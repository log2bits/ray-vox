use super::edit::Path;
use super::material::Material;
use super::node::{CellState, InteriorNode, LeafNode};
use super::Chunk;

impl Chunk {
	/// Rebuilds an interior node at `tree_depth` (0 = root, 1, 2) against a sorted edit slice.
	///
	/// Uses path-copy: appends a new version of the node (and any modified descendants) without
	/// touching existing nodes. Old nodes become garbage collected by the compaction pass.
	///
	/// `old_node`: the node being replaced; `None` means the subtree didn't exist yet.
	/// `expand_fill`: when the caller's slot was a filled cell we're descending into, this
	///   is the fill material — every slot in this node starts as filled with that material.
	///
	/// Returns `true` if a new node was appended (subtree is non-empty). When `true`, the
	/// new node is `self.interior_nodes.last()` and its LOD material is at its `material_offset`.
	#[inline]
	pub(super) fn rebuild_interior(
		&mut self,
		old_node: Option<InteriorNode>,
		expand_fill: Option<Material>,
		tree_depth: u8,
		edits: &[(Path, Material)],
	) -> bool {
		debug_assert!(tree_depth <= 2);
		debug_assert!(!(old_node.is_some() && expand_fill.is_some()));

		let old = old_node.unwrap_or_default();
		let children_are_leaves = tree_depth == 2;

		// Per-slot outcome after processing edits.
		// Interior children: slot_interior_idxs[slot] holds the index in interior_nodes.
		// Leaf children:     slot_leaf_idxs[slot] holds the index in leaf_nodes.
		// Filled/empty:      only the lod/material is stored in slot_lods.
		let mut new_has_child = 0u64;
		let mut new_is_leaf = 0u64;
		let mut slot_lods = [Material::air(); 64];
		let mut slot_interior_idxs = [u32::MAX; 64];
		let mut slot_leaf_idxs = [u32::MAX; 64];

		let mut ei = 0;
		for slot in 0u8..64 {
			let start = ei;
			while ei < edits.len() && edits[ei].0.slot_at(tree_depth) == slot {
				ei += 1;
			}
			let slot_edits = &edits[start..ei];

			if slot_edits.is_empty() {
				carry_forward_slot(
					slot,
					old,
					expand_fill,
					&self.materials,
					&mut new_has_child,
					&mut new_is_leaf,
					&mut slot_lods,
					&mut slot_interior_idxs,
					&mut slot_leaf_idxs,
				);
				// Once the edit iterator is exhausted the remaining slots are all
				// carry-forwards. Bulk them in with a bit-iteration instead of
				// continuing the full 0..64 loop.
				if ei == edits.len() {
					if let Some(fill) = expand_fill {
						let visited = (1u64 << (slot + 1)).wrapping_sub(1);
						let unvisited = !visited;
						new_is_leaf |= unvisited;
						let mut mask = unvisited;
						while mask != 0 {
							let s = mask.trailing_zeros() as u8;
							slot_lods[s as usize] = fill;
							mask &= mask - 1;
						}
					} else {
						let visited_mask = (1u64 << (slot + 1)).wrapping_sub(1);
						let mut mask = old.occupancy() & !visited_mask;
						while mask != 0 {
							let s = mask.trailing_zeros() as u8;
							carry_forward_slot(
								s,
								old,
								None,
								&self.materials,
								&mut new_has_child,
								&mut new_is_leaf,
								&mut slot_lods,
								&mut slot_interior_idxs,
								&mut slot_leaf_idxs,
							);
							mask &= mask - 1;
						}
					}
					break;
				}
				continue;
			}

			// A terminating edit (depth == tree_depth + 1) sets this cell to filled or empty.
			// The last one wins; any sub-edits that sorted before it are also discarded.
			let last_terminating = slot_edits
				.iter()
				.rfind(|(p, _)| p.depth() == tree_depth + 1)
				.map(|(_, m)| *m);

			if let Some(mat) = last_terminating {
				if !mat.is_air() {
					new_is_leaf |= 1u64 << slot;
					slot_lods[slot as usize] = mat;
				}
				continue;
			}

			// Only sub-edits remain: recurse into a child node.
			let (child_is_leaf, old_child_idx, child_expand_fill) =
				child_descent_params(slot, old, expand_fill, children_are_leaves, &self.materials);

			if child_is_leaf {
				let old_leaf = old_child_idx.map(|i| self.leaf_nodes[i as usize]);
				if self.rebuild_leaf(old_leaf, child_expand_fill, slot_edits) {
					let child_idx = self.leaf_nodes.len() as u32 - 1;
					let lod = self.materials.get(self.leaf_nodes.last().unwrap().material_offset());
					new_has_child |= 1u64 << slot;
					new_is_leaf |= 1u64 << slot;
					slot_lods[slot as usize] = lod;
					slot_leaf_idxs[slot as usize] = child_idx;
				}
			} else {
				let old_interior = old_child_idx.map(|i| self.interior_nodes[i as usize]);
				if self.rebuild_interior(old_interior, child_expand_fill, tree_depth + 1, slot_edits) {
					let child_idx = self.interior_nodes.len() as u32 - 1;
					let lod =
						self.materials.get(self.interior_nodes.last().unwrap().material_offset());
					new_has_child |= 1u64 << slot;
					slot_lods[slot as usize] = lod;
					slot_interior_idxs[slot as usize] = child_idx;
				}
			}
		}

		let occupancy = new_has_child | new_is_leaf;
		if occupancy == 0 {
			return false;
		}

		// Build contiguous child arrays.
		// All children for this node must be laid out consecutively in their respective
		// arrays so the popcount-implicit addressing scheme works during traversal.
		// We extend the arrays first, then copy the child node structs into the reserved slots.
		// Subtrees of those children are already in the arrays at their own indices — we only
		// copy the top-level node structs here, which preserves sharing of deeper subtrees.
		let n_interior_children = (new_has_child & !new_is_leaf).count_ones() as usize;
		let n_leaf_children = (new_has_child & new_is_leaf).count_ones() as usize;

		let new_interior_ptr = self.interior_nodes.len() as u32;
		let new_leaf_ptr = self.leaf_nodes.len() as u32;

		self.interior_nodes
			.resize_with(self.interior_nodes.len() + n_interior_children, InteriorNode::default);
		self.leaf_nodes
			.resize_with(self.leaf_nodes.len() + n_leaf_children, LeafNode::default);

		let new_mat_offset = self.materials.len();

		let mut interior_rank = 0u32;
		let mut leaf_rank = 0u32;

		let mut mask = occupancy;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			let has = (new_has_child >> slot) & 1 != 0;
			let is_leaf = (new_is_leaf >> slot) & 1 != 0;

			self.materials.push(slot_lods[slot as usize]);

			if has && !is_leaf {
				let src = self.interior_nodes[slot_interior_idxs[slot as usize] as usize];
				self.interior_nodes[(new_interior_ptr + interior_rank) as usize] = src;
				interior_rank += 1;
			} else if has {
				let src = self.leaf_nodes[slot_leaf_idxs[slot as usize] as usize];
				self.leaf_nodes[(new_leaf_ptr + leaf_rank) as usize] = src;
				leaf_rank += 1;
			}

			mask &= mask - 1;
		}

		let lod = slot_lods[occupancy.trailing_zeros() as usize];
		let _ = lod; // used by caller via material_offset after this returns

		let mut new_node = InteriorNode::default();
		new_node.set_has_child(new_has_child);
		new_node.set_is_leaf(new_is_leaf);
		new_node.set_interior_offset(new_interior_ptr);
		new_node.set_leaf_offset(new_leaf_ptr);
		new_node.set_material_offset(new_mat_offset);
		self.interior_nodes.push(new_node);

		true
	}

	/// Rebuilds a leaf node against a sorted edit slice (all edits must be at path depth 4).
	///
	/// Returns `true` if a new node was appended. When `true`, the new node is
	/// `self.leaf_nodes.last()` and its LOD material is at its `material_offset`.
	#[inline]
	fn rebuild_leaf(
		&mut self,
		old_node: Option<LeafNode>,
		expand_fill: Option<Material>,
		edits: &[(Path, Material)],
	) -> bool {
		debug_assert!(!(old_node.is_some() && expand_fill.is_some()));

		let old = old_node.unwrap_or_default();

		let mut new_occupancy = 0u64;
		let mut slot_materials = [Material::air(); 64];

		let mut ei = 0;
		for slot in 0u8..64 {
			let start = ei;
			while ei < edits.len() && edits[ei].0.slot_at(3) == slot {
				ei += 1;
			}
			let slot_edits = &edits[start..ei];

			let mat = if !slot_edits.is_empty() {
				slot_edits.last().unwrap().1
			} else if let Some(fill) = expand_fill {
				fill
			} else if old.is_occupied(slot) {
				self.materials.get(old.material_index(slot))
			} else {
				if ei == edits.len() {
					let visited_mask = (1u64 << (slot + 1)).wrapping_sub(1);
					let mut mask = old.occupancy() & !visited_mask;
					while mask != 0 {
						let s = mask.trailing_zeros() as u8;
						let m = self.materials.get(old.material_index(s));
						if !m.is_air() {
							new_occupancy |= 1u64 << s;
							slot_materials[s as usize] = m;
						}
						mask &= mask - 1;
					}
					break;
				}
				continue;
			};

			if !mat.is_air() {
				new_occupancy |= 1u64 << slot;
				slot_materials[slot as usize] = mat;
			}
		}

		if new_occupancy == 0 {
			return false;
		}

		let new_mat_offset = self.materials.len();
		let mut mask = new_occupancy;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			self.materials.push(slot_materials[slot as usize]);
			mask &= mask - 1;
		}

		let mut new_node = LeafNode::default();
		new_node.set_occupancy(new_occupancy);
		new_node.set_material_offset(new_mat_offset);
		self.leaf_nodes.push(new_node);

		true
	}
}

// --- Private helpers ---

fn carry_forward_slot(
	slot: u8,
	old: InteriorNode,
	expand_fill: Option<Material>,
	materials: &super::PalettedVec<Material>,
	new_has_child: &mut u64,
	new_is_leaf: &mut u64,
	slot_lods: &mut [Material; 64],
	slot_interior_idxs: &mut [u32; 64],
	slot_leaf_idxs: &mut [u32; 64],
) {
	let (state, lod, child_idx) = if let Some(fill) = expand_fill {
		(CellState::Filled, fill, 0u32)
	} else {
		match old.state(slot) {
			CellState::Empty => return,
			CellState::Filled => (CellState::Filled, materials.get(old.material_index(slot)), 0),
			CellState::Interior => {
				let lod = materials.get(old.material_index(slot));
				(CellState::Interior, lod, old.interior_child_index(slot))
			}
			CellState::Leaf => {
				let lod = materials.get(old.material_index(slot));
				(CellState::Leaf, lod, old.leaf_child_index(slot))
			}
		}
	};

	slot_lods[slot as usize] = lod;
	match state {
		CellState::Empty => {}
		CellState::Filled => {
			*new_is_leaf |= 1u64 << slot;
		}
		CellState::Interior => {
			*new_has_child |= 1u64 << slot;
			slot_interior_idxs[slot as usize] = child_idx;
		}
		CellState::Leaf => {
			*new_has_child |= 1u64 << slot;
			*new_is_leaf |= 1u64 << slot;
			slot_leaf_idxs[slot as usize] = child_idx;
		}
	}
}

fn child_descent_params(
	slot: u8,
	old: InteriorNode,
	expand_fill: Option<Material>,
	children_are_leaves: bool,
	materials: &super::PalettedVec<Material>,
) -> (bool, Option<u32>, Option<Material>) {
	if let Some(fill) = expand_fill {
		return (children_are_leaves, None, Some(fill));
	}
	match old.state(slot) {
		CellState::Empty => (children_are_leaves, None, None),
		CellState::Filled => {
			let fill = materials.get(old.material_index(slot));
			(children_are_leaves, None, Some(fill))
		}
		CellState::Interior => (false, Some(old.interior_child_index(slot)), None),
		CellState::Leaf => (true, Some(old.leaf_child_index(slot)), None),
	}
}
