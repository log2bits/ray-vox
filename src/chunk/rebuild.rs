use super::edit::Path;
use super::material::Material;
use super::node::{CellState, InteriorNodeWide, LeafNode};
use super::{Chunk, Editing};
use crate::util::PalettedVec;

pub(super) enum RebuildResult {
	Empty,
	Filled(Material),
	Leaf(LeafNode),
	Interior(InteriorNodeWide),
}

impl Chunk<Editing> {
	#[inline]
	pub(super) fn rebuild_interior(
		&mut self,
		old_node: Option<InteriorNodeWide>,
		expand_fill: Option<Material>,
		tree_depth: u8,
		edits: &[(Path, Material)],
	) -> RebuildResult {
		debug_assert!(tree_depth <= 2);
		debug_assert!(!(old_node.is_some() && expand_fill.is_some()));

		let old = old_node.unwrap_or_default();
		let children_are_leaves = tree_depth == 2;

		let mut new_has_child = 0u64;
		let mut new_is_leaf = 0u64;
		let mut slot_lods = [Material::air(); 64];
		let mut slot_interior_nodes = [InteriorNodeWide::default(); 64];
		let mut slot_leaf_nodes = [LeafNode::default(); 64];

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
					&self.state.interior_nodes,
					&self.leaf_nodes,
					&mut new_has_child,
					&mut new_is_leaf,
					&mut slot_lods,
					&mut slot_interior_nodes,
					&mut slot_leaf_nodes,
				);

				if ei == edits.len() {
					if let Some(fill) = expand_fill {
						let visited = u64::MAX >> (63 - slot);
						let unvisited = !visited;
						new_is_leaf |= unvisited;
						let mut mask = unvisited;
						while mask != 0 {
							let s = mask.trailing_zeros() as u8;
							slot_lods[s as usize] = fill;
							mask &= mask - 1;
						}
					} else {
						let visited_mask = u64::MAX >> (63 - slot);
						let mut mask = old.occupancy() & !visited_mask;
						while mask != 0 {
							let s = mask.trailing_zeros() as u8;
							carry_forward_slot(
								s,
								old,
								None,
								&self.materials,
								&self.state.interior_nodes,
								&self.leaf_nodes,
								&mut new_has_child,
								&mut new_is_leaf,
								&mut slot_lods,
								&mut slot_interior_nodes,
								&mut slot_leaf_nodes,
							);
							mask &= mask - 1;
						}
					}
					break;
				}
				continue;
			}

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

			let (child_is_leaf, old_child_idx, child_expand_fill) =
				child_descent_params(slot, old, expand_fill, children_are_leaves, &self.materials);

			if child_is_leaf {
				let old_leaf = old_child_idx.map(|i| self.leaf_nodes[i as usize]);
				match self.rebuild_leaf(old_leaf, child_expand_fill, slot_edits) {
					RebuildResult::Empty => {}
					RebuildResult::Filled(mat) => {
						new_is_leaf |= 1u64 << slot;
						slot_lods[slot as usize] = mat;
					}
					RebuildResult::Leaf(leaf_node) => {
						let lod = self.materials.get(leaf_node.material_offset());
						new_has_child |= 1u64 << slot;
						new_is_leaf |= 1u64 << slot;
						slot_lods[slot as usize] = lod;
						slot_leaf_nodes[slot as usize] = leaf_node;
					}
					RebuildResult::Interior(_) => unreachable!(),
				}
			} else {
				let old_interior = old_child_idx.map(|i| self.state.interior_nodes[i as usize]);
				match self.rebuild_interior(
					old_interior,
					child_expand_fill,
					tree_depth + 1,
					slot_edits,
				) {
					RebuildResult::Empty => {}
					RebuildResult::Filled(mat) => {
						new_is_leaf |= 1u64 << slot;
						slot_lods[slot as usize] = mat;
					}
					RebuildResult::Interior(interior_node) => {
						let lod = self.materials.get(interior_node.material_offset());
						new_has_child |= 1u64 << slot;
						slot_lods[slot as usize] = lod;
						slot_interior_nodes[slot as usize] = interior_node;
					}
					RebuildResult::Leaf(_) => unreachable!(),
				}
			}
		}

		let occupancy = new_has_child | new_is_leaf;
		if occupancy == 0 {
			return RebuildResult::Empty;
		}

		if new_has_child == 0 && new_is_leaf == !0u64 {
			let first = slot_lods[0];
			if slot_lods.iter().all(|&m| m == first) {
				return RebuildResult::Filled(first);
			}
		}

		let n_interior_children = (new_has_child & !new_is_leaf).count_ones() as usize;
		let n_leaf_children = (new_has_child & new_is_leaf).count_ones() as usize;

		let new_interior_ptr = self.state.interior_nodes.len() as u32;
		let new_leaf_ptr = self.leaf_nodes.len() as u32;

		self.state.interior_nodes.resize_with(
			self.state.interior_nodes.len() + n_interior_children,
			InteriorNodeWide::default,
		);
		self.leaf_nodes
			.resize_with(self.leaf_nodes.len() + n_leaf_children, LeafNode::default);

		let new_mat_offset = self.materials.len();

		let mut interior_rank = 0u32;
		let mut leaf_rank = 0u32;

		let mut mask = occupancy;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			let has = (new_has_child >> slot) & 1 != 0;
			let is_leaf_bit = (new_is_leaf >> slot) & 1 != 0;

			self.materials.push(slot_lods[slot as usize]);

			if has && !is_leaf_bit {
				self.state.interior_nodes[(new_interior_ptr + interior_rank) as usize] =
					slot_interior_nodes[slot as usize];
				interior_rank += 1;
			} else if has {
				self.leaf_nodes[(new_leaf_ptr + leaf_rank) as usize] =
					slot_leaf_nodes[slot as usize];
				leaf_rank += 1;
			}

			mask &= mask - 1;
		}

		let mut new_node = InteriorNodeWide::default();
		new_node.set_has_child(new_has_child);
		new_node.set_is_leaf(new_is_leaf);
		new_node.set_interior_offset(new_interior_ptr);
		new_node.set_leaf_offset(new_leaf_ptr);
		new_node.set_material_offset(new_mat_offset);

		RebuildResult::Interior(new_node)
	}

	#[inline]
	pub(super) fn rebuild_leaf(
		&mut self,
		old_node: Option<LeafNode>,
		expand_fill: Option<Material>,
		edits: &[(Path, Material)],
	) -> RebuildResult {
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
					let visited_mask = u64::MAX >> (63 - slot);
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
			return RebuildResult::Empty;
		}

		if new_occupancy == !0u64 {
			let first = slot_materials[0];
			if slot_materials.iter().all(|&m| m == first) {
				return RebuildResult::Filled(first);
			}
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

		RebuildResult::Leaf(new_node)
	}
}

// --- Private helpers ---

fn carry_forward_slot(
	slot: u8,
	old: InteriorNodeWide,
	expand_fill: Option<Material>,
	materials: &PalettedVec<Material>,
	interior_nodes: &[InteriorNodeWide],
	leaf_nodes: &[LeafNode],
	new_has_child: &mut u64,
	new_is_leaf: &mut u64,
	slot_lods: &mut [Material; 64],
	slot_interior_nodes: &mut [InteriorNodeWide; 64],
	slot_leaf_nodes: &mut [LeafNode; 64],
) {
	let (state, lod, interior_node, leaf_node) = if let Some(fill) = expand_fill {
		(
			CellState::Filled,
			fill,
			InteriorNodeWide::default(),
			LeafNode::default(),
		)
	} else {
		match old.state(slot) {
			CellState::Empty => return,
			CellState::Filled => {
				let lod = materials.get(old.material_index(slot));
				(
					CellState::Filled,
					lod,
					InteriorNodeWide::default(),
					LeafNode::default(),
				)
			}
			CellState::Interior => {
				let lod = materials.get(old.material_index(slot));
				let child_idx = old.interior_child_index(slot);
				let node = interior_nodes[child_idx as usize];
				(CellState::Interior, lod, node, LeafNode::default())
			}
			CellState::Leaf => {
				let lod = materials.get(old.material_index(slot));
				let child_idx = old.leaf_child_index(slot);
				let node = leaf_nodes[child_idx as usize];
				(CellState::Leaf, lod, InteriorNodeWide::default(), node)
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
			slot_interior_nodes[slot as usize] = interior_node;
		}
		CellState::Leaf => {
			*new_has_child |= 1u64 << slot;
			*new_is_leaf |= 1u64 << slot;
			slot_leaf_nodes[slot as usize] = leaf_node;
		}
	}
}

fn child_descent_params(
	slot: u8,
	old: InteriorNodeWide,
	expand_fill: Option<Material>,
	children_are_leaves: bool,
	materials: &PalettedVec<Material>,
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
