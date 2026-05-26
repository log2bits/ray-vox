use super::edit::Path;
use super::material::Material;
use super::node::{CellState, InteriorNodeWide, LeafNode};
use super::MutableChunk;

/// Outcome of rebuilding a single tree position. Used both as a recursive return value
/// and as the per-slot intermediate when assembling a parent.
///
/// The `Material` in `Leaf`/`Interior` is the LOD that should be stored in the parent's
/// material slab for this slot, so callers never have to re-fetch it from the chunk's
/// materials by `material_offset`.
pub(super) enum RebuildResult {
	Empty,
	Filled(Material),
	Leaf(Material, LeafNode),
	Interior(Material, InteriorNodeWide),
}

/// Per-slot accumulator for one `rebuild_interior` call. Built up as slots are processed,
/// then either collapsed into a `Filled`/`Empty` result, or materialized into the chunk's
/// arrays as a new contiguous child block + parent struct.
struct SlotAccumulator {
	has_child: u64,
	is_leaf: u64,
	lods: [Material; 64],
	interior_nodes: [InteriorNodeWide; 64],
	leaf_nodes: [LeafNode; 64],
}

impl Default for SlotAccumulator {
	fn default() -> Self {
		Self {
			has_child: 0,
			is_leaf: 0,
			lods: [Material::air(); 64],
			interior_nodes: [InteriorNodeWide::default(); 64],
			leaf_nodes: [LeafNode::default(); 64],
		}
	}
}

impl SlotAccumulator {
	fn record(&mut self, slot: u8, outcome: RebuildResult) {
		match outcome {
			RebuildResult::Empty => {}
			RebuildResult::Filled(mat) => {
				self.is_leaf |= 1u64 << slot;
				self.lods[slot as usize] = mat;
			}
			RebuildResult::Leaf(lod, node) => {
				self.has_child |= 1u64 << slot;
				self.is_leaf |= 1u64 << slot;
				self.lods[slot as usize] = lod;
				self.leaf_nodes[slot as usize] = node;
			}
			RebuildResult::Interior(lod, node) => {
				self.has_child |= 1u64 << slot;
				self.lods[slot as usize] = lod;
				self.interior_nodes[slot as usize] = node;
			}
		}
	}

	fn occupancy(&self) -> u64 {
		self.has_child | self.is_leaf
	}

	/// `Some(mat)` if every occupied slot is `Filled` with the same material.
	fn detect_uniform_filled(&self) -> Option<Material> {
		if self.has_child != 0 || self.is_leaf != !0u64 {
			return None;
		}
		let first = self.lods[0];
		self.lods.iter().all(|&m| m == first).then_some(first)
	}
}

impl MutableChunk {
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
		let mut accum = SlotAccumulator::default();

		let mut ei = 0;
		for slot in 0u8..64 {
			let start = ei;
			while ei < edits.len() && edits[ei].0.slot_at(tree_depth) == slot {
				ei += 1;
			}
			let slot_edits = &edits[start..ei];

			if slot_edits.is_empty() {
				let outcome = self.carry_forward_slot(slot, old, expand_fill);
				accum.record(slot, outcome);

				if ei == edits.len() {
					self.bulk_carry_forward(slot, old, expand_fill, &mut accum);
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
					accum.record(slot, RebuildResult::Filled(mat));
				}
				continue;
			}

			let result = self.rebuild_child(
				slot,
				old,
				expand_fill,
				children_are_leaves,
				tree_depth,
				slot_edits,
			);
			accum.record(slot, result);
		}

		if accum.occupancy() == 0 {
			return RebuildResult::Empty;
		}
		if let Some(mat) = accum.detect_uniform_filled() {
			return RebuildResult::Filled(mat);
		}
		self.materialize_interior(accum)
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

			let mat = if let Some(&(_, m)) = slot_edits.last() {
				m
			} else if let Some(fill) = expand_fill {
				fill
			} else if old.is_occupied(slot) {
				self.materials.get(old.material_index(slot))
			} else {
				if ei == edits.len() {
					self.bulk_carry_forward_leaf(
						slot,
						old,
						&mut new_occupancy,
						&mut slot_materials,
					);
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

		let first_slot = new_occupancy.trailing_zeros() as u8;
		let lod = slot_materials[first_slot as usize];

		let new_mat_offset = self.materials.len();
		let mut mask = new_occupancy;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			self.materials.push(slot_materials[slot as usize]);
			mask &= mask - 1;
		}

		let mut node = LeafNode::default();
		node.set_occupancy(new_occupancy);
		node.set_material_offset(new_mat_offset);
		RebuildResult::Leaf(lod, node)
	}

	fn carry_forward_slot(
		&self,
		slot: u8,
		old: InteriorNodeWide,
		expand_fill: Option<Material>,
	) -> RebuildResult {
		if let Some(fill) = expand_fill {
			return RebuildResult::Filled(fill);
		}
		match old.state(slot) {
			CellState::Empty => RebuildResult::Empty,
			CellState::Filled => {
				let lod = self.materials.get(old.material_index(slot));
				RebuildResult::Filled(lod)
			}
			CellState::Interior => {
				let lod = self.materials.get(old.material_index(slot));
				let node = self.interior_nodes[old.interior_child_index(slot) as usize];
				RebuildResult::Interior(lod, node)
			}
			CellState::Leaf => {
				let lod = self.materials.get(old.material_index(slot));
				let node = self.leaf_nodes[old.leaf_child_index(slot) as usize];
				RebuildResult::Leaf(lod, node)
			}
		}
	}

	fn bulk_carry_forward(
		&self,
		after_slot: u8,
		old: InteriorNodeWide,
		expand_fill: Option<Material>,
		accum: &mut SlotAccumulator,
	) {
		let visited = u64::MAX >> (63 - after_slot);
		let candidates = match expand_fill {
			Some(_) => !visited,
			None => old.occupancy() & !visited,
		};
		let mut mask = candidates;
		while mask != 0 {
			let s = mask.trailing_zeros() as u8;
			let outcome = self.carry_forward_slot(s, old, expand_fill);
			accum.record(s, outcome);
			mask &= mask - 1;
		}
	}

	fn bulk_carry_forward_leaf(
		&self,
		after_slot: u8,
		old: LeafNode,
		new_occupancy: &mut u64,
		slot_materials: &mut [Material; 64],
	) {
		let visited = u64::MAX >> (63 - after_slot);
		let mut mask = old.occupancy() & !visited;
		while mask != 0 {
			let s = mask.trailing_zeros() as u8;
			let m = self.materials.get(old.material_index(s));
			if !m.is_air() {
				*new_occupancy |= 1u64 << s;
				slot_materials[s as usize] = m;
			}
			mask &= mask - 1;
		}
	}

	fn rebuild_child(
		&mut self,
		slot: u8,
		old: InteriorNodeWide,
		expand_fill: Option<Material>,
		children_are_leaves: bool,
		tree_depth: u8,
		slot_edits: &[(Path, Material)],
	) -> RebuildResult {
		let (child_is_leaf, old_child_idx, child_expand_fill) =
			child_descent_params(slot, old, expand_fill, children_are_leaves, &self.materials);

		if child_is_leaf {
			let old_leaf = old_child_idx.map(|i| self.leaf_nodes[i as usize]);
			self.rebuild_leaf(old_leaf, child_expand_fill, slot_edits)
		} else {
			let old_interior = old_child_idx.map(|i| self.interior_nodes[i as usize]);
			self.rebuild_interior(old_interior, child_expand_fill, tree_depth + 1, slot_edits)
		}
	}

	fn materialize_interior(&mut self, accum: SlotAccumulator) -> RebuildResult {
		let occupancy = accum.occupancy();
		let n_interior = (accum.has_child & !accum.is_leaf).count_ones() as usize;
		let n_leaf = (accum.has_child & accum.is_leaf).count_ones() as usize;

		let interior_ptr = self.interior_nodes.len() as u32;
		let leaf_ptr = self.leaf_nodes.len() as u32;
		self.interior_nodes.resize_with(
			self.interior_nodes.len() + n_interior,
			InteriorNodeWide::default,
		);
		self.leaf_nodes
			.resize_with(self.leaf_nodes.len() + n_leaf, LeafNode::default);

		let mat_offset = self.materials.len();
		let mut interior_rank = 0u32;
		let mut leaf_rank = 0u32;

		let mut mask = occupancy;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			self.materials.push(accum.lods[slot as usize]);

			let has = (accum.has_child >> slot) & 1 != 0;
			let is_leaf_bit = (accum.is_leaf >> slot) & 1 != 0;
			if has && !is_leaf_bit {
				self.interior_nodes[(interior_ptr + interior_rank) as usize] =
					accum.interior_nodes[slot as usize];
				interior_rank += 1;
			} else if has {
				self.leaf_nodes[(leaf_ptr + leaf_rank) as usize] = accum.leaf_nodes[slot as usize];
				leaf_rank += 1;
			}
			mask &= mask - 1;
		}

		let first_slot = occupancy.trailing_zeros() as u8;
		let lod = accum.lods[first_slot as usize];

		let mut node = InteriorNodeWide::default();
		node.set_has_child(accum.has_child);
		node.set_is_leaf(accum.is_leaf);
		node.set_interior_offset(interior_ptr);
		node.set_leaf_offset(leaf_ptr);
		node.set_material_offset(mat_offset);
		RebuildResult::Interior(lod, node)
	}
}

fn child_descent_params(
	slot: u8,
	old: InteriorNodeWide,
	expand_fill: Option<Material>,
	children_are_leaves: bool,
	materials: &crate::util::PalettedVec<Material>,
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
