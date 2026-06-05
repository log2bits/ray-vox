use super::edit::Path;
use super::material::Material;
use super::node::{CellState, ChildMasks, InteriorNodeWide, LeafNode};
use super::MutableChunk;
use crate::util::types::Mask64;

/// What was at an interior position before edits ran.
#[derive(Copy, Clone)]
pub enum InteriorBase {
	Empty,
	Fill(Material),
	Existing(InteriorNodeWide),
}

impl InteriorBase {
	fn occupied_mask(self) -> Mask64 {
		match self {
			Self::Empty => Mask64::EMPTY,
			Self::Fill(_) => Mask64::FULL,
			Self::Existing(n) => n.masks.occupancy(),
		}
	}
}

/// What was at a leaf position before edits ran.
#[derive(Copy, Clone)]
pub enum LeafBase {
	Empty,
	Fill(Material),
	Existing(LeafNode),
}

impl LeafBase {
	fn occupied_mask(self) -> Mask64 {
		match self {
			Self::Empty => Mask64::EMPTY,
			Self::Fill(_) => Mask64::FULL,
			Self::Existing(n) => n.occupancy,
		}
	}
}

/// Groups a sorted edit slice by slot at the given depth. Yields one (slot, edits)
/// pair per slot, in ascending order.
struct EditGroups<'a> {
	edits: &'a [(Path, Material)],
	depth: u8,
	i: usize,
}

impl<'a> EditGroups<'a> {
	fn new(edits: &'a [(Path, Material)], depth: u8) -> Self {
		Self { edits, depth, i: 0 }
	}
}

impl<'a> Iterator for EditGroups<'a> {
	type Item = (u8, &'a [(Path, Material)]);
	fn next(&mut self) -> Option<Self::Item> {
		if self.i >= self.edits.len() {
			return None;
		}
		let slot = self.edits[self.i].0.slot_at(self.depth);
		let start = self.i;
		while self.i < self.edits.len() && self.edits[self.i].0.slot_at(self.depth) == slot {
			self.i += 1;
		}
		Some((slot, &self.edits[start..self.i]))
	}
}

enum SlotWork<'a> {
	Carry(u8),
	Edited(u8, &'a [(Path, Material)]),
}

/// Walks the union of carry-forward slots and edited slots in ascending order.
/// Skips slots with no pre-edit content and no edits.
struct SlotMerge<'a> {
	carry: Mask64,
	groups: EditGroups<'a>,
	peeked: Option<(u8, &'a [(Path, Material)])>,
}

impl<'a> SlotMerge<'a> {
	fn new(carry: Mask64, edits: &'a [(Path, Material)], depth: u8) -> Self {
		let mut groups = EditGroups::new(edits, depth);
		let peeked = groups.next();
		Self { carry, groups, peeked }
	}

	fn next_carry(&self) -> Option<u8> {
		if self.carry.is_empty() {
			None
		} else {
			Some(self.carry.raw().trailing_zeros() as u8)
		}
	}

	fn consume_carry(&mut self, slot: u8) {
		self.carry &= !Mask64::bit(slot);
	}

	fn take_edited(&mut self) -> Option<SlotWork<'a>> {
		let (slot, grp) = self.peeked.take()?;
		self.peeked = self.groups.next();
		Some(SlotWork::Edited(slot, grp))
	}
}

impl<'a> Iterator for SlotMerge<'a> {
	type Item = SlotWork<'a>;
	fn next(&mut self) -> Option<Self::Item> {
		let c = self.next_carry();
		let e = self.peeked.map(|(s, _)| s);
		match (c, e) {
			(None, None) => None,
			(Some(c), None) => {
				self.consume_carry(c);
				Some(SlotWork::Carry(c))
			}
			(None, Some(_)) => self.take_edited(),
			(Some(c), Some(e)) if c < e => {
				self.consume_carry(c);
				Some(SlotWork::Carry(c))
			}
			(Some(c), Some(e)) => {
				if c == e {
					self.consume_carry(c);
				}
				self.take_edited()
			}
		}
	}
}

/// Outcome of rebuilding one tree position. The Material on Leaf/Interior is the
/// LOD the parent stores for this slot.
pub enum RebuildResult {
	Empty,
	Filled(Material),
	Leaf(Material, LeafNode),
	Interior(Material, InteriorNodeWide),
}

/// Most-frequent material among occupied slots. Lowest slot wins ties, which
/// keeps the result deterministic for DAG dedup.
pub fn mode_over(occupancy: Mask64, mats: &[Material; 64]) -> Material {
	let mut keys = [Material::air(); 64];
	let mut counts = [0u32; 64];
	let mut n = 0usize;
	let mut best = (Material::air(), 0u32);

	for slot in occupancy.iter_slots() {
		let m = mats[slot as usize];
		let j = match keys[..n].iter().position(|&k| k == m) {
			Some(j) => j,
			None => {
				keys[n] = m;
				counts[n] = 0;
				let j = n;
				n += 1;
				j
			}
		};
		counts[j] += 1;
		if counts[j] > best.1 {
			best = (m, counts[j]);
		}
	}
	best.0
}

/// Per-slot accumulator for assembling one interior node. Records each slot's
/// outcome, then finish collapses or materializes.
pub struct ChildSet {
	masks: ChildMasks,
	lods: [Material; 64],
	interior_nodes: [InteriorNodeWide; 64],
	leaf_nodes: [LeafNode; 64],
}

impl Default for ChildSet {
	fn default() -> Self {
		Self {
			masks: ChildMasks::default(),
			lods: [Material::air(); 64],
			interior_nodes: [InteriorNodeWide::default(); 64],
			leaf_nodes: [LeafNode::default(); 64],
		}
	}
}

impl ChildSet {
	pub fn record(&mut self, slot: u8, outcome: RebuildResult) {
		let bit = Mask64::bit(slot);
		let i = slot as usize;
		match outcome {
			RebuildResult::Empty => {}
			RebuildResult::Filled(mat) => {
				self.masks.is_leaf |= bit;
				self.lods[i] = mat;
			}
			RebuildResult::Leaf(lod, node) => {
				self.masks.has_child |= bit;
				self.masks.is_leaf |= bit;
				self.lods[i] = lod;
				self.leaf_nodes[i] = node;
			}
			RebuildResult::Interior(lod, node) => {
				self.masks.has_child |= bit;
				self.lods[i] = lod;
				self.interior_nodes[i] = node;
			}
		}
	}

	fn uniform_filled(&self) -> Option<Material> {
		if !self.masks.has_child.is_empty() || self.masks.is_leaf != Mask64::FULL {
			return None;
		}
		let first = self.lods[0];
		self.lods.iter().all(|&m| m == first).then_some(first)
	}

	fn representative(&self) -> Material {
		mode_over(self.masks.occupancy(), &self.lods)
	}

	pub fn finish(self, chunk: &mut MutableChunk) -> RebuildResult {
		if self.masks.occupancy().is_empty() {
			return RebuildResult::Empty;
		}
		if let Some(mat) = self.uniform_filled() {
			return RebuildResult::Filled(mat);
		}
		chunk.materialize_interior(self)
	}
}

/// Per-slot accumulator for assembling one leaf node.
pub struct VoxelSet {
	occupancy: Mask64,
	materials: [Material; 64],
}

impl Default for VoxelSet {
	fn default() -> Self {
		Self {
			occupancy: Mask64::EMPTY,
			materials: [Material::air(); 64],
		}
	}
}

impl VoxelSet {
	pub fn record(&mut self, slot: u8, mat: Material) {
		if mat.is_air() {
			return;
		}
		self.occupancy |= Mask64::bit(slot);
		self.materials[slot as usize] = mat;
	}

	fn uniform_filled(&self) -> Option<Material> {
		if self.occupancy != Mask64::FULL {
			return None;
		}
		let first = self.materials[0];
		self.materials.iter().all(|&m| m == first).then_some(first)
	}

	fn representative(&self) -> Material {
		mode_over(self.occupancy, &self.materials)
	}

	pub fn finish(self, chunk: &mut MutableChunk) -> RebuildResult {
		if self.occupancy.is_empty() {
			return RebuildResult::Empty;
		}
		if let Some(mat) = self.uniform_filled() {
			return RebuildResult::Filled(mat);
		}
		let lod = self.representative();
		let new_mat_offset = chunk.materials.len();
		for slot in self.occupancy.iter_slots() {
			chunk.materials.push(self.materials[slot as usize]);
		}
		let mut node = LeafNode::default();
		node.occupancy = self.occupancy;
		node.set_material_offset(new_mat_offset);
		RebuildResult::Leaf(lod, node)
	}
}

impl MutableChunk {
	pub fn rebuild_interior(
		&mut self,
		base: InteriorBase,
		tree_depth: u8,
		edits: &[(Path, Material)],
	) -> RebuildResult {
		debug_assert!(tree_depth <= 2);
		let children_are_leaves = tree_depth == 2;
		let mut set = ChildSet::default();

		for work in SlotMerge::new(base.occupied_mask(), edits, tree_depth) {
			let (slot, outcome) = match work {
				SlotWork::Carry(slot) => (slot, self.carry_interior(base, slot)),
				SlotWork::Edited(slot, grp) => (
					slot,
					self.resolve_interior(base, slot, children_are_leaves, tree_depth, grp),
				),
			};
			set.record(slot, outcome);
		}
		set.finish(self)
	}

	pub fn rebuild_leaf(
		&mut self,
		base: LeafBase,
		edits: &[(Path, Material)],
	) -> RebuildResult {
		let mut set = VoxelSet::default();
		for work in SlotMerge::new(base.occupied_mask(), edits, 3) {
			let (slot, mat) = match work {
				SlotWork::Carry(slot) => (slot, self.carry_leaf(base, slot)),
				SlotWork::Edited(slot, grp) => (slot, grp.last().unwrap().1),
			};
			set.record(slot, mat);
		}
		set.finish(self)
	}

	/// Read the material a slot stored in this node's slab.
	#[inline]
	fn slab_mat(&self, n: &InteriorNodeWide, slot: u8) -> Material {
		self.materials.get(n.material_index(slot))
	}

	fn carry_interior(&self, base: InteriorBase, slot: u8) -> RebuildResult {
		match base {
			InteriorBase::Empty => RebuildResult::Empty,
			InteriorBase::Fill(mat) => RebuildResult::Filled(mat),
			InteriorBase::Existing(n) => match n.masks.state(slot) {
				CellState::Empty => RebuildResult::Empty,
				CellState::Filled => RebuildResult::Filled(self.slab_mat(&n, slot)),
				CellState::Interior => RebuildResult::Interior(
					self.slab_mat(&n, slot),
					self.interior_nodes[n.interior_child_index(slot) as usize],
				),
				CellState::Leaf => RebuildResult::Leaf(
					self.slab_mat(&n, slot),
					self.leaf_nodes[n.leaf_child_index(slot) as usize],
				),
			},
		}
	}

	fn carry_leaf(&self, base: LeafBase, slot: u8) -> Material {
		match base {
			LeafBase::Empty => Material::air(),
			LeafBase::Fill(mat) => mat,
			LeafBase::Existing(leaf) => self.materials.get(leaf.material_index(slot)),
		}
	}

	/// Apply a slot's edit group. Splits at the terminating-vs-deeper boundary
	/// (depth == tree_depth + 1 is terminating). Terminating sets the child base,
	/// deeper edits recurse on top.
	fn resolve_interior(
		&mut self,
		base: InteriorBase,
		slot: u8,
		children_are_leaves: bool,
		tree_depth: u8,
		group: &[(Path, Material)],
	) -> RebuildResult {
		let split = group.partition_point(|(p, _)| p.depth() <= tree_depth + 1);
		let (terminating, deeper) = group.split_at(split);
		let last_terminating = terminating.last().map(|&(_, m)| m);

		if deeper.is_empty() {
			return match last_terminating {
				Some(mat) if mat.is_air() => RebuildResult::Empty,
				Some(mat) => RebuildResult::Filled(mat),
				None => self.carry_interior(base, slot),
			};
		}

		if children_are_leaves {
			let child_base = match last_terminating {
				Some(mat) if mat.is_air() => LeafBase::Empty,
				Some(mat) => LeafBase::Fill(mat),
				None => self.descend_to_leaf(base, slot),
			};
			self.rebuild_leaf(child_base, deeper)
		} else {
			let child_base = match last_terminating {
				Some(mat) if mat.is_air() => InteriorBase::Empty,
				Some(mat) => InteriorBase::Fill(mat),
				None => self.descend_to_interior(base, slot),
			};
			self.rebuild_interior(child_base, tree_depth + 1, deeper)
		}
	}

	fn descend_to_interior(&self, base: InteriorBase, slot: u8) -> InteriorBase {
		match base {
			InteriorBase::Empty => InteriorBase::Empty,
			InteriorBase::Fill(m) => InteriorBase::Fill(m),
			InteriorBase::Existing(n) => match n.masks.state(slot) {
				CellState::Empty => InteriorBase::Empty,
				CellState::Filled => InteriorBase::Fill(self.slab_mat(&n, slot)),
				CellState::Interior => {
					InteriorBase::Existing(self.interior_nodes[n.interior_child_index(slot) as usize])
				}
				CellState::Leaf => unreachable!("expected interior child at slot {slot}, found leaf"),
			},
		}
	}

	fn descend_to_leaf(&self, base: InteriorBase, slot: u8) -> LeafBase {
		match base {
			InteriorBase::Empty => LeafBase::Empty,
			InteriorBase::Fill(m) => LeafBase::Fill(m),
			InteriorBase::Existing(n) => match n.masks.state(slot) {
				CellState::Empty => LeafBase::Empty,
				CellState::Filled => LeafBase::Fill(self.slab_mat(&n, slot)),
				CellState::Leaf => LeafBase::Existing(self.leaf_nodes[n.leaf_child_index(slot) as usize]),
				CellState::Interior => unreachable!("expected leaf child at slot {slot}, found interior"),
			},
		}
	}

	fn materialize_interior(&mut self, set: ChildSet) -> RebuildResult {
		let masks = set.masks;
		let interior_ptr = self.interior_nodes.len() as u32;
		let leaf_ptr = self.leaf_nodes.len() as u32;
		let mat_offset = self.materials.len();

		// Push child blocks in slot order so popcount-implicit indexing lines up.
		for slot in masks.interiors().iter_slots() {
			self.interior_nodes.push(set.interior_nodes[slot as usize]);
		}
		for slot in masks.leaves().iter_slots() {
			self.leaf_nodes.push(set.leaf_nodes[slot as usize]);
		}
		for slot in masks.occupancy().iter_slots() {
			self.materials.push(set.lods[slot as usize]);
		}

		let lod = set.representative();
		let mut node = InteriorNodeWide::default();
		node.masks = masks;
		node.set_interior_offset(interior_ptr);
		node.set_leaf_offset(leaf_ptr);
		node.set_material_offset(mat_offset);
		RebuildResult::Interior(lod, node)
	}
}
