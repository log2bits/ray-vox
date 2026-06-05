use crate::util::types::Mask64;
use bytemuck::{Pod, Zeroable};

pub const MAX_INTERIOR_NODES: u32 = 64 * 64 + 64 + 1;
pub const MAX_LEAF_NODES: u32 = 64 * 64 * 64 + 64 * 64 + 64 + 1;

const _: () = assert!(MAX_INTERIOR_NODES <= 1 << 13);
const _: () = assert!(MAX_LEAF_NODES <= 1 << 19);

pub enum CellState {
	Empty,
	Filled,
	Interior,
	Leaf,
}

/// The two masks that determine each slot's state. Both InteriorNode and
/// InteriorNodeWide embed this so the state and rank queries live in one place.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct ChildMasks {
	pub has_child: Mask64,
	pub is_leaf: Mask64,
}

impl ChildMasks {
	#[inline]
	pub fn new(has_child: u64, is_leaf: u64) -> Self {
		Self {
			has_child: Mask64(has_child),
			is_leaf: Mask64(is_leaf),
		}
	}

	#[inline]
	pub fn state(&self, slot: u8) -> CellState {
		match (self.has_child.contains(slot), self.is_leaf.contains(slot)) {
			(false, false) => CellState::Empty,
			(false, true) => CellState::Filled,
			(true, false) => CellState::Interior,
			(true, true) => CellState::Leaf,
		}
	}

	#[inline]
	pub fn set_state(&mut self, slot: u8, state: CellState) {
		let bit = Mask64::bit(slot);
		let (has, leaf) = match state {
			CellState::Empty => (false, false),
			CellState::Filled => (false, true),
			CellState::Interior => (true, false),
			CellState::Leaf => (true, true),
		};
		if has { self.has_child |= bit; } else { self.has_child &= !bit; }
		if leaf { self.is_leaf |= bit; } else { self.is_leaf &= !bit; }
	}

	#[inline]
	pub fn occupancy(&self) -> Mask64 {
		self.has_child | self.is_leaf
	}

	#[inline]
	pub fn interiors(&self) -> Mask64 {
		self.has_child & !self.is_leaf
	}

	#[inline]
	pub fn leaves(&self) -> Mask64 {
		self.has_child & self.is_leaf
	}

	#[inline]
	pub fn filled(&self) -> Mask64 {
		!self.has_child & self.is_leaf
	}

	#[inline]
	pub fn interior_rank(&self, slot: u8) -> u32 {
		self.interiors().popcount_below(slot)
	}

	#[inline]
	pub fn leaf_rank(&self, slot: u8) -> u32 {
		self.leaves().popcount_below(slot)
	}

	#[inline]
	pub fn material_rank(&self, slot: u8) -> u32 {
		self.occupancy().popcount_below(slot)
	}
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct InteriorNode {
	pub masks: ChildMasks,
	// Bits 0..13 hold interior_ptr, bits 13..32 hold leaf_ptr.
	node_offsets: u32,
	material_offset: u32,
}

/// Edit-time form of an interior node. Full u32 offsets instead of packed bits.
#[derive(Copy, Clone, Default)]
pub struct InteriorNodeWide {
	pub masks: ChildMasks,
	interior_offset: u32,
	leaf_offset: u32,
	material_offset: u32,
}

#[derive(Copy, Clone, Default)]
pub struct LeafNode {
	pub occupancy: Mask64,
	material_offset: u32,
}

impl InteriorNode {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn material_offset(&self) -> u32 {
		self.material_offset
	}

	pub fn set_material_offset(&mut self, offset: u32) {
		self.material_offset = offset;
	}

	pub fn interior_offset(&self) -> u32 {
		self.node_offsets & 0x1FFF
	}

	pub fn set_interior_offset(&mut self, offset: u32) {
		debug_assert!(
			offset < MAX_INTERIOR_NODES,
			"interior_offset {} exceeds theoretical max {}",
			offset,
			MAX_INTERIOR_NODES,
		);
		self.node_offsets = (self.node_offsets & 0xFFFFE000) | (offset & 0x1FFF);
	}

	pub fn leaf_offset(&self) -> u32 {
		(self.node_offsets >> 13) & 0x7FFFF
	}

	pub fn set_leaf_offset(&mut self, offset: u32) {
		debug_assert!(
			offset < MAX_LEAF_NODES,
			"leaf_offset {} exceeds theoretical max {}",
			offset,
			MAX_LEAF_NODES,
		);
		self.node_offsets = (self.node_offsets & 0x1FFF) | ((offset & 0x7FFFF) << 13);
	}

	pub fn interior_child_index(&self, slot: u8) -> u32 {
		self.interior_offset() + self.masks.interior_rank(slot)
	}

	pub fn leaf_child_index(&self, slot: u8) -> u32 {
		self.leaf_offset() + self.masks.leaf_rank(slot)
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + self.masks.material_rank(slot)
	}
}

impl InteriorNodeWide {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn interior_offset(&self) -> u32 {
		self.interior_offset
	}

	pub fn set_interior_offset(&mut self, offset: u32) {
		self.interior_offset = offset;
	}

	pub fn leaf_offset(&self) -> u32 {
		self.leaf_offset
	}

	pub fn set_leaf_offset(&mut self, offset: u32) {
		self.leaf_offset = offset;
	}

	pub fn material_offset(&self) -> u32 {
		self.material_offset
	}

	pub fn set_material_offset(&mut self, offset: u32) {
		self.material_offset = offset;
	}

	pub fn interior_child_index(&self, slot: u8) -> u32 {
		self.interior_offset + self.masks.interior_rank(slot)
	}

	pub fn leaf_child_index(&self, slot: u8) -> u32 {
		self.leaf_offset + self.masks.leaf_rank(slot)
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + self.masks.material_rank(slot)
	}
}

impl LeafNode {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn material_offset(&self) -> u32 {
		self.material_offset
	}

	pub fn set_material_offset(&mut self, offset: u32) {
		self.material_offset = offset;
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + self.occupancy.popcount_below(slot)
	}
}
