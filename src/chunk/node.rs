use bytemuck::{Pod, Zeroable};

pub const MAX_INTERIOR_NODES: u32 = 64 * 64 + 64 + 1; // 4161
pub const MAX_LEAF_NODES: u32 = 64 * 64 * 64 + 64 * 64 + 64 + 1; // 266305

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct InteriorNode {
	has_child: u64,
	is_leaf: u64,
	node_offsets: u32, // packed: [12..0] interior_ptr (13 bits), [31..13] leaf_ptr (19 bits)
	material_offset: u32,
}

/// Wide form used only during editing - full u32 offsets, no bit-packing.
#[derive(Copy, Clone, Default)]
pub struct InteriorNodeWide {
	has_child: u64,
	is_leaf: u64,
	interior_offset: u32,
	leaf_offset: u32,
	material_offset: u32,
}

#[derive(Copy, Clone, Default)]
pub struct LeafNode {
	occupancy: u64,
	material_offset: u32,
}

pub enum CellState {
	Empty,
	Filled,
	Interior,
	Leaf,
}

impl InteriorNode {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn has_child(&self) -> u64 {
		self.has_child
	}

	pub fn set_has_child(&mut self, mask: u64) {
		self.has_child = mask;
	}

	pub fn is_leaf(&self) -> u64 {
		self.is_leaf
	}

	pub fn set_is_leaf(&mut self, mask: u64) {
		self.is_leaf = mask;
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

	pub fn state(&self, slot: u8) -> CellState {
		match ((self.has_child >> slot) & 1, (self.is_leaf >> slot) & 1) {
			(0, 0) => CellState::Empty,
			(0, 1) => CellState::Filled,
			(1, 0) => CellState::Interior,
			(1, 1) => CellState::Leaf,
			_ => unreachable!(),
		}
	}

	pub fn set_state(&mut self, slot: u8, state: CellState) {
		match state {
			CellState::Empty => {
				self.has_child &= !(1u64 << slot);
				self.is_leaf &= !(1u64 << slot);
			}
			CellState::Filled => {
				self.has_child &= !(1u64 << slot);
				self.is_leaf |= 1u64 << slot;
			}
			CellState::Interior => {
				self.has_child |= 1u64 << slot;
				self.is_leaf &= !(1u64 << slot);
			}
			CellState::Leaf => {
				self.has_child |= 1u64 << slot;
				self.is_leaf |= 1u64 << slot;
			}
		}
	}

	pub fn occupancy(&self) -> u64 {
		self.has_child | self.is_leaf
	}

	pub fn occupied_count(&self) -> u32 {
		self.occupancy().count_ones()
	}

	pub fn is_occupied(&self, slot: u8) -> bool {
		(self.occupancy() >> slot) & 1 != 0
	}

	pub fn is_empty_slot(&self, slot: u8) -> bool {
		(self.occupancy() >> slot) & 1 == 0
	}

	pub fn interior_child_count(&self) -> u32 {
		(self.has_child & !self.is_leaf).count_ones()
	}

	pub fn leaf_child_count(&self) -> u32 {
		(self.has_child & self.is_leaf).count_ones()
	}

	pub fn filled_count(&self) -> u32 {
		(!self.has_child & self.is_leaf).count_ones()
	}

	pub fn interior_child_index(&self, slot: u8) -> u32 {
		self.interior_offset() + popcount_below(self.has_child & !self.is_leaf, slot)
	}

	pub fn leaf_child_index(&self, slot: u8) -> u32 {
		self.leaf_offset() + popcount_below(self.has_child & self.is_leaf, slot)
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + popcount_below(self.occupancy(), slot)
	}
}

impl InteriorNodeWide {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn has_child(&self) -> u64 {
		self.has_child
	}

	pub fn set_has_child(&mut self, mask: u64) {
		self.has_child = mask;
	}

	pub fn is_leaf(&self) -> u64 {
		self.is_leaf
	}

	pub fn set_is_leaf(&mut self, mask: u64) {
		self.is_leaf = mask;
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

	pub fn state(&self, slot: u8) -> CellState {
		match ((self.has_child >> slot) & 1, (self.is_leaf >> slot) & 1) {
			(0, 0) => CellState::Empty,
			(0, 1) => CellState::Filled,
			(1, 0) => CellState::Interior,
			(1, 1) => CellState::Leaf,
			_ => unreachable!(),
		}
	}

	pub fn occupancy(&self) -> u64 {
		self.has_child | self.is_leaf
	}

	pub fn occupied_count(&self) -> u32 {
		self.occupancy().count_ones()
	}

	pub fn is_occupied(&self, slot: u8) -> bool {
		(self.occupancy() >> slot) & 1 != 0
	}

	pub fn is_empty_slot(&self, slot: u8) -> bool {
		(self.occupancy() >> slot) & 1 == 0
	}

	pub fn interior_child_count(&self) -> u32 {
		(self.has_child & !self.is_leaf).count_ones()
	}

	pub fn leaf_child_count(&self) -> u32 {
		(self.has_child & self.is_leaf).count_ones()
	}

	pub fn filled_count(&self) -> u32 {
		(!self.has_child & self.is_leaf).count_ones()
	}

	pub fn interior_child_index(&self, slot: u8) -> u32 {
		self.interior_offset + popcount_below(self.has_child & !self.is_leaf, slot)
	}

	pub fn leaf_child_index(&self, slot: u8) -> u32 {
		self.leaf_offset + popcount_below(self.has_child & self.is_leaf, slot)
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + popcount_below(self.occupancy(), slot)
	}
}

impl LeafNode {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn occupancy(&self) -> u64 {
		self.occupancy
	}

	pub fn set_occupancy(&mut self, mask: u64) {
		self.occupancy = mask;
	}

	pub fn material_offset(&self) -> u32 {
		self.material_offset
	}

	pub fn set_material_offset(&mut self, offset: u32) {
		self.material_offset = offset;
	}

	pub fn occupied_count(&self) -> u32 {
		self.occupancy.count_ones()
	}

	pub fn is_occupied(&self, slot: u8) -> bool {
		(self.occupancy >> slot) & 1 != 0
	}

	pub fn is_empty_slot(&self, slot: u8) -> bool {
		(self.occupancy >> slot) & 1 == 0
	}

	pub fn set_occupied(&mut self, slot: u8) {
		self.occupancy |= 1u64 << slot;
	}

	pub fn clear_occupied(&mut self, slot: u8) {
		self.occupancy &= !(1u64 << slot);
	}

	pub fn material_index(&self, slot: u8) -> u32 {
		self.material_offset + popcount_below(self.occupancy, slot)
	}
}

pub fn popcount_below(mask: u64, slot: u8) -> u32 {
	(mask & ((1u64 << slot) - 1)).count_ones()
}
