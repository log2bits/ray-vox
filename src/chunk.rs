mod edit;

pub use edit::VoxelEdit;

use crate::{
	tree::{Edit, EditPacket, Tree, DELETE},
	types::{Lut, Voxel},
};

pub const DEPTH: usize = 4;
pub const SIDE: u32 = 256; // 4^DEPTH

#[derive(Clone)]
pub struct Chunk {
	pub tree: Tree<DEPTH>,
	pub materials: Lut<Voxel>,
}

impl Chunk {
	pub fn new() -> Self {
		let mut materials = Lut::new();
		materials.values.push(Voxel::air()); // slot 0 = air (DELETE)
		Self { tree: Tree::new(1), materials }
	}

	pub fn memory_bytes(&self) -> usize {
		self.tree.bytes() + self.materials.values.len() * std::mem::size_of::<Voxel>()
	}

	// pos is chunk-local: each component in [0, 255]
	pub fn get_voxel(&self, pos: [u8; 3]) -> Option<Voxel> {
		if !self.tree.occupied {
			return None;
		}
		if self.tree.is_leaf {
			let idx = self.tree.value;
			return if idx == DELETE { None } else { Some(self.materials.get(idx)) };
		}
		let count = self.tree.levels[0].node_count();
		if count == 0 {
			return None;
		}
		let mut node = count - 1;
		for d in 0..DEPTH {
			let shift = 2 * (DEPTH - 1 - d) as u32;
			let sx = ((pos[0] as u32) >> shift) & 3;
			let sy = ((pos[1] as u32) >> shift) & 3;
			let sz = ((pos[2] as u32) >> shift) & 3;
			let slot = (sx | (sy << 2) | (sz << 4)) as u8;
			let level = &self.tree.levels[d];
			if !level.is_occupied(node, slot) {
				return None;
			}
			let idx = level.get_value(node, slot);
			if level.is_leaf(node, slot) || d + 1 == DEPTH {
				return if idx == DELETE { None } else { Some(self.materials.get(idx)) };
			}
			node = level.get_child(node, slot);
		}
		None
	}

	pub fn has_pending_edits(&self) -> bool {
		!self.tree.edits.packets.is_empty()
	}

	// Append a player voxel edit. Adds to the last unsorted packet, or starts a
	// new one if the last packet is sorted.
	pub fn queue_edit(&mut self, edit: VoxelEdit) {
		let value = match edit.voxel {
			None => DELETE,
			Some(v) => self.materials.get_or_add(v),
		};
		let pos = edit.pos.map(|p| p as u64);
		self.tree.queue_edit(Edit::new(value, pos, DEPTH as u8, 1));
	}

	// Append a pre-sorted packet of shape edits from the coverage walk.
	// Remaps the packet's raw voxel values into this chunk's material table indices.
	// The packet's per-value indices into its own LUT are unchanged; only the LUT
	// values are replaced (raw Voxel u32 → material index).
	pub fn add_shape_packet(&mut self, mut packet: EditPacket<DEPTH>) {
		for raw in &mut packet.lut.values {
			*raw = self.materials.get_or_add(Voxel::from(*raw));
		}
		self.tree.queue_edit_packet(packet);
	}

	// Apply all pending edits to the tree and compact.
	pub fn flush_edits(&mut self) {
		self.tree.apply_edits();
		self.tree.compact();
	}
}
