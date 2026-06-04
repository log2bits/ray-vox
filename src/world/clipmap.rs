use crate::util::types::{ChunkHandle, ChunkId, LodLevel, LodLevelBitmask, WorldPos};
use std::array::from_fn;

pub struct Clipmap {
	pub occupancy: [LodLevelBitmask; LodLevel::LEVELS as usize],
	pub camera_pos: WorldPos,
	pub pending_remap: Vec<RemapOp>,
	pub pending_camera_pos: WorldPos,
}

pub enum RemapOp {
	Split((ChunkHandle, [Option<ChunkHandle>; 64])),
	Merge([Option<ChunkHandle>; 64], ChunkHandle),
	Add(ChunkHandle),
	Delete(ChunkHandle),
}

impl Clipmap {
	pub fn set_origin(&mut self, new_camera_pos: WorldPos) {
		// TODO
		// Lots of fun logic in the future!
		self.pending_camera_pos = new_camera_pos;
	}

	pub fn remap_completed(&mut self, index: u32) {
		self.pending_remap.remove(index as usize);
		if self.pending_remap.is_empty() {
			self.camera_pos = self.pending_camera_pos;
		}
	}

	pub fn is_occupied(&self, handle: ChunkHandle) -> bool {
		self.occupancy[handle.lod().level() as usize].get(handle.bit_index())
	}

	pub fn set_occupied(&mut self, handle: ChunkHandle) {
		self.occupancy[handle.lod().level() as usize].set(handle.bit_index());
	}

	pub fn clear_occupied(&mut self, handle: ChunkHandle) {
		self.occupancy[handle.lod().level() as usize].clear(handle.bit_index());
	}
}

pub const fn total_chunk_count() -> usize {
	LodLevel::CHUNKS_PER_LEVEL as usize * LodLevel::LEVELS as usize
}
