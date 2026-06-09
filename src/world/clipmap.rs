use crate::util::types::{ChunkHandle, ChunkId, LodLevel, LodLevelBitmask, WorldPos};
use std::collections::HashMap;

pub struct Clipmap {
	pub occupancy: [LodLevelBitmask; LodLevel::LEVELS as usize],
	pub resident: HashMap<ChunkHandle, ChunkId>,
	pub camera_pos: WorldPos,
	pub pending_remap: Vec<RemapOp>,
	pub pending_camera_pos: WorldPos,
}

#[derive(Clone)]
pub enum RemapOp {
	Add(ChunkHandle, ChunkId),
	Delete(ChunkHandle),
}

impl RemapOp {
	pub fn handle(&self) -> ChunkHandle {
		match self {
			RemapOp::Add(h, _) | RemapOp::Delete(h) => *h,
		}
	}
}

impl Clipmap {
	pub fn new() -> Self {
		Self {
			occupancy: std::array::from_fn(|_| LodLevelBitmask::new()),
			resident: HashMap::new(),
			camera_pos: WorldPos::new(0, 0, 0),
			pending_remap: Vec::new(),
			pending_camera_pos: WorldPos::new(0, 0, 0),
		}
	}

	pub fn set_origin(&mut self, new_camera_pos: WorldPos) {
		self.pending_camera_pos = new_camera_pos;

		for level in 0..LodLevel::LEVELS {
			let lod = LodLevel::new(level);
			let new_origin = lod.level_origin(new_camera_pos);
			let chunk_size = lod.chunk_size();

			for x in 0..LodLevel::GRID_SIZE {
				for y in 0..LodLevel::GRID_SIZE {
					for z in 0..LodLevel::GRID_SIZE {
						let new_world_origin = WorldPos::new(
							new_origin.x() + (x as i32) * chunk_size,
							new_origin.y() + (y as i32) * chunk_size,
							new_origin.z() + (z as i32) * chunk_size,
						);
						let new_id = ChunkId::new(new_world_origin, lod);
						let handle = new_id.handle();

						match self.resident.get(&handle).copied() {
							Some(prev) if prev == new_id => {}
							Some(_) => {
								self.pending_remap.push(RemapOp::Delete(handle));
								self.pending_remap.push(RemapOp::Add(handle, new_id));
							}
							None => {
								self.pending_remap.push(RemapOp::Add(handle, new_id));
							}
						}
					}
				}
			}
		}
	}

	pub fn apply_remap(&mut self, op: &RemapOp) {
		match op {
			RemapOp::Add(handle, chunk_id) => self.assign(*handle, *chunk_id),
			RemapOp::Delete(handle) => {
				self.evict(*handle);
			}
		}
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

	pub fn assign(&mut self, handle: ChunkHandle, chunk_id: ChunkId) {
		self.resident.insert(handle, chunk_id);
		self.occupancy[handle.lod().level() as usize].set(handle.bit_index());
	}

	pub fn evict(&mut self, handle: ChunkHandle) -> Option<ChunkId> {
		let prev = self.resident.remove(&handle);
		if prev.is_some() {
			self.occupancy[handle.lod().level() as usize].clear(handle.bit_index());
		}
		prev
	}

	pub fn chunk_id_of(&self, handle: ChunkHandle) -> Option<ChunkId> {
		self.resident.get(&handle).copied()
	}

	pub fn resident_chunks(&self) -> impl Iterator<Item = (ChunkHandle, ChunkId)> + '_ {
		self.resident.iter().map(|(&h, &id)| (h, id))
	}
}

impl Default for Clipmap {
	fn default() -> Self {
		Self::new()
	}
}

pub const fn total_chunk_count() -> usize {
	LodLevel::CHUNKS_PER_LEVEL as usize * LodLevel::LEVELS as usize
}
