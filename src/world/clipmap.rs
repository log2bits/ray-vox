use crate::util::types::{ChunkHandle, ChunkId, LodLevel, WorldPos};
use std::array::from_fn;

pub struct Clipmap {
	pub occupancy: [[u32; 16]; 11],
	pub origin: WorldPos,
	pub pending_remap: Vec<RemapOp>,
	pub pending_origin: WorldPos,
}

pub enum RemapOp {
	Move(ChunkHandle, ChunkHandle),
	Split((ChunkHandle, [ChunkHandle; 64])),
	Merge([ChunkHandle; 64], ChunkHandle),
	Add(ChunkHandle),
	Delete(ChunkHandle),
}

impl Clipmap {
	/// World-space origin of the chunk grid at the given LOD level.
	pub fn level_origin(&self, lod: LodLevel) -> WorldPos {
		if lod.is_coarsest() {
			return WorldPos::splat(-(1i32 << 30));
		}
		let cs = lod.chunk_size();
		WorldPos::from(from_fn(|i| {
			let snapped = (self.origin[i] + (cs >> 1)) & !(cs - 1);
			snapped - (cs << 2)
		}))
	}

	/// Returns true if the given chunk falls within the cutout —
	/// the region covered by the next finer LOD level.
	pub fn in_cutout(&self, id: ChunkId) -> bool {
		let Some(fine_lod) = id.lod.finer() else {
			return false;
		};
		let fine_origin = self.level_origin(fine_lod);
		let cs = id.lod.chunk_size();
		(0..3)
			.all(|i| id.origin[i] >= fine_origin[i] && id.origin[i] + cs <= fine_origin[i] + 2 * cs)
	}

	pub fn handle_world_origin(&self, handle: ChunkHandle) -> WorldPos {
		let cs = handle.lod().chunk_size();
		self.level_origin(handle.lod())
			+ WorldPos::new(
				handle.x() as i32 * cs,
				handle.y() as i32 * cs,
				handle.z() as i32 * cs,
			)
	}

	pub fn handle_to_id(&self, handle: ChunkHandle) -> ChunkId {
		ChunkId::new(self.handle_world_origin(handle), handle.lod())
	}

	/// Returns None if the chunk falls outside the current clipmap window.
	pub fn id_to_handle(&self, id: ChunkId) -> Option<ChunkHandle> {
		let level_origin = self.level_origin(id.lod);
		let cs = id.lod.chunk_size();
		let coords = from_fn::<i32, 3, _>(|i| (id.origin[i] - level_origin[i]) / cs);
		slot_in_bounds(coords)
			.then(|| ChunkHandle::new(id.lod, coords[0] as u8, coords[1] as u8, coords[2] as u8))
	}

	/// Returns None if the position falls outside the current clipmap window.
	pub fn world_to_handle(&self, pos: WorldPos, lod: LodLevel) -> Option<ChunkHandle> {
		let level_origin = self.level_origin(lod);
		let cs = lod.chunk_size();
		let coords = from_fn::<i32, 3, _>(|i| (pos[i] - level_origin[i]) / cs);
		slot_in_bounds(coords)
			.then(|| ChunkHandle::new(lod, coords[0] as u8, coords[1] as u8, coords[2] as u8))
	}

	pub fn set_origin(&mut self, new_origin: WorldPos) {
		todo!();

		self.pending_origin = new_origin;
	}

	pub fn remap_completed(&mut self, index: u32) {
		self.pending_remap.remove(index as usize);
		if self.pending_remap.is_empty() {
			self.origin = self.pending_origin;
		}
	}

	pub fn is_occupied(&self, handle: ChunkHandle) -> bool {
		todo!();
	}

	pub fn set_occupied(&mut self, handle: ChunkHandle) {
		todo!();
	}
}

pub const fn total_chunk_count() -> usize {
	let per_level = 8 * 8 * 8;
	per_level * 11 // 5632
}
