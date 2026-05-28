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
		let new_level_origin_finest = {
			let saved = self.origin;
			self.origin = new_origin;
			let r = self.level_origin(LodLevel::FINEST);
			self.origin = saved;
			r
		};
		if new_level_origin_finest == self.level_origin(LodLevel::FINEST) {
			self.pending_origin = new_origin;
			return;
		}

		let old_origins: [WorldPos; 11] = from_fn(|d| self.level_origin(LodLevel::new(d as u8)));
		let new_origins: [WorldPos; 11] = {
			let saved = self.origin;
			self.origin = new_origin;
			let r = from_fn(|d| self.level_origin(LodLevel::new(d as u8)));
			self.origin = saved;
			r
		};

		let mut ops: Vec<RemapOp> = Vec::new();

		for d in 0..11u8 {
			let lod = LodLevel::new(d);
			let chunk_size = lod.chunk_size() as i64;
			let old_origin = old_origins[d as usize];
			let new_origin_d = new_origins[d as usize];

			// Helpers to check cutout membership against old and new configs
			// without mutating self.origin inside the loop.
			let in_old_cutout = |world: [i64; 3]| -> bool {
				let Some(fine_lod) = lod.finer() else {
					return false;
				};
				let fine_origin = old_origins[d as usize + 1];
				let cs = lod.chunk_size() as i64;
				(0..3).all(|i| {
					world[i] >= fine_origin[i] as i64
						&& world[i] + cs <= fine_origin[i] as i64 + 2 * cs
				})
			};
			let in_new_cutout = |world: [i64; 3]| -> bool {
				let Some(fine_lod) = lod.finer() else {
					return false;
				};
				let fine_origin = new_origins[d as usize + 1];
				let cs = lod.chunk_size() as i64;
				(0..3).all(|i| {
					world[i] >= fine_origin[i] as i64
						&& world[i] + cs <= fine_origin[i] as i64 + 2 * cs
				})
			};

			// Scan old slots: emit Move, Split, or Delete.
			for x in 0u8..8 {
				for y in 0u8..8 {
					for z in 0u8..8 {
						let old_handle = ChunkHandle::new(lod, x, y, z);
						if !self.is_occupied(old_handle) {
							continue;
						}

						let world = [
							old_origin[0] as i64 + x as i64 * chunk_size,
							old_origin[1] as i64 + y as i64 * chunk_size,
							old_origin[2] as i64 + z as i64 * chunk_size,
						];

						let new_slot = from_fn::<i32, 3, _>(|i| {
							((world[i] - new_origin_d[i] as i64) / chunk_size) as i32
						});

						if slot_in_bounds(new_slot) {
							if in_new_cutout(world) {
								let fine_lod = lod.finer().unwrap();
								let fine_handles = fine_handles_covering(
									world,
									fine_lod,
									new_origins[d as usize + 1],
								);
								ops.push(RemapOp::Split((old_handle, fine_handles)));
							} else {
								let new_handle = ChunkHandle::new(
									lod,
									new_slot[0] as u8,
									new_slot[1] as u8,
									new_slot[2] as u8,
								);
								if old_handle != new_handle {
									ops.push(RemapOp::Move(old_handle, new_handle));
								}
							}
						} else {
							ops.push(RemapOp::Delete(old_handle));
						}
					}
				}
			}

			// Scan new slots: emit Merge or Add.
			for x in 0u8..8 {
				for y in 0u8..8 {
					for z in 0u8..8 {
						let world = [
							new_origin_d[0] as i64 + x as i64 * chunk_size,
							new_origin_d[1] as i64 + y as i64 * chunk_size,
							new_origin_d[2] as i64 + z as i64 * chunk_size,
						];

						if in_new_cutout(world) {
							continue;
						}

						let new_handle = ChunkHandle::new(lod, x, y, z);
						let old_slot = from_fn::<i32, 3, _>(|i| {
							((world[i] - old_origin[i] as i64) / chunk_size) as i32
						});

						if slot_in_bounds(old_slot) {
							if in_old_cutout(world) {
								let fine_lod = lod.finer().unwrap();
								let fine_handles = fine_handles_covering(
									world,
									fine_lod,
									old_origins[d as usize + 1],
								);
								if self.any_occupied(&fine_handles) {
									ops.push(RemapOp::Merge(fine_handles, new_handle));
								}
							}
							// else: maps to old active slot, already emitted as Move above
						} else {
							ops.push(RemapOp::Add(new_handle));
						}
					}
				}
			}
		}

		self.pending_remap = ops;
		self.pending_origin = new_origin;
	}

	pub fn remap_completed(&mut self, index: u32) {
		self.pending_remap.remove(index as usize);
		if self.pending_remap.is_empty() {
			self.origin = self.pending_origin;
		}
	}

	pub fn is_occupied(&self, handle: ChunkHandle) -> bool {
		let index = handle.x() as u32 + handle.y() as u32 * 8 + handle.z() as u32 * 64;
		self.occupancy[u8::from(handle.lod()) as usize][(index / 32) as usize]
			& (1u32 << (index % 32))
			!= 0
	}

	pub fn set_occupied(&mut self, handle: ChunkHandle) {
		let index = handle.x() as u32 + handle.y() as u32 * 8 + handle.z() as u32 * 64;
		self.occupancy[u8::from(handle.lod()) as usize][(index / 32) as usize] |=
			1u32 << (index % 32);
	}

	fn any_occupied(&self, handles: &[ChunkHandle; 64]) -> bool {
		handles.iter().any(|&h| self.is_occupied(h))
	}
}

pub const fn total_chunk_count() -> usize {
	let per_level = 8 * 8 * 8 - 2 * 2 * 2; // 8^3 minus inner 2^3 cutout
	per_level * 11 // 5544
}

fn slot_in_bounds(slot: [i32; 3]) -> bool {
	slot.iter().all(|&v| v >= 0 && v < 8)
}

fn fine_handles_covering(
	coarse_world: [i64; 3],
	fine_lod: LodLevel,
	fine_level_origin: WorldPos,
) -> [ChunkHandle; 64] {
	let fine_chunk_size = fine_lod.chunk_size() as i64;
	from_fn(|i| {
		let offset = [i % 4, (i / 4) % 4, i / 16];
		let fine_world = [
			coarse_world[0] + offset[0] as i64 * fine_chunk_size,
			coarse_world[1] + offset[1] as i64 * fine_chunk_size,
			coarse_world[2] + offset[2] as i64 * fine_chunk_size,
		];
		let slot = from_fn::<u8, 3, _>(|j| {
			((fine_world[j] - fine_level_origin[j] as i64) / fine_chunk_size) as u8
		});
		ChunkHandle::new(fine_lod, slot[0], slot[1], slot[2])
	})
}
