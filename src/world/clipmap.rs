use std::array::from_fn;
use crate::world::WorldPosition;

pub struct Clipmap {
	pub occupancy: [[u32; 16]; 11],
	pub origin: WorldPosition,
	pub pending_remap: Vec<RemapOp>,
	pub pending_origin: WorldPosition,
}

pub enum RemapOp {
	Move(ChunkHandle, ChunkHandle),
	Split((ChunkHandle, [ChunkHandle; 64])),
	Merge([ChunkHandle; 64], ChunkHandle),
	Add(ChunkHandle),
	Delete(ChunkHandle),
}

impl Clipmap {
	pub fn level_origin(&self, depth: u8) -> WorldPosition {
		if depth == 0 {
			return WorldPosition { position: [-(1i32 << 30); 3] };
		}
		let cs = chunk_size_at_depth(depth);
		WorldPosition {
			position: from_fn(|i| {
				let snapped = (self.origin.position[i] + (cs >> 1)) & !(cs - 1);
				snapped - (cs << 2)
			}),
		}
	}

	pub fn set_origin(&mut self, new_origin: WorldPosition) {
		let new_level_origin_10 = {
			let saved = self.origin;
			self.origin = new_origin;
			let r = self.level_origin(10);
			self.origin = saved;
			r
		};
		if new_level_origin_10 == self.level_origin(10) {
			self.pending_origin = new_origin;
			return;
		}

		let old_origins: [WorldPosition; 11] = from_fn(|d| self.level_origin(d as u8));
		let new_origins: [WorldPosition; 11] = {
			let saved = self.origin;
			self.origin = new_origin;
			let r = from_fn(|d| self.level_origin(d as u8));
			self.origin = saved;
			r
		};

		let mut ops: Vec<RemapOp> = Vec::new();

		for depth in 0..11u8 {
			let chunk_size = chunk_size_at_depth(depth) as i64;
			let old_level_origin = old_origins[depth as usize];
			let new_level_origin = new_origins[depth as usize];

			// Scan old active slots: emit Move, Split, or Delete.
			for old_slot in active_slots() {
				let [old_x, old_y, old_z] = old_slot;
				let old_handle = ChunkHandle::new(depth, old_x, old_y, old_z);
				if !self.is_occupied(old_handle) {
					continue;
				}

				let world = [
					old_level_origin.position[0] as i64 + old_x as i64 * chunk_size,
					old_level_origin.position[1] as i64 + old_y as i64 * chunk_size,
					old_level_origin.position[2] as i64 + old_z as i64 * chunk_size,
				];

				let new_slot = [
					((world[0] - new_level_origin.position[0] as i64) / chunk_size) as i32,
					((world[1] - new_level_origin.position[1] as i64) / chunk_size) as i32,
					((world[2] - new_level_origin.position[2] as i64) / chunk_size) as i32,
				];

				if slot_in_bounds(new_slot) {
					if in_cutout(new_slot) && depth < 10 {
						let fine_handles = fine_handles_covering(
							world,
							depth + 1,
							new_origins[depth as usize + 1],
						);
						ops.push(RemapOp::Split((old_handle, fine_handles)));
					} else if !in_cutout(new_slot) {
						let new_handle = ChunkHandle::new(
							depth,
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

			// Scan new active slots: emit Merge or Add.
			for new_slot in active_slots() {
				let [new_x, new_y, new_z] = new_slot;
				let new_handle = ChunkHandle::new(depth, new_x, new_y, new_z);

				let world = [
					new_level_origin.position[0] as i64 + new_x as i64 * chunk_size,
					new_level_origin.position[1] as i64 + new_y as i64 * chunk_size,
					new_level_origin.position[2] as i64 + new_z as i64 * chunk_size,
				];

				let old_slot = [
					((world[0] - old_level_origin.position[0] as i64) / chunk_size) as i32,
					((world[1] - old_level_origin.position[1] as i64) / chunk_size) as i32,
					((world[2] - old_level_origin.position[2] as i64) / chunk_size) as i32,
				];

				if slot_in_bounds(old_slot) {
					if in_cutout(old_slot) && depth < 10 {
						let fine_handles = fine_handles_covering(
							world,
							depth + 1,
							old_origins[depth as usize + 1],
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

		self.pending_remap = ops;
		self.pending_origin = new_origin;
	}

	fn is_occupied(&self, handle: ChunkHandle) -> bool {
		let index = handle.x() as u32 + handle.y() as u32 * 8 + handle.z() as u32 * 64;
		let block = (index / 32) as usize;
		let offset = index % 32;
		self.occupancy[handle.depth() as usize][block] & (1u32 << offset) != 0
	}

	fn any_occupied(&self, handles: &[ChunkHandle; 64]) -> bool {
		handles.iter().any(|&h| self.is_occupied(h))
	}

	pub fn remap_completed(&mut self, index: u32) {
		self.pending_remap.remove(index as usize);
		if self.pending_remap.is_empty() {
			self.origin = self.pending_origin;
		}
	}

	pub fn set_occupied(&mut self, handle: ChunkHandle) {
		let index = handle.x() + (handle.y() * 8) + (handle.z() * 64);
		let offset = index % 32;
		let block = index / 32;
		let mask = 1u32 << offset;
		self.occupancy[handle.depth() as usize][block as usize] |= mask;
	}
}

/// A u16 handle encoding a chunk's clipmap position.
/// Depth 0 is the coarsest level, depth 10 is the finest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ChunkHandle(u16);

impl ChunkHandle {
	pub fn new(depth: u8, x: u8, y: u8, z: u8) -> Self {
		debug_assert!(depth < 11);
		debug_assert!(x < 8 && y < 8 && z < 8);
		ChunkHandle(((depth as u16) << 9) | ((x as u16) << 6) | ((y as u16) << 3) | (z as u16))
	}

	pub fn depth(self) -> u8 {
		((self.0 >> 9) & 0xF) as u8
	}

	pub fn x(self) -> u8 {
		((self.0 >> 6) & 0x7) as u8
	}

	pub fn y(self) -> u8 {
		((self.0 >> 3) & 0x7) as u8
	}

	pub fn z(self) -> u8 {
		(self.0 & 0x7) as u8
	}

	pub fn xyz(self) -> [u8; 3] {
		[self.x(), self.y(), self.z()]
	}
}

/// Depth 0 is the coarsest (chunk covers 2^28 world units).
/// Depth 10 is the finest (chunk covers 256 world units).
pub fn chunk_size_at_depth(depth: u8) -> i32 {
	1i32 << (28 - 2 * depth as u32)
}

pub const fn total_chunk_count() -> usize {
	let per_depth = 8 * 8 * 8 - 2 * 2 * 2;
	per_depth * 11 // 5544
}

fn in_cutout(slot: [i32; 3]) -> bool {
	slot.iter().all(|&v| v == 3 || v == 4)
}

fn slot_in_bounds(slot: [i32; 3]) -> bool {
	slot.iter().all(|&v| v >= 0 && v < 8)
}

fn active_slots() -> impl Iterator<Item = [u8; 3]> {
	(0u8..8)
		.flat_map(|z| (0u8..8).flat_map(move |y| (0u8..8).map(move |x| [x, y, z])))
		.filter(|&[x, y, z]| !in_cutout([x as i32, y as i32, z as i32]))
}

fn fine_handles_covering(
	coarse_world: [i64; 3],
	fine_depth: u8,
	fine_level_origin: WorldPosition,
) -> [ChunkHandle; 64] {
	let fine_chunk_size = chunk_size_at_depth(fine_depth) as i64;
	from_fn(|i| {
		let offset = [i % 4, (i / 4) % 4, i / 16];
		let fine_world = [
			coarse_world[0] + offset[0] as i64 * fine_chunk_size,
			coarse_world[1] + offset[1] as i64 * fine_chunk_size,
			coarse_world[2] + offset[2] as i64 * fine_chunk_size,
		];
		let slot = [
			((fine_world[0] - fine_level_origin.position[0] as i64) / fine_chunk_size) as u8,
			((fine_world[1] - fine_level_origin.position[1] as i64) / fine_chunk_size) as u8,
			((fine_world[2] - fine_level_origin.position[2] as i64) / fine_chunk_size) as u8,
		];
		ChunkHandle::new(fine_depth, slot[0], slot[1], slot[2])
	})
}
