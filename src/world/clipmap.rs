use std::array::from_fn;

pub struct Clipmap {
	pub occupancy: [LevelOccupancy; 11],
	pub origin: [i32; 3],
}

impl Clipmap {
	pub fn level_origin(&self, level: u8) -> [i32; 3] {
		let chunk_size = chunk_size_at_level(level) as i64;
		from_fn(|i| {
			let origin = self.origin[i] as i64;
			let snapped = ((origin + chunk_size / 2) / chunk_size) * chunk_size;
			(snapped - chunk_size * 4) as i32
		})
	}
}

pub struct LevelOccupancy {
	pub bits: [u32; 16],
}

/// A u16 handle encoding a chunk's clipmap position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ChunkHandle(u16);

impl ChunkHandle {
	pub fn new(level: u8, x: u8, y: u8, z: u8) -> Self {
		debug_assert!(level < 11);
		debug_assert!(x < 8 && y < 8 && z < 8);
		ChunkHandle(((level as u16) << 9) | ((x as u16) << 6) | ((y as u16) << 3) | (z as u16))
	}

	pub fn level(self) -> u8 {
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

pub fn chunk_size_at_level(level: u8) -> i32 {
	256 * 4i32.pow(level as u32)
}

pub const fn total_chunk_count() -> usize {
	let per_level = 8 * 8 * 8 - 2 * 2 * 2;
	per_level * 11 // 5544
}
