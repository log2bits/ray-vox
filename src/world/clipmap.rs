use std::array::from_fn;

pub struct Clipmap {
	pub occupancy: [DepthOccupancy; 11],
	pub origin: [i32; 3],
}

impl Clipmap {
	pub fn depth_origin(&self, depth: u8) -> [i32; 3] {
		let chunk_size = chunk_size_at_depth(depth) as i64;
		from_fn(|i| {
			let origin = self.origin[i] as i64;
			let snapped = ((origin + chunk_size / 2) / chunk_size) * chunk_size;
			(snapped - chunk_size * 4) as i32
		})
	}
}

pub struct DepthOccupancy {
	pub bits: [u32; 16],
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
	256 * 4i32.pow((10 - depth) as u32)
}

pub const fn total_chunk_count() -> usize {
	let per_depth = 8 * 8 * 8 - 2 * 2 * 2;
	per_depth * 11 // 5544
}
