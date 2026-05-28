use std::ops::{Add, Index, Mul, Sub};

/// A position in world space, in world units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldPos([i32; 3]);

impl WorldPos {
	pub fn new(x: i32, y: i32, z: i32) -> Self {
		WorldPos([x, y, z])
	}
	pub fn splat(v: i32) -> Self {
		WorldPos([v, v, v])
	}

	pub fn x(self) -> i32 {
		self.0[0]
	}
	pub fn y(self) -> i32 {
		self.0[1]
	}
	pub fn z(self) -> i32 {
		self.0[2]
	}
	pub fn to_array(self) -> [i32; 3] {
		self.0
	}

	pub fn map(self, f: impl Fn(i32) -> i32) -> Self {
		WorldPos([f(self.0[0]), f(self.0[1]), f(self.0[2])])
	}

	/// Snap to the origin of the chunk containing this position at the given LOD.
	pub fn snap_to_chunk(self, lod: LodLevel) -> ChunkId {
		let cs = lod.chunk_size();
		ChunkId {
			origin: self.map(|v| v & !(cs - 1)),
			lod,
		}
	}

	/// Voxel coordinates within the chunk at the given LOD.
	pub fn to_chunk_pos(self, chunk_origin: WorldPos, lod: LodLevel) -> ChunkPos {
		let vs = lod.voxel_size();
		ChunkPos([
			((self[0] - chunk_origin[0]) / vs) as u8,
			((self[1] - chunk_origin[1]) / vs) as u8,
			((self[2] - chunk_origin[2]) / vs) as u8,
		])
	}
}

impl From<[i32; 3]> for WorldPos {
	fn from(arr: [i32; 3]) -> Self {
		WorldPos(arr)
	}
}

impl From<WorldPos> for [i32; 3] {
	fn from(p: WorldPos) -> Self {
		p.0
	}
}

impl Index<usize> for WorldPos {
	type Output = i32;
	fn index(&self, i: usize) -> &i32 {
		&self.0[i]
	}
}

impl Add for WorldPos {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
		WorldPos([self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2]])
	}
}

impl Sub for WorldPos {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self {
		WorldPos([self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2]])
	}
}

impl Mul<i32> for WorldPos {
	type Output = Self;
	fn mul(self, rhs: i32) -> Self {
		WorldPos([self[0] * rhs, self[1] * rhs, self[2] * rhs])
	}
}

/// A voxel position local to a chunk. Each axis is 0..=255.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChunkPos([u8; 3]);

impl ChunkPos {
	pub fn new(x: u8, y: u8, z: u8) -> Self {
		ChunkPos([x, y, z])
	}

	pub fn x(self) -> u8 {
		self.0[0]
	}
	pub fn y(self) -> u8 {
		self.0[1]
	}
	pub fn z(self) -> u8 {
		self.0[2]
	}
	pub fn to_array(self) -> [u8; 3] {
		self.0
	}
}

impl From<[u8; 3]> for ChunkPos {
	fn from(arr: [u8; 3]) -> Self {
		ChunkPos(arr)
	}
}

impl From<ChunkPos> for [u8; 3] {
	fn from(p: ChunkPos) -> Self {
		p.0
	}
}

impl Index<usize> for ChunkPos {
	type Output = u8;
	fn index(&self, i: usize) -> &u8 {
		&self.0[i]
	}
}

/// LOD level within the clipmap.
/// 0 is coarsest (2^28 world units per chunk), 10 is finest (256 world units per chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LodLevel(u8);

impl LodLevel {
	pub const COARSEST: Self = LodLevel(0);
	pub const FINEST: Self = LodLevel(10);

	pub fn new(level: u8) -> Self {
		debug_assert!(level <= 10, "LodLevel {level} out of range 0..=10");
		LodLevel(level)
	}

	/// Size of one chunk in world units at this LOD.
	pub fn chunk_size(self) -> i32 {
		1i32 << (28 - 2 * self.0 as u32)
	}

	/// Size of one voxel in world units at this LOD.
	/// Each chunk is always 256 voxels per axis.
	pub fn voxel_size(self) -> i32 {
		self.chunk_size() / 256
	}

	pub fn is_finest(self) -> bool {
		self == Self::FINEST
	}
	pub fn is_coarsest(self) -> bool {
		self == Self::COARSEST
	}

	pub fn finer(self) -> Option<Self> {
		(self.0 < 10).then(|| LodLevel(self.0 + 1))
	}
	pub fn coarser(self) -> Option<Self> {
		(self.0 > 0).then(|| LodLevel(self.0 - 1))
	}
}

impl From<u8> for LodLevel {
	fn from(v: u8) -> Self {
		LodLevel::new(v)
	}
}

impl From<LodLevel> for u8 {
	fn from(l: LodLevel) -> Self {
		l.0
	}
}

/// A chunk identified by its slot in the clipmap grid.
/// Encodes LOD + (x, y, z) slot (0..8 per axis) into a u16.
/// Ephemeral - shifts as the camera moves. Not stable across clipmap repositions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ChunkHandle(u16);

impl ChunkHandle {
	pub fn new(lod: LodLevel, x: u8, y: u8, z: u8) -> Self {
		debug_assert!(x < 8 && y < 8 && z < 8);
		ChunkHandle((u8::from(lod) as u16) << 9 | (x as u16) << 6 | (y as u16) << 3 | z as u16)
	}

	pub fn lod(self) -> LodLevel {
		LodLevel(((self.0 >> 9) & 0xF) as u8)
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

impl From<u16> for ChunkHandle {
	fn from(raw: u16) -> Self {
		ChunkHandle(raw)
	}
}

impl From<ChunkHandle> for u16 {
	fn from(h: ChunkHandle) -> Self {
		h.0
	}
}

/// A chunk identified by its world-space origin and LOD level.
/// Stable across clipmap repositions - use as the key for persistent chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkId {
	pub origin: WorldPos,
	pub lod: LodLevel,
}

impl ChunkId {
	pub fn new(origin: WorldPos, lod: LodLevel) -> Self {
		ChunkId { origin, lod }
	}

	/// The far corner of this chunk in world space.
	pub fn max_corner(self) -> WorldPos {
		self.origin + WorldPos::splat(self.lod.chunk_size())
	}
}
