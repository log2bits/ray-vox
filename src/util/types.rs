use std::array::from_fn;
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

	pub fn from_fn(f: impl Fn(usize) -> i32) -> Self {
		WorldPos(from_fn(f))
	}

	/// Snap to the origin of the chunk containing this position at the given LOD.
	pub fn chunk_id(self, lod: LodLevel) -> ChunkId {
		let chunk_size = lod.chunk_size();
		ChunkId {
			origin: self.map(|x| align_down(x, chunk_size)),
			lod,
		}
	}

	/// Voxel coordinates within the chunk at the given LOD.
	pub fn chunk_pos(self, chunk_origin: WorldPos, lod: LodLevel) -> ChunkPos {
		let voxel_size = lod.voxel_size();
		ChunkPos::from_fn(|i| ((self[i] - chunk_origin[i]) / voxel_size) as u8)
	}

	pub fn chunk_handle(self, lod: LodLevel) -> ChunkHandle {
		self.chunk_id(lod).handle()
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
	pub fn from_fn(f: impl Fn(usize) -> u8) -> Self {
		ChunkPos(from_fn(f))
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
	pub const GRID_SIZE: u32 = 8;
	pub const LEVELS: u8 = 11;
	pub const CHUNKS_PER_LEVEL: u32 = Self::GRID_SIZE * Self::GRID_SIZE * Self::GRID_SIZE;
	pub const COARSEST: LodLevel = LodLevel(0);
	pub const FINEST: LodLevel = LodLevel(Self::LEVELS - 1);

	pub fn new(level: u8) -> Self {
		debug_assert!(level < Self::LEVELS, "LodLevel {level} out of range");
		LodLevel(level)
	}

	pub fn level(self) -> u8 {
		self.0
	}

	/// Size of one chunk in world units at this LOD.
	pub fn chunk_size(self) -> i32 {
		256 * 4i32.pow((Self::LEVELS as u32 - 1) - self.0 as u32)
	}

	/// Size of one voxel in world units at this LOD.
	/// Each chunk is always 256 voxels per axis.
	pub fn voxel_size(self) -> i32 {
		self.chunk_size() / 256
	}

	pub fn is_coarsest(self) -> bool {
		self == Self::COARSEST
	}

	pub fn is_finest(self) -> bool {
		self == Self::FINEST
	}

	pub fn level_origin(self, camera_pos: WorldPos) -> WorldPos {
		let chunk_size = self.chunk_size();
		let half = chunk_size / 2;
		let total_span = chunk_size * LodLevel::GRID_SIZE as i32;
		let snapped = camera_pos.chunk_id(self).origin;

		WorldPos::from_fn(|i| {
			let offset = if camera_pos[i] - snapped[i] < half {
				4
			} else {
				3
			};
			(snapped[i] - chunk_size * offset).clamp(i32::MIN, i32::MAX - total_span)
		})
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
/// Slots are fixed to world space via toroidal addressing: slot = (chunk_origin / chunk_size) % 8.
/// Stable across camera movement - only validity (is_in_range) changes as the camera moves.
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

	/// World-space origin of this slot. Independent of camera position.
	pub fn world_origin(self) -> WorldPos {
		let chunk_size = self.lod().chunk_size();
		let xyz = self.xyz();
		WorldPos::from_fn(|i| xyz[i] as i32 * chunk_size)
	}

	pub fn id(self) -> ChunkId {
		ChunkId::new(self.world_origin(), self.lod())
	}

	/// Whether this slot's world region is currently within the camera's view window.
	pub fn is_in_range(self, camera_pos: WorldPos) -> bool {
		let lod = self.lod();
		let level_min = lod.level_origin(camera_pos);
		let level_max = level_min + WorldPos::splat(lod.chunk_size() * LodLevel::GRID_SIZE as i32);
		let origin = self.world_origin();
		(0..3).all(|i| origin[i] >= level_min[i] && origin[i] < level_max[i])
	}

	pub fn bit_index(self) -> u32 {
		(self.x() as u32)
			+ self.y() as u32 * LodLevel::GRID_SIZE
			+ self.z() as u32 * LodLevel::GRID_SIZE.pow(2)
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

	/// Returns the slot handle for this chunk. Always succeeds - use `handle.is_in_range(camera_pos)`
	/// to check whether the slot is currently within the camera's view window.
	pub fn handle(self) -> ChunkHandle {
		let chunk_size = self.lod.chunk_size();
		let [x, y, z] = std::array::from_fn(|i| {
			(self.origin[i] / chunk_size).rem_euclid(LodLevel::GRID_SIZE as i32) as u8
		});
		ChunkHandle::new(self.lod, x, y, z)
	}
}

pub struct LodLevelBitmask([u32; LodLevel::CHUNKS_PER_LEVEL as usize / 32]);

impl LodLevelBitmask {
	pub const fn new() -> Self {
		LodLevelBitmask([0; LodLevel::CHUNKS_PER_LEVEL as usize / 32])
	}

	pub fn get(&self, bit: u32) -> bool {
		(self.0[bit as usize / 32] >> (bit % 32)) & 1 == 1
	}

	pub fn set(&mut self, bit: u32) {
		self.0[bit as usize / 32] |= 1 << (bit % 32);
	}

	pub fn clear(&mut self, bit: u32) {
		self.0[bit as usize / 32] &= !(1 << (bit % 32));
	}
}

/// Needs alignment to always be a power of 2
#[inline(always)]
fn align_down(v: i32, alignment: i32) -> i32 {
	v & !(alignment - 1)
}
