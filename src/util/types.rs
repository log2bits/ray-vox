use bytemuck::{Pod, Zeroable};
use std::array::from_fn;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, Mul, Not, Sub};

/// A set of slots indexed 0..63. Wraps u64 with iteration and popcount helpers.
/// repr(transparent) so it can stand in for u64 in GPU-uploaded structs.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Mask64(pub u64);

impl Mask64 {
	pub const EMPTY: Self = Mask64(0);
	pub const FULL: Self = Mask64(u64::MAX);

	#[inline]
	pub fn bit(slot: u8) -> Self {
		Mask64(1u64 << slot)
	}

	#[inline]
	pub fn raw(self) -> u64 {
		self.0
	}

	#[inline]
	pub fn contains(self, slot: u8) -> bool {
		(self.0 >> slot) & 1 != 0
	}

	#[inline]
	pub fn count(self) -> u32 {
		self.0.count_ones()
	}

	#[inline]
	pub fn is_empty(self) -> bool {
		self.0 == 0
	}

	/// Count of bits strictly below slot.
	#[inline]
	pub fn popcount_below(self, slot: u8) -> u32 {
		(self.0 & ((1u64 << slot) - 1)).count_ones()
	}

	/// Yield set slot indices in ascending order.
	#[inline]
	pub fn iter_slots(self) -> Mask64Iter {
		Mask64Iter(self.0)
	}
}

pub struct Mask64Iter(u64);

impl Iterator for Mask64Iter {
	type Item = u8;
	#[inline]
	fn next(&mut self) -> Option<u8> {
		if self.0 == 0 {
			return None;
		}
		let slot = self.0.trailing_zeros() as u8;
		self.0 &= self.0 - 1;
		Some(slot)
	}
}

impl From<u64> for Mask64 {
	#[inline]
	fn from(v: u64) -> Self {
		Mask64(v)
	}
}

impl From<Mask64> for u64 {
	#[inline]
	fn from(m: Mask64) -> Self {
		m.0
	}
}

impl BitAnd for Mask64 {
	type Output = Self;
	#[inline]
	fn bitand(self, rhs: Self) -> Self {
		Mask64(self.0 & rhs.0)
	}
}

impl BitOr for Mask64 {
	type Output = Self;
	#[inline]
	fn bitor(self, rhs: Self) -> Self {
		Mask64(self.0 | rhs.0)
	}
}

impl BitXor for Mask64 {
	type Output = Self;
	#[inline]
	fn bitxor(self, rhs: Self) -> Self {
		Mask64(self.0 ^ rhs.0)
	}
}

impl Not for Mask64 {
	type Output = Self;
	#[inline]
	fn not(self) -> Self {
		Mask64(!self.0)
	}
}

impl BitAndAssign for Mask64 {
	#[inline]
	fn bitand_assign(&mut self, rhs: Self) {
		self.0 &= rhs.0;
	}
}

impl BitOrAssign for Mask64 {
	#[inline]
	fn bitor_assign(&mut self, rhs: Self) {
		self.0 |= rhs.0;
	}
}

impl BitXorAssign for Mask64 {
	#[inline]
	fn bitxor_assign(&mut self, rhs: Self) {
		self.0 ^= rhs.0;
	}
}

/// Position in world units.
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

	/// Origin of the chunk containing this position at the given LOD.
	pub fn chunk_id(self, lod: LodLevel) -> ChunkId {
		let chunk_size = lod.chunk_size();
		ChunkId {
			origin: self.map(|x| align_down(x, chunk_size)),
			lod,
		}
	}

	/// Voxel coordinates within the chunk.
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

/// Voxel position local to a chunk. Each axis is 0..=255.
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

/// LOD level. 0 is coarsest (2^28 units per chunk), 10 is finest (256 units per chunk).
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

	/// Size of one chunk in world units.
	pub fn chunk_size(self) -> i32 {
		256 * 4i32.pow((Self::LEVELS as u32 - 1) - self.0 as u32)
	}

	/// Size of one voxel in world units. Each chunk is always 256 voxels per axis.
	pub fn voxel_size(self) -> i32 {
		self.chunk_size() / 256
	}

	pub fn is_coarsest(self) -> bool {
		self == Self::COARSEST
	}

	pub fn is_finest(self) -> bool {
		self == Self::FINEST
	}

	/// One LOD coarser, or None at the coarsest level.
	pub fn coarser(self) -> Option<LodLevel> {
		if self.is_coarsest() { None } else { Some(LodLevel(self.0 - 1)) }
	}

	/// One LOD finer, or None at the finest level.
	pub fn finer(self) -> Option<LodLevel> {
		if self.is_finest() { None } else { Some(LodLevel(self.0 + 1)) }
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

/// Chunk handle: LOD + (x, y, z) slot (0..8 per axis) packed into a u16. Slots are
/// fixed to world space via toroidal addressing (slot = origin / chunk_size mod 8).
/// Stable across camera movement, only is_in_range changes.
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

	/// World-space origin of this slot.
	pub fn world_origin(self) -> WorldPos {
		let chunk_size = self.lod().chunk_size();
		let xyz = self.xyz();
		WorldPos::from_fn(|i| xyz[i] as i32 * chunk_size)
	}

	pub fn id(self) -> ChunkId {
		ChunkId::new(self.world_origin(), self.lod())
	}

	/// True if this slot's region lies inside the camera's current view window.
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

/// Chunk identified by world-space origin and LOD. Stable across clipmap moves,
/// use as the key for persistent chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkId {
	pub origin: WorldPos,
	pub lod: LodLevel,
}

impl ChunkId {
	pub fn new(origin: WorldPos, lod: LodLevel) -> Self {
		ChunkId { origin, lod }
	}

	/// Far corner of this chunk.
	pub fn max_corner(self) -> WorldPos {
		self.origin + WorldPos::splat(self.lod.chunk_size())
	}

	/// Slot handle for this chunk. Always succeeds, use is_in_range to check validity.
	pub fn handle(self) -> ChunkHandle {
		let chunk_size = self.lod.chunk_size();
		let [x, y, z] = std::array::from_fn(|i| {
			(self.origin[i] / chunk_size).rem_euclid(LodLevel::GRID_SIZE as i32) as u8
		});
		ChunkHandle::new(self.lod, x, y, z)
	}

	/// Bounding box of this chunk.
	pub fn aabb(self) -> Aabb {
		Aabb { min: self.origin, max: self.max_corner() }
	}

	/// One-step-coarser chunk that contains this one. None at the coarsest level.
	pub fn parent(self) -> Option<ChunkId> {
		let parent_lod = self.lod.coarser()?;
		Some(ChunkId {
			origin: self.origin.map(|c| align_down(c, parent_lod.chunk_size())),
			lod: parent_lod,
		})
	}

	/// The 64 finer chunks that tile this one. None at the finest level. Slot order
	/// matches Path::from_coords (slot = ((x&3) << 4) | ((y&3) << 2) | (z&3)) so the
	/// array feeds straight into merge_lod.
	pub fn children(self) -> Option<[ChunkId; 64]> {
		let child_lod = self.lod.finer()?;
		let child_size = child_lod.chunk_size();
		Some(std::array::from_fn(|slot| {
			let x = ((slot >> 4) & 3) as i32;
			let y = ((slot >> 2) & 3) as i32;
			let z = (slot & 3) as i32;
			ChunkId {
				origin: WorldPos::from_fn(|i| self.origin[i] + [x, y, z][i] * child_size),
				lod: child_lod,
			}
		}))
	}
}

/// World-space axis-aligned bounding box. Half-open: min inclusive, max exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Aabb {
	pub min: WorldPos,
	pub max: WorldPos,
}

impl Aabb {
	pub fn new(min: WorldPos, max: WorldPos) -> Self {
		Self { min, max }
	}

	pub fn from_chunk(id: ChunkId) -> Self {
		id.aabb()
	}

	pub fn intersects(&self, other: &Aabb) -> bool {
		(0..3).all(|i| self.min[i] < other.max[i] && other.min[i] < self.max[i])
	}

	pub fn contains(&self, p: WorldPos) -> bool {
		(0..3).all(|i| self.min[i] <= p[i] && p[i] < self.max[i])
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

/// alignment must be a power of two.
#[inline(always)]
fn align_down(v: i32, alignment: i32) -> i32 {
	v & !(alignment - 1)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn pos(x: i32, y: i32, z: i32) -> WorldPos {
		WorldPos::new(x, y, z)
	}

	#[test]
	fn aabb_intersects_and_contains() {
		let a = Aabb::new(pos(0, 0, 0), pos(10, 10, 10));
		let b = Aabb::new(pos(5, 5, 5), pos(15, 15, 15));
		let c = Aabb::new(pos(20, 20, 20), pos(30, 30, 30));

		assert!(a.intersects(&b));
		assert!(!a.intersects(&c));
		assert!(a.contains(pos(0, 0, 0)));
		assert!(a.contains(pos(9, 9, 9)));
		assert!(!a.contains(pos(10, 0, 0)));
		assert!(!a.contains(pos(-1, 0, 0)));
	}

	#[test]
	fn lod_parent_child_roundtrip() {
		let fine = ChunkId::new(pos(0, 0, 0), LodLevel::FINEST);
		let parent = fine.parent().expect("finest has a parent");
		assert!(parent.aabb().contains(fine.origin));

		let children = parent.children().expect("non-finest has children");
		assert_eq!(children.len(), 64);
		assert!(children.iter().any(|c| *c == fine));
	}

	#[test]
	fn lod_coarsest_has_no_parent_finest_has_no_children() {
		let coarsest = ChunkId::new(pos(0, 0, 0), LodLevel::COARSEST);
		assert!(coarsest.parent().is_none());

		let finest = ChunkId::new(pos(0, 0, 0), LodLevel::FINEST);
		assert!(finest.children().is_none());
	}

	#[test]
	fn children_slot_order_matches_path_encoding() {
		let parent = ChunkId::new(pos(0, 0, 0), LodLevel::new(5));
		let child_size = LodLevel::new(6).chunk_size();
		let children = parent.children().unwrap();

		for slot in 0..64 {
			let x = ((slot >> 4) & 3) as i32;
			let y = ((slot >> 2) & 3) as i32;
			let z = (slot & 3) as i32;
			let expected = pos(x * child_size, y * child_size, z * child_size);
			assert_eq!(children[slot].origin, expected, "slot {slot}");
		}
	}
}
