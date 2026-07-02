use bytemuck::{Pod, Zeroable};
use std::array::from_fn;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, Mul, Not, Sub};

// Voxels along one axis in a chunk. Fixed at 256 so chunk-local coords fit in u8.
pub const CHUNK_SIZE: i32 = 256;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Mask64([u32; 2]);

impl Mask64 {
	pub const EMPTY: Self = Mask64([0, 0]);
	pub const FULL: Self = Mask64([u32::MAX, u32::MAX]);

	#[inline]
	pub const fn new(v: u64) -> Self {
		Mask64([v as u32, (v >> 32) as u32])
	}

	#[inline]
	pub fn bit(slot: u8) -> Self {
		Self::new(1u64 << slot)
	}

	#[inline]
	pub fn raw(self) -> u64 {
		(self.0[0] as u64) | ((self.0[1] as u64) << 32)
	}

	#[inline]
	pub fn contains(self, slot: u8) -> bool {
		(self.raw() >> slot) & 1 != 0
	}

	#[inline]
	pub fn count(self) -> u32 {
		self.raw().count_ones()
	}

	#[inline]
	pub fn is_empty(self) -> bool {
		self.0[0] == 0 && self.0[1] == 0
	}

	#[inline]
	pub fn popcount_below(self, slot: u8) -> u32 {
		(self.raw() & ((1u64 << slot) - 1)).count_ones()
	}

	#[inline]
	pub fn iter_slots(self) -> Mask64Iter {
		Mask64Iter(self.raw())
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
		Mask64::new(v)
	}
}

impl From<Mask64> for u64 {
	#[inline]
	fn from(m: Mask64) -> Self {
		m.raw()
	}
}

impl BitAnd for Mask64 {
	type Output = Self;
	#[inline]
	fn bitand(self, rhs: Self) -> Self {
		Mask64([self.0[0] & rhs.0[0], self.0[1] & rhs.0[1]])
	}
}

impl BitOr for Mask64 {
	type Output = Self;
	#[inline]
	fn bitor(self, rhs: Self) -> Self {
		Mask64([self.0[0] | rhs.0[0], self.0[1] | rhs.0[1]])
	}
}

impl BitXor for Mask64 {
	type Output = Self;
	#[inline]
	fn bitxor(self, rhs: Self) -> Self {
		Mask64([self.0[0] ^ rhs.0[0], self.0[1] ^ rhs.0[1]])
	}
}

impl Not for Mask64 {
	type Output = Self;
	#[inline]
	fn not(self) -> Self {
		Mask64([!self.0[0], !self.0[1]])
	}
}

impl BitAndAssign for Mask64 {
	#[inline]
	fn bitand_assign(&mut self, rhs: Self) {
		self.0[0] &= rhs.0[0];
		self.0[1] &= rhs.0[1];
	}
}

impl BitOrAssign for Mask64 {
	#[inline]
	fn bitor_assign(&mut self, rhs: Self) {
		self.0[0] |= rhs.0[0];
		self.0[1] |= rhs.0[1];
	}
}

impl BitXorAssign for Mask64 {
	#[inline]
	fn bitxor_assign(&mut self, rhs: Self) {
		self.0[0] ^= rhs.0[0];
		self.0[1] ^= rhs.0[1];
	}
}

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

	// The chunk that contains this world position.
	pub fn chunk_id(self) -> ChunkId {
		ChunkId {
			origin: self.map(|c| align_down(c, CHUNK_SIZE)),
		}
	}

	// This position expressed as a chunk-local (u8, u8, u8).
	pub fn chunk_pos(self, chunk_origin: WorldPos) -> ChunkPos {
		ChunkPos::from_fn(|i| (self[i] - chunk_origin[i]) as u8)
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

// One chunk, identified by the world-space corner of its bounding box. The
// chunk covers [origin, origin + CHUNK_SIZE) along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkId {
	pub origin: WorldPos,
}

impl ChunkId {
	pub fn new(origin: WorldPos) -> Self {
		ChunkId { origin }
	}

	pub fn max_corner(self) -> WorldPos {
		self.origin + WorldPos::splat(CHUNK_SIZE)
	}

	pub fn aabb(self) -> Aabb {
		Aabb { min: self.origin, max: self.max_corner() }
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Aabb {
	pub min: WorldPos,
	pub max: WorldPos,
}

impl Aabb {
	pub fn new(min: WorldPos, max: WorldPos) -> Self {
		Self { min, max }
	}

	pub fn all() -> Self {
		Self::new(WorldPos::splat(i32::MIN), WorldPos::splat(i32::MAX))
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
	fn chunk_id_snaps_to_multiples_of_chunk_size() {
		let p = WorldPos::new(300, -10, 512);
		let id = p.chunk_id();
		assert_eq!(id.origin, WorldPos::new(256, -256, 512));
	}

	#[test]
	fn chunk_pos_of_world_pos_inside_chunk() {
		let p = WorldPos::new(300, -10, 512);
		let chunk_origin = p.chunk_id().origin;
		let local = p.chunk_pos(chunk_origin);
		assert_eq!(<[u8; 3]>::from(local), [44, 246, 0]);
	}
}
