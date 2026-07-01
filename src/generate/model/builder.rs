use super::Model;
use crate::Chunk;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::util::types::{Aabb, ChunkId, LodLevel, WorldPos};
use ahash::AHasher;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};

#[derive(Clone, Copy)]
pub struct WorldEdit {
	pub pos: WorldPos,
	pub material: Material,
}

pub struct ModelBuilder {
	// Sharded packet map. Each shard is a small HashMap behind its own Mutex,
	// so concurrent `add` calls only contend when they hash to the same shard.
	shards: Box<[Mutex<HashMap<ChunkId, EditPacket>>]>,
	bounds: AtomicBounds,
}

impl ModelBuilder {
	pub fn new() -> Self {
		let shards = (rayon::current_num_threads().max(1) * 4).next_power_of_two();
		Self {
			shards: (0..shards).map(|_| Mutex::new(HashMap::new())).collect(),
			bounds: AtomicBounds::new(),
		}
	}

	pub fn add(&self, edit: WorldEdit) {
		self.bounds.observe(edit.pos);
		let chunk_id = edit.pos.chunk_id(LodLevel::FINEST);
		let local = edit.pos.chunk_pos(chunk_id.origin, LodLevel::FINEST);
		let shard_idx = shard_of(chunk_id, self.shards.len());
		let mut shard = self.shards[shard_idx].lock().unwrap();
		shard.entry(chunk_id)
			.or_default()
			.push(Path::from_coords(local, 4), edit.material);
	}

	pub fn bake(self) -> Model {
		let bounds = self.bounds.finalize();
		let by_chunk: HashMap<ChunkId, EditPacket> = self.shards
			.into_vec()
			.into_iter()
			.flat_map(|s| s.into_inner().unwrap().into_iter())
			.collect();

		let chunks: HashMap<ChunkId, Chunk> = by_chunk
			.into_par_iter()
			.filter_map(|(id, mut packet)| {
				packet.sort();
				let chunk = Chunk::new().edit(&DiscreteSource::new(&packet.edits));
				if chunk.is_empty() { None } else { Some((id, chunk)) }
			})
			.collect();

		Model { chunks, bounds }
	}
}

impl Default for ModelBuilder {
	fn default() -> Self { Self::new() }
}

struct AtomicBounds {
	min: [AtomicI32; 3],
	max: [AtomicI32; 3],
}

impl AtomicBounds {
	fn new() -> Self {
		Self {
			min: [AtomicI32::new(i32::MAX), AtomicI32::new(i32::MAX), AtomicI32::new(i32::MAX)],
			max: [AtomicI32::new(i32::MIN), AtomicI32::new(i32::MIN), AtomicI32::new(i32::MIN)],
		}
	}

	fn observe(&self, p: WorldPos) {
		atomic_min(&self.min[0], p.x());
		atomic_min(&self.min[1], p.y());
		atomic_min(&self.min[2], p.z());
		atomic_max(&self.max[0], p.x() + 1);
		atomic_max(&self.max[1], p.y() + 1);
		atomic_max(&self.max[2], p.z() + 1);
	}

	fn finalize(&self) -> Aabb {
		let lo = WorldPos::new(
			self.min[0].load(Ordering::Relaxed),
			self.min[1].load(Ordering::Relaxed),
			self.min[2].load(Ordering::Relaxed),
		);
		let hi = WorldPos::new(
			self.max[0].load(Ordering::Relaxed),
			self.max[1].load(Ordering::Relaxed),
			self.max[2].load(Ordering::Relaxed),
		);
		Aabb::new(lo, hi)
	}
}

#[inline]
fn atomic_min(a: &AtomicI32, v: i32) {
	let mut cur = a.load(Ordering::Relaxed);
	while v < cur {
		match a.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
			Ok(_) => return,
			Err(actual) => cur = actual,
		}
	}
}

#[inline]
fn atomic_max(a: &AtomicI32, v: i32) {
	let mut cur = a.load(Ordering::Relaxed);
	while v > cur {
		match a.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
			Ok(_) => return,
			Err(actual) => cur = actual,
		}
	}
}

#[inline]
fn shard_of(chunk_id: ChunkId, num_shards: usize) -> usize {
	let mut h = AHasher::default();
	chunk_id.hash(&mut h);
	(h.finish() as usize) & (num_shards - 1)
}
