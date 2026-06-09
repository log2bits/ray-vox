use super::Model;
use crate::chunk::Chunk;
use crate::chunk::build::{Sample, Source, VoxelSample};
use crate::chunk::material::Material;
use crate::chunk::node::pack_slot;
use crate::chunk::sources::LocalEdit;
use crate::generate::Edit;
use crate::util::types::{Aabb, ChunkId, ChunkPos, LodLevel, WorldPos};
use std::cell::Cell;
use std::sync::Arc;

pub struct ModelStamp {
	pub model: Arc<Model>,
	pub position: WorldPos,
}

impl ModelStamp {
	pub fn new(model: Arc<Model>, position: WorldPos) -> Self {
		Self { model, position }
	}
}

impl Edit for ModelStamp {
	fn bounds(&self) -> Aabb {
		Aabb::new(
			self.position + self.model.bounds.min,
			self.position + self.model.bounds.max,
		)
	}

	fn make_local<'a>(&'a self, chunk_id: ChunkId) -> Option<Box<dyn LocalEdit + 'a>> {
		if !self.bounds().intersects(&chunk_id.aabb()) {
			return None;
		}
		Some(Box::new(ModelStampSource::new(&self.model, self.position, chunk_id)))
	}

	fn apply(&self, chunk_id: ChunkId, base: Chunk) -> Chunk {
		if !self.bounds().intersects(&chunk_id.aabb()) {
			return base;
		}
		let source = ModelStampSource::new(&self.model, self.position, chunk_id);
		base.edit(&source)
	}
}

pub struct ModelStampSource<'a> {
	model: &'a Model,
	voxel_offset: [i32; 3],
	lod: LodLevel,
	bounds_lo: [i32; 3],
	bounds_hi: [i32; 3],
	leaf_cache: Cell<Option<LeafCache>>,
	chunk_cache: Cell<Option<ChunkCache<'a>>>,
}

#[derive(Clone, Copy)]
struct LeafCache {
	leaf_origin: [i32; 3],
	occupancy: crate::util::types::Mask64,
	materials: [Material; 64],
}

#[derive(Clone, Copy)]
struct ChunkCache<'a> {
	chunk_id: ChunkId,
	chunk: Option<&'a Chunk>,
}

impl<'a> Clone for ModelStampSource<'a> {
	fn clone(&self) -> Self {
		Self {
			model: self.model,
			voxel_offset: self.voxel_offset,
			lod: self.lod,
			bounds_lo: self.bounds_lo,
			bounds_hi: self.bounds_hi,
			leaf_cache: Cell::new(None),
			chunk_cache: Cell::new(None),
		}
	}
}

impl<'a> ModelStampSource<'a> {
	pub fn new(model: &'a Model, stamp_position: WorldPos, target_chunk_id: ChunkId) -> Self {
		let voxel_size = target_chunk_id.lod.voxel_size();
		let delta = target_chunk_id.origin - stamp_position;
		let voxel_offset = [
			delta.x().div_euclid(voxel_size),
			delta.y().div_euclid(voxel_size),
			delta.z().div_euclid(voxel_size),
		];
		let bounds_lo_model = [
			model.bounds.min.x().div_euclid(voxel_size),
			model.bounds.min.y().div_euclid(voxel_size),
			model.bounds.min.z().div_euclid(voxel_size),
		];
		let bounds_hi_model = [
			(model.bounds.max.x() + voxel_size - 1).div_euclid(voxel_size),
			(model.bounds.max.y() + voxel_size - 1).div_euclid(voxel_size),
			(model.bounds.max.z() + voxel_size - 1).div_euclid(voxel_size),
		];
		let bounds_lo = [
			bounds_lo_model[0] - voxel_offset[0],
			bounds_lo_model[1] - voxel_offset[1],
			bounds_lo_model[2] - voxel_offset[2],
		];
		let bounds_hi = [
			bounds_hi_model[0] - voxel_offset[0],
			bounds_hi_model[1] - voxel_offset[1],
			bounds_hi_model[2] - voxel_offset[2],
		];
		Self {
			model,
			voxel_offset,
			lod: target_chunk_id.lod,
			bounds_lo,
			bounds_hi,
			leaf_cache: Cell::new(None),
			chunk_cache: Cell::new(None),
		}
	}

	#[inline]
	fn get_model_chunk(&self, chunk_id: ChunkId) -> Option<&'a Chunk> {
		if let Some(c) = self.chunk_cache.get() {
			if c.chunk_id == chunk_id {
				return c.chunk;
			}
		}
		let chunk = self.model.chunks.get(&chunk_id);
		self.chunk_cache.set(Some(ChunkCache { chunk_id, chunk }));
		chunk
	}

	#[inline]
	fn lookup_voxel(&self, chunk_local: [i32; 3]) -> Option<Material> {
		let model_voxel = [
			chunk_local[0] + self.voxel_offset[0],
			chunk_local[1] + self.voxel_offset[1],
			chunk_local[2] + self.voxel_offset[2],
		];

		// Cache hit: same 4×4×4 leaf as last call.
		let leaf_origin = [
			model_voxel[0] & !3,
			model_voxel[1] & !3,
			model_voxel[2] & !3,
		];
		if let Some(cache) = self.leaf_cache.get() {
			if cache.leaf_origin == leaf_origin {
				return self.read_cached(&cache, model_voxel);
			}
		}

		// Miss: walk the model tree, populate cache, return result.
		self.populate_cache_and_read(leaf_origin, model_voxel)
	}

	#[inline]
	fn read_cached(&self, cache: &LeafCache, model_voxel: [i32; 3]) -> Option<Material> {
		let slot = pack_slot(model_voxel);
		if cache.occupancy.contains(slot) {
			Some(cache.materials[slot as usize])
		} else {
			None
		}
	}

	fn populate_cache_and_read(&self, leaf_origin: [i32; 3], model_voxel: [i32; 3]) -> Option<Material> {
		let chunk_voxel_origin = [
			leaf_origin[0].div_euclid(256) * 256,
			leaf_origin[1].div_euclid(256) * 256,
			leaf_origin[2].div_euclid(256) * 256,
		];
		let voxel_size = self.lod.voxel_size();
		let chunk_world_origin = WorldPos::new(
			chunk_voxel_origin[0] * voxel_size,
			chunk_voxel_origin[1] * voxel_size,
			chunk_voxel_origin[2] * voxel_size,
		);
		let chunk_id = ChunkId::new(chunk_world_origin, self.lod);
		let chunk = self.get_model_chunk(chunk_id)?;
		let (occupancy, materials) = resolve_leaf(chunk, [
			(leaf_origin[0] - chunk_voxel_origin[0]) as u8,
			(leaf_origin[1] - chunk_voxel_origin[1]) as u8,
			(leaf_origin[2] - chunk_voxel_origin[2]) as u8,
		]);
		let cache = LeafCache { leaf_origin, occupancy, materials };
		self.leaf_cache.set(Some(cache));
		self.read_cached(&cache, model_voxel)
	}
}

/// Returns the slot indices `[sx, sy, sz]` if `[lo, lo+side)` fits in exactly one
/// `child_side`-sized child of the cell starting at `current_lo`, else `None`.
#[inline]
fn fits_in_child(lo: [i32; 3], side: i32, current_lo: [i32; 3], child_side: i32) -> Option<[i32; 3]> {
	let mut s = [0i32; 3];
	for axis in 0..3 {
		let a = (lo[axis] - current_lo[axis]) / child_side;
		let b = (lo[axis] + side - 1 - current_lo[axis]) / child_side;
		if a != b {
			return None;
		}
		s[axis] = a;
	}
	Some(s)
}

/// Classify a `[query_lo, query_lo + query_side)` region against a model chunk's tree.
/// Returns Passthrough/Fill if the region is uniformly empty/filled, else Subdivide.
fn classify_in_chunk(chunk: &Chunk, query_lo: [i32; 3], query_side: i32) -> Sample {
	use crate::chunk::Child;
	if chunk.is_empty() {
		return Sample::Passthrough;
	}
	if chunk.is_uniform() {
		return Sample::Fill(chunk.chunk_lod());
	}

	let mut current_lo = [0i32; 3];
	let mut current_side = 256i32;

	if chunk.interior_nodes.is_empty() {
		let leaf = &chunk.leaf_nodes[0];
		let child_side = 64;
		let s = match fits_in_child(query_lo, query_side, current_lo, child_side) {
			Some(s) => s,
			None => return Sample::Subdivide,
		};
		let slot = pack_slot(s);
		if !leaf.occupancy.contains(slot) {
			return Sample::Passthrough;
		}
		return Sample::Fill(chunk.materials.get(leaf.material_index(slot)));
	}

	let mut idx = chunk.root_idx();
	loop {
		let child_side = current_side / 4;
		let s = match fits_in_child(query_lo, query_side, current_lo, child_side) {
			Some(s) => s,
			None => return Sample::Subdivide,
		};
		let slot = pack_slot(s);
		match chunk.child(idx, slot) {
			Child::Empty => return Sample::Passthrough,
			Child::Filled(m) => return Sample::Fill(m),
			Child::Interior(c) => {
				if child_side == query_side {
					return Sample::Subdivide;
				}
				idx = c;
				current_lo = [
					current_lo[0] + s[0] * child_side,
					current_lo[1] + s[1] * child_side,
					current_lo[2] + s[2] * child_side,
				];
				current_side = child_side;
			}
			Child::Leaf(leaf_idx) => {
				if child_side == 4 {
					return Sample::Subdivide;
				}
				let leaf = &chunk.leaf_nodes[leaf_idx as usize];
				let cell_lo = [
					current_lo[0] + s[0] * child_side,
					current_lo[1] + s[1] * child_side,
					current_lo[2] + s[2] * child_side,
				];
				let slot_side = child_side / 4;
				let inner = match fits_in_child(query_lo, query_side, cell_lo, slot_side) {
					Some(s) => s,
					None => return Sample::Subdivide,
				};
				let inner_slot = pack_slot(inner);
				if !leaf.occupancy.contains(inner_slot) {
					return Sample::Passthrough;
				}
				return Sample::Fill(chunk.materials.get(leaf.material_index(inner_slot)));
			}
		}
	}
}

/// Resolve a 4×4×4 leaf-aligned region of a chunk into (occupancy, materials).
/// `leaf_origin` is in chunk-local voxel coords and must be a multiple of 4 per axis.
fn resolve_leaf(chunk: &Chunk, leaf_origin: [u8; 3]) -> (crate::util::types::Mask64, [Material; 64]) {
	use crate::chunk::Child;
	use crate::chunk::edit::Path;
	use crate::util::types::Mask64;

	let pos = ChunkPos::new(leaf_origin[0], leaf_origin[1], leaf_origin[2]);
	if chunk.interior_nodes.is_empty() && chunk.leaf_nodes.is_empty() {
		let m = chunk.chunk_lod();
		if m.is_air() {
			return (Mask64::EMPTY, [Material::air(); 64]);
		}
		return (Mask64::FULL, [m; 64]);
	}
	if chunk.interior_nodes.is_empty() {
		let leaf = &chunk.leaf_nodes[0];
		let path = Path::from_coords(pos, 4);
		let slot = path.slot_at(0);
		let m = if leaf.occupancy.contains(slot) {
			chunk.materials.get(leaf.material_index(slot))
		} else {
			Material::air()
		};
		if m.is_air() {
			return (Mask64::EMPTY, [Material::air(); 64]);
		}
		return (Mask64::FULL, [m; 64]);
	}

	let path = Path::from_coords(pos, 4);
	let mut idx = chunk.root_idx();
	for d in 0..3u8 {
		let slot = path.slot_at(d);
		match chunk.child(idx, slot) {
			Child::Empty => return (Mask64::EMPTY, [Material::air(); 64]),
			Child::Filled(m) => return (Mask64::FULL, [m; 64]),
			Child::Interior(c) => idx = c,
			Child::Leaf(leaf_idx) => {
				let leaf = &chunk.leaf_nodes[leaf_idx as usize];
				if d == 2 {
					// Voxel-level leaf: 64 slots = 64 voxels of our 4×4×4 cell.
					let mut mats = [Material::air(); 64];
					for s in leaf.occupancy.iter_slots() {
						mats[s as usize] = chunk.materials.get(leaf.material_index(s));
					}
					return (leaf.occupancy, mats);
				} else {
					// Demoted leaf above voxel level: one slot covers our whole cell.
					let lslot = path.slot_at(d + 1);
					let m = if leaf.occupancy.contains(lslot) {
						chunk.materials.get(leaf.material_index(lslot))
					} else {
						Material::air()
					};
					if m.is_air() {
						return (Mask64::EMPTY, [Material::air(); 64]);
					}
					return (Mask64::FULL, [m; 64]);
				}
			}
		}
	}
	(Mask64::EMPTY, [Material::air(); 64])
}

impl<'a> Source for ModelStampSource<'a> {
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], _depth: u8) -> Sample {
		if hi[0] <= self.bounds_lo[0] || lo[0] >= self.bounds_hi[0]
			|| hi[1] <= self.bounds_lo[1] || lo[1] >= self.bounds_hi[1]
			|| hi[2] <= self.bounds_lo[2] || lo[2] >= self.bounds_hi[2]
		{
			return Sample::Passthrough;
		}
		let side = hi[0] - lo[0];
		let model_lo = [
			lo[0] + self.voxel_offset[0],
			lo[1] + self.voxel_offset[1],
			lo[2] + self.voxel_offset[2],
		];
		let chunk_a = [
			model_lo[0].div_euclid(256),
			model_lo[1].div_euclid(256),
			model_lo[2].div_euclid(256),
		];
		let chunk_b = [
			(model_lo[0] + side - 1).div_euclid(256),
			(model_lo[1] + side - 1).div_euclid(256),
			(model_lo[2] + side - 1).div_euclid(256),
		];
		if chunk_a != chunk_b {
			return Sample::Subdivide;
		}
		let voxel_size = self.lod.voxel_size();
		let chunk_world_origin = WorldPos::new(
			chunk_a[0] * 256 * voxel_size,
			chunk_a[1] * 256 * voxel_size,
			chunk_a[2] * 256 * voxel_size,
		);
		let chunk_id = ChunkId::new(chunk_world_origin, self.lod);
		let chunk = match self.get_model_chunk(chunk_id) {
			Some(c) => c,
			None => return Sample::Passthrough,
		};
		let local_lo = [
			model_lo[0] - chunk_a[0] * 256,
			model_lo[1] - chunk_a[1] * 256,
			model_lo[2] - chunk_a[2] * 256,
		];
		classify_in_chunk(chunk, local_lo, side)
	}

	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		match self.lookup_voxel(v) {
			Some(m) => VoxelSample::Set(m),
			None => VoxelSample::Passthrough,
		}
	}
}

crate::impl_local_edit!(ModelStampSource<'a>, |s| [s.bounds_lo, s.bounds_hi]);
