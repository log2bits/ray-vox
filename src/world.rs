mod generate;
mod lod;
mod pool;

pub use lod::PointOfInterest;
pub use pool::ChunkPool;

use crate::{
	chunk::{Chunk, VoxelEdit},
	shape::{ContextShape, Shape},
	tree::{Aabb, Ray, RayHit, Tree},
	types::Voxel,
};
use std::collections::HashMap;
use std::time::Duration;

pub const WORLD_DEPTH: usize = 28;

pub struct World {
	pub world_tree: Tree<WORLD_DEPTH>,
	pub pool: ChunkPool,
	pub shape_edits: Vec<ShapeEdit>,
	// Persistent chunks keyed on LOD-0 chunk coordinates.
	// Active = chunk lives in the pool at full resolution.
	// Resident = chunk is in CPU memory, out of LOD-0 range.
	pub persistent_chunks: HashMap<[i64; 3], PersistentChunk>,
}

// A chunk that has received player voxel edits and is stored permanently.
pub enum PersistentChunk {
	Resident(Chunk), // held in CPU memory; not in the pool
	Active(u32),     // handle into pool; chunk lives there at full resolution
}

/// A lazy edit stored in the world's shape edit list. Evaluated at `generate_chunk` time
/// to produce an `EditPacket`, which is then applied to the chunk's tree.
/// Ephemeral chunks are generated entirely from this list; persistent chunks bake it once.
pub enum ShapeEdit {
	/// Write-only: pure geometry, no tree reads. Parallelizable across chunks.
	Write {
		aabb: Aabb,
		min_lod: u8,
		shape: Box<dyn Shape>,
	},
	/// Read+write: may query the tree during coverage evaluation.
	/// Applied sequentially after all Write edits in the same stage are flushed.
	/// Not yet implemented.
	ReadWrite {
		aabb: Aabb,
		min_lod: u8,
		shape: Box<dyn ContextShape>,
	},
}

impl ShapeEdit {
	pub fn write(aabb: Aabb, min_lod: u8, shape: Box<dyn Shape>) -> Self {
		ShapeEdit::Write { aabb, min_lod, shape }
	}

	pub fn aabb(&self) -> &Aabb {
		match self {
			ShapeEdit::Write { aabb, .. } | ShapeEdit::ReadWrite { aabb, .. } => aabb,
		}
	}

	pub fn min_lod(&self) -> u8 {
		match self {
			ShapeEdit::Write { min_lod, .. } | ShapeEdit::ReadWrite { min_lod, .. } => *min_lod,
		}
	}
}

pub struct WorldHit {
	pub chunk_pos: [i64; 3],
	pub local_pos: [u8; 3],
	pub normal: [i32; 3],
	pub voxel: Voxel,
}

impl World {
	pub fn new() -> Self {
		Self {
			world_tree: Tree::new(1),
			pool: ChunkPool::new(),
			shape_edits: Vec::new(),
			persistent_chunks: HashMap::new(),
		}
	}
	pub fn add_shape_edit(&mut self, edit: ShapeEdit) {
		self.shape_edits.push(edit);
	}
	// Queue a player voxel edit. Creates a persistent chunk for this position if
	// one doesn't exist yet (baking current shape edit state first).
	pub fn queue_voxel_edit(&mut self, chunk_pos: [i64; 3], edit: VoxelEdit) {
		todo!()
	}
	// Flush pending chunk edits within the given time budget, processing
	// closest-to-camera chunks first. Call once per frame.
	pub fn flush_edits(&mut self, budget: Duration, camera_chunk: [i64; 3]) {
		todo!()
	}
	pub fn tick_lod(&mut self, interests: &[PointOfInterest]) {
		todo!()
	}
	// Trace a ray through the world tree then the hit chunk's tree.
	pub fn trace_ray(&self, ray: &Ray) -> Option<WorldHit> {
		todo!()
	}
}
