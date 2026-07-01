pub mod builder;
pub mod rvox;

pub use builder::{ModelBuilder, WorldEdit};

#[cfg(test)]
mod tests;

use crate::Chunk;
use crate::util::types::{Aabb, ChunkId, LodLevel, WorldPos};
use std::collections::HashMap;

pub struct Model {
	pub chunks: HashMap<ChunkId, Chunk>,
	pub bounds: Aabb,
}

impl Model {
	pub fn empty(bounds: Aabb) -> Self {
		Self { chunks: HashMap::new(), bounds }
	}

	pub fn builder() -> ModelBuilder {
		ModelBuilder::new()
	}

	pub fn chunk_count(&self) -> usize {
		self.chunks.len()
	}

	pub fn chunk_at(&self, lod: LodLevel, pos: WorldPos) -> Option<&Chunk> {
		let id = pos.chunk_id(lod);
		self.chunks.get(&id)
	}
}
