pub mod builder;
pub mod coarsen;
pub mod rvox;
pub mod stamp;

pub use builder::{ModelBuilder, WorldEdit};

#[cfg(test)]
mod tests;

use crate::Chunk;
use crate::util::types::{Aabb, ChunkId, LodLevel, WorldPos};
use coarsen::coarsen;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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

	pub fn chunks_at_lod(&self, lod: LodLevel) -> impl Iterator<Item = (&ChunkId, &Chunk)> {
		self.chunks.iter().filter(move |(id, _)| id.lod == lod)
	}

	pub fn chunk_count(&self) -> usize {
		self.chunks.len()
	}

	pub fn chunk_at(&self, lod: LodLevel, pos: WorldPos) -> Option<&Chunk> {
		let id = pos.chunk_id(lod);
		self.chunks.get(&id)
	}

	pub fn build_mip_pyramid(&mut self) {
		let mut current_lod = match self.chunks.keys().map(|id| id.lod).max() {
			Some(lod) => lod,
			None => return,
		};

		while let Some(parent_lod) = current_lod.coarser() {
			let mut parent_ids: HashSet<ChunkId> = HashSet::new();
			for (id, _) in self.chunks.iter().filter(|(id, _)| id.lod == current_lod) {
				if let Some(parent) = id.parent() {
					parent_ids.insert(parent);
				}
			}

			let new_parents: Vec<(ChunkId, Chunk)> = parent_ids
				.into_par_iter()
				.filter_map(|parent_id| {
					let children_ids = parent_id.children()?;
					let children_refs: [Option<&Chunk>; 64] =
						std::array::from_fn(|i| self.chunks.get(&children_ids[i]));
					let parent_chunk = coarsen(&children_refs);
					if parent_chunk.is_empty() { None } else { Some((parent_id, parent_chunk)) }
				})
				.collect();
			for (parent_id, parent_chunk) in new_parents {
				self.chunks.insert(parent_id, parent_chunk);
			}

			current_lod = parent_lod;
		}
	}
}
