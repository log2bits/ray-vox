use super::Model;
use crate::Chunk;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::util::types::{Aabb, ChunkId, WorldPos};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct WorldEdit {
	pub pos: WorldPos,
	pub material: Material,
}

// Collects world-space voxel writes and bakes them into a Model.
// The builder is single-threaded: callers gather edits (typically via a
// rayon collect) and hand the finished Vec over to bake().
#[derive(Default)]
pub struct ModelBuilder {
	edits: Vec<WorldEdit>,
}

impl ModelBuilder {
	pub fn new() -> Self {
		Self { edits: Vec::new() }
	}

	pub fn from_edits(edits: Vec<WorldEdit>) -> Self {
		Self { edits }
	}

	pub fn push(&mut self, edit: WorldEdit) {
		self.edits.push(edit);
	}

	pub fn extend<I: IntoIterator<Item = WorldEdit>>(&mut self, iter: I) {
		self.edits.extend(iter);
	}

	pub fn len(&self) -> usize {
		self.edits.len()
	}

	pub fn is_empty(&self) -> bool {
		self.edits.is_empty()
	}

	pub fn bake(self) -> Model {
		let bounds = bounds_of(&self.edits);

		// Group edits by the chunk they land in. Sequential accumulation
		// dominates on realistic inputs (millions of tiny hashmap inserts),
		// then baking each chunk is done in parallel.
		let mut per_chunk: HashMap<ChunkId, EditPacket> = HashMap::new();
		for edit in self.edits {
			let chunk_id = edit.pos.chunk_id();
			let local = edit.pos.chunk_pos(chunk_id.origin);
			per_chunk
				.entry(chunk_id)
				.or_default()
				.push(Path::from_coords(local, 4), edit.material);
		}

		let chunks: HashMap<ChunkId, Chunk> = per_chunk
			.into_par_iter()
			.filter_map(|(chunk_id, mut packet)| {
				packet.sort();
				let chunk = Chunk::new().edit(&DiscreteSource::new(&packet.edits));
				if chunk.is_empty() { None } else { Some((chunk_id, chunk)) }
			})
			.collect();

		Model { chunks, bounds }
	}
}

impl FromIterator<WorldEdit> for ModelBuilder {
	fn from_iter<I: IntoIterator<Item = WorldEdit>>(iter: I) -> Self {
		Self { edits: iter.into_iter().collect() }
	}
}

fn bounds_of(edits: &[WorldEdit]) -> Aabb {
	if edits.is_empty() {
		return Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(0, 0, 0));
	}
	let mut min = WorldPos::splat(i32::MAX);
	let mut max = WorldPos::splat(i32::MIN);
	for edit in edits {
		min = WorldPos::from_fn(|i| min[i].min(edit.pos[i]));
		max = WorldPos::from_fn(|i| max[i].max(edit.pos[i] + 1));
	}
	Aabb::new(min, max)
}
