use super::World;
use crate::Chunk;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::util::types::{ChunkId, CHUNK_SIZE, WorldPos};
use rayon::prelude::*;
use std::collections::HashMap;

// One voxel write at a world-space position. Batch entry point for the
// vox importer and any other bulk-voxel loader.
#[derive(Clone, Copy)]
pub struct WorldEdit {
	pub pos: WorldPos,
	pub material: Material,
}

impl World {
	// Build a World sized to fit the edit bounds, then bake chunks in parallel.
	pub fn from_edits(edits: Vec<WorldEdit>) -> World {
		if edits.is_empty() {
			return World::new([0, 0, 0]);
		}

		let (world_min, world_max_exclusive) = world_bounds(&edits);
		let origin = world_min.chunk_id().origin; // snap to a chunk boundary
		let chunk_grid_dim = grid_dim_for_bounds(origin, world_max_exclusive);

		// Group edits by chunk. Sequential inserts dominate here since even
		// millions of hashmap ops finish well before the per-chunk bake.
		let mut per_chunk: HashMap<ChunkId, EditPacket> = HashMap::new();
		for edit in edits {
			let chunk_id = edit.pos.chunk_id();
			let local = edit.pos.chunk_pos(chunk_id.origin);
			per_chunk
				.entry(chunk_id)
				.or_default()
				.push(Path::from_coords(local, 4), edit.material);
		}

		let baked: Vec<(ChunkId, Chunk)> = per_chunk
			.into_par_iter()
			.filter_map(|(chunk_id, mut packet)| {
				packet.sort();
				let chunk = Chunk::new().edit(&DiscreteSource::new(&packet.edits));
				if chunk.is_empty() { None } else { Some((chunk_id, chunk)) }
			})
			.collect();

		let mut world = World::with_origin(chunk_grid_dim, origin);
		for (chunk_id, chunk) in baked {
			let grid_pos = grid_pos_for(chunk_id.origin, origin);
			world.set_chunk(grid_pos, chunk);
		}
		world
	}
}

fn world_bounds(edits: &[WorldEdit]) -> (WorldPos, WorldPos) {
	let mut min = WorldPos::splat(i32::MAX);
	let mut max_exclusive = WorldPos::splat(i32::MIN);
	for edit in edits {
		min = WorldPos::from_fn(|i| min[i].min(edit.pos[i]));
		max_exclusive = WorldPos::from_fn(|i| max_exclusive[i].max(edit.pos[i] + 1));
	}
	(min, max_exclusive)
}

fn grid_dim_for_bounds(origin: WorldPos, world_max_exclusive: WorldPos) -> [u32; 3] {
	let mut dim = [0u32; 3];
	for axis in 0..3 {
		let span = world_max_exclusive[axis] - origin[axis];
		let cells = (span + CHUNK_SIZE - 1) / CHUNK_SIZE;
		dim[axis] = cells.max(0) as u32;
	}
	dim
}

fn grid_pos_for(chunk_origin: WorldPos, world_origin: WorldPos) -> [u32; 3] {
	[
		((chunk_origin.x() - world_origin.x()) / CHUNK_SIZE) as u32,
		((chunk_origin.y() - world_origin.y()) / CHUNK_SIZE) as u32,
		((chunk_origin.z() - world_origin.z()) / CHUNK_SIZE) as u32,
	]
}
