use super::World;
use crate::Chunk;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::util::types::{ChunkId, CHUNK_SIZE, WorldPos};
use rayon::prelude::*;
use std::collections::HashMap;

// A single voxel write at a world-space position. The batch entry point for
// building a World from many voxels at once (used by the vox importer).
#[derive(Clone, Copy)]
pub struct WorldEdit {
	pub pos: WorldPos,
	pub material: Material,
}

impl World {
	// Build a World sized to fit all the given edits. The grid origin snaps
	// down to a chunk boundary, and the grid dimensions cover every chunk
	// touched by the edit bounds. Chunks are baked in parallel with rayon.
	pub fn from_edits(edits: Vec<WorldEdit>) -> World {
		if edits.is_empty() {
			return World::new([0, 0, 0]);
		}

		let (world_min, world_max_exclusive) = world_bounds(&edits);
		let origin = world_min.chunk_id().origin;
		let chunk_grid_dim = grid_dim_for_bounds(origin, world_max_exclusive);

		// Group edits by which chunk they land in. Sequential accumulation
		// dominates on realistic inputs (millions of tiny hashmap inserts).
		let mut per_chunk: HashMap<ChunkId, EditPacket> = HashMap::new();
		for edit in edits {
			let chunk_id = edit.pos.chunk_id();
			let local = edit.pos.chunk_pos(chunk_id.origin);
			per_chunk
				.entry(chunk_id)
				.or_default()
				.push(Path::from_coords(local, 4), edit.material);
		}

		// Bake each chunk in parallel.
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
