pub mod import;
pub mod terrain;
pub mod volume;

use crate::chunk::edit::{ChunkEdits, Path};
use crate::chunk::material::Material;
use crate::util::types::{ClipmapChunkId, WorldPos};
use crate::world::clipmap::{Clipmap, chunk_size_at_depth};
use std::array::from_fn;

/// Lazy edits span the world but are evaluated one chunk at a time on demand.
/// Terrain, volumes, and other procedural content implement this.
pub trait LazyEdit: Send + Sync {
	fn generate(&self, handle: ClipmapChunkId) -> ChunkEdits;
}

/// Eager edits are evaluated once upfront and produce edits for every chunk they cover.
/// The world applies these to persistent chunks immediately and discards the generator.
pub trait EagerEdit: Send + Sync {
	fn generate(&self) -> Vec<ChunkEdits>;
}

pub enum CellState {
	Empty,
	Uniform(Material),
	Subdivide,
}

pub fn chunk_world_origin(handle: ClipmapChunkId, clipmap: &Clipmap) -> WorldPos {
	let level_origin = clipmap.level_origin(handle.depth());
	let cs = chunk_size_at_depth(handle.depth());
	WorldPos {
		pos: from_fn(|i| level_origin.pos[i] + handle.xyz()[i] as i32 * cs),
	}
}

pub fn voxel_size(depth: u8) -> i32 {
	chunk_size_at_depth(depth) / 256
}

/// Analytical volume defined by AABB overlap classification.
///
/// Implement this for any SDF-based shape. A blanket [`LazyEdit`] impl is provided.
///
/// `material` returning `None` means the volume's color varies within the region
/// and subdivision must continue. At depth 4 (single voxel), `material` must
/// return `Some`.
pub trait Volume: Send + Sync {
	fn overlap(&self, world_min: WorldPos, world_max: WorldPos) -> Overlap;
	fn material(&self, world_min: WorldPos, world_max: WorldPos) -> Option<Material>;
}

impl<T: Volume> LazyEdit for T {
	fn generate(&self, handle: ClipmapChunkId, clipmap: &Clipmap) -> ChunkEdits {
		let world_origin = chunk_world_origin(handle, clipmap);
		let vs = voxel_size(handle.depth());
		let mut edits = ChunkEdits::new(world_origin);
		stamp_volume(self, &mut edits, world_origin, vs);
		edits
	}
}

pub(crate) fn stamp_volume(
	volume: &impl Volume,
	edits: &mut ChunkEdits,
	world_origin: WorldPos,
	voxel_size: i32,
) {
	stamp_cell(volume, edits, world_origin, voxel_size, [0, 0, 0], 0);
}

fn stamp_cell(
	volume: &impl Volume,
	edits: &mut ChunkEdits,
	world_origin: WorldPos,
	voxel_size: i32,
	local: [i32; 3],
	depth: u8,
) {
	let size = 256i32 >> (depth * 2);
	let world_min = WorldPos {
		pos: from_fn(|i| world_origin.pos[i] + local[i] * voxel_size),
	};
	let world_max = WorldPos {
		pos: from_fn(|i| world_min.pos[i] + size * voxel_size),
	};

	match volume.overlap(world_min, world_max) {
		Overlap::Empty => {}
		Overlap::Full => match volume.material(world_min, world_max) {
			Some(mat) => {
				let path = if depth == 0 {
					Path::from(0u32)
				} else {
					Path::from_coords([local[0] as u8, local[1] as u8, local[2] as u8], depth)
				};
				edits.push(path, mat);
			}
			None => subdivide(volume, edits, world_origin, voxel_size, local, depth),
		},
		Overlap::Partial => {
			if depth == 4 {
				if let Some(mat) = volume.material(world_min, world_max) {
					edits.push(
						Path::from_coords([local[0] as u8, local[1] as u8, local[2] as u8], 4),
						mat,
					);
				}
			} else {
				subdivide(volume, edits, world_origin, voxel_size, local, depth);
			}
		}
	}
}

#[inline]
fn subdivide(
	volume: &impl Volume,
	edits: &mut ChunkEdits,
	world_origin: WorldPos,
	voxel_size: i32,
	local: [i32; 3],
	depth: u8,
) {
	let child_size = (256i32 >> (depth * 2)) / 4;
	for dz in 0..4i32 {
		for dy in 0..4i32 {
			for dx in 0..4i32 {
				stamp_cell(
					volume,
					edits,
					world_origin,
					voxel_size,
					[
						local[0] + dx * child_size,
						local[1] + dy * child_size,
						local[2] + dz * child_size,
					],
					depth + 1,
				);
			}
		}
	}
}
