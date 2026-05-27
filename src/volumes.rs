pub mod rect;
pub mod sphere;
pub mod terrain;

pub use rect::Rect;
pub use sphere::Sphere;

use std::array::from_fn;

use crate::Chunk;
use crate::chunk::edit::{Edits, Path};
use crate::chunk::material::Material;
use crate::world::clipmap::{ChunkHandle, Clipmap, chunk_size_at_depth};

pub trait Volume: Send + Sync {
	fn overlaps(&self, handle: ChunkHandle, clipmap: &Clipmap) -> bool;
	fn apply(&self, chunk: Chunk, handle: ChunkHandle, clipmap: &Clipmap) -> Chunk;
}

pub(crate) enum Overlap {
	Empty,
	Partial,
	Full,
}

pub(crate) trait Shape {
	fn overlap(&self, world_min: [i32; 3], world_max: [i32; 3]) -> Overlap;
	fn material(&self) -> Material;
}

pub(crate) fn chunk_world_origin(handle: ChunkHandle, clipmap: &Clipmap) -> [i32; 3] {
	let level_origin = clipmap.level_origin(handle.depth());
	let cs = chunk_size_at_depth(handle.depth());
	from_fn(|i| level_origin[i] + handle.xyz()[i] as i32 * cs)
}

pub(crate) fn voxel_size(depth: u8) -> i32 {
	chunk_size_at_depth(depth) / 256
}

pub(crate) fn stamp(shape: &impl Shape, edits: &mut Edits, world_origin: [i32; 3], voxel_size: i32) {
	stamp_cell(shape, edits, world_origin, voxel_size, [0, 0, 0], 0);
}

fn stamp_cell(
	shape: &impl Shape,
	edits: &mut Edits,
	world_origin: [i32; 3],
	voxel_size: i32,
	local: [i32; 3],
	depth: u8,
) {
	let size = 256i32 >> (depth * 2);
	let world_min: [i32; 3] = from_fn(|i| world_origin[i] + local[i] * voxel_size);
	let world_max: [i32; 3] = from_fn(|i| world_min[i] + size * voxel_size);

	match shape.overlap(world_min, world_max) {
		Overlap::Empty => {}
		Overlap::Full => {
			let path = if depth == 0 {
				Path::from(0u32)
			} else {
				Path::from_coords([local[0] as u8, local[1] as u8, local[2] as u8], depth)
			};
			edits.push(path, shape.material());
		}
		Overlap::Partial => {
			if depth == 4 {
				edits.push(
					Path::from_coords([local[0] as u8, local[1] as u8, local[2] as u8], 4),
					shape.material(),
				);
			} else {
				let child_size = size / 4;
				for dz in 0..4i32 {
					for dy in 0..4i32 {
						for dx in 0..4i32 {
							stamp_cell(
								shape,
								edits,
								world_origin,
								voxel_size,
								[local[0] + dx * child_size, local[1] + dy * child_size, local[2] + dz * child_size],
								depth + 1,
							);
						}
					}
				}
			}
		}
	}
}
