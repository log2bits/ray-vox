use std::array::from_fn;

use crate::Chunk;
use crate::chunk::edit::Edits;
use crate::chunk::material::Material;
use crate::volumes::{Overlap, Shape, Volume, chunk_world_origin, stamp, voxel_size};
use crate::world::clipmap::{ChunkHandle, Clipmap, chunk_size_at_depth};

pub struct Rect {
	pub min: [i32; 3],
	pub max: [i32; 3],
	pub material: Material,
}

impl Shape for Rect {
	fn overlap(&self, world_min: [i32; 3], world_max: [i32; 3]) -> Overlap {
		for i in 0..3 {
			if self.max[i] <= world_min[i] || self.min[i] >= world_max[i] {
				return Overlap::Empty;
			}
		}
		for i in 0..3 {
			if self.min[i] > world_min[i] || self.max[i] < world_max[i] {
				return Overlap::Partial;
			}
		}
		Overlap::Full
	}

	fn material(&self) -> Material {
		self.material
	}
}

impl Volume for Rect {
	fn overlaps(&self, handle: ChunkHandle, clipmap: &Clipmap) -> bool {
		let world_origin = chunk_world_origin(handle, clipmap);
		let cs = chunk_size_at_depth(handle.depth());
		let world_max: [i32; 3] = from_fn(|i| world_origin[i] + cs);
		!matches!(self.overlap(world_origin, world_max), Overlap::Empty)
	}

	fn apply(&self, chunk: Chunk, handle: ChunkHandle, clipmap: &Clipmap) -> Chunk {
		let world_origin = chunk_world_origin(handle, clipmap);
		let vs = voxel_size(handle.depth());
		let mut edits = Edits::new();
		stamp(self, &mut edits, world_origin, vs);
		chunk.apply_edits(edits)
	}
}
