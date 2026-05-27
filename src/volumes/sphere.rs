use std::array::from_fn;

use crate::Chunk;
use crate::chunk::edit::Edits;
use crate::chunk::material::Material;
use crate::volumes::{Overlap, Shape, Volume, chunk_world_origin, stamp, voxel_size};
use crate::world::clipmap::{ChunkHandle, Clipmap, chunk_size_at_depth};

pub struct Sphere {
	pub center: [f32; 3],
	pub radius: f32,
	pub material: Material,
}

impl Shape for Sphere {
	fn overlap(&self, world_min: [i32; 3], world_max: [i32; 3]) -> Overlap {
		let r2 = self.radius * self.radius;
		let [cx, cy, cz] = self.center;

		let nx = cx.clamp(world_min[0] as f32, world_max[0] as f32);
		let ny = cy.clamp(world_min[1] as f32, world_max[1] as f32);
		let nz = cz.clamp(world_min[2] as f32, world_max[2] as f32);
		let dx = cx - nx;
		let dy = cy - ny;
		let dz = cz - nz;

		if dx * dx + dy * dy + dz * dz > r2 {
			return Overlap::Empty;
		}

		let fx = (cx - world_min[0] as f32).abs().max((cx - world_max[0] as f32).abs());
		let fy = (cy - world_min[1] as f32).abs().max((cy - world_max[1] as f32).abs());
		let fz = (cz - world_min[2] as f32).abs().max((cz - world_max[2] as f32).abs());

		if fx * fx + fy * fy + fz * fz <= r2 { Overlap::Full } else { Overlap::Partial }
	}

	fn material(&self) -> Material {
		self.material
	}
}

impl Sphere {
	pub fn build_edits(&self, world_origin: [i32; 3], voxel_size: i32) -> Edits {
		let mut edits = Edits::new();
		stamp(self, &mut edits, world_origin, voxel_size);
		edits
	}
}

impl Volume for Sphere {
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
