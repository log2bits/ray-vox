use crate::chunk::build::{Sample, Source, VoxelSample};
use crate::chunk::material::Material;
use crate::chunk::sources::LocalEdit;
use crate::chunk::Chunk;
use crate::generate::Edit;
use crate::util::types::{Aabb, ChunkId, WorldPos};

pub struct Sphere {
	pub center: WorldPos,
	pub radius: i32,
	pub material: Material,
}

impl Sphere {
	pub fn new(center: WorldPos, radius: i32, material: Material) -> Self {
		Self { center, radius, material }
	}

	// Convert this sphere into chunk-local coordinates. Returns None when the
	// sphere has non-positive radius and therefore covers no voxels.
	pub fn local(&self, chunk: ChunkId) -> Option<LocalSphere> {
		if self.radius <= 0 {
			return None;
		}
		let center = [
			self.center.x() - chunk.origin.x(),
			self.center.y() - chunk.origin.y(),
			self.center.z() - chunk.origin.z(),
		];
		Some(LocalSphere {
			center,
			radius_squared: self.radius * self.radius,
			material: self.material,
		})
	}
}

impl Edit for Sphere {
	fn bounds(&self) -> Aabb {
		let r = WorldPos::splat(self.radius);
		Aabb::new(self.center - r, self.center + r)
	}

	fn make_local<'a>(&'a self, chunk_id: ChunkId) -> Option<Box<dyn LocalEdit + 'a>> {
		Some(Box::new(self.local(chunk_id)?))
	}

	fn apply(&self, chunk_id: ChunkId, base: Chunk) -> Chunk {
		match self.local(chunk_id) {
			None => base,
			Some(local) => base.edit(&local),
		}
	}
}

#[derive(Clone)]
pub struct LocalSphere {
	pub center: [i32; 3],
	pub radius_squared: i32,
	pub material: Material,
}

impl LocalSphere {
	#[inline]
	fn contains(&self, voxel: [i32; 3]) -> bool {
		let dx = voxel[0] - self.center[0];
		let dy = voxel[1] - self.center[1];
		let dz = voxel[2] - self.center[2];
		dx * dx + dy * dy + dz * dz <= self.radius_squared
	}
}

impl Source for LocalSphere {
	#[inline]
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], _depth: u8) -> Sample {
		let mut nearest_sq = 0;
		let mut farthest_sq = 0;
		for axis in 0..3 {
			let c = self.center[axis];
			let nearest = c.clamp(lo[axis], hi[axis]) - c;
			nearest_sq += nearest * nearest;
			let to_lo = lo[axis] - c;
			let to_hi = hi[axis] - c;
			let farthest = if to_lo * to_lo > to_hi * to_hi { to_lo } else { to_hi };
			farthest_sq += farthest * farthest;
		}
		if nearest_sq > self.radius_squared {
			Sample::Passthrough
		} else if farthest_sq <= self.radius_squared {
			Sample::Fill(self.material)
		} else {
			Sample::Subdivide
		}
	}

	#[inline]
	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		if self.contains(v) {
			VoxelSample::Fill(self.material)
		} else {
			VoxelSample::Passthrough
		}
	}
}

crate::impl_local_edit!(LocalSphere, |s| {
	let r = (s.radius_squared as f32).sqrt() as i32 + 1;
	[
		[s.center[0] - r, s.center[1] - r, s.center[2] - r],
		[s.center[0] + r + 1, s.center[1] + r + 1, s.center[2] + r + 1],
	]
});
