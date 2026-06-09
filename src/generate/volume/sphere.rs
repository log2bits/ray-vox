use crate::chunk::build::{Sample, Source, VoxelSample};
use crate::chunk::material::Material;
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

	pub fn local(&self, chunk: ChunkId) -> Option<LocalSphere> {
		let voxel_size = chunk.lod.voxel_size();
		let center = [
			(self.center.x() - chunk.origin.x()) / voxel_size,
			(self.center.y() - chunk.origin.y()) / voxel_size,
			(self.center.z() - chunk.origin.z()) / voxel_size,
		];
		let radius = self.radius / voxel_size;
		if radius <= 0 {
			return None;
		}
		Some(LocalSphere {
			center,
			radius_squared: radius * radius,
			material: self.material,
		})
	}
}

impl Edit for Sphere {
	fn bounds(&self) -> Aabb {
		let r = self.radius;
		Aabb::new(
			WorldPos::new(self.center.x() - r, self.center.y() - r, self.center.z() - r),
			WorldPos::new(self.center.x() + r, self.center.y() + r, self.center.z() + r),
		)
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
	fn box_coverage(&self, lo: [i32; 3], hi: [i32; 3]) -> SphereCoverage {
		let mut nearest_squared = 0;
		let mut farthest_squared = 0;
		for axis in 0..3 {
			let c = self.center[axis];
			let nearest = c.clamp(lo[axis], hi[axis]) - c;
			nearest_squared += nearest * nearest;
			let to_lo = lo[axis] - c;
			let to_hi = hi[axis] - c;
			let farthest = if to_lo * to_lo > to_hi * to_hi { to_lo } else { to_hi };
			farthest_squared += farthest * farthest;
		}
		if nearest_squared > self.radius_squared {
			SphereCoverage::Outside
		} else if farthest_squared <= self.radius_squared {
			SphereCoverage::Inside
		} else {
			SphereCoverage::Straddle
		}
	}

	#[inline]
	fn contains(&self, voxel: [i32; 3]) -> bool {
		let dx = voxel[0] - self.center[0];
		let dy = voxel[1] - self.center[1];
		let dz = voxel[2] - self.center[2];
		dx * dx + dy * dy + dz * dz <= self.radius_squared
	}
}

enum SphereCoverage { Outside, Inside, Straddle }

impl Source for LocalSphere {
	#[inline]
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], _depth: u8) -> Sample {
		match self.box_coverage(lo, hi) {
			SphereCoverage::Outside => Sample::Passthrough,
			SphereCoverage::Inside => Sample::Fill(self.material),
			SphereCoverage::Straddle => Sample::Subdivide,
		}
	}

	#[inline]
	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		if self.contains(v) {
			VoxelSample::Set(self.material)
		} else {
			VoxelSample::Passthrough
		}
	}
}
