//! Sphere volume Edit. Pure geometry — the tree walk lives in the parent
//! module.

use crate::chunk::edit::EditPacket;
use crate::chunk::material::Material;
use crate::generate::volume::{walk, Coverage, Volume};
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
}

impl Edit for Sphere {
	fn bounds(&self) -> Aabb {
		let r = self.radius;
		Aabb::new(
			WorldPos::new(self.center.x() - r, self.center.y() - r, self.center.z() - r),
			WorldPos::new(self.center.x() + r, self.center.y() + r, self.center.z() + r),
		)
	}

	fn sample(&self, chunk: ChunkId) -> EditPacket {
		let voxel_size = chunk.lod.voxel_size();
		let center = [
			(self.center.x() - chunk.origin.x()) / voxel_size,
			(self.center.y() - chunk.origin.y()) / voxel_size,
			(self.center.z() - chunk.origin.z()) / voxel_size,
		];
		let radius = self.radius / voxel_size;
		if radius <= 0 {
			return EditPacket::from_sorted(Vec::new());
		}
		walk(&LocalSphere { center, radius_squared: radius * radius }, self.material)
	}
}

/// Sphere in chunk-local voxel coordinates. Built once per `sample` call.
struct LocalSphere {
	center: [i32; 3],
	radius_squared: i32,
}

impl Volume for LocalSphere {
	#[inline]
	fn classify(&self, lo: [i32; 3], hi: [i32; 3]) -> Coverage {
		let mut nearest_squared = 0;
		let mut farthest_squared = 0;
		for axis in 0..3 {
			let center = self.center[axis];
			let nearest = center.clamp(lo[axis], hi[axis]) - center;
			nearest_squared += nearest * nearest;
			let to_lo = lo[axis] - center;
			let to_hi = hi[axis] - center;
			let farthest = if to_lo * to_lo > to_hi * to_hi { to_lo } else { to_hi };
			farthest_squared += farthest * farthest;
		}
		if nearest_squared > self.radius_squared {
			Coverage::Outside
		} else if farthest_squared <= self.radius_squared {
			Coverage::Inside
		} else {
			Coverage::Straddle
		}
	}

	#[inline]
	fn contains_voxel(&self, voxel: [i32; 3]) -> bool {
		let dx = voxel[0] - self.center[0];
		let dy = voxel[1] - self.center[1];
		let dz = voxel[2] - self.center[2];
		dx * dx + dy * dy + dz * dz <= self.radius_squared
	}
}
