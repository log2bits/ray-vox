use super::{Coverage, Shape};
use crate::{tree::Aabb, types::Voxel};

pub struct CheckeredSphere {
    pub center:   [i64; 3],
    pub radius:   i64,
    pub material_a: Voxel,
    pub material_b: Voxel,
}

impl Shape for CheckeredSphere {
    fn aabb(&self) -> Aabb {
        let radius = self.radius.max(0);
        let extent = radius.saturating_add(1);
        Aabb {
            min: self.center.map(|c| c.saturating_sub(radius)),
            max: self.center.map(|c| c.saturating_add(extent)),
        }
    }

    fn coverage(&self, node_aabb: Aabb, lod: u8) -> Coverage {
        let radius    = self.radius.max(0) as i128;
        let radius_sq = radius * radius;
        let sample_max = node_aabb.max.map(|v| v - 1);

        let min_dist_sq =
            axis_distance_sq(self.center, node_aabb.min, sample_max, nearest_distance);
        if min_dist_sq > radius_sq {
            return Coverage::Empty;
        }

        if lod == 0 {
            // Single voxel — pick colour from 3D checkerboard parity.
            let parity = (node_aabb.min[0] + node_aabb.min[1] + node_aabb.min[2]) & 1;
            let mat = if parity == 0 { self.material_a } else { self.material_b };
            return Coverage::Full(mat);
        }

        Coverage::Partial
    }
}

pub struct Sphere {
	pub center: [i64; 3],
	pub radius: i64,
	pub material: Voxel,
}

impl Shape for Sphere {
	fn aabb(&self) -> Aabb {
		let radius = self.radius.max(0);
		let extent = radius.saturating_add(1);

		Aabb {
			min: self.center.map(|c| c.saturating_sub(radius)),
			max: self.center.map(|c| c.saturating_add(extent)),
		}
	}

	fn coverage(&self, node_aabb: Aabb, lod: u8) -> Coverage {
		let radius = self.radius.max(0) as i128;
		let radius_sq = radius * radius;
		let sample_max = node_aabb.max.map(|v| v - 1);

		let min_dist_sq =
			axis_distance_sq(self.center, node_aabb.min, sample_max, nearest_distance);
		if min_dist_sq > radius_sq {
			return Coverage::Empty;
		}

		let max_dist_sq =
			axis_distance_sq(self.center, node_aabb.min, sample_max, farthest_distance);
		if max_dist_sq <= radius_sq || lod == 0 {
			return Coverage::Full(self.material);
		}

		Coverage::Partial
	}
}

fn axis_distance_sq(
	center: [i64; 3],
	min: [i64; 3],
	max: [i64; 3],
	axis_distance: fn(i64, i64, i64) -> i128,
) -> i128 {
	(0..3)
		.map(|axis| axis_distance(center[axis], min[axis], max[axis]))
		.map(|distance| distance * distance)
		.sum()
}

fn nearest_distance(center: i64, min: i64, max: i64) -> i128 {
	if center < min {
		(min - center) as i128
	} else if center > max {
		(center - max) as i128
	} else {
		0
	}
}

fn farthest_distance(center: i64, min: i64, max: i64) -> i128 {
	let to_min = (center as i128 - min as i128).abs();
	let to_max = (center as i128 - max as i128).abs();
	to_min.max(to_max)
}
