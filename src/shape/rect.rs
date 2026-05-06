use super::{Coverage, Shape};
use crate::{tree::Aabb, types::Voxel};

pub struct Rect {
	pub min: [i64; 3],
	pub max: [i64; 3],
	pub material: Voxel,
}

impl Shape for Rect {
	fn aabb(&self) -> Aabb {
		Aabb { min: self.min, max: self.max }
	}
	fn coverage(&self, node_aabb: Aabb, _lod: u8) -> Coverage {
		let rect = self.aabb();
		if !rect.overlaps(&node_aabb) {
			Coverage::Empty
		} else if rect.contains(&node_aabb) {
			Coverage::Full(self.material)
		} else {
			Coverage::Partial
		}
	}
}
