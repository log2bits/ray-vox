use crate::chunk::material::Material;
use crate::generate::{Overlap, Volume};
use crate::util::types::WorldPos;

pub struct Rect {
	pub min: WorldPos,
	pub max: WorldPos,
	pub material: Material,
}

impl Volume for Rect {
	fn overlap(&self, world_min: WorldPos, world_max: WorldPos) -> Overlap {
		for i in 0..3 {
			if self.max.pos[i] <= world_min.pos[i] || self.min.pos[i] >= world_max.pos[i] {
				return Overlap::Empty;
			}
		}
		for i in 0..3 {
			if self.min.pos[i] > world_min.pos[i] || self.max.pos[i] < world_max.pos[i] {
				return Overlap::Partial;
			}
		}
		Overlap::Full
	}

	fn material(&self, _world_min: WorldPos, _world_max: WorldPos) -> Option<Material> {
		Some(self.material)
	}
}
