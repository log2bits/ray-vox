use crate::chunk::material::Material;
use crate::generate::{Overlap, Volume};
use crate::world::WorldPosition;

pub struct Rect {
    pub min: WorldPosition,
    pub max: WorldPosition,
    pub material: Material,
}

impl Volume for Rect {
    fn overlap(&self, world_min: WorldPosition, world_max: WorldPosition) -> Overlap {
        for i in 0..3 {
            if self.max.position[i] <= world_min.position[i]
                || self.min.position[i] >= world_max.position[i]
            {
                return Overlap::Empty;
            }
        }
        for i in 0..3 {
            if self.min.position[i] > world_min.position[i]
                || self.max.position[i] < world_max.position[i]
            {
                return Overlap::Partial;
            }
        }
        Overlap::Full
    }

    fn material(&self, _world_min: WorldPosition, _world_max: WorldPosition) -> Option<Material> {
        Some(self.material)
    }
}
