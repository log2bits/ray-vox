use crate::chunk::material::Material;
use crate::generate::{Overlap, Volume};
use crate::world::WorldPosition;

pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub material: Material,
}

impl Volume for Sphere {
    fn overlap(&self, world_min: WorldPosition, world_max: WorldPosition) -> Overlap {
        let r2 = self.radius * self.radius;
        let [cx, cy, cz] = self.center;

        let nx = cx.clamp(world_min.position[0] as f32, world_max.position[0] as f32);
        let ny = cy.clamp(world_min.position[1] as f32, world_max.position[1] as f32);
        let nz = cz.clamp(world_min.position[2] as f32, world_max.position[2] as f32);
        let dx = cx - nx;
        let dy = cy - ny;
        let dz = cz - nz;

        if dx * dx + dy * dy + dz * dz > r2 {
            return Overlap::Empty;
        }

        let fx = (cx - world_min.position[0] as f32).abs().max((cx - world_max.position[0] as f32).abs());
        let fy = (cy - world_min.position[1] as f32).abs().max((cy - world_max.position[1] as f32).abs());
        let fz = (cz - world_min.position[2] as f32).abs().max((cz - world_max.position[2] as f32).abs());

        if fx * fx + fy * fy + fz * fz <= r2 {
            Overlap::Full
        } else {
            Overlap::Partial
        }
    }

    fn material(&self, _world_min: WorldPosition, _world_max: WorldPosition) -> Option<Material> {
        Some(self.material)
    }
}
