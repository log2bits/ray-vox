use crate::chunk::material::Material;
use crate::volumes::{Containment, Volume};

pub struct Sphere {
	pub center: [f32; 3],
	pub radius: f32,
	pub material: Material,
}

impl Volume for Sphere {
	fn material(&self) -> Material {
		self.material
	}

	fn containment(&self, min: [i32; 3], max: [i32; 3]) -> Containment {
		let r2 = self.radius * self.radius;
		let [cx, cy, cz] = self.center;

		// Closest point on AABB to sphere center.
		let nx = cx.clamp(min[0] as f32, max[0] as f32);
		let ny = cy.clamp(min[1] as f32, max[1] as f32);
		let nz = cz.clamp(min[2] as f32, max[2] as f32);
		let dx = cx - nx;
		let dy = cy - ny;
		let dz = cz - nz;

		if dx * dx + dy * dy + dz * dz > r2 {
			return Containment::Empty;
		}

		// Farthest point on AABB from sphere center (worst-case corner).
		let fx = (cx - min[0] as f32).abs().max((cx - max[0] as f32).abs());
		let fy = (cy - min[1] as f32).abs().max((cy - max[1] as f32).abs());
		let fz = (cz - min[2] as f32).abs().max((cz - max[2] as f32).abs());

		if fx * fx + fy * fy + fz * fz <= r2 {
			Containment::Full
		} else {
			Containment::Partial
		}
	}
}
