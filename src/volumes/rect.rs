use crate::chunk::material::Material;
use crate::volumes::{Containment, Volume};

pub struct Rect {
	pub min: [i32; 3],
	pub max: [i32; 3],
	pub material: Material,
}

impl Volume for Rect {
	fn material(&self) -> Material {
		self.material
	}

	fn containment(&self, min: [i32; 3], max: [i32; 3]) -> Containment {
		if self.max[0] <= min[0]
			|| self.min[0] >= max[0]
			|| self.max[1] <= min[1]
			|| self.min[1] >= max[1]
			|| self.max[2] <= min[2]
			|| self.min[2] >= max[2]
		{
			return Containment::Empty;
		}
		if self.min[0] <= min[0]
			&& self.max[0] >= max[0]
			&& self.min[1] <= min[1]
			&& self.max[1] >= max[1]
			&& self.min[2] <= min[2]
			&& self.max[2] >= max[2]
		{
			return Containment::Full;
		}
		Containment::Partial
	}
}
