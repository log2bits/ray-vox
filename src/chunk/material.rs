use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Pod, Zeroable)]
#[repr(transparent)]
pub struct Material(u32);

impl Material {
	pub fn from_rgb_pbr_id(rgb: [u8; 3], pbr_id: u8) -> Self {
		let r = (rgb[0] as u32) << 24;
		let g = (rgb[1] as u32) << 16;
		let b = (rgb[2] as u32) << 8;
		let pbr_id = (pbr_id as u32 & 0xF) << 4;
		Material(r | g | b | pbr_id)
	}

	pub fn air() -> Self {
		Material(0)
	}

	pub fn is_air(self) -> bool {
		self.0 == 0
	}

	pub fn rgb(self) -> [u8; 3] {
		[
			(self.0 >> 24) as u8,
			(self.0 >> 16) as u8,
			(self.0 >> 8) as u8,
		]
	}

	pub fn pbr_id(self) -> u8 {
		((self.0 >> 4) & 0xF) as u8
	}
}

impl From<u32> for Material {
	fn from(v: u32) -> Self {
		Material(v)
	}
}

impl From<Material> for u32 {
	fn from(v: Material) -> Self {
		v.0
	}
}
