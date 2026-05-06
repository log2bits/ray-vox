/// 32-bit voxel value.
///
/// bits 31-8: rgb (24-bit linear)
/// bits  7-4: roughness nibble (0 = mirror, 15 = fully diffuse)
/// bit      3: emissive
/// bit      2: metallic
/// bit      1: transparent
/// bit      0: textured
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Voxel(u32);

impl Voxel {
	pub fn from_rgb_flags(
		rgb: [u8; 3],
		roughness: u8,
		emissive: bool,
		metallic: bool,
		transparent: bool,
		textured: bool,
	) -> Self {
		let r = (rgb[0] as u32) << 24;
		let g = (rgb[1] as u32) << 16;
		let b = (rgb[2] as u32) << 8;
		let rough = ((roughness & 0xf) as u32) << 4;
		let e = if emissive { 1 << 3 } else { 0 };
		let m = if metallic { 1 << 2 } else { 0 };
		let t = if transparent { 1 << 1 } else { 0 };
		let tex = if textured { 1 } else { 0 };
		Voxel(r | g | b | rough | e | m | t)
	}

	pub fn air() -> Self {
		Voxel(0)
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

	pub fn roughness(self) -> u8 {
		((self.0 >> 4) & 0xf) as u8
	}

	pub fn emissive(self) -> bool {
		(self.0 >> 3) & 1 != 0
	}

	pub fn metallic(self) -> bool {
		(self.0 >> 2) & 1 != 0
	}

	pub fn transparent(self) -> bool {
		(self.0 >> 1) & 1 != 0
	}

	pub fn textured(self) -> bool {
		self.0 & 1 != 0
	}
}

impl From<u32> for Voxel {
	fn from(v: u32) -> Self {
		Voxel(v)
	}
}

impl From<Voxel> for u32 {
	fn from(v: Voxel) -> Self {
		v.0
	}
}
