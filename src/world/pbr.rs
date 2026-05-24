#[derive(PartialEq, Clone, Copy)]
pub struct Pbr {
	roughness: u8,
	emissive: u8,
	scattering: u8,
	absorption: [u8; 3],
	ior: u8,
	anisotropy: u8,
}
