use crate::tree::Ray;

#[derive(Clone, Copy)]
pub struct CameraPos {
	pub chunk: [i64; 3], // LOD-0 chunk coordinates
	pub local: [f32; 3], // offset within chunk in [0, 256)
	pub yaw:   f32,      // radians, 0 = +Z, positive = right
	pub pitch: f32,      // radians, 0 = horizontal, positive = up
	pub fov_y: f32,      // vertical field of view in radians
}

impl CameraPos {
	pub fn new(local: [f32; 3], yaw: f32, pitch: f32) -> Self {
		Self { chunk: [0; 3], local, yaw, pitch, fov_y: std::f32::consts::FRAC_PI_4 }
	}

	pub fn forward(&self) -> [f32; 3] {
		let (sy, cy) = self.yaw.sin_cos();
		let (sp, cp) = self.pitch.sin_cos();
		[cy * cp, sp, sy * cp]
	}

	pub fn right(&self) -> [f32; 3] {
		let (sy, cy) = self.yaw.sin_cos();
		[sy, 0.0, -cy]
	}

	pub fn up(&self) -> [f32; 3] {
		let f = self.forward();
		let r = self.right();
		// up = right × forward
		[
			r[1] * f[2] - r[2] * f[1],
			r[2] * f[0] - r[0] * f[2],
			r[0] * f[1] - r[1] * f[0],
		]
	}

	/// Camera position in chunk-local voxel coords (ignores chunk field for MVP).
	pub fn world_pos(&self) -> [f32; 3] {
		self.local
	}

	pub fn ray(&self) -> Ray {
		Ray { origin: self.local, dir: self.forward() }
	}
}
