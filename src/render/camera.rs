// A simple orbit camera. The eye orbits around a target point at a fixed
// distance. Yaw rotates around the Y axis, pitch rotates around the horizon.
// The renderer reads out the eye position and the three basis vectors.
pub struct OrbitCamera {
	pub target: [f32; 3],
	pub distance: f32,
	pub yaw: f32,
	pub pitch: f32,
	pub fov_y: f32,
	pub aspect: f32,
}

impl OrbitCamera {
	pub fn new(target: [f32; 3], distance: f32) -> Self {
		Self {
			target,
			distance,
			yaw: 0.0,
			pitch: 0.3,
			fov_y: 60f32.to_radians(),
			aspect: 1.0,
		}
	}

	pub fn eye(&self) -> [f32; 3] {
		let cos_pitch = self.pitch.cos();
		let sin_pitch = self.pitch.sin();
		let cos_yaw = self.yaw.cos();
		let sin_yaw = self.yaw.sin();
		[
			self.target[0] + self.distance * cos_pitch * sin_yaw,
			self.target[1] + self.distance * sin_pitch,
			self.target[2] + self.distance * cos_pitch * cos_yaw,
		]
	}

	// Unit vector pointing from the eye toward the target.
	pub fn forward(&self) -> [f32; 3] {
		let eye = self.eye();
		let dx = self.target[0] - eye[0];
		let dy = self.target[1] - eye[1];
		let dz = self.target[2] - eye[2];
		let length = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
		[dx / length, dy / length, dz / length]
	}

	// Right basis vector (cross of forward and world-up), unit length.
	pub fn right(&self) -> [f32; 3] {
		let f = self.forward();
		let world_up = [0.0f32, 1.0, 0.0];
		let x = f[1] * world_up[2] - f[2] * world_up[1];
		let y = f[2] * world_up[0] - f[0] * world_up[2];
		let z = f[0] * world_up[1] - f[1] * world_up[0];
		let length = (x * x + y * y + z * z).sqrt().max(1e-6);
		[x / length, y / length, z / length]
	}

	// Camera-up basis vector, unit length. Computed as right x forward so it
	// stays orthogonal even when the camera pitches.
	pub fn up(&self) -> [f32; 3] {
		let f = self.forward();
		let r = self.right();
		[
			r[1] * f[2] - r[2] * f[1],
			r[2] * f[0] - r[0] * f[2],
			r[0] * f[1] - r[1] * f[0],
		]
	}

	// Half-tangent of the vertical field of view, used by the shader to scale
	// pixel offsets into ray directions.
	pub fn fov_scale(&self) -> f32 {
		(self.fov_y * 0.5).tan()
	}
}
