// Free-flying first-person camera. Yaw is rotation around world-up (Y); at
// yaw = 0 the camera looks along +Z, and positive yaw turns clockwise from
// above. Pitch tilts up/down and gets clamped by the input handler to avoid
// gimbal flip.
pub struct Camera {
	pub position: [f32; 3],
	pub yaw: f32,
	pub pitch: f32,
	pub fov_y: f32,
	pub aspect: f32,
}

impl Camera {
	pub fn new(position: [f32; 3]) -> Self {
		Self {
			position,
			yaw: 0.0,
			pitch: 0.0,
			fov_y: 60f32.to_radians(),
			aspect: 1.0,
		}
	}

	pub fn eye(&self) -> [f32; 3] {
		self.position
	}

	pub fn forward(&self) -> [f32; 3] {
		let cos_pitch = self.pitch.cos();
		let sin_pitch = self.pitch.sin();
		let cos_yaw = self.yaw.cos();
		let sin_yaw = self.yaw.sin();
		[cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw]
	}

	pub fn right(&self) -> [f32; 3] {
		let cos_yaw = self.yaw.cos();
		let sin_yaw = self.yaw.sin();
		[cos_yaw, 0.0, -sin_yaw]
	}

	pub fn up(&self) -> [f32; 3] {
		let forward = self.forward();
		let right = self.right();
		[
			forward[1] * right[2] - forward[2] * right[1],
			forward[2] * right[0] - forward[0] * right[2],
			forward[0] * right[1] - forward[1] * right[0],
		]
	}

	pub fn fov_scale(&self) -> f32 {
		(self.fov_y * 0.5).tan()
	}

	pub fn translate_forward(&mut self, distance: f32) {
		let forward = self.forward();
		self.position[0] += forward[0] * distance;
		self.position[1] += forward[1] * distance;
		self.position[2] += forward[2] * distance;
	}

	pub fn translate_right(&mut self, distance: f32) {
		let right = self.right();
		self.position[0] += right[0] * distance;
		self.position[1] += right[1] * distance;
		self.position[2] += right[2] * distance;
	}

	pub fn translate_world_up(&mut self, distance: f32) {
		self.position[1] += distance;
	}

	// Apply mouse-motion yaw and pitch deltas; pitch clamps clear of the poles.
	pub fn apply_look_delta(&mut self, yaw_delta: f32, pitch_delta: f32) {
		self.yaw += yaw_delta;
		let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
		self.pitch = (self.pitch + pitch_delta).clamp(-pitch_limit, pitch_limit);
	}
}
