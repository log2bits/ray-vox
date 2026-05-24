#[derive(Default, Clone)]
pub struct Lut<T> {
	pub values: Vec<T>,
}

impl<T: PartialEq + Copy> Lut<T> {
	pub fn new() -> Self {
		Self { values: Vec::new() }
	}

	pub fn len(&self) -> u32 {
		self.values.len() as u32
	}

	pub fn is_empty(&self) -> bool {
		self.values.is_empty()
	}

	pub fn get(&self, idx: u32) -> T {
		self.values[idx as usize]
	}

	pub fn get_or_add(&mut self, value: T) -> u32 {
		self.values
			.iter()
			.position(|&v| v == value)
			.unwrap_or_else(|| {
				self.values.push(value);
				self.values.len() - 1
			}) as u32
	}
}
