use crate::util::Lut;
use crate::util::PackedVec;

#[derive(Clone)]
pub struct PalettedVec<T> {
	pub lut: Lut<T>,
	pub indices: PackedVec,
}

impl<T: PartialEq + Copy + Into<u32>> PalettedVec<T> {
	pub fn new() -> Self {
		Self {
			lut: Lut::new(),
			indices: PackedVec::new(),
		}
	}

	pub fn push(&mut self, value: T) {
		let idx = self.lut.get_or_add(value);
		self.indices.push(idx);
	}

	pub fn get(&self, index: u32) -> T {
		self.lut.get(self.indices.get(index))
	}

	pub fn len(&self) -> u32 {
		self.indices.len()
	}

	pub fn is_empty(&self) -> bool {
		self.indices.is_empty()
	}

	pub fn clear(&mut self) {
		self.lut.clear();
		self.indices.clear();
	}
}
