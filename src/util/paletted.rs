use crate::util::Lut;
use crate::util::PackedVec;

pub struct PalettedVec {
	pub lut: Lut<u32>,
	pub indices: PackedVec,
}

impl PalettedVec {
	pub fn new() -> Self {
		Self {
			lut: Lut::new(),
			indices: PackedVec::new(),
		}
	}

	pub fn push(&mut self, value: u32) {
		let idx = self.lut.get_or_add(value);
		self.indices.push(idx);
	}

	pub fn get(&self, index: u32) -> u32 {
		self.lut.get(self.indices.get(index))
	}

	pub fn len(&self) -> u32 {
		self.indices.len()
	}

	pub fn is_empty(&self) -> bool {
		self.indices.is_empty()
	}
}
