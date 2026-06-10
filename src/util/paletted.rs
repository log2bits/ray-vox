use crate::util::Lut;
use crate::util::PackedVec;
use bytemuck::{Pod, Zeroable};
use std::hash::Hash;

#[derive(Clone, Default)]
pub struct PalettedVec<T> {
	pub lut: Lut<T>,
	pub indices: PackedVec,
}

impl<T: PartialEq + Eq + Hash + Copy + Into<u32>> PalettedVec<T> {
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

	pub fn shrink_to_fit(&mut self) {
		self.lut.shrink_to_fit();
	}

	pub fn bit_width(&self) -> u32 {
		self.indices.bits
	}
}

impl<T> PalettedVec<T>
where
	T: PartialEq + Eq + Hash + Copy + Into<u32> + Pod + Zeroable,
{
	pub fn byte_size(&self) -> u32 {
		4 + (self.lut.values.len() * std::mem::size_of::<T>()) as u32 + self.indices.byte_size()
	}

	pub fn write_bytes<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
		let lut_count = [self.lut.values.len() as u32];
		w.write_all(bytemuck::cast_slice(&lut_count))?;
		w.write_all(bytemuck::cast_slice(&self.lut.values))?;
		self.indices.write_bytes(w)?;
		Ok(())
	}

	pub fn read_bytes<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
		let mut lut_count = [0u32; 1];
		r.read_exact(bytemuck::cast_slice_mut(&mut lut_count))?;
		let mut values = vec![T::zeroed(); lut_count[0] as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut values))?;
		let indices = PackedVec::read_bytes(r)?;
		let mut lut = Lut::new();
		lut.values = values;
		Ok(Self { lut, indices })
	}
}
