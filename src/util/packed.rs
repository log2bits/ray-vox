#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedVec {
	pub words: Vec<u32>,
	pub bits: u32,
	pub len: u32,
}

impl Default for PackedVec {
	fn default() -> Self {
		Self::new()
	}
}

impl PackedVec {
	pub fn new() -> Self {
		Self {
			words: Vec::new(),
			bits: 1,
			len: 0,
		}
	}

	pub fn with_bits(bits: u32) -> Self {
		assert!(bits > 0 && bits <= 32);
		Self {
			words: Vec::new(),
			bits,
			len: 0,
		}
	}

	pub fn len(&self) -> u32 {
		self.len
	}

	pub fn is_empty(&self) -> bool {
		self.len == 0
	}

	pub fn clear(&mut self) {
		self.words.clear();
		self.bits = 1;
		self.len = 0;
	}

	pub fn truncate(&mut self, new_len: u32) {
		assert!(new_len <= self.len);
		self.len = new_len;
		let bits_used = self.len * self.bits;
		let words_needed = ((bits_used + 31) / 32) as usize;
		let leftover = bits_used % 32;
		if leftover > 0 {
			if let Some(w) = self.words.get_mut(words_needed.saturating_sub(1)) {
				*w &= (1u32 << leftover) - 1;
			}
		}
		self.words.truncate(words_needed);
	}

	#[inline]
	pub fn push(&mut self, value: u32) {
		self.ensure_width(value);
		let bit_pos = self.len * self.bits;
		let word_idx = (bit_pos / 32) as usize;
		let bit_offset = bit_pos % 32;

		if word_idx >= self.words.len() {
			self.words.push(0);
		}
		self.words[word_idx] |= value << bit_offset;

		let bits_written = 32 - bit_offset;
		if bits_written < self.bits {
			self.words.push(0);
			self.words[word_idx + 1] |= value >> bits_written;
		}
		self.len += 1;
	}

	#[inline]
	pub fn get(&self, index: u32) -> u32 {
		debug_assert!(
			index < self.len,
			"index {index} out of bounds (len={})",
			self.len
		);
		let bit_pos = index * self.bits;
		let word_idx = (bit_pos / 32) as usize;
		let bit_offset = bit_pos % 32;
		let mask = Self::mask(self.bits);
		let primary = self.words[word_idx] >> bit_offset;
		if bit_offset + self.bits > 32 {
			let bits_read = 32 - bit_offset;
			let secondary = self.words[word_idx + 1] << bits_read;
			(primary | secondary) & mask
		} else {
			primary & mask
		}
	}

	#[inline]
	pub fn set(&mut self, index: u32, value: u32) {
		debug_assert!(
			index < self.len,
			"index {index} out of bounds (len={})",
			self.len
		);
		self.ensure_width(value);
		self.set_raw(index, value);
	}

	pub fn insert(&mut self, index: u32, value: u32) {
		assert!(index <= self.len);
		self.ensure_width(value);
		let bit_pos = self.len * self.bits;
		let word_idx = (bit_pos / 32) as usize;
		let bit_offset = bit_pos % 32;
		if word_idx >= self.words.len() {
			self.words.push(0);
		}
		if bit_offset + self.bits > 32 && word_idx + 1 >= self.words.len() {
			self.words.push(0);
		}
		self.len += 1;
		let mut i = self.len - 1;
		while i > index {
			let v = self.get(i - 1);
			self.set_raw(i, v);
			i -= 1;
		}
		self.set_raw(index, value);
	}

	pub fn remove(&mut self, index: u32) -> u32 {
		assert!(index < self.len);
		let value = self.get(index);
		let mut i = index;
		while i < self.len - 1 {
			let v = self.get(i + 1);
			self.set_raw(i, v);
			i += 1;
		}
		self.len -= 1;
		let bits_used = self.len * self.bits;
		let words_needed = ((bits_used + 31) / 32) as usize;
		let leftover = bits_used % 32;
		if leftover > 0 {
			if let Some(w) = self.words.get_mut(words_needed.saturating_sub(1)) {
				*w &= (1u32 << leftover) - 1;
			}
		}
		self.words.truncate(words_needed);
		value
	}

	pub fn repack(&self, new_bits: u32) -> Self {
		assert!(new_bits > 0 && new_bits <= 32);
		let mut new = Self::with_bits(new_bits);
		new.words
			.reserve(((self.len as usize * new_bits as usize) + 31) / 32);
		for i in 0..self.len {
			new.push(self.get(i));
		}
		new
	}

	#[inline]
	fn set_raw(&mut self, index: u32, value: u32) {
		let bit_pos = index * self.bits;
		let word_idx = (bit_pos / 32) as usize;
		let bit_offset = bit_pos % 32;
		let mask = Self::mask(self.bits);
		self.words[word_idx] &= !(mask << bit_offset);
		self.words[word_idx] |= value << bit_offset;
		if bit_offset + self.bits > 32 {
			let bits_written = 32 - bit_offset;
			let straddle_mask = mask >> bits_written;
			self.words[word_idx + 1] &= !straddle_mask;
			self.words[word_idx + 1] |= value >> bits_written;
		}
	}

	#[inline]
	fn ensure_width(&mut self, value: u32) {
		if self.bits < 32 && value >> self.bits != 0 {
			*self = self.repack(32 - value.leading_zeros());
		}
	}

	#[inline]
	fn mask(bits: u32) -> u32 {
		if bits == 32 {
			u32::MAX
		} else {
			(1u32 << bits) - 1
		}
	}

	pub fn byte_size(&self) -> u32 {
		12 + (self.words.len() * 4) as u32
	}

	pub fn write_bytes<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
		let header = [self.len, self.bits, self.words.len() as u32];
		w.write_all(bytemuck::cast_slice(&header))?;
		w.write_all(bytemuck::cast_slice(&self.words))?;
		Ok(())
	}

	pub fn read_bytes<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
		let mut header = [0u32; 3];
		r.read_exact(bytemuck::cast_slice_mut(&mut header))?;
		let [len, bits, words_count] = header;
		let mut words = vec![0u32; words_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut words))?;
		Ok(Self { words, bits, len })
	}
}
