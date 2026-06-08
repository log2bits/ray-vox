use ahash::AHashMap;
use std::hash::Hash;

#[derive(Default, Clone)]
pub struct Lut<T> {
	pub values: Vec<T>,
	/// Most-recent (value, index) hit. Collapses long same-value run callers
	/// (uniform fills) to a single equality check.
	last_hit: Option<(T, u32)>,
	/// O(1) intern map. Drop with `shrink_to_fit` for storage.
	index: AHashMap<T, u32>,
}

impl<T: PartialEq + Eq + Hash + Copy> Lut<T> {
	pub fn new() -> Self {
		Self {
			values: Vec::new(),
			last_hit: None,
			index: AHashMap::new(),
		}
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
		if let Some((v, i)) = self.last_hit {
			if v == value {
				return i;
			}
		}
		let values = &mut self.values;
		let idx = *self.index.entry(value).or_insert_with(|| {
			values.push(value);
			(values.len() - 1) as u32
		});
		self.last_hit = Some((value, idx));
		idx
	}

	pub fn clear(&mut self) {
		self.values.clear();
		self.last_hit = None;
		self.index.clear();
	}

	/// Drop the build-time acceleration map. Call before storing long-term.
	pub fn shrink_to_fit(&mut self) {
		self.index = AHashMap::new();
	}

	pub fn bytes(&self) -> u32 {
		24 + (std::mem::size_of::<T>() * self.values.len()) as u32
	}
}
