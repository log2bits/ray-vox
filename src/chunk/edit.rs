use crate::chunk::Material;
use crate::util::types::ChunkPos;
use radsort::sort_by_key;

#[derive(Default, Clone)]
pub struct EditPacket {
	pub edits: Vec<(Path, Material)>,
	pub sorted: bool,
}

impl EditPacket {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn from_sorted(edits: Vec<(Path, Material)>) -> Self {
		Self { edits, sorted: true }
	}

	pub fn push(&mut self, path: Path, material: Material) {
		if self.sorted {
			if let Some(&(last, _)) = self.edits.last() {
				if u32::from(path) < u32::from(last) {
					self.sorted = false;
				}
			}
		}
		self.edits.push((path, material));
	}

	pub fn sort(&mut self) {
		if !self.sorted {
			sort_by_key(&mut self.edits, |&(path, _)| u32::from(path));
			self.sorted = true;
		}
	}

	pub fn len(&self) -> usize {
		self.edits.len()
	}

	pub fn is_empty(&self) -> bool {
		self.edits.is_empty()
	}
}

#[derive(Default, Clone, Copy)]
pub struct Path(pub u32);

impl From<[u8; 4]> for Path {
	fn from(bytes: [u8; 4]) -> Self {
		Path(u32::from_be_bytes(bytes))
	}
}

impl From<Path> for u32 {
	fn from(p: Path) -> Self {
		p.0
	}
}

impl From<u32> for Path {
	fn from(v: u32) -> Self {
		Path(v)
	}
}

impl Path {
	pub fn from_coords(position: ChunkPos, depth: u8) -> Self {
		let [x, y, z] = <[u8; 3]>::from(position);
		let slot = |shift: u8| -> u8 {
			((((x >> shift) & 3) << 4) | (((y >> shift) & 3) << 2) | ((z >> shift) & 3)) + 1
		};
		let b0 = if depth > 0 { slot(6) } else { 0 };
		let b1 = if depth > 1 { slot(4) } else { 0 };
		let b2 = if depth > 2 { slot(2) } else { 0 };
		let b3 = if depth > 3 { slot(0) } else { 0 };
		Path(u32::from_be_bytes([b0, b1, b2, b3]))
	}

	pub fn depth(&self) -> u8 {
		self.0.to_be_bytes().iter().take_while(|&&b| b != 0).count() as u8
	}

	pub fn slot_at(&self, depth: u8) -> u8 {
		self.0.to_be_bytes()[depth as usize] - 1
	}

	pub fn is_root(&self) -> bool {
		self.0 == 0
	}

	pub fn to_coords(&self) -> (ChunkPos, u8) {
		let depth = self.depth();
		let bytes = self.0.to_be_bytes();
		let mut x = 0u8;
		let mut y = 0u8;
		let mut z = 0u8;
		for d in 0..depth {
			let raw = bytes[d as usize] - 1;
			let shift = 6 - 2 * d;
			x |= ((raw >> 4) & 3) << shift;
			y |= ((raw >> 2) & 3) << shift;
			z |= (raw & 3) << shift;
		}
		(ChunkPos::new(x, y, z), depth)
	}

	pub fn common_depth(&self, other: &Path) -> usize {
		self.0
			.to_be_bytes()
			.iter()
			.zip(other.0.to_be_bytes().iter())
			.take_while(|(a, b)| a == b)
			.count()
	}
}
