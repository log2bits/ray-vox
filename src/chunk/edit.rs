use crate::chunk::Material;
use crate::util::types::WorldPos;
use radsort::sort_by_key;
use std::ops::Range;

const SORTED_RUN_THRESHOLD: u32 = 64;

#[derive(Default, Clone)]
pub struct ChunkEdits {
	pub world_pos: WorldPos,
	pub edits: Vec<(Path, Material)>,
	pub ranges: Vec<SortableRange>,
	last_path: Path,
	run_start: u32,
	run_len: u32,
}

#[derive(Clone)]
pub struct SortableRange {
	pub range: Range<u32>,
	pub sorted: bool,
}

impl ChunkEdits {
	pub fn new(world_pos: WorldPos) -> Self {
		Self {
			world_pos,
			edits: Vec::new(),
			ranges: Vec::new(),
			last_path: Path::from(0u32),
			run_start: 0,
			run_len: 0,
		}
	}

	pub fn push(&mut self, path: Path, material: Material) {
		let path: Path = path.into();
		let idx = self.edits.len() as u32;
		self.edits.push((path, material));

		let depth_changed = idx == 0 || path.depth() != self.last_path.depth();
		let out_of_order = !depth_changed && path.0 < self.last_path.0;

		if depth_changed {
			self.ranges.push(SortableRange {
				range: idx..idx + 1,
				sorted: false,
			});
			self.run_start = idx;
			self.run_len = 1;
		} else if out_of_order {
			self.run_start = idx;
			self.run_len = 1;
			match self.ranges.last_mut() {
				Some(batch) if !batch.sorted => batch.range.end = idx + 1,
				_ => self.ranges.push(SortableRange {
					range: idx..idx + 1,
					sorted: false,
				}),
			}
		} else {
			self.run_len += 1;
			self.ranges.last_mut().unwrap().range.end = idx + 1;
			if self.run_len == SORTED_RUN_THRESHOLD {
				self.promote_run_to_sorted();
			}
		}

		self.last_path = path;
	}

	pub fn sort(&mut self) {
		for batch in &mut self.ranges {
			if !batch.sorted {
				let slice = &mut self.edits[batch.range.start as usize..batch.range.end as usize];
				sort_by_key(slice, |&(path, _)| u32::from(path));
				batch.sorted = true;
			}
		}
	}

	fn promote_run_to_sorted(&mut self) {
		let last = self.ranges.last_mut().unwrap();
		let batch_start = last.range.start;
		let batch_end = last.range.end;
		if batch_start < self.run_start {
			last.range = batch_start..self.run_start;
			self.ranges.push(SortableRange {
				range: self.run_start..batch_end,
				sorted: true,
			});
		} else {
			last.range = self.run_start..batch_end;
			last.sorted = true;
		}
	}
}

impl Default for WorldPos {
	fn default() -> Self {
		WorldPos::from([0; 3])
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
	pub fn from_coords(position: [u8; 3], depth: u8) -> Self {
		let [x, y, z] = position;
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

	pub fn common_depth(&self, other: &Path) -> usize {
		self.0
			.to_be_bytes()
			.iter()
			.zip(other.0.to_be_bytes().iter())
			.take_while(|(a, b)| a == b)
			.count()
	}
}
