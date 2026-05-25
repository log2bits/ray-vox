use std::ops::Range;

use radsort::sort_by_key;

const SORTED_RUN_THRESHOLD: u32 = 64;

#[derive(Default)]
pub struct Edits {
	pub edits: Vec<(u32, u32)>, // (path, material)
	pub batches: Vec<EditBatch>,
	last_path: u32,
	run_start: u32,
	run_len: u32,
}

pub enum EditBatch {
	Sorted(Range<u32>),
	Unsorted(Range<u32>),
}

impl EditBatch {
	pub fn range(&self) -> &Range<u32> {
		match self {
			Self::Sorted(r) | Self::Unsorted(r) => r,
		}
	}

	pub fn range_mut(&mut self) -> &mut Range<u32> {
		match self {
			Self::Sorted(r) | Self::Unsorted(r) => r,
		}
	}
}

impl Edits {
	pub fn new() -> Self {
		Self {
			edits: Vec::new(),
			batches: Vec::new(),
			last_path: 0,
			run_start: 0,
			run_len: 0,
		}
	}

	pub fn push(&mut self, path: impl Into<u32>, material: u32) {
		let path = path.into();
		let idx = self.edits.len() as u32;
		self.edits.push((path, material));

		let depth_changed = idx == 0 || path_depth(path) != path_depth(self.last_path);
		let out_of_order = !depth_changed && path < self.last_path;

		if depth_changed {
			self.batches.push(EditBatch::Unsorted(idx..idx + 1));
			self.run_start = idx;
			self.run_len = 1;
		} else if out_of_order {
			self.run_start = idx;
			self.run_len = 1;
			match self.batches.last_mut() {
				Some(EditBatch::Unsorted(range)) => range.end = idx + 1,
				_ => self.batches.push(EditBatch::Unsorted(idx..idx + 1)),
			}
		} else {
			self.run_len += 1;
			self.batches.last_mut().unwrap().range_mut().end = idx + 1;
			if self.run_len == SORTED_RUN_THRESHOLD {
				self.promote_run_to_sorted();
			}
		}

		self.last_path = path;
	}

	pub fn sort(&mut self) {
		for batch in &mut self.batches {
			if let EditBatch::Unsorted(range) = batch {
				let slice = &mut self.edits[range.start as usize..range.end as usize];
				sort_by_key(slice, |&(path, _)| path);
				*batch = EditBatch::Sorted(range.clone());
			}
		}
	}

	fn promote_run_to_sorted(&mut self) {
		let last = self.batches.last_mut().unwrap();
		let batch_start = last.range().start;
		let batch_end = last.range().end;
		if batch_start < self.run_start {
			*last = EditBatch::Unsorted(batch_start..self.run_start);
			self.batches
				.push(EditBatch::Sorted(self.run_start..batch_end));
		} else {
			*last = EditBatch::Sorted(self.run_start..batch_end);
		}
	}
}

fn path_depth(path: u32) -> u8 {
	path.to_be_bytes().iter().take_while(|&&b| b != 0).count() as u8
}

pub struct Path(u32);

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
}
