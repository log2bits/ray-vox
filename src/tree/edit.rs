mod apply;
mod sort;

use crate::types::{BitpackedArray, Lut};

pub const DELETE: u32 = 0;

// Each byte is 1..=64 (slot index + 1). Trailing 0s are unused.
// depth() = number of filled bytes = how far down the tree this edit reaches.
// depth=0 means root (covers whole tree), depth=DEPTH means single voxel.
// Lexicographic order = preorder traversal order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreePath<const DEPTH: usize>([u8; DEPTH]);

impl<const DEPTH: usize> TreePath<DEPTH> {
	// depth: 0 = whole tree, DEPTH = single voxel.
	pub fn new(position: [u64; 3], depth: u8, leaf_size: u64) -> Self {
		debug_assert!(depth as usize <= DEPTH);
		let leaf = position.map(|p| p / leaf_size);
		let mut path = [0u8; DEPTH];
		for d in 0..depth as usize {
			let shift = 2 * (DEPTH - 1 - d);
			let [x, y, z] = leaf.map(|p| ((p >> shift) & 3) as u8);
			path[d] = (x | (y << 2) | (z << 4)) + 1;
		}
		Self(path)
	}

	pub fn from_packed(path: [u8; DEPTH]) -> Self {
		Self(path)
	}

	pub fn from_raw(indices: [u8; DEPTH], depth: u8) -> Self {
		debug_assert!(depth as usize <= DEPTH);
		let mut path = [0u8; DEPTH];
		for i in 0..depth as usize {
			path[i] = indices[i] + 1;
		}
		Self(path)
	}

	pub fn depth(&self) -> u8 {
		self.0.iter().position(|&b| b == 0).unwrap_or(DEPTH) as u8
	}

	pub fn as_bytes(&self) -> &[u8; DEPTH] {
		&self.0
	}

	pub fn to_raw(&self) -> ([u8; DEPTH], u8) {
		let depth = self.depth();
		let mut out = [0u8; DEPTH];
		for i in 0..depth as usize {
			out[i] = self.0[i] - 1;
		}
		(out, depth)
	}
}

pub struct Edit<const DEPTH: usize> {
	pub path: TreePath<DEPTH>,
	pub value: u32,
}

impl<const DEPTH: usize> Edit<DEPTH> {
	// depth: 0 = whole tree, DEPTH = single voxel.
	pub fn new(value: u32, position: [u64; 3], depth: u8, leaf_size: u64) -> Self {
		Self { path: TreePath::new(position, depth, leaf_size), value }
	}
}

#[derive(Clone)]
pub struct EditPacket<const DEPTH: usize> {
	pub paths: Vec<TreePath<DEPTH>>,
	pub lut: Lut<u32>,
	pub values: BitpackedArray,
	pub sorted: bool,
}

impl<const DEPTH: usize> EditPacket<DEPTH> {
	pub fn new(sorted: bool) -> Self {
		Self {
			paths: Vec::new(),
			lut: Lut::new(),
			values: BitpackedArray::new(),
			sorted,
		}
	}

	pub fn add_edit(&mut self, edit: Edit<DEPTH>) {
		if !self.sorted {
			for i in 0..self.paths.len() {
				if self.paths[i] == edit.path {
					let lut_index = self.lut.get_or_add(edit.value);
					self.values.set(i as u32, lut_index);
					return;
				}
			}
		}
		self.paths.push(edit.path);
		self.values.push(self.lut.get_or_add(edit.value));
	}
}

#[derive(Default, Clone)]
pub struct OrderedEdits<const DEPTH: usize> {
	pub packets: Vec<EditPacket<DEPTH>>,
}

impl<const DEPTH: usize> OrderedEdits<DEPTH> {
	pub fn add_edit(&mut self, edit: Edit<DEPTH>) {
		let needs_new = self.packets.last().map_or(true, |p| p.sorted);
		if needs_new {
			self.packets.push(EditPacket::new(false));
		}
		self.packets.last_mut().unwrap().add_edit(edit);
	}

	pub fn add_edit_packet(&mut self, packet: EditPacket<DEPTH>) {
		self.packets.push(packet);
	}
}
