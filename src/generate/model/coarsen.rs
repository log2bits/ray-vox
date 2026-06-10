use crate::chunk::build::{Sample, Source, VoxelSample, build_chunk};
use crate::chunk::material::Material;
use crate::chunk::node::{CellState, pack_slot};
use crate::chunk::{Child, Chunk};

pub fn coarsen(finer: &[Option<&Chunk>; 64]) -> Chunk {
	build_chunk(&CoarsenSource::new(finer))
}

#[derive(Clone, Copy)]
struct CoarsenSource<'a> {
	children: &'a [Option<&'a Chunk>; 64],
	cursor: Cursor<'a>,
	coarse_depth: u8,
}

#[derive(Clone, Copy)]
enum Cursor<'a> {
	Root,
	Empty,
	Filled(Material),
	Interior { chunk: &'a Chunk, idx: u32 },
	Leaf { chunk: &'a Chunk, idx: u32 },
}

impl<'a> CoarsenSource<'a> {
	fn new(children: &'a [Option<&'a Chunk>; 64]) -> Self {
		Self { children, cursor: Cursor::Root, coarse_depth: 0 }
	}

	fn cursor_at_chunk_root(chunk: &'a Chunk) -> Cursor<'a> {
		if chunk.is_empty() {
			Cursor::Empty
		} else if chunk.is_uniform() {
			Cursor::Filled(chunk.materials.get(0))
		} else if chunk.interior_nodes.is_empty() {
			Cursor::Leaf { chunk, idx: 0 }
		} else {
			Cursor::Interior { chunk, idx: chunk.root_idx() }
		}
	}

	fn descend_interior(chunk: &'a Chunk, idx: u32, slot: u8) -> Cursor<'a> {
		match chunk.child(idx, slot) {
			Child::Empty => Cursor::Empty,
			Child::Filled(m) => Cursor::Filled(m),
			Child::Interior(c) => Cursor::Interior { chunk, idx: c },
			Child::Leaf(c) => Cursor::Leaf { chunk, idx: c },
		}
	}

	fn descend_leaf(chunk: &'a Chunk, idx: u32, slot: u8) -> Cursor<'a> {
		let leaf = &chunk.leaf_nodes[idx as usize];
		if leaf.occupancy.contains(slot) {
			Cursor::Filled(chunk.materials.get(leaf.material_index(slot)))
		} else {
			Cursor::Empty
		}
	}
}

impl<'a> Source for CoarsenSource<'a> {
	fn classify(&self, _lo: [i32; 3], _hi: [i32; 3], _depth: u8) -> Sample {
		match self.cursor {
			Cursor::Root => Sample::Subdivide,
			Cursor::Empty => Sample::Passthrough,
			Cursor::Filled(m) => Sample::Fill(m),
			Cursor::Interior { .. } | Cursor::Leaf { .. } => Sample::Subdivide,
		}
	}

	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		let slot = pack_slot(v);
		match self.cursor {
			Cursor::Root => unreachable!("coarsen root reached voxel level"),
			Cursor::Empty => VoxelSample::Passthrough,
			Cursor::Filled(m) => VoxelSample::Fill(m),
			Cursor::Interior { chunk, idx } => {
				let n = &chunk.interior_nodes[idx as usize];
				match n.masks.state(slot) {
					CellState::Empty => VoxelSample::Passthrough,
					_ => VoxelSample::Fill(chunk.materials.get(n.material_index(slot))),
				}
			}
			Cursor::Leaf { chunk, idx } => {
				let leaf = &chunk.leaf_nodes[idx as usize];
				if leaf.occupancy.contains(slot) {
					VoxelSample::Fill(chunk.materials.get(leaf.material_index(slot)))
				} else {
					VoxelSample::Passthrough
				}
			}
		}
	}

	fn descend(&self, slot: u8) -> Self {
		let child_depth = self.coarse_depth + 1;
		let new_cursor = match self.cursor {
			Cursor::Root => match self.children[slot as usize] {
				None => Cursor::Empty,
				Some(c) => Self::cursor_at_chunk_root(c),
			},
			Cursor::Empty => Cursor::Empty,
			Cursor::Filled(m) => Cursor::Filled(m),
			Cursor::Interior { chunk, idx } => Self::descend_interior(chunk, idx, slot),
			Cursor::Leaf { chunk, idx } => Self::descend_leaf(chunk, idx, slot),
		};
		Self { children: self.children, cursor: new_cursor, coarse_depth: child_depth }
	}
}
