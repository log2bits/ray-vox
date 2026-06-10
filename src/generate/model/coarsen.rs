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
		Self { children, cursor: Cursor::Root }
	}

	fn cursor_from_child(chunk: &'a Chunk, c: Child) -> Cursor<'a> {
		match c {
			Child::Empty => Cursor::Empty,
			Child::Filled(m) => Cursor::Filled(m),
			Child::Interior(idx) => Cursor::Interior { chunk, idx },
			Child::Leaf(idx) => Cursor::Leaf { chunk, idx },
		}
	}

	fn cursor_at_chunk_root(chunk: &'a Chunk) -> Cursor<'a> {
		Self::cursor_from_child(chunk, chunk.root_child())
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
			Cursor::Leaf { chunk, idx } => chunk.leaf_voxel_sample(idx, slot),
		}
	}

	fn descend(&self, slot: u8) -> Self {
		let new_cursor = match self.cursor {
			Cursor::Root => match self.children[slot as usize] {
				None => Cursor::Empty,
				Some(c) => Self::cursor_at_chunk_root(c),
			},
			Cursor::Empty => Cursor::Empty,
			Cursor::Filled(m) => Cursor::Filled(m),
			Cursor::Interior { chunk, idx } => {
				Self::cursor_from_child(chunk, chunk.descend_child(Child::Interior(idx), slot))
			}
			Cursor::Leaf { chunk, idx } => {
				Self::cursor_from_child(chunk, chunk.descend_child(Child::Leaf(idx), slot))
			}
		};
		Self { children: self.children, cursor: new_cursor }
	}
}
