use super::build::{Sample, Source, VoxelSample};
use super::edit::Path;
use super::material::Material;
use super::{Child, Chunk};
use crate::util::types::ChunkPos;

#[derive(Clone, Copy)]
pub struct ChunkSource<'a> {
	chunk: &'a Chunk,
	state: ChunkCursor,
}

#[derive(Clone, Copy)]
enum ChunkCursor {
	Empty,
	Filled(Material),
	Interior(u32),
	Leaf(u32),
}

impl<'a> ChunkSource<'a> {
	pub fn new(chunk: &'a Chunk) -> Self {
		let state = if chunk.is_empty() {
			ChunkCursor::Empty
		} else if chunk.is_uniform() {
			ChunkCursor::Filled(chunk.materials.get(0))
		} else if chunk.interior_nodes.is_empty() {
			ChunkCursor::Leaf(0)
		} else {
			ChunkCursor::Interior(chunk.root_idx())
		};
		Self { chunk, state }
	}
}

impl Source for ChunkSource<'_> {
	#[inline]
	fn classify(&self, _lo: [i32; 3], _hi: [i32; 3], _depth: u8) -> Sample {
		match self.state {
			ChunkCursor::Empty => Sample::Passthrough,
			ChunkCursor::Filled(m) => Sample::Fill(m),
			ChunkCursor::Interior(_) | ChunkCursor::Leaf(_) => Sample::Subdivide,
		}
	}

	#[inline]
	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		match self.state {
			ChunkCursor::Empty => VoxelSample::Passthrough,
			ChunkCursor::Filled(m) => VoxelSample::Set(m),
			ChunkCursor::Leaf(idx) => {
				let leaf = &self.chunk.leaf_nodes[idx as usize];
				let slot = leaf_slot(v);
				if leaf.occupancy.contains(slot) {
					VoxelSample::Set(self.chunk.materials.get(leaf.material_index(slot)))
				} else {
					VoxelSample::Passthrough
				}
			}
			ChunkCursor::Interior(_) => unreachable!("interior node reached at voxel level"),
		}
	}

	#[inline]
	fn descend(&self, slot: u8) -> Self {
		let state = match self.state {
			ChunkCursor::Empty => ChunkCursor::Empty,
			ChunkCursor::Filled(m) => ChunkCursor::Filled(m),
			ChunkCursor::Interior(idx) => match self.chunk.child(idx, slot) {
				Child::Empty => ChunkCursor::Empty,
				Child::Filled(m) => ChunkCursor::Filled(m),
				Child::Interior(c) => ChunkCursor::Interior(c),
				Child::Leaf(c) => ChunkCursor::Leaf(c),
			},
			ChunkCursor::Leaf(leaf_idx) => {
				let leaf = &self.chunk.leaf_nodes[leaf_idx as usize];
				if leaf.occupancy.contains(slot) {
					ChunkCursor::Filled(self.chunk.materials.get(leaf.material_index(slot)))
				} else {
					ChunkCursor::Empty
				}
			}
		};
		Self { chunk: self.chunk, state }
	}
}

#[inline]
fn leaf_slot(v: [i32; 3]) -> u8 {
	let [x, y, z] = v;
	(((x & 3) << 4) | ((y & 3) << 2) | (z & 3)) as u8
}

#[derive(Clone, Copy)]
pub struct DiscreteSource<'a> {
	edits: &'a [(Path, Material)],
	depth: u8,
	inherited: Option<Material>,
}

impl<'a> DiscreteSource<'a> {
	pub fn new(edits: &'a [(Path, Material)]) -> Self {
		let (deeper, inherited) = absorb_terminators(edits, 0, None);
		Self { edits: deeper, depth: 0, inherited }
	}
}

#[inline]
fn absorb_terminators<'a>(
	edits: &'a [(Path, Material)],
	depth: u8,
	prior: Option<Material>,
) -> (&'a [(Path, Material)], Option<Material>) {
	let term_end = edits.partition_point(|(p, _)| p.depth() <= depth);
	let new_inherited = if term_end > 0 {
		Some(edits[term_end - 1].1)
	} else {
		prior
	};
	(&edits[term_end..], new_inherited)
}

impl<'a> Source for DiscreteSource<'a> {
	fn classify(&self, _lo: [i32; 3], _hi: [i32; 3], _depth: u8) -> Sample {
		if self.edits.is_empty() {
			match self.inherited {
				Some(m) => Sample::Fill(m),
				None => Sample::Passthrough,
			}
		} else {
			Sample::Subdivide
		}
	}

	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		let pos = ChunkPos::new(v[0] as u8, v[1] as u8, v[2] as u8);
		let target = u32::from(Path::from_coords(pos, 4));
		for &(p, m) in self.edits.iter().rev() {
			if u32::from(p) == target {
				return VoxelSample::Set(m);
			}
		}
		match self.inherited {
			Some(m) => VoxelSample::Set(m),
			None => VoxelSample::Passthrough,
		}
	}

	fn descend(&self, slot: u8) -> Self {
		let s_start = self.edits.partition_point(|(p, _)| p.slot_at(self.depth) < slot);
		let s_end = self.edits.partition_point(|(p, _)| p.slot_at(self.depth) <= slot);
		let slot_slice = &self.edits[s_start..s_end];
		let child_depth = self.depth + 1;
		let (deeper, inherited) = absorb_terminators(slot_slice, child_depth, self.inherited);
		Self { edits: deeper, depth: child_depth, inherited }
	}
}

#[derive(Clone, Copy)]
pub struct Overlay<B, T> {
	pub base: B,
	pub top: T,
}

impl<B, T> Overlay<B, T> {
	pub fn new(base: B, top: T) -> Self {
		Self { base, top }
	}
}

impl<B: Source, T: Source> Source for Overlay<B, T> {
	#[inline]
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], depth: u8) -> Sample {
		match self.top.classify(lo, hi, depth) {
			Sample::Passthrough => self.base.classify(lo, hi, depth),
			other => other,
		}
	}

	#[inline]
	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		match self.top.voxel(v) {
			VoxelSample::Passthrough => self.base.voxel(v),
			set => set,
		}
	}

	#[inline]
	fn descend(&self, slot: u8) -> Self {
		Self {
			base: self.base.descend(slot),
			top: self.top.descend(slot),
		}
	}
}
