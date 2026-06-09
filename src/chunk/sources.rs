use super::build::{Sample, Source, VoxelSample};
use super::edit::Path;
use super::material::Material;
use super::{Child, Chunk};
use crate::util::types::ChunkPos;

pub trait LocalEdit: Send {
	fn bounds_local(&self) -> [[i32; 3]; 2];
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], depth: u8) -> Sample;
	fn voxel(&self, v: [i32; 3]) -> VoxelSample;
}

pub struct CompositeSource<'a> {
	edits: &'a [Box<dyn LocalEdit + 'a>],
	active: Vec<u16>,
	cell_lo: [i32; 3],
	cell_side: i32,
}

impl<'a> CompositeSource<'a> {
	pub fn new(edits: &'a [Box<dyn LocalEdit + 'a>], chunk_side: i32) -> Self {
		let active: Vec<u16> = (0..edits.len() as u16).collect();
		Self {
			edits,
			active,
			cell_lo: [0, 0, 0],
			cell_side: chunk_side,
		}
	}
}

impl<'a> Clone for CompositeSource<'a> {
	fn clone(&self) -> Self {
		Self {
			edits: self.edits,
			active: self.active.clone(),
			cell_lo: self.cell_lo,
			cell_side: self.cell_side,
		}
	}
}

impl<'a> Source for CompositeSource<'a> {
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], depth: u8) -> Sample {
		for &i in self.active.iter().rev() {
			match self.edits[i as usize].classify(lo, hi, depth) {
				Sample::Passthrough => continue,
				other => return other,
			}
		}
		Sample::Passthrough
	}

	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		for &i in self.active.iter().rev() {
			let [b_lo, b_hi] = self.edits[i as usize].bounds_local();
			if !point_in_box(v, b_lo, b_hi) {
				continue;
			}
			match self.edits[i as usize].voxel(v) {
				VoxelSample::Passthrough => continue,
				set => return set,
			}
		}
		VoxelSample::Passthrough
	}

	fn descend(&self, slot: u8) -> Self {
		let child_side = self.cell_side / 4;
		let sx = ((slot >> 4) & 3) as i32;
		let sy = ((slot >> 2) & 3) as i32;
		let sz = (slot & 3) as i32;
		let child_lo = [
			self.cell_lo[0] + sx * child_side,
			self.cell_lo[1] + sy * child_side,
			self.cell_lo[2] + sz * child_side,
		];
		let child_hi = [
			child_lo[0] + child_side,
			child_lo[1] + child_side,
			child_lo[2] + child_side,
		];
		let mut active = Vec::with_capacity(self.active.len());
		for &i in &self.active {
			let [b_lo, b_hi] = self.edits[i as usize].bounds_local();
			if box_intersects(b_lo, b_hi, child_lo, child_hi) {
				active.push(i);
			}
		}
		Self {
			edits: self.edits,
			active,
			cell_lo: child_lo,
			cell_side: child_side,
		}
	}
}

#[inline]
fn box_intersects(a_lo: [i32; 3], a_hi: [i32; 3], b_lo: [i32; 3], b_hi: [i32; 3]) -> bool {
	a_hi[0] > b_lo[0] && b_hi[0] > a_lo[0]
		&& a_hi[1] > b_lo[1] && b_hi[1] > a_lo[1]
		&& a_hi[2] > b_lo[2] && b_hi[2] > a_lo[2]
}

#[inline]
fn point_in_box(p: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> bool {
	p[0] >= lo[0] && p[0] < hi[0]
		&& p[1] >= lo[1] && p[1] < hi[1]
		&& p[2] >= lo[2] && p[2] < hi[2]
}

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
