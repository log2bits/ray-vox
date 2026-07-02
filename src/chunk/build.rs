use ahash::AHashMap;

use super::material::Material;
use super::node::{ChildMasks, InteriorNode, LeafNode, unpack_slot};
use super::{Child, Chunk};
use crate::util::PalettedVec;
use crate::util::types::{CHUNK_SIZE, Mask64};

#[derive(Copy, Clone, Debug)]
pub enum Sample {
	Passthrough,
	Fill(Material),
	Subdivide,
}

#[derive(Copy, Clone, Debug)]
pub enum VoxelSample {
	Passthrough,
	Fill(Material),
}

pub trait Source: Sized + Clone {
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], depth: u8) -> Sample;
	fn voxel(&self, v: [i32; 3]) -> VoxelSample;
	#[inline]
	fn descend(&self, _slot: u8) -> Self {
		self.clone()
	}
}

const INTERIOR_DEPTHS: u8 = 3;

#[derive(Clone)]
struct LeafData {
	occupancy: Mask64,
	materials: [Material; 64],
}

// Scratch tree built cell-by-cell from the Source. Nodes are indexed by push
// order; the Serializer walks it afterward to produce the final Chunk layout.
#[derive(Default)]
struct Arena {
	leaves: Vec<LeafData>,
	interiors: Vec<[Child; 64]>,
}

impl Arena {
	#[inline]
	fn push_leaf(&mut self, data: LeafData) -> Child {
		let id = self.leaves.len() as u32;
		self.leaves.push(data);
		Child::Leaf(id)
	}

	#[inline]
	fn push_interior(&mut self, children: [Child; 64]) -> Child {
		let id = self.interiors.len() as u32;
		self.interiors.push(children);
		Child::Interior(id)
	}
}

// Collapse 64 child cells to one Child. All same material -> Filled; mix of
// Filled/Empty only -> Leaf; anything nested -> Interior.
fn classify_children(arena: &mut Arena, children: [Child; 64]) -> Child {
	let mut has_any = false;
	let mut has_nested = false;
	let mut uniform_fill: Option<Material> = None;
	let mut all_same = true;
	for c in &children {
		match c {
			Child::Empty => {}
			Child::Filled(m) => {
				has_any = true;
				match uniform_fill {
					None => uniform_fill = Some(*m),
					Some(prev) if prev == *m => {}
					_ => all_same = false,
				}
			}
			Child::Leaf(_) | Child::Interior(_) => {
				has_any = true;
				has_nested = true;
				all_same = false;
			}
		}
	}
	if !has_any {
		return Child::Empty;
	}
	if !has_nested {
		if all_same && uniform_fill.is_some() && children.iter().all(|c| matches!(c, Child::Filled(_))) {
			return Child::Filled(uniform_fill.unwrap());
		}
		let mut occupancy = Mask64::EMPTY;
		let mut materials = [Material::air(); 64];
		for (slot, c) in children.iter().enumerate() {
			if let Child::Filled(m) = c {
				occupancy |= Mask64::bit(slot as u8);
				materials[slot] = *m;
			}
		}
		return arena.push_leaf(LeafData { occupancy, materials });
	}
	arena.push_interior(children)
}

fn build_cell<S: Source>(
	arena: &mut Arena,
	source: &S,
	lo: [i32; 3],
	side: i32,
	depth: u8,
) -> Child {
	let hi = [lo[0] + side, lo[1] + side, lo[2] + side];
	match source.classify(lo, hi, depth) {
		Sample::Passthrough => Child::Empty,
		Sample::Fill(m) if m.is_air() => Child::Empty,
		Sample::Fill(m) => Child::Filled(m),
		Sample::Subdivide => {
			if depth == INTERIOR_DEPTHS {
				build_leaf(arena, source, lo)
			} else {
				let child_side = side / 4;
				let mut children: [Child; 64] = [Child::Empty; 64];
				for slot in 0..64u8 {
					let [sx, sy, sz] = unpack_slot(slot);
					let child_lo = [
						lo[0] + sx * child_side,
						lo[1] + sy * child_side,
						lo[2] + sz * child_side,
					];
					let child_source = source.descend(slot);
					children[slot as usize] = build_cell(arena, &child_source, child_lo, child_side, depth + 1);
				}
				classify_children(arena, children)
			}
		}
	}
}

fn build_leaf<S: Source>(arena: &mut Arena, source: &S, lo: [i32; 3]) -> Child {
	let mut occupancy = Mask64::EMPTY;
	let mut materials = [Material::air(); 64];
	for x in 0..4i32 {
		for y in 0..4i32 {
			for z in 0..4i32 {
				let m = match source.voxel([lo[0] + x, lo[1] + y, lo[2] + z]) {
					VoxelSample::Passthrough => Material::air(),
					VoxelSample::Fill(m) => m,
				};
				if !m.is_air() {
					let slot = ((x as u8) << 4) | ((y as u8) << 2) | z as u8;
					occupancy |= Mask64::bit(slot);
					materials[slot as usize] = m;
				}
			}
		}
	}
	if occupancy.is_empty() {
		return Child::Empty;
	}
	if occupancy == Mask64::FULL {
		let first = materials[0];
		if materials.iter().all(|&m| m == first) {
			return Child::Filled(first);
		}
	}
	arena.push_leaf(LeafData { occupancy, materials })
}

// Lowers the arena tree into the packed on-disk Chunk layout. Each parent
// stores its interior and leaf children contiguously in the output arrays
// (indexed by popcount), so we stage each child aside until the parent copies
// it into place.
struct Serializer {
	staged_interiors: Vec<InteriorNode>,
	staged_leaves: Vec<LeafNode>,
	out_interiors: Vec<InteriorNode>,
	out_leaves: Vec<LeafNode>,
	out_materials: PalettedVec<Material>,
	raw_materials: Vec<Material>, // parallel view of out_materials for dedup compares
	run_index: AHashMap<u64, u32>,
}

impl Serializer {
	fn new() -> Self {
		Self {
			staged_interiors: Vec::new(),
			staged_leaves: Vec::new(),
			out_interiors: Vec::new(),
			out_leaves: Vec::new(),
			out_materials: PalettedVec::new(),
			raw_materials: Vec::new(),
			run_index: AHashMap::new(),
		}
	}

	// Emit a material run with two-tier dedup: exact-hash first, then
	// tail-overlap extend on miss. Collapses uniform regions to near zero.
	fn emit_material_run(&mut self, run: &[Material]) -> u32 {
		let full_hash = hash_run(run);
		if let Some(&offset) = self.run_index.get(&full_hash) {
			if self.run_matches_at(offset, run) {
				return offset;
			}
		}

		let array_len = self.raw_materials.len();
		let max_overlap = run.len().min(array_len);
		let mut overlap = 0;
		for i in (1..=max_overlap).rev() {
			if self.raw_materials[array_len - i..array_len] == run[..i] {
				overlap = i;
				break;
			}
		}

		let offset = (array_len - overlap) as u32;
		for &m in &run[overlap..] {
			self.out_materials.push(m);
			self.raw_materials.push(m);
		}

		self.run_index.entry(full_hash).or_insert(offset);
		offset
	}

	fn run_matches_at(&self, offset: u32, run: &[Material]) -> bool {
		let start = offset as usize;
		if start + run.len() > self.raw_materials.len() {
			return false;
		}
		&self.raw_materials[start..start + run.len()] == run
	}

	// Lower an arena cell into staged storage. Empty and Filled pass through
	// unchanged; Leaf and Interior return a Child pointing at the staged slot.
	fn lower(&mut self, arena: &Arena, cell: Child) -> Child {
		match cell {
			Child::Empty => Child::Empty,
			Child::Filled(m) => Child::Filled(m),
			Child::Leaf(id) => {
				let leaf = &arena.leaves[id as usize];
				let mut run = [Material::air(); 64];
				let mut run_len = 0;
				for slot in leaf.occupancy.iter_slots() {
					run[run_len] = leaf.materials[slot as usize];
					run_len += 1;
				}
				let material_offset = self.emit_material_run(&run[..run_len]);
				let mut node = LeafNode::default();
				node.occupancy = leaf.occupancy;
				node.set_material_offset(material_offset);
				let staged_index = self.staged_leaves.len() as u32;
				self.staged_leaves.push(node);
				Child::Leaf(staged_index)
			}
			Child::Interior(id) => {
				let children = arena.interiors[id as usize];
				let mut lowered: [Child; 64] = [Child::Empty; 64];
				let mut filled_materials = [Material::air(); 64];
				let mut masks = ChildMasks::default();
				for slot in 0..64u8 {
					let child = self.lower(arena, children[slot as usize]);
					if let Child::Filled(m) = child {
						filled_materials[slot as usize] = m;
					}
					lowered[slot as usize] = child;
					masks.set_state(slot, child.state());
				}

				// Copy this parent's children into out_ contiguously in slot
				// order so the popcount scheme on masks indexes into them.
				let interior_ptr = self.out_interiors.len() as u32;
				let leaf_ptr = self.out_leaves.len() as u32;
				for slot in masks.interiors().iter_slots() {
					if let Child::Interior(staged_index) = lowered[slot as usize] {
						self.out_interiors.push(self.staged_interiors[staged_index as usize]);
					}
				}
				for slot in masks.leaves().iter_slots() {
					if let Child::Leaf(staged_index) = lowered[slot as usize] {
						self.out_leaves.push(self.staged_leaves[staged_index as usize]);
					}
				}

				// Material run only carries filled cells. Interior/leaf slots
				// let the child node carry its own materials.
				let mut run = [Material::air(); 64];
				let mut run_len = 0;
				for slot in masks.filled().iter_slots() {
					run[run_len] = filled_materials[slot as usize];
					run_len += 1;
				}
				let material_offset = self.emit_material_run(&run[..run_len]);

				let mut node = InteriorNode::default();
				node.masks = masks;
				node.set_interior_offset(interior_ptr);
				node.set_leaf_offset(leaf_ptr);
				node.set_material_offset(material_offset);
				let staged_index = self.staged_interiors.len() as u32;
				self.staged_interiors.push(node);
				Child::Interior(staged_index)
			}
		}
	}
}

pub fn build_chunk<S: Source>(source: &S) -> Chunk {
	let mut arena = Arena::default();
	let root = build_cell(&mut arena, source, [0, 0, 0], CHUNK_SIZE, 0);

	// Empty and uniform chunks skip node storage; a uniform chunk keeps only
	// its single material at materials[0].
	if let Child::Empty = root {
		return Chunk::new();
	}
	if let Child::Filled(m) = root {
		let mut materials = PalettedVec::new();
		materials.push(m);
		materials.shrink_to_fit();
		return Chunk {
			leaf_nodes: Vec::new(),
			materials,
			interior_nodes: Vec::new(),
		};
	}

	let mut serializer = Serializer::new();
	let root_cell = serializer.lower(&arena, root);

	// The root has no parent to copy it into place; push it directly.
	match root_cell {
		Child::Interior(staged_index) => {
			serializer.out_interiors.push(serializer.staged_interiors[staged_index as usize]);
		}
		Child::Leaf(staged_index) => {
			serializer.out_leaves.push(serializer.staged_leaves[staged_index as usize]);
		}
		_ => {}
	}

	let mut materials = serializer.out_materials;
	materials.shrink_to_fit();
	Chunk {
		leaf_nodes: serializer.out_leaves,
		materials,
		interior_nodes: serializer.out_interiors,
	}
}

#[inline(always)]
fn fnv_start() -> u64 {
	0xcbf29ce484222325
}

#[inline(always)]
fn fnv_mix(hash: u64, value: u32) -> u64 {
	(hash ^ value as u64).wrapping_mul(0x100000001b3)
}

fn hash_run(run: &[Material]) -> u64 {
	let mut h = fnv_start();
	for &m in run {
		h = fnv_mix(h, u32::from(m));
	}
	h
}
