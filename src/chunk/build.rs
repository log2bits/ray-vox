use ahash::AHashMap;

use super::material::Material;
use super::node::{CellState, ChildMasks, InteriorNode, LeafNode};
use super::Chunk;
use crate::util::PalettedVec;
use crate::util::types::Mask64;

#[derive(Copy, Clone, Debug)]
pub enum Sample {
	Passthrough,
	Fill(Material),
	Subdivide,
}

#[derive(Copy, Clone, Debug)]
pub enum VoxelSample {
	Passthrough,
	Set(Material),
}

pub trait Source: Sized + Clone {
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], depth: u8) -> Sample;
	fn voxel(&self, v: [i32; 3]) -> VoxelSample;
	#[inline]
	fn descend(&self, _slot: u8) -> Self {
		self.clone()
	}
}

pub const CHUNK_SIDE: i32 = 256;
const INTERIOR_DEPTHS: u8 = 3;

#[derive(Clone)]
struct LeafData {
	occupancy: Mask64,
	materials: [Material; 64],
}

#[derive(Copy, Clone)]
enum Cell {
	Empty,
	Filled(Material),
	Leaf(u32),
	Interior(u32),
}

#[derive(Default)]
struct Arena {
	leaves: Vec<LeafData>,
	interiors: Vec<[Cell; 64]>,
}

impl Arena {
	#[inline]
	fn push_leaf(&mut self, data: LeafData) -> Cell {
		let id = self.leaves.len() as u32;
		self.leaves.push(data);
		Cell::Leaf(id)
	}

	#[inline]
	fn push_interior(&mut self, children: [Cell; 64]) -> Cell {
		let id = self.interiors.len() as u32;
		self.interiors.push(children);
		Cell::Interior(id)
	}
}

#[inline]
fn slot_xyz(slot: u8) -> (i32, i32, i32) {
	(((slot >> 4) & 3) as i32, ((slot >> 2) & 3) as i32, (slot & 3) as i32)
}

fn classify_children(arena: &mut Arena, children: [Cell; 64]) -> Cell {
	let mut nonempty = false;
	let mut any_nested = false;
	let mut uniform_fill: Option<Material> = None;
	let mut all_same = true;
	for c in &children {
		match c {
			Cell::Empty => {}
			Cell::Filled(m) => {
				nonempty = true;
				match uniform_fill {
					None => uniform_fill = Some(*m),
					Some(prev) if prev == *m => {}
					_ => all_same = false,
				}
			}
			Cell::Leaf(_) | Cell::Interior(_) => {
				nonempty = true;
				any_nested = true;
				all_same = false;
			}
		}
	}
	if !nonempty {
		return Cell::Empty;
	}
	if !any_nested {
		if all_same && uniform_fill.is_some() && children.iter().all(|c| matches!(c, Cell::Filled(_))) {
			return Cell::Filled(uniform_fill.unwrap());
		}
		let mut occ = Mask64::EMPTY;
		let mut mats = [Material::air(); 64];
		for (i, c) in children.iter().enumerate() {
			if let Cell::Filled(m) = c {
				occ |= Mask64::bit(i as u8);
				mats[i] = *m;
			}
		}
		return arena.push_leaf(LeafData { occupancy: occ, materials: mats });
	}
	arena.push_interior(children)
}

fn build_cell<S: Source>(
	arena: &mut Arena,
	source: &S,
	lo: [i32; 3],
	side: i32,
	depth: u8,
) -> Cell {
	let hi = [lo[0] + side, lo[1] + side, lo[2] + side];
	match source.classify(lo, hi, depth) {
		Sample::Passthrough => Cell::Empty,
		Sample::Fill(m) if m.is_air() => Cell::Empty,
		Sample::Fill(m) => Cell::Filled(m),
		Sample::Subdivide => {
			if depth == INTERIOR_DEPTHS {
				build_leaf(arena, source, lo)
			} else {
				let child_side = side / 4;
				let mut children: [Cell; 64] = [Cell::Empty; 64];
				for slot in 0..64u8 {
					let (sx, sy, sz) = slot_xyz(slot);
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

fn build_leaf<S: Source>(arena: &mut Arena, source: &S, lo: [i32; 3]) -> Cell {
	let mut occ = Mask64::EMPTY;
	let mut mats = [Material::air(); 64];
	for x in 0..4i32 {
		for y in 0..4i32 {
			for z in 0..4i32 {
				let m = match source.voxel([lo[0] + x, lo[1] + y, lo[2] + z]) {
					VoxelSample::Passthrough => Material::air(),
					VoxelSample::Set(m) => m,
				};
				if !m.is_air() {
					let slot = ((x as u8) << 4) | ((y as u8) << 2) | z as u8;
					occ |= Mask64::bit(slot);
					mats[slot as usize] = m;
				}
			}
		}
	}
	if occ.is_empty() {
		return Cell::Empty;
	}
	if occ == Mask64::FULL {
		let first = mats[0];
		if mats.iter().all(|&m| m == first) {
			return Cell::Filled(first);
		}
	}
	arena.push_leaf(LeafData { occupancy: occ, materials: mats })
}

pub fn mode_over(occupancy: Mask64, mats: &[Material; 64]) -> Material {
	let mut keys = [Material::air(); 64];
	let mut counts = [0u32; 64];
	let mut n = 0usize;
	let mut best = (Material::air(), 0u32);
	for slot in occupancy.iter_slots() {
		let m = mats[slot as usize];
		let j = match keys[..n].iter().position(|&k| k == m) {
			Some(j) => j,
			None => {
				keys[n] = m;
				counts[n] = 0;
				let j = n;
				n += 1;
				j
			}
		};
		counts[j] += 1;
		if counts[j] > best.1 {
			best = (m, counts[j]);
		}
	}
	best.0
}

struct Serializer {
	canonical_interiors: Vec<InteriorNode>,
	canonical_leaves: Vec<LeafNode>,
	out_interiors: Vec<InteriorNode>,
	out_leaves: Vec<LeafNode>,
	out_materials: PalettedVec<Material>,
	run_index: AHashMap<u64, u32>,
	leaf_sig: AHashMap<u64, u32>,
	int_sig: AHashMap<u64, u32>,
}

impl Serializer {
	fn new() -> Self {
		Self {
			canonical_interiors: Vec::new(),
			canonical_leaves: Vec::new(),
			out_interiors: Vec::new(),
			out_leaves: Vec::new(),
			out_materials: PalettedVec::new(),
			run_index: AHashMap::new(),
			leaf_sig: AHashMap::new(),
			int_sig: AHashMap::new(),
		}
	}

	fn emit_material_run(&mut self, run: &[Material]) -> u32 {
		let hash = hash_run(run);
		if let Some(&offset) = self.run_index.get(&hash) {
			if self.run_matches_at(offset, run) {
				return offset;
			}
		}
		let offset = self.out_materials.len();
		for &m in run {
			self.out_materials.push(m);
		}
		self.run_index.entry(hash).or_insert(offset);
		offset
	}

	fn run_matches_at(&self, offset: u32, run: &[Material]) -> bool {
		if offset + run.len() as u32 > self.out_materials.len() {
			return false;
		}
		run.iter()
			.enumerate()
			.all(|(i, &m)| self.out_materials.get(offset + i as u32) == m)
	}

	fn leaf_representative(&self, canonical_idx: u32) -> Material {
		let leaf = self.canonical_leaves[canonical_idx as usize];
		let mut mats = [Material::air(); 64];
		for slot in leaf.occupancy.iter_slots() {
			mats[slot as usize] = self.out_materials.get(leaf.material_index(slot));
		}
		mode_over(leaf.occupancy, &mats)
	}

	fn intern_leaf(&mut self, leaf: &LeafData) -> u32 {
		let mut hash = fnv_start();
		hash = fnv_mix(hash, fold_mask(leaf.occupancy));
		let mut run = [Material::air(); 64];
		let mut len = 0;
		for slot in leaf.occupancy.iter_slots() {
			run[len] = leaf.materials[slot as usize];
			hash = fnv_mix(hash, run[len].into());
			len += 1;
		}
		if let Some(&cand) = self.leaf_sig.get(&hash) {
			let c = self.canonical_leaves[cand as usize];
			if c.occupancy == leaf.occupancy {
				let ok = leaf.occupancy.iter_slots().all(|s| {
					self.out_materials.get(c.material_index(s)) == leaf.materials[s as usize]
				});
				if ok {
					return cand;
				}
			}
		}
		let mat_offset = self.emit_material_run(&run[..len]);
		let mut node = LeafNode::default();
		node.occupancy = leaf.occupancy;
		node.set_material_offset(mat_offset);
		let new_idx = self.canonical_leaves.len() as u32;
		self.canonical_leaves.push(node);
		self.leaf_sig.entry(hash).or_insert(new_idx);
		new_idx
	}

	fn lower(&mut self, arena: &Arena, cell: Cell) -> (Cell, Material) {
		match cell {
			Cell::Empty => (Cell::Empty, Material::air()),
			Cell::Filled(m) => (Cell::Filled(m), m),
			Cell::Leaf(id) => {
				let leaf = arena.leaves[id as usize].clone();
				let canonical = self.intern_leaf(&leaf);
				(Cell::Leaf(canonical), self.leaf_representative(canonical))
			}
			Cell::Interior(id) => {
				let children = arena.interiors[id as usize];
				let mut lowered: [Cell; 64] = [Cell::Empty; 64];
				let mut lods = [Material::air(); 64];
				let mut masks = ChildMasks::default();
				for slot in 0..64u8 {
					let (c, lod) = self.lower(arena, children[slot as usize]);
					lowered[slot as usize] = c;
					lods[slot as usize] = lod;
					let state = match c {
						Cell::Empty => CellState::Empty,
						Cell::Filled(_) => CellState::Filled,
						Cell::Interior(_) => CellState::Interior,
						Cell::Leaf(_) => CellState::Leaf,
					};
					masks.set_state(slot, state);
				}

				let mut hash = fnv_start();
				hash = fnv_mix(hash, fold_mask(masks.has_child));
				hash = fnv_mix(hash, fold_mask(masks.is_leaf));
				for slot in masks.occupancy().iter_slots() {
					hash = fnv_mix(hash, lods[slot as usize].into());
					let canon = match lowered[slot as usize] {
						Cell::Interior(c) | Cell::Leaf(c) => c,
						_ => 0,
					};
					hash = fnv_mix(hash, canon);
				}
				if let Some(&cand) = self.int_sig.get(&hash) {
					if self.interior_eq(cand, masks, &lods, &lowered) {
						return (Cell::Interior(cand), mode_over(masks.occupancy(), &lods));
					}
				}

				let interior_ptr = self.out_interiors.len() as u32;
				let leaf_ptr = self.out_leaves.len() as u32;
				for slot in masks.interiors().iter_slots() {
					if let Cell::Interior(c) = lowered[slot as usize] {
						let n = self.canonical_interiors[c as usize];
						self.out_interiors.push(n);
					}
				}
				for slot in masks.leaves().iter_slots() {
					if let Cell::Leaf(c) = lowered[slot as usize] {
						let n = self.canonical_leaves[c as usize];
						self.out_leaves.push(n);
					}
				}
				let mut run = [Material::air(); 64];
				let mut run_len = 0;
				for slot in masks.occupancy().iter_slots() {
					run[run_len] = lods[slot as usize];
					run_len += 1;
				}
				let mat_offset = self.emit_material_run(&run[..run_len]);
				let mut out = InteriorNode::default();
				out.masks = masks;
				out.set_interior_offset(interior_ptr);
				out.set_leaf_offset(leaf_ptr);
				out.set_material_offset(mat_offset);
				let new_idx = self.canonical_interiors.len() as u32;
				self.canonical_interiors.push(out);
				self.int_sig.entry(hash).or_insert(new_idx);
				(Cell::Interior(new_idx), mode_over(masks.occupancy(), &lods))
			}
		}
	}

	fn interior_eq(
		&self,
		cand: u32,
		masks: ChildMasks,
		lods: &[Material; 64],
		lowered: &[Cell; 64],
	) -> bool {
		let c = self.canonical_interiors[cand as usize];
		if c.masks.has_child != masks.has_child || c.masks.is_leaf != masks.is_leaf {
			return false;
		}
		for slot in masks.occupancy().iter_slots() {
			if self.out_materials.get(c.material_index(slot)) != lods[slot as usize] {
				return false;
			}
		}
		let mut interior_rank = 0u32;
		let int_base = c.interior_offset();
		for slot in masks.interiors().iter_slots() {
			let want = match lowered[slot as usize] {
				Cell::Interior(want) => want,
				_ => return false,
			};
			let got = &self.out_interiors[(int_base + interior_rank) as usize];
			let want_node = &self.canonical_interiors[want as usize];
			if got.masks.has_child != want_node.masks.has_child
				|| got.masks.is_leaf != want_node.masks.is_leaf
				|| got.material_offset() != want_node.material_offset()
				|| got.interior_offset() != want_node.interior_offset()
				|| got.leaf_offset() != want_node.leaf_offset()
			{
				return false;
			}
			interior_rank += 1;
		}
		let mut leaf_rank = 0u32;
		let leaf_base = c.leaf_offset();
		for slot in masks.leaves().iter_slots() {
			let want = match lowered[slot as usize] {
				Cell::Leaf(want) => want,
				_ => return false,
			};
			let got = &self.out_leaves[(leaf_base + leaf_rank) as usize];
			let want_node = &self.canonical_leaves[want as usize];
			if got.occupancy != want_node.occupancy
				|| got.material_offset() != want_node.material_offset()
			{
				return false;
			}
			leaf_rank += 1;
		}
		true
	}
}

pub fn build_chunk<S: Source>(source: &S) -> Chunk {
	let mut arena = Arena::default();
	let root = build_cell(&mut arena, source, [0, 0, 0], CHUNK_SIDE, 0);

	let mut ser = Serializer::new();
	match root {
		Cell::Empty => return Chunk::new(),
		Cell::Filled(m) => {
			ser.out_materials.push(m);
			let mut materials = ser.out_materials;
			materials.shrink_to_fit();
			return Chunk { leaf_nodes: Vec::new(), materials, interior_nodes: Vec::new() };
		}
		_ => {}
	}

	ser.out_materials.push(Material::air());
	let (root_cell, chunk_lod) = ser.lower(&arena, root);

	let mut head = Vec::with_capacity(ser.out_materials.len() as usize);
	head.push(chunk_lod);
	for i in 1..ser.out_materials.len() {
		head.push(ser.out_materials.get(i));
	}
	ser.out_materials.clear();
	for m in head {
		ser.out_materials.push(m);
	}

	match root_cell {
		Cell::Interior(c) => {
			let n = ser.canonical_interiors[c as usize];
			ser.out_interiors.push(n);
		}
		Cell::Leaf(c) => {
			let n = ser.canonical_leaves[c as usize];
			ser.out_leaves.push(n);
		}
		_ => {}
	}

	let mut materials = ser.out_materials;
	materials.shrink_to_fit();
	Chunk {
		leaf_nodes: ser.out_leaves,
		materials,
		interior_nodes: ser.out_interiors,
	}
}

#[inline]
fn fold_mask(m: Mask64) -> u32 {
	let v = m.raw();
	(v as u32) ^ ((v >> 32) as u32)
}

#[inline(always)]
fn fnv_start() -> u64 { 0xcbf29ce484222325 }

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
