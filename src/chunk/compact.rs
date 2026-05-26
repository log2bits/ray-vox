use ahash::AHashMap;

use super::material::Material;
use super::node::{CellState, InteriorNode, InteriorNodeWide, LeafNode};
use super::{Chunk, MutableChunk};
use crate::util::PalettedVec;

/// Compacts and compresses a `MutableChunk` into a `Chunk`.
///
/// Removes orphan nodes left by path-copy edits, deduplicates identical subtrees
/// (DAG sharing), and packs interior node offsets into the 13/19-bit compressed layout.
pub(crate) fn compress(src: MutableChunk) -> Chunk {
	if src.interior_nodes.is_empty() {
		return Chunk {
			leaf_nodes: src.leaf_nodes,
			materials: src.materials,
			interior_nodes: Vec::new(),
		};
	}

	let root = src.interior_nodes.len() as u32 - 1;
	let (by_depth, reachable_leaf) = collect_reachable(&src.interior_nodes, root);

	let mut compressor =
		Compressor::new(&src.interior_nodes, &src.leaf_nodes, &src.materials);
	compressor.dedup_leaves(&reachable_leaf);
	for depth in (0..3).rev() {
		compressor.dedup_interior_at_depth(&by_depth[depth]);
	}
	compressor.push_root();

	Chunk {
		leaf_nodes: compressor.out_leaf,
		materials: compressor.out_mat,
		interior_nodes: compressor.out_interior,
	}
}

/// Working state for one compress pass.
///
/// Holds the source arrays (read-only), the remap tables from old indices to canonical pool
/// indices, the canonical pools themselves, and the output arrays where the final contiguous
/// child blocks are assembled.
struct Compressor<'a> {
	src_interior: &'a [InteriorNodeWide],
	src_leaf: &'a [LeafNode],
	src_mat: &'a PalettedVec<Material>,

	int_remap: Vec<u32>,
	leaf_remap: Vec<u32>,

	// Interior nodes that were demoted to leaf nodes (has_child == 0, only Filled/Empty slots).
	demoted: Vec<bool>,
	demoted_remap: Vec<u32>,

	canonical_interiors: Vec<InteriorNode>,
	canonical_leaves: Vec<LeafNode>,

	out_interior: Vec<InteriorNode>,
	out_leaf: Vec<LeafNode>,
	out_mat: PalettedVec<Material>,

	leaf_sig_map: AHashMap<u128, u32>,
	int_sig_map: AHashMap<u128, u32>,
}

impl<'a> Compressor<'a> {
	fn new(
		src_interior: &'a [InteriorNodeWide],
		src_leaf: &'a [LeafNode],
		src_mat: &'a PalettedVec<Material>,
	) -> Self {
		Self {
			src_interior,
			src_leaf,
			src_mat,
			int_remap: vec![0u32; src_interior.len()],
			leaf_remap: vec![0u32; src_leaf.len()],
			demoted: vec![false; src_interior.len()],
			demoted_remap: vec![0u32; src_interior.len()],
			canonical_interiors: Vec::new(),
			canonical_leaves: Vec::new(),
			out_interior: Vec::new(),
			out_leaf: Vec::new(),
			out_mat: PalettedVec::new(),
			leaf_sig_map: AHashMap::new(),
			int_sig_map: AHashMap::new(),
		}
	}

	fn dedup_leaves(&mut self, reachable: &[u32]) {
		for &old_idx in reachable {
			let node = self.src_leaf[old_idx as usize];
			let sig = self.leaf_sig(node);

			let canonical_idx = if let Some(&idx) = self.leaf_sig_map.get(&sig) {
				idx
			} else {
				let canonical_idx = self.canonical_leaves.len() as u32;
				let new_mat_offset = self.out_mat.len();
				let mut mask = node.occupancy();
				while mask != 0 {
					let slot = mask.trailing_zeros() as u8;
					self.out_mat
						.push(self.src_mat.get(node.material_index(slot)));
					mask &= mask - 1;
				}
				let mut out = LeafNode::default();
				out.set_occupancy(node.occupancy());
				out.set_material_offset(new_mat_offset);
				self.canonical_leaves.push(out);
				self.leaf_sig_map.insert(sig, canonical_idx);
				canonical_idx
			};
			self.leaf_remap[old_idx as usize] = canonical_idx;
		}
	}

	fn dedup_interior_at_depth(&mut self, reachable: &[u32]) {
		self.int_sig_map.clear();
		for &old_idx in reachable {
			let node = self.src_interior[old_idx as usize];

			// Interior node with no actual children (only Filled/Empty slots): demote to leaf.
			// occupancy() == is_leaf() because has_child == 0.
			if node.has_child() == 0 {
				let canonical = self.dedup_as_demoted_leaf(node);
				self.demoted[old_idx as usize] = true;
				self.demoted_remap[old_idx as usize] = canonical;
				continue;
			}

			let sig = self.interior_sig(node);
			let new_idx = if let Some(&idx) = self.int_sig_map.get(&sig) {
				idx
			} else {
				let new_idx = self.emit_interior_node(node);
				self.int_sig_map.insert(sig, new_idx);
				new_idx
			};
			self.int_remap[old_idx as usize] = new_idx;
		}
	}

	/// Converts a childless interior node (only Filled/Empty slots) into a canonical leaf node.
	fn dedup_as_demoted_leaf(&mut self, node: InteriorNodeWide) -> u32 {
		// Filled slots: has_child=0, is_leaf=1. So occupancy for the leaf = node.is_leaf().
		let occ = node.is_leaf();
		let mut hash = fnv_start();
		let mut mask = occ;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			hash = fnv_mix(hash, u32::from(self.src_mat.get(node.material_index(slot))));
			mask &= mask - 1;
		}
		let sig = ((occ as u128) << 64) | (hash as u128);

		if let Some(&idx) = self.leaf_sig_map.get(&sig) {
			return idx;
		}

		let canonical_idx = self.canonical_leaves.len() as u32;
		let new_mat_offset = self.out_mat.len();
		let mut mask = occ;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			self.out_mat.push(self.src_mat.get(node.material_index(slot)));
			mask &= mask - 1;
		}
		let mut out = LeafNode::default();
		out.set_occupancy(occ);
		out.set_material_offset(new_mat_offset);
		self.canonical_leaves.push(out);
		self.leaf_sig_map.insert(sig, canonical_idx);
		canonical_idx
	}

	/// Emits a canonical for `node`: appends its child blocks to the output arrays, packs offsets,
	/// and stores the resulting `InteriorNode` in the canonical pool. Returns the canonical's index.
	fn emit_interior_node(&mut self, node: InteriorNodeWide) -> u32 {
		let src_has_child = node.has_child();
		let src_is_leaf = node.is_leaf();

		// Build output masks: demoted Interior children become Leaf in the output.
		let out_has_child = src_has_child;
		let mut out_is_leaf = src_is_leaf;
		{
			let mut mask = src_has_child & !src_is_leaf;
			while mask != 0 {
				let slot = mask.trailing_zeros() as u8;
				let old_child = node.interior_child_index(slot);
				if self.demoted[old_child as usize] {
					out_is_leaf |= 1u64 << slot;
				}
				mask &= mask - 1;
			}
		}

		let n_interior = (out_has_child & !out_is_leaf).count_ones() as usize;
		let n_leaf = (out_has_child & out_is_leaf).count_ones() as usize;

		let new_interior_ptr = self.out_interior.len() as u32;
		let new_leaf_ptr = self.out_leaf.len() as u32;
		self.out_interior
			.resize_with(self.out_interior.len() + n_interior, InteriorNode::default);
		self.out_leaf
			.resize_with(self.out_leaf.len() + n_leaf, LeafNode::default);

		let new_mat_offset = self.out_mat.len();
		let mut int_rank = 0u32;
		let mut leaf_rank = 0u32;

		let mut mask = src_has_child | src_is_leaf;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			self.out_mat
				.push(self.src_mat.get(node.material_index(slot)));

			// Use output state (which may differ from source state for demoted children).
			let out_state = match ((out_has_child >> slot) & 1, (out_is_leaf >> slot) & 1) {
				(0, 1) => CellState::Filled,
				(1, 0) => CellState::Interior,
				(1, 1) => CellState::Leaf,
				_ => CellState::Empty,
			};

			match out_state {
				CellState::Interior => {
					let canonical = self.int_remap[node.interior_child_index(slot) as usize];
					self.out_interior[(new_interior_ptr + int_rank) as usize] =
						self.canonical_interiors[canonical as usize];
					int_rank += 1;
				}
				CellState::Leaf => {
					let canonical = match node.state(slot) {
						CellState::Leaf => {
							self.leaf_remap[node.leaf_child_index(slot) as usize]
						}
						CellState::Interior => {
							// Was demoted to leaf.
							self.demoted_remap[node.interior_child_index(slot) as usize]
						}
						_ => unreachable!(),
					};
					self.out_leaf[(new_leaf_ptr + leaf_rank) as usize] =
						self.canonical_leaves[canonical as usize];
					leaf_rank += 1;
				}
				_ => {}
			}
			mask &= mask - 1;
		}

		let new_idx = self.canonical_interiors.len() as u32;
		let mut out = InteriorNode::default();
		out.set_has_child(out_has_child);
		out.set_is_leaf(out_is_leaf);
		out.set_interior_offset(new_interior_ptr);
		out.set_leaf_offset(new_leaf_ptr);
		out.set_material_offset(new_mat_offset);
		self.canonical_interiors.push(out);

		new_idx
	}

	/// Push the root canonical onto `out_interior` so it sits at `out_interior.len() - 1`,
	/// where the shader expects it.
	fn push_root(&mut self) {
		if let Some(&root) = self.canonical_interiors.last() {
			self.out_interior.push(root);
		}
	}

	fn leaf_sig(&self, node: LeafNode) -> u128 {
		let occ = node.occupancy();
		let mut hash = fnv_start();
		let mut mask = occ;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			hash = fnv_mix(hash, u32::from(self.src_mat.get(node.material_index(slot))));
			mask &= mask - 1;
		}
		((occ as u128) << 64) | (hash as u128)
	}

	fn interior_sig(&self, node: InteriorNodeWide) -> u128 {
		let src_has_child = node.has_child();
		let src_is_leaf = node.is_leaf();

		// Reflect demotions in the signature so parents with identical demoted-child
		// patterns still deduplicate correctly.
		let out_has_child = src_has_child;
		let mut out_is_leaf = src_is_leaf;
		{
			let mut mask = src_has_child & !src_is_leaf;
			while mask != 0 {
				let slot = mask.trailing_zeros() as u8;
				let old_child = node.interior_child_index(slot);
				if self.demoted[old_child as usize] {
					out_is_leaf |= 1u64 << slot;
				}
				mask &= mask - 1;
			}
		}

		let masks =
			(out_has_child ^ out_is_leaf.rotate_left(32)) ^ (out_has_child >> 32 | out_is_leaf << 32);

		let mut hash = fnv_start();
		let mut mask = src_has_child | src_is_leaf;
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			hash = fnv_mix(hash, u32::from(self.src_mat.get(node.material_index(slot))));

			let child_ref = match node.state(slot) {
				CellState::Interior => {
					let old_child = node.interior_child_index(slot);
					if self.demoted[old_child as usize] {
						self.demoted_remap[old_child as usize]
					} else {
						self.int_remap[old_child as usize]
					}
				}
				CellState::Leaf => self.leaf_remap[node.leaf_child_index(slot) as usize],
				_ => 0,
			};
			hash = fnv_mix(hash, child_ref);
			mask &= mask - 1;
		}
		((masks as u128) << 64) | (hash as u128)
	}
}

fn collect_reachable(interior: &[InteriorNodeWide], root: u32) -> ([Vec<u32>; 3], Vec<u32>) {
	let mut by_depth: [Vec<u32>; 3] = Default::default();
	let mut reachable_leaf: Vec<u32> = Vec::new();
	by_depth[0].push(root);

	for depth in 0..3usize {
		let mut i = 0;
		while i < by_depth[depth].len() {
			let node = interior[by_depth[depth][i] as usize];
			i += 1;

			if depth < 2 {
				let mut mask = node.has_child() & !node.is_leaf();
				let base = node.interior_offset();
				let mut rank = 0u32;
				while mask != 0 {
					by_depth[depth + 1].push(base + rank);
					rank += 1;
					mask &= mask - 1;
				}
			}

			let mut mask = node.has_child() & node.is_leaf();
			let base = node.leaf_offset();
			let mut rank = 0u32;
			while mask != 0 {
				reachable_leaf.push(base + rank);
				rank += 1;
				mask &= mask - 1;
			}
		}
		by_depth[depth].sort_unstable();
		by_depth[depth].dedup();
	}

	reachable_leaf.sort_unstable();
	reachable_leaf.dedup();
	(by_depth, reachable_leaf)
}

#[inline(always)]
fn fnv_start() -> u64 {
	0xcbf29ce484222325
}

#[inline(always)]
fn fnv_mix(hash: u64, value: u32) -> u64 {
	(hash ^ value as u64).wrapping_mul(0x100000001b3)
}
