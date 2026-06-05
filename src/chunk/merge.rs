//! Build a coarser chunk from a 4x4x4 block of finer chunks.
//!
//! Each fine chunk fills one slot of the coarse root. Coarsening stops at fine
//! depth 2 (one level above the fine voxels) and turns that node into a coarse
//! leaf. Fine voxel leaves are never read because the cached representatives in
//! the parent slab already hold what the coarse cell needs.

use super::rebuild::{ChildSet, RebuildResult, VoxelSet};
use super::{Child, Chunk, MutableChunk};

/// Merge 64 finer chunks into one coarser chunk. Slot index matches the tree's
/// bit-interleave so the array can come straight from ChunkId::children.
pub fn merge_lod(children: [Option<&Chunk>; 64]) -> Chunk {
	let mut out = MutableChunk::empty();
	let mut root = ChildSet::default();

	for (slot, child) in children.iter().enumerate() {
		let outcome = match child {
			None => RebuildResult::Empty,
			Some(c) if c.is_empty() => RebuildResult::Empty,
			Some(c) if c.is_uniform() => RebuildResult::Filled(c.materials.get(0)),
			Some(c) => coarsen(c, c.root_idx(), 0, &mut out),
		};
		root.record(slot as u8, outcome);
	}

	match root.finish(&mut out) {
		RebuildResult::Empty => {}
		RebuildResult::Filled(mat) => {
			out.materials.clear();
			out.materials.push(mat);
		}
		RebuildResult::Interior(_, new_root) => {
			out.interior_nodes.push(new_root);
		}
		RebuildResult::Leaf(..) => unreachable!("merge root cannot be a single leaf"),
	}
	out.bake()
}

/// Coarsen the subtree at fine_idx into a RebuildResult for its slot in the coarse
/// tree. At fine depth 2 the subtree collapses into a coarse leaf instead of recursing.
fn coarsen(fine: &Chunk, fine_idx: u32, fine_depth: u8, out: &mut MutableChunk) -> RebuildResult {
	if fine_depth == 2 {
		return clip_to_leaf(fine, fine_idx, out);
	}

	let mut set = ChildSet::default();
	let occ = fine.interior_nodes[fine_idx as usize].masks.occupancy();
	for slot in occ.iter_slots() {
		let outcome = match fine.child(fine_idx, slot) {
			Child::Empty => RebuildResult::Empty,
			Child::Filled(m) => RebuildResult::Filled(m),
			Child::Interior(child_idx) => coarsen(fine, child_idx, fine_depth + 1, out),
			// A leaf above fine depth 2 is a demoted interior. Its cells are
			// already at or above coarse-voxel size, so copy it verbatim.
			Child::Leaf(leaf_idx) => copy_leaf(fine, leaf_idx, out),
		};
		set.record(slot, outcome);
	}
	set.finish(out)
}

/// Collapse a fine depth-2 interior into a coarse leaf. The fine slab already holds
/// per-slot representatives, so push them through a VoxelSet for collapse and mode pick.
fn clip_to_leaf(fine: &Chunk, fine_idx: u32, out: &mut MutableChunk) -> RebuildResult {
	let n = &fine.interior_nodes[fine_idx as usize];
	let mut set = VoxelSet::default();
	for slot in n.masks.occupancy().iter_slots() {
		let m = fine.materials.get(n.material_index(slot));
		set.record(slot, m);
	}
	set.finish(out)
}

/// Copy a fine leaf into the coarse arrays. Materials are re-interned into out.materials.
fn copy_leaf(fine: &Chunk, leaf_idx: u32, out: &mut MutableChunk) -> RebuildResult {
	let leaf = fine.leaf_nodes[leaf_idx as usize];
	let mut set = VoxelSet::default();
	for slot in leaf.occupancy.iter_slots() {
		let m = fine.materials.get(leaf.material_index(slot));
		set.record(slot, m);
	}
	set.finish(out)
}

