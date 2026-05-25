use ahash::AHashMap;

use super::material::Material;
use super::node::{CellState, InteriorNode, LeafNode};
use super::Chunk;
use crate::util::PalettedVec;

impl Chunk {
	/// Compacts the chunk by removing garbage nodes left by the path-copy edit pass,
	/// then deduplicates identical subtrees (DAG sharing).
	///
	/// Interior nodes with no pointer children (only filled/empty slots) are demoted
	/// to leaf nodes, saving 12 bytes each and enabling cross-type deduplication.
	///
	/// After this call the arrays are fully packed, the root is the last element of
	/// `interior_nodes` (or the sole element of `leaf_nodes` if the root was demoted),
	/// and every structurally-identical subtree is stored once.
	pub fn compact(&mut self) {
		if self.interior_nodes.is_empty() {
			return;
		}

		let root = self.interior_nodes.len() as u32 - 1;
		let (by_depth, reachable_leaf) = collect_reachable(&self.interior_nodes, root);

		let mut new_interior: Vec<InteriorNode> = Vec::new();
		let mut new_leaf: Vec<LeafNode> = Vec::new();
		let mut new_mat: PalettedVec<Material> = PalettedVec::new();

		// Shared leaf dedup map: used for both original leaves and demoted interior nodes
		// so equal content always maps to the same new leaf index.
		let mut leaf_sig_map: AHashMap<u128, u32> = AHashMap::new();

		let leaf_remap = dedup_leaves(
			&self.leaf_nodes,
			&self.materials,
			&reachable_leaf,
			&mut new_leaf,
			&mut new_mat,
			&mut leaf_sig_map,
		);

		// int_remap[i] = new index in new_interior (or new_leaf if int_is_demoted[i])
		let mut int_remap = vec![0u32; self.interior_nodes.len()];
		let mut int_is_demoted = vec![false; self.interior_nodes.len()];
		let mut int_sig_map: AHashMap<u128, u32> = AHashMap::new();

		// Process bottom-up (depth 2 → 0) so children are always remapped before parents.
		for depth in (0..3usize).rev() {
			for &old_idx in &by_depth[depth] {
				let node = self.interior_nodes[old_idx as usize];

				if node.has_child() == 0 {
					// No pointer children — demote to leaf. Dedup against original leaves
					// via the shared map so equal content always shares one node.
					let sig = childless_leaf_sig(node, &self.materials);
					let new_idx = if let Some(&idx) = leaf_sig_map.get(&sig) {
						idx
					} else {
						let new_idx = new_leaf.len() as u32;
						let new_mat_offset = new_mat.len();
						let occ = node.is_leaf(); // has_child == 0 → occupancy == is_leaf
						let mut mask = occ;
						while mask != 0 {
							let s = mask.trailing_zeros() as u8;
							new_mat.push(self.materials.get(node.material_index(s)));
							mask &= mask - 1;
						}
						let mut out = LeafNode::default();
						out.set_occupancy(occ);
						out.set_material_offset(new_mat_offset);
						new_leaf.push(out);
						leaf_sig_map.insert(sig, new_idx);
						new_idx
					};
					int_remap[old_idx as usize] = new_idx;
					int_is_demoted[old_idx as usize] = true;
				} else {
					let sig = interior_sig(
						node,
						&self.materials,
						&int_remap,
						&int_is_demoted,
						&leaf_remap,
					);
					let new_idx = if let Some(&idx) = int_sig_map.get(&sig) {
						idx
					} else {
						let new_idx = emit_interior_node(
							node,
							&self.materials,
							&int_remap,
							&int_is_demoted,
							&leaf_remap,
							&mut new_interior,
							&mut new_leaf,
							&mut new_mat,
						);
						int_sig_map.insert(sig, new_idx);
						new_idx
					};
					int_remap[old_idx as usize] = new_idx;
				}
			}
		}

		self.interior_nodes = new_interior;
		self.leaf_nodes = new_leaf;
		self.materials = new_mat;
	}
}

// --- Reachability ---

/// BFS from `root`, partitioning reachable interior nodes by tree depth (0–2)
/// and collecting all reachable leaf node indices.
fn collect_reachable(
	interior: &[InteriorNode],
	root: u32,
) -> ([Vec<u32>; 3], Vec<u32>) {
	let mut by_depth: [Vec<u32>; 3] = Default::default();
	let mut reachable_leaf: Vec<u32> = Vec::new();

	by_depth[0].push(root);

	for depth in 0..3usize {
		let mut i = 0;
		// Iterate by index so we can push into by_depth[depth+1] while reading by_depth[depth].
		while i < by_depth[depth].len() {
			let node_idx = by_depth[depth][i];
			i += 1;

			let node = interior[node_idx as usize];
			let has_child = node.has_child();
			let is_leaf = node.is_leaf();

			if depth < 2 {
				let mut mask = has_child & !is_leaf;
				let base = node.interior_offset();
				let mut rank = 0u32;
				while mask != 0 {
					by_depth[depth + 1].push(base + rank);
					rank += 1;
					mask &= mask - 1;
				}
			}

			let mut mask = has_child & is_leaf;
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

// --- Leaf dedup ---

fn dedup_leaves(
	old_leaves: &[LeafNode],
	old_mat: &PalettedVec<Material>,
	reachable: &[u32],
	new_leaves: &mut Vec<LeafNode>,
	new_mat: &mut PalettedVec<Material>,
	sig_to_new: &mut AHashMap<u128, u32>,
) -> Vec<u32> {
	let mut remap = vec![0u32; old_leaves.len()];

	for &old_idx in reachable {
		let node = old_leaves[old_idx as usize];
		let sig = leaf_sig(node, old_mat);

		let new_idx = if let Some(&idx) = sig_to_new.get(&sig) {
			idx
		} else {
			let new_idx = new_leaves.len() as u32;
			let new_mat_offset = new_mat.len();
			let mut mask = node.occupancy();
			while mask != 0 {
				let slot = mask.trailing_zeros() as u8;
				new_mat.push(old_mat.get(node.material_index(slot)));
				mask &= mask - 1;
			}
			let mut out = LeafNode::default();
			out.set_occupancy(node.occupancy());
			out.set_material_offset(new_mat_offset);
			new_leaves.push(out);
			sig_to_new.insert(sig, new_idx);
			new_idx
		};

		remap[old_idx as usize] = new_idx;
	}

	remap
}

// --- Interior emit ---

/// Appends a single interior node (and a fresh contiguous block of its children)
/// to the output arrays, accounting for demoted children. Returns the new node's
/// index in `new_interior`.
fn emit_interior_node(
	node: InteriorNode,
	old_mat: &PalettedVec<Material>,
	int_remap: &[u32],
	int_is_demoted: &[bool],
	leaf_remap: &[u32],
	new_interior: &mut Vec<InteriorNode>,
	new_leaf: &mut Vec<LeafNode>,
	new_mat: &mut PalettedVec<Material>,
) -> u32 {
	let has_child = node.has_child();
	let is_leaf = node.is_leaf();

	// Compute effective is_leaf mask: interior children that were demoted become leaf children.
	let mut eff_is_leaf = is_leaf;
	{
		let mut mask = has_child & !is_leaf;
		while mask != 0 {
			let s = mask.trailing_zeros() as u8;
			if int_is_demoted[node.interior_child_index(s) as usize] {
				eff_is_leaf |= 1u64 << s;
			}
			mask &= mask - 1;
		}
	}
	let eff_interior_mask = has_child & !eff_is_leaf;
	let eff_leaf_mask = has_child & eff_is_leaf;

	let n_interior = eff_interior_mask.count_ones() as usize;
	let n_leaf = eff_leaf_mask.count_ones() as usize;

	let new_interior_ptr = new_interior.len() as u32;
	let new_leaf_ptr = new_leaf.len() as u32;
	new_interior.resize_with(new_interior.len() + n_interior, InteriorNode::default);
	new_leaf.resize_with(new_leaf.len() + n_leaf, LeafNode::default);

	let new_mat_offset = new_mat.len();
	let mut int_rank = 0u32;
	let mut leaf_rank = 0u32;

	let mut mask = has_child | is_leaf;
	while mask != 0 {
		let slot = mask.trailing_zeros() as u8;
		new_mat.push(old_mat.get(node.material_index(slot)));

		match node.state(slot) {
			CellState::Interior => {
				let old_child = node.interior_child_index(slot);
				if int_is_demoted[old_child as usize] {
					let src = new_leaf[int_remap[old_child as usize] as usize];
					new_leaf[(new_leaf_ptr + leaf_rank) as usize] = src;
					leaf_rank += 1;
				} else {
					let src = new_interior[int_remap[old_child as usize] as usize];
					new_interior[(new_interior_ptr + int_rank) as usize] = src;
					int_rank += 1;
				}
			}
			CellState::Leaf => {
				let src = new_leaf[leaf_remap[node.leaf_child_index(slot) as usize] as usize];
				new_leaf[(new_leaf_ptr + leaf_rank) as usize] = src;
				leaf_rank += 1;
			}
			_ => {}
		}

		mask &= mask - 1;
	}

	let new_idx = new_interior.len() as u32;
	let mut out = InteriorNode::default();
	out.set_has_child(has_child);
	out.set_is_leaf(eff_is_leaf);
	out.set_interior_offset(new_interior_ptr);
	out.set_leaf_offset(new_leaf_ptr);
	out.set_material_offset(new_mat_offset);
	new_interior.push(out);

	new_idx
}

// --- Signatures ---

/// Leaf signature: 64-bit occupancy in the high half, FNV-1a hash of materials in the low half.
fn leaf_sig(node: LeafNode, mat: &PalettedVec<Material>) -> u128 {
	let occ = node.occupancy();
	let mut hash = fnv_start();
	let mut mask = occ;
	while mask != 0 {
		let slot = mask.trailing_zeros() as u8;
		hash = fnv_mix(hash, u32::from(mat.get(node.material_index(slot))));
		mask &= mask - 1;
	}
	((occ as u128) << 64) | (hash as u128)
}

/// Same structure as leaf_sig but for a childless interior node (has_child == 0).
/// Produces the same signature as a leaf with equal occupancy and materials, enabling
/// cross-type deduplication via the shared leaf_sig_map.
fn childless_leaf_sig(node: InteriorNode, mat: &PalettedVec<Material>) -> u128 {
	let occ = node.is_leaf(); // has_child == 0 → occupancy == is_leaf
	let mut hash = fnv_start();
	let mut mask = occ;
	while mask != 0 {
		let slot = mask.trailing_zeros() as u8;
		hash = fnv_mix(hash, u32::from(mat.get(node.material_index(slot))));
		mask &= mask - 1;
	}
	((occ as u128) << 64) | (hash as u128)
}

/// Interior node signature: masks in the high half, FNV hash of (material, child-ref) pairs.
///
/// Demoted interior children are accounted for: they appear as leaf children in the
/// effective masks and their leaf indices are used in the hash (same index space as
/// original leaf children, so equal content produces equal signatures).
fn interior_sig(
	node: InteriorNode,
	mat: &PalettedVec<Material>,
	int_remap: &[u32],
	int_is_demoted: &[bool],
	leaf_remap: &[u32],
) -> u128 {
	let has_child = node.has_child();
	let is_leaf = node.is_leaf();

	// Effective is_leaf after demotion
	let mut eff_is_leaf = is_leaf;
	{
		let mut mask = has_child & !is_leaf;
		while mask != 0 {
			let s = mask.trailing_zeros() as u8;
			if int_is_demoted[node.interior_child_index(s) as usize] {
				eff_is_leaf |= 1u64 << s;
			}
			mask &= mask - 1;
		}
	}

	let masks = (has_child ^ eff_is_leaf.rotate_left(32)) ^ (has_child >> 32 | eff_is_leaf << 32);

	let mut hash = fnv_start();
	let mut mask = has_child | is_leaf;
	while mask != 0 {
		let slot = mask.trailing_zeros() as u8;
		hash = fnv_mix(hash, u32::from(mat.get(node.material_index(slot))));

		// For child references: leaf indices (original or demoted) share index space,
		// interior indices are marked with the high bit to avoid collision.
		let child_ref = match node.state(slot) {
			CellState::Interior => {
				let old_child = node.interior_child_index(slot);
				if int_is_demoted[old_child as usize] {
					int_remap[old_child as usize] // leaf index
				} else {
					int_remap[old_child as usize] | 0x8000_0000 // interior index, marked
				}
			}
			CellState::Leaf => leaf_remap[node.leaf_child_index(slot) as usize],
			_ => 0,
		};
		hash = fnv_mix(hash, child_ref);

		mask &= mask - 1;
	}

	((masks as u128) << 64) | (hash as u128)
}

// --- FNV-1a helpers ---

#[inline(always)]
fn fnv_start() -> u64 {
	0xcbf29ce484222325
}

#[inline(always)]
fn fnv_mix(hash: u64, value: u32) -> u64 {
	(hash ^ value as u64).wrapping_mul(0x100000001b3)
}
