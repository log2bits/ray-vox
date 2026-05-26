use ahash::AHashMap;

use super::material::Material;
use super::node::{CellState, InteriorNode, InteriorNodeWide, LeafNode};
use super::{Chunk, Compressed, Editing};
use crate::util::PalettedVec;

/// Compacts and compresses a `Chunk<Editing>` into a `Chunk<Compressed>`.
///
/// Removes orphan nodes left by path-copy edits, deduplicates identical subtrees
/// (DAG sharing), and packs interior node offsets into the 13/19-bit compressed layout.
pub fn compress(src: Chunk<Editing>) -> Chunk<Compressed> {
    if src.state.interior_nodes.is_empty() {
        return Chunk {
            leaf_nodes: src.leaf_nodes,
            materials: src.materials,
            state: Compressed {
                interior_nodes: Vec::new(),
            },
        };
    }

    let root = src.state.interior_nodes.len() as u32 - 1;
    let (by_depth, reachable_leaf) = collect_reachable(&src.state.interior_nodes, root);

    let mut new_interior: Vec<InteriorNode> = Vec::new();
    let mut new_leaf: Vec<LeafNode> = Vec::new();
    let mut new_mat: PalettedVec<Material> = PalettedVec::new();
    let mut canonical_interiors: Vec<InteriorNode> = Vec::new();

    // dedup_leaves builds canonical leaf nodes (for signature/material dedup) without adding to
    // new_leaf. emit_interior_node later copies canonical leaves into the parent's contiguous block.
    let (leaf_remap, canonical_leaves) = dedup_leaves(
        &src.leaf_nodes,
        &src.materials,
        &reachable_leaf,
        &mut new_mat,
    );

    let mut int_remap = vec![0u32; src.state.interior_nodes.len()];
    for depth in (0..3usize).rev() {
        let pairs = dedup_interior_at_depth(
            &src.state.interior_nodes,
            &src.materials,
            &by_depth[depth],
            &int_remap,
            &leaf_remap,
            &canonical_leaves,
            &mut canonical_interiors,
            &mut new_interior,
            &mut new_leaf,
            &mut new_mat,
        );
        for (old, new) in pairs {
            int_remap[old as usize] = new;
        }
    }

    // Push the root canonical so it sits at new_interior.len() - 1 where the shader expects it.
    if let Some(&root) = canonical_interiors.last() {
        new_interior.push(root);
    }

    Chunk {
        leaf_nodes: new_leaf,
        materials: new_mat,
        state: Compressed {
            interior_nodes: new_interior,
        },
    }
}

// --- Reachability ---

fn collect_reachable(
    interior: &[InteriorNodeWide],
    root: u32,
) -> ([Vec<u32>; 3], Vec<u32>) {
    let mut by_depth: [Vec<u32>; 3] = Default::default();
    let mut reachable_leaf: Vec<u32> = Vec::new();

    by_depth[0].push(root);

    for depth in 0..3usize {
        let mut i = 0;
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

/// Returns (remap, canonical_leaves).
///
/// `remap[old_idx]` gives an index into `canonical_leaves`.
/// Materials for each canonical leaf are pushed to `new_mat`.
/// `canonical_leaves` is NOT added to the output leaf array — it is a lookup table
/// used by `emit_interior_node` to copy leaves into parent contiguous blocks.
fn dedup_leaves(
    old_leaves: &[LeafNode],
    old_mat: &PalettedVec<Material>,
    reachable: &[u32],
    new_mat: &mut PalettedVec<Material>,
) -> (Vec<u32>, Vec<LeafNode>) {
    let mut remap = vec![0u32; old_leaves.len()];
    let mut canonical: Vec<LeafNode> = Vec::new();
    let mut sig_to_canonical: AHashMap<u128, u32> = AHashMap::new();

    for &old_idx in reachable {
        let node = old_leaves[old_idx as usize];
        let sig = leaf_sig(node, old_mat);

        let canonical_idx = if let Some(&idx) = sig_to_canonical.get(&sig) {
            idx
        } else {
            let canonical_idx = canonical.len() as u32;
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
            canonical.push(out);
            sig_to_canonical.insert(sig, canonical_idx);
            canonical_idx
        };

        remap[old_idx as usize] = canonical_idx;
    }

    (remap, canonical)
}

// --- Interior dedup ---

fn dedup_interior_at_depth(
    old_interior: &[InteriorNodeWide],
    old_mat: &PalettedVec<Material>,
    reachable: &[u32],
    int_remap: &[u32],
    leaf_remap: &[u32],
    canonical_leaves: &[LeafNode],
    canonical_interiors: &mut Vec<InteriorNode>,
    new_interior: &mut Vec<InteriorNode>,
    new_leaf: &mut Vec<LeafNode>,
    new_mat: &mut PalettedVec<Material>,
) -> Vec<(u32, u32)> {
    let mut sig_to_new: AHashMap<u128, u32> = AHashMap::new();
    let mut pairs = Vec::with_capacity(reachable.len());

    for &old_idx in reachable {
        let node = old_interior[old_idx as usize];
        let sig = interior_sig(node, old_mat, int_remap, leaf_remap);

        let new_idx = if let Some(&idx) = sig_to_new.get(&sig) {
            idx
        } else {
            let new_idx = emit_interior_node(
                node, old_mat, int_remap, leaf_remap, canonical_leaves,
                canonical_interiors, new_interior, new_leaf, new_mat,
            );
            sig_to_new.insert(sig, new_idx);
            new_idx
        };

        pairs.push((old_idx, new_idx));
    }

    pairs
}

/// Appends a single interior node (and a fresh contiguous block of its children)
/// to the compressed output arrays. Packs offsets into the 13/19-bit layout.
/// Returns the new node's index in `new_interior`.
fn emit_interior_node(
    node: InteriorNodeWide,
    old_mat: &PalettedVec<Material>,
    int_remap: &[u32],
    leaf_remap: &[u32],
    canonical_leaves: &[LeafNode],
    canonical_interiors: &mut Vec<InteriorNode>,
    new_interior: &mut Vec<InteriorNode>,
    new_leaf: &mut Vec<LeafNode>,
    new_mat: &mut PalettedVec<Material>,
) -> u32 {
    let has_child = node.has_child();
    let is_leaf = node.is_leaf();

    let n_interior = (has_child & !is_leaf).count_ones() as usize;
    let n_leaf = (has_child & is_leaf).count_ones() as usize;

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
                let remapped = int_remap[node.interior_child_index(slot) as usize];
                let src = canonical_interiors[remapped as usize];
                new_interior[(new_interior_ptr + int_rank) as usize] = src;
                int_rank += 1;
            }
            CellState::Leaf => {
                let canonical_idx = leaf_remap[node.leaf_child_index(slot) as usize];
                new_leaf[(new_leaf_ptr + leaf_rank) as usize] = canonical_leaves[canonical_idx as usize];
                leaf_rank += 1;
            }
            _ => {}
        }

        mask &= mask - 1;
    }

    let new_idx = canonical_interiors.len() as u32;
    let mut out = InteriorNode::default();
    out.set_has_child(has_child);
    out.set_is_leaf(is_leaf);
    out.set_interior_offset(new_interior_ptr);
    out.set_leaf_offset(new_leaf_ptr);
    out.set_material_offset(new_mat_offset);
    canonical_interiors.push(out);

    new_idx
}

// --- Signatures ---

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

fn interior_sig(
    node: InteriorNodeWide,
    mat: &PalettedVec<Material>,
    int_remap: &[u32],
    leaf_remap: &[u32],
) -> u128 {
    let has_child = node.has_child();
    let is_leaf = node.is_leaf();

    let masks = (has_child ^ is_leaf.rotate_left(32)) ^ (has_child >> 32 | is_leaf << 32);

    let mut hash = fnv_start();
    let mut mask = has_child | is_leaf;
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;

        hash = fnv_mix(hash, u32::from(mat.get(node.material_index(slot))));

        let child_ref = match node.state(slot) {
            CellState::Interior => int_remap[node.interior_child_index(slot) as usize],
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
