use ahash::AHashMap;

use super::material::Material;
use super::node::{CellState, ChildMasks, InteriorNode, InteriorNodeWide, LeafNode};
use super::rebuild::mode_over;
use super::{Chunk, MutableChunk};
use crate::util::PalettedVec;
use crate::util::types::Mask64;

/// Compact a MutableChunk into a frozen Chunk. Drops orphan nodes left by path-copy
/// edits, dedupes identical subtrees, and packs offsets into the 13/19-bit layout.
pub fn compress(src: MutableChunk) -> Chunk {
	if src.interior_nodes.is_empty() {
		return Chunk {
			leaf_nodes: src.leaf_nodes,
			materials: src.materials,
			interior_nodes: Vec::new(),
		};
	}

	let root = src.interior_nodes.len() as u32 - 1;
	let (by_depth, reachable_leaves) = collect_reachable(&src.interior_nodes, root);

	let mut c = Compressor::new(&src.interior_nodes, &src.leaf_nodes, &src.materials);

	// materials[0] is always the chunk-level LOD. Node material_offsets land at 1+.
	let chunk_lod = compute_chunk_lod(&src.interior_nodes[root as usize], &src.materials);
	c.push_material(chunk_lod);

	c.dedup_leaves(&reachable_leaves);
	for depth in (0..3).rev() {
		c.dedup_interior_at_depth(&by_depth[depth]);
	}
	c.push_root(root);

	Chunk {
		leaf_nodes: c.out_leaves,
		materials: c.out_materials,
		interior_nodes: c.out_interiors,
	}
}

/// Working state for one compress pass.
///
/// Dedup uses hash-and-verify: maps hold u64 content hashes, hits run a full
/// content compare before merging. A collision with no content match emits a
/// duplicate canonical, which is statistically unmeasurable but never wrong.
struct Compressor<'a> {
	src_interiors: &'a [InteriorNodeWide],
	src_leaves: &'a [LeafNode],
	src_materials: &'a PalettedVec<Material>,

	int_remap: Vec<u32>,
	leaf_remap: Vec<u32>,
	demoted: Vec<bool>,
	demoted_remap: Vec<u32>,

	canonical_interiors: Vec<InteriorNode>,
	canonical_leaves: Vec<LeafNode>,
	/// Per-canonical child canonical indices in slot order (interiors then leaves).
	/// Lets interior_eq compare child structure without walking the output arrays.
	canonical_int_children: Vec<Vec<u32>>,

	out_interiors: Vec<InteriorNode>,
	out_leaves: Vec<LeafNode>,
	out_materials: PalettedVec<Material>,
	/// Transient value-to-index map so push_material is O(1) instead of doing the
	/// linear scan PalettedVec::push would. Dropped before the frozen Chunk returns.
	palette_index: AHashMap<Material, u32>,

	leaf_sig_map: AHashMap<u64, u32>,
	int_sig_map: AHashMap<u64, u32>,
}

impl<'a> Compressor<'a> {
	fn new(
		src_interiors: &'a [InteriorNodeWide],
		src_leaves: &'a [LeafNode],
		src_materials: &'a PalettedVec<Material>,
	) -> Self {
		Self {
			src_interiors,
			src_leaves,
			src_materials,
			int_remap: vec![0; src_interiors.len()],
			leaf_remap: vec![0; src_leaves.len()],
			demoted: vec![false; src_interiors.len()],
			demoted_remap: vec![0; src_interiors.len()],
			canonical_interiors: Vec::new(),
			canonical_leaves: Vec::new(),
			canonical_int_children: Vec::new(),
			out_interiors: Vec::new(),
			out_leaves: Vec::new(),
			out_materials: PalettedVec::new(),
			palette_index: AHashMap::new(),
			leaf_sig_map: AHashMap::new(),
			int_sig_map: AHashMap::new(),
		}
	}

	/// O(1) replacement for out_materials.push. Preserves first-occurrence ordering
	/// so palette indices and bit packing come out byte-identical.
	fn push_material(&mut self, m: Material) {
		let idx = *self.palette_index.entry(m).or_insert_with(|| {
			let i = self.out_materials.lut.values.len() as u32;
			self.out_materials.lut.values.push(m);
			i
		});
		self.out_materials.indices.push(idx);
	}

	#[inline]
	fn src_slab_mat(&self, n: &InteriorNodeWide, slot: u8) -> Material {
		self.src_materials.get(n.material_index(slot))
	}

	#[inline]
	fn out_slab_mat(&self, n: &InteriorNode, slot: u8) -> Material {
		self.out_materials.get(n.material_index(slot))
	}

	/// Same masks but with demoted interior children re-labeled as leaves.
	fn apply_demotions(&self, node: &InteriorNodeWide) -> ChildMasks {
		let src = node.masks;
		let mut is_leaf = src.is_leaf;
		for slot in src.interiors().iter_slots() {
			let child = node.interior_child_index(slot);
			if self.demoted[child as usize] {
				is_leaf |= Mask64::bit(slot);
			}
		}
		ChildMasks { has_child: src.has_child, is_leaf }
	}

	/// Canonical index of the child at this slot. Routes through int_remap,
	/// demoted_remap, or leaf_remap based on the pre-demotion state.
	fn child_canonical(&self, node: &InteriorNodeWide, slot: u8) -> u32 {
		match node.masks.state(slot) {
			CellState::Interior => {
				let child = node.interior_child_index(slot);
				if self.demoted[child as usize] {
					self.demoted_remap[child as usize]
				} else {
					self.int_remap[child as usize]
				}
			}
			CellState::Leaf => self.leaf_remap[node.leaf_child_index(slot) as usize],
			_ => 0,
		}
	}

	fn dedup_leaves(&mut self, reachable: &[u32]) {
		for &old_idx in reachable {
			let node = self.src_leaves[old_idx as usize];
			let hash = self.leaf_hash(node);
			let canonical = match self.leaf_sig_map.get(&hash) {
				Some(&cand) if self.leaf_eq(node, cand) => cand,
				_ => {
					let new_idx = self.emit_canonical_leaf(node.occupancy, |c, s| {
						c.src_materials.get(node.material_index(s))
					});
					self.leaf_sig_map.entry(hash).or_insert(new_idx);
					new_idx
				}
			};
			self.leaf_remap[old_idx as usize] = canonical;
		}
	}

	fn dedup_interior_at_depth(&mut self, reachable: &[u32]) {
		self.int_sig_map.clear();
		for &old_idx in reachable {
			let node = self.src_interiors[old_idx as usize];

			// No real children means only Filled or Empty slots. Demote to a leaf.
			if node.masks.has_child.is_empty() {
				let canonical = self.dedup_as_demoted_leaf(node);
				self.demoted[old_idx as usize] = true;
				self.demoted_remap[old_idx as usize] = canonical;
				continue;
			}

			let hash = self.interior_hash(node);
			let canonical = match self.int_sig_map.get(&hash) {
				Some(&cand) if self.interior_eq(node, cand) => cand,
				_ => {
					let new_idx = self.emit_interior(node);
					self.int_sig_map.entry(hash).or_insert(new_idx);
					new_idx
				}
			};
			self.int_remap[old_idx as usize] = canonical;
		}
	}

	/// Turn a childless interior into a canonical leaf.
	fn dedup_as_demoted_leaf(&mut self, node: InteriorNodeWide) -> u32 {
		let occupancy = node.masks.is_leaf;
		let mut hash = fnv_start();
		for slot in occupancy.iter_slots() {
			hash = fnv_mix(hash, self.src_slab_mat(&node, slot).into());
		}
		if let Some(&cand) = self.leaf_sig_map.get(&hash) {
			if self.demoted_eq(node, occupancy, cand) {
				return cand;
			}
		}
		let new_idx = self.emit_canonical_leaf(occupancy, |c, s| c.src_slab_mat(&node, s));
		self.leaf_sig_map.entry(hash).or_insert(new_idx);
		new_idx
	}

	fn emit_canonical_leaf(
		&mut self,
		occupancy: Mask64,
		mut mat_at: impl FnMut(&Self, u8) -> Material,
	) -> u32 {
		let canonical_idx = self.canonical_leaves.len() as u32;
		let mat_offset = self.out_materials.len();
		for slot in occupancy.iter_slots() {
			let m = mat_at(self, slot);
			self.push_material(m);
		}
		let mut out = LeafNode::default();
		out.occupancy = occupancy;
		out.set_material_offset(mat_offset);
		self.canonical_leaves.push(out);
		canonical_idx
	}

	/// Emit a canonical interior. Appends child blocks in slot order, packs offsets,
	/// and records child canonical indices for later eq verification.
	fn emit_interior(&mut self, node: InteriorNodeWide) -> u32 {
		let masks = self.apply_demotions(&node);
		let n_children = masks.has_child.count() as usize + masks.filled().count() as usize;

		let interior_ptr = self.out_interiors.len() as u32;
		let leaf_ptr = self.out_leaves.len() as u32;
		let mat_offset = self.out_materials.len();

		// Materials: one per occupied slot in slot order.
		for slot in masks.occupancy().iter_slots() {
			let mat = self.src_slab_mat(&node, slot);
			self.push_material(mat);
		}

		// Interior and leaf child blocks: separately in slot order so popcount-rank
		// indexing lines up.
		let mut children: Vec<u32> = Vec::with_capacity(n_children);
		for slot in masks.interiors().iter_slots() {
			let canonical = self.child_canonical(&node, slot);
			let child_node = self.canonical_interiors[canonical as usize];
			self.out_interiors.push(child_node);
			children.push(canonical);
		}
		for slot in masks.leaves().iter_slots() {
			let canonical = self.child_canonical(&node, slot);
			let child_node = self.canonical_leaves[canonical as usize];
			self.out_leaves.push(child_node);
			children.push(canonical);
		}

		let new_idx = self.canonical_interiors.len() as u32;
		let mut out = InteriorNode::default();
		out.masks = masks;
		out.set_interior_offset(interior_ptr);
		out.set_leaf_offset(leaf_ptr);
		out.set_material_offset(mat_offset);
		self.canonical_interiors.push(out);
		self.canonical_int_children.push(children);
		new_idx
	}

	/// Push the root canonical to its output array. Normal roots go to out_interiors
	/// at len-1 where the shader looks. Demoted roots go to out_leaves instead.
	fn push_root(&mut self, root_old_idx: u32) {
		if self.demoted[root_old_idx as usize] {
			let canonical = self.demoted_remap[root_old_idx as usize];
			self.out_leaves.push(self.canonical_leaves[canonical as usize]);
		} else if let Some(&root) = self.canonical_interiors.last() {
			self.out_interiors.push(root);
		}
	}

	fn leaf_hash(&self, node: LeafNode) -> u64 {
		let mut hash = fnv_start();
		hash = fnv_mix(hash, fold_mask(node.occupancy));
		for slot in node.occupancy.iter_slots() {
			hash = fnv_mix(hash, self.src_materials.get(node.material_index(slot)).into());
		}
		hash
	}

	fn interior_hash(&self, node: InteriorNodeWide) -> u64 {
		let masks = self.apply_demotions(&node);
		let mut hash = fnv_start();
		hash = fnv_mix(hash, fold_mask(masks.has_child));
		hash = fnv_mix(hash, fold_mask(masks.is_leaf));
		for slot in node.masks.occupancy().iter_slots() {
			hash = fnv_mix(hash, self.src_slab_mat(&node, slot).into());
			hash = fnv_mix(hash, self.child_canonical(&node, slot));
		}
		hash
	}

	fn leaf_eq(&self, node: LeafNode, canonical_idx: u32) -> bool {
		let c = self.canonical_leaves[canonical_idx as usize];
		if c.occupancy != node.occupancy {
			return false;
		}
		node.occupancy.iter_slots().all(|slot| {
			self.src_materials.get(node.material_index(slot))
				== self.out_materials.get(c.material_index(slot))
		})
	}

	fn demoted_eq(&self, node: InteriorNodeWide, occupancy: Mask64, canonical_idx: u32) -> bool {
		let c = self.canonical_leaves[canonical_idx as usize];
		if c.occupancy != occupancy {
			return false;
		}
		occupancy.iter_slots().all(|slot| {
			self.src_slab_mat(&node, slot) == self.out_materials.get(c.material_index(slot))
		})
	}

	fn interior_eq(&self, node: InteriorNodeWide, canonical_idx: u32) -> bool {
		let masks = self.apply_demotions(&node);
		let c = self.canonical_interiors[canonical_idx as usize];
		if c.masks.has_child != masks.has_child || c.masks.is_leaf != masks.is_leaf {
			return false;
		}
		// Per-slot materials.
		for slot in masks.occupancy().iter_slots() {
			if self.src_slab_mat(&node, slot) != self.out_slab_mat(&c, slot) {
				return false;
			}
		}
		// Child canonical indices in the same slot-order interior-then-leaf used at
		// emit time.
		let children = &self.canonical_int_children[canonical_idx as usize];
		let mut i = 0;
		for slot in masks.interiors().iter_slots() {
			if children[i] != self.child_canonical(&node, slot) {
				return false;
			}
			i += 1;
		}
		for slot in masks.leaves().iter_slots() {
			if children[i] != self.child_canonical(&node, slot) {
				return false;
			}
			i += 1;
		}
		true
	}
}

/// Mode of the root's per-slot representatives. The chunk's whole-volume LOD.
fn compute_chunk_lod(root: &InteriorNodeWide, materials: &PalettedVec<Material>) -> Material {
	let occ = root.masks.occupancy();
	if occ.is_empty() {
		return Material::air();
	}
	let mut mats = [Material::air(); 64];
	for slot in occ.iter_slots() {
		mats[slot as usize] = materials.get(root.material_index(slot));
	}
	mode_over(occ, &mats)
}

/// BFS from the root: returns reachable interior indices grouped by depth and a
/// flat list of reachable leaf indices.
fn collect_reachable(interiors: &[InteriorNodeWide], root: u32) -> ([Vec<u32>; 3], Vec<u32>) {
	let mut by_depth: [Vec<u32>; 3] = Default::default();
	let mut leaves: Vec<u32> = Vec::new();
	by_depth[0].push(root);

	for depth in 0..3 {
		let mut i = 0;
		while i < by_depth[depth].len() {
			let node = interiors[by_depth[depth][i] as usize];
			i += 1;
			if depth < 2 {
				let base = node.interior_offset();
				for (rank, _) in node.masks.interiors().iter_slots().enumerate() {
					by_depth[depth + 1].push(base + rank as u32);
				}
			}
			let base = node.leaf_offset();
			for (rank, _) in node.masks.leaves().iter_slots().enumerate() {
				leaves.push(base + rank as u32);
			}
		}
		by_depth[depth].sort_unstable();
		by_depth[depth].dedup();
	}
	leaves.sort_unstable();
	leaves.dedup();
	(by_depth, leaves)
}

#[inline]
fn fold_mask(m: Mask64) -> u32 {
	let v = m.raw();
	(v as u32) ^ ((v >> 32) as u32)
}

#[inline(always)]
fn fnv_start() -> u64 {
	0xcbf29ce484222325
}

#[inline(always)]
fn fnv_mix(hash: u64, value: u32) -> u64 {
	(hash ^ value as u64).wrapping_mul(0x100000001b3)
}
