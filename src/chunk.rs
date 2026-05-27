pub mod compact;
pub mod edit;
pub mod material;
pub mod node;
pub mod rebuild;
pub mod stats;

use crate::util::PalettedVec;
use bytemuck;
use edit::ChunkEdits;
use edit::Path;
use material::Material;
use node::{InteriorNode, InteriorNodeWide, LeafNode};

/// A chunk in its compressed, GPU-ready form.
///
/// To apply edits use [`Chunk::apply_edits`], which decompresses internally,
/// applies all edits, re-compresses, and returns a new `Chunk`.
pub struct Chunk {
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec<Material>,
	pub interior_nodes: Vec<InteriorNode>,
}

impl Chunk {
	pub fn new() -> Self {
		Self {
			leaf_nodes: Vec::new(),
			materials: PalettedVec::new(),
			interior_nodes: Vec::new(),
		}
	}

	pub fn is_root_leaf(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.len() == 1
	}

	pub fn is_uniform(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.len() == 1
	}

	pub fn is_empty(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.is_empty()
	}

	pub fn gpu_size_bytes(&self) -> u32 {
		let header = 3 * size_of::<u32>(); // interior_count, leaf_count, material_count
		let interior = self.interior_nodes.len() * size_of::<InteriorNode>();
		let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
		let lut = self.materials.lut.len() as usize * size_of::<Material>();
		let indices = self.materials.indices.words.len() * size_of::<u32>();
		(header + interior + leaf + lut + indices) as u32
	}

	pub fn gpu_bytes(&self) -> &[u8] {
		bytemuck::cast_slice(&self.interior_nodes)
	}

	/// Apply a set of edits to this chunk and return the updated chunk.
	///
	/// Internally: decompresses → applies edits → compresses.
	/// Compression happens once at the end, not after each individual edit.
	pub fn apply_edits(self, edits: ChunkEdits) -> Chunk {
		let mut editing = self.into_editing();
		editing.edits = edits;
		editing.flush_edits();
		editing.into_compressed()
	}

	fn into_editing(self) -> MutableChunk {
		let wide: Vec<InteriorNodeWide> = self
			.interior_nodes
			.iter()
			.map(|n| {
				let mut w = InteriorNodeWide::default();
				w.set_has_child(n.has_child());
				w.set_is_leaf(n.is_leaf());
				w.set_interior_offset(n.interior_offset());
				w.set_leaf_offset(n.leaf_offset());
				w.set_material_offset(n.material_offset());
				w
			})
			.collect();
		MutableChunk {
			leaf_nodes: self.leaf_nodes,
			materials: self.materials,
			interior_nodes: wide,
			edits: ChunkEdits::default(),
		}
	}
}

impl Default for Chunk {
	fn default() -> Self {
		Self::new()
	}
}

impl Clone for Chunk {
	fn clone(&self) -> Self {
		Self {
			leaf_nodes: self.leaf_nodes.clone(),
			materials: self.materials.clone(),
			interior_nodes: self.interior_nodes.clone(),
		}
	}
}

pub(crate) struct MutableChunk {
	pub(crate) leaf_nodes: Vec<LeafNode>,
	pub(crate) materials: PalettedVec<Material>,
	pub(crate) interior_nodes: Vec<InteriorNodeWide>,
	pub(crate) edits: ChunkEdits,
}

impl MutableChunk {
	/// Applies all accumulated edits (sorts, batches, rebuilds) without compressing.
	pub(crate) fn flush_edits(&mut self) {
		let mut edits = std::mem::take(&mut self.edits);
		edits.sort();
		for batch in &edits.ranges {
			let start = batch.range.start as usize;
			let end = batch.range.end as usize;
			let slice = &edits.edits[start..end];
			self.apply_batch(slice);
		}
	}

	pub(crate) fn apply_batch(&mut self, batch: &[(Path, Material)]) {
		let root_edit_count = batch.partition_point(|(p, _): &(Path, Material)| p.is_root());
		let sub_batch = &batch[root_edit_count..];

		if root_edit_count > 0 {
			let (_, fill_mat) = batch[root_edit_count - 1];
			self.interior_nodes.clear();
			self.leaf_nodes.clear();
			self.materials.clear();
			if !fill_mat.is_air() {
				self.materials.push(fill_mat);
			}
			if sub_batch.is_empty() {
				return;
			}
		}

		if sub_batch.is_empty() {
			return;
		}

		let old_root = self.interior_nodes.last().copied();
		let expand_fill = if old_root.is_none() && self.materials.len() == 1 {
			Some(self.materials.get(0))
		} else {
			None
		};

		match self.rebuild_interior(old_root, expand_fill, 0, sub_batch) {
			rebuild::RebuildResult::Empty => {}
			rebuild::RebuildResult::Filled(mat) => {
				self.interior_nodes.clear();
				self.leaf_nodes.clear();
				self.materials.clear();
				self.materials.push(mat);
			}
			rebuild::RebuildResult::Interior(_, new_root) => {
				self.interior_nodes.push(new_root);
			}
			rebuild::RebuildResult::Leaf(..) => unreachable!(),
		}
	}

	pub(crate) fn into_compressed(self) -> Chunk {
		compact::compress(self)
	}
}

impl Clone for MutableChunk {
	fn clone(&self) -> Self {
		Self {
			leaf_nodes: self.leaf_nodes.clone(),
			materials: self.materials.clone(),
			interior_nodes: self.interior_nodes.clone(),
			edits: self.edits.clone(),
		}
	}
}
