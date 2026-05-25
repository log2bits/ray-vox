pub mod compact;
pub mod edit;
pub mod material;
pub mod node;
pub mod rebuild;

use crate::util::PalettedVec;
use edit::Edits;
use edit::Path;
use material::Material;
use node::InteriorNode;
use node::LeafNode;

pub struct Chunk {
	pub interior_nodes: Vec<InteriorNode>,
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec<Material>,
	pub edits: Edits,
}

impl Chunk {
	pub fn new() -> Self {
		Self {
			interior_nodes: Vec::new(),
			leaf_nodes: Vec::new(),
			materials: PalettedVec::new(),
			edits: Edits::new(),
		}
	}

	pub fn gpu_size_bytes(&self) -> u32 {
		let header = 3 * size_of::<u32>(); // interior_count, leaf_count, material_count
		let interior = self.interior_nodes.len() * size_of::<InteriorNode>();
		let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
		let lut = self.materials.lut.len() as usize * size_of::<u32>();
		let indices = self.materials.indices.words.len() * size_of::<u32>();
		(header + interior + leaf + lut + indices) as u32
	}

	pub fn apply_edits(&mut self) {
		let mut edits = std::mem::take(&mut self.edits);
		edits.sort();
		for batch in &edits.batches {
			let start = batch.range().start as usize;
			let end = batch.range().end as usize;
			let slice = &edits.edits[start..end];
			self.apply_batch(slice);
		}
		self.compact();
	}

	pub fn apply_batch(&mut self, batch: &[(Path, Material)]) {
		// Root edits sort to the front (path value 0). The last one determines the
		// starting state; any edits before it are overridden.
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

		// Determine the current state of the chunk before applying sub_batch.
		// If the chunk has no tree yet but a single fill material, we expand that fill
		// into a virtual fully-filled root as we descend into it.
		let old_root = self.interior_nodes.last().copied();
		let expand_fill = if old_root.is_none() && self.materials.len() == 1 {
			Some(self.materials.get(0))
		} else {
			None
		};

		self.rebuild_interior(old_root, expand_fill, 0, sub_batch);
		// The new root is now the last element of interior_nodes.
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
}
