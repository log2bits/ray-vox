mod edit;
mod material;
mod node;

use crate::chunk::edit::Edits;
use crate::chunk::node::InteriorNode;
use crate::chunk::node::LeafNode;
use crate::util::PalettedVec;

pub struct Chunk {
	pub interior_nodes: Vec<InteriorNode>,
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec,
	pub edits: Edits,
}

enum StackFrame {
	Interior(u32),
	Leaf(u32),
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
	}

	pub fn apply_batch(&mut self, batch: &[(u32, u32)]) {
		let mut stack: Vec<(u32, u8)> = Vec::new(); // (node_index, slot)
		let mut prev_path = 0u32;
		for (path, material) in batch {}
	}

	pub fn is_uniform(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.len() == 1
	}

	pub fn is_empty(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.is_empty()
	}
}
