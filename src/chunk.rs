mod edit;
mod material;
mod node;

use crate::chunk::node::InteriorNode;
use crate::chunk::node::LeafNode;
use crate::util::PalettedVec;

pub struct Chunk {
	pub interior_nodes: Vec<InteriorNode>,
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec,
}

impl Chunk {
	pub fn gpu_size_bytes(&self) -> u32 {
		let header = 3 * size_of::<u32>(); // interior_count, leaf_count, material_count
		let interior = self.interior_nodes.len() * size_of::<InteriorNode>();
		let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
		let lut = self.materials.lut.len() as usize * size_of::<u32>();
		let indices = self.materials.indices.words.len() * size_of::<u32>();
		(header + interior + leaf + lut + indices) as u32
	}
}
