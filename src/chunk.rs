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
