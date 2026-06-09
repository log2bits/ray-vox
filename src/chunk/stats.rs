use super::{Child, Chunk};

const DEPTH: u32 = 4;
const SIDE: u64 = 256;

impl Chunk {
	pub fn stored_volume(&self) -> u64 {
		if self.is_empty() {
			return 0;
		}
		if self.is_uniform() {
			return SIDE.pow(3);
		}
		if self.interior_nodes.is_empty() {
			return 0;
		}
		self.subtree_voxels(self.root_idx(), 0)
	}

	fn subtree_voxels(&self, idx: u32, depth: u32) -> u64 {
		let mut total = 0u64;
		for slot in self.interior_nodes[idx as usize].masks.occupancy().iter_slots() {
			total += match self.child(idx, slot) {
				Child::Empty => 0,
				Child::Filled(_) => slot_voxels(depth),
				Child::Interior(child_idx) => self.subtree_voxels(child_idx, depth + 1),
				Child::Leaf(leaf_idx) => self.leaf_nodes[leaf_idx as usize].occupancy.count() as u64,
			};
		}
		total
	}
}

fn slot_voxels(depth: u32) -> u64 {
	let s = 4u64.pow(DEPTH - depth - 1);
	s * s * s
}
