use super::node::CellState;
use super::Chunk;

const DEPTH: u32 = 4;
const SIDE: u64 = 256;

impl Chunk {
	/// Voxels actually filled with some material (any non-air, any depth).
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

	fn root_idx(&self) -> u32 {
		(self.interior_nodes.len() - 1) as u32
	}

	fn subtree_voxels(&self, idx: u32, depth: u32) -> u64 {
		let n = self.interior_nodes[idx as usize];
		let mut total = 0u64;
		let mut mask = n.occupancy();
		while mask != 0 {
			let slot = mask.trailing_zeros() as u8;
			total += match n.state(slot) {
				CellState::Empty => 0,
				CellState::Filled => slot_voxels(depth),
				CellState::Interior => self.subtree_voxels(n.interior_child_index(slot), depth + 1),
				CellState::Leaf => {
					let leaf = self.leaf_nodes[n.leaf_child_index(slot) as usize];
					leaf.occupied_count() as u64
				}
			};
			mask &= mask - 1;
		}
		total
	}
}

/// Number of voxels a single slot at `depth` covers: (4^(DEPTH - depth - 1))^3.
fn slot_voxels(depth: u32) -> u64 {
	let s = 4u64.pow(DEPTH - depth - 1);
	s * s * s
}
