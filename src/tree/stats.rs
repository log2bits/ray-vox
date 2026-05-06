use crate::tree::{Level, Tree};

impl<const DEPTH: usize> Tree<DEPTH> {
	pub fn bytes(&self) -> usize {
		self.levels.iter().map(|l| l.bytes()).sum()
	}

	// Leaves physically stored (unique nodes only, no SVDAG path-following).
	pub fn unique_leaf_count(&self) -> u64 {
		if self.is_leaf { return 1; }
		self.levels.iter().map(|l| l.leaf_count()).sum()
	}

	// Volume physically stored (unique nodes only).
	pub fn unique_volume(&self) -> u64 {
		if self.is_leaf { return (self.leaf_size * self.side_len() as u64).pow(3); }
		let mut total = 0u64;
		for d in 0..DEPTH {
			let side = self.leaf_size * 4u64.pow((DEPTH - d - 1) as u32);
			total += self.levels[d].leaf_count() * side * side * side;
		}
		total
	}

	// Counts represented leaves, following SVDAG sharing (shared nodes counted once per path).
	pub fn leaf_count(&self) -> u64 {
		if !self.occupied { return 0; }
		if self.is_leaf { return 1; }
		if self.levels[0].node_count() == 0 { return 0; }
		let root = self.levels[0].node_count() - 1;
		geo_leaf_count::<DEPTH>(&self.levels, 0, root)
	}

	// Total cells covered by leaves, following SVDAG sharing.
	// A leaf slot at depth d covers (leaf_size * 4^(DEPTH-d-1))^3 cells.
	pub fn stored_volume(&self) -> u64 {
		if !self.occupied { return 0; }
		if self.is_leaf { return (self.leaf_size * self.side_len() as u64).pow(3); }
		if self.levels[0].node_count() == 0 { return 0; }
		let root = self.levels[0].node_count() - 1;
		geo_stored_volume::<DEPTH>(&self.levels, 0, root, self.leaf_size)
	}

	// Estimated bytes an ESVO (leaf collapse, no dedup) would need.
	// Follows SVDAG sharing, counting shared nodes once per path.
	pub fn esvo_bytes(&self) -> usize {
		if !self.occupied || self.is_leaf { return 0; }
		if self.levels[0].node_count() == 0 { return 0; }
		let bpn = self.bytes_per_node();
		let root = self.levels[0].node_count() - 1;
		geo_esvo_bytes::<DEPTH>(&self.levels, 0, root, &bpn)
	}

	// Estimated bytes a plain SVO (all leaves at deepest level, no dedup) would need.
	// Uniform regions collapsed by ESVO are expanded back to full depth.
	pub fn svo_bytes(&self) -> usize {
		if !self.occupied || self.is_leaf { return 0; }
		if self.levels[0].node_count() == 0 { return 0; }
		let bpn = self.bytes_per_node();
		let root = self.levels[0].node_count() - 1;
		geo_svo_bytes::<DEPTH>(&self.levels, 0, root, &bpn)
	}

	fn bytes_per_node(&self) -> Vec<usize> {
		self.levels.iter()
			.map(|l| if l.node_count() > 0 { l.bytes() / l.node_count() as usize } else { 20 })
			.collect()
	}
}

fn geo_leaf_count<const DEPTH: usize>(levels: &[Level], d: usize, node: u32) -> u64 {
	let level = &levels[d];
	let occ  = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	let base = level.children_offset[node as usize];
	let is_leaf_level = d + 1 == DEPTH;
	let mut count = 0u64;
	let mut mask = occ;
	while mask != 0 {
		let s    = mask.trailing_zeros() as usize;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		if (leaf >> s) & 1 != 0 || is_leaf_level {
			count += 1;
		} else {
			let child = level.node_children.get(base + rank);
			count += geo_leaf_count::<DEPTH>(levels, d + 1, child);
		}
		mask &= mask - 1;
	}
	count
}

fn geo_stored_volume<const DEPTH: usize>(levels: &[Level], d: usize, node: u32, leaf_size: u64) -> u64 {
	let level = &levels[d];
	let occ  = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	let base = level.children_offset[node as usize];
	let is_leaf_level = d + 1 == DEPTH;
	let side = leaf_size * 4u64.pow((DEPTH - d - 1) as u32);
	let mut total = 0u64;
	let mut mask = occ;
	while mask != 0 {
		let s    = mask.trailing_zeros() as usize;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		if (leaf >> s) & 1 != 0 || is_leaf_level {
			total += side * side * side;
		} else {
			let child = level.node_children.get(base + rank);
			total += geo_stored_volume::<DEPTH>(levels, d + 1, child, leaf_size);
		}
		mask &= mask - 1;
	}
	total
}

fn geo_esvo_bytes<const DEPTH: usize>(levels: &[Level], d: usize, node: u32, bpn: &[usize]) -> usize {
	let mut total = bpn[d];
	let is_leaf_level = d + 1 == DEPTH;
	if is_leaf_level { return total; }
	let level = &levels[d];
	let occ  = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	let base = level.children_offset[node as usize];
	let mut mask = occ & !leaf;
	while mask != 0 {
		let s    = mask.trailing_zeros() as usize;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		let child = level.node_children.get(base + rank);
		total += geo_esvo_bytes::<DEPTH>(levels, d + 1, child, bpn);
		mask &= mask - 1;
	}
	total
}

fn geo_svo_bytes<const DEPTH: usize>(levels: &[Level], d: usize, node: u32, bpn: &[usize]) -> usize {
	let mut total = bpn[d];
	let is_leaf_level = d + 1 == DEPTH;
	if is_leaf_level { return total; }
	let level = &levels[d];
	let occ  = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	let base = level.children_offset[node as usize];
	let mut mask = occ;
	while mask != 0 {
		let s    = mask.trailing_zeros() as usize;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		if (leaf >> s) & 1 != 0 {
			// One leaf at level d expands to: 1 node at d+1, 64 at d+2, 64^2 at d+3, ...
			for depth_below in 1..=(DEPTH - d - 1) {
				total += 64usize.pow((depth_below - 1) as u32) * bpn[d + depth_below];
			}
		} else {
			let child = level.node_children.get(base + rank);
			total += geo_svo_bytes::<DEPTH>(levels, d + 1, child, bpn);
		}
		mask &= mask - 1;
	}
	total
}
