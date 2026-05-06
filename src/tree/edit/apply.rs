use crate::tree::{Level, Tree};
use super::{Edit, EditPacket, OrderedEdits, DELETE};

impl<const DEPTH: usize> Tree<DEPTH> {
	pub fn queue_edit(&mut self, edit: Edit<DEPTH>) {
		self.edits.add_edit(edit);
	}

	pub fn queue_edit_packet(&mut self, packet: EditPacket<DEPTH>) {
		self.edits.add_edit_packet(packet);
	}

	pub fn apply_edits(&mut self) {
		let edits = std::mem::take(&mut self.edits);
		for packet in edits.packets {
			self.apply_edit_packet(packet);
		}
	}

	fn apply_ordered_edits(&mut self, edits: OrderedEdits<DEPTH>) {
		for packet in edits.packets {
			self.apply_edit_packet(packet);
		}
	}

	fn apply_edit_packet(&mut self, mut packet: EditPacket<DEPTH>) {
		if packet.paths.is_empty() {
			return;
		}
		packet.sort();

		let edits: Vec<([u8; DEPTH], usize, u32)> = packet.paths.iter()
			.enumerate()
			.map(|(i, path)| {
				let val = packet.lut.get(packet.values.get(i as u32));
				let (raw, depth) = path.to_raw();
				(raw, depth as usize, val)
			})
			.collect();

		self.apply_edit_slice(&edits);
	}

	fn apply_edit_slice(&mut self, edits: &[([u8; DEPTH], usize, u32)]) {
		// depth=0 edit covers the whole tree.
		if let Some(&(_, _, val)) = edits.iter().rfind(|(_, d, _)| *d == 0) {
			self.occupied = val != DELETE;
			self.is_leaf = val != DELETE;
			self.value = val;
			return;
		}

		self.occupied = true;

		// Ensure a root node exists to descend into.
		let root = if self.is_leaf {
			self.is_leaf = false;
			let val = self.value;
			alloc_expanded(&mut self.levels, 0, val)
		} else if self.levels[0].node_count() == 0 {
			alloc_empty(&mut self.levels, 0)
		} else {
			// Root is always the last node in levels[0] (most recently appended).
			self.levels[0].node_count() - 1
		};

		let new_root = rebuild(&mut self.levels, 0, root, edits);

		if self.levels[0].occupancy_mask[new_root as usize] == 0 {
			self.occupied = false;
		}
	}

}

// Allocate an empty node at the given tree depth.
fn alloc_empty(levels: &mut [Level], d: usize) -> u32 {
	let offset = levels[d].children_len();
	levels[d].push_node(0, 0, offset)
}

// Allocate a node at the given tree depth with all 64 slots as leaves of value `val`.
fn alloc_expanded(levels: &mut [Level], d: usize, val: u32) -> u32 {
	levels[d].push_leaf(u64::MAX, val)
}

// If all 64 slots are occupied leaves with the same value, return that value.
fn uniform_leaf_value(level: &Level, node: u32) -> Option<u32> {
	let occ = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	if occ != u64::MAX || leaf != u64::MAX { return None; }
	let base = level.children_offset[node as usize];
	let v = level.values.get(base);
	for i in 1..64 {
		if level.values.get(base + i) != v { return None; }
	}
	Some(v)
}

// LOD representative for a node: mode of occupied children's values.
// Returns DELETE if fewer than half of the 64 slots are occupied.
fn lod_value(level: &Level, node: u32) -> u32 {
	let occ      = level.occupancy_mask[node as usize];
	let occupied = occ.count_ones() as usize;
	if occupied < 32 {
		return DELETE;
	}
	let base = level.children_offset[node as usize];
	let mut best_val   = level.values.get(base);
	let mut best_count = 0usize;
	for rank in 0..occupied {
		let val   = level.values.get(base + rank as u32);
		let count = (0..occupied).filter(|&r| level.values.get(base + r as u32) == val).count();
		if count > best_count {
			best_count = count;
			best_val   = val;
		}
		if best_count > occupied / 2 { break; } // majority found, can't be beaten
	}
	best_val
}

// Path-copy edit: reads node at `node_idx`, appends a new modified copy, returns new index.
// Unedited slots share the same child indices as the original (SVDAG sharing).
// Only touches levels[d..], so levels[0..d] can be borrowed separately.
fn rebuild<const DEPTH: usize>(
	levels: &mut [Level],
	d: usize,
	node_idx: u32,
	edits: &[([u8; DEPTH], usize, u32)],
) -> u32 {
	if edits.is_empty() {
		return node_idx; // no edits — share this node unchanged
	}

	let is_leaf_level = d + 1 == DEPTH;

	// Phase 1: read current node data into locals (releases borrow on levels[d]).
	let occ = levels[d].occupancy_mask[node_idx as usize];
	let leaf = levels[d].leaf_mask[node_idx as usize];
	let base = levels[d].children_offset[node_idx as usize];

	let mut slot_val = [0u32; 64];
	let mut slot_child = [0u32; 64];
	let mut mask = occ;
	while mask != 0 {
		let s = mask.trailing_zeros() as u8;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		slot_val[s as usize] = levels[d].values.get(base + rank);
		if !is_leaf_level && (leaf >> s) & 1 == 0 {
			slot_child[s as usize] = levels[d].node_children.get(base + rank);
		}
		mask &= mask - 1;
	}

	// Phase 2: process each slot, recursing into deeper levels as needed.
	// Results stored in arrays; levels[d] not touched until phase 3.
	let mut res_val   = [0u32; 64];
	let mut res_child = [0u32; 64];
	let mut res_leaf  = [false; 64];
	let mut res_occ   = [false; 64];

	let mut ei = 0;
	for s in 0u8..64 {
		let start = ei;
		while ei < edits.len() && edits[ei].0[d] == s { ei += 1; }
		let slot_edits = &edits[start..ei];

		let occupied  = (occ  >> s) & 1 != 0;
		let is_leaf_s = occupied && (leaf >> s) & 1 != 0;

		if slot_edits.is_empty() {
			if !occupied { continue; }
			res_val[s as usize]   = slot_val[s as usize];
			res_child[s as usize] = slot_child[s as usize];
			res_leaf[s as usize]  = is_leaf_s || is_leaf_level;
			res_occ[s as usize]   = true;
			continue;
		}

		// A depth-(d+1) edit targets this exact slot — take the last one.
		if let Some(&(_, _, val)) = slot_edits.iter().rfind(|(_, ed, _)| *ed == d + 1) {
			if val != DELETE {
				res_val[s as usize]  = val;
				res_leaf[s as usize] = true;
				res_occ[s as usize]  = true;
			}
			continue;
		}

		if is_leaf_level { continue; } // sub-voxel edits at leaf level don't apply

		// Descend: expand slot into a child node if needed, then recurse.
		let child = if occupied && !is_leaf_s {
			slot_child[s as usize]
		} else if is_leaf_s {
			alloc_expanded(levels, d + 1, slot_val[s as usize])
		} else {
			alloc_empty(levels, d + 1)
		};

		let new_child = rebuild::<DEPTH>(levels, d + 1, child, slot_edits);

		if levels[d + 1].occupancy_mask[new_child as usize] == 0 {
			continue; // child became empty — drop this slot
		}

		// Collapse uniform child back to a leaf slot.
		if let Some(uval) = uniform_leaf_value(&levels[d + 1], new_child) {
			res_val[s as usize]  = uval;
			res_leaf[s as usize] = true;
			res_occ[s as usize]  = true;
		} else {
			res_val[s as usize]   = lod_value(&levels[d + 1], new_child);
			res_child[s as usize] = new_child;
			res_leaf[s as usize]  = false;
			res_occ[s as usize]   = true;
		}
	}

	// Phase 3: append the new node to levels[d].
	let new_node = levels[d].node_count();
	let new_offset = levels[d].children_len();
	let mut new_occ  = 0u64;
	let mut new_leaf = 0u64;

	for s in 0u8..64 {
		if res_occ[s as usize] {
			levels[d].push_child(res_child[s as usize], res_val[s as usize]);
			new_occ |= 1u64 << s;
			if res_leaf[s as usize] { new_leaf |= 1u64 << s; }
		}
	}

	levels[d].push_node(new_occ, new_leaf, new_offset);
	new_node
}
