pub struct InteriorNode {
	has_child: u64,
	is_leaf: u64,
	node_offsets: u32,
	material_offset: u32,
}

pub struct LeafNode {
	occupancy: u64,
	material_offset: u32,
}
