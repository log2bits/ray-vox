pub struct InteriorNode {
	pub has_child: u64,
	pub is_leaf: u64,
	pub node_offsets: u32,
	pub material_offset: u32,
}

pub struct LeafNode {
	pub occupancy: u64,
	pub material_offset: u32,
}
