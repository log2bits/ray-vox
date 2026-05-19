use crate::chunk::Chunk;

/// GPU buffer layout (all u32 words, no header — metadata goes in uniforms):
///
/// For each level d in 0..4:
///   node_count[d] words : occupancy_lo
///   node_count[d] words : occupancy_hi
///   node_count[d] words : leaf_lo
///   node_count[d] words : leaf_hi
///   node_count[d] words : children_offset
///   slot_count[d] words : values        (bitpacked; slot_count = word count of packed data)
///   nc_word_count[d] words : node_children (bitpacked; nc_word_count = word count)
///
/// After all levels: material table [u32; material_count]  (raw Voxel values)
pub fn serialize_chunk(chunk: &Chunk) -> (Vec<u32>, ChunkMeta) {
	let tree = &chunk.tree;
	let mut buf = Vec::<u32>::new();

	let mut node_counts = [0u32; 4];
	let mut slot_counts = [0u32; 4];
	let mut level_offsets = [0u32; 4];
	let mut values_bits = [0u32; 4];
	let mut nc_bits = [0u32; 4];

	for d in 0..4 {
		let level = &tree.levels[d];
		let nc = level.node_count() as usize;

		node_counts[d] = nc as u32;
		slot_counts[d] = level.values.data.len() as u32;
		values_bits[d] = level.values.bits as u32;
		nc_bits[d] = level.node_children.bits as u32;
		level_offsets[d] = buf.len() as u32;

		for n in 0..nc {
			buf.push(level.occupancy_mask[n] as u32);
		}
		for n in 0..nc {
			buf.push((level.occupancy_mask[n] >> 32) as u32);
		}
		for n in 0..nc {
			buf.push(level.leaf_mask[n] as u32);
		}
		for n in 0..nc {
			buf.push((level.leaf_mask[n] >> 32) as u32);
		}
		for n in 0..nc {
			buf.push(level.children_offset[n]);
		}
		buf.extend_from_slice(&level.values.data);
		buf.extend_from_slice(&level.node_children.data);
	}

	let material_offset = buf.len() as u32;
	let material_count = chunk.materials.values.len() as u32;
	for v in &chunk.materials.values {
		buf.push(u32::from(*v));
	}

	// Pad to 4-word alignment so wgpu buffer copies are happy.
	while buf.len() % 4 != 0 {
		buf.push(0);
	}

	let meta = ChunkMeta {
		node_counts,
		slot_counts,
		level_offsets,
		values_bits,
		nc_bits,
		material_count,
		material_offset,
		tree_occupied: tree.occupied as u32,
		tree_is_leaf: tree.is_leaf as u32,
		tree_leaf_value: tree.value,
	};

	(buf, meta)
}

#[derive(Clone, Copy)]
pub struct ChunkMeta {
	pub node_counts: [u32; 4],
	pub slot_counts: [u32; 4],
	pub level_offsets: [u32; 4],
	pub values_bits: [u32; 4],
	pub nc_bits: [u32; 4],
	pub material_count: u32,
	pub material_offset: u32,
	pub tree_occupied: u32,
	pub tree_is_leaf: u32,
	pub tree_leaf_value: u32,
}
