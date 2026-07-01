use ray_vox::chunk::{Child, Chunk};
use ray_vox::generate::model::Model;

const FILE_HEADER_BYTES: u64 = 4 + 4 + 12 + 12 + 4;
const CHUNK_ID_BYTES: u64 = 12 + 4;
const CHUNK_NODE_HEADER_BYTES: u64 = 8;
const PALETTE_HEADER_BYTES: u64 = 4;
const PACKED_HEADER_BYTES: u64 = 12;
const INTERIOR_NODE_BYTES: u64 = 24;
const LEAF_NODE_BYTES: u64 = 12;
const MATERIAL_ENTRY_BYTES: u64 = 4;
const WORD_BYTES: u64 = 4;

fn main() -> anyhow::Result<()> {
	let path = std::env::args().nth(1).unwrap_or_else(|| "assets/castle.rvox".to_string());
	println!("loading {}", path);

	let bytes = std::fs::read(&path)?;
	let on_disk = bytes.len() as u64;
	let mut cursor = std::io::Cursor::new(&bytes);
	let model = Model::load_rvox(&mut cursor)
		.map_err(|e| anyhow::anyhow!("load failed: {}", e))?;

	let mut interior_bytes: u64 = 0;
	let mut leaf_bytes: u64 = 0;
	let mut palette_bytes: u64 = 0;
	let mut indices_bytes: u64 = 0;
	let mut per_chunk_meta_bytes: u64 = 0;

	let mut actual_material_entries: u64 = 0;
	let mut naive_material_entries: u64 = 0;
	let mut naive_indices_bytes: u64 = 0;

	let mut actual_interior_nodes: u64 = 0;
	let mut actual_leaf_nodes: u64 = 0;
	let mut naive_interior_refs: u64 = 0;
	let mut naive_leaf_refs: u64 = 0;

	for chunk in model.chunks.values() {
		let interior = INTERIOR_NODE_BYTES * chunk.interior_nodes.len() as u64;
		let leaf = LEAF_NODE_BYTES * chunk.leaf_nodes.len() as u64;
		let palette = MATERIAL_ENTRY_BYTES * chunk.materials.lut.values.len() as u64;
		let indices = WORD_BYTES * chunk.materials.indices.words.len() as u64;
		let meta = CHUNK_ID_BYTES + CHUNK_NODE_HEADER_BYTES + PALETTE_HEADER_BYTES + PACKED_HEADER_BYTES;

		interior_bytes += interior;
		leaf_bytes += leaf;
		palette_bytes += palette;
		indices_bytes += indices;
		per_chunk_meta_bytes += meta;

		actual_material_entries += chunk.materials.indices.len as u64;

		let mut chunk_naive_entries: u64 = 0;
		for node in &chunk.interior_nodes {
			chunk_naive_entries += node.masks.filled().count() as u64;
		}
		for leaf in &chunk.leaf_nodes {
			chunk_naive_entries += leaf.occupancy.count() as u64;
		}
		naive_material_entries += chunk_naive_entries;

		let bits = chunk.materials.indices.bits as u64;
		let naive_words = (chunk_naive_entries * bits + 31) / 32;
		naive_indices_bytes += naive_words * 4;

		actual_interior_nodes += chunk.interior_nodes.len() as u64;
		actual_leaf_nodes += chunk.leaf_nodes.len() as u64;
		let (interior_refs, leaf_refs) = count_tree_refs(chunk);
		naive_interior_refs += interior_refs;
		naive_leaf_refs += leaf_refs;
	}

	let metadata_bytes = FILE_HEADER_BYTES + per_chunk_meta_bytes;
	let materials_bytes = palette_bytes + indices_bytes;
	let accounted = metadata_bytes + interior_bytes + leaf_bytes + materials_bytes;

	println!("on-disk file size: {} bytes ({:.2} MB)", on_disk, on_disk as f64 / 1_048_576.0);
	println!("accounted total:   {} bytes ({:.2} MB)", accounted, accounted as f64 / 1_048_576.0);
	if accounted != on_disk {
		println!("  (mismatch: {} bytes)", on_disk as i64 - accounted as i64);
	}
	println!("chunks: {}\n", model.chunks.len());

	let total = accounted as f64;
	let pct = |bytes: u64| 100.0 * bytes as f64 / total;

	println!("breakdown:");
	println!(
		"  metadata        {:>12} bytes  {:>6.2}%   (file hdr {} + per-chunk hdr/id {})",
		metadata_bytes, pct(metadata_bytes), FILE_HEADER_BYTES, per_chunk_meta_bytes,
	);
	println!("  interior nodes  {:>12} bytes  {:>6.2}%", interior_bytes, pct(interior_bytes));
	println!("  leaf nodes      {:>12} bytes  {:>6.2}%", leaf_bytes, pct(leaf_bytes));
	println!(
		"  materials       {:>12} bytes  {:>6.2}%   (palette {} + bitpacked indices {})",
		materials_bytes, pct(materials_bytes), palette_bytes, indices_bytes,
	);
	println!(
		"    palette LUT     {:>12} bytes  {:>6.2}%",
		palette_bytes, pct(palette_bytes),
	);
	println!(
		"    indices         {:>12} bytes  {:>6.2}%",
		indices_bytes, pct(indices_bytes),
	);

	let saved_entries = naive_material_entries.saturating_sub(actual_material_entries);
	let saved_bytes = naive_indices_bytes.saturating_sub(indices_bytes);
	let naive_file = on_disk + saved_bytes;
	println!("\nexact-run material dedup:");
	println!(
		"  material entries:  {} actual vs {} naive  ({:.2}x sharing, {:.2}% removed)",
		actual_material_entries,
		naive_material_entries,
		naive_material_entries as f64 / actual_material_entries.max(1) as f64,
		100.0 * saved_entries as f64 / naive_material_entries.max(1) as f64,
	);
	println!(
		"  indices bytes:     {} actual vs {} naive  (saved {} bytes = {:.2} MB)",
		indices_bytes,
		naive_indices_bytes,
		saved_bytes,
		saved_bytes as f64 / 1_048_576.0,
	);
	println!(
		"  file size:         {:.2} MB actual vs {:.2} MB without run dedup  ({:.2}x reduction overall)",
		on_disk as f64 / 1_048_576.0,
		naive_file as f64 / 1_048_576.0,
		naive_file as f64 / on_disk.max(1) as f64,
	);

	let naive_interior_bytes = naive_interior_refs * INTERIOR_NODE_BYTES;
	let naive_leaf_bytes = naive_leaf_refs * LEAF_NODE_BYTES;
	let saved_interior_bytes = naive_interior_bytes.saturating_sub(interior_bytes);
	let saved_leaf_bytes = naive_leaf_bytes.saturating_sub(leaf_bytes);
	let saved_node_bytes = saved_interior_bytes + saved_leaf_bytes;
	let naive_file_with_node_dag = on_disk + saved_node_bytes;
	println!("\nnode-level DAG dedup (subtree sharing):");
	println!(
		"  interior nodes: {} unique vs {} references  ({:.2}x sharing, {:.2}% removed)",
		actual_interior_nodes,
		naive_interior_refs,
		naive_interior_refs as f64 / actual_interior_nodes.max(1) as f64,
		100.0 * (naive_interior_refs - actual_interior_nodes) as f64 / naive_interior_refs.max(1) as f64,
	);
	println!(
		"  leaf nodes:     {} unique vs {} references  ({:.2}x sharing, {:.2}% removed)",
		actual_leaf_nodes,
		naive_leaf_refs,
		naive_leaf_refs as f64 / actual_leaf_nodes.max(1) as f64,
		100.0 * (naive_leaf_refs - actual_leaf_nodes) as f64 / naive_leaf_refs.max(1) as f64,
	);
	println!(
		"  node bytes:     {} actual vs {} naive  (saved {:.2} MB)",
		interior_bytes + leaf_bytes,
		naive_interior_bytes + naive_leaf_bytes,
		saved_node_bytes as f64 / 1_048_576.0,
	);
	println!(
		"  file size:      {:.2} MB actual vs {:.2} MB without node DAG  ({:.2}x reduction overall)",
		on_disk as f64 / 1_048_576.0,
		naive_file_with_node_dag as f64 / 1_048_576.0,
		naive_file_with_node_dag as f64 / on_disk.max(1) as f64,
	);

	Ok(())
}

fn count_tree_refs(chunk: &Chunk) -> (u64, u64) {
	let mut interior_refs = 0u64;
	let mut leaf_refs = 0u64;
	match chunk.root_child() {
		Child::Empty | Child::Filled(_) => {}
		Child::Leaf(_) => leaf_refs += 1,
		Child::Interior(idx) => {
			interior_refs += 1;
			walk(chunk, idx, &mut interior_refs, &mut leaf_refs);
		}
	}
	(interior_refs, leaf_refs)
}

fn walk(chunk: &Chunk, idx: u32, interior_refs: &mut u64, leaf_refs: &mut u64) {
	for slot in chunk.interior_nodes[idx as usize].masks.occupancy().iter_slots() {
		match chunk.child(idx, slot) {
			Child::Empty | Child::Filled(_) => {}
			Child::Leaf(_) => *leaf_refs += 1,
			Child::Interior(c) => {
				*interior_refs += 1;
				walk(chunk, c, interior_refs, leaf_refs);
			}
		}
	}
}
