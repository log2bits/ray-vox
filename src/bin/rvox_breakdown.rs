use ray_vox::world::World;

const FILE_HEADER_BYTES: u64 = 4 + 4 + 12 + 12 + 4;
const CHUNK_HEADER_BYTES: u64 = 12 + 8 + 4 + 12;
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
	let world = World::load_rvox(&mut cursor)
		.map_err(|e| anyhow::anyhow!("load failed: {}", e))?;

	let mut interior_bytes: u64 = 0;
	let mut leaf_bytes: u64 = 0;
	let mut palette_bytes: u64 = 0;
	let mut indices_bytes: u64 = 0;
	let mut per_chunk_meta_bytes: u64 = 0;
	let mut non_empty_chunks: u64 = 0;

	let mut actual_material_entries: u64 = 0;
	let mut naive_material_entries: u64 = 0;
	let mut naive_indices_bytes: u64 = 0;

	for chunk in world.chunks.iter().filter_map(|c| c.as_ref()) {
		non_empty_chunks += 1;
		let interior = INTERIOR_NODE_BYTES * chunk.interior_nodes.len() as u64;
		let leaf = LEAF_NODE_BYTES * chunk.leaf_nodes.len() as u64;
		let palette = MATERIAL_ENTRY_BYTES * chunk.materials.lut.values.len() as u64;
		let indices = WORD_BYTES * chunk.materials.indices.words.len() as u64;

		interior_bytes += interior;
		leaf_bytes += leaf;
		palette_bytes += palette;
		indices_bytes += indices;
		per_chunk_meta_bytes += CHUNK_HEADER_BYTES;

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
	}

	let metadata_bytes = FILE_HEADER_BYTES + per_chunk_meta_bytes;
	let materials_bytes = palette_bytes + indices_bytes;
	let accounted = metadata_bytes + interior_bytes + leaf_bytes + materials_bytes;

	println!("on-disk file size: {} bytes ({:.2} MB)", on_disk, on_disk as f64 / 1_048_576.0);
	println!("accounted total:   {} bytes ({:.2} MB)", accounted, accounted as f64 / 1_048_576.0);
	if accounted != on_disk {
		println!("  (mismatch: {} bytes)", on_disk as i64 - accounted as i64);
	}
	println!(
		"chunks: {} non-empty (grid {} x {} x {})\n",
		non_empty_chunks,
		world.chunk_grid_dim[0], world.chunk_grid_dim[1], world.chunk_grid_dim[2],
	);

	let total = accounted as f64;
	let pct = |bytes: u64| 100.0 * bytes as f64 / total;

	println!("breakdown:");
	println!(
		"  metadata        {:>12} bytes  {:>6.2}%   (file hdr {} + per-chunk hdr/pos {})",
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

	Ok(())
}
