use ray_vox::world::World;
use std::io::BufWriter;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let args: Vec<String> = std::env::args().collect();
	if args.len() < 3 {
		eprintln!("usage: {} <input.vox> <output.rvox>", args[0]);
		std::process::exit(1);
	}
	let input_path = &args[1];
	let output_path = &args[2];

	println!("reading {}...", input_path);
	let start = Instant::now();
	let bytes = std::fs::read(input_path)?;
	println!("  {} bytes in {:?}", bytes.len(), start.elapsed());

	let vox_data = dot_vox::load_bytes(&bytes).map_err(|e| anyhow::anyhow!("parse failed: {}", e))?;
	let source_voxel_count: usize = vox_data.models.iter().map(|m| m.voxels.len()).sum();
	let source_distinct_rgb = {
		let mut colors = std::collections::HashSet::new();
		for c in &vox_data.palette {
			colors.insert((c.r, c.g, c.b));
		}
		colors.len()
	};
	println!(
		"  source: {} voxels across {} models, palette has {} distinct RGB values ({} entries total)",
		source_voxel_count, vox_data.models.len(), source_distinct_rgb, vox_data.palette.len(),
	);

	println!("\nimporting...");
	let start = Instant::now();
	let world = ray_vox::import::vox::import_vox(&bytes)
		.map_err(|e| anyhow::anyhow!("import failed: {}", e))?;
	let non_empty = world.chunks.iter().filter(|c| c.is_some()).count();
	println!(
		"  {} non-empty chunks (grid {:?}) in {:?}",
		non_empty, world.chunk_grid_dim, start.elapsed(),
	);

	let imported_voxels: u64 = world.chunks.iter()
		.filter_map(|c| c.as_ref())
		.map(|c| c.stored_volume())
		.sum();
	let distinct_materials: usize = world.chunks.iter()
		.filter_map(|c| c.as_ref())
		.flat_map(|c| c.materials.lut.values.iter().copied())
		.collect::<std::collections::HashSet<_>>()
		.len();
	println!(
		"  imported: {} non-air voxels, {} distinct materials across all chunks",
		imported_voxels, distinct_materials,
	);
	if (imported_voxels as usize) != source_voxel_count {
		let diff = (imported_voxels as i64) - (source_voxel_count as i64);
		println!("  WARNING: voxel-count mismatch ({:+})", diff);
	}

	println!(
		"\nworld origin: {:?}, grid: {} x {} x {} chunks",
		<[i32; 3]>::from(world.origin),
		world.chunk_grid_dim[0], world.chunk_grid_dim[1], world.chunk_grid_dim[2],
	);

	println!("\nwriting {}...", output_path);
	let start = Instant::now();
	let file = std::fs::File::create(output_path)?;
	let mut writer = BufWriter::new(file);
	world.save_rvox(&mut writer)
		.map_err(|e| anyhow::anyhow!("save failed: {}", e))?;
	drop(writer);
	let output_size = std::fs::metadata(output_path)?.len();
	println!("  {} bytes in {:?}", output_size, start.elapsed());

	println!("\nverifying round-trip...");
	let start = Instant::now();
	let bytes = std::fs::read(output_path)?;
	let mut cursor = std::io::Cursor::new(&bytes);
	let reloaded = World::load_rvox(&mut cursor)
		.map_err(|e| anyhow::anyhow!("reload failed: {}", e))?;
	let reloaded_non_empty = reloaded.chunks.iter().filter(|c| c.is_some()).count();
	assert_eq!(reloaded_non_empty, non_empty, "chunk count mismatch");
	println!("  loaded {} chunks in {:?}, counts match", reloaded_non_empty, start.elapsed());

	Ok(())
}
