use ray_vox::generate::model::Model;
use ray_vox::util::types::LodLevel;
use std::io::BufWriter;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let args: Vec<String> = std::env::args().collect();
	if args.len() < 3 {
		eprintln!("usage: {} <input.vox> <output.rvox>", args[0]);
		std::process::exit(1);
	}
	let input = &args[1];
	let output = &args[2];

	println!("reading {}...", input);
	let t = Instant::now();
	let bytes = std::fs::read(input)?;
	println!("  {} bytes in {:?}", bytes.len(), t.elapsed());

	let vox_data = dot_vox::load_bytes(&bytes).map_err(|e| anyhow::anyhow!("parse failed: {}", e))?;
	let src_voxel_count: usize = vox_data.models.iter().map(|m| m.voxels.len()).sum();
	let src_palette_distinct_rgb = {
		let mut s = std::collections::HashSet::new();
		for c in &vox_data.palette {
			s.insert((c.r, c.g, c.b));
		}
		s.len()
	};
	println!("  source: {} voxels across {} models, palette has {} distinct RGB values ({} entries total)",
		src_voxel_count, vox_data.models.len(), src_palette_distinct_rgb, vox_data.palette.len());

	println!("\nimporting (voxelize + mip pyramid)...");
	let t = Instant::now();
	let model = ray_vox::import::vox::import_vox(&bytes)
		.map_err(|e| anyhow::anyhow!("import failed: {}", e))?;
	let elapsed = t.elapsed();
	println!("  {} chunks across all LODs in {:?}", model.chunk_count(), elapsed);

	let imported_finest_voxels: u64 = model
		.chunks
		.iter()
		.filter(|(id, _)| id.lod == LodLevel::FINEST)
		.map(|(_, c)| c.stored_volume())
		.sum();
	let imported_distinct_materials: usize = model
		.chunks
		.iter()
		.filter(|(id, _)| id.lod == LodLevel::FINEST)
		.flat_map(|(_, c)| c.materials.lut.values.iter().copied())
		.collect::<std::collections::HashSet<_>>()
		.len();
	println!(
		"  imported (finest LOD): {} non-air voxels, {} distinct materials across all chunks",
		imported_finest_voxels, imported_distinct_materials,
	);
	if (imported_finest_voxels as usize) != src_voxel_count {
		let diff = (imported_finest_voxels as i64) - (src_voxel_count as i64);
		println!("  WARNING: voxel-count mismatch ({:+})", diff);
	}

	println!("\nchunks per LOD:");
	for level in 0..LodLevel::LEVELS {
		let lod = LodLevel::new(level);
		let count = model.chunks_at_lod(lod).count();
		if count > 0 {
			println!("  LOD {:>2} (chunk_size={:>11}): {} chunks", level, lod.chunk_size(), count);
		}
	}
	println!("\nbounds: min={:?} max={:?}",
		<[i32; 3]>::from(model.bounds.min),
		<[i32; 3]>::from(model.bounds.max),
	);

	println!("\nwriting {}...", output);
	let t = Instant::now();
	let file = std::fs::File::create(output)?;
	let mut writer = BufWriter::new(file);
	model.save_rvox(&mut writer)
		.map_err(|e| anyhow::anyhow!("save failed: {}", e))?;
	drop(writer);
	let elapsed = t.elapsed();
	let out_size = std::fs::metadata(output)?.len();
	println!("  {} bytes in {:?}", out_size, elapsed);

	println!("\nverifying round-trip...");
	let t = Instant::now();
	let bytes = std::fs::read(output)?;
	let mut cursor = std::io::Cursor::new(&bytes);
	let reloaded = Model::load_rvox(&mut cursor)
		.map_err(|e| anyhow::anyhow!("reload failed: {}", e))?;
	assert_eq!(reloaded.chunk_count(), model.chunk_count(), "chunk count mismatch");
	println!("  loaded {} chunks in {:?}, counts match", reloaded.chunk_count(), t.elapsed());

	Ok(())
}
