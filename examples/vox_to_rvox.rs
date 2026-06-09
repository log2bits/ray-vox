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

	println!("\nimporting (voxelize + mip pyramid)...");
	let t = Instant::now();
	let model = Model::import_vox(&bytes)
		.map_err(|e| anyhow::anyhow!("import failed: {}", e))?;
	let elapsed = t.elapsed();
	println!("  {} chunks across all LODs in {:?}", model.chunk_count(), elapsed);

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
