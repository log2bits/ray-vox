use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let path = std::env::args().nth(1).unwrap_or_else(|| "assets/castle.vox".to_string());
	let runs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);

	println!("reading {}", path);
	let bytes = std::fs::read(&path)?;

	let t = Instant::now();
	let parsed = dot_vox::load_bytes(&bytes).map_err(|e| anyhow::anyhow!(e))?;
	let parse_ms = t.elapsed().as_secs_f64() * 1000.0;
	let total_voxels: usize = parsed.models.iter().map(|m| m.voxels.len()).sum();
	println!("parse: {:.3} ms ({} voxels across {} models)", parse_ms, total_voxels, parsed.models.len());

	println!("\nrunning {} import_vox iterations...", runs);
	let mut times_ms = Vec::with_capacity(runs);
	let mut last_chunks = 0;
	for i in 0..runs {
		let t = Instant::now();
		let model = ray_vox::import::vox::import_vox(&bytes).map_err(|e| anyhow::anyhow!(e))?;
		let ms = t.elapsed().as_secs_f64() * 1000.0;
		times_ms.push(ms);
		last_chunks = model.chunk_count();
		println!("  run {}: {:.3} ms ({} chunks)", i + 1, ms, last_chunks);
	}

	times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let min = times_ms[0];
	let median = times_ms[runs / 2];
	let mean = times_ms.iter().sum::<f64>() / runs as f64;
	println!(
		"\nimport_vox: min {:.3} ms, median {:.3} ms, mean {:.3} ms",
		min, median, mean,
	);
	println!(
		"chunk-build only (import_vox - parse): min {:.3} ms",
		min - parse_ms,
	);

	Ok(())
}
