use ray_vox::World;
use ray_vox::chunk::material::Material;
use ray_vox::generate::volume::terrain::Terrain;
use ray_vox::util::types::{LodLevel, WorldPos};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let mut args = std::env::args().skip(1);
	let base_height: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);
	let amplitude: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);
	let octaves: u8 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
	let base_period: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(100_000);

	let white = Material::from_rgb_pbr_id([255, 255, 255], 0);
	let terrain = Terrain::new(0xC0FFEE, base_height, amplitude, octaves, base_period, white);

	println!("terrain params: base_height={}, amplitude={}, octaves={}, base_period={}",
		base_height, amplitude, octaves, base_period);

	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));

	let t = Instant::now();
	world.apply_edit(Arc::new(terrain));
	println!("\napply_edit: {:?}", t.elapsed());

	let t = Instant::now();
	world.drive_remaps(usize::MAX);
	let elapsed = t.elapsed();
	let bake_count = world.clipmap.pending_remap.len()
		+ world.clipmap.resident.len();
	println!("drive_remaps: {:?}", elapsed);
	println!("  resident chunks: {}", world.clipmap.resident.len());
	println!("  chunks/sec: {:.1}",
		world.clipmap.resident.len() as f64 / elapsed.as_secs_f64());
	let _ = bake_count;

	let total_bytes: u64 = world.chunk_pool.chunks.values().map(|c| c.byte_size() as u64).sum();
	println!("\nresident: {} chunks, {} bytes ({:.2} MB)",
		world.clipmap.resident.len(),
		total_bytes,
		total_bytes as f64 / 1e6,
	);

	let mut per_lod: Vec<(usize, u64)> = vec![(0, 0); LodLevel::LEVELS as usize];
	for (handle, id) in world.clipmap.resident.iter() {
		let l = id.lod.level() as usize;
		per_lod[l].0 += 1;
		if let Some(chunk) = world.chunk_pool.chunks.get(handle) {
			per_lod[l].1 += chunk.byte_size() as u64;
		}
	}
	println!("\nper-LOD breakdown:");
	for (l, (count, bytes)) in per_lod.iter().enumerate() {
		if *count > 0 {
			println!("  LOD {:>2}: {:>5} chunks, {:>11} bytes ({:>6.2} MB)",
				l, count, bytes, *bytes as f64 / 1e6,
			);
		}
	}

	let active_per_lod: Vec<u8> = (0..LodLevel::LEVELS)
		.map(|l| {
			let vs = LodLevel::new(l).voxel_size();
			let mut active = 0u8;
			let mut amp = amplitude / 2;
			while active < octaves && amp >= vs {
				active += 1;
				amp /= 2;
			}
			active
		})
		.collect();
	println!("\nactive octaves per LOD:");
	for (l, a) in active_per_lod.iter().enumerate() {
		println!("  LOD {:>2} (voxel_size={:>10}): {} active octaves",
			l, LodLevel::new(l as u8).voxel_size(), a);
	}

	Ok(())
}
