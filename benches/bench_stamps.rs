use ray_vox::World;
use ray_vox::generate::model::Model;
use ray_vox::generate::model::stamp::ModelStamp;
use ray_vox::util::types::{ChunkHandle, LodLevel, WorldPos};
use ray_vox::world::clipmap::RemapOp;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let mut args = std::env::args().skip(1);
	let path = args.next().unwrap_or_else(|| "assets/castle.rvox".into());
	let grid_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);

	let bytes = std::fs::read(&path)?;
	let mut cursor = std::io::Cursor::new(&bytes);
	let model = Arc::new(
		Model::load_rvox(&mut cursor).map_err(|e| anyhow::anyhow!("{}", e))?
	);
	let extent = model.bounds.max;
	println!("loaded {}: {} chunks, extent {:?}",
		path,
		model.chunk_count(),
		<[i32; 3]>::from(extent),
	);

	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	world.drive_remaps(usize::MAX);

	let spacing_x = extent.x() + 256;
	let spacing_y = extent.y() + 256;
	let total = (grid_n * grid_n) as usize;

	println!("\nstamping {}×{} = {} copies, spacing ({}, {})",
		grid_n, grid_n, total, spacing_x, spacing_y,
	);

	let t = Instant::now();
	for gx in 0..grid_n {
		for gy in 0..grid_n {
			let pos = WorldPos::new(gx * spacing_x, gy * spacing_y, 0);
			world.apply_edit(Arc::new(ModelStamp::new(model.clone(), pos)));
		}
	}
	let apply_elapsed = t.elapsed();
	println!("  apply_edit × {}: {:?} ({:.1} µs/stamp)",
		total,
		apply_elapsed,
		apply_elapsed.as_micros() as f64 / total as f64,
	);
	let queued_ops = world.clipmap.pending_remap.len();
	let unique_handles: HashSet<ChunkHandle> = world.clipmap.pending_remap.iter()
		.filter_map(|op| match op {
			RemapOp::Add(h, _) => Some(*h),
			_ => None,
		})
		.collect();
	let bakes_attempted = unique_handles.len();
	println!("  pending_remap: {} ops ({} unique handles)", queued_ops, bakes_attempted);

	// Timing breakdown: pick one handle per LOD, time its solo bake.
	{
		let mut per_lod_handles: Vec<(LodLevel, ChunkHandle)> = Vec::new();
		let mut seen = HashSet::new();
		for op in &world.clipmap.pending_remap {
			if let RemapOp::Add(h, id) = op {
				if seen.insert(id.lod) {
					per_lod_handles.push((id.lod, *h));
				}
			}
		}
		per_lod_handles.sort_by_key(|(lod, _)| lod.level());
		println!("\nsolo bake timing per LOD (one chunk each, for reference):");
		for (lod, handle) in &per_lod_handles {
			let chunk_id = world.clipmap.resident.get(handle).copied()
				.or_else(|| {
					world.clipmap.pending_remap.iter().find_map(|op| {
						if let RemapOp::Add(h, id) = op {
							if h == handle { Some(*id) } else { None }
						} else { None }
					})
				});
			if let Some(id) = chunk_id {
				let aabb = id.aabb();
				let stamp_count = world.edits.iter().filter(|e| e.bounds().intersects(&aabb)).count();
				let arcs: Vec<_> = world.edits.iter()
					.filter(|e| e.bounds().intersects(&aabb))
					.cloned()
					.collect();
				let t = Instant::now();
				let chunk = ray_vox::world::bake_pure(id, &arcs);
				let elapsed = t.elapsed();
				println!("  LOD {:>2}: {:>7.2?}  ({} overlapping stamps, output {} bytes)",
					lod.level(), elapsed, stamp_count, chunk.byte_size(),
				);
			}
		}
	}

	let t = Instant::now();
	world.drive_remaps(usize::MAX);
	let bake_elapsed = t.elapsed();
	println!("\n  drive_remaps: {:?}", bake_elapsed);
	println!("  bakes attempted: {} ({:.1} chunks/sec)",
		bakes_attempted,
		bakes_attempted as f64 / bake_elapsed.as_secs_f64(),
	);
	println!("  bakes producing content (final resident): {} ({:.1} chunks/sec)",
		world.clipmap.resident.len(),
		world.clipmap.resident.len() as f64 / bake_elapsed.as_secs_f64(),
	);
	println!("  empty-filtered: {} ({:.1}%)",
		bakes_attempted - world.clipmap.resident.len(),
		100.0 * (bakes_attempted - world.clipmap.resident.len()) as f64 / bakes_attempted as f64,
	);

	let resident = world.clipmap.resident.len();
	let total_bytes: u64 = world.chunk_pool.chunks.values().map(|c| c.byte_size() as u64).sum();
	println!("\nresident: {} chunks, {} bytes ({:.2} MB)",
		resident, total_bytes, total_bytes as f64 / 1e6,
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

	let raw_per_stamp = 17_534_704u64;
	let raw_total = raw_per_stamp * total as u64;
	println!("\nvs naive (one full model per stamp): {} bytes ({:.2} GB), so {:.1}× smaller",
		raw_total,
		raw_total as f64 / 1e9,
		raw_total as f64 / total_bytes.max(1) as f64,
	);

	Ok(())
}
