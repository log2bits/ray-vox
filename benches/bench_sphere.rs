use ray_vox::chunk::edit::{EditPacket, Path};
use ray_vox::chunk::material::Material;
use ray_vox::chunk::sources::DiscreteSource;
use ray_vox::generate::volume::sphere::Sphere;
use ray_vox::generate::Edit;
use ray_vox::util::types::{ChunkId, ChunkPos, WorldPos};
use ray_vox::Chunk;
use std::time::{Duration, Instant};

const ITERS: usize = 30;

fn summarize(name: &str, mut samples: Vec<Duration>) {
	samples.sort();
	let min = samples[0];
	let median = samples[samples.len() / 2];
	let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
	let max = *samples.last().unwrap();
	println!(
		"  {:<28} min={:>8.3?}  median={:>8.3?}  mean={:>8.3?}  max={:>8.3?}",
		name, min, median, mean, max,
	);
}

fn bake_sphere_from_empty(radius: i32, material: Material) -> Chunk {
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let sphere = Sphere::new(WorldPos::new(128, 128, 128), radius, material);
	sphere.apply(chunk_id, Chunk::new())
}

fn stone() -> Material { Material::from(0x80808040) }
fn air() -> Material { Material::air() }

fn chunk_stats(name: &str, chunk: &Chunk) {
	let interior_bytes = chunk.interior_nodes.len() * std::mem::size_of_val(&chunk.interior_nodes[0]);
	let leaf_bytes = chunk.leaf_nodes.len() * std::mem::size_of_val(&chunk.leaf_nodes[0]);
	let palette_bytes = chunk.materials.lut.values.len() * std::mem::size_of::<Material>();
	let indices_bytes = chunk.materials.indices.words.len() * 4;
	let total = chunk.byte_size() as usize;
	let voxels = chunk.stored_volume();
	let bytes_per_voxel = if voxels > 0 { total as f64 / voxels as f64 } else { 0.0 };
	println!(
		"  {:<28} {:>10} B  ({:>7} voxels, {:>6.4} B/voxel)  int={} leaf={} pal={}B idx={}B@{}bit",
		name,
		total,
		voxels,
		bytes_per_voxel,
		chunk.interior_nodes.len(),
		chunk.leaf_nodes.len(),
		palette_bytes,
		indices_bytes,
		chunk.materials.indices.bits,
	);
	let _ = interior_bytes;
	let _ = leaf_bytes;
}

fn bench_sphere_bake(radius: i32) {
	let mut times = Vec::with_capacity(ITERS);
	for _ in 0..ITERS {
		let start = Instant::now();
		let baked = bake_sphere_from_empty(radius, stone());
		times.push(start.elapsed());
		std::hint::black_box(baked);
	}
	summarize(&format!("sphere r={:>3} bake", radius), times);
}

fn bench_sphere_over_filled(radius: i32) {
	let mut edits = EditPacket::default();
	edits.push(Path::from(0u32), stone());
	edits.sort();
	let filled = Chunk::new().edit(&DiscreteSource::new(&edits.edits));

	let mut times = Vec::with_capacity(ITERS);
	for _ in 0..ITERS {
		let base = filled.clone();
		let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
		let start = Instant::now();
		let baked = Sphere::new(WorldPos::new(128, 128, 128), radius, air()).apply(chunk_id, base);
		times.push(start.elapsed());
		std::hint::black_box(baked);
	}
	summarize(&format!("sphere r={:>3} carve", radius), times);
}

fn bench_single_voxel_edits(count: usize) {
	let mut times = Vec::with_capacity(ITERS);
	for _ in 0..ITERS {
		let mut packet = EditPacket::default();
		let mut rng_state: u32 = 0xC0FFEEu32.wrapping_add(count as u32);
		for _ in 0..count {
			rng_state = rng_state.wrapping_mul(0x100_0193).wrapping_add(0xdead_beef);
			let x = (rng_state & 0xFF) as u8;
			let y = ((rng_state >> 8) & 0xFF) as u8;
			let z = ((rng_state >> 16) & 0xFF) as u8;
			packet.push(Path::from_coords(ChunkPos::new(x, y, z), 4), stone());
		}
		let start = Instant::now();
		packet.sort();
		let chunk = Chunk::new().edit(&DiscreteSource::new(&packet.edits));
		times.push(start.elapsed());
		std::hint::black_box(chunk);
	}
	summarize(&format!("{:>6} single-voxel edits", count), times);
}

fn main() {
	println!("=== CPU chunk edits (single 256^3 chunk, {} iterations) ===", ITERS);
	bench_sphere_bake(16);
	bench_sphere_bake(32);
	bench_sphere_bake(64);
	bench_sphere_bake(96);
	bench_sphere_bake(128);
	println!();
	bench_sphere_over_filled(32);
	bench_sphere_over_filled(64);
	bench_sphere_over_filled(128);
	println!();
	bench_single_voxel_edits(1_000);
	bench_single_voxel_edits(10_000);
	bench_single_voxel_edits(100_000);

	println!("\n=== Compression: sphere size vs stored volume ===");
	for &r in &[16, 32, 64, 96, 128] {
		let chunk = bake_sphere_from_empty(r, stone());
		chunk_stats(&format!("sphere r={:>3}", r), &chunk);
	}

	// Stone shell with an ember core stresses the material LUT past 1 bit.
	{
		let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
		let ember = Material::from_rgb_pbr_id([0xE0, 0x50, 0x30], 0);
		let shell = Sphere::new(WorldPos::new(128, 128, 128), 128, stone()).apply(chunk_id, Chunk::new());
		let core = Sphere::new(WorldPos::new(128, 128, 128), 64, ember).apply(chunk_id, shell);
		chunk_stats("stone+ember r=128 with core", &core);
	}
}
