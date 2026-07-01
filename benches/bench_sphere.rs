use ray_vox::chunk::build;
use ray_vox::chunk::material::Material;
use ray_vox::generate::volume::sphere::Sphere;
use ray_vox::generate::Edit;
use ray_vox::util::types::{ChunkId, WorldPos};
use ray_vox::Chunk;
use std::time::{Duration, Instant};

const ITERS: usize = 25;
const RADIUS: i32 = 128;

fn summarize(name: &str, mut samples: Vec<Duration>) {
	samples.sort();
	let min = samples[0];
	let med = samples[samples.len() / 2];
	let max = *samples.last().unwrap();
	let sum: Duration = samples.iter().sum();
	let mean = sum / samples.len() as u32;
	println!(
		"  {:<22} min={:>10.3?}  median={:>10.3?}  mean={:>10.3?}  max={:>10.3?}",
		name, min, med, mean, max,
	);
}

fn main() {
	let stone = Material::from(0x80808040);
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let center = WorldPos::new(128, 128, 128);

	let mut t_build = Vec::with_capacity(ITERS);
	for _ in 0..ITERS {
		let s = Instant::now();
		let baked = Sphere::new(center, RADIUS, stone).apply(chunk_id, Chunk::new());
		t_build.push(s.elapsed());
		std::hint::black_box(baked);
	}

	let mut t_direct = Vec::with_capacity(ITERS);
	for _ in 0..ITERS {
		let local = Sphere::new(center, RADIUS, stone).local(chunk_id).unwrap();
		let s = Instant::now();
		let baked = build::build_chunk(&local);
		t_direct.push(s.elapsed());
		std::hint::black_box(baked);
	}

	println!("sphere r={} into empty 256³ chunk, {} iterations", RADIUS, ITERS);
	summarize("Sphere::apply (overlay)", t_build);
	summarize("build_chunk (no base)", t_direct);

	let c = Sphere::new(center, RADIUS, stone).apply(chunk_id, Chunk::new());
	let interior_bytes = c.interior_nodes.len() * std::mem::size_of_val(&c.interior_nodes[0]);
	let leaf_bytes = c.leaf_nodes.len() * std::mem::size_of_val(&c.leaf_nodes[0]);
	let lut_bytes = c.materials.lut.values.len() * std::mem::size_of::<Material>();
	let idx_bytes = c.materials.indices.words.len() * 4;
	println!(
		"\nstorage: gpu={} interiors={}({}b) leaves={}({}b) lut={}({}b) indices={}({}b @ {}bit) stored_vol={}",
		c.byte_size(),
		c.interior_nodes.len(), interior_bytes,
		c.leaf_nodes.len(), leaf_bytes,
		c.materials.lut.values.len(), lut_bytes,
		c.materials.indices.len(), idx_bytes, c.materials.indices.bits,
		c.stored_volume(),
	);
}
