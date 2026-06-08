// Per-stage timing for placing one r=128 sphere into an empty chunk.
// Runs N iterations and prints min/median/max per stage.

use ray_vox::chunk::compact;
use ray_vox::chunk::material::Material;
use ray_vox::generate::volume::sphere::Sphere;
use ray_vox::generate::Edit;
use ray_vox::util::types::{ChunkId, LodLevel, WorldPos};
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
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let center = WorldPos::new(128, 128, 128);

	let mut t_emit = Vec::with_capacity(ITERS);
	let mut t_into_mutable = Vec::with_capacity(ITERS);
	let mut t_sort = Vec::with_capacity(ITERS);
	let mut t_apply = Vec::with_capacity(ITERS);
	let mut t_compress = Vec::with_capacity(ITERS);
	let mut t_total = Vec::with_capacity(ITERS);
	let mut edit_count = 0usize;

	for _ in 0..ITERS {
		let chunk = Chunk::new();

		let total_start = Instant::now();

		let s = Instant::now();
		let mut packet = Sphere::new(center, RADIUS, stone).sample(chunk_id);
		t_emit.push(s.elapsed());
		edit_count = packet.edits.len();

		let s = Instant::now();
		let mut mc = chunk.into_mutable();
		t_into_mutable.push(s.elapsed());

		let s = Instant::now();
		packet.sort();
		t_sort.push(s.elapsed());

		let s = Instant::now();
		mc.apply_batch(&packet.edits);
		t_apply.push(s.elapsed());

		let s = Instant::now();
		let baked = compact::compress(mc);
		t_compress.push(s.elapsed());

		t_total.push(total_start.elapsed());
		std::hint::black_box(baked);
	}

	println!(
		"sphere r={} ({} edits), {} iterations",
		RADIUS, edit_count, ITERS
	);
	summarize("emit packet", t_emit);
	summarize("into_mutable", t_into_mutable);
	summarize("packet.sort", t_sort);
	summarize("apply_batch", t_apply);
	summarize("compact::compress", t_compress);
	summarize("TOTAL", t_total);

	// Storage breakdown for one baked chunk.
	let packet = Sphere::new(center, RADIUS, stone).sample(chunk_id);
	let mut mc = Chunk::new().into_mutable();
	mc.queue_edit(packet);
	let c = mc.bake();
	let interior_bytes = c.interior_nodes.len() * std::mem::size_of_val(&c.interior_nodes[0]);
	let leaf_bytes = c.leaf_nodes.len() * std::mem::size_of_val(&c.leaf_nodes[0]);
	let lut_bytes = c.materials.lut.values.len() * std::mem::size_of::<Material>();
	let idx_bytes = c.materials.indices.words.len() * 4;
	println!(
		"\nstorage: gpu={} interiors={}({}b) leaves={}({}b) lut={}({}b) indices={}({}b @ {}bit) stored_vol={}",
		c.gpu_size_bytes(),
		c.interior_nodes.len(), interior_bytes,
		c.leaf_nodes.len(), leaf_bytes,
		c.materials.lut.values.len(), lut_bytes,
		c.materials.indices.len(), idx_bytes, c.materials.indices.bits,
		c.stored_volume(),
	);
}
