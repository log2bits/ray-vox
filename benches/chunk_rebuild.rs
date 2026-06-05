use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ray_vox::chunk::edit::{EditPacket, Path};
use ray_vox::chunk::material::Material;
use ray_vox::generate::volume::sphere::Sphere;
use ray_vox::util::types::{ChunkId, LodLevel, WorldPos};
use ray_vox::Chunk;

fn chunk_at_origin() -> ChunkId {
	ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST)
}

fn sphere_packet(radius: i32, material: Material) -> EditPacket {
	Sphere::generate(radius, chunk_at_origin(), WorldPos::new(128, 128, 128), material)
}

fn bake_one(chunk: Chunk, packet: EditPacket) -> Chunk {
	let mut mc = chunk.into_mutable();
	mc.queue_edit(packet);
	mc.bake()
}

fn bench_apply_edits(c: &mut Criterion) {
	let stone = Material::from(0x80808040);
	let air = Material::air();

	let mut g = c.benchmark_group("small_workloads");
	g.throughput(Throughput::Elements(1));
	g.bench_function("fill_only", |b| {
		b.iter(|| {
			let mut p = EditPacket::default();
			p.push(Path::from(0u32), stone);
			black_box(bake_one(Chunk::new(), p))
		})
	});
	g.bench_function("fill_plus_small_carve", |b| {
		b.iter(|| {
			let mut fill = EditPacket::default();
			fill.push(Path::from(0u32), stone);
			let mid = bake_one(Chunk::new(), fill);
			black_box(bake_one(mid, sphere_packet(12, air)))
		})
	});
	g.finish();

	let mut g = c.benchmark_group("big_sphere");
	for &radius in &[32i32, 64, 96, 128] {
		let count = sphere_packet(radius, stone).len();
		g.throughput(Throughput::Elements(count as u64));

		g.bench_with_input(BenchmarkId::new("place", radius), &radius, |b, &r| {
			b.iter_batched(
				|| sphere_packet(r, stone),
				|p| black_box(bake_one(Chunk::new(), p)),
				criterion::BatchSize::SmallInput,
			)
		});

		g.bench_with_input(BenchmarkId::new("carve", radius), &radius, |b, &r| {
			let mut fill = EditPacket::default();
			fill.push(Path::from(0u32), stone);
			let prefilled = bake_one(Chunk::new(), fill);
			b.iter_batched(
				|| (prefilled.clone(), sphere_packet(r, air)),
				|(chunk, p)| black_box(bake_one(chunk, p)),
				criterion::BatchSize::SmallInput,
			)
		});
	}
	g.finish();

	let mut g = c.benchmark_group("emit");
	g.throughput(Throughput::Elements(1));
	g.bench_function("sphere_r128", |b| {
		b.iter(|| black_box(sphere_packet(128, stone)))
	});
	g.finish();
}

criterion_group!(benches, bench_apply_edits);
criterion_main!(benches);
