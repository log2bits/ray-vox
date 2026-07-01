use super::build::build_chunk;
use super::edit::{EditPacket, Path};
use super::material::Material;
use super::sources::{DiscreteSource, Overlay};
use super::Chunk;
use crate::generate::volume::sphere::Sphere;
use crate::generate::Edit;
use crate::util::types::{ChunkId, ChunkPos, WorldPos};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

fn mat(v: u32) -> Material {
	Material::from(v)
}

fn path_to_cube(path: Path) -> ([u8; 3], u8) {
	let (corner, depth) = path.to_coords();
	let side = 1u8 << (8u8.saturating_sub(2 * depth)).min(7);
	([corner.x(), corner.y(), corner.z()], side)
}

#[derive(Default, Clone)]
struct BruteChunk {
	voxels: HashMap<u32, Material>,
	background: Material,
}

fn pack(x: u8, y: u8, z: u8) -> u32 {
	(x as u32) | ((y as u32) << 8) | ((z as u32) << 16)
}

fn bake_one(chunk: Chunk, mut packet: EditPacket) -> Chunk {
	packet.sort();
	chunk.edit(&DiscreteSource::new(&packet.edits))
}

impl BruteChunk {
	fn voxel_at(&self, p: ChunkPos) -> Material {
		let key = pack(p.x(), p.y(), p.z());
		self.voxels.get(&key).copied().unwrap_or(self.background)
	}

	fn paint(&mut self, path: Path, m: Material) {
		let depth = path.depth();
		if depth == 0 {
			self.background = m;
			self.voxels.clear();
			return;
		}
		let ([x0, y0, z0], side) = path_to_cube(path);
		let end = |c: u8| c as u16 + side as u16;
		for x in x0 as u16..end(x0) {
			for y in y0 as u16..end(y0) {
				for z in z0 as u16..end(z0) {
					let k = pack(x as u8, y as u8, z as u8);
					if m == self.background {
						self.voxels.remove(&k);
					} else {
						self.voxels.insert(k, m);
					}
				}
			}
		}
	}
}

fn brute_apply(brute: &mut BruteChunk, mut edits: EditPacket) {
	edits.sort();
	for &(path, m) in &edits.edits {
		brute.paint(path, m);
	}
}

fn assert_match(chunk: &Chunk, brute: &BruteChunk, sample_points: &[ChunkPos]) {
	for &p in sample_points {
		let got = chunk.voxel_at(p);
		let want = brute.voxel_at(p);
		assert_eq!(
			got, want,
			"voxel_at({:?}, {:?}, {:?}) = {:?}, expected {:?}",
			p.x(),
			p.y(),
			p.z(),
			got,
			want,
		);
	}
}

fn sample_grid() -> Vec<ChunkPos> {
	let mut v = Vec::new();
	for x in (0..=255).step_by(16) {
		for y in (0..=255).step_by(16) {
			for z in (0..=255).step_by(16) {
				v.push(ChunkPos::new(x, y, z));
			}
		}
	}
	v
}

#[test]
fn empty_chunk_is_all_air() {
	let c = Chunk::new();
	for p in sample_grid() {
		assert_eq!(c.voxel_at(p), Material::air());
	}
}

#[test]
fn single_voxel_edit() {
	let mut edits = EditPacket::default();
	let p = ChunkPos::new(10, 20, 30);
	let m = mat(0x11223340);
	edits.push(Path::from_coords(p, 4), m);

	let chunk = bake_one(Chunk::new(), edits);
	assert_eq!(chunk.voxel_at(p), m);
	assert_eq!(chunk.voxel_at(ChunkPos::new(11, 20, 30)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(10, 21, 30)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(0, 0, 0)), Material::air());
}

#[test]
fn root_fill_then_carve() {
	let fill = mat(0xAABBCC40);
	let mut edits = EditPacket::default();
	edits.push(Path::from(0u32), fill);

	let mut chunk = bake_one(Chunk::new(), edits);
	for p in sample_grid() {
		assert_eq!(chunk.voxel_at(p), fill);
	}

	let carve = ChunkPos::new(100, 100, 100);
	let mut e2 = EditPacket::default();
	e2.push(Path::from_coords(carve, 4), Material::air());
	chunk = bake_one(chunk, e2);

	assert_eq!(chunk.voxel_at(carve), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(101, 100, 100)), fill);
	assert_eq!(chunk.voxel_at(ChunkPos::new(0, 0, 0)), fill);
}

#[test]
fn depth3_cube_edit() {
	let mut edits = EditPacket::default();
	let m = mat(0x33445540);
	let corner = ChunkPos::new(8, 12, 16);
	edits.push(Path::from_coords(corner, 3), m);
	let chunk = bake_one(Chunk::new(), edits);

	for dx in 0..4u8 {
		for dy in 0..4u8 {
			for dz in 0..4u8 {
				let p = ChunkPos::new(corner.x() + dx, corner.y() + dy, corner.z() + dz);
				assert_eq!(chunk.voxel_at(p), m, "inside cube at {:?}", p);
			}
		}
	}
	assert_eq!(
		chunk.voxel_at(ChunkPos::new(corner.x() + 4, corner.y(), corner.z())),
		Material::air()
	);
}

#[test]
fn dedup_handles_identical_subtrees() {
	let m = mat(0x55667740);
	let mut edits = EditPacket::default();
	let positions = [
		ChunkPos::new(0, 0, 0),
		ChunkPos::new(16, 16, 16),
		ChunkPos::new(0, 16, 32),
		ChunkPos::new(48, 0, 48),
	];
	for &p in &positions {
		edits.push(Path::from_coords(p, 3), m);
	}

	let chunk = bake_one(Chunk::new(), edits);

	for &p in &positions {
		for dx in 0..4u8 {
			for dy in 0..4u8 {
				for dz in 0..4u8 {
					let q = ChunkPos::new(p.x() + dx, p.y() + dy, p.z() + dz);
					assert_eq!(chunk.voxel_at(q), m, "inside cube at {:?}", q);
				}
			}
		}
	}
	assert_eq!(chunk.voxel_at(ChunkPos::new(8, 8, 8)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(100, 100, 100)), Material::air());

	// Material LUT holds a single entry for m. Without run-dedup, each of the
	// four identical leaves would have contributed its own material run of 64
	// copies of m.
	assert_eq!(chunk.materials.lut.len(), 1);
}

fn uniform_chunk(m: Material) -> Chunk {
	let mut e = EditPacket::default();
	e.push(Path::from(0u32), m);
	bake_one(Chunk::new(), e)
}

#[test]
fn deep_edit_into_demoted_leaf_region() {
	let m1 = mat(0x11111140);
	let m2 = mat(0x22222240);
	let m3 = mat(0x33333340);
	let mut p1 = EditPacket::default();
	p1.push(Path::from_coords(ChunkPos::new(0, 0, 0), 2), m1);
	p1.push(Path::from_coords(ChunkPos::new(16, 0, 0), 2), m2);
	let mid = bake_one(Chunk::new(), p1);

	assert_eq!(mid.voxel_at(ChunkPos::new(0, 0, 0)), m1);
	assert_eq!(mid.voxel_at(ChunkPos::new(16, 0, 0)), m2);

	let target = ChunkPos::new(32, 0, 0);
	let mut p2 = EditPacket::default();
	p2.push(Path::from_coords(target, 4), m3);
	let chunk = bake_one(mid, p2);

	assert_eq!(chunk.voxel_at(target), m3);
	assert_eq!(chunk.voxel_at(ChunkPos::new(0, 0, 0)), m1);
	assert_eq!(chunk.voxel_at(ChunkPos::new(16, 0, 0)), m2);
}

#[test]
fn sphere_paints_inside_and_leaves_outside_air() {
	let m = mat(0x778899AA);
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let chunk = Sphere::new(WorldPos::new(128, 128, 128), 20, m).apply(chunk_id, Chunk::new());

	for (offset, expected) in [
		([0, 0, 0], m),
		([10, 5, -5], m),
		([20, 0, 0], m),
		([21, 0, 0], Material::air()),
		([100, 0, 0], Material::air()),
	] {
		let pos = ChunkPos::new(
			(128 + offset[0]) as u8,
			(128 + offset[1]) as u8,
			(128 + offset[2]) as u8,
		);
		assert_eq!(chunk.voxel_at(pos), expected, "offset {:?}", offset);
	}
}

#[test]
fn sphere_carve_leaves_air_hole_in_filled_chunk() {
	let stone = mat(0x80808040);
	let mut fill = EditPacket::default();
	fill.push(Path::from(0u32), stone);
	let solid = bake_one(Chunk::new(), fill);

	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let chunk = Sphere::new(WorldPos::new(128, 128, 128), 20, Material::air())
		.apply(chunk_id, solid);

	assert_eq!(chunk.voxel_at(ChunkPos::new(128, 128, 128)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(138, 128, 128)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(150, 128, 128)), stone);
	assert_eq!(chunk.voxel_at(ChunkPos::new(0, 0, 0)), stone);
}

#[test]
fn empty_and_uniform_chunks_use_the_materials_zero_shortcut() {
	let empty = Chunk::new();
	assert!(empty.materials.is_empty());
	assert!(empty.interior_nodes.is_empty() && empty.leaf_nodes.is_empty());
	assert_eq!(empty.voxel_at(ChunkPos::new(0, 0, 0)), Material::air());

	let m = mat(0xCAFEBE40);
	let u = uniform_chunk(m);
	assert!(u.is_uniform());
	assert_eq!(u.materials.len(), 1);
	assert_eq!(u.uniform_material(), m);
	assert!(u.interior_nodes.is_empty() && u.leaf_nodes.is_empty());
	assert_eq!(u.voxel_at(ChunkPos::new(0, 0, 0)), m);
	assert_eq!(u.voxel_at(ChunkPos::new(255, 255, 255)), m);
}

#[test]
fn property_random_voxel_edits() {
	let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
	let palette = [
		mat(0x11111140),
		mat(0x22222240),
		mat(0x33333340),
		Material::air(),
	];

	for trial in 0..32 {
		let mut edits = EditPacket::default();
		let mut brute = BruteChunk::default();
		let n: u32 = rng.r#gen_range(1..=64);
		let mut touched: Vec<ChunkPos> = Vec::new();
		for _ in 0..n {
			let x = rng.r#gen::<u8>();
			let y = rng.r#gen::<u8>();
			let z = rng.r#gen::<u8>();
			let p = ChunkPos::new(x, y, z);
			let m = palette[rng.r#gen_range(0..palette.len())];
			edits.push(Path::from_coords(p, 4), m);
			touched.push(p);
		}

		let edits_for_chunk = edits.clone();
		brute_apply(&mut brute, edits);
		let chunk = bake_one(Chunk::new(), edits_for_chunk);

		let mut samples = sample_grid();
		samples.extend(touched.iter().copied());
		assert_match(&chunk, &brute, &samples);
		let _ = trial;
	}
}

#[test]
fn property_fill_then_random_carves() {
	let mut rng = SmallRng::seed_from_u64(0xBEEF);
	let fill = mat(0x77777740);

	for _ in 0..16 {
		let mut e1 = EditPacket::default();
		e1.push(Path::from(0u32), fill);
		let mut e2 = EditPacket::default();
		let mut brute = BruteChunk::default();
		brute_apply(&mut brute, e1.clone());

		let mut touched = Vec::new();
		let n: u32 = rng.r#gen_range(1..=32);
		for _ in 0..n {
			let p = ChunkPos::new(rng.r#gen(), rng.r#gen(), rng.r#gen());
			let m = if rng.r#gen_bool(0.5) {
				Material::air()
			} else {
				mat(rng.r#gen_range(0x10..0xF0) << 8 | 0x40)
			};
			e2.push(Path::from_coords(p, 4), m);
			touched.push(p);
		}

		brute_apply(&mut brute, e2.clone());
		let chunk = bake_one(bake_one(Chunk::new(), e1), e2);

		let mut samples = sample_grid();
		samples.extend(touched);
		assert_match(&chunk, &brute, &samples);
	}
}

#[test]
fn property_multi_packet_mixed_depths() {
	let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
	let palette = [
		mat(0x11111140),
		mat(0x22222240),
		mat(0x33333340),
		mat(0x44444440),
		Material::air(),
	];

	for _ in 0..16 {
		let mut chunk = Chunk::new();
		let mut brute = BruteChunk::default();
		let mut touched: Vec<ChunkPos> = Vec::new();

		let packets: u32 = rng.r#gen_range(2..=5);
		for _ in 0..packets {
			let mut packet = EditPacket::default();
			let edits: u32 = rng.r#gen_range(1..=16);
			for _ in 0..edits {
				let depth: u8 = rng.r#gen_range(1..=4);
				let step: u32 = 1 << (2 * (4 - depth));
				let max_cell = 256u32 / step;
				let cx = rng.r#gen_range(0..max_cell);
				let cy = rng.r#gen_range(0..max_cell);
				let cz = rng.r#gen_range(0..max_cell);
				let pos = ChunkPos::new(
					(cx * step) as u8,
					(cy * step) as u8,
					(cz * step) as u8,
				);
				let m = palette[rng.r#gen_range(0..palette.len())];
				packet.push(Path::from_coords(pos, depth), m);
				touched.push(pos);
			}
			let packet_for_brute = packet.clone();
			chunk = bake_one(chunk, packet);
			brute_apply(&mut brute, packet_for_brute);
		}

		let mut samples = sample_grid();
		samples.extend(touched);
		assert_match(&chunk, &brute, &samples);
	}
}

#[test]
fn overlay_last_writer_wins_for_two_overlapping_spheres() {
	let red = mat(0x11111140);
	let blue = mat(0x22222240);
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let s1 = Sphere::new(WorldPos::new(100, 128, 128), 40, red).local(chunk_id).unwrap();
	let s2 = Sphere::new(WorldPos::new(156, 128, 128), 40, blue).local(chunk_id).unwrap();
	let chunk = build_chunk(&Overlay::new(s1, s2));

	assert_eq!(chunk.voxel_at(ChunkPos::new(190, 128, 128)), blue);
	assert_eq!(chunk.voxel_at(ChunkPos::new(70, 128, 128)), red);
	assert_eq!(chunk.voxel_at(ChunkPos::new(128, 128, 128)), blue);
	assert_eq!(chunk.voxel_at(ChunkPos::new(0, 0, 0)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(255, 255, 255)), Material::air());
}

#[test]
fn edit_sphere_over_detailed_base_preserves_untouched_voxels() {
	let palette = [mat(0x11111140), mat(0x22222240), mat(0x33333340)];
	let mut edits = EditPacket::default();
	for (i, &m) in palette.iter().enumerate() {
		for slot in 0..16usize {
			let pos = ChunkPos::new(
				((i * 64 + (slot & 3) * 16) % 256) as u8,
				((slot >> 2) * 16) as u8,
				((i * 64) % 256) as u8,
			);
			edits.push(Path::from_coords(pos, 3), m);
		}
	}
	let base = bake_one(Chunk::new(), edits);

	let outside_samples = [
		ChunkPos::new(8, 8, 8),
		ChunkPos::new(200, 200, 8),
		ChunkPos::new(8, 200, 8),
	];
	let pre: Vec<_> = outside_samples.iter().map(|p| (*p, base.voxel_at(*p))).collect();

	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0));
	let carved = Sphere::new(WorldPos::new(128, 128, 128), 30, Material::air())
		.apply(chunk_id, base);

	assert_eq!(carved.voxel_at(ChunkPos::new(128, 128, 128)), Material::air());
	assert_eq!(carved.voxel_at(ChunkPos::new(140, 128, 128)), Material::air());
	for (p, before) in pre {
		assert_eq!(carved.voxel_at(p), before, "pos {:?} should be unchanged", p);
	}
}

#[test]
fn overlay_passthrough_reveals_base() {
	use super::build::{Sample, Source, VoxelSample};

	#[derive(Clone, Copy)]
	struct NoOp;
	impl Source for NoOp {
		fn classify(&self, _: [i32; 3], _: [i32; 3], _: u8) -> Sample { Sample::Passthrough }
		fn voxel(&self, _: [i32; 3]) -> VoxelSample { VoxelSample::Passthrough }
	}

	let fill = mat(0xAABBCC40);
	let mut e = EditPacket::default();
	e.push(Path::from(0u32), fill);
	let base = bake_one(Chunk::new(), e);
	let out = base.clone().edit(&NoOp);
	for p in sample_grid() {
		assert_eq!(out.voxel_at(p), fill, "passthrough should preserve base at {:?}", p);
	}
}
