//! Chunk edit application tests. BruteChunk is a sparse reference that applies
//! edits the same way the engine does (single-depth batches, in order). Property
//! tests compare BruteChunk to Chunk::voxel_at across random edits.

use super::edit::{EditPacket, Path};
use super::material::Material;
use super::merge::merge_lod;
use super::rebuild::mode_over;
use super::Chunk;
use crate::util::types::{ChunkPos, Mask64};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

fn mat(v: u32) -> Material {
	Material::from(v)
}

/// Returns the (corner, side) of the voxel cube the path covers.
/// Depth 0 is handled by the caller since it covers the whole chunk.
fn path_to_cube(path: Path) -> ([u8; 3], u8) {
	let (corner, depth) = path.to_coords();
	let side = 1u8 << (8u8.saturating_sub(2 * depth)).min(7);
	([corner.x(), corner.y(), corner.z()], side)
}

#[derive(Default, Clone)]
struct BruteChunk {
	// Sparse storage. Absent keys take the background value.
	voxels: HashMap<u32, Material>,
	background: Material,
}

fn pack(x: u8, y: u8, z: u8) -> u32 {
	(x as u32) | ((y as u32) << 8) | ((z as u32) << 16)
}

/// Test helper: apply one packet to a chunk and bake. Tests don't need the
/// queueing flow, so we keep this one-liner here instead of polluting Chunk's API.
fn bake_one(chunk: Chunk, packet: EditPacket) -> Chunk {
	let mut mc = chunk.into_mutable();
	mc.queue_edit(packet);
	mc.bake()
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

/// Apply a packet to BruteChunk in path-sorted order, mirroring bake.
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

	// Carve a single voxel back to air in a separate batch.
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
	// A depth-3 path covers a 4x4x4 voxel cube.
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
fn mode_over_picks_most_frequent_material() {
	let a = mat(0x11111140);
	let b = mat(0x22222240);
	let c = mat(0x33333340);

	let mut mats = [Material::air(); 64];
	mats[0] = a;
	mats[1] = a;
	mats[2] = a;
	mats[5] = b;
	mats[10] = c;
	let occ = Mask64(1 | 1 << 1 | 1 << 2 | 1 << 5 | 1 << 10);
	assert_eq!(mode_over(occ, &mats), a);
}

#[test]
fn mode_over_breaks_ties_by_lowest_slot() {
	let a = mat(0xAAAAAA40);
	let b = mat(0xBBBBBB40);

	let mut mats = [Material::air(); 64];
	mats[1] = a;
	mats[3] = b;
	mats[5] = a;
	mats[7] = b;
	let occ = Mask64((1u64 << 1) | (1 << 3) | (1 << 5) | (1 << 7));
	assert_eq!(mode_over(occ, &mats), a);

	let mut mats = [Material::air(); 64];
	mats[1] = b;
	mats[3] = a;
	mats[5] = b;
	mats[7] = a;
	assert_eq!(mode_over(occ, &mats), b);
}

#[test]
fn dedup_handles_identical_subtrees() {
	// Several cubes of the same material at positions with low 4 bits zero produce
	// structurally identical depth-2 interiors, all sharing the same content hash.
	// The verify step has to confirm equality before merging.
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

	// One material across all cubes, so the palette has one entry.
	assert_eq!(chunk.materials.lut.len(), 1);
}

fn uniform_chunk(m: Material) -> Chunk {
	let mut e = EditPacket::default();
	e.push(Path::from(0u32), m);
	bake_one(Chunk::new(), e)
}

#[test]
fn merge_all_empty_is_empty() {
	let coarse = merge_lod([(); 64].map(|_| None));
	assert!(coarse.is_empty());
}

#[test]
fn merge_all_uniform_same_is_uniform() {
	let m = mat(0x99AABB40);
	let fine = uniform_chunk(m);
	let arr = [Some(&fine); 64];
	let coarse = merge_lod(arr);
	assert!(coarse.is_uniform(), "expected uniform coarse chunk");
	for p in sample_grid() {
		assert_eq!(coarse.voxel_at(p), m);
	}
}

#[test]
fn merge_routes_content_to_correct_coarse_position() {
	// One fine voxel at fine (0,0,0) in slot-0 of the coarse root.
	let m = mat(0xABCDEF40);
	let mut e = EditPacket::default();
	e.push(Path::from_coords(ChunkPos::new(0, 0, 0), 4), m);
	let fine = bake_one(Chunk::new(), e);

	let mut children: [Option<&Chunk>; 64] = [None; 64];
	children[0] = Some(&fine);
	let coarse = merge_lod(children);

	// The lone non-air voxel wins the mode for the coarse voxel covering fine (0..4)^3.
	assert_eq!(coarse.voxel_at(ChunkPos::new(0, 0, 0)), m);
	assert_eq!(coarse.voxel_at(ChunkPos::new(128, 128, 128)), Material::air());
}

#[test]
fn merge_mixes_two_uniform_chunks_at_distinct_slots() {
	let a = mat(0x11111140);
	let b = mat(0x22222240);
	let chunk_a = uniform_chunk(a);
	let chunk_b = uniform_chunk(b);

	let mut children: [Option<&Chunk>; 64] = [None; 64];
	children[0] = Some(&chunk_a);
	// Slot 63 = (x=3, y=3, z=3) in the slot bit-packing, covering coarse cells
	// from (192, 192, 192) to (255, 255, 255) since each slot covers 64 coarse voxels.
	children[63] = Some(&chunk_b);
	let coarse = merge_lod(children);

	assert_eq!(coarse.voxel_at(ChunkPos::new(0, 0, 0)), a);
	assert_eq!(coarse.voxel_at(ChunkPos::new(63, 63, 63)), a);
	assert_eq!(coarse.voxel_at(ChunkPos::new(64, 64, 64)), Material::air());
	assert_eq!(coarse.voxel_at(ChunkPos::new(192, 192, 192)), b);
	assert_eq!(coarse.voxel_at(ChunkPos::new(255, 255, 255)), b);
}

#[test]
fn chunk_lod_at_materials_zero_invariant() {
	let empty = Chunk::new();
	assert_eq!(empty.chunk_lod(), Material::air());
	assert!(empty.materials.is_empty());

	let m = mat(0xCAFEBE40);
	let u = uniform_chunk(m);
	assert_eq!(u.chunk_lod(), m);
	assert_eq!(u.materials.len(), 1);
	assert_eq!(u.materials.get(0), m);
	assert!(u.interior_nodes.is_empty() && u.leaf_nodes.is_empty());

	let mut e = EditPacket::default();
	e.push(Path::from_coords(ChunkPos::new(0, 0, 0), 3), m);
	let c = bake_one(Chunk::new(), e);
	assert!(!c.interior_nodes.is_empty() || !c.leaf_nodes.is_empty());
	assert_eq!(c.chunk_lod(), m);
	assert_eq!(c.materials.get(0), m);

	let a = mat(0x11111140);
	let b = mat(0x22222240);
	let mut e = EditPacket::default();
	e.push(Path::from_coords(ChunkPos::new(0, 0, 0), 3), a);
	e.push(Path::from_coords(ChunkPos::new(0, 0, 16), 3), a);
	e.push(Path::from_coords(ChunkPos::new(0, 16, 0), 3), a);
	e.push(Path::from_coords(ChunkPos::new(0, 0, 32), 3), b);
	let c = bake_one(Chunk::new(), e);
	assert_eq!(c.chunk_lod(), a);
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
