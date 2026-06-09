
use super::coarsen::coarsen;
use super::Model;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::Chunk;
use crate::util::types::{Aabb, ChunkId, ChunkPos, LodLevel, WorldPos};

fn mat(v: u32) -> Material { Material::from(v) }

fn one_voxel_chunk(pos: ChunkPos, m: Material) -> Chunk {
	let mut packet = EditPacket::default();
	packet.push(Path::from_coords(pos, 4), m);
	packet.sort();
	Chunk::new().edit(&DiscreteSource::new(&packet.edits))
}

fn uniform_chunk(m: Material) -> Chunk {
	let mut packet = EditPacket::default();
	packet.push(Path::from(0u32), m);
	Chunk::new().edit(&DiscreteSource::new(&packet.edits))
}


#[test]
fn coarsen_all_empty_gives_empty() {
	let children: [Option<&Chunk>; 64] = [None; 64];
	let coarse = coarsen(&children);
	assert!(coarse.is_empty());
}

#[test]
fn coarsen_all_uniform_same_material_gives_uniform() {
	let m = mat(0x80808040);
	let fine = uniform_chunk(m);
	let children = [Some(&fine); 64];
	let coarse = coarsen(&children);
	assert!(coarse.is_uniform(), "expected uniform coarse chunk");
	assert_eq!(coarse.chunk_lod(), m);
}

#[test]
fn coarsen_single_fine_voxel_coarsens_to_air() {
	// Under the "majority-or-air" LOD rule, one occupied voxel out of 64 fine voxels
	// per coarse cell loses to air at the first mode_over step, then every coarser
	// level sees 1 non-air slot out of 64 and also picks air. Sparse content is
	// expected to disappear in the coarse representation.
	let m = mat(0xABCDEF40);
	let fine = one_voxel_chunk(ChunkPos::new(0, 0, 0), m);
	let mut children: [Option<&Chunk>; 64] = [None; 64];
	children[0] = Some(&fine);
	let coarse = coarsen(&children);

	assert_eq!(coarse.voxel_at(ChunkPos::new(0, 0, 0)), Material::air());
	assert_eq!(coarse.voxel_at(ChunkPos::new(1, 0, 0)), Material::air());
}

#[test]
fn coarsen_two_uniform_children_at_distinct_slots() {
	let a = mat(0x11111140);
	let b = mat(0x22222240);
	let fine_a = uniform_chunk(a);
	let fine_b = uniform_chunk(b);

	let mut children: [Option<&Chunk>; 64] = [None; 64];
	children[0] = Some(&fine_a);
	children[63] = Some(&fine_b); // slot (3, 3, 3)
	let coarse = coarsen(&children);

	assert_eq!(coarse.voxel_at(ChunkPos::new(0, 0, 0)), a);
	assert_eq!(coarse.voxel_at(ChunkPos::new(63, 63, 63)), a);
	assert_eq!(coarse.voxel_at(ChunkPos::new(192, 192, 192)), b);
	assert_eq!(coarse.voxel_at(ChunkPos::new(255, 255, 255)), b);
	assert_eq!(coarse.voxel_at(ChunkPos::new(128, 128, 128)), Material::air());
}


#[test]
fn mip_pyramid_builds_one_level_up_from_finest() {
	let m = mat(0x12345640);
	let fine_chunk = uniform_chunk(m);

	let bounds = Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(256, 256, 256));
	let mut model = Model::empty(bounds);
	let fine_id = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	model.chunks.insert(fine_id, fine_chunk);

	let initial_count = model.chunks.len();
	model.build_mip_pyramid();
	assert!(model.chunks.len() > initial_count, "pyramid should add coarser LODs");

	let parent = fine_id.parent().expect("finest has a parent");
	let coarse = model.chunks.get(&parent).expect("parent should be filled in");
	let local = WorldPos::new(0, 0, 0).chunk_pos(parent.origin, parent.lod);
	assert_eq!(coarse.voxel_at(local), m);
}

#[test]
fn mip_pyramid_drops_empty_parents() {
	let bounds = Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(256, 256, 256));
	let mut model = Model::empty(bounds);
	model.build_mip_pyramid();
	assert_eq!(model.chunks.len(), 0);
}


#[test]
fn rvox_round_trip_preserves_voxel_content() {
	let m = mat(0xCAFEBE40);
	let fine_chunk = one_voxel_chunk(ChunkPos::new(10, 20, 30), m);

	let bounds = Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(256, 256, 256));
	let mut original = Model::empty(bounds);
	original.chunks.insert(
		ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST),
		fine_chunk,
	);
	original.build_mip_pyramid();
	let original_count = original.chunks.len();

	let mut buf: Vec<u8> = Vec::new();
	original.save_rvox(&mut buf).expect("save");
	let mut cursor = std::io::Cursor::new(&buf);
	let loaded = Model::load_rvox(&mut cursor).expect("load");

	assert_eq!(loaded.bounds, original.bounds);
	assert_eq!(loaded.chunks.len(), original_count);

	for (id, orig_chunk) in &original.chunks {
		let loaded_chunk = loaded.chunks.get(id).expect("chunk preserved");
		for &p in &[
			ChunkPos::new(10, 20, 30),
			ChunkPos::new(0, 0, 0),
			ChunkPos::new(255, 255, 255),
			ChunkPos::new(128, 128, 128),
		] {
			assert_eq!(
				orig_chunk.voxel_at(p),
				loaded_chunk.voxel_at(p),
				"voxel mismatch at {:?} in chunk {:?}", p, id,
			);
		}
	}
}

#[test]
fn rvox_load_rejects_bad_magic() {
	let mut bad: Vec<u8> = vec![0; 100];
	bad[0..4].copy_from_slice(b"XXXX");
	let mut cursor = std::io::Cursor::new(&bad);
	let err = Model::load_rvox(&mut cursor);
	assert!(matches!(err, Err(super::rvox::RvoxError::BadMagic)));
}


#[test]
fn import_vox_from_synthetic_bytes() {
	use dot_vox::{Color, DotVoxData, Model as VoxModel, Size, Voxel};

	let voxels = vec![
		Voxel { x: 1, y: 2, z: 3, i: 0 }, // palette index 0 = red
		Voxel { x: 5, y: 6, z: 7, i: 1 }, // palette index 1 = blue
	];
	let model = VoxModel {
		size: Size { x: 16, y: 16, z: 16 },
		voxels,
	};
	let mut palette: Vec<Color> = vec![Color { r: 0, g: 0, b: 0, a: 255 }; 256];
	palette[0] = Color { r: 0xFF, g: 0x10, b: 0x10, a: 0xFF };
	palette[1] = Color { r: 0x10, g: 0x10, b: 0xFF, a: 0xFF };
	let data = DotVoxData {
		version: 150,
		index_map: (0..255u8).collect(),
		models: vec![model],
		palette,
		materials: Vec::new(),
		scenes: Vec::new(),
		layers: Vec::new(),
	};
	let mut bytes: Vec<u8> = Vec::new();
	data.write_vox(&mut bytes).expect("write_vox");

	let model = Model::import_vox(&bytes).expect("import");
	assert!(model.chunks.len() > 0, "model should have chunks");

	let finest_id = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let fine = model.chunks.get(&finest_id).expect("finest chunk");
	let red = fine.voxel_at(ChunkPos::new(0, 0, 0));
	let blue = fine.voxel_at(ChunkPos::new(4, 4, 4));
	assert_ne!(red, Material::air(), "red voxel should be set");
	assert_ne!(blue, Material::air(), "blue voxel should be set");
	assert_ne!(red, blue, "red and blue should be different materials");
}

#[test]
fn stamp_places_voxels_at_world_position() {
	use super::stamp::ModelStamp;
	use crate::generate::Edit;
	use std::sync::Arc;

	let m = mat(0x778899AA);
	let stone_chunk = one_voxel_chunk(ChunkPos::new(5, 6, 7), m);
	let mut model = Model::empty(Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(8, 8, 8)));
	model.chunks.insert(ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST), stone_chunk);

	let position = WorldPos::new(100, 200, 50);
	let stamp = ModelStamp::new(Arc::new(model), position);

	let target = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let baked = stamp.apply(target, Chunk::new());

	assert_eq!(baked.voxel_at(ChunkPos::new(105, 206, 57)), m);
	assert_eq!(baked.voxel_at(ChunkPos::new(0, 0, 0)), Material::air());
	assert_eq!(baked.voxel_at(ChunkPos::new(106, 206, 57)), Material::air());
}

#[test]
fn stamp_outside_chunk_returns_base_unchanged() {
	use super::stamp::ModelStamp;
	use crate::generate::Edit;
	use std::sync::Arc;

	let m = mat(0x12345678);
	let chunk = one_voxel_chunk(ChunkPos::new(0, 0, 0), m);
	let mut model = Model::empty(Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(4, 4, 4)));
	model.chunks.insert(ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST), chunk);

	let far_position = WorldPos::new(10_000, 10_000, 10_000);
	let stamp = ModelStamp::new(Arc::new(model), far_position);

	let target = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let base = uniform_chunk(mat(0x0000FF40));
	let baked = stamp.apply(target, base.clone());

	assert_eq!(baked.voxel_at(ChunkPos::new(0, 0, 0)), base.voxel_at(ChunkPos::new(0, 0, 0)));
	assert_eq!(baked.voxel_at(ChunkPos::new(128, 128, 128)), base.voxel_at(ChunkPos::new(128, 128, 128)));
}

#[test]
fn stamp_at_unaligned_position_routes_voxels_correctly() {
	use super::stamp::ModelStamp;
	use crate::generate::Edit;
	use std::sync::Arc;

	let m_a = mat(0xAAAAAAAA);
	let m_b = mat(0xBBBBBBBB);
	let chunk0 = one_voxel_chunk(ChunkPos::new(255, 0, 0), m_a);
	let chunk1 = one_voxel_chunk(ChunkPos::new(0, 0, 0), m_b);
	let mut model = Model::empty(Aabb::new(WorldPos::new(0, 0, 0), WorldPos::new(512, 4, 4)));
	model.chunks.insert(ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST), chunk0);
	model.chunks.insert(ChunkId::new(WorldPos::new(256, 0, 0), LodLevel::FINEST), chunk1);

	let position = WorldPos::new(123, 0, 0);
	let stamp = ModelStamp::new(Arc::new(model), position);

	let target0 = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let baked0 = stamp.apply(target0, Chunk::new());
	for x in 0..=255u8 {
		assert_eq!(baked0.voxel_at(ChunkPos::new(x, 0, 0)), Material::air(),
			"target0 should be all air at x={}, got non-air", x);
	}

	let target1 = ChunkId::new(WorldPos::new(256, 0, 0), LodLevel::FINEST);
	let baked1 = stamp.apply(target1, Chunk::new());
	assert_eq!(baked1.voxel_at(ChunkPos::new(122, 0, 0)), m_a,
		"model chunk0 voxel (255) at world x=378 should land at local 122 of world chunk 1");
	assert_eq!(baked1.voxel_at(ChunkPos::new(123, 0, 0)), m_b,
		"model chunk1 voxel (0) at world x=379 should land at local 123 of world chunk 1");
}
