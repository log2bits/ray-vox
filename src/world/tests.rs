use super::World;
use crate::chunk::material::Material;
use crate::generate::volume::sphere::Sphere;
use crate::util::types::{ChunkPos, WorldPos};
use std::sync::Arc;

fn mat(v: u32) -> Material { Material::from(v) }

#[test]
fn empty_world_has_no_baked_chunks() {
	let world = World::new([2, 2, 2]);
	assert_eq!(world.chunk_slot_count(), 8);
	assert!(world.chunks.iter().all(|c| c.is_none()));
}

#[test]
fn slot_index_round_trips_grid_position() {
	let world = World::new([4, 3, 5]);
	for gz in 0..5 {
		for gy in 0..3 {
			for gx in 0..4 {
				let grid_pos = [gx, gy, gz];
				let index = world.slot_index(grid_pos).unwrap();
				assert_eq!(world.slot_grid_pos(index), grid_pos);
			}
		}
	}
}

#[test]
fn slot_index_returns_none_out_of_range() {
	let world = World::new([2, 2, 2]);
	assert!(world.slot_index([2, 0, 0]).is_none());
	assert!(world.slot_index([0, 2, 0]).is_none());
	assert!(world.slot_index([0, 0, 2]).is_none());
}

#[test]
fn chunk_world_origin_uses_grid_and_world_origin() {
	let world = World::with_origin([4, 4, 4], WorldPos::new(-256, 0, 512));
	assert_eq!(world.chunk_world_origin([0, 0, 0]), WorldPos::new(-256, 0, 512));
	assert_eq!(world.chunk_world_origin([2, 1, 3]), WorldPos::new(-256 + 2 * 256, 256, 512 + 3 * 256));
}

#[test]
fn sphere_edit_populates_only_overlapping_chunks() {
	let mut world = World::new([3, 3, 3]);
	let center = world.chunk_world_origin([1, 1, 1]) + WorldPos::new(128, 128, 128);
	world.apply_edit(Arc::new(Sphere::new(center, 20, mat(0x11111140))));

	for gz in 0..3 {
		for gy in 0..3 {
			for gx in 0..3 {
				let is_center = [gx, gy, gz] == [1, 1, 1];
				assert_eq!(
					world.chunk_at([gx, gy, gz]).is_some(),
					is_center,
					"unexpected residency at {:?}", [gx, gy, gz],
				);
			}
		}
	}
}

#[test]
fn later_edits_override_earlier_ones_in_overlap() {
	let mut world = World::new([1, 1, 1]);
	let red = mat(0x11111140);
	let blue = mat(0x22222240);
	let center = WorldPos::new(128, 128, 128);
	world.apply_edit(Arc::new(Sphere::new(center, 40, red)));
	world.apply_edit(Arc::new(Sphere::new(center, 40, blue)));

	let chunk = world.chunk_at([0, 0, 0]).expect("center chunk baked");
	assert_eq!(chunk.voxel_at(ChunkPos::new(128, 128, 128)), blue);
}

#[test]
fn sphere_spanning_chunk_boundary_paints_both_sides() {
	let mut world = World::new([2, 1, 1]);
	let center = WorldPos::new(256, 128, 128);
	world.apply_edit(Arc::new(Sphere::new(center, 20, mat(0x33333340))));

	let left = world.chunk_at([0, 0, 0]).expect("left chunk baked");
	let right = world.chunk_at([1, 0, 0]).expect("right chunk baked");
	assert_ne!(left.voxel_at(ChunkPos::new(255, 128, 128)), Material::air());
	assert_ne!(right.voxel_at(ChunkPos::new(0, 128, 128)), Material::air());
}

#[test]
fn edit_outside_grid_touches_no_chunks() {
	let mut world = World::new([1, 1, 1]);
	let far = WorldPos::new(10_000, 10_000, 10_000);
	world.apply_edit(Arc::new(Sphere::new(far, 20, mat(0x40))));
	assert!(world.chunks.iter().all(|c| c.is_none()));
}

#[test]
fn air_edit_carves_a_hole() {
	let mut world = World::new([1, 1, 1]);
	let stone = mat(0x80808040);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 60, stone)));
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 20, Material::air())));

	let chunk = world.chunk_at([0, 0, 0]).expect("chunk baked");
	assert_eq!(chunk.voxel_at(ChunkPos::new(128, 128, 128)), Material::air());
	assert_eq!(chunk.voxel_at(ChunkPos::new(150, 128, 128)), stone);
}

#[test]
fn from_edits_sizes_grid_to_bounds_and_places_chunks() {
	use crate::world::WorldEdit;
	let m = mat(0xAB);
	let edits = vec![
		WorldEdit { pos: WorldPos::new(10, 20, 30), material: m },
		WorldEdit { pos: WorldPos::new(260, 20, 30), material: m },
	];
	let world = World::from_edits(edits);

	// Two voxels straddle the x=256 chunk boundary, so we need at least
	// 2 chunks along x and 1 each along y and z. Origin snaps to 0,0,0.
	assert_eq!(world.origin, WorldPos::new(0, 0, 0));
	assert_eq!(world.chunk_grid_dim, [2, 1, 1]);
	assert_eq!(world.chunk_at([0, 0, 0]).map(|c| c.voxel_at(ChunkPos::new(10, 20, 30))), Some(m));
	assert_eq!(world.chunk_at([1, 0, 0]).map(|c| c.voxel_at(ChunkPos::new(4, 20, 30))), Some(m));
}

#[test]
fn rvox_round_trip_preserves_chunks() {
	let mut world = World::new([2, 1, 1]);
	let m = mat(0xCAFEBE);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(256, 128, 128), 30, m)));
	let non_empty_before = world.chunks.iter().filter(|c| c.is_some()).count();

	let mut buf: Vec<u8> = Vec::new();
	world.save_rvox(&mut buf).expect("save");
	let mut cursor = std::io::Cursor::new(&buf);
	let loaded = World::load_rvox(&mut cursor).expect("load");

	assert_eq!(loaded.chunk_grid_dim, world.chunk_grid_dim);
	assert_eq!(loaded.origin, world.origin);
	assert_eq!(loaded.chunks.iter().filter(|c| c.is_some()).count(), non_empty_before);
	for grid_pos in [[0u32, 0, 0], [1, 0, 0]] {
		let before = world.chunk_at(grid_pos);
		let after = loaded.chunk_at(grid_pos);
		match (before, after) {
			(Some(b), Some(a)) => {
				assert_eq!(b.voxel_at(ChunkPos::new(255, 128, 128)), a.voxel_at(ChunkPos::new(255, 128, 128)));
				assert_eq!(b.voxel_at(ChunkPos::new(0, 128, 128)), a.voxel_at(ChunkPos::new(0, 128, 128)));
			}
			(None, None) => {}
			_ => panic!("residency mismatch at {:?}", grid_pos),
		}
	}
}

#[test]
fn rvox_load_rejects_bad_magic() {
	let mut bad: Vec<u8> = vec![0; 100];
	bad[0..4].copy_from_slice(b"XXXX");
	let mut cursor = std::io::Cursor::new(&bad);
	let err = World::load_rvox(&mut cursor);
	assert!(matches!(err, Err(crate::world::RvoxError::BadMagic)));
}

#[test]
fn import_vox_from_synthetic_bytes_builds_a_world() {
	use dot_vox::{Color, DotVoxData, Model as VoxModel, Size, Voxel};

	let voxels = vec![
		Voxel { x: 1, y: 2, z: 3, i: 0 },
		Voxel { x: 5, y: 6, z: 7, i: 1 },
	];
	let vox_model = VoxModel { size: Size { x: 16, y: 16, z: 16 }, voxels };
	let mut palette: Vec<Color> = vec![Color { r: 0, g: 0, b: 0, a: 255 }; 256];
	palette[0] = Color { r: 0xFF, g: 0x10, b: 0x10, a: 0xFF };
	palette[1] = Color { r: 0x10, g: 0x10, b: 0xFF, a: 0xFF };
	let data = DotVoxData {
		version: 150,
		index_map: (0..255u8).collect(),
		models: vec![vox_model],
		palette,
		materials: Vec::new(),
		scenes: Vec::new(),
		layers: Vec::new(),
	};
	let mut bytes: Vec<u8> = Vec::new();
	data.write_vox(&mut bytes).expect("write_vox");

	let world = crate::import::vox::import_vox(&bytes).expect("import");
	let chunk = world.chunk_at([0, 0, 0]).expect("origin chunk populated");
	// The importer swaps Y and Z to convert MagicaVoxel Z-up into ray-vox
	// Y-up, so MV voxel (1, 2, 3) lands at ray-vox (1, 3, 2), and (5, 6, 7)
	// lands at (5, 7, 6).
	let red = chunk.voxel_at(ChunkPos::new(1, 3, 2));
	let blue = chunk.voxel_at(ChunkPos::new(5, 7, 6));
	assert_ne!(red, Material::air());
	assert_ne!(blue, Material::air());
	assert_ne!(red, blue);
}
