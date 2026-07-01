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
