
use super::clipmap::RemapOp;
use super::World;
use crate::Chunk;
use crate::chunk::material::Material;
use crate::generate::volume::sphere::Sphere;
use crate::util::types::{ChunkHandle, ChunkId, ChunkPos, LodLevel, WorldPos};
use std::sync::Arc;

fn mat(v: u32) -> Material { Material::from(v) }

fn finest_origin() -> (ChunkHandle, ChunkId) {
	let lod = LodLevel::FINEST;
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0), lod);
	let handle = chunk_id.handle();
	(handle, chunk_id)
}

#[test]
fn apply_edit_registers_at_overlapping_resident_handles_only() {
	let mut world = World::new();
	let (h0, id0) = finest_origin();
	world.clipmap.assign(h0, id0);

	let far_id = ChunkId::new(WorldPos::new(1_000_000, 0, 0), LodLevel::FINEST);
	let h_far = far_id.handle();
	world.clipmap.assign(h_far, far_id);

	let s = Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 20, mat(0x11111140)));
	world.apply_edit(s);

	assert_eq!(world.by_handle.get(&h0).map(|v| v.len()), Some(1));
	assert!(world.by_handle.get(&h_far).is_none());
}

#[test]
fn bake_applies_edits_in_insertion_order() {
	let mut world = World::new();
	let (h, id) = finest_origin();
	world.clipmap.assign(h, id);

	let red = mat(0x11111140);
	let blue = mat(0x22222240);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 40, red)));
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 40, blue)));

	let chunk = world.bake(h);
	assert_eq!(chunk.voxel_at(ChunkPos::new(128, 128, 128)), blue);
}

#[test]
fn bake_of_resident_handle_with_no_edits_is_empty() {
	let mut world = World::new();
	let (h, id) = finest_origin();
	world.clipmap.assign(h, id);

	let chunk = world.bake(h);
	assert!(chunk.is_empty());
}

#[test]
fn remap_add_seeds_by_handle_from_existing_edit_log() {
	let mut world = World::new();
	let s = Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 20, mat(0x11111140)));
	world.apply_edit(s);
	assert!(world.by_handle.is_empty());

	let (h, id) = finest_origin();
	world.process_remap(&RemapOp::Add(h, id));
	assert_eq!(world.by_handle.get(&h).map(|v| v.len()), Some(1));
}

#[test]
fn remap_delete_drops_by_handle_entry() {
	let mut world = World::new();
	let (h, id) = finest_origin();
	world.clipmap.assign(h, id);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 20, mat(0x11111140))));
	assert!(world.by_handle.contains_key(&h));

	world.process_remap(&RemapOp::Delete(h));
	assert!(!world.by_handle.contains_key(&h));
	assert!(world.clipmap.chunk_id_of(h).is_none());
}

#[test]
fn bake_unassigned_handle_returns_empty_chunk() {
	let world = World::new();
	let handle = ChunkHandle::new(LodLevel::FINEST, 0, 0, 0);
	let chunk = world.bake(handle);
	assert!(chunk.is_empty());
}

#[test]
fn set_origin_with_no_movement_after_full_populate_produces_no_ops() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	let ops: Vec<_> = world.clipmap.pending_remap.drain(..).collect();
	for op in &ops {
		world.process_remap(op);
	}
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	assert!(
		world.clipmap.pending_remap.is_empty(),
		"got {} pending ops on a no-op set_origin",
		world.clipmap.pending_remap.len(),
	);
}

#[test]
fn first_set_origin_populates_every_slot_at_every_lod() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));

	let expected =
		LodLevel::LEVELS as usize * LodLevel::CHUNKS_PER_LEVEL as usize;
	let adds = world.clipmap.pending_remap.iter()
		.filter(|op| matches!(op, RemapOp::Add(_, _)))
		.count();
	let deletes = world.clipmap.pending_remap.iter()
		.filter(|op| matches!(op, RemapOp::Delete(_)))
		.count();
	assert_eq!(adds, expected, "fresh world should produce one Add per slot");
	assert_eq!(deletes, 0, "no prior residency, no Deletes expected");
}

#[test]
fn set_origin_moving_one_finest_chunk_shifts_a_single_finest_face() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	let initial: Vec<_> = world.clipmap.pending_remap.drain(..).collect();
	for op in &initial {
		world.process_remap(op);
	}

	let shift = LodLevel::FINEST.chunk_size();
	world.clipmap.set_origin(WorldPos::new(shift, 0, 0));

	let face = LodLevel::GRID_SIZE * LodLevel::GRID_SIZE;
	let adds = world.clipmap.pending_remap.iter()
		.filter(|op| matches!(op, RemapOp::Add(_, _)))
		.count();
	let deletes = world.clipmap.pending_remap.iter()
		.filter(|op| matches!(op, RemapOp::Delete(_)))
		.count();
	assert_eq!(adds as u32, face);
	assert_eq!(deletes as u32, face);
}

#[test]
fn chunk_pool_apply_remap_drops_chunk_on_delete() {
	let mut world = World::new();
	let (h, id) = finest_origin();
	world.clipmap.assign(h, id);
	world.chunk_pool.insert(h, Chunk::new());
	assert!(world.chunk_pool.contains(h));

	world.process_remap(&RemapOp::Delete(h));
	assert!(!world.chunk_pool.contains(h));
	assert!(world.chunk_pool.allocations.get(&h).is_none());
}

#[test]
fn process_remap_keeps_all_three_subsystems_in_sync_on_add() {
	let mut world = World::new();
	let (h, id) = finest_origin();

	let s = Arc::new(Sphere::new(WorldPos::new(128, 128, 128), 20, mat(0x10)));
	world.apply_edit(s);
	assert!(world.by_handle.is_empty());

	world.process_remap(&RemapOp::Add(h, id));

	assert_eq!(world.clipmap.chunk_id_of(h), Some(id));
	assert!(!world.chunk_pool.contains(h));
	assert_eq!(world.by_handle.get(&h).map(|v| v.len()), Some(1));

	let chunk = world.bake(h);
	world.chunk_pool.insert(h, chunk);
	assert!(world.chunk_pool.contains(h));
}


#[test]
fn empty_world_drive_remaps_creates_no_chunks() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	assert!(!world.clipmap.pending_remap.is_empty());

	world.drive_remaps(usize::MAX);

	assert!(world.clipmap.resident.is_empty(), "no chunks should be resident");
	assert!(world.chunk_pool.chunks.is_empty(), "no chunks in pool");
	assert!(world.by_handle.is_empty(), "no edit registrations");
	assert!(world.clipmap.pending_remap.is_empty(), "all ops drained");
}

#[test]
fn single_sphere_edit_populates_only_overlapping_chunks() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	world.drive_remaps(usize::MAX);
	assert!(world.clipmap.resident.is_empty());

	let sphere = Sphere::new(WorldPos::new(0, 0, 0), 50, mat(0x11111140));
	world.apply_edit(Arc::new(sphere));
	world.drive_remaps(usize::MAX);

	let count = world.clipmap.resident.len();
	assert!(count > 0, "sphere should create at least one chunk");
	assert!(count < 100, "got {} chunks for a single small sphere", count);
	assert_eq!(count, world.chunk_pool.chunks.len(), "pool tracks resident set");

	let bounds = world.edits[0].bounds();
	for (_, id) in world.clipmap.resident_chunks() {
		assert!(id.aabb().intersects(&bounds), "non-overlapping chunk resident");
	}
}

#[test]
fn drive_remaps_respects_budget() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	world.drive_remaps(usize::MAX);

	let sphere = Sphere::new(WorldPos::new(0, 0, 0), 1024, mat(0x10));
	world.apply_edit(Arc::new(sphere));
	let queued = world.clipmap.pending_remap.len();
	assert!(queued > 5, "test needs more than 5 Adds queued, got {}", queued);

	world.drive_remaps(5);
	assert!(world.clipmap.resident.len() <= 5,
		"more chunks baked than budget allowed");
	assert!(world.clipmap.pending_remap.len() < queued,
		"some ops should have been processed");
}

#[test]
fn apply_edit_to_already_baked_chunk_marks_dirty_and_rebakes() {
	let mut world = World::new();
	world.clipmap.set_origin(WorldPos::new(0, 0, 0));
	world.drive_remaps(usize::MAX);

	let red = mat(0x11111140);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(0, 0, 0), 50, red)));
	world.drive_remaps(usize::MAX);
	let initial_count = world.clipmap.resident.len();
	assert!(initial_count > 0);

	let (handle, _) = world.clipmap.resident_chunks()
		.find(|(_, id)| id.lod == LodLevel::FINEST)
		.expect("at least one finest chunk should be resident");
	let red_chunk = world.chunk_pool.get(handle).expect("baked").clone();

	let blue = mat(0x22222240);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(0, 0, 0), 50, blue)));
	assert!(world.needs_rebake.contains(&handle),
		"already-resident handle should be marked needs_rebake after overlapping apply_edit");

	world.drive_remaps(usize::MAX);
	let blue_chunk = world.chunk_pool.get(handle).expect("still baked");
	let centre_voxel_red = red_chunk.voxel_at(ChunkPos::new(0, 0, 0));
	let centre_voxel_blue = blue_chunk.voxel_at(ChunkPos::new(0, 0, 0));
	if centre_voxel_red != Material::air() {
		assert_eq!(centre_voxel_blue, blue, "rebake should produce blue at centre");
	}
}

#[test]
fn parallel_bake_results_match_sequential() {
	let make_world = || {
		let mut w = World::new();
		w.clipmap.set_origin(WorldPos::new(0, 0, 0));
		w.drive_remaps(usize::MAX);
		w.apply_edit(Arc::new(Sphere::new(WorldPos::new(0, 0, 0), 80, mat(0x11111140))));
		w.apply_edit(Arc::new(Sphere::new(WorldPos::new(64, 0, 0), 64, mat(0x22222240))));
		w.apply_edit(Arc::new(Sphere::new(WorldPos::new(0, 64, 0), 48, Material::air())));
		w
	};

	let mut parallel = make_world();
	parallel.drive_remaps(usize::MAX);

	let mut sequential = make_world();
	while !sequential.clipmap.pending_remap.is_empty()
		|| !sequential.needs_rebake.is_empty()
	{
		sequential.drive_remaps(1);
	}

	assert_eq!(parallel.clipmap.resident.len(), sequential.clipmap.resident.len());
	assert_eq!(parallel.chunk_pool.chunks.len(), sequential.chunk_pool.chunks.len());

	for (handle, id) in parallel.clipmap.resident_chunks() {
		let p_chunk = parallel.chunk_pool.get(handle).expect("parallel chunk");
		let s_chunk = sequential.chunk_pool.get(handle).expect("sequential chunk");
		assert_eq!(sequential.clipmap.chunk_id_of(handle), Some(id));
		for &(x, y, z) in &[(0u8, 0, 0), (128, 128, 128), (255, 255, 255), (64, 64, 64)] {
			let pos = ChunkPos::new(x, y, z);
			assert_eq!(p_chunk.voxel_at(pos), s_chunk.voxel_at(pos),
				"mismatch at handle {:?} pos {:?}", handle, pos);
		}
	}
}
