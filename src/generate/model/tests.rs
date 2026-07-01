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
		Voxel { x: 1, y: 2, z: 3, i: 0 },
		Voxel { x: 5, y: 6, z: 7, i: 1 },
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

	let model = crate::import::vox::import_vox(&bytes).expect("import");
	assert!(model.chunks.len() > 0, "model should have chunks");

	let finest_id = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);
	let fine = model.chunks.get(&finest_id).expect("finest chunk");
	let red = fine.voxel_at(ChunkPos::new(1, 2, 3));
	let blue = fine.voxel_at(ChunkPos::new(5, 6, 7));
	assert_ne!(red, Material::air(), "red voxel should be set");
	assert_ne!(blue, Material::air(), "blue voxel should be set");
	assert_ne!(red, blue, "red and blue should be different materials");
}
