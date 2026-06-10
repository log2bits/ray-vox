use super::Model;
use crate::Chunk;
use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::chunk::sources::DiscreteSource;
use crate::util::types::{Aabb, ChunkId, LodLevel, WorldPos};
use dot_vox::{DotVoxData, SceneNode};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ImportError {
	BadVox(String),
}

impl std::fmt::Display for ImportError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			ImportError::BadVox(s) => write!(f, "bad vox file: {}", s),
		}
	}
}

impl std::error::Error for ImportError {}

impl Model {
	pub fn import_vox(bytes: &[u8]) -> Result<Self, ImportError> {
		let data = dot_vox::load_bytes(bytes).map_err(|e| ImportError::BadVox(e.into()))?;

		let palette: Vec<Material> = data.palette.iter()
			.map(|c| Material::from_rgb_pbr_id([c.r, c.g, c.b], 0))
			.collect();

		let mut voxel_buf: VoxelBuf = VoxelBuf::default();
		let origin = WorldPos::new(0, 0, 0);
		if data.scenes.is_empty() {
			for (i, _model) in data.models.iter().enumerate() {
				emit_model_at_corner(&data, i as u32, origin, &palette, &mut voxel_buf);
			}
		} else {
			walk_scene(&data, 0, origin, &palette, &mut voxel_buf);
		}

		if voxel_buf.entries.is_empty() {
			return Ok(Model {
				chunks: HashMap::new(),
				bounds: Aabb::new(origin, origin),
			});
		}
		let (mut min_x, mut min_y, mut min_z) = (i32::MAX, i32::MAX, i32::MAX);
		let (mut max_x, mut max_y, mut max_z) = (i32::MIN, i32::MIN, i32::MIN);
		for &(p, _) in &voxel_buf.entries {
			min_x = min_x.min(p.x()); min_y = min_y.min(p.y()); min_z = min_z.min(p.z());
			max_x = max_x.max(p.x() + 1); max_y = max_y.max(p.y() + 1); max_z = max_z.max(p.z() + 1);
		}
		let shift = WorldPos::new(-min_x, -min_y, -min_z);

		let fine = LodLevel::FINEST;
		let mut by_chunk: HashMap<ChunkId, EditPacket> = HashMap::new();
		for &(world, material) in &voxel_buf.entries {
			let p = world + shift;
			let chunk_id = p.chunk_id(fine);
			let local = p.chunk_pos(chunk_id.origin, fine);
			by_chunk.entry(chunk_id).or_default()
				.push(Path::from_coords(local, 4), material);
		}

		let mut chunks: HashMap<ChunkId, Chunk> = HashMap::new();
		for (id, mut packet) in by_chunk {
			packet.sort();
			let chunk = Chunk::new().edit(&DiscreteSource::new(&packet.edits));
			if !chunk.is_empty() {
				chunks.insert(id, chunk);
			}
		}

		let bounds = Aabb::new(
			WorldPos::new(0, 0, 0),
			WorldPos::new(max_x - min_x, max_y - min_y, max_z - min_z),
		);

		let mut out = Model { chunks, bounds };
		out.build_mip_pyramid();
		Ok(out)
	}
}

#[derive(Default)]
struct VoxelBuf {
	entries: Vec<(WorldPos, Material)>,
}

fn walk_scene(
	data: &DotVoxData,
	node_idx: u32,
	offset: WorldPos,
	palette: &[Material],
	buf: &mut VoxelBuf,
) {
	let node = match data.scenes.get(node_idx as usize) {
		Some(n) => n,
		None => return,
	};
	match node {
		SceneNode::Transform { frames, child, .. } => {
			let mut new_offset = offset;
			if let Some(frame) = frames.first() {
				if let Some(p) = frame.position() {
					new_offset = offset + WorldPos::new(p.x, p.y, p.z);
				}
			}
			walk_scene(data, *child, new_offset, palette, buf);
		}
		SceneNode::Group { children, .. } => {
			for &c in children {
				walk_scene(data, c, offset, palette, buf);
			}
		}
		SceneNode::Shape { models, .. } => {
			for sm in models {
				emit_model(data, sm.model_id, offset, palette, buf);
			}
		}
	}
}

fn emit_model(
	data: &DotVoxData,
	model_id: u32,
	center: WorldPos,
	palette: &[Material],
	buf: &mut VoxelBuf,
) {
	let model = match data.models.get(model_id as usize) {
		Some(m) => m,
		None => return,
	};
	let corner = center - WorldPos::new(
		model.size.x as i32 / 2,
		model.size.y as i32 / 2,
		model.size.z as i32 / 2,
	);
	emit_model_at_corner(data, model_id, corner, palette, buf);
}

fn emit_model_at_corner(
	data: &DotVoxData,
	model_id: u32,
	corner: WorldPos,
	palette: &[Material],
	buf: &mut VoxelBuf,
) {
	let model = match data.models.get(model_id as usize) {
		Some(m) => m,
		None => return,
	};
	for v in &model.voxels {
		let world = corner + WorldPos::new(v.x as i32, v.y as i32, v.z as i32);
		let m = palette.get(v.i as usize).copied().unwrap_or(Material::air());
		if m.is_air() {
			continue;
		}
		buf.entries.push((world, m));
	}
}

