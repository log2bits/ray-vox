use crate::chunk::material::Material;
use crate::generate::model::{Model, ModelBuilder, WorldEdit};
use crate::util::types::WorldPos;
use dot_vox::{DotVoxData, SceneNode};
use rayon::prelude::*;

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

pub fn import_vox(bytes: &[u8]) -> Result<Model, ImportError> {
	let data = dot_vox::load_bytes(bytes).map_err(|e| ImportError::BadVox(e.into()))?;

	let palette: Vec<Material> = data.palette.iter()
		.map(|c| Material::from_rgb_pbr_id([c.r, c.g, c.b], 0))
		.collect();

	// Scene traversal is cheap (no voxel iteration). Collect the placements
	// here, then emit voxels into the builder in parallel.
	let mut instances: Vec<Instance> = Vec::new();
	let origin = WorldPos::new(0, 0, 0);
	if data.scenes.is_empty() {
		for i in 0..data.models.len() {
			instances.push(Instance { model_id: i as u32, corner: origin });
		}
	} else {
		walk_scene(&data, 0, origin, &mut instances);
	}

	let builder = ModelBuilder::new();
	instances.par_iter().for_each(|inst| {
		let model = match data.models.get(inst.model_id as usize) {
			Some(m) => m,
			None => return,
		};
		for v in &model.voxels {
			let m = match palette.get(v.i as usize) {
				Some(m) if !m.is_air() => *m,
				_ => continue,
			};
			let pos = inst.corner + WorldPos::new(v.x as i32, v.y as i32, v.z as i32);
			builder.add(WorldEdit { pos, material: m });
		}
	});

	Ok(builder.bake())
}

#[derive(Clone, Copy)]
struct Instance {
	model_id: u32,
	corner: WorldPos,
}

fn walk_scene(data: &DotVoxData, node_idx: u32, offset: WorldPos, out: &mut Vec<Instance>) {
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
			walk_scene(data, *child, new_offset, out);
		}
		SceneNode::Group { children, .. } => {
			for &c in children {
				walk_scene(data, c, offset, out);
			}
		}
		SceneNode::Shape { models, .. } => {
			for sm in models {
				if let Some(m) = data.models.get(sm.model_id as usize) {
					let corner = offset - WorldPos::new(
						m.size.x as i32 / 2,
						m.size.y as i32 / 2,
						m.size.z as i32 / 2,
					);
					out.push(Instance { model_id: sm.model_id, corner });
				}
			}
		}
	}
}
