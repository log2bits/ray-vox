use crate::chunk::material::Material;
use crate::util::types::WorldPos;
use crate::world::{World, WorldEdit};
use dot_vox::{DotVoxData, Rotation, SceneNode};
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

pub fn import_vox(bytes: &[u8]) -> Result<World, ImportError> {
	let data = dot_vox::load_bytes(bytes).map_err(|e| ImportError::BadVox(e.into()))?;

	let palette: Vec<Material> = data.palette.iter()
		.map(|c| Material::from_rgb_pbr_id([c.r, c.g, c.b], 0))
		.collect();

	// Scene traversal is cheap; collect placements first, emit voxels in parallel.
	let mut instances: Vec<Instance> = Vec::new();
	if data.scenes.is_empty() {
		for i in 0..data.models.len() {
			instances.push(Instance {
				model_id: i as u32,
				corner: WorldPos::new(0, 0, 0),
				rotation_cols: IDENTITY_ROTATION,
			});
		}
	} else {
		walk_scene(&data, 0, WorldPos::new(0, 0, 0), Rotation::IDENTITY, &mut instances);
	}

	let world_edits: Vec<WorldEdit> = instances
		.par_iter()
		.flat_map_iter(|instance| {
			let model = data.models.get(instance.model_id as usize);
			let palette = &palette;
			let corner = instance.corner;
			let rotation_cols = instance.rotation_cols;
			model
				.into_iter()
				.flat_map(|m| m.voxels.iter())
				.filter_map(move |v| {
					let material = *palette.get(v.i as usize)?;
					if material.is_air() {
						return None;
					}
					// Rotate around the shape origin, then translate to the
					// corner (MagicaVoxel Z-up frame).
					let local = [v.x as i32, v.y as i32, v.z as i32];
					let rotated = apply_rotation_cols(rotation_cols, local);
					let mv_pos = corner + WorldPos::new(rotated[0], rotated[1], rotated[2]);
					// Swap Y and Z at the very end to convert MV Z-up to ray-vox Y-up.
					let pos = WorldPos::new(mv_pos.x(), mv_pos.z(), mv_pos.y());
					Some(WorldEdit { pos, material })
				})
		})
		.collect();

	Ok(World::from_edits(world_edits))
}

// One shape placement pulled out of the scene graph. corner is where the
// shape's local (0, 0, 0) lands after all transforms; rotation_cols is the
// accumulated rotation as an integer matrix so it stays a cheap per-voxel op.
#[derive(Clone, Copy)]
struct Instance {
	model_id: u32,
	corner: WorldPos,
	rotation_cols: [[i8; 3]; 3],
}

// Identity in the same column-major shape rotation_to_i8_cols produces.
const IDENTITY_ROTATION: [[i8; 3]; 3] = [
	[1, 0, 0],
	[0, 1, 0],
	[0, 0, 1],
];

// Walk the vox scene graph, accumulating translation and rotation through
// every enclosing Transform. MagicaVoxel semantics: a child Transform with
// local (T_local, R_local) inside parent (T, R) yields T + R * T_local and
// R * R_local. A shape's world corner is T - R * (size / 2) because the
// transform position marks the shape's rotated center.
fn walk_scene(
	data: &DotVoxData,
	node_idx: u32,
	accumulated_translation: WorldPos,
	accumulated_rotation: Rotation,
	out: &mut Vec<Instance>,
) {
	let node = match data.scenes.get(node_idx as usize) {
		Some(n) => n,
		None => return,
	};
	match node {
		SceneNode::Transform { frames, child, .. } => {
			let mut new_translation = accumulated_translation;
			let mut new_rotation = accumulated_rotation;
			if let Some(frame) = frames.first() {
				if let Some(local_position) = frame.position() {
					let local_translation = WorldPos::new(
						local_position.x,
						local_position.y,
						local_position.z,
					);
					let rotated_local_translation =
						rotate_worldpos(local_translation, accumulated_rotation);
					new_translation = accumulated_translation + rotated_local_translation;
				}
				if let Some(local_rotation) = frame.orientation() {
					new_rotation = accumulated_rotation * local_rotation;
				}
			}
			walk_scene(data, *child, new_translation, new_rotation, out);
		}
		SceneNode::Group { children, .. } => {
			for &c in children {
				walk_scene(data, c, accumulated_translation, accumulated_rotation, out);
			}
		}
		SceneNode::Shape { models, .. } => {
			let rotation_cols = rotation_to_i8_cols(accumulated_rotation);
			for sm in models {
				if let Some(m) = data.models.get(sm.model_id as usize) {
					let half_size = WorldPos::new(
						m.size.x as i32 / 2,
						m.size.y as i32 / 2,
						m.size.z as i32 / 2,
					);
					let rotated_half_size = rotate_worldpos(half_size, accumulated_rotation);
					let corner = accumulated_translation - rotated_half_size;
					out.push(Instance {
						model_id: sm.model_id,
						corner,
						rotation_cols,
					});
				}
			}
		}
	}
}

fn rotate_worldpos(v: WorldPos, rotation: Rotation) -> WorldPos {
	let cols = rotation_to_i8_cols(rotation);
	let rotated = apply_rotation_cols(cols, [v.x(), v.y(), v.z()]);
	WorldPos::new(rotated[0], rotated[1], rotated[2])
}

// dot_vox's Rotation is a byte-packed signed permutation matrix. Entries are
// always -1, 0, or 1, so an i8 3x3 column-major matrix stores it losslessly.
fn rotation_to_i8_cols(rotation: Rotation) -> [[i8; 3]; 3] {
	let f = rotation.to_cols_array_2d();
	let mut cols = [[0i8; 3]; 3];
	for i in 0..3 {
		for j in 0..3 {
			cols[i][j] = f[i][j] as i8;
		}
	}
	cols
}

// Apply cols to v. Output row j = sum over columns i of cols[i][j] * v[i].
fn apply_rotation_cols(cols: [[i8; 3]; 3], v: [i32; 3]) -> [i32; 3] {
	let mut result = [0i32; 3];
	for j in 0..3 {
		result[j] = (cols[0][j] as i32) * v[0]
			+ (cols[1][j] as i32) * v[1]
			+ (cols[2][j] as i32) * v[2];
	}
	result
}
