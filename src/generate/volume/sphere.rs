//! Sphere volume generator. Emits a path-sorted EditPacket that paints or carves
//! a sphere into a chunk at any LOD.
//!
//! Tree-walk classification: at each cell, check closest and farthest corners to
//! the sphere center. Outside skips. Inside emits one fill at the current depth.
//! Partial recurses, with the leaf level inlined for speed.

use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::util::types::{ChunkId, ChunkPos, WorldPos};

pub struct Sphere;

impl Sphere {
	/// Build edits to paint a sphere of the given radius centered at world_center
	/// into the given chunk. The returned packet is path-sorted.
	pub fn generate(
		radius: i32,
		chunk: ChunkId,
		world_center: WorldPos,
		material: Material,
	) -> EditPacket {
		let voxel_size = chunk.lod.voxel_size();
		// i32 throughout so out-of-chunk centers can't underflow.
		let center = [
			(world_center.x() - chunk.origin.x()) / voxel_size,
			(world_center.y() - chunk.origin.y()) / voxel_size,
			(world_center.z() - chunk.origin.z()) / voxel_size,
		];
		let r = radius / voxel_size;
		if r <= 0 {
			return EditPacket::from_sorted(Vec::new());
		}
		let mut edits = Vec::new();
		emit_cell(&mut edits, center, r * r, material, [0, 0, 0], 0);
		EditPacket::from_sorted(edits)
	}
}

#[inline]
fn emit_cell(
	edits: &mut Vec<(Path, Material)>,
	center: [i32; 3],
	r_sq: i32,
	material: Material,
	origin: [u32; 3],
	depth: u8,
) {
	let side = 1u32 << (2 * (4 - depth));
	let lo = [origin[0] as i32, origin[1] as i32, origin[2] as i32];
	let hi = [lo[0] + side as i32, lo[1] + side as i32, lo[2] + side as i32];

	let mut near_sq = 0;
	let mut far_sq = 0;
	for i in 0..3 {
		let c = center[i];
		let near = c.clamp(lo[i], hi[i]) - c;
		near_sq += near * near;
		let d_lo = lo[i] - c;
		let d_hi = hi[i] - c;
		let far = if d_lo * d_lo > d_hi * d_hi { d_lo } else { d_hi };
		far_sq += far * far;
	}

	if near_sq > r_sq {
		return;
	}
	if far_sq <= r_sq {
		let pos = ChunkPos::new(origin[0] as u8, origin[1] as u8, origin[2] as u8);
		edits.push((Path::from_coords(pos, depth), material));
		return;
	}

	if depth == 3 {
		emit_partial_leaf(edits, center, r_sq, material, origin);
		return;
	}

	let child_side = side / 4;
	for x in 0..4u32 {
		for y in 0..4u32 {
			for z in 0..4u32 {
				emit_cell(
					edits,
					center,
					r_sq,
					material,
					[
						origin[0] + x * child_side,
						origin[1] + y * child_side,
						origin[2] + z * child_side,
					],
					depth + 1,
				);
			}
		}
	}
}

/// Voxel-level fill inside one 4-voxel cell. Hoists per-axis distance squared
/// out of inner loops and shares the high Path bytes across all 64 voxels.
#[inline]
fn emit_partial_leaf(
	edits: &mut Vec<(Path, Material)>,
	center: [i32; 3],
	r_sq: i32,
	material: Material,
	origin: [u32; 3],
) {
	let x0 = origin[0] as u8;
	let y0 = origin[1] as u8;
	let z0 = origin[2] as u8;
	let prefix = depth4_path_prefix(x0, y0, z0);

	let dx0 = origin[0] as i32 - center[0];
	let dy0 = origin[1] as i32 - center[1];
	let dz0 = origin[2] as i32 - center[2];

	for vx in 0..4i32 {
		let dx = dx0 + vx;
		let dx_sq = dx * dx;
		for vy in 0..4i32 {
			let dy = dy0 + vy;
			let dxy_sq = dx_sq + dy * dy;
			for vz in 0..4i32 {
				let dz = dz0 + vz;
				if dxy_sq + dz * dz <= r_sq {
					let b3 = (((vx as u32) << 4) | ((vy as u32) << 2) | vz as u32) + 1;
					edits.push((Path::from(prefix | b3), material));
				}
			}
		}
	}
}

/// Top three Path bytes for the depth-4 cell at this corner. Corner must be
/// aligned to a 4-voxel boundary.
#[inline]
fn depth4_path_prefix(x: u8, y: u8, z: u8) -> u32 {
	let slot = |shift: u8| -> u32 {
		((((x as u32 >> shift) & 3) << 4) | (((y as u32 >> shift) & 3) << 2) | ((z as u32 >> shift) & 3)) + 1
	};
	let b0 = slot(6);
	let b1 = slot(4);
	let b2 = slot(2);
	(b0 << 24) | (b1 << 16) | (b2 << 8)
}
