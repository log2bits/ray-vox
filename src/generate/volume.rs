//! Shape-agnostic tree walk for volume edits. A Volume describes geometry only
//! (box-vs-shape classification + per-voxel test); this module owns the
//! traversal, path encoding, and the inlined 4x4x4 leaf level.

pub mod sphere;

use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::util::types::ChunkPos;

/// Box-vs-shape result for one tree cell.
pub enum Coverage {
	Outside,
	Inside,
	Straddle,
}

/// A geometric volume in chunk-local voxel coordinates. Implementors describe
/// their shape; the walk in this module handles tree traversal and emission.
pub trait Volume {
	/// Classify the axis-aligned box `[lo, hi)`.
	fn classify(&self, lo: [i32; 3], hi: [i32; 3]) -> Coverage;
	/// Is the unit voxel at integer coord `v` inside the shape?
	fn contains_voxel(&self, v: [i32; 3]) -> bool;
}

/// Walk a chunk's tree and emit a path-sorted EditPacket painting the volume
/// with `material`.
pub fn walk<V: Volume>(volume: &V, material: Material) -> EditPacket {
	let mut edits = Vec::new();
	emit_cell(volume, material, &mut edits, [0, 0, 0], 0);
	EditPacket::from_sorted(edits)
}

#[inline]
fn emit_cell<V: Volume>(
	volume: &V,
	material: Material,
	edits: &mut Vec<(Path, Material)>,
	origin: [u32; 3],
	depth: u8,
) {
	let side = 1u32 << (2 * (4 - depth));
	let lo = [origin[0] as i32, origin[1] as i32, origin[2] as i32];
	let hi = [lo[0] + side as i32, lo[1] + side as i32, lo[2] + side as i32];

	match volume.classify(lo, hi) {
		Coverage::Outside => return,
		Coverage::Inside => {
			let pos = ChunkPos::new(origin[0] as u8, origin[1] as u8, origin[2] as u8);
			edits.push((Path::from_coords(pos, depth), material));
			return;
		}
		Coverage::Straddle => {}
	}

	if depth == 3 {
		emit_partial_leaf(volume, material, edits, origin);
		return;
	}

	let child_side = side / 4;
	for x in 0..4u32 {
		for y in 0..4u32 {
			for z in 0..4u32 {
				emit_cell(
					volume,
					material,
					edits,
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

/// Voxel-level fill inside one 4-voxel cell. Shares the high Path bytes across
/// all 64 voxels via the depth-4 prefix.
#[inline]
fn emit_partial_leaf<V: Volume>(
	volume: &V,
	material: Material,
	edits: &mut Vec<(Path, Material)>,
	origin: [u32; 3],
) {
	let prefix = depth4_path_prefix(origin[0] as u8, origin[1] as u8, origin[2] as u8);
	let origin_x = origin[0] as i32;
	let origin_y = origin[1] as i32;
	let origin_z = origin[2] as i32;
	for x in 0..4i32 {
		for y in 0..4i32 {
			for z in 0..4i32 {
				if volume.contains_voxel([origin_x + x, origin_y + y, origin_z + z]) {
					let leaf_byte = (((x as u32) << 4) | ((y as u32) << 2) | z as u32) + 1;
					edits.push((Path::from(prefix | leaf_byte), material));
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
