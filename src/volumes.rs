pub mod rect;
pub mod sphere;
pub mod terrain;

use crate::chunk::edit::{Edits, Path};
use crate::chunk::material::Material;

pub use rect::Rect;
pub use sphere::Sphere;

pub enum Containment {
	/// Volume does not overlap this AABB at all.
	Empty,
	/// Volume partially overlaps — subdivide and test children.
	Partial,
	/// Volume fully contains this AABB — fill it without recursing.
	Full,
}

pub trait Volume {
	fn material(&self) -> Material;
	fn containment(&self, min: [i32; 3], max: [i32; 3]) -> Containment;
}

/// Pushes edits into `edits` that fill the voxels covered by `volume`.
///
/// The recursive descent mirrors the chunk's 4-level tree structure so fully-covered
/// subtrees are expressed as a single coarse edit rather than individual voxels.
/// Call `chunk.apply_edits()` after all volumes have been stamped.
pub fn stamp(volume: &impl Volume, edits: &mut Edits) {
	stamp_cell(volume, edits, [0, 0, 0], 0);
}



/// `depth` 0 = whole chunk (256³), 1–4 mirror the tree levels.
fn stamp_cell(volume: &impl Volume, edits: &mut Edits, origin: [i32; 3], depth: u8) {
	// Cell size in voxels: 256, 64, 16, 4, 1
	let size = 256i32 >> (depth * 2);
	let max = [origin[0] + size, origin[1] + size, origin[2] + size];

	match volume.containment(origin, max) {
		Containment::Empty => {}
		Containment::Full => {
			let path = if depth == 0 {
				Path::from(0u32) // root fill
			} else {
				Path::from_coords([origin[0] as u8, origin[1] as u8, origin[2] as u8], depth)
			};
			edits.push(path, volume.material());
		}
		Containment::Partial => {
			if depth == 4 {
				// Single voxel — Partial shouldn't happen but treat as Full.
				let path =
					Path::from_coords([origin[0] as u8, origin[1] as u8, origin[2] as u8], 4);
				edits.push(path, volume.material());
			} else {
				let child_size = size / 4;
				for dx in 0..4i32 {
					for dy in 0..4i32 {
						for dz in 0..4i32 {
							stamp_cell(
								volume,
								edits,
								[
									origin[0] + dx * child_size,
									origin[1] + dy * child_size,
									origin[2] + dz * child_size,
								],
								depth + 1,
							);
						}
					}
				}
			}
		}
	}
}
