mod rect;
mod sphere;
mod terrain;

pub use rect::Rect;
pub use sphere::{CheckeredSphere, Sphere};
pub use terrain::Terrain;

use crate::{
	tree::{Aabb, EditPacket, TreePath},
	types::Voxel,
};

pub enum Coverage {
	Full(Voxel),
	Partial,
	Empty,
}

/// A write-only shape. `coverage` is a pure function of geometry — no tree reads.
/// These are evaluated in parallel across chunks during `generate_chunk`.
/// All edit sources ultimately resolve to an `EditPacket` before being applied to the tree.
pub trait Shape: Send + Sync {
	fn aabb(&self) -> Aabb;
	// lod: 0 = single voxel, n = node covers 4^n voxels per side.
	fn coverage(&self, node_aabb: Aabb, lod: u8) -> Coverage;
}

/// A read+write shape. Can query the current tree state during coverage evaluation,
/// enabling edits that depend on previously-applied edits (e.g. topsoiling after terrain).
/// Applied sequentially after all write-only shapes in the same stage are flushed.
/// Not yet implemented.
pub trait ContextShape: Send + Sync {
	fn aabb(&self) -> Aabb;
	// TODO: coverage_with_ctx(&self, node_aabb: Aabb, lod: u8, tree: &Tree) -> Coverage
	// Needs a type-erased tree reference; design TBD when implementing read+write edits.
}

/// Runs the coverage walk for a write-only shape, producing a sorted `EditPacket`.
pub fn edit_packet_for_shape<const DEPTH: usize>(
	shape: &dyn Shape,
	root_aabb: Aabb,
) -> EditPacket<DEPTH> {
	assert!(
		DEPTH <= u8::MAX as usize,
		"shape edit packets only support depths up to {}",
		u8::MAX
	);

	let mut packet = EditPacket::new(true);
	let shape_aabb = shape.aabb();

	if !shape_aabb.overlaps(&root_aabb) {
		return packet;
	}

	let mut path = [0u8; DEPTH];
	collect_shape_edits(shape, shape_aabb, root_aabb, 0, &mut path, &mut packet);

	packet
}

/// Placeholder for read+write shape evaluation. Not yet implemented.
pub fn edit_packet_for_context_shape<const DEPTH: usize>(
	_shape: &dyn ContextShape,
	_root_aabb: Aabb,
) -> EditPacket<DEPTH> {
	todo!("read+write shape evaluation not yet implemented")
}

fn collect_shape_edits<const DEPTH: usize>(
	shape: &dyn Shape,
	shape_aabb: Aabb,
	node_aabb: Aabb,
	depth: usize,
	path: &mut [u8; DEPTH],
	packet: &mut EditPacket<DEPTH>,
) {
	if !shape_aabb.overlaps(&node_aabb) {
		return;
	}

	match shape.coverage(node_aabb, (DEPTH - depth) as u8) {
		Coverage::Empty => {}
		Coverage::Full(voxel) => push_shape_edit(path, depth, voxel, packet),
		Coverage::Partial => {
			debug_assert!(depth < DEPTH, "partial coverage at leaf level");
			for slot in 0u8..64 {
				path[depth] = slot + 1;
				collect_shape_edits(
					shape,
					shape_aabb,
					node_aabb.split_at_slot(slot as u32),
					depth + 1,
					path,
					packet,
				);
			}
		}
	}
}

fn push_shape_edit<const DEPTH: usize>(
	path: &[u8; DEPTH],
	depth: usize,
	voxel: Voxel,
	packet: &mut EditPacket<DEPTH>,
) {
	let mut buf = [0u8; DEPTH];
	buf[..depth].copy_from_slice(&path[..depth]);
	packet.paths.push(TreePath::from_packed(buf));
	packet.values.push(packet.lut.get_or_add(voxel.into()));
}
