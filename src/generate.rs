pub mod model;
pub mod volume;

use crate::chunk::edit::EditPacket;
use crate::util::types::{Aabb, ChunkId};

/// Resolution-independent description of a world modification.
///
/// An Edit is sampled per chunk at that chunk's LOD: the cost scales with the
/// output (how many voxels the chunk has at its level), not with the shape's
/// nominal size. A trillion-voxel sphere sampled against a coarse distant chunk
/// produces a coarse cheap packet; the same sphere against a fine chunk near the
/// camera produces full detail. The clipmap's chunk-LOD assignment is what makes
/// distance-based cost falloff automatic.
///
/// The world keeps edits in a list, culls by `bounds()` overlap when generating
/// a chunk, and calls `sample` on each surviving edit.
pub trait Edit: Send + Sync {
	/// World-space extent. Used to skip chunks the edit can't possibly touch.
	fn bounds(&self) -> Aabb;

	/// This edit's contribution to one chunk, sampled at the chunk's LOD.
	/// Returns an empty packet when the edit produces no voxels in this chunk.
	fn sample(&self, chunk: ChunkId) -> EditPacket;
}
