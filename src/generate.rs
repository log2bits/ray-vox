pub mod model;
pub mod volume;

use crate::chunk::Chunk;
use crate::util::types::{Aabb, ChunkId};

pub trait Edit: Send + Sync {
	fn bounds(&self) -> Aabb;
	fn apply(&self, chunk_id: ChunkId, base: Chunk) -> Chunk;
}
