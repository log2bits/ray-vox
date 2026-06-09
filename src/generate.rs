pub mod model;
pub mod volume;

use crate::chunk::Chunk;
use crate::chunk::build::CHUNK_SIDE;
use crate::chunk::sources::{ChunkSource, CompositeSource, LocalEdit, Overlay};
use crate::util::types::{Aabb, ChunkId};

pub trait Edit: Send + Sync {
	fn bounds(&self) -> Aabb;

	fn make_local<'a>(&'a self, chunk_id: ChunkId) -> Option<Box<dyn LocalEdit + 'a>>;

	fn apply(&self, chunk_id: ChunkId, base: Chunk) -> Chunk {
		let local = match self.make_local(chunk_id) {
			Some(l) => l,
			None => return base,
		};
		let locals: [Box<dyn LocalEdit + '_>; 1] = [local];
		let composite = CompositeSource::new(&locals, CHUNK_SIDE);
		let chunk_source = ChunkSource::new(&base);
		let overlay = Overlay::new(chunk_source, composite);
		crate::chunk::build::build_chunk(&overlay)
	}
}
