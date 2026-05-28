use crate::chunk::edit::ChunkEdits;
use crate::generate::LazyEdit;
use crate::util::types::ClipmapChunkId;
use crate::world::clipmap::Clipmap;

pub struct Terrain;

impl LazyEdit for Terrain {
	fn generate(&self, _handle: ClipmapChunkId, _clipmap: &Clipmap) -> ChunkEdits {
		todo!()
	}
}
