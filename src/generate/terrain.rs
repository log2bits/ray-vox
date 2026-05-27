use crate::chunk::edit::ChunkEdits;
use crate::generate::LazyEdit;
use crate::world::clipmap::{ChunkHandle, Clipmap};

pub struct Terrain;

impl LazyEdit for Terrain {
    fn generate(&self, _handle: ChunkHandle, _clipmap: &Clipmap) -> ChunkEdits {
        todo!()
    }
}
