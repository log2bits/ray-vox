use crate::chunk::edit::ChunkEdits;
use crate::generate::EagerEdit;
use crate::world::clipmap::Clipmap;

pub struct VoxelImport;

impl EagerEdit for VoxelImport {
    fn generate(&self, _clipmap: &Clipmap) -> Vec<ChunkEdits> {
        todo!()
    }
}
