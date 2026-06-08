pub mod chunk_pool;
pub mod clipmap;
pub mod pbr;

use crate::Chunk;
use crate::util::Lut;
use crate::util::types::WorldPos;
use chunk_pool::ChunkPool;
pub use pbr::Pbr;
use std::collections::HashMap;

use crate::generate::Edit;
use clipmap::Clipmap;

pub struct World {
	pub edits: Vec<Box<dyn Edit>>,
	pub chunk_pool: ChunkPool,
	pub persistent_chunks: HashMap<WorldPos, Chunk>,
	pub clipmap: Clipmap,
	pub pbr_lut: Lut<Pbr>,
}
