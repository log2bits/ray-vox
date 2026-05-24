pub mod chunk_pool;
pub mod clipmap;
pub mod pbr;

use crate::util::Lut;
use chunk_pool::ChunkPool;
pub use pbr::Pbr;

use crate::Chunk;
use clipmap::Clipmap;

pub struct World {
	pub chunk_pool: ChunkPool,
	pub persistent_chunks: Vec<Chunk>,
	pub clipmap: Clipmap,
	pub pbr_lut: Lut<Pbr>,
}
