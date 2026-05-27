pub mod chunk_pool;
pub mod clipmap;
pub mod pbr;

use crate::util::Lut;
use chunk_pool::ChunkPool;
pub use pbr::Pbr;

use crate::volumes::Volume;
use clipmap::Clipmap;

pub struct World {
	pub volumes: Vec<Box<dyn Volume>>,
	pub chunk_pool: ChunkPool,
	pub clipmap: Clipmap,
	pub pbr_lut: Lut<Pbr>,
}
