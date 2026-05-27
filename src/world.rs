pub mod chunk_pool;
pub mod clipmap;
pub mod pbr;

use std::collections::HashMap;

use crate::Chunk;
use crate::util::Lut;
use chunk_pool::ChunkPool;
pub use pbr::Pbr;

use crate::generate::LazyEdit;
use clipmap::Clipmap;

pub struct World {
	pub layers: Vec<Box<dyn LazyEdit>>,
	pub chunk_pool: ChunkPool,
	pub persistent_chunks: HashMap<WorldPosition, Chunk>,
	pub clipmap: Clipmap,
	pub pbr_lut: Lut<Pbr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldPosition {
	pub position: [i32; 3],
}

impl WorldPosition {
	pub fn new(x: i32, y: i32, z: i32) -> Self {
		Self {
			position: [x, y, z],
		}
	}
}

impl From<[i32; 3]> for WorldPosition {
	fn from(position: [i32; 3]) -> Self {
		Self { position }
	}
}
