pub mod import;
pub mod terrain;
pub mod volume;

use crate::chunk::edit::{EditPacket, Path};
use crate::chunk::material::Material;
use crate::util::types::{ChunkId, WorldPos};
use crate::world::clipmap::Clipmap;
use std::array::from_fn;

/// Lazy edits span the world but are evaluated one chunk at a time on demand.
/// Terrain, volumes, and other procedural content implement this.
pub trait LazyEdit: Send + Sync {
	fn generate(&self, handle: ChunkId) -> EditPacket;
}

/// Eager edits are evaluated once upfront and produce edits for every chunk they cover.
/// The world applies these to persistent chunks immediately and discards the generator.
pub trait EagerEdit: Send + Sync {
	fn generate(&self) -> Vec<EditPacket>;
}

pub enum CellState {
	Empty,
	Uniform(Material),
	Subdivide,
}
