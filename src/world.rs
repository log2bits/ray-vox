pub mod pbr;

#[cfg(test)]
mod tests;

use crate::Chunk;
use crate::generate::Edit;
use crate::util::Lut;
use crate::util::types::{Aabb, ChunkId, CHUNK_SIZE, WorldPos};
pub use pbr::Pbr;
use std::sync::Arc;

// A fixed 3D grid of chunks in world space. The grid covers
// `[origin, origin + chunk_grid_dim * CHUNK_SIZE)` along each axis.
// Chunk slots are either baked (`Some`) or empty air (`None`).
pub struct World {
	pub chunk_grid_dim: [u32; 3],
	pub origin: WorldPos,
	pub edits: Vec<Arc<dyn Edit>>,
	pub chunks: Vec<Option<Chunk>>,
	pub pbr_lut: Lut<Pbr>,
}

impl World {
	pub fn new(chunk_grid_dim: [u32; 3]) -> Self {
		Self::with_origin(chunk_grid_dim, WorldPos::new(0, 0, 0))
	}

	pub fn with_origin(chunk_grid_dim: [u32; 3], origin: WorldPos) -> Self {
		let len = chunk_grid_dim[0] as usize
			* chunk_grid_dim[1] as usize
			* chunk_grid_dim[2] as usize;
		let mut chunks = Vec::with_capacity(len);
		chunks.resize_with(len, || None);
		Self {
			chunk_grid_dim,
			origin,
			edits: Vec::new(),
			chunks,
			pbr_lut: Lut::new(),
		}
	}

	pub fn chunk_slot_count(&self) -> usize {
		self.chunks.len()
	}

	// Flatten a 3D grid position into the storage index. Returns None if the
	// position is out of bounds.
	pub fn slot_index(&self, grid_pos: [u32; 3]) -> Option<usize> {
		if grid_pos[0] >= self.chunk_grid_dim[0]
			|| grid_pos[1] >= self.chunk_grid_dim[1]
			|| grid_pos[2] >= self.chunk_grid_dim[2]
		{
			return None;
		}
		let [gx, gy, gz] = grid_pos;
		let [dx, dy, _] = self.chunk_grid_dim;
		Some(gx as usize + gy as usize * dx as usize + gz as usize * dx as usize * dy as usize)
	}

	pub fn slot_grid_pos(&self, index: usize) -> [u32; 3] {
		let [dx, dy, _] = self.chunk_grid_dim;
		let dx = dx as usize;
		let dy = dy as usize;
		let z = index / (dx * dy);
		let rem = index % (dx * dy);
		let y = rem / dx;
		let x = rem % dx;
		[x as u32, y as u32, z as u32]
	}

	pub fn chunk_world_origin(&self, grid_pos: [u32; 3]) -> WorldPos {
		WorldPos::new(
			self.origin.x() + grid_pos[0] as i32 * CHUNK_SIZE,
			self.origin.y() + grid_pos[1] as i32 * CHUNK_SIZE,
			self.origin.z() + grid_pos[2] as i32 * CHUNK_SIZE,
		)
	}

	pub fn chunk_id_at(&self, grid_pos: [u32; 3]) -> ChunkId {
		ChunkId::new(self.chunk_world_origin(grid_pos))
	}

	pub fn chunk_at(&self, grid_pos: [u32; 3]) -> Option<&Chunk> {
		self.slot_index(grid_pos).and_then(|i| self.chunks[i].as_ref())
	}

	pub fn set_chunk(&mut self, grid_pos: [u32; 3], chunk: Chunk) {
		if let Some(i) = self.slot_index(grid_pos) {
			self.chunks[i] = if chunk.is_empty() { None } else { Some(chunk) };
		}
	}

	pub fn clear_chunk(&mut self, grid_pos: [u32; 3]) {
		if let Some(i) = self.slot_index(grid_pos) {
			self.chunks[i] = None;
		}
	}

	// Apply an edit to every grid cell whose chunk AABB overlaps its bounds,
	// baking each affected chunk in place.
	pub fn apply_edit(&mut self, edit: Arc<dyn Edit>) {
		let bounds = edit.bounds();
		let [lo, hi] = self.grid_range_for_bounds(bounds);
		for gz in lo[2]..hi[2] {
			for gy in lo[1]..hi[1] {
				for gx in lo[0]..hi[0] {
					let grid_pos = [gx, gy, gz];
					let chunk_id = self.chunk_id_at(grid_pos);
					let base = self.chunk_at(grid_pos).cloned().unwrap_or_else(Chunk::new);
					let baked = edit.apply(chunk_id, base);
					self.set_chunk(grid_pos, baked);
				}
			}
		}
		self.edits.push(edit);
	}

	// The rectangular subset of grid cells whose AABBs intersect `bounds`,
	// clamped to the grid. Returns `[lo, hi)` exclusive-upper ranges.
	fn grid_range_for_bounds(&self, bounds: Aabb) -> [[u32; 3]; 2] {
		let mut lo = [0u32; 3];
		let mut hi = [0u32; 3];
		for axis in 0..3 {
			let axis_lo = bounds.min[axis] - self.origin[axis];
			let axis_hi = bounds.max[axis] - self.origin[axis];
			let cell_lo = axis_lo.div_euclid(CHUNK_SIZE).max(0);
			let cell_hi_inclusive = (axis_hi - 1).div_euclid(CHUNK_SIZE);
			let cell_hi = (cell_hi_inclusive + 1).max(0);
			lo[axis] = (cell_lo as u32).min(self.chunk_grid_dim[axis]);
			hi[axis] = (cell_hi as u32).min(self.chunk_grid_dim[axis]);
		}
		[lo, hi]
	}
}

impl Default for World {
	fn default() -> Self {
		Self::new([1, 1, 1])
	}
}
