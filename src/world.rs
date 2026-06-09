pub mod chunk_pool;
pub mod clipmap;
pub mod pbr;

#[cfg(test)]
mod tests;

use crate::Chunk;
use crate::chunk::build::{build_chunk, CHUNK_SIDE};
use crate::chunk::sources::{CompositeSource, LocalEdit};
use crate::generate::Edit;
use crate::util::Lut;
use crate::util::types::{ChunkHandle, ChunkId, LodLevel, WorldPos};
use chunk_pool::ChunkPool;
use clipmap::{Clipmap, RemapOp};
pub use pbr::Pbr;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn dedup_pending_remap(pending: &mut Vec<RemapOp>) {
	if pending.len() < 2 {
		return;
	}
	let mut last: HashMap<ChunkHandle, u32> = HashMap::with_capacity(pending.len());
	for (i, op) in pending.iter().enumerate() {
		last.insert(op.handle(), i as u32);
	}
	let mut i = 0u32;
	pending.retain(|op| {
		let keep = last.get(&op.handle()) == Some(&i);
		i += 1;
		keep
	});
}

pub fn bake_pure(chunk_id: ChunkId, edits: &[Arc<dyn Edit>]) -> Chunk {
	let locals: Vec<Box<dyn LocalEdit + '_>> = edits.iter()
		.filter_map(|e| e.make_local(chunk_id))
		.collect();
	if locals.is_empty() {
		return Chunk::new();
	}
	let composite = CompositeSource::new(&locals, CHUNK_SIDE);
	build_chunk(&composite)
}

struct BakeJob {
	handle: ChunkHandle,
	chunk_id: ChunkId,
	edit_ids: Vec<u32>,
	edit_refs: Vec<Arc<dyn Edit>>,
}

pub struct World {
	pub edits: Vec<Arc<dyn Edit>>,
	pub by_handle: HashMap<ChunkHandle, Vec<u32>>,
	pub needs_rebake: HashSet<ChunkHandle>,
	pub chunk_pool: ChunkPool,
	pub clipmap: Clipmap,
	pub pbr_lut: Lut<Pbr>,
}

impl World {
	pub fn new() -> Self {
		Self {
			edits: Vec::new(),
			by_handle: HashMap::new(),
			needs_rebake: HashSet::new(),
			chunk_pool: ChunkPool::new(),
			clipmap: Clipmap::new(),
			pbr_lut: Lut::new(),
		}
	}

	pub fn apply_edit(&mut self, edit: Arc<dyn Edit>) -> u32 {
		let bounds = edit.bounds();
		let id = self.edits.len() as u32;
		let camera_pos = self.clipmap.camera_pos;
		self.edits.push(edit);

		for level in 0..LodLevel::LEVELS {
			let lod = LodLevel::new(level);
			let window_origin = lod.level_origin(camera_pos);
			let chunk_size = lod.chunk_size();
			for x in 0..LodLevel::GRID_SIZE {
				for y in 0..LodLevel::GRID_SIZE {
					for z in 0..LodLevel::GRID_SIZE {
						let chunk_origin = WorldPos::new(
							window_origin.x() + (x as i32) * chunk_size,
							window_origin.y() + (y as i32) * chunk_size,
							window_origin.z() + (z as i32) * chunk_size,
						);
						let chunk_id = ChunkId::new(chunk_origin, lod);
						if !chunk_id.aabb().intersects(&bounds) {
							continue;
						}
						let handle = chunk_id.handle();
						match self.clipmap.chunk_id_of(handle) {
							Some(resident) if resident == chunk_id => {
								self.by_handle.entry(handle).or_default().push(id);
								self.needs_rebake.insert(handle);
							}
							_ => {
								self.clipmap.pending_remap.push(RemapOp::Add(handle, chunk_id));
							}
						}
					}
				}
			}
		}

		id
	}

	pub fn bake(&self, handle: ChunkHandle) -> Chunk {
		let chunk_id = match self.clipmap.chunk_id_of(handle) {
			Some(id) => id,
			None => return Chunk::new(),
		};
		let edits: Vec<Arc<dyn Edit>> = self.by_handle.get(&handle)
			.map(|ids| ids.iter().map(|&i| self.edits[i as usize].clone()).collect())
			.unwrap_or_default();
		bake_pure(chunk_id, &edits)
	}

	pub fn drive_remaps(&mut self, budget: usize) {
		dedup_pending_remap(&mut self.clipmap.pending_remap);
		let add_jobs = self.collect_add_jobs(budget);
		let remaining = budget - add_jobs.len();
		let rebake_jobs = self.collect_rebake_jobs(remaining);

		let jobs: Vec<BakeJob> = add_jobs.into_iter().chain(rebake_jobs).collect();
		let results: Vec<(BakeJob, Chunk)> = jobs.into_par_iter()
			.map(|job| {
				let chunk = bake_pure(job.chunk_id, &job.edit_refs);
				(job, chunk)
			})
			.collect();

		for (job, chunk) in results {
			self.needs_rebake.remove(&job.handle);
			let already_resident = self.clipmap.chunk_id_of(job.handle) == Some(job.chunk_id);
			if chunk.is_empty() {
				if already_resident {
					self.clipmap.evict(job.handle);
					self.chunk_pool.remove(job.handle);
					self.by_handle.remove(&job.handle);
				}
				continue;
			}
			if !already_resident && self.clipmap.chunk_id_of(job.handle).is_some() {
				self.clipmap.evict(job.handle);
				self.chunk_pool.remove(job.handle);
			}
			self.clipmap.assign(job.handle, job.chunk_id);
			self.by_handle.insert(job.handle, job.edit_ids);
			self.chunk_pool.insert(job.handle, chunk);
		}
	}

	fn collect_add_jobs(&mut self, budget: usize) -> Vec<BakeJob> {
		let mut jobs: Vec<BakeJob> = Vec::with_capacity(budget.min(64));
		while jobs.len() < budget {
			let op = match self.clipmap.pending_remap.pop() {
				Some(op) => op,
				None => break,
			};
			match op {
				RemapOp::Add(handle, chunk_id) => {
					if self.clipmap.chunk_id_of(handle) == Some(chunk_id) {
						continue;
					}
					let aabb = chunk_id.aabb();
					let edit_ids: Vec<u32> = self.edits.iter().enumerate()
						.filter_map(|(i, e)| {
							e.bounds().intersects(&aabb).then_some(i as u32)
						})
						.collect();
					if edit_ids.is_empty() {
						continue;
					}
					let edit_refs: Vec<Arc<dyn Edit>> = edit_ids.iter()
						.map(|&i| self.edits[i as usize].clone())
						.collect();
					jobs.push(BakeJob { handle, chunk_id, edit_ids, edit_refs });
				}
				RemapOp::Delete(handle) => {
					self.clipmap.evict(handle);
					self.chunk_pool.remove(handle);
					self.by_handle.remove(&handle);
					self.needs_rebake.remove(&handle);
				}
			}
		}
		jobs
	}

	fn collect_rebake_jobs(&self, budget: usize) -> Vec<BakeJob> {
		self.needs_rebake.iter().take(budget).filter_map(|&handle| {
			let chunk_id = self.clipmap.chunk_id_of(handle)?;
			let edit_ids: Vec<u32> = self.by_handle.get(&handle).cloned().unwrap_or_default();
			let edit_refs: Vec<Arc<dyn Edit>> = edit_ids.iter()
				.map(|&i| self.edits[i as usize].clone())
				.collect();
			Some(BakeJob { handle, chunk_id, edit_ids, edit_refs })
		}).collect()
	}

	pub fn process_remap(&mut self, op: &RemapOp) {
		self.clipmap.apply_remap(op);
		self.chunk_pool.apply_remap(op);
		self.apply_remap_to_edits(op);
	}

	fn apply_remap_to_edits(&mut self, op: &RemapOp) {
		match op {
			RemapOp::Add(handle, chunk_id) => {
				let aabb = chunk_id.aabb();
				let mut hits: Vec<u32> = Vec::new();
				for (i, e) in self.edits.iter().enumerate() {
					if e.bounds().intersects(&aabb) {
						hits.push(i as u32);
					}
				}
				if hits.is_empty() {
					self.by_handle.remove(handle);
				} else {
					self.by_handle.insert(*handle, hits);
				}
			}
			RemapOp::Delete(handle) => {
				self.by_handle.remove(handle);
			}
		}
	}
}
