use crate::chunk::Chunk;
use crate::util::types::ChunkHandle;
use crate::world::clipmap::RemapOp;
use std::collections::{HashMap, HashSet};

pub struct ChunkPool {
	pub chunks: HashMap<ChunkHandle, Chunk>,
	pub allocations: HashMap<ChunkHandle, Allocation>,
	pub free_list: Vec<Allocation>,
	pub dirty: HashSet<ChunkHandle>,
	// TODO: gpu_buffer: wgpu::Buffer,
	// TODO: staging_belt: wgpu::util::StagingBelt,
}

#[derive(Clone, Copy, Debug)]
pub struct Allocation {
	pub offset: u32,
	pub size: u32,
}

impl ChunkPool {
	pub fn new() -> Self {
		Self {
			chunks: HashMap::new(),
			allocations: HashMap::new(),
			free_list: Vec::new(),
			dirty: HashSet::new(),
		}
	}

	pub fn insert(&mut self, handle: ChunkHandle, chunk: Chunk) {
		if let Some(prev) = self.allocations.get(&handle).copied() {
			if prev.size >= chunk.byte_size() {
				self.chunks.insert(handle, chunk);
				self.dirty.insert(handle);
				return;
			}
			self.free_list.push(prev);
			self.allocations.remove(&handle);
		}
		let alloc = self.alloc(chunk.byte_size());
		self.allocations.insert(handle, alloc);
		self.chunks.insert(handle, chunk);
		self.dirty.insert(handle);
	}

	pub fn remove(&mut self, handle: ChunkHandle) -> Option<Chunk> {
		if let Some(alloc) = self.allocations.remove(&handle) {
			self.free_list.push(alloc);
			self.coalesce();
		}
		self.dirty.remove(&handle);
		self.chunks.remove(&handle)
	}

	pub fn get(&self, handle: ChunkHandle) -> Option<&Chunk> {
		self.chunks.get(&handle)
	}

	pub fn get_mut(&mut self, handle: ChunkHandle) -> Option<&mut Chunk> {
		self.dirty.insert(handle);
		self.chunks.get_mut(&handle)
	}

	pub fn contains(&self, handle: ChunkHandle) -> bool {
		self.chunks.contains_key(&handle)
	}

	pub fn high_water_mark(&self) -> u32 {
		self.allocations
			.values()
			.map(|a| a.offset + a.size)
			.max()
			.unwrap_or(0)
	}

	pub fn apply_remap(&mut self, op: &RemapOp) {
		match op {
			RemapOp::Add(_, _) => {}
			RemapOp::Delete(h) => {
				self.remove(*h);
			}
		}
	}

	// TODO: takes queue: &wgpu::Queue, staging_belt: &mut wgpu::util::StagingBelt
	pub fn flush(&mut self) {
		for handle in self.dirty.drain() {
			let Some(chunk) = self.chunks.get(&handle) else {
				continue;
			};
			let Some(alloc) = self.allocations.get(&handle) else {
				continue;
			};
			let _ = (chunk, alloc);
		}
	}

	fn alloc(&mut self, size: u32) -> Allocation {
		if let Some(idx) = self.free_list.iter().position(|s| s.size >= size) {
			let slot = self.free_list[idx];
			if slot.size > size {
				self.free_list[idx] = Allocation {
					offset: slot.offset + size,
					size: slot.size - size,
				};
			} else {
				self.free_list.swap_remove(idx);
			}
			Allocation { offset: slot.offset, size }
		} else {
			Allocation { offset: self.high_water_mark(), size }
		}
	}

	fn coalesce(&mut self) {
		self.free_list.sort_unstable_by_key(|s| s.offset);
		let mut i = 0;
		while i + 1 < self.free_list.len() {
			let end = self.free_list[i].offset + self.free_list[i].size;
			if end == self.free_list[i + 1].offset {
				self.free_list[i].size += self.free_list[i + 1].size;
				self.free_list.remove(i + 1);
			} else {
				i += 1;
			}
		}
	}
}

impl Default for ChunkPool {
	fn default() -> Self {
		Self::new()
	}
}
