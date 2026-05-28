use crate::chunk::Chunk;
use crate::util::types::ClipmapChunkId;
use std::collections::{HashMap, HashSet};

pub struct ChunkPool {
	// CPU side
	pub chunks: HashMap<ClipmapChunkId, Chunk>,

	// GPU allocation tracking
	pub allocations: HashMap<ClipmapChunkId, Allocation>,
	pub free_list: Vec<Allocation>,
	pub dirty: HashSet<ClipmapChunkId>,
	// TODO: gpu_buffer: wgpu::Buffer,
	// TODO: staging_belt: wgpu::util::StagingBelt,
}

#[derive(Clone, Copy, Debug)]
pub struct Allocation {
	pub offset: u32, // bytes
	pub size: u32,   // bytes
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

	pub fn insert(&mut self, handle: ClipmapChunkId, chunk: Chunk) {
		let alloc = self.alloc(chunk.gpu_size_bytes());
		self.allocations.insert(handle, alloc);
		self.chunks.insert(handle, chunk);
		self.dirty.insert(handle);
	}

	pub fn remove(&mut self, handle: ClipmapChunkId) -> Option<Chunk> {
		if let Some(alloc) = self.allocations.remove(&handle) {
			self.free_list.push(alloc);
			self.coalesce();
		}
		self.dirty.remove(&handle);
		self.chunks.remove(&handle)
	}

	pub fn get(&self, handle: ClipmapChunkId) -> Option<&Chunk> {
		self.chunks.get(&handle)
	}

	/// Mutable access marks the chunk dirty for re-upload.
	pub fn get_mut(&mut self, handle: ClipmapChunkId) -> Option<&mut Chunk> {
		self.dirty.insert(handle);
		self.chunks.get_mut(&handle)
	}

	pub fn contains(&self, handle: ClipmapChunkId) -> bool {
		self.chunks.contains_key(&handle)
	}

	/// The current high-water mark of GPU memory usage in bytes.
	/// The GPU buffer must be at least this large.
	pub fn high_water_mark(&self) -> u32 {
		self.allocations
			.values()
			.map(|a| a.offset + a.size)
			.max()
			.unwrap_or(0)
	}

	/// Upload all dirty chunks to the GPU and clear the dirty set.
	/// TODO: takes queue: &wgpu::Queue, staging_belt: &mut wgpu::util::StagingBelt
	pub fn flush(&mut self) {
		for handle in self.dirty.drain() {
			let Some(chunk) = self.chunks.get(&handle) else {
				continue;
			};
			let Some(alloc) = self.allocations.get(&handle) else {
				continue;
			};
			// TODO: let mut view = staging_belt.write_buffer(
			//     encoder,
			//     &self.gpu_buffer,
			//     alloc.offset as u64,
			//     alloc.size,
			//     device,
			// );
			// TODO: chunk.write_to(&mut view);
			let _ = (chunk, alloc); // remove when wgpu calls are added
		}
	}

	/// First-fit allocation. Grows past the current high-water mark if no
	/// free slot is large enough.
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
			Allocation {
				offset: slot.offset,
				size,
			}
		} else {
			Allocation {
				offset: self.high_water_mark(),
				size,
			}
		}
	}

	/// Sort free list by offset and merge adjacent slots.
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
