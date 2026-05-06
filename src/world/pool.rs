use crate::chunk::Chunk;

pub struct ChunkPool {
	chunks: Vec<Option<Chunk>>,
	dirty: Vec<bool>,
	free: Vec<u32>,
}

impl ChunkPool {
	pub fn new() -> Self {
		Self {
			chunks: vec![None], // slot 0 reserved; handles start at 1
			dirty: vec![false],
			free: Vec::new(),
		}
	}

	pub fn alloc(&mut self, chunk: Chunk) -> u32 {
		if let Some(handle) = self.free.pop() {
			self.chunks[handle as usize] = Some(chunk);
			self.dirty[handle as usize] = true;
			handle
		} else {
			let handle = self.chunks.len() as u32;
			self.chunks.push(Some(chunk));
			self.dirty.push(true);
			handle
		}
	}

	pub fn free(&mut self, handle: u32) {
		self.chunks[handle as usize] = None;
		self.dirty[handle as usize] = false;
		self.free.push(handle);
	}

	pub fn get(&self, handle: u32) -> Option<&Chunk> {
		self.chunks.get(handle as usize)?.as_ref()
	}

	pub fn get_mut(&mut self, handle: u32) -> Option<&mut Chunk> {
		self.chunks.get_mut(handle as usize)?.as_mut()
	}

	pub fn mark_dirty(&mut self, handle: u32) {
		if (handle as usize) < self.chunks.len() {
			self.dirty[handle as usize] = true;
		}
	}

	// Handles that need re-upload to GPU. Clears the dirty flags.
	pub fn take_dirty(&mut self) -> Vec<u32> {
		self.dirty
			.iter_mut()
			.enumerate()
			.filter_map(|(i, dirty)| {
				if *dirty && self.chunks[i].is_some() {
					*dirty = false;
					Some(i as u32)
				} else {
					None
				}
			})
			.collect()
	}
}
