use crate::Chunk;
use crate::world::World;

// A directory entry of this value means the corresponding grid cell has no
// chunk data. The shader treats it as air.
pub const EMPTY_CHUNK_SENTINEL: u32 = u32::MAX;

// A CPU-side snapshot of a World laid out ready for GPU upload. The renderer
// takes this and copies it verbatim into two storage buffers.
//
// Layout:
//   directory[flat_grid_index] -> starting u32 offset into chunk_data, or
//                                 EMPTY_CHUNK_SENTINEL.
//   chunk_data                 -> concatenated per-chunk blobs, each in the
//                                 same format Chunk::write_bytes produces.
pub struct GpuWorldSnapshot {
	pub chunk_grid_dim: [u32; 3],
	pub world_origin: [f32; 3],
	pub directory: Vec<u32>,
	pub chunk_data: Vec<u32>,
}

impl GpuWorldSnapshot {
	pub fn from_world(world: &World) -> Self {
		let directory_len = world.chunks.len().max(1);
		let mut directory = vec![EMPTY_CHUNK_SENTINEL; directory_len];
		let mut chunk_data: Vec<u32> = Vec::new();

		for (slot_index, chunk) in world.chunks.iter().enumerate() {
			let Some(chunk) = chunk.as_ref() else { continue };
			let word_offset = chunk_data.len() as u32;
			directory[slot_index] = word_offset;
			append_chunk_words(&mut chunk_data, chunk);
		}

		// Storage buffers must have at least one word; pad if the world is
		// completely empty.
		if chunk_data.is_empty() {
			chunk_data.push(0);
		}

		let world_origin = [
			world.origin.x() as f32,
			world.origin.y() as f32,
			world.origin.z() as f32,
		];

		Self {
			chunk_grid_dim: world.chunk_grid_dim,
			world_origin,
			directory,
			chunk_data,
		}
	}

	pub fn directory_byte_size(&self) -> u64 {
		(self.directory.len() * std::mem::size_of::<u32>()) as u64
	}

	pub fn chunk_data_byte_size(&self) -> u64 {
		(self.chunk_data.len() * std::mem::size_of::<u32>()) as u64
	}
}

// Serialize a single chunk into the shared chunk_data buffer. Uses
// Chunk::write_bytes so the CPU and GPU formats stay in lockstep.
fn append_chunk_words(chunk_data: &mut Vec<u32>, chunk: &Chunk) {
	let mut bytes: Vec<u8> = Vec::with_capacity(chunk.byte_size() as usize);
	chunk.write_bytes(&mut bytes).expect("chunk serialization to Vec<u8> cannot fail");

	// The on-disk chunk layout is all u32-aligned fields, so the byte count is
	// always a multiple of 4. Reinterpret directly.
	debug_assert!(bytes.len() % 4 == 0, "chunk byte size must be multiple of 4");
	let word_count = bytes.len() / 4;
	let base = chunk_data.len();
	chunk_data.resize(base + word_count, 0);
	let dest: &mut [u8] = bytemuck::cast_slice_mut(&mut chunk_data[base..]);
	dest.copy_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::chunk::material::Material;
	use crate::generate::volume::sphere::Sphere;
	use crate::util::types::WorldPos;
	use std::sync::Arc;

	#[test]
	fn empty_world_produces_sentinel_directory_and_padded_data() {
		let world = World::new([2, 2, 2]);
		let snapshot = GpuWorldSnapshot::from_world(&world);
		assert_eq!(snapshot.chunk_grid_dim, [2, 2, 2]);
		assert_eq!(snapshot.directory.len(), 8);
		assert!(snapshot.directory.iter().all(|&e| e == EMPTY_CHUNK_SENTINEL));
		assert_eq!(snapshot.chunk_data.len(), 1, "buffer padded to at least one word");
	}

	#[test]
	fn populated_chunks_get_word_offsets_into_data_buffer() {
		let mut world = World::new([2, 1, 1]);
		let stone = Material::from(0x80808040);
		world.apply_edit(Arc::new(Sphere::new(WorldPos::new(256, 128, 128), 30, stone)));

		let snapshot = GpuWorldSnapshot::from_world(&world);
		let non_sentinel_entries: Vec<u32> = snapshot
			.directory
			.iter()
			.copied()
			.filter(|&e| e != EMPTY_CHUNK_SENTINEL)
			.collect();
		assert_eq!(non_sentinel_entries.len(), 2, "sphere spans two chunks");
		assert_eq!(non_sentinel_entries[0], 0, "first chunk starts at offset 0");
		assert!(
			non_sentinel_entries[1] > 0
				&& (non_sentinel_entries[1] as usize) < snapshot.chunk_data.len(),
			"second chunk offset lands inside chunk_data",
		);
	}
}
