use super::{ShapeEdit, World};
use crate::{chunk::Chunk, shape::edit_packet_for_shape, tree::Aabb};

impl World {
	pub fn generate_chunk(&self, chunk_pos: [i64; 3], lod: u8) -> Chunk {
		// Side length in world voxels for this LOD chunk.
		let side = 256i64 * (4i64.pow(lod as u32));
		let chunk_aabb = Aabb {
			min: chunk_pos.map(|p| p * side),
			max: chunk_pos.map(|p| p * side + side),
		};

		let mut chunk = Chunk::new();

		for edit in &self.shape_edits {
			if edit.min_lod() > lod {
				continue;
			}
			if !edit.aabb().overlaps(&chunk_aabb) {
				continue;
			}
			if let ShapeEdit::Write { shape, .. } = edit {
				let packet = edit_packet_for_shape::<{ crate::chunk::DEPTH }>(
					shape.as_ref(),
					chunk_aabb,
				);
				if !packet.paths.is_empty() {
					chunk.add_shape_packet(packet);
				}
			}
		}

		chunk.flush_edits();
		chunk
	}
}
