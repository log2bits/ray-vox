pub mod compact;
pub mod edit;
pub mod material;
pub mod merge;
pub mod node;
pub mod rebuild;
pub mod stats;

#[cfg(test)]
mod tests;

use crate::util::PalettedVec;
use crate::util::types::ChunkPos;
use bytemuck;
use edit::EditPacket;
use edit::Path;
use material::Material;
use node::{CellState, InteriorNode, InteriorNodeWide, LeafNode};

/// What sits at one slot of an interior node. Returned by Chunk::child.
pub enum Child {
	Empty,
	Filled(Material),
	Interior(u32),
	Leaf(u32),
}

/// A chunk in its compressed, GPU-ready form.
pub struct Chunk {
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec<Material>,
	pub interior_nodes: Vec<InteriorNode>,
}

impl Chunk {
	pub fn new() -> Self {
		Self {
			leaf_nodes: Vec::new(),
			materials: PalettedVec::new(),
			interior_nodes: Vec::new(),
		}
	}

	pub fn is_uniform(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.len() == 1
	}

	pub fn is_empty(&self) -> bool {
		self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() && self.materials.is_empty()
	}

	/// Whole-chunk representative material, stored at materials[0]. When both
	/// node arrays are empty, this is the chunk's uniform fill.
	pub fn chunk_lod(&self) -> Material {
		if self.materials.is_empty() {
			Material::air()
		} else {
			self.materials.get(0)
		}
	}

	/// Index of the root interior. Only valid when interior_nodes is non-empty.
	pub fn root_idx(&self) -> u32 {
		(self.interior_nodes.len() - 1) as u32
	}

	/// Resolve a slot of an interior into its child kind plus index.
	pub fn child(&self, node_idx: u32, slot: u8) -> Child {
		let n = &self.interior_nodes[node_idx as usize];
		match n.masks.state(slot) {
			CellState::Empty => Child::Empty,
			CellState::Filled => Child::Filled(self.materials.get(n.material_index(slot))),
			CellState::Interior => Child::Interior(n.interior_child_index(slot)),
			CellState::Leaf => Child::Leaf(n.leaf_child_index(slot)),
		}
	}

	/// Material at a voxel position. Air for empty space.
	pub fn voxel_at(&self, pos: ChunkPos) -> Material {
		if self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() {
			return self.chunk_lod();
		}
		let path = Path::from_coords(pos, 4);
		if self.interior_nodes.is_empty() {
			// Root-leaf: a single leaf at the top, each cell covering 64^3 voxels.
			let leaf = &self.leaf_nodes[0];
			let slot = path.slot_at(0);
			return if leaf.occupancy.contains(slot) {
				self.materials.get(leaf.material_index(slot))
			} else {
				Material::air()
			};
		}
		let mut idx = self.root_idx();
		for d in 0..3u8 {
			let slot = path.slot_at(d);
			match self.child(idx, slot) {
				Child::Empty => return Material::air(),
				Child::Filled(m) => return m,
				Child::Interior(child) => idx = child,
				Child::Leaf(leaf_idx) => {
					// Compact can demote interiors into leaves at any depth, so use
					// the path byte one level deeper than the current interior.
					let leaf = &self.leaf_nodes[leaf_idx as usize];
					let lslot = path.slot_at(d + 1);
					return if leaf.occupancy.contains(lslot) {
						self.materials.get(leaf.material_index(lslot))
					} else {
						Material::air()
					};
				}
			}
		}
		Material::air()
	}

	pub fn gpu_size_bytes(&self) -> u32 {
		let header = 3 * size_of::<u32>();
		let interior = self.interior_nodes.len() * size_of::<InteriorNode>();
		let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
		let lut = self.materials.lut.len() as usize * size_of::<Material>();
		let indices = self.materials.indices.words.len() * size_of::<u32>();
		(header + interior + leaf + lut + indices) as u32
	}

	pub fn gpu_bytes(&self) -> &[u8] {
		bytemuck::cast_slice(&self.interior_nodes)
	}

	/// Switch into the editable form. Use queue_edit + bake to apply edits.
	pub fn into_mutable(self) -> MutableChunk {
		let wide: Vec<InteriorNodeWide> = self
			.interior_nodes
			.iter()
			.map(|n| {
				let mut w = InteriorNodeWide::default();
				w.masks = n.masks;
				w.set_interior_offset(n.interior_offset());
				w.set_leaf_offset(n.leaf_offset());
				w.set_material_offset(n.material_offset());
				w
			})
			.collect();
		MutableChunk {
			leaf_nodes: self.leaf_nodes,
			materials: self.materials,
			interior_nodes: wide,
			pending: Vec::new(),
		}
	}
}

impl Default for Chunk {
	fn default() -> Self {
		Self::new()
	}
}

impl Clone for Chunk {
	fn clone(&self) -> Self {
		Self {
			leaf_nodes: self.leaf_nodes.clone(),
			materials: self.materials.clone(),
			interior_nodes: self.interior_nodes.clone(),
		}
	}
}

pub struct MutableChunk {
	pub leaf_nodes: Vec<LeafNode>,
	pub materials: PalettedVec<Material>,
	pub interior_nodes: Vec<InteriorNodeWide>,
	pub pending: Vec<EditPacket>,
}

impl MutableChunk {
	/// Empty builder. No nodes, no materials, no pending edits.
	pub fn empty() -> Self {
		Self {
			leaf_nodes: Vec::new(),
			materials: PalettedVec::new(),
			interior_nodes: Vec::new(),
			pending: Vec::new(),
		}
	}

	/// Queue an edit packet for the next bake. Packets are applied in queue order.
	pub fn queue_edit(&mut self, packet: EditPacket) {
		self.pending.push(packet);
	}

	/// Apply all queued packets and compress. Returns a frozen Chunk.
	pub fn bake(mut self) -> Chunk {
		let pending = std::mem::take(&mut self.pending);
		for mut packet in pending {
			packet.sort();
			self.apply_batch(&packet.edits);
		}
		compact::compress(self)
	}

	pub fn apply_batch(&mut self, batch: &[(Path, Material)]) {
		let root_edit_count = batch.partition_point(|(p, _): &(Path, Material)| p.is_root());
		let sub_batch = &batch[root_edit_count..];

		if root_edit_count > 0 {
			let (_, fill_mat) = batch[root_edit_count - 1];
			self.interior_nodes.clear();
			self.leaf_nodes.clear();
			self.materials.clear();
			if !fill_mat.is_air() {
				self.materials.push(fill_mat);
			}
			if sub_batch.is_empty() {
				return;
			}
		}

		if sub_batch.is_empty() {
			return;
		}

		let base = if let Some(root) = self.interior_nodes.last().copied() {
			rebuild::InteriorBase::Existing(root)
		} else if self.materials.len() == 1 {
			rebuild::InteriorBase::Fill(self.materials.get(0))
		} else {
			rebuild::InteriorBase::Empty
		};

		match self.rebuild_interior(base, 0, sub_batch) {
			rebuild::RebuildResult::Empty => {}
			rebuild::RebuildResult::Filled(mat) => {
				self.interior_nodes.clear();
				self.leaf_nodes.clear();
				self.materials.clear();
				self.materials.push(mat);
			}
			rebuild::RebuildResult::Interior(_, new_root) => {
				self.interior_nodes.push(new_root);
			}
			rebuild::RebuildResult::Leaf(..) => unreachable!(),
		}
	}

}

impl Clone for MutableChunk {
	fn clone(&self) -> Self {
		Self {
			leaf_nodes: self.leaf_nodes.clone(),
			materials: self.materials.clone(),
			interior_nodes: self.interior_nodes.clone(),
			pending: self.pending.clone(),
		}
	}
}
