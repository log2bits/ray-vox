pub mod build;
pub mod edit;
pub mod material;
pub mod node;
pub mod sources;
pub mod stats;

#[cfg(test)]
mod tests;

use crate::util::PalettedVec;
use crate::util::types::ChunkPos;
use bytemuck;
use build::Source;
use edit::Path;
use material::Material;
use node::{CellState, InteriorNode, LeafNode};
use sources::{ChunkSource, Overlay};

pub enum Child {
	Empty,
	Filled(Material),
	Interior(u32),
	Leaf(u32),
}

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

	pub fn chunk_lod(&self) -> Material {
		if self.materials.is_empty() {
			Material::air()
		} else {
			self.materials.get(0)
		}
	}

	pub fn root_idx(&self) -> u32 {
		(self.interior_nodes.len() - 1) as u32
	}

	pub fn child(&self, node_idx: u32, slot: u8) -> Child {
		let n = &self.interior_nodes[node_idx as usize];
		match n.masks.state(slot) {
			CellState::Empty => Child::Empty,
			CellState::Filled => Child::Filled(self.materials.get(n.material_index(slot))),
			CellState::Interior => Child::Interior(n.interior_child_index(slot)),
			CellState::Leaf => Child::Leaf(n.leaf_child_index(slot)),
		}
	}

	pub fn voxel_at(&self, pos: ChunkPos) -> Material {
		if self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() {
			return self.chunk_lod();
		}
		let path = Path::from_coords(pos, 4);
		if self.interior_nodes.is_empty() {
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

	pub const HEADER_BYTES: u32 = 6 * 4;

	pub fn byte_size(&self) -> u32 {
		let header = Chunk::HEADER_BYTES as usize;
		let interior = self.interior_nodes.len() * size_of::<InteriorNode>();
		let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
		let lut = self.materials.lut.values.len() * size_of::<Material>();
		let indices = self.materials.indices.words.len() * size_of::<u32>();
		(header + interior + leaf + lut + indices) as u32
	}

	fn header_words(&self) -> [u32; 6] {
		[
			self.interior_nodes.len() as u32,
			self.leaf_nodes.len() as u32,
			self.materials.lut.values.len() as u32,
			self.materials.indices.len,
			self.materials.indices.bits,
			self.materials.indices.words.len() as u32,
		]
	}

	pub fn write_bytes<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
		w.write_all(bytemuck::cast_slice(&self.header_words()))?;
		w.write_all(bytemuck::cast_slice(&self.interior_nodes))?;
		w.write_all(bytemuck::cast_slice(&self.leaf_nodes))?;
		w.write_all(bytemuck::cast_slice(&self.materials.lut.values))?;
		w.write_all(bytemuck::cast_slice(&self.materials.indices.words))?;
		Ok(())
	}

	pub fn read_bytes<R: std::io::Read>(r: &mut R) -> std::io::Result<Chunk> {
		let mut header = [0u32; 6];
		r.read_exact(bytemuck::cast_slice_mut(&mut header))?;
		let [interior_count, leaf_count, lut_count, indices_len, indices_bits, indices_words] = header;

		let mut interior_nodes = vec![InteriorNode::default(); interior_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut interior_nodes))?;

		let mut leaf_nodes = vec![LeafNode::default(); leaf_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut leaf_nodes))?;

		let mut lut_values = vec![Material::default(); lut_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut lut_values))?;

		let mut indices_words_vec = vec![0u32; indices_words as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut indices_words_vec))?;

		let mut materials = PalettedVec::new();
		materials.lut.values = lut_values;
		materials.indices = crate::util::PackedVec {
			words: indices_words_vec,
			bits: indices_bits,
			len: indices_len,
		};
		Ok(Chunk { leaf_nodes, materials, interior_nodes })
	}

	pub fn edit<S: Source>(self, source: &S) -> Chunk {
		let base = ChunkSource::new(&self);
		let overlay = Overlay::new(base, source.clone());
		build::build_chunk(&overlay)
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
