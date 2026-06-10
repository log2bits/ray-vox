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
use build::{Source, VoxelSample};
use edit::Path;
use material::Material;
use node::{CellState, InteriorNode, LeafNode};
use sources::{ChunkSource, Overlay};

#[derive(Copy, Clone)]
pub enum Child {
	Empty,
	Filled(Material),
	Interior(u32),
	Leaf(u32),
}

impl Child {
	pub fn state(self) -> CellState {
		match self {
			Child::Empty => CellState::Empty,
			Child::Filled(_) => CellState::Filled,
			Child::Interior(_) => CellState::Interior,
			Child::Leaf(_) => CellState::Leaf,
		}
	}
}

#[derive(Clone, Default)]
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

	/// The `Child` representing this chunk's root cell.
	///
	/// Collapses the four storage shapes (empty, uniform, single leaf at root,
	/// or interior tree) into one state for downstream traversal.
	pub fn root_child(&self) -> Child {
		if self.is_empty() {
			Child::Empty
		} else if self.is_uniform() {
			Child::Filled(self.materials.get(0))
		} else if self.interior_nodes.is_empty() {
			Child::Leaf(0)
		} else {
			Child::Interior(self.root_idx())
		}
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

	#[inline]
	fn leaf_voxel(&self, leaf: &LeafNode, slot: u8) -> Material {
		if leaf.occupancy.contains(slot) {
			self.materials.get(leaf.material_index(slot))
		} else {
			Material::air()
		}
	}

	/// Descend one tree level from `state` through `slot`. Empty/Filled propagate
	/// unchanged; Interior reads the child; Leaf reads the leaf-slot's voxel and
	/// collapses to Empty (air) or Filled.
	pub fn descend_child(&self, state: Child, slot: u8) -> Child {
		match state {
			Child::Empty => Child::Empty,
			Child::Filled(m) => Child::Filled(m),
			Child::Interior(idx) => self.child(idx, slot),
			Child::Leaf(leaf_idx) => {
				let leaf = &self.leaf_nodes[leaf_idx as usize];
				if leaf.occupancy.contains(slot) {
					Child::Filled(self.materials.get(leaf.material_index(slot)))
				} else {
					Child::Empty
				}
			}
		}
	}

	/// Voxel sample for the cell at `slot` inside a depth-3 leaf.
	pub fn leaf_voxel_sample(&self, leaf_idx: u32, slot: u8) -> VoxelSample {
		match self.leaf_voxel(&self.leaf_nodes[leaf_idx as usize], slot) {
			m if m.is_air() => VoxelSample::Passthrough,
			m => VoxelSample::Fill(m),
		}
	}

	pub fn voxel_at(&self, pos: ChunkPos) -> Material {
		if self.interior_nodes.is_empty() && self.leaf_nodes.is_empty() {
			return self.chunk_lod();
		}
		let path = Path::from_coords(pos, 4);
		if self.interior_nodes.is_empty() {
			return self.leaf_voxel(&self.leaf_nodes[0], path.slot_at(0));
		}
		let mut idx = self.root_idx();
		for d in 0..3u8 {
			let slot = path.slot_at(d);
			match self.child(idx, slot) {
				Child::Empty => return Material::air(),
				Child::Filled(m) => return m,
				Child::Interior(child) => idx = child,
				Child::Leaf(leaf_idx) => {
					return self.leaf_voxel(&self.leaf_nodes[leaf_idx as usize], path.slot_at(d + 1));
				}
			}
		}
		Material::air()
	}

	pub fn byte_size(&self) -> u32 {
		8 + (self.interior_nodes.len() * size_of::<InteriorNode>()
			+ self.leaf_nodes.len() * size_of::<LeafNode>()) as u32
			+ self.materials.byte_size()
	}

	pub fn write_bytes<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
		let header = [self.interior_nodes.len() as u32, self.leaf_nodes.len() as u32];
		w.write_all(bytemuck::cast_slice(&header))?;
		w.write_all(bytemuck::cast_slice(&self.interior_nodes))?;
		w.write_all(bytemuck::cast_slice(&self.leaf_nodes))?;
		self.materials.write_bytes(w)?;
		Ok(())
	}

	pub fn read_bytes<R: std::io::Read>(r: &mut R) -> std::io::Result<Chunk> {
		let mut header = [0u32; 2];
		r.read_exact(bytemuck::cast_slice_mut(&mut header))?;
		let [interior_count, leaf_count] = header;

		let mut interior_nodes = vec![InteriorNode::default(); interior_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut interior_nodes))?;

		let mut leaf_nodes = vec![LeafNode::default(); leaf_count as usize];
		r.read_exact(bytemuck::cast_slice_mut(&mut leaf_nodes))?;

		let materials = PalettedVec::read_bytes(r)?;
		Ok(Chunk { leaf_nodes, materials, interior_nodes })
	}

	pub fn edit<S: Source>(self, source: &S) -> Chunk {
		let base = ChunkSource::new(&self);
		let overlay = Overlay::new(base, source.clone());
		build::build_chunk(&overlay)
	}
}
