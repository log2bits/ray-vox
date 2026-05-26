pub mod compact;
pub mod edit;
pub mod material;
pub mod node;
pub mod rebuild;

use crate::util::PalettedVec;
use bytemuck;
use edit::Edits;
use edit::Path;
use material::Material;
use node::{InteriorNode, InteriorNodeWide, LeafNode};

pub struct Compressed {
    pub interior_nodes: Vec<InteriorNode>,
}

pub struct Editing {
    pub interior_nodes: Vec<InteriorNodeWide>,
    pub edits: Edits,
}

pub struct Chunk<S> {
    pub leaf_nodes: Vec<LeafNode>,
    pub materials: PalettedVec<Material>,
    pub state: S,
}

pub trait ChunkInteriorLen {
    fn interior_len(&self) -> usize;
}

impl ChunkInteriorLen for Compressed {
    fn interior_len(&self) -> usize {
        self.interior_nodes.len()
    }
}

impl ChunkInteriorLen for Editing {
    fn interior_len(&self) -> usize {
        self.interior_nodes.len()
    }
}

impl<S: ChunkInteriorLen> Chunk<S> {
    pub fn is_root_leaf(&self) -> bool {
        self.state.interior_len() == 0 && self.leaf_nodes.len() == 1
    }

    pub fn is_uniform(&self) -> bool {
        self.state.interior_len() == 0 && self.leaf_nodes.is_empty() && self.materials.len() == 1
    }

    pub fn is_empty(&self) -> bool {
        self.state.interior_len() == 0
            && self.leaf_nodes.is_empty()
            && self.materials.is_empty()
    }
}

impl Chunk<Compressed> {
    pub fn new() -> Self {
        Self {
            leaf_nodes: Vec::new(),
            materials: PalettedVec::new(),
            state: Compressed {
                interior_nodes: Vec::new(),
            },
        }
    }

    pub fn gpu_size_bytes(&self) -> u32 {
        let header = 3 * size_of::<u32>(); // interior_count, leaf_count, material_count
        let interior = self.state.interior_nodes.len() * size_of::<InteriorNode>();
        let leaf = self.leaf_nodes.len() * size_of::<LeafNode>();
        let lut = self.materials.lut.len() as usize * size_of::<u32>();
        let indices = self.materials.indices.words.len() * size_of::<u32>();
        (header + interior + leaf + lut + indices) as u32
    }

    pub fn gpu_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.state.interior_nodes)
    }

    pub fn decompress(self) -> Chunk<Editing> {
        let wide: Vec<InteriorNodeWide> = self
            .state
            .interior_nodes
            .iter()
            .map(|n| {
                let mut w = InteriorNodeWide::default();
                w.set_has_child(n.has_child());
                w.set_is_leaf(n.is_leaf());
                w.set_interior_offset(n.interior_offset());
                w.set_leaf_offset(n.leaf_offset());
                w.set_material_offset(n.material_offset());
                w
            })
            .collect();
        Chunk {
            leaf_nodes: self.leaf_nodes,
            materials: self.materials,
            state: Editing {
                interior_nodes: wide,
                edits: Edits::new(),
            },
        }
    }
}

impl Default for Chunk<Compressed> {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Chunk<Compressed> {
    fn clone(&self) -> Self {
        Self {
            leaf_nodes: self.leaf_nodes.clone(),
            materials: self.materials.clone(),
            state: Compressed {
                interior_nodes: self.state.interior_nodes.clone(),
            },
        }
    }
}

impl Chunk<Editing> {
    pub fn new() -> Self {
        Self {
            leaf_nodes: Vec::new(),
            materials: PalettedVec::new(),
            state: Editing {
                interior_nodes: Vec::new(),
                edits: Edits::new(),
            },
        }
    }

    pub fn push_edit(&mut self, path: Path, material: Material) {
        self.state.edits.push(path, material);
    }

    pub fn apply_edits(&mut self) {
        let mut edits = std::mem::take(&mut self.state.edits);
        edits.sort();
        for batch in &edits.batches {
            let start = batch.range().start as usize;
            let end = batch.range().end as usize;
            let slice = &edits.edits[start..end];
            self.apply_batch(slice);
        }
    }

    pub fn apply_batch(&mut self, batch: &[(Path, Material)]) {
        let root_edit_count = batch.partition_point(|(p, _): &(Path, Material)| p.is_root());
        let sub_batch = &batch[root_edit_count..];

        if root_edit_count > 0 {
            let (_, fill_mat) = batch[root_edit_count - 1];
            self.state.interior_nodes.clear();
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

        let old_root = self.state.interior_nodes.last().copied();
        let expand_fill = if old_root.is_none() && self.materials.len() == 1 {
            Some(self.materials.get(0))
        } else {
            None
        };

        match self.rebuild_interior(old_root, expand_fill, 0, sub_batch) {
            rebuild::RebuildResult::Empty => {}
            rebuild::RebuildResult::Filled(mat) => {
                self.state.interior_nodes.clear();
                self.leaf_nodes.clear();
                self.materials.clear();
                self.materials.push(mat);
            }
            rebuild::RebuildResult::Interior(new_root) => {
                self.state.interior_nodes.push(new_root);
            }
            rebuild::RebuildResult::Leaf(_) => unreachable!(),
        }
    }

    pub fn compress(self) -> Chunk<Compressed> {
        compact::compress(self)
    }
}

impl Default for Chunk<Editing> {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Chunk<Editing> {
    fn clone(&self) -> Self {
        Self {
            leaf_nodes: self.leaf_nodes.clone(),
            materials: self.materials.clone(),
            state: Editing {
                interior_nodes: self.state.interior_nodes.clone(),
                edits: self.state.edits.clone(),
            },
        }
    }
}

// Convenience: create a fresh Chunk<Editing> directly without going through decompress.
pub fn new_editing() -> Chunk<Editing> {
    Chunk::<Editing>::new()
}
