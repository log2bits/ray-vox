use ray_vox::{
	Chunk,
	chunk::{material::Material, node::CellState},
	generate::{LazyEdit, volume::sphere::Sphere},
	world::{WorldPosition, clipmap::{ChunkHandle, Clipmap}},
};

fn blue() -> Material {
	Material::from_rgb_pbr_id([100, 150, 200], 0)
}

fn validate(chunk: &Chunk, label: &str) {
	if chunk.interior_nodes.is_empty() {
		return;
	}
	let root = chunk.interior_nodes.len() - 1;
	let mut queue: Vec<(usize, u8)> = vec![(root, 0)];
	let mut i = 0;
	while i < queue.len() {
		let (idx, depth) = queue[i];
		i += 1;
		let node = chunk.interior_nodes[idx];
		for slot in 0u8..64 {
			match node.state(slot) {
				CellState::Interior => {
					let child_idx = node.interior_child_index(slot) as usize;
					queue.push((child_idx, depth + 1));
				}
				CellState::Leaf => {
					if depth < 2 {
						eprintln!(
							"{label}: INVALID Leaf child at depth={depth} slot={slot} node_idx={idx}"
						);
					}
					let _ = node.leaf_child_index(slot);
				}
				_ => {}
			}
		}
	}
}

fn main() {
	// At depth 10 with origin [0,0,0], the chunk at slot (4,4,4) has
	// world origin [0,0,0] and voxel_size=1.
	let clipmap = Clipmap {
		occupancy: [[0u32; 16]; 11],
		origin: WorldPosition::from([0; 3]),
		pending_remap: Vec::new(),
		pending_origin: WorldPosition::from([0; 3]),
	};
	let handle = ChunkHandle::new(10, 4, 4, 4);

	let sphere = Sphere { center: [128.0; 3], radius: 32.0, material: blue() };

	let c1 = Chunk::new().apply_edits(sphere.generate(handle, &clipmap));
	validate(&c1, "c1-after-apply");
	println!("c1 done: interior={} leaf={}", c1.interior_nodes.len(), c1.leaf_nodes.len());

	let c2 = c1.clone().apply_edits(sphere.generate(handle, &clipmap));
	validate(&c2, "c2-after-apply");
	println!("c2 done: interior={} leaf={}", c2.interior_nodes.len(), c2.leaf_nodes.len());
}
