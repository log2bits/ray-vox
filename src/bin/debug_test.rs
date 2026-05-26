use ray_vox::{
	chunk::{Chunk, material::Material, node::CellState},
	volumes::{self, Sphere},
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
	use ray_vox::chunk::edit::Edits;

	// Build c1: sphere stamp
	let mut edits1 = Edits::new();
	volumes::stamp(
		&Sphere {
			center: [128.0; 3],
			radius: 32.0,
			material: blue(),
		},
		&mut edits1,
	);
	let c1 = Chunk::new().apply_edits(edits1);
	validate(&c1, "c1-after-apply_edits");
	println!(
		"c1 done: interior={} leaf={}",
		c1.interior_nodes.len(),
		c1.leaf_nodes.len()
	);

	// Build c2: same stamp again on top of c1
	let mut edits2 = Edits::new();
	volumes::stamp(
		&Sphere {
			center: [128.0; 3],
			radius: 32.0,
			material: blue(),
		},
		&mut edits2,
	);
	let c2 = c1.clone().apply_edits(edits2);
	validate(&c2, "c2-after-compress");
	println!(
		"c2 done: interior={} leaf={}",
		c2.interior_nodes.len(),
		c2.leaf_nodes.len()
	);
}
