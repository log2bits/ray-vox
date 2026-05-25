use ray_vox::{
    chunk::{material::Material, node::CellState, Chunk, Compressed, Editing},
    volumes::{self, Sphere},
};

fn blue() -> Material {
    Material::from_rgb_pbr_id([100, 150, 200], 0)
}

fn validate(chunk: &Chunk<Compressed>, label: &str) {
    if chunk.state.interior_nodes.is_empty() {
        return;
    }
    let root = chunk.state.interior_nodes.len() - 1;
    let mut queue: Vec<(usize, u8)> = vec![(root, 0)];
    let mut i = 0;
    while i < queue.len() {
        let (idx, depth) = queue[i];
        i += 1;
        let node = chunk.state.interior_nodes[idx];
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

fn validate_editing(chunk: &Chunk<Editing>, label: &str) {
    if chunk.state.interior_nodes.is_empty() {
        return;
    }
    let root = chunk.state.interior_nodes.len() - 1;
    let mut queue: Vec<(usize, u8)> = vec![(root, 0)];
    let mut i = 0;
    while i < queue.len() {
        let (idx, depth) = queue[i];
        i += 1;
        let node = chunk.state.interior_nodes[idx];
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
    let mut c1 = Chunk::<Editing>::new();
    volumes::stamp(
        &Sphere { center: [128.0; 3], radius: 32.0, material: blue() },
        &mut c1.state.edits,
    );
    c1.apply_edits();
    validate_editing(&c1, "c1-after-apply_edits");
    println!(
        "c1 done: interior={} leaf={}",
        c1.state.interior_nodes.len(),
        c1.leaf_nodes.len()
    );

    let mut c2 = c1.clone();
    volumes::stamp(
        &Sphere { center: [128.0; 3], radius: 32.0, material: blue() },
        &mut c2.state.edits,
    );
    {
        let mut edits = std::mem::take(&mut c2.state.edits);
        edits.sort();
        for (i, batch) in edits.batches.iter().enumerate() {
            let start = batch.range().start as usize;
            let end = batch.range().end as usize;
            c2.apply_batch(&edits.edits[start..end]);
            let depth_of_batch = edits.edits[start].0.depth();
            validate_editing(&c2, &format!("c2-batch{i}(depth={depth_of_batch})"));
            if i < 5 {
                println!(
                    "  batch {i}: n_edits={} depth={} interior={} leaf={}",
                    end - start,
                    depth_of_batch,
                    c2.state.interior_nodes.len(),
                    c2.leaf_nodes.len()
                );
            }
        }
    }
    let c2 = c2.compress();
    validate(&c2, "c2-after-compress");
    println!(
        "c2 done: interior={} leaf={}",
        c2.state.interior_nodes.len(),
        c2.leaf_nodes.len()
    );
}
