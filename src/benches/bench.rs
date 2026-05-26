use std::time::Instant;

use ray_vox::{
    chunk::{
        edit::{Edits, Path},
        material::Material,
        node::{CellState, InteriorNode, LeafNode},
        Chunk, Compressed, Editing,
    },
    volumes::{self, Sphere},
};

const DEPTH: usize = 4;
const SIDE: u64 = 256;

// --- Setup ---

fn aligned_center(radius: f32) -> [f32; 3] {
    for &n in &[4f32, 16.0, 64.0, 256.0] {
        if radius <= n / 2.0 - 1.0 {
            return [n / 2.0; 3];
        }
    }
    [128.0; 3]
}

fn blue() -> Material { Material::from_rgb_pbr_id([100, 150, 200], 0) }
fn red()  -> Material { Material::from_rgb_pbr_id([200, 80, 80],   0) }

fn sphere_edits(radius: f32, material: Material) -> Edits {
    let mut edits = Edits::new();
    let center = aligned_center(radius);
    volumes::stamp(&Sphere { center, radius, material }, &mut edits);
    edits
}

fn apply_edits(chunk: &mut Chunk<Editing>, edits: Edits) {
    chunk.state.edits = edits;
    chunk.apply_edits();
}

// --- get_voxel ---

fn get_voxel(chunk: &Chunk<Compressed>, pos: [u8; 3]) -> Option<Material> {
    if chunk.is_empty() { return None; }
    if chunk.is_uniform() { return Some(chunk.materials.get(0)); }
    if chunk.state.interior_nodes.is_empty() { return None; }

    let path = Path::from_coords(pos, 4);
    let interior = &chunk.state.interior_nodes;
    let leaves = &chunk.leaf_nodes;
    let mut node_idx = (interior.len() - 1) as u32;

    for depth in 0..=2u8 {
        let slot = path.slot_at(depth);
        let node = interior[node_idx as usize];
        match node.state(slot) {
            CellState::Empty => return None,
            CellState::Filled => {
                return Some(chunk.materials.get(node.material_index(slot)));
            }
            CellState::Interior => {
                node_idx = node.interior_child_index(slot);
            }
            CellState::Leaf => {
                let leaf_idx = node.leaf_child_index(slot);
                let leaf_slot = path.slot_at(3);
                let leaf = leaves[leaf_idx as usize];
                return if leaf.is_occupied(leaf_slot) {
                    Some(chunk.materials.get(leaf.material_index(leaf_slot)))
                } else {
                    None
                };
            }
        }
    }
    None
}

// --- Validation ---

fn assert_chunk_valid(chunk: &Chunk<Compressed>) {
    let interior = &chunk.state.interior_nodes;
    let leaves = &chunk.leaf_nodes;
    if interior.is_empty() { return; }
    let root = (interior.len() - 1) as u32;
    validate_interior(interior, leaves, root, 0);
}

fn validate_interior(interior: &[InteriorNode], leaves: &[LeafNode], idx: u32, depth: u8) {
    let n = interior[idx as usize];
    let mut mask = n.has_child() & !n.is_leaf();
    if depth < 2 {
        let base = n.interior_offset();
        let count = mask.count_ones();
        assert!(
            base + count <= interior.len() as u32,
            "depth={depth} node={idx}: interior children out of bounds (base={base} count={count} len={})",
            interior.len(),
        );
        let mut rank = 0u32;
        while mask != 0 {
            let child = base + rank;
            validate_interior(interior, leaves, child, depth + 1);
            rank += 1;
            mask &= mask - 1;
        }
    }
    let leaf_base = n.leaf_offset();
    let leaf_count = (n.has_child() & n.is_leaf()).count_ones();
    assert!(
        leaf_base + leaf_count <= leaves.len() as u32,
        "depth={depth} node={idx}: leaf children out of bounds (base={leaf_base} count={leaf_count} len={})",
        leaves.len(),
    );
}

// --- Stored volume ---

fn stored_volume(chunk: &Chunk<Compressed>) -> u64 {
    if chunk.is_empty() { return 0; }
    if chunk.is_uniform() { return SIDE * SIDE * SIDE; }
    if chunk.state.interior_nodes.is_empty() { return 0; }
    let root = (chunk.state.interior_nodes.len() - 1) as u32;
    node_stored_volume(&chunk.state.interior_nodes, &chunk.leaf_nodes, root, 0)
}

fn node_stored_volume(interior: &[InteriorNode], leaves: &[LeafNode], idx: u32, depth: u8) -> u64 {
    let n = interior[idx as usize];
    let mut total = 0u64;
    let mut mask = n.occupancy();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        match n.state(slot) {
            CellState::Interior => {
                let child = n.interior_child_index(slot);
                total += node_stored_volume(interior, leaves, child, depth + 1);
            }
            CellState::Leaf => {
                let leaf = leaves[n.leaf_child_index(slot) as usize];
                total += leaf.occupied_count() as u64;
            }
            CellState::Filled => {
                let side = SIDE >> (2 * (depth as u64 + 1));
                total += side * side * side;
            }
            CellState::Empty => {}
        }
        mask &= mask - 1;
    }
    total
}

// --- SVO / ESVO byte estimates ---
//
// SVO:  expand every ESVO-collapsed (Filled) slot to full node depth.
//       Uses the same 16-bit child-index model as the lattice benchmark.
// ESVO: follow all paths from root; shared (DAG) nodes counted once per path.
//       Uses actual ray-vox node sizes: InteriorNode=24 B, LeafNode=12 B.

fn node_cost(d: usize, bits: usize) -> usize {
    let child_bits: usize = if d + 1 < DEPTH { 16 } else { 0 };
    20 + (64 * (bits + child_bits) + 7) / 8
}

fn svo_bytes(chunk: &Chunk<Compressed>, bits: usize) -> usize {
    if chunk.is_empty() { return 0; }
    if chunk.is_uniform() {
        // Fully filled: one node at each level, 64× children per level
        return (0..DEPTH).map(|d| 64usize.pow(d as u32) * node_cost(d, bits)).sum();
    }
    if chunk.state.interior_nodes.is_empty() { return 0; }
    let root = (chunk.state.interior_nodes.len() - 1) as u32;
    svo_node_bytes(&chunk.state.interior_nodes, 0, root, bits)
}

fn svo_node_bytes(interior: &[InteriorNode], depth: usize, idx: u32, bits: usize) -> usize {
    let n = interior[idx as usize];
    let mut total = node_cost(depth, bits);
    if depth + 1 >= DEPTH { return total; }

    let mut mask = n.occupancy();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        match n.state(slot) {
            CellState::Interior => {
                let child = n.interior_child_index(slot);
                total += svo_node_bytes(interior, depth + 1, child, bits);
            }
            CellState::Leaf => {
                total += node_cost(depth + 1, bits);
            }
            CellState::Filled => {
                for k in 1..=(DEPTH - depth - 1) {
                    total += 64usize.pow((k - 1) as u32) * node_cost(depth + k, bits);
                }
            }
            CellState::Empty => {}
        }
        mask &= mask - 1;
    }
    total
}

fn esvo_bytes(chunk: &Chunk<Compressed>) -> usize {
    if chunk.is_empty() || chunk.is_uniform() { return 0; }
    if chunk.state.interior_nodes.is_empty() { return 0; }
    let root = (chunk.state.interior_nodes.len() - 1) as u32;
    esvo_node_bytes(&chunk.state.interior_nodes, &chunk.leaf_nodes, root)
}

fn esvo_node_bytes(interior: &[InteriorNode], leaves: &[LeafNode], idx: u32) -> usize {
    let n = interior[idx as usize];
    let n_leaves = (n.has_child() & n.is_leaf()).count_ones() as usize;
    let mut total = std::mem::size_of::<InteriorNode>() + n_leaves * std::mem::size_of::<LeafNode>();

    let mut mask = n.has_child() & !n.is_leaf();
    let base = n.interior_offset();
    let mut rank = 0u32;
    while mask != 0 {
        total += esvo_node_bytes(interior, leaves, base + rank);
        rank += 1;
        mask &= mask - 1;
    }
    total
}

// --- Formatting helpers ---

fn fmt_bytes(n: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    let f = n as f64;
    if      f < KB { format!("{} B", n) }
    else if f < MB { format!("{:.2} KB", f / KB) }
    else if f < GB { format!("{:.2} MB", f / MB) }
    else           { format!("{:.2} GB", f / GB) }
}

fn fmt_f_commas(f: f64, decimals: usize) -> String {
    let formatted = format!("{:.prec$}", f, prec = decimals);
    let dot = formatted.find('.');
    let int_s = &formatted[..dot.unwrap_or(formatted.len())];
    let frac_s = dot.map(|d| &formatted[d..]).unwrap_or("");
    let mut rev = String::new();
    for (i, c) in int_s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { rev.push(','); }
        rev.push(c);
    }
    format!("{}{}", rev.chars().rev().collect::<String>(), frac_s)
}

fn fmt_pct(pct: f64) -> String {
    if pct >= 100.0 { return "100%".to_string(); }
    if pct <= 0.0   { return "0%".to_string(); }
    for d in 1..=10 {
        let s = format!("{:.prec$}%", pct, prec = d);
        if !s.starts_with("100") { return s; }
    }
    format!("{:.10}%", pct)
}

// --- Stats ---

fn print_chunk_stats(label: &str, chunk: &Chunk<Compressed>) {
    let mat_count    = chunk.materials.lut.len() as usize;
    let mat_bytes    = mat_count * std::mem::size_of::<Material>();
    let interior_b   = chunk.state.interior_nodes.len() * std::mem::size_of::<InteriorNode>();
    let leaf_b       = chunk.leaf_nodes.len() * std::mem::size_of::<LeafNode>();
    let idx_b        = chunk.materials.indices.words.len() * 4;
    let tree_bytes   = interior_b + leaf_b + idx_b;
    let total_bytes  = tree_bytes + mat_bytes;

    let occupied   = stored_volume(chunk).max(1);
    let bpv        = (total_bytes * 8) as f64 / occupied as f64;
    let bpv_str    = if bpv >= 0.01 { format!("{bpv:.2}") } else { format!("{bpv:.4}") };
    let total_vol  = SIDE * SIDE * SIDE;
    let pct_empty  = (total_vol - occupied.min(total_vol)) as f64 / total_vol as f64 * 100.0;

    let bits_per_idx: usize = if mat_count <= 2 { 1 }
        else if mat_count <=   4 { 2 }
        else if mat_count <=  16 { 4 }
        else if mat_count <= 256 { 8 }
        else if mat_count <= 65536 { 16 }
        else { 32 };

    let flat_u32    = SIDE * SIDE * SIDE * 4;
    let lut_flat    = (SIDE * SIDE * SIDE * bits_per_idx as u64 + 7) / 8 + mat_bytes as u64;
    let lut_svo     = svo_bytes(chunk, bits_per_idx) as u64 + mat_bytes as u64;
    let lut_esvo    = esvo_bytes(chunk) as u64 + mat_bytes as u64;
    let lut_esvodag = total_bytes as u64;

    let r_lut   = flat_u32  as f64 / lut_flat.max(1)    as f64;
    let r_svo   = lut_flat  as f64 / lut_svo.max(1)     as f64;
    let r_esvo  = lut_svo   as f64 / lut_esvo.max(1)    as f64;
    let r_dag   = lut_esvo  as f64 / lut_esvodag.max(1) as f64;
    let r_total = flat_u32  as f64 / lut_esvodag.max(1) as f64;

    // val: material index bits (single PalettedVec, same at all levels)
    // child: interior ptr = 13 bits (depths 0-1), leaf ptr = 19 bits (depth 2), 0 at leaf level
    let val_bits   = vec![chunk.materials.indices.bits as u8; DEPTH];
    let child_bits = vec![13u8, 13, 19, 0];

    println!(
        "  {label}: {} | {bpv_str} bits/voxel | {} empty \
         | total {}x (LUT {}x  SVO {}x  ESVO {}x  DAG {}x) \
         | val:{val_bits:?} child:{child_bits:?}",
        fmt_bytes(total_bytes),
        fmt_pct(pct_empty),
        fmt_f_commas(r_total, 1),
        fmt_f_commas(r_lut, 1),
        fmt_f_commas(r_svo, 2),
        fmt_f_commas(r_esvo, 1),
        fmt_f_commas(r_dag, 1),
    );
}

// --- Edit depth distribution ---

fn print_edit_depths(label: &str, edits: &Edits) {
    let mut counts = [0u32; DEPTH + 1];
    for (path, _) in &edits.edits {
        let d = path.depth() as usize;
        if d < counts.len() { counts[d] += 1; }
    }
    print!("  {label} ({} edits):", edits.edits.len());
    for (depth, &count) in counts.iter().enumerate() {
        if count > 0 { print!(" D{depth}={count}"); }
    }
    println!();
}

// --- Tests ---

fn run_tests() {
    // 1. Empty chunk stays empty after no-op stamp.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, Edits::new());
        let chunk = chunk.compress();
        assert!(chunk.is_empty());
        assert_eq!(chunk.materials.lut.len(), 0);
    }

    // 2. Single voxel edit adds exactly one material entry.
    {
        let mut chunk = Chunk::<Editing>::new();
        chunk.push_edit(Path::from_coords([0, 0, 0], 4), blue());
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(!chunk.is_empty());
        assert_eq!(chunk.materials.lut.len(), 1);
        assert_chunk_valid(&chunk);
    }

    // 3. Two different materials produce two material entries.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(80.0, blue()));
        apply_edits(&mut chunk, sphere_edits(40.0, red()));
        let chunk = chunk.compress();
        assert_eq!(chunk.materials.lut.len(), 2);
        assert_chunk_valid(&chunk);
    }

    // 4. Same material applied twice still produces one LUT entry.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(80.0, blue()));
        apply_edits(&mut chunk, sphere_edits(40.0, blue()));
        let chunk = chunk.compress();
        assert_eq!(chunk.materials.lut.len(), 1);
        assert_chunk_valid(&chunk);
    }

    // 5. Sphere produces a valid chunk.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(100.0, blue()));
        let chunk = chunk.compress();
        assert!(!chunk.is_empty());
        assert_chunk_valid(&chunk);
    }

    // 6. Sphere then air-sphere produces a valid (possibly empty) chunk.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(100.0, blue()));
        apply_edits(&mut chunk, sphere_edits(100.0, Material::air()));
        let chunk = chunk.compress();
        assert_chunk_valid(&chunk);
    }

    // 7. Delete a single voxel from an otherwise solid chunk.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(512.0, blue()));
        chunk.push_edit(Path::from_coords([0, 0, 0], 4), Material::air());
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(!chunk.is_empty());
        assert_chunk_valid(&chunk);
    }

    // 8. Idempotency: applying same edits twice gives same structure.
    {
        let mut c1 = Chunk::<Editing>::new();
        let mut c2 = Chunk::<Editing>::new();
        apply_edits(&mut c1, sphere_edits(60.0, blue()));
        apply_edits(&mut c2, sphere_edits(60.0, blue()));
        apply_edits(&mut c2, sphere_edits(60.0, blue()));
        let c1 = c1.compress();
        let c2 = c2.compress();
        assert_chunk_valid(&c1);
        assert_chunk_valid(&c2);
        assert_eq!(c1.materials.lut.len(), c2.materials.lut.len());
    }

    // 9. get_voxel returns the correct material after a voxel edit.
    {
        let mut chunk = Chunk::<Editing>::new();
        let v = blue();
        chunk.push_edit(Path::from_coords([10, 20, 30], 4), v);
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert_eq!(get_voxel(&chunk, [10, 20, 30]), Some(v));
        assert_eq!(get_voxel(&chunk, [10, 20, 31]), None);
    }

    // 10. Root-level fill produces a uniform chunk (no tree nodes).
    {
        let mut chunk = Chunk::<Editing>::new();
        chunk.push_edit(Path::from(0u32), blue()); // root fill
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(chunk.is_uniform());
        assert!(chunk.state.interior_nodes.is_empty());
    }

    // 11. Root-level air over filled chunk empties it.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(100.0, blue()));
        chunk.push_edit(Path::from(0u32), Material::air()); // root air fill
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(chunk.is_empty());
    }

    // 12. Sub-voxel edit after root fill expands the tree.
    {
        let mut chunk = Chunk::<Editing>::new();
        chunk.push_edit(Path::from(0u32), blue());
        chunk.apply_edits();
        chunk.push_edit(Path::from_coords([0, 0, 0], 4), red());
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(!chunk.is_uniform());
        assert_chunk_valid(&chunk);
    }

    // 13. A single voxel produces expected tree depth (≥3 interior nodes).
    {
        let mut chunk = Chunk::<Editing>::new();
        chunk.push_edit(Path::from_coords([0, 0, 0], 4), blue());
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert!(chunk.state.interior_nodes.len() >= 3, "expected ≥3 interior nodes for single voxel");
        assert_eq!(chunk.leaf_nodes.len(), 1);
        assert_chunk_valid(&chunk);
    }

    // 14. Max-fill sphere has ≤ MAX_INTERIOR_NODES interior nodes.
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(512.0, blue()));
        apply_edits(&mut chunk, sphere_edits(64.0, red()));
        let chunk = chunk.compress();
        assert!(
            chunk.state.interior_nodes.len() <= 4161,
            "interior nodes {} exceed theoretical max 4161",
            chunk.state.interior_nodes.len()
        );
        assert_chunk_valid(&chunk);
    }

    // 15. Re-stamping a sphere gives the same node count (idempotent structure).
    {
        let mut c1 = Chunk::<Editing>::new();
        apply_edits(&mut c1, sphere_edits(60.0, blue()));
        let c1 = c1.compress();

        let mut c2 = c1.clone().decompress();
        apply_edits(&mut c2, sphere_edits(60.0, blue()));
        let c2 = c2.compress();

        assert_eq!(
            c1.state.interior_nodes.len(),
            c2.state.interior_nodes.len(),
            "idempotent stamp should produce same tree"
        );
        assert_eq!(c1.leaf_nodes.len(), c2.leaf_nodes.len());
    }

    // 16. Two materials are both retrievable via get_voxel.
    {
        let mut chunk = Chunk::<Editing>::new();
        chunk.push_edit(Path::from_coords([0, 0, 0], 4), blue());
        chunk.push_edit(Path::from_coords([255, 255, 255], 4), red());
        chunk.apply_edits();
        let chunk = chunk.compress();
        assert_eq!(get_voxel(&chunk, [0, 0, 0]), Some(blue()));
        assert_eq!(get_voxel(&chunk, [255, 255, 255]), Some(red()));
        assert_eq!(get_voxel(&chunk, [128, 128, 128]), None);
        assert_chunk_valid(&chunk);
    }

    // 17. Filling a 4³ leaf region with one material collapses to Filled on the parent.
    {
        let mut chunk = Chunk::<Editing>::new();
        // Fill a 4³ leaf region (positions [0..4, 0..4, 0..4]) with blue.
        for x in 0u8..4 {
            for y in 0u8..4 {
                for z in 0u8..4 {
                    chunk.push_edit(Path::from_coords([x, y, z], 4), blue());
                }
            }
        }
        chunk.apply_edits();
        // The leaf at [0,0,0] depth-3 should collapse to a Filled slot — no leaf nodes.
        assert_eq!(chunk.leaf_nodes.len(), 0, "uniform 4³ region should collapse to Filled, not Leaf");
    }

    println!("all tests passed");
}

// --- Timing ---

fn time_one<F: FnOnce()>(f: F) -> std::time::Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}

fn fmt_duration(d: std::time::Duration) -> String {
    let ns = d.as_nanos();
    if      ns < 1_000         { format!("{ns}ns") }
    else if ns < 1_000_000     { format!("{:.1}µs", ns as f64 / 1_000.0) }
    else if ns < 1_000_000_000 { format!("{:.2}ms", ns as f64 / 1_000_000.0) }
    else                       { format!("{:.3}s",  ns as f64 / 1_000_000_000.0) }
}

// --- Benchmark data ---

fn make_grid_spheres() -> Chunk<Compressed> {
    let mut chunk = Chunk::<Editing>::new();
    for x in 0..16i32 {
        for y in 0..16i32 {
            for z in 0..16i32 {
                let center = [
                    x as f32 * 16.0 + 8.0,
                    y as f32 * 16.0 + 8.0,
                    z as f32 * 16.0 + 8.0,
                ];
                apply_edits(&mut chunk, sphere_edits_center(7.0, center, blue()));
            }
        }
    }
    chunk.compress()
}

fn sphere_edits_center(radius: f32, center: [f32; 3], material: Material) -> Edits {
    let mut edits = Edits::new();
    volumes::stamp(&Sphere { center, radius, material }, &mut edits);
    edits
}

// --- Main ---

fn main() {
    run_tests();

    let radii: Vec<f32> = (0..=7).map(|i| (1u32 << i) as f32).collect();

    println!("\nsphere edit packet depth distribution:");
    for &r in &radii {
        print_edit_depths(&format!("r={r:3}"), &sphere_edits(r, blue()));
    }

    println!("\nsphere stats:");
    for &r in &radii {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(r, blue()));
        print_chunk_stats(&format!("r={r:3}"), &chunk.compress());
    }

    println!("\napply_sphere_fresh:");
    for &r in &radii {
        let edits = sphere_edits(r, blue());
        let d = time_one(|| {
            let mut chunk = Chunk::<Editing>::new();
            apply_edits(&mut chunk, edits.clone());
            std::hint::black_box(chunk.compress());
        });
        println!("  r={r:3}: {}", fmt_duration(d));
    }

    let mut full_chunk_edit = Chunk::<Editing>::new();
    apply_edits(&mut full_chunk_edit, sphere_edits(512.0, blue()));
    let full_chunk = full_chunk_edit.compress();

    println!("\napply_sphere_onto_full:");
    for &r in &radii {
        let edits = sphere_edits(r, blue());
        let d = time_one(|| {
            let mut chunk = full_chunk.clone().decompress();
            apply_edits(&mut chunk, edits.clone());
            std::hint::black_box(chunk.compress());
        });
        println!("  r={r:3}: {}", fmt_duration(d));
    }

    println!("\nr=512 sphere (entire chunk collapses to root leaf):");
    print_chunk_stats("stats", &full_chunk);

    println!("\nsingle r=7 sphere:");
    {
        let mut chunk = Chunk::<Editing>::new();
        apply_edits(&mut chunk, sphere_edits(7.0, blue()));
        print_chunk_stats("stats", &chunk.compress());
    }

    println!("\ngrid spheres (4096 × r=7, aligned for DAG dedup):");
    print_chunk_stats("stats", &make_grid_spheres());
}
