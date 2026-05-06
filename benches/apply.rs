use std::time::Instant;
use lattice::{
	chunk::{self, Chunk, VoxelEdit},
	shape::{Sphere, edit_packet_for_shape},
	tree::{lod, Aabb, Edit, EditPacket, Level, Tree, DELETE},
	types::Voxel,
};

const DEPTH: usize = chunk::DEPTH;
const SIDE: u64 = chunk::SIDE as u64;
type TestTree = Tree<DEPTH>;

// --- Setup ---

fn root_aabb() -> Aabb {
	Aabb { min: [0, 0, 0], max: [SIDE as i64; 3] }
}

fn aligned_center(radius: i64) -> [i64; 3] {
	for &n in &[4i64, 16, 64, 256] {
		if radius <= n / 2 - 1 {
			return [n / 2; 3];
		}
	}
	[128; 3]
}

fn blue() -> Voxel { Voxel::from_rgb_flags([100, 150, 200], 0, false, false, false, false) }
fn red()  -> Voxel { Voxel::from_rgb_flags([200, 80, 80],   0, false, false, false, false) }

fn sphere_packet(radius: i64, material: Voxel) -> EditPacket<DEPTH> {
	let sphere = Sphere { center: aligned_center(radius), radius, material };
	edit_packet_for_shape::<DEPTH>(&sphere, root_aabb())
}

fn apply_chunk(chunk: &mut Chunk, packet: EditPacket<DEPTH>) {
	chunk.add_shape_packet(packet);
	chunk.flush_edits();
}

fn apply_tree(tree: &mut TestTree, packet: EditPacket<DEPTH>) {
	tree.queue_edit_packet(packet);
	tree.apply_edits();
}

// --- SVO bytes estimate ---
//
// Walks the ESVO tree. For each ESVO leaf at level d, counts the SVO expansion:
// 1 node at d+1, 64 at d+2, 64^2 at d+3, ... all the way to the leaf level.
// Uses a fixed per-node cost: 20B structural overhead + packed slot storage.
// Each slot takes bits_per_val bits for its value + 16 bits for its child index
// (except at the leaf level where no child index is needed).
// 16-bit child indices are a conservative estimate; actual SVO may use fewer bits for
// sparse trees, but this keeps the SVO ≥ ESVO invariant.

fn node_cost(d: usize, bits_per_val: usize) -> usize {
	let child_bits = if d + 1 < DEPTH { 16usize } else { 0 };
	20 + (64 * (bits_per_val + child_bits) + 7) / 8
}

fn svo_bytes_estimate(t: &TestTree, bits_per_val: usize) -> usize {
	if !t.occupied { return 0; }
	if t.is_leaf {
		return (0..DEPTH).map(|d| 64usize.pow(d as u32) * node_cost(d, bits_per_val)).sum();
	}
	if t.levels[0].node_count() == 0 { return 0; }
	let root = t.levels[0].node_count() - 1;
	svo_node_bytes(&t.levels, 0, root, bits_per_val)
}

fn svo_node_bytes(levels: &[Level], d: usize, node: u32, bits_per_val: usize) -> usize {
	let level = &levels[d];
	let occ  = level.occupancy_mask[node as usize];
	let leaf = level.leaf_mask[node as usize];
	let base = level.children_offset[node as usize];
	let is_leaf_level = d + 1 == levels.len();

	let mut total = node_cost(d, bits_per_val);
	if is_leaf_level { return total; }

	let mut mask = occ;
	while mask != 0 {
		let s    = mask.trailing_zeros() as usize;
		let rank = (occ & ((1u64 << s) - 1)).count_ones();
		if (leaf >> s) & 1 != 0 {
			// One ESVO leaf expands to: 1 node at d+1, 64 at d+2, 64^2 at d+3, ...
			for k in 1..=(levels.len() - d - 1) {
				total += 64usize.pow((k - 1) as u32) * node_cost(d + k, bits_per_val);
			}
		} else {
			let child = level.node_children.get(base + rank);
			total += svo_node_bytes(levels, d + 1, child, bits_per_val);
		}
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
	if f < KB      { format!("{} B", n) }
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

fn print_chunk_stats(label: &str, chunk: &Chunk) {
	let mut t = chunk.tree.clone();
	t.compact();

	let mat_count    = chunk.materials.values.len();
	let mat_bytes    = mat_count * 4;
	let tree_bytes   = t.bytes();
	let total_bytes  = tree_bytes + mat_bytes;
	let occupied     = t.stored_volume().max(1);
	let bpv          = (total_bytes * 8) as f64 / occupied as f64;
	let bpv_str      = if bpv >= 0.01 { format!("{bpv:.2}") } else { format!("{bpv:.4}") };
	let total_vol    = SIDE * SIDE * SIDE;
	let pct_empty    = (total_vol - occupied.min(total_vol)) as f64 / total_vol as f64 * 100.0;

	// Bits per material index: smallest power-of-2 >= log2(mat_count).
	let bits_per_idx: usize = if mat_count <= 2 { 1 }
		else if mat_count <= 4  { 2 }
		else if mat_count <= 16 { 4 }
		else if mat_count <= 256 { 8 }
		else if mat_count <= 65536 { 16 }
		else { 32 };

	// Compression chain (bytes at each step).
	let flat_u32  = SIDE * SIDE * SIDE * 4;
	let lut_flat  = (SIDE * SIDE * SIDE * bits_per_idx as u64 + 7) / 8 + mat_bytes as u64;
	let lut_svo   = svo_bytes_estimate(&t, bits_per_idx) as u64 + mat_bytes as u64;
	let lut_esvo  = t.esvo_bytes() as u64 + mat_bytes as u64;
	let lut_esvodag = total_bytes as u64;

	let r_lut   = flat_u32  as f64 / lut_flat.max(1)    as f64;
	let r_svo   = lut_flat  as f64 / lut_svo.max(1)     as f64;
	let r_esvo  = lut_svo   as f64 / lut_esvo.max(1)    as f64;
	let r_dag   = lut_esvo  as f64 / lut_esvodag.max(1) as f64;
	let r_total = flat_u32  as f64 / lut_esvodag.max(1) as f64;

	let val_bits:   Vec<u8> = t.levels.iter().map(|l| l.values.bits).collect();
	let child_bits: Vec<u8> = t.levels.iter().map(|l| l.node_children.bits).collect();

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

// --- Validation ---

fn assert_chunk_valid(chunk: &Chunk) {
	assert_tree_valid(&chunk.tree);
}

fn assert_tree_valid(tree: &TestTree) {
	let mut t = tree.clone();
	t.compact();
	for d in 0..DEPTH {
		let level = &t.levels[d];
		for n in 0..level.node_count() {
			let occ   = level.occupancy_mask[n as usize];
			let leaf  = level.leaf_mask[n as usize];
			let base  = level.children_offset[n as usize];
			let count = occ.count_ones();
			assert!(
				base + count <= level.node_children.len(),
				"level {d} node {n}: children out of bounds (base={base} count={count} len={})",
				level.node_children.len()
			);
			if d + 1 < DEPTH {
				let mut mask = occ & !leaf;
				while mask != 0 {
					let slot = mask.trailing_zeros() as u8;
					let rank = (occ & ((1u64 << slot) - 1)).count_ones();
					let child = level.node_children.get(base + rank);
					assert!(
						(child as usize) < t.levels[d + 1].node_count() as usize,
						"level {d} node {n} slot {slot}: child {child} out of range",
					);
					mask &= mask - 1;
				}
			}
		}
	}
}

// --- Tests ---

fn run_tests() {
	// 1. Empty chunk stays empty after empty packet.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, EditPacket::new(true));
		assert!(!chunk.tree.occupied);
		assert_eq!(chunk.materials.values.len(), 1); // slot 0 = air (reserved)
	}

	// 2. Single voxel edit adds exactly one material entry.
	{
		let mut chunk = Chunk::new();
		chunk.queue_edit(VoxelEdit { pos: [0, 0, 0], voxel: Some(blue()) });
		chunk.flush_edits();
		assert!(chunk.tree.occupied);
		assert_eq!(chunk.materials.values.len(), 2); // slot 0 = air, slot 1 = blue
		assert_chunk_valid(&chunk);
	}

	// 3. Two different materials produce two material entries.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(80, blue()));
		apply_chunk(&mut chunk, sphere_packet(40, red()));
		assert_eq!(chunk.materials.values.len(), 3); // slot 0 = air, 1 = blue, 2 = red
		assert_chunk_valid(&chunk);
	}

	// 4. Same material applied twice still produces one entry.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(80, blue()));
		apply_chunk(&mut chunk, sphere_packet(40, blue()));
		assert_eq!(chunk.materials.values.len(), 2); // slot 0 = air, slot 1 = blue
		assert_chunk_valid(&chunk);
	}

	// 5. Sphere produces valid chunk.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(100, blue()));
		assert!(chunk.tree.occupied);
		assert_chunk_valid(&chunk);
	}

	// 6. Sphere then delete-sphere.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(100, blue()));
		let del = Sphere { center: [SIDE as i64 / 2; 3], radius: 100, material: Voxel::air() };
		apply_chunk(&mut chunk, edit_packet_for_shape::<DEPTH>(&del, root_aabb()));
		assert_chunk_valid(&chunk);
	}

	// 7. Delete a single voxel from an otherwise solid chunk.
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(512, blue()));
		chunk.queue_edit(VoxelEdit { pos: [0, 0, 0], voxel: None });
		chunk.flush_edits();
		assert!(chunk.tree.occupied);
		assert_chunk_valid(&chunk);
	}

	// 8. Idempotency: applying same packet twice matches single application.
	{
		let mut c1 = Chunk::new();
		let mut c2 = Chunk::new();
		let p = sphere_packet(60, blue());
		apply_chunk(&mut c1, p.clone());
		apply_chunk(&mut c2, p.clone());
		apply_chunk(&mut c2, p);
		assert_chunk_valid(&c1);
		assert_chunk_valid(&c2);
		assert_eq!(c1.materials.values.len(), c2.materials.values.len());
	}

	// 9. get_voxel returns the correct material after round-trip.
	{
		let mut chunk = Chunk::new();
		let v = blue();
		chunk.queue_edit(VoxelEdit { pos: [10, 20, 30], voxel: Some(v) });
		chunk.flush_edits();
		assert_eq!(chunk.get_voxel([10, 20, 30]), Some(v));
		assert_eq!(chunk.get_voxel([10, 20, 31]), None);
	}

	// --- Tree-internal tests (raw u32 values, LOD) ---

	// 10. Root-level edit collapses to leaf.
	{
		let mut tree = TestTree::new(1);
		apply_tree(&mut tree, sphere_packet(100, blue()));
		let mut p = EditPacket::new(false);
		p.add_edit(Edit::new(99, [0, 0, 0], 0, 1));
		apply_tree(&mut tree, p);
		assert!(tree.is_leaf && tree.value == 99);
	}

	// 11. Root DELETE clears tree.
	{
		let mut tree = TestTree::new(1);
		apply_tree(&mut tree, sphere_packet(100, blue()));
		let mut p = EditPacket::new(false);
		p.add_edit(Edit::new(DELETE, [0, 0, 0], 0, 1));
		apply_tree(&mut tree, p);
		assert!(!tree.occupied);
	}

	// 12. Expanding a root leaf with a sub-voxel edit.
	{
		let mut tree = TestTree::new(1);
		let mut p = EditPacket::new(false);
		p.add_edit(Edit::new(1, [0, 0, 0], 0, 1));
		apply_tree(&mut tree, p);
		let mut p2 = EditPacket::new(false);
		p2.add_edit(Edit::new(2, [0, 0, 0], DEPTH as u8, 1));
		apply_tree(&mut tree, p2);
		assert!(tree.occupied && !tree.is_leaf);
		assert_tree_valid(&tree);
	}

	// 13. Merge 64 empty children → empty tree.
	{
		let children: [TestTree; 64] = std::array::from_fn(|_| TestTree::new(1));
		let merged = lod::merge(&children);
		assert!(!merged.occupied);
	}

	// 14. Merge 64 uniform leaf children → occupied tree.
	{
		let children: [TestTree; 64] = std::array::from_fn(|_| {
			let mut t = TestTree::new(1);
			t.occupied = true;
			t.is_leaf = true;
			t.value = 42;
			t
		});
		let merged = lod::merge(&children);
		assert!(merged.occupied);
		assert_tree_valid(&merged);
	}

	// 15. Merge sphere children, split back, each child occupied.
	{
		let mut children: [TestTree; 64] = std::array::from_fn(|_| TestTree::new(1));
		for child in children.iter_mut() {
			apply_tree(child, sphere_packet(2, blue()));
		}
		let merged = lod::merge(&children);
		assert!(merged.occupied);
		assert_tree_valid(&merged);
		let split = lod::split(&merged);
		for t in &split {
			assert!(t.occupied, "expected occupied after split");
		}
	}

	// 16. Split a full leaf → 64 leaf children.
	{
		let mut tree = TestTree::new(4);
		tree.occupied = true;
		tree.is_leaf = true;
		tree.value = 7;
		let split = lod::split(&tree);
		for t in &split {
			assert!(t.occupied && t.is_leaf && t.value == 7);
		}
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

fn make_grid_spheres() -> Chunk {
	let aabb = root_aabb();
	let mut chunk = Chunk::new();
	for x in 0..16i64 {
		for y in 0..16i64 {
			for z in 0..16i64 {
				let center = [x * 16 + 8, y * 16 + 8, z * 16 + 8];
				let sphere = Sphere { center, radius: 7, material: blue() };
				apply_chunk(&mut chunk, edit_packet_for_shape::<DEPTH>(&sphere, aabb));
			}
		}
	}
	chunk
}

fn print_packet_levels(label: &str, packet: &EditPacket<DEPTH>) {
	let mut counts = [0u32; DEPTH + 1];
	for path in &packet.paths {
		counts[path.depth() as usize] += 1;
	}
	print!("  {label} ({} edits):", packet.paths.len());
	for (depth, &count) in counts.iter().enumerate() {
		if count > 0 { print!(" D{depth}={count}"); }
	}
	println!();
}

// --- Main ---

fn main() {
	run_tests();

	let radii: Vec<i64> = (0..=7).map(|i| 1i64 << i).collect();

	println!("\nsphere edit packet depth distribution:");
	for &r in &radii {
		print_packet_levels(&format!("r={r:3}"), &sphere_packet(r, blue()));
	}

	println!("\nsphere stats:");
	for &r in &radii {
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(r, blue()));
		print_chunk_stats(&format!("r={r:3}"), &chunk);
	}

	println!("\napply_sphere_fresh:");
	for &r in &radii {
		let packet = sphere_packet(r, blue());
		let d = time_one(|| {
			let mut chunk = Chunk::new();
			apply_chunk(&mut chunk, packet.clone());
			std::hint::black_box(chunk);
		});
		println!("  r={r:3}: {}", fmt_duration(d));
	}

	let mut full_chunk = Chunk::new();
	apply_chunk(&mut full_chunk, sphere_packet(512, blue()));

	println!("\napply_sphere_onto_full:");
	for &r in &radii {
		let packet = sphere_packet(r, blue());
		let d = time_one(|| {
			let mut chunk = full_chunk.clone();
			apply_chunk(&mut chunk, packet.clone());
			std::hint::black_box(chunk);
		});
		println!("  r={r:3}: {}", fmt_duration(d));
	}

	println!("\nr=512 sphere (entire chunk collapses to root leaf):");
	print_chunk_stats("stats", &full_chunk);

	println!("\nsingle r=7 sphere:");
	{
		let mut chunk = Chunk::new();
		apply_chunk(&mut chunk, sphere_packet(7, blue()));
		print_chunk_stats("stats", &chunk);
	}

	println!("\ngrid spheres (4096 × r=7, aligned for DAG dedup):");
	print_chunk_stats("stats", &make_grid_spheres());
}
