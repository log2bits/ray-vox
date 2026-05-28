use std::time::Instant;

use ray_vox::{
	chunk::{
		Chunk,
		edit::{ChunkEdits, Path},
		material::Material,
		node::{CellState, InteriorNode, LeafNode},
	},
	generate::volume::sphere::Sphere,
	world::WorldPos,
};

const DEPTH: usize = 4;
const SIDE: u64 = 256;

fn aligned_center(radius: f32) -> [f32; 3] {
	for &n in &[4f32, 16.0, 64.0, 256.0] {
		if radius <= n / 2.0 - 1.0 {
			return [n / 2.0; 3];
		}
	}
	[128.0; 3]
}

fn blue() -> Material {
	Material::from_rgb_pbr_id([100, 150, 200], 0)
}
fn red() -> Material {
	Material::from_rgb_pbr_id([200, 80, 80], 0)
}

fn sphere_edits(radius: f32, material: Material) -> ChunkEdits {
	use ray_vox::generate::LazyEdit;
	use ray_vox::world::clipmap::{Clipmap, ClipmapChunkId};
	let center = aligned_center(radius);
	let clipmap = Clipmap {
		occupancy: [[0u32; 16]; 11],
		origin: WorldPos::from([0; 3]),
		pending_remap: Vec::new(),
		pending_origin: WorldPos::from([0; 3]),
	};
	Sphere {
		center,
		radius,
		material,
	}
	.generate(ClipmapChunkId::new(10, 4, 4, 4), &clipmap)
}

fn get_voxel(chunk: &Chunk, pos: [u8; 3]) -> Option<Material> {
	if chunk.is_empty() {
		return None;
	}
	if chunk.is_uniform() {
		return Some(chunk.materials.get(0));
	}
	if chunk.interior_nodes.is_empty() {
		return None;
	}

	let path = Path::from_coords(pos, 4);
	let interior = &chunk.interior_nodes;
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
				let leaf_slot = path.slot_at(depth + 1);
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

fn assert_chunk_valid(chunk: &Chunk) {
	let interior = &chunk.interior_nodes;
	let leaves = &chunk.leaf_nodes;
	if interior.is_empty() {
		return;
	}
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

fn fmt_bytes(n: usize) -> String {
	const KB: f64 = 1024.0;
	const MB: f64 = 1024.0 * KB;
	const GB: f64 = 1024.0 * MB;
	let f = n as f64;
	if f < KB {
		format!("{} B", n)
	} else if f < MB {
		format!("{:.2} KB", f / KB)
	} else if f < GB {
		format!("{:.2} MB", f / MB)
	} else {
		format!("{:.2} GB", f / GB)
	}
}

fn fmt_f_commas(f: f64, decimals: usize) -> String {
	let formatted = format!("{:.prec$}", f, prec = decimals);
	let dot = formatted.find('.');
	let int_s = &formatted[..dot.unwrap_or(formatted.len())];
	let frac_s = dot.map(|d| &formatted[d..]).unwrap_or("");
	let mut rev = String::new();
	for (i, c) in int_s.chars().rev().enumerate() {
		if i > 0 && i % 3 == 0 {
			rev.push(',');
		}
		rev.push(c);
	}
	format!("{}{}", rev.chars().rev().collect::<String>(), frac_s)
}

fn fmt_pct(pct: f64) -> String {
	if pct >= 100.0 {
		return "100%".to_string();
	}
	if pct <= 0.0 {
		return "0%".to_string();
	}
	for d in 1..=10 {
		let s = format!("{:.prec$}%", pct, prec = d);
		if !s.starts_with("100") {
			return s;
		}
	}
	format!("{:.10}%", pct)
}

fn print_chunk_stats(label: &str, chunk: &Chunk) {
	let total_bytes = chunk.gpu_size_bytes() as usize;

	let occupied = chunk.stored_volume().max(1);
	let bpv = (total_bytes * 8) as f64 / occupied as f64;
	let bpv_str = if bpv >= 0.01 {
		format!("{bpv:.2}")
	} else {
		format!("{bpv:.4}")
	};
	let total_vol = SIDE * SIDE * SIDE;
	let pct_empty = (total_vol - occupied.min(total_vol)) as f64 / total_vol as f64 * 100.0;

	let flat_u32 = SIDE * SIDE * SIDE * 4;

	let r_total = flat_u32 as f64 / total_bytes as f64;

	// val: material index bits (single PalettedVec, same at all levels)
	// child: interior ptr = 13 bits (depths 0-1), leaf ptr = 19 bits (depth 2), 0 at leaf level
	let val_bits = vec![chunk.materials.indices.bits as u8; DEPTH];
	let child_bits = vec![13u8, 13, 19, 0];

	println!(
		"  {label}: {} | {bpv_str} bits/voxel | {} empty \
         | total {}x \
         | val:{val_bits:?} child:{child_bits:?}",
		fmt_bytes(total_bytes),
		fmt_pct(pct_empty),
		fmt_f_commas(r_total, 1),
	);
}

fn print_edit_depths(label: &str, edits: &ChunkEdits) {
	let mut counts = [0u32; DEPTH + 1];
	for (path, _) in &edits.edits {
		let d = path.depth() as usize;
		if d < counts.len() {
			counts[d] += 1;
		}
	}
	print!("  {label} ({} edits):", edits.edits.len());
	for (depth, &count) in counts.iter().enumerate() {
		if count > 0 {
			print!(" D{depth}={count}");
		}
	}
	println!();
}

fn run_tests() {
	// 1. Empty chunk stays empty after no-op stamp.
	{
		let chunk = Chunk::new().apply_edits(ChunkEdits::new(WorldPos::from([0; 3])));
		assert!(chunk.is_empty());
		assert_eq!(chunk.materials.lut.len(), 0);
	}

	// 2. Single voxel edit adds exactly one material entry.
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from_coords([0, 0, 0], 4), blue());
		let chunk = Chunk::new().apply_edits(edits);
		assert!(!chunk.is_empty());
		assert_eq!(chunk.materials.lut.len(), 1);
		assert_chunk_valid(&chunk);
	}

	// 3. Two different materials produce two material entries.
	{
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(80.0, blue()))
			.apply_edits(sphere_edits(40.0, red()));
		assert_eq!(chunk.materials.lut.len(), 2);
		assert_chunk_valid(&chunk);
	}

	// 4. Same material applied twice still produces one LUT entry.
	{
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(80.0, blue()))
			.apply_edits(sphere_edits(40.0, blue()));
		assert_eq!(chunk.materials.lut.len(), 1);
		assert_chunk_valid(&chunk);
	}

	// 5. Sphere produces a valid chunk.
	{
		let chunk = Chunk::new().apply_edits(sphere_edits(100.0, blue()));
		assert!(!chunk.is_empty());
		assert_chunk_valid(&chunk);
	}

	// 6. Sphere then air-sphere produces a valid (possibly empty) chunk.
	{
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(100.0, blue()))
			.apply_edits(sphere_edits(100.0, Material::air()));
		assert_chunk_valid(&chunk);
	}

	// 7. Delete a single voxel from an otherwise solid chunk.
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from_coords([0, 0, 0], 4), Material::air());
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(512.0, blue()))
			.apply_edits(edits);
		assert!(!chunk.is_empty());
		assert_chunk_valid(&chunk);
	}

	// 8. Idempotency: applying same edits twice gives same structure.
	{
		let c1 = Chunk::new().apply_edits(sphere_edits(60.0, blue()));
		let c2 = Chunk::new()
			.apply_edits(sphere_edits(60.0, blue()))
			.apply_edits(sphere_edits(60.0, blue()));
		assert_chunk_valid(&c1);
		assert_chunk_valid(&c2);
		assert_eq!(c1.materials.lut.len(), c2.materials.lut.len());
	}

	// 9. get_voxel returns the correct material after a voxel edit.
	{
		let v = blue();
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from_coords([10, 20, 30], 4), v);
		let chunk = Chunk::new().apply_edits(edits);
		assert_eq!(get_voxel(&chunk, [10, 20, 30]), Some(v));
		assert_eq!(get_voxel(&chunk, [10, 20, 31]), None);
	}

	// 10. Root-level fill produces a uniform chunk (no tree nodes).
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from(0u32), blue()); // root fill
		let chunk = Chunk::new().apply_edits(edits);
		assert!(chunk.is_uniform());
		assert!(chunk.interior_nodes.is_empty());
	}

	// 11. Root-level air over filled chunk empties it.
	{
		let mut air_edits = ChunkEdits::new(WorldPos::from([0; 3]));
		air_edits.push(Path::from(0u32), Material::air()); // root air fill
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(100.0, blue()))
			.apply_edits(air_edits);
		assert!(chunk.is_empty());
	}

	// 12. Sub-voxel edit after root fill expands the tree.
	{
		let mut fill_edits = ChunkEdits::new(WorldPos::from([0; 3]));
		fill_edits.push(Path::from(0u32), blue());
		let mut sub_edits = ChunkEdits::new(WorldPos::from([0; 3]));
		sub_edits.push(Path::from_coords([0, 0, 0], 4), red());
		let chunk = Chunk::new().apply_edits(fill_edits).apply_edits(sub_edits);
		assert!(!chunk.is_uniform());
		assert_chunk_valid(&chunk);
	}

	// 13. A single voxel produces expected tree depth (≥3 interior nodes).
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from_coords([0, 0, 0], 4), blue());
		let chunk = Chunk::new().apply_edits(edits);
		assert!(
			chunk.interior_nodes.len() >= 3,
			"expected ≥3 interior nodes for single voxel"
		);
		assert_eq!(chunk.leaf_nodes.len(), 1);
		assert_chunk_valid(&chunk);
	}

	// 14. Max-fill sphere has ≤ MAX_INTERIOR_NODES interior nodes.
	{
		let chunk = Chunk::new()
			.apply_edits(sphere_edits(512.0, blue()))
			.apply_edits(sphere_edits(64.0, red()));
		assert!(
			chunk.interior_nodes.len() <= 4161,
			"interior nodes {} exceed theoretical max 4161",
			chunk.interior_nodes.len()
		);
		assert_chunk_valid(&chunk);
	}

	// 15. Re-stamping a sphere gives the same node count (idempotent structure).
	{
		let c1 = Chunk::new().apply_edits(sphere_edits(60.0, blue()));
		let c2 = c1.clone().apply_edits(sphere_edits(60.0, blue()));

		assert_eq!(
			c1.interior_nodes.len(),
			c2.interior_nodes.len(),
			"idempotent stamp should produce same tree"
		);
		assert_eq!(c1.leaf_nodes.len(), c2.leaf_nodes.len());
	}

	// 16. Two materials are both retrievable via get_voxel.
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		edits.push(Path::from_coords([0, 0, 0], 4), blue());
		edits.push(Path::from_coords([255, 255, 255], 4), red());
		let chunk = Chunk::new().apply_edits(edits);
		assert_eq!(get_voxel(&chunk, [0, 0, 0]), Some(blue()));
		assert_eq!(get_voxel(&chunk, [255, 255, 255]), Some(red()));
		assert_eq!(get_voxel(&chunk, [128, 128, 128]), None);
		assert_chunk_valid(&chunk);
	}

	// 17. Filling a 4³ leaf region with one material collapses properly.
	// During editing the region collapses to a Filled slot (no leaf nodes in the wide
	// tree).  After compression, childless interior nodes are demoted to leaf nodes,
	// so the compressed chunk may have exactly one leaf node.  What must NOT happen
	// is the chunk being treated as empty or producing an invalid tree.
	{
		let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
		// Fill a 4³ leaf region (positions [0..4, 0..4, 0..4]) with blue.
		for x in 0u8..4 {
			for y in 0u8..4 {
				for z in 0u8..4 {
					edits.push(Path::from_coords([x, y, z], 4), blue());
				}
			}
		}
		let chunk = Chunk::new().apply_edits(edits);
		assert!(!chunk.is_empty(), "uniform 4³ region should not be empty");
		assert_chunk_valid(&chunk);
	}

	println!("all tests passed");
}

fn time_one<F: FnOnce()>(f: F) -> std::time::Duration {
	let t = Instant::now();
	f();
	t.elapsed()
}

fn fmt_duration(d: std::time::Duration) -> String {
	let ns = d.as_nanos();
	if ns < 1_000 {
		format!("{ns}ns")
	} else if ns < 1_000_000 {
		format!("{:.1}µs", ns as f64 / 1_000.0)
	} else if ns < 1_000_000_000 {
		format!("{:.2}ms", ns as f64 / 1_000_000.0)
	} else {
		format!("{:.3}s", ns as f64 / 1_000_000_000.0)
	}
}

fn green() -> Material {
	Material::from_rgb_pbr_id([60, 120, 40], 0)
}
fn brown() -> Material {
	Material::from_rgb_pbr_id([100, 70, 40], 0)
}

// Simple value noise: hash two ints into a float in [-1, 1].
fn hash2(x: i32, z: i32) -> f32 {
	let h = (x.wrapping_mul(374761393) ^ z.wrapping_mul(668265263)).wrapping_mul(1274126177);
	(h & 0xFFFF) as f32 / 32767.5 - 1.0
}

// Bilinear value noise over a [0,16) grid with grid spacing 4.
fn noise2(x: u8, z: u8) -> f32 {
	let gx = (x / 4) as i32;
	let gz = (z / 4) as i32;
	let tx = (x % 4) as f32 / 4.0;
	let tz = (z % 4) as f32 / 4.0;
	// Smoothstep.
	let sx = tx * tx * (3.0 - 2.0 * tx);
	let sz = tz * tz * (3.0 - 2.0 * tz);
	let v00 = hash2(gx, gz);
	let v10 = hash2(gx + 1, gz);
	let v01 = hash2(gx, gz + 1);
	let v11 = hash2(gx + 1, gz + 1);
	let top = v00 + sx * (v10 - v00);
	let bot = v01 + sx * (v11 - v01);
	top + sz * (bot - top)
}

// Returns the surface y for a given column. Range [6, 10] roughly.
fn terrain_height(x: u8, z: u8) -> u8 {
	let h = 8.0 + 2.0 * noise2(x, z);
	h.round().clamp(0.0, 15.0) as u8
}

fn make_minecraft_chunk() -> Chunk {
	let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
	for bx in 0u8..16 {
		for bz in 0u8..16 {
			let surface = terrain_height(bx, bz); // block-space y, 0..15
			for by in 0u8..surface {
				// Each 16x16x16 block aligns to a depth-2 node.
				edits.push(Path::from_coords([bx * 16, by * 16, bz * 16], 2), brown());
			}
			edits.push(
				Path::from_coords([bx * 16, surface * 16, bz * 16], 2),
				green(),
			);
		}
	}
	Chunk::new().apply_edits(edits)
}

fn make_flat_minecraft_chunk() -> Chunk {
	let mut edits = ChunkEdits::new(WorldPos::from([0; 3]));
	for bx in 0u8..16 {
		for bz in 0u8..16 {
			for by in 0u8..8 {
				edits.push(Path::from_coords([bx * 16, by * 16, bz * 16], 2), brown());
			}
			edits.push(Path::from_coords([bx * 16, 8 * 16, bz * 16], 2), green());
		}
	}
	Chunk::new().apply_edits(edits)
}

fn make_grid_spheres() -> Chunk {
	let mut chunk = Chunk::new();
	for x in 0..16i32 {
		for y in 0..16i32 {
			for z in 0..16i32 {
				let center = [
					x as f32 * 16.0 + 8.0,
					y as f32 * 16.0 + 8.0,
					z as f32 * 16.0 + 8.0,
				];
				chunk = chunk.apply_edits(sphere_edits_center(7.0, center, blue()));
			}
		}
	}
	chunk
}

fn sphere_edits_center(radius: f32, center: [f32; 3], material: Material) -> ChunkEdits {
	use ray_vox::generate::LazyEdit;
	use ray_vox::world::clipmap::{Clipmap, ClipmapChunkId};
	let clipmap = Clipmap {
		occupancy: [[0u32; 16]; 11],
		origin: WorldPos::from([0; 3]),
		pending_remap: Vec::new(),
		pending_origin: WorldPos::from([0; 3]),
	};
	Sphere {
		center,
		radius,
		material,
	}
	.generate(ClipmapChunkId::new(10, 4, 4, 4), &clipmap)
}

fn main() {
	run_tests();

	let radii: Vec<f32> = (0..=7).map(|i| (1u32 << i) as f32).collect();

	println!("\nsphere edit packet depth distribution:");
	for &r in &radii {
		print_edit_depths(&format!("r={r:3}"), &sphere_edits(r, blue()));
	}

	println!("\nsphere stats:");
	for &r in &radii {
		let chunk = Chunk::new().apply_edits(sphere_edits(r, blue()));
		print_chunk_stats(&format!("r={r:3}"), &chunk);
	}

	println!("\napply_sphere_fresh:");
	for &r in &radii {
		let edits = sphere_edits(r, blue());
		let d = time_one(|| {
			let chunk = Chunk::new().apply_edits(edits.clone());
			std::hint::black_box(chunk);
		});
		println!("  r={r:3}: {}", fmt_duration(d));
	}

	let full_chunk = Chunk::new().apply_edits(sphere_edits(512.0, blue()));

	println!("\napply_sphere_onto_full:");
	for &r in &radii {
		let edits = sphere_edits(r, blue());
		let d = time_one(|| {
			let chunk = full_chunk.clone().apply_edits(edits.clone());
			std::hint::black_box(chunk);
		});
		println!("  r={r:3}: {}", fmt_duration(d));
	}

	println!("\nr=512 sphere (entire chunk collapses to root leaf):");
	print_chunk_stats("stats", &full_chunk);

	println!("\nsingle r=7 sphere:");
	{
		let chunk = Chunk::new().apply_edits(sphere_edits(7.0, blue()));
		print_chunk_stats("stats", &chunk);
	}

	println!("\ngrid spheres (4096 × r=7, aligned for DAG dedup):");
	print_chunk_stats("stats", &make_grid_spheres());

	println!("\nminecraft-like flat world (16×16 columns, 2-material, 8 brown + 1 green):");
	let flat = make_flat_minecraft_chunk();
	debug_chunk("flat", &flat);
	print_chunk_stats("stats", &flat);

	println!("\nminecraft-like terrain (16×16 columns, 2-material, perlin ±2):");
	print_chunk_stats("stats", &make_minecraft_chunk());
}

fn debug_chunk(label: &str, chunk: &Chunk) {
	println!(
		"  {label}: {} interior nodes, {} leaf nodes, {} mat entries",
		chunk.interior_nodes.len(),
		chunk.leaf_nodes.len(),
		chunk.materials.lut.len(),
	);
}
