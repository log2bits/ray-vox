// Fragment shader that ray traces a fixed 3D grid of chunks. Each chunk is a
// sparse 64-tree of voxels stored in the same layout Chunk::write_bytes
// produces on the CPU. A ray steps through the world chunk by chunk via the
// directory buffer; inside a populated chunk it steps voxel by voxel and asks
// the tree what material lives at each voxel.

struct Uniforms {
	camera_eye: vec4<f32>,
	camera_right: vec4<f32>,
	camera_up: vec4<f32>,
	camera_forward: vec4<f32>,
	world_origin: vec4<f32>,
	chunk_grid_dim: vec4<u32>,
	resolution: vec2<f32>,
	fov_scale: f32,
	aspect: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> chunk_directory: array<u32>;
@group(0) @binding(2) var<storage, read> chunk_data: array<u32>;

const CHUNK_SIZE_F: f32 = 256.0;
const CHUNK_SIZE_I: i32 = 256;
const EMPTY_CHUNK_SENTINEL: u32 = 0xFFFFFFFFu;
const MAX_WORLD_STEPS: u32 = 256u;
const MAX_CHUNK_STEPS: u32 = 512u;
const RAY_EPSILON: f32 = 1e-4;
const SUN_DIRECTION: vec3<f32> = vec3<f32>(0.4, 0.9, 0.3);
const SKY_TOP: vec3<f32> = vec3<f32>(0.45, 0.65, 0.95);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.85, 0.88, 0.90);

// A chunk blob starts with:
//   [0]       interior_count
//   [1]       leaf_count
//   [2..]     interior nodes (6 u32 each)
//   [..]      leaf nodes (3 u32 each)
//   [..]      palette_count (1 u32)
//   [..]      palette entries (palette_count u32)
//   [..]      packed indices header: len, bits, word_count
//   [..]      packed word storage
// read_chunk_header walks the four counts and returns starting word offsets
// for each region so downstream reads can jump straight in.
struct ChunkHeader {
	interior_base: u32,
	leaf_base: u32,
	palette_base: u32,
	indices_base: u32,
	indices_bits: u32,
	interior_count: u32,
	leaf_count: u32,
	palette_count: u32,
};

fn read_chunk_header(chunk_offset: u32) -> ChunkHeader {
	let interior_count = chunk_data[chunk_offset];
	let leaf_count = chunk_data[chunk_offset + 1u];
	let interior_base = chunk_offset + 2u;
	let leaf_base = interior_base + interior_count * 6u;
	let palette_header = leaf_base + leaf_count * 3u;
	let palette_count = chunk_data[palette_header];
	let palette_base = palette_header + 1u;
	let indices_header = palette_base + palette_count;
	let indices_bits = chunk_data[indices_header + 1u];
	let indices_base = indices_header + 3u;
	return ChunkHeader(
		interior_base, leaf_base, palette_base, indices_base,
		indices_bits, interior_count, leaf_count, palette_count,
	);
}

fn palette_lookup(header: ChunkHeader, palette_index: u32) -> u32 {
	return chunk_data[header.palette_base + palette_index];
}

// Extract the palette index at a bit-packed position, then resolve it through
// the palette LUT into a full voxel value.
fn read_material_at_index(header: ChunkHeader, packed_position: u32) -> u32 {
	let bits = header.indices_bits;
	let bit_pos = packed_position * bits;
	let word_index = bit_pos / 32u;
	let bit_offset = bit_pos % 32u;
	let primary_word = chunk_data[header.indices_base + word_index];
	var raw = primary_word >> bit_offset;
	if (bit_offset + bits > 32u) {
		let secondary_word = chunk_data[header.indices_base + word_index + 1u];
		raw = raw | (secondary_word << (32u - bit_offset));
	}
	var mask: u32;
	if (bits == 32u) {
		mask = 0xFFFFFFFFu;
	} else {
		mask = (1u << bits) - 1u;
	}
	let palette_index = raw & mask;
	return palette_lookup(header, palette_index);
}

struct InteriorNode {
	has_child_lo: u32,
	has_child_hi: u32,
	is_leaf_lo: u32,
	is_leaf_hi: u32,
	node_offsets: u32,
	material_offset: u32,
};

fn read_interior(header: ChunkHeader, index: u32) -> InteriorNode {
	let base = header.interior_base + index * 6u;
	return InteriorNode(
		chunk_data[base],
		chunk_data[base + 1u],
		chunk_data[base + 2u],
		chunk_data[base + 3u],
		chunk_data[base + 4u],
		chunk_data[base + 5u],
	);
}

struct LeafNode {
	occupancy_lo: u32,
	occupancy_hi: u32,
	material_offset: u32,
};

fn read_leaf(header: ChunkHeader, index: u32) -> LeafNode {
	let base = header.leaf_base + index * 3u;
	return LeafNode(
		chunk_data[base],
		chunk_data[base + 1u],
		chunk_data[base + 2u],
	);
}

// Test whether bit `slot` (0..63) is set in a 64-bit mask stored as two u32.
fn bit64_is_set(low: u32, high: u32, slot: u32) -> bool {
	if (slot < 32u) {
		return (low & (1u << slot)) != 0u;
	}
	return (high & (1u << (slot - 32u))) != 0u;
}

// Count set bits below `slot` in a 64-bit mask stored as two u32. Explicit
// guards keep us clear of the "1u << 32" undefined-behavior corner.
fn popcount_below_64(low: u32, high: u32, slot: u32) -> u32 {
	if (slot == 0u) { return 0u; }
	if (slot < 32u) {
		let mask = (1u << slot) - 1u;
		return countOneBits(low & mask);
	}
	if (slot == 32u) {
		return countOneBits(low);
	}
	let high_bits = slot - 32u;
	let high_mask = (1u << high_bits) - 1u;
	return countOneBits(low) + countOneBits(high & high_mask);
}

// State of a slot inside an interior node:
//   0 = empty, 1 = filled (material stored inline), 2 = interior child,
//   3 = leaf child.
fn interior_slot_state(node: InteriorNode, slot: u32) -> u32 {
	let has_child = bit64_is_set(node.has_child_lo, node.has_child_hi, slot);
	let is_leaf = bit64_is_set(node.is_leaf_lo, node.is_leaf_hi, slot);
	var state: u32 = 0u;
	if (has_child) { state = state + 2u; }
	if (is_leaf)   { state = state + 1u; }
	return state;
}

fn interior_filled_material(header: ChunkHeader, node: InteriorNode, slot: u32) -> u32 {
	let filled_lo = (~node.has_child_lo) & node.is_leaf_lo;
	let filled_hi = (~node.has_child_hi) & node.is_leaf_hi;
	let rank = popcount_below_64(filled_lo, filled_hi, slot);
	return read_material_at_index(header, node.material_offset + rank);
}

fn interior_child_at_slot_interior(node: InteriorNode, slot: u32) -> u32 {
	let interiors_lo = node.has_child_lo & (~node.is_leaf_lo);
	let interiors_hi = node.has_child_hi & (~node.is_leaf_hi);
	let rank = popcount_below_64(interiors_lo, interiors_hi, slot);
	let interior_ptr = node.node_offsets & 0x1FFFu;
	return interior_ptr + rank;
}

fn interior_child_at_slot_leaf(node: InteriorNode, slot: u32) -> u32 {
	let leaves_lo = node.has_child_lo & node.is_leaf_lo;
	let leaves_hi = node.has_child_hi & node.is_leaf_hi;
	let rank = popcount_below_64(leaves_lo, leaves_hi, slot);
	let leaf_ptr = (node.node_offsets >> 13u) & 0x7FFFFu;
	return leaf_ptr + rank;
}

fn leaf_voxel_material(header: ChunkHeader, leaf: LeafNode, slot: u32) -> u32 {
	if (!bit64_is_set(leaf.occupancy_lo, leaf.occupancy_hi, slot)) {
		return 0u;
	}
	let rank = popcount_below_64(leaf.occupancy_lo, leaf.occupancy_hi, slot);
	return read_material_at_index(header, leaf.material_offset + rank);
}

// Pack the two bits per axis at tree depth `depth` into a 6-bit slot index.
// The chunk is 4^4 = 256 voxels per side, so depth 0 uses bits [7..6],
// depth 1 uses [5..4], depth 2 uses [3..2], depth 3 uses [1..0].
fn path_slot(voxel: vec3<u32>, depth: u32) -> u32 {
	let shift = (3u - depth) * 2u;
	let x = (voxel.x >> shift) & 3u;
	let y = (voxel.y >> shift) & 3u;
	let z = (voxel.z >> shift) & 3u;
	return (x << 4u) | (y << 2u) | z;
}

// Look up the material at a chunk-local voxel coord (each axis in 0..255).
// Returns 0 for air. Walks the tree from the root, taking at most three
// interior descents before hitting a leaf, filled cell, or air.
fn chunk_voxel_at(header: ChunkHeader, voxel: vec3<u32>) -> u32 {
	if (header.interior_count == 0u && header.leaf_count == 0u) {
		if (header.palette_count == 1u) {
			return palette_lookup(header, 0u);
		}
		return 0u;
	}
	if (header.interior_count == 0u) {
		let root_leaf = read_leaf(header, 0u);
		return leaf_voxel_material(header, root_leaf, path_slot(voxel, 0u));
	}
	var node_index = header.interior_count - 1u;
	for (var depth: u32 = 0u; depth < 3u; depth = depth + 1u) {
		let node = read_interior(header, node_index);
		let slot = path_slot(voxel, depth);
		let state = interior_slot_state(node, slot);
		if (state == 0u) {
			return 0u;
		}
		if (state == 1u) {
			return interior_filled_material(header, node, slot);
		}
		if (state == 2u) {
			node_index = interior_child_at_slot_interior(node, slot);
			continue;
		}
		let leaf_index = interior_child_at_slot_leaf(node, slot);
		let leaf = read_leaf(header, leaf_index);
		return leaf_voxel_material(header, leaf, path_slot(voxel, depth + 1u));
	}
	return 0u;
}

struct AabbHit {
	hit: bool,
	t_near: f32,
	t_far: f32,
};

fn ray_aabb_intersect(origin: vec3<f32>, dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> AabbHit {
	let inv_dir = 1.0 / dir;
	let t_lo = (box_min - origin) * inv_dir;
	let t_hi = (box_max - origin) * inv_dir;
	let t_min_per_axis = min(t_lo, t_hi);
	let t_max_per_axis = max(t_lo, t_hi);
	let t_near = max(max(t_min_per_axis.x, t_min_per_axis.y), t_min_per_axis.z);
	let t_far = min(min(t_max_per_axis.x, t_max_per_axis.y), t_max_per_axis.z);
	return AabbHit(t_near <= t_far && t_far > 0.0, t_near, t_far);
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
	let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
	return mix(SKY_HORIZON, SKY_TOP, t);
}

// Unpack the 24-bit linear RGB tint from a voxel u32 (bits 31..8).
fn material_rgb(material: u32) -> vec3<f32> {
	let r = f32((material >> 24u) & 0xFFu) / 255.0;
	let g = f32((material >> 16u) & 0xFFu) / 255.0;
	let b = f32((material >>  8u) & 0xFFu) / 255.0;
	return vec3<f32>(r, g, b);
}

fn shade(albedo: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
	let sun = normalize(SUN_DIRECTION);
	let ambient = 0.35;
	let diffuse = max(dot(normal, sun), 0.0) * 0.7;
	return albedo * (ambient + diffuse);
}

struct VoxelHit {
	hit: bool,
	material: u32,
	normal: vec3<f32>,
};

fn axis_to_normal(axis: u32, step: vec3<i32>) -> vec3<f32> {
	if (axis == 0u) { return vec3<f32>(-f32(step.x), 0.0, 0.0); }
	if (axis == 1u) { return vec3<f32>(0.0, -f32(step.y), 0.0); }
	return vec3<f32>(0.0, 0.0, -f32(step.z));
}

// Voxel-resolution DDA inside a single chunk. Starts at origin_local (a
// chunk-local position in world units) and steps one voxel at a time,
// consulting chunk_voxel_at for occupancy. entry_axis is the axis whose face
// the ray crossed to enter this chunk; it seeds the normal for a hit on the
// very first voxel we test.
fn trace_chunk(header: ChunkHeader, origin_local: vec3<f32>, dir: vec3<f32>, entry_axis: u32) -> VoxelHit {
	let start = clamp(origin_local, vec3<f32>(0.0), vec3<f32>(CHUNK_SIZE_F - RAY_EPSILON));
	var voxel = vec3<i32>(floor(start));
	let step = vec3<i32>(sign(dir));
	let next_boundary = vec3<f32>(voxel + max(step, vec3<i32>(0)));
	var t_next = (next_boundary - origin_local) / dir;
	let t_delta = abs(1.0 / dir);
	var last_axis = entry_axis;

	for (var step_index: u32 = 0u; step_index < MAX_CHUNK_STEPS; step_index = step_index + 1u) {
		if (voxel.x < 0 || voxel.x >= CHUNK_SIZE_I
			|| voxel.y < 0 || voxel.y >= CHUNK_SIZE_I
			|| voxel.z < 0 || voxel.z >= CHUNK_SIZE_I) {
			break;
		}
		let material = chunk_voxel_at(header, vec3<u32>(voxel));
		if (material != 0u) {
			return VoxelHit(true, material, axis_to_normal(last_axis, step));
		}
		if (t_next.x < t_next.y && t_next.x < t_next.z) {
			voxel.x = voxel.x + step.x;
			t_next.x = t_next.x + t_delta.x;
			last_axis = 0u;
		} else if (t_next.y < t_next.z) {
			voxel.y = voxel.y + step.y;
			t_next.y = t_next.y + t_delta.y;
			last_axis = 1u;
		} else {
			voxel.z = voxel.z + step.z;
			t_next.z = t_next.z + t_delta.z;
			last_axis = 2u;
		}
	}
	return VoxelHit(false, 0u, vec3<f32>(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 3.0, -1.0),
		vec2<f32>(-1.0,  3.0),
	);
	return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_position: vec4<f32>) -> @location(0) vec4<f32> {
	// Ray from camera through this pixel.
	let ndc_xy = (frag_position.xy / uniforms.resolution) * 2.0 - vec2<f32>(1.0);
	let ray_dir = normalize(
		uniforms.camera_forward.xyz
		+ uniforms.camera_right.xyz * ndc_xy.x * uniforms.aspect * uniforms.fov_scale
		- uniforms.camera_up.xyz * ndc_xy.y * uniforms.fov_scale
	);
	let ray_origin = uniforms.camera_eye.xyz;

	// Nudge zero-magnitude components so 1/dir stays finite. DDA is happy
	// with any non-zero direction.
	var safe_dir = ray_dir;
	if (abs(safe_dir.x) < RAY_EPSILON) { safe_dir.x = RAY_EPSILON; }
	if (abs(safe_dir.y) < RAY_EPSILON) { safe_dir.y = RAY_EPSILON; }
	if (abs(safe_dir.z) < RAY_EPSILON) { safe_dir.z = RAY_EPSILON; }

	let grid_dim_f = vec3<f32>(
		f32(uniforms.chunk_grid_dim.x),
		f32(uniforms.chunk_grid_dim.y),
		f32(uniforms.chunk_grid_dim.z),
	);
	let grid_min = uniforms.world_origin.xyz;
	let grid_max = grid_min + grid_dim_f * CHUNK_SIZE_F;

	let world_hit = ray_aabb_intersect(ray_origin, safe_dir, grid_min, grid_max);
	if (!world_hit.hit) {
		return vec4<f32>(sky_color(ray_dir), 1.0);
	}

	// Which face of the world AABB the ray entered through, and whether we
	// started inside (t_near <= 0).
	let entry_t_world = max(world_hit.t_near, 0.0);
	let entry_pos_world = ray_origin + safe_dir * (entry_t_world + RAY_EPSILON);

	var chunk_pos = vec3<i32>(floor((entry_pos_world - grid_min) / CHUNK_SIZE_F));
	let chunk_step = vec3<i32>(sign(safe_dir));
	let chunk_boundary = grid_min + vec3<f32>(chunk_pos + max(chunk_step, vec3<i32>(0))) * CHUNK_SIZE_F;
	var chunk_t_next = (chunk_boundary - ray_origin) / safe_dir;
	let chunk_t_delta = CHUNK_SIZE_F * abs(1.0 / safe_dir);

	// Deduce the entry face by finding the axis whose t_lo equals t_near.
	let t_lo = (grid_min - ray_origin) / safe_dir;
	let t_hi = (grid_max - ray_origin) / safe_dir;
	let t_min_per_axis = min(t_lo, t_hi);
	var world_entry_axis: u32 = 0u;
	if (t_min_per_axis.y > t_min_per_axis.x) { world_entry_axis = 1u; }
	if (t_min_per_axis.z > t_min_per_axis[world_entry_axis]) { world_entry_axis = 2u; }

	var current_entry_t = entry_t_world;
	var current_entry_axis = world_entry_axis;

	let grid_ix = i32(uniforms.chunk_grid_dim.x);
	let grid_iy = i32(uniforms.chunk_grid_dim.y);
	let grid_iz = i32(uniforms.chunk_grid_dim.z);

	for (var step_index: u32 = 0u; step_index < MAX_WORLD_STEPS; step_index = step_index + 1u) {
		if (chunk_pos.x < 0 || chunk_pos.x >= grid_ix
			|| chunk_pos.y < 0 || chunk_pos.y >= grid_iy
			|| chunk_pos.z < 0 || chunk_pos.z >= grid_iz) {
			break;
		}
		let flat_index = u32(chunk_pos.x)
			+ u32(chunk_pos.y) * uniforms.chunk_grid_dim.x
			+ u32(chunk_pos.z) * uniforms.chunk_grid_dim.x * uniforms.chunk_grid_dim.y;
		let chunk_offset = chunk_directory[flat_index];
		if (chunk_offset != EMPTY_CHUNK_SENTINEL) {
			let chunk_origin_world = grid_min + vec3<f32>(chunk_pos) * CHUNK_SIZE_F;
			let local_entry = ray_origin + safe_dir * (current_entry_t + RAY_EPSILON) - chunk_origin_world;
			let header = read_chunk_header(chunk_offset);
			let hit = trace_chunk(header, local_entry, safe_dir, current_entry_axis);
			if (hit.hit) {
				let albedo = material_rgb(hit.material);
				return vec4<f32>(shade(albedo, hit.normal), 1.0);
			}
		}
		// Step to the next chunk along the smallest t_next axis.
		if (chunk_t_next.x < chunk_t_next.y && chunk_t_next.x < chunk_t_next.z) {
			current_entry_t = chunk_t_next.x;
			chunk_pos.x = chunk_pos.x + chunk_step.x;
			chunk_t_next.x = chunk_t_next.x + chunk_t_delta.x;
			current_entry_axis = 0u;
		} else if (chunk_t_next.y < chunk_t_next.z) {
			current_entry_t = chunk_t_next.y;
			chunk_pos.y = chunk_pos.y + chunk_step.y;
			chunk_t_next.y = chunk_t_next.y + chunk_t_delta.y;
			current_entry_axis = 1u;
		} else {
			current_entry_t = chunk_t_next.z;
			chunk_pos.z = chunk_pos.z + chunk_step.z;
			chunk_t_next.z = chunk_t_next.z + chunk_t_delta.z;
			current_entry_axis = 2u;
		}
	}
	return vec4<f32>(sky_color(ray_dir), 1.0);
}
