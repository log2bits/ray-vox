// Ray traces a fixed 3D grid of chunks. Each chunk is a sparse 64-tree in
// the on-disk Chunk::write_bytes layout. Rays step chunk to chunk via the
// directory buffer and refine inside a chunk with a tree-DDA.

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
    render_mode: u32,
    // WGSL rounds struct size up to the max field alignment (16), so
    // render_mode at offset 112 pads out to 128 automatically. The Rust
    // Uniforms struct matches with a trailing [u32; 3].
};

// Per-pixel memory-read counter. Reset at the top of fs_main and bumped
// inside every loop that reads chunk data. Shown as a heatmap when
// render_mode == 1.
var<private> g_probe_count: u32 = 0u;

// Saturation ceiling for the heatmap. Tuned so castle-scale grazing rays
// approach white without pinning the whole screen at the top of the palette.
const HEATMAP_MAX_PROBES: f32 = 150.0;

// Five-stop palette: black, deep purple, magenta, orange, near-white.
fn heatmap_color(intensity: f32) -> vec3<f32> {
    let stops = 4.0;
    let scaled = clamp(intensity, 0.0, 1.0) * stops;
    let black = vec3<f32>(0.0, 0.0, 0.0);
    let deep_purple = vec3<f32>(0.30, 0.05, 0.55);
    let magenta = vec3<f32>(0.95, 0.15, 0.45);
    let orange = vec3<f32>(1.00, 0.60, 0.10);
    let near_white = vec3<f32>(1.00, 1.00, 0.85);
    if scaled < 1.0 {
        return mix(black, deep_purple, scaled);
    }
    if scaled < 2.0 {
        return mix(deep_purple, magenta, scaled - 1.0);
    }
    if scaled < 3.0 {
        return mix(magenta, orange, scaled - 2.0);
    }
    return mix(orange, near_white, clamp(scaled - 3.0, 0.0, 1.0));
}

fn probe_pixel() -> vec4<f32> {
    let intensity = f32(g_probe_count) / HEATMAP_MAX_PROBES;
    return vec4<f32>(heatmap_color(intensity), 1.0);
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> chunk_directory: array<u32>;
@group(0) @binding(2) var<storage, read> chunk_data: array<u32>;

const CHUNK_SIZE_F: f32 = 256.0;
const CHUNK_SIZE_I: i32 = 256;
const EMPTY_CHUNK_SENTINEL: u32 = 0xFFFFFFFFu;
const MAX_WORLD_STEPS: u32 = 256u;
// Amanatides-Woo DDA needs up to 3N axis crossings through an NxNxN grid,
// so a 256^3 chunk corner-to-corner tops out at 768 steps. Rounded up so
// grazing rays don't fall short and leak the sky through populated chunks.
const MAX_CHUNK_STEPS: u32 = 1024u;
const RAY_EPSILON: f32 = 1e-4;
const SUN_DIRECTION: vec3<f32> = vec3<f32>(0.4, 0.9, 0.3);
const SKY_TOP: vec3<f32> = vec3<f32>(0.45, 0.65, 0.95);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.85, 0.88, 0.90);

// Chunk blob layout: interior_count, leaf_count, interior_nodes (6 u32 each),
// leaf_nodes (3 u32 each), palette_count, palette entries, packed-indices
// header (len, bits, word_count), and finally the packed words.
// read_chunk_header walks the counts once and returns starting word offsets.
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

// Read the palette index at a bit-packed position and resolve it into a
// full voxel value via the palette LUT.
fn read_material_at_index(header: ChunkHeader, packed_position: u32) -> u32 {
    let bits = header.indices_bits;
    let bit_pos = packed_position * bits;
    let word_index = bit_pos / 32u;
    let bit_offset = bit_pos % 32u;
    let primary_word = chunk_data[header.indices_base + word_index];
    var raw = primary_word >> bit_offset;
    if bit_offset + bits > 32u {
        let secondary_word = chunk_data[header.indices_base + word_index + 1u];
        raw = raw | (secondary_word << (32u - bit_offset));
    }
    var mask: u32;
    if bits == 32u {
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

// True if the slot bit (0..63) is set in a 64-bit mask split across two u32.
fn bit64_is_set(low: u32, high: u32, slot: u32) -> bool {
    if slot < 32u {
        return (low & (1u << slot)) != 0u;
    }
    return (high & (1u << (slot - 32u))) != 0u;
}

// Count set bits below slot in a 64-bit mask split across two u32. Explicit
// guards keep us clear of the 1u << 32 undefined-behavior corner.
fn popcount_below_64(low: u32, high: u32, slot: u32) -> u32 {
    if slot == 0u { return 0u; }
    if slot < 32u {
        let mask = (1u << slot) - 1u;
        return countOneBits(low & mask);
    }
    if slot == 32u {
        return countOneBits(low);
    }
    let high_bits = slot - 32u;
    let high_mask = (1u << high_bits) - 1u;
    return countOneBits(low) + countOneBits(high & high_mask);
}

// Slot state inside an interior node: 0 empty, 1 filled (inline material),
// 2 interior child, 3 leaf child.
fn interior_slot_state(node: InteriorNode, slot: u32) -> u32 {
    let has_child = bit64_is_set(node.has_child_lo, node.has_child_hi, slot);
    let is_leaf = bit64_is_set(node.is_leaf_lo, node.is_leaf_hi, slot);
    var state: u32 = 0u;
    if has_child { state = state + 2u; }
    if is_leaf { state = state + 1u; }
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
    if !bit64_is_set(leaf.occupancy_lo, leaf.occupancy_hi, slot) {
        return 0u;
    }
    let rank = popcount_below_64(leaf.occupancy_lo, leaf.occupancy_hi, slot);
    return read_material_at_index(header, leaf.material_offset + rank);
}

// Pack the depth-d nibble of each voxel axis into a 6-bit slot index. The
// chunk is 4^4 voxels per side: depth 0 uses bits 6..7, 1 uses 4..5, 2 uses
// 2..3, 3 uses 0..1.
fn path_slot(voxel: vec3<u32>, depth: u32) -> u32 {
    let shift = (3u - depth) * 2u;
    let x = (voxel.x >> shift) & 3u;
    let y = (voxel.y >> shift) & 3u;
    let z = (voxel.z >> shift) & 3u;
    return (x << 4u) | (y << 2u) | z;
}

// chunk_voxel_at was removed; trace_chunk below replaces it with a
// stack-based tree DDA that skips empty subtrees and reuses ancestors.

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
    let b = f32((material >> 8u) & 0xFFu) / 255.0;
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
    if axis == 0u { return vec3<f32>(-f32(step.x), 0.0, 0.0); }
    if axis == 1u { return vec3<f32>(0.0, -f32(step.y), 0.0); }
    return vec3<f32>(0.0, 0.0, -f32(step.z));
}

// Snap the entry-face coord to a safe inset. Absorbs precision loss from
// the world-to-local subtraction so floor never lands us in the wrong cell.
fn snap_chunk_entry(origin_local: vec3<f32>, dir: vec3<f32>, entry_axis: u32, entered_from_outside: bool) -> vec3<f32> {
    var pos = clamp(origin_local, vec3<f32>(0.0), vec3<f32>(CHUNK_SIZE_F));
    if entered_from_outside {
        let face_inset: f32 = 1.0 / 16.0;
        if entry_axis == 0u {
            if dir.x > 0.0 { pos.x = face_inset; }
			else { pos.x = CHUNK_SIZE_F - face_inset; }
        } else if entry_axis == 1u {
            if dir.y > 0.0 { pos.y = face_inset; }
			else { pos.y = CHUNK_SIZE_F - face_inset; }
        } else {
            if dir.z > 0.0 { pos.z = face_inset; }
			else { pos.z = CHUNK_SIZE_F - face_inset; }
        }
    }
    return pos;
}

// Rare case: chunk whose root is a leaf. Its 64 slots each cover a 64-voxel
// sub-cell, so DDA at 64-voxel granularity finds any hit in a few steps.
fn trace_root_leaf_chunk(
    header: ChunkHeader,
    origin_local: vec3<f32>,
    dir: vec3<f32>,
    entry_axis: u32,
    entered_from_outside: bool,
) -> VoxelHit {
    var pos = snap_chunk_entry(origin_local, dir, entry_axis, entered_from_outside);
    let step_sign = vec3<i32>(sign(dir));
    let inv_dir = 1.0 / dir;
    var last_axis = entry_axis;
    let leaf = read_leaf(header, 0u);
    g_probe_count = g_probe_count + 1u;

    for (var step_index: u32 = 0u; step_index < MAX_CHUNK_STEPS; step_index = step_index + 1u) {
        let voxel_i = vec3<i32>(floor(pos));
        if voxel_i.x < 0 || voxel_i.x >= CHUNK_SIZE_I
			|| voxel_i.y < 0 || voxel_i.y >= CHUNK_SIZE_I
			|| voxel_i.z < 0 || voxel_i.z >= CHUNK_SIZE_I {
            break;
        }
        g_probe_count = g_probe_count + 1u;
        let slot = path_slot(vec3<u32>(voxel_i), 0u);
        if bit64_is_set(leaf.occupancy_lo, leaf.occupancy_hi, slot) {
            let material = leaf_voxel_material(header, leaf, slot);
            return VoxelHit(true, material, axis_to_normal(last_axis, step_sign));
        }
        let cell_mask = ~63i;
        let cell_origin_i = vec3<i32>(voxel_i.x & cell_mask, voxel_i.y & cell_mask, voxel_i.z & cell_mask);
        let cell_origin_f = vec3<f32>(cell_origin_i);
        let side_pos = cell_origin_f + vec3<f32>(max(step_sign, vec3<i32>(0)) * 64i);
        let side_dist = (side_pos - pos) * inv_dir;
        let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);
        if side_dist.x <= side_dist.y && side_dist.x <= side_dist.z { last_axis = 0u; }
		else if side_dist.y <= side_dist.z { last_axis = 1u; }
		else { last_axis = 2u; }
        var neighbour_origin_i = cell_origin_i;
        if last_axis == 0u { neighbour_origin_i.x = cell_origin_i.x + step_sign.x * 64i; }
		else if last_axis == 1u { neighbour_origin_i.y = cell_origin_i.y + step_sign.y * 64i; }
		else { neighbour_origin_i.z = cell_origin_i.z + step_sign.z * 64i; }
        let neighbour_min = vec3<f32>(neighbour_origin_i);
        let neighbour_max = neighbour_min + vec3<f32>(64.0);
        let inset: f32 = 1.0 / 1024.0;
        pos = clamp(pos + dir * tmax, neighbour_min + vec3<f32>(inset), neighbour_max - vec3<f32>(inset));
    }
    return VoxelHit(false, 0u, vec3<f32>(0.0));
}

// Tree-DDA inside one chunk. The ray descends only when it enters a new
// interior slot, caches the active depth-1 interior on the stack, and skips
// whole empty sub-cells in one DDA step at whatever depth we're at.
//
// Slot widths in voxels: depth 0 = 64, depth 1 = 16, depth 2 = 4, depth 3 = 1.
//
// The stack only needs the depth-1 interior index: the root is always at
// header.interior_count - 1, and the depth-2 node reloads on descent. 13
// bits (max interior_count is 8192) fit trivially in a u32.
fn trace_chunk(
    header: ChunkHeader,
    origin_local: vec3<f32>,
    dir: vec3<f32>,
    entry_axis: u32,
    entered_from_outside: bool,
) -> VoxelHit {
    // Uniform chunk: no nodes, one palette entry, whole chunk is that material.
    if header.interior_count == 0u && header.leaf_count == 0u {
        if header.palette_count == 1u {
            let material = palette_lookup(header, 0u);
            return VoxelHit(true, material, axis_to_normal(entry_axis, vec3<i32>(sign(dir))));
        }
        return VoxelHit(false, 0u, vec3<f32>(0.0));
    }
    if header.interior_count == 0u {
        return trace_root_leaf_chunk(header, origin_local, dir, entry_axis, entered_from_outside);
    }

    var pos = snap_chunk_entry(origin_local, dir, entry_axis, entered_from_outside);
    let step_sign = vec3<i32>(sign(dir));
    let inv_dir = 1.0 / dir;
    var last_axis = entry_axis;

    let root_node_idx = header.interior_count - 1u;
    var stack_d1: u32 = 0u;
    var current_depth: u32 = 0u;
    var current_node_idx: u32 = root_node_idx;
    var current_node = read_interior(header, root_node_idx);
    g_probe_count = g_probe_count + 1u;

    for (var step_index: u32 = 0u; step_index < MAX_CHUNK_STEPS; step_index = step_index + 1u) {
        let voxel_i = vec3<i32>(floor(pos));
        if voxel_i.x < 0 || voxel_i.x >= CHUNK_SIZE_I
			|| voxel_i.y < 0 || voxel_i.y >= CHUNK_SIZE_I
			|| voxel_i.z < 0 || voxel_i.z >= CHUNK_SIZE_I {
            break;
        }
        let voxel_u = vec3<u32>(voxel_i);
        g_probe_count = g_probe_count + 1u;

        // Descend interior children as far as we can. Interior descent tops
        // out at depth 2, because depth 3 is always a leaf.
        var slot: u32 = path_slot(voxel_u, current_depth);
        var state: u32 = interior_slot_state(current_node, slot);
        loop {
            if state != 2u || current_depth >= 2u { break; }
            if current_depth == 1u { stack_d1 = current_node_idx; }
            current_node_idx = interior_child_at_slot_interior(current_node, slot);
            current_node = read_interior(header, current_node_idx);
            g_probe_count = g_probe_count + 1u;
            current_depth = current_depth + 1u;
            slot = path_slot(voxel_u, current_depth);
            state = interior_slot_state(current_node, slot);
        }

        if state == 1u {
            let material = interior_filled_material(header, current_node, slot);
            return VoxelHit(true, material, axis_to_normal(last_axis, step_sign));
        }

        // effective_depth is the depth at which we DDA-step below. Empty
        // interior slots skip a full slot's width; a leaf slot drops one
        // level so we scan its 64 sub-cells at their native size instead.
        var effective_depth = current_depth;
        if state == 3u {
            let leaf_idx = interior_child_at_slot_leaf(current_node, slot);
            let leaf = read_leaf(header, leaf_idx);
            g_probe_count = g_probe_count + 1u;
            let leaf_slot = path_slot(voxel_u, current_depth + 1u);
            if bit64_is_set(leaf.occupancy_lo, leaf.occupancy_hi, leaf_slot) {
                let material = leaf_voxel_material(header, leaf, leaf_slot);
                return VoxelHit(true, material, axis_to_normal(last_axis, step_sign));
            }
            effective_depth = current_depth + 1u;
        }

        // DDA one slot forward at effective_depth. Cells at that depth are
        // 4^(3 - effective_depth) voxels wide (64, 16, 4, or 1).
        let slot_shift = (3u - effective_depth) * 2u;
        let slot_width_i = 1i << slot_shift;
        let cell_mask = ~(slot_width_i - 1);
        let cell_origin_i = vec3<i32>(voxel_i.x & cell_mask, voxel_i.y & cell_mask, voxel_i.z & cell_mask);
        let cell_origin_f = vec3<f32>(cell_origin_i);
        let side_pos = cell_origin_f + vec3<f32>(max(step_sign, vec3<i32>(0)) * slot_width_i);
        let side_dist = (side_pos - pos) * inv_dir;
        let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

        if side_dist.x <= side_dist.y && side_dist.x <= side_dist.z { last_axis = 0u; }
		else if side_dist.y <= side_dist.z { last_axis = 1u; }
		else { last_axis = 2u; }

        // Advance and clamp into the neighbour cell so precision loss can't
        // push us onto the wrong side of the face.
        var neighbour_origin_i = cell_origin_i;
        if last_axis == 0u { neighbour_origin_i.x = cell_origin_i.x + step_sign.x * slot_width_i; }
		else if last_axis == 1u { neighbour_origin_i.y = cell_origin_i.y + step_sign.y * slot_width_i; }
		else { neighbour_origin_i.z = cell_origin_i.z + step_sign.z * slot_width_i; }
        let neighbour_min = vec3<f32>(neighbour_origin_i);
        let neighbour_max = neighbour_min + vec3<f32>(f32(slot_width_i));
        let inset: f32 = 1.0 / 1024.0;
        pos = clamp(pos + dir * tmax, neighbour_min + vec3<f32>(inset), neighbour_max - vec3<f32>(inset));

        // Ascend by the highest bit that changed in the integer voxel coord.
        // Bit position gives us the tree depth of the boundary we crossed:
        // bits 6..7 map to depth 0, 4..5 to 1, 2..3 to 2, 0..1 to 3.
        let new_voxel_i = vec3<i32>(floor(pos));
        let diff_or = u32(voxel_i.x ^ new_voxel_i.x)
			| u32(voxel_i.y ^ new_voxel_i.y)
			| u32(voxel_i.z ^ new_voxel_i.z);
        if diff_or != 0u {
            let highest_bit = 31u - countLeadingZeros(diff_or);
            let crossed_depth = 3u - min(highest_bit >> 1u, 3u);
            if crossed_depth < current_depth {
                current_depth = crossed_depth;
                if current_depth == 0u {
                    current_node_idx = root_node_idx;
                    current_node = read_interior(header, root_node_idx);
                } else {
                    current_node_idx = stack_d1;
                    current_node = read_interior(header, stack_d1);
                }
                g_probe_count = g_probe_count + 1u;
            }
        }
    }
    return VoxelHit(false, 0u, vec3<f32>(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
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

    // Nudge zero-magnitude components so 1/dir stays finite, preserving
    // sign. Naive "if abs < eps: = eps" flips the sign of tiny negative
    // values, which makes chunk_step point the wrong way and sends the DDA
    // down the wrong axis. That specific bug produces circular ringing
    // bands centered on axis-aligned view directions.
    var safe_dir = ray_dir;
    if safe_dir.x >= 0.0 && safe_dir.x < RAY_EPSILON { safe_dir.x = RAY_EPSILON; }
    if safe_dir.x < 0.0 && safe_dir.x > -RAY_EPSILON { safe_dir.x = -RAY_EPSILON; }
    if safe_dir.y >= 0.0 && safe_dir.y < RAY_EPSILON { safe_dir.y = RAY_EPSILON; }
    if safe_dir.y < 0.0 && safe_dir.y > -RAY_EPSILON { safe_dir.y = -RAY_EPSILON; }
    if safe_dir.z >= 0.0 && safe_dir.z < RAY_EPSILON { safe_dir.z = RAY_EPSILON; }
    if safe_dir.z < 0.0 && safe_dir.z > -RAY_EPSILON { safe_dir.z = -RAY_EPSILON; }

    let grid_dim_f = vec3<f32>(
        f32(uniforms.chunk_grid_dim.x),
        f32(uniforms.chunk_grid_dim.y),
        f32(uniforms.chunk_grid_dim.z),
    );
    let grid_min = uniforms.world_origin.xyz;
    let grid_max = grid_min + grid_dim_f * CHUNK_SIZE_F;

    // Reset the per-pixel probe counter used by the heatmap view.
    g_probe_count = 0u;

    let world_hit = ray_aabb_intersect(ray_origin, safe_dir, grid_min, grid_max);
    if !world_hit.hit {
        if uniforms.render_mode == 1u {
            return probe_pixel();
        }
        return vec4<f32>(sky_color(ray_dir), 1.0);
    }

    // Position-based world DDA. Each iteration we recompute the distance
    // to the current chunk's exit face from the ACTUAL current position,
    // not from an accumulated t. After advancing, we clamp the new position
    // strictly inside the neighbour cell's AABB. That clamp is the article's
    // robustness fix and it kills the last of the corner-ringing artifacts.
    let chunk_step = vec3<i32>(sign(safe_dir));
    let inv_safe_dir = 1.0 / safe_dir;
    let grid_ix = i32(uniforms.chunk_grid_dim.x);
    let grid_iy = i32(uniforms.chunk_grid_dim.y);
    let grid_iz = i32(uniforms.chunk_grid_dim.z);

    // Enter the world at t_near (or stay at the camera if it's already
    // inside). This position is our source of truth for the whole traversal.
    let entry_t_world = max(world_hit.t_near, 0.0);
    var pos_world = ray_origin + safe_dir * entry_t_world;

    // Which axis carries the entry face. Written as a straight if-chain so
    // no backend has to deal with dynamic vector indexing.
    let t_lo = (grid_min - ray_origin) * inv_safe_dir;
    let t_hi = (grid_max - ray_origin) * inv_safe_dir;
    let t_min_per_axis = min(t_lo, t_hi);
    var current_entry_axis: u32 = 0u;
    var largest_t_min = t_min_per_axis.x;
    if t_min_per_axis.y > largest_t_min {
        current_entry_axis = 1u;
        largest_t_min = t_min_per_axis.y;
    }
    if t_min_per_axis.z > largest_t_min {
        current_entry_axis = 2u;
    }
    var entered_from_outside = entry_t_world > 0.0;

    var chunk_pos = vec3<i32>(floor((pos_world - grid_min) / CHUNK_SIZE_F));
    // Guard against a corner-case landing exactly on the far face.
    chunk_pos = clamp(chunk_pos, vec3<i32>(0), vec3<i32>(grid_ix - 1, grid_iy - 1, grid_iz - 1));

    for (var step_index: u32 = 0u; step_index < MAX_WORLD_STEPS; step_index = step_index + 1u) {
        if chunk_pos.x < 0 || chunk_pos.x >= grid_ix
			|| chunk_pos.y < 0 || chunk_pos.y >= grid_iy
			|| chunk_pos.z < 0 || chunk_pos.z >= grid_iz {
            break;
        }
        g_probe_count = g_probe_count + 1u;
        let chunk_origin_world = grid_min + vec3<f32>(chunk_pos) * CHUNK_SIZE_F;
        let flat_index = u32(chunk_pos.x)
			+ u32(chunk_pos.y) * uniforms.chunk_grid_dim.x
			+ u32(chunk_pos.z) * uniforms.chunk_grid_dim.x * uniforms.chunk_grid_dim.y;
        let chunk_offset = chunk_directory[flat_index];
        if chunk_offset != EMPTY_CHUNK_SENTINEL {
            let local_entry = pos_world - chunk_origin_world;
            let header = read_chunk_header(chunk_offset);
            let hit = trace_chunk(header, local_entry, safe_dir, current_entry_axis, entered_from_outside);
            if hit.hit {
                if uniforms.render_mode == 1u {
                    return probe_pixel();
                }
                let albedo = material_rgb(hit.material);
                return vec4<f32>(shade(albedo, hit.normal), 1.0);
            }
        }
        // Distance from current position to each of this chunk's exit faces.
        // We compute freshly every iteration so there's no drift from
        // accumulated t deltas.
        let side_pos = chunk_origin_world + vec3<f32>(max(chunk_step, vec3<i32>(0))) * CHUNK_SIZE_F;
        let side_dist = (side_pos - pos_world) * inv_safe_dir;
        let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

        // Decide which face we're crossing and advance chunk_pos by exactly
        // one along that axis.
        var next_chunk_pos = chunk_pos;
        if side_dist.x <= side_dist.y && side_dist.x <= side_dist.z {
            next_chunk_pos.x = chunk_pos.x + chunk_step.x;
            current_entry_axis = 0u;
        } else if side_dist.y <= side_dist.z {
            next_chunk_pos.y = chunk_pos.y + chunk_step.y;
            current_entry_axis = 1u;
        } else {
            next_chunk_pos.z = chunk_pos.z + chunk_step.z;
            current_entry_axis = 2u;
        }

        // Move to the boundary, then clamp strictly inside the neighbour
        // chunk's AABB. That clamp is what makes the DDA robust: precision
        // loss in pos + dir * tmax can't push the ray onto the wrong side.
        let inset: f32 = 1.0 / 256.0;
        let neighbour_origin = grid_min + vec3<f32>(next_chunk_pos) * CHUNK_SIZE_F;
        let neighbour_min = neighbour_origin + vec3<f32>(inset);
        let neighbour_max = neighbour_origin + vec3<f32>(CHUNK_SIZE_F - inset);
        pos_world = clamp(pos_world + safe_dir * tmax, neighbour_min, neighbour_max);

        chunk_pos = next_chunk_pos;
        entered_from_outside = true;
    }
    if uniforms.render_mode == 1u {
        return probe_pixel();
    }
    return vec4<f32>(sky_color(ray_dir), 1.0);
}
