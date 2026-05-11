struct Uniforms {
    cam_pos:         vec3<f32>,
    fov_half_tan:    f32,
    cam_right:       vec3<f32>,
    viewport_w:      f32,
    cam_up:          vec3<f32>,
    viewport_h:      f32,
    cam_forward:     vec3<f32>,
    _pad0:           f32,
    node_counts:     vec4<u32>,
    slot_counts:     vec4<u32>,
    level_offsets:   vec4<u32>,
    material_count:  u32,
    material_offset: u32,
    tree_occupied:   u32,
    tree_is_leaf:    u32,
    tree_leaf_value: u32,
}

@group(0) @binding(0) var<uniform>       u:   Uniforms;
@group(0) @binding(1) var<storage, read> buf: array<u32>;

// ---- bit helpers --------------------------------------------------------

fn bit_set(lo: u32, hi: u32, bit: u32) -> bool {
    if bit < 32u { return ((lo >> bit) & 1u) != 0u; }
    return ((hi >> (bit - 32u)) & 1u) != 0u;
}

fn popcount_below(lo: u32, hi: u32, bit: u32) -> u32 {
    if bit < 32u { return countOneBits(lo & ((1u << bit) - 1u)); }
    return countOneBits(lo) + countOneBits(hi & ((1u << (bit - 32u)) - 1u));
}

// ---- fractional coordinate helpers (coords in [1, 2)) ------------------
//
// The chunk lives in [1.0, 2.0)^3.  Each tree level consumes 2 mantissa bits;
// scale_exp runs 21→19→17→15 (root→leaf).
// Cell size at scale_exp e = 2^(e-23) * 256 voxels.

fn cell_index(pos: vec3<f32>, scale_exp: u32) -> u32 {
    let b = (bitcast<vec3<u32>>(pos) >> vec3(scale_exp)) & vec3(3u);
    return b.x | (b.y << 2u) | (b.z << 4u);
}

fn floor_scale(pos: vec3<f32>, scale_exp: u32) -> vec3<f32> {
    let mask = ~((1u << scale_exp) - 1u);
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(pos) & vec3(mask));
}

fn mirror_coord(v: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(v) ^ 0x7FFFFFu);
}

// ---- material / shading -------------------------------------------------

fn material_color(mat: u32) -> vec3<f32> {
    let raw = buf[u.material_offset + mat];
    let r = f32((raw >> 24u) & 0xffu) / 255.0;
    let g = f32((raw >> 16u) & 0xffu) / 255.0;
    let b = f32((raw >>  8u) & 0xffu) / 255.0;
    return vec3(r, g, b);
}

fn shade(mat: u32, normal: vec3<f32>) -> vec4<f32> {
    let diffuse = max(dot(normal, normalize(SUN_DIR)), 0.0);
    let light   = AMBIENT + (1.0 - AMBIENT) * diffuse;
    return vec4(material_color(mat) * light, 1.0);
}

// ---- constants ----------------------------------------------------------

const SKY:            vec4<f32> = vec4(0.12, 0.15, 0.22, 1.0);
const MIN_VOX_PIXELS: f32       = 4.0;
const SUN_DIR:        vec3<f32> = vec3(0.6, 0.8, 0.2);
const AMBIENT:        f32       = 0.15;

const SCALE_EXP_ROOT: u32 = 21u;
const SCALE_EXP_LEAF: u32 = 15u;

// ---- DDA ray traversal --------------------------------------------------

fn ray_cast(origin_chunk: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    if u.tree_occupied == 0u { return SKY; }

    if u.tree_is_leaf != 0u {
        return shade(u.tree_leaf_value, vec3(0.0, 1.0, 0.0));
    }

    // Mirror all direction components to negative so the exit face is always cellMin.
    let do_mirror = vec3(dir.x > 0.0, dir.y > 0.0, dir.z > 0.0);

    // Map origin from chunk voxels [0,256] to fractional [1,2), then mirror.
    var frac_origin = vec3(1.0) + origin_chunk / 256.0;
    frac_origin = vec3(
        select(frac_origin.x, select(mirror_coord(frac_origin.x), 3.0 - frac_origin.x, frac_origin.x < 1.0 || frac_origin.x >= 2.0), do_mirror.x),
        select(frac_origin.y, select(mirror_coord(frac_origin.y), 3.0 - frac_origin.y, frac_origin.y < 1.0 || frac_origin.y >= 2.0), do_mirror.y),
        select(frac_origin.z, select(mirror_coord(frac_origin.z), 3.0 - frac_origin.z, frac_origin.z < 1.0 || frac_origin.z >= 2.0), do_mirror.z),
    );
    let inv_dir = 1.0 / -abs(dir);

    var pos = clamp(frac_origin, vec3(1.0), vec3(bitcast<f32>(0x3FFFFFFFu)));

    // XOR applied to cell_index output to undo coordinate reflection.
    var mirror_mask = 0u;
    if do_mirror.x { mirror_mask |= 3u; }
    if do_mirror.y { mirror_mask |= 3u << 2u; }
    if do_mirror.z { mirror_mask |= 3u << 4u; }

    let pixel_ws_k = 2.0 * u.fov_half_tan / u.viewport_h;

    var stack     = array<u32, 4>(0u, 0u, 0u, 0u);
    var scale_exp = SCALE_EXP_ROOT;
    var node      = 0u;
    var normal    = vec3(0.0);

    for (var iter = 0; iter < 256; iter++) {
        // Load current level's occupancy data once; the inner loop reuses/updates it
        // during consecutive descents without returning to the outer loop.
        var child_idx  = cell_index(pos, scale_exp) ^ mirror_mask;
        var depth      = (SCALE_EXP_ROOT - scale_exp) >> 1u;
        var node_count = u.node_counts[depth];
        var slot_count = u.slot_counts[depth];
        var level_base = u.level_offsets[depth];
        var occ_lo     = buf[level_base + node];
        var occ_hi     = buf[level_base + node_count + node];

        // Inner descent loop: descend while the slot is occupied, stopping only when
        // we find an empty slot, a leaf, or the LOD threshold fires.
        while bit_set(occ_lo, occ_hi, child_idx) {
            let is_leaf = bit_set(
                buf[level_base + 2u * node_count + node],
                buf[level_base + 3u * node_count + node],
                child_idx,
            );
            let rank         = popcount_below(occ_lo, occ_hi, child_idx);
            let child_offset = buf[level_base + 4u * node_count + node];
            let vals_base    = level_base + 5u * node_count;
            let lod_val      = buf[vals_base + child_offset + rank];

            // LOD: stop when the grandchild voxel size drops below MIN_VOX_PIXELS.
            // grandchild_vox at depth d: 256 >> (2*(d+2)) → 16,4,1,0
            let grandchild_vox = 256u >> (2u * (depth + 2u));
            let dist_vox       = length((pos - frac_origin) * 256.0);
            let lod_stop       = f32(grandchild_vox) <= dist_vox * pixel_ws_k * MIN_VOX_PIXELS;

            if is_leaf || scale_exp == SCALE_EXP_LEAF || lod_stop {
                if lod_val != 0u { return shade(lod_val, normal); }
                break; // lod_val==0: LOD is air — advance past this node
            }

            // Descend: push child onto stack, update level data for next iteration.
            stack[depth + 1u] = buf[vals_base + slot_count + child_offset + rank];
            node      = stack[depth + 1u];
            scale_exp -= 2u;

            child_idx  = cell_index(pos, scale_exp) ^ mirror_mask;
            depth      = (SCALE_EXP_ROOT - scale_exp) >> 1u;
            node_count = u.node_counts[depth];
            slot_count = u.slot_counts[depth];
            level_base = u.level_offsets[depth];
            occ_lo     = buf[level_base + node];
            occ_hi     = buf[level_base + node_count + node];
        }

        // --- advance the ray to the next cell ---

        // Coarse 2³ skip: if the 2x2x2 octant containing pos is fully empty, double the step.
        var adv_scale_exp = scale_exp;
        let octant_base   = child_idx & 0x2Au;
        let occ_word      = select(occ_lo, occ_hi, (octant_base & 32u) != 0u);
        if ((occ_word >> (octant_base & 31u)) & 0x00330033u) == 0u {
            adv_scale_exp += 1u;
        }

        let cell_min  = floor_scale(pos, adv_scale_exp);
        let side_dist = (cell_min - frac_origin) * inv_dir;
        let t_exit    = min(min(side_dist.x, side_dist.y), side_dist.z);

        let adv_bits          = (1u << adv_scale_exp) - 1u;
        let neighbor_max_bits = bitcast<vec3<u32>>(cell_min) +
            select(vec3(adv_bits), vec3(0xFFFFFFFFu), side_dist == vec3(t_exit));
        pos = min(frac_origin - abs(dir) * t_exit, bitcast<vec3<f32>>(neighbor_max_bits));

        // Normal: exit face = axis with minimum side_dist; flip sign for mirrored axes.
        if side_dist.x <= side_dist.y && side_dist.x <= side_dist.z {
            normal = vec3(select(1.0, -1.0, do_mirror.x), 0.0, 0.0);
        } else if side_dist.y <= side_dist.z {
            normal = vec3(0.0, select(1.0, -1.0, do_mirror.y), 0.0);
        } else {
            normal = vec3(0.0, 0.0, select(1.0, -1.0, do_mirror.z));
        }

        // Ancestor ascent: XOR old and new pos to find the highest changed bit,
        // then pop the stack to that level.
        // Tree scale_exp values are odd (21,19,17,15); round up to next odd if even.
        let changed_bits = (bitcast<vec3<u32>>(pos) ^ bitcast<vec3<u32>>(cell_min)) & vec3(0xFFAAAAAAu);
        var ascend_exp   = firstLeadingBit(changed_bits.x | changed_bits.y | changed_bits.z);
        if (ascend_exp & 1u) == 0u { ascend_exp += 1u; }
        if ascend_exp > scale_exp {
            scale_exp = ascend_exp;
            if scale_exp > SCALE_EXP_ROOT { break; }
            node = stack[(SCALE_EXP_ROOT - scale_exp) >> 1u];
        }
    }

    return SKY;
}

// ---- vertex / fragment --------------------------------------------------

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x   = f32((vid & 1u) << 1u);
    let y   = f32((vid & 2u));
    let ndc = vec2(x * 2.0 - 1.0, y * 2.0 - 1.0);
    var out: VsOut;
    out.clip_pos = vec4(ndc, 0.0, 1.0);
    out.ndc      = ndc;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let aspect  = u.viewport_w / u.viewport_h;
    let ray_dir = normalize(
        u.cam_forward
        + in.ndc.x * aspect * u.fov_half_tan * u.cam_right
        - in.ndc.y *          u.fov_half_tan * u.cam_up
    );
    return ray_cast(u.cam_pos, ray_dir);
}
