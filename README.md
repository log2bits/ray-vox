# ray-vox

A voxel renderer that ray traces everything (no rasterization). Rust + WebGPU.

## Data Structure

It’s called CBEPSV64DAG. An acronym pile-up:

- C, clipmapped. Many trees inside one big clipmap.
- B, bitpacked. Every value uses only as many bits as it needs.
- E, [efficient](https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf). Leaf nodes allowed anywhere in the tree.
- P, [pointerless](https://www.cai.sk/ojs/index.php/cai/article/view/2020_3_587). Popcount-implicit offsets, no per-cell pointers.
- S, sparse. Empty space costs nothing.
- V, voxel.
- 64, 4^3 branching [tetrahexacontree](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/).
- DAG, identical subtrees share storage.

## Priorities

1. Fast ray traversal, no hardware RT.
1. Heavy compression.
1. Quick edits.

# Storage

## Chunks

Each node has two 64-bit masks, `has_child` and `is_leaf`. Together they pick one of four cell states:

|has_child|is_leaf|meaning                                       |
|:-------:|:-----:|----------------------------------------------|
|0        |0      |empty                                         |
|0        |1      |filled, one material stored inline            |
|1        |0      |interior, has a child node with more branching|
|1        |1      |leaf, has a child node holding voxel materials|

A cell is occupied if either bit is set.

- Every chunk is depth 4 regardless of clipmap level. 256^3 voxels each, chunk-local coords fit in `[u8; 3]`.
- AoS layout. Interior nodes are 24 bytes, leaf nodes are 12 bytes. One fetch per visit, no SoA scatter across five arrays.
- Interior and leaf nodes live in separate arrays. The parent addresses both with a single packed pointer field.
- Materials array is fully packed. Doubles as LOD storage and leaf storage.
- DAG dedupes identical subtrees.
- Per-chunk material LUT. Bit width scales from 1 bit (2 materials) up to 32 bits (unique per voxel). LUT entries are full voxel values, so the hot traversal path only touches small indices.
- Persistent chunks (player edits) stick around. Procedural chunks generate on demand and discard out of range.

## Interior Node (24 bytes)

```
has_child   u64   // bit i set: cell i has a child node (interior or leaf)
is_leaf     u64   // bit i set: cell i is a leaf
ptrs        u32   // [12..0]  interior_ptr  (13 bits, into interior array)
                  // [31..13] leaf_ptr      (19 bits, into leaf array)
mat_offset  u32   // bit offset into chunk's materials array
```

`popcount_below(m, slot)` means `popcount(m & ((1 << slot) - 1))`.

Looking up a child at a given slot:

|state                            |location                                                             |
|---------------------------------|---------------------------------------------------------------------|
|interior (has_child=1, is_leaf=0)|`interior_ptr + popcount_below(has_child & !is_leaf, slot)`          |
|leaf (has_child=1, is_leaf=1)    |`leaf_ptr + popcount_below(has_child & is_leaf, slot)`               |
|filled (has_child=0, is_leaf=1)  |`mat_offset + popcount_below(!has_child & is_leaf, slot) * mat_width`|
|empty (has_child=0, is_leaf=0)   |nothing stored                                                       |

Pointer packing:

- Max interior nodes per chunk: 64^2 + 64 + 1 = 4161, fits in 13 bits (8192 capacity).
- Max leaf nodes per chunk: about 64^3 + 4160 = 266K, fits in 19 bits (524K capacity).
- 13 + 19 = 32 exactly. No waste.
- Extraction is one shift and one mask.
- Static-assert these bounds so any change to depth or branching gets caught.

## Leaf Node (12 bytes)

```
occ_mask    u64   // bit i set: cell i has a voxel
mat_offset  u32   // bit offset into chunk's materials array
```

- One mask is enough. A leaf node’s cells are voxels, so the mask is just occupancy.
- Cell `i` reads from: `mat_offset + popcount_below(occ_mask, i) * mat_width`.
- Leaves only exist for partially-occupied regions where cells actually differ.
- Uniform regions are stored as a `filled` cell on the parent instead. No node, one material entry.

## Material Array (per chunk)

```
[K palette entries x u32]    // full voxel values
[bitpacked indices]          // mat_width bits each, popcount-indexed
```

`mat_width = next_pow2(ceil(log2(K)))`. Power-of-two widths mean extraction is shift and mask, never divide:

|K (unique materials)|bits per slot|
|:------------------:|:-----------:|
|2                   |1            |
|3..4                |2            |
|5..16               |4            |
|17..256             |8            |
|257..65536          |16           |
|65537+              |32           |

- Both interior `mat_offset` (filled children) and leaf `mat_offset` (cells) write into the same shared per-chunk bitpacked array.
- Mid-tree fills and bottom-level leaf entries sit side by side.

## World Clipmap

- 28 levels covering the 2^64 coord space.
- Each level is 8^3 chunks, scaling 4x outward.
- Top level is 4*4*4 (exact 2^64 fit).
- All other levels are 8*8*8 with a 2*2*2 inner cutout that the next finer level fills.
- LOD boundary is always at least 3 cells from the camera. Coarse LOD never shows up close.
- Storage: 28 KB of chunk handles plus 1.8 KB occupancy bitmask.
- Chunk handles are u16. Max around 14K chunks across all levels, well under the 65K cap.

## Chunk Handle (u16)

```
[15..14]  unused
[13..9]   level    5 bits  (0..27)
[8..6]    x        3 bits  (0..7)
[5..3]    y        3 bits  (0..7)
[2..0]    z        3 bits  (0..7)
```

World-space recovery is O(1), no lookup table:

```
chunk_world = clipmap_origin[level] + (x, y, z) * chunk_size[level]
voxel_world = chunk_world + decode_path(path) * voxel_size[level]
```

- Handle encodes the full clipmap position directly.
- Any face ID or emissive ID can pull its world position back out.
- Direction vectors between two encoded positions are just a subtraction.

## Editing

- World keeps an ordered list of all procedural edits.
- Stale or new chunks find overlapping edits via AABB and apply them in order.
- Each chunk has its own ordered edit packet list.
- Procedural edits arrive pre-sorted. Player edits sort lazily.
- One edit is a `(path, material)` pair.
- Path is u32, split into four 8-bit blocks. Each block: 6 bits for a slot index (1..64), 0 means terminate at that level.
- Materials live in a bitpacked list with a LUT (same format as chunk material array).
- Voxel value 0 means air or delete.
- After an edit, the tree recompacts and DAG dedup runs again.

## GPU Memory

- Chunks live in a pool with a free list, since chunk size varies a lot.
- Chunk offset table maps the u16 handle to the actual memory offset.
- Edits only re-upload the affected chunk.

# Rendering

## Ray Tracing

- DDA with an ancestor stack. Neighbor steps pop or push the stack instead of restarting from the clipmap root.
- O(1) common ancestor via diffbits. XOR old and new position, clz, index the stack.
- 1.7 KB occupancy clipmap as an L1-resident fast reject. Sky-bound shadow rays never touch chunk data.
- 2^3 coarse occupancy grouping skips 8-cell empty regions in one mask test.
- Ray-octant mirroring bakes direction sign into the coord system. No per-step direction branches.
- Fractional coord system in `[1.0, 2.0)`. Cell index and floor-to-scale are float mantissa bit ops.
- After each step, position clamps to the neighbor bounding box. No bias offset.
- Ancestor stack in workgroup shared memory. Lower register pressure, higher occupancy.
- Camera is integer chunk coords plus a chunk-local f32 offset. f32 is enough for all traversal math.
- LOD-aware shape coverage skips sub-voxel detail at coarse levels.

## Per-Face Lighting

Lighting is cached per voxel face in a GPU hashmap. Three passes, so reads and writes can’t race and shadow rays can batch:

1. Traversal. Rays write a face ID per pixel into the G-buffer. No hashmap access yet.
1. Lookup. Face IDs get deduped (many pixels share a face). Each unique face probes the hashmap. Hits return cached light, misses queue shadow rays. A screen-space temporal layer also compares the current face ID against last frame’s at the same pixel. For static geometry this catches most lookups before the hashmap ever gets touched.
1. Shadow rays and writeback. Uncached faces dispatch shadow rays. Results write back to the hashmap.

Final shading:

```
final_color = albedo * rgb
```

where `rgb` is the total blended incident light from everything cached in the hashmap value.

### Lighting Model

```
total = direct sun shadow ray (one per visible face)
      + up to 4 cached emissive contributions per face (direct rays)
      + multi-bounce GI (progressive accumulation)
      + sky ambient
```

- GI uses progressive accumulation. A fixed ray budget per frame blends into cached results over many frames, converging toward a path-traced reference without a full per-frame solve.
- Off-screen emissive voxels get discovered by random bounce rays.
- Once found, they enter the global emissive pool and light visible faces directly.
- Newly-discovered emissives also fire rays outward, so face caches on surfaces the light can see fill in without waiting for camera-side rays to make the connection.

### Hashmap

- About 1 to 2 million slots, 16 bytes per slot, around 32 MB total.
- Load factor about 0.5.

Key (8 bytes, u64). Stable face identifier, never mutated:

```
[63..48]  chunk_id        u16    // clipmap-encoded chunk position
[47..16]  path            u8;4   // 6-bit slot + 2 unused per level
[15..8]   face_direction  u8     // bits 7..5: direction 0..5, bits 4..0: unused
[7..0]    reserved        u8
```

- Path encoding matches the edit format.
- Chunk handle gives O(1) world position.

Value (8 bytes):

```
rgb            u8;3   // blended incident light from all sources
generation     u8     // compared against a global frame counter, stale entries don't need explicit eviction
emissive_ids   u8;4   // indices into the emissive pool (0 means empty)
```

### Emissive Pool

- Global pool of up to 255 discovered emissive voxels.
- Whole pool is 255 * 8 = 2040 bytes. Fits in L1, stays resident during shading.
- u8 indices. 0 reserved as the empty sentinel.
- When the pool fills, lowest-influence entry (intensity over distance squared) gets evicted.

Entry (8 bytes):

```
[63..48]  chunk_id  u16
[47..16]  path      u8;4
[15..0]   unused    u16
```

- No face direction. Emissives are position-only.
- Color is fetched from voxel data at ray dispatch, since the chunk has to be loaded anyway.

# Voxel Format (u32)

|Bits |Field      |Notes                         |
|-----|-----------|------------------------------|
|31..8|RGB color  |24-bit linear RGB             |
|7..4 |Roughness  |0 = mirror, 15 = fully diffuse|
|3    |Emissive   |Emits light at albedo color   |
|2    |Metallic   |Albedo tints specular         |
|1    |Transparent|Refracts, doesn’t reflect     |
|0    |Textured   |Random color variation        |

- Voxel value 0 is air. Zero state of every bitpacked array, costs nothing to store.

# Voxel Import Formats

## Minecraft (.mca)

- Each block ID becomes one material in the chunk LUT.
- Each block is 16^3 voxels of one material, which lines up exactly with one Level-1 cell.
- Block face textures live in a GPU-side table. On ray hit, the GPU resolves color from the per-block face texture table.
- Different faces can return different colors.
- Each face texture has its own color LUT. Fewer than 256 unique colors per block, so each voxel takes 1 byte at hit time.
- Compresses tighter than `.mca` itself thanks to DAG dedup, bitpacking, and the 16^3 structural alignment.
- Catch: non-cube models (lecterns, stairs, fences) get voxelized. Some end up looking off.

## MagicaVoxel (.vox)

- Direct translation. Materials and palette map across without much fuss.

## glTF (.gltf, .glb)

- Bin triangles into chunks.
- Triangle/voxel intersection per voxel.
- Configurable voxel scale and palette.
- PBR material extraction (ideally).

# Resources

## References

|Reference                                                                                                               |Why it’s here                                                                           |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
|[Sparse 64-trees guide](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)                                |Traversal algorithm: ancestor stack, coarse occupancy, octant mirroring, mantissa tricks|
|[Aokana (2505.02017)](https://arxiv.org/abs/2505.02017)                                                                 |Chunked SVDAG with LOD streaming, validates uniform-resolution chunks                   |
|[Hybrid Voxel Formats (2410.14128)](https://arxiv.org/abs/2410.14128)                                                   |Systematic comparison of voxel storage formats                                          |
|[High-Resolution SVDAGs](https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/highResolutionSparseVoxelDAGs.pdf)|The original SVDAG paper                                                                |
|[ESVO](https://www.researchgate.net/publication/47645140_Efficient_Sparse_Voxel_Octrees)                                |Laine and Karras, SVO traversal and beam optimization                                   |
|[Voxelis Bible](https://github.com/WildPixelGames/voxelis)                                                              |SVO-DAG deep dive: batching, CoW, SoA, LOD, hash consing                                |
|[Amanatides and Woo DDA](http://www.cse.yorku.ca/~amana/research/grid.pdf)                                              |DDA for grid traversal                                                                  |
|[Erosion filter](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html)                             |LOD-friendly per-point erosion                                                          |

## Channels

|Channel                                               |Focus                                       |
|------------------------------------------------------|--------------------------------------------|
|[Douglas Dwyer](https://www.youtube.com/@DouglasDwyer)|Octo engine, Rust + WebGPU, path-traced GI  |
|[John Lin (Voxely)](https://www.youtube.com/@johnlin) |Path-traced voxel sandbox, per-face lighting|
|[Gabe Rundlett](https://www.youtube.com/@GabeRundlett)|C++ voxel engine, Daxa/Vulkan               |
|[Ethan Gore](https://www.youtube.com/@EthanGore)      |Engine dev, binary greedy meshing           |
|[VoxelRifts](https://www.youtube.com/@VoxelRifts)     |Voxel programming explainers                |
|[SimonDev](https://www.youtube.com/@simondev758)      |Radiance cascades                           |

## Projects

|Project                                                    |Description                           |
|-----------------------------------------------------------|--------------------------------------|
|[VoxelRT](https://github.com/dubiousconst282/VoxelRT)      |Tree64, brickmap, XBrickMap benchmarks|
|[Voxelis](https://github.com/WildPixelGames/voxelis)       |Rust SVO-DAG with batching, CoW, LOD  |
|[Octo Engine](https://github.com/DouglasDwyer/octo-release)|Rust + WebGPU voxel engine            |
|[tree64](https://github.com/expenses/tree64)               |Rust sparse 64-tree                   |
|[HashDAG](https://github.com/Phyronnaz/HashDAG)            |Reference implementation              |
|[gvox](https://github.com/GabeRundlett/gvox)               |Voxel format translation library      |

## More

|Resource                                                           |Description                       |
|-------------------------------------------------------------------|----------------------------------|
|[Voxel.Wiki](https://voxel.wiki)                                   |Community hub                     |
|[Voxely.net blog](https://voxely.net/blog/)                        |John Lin’s design posts           |
|[Brickmap rundown](https://uygarb.dev/posts/0003_brickmap_rundown/)|Brickmap and brickgrid explanation|
|[Branchless DDA (ShaderToy)](https://www.shadertoy.com/view/XdtcRM)|Branchless 3D DDA reference       |