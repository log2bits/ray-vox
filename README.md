# ray-vox

A voxel renderer built around software ray tracing (no rasterization), built in Rust + WebGPU.

# Primary Data Structure
Heavily modified sparse voxel data structure. The acronym is an abomination: **CBEPSV64DAG**
- C - Clipmapped (many individual trees in one big clipmap)
- B - Bitpacked (values and offsets use only as many bits as required)
- E - [Efficient](https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf) (allows for leaf nodes anywhere in the tree)
- P - [Pointerless](https://www.cai.sk/ojs/index.php/cai/article/view/2020_3_587) (implicit offsets are stored instead of explicit pointers)
- S - Sparse (only occupied nodes are stored, empty space is stored efficiently)
- V - Voxel (Volumetric Pixel)
- 64 - Tetrahexacontree (or [64-tree](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/))
- DAG - Directed Acyclic Graph

# Priorities
1. Extremely fast ray traversal (as fast as I can go without resorting to hardware RT)
2. Wonderful compression
3. Quick editability

# Optimizations

## Storage

### Chunk
- All chunks have the same depth (4), regardless of which clipmap level they exist in.
- Depth 4 chunks = 256^3 voxels, allowing chunk local coordinates can be stored as [u8; 3]
- Two masks per node (branch_mask and terminal_mask) encode four child states:

  branch + terminal = leaf node,
  
  branch + not terminal = interior node,
  
  not branch + terminal = filled node (single material, no node stored),

  not branch + not terminal = empty.
  
  Occupied children = branch_mask | terminal_mask.

- Materials array is fully packed, doubling as LOD storage and leaf storage.
- AoS node layout: internal nodes are 20 bytes (5 u32s), leaf nodes are 8 bytes (2 u32s). All fields for a node are fetched together instead of 5 separate SoA array reads per node visit. Internal and leaf nodes live in separate arrays.
- DAG allows identical subtrees to share memory
- Materials are stored in a per-chunk bitpacked array with a LUT. Bit width scales with unique material count: a chunk with 2 materials uses 1 bit per slot, 4 materials uses 2 bits, up to 8 bits for 256 materials.
- The LUT maps compact indices to full voxel values, so the hot traversal path only touches small indices.
- Persistent chunks (player-edited) are stored permanently. Procedural chunks are generated on demand and discarded when out of range.

### Internal Node Format (20 bytes)

```
[31..0]  branch_lo   u32  - branch mask bits 0-31  (children that are interior or leaf nodes)
[31..0]  branch_hi   u32  - branch mask bits 32-63
[31..0]  terminal_lo u32  - terminal mask bits 0-31 (children that are leaf or filled nodes)
[31..0]  terminal_hi u32  - terminal mask bits 32-63
[31..0]  child_ptr   u32  - index of first child in whichever array (interior or leaf)
```

The four combinations of branch and terminal bits determine child type. Interior nodes are indexed from the interior node array, leaf nodes from the leaf node array, and filled nodes contribute a single entry directly to the materials array. No node is allocated for filled children.

### Leaf Node Format (8 bytes)

```
[31..0]  occ_lo     u32  - occupancy mask bits 0-31
[31..0]  occ_hi     u32  - occupancy mask bits 32-63
```

Leaf nodes exist only for partially occupied regions (branch + terminal). Fully uniform regions use the filled encoding instead and store no node at all, only a single material entry in the parent's materials array.

### Material Array (per chunk)

```
[palette_entry_0]     u32  - full Voxel value
[palette_entry_1]     u32
...
[palette_entry_K-1]   u32
[bitpacked indices]        - next_pow2(ceil(log2(K))) bits per occupied leaf slot
```

Bit width is a single chunk-level uniform, fixed to the next power of 2 so indices can be extracted with a bitshift and mask rather than a division. At 2 materials: 1 bit/slot. At 3-4 materials: 2 bits/slot. At 5-16 materials: 4 bits/slot. At 17-256 materials: 8 bits/slot.

The array is fully packed - only occupied slots have entries, indexed via `child_ptr + popcount_below(occ_mask, slot)`. Uniform nodes higher up in the tree write into the same array as bottom-level leaf nodes.

### World Clipmap
- World stores chunks in a 28 level clipmap. (spans the 2^64 coordinate space)
- Each clipmap is 8^3 chunks, and gets 4x bigger each time.
- Clipmap stores chunk handles (u16 * 8^3 * 28) = 28 KB and an occupancy bitmask (u1 * 8^3 * 28) = 1.8 KB
- Top level is 4x4x4 cells (exact 2^64 fit)
- All other levels are 8×8×8 with a 2×2×2 inner cutout filled by the next finer level.
- LOD boundaries always 3 cells from the camera; coarse LOD is never visible up close.
- Chunk handles are u16 (max 14,336 chunks across all levels fits comfortably in u16's 65,536 range)

### Chunk Handle Encoding (u16)

The chunk handle encodes the full clipmap position directly, making world-space position recovery O(1) with no table lookup:

```
[15..14] unused
[13..9]  level  (5 bits, 0-27)
[8..6]   x      (3 bits, 0-7)
[5..3]   y      (3 bits, 0-7)
[2..0]   z      (3 bits, 0-7)
```

World-space position recovery:
```
chunk_world_pos = clipmap_origin[level] + (x, y, z) * chunk_size[level]
voxel_world_pos = chunk_world_pos + decode_path(path) * voxel_size[level]
```

This means any face ID or emissive voxel ID can have its world-space position recovered in O(1), enabling direction vectors between any two encoded positions as a simple subtraction.

### Editing
- The world stores an ordered list of all procedural edits. When a chunk is stale or new, we find all edits that overlap the chunk (using AABB) and apply them.
- Each chunk stores its own ordered list of edit packets, which are applied sequentially.
- Packets that come from procedural edits are sent in pre-sorted.
- Packets that aren't sorted are lazily sorted later.
- Each individual edit in a chunk is stored as a path, and a material.
- The path can be any height in the tree, so we must store it as a u32. This is split into 4 blocks of 8 bits. The first 6 bits of each block store 1-64 for the slot, and 0 if the path terminates there.
- The materials are stored in a list of bitpacked values with a LUT.
- A voxel value of all zeros represents air or a deletion.
- After an edit is applied, the tree is compacted, allowing DAG deduplication.

### GPU Memory
- Chunks are stored as a large pool in a free-list. Chunks have highly variable memory length, so the free-list is essential.
- Chunk offset table to map the u16 chunk handle to the offset in memory
- Edits re-upload only the affected chunk.

## Rendering

### Ray Tracing
- DDA with ancestor stack; neighbor steps pop/push the stack rather than restarting from the clipmap root.
- Common ancestor found in O(1) via diffbits: XOR old and new position, clz, index stack directly.
- Occupancy clipmap (1.7 KB) checked first as an L1-resident fast-reject. Shadow rays through empty sky never touch chunk data.
- 2³ coarse occupancy grouping skips 8-cell empty regions in a single mask test.
- Ray-octant mirroring bakes direction sign into the coordinate system, removing per-step branches in the inner loop.
- Fractional coordinate system \[1.0, 2.0): cell index and floor-to-scale computed via float mantissa bit ops.
- Position clamped to neighbor bounding box after each step rather than using a bias offset.
- Ancestor stack in workgroup shared memory to reduce register pressure and improve occupancy.
- Camera stored as integer chunk coordinates plus a chunk-local float offset; f32 is sufficient for all traversal math.
- LOD-aware shape coverage skips sub-voxel detail passes at coarse levels.

### Per Face Lighting

#### Overview
Lighting is cached per voxel face in a GPU hashmap. Rendering is split into three passes to avoid read/write hazards and allow batched shadow ray dispatch:

- **Pass 1 - Traversal:** rays traverse the scene and write a face ID per pixel into a G-buffer. No hashmap access.
- **Pass 2 - Lighting lookup:** face IDs are deduplicated (many pixels share the same face). Each unique face is looked up in the hashmap. Cache hits return cached lighting. Cache misses are queued for shadow ray dispatch. A screen-space temporal layer compares the current face ID against last frame's stored face ID at the same pixel - for static geometry this absorbs the majority of lookups before touching the hashmap.
- **Pass 3 - Shadow rays & writeback:** shadow rays are dispatched for uncached faces. Results are written back to the hashmap.

Final shading:
```
final_color = albedo * rgb
```

where `rgb` is the total blended incident light from all sources cached in the hashmap value.

#### Lighting Model
```
total lighting = direct shadow ray (one per visible face to the sun)
              + emissive contributions (up to 4 cached emissive voxels per face, direct rays)
              + multiple bounce lighting for global illumination
              + sky ambient
```

Multi-bounce GI is handled via progressive accumulation - a fixed ray budget per frame is blended into cached results over many frames, converging toward correct path-traced output without a full per-frame solve.

Emissive voxels off-screen are discovered via random bounce rays. Once discovered they are added to the global emissive voxel pool and can light visible faces directly. Rays are also fired outward from newly discovered emissive voxels (bidirectional), seeding face caches for surfaces the light can see without waiting for camera-side rays to find the connection.

#### Hashmap

~1-2M slots, 16 bytes per slot, ~32MB total. Load factor ~0.5.

**Key (8 bytes / u64) - stable face identifier, never mutated:**

```
[63..48] chunk_id      u16  - clipmap-encoded chunk position (see Chunk Handle Encoding)
[47..16] path          [u8; 4] - 6 bits slot (1-64, 0=termination), 2 bits unused per level
[15..8]  face_direction u8  - bits 7-5: face direction (0-5), bits 4-0: unused
[7..0]   reserved      u8
```

The path encoding is identical to the edit path format. The chunk handle encodes the full clipmap position, so world-space position of any face is recoverable in O(1) - see Chunk Handle Encoding above.

**Value (8 bytes):**

```
rgb:          [u8; 3]  - total blended incident lighting from all sources
generation:   u8       - compared against a global frame counter; stale entries need no explicit eviction
emissive_ids: [u8; 4]  - indices into global emissive voxel pool (0 = empty slot, 1-255 = valid)
```

#### Emissive Voxel Pool

A global pool of up to 255 discovered emissive voxels. The entire pool is 255 × 8 = **2040 bytes**, fitting comfortably in L1 cache and remaining resident during shading passes. Indices are u8 (0 reserved as empty sentinel). When the pool is full, the lowest-influence entry (by intensity / distance²) is evicted.

**Pool entry (8 bytes):**

```
[63..48] chunk_id  u16     - clipmap-encoded chunk position
[47..16] path      [u8; 4] - path to voxel within chunk, same encoding as face ID path
[15..0]  unused    u16
```

No face direction is stored - emissive voxels are identified by position only. Color is fetched from the voxel data at ray dispatch time since the chunk must be accessed anyway to begin traversal.

## Voxel Format

| Bits | Field       | Notes                          |
| ---- | ----------- | ------------------------------ |
| 31–8 | RGB color   | 24-bit linear RGB              |
| 7–4  | Roughness   | 0 = mirror, 15 = fully diffuse |
| 3    | Emissive    | Emits light at albedo color    |
| 2    | Metallic    | Albedo tints specular          |
| 1    | Transparent | Refracts rather than reflects  |
| 0    | Textured    | Add random variation to color  |

Voxel value 0 is reserved as **air**. It is the zero-state of the bitpacked arrays and requires no storage.

## Planned Voxel Import Formats

### Minecraft Worlds (.mca)
- Every block is given a unique material, and a block face texture lookup is sent to the GPU
- Every block is represented as 16^3 voxels of the same material
- GPU looks up color on hit from block texture table (different faces can have different colors too!)
- Each set of block textures uses a color LUT to keep each color under 1 byte (there are less than 256 unique colors in each minecraft block)
- This should allow for compression greater than even minecraft's .mca since we have DAG deduplication and bitpacking
- Only limitation is that all block models must be remapped to voxels so some blocks may look off (the lectern for example)

### MagicaVoxel (.vox)
- Quite straightforward

### GLTF (.gltf/.glb)
- Bin triangles into chunks
- Do triangle voxel intersection tests
- Configurable voxel scale
- Configurable color palette
- PBR material extraction (ideally)

## Resources

### References

| Reference | Why it matters |
|---|---|
| [Guide to sparse 64-trees](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) | Traversal algorithm: ancestor stack, coarse occupancy, octant mirroring, mantissa tricks |
| [Aokana (2505.02017)](https://arxiv.org/abs/2505.02017) | Chunked SVDAG with LOD streaming; validates the uniform-chunk-resolution approach |
| [Hybrid Voxel Formats (2410.14128)](https://arxiv.org/abs/2410.14128) | Systematic comparison of voxel storage formats |
| [High Resolution SVDAGs](https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/highResolutionSparseVoxelDAGs.pdf) | Original SVDAG paper |
| [Efficient Sparse Voxel Octrees](https://www.researchgate.net/publication/47645140_Efficient_Sparse_Voxel_Octrees) | Laine & Karras; SVO traversal and beam optimization |
| [Voxelis Bible](https://github.com/WildPixelGames/voxelis) | SVO-DAG deep dive: batching, CoW, SoA, LOD, hash consing |
| [Amanatides & Woo DDA](http://www.cse.yorku.ca/~amana/research/grid.pdf) | DDA algorithm for grid traversal |
| [Fast and Gorgeous Erosion Filter](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html) | LOD-friendly per-point erosion |

### Channels

| Channel | Focus |
|---|---|
| [Douglas Dwyer](https://www.youtube.com/@DouglasDwyer) | Octo voxel engine, Rust + WebGPU, path-traced GI |
| [John Lin (Voxely)](https://www.youtube.com/@johnlin) | Path-traced voxel sandbox, per-face lighting pipeline |
| [Gabe Rundlett](https://www.youtube.com/@GabeRundlett) | C++ voxel engine, Daxa/Vulkan |
| [Ethan Gore](https://www.youtube.com/@EthanGore) | Voxel engine dev, binary greedy meshing |
| [VoxelRifts](https://www.youtube.com/@VoxelRifts) | Voxel programming explainers |
| [SimonDev](https://www.youtube.com/@simondev758) | Radiance cascades |

### Projects

| Project | Description |
|---|---|
| [VoxelRT](https://github.com/dubiousconst282/VoxelRT) | Tree64, brickmap, XBrickMap benchmarks |
| [Voxelis](https://github.com/WildPixelGames/voxelis) | Rust SVO-DAG with batching, CoW, LOD |
| [Octo Engine](https://github.com/DouglasDwyer/octo-release) | Rust + WebGPU voxel engine |
| [tree64](https://github.com/expenses/tree64) | Rust sparse 64-tree |
| [HashDAG](https://github.com/Phyronnaz/HashDAG) | HashDAG reference implementation |
| [gvox](https://github.com/GabeRundlett/gvox) | Voxel format translation library |

### More

| Resource | Description |
|---|---|
| [Voxel.Wiki](https://voxel.wiki) | Community hub |
| [Voxely.net blog](https://voxely.net/blog/) | John Lin's design posts |
| [A Rundown on Brickmaps](https://uygarb.dev/posts/0003_brickmap_rundown/) | Brickmap/brickgrid explanation |
| [Branchless DDA (ShaderToy)](https://www.shadertoy.com/view/XdtcRM) | Branchless 3D DDA reference |
