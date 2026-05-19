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
- A leaf mask per node allows larger voxels higher up in the tree, avoiding unnecessary descent
- Materials array is fully packed, doubling as LOD storage and leaf storage.
- Materials are stored in bitpacked arrays based on how many unique voxel materials exist in said chunk. 
- The bitpacked arrays have power-of-2 bitwidths that scale based on content.
- LUT for each chunk allow bitpacking to work efficiently
- SoA layout per level
- DAG allows identical subtrees to share memory
- Persistent chunks (player-edited) are stored permanently. Procedural chunks are generated on demand and discarded when out of range.

### World Clipmap
- World stores chunks in a 28 level clipmap. (spans the 2^64 coordinate space)
- Each clipmap is 8^3 chunks, and gets 4x bigger each time.
- Clipmap stores chunk handles (u32 * 8^3 * 28) = 57 KB and an occupancy bitmask (u1 * 8^3 * 28) = 1.8 KB
- Top level is 4x4x4 cells (exact 2^64 fit)
- All other levels are 8×8×8 with a 2×2×2 inner cutout filled by the next finer level.
- LOD boundaries always 3 cells from the camera; coarse LOD is never visible up close.

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
- Chunk offset table to map the u32 chunk handle to the offset in memory
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
- GPU hashmap maps voxel face world coordinates to lighting information, and is lookup up when drawing to the screen.
- Per-face temporal cache
- One shadow ray per visible face

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
