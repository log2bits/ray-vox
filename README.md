# Lattice

A voxel renderer with path tracing, built in Rust + WebGPU.

---

# Data Structure
Custom **NBEPSVCDAG**
- N - Nested (chunk trees are nested within one big world tree)
- B - Bitpacked (values and offsets use only as many bits as required)
- E - [Efficient](https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf) (allows for leaf nodes anywhere in the tree)
- P - [Pointerless](https://www.cai.sk/ojs/index.php/cai/article/view/2020_3_587) (implicit offsets are stored instead of explicit pointers)
- S - Sparse (only occupied nodes are stored, empty space is stored efficiently)
- V - Voxel (Volumetric Pixel)
- C - Tetrahexa**contree** (or [64-tree](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/))
- DAG - Directed Acyclic Graph

---

# Priorities

1. Extremely fast ray traversal
2. Wonderful compression
3. Moderately fast editability

---

# Optimizations

### Storage

1. Both the world and each chunk use the same sparse 64-tree structure. The world tree (28 levels deep) stores chunk pool handles at its leaves; chunk trees (4 levels deep) store material indices. Same traversal algorithm, same flat SoA layout, same GPU buffer format.
2. A leaf mask per node marks subtrees that are uniform (single value), avoiding unnecessary descent.
3. The values array is fully packed with one entry per occupied slot and does double duty: leaf slots store the exact uniform value, non-leaf slots store the LOD representative. One array, no separate LOD field, no gaps.
4. The node children and values arrays share a single children offset with lock-step indexing. Leaf slots in the children array hold zero; only non-leaf slots carry child indices.
5. Per-chunk material tables deduplicate on the full 32-bit voxel value. The world tree has no material table — its values are chunk handles, not materials.
6. Bitpacked widths are powers of 2 and scale with content: 1 bit for ≤2 values, 2 for ≤4, 4 for ≤16, 8 for ≤256, etc.
7. SoA layout per level: GPU warps reading one field across many nodes hit contiguous memory.
8. Chunk levels stored top-down (coarsest first) so file reads for LOD can stop early.

### Editing

1. Each chunk owns an ordered list of edit packets applied sequentially. Later packets overwrite earlier ones. Each packet holds a list of tree path + value pairs and a sorted flag.
2. All edit sources — coverage walks, direct voxel edits, voxel imports — resolve to an edit packet before being applied to the tree. Edit packets are the single stable interface between edit generation and tree application.
3. There are two edit pathways: **lazy** (shape edits stored in the world's shape list, evaluated at chunk generation time) and **eager** (edit packets constructed directly and queued on the tree). Procedural shapes use lazy; player edits and imports use eager.
4. Chunk generation is parallelized across chunks with rayon. Each chunk reads the world's shape edit list (immutable) and writes only to its own tree — no coordination needed.
5. Each edit carries a position, a level (0 = single voxel, n = 4^n voxel cube), and a value (0 = delete/air). Level > 0 lets shape edits collapse entire uniform subtrees without expanding to individual voxels.
6. Applying edits sorts any unsorted packets by Morton key, then applies all packets in order. Cost is O(depth × edits) per packet.
7. The edit walk is top-down: for each node, find children with edits in their range. Children with no edits are copied unchanged. Children with edits are recursed into, producing new nodes appended to the level arrays.
8. After each flush, the tree is compacted — level arrays are rebuilt with only reachable nodes, removing orphans and enabling DAG deduplication.

### Shape Editing

Individual voxel edits are just one case. The edit walk generalizes to any shape that implements a coverage query:

- **Full coverage** (node AABB entirely inside shape): set subtree to a uniform leaf with the fill material, stop recursing.
- **No coverage** (node AABB entirely outside shape): copy subtree unchanged, stop recursing.
- **Partial coverage**: recurse into all 64 children.

This gives O(surface area) cost for any convex or SDF shape. A 64³ rectangular fill touches a handful of border nodes rather than 262k individual voxels. Works for rectangles, spheres, capsules, or any SDF. Heightmap terrain generation is a column-wise variant of the same walk.

At leaf level, coverage is evaluated per-voxel, so multi-material shapes (geology layers, dithered boundaries) can return different fill materials based on position.

Write-only shapes are pure geometry — no tree reads, parallelizable across chunks. Read+write shapes can query the current tree state during coverage evaluation, enabling edits that depend on prior edits (e.g. topsoiling after terrain). Read+write shapes are applied sequentially after all write-only shapes in the same stage are flushed. (Not yet implemented.)

### Rendering

1. All rendering is done in WGSL on the GPU. No CPU ray traversal in the hot path.
2. Ray traversal uses a DDA algorithm with an ancestor stack that caches parent node indices. Stepping into a neighbor cell does not restart from root — the stack is popped/pushed instead.
3. Coarse occupancy groups the 64-bit mask into 8 regions of 2×2×2, enabling 8-cell skips over large empty regions.
4. Coordinate flipping maps all rays into the negative octant, halving branch count in the DDA inner loop.
5. Camera position is stored as integer chunk coordinates plus a chunk-local vec3 offset always within [0, 256] voxels. All traversal happens in chunk-local space: when a ray exits a chunk, the integer chunk coordinates are stepped and the ray origin is re-expressed relative to the new chunk. No large floats ever enter traversal math, so f32 is sufficient for local coordinates regardless of world size.
6. For player interaction (placing or removing voxels), the same DDA algorithm runs on the CPU to find the exact hit position in world voxel coordinates. This is not in the rendering hot path.
7. Future: partial chunk upload — currently entire chunks are re-uploaded on edit; a future optimization would only send the top N levels to VRAM based on camera distance, with the GPU reading the LOD representative from the parent node when it runs out of uploaded data.

### Later (path tracing)

1. Bidirectional: rays from the camera and rays from light sources (sun, emissive voxels).
2. Emissive voxels discovered by camera rays get added to the light list dynamically.
3. Per-face unique ID with lighting averaged across that face, doing spatial and temporal accumulation in one pass.
4. Averaging strength tied to roughness: mirror faces accumulate slowly, diffuse faces accumulate aggressively.
5. Faces not updated recently get evicted from the lighting buffer.

---

## Voxel Format

| Bits | Field | Notes |
|------|-------|-------|
| 31–8 | RGB color | 24-bit linear RGB |
| 7–4 | Roughness | 0 = mirror, 15 = fully diffuse |
| 3 | Emissive | Emits light at albedo color |
| 2 | Metallic | Albedo tints specular |
| 1 | Transparent | Refracts rather than reflects |
| 0 | Textured | Sample from a texture atlas |

Voxel value 0 is reserved as **air** (empty). It is the zero-state of the bitpacked arrays and requires no storage. No valid material can have value 0.

---

## Tree

The core data structure, used for both the world tree and every chunk tree.

Each tree is a sparse 64-tree stored in a flat SoA (struct-of-arrays) layout. Each level holds a list of nodes, and each node has:

- **Occupancy mask** (64 bits): which of the node's 64 slots contain anything.
- **Leaf mask** (64 bits): which occupied slots are leaves (uniform subtrees). If a slot is occupied and in the leaf mask, it holds a direct value. If occupied but not in the leaf mask, it holds a child node index to descend into.
- **Children offset**: the start index in the values and node children arrays for this node's occupied slots. The rank of a slot (popcount of the occupancy mask below it) gives the index offset within that block.
- **Values array** (bitpacked): one entry per occupied slot. Leaf slots hold the leaf value (material index or chunk handle). Non-leaf slots hold the LOD representative for that subtree (mode of occupied child values). Same array, no separate LOD storage.
- **Node children array** (bitpacked): one entry per occupied slot. Non-leaf slots hold child node indices. Leaf slots hold zero. Empty at the bottommost level.

Bitpacked widths are the smallest power-of-2 that fits the largest value stored, so a tree with only two unique materials uses 1-bit values throughout. Widths can differ between the values and node children arrays at the same level.

The tree root is represented by a small header: a flag for whether the tree is entirely empty, a flag for whether it is a uniform leaf (and the leaf value), and otherwise the index of the root node in the first level's arrays.

After applying edits, the tree is compacted: the level arrays are rebuilt with only nodes reachable from the root, in a canonical order that enables DAG deduplication (identical subtrees share the same node index).

---

## Chunk Tree

A depth-4 tree covering 256³ voxels. Chunk depth is fixed — depth-4 is the only chunk size used at every LOD level. Positions within a chunk are always a three-component byte coordinate (each component 0–255), keeping chunk-local math cheap and free of range checks. World-space positions use 64-bit signed integers only when crossing chunk boundaries.

Leaf values in a chunk tree are indices into the chunk's **material table** — a deduplicated list of 32-bit voxel values stored alongside the tree. Index 0 is reserved for air and is never stored in the table; actual material indices start at 1. A chunk with a single material type needs only 1-bit values throughout the entire tree.

The material table lives on the CPU. On the GPU, material values are inlined directly into the chunk data buffer at serialization time and do not require a separate indirection step during traversal.

---

## World Tree

A depth-28 tree covering a 2^64 voxel world. Leaf values are chunk pool handles (indices into the pool). Handle 0 is reserved for empty; active handles start at 1. No material table — handle values go directly into the tree.

Tree depth encodes LOD: a leaf at depth 28 is a LOD-0 chunk (256³ voxels, 1³ per leaf voxel). A leaf at depth 20 is a LOD-8 chunk (same depth-4 structure, but each leaf voxel covers 4^8 original voxels). The world tree is maintained by the CPU LOD system using a points-of-interest walk; see the LOD section.

GPU ray traversal descends the world tree using the same DDA and ancestor stack as per-chunk traversal. On hitting a leaf node, the shader reads the chunk handle, looks up its byte offset in the chunk offset table, and continues traversal in that chunk's tree. Both trees use the same flat SoA buffer format, so no structural switch is needed mid-ray.

---

## Tree Construction

Trees start empty and grow entirely through edits. There is no separate build path — initial generation, player edits, LOD aggregation, and voxel imports all go through the same edit system. A voxel import is just an empty tree with the imported voxels applied as a single sorted edit packet.

---

## World

The world owns a world tree (depth-28), a chunk pool (flat pool of loaded chunks indexed by handle), a shape edit list, and a map of persistent chunks.

**Ephemeral chunks** are generated on demand from the shape edit list and discarded when out of range. They are not stored on disk.

**Persistent chunks** are chunks that have received player voxel edits. The first player edit to a chunk triggers its creation: the full shape edit content is baked at that point, the edit is applied on top, and the resulting tree is stored permanently. The tree is the ground truth — the shape edit list is not re-run after baking. Subsequent edits go directly to the stored tree.

A persistent chunk is either **active** (currently in the pool at full LOD-0 resolution) or **resident** (held in CPU memory, out of LOD-0 range). When the camera moves away and the chunk coarsens, it is moved from the pool into resident storage; a derived LOD chunk takes the pool slot. When the camera returns, the resident chunk is promoted back to active. The resident data is always the ground truth; the LOD-derived chunk is ephemeral.

Shape edits are stored as a global ordered list. Each entry stores a tight axis-aligned bounding box in world voxel coordinates, a minimum LOD level, and a shape. When generating a chunk, the list is filtered by AABB overlap before invoking any shape logic — for flat terrain whose AABB has a small Y range, sky chunks are rejected with a single comparison. For large edit counts, the list can be indexed with a BVH for O(log n) per-chunk queries.

When a new shape edit is added and its AABB overlaps an existing persistent chunk, it is applied immediately to that chunk's tree. Player voxels outside the shape's coverage are untouched. The edit is also appended to the shape edit list so future ephemeral chunk generation includes it.

---

## Terrain

Procedural terrain is generated chunk by chunk on demand. The solid collapse optimization makes this extremely efficient: below the surface is a large uniform solid region, above is air — both terminate high in the tree. Only the thin surface layer needs full leaf resolution.

Terrain is generated via noise-based heightmaps with erosion and layered geology. The shape edit API drives generation: for each chunk, test columns against the heightmap using the AABB walk, collapsing solid and empty regions immediately without ever expanding them to individual voxels.

Material transitions between geology layers use a dithered boundary: instead of a hard horizontal cut, each voxel samples a hash of its world position against a blend factor to decide which material it belongs to. This produces natural-looking stochastic transitions at zero memory cost.

Terrain features like caves, boulders, and overhangs are driven by the same shape API, making them entries in the shape edit list rather than special cases. Each shape is LOD-aware: at coarse levels, the coverage implementation skips noise layers and detail passes whose contribution would be sub-voxel at that scale.

---

## LOD

LOD is a cascade of chunks where every LOD level uses the same depth-4 structure. Every chunk in memory is identical in format regardless of LOD level. What changes between LOD levels is only the physical size of each leaf voxel. This mirrors the approach in [Aokana (2505.02017)](https://arxiv.org/abs/2505.02017), which uses uniform chunk resolution across all LOD levels, but with a 64-tree instead of an octree.

A LOD-0 chunk covers 256³ world voxels, each leaf = 1³ voxels. A LOD-1 chunk covers 1024³ world voxels, each leaf = 4³ voxels. LOD-2 covers 4096³, each leaf = 16³, and so on. Coverage grows as 256 × 4^k per side at LOD-k, reaching 2^64 at LOD-28.

The 4x scale factor per level (vs Aokana's 2x octree factor) means only 28 LOD levels are needed to span a 2^64 voxel world, instead of 56.

**Construction**: a LOD-k chunk is built by aggregating 64 LOD-(k-1) chunks, exactly as Aokana aggregates 8 octree chunks, but with 64 children instead of 8. The output material for each slot samples from the corresponding region of input chunks using a density threshold; the output value is the mode of non-empty inputs. For persistent chunks there is a shortcut: level-2 nodes (covering exactly 4³ original voxels) are lifted directly out of the existing tree without recomputation. For ephemeral chunks the shape edit list is re-run at LOD-k resolution, skipping sub-voxel detail.

**Coarsen and split**: LOD transitions are explicit world-tree operations. Coarsening takes 64 child handles, aggregates them into one new coarser chunk, frees the 64 old pool slots, and returns the new handle. Splitting takes one coarser handle, spawns 64 finer chunks initialized from the parent, frees the parent slot, and returns the 64 new handles. Each new finer chunk is marked for shape resolution to fill in sub-voxel detail the coarser chunk lacked.

**World tree LOD maintenance**: The CPU walks the world tree top-down each frame to enforce a simple invariant: the only non-leaf paths are those leading directly to an active point of interest. Every other occupied slot must be a leaf chunk. Any non-leaf child that no point of interest passes through is marked for consolidation via coarsening and replaced with a leaf entry.

A point of interest has a world position and a max depth. The camera is always a point of interest at max depth 28 (full LOD-0 resolution). Other game objects (spyglass targets, NPCs, etc.) can register as points of interest at whatever depth gives sufficient detail at their range.

---

## GPU Memory

The GPU needs four things: scene metadata (camera position, sun direction, etc.), the world tree, a chunk offset table, and a chunk data buffer.

### Chunk Data Buffer

A single large read-only storage buffer holding all chunk trees packed end-to-end. Each chunk occupies a variable-length region determined by its tree's content — an empty or near-empty chunk may be a handful of bytes; a dense chunk with many unique materials may be tens or hundreds of kilobytes.

The CPU maintains a **free list** of unoccupied byte ranges within the buffer, sorted by offset. When a chunk is added or grows after an edit:

1. Search the free list for the first block large enough to hold the chunk (first-fit). If none exists, grow the buffer.
2. Serialize the chunk directly into that block (see zero-copy section below).
3. Update the chunk's entry in the offset table.
4. If the chunk previously occupied a different block, return that block to the free list and immediately **coalesce** it with any adjacent free blocks to prevent fragmentation from accumulating as small unusable gaps.

When a chunk shrinks, it stays in place — no relocation, and the tail of its allocation remains part of its reserved range until the next coarsen/split or an explicit defrag pass.

When a chunk is removed, its range is returned to the free list with adjacent coalescing.

Over time, edits produce gaps. When fragmentation exceeds a threshold, a **defrag pass** runs entirely on the GPU: `copy_buffer_to_buffer` packs live chunks together, then the CPU updates the offset table to match the new positions. The world tree is untouched. Defrag runs at GPU memory bandwidth (200–400 GB/s on mid-range hardware), not PCIe bandwidth, so compacting several hundred megabytes takes a few milliseconds.

### Chunk Offset Table

A fixed-size GPU buffer, one u32 per pool handle. The entry for a given handle is the byte offset of that chunk's data in the chunk data buffer. Offsets are stored in units of 16-byte alignment, extending addressable range to 64 GB within a u32. Handle 0 is unused (reserved for empty). The world tree stores handles; the shader does one extra read — handle → offset — before descending into the chunk tree.

The separate offset table is what makes defrag cheap. When chunks move, only their offset table entries need updating. The world tree, which stores handles, is untouched.

### Zero-Copy Uploads

When a chunk is serialized to the GPU, it is written **directly into a mapped staging buffer** (a GPU-driver-maintained ring of CPU-writable memory). There is no intermediate heap allocation. The serialization walks the chunk's internal arrays and writes their bitpacked contents one word at a time into the staging buffer. The GPU driver then issues a DMA transfer from the staging buffer to the chunk data buffer in VRAM.

On a **discrete GPU without Resizable BAR**, the PCIe DMA transfer is unavoidable — some data must cross the bus. But there is no extra CPU-side copy: the staging ring eliminates the intermediate `Vec<u8>`. On **Apple Silicon or integrated GPUs** with unified memory, even the PCIe step disappears and the staging write is the final write.

### Chunk GPU Format

Each chunk in the buffer is laid out as a header followed by per-level data in top-down order (coarsest level first):

- **Header**: for each level, the node count, the bit width of the values array, and the bit width of the node children array. Also a byte offset to each level's data within the chunk's buffer region, and the inlined material table (full 32-bit voxel values, in material index order starting at index 1).
- **Per level**: occupancy masks, leaf masks, and children offsets as flat arrays of u32/u64. Then the values array (bitpacked) and node children array (bitpacked), interleaved or concatenated.

Because bit widths are powers of 2 (1, 2, 4, 8, 16, 32), elements never straddle a 32-bit word boundary. Extraction in the shader is a shift and mask — no branching on element boundaries, no cross-word handling. The shader reads the bit width from the chunk header once at chunk entry and uses it for all subsequent reads from that level.

### Shader Bindings

The shader accesses three GPU buffers:

- The world tree buffer (flat u32 array in the same SoA format as chunk trees)
- The chunk offset table (one u32 per handle)
- The chunk data buffer (all chunks packed end-to-end)

When a ray hits a leaf node in the world tree with handle `h`, it reads the offset table at index `h` and jumps to that offset in the chunk data buffer to continue traversal. One extra memory read per chunk boundary crossing.

---

## Resources

### References

| Reference | Why it matters |
|---|---|
| [Guide to sparse 64-trees](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) | The traversal algorithm. Ancestor stack, coarse occupancy, flipped coordinates. |
| [Aokana (2505.02017)](https://arxiv.org/abs/2505.02017) | Chunked SVDAG with LOD streaming. Validates the shallow-tree-per-chunk approach. |
| [Hybrid Voxel Formats (2410.14128)](https://arxiv.org/abs/2410.14128) | Systematic comparison of voxel storage formats and their tradeoffs. |
| [High Resolution SVDAGs](https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/highResolutionSparseVoxelDAGs.pdf) | Original SVDAG paper. Bottom-up DAG reduction, GPU traversal. |
| [Efficient Sparse Voxel Octrees](https://www.researchgate.net/publication/47645140_Efficient_Sparse_Voxel_Octrees) | Laine & Karras. Foundation for SVO traversal and beam optimization. |
| [Voxelis Bible](https://github.com/WildPixelGames/voxelis) | SVO-DAG deep dive: batching, CoW, SoA, LOD, hash consing. |
| [Amanatides & Woo DDA](http://www.cse.yorku.ca/~amana/research/grid.pdf) | The DDA algorithm for voxel ray traversal. |
| [Compressing color data for voxels (Dolonius 2017)](https://dl.acm.org/doi/10.1145/3023368.3023381) | DFS-order color arrays, block compression for SVDAG attributes. |
| [Fast and Gorgeous Erosion Filter](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html) | Per-point erosion filter (no simulation). Evaluates in isolation, LOD-friendly, outputs height + derivatives + ridge map. |

### Channels

| Channel | Focus |
|---|---|
| [Douglas Dwyer](https://www.youtube.com/@DouglasDwyer) | Octo voxel engine, Rust + WebGPU, path-traced GI |
| [John Lin (Voxely)](https://www.youtube.com/@johnlin) | Path-traced voxel sandbox, RTX |
| [Gabe Rundlett](https://www.youtube.com/@GabeRundlett) | C++ voxel engine, Daxa/Vulkan |
| [Ethan Gore](https://www.youtube.com/@EthanGore) | Voxel engine dev, binary greedy meshing |
| [VoxelRifts](https://www.youtube.com/@VoxelRifts) | Voxel programming explainers |
| [SimonDev](https://www.youtube.com/@simondev758) | Radiance Cascades intro |

### Projects

| Project | Description |
|---|---|
| [voxquant](https://github.com/) | glTF voxelizer, source of rasterization algorithms |
| [VoxelRT](https://github.com/dubiousconst282/VoxelRT) | Tree64, brickmap, XBrickMap benchmarks |
| [Voxelis](https://github.com/WildPixelGames/voxelis) | Rust SVO-DAG with batching, CoW, LOD |
| [Octo Engine](https://github.com/DouglasDwyer/octo-release) | Rust + WebGPU voxel engine |
| [tree64](https://github.com/expenses/tree64) | Rust sparse 64-tree with hashing |
| [HashDAG](https://github.com/Phyronnaz/HashDAG) | HashDAG reference implementation |
| [gvox](https://github.com/GabeRundlett/gvox) | Voxel format translation library |

### More

| Resource | Description |
|---|---|
| [Voxel.Wiki](https://voxel.wiki) | Community hub for voxel rendering resources |
| [Voxely.net blog](https://voxely.net/blog/) | John Lin's voxel engine design posts |
| [A Rundown on Brickmaps](https://uygarb.dev/posts/0003_brickmap_rundown/) | Brickmap/brickgrid explanation |
| [Radiance Cascades 3D (ShaderToy)](https://www.shadertoy.com/view/X3XfRM) | Surface-based 3D radiance cascades |
| [Branchless DDA (ShaderToy)](https://www.shadertoy.com/view/XdtcRM) | Clean branchless 3D DDA reference |
