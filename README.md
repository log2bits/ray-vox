# ray-vox

A voxel renderer that ray traces everything (no rasterization). Rust + WebGPU.

The core data structure is a compact bitpacked sparse tree, one per chunk, arranged in a fixed 3D grid on the GPU. I call it a CBEPSV64:

- C, chunked. Independent trees arranged in a fixed 3D grid.
- B, bitpacked. Every value uses only as many bits as it needs.
- E, [efficient](https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf). Leaf nodes allowed anywhere in the tree.
- P, [pointerless](https://www.cai.sk/ojs/index.php/cai/article/view/2020_3_587). Popcount-implicit offsets, no per-cell pointers.
- S, sparse. Empty space costs nothing.
- V, voxel.
- 64, 4^3 branching [tetrahexacontree](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/).

## Priorities

1. Fast ray traversal, no hardware RT.
2. Heavy compression.
3. Quick CPU-side edits.

# Implemented

## Chunks

Each node has two 64-bit masks, `has_child` and `is_leaf`. Together they pick one of four cell states:

| has_child | is_leaf | meaning                                                |
|:---------:|:-------:|--------------------------------------------------------|
|     0     |    0    | empty                                                  |
|     0     |    1    | filled, one material stored inline                     |
|     1     |    0    | interior, has a child node with more branching         |
|     1     |    1    | leaf, has a child node holding voxel materials         |

A cell is occupied if either bit is set.

- Every chunk is depth 4. 256^3 voxels each, chunk-local coords fit in `[u8; 3]`.
- AoS layout. Interior nodes are 24 bytes, leaf nodes are 12 bytes. One fetch per visit, no SoA scatter across five arrays.
- Interior and leaf nodes live in separate arrays. The parent addresses both with a single packed pointer field.
- Per-chunk material LUT. Bit width scales from 1 bit (2 materials) up to 32 bits (unique per voxel). LUT entries are full voxel values, so the hot traversal path only touches small indices.
- A chunk with no nodes and an empty materials array is entirely air. A chunk with no nodes and exactly one material entry is entirely that material. Both are valid states requiring no tree traversal.

## Interior Node (24 bytes)

```
has_child   u64   // bit i set: cell i has a child node (interior or leaf)
is_leaf     u64   // bit i set: cell i is a leaf
ptrs        u32   // [12..0]  interior_ptr  (13 bits, into interior array)
                  // [31..13] leaf_ptr      (19 bits, into leaf array)
mat_offset  u32   // bit offset into chunk's materials array (filled cells only)
```

`popcount_below(m, slot)` means `popcount(m & ((1 << slot) - 1))`.

Looking up a child at a given slot:

| state                                  | location                                                                |
|----------------------------------------|-------------------------------------------------------------------------|
| interior (has_child=1, is_leaf=0)      | `interior_ptr + popcount_below(has_child & !is_leaf, slot)`             |
| leaf (has_child=1, is_leaf=1)          | `leaf_ptr + popcount_below(has_child & is_leaf, slot)`                  |
| filled (has_child=0, is_leaf=1)        | `mat_offset + popcount_below(!has_child & is_leaf, slot) * mat_width`   |
| empty (has_child=0, is_leaf=0)         | nothing stored                                                          |

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

- One mask is enough. A leaf node's cells are voxels, so the mask is just occupancy.
- Cell `i` reads from: `mat_offset + popcount_below(occ_mask, i) * mat_width`.
- Leaves only exist for partially-occupied regions where cells actually differ.
- Uniform regions are stored as a `filled` cell on the parent instead. No node, one material entry.

## Material Array (per chunk)

```
[K palette entries x u32]    // full voxel values
[bitpacked indices]          // mat_width bits each, popcount-indexed
```

`mat_width = ceil(log2(K))`. Exact bit widths minimise memory and maximise cache usage:

| K (unique materials) | bits per slot |
|:--------------------:|:-------------:|
|         2            |       1       |
|         3..4         |       2       |
|         5..8         |       3       |
|         9..16        |       4       |
|         17..32       |       5       |
|         ...          |      ...      |
|       65537+         |      32       |

- Word index and bit offset use shifts and masks, never division: `word = bit_pos >> 5`, `offset = bit_pos & 31`.
- Entries can straddle a 32-bit word boundary. Extraction always loads two words and masks unconditionally, keeping the code branchless and warp-coherent.
- Interior and leaf `mat_offset` fields write into the same shared per-chunk bitpacked array.

### Material-run dedup

Every interior node emits one run of materials (its filled cells), and every leaf emits one run (its occupied cells). Runs frequently repeat: uniform regions of a single material produce the same run over and over.

Two-tier dedup at emit time:

1. **Full-run exact match** via a hash-and-verify index. If an identical run was emitted before, the new node points at the same offset.
2. **Overlap-extend.** If the tail of the materials array matches the head of the new run, only the missing suffix is appended and the offset points back into the overlap.

Collapses the materials slab to near-zero for uniform-material regions — ~99% reduction on a single-material r=128 sphere.

## Editing

- Chunks are edited CPU-side. Once uploaded to the GPU they are immutable — a re-edit rebuilds the affected chunks on the CPU and re-uploads.
- Two edit kinds:
  - **Single-voxel**: a `(path, material)` pair. Path is u32, four 6-bit slot indices; a slot of 0 terminates early.
  - **Volume**: a shape with a world-space AABB and a per-cell `classify(lo, hi, depth) → Passthrough | Fill(m) | Subdivide`. Cost is resolution-adaptive: a big uniform region collapses to one Fill without visiting every voxel it covers. A filled sphere covering millions of voxels only recurses at the boundary.
- Multiple edits compose as layered sources: last write wins, and lower layers show through where an upper layer says Passthrough.
- Voxel value 0 means air or delete.

## Voxel Format (u32)

| Bits  | Field          | Notes                                     |
|-------|----------------|-------------------------------------------|
| 31..8 | RGB            | 24-bit linear RGB, tints the material     |
| 7..4  | Material index | 0-15, indexes into the PBR material table |
| 3..0  | Unused         |                                           |

- Voxel value 0 is air. Zero state of every bitpacked array, costs nothing to store.
- RGB is a tint on top of the material's behavior, not a raw surface color. Two voxels with the same material index but different RGB values give different shades of the same physical behavior. The material table defines how light interacts with the surface; the voxel's RGB just colors it.

## Material Table

16 entries, 8 bytes each, 128 bytes total. Fits permanently in L1.

| Parameter            | Bits | Notes                                       |
|----------------------|------|---------------------------------------------|
| Roughness + metallic | 8    | High bit = metallic flag, low 7 = roughness |
| Emissive strength    | 8    |                                             |
| Scattering coeff     | 8    | How often rays scatter inside the medium    |
| Absorption RGB       | 24   | 8 bits per channel                          |
| IOR                  | 8    | Index of refraction                         |
| Anisotropy g         | 8    | Phase function for volumetric scatter       |

- Roughness and metallic share one byte. High bit is metallic, low 7 bits are roughness (0 = mirror, 127 = fully diffuse).
- Scattering and absorption describe participating media. Scattering controls how often rays bounce inside the medium. Absorption controls how much light is lost per unit distance and at which wavelengths, which is what gives deep water its blue-green tint.
- Anisotropy g is the Henyey-Greenstein phase function parameter. g = 0 scatters evenly in all directions (fog), g > 0 scatters forward (clouds). Clouds without forward scattering look flat and wrong — the bright halo around clouds when the sun is behind them comes entirely from this.
- Non-volumetric materials zero out scattering, absorption, and g. They're just ignored.
- 4 bits are left unused in the voxel format, reserved for future use.

## Voxel Import: MagicaVoxel (.vox)

- Direct translation. Materials and palette map across without much fuss.
- Compiled offline into `.rvox`, a compact binary layout matching the CPU chunk representation. The renderer only ever loads `.rvox`.

# Planned

## Renderer

Fixed-grid multi-chunk ray tracing. One storage buffer of concatenated chunk bytes, one directory buffer mapping grid position to chunk offset, one WGSL kernel that walks the tree per chunk and steps between chunks via directory lookups.

### Ray Tracing

- DDA with an ancestor stack. Neighbor steps pop or push the stack instead of restarting from the chunk root.
- O(1) common ancestor via diffbits. XOR old and new position, clz, index the stack.
- 2^3 coarse occupancy grouping skips 8-cell empty regions in one mask test.
- Ray-octant mirroring bakes direction sign into the coord system. No per-step direction branches.
- Fractional coord system in `[1.0, 2.0)`. Cell index and floor-to-scale are float mantissa bit ops.
- After each step, position clamps to the neighbor bounding box. No bias offset.
- Ancestor stack in workgroup shared memory. Lower register pressure, higher occupancy.
- Camera is integer grid coords plus a chunk-local f32 offset. f32 is enough for all traversal math.

### Per-Face Lighting

Lighting is cached per voxel face in a GPU hashmap. Three passes, so reads and writes can't race and shadow rays can batch:

1. Traversal. Rays write a face ID per pixel into the G-buffer. No hashmap access yet.
2. Lookup. Face IDs get deduped (many pixels share a face). Each unique face probes the hashmap. Hits return cached light, misses queue shadow rays. A screen-space temporal layer also compares the current face ID against last frame's at the same pixel. For static geometry this catches most lookups before the hashmap ever gets touched.
3. Shadow rays and writeback. Uncached faces dispatch shadow rays. Results write back to the hashmap.

Final shading:

```
final_color = albedo * rgb
```

where `rgb` is the total blended incident light from everything cached in the hashmap value.

#### Lighting Model

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

#### Hashmap

- About 1 to 2 million slots, 16 bytes per slot, around 32 MB total.
- Load factor about 0.5.

Key (8 bytes, u64). Stable face identifier, never mutated:

```
[63..48]  chunk_id        u16    // grid-encoded chunk position
[47..16]  path            u8;4   // 6-bit slot + 2 unused per level
[15..8]   face_direction  u8     // bits 7..5: direction 0..5, bits 4..0: unused
[7..0]    reserved        u8
```

Value (8 bytes):

```
rgb            u8;3   // blended incident light from all sources
generation     u8     // compared against a global frame counter, stale entries don't need explicit eviction
emissive_ids   u8;4   // indices into the emissive pool (0 means empty)
```

#### Emissive Pool

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

## World Clipmap (with LOD)

Beyond the fixed-grid renderer, the world grows to a clipmap so far-away chunks are cheap and the addressable space is unbounded.

- 11 levels (0–10) covering the i32 coord space.
- Level 0 is the coarsest, level 10 is the finest. Same convention as the chunk tree.
- All levels are 8^3 chunks.
- `chunk_size(d) = 256 * 4^(10 - d)`. At depth 0: 2^28 per chunk, 8 * 2^28 = 2^31 total.
- Coarsest origin at `-(1 << 30)`. Fixed in world space, never moves.
- Finer levels snap to their own grid. Origin = camera rounded to `chunk_size(d)`, minus `4 * chunk_size(d)`.
- Levels overlap rather than partition. Finer level wins at traversal.
- 4× LOD scaling. Matches the 64-tree's branching: coarser LOD is the same chunk read one depth shallower.
- Storage: ~11 KB of chunk handles plus 704 bytes occupancy bitmask.
- Chunk handles are u16. Max 5632 chunks across all depths, well under the 65K cap.

### Chunk Handle (u16)

```
[15..13]  unused
[12..9]   depth    4 bits  (0..10)
[8..6]    x        3 bits  (0..7)
[5..3]    y        3 bits  (0..7)
[2..0]    z        3 bits  (0..7)
```

World-space recovery is O(1), no lookup table:

```
chunk_world = clipmap_origin[depth] + (x, y, z) * chunk_size[depth]
voxel_world = chunk_world + decode_path(path) * voxel_size[depth]
```

- Handle encodes the full clipmap position directly.
- Any face ID or emissive ID can pull its world position back out.
- Direction vectors between two encoded positions are just a subtraction.

## GPU Memory

Once the chunk grid isn't fixed at load time, memory management gets a proper pool:

- Chunks live in a pool with a free list, since chunk size varies a lot.
- Chunk offset table maps the u16 handle to the actual memory offset.
- Edits only re-upload the affected chunk.
- CPU side stores chunks as typed structs (`Vec<InteriorNode>`, `Vec<LeafNode>`, material array).
- GPU side is one flat contiguous buffer. Upload serializes the scattered heap allocations into the packed GPU layout.
- On discrete GPU: CPU → staging buffer → VRAM via PCIe.
- On unified memory: CPU → GPU-visible buffer directly, driver elides the transfer.
- wgpu's `StagingBelt` abstracts both paths — same code runs optimally on all hardware.
- Node types are `bytemuck::Pod` so serialization is raw `memcpy` with no per-field encoding.
- Dirty tracking: only chunks modified since last frame are re-uploaded.

## Editing After Upload

Today edits are CPU-only. Planned: copy-on-write rebake of only the affected chunks, partial re-upload via the chunk pool's dirty set. Interior LOD material tracking may come back if a clipmap needs it for coarse levels.

## Model Stamping

Models are precomputed voxelized assets, stored once and stamped many times.

- A full mip pyramid is built at import time and stored alongside the finest level. The pyramid is ~1.6% larger than the finest level alone, since each level has 1/64 the chunks of the one below.
- Stamping a model places it in the world at a chunk-grid-snapped translation. Where a world chunk aligns 1:1 with one of the model's mip chunks, the world cell references the model's chunk by handle (one copy in RAM and VRAM, the instancing win). Where the model overlaps terrain or another stamp, the cell bakes a composite chunk.
- Editing a stamped instance triggers copy-on-write on the affected chunks for that instance.
- Placement is translation + chunk-snap only. Arbitrary rotation would lose the instance (different voxel grid per placement); 90° rotations are possible via separately cached rotated variants.

## More Voxel Import Formats

### Minecraft (.mca)

- Each block ID becomes one material in the chunk LUT.
- Each block is 16^3 voxels of one material, which lines up exactly with one Level-1 cell.
- Block face textures live in a GPU-side table. On ray hit, the GPU resolves color from the per-block face texture table.
- Different faces can return different colors.
- Each face texture has its own color LUT. Fewer than 256 unique colors per block, so each voxel takes 1 byte at hit time.
- Compresses tighter than `.mca` itself thanks to bitpacking and the 16^3 structural alignment.
- Catch: non-cube models (lecterns, stairs, fences) get voxelized. Some end up looking off.

### glTF (.gltf, .glb)

- Bin triangles into chunks.
- Triangle/voxel intersection per voxel.
- Configurable voxel scale and palette.
- PBR material extraction.

# Resources

## References

| Reference | Why it's here |
|---|---|
| [Sparse 64-trees guide](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) | Traversal algorithm: ancestor stack, coarse occupancy, octant mirroring, mantissa tricks |
| [Aokana (2505.02017)](https://arxiv.org/abs/2505.02017) | Chunked SVDAG with LOD streaming, validates uniform-resolution chunks |
| [Hybrid Voxel Formats (2410.14128)](https://arxiv.org/abs/2410.14128) | Systematic comparison of voxel storage formats |
| [High-Resolution SVDAGs](https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/highResolutionSparseVoxelDAGs.pdf) | The original SVDAG paper |
| [ESVO](https://www.researchgate.net/publication/47645140_Efficient_Sparse_Voxel_Octrees) | Laine and Karras, SVO traversal and beam optimization |
| [Voxelis Bible](https://github.com/WildPixelGames/voxelis) | SVO-DAG deep dive: batching, CoW, SoA, LOD, hash consing |
| [Amanatides and Woo DDA](http://www.cse.yorku.ca/~amana/research/grid.pdf) | DDA for grid traversal |
| [Erosion filter](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html) | LOD-friendly per-point erosion |

## Channels

| Channel | Focus |
|---|---|
| [Douglas Dwyer](https://www.youtube.com/@DouglasDwyer) | Octo engine, Rust + WebGPU, path-traced GI |
| [John Lin (Voxely)](https://www.youtube.com/@johnlin) | Path-traced voxel sandbox, per-face lighting |
| [Gabe Rundlett](https://www.youtube.com/@GabeRundlett) | C++ voxel engine, Daxa/Vulkan |
| [Ethan Gore](https://www.youtube.com/@EthanGore) | Engine dev, binary greedy meshing |
| [VoxelRifts](https://www.youtube.com/@VoxelRifts) | Voxel programming explainers |
| [SimonDev](https://www.youtube.com/@simondev758) | Radiance cascades |

## Projects

| Project | Description |
|---|---|
| [VoxelRT](https://github.com/dubiousconst282/VoxelRT) | Tree64, brickmap, XBrickMap benchmarks |
| [Voxelis](https://github.com/WildPixelGames/voxelis) | Rust SVO-DAG with batching, CoW, LOD |
| [Octo Engine](https://github.com/DouglasDwyer/octo-release) | Rust + WebGPU voxel engine |
| [tree64](https://github.com/expenses/tree64) | Rust sparse 64-tree |
| [HashDAG](https://github.com/Phyronnaz/HashDAG) | Reference implementation |
| [gvox](https://github.com/GabeRundlett/gvox) | Voxel format translation library |

## More

| Resource | Description |
|---|---|
| [Voxel.Wiki](https://voxel.wiki) | Community hub |
| [Voxely.net blog](https://voxely.net/blog/) | John Lin's design posts |
| [Brickmap rundown](https://uygarb.dev/posts/0003_brickmap_rundown/) | Brickmap and brickgrid explanation |
| [Branchless DDA (ShaderToy)](https://www.shadertoy.com/view/XdtcRM) | Branchless 3D DDA reference |
