//! Voxelize external formats (gltf, MagicaVoxel .vox, Minecraft .mca) into a
//! Model. Each importer produces the finest-LOD chunks; the mip pyramid is
//! built once via merge_lod and saved alongside in the .rvox.
