//! Models: precomputed instanced mip pyramids of chunks.
//!
//! Loaded from external formats (gltf/MagicaVoxel/Minecraft) at import time,
//! serialized to .rvox in a layout matching the GPU buffer, then stamped into
//! the world. A clipmap chunk that aligns 1:1 with one of a model's mip chunks
//! references it by handle (one copy in RAM and VRAM). Composite cells where
//! the model overlaps terrain or another stamp bake a fresh chunk.

pub mod import;
pub mod rvox;
