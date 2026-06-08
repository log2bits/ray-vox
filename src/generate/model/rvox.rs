//! .rvox file format: serialized Model with layout matching the GPU buffer
//! representation, so loads upload with minimal transformation. Versioned
//! header + per-LOD chunk arrays. Chunk-local offsets are position-independent,
//! so chunks relocate cleanly into any GPU buffer offset.
