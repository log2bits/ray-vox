use super::Model;
use crate::chunk::Chunk;
use crate::util::types::{Aabb, ChunkId, LodLevel, WorldPos};
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"RVOX";
const VERSION: u32 = 2;

#[derive(Debug)]
pub enum RvoxError {
	Io(io::Error),
	BadMagic,
	UnsupportedVersion(u32),
	Truncated,
}

impl From<io::Error> for RvoxError {
	fn from(e: io::Error) -> Self { RvoxError::Io(e) }
}

impl std::fmt::Display for RvoxError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			RvoxError::Io(e) => write!(f, "rvox io error: {}", e),
			RvoxError::BadMagic => write!(f, "rvox file does not start with 'RVOX'"),
			RvoxError::UnsupportedVersion(v) => write!(f, "rvox version {} not supported", v),
			RvoxError::Truncated => write!(f, "rvox file truncated"),
		}
	}
}

impl std::error::Error for RvoxError {}

impl Model {
	pub fn save_rvox<W: Write>(&self, w: &mut W) -> Result<(), RvoxError> {
		w.write_all(MAGIC)?;
		write_u32(w, VERSION)?;
		write_worldpos(w, self.bounds.min)?;
		write_worldpos(w, self.bounds.max)?;
		write_u32(w, self.chunks.len() as u32)?;
		for (id, chunk) in &self.chunks {
			write_chunk_id(w, *id)?;
			chunk.write_bytes(w)?;
		}
		Ok(())
	}

	pub fn load_rvox<R: Read>(r: &mut R) -> Result<Model, RvoxError> {
		let mut magic = [0u8; 4];
		r.read_exact(&mut magic)?;
		if &magic != MAGIC {
			return Err(RvoxError::BadMagic);
		}
		let version = read_u32(r)?;
		if version != VERSION {
			return Err(RvoxError::UnsupportedVersion(version));
		}
		let min = read_worldpos(r)?;
		let max = read_worldpos(r)?;
		let chunk_count = read_u32(r)?;

		let mut model = Model::empty(Aabb::new(min, max));
		for _ in 0..chunk_count {
			let id = read_chunk_id(r)?;
			let chunk = Chunk::read_bytes(r)?;
			model.chunks.insert(id, chunk);
		}
		Ok(model)
	}
}

fn write_chunk_id<W: Write>(w: &mut W, id: ChunkId) -> Result<(), RvoxError> {
	write_worldpos(w, id.origin)?;
	w.write_all(&[u8::from(id.lod), 0, 0, 0])?;
	Ok(())
}

fn read_chunk_id<R: Read>(r: &mut R) -> Result<ChunkId, RvoxError> {
	let origin = read_worldpos(r)?;
	let mut lod_bytes = [0u8; 4];
	r.read_exact(&mut lod_bytes)?;
	Ok(ChunkId::new(origin, LodLevel::new(lod_bytes[0])))
}

fn write_worldpos<W: Write>(w: &mut W, p: WorldPos) -> Result<(), RvoxError> {
	write_i32(w, p.x())?;
	write_i32(w, p.y())?;
	write_i32(w, p.z())?;
	Ok(())
}

fn read_worldpos<R: Read>(r: &mut R) -> Result<WorldPos, RvoxError> {
	let x = read_i32(r)?;
	let y = read_i32(r)?;
	let z = read_i32(r)?;
	Ok(WorldPos::new(x, y, z))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), RvoxError> {
	w.write_all(&v.to_le_bytes())?;
	Ok(())
}

fn write_i32<W: Write>(w: &mut W, v: i32) -> Result<(), RvoxError> {
	w.write_all(&v.to_le_bytes())?;
	Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, RvoxError> {
	let mut buf = [0u8; 4];
	r.read_exact(&mut buf)?;
	Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, RvoxError> {
	let mut buf = [0u8; 4];
	r.read_exact(&mut buf)?;
	Ok(i32::from_le_bytes(buf))
}
