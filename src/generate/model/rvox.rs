use super::Model;
use crate::chunk::Chunk;
use crate::util::types::{Aabb, ChunkId, WorldPos};
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"RVOX";
const VERSION: u32 = 3;

#[derive(Debug)]
pub enum RvoxError {
	Io(io::Error),
	BadMagic,
	UnsupportedVersion(u32),
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
		}
	}
}

impl std::error::Error for RvoxError {}

impl Model {
	pub fn save_rvox<W: Write>(&self, w: &mut W) -> Result<(), RvoxError> {
		w.write_all(MAGIC)?;
		write_le(w, VERSION)?;
		write_worldpos(w, self.bounds.min)?;
		write_worldpos(w, self.bounds.max)?;
		write_le(w, self.chunks.len() as u32)?;
		for (id, chunk) in &self.chunks {
			write_worldpos(w, id.origin)?;
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
		let version = read_le(r)?;
		if version != VERSION {
			return Err(RvoxError::UnsupportedVersion(version));
		}
		let min = read_worldpos(r)?;
		let max = read_worldpos(r)?;
		let chunk_count = read_le(r)?;

		let mut model = Model::empty(Aabb::new(min, max));
		for _ in 0..chunk_count {
			let origin = read_worldpos(r)?;
			let chunk = Chunk::read_bytes(r)?;
			model.chunks.insert(ChunkId::new(origin), chunk);
		}
		Ok(model)
	}
}

// Header fields are written little-endian. Chunk node/material bytes are
// bytemuck-cast and written native-endian, so files are only portable across
// little-endian hosts (fine in practice for x86 and aarch64).

fn write_worldpos<W: Write>(w: &mut W, p: WorldPos) -> Result<(), RvoxError> {
	write_le(w, p.x() as u32)?;
	write_le(w, p.y() as u32)?;
	write_le(w, p.z() as u32)?;
	Ok(())
}

fn read_worldpos<R: Read>(r: &mut R) -> Result<WorldPos, RvoxError> {
	let x = read_le(r)? as i32;
	let y = read_le(r)? as i32;
	let z = read_le(r)? as i32;
	Ok(WorldPos::new(x, y, z))
}

fn write_le<W: Write>(w: &mut W, v: u32) -> Result<(), RvoxError> {
	w.write_all(&v.to_le_bytes())?;
	Ok(())
}

fn read_le<R: Read>(r: &mut R) -> Result<u32, RvoxError> {
	let mut buf = [0u8; 4];
	r.read_exact(&mut buf)?;
	Ok(u32::from_le_bytes(buf))
}
