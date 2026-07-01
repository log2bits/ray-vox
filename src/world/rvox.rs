use super::World;
use crate::chunk::Chunk;
use crate::util::types::WorldPos;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"RVOX";
const VERSION: u32 = 4;

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

// The file format is:
//   MAGIC (4)
//   VERSION (4)
//   chunk_grid_dim (3 x u32)
//   world_origin (3 x i32)
//   non_empty_chunk_count (u32)
//   for each non-empty chunk:
//     grid_pos (3 x u32)
//     chunk bytes (see Chunk::write_bytes)
//
// Header fields are little-endian. Chunk node and material bytes are
// bytemuck-cast native-endian, so files are only portable across
// little-endian hosts (fine in practice for x86 and aarch64).
impl World {
	pub fn save_rvox<W: Write>(&self, w: &mut W) -> Result<(), RvoxError> {
		w.write_all(MAGIC)?;
		write_u32(w, VERSION)?;
		for axis in 0..3 {
			write_u32(w, self.chunk_grid_dim[axis])?;
		}
		write_worldpos(w, self.origin)?;
		let non_empty = self.chunks.iter().filter(|c| c.is_some()).count() as u32;
		write_u32(w, non_empty)?;
		for (index, slot) in self.chunks.iter().enumerate() {
			let Some(chunk) = slot else { continue };
			let grid_pos = self.slot_grid_pos(index);
			for axis in 0..3 {
				write_u32(w, grid_pos[axis])?;
			}
			chunk.write_bytes(w)?;
		}
		Ok(())
	}

	pub fn load_rvox<R: Read>(r: &mut R) -> Result<World, RvoxError> {
		let mut magic = [0u8; 4];
		r.read_exact(&mut magic)?;
		if &magic != MAGIC {
			return Err(RvoxError::BadMagic);
		}
		let version = read_u32(r)?;
		if version != VERSION {
			return Err(RvoxError::UnsupportedVersion(version));
		}
		let chunk_grid_dim = [read_u32(r)?, read_u32(r)?, read_u32(r)?];
		let world_origin = read_worldpos(r)?;
		let chunk_count = read_u32(r)?;

		let mut world = World::with_origin(chunk_grid_dim, world_origin);
		for _ in 0..chunk_count {
			let grid_pos = [read_u32(r)?, read_u32(r)?, read_u32(r)?];
			let chunk = Chunk::read_bytes(r)?;
			world.set_chunk(grid_pos, chunk);
		}
		Ok(world)
	}
}

fn write_worldpos<W: Write>(w: &mut W, p: WorldPos) -> Result<(), RvoxError> {
	write_u32(w, p.x() as u32)?;
	write_u32(w, p.y() as u32)?;
	write_u32(w, p.z() as u32)?;
	Ok(())
}

fn read_worldpos<R: Read>(r: &mut R) -> Result<WorldPos, RvoxError> {
	let x = read_u32(r)? as i32;
	let y = read_u32(r)? as i32;
	let z = read_u32(r)? as i32;
	Ok(WorldPos::new(x, y, z))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), RvoxError> {
	w.write_all(&v.to_le_bytes())?;
	Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, RvoxError> {
	let mut buf = [0u8; 4];
	r.read_exact(&mut buf)?;
	Ok(u32::from_le_bytes(buf))
}
