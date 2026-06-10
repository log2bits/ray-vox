use crate::chunk::Chunk;
use crate::chunk::build::{Sample, Source, VoxelSample};
use crate::chunk::material::Material;
use crate::chunk::sources::LocalEdit;
use crate::generate::Edit;
use crate::util::types::{Aabb, ChunkId, WorldPos};

pub struct Terrain {
	pub seed: u32,
	pub base_height: i32,
	pub total_amplitude: i32,
	pub octave_count: u8,
	pub base_period: i32,
	pub material: Material,
}

impl Terrain {
	pub fn new(
		seed: u32,
		base_height: i32,
		total_amplitude: i32,
		octave_count: u8,
		base_period: i32,
		material: Material,
	) -> Self {
		Self { seed, base_height, total_amplitude, octave_count, base_period, material }
	}

	pub fn local(&self, chunk: ChunkId) -> LocalTerrain {
		let voxel_size = chunk.lod.voxel_size();
		let mut active_octaves: u8 = 0;
		let mut amp = self.total_amplitude / 2;
		while active_octaves < self.octave_count && amp >= voxel_size {
			active_octaves += 1;
			amp /= 2;
		}
		LocalTerrain {
			seed: self.seed,
			chunk_origin: chunk.origin,
			voxel_size,
			base_height: self.base_height,
			base_period: self.base_period,
			total_amplitude: self.total_amplitude,
			active_octaves,
			material: self.material,
		}
	}
}

impl Edit for Terrain {
	fn bounds(&self) -> Aabb {
		Aabb::all()
	}

	fn make_local<'a>(&'a self, chunk_id: ChunkId) -> Option<Box<dyn LocalEdit + 'a>> {
		Some(Box::new(self.local(chunk_id)))
	}

	fn apply(&self, chunk_id: ChunkId, base: Chunk) -> Chunk {
		base.edit(&self.local(chunk_id))
	}
}

#[derive(Clone)]
pub struct LocalTerrain {
	seed: u32,
	chunk_origin: WorldPos,
	voxel_size: i32,
	base_height: i32,
	base_period: i32,
	total_amplitude: i32,
	active_octaves: u8,
	material: Material,
}

impl LocalTerrain {
	fn height(&self, wx: i32, wz: i32) -> i32 {
		let mut total = 0.0f32;
		let mut amp = (self.total_amplitude / 2) as f32;
		let mut period = self.base_period;
		for i in 0..self.active_octaves {
			total += amp * value_noise(self.seed.wrapping_add(i as u32), wx, wz, period);
			amp *= 0.5;
			period = (period / 2).max(1);
		}
		self.base_height + total as i32
	}

	fn height_bounds(&self, wx_lo: i32, wx_hi: i32, wz_lo: i32, wz_hi: i32) -> (i32, i32) {
		let corners = [
			self.height(wx_lo, wz_lo),
			self.height(wx_hi - 1, wz_lo),
			self.height(wx_lo, wz_hi - 1),
			self.height(wx_hi - 1, wz_hi - 1),
		];
		let mut min = corners[0];
		let mut max = corners[0];
		for &c in &corners[1..] {
			if c < min { min = c; }
			if c > max { max = c; }
		}
		let cell_side = (wx_hi - wx_lo).max(wz_hi - wz_lo);
		let mut slack = 0.0f32;
		let mut amp = (self.total_amplitude / 2) as f32;
		let mut period = self.base_period;
		for _ in 0..self.active_octaves {
			if period <= cell_side {
				slack += amp;
			}
			amp *= 0.5;
			period = (period / 2).max(1);
		}
		(min - slack as i32, max + slack as i32)
	}
}

impl Source for LocalTerrain {
	#[inline]
	fn classify(&self, lo: [i32; 3], hi: [i32; 3], _depth: u8) -> Sample {
		let wx_lo = self.chunk_origin.x() + lo[0] * self.voxel_size;
		let wx_hi = self.chunk_origin.x() + hi[0] * self.voxel_size;
		let wy_lo = self.chunk_origin.y() + lo[1] * self.voxel_size;
		let wy_hi = self.chunk_origin.y() + hi[1] * self.voxel_size;
		let wz_lo = self.chunk_origin.z() + lo[2] * self.voxel_size;
		let wz_hi = self.chunk_origin.z() + hi[2] * self.voxel_size;
		let (h_min, h_max) = self.height_bounds(wx_lo, wx_hi, wz_lo, wz_hi);
		if wy_hi <= h_min {
			return Sample::Fill(self.material);
		}
		if wy_lo >= h_max {
			return Sample::Passthrough;
		}
		Sample::Subdivide
	}

	#[inline]
	fn voxel(&self, v: [i32; 3]) -> VoxelSample {
		let wx = self.chunk_origin.x() + v[0] * self.voxel_size;
		let wy = self.chunk_origin.y() + v[1] * self.voxel_size;
		let wz = self.chunk_origin.z() + v[2] * self.voxel_size;
		if wy < self.height(wx, wz) {
			VoxelSample::Fill(self.material)
		} else {
			VoxelSample::Passthrough
		}
	}
}

crate::impl_local_edit!(LocalTerrain, |_s| {
	[[i32::MIN / 2; 3], [i32::MAX / 2; 3]]
});

#[inline]
fn value_noise(seed: u32, wx: i32, wz: i32, period: i32) -> f32 {
	let p = period.max(1);
	let ix = wx.div_euclid(p);
	let iz = wz.div_euclid(p);
	let fx = wx.rem_euclid(p) as f32 / p as f32;
	let fz = wz.rem_euclid(p) as f32 / p as f32;
	let v00 = lattice(seed, ix, iz);
	let v10 = lattice(seed, ix + 1, iz);
	let v01 = lattice(seed, ix, iz + 1);
	let v11 = lattice(seed, ix + 1, iz + 1);
	let sx = smoothstep(fx);
	let sz = smoothstep(fz);
	lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sz)
}

#[inline]
fn lattice(seed: u32, x: i32, z: i32) -> f32 {
	let mut h = seed;
	h = h.wrapping_add((x as u32).wrapping_mul(0x9E3779B9));
	h ^= h >> 16;
	h = h.wrapping_mul(0x85EBCA6B);
	h = h.wrapping_add((z as u32).wrapping_mul(0xC2B2AE35));
	h ^= h >> 13;
	h = h.wrapping_mul(0xC2B2AE35);
	h ^= h >> 16;
	(h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[inline]
fn smoothstep(t: f32) -> f32 {
	t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
	a + (b - a) * t
}
