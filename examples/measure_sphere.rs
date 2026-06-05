// Measure gpu_size_bytes + indices breakdown for the big sphere.
use ray_vox::Chunk;
use ray_vox::chunk::edit::EditPacket;
use ray_vox::chunk::material::Material;
use ray_vox::generate::volume::sphere::Sphere;
use ray_vox::util::types::{ChunkId, LodLevel, WorldPos};

fn main() {
	let stone = Material::from(0x80808040);
	let chunk_id = ChunkId::new(WorldPos::new(0, 0, 0), LodLevel::FINEST);

	let cases: &[(&str, i32, Material)] = &[
		("sphere r=128 place", 128, stone),
		("sphere r=64 place",   64, stone),
	];

	for (name, r, m) in cases {
		let packet = Sphere::generate(*r, chunk_id, WorldPos::new(128, 128, 128), *m);
		let mut mc = Chunk::new().into_mutable();
		mc.queue_edit(packet);
		let c = mc.bake();
		let interior_bytes = c.interior_nodes.len() * std::mem::size_of_val(&c.interior_nodes[0]);
		let leaf_bytes = c.leaf_nodes.len() * std::mem::size_of_val(&c.leaf_nodes[0]);
		let lut_bytes = c.materials.lut.values.len() * std::mem::size_of::<Material>();
		let idx_bytes = c.materials.indices.words.len() * 4;
		let idx_count = c.materials.indices.len();
		let bit_width = c.materials.indices.bits;
		println!(
			"{name}: gpu={} interiors={}({}b) leaves={}({}b) lut={}({}b) indices={}({}b @ {}bit) stored_vol={}",
			c.gpu_size_bytes(),
			c.interior_nodes.len(), interior_bytes,
			c.leaf_nodes.len(), leaf_bytes,
			c.materials.lut.values.len(), lut_bytes,
			idx_count, idx_bytes, bit_width,
			c.stored_volume(),
		);
	}
}
