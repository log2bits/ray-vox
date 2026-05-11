pub mod camera;
mod pipeline;
mod present;
mod serialize;
mod upload;

pub use camera::CameraPos;
pub use serialize::ChunkMeta;

use bytemuck::Zeroable;
use crate::chunk::Chunk;
use pipeline::RenderPipeline;
use serialize::serialize_chunk;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
	cam_pos:      [f32; 3],
	fov_half_tan: f32,
	cam_right:    [f32; 3],
	viewport_w:   f32,
	cam_up:       [f32; 3],
	viewport_h:   f32,
	cam_forward:  [f32; 3],
	_pad0:        f32,
	node_counts:  [u32; 4],
	slot_counts:  [u32; 4],
	level_offsets:[u32; 4],
	material_count:  u32,
	material_offset: u32,
	tree_occupied:   u32,
	tree_is_leaf:    u32,
	tree_leaf_value: u32,
	_pad1: [u32; 3],
}

pub struct Renderer {
	device:        wgpu::Device,
	queue:         wgpu::Queue,
	surface:       wgpu::Surface<'static>,
	config:        wgpu::SurfaceConfiguration,
	pipeline:      RenderPipeline,
	uniforms_buf:  wgpu::Buffer,
	chunk_buf:     wgpu::Buffer,
	bind_group:    wgpu::BindGroup,
	chunk_meta:    ChunkMeta,
	egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
	pub async fn new(window: std::sync::Arc<winit::window::Window>, chunk: &Chunk) -> Self {
		let size = window.inner_size();
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			..Default::default()
		});
		let surface = instance.create_surface(window).unwrap();
		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				compatible_surface: Some(&surface),
				..Default::default()
			})
			.await
			.expect("no adapter");
		let (device, queue) = adapter
			.request_device(&wgpu::DeviceDescriptor::default(), None)
			.await
			.expect("no device");

		let caps   = surface.get_capabilities(&adapter);
		let format = caps.formats[0];
		let config = wgpu::SurfaceConfiguration {
			usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
			format,
			width:        size.width.max(1),
			height:       size.height.max(1),
			present_mode: wgpu::PresentMode::AutoNoVsync,
			alpha_mode:   caps.alpha_modes[0],
			view_formats: vec![],
			desired_maximum_frame_latency: 2,
		};
		surface.configure(&device, &config);

		let pipeline = RenderPipeline::new(&device, format);

		let (chunk_data, meta) = serialize_chunk(chunk);
		let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label:    Some("chunk_data"),
			contents: bytemuck::cast_slice(&chunk_data),
			usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		});

		let uniforms = GpuUniforms::zeroed();
		let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label:    Some("uniforms"),
			contents: bytemuck::bytes_of(&uniforms),
			usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		});

		let bind_group = Self::make_bind_group(&device, &pipeline, &uniforms_buf, &chunk_buf);

		let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

		Self {
			device,
			queue,
			surface,
			config,
			pipeline,
			uniforms_buf,
			chunk_buf,
			bind_group,
			chunk_meta: meta,
			egui_renderer,
		}
	}

	pub fn vram_bytes(&self) -> u64 {
		self.uniforms_buf.size() + self.chunk_buf.size()
	}

	pub fn set_chunk(&mut self, chunk: &Chunk) {
		let (chunk_data, meta) = serialize_chunk(chunk);
		self.chunk_meta = meta;

		let needed = (chunk_data.len() * 4) as u64;
		if self.chunk_buf.size() < needed {
			self.chunk_buf = self.device.create_buffer_init(
				&wgpu::util::BufferInitDescriptor {
					label:    Some("chunk_data"),
					contents: bytemuck::cast_slice(&chunk_data),
					usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
				},
			);
			self.bind_group = Self::make_bind_group(
				&self.device,
				&self.pipeline,
				&self.uniforms_buf,
				&self.chunk_buf,
			);
		} else {
			self.queue.write_buffer(&self.chunk_buf, 0, bytemuck::cast_slice(&chunk_data));
		}
	}

	pub fn render(
		&mut self,
		camera:      &CameraPos,
		egui_ctx:    &egui::Context,
		egui_output: egui::FullOutput,
	) {
		let (w, h) = (self.config.width as f32, self.config.height as f32);
		let m = &self.chunk_meta;

		let uniforms = GpuUniforms {
			cam_pos:         camera.world_pos(),
			fov_half_tan:    (camera.fov_y * 0.5).tan(),
			cam_right:       camera.right(),
			viewport_w:      w,
			cam_up:          camera.up(),
			viewport_h:      h,
			cam_forward:     camera.forward(),
			_pad0:           0.0,
			node_counts:     m.node_counts,
			slot_counts:     m.slot_counts,
			level_offsets:   m.level_offsets,
			material_count:  m.material_count,
			material_offset: m.material_offset,
			tree_occupied:   m.tree_occupied,
			tree_is_leaf:    m.tree_is_leaf,
			tree_leaf_value: m.tree_leaf_value,
			_pad1:           [0; 3],
		};
		self.queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

		let frame = match self.surface.get_current_texture() {
			Ok(f) => f,
			Err(_) => return,
		};
		let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

		// Tessellate egui output
		let ppp = egui_output.pixels_per_point;
		let primitives = egui_ctx.tessellate(egui_output.shapes, ppp);
		let screen_desc = egui_wgpu::ScreenDescriptor {
			size_in_pixels: [self.config.width, self.config.height],
			pixels_per_point: ppp,
		};
		for (id, delta) in &egui_output.textures_delta.set {
			self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
		}

		let mut enc = self.device.create_command_encoder(
			&wgpu::CommandEncoderDescriptor { label: Some("frame") },
		);

		// Voxel pass
		{
			let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("voxels"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view:           &view,
					resolve_target: None,
					ops:            wgpu::Operations {
						load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.12, g: 0.15, b: 0.22, a: 1.0 }),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes:         None,
				occlusion_query_set:      None,
			});
			pass.set_pipeline(&self.pipeline.pipeline);
			pass.set_bind_group(0, &self.bind_group, &[]);
			pass.draw(0..3, 0..1);
		}

		// egui pass (renders on top)
		self.egui_renderer.update_buffers(&self.device, &self.queue, &mut enc, &primitives, &screen_desc);
		{
			let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("egui"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view:           &view,
					resolve_target: None,
					ops:            wgpu::Operations {
						load:  wgpu::LoadOp::Load,
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes:         None,
				occlusion_query_set:      None,
			});
			self.egui_renderer.render(&mut pass.forget_lifetime(), &primitives, &screen_desc);
		}

		for id in &egui_output.textures_delta.free {
			self.egui_renderer.free_texture(id);
		}

		self.queue.submit([enc.finish()]);
		frame.present();
	}

	pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
		if size.width == 0 || size.height == 0 { return; }
		self.config.width  = size.width;
		self.config.height = size.height;
		self.surface.configure(&self.device, &self.config);
	}

	fn make_bind_group(
		device: &wgpu::Device,
		pipeline: &RenderPipeline,
		uniforms_buf: &wgpu::Buffer,
		chunk_buf: &wgpu::Buffer,
	) -> wgpu::BindGroup {
		device.create_bind_group(&wgpu::BindGroupDescriptor {
			label:  Some("chunk_bg"),
			layout: &pipeline.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding:  0,
					resource: uniforms_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding:  1,
					resource: chunk_buf.as_entire_binding(),
				},
			],
		})
	}
}
