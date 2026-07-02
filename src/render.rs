pub mod camera;
pub mod gpu_world;

use bytemuck::{Pod, Zeroable};
use camera::Camera;
use gpu_world::GpuWorldSnapshot;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

// Uploaded to the fragment shader every frame. Layout matches Uniforms in
// shaders/render.wgsl; vec3s pad to vec4 for std140 alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
	camera_eye: [f32; 4],
	camera_right: [f32; 4],
	camera_up: [f32; 4],
	camera_forward: [f32; 4],
	world_origin: [f32; 4],
	chunk_grid_dim: [u32; 4],
	resolution: [f32; 2],
	fov_scale: f32,
	aspect: f32,
	render_mode: u32, // 0 = normal, 1 = heatmap
	_padding: [u32; 3], // keeps the struct 16-byte aligned end-to-end
}

// What the fragment shader draws each pixel as.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderMode {
	// Ray-traced voxels with Lambert shading.
	Normal,
	// Per-pixel memory-read heatmap: black is cold, then purple, magenta,
	// orange, and finally white for the hottest rays (typically grazing).
	Heatmap,
}

impl RenderMode {
	fn as_uniform(self) -> u32 {
		match self {
			RenderMode::Normal => 0,
			RenderMode::Heatmap => 1,
		}
	}
}

pub struct Renderer {
	window: Arc<Window>,
	surface: wgpu::Surface<'static>,
	device: wgpu::Device,
	queue: wgpu::Queue,
	surface_format: wgpu::TextureFormat,
	surface_config: wgpu::SurfaceConfiguration,
	pipeline: wgpu::RenderPipeline,
	bind_group_layout: wgpu::BindGroupLayout,
	bind_group: Option<wgpu::BindGroup>,
	uniforms_buffer: wgpu::Buffer,
	directory_buffer: Option<wgpu::Buffer>,
	chunk_data_buffer: Option<wgpu::Buffer>,
	snapshot: Option<GpuWorldSnapshot>,
	render_mode: RenderMode,
}

impl Renderer {
	pub async fn new(window: Arc<Window>) -> Self {
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::PRIMARY,
			..Default::default()
		});

		let surface = instance
			.create_surface(window.clone())
			.expect("create surface");

		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::HighPerformance,
				compatible_surface: Some(&surface),
				force_fallback_adapter: false,
			})
			.await
			.expect("no suitable GPU adapter");

		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					label: Some("ray-vox device"),
					required_features: wgpu::Features::empty(),
					required_limits: wgpu::Limits {
						max_storage_buffer_binding_size: 256 * 1024 * 1024,
						max_buffer_size: 256 * 1024 * 1024,
						..wgpu::Limits::default()
					},
					memory_hints: wgpu::MemoryHints::Performance,
				},
				None,
			)
			.await
			.expect("failed to acquire device");

		let surface_caps = surface.get_capabilities(&adapter);
		let surface_format = surface_caps
			.formats
			.iter()
			.copied()
			.find(|f| f.is_srgb())
			.unwrap_or(surface_caps.formats[0]);

		// Prefer an uncapped present mode so FPS reflects real throughput,
		// not display refresh. Mailbox (uncapped, no tearing) beats Immediate
		// (uncapped, tears); Fifo (vsync-capped) is the final fallback.
		let present_mode = if surface_caps
			.present_modes
			.contains(&wgpu::PresentMode::Mailbox)
		{
			wgpu::PresentMode::Mailbox
		} else if surface_caps
			.present_modes
			.contains(&wgpu::PresentMode::Immediate)
		{
			wgpu::PresentMode::Immediate
		} else {
			wgpu::PresentMode::Fifo
		};

		let window_size = window.inner_size();
		let surface_config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: surface_format,
			width: window_size.width.max(1),
			height: window_size.height.max(1),
			present_mode,
			desired_maximum_frame_latency: 2,
			alpha_mode: surface_caps.alpha_modes[0],
			view_formats: Vec::new(),
		};
		surface.configure(&device, &surface_config);

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("ray-vox render shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("ray-vox bind group layout"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: true },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 2,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Storage { read_only: true },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
			],
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("ray-vox pipeline layout"),
			bind_group_layouts: &[&bind_group_layout],
			push_constant_ranges: &[],
		});

		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("ray-vox render pipeline"),
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: Some("vs_main"),
				buffers: &[],
				compilation_options: Default::default(),
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: Some("fs_main"),
				targets: &[Some(wgpu::ColorTargetState {
					format: surface_format,
					blend: None,
					write_mask: wgpu::ColorWrites::ALL,
				})],
				compilation_options: Default::default(),
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				polygon_mode: wgpu::PolygonMode::Fill,
				unclipped_depth: false,
				conservative: false,
			},
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			cache: None,
		});

		let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("ray-vox uniforms"),
			size: std::mem::size_of::<Uniforms>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		Self {
			window,
			surface,
			device,
			queue,
			surface_format,
			surface_config,
			pipeline,
			bind_group_layout,
			bind_group: None,
			uniforms_buffer,
			directory_buffer: None,
			chunk_data_buffer: None,
			snapshot: None,
			render_mode: RenderMode::Normal,
		}
	}

	pub fn set_render_mode(&mut self, mode: RenderMode) {
		self.render_mode = mode;
	}

	pub fn render_mode(&self) -> RenderMode {
		self.render_mode
	}

	pub fn window(&self) -> &Window {
		&self.window
	}

	pub fn surface_format(&self) -> wgpu::TextureFormat {
		self.surface_format
	}

	pub fn resize(&mut self, new_width: u32, new_height: u32) {
		let width = new_width.max(1);
		let height = new_height.max(1);
		if width == self.surface_config.width && height == self.surface_config.height {
			return;
		}
		self.surface_config.width = width;
		self.surface_config.height = height;
		self.surface.configure(&self.device, &self.surface_config);
	}

	// Upload a World snapshot and rebuild the bind group. Call once per world
	// change; the shader reads the two storage buffers on every draw.
	pub fn upload_world(&mut self, snapshot: GpuWorldSnapshot) {
		let directory_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("ray-vox chunk directory"),
			contents: bytemuck::cast_slice(&snapshot.directory),
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		});

		let chunk_data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("ray-vox chunk data"),
			contents: bytemuck::cast_slice(&snapshot.chunk_data),
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
		});

		let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("ray-vox bind group"),
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: self.uniforms_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: directory_buffer.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: chunk_data_buffer.as_entire_binding(),
				},
			],
		});

		self.directory_buffer = Some(directory_buffer);
		self.chunk_data_buffer = Some(chunk_data_buffer);
		self.bind_group = Some(bind_group);
		self.snapshot = Some(snapshot);
	}

	pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
		let Some(bind_group) = self.bind_group.as_ref() else {
			return Ok(()); // no world uploaded yet
		};

		let uniforms = self.build_uniforms(camera);
		self.queue
			.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

		let frame = self.surface.get_current_texture()?;
		let view = frame
			.texture
			.create_view(&wgpu::TextureViewDescriptor::default());
		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("ray-vox frame encoder"),
			});

		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("ray-vox main pass"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			pass.set_pipeline(&self.pipeline);
			pass.set_bind_group(0, bind_group, &[]);
			pass.draw(0..3, 0..1);
		}

		self.queue.submit(std::iter::once(encoder.finish()));
		frame.present();
		Ok(())
	}

	fn build_uniforms(&self, camera: &Camera) -> Uniforms {
		let world_origin = self
			.snapshot
			.as_ref()
			.map(|s| s.world_origin)
			.unwrap_or([0.0, 0.0, 0.0]);
		let chunk_grid_dim = self
			.snapshot
			.as_ref()
			.map(|s| s.chunk_grid_dim)
			.unwrap_or([0, 0, 0]);
		let eye = camera.eye();
		let right = camera.right();
		let up = camera.up();
		let forward = camera.forward();
		let resolution = [
			self.surface_config.width as f32,
			self.surface_config.height as f32,
		];
		Uniforms {
			camera_eye: [eye[0], eye[1], eye[2], 0.0],
			camera_right: [right[0], right[1], right[2], 0.0],
			camera_up: [up[0], up[1], up[2], 0.0],
			camera_forward: [forward[0], forward[1], forward[2], 0.0],
			world_origin: [world_origin[0], world_origin[1], world_origin[2], 0.0],
			chunk_grid_dim: [chunk_grid_dim[0], chunk_grid_dim[1], chunk_grid_dim[2], 0],
			resolution,
			fov_scale: camera.fov_scale(),
			aspect: camera.aspect,
			render_mode: self.render_mode.as_uniform(),
			_padding: [0; 3],
		}
	}
}
