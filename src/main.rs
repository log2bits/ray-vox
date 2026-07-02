use ray_vox::render::{RenderMode, Renderer};
use ray_vox::render::camera::Camera;
use ray_vox::render::gpu_world::GpuWorldSnapshot;
use ray_vox::world::World;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const CASTLE_PATH: &str = "assets/castle.rvox";
const NORMAL_MOVE_SPEED_UNITS_PER_SECOND: f32 = 600.0;
const BOOST_MULTIPLIER: f32 = 4.0;
const MOUSE_LOOK_SENSITIVITY_RADIANS_PER_PIXEL: f32 = 0.0025;
const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct HeldMovementKeys {
	forward: bool,
	backward: bool,
	left: bool,
	right: bool,
	up: bool,
	down: bool,
	boost: bool,
}

struct App {
	world: World,
	renderer: Option<Renderer>,
	camera: Camera,
	last_frame_time: Instant,
	held_keys: HeldMovementKeys,
	mouse_look_active: bool,
	frames_since_fps_update: u32, // rolling window for the title-bar FPS
	last_fps_update: Instant,
}

impl App {
	fn new(world: World) -> Self {
		let camera = camera_starting_pose(&world);
		let now = Instant::now();
		Self {
			world,
			renderer: None,
			camera,
			last_frame_time: now,
			held_keys: HeldMovementKeys::default(),
			mouse_look_active: false,
			frames_since_fps_update: 0,
			last_fps_update: now,
		}
	}
}

// Position the camera back along -Z and above center, looking at the world.
fn camera_starting_pose(world: &World) -> Camera {
	let span = [
		world.chunk_grid_dim[0] as f32 * 256.0,
		world.chunk_grid_dim[1] as f32 * 256.0,
		world.chunk_grid_dim[2] as f32 * 256.0,
	];
	let origin = [
		world.origin.x() as f32,
		world.origin.y() as f32,
		world.origin.z() as f32,
	];
	let center = [
		origin[0] + span[0] * 0.5,
		origin[1] + span[1] * 0.5,
		origin[2] + span[2] * 0.5,
	];
	let max_span = span[0].max(span[1]).max(span[2]).max(256.0);
	let position = [
		center[0],
		center[1] + max_span * 0.3,
		center[2] - max_span * 1.1,
	];
	let mut camera = Camera::new(position);
	camera.yaw = 0.0; // looks along +Z
	camera.pitch = -0.25; // tips downward
	camera
}

impl ApplicationHandler for App {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.renderer.is_some() {
			return;
		}
		let window_attributes = Window::default_attributes()
			.with_title("ray-vox")
			.with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
		let window = Arc::new(
			event_loop
				.create_window(window_attributes)
				.expect("failed to create window"),
		);

		let mut renderer = pollster::block_on(Renderer::new(window.clone()));
		let snapshot = GpuWorldSnapshot::from_world(&self.world);
		let directory_bytes = snapshot.directory_byte_size();
		let chunk_data_bytes = snapshot.chunk_data_byte_size();
		renderer.upload_world(snapshot);
		println!(
			"uploaded to GPU: directory {} bytes, chunk_data {:.2} MB",
			directory_bytes,
			chunk_data_bytes as f64 / 1_048_576.0,
		);
		println!(
			"controls: WASD move, space/ctrl up/down, hold LMB to look, shift to boost, \
			 1 = normal / 2 = heatmap, escape to quit"
		);

		let inner_size = window.inner_size();
		self.camera.aspect = inner_size.width as f32 / inner_size.height.max(1) as f32;
		self.renderer = Some(renderer);
		window.request_redraw();
	}

	fn window_event(
		&mut self,
		event_loop: &ActiveEventLoop,
		_window_id: WindowId,
		event: WindowEvent,
	) {
		let Some(renderer) = self.renderer.as_mut() else { return };
		match event {
			WindowEvent::CloseRequested => event_loop.exit(),
			WindowEvent::Resized(new_size) => {
				renderer.resize(new_size.width, new_size.height);
				self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
			}
			WindowEvent::RedrawRequested => {
				let now = Instant::now();
				let delta_seconds = (now - self.last_frame_time).as_secs_f32();
				self.last_frame_time = now;
				// Inlined here so we don't borrow self mutably while holding
				// the renderer borrow above.
				let mut speed = NORMAL_MOVE_SPEED_UNITS_PER_SECOND;
				if self.held_keys.boost { speed *= BOOST_MULTIPLIER; }
				let step = speed * delta_seconds;
				if self.held_keys.forward  { self.camera.translate_forward(step); }
				if self.held_keys.backward { self.camera.translate_forward(-step); }
				if self.held_keys.right    { self.camera.translate_right(step); }
				if self.held_keys.left     { self.camera.translate_right(-step); }
				if self.held_keys.up       { self.camera.translate_world_up(step); }
				if self.held_keys.down     { self.camera.translate_world_up(-step); }
				match renderer.render(&self.camera) {
					Ok(()) => {}
					Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
						let size = renderer.window().inner_size();
						renderer.resize(size.width, size.height);
					}
					Err(wgpu::SurfaceError::OutOfMemory) => {
						eprintln!("GPU out of memory");
						event_loop.exit();
					}
					Err(other) => eprintln!("render error: {:?}", other),
				}
				// Update the title bar every FPS_UPDATE_INTERVAL so the FPS
				// number doesn't jitter every frame.
				self.frames_since_fps_update += 1;
				let elapsed = now - self.last_fps_update;
				if elapsed >= FPS_UPDATE_INTERVAL {
					let fps = self.frames_since_fps_update as f32 / elapsed.as_secs_f32();
					let mode_label = match renderer.render_mode() {
						RenderMode::Normal => "normal",
						RenderMode::Heatmap => "heatmap",
					};
					renderer
						.window()
						.set_title(&format!("ray-vox [{}] {:.1} fps", mode_label, fps));
					self.frames_since_fps_update = 0;
					self.last_fps_update = now;
				}
			}
			WindowEvent::KeyboardInput {
				event:
					KeyEvent {
						physical_key: PhysicalKey::Code(code),
						state,
						..
					},
				..
			} => {
				let pressed = state == ElementState::Pressed;
				match code {
					KeyCode::Escape => {
						if pressed {
							event_loop.exit();
						}
					}
					KeyCode::KeyW => self.held_keys.forward = pressed,
					KeyCode::KeyS => self.held_keys.backward = pressed,
					KeyCode::KeyA => self.held_keys.left = pressed,
					KeyCode::KeyD => self.held_keys.right = pressed,
					KeyCode::Space => self.held_keys.up = pressed,
					KeyCode::ControlLeft | KeyCode::ControlRight => self.held_keys.down = pressed,
					KeyCode::ShiftLeft | KeyCode::ShiftRight => self.held_keys.boost = pressed,
					KeyCode::Digit1 => {
						if pressed {
							renderer.set_render_mode(RenderMode::Normal);
						}
					}
					KeyCode::Digit2 => {
						if pressed {
							renderer.set_render_mode(RenderMode::Heatmap);
						}
					}
					_ => {}
				}
			}
			WindowEvent::MouseInput {
				state,
				button: MouseButton::Left,
				..
			} => {
				let pressing = state == ElementState::Pressed;
				if pressing && !self.mouse_look_active {
					// Lock the cursor if possible; some platforms only allow Confined.
					let window = renderer.window();
					let _ = window
						.set_cursor_grab(CursorGrabMode::Locked)
						.or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
					window.set_cursor_visible(false);
					self.mouse_look_active = true;
				} else if !pressing && self.mouse_look_active {
					let window = renderer.window();
					let _ = window.set_cursor_grab(CursorGrabMode::None);
					window.set_cursor_visible(true);
					self.mouse_look_active = false;
				}
			}
			_ => {}
		}
	}

	fn device_event(
		&mut self,
		_event_loop: &ActiveEventLoop,
		_device_id: DeviceId,
		event: DeviceEvent,
	) {
		if !self.mouse_look_active {
			return;
		}
		if let DeviceEvent::MouseMotion { delta } = event {
			let (dx, dy) = delta;
			// Mouse right (dx > 0) increases yaw so the camera turns right.
			// Mouse down (dy > 0) decreases pitch so the camera looks down.
			self.camera.apply_look_delta(
				(dx as f32) * MOUSE_LOOK_SENSITIVITY_RADIANS_PER_PIXEL,
				-(dy as f32) * MOUSE_LOOK_SENSITIVITY_RADIANS_PER_PIXEL,
			);
		}
	}

	fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
		if let Some(renderer) = self.renderer.as_ref() {
			renderer.window().request_redraw();
		}
	}
}

fn load_world_from_disk_or_fallback() -> World {
	if Path::new(CASTLE_PATH).exists() {
		println!("loading {}...", CASTLE_PATH);
		match std::fs::read(CASTLE_PATH) {
			Ok(bytes) => {
				let mut cursor = std::io::Cursor::new(&bytes);
				match World::load_rvox(&mut cursor) {
					Ok(world) => {
						let non_empty = world.chunks.iter().filter(|c| c.is_some()).count();
						println!(
							"loaded {} non-empty chunks (grid {:?}, origin {:?})",
							non_empty,
							world.chunk_grid_dim,
							<[i32; 3]>::from(world.origin),
						);
						return world;
					}
					Err(err) => eprintln!("failed to parse {}: {}", CASTLE_PATH, err),
				}
			}
			Err(err) => eprintln!("failed to read {}: {}", CASTLE_PATH, err),
		}
	} else {
		println!("no {}; building a small demo world instead", CASTLE_PATH);
	}
	build_demo_world()
}

fn build_demo_world() -> World {
	use ray_vox::chunk::material::Material;
	use ray_vox::generate::volume::sphere::Sphere;
	use ray_vox::util::types::WorldPos;

	let mut world = World::new([2, 2, 2]);
	let stone = Material::from_rgb_pbr_id([0x88, 0x88, 0x90], 0);
	let ember = Material::from_rgb_pbr_id([0xE0, 0x50, 0x30], 0);
	let moss = Material::from_rgb_pbr_id([0x40, 0xB0, 0x50], 0);
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(256, 256, 256), 180, stone)));
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(256, 400, 256), 55, ember)));
	world.apply_edit(Arc::new(Sphere::new(WorldPos::new(360, 250, 340), 45, moss)));
	world
}

pub fn main() {
	// Surface wgpu warnings on stderr. RUST_LOG=info shows device/pipeline
	// setup chatter; WGPU_LOG=warn filters to wgpu-only messages.
	env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

	let world = load_world_from_disk_or_fallback();
	let event_loop = EventLoop::new().expect("failed to create event loop");
	event_loop.set_control_flow(ControlFlow::Poll);
	let mut app = App::new(world);
	event_loop.run_app(&mut app).expect("event loop failed");
}
