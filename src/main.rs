use std::sync::Arc;
use std::collections::HashSet;

use lattice::{
	render::{CameraPos, Renderer},
	shape::{CheckeredSphere, Shape},
	types::Voxel,
	world::{ShapeEdit, World},
};
use winit::{
	application::ApplicationHandler,
	event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
	event_loop::{ActiveEventLoop, EventLoop},
	keyboard::{KeyCode, PhysicalKey},
	window::{CursorGrabMode, Window, WindowId},
};

const MOVE_SPEED:   f32 = 120.0;
const MOUSE_SENS:   f32 = 0.002;
const PITCH_LIMIT:  f32 = std::f32::consts::FRAC_PI_2 - 0.01;

struct App {
	window:    Option<Arc<Window>>,
	renderer:  Option<Renderer>,
	camera:    CameraPos,
	keys:      HashSet<KeyCode>,
	mouse_captured: bool,
	last_frame: std::time::Instant,
	fps:        f32,
	fps_accum:  f32,
	fps_frames: u32,
	egui_ctx:   egui::Context,
	egui_state: Option<egui_winit::State>,
}

impl App {
	fn new() -> Self {
		Self {
			window:    None,
			renderer:  None,
			camera:    CameraPos::new([-80.0, 128.0, 128.0], 0.0, 0.0),
			keys:      HashSet::new(),
			mouse_captured: false,
			last_frame: std::time::Instant::now(),
			fps:        0.0,
			fps_accum:  0.0,
			fps_frames: 0,
			egui_ctx:   egui::Context::default(),
			egui_state: None,
		}
	}

	fn capture_mouse(&mut self, captured: bool) {
		let Some(w) = &self.window else { return };
		if captured {
			let _ = w.set_cursor_grab(CursorGrabMode::Locked)
				.or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
			w.set_cursor_visible(false);
		} else {
			let _ = w.set_cursor_grab(CursorGrabMode::None);
			w.set_cursor_visible(true);
		}
		self.mouse_captured = captured;
	}

	fn update(&mut self) {
		let now = std::time::Instant::now();
		let dt = now.duration_since(self.last_frame).as_secs_f32();
		self.last_frame = now;

		let speed = MOVE_SPEED * dt;
		let forward = self.camera.forward();
		let right   = self.camera.right();

		let mut move_dir = [0f32; 3];
		if self.keys.contains(&KeyCode::KeyW) {
			for i in 0..3 { move_dir[i] += forward[i]; }
		}
		if self.keys.contains(&KeyCode::KeyS) {
			for i in 0..3 { move_dir[i] -= forward[i]; }
		}
		if self.keys.contains(&KeyCode::KeyD) {
			for i in 0..3 { move_dir[i] += right[i]; }
		}
		if self.keys.contains(&KeyCode::KeyA) {
			for i in 0..3 { move_dir[i] -= right[i]; }
		}
		if self.keys.contains(&KeyCode::Space) {
			move_dir[1] += 1.0;
		}
		if self.keys.contains(&KeyCode::ShiftLeft) {
			move_dir[1] -= 1.0;
		}

		let len = (move_dir[0]*move_dir[0] + move_dir[1]*move_dir[1] + move_dir[2]*move_dir[2]).sqrt();
		if len > 0.0 {
			for i in 0..3 { self.camera.local[i] += move_dir[i] / len * speed; }
		}

		self.fps_accum  += dt;
		self.fps_frames += 1;
		if self.fps_accum >= 0.5 {
			self.fps = self.fps_frames as f32 / self.fps_accum;
			self.fps_accum  = 0.0;
			self.fps_frames = 0;
		}
	}
}

impl ApplicationHandler for App {
	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		let window = Arc::new(
			event_loop
				.create_window(Window::default_attributes().with_title("Lattice"))
				.unwrap(),
		);

		let egui_state = egui_winit::State::new(
			self.egui_ctx.clone(),
			egui::ViewportId::ROOT,
			window.as_ref(),
			None,
			None,
			None,
		);
		self.egui_state = Some(egui_state);

		let mut world = World::new();
		let white = Voxel::from_rgb_flags([255, 255, 255], 15, false, false, false, false);
		let grey  = Voxel::from_rgb_flags([128, 128, 128], 15, false, false, false, false);
		let sphere = CheckeredSphere { center: [128, 128, 128], radius: 64, material_a: white, material_b: grey };
		world.add_shape_edit(ShapeEdit::write(sphere.aabb(), 0, Box::new(sphere)));
		let chunk = world.generate_chunk([0, 0, 0], 0);

		println!("{}", chunk.memory_bytes());

		let renderer = pollster::block_on(Renderer::new(window.clone(), &chunk));

		self.renderer  = Some(renderer);
		self.window    = Some(window);
		self.last_frame = std::time::Instant::now();
	}

	fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
		if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
			if self.mouse_captured {
				self.camera.yaw   -= dx as f32 * MOUSE_SENS;
				self.camera.pitch -= dy as f32 * MOUSE_SENS;
				self.camera.pitch  = self.camera.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
			}
		}
	}

	fn window_event(
		&mut self,
		event_loop: &ActiveEventLoop,
		_id: WindowId,
		event: WindowEvent,
	) {
		// Forward events to egui; when mouse is captured, don't let egui consume them.
		let egui_consumed = if let (Some(state), Some(w)) = (&mut self.egui_state, &self.window) {
			let response = state.on_window_event(w, &event);
			response.consumed && !self.mouse_captured
		} else {
			false
		};
		if egui_consumed { return; }

		match event {
			WindowEvent::CloseRequested => event_loop.exit(),

			WindowEvent::Resized(size) => {
				if let Some(r) = &mut self.renderer {
					r.resize(size);
				}
			}

			WindowEvent::MouseInput {
				state: ElementState::Pressed,
				button: winit::event::MouseButton::Left,
				..
			} => {
				if !self.mouse_captured {
					self.capture_mouse(true);
				}
			}

			WindowEvent::KeyboardInput { event, .. } => {
				if let PhysicalKey::Code(key) = event.physical_key {
					match event.state {
						ElementState::Pressed  => { self.keys.insert(key); }
						ElementState::Released => { self.keys.remove(&key); }
					}
					if key == KeyCode::Escape && event.state == ElementState::Pressed {
						if self.mouse_captured {
							self.capture_mouse(false);
						} else {
							event_loop.exit();
						}
					}
				}
			}

			WindowEvent::RedrawRequested => {
				self.update();

				let (Some(renderer), Some(window), Some(egui_state)) =
					(&mut self.renderer, &self.window, &mut self.egui_state)
				else {
					return;
				};

				let raw_input = egui_state.take_egui_input(window);
				let fps = self.fps;
				let vram_mb = renderer.vram_bytes() as f64 / (1024.0 * 1024.0);
				let full_output = self.egui_ctx.run(raw_input, |ctx| {
					egui::Area::new(egui::Id::new("fps_counter"))
						.fixed_pos([10.0, 10.0])
						.show(ctx, |ui| {
							ui.set_min_width(120.0);
							let style = |s: &str| egui::RichText::new(s).color(egui::Color32::WHITE).size(18.0);
							ui.label(style(&format!("{fps:.0} fps")));
							ui.label(style(&format!("{vram_mb:.1} MB VRAM")));
						});
				});

				let platform_output = full_output.platform_output.clone();
				let camera = self.camera;
				let egui_ctx = self.egui_ctx.clone();
				renderer.render(&camera, &egui_ctx, full_output);
				egui_state.handle_platform_output(window, platform_output);

				window.request_redraw();
			}

			_ => {}
		}
	}
}

fn main() {
	let event_loop = EventLoop::new().unwrap();
	let mut app = App::new();
	event_loop.run_app(&mut app).unwrap();
}
