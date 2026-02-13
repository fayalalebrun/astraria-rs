use anyhow::Result;
/// Main application structure that coordinates all subsystems
/// Replaces the LibGDX Game class from the original implementation
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    AstrariaResult, assets::AssetManager, input::InputHandler, physics::PhysicsSimulation,
    renderer::Renderer, scenario::BodyType, ui::UserInterface,
};

#[cfg(feature = "web")]
use wasm_bindgen::JsCast;

#[cfg(feature = "web")]
use std::rc::Rc;
#[cfg(feature = "web")]
use std::cell::RefCell;

pub struct AstrariaApp {
    window: Option<Window>,
    renderer: Option<Renderer>,
    physics: Option<PhysicsSimulation>,
    ui: Option<UserInterface>,
    input_handler: Option<InputHandler>,
    asset_manager: Option<AssetManager>,
    #[cfg(feature = "native")]
    last_frame_time: std::time::Instant,
    #[cfg(feature = "web")]
    last_frame_time: f64,
    scenario_file: String,
    focus_body_index: usize,
    #[cfg(feature = "web")]
    initialized: bool,
}

impl AstrariaApp {
    pub fn new() -> Result<Self> {
        Self::new_with_scenario("Solar_System_2K.txt".to_string())
    }

    pub fn new_with_scenario(scenario_file: String) -> Result<Self> {
        Self::new_with_scenario_and_focus(scenario_file, 0)
    }

    #[cfg(feature = "native")]
    pub fn new_with_scenario_and_focus(
        scenario_file: String,
        focus_body_index: usize,
    ) -> Result<Self> {
        Ok(Self {
            window: None,
            renderer: None,
            physics: None,
            ui: None,
            input_handler: None,
            asset_manager: None,
            last_frame_time: std::time::Instant::now(),
            scenario_file,
            focus_body_index,
        })
    }

    #[cfg(feature = "web")]
    pub fn new_with_scenario_and_focus(
        scenario_file: String,
        focus_body_index: usize,
    ) -> Result<Self> {
        Ok(Self {
            window: None,
            renderer: None,
            physics: None,
            ui: None,
            input_handler: None,
            asset_manager: None,
            last_frame_time: 0.0,
            scenario_file,
            focus_body_index,
            initialized: false,
        })
    }

    #[cfg(feature = "native")]
    pub fn run(mut self) -> Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    #[cfg(feature = "web")]
    pub fn run(self) -> Result<()> {
        use winit::platform::web::EventLoopExtWebSys;

        let event_loop = EventLoop::new()?;
        let app = Rc::new(RefCell::new(self));

        // Clone for the event loop
        let app_clone = Rc::clone(&app);

        event_loop.spawn(move |event, elwt| {
            use winit::event::Event;

            // Use try_borrow_mut to avoid panics when async initialization holds the borrow
            match event {
                Event::Resumed => {
                    if let Ok(mut app) = app_clone.try_borrow_mut() {
                        app.web_resumed(elwt, Rc::clone(&app_clone));
                    }
                }
                Event::AboutToWait => {
                    if let Ok(mut app) = app_clone.try_borrow_mut() {
                        app.about_to_wait(elwt);
                    }
                }
                Event::WindowEvent { window_id, event } => {
                    if let Ok(mut app) = app_clone.try_borrow_mut() {
                        app.window_event(elwt, window_id, event);
                    }
                }
                _ => {}
            }
        });

        Ok(())
    }

    #[cfg(feature = "web")]
    fn web_resumed(&mut self, event_loop: &ActiveEventLoop, app: Rc<RefCell<Self>>) {
        use winit::platform::web::WindowAttributesExtWebSys;

        if self.window.is_none() {
            // Get the canvas element from the page
            let canvas = web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.get_element_by_id("astraria-canvas"))
                .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok());

            let window_attributes = if let Some(canvas) = canvas {
                Window::default_attributes()
                    .with_title("Astraria - 3D Orbital Mechanics Simulator")
                    .with_canvas(Some(canvas))
            } else {
                Window::default_attributes()
                    .with_title("Astraria - 3D Orbital Mechanics Simulator")
                    .with_inner_size(PhysicalSize::new(1280, 720))
            };

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    self.window = Some(window);
                    self.last_frame_time = Self::get_current_time_web();

                    // Spawn async initialization
                    wasm_bindgen_futures::spawn_local(async move {
                        log::info!("Inside spawn_local before borrow");
                        {
                            log::info!("Taking borrow_mut for initialization");
                            let mut app_ref = app.borrow_mut();
                            log::info!("Starting async initialization...");
                            if let Err(e) = app_ref.initialize_web().await {
                                log::error!("Failed to initialize: {:?}", e);
                            }
                            log::info!("initialize_web completed, dropping borrow");
                        }
                        log::info!("Borrow dropped, requesting first redraw...");
                        match app.try_borrow() {
                            Ok(app_ref) => {
                                log::info!("Got borrow, checking window");
                                if let Some(window) = &app_ref.window {
                                    log::info!("Calling window.request_redraw()");
                                    window.request_redraw();
                                } else {
                                    log::error!("No window available for redraw request");
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to borrow app for redraw: {:?}", e);
                            }
                        }
                        log::info!("spawn_local task completing");
                    });
                }
                Err(e) => {
                    log::error!("Failed to create window: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    #[cfg(feature = "web")]
    async fn initialize_web(&mut self) -> AstrariaResult<()> {
        log::info!("Initializing Astraria application (web)...");

        let window = self.window.as_ref().ok_or_else(|| {
            crate::AstrariaError::Graphics("No window available".to_string())
        })?;

        // Initialize asset manager first
        self.asset_manager = Some(AssetManager::new().await?);

        // Initialize renderer
        let renderer = Renderer::new(window, self.asset_manager.as_mut().unwrap()).await?;
        self.renderer = Some(renderer);

        // Load atmospheric assets required by shaders after renderer is ready
        if let (Some(asset_manager), Some(renderer)) = (&mut self.asset_manager, &self.renderer) {
            asset_manager
                .load_atmospheric_assets(renderer.device(), renderer.queue())
                .await?;
        }

        // Initialize physics simulation
        self.physics = Some(PhysicsSimulation::new());

        // Initialize input handler
        self.input_handler = Some(InputHandler::new());

        // Initialize UI
        let ui = UserInterface::new(window, self.renderer.as_ref().unwrap())?;
        self.ui = Some(ui);

        // Load default scenario if available
        self.load_default_scenario().await?;

        // Position camera to focus on the specified body
        self.position_camera_on_focus_body().await?;

        // Force a resize to sync with actual canvas size (may differ from initial size)
        // Get canvas size directly from DOM for more accurate dimensions
        let canvas_size = web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.get_element_by_id("astraria-canvas"))
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .map(|canvas| {
                // Use clientWidth/clientHeight for CSS-sized canvas
                let width = canvas.client_width() as u32;
                let height = canvas.client_height() as u32;
                log::info!("Canvas client size from DOM: {}x{}", width, height);
                PhysicalSize::new(width.max(1), height.max(1))
            });

        if let Some(canvas_size) = canvas_size {
            log::info!("Post-init resize to canvas size: {}x{}", canvas_size.width, canvas_size.height);
            if let Some(renderer) = &mut self.renderer {
                renderer.resize(canvas_size)?;
            }
            if let Some(ui) = &mut self.ui {
                ui.resize(canvas_size)?;
            }
        } else if let Some(window) = &self.window {
            // Fallback to window.inner_size()
            let current_size = window.inner_size();
            log::info!("Post-init resize to window size (fallback): {}x{}", current_size.width, current_size.height);
            if let Some(renderer) = &mut self.renderer {
                renderer.resize(current_size)?;
            }
            if let Some(ui) = &mut self.ui {
                ui.resize(current_size)?;
            }
        }

        self.initialized = true;
        log::info!("Astraria initialization complete!");
        Ok(())
    }

    async fn initialize(&mut self, window: &Window) -> AstrariaResult<()> {
        log::info!("Initializing Astraria application...");

        // Initialize asset manager first
        self.asset_manager = Some(AssetManager::new().await?);

        // Initialize renderer
        let renderer = Renderer::new(window, self.asset_manager.as_mut().unwrap()).await?;
        self.renderer = Some(renderer);

        // Load atmospheric assets required by shaders after renderer is ready
        if let (Some(asset_manager), Some(renderer)) = (&mut self.asset_manager, &self.renderer) {
            asset_manager
                .load_atmospheric_assets(renderer.device(), renderer.queue())
                .await?;
        }

        // Initialize physics simulation
        self.physics = Some(PhysicsSimulation::new());

        // Initialize input handler
        self.input_handler = Some(InputHandler::new());

        // Initialize UI
        let ui = UserInterface::new(window, self.renderer.as_ref().unwrap())?;
        self.ui = Some(ui);

        // Load default scenario if available
        self.load_default_scenario().await?;

        // Position camera to focus on the specified body
        self.position_camera_on_focus_body().await?;

        log::info!("Astraria initialization complete!");
        Ok(())
    }

    async fn load_default_scenario(&mut self) -> AstrariaResult<()> {
        // Try to load the specified scenario file
        log::info!(
            "App: Attempting to load scenario file: {}",
            self.scenario_file
        );
        if let Some(asset_manager) = &mut self.asset_manager {
            if let Ok(scenario_data) = asset_manager.load_scenario(&self.scenario_file).await {
                log::info!("App: Successfully loaded scenario data, parsing...");

                // Parse the scenario
                let scenario = crate::scenario::ScenarioParser::parse(&scenario_data)?;
                log::info!("App: Parsed scenario with {} bodies", scenario.bodies.len());

                // Load textures for all bodies in the scenario
                if let Some(renderer) = &mut self.renderer {
                    log::info!("App: Loading scenario textures...");
                    let (textures_before, models_before, cubemaps_before) =
                        asset_manager.cache_stats();
                    log::info!(
                        "App: AssetManager cache before scenario textures: textures={}, models={}, cubemaps={}",
                        textures_before,
                        models_before,
                        cubemaps_before
                    );

                    renderer
                        .main_renderer()
                        .load_scenario_textures(asset_manager, &scenario)
                        .await?;

                    let (textures_after, models_after, cubemaps_after) =
                        asset_manager.cache_stats();
                    log::info!(
                        "App: AssetManager cache after scenario textures: textures={}, models={}, cubemaps={}",
                        textures_after,
                        models_after,
                        cubemaps_after
                    );
                    log::info!(
                        "App: Scenario textures loaded successfully - added {} textures",
                        textures_after - textures_before
                    );
                } else {
                    log::error!("App: Renderer not initialized when loading scenario textures");
                }

                // Pass the raw scenario data to physics (it will re-parse, but that's OK for now)
                if let Some(physics) = &mut self.physics {
                    physics.load_scenario(scenario_data)?;
                    log::info!("App: Physics loaded scenario: {}", self.scenario_file);
                } else {
                    log::error!("App: Physics system not initialized when loading scenario");
                }
            } else {
                log::warn!(
                    "App: Could not load scenario '{}', starting with empty simulation",
                    self.scenario_file
                );
            }
        } else {
            log::error!("App: Asset manager not initialized when loading scenario");
        }
        Ok(())
    }

    async fn position_camera_on_focus_body(&mut self) -> AstrariaResult<()> {
        if let (Some(physics), Some(renderer)) = (&self.physics, &mut self.renderer) {
            let bodies = physics.get_bodies()?;
            if let Some(focus_body) = bodies.get(self.focus_body_index) {
                log::info!(
                    "Positioning camera to focus on body '{}' at index {}",
                    focus_body.name,
                    self.focus_body_index
                );

                // Calculate camera distance based on body radius
                let radius = match &focus_body.body_type {
                    BodyType::Planet { radius, .. } => radius,
                    BodyType::Star { radius, .. } => radius,
                    BodyType::PlanetAtmo { radius, .. } => radius,
                    BodyType::BlackHole { radius } => radius,
                };

                // Position camera at 3x radius distance for good view
                let camera_distance = radius * 3.0;
                let body_position = focus_body.position;

                // Use simplified look_at with distance parameter
                renderer.set_camera_look_at(body_position, camera_distance as f64);

                log::info!(
                    "Camera positioned at distance {:.2e} looking at '{}' at ({:.2e}, {:.2e}, {:.2e})",
                    camera_distance,
                    focus_body.name,
                    body_position.x,
                    body_position.y,
                    body_position.z
                );
            } else {
                log::warn!(
                    "Focus body index {} is out of range, using default camera position",
                    self.focus_body_index
                );
            }
        }
        Ok(())
    }

    fn update(&mut self, delta_time: f32) -> AstrariaResult<()> {
        // Update physics simulation
        if let Some(physics) = &mut self.physics {
            physics.update(delta_time)?;
        }

        // Update input handler
        if let Some(input_handler) = &mut self.input_handler {
            input_handler.update(delta_time);

            // Handle camera input and update camera movement
            if let Some(renderer) = &mut self.renderer {
                renderer.handle_camera_input(input_handler, delta_time)?;
                renderer.update_camera(delta_time);
            }
        }

        // Update UI
        if let Some(ui) = &mut self.ui {
            ui.update(delta_time, self.physics.as_ref(), &mut self.renderer)?;

            // Handle UI actions
            let actions = ui.take_actions();
            for action in actions {
                self.handle_ui_action(action)?;
            }
        }

        Ok(())
    }

    fn handle_ui_action(&mut self, action: crate::ui::UiAction) -> AstrariaResult<()> {
        use crate::ui::UiAction;

        match action {
            UiAction::FocusCameraOnObject {
                object_index,
                position,
                radius,
            } => {
                log::info!(
                    "Focusing camera on object {} at position ({:.2e}, {:.2e}, {:.2e}) with radius {:.2e}",
                    object_index,
                    position.x,
                    position.y,
                    position.z,
                    radius
                );

                if let Some(renderer) = &mut self.renderer {
                    // Position camera at 3x radius distance for good view (matching Java behavior)
                    let camera_distance = (radius * 3.0).max(1000.0); // Minimum 1000m distance
                    renderer.set_camera_look_at(position, camera_distance);

                    log::info!(
                        "Camera positioned at distance {:.2e} meters looking at object",
                        camera_distance
                    );
                }
            }
            UiAction::ClearCameraFocus => {
                log::info!("Clearing camera focus - camera now in free mode");
                // Camera focus is cleared - user can now move freely
                // No specific action needed as the camera will respond to user input
            }
        }

        Ok(())
    }

    fn render(&mut self) -> AstrariaResult<()> {
        if let (Some(renderer), Some(physics), Some(asset_manager), Some(ui), Some(window)) = (
            &mut self.renderer,
            &self.physics,
            &self.asset_manager,
            &mut self.ui,
            &self.window,
        ) {
            renderer.begin_frame()?;

            // Render 3D scene
            renderer.render_scene(physics, asset_manager)?;

            // Prepare and render UI overlay
            let (screen_descriptor, clipped_primitives) =
                ui.prepare(renderer, window, Some(physics))?;
            renderer.render_ui_overlay(ui, &clipped_primitives, &screen_descriptor)?;

            renderer.end_frame()?;
        }

        Ok(())
    }

    fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        window: &Window,
    ) -> AstrariaResult<bool> {
        // Let input handler process camera-related events first
        if let Some(input_handler) = &mut self.input_handler {
            // Give input handler priority for mouse events (needed for camera controls)
            match event {
                WindowEvent::MouseInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseWheel { .. } => {
                    if input_handler.handle_event(event)? {
                        return Ok(true); // Event consumed by input handler
                    }
                }
                _ => {}
            }
        }

        // Let UI handle the event next (but only if input handler didn't consume it)
        if let Some(ui) = &mut self.ui {
            if ui.handle_event(event, window)? {
                return Ok(true); // Event consumed by UI
            }
        }

        // Finally let input handler process keyboard events
        if let Some(input_handler) = &mut self.input_handler {
            if input_handler.handle_event(event)? {
                return Ok(true); // Event consumed by input handler
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, shutting down...");
                return Ok(false);
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(*physical_size)?;
                }
                if let Some(ui) = &mut self.ui {
                    ui.resize(*physical_size)?;
                }
            }
            _ => {}
        }

        Ok(true)
    }
}

#[cfg(feature = "native")]
impl ApplicationHandler for AstrariaApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Astraria - 3D Orbital Mechanics Simulator")
                .with_inner_size(PhysicalSize::new(1280, 720))
                .with_min_inner_size(PhysicalSize::new(800, 600));

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    if let Err(e) = pollster::block_on(self.initialize(&window)) {
                        log::error!("Failed to initialize application: {e}");
                        event_loop.exit();
                        return;
                    }
                    self.window = Some(window);
                    self.last_frame_time = std::time::Instant::now();
                }
                Err(e) => {
                    log::error!("Failed to create window: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        _event: winit::event::DeviceEvent,
    ) {
        // Handle device events if needed
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window.id() != window_id {
                return;
            }
        }

        if matches!(event, WindowEvent::RedrawRequested) {
            let current_time = std::time::Instant::now();
            let delta_time = current_time
                .duration_since(self.last_frame_time)
                .as_secs_f32();
            self.last_frame_time = current_time;

            if let Err(e) = self.update(delta_time) {
                log::error!("Update error: {e}");
            }

            if let Err(e) = self.render() {
                log::error!("Render error: {e}");
            }
            return;
        }

        // Handle the window event
        // Extract window temporarily to avoid borrow conflicts
        if let Some(window) = self.window.take() {
            let result = self.handle_window_event(&event, &window);
            self.window = Some(window); // Put it back

            match result {
                Ok(true) => {
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
                Ok(false) => {
                    event_loop.exit();
                }
                Err(e) => {
                    log::error!("Error handling window event: {e}");
                    event_loop.exit();
                }
            }
        }
    }
}


#[cfg(feature = "web")]
impl AstrariaApp {
    fn get_current_time_web() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Only request redraws after initialization is complete
        if self.initialized {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window.id() != window_id {
                return;
            }
        }

        if matches!(event, WindowEvent::RedrawRequested) {
            if !self.initialized {
                // Still initializing, skip rendering
                log::debug!("RedrawRequested but not initialized yet");
                return;
            }

            let current_time = Self::get_current_time_web();
            let delta_time = ((current_time - self.last_frame_time) / 1000.0) as f32;
            self.last_frame_time = current_time;

            // Only update/render if we have all components
            if self.renderer.is_some() {
                if let Err(e) = self.update(delta_time) {
                    log::error!("Update error: {e}");
                }

                if let Err(e) = self.render() {
                    log::error!("Render error: {e}");
                }

                // Continue rendering
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            } else {
                log::warn!("RedrawRequested but renderer is None");
            }
            return;
        }

        // Handle the window event
        if let Some(window) = self.window.take() {
            let result = self.handle_window_event(&event, &window);
            self.window = Some(window);

            match result {
                Ok(true) => {
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
                Ok(false) => {
                    event_loop.exit();
                }
                Err(e) => {
                    log::error!("Error handling window event: {e}");
                    event_loop.exit();
                }
            }
        }
    }
}

impl Drop for AstrariaApp {
    fn drop(&mut self) {
        log::info!("Shutting down Astraria application");

        // Stop physics simulation first
        if let Some(mut physics) = self.physics.take() {
            physics.shutdown();
        }

        // Clean up other resources
        // (Rust's drop semantics will handle the rest)
    }
}
