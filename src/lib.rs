pub mod app;
pub mod assets;
pub mod generated_shaders;
pub mod graphics;
pub mod input;
pub mod math;
pub mod physics;
pub mod renderer;
pub mod scenario;
pub mod ui;

pub use app::AstrariaApp;

use anyhow::Result;

#[derive(thiserror::Error, Debug)]
pub enum AstrariaError {
    #[error("Graphics error: {0}")]
    Graphics(String),
    #[error("Physics simulation error: {0}")]
    Physics(String),
    #[error("Asset loading error: {0}")]
    AssetLoading(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Rendering error: {0}")]
    RenderingError(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse float error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
}

pub type AstrariaResult<T> = Result<T, AstrariaError>;

// Web entry point
#[cfg(feature = "web")]
pub mod web {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn web_main() {
        // Set up panic hook for better error messages
        console_error_panic_hook::set_once();

        // Initialize logging to browser console
        console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

        log::info!("Starting Astraria in browser...");

        // Run the application
        if let Err(e) = run_web() {
            log::error!("Failed to run Astraria: {:?}", e);
        }
    }

    fn run_web() -> anyhow::Result<()> {
        use crate::AstrariaApp;

        let app = AstrariaApp::new_with_scenario_and_focus(
            "Solar_System_2K.txt".to_string(),
            0,
        )?;

        // On web, run() uses spawn_app which is non-blocking
        app.run()?;

        Ok(())
    }
}
