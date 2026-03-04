mod ui;

use iced::{Application, Settings};
use tracing_subscriber::EnvFilter;

fn main() -> iced::Result {
    // Initialize logging.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("embedding=debug".parse().unwrap()))
        .init();

    tracing::info!("Starting embedding-debugger");

    let mut settings = Settings::default();
    settings.window.size = iced::Size::new(1280.0, 800.0);
    settings.antialiasing = true;

    ui::app::App::run(settings)
}
