use iced::widget::{column, container, row};
use iced::{Application, Command, Element, Length, Theme, Color};

use embedding_core::{EmbeddingSet, ProjectedPoint, Projector, TokenCollection};
use embedding_inference::{EmbeddingClient, ProviderConfig, ProviderKind};
use embedding_viz::PointCloud;

use super::sidebar;
use super::status_bar;
use super::viewer;

/// Top-level application state.
pub struct App {
    /// Current provider configuration.
    pub provider_config: ProviderConfig,
    /// Loaded token collection (if any).
    pub tokens: Option<TokenCollection>,
    /// Generated embeddings (if any).
    pub embeddings: Option<EmbeddingSet>,
    /// 3D projected points for visualization.
    pub projected: Vec<ProjectedPoint>,
    /// The point cloud visualization state.
    pub point_cloud: PointCloud,
    /// Sidebar UI state.
    pub sidebar: sidebar::SidebarState,
    /// Status message displayed at the bottom.
    pub status: String,
    /// Whether an embedding request is in flight.
    pub loading: bool,
}

/// Messages that can be sent to update the application.
#[derive(Debug, Clone)]
pub enum Message {
    // -- Sidebar messages --
    ProviderChanged(String),
    ModelChanged(String),
    ApiKeyChanged(String),
    DimensionsChanged(String),
    EndpointChanged(String),
    DeploymentChanged(String),

    // -- Token loading --
    LoadTokensPressed,
    TokensLoaded(Result<TokenCollection, String>),

    // -- Embedding generation --
    GeneratePressed,
    EmbeddingsGenerated(Result<EmbeddingSet, String>),

    // -- Viewer interaction --
    CameraRotate { dx: f32, dy: f32 },
    CameraZoom(f32),
    CameraPan { dx: f32, dy: f32 },

    // -- Misc --
    Noop,
}

impl Application for App {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let app = Self {
            provider_config: ProviderConfig::default(),
            tokens: None,
            embeddings: None,
            projected: Vec::new(),
            point_cloud: PointCloud::default(),
            sidebar: sidebar::SidebarState::default(),
            status: "Ready — load a token collection to get started.".into(),
            loading: false,
        };
        (app, Command::none())
    }

    fn title(&self) -> String {
        "Embedding Debugger".to_string()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            // -- Provider configuration --
            Message::ProviderChanged(provider) => {
                self.provider_config.kind = match provider.as_str() {
                    "OpenAI" => ProviderKind::OpenAI,
                    "Azure" => ProviderKind::Azure,
                    _ => ProviderKind::OpenRouter,
                };
                // Reset model placeholder for the selected provider.
                if self.sidebar.model_input.is_empty() {
                    self.provider_config.model = match provider.as_str() {
                        "OpenAI" => "text-embedding-3-small".into(),
                        _ => "openai/text-embedding-3-small".into(),
                    };
                }
                self.sidebar.provider_input = provider;
                Command::none()
            }
            Message::ModelChanged(model) => {
                self.provider_config.model = model.clone();
                self.sidebar.model_input = model;
                Command::none()
            }
            Message::ApiKeyChanged(key) => {
                self.provider_config.api_key = if key.is_empty() { None } else { Some(key.clone()) };
                self.sidebar.api_key_input = key;
                Command::none()
            }
            Message::DimensionsChanged(dims) => {
                self.sidebar.dimensions_input = dims.clone();
                self.provider_config.dimensions = dims.parse().ok();
                Command::none()
            }
            Message::EndpointChanged(url) => {
                self.provider_config.endpoint = if url.is_empty() { None } else { Some(url.clone()) };
                self.sidebar.endpoint_input = url;
                Command::none()
            }
            Message::DeploymentChanged(name) => {
                self.provider_config.deployment_name = if name.is_empty() { None } else { Some(name.clone()) };
                self.sidebar.deployment_input = name;
                Command::none()
            }

            // -- Token loading --
            Message::LoadTokensPressed => {
                self.status = "Loading tokens...".into();
                // TODO: Open a file dialog (rfd) and load tokens.
                Command::none()
            }
            Message::TokensLoaded(result) => {
                match result {
                    Ok(collection) => {
                        self.status = format!(
                            "Loaded {} tokens from '{}'.",
                            collection.len(),
                            collection.name
                        );
                        self.tokens = Some(collection);
                    }
                    Err(e) => {
                        self.status = format!("Failed to load tokens: {e}");
                    }
                }
                Command::none()
            }

            // -- Embedding generation --
            Message::GeneratePressed => {
                if let Some(tokens) = &self.tokens {
                    self.loading = true;
                    self.status = format!(
                        "Generating embeddings for {} tokens with {}...",
                        tokens.len(),
                        self.provider_config.model
                    );

                    let config = self.provider_config.clone();
                    let texts: Vec<String> = tokens.texts().into_iter().map(|s| s.to_string()).collect();

                    Command::perform(
                        async move {
                            let client = EmbeddingClient::new(config);
                            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                            client
                                .embed(&text_refs)
                                .await
                                .map_err(|e| e.to_string())
                        },
                        Message::EmbeddingsGenerated,
                    )
                } else {
                    self.status = "No tokens loaded. Load a token collection first.".into();
                    Command::none()
                }
            }
            Message::EmbeddingsGenerated(result) => {
                self.loading = false;
                match result {
                    Ok(set) => {
                        self.status = format!(
                            "Generated {} embeddings ({}D) — projecting to 3D...",
                            set.len(),
                            set.dimensions
                        );

                        let vectors = set.vectors_as_matrix();
                        let mut projected = Projector::pca(&vectors);
                        Projector::normalize(&mut projected);
                        self.projected = projected;

                        let colors = self.category_colors();
                        self.point_cloud.set_points(&self.projected, &colors);
                        self.embeddings = Some(set);

                        self.status = format!(
                            "Showing {} points in 3D. Drag to rotate, scroll to zoom.",
                            self.projected.len()
                        );
                    }
                    Err(e) => {
                        self.status = format!("Embedding generation failed: {e}");
                    }
                }
                Command::none()
            }

            // -- Camera controls --
            Message::CameraRotate { dx, dy } => {
                self.point_cloud.camera.rotate(dx * 0.01, dy * 0.01);
                Command::none()
            }
            Message::CameraZoom(delta) => {
                self.point_cloud.camera.zoom(delta * 0.1);
                Command::none()
            }
            Message::CameraPan { dx, dy } => {
                self.point_cloud.camera.pan(dx * 0.005, dy * 0.005);
                Command::none()
            }

            Message::Noop => Command::none(),
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let sidebar = sidebar::view(&self.sidebar, &self.tokens, self.loading);
        let viewer = viewer::view(&self.point_cloud, &self.embeddings);
        let status = status_bar::view(&self.status);

        let main_content = row![
            sidebar,
            viewer,
        ]
        .spacing(0)
        .height(Length::Fill);

        container(
            column![
                main_content,
                status,
            ]
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }
}

impl App {
    /// Generate a color for each token based on its category.
    fn category_colors(&self) -> Vec<Color> {
        let palette = [
            Color::from_rgb(0.4, 0.7, 1.0),
            Color::from_rgb(1.0, 0.5, 0.3),
            Color::from_rgb(0.3, 0.9, 0.4),
            Color::from_rgb(0.9, 0.3, 0.7),
            Color::from_rgb(1.0, 0.9, 0.2),
            Color::from_rgb(0.6, 0.4, 0.9),
            Color::from_rgb(0.2, 0.9, 0.8),
            Color::from_rgb(0.9, 0.6, 0.2),
        ];

        let tokens = match &self.tokens {
            Some(t) => &t.tokens,
            None => return Vec::new(),
        };

        let mut categories: Vec<String> = Vec::new();
        for token in tokens {
            if let Some(ref cat) = token.category {
                if !categories.contains(cat) {
                    categories.push(cat.clone());
                }
            }
        }

        tokens
            .iter()
            .map(|token| {
                if let Some(ref cat) = token.category {
                    let idx = categories.iter().position(|c| c == cat).unwrap_or(0);
                    palette[idx % palette.len()]
                } else {
                    Color::WHITE
                }
            })
            .collect()
    }
}
