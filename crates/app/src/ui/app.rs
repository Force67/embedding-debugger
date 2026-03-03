use iced::widget::{column, container, row};
use iced::{Application, Command, Element, Length, Theme, Color};

use embedding_core::{EmbeddingSet, ProjectedPoint, Projector, TokenCollection};
use embedding_inference::{EmbeddingClient, ProviderConfig, ProviderKind};
use embedding_viz::{ArcballCamera, PointCloud, PointSelection};

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
    /// Currently selected point info (if any).
    pub selected_point: Option<SelectedPointInfo>,
    /// Mirrored camera state from the viewer (for axis indicator).
    pub viewer_camera: ArcballCamera,
    /// Current bubble/point render size.
    pub bubble_size: f32,
    /// Current search query for fuzzy point finding.
    pub search_query: String,
}

/// Information about a selected point (resolved with token label).
#[derive(Debug, Clone)]
pub struct SelectedPointInfo {
    pub index: usize,
    pub label: String,
    pub position: [f32; 3],
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
    /// User edited the free-text token field.
    TokensTextChanged(String),
    /// Parse and use whatever is in the text field.
    UseTypedTokens,

    // -- Embedding generation --
    GeneratePressed,
    EmbeddingsGenerated(Result<EmbeddingSet, String>),

    // -- Viewer interaction --
    PointSelected(Option<PointSelection>),
    ViewerCameraChanged(ArcballCamera),

    // -- Visualization controls --
    /// Bubble size slider moved (range 2–30).
    BubbleSizeChanged(f32),
    /// Search bar text changed — fuzzy-selects the best matching point.
    SearchQueryChanged(String),

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
            selected_point: None,
            viewer_camera: ArcballCamera::default(),
            bubble_size: 8.0,
            search_query: String::new(),
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
                Command::perform(
                    async {
                        let handle = rfd::AsyncFileDialog::new()
                            .set_title("Open token file")
                            .add_filter("Token files", &["json", "txt"])
                            .add_filter("All files", &["*"])
                            .pick_file()
                            .await;

                        match handle {
                            None => Err("No file selected.".to_string()),
                            Some(h) => {
                                let path = h.path().to_path_buf();
                                let ext = path
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .unwrap_or("");
                                if ext == "json" {
                                    TokenCollection::from_json_file(&path)
                                        .map_err(|e| e.to_string())
                                } else {
                                    let name = path
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("tokens")
                                        .to_string();
                                    TokenCollection::from_text_file(&path, name)
                                        .map_err(|e| e.to_string())
                                }
                            }
                        }
                    },
                    Message::TokensLoaded,
                )
            }
            Message::TokensLoaded(result) => {
                match result {
                    Ok(collection) => {
                        self.status = format!(
                            "Loaded {} tokens from '{}'.",
                            collection.len(),
                            collection.name
                        );
                        // Mirror into the text field so the user can see/edit.
                        self.sidebar.tokens_text = collection.texts().join(", ");
                        self.tokens = Some(collection);
                    }
                    Err(e) => {
                        self.status = format!("Failed to load tokens: {e}");
                    }
                }
                Command::none()
            }
            Message::TokensTextChanged(s) => {
                self.sidebar.tokens_text = s;
                Command::none()
            }
            Message::UseTypedTokens => {
                let raw = self.sidebar.tokens_text.clone();
                // Split on commas or newlines, trim whitespace, drop empty.
                let tokens: Vec<embedding_core::Token> = raw
                    .split([',', '\n'])
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(embedding_core::Token::new)
                    .collect();
                if tokens.is_empty() {
                    self.status = "No tokens found in the input field.".into();
                } else {
                    let count = tokens.len();
                    let mut col = TokenCollection::new("Manual");
                    for t in tokens {
                        col.push(t);
                    }
                    self.tokens = Some(col);
                    self.status = format!("Using {count} tokens from text input.");
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

            // -- Viewer interaction --
            Message::PointSelected(sel) => {
                self.selected_point = sel.map(|s| {
                    let label = self.tokens
                        .as_ref()
                        .and_then(|tc| tc.tokens.get(s.index))
                        .map(|t| t.text.clone())
                        .unwrap_or_else(|| format!("Point {}", s.index));
                    SelectedPointInfo {
                        index: s.index,
                        label,
                        position: s.position,
                    }
                });
                Command::none()
            }
            Message::ViewerCameraChanged(cam) => {
                self.viewer_camera = cam;
                Command::none()
            }

            Message::BubbleSizeChanged(size) => {
                self.bubble_size = size;
                self.point_cloud.set_point_size(size);
                Command::none()
            }

            Message::SearchQueryChanged(query) => {
                self.search_query = query.clone();
                if query.trim().is_empty() {
                    self.selected_point = None;
                } else if let Some(tokens) = &self.tokens {
                    if let Some((idx, score)) = fuzzy_find(&query, tokens) {
                        if score > 0 {
                            let point = self.point_cloud.points.iter().find(|p| p.index == idx);
                            if let Some(p) = point {
                                let pos = p.position;
                                let label = tokens.tokens[idx].text.clone();
                                self.selected_point = Some(SelectedPointInfo {
                                    index: idx,
                                    label,
                                    position: pos,
                                });
                            }
                        }
                    }
                }
                Command::none()
            }

            Message::Noop => Command::none(),
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let sidebar = sidebar::view(&self.sidebar, &self.tokens, self.loading, self.bubble_size);
        let viewer = viewer::view(
            &self.point_cloud,
            &self.embeddings,
            &self.selected_point,
            &self.viewer_camera,
            &self.search_query,
        );
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

// ─── Fuzzy search ─────────────────────────────────────────────────────────────

/// Find the best-matching token index for `query`.
/// Returns `(index, score)` where higher score = better match, or `None`.
///
/// Scoring (highest wins):
///   3 — exact match (case-insensitive)
///   2 — query is a prefix of the token
///   1 — token contains the query as a substring
///   0 — lowest edit-distance fallback (always returns something if tokens exist)
pub fn fuzzy_find(query: &str, tokens: &embedding_core::TokenCollection) -> Option<(usize, u32)> {
    if tokens.is_empty() {
        return None;
    }
    let q = query.to_lowercase();

    // Priority passes: exact → prefix → contains
    for (tier, pred) in [
        (3u32, Box::new(|t: &str| t.to_lowercase() == q) as Box<dyn Fn(&str) -> bool>),
        (2,    Box::new(|t: &str| t.to_lowercase().starts_with(&q))),
        (1,    Box::new(|t: &str| t.to_lowercase().contains(&q))),
    ] {
        if let Some(idx) = tokens.tokens.iter().position(|t| pred(&t.text)) {
            return Some((idx, tier));
        }
    }

    // Edit-distance fallback — find token with minimum distance.
    let best = tokens.tokens
        .iter()
        .enumerate()
        .min_by_key(|(_, t)| edit_distance(&q, &t.text.to_lowercase()));

    best.map(|(idx, _)| (idx, 0))
}

/// Simple Levenshtein edit distance (bounded to avoid O(n²) blowup on long strings).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().take(32).collect();
    let b: Vec<char> = b.chars().take(32).collect();
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n { dp[i][0] = i; }
    for j in 0..=m { dp[0][j] = j; }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[n][m]
}
