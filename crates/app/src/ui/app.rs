use std::cmp::Ordering;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use iced::widget::{column, container, row};
use iced::{Application, Color, Command, Element, Length, Theme};

use embedding_core::{
    EmbeddingSet, ProjectedPoint, ProjectionMethod, Projector, TokenCollection, TsneParams,
};
use embedding_inference::{EmbeddingClient, ProviderConfig, ProviderKind};
use embedding_viz::{ArcballCamera, PointCloud, PointSelection};
use tracing::warn;

use super::credentials;
use super::persistence::{self, ProviderProfile};
use super::sidebar;
use super::status_bar;
use super::viewer;

const SESSION_SAVE_DEBOUNCE: Duration = Duration::from_millis(750);
const NEAR_DUP_THRESHOLD: f32 = 0.999;
const NEAR_DUP_MAX_SAMPLE: usize = 1200;
const NEAR_DUP_MAX_EXAMPLES: usize = 5;
const NEIGHBOR_COUNT: usize = 8;

/// Top-level application state.
pub struct App {
    /// Current provider configuration.
    pub provider_config: ProviderConfig,
    /// Saved non-secret provider profiles.
    pub provider_profiles: Vec<ProviderProfile>,
    /// Active profile name, if any.
    pub active_profile: Option<String>,
    /// Loaded token collection (if any).
    pub tokens: Option<TokenCollection>,
    /// Generated embeddings (if any).
    pub embeddings: Option<EmbeddingSet>,
    /// Embedding diagnostics for the latest run.
    pub diagnostics: Option<EmbeddingDiagnostics>,
    /// Top nearest neighbors for the selected point.
    pub neighbors: Vec<NeighborMatch>,
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
    /// Whether a (potentially slow) projection is running.
    pub projecting: bool,
    /// Currently selected point info (if any).
    pub selected_point: Option<SelectedPointInfo>,
    /// Selection requested from session restore and applied when projection is available.
    pub pending_selected_index: Option<usize>,
    /// Mirrored camera state from the viewer (for axis indicator).
    pub viewer_camera: ArcballCamera,
    /// Current bubble/point render size.
    pub bubble_size: f32,
    /// Current search query for fuzzy point finding.
    pub search_query: String,
    /// Which dimensionality-reduction algorithm to use.
    pub projection_method: ProjectionMethod,
    /// t-SNE perplexity (5–50).
    pub tsne_perplexity: f32,
    /// Last time session state was flushed to disk.
    pub last_session_save: Instant,
}

/// Information about a selected point (resolved with token label).
#[derive(Debug, Clone)]
pub struct SelectedPointInfo {
    pub index: usize,
    pub label: String,
    pub position: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct NeighborMatch {
    pub index: usize,
    pub label: String,
    pub cosine: f32,
    pub l2: f32,
}

#[derive(Debug, Clone)]
pub struct NearDuplicateExample {
    pub left: String,
    pub right: String,
    pub cosine: f32,
}

#[derive(Debug, Clone)]
pub struct EmbeddingDiagnostics {
    pub count: usize,
    pub dimensions: usize,
    pub finite_vectors: usize,
    pub non_finite_vectors: usize,
    pub zero_vectors: usize,
    pub exact_duplicates: usize,
    pub near_duplicates: usize,
    pub near_duplicate_pairs_sampled: usize,
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    pub std_norm: f32,
    pub near_examples: Vec<NearDuplicateExample>,
}

/// Messages that can be sent to update the application.
#[derive(Debug, Clone)]
pub enum Message {
    // -- Sidebar messages --
    ProfileSelected(String),
    ProfileNameChanged(String),
    SaveProfilePressed,
    DeleteProfilePressed,
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
    /// Projection method changed (PCA / t-SNE).
    ProjectionMethodChanged(String),
    /// t-SNE perplexity slider changed.
    PerplexityChanged(f32),
    /// Async projection finished.
    ProjectionComplete(Vec<ProjectedPoint>),
}

impl Application for App {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let mut provider_config = ProviderConfig::default();
        let mut provider_profiles = Vec::new();
        let mut active_profile = None;

        let mut sidebar = sidebar::SidebarState::default();
        let mut status = "Ready — load a token collection to get started.".to_string();
        let mut bubble_size = 8.0_f32;
        let mut search_query = String::new();
        let mut projection_method = ProjectionMethod::Pca;
        let mut tsne_perplexity = 30.0_f32;
        let mut pending_selected_index = None;
        let mut viewer_camera = ArcballCamera::default();

        match persistence::load_session() {
            Ok(session) => {
                provider_profiles = session.profiles;
                active_profile = session.selected_profile.clone();

                if let Some(ref profile_name) = active_profile {
                    if let Some(profile) =
                        provider_profiles.iter().find(|p| &p.name == profile_name)
                    {
                        profile.apply_to(&mut provider_config);
                    } else {
                        active_profile = None;
                    }
                }

                if active_profile.is_none() {
                    provider_config.kind = session.current_provider;
                    provider_config.model = session.current_model;
                    provider_config.deployment_name = session.current_deployment_name;
                    provider_config.endpoint = session.current_endpoint;
                    provider_config.dimensions = session.current_dimensions;
                }

                bubble_size = session.bubble_size.clamp(2.0, 30.0);
                search_query = session.search_query;
                sidebar.tokens_text = session.tokens_text;
                tsne_perplexity = session.tsne_perplexity.clamp(5.0, 50.0);
                projection_method = if session.projection_method == "t-SNE" {
                    ProjectionMethod::TSne
                } else {
                    ProjectionMethod::Pca
                };
                pending_selected_index = session.selected_index;
                viewer_camera = session.camera.to_camera();

                sidebar.selected_profile = active_profile.clone();
                sidebar.profile_name_input = active_profile.clone().unwrap_or_default();
                sidebar.projection_input = if projection_method == ProjectionMethod::TSne {
                    "t-SNE".to_string()
                } else {
                    "PCA".to_string()
                };
                sidebar.perplexity_input = tsne_perplexity;
            }
            Err(e) => {
                warn!(error = %e, "Failed to load session state");
            }
        }

        provider_config.api_key = credentials::load_api_key(provider_config.kind);

        sidebar.provider_input = provider_config.kind.to_string();
        sidebar.model_input = provider_config.model.clone();
        sidebar.api_key_input = provider_config.api_key.clone().unwrap_or_default();
        sidebar.dimensions_input = provider_config
            .dimensions
            .map(|d| d.to_string())
            .unwrap_or_default();
        sidebar.endpoint_input = provider_config.endpoint.clone().unwrap_or_default();
        sidebar.deployment_input = provider_config.deployment_name.clone().unwrap_or_default();

        let mut app = Self {
            provider_config,
            provider_profiles,
            active_profile,
            tokens: None,
            embeddings: None,
            diagnostics: None,
            neighbors: Vec::new(),
            projected: Vec::new(),
            point_cloud: PointCloud::default(),
            sidebar,
            status: status.clone(),
            loading: false,
            projecting: false,
            selected_point: None,
            pending_selected_index,
            viewer_camera,
            bubble_size,
            search_query,
            projection_method,
            tsne_perplexity,
            last_session_save: Instant::now(),
        };

        if let Some(collection) = parse_manual_collection(&app.sidebar.tokens_text) {
            let n = collection.len();
            app.tokens = Some(collection);
            status = format!("Restored {n} tokens from previous session.");
            app.status = status;
        }

        app.persist_session(true);
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
            // -- Profile management --
            Message::ProfileSelected(name) => {
                if self.apply_profile(&name) {
                    self.status = format!("Loaded profile '{name}'.");
                } else {
                    self.status = format!("Profile '{name}' not found.");
                }
                self.persist_session(true);
                Command::none()
            }
            Message::ProfileNameChanged(name) => {
                self.sidebar.profile_name_input = name;
                Command::none()
            }
            Message::SaveProfilePressed => {
                let name = self.sidebar.profile_name_input.trim().to_string();
                if name.is_empty() {
                    self.status = "Profile name cannot be empty.".into();
                    return Command::none();
                }

                let profile = ProviderProfile::from_config(name.clone(), &self.provider_config);
                if let Some(existing) = self.provider_profiles.iter_mut().find(|p| p.name == name) {
                    *existing = profile;
                } else {
                    self.provider_profiles.push(profile);
                    self.provider_profiles
                        .sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
                }

                self.active_profile = Some(name.clone());
                self.sidebar.selected_profile = Some(name.clone());
                self.sidebar.profile_name_input = name.clone();
                self.status = format!("Saved profile '{name}'.");
                self.persist_session(true);
                Command::none()
            }
            Message::DeleteProfilePressed => {
                let Some(name) = self.sidebar.selected_profile.clone() else {
                    self.status = "No profile selected to delete.".into();
                    return Command::none();
                };

                let before = self.provider_profiles.len();
                self.provider_profiles.retain(|p| p.name != name);
                if self.provider_profiles.len() == before {
                    self.status = format!("Profile '{name}' not found.");
                    return Command::none();
                }

                if self.active_profile.as_deref() == Some(name.as_str()) {
                    self.active_profile = None;
                }
                self.sidebar.selected_profile = None;
                self.sidebar.profile_name_input.clear();
                self.status = format!("Deleted profile '{name}'.");
                self.persist_session(true);
                Command::none()
            }

            // -- Provider configuration --
            Message::ProviderChanged(provider) => {
                self.active_profile = None;
                self.sidebar.selected_profile = None;

                self.provider_config.kind = match provider.as_str() {
                    "OpenAI" => ProviderKind::OpenAI,
                    "Azure" => ProviderKind::Azure,
                    _ => ProviderKind::OpenRouter,
                };

                if self.sidebar.model_input.is_empty() {
                    self.provider_config.model = match provider.as_str() {
                        "OpenAI" => "text-embedding-3-small".into(),
                        _ => "openai/text-embedding-3-small".into(),
                    };
                }

                self.provider_config.api_key = credentials::load_api_key(self.provider_config.kind);
                self.sidebar.api_key_input =
                    self.provider_config.api_key.clone().unwrap_or_default();
                self.sidebar.provider_input = provider;
                self.persist_session(true);
                Command::none()
            }
            Message::ModelChanged(model) => {
                self.provider_config.model = model.clone();
                self.sidebar.model_input = model;
                self.persist_session(true);
                Command::none()
            }
            Message::ApiKeyChanged(key) => {
                self.provider_config.api_key = if key.is_empty() {
                    None
                } else {
                    Some(key.clone())
                };
                self.sidebar.api_key_input = key;
                if let Err(e) = credentials::store_api_key(
                    self.provider_config.kind,
                    &self.sidebar.api_key_input,
                ) {
                    warn!(
                        provider = %self.provider_config.kind,
                        error = %e,
                        "Failed to persist API key in OS keyring"
                    );
                }
                self.persist_session(true);
                Command::none()
            }
            Message::DimensionsChanged(dims) => {
                self.sidebar.dimensions_input = dims.clone();
                self.provider_config.dimensions = dims.parse().ok();
                self.persist_session(true);
                Command::none()
            }
            Message::EndpointChanged(url) => {
                self.provider_config.endpoint = if url.is_empty() {
                    None
                } else {
                    Some(url.clone())
                };
                self.sidebar.endpoint_input = url;
                self.persist_session(true);
                Command::none()
            }
            Message::DeploymentChanged(name) => {
                self.provider_config.deployment_name = if name.is_empty() {
                    None
                } else {
                    Some(name.clone())
                };
                self.sidebar.deployment_input = name;
                self.persist_session(true);
                Command::none()
            }

            // -- Token loading --
            Message::LoadTokensPressed => Command::perform(
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
                            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                            if ext == "json" {
                                TokenCollection::from_json_file(&path).map_err(|e| e.to_string())
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
            ),
            Message::TokensLoaded(result) => {
                match result {
                    Ok(collection) => {
                        self.status = format!(
                            "Loaded {} tokens from '{}'.",
                            collection.len(),
                            collection.name
                        );
                        self.sidebar.tokens_text = collection.texts().join(", ");
                        self.tokens = Some(collection);
                    }
                    Err(e) => {
                        self.status = format!("Failed to load tokens: {e}");
                    }
                }
                self.persist_session(true);
                Command::none()
            }
            Message::TokensTextChanged(s) => {
                self.sidebar.tokens_text = s;
                self.persist_session(true);
                Command::none()
            }
            Message::UseTypedTokens => {
                if let Some(collection) = parse_manual_collection(&self.sidebar.tokens_text) {
                    let count = collection.len();
                    self.tokens = Some(collection);
                    self.status = format!("Using {count} tokens from text input.");
                } else {
                    self.status = "No tokens found in the input field.".into();
                }
                self.persist_session(true);
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
                    let texts: Vec<String> =
                        tokens.texts().into_iter().map(|s| s.to_string()).collect();

                    Command::perform(
                        async move {
                            let client = EmbeddingClient::new(config);
                            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                            client.embed(&text_refs).await.map_err(|e| e.to_string())
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
                            "Generated {} embeddings ({}D) — projecting to 3D…",
                            set.len(),
                            set.dimensions
                        );
                        self.diagnostics = Some(compute_embedding_diagnostics(&set));
                        self.embeddings = Some(set);
                        if let Some(sel) = self.selected_point.as_ref().map(|s| s.index) {
                            self.neighbors = self.compute_neighbors(sel, NEIGHBOR_COUNT);
                        } else {
                            self.neighbors.clear();
                        }
                        self.persist_session(true);
                        return self.start_projection();
                    }
                    Err(e) => {
                        self.status = format!("Embedding generation failed: {e}");
                    }
                }
                Command::none()
            }

            // -- Viewer interaction --
            Message::PointSelected(sel) => {
                if let Some(s) = sel {
                    self.selected_point = Some(SelectedPointInfo {
                        index: s.index,
                        label: self.label_for_index(s.index),
                        position: s.position,
                    });
                    self.pending_selected_index = Some(s.index);
                    self.neighbors = self.compute_neighbors(s.index, NEIGHBOR_COUNT);
                } else {
                    self.selected_point = None;
                    self.pending_selected_index = None;
                    self.neighbors.clear();
                }
                self.persist_session(true);
                Command::none()
            }
            Message::ViewerCameraChanged(cam) => {
                self.viewer_camera = cam;
                self.persist_session(false);
                Command::none()
            }

            Message::BubbleSizeChanged(size) => {
                self.bubble_size = size;
                self.point_cloud.set_point_size(size);
                self.persist_session(false);
                Command::none()
            }

            Message::SearchQueryChanged(query) => {
                self.search_query = query.clone();
                if query.trim().is_empty() {
                    self.selected_point = None;
                    self.pending_selected_index = None;
                    self.neighbors.clear();
                } else if let Some(tokens) = &self.tokens {
                    if let Some((idx, _score)) = fuzzy_find(&query, tokens) {
                        self.pending_selected_index = Some(idx);
                        self.neighbors = self.compute_neighbors(idx, NEIGHBOR_COUNT);
                        if let Some(point) = self.point_cloud.points.iter().find(|p| p.index == idx)
                        {
                            self.selected_point = Some(SelectedPointInfo {
                                index: idx,
                                label: self.label_for_index(idx),
                                position: point.position,
                            });
                        }
                    }
                }
                self.persist_session(false);
                Command::none()
            }

            Message::ProjectionMethodChanged(s) => {
                self.projection_method = if s == "t-SNE" {
                    ProjectionMethod::TSne
                } else {
                    ProjectionMethod::Pca
                };
                self.sidebar.projection_input = s;
                self.persist_session(true);
                if self.embeddings.is_some() {
                    return self.start_projection();
                }
                Command::none()
            }

            Message::PerplexityChanged(v) => {
                self.tsne_perplexity = v;
                self.sidebar.perplexity_input = v;
                self.persist_session(false);
                Command::none()
            }

            Message::ProjectionComplete(mut pts) => {
                self.projecting = false;
                Projector::normalize(&mut pts);
                let n = pts.len();
                self.projected = pts;
                let colors = self.cluster_colors();
                self.point_cloud.set_points(&self.projected, &colors);
                self.point_cloud.set_point_size(self.bubble_size);

                let restore_idx = self
                    .pending_selected_index
                    .or_else(|| self.selected_point.as_ref().map(|s| s.index));
                if let Some(idx) = restore_idx {
                    self.select_index_if_visible(idx);
                }

                self.status = format!(
                    "Showing {n} points in 3D ({method}). Drag to rotate, scroll to zoom.",
                    method = match self.projection_method {
                        ProjectionMethod::Pca => "PCA",
                        ProjectionMethod::TSne => "t-SNE",
                    }
                );
                self.persist_session(true);
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let profile_names: Vec<String> = self
            .provider_profiles
            .iter()
            .map(|p| p.name.clone())
            .collect();

        let sidebar = sidebar::view(
            &self.sidebar,
            profile_names,
            &self.tokens,
            self.loading || self.projecting,
            self.bubble_size,
        );
        let viewer = viewer::view(
            &self.point_cloud,
            &self.embeddings,
            &self.selected_point,
            &self.viewer_camera,
            &self.search_query,
            &self.diagnostics,
            &self.neighbors,
        );
        let status = status_bar::view(&self.status);

        let main_content = row![sidebar, viewer].spacing(0).height(Length::Fill);

        container(column![main_content, status])
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

impl App {
    /// Color each token by its k-means cluster in the projected 3D space.
    fn cluster_colors(&self) -> Vec<Color> {
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

        if self.projected.is_empty() {
            return Vec::new();
        }

        let k = palette.len().min(self.projected.len());
        let assignments = Projector::kmeans(&self.projected, k);
        assignments
            .iter()
            .map(|&cluster| palette[cluster % palette.len()])
            .collect()
    }

    /// Launch an async projection task using the current method and embeddings.
    fn start_projection(&mut self) -> Command<Message> {
        let set = match &self.embeddings {
            Some(s) => s,
            None => return Command::none(),
        };

        self.projecting = true;
        let method = self.projection_method;
        let perplexity = self.tsne_perplexity;
        let raw: Vec<Vec<f32>> = set.embeddings.iter().map(|e| e.vector.clone()).collect();

        self.status = format!(
            "Projecting with {}…",
            match method {
                ProjectionMethod::Pca => "PCA",
                ProjectionMethod::TSne => "t-SNE",
            }
        );

        Command::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let refs: Vec<&[f32]> = raw.iter().map(|v| v.as_slice()).collect();
                    match method {
                        ProjectionMethod::Pca => Projector::pca(&refs),
                        ProjectionMethod::TSne => {
                            let params = TsneParams {
                                perplexity,
                                ..Default::default()
                            };
                            Projector::tsne(&refs, &params)
                        }
                    }
                })
                .await
                .unwrap_or_default()
            },
            Message::ProjectionComplete,
        )
    }

    fn apply_profile(&mut self, name: &str) -> bool {
        let Some(profile) = self
            .provider_profiles
            .iter()
            .find(|p| p.name == name)
            .cloned()
        else {
            return false;
        };

        profile.apply_to(&mut self.provider_config);
        self.provider_config.api_key = credentials::load_api_key(self.provider_config.kind);

        self.active_profile = Some(profile.name.clone());
        self.sidebar.selected_profile = Some(profile.name.clone());
        self.sidebar.profile_name_input = profile.name;
        self.sync_sidebar_from_provider();
        true
    }

    fn sync_sidebar_from_provider(&mut self) {
        self.sidebar.provider_input = self.provider_config.kind.to_string();
        self.sidebar.model_input = self.provider_config.model.clone();
        self.sidebar.api_key_input = self.provider_config.api_key.clone().unwrap_or_default();
        self.sidebar.dimensions_input = self
            .provider_config
            .dimensions
            .map(|d| d.to_string())
            .unwrap_or_default();
        self.sidebar.endpoint_input = self.provider_config.endpoint.clone().unwrap_or_default();
        self.sidebar.deployment_input = self
            .provider_config
            .deployment_name
            .clone()
            .unwrap_or_default();
    }

    fn label_for_index(&self, idx: usize) -> String {
        self.tokens
            .as_ref()
            .and_then(|tc| tc.tokens.get(idx))
            .map(|t| t.text.clone())
            .or_else(|| {
                self.embeddings
                    .as_ref()
                    .and_then(|set| set.embeddings.get(idx))
                    .map(|e| e.label.clone())
            })
            .unwrap_or_else(|| format!("Point {idx}"))
    }

    fn select_index_if_visible(&mut self, idx: usize) {
        if let Some(point) = self.point_cloud.points.iter().find(|p| p.index == idx) {
            self.selected_point = Some(SelectedPointInfo {
                index: idx,
                label: self.label_for_index(idx),
                position: point.position,
            });
            self.pending_selected_index = Some(idx);
            self.neighbors = self.compute_neighbors(idx, NEIGHBOR_COUNT);
        }
    }

    fn compute_neighbors(&self, index: usize, k: usize) -> Vec<NeighborMatch> {
        let Some(set) = &self.embeddings else {
            return Vec::new();
        };
        if index >= set.embeddings.len() {
            return Vec::new();
        }

        let src = &set.embeddings[index].vector;
        let src_norm = l2_norm(src);
        if src_norm <= 1e-8 {
            return Vec::new();
        }

        let mut matches: Vec<NeighborMatch> = set
            .embeddings
            .iter()
            .enumerate()
            .filter_map(|(i, emb)| {
                if i == index {
                    return None;
                }
                if emb.vector.len() != src.len() {
                    return None;
                }
                let norm = l2_norm(&emb.vector);
                if norm <= 1e-8 {
                    return None;
                }

                let mut dot = 0.0_f32;
                let mut dist2 = 0.0_f32;
                for (&a, &b) in src.iter().zip(emb.vector.iter()) {
                    if !a.is_finite() || !b.is_finite() {
                        return None;
                    }
                    dot += a * b;
                    let d = a - b;
                    dist2 += d * d;
                }

                Some(NeighborMatch {
                    index: i,
                    label: self.label_for_index(i),
                    cosine: dot / (src_norm * norm),
                    l2: dist2.sqrt(),
                })
            })
            .collect();

        matches.sort_by(|a, b| {
            b.cosine
                .partial_cmp(&a.cosine)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.l2.partial_cmp(&b.l2).unwrap_or(Ordering::Equal))
        });
        matches.truncate(k);
        matches
    }

    fn session_snapshot(&self) -> persistence::SessionState {
        persistence::SessionState {
            profiles: self.provider_profiles.clone(),
            selected_profile: self.active_profile.clone(),
            current_provider: self.provider_config.kind,
            current_model: self.provider_config.model.clone(),
            current_deployment_name: self.provider_config.deployment_name.clone(),
            current_endpoint: self.provider_config.endpoint.clone(),
            current_dimensions: self.provider_config.dimensions,
            projection_method: if self.projection_method == ProjectionMethod::TSne {
                "t-SNE".to_string()
            } else {
                "PCA".to_string()
            },
            tsne_perplexity: self.tsne_perplexity,
            bubble_size: self.bubble_size,
            search_query: self.search_query.clone(),
            tokens_text: self.sidebar.tokens_text.clone(),
            selected_index: self
                .selected_point
                .as_ref()
                .map(|s| s.index)
                .or(self.pending_selected_index),
            camera: persistence::CameraSnapshot::from_camera(&self.viewer_camera),
        }
    }

    fn persist_session(&mut self, force: bool) {
        if !force && self.last_session_save.elapsed() < SESSION_SAVE_DEBOUNCE {
            return;
        }

        let snapshot = self.session_snapshot();
        if let Err(e) = persistence::save_session(&snapshot) {
            warn!(error = %e, "Failed to save session state");
            return;
        }

        self.last_session_save = Instant::now();
    }
}

fn parse_manual_collection(raw: &str) -> Option<TokenCollection> {
    let tokens: Vec<embedding_core::Token> = raw
        .split([',', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(embedding_core::Token::new)
        .collect();

    if tokens.is_empty() {
        return None;
    }

    let mut col = TokenCollection::new("Manual");
    for t in tokens {
        col.push(t);
    }
    Some(col)
}

fn l2_norm(v: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in v {
        if !x.is_finite() {
            return 0.0;
        }
        sum += x * x;
    }
    sum.sqrt()
}

fn compute_embedding_diagnostics(set: &EmbeddingSet) -> EmbeddingDiagnostics {
    let mut norms = Vec::with_capacity(set.embeddings.len());
    let mut non_finite_vectors = 0_usize;
    let mut zero_vectors = 0_usize;

    let mut exact_hash_counts: HashMap<u64, usize> = HashMap::new();
    let mut candidates = Vec::new();

    for (idx, emb) in set.embeddings.iter().enumerate() {
        let mut has_non_finite = false;
        let mut norm2 = 0.0_f32;
        let mut hasher = DefaultHasher::new();

        for &x in &emb.vector {
            if !x.is_finite() {
                has_non_finite = true;
                break;
            }
            x.to_bits().hash(&mut hasher);
            norm2 += x * x;
        }

        if has_non_finite {
            non_finite_vectors += 1;
            continue;
        }

        let norm = norm2.sqrt();
        if norm <= 1e-8 {
            zero_vectors += 1;
        }

        norms.push(norm);
        *exact_hash_counts.entry(hasher.finish()).or_insert(0) += 1;
        if norm > 1e-8 {
            candidates.push(idx);
        }
    }

    let exact_duplicates = exact_hash_counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum();

    let (mean_norm, std_norm, min_norm, max_norm) = if norms.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let mean = norms.iter().sum::<f32>() / norms.len() as f32;
        let var = norms
            .iter()
            .map(|n| {
                let d = *n - mean;
                d * d
            })
            .sum::<f32>()
            / norms.len() as f32;
        let min = norms.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
        let max = norms
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        (mean, var.sqrt(), min, max)
    };

    let sample_n = candidates.len().min(NEAR_DUP_MAX_SAMPLE);
    let sample = &candidates[..sample_n];
    let mut near_duplicates = 0_usize;
    let mut near_examples = Vec::<(usize, usize, f32)>::new();

    for i in 0..sample_n {
        for j in (i + 1)..sample_n {
            let ai = sample[i];
            let bi = sample[j];
            let a = &set.embeddings[ai].vector;
            let b = &set.embeddings[bi].vector;
            if a.len() != b.len() {
                continue;
            }

            let mut dot = 0.0_f32;
            let mut norm_a2 = 0.0_f32;
            let mut norm_b2 = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                norm_a2 += x * x;
                norm_b2 += y * y;
            }

            let denom = (norm_a2.sqrt() * norm_b2.sqrt()).max(1e-8);
            let cosine = dot / denom;
            if cosine >= NEAR_DUP_THRESHOLD {
                near_duplicates += 1;
                near_examples.push((ai, bi, cosine));
            }
        }
    }

    near_examples.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    near_examples.truncate(NEAR_DUP_MAX_EXAMPLES);

    let near_examples = near_examples
        .into_iter()
        .map(|(ai, bi, cosine)| NearDuplicateExample {
            left: set.embeddings[ai].label.clone(),
            right: set.embeddings[bi].label.clone(),
            cosine,
        })
        .collect();

    EmbeddingDiagnostics {
        count: set.len(),
        dimensions: set.dimensions,
        finite_vectors: set.len().saturating_sub(non_finite_vectors),
        non_finite_vectors,
        zero_vectors,
        exact_duplicates,
        near_duplicates,
        near_duplicate_pairs_sampled: sample_n.saturating_mul(sample_n.saturating_sub(1)) / 2,
        min_norm,
        max_norm,
        mean_norm,
        std_norm,
        near_examples,
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

    for (tier, pred) in [
        (
            3u32,
            Box::new(|t: &str| t.to_lowercase() == q) as Box<dyn Fn(&str) -> bool>,
        ),
        (2, Box::new(|t: &str| t.to_lowercase().starts_with(&q))),
        (1, Box::new(|t: &str| t.to_lowercase().contains(&q))),
    ] {
        if let Some(idx) = tokens.tokens.iter().position(|t| pred(&t.text)) {
            return Some((idx, tier));
        }
    }

    let best = tokens
        .tokens
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
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
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
