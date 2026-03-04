use std::fs;
use std::path::PathBuf;

use directories::ProjectDirs;
use embedding_inference::{ProviderConfig, ProviderKind};
use embedding_viz::ArcballCamera;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub name: String,
    pub kind: ProviderKind,
    pub model: String,
    pub deployment_name: Option<String>,
    pub endpoint: Option<String>,
    pub dimensions: Option<usize>,
}

impl ProviderProfile {
    pub fn from_config(name: String, cfg: &ProviderConfig) -> Self {
        Self {
            name,
            kind: cfg.kind,
            model: cfg.model.clone(),
            deployment_name: cfg.deployment_name.clone(),
            endpoint: cfg.endpoint.clone(),
            dimensions: cfg.dimensions,
        }
    }

    pub fn apply_to(&self, cfg: &mut ProviderConfig) {
        cfg.kind = self.kind;
        cfg.model = self.model.clone();
        cfg.deployment_name = self.deployment_name.clone();
        cfg.endpoint = self.endpoint.clone();
        cfg.dimensions = self.dimensions;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSnapshot {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
}

impl CameraSnapshot {
    pub fn from_camera(camera: &ArcballCamera) -> Self {
        Self {
            target: [camera.target.x, camera.target.y, camera.target.z],
            distance: camera.distance,
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov: camera.fov,
        }
    }

    pub fn to_camera(&self) -> ArcballCamera {
        let mut camera = ArcballCamera::default();
        camera.target.x = self.target[0];
        camera.target.y = self.target[1];
        camera.target.z = self.target[2];
        camera.distance = self.distance;
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        camera.fov = self.fov;
        camera
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub profiles: Vec<ProviderProfile>,
    pub selected_profile: Option<String>,

    pub current_provider: ProviderKind,
    pub current_model: String,
    pub current_deployment_name: Option<String>,
    pub current_endpoint: Option<String>,
    pub current_dimensions: Option<usize>,

    pub projection_method: String,
    pub tsne_perplexity: f32,
    pub bubble_size: f32,
    pub search_query: String,
    pub tokens_text: String,
    pub selected_index: Option<usize>,
    pub camera: CameraSnapshot,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            profiles: Vec::new(),
            selected_profile: None,
            current_provider: ProviderKind::OpenRouter,
            current_model: "openai/text-embedding-3-small".to_string(),
            current_deployment_name: None,
            current_endpoint: None,
            current_dimensions: None,
            projection_method: "PCA".to_string(),
            tsne_perplexity: 30.0,
            bubble_size: 8.0,
            search_query: String::new(),
            tokens_text: String::new(),
            selected_index: None,
            camera: CameraSnapshot::from_camera(&ArcballCamera::default()),
        }
    }
}

fn session_path() -> Result<PathBuf, String> {
    let dirs = ProjectDirs::from("io", "force67", "embedding-debugger")
        .ok_or_else(|| "Unable to determine config directory".to_string())?;
    Ok(dirs.config_dir().join("session.json"))
}

pub fn load_session() -> Result<SessionState, String> {
    let path = session_path()?;
    if !path.exists() {
        return Ok(SessionState::default());
    }
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read '{}': {e}", path.display()))?;
    serde_json::from_str::<SessionState>(&raw)
        .map_err(|e| format!("Failed to parse '{}': {e}", path.display()))
}

pub fn save_session(state: &SessionState) -> Result<(), String> {
    let path = session_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create '{}': {e}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(state)
        .map_err(|e| format!("Failed to serialize session: {e}"))?;
    fs::write(&path, raw).map_err(|e| format!("Failed to write '{}': {e}", path.display()))
}
