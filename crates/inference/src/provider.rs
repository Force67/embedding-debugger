use serde::{Deserialize, Serialize};

/// Supported embedding API providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    OpenRouter,
    OpenAI,
    Azure,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenRouter => write!(f, "OpenRouter"),
            Self::OpenAI => write!(f, "OpenAI"),
            Self::Azure => write!(f, "Azure"),
        }
    }
}

/// Configuration for connecting to an embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    /// API key.
    pub api_key: Option<String>,
    /// Model identifier (e.g. "openai/text-embedding-3-small").
    /// For Azure this is the deployment name used in the request body.
    pub model: String,
    /// Azure: deployment name (e.g. "text-embedding-3-large").
    /// When set this overrides `model` as the value sent to the API.
    pub deployment_name: Option<String>,
    /// Azure: full endpoint URL
    ///   e.g. "https://myresource.cognitiveservices.azure.com/openai/v1/"
    /// OpenAI-compatible providers: custom base URL.
    pub endpoint: Option<String>,
    /// Optional: requested embedding dimensions (if the model supports it).
    pub dimensions: Option<usize>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            kind: ProviderKind::OpenRouter,
            api_key: None,
            model: "openai/text-embedding-3-small".to_string(),
            deployment_name: None,
            endpoint: None,
            dimensions: None,
        }
    }
}
