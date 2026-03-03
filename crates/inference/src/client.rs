use anyhow::{anyhow, Result};
use embedding_core::{Embedding, EmbeddingSet};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::provider::{ProviderConfig, ProviderKind};

// ---------- Wire types for the embeddings REST API ----------

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [&'a str],
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedObject>,
    model: String,
}

#[derive(Deserialize)]
struct EmbedObject {
    embedding: Vec<f32>,
    index: usize,
}

// -----------------------------------------------------------

/// Client for generating embeddings via a configured provider.
pub struct EmbeddingClient {
    config: ProviderConfig,
    http: reqwest::Client,
}

impl EmbeddingClient {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Generate embeddings for a batch of texts.
    pub async fn embed(&self, texts: &[&str]) -> Result<EmbeddingSet> {
        info!(
            model = %self.config.model,
            count = texts.len(),
            "Generating embeddings"
        );

        match self.config.kind {
            ProviderKind::Azure => self.embed_azure(texts).await,
            ProviderKind::OpenAI => self.embed_openai_compat(texts).await,
            ProviderKind::OpenRouter => self.embed_openai_compat(texts).await,
        }
    }

    // ---- Azure OpenAI (both traditional and Azure AI Foundry endpoints) ------
    //
    // Two supported URL styles:
    //
    //  a) New-style / OpenAI-compatible (Azure AI Foundry, cognitiveservices.azure.com):
    //       https://<resource>.cognitiveservices.azure.com/openai/v1/
    //       → POST {endpoint}/embeddings
    //       → model sent in request body as deployment name
    //
    //  b) Traditional Azure OpenAI (*.openai.azure.com):
    //       https://<resource>.openai.azure.com/
    //       → POST {endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01
    //       → deployment encoded in URL, model field still echoed in body
    //
    // Both accept the `api-key` header for API-key authentication.

    async fn embed_azure(&self, texts: &[&str]) -> Result<EmbeddingSet> {
        let endpoint = self
            .config
            .endpoint
            .as_deref()
            .ok_or_else(|| anyhow!("Azure requires an endpoint URL"))?;

        let api_key = self
            .config
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow!("Azure requires an api_key"))?;

        // Deployment name drives both the URL path and the model field.
        let deployment = self
            .config
            .deployment_name
            .as_deref()
            .unwrap_or(self.config.model.as_str());

        let base = endpoint.trim_end_matches('/');

        // Detect endpoint style from the URL.
        // New-style endpoints contain "/openai/v1" in the path.
        // Traditional endpoints are bare resource roots (*.openai.azure.com).
        let url = if base.contains("/openai/v1") {
            // New Azure AI Foundry / OpenAI-compatible endpoint — append /embeddings.
            format!("{base}/embeddings")
        } else {
            // Traditional Azure OpenAI — deployment goes in the URL path.
            format!(
                "{base}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01"
            )
        };

        let body = EmbedRequest {
            input: texts,
            model: deployment,
            dimensions: self.config.dimensions,
        };

        let mut headers = HeaderMap::new();
        headers.insert("api-key", HeaderValue::from_str(api_key)?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let resp: EmbedResponse = self
            .http
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        self.build_set(texts, &resp)
    }

    // ---- OpenAI / OpenRouter ------------------------------------------------

    async fn embed_openai_compat(&self, texts: &[&str]) -> Result<EmbeddingSet> {
        let base_url = self
            .config
            .endpoint
            .as_deref()
            .unwrap_or(match self.config.kind {
                ProviderKind::OpenAI => "https://api.openai.com/v1/",
                _ => "https://openrouter.ai/api/v1/",
            });

        let api_key = self
            .config
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow!("An api_key is required"))?;

        let url = format!("{}/embeddings", base_url.trim_end_matches('/'));

        let body = EmbedRequest {
            input: texts,
            model: &self.config.model,
            dimensions: self.config.dimensions,
        };

        let resp: EmbedResponse = self
            .http
            .post(&url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        self.build_set(texts, &resp)
    }

    // ---- Shared helpers -----------------------------------------------------

    fn build_set(&self, texts: &[&str], resp: &EmbedResponse) -> Result<EmbeddingSet> {
        let dims = resp
            .data
            .first()
            .map(|e| e.embedding.len())
            .unwrap_or(self.config.dimensions.unwrap_or(1536));

        let mut set = EmbeddingSet::new(&resp.model, dims);

        // The API returns embeddings sorted by index, but respect `index` just in case.
        let mut pairs: Vec<(usize, Vec<f32>)> = resp
            .data
            .iter()
            .map(|e| (e.index, e.embedding.clone()))
            .collect();
        pairs.sort_by_key(|(i, _)| *i);

        for (i, vector) in pairs {
            let label = texts.get(i).copied().unwrap_or("");
            set.push(Embedding::new(label, vector));
        }

        Ok(set)
    }

    /// Deterministic placeholder — used in tests / when no provider is configured.
    #[allow(dead_code)]
    fn placeholder_vector(index: usize, dims: usize) -> Vec<f32> {
        (0..dims)
            .map(|d| {
                let seed = (index * 7919 + d * 104729) as f32;
                (seed.sin() * 0.5).clamp(-1.0, 1.0)
            })
            .collect()
    }
}
