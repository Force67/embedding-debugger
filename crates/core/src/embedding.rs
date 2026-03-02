use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A single embedding vector with its associated label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Human-readable label (the token or term this embedding represents).
    pub label: String,
    /// The raw high-dimensional embedding vector.
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(label: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            label: label.into(),
            vector,
        }
    }

    /// Return the dimensionality of this embedding.
    pub fn dims(&self) -> usize {
        self.vector.len()
    }

    /// Convert to an nalgebra dynamic vector for math operations.
    pub fn as_dvector(&self) -> DVector<f32> {
        DVector::from_vec(self.vector.clone())
    }

    /// Compute cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let a = self.as_dvector();
        let b = other.as_dvector();
        let dot = a.dot(&b);
        let norm_a = a.norm();
        let norm_b = b.norm();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

/// A collection of embeddings, typically produced from a single batch request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingSet {
    /// The model that generated these embeddings.
    pub model: String,
    /// The dimensionality of the embedding space.
    pub dimensions: usize,
    /// All embeddings in this set.
    pub embeddings: Vec<Embedding>,
}

impl EmbeddingSet {
    pub fn new(model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            model: model.into(),
            dimensions,
            embeddings: Vec::new(),
        }
    }

    pub fn push(&mut self, embedding: Embedding) {
        self.embeddings.push(embedding);
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get only the vectors as a flat matrix (rows = points, cols = dims).
    pub fn vectors_as_matrix(&self) -> Vec<&[f32]> {
        self.embeddings.iter().map(|e| e.vector.as_slice()).collect()
    }
}
