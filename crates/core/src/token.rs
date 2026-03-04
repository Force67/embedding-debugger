use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenError {
    #[error("failed to read token file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse token file: {0}")]
    Parse(String),
}

/// A single token or term to be embedded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// The text content of the token.
    pub text: String,
}

impl Token {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A collection of tokens/terms to be embedded and visualized.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenCollection {
    pub name: String,
    pub tokens: Vec<Token>,
}

impl TokenCollection {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tokens: Vec::new(),
        }
    }

    pub fn push(&mut self, token: Token) {
        self.tokens.push(token);
    }

    /// Load tokens from a JSON file.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "name": "Sample Collection",
    ///   "tokens": [
    ///     { "text": "king" },
    ///     { "text": "queen" },
    ///     { "text": "apple" }
    ///   ]
    /// }
    /// ```
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, TokenError> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| TokenError::Parse(e.to_string()))
    }

    /// Load tokens from a simple text file (one token per line).
    pub fn from_text_file(
        path: impl AsRef<Path>,
        name: impl Into<String>,
    ) -> Result<Self, TokenError> {
        let content = std::fs::read_to_string(path)?;
        let tokens = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Token::new(line.trim()))
            .collect();

        Ok(Self {
            name: name.into(),
            tokens,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get all token texts as strings (for sending to the embedding API).
    pub fn texts(&self) -> Vec<&str> {
        self.tokens.iter().map(|t| t.text.as_str()).collect()
    }
}
