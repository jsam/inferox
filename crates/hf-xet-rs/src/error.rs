//! Error types for hf-xet-rs

/// Error type for hf-xet-rs operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Network error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Repository not found
    #[error("Repository not found: {0}")]
    RepoNotFound(String),

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// Chunk reconstruction failed
    #[error("Chunk reconstruction failed: {0}")]
    Reconstruction(String),

    /// Invalid XET pointer
    #[error("Invalid XET pointer: {0}")]
    InvalidPointer(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for hf-xet-rs operations
pub type Result<T> = std::result::Result<T, Error>;
