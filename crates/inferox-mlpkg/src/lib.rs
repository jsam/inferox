//! Model package format for Inferox
//!
//! This crate provides functionality for packaging and loading ML models
//! with their weights in a portable format. It integrates with Hugging Face
//! Hub for efficient model downloads using XET technology.

#![warn(missing_docs)]

use std::path::PathBuf;

/// Model package errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Hugging Face download error
    #[error("HF download error: {0}")]
    HfDownload(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid package format
    #[error("Invalid package format: {0}")]
    InvalidFormat(String),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

/// Result type for model package operations
pub type Result<T> = std::result::Result<T, Error>;

/// Model package metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackageMetadata {
    /// Package format version
    pub version: String,

    /// Model name
    pub name: String,

    /// Model repository ID on Hugging Face
    pub repo_id: String,

    /// Model revision/commit
    pub revision: String,

    /// List of files included in the package
    pub files: Vec<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Model package manager
pub struct PackageManager {
    #[allow(dead_code)]
    cache_dir: PathBuf,
    hf_client: hf_xet_rs::HfXetClient,
}

impl PackageManager {
    /// Create a new package manager
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let hf_client = hf_xet_rs::HfXetClient::with_cache_dir(cache_dir.clone())
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(Self {
            cache_dir,
            hf_client,
        })
    }

    /// Download a model package from Hugging Face
    pub async fn download(&self, repo_id: &str, revision: Option<&str>) -> Result<PathBuf> {
        let path = self
            .hf_client
            .snapshot_download(repo_id, revision, None, None)
            .await
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(path)
    }

    /// Download specific files from a repository
    pub async fn download_files(
        &self,
        repo_id: &str,
        patterns: &[&str],
        revision: Option<&str>,
    ) -> Result<Vec<PathBuf>> {
        let paths = self
            .hf_client
            .download_files(repo_id, patterns, revision)
            .await
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(paths)
    }

    /// Load package metadata
    pub fn load_metadata(&self, package_path: &std::path::Path) -> Result<PackageMetadata> {
        let metadata_path = package_path.join("metadata.json");
        let content = std::fs::read_to_string(metadata_path)?;
        let metadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }
}
