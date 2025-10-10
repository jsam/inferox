//! Data types for HfXet

use serde::{Deserialize, Serialize};

/// Repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoInfo {
    /// Repository ID
    #[serde(rename = "modelId")]
    pub id: String,
    
    /// Repository author
    #[serde(default)]
    pub author: String,
    
    /// Current commit SHA
    #[serde(default)]
    pub sha: String,
    
    /// Last modified timestamp
    #[serde(rename = "lastModified", default = "default_datetime")]
    pub last_modified: chrono::DateTime<chrono::Utc>,
    
    /// Whether repository is private
    #[serde(default)]
    pub private: bool,
    
    /// Whether repository is disabled
    #[serde(default)]
    pub disabled: bool,
    
    /// Number of downloads
    #[serde(default)]
    pub downloads: u64,
    
    /// Number of likes
    #[serde(default)]
    pub likes: u64,
    
    /// Repository tags
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Files in repository
    #[serde(default)]
    pub siblings: Vec<FileInfo>,
}

fn default_datetime() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

/// File information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// File path
    #[serde(rename = "rfilename")]
    pub path: String,
    
    /// File size in bytes (optional as API doesn't always provide it)
    #[serde(default)]
    pub size: u64,
    
    /// Blob ID
    #[serde(default)]
    pub blob_id: String,
    
    /// LFS pointer if file is stored in LFS
    #[serde(default)]
    pub lfs: Option<LfsPointer>,
}

/// LFS pointer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfsPointer {
    /// OID hash
    pub oid: String,
    
    /// File size
    pub size: u64,
}

/// XET pointer for XET-backed files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetPointer {
    /// XET hash
    pub xet_hash: String,
    
    /// File size
    pub size: u64,
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    /// Total size of cached data in bytes
    pub total_size: u64,
    
    /// Number of chunks in cache
    pub chunk_count: usize,
    
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// File name being downloaded
    pub file_name: String,
    
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    
    /// Total bytes to download
    pub total_bytes: u64,
    
    /// Chunks completed
    pub chunks_completed: usize,
    
    /// Total chunks
    pub chunks_total: usize,
}
