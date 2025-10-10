//! Configuration for HfXetClient

use std::path::PathBuf;

/// Configuration for XET client
#[derive(Debug, Clone)]
pub struct XetConfig {
    /// Cache directory path
    pub cache_dir: PathBuf,
    
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    
    /// Hugging Face Hub endpoint
    pub hub_endpoint: String,
    
    /// CAS endpoint
    pub cas_endpoint: String,
    
    /// Number of concurrent chunk downloads
    pub max_concurrent_chunks: usize,
    
    /// HTTP timeout in seconds
    pub timeout_secs: u64,
    
    /// Retry attempts for failed requests
    pub max_retries: u32,
}

impl Default for XetConfig {
    fn default() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("huggingface"),
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10 GB
            hub_endpoint: "https://huggingface.co".to_string(),
            cas_endpoint: "https://cas.huggingface.co".to_string(),
            max_concurrent_chunks: 8,
            timeout_secs: 300,
            max_retries: 3,
        }
    }
}

impl XetConfig {
    /// Create a new config with custom cache directory
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }
    
    /// Set maximum cache size
    pub fn with_max_cache_size(mut self, size: u64) -> Self {
        self.max_cache_size = size;
        self
    }
    
    /// Set Hub endpoint
    pub fn with_hub_endpoint(mut self, endpoint: String) -> Self {
        self.hub_endpoint = endpoint;
        self
    }
    
    /// Set CAS endpoint
    pub fn with_cas_endpoint(mut self, endpoint: String) -> Self {
        self.cas_endpoint = endpoint;
        self
    }
}
