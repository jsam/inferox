//! Main HfXet client implementation

use crate::{
    cache::ChunkCache,
    config::XetConfig,
    hf_api::HfApiClient,
    xet::XetProtocolClient,
    CacheStats, FileInfo, RepoInfo, Result,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::fs;

/// Main client for Hugging Face downloads with XET
pub struct HfXetClient {
    config: XetConfig,
    hf_api: HfApiClient,
    #[allow(dead_code)]
    xet_client: XetProtocolClient,
    chunk_cache: ChunkCache,
    token: Option<String>,
    cache_hits: Arc<AtomicU64>,
    cache_misses: Arc<AtomicU64>,
}

impl HfXetClient {
    /// Create a new client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(XetConfig::default())
    }
    
    /// Create a client with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        let config = XetConfig::default().with_cache_dir(cache_dir);
        Self::with_config(config)
    }
    
    /// Create a client with custom configuration
    pub fn with_config(config: XetConfig) -> Result<Self> {
        let token = Self::load_token();
        
        let hf_api = HfApiClient::new(config.hub_endpoint.clone(), token.clone())?;
        
        let chunk_cache = ChunkCache::new(
            config.cache_dir.join("xet").join("chunks"),
            config.max_cache_size,
        )?;
        
        let xet_client = XetProtocolClient::new(
            config.cas_endpoint.clone(),
            ChunkCache::new(
                config.cache_dir.join("xet").join("chunks"),
                config.max_cache_size,
            )?,
        )?;
        
        Ok(Self {
            config,
            hf_api,
            xet_client,
            chunk_cache,
            token,
            cache_hits: Arc::new(AtomicU64::new(0)),
            cache_misses: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Set authentication token
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }
    
    /// Download a specific file from a repository
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        let file_info = self.hf_api.file_info(repo_id, filename, revision).await?;
        
        let snapshot_dir = self.snapshot_dir(repo_id, revision.unwrap_or("main"));
        fs::create_dir_all(&snapshot_dir).await?;
        
        let output_path = snapshot_dir.join(filename);
        
        if output_path.exists() {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(output_path);
        }
        
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        if file_info.lfs.is_some() {
            self.download_lfs_file(repo_id, filename, revision, &output_path).await?;
        } else {
            let bytes = self.hf_api.download_file(repo_id, filename, revision).await?;
            fs::write(&output_path, bytes).await?;
        }
        
        Ok(output_path)
    }
    
    /// Download multiple files matching patterns
    pub async fn download_files(
        &self,
        repo_id: &str,
        patterns: &[&str],
        revision: Option<&str>,
    ) -> Result<Vec<PathBuf>> {
        let files = self.hf_api.list_files(repo_id, revision).await?;
        
        let mut paths = Vec::new();
        
        for file in files {
            if Self::matches_patterns(&file.path, patterns) {
                let path = self.download_file(repo_id, &file.path, revision).await?;
                paths.push(path);
            }
        }
        
        Ok(paths)
    }
    
    /// Download an entire repository snapshot
    pub async fn snapshot_download(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        allow_patterns: Option<&[&str]>,
        ignore_patterns: Option<&[&str]>,
    ) -> Result<PathBuf> {
        let files = self.hf_api.list_files(repo_id, revision).await?;
        
        for file in files {
            let should_download = allow_patterns
                .map(|patterns| Self::matches_patterns(&file.path, patterns))
                .unwrap_or(true)
                && !ignore_patterns
                    .map(|patterns| Self::matches_patterns(&file.path, patterns))
                    .unwrap_or(false);
            
            if should_download {
                self.download_file(repo_id, &file.path, revision).await?;
            }
        }
        
        Ok(self.snapshot_dir(repo_id, revision.unwrap_or("main")))
    }
    
    /// Get repository metadata
    pub async fn repo_info(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<RepoInfo> {
        self.hf_api.repo_info(repo_id, revision).await
    }
    
    /// List files in a repository
    pub async fn list_files(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<Vec<FileInfo>> {
        self.hf_api.list_files(repo_id, revision).await
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        let total_size = self.chunk_cache.size()?;
        let chunk_count = self.chunk_cache.count()?;
        
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        
        Ok(CacheStats {
            total_size,
            chunk_count,
            hit_rate,
        })
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) -> Result<()> {
        self.chunk_cache.clear()
    }
    
    /// Prune cache to target size
    pub fn prune_cache(&mut self, target_size: u64) -> Result<u64> {
        self.chunk_cache.evict_lru(target_size)
    }
    
    fn load_token() -> Option<String> {
        if let Ok(token) = std::env::var("HF_TOKEN") {
            return Some(token);
        }
        
        if let Some(cache_dir) = dirs::cache_dir() {
            let token_path = cache_dir.join("huggingface").join("token");
            if let Ok(token) = std::fs::read_to_string(token_path) {
                return Some(token.trim().to_string());
            }
        }
        
        None
    }
    
    fn snapshot_dir(&self, repo_id: &str, revision: &str) -> PathBuf {
        self.config
            .cache_dir
            .join("snapshots")
            .join(repo_id)
            .join(revision)
    }
    
    fn matches_patterns(path: &str, patterns: &[&str]) -> bool {
        patterns.iter().any(|pattern| {
            if pattern.contains('*') {
                let pattern = pattern.replace("*.", "");
                path.ends_with(&pattern) || path.contains(&pattern)
            } else {
                path == *pattern || path.ends_with(pattern)
            }
        })
    }
    
    async fn download_lfs_file(
        &self,
        repo_id: &str,
        filename: &str,
        revision: Option<&str>,
        output_path: &std::path::Path,
    ) -> Result<()> {
        let bytes = self.hf_api.download_file(repo_id, filename, revision).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_create_client() {
        let dir = tempdir().unwrap();
        let client = HfXetClient::with_cache_dir(dir.path().to_path_buf());
        assert!(client.is_ok());
    }
    
    #[test]
    fn test_matches_patterns() {
        assert!(HfXetClient::matches_patterns("model.safetensors", &["*.safetensors"]));
        assert!(HfXetClient::matches_patterns("config.json", &["config.json"]));
        assert!(!HfXetClient::matches_patterns("other.txt", &["*.safetensors"]));
    }
}
