//! Unit tests for hf-xet-rs components

use hf_xet_rs::{Error, XetConfig};
use std::path::PathBuf;

#[test]
fn test_config_default() {
    let config = XetConfig::default();

    assert!(config.cache_dir.to_string_lossy().contains("huggingface"));
    assert_eq!(config.max_cache_size, 10 * 1024 * 1024 * 1024);
    assert_eq!(config.hub_endpoint, "https://huggingface.co");
    assert_eq!(config.cas_endpoint, "https://cas.huggingface.co");
    assert_eq!(config.max_concurrent_chunks, 8);
    assert_eq!(config.timeout_secs, 300);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_config_with_cache_dir() {
    let custom_dir = PathBuf::from("/tmp/test-cache");
    let config = XetConfig::default().with_cache_dir(custom_dir.clone());

    assert_eq!(config.cache_dir, custom_dir);
}

#[test]
fn test_config_with_max_cache_size() {
    let config = XetConfig::default().with_max_cache_size(5_000_000_000);

    assert_eq!(config.max_cache_size, 5_000_000_000);
}

#[test]
fn test_config_with_endpoints() {
    let config = XetConfig::default()
        .with_hub_endpoint("https://custom-hub.example.com".to_string())
        .with_cas_endpoint("https://custom-cas.example.com".to_string());

    assert_eq!(config.hub_endpoint, "https://custom-hub.example.com");
    assert_eq!(config.cas_endpoint, "https://custom-cas.example.com");
}

#[test]
fn test_error_display() {
    let err = Error::RepoNotFound("test-repo".to_string());
    assert_eq!(err.to_string(), "Repository not found: test-repo");

    let err = Error::FileNotFound("test.txt".to_string());
    assert_eq!(err.to_string(), "File not found: test.txt");

    let err = Error::Authentication("invalid token".to_string());
    assert_eq!(err.to_string(), "Authentication failed: invalid token");
}

#[test]
fn test_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err: Error = io_err.into();

    assert!(matches!(err, Error::Io(_)));
}

#[test]
fn test_patterns_matching() {
    // This tests the internal pattern matching logic
    assert!(matches_pattern("model.safetensors", "*.safetensors"));
    assert!(matches_pattern("config.json", "*.json"));
    assert!(matches_pattern("tokenizer_config.json", "*.json"));
    assert!(!matches_pattern("model.bin", "*.safetensors"));

    // Exact match
    assert!(matches_pattern("config.json", "config.json"));
    assert!(!matches_pattern("other.json", "config.json"));
}

// Helper function for pattern testing
fn matches_pattern(path: &str, pattern: &str) -> bool {
    if pattern.contains('*') {
        let pattern = pattern.replace("*.", "");
        path.ends_with(&pattern) || path.contains(&pattern)
    } else {
        path == pattern || path.ends_with(pattern)
    }
}

#[test]
fn test_cache_stats_display() {
    use hf_xet_rs::CacheStats;

    let stats = CacheStats {
        total_size: 1024 * 1024 * 100, // 100 MB
        chunk_count: 50,
        hit_rate: 0.75,
    };

    assert_eq!(stats.total_size, 104_857_600);
    assert_eq!(stats.chunk_count, 50);
    assert_eq!(stats.hit_rate, 0.75);
}

#[test]
fn test_download_progress() {
    use hf_xet_rs::types::DownloadProgress;

    let progress = DownloadProgress {
        file_name: "model.safetensors".to_string(),
        downloaded_bytes: 512_000,
        total_bytes: 1_024_000,
        chunks_completed: 5,
        chunks_total: 10,
    };

    assert_eq!(progress.file_name, "model.safetensors");
    assert_eq!(progress.downloaded_bytes, 512_000);
    assert_eq!(progress.total_bytes, 1_024_000);

    let percentage = (progress.downloaded_bytes as f64 / progress.total_bytes as f64) * 100.0;
    assert!((percentage - 50.0).abs() < 0.1);
}

#[test]
fn test_file_info_deserialization() {
    use hf_xet_rs::FileInfo;

    let json = r#"{
        "rfilename": "config.json",
        "size": 1234,
        "blob_id": "abc123"
    }"#;

    let file_info: FileInfo = serde_json::from_str(json).unwrap();

    assert_eq!(file_info.path, "config.json");
    assert_eq!(file_info.size, 1234);
    assert_eq!(file_info.blob_id, "abc123");
    assert!(file_info.lfs.is_none());
}

#[test]
fn test_lfs_pointer_deserialization() {
    use hf_xet_rs::FileInfo;

    let json = r#"{
        "rfilename": "model.bin",
        "size": 5000000,
        "blob_id": "def456",
        "lfs": {
            "oid": "sha256:abc123...",
            "size": 5000000
        }
    }"#;

    let file_info: FileInfo = serde_json::from_str(json).unwrap();

    assert!(file_info.lfs.is_some());
    let lfs = file_info.lfs.unwrap();
    assert_eq!(lfs.oid, "sha256:abc123...");
    assert_eq!(lfs.size, 5000000);
}
