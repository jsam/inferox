//! Integration tests for hf-xet-rs
//!
//! These tests require network access and a valid HF token.
//! Run with: cargo test --test integration_tests -- --ignored

use hf_xet_rs::{HfXetClient, Result};
use std::path::PathBuf;
use tempfile::tempdir;

fn load_test_token() -> Option<String> {
    // Try to load from .hftoken file in project root
    let token_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(".hftoken"));

    if let Some(path) = token_path {
        if let Ok(token) = std::fs::read_to_string(path) {
            return Some(token.trim().to_string());
        }
    }

    // Fallback to environment variable
    std::env::var("HF_TOKEN").ok()
}

#[tokio::test]
#[ignore] // Run with --ignored flag
async fn test_download_small_model() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    // Download a tiny test model
    let path = client
        .download_file("hf-internal-testing/tiny-random-bert", "config.json", None)
        .await?;

    assert!(path.exists());
    assert!(path.is_file());

    // Verify content
    let content = tokio::fs::read_to_string(&path).await?;
    assert!(content.contains("\"model_type\""));

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_repo_info() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    let info = client
        .repo_info("hf-internal-testing/tiny-random-bert", None)
        .await?;

    assert_eq!(info.id, "hf-internal-testing/tiny-random-bert");
    assert!(!info.siblings.is_empty());

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_list_files() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    let files = client
        .list_files("hf-internal-testing/tiny-random-bert", None)
        .await?;

    assert!(!files.is_empty());
    assert!(files.iter().any(|f| f.path.contains("config.json")));

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_download_multiple_files() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    let paths = client
        .download_files("hf-internal-testing/tiny-random-bert", &["*.json"], None)
        .await?;

    assert!(!paths.is_empty());
    assert!(paths.iter().all(|p| p.extension().unwrap() == "json"));

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_snapshot_download() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    let snapshot_dir = client
        .snapshot_download(
            "hf-internal-testing/tiny-random-bert",
            None,
            Some(&["*.json", "*.txt"]),
            None,
        )
        .await?;

    assert!(snapshot_dir.exists());
    assert!(snapshot_dir.is_dir());

    // Verify some files exist
    assert!(snapshot_dir.join("config.json").exists());

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_cache_reuse() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    // First download
    let path1 = client
        .download_file("hf-internal-testing/tiny-random-bert", "config.json", None)
        .await?;

    let metadata1 = tokio::fs::metadata(&path1).await?;
    let mtime1 = metadata1.modified()?;

    // Second download (should use cache)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    let path2 = client
        .download_file("hf-internal-testing/tiny-random-bert", "config.json", None)
        .await?;

    let metadata2 = tokio::fs::metadata(&path2).await?;
    let mtime2 = metadata2.modified()?;

    // Same path and modification time = cache hit
    assert_eq!(path1, path2);
    assert_eq!(mtime1, mtime2);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_cache_stats() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    // Download something
    client
        .download_file("hf-internal-testing/tiny-random-bert", "config.json", None)
        .await?;

    let _stats = client.cache_stats()?;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_clear_cache() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    // Download something
    client
        .download_file("hf-internal-testing/tiny-random-bert", "config.json", None)
        .await?;

    // Clear cache
    client.clear_cache()?;

    let stats = client.cache_stats()?;
    assert_eq!(stats.chunk_count, 0);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_download_with_revision() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())?.with_token(token);

    let path = client
        .download_file(
            "hf-internal-testing/tiny-random-bert",
            "config.json",
            Some("main"),
        )
        .await?;

    assert!(path.exists());
    assert!(path.to_string_lossy().contains("main"));

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_file_not_found() {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())
        .unwrap()
        .with_token(token);

    let result = client
        .download_file(
            "hf-internal-testing/tiny-random-bert",
            "nonexistent.json",
            None,
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
#[ignore]
async fn test_repo_not_found() {
    let token = load_test_token().expect("HF token required for integration tests");

    let dir = tempdir().unwrap();
    let client = HfXetClient::with_cache_dir(dir.path().to_path_buf())
        .unwrap()
        .with_token(token);

    let result = client
        .repo_info("nonexistent/model-that-does-not-exist", None)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
#[ignore]
async fn test_download_depth_anything_small() -> Result<()> {
    let token = load_test_token().expect("HF token required for integration tests");

    let cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let client = HfXetClient::with_cache_dir(cache_dir.clone())?.with_token(token);

    let repo_id = "LiheYoung/depth-anything-small-hf";

    let snapshot_dir = client
        .snapshot_download(repo_id, Some("main"), None, None)
        .await?;

    assert!(snapshot_dir.exists());
    assert!(snapshot_dir.is_dir());

    assert!(snapshot_dir.join("config.json").exists());
    assert!(snapshot_dir.join("preprocessor_config.json").exists());

    let files = client.list_files(repo_id, Some("main")).await?;
    assert!(!files.is_empty());

    let config_files: Vec<_> = files.iter().filter(|f| f.path.ends_with(".json")).collect();
    assert!(!config_files.is_empty());

    for file in config_files {
        let file_path = snapshot_dir.join(&file.path);
        assert!(
            file_path.exists(),
            "Expected file {} to exist at {:?}",
            file.path,
            file_path
        );
    }

    let _stats = client.cache_stats()?;

    Ok(())
}
