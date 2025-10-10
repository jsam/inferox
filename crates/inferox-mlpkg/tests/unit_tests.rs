//! Unit tests for inferox-mlpkg

use inferox_mlpkg::{Error, PackageManager, PackageMetadata};
use tempfile::tempdir;

#[test]
fn test_package_manager_creation() {
    let dir = tempdir().unwrap();
    let manager = PackageManager::new(dir.path().to_path_buf());
    assert!(manager.is_ok());
}

#[test]
fn test_error_display() {
    let err = Error::HfDownload("download failed".to_string());
    assert_eq!(err.to_string(), "HF download error: download failed");

    let err = Error::InvalidFormat("bad format".to_string());
    assert_eq!(err.to_string(), "Invalid package format: bad format");

    let err = Error::Generic("generic error".to_string());
    assert_eq!(err.to_string(), "generic error");
}

#[test]
fn test_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err: Error = io_err.into();
    assert!(matches!(err, Error::Io(_)));
}

#[test]
fn test_package_metadata_creation() {
    let metadata = PackageMetadata {
        version: "1.0".to_string(),
        name: "test-model".to_string(),
        repo_id: "org/model".to_string(),
        revision: "main".to_string(),
        files: vec!["model.safetensors".to_string(), "config.json".to_string()],
        created_at: chrono::Utc::now(),
    };

    assert_eq!(metadata.version, "1.0");
    assert_eq!(metadata.name, "test-model");
    assert_eq!(metadata.repo_id, "org/model");
    assert_eq!(metadata.files.len(), 2);
}

#[test]
fn test_package_metadata_serialization() {
    let metadata = PackageMetadata {
        version: "1.0".to_string(),
        name: "test-model".to_string(),
        repo_id: "org/model".to_string(),
        revision: "main".to_string(),
        files: vec!["model.safetensors".to_string()],
        created_at: chrono::Utc::now(),
    };

    let json = serde_json::to_string(&metadata).unwrap();
    assert!(json.contains("test-model"));
    assert!(json.contains("org/model"));

    let deserialized: PackageMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, metadata.name);
    assert_eq!(deserialized.repo_id, metadata.repo_id);
}
