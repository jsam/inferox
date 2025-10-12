//! Unit tests for inferox-mlpkg

use inferox_mlpkg::{
    ArchitectureFamily, BackendType, Error, ModelInfo, PackageManager, PackageMetadata,
};
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

#[test]
fn test_backend_type_as_str() {
    assert_eq!(BackendType::Candle.as_str(), "candle");
    assert_eq!(BackendType::Torch.as_str(), "torch");
    assert_eq!(BackendType::TFLite.as_str(), "tflite");
}

#[test]
fn test_backend_type_serialization() {
    let backend = BackendType::Candle;
    let json = serde_json::to_string(&backend).unwrap();
    assert_eq!(json, "\"Candle\"");

    let deserialized: BackendType = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, BackendType::Candle);
}

#[test]
fn test_architecture_family_serialization() {
    let arch = ArchitectureFamily::EncoderOnly;
    let json = serde_json::to_string(&arch).unwrap();
    assert_eq!(json, "\"EncoderOnly\"");

    let deserialized: ArchitectureFamily = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, ArchitectureFamily::EncoderOnly);
}

#[test]
fn test_model_info_creation() {
    let info = ModelInfo {
        model_type: "bert".to_string(),
        repo_id: "bert-base-uncased".to_string(),
        architecture_family: ArchitectureFamily::EncoderOnly,
        supported_backends: vec![BackendType::Candle],
        device: Some("cpu".to_string()),
        hidden_size: Some(768),
        num_layers: Some(12),
        vocab_size: Some(30522),
    };

    assert_eq!(info.model_type, "bert");
    assert_eq!(info.architecture_family, ArchitectureFamily::EncoderOnly);
    assert_eq!(info.supported_backends.len(), 1);
    assert_eq!(info.hidden_size, Some(768));
}

#[test]
fn test_model_info_serialization() {
    let info = ModelInfo {
        model_type: "gpt2".to_string(),
        repo_id: "gpt2".to_string(),
        architecture_family: ArchitectureFamily::DecoderOnly,
        supported_backends: vec![BackendType::Candle, BackendType::Torch],
        device: Some("cpu".to_string()),
        hidden_size: Some(768),
        num_layers: Some(12),
        vocab_size: Some(50257),
    };

    let json = serde_json::to_string(&info).unwrap();
    let deserialized: ModelInfo = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.model_type, info.model_type);
    assert_eq!(
        deserialized.architecture_family,
        ArchitectureFamily::DecoderOnly
    );
    assert_eq!(deserialized.supported_backends.len(), 2);
}

#[test]
fn test_config_parser_bert() {
    use std::io::Write;
    let temp_dir = tempdir().unwrap();
    let config_path = temp_dir.path().join("config.json");

    let config_content = r#"{
        "model_type": "bert",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "vocab_size": 30522
    }"#;

    let mut file = std::fs::File::create(&config_path).unwrap();
    file.write_all(config_content.as_bytes()).unwrap();

    // Can't test private functions directly, but we can test through public API later
    // For now, just verify the JSON is valid
    let parsed: serde_json::Value = serde_json::from_str(config_content).unwrap();
    assert_eq!(parsed["model_type"], "bert");
    assert_eq!(parsed["hidden_size"], 768);
}

#[test]
fn test_architecture_family_values() {
    // Test all architecture family variants exist
    let _encoder = ArchitectureFamily::EncoderOnly;
    let _decoder = ArchitectureFamily::DecoderOnly;
    let _enc_dec = ArchitectureFamily::EncoderDecoder;
    let _vision = ArchitectureFamily::Vision;
    let _multimodal = ArchitectureFamily::Multimodal;
}
