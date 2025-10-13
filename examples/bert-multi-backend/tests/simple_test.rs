use inferox_engine::{EngineConfig, InferoxEngine};
use inferox_mlpkg::PackageManager;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[tokio::test]
#[ignore]
async fn test_engine_simple_api() {
    println!("=== Test: Engine Simple API ===\n");

    let workspace_root = workspace_root();
    let candle_package = workspace_root.join("target/mlpkg/bert-candle");
    let tch_package = workspace_root.join("target/mlpkg/bert-tch");

    assert!(candle_package.exists(), "Candle package not found");
    assert!(tch_package.exists(), "Tch package not found");

    let cache_dir = std::env::temp_dir().join("inferox-multi-backend-test");
    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    println!("1. Loading models...");
    let candle_pkg = manager.load_package(&candle_package).expect("Failed to load candle package");
    let tch_pkg = manager.load_package(&tch_package).expect("Failed to load tch package");

    let (candle_model, _) = manager.load_model(&candle_pkg).expect("Failed to load candle model");
    let candle_model = candle_model.as_candle().expect("Expected Candle model");

    let (tch_model, _) = manager.load_model(&tch_pkg).expect("Failed to load tch model");
    let tch_model = tch_model.as_tch().expect("Expected Tch model");

    println!("2. Creating engine...");
    let mut engine = InferoxEngine::new(EngineConfig::default());

    println!("3. Registering models with route names...");
    engine.register_model("api/v1/bert.candle", candle_model, None);
    engine.register_model("api/v1/bert.pytorch", tch_model, None);

    let models = engine.list_models();
    println!("   ✓ Registered models: {:?}", models);
    assert_eq!(models.len(), 2);
    assert!(models.contains(&"api/v1/bert.candle".to_string()));
    assert!(models.contains(&"api/v1/bert.pytorch".to_string()));

    println!("4. Testing inference via 'api/v1/bert.candle'...");
    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];
    let output = engine.infer("api/v1/bert.candle", input_ids.clone())
        .expect("Inference failed");

    println!("   ✓ Output shape: {:?}", output.shape);
    println!("   ✓ Output data size: {}", output.data.len());
    assert_eq!(output.shape, vec![1, 6, 768]);

    println!("5. Testing inference via 'api/v1/bert.pytorch'...");
    let output = engine.infer("api/v1/bert.pytorch", input_ids)
        .expect("Inference failed");

    println!("   ✓ Output shape: {:?}", output.shape);
    assert_eq!(output.shape, vec![1, 6, 768]);

    println!("\n✅ Test passed: Single-step registration and inference!\n");
}
