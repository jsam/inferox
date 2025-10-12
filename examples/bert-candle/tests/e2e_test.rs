use inferox_candle::CandleBackend;
use inferox_core::{Backend, Tensor, TensorBuilder};
use inferox_mlpkg::PackageManager;
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn test_bert_package_end_to_end() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let package_path = workspace_root.join("target/mlpkg/bert-candle");

    if !package_path.exists() {
        panic!(
            "Package not found at {:?}. Run: cargo build --release -p bert-candle",
            package_path
        );
    }

    let cache_dir = std::env::temp_dir().join("inferox-test-cache");
    println!("Cache directory: {:?}", cache_dir);
    println!("1. Loading package from {:?}", package_path);

    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let package = manager
        .load_package(&package_path)
        .expect("Failed to load package");

    println!("   Model: {}", package.metadata.name);
    println!("   Repo: {}", package.metadata.repo_id);
    println!("   Architecture: {:?}", package.info.architecture_family);
    println!(
        "   Supported backends: {:?}",
        package.info.supported_backends
    );

    println!("2. Loading model with libloading...");
    let (loaded_model, device) = manager.load_model(&package).expect("Failed to load model");

    let model = loaded_model.as_candle().expect("Expected Candle model");
    let candle_device = device.as_candle().expect("Expected Candle device");

    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}", model.metadata());
    println!("   Device: {:?}", candle_device);

    println!("3. Creating backend and test input...");
    let backend = CandleBackend::with_device(candle_device.clone());

    let input_ids: Vec<i64> = vec![101, 2023, 2003, 1037, 3231, 102];
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(input_ids.clone(), &[1, input_ids.len()])
        .expect("Failed to create input tensor");

    println!("   Input shape: {:?}", input_tensor.shape());
    println!("   Input IDs: {:?}", input_ids);

    println!("4. Running inference...");
    let output = model.forward(input_tensor).expect("Forward pass failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1, "Batch size should be 1");
    assert_eq!(
        output.shape()[1],
        input_ids.len(),
        "Sequence length should match input"
    );
    assert_eq!(output.shape()[2], 768, "Hidden size should be 768");

    println!("\n✓ End-to-end test passed!");
    println!("  - Package loaded successfully");
    println!("  - Model loaded via libloading");
    println!("  - Inference produced correct output shape");
}

#[tokio::test]
#[ignore]
async fn test_bert_with_inferox_engine() {
    use inferox_engine::{EngineConfig, InferoxEngine};

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let package_path = workspace_root.join("target/mlpkg/bert-candle");

    if !package_path.exists() {
        panic!(
            "Package not found at {:?}. Run: cargo build --release -p bert-candle",
            package_path
        );
    }

    println!("1. Setting up Inferox Engine...");
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);

    let cache_dir = std::env::temp_dir().join("inferox-test-cache");
    println!("Cache directory: {:?}", cache_dir);
    println!("2. Loading BERT model package...");

    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let package = manager
        .load_package(&package_path)
        .expect("Failed to load package");

    let (model, _device) = manager.load_model(&package).expect("Failed to load model");

    let model = model.as_candle().expect("Expected Candle model");
    let model_name = model.name().to_string();

    println!("3. Registering model with engine...");
    engine.register_model(&model_name, model, None);

    println!("   Available models: {:?}", engine.list_models());

    println!("4. Running inference through engine...");
    let backend = CandleBackend::cpu().expect("Failed to create backend");
    let input_ids: Vec<i64> = vec![101, 7592, 1010, 2088, 999, 102];
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(input_ids.clone(), &[1, input_ids.len()])
        .expect("Failed to create input tensor");

    let output = engine
        .infer_typed::<CandleBackend>(&model_name, input_tensor)
        .expect("Inference failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], input_ids.len());
    assert_eq!(output.shape()[2], 768);

    println!("\n✓ Engine integration test passed!");
    println!("  - Model registered with engine");
    println!("  - Inference through engine API successful");
}
