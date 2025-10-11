use inferox_core::{Backend, Tensor, TensorBuilder};
use inferox_mlpkg::{BackendType, PackageManager};
use inferox_tch::TchBackend;
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn test_bert_tch_package_end_to_end() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let package_path = workspace_root.join("target/mlpkg/bert-tch");

    if !package_path.exists() {
        panic!(
            "Package not found at {:?}. Run: cargo build --release -p bert-tch",
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
    let model = manager
        .load_model(&package, BackendType::Tch)
        .expect("Failed to load model");

    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}", model.metadata());

    println!("3. Creating backend and test input...");
    let backend = TchBackend::cpu().expect("Failed to create backend");

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
async fn test_bert_tch_with_inferox_engine() {
    use inferox_engine::{EngineConfig, InferoxEngine};

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let package_path = workspace_root.join("target/mlpkg/bert-tch");

    if !package_path.exists() {
        panic!(
            "Package not found at {:?}. Run: cargo build --release -p bert-tch",
            package_path
        );
    }

    println!("1. Setting up Inferox Engine...");
    let backend = TchBackend::cpu().expect("Failed to create backend");
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(backend.clone(), config);

    let cache_dir = std::env::temp_dir().join("inferox-test-cache");
    println!("Cache directory: {:?}", cache_dir);
    println!("2. Loading BERT model package...");

    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let package = manager
        .load_package(&package_path)
        .expect("Failed to load package");

    println!("   Model: {}", package.metadata.name);
    println!("   Repo: {}", package.metadata.repo_id);
    println!(
        "Supported backends: {:?}",
        package.info.supported_backends
    );

    let model = manager
        .load_model(&package, BackendType::Tch)
        .expect("Failed to load model");

    let model_name = model.name().to_string();

    println!("3. Registering model with engine...");
    engine.register_boxed_model(model);

    println!("   Input shape: {:?}", [1, 6]);
    let input_ids: Vec<i64> = vec![101, 2023, 2003, 1037, 3231, 102];
    println!("   Input IDs: {:?}", input_ids);

    println!("4. Running inference through engine...");
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(input_ids.clone(), &[1, input_ids.len()])
        .expect("Failed to create input tensor");

    println!(
        "   Available models: {:?}",
        engine
            .list_models()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    );

    let output = engine
        .infer(&model_name, input_tensor)
        .expect("Inference failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], input_ids.len());
    assert_eq!(output.shape()[2], 768);

    println!("\n✓ Engine integration test passed!");
    println!("  - Model registered with engine");
    println!("  - Inference through engine API successful");
}
