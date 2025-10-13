use inferox_candle::CandleBackend;
use inferox_core::{Backend, Model, Tensor, TensorBuilder};
use inferox_engine::{EngineConfig, InferoxEngine};
use libloading::{Library, Symbol};
use std::path::PathBuf;

type BoxedCandleModel = Box<
    dyn Model<
        Backend = CandleBackend,
        Input = <CandleBackend as Backend>::Tensor,
        Output = <CandleBackend as Backend>::Tensor,
    >,
>;
type ModelFactory = fn() -> BoxedCandleModel;

fn load_model_from_library(path: &str) -> Result<BoxedCandleModel, Box<dyn std::error::Error>> {
    unsafe {
        let lib = Library::new(path)?;
        let factory: Symbol<ModelFactory> = lib.get(b"create_model")?;
        let model = factory();
        std::mem::forget(lib);
        Ok(model)
    }
}

#[test]
#[ignore]
fn test_mlp_classifier_end_to_end() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    #[cfg(target_os = "macos")]
    let lib_path = workspace_root.join("target/release/libmlp_classifier.dylib");
    #[cfg(target_os = "linux")]
    let lib_path = workspace_root.join("target/release/libmlp_classifier.so");
    #[cfg(target_os = "windows")]
    let lib_path = workspace_root.join("target/release/mlp_classifier.dll");

    if !lib_path.exists() {
        panic!(
            "Model library not found at {:?}. Run: cargo build --release -p mlp-classifier",
            lib_path
        );
    }

    println!("1. Loading MLP classifier model from {:?}", lib_path);
    let model = load_model_from_library(lib_path.to_str().unwrap())
        .expect("Failed to load model via libloading");

    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}", model.metadata());

    println!("2. Creating backend and test input...");
    let backend = CandleBackend::cpu().expect("Failed to create CPU backend");

    let input_data: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(input_data.clone(), &[1, 10])
        .expect("Failed to create input tensor");

    println!("   Input shape: {:?}", input_tensor.shape());
    println!("   Input data: {:?}", input_data);

    println!("3. Running inference...");
    let output = model.forward(input_tensor).expect("Forward pass failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1, "Batch size should be 1");
    assert_eq!(output.shape()[1], 3, "Output size should be 3 (classes)");

    println!("\n✓ End-to-end test passed!");
    println!("  - Model loaded via libloading");
    println!("  - Inference produced correct output shape");
}

#[test]
#[ignore]
fn test_mlp_small_end_to_end() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    #[cfg(target_os = "macos")]
    let lib_path = workspace_root.join("target/release/libmlp_small.dylib");
    #[cfg(target_os = "linux")]
    let lib_path = workspace_root.join("target/release/libmlp_small.so");
    #[cfg(target_os = "windows")]
    let lib_path = workspace_root.join("target/release/mlp_small.dll");

    if !lib_path.exists() {
        panic!(
            "Model library not found at {:?}. Run: cargo build --release -p mlp-small",
            lib_path
        );
    }

    println!("1. Loading MLP small model from {:?}", lib_path);
    let model = load_model_from_library(lib_path.to_str().unwrap())
        .expect("Failed to load model via libloading");

    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}", model.metadata());

    println!("2. Creating backend and test input...");
    let backend = CandleBackend::cpu().expect("Failed to create CPU backend");

    let input_data: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(input_data.clone(), &[1, 5])
        .expect("Failed to create input tensor");

    println!("   Input shape: {:?}", input_tensor.shape());
    println!("   Input data: {:?}", input_data);

    println!("3. Running inference...");
    let output = model.forward(input_tensor).expect("Forward pass failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1, "Batch size should be 1");
    assert_eq!(output.shape()[1], 2, "Output size should be 2");

    println!("\n✓ End-to-end test passed!");
    println!("  - Model loaded via libloading");
    println!("  - Inference produced correct output shape");
}

#[test]
#[ignore]
fn test_mlp_with_inferox_engine() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    #[cfg(target_os = "macos")]
    let classifier_path = workspace_root.join("target/release/libmlp_classifier.dylib");
    #[cfg(target_os = "linux")]
    let classifier_path = workspace_root.join("target/release/libmlp_classifier.so");
    #[cfg(target_os = "windows")]
    let classifier_path = workspace_root.join("target/release/mlp_classifier.dll");

    #[cfg(target_os = "macos")]
    let small_path = workspace_root.join("target/release/libmlp_small.dylib");
    #[cfg(target_os = "linux")]
    let small_path = workspace_root.join("target/release/libmlp_small.so");
    #[cfg(target_os = "windows")]
    let small_path = workspace_root.join("target/release/mlp_small.dll");

    if !classifier_path.exists() || !small_path.exists() {
        panic!(
            "Model libraries not found. Run: cargo build --release -p mlp-classifier -p mlp-small"
        );
    }

    println!("1. Setting up Inferox Engine...");
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);

    println!("2. Loading multiple MLP models...");

    println!("   Loading classifier model...");
    let classifier_model = load_model_from_library(classifier_path.to_str().unwrap())
        .expect("Failed to load classifier model");
    let classifier_name = classifier_model.name().to_string();
    engine.register_model(&classifier_name, classifier_model, None);
    println!("   ✓ Registered: {}", classifier_name);

    println!("   Loading small model...");
    let small_model =
        load_model_from_library(small_path.to_str().unwrap()).expect("Failed to load small model");
    let small_name = small_model.name().to_string();
    engine.register_model(&small_name, small_model, None);
    println!("   ✓ Registered: {}", small_name);

    println!("\n3. Listing registered models...");
    let models = engine.list_models_with_metadata();
    println!("   Available models: {}", models.len());
    for (name, metadata) in &models {
        println!(
            "     - {} v{}: {}",
            name, metadata.version, metadata.description
        );
    }
    assert_eq!(models.len(), 2, "Should have 2 models registered");

    println!("\n4. Running inference on classifier model...");
    let backend = CandleBackend::cpu().expect("Failed to create CPU backend");

    let classifier_input: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
    let classifier_tensor = backend
        .tensor_builder()
        .build_from_vec(classifier_input.clone(), &[1, 10])
        .expect("Failed to create classifier input");

    println!("   Input shape: {:?}", classifier_tensor.shape());
    println!("   Input data: {:?}", classifier_input);

    let classifier_output = engine
        .infer_typed::<CandleBackend>(&classifier_name, classifier_tensor)
        .expect("Classifier inference failed");

    println!("   Output shape: {:?}", classifier_output.shape());
    assert_eq!(classifier_output.shape()[0], 1);
    assert_eq!(classifier_output.shape()[1], 3);

    println!("\n5. Running inference on small model...");
    let small_input: Vec<f32> = (0..5).map(|i| i as f32 * 0.2).collect();
    let small_tensor = backend
        .tensor_builder()
        .build_from_vec(small_input.clone(), &[1, 5])
        .expect("Failed to create small input");

    println!("   Input shape: {:?}", small_tensor.shape());
    println!("   Input data: {:?}", small_input);

    let small_output = engine
        .infer_typed::<CandleBackend>(&small_name, small_tensor)
        .expect("Small model inference failed");

    println!("   Output shape: {:?}", small_output.shape());
    assert_eq!(small_output.shape()[0], 1);
    assert_eq!(small_output.shape()[1], 2);

    println!("\n6. Testing batch inference...");
    let batch_inputs = vec![
        backend
            .tensor_builder()
            .build_from_vec(vec![0.1; 10], &[1, 10])
            .unwrap(),
        backend
            .tensor_builder()
            .build_from_vec(vec![0.5; 10], &[1, 10])
            .unwrap(),
        backend
            .tensor_builder()
            .build_from_vec(vec![0.9; 10], &[1, 10])
            .unwrap(),
    ];

    let batch_outputs = engine
        .infer_batch::<CandleBackend>(&classifier_name, batch_inputs)
        .expect("Batch inference failed");

    println!("   Batch size: {}", batch_outputs.len());
    assert_eq!(batch_outputs.len(), 3, "Should have 3 outputs");
    for (i, output) in batch_outputs.iter().enumerate() {
        println!("   Batch {} output shape: {:?}", i, output.shape());
        assert_eq!(output.shape()[1], 3);
    }

    println!("\n7. Testing model info retrieval...");
    let classifier_info = engine
        .model_info(&classifier_name)
        .expect("Failed to get classifier info");
    println!("   Classifier info: {:?}", classifier_info);
    assert_eq!(classifier_info.version, "1.0.0");
    assert!(classifier_info.description.contains("10"));

    let small_info = engine
        .model_info(&small_name)
        .expect("Failed to get small info");
    println!("   Small model info: {:?}", small_info);
    assert_eq!(small_info.version, "1.0.0");
    assert!(small_info.description.contains("5"));

    println!("\n✓ Engine integration test passed!");
    println!("  - Multiple models registered with engine");
    println!("  - Individual inference successful for both models");
    println!("  - Batch inference successful");
    println!("  - Model metadata retrieval successful");
}

#[test]
#[ignore]
fn test_mlp_engine_model_not_found() {
    println!("Testing error handling for non-existent model...");

    let config = EngineConfig::default();
    let engine = InferoxEngine::new(config);

    let backend = CandleBackend::cpu().expect("Failed to create CPU backend");
    let input = backend
        .tensor_builder()
        .build_from_vec(vec![1.0f32; 10], &[1, 10])
        .expect("Failed to create input");

    let result = engine.infer_typed::<CandleBackend>("nonexistent-model", input);

    assert!(
        result.is_err(),
        "Should return error for non-existent model"
    );
    println!("   ✓ Correctly returned error for non-existent model");

    println!("\n✓ Error handling test passed!");
}
