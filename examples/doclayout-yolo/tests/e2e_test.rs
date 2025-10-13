use doclayout_yolo::DocLayoutYOLO;
use inferox_core::{Backend, DataType, Model, Tensor, TensorBuilder};
use inferox_engine::{EngineConfig, InferoxEngine};
use inferox_tch::TchBackend;
use std::path::PathBuf;
use tch::Device;

fn get_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt")
}

const MODEL_INPUT_SIZE: usize = 1280;

#[tokio::test]
#[ignore]
async fn test_doclayout_yolo_direct_inference() {
    let model_path = get_model_path();
    
    if !model_path.exists() {
        panic!(
            "Model not found at {:?}. Please download the model first.",
            model_path
        );
    }
    
    println!("=== DocLayout-YOLO Direct Inference Test ===\n");
    println!("1. Loading TorchScript model from: {:?}", model_path);
    
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load DocLayout-YOLO model");
    
    println!("   ✓ Model loaded successfully");
    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}\n", model.metadata());
    
    println!("2. Creating test input (dummy image tensor)...");
    let batch_size = 1;
    let channels = 3;
    let height = MODEL_INPUT_SIZE;
    let width = MODEL_INPUT_SIZE;
    
    let backend = TchBackend::cpu().expect("Failed to create backend");
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(
            vec![0.5f32; batch_size * channels * height * width],
            &[batch_size, channels, height, width],
        )
        .expect("Failed to create input tensor");
    
    println!("   Input shape: {:?}", input_tensor.shape());
    println!("   Input dtype: {}\n", input_tensor.dtype().name());
    
    println!("3. Running inference...");
    let output = model.forward(input_tensor)
        .expect("Forward pass failed");
    
    println!("   ✓ Inference completed successfully");
    println!("   Output shape: {:?}", output.shape());
    println!("   Output dtype: {}\n", output.dtype().name());
    
    println!("✅ Direct inference test passed!");
}

#[tokio::test]
#[ignore]
async fn test_doclayout_yolo_with_engine() {
    let model_path = get_model_path();
    
    if !model_path.exists() {
        panic!(
            "Model not found at {:?}. Please download the model first.",
            model_path
        );
    }
    
    println!("=== DocLayout-YOLO Engine Integration Test ===\n");
    println!("1. Setting up Inferox Engine...");
    
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);
    
    println!("   ✓ Engine created\n");
    
    println!("2. Loading DocLayout-YOLO model...");
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load DocLayout-YOLO model");
    
    let model_name = model.name().to_string();
    println!("   ✓ Model loaded: {}", model_name);
    println!("   Metadata: {:?}\n", model.metadata());
    
    println!("3. Registering model with engine...");
    engine.register_model(&model_name, Box::new(model), None);
    
    println!("   ✓ Model registered");
    println!("   Available models: {:?}\n", engine.list_models());
    
    println!("4. Creating test input...");
    let batch_size = 1;
    let channels = 3;
    let height = MODEL_INPUT_SIZE;
    let width = MODEL_INPUT_SIZE;
    
    let backend = TchBackend::cpu().expect("Failed to create backend");
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(
            vec![0.5f32; batch_size * channels * height * width],
            &[batch_size, channels, height, width],
        )
        .expect("Failed to create input tensor");
    
    println!("   Input shape: {:?}\n", input_tensor.shape());
    
    println!("5. Running inference through engine...");
    let output = engine
        .infer_typed::<TchBackend>(&model_name, input_tensor)
        .expect("Inference failed");
    
    println!("   ✓ Inference completed successfully");
    println!("   Output shape: {:?}", output.shape());
    println!("   Output dtype: {}\n", output.dtype().name());
    
    println!("✅ Engine integration test passed!");
    println!("  - Model loaded via TorchScript");
    println!("  - Model registered with engine");
    println!("  - Inference through engine API successful");
}

#[tokio::test]
#[ignore]
async fn test_doclayout_yolo_batch_inference() {
    let model_path = get_model_path();
    
    if !model_path.exists() {
        panic!(
            "Model not found at {:?}. Please download the model first.",
            model_path
        );
    }
    
    println!("=== DocLayout-YOLO Batch Inference Test ===\n");
    println!("1. Loading model...");
    
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load DocLayout-YOLO model");
    
    println!("   ✓ Model loaded\n");
    
    println!("2. Creating batch of test inputs (2 images)...");
    let backend = TchBackend::cpu().expect("Failed to create backend");
    
    let image1 = backend
        .tensor_builder()
        .build_from_vec(
            vec![0.3f32; 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE],
            &[1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        )
        .expect("Failed to create image1");
    
    let image2 = backend
        .tensor_builder()
        .build_from_vec(
            vec![0.7f32; 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE],
            &[1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        )
        .expect("Failed to create image2");
    
    println!("   ✓ Created 2 test images\n");
    
    println!("3. Running batch inference...");
    let outputs = model.detect_batch(vec![image1, image2])
        .expect("Batch inference failed");
    
    println!("   ✓ Batch inference completed");
    println!("   Number of outputs: {}", outputs.len());
    
    for (i, output) in outputs.iter().enumerate() {
        println!("   Output {}: shape = {:?}", i + 1, output.shape());
    }
    
    println!("\n✅ Batch inference test passed!");
}

#[tokio::test]
#[ignore]
async fn test_doclayout_yolo_model_info() {
    let model_path = get_model_path();
    
    if !model_path.exists() {
        panic!(
            "Model not found at {:?}. Please download the model first.",
            model_path
        );
    }
    
    println!("=== DocLayout-YOLO Model Info Test ===\n");
    
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load DocLayout-YOLO model");
    
    let metadata = model.metadata();
    
    println!("Model Information:");
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Description: {}", metadata.description);
    println!("  Author: {}", metadata.author);
    println!("  License: {}", metadata.license);
    println!("  Tags: {:?}", metadata.tags);
    
    assert_eq!(metadata.name, "doclayout-yolo");
    assert_eq!(metadata.author, "juliozhao");
    assert_eq!(metadata.license, "Apache-2.0");
    assert!(metadata.tags.contains(&"yolo".to_string()));
    assert!(metadata.tags.contains(&"document".to_string()));
    assert!(metadata.tags.contains(&"layout".to_string()));
    assert!(metadata.tags.contains(&"detection".to_string()));
    assert!(metadata.tags.contains(&"torchscript".to_string()));
    
    println!("\n✅ Model info test passed!");
}
