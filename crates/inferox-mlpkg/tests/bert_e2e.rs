use inferox_mlpkg::{BackendType, PackageManager};
use std::process::Command;
use tempfile::tempdir;

#[tokio::test]
#[ignore]
async fn test_bert_end_to_end() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let cache_dir = temp_dir.path().to_path_buf();

    let manager = PackageManager::new(cache_dir.clone()).expect("Failed to create PackageManager");

    println!("1. Downloading and packaging bert-base-uncased...");
    let package = manager
        .download_and_package("bert-base-uncased", None, &[BackendType::Candle])
        .await
        .expect("Failed to download and package");

    println!("2. Package created at: {:?}", package.path());
    println!("   Model type: {}", package.info().model_type);
    println!("   Architecture: {:?}", package.info().architecture_family);

    println!("3. Compiling bert-candle library...");
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current dir")
        .join("../..")
        .canonicalize()
        .expect("Failed to canonicalize workspace root");

    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg("bert-candle")
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to compile bert-candle");

    if !output.status.success() {
        panic!(
            "bert-candle compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("4. Installing compiled library into package...");
    #[cfg(target_os = "macos")]
    let lib_path = workspace_root.join("target/release/libbert_candle.dylib");
    #[cfg(target_os = "windows")]
    let lib_path = workspace_root.join("target/release/bert_candle.dll");
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let lib_path = workspace_root.join("target/release/libbert_candle.so");

    manager
        .install_model_library(&package, &lib_path)
        .expect("Failed to install library");

    println!("5. Loading model using libloading...");
    let model = manager
        .load_model(&package, BackendType::Candle)
        .expect("Failed to load model");

    println!("   Model name: {}", model.name());
    println!("   Model metadata: {:?}", model.metadata());

    println!("6. Creating test input tensor (token IDs)...");
    use inferox_candle::CandleTensor;
    use inferox_core::Tensor;

    let input_ids = vec![101u32, 2023, 2003, 1037, 3231];
    let input_tensor = candle_core::Tensor::from_vec(input_ids, &[1, 5], &candle_core::Device::Cpu)
        .expect("Failed to create tensor");
    let input_tensor = input_tensor
        .to_dtype(candle_core::DType::U32)
        .expect("Failed to convert to U32");
    let candle_input = CandleTensor::from(input_tensor);

    println!("7. Running inference...");
    let output = model.forward(candle_input).expect("Forward pass failed");

    println!("   Output shape: {:?}", output.shape());

    assert_eq!(output.shape()[0], 1, "Batch size should be 1");
    assert_eq!(output.shape()[1], 5, "Sequence length should be 5");
    assert_eq!(output.shape()[2], 768, "Hidden size should be 768");

    println!("\n✓ BERT end-to-end test passed!");
}
