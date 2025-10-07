use inferox_candle::{CandleBackend, CandleTensor};
use inferox_core::{Backend, Model, Tensor, TensorBuilder};
use inferox_engine::{EngineConfig, InferoxEngine};
use libloading::{Library, Symbol};
use std::env;

type BoxedCandleModel =
    Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>>;
type ModelFactory = fn() -> BoxedCandleModel;

fn load_model_from_binary(path: &str) -> Result<BoxedCandleModel, Box<dyn std::error::Error>> {
    unsafe {
        let lib = Library::new(path)?;
        let factory: Symbol<ModelFactory> = lib.get(b"create_model")?;
        let model = factory();
        std::mem::forget(lib);
        Ok(model)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model-library-1.dylib> [model-library-2.dylib] ...",
            args[0]
        );
        eprintln!("\nExample (from workspace root):");
        eprintln!(
            "  {} target/release/libmlp_classifier.dylib target/release/libmlp_small.dylib",
            args[0]
        );
        eprintln!("\nAvailable models after building:");
        eprintln!("  cargo build --release -p mlp-classifier -p mlp-small");
        eprintln!("  - target/release/libmlp_classifier.dylib");
        eprintln!("  - target/release/libmlp_small.dylib");
        std::process::exit(1);
    }

    println!("Inferox MLP Engine");
    println!("==================\n");

    let backend = CandleBackend::cpu()?;
    println!("✓ Created CPU backend");

    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(backend.clone(), config);

    for model_path in &args[1..] {
        println!("\nLoading model from: {}", model_path);
        let model = load_model_from_binary(model_path)?;
        let model_name = model.name().to_string();
        let metadata = model.metadata();

        engine.register_boxed_model(model);
        println!("✓ Registered '{}' - {}", model_name, metadata.description);
    }

    println!("\n{} models loaded\n", args.len() - 1);
    println!("Available models:");
    for (name, metadata) in engine.list_models() {
        println!(
            "  - {} v{}: {}",
            name, metadata.version, metadata.description
        );
    }

    println!("\nRunning test inference on all models:");
    for (name, _) in engine.list_models() {
        let input_size = if name == "classifier" { 10 } else { 5 };

        let input_data: Vec<f32> = (0..input_size).map(|i| i as f32 * 0.1).collect();
        let input = backend
            .tensor_builder()
            .build_from_vec(input_data, &[1, input_size])?;

        let output = engine.infer(name, input)?;
        println!("  {} -> output shape: {:?}", name, output.shape());
    }

    println!("\n✓ All models working!");

    Ok(())
}
