<div align="center">

# Inferox

**Safe, Fast, and Modular ML Inference Engine for Rust**

[![PR Checks](https://img.shields.io/github/actions/workflow/status/jsam/inferox/pr-checks.yml?style=flat-square&label=checks)](https://github.com/jsam/inferox/actions/workflows/pr-checks.yml)
[![Coverage](https://img.shields.io/codecov/c/github/jsam/inferox?style=flat-square)](https://codecov.io/gh/jsam/inferox)
[![Crates.io](https://img.shields.io/crates/v/inferox-core?style=flat-square)](https://crates.io/crates/inferox-core)
[![Documentation](https://img.shields.io/docsrs/inferox-core?style=flat-square)](https://docs.rs/inferox-core)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange?style=flat-square)](https://www.rust-lang.org)
[![Downloads](https://img.shields.io/crates/d/inferox-core?style=flat-square)](https://crates.io/crates/inferox-core)
[![Stars](https://img.shields.io/github/stars/jsam/inferox?style=flat-square)](https://github.com/jsam/inferox/stargazers)

</div>

---

## Overview

Inferox is a high-performance ML inference engine built in Rust, designed with a **two-pillar architecture** that separates model compilation from runtime execution. Compile your model architectures into shared libraries (`.so`/`.dylib`) and load them dynamically into the engine with complete type safety.

### Key Features

- 🔒 **Type-Safe Dynamic Loading**: Load models as trait objects - no manual FFI required
- 🚀 **Multiple Backend Support**: Candle backend implemented, extensible for ONNX, TensorFlow, etc.
- 🎯 **Zero-Copy Inference**: Efficient tensor operations without unnecessary allocations
- 🔧 **Hot Reloadable**: Swap model libraries without recompiling the engine
- 🦀 **Pure Rust**: Memory safety and RAII throughout, minimal `unsafe` confined to `libloading`

## Architecture

```
┌─────────────────────────────────────────────┐
│  Model Library (libmlp_classifier.dylib)    │
│  ┌──────────────────────────────────────┐   │
│  │ #[no_mangle]                         │   │
│  │ pub fn create_model()                │   │
│  │   -> Box<dyn Model>                  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Engine (loads via libloading)              │
│  ┌──────────────────────────────────────┐   │
│  │ let lib = Library::new(path)         │   │
│  │ let factory = lib.get("create_model")│   │
│  │ let model: Box<dyn Model> = factory()│   │
│  │ engine.register_boxed_model(model)   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  InferoxEngine manages all models           │
│  - Type-safe trait interface                │
│  - RAII memory management                   │
│  - No unsafe in user code                   │
└─────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

**Required:**
- Rust 1.70+ (`rustup install stable`)
- Python 3.8+ (for PyTorch/tch backend support)

**Optional (for tch backend):**
- PyTorch 2.4.0 (`pip3 install torch==2.4.0`)

### Installation

```bash
# Clone the repository
git clone https://github.com/jsam/inferox.git
cd inferox

# Install all dependencies (PyTorch, tools, hooks)
make install

# Build the project
make build

# Run tests
make test
```

### 1. Define Your Model Architecture

```rust
use inferox_core::{Model, ModelMetadata, InferoxError};
use inferox_candle::{CandleBackend, CandleTensor};
use candle_nn::{Linear, VarBuilder};

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    name: String,
}

impl Model for MLP {
    type Backend = CandleBackend;
    type Input = CandleTensor;
    type Output = CandleTensor;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output, InferoxError> {
        let x = self.fc1.forward(&input.inner())?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc3.forward(&x)?;
        Ok(CandleTensor::new(x))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata::new("mlp", "1.0.0")
            .with_description("Multi-Layer Perceptron")
    }
}
```

### 2. Compile to Shared Library

Create a library crate with `crate-type = ["cdylib"]`:

```rust
// models/classifier/src/lib.rs
use inferox_candle::{CandleBackend, CandleModelBuilder, CandleTensor};
use inferox_core::Model;
use candle_core::Device;

#[no_mangle]
pub fn create_model() -> Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>> {
    let builder = CandleModelBuilder::new(Device::Cpu);
    let model = MLP::new("classifier", 10, 8, 3, builder.var_builder())
        .expect("Failed to create classifier model");
    Box::new(model)
}
```

Build the model:

```bash
cargo build --release -p mlp-classifier
```

### 3. Load and Run in Engine

```rust
use inferox_engine::{InferoxEngine, EngineConfig};
use inferox_candle::CandleBackend;
use libloading::{Library, Symbol};

type ModelFactory = fn() -> Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CandleBackend::cpu();
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(backend.clone(), config);
    
    // Load model from shared library
    unsafe {
        let lib = Library::new("target/release/libmlp_classifier.dylib")?;
        let factory: Symbol<ModelFactory> = lib.get(b"create_model")?;
        let model = factory();
        engine.register_boxed_model(model);
        std::mem::forget(lib);
    }
    
    // Run inference
    let input = backend.tensor_builder().build_from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        &[1, 10]
    )?;
    
    let output = engine.infer("classifier", input)?;
    println!("Output: {:?}", output.to_vec2::<f32>()?);
    
    Ok(())
}
```

## Project Structure

```
inferox/
├── crates/
│   ├── inferox-core/        # Core traits and types
│   ├── inferox-candle/      # Candle backend implementation
│   └── inferox-engine/      # Inference engine runtime
├── examples/
│   └── mlp/                 # MLP example with dynamic loading
│       ├── src/
│       │   ├── lib.rs       # MLP architecture
│       │   └── main.rs      # Engine runtime
│       └── models/
│           ├── classifier/  # Compiled to .dylib/.so
│           └── small/       # Compiled to .dylib/.so
├── Makefile                 # Development commands
└── .github/workflows/       # CI/CD pipelines
```

## Core Components

### `inferox-core`

Core trait definitions for backends, tensors, and models:

- `Backend` - Hardware abstraction (CPU, CUDA, Metal, etc.)
- `Tensor` - N-dimensional array operations
- `Model` - Model trait with forward pass, metadata, and state management
- `DataType` - Numeric type system with safe conversions

### `inferox-candle`

Candle backend implementation using Hugging Face's [Candle](https://github.com/huggingface/candle):

- `CandleBackend` - Backend for Candle tensors
- `CandleTensor` - Tensor wrapper with type-safe operations
- `CandleModelBuilder` - Model initialization with weight loading
- `CandleVarMap` - Weight management and serialization

### `inferox-engine`

High-level inference engine with model management:

- `InferoxEngine` - Multi-model inference orchestration
- `InferenceSession` - Stateful inference with context
- `EngineConfig` - Runtime configuration (batch size, device, etc.)
- Dynamic model loading via trait objects

## Examples

### MLP Example

A complete example demonstrating the two-pillar architecture:

```bash
# Build model libraries
make models

# Run the engine with multiple models
cargo run --bin mlp --release -- \
  target/release/libmlp_classifier.dylib \
  target/release/libmlp_small.dylib
```

Output:

```
Inferox MLP Engine
==================

✓ Created CPU backend

Loading model from: target/release/libmlp_classifier.dylib
✓ Registered 'classifier' - Multi-Layer Perceptron (10 → 8 → 8 → 3)

Loading model from: target/release/libmlp_small.dylib
✓ Registered 'small' - Multi-Layer Perceptron (5 → 4 → 4 → 2)

2 models loaded

Available models:
  - classifier v1.0.0: Multi-Layer Perceptron (10 → 8 → 8 → 3)
  - small v1.0.0: Multi-Layer Perceptron (5 → 4 → 4 → 2)

Running test inference on all models:
  classifier -> output shape: [1, 3]
  small -> output shape: [1, 2]

✓ All models working!
```

See [examples/mlp/README.md](examples/mlp/README.md) for detailed documentation.

## Development

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo
- Python 3.8+ (for tch backend)
- PyTorch 2.4.0 (for tch backend): `pip3 install torch==2.4.0`

### Setup

```bash
# Install all development dependencies
make install

# This will:
# - Install PyTorch 2.4.0
# - Install Python packages (huggingface_hub, safetensors)
# - Install Rust tools (cargo-tarpaulin)
# - Set up git hooks
```

### Environment Variables (for tch backend)

If you're working with the tch backend, you need to set these environment variables:

```bash
# Tell tch-rs to use PyTorch installation
export LIBTORCH_USE_PYTORCH=1

# macOS: Set library path for runtime
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$DYLD_LIBRARY_PATH"

# Linux: Set library path for runtime  
export LD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

**Tip:** Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

### Building

```bash
# Build all crates
make build

# Build in release mode
make build-release

# Build model libraries
make models

# Build examples
make examples
```

### Testing

```bash
# Run tests + quick lint (recommended)
make test

# Run tests only
make test-quick

# Run specific crate tests
make test-core
make test-candle
make test-engine
```

### Linting and Formatting

```bash
# Format code
make format

# Run clippy linter
make lint

# Quick pre-commit checks
make pre-commit
```

### Documentation

```bash
# Generate and open docs
make doc

# Generate docs including private items
make doc-private
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Format Check**: Ensures code follows `rustfmt` standards
- **Clippy Lint**: Catches common mistakes and anti-patterns
- **Test Suite**: Runs on Ubuntu and macOS with stable Rust
- **Model Libraries**: Verifies model binaries build correctly
- **Documentation**: Ensures docs build without warnings
- **Examples**: Validates all examples compile and run

See [.github/workflows/pr-checks.yml](.github/workflows/pr-checks.yml) for the complete pipeline.

## Roadmap

- [x] Core trait system
- [x] Candle backend
- [x] Inference engine
- [x] Dynamic model loading
- [x] MLP example
- [ ] ResNet18 example
- [ ] ONNX backend
- [ ] Batch inference optimization
- [ ] Model quantization support
- [ ] GPU acceleration (CUDA, Metal)
- [ ] Production deployment guide

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run `make pre-commit` before committing
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- --- -->
<!-- 
<div align="center">
Built with ❤️ by the InputLayer team
</div> -->