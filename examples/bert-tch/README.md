# BERT-Tch Example

BERT base model implementation using the `inferox-tch` backend (PyTorch tch-rs bindings).

## Prerequisites

This example requires **LibTorch** (PyTorch C++ library) to be installed.

### Installing LibTorch

#### Option 1: Download from PyTorch (Recommended)

1. Download LibTorch from https://pytorch.org/get-started/locally/
   - Select: Stable → (Your OS) → C++/Java → CPU or CUDA
   - For macOS: Download the CPU version

2. Extract and set environment variables:

```bash
# Download (example for macOS CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip

# Set environment variables
export LIBTORCH=$PWD/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH  # macOS
# or
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH  # Linux
```

#### Option 2: Use Existing PyTorch Installation

If you have PyTorch installed via pip/conda:

```bash
export LIBTORCH_USE_PYTORCH=1
```

## Building

Once LibTorch is installed:

```bash
# From workspace root
make test-bert-tch
```

Or manually:

```bash
cd examples/bert-tch
cargo build --release

# Trigger package assembly
touch build.rs
cargo build --release

# Verify package created
ls ../../target/mlpkg/bert-tch
```

## Running Tests

```bash
cd examples/bert-tch
cargo test --test e2e_test -- --ignored --nocapture
```

## Architecture

This example demonstrates:

1. **Manual BERT Implementation** - Full BERT architecture using `tch::nn` modules:
   - Embeddings (word + position + token type)
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - 12 encoder layers

2. **Safetensors Weight Loading** - Direct loading from HuggingFace format:
   - Parses safetensors file
   - Creates tch::Tensor from raw data
   - Loads into nn::VarStore

3. **Model Packaging** - Uses `inferox-mlpkg`:
   - Downloads bert-base-uncased from HF Hub
   - Copies config.json and model.safetensors
   - Compiles as cdylib
   - Assembles to `target/mlpkg/bert-tch/`

4. **Dynamic Loading** - Loads via libloading:
   - Exports `create_model()` function
   - Compatible with InferoxEngine

## Package Structure

```
target/mlpkg/bert-tch/
├── metadata.json          # Package metadata
├── model_info.json        # Model architecture info
├── backends/
│   └── tch/
│       ├── libbert_tch.dylib  # Compiled model library
│       ├── config.json         # BERT config
│       └── model.safetensors   # Model weights
```

## Usage Example

```rust
use inferox_tch::TchBackend;
use inferox_core::{Backend, TensorBuilder};
use inferox_mlpkg::{PackageManager, BackendType};

// Create backend
let backend = TchBackend::cpu()?;

// Load package
let manager = PackageManager::new(cache_dir)?;
let package = manager.load_package(&package_path)?;

// Load model (backend determined from model.toml)
let loaded_model = manager.load_model(&package)?;
let model = loaded_model.as_tch().expect("Expected Tch model");

// Run inference
let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];  // [CLS] this is a test [SEP]
let input_tensor = backend
    .tensor_builder()
    .build_from_vec(input_ids, &[1, 6])?;

let output = model.forward(input_tensor)?;
// Output shape: [1, 6, 768]
```

## Notes

- **Workspace Exclusion**: This example is excluded from the main workspace because it requires LibTorch
- **CI**: Not run in CI due to LibTorch requirement
- **Performance**: Uses PyTorch's optimized CUDA/MPS kernels when available
- **Compatibility**: Works with any HuggingFace BERT model weights

## Troubleshooting

### "cannot find -ltorch" error

LibTorch is not installed or environment variables are not set. Follow the installation steps above.

### "failed to run custom build command for `torch-sys`"

The `tch` crate's build script cannot find LibTorch. Ensure `LIBTORCH` or `LIBTORCH_USE_PYTORCH` is set.

### Package assembly fails

Make sure to run the build twice to trigger package assembly:
```bash
cargo build --release
touch build.rs
cargo build --release
```

## Comparison with BERT-Candle

| Feature | bert-tch | bert-candle |
|---------|----------|-------------|
| Backend | PyTorch (tch-rs) | Candle (Rust-native) |
| Installation | Requires LibTorch | Pure Rust |
| Implementation | Manual BERT | Uses candle-transformers |
| Weight Loading | Safetensors manual | VarBuilder |
| Performance | Mature PyTorch kernels | Fast Rust kernels |
| Binary Size | Large (~200MB) | Small (~10MB) |
