# inferox-tch

PyTorch (tch-rs) backend implementation for Inferox.

## Overview

This crate provides a PyTorch backend for the Inferox ML inference library using the [tch-rs](https://github.com/LaurentMazare/tch-rs) bindings to libtorch. It implements all core Inferox traits (`Backend`, `Tensor`, `Device`, `TensorBuilder`) to enable PyTorch-based model inference within the Inferox ecosystem.

## Features

- **Full PyTorch Integration**: Direct bindings to libtorch via tch-rs
- **GPU Support**: CUDA and MPS (Apple Silicon) device support
- **Type Safety**: Rust-native tensor operations with type safety
- **Consistent API**: Follows the same patterns as other Inferox backends (inferox-candle)
- **Comprehensive Testing**: Full test coverage for all backend operations

## Prerequisites

### Installing LibTorch

The `tch-rs` crate requires LibTorch (the C++ backend of PyTorch) to be installed. You have several options:

#### Option 1: Download LibTorch (Recommended)

1. Download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/)
2. Extract the archive
3. Set the `LIBTORCH` environment variable:

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH  # macOS
```

#### Option 2: Use System PyTorch

If you have PyTorch installed via pip/conda:

```bash
export LIBTORCH_USE_PYTORCH=1
```

#### Option 3: System-wide Install

Install libtorch system-wide (e.g., `/usr/lib/libtorch.so` on Linux).

For detailed setup instructions, see the [tch-rs README](https://github.com/LaurentMazare/tch-rs).

## Usage

```rust
use inferox_core::{Backend, TensorBuilder, DType};
use inferox_tch::TchBackend;

// Create a CPU backend
let backend = TchBackend::cpu()?;

// Or use CUDA if available
let backend = TchBackend::cuda(0)?;

// Build tensors
let builder = backend.tensor_builder();
let tensor = builder.zeros(&[2, 3], DType::F32)?;

// Get available devices
let devices = backend.devices()?;
```

## Architecture

The crate follows the same structure as `inferox-candle`:

```
src/
├── lib.rs           - Public API exports
├── backend.rs       - TchBackend implementation
├── device.rs        - TchDeviceWrapper for device management
├── tensor.rs        - TchTensor wrapper and DataType implementation
└── tensor_builder.rs - TensorBuilder implementation
```

### Key Components

- **TchBackend**: Implements `Backend` trait, manages device and tensor creation
- **TchTensor**: Wraps `tch::Tensor`, implements `Tensor` trait
- **TchDeviceWrapper**: Wraps `tch::Device`, implements `Device` trait
- **TchTensorBuilder**: Implements `TensorBuilder` for creating tensors
- **TchDTypeWrapper**: Wraps `tch::Kind`, implements `DataType` trait

## Integration with Inferox Engine

This backend can be used with `inferox-engine` for dynamic model loading:

```rust
use inferox_engine::Engine;
use inferox_tch::TchBackend;

let backend = TchBackend::cpu()?;
let mut engine = Engine::new(backend);

// Load and register models...
```

## Supported Data Types

- `f32` (Float)
- `f64` (Double)
- `i32` (Int)
- `i64` (Int64)
- `u8` (Uint8)
- `i8` (Int8)
- `i16` (Int16)
- `f16` (Half)
- `bf16` (BFloat16)
- `bool` (Bool)

## Device Support

- **CPU**: Always available
- **CUDA**: Available when CUDA is installed and `tch::Cuda::is_available()` returns true
- **MPS**: Apple Silicon GPU support via `TchDevice::Mps`
- **Vulkan**: Basic support (maps to CPU in DeviceId)

## Development

### Running Tests

Tests require LibTorch to be installed:

```bash
cargo test -p inferox-tch
```

### Building

```bash
cargo build -p inferox-tch
```

### CI/CD Considerations

For CI pipelines, you'll need to:

1. Install LibTorch in the CI environment
2. Set appropriate environment variables
3. Consider using CPU-only LibTorch for faster builds

Example for GitHub Actions:

```yaml
- name: Install LibTorch
  run: |
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
    echo "LIBTORCH=$PWD/libtorch" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=$PWD/libtorch/lib:$LD_LIBRARY_PATH" >> $GITHUB_ENV
```

## Comparison with inferox-candle

| Feature | inferox-tch | inferox-candle |
|---------|-------------|----------------|
| Backend | PyTorch (libtorch) | Candle (Rust-native) |
| Installation | Requires libtorch | Pure Rust, no deps |
| Performance | Mature, optimized | Fast, growing |
| GPU Support | CUDA, MPS | CUDA, Metal |
| Ecosystem | Full PyTorch ecosystem | Rust-native models |
| Binary Size | Larger (libtorch) | Smaller |

## License

MIT OR Apache-2.0
