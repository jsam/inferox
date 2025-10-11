# RFC 0002: PyTorch (tch-rs) Backend for Inferox

- **Status**: Proposed
- **Created**: 2025-01-11
- **Author**: Claude Code
- **Related RFCs**: [RFC 0001: mlpkg Multi-Backend Support](./0001-mlpkg-multi-backend.md)

## Summary

This RFC proposes adding a PyTorch backend implementation (`inferox-tch`) to the Inferox ecosystem using the `tch-rs` Rust bindings to libtorch. This will enable users to run PyTorch-trained models within Inferox while maintaining API consistency with existing backends.

## Motivation

### Why Add a PyTorch Backend?

1. **Ecosystem Compatibility**: PyTorch is one of the most popular ML frameworks. Supporting it allows Inferox to work with the vast ecosystem of PyTorch models.

2. **Production-Ready Performance**: LibTorch is a mature, highly-optimized inference runtime with years of development and optimization.

3. **Model Portability**: Users can train models in PyTorch (Python) and deploy them in Inferox (Rust) without rewriting model code.

4. **Multi-Backend Strategy**: Following RFC 0001, this demonstrates the multi-backend architecture by adding a second major backend alongside Candle.

5. **GPU Support**: Leverage PyTorch's mature CUDA and MPS (Apple Silicon) implementations.

### Use Cases

- **PyTorch Model Deployment**: Run existing PyTorch models in production Rust services
- **Cross-Framework Comparison**: Compare performance between Candle and PyTorch backends
- **Gradual Migration**: Migrate from PyTorch to Rust-native solutions incrementally
- **Backend Fallback**: Use PyTorch as fallback when Candle doesn't support specific operations

## Design

### Architecture Overview

The `inferox-tch` crate follows the same architectural pattern as `inferox-candle`:

```
┌─────────────────────────────────────┐
│      inferox-engine (Engine)        │
│  Model Registry & Dynamic Loading   │
└──────────────┬──────────────────────┘
               │ uses Backend trait
               ▼
┌──────────────────────────────────────┐
│         inferox-core                 │
│  Backend, Tensor, Device traits     │
└──────┬───────────────────────┬───────┘
       │                       │
       ▼                       ▼
┌──────────────┐      ┌──────────────┐
│ inferox-tch  │      │inferox-candle│
│ (PyTorch)    │      │ (Rust-native)│
└──────────────┘      └──────────────┘
```

### Core Components

#### 1. TchBackend

Implements the `Backend` trait from `inferox-core`:

```rust
pub struct TchBackend {
    device: tch::Device,
}

impl Backend for TchBackend {
    type Tensor = TchTensor;
    type Error = tch::TchError;
    type Device = TchDeviceWrapper;
    type TensorBuilder = TchTensorBuilder;
    
    fn name(&self) -> &str {
        "tch"
    }
    
    fn devices(&self) -> Result<Vec<Self::Device>, Self::Error> {
        // Lists available CPU and CUDA devices
    }
    
    fn default_device(&self) -> Self::Device {
        // Returns configured device
    }
    
    fn tensor_builder(&self) -> Self::TensorBuilder {
        // Creates tensor builder for this device
    }
}
```

**Key Features:**
- Auto-detection of CUDA availability
- Support for CPU, CUDA, MPS (Apple Silicon), and Vulkan devices
- Thread-safe, cloneable backend instance
- Factory methods: `new()`, `cpu()`, `cuda(ordinal)`

#### 2. TchTensor

Wraps `tch::Tensor` and implements `Tensor` trait:

```rust
pub struct TchTensor(pub tch::Tensor);

impl Tensor for TchTensor {
    type Dtype = TchDTypeWrapper;
    type Backend = TchBackend;
    
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> Self::Dtype;
    fn device(&self) -> TchDeviceWrapper;
    fn to_device(&self, device: &TchDeviceWrapper) -> Result<Self, tch::TchError>;
    fn reshape(&self, shape: &[usize]) -> Result<Self, tch::TchError>;
    fn contiguous(&self) -> Result<Self, tch::TchError>;
}
```

**Key Features:**
- Zero-copy wrapping of `tch::Tensor`
- Shape, dtype, and device introspection
- Device transfer operations
- Tensor reshaping and layout operations

#### 3. TchDeviceWrapper

Wraps `tch::Device` and implements `Device` trait:

```rust
pub struct TchDeviceWrapper(pub(crate) tch::Device);

impl Device for TchDeviceWrapper {
    fn id(&self) -> DeviceId {
        match self.0 {
            TchDevice::Cpu => DeviceId::Cpu,
            TchDevice::Cuda(ordinal) => DeviceId::Cuda(ordinal),
            TchDevice::Mps => DeviceId::Metal(0),
            TchDevice::Vulkan => DeviceId::Cpu,
        }
    }
    
    fn is_available(&self) -> bool {
        // Checks CUDA availability for GPU devices
    }
    
    fn memory_info(&self) -> Option<MemoryInfo> {
        // Currently returns None, could be extended
    }
}
```

**Key Features:**
- Maps PyTorch devices to Inferox `DeviceId` enum
- Runtime availability checking for CUDA
- MPS support for Apple Silicon

#### 4. TchTensorBuilder

Implements `TensorBuilder` for tensor creation:

```rust
pub struct TchTensorBuilder {
    pub(crate) device: tch::Device,
}

impl TensorBuilder<TchBackend> for TchTensorBuilder {
    fn build_from_slice<T: NumericType>(...) -> Result<TchTensor, tch::TchError>;
    fn build_from_vec<T: NumericType>(...) -> Result<TchTensor, tch::TchError>;
    fn zeros(...) -> Result<TchTensor, tch::TchError>;
    fn ones(...) -> Result<TchTensor, tch::TchError>;
    fn randn(...) -> Result<TchTensor, tch::TchError>;
    fn with_device(self, device: TchDeviceWrapper) -> Self;
}
```

**Key Features:**
- Tensor creation from Rust data (slices, vectors)
- Factory functions for common initializations
- Type-safe numeric type conversions
- Device-aware tensor creation

#### 5. TchDTypeWrapper

Wraps `tch::Kind` for data type representation:

```rust
pub struct TchDTypeWrapper(pub tch::Kind);

impl DataType for TchDTypeWrapper {
    fn name(&self) -> &str {
        // Maps tch::Kind to string names
    }
    
    fn size(&self) -> usize {
        // Returns size in bytes
    }
}
```

**Supported Types:**
- f32, f64 (Float, Double)
- i32, i64, i8, i16 (Int variants)
- u8 (Uint8)
- f16, bf16 (Half, BFloat16)
- bool

### Integration with inferox-mlpkg

The multi-backend package system (RFC 0001) will automatically support the tch backend:

```rust
// model_info.json in package
{
    "model_type": "bert",
    "repo_id": "bert-base-uncased",
    "architecture_family": "EncoderOnly",
    "supported_backends": ["Tch"],  // or ["Candle", "Tch"] for multi-backend
    "hidden_size": 768,
    "num_layers": 12,
    "vocab_size": 30522
}
```

**Package Structure:**
```
bert-base-uncased/
├── metadata.json
├── model_info.json
└── backends/
    ├── candle/
    │   ├── libmodel.dylib
    │   ├── config.json
    │   └── model.safetensors
    └── tch/
        ├── libmodel.dylib    # Compiled with inferox-tch
        ├── config.json
        └── model.pt          # PyTorch weights
```

### Dependencies

**Direct Dependencies:**
- `inferox-core` (0.1.0) - Core traits
- `tch` (0.17) - PyTorch Rust bindings
- `thiserror` (1.0) - Error handling

**External Requirements:**
- LibTorch (C++ PyTorch library)
- CUDA Toolkit (optional, for GPU support)

## Implementation Plan

### Phase 1: Core Implementation ✅

- [x] Create `inferox-tch` crate structure
- [x] Implement `TchBackend`
- [x] Implement `TchTensor` and `TchDTypeWrapper`
- [x] Implement `TchDeviceWrapper`
- [x] Implement `TchTensorBuilder`
- [x] Add comprehensive unit tests
- [x] Add README with setup instructions

### Phase 2: CI/CD Setup ✅

- [x] Add LibTorch installation to CI pipelines (excluded tch from CI)
- [x] Create CPU-only test configuration for CI
- [x] Add platform-specific build instructions (Linux, macOS, Windows)
- [x] Update workspace documentation

### Phase 3: Example Implementation (In Progress)

**Goal:** Create BERT example using inferox-tch backend with full E2E testing

#### Architecture
```
examples/bert-tch/
├── Cargo.toml              # cdylib package with tch dependencies
├── build.rs                # Package assembly via BuildScriptRunner
├── src/
│   └── lib.rs              # BERT implementation using tch-rs
└── tests/
    └── e2e_test.rs         # E2E tests (package loading + engine integration)
```

#### Implementation Approach

**BERT Model Loading Strategy:**
- **Option A:** Manual BERT implementation using tch::nn modules (complex, flexible)
- **Option B:** Load TorchScript model via tch::CModule (simpler, recommended) ✅
- **Option C:** Use safetensors + manual tensor loading (medium complexity)

**Selected: Option B (TorchScript)** - Most reliable, leverages PyTorch's maturity

#### Components

**1. Model Wrapper (`BertModelWrapper`)**
```rust
pub struct BertModelWrapper {
    name: String,
    model: tch::CModule,  // TorchScript module
}

impl Model for BertModelWrapper {
    type Backend = TchBackend;
    type Input = TchTensor;
    type Output = TchTensor;
    
    fn forward(&self, input: TchTensor) -> Result<TchTensor, tch::TchError> {
        // Convert to IValue, run model, extract output
    }
}
```

**2. Weight Loading**
- Safetensors → tch::Tensor conversion using `safetensors` crate
- HF weight name mapping (e.g., `bert.encoder.layer.0.attention.self.query.weight`)
- Dtype and device handling

**3. Package Assembly**
- BuildScriptRunner downloads bert-base-uncased from HF Hub
- Copies config.json and model.safetensors
- Compiles cdylib and assembles to target/mlpkg/bert-tch/

**4. E2E Tests**
- Test 1: Package loading + direct inference
- Test 2: Integration with InferoxEngine
- Validation: Output shape [1, seq_len, 768]

#### Tasks

- [ ] Create project structure and Cargo.toml
- [ ] Implement BertModelWrapper with TorchScript loading
- [ ] Add safetensors weight loading
- [ ] Implement create_model() export function
- [ ] Add build.rs with BuildScriptRunner
- [ ] Write E2E test: package loading
- [ ] Write E2E test: engine integration
- [ ] Add Makefile target: test-bert-tch
- [ ] Document usage in README
- [ ] Performance benchmarking vs. Candle backend

### Phase 4: Integration with inferox-mlpkg

- [ ] Update `BackendType` enum to include `Tch`
- [ ] Add PyTorch weight loading (`model.pt`, `model.pth`)
- [ ] Support `torch.jit.save` serialized models
- [ ] Update package assembly to handle PyTorch models

### Phase 5: Documentation & Examples

- [ ] Add migration guide from PyTorch to Inferox
- [ ] Document model conversion workflow
- [ ] Create example Jupyter notebooks
- [ ] Add performance comparison benchmarks

## Trade-offs and Alternatives

### Advantages of inferox-tch

**Pros:**
1. **Mature Ecosystem**: Leverage years of PyTorch optimization
2. **Model Compatibility**: Use existing PyTorch models without conversion
3. **GPU Performance**: Battle-tested CUDA kernels
4. **Community Support**: Large PyTorch user base
5. **Feature Completeness**: Full operator coverage

**Cons:**
1. **Binary Size**: LibTorch is large (~200MB+ uncompressed)
2. **Build Complexity**: Requires external LibTorch installation
3. **Runtime Dependencies**: Must distribute or install LibTorch
4. **Less "Rust-Native"**: Depends on C++ library

### Comparison with inferox-candle

| Aspect | inferox-tch | inferox-candle |
|--------|-------------|----------------|
| **Installation** | Complex (LibTorch required) | Simple (pure Rust) |
| **Binary Size** | Large (200MB+) | Small (10-20MB) |
| **Build Time** | Slow (C++ linking) | Fast (Rust only) |
| **Performance** | Mature, optimized | Fast, improving |
| **GPU Support** | CUDA, MPS | CUDA, Metal |
| **Model Compatibility** | PyTorch native | Requires conversion |
| **Operator Coverage** | Complete | Growing |
| **Deployment** | Heavier | Lighter |

### Alternative Approaches Considered

#### 1. ONNX Runtime Backend

**Why not chosen:**
- ONNX adds another conversion step (PyTorch → ONNX → Inferox)
- Less direct access to PyTorch ecosystem
- Still requires external C++ runtime

**When to use:**
- Cross-framework model deployment
- When model source is ONNX already
- Could be implemented as RFC 0003 in future

#### 2. Pure Rust PyTorch Reimplementation

**Why not chosen:**
- Massive engineering effort
- Candle already serves this purpose
- Difficult to match PyTorch's operator coverage

**When to use:**
- This is essentially what Candle provides

#### 3. JIT/Script Module Support Only

**Why not chosen:**
- TorchScript has limitations
- Regular PyTorch models more common
- Can add as incremental feature later

### Why Both Candle and Tch?

Having both backends provides:

1. **Choice**: Users pick based on their constraints
2. **Fallback**: If one backend lacks an op, try the other
3. **Performance Comparison**: Benchmark and optimize
4. **Migration Path**: Start with Tch, move to Candle gradually
5. **Proof of Concept**: Validates multi-backend architecture

## Testing Strategy

### Unit Tests

Each component has comprehensive unit tests:

```rust
#[cfg(test)]
mod tests {
    // Backend tests
    - test_cpu_backend()
    - test_cuda_backend()  // skipped if CUDA unavailable
    - test_backend_devices()
    - test_backend_tensor_builder()
    
    // Tensor tests
    - test_tensor_shape()
    - test_tensor_dtype()
    - test_tensor_device_transfer()
    - test_tensor_reshape()
    
    // Device tests
    - test_device_availability()
    - test_device_id_mapping()
    
    // Builder tests
    - test_build_from_slice()
    - test_zeros_ones_randn()
    - test_dtype_conversions()
}
```

### Integration Tests

- [ ] Load PyTorch model and run inference
- [ ] Compare output with PyTorch Python reference
- [ ] Test device transfers (CPU ↔ CUDA)
- [ ] Test with various dtypes

### CI/CD Considerations

**CPU-Only CI:**
- Install CPU-only LibTorch
- Run all unit tests
- Skip CUDA-specific tests

**GPU CI (optional):**
- Requires GPU runners
- Install CUDA + LibTorch GPU
- Run CUDA tests

## Security & Safety

### Memory Safety

- **tch-rs** uses `unsafe` internally for FFI
- Wrapper types provide safe Rust API
- Reference counting handled by LibTorch

### Dependency Security

- **LibTorch**: Maintained by PyTorch/Meta
- **tch-rs**: Community-maintained, active development
- Scan dependencies for vulnerabilities regularly

### Runtime Safety

- Check CUDA availability before use
- Handle device OOM errors gracefully
- Validate tensor operations before execution

## Documentation Requirements

### User Documentation

- [x] README with installation instructions
- [ ] API documentation (rustdoc)
- [ ] Migration guide from PyTorch Python
- [ ] Troubleshooting guide for LibTorch setup

### Developer Documentation

- [x] RFC (this document)
- [ ] Architecture diagrams
- [ ] Contribution guidelines specific to tch backend
- [ ] Testing guidelines

## Future Extensions

### Potential Enhancements

1. **TorchScript Support**: Load `torch.jit.script` models
2. **Quantization**: INT8/FP16 quantized models
3. **Dynamic Batching**: Batch multiple requests
4. **Memory Pooling**: Custom allocators for performance
5. **Streaming**: Process data in chunks
6. **Custom Ops**: Register custom PyTorch operators
7. **Model Compilation**: Integrate with `torch.compile`

### Integration Opportunities

- **inferox-engine**: Dynamic backend selection
- **inferox-mlpkg**: PyTorch model packages
- **ONNX bridge**: Export/import via ONNX format

## Conclusion

The `inferox-tch` backend brings PyTorch's mature inference capabilities to Inferox while maintaining API consistency with other backends. It demonstrates the multi-backend architecture's flexibility and provides users with production-ready PyTorch inference in Rust.

### Success Criteria

1. ✅ All core traits implemented
2. ✅ Unit tests pass on CPU
3. ✅ CI/CD pipeline configured (tch excluded from CI)
4. [ ] Example model runs successfully
5. [ ] Documentation complete
6. [ ] Performance benchmarks available

### Next Steps

1. ✅ Configure CI (tch excluded from CI workflows)
2. Create `examples/bert-tch` demonstration
3. Integrate with `inferox-mlpkg` package system
4. Add performance benchmarks vs. Candle
5. Write migration guide for PyTorch users

## References

- [RFC 0001: mlpkg Multi-Backend Support](./0001-mlpkg-multi-backend.md)
- [tch-rs Repository](https://github.com/LaurentMazare/tch-rs)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [LibTorch Download](https://pytorch.org/get-started/locally/)
- [Inferox Core Traits](../crates/inferox-core/src/)
