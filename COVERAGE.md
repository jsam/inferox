# Test Coverage Implementation Checklist

Track progress by marking items as complete: `[ ]` → `[x]`

---

## Phase 1: inferox-core Tests (Critical - 0% → 90%)

### Backend Tests (`crates/inferox-core/tests/backend_test.rs`)
- [ ] Test Backend trait with CandleBackend
- [ ] Test Backend trait with TchBackend
- [ ] Test backend device selection (CPU)
- [ ] Test backend device selection (CUDA)
- [ ] Test backend device selection (Metal/MPS)
- [ ] Test backend tensor_builder() creation
- [ ] Test backend error handling

### Tensor Tests (`crates/inferox-core/tests/tensor_test.rs`)
- [ ] Test Tensor::shape() for various dimensions
- [ ] Test Tensor::dtype() for all supported types
- [ ] Test Tensor::device() accessor
- [ ] Test TensorBuilder::build_from_vec() with u8
- [ ] Test TensorBuilder::build_from_vec() with u16
- [ ] Test TensorBuilder::build_from_vec() with u32
- [ ] Test TensorBuilder::build_from_vec() with i8
- [ ] Test TensorBuilder::build_from_vec() with i16
- [ ] Test TensorBuilder::build_from_vec() with i32
- [ ] Test TensorBuilder::build_from_vec() with i64
- [ ] Test TensorBuilder::build_from_vec() with f32
- [ ] Test TensorBuilder::build_from_vec() with f64
- [ ] Test TensorBuilder::build_from_slice()
- [ ] Test TensorBuilder with empty tensor
- [ ] Test TensorBuilder with invalid shape (size mismatch)
- [ ] Test TensorBuilder with negative dimensions (should error)
- [ ] Test tensor shape validation
- [ ] Test tensor dtype conversions
- [ ] Test tensor device transfers (CPU → CUDA)
- [ ] Test tensor device transfers (CUDA → CPU)

### Model Tests (`crates/inferox-core/tests/model_test.rs`)
- [ ] Test Model trait with dummy Candle implementation
- [ ] Test Model trait with dummy Tch implementation
- [ ] Test Model::name() accessor
- [ ] Test Model::metadata() accessor
- [ ] Test Model::forward() with valid input
- [ ] Test Model::forward() with invalid input
- [ ] Test TypeErasedModel wrapper creation
- [ ] Test TypeErasedModel::forward_any() with Candle
- [ ] Test TypeErasedModel::forward_any() with Tch
- [ ] Test AnyModel trait implementation
- [ ] Test type erasure round-trip (concrete → erased → concrete)
- [ ] Test type mismatch error in downcasting

### Error Tests (`crates/inferox-core/tests/error_test.rs`)
- [ ] Test InferoxError::ModelNotFound variant
- [ ] Test InferoxError::InvalidInput variant
- [ ] Test InferoxError::Backend variant
- [ ] Test InferoxError::DeviceError variant
- [ ] Test InferoxError display messages
- [ ] Test error conversion from IO errors
- [ ] Test error conversion from backend-specific errors
- [ ] Test error propagation through Result types

### Device Tests (`crates/inferox-core/tests/device_test.rs`)
- [ ] Test Device creation for CPU
- [ ] Test Device creation for CUDA
- [ ] Test Device creation for Metal/MPS
- [ ] Test Device::is_cpu()
- [ ] Test Device::is_cuda()
- [ ] Test Device::is_metal()
- [ ] Test DeviceId parsing
- [ ] Test MemoryInfo tracking

### DType Tests (`crates/inferox-core/tests/dtype_test.rs`)
- [ ] Test DType enum variants
- [ ] Test DType size calculations
- [ ] Test DType::is_float()
- [ ] Test DType::is_int()
- [ ] Test DataType trait implementations
- [ ] Test NumericType conversions

---

## Phase 2: inferox-engine Tests (Critical - 30% → 85%)

### Engine Core Tests (expand `crates/inferox-engine/src/engine.rs`)
- [ ] Test InferoxEngine::new() with default config
- [ ] Test InferoxEngine::new() with custom config
- [ ] Test register_model() with CandleBackend model
- [ ] Test register_model() with TchBackend model
- [ ] Test register_model() with duplicate names (should replace)
- [ ] Test register_model() with multiple different models
- [ ] Test infer() with correct backend type
- [ ] Test infer() with model not found error
- [ ] Test infer() with type mismatch error
- [ ] Test infer() with valid input tensor
- [ ] Test infer() with invalid input tensor
- [ ] Test infer_batch() with single item
- [ ] Test infer_batch() with multiple items
- [ ] Test infer_batch() with empty batch
- [ ] Test infer_batch() with large batch (100+ items)
- [ ] Test list_models() returns all registered models
- [ ] Test list_models() with empty engine
- [ ] Test model_info() retrieves correct metadata
- [ ] Test model_info() with non-existent model
- [ ] Test unregister_model() (if implemented)
- [ ] Test engine with zero models

### Multi-Backend Tests (`crates/inferox-engine/tests/multi_backend_test.rs`)
- [ ] Test Candle model registration
- [ ] Test Tch model registration
- [ ] Test both backends in same engine
- [ ] Test inferring on Candle model
- [ ] Test inferring on Tch model
- [ ] Test switching between backends in same session
- [ ] Test concurrent inference on different backends
- [ ] Test backend isolation (no cross-contamination)
- [ ] Test model name uniqueness across backends

### Session Tests (`crates/inferox-engine/tests/session_test.rs`)
- [ ] Test InferenceSession::new() creation
- [ ] Test session.run() with valid input
- [ ] Test session.run() with multiple calls
- [ ] Test session.set_batch_size()
- [ ] Test session.store_state() with string value
- [ ] Test session.store_state() with custom type
- [ ] Test session.get_state() retrieval
- [ ] Test session.get_state() with non-existent key
- [ ] Test session.get_state() with wrong type
- [ ] Test session with multiple models
- [ ] Test session error handling
- [ ] Test session state isolation between sessions

### Config Tests (`crates/inferox-engine/tests/config_test.rs`)
- [ ] Test EngineConfig::default() values
- [ ] Test EngineConfig::new()
- [ ] Test with_max_batch_size() builder
- [ ] Test with_profiling() builder
- [ ] Test with_memory_pool_size() builder
- [ ] Test builder chaining
- [ ] Test config validation
- [ ] Test invalid config values

### Concurrency Tests (`crates/inferox-engine/tests/concurrency_test.rs`)
- [ ] Test concurrent model registration
- [ ] Test concurrent inference requests (same model)
- [ ] Test concurrent inference requests (different models)
- [ ] Test thread safety of model storage
- [ ] Test race conditions in model lookup
- [ ] Test concurrent session creation

---

## Phase 3: inferox-mlpkg Tests (High Priority - 40% → 80%)

### Package Assembly Tests (`crates/inferox-mlpkg/tests/package_assembly_test.rs`)
- [ ] Test assemble_package() with Candle backend
- [ ] Test assemble_package() with Tch backend
- [ ] Test package directory structure creation
- [ ] Test backends/ subdirectory creation
- [ ] Test backends/candle/ directory creation
- [ ] Test backends/tch/ directory creation
- [ ] Test metadata.json creation
- [ ] Test metadata.json content validity
- [ ] Test model_info.json creation
- [ ] Test model_info.json content validity
- [ ] Test library copying to backend directory (.dylib)
- [ ] Test library copying to backend directory (.so)
- [ ] Test library copying to backend directory (.dll)
- [ ] Test weight file copying (safetensors)
- [ ] Test config file copying (config.json)
- [ ] Test file permissions after copying
- [ ] Test package assembly with missing library
- [ ] Test package assembly with missing config
- [ ] Test package assembly with missing weights

### Package Loading Tests (`crates/inferox-mlpkg/tests/package_loading_test.rs`)
- [ ] Test load_package() from valid package
- [ ] Test load_package() metadata parsing
- [ ] Test load_package() model_info parsing
- [ ] Test load_package() with missing metadata.json
- [ ] Test load_package() with invalid metadata.json
- [ ] Test load_package() with missing model_info.json
- [ ] Test load_package() with invalid model_info.json
- [ ] Test load_package() with missing backend directory
- [ ] Test package validation logic
- [ ] Test package version compatibility

### Model Loading Tests (`crates/inferox-mlpkg/tests/model_loading_test.rs`)
- [ ] Test load_model() with Candle backend
- [ ] Test load_model() with Tch backend
- [ ] Test library symbol resolution (create_model)
- [ ] Test create_model() function invocation
- [ ] Test device wrapper creation (Candle)
- [ ] Test device wrapper creation (Tch)
- [ ] Test LoadedModel enum wrapping
- [ ] Test load_model() with missing library file
- [ ] Test load_model() with corrupt library file
- [ ] Test load_model() with wrong backend library
- [ ] Test load_model() with missing create_model symbol

### Build Script Tests (`crates/inferox-mlpkg/tests/build_script_test.rs`)
- [ ] Test BuildScriptRunner::new() creation
- [ ] Test BuildScriptRunner::run() flow
- [ ] Test backend parsing from model.toml (candle)
- [ ] Test backend parsing from model.toml (Candle)
- [ ] Test backend parsing from model.toml (tch)
- [ ] Test backend parsing from model.toml (Tch)
- [ ] Test case-insensitive backend parsing
- [ ] Test invalid backend value error
- [ ] Test missing backend field error
- [ ] Test missing model.toml file error
- [ ] Test INFEROX_PACKAGE_DIR environment variable
- [ ] Test package directory creation
- [ ] Test library path resolution

### DeviceWrapper & LoadedModel Tests (expand `crates/inferox-mlpkg/tests/unit_tests.rs`)
- [ ] Test DeviceWrapper::as_candle() success
- [ ] Test DeviceWrapper::as_candle() with wrong type
- [ ] Test DeviceWrapper::as_tch() success
- [ ] Test DeviceWrapper::as_tch() with wrong type
- [ ] Test DeviceWrapper::device_type() for CPU
- [ ] Test DeviceWrapper::device_type() for CUDA
- [ ] Test LoadedModel::as_candle() success
- [ ] Test LoadedModel::as_candle() with wrong type
- [ ] Test LoadedModel::as_tch() success
- [ ] Test LoadedModel::as_tch() with wrong type
- [ ] Test LoadedModel::name() accessor
- [ ] Test LoadedModel::backend_type() detection

### HF Integration Tests (`crates/inferox-mlpkg/tests/hf_integration_test.rs`)
- [ ] Test download_and_package() with small model
- [ ] Test download_and_package() with bert-base-uncased
- [ ] Test download with revision="main"
- [ ] Test download with specific commit SHA
- [ ] Test download with HF token authentication
- [ ] Test download private model with token
- [ ] Test download model not found error
- [ ] Test download network failure handling
- [ ] Test download timeout handling
- [ ] Test download resume after interruption
- [ ] Test concurrent downloads

---

## Phase 4: inferox-candle Tests (Medium Priority - 20% → 75%)

### Tensor Tests (`crates/inferox-candle/tests/tensor_test.rs`)
- [ ] Test CandleTensor creation from Vec<f32>
- [ ] Test CandleTensor creation from Vec<f64>
- [ ] Test CandleTensor creation from Vec<i64>
- [ ] Test CandleTensor creation from Vec<u32>
- [ ] Test CandleTensor::shape()
- [ ] Test CandleTensor::dtype()
- [ ] Test CandleTensor::device()
- [ ] Test tensor reshape operation
- [ ] Test tensor transpose operation
- [ ] Test tensor slice operation
- [ ] Test tensor device transfer (CPU → CUDA)
- [ ] Test tensor device transfer (CUDA → CPU)
- [ ] Test tensor dtype conversion (f32 → f64)
- [ ] Test tensor dtype conversion (i64 → f32)
- [ ] Test tensor operations (add, mul, etc.)

### Backend Tests (`crates/inferox-candle/tests/backend_test.rs`)
- [ ] Test CandleBackend::cpu() creation
- [ ] Test CandleBackend::cuda() creation
- [ ] Test CandleBackend::cuda() with device_id
- [ ] Test CandleBackend::metal() creation (macOS)
- [ ] Test CandleBackend::with_device()
- [ ] Test backend device detection
- [ ] Test CUDA availability check
- [ ] Test Metal availability check
- [ ] Test backend tensor_builder() creation
- [ ] Test multi-GPU support

### TensorBuilder Tests (`crates/inferox-candle/tests/tensor_builder_test.rs`)
- [ ] Test build_from_vec() with Vec<f32>
- [ ] Test build_from_vec() with Vec<f64>
- [ ] Test build_from_vec() with Vec<i32>
- [ ] Test build_from_vec() with Vec<i64>
- [ ] Test build_from_vec() with Vec<u8>
- [ ] Test build_from_vec() with 1D shape
- [ ] Test build_from_vec() with 2D shape
- [ ] Test build_from_vec() with 3D shape
- [ ] Test build_from_vec() with 4D shape
- [ ] Test build_from_slice()
- [ ] Test empty tensor creation
- [ ] Test large tensor creation (>1GB)
- [ ] Test invalid shape error (size mismatch)
- [ ] Test invalid shape error (negative dimension)
- [ ] Test out of memory error

### Device Tests (`crates/inferox-candle/tests/device_test.rs`)
- [ ] Test device availability checking
- [ ] Test device switching
- [ ] Test out of memory handling
- [ ] Test device synchronization
- [ ] Test memory tracking

---

## Phase 5: inferox-tch Tests (Medium Priority - 20% → 75%)

### Tensor Tests (`crates/inferox-tch/tests/tensor_test.rs`)
- [ ] Test TchTensor creation from Vec<f32>
- [ ] Test TchTensor creation from Vec<f64>
- [ ] Test TchTensor creation from Vec<i64>
- [ ] Test TchTensor::shape()
- [ ] Test TchTensor::dtype()
- [ ] Test TchTensor::device()
- [ ] Test tensor operations (add, mul, matmul)
- [ ] Test tensor indexing
- [ ] Test tensor slicing
- [ ] Test tensor broadcasting
- [ ] Test tensor device transfer
- [ ] Test tensor dtype conversion
- [ ] Test gradient tracking (if applicable)

### Backend Tests (`crates/inferox-tch/tests/backend_test.rs`)
- [ ] Test TchBackend::cpu() creation
- [ ] Test TchBackend::cuda() creation
- [ ] Test TchBackend::cuda() with device_id
- [ ] Test LibTorch initialization
- [ ] Test CUDA availability detection
- [ ] Test backend with custom device
- [ ] Test backend tensor_builder() creation

### Integration Tests (`crates/inferox-tch/tests/integration_test.rs`)
- [ ] Test loading PyTorch .pt model
- [ ] Test loading PyTorch .pth model
- [ ] Test model saving to PyTorch format
- [ ] Test PyTorch tensor compatibility
- [ ] Test JIT model support (if applicable)
- [ ] Test TorchScript models

### Error Handling Tests (`crates/inferox-tch/tests/error_test.rs`)
- [ ] Test LibTorch error propagation
- [ ] Test CUDA errors
- [ ] Test device unavailable error
- [ ] Test invalid operations
- [ ] Test out of memory error

---

## Phase 6: hf-xet-rs Enhancement (Medium Priority - 70% → 85%)

### Advanced Cache Tests (`crates/hf-xet-rs/tests/cache_advanced_test.rs`)
- [ ] Test cache size limit enforcement
- [ ] Test cache eviction when limit reached
- [ ] Test LRU eviction order
- [ ] Test concurrent cache access (thread safety)
- [ ] Test concurrent put operations
- [ ] Test concurrent get operations
- [ ] Test cache corruption detection
- [ ] Test cache corruption recovery
- [ ] Test cache metadata persistence

### XET Reconstruction Tests (`crates/hf-xet-rs/tests/xet_reconstruction_test.rs`)
- [ ] Test chunk reconstruction from XET format
- [ ] Test multi-chunk file reconstruction
- [ ] Test corrupted chunk detection
- [ ] Test corrupted chunk recovery
- [ ] Test missing chunk handling
- [ ] Test partial chunk data
- [ ] Test reconstruction performance

### Error Recovery Tests (`crates/hf-xet-rs/tests/error_recovery_test.rs`)
- [ ] Test retry logic on failure
- [ ] Test exponential backoff calculation
- [ ] Test max retries enforcement
- [ ] Test timeout scenarios
- [ ] Test network timeout handling
- [ ] Test partial file recovery
- [ ] Test resume download after failure

### Edge Cases Tests (`crates/hf-xet-rs/tests/edge_cases_test.rs`)
- [ ] Test very large file download (>10GB)
- [ ] Test many small files (>1000)
- [ ] Test special characters in filenames
- [ ] Test unicode filenames
- [ ] Test symlink handling
- [ ] Test empty repository
- [ ] Test repository with no models
- [ ] Test concurrent downloads to same file

---

## Phase 7: Examples Enhancement (Low Priority)

### MLP Tests (`examples/mlp/tests/`)
- [ ] Test different input dimensions (5, 10, 20, 50)
- [ ] Test batch size variations (1, 4, 8, 16, 32)
- [ ] Test invalid input shape error
- [ ] Test invalid input type error
- [ ] Test model weight modification
- [ ] Test model serialization
- [ ] Test model deserialization
- [ ] Test memory leak detection

### MLP Benchmarks (`examples/mlp/benches/performance_bench.rs`)
- [ ] Benchmark single inference latency
- [ ] Benchmark batch inference throughput
- [ ] Benchmark with different batch sizes
- [ ] Benchmark memory usage
- [ ] Compare with baseline implementation

### BERT-Candle Tests (`examples/bert-candle/tests/`)
- [ ] Test sequence length variations (16, 32, 64, 128, 256, 512)
- [ ] Test batch inference (batch size 1-16)
- [ ] Test attention mask handling
- [ ] Test token type IDs handling
- [ ] Test position IDs handling
- [ ] Test padding handling
- [ ] Test truncation handling
- [ ] Test maximum sequence length enforcement (512)
- [ ] Test invalid token IDs error
- [ ] Test sequence too long error
- [ ] Test empty input error

### BERT-Tch Tests (`examples/bert-tch/tests/`)
- [ ] Test sequence length variations
- [ ] Test batch inference
- [ ] Test model comparison with Candle (output consistency)
- [ ] Test performance comparison with Candle
- [ ] Test memory usage comparison

### BERT Error Handling (`examples/bert-*/tests/error_handling_test.rs`)
- [ ] Test model loading failures
- [ ] Test inference failures
- [ ] Test out of memory handling
- [ ] Test invalid configuration

### Multi-Backend BERT E2E (`examples/bert-multi-backend/tests/e2e_test.rs`)
- [ ] Create new test module `examples/bert-multi-backend/`
- [ ] Test loading both bert-candle and bert-tch packages
- [ ] Test registering both models in single engine
- [ ] Test routing inference to Candle model based on request metadata
- [ ] Test routing inference to Tch model based on request metadata
- [ ] Test consolidated response format (unified tensor type)
- [ ] Test response tensor type conversion (Candle → unified)
- [ ] Test response tensor type conversion (Tch → unified)
- [ ] Test backend selection via request metadata field
- [ ] Test fallback when preferred backend unavailable
- [ ] Test concurrent requests to both backends
- [ ] Test output consistency between backends (numerical comparison)
- [ ] Test performance comparison between backends
- [ ] Test error handling when one backend fails
- [ ] Test graceful degradation when backend unavailable

---

## Phase 8: Integration & E2E Tests (Medium Priority)

### Cross-Component Integration (`tests/integration/end_to_end_test.rs`)
- [ ] Test full flow: download → package → load → infer (Candle)
- [ ] Test full flow: download → package → load → infer (Tch)
- [ ] Test multi-model engine with Candle + Tch
- [ ] Test model hot-swapping
- [ ] Test concurrent inference requests (10+)
- [ ] Test concurrent inference requests (100+)
- [ ] Test model unload and reload

### Cross-Backend Tests (`tests/integration/cross_backend_test.rs`)
- [ ] Test same model on Candle vs Tch
- [ ] Test output consistency between backends
- [ ] Test numerical differences (tolerance check)
- [ ] Test performance comparison
- [ ] Test memory usage comparison

### Memory Tests (`tests/integration/memory_test.rs`)
- [ ] Test memory leak detection (valgrind/asan)
- [ ] Test large model loading (>1GB)
- [ ] Test multiple model loading (5+ models)
- [ ] Test memory cleanup on model unload
- [ ] Test memory pool usage
- [ ] Test peak memory tracking

### Error Propagation Tests (`tests/integration/error_propagation_test.rs`)
- [ ] Test error from model → engine → API
- [ ] Test error from backend → model → engine
- [ ] Test error from mlpkg → engine
- [ ] Test error recovery strategies
- [ ] Test graceful degradation

### Failure Modes Tests (`tests/integration/failure_modes_test.rs`)
- [ ] Test network failure during download
- [ ] Test disk space exhaustion during download
- [ ] Test disk space exhaustion during package assembly
- [ ] Test corrupt model file loading
- [ ] Test missing library dependencies
- [ ] Test incompatible backend version

---

## Phase 9: Performance & Stress Tests (Low Priority)

### Inference Benchmarks (`benches/inference_latency.rs`)
- [ ] Benchmark Candle single inference latency
- [ ] Benchmark Tch single inference latency
- [ ] Benchmark Candle batch inference (sizes: 1,4,8,16,32)
- [ ] Benchmark Tch batch inference (sizes: 1,4,8,16,32)
- [ ] Compare with baseline PyTorch
- [ ] Compare with baseline Candle (direct)

### Throughput Benchmarks (`benches/throughput.rs`)
- [ ] Benchmark Candle throughput (inferences/sec)
- [ ] Benchmark Tch throughput (inferences/sec)
- [ ] Test scaling with batch size
- [ ] Test scaling with concurrent requests

### Memory Overhead Benchmarks (`benches/memory_overhead.rs`)
- [ ] Measure type erasure overhead
- [ ] Measure engine overhead
- [ ] Compare with direct backend usage
- [ ] Measure per-model memory overhead

### Stress Tests (`tests/stress/`)
- [ ] Test 100 concurrent inference requests
- [ ] Test 1000 concurrent inference requests
- [ ] Test thread safety under stress
- [ ] Test loading 10GB+ model
- [ ] Test OOM handling
- [ ] Test 1M+ inference requests
- [ ] Test long-running process (24h)
- [ ] Test memory leak detection (long-running)

---

## Success Criteria

- [ ] Overall code coverage ≥80%
- [ ] inferox-core coverage ≥90%
- [ ] inferox-engine coverage ≥85%
- [ ] inferox-mlpkg coverage ≥80%
- [ ] inferox-candle coverage ≥75%
- [ ] inferox-tch coverage ≥75%
- [ ] hf-xet-rs coverage ≥85%
- [ ] All critical paths have tests
- [ ] All public APIs have tests
- [ ] Integration tests cover major workflows
- [ ] Error cases are tested
- [ ] CI/CD runs all tests on every PR
- [ ] No flaky tests in CI
- [ ] Performance benchmarks establish baselines

---

## Progress Summary

Total items: ~450 test cases

**Phase 1 (inferox-core)**: 0/48 (0%)
**Phase 2 (inferox-engine)**: 0/38 (0%)
**Phase 3 (inferox-mlpkg)**: 31/93 (33%)
**Phase 4 (inferox-candle)**: 0/45 (0%)
**Phase 5 (inferox-tch)**: 0/28 (0%)
**Phase 6 (hf-xet-rs)**: 87/108 (81%)
**Phase 7 (Examples)**: 8/46 (17%)
**Phase 8 (Integration)**: 0/23 (0%)
**Phase 9 (Performance)**: 0/16 (0%)

**Overall Progress**: 126/445 (28%)

---

Last Updated: 2025-10-12
