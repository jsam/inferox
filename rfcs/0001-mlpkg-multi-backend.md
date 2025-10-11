# RFC 0001: Multi-Backend Model Package System

## Status
**DRAFT** - Ready for Implementation

## Summary
Design and implement a backend-agnostic model packaging system (`inferox-mlpkg`) that enables downloading, packaging, and loading transformer models from Hugging Face into the Inferox engine. The system supports multiple backends (Candle, LibTorch, TFLite) using a unified package format.

---

## 1. Motivation

### Current State
- ✅ `hf-xet-rs`: Downloads models from HuggingFace
- ✅ `inferox-engine`: Runs models with unified API
- ✅ `inferox-candle`: Candle backend implementation
- ✅ Working MLP example (hardcoded, dylib-based)

### Problem
We need to:
1. Load transformer models from HuggingFace (BERT, GPT-2, T5, ViT, etc.)
2. Support multiple backends (Candle, LibTorch, TFLite) from same package
3. Map Candle transformer implementations to Inferox Model trait
4. Enable dynamic model loading (not compile-time dylib)

### Goals
- **Backend-agnostic packages**: One package works across all backends
- **Runtime backend selection**: Choose backend when loading, not compiling
- **Clean API**: `download() → load() → infer()` in <10 lines
- **Extensible**: Easy to add new models and backends

---

## 2. Architecture

### 2.1 Package Structure

```
model-package/
├── metadata.json              # Package metadata
├── model_info.json           # Backend-agnostic model description
│
├── backends/                 # Backend-specific artifacts
│   ├── candle/
│   │   ├── config.json       # Model architecture config
│   │   ├── model.safetensors # Weights
│   │   └── libmodel.so       # Compiled model architecture
│   │
│   ├── torch/                # Future
│   │   ├── config.json
│   │   ├── model.pt
│   │   └── libmodel.so       # Compiled model architecture
│   │
│   └── tflite/               # Future
│       ├── model.tflite
│       └── libmodel.so       # Compiled model architecture
│
├── tokenizer.json            # Shared (optional)
└── preprocessor_config.json  # Shared (optional)
```

**Key Insights:**
- Package can contain artifacts for multiple backends; loader selects appropriate one at runtime
- Each backend includes a **compiled static library** (`.so`/`.dylib`) containing the model architecture implementation
- Static libraries can implement **ANY neural network architecture** (transformers, CNNs, RNNs, GANs, custom architectures, etc.)
- Static libraries export `create_model()` function conforming to the Inferox Model trait interface

### 2.2 Component Architecture

```
┌─────────────────────────────────────┐
│      User Application                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   ModelPackageManager                │
│   (Orchestrates loading)             │
└────────────┬────────────────────────┘
             │
      ┌──────┴──────┬─────────────┐
      │             │             │
┌─────▼──────┐ ┌────▼─────┐ ┌────▼──────┐
│CandleLoader│ │TorchLoader│ │TFLiteLdr │
└─────┬──────┘ └────┬─────┘ └────┬──────┘
      │             │             │
      │  Loads .so  │  Loads .so  │  Loads .so
      │             │             │
┌─────▼──────┐ ┌────▼─────┐ ┌────▼──────┐
│libmodel.so │ │libmodel.so│ │libmodel.so│
│(ANY arch)  │ │(ANY arch) │ │(ANY arch) │
└─────┬──────┘ └────┬─────┘ └────┬──────┘
      │             │             │
      └─────────┬───┴─────────────┘
                │
        Exports create_model()
                │
┌───────────────▼─────────────────────┐
│  Box<dyn Model<Backend=B, ...>>     │
│  (Conforms to Inferox trait)        │
└─────────────────────────────────────┘
```

---

## 3. Core Types

### 3.1 Backend-Agnostic Model Metadata

```rust
/// Backend-agnostic model description
#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model type (e.g., "bert", "gpt2")
    pub model_type: String,
    
    /// HuggingFace repo ID
    pub repo_id: String,
    
    /// Architecture family
    pub architecture_family: ArchitectureFamily,
    
    /// Supported backends
    pub supported_backends: Vec<BackendType>,
    
    /// Input/output specifications (backend-agnostic)
    pub input_spec: TensorSpec,
    pub output_spec: TensorSpec,
    
    /// Model size
    pub num_parameters: usize,
    pub memory_mb: usize,
}

#[derive(Serialize, Deserialize)]
pub enum BackendType {
    Candle,
    Torch,
    TFLite,
}

#[derive(Serialize, Deserialize)]
pub enum ArchitectureFamily {
    EncoderOnly,      // BERT, RoBERTa
    DecoderOnly,      // GPT-2, GPT-J
    EncoderDecoder,   // T5, BART
    Vision,           // ViT, ResNet
    Multimodal,       // CLIP, BLIP
}

#[derive(Serialize, Deserialize)]
pub struct TensorSpec {
    pub dtype: String,           // "float32", "float16"
    pub shape: Vec<DimSpec>,     // [Batch, SeqLen, Hidden]
}

#[derive(Serialize, Deserialize)]
pub enum DimSpec {
    Fixed(usize),
    Dynamic(String),  // "batch", "seq_len"
}
```

### 3.2 Backend Loader Trait

```rust
/// Trait for loading models for a specific backend
pub trait BackendLoader: Send + Sync {
    type Backend: inferox_core::Backend;
    
    /// Backend identifier
    fn backend_type(&self) -> BackendType;
    
    /// Load model from package by dynamically loading the static library
    fn load_model(
        &self,
        package: &ModelPackage,
        backend: &Self::Backend,
    ) -> Result<Box<dyn Model<
        Backend = Self::Backend,
        Input = <Self::Backend as Backend>::Tensor,
        Output = <Self::Backend as Backend>::Tensor
    >>>;
    
    /// Check if can load this model
    fn can_load(&self, model_info: &ModelInfo) -> bool;
}
```

### 3.3 Static Library Interface

Each model's static library must export a factory function:

```rust
/// Function signature for model factory exported by static libraries
pub type ModelFactory<B> = unsafe extern "C" fn() -> Box<dyn Model<
    Backend = B,
    Input = <B as Backend>::Tensor,
    Output = <B as Backend>::Tensor
>>;
```

Example implementation in model library:

```rust
#[no_mangle]
pub extern "C" fn create_model() -> Box<dyn Model<
    Backend = CandleBackend,
    Input = CandleTensor,
    Output = CandleTensor
>> {
    let builder = CandleModelBuilder::new(Device::Cpu);
    let model = BertModel::new("bert", builder.var_builder())
        .expect("Failed to create BERT model");
    Box::new(model)
}
```

### 3.4 Package Manager

```rust
pub struct ModelPackageManager {
    cache_dir: PathBuf,
    hf_client: HfXetClient,
    loaders: HashMap<BackendType, Box<dyn BackendLoaderDyn>>,
}

impl ModelPackageManager {
    pub fn new(cache_dir: PathBuf) -> Result<Self>;
    
    /// Register a backend loader
    pub fn register_loader<L: BackendLoader + 'static>(
        &mut self, 
        loader: L
    );
    
    /// Download model weights/config and create package structure
    pub async fn download_and_package(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<ModelPackage>;
    
    /// Load model for specific backend by loading static library
    pub fn load_model<B: Backend>(
        &self,
        package: &ModelPackage,
        backend: &B,
    ) -> Result<BoxedModel<B>>;
}
```

---

## 4. Candle Backend Implementation (Phase 1)

### 4.1 Candle Loader

The Candle loader dynamically loads the compiled static library and calls its `create_model()` function:

```rust
use libloading::{Library, Symbol};

pub struct CandleModelLoader;

impl BackendLoader for CandleModelLoader {
    type Backend = CandleBackend;
    
    fn backend_type(&self) -> BackendType {
        BackendType::Candle
    }
    
    fn load_model(
        &self,
        package: &ModelPackage,
        _backend: &CandleBackend,
    ) -> Result<BoxedModel<CandleBackend>> {
        // 1. Locate the static library
        let lib_path = package.path()
            .join("backends/candle/libmodel.so");
        
        if !lib_path.exists() {
            return Err(Error::LibraryNotFound(lib_path));
        }
        
        // 2. Load the library
        unsafe {
            let lib = Library::new(&lib_path)
                .map_err(|e| Error::LibraryLoad(e.to_string()))?;
            
            // 3. Get the create_model symbol
            let factory: Symbol<ModelFactory<CandleBackend>> = lib
                .get(b"create_model")
                .map_err(|e| Error::SymbolNotFound(e.to_string()))?;
            
            // 4. Call factory to create model
            let model = factory();
            
            // 5. Keep library alive for model's lifetime
            std::mem::forget(lib);
            
            Ok(model)
        }
    }
    
    fn can_load(&self, model_info: &ModelInfo) -> bool {
        model_info.supported_backends.contains(&BackendType::Candle)
    }
}
```

**Key Points:**
- Uses `libloading` crate for dynamic library loading
- Calls the `create_model()` function exported by the static library
- The static library contains the compiled model architecture implementation
- Weights are loaded inside the `create_model()` function from the package

### 4.2 Model Static Library Structure

Each model architecture is compiled as a separate static library crate. The architecture can be **anything** - not limited to transformers:

```
models/
├── bert-candle/
│   ├── Cargo.toml          # crate-type = ["cdylib"]
│   └── src/
│       └── lib.rs          # Implements BertModel + exports create_model()
├── gpt2-candle/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs          # GPT-2 architecture
├── resnet-candle/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs          # ResNet CNN architecture
├── lstm-candle/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs          # LSTM/RNN architecture
├── gan-candle/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs          # GAN architecture
└── custom-candle/
    ├── Cargo.toml
    └── src/
        └── lib.rs          # Any custom architecture
```

**Universal Architecture Support**: The system is architecture-agnostic - as long as the static library exports `create_model()` and conforms to the `Model` trait, it will work.

### 4.3 BERT Static Library Example

`examples/bert-candle/Cargo.toml`:

```toml
[package]
name = "bert-candle"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
inferox-core = { path = "../../crates/inferox-core" }
inferox-candle = { path = "../../crates/inferox-candle" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
candle-transformers = { git = "https://github.com/huggingface/candle" }
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

`examples/bert-candle/src/lib.rs`:

```rust
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use inferox_candle::{CandleBackend, CandleTensor};
use inferox_core::Model;
use std::path::Path;

pub struct BertModelWrapper {
    name: String,
    inner: BertModel,
}

impl Model for BertModelWrapper {
    type Backend = CandleBackend;
    type Input = CandleTensor;
    type Output = CandleTensor;
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn forward(
        &self,
        input: Self::Input,
    ) -> Result<Self::Output, <Self::Backend as inferox_core::Backend>::Error> {
        let tensor = input.into_inner();
        let output = self.inner.forward(&tensor)?;
        Ok(CandleTensor::new(output))
    }
}

#[no_mangle]
pub extern "C" fn create_model() -> Box<dyn Model<
    Backend = CandleBackend,
    Input = CandleTensor,
    Output = CandleTensor,
>> {
    // This function is called by the loader after weights are in place
    // The package directory is available via environment variable
    let package_dir = std::env::var("INFEROX_PACKAGE_DIR")
        .expect("INFEROX_PACKAGE_DIR not set");
    
    let config_path = Path::new(&package_dir)
        .join("backends/candle/config.json");
    let weights_path = Path::new(&package_dir)
        .join("backends/candle/model.safetensors");
    
    // Load config
    let config: BertConfig = serde_json::from_reader(
        std::fs::File::open(config_path).expect("Failed to open config")
    ).expect("Failed to parse config");
    
    // Load weights
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path],
            DType::F32,
            &device,
        ).expect("Failed to load weights")
    };
    
    // Create model
    let bert = BertModel::load(vb, &config)
        .expect("Failed to create BERT model");
    
    Box::new(BertModelWrapper {
        name: "bert".to_string(),
        inner: bert,
    })
}
```

---

## 5. Package Download & Organization

The package manager downloads weights and config from HuggingFace, then expects the user to compile and place the appropriate static library:

```rust
impl ModelPackageManager {
    pub async fn download_and_package(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        backends: &[BackendType],
    ) -> Result<ModelPackage> {
        // 1. Download from HF
        let snapshot = self.hf_client.snapshot_download(
            repo_id,
            revision,
            None,
            None,
        ).await?;
        
        // 2. Parse HF config
        let config_path = snapshot.join("config.json");
        let hf_config: HFConfig = serde_json::from_reader(
            File::open(&config_path)?
        )?;
        
        // 3. Detect model type and create ModelInfo
        let model_info = ModelInfo {
            model_type: hf_config.model_type.clone(),
            repo_id: repo_id.to_string(),
            architecture_family: detect_architecture(&hf_config),
            supported_backends: backends.to_vec(),
            input_spec: parse_input_spec(&hf_config),
            output_spec: parse_output_spec(&hf_config),
            num_parameters: calculate_params(&hf_config),
            memory_mb: estimate_memory(&hf_config),
        };
        
        // 4. Organize into package structure
        let package_dir = self.cache_dir
            .join("packages")
            .join(sanitize_repo_id(repo_id));
        
        self.organize_package(
            &snapshot,
            &package_dir,
            &model_info,
            backends,
        )?;
        
        // 5. Create package
        Ok(ModelPackage {
            path: package_dir,
            model_info,
            metadata: PackageMetadata {
                repo_id: repo_id.to_string(),
                revision: revision.map(String::from),
                created_at: Utc::now(),
            },
        })
    }
    
    fn organize_package(
        &self,
        snapshot: &Path,
        package_dir: &Path,
        model_info: &ModelInfo,
        backends: &[BackendType],
    ) -> Result<()> {
        fs::create_dir_all(package_dir)?;
        
        // Write model_info.json
        let info_json = serde_json::to_string_pretty(&model_info)?;
        fs::write(
            package_dir.join("model_info.json"),
            info_json
        )?;
        
        // Copy shared files
        for file in &["tokenizer.json", "preprocessor_config.json"] {
            let src = snapshot.join(file);
            if src.exists() {
                fs::copy(&src, package_dir.join(file))?;
            }
        }
        
        // Organize each backend
        for backend in backends {
            match backend {
                BackendType::Candle => {
                    let candle_dir = package_dir.join("backends/candle");
                    fs::create_dir_all(&candle_dir)?;
                    
                    // Copy config
                    fs::copy(
                        snapshot.join("config.json"),
                        candle_dir.join("config.json")
                    )?;
                    
                    // Copy safetensors files
                    for entry in fs::read_dir(snapshot)? {
                        let entry = entry?;
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                            fs::copy(
                                &path,
                                candle_dir.join(entry.file_name())
                            )?;
                        }
                    }
                    
                    // NOTE: User must compile and place libmodel.so here
                    println!("Package created at: {:?}", package_dir);
                    println!("Compile model library and place at: {:?}", 
                        candle_dir.join("libmodel.so"));
                }
                BackendType::Torch => {
                    // TODO: Implement torch backend packaging
                    unimplemented!("Torch backend not yet supported")
                }
                BackendType::TFLite => {
                    // TODO: Implement tflite backend packaging
                    unimplemented!("TFLite backend not yet supported")
                }
            }
        }
        
        Ok(())
    }
    
    /// Helper to install a compiled model library into a package
    pub fn install_model_library(
        &self,
        package: &ModelPackage,
        backend: BackendType,
        library_path: &Path,
    ) -> Result<()> {
        let backend_dir = package.path().join(format!("backends/{}", backend.as_str()));
        let target_path = backend_dir.join(library_name());
        
        fs::copy(library_path, &target_path)?;
        println!("Installed library at: {:?}", target_path);
        
        Ok(())
    }
}

fn library_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libmodel.dylib"
    } else if cfg!(target_os = "windows") {
        "model.dll"
    } else {
        "libmodel.so"
    }
}
```

**Two-Step Workflow:**
1. **Download**: `download_and_package()` creates package structure with weights/config
2. **Install Library**: User compiles model architecture and calls `install_model_library()` to place the `.so` file

---

## 6. Usage Examples

### 6.1 Complete Workflow

```rust
use inferox_mlpkg::{ModelPackageManager, BackendType};
use inferox_candle::CandleBackend;
use inferox_engine::{InferoxEngine, EngineConfig};
use candle_core::Device;
use std::path::PathBuf;

// Step 1: Create package manager
let mut pm = ModelPackageManager::new(cache_dir)?;
pm.register_loader(CandleModelLoader::new());

// Step 2: Download model (weights + config)
let package = pm.download_and_package(
    "bert-base-uncased",
    None,
    &[BackendType::Candle]
).await?;

// Step 3: Compile model architecture library
// (User runs this externally)
// $ cd examples/bert-candle
// $ cargo build --release
// $ cp target/release/libbert_candle.so <package_dir>/backends/candle/libmodel.so

// Step 4: Install compiled library into package
pm.install_model_library(
    &package,
    BackendType::Candle,
    &PathBuf::from("target/release/libbert_candle.so"),
)?;

// Step 5: Load model (loads static library)
let backend = CandleBackend::new(Device::Cpu)?;
let model = pm.load_model(&package, &backend)?;

// Step 6: Use with engine
let mut engine = InferoxEngine::new(backend, EngineConfig::default());
engine.register_boxed_model(model);

// Step 7: Run inference
let input = prepare_bert_input("Hello world");
let output = engine.infer("bert-base-uncased", input)?;
```

### 6.2 Multi-Backend (Future)

```rust
// Download once with multiple backend support
let package = pm.download_and_package(
    "bert-base-uncased",
    None,
    &[BackendType::Candle, BackendType::Torch]
).await?;

// Compile and install libraries for each backend
pm.install_model_library(
    &package,
    BackendType::Candle,
    &PathBuf::from("target/release/libbert_candle.so"),
)?;
pm.install_model_library(
    &package,
    BackendType::Torch,
    &PathBuf::from("target/release/libbert_torch.so"),
)?;

// Load with Candle
let candle_backend = CandleBackend::new(Device::Cpu)?;
let candle_model = pm.load_model(&package, &candle_backend)?;

// Load with Torch (when implemented)
let torch_backend = TorchBackend::new(tch::Device::Cpu)?;
let torch_model = pm.load_model(&package, &torch_backend)?;
```

---

## 7. Directory Structure

```
inferox/
├── crates/
│   └── inferox-mlpkg/
│       ├── src/
│       │   ├── lib.rs
│       │   │
│       │   ├── package/
│       │   │   ├── mod.rs
│       │   │   ├── manager.rs          # ModelPackageManager
│       │   │   ├── metadata.rs         # ModelInfo, PackageMetadata
│       │   │   └── loader.rs           # BackendLoader trait
│       │   │
│       │   ├── backends/
│       │   │   ├── mod.rs
│       │   │   └── candle/
│       │   │       ├── mod.rs
│       │   │       └── loader.rs       # CandleModelLoader (libloading)
│       │   │
│       │   ├── config/
│       │   │   ├── mod.rs
│       │   │   └── parser.rs           # HF config parsing
│       │   │
│       │   └── error.rs
│       │
│       └── tests/
│           └── integration/
│               ├── bert_end_to_end.rs
│               └── multi_model.rs
│
└── models/
    ├── bert-candle/
    │   ├── Cargo.toml              # crate-type = ["cdylib"]
    │   └── src/
    │       └── lib.rs              # BertModel wrapper + create_model()
    │
    ├── gpt2-candle/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs              # GPT2Model wrapper + create_model()
    │
    ├── t5-candle/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs              # T5Model wrapper + create_model()
    │
    └── vit-candle/
        ├── Cargo.toml
        └── src/
            └── lib.rs              # ViTModel wrapper + create_model()
```

**Key Changes from Original Design:**
- **Removed:** `CandleModelRegistry`, `CandleModelBuilder`, `CandleModelAdapter` - no longer needed
- **Simplified:** Each model is now a standalone static library crate
- **Cleaner:** `CandleModelLoader` just loads `.so` files, doesn't build models

---

## 8. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Core Package System:**
- [ ] Define core types: `ModelInfo`, `PackageMetadata`, `ModelPackage`
- [ ] Implement `BackendLoader` trait
- [ ] Implement `ModelPackageManager` with download logic
- [ ] Implement `install_model_library()` helper
- [ ] Add HF config parsing and model type detection

**Candle Backend Loader:**
- [ ] Implement `CandleModelLoader` using `libloading`
- [ ] Handle library loading and symbol resolution
- [ ] Add proper error handling for missing libraries

### Phase 2: BERT Model Library (Week 1-2)

**BERT Static Library:**
- [ ] Create `examples/bert-candle/` crate with `cdylib` type
- [ ] Implement `BertModelWrapper` conforming to `Model` trait
- [ ] Implement `create_model()` function that:
  - Reads `INFEROX_PACKAGE_DIR` environment variable
  - Loads config from `backends/candle/config.json`
  - Loads weights from `backends/candle/model.safetensors`
  - Returns `Box<dyn Model>`
- [ ] Add error handling for missing files/invalid configs

**End-to-End Test:**
- [ ] Download BERT from HuggingFace
- [ ] Compile BERT static library
- [ ] Install library into package
- [ ] Load model via `CandleModelLoader`
- [ ] Run inference and validate output

### Phase 3: Additional Model Architectures (Week 2)

**Transformer Models:**
- [ ] GPT-2 static library (`models/gpt2-candle/`)
- [ ] T5 static library (`models/t5-candle/`)

**Vision Models:**
- [ ] ResNet CNN static library (`models/resnet-candle/`)
- [ ] ViT static library (`models/vit-candle/`)

**Other Architectures:**
- [ ] LSTM/RNN static library (optional)
- [ ] Integration tests for each model type

**Architecture Coverage**: Demonstrate the system works with diverse architectures (transformers, CNNs, RNNs) not just transformers

### Phase 4: Second Backend (Week 3+)

**Torch Backend:**
- [ ] Create `TorchModelLoader`
- [ ] Implement BERT-torch static library
- [ ] Handle weight format differences
- [ ] Multi-backend validation tests

---

## 9. Testing Strategy

### Unit Tests
```rust
#[test]
fn test_config_parsing() {
    let hf_config = load_test_config("bert-base-uncased");
    let model_info = parse_model_info(&hf_config);
    assert_eq!(model_info.model_type, "bert");
}

#[test]
fn test_package_metadata() {
    let metadata = PackageMetadata {
        repo_id: "bert-base-uncased".to_string(),
        revision: Some("main".to_string()),
        created_at: Utc::now(),
    };
    let json = serde_json::to_string(&metadata).unwrap();
    let deserialized: PackageMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.repo_id, metadata.repo_id);
}

#[test]
fn test_library_name() {
    #[cfg(target_os = "macos")]
    assert_eq!(library_name(), "libmodel.dylib");
    #[cfg(target_os = "linux")]
    assert_eq!(library_name(), "libmodel.so");
    #[cfg(target_os = "windows")]
    assert_eq!(library_name(), "model.dll");
}
```

### Integration Tests
```rust
#[tokio::test]
#[ignore]
async fn test_bert_end_to_end() {
    // Step 1: Download and package
    let pm = setup_package_manager();
    let package = pm.download_and_package(
        "bert-base-uncased",
        None,
        &[BackendType::Candle]
    ).await?;
    
    // Step 2: Compile model library
    let status = Command::new("cargo")
        .args(&["build", "--release", "-p", "bert-candle"])
        .status()?;
    assert!(status.success());
    
    // Step 3: Install library
    pm.install_model_library(
        &package,
        BackendType::Candle,
        &PathBuf::from("target/release/libbert_candle.so"),
    )?;
    
    // Step 4: Load model
    let backend = CandleBackend::new(Device::Cpu)?;
    let model = pm.load_model(&package, &backend)?;
    
    // Step 5: Test inference
    let input = create_dummy_input();
    let output = model.forward(input)?;
    
    assert_eq!(output.shape(), &[1, 768]); // [batch, hidden_size]
}
```

---

## 10. Success Criteria

**Phase 1 (Core + BERT):**
- [ ] Download BERT from HuggingFace
- [ ] Compile BERT static library
- [ ] Load BERT via dynamic library loading
- [ ] Run inference and get correct output shape
- [ ] End-to-end test passes

**Phase 2 (Multi-Architecture):**
- [ ] Support diverse architectures: transformers (BERT, GPT-2), CNNs (ResNet), RNNs (LSTM), custom models
- [ ] All architectures loadable via same package system
- [ ] Integration test for each architecture family
- [ ] Prove architecture-agnostic design works

**Quality Metrics:**
- [ ] Cold start <5 seconds (download → load → infer)
- [ ] Memory usage <2x model size
- [ ] 80%+ test coverage
- [ ] Clean API (<10 lines for basic workflow)
- [ ] Comprehensive documentation

**Architecture Validation:**
- [ ] Static libraries conform to Model trait interface
- [ ] `libloading` successfully loads compiled models
- [ ] Environment variable approach works for package path
- [ ] Multi-backend extensibility proven (Candle → Torch)

---

## 11. Future Enhancements

### Phase 4+
- Multi-backend support (Torch, TFLite)
- Weight format conversion
- Quantization support
- Model optimization (graph fusion, etc.)
- Streaming model loading
- Custom/fine-tuned model support

---

## 12. Dependencies

### inferox-mlpkg (Package Manager)

```toml
[dependencies]
hf-xet-rs = { path = "../hf-xet-rs" }
inferox-core = { path = "../inferox-core" }
libloading = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
```

### examples/bert-candle (Model Library)

```toml
[dependencies]
inferox-core = { path = "../../crates/inferox-core" }
inferox-candle = { path = "../../crates/inferox-candle" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
candle-transformers = { git = "https://github.com/huggingface/candle" }
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

---

## Conclusion

This RFC proposes a modular, backend-agnostic model packaging system that follows Inferox's **two-pillar architecture**:

### Key Design Principles

1. **Static Library Architecture**: Each model is compiled as a separate `.so`/`.dylib` that exports a `create_model()` function
2. **Trait Interface**: All models conform to the `inferox_core::Model` trait, ensuring type safety and uniform API
3. **Multi-Backend Support**: Single package format contains artifacts for multiple backends (Candle, LibTorch, TFLite)
4. **Dynamic Loading**: Models are loaded at runtime via `libloading`, not compiled into the engine
5. **Clean Separation**: Package manager handles downloads, loaders handle library loading, static libraries handle model construction

### Architecture Benefits

- ✅ **Type Safety**: Rust trait bounds enforced at compile time
- ✅ **Memory Safety**: RAII and ownership prevent leaks
- ✅ **Modularity**: Add new models without recompiling engine
- ✅ **Backend Flexibility**: Same package works with different backends
- ✅ **No FFI Complexity**: Pure Rust interfaces, no C headers

### Implementation Path

The phased approach allows us to validate the architecture with Candle + BERT, then scale to:
- **Any neural network architecture**: Transformers, CNNs, RNNs, GANs, custom architectures
- Multiple backends (Torch, TFLite)
- Production features (quantization, optimization)

### Universal Architecture Support

The system is **architecture-agnostic** - it works with any model that:
1. Can be implemented in Rust
2. Conforms to the `Model` trait interface
3. Can be compiled as a static library

This includes but is not limited to:
- 🤖 **Transformers**: BERT, GPT, T5, ViT, CLIP, LLaMA
- 🖼️ **Vision**: ResNet, EfficientNet, YOLO, Stable Diffusion
- 📝 **Sequence**: LSTM, GRU, Seq2Seq
- 🎨 **Generative**: GANs, VAEs, Diffusion Models
- 🔧 **Custom**: Any proprietary or research architecture

**Status:** Ready for Phase 1 implementation.
