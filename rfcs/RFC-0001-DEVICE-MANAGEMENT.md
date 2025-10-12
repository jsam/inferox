# RFC-0001: Device Management and Model Hot-Reload

**Status:** Draft  
**Created:** 2025-10-12  
**Author:** Claude Code

## Summary

This RFC proposes a comprehensive device management system for Inferox that allows:
1. Models to specify their target device at load time (from model.toml)
2. Runtime device management operations (hot-reload to different devices)
3. Clear separation between "happy path" inference (no reload) and management operations (reload allowed)

## Motivation

### Current State Analysis

**Device Handling Today:**
- **Backend Level**: Both `CandleBackend` and `TchBackend` have device information
  - `CandleBackend::cpu()`, `CandleBackend::with_device(device)`
  - `TchBackend::cpu()`, `TchBackend::cuda(ordinal)`, `TchBackend::with_device(device)`
- **Model.toml**: Device specified in package metadata (`device = "cpu"`)
- **PackageManager**: Parses device from model.toml and creates backend with correct device (inferox-mlpkg/src/lib.rs:682-706)
- **Router**: Hardcodes `Backend::cpu()` in inference hotpath (examples/bert-multi-backend/src/router.rs:119, 146)

**Problems:**
1. **Inference hotpath creates backends**: Router creates new backend instances on every request
2. **Device ignored during routing**: Model.toml device is used at load time but ignored during inference
3. **No device migration**: Cannot move model from CPU → GPU or GPU:0 → GPU:1 without process restart
4. **Mixed responsibilities**: Device management split between Backend, Model, PackageManager, and Router

### Use Cases

**UC1: Startup with Device Specification**
```
Process starts → Load bert.mlpkg (device="cuda:0") → Engine registers model on GPU:0 → Inference uses GPU:0
```

**UC2: Management Operation - Device Migration**
```
Model running on CPU → Admin API: move_model("bert", "cuda:0") → Engine unloads CPU model → Reloads on GPU:0 → Inference continues on GPU:0
```

**UC3: Multi-Model with Different Devices**
```
bert-cpu.mlpkg (device="cpu") → GPU:0
bert-gpu.mlpkg (device="cuda:0") → GPU:0
gpt-gpu.mlpkg (device="cuda:1") → GPU:1
```

## Detailed Design

### 1. Core Device Abstraction (inferox-core)

**Current State:**
```rust
// inferox-core/src/device.rs
pub enum DeviceId {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Custom(String),
}

pub trait Device: Clone + Send + Sync {
    fn id(&self) -> DeviceId;
    fn is_available(&self) -> bool;
    fn memory_info(&self) -> Option<MemoryInfo>;
}
```

**Proposed Enhancement:**
```rust
// inferox-core/src/device.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceId {
    Cpu,
    Cuda(usize),      // CUDA GPU with ordinal
    Metal(usize),     // Metal GPU with ordinal
    Custom(String),
}

impl DeviceId {
    /// Parse device string from model.toml format
    /// Examples: "cpu", "cuda", "cuda:0", "cuda:1", "mps", "metal:0"
    pub fn parse(s: &str) -> Result<Self, DeviceError> {
        match s {
            "cpu" => Ok(DeviceId::Cpu),
            "cuda" => Ok(DeviceId::Cuda(0)),
            "mps" | "metal" => Ok(DeviceId::Metal(0)),
            s if s.starts_with("cuda:") => {
                let idx = s.strip_prefix("cuda:").unwrap()
                    .parse::<usize>()
                    .map_err(|_| DeviceError::InvalidFormat(s.to_string()))?;
                Ok(DeviceId::Cuda(idx))
            }
            s if s.starts_with("metal:") => {
                let idx = s.strip_prefix("metal:").unwrap()
                    .parse::<usize>()
                    .map_err(|_| DeviceError::InvalidFormat(s.to_string()))?;
                Ok(DeviceId::Metal(idx))
            }
            _ => Ok(DeviceId::Custom(s.to_string())),
        }
    }
    
    /// Convert to string format (for serialization)
    pub fn to_string(&self) -> String {
        match self {
            DeviceId::Cpu => "cpu".to_string(),
            DeviceId::Cuda(idx) => format!("cuda:{}", idx),
            DeviceId::Metal(idx) => format!("metal:{}", idx),
            DeviceId::Custom(s) => s.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Device not available: {0}")]
    NotAvailable(String),
    
    #[error("Invalid device format: {0}")]
    InvalidFormat(String),
    
    #[error("Device migration failed: {0}")]
    MigrationFailed(String),
}

pub trait Device: Clone + Send + Sync {
    fn id(&self) -> DeviceId;
    fn is_available(&self) -> bool;
    fn memory_info(&self) -> Option<MemoryInfo>;
}
```

### 2. Model Device Awareness (inferox-core)

**Current State:**
```rust
pub trait Model: Send + Sync {
    type Backend: Backend;
    type Input;
    type Output;

    fn name(&self) -> &str;
    fn forward(&self, input: Self::Input) -> Result<Self::Output, <Self::Backend as Backend>::Error>;
    fn metadata(&self) -> ModelMetadata { ModelMetadata::default() }
    fn memory_requirements(&self) -> MemoryRequirements { MemoryRequirements::default() }
}
```

**Proposed Enhancement:**
```rust
pub trait Model: Send + Sync {
    type Backend: Backend;
    type Input;
    type Output;

    fn name(&self) -> &str;
    fn forward(&self, input: Self::Input) -> Result<Self::Output, <Self::Backend as Backend>::Error>;
    fn metadata(&self) -> ModelMetadata { ModelMetadata::default() }
    fn memory_requirements(&self) -> MemoryRequirements { MemoryRequirements::default() }
    
    /// Get the device this model is currently loaded on
    fn device(&self) -> <Self::Backend as Backend>::Device;
}

/// Extended metadata to include device information
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub tags: Vec<String>,
    pub device: Option<DeviceId>,  // NEW: Track which device model is on
    pub custom: HashMap<String, String>,
}
```

### 3. Engine Device Management (inferox-engine)

**Current State:**
```rust
pub struct InferoxEngine {
    models: HashMap<String, Box<dyn AnyModel>>,
    _config: EngineConfig,
}

impl InferoxEngine {
    pub fn register_model<B: Backend + 'static>(&mut self, model: BoxedModel<B>);
    pub fn infer<B: Backend + 'static>(&self, model_name: &str, input: B::Tensor) -> Result<B::Tensor>;
    pub fn list_models(&self) -> Vec<(&str, ModelMetadata)>;
}
```

**Proposed Enhancement:**
```rust
use std::sync::RwLock;

/// Model entry tracking device and backend info
struct ModelEntry {
    model: Box<dyn AnyModel>,
    device: DeviceId,
    backend_type: BackendType,
}

pub struct InferoxEngine {
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,  // RwLock for hot-reload
    _config: EngineConfig,
}

impl InferoxEngine {
    /// Register model with automatic device detection from metadata
    pub fn register_model<B: Backend + 'static>(&mut self, model: BoxedModel<B>)
    where
        B::Tensor: 'static,
    {
        let name = model.name().to_string();
        let device = model.device().id();  // Get device from model
        let backend_type = infer_backend_type::<B>();
        
        let erased = TypeErasedModel::new(model);
        let entry = ModelEntry {
            model: Box::new(erased),
            device,
            backend_type,
        };
        
        self.models.write().unwrap().insert(name, entry);
    }
    
    /// Get device for a specific model
    pub fn model_device(&self, model_name: &str) -> Option<DeviceId> {
        self.models.read().unwrap()
            .get(model_name)
            .map(|entry| entry.device.clone())
    }
    
    /// Hot-reload model to different device (MANAGEMENT OPERATION)
    /// 
    /// This is NOT called on the inference hotpath. Only for admin operations.
    /// 
    /// Steps:
    /// 1. Validate new device is available
    /// 2. Load model on new device
    /// 3. Atomically swap old model with new model
    /// 4. Drop old model (frees old device memory)
    pub fn reload_model_to_device(
        &mut self,
        model_name: &str,
        new_device: DeviceId,
        loader: impl FnOnce(DeviceId) -> Result<Box<dyn AnyModel>, Box<dyn std::error::Error>>
    ) -> Result<(), EngineError> {
        // 1. Validate device
        if !self.is_device_available(&new_device) {
            return Err(EngineError::DeviceNotAvailable(new_device));
        }
        
        // 2. Load new model on target device
        let new_model = loader(new_device.clone())
            .map_err(|e| EngineError::ModelLoadFailed(e.to_string()))?;
        
        let backend_type = self.models.read().unwrap()
            .get(model_name)
            .map(|e| e.backend_type)
            .ok_or_else(|| EngineError::ModelNotFound(model_name.to_string()))?;
        
        let new_entry = ModelEntry {
            model: new_model,
            device: new_device.clone(),
            backend_type,
        };
        
        // 3. Atomic swap (write lock held briefly)
        let mut models = self.models.write().unwrap();
        let old_entry = models.insert(model_name.to_string(), new_entry);
        drop(models);  // Release lock immediately
        
        // 4. Old model dropped here, freeing old device memory
        drop(old_entry);
        
        Ok(())
    }
    
    /// Remove model from engine (frees device memory)
    pub fn unload_model(&mut self, model_name: &str) -> Result<(), EngineError> {
        self.models.write().unwrap()
            .remove(model_name)
            .ok_or_else(|| EngineError::ModelNotFound(model_name.to_string()))?;
        Ok(())
    }
    
    /// List models with device information
    pub fn list_models_with_devices(&self) -> Vec<(String, ModelMetadata, DeviceId)> {
        self.models.read().unwrap()
            .iter()
            .map(|(name, entry)| {
                let mut metadata = entry.model.metadata();
                metadata.device = Some(entry.device.clone());
                (name.clone(), metadata, entry.device.clone())
            })
            .collect()
    }
    
    fn is_device_available(&self, device: &DeviceId) -> bool {
        // Check device availability based on backend capabilities
        match device {
            DeviceId::Cpu => true,
            DeviceId::Cuda(idx) => {
                #[cfg(feature = "tch")]
                {
                    tch::Cuda::is_available() && (*idx as i64) < tch::Cuda::device_count()
                }
                #[cfg(not(feature = "tch"))]
                false
            }
            DeviceId::Metal(_) => {
                #[cfg(target_os = "macos")]
                true
                #[cfg(not(target_os = "macos"))]
                false
            }
            DeviceId::Custom(_) => false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Device not available: {0:?}")]
    DeviceNotAvailable(DeviceId),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Backend error: {0}")]
    Backend(String),
}
```

### 4. Router Device Awareness (bert-multi-backend example)

**Current Problem:**
```rust
// WRONG: Creates new backend on every inference!
fn route_to_candle(&self, model_name: &str, input_ids: Vec<i64>) -> Result<...> {
    let backend = CandleBackend::cpu()  // ❌ Hardcoded CPU!
        .map_err(|e| RouterError::Engine(...))?;
    
    let input_tensor = backend.tensor_builder()
        .build_from_vec(input_ids.clone(), &[1, input_ids.len()])?;
    
    self.engine.infer::<CandleBackend>(model_name, input_tensor)?;
}
```

**Proposed Solution:**
```rust
pub struct InferenceRouter {
    engine: Arc<InferoxEngine>,
    routes: HashMap<String, ModelRoute>,
    backend_cache: HashMap<(BackendType, DeviceId), Arc<dyn Any + Send + Sync>>,  // Cache backends
}

impl InferenceRouter {
    /// Get or create backend for specific device
    fn get_backend_candle(&self, device: &DeviceId) -> Result<Arc<CandleBackend>, RouterError> {
        let cache_key = (BackendType::Candle, device.clone());
        
        // Try to get from cache
        if let Some(backend) = self.backend_cache.get(&cache_key) {
            return Ok(backend.clone().downcast::<CandleBackend>().unwrap());
        }
        
        // Create new backend for this device
        let candle_device = match device {
            DeviceId::Cpu => candle_core::Device::Cpu,
            DeviceId::Cuda(idx) => candle_core::Device::new_cuda(*idx)
                .map_err(|e| RouterError::Engine(format!("CUDA device error: {}", e)))?,
            DeviceId::Metal(idx) => candle_core::Device::new_metal(*idx)
                .map_err(|e| RouterError::Engine(format!("Metal device error: {}", e)))?,
            DeviceId::Custom(s) => return Err(RouterError::Engine(format!("Unsupported device: {}", s))),
        };
        
        let backend = Arc::new(CandleBackend::with_device(candle_device));
        self.backend_cache.insert(cache_key, backend.clone() as Arc<dyn Any + Send + Sync>);
        
        Ok(backend)
    }
    
    fn route_to_candle(&self, model_name: &str, input_ids: Vec<i64>) -> Result<UnifiedTensorResponse, RouterError> {
        let start = Instant::now();
        
        // 1. Get model's device from engine
        let device = self.engine.model_device(model_name)
            .ok_or_else(|| RouterError::Engine(format!("Model not found: {}", model_name)))?;
        
        // 2. Get backend for that specific device (cached)
        let backend = self.get_backend_candle(&device)?;
        
        // 3. Build input tensor on correct device
        let input_tensor = backend
            .tensor_builder()
            .build_from_vec(input_ids.clone(), &[1, input_ids.len()])
            .map_err(|e| RouterError::TensorConversion(format!("Failed to create tensor: {}", e)))?;
        
        // 4. Inference on model's device
        let output = self.engine
            .infer::<CandleBackend>(model_name, input_tensor)
            .map_err(|e| RouterError::Inference(format!("{:?}", e)))?;
        
        UnifiedTensorResponse::from_candle(output, start)
            .map_err(|e| RouterError::TensorConversion(e))
    }
}
```

### 5. PackageManager Integration

**Current State:** PackageManager already parses device from model.toml (line 682)

**Proposed Enhancement:**
```rust
impl PackageManager {
    /// Load model with explicit device override
    pub fn load_model_with_device(
        &self, 
        package: &ModelPackage,
        device_override: Option<DeviceId>
    ) -> Result<(LoadedModel, DeviceWrapper)> {
        let device = if let Some(override_device) = device_override {
            override_device
        } else {
            // Use device from package.info.device
            package.info.device.as_deref()
                .and_then(|s| DeviceId::parse(s).ok())
                .unwrap_or(DeviceId::Cpu)
        };
        
        // Load model on specified device...
        match backend_type {
            BackendType::Candle => {
                let candle_device = self.device_id_to_candle(&device)?;
                // ... existing load logic with candle_device
            }
            BackendType::Tch => {
                let tch_device = self.device_id_to_tch(&device)?;
                // ... existing load logic with tch_device
            }
            _ => unimplemented!()
        }
    }
    
    fn device_id_to_candle(&self, device: &DeviceId) -> Result<candle_core::Device> {
        match device {
            DeviceId::Cpu => Ok(candle_core::Device::Cpu),
            DeviceId::Cuda(idx) => candle_core::Device::new_cuda(*idx)
                .map_err(|e| Error::Generic(format!("CUDA error: {}", e))),
            DeviceId::Metal(idx) => candle_core::Device::new_metal(*idx)
                .map_err(|e| Error::Generic(format!("Metal error: {}", e))),
            DeviceId::Custom(s) => Err(Error::Generic(format!("Unsupported device: {}", s))),
        }
    }
    
    fn device_id_to_tch(&self, device: &DeviceId) -> Result<tch::Device> {
        match device {
            DeviceId::Cpu => Ok(tch::Device::Cpu),
            DeviceId::Cuda(idx) => Ok(tch::Device::Cuda(*idx)),
            DeviceId::Custom(s) if s == "mps" => {
                // Tch might support MPS in future
                Ok(tch::Device::Cpu)
            }
            _ => Err(Error::Generic(format!("Unsupported Tch device: {:?}", device))),
        }
    }
}
```

## Implementation Plan

### Phase 1: Core Device Abstraction (inferox-core)
- [ ] Add `DeviceId::parse()` and `DeviceId::to_string()`
- [ ] Add `DeviceError` enum
- [ ] Add `Model::device()` trait method
- [ ] Update `ModelMetadata` to include `device: Option<DeviceId>`
- [ ] Tests for device parsing and serialization

### Phase 2: Backend Device Tracking
- [ ] Update `CandleBackend` to expose device via `Model::device()`
- [ ] Update `TchBackend` to expose device via `Model::device()`
- [ ] Add device info to backend tests

### Phase 3: Engine Device Management (inferox-engine)
- [ ] Refactor `InferoxEngine` to use `RwLock<HashMap<String, ModelEntry>>`
- [ ] Implement `model_device()` method
- [ ] Implement `reload_model_to_device()` for hot-reload
- [ ] Implement `unload_model()` method
- [ ] Implement `list_models_with_devices()`
- [ ] Add comprehensive tests for device management

### Phase 4: Router Update (bert-multi-backend)
- [ ] Add backend caching by (BackendType, DeviceId)
- [ ] Update `route_to_candle()` to query model device
- [ ] Update `route_to_tch()` to query model device
- [ ] Remove hardcoded `Backend::cpu()` calls
- [ ] Add device-aware tests

### Phase 5: PackageManager Enhancement
- [ ] Add `load_model_with_device()` method
- [ ] Add `device_id_to_candle()` helper
- [ ] Add `device_id_to_tch()` helper
- [ ] Update existing `load_model()` to use new device logic

### Phase 6: Integration Tests
- [ ] Test startup with CPU device
- [ ] Test startup with CUDA device
- [ ] Test hot-reload CPU → CUDA
- [ ] Test hot-reload CUDA:0 → CUDA:1
- [ ] Test concurrent inference during reload
- [ ] Test device validation errors

## Migration Path

**For Existing Code:**
1. Default `Model::device()` implementation returns `Backend::default_device()`
2. Existing models continue to work without changes
3. Router code requires update to remove hardcoded `cpu()` calls

**For New Code:**
1. Specify device in model.toml
2. Use `load_model()` or `load_model_with_device()`
3. Router automatically uses correct device

## Performance Considerations

**Inference Hotpath:**
- ✅ No device changes during inference
- ✅ Backend caching avoids repeated creation
- ✅ RwLock allows concurrent reads (multiple inferences)
- ✅ Device lookup is O(1) HashMap access

**Management Operations:**
- ⚠️ `reload_model_to_device()` requires write lock (brief)
- ⚠️ Model loading is expensive (expected for mgmt operation)
- ✅ Atomic swap minimizes downtime during reload

## Alternatives Considered

### Alternative 1: Device per Request
**Rejected:** Would require model reload on every request, violates "no hotpath reload" requirement.

### Alternative 2: Separate Engine per Device
**Rejected:** Complex to manage, doesn't support hot-reload, wastes memory.

### Alternative 3: Device as Generic Parameter
**Rejected:** Would require recompiling model code for each device, too inflexible.

## Open Questions

1. **Should we support automatic device selection (e.g., "auto" → picks best available)?**
   - Proposal: Add `DeviceId::Auto` that resolves at load time

2. **How to handle multi-GPU model parallelism?**
   - Out of scope for this RFC, future enhancement

3. **Should device migration be async?**
   - Proposal: Start with sync, add async in future if needed

4. **Memory management during migration?**
   - Current: Old model dropped after new model loaded
   - Alternative: Could use memory pool for staging

## References

- Current codebase: inferox-core/src/device.rs
- Backend implementations: inferox-candle, inferox-tch
- Package format: inferox-mlpkg
- Example usage: examples/bert-multi-backend
