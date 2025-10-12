//! Model package format for Inferox
//!
//! This crate provides functionality for packaging and loading ML models
//! with their weights in a portable format. It integrates with Hugging Face
//! Hub for efficient model downloads using XET technology.

#![warn(missing_docs)]

use std::path::{Path, PathBuf};

/// Model package errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Hugging Face download error
    #[error("HF download error: {0}")]
    HfDownload(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid package format
    #[error("Invalid package format: {0}")]
    InvalidFormat(String),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

/// Result type for model package operations
pub type Result<T> = std::result::Result<T, Error>;

/// Backend type for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendType {
    /// Candle backend
    Candle,
    /// LibTorch backend (tch-rs)
    Tch,
    /// PyTorch backend
    Torch,
    /// TensorFlow Lite backend
    TFLite,
}

/// Device wrapper that can hold device for any backend
pub enum DeviceWrapper {
    /// Candle device
    Candle(candle_core::Device),
    #[cfg(feature = "tch")]
    /// Tch device
    Tch(inferox_tch::TchBackend),
}

impl DeviceWrapper {
    /// Get device type as string
    pub fn device_type(&self) -> &str {
        match self {
            DeviceWrapper::Candle(device) => {
                if device.is_cpu() {
                    "cpu"
                } else if device.is_cuda() {
                    "cuda"
                } else if device.is_metal() {
                    "mps"
                } else {
                    "unknown"
                }
            }
            #[cfg(feature = "tch")]
            DeviceWrapper::Tch(_) => "tch",
        }
    }

    /// Unwrap as Candle device
    pub fn as_candle(self) -> Option<candle_core::Device> {
        match self {
            DeviceWrapper::Candle(device) => Some(device),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    #[cfg(feature = "tch")]
    /// Unwrap as Tch backend (which includes device)
    pub fn as_tch(self) -> Option<inferox_tch::TchBackend> {
        match self {
            DeviceWrapper::Tch(backend) => Some(backend),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// Loaded model wrapper that can hold any backend type
pub enum LoadedModel {
    /// Candle backend model
    Candle(
        Box<
            dyn inferox_core::Model<
                Backend = inferox_candle::CandleBackend,
                Input = inferox_candle::CandleTensor,
                Output = inferox_candle::CandleTensor,
            >,
        >,
    ),
    #[cfg(feature = "tch")]
    /// Tch backend model
    Tch(
        Box<
            dyn inferox_core::Model<
                Backend = inferox_tch::TchBackend,
                Input = inferox_tch::TchTensor,
                Output = inferox_tch::TchTensor,
            >,
        >,
    ),
}

impl LoadedModel {
    /// Get the model name
    pub fn name(&self) -> &str {
        match self {
            LoadedModel::Candle(model) => model.name(),
            #[cfg(feature = "tch")]
            LoadedModel::Tch(model) => model.name(),
        }
    }

    /// Get the backend type of this model
    pub fn backend_type(&self) -> BackendType {
        match self {
            LoadedModel::Candle(_) => BackendType::Candle,
            #[cfg(feature = "tch")]
            LoadedModel::Tch(_) => BackendType::Tch,
        }
    }

    /// Unwrap as Candle model
    pub fn as_candle(
        self,
    ) -> Option<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_candle::CandleBackend,
                Input = inferox_candle::CandleTensor,
                Output = inferox_candle::CandleTensor,
            >,
        >,
    > {
        match self {
            LoadedModel::Candle(model) => Some(model),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    #[cfg(feature = "tch")]
    /// Unwrap as Tch model
    pub fn as_tch(
        self,
    ) -> Option<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_tch::TchBackend,
                Input = inferox_tch::TchTensor,
                Output = inferox_tch::TchTensor,
            >,
        >,
    > {
        match self {
            LoadedModel::Tch(model) => Some(model),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

impl BackendType {
    /// Get the string representation of the backend type
    pub fn as_str(&self) -> &str {
        match self {
            BackendType::Candle => "candle",
            BackendType::Tch => "tch",
            BackendType::Torch => "torch",
            BackendType::TFLite => "tflite",
        }
    }
}

/// Architecture family classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArchitectureFamily {
    /// Encoder-only models (BERT, RoBERTa)
    EncoderOnly,
    /// Decoder-only models (GPT-2, GPT-J)
    DecoderOnly,
    /// Encoder-decoder models (T5, BART)
    EncoderDecoder,
    /// Vision models (ResNet, ViT)
    Vision,
    /// Multimodal models (CLIP, BLIP)
    Multimodal,
}

/// Model information (architecture-agnostic metadata)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Model type identifier (e.g., "bert", "gpt2", "resnet")
    pub model_type: String,
    /// HuggingFace repository ID
    pub repo_id: String,
    /// Architecture family classification
    pub architecture_family: ArchitectureFamily,
    /// Supported backends for this model
    pub supported_backends: Vec<BackendType>,
    /// Default device for model execution (e.g., "cpu", "cuda", "cuda:0", "mps")
    pub device: Option<String>,
    /// Hidden size (for transformers)
    pub hidden_size: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Vocabulary size (for language models)
    pub vocab_size: Option<usize>,
}

/// Model package metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackageMetadata {
    /// Package format version
    pub version: String,

    /// Model name
    pub name: String,

    /// Model repository ID on Hugging Face
    pub repo_id: String,

    /// Model revision/commit
    pub revision: String,

    /// List of files included in the package
    pub files: Vec<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Model package structure
pub struct ModelPackage {
    /// Package directory path
    pub path: PathBuf,
    /// Model information
    pub info: ModelInfo,
    /// Package metadata
    pub metadata: PackageMetadata,
}

impl ModelPackage {
    /// Get the package directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the model information
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Get the package metadata
    pub fn metadata(&self) -> &PackageMetadata {
        &self.metadata
    }
}

/// HuggingFace config structure (simplified)
#[derive(Debug, serde::Deserialize)]
struct HFConfig {
    model_type: Option<String>,
    #[allow(dead_code)]
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    vocab_size: Option<usize>,
}

/// Parse HuggingFace config.json file
fn parse_hf_config(config_path: &Path) -> Result<HFConfig> {
    let content = std::fs::read_to_string(config_path)?;
    let config: HFConfig = serde_json::from_str(&content)?;
    Ok(config)
}

/// Detect architecture family from HF config
fn detect_architecture(config: &HFConfig) -> Result<ArchitectureFamily> {
    if let Some(ref model_type) = config.model_type {
        match model_type.as_str() {
            "bert" | "roberta" | "albert" | "distilbert" => Ok(ArchitectureFamily::EncoderOnly),
            "gpt2" | "gptj" | "gpt_neo" | "llama" => Ok(ArchitectureFamily::DecoderOnly),
            "t5" | "bart" | "mbart" => Ok(ArchitectureFamily::EncoderDecoder),
            "vit" | "resnet" | "convnext" => Ok(ArchitectureFamily::Vision),
            "clip" | "blip" => Ok(ArchitectureFamily::Multimodal),
            _ => Err(Error::Generic(format!(
                "Unknown model type: {}",
                model_type
            ))),
        }
    } else {
        Err(Error::Generic("No model_type in config".to_string()))
    }
}

/// Create ModelInfo from HF config
fn create_model_info(repo_id: &str, config: &HFConfig) -> Result<ModelInfo> {
    Ok(ModelInfo {
        model_type: config
            .model_type
            .clone()
            .ok_or_else(|| Error::Generic("Missing model_type".to_string()))?,
        repo_id: repo_id.to_string(),
        architecture_family: detect_architecture(config)?,
        supported_backends: vec![BackendType::Candle],
        device: Some("cpu".to_string()), // Default device
        hidden_size: config.hidden_size,
        num_layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
    })
}

/// Model package manager
pub struct PackageManager {
    #[allow(dead_code)]
    cache_dir: PathBuf,
    hf_client: hf_xet_rs::HfXetClient,
}

impl PackageManager {
    /// Create a new package manager
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let hf_client = hf_xet_rs::HfXetClient::with_cache_dir(cache_dir.clone())
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(Self {
            cache_dir,
            hf_client,
        })
    }

    /// Download a model package from Hugging Face
    pub async fn download(&self, repo_id: &str, revision: Option<&str>) -> Result<PathBuf> {
        let path = self
            .hf_client
            .snapshot_download(repo_id, revision, None, None)
            .await
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(path)
    }

    /// Download specific files from a repository
    pub async fn download_files(
        &self,
        repo_id: &str,
        patterns: &[&str],
        revision: Option<&str>,
    ) -> Result<Vec<PathBuf>> {
        let paths = self
            .hf_client
            .download_files(repo_id, patterns, revision)
            .await
            .map_err(|e| Error::HfDownload(e.to_string()))?;

        Ok(paths)
    }

    /// Load package metadata
    pub fn load_metadata(&self, package_path: &std::path::Path) -> Result<PackageMetadata> {
        let metadata_path = package_path.join("metadata.json");
        let content = std::fs::read_to_string(metadata_path)?;
        let metadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    /// Load complete model package from directory
    ///
    /// This reads both metadata.json and model_info.json from the package directory
    /// and returns a complete ModelPackage ready for loading.
    pub fn load_package(&self, package_path: &std::path::Path) -> Result<ModelPackage> {
        let metadata_path = package_path.join("metadata.json");
        let model_info_path = package_path.join("model_info.json");

        let metadata_content = std::fs::read_to_string(&metadata_path)
            .map_err(|e| Error::InvalidFormat(format!("Failed to read metadata.json: {}", e)))?;
        let metadata: PackageMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| Error::InvalidFormat(format!("Failed to parse metadata.json: {}", e)))?;

        let model_info_content = std::fs::read_to_string(&model_info_path)
            .map_err(|e| Error::InvalidFormat(format!("Failed to read model_info.json: {}", e)))?;
        let info: ModelInfo = serde_json::from_str(&model_info_content)
            .map_err(|e| Error::InvalidFormat(format!("Failed to parse model_info.json: {}", e)))?;

        Ok(ModelPackage {
            path: package_path.to_path_buf(),
            info,
            metadata,
        })
    }

    /// Download model from HuggingFace and organize into package structure
    pub async fn download_and_package(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        backends: &[BackendType],
    ) -> Result<ModelPackage> {
        // 1. Download snapshot using existing download() method
        let snapshot_path = self.download(repo_id, revision).await?;

        // 2. Parse config
        let config_path = snapshot_path.join("config.json");
        let hf_config = parse_hf_config(&config_path)?;
        let model_info = create_model_info(repo_id, &hf_config)?;

        // 3. Create package directory
        let package_dir = self
            .cache_dir
            .join("packages")
            .join(repo_id.replace('/', "_"));
        std::fs::create_dir_all(&package_dir)?;

        // 4. Write model_info.json
        let info_json = serde_json::to_string_pretty(&model_info)?;
        std::fs::write(package_dir.join("model_info.json"), info_json)?;

        // 5. Organize backend directories
        for backend in backends {
            let backend_dir = package_dir.join("backends").join(backend.as_str());
            std::fs::create_dir_all(&backend_dir)?;

            // Copy config
            std::fs::copy(&config_path, backend_dir.join("config.json"))?;

            // Copy weight files (safetensors for candle)
            if *backend == BackendType::Candle {
                for entry in std::fs::read_dir(&snapshot_path)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                        let filename = entry.file_name();
                        std::fs::copy(&path, backend_dir.join(filename))?;
                    }
                }
            }
        }

        // 6. Create metadata
        let metadata = PackageMetadata {
            version: "1.0".to_string(),
            name: model_info.model_type.clone(),
            repo_id: repo_id.to_string(),
            revision: revision.unwrap_or("main").to_string(),
            files: vec![],
            created_at: chrono::Utc::now(),
        };

        // 7. Write metadata.json
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(package_dir.join("metadata.json"), metadata_json)?;

        Ok(ModelPackage {
            path: package_dir,
            info: model_info,
            metadata,
        })
    }

    /// Install compiled model library into package
    ///
    /// The backend is automatically determined from the package's supported_backends metadata.
    /// Uses the first backend in the supported_backends list.
    pub fn install_model_library(&self, package: &ModelPackage, library_path: &Path) -> Result<()> {
        let backend = package.info.supported_backends.first().ok_or_else(|| {
            Error::InvalidFormat("No supported backends found in package metadata".to_string())
        })?;

        let backend_dir = package.path.join("backends").join(backend.as_str());

        #[cfg(target_os = "macos")]
        let lib_name = "libmodel.dylib";
        #[cfg(target_os = "windows")]
        let lib_name = "model.dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let lib_name = "libmodel.so";

        let target_path = backend_dir.join(lib_name);
        std::fs::copy(library_path, &target_path)?;

        println!(
            "Installed library at: {:?} (backend: {})",
            target_path,
            backend.as_str()
        );
        Ok(())
    }

    /// Assemble complete package from compiled library and HuggingFace model
    pub async fn assemble_package(
        repo_id: &str,
        library_path: &Path,
        output_dir: &Path,
    ) -> Result<ModelPackage> {
        let temp_cache = std::env::temp_dir().join("inferox-mlpkg-cache");
        std::fs::create_dir_all(&temp_cache)?;

        let manager = PackageManager::new(temp_cache)?;

        let package = manager
            .download_and_package(repo_id, None, &[BackendType::Candle])
            .await?;

        manager.install_model_library(&package, library_path)?;

        if output_dir.exists() {
            std::fs::remove_dir_all(output_dir)?;
        }
        copy_dir_all(package.path(), output_dir)?;

        Ok(ModelPackage {
            path: output_dir.to_path_buf(),
            info: package.info,
            metadata: package.metadata,
        })
    }

    /// Load Candle model from compiled library in package
    /// Backend type is automatically determined from package.info.supported_backends[0]
    fn load_model_candle_internal(
        &self,
        package: &ModelPackage,
    ) -> Result<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_candle::CandleBackend,
                Input = inferox_candle::CandleTensor,
                Output = inferox_candle::CandleTensor,
            >,
        >,
    > {
        let backend_type =
            package.info.supported_backends.first().ok_or_else(|| {
                Error::InvalidFormat("No supported backends in package".to_string())
            })?;

        let backend_dir = package.path.join("backends").join(backend_type.as_str());

        #[cfg(target_os = "macos")]
        let lib_name = "libmodel.dylib";
        #[cfg(target_os = "windows")]
        let lib_name = "model.dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let lib_name = "libmodel.so";

        let library_path = backend_dir.join(lib_name);

        if !library_path.exists() {
            return Err(Error::InvalidFormat(format!(
                "Model library not found: {:?}",
                library_path
            )));
        }

        std::env::set_var("INFEROX_PACKAGE_DIR", &backend_dir);

        type BoxedCandleModel = Box<
            dyn inferox_core::Model<
                Backend = inferox_candle::CandleBackend,
                Input = inferox_candle::CandleTensor,
                Output = inferox_candle::CandleTensor,
            >,
        >;
        type ModelFactory = fn() -> BoxedCandleModel;

        unsafe {
            let lib = libloading::Library::new(&library_path)
                .map_err(|e| Error::Generic(format!("Failed to load library: {}", e)))?;

            let factory: libloading::Symbol<ModelFactory> = lib
                .get(b"create_model")
                .map_err(|e| Error::Generic(format!("Failed to get create_model symbol: {}", e)))?;

            let model = factory();

            std::mem::forget(lib);

            Ok(model)
        }
    }

    #[cfg(feature = "tch")]
    /// Load Tch model from compiled library in package
    /// Backend type is automatically determined from package.info.supported_backends[0]
    fn load_model_tch_internal(
        &self,
        package: &ModelPackage,
    ) -> Result<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_tch::TchBackend,
                Input = inferox_tch::TchTensor,
                Output = inferox_tch::TchTensor,
            >,
        >,
    > {
        let backend_type =
            package.info.supported_backends.first().ok_or_else(|| {
                Error::InvalidFormat("No supported backends in package".to_string())
            })?;

        let backend_dir = package.path.join("backends").join(backend_type.as_str());

        #[cfg(target_os = "macos")]
        let lib_name = "libmodel.dylib";
        #[cfg(target_os = "windows")]
        let lib_name = "model.dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let lib_name = "libmodel.so";

        let library_path = backend_dir.join(lib_name);

        if !library_path.exists() {
            return Err(Error::InvalidFormat(format!(
                "Model library not found: {:?}",
                library_path
            )));
        }

        std::env::set_var("INFEROX_PACKAGE_DIR", package.path.to_str().unwrap());

        type BoxedTchModel = Box<
            dyn inferox_core::Model<
                Backend = inferox_tch::TchBackend,
                Input = inferox_tch::TchTensor,
                Output = inferox_tch::TchTensor,
            >,
        >;
        type ModelFactory = fn() -> BoxedTchModel;

        unsafe {
            let lib = libloading::Library::new(&library_path)
                .map_err(|e| Error::Generic(format!("Failed to load library: {}", e)))?;

            let factory: libloading::Symbol<ModelFactory> = lib
                .get(b"create_model")
                .map_err(|e| Error::Generic(format!("Failed to get create_model symbol: {}", e)))?;

            let model = factory();

            std::mem::forget(lib);

            Ok(model)
        }
    }

    /// Load model from compiled library in package
    /// Backend type and device are automatically determined from model.toml
    ///
    /// Returns a tuple of (LoadedModel, DeviceWrapper) where both are determined from package metadata
    pub fn load_model(&self, package: &ModelPackage) -> Result<(LoadedModel, DeviceWrapper)> {
        let backend_type =
            package.info.supported_backends.first().ok_or_else(|| {
                Error::InvalidFormat("No supported backends in package".to_string())
            })?;

        let device_str = package.info.device.as_deref().unwrap_or("cpu");

        match backend_type {
            BackendType::Candle => {
                // Parse device for Candle
                let device = if device_str.starts_with("cuda") {
                    if let Some(idx_str) = device_str.strip_prefix("cuda:") {
                        let idx: usize = idx_str.parse().map_err(|_| {
                            Error::InvalidFormat(format!("Invalid CUDA device: {}", device_str))
                        })?;
                        candle_core::Device::new_cuda(idx).map_err(|e| {
                            Error::Generic(format!("Failed to create CUDA device: {}", e))
                        })?
                    } else {
                        candle_core::Device::new_cuda(0).map_err(|e| {
                            Error::Generic(format!("Failed to create CUDA device: {}", e))
                        })?
                    }
                } else if device_str == "mps" || device_str == "metal" {
                    candle_core::Device::new_metal(0).map_err(|e| {
                        Error::Generic(format!("Failed to create Metal device: {}", e))
                    })?
                } else {
                    candle_core::Device::Cpu
                };

                let model = self.load_model_candle_internal(package)?;
                Ok((LoadedModel::Candle(model), DeviceWrapper::Candle(device)))
            }
            #[cfg(feature = "tch")]
            BackendType::Tch => {
                // Parse device for Tch and create backend
                let backend = if device_str.starts_with("cuda") {
                    if let Some(idx_str) = device_str.strip_prefix("cuda:") {
                        let idx: usize = idx_str.parse().map_err(|_| {
                            Error::InvalidFormat(format!("Invalid CUDA device: {}", device_str))
                        })?;
                        inferox_tch::TchBackend::cuda(idx).map_err(|e| {
                            Error::Generic(format!("Failed to create CUDA backend: {}", e))
                        })?
                    } else {
                        inferox_tch::TchBackend::cuda(0).map_err(|e| {
                            Error::Generic(format!("Failed to create CUDA backend: {}", e))
                        })?
                    }
                } else {
                    inferox_tch::TchBackend::cpu().map_err(|e| {
                        Error::Generic(format!("Failed to create CPU backend: {}", e))
                    })?
                };

                let model = self.load_model_tch_internal(package)?;
                Ok((LoadedModel::Tch(model), DeviceWrapper::Tch(backend)))
            }
            #[cfg(not(feature = "tch"))]
            BackendType::Tch => Err(Error::Generic(
                "Tch backend not enabled. Compile with --features tch".to_string(),
            )),
            _ => Err(Error::Generic(format!(
                "Backend {:?} not supported yet",
                backend_type
            ))),
        }
    }

    /// Load model from compiled library in package and unwrap as Candle model
    /// Backend type is automatically determined from package.info.supported_backends
    pub fn load_model_candle(
        &self,
        package: &ModelPackage,
    ) -> Result<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_candle::CandleBackend,
                Input = inferox_candle::CandleTensor,
                Output = inferox_candle::CandleTensor,
            >,
        >,
    > {
        self.load_model_candle_internal(package)
    }

    #[cfg(feature = "tch")]
    /// Load model from compiled library in package and unwrap as Tch model
    /// Backend type is automatically determined from package.info.supported_backends[0]
    pub fn load_model_tch(
        &self,
        package: &ModelPackage,
    ) -> Result<
        Box<
            dyn inferox_core::Model<
                Backend = inferox_tch::TchBackend,
                Input = inferox_tch::TchTensor,
                Output = inferox_tch::TchTensor,
            >,
        >,
    > {
        self.load_model_tch_internal(package)
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Build script helper for assembling model packages
pub struct BuildScriptRunner {
    model_name: String,
}

impl BuildScriptRunner {
    /// Create a new build script runner
    ///
    /// # Arguments
    /// * `model_name` - Name of the model (e.g., "bert-candle")
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }

    /// Run the build script assembly process
    ///
    /// Call this from your build.rs:
    /// ```no_run
    /// inferox_mlpkg::BuildScriptRunner::new("bert-candle").run();
    /// ```
    pub fn run(&self) {
        use std::env;

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=model.toml");

        let profile = env::var("PROFILE").unwrap_or_default();
        if profile != "release" {
            return;
        }

        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let model_toml_path = manifest_dir.join("model.toml");

        let model_toml_content = match std::fs::read_to_string(&model_toml_path) {
            Ok(content) => content,
            Err(_) => {
                println!("cargo:warning=model.toml not found - skipping package assembly");
                return;
            }
        };

        let repo_id = model_toml_content
            .lines()
            .find(|line| line.starts_with("repo_id"))
            .and_then(|line| line.split('=').nth(1))
            .map(|s| s.trim().trim_matches('"'))
            .expect("repo_id not found in model.toml");

        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

        #[cfg(target_os = "macos")]
        let lib_ext = "dylib";
        #[cfg(target_os = "windows")]
        let lib_ext = "dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let lib_ext = "so";

        let lib_name = if cfg!(target_os = "windows") {
            format!("{}.{}", self.model_name.replace("-", "_"), lib_ext)
        } else {
            format!("lib{}.{}", self.model_name.replace("-", "_"), lib_ext)
        };

        let lib_path = workspace_root.join("target").join(&profile).join(&lib_name);

        if !lib_path.exists() {
            println!(
                "cargo:warning=Library not yet built - package will be assembled on next build"
            );
            return;
        }

        let output_dir = workspace_root
            .join("target")
            .join("mlpkg")
            .join(&self.model_name);

        println!("cargo:warning=Assembling package from {}", repo_id);

        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(async {
            PackageManager::assemble_package(repo_id, &lib_path, &output_dir).await
        }) {
            Ok(_) => {
                println!("cargo:warning=✓ Package ready at {}", output_dir.display());
            }
            Err(e) => {
                eprintln!("\n========================================");
                eprintln!("Package assembly FAILED:");
                eprintln!("{:?}", e);
                eprintln!("========================================\n");
                println!("cargo:warning=Package assembly failed: {:?}", e);
                panic!("Package assembly failed: {:?}", e);
            }
        }
    }
}
