use inferox_core::{
    AnyModel, Backend, InferoxError, Model, ModelMetadata, TensorBuilder, TypeErasedModel,
};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug)]
pub struct DynError(Box<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for DynError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DynError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

pub struct EngineConfig {
    pub max_batch_size: usize,
    pub enable_profiling: bool,
    pub memory_pool_size: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            enable_profiling: false,
            memory_pool_size: None,
        }
    }
}

impl EngineConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn with_memory_pool_size(mut self, size: Option<usize>) -> Self {
        self.memory_pool_size = size;
        self
    }
}

type BoxedModel<B> =
    Box<dyn Model<Backend = B, Input = <B as Backend>::Tensor, Output = <B as Backend>::Tensor>>;

struct ModelEntry {
    model: Box<dyn AnyModel>,
    backend_name: String,
    device: inferox_core::DeviceId,
}

pub struct InferenceOutput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct InferoxEngine {
    models: HashMap<String, ModelEntry>,
    _config: EngineConfig,
}

impl InferoxEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            models: HashMap::new(),
            _config: config,
        }
    }

    pub fn register_model<B: Backend + 'static>(
        &mut self,
        name: impl Into<String>,
        model: BoxedModel<B>,
        device: Option<inferox_core::DeviceId>,
    ) where
        B::Tensor: 'static,
    {
        let name = name.into();
        let backend_name = std::any::type_name::<B>().to_string();
        let device = device.unwrap_or(inferox_core::DeviceId::Cpu);
        let erased = TypeErasedModel::new(model);
        let entry = ModelEntry {
            model: Box::new(erased),
            backend_name,
            device,
        };
        self.models.insert(name, entry);
    }

    pub fn remove_model(&mut self, name: &str) -> Result<(), InferoxError<DynError>> {
        self.models
            .remove(name)
            .ok_or_else(|| InferoxError::ModelNotFound(name.to_string()))?;
        Ok(())
    }

    pub fn reload_model_to_device<B: Backend + 'static>(
        &mut self,
        name: &str,
        new_device: inferox_core::DeviceId,
        loader: impl FnOnce(
            inferox_core::DeviceId,
        ) -> Result<BoxedModel<B>, Box<dyn std::error::Error + Send + Sync>>,
    ) -> Result<(), InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        let entry = self
            .models
            .get(name)
            .ok_or_else(|| InferoxError::ModelNotFound(name.to_string()))?;

        let backend_name = entry.backend_name.clone();

        let new_model =
            loader(new_device.clone()).map_err(|e| InferoxError::Backend(DynError(e)))?;

        let erased = TypeErasedModel::new(new_model);
        let new_entry = ModelEntry {
            model: Box::new(erased),
            backend_name,
            device: new_device,
        };

        self.models.insert(name.to_string(), new_entry);
        Ok(())
    }

    pub fn infer(
        &self,
        name: &str,
        input_ids: Vec<i64>,
    ) -> Result<InferenceOutput, InferoxError<DynError>> {
        let entry = self
            .models
            .get(name)
            .ok_or_else(|| InferoxError::ModelNotFound(name.to_string()))?;

        let shape = vec![1, input_ids.len()];

        if entry.backend_name.contains("CandleBackend") {
            let candle_device = match &entry.device {
                inferox_core::DeviceId::Cpu => candle_core::Device::Cpu,
                inferox_core::DeviceId::Cuda(idx) => candle_core::Device::new_cuda(*idx)
                    .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?,
                inferox_core::DeviceId::Metal(idx) => candle_core::Device::new_metal(*idx)
                    .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?,
                _ => candle_core::Device::Cpu,
            };
            let backend = inferox_candle::CandleBackend::with_device(candle_device);

            let input_tensor = backend
                .tensor_builder()
                .build_from_vec(input_ids, &shape)
                .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?;

            let output_tensor =
                self.infer_typed::<inferox_candle::CandleBackend>(name, input_tensor)?;

            let candle_tensor: candle_core::Tensor = output_tensor.into();
            let output_shape = candle_tensor.shape().dims().to_vec();
            let flattened = candle_tensor
                .flatten_all()
                .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?;
            let data = flattened
                .to_vec1::<f32>()
                .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?;

            Ok(InferenceOutput {
                data,
                shape: output_shape,
            })
        } else if entry.backend_name.contains("TchBackend") {
            #[cfg(feature = "tch")]
            {
                let tch_device = match &entry.device {
                    inferox_core::DeviceId::Cpu => tch::Device::Cpu,
                    inferox_core::DeviceId::Cuda(idx) => tch::Device::Cuda(*idx),
                    _ => tch::Device::Cpu,
                };
                let backend = inferox_tch::TchBackend::with_device(tch_device);

                let input_tensor = backend
                    .tensor_builder()
                    .build_from_vec(input_ids, &shape)
                    .map_err(|e| InferoxError::Backend(DynError(Box::new(e))))?;

                let output_tensor =
                    self.infer_typed::<inferox_tch::TchBackend>(name, input_tensor)?;

                let tch_tensor: tch::Tensor = output_tensor.into();
                let output_shape: Vec<usize> =
                    tch_tensor.size().iter().map(|&x| x as usize).collect();
                let data: Vec<f32> = tch_tensor
                    .flatten(0, -1)
                    .try_into()
                    .map_err(|e: tch::TchError| InferoxError::Backend(DynError(Box::new(e))))?;

                Ok(InferenceOutput {
                    data,
                    shape: output_shape,
                })
            }
            #[cfg(not(feature = "tch"))]
            {
                Err(InferoxError::Backend(DynError(
                    "Tch backend not available".into(),
                )))
            }
        } else {
            Err(InferoxError::Backend(DynError(
                format!("Unknown backend: {}", entry.backend_name).into(),
            )))
        }
    }

    pub fn infer_typed<B: Backend + 'static>(
        &self,
        name: &str,
        input: B::Tensor,
    ) -> Result<B::Tensor, InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        let entry = self
            .models
            .get(name)
            .ok_or_else(|| InferoxError::ModelNotFound(name.to_string()))?;

        let boxed_input: Box<dyn Any> = Box::new(input);
        let boxed_output = entry
            .model
            .forward_any(boxed_input)
            .map_err(|e| InferoxError::Backend(DynError(e)))?;

        let output = boxed_output
            .downcast::<B::Tensor>()
            .map_err(|_| InferoxError::Backend(DynError("Output type mismatch".into())))?;

        Ok(*output)
    }

    pub fn infer_batch<B: Backend + 'static>(
        &self,
        name: &str,
        batch: Vec<B::Tensor>,
    ) -> Result<Vec<B::Tensor>, InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        batch
            .into_iter()
            .map(|input| self.infer_typed::<B>(name, input))
            .collect()
    }

    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn list_models_with_metadata(&self) -> Vec<(&str, ModelMetadata)> {
        self.models
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.model.metadata()))
            .collect()
    }

    pub fn model_info(&self, model_name: &str) -> Option<ModelMetadata> {
        self.models.get(model_name).map(|e| e.model.metadata())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_candle::CandleBackend;
    use inferox_core::{Backend, Model, Tensor, TensorBuilder};

    struct DummyModel {
        name: String,
    }

    impl Model for DummyModel {
        type Backend = CandleBackend;
        type Input = <CandleBackend as Backend>::Tensor;
        type Output = <CandleBackend as Backend>::Tensor;

        fn name(&self) -> &str {
            &self.name
        }

        fn forward(
            &self,
            input: Self::Input,
        ) -> Result<Self::Output, <Self::Backend as Backend>::Error> {
            Ok(input)
        }
    }

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert!(!config.enable_profiling);
        assert_eq!(config.memory_pool_size, None);
    }

    #[test]
    fn test_engine_config_new() {
        let config = EngineConfig::new();
        assert_eq!(config.max_batch_size, 32);
    }

    #[test]
    fn test_engine_config_with_max_batch_size() {
        let config = EngineConfig::new().with_max_batch_size(64);
        assert_eq!(config.max_batch_size, 64);
    }

    #[test]
    fn test_engine_config_with_profiling() {
        let config = EngineConfig::new().with_profiling(true);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_engine_config_with_memory_pool_size() {
        let config = EngineConfig::new().with_memory_pool_size(Some(1024));
        assert_eq!(config.memory_pool_size, Some(1024));
    }

    #[test]
    fn test_engine_new() {
        let config = EngineConfig::default();
        let engine = InferoxEngine::new(config);
        assert_eq!(engine.list_models().len(), 0);
    }

    #[test]
    fn test_register_model() {
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test_model".to_string(),
            });
        engine.register_model("test_model", model, None);

        assert_eq!(engine.list_models().len(), 1);
    }

    #[test]
    fn test_register_model_boxed() {
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "boxed_model".to_string(),
            });
        engine.register_model("boxed_model", model, None);

        assert_eq!(engine.list_models().len(), 1);
    }

    #[test]
    fn test_infer() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test".to_string(),
            });
        engine.register_model("test", model, None);

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .unwrap();
        let output = engine.infer_typed::<CandleBackend>("test", input).unwrap();
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_infer_model_not_found() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let engine = InferoxEngine::new(config);

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32], &[1, 1])
            .unwrap();
        let result = engine.infer_typed::<CandleBackend>("nonexistent", input);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_batch() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test".to_string(),
            });
        engine.register_model("test", model, None);

        let input1 = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0], &[1, 2])
            .unwrap();
        let input2 = backend
            .tensor_builder()
            .build_from_vec(vec![3.0f32, 4.0], &[1, 2])
            .unwrap();

        let outputs = engine
            .infer_batch::<CandleBackend>("test", vec![input1, input2])
            .unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_list_models() {
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model1: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "model1".to_string(),
            });
        let model2: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "model2".to_string(),
            });

        engine.register_model("model1", model1, None);
        engine.register_model("model2", model2, None);

        let models = engine.list_models();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_model_info() {
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test".to_string(),
            });
        engine.register_model("test", model, None);

        let info = engine.model_info("test");
        assert!(info.is_some());

        let no_info = engine.model_info("nonexistent");
        assert!(no_info.is_none());
    }
}
