use inferox_core::{AnyModel, Backend, InferoxError, Model, ModelMetadata, TypeErasedModel};
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

pub struct InferoxEngine {
    models: HashMap<String, Box<dyn AnyModel>>,
    _config: EngineConfig,
}

impl InferoxEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            models: HashMap::new(),
            _config: config,
        }
    }

    pub fn register_model<B: Backend + 'static>(&mut self, model: BoxedModel<B>)
    where
        B::Tensor: 'static,
    {
        let name = model.name().to_string();
        let erased = TypeErasedModel::new(model);
        self.models.insert(name, Box::new(erased));
    }

    pub fn infer<B: Backend + 'static>(
        &self,
        model_name: &str,
        input: B::Tensor,
    ) -> Result<B::Tensor, InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        let model = self
            .models
            .get(model_name)
            .ok_or_else(|| InferoxError::ModelNotFound(model_name.to_string()))?;

        let boxed_input: Box<dyn Any> = Box::new(input);
        let boxed_output = model
            .forward_any(boxed_input)
            .map_err(|e| InferoxError::Backend(DynError(e)))?;

        let output = boxed_output
            .downcast::<B::Tensor>()
            .map_err(|_| InferoxError::Backend(DynError("Output type mismatch".into())))?;

        Ok(*output)
    }

    pub fn infer_batch<B: Backend + 'static>(
        &self,
        model_name: &str,
        batch: Vec<B::Tensor>,
    ) -> Result<Vec<B::Tensor>, InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        batch
            .into_iter()
            .map(|input| self.infer::<B>(model_name, input))
            .collect()
    }

    pub fn list_models(&self) -> Vec<(&str, ModelMetadata)> {
        self.models
            .iter()
            .map(|(name, model)| (name.as_str(), model.metadata()))
            .collect()
    }

    pub fn model_info(&self, model_name: &str) -> Option<ModelMetadata> {
        self.models.get(model_name).map(|m| m.metadata())
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
        engine.register_model(model);

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
        engine.register_model(model);

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
        engine.register_model(model);

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .unwrap();
        let output = engine.infer::<CandleBackend>("test", input).unwrap();
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
        let result = engine.infer::<CandleBackend>("nonexistent", input);
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
        engine.register_model(model);

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

        engine.register_model(model1);
        engine.register_model(model2);

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
        engine.register_model(model);

        let info = engine.model_info("test");
        assert!(info.is_some());

        let no_info = engine.model_info("nonexistent");
        assert!(no_info.is_none());
    }
}
