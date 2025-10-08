use inferox_core::{Backend, InferoxError, Model, ModelMetadata};
use std::collections::HashMap;

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

pub struct InferoxEngine<B: Backend> {
    backend: B,
    models: HashMap<String, BoxedModel<B>>,
    _config: EngineConfig,
}

impl<B: Backend> InferoxEngine<B> {
    pub fn new(backend: B, config: EngineConfig) -> Self {
        Self {
            backend,
            models: HashMap::new(),
            _config: config,
        }
    }

    pub fn register_model<M>(&mut self, model: M)
    where
        M: Model<Backend = B, Input = B::Tensor, Output = B::Tensor> + 'static,
    {
        let name = model.name().to_string();
        self.models.insert(name, Box::new(model));
    }

    pub fn register_boxed_model(
        &mut self,
        model: Box<dyn Model<Backend = B, Input = B::Tensor, Output = B::Tensor>>,
    ) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }

    pub fn infer(
        &self,
        model_name: &str,
        input: B::Tensor,
    ) -> Result<B::Tensor, InferoxError<B::Error>> {
        let model = self
            .models
            .get(model_name)
            .ok_or_else(|| InferoxError::ModelNotFound(model_name.to_string()))?;

        model.forward(input).map_err(InferoxError::Backend)
    }

    pub fn infer_batch(
        &self,
        model_name: &str,
        batch: Vec<B::Tensor>,
    ) -> Result<Vec<B::Tensor>, InferoxError<B::Error>> {
        let model = self
            .models
            .get(model_name)
            .ok_or_else(|| InferoxError::ModelNotFound(model_name.to_string()))?;

        batch
            .into_iter()
            .map(|input| model.forward(input))
            .collect::<Result<Vec<_>, _>>()
            .map_err(InferoxError::Backend)
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

    pub fn backend(&self) -> &B {
        &self.backend
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
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let engine = InferoxEngine::new(backend, config);
        assert_eq!(engine.list_models().len(), 0);
    }

    #[test]
    fn test_register_model() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend, config);

        let model = DummyModel {
            name: "test_model".to_string(),
        };
        engine.register_model(model);

        assert_eq!(engine.list_models().len(), 1);
    }

    #[test]
    fn test_register_boxed_model() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend, config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "boxed_model".to_string(),
            });
        engine.register_boxed_model(model);

        assert_eq!(engine.list_models().len(), 1);
    }

    #[test]
    fn test_infer() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend.clone(), config);

        let model = DummyModel {
            name: "test".to_string(),
        };
        engine.register_model(model);

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .unwrap();
        let output = engine.infer("test", input).unwrap();
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_infer_model_not_found() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let engine = InferoxEngine::new(backend.clone(), config);

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32], &[1, 1])
            .unwrap();
        let result = engine.infer("nonexistent", input);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_batch() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend.clone(), config);

        let model = DummyModel {
            name: "test".to_string(),
        };
        engine.register_model(model);

        let input1 = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0], &[1, 2])
            .unwrap();
        let input2 = backend
            .tensor_builder()
            .build_from_vec(vec![3.0f32, 4.0], &[1, 2])
            .unwrap();

        let outputs = engine.infer_batch("test", vec![input1, input2]).unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_list_models() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend, config);

        let model1 = DummyModel {
            name: "model1".to_string(),
        };
        let model2 = DummyModel {
            name: "model2".to_string(),
        };

        engine.register_model(model1);
        engine.register_model(model2);

        let models = engine.list_models();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_model_info() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let mut engine = InferoxEngine::new(backend, config);

        let model = DummyModel {
            name: "test".to_string(),
        };
        engine.register_model(model);

        let info = engine.model_info("test");
        assert!(info.is_some());

        let no_info = engine.model_info("nonexistent");
        assert!(no_info.is_none());
    }

    #[test]
    fn test_backend_accessor() {
        let backend = CandleBackend::cpu().unwrap();
        let config = EngineConfig::default();
        let engine = InferoxEngine::new(backend, config);

        assert_eq!(engine.backend().name(), "candle");
    }
}
