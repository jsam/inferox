use crate::{DynError, InferoxEngine};
use inferox_core::{Backend, InferoxError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

pub struct SessionContext {
    pub batch_size: usize,
    pub sequence_length: Option<usize>,
    pub cache: HashMap<String, Box<dyn Any + Send + Sync>>,
}

pub struct InferenceSession {
    engine: Arc<InferoxEngine>,
    model_name: String,
    context: SessionContext,
}

impl InferenceSession {
    pub fn new(engine: Arc<InferoxEngine>, model_name: String) -> Self {
        Self {
            engine,
            model_name,
            context: SessionContext {
                batch_size: 1,
                sequence_length: None,
                cache: HashMap::new(),
            },
        }
    }

    pub fn run<B: Backend + 'static>(
        &mut self,
        input: B::Tensor,
    ) -> Result<B::Tensor, InferoxError<DynError>>
    where
        B::Tensor: 'static,
    {
        self.engine.infer::<B>(&self.model_name, input)
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.context.batch_size = batch_size;
    }

    pub fn store_state<T: Any + Send + Sync>(&mut self, key: String, value: T) {
        self.context.cache.insert(key, Box::new(value));
    }

    pub fn get_state<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.context.cache.get(key)?.downcast_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferoxEngine;
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
    fn test_session_new() {
        let config = crate::EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test".to_string(),
            });
        engine.register_model(model);

        let engine_arc = Arc::new(engine);
        let session = InferenceSession::new(engine_arc, "test".to_string());

        assert_eq!(session.context.batch_size, 1);
        assert_eq!(session.context.sequence_length, None);
    }

    #[test]
    fn test_session_run() {
        let backend = CandleBackend::cpu().unwrap();
        let config = crate::EngineConfig::default();
        let mut engine = InferoxEngine::new(config);

        let model: Box<dyn Model<Backend = CandleBackend, Input = _, Output = _>> =
            Box::new(DummyModel {
                name: "test".to_string(),
            });
        engine.register_model(model);

        let engine_arc = Arc::new(engine);
        let mut session = InferenceSession::new(engine_arc, "test".to_string());

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3])
            .unwrap();

        let output = session.run::<CandleBackend>(input).unwrap();
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_session_set_batch_size() {
        let config = crate::EngineConfig::default();
        let engine = InferoxEngine::new(config);
        let engine_arc = Arc::new(engine);
        let mut session = InferenceSession::new(engine_arc, "test".to_string());

        session.set_batch_size(16);
        assert_eq!(session.context.batch_size, 16);
    }

    #[test]
    fn test_session_state_storage() {
        let config = crate::EngineConfig::default();
        let engine = InferoxEngine::new(config);
        let engine_arc = Arc::new(engine);
        let mut session = InferenceSession::new(engine_arc, "test".to_string());

        session.store_state("counter".to_string(), 42usize);
        let value = session.get_state::<usize>("counter");
        assert_eq!(value, Some(&42));
    }

    #[test]
    fn test_session_state_retrieval_wrong_type() {
        let config = crate::EngineConfig::default();
        let engine = InferoxEngine::new(config);
        let engine_arc = Arc::new(engine);
        let mut session = InferenceSession::new(engine_arc, "test".to_string());

        session.store_state("counter".to_string(), 42usize);
        let value = session.get_state::<String>("counter");
        assert_eq!(value, None);
    }
}
