use crate::InferoxEngine;
use inferox_core::{Backend, InferoxError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

pub struct SessionContext {
    pub batch_size: usize,
    pub sequence_length: Option<usize>,
    pub cache: HashMap<String, Box<dyn Any + Send + Sync>>,
}

pub struct InferenceSession<B: Backend> {
    engine: Arc<InferoxEngine<B>>,
    model_name: String,
    context: SessionContext,
}

impl<B: Backend> InferenceSession<B> {
    pub fn new(engine: Arc<InferoxEngine<B>>, model_name: String) -> Self {
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

    pub fn run(&mut self, input: B::Tensor) -> Result<B::Tensor, InferoxError<B::Error>> {
        self.engine.infer(&self.model_name, input)
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
