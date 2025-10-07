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

pub struct InferoxEngine<B: Backend> {
    backend: B,
    models: HashMap<String, Box<dyn Model<Backend = B, Input = B::Tensor, Output = B::Tensor>>>,
    config: EngineConfig,
}

impl<B: Backend> InferoxEngine<B> {
    pub fn new(backend: B, config: EngineConfig) -> Self {
        Self {
            backend,
            models: HashMap::new(),
            config,
        }
    }
    
    pub fn register_model<M>(&mut self, model: M) 
    where
        M: Model<Backend = B, Input = B::Tensor, Output = B::Tensor> + 'static,
    {
        let name = model.name().to_string();
        self.models.insert(name, Box::new(model));
    }
    
    pub fn register_boxed_model(&mut self, model: Box<dyn Model<Backend = B, Input = B::Tensor, Output = B::Tensor>>) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }
    
    pub fn infer(
        &self,
        model_name: &str,
        input: B::Tensor,
    ) -> Result<B::Tensor, InferoxError<B::Error>> {
        let model = self.models.get(model_name)
            .ok_or_else(|| InferoxError::ModelNotFound(model_name.to_string()))?;
        
        model.forward(input)
            .map_err(InferoxError::Backend)
    }
    
    pub fn infer_batch(
        &self,
        model_name: &str,
        batch: Vec<B::Tensor>,
    ) -> Result<Vec<B::Tensor>, InferoxError<B::Error>> {
        let model = self.models.get(model_name)
            .ok_or_else(|| InferoxError::ModelNotFound(model_name.to_string()))?;
        
        batch.into_iter()
            .map(|input| model.forward(input))
            .collect::<Result<Vec<_>, _>>()
            .map_err(InferoxError::Backend)
    }
    
    pub fn list_models(&self) -> Vec<(&str, ModelMetadata)> {
        self.models.iter()
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
