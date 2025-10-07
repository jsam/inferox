use crate::Backend;
use std::collections::HashMap;
use std::path::Path;

pub trait Model: Send + Sync {
    type Backend: Backend;
    type Input;
    type Output;
    
    fn name(&self) -> &str;
    
    fn forward(
        &self,
        input: Self::Input,
    ) -> Result<Self::Output, <Self::Backend as Backend>::Error>;
    
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata::default()
    }
    
    fn memory_requirements(&self) -> MemoryRequirements {
        MemoryRequirements::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub tags: Vec<String>,
    pub custom: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryRequirements {
    pub parameters: usize,
    pub activations: usize,
    pub peak: usize,
}

pub trait BatchedModel: Model {
    fn forward_batch(
        &self,
        batch: Vec<Self::Input>,
    ) -> Result<Vec<Self::Output>, <Self::Backend as Backend>::Error> {
        batch.into_iter()
            .map(|input| self.forward(input))
            .collect()
    }
}

pub trait SaveLoadModel: Model {
    fn save(&self, path: &Path) -> Result<(), <Self::Backend as Backend>::Error>;
    
    fn load(&mut self, path: &Path) -> Result<(), <Self::Backend as Backend>::Error>;
}
