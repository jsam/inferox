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
        batch.into_iter().map(|input| self.forward(input)).collect()
    }
}

pub trait SaveLoadModel: Model {
    fn save(&self, path: &Path) -> Result<(), <Self::Backend as Backend>::Error>;

    fn load(&mut self, path: &Path) -> Result<(), <Self::Backend as Backend>::Error>;
}

use std::any::Any;

pub trait AnyModel: Send + Sync {
    fn name(&self) -> &str;
    fn metadata(&self) -> ModelMetadata;
    fn forward_any(
        &self,
        input: Box<dyn Any>,
    ) -> Result<Box<dyn Any>, Box<dyn std::error::Error + Send + Sync>>;
    fn backend_name(&self) -> &str;
    fn as_any(&self) -> &dyn Any;
}

pub struct TypeErasedModel<B: Backend + 'static> {
    inner: Box<dyn Model<Backend = B, Input = B::Tensor, Output = B::Tensor>>,
}

impl<B: Backend + 'static> TypeErasedModel<B>
where
    B::Tensor: 'static,
{
    pub fn new(model: Box<dyn Model<Backend = B, Input = B::Tensor, Output = B::Tensor>>) -> Self {
        Self { inner: model }
    }
}

impl<B: Backend + 'static> AnyModel for TypeErasedModel<B>
where
    B::Tensor: 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn metadata(&self) -> ModelMetadata {
        self.inner.metadata()
    }

    fn forward_any(
        &self,
        input: Box<dyn Any>,
    ) -> Result<Box<dyn Any>, Box<dyn std::error::Error + Send + Sync>> {
        let typed_input = input
            .downcast::<B::Tensor>()
            .map_err(|_| "Input type mismatch")?;

        let output = self
            .inner
            .forward(*typed_input)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        Ok(Box::new(output))
    }

    fn backend_name(&self) -> &str {
        std::any::type_name::<B>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
