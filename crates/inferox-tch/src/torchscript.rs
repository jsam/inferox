use crate::{TchBackend, TchTensor};
use inferox_core::{Model, ModelMetadata};
use std::path::Path;
use tch::{CModule, Device, IValue, Tensor};

pub struct TorchScriptModel {
    name: String,
    module: CModule,
    metadata: ModelMetadata,
}

impl TorchScriptModel {
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, tch::TchError> {
        let module = CModule::load_on_device(path.as_ref(), device)?;
        
        let name = path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("torchscript_model")
            .to_string();
        
        Ok(Self {
            name: name.clone(),
            module,
            metadata: Self::default_metadata(&name),
        })
    }
    
    pub fn load_with_metadata(
        path: impl AsRef<Path>,
        device: Device,
        metadata: ModelMetadata,
    ) -> Result<Self, tch::TchError> {
        let module = CModule::load_on_device(path.as_ref(), device)?;
        let name = metadata.name.clone();
        Ok(Self { name, module, metadata })
    }
    
    fn default_metadata(name: &str) -> ModelMetadata {
        ModelMetadata {
            name: name.to_string(),
            version: "1.0".to_string(),
            description: format!("TorchScript model: {}", name),
            author: "Inferox".to_string(),
            license: "Unknown".to_string(),
            tags: vec!["torchscript".to_string(), "tch".to_string()],
            custom: Default::default(),
        }
    }
    
    pub fn forward_multi(
        &self,
        inputs: Vec<TchTensor>,
    ) -> Result<Vec<TchTensor>, tch::TchError> {
        let input_tensors: Vec<Tensor> = inputs.into_iter()
            .map(|t| t.0)
            .collect();
        
        let output_ivalue = self.module.forward_is(&input_tensors
            .iter()
            .map(|t| IValue::Tensor(t.shallow_clone()))
            .collect::<Vec<_>>())?;
        
        Self::parse_ivalue_output(output_ivalue)
    }
    
    pub fn forward_batch(&self, inputs: Vec<TchTensor>) -> Result<Vec<TchTensor>, tch::TchError> {
        let input_tensors: Vec<Tensor> = inputs.into_iter().map(|t| t.0).collect();
        let batched = Tensor::cat(&input_tensors, 0);
        let output = self.module.forward_ts(&[batched])?;
        
        let batch_size = input_tensors.len() as i64;
        let mut outputs = Vec::with_capacity(batch_size as usize);
        for i in 0..batch_size {
            outputs.push(TchTensor(output.get(i)));
        }
        Ok(outputs)
    }
    
    pub fn method(
        &self,
        method_name: &str,
        inputs: Vec<TchTensor>,
    ) -> Result<Vec<TchTensor>, tch::TchError> {
        let input_tensors: Vec<Tensor> = inputs.into_iter()
            .map(|t| t.0)
            .collect();
        
        let output_ivalue = self.module.method_is(
            method_name,
            &input_tensors
                .iter()
                .map(|t| IValue::Tensor(t.shallow_clone()))
                .collect::<Vec<_>>()
        )?;
        
        Self::parse_ivalue_output(output_ivalue)
    }
    
    fn parse_ivalue_output(ivalue: IValue) -> Result<Vec<TchTensor>, tch::TchError> {
        match ivalue {
            IValue::Tensor(t) => Ok(vec![TchTensor(t)]),
            IValue::TensorList(tensors) => {
                Ok(tensors.into_iter().map(TchTensor).collect())
            }
            IValue::Tuple(values) => {
                values.into_iter()
                    .map(|v| match v {
                        IValue::Tensor(t) => Ok(TchTensor(t)),
                        _ => Err(tch::TchError::Torch(
                            "Unsupported output type in tuple".to_string()
                        )),
                    })
                    .collect()
            }
            _ => Err(tch::TchError::Torch(
                format!("Unsupported output type: {:?}", ivalue)
            )),
        }
    }
}

impl Model for TorchScriptModel {
    type Backend = TchBackend;
    type Input = TchTensor;
    type Output = TchTensor;
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output, tch::TchError> {
        let output = self.module.forward_ts(&[input.0])?;
        Ok(TchTensor(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};
    
    #[test]
    fn test_torchscript_model_creation() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0".to_string(),
            description: "Test TorchScript model".to_string(),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            tags: vec!["test".to_string()],
            custom: Default::default(),
        };
        
        assert_eq!(metadata.name, "test_model");
    }
    
    #[test]
    fn test_default_metadata() {
        let metadata = TorchScriptModel::default_metadata("my_model");
        assert_eq!(metadata.name, "my_model");
        assert!(metadata.tags.contains(&"torchscript".to_string()));
    }
}
