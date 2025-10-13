use inferox_core::{Model, ModelMetadata};
use inferox_tch::{TchBackend, TchTensor, TorchScriptModel};
use std::path::Path;
use tch::Device;

pub struct DocLayoutYOLO {
    model: TorchScriptModel,
}

impl DocLayoutYOLO {
    pub fn from_pretrained(
        model_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, tch::TchError> {
        let metadata = ModelMetadata {
            name: "doclayout-yolo".to_string(),
            version: "1.0".to_string(),
            description: "DocLayout-YOLO: Document layout detection using YOLO architecture".to_string(),
            author: "juliozhao".to_string(),
            license: "Apache-2.0".to_string(),
            tags: vec![
                "yolo".to_string(),
                "document".to_string(),
                "layout".to_string(),
                "detection".to_string(),
                "torchscript".to_string(),
            ],
            custom: Default::default(),
        };
        
        let model = TorchScriptModel::load_with_metadata(
            model_path,
            device,
            metadata,
        )?;
        
        Ok(Self { model })
    }
    
    pub fn detect(&self, image: TchTensor) -> Result<TchTensor, tch::TchError> {
        self.model.forward(image)
    }
    
    pub fn detect_batch(&self, images: Vec<TchTensor>) -> Result<Vec<TchTensor>, tch::TchError> {
        self.model.forward_batch(images)
    }
}

impl Model for DocLayoutYOLO {
    type Backend = TchBackend;
    type Input = TchTensor;
    type Output = TchTensor;
    
    fn name(&self) -> &str {
        self.model.name()
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.model.metadata()
    }
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output, tch::TchError> {
        self.model.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::Model;
    
    #[test]
    fn test_doclayout_yolo_metadata() {
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/doclayout_yolo_docstructbench_imgsz1024.pt");
        
        if !model_path.exists() {
            eprintln!("Model not found at {:?}, skipping test", model_path);
            return;
        }
        
        let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
            .expect("Failed to load model");
        
        let metadata = model.metadata();
        assert_eq!(metadata.name, "doclayout-yolo");
        assert_eq!(metadata.license, "Apache-2.0");
        assert!(metadata.tags.contains(&"yolo".to_string()));
        assert!(metadata.tags.contains(&"document".to_string()));
    }
}
