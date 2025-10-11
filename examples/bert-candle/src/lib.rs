use candle_core::{DType, Device, Tensor as InternalTensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use inferox_candle::{CandleBackend, CandleTensor};
use inferox_core::{Model, ModelMetadata};
use std::path::PathBuf;

pub struct BertModelWrapper {
    name: String,
    inner: BertModel,
}

impl Model for BertModelWrapper {
    type Backend = CandleBackend;
    type Input = CandleTensor;
    type Output = CandleTensor;

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: self.name.clone(),
            version: "1.0".to_string(),
            description: "BERT model loaded from Hugging Face".to_string(),
            author: "Inferox".to_string(),
            license: "MIT".to_string(),
            tags: vec!["bert".to_string(), "transformer".to_string()],
            custom: Default::default(),
        }
    }

    fn forward(&self, input: Self::Input) -> Result<Self::Output, candle_core::Error> {
        let input_tensor: InternalTensor = input.into();

        let input_tensor = input_tensor.to_dtype(DType::U32)?;

        let token_type_ids = input_tensor.zeros_like()?;

        let output = self.inner.forward(&input_tensor, &token_type_ids, None)?;

        Ok(CandleTensor::from(output))
    }
}

type BoxedCandleModel =
    Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>>;

#[no_mangle]
pub fn create_model() -> BoxedCandleModel {
    let package_dir = std::env::var("INFEROX_PACKAGE_DIR")
        .expect("INFEROX_PACKAGE_DIR environment variable not set");

    let package_path = PathBuf::from(package_dir);
    let config_path = package_path.join("config.json");
    let weights_path = package_path.join("model.safetensors");

    let config_str = std::fs::read_to_string(&config_path).expect("Failed to read config.json");
    let config: Config = serde_json::from_str(&config_str).expect("Failed to parse config.json");

    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            .expect("Failed to load weights from safetensors")
    };

    let vb = vb.rename_f(|name| {
        let prefixed = format!("bert.{}", name);
        if prefixed.contains("LayerNorm") {
            prefixed.replace("weight", "gamma").replace("bias", "beta")
        } else {
            prefixed
        }
    });

    let bert_model = BertModel::load(vb, &config).expect("Failed to load BERT model");

    let wrapper = BertModelWrapper {
        name: "bert".to_string(),
        inner: bert_model,
    };

    Box::new(wrapper)
}
