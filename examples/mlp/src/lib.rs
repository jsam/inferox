use candle_nn::{Linear, Module, VarBuilder};
use inferox_candle::{CandleBackend, CandleTensor};
use inferox_core::{Model, ModelMetadata};

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    name: String,
}

impl MLP {
    pub fn new(
        name: &str,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, candle_core::Error> {
        Ok(Self {
            fc1: candle_nn::linear(input_dim, hidden_dim, vb.pp("fc1"))?,
            fc2: candle_nn::linear(hidden_dim, hidden_dim, vb.pp("fc2"))?,
            fc3: candle_nn::linear(hidden_dim, output_dim, vb.pp("fc3"))?,
            input_dim,
            hidden_dim,
            output_dim,
            name: name.to_string(),
        })
    }
}

impl Model for MLP {
    type Backend = CandleBackend;
    type Input = CandleTensor;
    type Output = CandleTensor;

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: Self::Input) -> Result<Self::Output, candle_core::Error> {
        let x = self.fc1.forward(&input.0)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc3.forward(&x)?;
        Ok(CandleTensor(x))
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: "MLP".to_string(),
            version: "1.0.0".to_string(),
            description: format!(
                "Multi-Layer Perceptron ({} → {} → {} → {})",
                self.input_dim, self.hidden_dim, self.hidden_dim, self.output_dim
            ),
            author: "Inferox Contributors".to_string(),
            license: "MIT OR Apache-2.0".to_string(),
            tags: vec!["neural-network".to_string(), "classification".to_string()],
            custom: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use inferox_candle::CandleModelBuilder;
    use inferox_core::{Backend, Tensor, TensorBuilder};

    #[test]
    fn test_mlp_creation() {
        let builder = CandleModelBuilder::new(Device::Cpu);
        let vb = builder.var_builder();

        let model = MLP::new("test", 10, 20, 5, vb).unwrap();
        assert_eq!(model.name(), "test");
    }

    #[test]
    fn test_mlp_forward() {
        let backend = inferox_candle::CandleBackend::cpu().unwrap();
        let builder = CandleModelBuilder::new(Device::Cpu);
        let vb = builder.var_builder();

        let model = MLP::new("test", 10, 20, 5, vb).unwrap();

        let input = backend
            .tensor_builder()
            .build_from_vec(vec![1.0f32; 10], &[1, 10])
            .unwrap();

        let output = model.forward(input).unwrap();
        assert_eq!(output.shape(), &[1, 5]);
    }
}
