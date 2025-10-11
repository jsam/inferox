use inferox_core::{Model, ModelMetadata};
use inferox_tch::{TchBackend, TchTensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug, Deserialize)]
struct BertConfig {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: i64,
    type_vocab_size: i64,
    vocab_size: i64,
    layer_norm_eps: f64,
}

pub struct BertModelWrapper {
    name: String,
    vs: nn::VarStore,
    config: BertConfig,
}

impl BertModelWrapper {
    fn load_from_safetensors(
        package_dir: &PathBuf,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = package_dir.join("config.json");
        let weights_path = package_dir.join("model.safetensors");

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;

        let mut vs = nn::VarStore::new(device);

        let weights_data = std::fs::read(&weights_path)?;
        let tensors = safetensors::SafeTensors::deserialize(&weights_data)?;

        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();

            let mapped_name = map_weight_name(name);

            let tensor_data = tensor_view.data();
            let dtype = match tensor_view.dtype() {
                safetensors::Dtype::F32 => Kind::Float,
                safetensors::Dtype::F64 => Kind::Double,
                safetensors::Dtype::I32 => Kind::Int,
                safetensors::Dtype::I64 => Kind::Int64,
                _ => Kind::Float,
            };

            let tensor = Tensor::from_data_size(
                tensor_data,
                &shape,
                dtype,
            )
            .to_device(device);

            vs.variables_
                .lock()
                .unwrap()
                .named_variables
                .insert(mapped_name, tensor);
        }

        Ok(Self {
            name: "bert".to_string(),
            vs,
            config,
        })
    }

    fn forward_impl(&self, input_ids: &Tensor) -> Result<Tensor, tch::TchError> {
        let variables = self.vs.variables_.lock().unwrap();
        
        let embeddings = self.get_embeddings(input_ids, &variables)?;
        
        let mut hidden_states = embeddings;
        for layer_idx in 0..self.config.num_hidden_layers {
            hidden_states = self.apply_layer(&hidden_states, layer_idx, &variables)?;
        }

        Ok(hidden_states)
    }

    fn get_embeddings(
        &self,
        input_ids: &Tensor,
        variables: &std::sync::MutexGuard<nn::Variables>,
    ) -> Result<Tensor, tch::TchError> {
        let word_embeddings = variables
            .named_variables
            .get("bert.embeddings.word_embeddings.weight")
            .ok_or_else(|| tch::TchError::FileFormat("Missing word embeddings".into()))?;

        let position_embeddings = variables
            .named_variables
            .get("bert.embeddings.position_embeddings.weight")
            .ok_or_else(|| tch::TchError::FileFormat("Missing position embeddings".into()))?;

        let token_type_embeddings = variables
            .named_variables
            .get("bert.embeddings.token_type_embeddings.weight")
            .ok_or_else(|| tch::TchError::FileFormat("Missing token type embeddings".into()))?;

        let seq_length = input_ids.size()[1];
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input_ids.device()))
            .unsqueeze(0);

        let token_type_ids = Tensor::zeros_like(input_ids);

        let word_embeds = word_embeddings.index_select(0, input_ids);
        let position_embeds = position_embeddings.index_select(0, &position_ids);
        let token_type_embeds = token_type_embeddings.index_select(0, &token_type_ids);

        let embeddings = word_embeds + position_embeds + token_type_embeds;

        let layer_norm_weight = variables
            .named_variables
            .get("bert.embeddings.LayerNorm.weight")
            .ok_or_else(|| tch::TchError::FileFormat("Missing LayerNorm weight".into()))?;

        let layer_norm_bias = variables
            .named_variables
            .get("bert.embeddings.LayerNorm.bias")
            .ok_or_else(|| tch::TchError::FileFormat("Missing LayerNorm bias".into()))?;

        let embeddings = embeddings.layer_norm(
            &[self.config.hidden_size],
            Some(layer_norm_weight),
            Some(layer_norm_bias),
            self.config.layer_norm_eps,
            true,
        );

        Ok(embeddings)
    }

    fn apply_layer(
        &self,
        hidden_states: &Tensor,
        layer_idx: i64,
        variables: &std::sync::MutexGuard<nn::Variables>,
    ) -> Result<Tensor, tch::TchError> {
        let prefix = format!("bert.encoder.layer.{}", layer_idx);

        let attention_output = self.apply_attention(hidden_states, &prefix, variables)?;

        let intermediate = self.apply_intermediate(&attention_output, &prefix, variables)?;

        let layer_output = self.apply_output(&intermediate, &attention_output, &prefix, variables)?;

        Ok(layer_output)
    }

    fn apply_attention(
        &self,
        hidden_states: &Tensor,
        prefix: &str,
        variables: &std::sync::MutexGuard<nn::Variables>,
    ) -> Result<Tensor, tch::TchError> {
        let query_weight = self.get_variable(variables, &format!("{}.attention.self.query.weight", prefix))?;
        let query_bias = self.get_variable(variables, &format!("{}.attention.self.query.bias", prefix))?;
        let key_weight = self.get_variable(variables, &format!("{}.attention.self.key.weight", prefix))?;
        let key_bias = self.get_variable(variables, &format!("{}.attention.self.key.bias", prefix))?;
        let value_weight = self.get_variable(variables, &format!("{}.attention.self.value.weight", prefix))?;
        let value_bias = self.get_variable(variables, &format!("{}.attention.self.value.bias", prefix))?;

        let query = hidden_states.linear(query_weight, Some(query_bias));
        let key = hidden_states.linear(key_weight, Some(key_bias));
        let value = hidden_states.linear(value_weight, Some(value_bias));

        let attention_head_size = self.config.hidden_size / self.config.num_attention_heads;
        let query = self.transpose_for_scores(&query, attention_head_size);
        let key = self.transpose_for_scores(&key, attention_head_size);
        let value = self.transpose_for_scores(&value, attention_head_size);

        let attention_scores = query.matmul(&key.transpose(-1, -2));
        let attention_scores = attention_scores / (attention_head_size as f64).sqrt();
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        let context = attention_probs.matmul(&value);
        let context = context.transpose(1, 2).contiguous();
        let context = context.view([
            context.size()[0],
            context.size()[1],
            self.config.hidden_size,
        ]);

        let dense_weight = self.get_variable(variables, &format!("{}.attention.output.dense.weight", prefix))?;
        let dense_bias = self.get_variable(variables, &format!("{}.attention.output.dense.bias", prefix))?;
        let attention_output = context.linear(dense_weight, Some(dense_bias));

        let layer_norm_weight = self.get_variable(variables, &format!("{}.attention.output.LayerNorm.weight", prefix))?;
        let layer_norm_bias = self.get_variable(variables, &format!("{}.attention.output.LayerNorm.bias", prefix))?;

        let attention_output = (attention_output + hidden_states).layer_norm(
            &[self.config.hidden_size],
            Some(layer_norm_weight),
            Some(layer_norm_bias),
            self.config.layer_norm_eps,
            true,
        );

        Ok(attention_output)
    }

    fn apply_intermediate(
        &self,
        hidden_states: &Tensor,
        prefix: &str,
        variables: &std::sync::MutexGuard<nn::Variables>,
    ) -> Result<Tensor, tch::TchError> {
        let dense_weight = self.get_variable(variables, &format!("{}.intermediate.dense.weight", prefix))?;
        let dense_bias = self.get_variable(variables, &format!("{}.intermediate.dense.bias", prefix))?;

        let intermediate = hidden_states.linear(dense_weight, Some(dense_bias));
        let intermediate = intermediate.gelu("none");

        Ok(intermediate)
    }

    fn apply_output(
        &self,
        intermediate: &Tensor,
        attention_output: &Tensor,
        prefix: &str,
        variables: &std::sync::MutexGuard<nn::Variables>,
    ) -> Result<Tensor, tch::TchError> {
        let dense_weight = self.get_variable(variables, &format!("{}.output.dense.weight", prefix))?;
        let dense_bias = self.get_variable(variables, &format!("{}.output.dense.bias", prefix))?;

        let output = intermediate.linear(dense_weight, Some(dense_bias));

        let layer_norm_weight = self.get_variable(variables, &format!("{}.output.LayerNorm.weight", prefix))?;
        let layer_norm_bias = self.get_variable(variables, &format!("{}.output.LayerNorm.bias", prefix))?;

        let output = (output + attention_output).layer_norm(
            &[self.config.hidden_size],
            Some(layer_norm_weight),
            Some(layer_norm_bias),
            self.config.layer_norm_eps,
            true,
        );

        Ok(output)
    }

    fn transpose_for_scores(&self, tensor: &Tensor, attention_head_size: i64) -> Tensor {
        let size = tensor.size();
        tensor
            .view([size[0], size[1], self.config.num_attention_heads, attention_head_size])
            .transpose(1, 2)
    }

    fn get_variable<'a>(
        &self,
        variables: &'a std::sync::MutexGuard<nn::Variables>,
        name: &str,
    ) -> Result<&'a Tensor, tch::TchError> {
        variables
            .named_variables
            .get(name)
            .ok_or_else(|| tch::TchError::FileFormat(format!("Missing variable: {}", name).into()))
    }
}

impl Model for BertModelWrapper {
    type Backend = TchBackend;
    type Input = TchTensor;
    type Output = TchTensor;

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            name: self.name.clone(),
            version: "1.0".to_string(),
            description: "BERT model loaded from Hugging Face using tch-rs".to_string(),
            author: "Inferox".to_string(),
            license: "MIT".to_string(),
            tags: vec!["bert".to_string(), "transformer".to_string(), "tch".to_string()],
            custom: Default::default(),
        }
    }

    fn forward(&self, input: Self::Input) -> Result<Self::Output, tch::TchError> {
        let input_tensor: Tensor = input.0;
        let input_tensor = input_tensor.to_kind(Kind::Int64);

        let output = self.forward_impl(&input_tensor)?;

        Ok(TchTensor(output))
    }
}

fn map_weight_name(name: &str) -> String {
    name.to_string()
}

type BoxedTchModel =
    Box<dyn Model<Backend = TchBackend, Input = TchTensor, Output = TchTensor>>;

#[no_mangle]
pub fn create_model() -> BoxedTchModel {
    let package_dir = std::env::var("INFEROX_PACKAGE_DIR")
        .expect("INFEROX_PACKAGE_DIR environment variable not set");

    let package_path = PathBuf::from(package_dir);

    let device = Device::Cpu;

    let model = BertModelWrapper::load_from_safetensors(&package_path, device)
        .expect("Failed to load BERT model from safetensors");

    Box::new(model)
}
