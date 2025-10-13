use inferox_candle::CandleTensor;
use inferox_core::Tensor;
use inferox_mlpkg::BackendType;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[cfg(feature = "tch")]
use inferox_tch::TchTensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTensorResponse {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub backend_used: BackendType,
    pub latency_ms: f64,
}

impl UnifiedTensorResponse {
    pub fn from_candle(tensor: CandleTensor, start_time: Instant) -> Result<Self, String> {
        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let shape = tensor.shape().to_vec();
        
        let candle_tensor: candle_core::Tensor = tensor.into();
        let flattened = candle_tensor
            .flatten_all()
            .map_err(|e| format!("Failed to flatten tensor: {}", e))?;
        
        let data = flattened
            .to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert to f32: {}", e))?;

        Ok(Self {
            shape,
            data,
            backend_used: BackendType::Candle,
            latency_ms,
        })
    }

    #[cfg(feature = "tch")]
    pub fn from_tch(tensor: TchTensor, start_time: Instant) -> Result<Self, String> {
        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let shape = tensor.shape().to_vec();
        
        let tch_tensor: tch::Tensor = tensor.into();
        let flattened = tch_tensor.flatten(0, -1);
        
        let data: Vec<f32> = flattened
            .to_kind(tch::Kind::Float)
            .try_into()
            .map_err(|e| format!("Failed to convert to Vec<f32>: {:?}", e))?;

        Ok(Self {
            shape,
            data,
            backend_used: BackendType::Tch,
            latency_ms,
        })
    }

    pub fn compare_with(&self, other: &Self, tolerance: f32) -> bool {
        if self.shape != other.shape {
            return false;
        }

        if self.data.len() != other.data.len() {
            return false;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() < tolerance)
    }

    pub fn max_difference(&self, other: &Self) -> Option<f32> {
        if self.shape != other.shape || self.data.len() != other.data.len() {
            return None;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }
}
