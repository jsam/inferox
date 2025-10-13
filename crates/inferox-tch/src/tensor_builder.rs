use crate::{TchBackend, TchDeviceWrapper, TchTensor};
use inferox_core::{DataType, NumericType, TensorBuilder};
use tch::{Device as TchDevice, Kind as TchKind, Tensor as InternalTensor};

pub struct TchTensorBuilder {
    pub(crate) device: TchDevice,
}

impl TensorBuilder<TchBackend> for TchTensorBuilder {
    fn build_from_slice<T: NumericType>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> Result<TchTensor, tch::TchError> {
        let data_f32 = T::as_f32_slice(data);
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = InternalTensor::from_slice(&data_f32)
            .reshape(&shape_i64)
            .to_device(self.device);
        Ok(TchTensor(tensor))
    }

    fn build_from_vec<T: NumericType>(
        &self,
        data: Vec<T>,
        shape: &[usize],
    ) -> Result<TchTensor, tch::TchError> {
        self.build_from_slice(&data, shape)
    }

    fn zeros(&self, shape: &[usize], dtype: impl DataType) -> Result<TchTensor, tch::TchError> {
        let dtype_str = dtype.name();
        let tch_kind = match dtype_str {
            "f32" => TchKind::Float,
            "f64" => TchKind::Double,
            "i32" => TchKind::Int,
            "i64" => TchKind::Int64,
            "u8" => TchKind::Uint8,
            "i8" => TchKind::Int8,
            "i16" => TchKind::Int16,
            "bf16" => TchKind::BFloat16,
            "f16" => TchKind::Half,
            "bool" => TchKind::Bool,
            _ => TchKind::Float,
        };
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = InternalTensor::zeros(&shape_i64, (tch_kind, self.device));
        Ok(TchTensor(tensor))
    }

    fn ones(&self, shape: &[usize], dtype: impl DataType) -> Result<TchTensor, tch::TchError> {
        let dtype_str = dtype.name();
        let tch_kind = match dtype_str {
            "f32" => TchKind::Float,
            "f64" => TchKind::Double,
            "i32" => TchKind::Int,
            "i64" => TchKind::Int64,
            "u8" => TchKind::Uint8,
            "i8" => TchKind::Int8,
            "i16" => TchKind::Int16,
            "bf16" => TchKind::BFloat16,
            "f16" => TchKind::Half,
            "bool" => TchKind::Bool,
            _ => TchKind::Float,
        };
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = InternalTensor::ones(&shape_i64, (tch_kind, self.device));
        Ok(TchTensor(tensor))
    }

    fn randn(&self, shape: &[usize], dtype: impl DataType) -> Result<TchTensor, tch::TchError> {
        let dtype_str = dtype.name();
        let tch_kind = match dtype_str {
            "f32" => TchKind::Float,
            "f64" => TchKind::Double,
            _ => TchKind::Float,
        };
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = InternalTensor::randn(&shape_i64, (tch_kind, self.device));
        Ok(TchTensor(tensor))
    }

    fn with_device(mut self, device: TchDeviceWrapper) -> Self {
        self.device = device.0;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::{DType, Tensor, TensorBuilder};

    #[test]
    fn test_build_from_slice() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = builder.build_from_slice(&data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_build_from_vec() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = builder.build_from_vec(data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_zeros() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.zeros(&[2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_zeros_f64() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.zeros(&[2, 3], DType::F64).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_zeros_i64() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.zeros(&[2, 3], DType::I64).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_zeros_u8() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.zeros(&[2, 3], DType::U8).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_ones() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.ones(&[2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_ones_f64() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.ones(&[2, 3], DType::F64).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_randn() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.randn(&[2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_randn_f64() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let tensor = builder.randn(&[2, 3], DType::F64).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }

    #[test]
    fn test_with_device() {
        let builder = TchTensorBuilder {
            device: TchDevice::Cpu,
        };
        let device_wrapper = TchDeviceWrapper(TchDevice::Cpu);
        let new_builder = builder.with_device(device_wrapper);
        let tensor = new_builder.zeros(&[2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape().len(), 2);
    }
}
