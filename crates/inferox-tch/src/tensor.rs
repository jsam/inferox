use crate::{TchBackend, TchDeviceWrapper};
use inferox_core::{DataType, Tensor as TensorTrait};
use tch::{Kind as TchKind, Tensor as InternalTensor};

pub struct TchTensor(pub InternalTensor);

impl Clone for TchTensor {
    fn clone(&self) -> Self {
        TchTensor(self.0.shallow_clone())
    }
}

unsafe impl Send for TchTensor {}
unsafe impl Sync for TchTensor {}

#[derive(Clone, Copy)]
pub struct TchDTypeWrapper(pub TchKind);

impl DataType for TchDTypeWrapper {
    fn name(&self) -> &str {
        match self.0 {
            TchKind::Float => "f32",
            TchKind::Double => "f64",
            TchKind::Int => "i32",
            TchKind::Int64 => "i64",
            TchKind::Uint8 => "u8",
            TchKind::Int8 => "i8",
            TchKind::Int16 => "i16",
            TchKind::Half => "f16",
            TchKind::BFloat16 => "bf16",
            TchKind::Bool => "bool",
            _ => "unknown",
        }
    }

    fn size(&self) -> usize {
        match self.0 {
            TchKind::Double | TchKind::Int64 => 8,
            TchKind::Float | TchKind::Int => 4,
            TchKind::Half | TchKind::BFloat16 | TchKind::Int16 => 2,
            TchKind::Uint8 | TchKind::Int8 | TchKind::Bool => 1,
            _ => 0,
        }
    }
}

impl TensorTrait for TchTensor {
    type Dtype = TchDTypeWrapper;
    type Backend = TchBackend;

    fn shape(&self) -> &[usize] {
        let size = self.0.size();
        Box::leak(
            size.into_iter()
                .map(|x| x as usize)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    fn dtype(&self) -> Self::Dtype {
        TchDTypeWrapper(self.0.kind())
    }

    fn device(&self) -> TchDeviceWrapper {
        TchDeviceWrapper(self.0.device())
    }

    fn to_device(
        &self,
        device: &TchDeviceWrapper,
    ) -> Result<Self, <Self::Backend as inferox_core::Backend>::Error> {
        Ok(TchTensor(self.0.to_device(device.0)))
    }

    fn reshape(
        &self,
        shape: &[usize],
    ) -> Result<Self, <Self::Backend as inferox_core::Backend>::Error> {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        Ok(TchTensor(self.0.reshape(&shape_i64)))
    }

    fn contiguous(&self) -> Result<Self, <Self::Backend as inferox_core::Backend>::Error> {
        Ok(TchTensor(self.0.contiguous()))
    }
}

impl From<InternalTensor> for TchTensor {
    fn from(tensor: InternalTensor) -> Self {
        Self(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::{Device, Tensor};

    #[test]
    fn test_tensor_shape() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let shape = tensor.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_tensor_dtype() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let dtype = tensor.dtype();
        assert_eq!(dtype.name(), "f32");
        assert_eq!(dtype.size(), 4);
    }

    #[test]
    fn test_tensor_device() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let device = tensor.device();
        assert_eq!(device.id(), inferox_core::DeviceId::Cpu);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        let shape = reshaped.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_tensor_from_internal() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor: TchTensor = t.into();
        let shape = tensor.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_tensor_clone() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let cloned = tensor.clone();
        assert_eq!(cloned.shape(), tensor.shape());
    }

    #[test]
    fn test_tensor_to_device() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let device = TchDeviceWrapper(tch::Device::Cpu);
        let moved = tensor.to_device(&device).unwrap();
        assert_eq!(moved.device().id(), inferox_core::DeviceId::Cpu);
    }

    #[test]
    fn test_tensor_contiguous() {
        let t = tch::Tensor::zeros(&[2, 3], (TchKind::Float, tch::Device::Cpu));
        let tensor = TchTensor(t);
        let contiguous = tensor.contiguous().unwrap();
        assert_eq!(contiguous.shape(), tensor.shape());
    }

    #[test]
    fn test_dtype_wrapper_all_types() {
        assert_eq!(TchDTypeWrapper(TchKind::Double).name(), "f64");
        assert_eq!(TchDTypeWrapper(TchKind::Int).name(), "i32");
        assert_eq!(TchDTypeWrapper(TchKind::Int64).name(), "i64");
        assert_eq!(TchDTypeWrapper(TchKind::Uint8).name(), "u8");
        assert_eq!(TchDTypeWrapper(TchKind::Int8).name(), "i8");
        assert_eq!(TchDTypeWrapper(TchKind::Int16).name(), "i16");
        assert_eq!(TchDTypeWrapper(TchKind::Half).name(), "f16");
        assert_eq!(TchDTypeWrapper(TchKind::BFloat16).name(), "bf16");
        assert_eq!(TchDTypeWrapper(TchKind::Bool).name(), "bool");
    }

    #[test]
    fn test_dtype_wrapper_sizes() {
        assert_eq!(TchDTypeWrapper(TchKind::Double).size(), 8);
        assert_eq!(TchDTypeWrapper(TchKind::Int64).size(), 8);
        assert_eq!(TchDTypeWrapper(TchKind::Float).size(), 4);
        assert_eq!(TchDTypeWrapper(TchKind::Int).size(), 4);
        assert_eq!(TchDTypeWrapper(TchKind::Half).size(), 2);
        assert_eq!(TchDTypeWrapper(TchKind::BFloat16).size(), 2);
        assert_eq!(TchDTypeWrapper(TchKind::Int16).size(), 2);
        assert_eq!(TchDTypeWrapper(TchKind::Uint8).size(), 1);
        assert_eq!(TchDTypeWrapper(TchKind::Int8).size(), 1);
        assert_eq!(TchDTypeWrapper(TchKind::Bool).size(), 1);
    }
}
