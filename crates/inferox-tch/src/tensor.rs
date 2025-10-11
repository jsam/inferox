use crate::{TchBackend, TchDeviceWrapper};
use inferox_core::{DataType, Tensor as TensorTrait};
use tch::{Kind as TchKind, Tensor as InternalTensor};

#[derive(Clone)]
pub struct TchTensor(pub InternalTensor);

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
    use inferox_core::Tensor;

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
}
