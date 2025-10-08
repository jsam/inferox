use crate::{CandleBackend, CandleDeviceWrapper};
use candle_core::{DType as CandleDType, Tensor as InternalTensor};
use inferox_core::{DataType, Tensor as TensorTrait};

#[derive(Clone)]
pub struct CandleTensor(pub InternalTensor);

#[derive(Clone, Copy)]
pub struct CandleDTypeWrapper(pub CandleDType);

impl DataType for CandleDTypeWrapper {
    fn name(&self) -> &str {
        match self.0 {
            CandleDType::F32 => "f32",
            CandleDType::F64 => "f64",
            CandleDType::U32 => "u32",
            CandleDType::I64 => "i64",
            CandleDType::U8 => "u8",
            CandleDType::BF16 => "bf16",
            CandleDType::F16 => "f16",
        }
    }

    fn size(&self) -> usize {
        match self.0 {
            CandleDType::F64 | CandleDType::I64 => 8,
            CandleDType::F32 | CandleDType::U32 => 4,
            CandleDType::F16 | CandleDType::BF16 => 2,
            CandleDType::U8 => 1,
        }
    }
}

impl TensorTrait for CandleTensor {
    type Dtype = CandleDTypeWrapper;
    type Backend = CandleBackend;

    fn shape(&self) -> &[usize] {
        self.0.dims()
    }

    fn dtype(&self) -> Self::Dtype {
        CandleDTypeWrapper(self.0.dtype())
    }

    fn device(&self) -> CandleDeviceWrapper {
        CandleDeviceWrapper(self.0.device().clone())
    }

    fn to_device(&self, device: &CandleDeviceWrapper) -> Result<Self, candle_core::Error> {
        Ok(CandleTensor(self.0.to_device(&device.0)?))
    }

    fn reshape(&self, shape: &[usize]) -> Result<Self, candle_core::Error> {
        Ok(CandleTensor(self.0.reshape(shape)?))
    }

    fn contiguous(&self) -> Result<Self, candle_core::Error> {
        Ok(CandleTensor(self.0.contiguous()?))
    }
}

impl From<InternalTensor> for CandleTensor {
    fn from(tensor: InternalTensor) -> Self {
        CandleTensor(tensor)
    }
}

impl From<CandleTensor> for InternalTensor {
    fn from(tensor: CandleTensor) -> Self {
        tensor.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use inferox_core::{DataType, Device as DeviceTrait, Tensor as TensorTrait};

    #[test]
    fn test_tensor_shape() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        assert_eq!(candle_tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_tensor_dtype() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let dtype = candle_tensor.dtype();
        assert_eq!(dtype.name(), "f32");
    }

    #[test]
    fn test_tensor_device() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let device = candle_tensor.device();
        assert_eq!(device.id(), inferox_core::DeviceId::Cpu);
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let reshaped = candle_tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
    }

    #[test]
    fn test_tensor_contiguous() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let contiguous = candle_tensor.contiguous().unwrap();
        assert_eq!(contiguous.shape(), &[2, 3]);
    }

    #[test]
    fn test_tensor_to_device() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let device = CandleDeviceWrapper(Device::Cpu);
        let moved = candle_tensor.to_device(&device).unwrap();
        assert_eq!(moved.shape(), &[2, 3]);
    }

    #[test]
    fn test_dtype_wrapper_name() {
        let dtype = CandleDTypeWrapper(CandleDType::F32);
        assert_eq!(dtype.name(), "f32");

        let dtype = CandleDTypeWrapper(CandleDType::F64);
        assert_eq!(dtype.name(), "f64");

        let dtype = CandleDTypeWrapper(CandleDType::U32);
        assert_eq!(dtype.name(), "u32");

        let dtype = CandleDTypeWrapper(CandleDType::I64);
        assert_eq!(dtype.name(), "i64");

        let dtype = CandleDTypeWrapper(CandleDType::U8);
        assert_eq!(dtype.name(), "u8");
    }

    #[test]
    fn test_dtype_wrapper_size() {
        let dtype = CandleDTypeWrapper(CandleDType::F64);
        assert_eq!(dtype.size(), 8);

        let dtype = CandleDTypeWrapper(CandleDType::I64);
        assert_eq!(dtype.size(), 8);

        let dtype = CandleDTypeWrapper(CandleDType::F32);
        assert_eq!(dtype.size(), 4);

        let dtype = CandleDTypeWrapper(CandleDType::U32);
        assert_eq!(dtype.size(), 4);

        let dtype = CandleDTypeWrapper(CandleDType::U8);
        assert_eq!(dtype.size(), 1);
    }

    #[test]
    fn test_tensor_from_internal() {
        let internal = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor: CandleTensor = internal.into();
        assert_eq!(candle_tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_internal_from_tensor() {
        let tensor = InternalTensor::zeros(&[2, 3], CandleDType::F32, &Device::Cpu).unwrap();
        let candle_tensor = CandleTensor(tensor);
        let internal: InternalTensor = candle_tensor.into();
        assert_eq!(internal.dims(), &[2, 3]);
    }
}
