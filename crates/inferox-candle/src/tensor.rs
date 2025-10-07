use candle_core::{Tensor as InternalTensor, DType as CandleDType};
use inferox_core::{Tensor as TensorTrait, DataType};
use crate::{CandleBackend, CandleDeviceWrapper};

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
