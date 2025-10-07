use candle_core::{Device as CandleDevice, Tensor as InternalTensor, DType as CandleDType};
use inferox_core::{DataType, DType, NumericType, TensorBuilder};
use crate::{CandleBackend, CandleDeviceWrapper, CandleTensor};

pub struct CandleTensorBuilder {
    pub(crate) device: CandleDevice,
}

impl TensorBuilder<CandleBackend> for CandleTensorBuilder {
    fn from_slice<T: NumericType>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> Result<CandleTensor, candle_core::Error> {
        let data_f32 = T::as_f32_slice(data);
        let tensor = InternalTensor::from_vec(data_f32, shape, &self.device)?;
        Ok(CandleTensor(tensor))
    }
    
    fn from_vec<T: NumericType>(
        &self,
        data: Vec<T>,
        shape: &[usize],
    ) -> Result<CandleTensor, candle_core::Error> {
        self.from_slice(&data, shape)
    }
    
    fn zeros(&self, shape: &[usize], dtype: impl DataType) -> Result<CandleTensor, candle_core::Error> {
        let dtype_str = dtype.name();
        let candle_dtype = match dtype_str {
            "f32" => CandleDType::F32,
            "f64" => CandleDType::F64,
            "i32" => CandleDType::U32,
            "i64" => CandleDType::I64,
            "u8" => CandleDType::U8,
            "bf16" => CandleDType::BF16,
            "f16" => CandleDType::F16,
            _ => CandleDType::F32,
        };
        let tensor = InternalTensor::zeros(shape, candle_dtype, &self.device)?;
        Ok(CandleTensor(tensor))
    }
    
    fn ones(&self, shape: &[usize], dtype: impl DataType) -> Result<CandleTensor, candle_core::Error> {
        let dtype_str = dtype.name();
        let candle_dtype = match dtype_str {
            "f32" => CandleDType::F32,
            "f64" => CandleDType::F64,
            "i32" => CandleDType::U32,
            "i64" => CandleDType::I64,
            "u8" => CandleDType::U8,
            "bf16" => CandleDType::BF16,
            "f16" => CandleDType::F16,
            _ => CandleDType::F32,
        };
        let tensor = InternalTensor::ones(shape, candle_dtype, &self.device)?;
        Ok(CandleTensor(tensor))
    }
    
    fn randn(&self, shape: &[usize], dtype: impl DataType) -> Result<CandleTensor, candle_core::Error> {
        let dtype_str = dtype.name();
        let candle_dtype = match dtype_str {
            "f32" => CandleDType::F32,
            "f64" => CandleDType::F64,
            _ => CandleDType::F32,
        };
        let tensor = InternalTensor::randn(0.0, 1.0, shape, &self.device)?.to_dtype(candle_dtype)?;
        Ok(CandleTensor(tensor))
    }
    
    fn with_device(mut self, device: CandleDeviceWrapper) -> Self {
        self.device = device.0;
        self
    }
}

pub fn map_dtype(dtype: DType) -> CandleDType {
    match dtype {
        DType::F32 => CandleDType::F32,
        DType::F64 => CandleDType::F64,
        DType::I32 => CandleDType::U32,
        DType::I64 => CandleDType::I64,
        DType::U8 => CandleDType::U8,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
    }
}
