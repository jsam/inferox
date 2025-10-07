use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use crate::CandleVarMap;

pub struct CandleModelBuilder {
    var_map: CandleVarMap,
    device: Device,
    dtype: DType,
}

impl CandleModelBuilder {
    pub fn new(device: Device) -> Self {
        Self {
            var_map: CandleVarMap::new(),
            device,
            dtype: DType::F32,
        }
    }
    
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    
    pub fn load_weights<P: AsRef<std::path::Path>>(mut self, path: P) 
        -> Result<Self, candle_core::Error> 
    {
        self.var_map.load(path)?;
        Ok(self)
    }
    
    pub fn var_builder(&self) -> VarBuilder {
        self.var_map.var_builder(self.dtype, &self.device)
    }
    
    pub fn var_map(&self) -> &CandleVarMap {
        &self.var_map
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
