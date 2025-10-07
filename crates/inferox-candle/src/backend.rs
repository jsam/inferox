use candle_core::Device as CandleDevice;
use inferox_core::Backend;
use crate::{CandleDeviceWrapper, CandleTensorBuilder, CandleTensor};

#[derive(Clone)]
pub struct CandleBackend {
    device: CandleDevice,
}

impl CandleBackend {
    pub fn new() -> Result<Self, candle_core::Error> {
        let device = CandleDevice::cuda_if_available(0)?;
        Ok(Self { device })
    }
    
    pub fn with_device(device: CandleDevice) -> Self {
        Self { device }
    }
    
    pub fn cpu() -> Result<Self, candle_core::Error> {
        Ok(Self {
            device: CandleDevice::Cpu,
        })
    }
}

impl Backend for CandleBackend {
    type Tensor = CandleTensor;
    type Error = candle_core::Error;
    type Device = CandleDeviceWrapper;
    type TensorBuilder = CandleTensorBuilder;
    
    fn name(&self) -> &str {
        "candle"
    }
    
    fn devices(&self) -> Result<Vec<Self::Device>, Self::Error> {
        let devices = vec![CandleDeviceWrapper(CandleDevice::Cpu)];
        
        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                for i in 0..candle_core::utils::get_num_threads() {
                    if let Ok(device) = CandleDevice::new_cuda(i) {
                        devices.push(CandleDeviceWrapper(device));
                    }
                }
            }
        }
        
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                if let Ok(device) = CandleDevice::new_metal(0) {
                    devices.push(CandleDeviceWrapper(device));
                }
            }
        }
        
        Ok(devices)
    }
    
    fn default_device(&self) -> Self::Device {
        CandleDeviceWrapper(self.device.clone())
    }
    
    fn tensor_builder(&self) -> Self::TensorBuilder {
        CandleTensorBuilder {
            device: self.device.clone(),
        }
    }
}

