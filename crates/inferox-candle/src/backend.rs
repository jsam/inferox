use crate::{CandleDeviceWrapper, CandleTensor, CandleTensorBuilder};
use candle_core::Device as CandleDevice;
use inferox_core::Backend;

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

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::{Backend, Device};

    #[test]
    fn test_cpu_backend() {
        let backend = CandleBackend::cpu().unwrap();
        assert_eq!(backend.name(), "candle");
    }

    #[test]
    fn test_backend_name() {
        let backend = CandleBackend::cpu().unwrap();
        assert_eq!(backend.name(), "candle");
    }

    #[test]
    fn test_backend_devices() {
        let backend = CandleBackend::cpu().unwrap();
        let devices = backend.devices().unwrap();
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_backend_default_device() {
        let backend = CandleBackend::cpu().unwrap();
        let device = backend.default_device();
        assert!(device.is_available());
    }

    #[test]
    fn test_backend_tensor_builder() {
        let backend = CandleBackend::cpu().unwrap();
        let _builder = backend.tensor_builder();
    }

    #[test]
    fn test_with_device() {
        let backend = CandleBackend::with_device(CandleDevice::Cpu);
        assert_eq!(backend.name(), "candle");
    }
}
