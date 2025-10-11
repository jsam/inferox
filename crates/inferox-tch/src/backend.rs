use crate::{TchDeviceWrapper, TchTensor, TchTensorBuilder};
use inferox_core::Backend;
use tch::Device as TchDevice;

#[derive(Clone)]
pub struct TchBackend {
    device: TchDevice,
}

impl TchBackend {
    pub fn new() -> Result<Self, tch::TchError> {
        let device = if tch::Cuda::is_available() {
            TchDevice::Cuda(0)
        } else {
            TchDevice::Cpu
        };
        Ok(Self { device })
    }

    pub fn with_device(device: TchDevice) -> Self {
        Self { device }
    }

    pub fn cpu() -> Result<Self, tch::TchError> {
        Ok(Self {
            device: TchDevice::Cpu,
        })
    }

    pub fn cuda(ordinal: usize) -> Result<Self, tch::TchError> {
        if !tch::Cuda::is_available() {
            return Err(tch::TchError::Torch("CUDA is not available".to_string()));
        }
        Ok(Self {
            device: TchDevice::Cuda(ordinal),
        })
    }
}

impl Backend for TchBackend {
    type Tensor = TchTensor;
    type Error = tch::TchError;
    type Device = TchDeviceWrapper;
    type TensorBuilder = TchTensorBuilder;

    fn name(&self) -> &str {
        "tch"
    }

    fn devices(&self) -> Result<Vec<Self::Device>, Self::Error> {
        let mut devices = vec![TchDeviceWrapper(TchDevice::Cpu)];

        if tch::Cuda::is_available() {
            let device_count = tch::Cuda::device_count();
            for i in 0..device_count {
                devices.push(TchDeviceWrapper(TchDevice::Cuda(i as usize)));
            }
        }

        Ok(devices)
    }

    fn default_device(&self) -> Self::Device {
        TchDeviceWrapper(self.device)
    }

    fn tensor_builder(&self) -> Self::TensorBuilder {
        TchTensorBuilder {
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::{Backend, Device};

    #[test]
    fn test_cpu_backend() {
        let backend = TchBackend::cpu().unwrap();
        assert_eq!(backend.name(), "tch");
    }

    #[test]
    fn test_backend_name() {
        let backend = TchBackend::cpu().unwrap();
        assert_eq!(backend.name(), "tch");
    }

    #[test]
    fn test_backend_devices() {
        let backend = TchBackend::cpu().unwrap();
        let devices = backend.devices().unwrap();
        assert!(!devices.is_empty());
        assert!(devices[0].is_available());
    }

    #[test]
    fn test_backend_default_device() {
        let backend = TchBackend::cpu().unwrap();
        let device = backend.default_device();
        assert!(device.is_available());
    }

    #[test]
    fn test_backend_tensor_builder() {
        let backend = TchBackend::cpu().unwrap();
        let _builder = backend.tensor_builder();
    }

    #[test]
    fn test_with_device() {
        let backend = TchBackend::with_device(TchDevice::Cpu);
        assert_eq!(backend.name(), "tch");
    }
}
