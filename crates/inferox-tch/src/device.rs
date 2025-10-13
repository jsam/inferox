use inferox_core::{Device, DeviceId, MemoryInfo};
use tch::Device as TchDevice;

#[derive(Clone, Copy)]
pub struct TchDeviceWrapper(pub(crate) TchDevice);

impl Device for TchDeviceWrapper {
    fn id(&self) -> DeviceId {
        match self.0 {
            TchDevice::Cpu => DeviceId::Cpu,
            TchDevice::Cuda(ordinal) => DeviceId::Cuda(ordinal),
            TchDevice::Mps => DeviceId::Metal(0),
            TchDevice::Vulkan => DeviceId::Cpu,
        }
    }

    fn is_available(&self) -> bool {
        match self.0 {
            TchDevice::Cpu => true,
            TchDevice::Cuda(_) => tch::Cuda::is_available(),
            TchDevice::Mps => false,
            TchDevice::Vulkan => false,
        }
    }

    fn memory_info(&self) -> Option<MemoryInfo> {
        None
    }
}

impl From<TchDevice> for TchDeviceWrapper {
    fn from(device: TchDevice) -> Self {
        Self(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::Device;

    #[test]
    fn test_cpu_device_id() {
        let device = TchDeviceWrapper(TchDevice::Cpu);
        assert_eq!(device.id(), DeviceId::Cpu);
    }

    #[test]
    fn test_device_is_available() {
        let device = TchDeviceWrapper(TchDevice::Cpu);
        assert!(device.is_available());
    }

    #[test]
    fn test_device_memory_info() {
        let device = TchDeviceWrapper(TchDevice::Cpu);
        assert!(device.memory_info().is_none());
    }

    #[test]
    fn test_device_from_tch_device() {
        let tch_device = TchDevice::Cpu;
        let device: TchDeviceWrapper = tch_device.into();
        assert_eq!(device.id(), DeviceId::Cpu);
    }

    #[test]
    fn test_cuda_device_id() {
        let device = TchDeviceWrapper(TchDevice::Cuda(0));
        assert_eq!(device.id(), DeviceId::Cuda(0));
    }

    #[test]
    fn test_mps_device() {
        let device = TchDeviceWrapper(TchDevice::Mps);
        assert_eq!(device.id(), DeviceId::Metal(0));
        assert!(!device.is_available());
    }

    #[test]
    fn test_vulkan_device() {
        let device = TchDeviceWrapper(TchDevice::Vulkan);
        assert_eq!(device.id(), DeviceId::Cpu);
        assert!(!device.is_available());
    }
}
