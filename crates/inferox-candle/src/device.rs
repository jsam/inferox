use candle_core::Device as CandleDevice;
use inferox_core::{Device, DeviceId, MemoryInfo};

#[derive(Clone)]
pub struct CandleDeviceWrapper(pub(crate) CandleDevice);

impl Device for CandleDeviceWrapper {
    fn id(&self) -> DeviceId {
        match &self.0 {
            CandleDevice::Cpu => DeviceId::Cpu,
            CandleDevice::Cuda(_) => DeviceId::Cuda(0),
            CandleDevice::Metal(_) => DeviceId::Metal(0),
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn memory_info(&self) -> Option<MemoryInfo> {
        None
    }
}

impl From<CandleDevice> for CandleDeviceWrapper {
    fn from(device: CandleDevice) -> Self {
        Self(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferox_core::Device;

    #[test]
    fn test_cpu_device_id() {
        let device = CandleDeviceWrapper(CandleDevice::Cpu);
        assert_eq!(device.id(), DeviceId::Cpu);
    }

    #[test]
    fn test_device_is_available() {
        let device = CandleDeviceWrapper(CandleDevice::Cpu);
        assert!(device.is_available());
    }

    #[test]
    fn test_device_memory_info() {
        let device = CandleDeviceWrapper(CandleDevice::Cpu);
        assert!(device.memory_info().is_none());
    }

    #[test]
    fn test_device_from_candle_device() {
        let candle_device = CandleDevice::Cpu;
        let device: CandleDeviceWrapper = candle_device.into();
        assert_eq!(device.id(), DeviceId::Cpu);
    }
}
