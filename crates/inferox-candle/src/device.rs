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
