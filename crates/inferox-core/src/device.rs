#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceId {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

pub trait Device: Clone + Send + Sync {
    fn id(&self) -> DeviceId;
    
    fn is_available(&self) -> bool;
    
    fn memory_info(&self) -> Option<MemoryInfo>;
}
