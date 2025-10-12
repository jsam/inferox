#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceId {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Custom(String),
}

impl DeviceId {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(DeviceId::Cpu),
            "cuda" => Ok(DeviceId::Cuda(0)),
            "mps" | "metal" => Ok(DeviceId::Metal(0)),
            s if s.starts_with("cuda:") => {
                let idx = s
                    .strip_prefix("cuda:")
                    .unwrap()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device: {}", s))?;
                Ok(DeviceId::Cuda(idx))
            }
            s if s.starts_with("metal:") => {
                let idx = s
                    .strip_prefix("metal:")
                    .unwrap()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid Metal device: {}", s))?;
                Ok(DeviceId::Metal(idx))
            }
            _ => Ok(DeviceId::Custom(s.to_string())),
        }
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Cpu => write!(f, "cpu"),
            DeviceId::Cuda(idx) => write!(f, "cuda:{}", idx),
            DeviceId::Metal(idx) => write!(f, "metal:{}", idx),
            DeviceId::Custom(s) => write!(f, "{}", s),
        }
    }
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
