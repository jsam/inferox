pub trait DataType: Copy + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn size(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
    BF16,
    F16,
}

impl DataType for DType {
    fn name(&self) -> &str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
        }
    }

    fn size(&self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 => 1,
        }
    }
}

pub trait NumericType: Copy + Send + Sync + 'static {
    fn dtype() -> DType;
    fn as_f32_slice(data: &[Self]) -> Vec<f32>;
}

impl NumericType for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn as_f32_slice(data: &[Self]) -> Vec<f32> {
        data.to_vec()
    }
}

impl NumericType for f64 {
    fn dtype() -> DType {
        DType::F64
    }

    fn as_f32_slice(data: &[Self]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }
}

impl NumericType for i32 {
    fn dtype() -> DType {
        DType::I32
    }

    fn as_f32_slice(data: &[Self]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }
}

impl NumericType for i64 {
    fn dtype() -> DType {
        DType::I64
    }

    fn as_f32_slice(data: &[Self]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }
}

impl NumericType for u8 {
    fn dtype() -> DType {
        DType::U8
    }

    fn as_f32_slice(data: &[Self]) -> Vec<f32> {
        data.iter().map(|&x| x as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_names() {
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F64.name(), "f64");
        assert_eq!(DType::I32.name(), "i32");
        assert_eq!(DType::I64.name(), "i64");
        assert_eq!(DType::U8.name(), "u8");
        assert_eq!(DType::BF16.name(), "bf16");
        assert_eq!(DType::F16.name(), "f16");
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F64.size(), 8);
        assert_eq!(DType::I64.size(), 8);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::BF16.size(), 2);
        assert_eq!(DType::U8.size(), 1);
    }

    #[test]
    fn test_numeric_type_dtype() {
        assert_eq!(f32::dtype(), DType::F32);
        assert_eq!(f64::dtype(), DType::F64);
        assert_eq!(i32::dtype(), DType::I32);
        assert_eq!(i64::dtype(), DType::I64);
        assert_eq!(u8::dtype(), DType::U8);
    }

    #[test]
    fn test_dtype_equality() {
        assert_eq!(DType::F32, DType::F32);
        assert_ne!(DType::F32, DType::F64);
    }

    #[test]
    fn test_as_f32_slice_f64() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = f64::as_f32_slice(&data);
        assert_eq!(result, vec![1.0f32, 2.0f32, 3.0f32]);
    }

    #[test]
    fn test_as_f32_slice_i32() {
        let data: Vec<i32> = vec![1, 2, 3];
        let result = i32::as_f32_slice(&data);
        assert_eq!(result, vec![1.0f32, 2.0f32, 3.0f32]);
    }

    #[test]
    fn test_as_f32_slice_i64() {
        let data: Vec<i64> = vec![1, 2, 3];
        let result = i64::as_f32_slice(&data);
        assert_eq!(result, vec![1.0f32, 2.0f32, 3.0f32]);
    }

    #[test]
    fn test_as_f32_slice_u8() {
        let data: Vec<u8> = vec![1, 2, 3];
        let result = u8::as_f32_slice(&data);
        assert_eq!(result, vec![1.0f32, 2.0f32, 3.0f32]);
    }
}
