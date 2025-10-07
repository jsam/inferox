use crate::{Backend, DataType, NumericType};

pub trait Tensor: Clone + Send + Sync {
    type Dtype: DataType;
    type Backend: Backend;
    
    fn shape(&self) -> &[usize];
    
    fn dtype(&self) -> Self::Dtype;
    
    fn device(&self) -> <Self::Backend as Backend>::Device;
    
    fn to_device(&self, device: &<Self::Backend as Backend>::Device) 
        -> Result<Self, <Self::Backend as Backend>::Error>;
    
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    
    fn reshape(&self, shape: &[usize]) -> Result<Self, <Self::Backend as Backend>::Error>;
    
    fn contiguous(&self) -> Result<Self, <Self::Backend as Backend>::Error>;
}

pub trait TensorBuilder<B: Backend + ?Sized> {
    fn from_slice<T: NumericType>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> Result<B::Tensor, B::Error>;
    
    fn from_vec<T: NumericType>(
        &self,
        data: Vec<T>,
        shape: &[usize],
    ) -> Result<B::Tensor, B::Error>;
    
    fn zeros(&self, shape: &[usize], dtype: impl DataType) -> Result<B::Tensor, B::Error>;
    
    fn ones(&self, shape: &[usize], dtype: impl DataType) -> Result<B::Tensor, B::Error>;
    
    fn randn(&self, shape: &[usize], dtype: impl DataType) -> Result<B::Tensor, B::Error>;
    
    fn with_device(self, device: B::Device) -> Self;
}
