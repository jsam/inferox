use crate::{Device, Tensor, TensorBuilder};

pub trait Backend: Send + Sync + 'static {
    type Tensor: Tensor;

    type Error: std::error::Error + Send + Sync + 'static;

    type Device: Device;

    type TensorBuilder: TensorBuilder<Self>;

    fn name(&self) -> &str;

    fn devices(&self) -> Result<Vec<Self::Device>, Self::Error>;

    fn default_device(&self) -> Self::Device;

    fn tensor_builder(&self) -> Self::TensorBuilder;
}
