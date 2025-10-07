use inferox_candle::{CandleBackend, CandleModelBuilder, CandleTensor};
use inferox_core::Model;
use candle_core::Device;
use mlp::MLP;

#[no_mangle]
pub fn create_model() -> Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>> {
    let builder = CandleModelBuilder::new(Device::Cpu);
    let model = MLP::new("classifier", 10, 8, 3, builder.var_builder())
        .expect("Failed to create classifier model");
    Box::new(model)
}
