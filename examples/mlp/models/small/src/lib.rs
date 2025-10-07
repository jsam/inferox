use candle_core::Device;
use inferox_candle::{CandleBackend, CandleModelBuilder, CandleTensor};
use inferox_core::Model;
use mlp::MLP;

#[no_mangle]
pub fn create_model(
) -> Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>> {
    let builder = CandleModelBuilder::new(Device::Cpu);
    let model =
        MLP::new("small", 5, 4, 2, builder.var_builder()).expect("Failed to create small model");
    Box::new(model)
}
