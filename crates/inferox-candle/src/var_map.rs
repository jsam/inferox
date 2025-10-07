use candle_nn::VarMap;
use std::path::Path;

pub struct CandleVarMap {
    inner: VarMap,
}

impl CandleVarMap {
    pub fn new() -> Self {
        Self {
            inner: VarMap::new(),
        }
    }

    pub fn inner(&self) -> &VarMap {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut VarMap {
        &mut self.inner
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), candle_core::Error> {
        self.inner.load(path)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), candle_core::Error> {
        self.inner.save(path)
    }

    pub fn var_builder(
        &self,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> candle_nn::VarBuilder<'_> {
        candle_nn::VarBuilder::from_varmap(&self.inner, dtype, device)
    }
}

impl Default for CandleVarMap {
    fn default() -> Self {
        Self::new()
    }
}
