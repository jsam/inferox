use crate::request::InferenceRequest;
use crate::response::UnifiedTensorResponse;
use inferox_candle::CandleBackend;
use inferox_core::{Backend, InferoxError, TensorBuilder};
use inferox_engine::{DynError, InferoxEngine};
use inferox_mlpkg::BackendType;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "tch")]
use inferox_tch::TchBackend;

#[derive(thiserror::Error, Debug)]
pub enum RouterError {
    #[error("Engine error: {0}")]
    Engine(String),
    
    #[error("Model route '{0}' not found")]
    RouteNotFound(String),
    
    #[error("Backend {0:?} not available")]
    BackendUnavailable(BackendType),
    
    #[error("Tensor conversion error: {0}")]
    TensorConversion(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
}

#[derive(Debug, Clone)]
pub struct ModelRoute {
    pub model_name: String,
    pub backend: BackendType,
}

pub struct InferenceRouter {
    engine: Arc<InferoxEngine>,
    routes: HashMap<String, ModelRoute>,
}

impl InferenceRouter {
    pub fn new(engine: Arc<InferoxEngine>) -> Self {
        Self {
            engine,
            routes: HashMap::new(),
        }
    }

    pub fn register_route(&mut self, route: impl Into<String>, model_name: String, backend: BackendType) {
        let route_key = route.into();
        self.routes.insert(
            route_key.clone(),
            ModelRoute { model_name, backend },
        );
    }

    pub fn with_route(mut self, route: impl Into<String>, model_name: String, backend: BackendType) -> Self {
        self.register_route(route, model_name, backend);
        self
    }
    
    pub fn validate_routes(&self) -> Result<(), RouterError> {
        let engine_models: HashMap<String, _> = self.engine.list_models()
            .into_iter()
            .map(|(name, _meta)| (name.to_string(), ()))
            .collect();
        
        for (route, model_route) in &self.routes {
            if !engine_models.contains_key(&model_route.model_name) {
                return Err(RouterError::Engine(format!(
                    "Route '{}' points to model '{}' which is not loaded in the engine. \
                     All models must be loaded at startup before registering routes.",
                    route, model_route.model_name
                )));
            }
        }
        
        Ok(())
    }

    pub fn infer(&self, request: InferenceRequest) -> Result<UnifiedTensorResponse, RouterError> {
        let start = Instant::now();
        
        let model_route = self
            .routes
            .get(&request.model_route)
            .ok_or_else(|| RouterError::RouteNotFound(request.model_route.clone()))?;

        let shape = vec![1, request.input_ids.len()];
        
        match model_route.backend {
            BackendType::Candle => {
                let backend = CandleBackend::cpu()
                    .map_err(|e| RouterError::Engine(format!("Failed to create Candle backend: {}", e)))?;

                let input_tensor = backend
                    .tensor_builder()
                    .build_from_vec(request.input_ids.clone(), &shape)
                    .map_err(|e| RouterError::TensorConversion(format!("Failed to create tensor: {}", e)))?;

                let output = self
                    .engine
                    .infer::<CandleBackend>(&model_route.model_name, input_tensor)
                    .map_err(|e| RouterError::Inference(format!("{:?}", e)))?;

                UnifiedTensorResponse::from_candle(output, start)
                    .map_err(|e| RouterError::TensorConversion(e))
            }
            #[cfg(feature = "tch")]
            BackendType::Tch => {
                let backend = TchBackend::cpu()
                    .map_err(|e| RouterError::Engine(format!("Failed to create Tch backend: {}", e)))?;

                let input_tensor = backend
                    .tensor_builder()
                    .build_from_vec(request.input_ids.clone(), &shape)
                    .map_err(|e| RouterError::TensorConversion(format!("Failed to create tensor: {}", e)))?;

                let output = self
                    .engine
                    .infer::<TchBackend>(&model_route.model_name, input_tensor)
                    .map_err(|e| RouterError::Inference(format!("{:?}", e)))?;

                UnifiedTensorResponse::from_tch(output, start)
                    .map_err(|e| RouterError::TensorConversion(e))
            }
            #[cfg(not(feature = "tch"))]
            BackendType::Tch => Err(RouterError::BackendUnavailable(BackendType::Tch)),
            _ => Err(RouterError::Engine(format!(
                "Unsupported backend: {:?}",
                model_route.backend
            ))),
        }
    }

    pub fn list_routes(&self) -> Vec<String> {
        self.routes.keys().cloned().collect()
    }

    pub fn get_route(&self, route: &str) -> Option<&ModelRoute> {
        self.routes.get(route)
    }
}
