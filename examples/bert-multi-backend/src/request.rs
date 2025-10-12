use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub input_ids: Vec<i64>,
    pub model_route: String,
}

impl InferenceRequest {
    pub fn new(input_ids: Vec<i64>, model_route: impl Into<String>) -> Self {
        Self { 
            input_ids, 
            model_route: model_route.into() 
        }
    }

    pub fn to_route(input_ids: Vec<i64>, route: impl Into<String>) -> Self {
        Self::new(input_ids, route)
    }
}
