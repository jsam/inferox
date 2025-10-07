use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferoxError<E: std::error::Error> {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Backend error: {0}")]
    Backend(E),
    
    #[error("Invalid input shape")]
    InvalidInput,
    
    #[error("Out of memory")]
    OutOfMemory,
}
