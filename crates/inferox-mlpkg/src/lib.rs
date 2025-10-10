//! Model package format for Inferox
//!
//! This crate provides functionality for packaging and loading ML models
//! with their weights in a portable format.

#![warn(missing_docs)]

/// Model package errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Generic error
    #[error("{0}")]
    Generic(String),
}

/// Result type for model package operations
pub type Result<T> = std::result::Result<T, Error>;
