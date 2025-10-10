//! Rust client for Hugging Face model downloads using XET technology
//!
//! This crate provides efficient downloads from Hugging Face Hub using XET's
//! chunk-based deduplication and caching mechanisms.

#![warn(missing_docs)]

pub mod cache;
pub mod client;
pub mod config;
pub mod error;
pub mod hf_api;
pub mod types;
pub mod xet;

pub use client::HfXetClient;
pub use config::XetConfig;
pub use error::{Error, Result};
pub use types::{CacheStats, FileInfo, LfsPointer, RepoInfo, XetPointer};
