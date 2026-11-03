//! AGPL-3.0-or-later CV engines (see the crate LICENSE and NOTICE).
//!
//! Native CV engines. Callers supply local weights and session loading; engines own
//! numeric preprocessing, execution and decoding. No download or application cache policy.

mod image_sampling;
pub mod person;
pub mod pose;
pub mod session;
pub mod upscale;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Engine(String),
    #[error("{0}")]
    InvalidPayload(String),
    #[error("{0}")]
    Canceled(String),
    /// Preserve the caller's typed load failure across the engine boundary.
    #[error("{0}")]
    Load(Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;

impl<R> From<ort::Error<R>> for Error {
    fn from(error: ort::Error<R>) -> Self {
        Self::Engine(format!("onnxruntime: {error}"))
    }
}
