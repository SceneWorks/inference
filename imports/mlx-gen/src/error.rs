//! Crate error type. Replaces the early `Box<dyn Error>` placeholder with a typed enum
//! (sc-2373, disciplined-hybrid architecture). `From<&str>`/`From<String>` are provided so
//! existing `"...".into()` / `format!(...).into()` error sites keep compiling, while
//! `#[from]` lets `?` lift `mlx_rs` and IO errors transparently.

use thiserror::Error;

/// Anything that can go wrong in mlx-gen.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX op (matmul, quantize, SDPA, …) failed on device.
    #[error("MLX op failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// A required tensor key was absent from a loaded checkpoint/adapter.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem error while traversing a model directory.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// safetensors load/save error from mlx-rs.
    #[error("safetensors I/O failed: {0}")]
    SafeTensors(#[from] mlx_rs::error::IoError),

    /// A contextual message (config/validation/adapter-shape errors).
    #[error("{0}")]
    Msg(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;
