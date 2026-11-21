//! Native MLX projection layers shared by media and language models.

pub mod convrot_int8;
pub mod int8_gemm;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Msg(value.to_owned())
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Msg(value)
    }
}
