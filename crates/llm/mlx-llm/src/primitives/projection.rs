//! A linear projection that is either dense or group-wise quantized.
//!
//! Moved to [`mlx_layers::llm_projection`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use mlx_layers::llm_projection::{KvProjection, Projection, QuantGrid, QuantSpec};
