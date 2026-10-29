//! Qwen-Image 2.1 host logic shared by `mlx-gen-qwen-image-2-1` and `candle-gen-qwen-image-2-1`
//! (sc-27329): the config parsers, the flow-match schedules, and the reference-conditioning
//! geometry and refusals. Nothing here holds a tensor; each engine re-exports these modules under
//! its own `config` / `scheduler` / `reference` paths.

pub mod config;
pub mod reference;
pub mod scheduler;
