//! The process-wide on/off switch the fused primitives ([`fused`](super::fused)), the NVFP4
//! decode GEMV ([`nvfp4_path`](super::nvfp4_path)), device positions
//! ([`device_positions`](super::device_positions)) and the CUDA-graph runner
//! ([`graph`](crate::decode::graph)) each switch through — re-exported from [`candle_layers::switch`]
//! (`SW-6957` / `SW-F335`).

pub use candle_layers::switch::{ProcessSwitch, SwitchGuard, SwitchLock};
