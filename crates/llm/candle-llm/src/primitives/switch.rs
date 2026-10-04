//! The process-wide on/off switch the fused primitives ([`fused`](super::fused)), the NVFP4
//! decode GEMV ([`nvfp4_path`](super::nvfp4_path)), device positions
//! ([`device_positions`](super::device_positions)) and the CUDA-graph runner
//! ([`graph`](crate::decode::graph)) each switch through — the one shared implementation,
//! [`core_llm::switch`] (sc-24446: mlx-llm's runtime switches take the same type), re-exported here.

pub use core_llm::switch::{ProcessSwitch, SwitchGuard, SwitchLock};
