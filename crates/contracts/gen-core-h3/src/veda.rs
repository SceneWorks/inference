//! Opt-in MiniMax-H3 learned sparse attention. An absent or disabled request preserves dense execution.

pub use gen_core::veda::{VedaBudget, VedaExecutionReport, VedaRequest};
/// Frozen tensor-neutral tiling, predictor and selection contract.
pub mod contract;
mod topk;
pub mod workspace;
