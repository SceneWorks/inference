//! Qwen-Image **60-layer dual-stream MMDiT** transformer. Port of the fork's `QwenTransformer`.
//!
//! Dual-stream: image and text tokens carry separate AdaLN modulation (from the timestep
//! embedding) and feed-forward, but attend **jointly** over the concatenated `[txt, img]` sequence.
//! Uses interleaved-complex 3D RoPE ([`rope`]), per-head q/k RMSNorm, affine-less LayerNorms, and
//! `AdaLayerNormContinuous` at the output. NCS (batch, seq, dim) tensors throughout.

pub mod attention;
pub mod block;
pub mod feed_forward;
pub mod norm_out;
pub mod rope;
pub mod time_text_embed;
pub mod timesteps;
#[allow(clippy::module_inception)]
pub mod transformer;

pub use attention::QwenJointAttention;
pub use block::QwenTransformerBlock;
pub use feed_forward::FeedForward;
pub use norm_out::AdaLayerNormContinuous;
pub use rope::QwenRope3d;
pub use transformer::{QwenTransformer, QwenTransformerConfig};

use mlx_rs::ops::matmul;
use mlx_rs::Array;

use mlx_gen::Result;

/// `y = x · Wᵀ` for a stored `[out, in]` weight (bias-less Linear).
pub(crate) fn matmul_t(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
