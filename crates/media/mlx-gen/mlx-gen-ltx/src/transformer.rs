//! S3 — the LTX-2.3 **DiT** (video stack): the preprocessor (patchify + adaLN-single) → 48 ×
//! `BasicAVTransformerBlock` (video-only path) → output projection → velocity. Port of the
//! `mlx_video` reference `models/ltx/{transformer,attention,adaln,feed_forward,ltx}.py`.
//!
//! Per-block math (S3a): **gated attention** (`to_gate_logits → 2·sigmoid`, zero-init identity), q/k
//! **RMSNorm** over the full inner_dim (pre-head, learned), **SPLIT 3-D RoPE** on q/k (reusing the S0
//! [`crate::rope`]), SDPA, `to_out`; **adaLN-single** with the 9-row `scale_shift_table` (gated 2.3
//! family: MSA rows 0..3, FF rows 3..6, text-cross-attn rows 6..9); **prompt adaLN** modulating the
//! text context; **FeedForward** = `proj_in → gelu(tanh) → proj_out`.
//!
//! Full forward (S3b): patchify_proj → adaLN-single (timestep → 9·dim) + prompt-adaLN (→ 2·dim) →
//! caption projection (Identity for 2.3) → SPLIT RoPE from the position grid → 48 blocks → output
//! (`LayerNorm` affine-false + final 2-row `scale_shift_table` modulated by the embedded timestep) →
//! `proj_out → 128` velocity. `denoised = latent − σ·velocity`.
//!
//! **Quant.** The shipped transformer stores the attn/ff Linears selectively quantized (U32 +
//! `scales` + `biases`) — there is no dense bf16 checkpoint. The **bits/group ride on the checkpoint's
//! `split_model.json`** ([`crate::config::SplitModel`]): `base_q8` at 8 bits, `base_q4` at 4 bits,
//! group 64 — read into [`Precision`], never hardcoded (sc-2686). The per-Linear predicate
//! (quantize iff the weights carry `.scales`) mirrors `generate_av.py`'s `_should_quantize`.
//!
//! [`Precision::quant_f32`] is the production quality target: **f32 activations × `quantized_matmul`**
//! (a single block is bit-exact to the reference at matched mlx 0.31.2). [`Precision::quant_bf16`]
//! mirrors the reference's own bf16 compute (the production-speed path). [`Precision::dense_f32`]
//! additionally dequantizes the weights to dense f32 — the S3a block-math gate.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::fast::{layer_norm, rms_norm as fast_rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, dequantize, divide, matmul, multiply, sigmoid, subtract};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::block_residency::BlockPlan;

/// sc-18797's rung-4 loader-identity and bit-identity suite. A `#[path]` submodule rather than a
/// `tests/` file because it observes the private loader surface (`AvBlocks`, `AvBlock`'s attentions)
/// that AC2's output-invisible failure mode can only be caught on.
#[cfg(test)]
#[path = "transformer_rung4_tests.rs"]
pub(crate) mod rung4_block_window_tests;
use mlx_gen::train::lora::LoraParams;
use mlx_gen::CancelFlag;

use mlx_gen::nn::{linear, quantized_matmul_with_bias, DitCompileGlue, DitGlue};
use mlx_gen::qkv::{
    self, AttnPrepSpec, QkNormSpec, QkvSource, RopeDtype, RopeSpec, RopeStyle, RopeTables,
    RotationAxes,
};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::rope::precompute_split_freqs_cis;

// sc-27342: the adapter-bearing Linear, attention / feed-forward, the video block and DiT, and
// the audio-video block and DiT live in sibling modules; precision and the compiled glue stay here.
mod attention;
mod av_block;
mod av_dit;
mod linear;
mod video_block;
mod video_dit;
use attention::*;
pub use av_block::*;
pub use av_dit::*;
pub use linear::*;
pub use video_block::*;
pub use video_dit::*;

/// adaLN-single sinusoidal timestep projection width (PixArt `Timesteps`).
const TIME_PROJ_DIM: i32 = 256;

/// How to run the (selectively quantized) DiT: the activation/compute dtype, whether quantized
/// weights stay packed (`quantized_matmul`) or are dequantized to dense, and the **checkpoint's**
/// quant geometry (`bits`/`group` from `split_model.json` — so Q4 and Q8 both load without a code
/// change; sc-2686). Construct via [`quant_f32`](Self::quant_f32) / [`quant_bf16`](Self::quant_bf16)
/// / [`dense_f32`](Self::dense_f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    mode: Mode,
    bits: i32,
    group: i32,
}

/// The compute mode (independent of the quant bit-width, which rides alongside in [`Precision`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// f32 activations, quantized weights **dequantized** to dense f32 — the S3a block-math gate.
    DenseF32,
    /// **f32 activations × `quantized_matmul`** — the production path / quality target. The full
    /// 48-layer velocity is bit-exact to the reference (mlx 0.31.2), required because the distilled
    /// stage-1 sampler is chaos-sensitive (sc-2842).
    QuantF32,
    /// bf16 activations × `quantized_matmul` — the reference's own compute dtype (production speed).
    QuantBf16,
}

impl Precision {
    /// f32 activations, quantized weights dequantized to dense f32 (the block-math gate).
    pub fn dense_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::DenseF32,
            bits,
            group,
        }
    }

    /// f32 activations × `quantized_matmul` (the production quality target).
    pub fn quant_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantF32,
            bits,
            group,
        }
    }

    /// bf16 activations × `quantized_matmul` (the reference's native production-speed path).
    pub fn quant_bf16(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantBf16,
            bits,
            group,
        }
    }

    pub(crate) fn dtype(self) -> Dtype {
        match self.mode {
            Mode::DenseF32 | Mode::QuantF32 => Dtype::Float32,
            Mode::QuantBf16 => Dtype::Bfloat16,
        }
    }

    /// Whether quantized weights are kept packed (`quantized_matmul`) vs dequantized to dense f32.
    fn keep_quant(self) -> bool {
        matches!(self.mode, Mode::QuantF32 | Mode::QuantBf16)
    }

    /// The same quant geometry re-targeted at a new compute/activation `dtype` (sc-4942 bf16 training
    /// cast). bf16 → `QuantBf16`; f32 preserves whether the base was kept packed (`QuantF32`) or
    /// dequantized to dense (`DenseF32`). Used by [`LtxDiT::cast_weights`] to flip the whole forward
    /// to bf16 activations once the leaves are cast.
    pub(crate) fn with_compute_dtype(self, dtype: Dtype) -> Self {
        let mode = match dtype {
            Dtype::Bfloat16 => Mode::QuantBf16,
            _ if self.keep_quant() => Mode::QuantF32,
            _ => Mode::DenseF32,
        };
        Self { mode, ..self }
    }
}

const SITE_MODULATE: &str = "ltx::transformer::modulate";
const SITE_GATED: &str = "ltx::transformer::gated";
const SITE_GELU_FFN: &str = "ltx::transformer::gelu_ffn";

fn modulate_impl(
    (x, scale, shift): (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    add(
        &multiply(x, &add(scale, &scalar(1.0).as_dtype(scale.dtype())?)?)?,
        shift,
    )
}

thread_local! {
    static RETAINED_COMPILE_GLUE: RefCell<Option<DitCompileGlue>> = const { RefCell::new(None) };
}

/// The LTX DiT's compiled glue (sc-2963) — adaLN `x·(1+scale)+shift`, gated residual `x + out·gate`
/// and the tanh-GELU FFN activation — through the shared retained/one-shot/eager dispatch
/// (sc-27339). Each is one fused kernel when the glue toggle is on, bit-identical to the eager form
/// and dtype-preserving (the modulate `1` is cast to `scale`'s dtype inside). The GELU fusion is by
/// far the biggest per-step glue cost at video sequence (the FFN expansion runs on `[B, S, ffn]`
/// tensors of tens-to-hundreds of millions of elements); off ⇒ core `gelu_tanh`, byte-for-byte the
/// previous behaviour.
static GLUE: DitGlue = DitGlue::new(
    [SITE_MODULATE, SITE_GATED, SITE_GELU_FFN],
    modulate_impl,
    &RETAINED_COMPILE_GLUE,
);

/// Exercise every retained handle owned by the LTX transformer once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &Array) -> Result<()> {
    GLUE.exercise_retained_compile_inventory(input)
}

/// Load a non-Linear param (norm weight, scale-shift table) cast to the compute dtype.
fn param(w: &Weights, key: &str, prec: Precision) -> Result<Array> {
    to_dtype(w.require(key)?, prec.dtype())
}

/// Add the learned keyframe absolute-position marker to single-pixel generated-keyframe tokens
/// (sc-18758; reference `apply_keyframes_absolute_embedding`). Applied to the patchified video
/// hidden states, immediately after `patchify_proj`. `embedding` is the model's `(1, inner_dim)`
/// `keyframes_abs_pos_embedding` parameter (`None` for a model built without
/// `use_keyframes_abs_pos_embedding`, matching `LTXModel._keyframes_embedding()`); `keyframes_mask`
/// is `(B, T, 1)`, `> 0` marking a keyframe token (`None` = no token marked). Either `None` makes this
/// an exact no-op — as does an embedding that is still zero-initialized (the loaded-but-untrained
/// state). The DFR token loops (sc-18789, [`crate::dfr`] / the token-native pipeline paths) thread
/// a real mask marking generated-keyframe slot tokens; grid paths with no slots pass `None`.
fn apply_keyframes_embedding(
    x: &Array,
    embedding: Option<&Array>,
    keyframes_mask: Option<&Array>,
) -> Result<Array> {
    let (Some(embedding), Some(mask)) = (embedding, keyframes_mask) else {
        return Ok(x.clone());
    };
    let gate = mask
        .gt(scalar(0.0f32).as_dtype(mask.dtype())?)?
        .as_dtype(x.dtype())?;
    let marker = multiply(&gate, &embedding.as_dtype(x.dtype())?)?;
    Ok(add(x, &marker)?)
}

/// `x · (1 + scale) + shift` (adaLN modulation), broadcasting `scale`/`shift` `(B, S', dim)` over the
/// token axis.
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    GLUE.modulate(x, scale, shift)
}

/// Gated residual `x + out·gate`.
fn gated(x: &Array, out: &Array, gate: &Array) -> Result<Array> {
    GLUE.gated(x, out, gate)
}

/// The tanh-GELU FFN activation.
fn gelu_ffn(x: &Array) -> Result<Array> {
    GLUE.gelu_ffn(x)
}

#[cfg(test)]
mod tests;
