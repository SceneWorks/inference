//! Lens joint (dual-stream) attention (`LensJointAttention`). **Fused** `img_qkv`/`txt_qkv`
//! projections (bias) split into per-stream q/k/v, per-head q/k RMSNorm, interleaved-complex RoPE on
//! both streams, then SDPA over the **`[img, txt]`**-concatenated sequence (matching the Lens
//! `_build_joint_attention_mask` which orders image tokens first), split back and projected
//! (`to_out.0` for image, `to_add_out` for text).

use mlx_rs::error::Result as MlxResult;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::split_sections;
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionPlan};
use mlx_gen::qkv::{
    self, AttnPrepSpec, QkNormSpec, QkvSource, RopeDtype, RopeSpec, RopeStyle, RopeTables,
    StreamOrder,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, load_weight};

/// QK-RMSNorm + block-norm epsilon (the Lens block builds `LensJointAttention(eps=1e-6)` via its own
/// `eps` default).
const RMS_EPS: f32 = 1e-6;

/// Load a biased diffusers `[out, in]` projection as an [`AdaptableLinear`] (the LoRA/LoKr adapter
/// targets, sc-3174). The dense forward is `x·Wᵀ + b`, identical to the sc-3168 [`super::Linear`].
///
/// Packed-detect (sc-8763): routes through [`crate::quant::lin`], so a pre-quantized turnkey loads the
/// packed triple `{prefix}.{weight,scales,biases}` directly. The dense path is cast to `dtype`; a
/// packed base carries its own compute dtype (`cast_weights` no-ops on it). The `quantize` after any
/// adapter merge no-ops on the already-packed base.
fn load_adaptable(w: &Weights, prefix: &str, dtype: Dtype) -> Result<AdaptableLinear> {
    let mut l = crate::quant::lin(w, prefix, true)?;
    l.cast_weights(dtype)?;
    Ok(l)
}

#[derive(Clone)]
pub struct LensJointAttention {
    img_qkv: AdaptableLinear,
    txt_qkv: AdaptableLinear,
    to_out: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    /// sc-5170 — run the joint SDPA inside an `mlx::checkpoint` so its backward recomputes the
    /// attention instead of retaining the `[heads, joint, joint]` probability matrix (the grad
    /// through `fast::scaled_dot_product_attention` decomposes to naive attention — MLX has no fused
    /// SDPA backward — and that one retained seq² array per block dominates the dense training
    /// working set). Numerically identical (same math, recomputed); inference never sets it (default
    /// off, zero cost), the trainer enables it unconditionally (LoRA + LoKr).
    ckpt_sdpa: bool,
}

impl LensJointAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            img_qkv: load_adaptable(w, &join(prefix, "img_qkv"), dtype)?,
            txt_qkv: load_adaptable(w, &join(prefix, "txt_qkv"), dtype)?,
            to_out: load_adaptable(w, &join(prefix, "to_out.0"), dtype)?,
            to_add_out: load_adaptable(w, &join(prefix, "to_add_out"), dtype)?,
            norm_q: load_weight(w, &join(prefix, "norm_q"), dtype)?,
            norm_k: load_weight(w, &join(prefix, "norm_k"), dtype)?,
            norm_added_q: load_weight(w, &join(prefix, "norm_added_q"), dtype)?,
            norm_added_k: load_weight(w, &join(prefix, "norm_added_k"), dtype)?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-5170). Training-only knob — see `ckpt_sdpa`.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Quantize the four projections to Q4/Q8 (sc-3175). Call **after** any adapter merge — the
    /// adapters are forward-time residuals over the (now quantized) base, exactly as the shared seam
    /// intends, so a quantized base + LoRA residual compose. The QK-norm weights stay full precision.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_qkv.quantize(bits, None)?;
        self.txt_qkv.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        self.to_add_out.quantize(bits, None)?;
        Ok(())
    }

    /// `img`/`txt`: `[B, seq, dim]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, img+txt]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        self.forward_with_attention(
            img,
            txt,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            mask,
            AttentionPlan::UNBOUNDED,
        )
    }

    /// Inference forward with the shared bounded-attention plan. The unbounded plan retains the
    /// original single-kernel path exactly; training continues to use [`Self::forward`] because an
    /// eval-per-chunk plan is not valid inside an autograd trace.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_attention(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<(Array, Array)> {
        let img_seq = img.shape()[1];
        let (h, hd) = (self.num_heads, self.head_dim);

        // SC-18319 — the shared prologue. Lens's knob selection: fused packed QKV (knob 9), per-head
        // QK-RMSNorm after the head split (knob 1), adjacent-pair/interleaved-complex RoPE on both
        // streams from per-stream tables (knobs 2, 5, 6), token-major rotation, and an `[img, txt]`
        // join (knob 11) matching `_build_joint_attention_mask`, which orders image tokens first.
        let stream =
            |lin: &AdaptableLinear, x: &Array, nq: &Array, nk: &Array, cos: &Array, sin: &Array| {
                let spec = AttnPrepSpec::new(h, hd)
                    .with_qk_norm(QkNormSpec::per_head(nq, nk, RMS_EPS))
                    .with_rope(RopeSpec {
                        style: RopeStyle::AdjacentPair,
                        q: Some(RopeTables::new(cos, sin)),
                        k: Some(RopeTables::new(cos, sin)),
                        // Knob 12 — the removed `apply_rope` ended in `.as_dtype(x.dtype())`, so
                        // the f32 tables' promotion is undone and SDPA sees the stream's own dtype.
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    });
                qkv::prepare(QkvSource::Packed(&lin.forward(x)?), &spec)
            };
        let img_heads = stream(
            &self.img_qkv,
            img,
            &self.norm_q,
            &self.norm_k,
            img_cos,
            img_sin,
        )?;
        let txt_heads = stream(
            &self.txt_qkv,
            txt,
            &self.norm_added_q,
            &self.norm_added_k,
            txt_cos,
            txt_sin,
        )?;
        let joint_qkv = StreamOrder::ImageFirst.join(&img_heads, &txt_heads)?;
        let (q, k, v) = (joint_qkv.q, joint_qkv.k, joint_qkv.v);

        let o = if self.ckpt_sdpa {
            // sc-5170: checkpoint just the joint SDPA. q/k/v are the threaded inputs (grads to the
            // QKV projections — and their LoRA — flow back through them); the f32 scale and the
            // additive mask are captured constants (the mask carries no trainable graph). The
            // backward recomputes the decomposed attention for THIS block alone, so the
            // `[heads, joint, joint]` probability matrix is a per-block transient, never 48×
            // retained.
            let scale = self.scale;
            let m = mask.cloned();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                let o = match m.as_ref() {
                    Some(mm) => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, mm, None)?
                    }
                    None => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, None, None)?
                    }
                };
                Ok(vec![o])
            });
            seg(&[q, k, v])?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("lens: checkpoint SDPA produced no output".into())
            })?
        } else {
            sdpa_budgeted_bhsd(&q, &k, &v, self.scale, mask, attention)?
        };
        let o = qkv::merge_heads(&o)?;

        // Split back at the image/text boundary (image first).
        let parts = split_sections(&o, &[img_seq], 1)?;
        let img_attn = self.to_out.forward(&parts[0])?;
        let txt_attn = self.to_add_out.forward(&parts[1])?;
        Ok((img_attn, txt_attn))
    }
}

impl AdaptableHost for LensJointAttention {
    /// Trained-file (diffusers/peft) module names → the fused attention projections (sc-3174). The
    /// Lens trainer's `DEFAULT_LORA_TARGET_MODULES` = `img_qkv` / `txt_qkv` / `to_out.0` / `to_add_out`
    /// (the QKV are fused `[3·inner, in]`, so a LoRA on them merges whole — no q/k/v split). `to_out`
    /// is a `ModuleList([Linear, Identity])`, addressed `to_out.0`; accept the bare `to_out` alias too.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["img_qkv"] => Some(&mut self.img_qkv),
            ["txt_qkv"] => Some(&mut self.txt_qkv),
            ["to_out"] | ["to_out", "0"] => Some(&mut self.to_out),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

// The interleaved-complex RoPE that used to live here is now
// `mlx_gen::qkv::apply_rope(.., RopeStyle::AdjacentPair, ..)` (SC-18319) — the identical
// `(real, imag)·(cos, sin)` expression, routed through the shared compiled-glue `nn::rope_rotate`.
