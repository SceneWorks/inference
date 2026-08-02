//! Vendored **inference** Z-Image DiT with a packed-load seam (sc-9408).
//!
//! A faithful copy of the stock `candle-transformers` `z_image::transformer` model at the workspace
//! candle pin (`c1e6756`), vendored because the stock `ZImageAttention` / `FeedForward` / `FinalLayer`
//! / block / model build their projections from frozen `candle_nn::Linear` with **no seam** — so they
//! cannot load a pre-quantized MLX-packed tier (`SceneWorks/z-image-turbo-mlx`, whose q4/q8 snapshots
//! store each quantized projection as the packed triple `{base}.weight` u32 + `.scales` + `.biases`).
//! Only the five structs that *own* those projections are vendored; everything else — `Config`,
//! `TimestepEmbedder` (its MLP stays dense — not in the packed base set), `QkNorm`, `RopeEmbedder`,
//! `LayerNormNoParams`, `apply_rotary_emb` / `patchify` / `unpatchify` / `create_coordinate_grid`, the
//! constants — is **reused** straight from the stock crate (the same reuse [`crate::dit`], the
//! training model, already does), so no logic drifts.
//!
//! Each vendored projection is a [`crate::quant::QLinear`], which **packed-detects** the `.scales`
//! sibling ([`QLinear::linear_detect`]): a packed tier builds the quantized weight straight from the
//! packed parts (Q4→`Q4_1` lossless, Q8→`Q8_0` requant, dequant-on-forward — sc-7702), a dense bf16
//! tier loads the dense weight unchanged. **The dense path is byte-identical to the stock model**: the
//! `parity_tests::vendored_dense_dit_matches_stock_forward` test (the stock-vs-vendored DiT numeric
//! parity test tracked by sc-9443) pins it — built from the same weights with no `.scales`, the
//! vendored forward matches the stock `ZImageTransformer2DModel` forward bit-for-bit. Unlike flux
//! (diffusers split-QKV vs BFL fused-QKV, which needs a load-time layout remap in its parity test),
//! z-image's vendored DiT shares the stock `z_image::transformer` layout key-for-key, so the same
//! `VarMap` feeds both models directly with no remap. This model is used only when the snapshot is a packed tier
//! ([`crate::pipeline`]); a dense snapshot keeps using the stock `ZImageTransformer2DModel`.

use candle_gen::candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen::candle_nn::{RmsNorm, VarBuilder};
use std::sync::Arc;

// The projection type is the shared residual-capable [`candle_gen::quant::AdaptLinear`] (sc-11105),
// aliased to the crate-local `QLinear` name so every `linear_detect` call site below stays unchanged.
// It carries an optional **forward-time additive LoRA/LoKr residual**, so a user LoRA (or the distill)
// can apply on a **packed q4/q8** tier with the base kept packed — the deltas ride unmerged, never
// folded into u32 codes ([`crate::adapters::install_additive`], mirroring qwen-image-edit in sc-11091).
// With no adapter attached the forward is byte-identical to the bare base, so the dense-parity test and
// every packed load are unchanged. (The crate's other packed seams — `packed_te`, the VAE dequant — keep
// using the plain `crate::quant::QLinear` enum; only the DiT needs the residual surface.)
use candle_gen::block_window::BlockPlan;
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::quant::{AdaptLinear as QLinear, PackedWeightSidecars};

/// Load one projection in a streamed block. Packed triples must resolve through the prepared
/// device-format cache; dense tensors keep the exact historical `VarBuilder` path.
fn streamed_linear_detect(
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilder,
    base: &str,
    bias: bool,
    sidecars: &PackedWeightSidecars,
    sidecar_prefix: &str,
) -> Result<QLinear> {
    let scales_key = format!("{base}.scales");
    if !vb.contains_tensor(&scales_key) {
        return QLinear::linear_detect(in_dim, out_dim, vb, base, bias);
    }

    let sidecar_base = format!("{sidecar_prefix}.{base}");
    if !sidecars.contains(&sidecar_base) {
        candle_gen::candle_core::bail!(
            "z-image streamed packed projection `{sidecar_base}` has no prepared device-format \
             sidecar"
        );
    }
    let dense_bias = if bias {
        Some(vb.get(out_dim, &format!("{base}.bias"))?)
    } else {
        None
    };
    let qtensor = sidecars.load(&sidecar_base, vb.device())?;
    let packed = candle_gen::quant::QLinear::from_qtensor_dequant(Arc::new(qtensor), dense_bias);
    Ok(QLinear::from_packed(packed, in_dim, out_dim))
}

fn plan_from_budget(budget: usize) -> AttentionPlan<'static> {
    let max_score_elements = if budget == usize::MAX {
        u64::MAX
    } else {
        budget as u64
    };
    AttentionPlan::budgeted(AttentionBudget::from_score_elements(
        max_score_elements,
        false,
    ))
}

fn into_candle_core(result: candle_gen::Result<Tensor>) -> Result<Tensor> {
    result.map_err(|error| match error {
        candle_gen::CandleError::Candle(error) => error,
        other => candle_gen::candle_core::Error::Msg(other.to_string()),
    })
}

// Reused verbatim from candle-transformers — frozen sub-modules + the patchify/RoPE helpers that hold
// no packed projection (identical reuse to `crate::dit`). Vendoring these would add drift for zero
// benefit. `TimestepEmbedder` is NOT reused — its `mlp.0`/`mlp.2` ARE packed in the MLX tier, so it is
// re-vendored below with a QLinear seam.
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, Config, LayerNormNoParams,
    QkNorm, RopeEmbedder, ADALN_EMBED_DIM, FREQUENCY_EMBEDDING_SIZE, MAX_PERIOD,
};

// ==================== TimestepEmbedder (packed seam) ====================

/// Sinusoidal timestep embedding + a 2-layer MLP whose `mlp.0` / `mlp.2` projections ARE packed in the
/// MLX tier — so, unlike [`crate::dit`]'s training model (which reuses the stock `TimestepEmbedder`),
/// the inference packed model re-vendors it with a [`QLinear`] seam. Same `timestep_embedding → linear1
/// → silu → linear2` math + `mlp.0`/`mlp.2` keys (both biased) as the stock `TimestepEmbedder`.
struct TimestepEmbedder {
    linear1: QLinear,
    linear2: QLinear,
    frequency_embedding_size: usize,
    /// The MLP input dtype — the model's compute dtype (`vb.dtype()`), matching the stock embedder's
    /// `self.linear1.weight().dtype()` (dense bf16 tier ⇒ bf16). QLinear's dense arm requires the
    /// activation dtype to match the weight; its packed arm dequants the weight to this dtype (parity).
    dtype: DType,
}

impl TimestepEmbedder {
    fn new(out_size: usize, mid_size: usize, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let mlp = vb.pp("mlp");
        let linear1 = QLinear::linear_detect(FREQUENCY_EMBEDDING_SIZE, mid_size, &mlp, "0", true)?;
        let linear2 = QLinear::linear_detect(mid_size, out_size, &mlp, "2", true)?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
            dtype,
        })
    }

    fn timestep_embedding(&self, t: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.to_dtype(dtype)
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let t_freq = self.timestep_embedding(t, device, self.dtype)?;
        let h = self.linear1.forward(&t_freq)?.silu()?;
        self.linear2.forward(&h)
    }

    /// Visit the two timestep-embedder projections (`{prefix}.mlp.0`, `{prefix}.mlp.2`) — dense in the
    /// MLX tier, but part of the general adaptable surface so a user LoRA that targets them applies
    /// additively on the dense base (equals the fold to f32 tolerance — sc-11105).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.mlp.0"), &mut self.linear1)?;
        f(&format!("{prefix}.mlp.2"), &mut self.linear2)?;
        Ok(())
    }
}

// ==================== ZImageAttention (packed seam) ====================

/// Z-Image attention with QK normalization and 3D RoPE, with the four projections held as
/// [`QLinear`] so a packed tier loads them straight from the packed parts. Numerically identical to
/// the stock `ZImageAttention` (the dense path builds the same `candle_nn::Linear`); the attention
/// dispatch (flash / SDPA / basic) is copied verbatim.
struct ZImageAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
    use_accelerated_attn: bool,
}

impl ZImageAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_sidecars(cfg, vb, None)
    }

    fn new_streamed(
        cfg: &Config,
        vb: VarBuilder,
        sidecars: &PackedWeightSidecars,
        sidecar_prefix: &str,
    ) -> Result<Self> {
        Self::new_with_sidecars(cfg, vb, Some((sidecars, sidecar_prefix)))
    }

    fn new_with_sidecars(
        cfg: &Config,
        vb: VarBuilder,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        // Packed bases are the full dotted key prefixes (the `.scales` siblings live directly under
        // `attention.to_q` … `attention.to_out.0`), so the detect uses the base string — never `.pp()`
        // past the sibling (the key-remap trap for `to_out.0`).
        let load = |in_dim, out_dim, base| match sidecars {
            Some((sidecars, prefix)) => {
                streamed_linear_detect(in_dim, out_dim, &vb, base, false, sidecars, prefix)
            }
            None => QLinear::linear_detect(in_dim, out_dim, &vb, base, false),
        };
        let to_q = load(dim, n_heads * head_dim, "to_q")?;
        let to_k = load(dim, cfg.n_kv_heads * head_dim, "to_k")?;
        let to_v = load(dim, cfg.n_kv_heads * head_dim, "to_v")?;
        let to_out = load(n_heads * head_dim, dim, "to_out.0")?;

        // The stock `QkNorm::new(head_dim, eps, vb.clone())` loads `attention.norm_q`/`norm_k` as
        // siblings of the projections (NOT nested under a `qk_norm` prefix) — reproduce exactly.
        let qk_norm = if cfg.qk_norm {
            Some(QkNorm::new(head_dim, 1e-5, vb.clone())?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
            use_accelerated_attn: cfg.use_accelerated_attn,
        })
    }

    fn forward_with_attention_plan(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_heads, self.head_dim))?;

        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };

        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let context = self.attention_dispatch(&q, &k, &v, attention_mask, scale, attention_plan)?;

        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok(self.to_out.forward(&context)?)
    }

    /// Visit the four attention projections (`{prefix}.to_q/to_k/to_v/to_out.0`) — the surface the
    /// Z-Image trainer's LoRA/LoKr adapts (sc-11105). `to_out` carries the diffusers `.0` suffix.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.to_q"), &mut self.to_q)?;
        f(&format!("{prefix}.to_k"), &mut self.to_k)?;
        f(&format!("{prefix}.to_v"), &mut self.to_v)?;
        f(&format!("{prefix}.to_out.0"), &mut self.to_out)?;
        Ok(())
    }

    /// Attention dispatch. The Z-Image DiT **always** passes an attention mask (from `prepare_inputs`),
    /// and the stock model's CUDA flash-attn path falls back to `attention_basic` whenever a mask is
    /// present (flash-attn can't take a custom mask) — so on CUDA the flash path is never actually taken
    /// here and this vendored copy needs no `candle-flash-attn` dependency. Metal keeps the fused SDPA
    /// path (it accepts an additive mask); everything else runs the materialized `attention_basic`. This
    /// is behaviorally identical to the stock dispatch for the mask-always inputs the DiT feeds.
    fn attention_dispatch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        if self.use_accelerated_attn && q.device().is_metal() {
            Ok(self.attention_metal(q, k, v, mask, scale)?)
        } else {
            self.attention_basic(q, k, v, mask, scale, attention_plan)
        }
    }

    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    fn attention_metal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let sdpa_mask = self.prepare_sdpa_mask(mask, q)?;
        candle_gen::candle_nn::ops::sdpa(q, k, v, sdpa_mask.as_ref(), false, scale as f32, 1.0)
    }

    fn attention_basic(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        // Build the optional additive `[B,1,1,seq]` mask up front. i32-overflow guard (sc-9116): the
        // image-token scores `[B, n, seq, seq]` reach `~24·16384² ≈ 6.4e9 > i32::MAX` at a 2048² render
        // (this is the CPU/CUDA `basic` fallback — the Metal path uses candle's fused `sdpa`), so chunk
        // over the query rows (byte-identical for common sizes) via the shared helper.
        let m = match mask {
            Some(m) => {
                let m = m.unsqueeze(1)?.unsqueeze(2)?.to_dtype(q.dtype())?;
                Some(((m - 1.0)? * 1e9)?)
            }
            None => None,
        };
        candle_gen::sdpa_planned_bhsd(
            q,
            k,
            v,
            scale,
            m.as_ref(),
            candle_gen::candle_nn::ops::softmax_last_dim,
            attention_plan,
        )
    }

    fn prepare_sdpa_mask(&self, mask: Option<&Tensor>, q: &Tensor) -> Result<Option<Tensor>> {
        match mask {
            Some(m) => {
                let (b, _, seq_len, _) = q.dims4()?;
                let m = m.unsqueeze(1)?.unsqueeze(2)?;
                let m = m.to_dtype(q.dtype())?;
                let m = ((m - 1.0)? * 1e9)?;
                let m = m.broadcast_as((b, self.n_heads, seq_len, seq_len))?;
                Ok(Some(m))
            }
            None => Ok(None),
        }
    }
}

// ==================== FeedForward (packed seam) ====================

/// SwiGLU feed-forward with the three projections held as [`QLinear`] (all packed in the tier). Same
/// `w1`/`w2`/`w3` keys + `silu(w1·x) * (w3·x) → w2` math as the stock `FeedForward`.
struct FeedForward {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: QLinear::linear_detect(dim, hidden_dim, &vb, "w1", false)?,
            w2: QLinear::linear_detect(hidden_dim, dim, &vb, "w2", false)?,
            w3: QLinear::linear_detect(dim, hidden_dim, &vb, "w3", false)?,
        })
    }

    fn new_streamed(
        dim: usize,
        hidden_dim: usize,
        vb: VarBuilder,
        sidecars: &PackedWeightSidecars,
        sidecar_prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            w1: streamed_linear_detect(
                dim,
                hidden_dim,
                &vb,
                "w1",
                false,
                sidecars,
                sidecar_prefix,
            )?,
            w2: streamed_linear_detect(
                hidden_dim,
                dim,
                &vb,
                "w2",
                false,
                sidecars,
                sidecar_prefix,
            )?,
            w3: streamed_linear_detect(
                dim,
                hidden_dim,
                &vb,
                "w3",
                false,
                sidecars,
                sidecar_prefix,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.w1.forward(x)?.silu()?;
        let x3 = self.w3.forward(x)?;
        self.w2.forward(&(x1 * x3)?)
    }

    /// Visit the three SwiGLU projections (`{prefix}.w1/w2/w3`) — part of the adaptable surface (sc-11105).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.w1"), &mut self.w1)?;
        f(&format!("{prefix}.w2"), &mut self.w2)?;
        f(&format!("{prefix}.w3"), &mut self.w3)?;
        Ok(())
    }
}

// ==================== FinalLayer (packed seam) ====================

/// The output head: `linear` + `adaLN_modulation.1` are packed; `norm_final` is param-free. Same
/// `silu(c)·adaln + 1` scale-then-project math as the stock `FinalLayer`.
struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: QLinear,
    adaln_silu: QLinear,
}

impl FinalLayer {
    fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNormNoParams::new(1e-6);
        let linear = QLinear::linear_detect(hidden_size, out_channels, &vb, "linear", true)?;
        let adaln_dim = hidden_size.min(ADALN_EMBED_DIM);
        // The stock builds this at `adaLN_modulation.1` (index `.0` is a param-free SiLU).
        let adaln_silu =
            QLinear::linear_detect(adaln_dim, hidden_size, &vb, "adaLN_modulation.1", true)?;
        Ok(Self {
            norm_final,
            linear,
            adaln_silu,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = self.adaln_silu.forward(&c.silu()?)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        self.linear.forward(&x)
    }

    /// Visit the output-head projections (`{prefix}.linear`, `{prefix}.adaLN_modulation.1`) — part of
    /// the adaptable surface (sc-11105). `norm_final` is param-free, not a projection.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.linear"), &mut self.linear)?;
        f(
            &format!("{prefix}.adaLN_modulation.1"),
            &mut self.adaln_silu,
        )?;
        Ok(())
    }
}

// ==================== ZImageTransformerBlock (packed seam) ====================

/// Z-Image transformer block; its `attention` / `feed_forward` / `adaLN_modulation.0` are packed, the
/// four RMSNorms are dense. Identical AdaLN-modulated (and non-modulated) forward to the stock block.
struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<QLinear>,
}

impl ZImageTransformerBlock {
    fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        Self::new_with_sidecars(cfg, modulation, vb, None)
    }

    fn new_streamed(
        cfg: &Config,
        modulation: bool,
        vb: VarBuilder,
        sidecars: &PackedWeightSidecars,
        block_index: usize,
    ) -> Result<Self> {
        let block_prefix = format!("layers.{block_index}");
        Self::new_with_sidecars(cfg, modulation, vb, Some((sidecars, &block_prefix)))
    }

    fn new_with_sidecars(
        cfg: &Config,
        modulation: bool,
        vb: VarBuilder,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attention = match sidecars {
            Some((sidecars, prefix)) => ZImageAttention::new_streamed(
                cfg,
                vb.pp("attention"),
                sidecars,
                &format!("{prefix}.attention"),
            )?,
            None => ZImageAttention::new(cfg, vb.pp("attention"))?,
        };
        let feed_forward = match sidecars {
            Some((sidecars, prefix)) => FeedForward::new_streamed(
                dim,
                hidden_dim,
                vb.pp("feed_forward"),
                sidecars,
                &format!("{prefix}.feed_forward"),
            )?,
            None => FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?,
        };

        let attention_norm1 =
            candle_gen::candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm1"))?;
        let attention_norm2 =
            candle_gen::candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm1 = candle_gen::candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?;
        let ffn_norm2 = candle_gen::candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?;

        let adaln_modulation = if modulation {
            let adaln_dim = dim.min(ADALN_EMBED_DIM);
            // Packed base `adaLN_modulation.0` (the `.0` is the linear; the stock nests via `.pp("0")`).
            let adaln = vb.pp("adaLN_modulation");
            Some(match sidecars {
                Some((sidecars, prefix)) => streamed_linear_detect(
                    adaln_dim,
                    4 * dim,
                    &adaln,
                    "0",
                    true,
                    sidecars,
                    &format!("{prefix}.adaLN_modulation"),
                )?,
                None => QLinear::linear_detect(adaln_dim, 4 * dim, &adaln, "0", true)?,
            })
        } else {
            None
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    fn forward_with_attention_plan(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        if let Some(ref adaln) = self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            let modulation = adaln.forward(adaln_input)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            let normed = self.attention_norm1.forward(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward_with_attention_plan(
                &scaled,
                attn_mask,
                cos,
                sin,
                attention_plan,
            )?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            Ok((x + gate_mlp.broadcast_mul(&ffn_out)?)?)
        } else {
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward_with_attention_plan(
                &normed,
                attn_mask,
                cos,
                sin,
                attention_plan,
            )?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            Ok((x + ffn_out)?)
        }
    }

    /// Visit every adaptable projection in the block under `{prefix}` — the attention + feed-forward
    /// projections, plus the optional `{prefix}.adaLN_modulation.0` (the four RMSNorms are dense, not
    /// projections). sc-11105.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attention
            .visit_adaptable_mut(&format!("{prefix}.attention"), f)?;
        self.feed_forward
            .visit_adaptable_mut(&format!("{prefix}.feed_forward"), f)?;
        if let Some(adaln) = self.adaln_modulation.as_mut() {
            f(&format!("{prefix}.adaLN_modulation.0"), adaln)?;
        }
        Ok(())
    }
}

// ==================== ZImageTransformer2DModel (packed seam) ====================

/// The packed-load inference twin of the stock `ZImageTransformer2DModel`. Built from the *same*
/// `transformer/` keys (the packed-detect siblings + the reused sub-module paths guarantee key
/// parity), so it loads a packed tier straight from the packed parts and — on a dense tier (no
/// `.scales`) — reproduces the stock forward bit-for-bit (`parity_tests`).
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: QLinear,
    x_embedder: QLinear,
    final_layer: FinalLayer,
    #[allow(dead_code)]
    x_pad_token: Tensor,
    #[allow(dead_code)]
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: TransformerLayers,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

/// Main-stack residency for ladder rung 4. The front-end, refiners, and final projection remain
/// resident; the 30 uniform `layers` blocks can instead be rebuilt from the retained read-only
/// safetensors view one window at a time.
enum TransformerLayers {
    Resident(Vec<ZImageTransformerBlock>),
    Streamed {
        weights: VarBuilder<'static>,
        sidecars: Option<Arc<PackedWeightSidecars>>,
        adapters: Option<crate::adapters::AdditivePlan>,
    },
}

impl ZImageTransformer2DModel {
    pub fn new(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::new_with_layers(cfg, vb, false, None)
    }

    /// Build the DiT with a host-backed main stack. This is the provider-side implementation of
    /// bounded transformer residency: no `layers.N` tensor is transferred until its window runs.
    pub fn new_block_streamed(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::new_with_layers(cfg, vb, true, None)
    }

    /// Build the streamed DiT with content-addressed device-format artifacts for every packed
    /// `layers.N` projection. Dense tensors continue to load from `vb` unchanged.
    pub fn new_block_streamed_with_sidecars(
        cfg: &Config,
        vb: VarBuilder<'static>,
        sidecars: Arc<PackedWeightSidecars>,
    ) -> Result<Self> {
        Self::new_with_layers(cfg, vb, true, Some(sidecars))
    }

    fn new_with_layers(
        cfg: &Config,
        vb: VarBuilder<'static>,
        stream_layers: bool,
        sidecars: Option<Arc<PackedWeightSidecars>>,
    ) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        // t_embedder stays dense (the MLX tier does not pack `t_embedder.*`) — reuse the stock struct.
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        let cap_embedder_norm = candle_gen::candle_nn::rms_norm(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear =
            QLinear::linear_detect(cfg.cap_feat_dim, cfg.dim, &vb.pp("cap_embedder"), "1", true)?;

        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder =
            QLinear::linear_detect(patch_dim, cfg.dim, &vb.pp("all_x_embedder"), "2-1", true)?;

        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer =
            FinalLayer::new(cfg.dim, out_channels, vb.pp("all_final_layer").pp("2-1"))?;

        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;

        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        let layers = if stream_layers {
            TransformerLayers::Streamed {
                weights: vb.clone(),
                sidecars,
                adapters: None,
            }
        } else {
            let mut layers = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                layers.push(ZImageTransformerBlock::new(
                    cfg,
                    true,
                    vb.pp("layers").pp(i),
                )?);
            }
            TransformerLayers::Resident(layers)
        };

        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// Narrow sibling seam for the bespoke Fun-ControlNet provider. Keeping the VACE injection math
    /// in `control.rs` must not expose the packed transformer's weights or layer representation: these
    /// helpers preserve the same packed attention path while containing future residency changes here.
    pub(crate) fn control_config(&self) -> &Config {
        &self.cfg
    }

    pub(crate) fn control_timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        self.t_embedder.forward(t)
    }

    pub(crate) fn control_embed_image(&self, patches: &Tensor) -> Result<Tensor> {
        self.x_embedder.forward(patches)
    }

    pub(crate) fn control_rope(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        self.rope_embedder.forward(position_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn control_refine_noise<F>(
        &self,
        mut hidden: Tensor,
        attention_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        attention_plan: AttentionPlan<'_>,
        mut after_layer: F,
    ) -> candle_gen::Result<Tensor>
    where
        F: FnMut(usize, Tensor) -> candle_gen::Result<Tensor>,
    {
        for (index, layer) in self.noise_refiner.iter().enumerate() {
            hidden = layer.forward_with_attention_plan(
                &hidden,
                Some(attention_mask),
                cos,
                sin,
                Some(adaln),
                attention_plan,
            )?;
            hidden = after_layer(index, hidden)?;
        }
        Ok(hidden)
    }

    pub(crate) fn control_embed_caption(&self, cap_feats: &Tensor) -> Result<Tensor> {
        let normalized = self.cap_embedder_norm.forward(cap_feats)?;
        self.cap_embedder_linear.forward(&normalized)
    }

    pub(crate) fn control_refine_context(
        &self,
        mut hidden: Tensor,
        attention_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        for layer in &self.context_refiner {
            hidden = layer.forward_with_attention_plan(
                &hidden,
                Some(attention_mask),
                cos,
                sin,
                None,
                attention_plan,
            )?;
        }
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn control_run_layers<F>(
        &self,
        mut hidden: Tensor,
        attention_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        mut after_layer: F,
    ) -> candle_gen::Result<Tensor>
    where
        F: FnMut(usize, Tensor) -> candle_gen::Result<Tensor>,
    {
        match &self.layers {
            TransformerLayers::Resident(layers) => {
                for (index, layer) in layers.iter().enumerate() {
                    hidden = layer.forward_with_attention_plan(
                        &hidden,
                        Some(attention_mask),
                        cos,
                        sin,
                        Some(adaln),
                        attention_plan,
                    )?;
                    hidden = after_layer(index, hidden)?;
                }
            }
            TransformerLayers::Streamed {
                weights,
                sidecars,
                adapters,
            } => {
                let block_plan = BlockPlan::new(self.cfg.n_layers, transformer_window)?;
                let uncancelled = candle_gen::gen_core::CancelFlag::default();
                let cancel = attention_plan.cancel.unwrap_or(&uncancelled);
                hidden = candle_gen::block_window::run_windowed(
                    self.device(),
                    &block_plan,
                    cancel,
                    hidden,
                    || Ok(weights.clone()),
                    |mut state, view, range| {
                        let first = range.start;
                        let mut blocks = range
                            .map(|index| {
                                match sidecars.as_deref() {
                                    Some(sidecars) => ZImageTransformerBlock::new_streamed(
                                        &self.cfg,
                                        true,
                                        view.pp("layers").pp(index),
                                        sidecars,
                                        index,
                                    ),
                                    None => ZImageTransformerBlock::new(
                                        &self.cfg,
                                        true,
                                        view.pp("layers").pp(index),
                                    ),
                                }
                                .map_err(candle_gen::CandleError::from)
                            })
                            .collect::<candle_gen::Result<Vec<_>>>()?;
                        if let Some(plan) = adapters {
                            for (offset, block) in blocks.iter_mut().enumerate() {
                                let index = first + offset;
                                block.visit_adaptable_mut(
                                    &format!("layers.{index}"),
                                    &mut |path, linear| {
                                        plan.apply_projection(path, linear, self.device())?;
                                        Ok(())
                                    },
                                )?;
                            }
                        }
                        for (offset, block) in blocks.iter().enumerate() {
                            candle_gen::check_cancel(cancel)?;
                            let index = first + offset;
                            state = block.forward_with_attention_plan(
                                &state,
                                Some(attention_mask),
                                cos,
                                sin,
                                Some(adaln),
                                attention_plan,
                            )?;
                            state = after_layer(index, state)?;
                        }
                        Ok(state)
                    },
                )?;
            }
        }
        Ok(hidden)
    }

    pub(crate) fn control_finish(&self, hidden: &Tensor, adaln: &Tensor) -> Result<Tensor> {
        self.final_layer.forward(hidden, adaln)
    }

    /// Forward pass — returns the **raw** DiT velocity `(B, C, F, H, W)` (the pipeline negates it).
    /// Byte-faithful to the stock model's forward (identical phases 1–13).
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with_attention_budget(
            x,
            t,
            cap_feats,
            cap_mask,
            candle_gen::ATTN_SCORES_BUDGET,
        )
    }

    /// [`Self::forward`] with an explicit attention-score budget. Query-row chunking preserves every
    /// query's complete key/value domain while bounding the transient score/probability tensors.
    pub fn forward_with_attention_budget(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
        attention_scores_budget: usize,
    ) -> Result<Tensor> {
        into_candle_core(self.forward_with_attention_plan(
            x,
            t,
            cap_feats,
            cap_mask,
            plan_from_budget(attention_scores_budget),
        ))
    }

    /// Request-scoped bounded forward. Only this path attaches a cancel flag; the raw-budget overload
    /// above remains the compatibility surface for tests and non-request-scoped correctness guards.
    pub fn forward_with_attention_plan(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_memory(
            x,
            t,
            cap_feats,
            cap_mask,
            attention_plan,
            self.cfg.n_layers.max(1),
        )
    }

    /// Request-scoped forward with both the bounded-attention plan and the admitted transformer
    /// window. Resident models ignore `transformer_window`; streamed models drive the shared Candle
    /// block-window scheduler and materialize every block in a window before executing any of them.
    pub fn forward_with_memory(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
    ) -> candle_gen::Result<Tensor> {
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = self.x_embedder.forward(&x_patches)?;
        let img_seq_len = x.dim(1)?;

        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let text_len = cap_feats.dim(1)?;
        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0), device)?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        let cap_normed = self.cap_embedder_norm.forward(cap_feats)?;
        let mut cap = self.cap_embedder_linear.forward(&cap_normed)?;

        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        for layer in &self.noise_refiner {
            x = layer.forward_with_attention_plan(
                &x,
                Some(&x_attn_mask),
                &x_cos,
                &x_sin,
                Some(&adaln_input),
                attention_plan,
            )?;
        }
        for layer in &self.context_refiner {
            cap = layer.forward_with_attention_plan(
                &cap,
                Some(&cap_attn_mask),
                &cap_cos,
                &cap_sin,
                None,
                attention_plan,
            )?;
        }

        let unified = Tensor::cat(&[&x, &cap], 1)?;
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        let mut unified = unified;
        match &self.layers {
            TransformerLayers::Resident(layers) => {
                for layer in layers {
                    unified = layer.forward_with_attention_plan(
                        &unified,
                        Some(&unified_attn_mask),
                        &unified_cos,
                        &unified_sin,
                        Some(&adaln_input),
                        attention_plan,
                    )?;
                }
            }
            TransformerLayers::Streamed {
                weights,
                sidecars,
                adapters,
            } => {
                let block_plan = BlockPlan::new(self.cfg.n_layers, transformer_window)?;
                let uncancelled = candle_gen::gen_core::CancelFlag::default();
                let cancel = attention_plan.cancel.unwrap_or(&uncancelled);
                unified = candle_gen::block_window::run_windowed(
                    self.device(),
                    &block_plan,
                    cancel,
                    unified,
                    || Ok(weights.clone()),
                    |mut state, view, range| {
                        // Materialize the whole window before the first forward. Loading inside the
                        // block loop would silently execute a window of one regardless of selection.
                        let first = range.start;
                        let mut blocks = range
                            .map(|index| {
                                match sidecars.as_deref() {
                                    Some(sidecars) => ZImageTransformerBlock::new_streamed(
                                        &self.cfg,
                                        true,
                                        view.pp("layers").pp(index),
                                        sidecars,
                                        index,
                                    ),
                                    None => ZImageTransformerBlock::new(
                                        &self.cfg,
                                        true,
                                        view.pp("layers").pp(index),
                                    ),
                                }
                                .map_err(candle_gen::CandleError::from)
                            })
                            .collect::<candle_gen::Result<Vec<_>>>()?;
                        if let Some(plan) = adapters {
                            for (offset, block) in blocks.iter_mut().enumerate() {
                                let index = first + offset;
                                block.visit_adaptable_mut(
                                    &format!("layers.{index}"),
                                    &mut |path, linear| {
                                        plan.apply_projection(path, linear, self.device())?;
                                        Ok(())
                                    },
                                )?;
                            }
                        }
                        for block in &blocks {
                            candle_gen::check_cancel(cancel)?;
                            state = block.forward_with_attention_plan(
                                &state,
                                Some(&unified_attn_mask),
                                &unified_cos,
                                &unified_sin,
                                Some(&adaln_input),
                                attention_plan,
                            )?;
                        }
                        Ok(state)
                    },
                )?;
            }
        }

        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;
        Ok(unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )?)
    }

    /// The device the DiT weights live on — the forward-time residual factors are read on the CPU and
    /// moved here at install (else the residual matmul is a device mismatch). sc-11105.
    pub fn device(&self) -> &Device {
        self.x_pad_token.device()
    }

    /// Walk every adaptable projection, invoking `f(path, &mut QLinear)` once each with the projection's
    /// canonical DiT dotted path — the same paths [`crate::adapters::classify_lora_key`] resolves a LoRA
    /// key to (the embedders, the per-block attention/feed-forward/adaLN projections across the three
    /// stacks, and the final layer). The additive installer
    /// ([`crate::adapters::install_additive`]) pushes a resolved LoRA/LoKr residual onto each matched
    /// projection so a user adapter applies on a packed q4/q8 tier with the base kept packed (sc-11105).
    pub(crate) fn visit_adaptable_for_install(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f("cap_embedder.1", &mut self.cap_embedder_linear)?;
        f("all_x_embedder.2-1", &mut self.x_embedder)?;
        self.t_embedder.visit_adaptable_mut("t_embedder", f)?;
        for (i, blk) in self.noise_refiner.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("noise_refiner.{i}"), f)?;
        }
        for (i, blk) in self.context_refiner.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("context_refiner.{i}"), f)?;
        }
        match &mut self.layers {
            TransformerLayers::Resident(layers) => {
                for (i, blk) in layers.iter_mut().enumerate() {
                    blk.visit_adaptable_mut(&format!("layers.{i}"), f)?;
                }
            }
            TransformerLayers::Streamed { weights, .. } => {
                for i in 0..self.cfg.n_layers {
                    let mut block =
                        ZImageTransformerBlock::new(&self.cfg, true, weights.pp("layers").pp(i))?;
                    let visited = block.visit_adaptable_mut(&format!("layers.{i}"), f);
                    let synchronized = weights.device().synchronize();
                    drop(block);
                    visited?;
                    synchronized?;
                }
            }
        }
        self.final_layer
            .visit_adaptable_mut("all_final_layer.2-1", f)?;
        Ok(())
    }

    pub(crate) fn retain_streamed_adapter_plan(&mut self, plan: crate::adapters::AdditivePlan) {
        if let TransformerLayers::Streamed { adapters, .. } = &mut self.layers {
            *adapters = Some(plan);
        }
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored DENSE path to the stock candle-transformers DiT: built from the *same*
    //! `VarMap`-backed weights (no `.scales`, so every `QLinear` takes the dense arm), the two must
    //! produce bit-identical forward output — the guard that the packed-seam vendoring changed nothing
    //! numerically on a dense tier. This is the z-image half of the stock-vs-vendored DiT numeric parity
    //! coverage tracked by sc-9443 (the flux half lives in `candle-gen-flux` `packed_dit.rs`, where the
    //! diffusers→BFL layout difference additionally requires a load-time QKV remap).
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::quant::{pack_mlx_affine, PackedConfig};
    use candle_transformers::models::z_image::preprocess::prepare_inputs;
    use candle_transformers::models::z_image::transformer::{
        Config, ZImageTransformer2DModel as StockModel,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A tiny Z-Image-shaped config (`head_dim` locked to 128 by `axes_dims=[32,48,48]`): a single head
    /// at `dim=128`, 2 main layers + 1 refiner each — exercises every vendored path cheaply on CPU.
    fn tiny_cfg() -> Config {
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 128;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.n_layers = 2;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 64;
        cfg.set_use_accelerated_attn(false);
        cfg
    }

    struct SidecarFixture(PathBuf);

    impl SidecarFixture {
        fn new(bits: usize) -> Self {
            let path = std::env::temp_dir().join(format!(
                "z-image-sc16510-sidecar-{}-{bits}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for SidecarFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn assert_streamed_projection_uses_byte_exact_sidecar(bits: usize) -> Result<()> {
        let fixture = SidecarFixture::new(bits);
        let values: Vec<f32> = (0..4 * 64)
            .map(|i| ((i * 17 + i / 11) % 47) as f32 / 13.0 - 1.0)
            .collect();
        let dense = Tensor::from_vec(values, (4, 64), &Device::Cpu)?;
        let (weight, scales, biases) = pack_mlx_affine(&dense, bits, 64)?;
        let bias = Tensor::from_vec(vec![0.25f32, -0.5, 0.75, -1.0], 4, &Device::Cpu)?;
        let source_path = fixture.0.join("model.safetensors");
        candle_gen::candle_core::safetensors::save(
            &HashMap::from([
                ("layers.0.attention.to_q.weight".to_owned(), weight),
                ("layers.0.attention.to_q.scales".to_owned(), scales),
                ("layers.0.attention.to_q.biases".to_owned(), biases),
                ("layers.0.attention.to_q.bias".to_owned(), bias),
            ]),
            &source_path,
        )?;
        // SAFETY: immutable fixture files live for the duration of the mapping.
        let source = unsafe { MmapedSafetensors::new(&source_path)? };
        let sidecars = PackedWeightSidecars::prepare_prefix_cancelable(
            &source,
            &fixture.0,
            PackedConfig {
                bits: bits as i32,
                group_size: 64,
            },
            &Device::Cpu,
            &candle_gen::gen_core::CancelFlag::default(),
            "layers.",
        )?;
        assert_eq!(sidecars.created_count(), 1);
        let vb = VarBuilder::from_backend(Box::new(source), DType::F32, Device::Cpu)
            .pp("layers")
            .pp(0)
            .pp("attention");
        let old = QLinear::linear_detect(64, 4, &vb, "to_q", true)?;
        let missing =
            streamed_linear_detect(64, 4, &vb, "to_q", true, &sidecars, "layers.1.attention")
                .err()
                .expect("a packed streamed projection must not fall back to source conversion");
        assert!(missing
            .to_string()
            .contains("no prepared device-format sidecar"));
        let prepared =
            streamed_linear_detect(64, 4, &vb, "to_q", true, &sidecars, "layers.0.attention")?;
        let input = Tensor::randn(0f32, 1f32, (3, 64), &Device::Cpu)?;
        let old_output = old.forward(&input)?;
        let prepared_output = prepared.forward(&input)?;
        assert_eq!(
            old_output.to_vec2::<f32>()?,
            prepared_output.to_vec2::<f32>()?,
            "q{bits} sidecar path changed the packed projection output"
        );
        Ok(())
    }

    #[test]
    fn streamed_q4_projection_uses_byte_exact_device_format_sidecar() -> Result<()> {
        assert_streamed_projection_uses_byte_exact_sidecar(4)
    }

    #[test]
    fn streamed_q8_projection_uses_byte_exact_device_format_sidecar() -> Result<()> {
        assert_streamed_projection_uses_byte_exact_sidecar(8)
    }

    #[test]
    fn vendored_dense_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The vendored model is built first, populating the VarMap with random weights; the stock model
        // then reads the SAME parameters (identical names/shapes), so any output difference is a
        // forward-logic difference. No `.scales` present, so every QLinear is `Dense`.
        let vendored = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let stock = StockModel::new(&cfg, vb).unwrap();

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let y_v = vendored
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();
        let y_s = stock
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();

        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "vendored dense DiT diverged from stock by {diff}"
        );
    }

    #[test]
    fn attention_query_chunking_matches_the_unbounded_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let full = model
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();
        let chunked = model
            .forward_with_attention_budget(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                2,
            )
            .unwrap();
        let diff = (full - chunked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "query-chunked attention changed the DiT output by {diff}"
        );
    }

    #[test]
    fn streamed_windows_match_resident_forward_including_ragged_tail() {
        let dev = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.n_layers = 3;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Populate one immutable host-backed source, then build the streamed front/refiner shell from
        // that same source. Two forces a 2+1 ragged tail; the larger published candidates exercise
        // the all-covering window without changing the contract's production range.
        let resident = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let expected = resident
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();

        for window in [1, 2, 4, 8, 15, 30] {
            let streamed = ZImageTransformer2DModel::new_block_streamed(&cfg, vb.clone()).unwrap();
            let actual = streamed
                .forward_with_memory(
                    &prepared.latents,
                    &t,
                    &prepared.cap_feats,
                    &prepared.cap_mask,
                    plan_from_budget(candle_gen::ATTN_SCORES_BUDGET),
                    window,
                )
                .unwrap();
            let diff = (expected.clone() - actual)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-5,
                "streamed window {window} changed the DiT output by {diff}"
            );
        }
    }

    #[test]
    fn streamed_window_rejects_zero_and_preserves_typed_cancellation() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Populate all layer tensors once; the streamed model retains only the source view.
        let _resident = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let streamed = ZImageTransformer2DModel::new_block_streamed(&cfg, vb).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let zero = streamed
            .forward_with_memory(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                plan_from_budget(candle_gen::ATTN_SCORES_BUDGET),
                0,
            )
            .expect_err("a zero-width transformer window must be rejected");
        assert!(zero.to_string().contains("window"));

        let cancel = candle_gen::gen_core::CancelFlag::default();
        cancel.cancel();
        let plan = plan_from_budget(candle_gen::ATTN_SCORES_BUDGET).with_cancel(&cancel);
        let canceled = streamed
            .forward_with_memory(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                plan,
                1,
            )
            .expect_err("a canceled streamed request must stop before producing output");
        assert!(matches!(canceled, candle_gen::CandleError::Canceled));
    }

    #[test]
    fn constrained_request_cancel_stops_inside_the_packed_dit_as_typed_canceled() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let request = candle_gen::gen_core::GenerationRequest::default();
        request.cancel.cancel();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(2, false))
            .with_cancel(&request.cancel);
        let error = model
            .forward_with_attention_plan(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                plan,
            )
            .expect_err("a canceled constrained request must not complete the DiT forward");
        assert!(matches!(error, candle_gen::CandleError::Canceled));
        let contract_error: candle_gen::gen_core::Error = error.into();
        assert!(matches!(
            contract_error,
            candle_gen::gen_core::Error::Canceled
        ));
    }

    /// **Additive install on the vendored DiT (sc-11105).** A bare-dotted LoRA over two real `layers.0`
    /// attention projections installs as forward-time residuals: the report counts both, the DiT forward
    /// shifts vs the un-adapted model, and no target is left unresolved — proving the visitor's canonical
    /// paths line up with `adapters::classify_lora_key` (a path mismatch would apply nothing and error).
    /// This exercises the packed-tier install wiring end-to-end on a dense-base fixture; the base-agnostic
    /// additive path is byte-equal on a packed base (the stays-packed / exact-residual property is proven
    /// at the `candle_gen::quant::AdaptLinear` unit level).
    #[test]
    fn install_additive_lora_on_vendored_dit_applies_and_shifts() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        use std::collections::HashMap as Map;
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Two DiTs from the SAME params: one baseline, one adapted.
        let base = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let mut adapted = ZImageTransformer2DModel::new(&cfg, vb).unwrap();

        // A LoRA over layers.0 `to_q` + `to_v` (both [dim, dim] at this tiny cfg), bare-dotted.
        let (out_dim, in_dim, rank) = (cfg.dim, cfg.dim, 2usize);
        let mut map: Map<String, Tensor> = Map::new();
        for proj in ["to_q", "to_v"] {
            let path = format!("layers.0.attention.{proj}");
            map.insert(
                format!("{path}.lora_A.weight"),
                Tensor::randn(0f32, 0.5f32, (rank, in_dim), &dev).unwrap(),
            );
            map.insert(
                format!("{path}.lora_B.weight"),
                Tensor::randn(0f32, 0.5f32, (out_dim, rank), &dev).unwrap(),
            );
        }
        let tmp = std::env::temp_dir().join(format!(
            "sc11105_install_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        let report = crate::adapters::install_additive(
            &mut adapted,
            &[AdapterSpec::new(tmp.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(report.applied, 2, "both to_q + to_v residuals installed");
        assert!(
            report.skipped_targets.is_empty(),
            "no unresolved targets (visitor paths match classify)"
        );

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let y_base = base
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();
        let y_adapt = adapted
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();
        let shift = (y_adapt - y_base)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-4,
            "additive LoRA did not shift the DiT forward ({shift})"
        );
    }

    /// [`crate::adapters::install_additive`] rejects an off-surface LoRA (no DiT target matched) and a
    /// LoHa adapter (no allocation-free structured form on a packed tier) — the loud guards that keep a
    /// misconfigured adapter from silently rendering unadapted on the packed tier (sc-11105).
    #[test]
    fn install_additive_rejects_off_surface_and_loha() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        use std::collections::HashMap as Map;
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        // Off-surface: a nonexistent block → no target matched → err.
        let mut dit = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let mut off: Map<String, Tensor> = Map::new();
        off.insert(
            "layers.99.attention.to_q.lora_A.weight".into(),
            Tensor::zeros((2, cfg.dim), DType::F32, &dev).unwrap(),
        );
        off.insert(
            "layers.99.attention.to_q.lora_B.weight".into(),
            Tensor::zeros((cfg.dim, 2), DType::F32, &dev).unwrap(),
        );
        let tmp1 =
            std::env::temp_dir().join(format!("sc11105_off_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&off, &tmp1).unwrap();
        let r = crate::adapters::install_additive(
            &mut dit,
            &[AdapterSpec::new(tmp1.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&tmp1).ok();
        assert!(
            r.is_err(),
            "an off-surface LoRA must error (no target matched)"
        );

        // LoHa: `hada_*` keys → rejected with a pointer to the dense tier.
        let mut dit2 = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        let mut loha: Map<String, Tensor> = Map::new();
        let path = "layers.0.attention.to_q";
        for k in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"] {
            loha.insert(
                format!("{path}.{k}"),
                Tensor::zeros((cfg.dim, 1), DType::F32, &dev).unwrap(),
            );
        }
        let tmp2 =
            std::env::temp_dir().join(format!("sc11105_loha_{}.safetensors", std::process::id()));
        candle_gen::candle_core::safetensors::save(&loha, &tmp2).unwrap();
        let r2 = crate::adapters::install_additive(
            &mut dit2,
            &[AdapterSpec::new(tmp2.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&tmp2).ok();
        assert!(
            r2.is_err(),
            "a LoHa adapter must be rejected on the packed additive path"
        );
    }
}
