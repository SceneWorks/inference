//! Z-Image DiT shared by native render, control, and training.
//!
//! Checkpoint names and block topology are the same for all routes. Ordinary packed
//! and additive inference use the shared Candle projection, including prepared sidecars
//! for streamed blocks. Dense edit, the VACE side blocks, and the trainer keep the
//! differentiable normalization and attention path. Their trainable projection leaves
//! preserve LoRA/LoKr, frozen training adapters, and the trainer's packed QKV preparation.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, TensorId, D};
use candle_gen::block_window::BlockPlan;
#[cfg(test)]
use candle_gen::gen_core::attention_budget::AttentionBudget;
use candle_gen::gen_core::attention_budget::AttentionPlan;
use candle_gen::quant::{AdaptLinear, PackedWeightSidecars};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{lora_linear, lora_linear_no_bias, LoraHost, LoraLinear};
use candle_gen::train::projection::FrozenProjectionPack;
use candle_nn::{RmsNorm, VarBuilder};
use candle_transformers::models::z_image::preprocess::{prepare_inputs, PreparedInputs};
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, Config, LayerNormNoParams,
    RopeEmbedder, ADALN_EMBED_DIM, FREQUENCY_EMBEDDING_SIZE, MAX_PERIOD,
};

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
) -> Result<AdaptLinear> {
    let scales_key = format!("{base}.scales");
    if !vb.contains_tensor(&scales_key) {
        return AdaptLinear::linear_detect(in_dim, out_dim, vb, base, bias);
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
    Ok(AdaptLinear::from_packed(packed, in_dim, out_dim))
}

fn plan_from_budget(budget: usize) -> AttentionPlan<'static> {
    AttentionPlan::budgeted(candle_gen::attention::attention_budget_from_usize(budget))
}

fn into_candle_core(result: candle_gen::Result<Tensor>) -> Result<Tensor> {
    result.map_err(|error| match error {
        candle_gen::CandleError::Candle(error) => error,
        other => candle_gen::candle_core::Error::Msg(other.to_string()),
    })
}

fn rms_diff(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    candle_gen::train::profile::cohort_reuse("z-image:shared-differentiable-RMS");
    candle_gen::train::norm::rms_norm_cast_before_weight(x, norm.weight(), norm.eps())
}

fn default_attention_plan() -> AttentionPlan<'static> {
    plan_from_budget(candle_gen::ATTN_SCORES_BUDGET)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Execution {
    Differentiable,
    Inference,
}

impl Execution {
    fn rms(self, norm: &RmsNorm, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Differentiable => rms_diff(norm, input),
            Self::Inference => __model_proof_output!(norm.forward(input)),
        }
    }

    fn attention_plan(self, requested: AttentionPlan<'_>) -> AttentionPlan<'_> {
        match self {
            Self::Differentiable => default_attention_plan(),
            Self::Inference => requested,
        }
    }
}

#[derive(Debug, Clone)]
enum Projection {
    Trainable(LoraLinear),
    Inference(AdaptLinear),
}

impl Projection {
    fn load(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        key: &str,
        bias: bool,
        execution: Execution,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        match execution {
            Execution::Differentiable => {
                let weights = vb.pp(key);
                let linear = if bias {
                    lora_linear(in_dim, out_dim, weights)?
                } else {
                    lora_linear_no_bias(in_dim, out_dim, weights)?
                };
                Ok(Self::Trainable(linear))
            }
            Execution::Inference => {
                let linear = match sidecars {
                    Some((sidecars, prefix)) => {
                        streamed_linear_detect(in_dim, out_dim, vb, key, bias, sidecars, prefix)?
                    }
                    None => AdaptLinear::linear_detect(in_dim, out_dim, vb, key, bias)?,
                };
                Ok(Self::Inference(linear))
            }
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Trainable(linear) => linear.forward(input),
            Self::Inference(linear) => linear.forward(input),
        }
    }

    fn trainable(&self) -> &LoraLinear {
        match self {
            Self::Trainable(linear) => linear,
            Self::Inference(_) => unreachable!("training projection required"),
        }
    }

    fn trainable_mut(&mut self) -> &mut LoraLinear {
        match self {
            Self::Trainable(linear) => linear,
            Self::Inference(_) => unreachable!("training projection required"),
        }
    }

    fn visit_inference(
        &mut self,
        path: &str,
        visit: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        match self {
            Self::Inference(linear) => visit(path, linear),
            Self::Trainable(_) => Err(candle_gen::CandleError::Msg(
                "z-image: additive inference requires inference projections".into(),
            )),
        }
    }
}

/// Embedders and the final layer have no trainable adapter targets. Their differentiable
/// route explicitly loads dense frozen projections rather than probing for packed siblings.
fn frozen_projection(
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilder,
    key: &str,
    execution: Execution,
) -> Result<AdaptLinear> {
    match execution {
        Execution::Differentiable => AdaptLinear::linear(in_dim, out_dim, vb.pp(key)),
        Execution::Inference => AdaptLinear::linear_detect(in_dim, out_dim, vb, key, true),
    }
}

struct TimestepEmbedder {
    linear1: AdaptLinear,
    linear2: AdaptLinear,
    frequency_embedding_size: usize,
    /// The MLP input dtype — the model's compute dtype (`vb.dtype()`), matching the stock embedder's
    /// `self.linear1.weight().dtype()` (dense bf16 tier ⇒ bf16). AdaptLinear's dense arm requires the
    /// activation dtype to match the weight; its packed arm dequants the weight to this dtype (parity).
    dtype: DType,
}

impl TimestepEmbedder {
    fn new(out_size: usize, mid_size: usize, vb: VarBuilder, execution: Execution) -> Result<Self> {
        let dtype = vb.dtype();
        let mlp = vb.pp("mlp");
        let linear1 = frozen_projection(FREQUENCY_EMBEDDING_SIZE, mid_size, &mlp, "0", execution)?;
        let linear2 = frozen_projection(mid_size, out_size, &mlp, "2", execution)?;
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
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.mlp.0"), &mut self.linear1)?;
        f(&format!("{prefix}.mlp.2"), &mut self.linear2)?;
        Ok(())
    }
}

/// The default LoRA target suffixes — the attention projections, matching the SDXL trainer's
/// [`SDXL_ATTN_TARGETS`](candle_gen::train::lora::SDXL_ATTN_TARGETS), the torch
/// `DEFAULT_LORA_TARGET_MODULES`, and the MLX Z-Image trainer. `to_out.0` is the first element of
/// diffusers' `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const Z_IMAGE_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

#[derive(Debug, Clone)]
struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_q: candle_nn::rms_norm(head_dim, eps, vb.pp("norm_q"))?,
            norm_k: candle_nn::rms_norm(head_dim, eps, vb.pp("norm_k"))?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, execution: Execution) -> Result<(Tensor, Tensor)> {
        Ok((
            execution.rms(&self.norm_q, q)?,
            execution.rms(&self.norm_k, k)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct ZImageAttention {
    qkv_pack: Option<FrozenProjectionPack>,
    to_q: Projection,
    to_k: Projection,
    to_v: Projection,
    to_out: Projection,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
    use_accelerated_attn: bool,
    execution: Execution,
}

impl ZImageAttention {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        execution: Execution,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        if execution == Execution::Differentiable {
            debug_assert_eq!(cfg.n_kv_heads, cfg.n_heads);
        }
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        // Packed bases are the full dotted key prefixes (the `.scales` siblings live directly under
        // `attention.to_q` … `attention.to_out.0`), so the detect uses the base string — never `.pp()`
        // past the sibling (the key-remap trap for `to_out.0`).
        let load = |in_dim, out_dim, base| {
            Projection::load(in_dim, out_dim, &vb, base, false, execution, sidecars)
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
            qkv_pack: None,
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
            use_accelerated_attn: cfg.use_accelerated_attn,
            execution,
        })
    }

    fn forward_with_attention_plan_and_pag(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
        pag_x_len: Option<usize>,
    ) -> candle_gen::Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        let (q, k, v) = match &self.qkv_pack {
            Some(pack) => {
                let [q, k, v] = pack
                    .forward_lora(
                        hidden_states,
                        &[
                            self.to_q.trainable(),
                            self.to_k.trainable(),
                            self.to_v.trainable(),
                        ],
                    )?
                    .try_into()
                    .map_err(|_| {
                        candle_gen::CandleError::Msg(
                            "z-image: packed QKV did not return three projections".into(),
                        )
                    })?;
                candle_gen::train::profile::cohort_reuse("zimage:actual-frozen-QKV-pack");
                (q, k, v)
            }
            None => (
                self.to_q.forward(hidden_states)?,
                self.to_k.forward(hidden_states)?,
                self.to_v.forward(hidden_states)?,
            ),
        };

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_heads, self.head_dim))?;

        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k, self.execution)?
        } else {
            (q, k)
        };

        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let context = crate::common::apply_attention_with_pag(&q, &k, &v, pag_x_len, |q, k, v| {
            self.attention_dispatch(q, k, v, attention_mask, scale, attention_plan)
        })?;

        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok(self.to_out.forward(&context)?)
    }

    /// Visit the four attention projections (`{prefix}.to_q/to_k/to_v/to_out.0`) — the surface the
    /// Z-Image trainer's LoRA/LoKr adapts (sc-11105). `to_out` carries the diffusers `.0` suffix.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.to_q.visit_inference(&format!("{prefix}.to_q"), f)?;
        self.to_k.visit_inference(&format!("{prefix}.to_k"), f)?;
        self.to_v.visit_inference(&format!("{prefix}.to_v"), f)?;
        self.to_out
            .visit_inference(&format!("{prefix}.to_out.0"), f)?;
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
        if self.execution == Execution::Differentiable {
            let mask = Self::additive_mask(mask, q.dtype())?;
            return candle_gen::fused_attention::training_sdpa_bhsd(
                q,
                k,
                v,
                scale,
                mask.as_ref(),
                |scores| candle_nn::ops::softmax(scores, D::Minus1),
                attention_plan,
            );
        }
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
        let sdpa_mask = self.prepare_sdpa_mask(mask, q, k.dim(2)?)?;
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
        let m = Self::additive_mask(mask, q.dtype())?;
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

    fn additive_mask(mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
        mask.map(|mask| {
            let mask = mask.unsqueeze(1)?.unsqueeze(2)?.to_dtype(dtype)?;
            (mask - 1.0)? * 1e9
        })
        .transpose()
    }

    fn prepare_sdpa_mask(
        &self,
        mask: Option<&Tensor>,
        q: &Tensor,
        k_len: usize,
    ) -> Result<Option<Tensor>> {
        Self::additive_mask(mask, q.dtype())?
            .map(|mask| {
                let (batch, _, queries, _) = q.dims4()?;
                mask.broadcast_as((batch, self.n_heads, queries, k_len))
            })
            .transpose()
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    w1: Projection,
    w2: Projection,
    w3: Projection,
}

impl FeedForward {
    fn new(
        dim: usize,
        hidden_dim: usize,
        vb: VarBuilder,
        execution: Execution,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        Ok(Self {
            w1: Projection::load(dim, hidden_dim, &vb, "w1", false, execution, sidecars)?,
            w2: Projection::load(hidden_dim, dim, &vb, "w2", false, execution, sidecars)?,
            w3: Projection::load(dim, hidden_dim, &vb, "w3", false, execution, sidecars)?,
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
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.w1.visit_inference(&format!("{prefix}.w1"), f)?;
        self.w2.visit_inference(&format!("{prefix}.w2"), f)?;
        self.w3.visit_inference(&format!("{prefix}.w3"), f)?;
        Ok(())
    }
}

struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: AdaptLinear,
    adaln_silu: AdaptLinear,
}

impl FinalLayer {
    fn new(
        hidden_size: usize,
        out_channels: usize,
        vb: VarBuilder,
        execution: Execution,
    ) -> Result<Self> {
        let norm_final = LayerNormNoParams::new(1e-6);
        let linear = frozen_projection(hidden_size, out_channels, &vb, "linear", execution)?;
        let adaln_dim = hidden_size.min(ADALN_EMBED_DIM);
        // The stock builds this at `adaLN_modulation.1` (index `.0` is a param-free SiLU).
        let adaln_silu =
            frozen_projection(adaln_dim, hidden_size, &vb, "adaLN_modulation.1", execution)?;
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
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&format!("{prefix}.linear"), &mut self.linear)?;
        f(
            &format!("{prefix}.adaLN_modulation.1"),
            &mut self.adaln_silu,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ZImageTransformerBlock {
    execution: Execution,
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<Projection>,
}

impl ZImageTransformerBlock {
    pub(crate) fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, modulation, vb, Execution::Differentiable, None)
    }

    fn new_inference(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, modulation, vb, Execution::Inference, None)
    }

    fn new_streamed(
        cfg: &Config,
        modulation: bool,
        vb: VarBuilder,
        sidecars: &PackedWeightSidecars,
        block_index: usize,
    ) -> Result<Self> {
        let block_prefix = format!("layers.{block_index}");
        Self::build(
            cfg,
            modulation,
            vb,
            Execution::Inference,
            Some((sidecars, &block_prefix)),
        )
    }

    fn build(
        cfg: &Config,
        modulation: bool,
        vb: VarBuilder,
        execution: Execution,
        sidecars: Option<(&PackedWeightSidecars, &str)>,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attn_prefix = sidecars.map(|(_, prefix)| format!("{prefix}.attention"));
        let ff_prefix = sidecars.map(|(_, prefix)| format!("{prefix}.feed_forward"));
        let attention = ZImageAttention::new(
            cfg,
            vb.pp("attention"),
            execution,
            sidecars
                .zip(attn_prefix.as_deref())
                .map(|((cache, _), prefix)| (cache, prefix)),
        )?;
        let feed_forward = FeedForward::new(
            dim,
            hidden_dim,
            vb.pp("feed_forward"),
            execution,
            sidecars
                .zip(ff_prefix.as_deref())
                .map(|((cache, _), prefix)| (cache, prefix)),
        )?;

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
            let prefix = sidecars.map(|(_, prefix)| format!("{prefix}.adaLN_modulation"));
            Some(Projection::load(
                adaln_dim,
                4 * dim,
                &adaln,
                "0",
                true,
                execution,
                sidecars
                    .zip(prefix.as_deref())
                    .map(|((cache, _), prefix)| (cache, prefix)),
            )?)
        } else {
            None
        };

        Ok(Self {
            execution,
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        let attention = &mut self.attention;
        for linear in [
            &mut attention.to_q,
            &mut attention.to_k,
            &mut attention.to_v,
            &mut attention.to_out,
        ] {
            f(linear.trainable_mut())?;
        }
        Ok(())
    }

    fn visit_linears(&self, f: &mut dyn FnMut(&LoraLinear)) {
        let a = &self.attention;
        for lin in [&a.to_q, &a.to_k, &a.to_v, &a.to_out] {
            f(lin.trainable());
        }
        let ff = &self.feed_forward;
        for lin in [&ff.w1, &ff.w2, &ff.w3] {
            f(lin.trainable());
        }
        if let Some(adaln) = &self.adaln_modulation {
            f(adaln.trainable());
        }
    }

    /// [`visit_linears`](Self::visit_linears), mutably.
    fn visit_linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.visit_lora_mut(f)?;
        let ff = &mut self.feed_forward;
        for lin in [&mut ff.w1, &mut ff.w2, &mut ff.w3] {
            f(lin.trainable_mut())?;
        }
        if let Some(adaln) = &mut self.adaln_modulation {
            f(adaln.trainable_mut())?;
        }
        Ok(())
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        into_candle_core(self.forward_with_attention_plan(
            x,
            attn_mask,
            cos,
            sin,
            adaln_input,
            default_attention_plan(),
        ))
    }

    pub(crate) fn forward_with_attention_plan(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_attention_plan_and_pag(
            x,
            attn_mask,
            cos,
            sin,
            adaln_input,
            attention_plan,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_attention_plan_and_pag(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
        attention_plan: AttentionPlan<'_>,
        pag_x_len: Option<usize>,
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

            let normed = self.execution.rms(&self.attention_norm1, x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward_with_attention_plan_and_pag(
                &scaled,
                attn_mask,
                cos,
                sin,
                attention_plan,
                pag_x_len,
            )?;
            let attn_out = self.execution.rms(&self.attention_norm2, &attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            let normed = self.execution.rms(&self.ffn_norm1, &x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.execution.rms(&self.ffn_norm2, &ffn_out)?;
            Ok((x + gate_mlp.broadcast_mul(&ffn_out)?)?)
        } else {
            let normed = self.execution.rms(&self.attention_norm1, x)?;
            let attn_out = self.attention.forward_with_attention_plan_and_pag(
                &normed,
                attn_mask,
                cos,
                sin,
                attention_plan,
                pag_x_len,
            )?;
            let attn_out = self.execution.rms(&self.attention_norm2, &attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.execution.rms(&self.ffn_norm1, &x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.execution.rms(&self.ffn_norm2, &ffn_out)?;
            Ok((x + ffn_out)?)
        }
    }

    /// Visit every adaptable projection in the block under `{prefix}` — the attention + feed-forward
    /// projections, plus the optional `{prefix}.adaLN_modulation.0` (the four RMSNorms are dense, not
    /// projections). sc-11105.
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attention
            .visit_adaptable_mut(&format!("{prefix}.attention"), f)?;
        self.feed_forward
            .visit_adaptable_mut(&format!("{prefix}.feed_forward"), f)?;
        if let Some(adaln) = self.adaln_modulation.as_mut() {
            adaln.visit_inference(&format!("{prefix}.adaLN_modulation.0"), f)?;
        }
        Ok(())
    }
}

/// The constant side tensors the **main** transformer layers and the final layer consume, produced
/// once by [`ZImageTransformer2DModel::forward_pre_main`]. None of these depend on an adapter `Var`
/// (the AdaLN conditioning is the frozen `t_embedder`'s output; the RoPE tables + masks are pure
/// functions of the token geometry), so the gradient-checkpointing path captures them as detached
/// constants shared across every recomputed main-layer segment — exactly how the SDXL trainer captures
/// the time embedding + encoder states. The image-token bookkeeping (`img_seq_len`/`orig_size`) is what
/// [`ZImageTransformer2DModel::velocity_out`] needs to narrow + unpatchify the image portion back out.
#[derive(Debug, Clone)]
pub struct MainContext {
    adaln_input: Tensor,
    unified_cos: Tensor,
    unified_sin: Tensor,
    unified_attn_mask: Tensor,
    img_seq_len: usize,
    orig_size: (usize, usize, usize),
}

#[derive(Debug, Clone)]
struct StreamGeometry {
    cos: Tensor,
    sin: Tensor,
    mask: Tensor,
}

struct PositionedGeometry {
    image_positions: Tensor,
    caption_positions: Tensor,
    image: StreamGeometry,
    caption: StreamGeometry,
}

#[derive(Debug, Clone)]
struct ConditioningGeometry {
    image: StreamGeometry,
    caption: StreamGeometry,
    unified: StreamGeometry,
}

/// One Z-Image graph with resident or bounded-window main layers.
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    execution: Execution,
    device: Device,
    preparation_owner: Arc<()>,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: AdaptLinear,
    x_embedder: AdaptLinear,
    final_layer: FinalLayer,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: TransformerLayers,
    rope_embedder: RopeEmbedder,
    cfg: Config,
    training_captions: Vec<(TensorId, Tensor, usize)>,
    training_geometry: std::collections::HashMap<
        (usize, usize, usize, usize, TensorId, TensorId),
        ConditioningGeometry,
    >,
    /// Per-instance scheduler trace for mutation-sensitive plain-forward tests. Test-only and
    /// mutex-backed so parallel tests never share recorder state.
    #[cfg(test)]
    streamed_window_trace: std::sync::Mutex<Vec<(usize, usize)>>,
    #[cfg(test)]
    prepare_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// Request-owned geometry conditioning. RoPE tables and the refined caption stream are independent of
/// the current denoise latent and timestep, so the render creates this once per CFG branch.
pub(crate) struct PreparedConditioning {
    geometry: (usize, usize, usize, usize, usize),
    owner: Arc<()>,
    dtype: DType,
    device: candle_gen::candle_core::DeviceLocation,
    orig_size: (usize, usize, usize),
    img_seq_len: usize,
    x_cos: Tensor,
    x_sin: Tensor,
    x_attn_mask: Tensor,
    cap: Tensor,
    unified_cos: Tensor,
    unified_sin: Tensor,
    unified_attn_mask: Tensor,
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
    pub fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Self::build(cfg, vb, Execution::Differentiable, None)
    }

    pub(crate) fn new_inference(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Self::build(cfg, vb, Execution::Inference, None)
    }

    /// Keep all main-stack blocks on the host until their admitted window runs.
    pub(crate) fn new_block_streamed(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::new_streamed(cfg, vb, None)
    }

    pub(crate) fn new_block_streamed_with_sidecars(
        cfg: &Config,
        vb: VarBuilder<'static>,
        sidecars: Arc<PackedWeightSidecars>,
    ) -> Result<Self> {
        Self::new_streamed(cfg, vb, Some(sidecars))
    }

    fn new_streamed(
        cfg: &Config,
        vb: VarBuilder<'static>,
        sidecars: Option<Arc<PackedWeightSidecars>>,
    ) -> Result<Self> {
        let layers = TransformerLayers::Streamed {
            weights: vb.clone(),
            sidecars,
            adapters: None,
        };
        Self::build(cfg, vb, Execution::Inference, Some(layers))
    }

    fn build(
        cfg: &Config,
        vb: VarBuilder<'_>,
        execution: Execution,
        streamed: Option<TransformerLayers>,
    ) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"), execution)?;

        let cap_embedder_norm = candle_gen::candle_nn::rms_norm(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear = frozen_projection(
            cfg.cap_feat_dim,
            cfg.dim,
            &vb.pp("cap_embedder"),
            "1",
            execution,
        )?;

        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = frozen_projection(
            patch_dim,
            cfg.dim,
            &vb.pp("all_x_embedder"),
            "2-1",
            execution,
        )?;

        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer = FinalLayer::new(
            cfg.dim,
            out_channels,
            vb.pp("all_final_layer").pp("2-1"),
            execution,
        )?;

        // The pad tokens are required checkpoint keys but are not used by the forward.
        vb.get((1, cfg.dim), "x_pad_token")?;
        vb.get((1, cfg.dim), "cap_pad_token")?;

        let build_stack = |name: &str, count, modulation| {
            (0..count)
                .map(|index| {
                    ZImageTransformerBlock::build(
                        cfg,
                        modulation,
                        vb.pp(name).pp(index),
                        execution,
                        None,
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        let noise_refiner = build_stack("noise_refiner", cfg.n_refiner_layers, true)?;
        let context_refiner = build_stack("context_refiner", cfg.n_refiner_layers, false)?;
        let layers = match streamed {
            Some(layers) => layers,
            None => TransformerLayers::Resident(build_stack("layers", cfg.n_layers, true)?),
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
            execution,
            device: device.clone(),
            preparation_owner: Arc::new(()),
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
            training_captions: Vec::new(),
            training_geometry: std::collections::HashMap::new(),
            #[cfg(test)]
            streamed_window_trace: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            prepare_calls: Default::default(),
        })
    }

    pub(crate) fn prepare_training_projections(&mut self) -> Result<()> {
        if !candle_gen::train::policy::projection_enabled() {
            return Ok(());
        }
        let TransformerLayers::Resident(layers) = &mut self.layers else {
            candle_core::bail!("z-image training requires resident layers")
        };
        for block in self
            .noise_refiner
            .iter_mut()
            .chain(self.context_refiner.iter_mut())
            .chain(layers.iter_mut())
        {
            let attention = &mut block.attention;
            attention.qkv_pack = FrozenProjectionPack::prepare_lora(&mut [
                attention.to_q.trainable_mut(),
                attention.to_k.trainable_mut(),
                attention.to_v.trainable_mut(),
            ])?;
        }
        Ok(())
    }

    /// Cache only the immutable caption embedder, before the adaptable context refiner.
    /// The caller replaces the original caption; no second activation copy is retained here.
    pub(crate) fn prepare_training_caption(
        &mut self,
        x: &Tensor,
        caption: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        let inputs = prepare_inputs(x, std::slice::from_ref(caption), x.device())?;
        let features = inputs.cap_feats.to_dtype(dtype)?;
        let projected = self.embed_caption(&features)?;
        self.training_captions.push((
            projected.id(),
            inputs.cap_mask.clone(),
            inputs.text_lengths[0],
        ));
        Ok(projected)
    }

    pub(crate) fn prepare_training_bucket(&mut self, x: &Tensor, caption: &Tensor) -> Result<()> {
        let inputs = self.training_inputs(x, caption)?;
        let geometry = self.build_geometry(&inputs.latents, &inputs.cap_feats, &inputs.cap_mask)?;
        let (b, _, f, h, w) = inputs.latents.dims5()?;
        self.training_geometry
            .insert((b, f, h, w, caption.id(), inputs.cap_mask.id()), geometry);
        Ok(())
    }

    pub(crate) fn training_inputs(&self, x: &Tensor, caption: &Tensor) -> Result<PreparedInputs> {
        match self
            .training_captions
            .iter()
            .find(|(id, _, _)| *id == caption.id())
        {
            Some((_, mask, length)) => {
                candle_gen::train::profile::cohort_reuse(
                    "zimage:frozen-caption-and-mask-storage-hit",
                );
                Ok(PreparedInputs {
                    latents: x.unsqueeze(2)?,
                    cap_feats: caption.clone(),
                    cap_mask: mask.clone(),
                    text_lengths: vec![*length],
                })
            }
            None => prepare_inputs(x, std::slice::from_ref(caption), x.device()),
        }
    }

    fn positioned_geometry(
        &self,
        x: &Tensor,
        caption: &Tensor,
        mask: &Tensor,
    ) -> Result<PositionedGeometry> {
        let (batch, _, frames, height, width) = x.dims5()?;
        let (patch, frame_patch) = (self.cfg.all_patch_size[0], self.cfg.all_f_patch_size[0]);
        let text_len = caption.dim(1)?;
        let grid = (frames / frame_patch, height / patch, width / patch);
        let image_positions = create_coordinate_grid(grid, (text_len + 1, 0, 0), x.device())?;
        let (image_cos, image_sin) = self.rope_embedder.forward(&image_positions)?;
        let caption_positions = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), x.device())?;
        let (caption_cos, caption_sin) = self.rope_embedder.forward(&caption_positions)?;

        Ok(PositionedGeometry {
            image_positions,
            caption_positions,
            image: StreamGeometry {
                cos: image_cos,
                sin: image_sin,
                mask: Tensor::ones((batch, grid.0 * grid.1 * grid.2), DType::U8, x.device())?,
            },
            caption: StreamGeometry {
                cos: caption_cos,
                sin: caption_sin,
                mask: mask.to_dtype(DType::U8)?,
            },
        })
    }

    fn join_geometry(&self, geometry: &PositionedGeometry) -> Result<StreamGeometry> {
        let positions = Tensor::cat(&[&geometry.image_positions, &geometry.caption_positions], 0)?;
        let (cos, sin) = self.rope_embedder.forward(&positions)?;
        let mask = Tensor::cat(&[&geometry.image.mask, &geometry.caption.mask], 1)?;
        Ok(StreamGeometry { cos, sin, mask })
    }

    fn build_geometry(
        &self,
        x: &Tensor,
        caption: &Tensor,
        mask: &Tensor,
    ) -> Result<ConditioningGeometry> {
        let geometry = self.positioned_geometry(x, caption, mask)?;
        let unified = self.join_geometry(&geometry)?;
        Ok(ConditioningGeometry {
            image: geometry.image,
            caption: geometry.caption,
            unified,
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
    pub(crate) fn refine_image<F>(
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

    pub(crate) fn embed_caption(&self, cap_feats: &Tensor) -> Result<Tensor> {
        let normalized = self.execution.rms(&self.cap_embedder_norm, cap_feats)?;
        self.cap_embedder_linear.forward(&normalized)
    }

    pub(crate) fn refine_caption(
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
        hidden: Tensor,
        attention_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        after_layer: F,
    ) -> candle_gen::Result<Tensor>
    where
        F: FnMut(usize, Tensor) -> candle_gen::Result<Tensor>,
    {
        self.run_layers(
            hidden,
            attention_mask,
            cos,
            sin,
            adaln,
            attention_plan,
            transformer_window,
            |_| None,
            after_layer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        mut hidden: Tensor,
        attention_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        pag_for_layer: impl Fn(usize) -> Option<usize>,
        mut after_layer: impl FnMut(usize, Tensor) -> candle_gen::Result<Tensor>,
    ) -> candle_gen::Result<Tensor> {
        match &self.layers {
            TransformerLayers::Resident(layers) => {
                for (index, layer) in layers.iter().enumerate() {
                    hidden = layer.forward_with_attention_plan_and_pag(
                        &hidden,
                        Some(attention_mask),
                        cos,
                        sin,
                        Some(adaln),
                        attention_plan,
                        pag_for_layer(index),
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
                        #[cfg(test)]
                        candle_gen::lock_recover(&self.streamed_window_trace)
                            .push((range.start, range.end));
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
                                    None => ZImageTransformerBlock::new_inference(
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
                            state = block.forward_with_attention_plan_and_pag(
                                &state,
                                Some(attention_mask),
                                cos,
                                sin,
                                Some(adaln),
                                attention_plan,
                                pag_for_layer(index),
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
        if self.execution == Execution::Differentiable {
            let (unified, ctx) = self.forward_pre_main(x, t, cap_feats, cap_mask)?;
            let unified = into_candle_core(self.run_layers(
                unified,
                &ctx.unified_attn_mask,
                &ctx.unified_cos,
                &ctx.unified_sin,
                &ctx.adaln_input,
                default_attention_plan(),
                self.plain_forward_transformer_window(),
                |_| None,
                |_, hidden| Ok(hidden),
            ))?;
            return self.velocity_out(&unified, &ctx);
        }
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
            self.plain_forward_transformer_window(),
        )
    }

    /// Compatibility forwards do not carry an admitted transformer window. Resident storage can
    /// traverse its already-live trunk in one pass; streamed storage must retain the provider's
    /// bounded default rather than silently materializing every block at once.
    fn plain_forward_transformer_window(&self) -> usize {
        match &self.layers {
            TransformerLayers::Resident(_) => self.cfg.n_layers.max(1),
            TransformerLayers::Streamed { .. } => {
                crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW
            }
        }
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
        let prepared = self.prepare_conditioning(x, cap_feats, cap_mask, attention_plan)?;
        self.forward_prepared_with_memory(x, t, &prepared, attention_plan, transformer_window)
    }

    /// Materialize every geometry/caption-only tensor for one request. This is deliberately an
    /// explicit handle instead of a model cache: changing latent geometry, dtype, or device yields a
    /// new handle and concurrent requests cannot exchange RoPE tables.
    pub(crate) fn prepare_conditioning(
        &self,
        x: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<PreparedConditioning> {
        #[cfg(test)]
        self.prepare_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (b, c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];
        let (_, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let geometry = self.positioned_geometry(x, cap_feats, cap_mask)?;
        let cap = self.refine_caption(
            self.embed_caption(cap_feats)?,
            &geometry.caption.mask,
            &geometry.caption.cos,
            &geometry.caption.sin,
            self.execution.attention_plan(attention_plan),
        )?;
        let unified = self.join_geometry(&geometry)?;
        Ok(PreparedConditioning {
            geometry: (b, c, f, h, w),
            owner: self.preparation_owner.clone(),
            dtype: x.dtype(),
            device: x.device().location(),
            orig_size,
            img_seq_len: geometry.image.mask.dim(1)?,
            x_cos: geometry.image.cos,
            x_sin: geometry.image.sin,
            x_attn_mask: geometry.image.mask,
            cap,
            unified_cos: unified.cos,
            unified_sin: unified.sin,
            unified_attn_mask: unified.mask,
        })
    }

    #[cfg(test)]
    fn forward_prepared(
        &self,
        x: &Tensor,
        t: &Tensor,
        prepared: &PreparedConditioning,
    ) -> Result<Tensor> {
        self.forward_prepared_with_pag(x, t, prepared, None)
    }

    pub(crate) fn forward_prepared_with_pag(
        &self,
        x: &Tensor,
        t: &Tensor,
        prepared: &PreparedConditioning,
        pag: Option<crate::common::PagStepContext<'_>>,
    ) -> Result<Tensor> {
        into_candle_core(self.forward_prepared_with_memory_and_pag(
            x,
            t,
            prepared,
            default_attention_plan(),
            self.plain_forward_transformer_window(),
            pag,
        ))
    }

    #[cfg(test)]
    fn prepare_calls(&self) -> usize {
        self.prepare_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn forward_prepared_with_memory(
        &self,
        x: &Tensor,
        t: &Tensor,
        prepared: &PreparedConditioning,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
    ) -> candle_gen::Result<Tensor> {
        self.forward_prepared_with_memory_and_pag(
            x,
            t,
            prepared,
            attention_plan,
            transformer_window,
            None,
        )
    }

    pub(crate) fn forward_prepared_with_memory_and_pag(
        &self,
        x: &Tensor,
        t: &Tensor,
        prepared: &PreparedConditioning,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        pag: Option<crate::common::PagStepContext<'_>>,
    ) -> candle_gen::Result<Tensor> {
        if x.dims5()? != prepared.geometry
            || x.dtype() != prepared.dtype
            || x.device().location() != prepared.device
        {
            return Err(candle_gen::CandleError::Msg(
                "z-image: prepared conditioning geometry, dtype, or device does not match latent"
                    .into(),
            ));
        }
        if !Arc::ptr_eq(&prepared.owner, &self.preparation_owner) {
            return Err(candle_gen::CandleError::Msg(
                "z-image: prepared conditioning belongs to another DiT or adapter installation"
                    .into(),
            ));
        }
        let attention_plan = self.execution.attention_plan(attention_plan);
        let (x, adaln_input, _) = self.embed_image(x, t)?;
        let x = self.refine_image(
            x,
            &prepared.x_attn_mask,
            &prepared.x_cos,
            &prepared.x_sin,
            &adaln_input,
            attention_plan,
            |_, hidden| Ok(hidden),
        )?;
        let unified = self.run_layers(
            Tensor::cat(&[&x, &prepared.cap], 1)?,
            &prepared.unified_attn_mask,
            &prepared.unified_cos,
            &prepared.unified_sin,
            &adaln_input,
            attention_plan,
            transformer_window,
            |i| pag.and_then(|ctx| ctx.layer_x_len(i, prepared.img_seq_len)),
            |_, hidden| Ok(hidden),
        )?;
        Ok(self.output_velocity(
            &unified,
            &adaln_input,
            prepared.img_seq_len,
            prepared.orig_size,
        )?)
    }

    fn embed_image(
        &self,
        latent: &Tensor,
        t: &Tensor,
    ) -> Result<(Tensor, Tensor, (usize, usize, usize))> {
        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln = __model_proof_output!(self.t_embedder.forward(&t_scaled))?;
        let (patches, orig_size) = patchify(
            latent,
            self.cfg.all_patch_size[0],
            self.cfg.all_f_patch_size[0],
        )?;
        let image = __model_proof_output!(self.x_embedder.forward(&patches))?;
        Ok((image, adaln, orig_size))
    }

    fn output_velocity(
        &self,
        unified: &Tensor,
        adaln: &Tensor,
        image_tokens: usize,
        orig_size: (usize, usize, usize),
    ) -> Result<Tensor> {
        let output = unified.narrow(1, 0, image_tokens)?;
        let output = __model_proof_output!(self.final_layer.forward(&output, adaln))?;
        unpatchify(
            &output,
            orig_size,
            self.cfg.all_patch_size[0],
            self.cfg.all_f_patch_size[0],
            self.cfg.in_channels,
        )
    }

    pub(crate) fn invalidate_prepared_conditioning(&mut self) {
        self.preparation_owner = Arc::new(());
    }

    #[cfg(test)]
    fn streamed_window_trace(&self) -> Vec<(usize, usize)> {
        candle_gen::lock_recover(&self.streamed_window_trace).clone()
    }

    /// The device the DiT weights live on — the forward-time residual factors are read on the CPU and
    /// moved here at install (else the residual matmul is a device mismatch). sc-11105.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Walk every adaptable projection, invoking `f(path, &mut AdaptLinear)` once each with the projection's
    /// canonical DiT dotted path — the same paths [`crate::adapters::classify_lora_key`] resolves a LoRA
    /// key to (the embedders, the per-block attention/feed-forward/adaLN projections across the three
    /// stacks, and the final layer). The additive installer
    /// ([`crate::adapters::install_additive`]) pushes a resolved LoRA/LoKr residual onto each matched
    /// projection so a user adapter applies on a packed q4/q8 tier with the base kept packed (sc-11105).
    pub(crate) fn visit_adaptable_for_install(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
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
                    let mut block = ZImageTransformerBlock::new_inference(
                        &self.cfg,
                        true,
                        weights.pp("layers").pp(i),
                    )?;
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

impl ZImageTransformer2DModel {
    /// The **pre-main** forward (DiT phases 1–10): timestep embed, patchify + embed the image, embed
    /// the caption, run the noise + context refiners, and concatenate `[image, text]` into the unified
    /// hidden state fed to the main layers. Returns `(unified, ctx)` where `ctx` ([`MainContext`]) holds
    /// the constant RoPE/mask/AdaLN tensors the main layers + final layer consume.
    ///
    /// The trainer's gradient-checkpointing path runs this **retained** (its graph kept live), so the
    /// refiner + embedder adapter `Var`s train through ordinary autograd; only the main `layers` stack
    /// is checkpointed (see [`crate::training`]).
    pub fn forward_pre_main(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<(Tensor, MainContext)> {
        let (b, _c, f, h, w) = x.dims5()?;
        let geometry =
            match self
                .training_geometry
                .get(&(b, f, h, w, cap_feats.id(), cap_mask.id()))
            {
                Some(geometry) => {
                    candle_gen::train::profile::cohort_reuse(
                        "zimage:exact-geometry-mask-storage-hit",
                    );
                    geometry.clone()
                }
                None => self.build_geometry(x, cap_feats, cap_mask)?,
            };
        let (x, adaln_input, orig_size) = self.embed_image(x, t)?;
        let img_seq_len = x.dim(1)?;

        // Only the frozen caption embedder can be reused. Every adaptable refiner
        // below is evaluated on each update and remains in the prelude autograd graph.
        let cap = if self
            .training_captions
            .iter()
            .any(|(id, _, _)| *id == cap_feats.id())
        {
            cap_feats.clone()
        } else {
            self.embed_caption(cap_feats)?
        };

        let plan = default_attention_plan();
        let x = into_candle_core(self.refine_image(
            x,
            &geometry.image.mask,
            &geometry.image.cos,
            &geometry.image.sin,
            &adaln_input,
            plan,
            |_, hidden| Ok(hidden),
        ))?;
        let cap = into_candle_core(self.refine_caption(
            cap,
            &geometry.caption.mask,
            &geometry.caption.cos,
            &geometry.caption.sin,
            plan,
        ))?;

        // 9. Concatenate [image, text]
        let unified = Tensor::cat(&[&x, &cap], 1)?;

        Ok((
            unified,
            MainContext {
                adaln_input,
                unified_cos: geometry.unified.cos,
                unified_sin: geometry.unified.sin,
                unified_attn_mask: geometry.unified.mask,
                img_seq_len,
                orig_size,
            },
        ))
    }

    /// One [`Segment`] per **main** transformer layer (DiT phase 11), each mapping `[unified] →
    /// [unified]` using `ctx`'s constant RoPE/mask/AdaLN tensors. This is the Z-Image analog of the
    /// SDXL UNet's `block_segments`: the bulk of the activation working set, fed to
    /// [`checkpointed_backward`](candle_gen::train::gradient_checkpoint::checkpointed_backward) so each
    /// layer is recomputed in the backward instead of retaining all of their activations at once. The
    /// trainer appends a final `[unified] → [loss]` segment (the [`velocity_out`](Self::velocity_out)
    /// head + the flow-match regression).
    pub fn main_layer_segments<'a>(&'a self, ctx: &'a MainContext) -> Vec<Segment<'a>> {
        self.resident_layers()
            .iter()
            .map(|layer| -> Segment<'a> {
                Box::new(move |st: &[Tensor]| {
                    Ok(vec![layer.forward(
                        &st[0],
                        Some(&ctx.unified_attn_mask),
                        &ctx.unified_cos,
                        &ctx.unified_sin,
                        Some(&ctx.adaln_input),
                    )?])
                })
            })
            .collect()
    }

    /// The **post-main** head (DiT phases 12–13): take the image-token portion of the unified hidden
    /// state, apply the final AdaLN layer, and unpatchify back to the raw velocity `(B, C, F, H, W)`.
    /// `unified_final` is the output of the last main layer; `ctx` is the [`MainContext`] from
    /// [`forward_pre_main`](Self::forward_pre_main).
    pub fn velocity_out(&self, unified_final: &Tensor, ctx: &MainContext) -> Result<Tensor> {
        self.output_velocity(
            unified_final,
            &ctx.adaln_input,
            ctx.img_seq_len,
            ctx.orig_size,
        )
    }

    fn resident_layers(&self) -> &[ZImageTransformerBlock] {
        match &self.layers {
            TransformerLayers::Resident(layers) => layers,
            TransformerLayers::Streamed { .. } => unreachable!("training requires resident layers"),
        }
    }
}

impl ZImageTransformer2DModel {
    /// Every block projection a frozen training adapter can reach (sc-25213), across the noise
    /// refiner, context refiner and main stacks — a superset of the trainable [`LoraHost`] walk,
    /// which stays the attention projections only.
    pub(crate) fn visit_block_linears(&self, f: &mut dyn FnMut(&LoraLinear)) {
        for blk in self
            .noise_refiner
            .iter()
            .chain(self.context_refiner.iter())
            .chain(self.resident_layers())
        {
            blk.visit_linears(f);
        }
    }

    /// [`visit_block_linears`](Self::visit_block_linears), mutably.
    pub(crate) fn visit_block_linears_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        let TransformerLayers::Resident(layers) = &mut self.layers else {
            return Err(candle_gen::CandleError::Msg(
                "z-image training requires resident layers".into(),
            ));
        };
        for blk in self
            .noise_refiner
            .iter_mut()
            .chain(self.context_refiner.iter_mut())
            .chain(layers.iter_mut())
        {
            blk.visit_linears_mut(f)?;
        }
        Ok(())
    }
}

impl LoraHost for ZImageTransformer2DModel {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        let TransformerLayers::Resident(layers) = &mut self.layers else {
            return Err(candle_gen::CandleError::Msg(
                "z-image training requires resident layers".into(),
            ));
        };
        for blk in self
            .noise_refiner
            .iter_mut()
            .chain(self.context_refiner.iter_mut())
            .chain(layers.iter_mut())
        {
            blk.visit_lora_mut(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored DiT to the stock candle-transformers DiT: built from the *same*
    //! `VarMap`-backed weights with no adapter installed, the two must produce bit-identical forward
    //! output. The regression guard that the vendoring (the `LoraLinear` swap + composable softmax)
    //! changed nothing numerically.
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::preprocess::prepare_inputs;
    use candle_transformers::models::z_image::transformer::{
        Config, ZImageTransformer2DModel as StockModel,
    };

    /// A tiny Z-Image-shaped config: `head_dim` is locked to 128 by `axes_dims=[32,48,48]` (the RoPE
    /// half-dims `16+24+24=64=head_dim/2`), so a single head at `dim=128` is the smallest valid DiT.
    /// 2 main layers + 1 refiner each exercises every vendored path cheaply on CPU.
    fn tiny_cfg() -> Config {
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 128; // head_dim = 128/1 = 128 (axes_dims sum)
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.n_layers = 2;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 64;
        cfg.use_accelerated_attn = false; // force the math/basic path on the stock side
        cfg
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The vendored model is built first, populating the VarMap with random weights; the stock
        // model then reads the SAME parameters (identical names/shapes), so any output difference is
        // a forward-logic difference, not a weight difference.
        let vendored = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let stock = StockModel::new(&cfg, vb).unwrap();

        // latent (1, 16, 4, 4) -> patchified 2x2 = 4 image tokens; tiny caption of 3 tokens.
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let y_v = __model_proof_output!(vendored.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
        .unwrap();
        let y_s = __model_proof_output!(stock.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
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
        assert!(diff < 1e-5, "vendored DiT diverged from stock by {diff}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::parity_tests::vendored_dit_matches_stock_forward");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn prepared_dense_conditioning_is_reused_by_actual_forwards_and_rejects_stale_keys() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let model =
            ZImageTransformer2DModel::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))
                .unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let inputs = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let mut prepared = model
            .prepare_conditioning(
                &inputs.latents,
                &inputs.cap_feats,
                &inputs.cap_mask,
                default_attention_plan(),
            )
            .unwrap();
        assert_eq!(model.prepare_calls(), 1);
        let first = model
            .forward_prepared(&inputs.latents, &t, &prepared)
            .unwrap();
        let second = model
            .forward_prepared(&inputs.latents, &t, &prepared)
            .unwrap();
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            model.prepare_calls(),
            1,
            "actual forwards must not reconstruct request conditioning"
        );
        let stale = Tensor::zeros((1, cfg.in_channels, 1, 6, 4), DType::F32, &dev).unwrap();
        assert!(model.forward_prepared(&stale, &t, &prepared).is_err());
        let wrong_dtype = inputs.latents.to_dtype(DType::F64).unwrap();
        assert!(model.forward_prepared(&wrong_dtype, &t, &prepared).is_err());
        prepared.device = candle_core::DeviceLocation::Metal { gpu_id: 5 };
        assert!(model
            .forward_prepared(&inputs.latents, &t, &prepared)
            .is_err());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::parity_tests::prepared_dense_conditioning_is_reused_by_actual_forwards_and_rejects_stale_keys");
    }

    /// The [`LoraHost`] walk reaches exactly `4 × (n_refiner·2 + n_layers)` projections — the four
    /// attention `LoraLinear`s in every noise-refiner, context-refiner, and main block.
    #[test]
    fn lora_host_visits_every_attention_projection() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        let mut paths: Vec<String> = Vec::new();
        model
            .visit_lora_mut(&mut |lin| {
                paths.push(lin.path().to_string());
                Ok(())
            })
            .unwrap();
        let expected = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(paths.len(), expected);
        // Every default target suffix resolves against at least one visited path.
        for suffix in Z_IMAGE_ATTN_TARGETS {
            assert!(
                paths
                    .iter()
                    .any(|p| p == suffix || p.ends_with(&format!(".{suffix}"))),
                "no visited projection matched suffix {suffix}"
            );
        }
        // Layer-0 main-block attention paths are present and correctly named.
        assert!(paths.contains(&"layers.0.attention.to_q".to_string()));
        assert!(paths.contains(&"layers.0.attention.to_out.0".to_string()));
    }
}

#[cfg(test)]
mod inference_tests {
    //! Pin the vendored DENSE path to the stock candle-transformers DiT: built from the *same*
    //! `VarMap`-backed weights (no `.scales`, so every `QLinear` takes the dense arm), the two must
    //! produce bit-identical forward output — the guard that the packed-seam vendoring changed nothing
    //! numerically on a dense tier. It uses the same checkpoint names as the pinned reference model.
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

    /// Owns its component directory outright. The hand-written `Drop` this replaces skipped a
    /// panicking test and drew its name from a recyclable PID; the guard does neither.
    struct SidecarFixture(tempfile::TempDir);

    impl SidecarFixture {
        fn new(bits: usize) -> Self {
            Self(
                tempfile::Builder::new()
                    .prefix(&format!("z-image-sc16510-sidecar-{bits}-"))
                    .tempdir()
                    .unwrap(),
            )
        }

        fn path(&self) -> &std::path::Path {
            self.0.path()
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
        let source_path = fixture.path().join("model.safetensors");
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
            fixture.path(),
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
        let old = AdaptLinear::linear_detect(64, 4, &vb, "to_q", true)?;
        let missing =
            streamed_linear_detect(64, 4, &vb, "to_q", true, &sidecars, "layers.1.attention")
                .expect_err("a packed streamed projection must not fall back to source conversion");
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
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_q4_projection_uses_byte_exact_device_format_sidecar() -> Result<()> {
        __model_proof_inspect!(
            assert_streamed_projection_uses_byte_exact_sidecar(4),
            |_| {
                println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::streamed_q4_projection_uses_byte_exact_device_format_sidecar");
            }
        )
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_q8_projection_uses_byte_exact_device_format_sidecar() -> Result<()> {
        __model_proof_inspect!(
            assert_streamed_projection_uses_byte_exact_sidecar(8),
            |_| {
                println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::streamed_q8_projection_uses_byte_exact_device_format_sidecar");
            }
        )
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dense_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The vendored model is built first, populating the VarMap with random weights; the stock model
        // then reads the SAME parameters (identical names/shapes), so any output difference is a
        // forward-logic difference. No `.scales` present, so every QLinear is `Dense`.
        let vendored = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
        let stock = StockModel::new(&cfg, vb).unwrap();

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let y_v = __model_proof_output!(vendored.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
        .unwrap();
        let y_s = __model_proof_output!(stock.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::vendored_dense_dit_matches_stock_forward");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn prepared_conditioning_reuses_actual_forward_tables_and_rejects_stale_geometry() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let model = ZImageTransformer2DModel::new_inference(
            &cfg,
            VarBuilder::from_varmap(&vm, DType::F32, &dev),
        )
        .unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (1, 3usize, cfg.cap_feat_dim), &dev).unwrap();
        let mask = Tensor::ones((1, 3), DType::U8, &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(1 << 20, false));
        let mut prepared = model
            .prepare_conditioning(&latent, &cap, &mask, plan)
            .unwrap();
        assert_eq!(model.prepare_calls(), 1);
        let a = model
            .forward_prepared_with_memory(&latent, &t, &prepared, plan, cfg.n_layers)
            .unwrap();
        let b = model
            .forward_prepared_with_memory(&latent, &t, &prepared, plan, cfg.n_layers)
            .unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff <= f32::EPSILON, "prepared forwards diverged by {diff}");
        assert_eq!(
            model.prepare_calls(),
            1,
            "actual packed forwards must not reconstruct request conditioning"
        );
        let stale = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 6, 4), &dev).unwrap();
        assert!(model
            .forward_prepared_with_memory(&stale, &t, &prepared, plan, cfg.n_layers)
            .is_err());
        let wrong_dtype = latent.to_dtype(DType::F64).unwrap();
        assert!(model
            .forward_prepared_with_memory(&wrong_dtype, &t, &prepared, plan, cfg.n_layers)
            .is_err());
        let another_model = ZImageTransformer2DModel::new_inference(
            &cfg,
            VarBuilder::from_varmap(&vm, DType::F32, &dev),
        )
        .unwrap();
        let err = another_model
            .forward_prepared_with_memory(&latent, &t, &prepared, plan, cfg.n_layers)
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("prepared conditioning belongs to another DiT"));
        prepared.device = candle_gen::candle_core::DeviceLocation::Cuda { gpu_id: 9 };
        assert!(model
            .forward_prepared_with_memory(&latent, &t, &prepared, plan, cfg.n_layers)
            .is_err());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::prepared_conditioning_reuses_actual_forward_tables_and_rejects_stale_geometry");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn attention_query_chunking_matches_the_unbounded_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = ZImageTransformer2DModel::new_inference(&cfg, vb).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let full = __model_proof_output!(model.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::attention_query_chunking_matches_the_unbounded_forward");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_windows_match_resident_forward_including_ragged_tail() {
        let dev = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.n_layers = 3;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Populate one immutable host-backed source, then build the streamed front/refiner shell from
        // that same source. Two forces a 2+1 ragged tail; the larger published candidates exercise
        // the all-covering window without changing the contract's production range.
        let resident = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let expected = __model_proof_output!(resident.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::streamed_windows_match_resident_forward_including_ragged_tail");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_window_rejects_zero_and_preserves_typed_cancellation() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Populate all layer tensors once; the streamed model retains only the source view.
        let _resident = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::streamed_window_rejects_zero_and_preserves_typed_cancellation");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn plain_forward_drives_streamed_storage_through_bounded_windows() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        assert!(
            cfg.n_layers > crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW,
            "the fixture must distinguish the bounded default from full-trunk residency"
        );
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let resident = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
        let streamed = ZImageTransformer2DModel::new_block_streamed(&cfg, vb).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let plan = plan_from_budget(candle_gen::ATTN_SCORES_BUDGET);

        let resident_output = resident
            .forward_with_attention_plan(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                plan,
            )
            .unwrap();
        let streamed_output = streamed
            .forward_with_attention_plan(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
                plan,
            )
            .unwrap();
        let diff = (&resident_output - &streamed_output)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "storage choice changed the output by {diff}");
        assert!(
            resident.streamed_window_trace().is_empty(),
            "resident storage must preserve its direct traversal"
        );
        assert_eq!(
            streamed.streamed_window_trace(),
            vec![(0, 1), (1, 2)],
            "the actual plain-forward wiring must schedule the streamed trunk one bounded window \
             at a time"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::plain_forward_drives_streamed_storage_through_bounded_windows");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn constrained_request_cancel_stops_inside_the_packed_dit_as_typed_canceled() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = ZImageTransformer2DModel::new_inference(&cfg, vb).unwrap();
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::constrained_request_cancel_stops_inside_the_packed_dit_as_typed_canceled");
    }

    /// **Additive install on the vendored DiT (sc-11105).** A bare-dotted LoRA over two real `layers.0`
    /// attention projections installs as forward-time residuals: the report counts both, the DiT forward
    /// shifts vs the un-adapted model, and no target is left unresolved — proving the visitor's canonical
    /// paths line up with `adapters::classify_lora_key` (a path mismatch would apply nothing and error).
    /// This exercises the packed-tier install wiring end-to-end on a dense-base fixture; the base-agnostic
    /// additive path is byte-equal on a packed base (the stays-packed / exact-residual property is proven
    /// at the `candle_gen::quant::AdaptLinear` unit level).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn install_additive_lora_on_vendored_dit_applies_and_shifts() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        use std::collections::HashMap as Map;
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Two DiTs from the SAME params: one baseline, one adapted.
        let base = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
        let mut adapted = ZImageTransformer2DModel::new_inference(&cfg, vb).unwrap();

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
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().join("sc11105_install.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        let report = crate::adapters::install_additive(
            &mut adapted,
            &[AdapterSpec::new(tmp.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(report.applied, 2, "both to_q + to_v residuals installed");
        assert!(
            report.skipped_targets.is_empty(),
            "no unresolved targets (visitor paths match classify)"
        );

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let y_base = __model_proof_output!(base.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
        .unwrap();
        let y_adapt = __model_proof_output!(adapted.forward(
            &prepared.latents,
            &t,
            &prepared.cap_feats,
            &prepared.cap_mask,
        ))
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::inference_tests::install_additive_lora_on_vendored_dit_applies_and_shifts");
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
        let mut dit = ZImageTransformer2DModel::new_inference(&cfg, vb.clone()).unwrap();
        let mut off: Map<String, Tensor> = Map::new();
        off.insert(
            "layers.99.attention.to_q.lora_A.weight".into(),
            Tensor::zeros((2, cfg.dim), DType::F32, &dev).unwrap(),
        );
        off.insert(
            "layers.99.attention.to_q.lora_B.weight".into(),
            Tensor::zeros((cfg.dim, 2), DType::F32, &dev).unwrap(),
        );
        let tmp1_tmp = tempfile::tempdir().unwrap();
        let tmp1 = tmp1_tmp.path().join("sc11105_off.safetensors");
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
        let mut dit2 = ZImageTransformer2DModel::new_inference(&cfg, vb).unwrap();
        let mut loha: Map<String, Tensor> = Map::new();
        let path = "layers.0.attention.to_q";
        for k in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"] {
            loha.insert(
                format!("{path}.{k}"),
                Tensor::zeros((cfg.dim, 1), DType::F32, &dev).unwrap(),
            );
        }
        let tmp2_tmp = tempfile::tempdir().unwrap();
        let tmp2 = tmp2_tmp.path().join("sc11105_loha.safetensors");
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
