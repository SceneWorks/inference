//! The Qwen-Image 2.1 single-stream DiT — a faithful port of diffusers'
//! `QwenImage21Transformer2DModel` (`transformer_qwenimage21.py` @
//! [`crate::UPSTREAM_DIFFUSERS_REVISION`]), prefill path (`QwenImage21AttnProcessor`, no KV cache).
//! The candle twin of `mlx-gen-qwen-image-2-1`'s `src/transformer.rs`.
//!
//! Text and image latents share **one joint sequence** ([`JointLayout`]): the projected text tokens
//! in order, condition-image blocks substituted at the vision slots the encoder reserved for them
//! (a later story), and the target image's tokens appended. Two behaviours distinguish 2.1:
//!
//! * **Block-causal attention** — `(q ≥ kv) or same_image_block`: the joint sequence is causal,
//!   every image block is internally bidirectional, and the target block (last) therefore sees
//!   everything. Realised exactly as upstream's SDPA processor does: one attention call per prefix
//!   segment (text segments with a causal mask over keys `[0, end)`, image segments unmasked) plus
//!   one unmasked call for the target over all keys.
//! * **`causal_condition`** — the shared modulation is computed for two timestep rows, the sampled
//!   `t` and `t = 0`; text/condition tokens read the `t = 0` row, target tokens their own.
//!
//! Numerics: the model computes in the `VarBuilder`'s dtype (f32 on the CPU parity lane, bf16 on a
//! GPU backend — [`crate::loader::compute_dtype`]); the zero-centred text norm, the affine-free
//! LayerNorms, the q/k RMSNorm, the RoPE rotation and the timestep sinusoids are computed in f32
//! and rounded back, as upstream does. The 3-axis RoPE (`axes_dims_rope = [16, 56, 56]`,
//! `θ = 10000`) rotates **adjacent** channel pairs (`view_as_complex`, `use_real=False`) — candle's
//! [`rope_i`] — unlike the text tower's half-split RoPE, and its table is built host-side in f32
//! because the centred image grid's positions are negative.
//!
//! Weight keys are the diffusers ones and every Linear is **bias-less** in this checkpoint:
//! `img_in`, `txt_in.{text_norm.weight, in_layer, out_layer}`,
//! `time_text_embed.timestep_embedder.linear_{1,2}`, `modulation.1`,
//! `transformer_blocks.{i}.attn.{to_q, to_k, to_v, to_out.0, norm_q.weight, norm_k.weight}`,
//! `transformer_blocks.{i}.img_mlp.{gate_layer, proj, out}`, `norm_out.linear`, `proj_out`. Each
//! Linear loads through [`AdaptLinear::linear_detect_gs`], so an already-MLX-packed snapshot binds
//! its `.scales`/`.biases` siblings without materialising a dense weight; with no packed sidecar the
//! dense path is a plain `candle_nn::Linear`.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_gen::candle_nn::ops::{rms_norm, softmax_last_dim};
use candle_gen::candle_nn::rotary_emb::rope_i;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::AdaptLinear;
use candle_gen::{CandleError as Error, Result};

use crate::config::TransformerConfig;

/// Rotary base of the joint-sequence RoPE (`QwenImage21Rope(theta=10000)`).
const ROPE_THETA: f32 = 10_000.0;
/// Sinusoidal timestep projection width (`QwenImage21TemporalTimesteps(timestep_dim=256)`).
const TIMESTEP_DIM: usize = 256;
/// `max_period` of the timestep sinusoid (`get_timestep_embedding`).
const TIMESTEP_MAX_PERIOD: f64 = 10_000.0;
/// `time_factor`: the transformer receives `sigma` and scales it by 1000 inside.
const TIME_FACTOR: f32 = 1000.0;
/// Group size of a pre-quantized (MLX-packed) DiT — the codebase-wide default (64).
const GROUP_SIZE: usize = 64;

/// The MLX group size a packed Linear of this input width would have been written at — mirrors the
/// MLX twin's `quantize`: 64 where the width allows, 32 otherwise (the released DiT's widths — 64 /
/// 4096 / 12288 — all take 64; only the miniature parity snapshot has a narrower one, and it is
/// never packed).
fn group_for(in_dim: usize) -> usize {
    if in_dim.is_multiple_of(GROUP_SIZE) {
        GROUP_SIZE
    } else {
        32
    }
}

/// A bias-less, packed-detecting `[out, in]` projection at `base` (relative to `vb`). `base` is the
/// full dotted key so a `to_out.0`-style nesting keeps its `.scales`/`.biases` siblings.
fn lin(in_dim: usize, out_dim: usize, vb: &VarBuilder, base: &str) -> Result<AdaptLinear> {
    Ok(AdaptLinear::linear_detect_gs(
        in_dim,
        out_dim,
        vb,
        base,
        false,
        group_for(in_dim),
    )?)
}

/// One run of the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `len` prompt tokens (strictly causal among themselves).
    Text { len: usize },
    /// One image block of `height × width` latent tokens (row-major), internally bidirectional.
    /// The **last** image segment of a layout is the target being denoised; any earlier one is a
    /// condition image (edit path).
    Image { height: usize, width: usize },
}

/// The joint text/image token layout the transformer attends over. Built by the pipeline; for
/// text-to-image it is `[Text { len }, Image { target }]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointLayout {
    pub segments: Vec<Segment>,
}

impl JointLayout {
    /// The text-to-image layout: the prompt then the target image.
    pub fn text_to_image(text_len: usize, height: usize, width: usize) -> Self {
        Self {
            segments: vec![
                Segment::Text { len: text_len },
                Segment::Image { height, width },
            ],
        }
    }

    fn validate(&self) -> Result<()> {
        match self.segments.last() {
            Some(Segment::Image { height, width }) if *height > 0 && *width > 0 => Ok(()),
            _ => Err(Error::Msg(
                "qwen_image_2_1: the joint layout must end with the non-empty target image block"
                    .into(),
            )),
        }
    }

    fn segment_len(segment: &Segment) -> usize {
        match segment {
            Segment::Text { len } => *len,
            Segment::Image { height, width } => height * width,
        }
    }

    pub fn total_len(&self) -> usize {
        self.segments.iter().map(Self::segment_len).sum()
    }

    /// Tokens of the target image block.
    pub fn target_tokens(&self) -> usize {
        self.segments.last().map_or(0, Self::segment_len)
    }

    /// Everything before the target block.
    pub fn prefix_len(&self) -> usize {
        self.total_len() - self.target_tokens()
    }

    /// `(start, end, is_text)` runs over the prefix — `_qwenimage21_prefix_segments`.
    pub fn prefix_segments(&self) -> Vec<(usize, usize, bool)> {
        let mut out = Vec::new();
        let mut cursor = 0;
        for segment in &self.segments[..self.segments.len().saturating_sub(1)] {
            let len = Self::segment_len(segment);
            if len > 0 {
                out.push((
                    cursor,
                    cursor + len,
                    matches!(segment, Segment::Text { .. }),
                ));
            }
            cursor += len;
        }
        out
    }

    /// `(frame, height, width)` RoPE position per token — `QwenImage21Rope.forward`: text tokens
    /// advance one shared position on all three axes; each image block freezes the frame axis at
    /// the position reached so far and lays its tokens on a zero-centred grid, then advances the
    /// shared position by `max(height, width)`.
    pub fn position_ids(&self) -> Vec<[i32; 3]> {
        let mut ids = Vec::with_capacity(self.total_len());
        let mut position: i32 = 0;
        for segment in &self.segments {
            match *segment {
                Segment::Text { len } => {
                    for _ in 0..len {
                        ids.push([position, position, position]);
                        position += 1;
                    }
                }
                Segment::Image { height, width } => {
                    let (h, w) = (height as i32, width as i32);
                    for y in -(h - h / 2)..(h / 2) {
                        for x in -(w - w / 2)..(w / 2) {
                            ids.push([position, y, x]);
                        }
                    }
                    position += h.max(w);
                }
            }
        }
        ids
    }

    /// `true` at the target image's tokens.
    pub fn target_mask(&self) -> Vec<bool> {
        let total = self.total_len();
        let target = self.target_tokens();
        (0..total).map(|i| i >= total - target).collect()
    }
}

/// Zero-centred RMSNorm (`QwenImage21ZeroCenterRMSNorm`): effective scale `weight + 1`, f32.
fn zero_center_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let ms = x32.sqr()?.mean_keepdim(D::Minus1)?;
    let rrms = (ms + eps)?.sqrt()?.recip()?;
    let scale = (weight.to_dtype(DType::F32)? + 1.0)?;
    Ok(x32
        .broadcast_mul(&rrms)?
        .broadcast_mul(&scale)?
        .to_dtype(dtype)?)
}

/// Affine-free LayerNorm over the last axis, computed in f32 (`mlx_rs::fast::layer_norm(x, None,
/// None, eps)`).
fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.mean_keepdim(D::Minus1)?;
    let centred = x32.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(centred
        .broadcast_div(&(var + eps)?.sqrt()?)?
        .to_dtype(dtype)?)
}

/// `x · (1 + scale)` — upstream's `1 + scale` in the modulation's own dtype.
fn scale_residual(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_mul(&(scale + 1.0)?.to_dtype(x.dtype())?)?)
}

/// Adjacent-pair complex rotation of `x` `[B, H, S, D]` by `cos`/`sin` `[S, D/2]`, in f32. (The MLX
/// twin rotates `[B, S, H, D]`; the rotation is per token and per head, so the transpose commutes.)
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?.contiguous()?;
    Ok(rope_i(&x32, cos, sin)?.to_dtype(dtype)?)
}

/// Per-axis RoPE sinusoid table from the integer position ids `[N, 3]` (mlx-gen's
/// `rope_sincos_from_ids`): for each axis `i` with `dim = axes_dim[i]`, `omega[k] =
/// theta^-(2k/dim)` for `k < dim/2`, angles `ids[:, i]·omega`, then `(cos, sin)` **concatenated
/// across the axes** on the last dim → `([N, Σ dim/2], [N, Σ dim/2])`, f32. Computed host-side: the
/// centred image grid's positions are negative, and the MLX twin's table is the same arithmetic.
fn rope_sincos_from_ids(
    ids: &[[i32; 3]],
    axes_dim: &[usize; 3],
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half: usize = axes_dim.iter().map(|d| d / 2).sum();
    let n = ids.len();
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    // `omega` is position-independent: build the concatenated `[Σ dim/2]` frequency vector once.
    let mut omega = Vec::with_capacity(half);
    let mut axis_of = Vec::with_capacity(half);
    for (axis, &dim) in axes_dim.iter().enumerate() {
        for k in 0..dim / 2 {
            omega.push(1.0f32 / theta.powf((2 * k) as f32 / dim as f32));
            axis_of.push(axis);
        }
    }
    for (row, id) in ids.iter().enumerate() {
        for (j, (&om, &axis)) in omega.iter().zip(&axis_of).enumerate() {
            let a = id[axis] as f32 * om;
            cos[row * half + j] = a.cos();
            sin[row * half + j] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, half), device)?,
        Tensor::from_vec(sin, (n, half), device)?,
    ))
}

/// diffusers `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True,
/// downscale_freq_shift=0, max_period)` in f32 → `[N, dim]` ordered `[cos | sin]` (the flip).
fn timestep_sincos(
    timesteps: &[f32],
    dim: usize,
    max_period: f64,
    device: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let neg_log = -(max_period.ln()) as f32;
    let denom = half as f32;
    let n = timesteps.len();
    let mut out = vec![0f32; n * dim];
    for (row, &t) in timesteps.iter().enumerate() {
        for i in 0..half {
            let a = t * (i as f32 * neg_log / denom).exp();
            out[row * dim + i] = a.cos();
            out[row * dim + half + i] = a.sin();
        }
    }
    Ok(Tensor::from_vec(out, (n, dim), device)?)
}

/// SiLU / swish: `x · sigmoid(x)`.
fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_gen::candle_nn::ops::silu(x)?)
}

/// GELU with the tanh approximation — candle's `Tensor::gelu` (`gelu_erf` is the exact one).
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// Block-causal SDPA over `q, k, v: [B, H, S, D]`, returning `[B, H, S, D]`. One call per prefix
/// segment — a text run gets an additive `-inf` mask over keys `[0, end)` whose diagonal is aligned
/// to the **last** key (query row `r` is absolute position `start + r` and may see keys
/// `[0, start + r]`), an image block is unmasked over the same keys — plus one unmasked call for
/// the target rows `[prefix_len, S)` over every key. The per-segment outputs concatenate back into
/// the full sequence.
///
/// Every call goes through [`candle_gen::ATTN_SCORES_BUDGET`], the F-003 i32-overflow guard, and
/// **not** the un-chunked `usize::MAX` sentinel: this family's joint sequence is long enough for
/// the guard to be load-bearing at the shipped sizes, not a formality. At the 1:1 2048² default the
/// target block alone is `2·(2048/32) = 128` latent tokens per side → 16384 image tokens, so the
/// unmasked target call's scores are `1 · 32 heads · 16384 · ~16.6k ≈ 8.7e9` elements — about 4×
/// `i32::MAX`, which candle's CUDA kernels index with. Unguarded that silently corrupts the tail of
/// the scores tensor: a green run, a wrong image. The guard chunks the query rows, which is
/// mathematically equivalent (each row's softmax is independent) though not bitwise equal to a
/// single pass; below the budget it returns the whole query axis, so the parity fixtures here —
/// whose scores are a few hundred elements — take the identical single-pass path.
fn block_causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    prefix_segments: &[(usize, usize, bool)],
    prefix_len: usize,
) -> Result<Tensor> {
    let (_b, _h, s, _d) = q.dims4()?;
    let device = q.device();
    let dtype = q.dtype();
    let mut outputs: Vec<Tensor> = Vec::with_capacity(prefix_segments.len() + 1);
    for &(start, end, is_text) in prefix_segments {
        let qs = q.narrow(2, start, end - start)?.contiguous()?;
        let ks = k.narrow(2, 0, end)?.contiguous()?;
        let vs = v.narrow(2, 0, end)?.contiguous()?;
        let mask = if is_text {
            let rows = end - start;
            let mut m = vec![0f32; rows * end];
            for (r, row) in m.chunks_mut(end).enumerate() {
                for entry in row.iter_mut().skip(start + r + 1) {
                    *entry = f32::NEG_INFINITY;
                }
            }
            Some(Tensor::from_vec(m, (1, 1, rows, end), device)?.to_dtype(dtype)?)
        } else {
            None
        };
        outputs.push(candle_gen::sdpa_budgeted_bhsd(
            &qs,
            &ks,
            &vs,
            scale,
            mask.as_ref(),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?);
    }
    let qt = q.narrow(2, prefix_len, s - prefix_len)?.contiguous()?;
    outputs.push(candle_gen::sdpa_budgeted_bhsd(
        &qt,
        &k.contiguous()?,
        &v.contiguous()?,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?);
    Ok(Tensor::cat(&outputs, 2)?)
}

struct Attention {
    to_q: AdaptLinear,
    to_k: AdaptLinear,
    to_v: AdaptLinear,
    to_out: AdaptLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl Attention {
    fn new(cfg: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            to_q: lin(inner, inner, vb, "to_q")?,
            to_k: lin(inner, inner, vb, "to_k")?,
            to_v: lin(inner, inner, vb, "to_v")?,
            to_out: lin(inner, inner, vb, "to_out.0")?,
            norm_q: vb.get(cfg.attention_head_dim, "norm_q.weight")?,
            norm_k: vb.get(cfg.attention_head_dim, "norm_k.weight")?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps,
        })
    }

    /// Block-causal attention over the joint sequence `x` `[B, S, inner]`.
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let heads = |y: Tensor| -> Result<Tensor> {
            Ok(y.reshape((b, s, h, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = heads(self.to_q.forward(x)?)?;
        let k = heads(self.to_k.forward(x)?)?;
        let v = heads(self.to_v.forward(x)?)?;
        let norm = |y: &Tensor, w: &Tensor| -> Result<Tensor> {
            Ok(
                rms_norm(&y.to_dtype(DType::F32)?, &w.to_dtype(DType::F32)?, self.eps)?
                    .to_dtype(v.dtype())?,
            )
        };
        let q = apply_rope(&norm(&q, &self.norm_q)?, cos, sin)?;
        let k = apply_rope(&norm(&k, &self.norm_k)?, cos, sin)?;
        let scale = (hd as f64).powf(-0.5);
        let o = block_causal_attention(&q, &k, &v, scale, prefix_segments, prefix_len)?;
        let o = o.transpose(1, 2)?.reshape((b, s, h * hd))?;
        Ok(self.to_out.forward(&o)?)
    }
}

struct FeedForward {
    gate_layer: AdaptLinear,
    proj: AdaptLinear,
    out: AdaptLinear,
}

impl FeedForward {
    fn new(cfg: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hidden = inner * cfg.mlp_ratio;
        Ok(Self {
            gate_layer: lin(inner, hidden, vb, "gate_layer")?,
            proj: lin(inner, hidden, vb, "proj")?,
            out: lin(hidden, inner, vb, "out")?,
        })
    }

    /// `out(silu(gate(x)) · proj(x))`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (silu(&self.gate_layer.forward(x)?)? * self.proj.forward(x)?)?;
        Ok(self.out.forward(&gated)?)
    }
}

/// Optional per-stage capture for the parity tests: every named intermediate, cast to f32, in
/// forward order. `None` costs nothing on the production path.
struct Trace<'a>(Option<&'a mut Vec<(String, Tensor)>>);

impl Trace<'_> {
    fn push(&mut self, name: impl Into<String>, value: &Tensor) -> Result<()> {
        if let Some(sink) = self.0.as_deref_mut() {
            sink.push((name.into(), value.to_dtype(DType::F32)?));
        }
        Ok(())
    }
}

/// Per-token modulation rows for one forward: `[1, S, inner]` each, already selected between the
/// sampled-timestep row (target tokens) and the `t = 0` row (prefix tokens).
struct Modulation {
    scale1: Tensor,
    gate1: Tensor,
    scale2: Tensor,
    gate2: Tensor,
}

struct Block {
    attn: Attention,
    mlp: FeedForward,
    eps: f64,
}

impl Block {
    fn new(cfg: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, &vb.pp("attn"))?,
            mlp: FeedForward::new(cfg, &vb.pp("img_mlp"))?,
            eps: cfg.eps as f64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        index: usize,
        x: &Tensor,
        m: &Modulation,
        cos: &Tensor,
        sin: &Tensor,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
        trace: &mut Trace<'_>,
    ) -> Result<Tensor> {
        let h = scale_residual(&layer_norm(x, self.eps)?, &m.scale1)?;
        let a = self
            .attn
            .forward(&h, cos, sin, prefix_segments, prefix_len)?;
        trace.push(format!("block_{index}_attn"), &a)?;
        let x = (x + m.gate1.tanh()?.broadcast_mul(&a)?)?;
        let h = scale_residual(&layer_norm(&x, self.eps)?, &m.scale2)?;
        let f = self.mlp.forward(&h)?;
        trace.push(format!("block_{index}_mlp"), &f)?;
        let out = (&x + m.gate2.tanh()?.broadcast_mul(&f)?)?;
        trace.push(format!("block_{index}_out"), &out)?;
        Ok(out)
    }
}

/// The single-stream Qwen-Image 2.1 transformer.
pub struct QwenImage21Transformer {
    cfg: TransformerConfig,
    device: Device,
    dtype: DType,
    img_in: AdaptLinear,
    txt_norm: Tensor,
    txt_in: AdaptLinear,
    txt_out: AdaptLinear,
    time_in: AdaptLinear,
    time_out: AdaptLinear,
    modulation: AdaptLinear,
    blocks: Vec<Block>,
    norm_out: AdaptLinear,
    proj_out: AdaptLinear,
}

impl QwenImage21Transformer {
    /// Build from diffusers-keyed weights (`img_in.weight`, `transformer_blocks.{i}.…`, …) on
    /// `vb`'s device, computing in `vb`'s dtype.
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, &vb.pp(format!("transformer_blocks.{i}")))?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
            img_in: lin(cfg.in_channels, inner, &vb, "img_in")?,
            txt_norm: vb.get(cfg.context_in_dim, "txt_in.text_norm.weight")?,
            txt_in: lin(cfg.context_in_dim, inner, &vb, "txt_in.in_layer")?,
            txt_out: lin(inner, inner, &vb, "txt_in.out_layer")?,
            time_in: lin(
                TIMESTEP_DIM,
                inner,
                &vb,
                "time_text_embed.timestep_embedder.linear_1",
            )?,
            time_out: lin(
                inner,
                inner,
                &vb,
                "time_text_embed.timestep_embedder.linear_2",
            )?,
            modulation: lin(inner, 4 * inner, &vb, "modulation.1")?,
            blocks,
            norm_out: lin(inner, inner, &vb, "norm_out.linear")?,
            proj_out: lin(inner, cfg.out_channels, &vb, "proj_out")?,
        })
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.cfg
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The dtype the model computes in — the `VarBuilder`'s (f32 on CPU, bf16 on a GPU backend).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// The joint RoPE table `(cos, sin)`, each `[S, head_dim/2]` f32, for `layout`.
    pub fn rope(&self, layout: &JointLayout) -> Result<(Tensor, Tensor)> {
        rope_sincos_from_ids(
            &layout.position_ids(),
            &self.cfg.axes_dims_rope,
            ROPE_THETA,
            &self.device,
        )
    }

    /// `[1, target_tokens, in_channels]` latents + `[1, text_len, context_in_dim]` text → the
    /// target's velocity `[1, target_tokens, out_channels]` (f32), for the text-to-image layout.
    pub fn forward(
        &self,
        latents: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let layout = JointLayout::text_to_image(encoder_hidden_states.dim(1)?, height, width);
        self.forward_joint(encoder_hidden_states, &[latents], timestep, &layout)
    }

    /// The general joint forward: `text` `[1, L, context_in_dim]` and one `[1, h·w, in_channels]`
    /// tensor per [`Segment::Image`] of `layout` in order (condition images first, target last).
    /// Returns the **target block's** velocity `[1, target_tokens, out_channels]` in f32.
    pub fn forward_joint(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<Tensor> {
        self.run_joint(text, images, timestep, layout, Trace(None))
    }

    /// [`Self::forward_joint`] that also captures every named intermediate (`txt_in`, `img_in`,
    /// `temb`, `modulation`, `block_{i}_attn` / `_mlp` / `_out`, `norm_out`, `proj_out`) in
    /// forward order — the localisation seam the parity tests read. Returns the velocity and the
    /// trace.
    pub fn forward_joint_traced(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let velocity = self.run_joint(text, images, timestep, layout, Trace(Some(&mut trace)))?;
        Ok((velocity, trace))
    }

    fn run_joint(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
        mut trace: Trace<'_>,
    ) -> Result<Tensor> {
        layout.validate()?;
        let dtype = self.compute_dtype();
        let eps = self.cfg.eps as f64;
        let image_segments = layout
            .segments
            .iter()
            .filter(|s| matches!(s, Segment::Image { .. }))
            .count();
        if images.len() != image_segments {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: layout has {image_segments} image blocks but {} latent tensors were given",
                images.len()
            )));
        }
        let text_len: usize = layout
            .segments
            .iter()
            .map(|s| match s {
                Segment::Text { len } => *len,
                Segment::Image { .. } => 0,
            })
            .sum();
        if text.dim(1)? != text_len {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: layout expects {text_len} text tokens, got {}",
                text.dim(1)?
            )));
        }

        // Projections into the joint width.
        let txt = zero_center_rms_norm(&text.to_dtype(dtype)?, &self.txt_norm, eps)?;
        let txt = self
            .txt_out
            .forward(&gelu_tanh(&self.txt_in.forward(&txt)?)?)?;
        trace.push("txt_in", &txt)?;
        let mut pieces: Vec<Tensor> = Vec::with_capacity(layout.segments.len());
        let (mut text_cursor, mut image_cursor) = (0usize, 0usize);
        for segment in &layout.segments {
            match *segment {
                Segment::Text { len } => {
                    pieces.push(txt.narrow(1, text_cursor, len)?);
                    text_cursor += len;
                }
                Segment::Image { height, width } => {
                    let img = images[image_cursor];
                    if img.dim(1)? != height * width {
                        return Err(Error::Msg(format!(
                            "qwen_image_2_1: image block {image_cursor} expects {} tokens, got {}",
                            height * width,
                            img.dim(1)?
                        )));
                    }
                    let projected = self.img_in.forward(&img.to_dtype(dtype)?)?;
                    trace.push("img_in", &projected)?;
                    pieces.push(projected);
                    image_cursor += 1;
                }
            }
        }
        let mut x = Tensor::cat(&pieces, 1)?.contiguous()?;
        let s = x.dim(1)?;

        let (cos, sin) = self.rope(layout)?;

        // Timestep rows: the sampled `t` and, under `causal_condition`, an extra `t = 0` row.
        let ts: Vec<f32> = if self.cfg.causal_condition {
            vec![timestep * TIME_FACTOR, 0.0]
        } else {
            vec![timestep * TIME_FACTOR]
        };
        let proj = timestep_sincos(&ts, TIMESTEP_DIM, TIMESTEP_MAX_PERIOD, &self.device)?
            .to_dtype(dtype)?;
        let temb = self
            .time_out
            .forward(&silu(&self.time_in.forward(&proj)?)?)?;
        trace.push("temb", &temb)?;
        let silu_temb = silu(&temb)?;
        let modulation = self.modulation.forward(&silu_temb)?; // [rows, 4·inner]
        trace.push("modulation", &modulation)?;
        let inner = self.cfg.inner_dim();
        // `_select_modulation_rows`: target tokens take row 0 (their sample's timestep), every
        // other token the trailing `t = 0` row. Expressed as `zero + (real − zero)·mask` for a 0/1
        // mask, which is exactly `where(mask, real, zero)`.
        let target_mask = if self.cfg.causal_condition {
            let m: Vec<f32> = layout
                .target_mask()
                .into_iter()
                .map(|b| if b { 1.0 } else { 0.0 })
                .collect();
            Some(Tensor::from_vec(m, (1, s, 1), &self.device)?.to_dtype(dtype)?)
        } else {
            None
        };
        let select = |rows: &Tensor, slot: usize, width: usize| -> Result<Tensor> {
            let real = rows
                .narrow(0, 0, 1)?
                .narrow(1, slot * width, width)?
                .reshape((1, 1, width))?;
            let Some(mask) = target_mask.as_ref() else {
                return Ok(real.broadcast_as((1, s, width))?.contiguous()?);
            };
            let zero = rows
                .narrow(0, 1, 1)?
                .narrow(1, slot * width, width)?
                .reshape((1, 1, width))?;
            Ok(zero.broadcast_add(&real.broadcast_sub(&zero)?.broadcast_mul(mask)?)?)
        };
        let m = Modulation {
            scale1: select(&modulation, 0, inner)?,
            gate1: select(&modulation, 1, inner)?,
            scale2: select(&modulation, 2, inner)?,
            gate2: select(&modulation, 3, inner)?,
        };
        let out_scale = select(&self.norm_out.forward(&silu_temb)?, 0, inner)?;

        let prefix_segments = layout.prefix_segments();
        let prefix_len = layout.prefix_len();
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                index,
                &x,
                &m,
                &cos,
                &sin,
                &prefix_segments,
                prefix_len,
                &mut trace,
            )?;
        }
        let x = scale_residual(&layer_norm(&x, eps)?, &out_scale)?;
        trace.push("norm_out", &x)?;
        let out = self.proj_out.forward(&x)?;
        trace.push("proj_out", &out)?;
        let target = layout.target_tokens();
        Ok(out.i((.., s - target..s, ..))?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The joint sequence is long enough at the SHIPPED sizes that the F-003 i32-overflow guard is
    /// load-bearing, so every SDPA call here must carry the budget rather than the un-chunked
    /// `usize::MAX` sentinel.
    ///
    /// Two halves, because either alone rots. The arithmetic half states *why* — at the 1:1 2048²
    /// default the unmasked target call's scores exceed `i32::MAX`, which candle's CUDA kernels
    /// index with, so an unguarded pass silently corrupts the tail and returns a wrong image behind
    /// a green exit code. The source half is what actually catches a regression: no fixture in this
    /// crate is within three orders of the budget, so a revert to `usize::MAX` would leave every
    /// parity test green.
    #[test]
    fn the_joint_attention_is_budgeted_because_the_shipped_sizes_overflow_i32() {
        // 1:1 2048²: 2·(2048/32) = 128 latent tokens per side.
        let target_tokens = (2 * (2048 / 32)) * (2 * (2048 / 32));
        assert_eq!(target_tokens, 16_384);
        let cfg = TransformerConfig::production();
        // The target call attends over the whole joint sequence; the prompt only lengthens it.
        let scores = cfg.num_attention_heads * target_tokens * target_tokens;
        assert!(
            scores > i32::MAX as usize,
            "{scores} scores elements must exceed i32::MAX ({}) for the guard to matter",
            i32::MAX
        );
        assert!(
            scores > candle_gen::ATTN_SCORES_BUDGET,
            "{scores} must exceed the budget, so the planner really chunks at the default preset"
        );

        // Only the SHIPPED half of this file: the scan must not count its own assertions, and a
        // `#[cfg(test)]` helper could not affect a render anyway.
        let shipped = include_str!("transformer.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("the file has a shipped half");
        let calls = shipped.matches("sdpa_budgeted_bhsd(").count();
        assert_eq!(calls, 2, "one prefix-segment call and one target call");
        assert_eq!(
            shipped.matches("candle_gen::ATTN_SCORES_BUDGET,").count(),
            calls,
            "every SDPA call must pass the budget"
        );
        // Code only — the prose above names the sentinel to explain why it is wrong.
        assert!(
            !shipped
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .any(|line| line.contains("usize::MAX")),
            "no SDPA call may pass the un-chunked sentinel"
        );
    }

    #[test]
    fn t2i_layout_positions_follow_upstream_rope() {
        // text 3, target 2x4: text positions 0,1,2 on every axis; image frame 3, height in
        // range(-1, 1), width in range(-2, 2), row-major.
        let layout = JointLayout::text_to_image(3, 2, 4);
        let ids = layout.position_ids();
        assert_eq!(ids.len(), 11);
        assert_eq!(&ids[..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(ids[3], [3, -1, -2]);
        assert_eq!(ids[6], [3, -1, 1]);
        assert_eq!(ids[7], [3, 0, -2]);
        assert_eq!(ids[10], [3, 0, 1]);
        assert_eq!(layout.prefix_len(), 3);
        assert_eq!(layout.target_tokens(), 8);
        assert_eq!(layout.prefix_segments(), vec![(0, 3, true)]);
        let mask = layout.target_mask();
        assert!(!mask[2] && mask[3] && mask[10]);
    }

    #[test]
    fn condition_image_blocks_keep_their_own_segments() {
        let layout = JointLayout {
            segments: vec![
                Segment::Text { len: 2 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
            ],
        };
        assert_eq!(
            layout.prefix_segments(),
            vec![(0, 2, true), (2, 6, false), (6, 7, true)]
        );
        let ids = layout.position_ids();
        // After the first block the shared position advanced by max(2, 2) = 2 → the trailing text
        // token sits at 4 and the target block freezes its frame axis at 5.
        assert_eq!(ids[6], [4, 4, 4]);
        assert_eq!(ids[7], [5, -1, -1]);
    }
}
