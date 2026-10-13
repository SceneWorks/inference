//! The Qwen-Image 2.1 single-stream DiT — a faithful port of diffusers'
//! `QwenImage21Transformer2DModel` (`transformer_qwenimage21.py` @
//! [`crate::UPSTREAM_DIFFUSERS_REVISION`]): the prefill path (`QwenImage21AttnProcessor`) and the
//! prefix-KV-cached decode path (`QwenImage21KVCache`, sc-25904). The candle twin of
//! `mlx-gen-qwen-image-2-1`'s `src/transformer.rs`.
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
//! Together those two make every block's **prefix** hidden states — and so its prefix keys and
//! values — independent of the denoise step: a prefix token never attends to the target and never
//! reads the sampled timestep. [`PrefixKvCache`] is upstream's `QwenImage21KVCache`: the first model
//! evaluation runs the full joint sequence and keeps each block's prefix K/V
//! ([`QwenImage21Transformer::forward_joint_extract`]); every later one feeds the blocks the target
//! tokens only and attends over `[cached prefix, target]`
//! ([`QwenImage21Transformer::forward_joint_cached`]).
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
use candle_gen::candle_nn::ops::{rms_norm, rms_norm_slow, softmax, softmax_last_dim};
use candle_gen::candle_nn::rotary_emb::{rope_i, rope_i_slow};
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

/// The per-block adapter targets — every Linear inside `transformer_blocks.{i}` — relative to the
/// block. The candle twin of `mlx_gen_qwen_image_2_1::transformer::BLOCK_ADAPTER_TARGETS` (same
/// spellings, same order), and the trainer's default target surface (sc-24160).
pub const BLOCK_ADAPTER_TARGETS: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
];

/// The adaptable Linears outside the blocks — the twin of the MLX crate's
/// `GLOBAL_ADAPTER_TARGETS`. Reachable by an explicit trainer `lora_target_modules` list, never by
/// the default.
pub const GLOBAL_ADAPTER_TARGETS: [&str; 8] = [
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
];

/// How the attention core's normalisation, rotation and softmax are realised.
///
/// * [`Ops::Fused`] — candle's fused kernels (`rms_norm`, `rope_i`, `softmax_last_dim`): the render
///   path, byte-for-byte what it has always been.
/// * [`Ops::Composable`] — their composable equivalents (`rms_norm_slow`, `rope_i_slow`, an
///   f32-upcast `softmax`): the **training** path (sc-24160). The fused kernels are `CustomOp`s with
///   no backward, so a loss built through them cannot reach the adapter factors. The softmax is
///   taken in f32 whatever the compute dtype, so a bf16 CUDA run does not exponentiate in bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ops {
    Fused,
    Composable,
}

/// Composable softmax over the last axis, computed in f32 and cast back.
fn softmax_f32(x: &Tensor) -> candle_core::Result<Tensor> {
    softmax(&x.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(x.dtype())
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
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, ops: Ops) -> Result<Tensor> {
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?.contiguous()?;
    let rotated = match ops {
        Ops::Fused => rope_i(&x32, cos, sin)?,
        Ops::Composable => rope_i_slow(&x32, cos, sin)?,
    };
    Ok(rotated.to_dtype(dtype)?)
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
/// `budget` is the F-003 i32-overflow guard's score-element budget, forwarded to every SDPA call.
/// Production passes [`candle_gen::ATTN_SCORES_BUDGET`] and **never** the un-chunked `usize::MAX`
/// sentinel: this family's joint sequence is long enough for the guard to be load-bearing at the
/// shipped sizes, not a formality. At the 1:1 2048² default the target block alone is
/// `2·(2048/32) = 128` latent tokens per side → 16384 image tokens, so the unmasked target call's
/// scores are `1 · 32 heads · 16384 · ~16.6k ≈ 8.7e9` elements — about 4× `i32::MAX`, which
/// candle's CUDA kernels index with. Unguarded that silently corrupts the tail of the scores
/// tensor: a green run, a wrong image.
///
/// The guard chunks the query rows, which is mathematically equivalent (each row's softmax is
/// independent) though not bitwise equal to a single pass. **That equivalence is a property of the
/// call shapes here, not a given** — a per-segment mask that does not narrow consistently along the
/// query axis would silently diverge under chunking — so it is a parameter rather than a constant
/// precisely so `chunked_and_unchunked_attention_agree_at_fixture_scale` can drive both branches on
/// the real fixture geometry. On CUDA at every shipped preset the target call always chunks; on the
/// fixtures it never would, which is why the test forces the budget instead of relying on size.
///
/// `softmax` is the score normaliser every call applies: the fused `softmax_last_dim` on the render
/// path, the composable f32 one on the training path ([`Ops`]).
#[allow(clippy::too_many_arguments)]
fn block_causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    prefix_segments: &[(usize, usize, bool)],
    prefix_len: usize,
    softmax: &dyn Fn(&Tensor) -> candle_core::Result<Tensor>,
    budget: usize,
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
            softmax,
            budget,
        )?);
    }
    let qt = q.narrow(2, prefix_len, s - prefix_len)?.contiguous()?;
    outputs.push(candle_gen::sdpa_budgeted_bhsd(
        &qt,
        &k.contiguous()?,
        &v.contiguous()?,
        scale,
        None,
        softmax,
        budget,
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

    /// `_qwenimage21_prepare_qkv` up to the cache bookkeeping: project, split into heads, QK-norm
    /// and rotate. Returns `(q, k, v)` as `[B, heads, S, head_dim]` with `q`/`k` post-norm **and
    /// post-RoPE**, `v` the raw `to_v` projection.
    fn project(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        ops: Ops,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let heads = |y: Tensor| -> Result<Tensor> {
            Ok(y.reshape((b, s, h, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = heads(self.to_q.forward(x)?)?;
        let k = heads(self.to_k.forward(x)?)?;
        let v = heads(self.to_v.forward(x)?)?;
        let norm = |y: &Tensor, w: &Tensor| -> Result<Tensor> {
            let (y32, w32) = (y.to_dtype(DType::F32)?, w.to_dtype(DType::F32)?);
            let normed = match ops {
                Ops::Fused => rms_norm(&y32, &w32, self.eps)?,
                Ops::Composable => rms_norm_slow(&y32, &w32, self.eps)?,
            };
            Ok(normed.to_dtype(v.dtype())?)
        };
        let q = apply_rope(&norm(&q, &self.norm_q)?, cos, sin, ops)?;
        let k = apply_rope(&norm(&k, &self.norm_k)?, cos, sin, ops)?;
        Ok((q, k, v))
    }

    /// The attention core both paths share: `q` `[B, heads, Sq, head_dim]` over `k`/`v`
    /// `[B, heads, Sk, head_dim]` through [`block_causal_attention`], then the head merge and
    /// `to_out`. This is the crate's **only** route to an SDPA call and the only place the scores
    /// budget is supplied, so the cached decode cannot leave the budgeted path the prefill is on.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
        ops: Ops,
    ) -> Result<Tensor> {
        let (b, _h, s, _d) = q.dims4()?;
        let (h, hd) = (self.heads, self.head_dim);
        let scale = (hd as f64).powf(-0.5);
        let softmax: &dyn Fn(&Tensor) -> candle_core::Result<Tensor> = match ops {
            Ops::Fused => &softmax_last_dim,
            Ops::Composable => &softmax_f32,
        };
        let o = block_causal_attention(
            q,
            k,
            v,
            scale,
            prefix_segments,
            prefix_len,
            softmax,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let o = o.transpose(1, 2)?.reshape((b, s, h * hd))?;
        Ok(self.to_out.forward(&o)?)
    }

    /// Block-causal attention over the joint sequence `x` `[B, S, inner]` — the prefill. With
    /// `capture`, the block's prefix `(k, v)` rows (exactly what the target's attention call
    /// consumes for those keys) are written to it as **owned, prefix-sized copies**
    /// ([`owned_prefix_rows`]): upstream's `kv_cache_mode == "extract"`. The copies are taken here,
    /// while the block's full K/V are alive anyway, and nothing else of this forward escapes it —
    /// the full K/V are released when it returns, exactly as without a capture.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
        ops: Ops,
        capture: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(x, cos, sin, ops)?;
        if let Some(capture) = capture {
            *capture = Some((
                owned_prefix_rows(&k, prefix_len)?,
                owned_prefix_rows(&v, prefix_len)?,
            ));
        }
        self.attend(&q, &k, &v, prefix_segments, prefix_len, ops)
    }

    /// The cached decode (`kv_cache_mode == "cached"`): `x` is the **target tokens only**
    /// `[B, T, inner]`, `cos`/`sin` their RoPE rows, and the target attends — unmasked, as the
    /// block-causal mask degenerates to for target rows — over `[cached prefix, target]` keys and
    /// values. With no prefix segments and a zero prefix length, [`block_causal_attention`] makes
    /// exactly its one target call: the same call, over the same keys, on the same budgeted path
    /// as the prefill's.
    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cached_k: &Tensor,
        cached_v: &Tensor,
        ops: Ops,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(x, cos, sin, ops)?;
        // The prefix is never empty: a layout without text does not get through the prefill
        // (`prelude`), so no cache can be extracted for one.
        let k = Tensor::cat(&[cached_k, &k], 2)?;
        let v = Tensor::cat(&[cached_v, &v], 2)?;
        self.attend(&q, &k, &v, &[], 0, ops)
    }
}

/// An owned, exactly-sized copy of the first `rows` tokens of `x` `[B, heads, S, head_dim]`.
///
/// A candle `narrow` is a **view**: it shares (and keeps alive) the storage it was cut from, so
/// caching a block's prefix rows as a view would pin that block's whole prefill K or V — target
/// rows included — for every later step. This is upstream's `key[:, cache_write_slice].clone()`
/// ("`clone()`, not `contiguous()`"), and the same trap: `contiguous()` returns the view itself
/// whenever the view already counts as contiguous (one head, or a full-length prefix).
/// `force_contiguous` always allocates a storage of the view's own shape and copies into it. The
/// view is detached first so the copy carries no backprop edge back to it either (a tracked
/// tensor's copy keeps its source alive through the op graph).
fn owned_prefix_rows(x: &Tensor, rows: usize) -> Result<Tensor> {
    Ok(x.narrow(2, 0, rows)?.detach().force_contiguous()?)
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

/// How one block's attention runs: the block-causal prefill over the whole joint sequence
/// (optionally capturing the prefix K/V), or the cached decode over the target tokens.
enum BlockAttention<'a> {
    Prefill {
        capture: Option<&'a mut Option<(Tensor, Tensor)>>,
    },
    Cached {
        k: &'a Tensor,
        v: &'a Tensor,
    },
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

    fn forward(
        &self,
        index: usize,
        x: &Tensor,
        m: &Modulation,
        geometry: &JointGeometry,
        ops: Ops,
        trace: &mut Trace<'_>,
    ) -> Result<Tensor> {
        self.forward_with(
            index,
            x,
            m,
            geometry,
            BlockAttention::Prefill { capture: None },
            ops,
            trace,
        )
    }

    /// The block body, shared by the prefill and the cached decode — only the attention differs.
    #[allow(clippy::too_many_arguments)]
    fn forward_with(
        &self,
        index: usize,
        x: &Tensor,
        m: &Modulation,
        geometry: &JointGeometry,
        attention: BlockAttention<'_>,
        ops: Ops,
        trace: &mut Trace<'_>,
    ) -> Result<Tensor> {
        let h = scale_residual(&layer_norm(x, self.eps)?, &m.scale1)?;
        let a = match attention {
            BlockAttention::Prefill { capture } => self.attn.forward(
                &h,
                &geometry.cos,
                &geometry.sin,
                &geometry.prefix_segments,
                geometry.prefix_len,
                ops,
                capture,
            )?,
            BlockAttention::Cached { k, v } => {
                self.attn
                    .forward_cached(&h, &geometry.cos, &geometry.sin, k, v, ops)?
            }
        };
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

/// The prefix K/V of every block for one prompt branch — the port of upstream's
/// `QwenImage21KVCache` (sc-25904), the candle twin of `mlx_gen_qwen_image_2_1::PrefixKvCache`.
///
/// Built **only** by [`QwenImage21Transformer::forward_joint_extract`], complete, and consumed by
/// [`QwenImage21Transformer::forward_joint_cached`]; there is no way to hold an empty or partly
/// filled one. It belongs to the branch (prompt + references + target geometry) it was extracted
/// from: true CFG keeps one for the positive branch and an independent one for the negative
/// branch, whose prompt is a different prefix.
///
/// Per block it stores `(k, v)` as `[1, heads, prefix_len, head_dim]` in the model's compute dtype
/// — `k` post-QK-norm **and post-RoPE**, `v` the raw `to_v` projection, i.e. exactly the rows the
/// target's attention consumes for the prefix keys. (Upstream and the MLX twin hold the same
/// values as `[1, prefix_len, heads, head_dim]`; this crate's attention runs heads-first, and
/// storing them that way saves a transposing copy of the whole prefix at every cached step.) Each
/// tensor owns a storage of exactly its own size: holding a cache holds [`Self::nbytes`] plus the
/// target block's RoPE rows, and nothing else of the forward that produced it.
pub struct PrefixKvCache {
    layers: Vec<(Tensor, Tensor)>,
    /// RoPE rows of the target block (`[target_tokens, head_dim / 2]` each, f32).
    target_cos: Tensor,
    target_sin: Tensor,
    prefix_len: usize,
    target_tokens: usize,
}

impl PrefixKvCache {
    /// Tokens of the cached prefix (text + condition-image blocks).
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Tokens of the target block this cache decodes for.
    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }

    /// Number of blocks cached — always the transformer's block count.
    pub fn num_blocks(&self) -> usize {
        self.layers.len()
    }

    /// Block `index`'s cached `(k, v)`, each `[1, heads, prefix_len, head_dim]`.
    pub fn layer(&self, index: usize) -> Option<(&Tensor, &Tensor)> {
        self.layers.get(index).map(|(k, v)| (k, v))
    }

    /// The geometry of a cached decode: the target block alone, with its own RoPE rows and no
    /// prefix segments.
    ///
    /// `modulation_mask[prefix_len:]` is all target tokens, so no token reads the `t = 0` row
    /// (`t0_tokens: 0`): `JointGeometry::select` broadcasts row 0 — the sampled timestep — to
    /// every token unchanged, exactly as upstream's `torch.where` and the MLX twin's
    /// `broadcast_to` do.
    fn target_geometry(&self, inner: usize) -> JointGeometry {
        JointGeometry {
            cos: self.target_cos.clone(),
            sin: self.target_sin.clone(),
            prefix_segments: Vec::new(),
            prefix_len: 0,
            seq_len: self.target_tokens,
            target_tokens: self.target_tokens,
            t0_tokens: 0,
            inner,
        }
    }

    /// Bytes the cached K/V hold: `2 · blocks · prefix_len · inner_dim · itemsize`.
    pub fn nbytes(&self) -> usize {
        self.layers
            .iter()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
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

    /// Visit every adaptable projection under its diffusers dotted key — the candle twin of the MLX
    /// host's adapter module map, and the walk `candle_gen::quant::install_dotted_adapters` (and the
    /// LoHa dense fold in [`crate::adapters`]) resolve adapter targets against. Each projection is
    /// visited exactly once, in load order:
    ///
    /// * `img_in`, `txt_in.in_layer`, `txt_in.out_layer`,
    ///   `time_text_embed.timestep_embedder.linear_{1,2}`, `modulation.1`;
    /// * per block `transformer_blocks.{i}.attn.{to_q, to_k, to_v, to_out.0}` and
    ///   `transformer_blocks.{i}.img_mlp.{gate_layer, proj, out}`;
    /// * `norm_out.linear`, `proj_out`.
    ///
    /// These are exactly the weight keys the DiT loads (minus the norm vectors), so a PEFT/diffusers
    /// adapter trained against the upstream module tree resolves 1:1.
    pub fn visit_adaptable_mut(
        &mut self,
        visitor: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_core::Result<()>,
    ) -> candle_core::Result<()> {
        visitor("img_in", &mut self.img_in)?;
        visitor("txt_in.in_layer", &mut self.txt_in)?;
        visitor("txt_in.out_layer", &mut self.txt_out)?;
        visitor(
            "time_text_embed.timestep_embedder.linear_1",
            &mut self.time_in,
        )?;
        visitor(
            "time_text_embed.timestep_embedder.linear_2",
            &mut self.time_out,
        )?;
        visitor("modulation.1", &mut self.modulation)?;
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("transformer_blocks.{index}");
            for (name, linear) in [
                ("attn.to_q", &mut block.attn.to_q),
                ("attn.to_k", &mut block.attn.to_k),
                ("attn.to_v", &mut block.attn.to_v),
                ("attn.to_out.0", &mut block.attn.to_out),
                ("img_mlp.gate_layer", &mut block.mlp.gate_layer),
                ("img_mlp.proj", &mut block.mlp.proj),
                ("img_mlp.out", &mut block.mlp.out),
            ] {
                visitor(&format!("{prefix}.{name}"), linear)?;
            }
        }
        visitor("norm_out.linear", &mut self.norm_out)?;
        visitor("proj_out", &mut self.proj_out)
    }

    /// The weight-free twin of [`Self::visit_adaptable_mut`]: every adaptable projection's dotted
    /// key and `[out, in]` base shape, in the same order, derived from `cfg` alone (sc-24157). The
    /// adapter preflight and the memory contract resolve adapter targets against this before any
    /// DiT weight is read; a test pins it to the visitor walk of a loaded DiT.
    pub fn adaptable_projections(cfg: &TransformerConfig) -> Vec<(String, (usize, usize))> {
        let inner = cfg.inner_dim();
        let hidden = inner * cfg.mlp_ratio;
        let mut out = vec![
            ("img_in".to_owned(), (inner, cfg.in_channels)),
            ("txt_in.in_layer".to_owned(), (inner, cfg.context_in_dim)),
            ("txt_in.out_layer".to_owned(), (inner, inner)),
            (
                "time_text_embed.timestep_embedder.linear_1".to_owned(),
                (inner, TIMESTEP_DIM),
            ),
            (
                "time_text_embed.timestep_embedder.linear_2".to_owned(),
                (inner, inner),
            ),
            ("modulation.1".to_owned(), (4 * inner, inner)),
        ];
        for index in 0..cfg.num_layers {
            for (name, shape) in [
                ("attn.to_q", (inner, inner)),
                ("attn.to_k", (inner, inner)),
                ("attn.to_v", (inner, inner)),
                ("attn.to_out.0", (inner, inner)),
                ("img_mlp.gate_layer", (hidden, inner)),
                ("img_mlp.proj", (hidden, inner)),
                ("img_mlp.out", (inner, hidden)),
            ] {
                out.push((format!("transformer_blocks.{index}.{name}"), shape));
            }
        }
        out.push(("norm_out.linear".to_owned(), (inner, inner)));
        out.push(("proj_out".to_owned(), (cfg.out_channels, inner)));
        out
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
        self.rope_for_ids(&layout.position_ids())
    }

    /// The RoPE rows for explicit `(frame, height, width)` positions — [`Self::rope`] is this over
    /// the whole layout, the cached decode over the target block's rows alone.
    fn rope_for_ids(&self, ids: &[[i32; 3]]) -> Result<(Tensor, Tensor)> {
        rope_sincos_from_ids(ids, &self.cfg.axes_dims_rope, ROPE_THETA, &self.device)
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
        self.run_joint(text, images, timestep, layout, Ops::Fused, Trace(None))
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
        let velocity = self.run_joint(
            text,
            images,
            timestep,
            layout,
            Ops::Fused,
            Trace(Some(&mut trace)),
        )?;
        Ok((velocity, trace))
    }

    fn run_joint(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
        ops: Ops,
        mut trace: Trace<'_>,
    ) -> Result<Tensor> {
        let prelude = self.prelude(text, images, timestep, layout, &mut trace)?;
        let geometry = &prelude.geometry;
        let m = geometry.modulation(&prelude.modulation)?;
        let out_scale = geometry.select(&prelude.out_rows, 0)?;
        let mut x = prelude.x.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(index, &x, &m, geometry, ops, &mut trace)?;
        }
        self.head(&x, &out_scale, geometry, &mut trace)
    }

    /// Everything before block 0: the text and image projections into one joint sequence, the
    /// RoPE table, and the shared modulation / `norm_out` rows (still **unselected** — one row per
    /// timestep; `JointGeometry::select` spreads them over the tokens).
    fn prelude(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
        trace: &mut Trace<'_>,
    ) -> Result<JointPrelude> {
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
                    // Keyed by segment: a multi-image layout (the edit path) would otherwise emit
                    // the SAME `img_in` key once per condition image, and every consumer that keys
                    // a trace by name — including `transformer_parity.rs`'s stage-count assertion —
                    // would silently stop holding instead of failing.
                    trace.push(format!("img_in_{image_cursor}"), &projected)?;
                    pieces.push(projected);
                    image_cursor += 1;
                }
            }
        }
        let x = Tensor::cat(&pieces, 1)?.contiguous()?;
        let s = x.dim(1)?;

        let (cos, sin) = self.rope(layout)?;

        let (modulation, out_rows) = self.modulation_rows(timestep, trace)?;
        // `_select_modulation_rows`: target tokens take row 0 (their sample's timestep), every
        // other token — the whole prefix, since the target is the trailing block — the `t = 0`
        // row (see `JointGeometry::select`).
        let t0_tokens = if self.cfg.causal_condition {
            layout.prefix_len()
        } else {
            0
        };
        Ok(JointPrelude {
            x,
            modulation,
            out_rows,
            geometry: JointGeometry {
                cos,
                sin,
                prefix_segments: layout.prefix_segments(),
                prefix_len: layout.prefix_len(),
                seq_len: s,
                target_tokens: layout.target_tokens(),
                t0_tokens,
                inner: self.cfg.inner_dim(),
            },
        })
    }

    /// The timestep embedding → `(modulation, norm_out)` rows, `[rows, 4·inner]` and
    /// `[rows, inner]`: row 0 is the sampled `t`; under `causal_condition` a trailing row is `t = 0`
    /// (the row every prefix token modulates from). Shared by the prefill and the cached decode,
    /// which keeps the two-row shape so the sampled row is computed identically on both.
    fn modulation_rows(&self, timestep: f32, trace: &mut Trace<'_>) -> Result<(Tensor, Tensor)> {
        let dtype = self.compute_dtype();
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
        Ok((modulation, self.norm_out.forward(&silu_temb)?))
    }

    // ── prefix KV cache (sc-25904) ──────────────────────────────────────────────────────────────

    /// The first model evaluation of a cached denoise — upstream's `kv_cache_mode="extract"`:
    /// [`Self::forward_joint`] over the full joint sequence, which also keeps every block's prefix
    /// K/V. Returns the target's velocity and the branch's [`PrefixKvCache`].
    ///
    /// Each block's cache entry is copied out **as that block runs** into a storage of its own
    /// size, so the extract never holds more than the prefill forward plus the cache it is
    /// building: `prefill peak + 2 · blocks · prefix_tokens · inner_dim · itemsize` (and the
    /// target's RoPE rows, `2 · target_tokens · head_dim / 2` f32).
    ///
    /// Refused unless the model is `causal_condition`: the cache is only valid because the prefix
    /// tokens modulate from `t = 0`, which is what makes their activations step-independent.
    pub fn forward_joint_extract(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<(Tensor, PrefixKvCache)> {
        let (velocity, cache, _) =
            self.run_joint_extract(text, images, timestep, layout, Trace(None))?;
        Ok((velocity, cache))
    }

    /// Every later model evaluation of a cached denoise — upstream's `kv_cache_mode="cached"`: the
    /// blocks receive **only** the target tokens `target` `[1, target_tokens, in_channels]`, and
    /// each block's target attention reads `cache`'s prefix K/V in front of its own. Returns the
    /// target's velocity `[1, target_tokens, out_channels]` in f32.
    ///
    /// Valid at any timestep and in any order — the cache does not depend on the step — so a
    /// solver that evaluates the model more than once per step, or off the schedule, is served by
    /// the same cache.
    pub fn forward_joint_cached(
        &self,
        target: &Tensor,
        timestep: f32,
        cache: &PrefixKvCache,
    ) -> Result<Tensor> {
        self.run_joint_cached(target, timestep, cache, Trace(None))
            .map(|(velocity, _)| velocity)
    }

    /// [`Self::forward_joint_cached`] that also captures the named intermediates (`img_in_target`,
    /// `temb`, `modulation`, `block_{i}_attn` / `_mlp` / `_out`, `norm_out`, `proj_out`) in forward
    /// order. Every joint-sequence stage is `[1, target_tokens, C]`: the tensors the blocks
    /// actually ran on.
    pub fn forward_joint_cached_traced(
        &self,
        target: &Tensor,
        timestep: f32,
        cache: &PrefixKvCache,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let (velocity, _) =
            self.run_joint_cached(target, timestep, cache, Trace(Some(&mut trace)))?;
        Ok((velocity, trace))
    }

    /// [`Self::forward_joint_extract`] plus the token count the block stack ran on (read off the
    /// stack's own output), for the denoise loop's report.
    pub(crate) fn extract_counted(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<(Tensor, PrefixKvCache, usize)> {
        self.run_joint_extract(text, images, timestep, layout, Trace(None))
    }

    /// [`Self::forward_joint_cached`] plus the token count the block stack ran on.
    pub(crate) fn cached_counted(
        &self,
        target: &Tensor,
        timestep: f32,
        cache: &PrefixKvCache,
    ) -> Result<(Tensor, usize)> {
        self.run_joint_cached(target, timestep, cache, Trace(None))
    }

    /// The typed refusal upstream raises for `kv_cache` without `causal_condition`.
    fn require_causal_condition(&self) -> Result<()> {
        if self.cfg.causal_condition {
            return Ok(());
        }
        Err(Error::Unsupported(
            "qwen_image_2_1: the prefix KV cache requires `causal_condition = true` — it is only \
             valid because text and condition-image tokens modulate from t = 0, which makes their \
             activations independent of the denoising step"
                .into(),
        ))
    }

    fn run_joint_extract(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
        mut trace: Trace<'_>,
    ) -> Result<(Tensor, PrefixKvCache, usize)> {
        self.require_causal_condition()?;
        let prelude = self.prelude(text, images, timestep, layout, &mut trace)?;
        let geometry = &prelude.geometry;
        let m = geometry.modulation(&prelude.modulation)?;
        let out_scale = geometry.select(&prelude.out_rows, 0)?;
        let mut layers: Vec<(Tensor, Tensor)> = Vec::with_capacity(self.blocks.len());
        let mut x = prelude.x.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            // The slot is filled inside the block's attention with owned, prefix-sized copies; the
            // block's full `[1, heads, prefix + target, head_dim]` K/V never leave it.
            let mut captured: Option<(Tensor, Tensor)> = None;
            x = block.forward_with(
                index,
                &x,
                &m,
                geometry,
                BlockAttention::Prefill {
                    capture: Some(&mut captured),
                },
                Ops::Fused,
                &mut trace,
            )?;
            layers.push(captured.ok_or_else(|| {
                Error::Msg(format!(
                    "qwen_image_2_1: block {index} captured no prefix K/V"
                ))
            })?);
        }
        let block_tokens = x.dim(1)?;
        let velocity = self.head(&x, &out_scale, geometry, &mut trace)?;

        // The target block's own RoPE rows, built at exactly their size rather than narrowed out
        // of (and so pinning) the joint table.
        let ids = layout.position_ids();
        let (target_cos, target_sin) = self.rope_for_ids(&ids[geometry.prefix_len..])?;
        let cache = PrefixKvCache {
            layers,
            target_cos,
            target_sin,
            prefix_len: geometry.prefix_len,
            target_tokens: geometry.target_tokens,
        };
        Ok((velocity, cache, block_tokens))
    }

    fn run_joint_cached(
        &self,
        target: &Tensor,
        timestep: f32,
        cache: &PrefixKvCache,
        mut trace: Trace<'_>,
    ) -> Result<(Tensor, usize)> {
        self.require_causal_condition()?;
        if cache.layers.len() != self.blocks.len() {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: a {}-block prefix KV cache was given to a {}-block DiT",
                cache.layers.len(),
                self.blocks.len()
            )));
        }
        if target.dim(1)? != cache.target_tokens {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the prefix KV cache was extracted for a {}-token target, got {}",
                cache.target_tokens,
                target.dim(1)?
            )));
        }
        let dtype = self.compute_dtype();

        let x = self.img_in.forward(&target.to_dtype(dtype)?)?;
        trace.push("img_in_target", &x)?;
        let (modulation, out_rows) = self.modulation_rows(timestep, &mut trace)?;
        let geometry = cache.target_geometry(self.cfg.inner_dim());
        let m = geometry.modulation(&modulation)?;
        let out_scale = geometry.select(&out_rows, 0)?;
        let mut x = x;
        for (index, (block, (k, v))) in self.blocks.iter().zip(&cache.layers).enumerate() {
            x = block.forward_with(
                index,
                &x,
                &m,
                &geometry,
                BlockAttention::Cached { k, v },
                Ops::Fused,
                &mut trace,
            )?;
        }
        let block_tokens = x.dim(1)?;
        Ok((
            self.head(&x, &out_scale, &geometry, &mut trace)?,
            block_tokens,
        ))
    }

    /// After the last block: the `norm_out` modulation, `proj_out`, and the target block's slice
    /// as f32.
    fn head(
        &self,
        x: &Tensor,
        out_scale: &Tensor,
        geometry: &JointGeometry,
        trace: &mut Trace<'_>,
    ) -> Result<Tensor> {
        let x = scale_residual(&layer_norm(x, self.cfg.eps as f64)?, out_scale)?;
        trace.push("norm_out", &x)?;
        let out = self.proj_out.forward(&x)?;
        trace.push("proj_out", &out)?;
        let (s, target) = (geometry.seq_len, geometry.target_tokens);
        Ok(out.i((.., s - target..s, ..))?.to_dtype(DType::F32)?)
    }

    // ── training (sc-24160) ─────────────────────────────────────────────────────────────────────

    /// Number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The **training** text-to-image forward: [`Self::forward`] with the attention core on its
    /// composable, differentiable ops ([`Ops::Composable`]), so `loss.backward()` reaches every
    /// trainable adapter factor. Same arguments, same `[1, h·w, out_channels]` f32 velocity.
    pub fn forward_train(
        &self,
        latents: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let layout = JointLayout::text_to_image(encoder_hidden_states.dim(1)?, height, width);
        self.forward_train_joint(encoder_hidden_states, &[latents], timestep, &layout)
    }

    /// The **training** joint forward (sc-24162): [`Self::forward_joint`] — the render path's
    /// general text/condition-image/target forward — on the composable, differentiable ops. Same
    /// arguments (`images` = condition latents in order, target last), same target-block-only
    /// `[1, target_tokens, out_channels]` f32 velocity, so a loss over it covers target tokens only.
    pub fn forward_train_joint(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<Tensor> {
        self.run_joint(text, images, timestep, layout, Ops::Composable, Trace(None))
    }

    /// The pre-block half of [`Self::forward_train`], for a gradient-checkpointed step: the joint
    /// sequence entering block 0 plus the modulation rows every block (and the head) reads. The
    /// global projections that produce it (`img_in`, `txt_in.*`, the timestep embedder,
    /// `modulation.1`, `norm_out.linear`) run here, retained, so an adapter on one of them trains
    /// through ordinary autograd.
    pub fn train_prelude(
        &self,
        latents: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        height: usize,
        width: usize,
    ) -> Result<JointPrelude> {
        let layout = JointLayout::text_to_image(encoder_hidden_states.dim(1)?, height, width);
        self.train_prelude_joint(encoder_hidden_states, &[latents], timestep, &layout)
    }

    /// The pre-block half of [`Self::forward_train_joint`] (sc-24162) — [`Self::train_prelude`]
    /// over a general joint layout (condition images first, target last).
    pub fn train_prelude_joint(
        &self,
        text: &Tensor,
        images: &[&Tensor],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<JointPrelude> {
        self.prelude(text, images, timestep, layout, &mut Trace(None))
    }

    /// Block `index` of [`Self::forward_train`] on its own — one checkpoint segment. `modulation`
    /// is [`JointPrelude::modulation`] (passed explicitly so a checkpointed backward can carry it
    /// across the segment boundary as a differentiable input).
    pub fn train_block(
        &self,
        index: usize,
        x: &Tensor,
        modulation: &Tensor,
        geometry: &JointGeometry,
    ) -> Result<Tensor> {
        let block = self.blocks.get(index).ok_or_else(|| {
            Error::Msg(format!(
                "qwen_image_2_1: block {index} out of range ({} blocks)",
                self.blocks.len()
            ))
        })?;
        let m = geometry.modulation(modulation)?;
        block.forward(index, x, &m, geometry, Ops::Composable, &mut Trace(None))
    }

    /// The post-block head of [`Self::forward_train`]: `out_rows` is [`JointPrelude::out_rows`].
    /// Returns the target's `[1, h·w, out_channels]` velocity in f32.
    pub fn train_head(
        &self,
        x: &Tensor,
        out_rows: &Tensor,
        geometry: &JointGeometry,
    ) -> Result<Tensor> {
        let out_scale = geometry.select(out_rows, 0)?;
        self.head(x, &out_scale, geometry, &mut Trace(None))
    }
}

/// The joint sequence's static geometry for one forward: the RoPE table, the block-causal segment
/// runs, and how many leading tokens read the `t = 0` modulation row. Constant across the blocks
/// (it carries no trainable dependency).
pub struct JointGeometry {
    cos: Tensor,
    sin: Tensor,
    prefix_segments: Vec<(usize, usize, bool)>,
    prefix_len: usize,
    seq_len: usize,
    target_tokens: usize,
    /// Leading tokens that modulate from the `t = 0` row: the whole prefix for a
    /// `causal_condition` prefill, `0` when every token reads the sampled row (a model without
    /// `causal_condition`, or a cached decode, whose sequence is the target alone).
    t0_tokens: usize,
    inner: usize,
}

impl JointGeometry {
    /// Tokens of the whole joint sequence.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// `_select_modulation_rows` for slot `slot` of `rows` (`[rows, k·inner]`) → `[1, S, inner]`:
    /// the first `t0_tokens` tokens take the trailing `t = 0` row, every
    /// later one row 0 (its sample's timestep).
    ///
    /// An **exact** select — upstream's `torch.where(mask, real, zero)`: each output row is a copy
    /// of one of the two input rows, in `rows`' own dtype. It is built as a concatenation of the
    /// two broadcast rows rather than a per-token `where`, which is the same thing because the
    /// target is always the trailing block of a joint layout (`JointLayout::validate`), so the
    /// `t = 0` tokens are exactly the leading `prefix_len`; a geometry whose prefix and target do
    /// not tile the sequence is refused rather than mis-selected. Differentiable in both rows.
    fn select(&self, rows: &Tensor, slot: usize) -> Result<Tensor> {
        let (s, width) = (self.seq_len, self.inner);
        let row = |index: usize| -> Result<Tensor> {
            Ok(rows
                .narrow(0, index, 1)?
                .narrow(1, slot * width, width)?
                .reshape((1, 1, width))?)
        };
        let real = row(0)?;
        if self.t0_tokens == 0 {
            return Ok(real.broadcast_as((1, s, width))?.contiguous()?);
        }
        if self.t0_tokens + self.target_tokens != s {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the modulation select needs the target block to be the trailing \
                 {} of {s} tokens, but {} leading tokens read the t = 0 row",
                self.target_tokens, self.t0_tokens
            )));
        }
        let zero = row(1)?;
        Ok(Tensor::cat(
            &[
                zero.broadcast_as((1, self.t0_tokens, width))?,
                real.broadcast_as((1, self.target_tokens, width))?,
            ],
            1,
        )?)
    }

    /// The four per-token block modulations from the `[rows, 4·inner]` shared modulation.
    fn modulation(&self, rows: &Tensor) -> Result<Modulation> {
        Ok(Modulation {
            scale1: self.select(rows, 0)?,
            gate1: self.select(rows, 1)?,
            scale2: self.select(rows, 2)?,
            gate2: self.select(rows, 3)?,
        })
    }
}

/// [`QwenImage21Transformer::train_prelude`]'s output: the differentiable state entering block 0
/// and the static [`JointGeometry`].
pub struct JointPrelude {
    /// `[1, S, inner]` — the joint sequence entering block 0.
    pub x: Tensor,
    /// `[rows, 4·inner]` — the shared modulation rows (row 0 the sampled timestep, row 1 `t = 0`
    /// under `causal_condition`).
    pub modulation: Tensor,
    /// `[rows, inner]` — the `norm_out.linear` rows the head modulates with.
    pub out_rows: Tensor,
    /// The static geometry every block and the head read.
    pub geometry: JointGeometry,
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
        // Both SDPA calls forward the parameter, and exactly one production call site supplies the
        // constant. This is what catches `block_causal_attention` hardcoding a budget of its own
        // and ignoring its argument — a mutation the equivalence test below cannot see, because
        // both of its branches would then take the same path and agree trivially.
        assert_eq!(
            shipped.matches("        budget,\n").count(),
            calls,
            "every SDPA call must forward the `budget` parameter"
        );
        assert_eq!(
            shipped.matches("candle_gen::ATTN_SCORES_BUDGET,\n").count(),
            1,
            "exactly one production call site supplies the budget constant"
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

    /// The chunked branch of the guard and the un-chunked one agree **on this crate's call
    /// shapes**, at the real fixture geometry.
    ///
    /// This is the half the arithmetic above cannot assert. The guard splits the query axis and
    /// narrows each per-segment mask with it; a mask that does not narrow consistently (a dropped
    /// head axis, a `[1, 1, 1, Sk]` broadcast standing in for a per-row one) is *self*-consistent
    /// in a single pass and silently wrong once chunked — and on CUDA the target call chunks at
    /// every shipped preset while on the fixtures it never would. So the budget is forced small
    /// here rather than waited for.
    ///
    /// Tolerance 1e-5, not equality: the driver's own docs say a different query-block `M` changes
    /// the GEMM accumulation order, so the two agree to ~1 ULP rather than bitwise (SC-15943 is the
    /// defect a sibling crate hit by asserting `== 0.0` on exactly this difference).
    #[test]
    fn chunked_and_unchunked_attention_agree_at_fixture_scale() {
        use candle_gen::attention::attention_budget_from_usize;

        // The `wide` fixture's geometry: 5 text tokens then a 2x4 target block.
        let layout = JointLayout::text_to_image(5, 2, 4);
        let (s, heads, head_dim) = (layout.total_len(), 4usize, 8usize);
        let dev = Device::Cpu;
        // Deterministic, non-degenerate q/k/v — a constant tensor would make every chunk boundary
        // agree for the wrong reason.
        let fill = |seed: usize| -> Tensor {
            let n = heads * s * head_dim;
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 11) % 97) as f32 / 97.0) - 0.5)
                .collect();
            Tensor::from_vec(v, (1, heads, s, head_dim), &dev).unwrap()
        };
        let (q, k, v) = (fill(0), fill(1), fill(2));
        let scale = (head_dim as f64).powf(-0.5);
        let segments = layout.prefix_segments();
        let prefix = layout.prefix_len();

        // A budget of one score element per query row forces the planner to chunk; assert that it
        // really does, so the test cannot pass by never taking the branch it exists to exercise.
        let rows_per_query = heads * s;
        let forced = rows_per_query;
        assert!(
            (attention_budget_from_usize(forced).query_block_rows(rows_per_query as u64, s as u64)
                as usize)
                < s,
            "the forced budget must actually chunk the query axis"
        );
        assert_eq!(
            attention_budget_from_usize(usize::MAX)
                .query_block_rows(rows_per_query as u64, s as u64) as usize,
            s,
            "the sentinel must leave the query axis whole"
        );

        let chunked = block_causal_attention(
            &q,
            &k,
            &v,
            scale,
            &segments,
            prefix,
            &softmax_last_dim,
            forced,
        )
        .unwrap();
        let single = block_causal_attention(
            &q,
            &k,
            &v,
            scale,
            &segments,
            prefix,
            &softmax_last_dim,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(chunked.dims(), single.dims());
        let diff = (&chunked - &single)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("chunked_vs_unchunked: max|Δ|={diff:.3e} bound=1.000e-5");
        assert!(
            diff <= 1e-5,
            "the chunked branch diverged from the single pass by {diff:.3e}"
        );
    }

    /// The cached decode's attention call — target queries over `[cached prefix, target]` keys,
    /// no segments — is the prefill's own target call: same rows out, and on the same budgeted
    /// path, where it chunks consistently although its key axis is longer than its query axis.
    ///
    /// At the shipped sizes this call **always** chunks (see the arithmetic test below), and no
    /// fixture would make it, so the budget is forced here. The cached keys are an owned copy of
    /// the prefix rows, as the cache holds them.
    #[test]
    fn the_cached_attention_call_is_the_prefill_target_call_chunked_or_not() {
        use candle_gen::attention::attention_budget_from_usize;

        // Two condition blocks around text, then a 4x6 target — the edit route's shape.
        let layout = JointLayout {
            segments: vec![
                Segment::Text { len: 5 },
                Segment::Image {
                    height: 2,
                    width: 3,
                },
                Segment::Text { len: 2 },
                Segment::Image {
                    height: 4,
                    width: 6,
                },
            ],
        };
        let (s, heads, head_dim) = (layout.total_len(), 4usize, 8usize);
        let (prefix, target) = (layout.prefix_len(), layout.target_tokens());
        let dev = Device::Cpu;
        let fill = |seed: usize| -> Tensor {
            let n = heads * s * head_dim;
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 11) % 97) as f32 / 97.0) - 0.5)
                .collect();
            Tensor::from_vec(v, (1, heads, s, head_dim), &dev).unwrap()
        };
        let (q, k, v) = (fill(0), fill(1), fill(2));
        let scale = (head_dim as f64).powf(-0.5);
        let want = block_causal_attention(
            &q,
            &k,
            &v,
            scale,
            &layout.prefix_segments(),
            prefix,
            &softmax_last_dim,
            usize::MAX,
        )
        .unwrap()
        .narrow(2, prefix, target)
        .unwrap();

        let rows = |t: &Tensor| t.narrow(2, prefix, target).unwrap().contiguous().unwrap();
        let cached_k = owned_prefix_rows(&k, prefix).unwrap();
        let cached_v = owned_prefix_rows(&v, prefix).unwrap();
        let k_cat = Tensor::cat(&[&cached_k, &rows(&k)], 2).unwrap();
        let v_cat = Tensor::cat(&[&cached_v, &rows(&v)], 2).unwrap();
        let rows_per_query = heads * s;
        let forced = rows_per_query;
        assert!(
            (attention_budget_from_usize(forced)
                .query_block_rows(rows_per_query as u64, target as u64) as usize)
                < target,
            "the forced budget must actually chunk the cached call's query axis"
        );
        for (label, budget) in [("single pass", usize::MAX), ("chunked", forced)] {
            let got = block_causal_attention(
                &rows(&q),
                &k_cat,
                &v_cat,
                scale,
                &[],
                0,
                &softmax_last_dim,
                budget,
            )
            .unwrap();
            assert_eq!(got.dims(), want.dims(), "{label}");
            let diff = (&got - &want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!("cached call vs prefill target rows ({label}): max|Δ|={diff:.3e}");
            assert!(
                diff <= 1e-5,
                "{label}: the cached call diverged from the prefill's target rows by {diff:.3e}"
            );
        }
    }

    /// Every tensor the cached path adds stays inside candle's CUDA index widths at the worst
    /// shipped request — the largest-area preset with ten references — **because** its attention
    /// is on the budgeted path; unbudgeted, its scores alone are ~14× `i32::MAX`.
    ///
    /// candle's CUDA kernels index elements with `i32` (softmax, elementwise, matmul) and launch
    /// with a `u32` element count (`LaunchConfig::for_num_elems`, the strided copy behind `cat`
    /// and `force_contiguous`). The numbers below are the element counts of each op the cached
    /// step and the extract's capture add; the source half of
    /// `the_joint_attention_is_budgeted_because_the_shipped_sizes_overflow_i32` is what pins the
    /// cached call to the budget.
    #[test]
    fn the_cached_path_stays_inside_cuda_index_widths_at_the_ten_reference_boundary() {
        use candle_gen::attention::attention_budget_from_usize;

        let cfg = TransformerConfig::production();
        let geometry = crate::memory_strategy::admission_geometry();
        let target = geometry.max_target_image_tokens as usize;
        let joint = geometry.max_joint_tokens as usize;
        let prefix = joint - target;
        assert_eq!((target, prefix, joint), (16_800, 41_216, 58_016));
        let (heads, head_dim) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let i32_max = i32::MAX as usize;

        // The cached step's one attention call: target queries over prefix + target keys.
        let scores = heads * target * joint;
        assert!(
            scores > 14 * i32_max && scores > u32::MAX as usize,
            "{scores} unbudgeted score elements must overflow both index widths"
        );
        let rows_per_query = heads * joint;
        let block = attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET)
            .query_block_rows(rows_per_query as u64, target as u64) as usize;
        assert!(
            (1..target).contains(&block),
            "the cached call must chunk at this size (block {block} of {target} rows)"
        );
        // One chunk's scores / probs (`matmul`, `· scale`, `softmax_last_dim`).
        assert!(block * rows_per_query <= candle_gen::ATTN_SCORES_BUDGET);
        assert!(candle_gen::ATTN_SCORES_BUDGET < i32_max);

        // `cat([cached, target], 2)` and the `kᵀ` / `v` the SDPA makes contiguous: one full-length
        // K or V. The capture's `narrow → force_contiguous` reads from a tensor of the same size
        // and writes the prefix rows. The cached step's modulation rows are `[1, target, inner]`.
        for (what, elements) in [
            ("a full-length K or V", heads * joint * head_dim),
            ("one cached prefix K or V", heads * prefix * head_dim),
            ("the per-token modulation", target * cfg.inner_dim()),
        ] {
            assert!(
                elements < i32_max,
                "{what} has {elements} elements, past i32::MAX"
            );
        }
    }

    /// The committed miniature DiT (the MLX twin's tiny snapshot), f32 on CPU.
    fn tiny_dit() -> QwenImage21Transformer {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/tiny-snapshot");
        let cfg = TransformerConfig::from_json_file(&root.join("transformer/config.json")).unwrap();
        let vb = candle_gen::loader::component_vb(
            &root,
            "transformer",
            DType::F32,
            &Device::Cpu,
            "select",
        )
        .unwrap();
        QwenImage21Transformer::new(&cfg, vb).unwrap()
    }

    /// Text, two reference blocks and a non-square target — and the prelude the prefill forward
    /// (`run_joint`, `run_joint_extract`) and the trainer's checkpointed pieces all start from.
    fn edit_prelude(dit: &QwenImage21Transformer) -> (JointLayout, JointPrelude) {
        let layout = JointLayout {
            segments: vec![
                Segment::Text { len: 3 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 2,
                    width: 4,
                },
                Segment::Text { len: 4 },
                Segment::Image {
                    height: 2,
                    width: 6,
                },
            ],
        };
        let c = dit.config();
        let fill = |tokens: usize, width: usize, seed: usize| -> Tensor {
            let v: Vec<f32> = (0..tokens * width)
                .map(|i| (((i * 31 + seed * 17) % 89) as f32 / 89.0) - 0.5)
                .collect();
            Tensor::from_vec(v, (1, tokens, width), &Device::Cpu).unwrap()
        };
        let text = fill(8, c.context_in_dim, 1);
        let images = [
            fill(4, c.in_channels, 2),
            fill(8, c.in_channels, 3),
            fill(12, c.in_channels, 4),
        ];
        let refs: Vec<&Tensor> = images.iter().collect();
        let prelude = dit.train_prelude_joint(&text, &refs, 0.6, &layout).unwrap();
        (layout, prelude)
    }

    fn f32_bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Two modulation rows (`[2, slots·inner]`) on which `zero + (real − zero)` visibly rounds in
    /// f32 — a large `t = 0` entry absorbs a small sampled one.
    fn rounding_rows(slots: usize, inner: usize) -> Tensor {
        let real: Vec<f32> = (0..slots * inner).map(|i| 0.1 + i as f32 * 0.013).collect();
        let zero: Vec<f32> = (0..slots * inner)
            .map(|i| if i % 2 == 0 { 1.0e9 } else { -3.7e4 })
            .collect();
        Tensor::from_vec([real, zero].concat(), (2, slots * inner), &Device::Cpu).unwrap()
    }

    /// The prefill's per-token modulation is an **exact** select — upstream's `torch.where`: every
    /// prefix token receives the `t = 0` row and every target token the sampled-timestep row, bit
    /// for bit, in all four block modulation slots and in the `norm_out` select. Asked of the
    /// tensors the blocks and the head are handed (`JointGeometry::modulation` / `select` on the
    /// prelude's own geometry), on the model's own rows and on rows where a blend
    /// `zero + (real − zero)·mask` rounds, in f32 and in **bf16** — the dtype production runs on
    /// CUDA/Metal, where that blend was off by percents and which no forward on this lane reaches
    /// (the select needs no matmul, so it does run in bf16 on CPU).
    #[test]
    fn the_prefill_selects_each_modulation_row_exactly() {
        let dit = tiny_dit();
        let (layout, prelude) = edit_prelude(&dit);
        let inner = dit.config().inner_dim();
        let geometry = &prelude.geometry;
        let (prefix, total) = (layout.prefix_len(), layout.total_len());
        assert_eq!((prefix, total, geometry.seq_len()), (20, 32, 32));
        // The mask upstream selects with: exactly the trailing target block.
        assert_eq!(
            layout.target_mask(),
            (0..total).map(|i| i >= prefix).collect::<Vec<_>>()
        );

        // `rows` → per-token `[1, S, inner]` must be row 1 on the prefix and row 0 on the target.
        let check = |what: &str, selected: &Tensor, rows: &Tensor, slot: usize| {
            assert_eq!(selected.dims(), &[1, total, inner], "{what}");
            assert_eq!(
                selected.dtype(),
                rows.dtype(),
                "{what}: the rows' own dtype"
            );
            let row = |index: usize| {
                f32_bits(
                    &rows
                        .narrow(0, index, 1)
                        .unwrap()
                        .narrow(1, slot * inner, inner)
                        .unwrap(),
                )
            };
            let (sampled, t0) = (row(0), row(1));
            assert_ne!(sampled, t0, "{what}: the two rows must differ");
            let got = f32_bits(selected);
            for token in 0..total {
                let want = if token < prefix { &t0 } else { &sampled };
                assert_eq!(
                    &got[token * inner..(token + 1) * inner],
                    &want[..],
                    "{what}: token {token} ({})",
                    if token < prefix { "prefix" } else { "target" }
                );
            }
        };
        for dtype in [DType::F32, DType::BF16] {
            for (name, modulation, out_rows) in [
                (
                    "model rows",
                    prelude.modulation.to_dtype(dtype).unwrap(),
                    prelude.out_rows.to_dtype(dtype).unwrap(),
                ),
                (
                    "rounding rows",
                    rounding_rows(4, inner).to_dtype(dtype).unwrap(),
                    rounding_rows(1, inner).to_dtype(dtype).unwrap(),
                ),
            ] {
                let m = geometry.modulation(&modulation).unwrap();
                for (slot, selected) in [&m.scale1, &m.gate1, &m.scale2, &m.gate2]
                    .into_iter()
                    .enumerate()
                {
                    check(
                        &format!("{dtype:?} {name} slot {slot}"),
                        selected,
                        &modulation,
                        slot,
                    );
                }
                check(
                    &format!("{dtype:?} {name} norm_out"),
                    &geometry.select(&out_rows, 0).unwrap(),
                    &out_rows,
                    0,
                );
            }
        }
    }

    /// The select is correct only because the `t = 0` tokens are the leading prefix and the target
    /// the trailing block. A geometry in which the two do not tile the sequence is refused, not
    /// mis-selected.
    #[test]
    fn a_geometry_whose_target_is_not_the_trailing_block_is_refused() {
        let dit = tiny_dit();
        let (_, prelude) = edit_prelude(&dit);
        let mut geometry = prelude.geometry;
        geometry.t0_tokens -= 1;
        let err = geometry
            .select(&prelude.out_rows, 0)
            .expect_err("19 t = 0 tokens + 12 target tokens do not tile 32");
        assert_eq!(
            err.to_string(),
            "qwen_image_2_1: the modulation select needs the target block to be the trailing 12 \
             of 32 tokens, but 19 leading tokens read the t = 0 row"
        );
    }

    /// The select stays differentiable in **both** rows: a loss over the selected rows sends each
    /// row exactly the gradient its tokens carry — the sampled row the target tokens', the `t = 0`
    /// row the prefix tokens' — and nothing to the other slots. Analytic, so the comparison is to
    /// f32 summation error.
    #[test]
    fn the_select_passes_each_rows_gradient_to_its_own_tokens() {
        let dit = tiny_dit();
        let (layout, prelude) = edit_prelude(&dit);
        let inner = dit.config().inner_dim();
        let (prefix, total) = (layout.prefix_len(), layout.total_len());
        let rows = candle_core::Var::from_tensor(&prelude.modulation).unwrap();
        let weights: Vec<f32> = (0..total * inner)
            .map(|i| (((i * 13) % 41) as f32 / 41.0) - 0.4)
            .collect();
        let w = Tensor::from_vec(weights.clone(), (1, total, inner), &Device::Cpu).unwrap();
        let slot = 2usize;
        let loss = (prelude.geometry.select(rows.as_tensor(), slot).unwrap() * &w)
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        let grad = grads
            .get(rows.as_tensor())
            .expect("the select must reach its rows")
            .to_vec2::<f32>()
            .unwrap();
        let column_sum = |tokens: std::ops::Range<usize>, channel: usize| -> f32 {
            tokens.map(|t| weights[t * inner + channel]).sum()
        };
        let mut reached = [false; 2];
        for (row, tokens) in [(0usize, prefix..total), (1, 0..prefix)] {
            for (column, got) in grad[row].iter().enumerate() {
                let want = if column / inner == slot {
                    column_sum(tokens.clone(), column % inner)
                } else {
                    0.0
                };
                reached[row] |= want.abs() > 0.1;
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "row {row} column {column}: gradient {got} vs {want}"
                );
            }
        }
        assert_eq!(reached, [true; 2], "both rows carry a real gradient");
    }

    /// End to end through the trainer's checkpoint pieces: a loss on the velocity reaches the
    /// shared modulation rows through `train_block`'s modulation input — the sampled row **and**
    /// the `t = 0` row — and the `norm_out` rows through `train_head`, and those gradients are the
    /// function's: each matches a central finite difference along a fixed direction.
    #[test]
    fn the_trainer_pieces_backpropagate_through_the_selected_rows() {
        use candle_core::Var;

        let dit = tiny_dit();
        let (_, prelude) = edit_prelude(&dit);
        let velocity = |modulation: &Tensor, out_rows: &Tensor| -> Tensor {
            let mut x = prelude.x.clone();
            for index in 0..dit.num_blocks() {
                x = dit
                    .train_block(index, &x, modulation, &prelude.geometry)
                    .unwrap();
            }
            dit.train_head(&x, out_rows, &prelude.geometry).unwrap()
        };
        let probe = |v: &Tensor| -> Tensor {
            let n = v.elem_count();
            let w: Vec<f32> = (0..n)
                .map(|i| (((i * 7) % 23) as f32 / 23.0) - 0.45)
                .collect();
            (v * Tensor::from_vec(w, v.dims(), &Device::Cpu).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
        };
        let direction = |like: &Tensor, seed: usize| -> Tensor {
            let n = like.elem_count();
            let d: Vec<f32> = (0..n)
                .map(|i| (((i * 29 + seed) % 53) as f32 / 53.0) - 0.5)
                .collect();
            Tensor::from_vec(d, like.dims(), &Device::Cpu).unwrap()
        };
        let modulation = Var::from_tensor(&prelude.modulation).unwrap();
        let out_rows = Var::from_tensor(&prelude.out_rows).unwrap();
        let loss = probe(&velocity(modulation.as_tensor(), out_rows.as_tensor()));
        let grads = loss.backward().unwrap();
        let scalar = |t: &Tensor| t.to_scalar::<f32>().unwrap();
        let abs_sum = |t: &Tensor| scalar(&t.abs().unwrap().sum_all().unwrap());

        let g_mod = grads
            .get(modulation.as_tensor())
            .expect("the modulation rows must receive a gradient");
        for (row, name) in [(0usize, "sampled"), (1, "t = 0")] {
            let mass = abs_sum(&g_mod.narrow(0, row, 1).unwrap());
            assert!(
                mass > 1e-3,
                "the {name} modulation row received no gradient ({mass:e})"
            );
        }
        let g_out = grads
            .get(out_rows.as_tensor())
            .expect("the norm_out rows must receive a gradient");
        assert!(abs_sum(&g_out.narrow(0, 0, 1).unwrap()) > 1e-3);

        let eps = 1e-2f64;
        for (name, grad, base, seed) in [
            ("modulation", g_mod, &prelude.modulation, 3usize),
            ("norm_out", g_out, &prelude.out_rows, 5),
        ] {
            let d = direction(base, seed);
            let analytic = scalar(&(grad * &d).unwrap().sum_all().unwrap());
            let at = |sign: f64| -> f32 {
                let moved = (base + (&d * (sign * eps)).unwrap()).unwrap();
                let v = if name == "modulation" {
                    velocity(&moved, &prelude.out_rows)
                } else {
                    velocity(&prelude.modulation, &moved)
                };
                scalar(&probe(&v))
            };
            let numeric = (at(1.0) - at(-1.0)) / (2.0 * eps as f32);
            let rel = (analytic - numeric).abs() / numeric.abs().max(1e-6);
            eprintln!(
                "{name}: analytic {analytic:.6e} finite-difference {numeric:.6e} rel {rel:.2e}"
            );
            assert!(
                numeric.abs() > 1e-3,
                "{name}: the direction must move the loss"
            );
            assert!(
                rel <= 1e-3,
                "{name}: gradient {analytic:e} vs finite difference {numeric:e} ({rel:.2e})"
            );
        }
    }

    /// The modulation a cached step's target tokens receive is the sampled-timestep row **bit for
    /// bit** — upstream's `torch.where`, the MLX twin's `broadcast_to`. The rows here are chosen so
    /// the prefill's blend `zero + (real − zero)·1` visibly rounds in f32 (a large `t = 0` entry
    /// absorbs a small sampled one), which is the error an all-ones mask would put on every target
    /// token; in the released checkpoint's bf16 the same blend is off by percents.
    #[test]
    fn a_cached_step_reads_the_sampled_modulation_row_exactly() {
        let dev = Device::Cpu;
        let (inner, tokens, slots) = (4usize, 3usize, 4usize);
        // Row 0 = sampled timestep, row 1 = `t = 0`.
        let real: Vec<f32> = (0..slots * inner).map(|i| 0.1 + i as f32 * 0.013).collect();
        let zero: Vec<f32> = (0..slots * inner)
            .map(|i| if i % 2 == 0 { 1.0e9 } else { -3.7e4 })
            .collect();
        let rows =
            Tensor::from_vec([real.clone(), zero].concat(), (2, slots * inner), &dev).unwrap();
        let table = Tensor::zeros((tokens, 2), DType::F32, &dev).unwrap();
        let cache = PrefixKvCache {
            layers: Vec::new(),
            target_cos: table.clone(),
            target_sin: table,
            prefix_len: 5,
            target_tokens: tokens,
        };
        let geometry = cache.target_geometry(inner);
        assert_eq!(geometry.seq_len(), tokens);
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for slot in 0..slots {
            let got = geometry.select(&rows, slot).unwrap();
            assert_eq!(got.dims(), &[1, tokens, inner]);
            let want: Vec<u32> = (0..tokens)
                .flat_map(|_| real[slot * inner..(slot + 1) * inner].iter())
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                bits(&got),
                want,
                "slot {slot}: every target token must read the sampled row unchanged"
            );
        }
        let m = geometry.modulation(&rows).unwrap();
        assert_eq!(
            bits(&m.gate2.narrow(1, 0, 1).unwrap()),
            real[3 * inner..]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// The cache holds its entries in the tensor's own dtype — the compute dtype, bf16 on a GPU
    /// backend — not an f32 upcast, and prepending one to a bf16 target K/V stays bf16. (candle's
    /// CPU backend has no bf16 matmul, so no forward here can run in bf16; the copy and the `cat`
    /// can.)
    #[test]
    fn a_cache_entry_keeps_the_compute_dtype() {
        let dev = Device::Cpu;
        let fill = |tokens: usize| -> Tensor {
            let v: Vec<f32> = (0..2 * tokens * 8).map(|i| i as f32 * 0.25 - 3.0).collect();
            Tensor::from_vec(v, (1, 2, tokens, 8), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let full = fill(6);
        let owned = owned_prefix_rows(&full, 2).unwrap();
        assert_eq!(owned.dtype(), DType::BF16);
        assert_eq!(owned.dims(), &[1, 2, 2, 8]);
        assert_eq!(
            owned
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            full.narrow(2, 0, 2)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let joined = Tensor::cat(&[&owned, &fill(4)], 2).unwrap();
        assert_eq!(joined.dtype(), DType::BF16);
        assert_eq!(joined.dims(), &[1, 2, 6, 8]);
    }

    /// A multi-image layout emits one DISTINCT trace key per image block.
    ///
    /// The edit path appends condition-image segments; keying them all `img_in` would give a
    /// duplicate-keyed trace, and every consumer that counts or looks up stages by name — starting
    /// with `transformer_parity.rs`'s stage-count assertion — would quietly stop meaning anything
    /// rather than fail.
    #[test]
    fn every_image_segment_gets_its_own_trace_key() {
        let shipped = include_str!("transformer.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("the file has a shipped half");
        assert!(
            shipped.contains("trace.push(format!(\"img_in_{image_cursor}\"), &projected)?;"),
            "the image projection must be traced under a per-segment key"
        );
        assert!(
            !shipped.contains("trace.push(\"img_in\""),
            "no image segment may push the un-indexed `img_in` key"
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
