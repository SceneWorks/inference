//! MMAudio's multimodal flow-matching transformer (**MM-DiT**, SD3-derived) + the Euler
//! flow-matching sampler, ported natively onto the workspace's pinned candle revision (sc-13439,
//! epic sc-12833). This is MMAudio's `MMAudio` network (`mmaudio/model/networks.py`) plus its
//! `FlowMatching` Euler loop (`mmaudio/model/flow_matching.py`) reconstructed from the reference
//! source and parity-verified against a PyTorch forward — not guessed. The **small_16k** config is
//! implemented ([`Config::small_16k`]).
//!
//! ## Architecture (reference `networks.py` + `transformer_layers.py`)
//!
//! The network denoises audio latents `(B, N=250, latent_dim=20)` conditioned on three streams —
//! CLIP visual `(B, 64, 1024)`, Synchformer sync `(B, 192, 768)`, CLIP text `(B, 77, 1024)` — at a
//! flow-matching timestep `t`.
//!
//! - **Input projections.** Each modality is projected to `hidden_dim=448`: audio via
//!   `ChannelLastConv1d(k=7) → SELU → ConvMLP(k=7)`; clip via `Linear → ConvMLP(k=3)`; sync via
//!   `ChannelLastConv1d(k=7) → SELU → ConvMLP(k=3)`; text via `Linear → MLP`. (`ConvMLP`/`MLP` are
//!   SwiGLU feed-forwards with the SD3 `2/3`-then-round-to-256 hidden rule → 1280.) The sync stream
//!   gets a per-frame `sync_pos_emb` added over its 8-frame segments, then is **nearest-exact
//!   upsampled** 192→250 to the audio-latent frame rate.
//! - **N₁ = 4 multimodal joint blocks** (`JointBlock`): latent+clip+text run their own pre-norm +
//!   adaLN + QKV, the three streams' Q/K/V are **concatenated along the token axis** and attended
//!   **jointly** (SD3 MM-DiT), then split back and each stream's post-attention residual/FFN is
//!   applied. The **last** joint block is `pre_only` for the non-audio streams: they still supply
//!   keys/values into the joint attention but are then **dropped**.
//! - **N₂ = 8 audio-only blocks** (`SingleBlock`, `MMDitSingleBlock` in the reference): the latent
//!   stream alone, self-attention + ConvMLP, frame-aligned adaLN.
//! - **Threefold conditioning injection.** (1) **Global adaLN** from the Fourier-timestep embedding
//!   (`TimestepEmbedder`) plus avg-pooled projected clip+text features drives the clip/text blocks
//!   and the final layer. (2) **Frame-aligned adaLN**: `extended_c = global_c + sync_f` is a
//!   *token-level* `(B, 250, D)` modulation (broadcast global + per-frame sync) driving the latent
//!   stream and every audio-only block. (3) **Aligned RoPE**: the latent stream uses RoPE built at
//!   its own frame rate; the clip stream's RoPE is **rescaled by `latent_seq_len/clip_seq_len =
//!   250/64 = 3.90625`** (= `31.25/8`) so the 64 visual tokens align to the 250-frame audio latent.
//!   Q/K carry per-head **RMSNorm** before RoPE.
//! - **Final block**: global-adaLN-modulated LayerNorm → `ChannelLastConv1d(448→20, k=7)` → flow.
//!
//! ## Sampler (reference `flow_matching.py`)
//!
//! Deterministic **Euler**, **25 steps**, integrating the learned velocity field from the Gaussian
//! prior (`t=0`) to data (`t=1`) over `linspace(0, 1, 26)`. **CFG 4.5**: at each step the flow is
//! `cfg·v(cond) + (1−cfg)·v(empty)`, where the empty conditions come from the learnable
//! `empty_clip_feat`/`empty_sync_feat`/`empty_string_feat`. Output latents are un-normalized by the
//! checkpoint's `latent_mean`/`latent_std` — the shape (`(B, 250, 20)`) the 16k VAE (sc-13440)
//! decodes.
//!
//! ## Weights + license
//!
//! `hkchengrex/MMAudio` @ [`HUB_REVISION`], file [`WEIGHTS_PATH`]
//! (`weights/mmaudio_small_16k.pth`, ~629 MB — the **network only**, not the 20 GB training
//! checkpoint). License: **CC-BY-NC 4.0** ([`WEIGHT_LICENSE`], sc-13332). Left **UNREGISTERED** this
//! slice — the shipping generator that wires generator→VAE→waveform and registers is sc-12843.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Result as CResult, Tensor, D};
use candle_audio::gen_core::WeightsSource;
use candle_audio::{AudioError, Result};
use candle_nn::ops::{silu, softmax_last_dim};
use candle_nn::{
    conv1d, conv1d_no_bias, linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, Module,
    VarBuilder,
};

// ---------------------------------------------------------------------------------------------
// Configuration (small_16k — reference `networks.py::small_16k` + `sequence_config.CONFIG_16K`)
// ---------------------------------------------------------------------------------------------

/// MM-DiT hyperparameters. Two presets are populated: the shipping **small_16k** (sc-13439) and the
/// 44.1 kHz quality-ceiling **large_44k_v2** (sc-13441). The struct was shaped from the start so 44k
/// presets add without touching call sites — the `v2` flag captures the reference `MMAudio(v2=True)`
/// architecture deltas (SiLU input-proj activations + the hidden-width Fourier timestep embedder).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Audio-latent channel count (VAE latent dim). small_16k = 20, large_44k_v2 = 40.
    pub latent_dim: usize,
    /// CLIP visual/text feature width (1024).
    pub clip_dim: usize,
    /// Synchformer sync feature width (768).
    pub sync_dim: usize,
    /// CLIP text feature width (1024).
    pub text_dim: usize,
    /// Transformer hidden width (`64 * num_heads`; small_16k = 448, large_44k_v2 = 896).
    pub hidden_dim: usize,
    /// Total DiT depth (`depth`; 12 / 21): `depth - fused_depth` joint blocks + `fused_depth` fused.
    pub depth: usize,
    /// Number of audio-only fused blocks (`fused_depth`; 8 / 14).
    pub fused_depth: usize,
    /// Attention heads (small_16k = 7, large_44k_v2 = 14).
    pub num_heads: usize,
    /// Audio-latent sequence length (250 for 8 s @ 16k; 345 for 8 s @ 44k).
    pub latent_seq_len: usize,
    /// CLIP visual token count (64).
    pub clip_seq_len: usize,
    /// Synchformer token count (192 = 24 segments × 8).
    pub sync_seq_len: usize,
    /// CLIP text token count (77).
    pub text_seq_len: usize,
    /// The reference `MMAudio(v2=…)` flag. `true` (large_44k_v2) swaps the input-projection
    /// activations to **SiLU** — audio/sync `Conv → SiLU → ConvMLP`, and clip/text gain an
    /// interposed `SiLU` between their `Linear` and the following ConvMLP/MLP (v1 has none) — and
    /// makes the `TimestepEmbedder` use `frequency_embedding_size = hidden_dim` with
    /// `max_period = 1` (v1 uses 256 / 10000). Everything else is dimension-driven.
    pub v2: bool,
}

impl Config {
    /// The `small_16k` preset — MMAudio's 16 kHz model.
    pub fn small_16k() -> Self {
        let num_heads = 7;
        Self {
            latent_dim: 20,
            clip_dim: 1024,
            sync_dim: 768,
            text_dim: 1024,
            hidden_dim: 64 * num_heads,
            depth: 12,
            fused_depth: 8,
            num_heads,
            latent_seq_len: 250,
            clip_seq_len: 64,
            sync_seq_len: 192,
            text_seq_len: 77,
            v2: false,
        }
    }

    /// The `large_44k_v2` preset — MMAudio's 44.1 kHz quality-ceiling model (1.03B; reference
    /// `networks.py::large_44k_v2`). 14 heads → hidden 896, depth 21 (7 joint + 14 fused), latent
    /// dim 40, latent seq 345 for the trained 8 s window, `v2=True`.
    pub fn large_44k_v2() -> Self {
        let num_heads = 14;
        Self {
            latent_dim: 40,
            clip_dim: 1024,
            sync_dim: 768,
            text_dim: 1024,
            hidden_dim: 64 * num_heads,
            depth: 21,
            fused_depth: 14,
            num_heads,
            latent_seq_len: 345,
            clip_seq_len: 64,
            sync_seq_len: 192,
            text_seq_len: 77,
            v2: true,
        }
    }

    /// Fourier timestep embedding width (`frequency_embedding_size`): v1 = 256, v2 = `hidden_dim`.
    pub fn freq_embed_size(&self) -> usize {
        if self.v2 {
            self.hidden_dim
        } else {
            FREQ_EMBED_SIZE
        }
    }

    /// Timestep embedding max period (`max_period`): v1 = 10000, v2 = 1.
    pub fn timestep_max_period(&self) -> f64 {
        if self.v2 {
            1.0
        } else {
            TIMESTEP_MAX_PERIOD
        }
    }

    /// Per-head dimension (`hidden_dim / num_heads` = 64).
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
    /// Number of multimodal joint blocks (`depth - fused_depth` = 4).
    pub fn num_joint_blocks(&self) -> usize {
        self.depth - self.fused_depth
    }
    /// Number of Synchformer segments (`sync_seq_len / 8` = 24).
    pub fn num_sync_segments(&self) -> usize {
        self.sync_seq_len / SYNC_FRAMES_PER_SEGMENT
    }
    /// Aligned-RoPE rescale for the clip stream (`latent_seq_len / clip_seq_len` = 3.90625).
    pub fn clip_rope_scaling(&self) -> f64 {
        self.latent_seq_len as f64 / self.clip_seq_len as f64
    }
}

/// Synchformer emits 8 feature frames per segment (`sync_pos_emb` second-to-last dim).
pub const SYNC_FRAMES_PER_SEGMENT: usize = 8;
/// SwiGLU feed-forward "multiple_of" rounding (SD3 `MLP`/`ConvMLP`).
const FF_MULTIPLE_OF: usize = 256;
/// MLP expansion ratio applied before the SwiGLU 2/3 reduction.
const MLP_RATIO: usize = 4;
/// RoPE / timestep base period.
const ROPE_THETA: f64 = 10000.0;
/// Fourier timestep embedding width (`frequency_embedding_size`, small_16k = 256).
const FREQ_EMBED_SIZE: usize = 256;
/// Timestep embedding max period (small_16k = 10000; `freq_scale = 10000/max_period = 1`).
const TIMESTEP_MAX_PERIOD: f64 = 10000.0;
/// Affine-free `nn.LayerNorm` epsilon (PyTorch default `1e-5`).
const LN_EPS: f64 = 1e-5;
/// `nn.RMSNorm` epsilon — reference leaves it `None`, so PyTorch uses `finfo(float32).eps`.
const RMS_EPS: f64 = 1.192_092_9e-7;
/// SELU coefficients (`torch.nn.SELU` constants).
const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
const SELU_SCALE: f64 = 1.050_700_987_355_480_5;
/// Classifier-free-guidance strength (MMAudio default 4.5).
pub const CFG_STRENGTH: f64 = 4.5;
/// Euler flow-matching steps (MMAudio default 25).
pub const NUM_STEPS: usize = 25;

/// SwiGLU hidden width: `round_up(2/3 · h4, 256)` (SD3 rule). For `h4 = hidden*4 = 1792` → 1280.
fn ff_hidden(h4: usize) -> usize {
    let hidden = 2 * h4 / 3;
    FF_MULTIPLE_OF * hidden.div_ceil(FF_MULTIPLE_OF)
}

// ---------------------------------------------------------------------------------------------
// Small primitives
// ---------------------------------------------------------------------------------------------

/// Affine-free LayerNorm over the last dim (`elementwise_affine=False`).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> CResult<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps)?.sqrt()?;
    xc.broadcast_div(&denom)
}

/// adaLN modulation: `x * (1 + scale) + shift` with broadcasting over the token axis.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> CResult<Tensor> {
    let scaled = x.broadcast_mul(&(scale + 1.0)?)?;
    scaled.broadcast_add(shift)
}

/// SELU (`scale · ELU(x, alpha)`).
fn selu(x: &Tensor) -> CResult<Tensor> {
    x.elu(SELU_ALPHA)? * SELU_SCALE
}

/// `nn.Conv1d` operating on channel-last `(B, L, C)` tensors (MMAudio `ChannelLastConv1d`).
struct ChannelLastConv1d {
    conv: Conv1d,
}

impl ChannelLastConv1d {
    fn load(
        in_c: usize,
        out_c: usize,
        k: usize,
        pad: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let cfg = Conv1dConfig {
            padding: pad,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = if bias {
            conv1d(in_c, out_c, k, cfg, vb)?
        } else {
            conv1d_no_bias(in_c, out_c, k, cfg, vb)?
        };
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // (B, L, C) -> (B, C, L) -> conv -> (B, C', L) -> (B, L, C')
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = self.conv.forward(&x)?;
        x.transpose(1, 2)?.contiguous()
    }
}

/// SwiGLU feed-forward built from `Linear`s (`MLP`) — `w2(silu(w1(x)) * w3(x))`, no bias.
struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn load(dim: usize, h4: usize, vb: VarBuilder) -> CResult<Self> {
        let h = ff_hidden(h4);
        Ok(Self {
            w1: linear_no_bias(dim, h, vb.pp("w1"))?,
            w2: linear_no_bias(h, dim, vb.pp("w2"))?,
            w3: linear_no_bias(dim, h, vb.pp("w3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let a = silu(&self.w1.forward(x)?)?;
        let b = self.w3.forward(x)?;
        self.w2.forward(&(a * b)?)
    }
}

/// SwiGLU feed-forward built from `ChannelLastConv1d`s (`ConvMLP`) — temporal (k=3 or 7).
struct ConvMlp {
    w1: ChannelLastConv1d,
    w2: ChannelLastConv1d,
    w3: ChannelLastConv1d,
}

impl ConvMlp {
    fn load(dim: usize, h4: usize, k: usize, pad: usize, vb: VarBuilder) -> CResult<Self> {
        let h = ff_hidden(h4);
        Ok(Self {
            w1: ChannelLastConv1d::load(dim, h, k, pad, false, vb.pp("w1"))?,
            w2: ChannelLastConv1d::load(h, dim, k, pad, false, vb.pp("w2"))?,
            w3: ChannelLastConv1d::load(dim, h, k, pad, false, vb.pp("w3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let a = silu(&self.w1.forward(x)?)?;
        let b = self.w3.forward(x)?;
        self.w2.forward(&(a * b)?)
    }
}

/// A linear-or-conv feed-forward (the block FFN is `ConvMLP` for k>1, `MLP` for k=1).
enum FeedForward {
    Conv(ConvMlp),
    Linear(Mlp),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            FeedForward::Conv(m) => m.forward(x),
            FeedForward::Linear(m) => m.forward(x),
        }
    }
}

/// The block's post-attention output projection (`linear1`): conv for k>1, linear for k=1.
enum OutProj {
    Conv(ChannelLastConv1d),
    Linear(Linear),
}

impl OutProj {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            OutProj::Conv(c) => c.forward(x),
            OutProj::Linear(l) => l.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Rotary embeddings (reference `ext/rotary_embeddings.py`)
// ---------------------------------------------------------------------------------------------

/// Precomputed interleaved-pair RoPE (`cos`/`sin` of shape `(seq, head_dim/2)`).
struct Rope {
    cos: Tensor, // (1, 1, seq, head_dim/2)
    sin: Tensor,
}

impl Rope {
    /// Build for `seq` positions over `head_dim` with an optional frequency rescale.
    fn new(seq: usize, head_dim: usize, freq_scaling: f64, device: &Device) -> CResult<Self> {
        let half = head_dim / 2;
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        for pos in 0..seq {
            for i in 0..half {
                let freq = (1.0 / ROPE_THETA.powf((2 * i) as f64 / head_dim as f64)) * freq_scaling;
                let ang = pos as f64 * freq;
                cos[pos * half + i] = ang.cos() as f32;
                sin[pos * half + i] = ang.sin() as f32;
            }
        }
        let cos = Tensor::from_vec(cos, (1, 1, seq, half), device)?;
        let sin = Tensor::from_vec(sin, (1, 1, seq, half), device)?;
        Ok(Self { cos, sin })
    }

    /// Apply to `(B, H, N, head_dim)`, rotating adjacent `(x0, x1)` pairs.
    fn apply(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, h, n, d) = x.dims4()?;
        let half = d / 2;
        let x = x.reshape((b, h, n, half, 2))?;
        let x0 = x.narrow(4, 0, 1)?.squeeze(4)?; // (B,H,N,half)
        let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
        let out0 = (x0.broadcast_mul(&self.cos)? - x1.broadcast_mul(&self.sin)?)?;
        let out1 = (x0.broadcast_mul(&self.sin)? + x1.broadcast_mul(&self.cos)?)?;
        // interleave back: stack on a new last axis then flatten -> (B,H,N,d)
        let stacked = Tensor::stack(&[&out0, &out1], 4)?; // (B,H,N,half,2)
        stacked.reshape((b, h, n, d))
    }
}

// ---------------------------------------------------------------------------------------------
// Attention (reference `transformer_layers.py::SelfAttention`)
// ---------------------------------------------------------------------------------------------

/// Per-stream QKV projection with per-head RMSNorm on Q/K and optional RoPE. Produces
/// `(B, H, N, head_dim)` tensors ready for joint concatenation.
struct Attention {
    qkv: Linear,
    q_norm: Tensor, // (head_dim,) RMSNorm weight
    k_norm: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder) -> CResult<Self> {
        let dim = cfg.hidden_dim;
        let hd = cfg.head_dim();
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            q_norm: vb.get(hd, "q_norm.weight")?,
            k_norm: vb.get(hd, "k_norm.weight")?,
            num_heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    fn rms(&self, x: &Tensor, weight: &Tensor) -> CResult<Tensor> {
        // x: (B,H,N,hd) — normalize over the last dim.
        let x = x.to_dtype(DType::F32)?;
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = (ms + RMS_EPS)?.sqrt()?;
        x.broadcast_div(&denom)?.broadcast_mul(weight)
    }

    /// `x`: `(B, N, D)` → `(q, k, v)` each `(B, H, N, head_dim)`.
    fn pre_attention(&self, x: &Tensor, rope: Option<&Rope>) -> CResult<(Tensor, Tensor, Tensor)> {
        let (b, n, _d) = x.dims3()?;
        let h = self.num_heads;
        let hd = self.head_dim;
        let qkv = self.qkv.forward(x)?; // (B, N, 3D)
                                        // reference `b n (h d j) -> b h n d j` with j=3: layout is [head, dim, {q,k,v}].
        let qkv = qkv.reshape((b, n, h, hd, 3))?;
        let take = |idx: usize| -> CResult<Tensor> {
            qkv.narrow(4, idx, 1)? // (B,N,H,hd,1)
                .squeeze(4)? // (B,N,H,hd)
                .permute([0, 2, 1, 3])? // (B,H,N,hd)
                .contiguous()
        };
        let q = take(0)?;
        let k = take(1)?;
        let v = take(2)?;
        let q = self.rms(&q, &self.q_norm)?;
        let k = self.rms(&k, &self.k_norm)?;
        let (q, k) = match rope {
            Some(r) => (r.apply(&q)?, r.apply(&k)?),
            None => (q, k),
        };
        Ok((q, k, v))
    }
}

/// Scaled-dot-product attention over `(B, H, N, hd)` → `(B, N, H·hd)` (no mask).
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> CResult<Tensor> {
    let (b, h, n, hd) = q.dims4()?;
    let scale = (hd as f64).powf(-0.5);
    let q = (q.contiguous()? * scale)?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let sim = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
    let attn = softmax_last_dim(&sim)?;
    let out = attn.matmul(&v)?; // (B,H,N,hd)
    out.transpose(1, 2)?.contiguous()?.reshape((b, n, h * hd))
}

// ---------------------------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------------------------

/// adaLN modulation head: `SiLU → Linear(dim, k·dim)` producing `k` chunks.
struct AdaLn {
    linear: Linear,
    chunks: usize,
}

impl AdaLn {
    fn load(dim: usize, chunks: usize, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            linear: linear(dim, chunks * dim, vb.pp("1"))?,
            chunks,
        })
    }
    /// `c`: `(B, 1, D)` (global) or `(B, N, D)` (frame-aligned) → `chunks` tensors of the same rank.
    fn forward(&self, c: &Tensor) -> CResult<Vec<Tensor>> {
        let m = self.linear.forward(&silu(c)?)?; // (..., chunks*D)
        let dim = m.dim(D::Minus1)? / self.chunks;
        (0..self.chunks)
            .map(|i| m.narrow(D::Minus1, i * dim, dim))
            .collect()
    }
}

/// A single-stream MM-DiT block (`MMDitSingleBlock`): frame/global-adaLN, self-attention, ConvMLP.
/// When `pre_only`, only the pre-attention Q/K/V is produced (used by the dropped non-audio streams
/// in the last joint block); post-attention is a no-op.
struct SingleBlock {
    attn: Attention,
    adaln: AdaLn,
    pre_only: bool,
    // present only when !pre_only:
    out_proj: Option<OutProj>,
    ffn: Option<FeedForward>,
}

/// Modulation tensors carried from pre- to post-attention.
struct BlockMod {
    gate_msa: Option<Tensor>,
    shift_mlp: Option<Tensor>,
    scale_mlp: Option<Tensor>,
    gate_mlp: Option<Tensor>,
}

impl SingleBlock {
    /// `kernel_size` selects conv (k>1) vs linear (k=1) for `linear1`/`ffn`.
    fn load(cfg: &Config, pre_only: bool, kernel_size: usize, vb: VarBuilder) -> CResult<Self> {
        let dim = cfg.hidden_dim;
        let attn = Attention::load(cfg, vb.pp("attn"))?;
        if pre_only {
            let adaln = AdaLn::load(dim, 2, vb.pp("adaLN_modulation"))?;
            Ok(Self {
                attn,
                adaln,
                pre_only,
                out_proj: None,
                ffn: None,
            })
        } else {
            let adaln = AdaLn::load(dim, 6, vb.pp("adaLN_modulation"))?;
            let h4 = dim * MLP_RATIO;
            let (out_proj, ffn) = if kernel_size == 1 {
                (
                    OutProj::Linear(linear(dim, dim, vb.pp("linear1"))?),
                    FeedForward::Linear(Mlp::load(dim, h4, vb.pp("ffn"))?),
                )
            } else {
                let pad = kernel_size / 2;
                (
                    OutProj::Conv(ChannelLastConv1d::load(
                        dim,
                        dim,
                        kernel_size,
                        pad,
                        true,
                        vb.pp("linear1"),
                    )?),
                    FeedForward::Conv(ConvMlp::load(dim, h4, kernel_size, pad, vb.pp("ffn"))?),
                )
            };
            Ok(Self {
                attn,
                adaln,
                pre_only,
                out_proj: Some(out_proj),
                ffn: Some(ffn),
            })
        }
    }

    /// `x`: `(B, N, D)`; `c`: `(B, 1, D)` or `(B, N, D)`.
    fn pre_attention(
        &self,
        x: &Tensor,
        c: &Tensor,
        rope: Option<&Rope>,
    ) -> CResult<((Tensor, Tensor, Tensor), BlockMod)> {
        let m = self.adaln.forward(c)?;
        if self.pre_only {
            let (shift_msa, scale_msa) = (&m[0], &m[1]);
            let xn = modulate(&layer_norm_no_affine(x, LN_EPS)?, shift_msa, scale_msa)?;
            let qkv = self.attn.pre_attention(&xn, rope)?;
            Ok((
                qkv,
                BlockMod {
                    gate_msa: None,
                    shift_mlp: None,
                    scale_mlp: None,
                    gate_mlp: None,
                },
            ))
        } else {
            let (shift_msa, scale_msa) = (&m[0], &m[1]);
            let xn = modulate(&layer_norm_no_affine(x, LN_EPS)?, shift_msa, scale_msa)?;
            let qkv = self.attn.pre_attention(&xn, rope)?;
            Ok((
                qkv,
                BlockMod {
                    gate_msa: Some(m[2].clone()),
                    shift_mlp: Some(m[3].clone()),
                    scale_mlp: Some(m[4].clone()),
                    gate_mlp: Some(m[5].clone()),
                },
            ))
        }
    }

    /// Apply the post-attention residual + FFN. No-op when `pre_only`.
    fn post_attention(&self, x: &Tensor, attn_out: &Tensor, m: &BlockMod) -> CResult<Tensor> {
        if self.pre_only {
            return Ok(x.clone());
        }
        let out_proj = self.out_proj.as_ref().unwrap();
        let ffn = self.ffn.as_ref().unwrap();
        let gate_msa = m.gate_msa.as_ref().unwrap();
        let x = (x + out_proj.forward(attn_out)?.broadcast_mul(gate_msa)?)?;
        let r = modulate(
            &layer_norm_no_affine(&x, LN_EPS)?,
            m.shift_mlp.as_ref().unwrap(),
            m.scale_mlp.as_ref().unwrap(),
        )?;
        &x + ffn
            .forward(&r)?
            .broadcast_mul(m.gate_mlp.as_ref().unwrap())?
    }

    /// Standalone forward for the audio-only fused blocks.
    fn forward(&self, x: &Tensor, c: &Tensor, rope: &Rope) -> CResult<Tensor> {
        let (qkv, m) = self.pre_attention(x, c, Some(rope))?;
        let attn_out = attention(&qkv.0, &qkv.1, &qkv.2)?;
        self.post_attention(x, &attn_out, &m)
    }
}

/// A multimodal joint block: latent+clip+text attend **jointly** over concatenated tokens.
struct JointBlock {
    latent: SingleBlock,
    clip: SingleBlock,
    text: SingleBlock,
    pre_only: bool,
}

impl JointBlock {
    fn load(cfg: &Config, pre_only: bool, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            latent: SingleBlock::load(cfg, false, 3, vb.pp("latent_block"))?,
            clip: SingleBlock::load(cfg, pre_only, 3, vb.pp("clip_block"))?,
            text: SingleBlock::load(cfg, pre_only, 1, vb.pp("text_block"))?,
            pre_only,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        latent: &Tensor,
        clip_f: &Tensor,
        text_f: &Tensor,
        global_c: &Tensor,
        extended_c: &Tensor,
        latent_rope: &Rope,
        clip_rope: &Rope,
    ) -> CResult<(Tensor, Tensor, Tensor)> {
        let (xq, xmod) = self
            .latent
            .pre_attention(latent, extended_c, Some(latent_rope))?;
        let (cq, cmod) = self.clip.pre_attention(clip_f, global_c, Some(clip_rope))?;
        let (tq, tmod) = self.text.pre_attention(text_f, global_c, None)?;

        let latent_len = latent.dim(1)?;
        let clip_len = clip_f.dim(1)?;

        // concat Q/K/V along the token axis (dim=2 of (B,H,N,hd)).
        let q = Tensor::cat(&[&xq.0, &cq.0, &tq.0], 2)?;
        let k = Tensor::cat(&[&xq.1, &cq.1, &tq.1], 2)?;
        let v = Tensor::cat(&[&xq.2, &cq.2, &tq.2], 2)?;
        let attn_out = attention(&q, &k, &v)?; // (B, Ntot, D)

        let x_attn = attn_out.narrow(1, 0, latent_len)?;
        let latent = self.latent.post_attention(latent, &x_attn, &xmod)?;
        if self.pre_only {
            return Ok((latent, clip_f.clone(), text_f.clone()));
        }
        let c_attn = attn_out.narrow(1, latent_len, clip_len)?;
        let t_attn = attn_out.narrow(1, latent_len + clip_len, text_f.dim(1)?)?;
        let clip_f = self.clip.post_attention(clip_f, &c_attn, &cmod)?;
        let text_f = self.text.post_attention(text_f, &t_attn, &tmod)?;
        Ok((latent, clip_f, text_f))
    }
}

/// Timestep embedder: sinusoidal Fourier features → 2-layer MLP (`TimestepEmbedder`).
struct TimestepEmbedder {
    fc1: Linear,
    fc2: Linear,
    freqs: Tensor, // (FREQ_EMBED_SIZE/2,)
}

impl TimestepEmbedder {
    fn load(cfg: &Config, vb: VarBuilder, device: &Device) -> CResult<Self> {
        let dim = cfg.hidden_dim;
        let freq_embed = cfg.freq_embed_size(); // 256 (v1) / hidden_dim (v2)
        let half = freq_embed / 2;
        // reference: freqs = (10000 / max_period) * (1 / 10000^(arange(0,F,2)/F)).
        let freq_scale = ROPE_THETA / cfg.timestep_max_period(); // 1.0 (v1) / 10000 (v2)
        let mut f = vec![0f32; half];
        for (i, fi) in f.iter_mut().enumerate() {
            *fi = (freq_scale / ROPE_THETA.powf((2 * i) as f64 / freq_embed as f64)) as f32;
        }
        Ok(Self {
            fc1: linear(freq_embed, dim, vb.pp("mlp.0"))?,
            fc2: linear(dim, dim, vb.pp("mlp.2"))?,
            freqs: Tensor::from_vec(f, (half,), device)?,
        })
    }

    /// `t`: `(B,)` → `(B, D)`.
    fn forward(&self, t: &Tensor) -> CResult<Tensor> {
        let b = t.dim(0)?;
        let half = self.freqs.dim(0)?;
        // args = t[:,None] * freqs[None] -> (B, half)
        let args = t
            .reshape((b, 1))?
            .broadcast_mul(&self.freqs.reshape((1, half))?)?;
        let emb = Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)?; // (B, freq_embed_size)
        let x = self.fc1.forward(&emb)?;
        self.fc2.forward(&silu(&x)?)
    }
}

/// The final flow head: global-adaLN LayerNorm → `ChannelLastConv1d(hidden → latent_dim)`.
struct FinalBlock {
    adaln: AdaLn,
    conv: ChannelLastConv1d,
}

impl FinalBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            adaln: AdaLn::load(cfg.hidden_dim, 2, vb.pp("adaLN_modulation"))?,
            conv: ChannelLastConv1d::load(
                cfg.hidden_dim,
                cfg.latent_dim,
                7,
                3,
                true,
                vb.pp("conv"),
            )?,
        })
    }
    fn forward(&self, latent: &Tensor, global_c: &Tensor) -> CResult<Tensor> {
        let m = self.adaln.forward(global_c)?;
        let x = modulate(&layer_norm_no_affine(latent, LN_EPS)?, &m[0], &m[1])?;
        self.conv.forward(&x)
    }
}

// ---------------------------------------------------------------------------------------------
// The MMAudio network + sampler
// ---------------------------------------------------------------------------------------------

/// Cached, latent/timestep-independent conditioning (reference `PreprocessedConditions`).
pub struct Conditions {
    clip_f: Tensor,   // (B, clip_seq_len, D)
    sync_f: Tensor,   // (B, latent_seq_len, D) — upsampled
    text_f: Tensor,   // (B, text_seq_len, D)
    clip_f_c: Tensor, // (B, D)
    text_f_c: Tensor, // (B, D)
}

/// MMAudio's MM-DiT flow-matching generator (small_16k).
pub struct MmAudioDit {
    cfg: Config,
    device: Device,

    audio_in_conv: ChannelLastConv1d,
    audio_in_ffn: ConvMlp,
    clip_in_lin: Linear,
    clip_in_ffn: ConvMlp,
    sync_in_conv: ChannelLastConv1d,
    sync_in_ffn: ConvMlp,
    text_in_lin: Linear,
    text_in_ffn: Mlp,

    clip_cond_proj: Linear,
    text_cond_proj: Linear,
    global_cond_mlp: Mlp,
    sync_pos_emb: Tensor, // (1,1,8,sync_dim)

    t_embed: TimestepEmbedder,
    joint_blocks: Vec<JointBlock>,
    fused_blocks: Vec<SingleBlock>,
    final_block: FinalBlock,

    latent_mean: Tensor, // (1,1,latent_dim)
    latent_std: Tensor,
    empty_clip_feat: Tensor,   // (1, clip_dim)
    empty_sync_feat: Tensor,   // (1, sync_dim)
    empty_string_feat: Tensor, // (text_seq_len, text_dim)

    latent_rope: Rope,
    clip_rope: Rope,
    // nearest-exact upsample indices sync_seq_len -> latent_seq_len
    sync_up_idx: Tensor,
}

impl MmAudioDit {
    /// Load the small_16k network from a `VarBuilder` rooted at the checkpoint's top level.
    pub fn load(cfg: Config, vb: VarBuilder, device: Device) -> CResult<Self> {
        let dim = cfg.hidden_dim;
        let h4 = dim * MLP_RATIO;

        let audio_in_conv =
            ChannelLastConv1d::load(cfg.latent_dim, dim, 7, 3, true, vb.pp("audio_input_proj.0"))?;
        let audio_in_ffn = ConvMlp::load(dim, h4, 7, 3, vb.pp("audio_input_proj.2"))?;
        let clip_in_lin = linear(cfg.clip_dim, dim, vb.pp("clip_input_proj.0"))?;
        // The audio/sync conv streams have an activation between conv (index 0) and ConvMLP
        // (index 2) in BOTH v1 (SELU) and v2 (SiLU), so their ffn is always at `.2`. The clip/text
        // linear streams have NO activation in v1 (ConvMLP/MLP at index 1) but a SiLU at index 1 in
        // v2, which pushes their ffn submodule to index 2 (the reference `nn.Sequential` layout).
        let lin_ffn_idx = if cfg.v2 { 2 } else { 1 };
        let clip_in_ffn = ConvMlp::load(
            dim,
            h4,
            3,
            1,
            vb.pp(format!("clip_input_proj.{lin_ffn_idx}")),
        )?;
        let sync_in_conv =
            ChannelLastConv1d::load(cfg.sync_dim, dim, 7, 3, true, vb.pp("sync_input_proj.0"))?;
        let sync_in_ffn = ConvMlp::load(dim, h4, 3, 1, vb.pp("sync_input_proj.2"))?;
        let text_in_lin = linear(cfg.text_dim, dim, vb.pp("text_input_proj.0"))?;
        let text_in_ffn = Mlp::load(dim, h4, vb.pp(format!("text_input_proj.{lin_ffn_idx}")))?;

        let clip_cond_proj = linear(dim, dim, vb.pp("clip_cond_proj"))?;
        let text_cond_proj = linear(dim, dim, vb.pp("text_cond_proj"))?;
        let global_cond_mlp = Mlp::load(dim, h4, vb.pp("global_cond_mlp"))?;
        let sync_pos_emb = vb.get(
            (1, 1, SYNC_FRAMES_PER_SEGMENT, cfg.sync_dim),
            "sync_pos_emb",
        )?;

        let t_embed = TimestepEmbedder::load(&cfg, vb.pp("t_embed"), &device)?;

        let n_joint = cfg.num_joint_blocks();
        let pre_only_idx = cfg.depth - cfg.fused_depth - 1;
        let mut joint_blocks = Vec::with_capacity(n_joint);
        for i in 0..n_joint {
            joint_blocks.push(JointBlock::load(
                &cfg,
                i == pre_only_idx,
                vb.pp(format!("joint_blocks.{i}")),
            )?);
        }
        let mut fused_blocks = Vec::with_capacity(cfg.fused_depth);
        for i in 0..cfg.fused_depth {
            fused_blocks.push(SingleBlock::load(
                &cfg,
                false,
                3,
                vb.pp(format!("fused_blocks.{i}")),
            )?);
        }
        let final_block = FinalBlock::load(&cfg, vb.pp("final_layer"))?;

        let latent_mean = vb.get((1, 1, cfg.latent_dim), "latent_mean")?;
        let latent_std = vb.get((1, 1, cfg.latent_dim), "latent_std")?;
        let empty_clip_feat = vb.get((1, cfg.clip_dim), "empty_clip_feat")?;
        let empty_sync_feat = vb.get((1, cfg.sync_dim), "empty_sync_feat")?;
        let empty_string_feat = vb.get((cfg.text_seq_len, cfg.text_dim), "empty_string_feat")?;

        let hd = cfg.head_dim();
        let latent_rope = Rope::new(cfg.latent_seq_len, hd, 1.0, &device)?;
        let clip_rope = Rope::new(cfg.clip_seq_len, hd, cfg.clip_rope_scaling(), &device)?;
        let sync_up_idx = nearest_exact_indices(cfg.sync_seq_len, cfg.latent_seq_len, &device)?;

        Ok(Self {
            cfg,
            device,
            audio_in_conv,
            audio_in_ffn,
            clip_in_lin,
            clip_in_ffn,
            sync_in_conv,
            sync_in_ffn,
            text_in_lin,
            text_in_ffn,
            clip_cond_proj,
            text_cond_proj,
            global_cond_mlp,
            sync_pos_emb,
            t_embed,
            joint_blocks,
            fused_blocks,
            final_block,
            latent_mean,
            latent_std,
            empty_clip_feat,
            empty_sync_feat,
            empty_string_feat,
            latent_rope,
            clip_rope,
            sync_up_idx,
        })
    }

    /// The generator config.
    pub fn config(&self) -> &Config {
        &self.cfg
    }
    /// The device the network lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The input-projection activation between the first conv/linear and the following FFN.
    /// v1 (`small_16k`): **SELU** after the conv streams, **nothing** after the linear streams.
    /// v2 (`large_44k_v2`): **SiLU** for every stream (the reference `MMAudio(v2=True)` blocks).
    fn conv_stream_act(&self, x: &Tensor) -> CResult<Tensor> {
        if self.cfg.v2 {
            silu(x)
        } else {
            selu(x)
        }
    }

    fn audio_input_proj(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.conv_stream_act(&self.audio_in_conv.forward(x)?)?;
        self.audio_in_ffn.forward(&x)
    }
    fn clip_input_proj(&self, x: &Tensor) -> CResult<Tensor> {
        // v1: Linear then ConvMLP, no activation between. v2: Linear → SiLU → ConvMLP.
        let x = self.clip_in_lin.forward(x)?;
        let x = if self.cfg.v2 { silu(&x)? } else { x };
        self.clip_in_ffn.forward(&x)
    }
    fn sync_input_proj(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.conv_stream_act(&self.sync_in_conv.forward(x)?)?;
        self.sync_in_ffn.forward(&x)
    }
    fn text_input_proj(&self, x: &Tensor) -> CResult<Tensor> {
        // v1: Linear then MLP, no activation between. v2: Linear → SiLU → MLP.
        let x = self.text_in_lin.forward(x)?;
        let x = if self.cfg.v2 { silu(&x)? } else { x };
        self.text_in_ffn.forward(&x)
    }

    /// Cache the latent/timestep-independent conditioning (reference `preprocess_conditions`).
    pub fn preprocess_conditions(
        &self,
        clip_f: &Tensor,
        sync_f: &Tensor,
        text_f: &Tensor,
    ) -> CResult<Conditions> {
        let bs = clip_f.dim(0)?;
        let segs = self.cfg.num_sync_segments();
        // (B, segs, 8, sync_dim) + sync_pos_emb, then flatten segments -> (B, sync_seq_len, sync_dim)
        let sync_f = sync_f
            .reshape((bs, segs, SYNC_FRAMES_PER_SEGMENT, self.cfg.sync_dim))?
            .broadcast_add(&self.sync_pos_emb)?
            .reshape((bs, self.cfg.sync_seq_len, self.cfg.sync_dim))?;

        let clip_f = self.clip_input_proj(clip_f)?;
        let sync_f = self.sync_input_proj(&sync_f)?;
        let text_f = self.text_input_proj(text_f)?;

        // nearest-exact upsample sync along the token axis: gather (B, latent_seq_len, D).
        let sync_f = sync_f.index_select(&self.sync_up_idx, 1)?;

        let clip_f_c = self.clip_cond_proj.forward(&clip_f.mean(1)?)?;
        let text_f_c = self.text_cond_proj.forward(&text_f.mean(1)?)?;
        Ok(Conditions {
            clip_f,
            sync_f,
            text_f,
            clip_f_c,
            text_f_c,
        })
    }

    /// Re-derive every sequence-length-dependent tensor for a new output duration — the port of the
    /// reference `MMAudio.update_seq_lengths` (`networks.py`), which `demo.py` calls after
    /// `seq_cfg.duration = duration` so a clip shorter (or longer, ≤ the trained 8 s window) than the
    /// 250-latent default is generated at its own length. Rebuilds the latent/clip RoPE tables (the
    /// clip rescale tracks the new `latent/clip` ratio) and the nearest-exact `sync → latent` upsample
    /// indices, and updates the cached [`Config`] sequence lengths that
    /// [`preprocess_conditions`](Self::preprocess_conditions) / [`predict_flow`](Self::predict_flow)
    /// assert against. The weights themselves are sequence-length-independent, so nothing else moves.
    pub fn update_seq_lengths(
        &mut self,
        latent_seq_len: usize,
        clip_seq_len: usize,
        sync_seq_len: usize,
    ) -> CResult<()> {
        let hd = self.cfg.head_dim();
        self.cfg.latent_seq_len = latent_seq_len;
        self.cfg.clip_seq_len = clip_seq_len;
        self.cfg.sync_seq_len = sync_seq_len;
        self.latent_rope = Rope::new(latent_seq_len, hd, 1.0, &self.device)?;
        self.clip_rope = Rope::new(clip_seq_len, hd, self.cfg.clip_rope_scaling(), &self.device)?;
        self.sync_up_idx = nearest_exact_indices(sync_seq_len, latent_seq_len, &self.device)?;
        Ok(())
    }

    /// The empty (CFG-negative) conditions built from a caller-supplied **negative text** feature —
    /// the port of the reference `get_empty_conditions(bs, negative_text_features=…)`. This is the
    /// path `demo.py`/`generate` actually take: the negative branch's text is `encode_text(negative)`
    /// (the default `negative=""` is still a real CLIP encoding), while the visual streams use the
    /// learned `empty_clip_feat`/`empty_sync_feat`. Distinct from [`empty_conditions`](Self::empty_conditions),
    /// whose text is the learned `empty_string_feat` (used only when no negative text is supplied).
    pub fn empty_conditions_with_text(
        &self,
        bs: usize,
        neg_text_f: &Tensor,
    ) -> CResult<Conditions> {
        let clip = self
            .empty_clip_feat
            .reshape((1, 1, self.cfg.clip_dim))?
            .broadcast_as((bs, self.cfg.clip_seq_len, self.cfg.clip_dim))?
            .contiguous()?;
        let sync = self
            .empty_sync_feat
            .reshape((1, 1, self.cfg.sync_dim))?
            .broadcast_as((bs, self.cfg.sync_seq_len, self.cfg.sync_dim))?
            .contiguous()?;
        self.preprocess_conditions(&clip, &sync, neg_text_f)
    }

    /// The empty (CFG-negative) conditions from the learnable empty features, expanded to `bs`.
    pub fn empty_conditions(&self, bs: usize) -> CResult<Conditions> {
        let clip = self
            .empty_clip_feat
            .reshape((1, 1, self.cfg.clip_dim))?
            .broadcast_as((bs, self.cfg.clip_seq_len, self.cfg.clip_dim))?
            .contiguous()?;
        let sync = self
            .empty_sync_feat
            .reshape((1, 1, self.cfg.sync_dim))?
            .broadcast_as((bs, self.cfg.sync_seq_len, self.cfg.sync_dim))?
            .contiguous()?;
        let text = self
            .empty_string_feat
            .reshape((1, self.cfg.text_seq_len, self.cfg.text_dim))?
            .broadcast_as((bs, self.cfg.text_seq_len, self.cfg.text_dim))?
            .contiguous()?;
        self.preprocess_conditions(&clip, &sync, &text)
    }

    /// Predict the flow (velocity) for `latent` `(B, N, latent_dim)` at timestep `t` `(B,)`.
    pub fn predict_flow(&self, latent: &Tensor, t: &Tensor, c: &Conditions) -> CResult<Tensor> {
        let latent = self.audio_input_proj(latent)?; // (B, N, D)
        let global_c = self
            .global_cond_mlp
            .forward(&(&c.clip_f_c + &c.text_f_c)?)?; // (B, D)
        let global_c = (self.t_embed.forward(t)?.unsqueeze(1)? + global_c.unsqueeze(1)?)?; // (B,1,D)
        let extended_c = global_c.broadcast_add(&c.sync_f)?; // (B, N, D)

        let mut latent = latent;
        let mut clip_f = c.clip_f.clone();
        let mut text_f = c.text_f.clone();
        for block in &self.joint_blocks {
            let (l, cf, tf) = block.forward(
                &latent,
                &clip_f,
                &text_f,
                &global_c,
                &extended_c,
                &self.latent_rope,
                &self.clip_rope,
            )?;
            latent = l;
            clip_f = cf;
            text_f = tf;
        }
        for block in &self.fused_blocks {
            latent = block.forward(&latent, &extended_c, &self.latent_rope)?;
        }
        self.final_block.forward(&latent, &global_c)
    }

    /// One CFG-combined flow at a scalar timestep: `cfg·v(cond) + (1−cfg)·v(empty)`.
    fn cfg_flow(
        &self,
        latent: &Tensor,
        t: f64,
        cond: &Conditions,
        empty: &Conditions,
        cfg: f64,
    ) -> CResult<Tensor> {
        let bs = latent.dim(0)?;
        let t = Tensor::full(t as f32, (bs,), &self.device)?;
        if cfg < 1.0 {
            return self.predict_flow(latent, &t, cond);
        }
        let vc = self.predict_flow(latent, &t, cond)?;
        let ve = self.predict_flow(latent, &t, empty)?;
        (vc * cfg)? + (ve * (1.0 - cfg))?
    }

    /// Un-normalize latents by the checkpoint's `latent_mean`/`latent_std` (the VAE input scale).
    pub fn unnormalize(&self, x: &Tensor) -> CResult<Tensor> {
        x.broadcast_mul(&self.latent_std)?
            .broadcast_add(&self.latent_mean)
    }

    /// Full deterministic Euler flow-matching sample from a fixed prior `x0` `(B, N, latent_dim)`.
    /// Returns the **un-normalized** latents the 16k VAE decodes (`(B, latent_seq_len, latent_dim)`).
    pub fn sample(
        &self,
        x0: &Tensor,
        cond: &Conditions,
        cfg: f64,
        steps: usize,
    ) -> CResult<Tensor> {
        let mut x = x0.clone();
        let empty = self.empty_conditions(x0.dim(0)?)?;
        // steps: linspace(0, 1, steps+1)
        for i in 0..steps {
            let t = i as f64 / steps as f64;
            let next = (i + 1) as f64 / steps as f64;
            let dt = next - t;
            let flow = self.cfg_flow(&x, t, cond, &empty, cfg)?;
            x = (x + (flow * dt)?)?;
        }
        self.unnormalize(&x)
    }

    /// Convenience: the standard 25-step / CFG-4.5 sample.
    pub fn sample_default(&self, x0: &Tensor, cond: &Conditions) -> CResult<Tensor> {
        self.sample(x0, cond, CFG_STRENGTH, NUM_STEPS)
    }
}

/// PyTorch `F.interpolate(mode='nearest-exact')` source indices: `floor((dst+0.5)·in/out)`.
fn nearest_exact_indices(in_len: usize, out_len: usize, device: &Device) -> CResult<Tensor> {
    let mut idx = vec![0u32; out_len];
    for (dst, id) in idx.iter_mut().enumerate() {
        let src = ((dst as f64 + 0.5) * in_len as f64 / out_len as f64).floor() as usize;
        *id = src.min(in_len - 1) as u32;
    }
    Tensor::from_vec(idx, (out_len,), device)
}

// ---------------------------------------------------------------------------------------------
// Weights / hub / license
// ---------------------------------------------------------------------------------------------

/// Stable identity for the weight-license entry (not a shipping provider id — unregistered slice).
pub const MODEL_ID: &str = "mmaudio_small_16k";

/// Hub pin: MMAudio's model repo (immutable commit SHA, F-029 discipline). Same repo as the
/// Synchformer checkpoint; the network weights are `weights/mmaudio_small_16k.pth`.
pub const HUB_REPO: &str = "hkchengrex/MMAudio";
pub const HUB_REVISION: &str = "eb13a1a98fdbec91753775c57b074ccdfc60587c";
/// The small_16k **network** checkpoint (~629 MB) — NOT the 20 GB training checkpoint.
pub const WEIGHTS_PATH: &str = "weights/mmaudio_small_16k.pth";

/// Stable identity for the large_44k_v2 network's weight-license entry (sc-13441). Not a shipping
/// provider id — the shipping generator that registers is [`crate::generator`]'s 44k sibling.
pub const MODEL_ID_44K: &str = "mmaudio_large_44k_v2";
/// The large_44k_v2 **network** checkpoint (~4.12 GB) — the network only, NOT the training checkpoint.
pub const WEIGHTS_PATH_44K: &str = "weights/mmaudio_large_44k_v2.pth";

/// License of the pinned large_44k_v2 network weights (sc-13441) — **CC-BY-NC-4.0**, the same
/// MMAudio checkpoint license as small_16k (all MMAudio HF checkpoints are CC-BY-NC-4.0). Surfaced
/// for the product licenses page; folded into the 44k provider's composite restriction.
pub const WEIGHT_LICENSE_44K: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "CC-BY-NC-4.0",
        name: "Creative Commons Attribution-NonCommercial 4.0 International",
        source_url: "https://huggingface.co/hkchengrex/MMAudio",
        attribution: Some(
            "MMAudio large_44k_v2 network (mmaudio_large_44k_v2.pth) © 2024 Ho Kei Cheng et al. \
             (arXiv:2412.15322); weights distributed via hkchengrex/MMAudio under CC-BY-NC 4.0",
        ),
        commercial_use: false,
        restriction: Some(
            "CC-BY-NC 4.0: non-commercial use only. The MMAudio code is MIT, but the released model \
             weights are NonCommercial; a commercial use needs a separate license from the authors. \
             Trained on VGGSound (research-oriented terms).",
        ),
    };

/// The large_44k_v2 network's weight-license entry (keyed by [`MODEL_ID_44K`]).
pub const WEIGHT_LICENSE_ENTRY_44K: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID_44K,
        component: None,
        license: WEIGHT_LICENSE_44K,
    };

/// License of the pinned MMAudio network weights (sc-13332) — surfaced for the product licenses
/// page. MMAudio's weights are released under **CC-BY-NC 4.0** (non-commercial); the code is MIT.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "CC-BY-NC-4.0",
        name: "Creative Commons Attribution-NonCommercial 4.0 International",
        source_url: "https://github.com/hkchengrex/MMAudio",
        attribution: Some(
            "MMAudio © 2024 Ho Kei Cheng et al. (arXiv:2412.15322); weights distributed via \
             hkchengrex/MMAudio under CC-BY-NC 4.0",
        ),
        commercial_use: false,
        restriction: Some(
            "CC-BY-NC 4.0: non-commercial use only. The MMAudio code is MIT, but the released \
             model weights are NonCommercial; a commercial use needs a separate license from the \
             authors. Trained on VGGSound (research-oriented terms).",
        ),
    };

/// This network's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation once a
/// shipping MMAudio generator registers it (sc-12843).
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: None,
        license: WEIGHT_LICENSE,
    };

/// Load a generator with an explicit [`Config`] from a `.pth` network state dict.
///
/// The `.pth` is the network state dict. The reference deletes the non-persistent
/// `t_embed.freqs`/`latent_rot`/`clip_rot` buffers before loading; we simply never request them
/// (RoPE and the timestep frequencies are recomputed identically), and every other key is consumed.
pub fn load_config_from_pth(cfg: Config, weights: &Path, device: &Device) -> Result<MmAudioDit> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "mmaudio DiT: weights file {} not found",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth(weights, DType::F32, device).map_err(AudioError::from)?;
    MmAudioDit::load(cfg, vb, device.clone()).map_err(AudioError::from)
}

/// Load the small_16k generator from a `mmaudio_small_16k.pth` file path.
pub fn load_from_pth(weights: &Path, device: &Device) -> Result<MmAudioDit> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID}: weights file {} not found (pass {WEIGHTS_PATH} in via the LoadSpec)",
            weights.display()
        )));
    }
    load_config_from_pth(Config::small_16k(), weights, device)
}

/// Load the large_44k_v2 generator from a `mmaudio_large_44k_v2.pth` file path (sc-13441).
pub fn load_large_44k_v2_from_pth(weights: &Path, device: &Device) -> Result<MmAudioDit> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID_44K}: weights file {} not found (pass {WEIGHTS_PATH_44K} in via the LoadSpec)",
            weights.display()
        )));
    }
    load_config_from_pth(Config::large_44k_v2(), weights, device)
}

/// Load from a [`WeightsSource`] (a `File` path to the `.pth`, or a `Dir` containing it under
/// `weights/` or at its root).
pub fn load(source: &WeightsSource, device: &Device) -> Result<MmAudioDit> {
    let path: PathBuf = match source {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => {
            let nested = d.join(WEIGHTS_PATH);
            if nested.exists() {
                nested
            } else {
                d.join("mmaudio_small_16k.pth")
            }
        }
    };
    load_from_pth(&path, device)
}

/// Load the large_44k_v2 generator from a [`WeightsSource`] (a `File` path to the `.pth`, or a `Dir`
/// containing it under `weights/` or at its root) — the 44k twin of [`load`] (sc-13666).
pub fn load_large_44k_v2(source: &WeightsSource, device: &Device) -> Result<MmAudioDit> {
    let path: PathBuf = match source {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => {
            let nested = d.join(WEIGHTS_PATH_44K);
            if nested.exists() {
                nested
            } else {
                d.join("mmaudio_large_44k_v2.pth")
            }
        }
    };
    load_large_44k_v2_from_pth(&path, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_small_16k_matches_reference() {
        let c = Config::small_16k();
        assert_eq!(c.hidden_dim, 448, "64 * 7 heads");
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.num_joint_blocks(), 4, "N1 = depth - fused_depth = 12 - 8");
        assert_eq!(c.fused_depth, 8, "N2 = 8 audio-only blocks");
        assert_eq!(c.num_sync_segments(), 24, "192 / 8");
        assert_eq!(c.latent_seq_len, 250);
        assert_eq!(c.clip_seq_len, 64);
        assert_eq!(c.sync_seq_len, 192);
        assert_eq!(c.text_seq_len, 77);
        assert_eq!(c.latent_dim, 20);
    }

    #[test]
    fn config_large_44k_v2_matches_reference() {
        let c = Config::large_44k_v2();
        assert_eq!(c.hidden_dim, 896, "64 * 14 heads");
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.num_heads, 14);
        assert_eq!(
            c.num_joint_blocks(),
            7,
            "N1 = depth - fused_depth = 21 - 14"
        );
        assert_eq!(c.fused_depth, 14, "N2 = 14 audio-only blocks");
        assert_eq!(c.depth, 21);
        assert_eq!(c.num_sync_segments(), 24, "192 / 8");
        assert_eq!(c.latent_seq_len, 345, "8 s @ 44.1 kHz");
        assert_eq!(c.clip_seq_len, 64);
        assert_eq!(c.sync_seq_len, 192);
        assert_eq!(c.text_seq_len, 77);
        assert_eq!(c.latent_dim, 40);
        assert!(c.v2, "large_44k_v2 uses the v2 architecture");
        // v2 timestep embedder: frequency_embedding_size = hidden_dim, max_period = 1.
        assert_eq!(c.freq_embed_size(), 896);
        assert!((c.timestep_max_period() - 1.0).abs() < 1e-12);
        // v1 leaves the 256 / 10000 defaults.
        let s = Config::small_16k();
        assert_eq!(s.freq_embed_size(), 256);
        assert!((s.timestep_max_period() - 10000.0).abs() < 1e-9);
    }

    #[test]
    fn clip_rope_scaling_is_alignment_factor() {
        let c = Config::small_16k();
        // 250/64 = 3.90625 = 31.25/8 — the visual-stream RoPE alignment to the audio frame rate.
        assert!((c.clip_rope_scaling() - 3.906_25).abs() < 1e-12);
        assert!((c.clip_rope_scaling() - 31.25 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn swiglu_hidden_matches_sd3_rule() {
        // hidden*4 = 1792 -> int(2*1792/3)=1194 -> round up to 256 -> 1280.
        assert_eq!(ff_hidden(448 * 4), 1280);
    }

    #[test]
    fn nearest_exact_indices_match_torch() {
        let dev = Device::Cpu;
        let idx = nearest_exact_indices(192, 250, &dev).unwrap();
        let v: Vec<u32> = idx.to_vec1().unwrap();
        assert_eq!(v.len(), 250);
        // reference F.interpolate('nearest-exact') 192->250, first entries + last.
        assert_eq!(&v[..8], &[0, 1, 1, 2, 3, 4, 4, 5]);
        assert_eq!(v[249], 191);
        assert!(v.iter().all(|&i| i < 192));
    }

    #[test]
    fn pre_only_index_is_last_joint_block() {
        let c = Config::small_16k();
        // reference `pre_only=(i == depth - fused_depth - 1)` -> i == 3 (last of 4 joint blocks).
        assert_eq!(c.depth - c.fused_depth - 1, 3);
    }

    #[test]
    fn modulate_is_affine() {
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 2, 3), DType::F32, &dev).unwrap();
        let shift = Tensor::new(&[[[1f32, 1., 1.]]], &dev).unwrap();
        let scale = Tensor::new(&[[[1f32, 1., 1.]]], &dev).unwrap();
        // x*(1+scale)+shift = 1*2+1 = 3
        let out = modulate(&x, &shift, &scale).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|&e| (e - 3.0).abs() < 1e-6));
    }

    #[test]
    fn rope_is_norm_preserving() {
        // A rotation preserves the L2 norm of each rotated pair.
        let dev = Device::Cpu;
        let rope = Rope::new(5, 64, 1.0, &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 7, 5, 64), &dev).unwrap();
        let y = rope.apply(&x).unwrap();
        let nx: f32 = x.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let ny: f32 = y.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!((nx - ny).abs() / nx < 1e-4, "rope must preserve norm");
    }

    #[test]
    fn weight_license_is_noncommercial_cc_by_nc() {
        assert!(WEIGHT_LICENSE.is_well_formed());
        assert_eq!(WEIGHT_LICENSE.spdx_id, "CC-BY-NC-4.0");
        // Non-commercial: read through the shared helper (and a runtime binding so this stays a
        // runtime assertion, not a const-folded one).
        assert!(
            !WEIGHT_LICENSE.is_permissive(),
            "CC-BY-NC is not permissive"
        );
        let commercial_use = WEIGHT_LICENSE.commercial_use;
        assert!(!commercial_use, "CC-BY-NC-4.0 forbids commercial use");
        assert!(WEIGHT_LICENSE.restriction.is_some());
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, MODEL_ID);
    }

    #[test]
    fn hub_revision_is_a_full_commit_sha() {
        assert_eq!(HUB_REVISION.len(), 40);
        assert!(HUB_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn missing_weights_file_errors_clearly() {
        let dev = Device::Cpu;
        let err = match load_from_pth(Path::new("/nonexistent/mmaudio_small_16k.pth"), &dev) {
            Ok(_) => panic!("loading a nonexistent path must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("not found"));
    }
}
