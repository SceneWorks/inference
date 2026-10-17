//! Leaf + stage building blocks of the Qwen-Image causal-Conv3d VAE, ported 1:1 from the frozen
//! fork (`~/repos/mflux/src/mflux/models/qwen/model/qwen_vae/`). Tensors are kept **NCTHW**
//! (channels-first) throughout — mirroring the fork — and transposed to channels-last only inside
//! the conv ops, since mlx convs are channels-last.
//!
//! Two notes carried from the fork:
//!  - The VAE "RMSNorm" is actually a **channel-L2 normalization** (`x / max(‖x‖₂ over C, eps) · √C
//!    · weight`), not feature-RMS. See [`rms_norm_channels`].
//!  - `Resample3d` constructs a temporal `time_conv` that the fork **never calls** in `forward`;
//!    for T2I (T=1) up/down-sampling is purely spatial. We don't port the unused `time_conv`.

use mlx_rs::ops::{add, divide, maximum, multiply, pad, split, sum_axes};
use mlx_rs::Array;

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::ios_quantum::Scores;
use mlx_gen::nn::{conv2d, conv3d, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// VAE channel-L2 norm eps (fork `QwenImageRMSNorm` default). Single source of truth — the parent
/// `vae` module imports this rather than redefining it.
pub(super) const NORM_EPS: f32 = 1e-12;

/// iOS decode-quantum labels (see [`mlx_gen::ios_quantum`]). Stable identifiers: a capture reads
/// them off a typed refusal.
pub const DECODE_RESNET_QUANTUM: &str = "Qwen VAE decode resnet";
pub const DECODE_ATTENTION_QUANTUM: &str = "Qwen VAE decode attention";
pub const DECODE_UPSAMPLE_QUANTUM: &str = "Qwen VAE decode upsample";

/// Scores per query chunk of the mid-block attention on a physical iOS device (16 Mi f32 = 64 MiB):
/// a 1024² decode's 128x128 latent scores 1024 query rows over its 16 384 keys at a time instead of
/// the whole 1 GiB `H·W x H·W` matrix; a 512² latent (64 x 64) fits in one chunk.
pub const DECODE_ATTENTION_SCORE_ELEMENTS: u64 = 16 * 1024 * 1024;

/// One decoder sub-stage as an iOS execution quantum priced at `scratch` — the stage's own
/// counted bytes (see each stage's `quantum_bytes`). Checked BEFORE `build` constructs the graph; a
/// pass-through everywhere except a physical iOS device or the CPU diagnostic fixture.
pub(super) fn decode_quantum(
    label: &str,
    scratch: u64,
    build: impl FnOnce() -> Result<Array>,
) -> Result<Array> {
    mlx_gen::ios_quantum::run(label, scratch, build)
}

/// `[B, C, T, H, W]` of an NCTHW tensor as `u64`s (negative dims read as 0).
pub(super) fn ncthw(x: &Array) -> [u64; 5] {
    let sh = x.shape();
    let d = |i: usize| sh.get(i).copied().unwrap_or(0).max(0) as u64;
    [d(0), d(1), d(2), d(3), d(4)]
}

/// Bytes per element of an activation.
pub(super) fn element_bytes(x: &Array) -> u64 {
    x.item_size() as u64
}

/// [`rms_norm_channels`] over `[B, C, T, H, W]`: the squared input and the normalized, scaled,
/// weighted output (2 full-width buffers; the scale and weight multiplies are donated in place) plus
/// the per-position sum, root and clamp (3 single-channel planes).
pub(super) fn channel_norm_bytes([b, c, t, h, w]: [u64; 5], eb: u64) -> u64 {
    let tb = mlx_gen::ios_quantum::tensor_bytes;
    (2 * tb(eb, &[b, c, t, h, w])).saturating_add(3 * tb(eb, &[b, 1, t, h, w]))
}

/// Channel-L2 normalization over axis 1: `x / max(‖x‖₂ over C, eps) · √C · weight`.
/// `weight` is 1-D `(C,)`; `x` is any rank with the channel axis at index 1 (NCTHW or NCHW).
pub fn rms_norm_channels(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let nd = shape.len();
    let c = shape[1];
    let sum_sq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let l2 = sum_sq.sqrt()?;
    let denom = maximum(&l2, Array::from_slice(&[eps], &[1]))?;
    let normed = divide(x, &denom)?;
    let scale = (c as f32).sqrt();
    let mut wshape = vec![1i32; nd];
    wshape[1] = c;
    let wt = weight.reshape(&wshape)?;
    let scaled = multiply(&normed, Array::from_slice(&[scale], &[1]))?;
    Ok(multiply(&scaled, &wt)?)
}

/// Causal 3-D conv: pad time on the **left only** (`2·pad_t, 0`) + symmetric H/W, then a valid
/// (padding-0) conv3d. NCTHW I/O. Weight is the fork's already-transposed MLX `[out,kD,kH,kW,in]`.
pub struct CausalConv3d {
    w: Array,
    b: Array,
    padding: i32,
}

impl CausalConv3d {
    pub fn from_weights(w: &Weights, prefix: &str, padding: i32) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.conv3d.weight"))?.clone(),
            b: w.require(&format!("{prefix}.conv3d.bias"))?.clone(),
            padding,
        })
    }

    /// Output channels of the loaded kernel (`[out, kD, kH, kW, in]`).
    pub(super) fn out_channels(&self) -> i32 {
        self.w.shape()[0]
    }

    /// Bytes [`forward`](Self::forward) allocates over an `[B, Cin, T, H, W]` input: the causally
    /// padded input and its channels-last contiguous copy (2 x `[B, Cin, T + 2p, H + 2p, W + 2p]`;
    /// for an unpadded 1x1x1 conv just the channels-last copy), MLX's explicit-GEMM unfold of every
    /// output position's `kD·kH·kW·Cin` patch (Metal and CPU both run Conv3d as unfold + GEMM), and
    /// the conv output and its bias sum (2 x `[B, Cout, T, H, W]`; the bias add is counted even
    /// where MLX fuses or donates it).
    pub(super) fn quantum_bytes(&self, [b, cin, t, h, w]: [u64; 5], eb: u64) -> u64 {
        let ws = self.w.shape();
        let d = |i: usize| ws.get(i).copied().unwrap_or(0).max(0) as u64;
        let (cout, kvol) = (d(0), d(1) * d(2) * d(3));
        let p = self.padding.max(0) as u64;
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        let staged = if p > 0 {
            2 * tb(eb, &[b, cin, t + 2 * p, h + 2 * p, w + 2 * p])
        } else {
            tb(eb, &[b, cin, t, h, w])
        };
        [
            staged,
            tb(eb, &[b, t, h, w, kvol, cin]),
            2 * tb(eb, &[b, cout, t, h, w]),
        ]
        .iter()
        .fold(0u64, |a, x| a.saturating_add(*x))
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let p = self.padding;
        let x = if p > 0 {
            pad(
                x_ncthw,
                &[(0, 0), (0, 0), (2 * p, 0), (p, p), (p, p)][..],
                None,
                None,
            )?
        } else {
            x_ncthw.clone()
        };
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

/// `norm1 → SiLU → conv1(3×3×3) → norm2 → SiLU → conv2`, residual (1×1×1 skip when channels differ).
pub struct ResBlock3D {
    norm1: Array,
    conv1: CausalConv3d,
    norm2: Array,
    conv2: CausalConv3d,
    skip: Option<CausalConv3d>,
}

impl ResBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let skip = if w
            .get(&format!("{prefix}.skip_conv.conv3d.weight"))
            .is_some()
        {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.skip_conv"),
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1"), 1)?,
            norm2: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2"), 1)?,
            skip,
        })
    }

    /// Bytes [`forward`](Self::forward) allocates over an `[B, Cin, T, H, W]` input, counting
    /// every buffer: the norm and SiLU at the input width, `conv1`, the norm and SiLU at its output
    /// width, `conv2`, the 1x1x1 skip conv when the widths differ, and the residual sum. MLX holds
    /// an op's buffers until its stream catches up, so two convs' unfolds are live together.
    pub(super) fn quantum_bytes(&self, x: [u64; 5], eb: u64) -> u64 {
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        let [b, _, t, h, w] = x;
        let mid = [b, self.conv1.out_channels().max(0) as u64, t, h, w];
        let out = [b, self.conv2.out_channels().max(0) as u64, t, h, w];
        [
            channel_norm_bytes(x, eb),
            tb(eb, &x),
            self.conv1.quantum_bytes(x, eb),
            channel_norm_bytes(mid, eb),
            tb(eb, &mid),
            self.conv2.quantum_bytes(mid, eb),
            self.skip.as_ref().map_or(0, |s| s.quantum_bytes(x, eb)),
            tb(eb, &out),
        ]
        .iter()
        .fold(0u64, |a, v| a.saturating_add(*v))
    }

    /// The output shape of [`forward`](Self::forward) over an `[B, C, T, H, W]` input.
    pub(super) fn out_shape(&self, [b, _, t, h, w]: [u64; 5]) -> [u64; 5] {
        [b, self.conv2.out_channels().max(0) as u64, t, h, w]
    }

    /// [`forward`](Self::forward) as one decode quantum.
    fn forward_decode(&self, x: &Array) -> Result<Array> {
        decode_quantum(
            DECODE_RESNET_QUANTUM,
            self.quantum_bytes(ncthw(x), element_bytes(x)),
            || self.forward(x),
        )
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = rms_norm_channels(x, &self.norm1, NORM_EPS)?;
        let h = self.conv1.forward(&silu(&h)?)?;
        let h = rms_norm_channels(&h, &self.norm2, NORM_EPS)?;
        let h = self.conv2.forward(&silu(&h)?)?;
        let residual = match &self.skip {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        Ok(add(&h, &residual)?)
    }
}

/// Per-frame spatial self-attention (single head, head_dim = C). NCTHW I/O. The fork builds Q/K/V
/// via a 1×1 Conv2d (`to_qkv`) over channels, attends over the H·W tokens, then projects.
pub struct AttentionBlock3D {
    norm: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
}

impl AttentionBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            qkv_w: w.require(&format!("{prefix}.to_qkv.weight"))?.clone(),
            qkv_b: w.require(&format!("{prefix}.to_qkv.bias"))?.clone(),
            proj_w: w.require(&format!("{prefix}.proj.weight"))?.clone(),
            proj_b: w.require(&format!("{prefix}.proj.bias"))?.clone(),
        })
    }

    /// The query rows one chunk of THIS attention scores under `plan` over an `[B, C, T, H, W]`
    /// input (`H·W` when unbounded).
    fn query_rows(&self, x: [u64; 5], plan: AttentionPlan<'_>) -> u64 {
        let [b, _, t, h, w] = x;
        let tokens = (h * w).min(i32::MAX as u64) as i32;
        let frames = (b * t).min(i32::MAX as u64) as i32;
        plan.budget
            .query_block(frames, 1, tokens, tokens)
            .clamp(0, tokens) as u64
    }

    /// The counted geometry of [`forward_bounded`](Self::forward_bounded) over an
    /// `[B, C, T, H, W]` input. Single head with `head_dim = C` (384 in the shipped VAE), which the
    /// fused Metal SDPA kernel does NOT serve (it is specialized for head dims 64/80/128), so MLX
    /// falls back to `matmul -> softmax -> matmul` and MATERIALIZES the `[B·T, 1, rows, H·W]` f32
    /// scores of each query chunk: [`Scores::Materialized`] with 1 live (the softmax is donated
    /// into the logits). Live `[B·T, H·W, C]` buffers, unchunked — one graph, whose buffers MLX
    /// holds until its stream catches up: the channel norm (2) and its per-position statistics (1),
    /// the channels-last copy (1), the qkv conv and its bias sum (2 x 3), contiguous q/k/v (3), the
    /// scaled q (1), the attention output (1), `proj` and its bias sum (2), the NCTHW copy back (1)
    /// and the residual sum (1) = 18. Chunked, every chunk is evaluated before the next is built,
    /// which synchronizes the stream and releases each buffer once its consumers have run: live at
    /// the peak are the qkv conv and its bias sum (6, kept alive by the k/v views every chunk
    /// reads), the per-position statistics (1), the finished output chunks (1) and their
    /// concatenation (1) = 9.
    pub(super) fn quantum_geometry(
        &self,
        x: [u64; 5],
        eb: u64,
        plan: AttentionPlan<'_>,
    ) -> mlx_gen::ios_quantum::AttentionBlockGeometry {
        let [b, c, t, h, w] = x;
        let tokens = h * w;
        let query_rows = self.query_rows(x, plan);
        mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: b * t,
            tokens,
            dim: c,
            heads: 1,
            context_tokens: 0,
            ff_hidden: 0,
            query_rows,
            element_bytes: eb,
            weight_bytes: 0,
            attn_hidden_live: if query_rows < tokens { 9 } else { 18 },
            context_live: 0,
            scores: Scores::Materialized {
                live: 1,
                element_bytes: eb,
            },
            mlp_hidden_live: 0,
            ff_live: 0,
        }
    }

    /// The attention plan a decode runs this block under: query-chunked at
    /// [`DECODE_ATTENTION_SCORE_ELEMENTS`] scores per chunk on a physical iOS device (and the CPU
    /// fixture that stands in for it), where the full-latent `H·W x H·W` scores would otherwise
    /// materialize at once; the historical single call everywhere else, so a Mac/CUDA decode of
    /// Qwen-Image, Krea and Anima is byte-identical to before.
    pub(super) fn decode_plan() -> AttentionPlan<'static> {
        if mlx_gen::ios_quantum::enabled() {
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                DECODE_ATTENTION_SCORE_ELEMENTS,
                true,
            ))
        } else {
            AttentionPlan::UNBOUNDED
        }
    }

    /// [`forward`](Self::forward) as one decode quantum, under [`Self::decode_plan`], priced per
    /// chunk ([`Self::quantum_geometry`]).
    fn forward_decode(&self, x_ncthw: &Array) -> Result<Array> {
        let plan = Self::decode_plan();
        let scratch = mlx_gen::ios_quantum::attention_block_bytes(self.quantum_geometry(
            ncthw(x_ncthw),
            element_bytes(x_ncthw),
            plan,
        ));
        decode_quantum(DECODE_ATTENTION_QUANTUM, scratch, || {
            self.forward_bounded(x_ncthw, plan)
        })
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        self.forward_bounded(x_ncthw, AttentionPlan::UNBOUNDED)
    }

    /// [`forward`](Self::forward) with the attention's query rows split per `plan`: each chunk of
    /// query rows attends over the COMPLETE keys/values (every row's softmax is over all keys and
    /// independent of the other rows), and the chunk outputs are concatenated back. Same math;
    /// [`AttentionPlan::UNBOUNDED`] is exactly the historical single SDPA call.
    pub fn forward_bounded(&self, x_ncthw: &Array, plan: AttentionPlan<'_>) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // NCTHW -> (B·T, C, H, W)
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?;
        // channel-L2 norm over C (axis 1), then to NHWC for the 1×1 convs.
        let normed = rms_norm_channels(&x, &self.norm, NORM_EPS)?.transpose_axes(&[0, 2, 3, 1])?;
        let qkv = conv2d(&normed, &self.qkv_w, Some(&self.qkv_b), 1, 0)?; // (BT,H,W,3C)
        let qkv = qkv.reshape(&[bt, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 2)?; // q,k,v each (BT, H·W, C)
                                        // single head: (BT, 1, H·W, C)
        let q = parts[0].expand_dims(1)?;
        let k = parts[1].expand_dims(1)?;
        let v = parts[2].expand_dims(1)?;
        let scale = (c as f32).powf(-0.5);
        let o = mlx_gen::attention::sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan)?;
        let o = o.reshape(&[bt, h, w, c])?;
        let o = conv2d(&o, &self.proj_w, Some(&self.proj_b), 1, 0)?; // (BT,H,W,C)
                                                                     // back to NCTHW
        let o = o
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, t, c, h, w])?
            .transpose_axes(&[0, 2, 1, 3, 4])?;
        Ok(add(&o, x_ncthw)?)
    }
}

/// Spatial 2× resample (up = nearest-2× + 3×3 conv to C/2; down = pad-(0,1) + stride-2 3×3 conv).
/// The fork's `time_conv` is never invoked, so only `resample_conv` is ported. NCTHW I/O.
pub struct Resample3d {
    conv_w: Array,
    conv_b: Array,
    upsample: bool,
}

impl Resample3d {
    pub fn from_weights(w: &Weights, prefix: &str, upsample: bool) -> Result<Self> {
        Ok(Self {
            conv_w: w
                .require(&format!("{prefix}.resample_conv.weight"))?
                .clone(),
            conv_b: w.require(&format!("{prefix}.resample_conv.bias"))?.clone(),
            upsample,
        })
    }

    /// Bytes [`forward`](Self::forward) allocates over an `[B, C, T, H, W]` input, at the
    /// resampled extent `H', W'` (2x for an upsampler): the channels-last copy of the input (1), the
    /// nearest-2x broadcast copy (`[B·T, H', W', C]`; for a downsampler, its padded copy instead),
    /// MLX's zero-padded conv input and its explicit-GEMM unfold of every output position's
    /// `3·3·C` patch, the conv output and its bias sum (2 x `[B·T, H', W', Cout]`) and the NCTHW
    /// copy back (1). The bias add is counted even where MLX fuses or donates it.
    pub(super) fn quantum_bytes(&self, [b, c, t, h, w]: [u64; 5], eb: u64) -> u64 {
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        let ws = self.conv_w.shape();
        let d = |i: usize| ws.get(i).copied().unwrap_or(0).max(0) as u64;
        let (cout, kh, kw) = (d(0), d(1), d(2));
        let bt = b * t;
        let (staged, conv_in, (oh, ow)) = if self.upsample {
            let (uh, uw) = (2 * h, 2 * w);
            // nearest-2x copy, then the conv's own 1-pixel zero padding.
            (
                tb(eb, &[bt, uh, uw, c]),
                tb(eb, &[bt, uh + 2, uw + 2, c]),
                (uh, uw),
            )
        } else {
            // bottom/right pad by 1, then a valid stride-2 conv.
            let padded = tb(eb, &[bt, h + 1, w + 1, c]);
            (padded, 0, (h.div_ceil(2), w.div_ceil(2)))
        };
        [
            tb(eb, &[bt, h, w, c]),
            staged,
            conv_in,
            tb(eb, &[bt, oh, ow, kh * kw * c]),
            2 * tb(eb, &[bt, oh, ow, cout]),
            tb(eb, &[bt, oh, ow, cout]),
        ]
        .iter()
        .fold(0u64, |a, x| a.saturating_add(*x))
    }

    /// The output shape of [`forward`](Self::forward) over an `[B, C, T, H, W]` input.
    pub(super) fn out_shape(&self, [b, _, t, h, w]: [u64; 5]) -> [u64; 5] {
        let c = self.conv_w.shape()[0].max(0) as u64;
        if self.upsample {
            [b, c, t, 2 * h, 2 * w]
        } else {
            [b, c, t, h.div_ceil(2), w.div_ceil(2)]
        }
    }

    /// [`forward`](Self::forward) as one decode quantum, priced at the resampled extent.
    fn forward_decode(&self, x_ncthw: &Array) -> Result<Array> {
        decode_quantum(
            DECODE_UPSAMPLE_QUANTUM,
            self.quantum_bytes(ncthw(x_ncthw), element_bytes(x_ncthw)),
            || self.forward(x_ncthw),
        )
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = if self.upsample {
            let up = upsample_nearest(&x, 2)?;
            conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?
        } else {
            // pad bottom/right by 1, then valid stride-2 conv (fork's asymmetric downsample).
            let padded = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            conv2d(&padded, &self.conv_w, Some(&self.conv_b), 2, 0)?
        };
        let nsh = x.shape();
        let (nc, nh, nw) = (nsh[3], nsh[1], nsh[2]);
        Ok(x.transpose_axes(&[0, 3, 1, 2])? // (BT, nc, nh, nw)
            .reshape(&[b, t, nc, nh, nw])?
            .transpose_axes(&[0, 2, 1, 3, 4])?) // NCTHW
    }
}

/// `resnet → attention → resnet` bottleneck.
pub struct MidBlock3D {
    resnet0: ResBlock3D,
    attn: AttentionBlock3D,
    resnet1: ResBlock3D,
}

impl MidBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            resnet0: ResBlock3D::from_weights(w, &format!("{prefix}.resnets.0"))?,
            attn: AttentionBlock3D::from_weights(w, &format!("{prefix}.attentions.0"))?,
            resnet1: ResBlock3D::from_weights(w, &format!("{prefix}.resnets.1"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }

    /// The block's first resnet and its attention (the CPU proofs price them directly).
    #[cfg(test)]
    pub(super) fn parts(&self) -> (&ResBlock3D, &AttentionBlock3D) {
        (&self.resnet0, &self.attn)
    }

    /// The counted scratch of each decode quantum [`Self::forward_decode`] runs over an
    /// `[B, C, T, H, W]` input, from shapes alone (no tensor is read), appended to `out`, with the
    /// attention under `plan`. Returns the block's output shape.
    pub(super) fn decode_quanta(
        &self,
        x: [u64; 5],
        eb: u64,
        plan: AttentionPlan<'_>,
        out: &mut Vec<(&'static str, u64)>,
    ) -> [u64; 5] {
        out.push((DECODE_RESNET_QUANTUM, self.resnet0.quantum_bytes(x, eb)));
        let x = self.resnet0.out_shape(x);
        out.push((
            DECODE_ATTENTION_QUANTUM,
            mlx_gen::ios_quantum::attention_block_bytes(self.attn.quantum_geometry(x, eb, plan)),
        ));
        out.push((DECODE_RESNET_QUANTUM, self.resnet1.quantum_bytes(x, eb)));
        self.resnet1.out_shape(x)
    }

    /// The decoder's [`forward`](Self::forward), one iOS quantum per resnet/attention.
    pub(super) fn forward_decode(&self, x: &Array) -> Result<Array> {
        let x = self.resnet0.forward_decode(x)?;
        let x = self.attn.forward_decode(&x)?;
        self.resnet1.forward_decode(&x)
    }
}

/// N resnets followed by an optional spatial downsample.
pub struct DownBlock3D {
    resnets: Vec<ResBlock3D>,
    downsampler: Option<Resample3d>,
}

impl DownBlock3D {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_res_blocks: usize,
        has_downsampler: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            resnets.push(ResBlock3D::from_weights(
                w,
                &format!("{prefix}.resnets.{i}"),
            )?);
        }
        let downsampler = if has_downsampler {
            Some(Resample3d::from_weights(
                w,
                &format!("{prefix}.downsamplers.0"),
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(d) = &self.downsampler {
            x = d.forward(&x)?;
        }
        Ok(x)
    }
}

/// `num_res_blocks + 1` resnets followed by an optional spatial upsample.
pub struct UpBlock3D {
    resnets: Vec<ResBlock3D>,
    upsampler: Option<Resample3d>,
}

impl UpBlock3D {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_res_blocks: usize,
        has_upsampler: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        for i in 0..(num_res_blocks + 1) {
            resnets.push(ResBlock3D::from_weights(
                w,
                &format!("{prefix}.resnets.{i}"),
            )?);
        }
        let upsampler = if has_upsampler {
            Some(Resample3d::from_weights(
                w,
                &format!("{prefix}.upsamplers.0"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(u) = &self.upsampler {
            x = u.forward(&x)?;
        }
        Ok(x)
    }

    /// The block's upsampler (the CPU proofs price it directly).
    #[cfg(test)]
    pub(super) fn upsampler(&self) -> Option<&Resample3d> {
        self.upsampler.as_ref()
    }

    /// The counted scratch of each decode quantum [`Self::forward_decode`] runs over an
    /// `[B, C, T, H, W]` input, from shapes alone, appended to `out`. Returns the output shape.
    pub(super) fn decode_quanta(
        &self,
        mut x: [u64; 5],
        eb: u64,
        out: &mut Vec<(&'static str, u64)>,
    ) -> [u64; 5] {
        for r in &self.resnets {
            out.push((DECODE_RESNET_QUANTUM, r.quantum_bytes(x, eb)));
            x = r.out_shape(x);
        }
        if let Some(u) = &self.upsampler {
            out.push((DECODE_UPSAMPLE_QUANTUM, u.quantum_bytes(x, eb)));
            x = u.out_shape(x);
        }
        x
    }

    /// The decoder's [`forward`](Self::forward), one iOS quantum per resnet/upsampler.
    pub(super) fn forward_decode(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward_decode(&x)?;
        }
        if let Some(u) = &self.upsampler {
            x = u.forward_decode(&x)?;
        }
        Ok(x)
    }
}
