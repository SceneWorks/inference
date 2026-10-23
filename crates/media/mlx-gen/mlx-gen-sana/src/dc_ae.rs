//! DC-AE (deep compression autoencoder) **decoder** — faithful mlx-rs port of diffusers
//! `AutoencoderDC` (`mit-han-lab/dc-ae-f32c32-sana-1.0`), epic 8485 spike sc-8486.
//!
//! Scope is the decode path only — the spike's GO/NO-GO question is whether the f32 deep-compression
//! decode reproduces cleanly on Metal. The whole decoder runs in **f32** (the checkpoint is f32 and
//! the linear-attention normalizer is f32 in the reference regardless).
//!
//! Layout: everything is **channels-last NHWC** (mlx-native); diffusers' `movedim(1,-1)` before its
//! channel-wise Linear/RMSNorm ops is therefore a no-op here, and the conv weights (PyTorch
//! `[O, I/groups, H, W]`) are transposed to mlx `[O, H, W, I/groups]` at load. The one place that
//! needs the channels-first view is the multi-head reshape inside the linear attention, handled
//! locally by transposing to NCHW for that block of math and back.
//!
//! Block fidelity (vs the exact diffusers source):
//!  - `ResBlock`: `conv1 → SiLU → conv2(no-bias) → RMSNorm(channel) → + residual`.
//!  - `EfficientViTBlock`: `SanaMultiscaleLinearAttention → GLUMBConv`.
//!  - `SanaMultiscaleLinearAttention`: per-pixel `to_q/k/v` (no bias) → multiscale depthwise+grouped
//!    QKV projections → per-head `ReLU(Q),ReLU(K)` linear attention with a `1/(Σ+eps)` normalizer,
//!    computed here without the reference's ones-row `F.pad` via the algebraically identical
//!    numerator/denominator split (same f32 sums) → `to_out`(no bias) → RMSNorm → + residual.
//!  - `GLUMBConv`: `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) → gated SiLU → conv_point
//!    (1×1 no-bias) → RMSNorm → + residual`.
//!  - `DCUpBlock2d` (interpolate): `nearest-upsample → conv`, plus a channel shortcut
//!    `repeat_interleave → pixel_shuffle`.

use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, divide, matmul, mean_axes, multiply, split_sections,
    sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d_general, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::{BlockType, DcAeConfig};
use crate::quantum::labels;

const F32: Dtype = Dtype::Float32;

/// One guarded DC-AE quantum: observe cancellation, then check the counted `scratch`
/// ([`crate::quantum`]) before constructing `build`'s graph.
fn quantum(
    cancel: &CancelFlag,
    label: &str,
    scratch: u64,
    build: impl FnOnce() -> Result<Array>,
) -> Result<Array> {
    // Observe cancellation before constructing every subsequent DC-AE quantum.
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    mlx_gen::ios_quantum::run(label, scratch, build)
}

/// `(batch, height, width, channels)` of an NHWC tensor as `u64`.
fn nhwc(x: &Array) -> (u64, u64, u64, u64) {
    let s = x.shape();
    (s[0] as u64, s[1] as u64, s[2] as u64, s[3] as u64)
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::relu(x)?)
}

/// `PixelShuffle2D(r)` over NHWC: `(N, H, W, C·r²) → (N, H·r, W·r, C)`. (Mirrors the LTX upsampler.)
fn pixel_shuffle_2d(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape();
    let (n, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let out_c = c / (r * r);
    let x = x.reshape(&[n, h, w, out_c, r, r])?;
    let x = x.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
    Ok(x.reshape(&[n, h * r, w * r, out_c])?)
}

/// `repeat_interleave` along the last (channel) axis: each channel duplicated `r` times in place.
fn repeat_interleave_last(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape();
    let c = sh[sh.len() - 1];
    let mut s1 = sh.to_vec();
    s1.push(1);
    let mut s2 = sh.to_vec();
    s2.push(r);
    let mut s3 = sh[..sh.len() - 1].to_vec();
    s3.push(c * r);
    Ok(broadcast_to(&x.reshape(&s1)?, &s2)?.reshape(&s3)?)
}

/// Channel-last RMSNorm over the last axis, computed in f32. `weight`/`bias` are per-channel `[C]`.
fn rms_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let rank = x.shape().len();
    let ax = (rank - 1) as i32;
    let xf = x.as_dtype(F32)?;
    let var = mean_axes(&multiply(&xf, &xf)?, &[ax], true)?;
    let denom = add(&var, scalar(eps))?.sqrt()?;
    let normed = divide(&xf, &denom)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Linear (no bias) over the last axis. `w_t` is the pre-transposed `[in, out]` weight.
fn linear_nb(x: &Array, w_t: &Array) -> Result<Array> {
    let sh = x.shape();
    let inn = sh[sh.len() - 1];
    let out = w_t.shape()[1];
    let n: i32 = sh[..sh.len() - 1].iter().product();
    let y = matmul(&x.reshape(&[n, inn])?, w_t)?;
    let mut outsh: Vec<i32> = sh[..sh.len() - 1].to_vec();
    outsh.push(out);
    Ok(y.reshape(&outsh)?)
}

/// A conv whose on-disk weight is PyTorch `[O, I/groups, H, W]`, transposed to mlx
/// `[O, H, W, I/groups]` at load. Stored f32.
struct Conv {
    w: Array,
    b: Option<Array>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?
            .as_dtype(F32)?;
        let b = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.as_dtype(F32)?)
        } else {
            None
        };
        Ok(Self {
            w: weight,
            b,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        conv2d_general(
            x,
            &self.w,
            self.b.as_ref(),
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )
    }
}

/// `ResBlock` (norm_type=rms_norm, act_fn=silu): `conv1 → SiLU → conv2(no-bias) → RMSNorm → +res`.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
    norm_w: Array,
    norm_b: Array,
    eps: f32,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(w, &format!("{prefix}.conv1"), 1, 1, 1, true)?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), 1, 1, 1, false)?,
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.as_dtype(F32)?,
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.as_dtype(F32)?,
            eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv1.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = rms_norm(&h, &self.norm_w, &self.norm_b, self.eps)?;
        Ok(add(&h, x)?)
    }
}

/// One multiscale QKV projection: depthwise `proj_in` (kernel k, groups=channels) → grouped 1×1
/// `proj_out` (groups = 3·num_heads). Both bias-free. Operates on the NHWC `[B,H,W,3·inner]` qkv.
struct MultiscaleProj {
    proj_in: Conv,
    proj_out: Conv,
}

impl MultiscaleProj {
    fn load(w: &Weights, prefix: &str, kernel: i32, channels: i32, num_heads: i32) -> Result<Self> {
        Ok(Self {
            proj_in: Conv::load(
                w,
                &format!("{prefix}.proj_in"),
                1,
                kernel / 2,
                channels,
                false,
            )?,
            proj_out: Conv::load(w, &format!("{prefix}.proj_out"), 1, 0, 3 * num_heads, false)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out.forward(&self.proj_in.forward(x)?)
    }
}

/// `SanaMultiscaleLinearAttention` (residual). `inner == in_channels` for SANA-1.0 (head_dim 32).
struct LinearAttn {
    to_q: Array, // [in, inner] pre-transposed
    to_k: Array,
    to_v: Array,
    to_out: Array, // [inner·(1+scales), out] pre-transposed
    projs: Vec<MultiscaleProj>,
    norm_w: Array,
    norm_b: Array,
    head_dim: i32,
    num_heads: i32,
    norm_eps: f32,
    attn_eps: f32,
}

impl LinearAttn {
    fn load(w: &Weights, prefix: &str, cfg: &DcAeConfig, channels: i32) -> Result<Self> {
        let head_dim = cfg.attention_head_dim;
        let num_heads = channels / head_dim; // mult=1.0 → inner == channels
        let lin = |name: &str| -> Result<Array> {
            Ok(w.require(&format!("{prefix}.{name}.weight"))?
                .transpose_axes(&[1, 0])?
                .as_dtype(F32)?)
        };
        let mut projs = Vec::new();
        for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
            projs.push(MultiscaleProj::load(
                w,
                &format!("{prefix}.to_qkv_multiscale.{i}"),
                *k,
                3 * channels, // proj operates over the concatenated [q,k,v] = 3·inner channels
                num_heads,
            )?);
        }
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out")?,
            projs,
            norm_w: w
                .require(&format!("{prefix}.norm_out.weight"))?
                .as_dtype(F32)?,
            norm_b: w
                .require(&format!("{prefix}.norm_out.bias"))?
                .as_dtype(F32)?,
            head_dim,
            num_heads,
            norm_eps: cfg.norm_eps,
            attn_eps: cfg.attn_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w) = (sh[0], sh[1], sh[2]);
        let hd = self.head_dim;
        // qkv (NHWC), then concat the multiscale projections along channel.
        let q = linear_nb(x, &self.to_q)?;
        let k = linear_nb(x, &self.to_k)?;
        let v = linear_nb(x, &self.to_v)?;
        let qkv = concatenate_axis(&[q, k, v], -1)?; // [B,H,W,3·inner]
        let mut multi = vec![qkv.clone()];
        for proj in &self.projs {
            multi.push(proj.forward(&qkv)?);
        }
        let hidden = concatenate_axis(&multi, -1)?; // [B,H,W, 3·inner·(1+scales)]

        // → channels-first for the per-head reshape: [B, C_tot, H, W] → [B, heads·(1+scales), 3·hd, HW]
        let hidden = hidden.transpose_axes(&[0, 3, 1, 2])?.as_dtype(F32)?;
        let hw = h * w;
        let groups = self.num_heads * (1 + self.projs.len() as i32);
        let hidden = hidden.reshape(&[b, groups, 3 * hd, hw])?;
        // chunk(3) over the 3·hd axis → q,k,v each [B, groups, hd, HW]
        let parts = split_sections(&hidden, &[hd, 2 * hd], 2)?;
        let q = relu(&parts[0])?;
        let k = relu(&parts[1])?;
        let v = &parts[2];

        // Linear attention with 1/(Σ+eps) normalizer. The reference pads `value` with a ones-row and
        // divides by the last row; computed here as the algebraically identical numerator/denominator
        // split (all f32): num = (V·Kᵀ)·Q ; den = (Σ_hw K)·Q.
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?; // [B,groups,HW,hd]
        let num = matmul(&matmul(v, &k_t)?, &q)?; // [B,groups,hd,HW]
        let k_sum = sum_axes(&k, &[3], true)?.transpose_axes(&[0, 1, 3, 2])?; // [B,groups,1,hd]
        let den = matmul(&k_sum, &q)?; // [B,groups,1,HW]
        let out = divide(&num, &add(&den, scalar(self.attn_eps))?)?; // broadcast over hd

        // → [B, inner·(1+scales), H, W] → NHWC, to_out, RMSNorm, residual.
        let out = out
            .reshape(&[b, groups * hd, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let out = linear_nb(&out, &self.to_out)?;
        let out = rms_norm(&out, &self.norm_w, &self.norm_b, self.norm_eps)?;
        Ok(add(&out, x)?)
    }
}

/// `GLUMBConv` (rms_norm, residual, expand_ratio 4).
struct GluMbConv {
    conv_inverted: Conv, // 1×1, in → 2·hidden
    conv_depth: Conv,    // 3×3 depthwise, 2·hidden → 2·hidden
    conv_point: Conv,    // 1×1 no-bias, hidden → out
    norm_w: Array,
    norm_b: Array,
    hidden: i32,
    eps: f32,
}

impl GluMbConv {
    fn load(w: &Weights, prefix: &str, channels: i32, eps: f32) -> Result<Self> {
        let hidden = 4 * channels;
        Ok(Self {
            conv_inverted: Conv::load(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: Conv::load(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: Conv::load(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.as_dtype(F32)?,
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.as_dtype(F32)?,
            hidden,
            eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv_inverted.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let parts = split_sections(&h, &[self.hidden], 3)?; // chunk(2) over channel (NHWC last axis)
        let h = multiply(&parts[0], &silu(&parts[1])?)?;
        let h = self.conv_point.forward(&h)?;
        let h = rms_norm(&h, &self.norm_w, &self.norm_b, self.eps)?;
        Ok(add(&h, x)?)
    }
}

enum Block {
    Res(ResBlock),
    Evit {
        attn: LinearAttn,
        conv_out: GluMbConv,
    },
}

impl Block {
    /// This block's counted quantum at its NHWC input `x`.
    fn quantum_bytes(&self, x: &Array) -> u64 {
        let (b, h, w, c) = nhwc(x);
        match self {
            Block::Res(_) => crate::quantum::dcae_res_block_bytes(b, h, w, c),
            Block::Evit { attn, .. } => {
                crate::quantum::dcae_evit_block_bytes(b, h, w, c, attn.projs.len() as u64)
            }
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Block::Res(b) => b.forward(x),
            Block::Evit { attn, conv_out } => conv_out.forward(&attn.forward(x)?),
        }
    }
}

/// `DCUpBlock2d` (interpolate path): `nearest-upsample → conv`, + `repeat_interleave → pixel_shuffle`
/// channel shortcut.
struct UpBlock {
    conv: Conv,
    repeats: i32,
}

impl UpBlock {
    /// Counted quantum producing the doubled extent from NHWC `x`.
    fn quantum_bytes(&self, x: &Array) -> u64 {
        let (b, h, w, c) = nhwc(x);
        crate::quantum::dcae_upsample_bytes(b, 2 * h, 2 * w, c, self.conv.w.shape()[0] as u64)
    }

    fn load(w: &Weights, prefix: &str, in_ch: i32, out_ch: i32) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(w, &format!("{prefix}.conv"), 1, 1, 1, true)?,
            repeats: out_ch * 4 / in_ch,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let up = self.conv.forward(&upsample_nearest(x, 2)?)?;
        let shortcut = pixel_shuffle_2d(&repeat_interleave_last(x, self.repeats)?, 2)?;
        Ok(add(&up, &shortcut)?)
    }
}

struct Stage {
    upsample: Option<UpBlock>,
    blocks: Vec<Block>,
}

/// The full DC-AE decoder.
pub struct DcAeDecoder {
    cfg: DcAeConfig,
    conv_in: Conv,
    in_shortcut_repeats: i32,
    stages: Vec<Stage>, // storage order: shallow(0) → deep(n-1); decode iterates deep → shallow
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: Conv,
}

impl DcAeDecoder {
    pub fn from_weights(w: &Weights, cfg: DcAeConfig) -> Result<Self> {
        let n = cfg.num_stages();
        let deepest = cfg.block_out_channels[n - 1];
        let conv_in = Conv::load(w, "decoder.conv_in", 1, 1, 1, true)?;

        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            let ch = cfg.block_out_channels[i];
            // Stages 0..n-1 carry an upsample (storage slot `.0`); the deepest stage does not, so its
            // blocks start at slot `.0`. Block weights live under `decoder.up_blocks.{i}.{slot}`.
            let has_up = i + 1 < n;
            let upsample = if has_up {
                Some(UpBlock::load(
                    w,
                    &format!("decoder.up_blocks.{i}.0"),
                    cfg.block_out_channels[i + 1],
                    ch,
                )?)
            } else {
                None
            };
            let offset = if has_up { 1 } else { 0 };
            let mut blocks = Vec::new();
            for j in 0..cfg.layers_per_block[i] {
                let prefix = format!("decoder.up_blocks.{i}.{}", j + offset);
                let block = match cfg.block_types[i] {
                    BlockType::Res => Block::Res(ResBlock::load(w, &prefix, cfg.norm_eps)?),
                    BlockType::EfficientVit => Block::Evit {
                        attn: LinearAttn::load(w, &format!("{prefix}.attn"), &cfg, ch)?,
                        conv_out: GluMbConv::load(
                            w,
                            &format!("{prefix}.conv_out"),
                            ch,
                            cfg.norm_eps,
                        )?,
                    },
                };
                blocks.push(block);
            }
            stages.push(Stage { upsample, blocks });
        }

        Ok(Self {
            in_shortcut_repeats: deepest / cfg.latent_channels,
            conv_in,
            stages,
            norm_out_w: w.require("decoder.norm_out.weight")?.as_dtype(F32)?,
            norm_out_b: w.require("decoder.norm_out.bias")?.as_dtype(F32)?,
            conv_out: Conv::load(w, "decoder.conv_out", 1, 1, 1, true)?,
            cfg,
        })
    }

    /// Decode a latent `[B, latent_channels, h, w]` (channels-first, diffusers-native; **already
    /// un-scaled** by the caller) into an image `[B, H=32·h, W=32·w, 3]` (channels-last, f32).
    pub fn decode(&self, latent_nchw: &Array, cancel: &CancelFlag) -> Result<Array> {
        // F-028: the DC-AE decode is one monolithic lazy graph after the pipeline's last per-step
        // check. At the ≤1024² envelope (F-032 cap) the window is bounded to seconds, so a cheap
        // entry check — honoring a cancel that arrived during the final denoise step before this
        // decode begins — is the warranted guard; per-tile checks would only be needed if the
        // envelope rises (the graph would then be split into per-tile sub-graphs, each eval'd).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        self.decode_tail_cancellable(&self.decode_head_cancellable(latent_nchw, cancel)?, cancel)
    }

    /// Number of shallow (high-resolution) decoder stages that are attention-free
    /// [`BlockType::Res`] stages — the tileable **tail** (sc-19753).
    ///
    /// The stored stage order is shallow→deep and the SANA-1.0 autoencoder is `[R,R,R,E,E,E]`, so
    /// this counts the leading `Res` run (3). The deep `EfficientVit` stages are the **head**:
    /// their ReLU-linear attention contracts over **every** `H·W` token
    /// (`LinearAttn::forward`'s `matmul(v, kᵀ)` and `sum_axes(&k, &[3])`), so evaluating one on a
    /// crop aggregates over that crop instead of the image — the same defect class as a per-tile
    /// GroupNorm. Robust to any shallow-`Res` / deep-`EfficientVit` split (the DC-AE family
    /// invariant); a config with no leading `Res` run yields 0 and forces a single-pass decode.
    ///
    /// Mirrors `candle_gen_sana::DcAeDecoder::num_tail_stages`, which the candle sibling has had
    /// since sc-11804; the MLX lane is what tiled the whole decoder.
    pub fn num_tail_stages(&self) -> usize {
        self.cfg
            .block_types
            .iter()
            .take_while(|&&t| t == BlockType::Res)
            .count()
    }

    /// The tail's latent→pixel upsample factor (`2^{tail stages that upsample}`) — the tile plan's
    /// spatial scale once the head has already run. For the shipped `[R,R,R,…]` tail this is ×8,
    /// **not** the ×32 of the whole decoder.
    pub fn tail_scale(&self) -> i32 {
        let tail = self.num_tail_stages();
        let ups = self.stages[..tail]
            .iter()
            .filter(|s| s.upsample.is_some())
            .count();
        1i32 << ups
    }

    /// The **head**: `conv_in` + the deepest-stage channel shortcut, then the deep `EfficientVit`
    /// stages down to — but not including — the shallow `Res` tail. Input is the NCHW latent;
    /// output is the tail's NHWC input feature map.
    ///
    /// Must run once on the whole latent: this is where every global-token-sum attention lives.
    pub fn decode_head(&self, latent_nchw: &Array) -> Result<Array> {
        self.decode_head_cancellable(latent_nchw, &CancelFlag::default())
    }

    /// [`Self::decode_head`], checking `cancel` before each bounded quantum.
    pub fn decode_head_cancellable(
        &self,
        latent_nchw: &Array,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let latent = latent_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?; // → NHWC
        let (b, lh, lw, lc) = nhwc(&latent);
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sana",
            "native-decoder",
            &[
                ("schema", 1),
                ("batch", b),
                ("height", lh),
                ("width", lw),
                ("channels", lc),
                ("stages", self.stages.len() as u64),
                ("tail_stages", self.num_tail_stages() as u64),
                ("input_width", self.conv_in.w.shape()[0] as u64),
                (
                    "dtype",
                    mlx_gen::sampler::sampling_dtype_code(latent.dtype()),
                ),
            ],
        );
        let mut h = quantum(
            cancel,
            labels::DECODER_INPUT,
            crate::quantum::dcae_decoder_input_bytes(
                b,
                lh,
                lw,
                lc,
                self.conv_in.w.shape()[0] as u64,
            ),
            || self.input(&latent),
        )?;
        let tail = self.num_tail_stages();
        for stage in self.stages[tail..].iter().rev() {
            h = stage.run_decoder(&h, cancel)?;
        }
        Ok(h)
    }

    /// The **tail**: the shallow `Res` stages (each a ×2 upsample plus `ResBlock`s) then
    /// `norm_out → ReLU → conv_out`. Purely spatially local — convolutions, nearest upsample, and
    /// a `rms_norm` that reduces only the **channel** axis at each position — so evaluating it on a
    /// crop of the head feature map is exact up to convolution padding at the crop boundary, which
    /// the trapezoidal overlap blend absorbs.
    pub fn decode_tail(&self, head_nhwc: &Array) -> Result<Array> {
        self.decode_tail_cancellable(head_nhwc, &CancelFlag::default())
    }

    /// [`Self::decode_tail`], checking `cancel` before each bounded quantum.
    pub fn decode_tail_cancellable(&self, head_nhwc: &Array, cancel: &CancelFlag) -> Result<Array> {
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let mut h = head_nhwc.clone();
        let tail = self.num_tail_stages();
        for stage in self.stages[..tail].iter().rev() {
            h = stage.run_decoder(&h, cancel)?;
        }
        let (b, hh, hw, c) = nhwc(&h);
        quantum(
            cancel,
            labels::DECODER_OUTPUT,
            crate::quantum::dcae_decoder_output_bytes(
                b,
                hh,
                hw,
                c,
                self.conv_out.w.shape()[0] as u64,
            ),
            || self.output(&h),
        )
    }

    /// The head-input quantum's body: `conv_in` plus the channel-repeat shortcut.
    fn input(&self, latent: &Array) -> Result<Array> {
        let shortcut = repeat_interleave_last(latent, self.in_shortcut_repeats)?;
        Ok(add(&self.conv_in.forward(latent)?, &shortcut)?)
    }

    /// The output quantum's body: `norm_out → ReLU → conv_out`.
    fn output(&self, h: &Array) -> Result<Array> {
        let h = rms_norm(h, &self.norm_out_w, &self.norm_out_b, self.cfg.norm_eps)?;
        let h = relu(&h)?;
        self.conv_out.forward(&h)
    }
}

impl Stage {
    /// One decoder stage: its upsample (if any) then its blocks, one guarded quantum each.
    fn run_decoder(&self, x: &Array, cancel: &CancelFlag) -> Result<Array> {
        let mut h = x.clone();
        if let Some(up) = &self.upsample {
            h = quantum(
                cancel,
                labels::DECODER_UPSAMPLE,
                up.quantum_bytes(&h),
                || up.forward(&h),
            )?;
        }
        for block in &self.blocks {
            h = quantum(
                cancel,
                labels::DECODER_BLOCK,
                block.quantum_bytes(&h),
                || block.forward(&h),
            )?;
        }
        Ok(h)
    }
}

// =================================================================================================
// DC-AE ENCODER (sc-10190, img2img) — the mirror of `DcAeDecoder`. Faithful port of the diffusers
// `AutoencoderDC.Encoder` for `dc-ae-f32c32-sana-1.0`. Reuses the decoder's ResBlock / LinearAttn /
// GluMbConv primitives; the net-new pieces are `pixel_unshuffle_2d` + the `DownBlock` (DCDownBlock2d
// in the checkpoint's `downsample_block_type = "Conv"` mode: a stride-2 conv, plus a pixel-unshuffle
// + grouped-channel-mean shortcut). Stages run shallow→deep (128→1024 ch, image→image/32).
// =================================================================================================

/// `PixelUnshuffle2D(r)` over NHWC — the exact inverse of [`pixel_shuffle_2d`]:
/// `(N, H, W, C) → (N, H/r, W/r, C·r²)`. The channel packing (`c·r² + rh·r + rw`) matches PyTorch
/// `F.pixel_unshuffle`, so the DCDownBlock2d shortcut lines up with diffusers bit-for-bit.
fn pixel_unshuffle_2d(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape();
    let (n, big_h, big_w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let (h, w) = (big_h / r, big_w / r);
    let x = x.reshape(&[n, h, r, w, r, c])?; // split H→(h,rh), W→(w,rw)
    let x = x.transpose_axes(&[0, 1, 3, 5, 2, 4])?; // → [n, h, w, c, rh, rw]
    Ok(x.reshape(&[n, h, w, c * r * r])?) // merge → c·r² + rh·r + rw
}

/// Channel-grouped mean over NHWC: split the last axis into `out_channels` contiguous groups and
/// average each (the diffusers `unflatten(1, (-1, group_size)).mean(dim=2)`, channel-last here).
/// `C` must be divisible by `out_channels`.
fn avg_group(x: &Array, out_channels: i32) -> Result<Array> {
    let sh = x.shape();
    let (n, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let group_size = c / out_channels;
    let g = x.reshape(&[n, h, w, out_channels, group_size])?;
    Ok(mean_axes(&g, &[4], false)?)
}

/// `DCDownBlock2d` in **`downsample_block_type = "Conv"`** mode (the real SANA config): a stride-2
/// conv does the 2× spatial downsample AND the channel change, plus a residual shortcut = the
/// pixel-unshuffled input averaged down to `out_channels`. (This is NOT the pixel-unshuffle-conv
/// variant — `downsample=False` in diffusers, so no unshuffle on the conv branch and the conv keeps
/// full `out_channels`.)
struct DownBlock {
    conv: Conv, // in→out, stride 2, pad 1, bias
    out_channels: i32,
}

impl DownBlock {
    /// Counted quantum from NHWC `x` to its half extent.
    fn quantum_bytes(&self, x: &Array) -> u64 {
        let (b, h, w, c) = nhwc(x);
        crate::quantum::dcae_downsample_bytes(b, h, w, c, self.out_channels as u64)
    }

    fn load(w: &Weights, prefix: &str, out_channels: i32) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(w, &format!("{prefix}.conv"), 2, 1, 1, true)?,
            out_channels,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let down = self.conv.forward(x)?; // [n, H/2, W/2, out]
        let shortcut = avg_group(&pixel_unshuffle_2d(x, 2)?, self.out_channels)?;
        Ok(add(&down, &shortcut)?)
    }
}

/// One encoder stage: `encoder_layers_per_block` blocks then an optional trailing downsample (all
/// stages but the deepest carry one). Storage order matches `encoder.down_blocks.{i}.{slot}` — blocks
/// at slots `0..num_layers`, the downsample at slot `num_layers`.
struct EncStage {
    blocks: Vec<Block>,
    downsample: Option<DownBlock>,
}

/// The full DC-AE **encoder**: `conv_in → [stage]×6 → conv_out (+ out_shortcut)`. The mirror of
/// [`DcAeDecoder`]. Runs in f32 (the checkpoint is f32).
pub struct DcAeEncoder {
    conv_in: Conv,
    stages: Vec<EncStage>, // shallow(0) → deep(n-1)
    conv_out: Conv,
    /// The out-shortcut averages `block_out_channels[-1]` down to this many groups (`latent_channels`).
    out_shortcut_groups: i32,
}

impl DcAeEncoder {
    pub fn from_weights(w: &Weights, cfg: DcAeConfig) -> Result<Self> {
        let n = cfg.num_stages();
        // conv_in: 3 → block_out_channels[0], regular 3×3 conv (layers_per_block[0] > 0).
        let conv_in = Conv::load(w, "encoder.conv_in", 1, 1, 1, true)?;

        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            let ch = cfg.block_out_channels[i];
            let num_layers = cfg.encoder_layers_per_block[i];
            let mut blocks = Vec::new();
            for j in 0..num_layers {
                let prefix = format!("encoder.down_blocks.{i}.{j}");
                let block = match cfg.block_types[i] {
                    BlockType::Res => Block::Res(ResBlock::load(w, &prefix, cfg.norm_eps)?),
                    BlockType::EfficientVit => Block::Evit {
                        attn: LinearAttn::load(w, &format!("{prefix}.attn"), &cfg, ch)?,
                        conv_out: GluMbConv::load(
                            w,
                            &format!("{prefix}.conv_out"),
                            ch,
                            cfg.norm_eps,
                        )?,
                    },
                };
                blocks.push(block);
            }
            // Stages 0..n-1 carry a trailing DCDownBlock2d (slot == num_layers), out_channels =
            // block_out_channels[i+1]; the deepest stage has none.
            let downsample = if i + 1 < n {
                Some(DownBlock::load(
                    w,
                    &format!("encoder.down_blocks.{i}.{num_layers}"),
                    cfg.block_out_channels[i + 1],
                )?)
            } else {
                None
            };
            stages.push(EncStage { blocks, downsample });
        }

        let conv_out = Conv::load(w, "encoder.conv_out", 1, 1, 1, true)?;

        Ok(Self {
            out_shortcut_groups: cfg.latent_channels,
            conv_in,
            stages,
            conv_out,
        })
    }

    /// Encode a preprocessed image `[B, 3, H, W]` (channels-first, `[-1, 1]`, the
    /// [`mlx_gen::img2img::preprocess_init_image`] output) into a **raw** DC-AE latent
    /// `[B, latent_channels, H/32, W/32]` (channels-first) — the diffusers `AutoencoderDC.encode`
    /// output, BEFORE the `scaling_factor` multiply (the caller applies it, mirroring the decode
    /// path's `latents / scaling_factor`).
    pub fn encode(&self, image_nchw: &Array, cancel: &CancelFlag) -> Result<Array> {
        // F-028: bounded monolithic graph (see `decode`); a cheap entry check honors the
        // already-cancelled contract at the img2img encode seam.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let x = image_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?; // → NHWC
        let h = self.encode_prefix(&x, cancel)?;
        self.encode_rest(&h, cancel)
    }

    /// Number of shallow encoder stages that are attention-free [`BlockType::Res`] stages — the
    /// spatially-local, tileable **prefix** (the mirror of [`DcAeDecoder::num_tail_stages`]). The
    /// deep `EfficientVit` stages contract their ReLU-linear attention over every token, so they
    /// run once on the whole feature map.
    pub fn num_prefix_stages(&self) -> usize {
        self.stages
            .iter()
            .take_while(|stage| stage.blocks.iter().all(|b| matches!(b, Block::Res(_))))
            .count()
    }

    /// The prefix's image→feature downsample factor (`2^{prefix downsamples}`; ×8 for SANA).
    pub fn prefix_scale(&self) -> i32 {
        let ups = self.stages[..self.num_prefix_stages()]
            .iter()
            .filter(|s| s.downsample.is_some())
            .count();
        1i32 << ups
    }

    /// The prefix's receptive radius in image pixels, rounded UP to a whole prefix cell: `conv_in`
    /// reaches 1 pixel, each `ResBlock`'s two 3x3 convs 2 pixels at its stage's scale, and each
    /// stride-2 3x3 downsample 1 pixel at its input scale. A tile carrying this halo on every side
    /// computes its core's prefix features from exactly the pixels the whole image would.
    pub fn prefix_halo(&self) -> i32 {
        let mut radius = 1i32;
        let mut scale = 1i32;
        for stage in &self.stages[..self.num_prefix_stages()] {
            radius += 2 * stage.blocks.len() as i32 * scale;
            if stage.downsample.is_some() {
                radius += scale;
                scale *= 2;
            }
        }
        let cell = self.prefix_scale();
        (radius + cell - 1) / cell * cell
    }

    /// `conv_in` then the attention-free prefix stages (blocks + downsamples), one guarded quantum
    /// each. NHWC f32 in; NHWC prefix features out.
    fn encode_prefix(&self, x: &Array, cancel: &CancelFlag) -> Result<Array> {
        let (b, xh, xw, xc) = nhwc(x);
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sana",
            "native-encoder",
            &[
                ("schema", 1),
                ("batch", b),
                ("height", xh),
                ("width", xw),
                ("channels", xc),
                ("stages", self.stages.len() as u64),
                ("prefix_stages", self.num_prefix_stages() as u64),
                ("input_width", self.conv_in.w.shape()[0] as u64),
                ("dtype", mlx_gen::sampler::sampling_dtype_code(x.dtype())),
            ],
        );
        let mut h = quantum(
            cancel,
            labels::ENCODER_INPUT,
            crate::quantum::dcae_encoder_input_bytes(
                b,
                xh,
                xw,
                xc,
                self.conv_in.w.shape()[0] as u64,
            ),
            || self.conv_in.forward(x),
        )?;
        for stage in &self.stages[..self.num_prefix_stages()] {
            h = stage.run_encoder(&h, cancel)?;
        }
        Ok(h)
    }

    /// The deep stages and `conv_out` (+ out-shortcut) on the WHOLE prefix feature map.
    fn encode_rest(&self, prefix: &Array, cancel: &CancelFlag) -> Result<Array> {
        let mut h = prefix.clone();
        for stage in &self.stages[self.num_prefix_stages()..] {
            h = stage.run_encoder(&h, cancel)?;
        }
        let (b, hh, hw, c) = nhwc(&h);
        // out_shortcut: conv_out(h) + mean_group(h → latent_channels).
        quantum(
            cancel,
            labels::ENCODER_OUTPUT,
            crate::quantum::dcae_encoder_output_bytes(
                b,
                hh,
                hw,
                c,
                self.out_shortcut_groups as u64,
            ),
            || self.output(&h),
        )
    }

    /// The encoder output quantum's body.
    fn output(&self, h: &Array) -> Result<Array> {
        let shortcut = avg_group(h, self.out_shortcut_groups)?;
        let latent_nhwc = add(&self.conv_out.forward(h)?, &shortcut)?;
        Ok(latent_nhwc.transpose_axes(&[0, 3, 1, 2])?)
    }

    /// **Bounded encode** (img2img on iOS, or a request's bounded-VAE rung): [`Self::encode`] with
    /// the attention-free prefix run per `core`-pixel tile instead of over the whole image.
    ///
    /// Each tile is the core plus [`Self::prefix_halo`] pixels of real image context on every side
    /// (clipped at the image border, where the whole-image pass zero-pads exactly as the tile
    /// does), its prefix features are cropped back to the core's cells, and the cores are stitched
    /// in place — no blending. Core and halo are whole prefix cells, so every stride-2 sampling grid
    /// stays aligned with the whole image's. The per-tile prefix quanta price the tile's own extent;
    /// the stitch is its own guarded quantum; the deep attention stages then run once on the
    /// stitched map exactly as in [`Self::encode`]. `core` is rounded up to a whole prefix cell. A
    /// config with no attention-free prefix, or an image within one core, runs [`Self::encode`].
    pub fn encode_tiled(
        &self,
        image_nchw: &Array,
        core: i32,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let cell = self.prefix_scale();
        let core = ((core.max(1) + cell - 1) / cell) * cell;
        let sh = image_nchw.shape();
        let (height, width) = (sh[2], sh[3]);
        if self.num_prefix_stages() == 0 || (height <= core && width <= core) {
            return self.encode(image_nchw, cancel);
        }
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let halo = self.prefix_halo();
        let x = image_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?; // → NHWC
        let starts = |extent: i32| (0..extent).step_by(core as usize).collect::<Vec<i32>>();
        let mut rows = Vec::new();
        for &y0 in &starts(height) {
            let mut row = Vec::new();
            for &x0 in &starts(width) {
                let (y1, x1) = ((y0 + core).min(height), (x0 + core).min(width));
                let (ty0, tx0) = ((y0 - halo).max(0), (x0 - halo).max(0));
                let (ty1, tx1) = ((y1 + halo).min(height), (x1 + halo).min(width));
                let tile = mlx_gen::attention::slice_axis(
                    &mlx_gen::attention::slice_axis(&x, 1, ty0, ty1 - ty0)?,
                    2,
                    tx0,
                    tx1 - tx0,
                )?;
                let features = self.encode_prefix(&tile, cancel)?;
                let crop = mlx_gen::attention::slice_axis(
                    &mlx_gen::attention::slice_axis(
                        &features,
                        1,
                        (y0 - ty0) / cell,
                        (y1 - y0) / cell,
                    )?,
                    2,
                    (x0 - tx0) / cell,
                    (x1 - x0) / cell,
                )?;
                row.push(crop);
            }
            rows.push(row);
        }
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (b, _, _, c) = nhwc(&rows[0][0]);
        let stitched = quantum(
            cancel,
            labels::ENCODER_STITCH,
            crate::quantum::dcae_encode_stitch_bytes(
                b,
                (height / cell) as u64,
                (width / cell) as u64,
                c,
            ),
            || stitch(&rows),
        )?;
        drop(rows);
        self.encode_rest(&stitched, cancel)
    }
}

/// Stitch row-major core crops into one NHWC map: each row along W, then the rows along H.
fn stitch(rows: &[Vec<Array>]) -> Result<Array> {
    let rows = rows
        .iter()
        .map(|row| Ok(concatenate_axis(&row.iter().collect::<Vec<_>>(), 2)?))
        .collect::<Result<Vec<Array>>>()?;
    Ok(concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1)?)
}

impl EncStage {
    /// One encoder stage: its blocks then its downsample (if any), one guarded quantum each.
    fn run_encoder(&self, x: &Array, cancel: &CancelFlag) -> Result<Array> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = quantum(
                cancel,
                labels::ENCODER_BLOCK,
                block.quantum_bytes(&h),
                || block.forward(&h),
            )?;
        }
        if let Some(down) = &self.downsample {
            h = quantum(
                cancel,
                labels::ENCODER_DOWNSAMPLE,
                down.quantum_bytes(&h),
                || down.forward(&h),
            )?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use mlx_rs::random::{key, normal};
    use mlx_rs::transforms::eval;

    #[test]
    fn pixel_unshuffle_inverts_pixel_shuffle() {
        // Both ops are exact index permutations, so unshuffle→shuffle must return the original
        // bit-for-bit. This pins the channel packing to PyTorch's `pixel_(un)shuffle` convention —
        // the one guarantee the DCDownBlock2d shortcut relies on to match diffusers.
        let x = normal::<f32>(&[2, 6, 8, 12], None, None, Some(&key(11).unwrap())).unwrap();
        let round = pixel_shuffle_2d(&pixel_unshuffle_2d(&x, 2).unwrap(), 2).unwrap();
        eval([&x, &round]).unwrap();
        assert_eq!(x.shape(), round.shape());
        assert_eq!(x.as_slice::<f32>(), round.as_slice::<f32>());
    }

    #[test]
    fn pixel_unshuffle_shape() {
        // (N,H,W,C) → (N,H/2,W/2,C·4).
        let x = normal::<f32>(&[1, 4, 4, 3], None, None, Some(&key(1).unwrap())).unwrap();
        let y = pixel_unshuffle_2d(&x, 2).unwrap();
        eval([&y]).unwrap();
        assert_eq!(y.shape(), &[1, 2, 2, 12]);
    }

    #[test]
    fn avg_group_averages_contiguous_groups() {
        // 8 channels → 2 groups of 4; group g = mean of channels [4g, 4g+4) (the out-shortcut op).
        let x = normal::<f32>(&[1, 1, 1, 8], None, None, Some(&key(3).unwrap())).unwrap();
        let g = avg_group(&x, 2).unwrap();
        eval([&g]).unwrap();
        let xs = x.as_slice::<f32>();
        let gs = g.as_slice::<f32>();
        let m0 = (xs[0] + xs[1] + xs[2] + xs[3]) / 4.0;
        let m1 = (xs[4] + xs[5] + xs[6] + xs[7]) / 4.0;
        assert!((gs[0] - m0).abs() < 1e-5, "group 0: {} vs {m0}", gs[0]);
        assert!((gs[1] - m1).abs() < 1e-5, "group 1: {} vs {m1}", gs[1]);
    }
}

/// Measured-peak proofs of every counted DC-AE quantum (ResBlock, EfficientViT block, up/down
/// sample, decoder head input / output, encoder input / output, and the bounded encode's stitch)
/// on the MLX CPU device with Metal trapped, in a child process: real modules from random f32
/// weights at small shapes where activations dominate weights.
#[cfg(all(test, target_os = "macos"))]
pub(crate) mod quantum_proofs {
    use super::*;
    use crate::quantum::MAX_RATIO;
    use mlx_gen::ios_quantum::{assert_tracks_measured, measured_peak_bytes};
    use std::collections::HashMap;

    pub(crate) fn config() -> DcAeConfig {
        use BlockType::{EfficientVit as E, Res as R};
        DcAeConfig {
            in_channels: 3,
            latent_channels: 4,
            attention_head_dim: 8,
            block_out_channels: vec![16, 32, 32],
            layers_per_block: vec![1, 1, 1],
            encoder_layers_per_block: vec![1, 1, 1],
            block_types: vec![R, R, E],
            qkv_multiscales: vec![5],
            upsample_interpolate: true,
            norm_eps: 1e-5,
            attn_eps: 1e-15,
            scaling_factor: 0.41407,
        }
    }

    struct W(HashMap<String, Array>, u64);
    impl W {
        fn put(&mut self, k: String, shape: &[i32]) {
            self.1 += 1;
            let a = mlx_rs::random::normal::<f32>(
                shape,
                None,
                None,
                Some(&mlx_rs::random::key(self.1).unwrap()),
            )
            .unwrap()
            .multiply(scalar(0.1))
            .unwrap();
            a.eval().unwrap();
            self.0.insert(k, a);
        }
        fn conv(&mut self, p: &str, o: i32, i: i32, k: i32, bias: bool) {
            self.put(format!("{p}.weight"), &[o, i, k, k]);
            if bias {
                self.put(format!("{p}.bias"), &[o]);
            }
        }
        fn affine(&mut self, p: &str, c: i32) {
            self.put(format!("{p}.weight"), &[c]);
            self.put(format!("{p}.bias"), &[c]);
        }
        fn res(&mut self, p: &str, c: i32) {
            self.conv(&format!("{p}.conv1"), c, c, 3, true);
            self.conv(&format!("{p}.conv2"), c, c, 3, false);
            self.affine(&format!("{p}.norm"), c);
        }
        fn evit(&mut self, p: &str, c: i32, cfg: &DcAeConfig) {
            let heads = c / cfg.attention_head_dim;
            for n in ["to_q", "to_k", "to_v"] {
                self.put(format!("{p}.attn.{n}.weight"), &[c, c]);
            }
            let scales = cfg.qkv_multiscales.len() as i32;
            self.put(format!("{p}.attn.to_out.weight"), &[c, c * (1 + scales)]);
            for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
                let m = format!("{p}.attn.to_qkv_multiscale.{i}");
                self.conv(&format!("{m}.proj_in"), 3 * c, 1, *k, false);
                self.conv(
                    &format!("{m}.proj_out"),
                    3 * c,
                    3 * c / (3 * heads),
                    1,
                    false,
                );
            }
            self.affine(&format!("{p}.attn.norm_out"), c);
            let hidden = 4 * c;
            self.conv(
                &format!("{p}.conv_out.conv_inverted"),
                2 * hidden,
                c,
                1,
                true,
            );
            self.conv(&format!("{p}.conv_out.conv_depth"), 2 * hidden, 1, 3, true);
            self.conv(&format!("{p}.conv_out.conv_point"), c, hidden, 1, false);
            self.affine(&format!("{p}.conv_out.norm"), c);
        }
    }

    pub(crate) fn weights(cfg: &DcAeConfig) -> Weights {
        Weights::from_map(weight_map(cfg))
    }

    /// [`weights`]' tensors by name (to write a fixture snapshot).
    pub(crate) fn weight_map(cfg: &DcAeConfig) -> HashMap<String, Array> {
        let mut w = W(HashMap::new(), 4000);
        let n = cfg.num_stages();
        let ch = &cfg.block_out_channels;
        w.conv("decoder.conv_in", ch[n - 1], cfg.latent_channels, 3, true);
        w.conv("encoder.conv_in", ch[0], cfg.in_channels, 3, true);
        for i in 0..n {
            let has_up = i + 1 < n;
            if has_up {
                w.conv(
                    &format!("decoder.up_blocks.{i}.0.conv"),
                    ch[i],
                    ch[i + 1],
                    3,
                    true,
                );
                let layers = cfg.encoder_layers_per_block[i];
                w.conv(
                    &format!("encoder.down_blocks.{i}.{layers}.conv"),
                    ch[i + 1],
                    ch[i],
                    3,
                    true,
                );
            }
            for j in 0..cfg.layers_per_block[i] {
                let p = format!("decoder.up_blocks.{i}.{}", j + i32::from(has_up));
                match cfg.block_types[i] {
                    BlockType::Res => w.res(&p, ch[i]),
                    BlockType::EfficientVit => w.evit(&p, ch[i], cfg),
                }
            }
            for j in 0..cfg.encoder_layers_per_block[i] {
                let p = format!("encoder.down_blocks.{i}.{j}");
                match cfg.block_types[i] {
                    BlockType::Res => w.res(&p, ch[i]),
                    BlockType::EfficientVit => w.evit(&p, ch[i], cfg),
                }
            }
        }
        w.affine("decoder.norm_out", ch[0]);
        w.conv("decoder.conv_out", cfg.in_channels, ch[0], 3, true);
        w.conv("encoder.conv_out", cfg.latent_channels, ch[n - 1], 3, true);
        w.0
    }

    fn rand(shape: &[i32], seed: u64) -> Array {
        let a = mlx_rs::random::normal::<f32>(
            shape,
            None,
            None,
            Some(&mlx_rs::random::key(seed).unwrap()),
        )
        .unwrap();
        a.eval().unwrap();
        a
    }

    fn materialize_decoder(d: &DcAeDecoder) {
        let mut all: Vec<&Array> = vec![&d.conv_in.w, &d.conv_out.w, &d.norm_out_w, &d.norm_out_b];
        for st in &d.stages {
            if let Some(up) = &st.upsample {
                all.push(&up.conv.w);
            }
        }
        mlx_rs::transforms::eval(all).unwrap();
    }

    #[test]
    #[ignore = "expensive fixture capture; run through the selected iOS capture workflow"]
    fn dcae_quanta_track_their_measured_peaks() {
        const CHILD: &str = "SANA_DCAE_QUANTUM_PROOF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "dc_ae::quantum_proofs::dcae_quanta_track_their_measured_peaks",
                    "--include-ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("IOS_CAPTURE_IN_CHILD", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        let cfg = config();
        let w = weights(&cfg);
        let dec = DcAeDecoder::from_weights(&w, cfg.clone()).unwrap();
        let enc = DcAeEncoder::from_weights(&w, cfg.clone()).unwrap();
        materialize_decoder(&dec);
        let mut proofs: Vec<(String, u64, u64)> = Vec::new();
        let mut prove = |label: &str, formula: u64, run: &dyn Fn() -> Result<Array>| {
            // Warm the graph once so lazily-converted weights (f32 casts at load) are resident.
            run().unwrap().eval().unwrap();
            let measured = measured_peak_bytes(|| Ok(vec![run()?])).unwrap();
            println!(
                "[{label}] formula {formula} measured {measured} ratio {:.3}",
                formula as f64 / measured as f64
            );
            proofs.push((label.to_string(), formula, measured));
        };
        // ResBlock (decoder stage 1, 32 ch) at 64x64.
        let x32 = rand(&[1, 64, 64, 32], 1);
        let res = &dec.stages[1].blocks[0];
        prove("DC-AE ResBlock", res.quantum_bytes(&x32), &|| {
            res.forward(&x32)
        });
        // EfficientViT block (stage 2, 32 ch, 4 heads, one 5x5 multiscale branch) at 64x64.
        let evit = &dec.stages[2].blocks[0];
        prove(
            "DC-AE EfficientViT block",
            evit.quantum_bytes(&x32),
            &|| evit.forward(&x32),
        );
        // Upsample 32 -> 16 ch, 32x32 -> 64x64.
        let up = dec.stages[0].upsample.as_ref().unwrap();
        let x_up = rand(&[1, 32, 32, 32], 2);
        prove("DC-AE upsample", up.quantum_bytes(&x_up), &|| {
            up.forward(&x_up)
        });
        // Downsample 16 -> 32 ch, 64x64 -> 32x32.
        let down = enc.stages[0].downsample.as_ref().unwrap();
        let x16 = rand(&[1, 64, 64, 16], 3);
        prove("DC-AE downsample", down.quantum_bytes(&x16), &|| {
            down.forward(&x16)
        });
        // Decoder head input: latent 4 ch -> 32 ch at 32x32.
        // As in `decode_head`, the NHWC f32 latent is formed inside the quantum.
        let latent = rand(&[1, 4, 32, 32], 4);
        prove(
            "DC-AE head input",
            crate::quantum::dcae_decoder_input_bytes(1, 32, 32, 4, 32),
            &|| dec.input(&latent.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?),
        );
        // Decoder output: 16 ch -> RGB at 96x96.
        let x_out = rand(&[1, 96, 96, 16], 5);
        prove(
            "DC-AE decoder output",
            crate::quantum::dcae_decoder_output_bytes(1, 96, 96, 16, 3),
            &|| dec.output(&x_out),
        );
        // Encoder input: RGB -> 16 ch at 96x96.
        // As in `encode`, the NHWC f32 image is formed inside the quantum.
        let image = rand(&[1, 3, 96, 96], 6);
        prove(
            "DC-AE encoder input",
            crate::quantum::dcae_encoder_input_bytes(1, 96, 96, 3, 16),
            &|| {
                enc.conv_in
                    .forward(&image.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?)
            },
        );
        // Encoder output: 32 ch -> 4-ch latent at 32x32.
        let x_lat = rand(&[1, 32, 32, 32], 7);
        prove(
            "DC-AE encoder output",
            crate::quantum::dcae_encoder_output_bytes(1, 32, 32, 32, 4),
            &|| enc.output(&x_lat),
        );
        for (label, formula, measured) in proofs {
            assert_tracks_measured(&label, formula, measured, MAX_RATIO);
        }
        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED dc_ae::quantum_proofs::dcae_quanta_track_their_measured_peaks");
        }
    }

    fn max_abs(a: &Array, b: &Array) -> f32 {
        a.as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The bounded encode tracks the whole-image encode within the tiled-decode parity standard —
    /// in fact to float reassociation: each tile carries the attention-free prefix's full receptive
    /// halo, so every core cell sees exactly the whole image's pixels. It must genuinely tile (both
    /// axes split, a border and an interior tile), and the halo is load-bearing: one cell less
    /// diverges observably. The stitch quantum's counted scratch is proven against its measured
    /// peak.
    #[test]
    fn tiled_encode_matches_the_whole_image_encode_and_prices_its_stitch() {
        const CHILD: &str = "SANA_TILED_ENCODE_CPU_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "dc_ae::quantum_proofs::tiled_encode_matches_the_whole_image_encode_and_prices_its_stitch",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        let cfg = config();
        let enc = DcAeEncoder::from_weights(&weights(&cfg), cfg).unwrap();
        let cancel = CancelFlag::new();
        // Position-dependent image so a seam would show.
        let (h, w) = (96, 80);
        let data: Vec<f32> = (0..3 * h * w)
            .map(|i| ((i as f32) * 0.013).sin() + ((i / w) % h) as f32 * 0.01)
            .collect();
        let image = Array::from_slice(&data, &[1, 3, h, w]);
        let whole = enc.encode(&image, &cancel).unwrap();
        let tiled = enc.encode_tiled(&image, 32, &cancel).unwrap();
        mlx_rs::transforms::eval([&whole, &tiled]).unwrap();
        assert_eq!(tiled.shape(), whole.shape());
        let scale = whole
            .as_slice::<f32>()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        let delta = max_abs(&tiled, &whole);
        println!("[tiled encode] max|delta| {delta:.3e} (scale {scale:.3e})");
        assert!(
            delta <= 1e-5 * scale.max(1.0),
            "tiled encode diverged: {delta}"
        );

        assert_eq!(enc.num_prefix_stages(), 2);
        assert_eq!(enc.prefix_scale(), 4);
        // conv_in 1 + stage0 2 + down 1, then at scale 2: stage1 4 + down 2 = 10 -> 12 (3 cells).
        assert_eq!(enc.prefix_halo(), 12);

        // Mutation control: the same tiling without its halo diverges observably, so the
        // comparison above discriminates.
        let x = image.transpose_axes(&[0, 2, 3, 1]).unwrap();
        let no_halo_rows: Vec<Vec<Array>> = (0..h)
            .step_by(32)
            .map(|y0| {
                (0..w)
                    .step_by(32)
                    .map(|x0| {
                        let tile = mlx_gen::attention::slice_axis(
                            &mlx_gen::attention::slice_axis(&x, 1, y0, 32.min(h - y0)).unwrap(),
                            2,
                            x0,
                            32.min(w - x0),
                        )
                        .unwrap();
                        enc.encode_prefix(&tile, &cancel).unwrap()
                    })
                    .collect()
            })
            .collect();
        let no_halo = enc
            .encode_rest(&stitch(&no_halo_rows).unwrap(), &cancel)
            .unwrap();
        no_halo.eval().unwrap();
        let seam = max_abs(&no_halo, &whole);
        println!("[tiled encode without halo] max|delta| {seam:.3e}");
        assert!(
            seam > 1e-3 * scale,
            "the halo-free control must diverge: {seam}"
        );

        // The stalled stitch allocation measurement is an individually selected capture below.
    }

    #[test]
    #[ignore = "expensive stalled fixture capture; run through the selected iOS capture workflow"]
    fn encode_stitch_tracks_its_measured_peak() {
        const CHILD: &str = "SANA_ENCODE_STITCH_PROOF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "dc_ae::quantum_proofs::encode_stitch_tracks_its_measured_peak",
                    "--include-ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("IOS_CAPTURE_IN_CHILD", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        // Stitch proof over a 3x3 grid of materialized 48x40x32 crops (the tiles' prefix outputs are
        // materialized by their own quanta before the stitch; shapes large enough that the
        // allocator's 16 KiB page rounding is negligible).
        let crops: Vec<Vec<Array>> = (0..3)
            .map(|r| {
                (0..3)
                    .map(|k| rand(&[1, 48, 40, 32], 100 + 3 * r + k))
                    .collect()
            })
            .collect();
        let formula = crate::quantum::dcae_encode_stitch_bytes(1, 3 * 48, 3 * 40, 32);
        let measured = measured_peak_bytes(|| Ok(vec![stitch(&crops)?])).unwrap();
        println!(
            "[DC-AE encode stitch] formula {formula} measured {measured} ratio {:.3}",
            formula as f64 / measured as f64
        );
        assert_tracks_measured("DC-AE encode stitch", formula, measured, MAX_RATIO);
        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED dc_ae::quantum_proofs::encode_stitch_tracks_its_measured_peak");
        }
    }
}
