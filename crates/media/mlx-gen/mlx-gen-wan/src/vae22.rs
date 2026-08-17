//! sc-2680 — the Wan **2.2 `vae22`** (z48, stride 4×16×16): the z48 video VAE used **only** by the
//! dense TI2V-5B (the 14B path uses the 2.1 z16 [`crate::vae::WanVae`]). Port of the `mlx_video`
//! reference `models/wan/vae22.py`, gated bit-for-bit against it (`tests/vae22_parity.rs`).
//!
//! Architecturally distinct from the 2.1 VAE:
//!  - **Channels-last `[B, T, H, W, C]`** throughout (the reference is native channels-last here),
//!    so the conv ops take the data directly (mlx convs are channels-last) — no NCTHW transposes.
//!  - **`RMS_norm` is a channel-L2 normalization over the LAST axis** with eps **1e-24** —
//!    `x / max(‖x‖₂ over C, 1e-24) · √C · γ` (the 2.1 VAE normalizes over axis 1 with eps 1e-12).
//!  - **Spatial 2×2 patchify / unpatchify**: the encoder packs the RGB `[B,T,2H,2W,3]` into 12
//!    channels before the conv stack; the decoder unpacks 12 → 3 after it (the extra ×2 spatial that
//!    makes the stride 16, not 8).
//!  - **Parameter-free `DupUp3D` / `AvgDown3D` shortcuts** (channel duplicate-reshape up / group-mean
//!    down) replace the 2.1 VAE's learned residual skips.
//!  - **`CausalConv3d` causal pad = `2·padding[0]` on the LEFT** (not the 2.1 `kt − st`): the
//!    encoder's temporal `downsample3d` `time_conv` uses `padding=0` → *no* causal pad (stride-2
//!    over an explicitly cache-prepended input), while the `(3,1,1)` upsample `time_conv` pads 2.
//!  - **Asymmetric channel widths**: encoder dim 160 → `[160,160,320,640,640]`, decoder dim 256 →
//!    `[1024,1024,1024,512,256]`.
//!  - **Causal temporal decode** (`first_chunk=True`): `T_lat → 1 + (T_lat−1)·4` (the leading
//!    causal-padding frames are trimmed), vs the 2.1 non-causal `T → T·4`.
//!
//! Everything runs **f32** (the reference upcasts the VAE to f32; f32 also sidesteps the bf16 NAX
//! kernel history). The 48-value `VAE22_MEAN`/`VAE22_STD` are architecture constants the port
//! hardcodes (and the fixture gates). Honors [[divergence-is-not-rounding-pattern]]: the sole
//! expected gap is the float-summation order between mlx `conv3d` and the reference's
//! conv2d-per-temporal-slice decomposition of the same convolution (bounded, like the 2.1 gate).

use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, divide, maximum, mean_axis, minimum, multiply, pad, split,
    subtract, sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::{WAN_Z48_MEAN as VAE22_MEAN, WAN_Z48_STD as VAE22_STD};
use mlx_gen::nn::{conv2d, conv3d, silu, upsample_nearest};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, Result};

use crate::vae_common::{
    contiguous, eval, last_t_axis, scalar, slice_axis, tile_decode_accumulate,
    validate_decoder_tiling, FeatCache,
};

/// Last-`CACHE_T` frames are carried across chunks as causal left-context during encode.
const CACHE_T: i32 = 2;
/// Channel-L2 norm floor (reference `mx.maximum(l2_sq, 1e-24)`).
const NORM_EPS: f32 = 1e-24;

/// vae22 fixed structure (dim_mult [1,2,4,4], 2 res-blocks/stage).
const DIM_MULT_LEN: usize = 4;
const NUM_RES_BLOCKS: usize = 2;
const DIM_MULT: [usize; DIM_MULT_LEN] = [1, 2, 4, 4];
const PRODUCTION_DEC_DIM: usize = 256;
const PRODUCTION_ENC_DIM: usize = 160;
const PRODUCTION_Z_DIM: usize = 48;
/// Decoder temporal-upsample per stage (`upsample3d` vs `upsample2d`).
const TEMPORAL_UPSAMPLE: [bool; 3] = [true, true, false];
/// Encoder temporal-downsample per stage (`downsample3d` vs `downsample2d`).
const TEMPORAL_DOWNSAMPLE: [bool; 3] = [false, true, true];

/// One tensor consumed by the fixed Wan2.2 z48 VAE loader topology.
///
/// This is intentionally generated from the same stage widths/counts used by [`Wan22Vae`]'s
/// constructors rather than maintained as a second flat checkpoint-key list. Header-only callers
/// use it to prove a converted checkpoint is load-exact before publishing asset residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Wan22TensorSpec {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

/// The loader-owned decoder/encoder tensor topology for one Wan2.2 VAE width configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Wan22WeightSchema {
    pub(crate) decoder: Vec<Wan22TensorSpec>,
    pub(crate) encoder: Vec<Wan22TensorSpec>,
}

impl Wan22WeightSchema {
    pub(crate) fn tensors(&self) -> impl Iterator<Item = &Wan22TensorSpec> {
        self.decoder.iter().chain(self.encoder.iter())
    }
}

#[derive(Clone, Copy, Debug)]
struct Wan22Topology {
    dec_dim: usize,
    enc_dim: usize,
    z_dim: usize,
}

impl Wan22Topology {
    fn from_i32(dec_dim: i32, enc_dim: i32, z_dim: i32) -> Result<Self> {
        let positive = |label: &str, value: i32| {
            usize::try_from(value)
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| Error::Msg(format!("vae22: {label} must be positive, got {value}")))
        };
        Ok(Self {
            dec_dim: positive("decoder width", dec_dim)?,
            enc_dim: positive("encoder width", enc_dim)?,
            z_dim: positive("latent width", z_dim)?,
        })
    }

    fn production() -> Self {
        Self {
            dec_dim: PRODUCTION_DEC_DIM,
            enc_dim: PRODUCTION_ENC_DIM,
            z_dim: PRODUCTION_Z_DIM,
        }
    }

    fn decoder_stage_dims(self) -> [usize; DIM_MULT_LEN + 1] {
        [
            self.dec_dim * DIM_MULT[DIM_MULT_LEN - 1],
            self.dec_dim * DIM_MULT[3],
            self.dec_dim * DIM_MULT[2],
            self.dec_dim * DIM_MULT[1],
            self.dec_dim * DIM_MULT[0],
        ]
    }

    fn encoder_stage_dims(self) -> [usize; DIM_MULT_LEN + 1] {
        [
            self.enc_dim,
            self.enc_dim * DIM_MULT[0],
            self.enc_dim * DIM_MULT[1],
            self.enc_dim * DIM_MULT[2],
            self.enc_dim * DIM_MULT[3],
        ]
    }

    fn schema(self) -> Wan22WeightSchema {
        let mut decoder = Vec::new();
        push_conv3d(&mut decoder, "conv2", self.z_dim, self.z_dim, 1, 1, 1);

        let decoder_dims = self.decoder_stage_dims();
        let decoder_middle = decoder_dims[0];
        push_conv3d(
            &mut decoder,
            "decoder.conv1",
            decoder_middle,
            self.z_dim,
            3,
            3,
            3,
        );
        push_residual(
            &mut decoder,
            "decoder.middle.0",
            decoder_middle,
            decoder_middle,
        );
        push_attention(&mut decoder, "decoder.middle.1", decoder_middle);
        push_residual(
            &mut decoder,
            "decoder.middle.2",
            decoder_middle,
            decoder_middle,
        );
        for stage in 0..DIM_MULT_LEN {
            let input = decoder_dims[stage];
            let output = decoder_dims[stage + 1];
            let prefix = format!("decoder.upsamples.{stage}");
            for block in 0..=NUM_RES_BLOCKS {
                push_residual(
                    &mut decoder,
                    &format!("{prefix}.upsamples.{block}"),
                    if block == 0 { input } else { output },
                    output,
                );
            }
            if stage != DIM_MULT_LEN - 1 {
                let resample = format!("{prefix}.upsamples.{}", NUM_RES_BLOCKS + 1);
                if TEMPORAL_UPSAMPLE.get(stage).copied().unwrap_or(false) {
                    push_conv3d(
                        &mut decoder,
                        &format!("{resample}.time_conv"),
                        output * 2,
                        output,
                        3,
                        1,
                        1,
                    );
                }
                push_conv2d(&mut decoder, &resample, "resample", output, output, 3, 3);
            }
        }
        push_head(&mut decoder, "decoder.head", decoder_dims[DIM_MULT_LEN], 12);

        // `convert_ti2v_5b` always emits the full encoder. The top-level `conv1` consumes the
        // encoder head's 2*z moments after sampling; it is part of the encoder half even though its
        // key is not under `encoder.*`.
        let mut encoder = Vec::new();
        push_conv3d(
            &mut encoder,
            "conv1",
            self.z_dim * 2,
            self.z_dim * 2,
            1,
            1,
            1,
        );
        let encoder_dims = self.encoder_stage_dims();
        push_conv3d(&mut encoder, "encoder.conv1", encoder_dims[0], 12, 3, 3, 3);
        for stage in 0..DIM_MULT_LEN {
            let input = encoder_dims[stage];
            let output = encoder_dims[stage + 1];
            let prefix = format!("encoder.downsamples.{stage}");
            for block in 0..NUM_RES_BLOCKS {
                push_residual(
                    &mut encoder,
                    &format!("{prefix}.downsamples.{block}"),
                    if block == 0 { input } else { output },
                    output,
                );
            }
            if stage < DIM_MULT_LEN - 1 {
                let resample = format!("{prefix}.downsamples.{NUM_RES_BLOCKS}");
                push_conv2d(&mut encoder, &resample, "resample", output, output, 3, 3);
                if TEMPORAL_DOWNSAMPLE.get(stage).copied().unwrap_or(false) {
                    push_conv3d(
                        &mut encoder,
                        &format!("{resample}.time_conv"),
                        output,
                        output,
                        3,
                        1,
                        1,
                    );
                }
            }
        }
        let encoder_middle = encoder_dims[DIM_MULT_LEN];
        push_residual(
            &mut encoder,
            "encoder.middle.0",
            encoder_middle,
            encoder_middle,
        );
        push_attention(&mut encoder, "encoder.middle.1", encoder_middle);
        push_residual(
            &mut encoder,
            "encoder.middle.2",
            encoder_middle,
            encoder_middle,
        );
        push_head(&mut encoder, "encoder.head", encoder_middle, self.z_dim * 2);

        Wan22WeightSchema { decoder, encoder }
    }

    fn validate_loaded_shapes(self, weights: &Weights, include_encoder: bool) -> Result<()> {
        let schema = self.schema();
        let specs = schema.decoder.iter().chain(
            include_encoder
                .then_some(schema.encoder.iter())
                .into_iter()
                .flatten(),
        );
        for spec in specs {
            let value = weights.require(&spec.name)?;
            let expected: Vec<i32> = spec
                .shape
                .iter()
                .map(|&dimension| {
                    i32::try_from(dimension).map_err(|_| {
                        Error::Msg(format!(
                            "vae22: tensor {} dimension {dimension} exceeds MLX i32 shape range",
                            spec.name
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            if value.shape() != expected {
                return Err(Error::Msg(format!(
                    "vae22: tensor {} has shape {:?}, expected {:?}",
                    spec.name,
                    value.shape(),
                    expected
                )));
            }
        }
        Ok(())
    }
}

fn push_tensor(tensors: &mut Vec<Wan22TensorSpec>, name: impl Into<String>, shape: &[usize]) {
    tensors.push(Wan22TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
    });
}

fn push_conv3d(
    tensors: &mut Vec<Wan22TensorSpec>,
    prefix: &str,
    output: usize,
    input: usize,
    kt: usize,
    kh: usize,
    kw: usize,
) {
    push_tensor(
        tensors,
        format!("{prefix}.weight"),
        &[output, kt, kh, kw, input],
    );
    push_tensor(tensors, format!("{prefix}.bias"), &[output]);
}

fn push_conv2d(
    tensors: &mut Vec<Wan22TensorSpec>,
    prefix: &str,
    leaf: &str,
    output: usize,
    input: usize,
    kh: usize,
    kw: usize,
) {
    push_tensor(
        tensors,
        format!("{prefix}.{leaf}_weight"),
        &[output, kh, kw, input],
    );
    push_tensor(tensors, format!("{prefix}.{leaf}_bias"), &[output]);
}

fn push_residual(tensors: &mut Vec<Wan22TensorSpec>, prefix: &str, input: usize, output: usize) {
    push_tensor(
        tensors,
        format!("{prefix}.residual.layer_0.gamma"),
        &[input],
    );
    push_conv3d(
        tensors,
        &format!("{prefix}.residual.layer_2"),
        output,
        input,
        3,
        3,
        3,
    );
    push_tensor(
        tensors,
        format!("{prefix}.residual.layer_3.gamma"),
        &[output],
    );
    push_conv3d(
        tensors,
        &format!("{prefix}.residual.layer_6"),
        output,
        output,
        3,
        3,
        3,
    );
    if input != output {
        push_conv3d(
            tensors,
            &format!("{prefix}.shortcut"),
            output,
            input,
            1,
            1,
            1,
        );
    }
}

fn push_attention(tensors: &mut Vec<Wan22TensorSpec>, prefix: &str, channels: usize) {
    push_tensor(tensors, format!("{prefix}.norm.gamma"), &[channels]);
    push_tensor(
        tensors,
        format!("{prefix}.to_qkv_weight"),
        &[channels * 3, 1, 1, channels],
    );
    push_tensor(tensors, format!("{prefix}.to_qkv_bias"), &[channels * 3]);
    push_tensor(
        tensors,
        format!("{prefix}.proj_weight"),
        &[channels, 1, 1, channels],
    );
    push_tensor(tensors, format!("{prefix}.proj_bias"), &[channels]);
}

fn push_head(tensors: &mut Vec<Wan22TensorSpec>, prefix: &str, input: usize, output: usize) {
    push_tensor(tensors, format!("{prefix}.layer_0.gamma"), &[input]);
    push_conv3d(
        tensors,
        &format!("{prefix}.layer_2"),
        output,
        input,
        3,
        3,
        3,
    );
}

/// Complete production schema emitted by [`crate::convert::convert_ti2v_5b`] and consumed by
/// [`Wan22Vae::from_weights`]. Both decoder and encoder are required for that canonical converter
/// identity even when a particular request is plain T2V.
pub(crate) fn production_weight_schema() -> Wan22WeightSchema {
    Wan22Topology::production().schema()
}

/// `x / max(‖x‖₂ over last axis, 1e-24) · √C · γ` — channel-L2 norm over the **last** axis (vae22's
/// `RMS_norm`). `gamma` carries `C` elements and broadcasts on the last axis.
///
/// **Dtype-preserving (sc-5039):** the channel-L2 reduction + divide always run in **f32** and the
/// result is cast back to `x`'s dtype. For an f32 input every cast is a no-op, so the f32 path stays
/// bit-for-bit identical to the reference (the [`vae22_parity`] gate). For a **bf16** decode this is
/// the one mandatory f32 island: bf16's 8-bit exponent matches f32's range, but a sum-of-squares
/// over 1024+ channels with the `1e-24` floor needs f32's mantissa, or the tiny denom makes the
/// divide blow up (the "NaX kernel history" the f32-everywhere choice originally sidestepped).
fn rms_norm_last(x: &Array, gamma: &Array) -> Result<Array> {
    let in_dtype = x.dtype();
    let sh = x.shape();
    let c = sh[sh.len() - 1];
    let xf = x.as_dtype(Dtype::Float32)?;
    let sum_sq = sum_axes(&multiply(&xf, &xf)?, &[(sh.len() - 1) as i32], true)?;
    let denom = maximum(&sum_sq, scalar(NORM_EPS))?.sqrt()?;
    let normed = divide(&xf, &denom)?;
    let scaled = multiply(&normed, scalar((c as f32).sqrt()))?;
    let out = multiply(&scaled, &gamma.as_dtype(Dtype::Float32)?)?;
    Ok(out.as_dtype(in_dtype)?)
}

/// np.repeat over the last axis: `[…, C]` → `[…, C·repeats]` with each channel duplicated `repeats`
/// times consecutively (the reference `mx.repeat(x, repeats, axis=-1)`).
fn repeat_last(x: &Array, repeats: i32) -> Result<Array> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let c = sh[sh.len() - 1];
    let mut sh_one = sh.to_vec();
    sh_one.push(1); // [..., C, 1]
    let mut sh_rep = sh.to_vec();
    sh_rep.push(repeats); // [..., C, repeats]
    let bc = broadcast_to(&x.reshape(&sh_one)?, &sh_rep)?;
    let mut sh_out = sh.to_vec();
    *sh_out.last_mut().unwrap() = c * repeats;
    Ok(bc.reshape(&sh_out)?)
}

/// Last `n` frames along the temporal axis (axis 1, channels-last): the reference `x[:, -n:]`.
fn last_t(x: &Array, n: i32) -> Result<Array> {
    last_t_axis(x, n, 1)
}

/// Temporal slice `x[:, start:end]` (axis 1).
fn slice_t(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 1, start, end)
}

/// vae22 3-D causal conv (channels-last `[B,T,H,W,C]`). Causal time pad = `2·pt` on the LEFT
/// (prepend `cache_x` first, then zero-pad any remainder); symmetric spatial pad `ph`/`pw`. Weight
/// is the reference's already-MLX `[O, kt, kh, kw, I]`. Implemented as a single mlx `conv3d` (vs the
/// reference's sum-of-conv2d decomposition — same convolution, bounded summation-order gap).
struct CausalConv3d22 {
    w: Array,
    b: Array,
    causal_pad_t: i32,
    ph: i32,
    pw: i32,
    st: i32,
}

impl CausalConv3d22 {
    /// `pt`/`ph`/`pw` are the reference `padding` (causal time pad = `2·pt`); `st` the temporal
    /// stride (2 only for the encoder `downsample3d` `time_conv`).
    fn from_weights(w: &Weights, prefix: &str, st: i32, pt: i32, ph: i32, pw: i32) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            causal_pad_t: 2 * pt,
            ph,
            pw,
            st,
        })
    }

    fn forward(&self, x_bthwc: &Array, cache_x: Option<&Array>) -> Result<Array> {
        let mut x = x_bthwc.clone();
        let mut pad_needed = self.causal_pad_t;
        if let Some(cx) = cache_x {
            if pad_needed > 0 {
                x = concatenate_axis(&[cx, &x], 1)?;
                pad_needed = (pad_needed - cx.shape()[1]).max(0);
            }
        }
        if pad_needed > 0 {
            x = pad(
                &x,
                &[(0, 0), (pad_needed, 0), (0, 0), (0, 0), (0, 0)][..],
                None,
                None,
            )?;
        }
        if self.ph > 0 || self.pw > 0 {
            x = pad(
                &x,
                &[
                    (0, 0),
                    (0, 0),
                    (self.ph, self.ph),
                    (self.pw, self.pw),
                    (0, 0),
                ][..],
                None,
                None,
            )?;
        }
        // x is already NDHWC (= [B,T,H,W,C]); valid conv with temporal stride st.
        conv3d(&x, &self.w, Some(&self.b), (self.st, 1, 1), (0, 0, 0))
    }
}

/// Cached conv: feed the *previous* slot as left-context, then store this chunk's last `CACHE_T`
/// frames. Mirrors the reference `cache_x = x[:, -CACHE_T:]` (+ 1-frame prepend when short).
fn cached_conv(conv: &CausalConv3d22, x: &Array, cache: &mut FeatCache) -> Result<Array> {
    let idx = cache.idx;
    let t = x.shape()[1];
    let mut cache_x = last_t(x, t.min(CACHE_T))?;
    if cache_x.shape()[1] < CACHE_T {
        if let Some(prev) = &cache.slots[idx] {
            cache_x = concatenate_axis(&[&last_t(prev, 1)?, &cache_x], 1)?;
        }
    }
    let y = conv.forward(x, cache.slots[idx].as_ref())?;
    cache.slots[idx] = Some(cache_x);
    cache.idx += 1;
    Ok(y)
}

/// `RMS → SiLU → conv(3³) → RMS → SiLU → conv(3³)` + residual (1³ shortcut when channels differ).
/// Reference list indices: `residual.layer_{0,2,3,6}` (the SiLU/Dropout gaps carry no params).
struct ResidualBlock {
    norm1: Array,
    conv1: CausalConv3d22,
    norm2: Array,
    conv2: CausalConv3d22,
    shortcut: Option<CausalConv3d22>,
}

impl ResidualBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.shortcut.weight")).is_some() {
            // 1×1×1 conv (kt=kh=kw=1, padding 0).
            Some(CausalConv3d22::from_weights(
                w,
                &format!("{prefix}.shortcut"),
                1,
                0,
                0,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: w
                .require(&format!("{prefix}.residual.layer_0.gamma"))?
                .clone(),
            conv1: CausalConv3d22::from_weights(
                w,
                &format!("{prefix}.residual.layer_2"),
                1,
                1,
                1,
                1,
            )?,
            norm2: w
                .require(&format!("{prefix}.residual.layer_3.gamma"))?
                .clone(),
            conv2: CausalConv3d22::from_weights(
                w,
                &format!("{prefix}.residual.layer_6"),
                1,
                1,
                1,
                1,
            )?,
            shortcut,
        })
    }

    fn shortcut(&self, x: &Array) -> Result<Array> {
        match &self.shortcut {
            Some(s) => s.forward(x, None),
            None => Ok(x.clone()),
        }
    }

    /// Decode path (no cache).
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = self
            .conv1
            .forward(&silu(&rms_norm_last(x, &self.norm1)?)?, None)?;
        eval(&y)?;
        let y = self
            .conv2
            .forward(&silu(&rms_norm_last(&y, &self.norm2)?)?, None)?;
        Ok(add(&y, &h)?)
    }

    /// Encode path (chunked, with `feat_cache`).
    fn forward_cached(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = silu(&rms_norm_last(x, &self.norm1)?)?;
        let y = cached_conv(&self.conv1, &y, cache)?;
        eval(&y)?;
        let y = silu(&rms_norm_last(&y, &self.norm2)?)?;
        let y = cached_conv(&self.conv2, &y, cache)?;
        Ok(add(&y, &h)?)
    }
}

/// Per-frame single-head spatial self-attention (head_dim = C). Channels-last `[B,T,H,W,C]` I/O.
struct AttentionBlock {
    norm: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
}

impl AttentionBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.gamma"))?.clone(),
            qkv_w: w.require(&format!("{prefix}.to_qkv_weight"))?.clone(),
            qkv_b: w.require(&format!("{prefix}.to_qkv_bias"))?.clone(),
            proj_w: w.require(&format!("{prefix}.proj_weight"))?.clone(),
            proj_b: w.require(&format!("{prefix}.proj_bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, t, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // Merge B·T, channel-L2 norm over C, then the 1×1 convs (channels-last).
        let xf = x.reshape(&[bt, h, w, c])?;
        let normed = rms_norm_last(&xf, &self.norm)?;
        let qkv = conv2d(&normed, &self.qkv_w, Some(&self.qkv_b), 1, 0)?; // (BT,H,W,3C)
        let qkv = qkv.reshape(&[bt, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 2)?; // q,k,v each (BT, H·W, C)
        let q = parts[0].expand_dims(1)?; // (BT, 1, H·W, C)
        let k = parts[1].expand_dims(1)?;
        let v = parts[2].expand_dims(1)?;
        let scale = (c as f32).powf(-0.5);
        let o = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.reshape(&[bt, h, w, c])?;
        let o = conv2d(&o, &self.proj_w, Some(&self.proj_b), 1, 0)?; // (BT,H,W,C)
        let o = o.reshape(&[b, t, h, w, c])?;
        Ok(add(&o, x)?)
    }
}

/// Parameter-free up shortcut: duplicate channels, reshape, interleave → spatial/temporal upsample.
struct DupUp3D {
    out_c: i32,
    factor_t: i32,
    factor_s: i32,
    repeats: i32,
}

impl DupUp3D {
    fn new(in_c: i32, out_c: i32, factor_t: i32, factor_s: i32) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            out_c,
            factor_t,
            factor_s,
            repeats: out_c * factor / in_c,
        }
    }

    fn forward(&self, x: &Array, first_chunk: bool) -> Result<Array> {
        let sh = x.shape();
        let (b, t, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let (ft, fs) = (self.factor_t, self.factor_s);
        let x = repeat_last(x, self.repeats)?; // [B,T,H,W,out_c·factor]
        let x = x.reshape(&[b, t, h, w, self.out_c, ft, fs, fs])?;
        let x = x.transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4])?; // [B,T,ft,H,fs,W,fs,out_c]
        let x = x.reshape(&[b, t * ft, h * fs, w * fs, self.out_c])?;
        if first_chunk {
            slice_t(&x, ft - 1, t * ft)
        } else {
            Ok(x)
        }
    }
}

/// Parameter-free down shortcut: group channels across spatial/temporal factors and average.
struct AvgDown3D {
    out_c: i32,
    factor_t: i32,
    factor_s: i32,
    factor: i32,
    group_size: i32,
}

impl AvgDown3D {
    fn new(in_c: i32, out_c: i32, factor_t: i32, factor_s: i32) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            out_c,
            factor_t,
            factor_s,
            factor,
            group_size: in_c * factor / out_c,
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, mut t, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (ft, fs) = (self.factor_t, self.factor_s);
        let mut x = x.clone();
        let pad_t = (ft - t % ft) % ft;
        if pad_t > 0 {
            x = pad(
                &x,
                &[(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)][..],
                None,
                None,
            )?;
            t += pad_t;
        }
        let x = x.reshape(&[b, t / ft, ft, h / fs, fs, w / fs, fs, c])?;
        let x = x.transpose_axes(&[0, 1, 3, 5, 7, 2, 4, 6])?; // [B,T',H',W',C,ft,fs,fs]
        let x = x.reshape(&[b, t / ft, h / fs, w / fs, c * self.factor])?;
        let x = x.reshape(&[b, t / ft, h / fs, w / fs, self.out_c, self.group_size])?;
        Ok(mean_axis(&x, 5, false)?)
    }
}

/// Decoder spatial 2× upsample (`resample_weight` = Conv2d C→C, 3×3 pad 1). `upsample3d` first
/// doubles T via a learned `time_conv` (C→2C interleaved); `upsample2d` is spatial-only.
struct UpsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d22>,
}

impl UpsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            // CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0)).
            Some(CausalConv3d22::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                1,
                1,
                0,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample_weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample_bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x: &Array, first_chunk: bool) -> Result<Array> {
        let sh = x.shape();
        let (b, _t0, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let mut x = x.clone();
        if let Some(tc) = &self.time_conv {
            // time_conv: C→2C; reshape the 2C into (stream, C), then interleave the two streams
            // along the temporal axis (`stack([s0,s1], axis=2).reshape(·, T·2, ·)` ≡ moving the
            // stream axis next to time and flattening).
            let t = x.shape()[1];
            if first_chunk && t > 1 {
                let first = slice_t(&x, 0, 1)?;
                let rest = slice_t(&x, 1, t)?;
                let tc_out = tc.forward(&rest, None)?.reshape(&[b, t - 1, h, w, 2, c])?;
                let inter = tc_out.transpose_axes(&[0, 1, 4, 2, 3, 5])?.reshape(&[
                    b,
                    (t - 1) * 2,
                    h,
                    w,
                    c,
                ])?;
                x = concatenate_axis(&[&first, &inter], 1)?;
            } else {
                let tc_out = tc.forward(&x, None)?.reshape(&[b, t, h, w, 2, c])?;
                x = tc_out
                    .transpose_axes(&[0, 1, 4, 2, 3, 5])?
                    .reshape(&[b, t * 2, h, w, c])?;
            }
            eval(&x)?;
        }
        // Per-frame nearest-2× spatial upsample + 3×3 conv (C→C).
        let t = x.shape()[1];
        let xs = x.reshape(&[b * t, h, w, c])?;
        let up = upsample_nearest(&xs, 2)?;
        let y = conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?;
        let c_out = y.shape()[3];
        Ok(y.reshape(&[b, t, h * 2, w * 2, c_out])?)
    }
}

/// Encoder spatial 2× downsample (ZeroPad-(0,1,0,1) + stride-2 3×3 conv C→C). `downsample3d` adds a
/// temporal stride-2 `time_conv` (padding 0) with chunk-cache: first chunk passes through, later
/// chunks fold the previous chunk's last frame as the (manually prepended) left-context.
struct DownsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d22>,
}

impl DownsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            // CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1), padding=0) → causal pad 0.
            Some(CausalConv3d22::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                2,
                0,
                0,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample_weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample_bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let sh = x.shape();
        let (b, t, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // Per-frame ZeroPad(0,1,0,1) + valid stride-2 conv.
        let xs = x.reshape(&[bt, h, w, c])?;
        let xp = pad(&xs, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
        let y = conv2d(&xp, &self.conv_w, Some(&self.conv_b), 2, 0)?;
        let (h2, w2, c2) = (y.shape()[1], y.shape()[2], y.shape()[3]);
        let mut x = y.reshape(&[b, t, h2, w2, c2])?;

        if let Some(tc) = &self.time_conv {
            let idx = cache.idx;
            if cache.slots[idx].is_none() {
                // First chunk: stash the spatially-downsampled x, skip the temporal conv.
                cache.slots[idx] = Some(x.clone());
            } else {
                let save_x = last_t(&x, 1)?;
                let prev_last = last_t(cache.slots[idx].as_ref().unwrap(), 1)?;
                // padding=0 time_conv, manual 1-frame prepend (no cache_x).
                x = tc.forward(&concatenate_axis(&[&prev_last, &x], 1)?, None)?;
                cache.slots[idx] = Some(save_x);
            }
            cache.idx += 1;
        }
        Ok(x)
    }
}

/// One decoder up-stage: `num_res_blocks+1` residual blocks (+ optional spatial/temporal upsample),
/// plus a parameter-free `DupUp3D` shortcut on the stage input (when `up_flag`).
struct UpResBlock {
    shortcut: Option<DupUp3D>,
    resblocks: Vec<ResidualBlock>,
    resample: Option<UpsampleBlock>,
}

impl UpResBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        in_c: i32,
        out_c: i32,
        num_res: usize,
        temporal: bool,
        up_flag: bool,
    ) -> Result<Self> {
        let shortcut = if up_flag {
            Some(DupUp3D::new(in_c, out_c, if temporal { 2 } else { 1 }, 2))
        } else {
            None
        };
        let mut resblocks = Vec::with_capacity(num_res);
        for i in 0..num_res {
            resblocks.push(ResidualBlock::from_weights(
                w,
                &format!("{prefix}.upsamples.{i}"),
            )?);
        }
        let resample = if up_flag {
            Some(UpsampleBlock::from_weights(
                w,
                &format!("{prefix}.upsamples.{num_res}"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self {
            shortcut,
            resblocks,
            resample,
        })
    }

    fn forward(&self, x: &Array, first_chunk: bool) -> Result<Array> {
        let mut x_main = x.clone();
        for rb in &self.resblocks {
            x_main = rb.forward(&x_main)?;
            eval(&x_main)?;
        }
        if let Some(up) = &self.resample {
            x_main = up.forward(&x_main, first_chunk)?;
            eval(&x_main)?;
        }
        match &self.shortcut {
            Some(sc) => {
                let x_shortcut = sc.forward(x, first_chunk)?;
                eval(&x_shortcut)?;
                Ok(add(&x_main, &x_shortcut)?)
            }
            None => Ok(x_main),
        }
    }
}

/// One encoder down-stage: `num_res_blocks` residual blocks (+ optional spatial/temporal downsample),
/// plus a parameter-free `AvgDown3D` shortcut on the stage input (always).
struct DownResBlock {
    shortcut: AvgDown3D,
    resblocks: Vec<ResidualBlock>,
    resample: Option<DownsampleBlock>,
}

impl DownResBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        in_c: i32,
        out_c: i32,
        num_res: usize,
        temporal: bool,
        down_flag: bool,
    ) -> Result<Self> {
        let shortcut = AvgDown3D::new(
            in_c,
            out_c,
            if temporal { 2 } else { 1 },
            if down_flag { 2 } else { 1 },
        );
        let mut resblocks = Vec::with_capacity(num_res);
        for i in 0..num_res {
            resblocks.push(ResidualBlock::from_weights(
                w,
                &format!("{prefix}.downsamples.{i}"),
            )?);
        }
        let resample = if down_flag {
            Some(DownsampleBlock::from_weights(
                w,
                &format!("{prefix}.downsamples.{num_res}"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self {
            shortcut,
            resblocks,
            resample,
        })
    }

    /// Encoder down-stage forward; the conv-cache slots it consumes are counted by
    /// [`Self::cached_convs`] (used once at build to size the cache).
    fn forward(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let x_shortcut = self.shortcut.forward(x)?;
        eval(&x_shortcut)?;
        let mut x = x.clone();
        for rb in &self.resblocks {
            x = rb.forward_cached(&x, cache)?;
            eval(&x)?;
        }
        if let Some(d) = &self.resample {
            x = d.forward(&x, cache)?;
            eval(&x)?;
        }
        Ok(add(&x, &x_shortcut)?)
    }

    /// Cached convs this stage consumes (2 per residual block + 1 for a `downsample3d` time_conv).
    fn cached_convs(&self) -> usize {
        self.resblocks.len() * 2
            + self
                .resample
                .as_ref()
                .map(|d| usize::from(d.time_conv.is_some()))
                .unwrap_or(0)
    }
}

/// Decoder output head: `RMS_norm → SiLU → CausalConv3d(·, out, 3³)`. `layer_0` gamma, `layer_2` conv.
struct Head22 {
    norm: Array,
    conv: CausalConv3d22,
}

impl Head22 {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.layer_0.gamma"))?.clone(),
            conv: CausalConv3d22::from_weights(w, &format!("{prefix}.layer_2"), 1, 1, 1, 1)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.conv
            .forward(&silu(&rms_norm_last(x, &self.norm)?)?, None)
    }

    fn forward_cached(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let y = silu(&rms_norm_last(x, &self.norm)?)?;
        cached_conv(&self.conv, &y, cache)
    }
}

/// `conv1 → [Res, Attn, Res] → upsamples → RMS+SiLU+conv` (z_dim → 12). Non-causal (single pass).
struct Decoder3d {
    conv1: CausalConv3d22,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    upsamples: Vec<UpResBlock>,
    head: Head22,
}

impl Decoder3d {
    /// `dec_dim` is the decoder base width (256 in production); the latent channel count rides on
    /// the `conv1` weight, so it isn't needed here.
    fn from_weights(w: &Weights, topology: Wan22Topology) -> Result<Self> {
        let p = "decoder";
        // The same widths generate the header-only schema used by memory admission.
        let dims = topology.decoder_stage_dims();
        let mut upsamples = Vec::new();
        for i in 0..DIM_MULT_LEN {
            let in_c = dims[i] as i32;
            let out_c = dims[i + 1] as i32;
            let temporal = TEMPORAL_UPSAMPLE.get(i).copied().unwrap_or(false);
            let up_flag = i != DIM_MULT_LEN - 1;
            upsamples.push(UpResBlock::from_weights(
                w,
                &format!("{p}.upsamples.{i}"),
                in_c,
                out_c,
                NUM_RES_BLOCKS + 1,
                temporal,
                up_flag,
            )?);
        }
        Ok(Self {
            conv1: CausalConv3d22::from_weights(w, &format!("{p}.conv1"), 1, 1, 1, 1)?,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            upsamples,
            head: Head22::from_weights(w, &format!("{p}.head"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_upsample_tail(&self.forward_middle(x)?)
    }

    /// The **globally-scoped** half: `conv1` → the three middle blocks, at latent resolution
    /// (sc-19753).
    ///
    /// `middle.1` is an [`AttentionBlock`] whose softmax spans every `H·W` spatial token of a frame
    /// ([`AttentionBlock::forward`]), so a spatial tile that runs it attends only to its own crop.
    /// The channel-L2 [`rms_norm_last`] used throughout this VAE reduces only the last (channel)
    /// axis and is genuinely tiling-invariant; the attention is the op that is not.
    fn forward_middle(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv1.forward(x, None)?;
        x = self.middle.0.forward(&x)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward(&x)?;
        eval(&x)?;
        Ok(x)
    }

    /// The **spatially-local** half: the `UpResBlock` stack and the `Head22` epilogue. Convolutions,
    /// nearest upsample, channel-duplicating shortcuts and the per-position channel-L2 norm only.
    fn forward_upsample_tail(&self, middle: &Array) -> Result<Array> {
        let mut x = middle.clone();
        for up in &self.upsamples {
            x = up.forward(&x, true)?;
            eval(&x)?;
        }
        self.head.forward(&x)
    }
}

/// `conv1 → downsamples → [Res, Attn, Res] → RMS+SiLU+conv` (12 → z_dim·2). Chunked + cached.
struct Encoder3d {
    conv1: CausalConv3d22,
    downsamples: Vec<DownResBlock>,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    head: Head22,
    cache_slots: usize,
}

impl Encoder3d {
    /// `enc_dim` is the encoder base width (160 in production), `z2` the head output (= z_dim·2).
    fn from_weights(w: &Weights, topology: Wan22Topology) -> Result<Self> {
        let p = "encoder";
        // The same widths generate the header-only schema used by memory admission.
        let dims = topology.encoder_stage_dims();
        let mut downsamples = Vec::new();
        let mut cache_slots = 1usize; // conv1
        for i in 0..DIM_MULT_LEN {
            let in_c = dims[i] as i32;
            let out_c = dims[i + 1] as i32;
            let temporal = TEMPORAL_DOWNSAMPLE.get(i).copied().unwrap_or(false);
            let down_flag = i < DIM_MULT_LEN - 1;
            let stage = DownResBlock::from_weights(
                w,
                &format!("{p}.downsamples.{i}"),
                in_c,
                out_c,
                NUM_RES_BLOCKS,
                temporal,
                down_flag,
            )?;
            cache_slots += stage.cached_convs();
            downsamples.push(stage);
        }
        cache_slots += 4; // middle: 2 cached residual blocks × 2 convs
        cache_slots += 1; // head conv
        Ok(Self {
            conv1: CausalConv3d22::from_weights(w, &format!("{p}.conv1"), 1, 1, 1, 1)?,
            downsamples,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            head: Head22::from_weights(w, &format!("{p}.head"))?,
            cache_slots,
        })
    }

    fn forward(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let mut x = cached_conv(&self.conv1, x, cache)?;
        for stage in &self.downsamples {
            x = stage.forward(&x, cache)?;
        }
        x = self.middle.0.forward_cached(&x, cache)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward_cached(&x, cache)?;
        eval(&x)?;
        self.head.forward_cached(&x, cache)
    }
}

/// Spatial 2×2 patchify: `[B,T,2H,2W,C] → [B,T,H,W,C·4]` (channel pack order C, r, q).
fn patchify(x: &Array, p: i32) -> Result<Array> {
    if p == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, t, hf, wf, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let (h, w) = (hf / p, wf / p);
    let x = x.reshape(&[b, t, h, p, w, p, c])?; // [B,T,H,q,W,r,C]
    let x = x.transpose_axes(&[0, 1, 2, 4, 6, 5, 3])?; // [B,T,H,W,C,r,q]
    Ok(x.reshape(&[b, t, h, w, c * p * p])?)
}

/// Inverse of [`patchify`]: `[B,T,H,W,C·4] → [B,T,2H,2W,C]`.
fn unpatchify(x: &Array, p: i32) -> Result<Array> {
    if p == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, t, h, w, cpacked) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let c = cpacked / (p * p);
    let x = x.reshape(&[b, t, h, w, c, p, p])?; // [B,T,H,W,C,r,q]
    let x = x.transpose_axes(&[0, 1, 2, 6, 3, 5, 4])?; // [B,T,H,q,W,r,C]
    Ok(x.reshape(&[b, t, h * p, w * p, c])?)
}

/// The Wan 2.2 z48 VAE: a decoder (always) + optional encoder (TI2V), with per-channel latent
/// normalization. Decode latent → video; encode video → normalized latent.
pub struct Wan22Vae {
    conv2: CausalConv3d22, // decoder-top pointwise (z → z)
    decoder: Decoder3d,
    encoder: Option<(CausalConv3d22, Encoder3d)>, // (encoder-top pointwise z·2 → z·2, encoder)
    z_dim: i32,
    mean: Array, // [1,1,1,1,z]
    std: Array,  // [1,1,1,1,z]
    /// Decoder compute dtype, **inferred from the loaded weights** (sc-5039): f32 by default, or
    /// bf16 when the caller pre-casts the weight map (`Weights::cast_all(Bfloat16)`) to halve the
    /// per-tile decoder-activation footprint. The latent denorm and the `RMS_norm` reduction always
    /// run f32; this only governs the conv-heavy body. Decode-only — encode stays whatever loaded.
    compute_dtype: Dtype,
}

/// Generic decode adapter for the Wan 2.2 z48 video VAE. The underlying implementation is
/// channels-last; the trait boundary normalizes its decoded output to NCTHW like the other decoders.
pub struct Wan22VideoDecoder<'a> {
    vae: &'a Wan22Vae,
}

impl<'a> Wan22VideoDecoder<'a> {
    pub fn new(vae: &'a Wan22Vae) -> Self {
        Self { vae }
    }

    fn to_ncthw(decoded: &Array) -> Result<Array> {
        let shape = decoded.shape();
        if shape.len() != 5 || shape[4] != 3 {
            return Err(Error::Msg(format!(
                "Wan z48 decoder produced invalid [B,T,H,W,3] output {shape:?}"
            )));
        }
        Ok(decoded.transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

impl LatentDecoder for Wan22VideoDecoder<'_> {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::WAN_Z48_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        Self::to_ncthw(&self.vae.decode(latents)?)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let shape = latents.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "Wan z48 video decoder expects [C,T,H,W], got {shape:?}"
            )));
        }
        validate_decoder_tiling(tiling, VaeTiling::WAN22, shape[1])?;
        Self::to_ncthw(&self.vae.decode_tiled(latents, tiling, cancel)?)
    }
}

impl Wan22Vae {
    /// Geometry owned by the concrete causal z48 decoder.
    pub const VAE_TILING: VaeTiling = VaeTiling::WAN22;

    /// Build from a weight map (`convert`-sanitized channels-last keys). Structure is fixed by the
    /// vae22 config; channel widths ride on the weights, so the same builder serves production (enc
    /// 160 / dec 256) and the tiny parity fixture. The encoder remains optional for decode-only
    /// callers, but any top-level `conv1.*` or `encoder.*` leaf requires its complete topology.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        Self::from_weights_dims(w, 256, 160, 48)
    }

    /// Build with explicit base widths + latent dim (the fixture uses tiny widths; `z_dim` stays 48
    /// in production so `VAE22_MEAN`/`STD` apply — a smaller `z_dim` fixture must inject its own).
    pub fn from_weights_dims(w: &Weights, dec_dim: i32, enc_dim: i32, z_dim: i32) -> Result<Self> {
        let topology = Wan22Topology::from_i32(dec_dim, enc_dim, z_dim)?;
        // Treat either the top-level encoder projection or any `encoder.*` leaf as the encoder
        // discriminator. Once present, the complete encoder topology is required; a partially
        // converted checkpoint must never build a deceptively decode-only VAE.
        let has_encoder = w
            .keys()
            .any(|key| key == "conv1.weight" || key == "conv1.bias" || key.starts_with("encoder."));
        topology.validate_loaded_shapes(w, has_encoder)?;
        let (mean, std) = Self::mean_std(w, z_dim)?;
        let encoder = if has_encoder {
            Some((
                CausalConv3d22::from_weights(w, "conv1", 1, 0, 0, 0)?, // 1×1×1 pointwise
                Encoder3d::from_weights(w, topology)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv2: CausalConv3d22::from_weights(w, "conv2", 1, 0, 0, 0)?, // 1×1×1 pointwise
            decoder: Decoder3d::from_weights(w, topology)?,
            encoder,
            z_dim,
            mean,
            std,
            // Infer the decode compute dtype from the loaded conv weights (bf16 when pre-cast).
            compute_dtype: w.require("conv2.weight")?.dtype(),
        })
    }

    /// Per-channel mean/std as `[1,1,1,1,z]` (channels-last). For the production `z_dim == 48` these
    /// are the hardcoded `VAE22_MEAN`/`STD`; a fixture with a different `z_dim` may supply its own via
    /// `vae22_mean`/`vae22_std` weight tensors (else this errors loudly rather than mis-normalize).
    fn mean_std(w: &Weights, z_dim: i32) -> Result<(Array, Array)> {
        if z_dim == 48 {
            let mean = Array::from_slice(&VAE22_MEAN, &[1, 1, 1, 1, 48]);
            let std = Array::from_slice(&VAE22_STD, &[1, 1, 1, 1, 48]);
            return Ok((mean, std));
        }
        let mean = w.get("vae22_mean").ok_or_else(|| {
            Error::Msg(format!(
                "vae22: z_dim={z_dim} != 48 needs a 'vae22_mean' weight (test fixture)"
            ))
        })?;
        let std = w.require("vae22_std")?;
        Ok((
            mean.reshape(&[1, 1, 1, 1, z_dim])?,
            std.reshape(&[1, 1, 1, 1, z_dim])?,
        ))
    }

    /// Decode a normalized latent `[z, T, H, W]` (channels-first, the denoise output convention) →
    /// video `[1, 1+(T−1)·4, 16·H, 16·W, 3]` (channels-last) in `[-1, 1]`. Denormalizes, runs the
    /// pointwise `conv2`, the causal (`first_chunk`) decoder, the 2×2 unpatchify, and clamps.
    pub fn decode(&self, latent_czthw: &Array) -> Result<Array> {
        let z = self.to_channels_last(latent_czthw)?; // [1,T,H,W,z]
        self.decode_cl(&z)
    }

    /// `[z, T, H, W]` → `[1, T, H, W, z]` (add batch, channels → last).
    fn to_channels_last(&self, latent_czthw: &Array) -> Result<Array> {
        Ok(latent_czthw.transpose_axes(&[1, 2, 3, 0])?.expand_dims(0)?)
    }

    /// Decode a channels-last normalized latent `[1, T, H, W, z]` → video `[1, T', 16H, 16W, 3]`.
    fn decode_cl(&self, z: &Array) -> Result<Array> {
        let denorm = add(&multiply(z, &self.std)?, &self.mean)?;
        // Enter the (optional) bf16 decode island: denorm stays f32, the conv-heavy body runs in
        // `compute_dtype` (no-op cast in f32 mode). The final clamp scalars are f32, so the output
        // promotes back to f32 regardless.
        let denorm = denorm.as_dtype(self.compute_dtype)?;
        let x = self.conv2.forward(&denorm, None)?;
        let out = self.decoder.forward(&x)?;
        let out = unpatchify(&out, 2)?;
        contiguous(&minimum(&maximum(&out, scalar(-1.0))?, scalar(1.0))?)
    }

    /// Decode with **tiling** for memory-bounded large/long video. Splits the channels-last middle
    /// feature map into overlapping tiles, runs the upsample tail + unpatchify + clamp on each, and
    /// trapezoidally blends. Falls back to single-pass [`Self::decode`] when `cfg` doesn't fire.
    /// vae22 upsamples 16× spatially, 4× temporally, **causally** ([`VaeTiling::WAN22`]).
    ///
    /// **Normalization semantics (sc-19753).** Denormalize, `conv2` and the decoder's middle blocks
    /// — including the spatial self-attention — run **once** on the full latent
    /// (`Decoder3d::forward_middle`); only the spatially-local tail is tiled. This previously ran
    /// the whole decoder per tile, so every spatial tile's `middle.1` softmax attended over its own
    /// crop's token set instead of the frame's. The middle blocks are shape-preserving at latent
    /// resolution, so the tile plan is unchanged.
    ///
    /// **Memory cost of the dense head**, as for the z16 sibling: the middle feature map is
    /// materialized whole, but at *latent* resolution and in `compute_dtype`, so it is a fraction of
    /// the full-size output accumulator the tiling already required.
    pub fn decode_tiled(
        &self,
        latent_czthw: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let z = self.to_channels_last(latent_czthw)?; // [1,T,H,W,z]
        let sh = z.shape();
        let (f, h, w) = (sh[1], sh[2], sh[3]);
        if !cfg.needs_tiling(Self::VAE_TILING, f, h, w) {
            return self.decode_cl(&z);
        }
        let denorm = add(&multiply(&z, &self.std)?, &self.mean)?;
        // The attention-bearing middle blocks run once, in `compute_dtype`, on the full latent.
        let middle = self.decoder.forward_middle(
            &self
                .conv2
                .forward(&denorm.as_dtype(self.compute_dtype)?, None)?,
        )?;
        let plan = cfg.plan(Self::VAE_TILING, f, h, w);

        // Channels-last: channel axis last, tiled axes [1, 2, 3]. Per-tile work adds the 2× spatial
        // unpatchify (vae22 upsamples 16× via decoder×8 + patch×2) before the clamp. The per-tile
        // body runs in `compute_dtype` (bf16 halves its activation peak, sc-5039); the f32 blend
        // accumulators are unchanged (the clamp scalars promote each tile back to f32).
        tile_decode_accumulate(&middle, &plan, [1, 2, 3], cancel, |tile| {
            let dec = self.decoder.forward_upsample_tail(tile)?;
            let dec = unpatchify(&dec, 2)?;
            Ok(minimum(&maximum(&dec, scalar(-1.0))?, scalar(1.0))?)
        })
    }

    /// Encode an image/video `[1, T, H, W, 3]` (channels-last, `T = 1 + 4·k`, values in `[-1, 1]`) →
    /// normalized latent `[1, T_lat, H/16, W/16, z]` (channels-last) via chunked causal encoding.
    /// Requires encoder weights.
    pub fn encode(&self, img_1thwc: &Array) -> Result<Array> {
        let (top_conv1, encoder) = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("vae22: encode requires encoder weights".into()))?;

        let x = patchify(img_1thwc, 2)?; // [1,T,H/2,W/2,12]
        let t = x.shape()[1];
        let num_chunks = 1 + (t - 1) / 4;
        let mut cache = FeatCache::new(encoder.cache_slots);
        let mut out: Option<Array> = None;
        for i in 0..num_chunks {
            cache.idx = 0;
            let chunk = if i == 0 {
                slice_t(&x, 0, 1)?
            } else {
                slice_t(&x, 1 + 4 * (i - 1), 1 + 4 * i)?
            };
            let chunk_out = encoder.forward(&chunk, &mut cache)?;
            out = Some(match out {
                None => chunk_out,
                Some(o) => concatenate_axis(&[&o, &chunk_out], 1)?,
            });
            eval(out.as_ref().unwrap())?;
        }
        // Pointwise conv1 (z·2 → z·2) then split mu = first z channels; normalize.
        let out = out.ok_or_else(|| Error::Msg("wan vae22: encode produced no chunks".into()))?;
        let out = top_conv1.forward(&out, None)?;
        let mu = slice_axis(&out, 4, 0, self.z_dim)?;
        let normed = divide(&subtract(&mu, &self.mean)?, &self.std)?;
        contiguous(&normed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_last_matches_closed_form() {
        // 2 channels: ‖x‖₂=√5, √C=√2 → out = x/√5·√2·γ.
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 1, 2]);
        let gamma = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let got = rms_norm_last(&x, &gamma).unwrap();
        let got = got.as_slice::<f32>();
        let s = (2.0f32).sqrt() / (5.0f32).sqrt();
        assert!((got[0] - s).abs() < 1e-6);
        assert!((got[1] - 2.0 * s).abs() < 1e-6);
    }

    #[test]
    fn patchify_unpatchify_roundtrip() {
        // [1,1,4,4,3] → patchify → [1,1,2,2,12] → unpatchify → original.
        let n = 4 * 4 * 3; // = 48 elements of the [1,1,4,4,3] tensor
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = Array::from_slice(&data, &[1, 1, 4, 4, 3]);
        let p = patchify(&x, 2).unwrap();
        assert_eq!(p.shape(), &[1, 1, 2, 2, 12]);
        let r = unpatchify(&p, 2).unwrap();
        assert_eq!(r.shape(), &[1, 1, 4, 4, 3]);
        assert_eq!(contiguous(&r).unwrap().as_slice::<f32>(), &data[..]);
    }

    #[test]
    fn repeat_last_duplicates_channels() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let r = repeat_last(&x, 2).unwrap();
        assert_eq!(r.as_slice::<f32>(), &[1.0, 1.0, 2.0, 2.0]);
    }
}
