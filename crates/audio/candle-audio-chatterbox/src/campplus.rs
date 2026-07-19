//! The **CAMPPlus speaker encoder** — the second ported S3Gen sub-network (sc-13236). A faithful
//! native-candle port of Chatterbox's `models/s3gen/xvector.py` `CAMPPlus` (a CosyVoice-derived
//! D-TDNN x-vector network, itself modified from 3D-Speaker/FunASR): an 80-bin Kaldi-fbank →
//! **192-d x-vector** TDNN/D-TDNN with CAM (context-aware masking) attention statistics.
//!
//! It is the `speaker_encoder.*` block of `s3gen.safetensors` (**937 tensors**). Its 192-d output
//! is the *S3Gen* speaker vector — a DIFFERENT vector from the 256-d `chatterbox_ve` GE2E embedding
//! (sc-12844) that conditions T3. The S3Gen flow consumes it after **L2-normalization** and the
//! `flow.spk_embed_affine_layer` **Linear 192→80** (which lives under `flow.*`, not
//! `speaker_encoder.*`, in the checkpoint — wired here as [`Campplus::spk_embed_flow`]).
//!
//! ## Weight layout (`speaker_encoder.*`, 937 tensors — verified against the pinned checkpoint)
//!
//! ```text
//!   head (FCM — a frequency-context 2-D residual front-end, m_channels = 32)
//!     head.conv1.weight [32,1,3,3]          Conv2d(1→32, k3, s1, p1)   + head.bn1  (BatchNorm2d)
//!     head.layer1.{0,1}                      2× BasicResBlock(32,32)  (block 0 downsamples freq ×2)
//!     head.layer2.{0,1}                      2× BasicResBlock(32,32)  (block 0 downsamples freq ×2)
//!     head.conv2.weight [32,32,3,3]         Conv2d(32→32, k3, s(2,1), p1) + head.bn2
//!       → reshape to 320 = 32·(80/8) channels, time preserved
//!   xvector (D-TDNN trunk, growth_rate = 32, bn_channels = 128, init_channels = 128)
//!     xvector.tdnn                           TDNNLayer(320→128, k5, s2)   (time ÷2)
//!     xvector.block1  12× CAMDenseTDNNLayer  dilation 1  → 128 + 12·32 = 512 ch
//!     xvector.transit1                       TransitLayer(512→256)
//!     xvector.block2  24× CAMDenseTDNNLayer  dilation 2  → 256 + 24·32 = 1024 ch
//!     xvector.transit2                       TransitLayer(1024→512)
//!     xvector.block3  16× CAMDenseTDNNLayer  dilation 2  → 512 + 16·32 = 1024 ch
//!     xvector.transit3                       TransitLayer(1024→512)
//!     xvector.out_nonlinear                  BatchNorm1d(512)+ReLU
//!     xvector.stats                          mean‖std pooling → 1024
//!     xvector.dense                          DenseLayer(1024→192, batchnorm_ / affine=False)
//! ```
//!
//! ## Faithfulness notes (verified against `resemble-ai/chatterbox` + torchaudio)
//!
//! - **Front-end**: `torchaudio.compliance.kaldi.fbank(num_mel_bins=80)` with the torchaudio
//!   defaults (16 kHz, 25 ms / 10 ms frames, `dither=0` → deterministic, Povey window,
//!   remove-DC-per-frame, pre-emphasis 0.97, power spectrum, HTK-mel `1127·ln(1+f/700)` triangular
//!   bank over a 512-pt FFT, `low_freq=20`, `high_freq=nyquist`, `log(max(·, f32::EPSILON))`), then
//!   the CosyVoice per-utterance **CMN** (`feat − feat.mean(time)`). All host-side `f32`, exact.
//! - **FCM strides only the frequency axis** (`stride=(s,1)`): reproduced as a stride-1 conv + an
//!   even-index subsample on the freq dim (bit-identical to a strided conv), so the fbank time
//!   dimension survives the head unchanged.
//! - **CAM layer**: `y = linear_local(x); m = σ(linear2(relu(linear1(mean_T(x) + segpool(x)))))`,
//!   `segpool` = 100-frame average pooled then broadcast back (ceil-mode, partial last segment
//!   averaged over its real length). Segment pooling is expressed as two small matmuls.
//! - **StatsPool**: mean and **unbiased** (`N−1`) std over time, concatenated `[mean‖std]`.
//! - The final `dense` nonlinearity is `batchnorm_` — a **BatchNorm1d with `affine=False`** (no
//!   γ/β), so only its running mean/var are read; there is no output ReLU.

use std::f32::consts::PI;
use std::path::Path;

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_audio::{AudioError, Result};
use candle_nn::ops::sigmoid;
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, conv2d_no_bias, linear, BatchNorm, BatchNormConfig, Conv1d,
    Conv1dConfig, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder,
};

use crate::config::S3_SR;
use crate::s3gen::S3GEN_WEIGHTS_FILE;
use crate::s3tokenizer::resample_to_16k;

// --------------------------------------------------------------------------------------------
// Architecture constants (transcribed from `CAMPPlus.__init__` / the FCM / the trunk builder).
// --------------------------------------------------------------------------------------------

/// Input mel-feature dimension (`feat_dim`, the Kaldi fbank bin count).
pub const FEAT_DIM: usize = 80;
/// Produced x-vector width (`embedding_size`).
pub const XVECTOR_DIM: usize = 192;
/// Flow-ready speaker-embedding width (`flow.spk_embed_affine_layer` output).
pub const SPK_EMBED_DIM: usize = 80;
/// D-TDNN dense growth per layer (`growth_rate`).
const GROWTH_RATE: usize = 32;
/// D-TDNN bottleneck width (`bn_size · growth_rate` = 4·32).
const BN_CHANNELS: usize = 128;
/// Trunk stem output width (`init_channels`).
const INIT_CHANNELS: usize = 128;
/// FCM base channel count (`m_channels`).
const M_CHANNELS: usize = 32;
/// `(num_layers, dilation)` for `block{1,2,3}` (kernel is 3 throughout).
const BLOCKS: [(usize, usize); 3] = [(12, 1), (24, 2), (16, 2)];
/// PyTorch BatchNorm default epsilon (BatchNorm1d/2d).
const BN_EPS: f64 = 1e-5;
/// CAM segment-pooling window (frames).
const SEG_LEN: usize = 100;

// --- Kaldi fbank parameters (torchaudio defaults + num_mel_bins=80, at 16 kHz) ---
/// Frame length in samples (25 ms @ 16 kHz).
const FRAME_LEN: usize = 400;
/// Frame shift in samples (10 ms @ 16 kHz).
const FRAME_SHIFT: usize = 160;
/// FFT size (`round_to_power_of_two(400)`).
const N_FFT: usize = 512;
/// One-sided FFT bins (`N_FFT/2 + 1`).
const N_BINS: usize = N_FFT / 2 + 1;
/// Pre-emphasis coefficient.
const PREEMPH: f32 = 0.97;
/// Mel low cutoff (Hz).
const LOW_FREQ: f32 = 20.0;
/// Mel high cutoff (Hz) — the 16 kHz Nyquist.
const HIGH_FREQ: f32 = 8000.0;

// --------------------------------------------------------------------------------------------
// Kaldi fbank front-end (host f32; deterministic — no dither).
// --------------------------------------------------------------------------------------------

/// Kaldi HTK mel scale: `1127 · ln(1 + f/700)`.
fn mel_scale(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Povey window of length `n` — `hann(n, periodic=False)^0.85` (`hann[i] = 0.5 − 0.5·cos(2πi/(n−1))`).
fn povey_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = 0.5 - 0.5 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos();
            h.powf(0.85)
        })
        .collect()
}

/// Kaldi triangular mel filterbank, mel-major `[FEAT_DIM][N_BINS]` (the final Nyquist bin column is
/// the zero-pad Kaldi appends). Matches `get_mel_banks(80, 512, 16000, 20, 8000)`.
fn mel_filterbank() -> Vec<f32> {
    let num_fft_bins = N_FFT / 2; // 256 (the Nyquist bin is the zero-padded column)
    let fft_bin_width = S3_SR as f32 / N_FFT as f32; // 31.25 Hz
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(HIGH_FREQ);
    let delta = (mel_high - mel_low) / (FEAT_DIM as f32 + 1.0);
    // Mel of each analysed FFT bin (bins 0..255; bin 256 is the zero column).
    let mel_bin: Vec<f32> = (0..num_fft_bins)
        .map(|i| mel_scale(fft_bin_width * i as f32))
        .collect();
    let mut fb = vec![0f32; FEAT_DIM * N_BINS];
    for b in 0..FEAT_DIM {
        let left = mel_low + b as f32 * delta;
        let center = mel_low + (b as f32 + 1.0) * delta;
        let right = mel_low + (b as f32 + 2.0) * delta;
        for (i, &m) in mel_bin.iter().enumerate() {
            let up = (m - left) / (center - left);
            let down = (right - m) / (right - center);
            fb[b * N_BINS + i] = up.min(down).max(0.0);
        }
        // fb[b * N_BINS + 256] stays 0 (the appended Nyquist column).
    }
    fb
}

/// Precomputed fbank tables reused across frames and clips.
struct Fbank {
    window: Vec<f32>,
    mel_fb: Vec<f32>,  // `[FEAT_DIM][N_BINS]`
    cos_tab: Vec<f32>, // `[N_BINS][FRAME_LEN]`, angle over N_FFT
    sin_tab: Vec<f32>,
}

impl Fbank {
    fn new() -> Self {
        // DFT twiddles for a 512-pt transform, but only the first FRAME_LEN samples are non-zero
        // after windowing (the rest is the zero pad), so the tables only span FRAME_LEN columns.
        let mut cos_tab = vec![0f32; N_BINS * FRAME_LEN];
        let mut sin_tab = vec![0f32; N_BINS * FRAME_LEN];
        for k in 0..N_BINS {
            for t in 0..FRAME_LEN {
                let ang = -2.0 * PI * k as f32 * t as f32 / N_FFT as f32;
                cos_tab[k * FRAME_LEN + t] = ang.cos();
                sin_tab[k * FRAME_LEN + t] = ang.sin();
            }
        }
        Self {
            window: povey_window(FRAME_LEN),
            mel_fb: mel_filterbank(),
            cos_tab,
            sin_tab,
        }
    }

    /// Compute log-mel fbank frames for a 16 kHz waveform, then apply CosyVoice CMN
    /// (`feat − feat.mean(time)`). Returns frame-major `[n_frames][FEAT_DIM]`.
    fn compute(&self, wav: &[f32]) -> Vec<f32> {
        if wav.len() < FRAME_LEN {
            return Vec::new();
        }
        // snip_edges=True framing.
        let n_frames = 1 + (wav.len() - FRAME_LEN) / FRAME_SHIFT;
        let mut feat = vec![0f32; n_frames * FEAT_DIM]; // frame-major [t * FEAT_DIM + b]
        let mut frame = vec![0f32; FRAME_LEN];
        let mut power = vec![0f32; N_BINS];
        let eps = f32::EPSILON; // torch.finfo(float32).eps
        for f in 0..n_frames {
            let start = f * FRAME_SHIFT;
            let src = &wav[start..start + FRAME_LEN];
            // remove_dc_offset: subtract the per-frame mean.
            let mean = src.iter().sum::<f32>() / FRAME_LEN as f32;
            // pre-emphasis on the DC-removed signal: x[j] -= 0.97·x[max(0, j-1)].
            let prev0 = src[0] - mean;
            frame[0] = prev0 - PREEMPH * prev0;
            let mut prev = prev0;
            for j in 1..FRAME_LEN {
                let cur = src[j] - mean;
                frame[j] = (cur - PREEMPH * prev) * self.window[j];
                prev = cur;
            }
            frame[0] *= self.window[0];
            // Power spectrum via a direct real-DFT over the 512-pt (zero-padded) windowed frame.
            for (k, p) in power.iter_mut().enumerate() {
                let (cos_k, sin_k) = (
                    &self.cos_tab[k * FRAME_LEN..],
                    &self.sin_tab[k * FRAME_LEN..],
                );
                let mut re = 0f32;
                let mut im = 0f32;
                for (t, &x) in frame.iter().enumerate() {
                    re += x * cos_k[t];
                    im += x * sin_k[t];
                }
                *p = re * re + im * im;
            }
            // Mel projection + log(max(·, eps)).
            for b in 0..FEAT_DIM {
                let filt = &self.mel_fb[b * N_BINS..(b + 1) * N_BINS];
                let mut acc = 0f32;
                for (k, &w) in filt.iter().enumerate() {
                    acc += w * power[k];
                }
                feat[f * FEAT_DIM + b] = acc.max(eps).ln();
            }
        }
        // CMN: subtract each mel bin's mean over time.
        for b in 0..FEAT_DIM {
            let mut m = 0f32;
            for f in 0..n_frames {
                m += feat[f * FEAT_DIM + b];
            }
            m /= n_frames as f32;
            for f in 0..n_frames {
                feat[f * FEAT_DIM + b] -= m;
            }
        }
        feat
    }
}

// --------------------------------------------------------------------------------------------
// Model modules.
// --------------------------------------------------------------------------------------------

/// `get_nonlinear("batchnorm-relu")` — a BatchNorm1d followed by ReLU (eval-mode forward).
struct BnRelu {
    bn: BatchNorm,
}

impl BnRelu {
    fn load(channels: usize, vb: VarBuilder) -> CandleResult<Self> {
        // The Sequential submodule is named `batchnorm` inside the `nonlinear*` prefix.
        let bn = batch_norm(channels, bn_cfg(true), vb.pp("batchnorm"))?;
        Ok(Self { bn })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.bn.forward_t(x, false)?.relu()
    }
}

fn bn_cfg(affine: bool) -> BatchNormConfig {
    BatchNormConfig {
        eps: BN_EPS,
        affine,
        ..Default::default()
    }
}

fn conv1d_cfg(stride: usize, padding: usize, dilation: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding,
        stride,
        dilation,
        groups: 1,
        cudnn_fwd_algo: None,
    }
}

/// A BasicResBlock (2-D) whose convs stride **only the frequency axis** (`stride=(s,1)`), realized
/// as stride-1 convs plus an even-index freq subsample.
struct BasicResBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
    freq_stride: usize,
}

impl BasicResBlock {
    fn load(in_ch: usize, planes: usize, stride: usize, vb: VarBuilder) -> CandleResult<Self> {
        let c3 = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv2d_no_bias(in_ch, planes, 3, c3, vb.pp("conv1"))?;
        let bn1 = batch_norm(planes, bn_cfg(true), vb.pp("bn1"))?;
        let conv2 = conv2d_no_bias(planes, planes, 3, c3, vb.pp("conv2"))?;
        let bn2 = batch_norm(planes, bn_cfg(true), vb.pp("bn2"))?;
        let shortcut = if stride != 1 || in_ch != planes {
            let c1 = Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            let sc = vb.pp("shortcut");
            Some((
                conv2d_no_bias(in_ch, planes, 1, c1, sc.pp("0"))?,
                batch_norm(planes, bn_cfg(true), sc.pp("1"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
            freq_stride: stride,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Main path: conv1 strides freq, conv2 does not.
        let out = subsample_freq(&self.conv1.forward(x)?, self.freq_stride)?;
        let out = self.bn1.forward_t(&out, false)?.relu()?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward_t(&out, false)?;
        // Residual (the shortcut also strides freq).
        let sc = match &self.shortcut {
            Some((conv, bn)) => {
                let s = subsample_freq(&conv.forward(x)?, self.freq_stride)?;
                bn.forward_t(&s, false)?
            }
            None => x.clone(),
        };
        out.add(&sc)?.relu()
    }
}

/// Even-index subsample along the frequency axis (dim 2 of `[B,C,F,T]`) — bit-identical to a
/// stride-`step` convolution's output when applied to the stride-1 conv result.
fn subsample_freq(x: &Tensor, step: usize) -> CandleResult<Tensor> {
    if step <= 1 {
        return Ok(x.clone());
    }
    let n = x.dim(2)?;
    let idx: Vec<u32> = (0..n as u32).step_by(step).collect();
    let idx = Tensor::from_vec(idx.clone(), idx.len(), x.device())?;
    x.index_select(&idx, 2)?.contiguous()
}

/// The FCM frequency-context front-end.
struct Fcm {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2d,
    bn2: BatchNorm,
    out_channels: usize,
}

impl Fcm {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let c3 = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv2d_no_bias(1, M_CHANNELS, 3, c3, vb.pp("conv1"))?;
        let bn1 = batch_norm(M_CHANNELS, bn_cfg(true), vb.pp("bn1"))?;
        let make_layer = |name: &str, vb: &VarBuilder| -> CandleResult<Vec<BasicResBlock>> {
            // 2 blocks; the first strides freq ×2, the second is stride 1.
            let l = vb.pp(name);
            Ok(vec![
                BasicResBlock::load(M_CHANNELS, M_CHANNELS, 2, l.pp("0"))?,
                BasicResBlock::load(M_CHANNELS, M_CHANNELS, 1, l.pp("1"))?,
            ])
        };
        let layer1 = make_layer("layer1", &vb)?;
        let layer2 = make_layer("layer2", &vb)?;
        let conv2 = conv2d_no_bias(M_CHANNELS, M_CHANNELS, 3, c3, vb.pp("conv2"))?;
        let bn2 = batch_norm(M_CHANNELS, bn_cfg(true), vb.pp("bn2"))?;
        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            conv2,
            bn2,
            out_channels: M_CHANNELS * (FEAT_DIM / 8),
        })
    }

    /// `[B, F, T]` → `[B, 320, T]` (frequency downsampled ×8, time preserved).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = x.unsqueeze(1)?; // [B, 1, F, T]
        let mut out = self
            .bn1
            .forward_t(&self.conv1.forward(&x)?, false)?
            .relu()?;
        for b in &self.layer1 {
            out = b.forward(&out)?;
        }
        for b in &self.layer2 {
            out = b.forward(&out)?;
        }
        // conv2 strides freq ×2.
        let out = subsample_freq(&self.conv2.forward(&out)?, 2)?;
        let out = self.bn2.forward_t(&out, false)?.relu()?;
        let (b, c, f, t) = out.dims4()?;
        out.reshape((b, c * f, t))
    }
}

/// The trunk stem `TDNNLayer(320→128, k5, s2)` (BatchNorm+ReLU nonlinearity).
struct Tdnn {
    linear: Conv1d,
    nonlinear: BnRelu,
}

impl Tdnn {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        // kernel 5, stride 2, dilation 1, "equal" padding (5-1)//2 = 2.
        let linear = conv1d_no_bias(320, INIT_CHANNELS, 5, conv1d_cfg(2, 2, 1), vb.pp("linear"))?;
        let nonlinear = BnRelu::load(INIT_CHANNELS, vb.pp("nonlinear"))?;
        Ok(Self { linear, nonlinear })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.nonlinear.forward(&self.linear.forward(x)?)
    }
}

/// The CAM (context-aware masking) layer inside each dense D-TDNN layer.
struct CamLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CamLayer {
    fn load(bn_ch: usize, out_ch: usize, dilation: usize, vb: VarBuilder) -> CandleResult<Self> {
        // kernel 3, stride 1, "equal" padding (3-1)//2·dilation.
        let pad = dilation;
        let linear_local = conv1d_no_bias(
            bn_ch,
            out_ch,
            3,
            conv1d_cfg(1, pad, dilation),
            vb.pp("linear_local"),
        )?;
        // Point-wise reduction/expansion carry a bias (Conv1d default).
        let linear1 = conv1d(bn_ch, bn_ch / 2, 1, conv1d_cfg(1, 0, 1), vb.pp("linear1"))?;
        let linear2 = conv1d(bn_ch / 2, out_ch, 1, conv1d_cfg(1, 0, 1), vb.pp("linear2"))?;
        Ok(Self {
            linear_local,
            linear1,
            linear2,
        })
    }

    /// `x [B, bn_ch, T]` → `[B, out_ch, T]`. `seg` is the shared segment-pooling operator for the
    /// current time length.
    fn forward(&self, x: &Tensor, seg: &SegPool) -> CandleResult<Tensor> {
        let y = self.linear_local.forward(x)?; // [B, out, T]
        let global = x.mean_keepdim(D::Minus1)?; // [B, bn_ch, 1]
        let context = seg.forward(x)?.broadcast_add(&global)?; // [B, bn_ch, T]
        let context = self.linear1.forward(&context)?.relu()?; // [B, bn_ch/2, T]
        let m = sigmoid(&self.linear2.forward(&context)?)?; // [B, out, T]
        y.mul(&m)
    }
}

/// One dense D-TDNN layer: `cam(nonlinear2(linear1(nonlinear1(x))))` producing `growth_rate` new
/// channels.
struct DenseTdnnLayer {
    nonlinear1: BnRelu,
    linear1: Conv1d,
    nonlinear2: BnRelu,
    cam: CamLayer,
}

impl DenseTdnnLayer {
    fn load(in_ch: usize, dilation: usize, vb: VarBuilder) -> CandleResult<Self> {
        let nonlinear1 = BnRelu::load(in_ch, vb.pp("nonlinear1"))?;
        let linear1 = conv1d_no_bias(in_ch, BN_CHANNELS, 1, conv1d_cfg(1, 0, 1), vb.pp("linear1"))?;
        let nonlinear2 = BnRelu::load(BN_CHANNELS, vb.pp("nonlinear2"))?;
        let cam = CamLayer::load(BN_CHANNELS, GROWTH_RATE, dilation, vb.pp("cam_layer"))?;
        Ok(Self {
            nonlinear1,
            linear1,
            nonlinear2,
            cam,
        })
    }
    fn forward(&self, x: &Tensor, seg: &SegPool) -> CandleResult<Tensor> {
        let h = self.nonlinear1.forward(x)?;
        let h = self.linear1.forward(&h)?;
        let h = self.nonlinear2.forward(&h)?;
        self.cam.forward(&h, seg)
    }
}

/// A dense block: each layer's output is channel-concatenated onto the running feature map.
struct DenseBlock {
    layers: Vec<DenseTdnnLayer>,
}

impl DenseBlock {
    fn load(
        num_layers: usize,
        in_ch: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let li = DenseTdnnLayer::load(
                in_ch + i * GROWTH_RATE,
                dilation,
                vb.pp(format!("tdnnd{}", i + 1)),
            )?;
            layers.push(li);
        }
        Ok(Self { layers })
    }
    fn forward(&self, x: &Tensor, seg: &SegPool) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            let out = layer.forward(&x, seg)?;
            x = Tensor::cat(&[&x, &out], 1)?;
        }
        Ok(x)
    }
}

/// A transition layer `linear(nonlinear(x))` halving the channel count (no linear bias).
struct TransitLayer {
    nonlinear: BnRelu,
    linear: Conv1d,
}

impl TransitLayer {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> CandleResult<Self> {
        let nonlinear = BnRelu::load(in_ch, vb.pp("nonlinear"))?;
        let linear = conv1d_no_bias(in_ch, out_ch, 1, conv1d_cfg(1, 0, 1), vb.pp("linear"))?;
        Ok(Self { nonlinear, linear })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.linear.forward(&self.nonlinear.forward(x)?)
    }
}

/// The `dense` head: `Conv1d(1024→192)` then a **non-affine** BatchNorm1d (`batchnorm_`, no ReLU).
struct DenseHead {
    linear: Conv1d,
    bn: BatchNorm,
}

impl DenseHead {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> CandleResult<Self> {
        let linear = conv1d_no_bias(in_ch, out_ch, 1, conv1d_cfg(1, 0, 1), vb.pp("linear"))?;
        let bn = batch_norm(out_ch, bn_cfg(false), vb.pp("nonlinear").pp("batchnorm"))?;
        Ok(Self { linear, bn })
    }
    /// `stats [B, 2C]` → `[B, out_ch]`.
    fn forward(&self, stats: &Tensor) -> CandleResult<Tensor> {
        let x = self.linear.forward(&stats.unsqueeze(D::Minus1)?)?; // [B, out, 1]
        let x = self.bn.forward_t(&x, false)?;
        x.squeeze(D::Minus1)
    }
}

/// Segment-pooling operator for a fixed time length `t`: `segpool(x)[·, i] = mean over the 100-frame
/// segment containing frame i`. Realized as `x @ down_t @ up` with two precomputed `f32` matrices.
struct SegPool {
    down_t: Tensor, // [T, num_seg] — per-segment averaging (normalized by real segment length)
    up: Tensor,     // [num_seg, T] — broadcast a segment value back over its frames
}

impl SegPool {
    fn new(t: usize, device: &Device) -> CandleResult<Self> {
        let num_seg = t.div_ceil(SEG_LEN);
        let mut down = vec![0f32; t * num_seg]; // [T, num_seg]
        let mut up = vec![0f32; num_seg * t]; // [num_seg, T]
        for s in 0..num_seg {
            let start = s * SEG_LEN;
            let end = ((s + 1) * SEG_LEN).min(t);
            let inv = 1.0 / (end - start) as f32; // ceil-mode: partial last segment / its real length
            for f in start..end {
                down[f * num_seg + s] = inv;
                up[s * t + f] = 1.0;
            }
        }
        Ok(Self {
            down_t: Tensor::from_vec(down, (t, num_seg), device)?,
            up: Tensor::from_vec(up, (num_seg, t), device)?,
        })
    }

    /// `x [B, C, T]` (B = 1) → `[B, C, T]` piecewise-constant segment means.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, c, t) = x.dims3()?;
        let x2 = x.reshape((c, t))?; // B == 1 in inference
        let seg_mean = x2.matmul(&self.down_t)?; // [C, num_seg]
        let seg_bc = seg_mean.matmul(&self.up)?; // [C, T]
        seg_bc.reshape((b, c, t))
    }
}

/// Mean‖unbiased-std statistics pooling over the time axis. `x [B, C, T]` → `[B, 2C]`.
fn stats_pool(x: &Tensor) -> CandleResult<Tensor> {
    let t = x.dim(D::Minus1)?;
    let mean = x.mean(D::Minus1)?; // [B, C]
    let centered = x.broadcast_sub(&mean.unsqueeze(D::Minus1)?)?;
    let var = (centered.sqr()?.sum(D::Minus1)? / (t as f64 - 1.0))?; // unbiased
    let std = var.sqrt()?;
    Tensor::cat(&[&mean, &std], D::Minus1)
}

// --------------------------------------------------------------------------------------------
// The loaded encoder.
// --------------------------------------------------------------------------------------------

/// The CAMPPlus speaker encoder plus the S3Gen flow's speaker-embedding affine head.
pub struct Campplus {
    fbank: Fbank,
    head: Fcm,
    tdnn: Tdnn,
    block1: DenseBlock,
    transit1: TransitLayer,
    block2: DenseBlock,
    transit2: TransitLayer,
    block3: DenseBlock,
    transit3: TransitLayer,
    out_nonlinear: BnRelu,
    dense: DenseHead,
    /// `flow.spk_embed_affine_layer` — Linear 192→80 the flow applies after L2-normalization.
    spk_affine: Linear,
    device: Device,
}

impl Campplus {
    /// Load from a Chatterbox snapshot directory (reads `s3gen.safetensors`, prefixes
    /// `speaker_encoder.*` for the encoder and `flow.spk_embed_affine_layer.*` for the 80-d head).
    pub fn from_snapshot(dir: &Path) -> Result<Self> {
        let path = dir.join(S3GEN_WEIGHTS_FILE);
        if !path.is_file() {
            return Err(AudioError::Msg(format!(
                "campplus: {} missing (the speaker-encoder weights live in the S3Gen checkpoint)",
                path.display()
            )));
        }
        let device = candle_audio::default_device()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, &device)?
        };
        Ok(Self::new(vb, device)?)
    }

    /// Build the encoder from a safetensors-root [`VarBuilder`] (the tensors are under
    /// `speaker_encoder.*` and `flow.spk_embed_affine_layer.*`).
    pub fn new(vb: VarBuilder, device: Device) -> CandleResult<Self> {
        let se = vb.pp("speaker_encoder");
        let head = Fcm::load(se.pp("head"))?;
        let xv = se.pp("xvector");
        let tdnn = Tdnn::load(xv.pp("tdnn"))?;
        // Track channel growth exactly as the reference builder does.
        let mut ch = INIT_CHANNELS;
        let (n1, d1) = BLOCKS[0];
        let block1 = DenseBlock::load(n1, ch, d1, xv.pp("block1"))?;
        ch += n1 * GROWTH_RATE;
        let transit1 = TransitLayer::load(ch, ch / 2, xv.pp("transit1"))?;
        ch /= 2;
        let (n2, d2) = BLOCKS[1];
        let block2 = DenseBlock::load(n2, ch, d2, xv.pp("block2"))?;
        ch += n2 * GROWTH_RATE;
        let transit2 = TransitLayer::load(ch, ch / 2, xv.pp("transit2"))?;
        ch /= 2;
        let (n3, d3) = BLOCKS[2];
        let block3 = DenseBlock::load(n3, ch, d3, xv.pp("block3"))?;
        ch += n3 * GROWTH_RATE;
        let transit3 = TransitLayer::load(ch, ch / 2, xv.pp("transit3"))?;
        ch /= 2;
        let out_nonlinear = BnRelu::load(ch, xv.pp("out_nonlinear"))?;
        let dense = DenseHead::load(ch * 2, XVECTOR_DIM, xv.pp("dense"))?;
        let spk_affine = linear(
            XVECTOR_DIM,
            SPK_EMBED_DIM,
            vb.pp("flow").pp("spk_embed_affine_layer"),
        )?;
        debug_assert_eq!(head.out_channels, 320);
        debug_assert_eq!(ch, 512);
        Ok(Self {
            fbank: Fbank::new(),
            head,
            tdnn,
            block1,
            transit1,
            block2,
            transit2,
            block3,
            transit3,
            out_nonlinear,
            dense,
            spk_affine,
            device,
        })
    }

    /// The raw 192-d CAMPPlus x-vector for a reference waveform (resampled to 16 kHz if needed).
    pub fn embed(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let v = self.embed_tensor(samples, sample_rate)?;
        Ok(v.flatten_all()?.to_vec1::<f32>()?)
    }

    /// The 80-d **flow-ready** speaker embedding: the 192-d x-vector L2-normalized, then the
    /// `flow.spk_embed_affine_layer` (Linear 192→80) the S3Gen flow consumes.
    pub fn spk_embed_flow(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let x = self.embed_tensor(samples, sample_rate)?; // [1, 192]
        let x = l2_normalize_tensor(&x)?;
        let x = self.spk_affine.forward(&x)?; // [1, 80]
        Ok(x.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Reference waveform → `[1, 192]` x-vector tensor.
    fn embed_tensor(&self, samples: &[f32], sample_rate: u32) -> CandleResult<Tensor> {
        let wav = resample_to_16k(samples, sample_rate);
        let feat = self.fbank.compute(&wav); // frame-major [T][FEAT_DIM], CMN applied
        let n_frames = feat.len() / FEAT_DIM;
        if n_frames == 0 {
            return Err(candle_audio::candle_core::Error::Msg(
                "campplus: reference clip too short to produce any fbank frame".into(),
            ));
        }
        // Model input is (B, T, F).
        let x = Tensor::from_vec(feat, (1, n_frames, FEAT_DIM), &self.device)?;
        self.forward(&x)
    }

    /// `(B, T, F)` fbank → `(B, 192)` x-vector.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = x.permute((0, 2, 1))?.contiguous()?; // (B, F, T)
        let x = self.head.forward(&x)?; // (B, 320, T)
        let x = self.tdnn.forward(&x)?; // (B, 128, T')
                                        // The segment-pooling operator depends only on the (shared) time length.
        let t = x.dim(D::Minus1)?;
        let seg = SegPool::new(t, &self.device)?;
        let x = self.block1.forward(&x, &seg)?;
        let x = self.transit1.forward(&x)?;
        let x = self.block2.forward(&x, &seg)?;
        let x = self.transit2.forward(&x)?;
        let x = self.block3.forward(&x, &seg)?;
        let x = self.transit3.forward(&x)?;
        let x = self.out_nonlinear.forward(&x)?;
        let stats = stats_pool(&x)?; // (B, 1024)
        self.dense.forward(&stats) // (B, 192)
    }
}

/// L2-normalize `x [1, N]` along the feature axis (`F.normalize(dim=1)`, eps 1e-12).
fn l2_normalize_tensor(x: &Tensor) -> CandleResult<Tensor> {
    let norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::INFINITY as f64)?;
    x.broadcast_div(&norm)
}

/// L2-normalize a host vector (zero-safe).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity of two equal-length vectors (the discriminative-conformance measure).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn povey_window_shape_and_bounds() {
        let w = povey_window(FRAME_LEN);
        assert_eq!(w.len(), FRAME_LEN);
        assert!(w[0].abs() < 1e-6, "Povey window vanishes at the edges");
        assert!(w.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // Peak near the centre.
        assert!(w[FRAME_LEN / 2] > 0.99);
    }

    #[test]
    fn mel_scale_is_monotone_htk() {
        assert!((mel_scale(0.0)).abs() < 1e-6);
        assert!(mel_scale(1000.0) > mel_scale(100.0));
        // HTK reference point: mel(1000) = 1127·ln(1+1000/700) ≈ 999.9.
        assert!((mel_scale(1000.0) - 999.98553).abs() < 1e-2);
    }

    #[test]
    fn mel_filterbank_shape_nonneg_and_zero_nyquist_column() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), FEAT_DIM * N_BINS);
        for b in 0..FEAT_DIM {
            let row = &fb[b * N_BINS..(b + 1) * N_BINS];
            assert!(row.iter().all(|&w| w >= 0.0));
            assert!(row.iter().any(|&w| w > 0.0), "mel bin {b} is all-zero");
            // The appended Nyquist column is always zero.
            assert_eq!(row[N_BINS - 1], 0.0);
        }
    }

    #[test]
    fn fbank_frame_count_and_cmn_zero_mean() {
        let fb = Fbank::new();
        // 1 s of a 220 Hz tone at 16 kHz.
        let sr = S3_SR as usize;
        let tone: Vec<f32> = (0..sr)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / sr as f32).sin() * 0.3)
            .collect();
        let feat = fb.compute(&tone);
        let n_frames = feat.len() / FEAT_DIM;
        // snip_edges: 1 + (16000-400)/160 = 98 frames.
        assert_eq!(n_frames, 1 + (sr - FRAME_LEN) / FRAME_SHIFT);
        assert!(feat.iter().all(|v| v.is_finite()));
        // CMN makes every mel bin zero-mean over time.
        for b in 0..FEAT_DIM {
            let m: f32 =
                (0..n_frames).map(|f| feat[f * FEAT_DIM + b]).sum::<f32>() / n_frames as f32;
            assert!(m.abs() < 1e-3, "mel bin {b} mean {m} not ~0 after CMN");
        }
    }

    #[test]
    fn subsample_freq_takes_even_indices() {
        let dev = Device::Cpu;
        // [1,1,4,2] with freq values 0..3 along dim 2.
        let x = Tensor::arange(0f32, 8f32, &dev)
            .unwrap()
            .reshape((1, 1, 4, 2))
            .unwrap();
        let y = subsample_freq(&x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        // Rows 0 and 2 survive → values [0,1] and [4,5].
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn seg_pool_is_piecewise_segment_mean() {
        let dev = Device::Cpu;
        // T = 150 → two segments: [0,100) and [100,150).
        let t = 150usize;
        let seg = SegPool::new(t, &dev).unwrap();
        // x[c=0, i] = i.
        let x = Tensor::arange(0f32, t as f32, &dev)
            .unwrap()
            .reshape((1, 1, t))
            .unwrap();
        let out = seg.forward(&x).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Segment 0 mean = mean(0..99) = 49.5; segment 1 mean = mean(100..149) = 124.5.
        assert!((v[0] - 49.5).abs() < 1e-3);
        assert!((v[99] - 49.5).abs() < 1e-3);
        assert!((v[100] - 124.5).abs() < 1e-3);
        assert!((v[149] - 124.5).abs() < 1e-3);
    }

    #[test]
    fn stats_pool_mean_and_unbiased_std_dims() {
        let dev = Device::Cpu;
        // Channel 0 = [1,2,3,4,5]: mean 3, unbiased std = sqrt(2.5) ≈ 1.5811.
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5.], (1, 1, 5), &dev).unwrap();
        let s = stats_pool(&x).unwrap();
        assert_eq!(s.dims(), &[1, 2]);
        let v: Vec<f32> = s.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - 3.0).abs() < 1e-5);
        assert!((v[1] - 2.5f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn l2_normalize_unit_and_cosine_bounds() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    /// End-to-end shape/wiring on synthetic weights: build the full CAMPPlus from an in-memory
    /// VarBuilder of the real tensor shapes and confirm a 192-d x-vector + an 80-d flow embedding.
    /// Real weights are exercised by the `#[ignore]`d conformance test.
    #[test]
    fn model_wires_end_to_end_with_synthetic_weights() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        materialize_shapes(&vb);
        let model = Campplus::new(vb, device).unwrap();

        // 2 s of a 200 Hz tone at 16 kHz → a full trunk pass.
        let sr = S3_SR;
        let tone: Vec<f32> = (0..2 * sr as usize)
            .map(|i| (2.0 * PI * 200.0 * i as f32 / sr as f32).sin() * 0.2)
            .collect();
        let x = model.embed(&tone, sr).unwrap();
        assert_eq!(x.len(), XVECTOR_DIM);
        assert!(x.iter().all(|v| v.is_finite()));
        let flow = model.spk_embed_flow(&tone, sr).unwrap();
        assert_eq!(flow.len(), SPK_EMBED_DIM);
        assert!(flow.iter().all(|v| v.is_finite()));
    }

    /// Materialize every `speaker_encoder.*` + `flow.spk_embed_affine_layer.*` tensor the model
    /// reads, at its real shape (VarMap fills them with the default initializer).
    fn materialize_shapes(vb: &VarBuilder) {
        let bn = |vb: &VarBuilder, c: usize, affine: bool| {
            let _ = vb.get(c, "running_mean").unwrap();
            let _ = vb.get(c, "running_var").unwrap();
            if affine {
                let _ = vb.get(c, "weight").unwrap();
                let _ = vb.get(c, "bias").unwrap();
            }
        };
        let se = vb.pp("speaker_encoder");
        // head
        let h = se.pp("head");
        let _ = h.get((M_CHANNELS, 1, 3, 3), "conv1.weight").unwrap();
        bn(&h.pp("bn1"), M_CHANNELS, true);
        let _ = h
            .get((M_CHANNELS, M_CHANNELS, 3, 3), "conv2.weight")
            .unwrap();
        bn(&h.pp("bn2"), M_CHANNELS, true);
        for lname in ["layer1", "layer2"] {
            let l = h.pp(lname);
            for i in 0..2 {
                let b = l.pp(i.to_string());
                let _ = b
                    .get((M_CHANNELS, M_CHANNELS, 3, 3), "conv1.weight")
                    .unwrap();
                bn(&b.pp("bn1"), M_CHANNELS, true);
                let _ = b
                    .get((M_CHANNELS, M_CHANNELS, 3, 3), "conv2.weight")
                    .unwrap();
                bn(&b.pp("bn2"), M_CHANNELS, true);
                if i == 0 {
                    let _ = b
                        .get((M_CHANNELS, M_CHANNELS, 1, 1), "shortcut.0.weight")
                        .unwrap();
                    bn(&b.pp("shortcut").pp("1"), M_CHANNELS, true);
                }
            }
        }
        // xvector
        let xv = se.pp("xvector");
        let _ = xv
            .get((INIT_CHANNELS, 320, 5), "tdnn.linear.weight")
            .unwrap();
        bn(&xv.pp("tdnn").pp("nonlinear"), INIT_CHANNELS, true);
        let mut ch = INIT_CHANNELS;
        for (bi, (n, _)) in BLOCKS.iter().enumerate() {
            let blk = xv.pp(format!("block{}", bi + 1));
            for i in 0..*n {
                let l = blk.pp(format!("tdnnd{}", i + 1));
                let inc = ch + i * GROWTH_RATE;
                bn(&l.pp("nonlinear1"), inc, true);
                let _ = l.get((BN_CHANNELS, inc, 1), "linear1.weight").unwrap();
                bn(&l.pp("nonlinear2"), BN_CHANNELS, true);
                let cam = l.pp("cam_layer");
                let _ = cam
                    .get((GROWTH_RATE, BN_CHANNELS, 3), "linear_local.weight")
                    .unwrap();
                let _ = cam
                    .get((BN_CHANNELS / 2, BN_CHANNELS, 1), "linear1.weight")
                    .unwrap();
                let _ = cam.get(BN_CHANNELS / 2, "linear1.bias").unwrap();
                let _ = cam
                    .get((GROWTH_RATE, BN_CHANNELS / 2, 1), "linear2.weight")
                    .unwrap();
                let _ = cam.get(GROWTH_RATE, "linear2.bias").unwrap();
            }
            ch += n * GROWTH_RATE;
            let tr = xv.pp(format!("transit{}", bi + 1));
            bn(&tr.pp("nonlinear"), ch, true);
            let _ = tr.get((ch / 2, ch, 1), "linear.weight").unwrap();
            ch /= 2;
        }
        bn(&xv.pp("out_nonlinear"), ch, true);
        let _ = xv
            .get((XVECTOR_DIM, ch * 2, 1), "dense.linear.weight")
            .unwrap();
        bn(&xv.pp("dense").pp("nonlinear"), XVECTOR_DIM, false);
        // flow affine
        let fa = vb.pp("flow").pp("spk_embed_affine_layer");
        let _ = fa.get((SPK_EMBED_DIM, XVECTOR_DIM), "weight").unwrap();
        let _ = fa.get(SPK_EMBED_DIM, "bias").unwrap();
    }
}
