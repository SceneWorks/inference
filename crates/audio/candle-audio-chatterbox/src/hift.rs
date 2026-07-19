//! The **S3Gen HiFTNet vocoder** — `HiFTGenerator` (sc-13238). The **fourth and last** of S3Gen's
//! networks: it turns the flow's 80-bin, 50 Hz log-mel into a 24 kHz waveform. A faithful native
//! candle port of Chatterbox's `models/s3gen/hifigan.py` (`HiFTGenerator`, HiFTNet-style: an NSF
//! harmonic-plus-noise source + an iSTFTNet head) and `models/s3gen/f0_predictor.py`
//! (`ConvRNNF0Predictor`), instantiated with the Chatterbox kwargs from `models/s3gen/s3gen.py`.
//!
//! It is the `mel2wav.*` block of `s3gen.safetensors` (**328 tensors**). Its pieces:
//!
//! - **[`ConvRnnF0Predictor`]** (`mel2wav.f0_predictor.*`, 17 tensors) — a 5-layer weight-normed
//!   `Conv1d(k=3)` + ELU stack (`condnet`) then a `Linear(512 → 1)` `classifier`, returning
//!   `|f0|` per mel frame (Hz). Despite the "RNN" in the reference name it carries no recurrent
//!   weights — it is a pure conv stack.
//! - **`SourceModuleHnNSF`** (`mel2wav.m_source.*`, 2 tensors — the `l_linear` merge) — the NSF
//!   harmonic-plus-noise excitation, `nb_harmonics = 8`: the F0 (nearest-upsampled to the 24 kHz
//!   rate) drives a bank of `nb_harmonics + 1 = 9` phase-accumulated sines with random per-harmonic
//!   phase offsets, voiced/unvoiced gating (`f0 > 10 Hz`) and additive noise, merged by
//!   `tanh(Linear(9 → 1))` into a single excitation waveform.
//! - the **[`HiftGenerator`]** body: `conv_pre` `Conv1d(80 → 512, 7)`, a weight-normed
//!   `ConvTranspose1d` upsample stack (`ups`, `upsample_rates = [8, 5, 3]`,
//!   `upsample_kernel_sizes = [16, 11, 7]`), three MRF `resblocks` per upsample stage
//!   (`resblock_kernel_sizes = [3, 7, 11]`, Snake activations), the source-injection path
//!   (`source_downs` strided convs of the excitation's STFT + `source_resblocks`) added in at each
//!   stage, a `reflection_pad` before the last stage, and `conv_post` `Conv1d(64 → 18, 7)`.
//! - the **iSTFT head** (`istft_params = {n_fft: 16, hop_len: 4}`): `conv_post`'s 18 channels split
//!   into `exp`-magnitude (9) and `sin`-phase (9); a length-`(T'−1)·4` inverse STFT reconstructs the
//!   waveform, served by the shared radix-2 [`candle_audio::dsp::istft`] (`n_fft = 16` is a power of
//!   two). The product `prod(8·5·3)·4 = 480` samples per mel frame takes the 50 Hz mel to 24 kHz.
//!
//! The weight-normed convs are stored in the newer PyTorch parametrization format
//! (`parametrizations.weight.original0` = `g` `[out, 1, 1]`, `original1` = `v` `[out, in, k]`); the
//! dense weight `w = g · v / ‖v‖` is reconstructed by `weight_norm_weight`, the same math as
//! candle-transformers' `encodec::conv1d_weight_norm`, adapted to those key names. `ConvTranspose1d`
//! weight-norm normalizes per **input** channel (the `weight_norm` default `dim = 0` over that
//! tensor's `[in, out, k]` layout), which the shared reconstruction handles unchanged.
//!
//! The NSF source's random phase offsets and noise are drawn from a seeded [`StdRng`] (the gen-core
//! reproducibility law: same mel + seed ⇒ byte-identical waveform), rather than torch's global RNG.

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_audio::{dsp, AudioError, Result};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::path::Path;

use crate::config::S3GEN_SR;
use crate::s3gen::S3GEN_WEIGHTS_FILE;

/// Mel width (`in_channels`).
const MEL_DIM: usize = 80;
/// Vocoder trunk width (`base_channels`).
const BASE_CH: usize = 512;
/// Number of harmonic overtones above F0 (`nb_harmonics`); the sine bank has `NB_HARMONICS + 1`
/// rows (fundamental + overtones) and the `l_linear` merge is `Linear(NB_HARMONICS + 1 → 1)`.
const NB_HARMONICS: usize = 8;
/// NSF sine amplitude (`nsf_alpha`).
const NSF_SINE_AMP: f32 = 0.1;
/// NSF additive-noise std for voiced frames (`nsf_sigma`).
const NSF_NOISE_STD: f32 = 0.003;
/// F0 (Hz) above which a frame is voiced (`nsf_voiced_threshold`).
const NSF_VOICED_THRESHOLD: f32 = 10.0;
/// Upsample-stage LeakyReLU slope (`lrelu_slope`).
const LRELU_SLOPE: f64 = 0.1;
/// The default `F.leaky_relu` slope used once before `conv_post`.
const LRELU_SLOPE_DEFAULT: f64 = 0.01;
/// Output clamp (`audio_limit`).
const AUDIO_LIMIT: f32 = 0.99;
/// iSTFT window / FFT size (`istft_params["n_fft"]`).
const ISTFT_N_FFT: usize = 16;
/// iSTFT hop (`istft_params["hop_len"]`).
const ISTFT_HOP: usize = 4;
/// One-sided iSTFT bin count (`n_fft / 2 + 1`).
const ISTFT_BINS: usize = ISTFT_N_FFT / 2 + 1;
/// `conv_post`/`source_downs` channel width for the source STFT (`n_fft + 2` = re ++ im bins).
const SOURCE_CH: usize = ISTFT_N_FFT + 2;
/// The three upsample rates (`upsample_rates`).
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
/// The three upsample kernel sizes (`upsample_kernel_sizes`).
const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
/// MRF resblock kernel sizes per upsample stage (`resblock_kernel_sizes`).
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
/// Source-path resblock kernel sizes (`source_resblock_kernel_sizes`).
const SOURCE_RESBLOCK_KERNELS: [usize; 3] = [7, 7, 11];
/// Dilations shared by every resblock (`resblock_dilation_sizes[i]`).
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

// =================================================================================================
// weight-norm reconstruction + conv loaders
// =================================================================================================

/// `padding = (kernel · dilation − dilation) / 2` — the reference `get_padding` for 'same'-length
/// resblock convs.
const fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

/// Reconstruct a weight-normed dense weight `w = g · v / ‖v‖` from the parametrization tensors
/// `parametrizations.weight.original0` (`g`, `[d0, 1, 1]`) and `original1` (`v`, `[d0, d1, k]`),
/// with `‖v‖` taken per `d0`-slice (the `weight_norm` default `dim = 0`). Same math as
/// candle-transformers `encodec::conv1d_weight_norm`, keyed on the newer `parametrizations.*` names
/// this checkpoint uses. `d0` is `out` for `Conv1d` and `in` for `ConvTranspose1d`.
fn weight_norm_weight(vb: &VarBuilder, d0: usize, d1: usize, k: usize) -> CandleResult<Tensor> {
    let g = vb.get((d0, 1, 1), "parametrizations.weight.original0")?;
    let v = vb.get((d0, d1, k), "parametrizations.weight.original1")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_mul(&g)?.broadcast_div(&norm)
}

/// Load a weight-normed `Conv1d(in_c → out_c, k)` with the given config from a
/// `parametrizations.weight.*` + `bias`-keyed [`VarBuilder`].
fn load_wn_conv1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> CandleResult<Conv1d> {
    let weight = weight_norm_weight(&vb, out_c, in_c, k)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// Load a weight-normed `ConvTranspose1d(in_c → out_c, k)`. Its weight is `[in_c, out_c, k]` and the
/// weight-norm is per **input** channel, which [`weight_norm_weight`] handles with `d0 = in_c`.
fn load_wn_conv_transpose1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> CandleResult<ConvTranspose1d> {
    let weight = weight_norm_weight(&vb, in_c, out_c, k)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

/// Load a plain (no weight-norm) `Conv1d(in_c → out_c, k)` — the `source_downs` convs.
fn load_conv1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> CandleResult<Conv1d> {
    let weight = vb.get((out_c, in_c, k), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// LeakyReLU: `max(x, slope · x)` (`0 ≤ slope ≤ 1`).
fn leaky_relu(x: &Tensor, slope: f64) -> CandleResult<Tensor> {
    x.maximum(&x.affine(slope, 0.0)?)
}

/// `nn.ReflectionPad1d((1, 0))`: prepend the element at index 1 along the time axis (a 1-sample
/// reflection at the left boundary, edge sample excluded).
fn reflect_pad_left1(x: &Tensor) -> CandleResult<Tensor> {
    let left = x.narrow(2, 1, 1)?;
    Tensor::cat(&[&left, x], 2)
}

// =================================================================================================
// Snake activation
// =================================================================================================

/// `Snake(channels)`: `x + (1 / (α + 1e-9)) · sin(α · x)²`, `α` a learned per-channel parameter
/// (`alpha_logscale = False`, so `α` is used directly). Shape `[B, C, T]` → same.
struct Snake {
    alpha: Tensor, // [1, C, 1]
}

impl Snake {
    fn load(channels: usize, vb: VarBuilder) -> CandleResult<Self> {
        let alpha = vb.get(channels, "alpha")?.reshape((1, channels, 1))?;
        Ok(Self { alpha })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin2 = ax.sin()?.sqr()?;
        let denom = self.alpha.affine(1.0, 1e-9)?; // α + 1e-9
        x.broadcast_add(&sin2.broadcast_div(&denom)?)
    }
}

// =================================================================================================
// ResBlock (HiFiGAN/BigVGAN MRF block with Snake activations)
// =================================================================================================

/// `ResBlock(channels, kernel, dilations)`: three `(Snake, dilated Conv1d, Snake, Conv1d)` residual
/// sub-blocks. `convs1[i]` dilates by `dilations[i]`; `convs2[i]` is dilation-1; both keep length.
struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    fn load(
        channels: usize,
        kernel: usize,
        dilations: [usize; 3],
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut activations1 = Vec::with_capacity(3);
        let mut activations2 = Vec::with_capacity(3);
        for (idx, &dil) in dilations.iter().enumerate() {
            convs1.push(load_wn_conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, dil),
                    stride: 1,
                    dilation: dil,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp("convs1").pp(idx),
            )?);
            convs2.push(load_wn_conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, 1),
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp("convs2").pp(idx),
            )?);
            activations1.push(Snake::load(channels, vb.pp("activations1").pp(idx))?);
            activations2.push(Snake::load(channels, vb.pp("activations2").pp(idx))?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for idx in 0..self.convs1.len() {
            let xt = self.activations1[idx].forward(&x)?;
            let xt = self.convs1[idx].forward(&xt)?;
            let xt = self.activations2[idx].forward(&xt)?;
            let xt = self.convs2[idx].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

// =================================================================================================
// ConvRNNF0Predictor
// =================================================================================================

/// `ConvRNNF0Predictor` (`f0_predictor.*`): a 5-layer weight-normed `Conv1d(k=3, pad=1)` + ELU
/// `condnet` then `Linear(512 → 1)`, returning `|f0|` per mel frame (Hz). No recurrent weights.
pub struct ConvRnnF0Predictor {
    condnet: Vec<Conv1d>,
    classifier: Linear,
}

impl ConvRnnF0Predictor {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        // condnet = Sequential(conv, ELU, conv, ELU, ...) → the convs are at even indices 0,2,4,6,8.
        let mut condnet = Vec::with_capacity(5);
        condnet.push(load_wn_conv1d(
            MEL_DIM,
            BASE_CH,
            3,
            cfg,
            vb.pp("condnet").pp(0),
        )?);
        for i in 1..5 {
            condnet.push(load_wn_conv1d(
                BASE_CH,
                BASE_CH,
                3,
                cfg,
                vb.pp("condnet").pp(2 * i),
            )?);
        }
        let classifier = candle_nn::linear(BASE_CH, 1, vb.pp("classifier"))?;
        Ok(Self {
            condnet,
            classifier,
        })
    }

    /// `mel [1, 80, T]` → `f0 [1, T]` (Hz, non-negative).
    fn forward(&self, mel: &Tensor) -> CandleResult<Tensor> {
        let mut x = mel.clone();
        for conv in &self.condnet {
            x = conv.forward(&x)?.elu(1.0)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T, 512]
        let x = self.classifier.forward(&x)?; // [1, T, 1]
        x.squeeze(D::Minus1)?.abs() // [1, T]
    }
}

// =================================================================================================
// HiftGenerator
// =================================================================================================

/// The assembled HiFTNet vocoder: an [`ConvRnnF0Predictor`] + NSF harmonic source, a weight-normed
/// `ConvTranspose1d` upsample trunk with MRF resblocks and source injection, and an iSTFT head.
pub struct HiftGenerator {
    f0_predictor: ConvRnnF0Predictor,
    // SourceModuleHnNSF's `l_linear` (Linear(9 → 1)) kept as host scalars for the host-side NSF DSP.
    l_linear_weight: Vec<f32>, // [NB_HARMONICS + 1]
    l_linear_bias: f32,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    window: Vec<f32>, // periodic Hann, length ISTFT_N_FFT
    device: Device,
}

impl HiftGenerator {
    /// Load the vocoder from a Chatterbox snapshot directory (reads `s3gen.safetensors`, prefix
    /// `mel2wav.*`).
    pub fn from_snapshot(dir: &Path) -> Result<Self> {
        let path = dir.join(S3GEN_WEIGHTS_FILE);
        if !path.is_file() {
            return Err(AudioError::Msg(format!(
                "hift: {} missing (the vocoder weights live in the S3Gen checkpoint)",
                path.display()
            )));
        }
        let device = candle_audio::default_device()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, &device)?
        };
        Ok(Self::load(vb.pp("mel2wav"), device)?)
    }

    /// Build from a `mel2wav.*`-rooted [`VarBuilder`].
    pub fn load(vb: VarBuilder, device: Device) -> CandleResult<Self> {
        let f0_predictor = ConvRnnF0Predictor::load(vb.pp("f0_predictor"))?;

        let l_linear = vb.pp("m_source").pp("l_linear");
        let l_linear_weight = l_linear
            .get((1, NB_HARMONICS + 1), "weight")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let l_linear_bias = l_linear.get(1, "bias")?.to_vec1::<f32>()?[0];

        let conv_pre = load_wn_conv1d(
            MEL_DIM,
            BASE_CH,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv_pre"),
        )?;

        // Upsample trunk: ConvTranspose1d(base/2^i → base/2^(i+1)), stride = rate, pad = (k − u) / 2.
        let mut ups = Vec::with_capacity(3);
        for i in 0..3 {
            let in_c = BASE_CH >> i;
            let out_c = BASE_CH >> (i + 1);
            let (u, k) = (UPSAMPLE_RATES[i], UPSAMPLE_KERNELS[i]);
            ups.push(load_wn_conv_transpose1d(
                in_c,
                out_c,
                k,
                ConvTranspose1dConfig {
                    padding: (k - u) / 2,
                    output_padding: 0,
                    stride: u,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("ups").pp(i),
            )?);
        }

        // Source-injection path. `downsample_cum_rates[::-1]` for rates [8, 5, 3] is [15, 3, 1]:
        // stage i downsamples the 18-channel source STFT to the length of upsample stage i's output.
        let downsample_cum: [usize; 3] = [15, 3, 1];
        let mut source_downs = Vec::with_capacity(3);
        let mut source_resblocks = Vec::with_capacity(3);
        for i in 0..3 {
            let out_c = BASE_CH >> (i + 1);
            let u = downsample_cum[i];
            let (k, cfg) = if u == 1 {
                (
                    1,
                    Conv1dConfig {
                        padding: 0,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                )
            } else {
                (
                    u * 2,
                    Conv1dConfig {
                        padding: u / 2,
                        stride: u,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                )
            };
            source_downs.push(load_conv1d(
                SOURCE_CH,
                out_c,
                k,
                cfg,
                vb.pp("source_downs").pp(i),
            )?);
            source_resblocks.push(ResBlock::load(
                out_c,
                SOURCE_RESBLOCK_KERNELS[i],
                RESBLOCK_DILATIONS,
                vb.pp("source_resblocks").pp(i),
            )?);
        }

        // MRF resblocks: three per upsample stage (kernels [3, 7, 11]).
        let mut resblocks = Vec::with_capacity(9);
        for i in 0..3 {
            let ch = BASE_CH >> (i + 1);
            for &k in &RESBLOCK_KERNELS {
                let idx = resblocks.len();
                resblocks.push(ResBlock::load(
                    ch,
                    k,
                    RESBLOCK_DILATIONS,
                    vb.pp("resblocks").pp(idx),
                )?);
            }
        }

        let conv_post = load_wn_conv1d(
            BASE_CH >> 3, // 64
            SOURCE_CH,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            f0_predictor,
            l_linear_weight,
            l_linear_bias,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            window: dsp::hann_window(ISTFT_N_FFT),
            device,
        })
    }

    /// The configured device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of 24 kHz waveform samples for a `t_mel`-frame mel: `prod(upsample_rates) · hop`
    /// (= 480) per frame.
    pub fn num_samples(t_mel: usize) -> usize {
        UPSAMPLE_RATES[0] * UPSAMPLE_RATES[1] * UPSAMPLE_RATES[2] * ISTFT_HOP * t_mel
    }

    /// Vocode an 80-bin log-mel into a 24 kHz waveform. `mel` is `[80, T]` (the flow's output) or
    /// `[1, 80, T]`; `seed` seeds the NSF source's random phase + noise (reproducibility law).
    /// Returns the waveform as a 1-D `[T_samples]` tensor (`T_samples = 480 · T`).
    pub fn decode(&self, mel: &Tensor, seed: u64) -> Result<Tensor> {
        let mel = match mel.rank() {
            2 => mel.unsqueeze(0)?,
            3 => mel.clone(),
            r => {
                return Err(AudioError::Msg(format!(
                    "hift: mel must be [80, T] or [1, 80, T], got a rank-{r} tensor"
                )))
            }
        };
        let (batch, bins, t_mel) = mel.dims3()?;
        if batch != 1 || bins != MEL_DIM {
            return Err(AudioError::Msg(format!(
                "hift: mel must be [1, {MEL_DIM}, T], got [{batch}, {bins}, {t_mel}]"
            )));
        }
        let mel = mel.contiguous()?;

        // F0 → NSF source excitation (host DSP) → its STFT as an 18-channel [1, 18, TT] tensor.
        let f0 = self.f0_predictor.forward(&mel)?; // [1, T]
        let f0_host = f0.flatten_all()?.to_vec1::<f32>()?;
        let source = self.source_excitation(&f0_host, seed);
        let spec = dsp::stft(&source, ISTFT_N_FFT, ISTFT_HOP, &self.window)?;
        let tt = spec.n_frames;
        let mut s_stft_host = Vec::with_capacity(SOURCE_CH * tt);
        s_stft_host.extend_from_slice(&spec.re); // 9 real-bin rows
        s_stft_host.extend_from_slice(&spec.im); // 9 imag-bin rows
        let s_stft = Tensor::from_vec(s_stft_host, (1, SOURCE_CH, tt), &self.device)?;

        // Upsample trunk with source injection.
        let mut x = self.conv_pre.forward(&mel)?; // [1, 512, T]
        for i in 0..3 {
            x = leaky_relu(&x, LRELU_SLOPE)?;
            x = self.ups[i].forward(&x)?;
            if i == 2 {
                x = reflect_pad_left1(&x)?;
            }
            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;
            x = x.add(&si)?;
            // Multi-receptive-field fusion: mean of the three resblocks fed the same `x`.
            let mut xs = self.resblocks[i * 3].forward(&x)?;
            xs = xs.add(&self.resblocks[i * 3 + 1].forward(&x)?)?;
            xs = xs.add(&self.resblocks[i * 3 + 2].forward(&x)?)?;
            x = (xs / 3.0)?;
        }
        x = leaky_relu(&x, LRELU_SLOPE_DEFAULT)?;
        let x = self.conv_post.forward(&x)?.contiguous()?; // [1, 18, T']
        let t_prime = x.dim(2)?;

        // iSTFT head: split into exp-magnitude (bins 0..9) and sin-phase (bins 9..18).
        let x_host = x.flatten_all()?.to_vec1::<f32>()?; // channel-major [18 · T']
        let mut magnitude = vec![0f32; ISTFT_BINS * t_prime];
        let mut phase = vec![0f32; ISTFT_BINS * t_prime];
        for b in 0..ISTFT_BINS {
            for t in 0..t_prime {
                // magnitude = exp(x[:n_fft/2+1]), clipped to ≤ 1e2 (the reference `torch.clip`).
                magnitude[b * t_prime + t] = x_host[b * t_prime + t].exp().min(1e2);
                // phase = sin(x[n_fft/2+1:]).
                phase[b * t_prime + t] = x_host[(ISTFT_BINS + b) * t_prime + t].sin();
            }
        }
        let wav = dsp::istft(
            &magnitude,
            &phase,
            t_prime,
            ISTFT_N_FFT,
            ISTFT_HOP,
            &self.window,
        )?;
        let wav: Vec<f32> = wav
            .into_iter()
            .map(|v| v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT))
            .collect();
        let n = wav.len();
        Ok(Tensor::from_vec(wav, n, &self.device)?)
    }

    /// The NSF harmonic-plus-noise source excitation (host `f32` DSP). `f0_frames` is the per-mel-
    /// frame F0 (Hz); it is nearest-upsampled by `480` to the 24 kHz rate, drives a
    /// `NB_HARMONICS + 1` bank of phase-accumulated sines with random per-harmonic phase offsets,
    /// voiced/unvoiced gating and additive noise, then merged by `tanh(l_linear)` into one waveform
    /// of length `480 · t_mel`. Split out so the NSF math is unit-testable without weights.
    fn source_excitation(&self, f0_frames: &[f32], seed: u64) -> Vec<f32> {
        source_excitation_inner(f0_frames, &self.l_linear_weight, self.l_linear_bias, seed)
    }
}

/// Waveform samples produced per mel frame (`prod(upsample_rates) · istft_hop = 480`).
const SAMPLES_PER_FRAME: usize =
    UPSAMPLE_RATES[0] * UPSAMPLE_RATES[1] * UPSAMPLE_RATES[2] * ISTFT_HOP;

/// The raw `[NB_HARMONICS + 1, len]` NSF sine bank (`sine_amp · sin(2π·cumsum(f0·(h+1)/sr) + φ_h)`),
/// the per-sample voiced mask, and the nearest-upsampled F0 — the numeric core of [`SineGen`], split
/// out so the harmonic/phase math is testable without the `l_linear` merge weights.
///
/// Returns `(sines, uv, f0_up)` where `sines[h * len + i]` is harmonic `h` at sample `i`, `uv[i]`
/// the voiced gate, and `f0_up[i]` the upsampled F0 (Hz). Noise is *not* added here (it is applied
/// in the merge) so the returned sines are deterministic given the phase seed.
fn nsf_sine_bank(f0_frames: &[f32], seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = NB_HARMONICS + 1;
    let sr = S3GEN_SR as f64;
    let len = f0_frames.len() * SAMPLES_PER_FRAME;

    // Random per-harmonic phase offset in [−π, π); the fundamental (h = 0) has no offset.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut phase_off = vec![0f32; dim];
    for slot in phase_off.iter_mut().skip(1) {
        let u: f32 = rng.random();
        *slot = (u * 2.0 - 1.0) * std::f32::consts::PI;
    }

    let mut sines = vec![0f32; dim * len];
    let mut uv = vec![0f32; len];
    let mut f0_up = vec![0f32; len];
    let mut cum = vec![0f64; dim]; // per-harmonic cumulative phase (cycles)
    for i in 0..len {
        let f0v = f0_frames[i / SAMPLES_PER_FRAME];
        f0_up[i] = f0v;
        uv[i] = if f0v > NSF_VOICED_THRESHOLD { 1.0 } else { 0.0 };
        for (h, cum_h) in cum.iter_mut().enumerate() {
            *cum_h += f0v as f64 * (h as f64 + 1.0) / sr;
            let theta = 2.0 * std::f64::consts::PI * cum_h.rem_euclid(1.0);
            sines[h * len + i] = NSF_SINE_AMP * (theta as f32 + phase_off[h]).sin();
        }
    }
    (sines, uv, f0_up)
}

/// [`HiftGenerator::source_excitation`] over explicit `l_linear` weights (free function so the full
/// NSF path is unit-testable without a loaded generator).
fn source_excitation_inner(
    f0_frames: &[f32],
    l_linear_weight: &[f32],
    l_linear_bias: f32,
    seed: u64,
) -> Vec<f32> {
    let dim = NB_HARMONICS + 1;
    let (sines, uv, _) = nsf_sine_bank(f0_frames, seed);
    let len = uv.len();
    // Additive noise is drawn after the sine phases (a fixed, deterministic RNG order).
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let mut out = vec![0f32; len];
    for i in 0..len {
        let noise_amp = uv[i] * NSF_NOISE_STD + (1.0 - uv[i]) * NSF_SINE_AMP / 3.0;
        let mut acc = l_linear_bias;
        for h in 0..dim {
            let noise: f32 = rng.sample::<f32, _>(StandardNormal) * noise_amp;
            let val = sines[h * len + i] * uv[i] + noise;
            acc += l_linear_weight[h] * val;
        }
        out[i] = acc.tanh();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    // -------------------------------------------------------------------------------------------
    // NSF source-module math (harmonic sine generation, voiced/unvoiced gating).
    // -------------------------------------------------------------------------------------------

    /// A constant voiced F0 produces a fundamental at the right frequency and `NB_HARMONICS + 1`
    /// harmonics at integer multiples of it (checked by positive-going zero-crossing counts).
    #[test]
    fn nsf_sine_bank_has_correct_fundamental_and_harmonics() {
        let f0 = 200.0f32; // Hz, well above the 10 Hz voiced threshold
        let frames = 100usize;
        let f0_frames = vec![f0; frames];
        let (sines, uv, _) = nsf_sine_bank(&f0_frames, 7);

        let len = frames * SAMPLES_PER_FRAME;
        assert_eq!(sines.len(), (NB_HARMONICS + 1) * len);
        assert_eq!(uv.len(), len);
        assert!(
            uv.iter().all(|&v| v == 1.0),
            "constant 200 Hz F0 is all-voiced"
        );

        let dur_s = len as f32 / S3GEN_SR as f32;
        for h in 0..=NB_HARMONICS {
            // Count upward zero crossings → one per period → cycles ≈ f0·(h+1)·dur.
            let row = &sines[h * len..(h + 1) * len];
            let mut crossings = 0usize;
            for w in row.windows(2) {
                if w[0] <= 0.0 && w[1] > 0.0 {
                    crossings += 1;
                }
            }
            let expected = f0 * (h as f32 + 1.0) * dur_s;
            let rel = (crossings as f32 - expected).abs() / expected;
            assert!(
                rel < 0.02,
                "harmonic {h}: {crossings} cycles vs expected {expected:.1} ({:.1}% off)",
                rel * 100.0
            );
        }
    }

    /// Unvoiced frames (`f0 ≤ 10 Hz`) gate the sine to (near) zero: the merged excitation is far
    /// smaller than for a voiced frame at the same amplitude scale.
    #[test]
    fn nsf_voiced_unvoiced_gating() {
        // uv from the sine bank: 0 Hz → unvoiced, 200 Hz → voiced.
        let (_, uv_silent, _) = nsf_sine_bank(&[0.0f32; 4], 1);
        assert!(uv_silent.iter().all(|&v| v == 0.0), "0 Hz is unvoiced");
        let (_, uv_voiced, _) = nsf_sine_bank(&[150.0f32; 4], 1);
        assert!(uv_voiced.iter().all(|&v| v == 1.0), "150 Hz is voiced");

        // Full merged excitation: gating the sine on for voiced frames makes the excitation
        // materially larger than the unvoiced case (which faithfully keeps a residual noise floor of
        // amplitude sine_amp/3 — it is not silence). A broken gate would make the two comparable.
        let w = vec![1.0f32; NB_HARMONICS + 1];
        let voiced = source_excitation_inner(&[150.0f32; 50], &w, 0.0, 3);
        let unvoiced = source_excitation_inner(&[0.0f32; 50], &w, 0.0, 3);
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        assert!(
            rms(&voiced) > 1.5 * rms(&unvoiced),
            "voiced excitation ({:.4}) should materially exceed unvoiced ({:.4})",
            rms(&voiced),
            rms(&unvoiced)
        );
        assert!(voiced.iter().all(|v| v.is_finite()));
    }

    /// The source excitation is deterministic for a fixed seed and its length is exactly
    /// `480 · t_mel` (the samples the whole vocoder must ultimately emit).
    #[test]
    fn source_excitation_is_deterministic_and_correctly_sized() {
        let w = vec![0.3f32; NB_HARMONICS + 1];
        let a = source_excitation_inner(&[120.0f32; 10], &w, 0.1, 42);
        let b = source_excitation_inner(&[120.0f32; 10], &w, 0.1, 42);
        assert_eq!(a, b, "same seed ⇒ byte-identical source");
        assert_eq!(a.len(), 10 * SAMPLES_PER_FRAME);
        assert_eq!(a.len(), 10 * 480);
    }

    // -------------------------------------------------------------------------------------------
    // Upsample / iSTFT length accounting.
    // -------------------------------------------------------------------------------------------

    /// Per-stage output lengths for rates [8, 5, 3] + the ×4 iSTFT, and the 480 samples/frame
    /// product — reproducing the reference conv/kernel/padding arithmetic exactly.
    #[test]
    fn upsample_and_istft_length_accounting() {
        // The 480 = 8·5·3·4 product.
        assert_eq!(SAMPLES_PER_FRAME, 480);
        assert_eq!(UPSAMPLE_RATES.iter().product::<usize>() * ISTFT_HOP, 480);

        for &t in &[1usize, 2, 8, 37, 250] {
            // conv_pre preserves length.
            let mut x = t;
            // ConvTranspose1d length: (in − 1)·stride − 2·pad + kernel  (dilation 1, output_pad 0).
            let expected = [8 * t, 40 * t, 120 * t];
            for i in 0..3 {
                let (u, k) = (UPSAMPLE_RATES[i], UPSAMPLE_KERNELS[i]);
                let pad = (k - u) / 2;
                x = (x - 1) * u + k - 2 * pad;
                assert_eq!(x, expected[i], "ups stage {i} length for t_mel={t}");
            }
            // The final reflection pad adds one sample.
            let x_padded = x + 1;
            assert_eq!(x_padded, 120 * t + 1);

            // Source STFT frame count (center=True): 1 + (480·t)/hop = 120·t + 1.
            let tt = 1 + (SAMPLES_PER_FRAME * t) / ISTFT_HOP;
            assert_eq!(tt, 120 * t + 1);

            // source_downs stride accounting matches each upsample stage's output length.
            let downsample_cum = [15usize, 3, 1];
            let src_lengths: Vec<usize> = (0..3)
                .map(|i| {
                    let u = downsample_cum[i];
                    if u == 1 {
                        // Conv1d(k=1, stride=1, pad=0) preserves length.
                        tt
                    } else {
                        let (k, pad) = (u * 2, u / 2);
                        (tt + 2 * pad - k) / u + 1
                    }
                })
                .collect();
            assert_eq!(src_lengths, vec![8 * t, 40 * t, 120 * t + 1]);

            // iSTFT (center=True, length (t_prime − 1)·hop) over t_prime = 120·t + 1 frames → 480·t.
            let t_prime = x_padded;
            let wave_len = (t_prime - 1) * ISTFT_HOP;
            assert_eq!(wave_len, 480 * t);
            assert_eq!(wave_len, HiftGenerator::num_samples(t));
        }
    }

    // -------------------------------------------------------------------------------------------
    // weight-norm reconstruction.
    // -------------------------------------------------------------------------------------------

    /// `weight_norm_weight` reproduces `g · v / ‖v‖` (per output channel) exactly.
    #[test]
    fn weight_norm_reconstruction_matches_reference() {
        let dev = Device::Cpu;
        let (out_c, in_c, k) = (3usize, 2usize, 4usize);
        let v: Vec<f32> = (0..out_c * in_c * k)
            .map(|i| (i as f32 * 0.1) - 1.0)
            .collect();
        let g: Vec<f32> = vec![2.0, -0.5, 1.5];
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // Pre-seed the two parametrization tensors so `weight_norm_weight`'s `.get` retrieves them.
        varmap.data().lock().unwrap().insert(
            "parametrizations.weight.original1".to_string(),
            candle_audio::candle_core::Var::from_tensor(
                &Tensor::from_vec(v.clone(), (out_c, in_c, k), &dev).unwrap(),
            )
            .unwrap(),
        );
        varmap.data().lock().unwrap().insert(
            "parametrizations.weight.original0".to_string(),
            candle_audio::candle_core::Var::from_tensor(
                &Tensor::from_vec(g.clone(), (out_c, 1, 1), &dev).unwrap(),
            )
            .unwrap(),
        );
        let w = weight_norm_weight(&vb, out_c, in_c, k).unwrap();
        let got = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for o in 0..out_c {
            let slice = &v[o * in_c * k..(o + 1) * in_c * k];
            let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            for j in 0..in_c * k {
                let want = g[o] * slice[j] / norm;
                assert!((got[o * in_c * k + j] - want).abs() < 1e-6);
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // Snake activation.
    // -------------------------------------------------------------------------------------------

    /// `Snake` computes `x + (1/α)·sin(αx)²` (α = 1 here): 0 at x = 0, matches the closed form.
    #[test]
    fn snake_activation_matches_closed_form() {
        let dev = Device::Cpu;
        let ch = 2usize;
        let varmap = VarMap::new();
        varmap.data().lock().unwrap().insert(
            "alpha".to_string(),
            candle_audio::candle_core::Var::from_tensor(
                &Tensor::from_vec(vec![1.0f32; ch], ch, &dev).unwrap(),
            )
            .unwrap(),
        );
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let snake = Snake::load(ch, vb).unwrap();
        let xs = vec![0.0f32, 0.5, 1.0, -1.0, 2.0, -0.3];
        let x = Tensor::from_vec(xs.clone(), (1, ch, 3), &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &xv) in xs.iter().enumerate() {
            let want = xv + (1.0 / (1.0 + 1e-9)) * (xv.sin()).powi(2);
            assert!(
                (got[i] - want).abs() < 1e-5,
                "snake at x={xv}: {} vs {want}",
                got[i]
            );
        }
    }

    // -------------------------------------------------------------------------------------------
    // Full forward on synthetic (random) weights: shape + length + finiteness.
    // -------------------------------------------------------------------------------------------

    /// End-to-end structural test on randomly-initialized weights: the vocoder maps an
    /// `[80, T]` mel to a finite `[480·T]` waveform, exercising the weight-norm reconstruction,
    /// the transposed-conv upsample trunk, source injection, MRF resblocks, and the iSTFT head.
    #[test]
    fn decode_maps_mel_to_480x_waveform_on_synthetic_weights() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        materialize_mel2wav_shapes(&vb);
        let hift = HiftGenerator::load(vb, dev.clone()).unwrap();

        let t_mel = 8usize;
        let mel = Tensor::randn(0f32, 1.0, (MEL_DIM, t_mel), &dev).unwrap();
        let wav = hift.decode(&mel, 20260719).unwrap();
        assert_eq!(wav.rank(), 1);
        assert_eq!(
            wav.dim(0).unwrap(),
            480 * t_mel,
            "waveform is 480·T samples"
        );
        let v = wav.to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "waveform must be finite");
        assert!(
            v.iter().all(|x| x.abs() <= AUDIO_LIMIT + 1e-6),
            "clamped to audio_limit"
        );

        // Deterministic under a fixed seed.
        let wav2 = hift.decode(&mel, 20260719).unwrap();
        assert_eq!(v, wav2.to_vec1::<f32>().unwrap());
    }

    // ---- synthetic-weight materializers (shapes exactly match mel2wav.*) ----

    /// Materialize a weight-normed conv's `parametrizations.weight.original0/1` (`[d0, 1, 1]` /
    /// `[d0, d1, k]`) + `bias` (`[bias_len]`) with random (non-zero, so `‖v‖ > 0`) values under `vb`.
    /// `d0` is the weight-norm dim (`out` for `Conv1d`, `in` for `ConvTranspose1d`), while the bias is
    /// always `out_channels` — so `bias_len` is passed explicitly.
    fn wn_conv(vb: &VarBuilder, d0: usize, d1: usize, k: usize, bias_len: usize) {
        let init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.5,
        };
        let _ = vb
            .get_with_hints((d0, 1, 1), "parametrizations.weight.original0", init)
            .unwrap();
        let _ = vb
            .get_with_hints((d0, d1, k), "parametrizations.weight.original1", init)
            .unwrap();
        let _ = vb.get(bias_len, "bias").unwrap();
    }

    fn plain_conv(vb: &VarBuilder, out_c: usize, in_c: usize, k: usize) {
        let init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.5,
        };
        let _ = vb.get_with_hints((out_c, in_c, k), "weight", init).unwrap();
        let _ = vb.get(out_c, "bias").unwrap();
    }

    fn resblock(vb: &VarBuilder, ch: usize, kernel: usize) {
        for group in ["convs1", "convs2"] {
            for idx in 0..3 {
                wn_conv(&vb.pp(group).pp(idx), ch, ch, kernel, ch);
            }
        }
        for group in ["activations1", "activations2"] {
            for idx in 0..3 {
                let _ = vb.pp(group).pp(idx).get(ch, "alpha").unwrap();
            }
        }
    }

    fn materialize_mel2wav_shapes(vb: &VarBuilder) {
        // f0_predictor: 5 weight-normed convs (even indices) + classifier.
        wn_conv(
            &vb.pp("f0_predictor").pp("condnet").pp(0),
            BASE_CH,
            MEL_DIM,
            3,
            BASE_CH,
        );
        for i in 1..5 {
            wn_conv(
                &vb.pp("f0_predictor").pp("condnet").pp(2 * i),
                BASE_CH,
                BASE_CH,
                3,
                BASE_CH,
            );
        }
        let _ = vb
            .pp("f0_predictor")
            .pp("classifier")
            .get((1, BASE_CH), "weight")
            .unwrap();
        let _ = vb
            .pp("f0_predictor")
            .pp("classifier")
            .get(1, "bias")
            .unwrap();

        // m_source.l_linear (Linear(9 → 1)).
        let _ = vb
            .pp("m_source")
            .pp("l_linear")
            .get((1, NB_HARMONICS + 1), "weight")
            .unwrap();
        let _ = vb.pp("m_source").pp("l_linear").get(1, "bias").unwrap();

        // conv_pre / conv_post.
        wn_conv(&vb.pp("conv_pre"), BASE_CH, MEL_DIM, 7, BASE_CH);
        wn_conv(&vb.pp("conv_post"), SOURCE_CH, BASE_CH >> 3, 7, SOURCE_CH);

        // ups (ConvTranspose1d weights are [in, out, k]; the bias is `out`).
        #[allow(clippy::needless_range_loop)]
        // `i` drives the channel shifts too, not just the index
        for i in 0..3 {
            let (in_c, out_c) = (BASE_CH >> i, BASE_CH >> (i + 1));
            wn_conv(&vb.pp("ups").pp(i), in_c, out_c, UPSAMPLE_KERNELS[i], out_c);
        }

        // source_downs (plain) + source_resblocks.
        let downsample_cum = [15usize, 3, 1];
        for i in 0..3 {
            let out_c = BASE_CH >> (i + 1);
            let u = downsample_cum[i];
            let k = if u == 1 { 1 } else { u * 2 };
            plain_conv(&vb.pp("source_downs").pp(i), out_c, SOURCE_CH, k);
            resblock(
                &vb.pp("source_resblocks").pp(i),
                out_c,
                SOURCE_RESBLOCK_KERNELS[i],
            );
        }

        // MRF resblocks (3 per stage).
        let mut idx = 0;
        for i in 0..3 {
            let ch = BASE_CH >> (i + 1);
            for &k in &RESBLOCK_KERNELS {
                resblock(&vb.pp("resblocks").pp(idx), ch, k);
                idx += 1;
            }
        }
    }
}
