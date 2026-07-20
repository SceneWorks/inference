//! MMAudio's **16 kHz BigVGAN vocoder** (sc-13440) — the Make-An-Audio-2 BigVGAN — ported natively
//! onto the pinned candle revision. This is the second stage of the 16k output path: it turns the
//! 80-band log-mel spectrogram the [`crate::vae`] decoder emits into a 16 kHz waveform.
//!
//! ## Faithful to `mmaudio/ext/bigvgan/{models,activations}.py` + `alias_free_torch/*`
//!
//! BigVGAN is an anti-aliased HiFi-GAN-style vocoder configured by `bigvgan_vocoder.yml`:
//! `num_mels=80`, `upsample_initial_channel=1536`, `upsample_rates=[4,4,2,2,2,2]` (product =
//! `256` = the mel hop), `upsample_kernel_sizes=[8,8,4,4,4,4]`, three `AMPBlock1` resblocks per
//! stage with `resblock_kernel_sizes=[3,7,11]` / `dilations=[1,3,5]`, and `snakebeta` anti-aliased
//! activations (`snake_logscale=True`).
//!
//! - **`conv_pre`** `Conv1d(80→1536, k7, pad3)`.
//! - **6 upsample stages**: a `ConvTranspose1d` (`stride=rate`, `pad=(k-rate)/2`) then the sum of 3
//!   `AMPBlock1`s divided by 3.
//! - **`AMPBlock1`**: for each of 3 (dilated `conv1`, `conv2`) pairs — `act1 → conv1 → act2 →
//!   conv2`, residual-added. Each conv is a `Conv1d(ch→ch, k, dilation, pad=get_padding(k,d))`.
//! - **`Activation1d`**: anti-aliased periodic nonlinearity — upsample ×2 (kaiser-sinc FIR) →
//!   `snakebeta` → downsample ×2 (kaiser-sinc FIR). The 12-tap kaiser-sinc filters are **loaded
//!   from the checkpoint** (`upsample.filter` / `downsample.lowpass.filter`, both `(1,1,12)`), not
//!   recomputed, so the anti-aliasing matches the reference exactly.
//! - **`snakebeta`** `x + (1/(exp(beta)+1e-9))·sin(x·exp(alpha))^2` (log-scale alpha/beta).
//! - **`activation_post` → `conv_post` `Conv1d(ch→1, k7, pad3)` → `tanh`**.
//!
//! ## Weight-norm reconstruction
//!
//! The generator checkpoint stores every conv under PyTorch `weight_norm` parametrization
//! (`weight_g` `(d0,1,1)` + `weight_v`), never a baked `weight`. `weight_norm_reconstruct`
//! rebuilds `w = g · v / ‖v‖` (L2 over all dims but 0, matching torch's default `dim=0`) at load —
//! for both `Conv1d` (`dim0 = out`) and `ConvTranspose1d` (`dim0 = in`).
//!
//! ## Anti-aliased depthwise convs
//!
//! Every channel shares one 12-tap filter, so the grouped (`groups=C`) up/down convs are computed
//! as a single non-grouped conv over a merged `(B·C, 1, L)` batch — identical arithmetic to torch's
//! per-channel grouped conv, but without candle's O(C) grouped-conv decomposition.

use candle_audio::candle_core::{DType, Result as CResult, Tensor};
use candle_nn::VarBuilder;

/// Mel bands the vocoder consumes (`num_mels`).
pub const NUM_MELS: usize = 80;
/// First conv width (`upsample_initial_channel`).
pub const UPSAMPLE_INITIAL_CHANNEL: usize = 1536;
/// Transposed-conv upsample strides (`upsample_rates`); product = 256 = the mel hop.
pub const UPSAMPLE_RATES: [usize; 6] = [4, 4, 2, 2, 2, 2];
/// Transposed-conv kernel sizes (`upsample_kernel_sizes`).
pub const UPSAMPLE_KERNEL_SIZES: [usize; 6] = [8, 8, 4, 4, 4, 4];
/// Resblock conv kernel sizes (`resblock_kernel_sizes`).
pub const RESBLOCK_KERNEL_SIZES: [usize; 3] = [3, 7, 11];
/// Resblock dilations (`resblock_dilation_sizes`), one triple per kernel.
pub const RESBLOCK_DILATIONS: [[usize; 3]; 3] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
/// Total upsampling factor (∏ `UPSAMPLE_RATES`) — samples produced per mel frame (16k path).
pub const HOP: usize = 256;

const SNAKE_NO_DIV_BY_ZERO: f64 = 1e-9;

/// BigVGAN generator hyperparameters (`bigvgan_vocoder.yml` for the 16k Make-An-Audio-2 vocoder;
/// the NVIDIA `config.json` for BigVGAN v2). Both variants share the topology (6 transposed-conv
/// upsample stages, 3 `AMPBlock1` resblocks per stage, snakebeta anti-aliased activations) and the
/// same checkpoint key layout — including the loaded 12-tap kaiser-sinc anti-aliasing filters — and
/// differ only in the numeric config below.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Input mel bands (`num_mels`): 80 (16k) / 128 (BigVGAN v2 44k).
    pub num_mels: usize,
    /// First conv width (`upsample_initial_channel`): 1536 for both variants.
    pub upsample_initial_channel: usize,
    /// Transposed-conv upsample strides (`upsample_rates`); ∏ = the mel hop (256 / 512).
    pub upsample_rates: [usize; 6],
    /// Transposed-conv kernel sizes (`upsample_kernel_sizes`).
    pub upsample_kernel_sizes: [usize; 6],
    /// Resblock conv kernel sizes (`resblock_kernel_sizes`).
    pub resblock_kernel_sizes: [usize; 3],
    /// Resblock dilations (`resblock_dilation_sizes`), one triple per kernel.
    pub resblock_dilations: [[usize; 3]; 3],
    /// Whether `conv_post` carries a bias (`use_bias_at_final`): true (16k) / false (BigVGAN v2).
    pub use_bias_at_final: bool,
    /// Final activation: `tanh` when true (`use_tanh_at_final`, 16k), else `clamp(-1, 1)` (v2).
    pub use_tanh_at_final: bool,
}

impl Config {
    /// The Make-An-Audio-2 **16 kHz** BigVGAN (`bigvgan_vocoder.yml`): hop 256, 80-band, `tanh` +
    /// bias at final.
    pub fn bigvgan_16k() -> Self {
        Self {
            num_mels: NUM_MELS,
            upsample_initial_channel: UPSAMPLE_INITIAL_CHANNEL,
            upsample_rates: UPSAMPLE_RATES,
            upsample_kernel_sizes: UPSAMPLE_KERNEL_SIZES,
            resblock_kernel_sizes: RESBLOCK_KERNEL_SIZES,
            resblock_dilations: RESBLOCK_DILATIONS,
            use_bias_at_final: true,
            use_tanh_at_final: true,
        }
    }

    /// NVIDIA **BigVGAN v2** `bigvgan_v2_44khz_128band_512x` (sc-13441): hop 512, 128-band,
    /// `upsample_rates=[8,4,2,2,2,2]` / `upsample_kernel_sizes=[16,8,4,4,4,4]`, and — per the NVIDIA
    /// `config.json` — **no** bias at final and **clamp** (not tanh) at final.
    pub fn bigvgan_v2_44khz_128band_512x() -> Self {
        Self {
            num_mels: 128,
            upsample_initial_channel: 1536,
            upsample_rates: [8, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: [16, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 7, 11],
            resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            use_bias_at_final: false,
            use_tanh_at_final: false,
        }
    }

    /// Total upsampling factor (∏ `upsample_rates`) — samples produced per mel frame (256 / 512).
    pub fn hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// HiFi-GAN `get_padding(kernel_size, dilation)` = `(k·d - d) / 2`.
fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

/// Rebuild a `weight_norm`-parametrized conv weight: `w = g · v / ‖v‖`, with `‖v‖` the L2 norm over
/// every dim except 0 (torch default `dim=0`). `g` is `(d0, 1, 1)`, `v` is `(d0, d1, k)`.
fn weight_norm_reconstruct(g: &Tensor, v: &Tensor) -> CResult<Tensor> {
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?; // (d0,1,1)
    v.broadcast_mul(g)?.broadcast_div(&norm)
}

/// A `weight_norm`-parametrized `Conv1d`, reconstructed at load. Bias is optional — BigVGAN v2's
/// `conv_post` has `use_bias_at_final=false`, so its checkpoint carries no `bias` key.
struct Conv1d {
    weight: Tensor,       // (out, in, k)
    bias: Option<Tensor>, // (out,)
    padding: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    fn load(
        vb: VarBuilder,
        out: usize,
        inc: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
    ) -> CResult<Self> {
        Self::load_opt_bias(vb, out, inc, k, stride, dilation, padding, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_opt_bias(
        vb: VarBuilder,
        out: usize,
        inc: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        bias: bool,
    ) -> CResult<Self> {
        let g = vb.get((out, 1, 1), "weight_g")?;
        let v = vb.get((out, inc, k), "weight_v")?;
        let weight = weight_norm_reconstruct(&g, &v)?;
        let bias = if bias {
            Some(vb.get(out, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            padding,
            stride,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
            None => Ok(y),
        }
    }
}

/// A `weight_norm`-parametrized `ConvTranspose1d` (bias included), reconstructed at load. The
/// checkpoint weight is `(in, out, k)` and `weight_g` is `(in, 1, 1)` — torch's default `dim=0`
/// weight-norm on a transposed conv normalizes per **input** channel.
struct ConvTranspose1d {
    weight: Tensor, // (in, out, k)
    bias: Tensor,   // (out,)
    padding: usize,
    stride: usize,
}

impl ConvTranspose1d {
    fn load(vb: VarBuilder, inc: usize, out: usize, k: usize, stride: usize) -> CResult<Self> {
        let g = vb.get((inc, 1, 1), "weight_g")?;
        let v = vb.get((inc, out, k), "weight_v")?;
        let weight = weight_norm_reconstruct(&g, &v)?;
        let bias = vb.get(out, "bias")?;
        Ok(Self {
            weight,
            bias,
            padding: (k - stride) / 2,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let y = x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?;
        y.broadcast_add(&self.bias.reshape((1, self.bias.dim(0)?, 1))?)
    }
}

/// Depthwise conv where every channel shares one filter, computed as a single non-grouped conv over
/// a merged `(B·C, 1, L)` batch. `stride` and `transpose` select the down (`conv1d`) vs up
/// (`conv_transpose1d`) direction.
fn shared_filter_conv(
    x: &Tensor,
    filter: &Tensor, // (1, 1, k)
    stride: usize,
    transpose: bool,
) -> CResult<Tensor> {
    let (b, c, l) = x.dims3()?;
    let merged = x.reshape((b * c, 1, l))?;
    let y = if transpose {
        merged.conv_transpose1d(filter, 0, 0, stride, 1, 1)?
    } else {
        merged.conv1d(filter, 0, stride, 1, 1)?
    };
    let lo = y.dim(2)?;
    y.reshape((b, c, lo))
}

/// `snakebeta` periodic activation (log-scale): `x + (1/(exp(beta)+eps))·sin(x·exp(alpha))^2`.
/// `alpha`/`beta` are `(1, C, 1)`.
fn snakebeta(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> CResult<Tensor> {
    let a = alpha.exp()?;
    let b = beta.exp()?;
    let inv_b = b.affine(1.0, SNAKE_NO_DIV_BY_ZERO)?.recip()?;
    let s = x.broadcast_mul(&a)?.sin()?.sqr()?; // sin(x·a)^2
    x.broadcast_add(&s.broadcast_mul(&inv_b)?)
}

/// Anti-aliased `Activation1d`: upsample ×2 (FIR) → `snakebeta` → downsample ×2 (FIR).
///
/// Filters are loaded from the checkpoint (`upsample.filter`, `downsample.lowpass.filter`). The
/// pad/crop constants are the reference's for `ratio=2, kernel_size=12`.
struct Activation1d {
    alpha: Tensor, // (1, C, 1)
    beta: Tensor,  // (1, C, 1)
    up_filter: Tensor,
    down_filter: Tensor,
}

impl Activation1d {
    // ratio = 2, kernel_size = 12.
    const UP_PAD: usize = 5; // kernel_size/ratio - 1
    const UP_CROP: usize = 15; // pad*stride + (ks-stride)//2 = pad*stride + (ks-stride+1)//2
    const DOWN_PAD_LEFT: usize = 5; // ks/2 - 1 (even kernel)
    const DOWN_PAD_RIGHT: usize = 6; // ks/2

    fn load(vb: VarBuilder, channels: usize) -> CResult<Self> {
        let alpha = vb.get(channels, "act.alpha")?.reshape((1, channels, 1))?;
        let beta = vb.get(channels, "act.beta")?.reshape((1, channels, 1))?;
        let up_filter = vb.get((1, 1, 12), "upsample.filter")?;
        let down_filter = vb.get((1, 1, 12), "downsample.lowpass.filter")?;
        Ok(Self {
            alpha,
            beta,
            up_filter,
            down_filter,
        })
    }

    fn upsample(&self, x: &Tensor) -> CResult<Tensor> {
        let l = x.dim(2)?;
        let xp = x.pad_with_same(2, Self::UP_PAD, Self::UP_PAD)?;
        // ratio * conv_transpose1d, then center-crop [UP_CROP : UP_CROP + 2L].
        let y = (shared_filter_conv(&xp, &self.up_filter, 2, true)? * 2.0)?;
        y.narrow(2, Self::UP_CROP, 2 * l)
    }

    fn downsample(&self, x: &Tensor) -> CResult<Tensor> {
        let xp = x.pad_with_same(2, Self::DOWN_PAD_LEFT, Self::DOWN_PAD_RIGHT)?;
        shared_filter_conv(&xp, &self.down_filter, 2, false)
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.upsample(x)?;
        let x = snakebeta(&x, &self.alpha, &self.beta)?;
        self.downsample(&x)
    }
}

/// `AMPBlock1`: three (dilated `conv1`, `conv2`) pairs with anti-aliased activations, each
/// residual-added.
struct AmpBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations: Vec<Activation1d>, // length 6: [a1_0, a2_0, a1_1, a2_1, a1_2, a2_2]
}

impl AmpBlock1 {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilation: [usize; 3],
    ) -> CResult<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        for (i, d) in dilation.iter().enumerate() {
            convs1.push(Conv1d::load(
                vb.pp("convs1").pp(i),
                channels,
                channels,
                kernel_size,
                1,
                *d,
                get_padding(kernel_size, *d),
            )?);
            convs2.push(Conv1d::load(
                vb.pp("convs2").pp(i),
                channels,
                channels,
                kernel_size,
                1,
                1,
                get_padding(kernel_size, 1),
            )?);
        }
        let mut activations = Vec::with_capacity(6);
        for i in 0..6 {
            activations.push(Activation1d::load(vb.pp("activations").pp(i), channels)?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // acts1 = activations[0::2], acts2 = activations[1::2].
        let mut x = x.clone();
        for i in 0..3 {
            let a1 = &self.activations[2 * i];
            let a2 = &self.activations[2 * i + 1];
            let xt = a1.forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = a2.forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

/// The assembled BigVGAN vocoder (`BigVGANVocoder`): mel `(B, num_mels, T)` → waveform
/// `(B, 1, hop·T)`. Both the 16k Make-An-Audio-2 vocoder and the NVIDIA BigVGAN v2 44k vocoder are
/// this one struct, [`Config`]-parameterized.
pub struct BigVganVocoder {
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AmpBlock1>, // len = num_upsamples * num_kernels (6 * 3)
    activation_post: Activation1d,
    conv_post: Conv1d,
    num_kernels: usize,
    use_tanh_at_final: bool,
}

impl BigVganVocoder {
    /// Load the 16k Make-An-Audio-2 vocoder from a `VarBuilder` rooted at the generator sub-tree
    /// (keys `conv_pre.*`, `ups.*`, `resblocks.*`, `activation_post.*`, `conv_post.*`).
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        Self::load_with_config(vb, Config::bigvgan_16k())
    }

    /// Load with an explicit [`Config`] (16k or NVIDIA BigVGAN v2 44k, sc-13441).
    pub fn load_with_config(vb: VarBuilder, cfg: Config) -> CResult<Self> {
        let num_upsamples = cfg.upsample_rates.len();
        let num_kernels = cfg.resblock_kernel_sizes.len();

        let conv_pre = Conv1d::load(
            vb.pp("conv_pre"),
            cfg.upsample_initial_channel,
            cfg.num_mels,
            7,
            1,
            1,
            3,
        )?;

        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let inc = cfg.upsample_initial_channel >> i;
            let out = cfg.upsample_initial_channel >> (i + 1);
            ups.push(ConvTranspose1d::load(
                vb.pp("ups").pp(i).pp(0),
                inc,
                out,
                k,
                u,
            )?);
        }

        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        for i in 0..num_upsamples {
            let ch = cfg.upsample_initial_channel >> (i + 1);
            for (j, (&k, &d)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilations.iter())
                .enumerate()
            {
                resblocks.push(AmpBlock1::load(
                    vb.pp("resblocks").pp(i * num_kernels + j),
                    ch,
                    k,
                    d,
                )?);
            }
        }

        let ch_last = cfg.upsample_initial_channel >> num_upsamples;
        let activation_post = Activation1d::load(vb.pp("activation_post"), ch_last)?;
        let conv_post = Conv1d::load_opt_bias(
            vb.pp("conv_post"),
            1,
            ch_last,
            7,
            1,
            1,
            3,
            cfg.use_bias_at_final,
        )?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_kernels,
            use_tanh_at_final: cfg.use_tanh_at_final,
        })
    }

    /// Vocode a mel spectrogram `(B, num_mels, T)` → waveform `(B, 1, hop·T)` in `[-1, 1]`.
    pub fn forward(&self, mel: &Tensor) -> CResult<Tensor> {
        let mut x = self.conv_pre.forward(mel)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x)?;
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let r = self.resblocks[i * self.num_kernels + j].forward(&x)?;
                xs = Some(match xs {
                    None => r,
                    Some(acc) => (acc + r)?,
                });
            }
            x = (xs.expect("num_kernels > 0") / self.num_kernels as f64)?;
        }
        x = self.activation_post.forward(&x)?;
        x = self.conv_post.forward(&x)?;
        // Final: tanh (16k) or clamp to [-1, 1] (BigVGAN v2's use_tanh_at_final=false).
        if self.use_tanh_at_final {
            x.tanh()
        } else {
            x.clamp(-1.0, 1.0)
        }
    }
}

/// Cast a mel tensor to the vocoder's compute dtype (f32).
pub fn to_compute_dtype(mel: &Tensor) -> CResult<Tensor> {
    mel.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn get_padding_matches_reference() {
        // get_padding(k, d) = (k·d - d) / 2 — chosen so a dilated conv preserves length.
        assert_eq!(get_padding(3, 1), 1);
        assert_eq!(get_padding(3, 3), 3);
        assert_eq!(get_padding(3, 5), 5);
        assert_eq!(get_padding(7, 1), 3);
        assert_eq!(get_padding(11, 1), 5);
    }

    #[test]
    fn bigvgan_configs_match_reference() {
        let c16 = Config::bigvgan_16k();
        assert_eq!(c16.hop(), 256, "16k hop = ∏[4,4,2,2,2,2]");
        assert_eq!(c16.num_mels, 80);
        assert!(c16.use_bias_at_final && c16.use_tanh_at_final);
        // NVIDIA BigVGAN v2 44k: hop 512, 128-band, no-bias + clamp at final.
        let v2 = Config::bigvgan_v2_44khz_128band_512x();
        assert_eq!(v2.hop(), 512, "44k hop = ∏[8,4,2,2,2,2]");
        assert_eq!(v2.num_mels, 128);
        assert_eq!(v2.upsample_kernel_sizes, [16, 8, 4, 4, 4, 4]);
        assert!(!v2.use_bias_at_final, "config.json use_bias_at_final=false");
        assert!(!v2.use_tanh_at_final, "config.json use_tanh_at_final=false");
    }

    #[test]
    fn weight_norm_reconstruct_matches_torch() {
        let dev = Device::Cpu;
        // v (out=2, in=2, k=3), g (2,1,1). w = g * v / ||v||_{dims 1,2}.
        let v = Tensor::randn(0f32, 1.0, (2, 2, 3), &dev).unwrap();
        let g = Tensor::from_slice(&[2.0f32, 0.5], (2, 1, 1), &dev).unwrap();
        let w = weight_norm_reconstruct(&g, &v).unwrap();
        let vv = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let gv = [2.0f32, 0.5];
        let wv = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for o in 0..2 {
            let mut nrm = 0f64;
            for j in 0..6 {
                nrm += (vv[o * 6 + j] as f64).powi(2);
            }
            let nrm = nrm.sqrt();
            for j in 0..6 {
                let expect = gv[o] as f64 * vv[o * 6 + j] as f64 / nrm;
                assert!((wv[o * 6 + j] as f64 - expect).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn snakebeta_matches_reference_scalar() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[0.5f32, -0.3, 1.2], (1, 3, 1), &dev).unwrap();
        // alpha=beta=0 (logscale) → exp(0)=1 → x + sin(x)^2 / (1 + 1e-9).
        let alpha = Tensor::zeros((1, 3, 1), DType::F32, &dev).unwrap();
        let beta = Tensor::zeros((1, 3, 1), DType::F32, &dev).unwrap();
        let out = snakebeta(&x, &alpha, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, xv) in [0.5f32, -0.3, 1.2].iter().enumerate() {
            let expect = xv + (1.0 / (1.0 + 1e-9)) * xv.sin().powi(2);
            assert!((out[i] - expect).abs() < 1e-6, "snakebeta[{i}]={}", out[i]);
        }
    }
}
