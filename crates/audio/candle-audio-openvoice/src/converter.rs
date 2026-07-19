//! The OpenVoice V2 flow-based VITS converter — faithful candle ports of the `SynthesizerTrn`
//! submodules the tone-color conversion path runs (`models.py` / `modules.py`):
//!
//! - `Wn` — the WaveNet residual stack (`modules.WN`) with `fused_add_tanh_sigmoid_multiply`
//!   gating and per-layer speaker conditioning via a `cond_layer`,
//! - [`PosteriorEncoder`] — `enc_q`: `pre` Conv1d → `Wn` → `proj` Conv1d, sampling
//!   `z = m + ε·τ·exp(logs)`,
//! - [`ResidualCouplingBlock`] — `flow`: `N_FLOWS` mean-only `ResidualCouplingLayer`s each
//!   followed by a channel flip, run forward (conditioned on the source tone color) then in
//!   reverse (conditioned on the target),
//! - [`Generator`] — `dec`: the HiFi-GAN vocoder (`conv_pre` + speaker `cond` bias, four
//!   ConvTranspose1d upsamples each summing three `ResBlock1`s, `conv_post` + tanh).
//!
//! Conversion runs a **single full-length sequence**, so VITS's `x_mask` is all ones and dropped
//! everywhere (a no-op multiply). Per `zero_g = true` ([`config::ZERO_G`]) the posterior encoder and
//! decoder are conditioned on a **zeroed** `g` (their `cond` layers contribute only their bias);
//! the whole timbre transfer therefore lives in the flow.

use candle_audio::candle_core::{Device, IndexOp, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::{
    conv1d, conv1d_no_bias, conv_transpose1d, ops, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Module, VarBuilder,
};

use crate::config;

/// `modules.WN`: `n_layers` dilated (here dilation-1) gated residual conv blocks with per-layer
/// speaker conditioning. `forward` takes the already-`pre`-projected `[1, hidden, T]` activation and
/// a `[1, gin, 1]` conditioning tensor (zeros for the zero-`g` posterior encoder).
struct Wn {
    in_layers: Vec<Conv1d>,
    res_skip_layers: Vec<Conv1d>,
    cond_layer: Conv1d,
    hidden: usize,
    n_layers: usize,
}

impl Wn {
    fn new(hidden: usize, kernel: usize, n_layers: usize, vb: VarBuilder) -> Result<Self> {
        let pad = (kernel - 1) / 2; // dilation 1
        let in_cfg = Conv1dConfig {
            padding: pad,
            ..Default::default()
        };
        let mut in_layers = Vec::with_capacity(n_layers);
        let mut res_skip_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            in_layers.push(conv1d(
                hidden,
                2 * hidden,
                kernel,
                in_cfg,
                vb.pp(format!("in_layers.{i}")),
            )?);
            let res_skip_channels = if i < n_layers - 1 { 2 * hidden } else { hidden };
            res_skip_layers.push(conv1d(
                hidden,
                res_skip_channels,
                1,
                Conv1dConfig::default(),
                vb.pp(format!("res_skip_layers.{i}")),
            )?);
        }
        let cond_layer = conv1d(
            config::GIN_CHANNELS,
            2 * hidden * n_layers,
            1,
            Conv1dConfig::default(),
            vb.pp("cond_layer"),
        )?;
        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            hidden,
            n_layers,
        })
    }

    fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let mut output: Option<Tensor> = None;
        let g = self.cond_layer.forward(g)?; // [1, 2·hidden·n_layers, 1]
        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(&x)?; // [1, 2·hidden, T]
            let offset = i * 2 * self.hidden;
            let g_l = g.i((.., offset..offset + 2 * self.hidden, ..))?; // [1, 2·hidden, 1]
            let in_act = x_in.broadcast_add(&g_l)?;
            let t_act = in_act.i((.., ..self.hidden, ..))?.tanh()?;
            let s_act = ops::sigmoid(&in_act.i((.., self.hidden.., ..))?)?;
            let acts = (t_act * s_act)?; // [1, hidden, T]
            let res_skip = self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res_acts = res_skip.i((.., ..self.hidden, ..))?;
                x = (x + res_acts)?;
                let skip = res_skip.i((.., self.hidden.., ..))?;
                output = Some(match output {
                    Some(o) => (o + skip)?,
                    None => skip,
                });
            } else {
                output = Some(match output {
                    Some(o) => (o + res_skip)?,
                    None => res_skip,
                });
            }
        }
        output.ok_or_else(|| AudioError::Msg("WN with zero layers".into()))
    }
}

/// `models.PosteriorEncoder` (`enc_q`): the linear spectrogram → latent `z` sampler.
pub struct PosteriorEncoder {
    pre: Conv1d,
    enc: Wn,
    proj: Conv1d,
    out_channels: usize,
}

impl PosteriorEncoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let pre = conv1d(
            config::SPEC_CHANNELS,
            config::HIDDEN_CHANNELS,
            1,
            Conv1dConfig::default(),
            vb.pp("pre"),
        )?;
        let enc = Wn::new(
            config::HIDDEN_CHANNELS,
            config::WN_KERNEL_SIZE,
            config::ENC_Q_N_LAYERS,
            vb.pp("enc"),
        )?;
        let proj = conv1d(
            config::HIDDEN_CHANNELS,
            config::INTER_CHANNELS * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("proj"),
        )?;
        Ok(Self {
            pre,
            enc,
            proj,
            out_channels: config::INTER_CHANNELS,
        })
    }

    /// `spec: [1, spec_channels, T]`, `g: [1, gin, 1]` (zeros under zero_g), `noise: [1, inter, T]`
    /// standard-Gaussian samples → latent `z = m + noise·τ·exp(logs)` `[1, inter, T]`.
    pub fn forward(&self, spec: &Tensor, g: &Tensor, tau: f32, noise: &Tensor) -> Result<Tensor> {
        let x = self.pre.forward(spec)?;
        let x = self.enc.forward(&x, g)?;
        let stats = self.proj.forward(&x)?;
        let m = stats.i((.., ..self.out_channels, ..))?;
        let logs = stats.i((.., self.out_channels.., ..))?;
        let z = (&m + ((noise * tau as f64)? * logs.exp()?)?)?;
        Ok(z)
    }
}

/// `modules.ResidualCouplingLayer` (mean-only): `pre` Conv1d → [`Wn`] → `post` Conv1d producing the
/// coupling mean `m`. Splits channels in half; the first half passes through and conditions the
/// affine shift of the second.
struct ResidualCouplingLayer {
    pre: Conv1d,
    enc: Wn,
    post: Conv1d,
    half: usize,
}

impl ResidualCouplingLayer {
    fn new(vb: VarBuilder) -> Result<Self> {
        let half = config::INTER_CHANNELS / 2;
        let pre = conv1d(
            half,
            config::HIDDEN_CHANNELS,
            1,
            Conv1dConfig::default(),
            vb.pp("pre"),
        )?;
        let enc = Wn::new(
            config::HIDDEN_CHANNELS,
            config::WN_KERNEL_SIZE,
            config::FLOW_N_LAYERS,
            vb.pp("enc"),
        )?;
        // mean_only ⇒ post emits `half` channels (the mean); logs ≡ 0.
        let post = conv1d(
            config::HIDDEN_CHANNELS,
            half,
            1,
            Conv1dConfig::default(),
            vb.pp("post"),
        )?;
        Ok(Self {
            pre,
            enc,
            post,
            half,
        })
    }

    fn mean(&self, x0: &Tensor, g: &Tensor) -> Result<Tensor> {
        let h = self.pre.forward(x0)?;
        let h = self.enc.forward(&h, g)?;
        Ok(self.post.forward(&h)?)
    }

    fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x0 = x.i((.., ..self.half, ..))?;
        let x1 = x.i((.., self.half.., ..))?;
        let m = self.mean(&x0, g)?;
        let x1 = (x1 + m)?; // logs ≡ 0 ⇒ exp(logs) = 1
        Tensor::cat(&[&x0, &x1], 1).map_err(Into::into)
    }

    fn reverse(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x0 = x.i((.., ..self.half, ..))?;
        let x1 = x.i((.., self.half.., ..))?;
        let m = self.mean(&x0, g)?;
        let x1 = (x1 - m)?;
        Tensor::cat(&[&x0, &x1], 1).map_err(Into::into)
    }
}

/// `torch.flip(x, [1])` — reverse the channel dimension.
fn flip_channels(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let idx: Vec<u32> = (0..c as u32).rev().collect();
    let idx = Tensor::from_vec(idx, c, x.device())?;
    Ok(x.index_select(&idx, 1)?)
}

/// `models.ResidualCouplingBlock` (`flow`): `N_FLOWS` `[coupling, flip]` pairs.
pub struct ResidualCouplingBlock {
    couplings: Vec<ResidualCouplingLayer>,
}

impl ResidualCouplingBlock {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let mut couplings = Vec::with_capacity(config::N_FLOWS);
        for i in 0..config::N_FLOWS {
            // Module list is `[coupling, flip] × N_FLOWS`, so couplings sit at even indices.
            couplings.push(ResidualCouplingLayer::new(
                vb.pp(format!("flows.{}", 2 * i)),
            )?);
        }
        Ok(Self { couplings })
    }

    /// Forward pass (training direction): each coupling then a channel flip, in order.
    pub fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for coupling in &self.couplings {
            x = coupling.forward(&x, g)?;
            x = flip_channels(&x)?;
        }
        Ok(x)
    }

    /// Reverse pass: iterate the `[coupling, flip]` module list in reverse — each flip (a plain
    /// channel reverse) then its coupling's inverse.
    pub fn reverse(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for coupling in self.couplings.iter().rev() {
            x = flip_channels(&x)?;
            x = coupling.reverse(&x, g)?;
        }
        Ok(x)
    }
}

/// `modules.ResBlock1`: three dilated conv pairs, each `leaky_relu → convs1[j] → leaky_relu →
/// convs2[j]`, residual-added.
struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn new(channels: usize, kernel: usize, dilations: &[usize], vb: VarBuilder) -> Result<Self> {
        let get_padding = |k: usize, d: usize| (k * d - d) / 2;
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (j, &d) in dilations.iter().enumerate() {
            convs1.push(conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, d),
                    dilation: d,
                    ..Default::default()
                },
                vb.pp(format!("convs1.{j}")),
            )?);
            convs2.push(conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, 1),
                    ..Default::default()
                },
                vb.pp(format!("convs2.{j}")),
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let xt = ops::leaky_relu(&x, config::LRELU_SLOPE)?;
            let xt = c1.forward(&xt)?;
            let xt = ops::leaky_relu(&xt, config::LRELU_SLOPE)?;
            let xt = c2.forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

/// `models.Generator` (`dec`): the HiFi-GAN vocoder that renders the flow's latent back to audio.
pub struct Generator {
    conv_pre: Conv1d,
    cond: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1d,
    num_kernels: usize,
}

impl Generator {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let conv_pre = conv1d(
            config::INTER_CHANNELS,
            config::UPSAMPLE_INITIAL_CHANNEL,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;
        let cond = conv1d(
            config::GIN_CHANNELS,
            config::UPSAMPLE_INITIAL_CHANNEL,
            1,
            Conv1dConfig::default(),
            vb.pp("cond"),
        )?;
        let mut ups = Vec::with_capacity(config::UPSAMPLE_RATES.len());
        for (i, (&u, &k)) in config::UPSAMPLE_RATES
            .iter()
            .zip(&config::UPSAMPLE_KERNEL_SIZES)
            .enumerate()
        {
            let in_ch = config::UPSAMPLE_INITIAL_CHANNEL / (1 << i);
            let out_ch = config::UPSAMPLE_INITIAL_CHANNEL / (1 << (i + 1));
            ups.push(conv_transpose1d(
                in_ch,
                out_ch,
                k,
                ConvTranspose1dConfig {
                    padding: (k - u) / 2,
                    output_padding: 0,
                    stride: u,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("ups.{i}")),
            )?);
        }
        let mut resblocks = Vec::new();
        let mut last_ch = 0usize;
        for i in 0..config::UPSAMPLE_RATES.len() {
            let ch = config::UPSAMPLE_INITIAL_CHANNEL / (1 << (i + 1));
            last_ch = ch;
            for (j, (&k, d)) in config::RESBLOCK_KERNEL_SIZES
                .iter()
                .zip(&config::RESBLOCK_DILATIONS)
                .enumerate()
            {
                resblocks.push(ResBlock1::new(
                    ch,
                    k,
                    d,
                    vb.pp(format!(
                        "resblocks.{}",
                        i * config::RESBLOCK_KERNEL_SIZES.len() + j
                    )),
                )?);
            }
        }
        let conv_post = conv1d_no_bias(
            last_ch,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;
        Ok(Self {
            conv_pre,
            cond,
            ups,
            resblocks,
            conv_post,
            num_kernels: config::RESBLOCK_KERNEL_SIZES.len(),
        })
    }

    /// `x: [1, inter, T]`, `g: [1, gin, 1]` (zeros under zero_g — `cond` contributes only its bias)
    /// → waveform `[1, 1, T·∏upsample_rates]`.
    pub fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(x)?;
        x = x.broadcast_add(&self.cond.forward(g)?)?;
        for i in 0..self.ups.len() {
            x = ops::leaky_relu(&x, config::LRELU_SLOPE)?;
            x = self.ups[i].forward(&x)?;
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let r = self.resblocks[i * self.num_kernels + j].forward(&x)?;
                xs = Some(match xs {
                    Some(acc) => (acc + r)?,
                    None => r,
                });
            }
            x = (xs.expect("num_kernels >= 1") / self.num_kernels as f64)?;
        }
        // Final activation uses torch's DEFAULT leaky-relu slope (0.01), not LRELU_SLOPE.
        x = ops::leaky_relu(&x, 0.01)?;
        x = self.conv_post.forward(&x)?;
        Ok(x.tanh()?)
    }
}

/// The assembled converter: `enc_q` + `flow` + `dec`, running OpenVoice's `voice_conversion`.
pub struct VoiceConverter {
    enc_q: PosteriorEncoder,
    flow: ResidualCouplingBlock,
    dec: Generator,
    device: Device,
}

impl VoiceConverter {
    pub fn new(vb: VarBuilder, device: Device) -> Result<Self> {
        Ok(Self {
            enc_q: PosteriorEncoder::new(vb.pp("enc_q"))?,
            flow: ResidualCouplingBlock::new(vb.pp("flow"))?,
            dec: Generator::new(vb.pp("dec"))?,
            device,
        })
    }

    /// `models.SynthesizerTrn.voice_conversion` under `zero_g = true`: posterior-encode the source
    /// spectrogram (zeroed `g`), flow-forward on the source tone color, flow-reverse on the target
    /// tone color, decode (zeroed `g`). Returns the mono waveform `[T·hop]`.
    ///
    /// `spec` is `[1, spec_channels, T]`; `g_src` / `g_tgt` are `[1, gin, 1]`; `noise` is a
    /// `[1, inter, T]` standard-Gaussian tensor; `cancel` is polled between the heavy stages.
    pub fn voice_conversion(
        &self,
        spec: &Tensor,
        g_src: &Tensor,
        g_tgt: &Tensor,
        tau: f32,
        noise: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let zero_g = Tensor::zeros((1, config::GIN_CHANNELS, 1), spec.dtype(), &self.device)?;
        let enc_g = if config::ZERO_G { &zero_g } else { g_src };
        let dec_g = if config::ZERO_G { &zero_g } else { g_tgt };

        let z = self.enc_q.forward(spec, enc_g, tau, noise)?;
        if cancel() {
            return Err(AudioError::Canceled);
        }
        let z_p = self.flow.forward(&z, g_src)?;
        if cancel() {
            return Err(AudioError::Canceled);
        }
        let z_hat = self.flow.reverse(&z_p, g_tgt)?;
        if cancel() {
            return Err(AudioError::Canceled);
        }
        let o = self.dec.forward(&z_hat, dec_g)?; // [1, 1, T·hop]
        let samples = o.i((0, 0))?.to_vec1::<f32>()?;
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::DType;

    #[test]
    fn flip_channels_reverses_channel_dim() {
        let dev = Device::Cpu;
        // [1, 3, 2] with channels [a,b,c] → [c,b,a].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &dev).unwrap();
        let f = flip_channels(&x).unwrap();
        let v: Vec<f32> = f.i((0,)).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // channel 2 (5,6), channel 1 (3,4), channel 0 (1,2).
        assert_eq!(v, [5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn flip_is_its_own_inverse() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 8, 5), &dev).unwrap();
        let back = flip_channels(&flip_channels(&x).unwrap()).unwrap();
        let a: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = back.flatten_all().unwrap().to_vec1().unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_dtype_ops_are_wired() {
        // Guard: the noise/tau arithmetic in PosteriorEncoder::forward must type-check on f32.
        let dev = Device::Cpu;
        let m = Tensor::zeros((1, 4, 3), DType::F32, &dev).unwrap();
        let noise = Tensor::ones((1, 4, 3), DType::F32, &dev).unwrap();
        let z = (&m + ((noise * 0.3f64).unwrap() * m.exp().unwrap()).unwrap()).unwrap();
        assert_eq!(z.dims(), &[1, 4, 3]);
    }
}
