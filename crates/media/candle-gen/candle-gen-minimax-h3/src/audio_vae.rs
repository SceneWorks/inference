//! The MiniMax-H3 audio VAE decode path (`dac_audio_vae.py::DacAudioVAE.decode`).
//!
//! A **DAC-lineage encoder + BigVGAN decoder**, not an LTX-style audio VAE. Only the decode half is
//! ported: the encoder, `mean_proj`, `logs_proj` and the `AttnProjection` `pre_block` are the
//! encode half and are deliberately absent (173 of the checkpoint's 1087 tensors).
//!
//! ```text
//! decode(z)          z  [B, 32, T]      latents, 40 Hz
//!   dec_in_proj                          1x1 conv, 32 -> 2048
//!   BigVGAN
//!     conv_pre                           2048 -> 1024, k7
//!     x7  ConvTranspose1d                stride 5,5,2,2,2,2,2  (800x)
//!         mean of 3 AMPBlock1            k 3/7/11, dilations 1,3,5
//!     activation_post                    anti-aliased SnakeBeta
//!     conv_post                          8 -> 1, k7, no bias
//!     clamp(-1, 1)                       NOT tanh (`use_tanh_at_final = false`)
//!                    ->  [B, 1, T*800]   mono waveform
//! ```
//!
//! **The decoder is mono.** `conv_post` emits one channel, and `config.yaml` says
//! `audio_channel: 1`, while `config.json` declares `output_channel: 2`. Stereo is therefore two
//! independent 32-channel latents decoded through the same weights — which is what the model card
//! means by independent stereo channels. [`MiniMaxH3AudioVae::decode_stereo`] takes that packing
//! explicitly as `[B, output_channels, latent_channels, T]` and rejects anything else, so a
//! mis-packed latent is a loud error rather than silent garbage.
//!
//! **Weight-norm.** Every conv but `dec_in_proj` ships as `weight_g`/`weight_v`
//! (`w = g · v / ‖v‖`, the norm taken over every axis but the first). The reference re-fuses them
//! at load through torch's parametrization; so does [`MiniMaxH3AudioVae::from_weights`]. The
//! reduction axis differs between the two conv kinds — `Conv1d` weights are `[out, in, k]` so the
//! norm is per **output** channel, `ConvTranspose1d` weights are `[in, out, k]` so it is per
//! **input** channel — and getting that backwards produces a plausible, wrong decode.
//!
//! Everything runs in **NCL**, which is candle's native conv layout *and* the reference's, so this
//! port needs none of the MLX twin's boundary transposes.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::AudioTrack;
use candle_gen::{CandleError, Result, Weights};

use crate::alias_free::{
    flip_kernel, transposed_conv1d, Activation1d, LowPassFilter1d, SnakeBeta, UpSample1d,
};
use crate::audio_config::{
    MiniMaxH3AudioVaeConfig, ACTIVATION_KERNEL_SIZE, ACTIVATION_RESAMPLE_RATIO,
};

/// `get_padding(kernel_size, dilation)` from `dac_utils.py`.
fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

/// Add an `[out]` conv bias to an NCL `[B, out, T]` activation.
fn add_bias(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let c = y.dims()[1];
    Ok(y.broadcast_add(&bias.reshape((1, c, 1))?)?)
}

/// Re-fuse a `weight_norm` pair into a dense weight: `g · v / ‖v‖`, the norm reduced over every
/// axis but axis 0 (torch's `dim=0` default).
fn fuse_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let shape = v.dims();
    if shape.len() != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 audio vae: weight_v must be rank 3, got {shape:?}"
        )));
    }
    if g.dims() != [shape[0], 1, 1] {
        return Err(CandleError::Msg(format!(
            "minimax-h3 audio vae: weight_g {:?} does not match weight_v {shape:?}",
            g.dims()
        )));
    }
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?.broadcast_mul(g)?)
}

/// A `Conv1d` in candle's NCL layout, with optional bias.
#[derive(Debug, Clone)]
struct Conv1d {
    /// `[out, in, kernel]` — torch's own layout, unchanged.
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    /// Load a `weight_norm`-parametrized conv (`weight_g` / `weight_v` / optional `bias`).
    fn weight_normed(
        w: &Weights,
        prefix: &str,
        padding: usize,
        dilation: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Self> {
        let g = w
            .require(&format!("{prefix}.weight_g"))?
            .to_dtype(DType::F32)?;
        let v = w
            .require(&format!("{prefix}.weight_v"))?
            .to_dtype(DType::F32)?;
        let fused = fuse_weight_norm(&g, &v)?;
        let bias = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?)
        } else {
            None
        };
        Ok(Self {
            weight: fused.to_dtype(dtype)?.contiguous()?,
            bias,
            padding,
            dilation,
        })
    }

    fn names(prefix: &str, bias: bool) -> Vec<String> {
        let mut out = vec![format!("{prefix}.weight_g"), format!("{prefix}.weight_v")];
        if bias {
            out.push(format!("{prefix}.bias"));
        }
        out
    }

    /// `[B, C_in, T]` → `[B, C_out, T']`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x
            .contiguous()?
            .conv1d(&self.weight, self.padding, 1, self.dilation, 1)?;
        match &self.bias {
            Some(b) => add_bias(&y, b),
            None => Ok(y),
        }
    }
}

/// A `ConvTranspose1d` upsampler in candle's NCL layout.
#[derive(Debug, Clone)]
struct ConvTranspose1d {
    /// `[out, in, kernel]`, kernel axis reversed — see [`transposed_conv1d`].
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1d {
    /// Torch stores `ConvTranspose1d` weights as `[in, out, kernel]`, so the weight-norm reduction
    /// is per **input** channel and the permute to candle's conv layout is `[1, 0, 2]`.
    ///
    /// The kernel axis is reversed here because the forward pass runs as a zero-insert plus a
    /// *forward* convolution (see [`transposed_conv1d`] for why both backends do it that way).
    fn weight_normed(
        w: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        dtype: DType,
    ) -> Result<Self> {
        let g = w
            .require(&format!("{prefix}.weight_g"))?
            .to_dtype(DType::F32)?;
        let v = w
            .require(&format!("{prefix}.weight_v"))?
            .to_dtype(DType::F32)?;
        let fused = fuse_weight_norm(&g, &v)?.permute((1, 0, 2))?.contiguous()?;
        Ok(Self {
            weight: flip_kernel(&fused)?.to_dtype(dtype)?,
            bias: w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?,
            stride,
            padding,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        vec![
            format!("{prefix}.weight_g"),
            format!("{prefix}.weight_v"),
            format!("{prefix}.bias"),
        ]
    }

    /// `[B, C_in, T]` → `[B, C_out, T·stride]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = transposed_conv1d(x, &self.weight, self.stride, self.padding)?;
        add_bias(&y, &self.bias)
    }
}

/// Load an anti-aliased `SnakeBeta` (`Activation1d(SnakeBeta(...))`) and its two stored filters.
fn load_activation(
    w: &Weights,
    prefix: &str,
    logscale: bool,
    dtype: DType,
) -> Result<Activation1d> {
    let alpha = w
        .require(&format!("{prefix}.act.alpha"))?
        .to_dtype(DType::F32)?;
    let beta = w
        .require(&format!("{prefix}.act.beta"))?
        .to_dtype(DType::F32)?;
    // The Kaiser-sinc taps ship as buffers, exactly as the reference registers them. They are
    // read (never re-derived) so the checkpoint stays the authority; `kaiser_sinc_filter1d`
    // reproducing them is asserted separately in the parity suite and the real-weight smoke.
    let up = w
        .require(&format!("{prefix}.upsample.filter"))?
        .to_dtype(DType::F32)?;
    let down = w
        .require(&format!("{prefix}.downsample.lowpass.filter"))?
        .to_dtype(DType::F32)?;
    for (tag, f) in [("upsample", &up), ("downsample", &down)] {
        if f.dims() != [1, 1, ACTIVATION_KERNEL_SIZE] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: {prefix}.{tag} filter is {:?}, expected \
                 [1, 1, {ACTIVATION_KERNEL_SIZE}]",
                f.dims()
            )));
        }
    }
    Ok(Activation1d::new(
        SnakeBeta::new(alpha, beta, logscale)?,
        UpSample1d::from_filter(up.to_dtype(dtype)?, ACTIVATION_RESAMPLE_RATIO)?,
        LowPassFilter1d::from_filter(down.to_dtype(dtype)?, ACTIVATION_RESAMPLE_RATIO)?,
    ))
}

fn activation_names(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}.act.alpha"),
        format!("{prefix}.act.beta"),
        format!("{prefix}.upsample.filter"),
        format!("{prefix}.downsample.lowpass.filter"),
    ]
}

/// `dac_bigvgan.py::AMPBlock1` — per dilation: `act → conv(dilated) → act → conv(dense)`, residual.
///
/// Public so `tests/audio_vae_parity.rs` can hold one block against the reference in isolation:
/// the activation pairing below is the kind of error an end-to-end golden hides.
#[derive(Debug, Clone)]
pub struct AmpBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acts1: Vec<Activation1d>,
    acts2: Vec<Activation1d>,
}

impl AmpBlock1 {
    /// Load one AMP block from a checkpoint under `prefix`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        kernel: usize,
        dilations: &[usize],
        logscale: bool,
        dtype: DType,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut acts1 = Vec::with_capacity(dilations.len());
        let mut acts2 = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            convs1.push(Conv1d::weight_normed(
                w,
                &format!("{prefix}.convs1.{i}"),
                get_padding(kernel, d),
                d,
                true,
                dtype,
            )?);
            convs2.push(Conv1d::weight_normed(
                w,
                &format!("{prefix}.convs2.{i}"),
                get_padding(kernel, 1),
                1,
                true,
                dtype,
            )?);
            // `acts1 = activations[::2]`, `acts2 = activations[1::2]` — interleaved, not two
            // contiguous halves. A port that split them 0..n / n..2n loads every tensor and
            // still pairs the wrong activation with each conv.
            acts1.push(load_activation(
                w,
                &format!("{prefix}.activations.{}", 2 * i),
                logscale,
                dtype,
            )?);
            acts2.push(load_activation(
                w,
                &format!("{prefix}.activations.{}", 2 * i + 1),
                logscale,
                dtype,
            )?);
        }
        Ok(Self {
            convs1,
            convs2,
            acts1,
            acts2,
        })
    }

    fn names(prefix: &str, dilations: &[usize]) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..dilations.len() {
            out.extend(Conv1d::names(&format!("{prefix}.convs1.{i}"), true));
            out.extend(Conv1d::names(&format!("{prefix}.convs2.{i}"), true));
            out.extend(activation_names(&format!("{prefix}.activations.{}", 2 * i)));
            out.extend(activation_names(&format!(
                "{prefix}.activations.{}",
                2 * i + 1
            )));
        }
        out
    }

    /// `[B, C, T]` → `[B, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.acts1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.acts2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = xt.add(&x)?;
        }
        Ok(x)
    }
}

/// `dac_bigvgan.py::BigVGAN` — the audio VAE's decoder.
///
/// Public so the parity suite can check the vocoder stack on its own, without `dec_in_proj`.
#[derive(Debug, Clone)]
pub struct BigVgan {
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AmpBlock1>,
    activation_post: Activation1d,
    conv_post: Conv1d,
    num_kernels: usize,
    use_tanh_at_final: bool,
}

impl BigVgan {
    /// Load the vocoder from a checkpoint under `prefix` (`"decoder"` in the published naming).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: DType,
    ) -> Result<Self> {
        let h = &cfg.bigvgan;
        let conv_pre = Conv1d::weight_normed(w, &format!("{prefix}.conv_pre"), 3, 1, true, dtype)?;
        let mut ups = Vec::with_capacity(h.num_upsamples());
        for (i, (&rate, &kernel)) in h
            .upsample_rates
            .iter()
            .zip(h.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(ConvTranspose1d::weight_normed(
                w,
                &format!("{prefix}.ups.{i}.0"),
                rate,
                (kernel - rate) / 2,
                dtype,
            )?);
        }
        let mut resblocks = Vec::with_capacity(h.num_upsamples() * h.num_kernels());
        for stage in 0..h.num_upsamples() {
            for (j, (&kernel, dilations)) in h
                .resblock_kernel_sizes
                .iter()
                .zip(h.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let idx = stage * h.num_kernels() + j;
                resblocks.push(AmpBlock1::from_weights(
                    w,
                    &format!("{prefix}.resblocks.{idx}"),
                    kernel,
                    dilations,
                    h.snake_logscale,
                    dtype,
                )?);
            }
        }
        let activation_post = load_activation(
            w,
            &format!("{prefix}.activation_post"),
            h.snake_logscale,
            dtype,
        )?;
        let conv_post = Conv1d::weight_normed(
            w,
            &format!("{prefix}.conv_post"),
            3,
            1,
            h.use_bias_at_final,
            dtype,
        )?;
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_kernels: h.num_kernels(),
            use_tanh_at_final: h.use_tanh_at_final,
        })
    }

    fn names(prefix: &str, cfg: &MiniMaxH3AudioVaeConfig) -> Vec<String> {
        let h = &cfg.bigvgan;
        let mut out = Conv1d::names(&format!("{prefix}.conv_pre"), true);
        for i in 0..h.num_upsamples() {
            out.extend(ConvTranspose1d::names(&format!("{prefix}.ups.{i}.0")));
        }
        for stage in 0..h.num_upsamples() {
            for (j, dilations) in h.resblock_dilation_sizes.iter().enumerate() {
                let idx = stage * h.num_kernels() + j;
                out.extend(AmpBlock1::names(
                    &format!("{prefix}.resblocks.{idx}"),
                    dilations,
                ));
            }
        }
        out.extend(activation_names(&format!("{prefix}.activation_post")));
        out.extend(Conv1d::names(
            &format!("{prefix}.conv_post"),
            h.use_bias_at_final,
        ));
        out
    }

    /// `[B, num_mels, T]` → `[B, 1, T·hop]`, both NCL.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(x)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x)?;
            let mut acc: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let y = self.resblocks[i * self.num_kernels + j].forward(&x)?;
                acc = Some(match acc {
                    Some(prev) => prev.add(&y)?,
                    None => y,
                });
            }
            let acc = acc.ok_or_else(|| {
                CandleError::Msg("minimax-h3 audio vae: a stage has no residual blocks".into())
            })?;
            x = (acc / (self.num_kernels as f64))?;
        }
        let x = self.activation_post.forward(&x)?;
        let x = self.conv_post.forward(&x)?;
        if self.use_tanh_at_final {
            Ok(x.tanh()?)
        } else {
            // `use_tanh_at_final = false` for this checkpoint: a hard clamp, not a tanh. The two
            // agree only near zero, so this is a genuinely different output curve.
            Ok(x.clamp(-1.0, 1.0)?)
        }
    }
}

/// The MiniMax-H3 audio VAE's decode half.
#[derive(Debug, Clone)]
pub struct MiniMaxH3AudioVae {
    /// `[latent_dim, latent_channels, 1]` — a 1×1 conv, kept as a conv rather than reshaped to a
    /// linear because NCL makes the channel axis the convolved one anyway.
    dec_in_w: Tensor,
    dec_in_b: Tensor,
    decoder: BigVgan,
    cfg: MiniMaxH3AudioVaeConfig,
    latents_mean: Tensor,
    latents_std: Tensor,
}

impl MiniMaxH3AudioVae {
    /// Every tensor name the decode path consumes, in the published checkpoint's naming.
    ///
    /// Used by the parity and real-weight tests to prove the mapping is exhaustive in both
    /// directions — a tensor the loader never reads would still decode to something plausible.
    pub fn tensor_names(cfg: &MiniMaxH3AudioVaeConfig) -> Vec<String> {
        let mut out = vec![
            "dec_in_proj.weight".to_string(),
            "dec_in_proj.bias".to_string(),
        ];
        out.extend(BigVgan::names("decoder", cfg));
        out
    }

    /// Load the decode half from a checkpoint in the published naming.
    pub fn from_weights(
        w: &Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let dec_in = w.require("dec_in_proj.weight")?.to_dtype(dtype)?;
        let shape = dec_in.dims().to_vec();
        // `nn.Conv1d(latent_channels, latent_dim, 1)`.
        if shape != [cfg.bigvgan.num_mels, cfg.latent_channels, 1] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: dec_in_proj.weight is {shape:?}, expected [{}, {}, 1]",
                cfg.bigvgan.num_mels, cfg.latent_channels
            )));
        }
        let dec_in_b = w.require("dec_in_proj.bias")?.to_dtype(dtype)?;
        let decoder = BigVgan::from_weights(w, "decoder", cfg, dtype)?;

        let n = cfg.latent_channels;
        let latents_mean =
            Tensor::from_vec(cfg.latents_mean.clone(), n, device)?.to_dtype(dtype)?;
        let latents_std = Tensor::from_vec(cfg.latents_std.clone(), n, device)?.to_dtype(dtype)?;
        Ok(Self {
            dec_in_w: dec_in.contiguous()?,
            dec_in_b,
            decoder,
            cfg: cfg.clone(),
            latents_mean,
            latents_std,
        })
    }

    /// The configuration this instance was built from.
    pub fn config(&self) -> &MiniMaxH3AudioVaeConfig {
        &self.cfg
    }

    /// Per-channel latent de-normalization, `z · latents_std + latents_mean`.
    ///
    /// Applies over the **channel** axis, which is the second-to-last for both the mono
    /// `[B, C, T]` and stereo `[B, 2, C, T]` packings. The pipeline applies this before decoding;
    /// [`Self::decode`] deliberately does not, so it stays a byte-for-byte analogue of the
    /// reference's `DacAudioVAE.decode`.
    pub fn denormalize(&self, z: &Tensor) -> Result<Tensor> {
        let rank = z.dims().len();
        if rank < 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: cannot de-normalize a rank-{rank} latent"
            )));
        }
        let channels = z.dims()[rank - 2];
        if channels != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: latent has {channels} channels, config declares {}",
                self.cfg.latent_channels
            )));
        }
        let mut shape = vec![1usize; rank];
        shape[rank - 2] = channels;
        let mean = self.latents_mean.reshape(shape.clone())?;
        let std = self.latents_std.reshape(shape)?;
        Ok(z.broadcast_mul(&std)?.broadcast_add(&mean)?)
    }

    /// `DacAudioVAE.decode`: `[B, latent_channels, T]` → `[B, 1, T · hop]`.
    ///
    /// Reference-exact — **no** de-normalization (see [`Self::denormalize`]).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let shape = z.dims();
        if shape.len() != 3 || shape[1] != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae decode: expected [B, {}, T], got {shape:?}",
                self.cfg.latent_channels
            )));
        }
        let x =
            z.to_dtype(self.dec_in_w.dtype())?
                .contiguous()?
                .conv1d(&self.dec_in_w, 0, 1, 1, 1)?;
        let x = add_bias(&x, &self.dec_in_b)?;
        self.decoder.forward(&x)
    }

    /// Decode a multi-channel latent: `[B, output_channels, latent_channels, T]` →
    /// `[B, output_channels, T · hop]`.
    ///
    /// The decoder is mono, so each output channel is decoded independently through the same
    /// weights — the channel axis is folded into the batch. The shape is checked against the
    /// configured `output_channels` **and** `latent_channels`, so transposing the two (32 vs 2)
    /// fails loudly instead of decoding nonsense.
    ///
    /// The DiT corroborates this packing: `transformer/config.json` declares
    /// `audio_in_channels: 32`, i.e. ONE 32-channel audio stream per token, not 64.
    pub fn decode_stereo(&self, z: &Tensor) -> Result<Tensor> {
        let shape = z.dims().to_vec();
        let channels = usize::from(self.cfg.output_channels);
        if shape.len() != 4 || shape[1] != channels || shape[2] != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae decode_stereo: expected [B, {channels}, {}, T], got \
                 {shape:?}",
                self.cfg.latent_channels
            )));
        }
        let (b, t) = (shape[0], shape[3]);
        let folded = z
            .contiguous()?
            .reshape((b * channels, self.cfg.latent_channels, t))?;
        let waves = self.decode(&folded)?;
        let samples = waves.dims()[2];
        Ok(waves.reshape((b, channels, samples))?)
    }

    /// The full pipeline-facing decode: de-normalize, decode every channel, interleave.
    ///
    /// `z` is `[1, output_channels, latent_channels, T]`; the result is a `gen-core`
    /// [`AudioTrack`] carrying interleaved f32 PCM (`L0, R0, L1, R1, …`) at the configured sample
    /// rate, with no stems — this model emits a mix, not source-separated parts.
    pub fn decode_audio_track(&self, z: &Tensor) -> Result<AudioTrack> {
        let batch = *z.dims().first().unwrap_or(&0);
        if batch != 1 {
            // The interleave below drops the batch axis; a B>1 latent would silently fold extra
            // batches into the sample stream.
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae decode_audio_track: expected batch size 1, got {batch}"
            )));
        }
        let denorm = self.denormalize(z)?;
        let waves = self.decode_stereo(&denorm)?;
        let shape = waves.dims().to_vec();
        let (channels, samples) = (shape[1], shape[2]);
        // (1, C, S) -> (S, C) -> interleaved.
        let interleaved = waves
            .reshape((channels, samples))?
            .t()?
            .contiguous()?
            .reshape(channels * samples)?
            .to_dtype(DType::F32)?;
        Ok(AudioTrack {
            samples: interleaved.to_vec1::<f32>()?,
            sample_rate: self.cfg.sample_rate,
            channels: self.cfg.output_channels,
            stems: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_matches_the_reference_formula() {
        // `get_padding(k, d) = (k*d - d) / 2`.
        assert_eq!(get_padding(3, 1), 1);
        assert_eq!(get_padding(3, 3), 3);
        assert_eq!(get_padding(3, 5), 5);
        assert_eq!(get_padding(7, 1), 3);
        assert_eq!(get_padding(11, 5), 25);
    }

    /// `w = g · v / ‖v‖` with the norm over axes 1..; a unit `g` renormalizes each slice.
    #[test]
    fn weight_norm_fusion_normalizes_per_leading_slice() {
        let dev = Device::Cpu;
        // Two output channels, one input channel, two taps: rows [3, 4] (norm 5) and [0, 1].
        let v = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 1.0], (2, 1, 2), &dev).unwrap();
        let g = Tensor::from_vec(vec![10.0f32, 2.0], (2, 1, 1), &dev).unwrap();
        let fused = fuse_weight_norm(&g, &v).unwrap();
        assert_eq!(fused.dims(), &[2, 1, 2]);
        let got = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in got.iter().zip([6.0f32, 8.0, 0.0, 2.0].iter()) {
            assert!((a - b).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn weight_norm_fusion_rejects_mismatched_pairs() {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 1, 2), &dev).unwrap();
        // `g` must be [out, 1, 1].
        let g = Tensor::from_vec(vec![1.0f32, 1.0], 2, &dev).unwrap();
        assert!(fuse_weight_norm(&g, &v).is_err());
        let g = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (3, 1, 1), &dev).unwrap();
        assert!(fuse_weight_norm(&g, &v).is_err());
        // `v` must be rank 3.
        let v2 = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &dev).unwrap();
        let g = Tensor::from_vec(vec![1.0f32, 1.0], (2, 1, 1), &dev).unwrap();
        assert!(fuse_weight_norm(&g, &v2).is_err());
    }

    /// The declared name list must have the right cardinality and no duplicates — 914 tensors for
    /// the shipped decoder, which `tests/real_weights.rs` checks against the published checkpoint.
    #[test]
    fn declared_tensor_names_are_unique_and_complete() {
        let cfg = MiniMaxH3AudioVaeConfig::default();
        let names = MiniMaxH3AudioVae::tensor_names(&cfg);
        let unique: std::collections::BTreeSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate declared tensor name");

        // 2 dec_in_proj + 3 conv_pre + 7*3 ups + 21*(2*3*3 + 6*4) + 4 activation_post + 2 conv_post
        assert_eq!(names.len(), 2 + 3 + 21 + 21 * (18 + 24) + 4 + 2);
        assert_eq!(names.len(), 914);

        // `conv_post` ships no bias: `use_bias_at_final = false`.
        assert!(!names.iter().any(|n| n == "decoder.conv_post.bias"));
        assert!(names.iter().any(|n| n == "decoder.conv_post.weight_g"));
        // Activation indices are interleaved 0..5 per AMP block, not two contiguous halves.
        assert!(names
            .iter()
            .any(|n| n == "decoder.resblocks.20.activations.5.downsample.lowpass.filter"));
        assert!(!names
            .iter()
            .any(|n| n == "decoder.resblocks.20.activations.6.act.alpha"));
    }

    /// `use_bias_at_final = true` would add exactly one tensor — proof the flag reaches the
    /// declared key set rather than being an inert field.
    #[test]
    fn use_bias_at_final_changes_the_declared_names() {
        let mut cfg = MiniMaxH3AudioVaeConfig::default();
        cfg.bigvgan.use_bias_at_final = true;
        let names = MiniMaxH3AudioVae::tensor_names(&cfg);
        assert_eq!(names.len(), 915);
        assert!(names.iter().any(|n| n == "decoder.conv_post.bias"));
    }
}
