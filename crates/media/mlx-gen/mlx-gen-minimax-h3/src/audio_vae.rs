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

use mlx_rs::ops::{add, clip, conv1d, divide, multiply, tanh};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::AudioTrack;
use mlx_gen::nn::linear;
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::alias_free::{
    flip_kernel, transposed_conv1d, Activation1d, LowPassFilter1d, SnakeBeta, UpSample1d,
};
use crate::audio_config::{
    MiniMaxH3AudioVaeConfig, ACTIVATION_KERNEL_SIZE, ACTIVATION_RESAMPLE_RATIO,
};

/// `get_padding(kernel_size, dilation)` from `dac_utils.py`.
fn get_padding(kernel: i32, dilation: i32) -> i32 {
    (kernel * dilation - dilation) / 2
}

/// Re-fuse a `weight_norm` pair into a dense weight: `g · v / ‖v‖`, the norm reduced over every
/// axis but axis 0 (torch's `dim=0` default).
fn fuse_weight_norm(g: &Array, v: &Array) -> Result<Array> {
    let shape = v.shape();
    if shape.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 audio vae: weight_v must be rank 3, got {shape:?}"
        )));
    }
    if g.shape() != [shape[0], 1, 1] {
        return Err(Error::Msg(format!(
            "minimax-h3 audio vae: weight_g {:?} does not match weight_v {shape:?}",
            g.shape()
        )));
    }
    let norm = v.square()?.sum_axes(&[1, 2], true)?.sqrt()?;
    Ok(multiply(g, &divide(v, &norm)?)?)
}

/// A `Conv1d` in MLX's NLC layout, with optional bias.
#[derive(Debug, Clone)]
struct Conv1d {
    /// `[out, kernel, in]`.
    weight: Array,
    bias: Option<Array>,
    padding: i32,
    dilation: i32,
}

impl Conv1d {
    /// Load a `weight_norm`-parametrized conv (`weight_g` / `weight_v` / optional `bias`).
    ///
    /// Torch stores `Conv1d` weights as `[out, in, kernel]`; MLX wants `[out, kernel, in]`.
    fn weight_normed(
        w: &mut Weights,
        prefix: &str,
        padding: i32,
        dilation: i32,
        bias: bool,
        dtype: Dtype,
    ) -> Result<Self> {
        let g = to_dtype(w.require(&format!("{prefix}.weight_g"))?, Dtype::Float32)?;
        let v = to_dtype(w.require(&format!("{prefix}.weight_v"))?, Dtype::Float32)?;
        let fused = fuse_weight_norm(&g, &v)?.transpose_axes(&[0, 2, 1])?;
        let bias = if bias {
            Some(to_dtype(w.require(&format!("{prefix}.bias"))?, dtype)?)
        } else {
            None
        };
        Ok(Self {
            weight: fused.as_dtype(dtype)?,
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

    /// `[B, T, C_in]` → `[B, T', C_out]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = conv1d(x, &self.weight, 1, self.padding, self.dilation, 1)?;
        match &self.bias {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

/// A `ConvTranspose1d` upsampler in MLX's NLC layout.
#[derive(Debug, Clone)]
struct ConvTranspose1d {
    /// `[out, kernel, in]`, kernel axis reversed — see [`transposed_conv1d`].
    weight: Array,
    bias: Array,
    stride: i32,
    padding: i32,
}

impl ConvTranspose1d {
    /// Torch stores `ConvTranspose1d` weights as `[in, out, kernel]`, so the weight-norm reduction
    /// is per **input** channel and the MLX permute is `[1, 2, 0]` rather than `[0, 2, 1]`.
    ///
    /// The kernel axis is reversed here because the forward pass runs as a zero-insert plus a
    /// *forward* convolution: MLX's `conv_transpose1d` evaluates in reduced precision on Metal
    /// (see [`transposed_conv1d`]), and these seven stages carry the widest channel counts in the
    /// decoder.
    fn weight_normed(
        w: &mut Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        let g = to_dtype(w.require(&format!("{prefix}.weight_g"))?, Dtype::Float32)?;
        let v = to_dtype(w.require(&format!("{prefix}.weight_v"))?, Dtype::Float32)?;
        let fused = fuse_weight_norm(&g, &v)?.transpose_axes(&[1, 2, 0])?;
        Ok(Self {
            weight: flip_kernel(&fused)?.as_dtype(dtype)?,
            bias: to_dtype(w.require(&format!("{prefix}.bias"))?, dtype)?,
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

    /// `[B, T, C_in]` → `[B, T·stride, C_out]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = transposed_conv1d(x, &self.weight, self.stride, self.padding)?;
        Ok(add(&y, &self.bias)?)
    }
}

/// Load an anti-aliased `SnakeBeta` (`Activation1d(SnakeBeta(...))`) and its two stored filters.
fn load_activation(
    w: &mut Weights,
    prefix: &str,
    logscale: bool,
    dtype: Dtype,
) -> Result<Activation1d> {
    let alpha = to_dtype(w.require(&format!("{prefix}.act.alpha"))?, Dtype::Float32)?;
    let beta = to_dtype(w.require(&format!("{prefix}.act.beta"))?, Dtype::Float32)?;
    // The Kaiser-sinc taps ship as buffers, exactly as the reference registers them. They are
    // read (never re-derived) so the checkpoint stays the authority; `kaiser_sinc_filter1d`
    // reproducing them is asserted separately in the parity suite and the real-weight smoke.
    let up = to_dtype(
        w.require(&format!("{prefix}.upsample.filter"))?,
        Dtype::Float32,
    )?;
    let down = to_dtype(
        w.require(&format!("{prefix}.downsample.lowpass.filter"))?,
        Dtype::Float32,
    )?;
    for (tag, f) in [("upsample", &up), ("downsample", &down)] {
        if f.shape() != [1, 1, ACTIVATION_KERNEL_SIZE] {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae: {prefix}.{tag} filter is {:?}, expected \
                 [1, 1, {ACTIVATION_KERNEL_SIZE}]",
                f.shape()
            )));
        }
    }
    Ok(Activation1d::new(
        SnakeBeta::new(alpha, beta, logscale)?,
        UpSample1d::from_filter(up.as_dtype(dtype)?, ACTIVATION_RESAMPLE_RATIO)?,
        LowPassFilter1d::from_filter(down.as_dtype(dtype)?, ACTIVATION_RESAMPLE_RATIO)?,
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
        w: &mut Weights,
        prefix: &str,
        kernel: i32,
        dilations: &[i32],
        logscale: bool,
        dtype: Dtype,
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

    fn names(prefix: &str, dilations: &[i32]) -> Vec<String> {
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

    /// `[B, T, C]` → `[B, T, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.acts1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.acts2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = add(&xt, &x)?;
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
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: Dtype,
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

    /// `[B, T, num_mels]` → `[B, T·hop, 1]`, both NLC.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv_pre.forward(x)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x)?;
            let mut acc: Option<Array> = None;
            for j in 0..self.num_kernels {
                let y = self.resblocks[i * self.num_kernels + j].forward(&x)?;
                acc = Some(match acc {
                    Some(prev) => add(&prev, &y)?,
                    None => y,
                });
            }
            let acc = acc.ok_or_else(|| {
                Error::Msg("minimax-h3 audio vae: a stage has no residual blocks".into())
            })?;
            x = divide(&acc, Array::from_f32(self.num_kernels as f32))?;
        }
        let x = self.activation_post.forward(&x)?;
        let x = self.conv_post.forward(&x)?;
        if self.use_tanh_at_final {
            Ok(tanh(&x)?)
        } else {
            // `use_tanh_at_final = false` for this checkpoint: a hard clamp, not a tanh. The two
            // agree only near zero, so this is a genuinely different output curve.
            Ok(clip(&x, (Array::from_f32(-1.0), Array::from_f32(1.0)))?)
        }
    }
}

/// The MiniMax-H3 audio VAE's decode half.
#[derive(Debug, Clone)]
pub struct MiniMaxH3AudioVae {
    dec_in_w: Array,
    dec_in_b: Array,
    decoder: BigVgan,
    cfg: MiniMaxH3AudioVaeConfig,
    latents_mean: Array,
    latents_std: Array,
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
        w: &mut Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let dec_in = to_dtype(w.require("dec_in_proj.weight")?, dtype)?;
        let shape = dec_in.shape().to_vec();
        // `nn.Conv1d(latent_channels, latent_dim, 1)`: a 1x1 conv is a pointwise linear, so the
        // trailing kernel axis is squeezed rather than convolved.
        if shape != [cfg.bigvgan.num_mels, cfg.latent_channels, 1] {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae: dec_in_proj.weight is {shape:?}, expected [{}, {}, 1]",
                cfg.bigvgan.num_mels, cfg.latent_channels
            )));
        }
        let dec_in_w = dec_in.reshape(&[cfg.bigvgan.num_mels, cfg.latent_channels])?;
        let dec_in_b = to_dtype(w.require("dec_in_proj.bias")?, dtype)?;
        let decoder = BigVgan::from_weights(w, "decoder", cfg, dtype)?;

        let n = cfg.latent_channels;
        let latents_mean = Array::from_slice(&cfg.latents_mean, &[1, 1, n]).as_dtype(dtype)?;
        let latents_std = Array::from_slice(&cfg.latents_std, &[1, 1, n]).as_dtype(dtype)?;
        Ok(Self {
            dec_in_w,
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
    pub fn denormalize(&self, z: &Array) -> Result<Array> {
        let rank = z.shape().len();
        if rank < 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae: cannot de-normalize a rank-{rank} latent"
            )));
        }
        let channels = z.shape()[rank - 2];
        if channels != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae: latent has {channels} channels, config declares {}",
                self.cfg.latent_channels
            )));
        }
        // Statistics are stored `[1, 1, C]`; reshape to broadcast against the channel axis.
        let mut shape = vec![1; rank];
        shape[rank - 2] = channels;
        let mean = self.latents_mean.reshape(&shape)?;
        let std = self.latents_std.reshape(&shape)?;
        Ok(add(&multiply(z, &std)?, &mean)?)
    }

    /// `DacAudioVAE.decode`: `[B, latent_channels, T]` → `[B, 1, T · hop]`.
    ///
    /// Reference-exact — **no** de-normalization (see [`Self::denormalize`]).
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let shape = z.shape();
        if shape.len() != 3 || shape[1] != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae decode: expected [B, {}, T], got {shape:?}",
                self.cfg.latent_channels
            )));
        }
        // NCL -> NLC once, at the boundary.
        let x = z.transpose_axes(&[0, 2, 1])?;
        let x = linear(&x, &self.dec_in_w, &self.dec_in_b)?;
        let y = self.decoder.forward(&x)?;
        Ok(y.transpose_axes(&[0, 2, 1])?)
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
    /// `audio_in_channels: 32`, i.e. ONE 32-channel audio stream per token, not 64. Whether the
    /// pipeline carries the stereo pair as a leading axis (this signature) or has already folded
    /// it into the batch is an sc-17146/17147 integration detail; either way the shape check above
    /// turns a mis-packed latent into an error rather than silent garbage.
    pub fn decode_stereo(&self, z: &Array) -> Result<Array> {
        let shape = z.shape().to_vec();
        let channels = i32::from(self.cfg.output_channels);
        if shape.len() != 4 || shape[1] != channels || shape[2] != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae decode_stereo: expected [B, {channels}, {}, T], got \
                 {shape:?}",
                self.cfg.latent_channels
            )));
        }
        let (b, t) = (shape[0], shape[3]);
        let folded = z.reshape(&[b * channels, self.cfg.latent_channels, t])?;
        let waves = self.decode(&folded)?;
        let samples = waves.shape()[2];
        Ok(waves.reshape(&[b, channels, samples])?)
    }

    /// The full pipeline-facing decode: de-normalize, decode every channel, interleave.
    ///
    /// `z` is `[1, output_channels, latent_channels, T]`; the result is a `gen-core`
    /// [`AudioTrack`] carrying interleaved f32 PCM (`L0, R0, L1, R1, …`) at the configured sample
    /// rate, with no stems — this model emits a mix, not source-separated parts.
    pub fn decode_audio_track(&self, z: &Array) -> Result<AudioTrack> {
        let batch = *z.shape().first().unwrap_or(&0);
        if batch != 1 {
            // The interleave below drops the batch axis; a B>1 latent would silently fold extra
            // batches into the sample stream.
            return Err(Error::Msg(format!(
                "minimax-h3 audio vae decode_audio_track: expected batch size 1, got {batch}"
            )));
        }
        let denorm = self.denormalize(z)?;
        let waves = self.decode_stereo(&denorm)?;
        let shape = waves.shape().to_vec();
        let (channels, samples) = (shape[1], shape[2]);
        // (1, C, S) -> (S, C) -> interleaved.
        let interleaved = waves
            .reshape(&[channels, samples])?
            .transpose_axes(&[1, 0])?
            .reshape(&[channels * samples])?
            .as_dtype(Dtype::Float32)?;
        Ok(AudioTrack {
            samples: interleaved.as_slice::<f32>().to_vec(),
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
        // Two output channels, one input channel, two taps: rows [3, 4] (norm 5) and [0, 1].
        let v = Array::from_slice(&[3.0f32, 4.0, 0.0, 1.0], &[2, 1, 2]);
        let g = Array::from_slice(&[10.0f32, 2.0], &[2, 1, 1]);
        let fused = fuse_weight_norm(&g, &v).unwrap();
        assert_eq!(fused.shape(), &[2, 1, 2]);
        let got = fused.as_slice::<f32>();
        for (a, b) in got.iter().zip([6.0f32, 8.0, 0.0, 2.0].iter()) {
            assert!((a - b).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn weight_norm_fusion_rejects_mismatched_pairs() {
        let v = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 1, 2]);
        // `g` must be [out, 1, 1].
        let g = Array::from_slice(&[1.0f32, 1.0], &[2]);
        assert!(fuse_weight_norm(&g, &v).is_err());
        let g = Array::from_slice(&[1.0f32, 1.0, 1.0], &[3, 1, 1]);
        assert!(fuse_weight_norm(&g, &v).is_err());
        // `v` must be rank 3.
        let v2 = Array::from_slice(&[1.0f32, 2.0], &[2, 1]);
        let g = Array::from_slice(&[1.0f32, 1.0], &[2, 1, 1]);
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
