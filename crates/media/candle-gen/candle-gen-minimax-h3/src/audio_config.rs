//! MiniMax-H3 audio-VAE configuration (`MiniMaxH3AudioVAE` / `DacAudioVAE`).
//!
//! The audio VAE ships in two published forms with **identical tensor names** (1087 each) but
//! different configs:
//!
//! | | `FL2VA/audio_vae/` | root `audio_vae/` |
//! |---|---|---|
//! | class | `MiniMaxH3AudioVAE` (remote code) | `AutoencoderKLMiniMaxH3Audio` (diffusers) |
//! | weights | `model.safetensors` | `diffusion_pytorch_model.safetensors` |
//! | config | `config.json` + **`config.yaml`** + **`metadata.json`** | `config.json` only |
//!
//! Only the FL2VA triple is loadable by the Apache-2.0 reference — its `from_pretrained` reads
//! `source_config_path` / `source_metadata_path` / `source_safetensors_path` out of `config.json`,
//! and the root directory ships none of those. [`MiniMaxH3AudioVaeConfig::from_source_files`] is
//! therefore the primary constructor and takes exactly the three documents the reference reads;
//! [`MiniMaxH3AudioVaeConfig::cross_check_diffusers_json`] validates the repackaged root config
//! against it.
//!
//! ## What no config file says
//!
//! `DacAudioVAE.__init__` selects the whole BigVGAN hyper-parameter block from a hardcoded
//! `if sample_rate == 32000` branch. Five of those knobs appear in **no** published config and
//! leave **no** tensor behind, so a port reconstructed from the checkpoint alone would silently
//! take BigVGAN's upstream defaults and be wrong:
//!
//! | knob | upstream default | this checkpoint | how a wrong value shows up |
//! |---|---|---|---|
//! | `activation` | — | `snakebeta` | — (a different activation would need other tensors) |
//! | `snake_logscale` | `False` | **`True`** | `alpha`/`beta` used raw instead of `exp(·)` |
//! | `use_tanh_at_final` | `True` | **`False`** | `tanh` instead of `clamp(-1, 1)` |
//! | `use_bias_at_final` | `True` | **`False`** | corroborated: `conv_post` ships no `bias` |
//! | `upsample_kernel_sizes` | — | `[9,9,4,4,4,4,4]` | corroborated by the `ups.*` shapes |
//!
//! Two more are structural rather than free:
//!
//! * **`latent_dim` is derived, not passed.** `from_pretrained` never forwards `latent_dim`, so the
//!   constructor computes `encoder_dim · 2^len(encoder_rates)` from its own `encoder_dim = 64`
//!   default and `metadata.json`'s five `encoder_rates` → 2048. `metadata.json` also *states*
//!   `latent_dim: 2048` and `encoder_dim: 64`, which this module cross-checks rather than reads.
//! * **`decoder_rates` is inert for the decoder.** `DacAudioVAE` stores it but the BigVGAN
//!   `upsample_rates` come from the `sample_rate` branch. They agree here, and
//!   [`MiniMaxH3AudioVaeConfig::from_source_files`] asserts they do — a checkpoint where they
//!   diverged would otherwise decode at the wrong rate with no error.

use candle_gen::{CandleError, Result};

/// Published output sample rate.
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;
/// Published latent width (`vae_latent_channels` / `latent_channels`).
pub const AUDIO_LATENT_CHANNELS: usize = 32;
/// Published output channel count — stereo, decoded as two independent mono waveforms.
pub const AUDIO_OUTPUT_CHANNELS: u16 = 2;
/// Latent token rate: `sample_rate / ∏ upsample_rates` = `32000 / 800`.
pub const AUDIO_TOKEN_RATE_HZ: u32 = 40;
/// The `encoder_dim` default `from_pretrained` leaves in place; `latent_dim` derives from it.
pub const DEFAULT_ENCODER_DIM: usize = 64;

/// `Activation1d`'s up/down ratio. A constructor default in `dac_alias_free_act.py`, which
/// `dac_bigvgan.py` never overrides — it is in no config file.
pub const ACTIVATION_RESAMPLE_RATIO: usize = 2;
/// `Activation1d`'s Kaiser-sinc kernel size, likewise a never-overridden constructor default.
/// Corroborated by every stored `filter` buffer being `[1, 1, 12]`.
pub const ACTIVATION_KERNEL_SIZE: usize = 12;

/// Per-channel latent de-normalization mean (`latents_mean`, 32 entries).
pub const AUDIO_LATENTS_MEAN: [f32; AUDIO_LATENT_CHANNELS] = [
    -0.020211687,
    0.38764665,
    -0.043982796,
    -0.28591514,
    0.08179686,
    -0.3578264,
    0.04062381,
    -0.015525345,
    -0.22336248,
    0.18210068,
    0.2941779,
    -0.07901168,
    -0.056815073,
    -0.36990282,
    -0.31616315,
    0.5905951,
    -0.05213957,
    0.01367316,
    -0.03691648,
    0.09732661,
    -0.33946624,
    -0.30685678,
    -0.24504599,
    -0.034698524,
    0.028680323,
    -0.2121778,
    -0.16782631,
    0.3221288,
    -0.12230559,
    0.43566048,
    -0.05025992,
    0.39792582,
];

/// Per-channel latent de-normalization standard deviation (`latents_std`, 32 entries).
pub const AUDIO_LATENTS_STD: [f32; AUDIO_LATENT_CHANNELS] = [
    1.6895524, 2.7626374, 1.7945344, 1.6801682, 1.6390227, 2.7788298, 1.765909, 1.6199758,
    2.6336524, 1.8539357, 2.5056498, 1.8110192, 1.9579657, 1.6685498, 1.492247, 3.2986703,
    1.9491805, 1.8720003, 1.833408, 1.648807, 1.6176958, 1.913145, 1.5695245, 1.694366, 1.8318421,
    1.5540638, 1.9344931, 1.5991982, 1.718046, 1.6307219, 1.8661226, 1.5613768,
];

/// The BigVGAN hyper-parameter block (`dac_audio_vae.py`'s `bigvgan_conf` → `AttrDict`).
#[derive(Debug, Clone, PartialEq)]
pub struct BigVganConfig {
    /// `conv_pre` input channels — the audio VAE's `latent_dim`.
    pub num_mels: usize,
    /// Per-stage transposed-conv strides. `∏` is the hop length.
    pub upsample_rates: Vec<usize>,
    /// Per-stage transposed-conv kernel sizes.
    pub upsample_kernel_sizes: Vec<usize>,
    /// `conv_pre` output channels; halves once per upsample stage.
    pub upsample_initial_channel: usize,
    /// One AMP block per (stage, kernel).
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilations per AMP block, parallel to `resblock_kernel_sizes`.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// `false` here — the final bound is `clamp(-1, 1)`, **not** `tanh`.
    pub use_tanh_at_final: bool,
    /// `false` here — `conv_post` has no bias tensor.
    pub use_bias_at_final: bool,
    /// `true` here — `SnakeBeta` exponentiates `alpha` and `beta`.
    pub snake_logscale: bool,
}

impl BigVganConfig {
    /// The `sample_rate`-selected block, mirroring `DacAudioVAE.__init__`.
    ///
    /// Only 16 kHz and 32 kHz exist upstream; every other rate is a hard error there and here.
    pub fn for_sample_rate(sample_rate: u32, num_mels: usize, decoder_dim: usize) -> Result<Self> {
        let (upsample_rates, upsample_kernel_sizes) = match sample_rate {
            16_000 => (vec![5, 5, 2, 2, 2, 2], vec![9, 9, 4, 4, 4, 4]),
            32_000 => (vec![5, 5, 2, 2, 2, 2, 2], vec![9, 9, 4, 4, 4, 4, 4]),
            other => {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio vae: unsupported sample_rate {other} (the reference \
                     implementation defines a BigVGAN configuration only for 16000 and 32000)"
                )))
            }
        };
        Ok(Self {
            num_mels,
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel: decoder_dim,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_tanh_at_final: false,
            use_bias_at_final: false,
            snake_logscale: true,
        })
    }

    /// Upsample stages.
    pub fn num_upsamples(&self) -> usize {
        self.upsample_rates.len()
    }

    /// AMP blocks per stage.
    pub fn num_kernels(&self) -> usize {
        self.resblock_kernel_sizes.len()
    }

    /// Channel width entering stage `i` (`upsample_initial_channel / 2^i`).
    pub fn stage_in_channels(&self, stage: usize) -> usize {
        self.upsample_initial_channel >> stage
    }

    /// Channel width leaving stage `i` — also the AMP-block width for that stage.
    pub fn stage_out_channels(&self, stage: usize) -> usize {
        self.upsample_initial_channel >> (stage + 1)
    }

    /// Total upsampling factor — samples emitted per latent token.
    pub fn hop_length(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    fn validate(&self) -> Result<()> {
        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() {
            return Err(CandleError::Msg(
                "minimax-h3 audio vae: upsample_rates and upsample_kernel_sizes differ in length"
                    .into(),
            ));
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(CandleError::Msg(
                "minimax-h3 audio vae: resblock kernel/dilation tables differ in length".into(),
            ));
        }
        if self.num_upsamples() == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 audio vae: a BigVGAN needs at least one upsample stage".into(),
            ));
        }
        if self.stage_out_channels(self.num_upsamples() - 1) < 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: upsample_initial_channel {} cannot halve {} times",
                self.upsample_initial_channel,
                self.num_upsamples()
            )));
        }
        for (i, (&k, &u)) in self
            .upsample_kernel_sizes
            .iter()
            .zip(self.upsample_rates.iter())
            .enumerate()
        {
            if k < u || !(k - u).is_multiple_of(2) {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio vae: stage {i} kernel {k} and rate {u} give a fractional \
                     transposed-conv padding"
                )));
            }
        }
        Ok(())
    }
}

/// The audio VAE's decode-path configuration.
///
/// Dimension-parametric so the same code runs the shipped 1024-wide decoder and the tiny committed
/// parity fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3AudioVaeConfig {
    /// Output sample rate (`metadata.json` `kwargs.sample_rate`; `config.yaml` `model_config.sr`).
    pub sample_rate: u32,
    /// Delivered channel count (`config.json` `output_channel`).
    pub output_channels: u16,
    /// Latent width (`config.yaml` `model_config.vae_latent_channels`).
    pub latent_channels: usize,
    /// `metadata.json` `kwargs.encoder_dim`; also what the constructor's `latent_dim` derives from.
    pub encoder_dim: usize,
    /// `metadata.json` `kwargs.encoder_rates` — the encode half is not ported, but its **length**
    /// sets `latent_dim`.
    pub encoder_rates: Vec<usize>,
    /// `metadata.json` `kwargs.decoder_rates`. Stored by the reference but **not** used to build
    /// the decoder; kept so the `sample_rate` branch can be checked against it.
    pub decoder_rates: Vec<usize>,
    /// `config.yaml` `model_config.decoder_dim` — BigVGAN's `upsample_initial_channel`.
    pub decoder_dim: usize,
    /// `metadata.json` `kwargs.attn_proj`. Encode-half only.
    pub attn_proj: bool,
    /// `metadata.json` `kwargs.decoder_type`; only `bigvgan` exists.
    pub decoder_type: String,
    /// Per-channel latent de-normalization mean (`config.json` `latents_mean`).
    pub latents_mean: Vec<f32>,
    /// Per-channel latent de-normalization std (`config.json` `latents_std`).
    pub latents_std: Vec<f32>,
    /// The `sample_rate`-selected BigVGAN block.
    pub bigvgan: BigVganConfig,
}

impl Default for MiniMaxH3AudioVaeConfig {
    /// The shipped `MiniMaxAI/MiniMax-H3` audio VAE.
    ///
    /// `tests/real_weights.rs` asserts [`Self::from_source_files`] on the published documents
    /// reproduces this exactly, so it is a pin rather than a second source of truth.
    fn default() -> Self {
        let encoder_rates = vec![2, 4, 4, 5, 5];
        let latent_dim = DEFAULT_ENCODER_DIM << encoder_rates.len();
        Self {
            sample_rate: AUDIO_SAMPLE_RATE,
            output_channels: AUDIO_OUTPUT_CHANNELS,
            latent_channels: AUDIO_LATENT_CHANNELS,
            encoder_dim: DEFAULT_ENCODER_DIM,
            encoder_rates,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_dim: 1024,
            attn_proj: true,
            decoder_type: "bigvgan".into(),
            latents_mean: AUDIO_LATENTS_MEAN.to_vec(),
            latents_std: AUDIO_LATENTS_STD.to_vec(),
            bigvgan: BigVganConfig::for_sample_rate(AUDIO_SAMPLE_RATE, latent_dim, 1024)
                .expect("32000 is a supported sample rate"),
        }
    }
}

impl MiniMaxH3AudioVaeConfig {
    /// Build from the three documents `MiniMaxH3AudioVAE.from_pretrained` reads.
    ///
    /// Every constructor argument is taken from the file the reference takes it from —
    /// `config.yaml` for `decoder_dim` / `vae_latent_channels`, `metadata.json` for the rest,
    /// `config.json` for `output_channel` and the latent statistics. Nothing here is hardcoded
    /// except the `sample_rate`-selected BigVGAN block, which the reference itself hardcodes.
    pub fn from_source_files(
        config_json: &str,
        config_yaml: &str,
        metadata_json: &str,
    ) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 audio vae config.json: {e}")))?;
        let metadata: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 audio vae metadata.json: {e}")))?;
        let kwargs = metadata
            .get("metadata")
            .and_then(|m| m.get("kwargs"))
            .ok_or_else(|| {
                CandleError::Msg("minimax-h3 audio vae metadata.json: no metadata.kwargs".into())
            })?;
        let model_config = parse_model_config_yaml(config_yaml)?;

        // `config.json` must actually point at the two source documents, or these are the wrong
        // files: the repackaged root config carries neither key and would otherwise be accepted.
        for key in [
            "source_config_path",
            "source_metadata_path",
            "source_safetensors_path",
        ] {
            if config
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio vae config.json: missing `{key}`; this looks like the \
                     diffusers-repackaged root config, which the reference cannot load — pass \
                     `FL2VA/audio_vae/config.json` (or use `cross_check_diffusers_json`)"
                )));
            }
        }

        let sample_rate = u32::try_from(int_of(kwargs, "sample_rate", "metadata.kwargs")?)
            .map_err(|_| CandleError::Msg("minimax-h3 audio vae: negative sample_rate".into()))?;
        let sr_yaml = yaml_int(&model_config, "sr")?;
        if sr_yaml != i64::from(sample_rate) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: config.yaml sr {sr_yaml} disagrees with metadata.json \
                 sample_rate {sample_rate}"
            )));
        }

        let decoder_dim =
            usize::try_from(yaml_int(&model_config, "decoder_dim")?).map_err(|_| {
                CandleError::Msg("minimax-h3 audio vae: decoder_dim out of range".into())
            })?;
        let latent_channels = usize::try_from(yaml_int(&model_config, "vae_latent_channels")?)
            .map_err(|_| {
                CandleError::Msg("minimax-h3 audio vae: vae_latent_channels out of range".into())
            })?;

        let encoder_rates = usize_vec(kwargs, "encoder_rates", "metadata.kwargs")?;
        let decoder_rates = usize_vec(kwargs, "decoder_rates", "metadata.kwargs")?;
        let decoder_type = kwargs
            .get("decoder_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                CandleError::Msg("minimax-h3 audio vae metadata.kwargs: no decoder_type".into())
            })?
            .to_string();
        if decoder_type != "bigvgan" {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: decoder_type `{decoder_type}` is not implemented (the \
                 reference defines only `bigvgan`)"
            )));
        }
        let attn_proj = kwargs
            .get("attn_proj")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                CandleError::Msg("minimax-h3 audio vae metadata.kwargs: no attn_proj".into())
            })?;

        // `from_pretrained` does not pass `encoder_dim` or `latent_dim`, so both come from the
        // constructor's defaults/derivation. `metadata.json` states them; treat that as a check.
        let encoder_dim = DEFAULT_ENCODER_DIM;
        if let Ok(declared) = int_of(kwargs, "encoder_dim", "metadata.kwargs") {
            if declared != encoder_dim as i64 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio vae: metadata.json declares encoder_dim {declared}, but \
                     `from_pretrained` never forwards it — the reference would build with \
                     {encoder_dim}"
                )));
            }
        }
        let latent_dim = derive_latent_dim(encoder_dim, encoder_rates.len())?;
        if let Ok(declared) = int_of(kwargs, "latent_dim", "metadata.kwargs") {
            if declared != latent_dim as i64 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio vae: metadata.json declares latent_dim {declared}, but \
                     encoder_dim {encoder_dim} · 2^{} derives {latent_dim}",
                    encoder_rates.len()
                )));
            }
        }

        // `decoder_dim` and `vae_latent_channels` are read from `config.yaml` by the reference and
        // ALSO stated in `metadata.json`. A disagreement means the two documents describe different
        // models; the reference would silently follow the YAML.
        for (key, from_yaml) in [
            ("decoder_dim", decoder_dim as i64),
            ("vae_latent_channels", latent_channels as i64),
        ] {
            if let Ok(declared) = int_of(kwargs, key, "metadata.kwargs") {
                if declared != from_yaml {
                    return Err(CandleError::Msg(format!(
                        "minimax-h3 audio vae: metadata.json declares {key} {declared} but \
                         config.yaml gives {from_yaml}, and the reference reads the YAML"
                    )));
                }
            }
        }

        let output_channels = u16::try_from(int_of(&config, "output_channel", "config.json")?)
            .map_err(|_| {
                CandleError::Msg("minimax-h3 audio vae: output_channel out of range".into())
            })?;
        let latents_mean = f32_vec(&config, "latents_mean", "config.json")?;
        let latents_std = f32_vec(&config, "latents_std", "config.json")?;

        let bigvgan = BigVganConfig::for_sample_rate(sample_rate, latent_dim, decoder_dim)?;
        // The reference stores `decoder_rates` but builds the decoder from the `sample_rate`
        // branch. They agree in the published checkpoint; a divergence would decode at the wrong
        // rate with no other symptom, so refuse it.
        if bigvgan.upsample_rates != decoder_rates {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: metadata.json decoder_rates {decoder_rates:?} disagree \
                 with the sample_rate-{sample_rate} BigVGAN upsample_rates {:?}",
                bigvgan.upsample_rates
            )));
        }

        let cfg = Self {
            sample_rate,
            output_channels,
            latent_channels,
            encoder_dim,
            encoder_rates,
            decoder_rates,
            decoder_dim,
            attn_proj,
            decoder_type,
            latents_mean,
            latents_std,
            bigvgan,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the diffusers-repackaged root `audio_vae/config.json` against this config.
    ///
    /// The root form declares the architecture directly (`decoder_rates`, `decoder_kernel_sizes`,
    /// `resblock_*`, `latent_dim`, `sampling_rate`) but carries **no** `output_channel` and none of
    /// the `snake_logscale` / `use_tanh_at_final` / `use_bias_at_final` knobs, so it cannot replace
    /// the source triple. What it *does* declare must agree.
    pub fn cross_check_diffusers_json(&self, config_json: &str) -> Result<()> {
        let root: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 audio vae root config.json: {e}")))?;
        let ctx = "root config.json";
        let mismatch = |field: &str, got: String, want: String| {
            CandleError::Msg(format!(
                "minimax-h3 audio vae: {ctx} {field} = {got}, source configs give {want}"
            ))
        };

        let sampling_rate = int_of(&root, "sampling_rate", ctx)?;
        if sampling_rate != i64::from(self.sample_rate) {
            return Err(mismatch(
                "sampling_rate",
                sampling_rate.to_string(),
                self.sample_rate.to_string(),
            ));
        }
        let checks: [(&str, i64, i64); 5] = [
            (
                "latent_channels",
                int_of(&root, "latent_channels", ctx)?,
                self.latent_channels as i64,
            ),
            // **`num_attention_heads` — the one key here that binds a CONSTANT rather than a
            // field.** The original `DacAudioVAE` hardcodes `num_heads=8` and it appears in none of
            // the source triple, so `MiniMaxH3AudioVaeConfig` has no field for it and
            // `crate::audio_vae_encoder::ATTN_PROJ_HEADS` carries it. The diffusers repackaging
            // DOES publish it, which makes this the only place the constant can be checked against
            // a document instead of against itself. Changing the head count changes `head_dim`,
            // hence the attention scale and the adaptive pool's window layout, while every tensor
            // shape stays identical — a runnable model with a wrong soundtrack conditioning, which
            // no shape assertion anywhere can catch.
            (
                "num_attention_heads",
                int_of(&root, "num_attention_heads", ctx)?,
                crate::audio_vae_encoder::ATTN_PROJ_HEADS as i64,
            ),
            (
                "latent_dim",
                int_of(&root, "latent_dim", ctx)?,
                self.bigvgan.num_mels as i64,
            ),
            (
                "decoder_dim",
                int_of(&root, "decoder_dim", ctx)?,
                self.decoder_dim as i64,
            ),
            (
                "encoder_dim",
                int_of(&root, "encoder_dim", ctx)?,
                self.encoder_dim as i64,
            ),
        ];
        for (field, got, want) in checks {
            if got != want {
                return Err(mismatch(field, got.to_string(), want.to_string()));
            }
        }

        let vectors: [(&str, Vec<usize>, &Vec<usize>); 4] = [
            (
                "decoder_rates",
                usize_vec(&root, "decoder_rates", ctx)?,
                &self.decoder_rates,
            ),
            (
                "decoder_kernel_sizes",
                usize_vec(&root, "decoder_kernel_sizes", ctx)?,
                &self.bigvgan.upsample_kernel_sizes,
            ),
            (
                "encoder_rates",
                usize_vec(&root, "encoder_rates", ctx)?,
                &self.encoder_rates,
            ),
            (
                "resblock_kernel_sizes",
                usize_vec(&root, "resblock_kernel_sizes", ctx)?,
                &self.bigvgan.resblock_kernel_sizes,
            ),
        ];
        for (field, got, want) in vectors {
            if &got != want {
                return Err(mismatch(field, format!("{got:?}"), format!("{want:?}")));
            }
        }

        let dilations = root
            .get("resblock_dilation_sizes")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 audio vae {ctx}: no resblock_dilation_sizes"
                ))
            })?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| {
                        CandleError::Msg(format!(
                            "minimax-h3 audio vae {ctx}: resblock_dilation_sizes is not nested"
                        ))
                    })?
                    .iter()
                    .map(|v| {
                        v.as_u64().map(|v| v as usize).ok_or_else(|| {
                            CandleError::Msg(format!(
                                "minimax-h3 audio vae {ctx}: non-integer dilation"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()
            })
            .collect::<Result<Vec<Vec<usize>>>>()?;
        if dilations != self.bigvgan.resblock_dilation_sizes {
            return Err(mismatch(
                "resblock_dilation_sizes",
                format!("{dilations:?}"),
                format!("{:?}", self.bigvgan.resblock_dilation_sizes),
            ));
        }

        for (field, want) in [
            ("latents_mean", &self.latents_mean),
            ("latents_std", &self.latents_std),
        ] {
            let got = f32_vec(&root, field, ctx)?;
            if &got != want {
                return Err(mismatch(
                    field,
                    format!("{} entries", got.len()),
                    format!("{} entries", want.len()),
                ));
            }
        }
        Ok(())
    }

    /// Samples emitted per latent token — `∏ upsample_rates`.
    pub fn hop_length(&self) -> usize {
        self.bigvgan.hop_length()
    }

    /// Latent tokens per second: `sample_rate / hop_length`.
    pub fn token_rate_hz(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length() as f32
    }

    fn validate(&self) -> Result<()> {
        self.bigvgan.validate()?;
        if self.latents_mean.len() != self.latent_channels
            || self.latents_std.len() != self.latent_channels
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae: latents_mean/std have {}/{} entries for {} latent channels",
                self.latents_mean.len(),
                self.latents_std.len(),
                self.latent_channels
            )));
        }
        if self.latents_std.iter().any(|s| *s <= 0.0) {
            return Err(CandleError::Msg(
                "minimax-h3 audio vae: latents_std has a non-positive entry".into(),
            ));
        }
        if self.output_channels == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 audio vae: output_channel is 0".into(),
            ));
        }
        Ok(())
    }
}

/// `encoder_dim · 2^len(encoder_rates)` — the constructor's `latent_dim` derivation.
fn derive_latent_dim(encoder_dim: usize, encoder_rate_count: usize) -> Result<usize> {
    if encoder_rate_count >= 24 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 audio vae: {encoder_rate_count} encoder_rates overflow latent_dim"
        )));
    }
    Ok(encoder_dim << encoder_rate_count)
}

fn int_of(v: &serde_json::Value, key: &str, ctx: &str) -> Result<i64> {
    v.get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| CandleError::Msg(format!("minimax-h3 audio vae {ctx}: no integer `{key}`")))
}

fn usize_vec(v: &serde_json::Value, key: &str, ctx: &str) -> Result<Vec<usize>> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CandleError::Msg(format!("minimax-h3 audio vae {ctx}: no array `{key}`")))?
        .iter()
        .map(|e| {
            e.as_u64().map(|e| e as usize).ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 audio vae {ctx}: `{key}` has a non-positive-integer"
                ))
            })
        })
        .collect()
}

fn f32_vec(v: &serde_json::Value, key: &str, ctx: &str) -> Result<Vec<f32>> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CandleError::Msg(format!("minimax-h3 audio vae {ctx}: no array `{key}`")))?
        .iter()
        .map(|e| {
            e.as_f64().map(|e| e as f32).ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 audio vae {ctx}: `{key}` has a non-number"
                ))
            })
        })
        .collect()
}

/// Parse the `model_config:` block of the audio VAE's `config.yaml`.
///
/// That file is four `key: integer` pairs under one top-level mapping and nothing else, so this is
/// a purpose-built reader rather than a YAML dependency. Anything outside that shape — nesting,
/// sequences, quoting, anchors, multiple documents — is rejected rather than silently mis-read.
fn parse_model_config_yaml(text: &str) -> Result<Vec<(String, i64)>> {
    let mut inside = false;
    let mut out: Vec<(String, i64)> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("");
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let body = line.trim_end().trim_start();
        if indent == 0 {
            if body == "model_config:" {
                inside = true;
                continue;
            }
            // Another top-level key ends the block we care about.
            inside = false;
            continue;
        }
        if !inside {
            continue;
        }
        let (key, value) = body.split_once(':').ok_or_else(|| {
            CandleError::Msg(format!(
                "minimax-h3 audio vae config.yaml line {}: expected `key: value`, got `{body}`",
                lineno + 1
            ))
        })?;
        let value = value.trim();
        if value.is_empty() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio vae config.yaml line {}: `{key}` has a nested value; this \
                 reader supports only scalar integers",
                lineno + 1
            )));
        }
        let parsed = value.parse::<i64>().map_err(|_| {
            CandleError::Msg(format!(
                "minimax-h3 audio vae config.yaml line {}: `{key}: {value}` is not an integer",
                lineno + 1
            ))
        })?;
        out.push((key.trim().to_string(), parsed));
    }
    if out.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 audio vae config.yaml: no `model_config:` entries".into(),
        ));
    }
    Ok(out)
}

fn yaml_int(entries: &[(String, i64)], key: &str) -> Result<i64> {
    entries
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "minimax-h3 audio vae config.yaml: no `model_config.{key}`"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `FL2VA/audio_vae/config.yaml`.
    const CONFIG_YAML: &str = "model_config:\n  sr: 32000\n  decoder_dim: 1024\n  \
                               audio_channel: 1\n  vae_latent_channels: 32\n";

    /// Verbatim `FL2VA/audio_vae/metadata.json`.
    const METADATA_JSON: &str = r#"{
      "metadata": { "kwargs": {
        "attn_proj": true, "decoder_dim": 1024,
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2], "decoder_type": "bigvgan",
        "encoder_dim": 64, "encoder_rates": [2, 4, 4, 5, 5],
        "latent_dim": 2048, "sample_rate": 32000, "vae_latent_channels": 32
      } } }"#;

    fn config_json() -> String {
        let mean: Vec<String> = AUDIO_LATENTS_MEAN.iter().map(|v| v.to_string()).collect();
        let std: Vec<String> = AUDIO_LATENTS_STD.iter().map(|v| v.to_string()).collect();
        format!(
            r#"{{"_class_name":"MiniMaxH3AudioVAE","output_channel":2,"sample_rate":32000,
               "source_config_path":"config.yaml","source_metadata_path":"metadata.json",
               "source_safetensors_path":"model.safetensors","latent_channels":32,
               "latents_mean":[{}],"latents_std":[{}]}}"#,
            mean.join(","),
            std.join(",")
        )
    }

    fn shipped() -> MiniMaxH3AudioVaeConfig {
        MiniMaxH3AudioVaeConfig::from_source_files(&config_json(), CONFIG_YAML, METADATA_JSON)
            .unwrap()
    }

    /// The published documents must reproduce [`MiniMaxH3AudioVaeConfig::default`] exactly, so the
    /// constant is a pin on the parse rather than a second source of truth.
    #[test]
    fn source_files_reproduce_the_shipped_config() {
        assert_eq!(shipped(), MiniMaxH3AudioVaeConfig::default());
    }

    /// The four decode-relevant numbers, and the two derivations no config states outright.
    #[test]
    fn shipped_geometry_is_the_published_one() {
        let cfg = shipped();
        assert_eq!(cfg.sample_rate, 32_000);
        assert_eq!(cfg.latent_channels, 32);
        assert_eq!(cfg.output_channels, 2);
        assert_eq!(cfg.decoder_dim, 1024);
        // latent_dim = 64 · 2^5, from the constructor default and the five encoder_rates.
        assert_eq!(cfg.bigvgan.num_mels, 2048);
        assert_eq!(cfg.hop_length(), 800);
        assert_eq!(cfg.token_rate_hz() as u32, AUDIO_TOKEN_RATE_HZ);
        // 7 stages, 21 AMP blocks, widths 512 … 8.
        assert_eq!(cfg.bigvgan.num_upsamples(), 7);
        assert_eq!(cfg.bigvgan.num_upsamples() * cfg.bigvgan.num_kernels(), 21);
        assert_eq!(cfg.bigvgan.stage_in_channels(0), 1024);
        assert_eq!(cfg.bigvgan.stage_out_channels(0), 512);
        assert_eq!(cfg.bigvgan.stage_out_channels(6), 8);
        // The three knobs that appear in no config file and leave no tensor behind.
        assert!(cfg.bigvgan.snake_logscale);
        assert!(!cfg.bigvgan.use_tanh_at_final);
        assert!(!cfg.bigvgan.use_bias_at_final);
    }

    /// The repackaged root config declares a subset; it must agree, and the fields it OMITS are
    /// exactly the ones that make it insufficient on its own.
    #[test]
    fn diffusers_root_config_cross_checks() {
        let root = r#"{"_class_name":"AutoencoderKLMiniMaxH3Audio","decoder_dim":1024,
          "decoder_kernel_sizes":[9,9,4,4,4,4,4],"decoder_rates":[5,5,2,2,2,2,2],
          "encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_channels":32,"latent_dim":2048,
          "num_attention_heads":8,"resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]],
          "resblock_kernel_sizes":[3,7,11],"sampling_rate":32000,"latents_mean":[LM],
          "latents_std":[LS]}"#
            .replace(
                "LM",
                &AUDIO_LATENTS_MEAN
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .replace(
                "LS",
                &AUDIO_LATENTS_STD
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        shipped().cross_check_diffusers_json(&root).unwrap();

        // It carries no `output_channel`, so it cannot stand in for the source triple.
        let parsed: serde_json::Value = serde_json::from_str(&root).unwrap();
        assert!(parsed.get("output_channel").is_none());
        assert!(
            MiniMaxH3AudioVaeConfig::from_source_files(&root, CONFIG_YAML, METADATA_JSON).is_err()
        );

        // A drifted architecture is rejected rather than absorbed.
        let bad = root.replace("[9,9,4,4,4,4,4]", "[9,9,4,4,4,4,8]");
        assert!(shipped().cross_check_diffusers_json(&bad).is_err());

        // **`num_attention_heads` is genuinely compared, and names itself when it disagrees.**
        // The doc on `ATTN_PROJ_HEADS` claims this check binds the constant to a published
        // document; a check that is never shown to REJECT is not a binding. Both directions, so a
        // comparison hardcoded to 8 on either side fails one of them.
        for wrong in [4, 16] {
            let drifted = root.replace(
                "\"num_attention_heads\":8",
                &format!("\"num_attention_heads\":{wrong}"),
            );
            let e = shipped()
                .cross_check_diffusers_json(&drifted)
                .expect_err("a drifted head count must be refused")
                .to_string();
            assert!(
                e.contains("num_attention_heads") && e.contains(&wrong.to_string()),
                "{wrong}: {e}"
            );
        }
        // …and a config that OMITS it is refused rather than defaulted to the constant.
        let absent = root.replace("\"num_attention_heads\":8,", "");
        let e = shipped()
            .cross_check_diffusers_json(&absent)
            .expect_err("a missing head count must be refused")
            .to_string();
        assert!(e.contains("num_attention_heads"), "{e}");
    }

    /// A `sample_rate` the reference has no BigVGAN block for is an error, not a guess.
    #[test]
    fn unsupported_sample_rate_is_rejected() {
        assert!(BigVganConfig::for_sample_rate(44_100, 2048, 1024).is_err());
        let sixteen = BigVganConfig::for_sample_rate(16_000, 2048, 1024).unwrap();
        assert_eq!(sixteen.num_upsamples(), 6);
        assert_eq!(sixteen.hop_length(), 400);
    }

    /// `decoder_rates` is inert for the decoder, so a checkpoint whose metadata disagreed with the
    /// `sample_rate` branch would decode at the wrong rate silently. Refuse it instead.
    #[test]
    fn decoder_rates_must_match_the_sample_rate_branch() {
        let drifted = METADATA_JSON.replace("[5, 5, 2, 2, 2, 2, 2]", "[5, 5, 2, 2, 2, 2, 4]");
        let err = MiniMaxH3AudioVaeConfig::from_source_files(&config_json(), CONFIG_YAML, &drifted)
            .unwrap_err()
            .to_string();
        assert!(err.contains("decoder_rates"), "{err}");
    }

    /// `latent_dim` and `encoder_dim` are derived, not read — but metadata states them, so a
    /// disagreement means the reference would build a different model than the file describes.
    #[test]
    fn declared_latent_dim_must_match_the_derivation() {
        let drifted = METADATA_JSON.replace("\"latent_dim\": 2048", "\"latent_dim\": 1024");
        let err = MiniMaxH3AudioVaeConfig::from_source_files(&config_json(), CONFIG_YAML, &drifted)
            .unwrap_err()
            .to_string();
        assert!(err.contains("latent_dim"), "{err}");

        let drifted = METADATA_JSON.replace("\"encoder_dim\": 64", "\"encoder_dim\": 32");
        let err = MiniMaxH3AudioVaeConfig::from_source_files(&config_json(), CONFIG_YAML, &drifted)
            .unwrap_err()
            .to_string();
        assert!(err.contains("encoder_dim"), "{err}");

        assert_eq!(derive_latent_dim(64, 5).unwrap(), 2048);
        assert!(derive_latent_dim(64, 40).is_err());
    }

    /// The two documents that both state `decoder_dim` / `vae_latent_channels` must agree. The
    /// reference reads the YAML, so a metadata-only change would otherwise be silently ignored.
    #[test]
    fn metadata_must_agree_with_the_yaml_it_duplicates() {
        for (from, to) in [
            ("\"decoder_dim\": 1024", "\"decoder_dim\": 512"),
            ("\"vae_latent_channels\": 32", "\"vae_latent_channels\": 16"),
        ] {
            let drifted = METADATA_JSON.replace(from, to);
            assert_ne!(
                drifted, METADATA_JSON,
                "the test's search string went stale"
            );
            let err =
                MiniMaxH3AudioVaeConfig::from_source_files(&config_json(), CONFIG_YAML, &drifted)
                    .unwrap_err()
                    .to_string();
            assert!(err.contains("config.yaml gives"), "{err}");
        }
    }

    /// The YAML reader accepts the shipped shape and refuses anything it cannot honestly parse.
    #[test]
    fn yaml_reader_is_narrow_and_loud() {
        let ok = parse_model_config_yaml(CONFIG_YAML).unwrap();
        assert_eq!(yaml_int(&ok, "sr").unwrap(), 32_000);
        assert_eq!(yaml_int(&ok, "audio_channel").unwrap(), 1);
        assert!(yaml_int(&ok, "missing").is_err());

        // Comments and blank lines are fine.
        assert_eq!(
            parse_model_config_yaml("# top\nmodel_config:\n\n  sr: 16000  # rate\n").unwrap(),
            vec![("sr".to_string(), 16_000)]
        );
        // Nested values, non-integers and an absent block are not silently coerced.
        assert!(parse_model_config_yaml("model_config:\n  nested:\n    a: 1\n").is_err());
        assert!(parse_model_config_yaml("model_config:\n  sr: thirty\n").is_err());
        assert!(parse_model_config_yaml("other:\n  sr: 1\n").is_err());
        // A second top-level key ends the block rather than leaking into it.
        assert_eq!(
            parse_model_config_yaml("model_config:\n  sr: 32000\nother_config:\n  sr: 1\n")
                .unwrap(),
            vec![("sr".to_string(), 32_000)]
        );
    }

    /// A config that cannot describe a real decoder is rejected at construction.
    #[test]
    fn validation_rejects_impossible_geometry() {
        let mut cfg = MiniMaxH3AudioVaeConfig::default();
        // 64 cannot halve 7 times and stay >= 1.
        cfg.bigvgan.upsample_initial_channel = 64;
        assert!(cfg.validate().is_err());

        let mut cfg = MiniMaxH3AudioVaeConfig::default();
        cfg.latents_std[3] = 0.0;
        assert!(cfg.validate().is_err());

        let mut cfg = MiniMaxH3AudioVaeConfig::default();
        cfg.latents_mean.pop();
        assert!(cfg.validate().is_err());

        // An odd (kernel - rate) has no integer transposed-conv padding.
        let mut cfg = MiniMaxH3AudioVaeConfig::default();
        cfg.bigvgan.upsample_kernel_sizes[0] = 8;
        assert!(cfg.validate().is_err());
    }
}
