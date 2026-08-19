//! LTX-2.3 (distilled 22B) model configuration — hardcoded constants for the shipped dense BF16
//! checkpoint (`ltx-2.3-22b-distilled.safetensors`). The mlx provider reads `embedded_config.json`
//! to support the quantized split checkpoints; we load the single dense file and pin the LTX-2.3
//! values directly (they are fixed for this model family).
//!
//! This is **video+audio**: the video-stack DiT, the Gemma-3-12B text encoder, the video connector,
//! and the video VAE decoder, plus the synchronized-audio stack (audio text head + connector, the
//! dual-modal AV DiT, the audio VAE decoder, and the vocoder — sc-5495) are all consumed. The 2-stage
//! latent upsampler, prompt-enhance, and fp8/on-the-fly quant are deferred to follow-up stories.
//! I2V/keyframes/IC-LoRA clips and inference LoRA/PEFT-LoKr are wired by the package pipeline.

/// Registry id (the distilled 22B text-to-video model).
pub const MODEL_ID: &str = "ltx_2_3_distilled";
/// Registry id for LoRA training.
///
/// Training targets the base LTX-2.3 family recipe, while generation remains the separately pinned
/// distilled engine [`MODEL_ID`]. Keep these ids distinct so callers can select the trainer without
/// changing the generator route.
pub const TRAINER_ID: &str = "ltx_2_3";

/// The `model_version` this engine's checkpoints declare, used to resolve generation params
/// (sc-18759 — see [`crate::params`]). This crate loads only `ltx_2_3` checkpoints today (the
/// `ltx_2_5_distilled` engine descriptor is sc-18778), so the literal is correct as written, not a
/// placeholder: once split-checkpoint loading (sc-18757) threads a loaded checkpoint's declared
/// `model_version` onto [`LtxGenerator`](crate::LtxGenerator), swap this constant for that field.
pub const CHECKPOINT_MODEL_VERSION: &str = "2.3.0";

// --- VAE compression factors + sampling defaults (mlx-gen-ltx positions.rs) ----------------------
/// Temporal VAE compression: pixel frames → latent frames is `(F-1)/8 + 1`.
pub const TEMPORAL_SCALE: usize = 8;
/// Spatial VAE compression (per axis): pixel H/W → latent H/W is `/32`.
pub const SPATIAL_SCALE: usize = 32;
/// Latent voxel channels (the DiT in/out + VAE latent channels).
pub const LATENT_CHANNELS: usize = 128;

/// Default output framerate.
pub const DEFAULT_FPS: u32 = 24;
/// Default pixel frame count — `% TEMPORAL_SCALE == 1` (49 → 7 latent frames). Kept modest for the
/// first-slice verification render; the request may override.
pub const DEFAULT_FRAMES: u32 = 49;
/// Default pixel width/height (multiples of `SPATIAL_SCALE`).
pub const DEFAULT_WIDTH: u32 = 704;
pub const DEFAULT_HEIGHT: u32 = 480;

/// Gemma prompt token budget (left-padded). The connector replaces the left-pad slots with its
/// learnable registers, so this caps the real-token context fed to the DiT cross-attention.
pub const TEXT_MAX_LENGTH: usize = 256;

/// Upper bound on a render's video **latent token count** (`t_lat · h_lat · w_lat`). This is the
/// AvDiT denoise-loop sequence length and the real memory driver — the video self-attn scores
/// `[b, h, s, s]` working set plus the per-token q/k/v activations across the 48 video-DiT layers.
/// `validate` bounded only `frames % TEMPORAL_SCALE == 1` with no upper limit, so a `frames: 2001`
/// request at 1280² (≈400k latent tokens) passed every guard except the VAE's and OOM'd mid-denoise
/// in the 22B loop instead of failing catchably up front (F-131, sc-11234). Sized against the target
/// GPU envelope: at 131072 tokens the per-layer f32 q/k/v working set is ≈6.4 GB — comfortably
/// generous for real clips (704×480 → ~400 latent frames; 1280² → ~80 latent frames ≈ 640 pixel
/// frames) while rejecting pathological requests. Overridable per-GPU via [`max_latent_tokens`].
pub const MAX_LATENT_TOKENS: usize = 131_072;

/// Resolve the latent-token cap: the `LTX_MAX_LATENT_TOKENS` env override (a positive integer) when
/// set, else [`MAX_LATENT_TOKENS`]. Mirrors the seedvr2 `SEEDVR2_BUDGET_GIB` per-GPU tuning knob so
/// a larger-VRAM worker can lift the ceiling without a recompile.
pub fn max_latent_tokens() -> usize {
    if let Ok(raw) = std::env::var("LTX_MAX_LATENT_TOKENS") {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    MAX_LATENT_TOKENS
}

/// Distilled single-stage rectified-flow sigma schedule (`DEFAULT_STAGE_1_SIGMAS`, 8 denoise steps:
/// σ goes 1.0 → 0.0, a complete generation). The 2-stage refinement (upsample + re-noise + the
/// `STAGE2` sigmas) is deferred to a follow-up; stage-1 alone at the target resolution is a full,
/// coherent render. The distilled model bakes guidance in → **no CFG**.
pub const STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875, 0.0,
];

/// The number of denoise steps the distilled [`STAGE1_SIGMAS`] schedule performs (`len − 1`). This is
/// the ONLY step count the distilled model supports — the σ waypoints are baked into training, so an
/// arbitrary `req.steps` cannot be honored by resampling without going out-of-distribution. `render`
/// runs this many steps unconditionally; [`crate::descriptor`]'s `validate` rejects any other explicit
/// `req.steps` rather than silently ignoring it (sc-9027 / F-043).
pub const NATIVE_STEPS: u32 = STAGE1_SIGMAS.len() as u32 - 1;

/// The LTX-2.3 video DiT (`AVTransformer3DModel`, video stack) dimensions.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub rope_max_pos: [i32; 3],
    pub timestep_scale_multiplier: f64,
    /// sc-18758 — the measured LTX-2.3→2.5 delta's video-stream half: `true` for 2.3 (both FFN
    /// Linears carry a bias, the reference default when `ff_bias` is absent from the config), `false`
    /// for 2.5 (the checkpoint carries no `ff.proj_{in,out}.bias`).
    pub ff_bias: bool,
    /// sc-18758 — whether the video stream carries a learned `(1, inner_dim)` absolute-position
    /// marker added to single-pixel generated-keyframe tokens (the DFR keyframe-slot path). `false`
    /// for 2.3 (no such tensor); `true` for 2.5.
    pub use_keyframes_abs_pos_embedding: bool,
}

impl TransformerConfig {
    pub fn ltx_2_3() -> Self {
        Self {
            num_layers: 48,
            num_heads: 32,
            head_dim: 128,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_pos: [20, 2048, 2048],
            timestep_scale_multiplier: 1000.0,
            ff_bias: true,
            use_keyframes_abs_pos_embedding: false,
        }
    }

    /// The LTX-2.5 video DiT: identical dims to 2.3 (per the epic orientation, the *entire* measured
    /// config delta is `ff_bias`/`use_keyframes_abs_pos_embedding`) with `ff_bias:false` and
    /// `use_keyframes_abs_pos_embedding:true`.
    pub fn ltx_2_5() -> Self {
        Self {
            ff_bias: false,
            use_keyframes_abs_pos_embedding: true,
            ..Self::ltx_2_3()
        }
    }

    /// Inner dim `heads × head_dim` = 4096.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// The dual-modal `AVTransformer3DModel` dims (sc-5495): the video stack ([`TransformerConfig`]) plus
/// the audio stack + the cross-modal RoPE geometry. The audio stack mirrors the video block at the
/// audio inner dim (heads 32 × head_dim 64 = 2048); the cross-modal attns + their 1-D time RoPE run
/// at `cross_inner` (2048). Fixed for the shipped LTX-2.3 checkpoint.
#[derive(Clone, Debug)]
pub struct AvConfig {
    pub video: TransformerConfig,
    pub audio_heads: usize,
    pub audio_head_dim: usize,
    /// 1-D audio-self RoPE max position (`audio_positional_embedding_max_pos = [20]`).
    pub audio_max_pos: i32,
    /// Cross-modal RoPE inner dim (`audio_cross_attention_dim`, 2048).
    pub cross_inner: usize,
    /// Cross-modal (time-axis) RoPE max position (`cross_pe_max_pos`, 20).
    pub cross_max_pos: i32,
    /// sc-18758 — `audio_ff_bias`, the audio-stream analog of `video.ff_bias`.
    pub audio_ff_bias: bool,
}

impl AvConfig {
    pub fn ltx_2_3() -> Self {
        Self {
            video: TransformerConfig::ltx_2_3(),
            audio_heads: 32,
            audio_head_dim: 64,
            audio_max_pos: 20,
            cross_inner: 2048,
            cross_max_pos: 20,
            audio_ff_bias: true,
        }
    }

    /// The LTX-2.5 dual-modal config: [`TransformerConfig::ltx_2_5`]'s video half; `audio_ff_bias`
    /// stays `true` — verified against the real shipped header (both the distilled and dev
    /// `transformer.safetensors`), the 2.5 metadata carries **no** `audio_ff_bias` key at all (only
    /// `ff_bias:false`), so it takes the reference's absent-key default (`True`) same as 2.3, and all
    /// 96 `transformer_blocks.*.audio_ff.net.{0.proj,2}.bias` tensors are present and required. Only
    /// the **video** FFN drops its bias.
    pub fn ltx_2_5() -> Self {
        Self {
            video: TransformerConfig::ltx_2_5(),
            ..Self::ltx_2_3()
        }
    }

    /// Audio inner dim `heads × head_dim` = 2048.
    pub fn audio_inner(&self) -> usize {
        self.audio_heads * self.audio_head_dim
    }
}

/// The 8-layer learnable-register text connector (video stream).
#[derive(Clone, Debug)]
pub struct ConnectorConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_registers: usize,
    pub max_pos: i32,
    pub norm_eps: f64,
    pub rope_theta: f64,
    /// sc-18758 — `connector_ff_bias` (reference `Embeddings1DConnectorConfigurator`/
    /// `AudioEmbeddings1DConnectorConfigurator`). Independent of the DiT's own `ff_bias`; neither the
    /// 2.3 nor the shipped 2.5 checkpoint sets it, so it is `true` on both in practice today — parsed
    /// for fidelity to the reference config surface, not because it is part of the measured delta.
    pub ff_bias: bool,
}

impl ConnectorConfig {
    pub fn ltx_2_3() -> Self {
        Self {
            num_layers: 8,
            num_heads: 32,
            head_dim: 128,
            num_registers: 128,
            max_pos: 4096,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            ff_bias: true,
        }
    }
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// The audio text connector (`audio_embeddings_connector`): 8 layers, heads 32 × head_dim 64 =
    /// 2048, 128 registers, max_pos 4096. Same structure as the video connector at the audio dim.
    pub fn ltx_2_3_audio() -> Self {
        Self {
            num_layers: 8,
            num_heads: 32,
            head_dim: 64,
            num_registers: 128,
            max_pos: 4096,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            ff_bias: true,
        }
    }
}

/// Gemma-3-12B (used as a text encoder — all hidden states extracted).
#[derive(Clone, Debug)]
pub struct GemmaConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_eps: f64,
    /// Global-attention RoPE base (layers where `(i+1) % sliding_window_pattern == 0`).
    pub rope_theta_global: f64,
    /// Local (sliding-window) RoPE base.
    pub rope_theta_local: f64,
    /// Every Nth layer is global attention (1-indexed): `(i+1) % pattern == 0`.
    pub sliding_window_pattern: usize,
    /// Attention scale denominator (query_pre_attn_scalar = head_dim for 12B → scale 256^-0.5).
    pub query_pre_attn_scalar: f64,
    /// Token-embedding vocabulary size (`[vocab, hidden]` table) — Gemma-3's 262144. Only used to size
    /// the packed-detecting `embed_tokens` loader's shape hint (sc-9417).
    pub vocab_size: usize,
}

impl GemmaConfig {
    pub fn gemma_3_12b() -> Self {
        Self {
            num_layers: 48,
            hidden_size: 3840,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            intermediate_size: 15360,
            rms_eps: 1e-6,
            rope_theta_global: 1_000_000.0,
            rope_theta_local: 10_000.0,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: 256.0,
            vocab_size: 262_144,
        }
    }
    pub fn is_global_layer(&self, i: usize) -> bool {
        (i + 1).is_multiple_of(self.sliding_window_pattern)
    }
}

// =================================================================================================
// Synchronized audio (sc-5495) — the LTX-2.3 audio VAE decoder + HiFi-GAN/BigVGAN vocoder + the
// audio-stream dimensions of the dual-modal `AVTransformer3DModel`. These mirror `mlx-gen-ltx`
// (`config.rs` / `positions.rs`), but the values are **hardcoded** to the shipped LTX-2.3 dense
// checkpoint rather than parsed from `embedded_config.json` (the original `Lightricks/LTX-2.3` repo
// ships no such file; the same values live in the safetensors `__metadata__["config"]` blob and are
// fixed for this model family). Channel counts still ride on the weight shapes at load time.
// =================================================================================================

// --- Audio latent geometry (mlx-gen-ltx positions.rs `AUDIO_*`) ----------------------------------
/// Audio VAE internal sample rate (`AUDIO_LATENT_SAMPLE_RATE`).
pub const AUDIO_LATENT_SAMPLE_RATE: i64 = 16000;
/// Mel hop length (`AUDIO_HOP_LENGTH`).
pub const AUDIO_HOP_LENGTH: i64 = 160;
/// Latent temporal downsample factor (`AUDIO_LATENT_DOWNSAMPLE_FACTOR`).
pub const AUDIO_LATENT_DOWNSAMPLE_FACTOR: i64 = 4;
/// Audio latent channels before patchifying (`AUDIO_LATENT_CHANNELS`).
pub const AUDIO_LATENT_CHANNELS: i64 = 8;
/// Audio latent mel bins (`AUDIO_MEL_BINS`) — the latent is `(1, 8, T, 16)`.
pub const AUDIO_MEL_BINS: i64 = 16;
/// `AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR` = 25.
pub const AUDIO_LATENTS_PER_SECOND: f64 = 25.0;

/// Python `round()` (round-half-to-even) — matches `compute_audio_frames`'s `round(...)`.
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Audio latent-frame count for a video duration — port of `compute_audio_frames`
/// (`round(num_video_frames / fps · AUDIO_LATENTS_PER_SECOND)`). Computed in f64 (Python floats).
pub fn compute_audio_frames(num_video_frames: usize, fps: f64) -> usize {
    let duration = num_video_frames as f64 / fps;
    py_round(duration * AUDIO_LATENTS_PER_SECOND).max(0) as usize
}

// --- Audio VAE decoder (`audio_vae.model.params.ddconfig`) ----------------------------------------
/// The LTX-2.3 audio VAE decoder structure (2-D conv autoencoder, causal-on-time, PixelNorm). Fixed
/// for the shipped checkpoint; channels are inferred from the weights at load (see `audio_vae.rs`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioVaeConfig {
    pub ch: i32,
    pub out_ch: i32,
    pub ch_mult: Vec<i32>,
    pub num_res_blocks: i32,
    pub z_channels: i32,
    pub mel_bins: i32,
    /// `mid_block_add_attention` — `false` for the shipped 2.3 (no `mid.attn_1` weights).
    pub mid_block_add_attention: bool,
}

impl AudioVaeConfig {
    /// The shipped LTX-2.3 audio-VAE structure.
    pub fn ltx_2_3() -> Self {
        Self {
            ch: 128,
            out_ch: 2,
            ch_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            z_channels: 8,
            mel_bins: 64,
            mid_block_add_attention: false,
        }
    }

    /// Number of resolution levels (`len(ch_mult)`); the decoder upsamples on levels `1..num`.
    pub fn num_resolutions(&self) -> usize {
        self.ch_mult.len()
    }
}

// --- Vocoder (`vocoder.{vocoder,bwe}`) ------------------------------------------------------------
/// One vocoder generator's config (HiFi-GAN / BigVGAN). Drives the `ConvTranspose1d` upsample
/// strides + the dilated ResBlock/AMPBlock kernel sizes/dilations (channel counts ride on the
/// weights). `is_bigvgan()` selects SnakeBeta+AMPBlock1 vs leaky-ReLU+ResBlock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocoderGenConfig {
    pub upsample_rates: Vec<i32>,
    pub upsample_kernel_sizes: Vec<i32>,
    pub resblock_kernel_sizes: Vec<i32>,
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    pub resblock: String,
    pub activation: String,
    pub use_tanh_at_final: bool,
    pub apply_final_activation: bool,
}

impl VocoderGenConfig {
    /// SnakeBeta + AMPBlock1 (BigVGAN) vs leaky-ReLU + ResBlock (HiFi-GAN).
    pub fn is_bigvgan(&self) -> bool {
        self.activation.eq_ignore_ascii_case("snakebeta")
            || self.resblock.eq_ignore_ascii_case("AMP1")
    }

    /// The shipped LTX-2.3 **core** vocoder (BigVGAN, 6× upsample → 16 kHz).
    pub fn ltx_2_3_core() -> Self {
        Self {
            upsample_rates: vec![5, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![11, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            resblock: "AMP1".into(),
            activation: "snakebeta".into(),
            use_tanh_at_final: false,
            apply_final_activation: true,
        }
    }

    /// The shipped LTX-2.3 **BWE** generator (BigVGAN, 5× upsample, 16 → 48 kHz; no final activation).
    pub fn ltx_2_3_bwe() -> Self {
        Self {
            upsample_rates: vec![6, 5, 2, 2, 2],
            upsample_kernel_sizes: vec![12, 11, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            resblock: "AMP1".into(),
            activation: "snakebeta".into(),
            use_tanh_at_final: false,
            apply_final_activation: false,
        }
    }
}

/// The full vocoder config: the core generator + the bandwidth-extension (BWE) stage. The shipped
/// 2.3 path is BigVGAN core (16 kHz) → BWE → 48 kHz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocoderConfig {
    pub core: VocoderGenConfig,
    pub bwe: Option<VocoderGenConfig>,
    /// Core generator output sample rate (the BWE input rate).
    pub output_sample_rate: i32,
    pub bwe_input_sample_rate: i32,
    pub bwe_output_sample_rate: i32,
    pub bwe_hop_length: i32,
    pub bwe_win_length: i32,
}

impl VocoderConfig {
    /// The shipped LTX-2.3 vocoder (BigVGAN core + BWE, 48 kHz stereo output).
    pub fn ltx_2_3() -> Self {
        Self {
            core: VocoderGenConfig::ltx_2_3_core(),
            bwe: Some(VocoderGenConfig::ltx_2_3_bwe()),
            output_sample_rate: 16000,
            bwe_input_sample_rate: 16000,
            bwe_output_sample_rate: 48000,
            bwe_hop_length: 80,
            bwe_win_length: 512,
        }
    }

    /// The audio-track sample rate: the BWE output when present, else the core output.
    pub fn final_sample_rate(&self) -> i32 {
        if self.bwe.is_some() {
            self.bwe_output_sample_rate
        } else {
            self.output_sample_rate
        }
    }
}

#[cfg(test)]
mod audio_config_tests {
    use super::*;

    #[test]
    fn compute_audio_frames_matches_reference() {
        // round(num_frames / fps · 25). 33f@24fps: 33/24·25 = 34.375 → 34.
        assert_eq!(compute_audio_frames(33, 24.0), 34);
        assert_eq!(compute_audio_frames(9, 24.0), 9);
        assert_eq!(compute_audio_frames(1, 24.0), 1);
        // 121f@24fps: 121/24·25 = 126.04 → 126.
        assert_eq!(compute_audio_frames(121, 24.0), 126);
    }

    #[test]
    fn vocoder_is_bigvgan_and_48khz() {
        let v = VocoderConfig::ltx_2_3();
        assert!(v.core.is_bigvgan());
        assert!(v.bwe.as_ref().unwrap().is_bigvgan());
        assert_eq!(v.final_sample_rate(), 48000);
        assert_eq!(v.core.upsample_rates.iter().product::<i32>(), 160);
    }

    #[test]
    fn audio_vae_levels() {
        let a = AudioVaeConfig::ltx_2_3();
        assert_eq!(a.num_resolutions(), 3);
        assert!(!a.mid_block_add_attention);
        assert_eq!(a.z_channels, 8);
    }
}

#[cfg(test)]
mod sc_18758_delta_tests {
    use super::*;

    /// sc-18758: 2.3's defaults reproduce the reference `ff_bias`/`audio_ff_bias` absent-key fallback
    /// (`True`) and `use_keyframes_abs_pos_embedding` absent-key fallback (`False`).
    #[test]
    fn ltx_2_3_defaults_match_the_reference_absent_key_fallback() {
        let cfg = AvConfig::ltx_2_3();
        assert!(cfg.video.ff_bias);
        assert!(cfg.audio_ff_bias);
        assert!(!cfg.video.use_keyframes_abs_pos_embedding);
        assert!(ConnectorConfig::ltx_2_3().ff_bias);
        assert!(ConnectorConfig::ltx_2_3_audio().ff_bias);
    }

    /// The 2.3→2.5 config delta, verified against the real shipped header (both distilled and dev
    /// `transformer.safetensors`): `ff_bias:false` (video only) + `use_keyframes_abs_pos_embedding:
    /// true`. `audio_ff_bias` is **not** part of the delta — the real 2.5 metadata carries no such
    /// key, so it stays the reference absent-key default (`True`) on both configs, same as 2.3.
    #[test]
    fn ltx_2_5_flips_exactly_the_measured_delta() {
        let c23 = AvConfig::ltx_2_3();
        let c25 = AvConfig::ltx_2_5();

        assert!(c23.video.ff_bias);
        assert!(!c25.video.ff_bias);
        assert!(!c23.video.use_keyframes_abs_pos_embedding);
        assert!(c25.video.use_keyframes_abs_pos_embedding);
        // audio_ff_bias is unchanged (true) on both — real header carries no `audio_ff_bias` key.
        assert!(c23.audio_ff_bias);
        assert!(c25.audio_ff_bias);

        // Not part of the delta — unaffected by the `ltx_2_5()` constructor.
        assert_eq!(c23.video.num_layers, c25.video.num_layers);
        assert_eq!(c23.video.num_heads, c25.video.num_heads);
        assert_eq!(c23.video.head_dim, c25.video.head_dim);
        assert_eq!(c23.audio_heads, c25.audio_heads);
        assert_eq!(c23.audio_head_dim, c25.audio_head_dim);
        assert_eq!(c23.cross_inner, c25.cross_inner);
        assert_eq!(c23.cross_max_pos, c25.cross_max_pos);
    }
}
