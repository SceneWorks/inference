//! LTX-2.x model configuration.
//!
//! Two sources, one set of structs:
//!
//! * **Hardcoded LTX-2.3 constants** (`*::ltx_2_3()`) for the shipped dense BF16 checkpoint. The
//!   original `Lightricks/LTX-2.3` repo ships no `embedded_config.json`; the same values live in the
//!   safetensors `__metadata__["config"]` blob and are fixed for that model family.
//! * **Per-component configs read from a split bundle** (`*::from_bundle`, sc-18757). LTX-2.5 ships
//!   one file per component, each carrying only its own config section, so every struct below reads
//!   **its own** component's section from **its own** file. Nothing is defaulted across components:
//!   a video-VAE file with no `config.vae` is an error, never a 2.3-shaped fallback.
//!
//! The 2.3 constants remain the *fallback within a section* (a key the section omits keeps its 2.3
//! value, exactly as the MLX sibling's `embedded_config.json` readers do); they are never a fallback
//! for a whole missing section.
//!
//! Legacy note — LTX-2.3 (distilled 22B): hardcoded constants for the shipped dense BF16
//! checkpoint (`ltx-2.3-22b-distilled.safetensors`). The mlx provider reads `embedded_config.json`
//! to support the quantized split checkpoints; we load the single dense file and pin the LTX-2.3
//! values directly (they are fixed for this model family).
//!
//! This is **video+audio**: the video-stack DiT, the Gemma-3-12B text encoder, the video connector,
//! and the video VAE decoder, plus the synchronized-audio stack (audio text head + connector, the
//! dual-modal AV DiT, the audio VAE decoder, and the vocoder — sc-5495) are all consumed. The 2-stage
//! latent upsampler, prompt-enhance, and fp8/on-the-fly quant are deferred to follow-up stories.
//! I2V/keyframes/IC-LoRA clips and inference LoRA/PEFT-LoKr are wired by the package pipeline.

use candle_gen::gen_core::ltx_checkpoint::{
    caption_feature_version, CaptionFeatureVersion, LtxBundle, LtxComponent,
};
use candle_gen::gen_core::{self, Error as GenError};
use serde_json::Value;

/// Registry id (the distilled 22B text-to-video model).
pub const MODEL_ID: &str = "ltx_2_3_distilled";
/// Registry id for LoRA training.
///
/// Training targets the base LTX-2.3 family recipe, while generation remains the separately pinned
/// distilled engine [`MODEL_ID`]. Keep these ids distinct so callers can select the trainer without
/// changing the generator route.
pub const TRAINER_ID: &str = "ltx_2_3";

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
        }
    }
    /// Inner dim `heads × head_dim` = 4096.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Read the video-stack dims from a `config.transformer` object (sc-18757). Keys the section
    /// omits keep their [`ltx_2_3`](Self::ltx_2_3) value; the section itself is never defaulted.
    pub fn from_transformer_config(t: &Value) -> Self {
        let mut cfg = Self::ltx_2_3();
        cfg.num_layers = get_usize(t, "num_layers", cfg.num_layers);
        cfg.num_heads = get_usize(t, "num_attention_heads", cfg.num_heads);
        cfg.head_dim = get_usize(t, "attention_head_dim", cfg.head_dim);
        cfg.norm_eps = get_f64(t, "norm_eps", cfg.norm_eps);
        cfg.rope_theta = get_f64(t, "positional_embedding_theta", cfg.rope_theta);
        cfg.rope_max_pos = get_i32_3(t, "positional_embedding_max_pos", cfg.rope_max_pos);
        cfg.timestep_scale_multiplier = get_f64(
            t,
            "timestep_scale_multiplier",
            cfg.timestep_scale_multiplier,
        );
        cfg
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
    /// Which caption feature extractor the checkpoint's config selects — resolved from the
    /// transformer section's four `caption_proj*` keys by
    /// [`caption_feature_version`], never from a per-model constant.
    /// [`ltx_2_3`](Self::ltx_2_3) carries the value the shipped LTX-2.3 config **resolves to**
    /// (V2, through the measured legacy carve-out), so the constant and the config agree by
    /// construction rather than by assertion.
    pub caption_feature_version: CaptionFeatureVersion,
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
            // Measured off the shipped `SceneWorks/ltx-2.3-mlx` `embedded_config.json`: it declares
            // only `caption_projection_{first,second}_linear: false` plus
            // `text_encoder_norm_type: "per_token_rms"`, which the carve-out resolves to V2.
            caption_feature_version: CaptionFeatureVersion::V2,
        }
    }
    /// Audio inner dim `heads × head_dim` = 2048.
    pub fn audio_inner(&self) -> usize {
        self.audio_heads * self.audio_head_dim
    }

    /// Read the full dual-modal dims from a `config.transformer` object (sc-18757).
    ///
    /// `cross_max_pos` mirrors `LTXModel.__init__`'s `cross_pe_max_pos = max(video_max_pos[0],
    /// audio_max_pos)` rather than being read directly — it is derived, not declared.
    ///
    /// Fallible because the caption feature-extractor selection is config-driven and an
    /// undetectable `caption_proj*` shape is a load error, not a default.
    pub fn from_transformer_config(t: &Value) -> gen_core::Result<Self> {
        let mut cfg = Self::ltx_2_3();
        cfg.video = TransformerConfig::from_transformer_config(t);
        cfg.audio_heads = get_usize(t, "audio_num_attention_heads", cfg.audio_heads);
        cfg.audio_head_dim = get_usize(t, "audio_attention_head_dim", cfg.audio_head_dim);
        cfg.cross_inner = get_usize(t, "audio_cross_attention_dim", cfg.cross_inner);
        cfg.audio_max_pos =
            get_i32_array_first(t, "audio_positional_embedding_max_pos", cfg.audio_max_pos);
        cfg.cross_max_pos = cfg.video.rope_max_pos[0].max(cfg.audio_max_pos);
        cfg.caption_feature_version = caption_feature_version(t)?;
        Ok(cfg)
    }

    /// Read the dual-modal dims from a split bundle's **transformer** component (sc-18757).
    pub fn from_bundle(bundle: &LtxBundle) -> gen_core::Result<Self> {
        let transformer = bundle.require(LtxComponent::Transformer)?;
        Self::from_transformer_config(transformer.config()?)
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
        }
    }

    /// Read the **video** connector dims from a `config.transformer` object (sc-18757). Both
    /// connectors are declared on the transformer's own section — they are part of the DiT's config,
    /// not a component of their own.
    pub fn from_transformer_config(t: &Value) -> Self {
        let mut cfg = Self::ltx_2_3();
        cfg.num_layers = get_usize(t, "connector_num_layers", cfg.num_layers);
        cfg.num_heads = get_usize(t, "connector_num_attention_heads", cfg.num_heads);
        cfg.head_dim = get_usize(t, "connector_attention_head_dim", cfg.head_dim);
        cfg.num_registers = get_usize(t, "connector_num_learnable_registers", cfg.num_registers);
        cfg.max_pos = get_i32_array_first(t, "connector_positional_embedding_max_pos", cfg.max_pos);
        cfg.norm_eps = get_f64(t, "norm_eps", cfg.norm_eps);
        cfg.rope_theta = get_f64(t, "positional_embedding_theta", cfg.rope_theta);
        cfg
    }

    /// Read the **audio** connector dims from a `config.transformer` object (sc-18757). Layer count,
    /// register count and RoPE geometry are shared with the video connector; only the head geometry
    /// has its own `audio_connector_*` keys.
    pub fn audio_from_transformer_config(t: &Value) -> Self {
        let mut cfg = Self::from_transformer_config(t);
        let audio = Self::ltx_2_3_audio();
        cfg.num_heads = get_usize(t, "audio_connector_num_attention_heads", audio.num_heads);
        cfg.head_dim = get_usize(t, "audio_connector_attention_head_dim", audio.head_dim);
        cfg
    }

    /// Read the video connector dims from a split bundle's transformer component (sc-18757).
    pub fn from_bundle(bundle: &LtxBundle) -> gen_core::Result<Self> {
        let transformer = bundle.require(LtxComponent::Transformer)?;
        Ok(Self::from_transformer_config(transformer.config()?))
    }

    /// Read the audio connector dims from a split bundle's transformer component (sc-18757).
    pub fn audio_from_bundle(bundle: &LtxBundle) -> gen_core::Result<Self> {
        let transformer = bundle.require(LtxComponent::Transformer)?;
        Ok(Self::audio_from_transformer_config(transformer.config()?))
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

    /// Read the decoder structure from an `audio_vae.model.params.ddconfig` object (sc-18757).
    pub fn from_ddconfig(dd: &Value) -> Self {
        let mut cfg = Self::ltx_2_3();
        cfg.ch = get_i32(dd, "ch", cfg.ch);
        cfg.out_ch = get_i32(dd, "out_ch", cfg.out_ch);
        if let Some(mult) = get_i32_vec(dd, "ch_mult") {
            cfg.ch_mult = mult;
        }
        cfg.num_res_blocks = get_i32(dd, "num_res_blocks", cfg.num_res_blocks);
        cfg.z_channels = get_i32(dd, "z_channels", cfg.z_channels);
        cfg.mel_bins = get_i32(dd, "mel_bins", cfg.mel_bins);
        cfg.mid_block_add_attention =
            get_bool(dd, "mid_block_add_attention", cfg.mid_block_add_attention);
        cfg
    }

    /// Read the decoder structure from a split bundle's **audio VAE** component (sc-18757).
    ///
    /// The `ddconfig` sits at `audio_vae.model.params.ddconfig`; a file that flattens it onto the
    /// `audio_vae` block itself is accepted too, since both spellings appear across LTX-2.x extracts
    /// — either way the values come from this component's own config. An absent `config.audio_vae`
    /// is an error, not the 2.3 shape.
    pub fn from_bundle(bundle: &LtxBundle) -> gen_core::Result<Self> {
        let audio = bundle.require(LtxComponent::AudioVae)?;
        let block = audio.config()?;
        let dd = block
            .get("model")
            .and_then(|v| v.get("params"))
            .and_then(|v| v.get("ddconfig"))
            .unwrap_or(block);
        Ok(Self::from_ddconfig(dd))
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

    /// Read one generator's structure from a `config.vocoder.{vocoder,bwe}` object (sc-18757),
    /// falling back per-key to `base` (the matching LTX-2.3 generator).
    fn read(v: &Value, base: &VocoderGenConfig) -> Self {
        let mut cfg = base.clone();
        if let Some(a) = get_i32_vec(v, "upsample_rates") {
            cfg.upsample_rates = a;
        }
        if let Some(a) = get_i32_vec(v, "upsample_kernel_sizes") {
            cfg.upsample_kernel_sizes = a;
        }
        if let Some(a) = get_i32_vec(v, "resblock_kernel_sizes") {
            cfg.resblock_kernel_sizes = a;
        }
        if let Some(rows) = v.get("resblock_dilation_sizes").and_then(Value::as_array) {
            let parsed: Vec<Vec<i32>> = rows
                .iter()
                .filter_map(|row| {
                    row.as_array()
                        .map(|r| r.iter().filter_map(json_i32).collect())
                })
                .collect();
            if !parsed.is_empty() {
                cfg.resblock_dilation_sizes = parsed;
            }
        }
        if let Some(s) = v.get("resblock").and_then(Value::as_str) {
            cfg.resblock = s.to_string();
        }
        if let Some(s) = v.get("activation").and_then(Value::as_str) {
            cfg.activation = s.to_lowercase();
        }
        cfg.use_tanh_at_final = get_bool(v, "use_tanh_at_final", cfg.use_tanh_at_final);
        cfg.apply_final_activation =
            get_bool(v, "apply_final_activation", cfg.apply_final_activation);
        cfg
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

    /// Read the core + BWE generators from a `config.vocoder` object (sc-18757).
    ///
    /// An absent `bwe` sub-object means a **single-stage** vocoder (the pre-2.3 shape upstream's
    /// `VocoderConfigurator` still supports) — the BWE stage is dropped, not defaulted to the 2.3
    /// one, because a checkpoint without BWE weights would otherwise be built with a randomly
    /// initialized second generator.
    pub fn from_vocoder_config(v: &Value) -> Self {
        let mut cfg = Self::ltx_2_3();
        cfg.core = VocoderGenConfig::read(
            v.get("vocoder").unwrap_or(v),
            &VocoderGenConfig::ltx_2_3_core(),
        );
        match v.get("bwe").filter(|b| b.is_object()) {
            Some(bwe) => {
                cfg.bwe = Some(VocoderGenConfig::read(
                    bwe,
                    &VocoderGenConfig::ltx_2_3_bwe(),
                ));
                cfg.bwe_input_sample_rate =
                    get_i32(bwe, "input_sampling_rate", cfg.bwe_input_sample_rate);
                cfg.output_sample_rate = cfg.bwe_input_sample_rate;
                cfg.bwe_output_sample_rate =
                    get_i32(bwe, "output_sampling_rate", cfg.bwe_output_sample_rate);
                cfg.bwe_hop_length = get_i32(bwe, "hop_length", cfg.bwe_hop_length);
                // Upstream builds the BWE mel-STFT window from `n_fft` (`filter_length` = `n_fft`).
                cfg.bwe_win_length = get_i32(bwe, "n_fft", cfg.bwe_win_length);
            }
            None => {
                cfg.bwe = None;
                cfg.output_sample_rate = get_i32(
                    v.get("vocoder").unwrap_or(v),
                    "output_sampling_rate",
                    cfg.output_sample_rate,
                );
            }
        }
        cfg
    }

    /// Read the vocoder from a split bundle's **audio VAE** component (sc-18757).
    ///
    /// The vocoder has no file of its own: LTX-2.5 ships it inside the audio-VAE component, whose
    /// metadata carries `config.audio_vae` **and** `config.vocoder`. An absent sibling section is an
    /// error naming the component, not the HiFi-GAN defaults.
    pub fn from_bundle(bundle: &LtxBundle) -> gen_core::Result<Self> {
        let audio = bundle.require(LtxComponent::AudioVae)?;
        Ok(Self::from_vocoder_config(audio.config_section("vocoder")?))
    }
}

/// The video-VAE structure a split bundle's video-VAE component declares (sc-18757).
///
/// candle's `LtxVideoVae` currently infers its block ladder from the weight shapes rather than from
/// a config, so this carries the declared facts the loader needs to *route* — which decoder family
/// the file holds, and the latent/patch geometry — instead of a full block list. Reading it proves
/// the component's own `config.vae` is present and well-formed before any weight is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoVaeDeclaration {
    /// `config.vae._class_name` — `CausalVideoAutoencoder` (conv) or `CausalDiffusionVAE` (DiffVAE).
    pub class_name: String,
    /// `latent_channels` — the DiT in/out width.
    pub latent_channels: i32,
    /// `patch_size` — the pixel-shuffle patchify factor.
    pub patch_size: i32,
}

impl VideoVaeDeclaration {
    /// The shipped LTX-2.3 convolutional VAE declaration.
    pub fn ltx_2_3() -> Self {
        Self {
            class_name: gen_core::ltx_checkpoint::CONV_VIDEO_VAE_CLASS.to_string(),
            latent_channels: LATENT_CHANNELS as i32,
            patch_size: 4,
        }
    }

    /// True when the file holds the diffusion decoder rather than the convolutional one.
    pub fn is_diffusion(&self) -> bool {
        self.class_name == gen_core::ltx_checkpoint::DIFFUSION_VIDEO_VAE_CLASS
    }

    /// Read the declaration from a `config.vae` object.
    pub fn from_vae_config(v: &Value) -> Self {
        let mut cfg = Self::ltx_2_3();
        if let Some(s) = v.get("_class_name").and_then(Value::as_str) {
            cfg.class_name = s.to_string();
        }
        // `CausalDiffusionVAE` nests the encoder/decoder geometry; the latent width it reports is
        // the encoder's `out_channels` (upstream `_prepare_video_encoder_kwargs`).
        cfg.latent_channels = get_i32(v, "latent_channels", cfg.latent_channels);
        let geometry = v.get("decoder").filter(|d| d.is_object()).unwrap_or(v);
        cfg.patch_size = get_i32(geometry, "patch_size", cfg.patch_size);
        cfg
    }

    /// Read the declaration from a split bundle's video-VAE component (sc-18757).
    ///
    /// `component` selects which of the two LTX-2.5 video VAEs to read — they are separate files
    /// with separate structures, so there is no silent pick between them. A component file with no
    /// `config.vae` is an error, never a 2.3-shaped default.
    pub fn from_bundle(bundle: &LtxBundle, component: LtxComponent) -> gen_core::Result<Self> {
        match component {
            LtxComponent::ConvVideoVae | LtxComponent::DiffusionVideoVae => {}
            other => {
                return Err(GenError::Msg(format!(
                    "ltx: `{}` is not a video VAE component",
                    other.id()
                )))
            }
        }
        let vae = bundle.require(component)?;
        Ok(Self::from_vae_config(vae.config()?))
    }
}

// --- JSON readers ---------------------------------------------------------------------------------
// Float-aware: the LTX converters emit some integer config values as floats (e.g.
// `av_ca_timestep_scale_multiplier: 1000.0`), which `as_i64` silently drops.

fn json_i32(n: &Value) -> Option<i32> {
    n.as_i64()
        .map(|x| x as i32)
        .or_else(|| n.as_f64().map(|f| f as i32))
}

fn get_i32(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key).and_then(json_i32).unwrap_or(default)
}

fn get_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(json_i32)
        .filter(|n| *n >= 0)
        .map_or(default, |n| n as usize)
}

fn get_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn get_i32_vec(v: &Value, key: &str) -> Option<Vec<i32>> {
    let out: Vec<i32> = v
        .get(key)?
        .as_array()?
        .iter()
        .filter_map(json_i32)
        .collect();
    (!out.is_empty()).then_some(out)
}

/// First element of an int array (the single-element `*_max_pos` arrays LTX ships).
fn get_i32_array_first(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(json_i32)
        .unwrap_or(default)
}

fn get_i32_3(v: &Value, key: &str, default: [i32; 3]) -> [i32; 3] {
    match v.get(key).and_then(Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut out = default;
            for (slot, value) in out.iter_mut().zip(a) {
                if let Some(n) = json_i32(value) {
                    *slot = n;
                }
            }
            out
        }
        _ => default,
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
mod component_config_tests {
    use super::*;

    /// An LTX-2.5-shaped `config.transformer` with values that differ from every 2.3 constant, so a
    /// field that is silently not parsed shows up as a failure rather than passing on the default.
    fn transformer_section() -> Value {
        serde_json::json!({
            "_class_name": "AVTransformer3DModel",
            "num_layers": 44,
            "num_attention_heads": 24,
            "attention_head_dim": 96,
            "norm_eps": 1e-5,
            "positional_embedding_theta": 20000.0,
            "positional_embedding_max_pos": [24, 4096, 4096],
            "timestep_scale_multiplier": 500.0,
            "audio_num_attention_heads": 16,
            "audio_attention_head_dim": 32,
            "audio_cross_attention_dim": 1024,
            "audio_positional_embedding_max_pos": [12],
            "connector_num_layers": 6,
            "connector_num_attention_heads": 20,
            "connector_attention_head_dim": 64,
            "connector_num_learnable_registers": 96,
            "connector_positional_embedding_max_pos": [2048],
            "audio_connector_num_attention_heads": 12,
            "audio_connector_attention_head_dim": 48
        })
    }

    #[test]
    fn the_transformer_section_drives_every_dit_field() {
        let cfg = AvConfig::from_transformer_config(&transformer_section())
            .expect("V1: no caption_proj* keys");
        assert_eq!(cfg.video.num_layers, 44);
        assert_eq!(cfg.video.num_heads, 24);
        assert_eq!(cfg.video.head_dim, 96);
        assert_eq!(cfg.video.inner_dim(), 24 * 96);
        assert!((cfg.video.norm_eps - 1e-5).abs() < f64::EPSILON);
        assert!((cfg.video.rope_theta - 20000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.video.rope_max_pos, [24, 4096, 4096]);
        // The converter emits this as a float; a non-float-aware reader would drop it.
        assert!((cfg.video.timestep_scale_multiplier - 500.0).abs() < f64::EPSILON);
        assert_eq!(cfg.audio_heads, 16);
        assert_eq!(cfg.audio_head_dim, 32);
        assert_eq!(cfg.audio_inner(), 16 * 32);
        assert_eq!(cfg.cross_inner, 1024);
        assert_eq!(cfg.audio_max_pos, 12);
        // `cross_pe_max_pos = max(video_max_pos[0], audio_max_pos)` — derived, not declared.
        assert_eq!(cfg.cross_max_pos, 24);
    }

    #[test]
    fn an_omitted_key_keeps_its_2_3_value_but_a_missing_section_never_does() {
        // Within a present section, an omitted key falls back — that is the documented behavior.
        let cfg = AvConfig::from_transformer_config(&serde_json::json!({"num_layers": 40}))
            .expect("V1: no caption_proj* keys");
        assert_eq!(cfg.video.num_layers, 40);
        assert_eq!(cfg.video.num_heads, AvConfig::ltx_2_3().video.num_heads);
    }

    #[test]
    fn both_connectors_read_their_own_keys_off_the_transformer_section() {
        let t = transformer_section();
        let video = ConnectorConfig::from_transformer_config(&t);
        assert_eq!(video.num_layers, 6);
        assert_eq!(video.num_heads, 20);
        assert_eq!(video.head_dim, 64);
        assert_eq!(video.num_registers, 96);
        assert_eq!(video.max_pos, 2048);
        let audio = ConnectorConfig::audio_from_transformer_config(&t);
        // Shared structure…
        assert_eq!(audio.num_layers, 6);
        assert_eq!(audio.num_registers, 96);
        assert_eq!(audio.max_pos, 2048);
        // …its own head geometry.
        assert_eq!(audio.num_heads, 12);
        assert_eq!(audio.head_dim, 48);
    }

    #[test]
    fn the_audio_vae_ddconfig_drives_the_decoder_structure() {
        let cfg = AudioVaeConfig::from_ddconfig(&serde_json::json!({
            "ch": 96, "out_ch": 1, "ch_mult": [1, 2, 4, 8], "num_res_blocks": 3,
            "z_channels": 16, "mel_bins": 128, "mid_block_add_attention": true
        }));
        assert_eq!(cfg.ch, 96);
        assert_eq!(cfg.out_ch, 1);
        assert_eq!(cfg.ch_mult, vec![1, 2, 4, 8]);
        assert_eq!(cfg.num_resolutions(), 4);
        assert_eq!(cfg.num_res_blocks, 3);
        assert_eq!(cfg.z_channels, 16);
        assert_eq!(cfg.mel_bins, 128);
        assert!(cfg.mid_block_add_attention);
    }

    #[test]
    fn the_vocoder_section_drives_both_generators_and_the_sample_rates() {
        let cfg = VocoderConfig::from_vocoder_config(&serde_json::json!({
            "vocoder": {
                "resblock": "AMP1", "activation": "snakebeta",
                "upsample_rates": [5, 2, 2, 2, 2, 2], "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4]
            },
            "bwe": {
                "resblock": "AMP1", "activation": "snakebeta",
                "input_sampling_rate": 16000, "output_sampling_rate": 48000,
                "hop_length": 80, "n_fft": 512
            }
        }));
        assert!(cfg.core.is_bigvgan());
        assert_eq!(cfg.core.upsample_rates.iter().product::<i32>(), 160);
        let bwe = cfg.bwe.as_ref().expect("BWE stage");
        assert!(bwe.is_bigvgan());
        // Upstream builds the BWE generator with `apply_final_activation=False`.
        assert!(!bwe.apply_final_activation);
        assert_eq!(cfg.output_sample_rate, 16000);
        assert_eq!(cfg.final_sample_rate(), 48000);
        assert_eq!(cfg.bwe_hop_length, 80);
        assert_eq!(cfg.bwe_win_length, 512);
    }

    #[test]
    fn a_vocoder_without_bwe_drops_the_stage_instead_of_defaulting_it() {
        // The pre-2.3 single-stage shape: a checkpoint with no BWE weights must not be built with a
        // randomly-initialized second generator borrowed from the 2.3 config.
        let cfg = VocoderConfig::from_vocoder_config(&serde_json::json!({
            "vocoder": {"resblock": "1", "activation": "leaky_relu", "output_sampling_rate": 24000}
        }));
        assert!(cfg.bwe.is_none());
        assert!(!cfg.core.is_bigvgan());
        assert_eq!(cfg.final_sample_rate(), 24000);
    }

    #[test]
    fn the_video_vae_declaration_distinguishes_conv_from_diffusion() {
        let conv = VideoVaeDeclaration::from_vae_config(&serde_json::json!({
            "_class_name": "CausalVideoAutoencoder", "latent_channels": 128, "patch_size": 4
        }));
        assert!(!conv.is_diffusion());
        assert_eq!(conv.latent_channels, 128);
        assert_eq!(conv.patch_size, 4);
        let diff = VideoVaeDeclaration::from_vae_config(&serde_json::json!({
            "_class_name": "CausalDiffusionVAE",
            "latent_channels": 128,
            "decoder": {"patch_size": 8, "head_dim": 64}
        }));
        assert!(diff.is_diffusion());
        // The DiffVAE nests its geometry under `decoder`.
        assert_eq!(diff.patch_size, 8);
    }
}
