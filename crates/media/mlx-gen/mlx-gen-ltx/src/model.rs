//! `mlx-gen-ltx` model entry: the LTX-2.3 **AudioVideo** descriptor, the config-driven `load`, the
//! public `generate`, and registry self-registration.
//!
//! **Scope (sc-2684):** the production path is the full **synchronized audio+video** generation
//! (`generate_av.py`) — prompt → Gemma-3 tokenizer → [`LtxTextEncoder::encode_av`] (video 4096 +
//! audio 2048 embeddings) → seeded noise → the joint 2-stage distilled denoise
//! ([`crate::pipeline::generate_av_latents`]:
//! both streams through the dual-modality [`AvDiT`] with cross-modal attention every step; the video is
//! 2× upsampled between stages, the audio is not) → [`LtxVideoVae`] decode → uint8 RGB frames **plus**
//! [`AudioDecoder`] → [`LtxVocoder`] → an [`mlx_gen::media::AudioTrack`]. The audio is always denoised
//! (it conditions the video via cross-modal attention), so the video differs from the video-only
//! sc-2679 building block (`LtxDiT`, audio disabled). `--no-audio` (`req.video_mode == "no_audio"`)
//! runs the full A/V denoise but skips the audio decode (`audio: None`).
//!
//! 16-bit-WAV write + peak-normalize + the `ffmpeg -c:v copy -c:a aac -shortest` mux are **host-side**
//! (the `AudioTrack` is the raw vocoder waveform — `generate_av.py`'s `audio_np` before `save_audio`),
//! matching how MP4 video muxing already lives outside the crate (the Wan sibling).
//!
//! The Gemma text-encoder weights are a **separate** snapshot (the base model dir holds only the
//! `connector`/transformer/vae); `resolve_gemma_dir` reads it from the **required**
//! `LoadSpec::text_encoder` slot (`mlx-community/gemma-3-12b-it-bf16`). As of sc-13664 there is no
//! env side-channel or on-disk HF-cache fallback — the caller provisions the path, and an absent slot
//! is a load-time error.
//!
//! **Quantization (sc-2686).** The transformer ships **selectively quantized** (attn/ff Linears
//! packed U32 + `scales`); the **bits/group ride on the checkpoint's `split_model.json`** —
//! `ltx_2_3_base_q4` is **Q4**, `ltx_2_3_base_q8` is **Q8**, group 64 — read into the DiT
//! [`Precision`], never hardcoded. `LoadSpec::quantize`, when set, only *asserts* the expected level
//! (LTX can't re-quantize a dense checkpoint — there is no dense LTX transformer; it ships pre-packed),
//! so a mismatch with the manifest is a load error. Connector / VAE / upsampler are dense bf16 (the
//! reference quantizes the transformer only); the Gemma text encoder is dense bf16 by default
//! (reference TE quant rides on the *Gemma* snapshot's `config.json`).
//!
//! **Precision.** Selected by `LoadSpec::precision`: `Bf16` (the default) → the reference's **native**
//! bf16 activations × quantized weights ([`Precision::quant_bf16`]) — the production-speed path;
//! `Fp32` → [`Precision::quant_f32`] (f32 activations × quantized weights) — the quality target. Both
//! are bit-exact to their reference golden (sc-2842). The latent statistics follow the path dtype (so
//! the upsampler + denoise run in that precision); the VAE decode stays f32 (a post-sampling quality
//! island, pixel-parity either way), and the Gemma backbone runs bf16 as the reference does. Distilled
//! 2-stage → **no CFG** (guidance baked in).
//!
//! **I2V (sc-2685):** a single conditioning [`Conditioning::Reference`] image is VAE-encoded at both
//! stage resolutions and injected into the **video** stream as a clean latent at frame 0 (per-frame
//! denoise mask, `image_strength` → `1 − strength`), threaded through the joint A/V denoise via
//! `generate_av_latents`' `video_cond` — the audio stays pure-noise, matching `generate_av.py`'s
//! I2V+Audio. The VAE **encoder** is loaded **lazily** on first encode, so pure-T2V runs never pay
//! its resident cost (F-048). LoRA/LoKr are sibling slices.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlx_rs::{random, Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{
    layout_for_declared_version, LtxBundle, LtxCheckpointLayout, LtxComponent,
};
use mlx_gen::gen_core::{reject_unknown_components, ArtifactSeal};
use mlx_gen::gen_core::{GenerationReport, GenerationWarning};
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    curated_sampler_names, default_seed, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Precision as LoadPrecision, Progress, Result, StepSupport, WeightsSource,
};

use crate::audio_vae::AudioDecoder;
use crate::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use crate::dev_sampler::TransformerVariant;
use crate::dfr::{generate_dfr_av_latents, DfrComponents, DfrRequest};
use crate::diff_vae::{DiffVaeMode, DiffVaeQuant, NaDiffusionDecoder, NaDiffusionDecoderConfig};
use crate::duration_head::DurationHead;
use crate::enhance::{self, EnhanceConfig, SampleParams};
use crate::gemma::{GemmaConfig, GemmaModel, GemmaQuant};
use crate::gemma4_te::Ltx25TextEncoder;
use crate::image_crf::{condition_image_for_checkpoint, default_image_recompress};
use crate::pipeline::{
    decode_audio_track, decode_to_frames_with_tiling, generate_av_latents_for_variant,
    generate_av_latents_iclora, preprocess_conditioning_clip, to_uint8_frames, StageClip,
    StageKeyframe, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
use crate::positions::{
    compute_audio_frames, create_audio_position_grid, create_position_grid_with, DEFAULT_FPS,
};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::{GemmaAssets, Ltx25Tokenizer, LtxTokenizer};
use crate::transformer::{AvDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;
use mlx_llm::core_llm::{ChatTemplate as CoreChatTemplate, JinjaChatTemplate, Message};
use mlx_llm::models::{gemma4_mm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig};
use mlx_llm::primitives::Weights as LlmWeights;
use mlx_llm::{CausalLm, ModelConfig, PrefixCache};
use sha2::{Digest, Sha256};

// sc-27342: the provider's descriptors, load paths, per-generate staging, generation body,
// conditioning, prompt enhancement and Generator surfaces live in sibling modules; the shared
// constants and the provider types stay here.
mod conditioning;
mod descriptor;
mod generate;
mod generator;
mod load;
mod prompt_enhance;
mod staging;
pub use conditioning::*;
pub use descriptor::*;
use generate::*;
pub(crate) use generator::*;
pub use load::*;
use prompt_enhance::*;

/// Public provider id: `"ltx_2_3"`.
pub const MODEL_ID: &str = "ltx_2_3";
/// Public provider id for the split-component Gemma-4 LTX-2.5 route.
pub const MODEL_25_ID: &str = "ltx_2_5";
/// The stock, locally staged Gemma-4 instruction snapshot used only for opt-in LTX-2.5 prompt
/// enhancement.  It lives below the single LTX rehost rather than in a job-time HF cache.
const LTX25_ENHANCER_COMPONENT: &str = "enhancer";

/// Byte budget of the LTX-2.5 Gemma-4 enhancer's cross-request prefix cache (the system-prompt
/// KV reused between text-to-video enhancements; sc-24437 made the cache byte-budgeted). It lives
/// on the provider for as long as the load, so the 2.5 memory contract charges it beside the
/// enhancer weights whenever the enhancer snapshot is staged
/// ([`crate::memory_strategy_2_5`]), and this constant is the one figure both read. 1 GiB holds
/// one enhancement's prompt plus rewrite at bf16 on the 12B enhancer with room to spare; an entry
/// larger than the budget is simply not kept — the enhancement itself is unaffected, only the
/// next request's reuse is lost.
pub(crate) const LTX25_ENHANCER_PREFIX_CACHE_BYTES: u64 = 1 << 30;
/// Rehost-owned inventory that must accompany the exact stock enhancer snapshot.
const LTX25_ENHANCER_MANIFEST_FILE: &str = "sceneworks_asset_manifest.json";
/// Canonical SC-18780 publication manifest. Its semantic inventory digest binds the source repo,
/// immutable source revision, exact file names, byte lengths, and SHA-256 values.
const LTX25_ENHANCER_MANIFEST: &str = include_str!("../assets/ltx25_enhancer_manifest.json");
const LTX25_ENHANCER_SCHEMA: &str = "sceneworks-ltx25-enhancer-v1";
const LTX25_ENHANCER_REPOSITORY: &str = "google/gemma-4-12b-it";
const LTX25_ENHANCER_REVISION: &str = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7";
const LTX25_ENHANCER_INVENTORY_SHA256: &str =
    "8e094af99d14d8639760b07be5d433306b404fa3380be1e5043a53700d731e7e";

/// Upstream LTX-2 default negative conditioning for the non-distilled guided pipeline.  A caller's
/// explicit `negative_prompt` replaces this text; an omitted field does not mean an all-zero or
/// positive-conditioning branch.
const DEFAULT_NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// The supported DiffVAE execution recipe.  The planner still chooses untiled versus tiled from
/// the live process budget; this is the upstream semantic mode fed into that planner.
const DEFAULT_DIFFVAE_MODE: DiffVaeMode = DiffVaeMode::ChunkedEager;

/// The `model_version` declared by the all-in-one LTX-2.3 checkpoint layout, used to resolve its
/// generation params (sc-18759 — see [`crate::params`]). The split LTX-2.5 route resolves its
/// version from the bundle instead of this legacy-layout constant.
const CHECKPOINT_MODEL_VERSION: &str = "2.3.0";

/// Neutral gray the replace_person mask blends toward (reference `_apply_replacement_mask`).
const REPLACE_NEUTRAL: u32 = 118;

/// Reference text-encoder token budget (`LTX2TextEncoder.encode` default `max_length=1024`).
const MAX_PROMPT_TOKENS: usize = 1024;
/// LTX-2 latent channels.
pub(crate) const LATENT_CHANNELS: i32 = 128;
/// Audio latent channels (pre-patchify) and mel bins — the audio latent is `(1, 8, T, 16)`.
const AUDIO_LATENT_CHANNELS: i32 = 8;
const AUDIO_MEL_BINS: i32 = 16;
/// VAE temporal compression (8×): `latent_frames = 1 + (frames − 1) / 8`.
pub(crate) const TEMPORAL_SCALE: u32 = 8;
/// Upper bound on requested `num_frames` (DoS guard, F-058): `generate` sizes the per-stage noise +
/// audio buffers directly from the request, so an unbounded `frames` (e.g. a hostile `8_000_001`,
/// which still satisfies `frames % 8 == 1`) would drive multi-hundred-GB allocations before any
/// error. `1025` (= 1 + 8·128, ~40 s at 25 fps) is far above any realistic request.
pub(crate) const MAX_FRAMES: u32 = 1025;
pub(crate) const MIN_SIZE: u32 = 64;
pub(crate) const MAX_SIZE: u32 = 1280;
/// VAE spatial compression (32×); stage-1 additionally halves resolution.
pub(crate) const SPATIAL_SCALE: u32 = 32;
/// The request width/height multiple `validate_request` enforces: `2 × SPATIAL_SCALE` (= 64).
/// Stage-1 renders at half resolution (`latent_dims` divides by `2 · SPATIAL_SCALE`), so a request
/// dimension must be a multiple of *twice* the 32× VAE spatial compression for that division to be
/// exact. This is the pinned-engine stride SceneWorks ties `requiresDimensionsMultipleOf` to
/// (sc-12587), mirroring `wan::config::SIZE_MULTIPLE_14B`. Divergent by backend on purpose: candle's
/// single-stage `ltx_2_3_distilled` uses `SIZE_MULTIPLE = SPATIAL_SCALE` (= 32).
pub const SIZE_MULTIPLE: u32 = 2 * SPATIAL_SCALE;
/// The folded denoise-step total surfaced as `Progress::Step.total` (F-050): stage-1 (`STAGE1_SIGMAS`,
/// 8 steps) + stage-2 (`STAGE2_SIGMAS`, 3 steps) of the baked distilled schedule.
const TOTAL_STEPS: u32 = (STAGE1_SIGMAS.len() + STAGE2_SIGMAS.len() - 2) as u32;

/// Folds the LTX two-stage distilled denoise into one monotone `1..=TOTAL_STEPS` progress bar
/// (F-050, sc-11133). Each stage forwards a **per-stage** 1-based `current` (the σ-derived value from
/// `step_gate` on the curated path, or `i+1` on the native path) that restarts at 1 when the next
/// stage begins; a curated 2nd-order solver (`heun`/`dpmpp_sde`) also forwards the same `current`
/// twice per step (predictor + corrector). [`observe`](StageProgressFold::observe) maps that raw
/// stream to an absolute, deduplicated, clamped `current`: it returns `Some(current)` only when the
/// global bar strictly advances, so the emitted sequence is monotone non-decreasing, reaches `total`,
/// and never overruns it.
struct StageProgressFold {
    total: u32,
    /// Steps completed in prior stages (added to the current stage's forwarded value).
    stage_offset: u32,
    /// The last per-stage value observed (a drop below it signals a new stage).
    last_fwd: u32,
    /// The last absolute `current` emitted (dedupe / monotonicity anchor).
    emitted: u32,
}

impl StageProgressFold {
    fn new(total: u32) -> Self {
        Self {
            total,
            stage_offset: 0,
            last_fwd: 0,
            emitted: 0,
        }
    }

    /// Observe one forwarded per-stage `current`; returns the absolute bar position to emit, or
    /// `None` if the bar did not advance (a multi-eval repeat).
    fn observe(&mut self, fwd: u32) -> Option<u32> {
        if fwd < self.last_fwd {
            // The forwarded value dropped — the next stage restarted its counter at 1. Fold the
            // completed stages in so this stage's steps stack on top rather than overwriting.
            self.stage_offset = self.emitted;
        }
        self.last_fwd = fwd;
        let abs = (self.stage_offset + fwd).min(self.total);
        if abs > self.emitted {
            self.emitted = abs;
            Some(abs)
        } else {
            None
        }
    }
}
/// I2V conditioning strength when neither the `Reference` nor `req.strength` supplies one (reference
/// CLI `--image-strength` default): `1.0` = full denoise, fully pinning the conditioned frame.
const DEFAULT_IMAGE_STRENGTH: f32 = 1.0;
/// I2V conditioned frame index (reference CLI `--image-frame-idx` default). Single-image I2V pins the
/// **first** latent frame; multi-keyframe / first-last-frame at other indices is parity-plus (the
/// [`crate::conditioning`] primitive supports any index, but the reference CLI only wires one).
const IMAGE_FRAME_IDX: i32 = 0;

/// Text assets selected by a checkpoint layout. Keeping this selection in the shared execution
/// shell avoids a provider-local copy of the mature audio/video conditioning implementation.
enum TextAssets {
    Gemma3 {
        dir: PathBuf,
        quant: Option<GemmaQuant>,
    },
    Gemma4 {
        bundle: LtxBundle,
        connector: PathBuf,
        offload_policy: mlx_gen::gen_core::OffloadPolicy,
        enhancer: PathBuf,
        enhancer_template: JinjaChatTemplate,
    },
}

enum Tokenizer {
    Gemma3(LtxTokenizer),
    Gemma4(Ltx25Tokenizer),
}

impl Tokenizer {
    fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        match self {
            Self::Gemma3(tokenizer) => tokenizer.encode(prompt, max_length),
            Self::Gemma4(tokenizer) => tokenizer.encode(prompt, max_length),
        }
    }
}

enum StagedTextEncoder {
    Gemma3(Box<LtxTextEncoder>),
    Gemma4(Box<Ltx25TextEncoder>),
}

/// [`GenerationReport::warnings`] code: the best-effort LTX-2.3 prompt enhancer failed (or returned
/// nothing) and the render used the original prompt (sc-27451).
pub const PROMPT_ENHANCER_FALLBACK: &str = "prompt_enhancer_fallback";

struct StagedTextConditioning {
    enhanced_prompt: Option<String>,
    /// Why the best-effort 2.3 prompt enhancer fell back to the original prompt (sc-27451), surfaced
    /// as a [`PROMPT_ENHANCER_FALLBACK`] render warning.
    enhancer_fallback: Option<String>,
    video: Array,
    audio: Array,
    negative_video: Option<Array>,
    negative_audio: Option<Array>,
}

/// The LTX-2.5-only components which turn the shared 2.3 A/V shell into the declared 2.5
/// provider.  Keeping the choice here means all three advanced axes flow through the ordinary
/// `Generator::generate` route instead of being descriptor-only capability claims.
enum LtxExecution {
    Ltx23,
    Ltx25(Box<Ltx25Execution>),
}

struct Ltx25Execution {
    variant: TransformerVariant,
    duration_head: DurationHead,
    temporal_upsampler: LatentUpsampler,
    decoder: Ltx25Decoder,
}

enum Ltx25Decoder {
    Conv,
    Diffusion {
        decoder: Box<NaDiffusionDecoder>,
        mode: DiffVaeMode,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ltx25DecoderSelection {
    Conv,
    DiffusionBudgeted(DiffVaeMode),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DfrPlan {
    canvas_frames: u32,
    requested_frames: u32,
    keyframe_positions: Vec<i64>,
    temporal_upsample_rounds: u32,
}

impl StagedTextEncoder {
    fn encode_av(&self, ids: &Array, mask: &Array) -> Result<(Array, Array)> {
        match self {
            Self::Gemma3(encoder) => encoder.encode_av(ids, mask),
            Self::Gemma4(encoder) => encoder.encode_av(ids, mask),
        }
    }
}

/// The loaded LTX-2.3 model: the assembled **AudioVideo** components + the cached descriptor. The
/// production path is the joint A/V denoise (`generate_av.py`) — the audio latents are always
/// denoised (the cross-modal attention couples them to the video every step), so the video stream
/// differs from the video-only sc-2679 building block. Audio is decoded into the output unless
/// `--no-audio` (`req.video_mode == "no_audio"`).
pub struct Ltx {
    descriptor: ModelDescriptor,
    memory_strategy: Option<mlx_gen::gen_core::MemoryProviderContract>,
    memory_tier: Option<mlx_gen::gen_core::MemoryNumericTier>,
    memory_overlay: Option<String>,
    tokenizer: Tokenizer,
    // sc-10976 (epic 10975): the two GIANTS — the ~24 GB Gemma-3-12B text encoder and the AvDiT — are
    // NOT held resident. They are built on demand inside `generate` (load → use → drop), mirroring
    // Wan's `root`-only struct (`mlx-gen-wan/src/model.rs`): the TE is freed (+ `clear_cache()`) before
    // the DiT materializes, so peak ≈ max(TE, DiT) instead of the sum. The lazy-build inputs live here.
    transformer_path: PathBuf,
    config: LtxConfig,
    dit_prec: Precision,
    adapters: Vec<AdapterSpec>,
    text_assets: TextAssets,
    execution: LtxExecution,
    /// The optional **uncensored** 4-bit Gemma enhancer snapshot dir (the amoral
    /// `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`, sc-2845), staged by the caller in the
    /// `uncensored_enhancer` [`LoadSpec::components`] entry (sc-13664). `None` unless the caller
    /// provisioned it; a request that sets `use_uncensored_enhancer` without it staged is a
    /// generate-time actionable error (no `$LTX_UNCENSORED_GEMMA_DIR` / HF-cache scan any more).
    uncensored_enhancer: Option<PathBuf>,
    /// Reuses the immutable Gemma-4 system-prefix KV between enhancement requests. I2V prefills
    /// remain request-local because their image embeddings are input-specific.
    enhancer_cache: Option<Rc<RefCell<PrefixCache>>>,
    // The small components (each ≪ the TE / DiT) stay resident — the AvDiT is the denoise peak and the
    // Gemma drop is the win (sc-10976). Staging these too would add parity risk for ~no memory gain.
    upsampler: LatentUpsampler,
    vae: LtxVideoVae,
    audio_decoder: AudioDecoder,
    vocoder: LtxVocoder,
    latent_mean: Array,
    latent_std: Array,
    audio_sample_rate: u32,
    stat_dt: Dtype,
}

/// Lazy split-bundle LTX-2.5 provider.  Resolving the metadata/configuration is deliberately part
/// of `load`; materialising multi-gigabyte tensors remains request-scoped, matching the staged LTX
/// route and keeping ordinary catalog construction weights-free.
pub struct Ltx25 {
    descriptor: ModelDescriptor,
    spec: LoadSpec,
    variant: TransformerVariant,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    memory_tier: mlx_gen::gen_core::MemoryNumericTier,
    enhancer_cache: Rc<RefCell<PrefixCache>>,
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION_25 = descriptor_25 => load_25
}

#[cfg(test)]
mod tests;
