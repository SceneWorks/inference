//! `mlx-gen-ltx` model entry: the LTX-2.3 **AudioVideo** descriptor, the config-driven `load`, the
//! public `generate`, and registry self-registration.
//!
//! **Scope (sc-2684):** the production path is the full **synchronized audio+video** generation
//! (`generate_av.py`) — prompt → Gemma-3 tokenizer → [`LtxTextEncoder::encode_av`] (video 4096 +
//! audio 2048 embeddings) → seeded noise → the joint 2-stage distilled denoise ([`generate_av_latents`]:
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

use std::path::PathBuf;

use mlx_rs::{random, Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{
    layout_for_declared_version, LtxBundle, LtxCheckpointLayout, LtxComponent,
};
use mlx_gen::gen_core::reject_unknown_components;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    curated_sampler_names, default_seed, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Precision as LoadPrecision, Progress, Result, StepSupport, WeightsSource,
};

use crate::audio_vae::AudioDecoder;
use crate::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use crate::dfr::{generate_dfr_av_latents, DfrComponents, DfrRequest};
use crate::diff_vae::{DiffVaeMode, DiffVaeQuant, NaDiffusionDecoder, NaDiffusionDecoderConfig};
use crate::duration_head::DurationHead;
use crate::enhance::{self, EnhanceConfig, SampleParams};
use crate::gemma::{GemmaConfig, GemmaModel, GemmaQuant};
use crate::gemma4_te::Ltx25TextEncoder;
use crate::image_crf::{condition_image_for_checkpoint, default_image_recompress};
use crate::pipeline::{
    decode_audio_track, decode_to_frames_with_tiling, generate_av_latents,
    generate_av_latents_iclora, preprocess_conditioning_clip, to_uint8_frames, StageClip,
    StageKeyframe, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
use crate::positions::{compute_audio_frames, create_audio_position_grid, create_position_grid};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::{Ltx25Tokenizer, LtxTokenizer};
use crate::transformer::{AvDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Public provider id: `"ltx_2_3"`.
pub const MODEL_ID: &str = "ltx_2_3";
/// Public provider id for the split-component Gemma-4 LTX-2.5 route.
pub const MODEL_25_ID: &str = "ltx_2_5";

/// The supported DiffVAE execution recipe.  The planner still chooses untiled versus tiled from
/// the live process budget; this is the upstream semantic mode fed into that planner.
const DEFAULT_DIFFVAE_MODE: DiffVaeMode = DiffVaeMode::ChunkedEager;

/// The `model_version` declared by the all-in-one LTX-2.3 checkpoint layout, used to resolve its
/// generation params (sc-18759 — see [`crate::params`]). The split LTX-2.5 route resolves its
/// version from the bundle instead of this legacy-layout constant.
const CHECKPOINT_MODEL_VERSION: &str = "2.3.0";

/// Neutral gray the replace_person mask blends toward (reference `_apply_replacement_mask`).
const REPLACE_NEUTRAL: u32 = 118;

/// Port of the worker's `_apply_replacement_mask` (native-LTX replace_person): blend each frame's
/// person region toward neutral gray 118 by `strength`, so the IC-LoRA keyframe-append regenerates it
/// while the background is preserved. Byte-exact to Pillow: `gate = int(L(mask) · strength)`, then
/// `out = composite(gray, frame, gate)` where `L` is PIL's RGB→L (`(R·19595 + G·38470 + B·7471 +
/// 0x8000) >> 16`) and `composite` blends with a single rounded division
/// `(gray·gate + frame·(255−gate) + 127) / 255` (verified vs Pillow). The mask must already match the
/// frame size (the host delivers per-frame masks at the output resolution).
pub fn apply_replacement_mask(frame: &Image, mask: &Image, strength: f32) -> Result<Image> {
    let strength = strength.clamp(0.0, 1.0);
    if (frame.width, frame.height) != (mask.width, mask.height) {
        return Err(Error::Msg(format!(
            "replace_person mask {}x{} must match frame {}x{}",
            mask.width, mask.height, frame.width, frame.height
        )));
    }
    let (width, height) = (frame.width as usize, frame.height as usize);
    let pixel_count = width.checked_mul(height).ok_or_else(|| {
        Error::Msg(format!(
            "replace_person frame dimensions {}x{} overflow the host pixel count",
            frame.width, frame.height
        ))
    })?;
    let expected = mlx_gen::gen_core::imageops::checked_image_buffer_len(width, height, 3)
        .ok_or_else(|| {
            Error::Msg(format!(
                "replace_person frame dimensions {}x{} overflow the RGB8 buffer size",
                frame.width, frame.height
            ))
        })?;
    if frame.pixels.len() != expected || mask.pixels.len() != expected {
        return Err(Error::Msg("replace_person frame/mask must be RGB8".into()));
    }
    let mut out = vec![0u8; expected];
    for i in 0..pixel_count {
        let (r, g, b) = (
            mask.pixels[i * 3] as u32,
            mask.pixels[i * 3 + 1] as u32,
            mask.pixels[i * 3 + 2] as u32,
        );
        let l = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16; // PIL RGB→L
        let gate = ((l as f32 * strength) as u32).min(255); // PIL .point(int(v·s))
        for c in 0..3 {
            let fpx = frame.pixels[i * 3 + c] as u32;
            // PIL `Image.composite` blend: single rounded division (not two-term MULDIV255).
            out[i * 3 + c] = ((REPLACE_NEUTRAL * gate + fpx * (255 - gate) + 127) / 255) as u8;
        }
    }
    Ok(Image {
        width: frame.width,
        height: frame.height,
        pixels: out,
    })
}

/// Reference text-encoder token budget (`LTX2TextEncoder.encode` default `max_length=1024`).
const MAX_PROMPT_TOKENS: usize = 1024;
/// LTX-2 latent channels.
const LATENT_CHANNELS: i32 = 128;
/// Audio latent channels (pre-patchify) and mel bins — the audio latent is `(1, 8, T, 16)`.
const AUDIO_LATENT_CHANNELS: i32 = 8;
const AUDIO_MEL_BINS: i32 = 16;
/// VAE temporal compression (8×): `latent_frames = 1 + (frames − 1) / 8`.
const TEMPORAL_SCALE: u32 = 8;
/// Upper bound on requested `num_frames` (DoS guard, F-058): `generate` sizes the per-stage noise +
/// audio buffers directly from the request, so an unbounded `frames` (e.g. a hostile `8_000_001`,
/// which still satisfies `frames % 8 == 1`) would drive multi-hundred-GB allocations before any
/// error. `1025` (= 1 + 8·128, ~40 s at 25 fps) is far above any realistic request.
pub(crate) const MAX_FRAMES: u32 = 1025;
pub(crate) const MIN_SIZE: u32 = 64;
pub(crate) const MAX_SIZE: u32 = 1280;
/// VAE spatial compression (32×); stage-1 additionally halves resolution.
const SPATIAL_SCALE: u32 = 32;
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

/// Stable identity + advertised capabilities for the LTX-2.3 AudioVideo model (produces video frames
/// + a synchronized audio track).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::LTX_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt.
            // I2V single-image conditioning (sc-2685) is wired via `Reference`; audio is always
            // produced (sc-2684). Q4/Q8-of-everything is a sibling slice.
            supports_negative_prompt: false,
            // Reference = single-image I2V (sc-2685); Keyframe = first_last_frame / multi-keyframe
            // (replace-latent, epic 3040); `MultiReference` is the ordered 1–4 identity carrier
            // for replace_person (composited to the provider-native one-latent IC-LoRA input);
            // VideoClip = extend_clip / video_bridge (IC-LoRA keyframe-append — requires an
            // IC-LoRA adapter via `spec.adapters`).
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::Keyframe,
                ConditioningKind::VideoClip,
                ConditioningKind::ControlClip,
            ],
            // LoRA (sc-2687) + LoKr (sc-2393) in generate: forward-time residuals + per-pass
            // strength over the full video+audio+cross-modal surface.
            supports_lora: true,
            supports_lokr: true,
            // Quantization is checkpoint-driven (split_model.json); load() rejects on-the-fly
            // spec.quantize that disagrees with the manifest. No on-the-fly re-quant available.
            supported_quants: &[],
            // Curated unified solvers (epic 7114, sc-7122): LTX exposes the SAMPLER axis but NO scheduler
            // (matching ComfyUI) — it keeps its baked distilled σ schedule (8+3 steps) and only swaps the
            // integrator (over the two-stream `MlxAvLatentOps`, joint video+audio). LTX is distilled, so
            // `euler` is the recommended integrator and stays the byte-exact default (unset sampler); the
            // others are exposed for parity with ComfyUI's menu. T2V only — the I2V/keyframe/clip paths
            // (per-token σ + post-step blend) stay native.
            samplers: curated_sampler_names(),
            // height/width must be divisible by SIZE_MULTIPLE (= 2×SPATIAL_SCALE; stage-1 runs at //2//32).
            supported_guidance_methods: vec![],
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: 1,
            // sc-19502 — THE FIX for this lane. `req.steps` was never read here at all: a
            // `steps: 30` request was accepted, the knob did nothing, and the baked 8-step stage-1
            // schedule rendered anyway, while the candle lane refused the same request outright.
            // A control that is binding on one backend and silently inert on the other is the
            // sc-11993 silent-coercion class, and the silent side is the worse one.
            //
            // The distilled schedule genuinely cannot be resampled to an arbitrary count without
            // going out-of-distribution, so the honest resolution is an explicit refusal on BOTH
            // lanes — not a knob that pretends to work here. Same derived constant, same shared
            // floor, same message as candle.
            supported_steps: StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS]),
            mac_only: true,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            // sc-18816: every generate builds/evaluates/drops Gemma before materializing the AvDiT,
            // then drops the AvDiT before VAE/audio decode. This is physical default behavior, not a
            // selectable Sequential control and not an evidence-composition edge.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// Stable identity and capability surface of the LTX-2.5 MLX route.  The ordinary 2.3
/// conditioning paths are shared by the two execution shells; the advanced axes below are open
/// only because the split route binds the duration head, temporal upsampler, and alternate decoder
/// into its request execution path.
pub fn descriptor_25() -> ModelDescriptor {
    let mut out = descriptor();
    out.id = MODEL_25_ID;
    out.capabilities.supports_lora = false;
    out.capabilities.supports_lokr = false;
    out.capabilities.supports_prompt_enhancement = false;
    out.capabilities.supports_auto_duration = true;
    out.capabilities.supports_generated_keyframes = true;
    out.capabilities.max_temporal_upsample_rounds =
        mlx_gen::gen_core::ltx_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS;
    out.capabilities.supports_diffusion_decoder = true;
    out
}

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

/// The LTX-2.5-only components which turn the shared 2.3 A/V shell into the declared 2.5
/// provider.  Keeping the choice here means all three advanced axes flow through the ordinary
/// `Generator::generate` route instead of being descriptor-only capability claims.
enum LtxExecution {
    Ltx23,
    Ltx25(Box<Ltx25Execution>),
}

struct Ltx25Execution {
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

/// Apply the shared frame-count resolver to the actual provider request.  The predictor is injected
/// solely so the provider-path test can prove that its result changes the allocated-frame plan;
/// production supplies [`DurationHead::predict_seconds`] above.
fn apply_predicted_frames(
    req: &GenerationRequest,
    fps: f32,
    predict_seconds: &mut dyn FnMut() -> Result<f32>,
) -> Result<GenerationRequest> {
    let mut core_predict = || predict_seconds().map_err(mlx_gen::gen_core::Error::from);
    let frames = mlx_gen::gen_core::duration_head::resolve_request_num_frames(
        req.frames,
        req.auto_duration,
        fps,
        TEMPORAL_SCALE,
        &mut core_predict,
    )?;
    let mut resolved = req.clone();
    if let Some(frames) = frames {
        resolved.frames = Some(frames);
    }
    Ok(resolved)
}

/// Map the two declared LTX-2.5 DFR controls onto the concrete latent pipeline shape.  A positive
/// generated-keyframe count inserts exactly that many evenly spaced slots; a temporal request
/// additionally pads to the shared DFR segment canvas so each temporal round has its real tile
/// anchors.  The same plan drives noise allocation and the actual `generate_dfr_av_latents` call.
fn plan_dfr_request(req: &GenerationRequest) -> Result<Option<DfrPlan>> {
    let requested_frames = req.frames.unwrap_or(1);
    let generated_count = req.num_generated_keyframes.unwrap_or(0);
    let temporal_upsample_rounds = req.temporal_upsample_rounds.unwrap_or(0);
    if generated_count == 0 && temporal_upsample_rounds == 0 {
        return Ok(None);
    }

    let (canvas_frames, automatic_positions) = if temporal_upsample_rounds > 0 {
        let (canvas, _, positions) = mlx_gen::gen_core::ltx_dfr::resolve_canvas(
            i64::from(requested_frames),
            i64::from(TEMPORAL_SCALE),
        )?;
        (canvas as u32, positions)
    } else {
        (requested_frames, Vec::new())
    };
    let keyframe_positions = if generated_count > 0 {
        mlx_gen::gen_core::ltx_dfr::evenly_spaced_keyframe_positions(
            generated_count,
            i64::from(canvas_frames),
        )
    } else {
        automatic_positions
    };
    if keyframe_positions.is_empty() {
        return Err(Error::Msg(
            "ltx_2_5: DFR needs at least two requested frames to place a generated-keyframe slot"
                .into(),
        ));
    }
    Ok(Some(DfrPlan {
        canvas_frames,
        requested_frames,
        keyframe_positions,
        temporal_upsample_rounds,
    }))
}

/// Choose the provider's real DFR denoise branch or its established two-stage branch.  Progress is
/// passed into the selected callback so the branch decision stays observable in a synthetic test
/// without constructing multi-gigabyte Gemma/DiT fixtures.
fn dispatch_dfr<T>(
    plan: Option<&DfrPlan>,
    on_step: &mut dyn FnMut(usize),
    dfr: impl FnOnce(&DfrPlan, &mut dyn FnMut(usize)) -> Result<T>,
    plain: impl FnOnce(&mut dyn FnMut(usize)) -> Result<T>,
) -> Result<T> {
    match plan {
        Some(plan) => dfr(plan, on_step),
        None => plain(on_step),
    }
}

/// Locate the Gemma-3-12B text-encoder snapshot from the **required** `LoadSpec::text_encoder` slot
/// (sc-8827; required as of sc-13664). The caller provisions the TE path through the spec — there is no
/// env side-channel and no on-disk HF-cache scan any more (epic 13657): an absent slot is a load-time,
/// actionable error, and a set-but-nonexistent path errors up front. `pub(crate)` so the trainer
/// (sc-3047) resolves the TE snapshot exactly as inference does (it too must be handed a
/// `LoadSpec::text_encoder`).
pub(crate) fn resolve_gemma_dir(
    override_src: Option<&WeightsSource>,
) -> Result<std::path::PathBuf> {
    let Some(WeightsSource::Dir(p) | WeightsSource::File(p)) = override_src else {
        return Err(Error::Msg(
            "ltx_2_3 requires the Gemma-3-12B text encoder — set LoadSpec::text_encoder to the \
             gemma-3-12b-it snapshot dir (e.g. LoadSpec { text_encoder: \
             Some(WeightsSource::Dir(...)), .. }). It is no longer auto-discovered from an \
             environment variable or the HF cache."
                .into(),
        ));
    };
    if !p.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_3: LoadSpec text_encoder path does not exist: {}",
            p.display()
        )));
    }
    Ok(p.clone())
}

/// Resolve the optional **uncensored** 4-bit Gemma enhancer snapshot dir (sc-2845
/// `--use-uncensored-enhancer`, reference `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`) from the
/// caller-staged `uncensored_enhancer` [`LoadSpec::components`] entry (sc-13664 — no
/// `$LTX_UNCENSORED_GEMMA_DIR`, no HF-cache scan). `None` when the caller did not provision it (the
/// feature is opt-in per request); a set-but-nonexistent path errors up front at load. A standalone
/// mlx_lm checkpoint (`model.` key prefix).
fn resolve_uncensored_enhancer(
    component: Option<&WeightsSource>,
) -> Result<Option<std::path::PathBuf>> {
    let Some(src) = component else {
        return Ok(None);
    };
    let (WeightsSource::Dir(p) | WeightsSource::File(p)) = src;
    if !p.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_3: the 'uncensored_enhancer' component path does not exist: {}",
            p.display()
        )));
    }
    Ok(Some(p.clone()))
}

/// Read the Gemma snapshot's `config.json` top-level `quantization` block — the reference TE-quant
/// trigger (`utils.apply_quantization`). `None` for the default `…-bf16` snapshot (no block). Only the
/// `affine` mode is consumed (the one `quantized_matmul`/`dequantize` implement); a non-affine mode is
/// a hard error rather than a silent mis-decode.
pub(crate) fn resolve_gemma_quant(gemma_dir: &std::path::Path) -> Result<Option<GemmaQuant>> {
    let path = gemma_dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("ltx_2_3: parse gemma config.json: {e}")))?;
    let Some(q) = v.get("quantization") else {
        return Ok(None);
    };
    if let Some(mode) = q.get("mode").and_then(|m| m.as_str()) {
        if mode != "affine" {
            return Err(Error::Msg(format!(
                "ltx_2_3: gemma quantization mode {mode:?} is not supported (only affine)"
            )));
        }
    }
    match (
        q.get("group_size").and_then(|x| x.as_i64()),
        q.get("bits").and_then(|x| x.as_i64()),
    ) {
        (Some(g), Some(b)) => Ok(Some(GemmaQuant {
            group: g as i32,
            bits: b as i32,
        })),
        _ => Ok(None),
    }
}

/// Load the model from a split-weight snapshot directory (the `ltx_2_3_base*` tree). Reads
/// `embedded_config.json`, locates the Gemma TE separately, and assembles every component.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ltx_2_3: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    // Named-component contract (sc-13658/sc-13664): LTX-2.3 reads only the optional `uncensored_enhancer`
    // component; its main Gemma TE rides the typed `text_encoder` slot, not this map. Reject any
    // unrecognized component key up front, then stage the (existence-checked) enhancer path if provided.
    reject_unknown_components(spec, &["uncensored_enhancer"], MODEL_ID)?;
    let uncensored_enhancer =
        resolve_uncensored_enhancer(spec.components.get("uncensored_enhancer"))?;
    // Layout gate (sc-18757): which LTX generation this tree holds is decided by its declared
    // `model_version` — NOT by which files are present and NOT by their names. This engine
    // implements the LTX-2.3 all-in-one component set; an LTX-2.5 split bundle carries a different
    // DiT, a Gemma 4 text encoder and a diffusion VAE, so it is refused **here**, by version, with
    // the version named. Without this gate a 2.5 tree fell through to the per-file existence check
    // below and died as "missing transformer.safetensors" — a filename-shaped error that points at
    // the wrong problem. `crate::bundle` resolves the split layout for the engine that implements it.
    let declared_version = crate::bundle::declared_model_version(root)?;
    if layout_for_declared_version(declared_version.as_deref()) == LtxCheckpointLayout::Split {
        return Err(Error::Msg(format!(
            "ltx_2_3: {} declares model_version {:?}, which ships as a split-component bundle \
             (per-component transformer / text encoder / video VAE / audio VAE / duration head / \
             latent upsamplers, each with its own config). The ltx_2_3 engine loads the all-in-one \
             LTX-2.3 layout only.",
            root.display(),
            declared_version.as_deref().unwrap_or_default(),
        )));
    }
    // Quantization geometry rides on the checkpoint's `split_model.json` (sc-2686): the transformer is
    // shipped selectively quantized (Q4 for `base_q4`, Q8 for `base_q8`), bits/group from the
    // manifest — never hardcoded. The per-Linear `.scales` predicate (in `transformer.rs`) then picks
    // which Linears are quantized, matching `generate_av.py`'s `_should_quantize`.
    let split = SplitModel::from_model_dir(root)?;
    // `spec.quantize`, when set, only *asserts* the expected level. LTX can't re-quantize a dense
    // checkpoint (there is no dense LTX transformer — it ships pre-packed from the reference
    // `convert.py`, which casts f32→bf16 before quantizing), so a mismatch is a hard load error
    // rather than a silent re-quant.
    if let Some(q) = spec.quantize {
        if !split.quantized {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} but {} carries no split_model.json quant manifest — \
                 LTX quant is checkpoint-driven; point at a quantized checkpoint (e.g. ltx_2_3_base_q4)",
                root.display()
            )));
        }
        if q.bits() != split.bits {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} (bits {}) disagrees with the checkpoint's \
                 split_model.json (bits {})",
                q.bits(),
                split.bits
            )));
        }
    }
    // Precision selection. `Bf16` (the [`LoadSpec`] default) → the reference's **native** bf16
    // activations × quantized weights — the production-speed path; `Fp32` → f32 activations ×
    // quantized weights — the quality target. Both are bit-exact to their reference golden (sc-2842;
    // the distilled stage-1 sampler is chaos-sensitive, so each per-forward is bit-exact). The latent
    // statistics (the upsampler's un-/re-normalize) follow the path dtype so the whole denoise stays
    // in that precision; the VAE decode stays f32 in both — a post-sampling quality island.
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (
            Precision::quant_bf16(split.bits, split.group),
            Dtype::Bfloat16,
        ),
        LoadPrecision::Fp32 => (
            Precision::quant_f32(split.bits, split.group),
            Dtype::Float32,
        ),
    };

    let config = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;
    let audio_vae_config = AudioVaeConfig::from_model_dir(root)?;
    let vocoder_config = VocoderConfig::from_model_dir(root)?;

    // Resolve the Gemma TE location + its quant (cheap — a path lookup + one `config.json` read). The
    // ~24 GB Gemma backbone and the `connector`/`transformer` weights are deliberately NOT loaded here:
    // sc-10976 defers them to `build_text_encoder`/`build_transformer`, built per-generate so the TE is
    // dropped (+ `clear_cache()`) before the DiT materializes (mirroring Wan). Fail-fast on a missing
    // component is preserved by a cheap existence check, so a broken tree still errors at load rather
    // than at first generate.
    let gemma_dir = resolve_gemma_dir(spec.text_encoder.as_ref())?;
    // Selectively quantize the Gemma backbone iff the snapshot's `config.json` says so (the reference
    // `apply_quantization` path; sc-2686). The default `…-bf16` snapshot ⇒ `None` ⇒ dense bf16 TE.
    let gemma_quant = resolve_gemma_quant(&gemma_dir)?;
    for f in ["connector.safetensors", "transformer.safetensors"] {
        if !root.join(f).exists() {
            return Err(Error::Msg(format!(
                "ltx_2_3: missing {f} in the model dir {}",
                root.display()
            )));
        }
    }
    // SC-19109: bind the canonical dense-bf16 Gemma route to the same resolved split manifest,
    // component paths and additive overlays the engine below will materialize. Quantized Gemma is a
    // supported generator route but has no matching evidence identity yet, so it deliberately loads
    // without a memory contract rather than borrowing the canonical route's measurements.
    let loaded_memory =
        crate::memory_strategy::contract_for_loaded(spec, &split, &gemma_dir, gemma_quant)?;
    let (memory_strategy, memory_tier, memory_overlay) = match loaded_memory {
        Some((contract, tier, overlay)) => (Some(contract), Some(tier), overlay),
        None => (None, None, None),
    };

    // The small components (each ≪ the TE / DiT) stay resident. The VAE *decoder* + audio VAE + vocoder
    // load here; the VAE *encoder* is still lazy on first I2V encode (F-048).
    let vae_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let audio_vae_w = Weights::from_file(root.join("audio_vae.safetensors"))?;
    let vocoder_w = Weights::from_file(root.join("vocoder.safetensors"))?;

    // Loaded through the path constructor, so a stamped checkpoint's declared config is
    // cross-checked against the structure the weights imply instead of the rank silently winning.
    let upsampler = LatentUpsampler::from_checkpoint(root.join("upsampler.safetensors"))?;
    // The VAE encoder serves I2V conditioning (sc-2685) but pure-T2V+A requests never touch it, so it
    // is loaded **lazily** on first encode — `vae_encoder.safetensors` (hundreds of MB) stays off the
    // resident set for T2V (F-048). `generate_av.py` supports I2V+Audio (image-conditioned video).
    let vae = LtxVideoVae::from_weights_lazy_encoder(
        &vae_w,
        root.join("vae_encoder.safetensors"),
        &vae_config,
    )?;
    // The audio VAE decoder + vocoder run f32 (post-sampling quality islands, gated bit-exact).
    let audio_decoder = AudioDecoder::from_weights(&audio_vae_w, &audio_vae_config)?;
    let vocoder = LtxVocoder::from_weights(&vocoder_w, &vocoder_config)?;
    let audio_sample_rate = vocoder_config.final_sample_rate() as u32;
    // The VAE `per_channel_statistics` double as the upsampler's latent norm, at the path dtype.
    let latent_mean = to_dtype(vae_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(vae_w.require("per_channel_statistics.std")?, stat_dt)?;
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(Box::new(Ltx {
        descriptor: descriptor(),
        memory_strategy,
        memory_tier,
        memory_overlay,
        tokenizer: Tokenizer::Gemma3(tokenizer),
        transformer_path: root.join("transformer.safetensors"),
        config,
        dit_prec,
        adapters: spec.adapters.clone(),
        text_assets: TextAssets::Gemma3 {
            dir: gemma_dir,
            quant: gemma_quant,
        },
        execution: LtxExecution::Ltx23,
        uncensored_enhancer,
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate,
        stat_dt,
    }))
}

/// Lazy split-bundle LTX-2.5 provider.  Resolving the metadata/configuration is deliberately part
/// of `load`; materialising multi-gigabyte tensors remains request-scoped, matching the staged LTX
/// route and keeping ordinary catalog construction weights-free.
pub struct Ltx25 {
    descriptor: ModelDescriptor,
    spec: LoadSpec,
}

/// Resolve the LTX-2.5 split layout through the ordinary provider registration. The actual tensor
/// assembly is request-scoped and invoked by [`Generator::generate`], not hidden behind a filename
/// convention or a separate loader entry point.
pub fn load_25(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let known = crate::bundle::split_component_ids();
    reject_unknown_components(spec, &known, MODEL_25_ID)?;
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "ltx_2_5: LoRA/LoKr adapters are not part of this provider route".into(),
        ));
    }
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    if bundle.layout() != LtxCheckpointLayout::Split {
        return Err(Error::Msg(format!(
            "ltx_2_5: expected an LTX-2.5 split-component bundle, got {}",
            bundle.layout().id()
        )));
    }
    // These are intentional load-path checks, not descriptor-only declarations.  In particular,
    // `from_bundle` selects the transformer's own 2.5 config and the Gemma check ties that DiT to
    // the packed encoder before a generation can allocate noise.  The duration and temporal
    // components are required here because the advertised request controls execute through them.
    let config = LtxConfig::from_bundle(&bundle)?;
    if !config.use_keyframes_abs_pos_embedding {
        return Err(Error::Msg(
            "ltx_2_5: transformer lacks use_keyframes_abs_pos_embedding, so generated-keyframe \
             slots must not be advertised or executed"
                .into(),
        ));
    }
    crate::bundle::assert_gemma_version(&bundle)?;
    for component in [
        LtxComponent::TextEncoder,
        LtxComponent::AudioVae,
        LtxComponent::DurationHead,
        LtxComponent::SpatialUpsampler,
        LtxComponent::TemporalUpsampler,
    ] {
        bundle.require(component)?;
    }
    bundle.require(ltx25_video_component(spec))?;
    Ok(Box::new(Ltx25 {
        descriptor: descriptor_25(),
        spec: spec.clone(),
    }))
}

/// Staging the diffusion-video-VAE component is the alternate-decoder selection contract.  A
/// bundle may contain both VAE variants for catalog discovery; only an explicit component choice
/// switches the provider away from its ordinary convolutional decoder.
fn ltx25_video_component(spec: &LoadSpec) -> LtxComponent {
    match ltx25_decoder_selection(spec) {
        Ltx25DecoderSelection::Conv => LtxComponent::ConvVideoVae,
        Ltx25DecoderSelection::DiffusionBudgeted(_) => LtxComponent::DiffusionVideoVae,
    }
}

fn ltx25_decoder_selection(spec: &LoadSpec) -> Ltx25DecoderSelection {
    if spec
        .components
        .contains_key(LtxComponent::DiffusionVideoVae.id())
    {
        Ltx25DecoderSelection::DiffusionBudgeted(DEFAULT_DIFFVAE_MODE)
    } else {
        Ltx25DecoderSelection::Conv
    }
}

/// Invoke the decoder's declared budget/mode route.  This tiny dispatch is intentionally shared
/// by the provider and its synthetic test, so replacing a DiffVAE decode with the conv path (or
/// dropping the budgeted call) cannot leave its selected mode unobserved.
fn decode_diffvae_budgeted<T>(
    mode: DiffVaeMode,
    decode: impl FnOnce(DiffVaeMode) -> Result<T>,
) -> Result<T> {
    decode(mode)
}

/// Converted LTX-2.5 tiers split each VAE encoder into its own file, while raw split bundles keep
/// the encoder beside its decoder. The bundle resolver owns component discovery; after it has
/// selected a component, prefer that component's documented tier encoder half when present.
fn ltx25_encoder_path(
    root: &std::path::Path,
    component: LtxComponent,
    fallback: &std::path::Path,
) -> PathBuf {
    let half = match component {
        LtxComponent::ConvVideoVae => root.join("vae_encoder.safetensors"),
        LtxComponent::DiffusionVideoVae => root.join("diffusion_vae_encoder.safetensors"),
        _ => return fallback.to_path_buf(),
    };
    if half.is_file() {
        half
    } else {
        fallback.to_path_buf()
    }
}

fn build_ltx25(spec: &LoadSpec) -> Result<Ltx> {
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    let config = LtxConfig::from_bundle(&bundle)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "ltx_2_5: split bundles must be loaded from their component directory".into(),
            ))
        }
    };
    let split = SplitModel::from_model_dir(root)?;
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (
            Precision::quant_bf16(split.bits, split.group),
            Dtype::Bfloat16,
        ),
        LoadPrecision::Fp32 => (
            Precision::quant_f32(split.bits, split.group),
            Dtype::Float32,
        ),
    };
    let decoder_selection = ltx25_decoder_selection(spec);
    let video_component = ltx25_video_component(spec);
    let video_path = bundle.require(video_component)?.path().to_path_buf();
    let encoder_path = ltx25_encoder_path(root, video_component, &video_path);
    let audio = bundle.require(LtxComponent::AudioVae)?.path().to_path_buf();
    let connector = bundle
        .require(LtxComponent::Transformer)?
        .path()
        .with_file_name("connector.safetensors");
    // The converter keeps the connector beside the transformer.  A separately staged connector is
    // not a public component because the transformer metadata remains its identity authority.
    if !connector.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_5: missing connector.safetensors beside transformer {}",
            bundle.require(LtxComponent::Transformer)?.path().display()
        )));
    }
    let video_w = Weights::from_file(&video_path)?;
    let audio_w = Weights::from_file(&audio)?;
    let vae_cfg = LtxVaeConfig::from_bundle(&bundle, video_component)?;
    let audio_cfg = AudioVaeConfig::from_bundle(&bundle)?;
    let vocoder_cfg = VocoderConfig::from_bundle(&bundle)?;
    let (vae, decoder) = match (video_component, decoder_selection) {
        (LtxComponent::ConvVideoVae, Ltx25DecoderSelection::Conv) => (
            LtxVideoVae::from_weights_lazy_encoder(&video_w, encoder_path, &vae_cfg)?,
            Ltx25Decoder::Conv,
        ),
        (LtxComponent::DiffusionVideoVae, Ltx25DecoderSelection::DiffusionBudgeted(mode)) => {
            let diff_cfg = NaDiffusionDecoderConfig::from_embedded_vae(
                bundle.require(LtxComponent::DiffusionVideoVae)?.config()?,
            )?;
            let quant = split.quantized.then_some(DiffVaeQuant {
                bits: split.bits,
                group: split.group,
            });
            let decoder = NaDiffusionDecoder::from_weights(&video_w, &diff_cfg, quant)?;
            // The diffusion VAE carries the same causal encoder used for image conditioning, but
            // no convolutional decoder.  Keeping it encoder-only prevents a staged DiffVAE route
            // from materialising and then accidentally decoding through the conv sibling.
            (
                LtxVideoVae::encoder_only_lazy(encoder_path, &vae_cfg)?,
                Ltx25Decoder::Diffusion {
                    decoder: Box::new(decoder),
                    mode,
                },
            )
        }
        _ => unreachable!("the staged LTX-2.5 decoder selection and component must agree"),
    };
    let audio_decoder = AudioDecoder::from_weights(&audio_w, &audio_cfg)?;
    let vocoder = LtxVocoder::from_weights(&audio_w, &vocoder_cfg)?;
    let spatial = bundle.require(LtxComponent::SpatialUpsampler)?.path();
    let upsampler = LatentUpsampler::from_checkpoint(spatial)?;
    let latent_mean = to_dtype(video_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(video_w.require("per_channel_statistics.std")?, stat_dt)?;
    let te_path = bundle
        .require(LtxComponent::TextEncoder)?
        .path()
        .to_path_buf();
    let tokenizer = Ltx25Tokenizer::from_packed_te_file(&te_path)?;
    let duration_w = Weights::from_file(bundle.require(LtxComponent::DurationHead)?.path())?;
    let duration_head = DurationHead::from_weights(&duration_w)?;
    let temporal_upsampler =
        LatentUpsampler::from_checkpoint(bundle.require(LtxComponent::TemporalUpsampler)?.path())?;

    Ok(Ltx {
        descriptor: descriptor_25(),
        memory_strategy: None,
        memory_tier: None,
        memory_overlay: None,
        tokenizer: Tokenizer::Gemma4(tokenizer),
        transformer_path: bundle
            .require(LtxComponent::Transformer)?
            .path()
            .to_path_buf(),
        config,
        dit_prec,
        adapters: Vec::new(),
        text_assets: TextAssets::Gemma4 {
            bundle,
            connector,
            offload_policy: spec.offload_policy,
        },
        execution: LtxExecution::Ltx25(Box::new(Ltx25Execution {
            duration_head,
            temporal_upsampler,
            decoder,
        })),
        uncensored_enhancer: None,
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate: vocoder_cfg.final_sample_rate() as u32,
        stat_dt,
    })
}

impl Ltx {
    /// Build the AudioVideo Gemma-3-12B text encoder from the resolved `gemma_dir` + the snapshot's
    /// `connector.safetensors` (sc-10976). Called per-generate and dropped (+ `clear_cache()`) before
    /// [`build_transformer`](Self::build_transformer) — mirroring Wan's `encode_text_staged` — so the
    /// ~24 GB TE and the DiT never co-reside. bf16 activations (the reference TE dtype); the backbone is
    /// selectively quantized iff `gemma_quant` is set (`None` ⇒ dense bf16, the default `…-bf16`
    /// snapshot). Identical construction to the pre-sc-10976 `load()`, only deferred.
    fn build_text_encoder(&self) -> Result<StagedTextEncoder> {
        match &self.text_assets {
            TextAssets::Gemma3 { dir, quant } => {
                let gemma_w = Weights::from_dir(dir)?;
                let connector_w = Weights::from_file(
                    self.transformer_path
                        .parent()
                        .expect("transformer path has a parent")
                        .join("connector.safetensors"),
                )?;
                Ok(StagedTextEncoder::Gemma3(Box::new(
                    LtxTextEncoder::from_weights_av(
                        &gemma_w,
                        &connector_w,
                        GemmaConfig::gemma_3_12b(),
                        *quant,
                        &self.config,
                        self.dit_prec.with_compute_dtype(Dtype::Bfloat16),
                    )?,
                )))
            }
            TextAssets::Gemma4 {
                bundle,
                connector,
                offload_policy,
            } => {
                let connector_w = Weights::from_file(connector)?;
                Ok(StagedTextEncoder::Gemma4(Box::new(
                    Ltx25TextEncoder::from_bundle_av(
                        bundle,
                        &connector_w,
                        &self.config,
                        self.dit_prec.with_compute_dtype(Dtype::Bfloat16),
                        *offload_policy,
                    )?,
                )))
            }
        }
    }

    /// Build the AvDiT from the snapshot's `transformer.safetensors` at the load-time precision,
    /// applying any LoRA/LoKr adapters (sc-2687/sc-2393). Called per-generate AFTER the text encoder is
    /// freed (sc-10976), so the 24 GB TE and the DiT never co-reside. LoRA is a forward-time residual
    /// over the (quantized/dense) base; `pass_scales` carries one strength per distilled denoise pass.
    fn build_transformer(&self) -> Result<AvDiT> {
        let transformer_w = Weights::from_file(&self.transformer_path)?;
        let mut transformer = AvDiT::from_weights(&transformer_w, &self.config, self.dit_prec)?;
        if !self.adapters.is_empty() {
            crate::adapters::apply_ltx_adapters(
                &mut transformer,
                &self.adapters,
                crate::pipeline::NUM_DENOISE_PASSES,
            )?;
        }
        Ok(transformer)
    }

    /// Stage the text phase (sc-10976): build the Gemma TE, run the optional prompt enhancer + the
    /// video/audio encode inside its scope, `eval` so the encode completes, then let the TE drop at
    /// end of scope. Returns the (optional) enhanced prompt + the `(video_ctx, audio_ctx)` embeddings.
    /// The caller `clear_cache()`s and proceeds to the DiT, which now loads into the freed footprint.
    fn stage_text_phase(&self, req: &GenerationRequest) -> Result<(Option<String>, Array, Array)> {
        let te = self.build_text_encoder()?;
        // Prompt enhancement (sc-2845) reuses the just-built Gemma backbone (the censored path); the
        // uncensored path loads its own. Running it here keeps BOTH TE uses inside one staged load.
        let enhanced = match &te {
            StagedTextEncoder::Gemma3(te) => self.maybe_enhance(te, req),
            StagedTextEncoder::Gemma4(_) => None,
        };
        let prompt = enhanced.as_deref().unwrap_or(req.prompt.as_str());
        let (ids, mask) = self.tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
        let (video_ctx, audio_ctx) = te.encode_av(&ids, &mask)?;
        // Force the encode to complete before `te` (the ~24 GB Gemma) drops at end of scope — otherwise
        // the lazy graph would keep the TE weights alive into the DiT phase, defeating the staging.
        mlx_rs::transforms::eval([&video_ctx, &audio_ctx])?;
        Ok((enhanced, video_ctx, audio_ctx))
    }

    fn ltx25_execution(&self) -> Result<&Ltx25Execution> {
        match &self.execution {
            LtxExecution::Ltx25(execution) => Ok(execution),
            LtxExecution::Ltx23 => Err(Error::Unsupported(
                "ltx_2_3: the LTX-2.5 duration/DFR/DiffVAE execution components are unavailable"
                    .into(),
            )),
        }
    }

    /// Apply an opt-in automatic frame prediction **after** the ordinary staged text encode, where the
    /// real DurationHead receives the connector outputs it was trained on.  The returned request
    /// carries the concrete frame count, so noise allocation and every downstream plan see the
    /// prediction rather than the original empty `frames` field.
    fn apply_auto_frames(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
    ) -> Result<GenerationRequest> {
        if req.auto_duration.is_none() {
            return Ok(req.clone());
        }
        let duration_head = &self.ltx25_execution()?.duration_head;
        apply_predicted_frames(req, req.fps.unwrap_or(24) as f32, &mut || {
            duration_head.predict_seconds(Some(video_ctx), Some(audio_ctx))
        })
    }

    fn dfr_plan(&self, req: &GenerationRequest) -> Result<Option<DfrPlan>> {
        match &self.execution {
            LtxExecution::Ltx23 => Ok(None),
            LtxExecution::Ltx25(_) => plan_dfr_request(req),
        }
    }

    /// Latent dims `(frames, stage1_h, stage1_w, stage2_h, stage2_w)` for a request.
    pub(crate) fn latent_dims(req: &GenerationRequest) -> (usize, usize, usize, usize, usize) {
        // Precondition: `validate_request`'s `SIZE_MULTIPLE`-divisibility check runs before any generate, so the
        // integer divisions below (`h/2/SPATIAL_SCALE`, `h/SPATIAL_SCALE`) are exact and lose no rows.
        // Make that implicit dependency explicit so a future direct caller that skips validation trips
        // here in debug/test instead of silently truncating the latent grid (Info/L-E).
        debug_assert!(
            req.height.is_multiple_of(SIZE_MULTIPLE) && req.width.is_multiple_of(SIZE_MULTIPLE),
            "ltx latent_dims: {}×{} is not {SIZE_MULTIPLE}-aligned — validate_request must run first",
            req.width,
            req.height
        );
        let frames = req.frames.unwrap_or(1).max(1);
        let latent_frames = 1 + (frames as usize - 1) / TEMPORAL_SCALE as usize;
        let (h, w) = (req.height, req.width);
        (
            latent_frames,
            (h / 2 / SPATIAL_SCALE) as usize,
            (w / 2 / SPATIAL_SCALE) as usize,
            (h / SPATIAL_SCALE) as usize,
            (w / SPATIAL_SCALE) as usize,
        )
    }

    /// Audio latent-frame count for the request (`compute_audio_frames(num_frames, fps)`).
    pub(crate) fn audio_frames(req: &GenerationRequest) -> usize {
        compute_audio_frames(
            req.frames.unwrap_or(1).max(1) as usize,
            req.fps.unwrap_or(24) as f64,
        )
    }

    /// `--no-audio` toggle: `req.video_mode == "no_audio"` runs the full A/V denoise but skips the
    /// audio decode + returns `audio: None` (the reference `--no-audio`).
    fn no_audio(req: &GenerationRequest) -> bool {
        matches!(
            req.video_mode.as_deref(),
            Some("no_audio") | Some("video_only")
        )
    }

    /// Decode through the component selected at provider load.  The DiffVAE arm deliberately
    /// calls its memory-budgeted decoder with the declared mode rather than borrowing the conv
    /// VAE's tiling selector: these are different planners over different intermediate shapes.
    fn decode_video(&self, req: &GenerationRequest, latents: &Array, seed: u64) -> Result<Array> {
        let decode_conv = || {
            let selected_tiling = crate::memory_strategy::decode_tiling(req)?;
            decode_to_frames_with_tiling(&self.vae, latents, &req.cancel, selected_tiling.as_ref())
        };
        match &self.execution {
            LtxExecution::Ltx23 => decode_conv(),
            LtxExecution::Ltx25(execution) => match &execution.decoder {
                Ltx25Decoder::Conv => decode_conv(),
                Ltx25Decoder::Diffusion { decoder, mode } => {
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let shape = latents.shape();
                    let noise_shape = decoder.config().noise_shape(shape[2], shape[3], shape[4]);
                    let key = random::key(seed.wrapping_add(4))?;
                    let noise = random::normal::<f32>(
                        &[
                            shape[0],
                            decoder.config().out_channels,
                            noise_shape[0],
                            noise_shape[1],
                            noise_shape[2],
                        ],
                        None,
                        None,
                        Some(&key),
                    )?;
                    let pixels = decode_diffvae_budgeted(*mode, |mode| {
                        decoder.decode_budgeted(latents, &noise, mode)
                    })?;
                    to_uint8_frames(&pixels)
                }
            },
        }
    }

    /// The A/V path from **staged** text embeddings + injected stage noise (the deterministic seam
    /// `generate` calls with RNG-drawn noise). The Gemma encode has already run and its ~24 GB weights
    /// dropped (sc-10976 — see [`stage_text_phase`](Self::stage_text_phase)); this resolves any optional
    /// I2V conditioning image (VAE-encoded at both stage resolutions — the video is image-conditioned,
    /// the audio stays pure-noise, matching `generate_av.py`'s I2V+Audio), then defers to
    /// [`generate_av_from_embeddings`](Self::generate_av_from_embeddings).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_noise(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // F-051: honor a cancel before the (unbounded) per-keyframe/per-clip VAE conditioning encodes,
        // ahead of the first denoise-loop check. (The Gemma TE encode already ran + dropped in the
        // staged text phase.)
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Materialize every conditioned latent inside one encoder-residency scope. The VAE encoder
        // is request-only: once these arrays are forced, no denoise or decode operation needs its
        // weights. Release it on success, cancellation, and error before the 48-block AvDiT loads.
        let conditioned = (|| {
            let keyframes = self.build_keyframes(req)?;
            let clips = self.build_clips(req)?;
            Ok::<_, Error>((keyframes, clips))
        })();
        if self.vae.release_encoder() {
            mlx_rs::memory::clear_cache();
        }
        let (kf_owned, clip_owned) = conditioned?;
        // Replace-latent conditioning: VAE-encode each keyframe at both stage resolutions (half/full).
        // I2V = a single `Reference` at frame 0; first_last_frame / multi-keyframe = `Keyframe`s.
        let keyframes: Vec<StageKeyframe> = kf_owned
            .iter()
            .map(|(s1, s2, idx, strength)| StageKeyframe {
                stage1: s1,
                stage2: s2,
                frame_idx: *idx,
                strength: *strength,
            })
            .collect();
        // In-context clips (extend_clip / video_bridge) — VAE-encoded at stage-1 resolution, appended
        // as IC-LoRA conditioning tokens in stage 1 only.
        let clips: Vec<StageClip> = clip_owned
            .iter()
            .map(|(s1, idx, strength)| StageClip {
                stage1: s1,
                frame_idx: *idx,
                strength: *strength,
            })
            .collect();
        self.generate_av_from_embeddings(
            req,
            video_ctx,
            audio_ctx,
            video_s1,
            video_s2,
            audio_s1,
            audio_s2,
            &keyframes,
            &clips,
            on_progress,
        )
    }

    /// The A/V path from **injected** text embeddings + noise — the pipeline-only seam (no Gemma), so
    /// the parity test can gate the joint 2-stage pipeline + video/audio decode against the reference
    /// conditioning. `video_ctx` `(1, ctx, 4096)`, `audio_ctx` `(1, ctx, 2048)`; video noise
    /// `(1,128,F,h,w)` per stage; audio noise `(1,8,T,16)` per stage (`T = audio_frames`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_av_from_embeddings(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        video_keyframes: &[StageKeyframe],
        video_clips: &[StageClip],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let dfr_plan = self.dfr_plan(req)?;
        // A temporal DFR request pads the canvas before the request's public trim point, so its
        // injected noises carry the authoritative latent geometry.  Ordinary LTX keeps the
        // historical request-derived dimensions exactly.
        let (lf, h1, w1, h2, w2) = match dfr_plan.as_ref() {
            Some(_) => (
                video_s1.shape()[2] as usize,
                video_s1.shape()[3] as usize,
                video_s1.shape()[4] as usize,
                video_s2.shape()[3] as usize,
                video_s2.shape()[4] as usize,
            ),
            None => Self::latent_dims(req),
        };
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);
        let audio_frames = match dfr_plan.as_ref() {
            Some(_) => audio_s1.shape()[2] as usize,
            None => Self::audio_frames(req),
        };
        let audio_pos = create_audio_position_grid(1, audio_frames);

        // Curated unified solver (epic 7114, sc-7122): a curated solver name routes the joint two-stream
        // T2V+A denoise through `generate_av_latents`' `denoise_av_curated` branch (LTX keeps its baked
        // distilled σ schedule — sampler-only, no scheduler). `validate` already rejected any name not in
        // the advertised menu; an unset sampler → the native distilled Euler (the byte-exact default).
        let seed = req.seed.unwrap_or_else(default_seed);
        let curated = req
            .sampler
            .as_deref()
            .filter(|s| mlx_gen::Solver::from_name(s).is_some());
        // The image/keyframe (I2V) + in-context-clip paths condition with per-token `σ·mask` and a
        // post-step `apply_denoise_mask` blend, which have no single-eval curated-sampler hook — they
        // stay on the native distilled Euler. Reject a curated request there rather than silently ignore.
        if curated.is_some() && (!video_clips.is_empty() || !video_keyframes.is_empty()) {
            return Err(Error::Msg(
                "ltx: curated samplers apply to text-to-video only; the image/keyframe (I2V) and \
                 in-context-clip paths use per-token-σ conditioning that stays on the native distilled \
                 Euler"
                    .into(),
            ));
        }
        if dfr_plan.is_some() && (!video_clips.is_empty() || curated.is_some()) {
            return Err(Error::Msg(
                "ltx_2_5: DFR generated-keyframe and temporal requests do not accept \
                 in-context clips or curated samplers"
                    .into(),
            ));
        }

        // F-050 (sc-11133): fold the two-stage (8 + 3 step) distilled schedule into one monotone
        // `1..=11` bar from the σ-DERIVED `current` each stage forwards — NOT a blind per-call counter.
        // The curated 2nd-order solvers LTX advertises (`heun`, `dpmpp_sde`) evaluate the model twice
        // per step, so `on_step` fires twice per step; the old counter reached 22 against `total: 11`
        // (>100% overrun). `StageProgressFold` detects the per-stage restart, folds prior stages in,
        // dedupes the multi-eval repeats, and clamps — so the bar stays monotone non-decreasing,
        // reaches `total`, and never overruns.
        let mut fold = StageProgressFold::new(TOTAL_STEPS);
        let mut on_step = |cur: usize| {
            if let Some(current) = fold.observe(cur as u32) {
                on_progress(Progress::Step {
                    current,
                    total: TOTAL_STEPS,
                });
            }
        };
        // extend_clip / video_bridge ride the IC-LoRA keyframe-append path (stage-1 in-context tokens);
        // everything else (T2V / I2V / first_last_frame) is the replace-latent path.
        // sc-10976: build the AvDiT now — AFTER the staged text phase freed the ~24 GB Gemma — so the
        // TE and the DiT never co-reside. Built past the curated-sampler validation above so a rejected
        // request doesn't pay the DiT load. Dropped + `clear_cache()`d below before the VAE decode.
        let transformer = self.build_transformer()?;
        let run_dfr = |plan: &DfrPlan, on_step: &mut dyn FnMut(usize)| {
            let execution = self.ltx25_execution()?;
            let dfr = generate_dfr_av_latents(
                &DfrComponents {
                    dit: &transformer,
                    spatial_upsampler: &self.upsampler,
                    temporal_upsampler: Some(&execution.temporal_upsampler),
                    latent_mean: &self.latent_mean,
                    latent_std: &self.latent_std,
                    video_ctx,
                    audio_ctx,
                    audio_pos: &audio_pos,
                },
                &DfrRequest {
                    canvas_frames: i64::from(plan.canvas_frames),
                    requested_frames: i64::from(plan.requested_frames),
                    keyframe_positions: &plan.keyframe_positions,
                    fps: req.fps.unwrap_or(24) as f32,
                    seed,
                    temporal_upsample_rounds: plan.temporal_upsample_rounds,
                    detailing_downscale: None,
                    video_keyframes,
                },
                video_s1,
                &pos1,
                video_s2,
                &pos2,
                audio_s1,
                audio_s2,
                &req.cancel,
                on_step,
            )?;
            Ok((dfr.video_latent, dfr.audio_latent, dfr.playback_fps as u32))
        };
        let run_plain = |on_step: &mut dyn FnMut(usize)| {
            if !video_clips.is_empty() {
                generate_av_latents_iclora(
                    &transformer,
                    &self.upsampler,
                    video_s1,
                    &pos1,
                    video_s2,
                    &pos2,
                    audio_s1,
                    audio_s2,
                    &audio_pos,
                    video_ctx,
                    audio_ctx,
                    &self.latent_mean,
                    &self.latent_std,
                    video_clips,
                    (LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32),
                    &req.cancel,
                    on_step,
                )
                .map(|(video, audio)| (video, audio, req.fps.unwrap_or(24)))
            } else {
                generate_av_latents(
                    &transformer,
                    &self.upsampler,
                    video_s1,
                    &pos1,
                    video_s2,
                    &pos2,
                    audio_s1,
                    audio_s2,
                    &audio_pos,
                    video_ctx,
                    audio_ctx,
                    &self.latent_mean,
                    &self.latent_std,
                    video_keyframes,
                    curated,
                    seed,
                    &req.cancel,
                    on_step,
                )
                .map(|(video, audio)| (video, audio, req.fps.unwrap_or(24)))
            }
        };
        let (video_latents, audio_latents, playback_fps) =
            dispatch_dfr(dfr_plan.as_ref(), &mut on_step, run_dfr, run_plain)?;

        // sc-10976: force the denoise output to materialize, then drop the DiT + free the allocator
        // cache so the VAE + audio decode run in the freed footprint (the AvDiT is the denoise peak and
        // nothing downstream needs it). Mirrors Wan's scope-block release before the VAE decode.
        mlx_rs::transforms::eval([&video_latents, &audio_latents])?;
        finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Denoise, || Ok(()))?;
        drop(transformer);
        mlx_rs::memory::clear_cache();

        on_progress(Progress::Decoding);
        let frames = self.decode_video(req, &video_latents, seed)?;
        let images = frames_to_images(&frames)?;
        // Audio always denoised (it conditions the video); decode it unless `--no-audio`.
        let audio = if Self::no_audio(req) {
            None
        } else {
            Some(decode_audio_track(
                &self.audio_decoder,
                &self.vocoder,
                &audio_latents,
                self.audio_sample_rate,
                &req.cancel,
            )?)
        };
        finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Decode, || Ok(()))?;
        Ok(GenerationOutput::Video {
            frames: images,
            fps: playback_fps,
            audio,
        })
    }

    /// Extract the single I2V conditioning image + its strength from the request. The per-reference
    /// strength wins over `req.strength`, falling back to [`DEFAULT_IMAGE_STRENGTH`]. LTX I2V
    /// conditions on exactly one image (multi-keyframe / first-last-frame is parity-plus), so more
    /// than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, f32)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "ltx_2_3: multiple reference images are not supported (single-image I2V \
                         only; multi-keyframe / first-last-frame is parity-plus, sc-2685)"
                            .into(),
                    ));
                }
                reference = Some((
                    image,
                    strength.or(req.strength).unwrap_or(DEFAULT_IMAGE_STRENGTH),
                ));
            }
        }
        Ok(reference)
    }

    /// Build the replace-latent keyframes (single-image I2V `Reference` at frame 0 + explicit
    /// `Keyframe`s) as owned `(stage1_latent, stage2_latent, latent_frame_idx, strength)` tuples, each
    /// VAE-encoded at both stage resolutions. [`Conditioning::Keyframe`]'s `frame_idx` is a **latent**
    /// frame index with Python-style negative indexing (`-1` = last latent frame), so first_last_frame
    /// is `[@0, @-1]` without the caller knowing the latent-frame count. Out-of-range indices error.
    fn build_keyframes(&self, req: &GenerationRequest) -> Result<Vec<(Array, Array, i32, f32)>> {
        let lf = Self::latent_dims(req).0 as i32;
        let mut out = Vec::new();
        if let Some((image, strength)) = self.resolve_reference(req)? {
            // F-051: check + materialize before each VAE encode so a cancel during this (unbounded)
            // conditioning stage is honored between encodes rather than only at the first denoise step.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let s1 = self.encode_conditioning(image, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(image, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, IMAGE_FRAME_IDX, strength));
        }
        if let Some(images) = req.conditioning.iter().find_map(|entry| match entry {
            Conditioning::MultiReference { images } => Some(images.as_slice()),
            _ => None,
        }) {
            // SC-20776: LTX's native IC-LoRA has exactly one frame-zero image-latent carrier.
            // Keep every ordered identity reference physically present by composing the closed
            // 1–4 surface before either VAE encode; do not accept it merely in the descriptor.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let composite = crate::conditioning::compose_ordered_character_references(
                images, req.width, req.height,
            )?;
            let s1 = self.encode_conditioning(&composite, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(&composite, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, IMAGE_FRAME_IDX, 1.0));
        }
        for kf in req.keyframes() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let idx = if kf.frame_idx < 0 {
                lf + kf.frame_idx
            } else {
                kf.frame_idx
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: keyframe latent frame index {} out of bounds for {lf} latent frames",
                    kf.frame_idx
                )));
            }
            let s1 = self.encode_conditioning(kf.image, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(kf.image, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, idx, kf.strength));
        }
        Ok(out)
    }

    /// Build the in-context conditioning clips ([`Conditioning::VideoClip`] — extend_clip /
    /// video_bridge) as owned `(stage1_clip_latent, output_frame_offset, strength)` tuples, each
    /// VAE-encoded at **stage-1** (half-res) resolution into `(1, 128, cf, h1, w1)`. `frame_idx` is a
    /// latent frame index with negative-from-end indexing (`-1` = last latent frame), resolved against
    /// the target latent-frame count `lf` and converted to the output-frame coordinate required by
    /// the appended-token RoPE path. Video conditioning is stage-1 only (reference
    /// `ICLoraPipeline`), so no stage-2 encode.
    fn build_clips(&self, req: &GenerationRequest) -> Result<Vec<(Array, i32, f32)>> {
        let lf = Self::latent_dims(req).0 as i32;
        let mut out = Vec::new();
        for clip in req.video_clips() {
            // F-051: honor a cancel between the (unbounded) per-clip VAE encodes.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if clip.frames.is_empty() {
                return Err(Error::Msg(
                    "ltx_2_3: video conditioning clip is empty".into(),
                ));
            }
            let idx = if clip.frame_idx < 0 {
                lf + clip.frame_idx
            } else {
                clip.frame_idx
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: clip latent frame index {} out of bounds for {lf} latent frames",
                    clip.frame_idx
                )));
            }
            let video = preprocess_conditioning_clip(clip.frames, req.width / 2, req.height / 2)?;
            let latent = self.vae.encode(&video)?;
            latent.eval()?;
            out.push((
                latent,
                crate::conditioning::latent_frame_to_output_offset(
                    idx,
                    crate::positions::TEMPORAL_SCALE,
                )?,
                clip.strength,
            ));
        }
        // replace_person: the masked control clip rides the same keyframe-append path. Build the
        // gray-neutralized control frames host-side (port of the worker's `_apply_replacement_mask`),
        // then append at `start_frame` with strength = masking_strength (the reference passes
        // `video_conditioning = [(masked_clip, masking_strength)]`). `mode` is carried on the contract
        // but does not change the math here — the per-frame mask already encodes the region (the native
        // LTX path is region-driven; `replacement_mode` only affects the diffusers WanVACE path).
        if let Some(cc) = req.control_clip() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if cc.frames.is_empty() {
                return Err(Error::Msg(
                    "ltx_2_3: replace_person control clip is empty".into(),
                ));
            }
            if cc.frames.len() != cc.mask.len() {
                return Err(Error::Msg(format!(
                    "ltx_2_3: replace_person frame count {} != mask count {}",
                    cc.frames.len(),
                    cc.mask.len()
                )));
            }
            let idx = if cc.start_frame < 0 {
                lf + cc.start_frame
            } else {
                cc.start_frame
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: replace_person start_frame {} out of bounds for {lf} latent frames",
                    cc.start_frame
                )));
            }
            let masked: Vec<Image> = cc
                .frames
                .iter()
                .zip(cc.mask.iter())
                .map(|(f, m)| apply_replacement_mask(f, m, cc.masking_strength))
                .collect::<Result<_>>()?;
            let video = preprocess_conditioning_clip(&masked, req.width / 2, req.height / 2)?;
            let latent = self.vae.encode(&video)?;
            latent.eval()?;
            out.push((
                latent,
                crate::conditioning::latent_frame_to_output_offset(
                    idx,
                    crate::positions::TEMPORAL_SCALE,
                )?,
                cc.masking_strength,
            ));
        }
        Ok(out)
    }

    /// VAE-encode the conditioning image at a stage's pixel resolution `(px_h, px_w)` → the f32 clean
    /// latent `(1, 128, 1, px_h/32, px_w/32)`. The encoder is an f32 quality island (like the VAE
    /// decode); the caller casts the latent to the path dtype.
    ///
    /// Re-compresses the image first at the checkpoint's resolved `default_image_crf` (sc-18759 —
    /// [`CHECKPOINT_MODEL_VERSION`] resolves to [`crate::params::LTX_2_3_PARAMS`]'s `crf: 33`)
    /// before the existing f32 normalize/layout in [`preprocess_conditioning_image`].
    fn encode_conditioning(&self, image: &Image, px_h: u32, px_w: u32) -> Result<Array> {
        let video = condition_image_for_checkpoint(
            image,
            px_w,
            px_h,
            CHECKPOINT_MODEL_VERSION,
            None,
            &mut default_image_recompress,
        )?; // f32 (1,3,1,px_h,px_w)
        self.vae.encode(&video)
    }

    /// Prompt enhancement (sc-2845). Returns the rewritten prompt when `req.enhance_prompt` is set and
    /// the enhancer produces non-empty output; `None` (use the original prompt) when off, or — matching
    /// `generate_av.py`'s try/except — on **any** enhancer failure or empty output. Failures are logged
    /// to stderr with the reference's `ENHANCER_FALLBACK:` token; success with `ENHANCED_PROMPT:`.
    fn maybe_enhance(&self, te: &LtxTextEncoder, req: &GenerationRequest) -> Option<String> {
        if !req.enhance_prompt {
            return None;
        }
        match self.run_enhance(te, req) {
            Ok(p) if !p.trim().is_empty() => {
                eprintln!("ENHANCED_PROMPT:{p}");
                Some(p)
            }
            Ok(_) => {
                eprintln!("ENHANCER_FALLBACK:EmptyOutput:prompt enhancer returned empty output");
                None
            }
            Err(e) => {
                eprintln!("ENHANCER_FALLBACK:{e}");
                None
            }
        }
    }

    /// Run the configured enhancer: the uncensored 4-bit Gemma (`use_uncensored_enhancer`) or the
    /// already-loaded text-encoder backbone. I2V (a `Reference` image present) selects the I2V system
    /// prompt **only on the uncensored path** — the reference's censored `enhance_t2v` always uses the
    /// T2V system prompt (`generate_av.py` never calls `enhance_i2v` there), which we match.
    fn run_enhance(&self, te: &LtxTextEncoder, req: &GenerationRequest) -> Result<String> {
        let is_i2v = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let temperature = req
            .enhance_temperature
            .unwrap_or(enhance::DEFAULT_TEMPERATURE);
        let cfg = EnhanceConfig {
            // F-012 twin: clamp the request-supplied budget so a huge `enhance_max_tokens` can't turn
            // a single `enhance_prompt=true` request into an effectively unbounded decode job.
            max_tokens: enhance::clamp_max_tokens(req.enhance_max_tokens),
            seed: req.seed.unwrap_or(enhance::DEFAULT_SEED),
        };
        if req.use_uncensored_enhancer {
            let (model, tokenizer) = self.load_uncensored_enhancer()?;
            let system = if is_i2v {
                enhance::I2V_SYSTEM_PROMPT
            } else {
                enhance::T2V_SYSTEM_PROMPT
            };
            enhance::enhance(
                &model,
                &tokenizer,
                system,
                &req.prompt,
                &cfg,
                &SampleParams::uncensored(temperature),
                Some(&req.cancel),
            )
        } else {
            let Tokenizer::Gemma3(tokenizer) = &self.tokenizer else {
                unreachable!("Gemma-4 routes are filtered before prompt enhancement")
            };
            enhance::enhance(
                te.gemma(),
                tokenizer,
                enhance::T2V_SYSTEM_PROMPT,
                &req.prompt,
                &cfg,
                &SampleParams::censored(temperature),
                Some(&req.cancel),
            )
        }
    }

    /// Load the separate uncensored 4-bit Gemma enhancer + its tokenizer on demand (the reference
    /// `enhance_with_model` loads it per call). A standalone mlx_lm checkpoint → `model.` key prefix;
    /// its `config.json` `quantization` block drives the 4-bit dequant. The snapshot dir was staged by
    /// the caller in the `uncensored_enhancer` component at load (sc-13664); a request that asks for the
    /// uncensored enhancer without it provisioned is an actionable error here (no env / HF-cache scan).
    fn load_uncensored_enhancer(&self) -> Result<(GemmaModel, LtxTokenizer)> {
        let dir = self.uncensored_enhancer.as_ref().ok_or_else(|| {
            Error::Msg(
                "ltx_2_3: use_uncensored_enhancer was requested but no 'uncensored_enhancer' \
                 component was staged in LoadSpec::components — provision the amoral 4-bit Gemma \
                 snapshot dir (TheCluster/amoral-gemma-3-12B-v2-mlx-4bit) via \
                 with_component(\"uncensored_enhancer\", WeightsSource::Dir(...))"
                    .into(),
            )
        })?;
        let w = Weights::from_dir(dir)?;
        let quant = resolve_gemma_quant(dir)?;
        let model =
            GemmaModel::from_weights_with_prefix(&w, GemmaConfig::gemma_3_12b(), quant, "model.")?;
        let tokenizer = LtxTokenizer::from_dir(dir)?;
        Ok((model, tokenizer))
    }
}

impl Generator for Ltx25 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        // Reuse the full conditioning/index validation that owns the LTX request surface.  The
        // descriptor supplied here is the 2.5 descriptor, so shared capability gates stay closed
        // for any axis not yet consumed by the assembled execution route.
        validate_request_for(MODEL_25_ID, &self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // The load-time metadata checks above prove the provider is a split 2.5 route; this is the
        // matching runtime reachability seam, where its Gemma-4, split DiT, component VAE/audio and
        // spatial-upscaler are actually materialised and driven through the ordinary LTX pipeline.
        let route = build_ltx25(&self.spec).map_err(mlx_gen::gen_core::Error::from)?;
        route.generate(req, on_progress)
    }
}

/// Capability-driven request validation (weight-free, so it's unit-testable without a load): the
/// shared capability floor ([`Capabilities::validate_request`] — size range, count, unsupported
/// guidance/negative/true_cfg/sampler/scheduler, only advertised conditioning kinds) plus the LTX
/// model-specific constraints: non-empty prompt, 64-aligned width/height (stage-1 runs at //2//32),
/// `num_frames = 1 + 8·k`, and all weight-free conditioning cardinality/shape/index constraints.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    validate_request_for(MODEL_ID, caps, req)
}

fn validate_request_for(
    model_id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{model_id}: prompt must not be empty")));
    }
    // DFR knobs (sc-18789): the 2.3 checkpoint has no learned keyframe-slot marker
    // (`use_keyframes_abs_pos_embedding: false`), so generated keyframe slots would be denoised as
    // unmarked tokens — wasted compute with no conditioning effect. Refuse up front like the
    // reference's `assert_generated_keyframes_supported`, typed so the worker distinguishes the
    // capability gap from a generic failure.
    //
    // These run BEFORE the shared floor on purpose (sc-18778). The floor now refuses the same two
    // knobs from `supports_generated_keyframes` / `max_temporal_upsample_rounds`, which this
    // descriptor leaves at their refusing defaults — so the floor would catch them anyway. Going
    // first is what keeps the more actionable message: the floor can only say "this engine does
    // not support it", while these name the checkpoint generation that does, which is what a
    // caller needs to act. Deleting them would not un-refuse the knobs, only blur the reason.
    if model_id == MODEL_ID && req.num_generated_keyframes.is_some_and(|n| n > 0) {
        return Err(Error::Unsupported(
            "ltx_2_3: num_generated_keyframes requires a generated-keyframe checkpoint \
             (use_keyframes_abs_pos_embedding, LTX >= 2.5)"
                .into(),
        ));
    }
    if model_id == MODEL_ID && req.temporal_upsample_rounds.is_some_and(|r| r > 0) {
        return Err(Error::Unsupported(
            "ltx_2_3: temporal_upsample_rounds requires the LTX-2.5 DFR pipeline (generated \
             keyframe slots + the temporal latent upsampler)"
                .into(),
        ));
    }
    caps.validate_request(model_id, req)?;
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "ltx_2_3: width/height must be divisible by {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames % 8 != 1 {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames must be 1 + 8·k (got {frames})"
            )));
        }
        if frames > MAX_FRAMES {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames {frames} exceeds the maximum {MAX_FRAMES}"
            )));
        }
    }
    // The 2.5 provider's DFR plan is the execution geometry consumed after the staged duration
    // prediction.  Validate the explicit-frame shape now so a malformed keyframe/temporal request
    // never pays for Gemma; auto-duration requests are resolved and planned at generation time.
    if model_id == MODEL_25_ID && req.frames.is_some() {
        let _ = plan_dfr_request(req)?;
    }
    let latent_frames = Ltx::latent_dims(req).0 as i32;
    let resolve_latent_index = |label: &str, idx: i32| -> Result<()> {
        let resolved = if idx < 0 { latent_frames + idx } else { idx };
        if resolved < 0 || resolved >= latent_frames {
            return Err(Error::Msg(format!(
                "ltx_2_3: {label} latent frame index {idx} out of bounds for {latent_frames} latent frames"
            )));
        }
        Ok(())
    };

    // Apply-or-reject at the weight-free boundary. Plain I2V consumes one `Reference`; the closed
    // replace_person composite is exactly one ControlClip + one 1–4 image MultiReference carrier.
    // Anything crossed is refused before the ~24 GB staged text phase, not silently discarded.
    let reference_count = req
        .conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::Reference { .. }))
        .count();
    if reference_count > 1 {
        return Err(Error::Msg(
            "ltx_2_3: multiple reference images are not supported (single-image I2V only)".into(),
        ));
    }
    let multi_references: Vec<&[Image]> = req
        .conditioning
        .iter()
        .filter_map(|c| match c {
            Conditioning::MultiReference { images } => Some(images.as_slice()),
            _ => None,
        })
        .collect();
    if multi_references.len() > 1 {
        return Err(Error::Msg(
            "ltx_2_3: replace_person accepts exactly one ordered MultiReference carrier".into(),
        ));
    }
    let control_clip_count = req
        .conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::ControlClip { .. }))
        .count();
    if control_clip_count > 1 {
        return Err(Error::Msg(
            "ltx_2_3: at most one ControlClip can be applied per request".into(),
        ));
    }
    let replace_person = control_clip_count == 1 || !multi_references.is_empty();
    if replace_person {
        if control_clip_count != 1 {
            return Err(Error::Msg(
                "ltx_2_3: replace_person requires exactly one ControlClip".into(),
            ));
        }
        let Some(images) = multi_references.first().copied() else {
            return Err(Error::Msg(
                "ltx_2_3: replace_person requires exactly one ordered MultiReference carrier"
                    .into(),
            ));
        };
        if reference_count != 0
            || req.conditioning.iter().any(|c| {
                matches!(
                    c,
                    Conditioning::Keyframe { .. } | Conditioning::VideoClip { .. }
                )
            })
        {
            return Err(Error::Msg(
                "ltx_2_3: replace_person cannot be mixed with Reference, Keyframe, or VideoClip conditioning"
                    .into(),
            ));
        }
        // This also verifies every RGB buffer and dimensions before model construction. The render
        // recomputes the same deterministic contact sheet for the VAE input.
        crate::conditioning::compose_ordered_character_references(images, req.width, req.height)?;
    }
    // F-054: range-validate every conditioning strength to [0, 1]. `strength > 1` → a negative denoise
    // mask (`1 − strength`) → negative per-token σ timesteps and extrapolating blends (silent garbage,
    // every stage "succeeds"); `< 0` (or NaN — `contains` is false for NaN) is likewise degenerate.
    // `apply_replacement_mask` clamps a local copy but the same value flows unclamped as the clip
    // strength, so reject it at the request boundary. The top-level img2img `strength` is also covered
    // by the shared floor's finiteness guard; the range check here is LTX-specific.
    let check_strength = |label: &str, s: f32| -> Result<()> {
        if !(0.0..=1.0).contains(&s) {
            return Err(Error::Msg(format!(
                "ltx_2_3: {label} strength must be in [0, 1] (got {s})"
            )));
        }
        Ok(())
    };
    if let Some(s) = req.strength {
        check_strength("img2img", s)?;
    }
    for c in &req.conditioning {
        match c {
            Conditioning::Reference {
                strength: Some(s), ..
            } => check_strength("reference", *s)?,
            Conditioning::Keyframe {
                frame_idx,
                strength,
                ..
            } => {
                resolve_latent_index("keyframe", *frame_idx)?;
                check_strength("keyframe", *strength)?;
            }
            Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            } => {
                if frames.is_empty() {
                    return Err(Error::Msg(
                        "ltx_2_3: video conditioning clip is empty".into(),
                    ));
                }
                resolve_latent_index("clip", *frame_idx)?;
                check_strength("video clip", *strength)?;
            }
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                ..
            } => {
                if frames.is_empty() {
                    return Err(Error::Msg(
                        "ltx_2_3: replace_person control clip is empty".into(),
                    ));
                }
                if frames.len() != mask.len() {
                    return Err(Error::Msg(format!(
                        "ltx_2_3: replace_person frame count {} != mask count {}",
                        frames.len(),
                        mask.len()
                    )));
                }
                resolve_latent_index("replace_person", *start_frame)?;
                if *start_frame != 0 {
                    return Err(Error::Msg(format!(
                        "ltx_2_3: replace_person ControlClip must start at latent frame 0 (got {start_frame})"
                    )));
                }
                check_strength("control clip masking", *masking_strength)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame.
pub(crate) fn frames_to_images(frames: &Array) -> Result<Vec<Image>> {
    let sh = frames.shape(); // (F, H, W, 3)
    let (f, h, w) = (sh[0] as usize, sh[1] as u32, sh[2] as u32);
    let data = frames.as_slice::<u8>();
    let per = (h as usize) * (w as usize) * 3;
    Ok((0..f)
        .map(|i| Image {
            width: w,
            height: h,
            pixels: data[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

/// Request-local calibration fault injection at a completed physical phase boundary.
///
/// The shared request floor accepts the hidden fault carrier only when a conformance harness pairs
/// it with explicit authorization. Production requests leave it unset, so the hook is inert.
/// Returning the completed value through
/// this helper is deliberate: when the named fault fires, phase-local resources are dropped before
/// the error reaches the caller, which is the cleanup behavior the harness measures with a warm
/// follow-up request.
fn finish_calibration_phase<T>(
    req: &GenerationRequest,
    phase: mlx_gen::gen_core::MemoryPhase,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let value = work()?;
    if req.memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    }) {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: injected memory-strategy calibration error at {phase:?}"
        )));
    }
    Ok(value)
}

// Hand-written rather than `impl_generator!`: SC-19109 makes the memory contract a property of the
// loaded provider, not only a static registry row. The ordinary descriptor/validate/generate arms
// are the same delegation the macro emitted.
impl Generator for Ltx {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        if self.descriptor.id == MODEL_ID {
            validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
        } else {
            validate_request_for(self.descriptor.id, &self.descriptor.capabilities, req)
                .map_err(Into::into)
        }
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                mlx_gen::gen_core::MemorySafetyDecision::Accept
            } else {
                mlx_gen::gen_core::MemorySafetyDecision::Reject {
                    reason: "ltx_2_3 loaded route has no calibrated memory-strategy contract"
                        .into(),
                }
            };
        };
        crate::memory_strategy::safety_check(
            contract,
            tier,
            self.memory_overlay.as_deref(),
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return Ok(None);
        };
        crate::memory_strategy::begin_request(
            contract,
            tier,
            self.memory_overlay.as_deref(),
            self.config.num_layers as usize,
            context,
        )
    }
}

impl Ltx {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        // sc-10976 staged text phase: build the Gemma TE, run the optional prompt enhancer (sc-2845 —
        // rewrites `req.prompt` before encoding; default off, falls back to the original prompt on any
        // failure, so the `enhance_prompt = false` parity seams are untouched) + the video/audio encode
        // inside one TE scope, then drop the ~24 GB Gemma before the DiT loads. Honor a cancel before
        // this (unbounded) text stage.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (enhanced, video_ctx, audio_ctx) =
            finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Conditioning, || {
                self.stage_text_phase(req)
            })?;
        // The TE is dropped; free the allocator cache so the DiT loads into the low-water footprint.
        mlx_rs::memory::clear_cache();
        let prompted = match enhanced {
            Some(prompt) => GenerationRequest {
                prompt,
                ..req.clone()
            },
            None => req.clone(),
        };
        let owned = self.apply_auto_frames(&prompted, &video_ctx, &audio_ctx)?;
        // Auto-duration supplies `frames` only after the text phase.  Re-run the ordinary provider
        // floor on that concrete request so its maximum/stride and DFR geometry bounds apply to the
        // prediction before any noise allocation.
        self.validate(&owned)?;
        let req = &owned;
        // DFR may pad the latent canvas before its temporal rounds, while the public request's
        // frame count remains the trim contract.  Allocate the four ordinary provider noises for
        // that execution canvas; `generate_av_from_embeddings` consumes the same plan below.
        let dfr_plan = self.dfr_plan(req)?;
        let noise_request = match dfr_plan.as_ref() {
            Some(plan) if plan.canvas_frames != req.frames.unwrap_or(1) => GenerationRequest {
                frames: Some(plan.canvas_frames),
                ..req.clone()
            },
            _ => req.clone(),
        };
        let (lf, h1, w1, h2, w2) = Self::latent_dims(&noise_request);
        let af = Self::audio_frames(&noise_request) as i32;
        let seed = req.seed.unwrap_or_else(default_seed);
        // Seeded noise at the path dtype (the reference seeds `normal(...).astype(model_dtype)`). RNG
        // is not portable to mlx-python, so the pixel/waveform parity gate injects the reference
        // samples via `generate_with_noise`. Distinct keys per stage/modality. I2V conditioning (when
        // a `Reference` is supplied) + the audio decode are handled inside `generate_with_noise`.
        let normal = |key: u64, shape: &[i32]| -> Result<Array> {
            let k = random::key(key)?;
            Ok(random::normal::<f32>(shape, None, None, Some(&k))?.as_dtype(self.stat_dt)?)
        };
        let video_s1 = normal(seed, &[1, LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32])?;
        let video_s2 = normal(
            seed.wrapping_add(1),
            &[1, LATENT_CHANNELS, lf as i32, h2 as i32, w2 as i32],
        )?;
        let audio_s1 = normal(
            seed.wrapping_add(2),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        let audio_s2 = normal(
            seed.wrapping_add(3),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        self.generate_with_noise(
            req,
            &video_ctx,
            &audio_ctx,
            &video_s1,
            &video_s2,
            &audio_s1,
            &audio_s2,
            on_progress,
        )
    }
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
mod tests {
    use super::*;
    // sc-19502: the derived stage-1 step count the descriptor advertises.
    use crate::pipeline::NATIVE_STEPS;

    #[test]
    fn ltx25_registered_route_enforces_geometry_and_temporal_constraints() {
        let generator = Ltx25 {
            descriptor: descriptor_25(),
            spec: LoadSpec::new(WeightsSource::Dir(PathBuf::from("."))),
        };
        let request = GenerationRequest {
            prompt: "a quiet moonlit harbor".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            ..Default::default()
        };
        Generator::validate(&generator, &request).expect("64px / 1+8*k request must pass");

        let mut geometry = request.clone();
        geometry.width = 672; // divisible by 32, but not the provider's 64px stage-1 stride.
        assert!(Generator::validate(&generator, &geometry)
            .unwrap_err()
            .to_string()
            .contains("divisible by 64"));

        let mut temporal = request;
        temporal.frames = Some(16);
        assert!(Generator::validate(&generator, &temporal)
            .unwrap_err()
            .to_string()
            .contains("1 + 8"));
    }

    #[test]
    fn ltx25_provider_admits_only_advanced_axes_it_executes() {
        let generator = Ltx25 {
            descriptor: descriptor_25(),
            spec: LoadSpec::new(WeightsSource::Dir(PathBuf::from("."))),
        };
        let request = GenerationRequest {
            prompt: "a quiet moonlit harbor".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            num_generated_keyframes: Some(1),
            ..Default::default()
        };
        assert!(Generator::validate(&generator, &request).is_ok());
        let mut temporal = request.clone();
        temporal.num_generated_keyframes = None;
        temporal.temporal_upsample_rounds = Some(1);
        assert!(Generator::validate(&generator, &temporal).is_ok());
        let mut automatic = request;
        automatic.num_generated_keyframes = None;
        automatic.auto_duration = Some(mlx_gen::gen_core::duration_head::AutoDurationRange {
            min_seconds: 1.0,
            max_seconds: 2.0,
        });
        assert!(Generator::validate(&generator, &automatic).is_ok());
        assert!(
            generator
                .descriptor()
                .capabilities
                .supports_diffusion_decoder
        );
    }

    /// The DurationHead predictor is not merely admitted by the descriptor: its value becomes the
    /// request's actual frame plan before the provider allocates video/audio noise.  The injected
    /// spy is the production seam (`DurationHead::predict_seconds`) and makes a deleted/bypassed
    /// call fail this test instead of leaving a default-frame false green.
    #[test]
    fn ltx25_auto_duration_prediction_drives_the_provider_frame_plan() {
        let request = GenerationRequest {
            prompt: "a slow orbit around a lighthouse".into(),
            width: 512,
            height: 512,
            fps: Some(24),
            auto_duration: Some(mlx_gen::gen_core::duration_head::AutoDurationRange {
                min_seconds: 2.0,
                max_seconds: 8.0,
            }),
            ..Default::default()
        };
        let calls = std::cell::Cell::new(0usize);
        let planned = apply_predicted_frames(&request, 24.0, &mut || {
            calls.set(calls.get() + 1);
            Ok(3.0)
        })
        .expect("duration head result must resolve to provider frames");
        assert_eq!(
            calls.get(),
            1,
            "the real predictor seam must be called once"
        );
        assert_ne!(
            planned.frames, request.frames,
            "prediction must affect the plan"
        );
        assert_eq!(planned.frames, Some(65));
        assert_ne!(Ltx::latent_dims(&planned).0, Ltx::latent_dims(&request).0);

        let explicit = GenerationRequest {
            frames: Some(17),
            ..request
        };
        let explicit_plan = apply_predicted_frames(&explicit, 24.0, &mut || {
            panic!("explicit frames must bypass the duration-head predictor")
        })
        .expect("explicit frame count wins");
        assert_eq!(explicit_plan.frames, Some(17));
    }

    /// This is the request-side DFR plan consumed by the ordinary provider's
    /// `generate_dfr_av_latents` branch.  A temporal request changes both the noise canvas and the
    /// trim contract; a count request changes the actual slot positions.  Either call being
    /// deleted or replaced with the plain two-stage plan makes one of these assertions fail.
    #[test]
    fn ltx25_dfr_request_plan_drives_temporal_and_generated_slot_execution() {
        let temporal = GenerationRequest {
            prompt: "fast tracking shot through a market".into(),
            width: 512,
            height: 512,
            frames: Some(153),
            temporal_upsample_rounds: Some(2),
            ..Default::default()
        };
        let temporal_plan = plan_dfr_request(&temporal)
            .expect("DFR temporal plan")
            .expect("temporal request must select DFR");
        assert_eq!(temporal_plan.requested_frames, 153);
        assert_eq!(
            temporal_plan.canvas_frames, 161,
            "DFR must pad before denoise"
        );
        assert_eq!(temporal_plan.temporal_upsample_rounds, 2);
        assert_eq!(temporal_plan.keyframe_positions, vec![32, 64, 96, 128, 160]);

        let dfr_calls = std::cell::Cell::new(0usize);
        let plain_calls = std::cell::Cell::new(0usize);
        let mut ignored_progress = |_| {};
        let branch = dispatch_dfr(
            Some(&temporal_plan),
            &mut ignored_progress,
            |plan, _| {
                dfr_calls.set(dfr_calls.get() + 1);
                Ok((plan.canvas_frames, plan.temporal_upsample_rounds))
            },
            |_| {
                plain_calls.set(plain_calls.get() + 1);
                Ok((0, 0))
            },
        )
        .expect("a DFR plan must dispatch to the DFR execution branch");
        assert_eq!(branch, (161, 2));
        assert_eq!(dfr_calls.get(), 1);
        assert_eq!(plain_calls.get(), 0);

        let generated = GenerationRequest {
            num_generated_keyframes: Some(3),
            frames: Some(121),
            ..temporal
        };
        let generated_plan = plan_dfr_request(&generated)
            .expect("generated-slot plan")
            .expect("generated keyframes must select DFR");
        assert_eq!(generated_plan.keyframe_positions, vec![30, 60, 90]);
        assert_eq!(
            mlx_gen::gen_core::ltx_dfr::dfr_target_frames(
                i64::from(temporal_plan.requested_frames),
                temporal_plan.temporal_upsample_rounds,
            ),
            609,
            "the DFR trim contract must survive two real temporal rounds"
        );
    }

    /// A staged diffusion VAE is a provider execution choice, not an extra file the conv path may
    /// quietly ignore.  The spy observes the exact budget/mode passed into the same dispatch that
    /// `decode_video` uses, so deleting that call or replacing it with `decode_seeded` goes red.
    #[test]
    fn ltx25_staged_diffvae_executes_the_budgeted_decoder_mode() {
        let conv = LoadSpec::new(WeightsSource::Dir(PathBuf::from(".")));
        assert_eq!(ltx25_decoder_selection(&conv), Ltx25DecoderSelection::Conv);

        let diffusion = conv.with_component(
            LtxComponent::DiffusionVideoVae.id(),
            WeightsSource::File(PathBuf::from("/tmp/vae_diffusion_decoder.safetensors")),
        );
        assert_eq!(
            ltx25_decoder_selection(&diffusion),
            Ltx25DecoderSelection::DiffusionBudgeted(DEFAULT_DIFFVAE_MODE)
        );
        assert_eq!(
            ltx25_video_component(&diffusion),
            LtxComponent::DiffusionVideoVae
        );

        let observed = std::cell::Cell::new(None);
        let output = decode_diffvae_budgeted(DEFAULT_DIFFVAE_MODE, |mode| {
            observed.set(Some(mode));
            Ok("budgeted-diffvae")
        })
        .expect("the selected decoder must be invoked");
        assert_eq!(output, "budgeted-diffvae");
        assert_eq!(observed.get(), Some(DiffVaeMode::ChunkedEager));
    }

    #[test]
    fn calibration_fault_is_request_local_and_phase_exact() {
        use mlx_gen::gen_core::{GenerationMemory, MemoryPhase};

        let plain = GenerationRequest::default();
        for phase in [
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ] {
            assert_eq!(
                finish_calibration_phase(&plain, phase, || Ok(17)).unwrap(),
                17
            );

            let phase_without_authorization = GenerationRequest {
                memory: Some(GenerationMemory {
                    calibration_error_phase: Some(phase),
                    calibration_fault_harness_authorized: false,
                    ..Default::default()
                }),
                ..Default::default()
            };
            assert_eq!(
                finish_calibration_phase(&phase_without_authorization, phase, || Ok(19)).unwrap(),
                19,
                "the hidden phase carrier is inert without explicit harness authorization"
            );

            let authorization_without_phase = GenerationRequest {
                memory: Some(GenerationMemory {
                    calibration_error_phase: None,
                    calibration_fault_harness_authorized: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            assert_eq!(
                finish_calibration_phase(&authorization_without_phase, phase, || Ok(21)).unwrap(),
                21,
                "authorization alone must not invent a fault phase"
            );

            let mut memory = GenerationMemory::default();
            memory.authorize_calibration_fault(phase);
            let injected = GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            };
            let error = finish_calibration_phase(&injected, phase, || Ok(17))
                .expect_err("the selected physical phase must return a typed error")
                .to_string();
            assert!(error.contains(MODEL_ID), "got: {error}");
            assert!(
                error.contains("injected memory-strategy calibration error"),
                "got: {error}"
            );
            assert!(error.contains(&format!("{phase:?}")), "got: {error}");

            let other = if phase == MemoryPhase::Decode {
                MemoryPhase::Denoise
            } else {
                MemoryPhase::Decode
            };
            assert_eq!(
                finish_calibration_phase(&injected, other, || Ok(23)).unwrap(),
                23,
                "a request-local fault must not spill into another phase"
            );
        }
    }

    #[test]
    fn calibration_fault_drops_completed_phase_state_before_warm_recovery() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Debug)]
        struct DropWitness(Arc<AtomicUsize>);
        impl Drop for DropWitness {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut memory = mlx_gen::gen_core::GenerationMemory::default();
        memory.authorize_calibration_fault(mlx_gen::gen_core::MemoryPhase::Denoise);
        let injected = GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        finish_calibration_phase(&injected, mlx_gen::gen_core::MemoryPhase::Denoise, || {
            Ok(DropWitness(drops.clone()))
        })
        .expect_err("the injected request must fail after phase state exists");
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let warm = finish_calibration_phase(
            &GenerationRequest::default(),
            mlx_gen::gen_core::MemoryPhase::Denoise,
            || Ok(DropWitness(drops.clone())),
        )
        .expect("an unmodified warm request must recover");
        drop(warm);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn calibrated_scope_finishes_injected_error_and_a_fresh_scope_recovers() {
        use mlx_gen::gen_core::{
            LoadSpec, MemoryPhase, MemoryRunOutcome, MemoryStrategy, Quant, WeightsSource,
        };

        fn fixture() -> (
            LoadSpec,
            mlx_gen::gen_core::MemoryProviderContract,
            mlx_gen::gen_core::MemoryBehaviorFixture,
        ) {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into()))
                .with_quant(Quant::Q8);
            let contract = crate::memory_strategy::weights_free_memory_strategy_contract(&spec)
                .expect("the provider-owned weights-free contract fixture");
            let fixture = crate::memory_strategy::registered_valid_fixtures(
                &spec,
                &contract,
                MemoryStrategy::StagedResidency,
            )
            .expect("the provider-owned behavior fixture")
            .pop()
            .expect("one staged-residency fixture");
            (spec, contract, fixture)
        }

        let (spec, contract, fixture) = fixture();
        let mut scope =
            crate::memory_strategy::registered_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .expect("the LTX provider must open a calibrated request scope");
        let mut injected = fixture.request.clone();
        scope.configure_request(&mut injected).unwrap();
        injected
            .memory
            .as_mut()
            .expect("the selected staged rung installs its request carrier")
            .authorize_calibration_fault(MemoryPhase::Denoise);
        scope.enter_phase(MemoryPhase::Conditioning).unwrap();
        scope.leave_phase(MemoryPhase::Conditioning).unwrap();
        scope.enter_phase(MemoryPhase::Denoise).unwrap();
        let error = finish_calibration_phase(&injected, MemoryPhase::Denoise, || Ok(()))
            .expect_err("the authorized provider fault must escape the physical boundary");
        scope.leave_phase(MemoryPhase::Denoise).unwrap();
        scope
            .finish(MemoryRunOutcome::Error {
                message: error.to_string(),
            })
            .expect("an injected provider error must complete scope cleanup");
        assert!(
            scope.finish(MemoryRunOutcome::Complete).is_err(),
            "the error outcome must terminally finish the request scope"
        );

        let mut recovery =
            crate::memory_strategy::registered_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .expect("a fresh provider request scope must open after the injected error");
        let mut recovered_request = fixture.request;
        recovery.configure_request(&mut recovered_request).unwrap();
        recovery.enter_phase(MemoryPhase::Conditioning).unwrap();
        recovery.leave_phase(MemoryPhase::Conditioning).unwrap();
        recovery.enter_phase(MemoryPhase::Denoise).unwrap();
        finish_calibration_phase(&recovered_request, MemoryPhase::Denoise, || Ok(()))
            .expect("the subsequent unmodified provider request must recover");
        recovery.leave_phase(MemoryPhase::Denoise).unwrap();
        recovery.enter_phase(MemoryPhase::Decode).unwrap();
        finish_calibration_phase(&recovered_request, MemoryPhase::Decode, || Ok(()))
            .expect("recovery must remain healthy through decode");
        recovery.leave_phase(MemoryPhase::Decode).unwrap();
        recovery.finish(MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn replacement_mask_rejects_wrapping_u32_dimensions_without_panicking() {
        let frame = Image {
            width: 65_536,
            height: 65_536,
            pixels: Vec::new(),
        };
        let mask = frame.clone();

        let err = apply_replacement_mask(&frame, &mask, 1.0)
            .expect_err("wrapped dimensions must return a typed error");
        assert!(
            err.to_string().contains("frame/mask must be RGB8"),
            "unexpected error: {err}"
        );
    }

    /// F-050 (sc-11133): drive `StageProgressFold` over the exact event stream a curated 2nd-order
    /// solver produces — each stage's σ-derived `current` forwarded TWICE per step (predictor +
    /// corrector) — and assert the folded bar is monotone, reaches `TOTAL_STEPS`, and never overruns.
    /// Before the fix the blind counter reached 22 against `total: 11`.
    #[test]
    fn stage_progress_fold_dedupes_multi_eval_and_reaches_total() {
        let mut fold = StageProgressFold::new(TOTAL_STEPS);
        let mut emitted = Vec::new();
        // Stage 1: 8 steps, each forwarded value seen twice (heun predictor+corrector), monotone
        // per-stage 1..=8. Stage 2: 3 steps, restarts at 1, forwarded 1..=3 twice each.
        let stage1: Vec<u32> = (1..=8).flat_map(|c| [c, c]).collect();
        let stage2: Vec<u32> = (1..=3).flat_map(|c| [c, c]).collect();
        for fwd in stage1.into_iter().chain(stage2) {
            if let Some(cur) = fold.observe(fwd) {
                emitted.push(cur);
            }
        }
        assert_eq!(
            emitted,
            (1..=11).collect::<Vec<_>>(),
            "the folded bar must be exactly 1..=11 (monotone, complete, no overrun)"
        );
    }

    /// The native distilled path forwards `i+1` once per step (no multi-eval); the fold must still
    /// produce the same clean `1..=11` sequence.
    #[test]
    fn stage_progress_fold_handles_single_eval_native_path() {
        let mut fold = StageProgressFold::new(TOTAL_STEPS);
        let mut emitted = Vec::new();
        for fwd in (1..=8).chain(1..=3) {
            if let Some(cur) = fold.observe(fwd) {
                emitted.push(cur);
            }
        }
        assert_eq!(emitted, (1..=11).collect::<Vec<_>>());
        assert_eq!(TOTAL_STEPS, 11);
    }

    #[test]
    fn descriptor_advertises_curated_samplers_no_scheduler() {
        // epic 7114 (sc-7122): LTX exposes the curated SAMPLER axis (the joint two-stream T2V+A denoise)
        // but NO scheduler — it keeps its baked distilled σ schedule. A non-empty scheduler list would be
        // a false capability.
        let caps = descriptor().capabilities;
        for s in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            assert!(
                caps.samplers.contains(&s),
                "curated sampler {s:?} should be advertised"
            );
        }
        assert!(
            caps.schedulers.is_empty(),
            "LTX is sampler-only (no scheduler axis — the distilled schedule is baked)"
        );
    }

    #[test]
    fn descriptor_declares_unconditional_staging_without_selectable_sequential_control() {
        let caps = descriptor().capabilities;
        assert!(!caps.supports_sequential_offload);
        assert_eq!(
            caps.staged_residency_availability(),
            mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged
        );
    }

    #[test]
    fn resolve_gemma_dir_rejects_nonexistent_spec_override() {
        // sc-8827/sc-13664: the `LoadSpec::text_encoder` slot is the only TE source, and a nonexistent
        // override path errors with the spec-side message up front — the worker drives the TE location
        // through the spec (no `$LTX_GEMMA_DIR` / HF-cache scan any more).
        let src = WeightsSource::Dir("/nonexistent/ltx_gemma".into());
        let err = resolve_gemma_dir(Some(&src)).unwrap_err().to_string();
        assert!(err.contains("LoadSpec text_encoder"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    /// sc-13664: `LoadSpec::text_encoder` is now **required** — an absent slot is a load-time,
    /// actionable error naming the slot (not a silent `$LTX_GEMMA_DIR` / HF-cache fallback).
    #[test]
    fn resolve_gemma_dir_requires_text_encoder_slot() {
        let err = resolve_gemma_dir(None).unwrap_err().to_string();
        assert!(err.contains("LoadSpec::text_encoder"), "got: {err}");
        assert!(!err.contains("LTX_GEMMA_DIR"), "got: {err}");
    }

    /// sc-13664: LTX's only recognized component is `uncensored_enhancer`; an unknown component key is
    /// rejected at load (typed `Unsupported`), and the enhancer path is validated up front when staged.
    #[test]
    fn load_rejects_unknown_component_and_validates_enhancer() {
        let bogus = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "bogus_component",
            WeightsSource::File("/x.safetensors".into()),
        );
        assert!(matches!(
            load(&bogus).err().expect("err"),
            Error::Unsupported(_)
        ));
        // A staged-but-nonexistent uncensored_enhancer errors up front, naming the component.
        let missing = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "uncensored_enhancer",
            WeightsSource::Dir("/nope/amoral".into()),
        );
        let err = load(&missing).err().expect("err").to_string();
        assert!(err.contains("uncensored_enhancer"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn latent_dims_matches_reference_formula() {
        // 256×256, 9 frames: latent_frames = 1+(9-1)/8 = 2; stage1 = H/2/32 = 4; stage2 = H/32 = 8.
        let req = GenerationRequest {
            width: 256,
            height: 256,
            frames: Some(9),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (2, 4, 4, 8, 8));
        // 512×768, 1 frame: latent_frames = 1; stage1 = 8×12; stage2 = 16×24.
        let req = GenerationRequest {
            width: 768,
            height: 512,
            frames: Some(1),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (1, 8, 12, 16, 24));
    }

    /// sc-18789: the DFR knobs are refused on the 2.3 checkpoint with a TYPED `Unsupported` — its
    /// transformer has no learned keyframe-slot marker, so slots would denoise as unmarked tokens
    /// (wasted compute, no conditioning). `0`/`None` stay accepted (off).
    #[test]
    fn validate_refuses_dfr_knobs_on_2_3_typed() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "p".into(),
            width: 640,
            height: 384,
            frames: Some(25),
            ..Default::default()
        };
        for (label, req) in [
            (
                "num_generated_keyframes",
                GenerationRequest {
                    num_generated_keyframes: Some(3),
                    ..base.clone()
                },
            ),
            (
                "temporal_upsample_rounds",
                GenerationRequest {
                    temporal_upsample_rounds: Some(1),
                    ..base.clone()
                },
            ),
        ] {
            let err = validate_request(&caps, &req).expect_err(label);
            assert!(matches!(err, Error::Unsupported(_)), "{label}: {err}");
            assert!(err.to_string().contains("2.5"), "{label}: {err}");
        }
        // Zero / unset are the off position, not a refusal.
        let off = GenerationRequest {
            num_generated_keyframes: Some(0),
            temporal_upsample_rounds: Some(0),
            ..base
        };
        validate_request(&caps, &off).expect("0 is off");
    }

    #[test]
    fn validate_request_enforces_constraints() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(33),
            ..Default::default()
        };
        assert!(validate_request(&caps, &base).is_ok());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                prompt: String::new(),
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                width: 500,
                ..base.clone()
            }
        )
        .is_err());
        // The pinned stride is `SIZE_MULTIPLE` (= 64 = 2×SPATIAL_SCALE), NOT the bare 32× VAE scale:
        // a size that is a multiple of SPATIAL_SCALE but not SIZE_MULTIPLE must still be rejected, and
        // the error must name the stride (sc-12587 — this is the value SceneWorks ties to).
        assert_eq!(SIZE_MULTIPLE, 64);
        let off_stride = validate_request(
            &caps,
            &GenerationRequest {
                width: 480, // 15×32 — a multiple of SPATIAL_SCALE but not SIZE_MULTIPLE
                ..base.clone()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("divisible by 64"),
            "expected the stride error, got: {off_stride}"
        );
        // A different in-range multiple of SIZE_MULTIPLE is accepted.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                width: 448, // 7×64
                ..base.clone()
            }
        )
        .is_ok());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(32),
                ..base.clone()
            }
        )
        .is_err());
        // F-058: an unbounded frame count that still satisfies `% 8 == 1` is rejected by the max.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(8_000_001), // 8_000_001 % 8 == 1, but far beyond MAX_FRAMES
                ..base.clone()
            }
        )
        .is_err());
        // The maximum itself is allowed; the next valid form above it is not.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(MAX_FRAMES),
                ..base.clone()
            }
        )
        .is_ok());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(MAX_FRAMES + 8),
                ..base.clone()
            }
        )
        .is_err());
    }

    /// sc-19502 — this lane used to accept ANY `req.steps` and render the baked 8-step schedule
    /// regardless, while candle refused the same request. Both now refuse it.
    ///
    /// Goes through `validate_request`, the function the generator actually calls
    /// (`impl_generator!`'s `validate` arm), rather than asserting on `descriptor().capabilities`
    /// directly — a declaration that no request path consults is exactly the inert-key failure this
    /// story exists to fix, and reading the field back would prove only that a struct literal
    /// contains what it contains.
    #[test]
    fn validate_request_refuses_an_off_schedule_step_count() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(9),
            ..Default::default()
        };
        let at = |steps: Option<u32>| {
            validate_request(
                &caps,
                &GenerationRequest {
                    steps,
                    ..base.clone()
                },
            )
        };

        // The advertised count and "the model picks" are both admitted — the common path (the
        // catalog's `defaults.steps` is 8) must not regress into a rejection.
        assert!(at(Some(NATIVE_STEPS)).is_ok());
        assert!(at(None).is_ok());

        // 30 is the case the story names: previously accepted here and silently ignored, refused on
        // candle. 1 and 4 cover the under side, which a FLOOR-shaped key would have admitted.
        for steps in [1u32, 4, 7, 9, 30, 50] {
            let err = at(Some(steps))
                .expect_err("an off-schedule step count must be refused, not silently ignored")
                .to_string();
            assert!(
                err.contains(MODEL_ID) && err.contains(&format!("steps={steps}")),
                "the refusal must name the model and the request: {err}"
            );
            assert!(
                err.contains(&NATIVE_STEPS.to_string()),
                "the refusal must name the legal value: {err}"
            );
        }
    }

    /// sc-19502 — the advertised step surface is DERIVED from the σ table this engine actually runs,
    /// so re-baking the schedule cannot leave a stale advertised count behind.
    ///
    /// The cross-lane half (that candle advertises the same 8) cannot be asserted here: no crate
    /// depends on both `mlx-gen-ltx` and `candle-gen-ltx`, and mlx is macOS-only. SceneWorks owns
    /// that guard, where one catalog entry demonstrably drives both backends.
    #[test]
    fn advertised_steps_are_derived_from_the_baked_schedule() {
        assert_eq!(
            NATIVE_STEPS as usize,
            crate::pipeline::STAGE1_SIGMAS.len() - 1
        );
        assert_eq!(NATIVE_STEPS, 8, "the distilled stage-1 schedule is 8 steps");
        assert_eq!(
            descriptor().capabilities.supported_steps,
            StepSupport::Exact(vec![NATIVE_STEPS]),
            "the descriptor must advertise exactly the baked schedule"
        );
    }

    #[test]
    fn validate_request_conditioning() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(9),
            ..Default::default()
        };
        let img = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        };
        // A single I2V `Reference` is accepted.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![Conditioning::Reference {
                    image: img.clone(),
                    strength: Some(0.8),
                }],
                ..base.clone()
            }
        )
        .is_ok());
        // Unsupported conditioning (e.g. Depth) is rejected.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img.clone() }],
                ..base.clone()
            }
        )
        .is_err());
        // More than one `Reference` is rejected during weight-free preflight (single-image I2V only).
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img.clone(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img.clone(),
                    strength: None,
                },
            ],
            ..base.clone()
        };
        let err = validate_request(&caps, &two).unwrap_err().to_string();
        assert!(err.contains("multiple reference images"), "{err}");

        let control = Conditioning::ControlClip {
            frames: vec![img.clone()],
            mask: vec![img.clone()],
            masking_strength: 0.8,
            start_frame: 0,
            mode: mlx_gen::ReplacementMode::FaceOnly,
        };
        // The full 1–4 ordered replace-person surface is admitted before weights load.
        for reference_count in 1..=4 {
            assert!(validate_request(
                &caps,
                &GenerationRequest {
                    conditioning: vec![
                        control.clone(),
                        Conditioning::MultiReference {
                            images: vec![img.clone(); reference_count],
                        },
                    ],
                    ..base.clone()
                }
            )
            .is_ok());
        }
        for conditioning in [
            vec![control.clone()],
            vec![Conditioning::MultiReference {
                images: vec![img.clone()],
            }],
            vec![
                control.clone(),
                Conditioning::Reference {
                    image: img.clone(),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![img.clone()],
                },
            ],
            vec![
                control.clone(),
                Conditioning::MultiReference {
                    images: vec![img.clone(); 5],
                },
            ],
        ] {
            assert!(validate_request(
                &caps,
                &GenerationRequest {
                    conditioning,
                    ..base.clone()
                }
            )
            .is_err());
        }

        // The render consumes `control_clip()` (the first match), so duplicates must fail rather
        // than silently discard the second after text encoding.
        let err = validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![control.clone(), control],
                ..base
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("at most one ControlClip"), "{err}");
    }

    #[test]
    fn validate_request_rejects_late_conditioning_shape_errors_preflight() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(49), // seven latent frames: accepted indices -7..=6.
            ..Default::default()
        };
        let img = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        };
        let rejects = |conditioning| {
            validate_request(
                &caps,
                &GenerationRequest {
                    conditioning,
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string()
        };

        let err = rejects(vec![Conditioning::Keyframe {
            image: img.clone(),
            frame_idx: 7,
            strength: 1.0,
        }]);
        assert!(err.contains("keyframe latent frame index 7"), "{err}");

        let err = rejects(vec![Conditioning::VideoClip {
            frames: vec![],
            frame_idx: 0,
            strength: 1.0,
        }]);
        assert!(err.contains("video conditioning clip is empty"), "{err}");
        let err = rejects(vec![Conditioning::VideoClip {
            frames: vec![img.clone()],
            frame_idx: -8,
            strength: 1.0,
        }]);
        assert!(err.contains("clip latent frame index -8"), "{err}");

        let err = rejects(vec![
            Conditioning::ControlClip {
                frames: vec![],
                mask: vec![],
                masking_strength: 1.0,
                start_frame: 0,
                mode: mlx_gen::ReplacementMode::FaceOnly,
            },
            Conditioning::MultiReference {
                images: vec![img.clone()],
            },
        ]);
        assert!(err.contains("control clip is empty"), "{err}");
        let err = rejects(vec![
            Conditioning::ControlClip {
                frames: vec![img.clone()],
                mask: vec![],
                masking_strength: 1.0,
                start_frame: 0,
                mode: mlx_gen::ReplacementMode::FaceOnly,
            },
            Conditioning::MultiReference {
                images: vec![img.clone()],
            },
        ]);
        assert!(err.contains("frame count 1 != mask count 0"), "{err}");
        let err = rejects(vec![
            Conditioning::ControlClip {
                frames: vec![img.clone()],
                mask: vec![img.clone()],
                masking_strength: 1.0,
                start_frame: 7,
                mode: mlx_gen::ReplacementMode::FaceOnly,
            },
            Conditioning::MultiReference { images: vec![img] },
        ]);
        assert!(err.contains("replace_person latent frame index 7"), "{err}");
    }

    #[test]
    fn frames_to_images_splits_per_frame() {
        // (F=2, H=1, W=2, 3): each frame = 6 bytes.
        let data: Vec<u8> = (0..12).collect();
        let frames = Array::from_slice(&data, &[2, 1, 2, 3]);
        let imgs = frames_to_images(&frames).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!((imgs[0].width, imgs[0].height), (2, 1));
        assert_eq!(imgs[0].pixels, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(imgs[1].pixels, vec![6, 7, 8, 9, 10, 11]);
    }
}
