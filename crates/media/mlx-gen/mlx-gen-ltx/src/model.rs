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

use mlx_gen::gen_core::ltx_checkpoint::{layout_for_declared_version, LtxCheckpointLayout};
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
use crate::enhance::{self, EnhanceConfig, SampleParams};
use crate::gemma::{GemmaConfig, GemmaModel, GemmaQuant};
use crate::image_crf::{condition_image_for_checkpoint, default_image_recompress};
use crate::pipeline::{
    decode_audio_track, decode_to_frames_with_tiling, generate_av_latents,
    generate_av_latents_iclora, preprocess_conditioning_clip, StageClip, StageKeyframe,
    STAGE1_SIGMAS, STAGE2_SIGMAS,
};
use crate::positions::{compute_audio_frames, create_audio_position_grid, create_position_grid};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{AvDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Public provider id: `"ltx_2_3"`.
pub const MODEL_ID: &str = "ltx_2_3";

/// The `model_version` this engine's checkpoints declare, used to resolve generation params
/// (sc-18759 — see [`crate::params`]). This crate loads only `ltx_2_3` checkpoints today (the
/// `ltx_2_5` engine descriptor is sc-18778), so the literal is correct as written, not a
/// placeholder: once split-checkpoint loading (sc-18757) threads a loaded checkpoint's declared
/// `model_version` onto [`Ltx`], swap this constant for that field.
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
    tokenizer: LtxTokenizer,
    // sc-10976 (epic 10975): the two GIANTS — the ~24 GB Gemma-3-12B text encoder and the AvDiT — are
    // NOT held resident. They are built on demand inside `generate` (load → use → drop), mirroring
    // Wan's `root`-only struct (`mlx-gen-wan/src/model.rs`): the TE is freed (+ `clear_cache()`) before
    // the DiT materializes, so peak ≈ max(TE, DiT) instead of the sum. The lazy-build inputs live here.
    root: PathBuf,
    config: LtxConfig,
    dit_prec: Precision,
    adapters: Vec<AdapterSpec>,
    gemma_dir: PathBuf,
    gemma_quant: Option<GemmaQuant>,
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
        tokenizer,
        root: root.clone(),
        config,
        dit_prec,
        adapters: spec.adapters.clone(),
        gemma_dir,
        gemma_quant,
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

impl Ltx {
    /// Build the AudioVideo Gemma-3-12B text encoder from the resolved `gemma_dir` + the snapshot's
    /// `connector.safetensors` (sc-10976). Called per-generate and dropped (+ `clear_cache()`) before
    /// [`build_transformer`](Self::build_transformer) — mirroring Wan's `encode_text_staged` — so the
    /// ~24 GB TE and the DiT never co-reside. bf16 activations (the reference TE dtype); the backbone is
    /// selectively quantized iff `gemma_quant` is set (`None` ⇒ dense bf16, the default `…-bf16`
    /// snapshot). Identical construction to the pre-sc-10976 `load()`, only deferred.
    fn build_text_encoder(&self) -> Result<LtxTextEncoder> {
        let gemma_w = Weights::from_dir(&self.gemma_dir)?;
        let connector_w = Weights::from_file(self.root.join("connector.safetensors"))?;
        LtxTextEncoder::from_weights_av(
            &gemma_w,
            &connector_w,
            GemmaConfig::gemma_3_12b(),
            self.gemma_quant,
            &self.config,
            // The DiT's own checkpoint geometry re-targeted at bf16 activations: the connector and
            // the feature-extractor Linear are packed on an LTX-2.5 `q4`/`q8` tier (sc-18775) and
            // dense on LTX-2.3, decided per tensor by whether `.scales` is present — so one
            // `Precision` serves both without a per-generation branch.
            self.dit_prec.with_compute_dtype(Dtype::Bfloat16),
        )
    }

    /// Build the AvDiT from the snapshot's `transformer.safetensors` at the load-time precision,
    /// applying any LoRA/LoKr adapters (sc-2687/sc-2393). Called per-generate AFTER the text encoder is
    /// freed (sc-10976), so the 24 GB TE and the DiT never co-reside. LoRA is a forward-time residual
    /// over the (quantized/dense) base; `pass_scales` carries one strength per distilled denoise pass.
    fn build_transformer(&self) -> Result<AvDiT> {
        let transformer_w = Weights::from_file(self.root.join("transformer.safetensors"))?;
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
        let enhanced = self.maybe_enhance(&te, req);
        let prompt = enhanced.as_deref().unwrap_or(req.prompt.as_str());
        let (ids, mask) = self.tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
        let (video_ctx, audio_ctx) = te.encode_av(&ids, &mask)?;
        // Force the encode to complete before `te` (the ~24 GB Gemma) drops at end of scope — otherwise
        // the lazy graph would keep the TE weights alive into the DiT phase, defeating the staging.
        mlx_rs::transforms::eval([&video_ctx, &audio_ctx])?;
        Ok((enhanced, video_ctx, audio_ctx))
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
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);
        let audio_pos = create_audio_position_grid(1, Self::audio_frames(req));

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
        let (video_latents, audio_latents) = if !video_clips.is_empty() {
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
                &mut on_step,
            )?
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
                &mut on_step,
            )?
        };

        // sc-10976: force the denoise output to materialize, then drop the DiT + free the allocator
        // cache so the VAE + audio decode run in the freed footprint (the AvDiT is the denoise peak and
        // nothing downstream needs it). Mirrors Wan's scope-block release before the VAE decode.
        mlx_rs::transforms::eval([&video_latents, &audio_latents])?;
        finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Denoise, || Ok(()))?;
        drop(transformer);
        mlx_rs::memory::clear_cache();

        on_progress(Progress::Decoding);
        let selected_tiling = crate::memory_strategy::decode_tiling(req)?;
        let frames = decode_to_frames_with_tiling(
            &self.vae,
            &video_latents,
            &req.cancel,
            selected_tiling.as_ref(),
        )?;
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
            fps: req.fps.unwrap_or(24),
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
            enhance::enhance(
                te.gemma(),
                &self.tokenizer,
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

/// Capability-driven request validation (weight-free, so it's unit-testable without a load): the
/// shared capability floor ([`Capabilities::validate_request`] — size range, count, unsupported
/// guidance/negative/true_cfg/sampler/scheduler, only advertised conditioning kinds) plus the LTX
/// model-specific constraints: non-empty prompt, 64-aligned width/height (stage-1 runs at //2//32),
/// `num_frames = 1 + 8·k`, and all weight-free conditioning cardinality/shape/index constraints.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("ltx_2_3: prompt must not be empty".into()));
    }
    caps.validate_request(MODEL_ID, req)?;
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
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
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
        let owned;
        let req = match enhanced {
            Some(prompt) => {
                owned = GenerationRequest {
                    prompt,
                    ..req.clone()
                };
                &owned
            }
            None => req,
        };
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let af = Self::audio_frames(req) as i32;
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

#[cfg(test)]
mod tests {
    use super::*;
    // sc-19502: the derived stage-1 step count the descriptor advertises.
    use crate::pipeline::NATIVE_STEPS;

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
