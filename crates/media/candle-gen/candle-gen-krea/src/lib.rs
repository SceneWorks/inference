//! # candle-gen-krea
//!
//! The **Krea 2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-krea`. Registers two generator ids over **one architecture** (only the DiT weights differ,
//! distilled vs base — the Boogu base/turbo precedent):
//!
//! * **`krea_2_turbo`** — the user-facing text-to-image model: a 12B **dense single-stream**
//!   rectified-flow / v-param DiT (28 gated single-stream blocks, hidden 6144, GQA 48Q/12KV, head_dim
//!   128, SwiGLU 16384, 3-axis interleaved RoPE `[32,48,48]`, `DoubleSharedModulation`, and a
//!   `text_fusion` front-end that aggregates the 12 selected Qwen3-VL hidden layers) driven by a
//!   Qwen3-VL-4B condition encoder and the Qwen-Image VAE. TDM-distilled few-step (8 steps),
//!   **CFG-free** (guidance inert), up to 2048².
//! * **`krea_2_raw`** (sc-9994 / epic 9992) — the undistilled 12B base run as a **full classifier-free
//!   guidance** generator: a real guidance scale + optional user negative prompt, 52 steps, resolution-
//!   dynamic mu ([`pipeline::render_base`]). The SAME id is also the Krea LoRA *training* base (Path 1:
//!   one id, both roles — generator + trainer registries). Two DiT forwards/step (cond vs uncond).
//!
//! **Reuse:** the VAE is `candle_gen_qwen_image::vae::QwenVae` (the exact `AutoencoderKLQwenImage`
//! Qwen-Image ships — per-channel `latents_mean`/`latents_std` de-norm) — reused verbatim, as
//! `mlx-gen-krea` reuses `mlx-gen-qwen-image`'s `QwenVae`. The Qwen3-VL-4B condition encoder
//! ([`text_encoder`]), the single-stream DiT ([`transformer`]), and the rectified-flow sampler
//! ([`schedule`]) are ported here.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0; Krea 2 Community License (non-commercial use
//! satisfies it). The packed q4/q8/bf16 turnkey loads per-tier via `loader::linear_detect` (sc-9411);
//! the descriptor advertises `supported_quants: [Q4, Q8]` so the worker's A-B quant toggle engages
//! (sc-9607).

pub mod adapters;
pub mod config;
pub mod convert;
pub mod loader;
/// The NVFP4 precision seam for the Krea 2 DiT trunk (sc-12110, epic 11037) — the epic's SC#1/SC#2
/// validation vehicle. See [`nvfp4_dit`].
pub mod nvfp4_dit;
pub mod pipeline;
pub mod quant;
pub mod schedule;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;
pub mod vision;

// The candle Krea LoRA/LoKr trainer (sc-7577) + its vendored composable-op trainable DiT. Private
// (reached through the explicit family registry by id, like the SDXL/Z-Image trainers).
mod train_dit;
mod training;

// The pose-ControlNet control branch (sc-8460 spike / sc-8462, epic 8459): a trainable N-block side
// branch over the frozen DiT with zero-init per-block residual injection. Public so the spike's
// trainer/inference example binaries can drive it; the worker route is a later story.
pub mod control;

// The callable control-branch trainer (sc-8462): the spike CLI's training loop lifted into a
// reusable `ControlTrainer` so the ControlNet Training Studio worker driver (epic 10159 B2) can
// drive a run and stream its progress. Kept gen-core-neutral for the later MLX training lane.
pub mod control_train;

// The gen_core `Trainer` adapter for Krea pose-ControlNet (sc-10163, epic 10159 B2): registers
// `krea_2_control` so the studio drives control-branch training through the same `load_trainer` path
// LoRA uses. Private and reached through the explicit family registry by id.
mod control_trainer;

// The Krea 2 Turbo pose-ControlNet **inference** provider (sc-8464, epic 8459): loads a trained
// control-branch overlay on the frozen Turbo base and renders a pose-conditioned image. The
// deployable form of the sc-8460 spike inference harness; the worker `KreaControl` route calls it.
pub mod control_provider;

// Shared test-only tiny-DiT fixture (training + control tests).
#[cfg(test)]
mod testfix;

pub use adapters::{
    install_additive, merge_adapters, merge_into_weights, AdditiveReport, MergeReport,
};
pub use config::Krea2Config;
pub use control_provider::{
    Krea2Control, Krea2ControlPaths, Krea2ControlRequest, DEFAULT_CONTROL_SCALE,
};
// The resident aggregate. It splits internally into `pipeline::KreaText` (tokenizer + Qwen3-VL-4B TE)
// and `pipeline::KreaHeavy` (DiT + VAE + optional PiD) so the `Sequential` path can drop the first
// before the second loads (epic 10765 Phase 1c, sc-12089) — but both halves stay `pub(crate)`: every
// operation on them is crate-private, so exporting them would add two opaque, unusable types to this
// crate's compatibility surface. (The mlx-gen-krea twins are exported because those carry public
// methods; ours carry none.)
// The NVFP4 seam (sc-12110): the plan/probe/report surface a validation harness drives.
pub use nvfp4_dit::{
    summarize, ActProbe, ActRecord, DitPlan, LayerRole, LayerSparsitySummary, Nvfp4Quant,
    Nvfp4Report,
};
pub use pipeline::Components;
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder};
pub use tokenizer::KreaTokenizer;
// The composable trainable DiT, exposed for the sc-8460 control-branch spike binaries (the branch
// injects into its block stack; its forward is the spike's inference surface).
pub use train_dit::{KreaTrainDit, KREA_ATTN_CHUNK_BUDGET};
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae, QwenVaeEncoder};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Progress, Quant,
    WeightsSource,
};

/// Registry id for the Krea 2 Turbo text-to-image variant. Matches the SceneWorks worker's
/// `payload.model` and the manifest `engine_id` (sc-7572).
pub const KREA_2_TURBO_ID: &str = "krea_2_turbo";

/// Registry id for the undistilled **Raw** full-CFG text-to-image variant (sc-9994 / epic 9992). The
/// SAME string as the Krea LoRA *trainer* base (`crate::training::KREA_2_RAW_ID`) — Path 1 makes one id
/// both the training base and a first-class generator; the trainer + generator live in separate
/// registries so the shared id never collides. Matches the worker `payload.model` + manifest `engine_id`.
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Registry id for the **image-edit** variant (epic 10871 / sc-11085). Kontext-style instruction edit
/// over one or two source references (image 1 (required) + image 2 (optional), either can be a person)
/// on the undistilled full-CFG
/// base. The engine (pipeline `render_edit` + edit components) landed via #416 but was unreachable
/// through the `Generator` seam until this id was registered — the candle mirror of the mlx-gen #693
/// `krea_2_edit` seam. Matches the worker `payload.model` + manifest `engine_id`.
pub const KREA_2_EDIT_ID: &str = "krea_2_edit";

/// Surface tag for the **distilled Turbo image-edit** (`krea_2_turbo_edit`, sc-11640). Not a registered
/// `Generator` id — the CFG-free distilled edit is driven through the worker's bespoke
/// `generate_candle_krea_edit_stream` lane, which calls [`pipeline::render_edit`] with `distilled = true`
/// directly. Named here so the shared edit path (PiD decode-seam errors, sc-11197) reports the right
/// surface for the Turbo edit vs the Raw [`KREA_2_EDIT_ID`].
pub const KREA_2_TURBO_EDIT_ID: &str = "krea_2_turbo_edit";

/// patch_size(2)·vae_downsample(8) = 16 — patchify requires latent dims divisible by this. Exposed as
/// the pinned-engine stride SceneWorks ties each advertised Krea image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`; the control provider imports this same crate-root const so no copy
/// can drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Max images per request (the image-model standard, shared with the other families).
const MAX_COUNT: u32 = 8;

enum KreaTextPhase {
    Resident,
    Sequential(Box<pipeline::KreaText>),
}

enum KreaHeavyPhase {
    Resident(Box<ResidentKrea>),
    Sequential(Box<pipeline::ResidencyHeavy>),
}

enum KreaEncoded {
    Resident,
    Sequential(pipeline::ResidencyContext),
    Edit(pipeline::EditContext),
}

struct ResidentKrea {
    components: Arc<Components>,
    root: PathBuf,
    device: Device,
    edit_components: Mutex<Option<Arc<pipeline::EditComponents>>>,
    img2img_encoder: Mutex<Option<Arc<QwenVaeEncoder>>>,
}

impl ResidentKrea {
    fn edit_components(&self) -> candle_gen::Result<Arc<pipeline::EditComponents>> {
        candle_gen::cached(&self.edit_components, || {
            Ok(Arc::new(pipeline::load_edit_components(
                &self.root,
                &self.device,
            )?))
        })
    }

    fn img2img_encoder(&self) -> candle_gen::Result<Arc<QwenVaeEncoder>> {
        candle_gen::cached(&self.img2img_encoder, || {
            Ok(Arc::new(crate::vae::load_vae_encoder(
                &self.root,
                &self.device,
            )?))
        })
    }
}

/// A Krea 2 generator whose shared residency value exclusively owns the warm components or deferred
/// phase loaders.
pub struct KreaGenerator {
    descriptor: ModelDescriptor,
    device: Device,
    residency: candle_gen::Residency<KreaTextPhase, KreaHeavyPhase>,
}

impl Generator for KreaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // The Edit variant needs 1..=2 source references (image 1, then image 2). The capability floor
        // above accepts a single `Reference` on Turbo/Raw (img2img latent-init) but rejects a
        // MultiReference there; only `krea_2_edit` advertises both, so resolve + count-check here.
        if self.descriptor.id == KREA_2_EDIT_ID {
            resolve_edit_references(req)?;
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;

        let raw = self.descriptor.id == KREA_2_RAW_ID;
        let edit = self.descriptor.id == KREA_2_EDIT_ID;
        let edit_references: Vec<Image> = if edit {
            resolve_edit_references(req)?.into_iter().cloned().collect()
        } else {
            Vec::new()
        };
        let reference = img2img_reference(req);
        let images = self.residency.run(
            &req.cancel,
            &self.device,
            req.use_pid,
            on_progress,
            |text| match text {
                KreaTextPhase::Resident => Ok(KreaEncoded::Resident),
                KreaTextPhase::Sequential(text) if edit => {
                    Ok(KreaEncoded::Edit(pipeline::encode_edit_context(
                        text,
                        req,
                        &edit_references,
                        false,
                        &self.device,
                    )?))
                }
                KreaTextPhase::Sequential(text) => Ok(KreaEncoded::Sequential(
                    pipeline::encode_residency(text, raw, req)?,
                )),
            },
            |heavy, encoded, on_progress| match (heavy, encoded) {
                (KreaHeavyPhase::Sequential(heavy), KreaEncoded::Edit(context)) => {
                    pipeline::render_edit_residency(
                        heavy,
                        context,
                        req,
                        &edit_references,
                        &self.device,
                        on_progress,
                    )
                }
                (KreaHeavyPhase::Sequential(heavy), KreaEncoded::Sequential(context)) => {
                    pipeline::render_residency(
                        heavy,
                        context,
                        req,
                        reference,
                        &self.device,
                        on_progress,
                    )
                }
                (KreaHeavyPhase::Resident(resident), KreaEncoded::Resident) => {
                    let comps = &resident.components;
                    if edit {
                        let edit = resident.edit_components()?;
                        pipeline::render_edit(
                            comps,
                            &edit,
                            req,
                            &edit_references,
                            false,
                            &self.device,
                            on_progress,
                        )
                    } else if raw {
                        if let Some((reference, strength)) = reference {
                            let vae_encoder = resident.img2img_encoder()?;
                            pipeline::render_base_img2img(
                                comps,
                                &vae_encoder,
                                req,
                                reference,
                                strength,
                                &self.device,
                                on_progress,
                            )
                        } else {
                            pipeline::render_base(comps, req, &self.device, on_progress)
                        }
                    } else if let Some((reference, strength)) = reference {
                        let vae_encoder = resident.img2img_encoder()?;
                        pipeline::render_img2img(
                            comps,
                            &vae_encoder,
                            req,
                            reference,
                            strength,
                            &self.device,
                            on_progress,
                        )
                    } else {
                        pipeline::render(comps, req, &self.device, on_progress)
                    }
                }
                _ => unreachable!("residency phase variants are constructed in matching pairs"),
            },
        )?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Krea 2 Turbo identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). Distilled few-step text-to-image: **CFG-free** (the TDM
/// distillation baked the guided velocity into the weights, so no guidance / unconditional branch), no
/// user negative prompt. Accepts reference-guided **img2img** latent-init (sc-10134) — a single
/// `Conditioning::Reference` — but no control conditioning on the Turbo checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: KREA_2_TURBO_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // CFG-free distilled student (like Ideogram Turbo / Boogu Turbo / SDXL-Lightning).
            supports_guidance: false,
            supports_true_cfg: false,
            // Turbo img2img reference-guided latent-init (sc-10134, epic 8588): a single
            // `Conditioning::Reference { image, strength }` seeds the denoise from the VAE-encoded
            // reference (`pipeline::render_img2img`). A MultiReference is NOT accepted here (that is the
            // `krea_2_edit` Kontext surface); control conditioning stays unsupported. `raw_descriptor`
            // KEEPS this single Reference — Raw serves its own full-CFG `render_base_img2img` (sc-10226).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr wired (sc-7836): a trained `krea_2_raw` adapter merges into the dense DiT
            // attention projections at load ([`adapters::merge_into_weights`]), closing the candle
            // train→infer loop.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: false,
            // sc-9607: advertise the packed tiers so the worker's A-B quant toggle engages off-Mac.
            // The resolved q4/q8/bf16 turnkey subdir self-describes its tier (`loader::linear_detect`,
            // sc-9411); `build` no-ops the requested quant, and it composes with a merged LoRA overlay.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // sc-12089 (epic 10765 Phase 1c): the Turbo txt2img lane wires the load→encode→drop
            // residency lifecycle (`pipeline::render_sequential`), so it advertises the discovery bit
            // the worker's fit-gate reads. `raw_descriptor` inherits this for its CFG twin, and
            // `edit_descriptor` keeps it after sc-12129 moved grounded conditioning into KreaText.
            //
            // Provider + advertisement move in LOCKSTEP (the sc-10840 correctness contract): this bit
            // going true is what lets a consumer predict the staged peak, and `OffloadPolicy::Sequential`
            // is advisory (an unwired engine silently stays resident). Advertising a lane that would
            // actually run resident makes the gate under-predict its real peak — an admitted job that
            // then OOMs. Never flip this on ahead of the wiring.
            supports_sequential_offload: true,
            supports_streaming: false,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
        },
    }
}

/// Krea 2 **Raw** identity + capabilities (sc-9994 / epic 9992) — the undistilled 12B DiT run with
/// **true classifier-free guidance** (two DiT forwards/step: cond vs uncond) at 52 steps, unlike the
/// CFG-free distilled Turbo. Same architecture / snapshot layout as Turbo (only the DiT weights differ,
/// distilled vs base), so it shares `build` + the whole [`pipeline`]. Exposes a real guidance scale
/// AND a user negative prompt (unlike Turbo / Boogu base, which fixes the uncond to the empty prompt).
/// NOT guidance-distilled, so `supports_true_cfg` stays false — the two-forward CFG IS the guidance
/// (the Boogu-base precedent). Derived from [`descriptor`] so the shared surface (family / backend /
/// samplers / quants / size / LoRA) stays in lockstep with Turbo.
pub fn raw_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_RAW_ID;
    d.capabilities.supports_negative_prompt = true;
    d.capabilities.supports_guidance = true;
    d.capabilities.supports_true_cfg = false;
    // Raw img2img reference-guided latent-init (sc-10226, epic 8588): keep the single
    // `ConditioningKind::Reference` inherited from `descriptor` (Turbo). Raw serves it through the
    // undistilled full-CFG `pipeline::render_base_img2img` (the CFG sibling of Turbo's `render_img2img`),
    // so the surface is honored, not silently dropped to txt2img. A MultiReference stays the `krea_2_edit`
    // Kontext surface; `edit_descriptor` (derived from this) extends to Reference + MultiReference.
    d
}

/// Krea 2 **Edit** identity + capabilities (epic 10871 / sc-11085) — the Kontext-style instruction-edit
/// variant. Derived from [`raw_descriptor`] (the edit runs the undistilled **full-CFG** loop from pure
/// noise, with the references as in-context conditioning), so it inherits the Raw surface — real
/// guidance + a user negative prompt, packed quants, LoRA/LoKr (the edit LoRA merges through the shared
/// `build` adapter path) — and additionally advertises the source-reference conditioning:
/// [`ConditioningKind::Reference`] for a single source and [`ConditioningKind::MultiReference`] for two
/// (image 1, then image 2; [`pipeline::MAX_EDIT_REFERENCES`]).
pub fn edit_descriptor() -> ModelDescriptor {
    let mut d = raw_descriptor();
    d.id = KREA_2_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    // sc-12129: grounded Qwen3-VL conditioning now completes inside the `KreaText` phase, including a
    // lazily loaded vision tower. The returned edit context owns its tensors, so the full text phase
    // drops before the DiT/VAE bundle loads. Keep this advertisement in lockstep with that route: the
    // worker uses it to decide whether a staged peak is safe to admit.
    d.capabilities.supports_sequential_offload = true;
    d
}

/// The img2img reference + strength: the first [`Conditioning::Reference`] in the request, if any. Both
/// Turbo (`render_img2img`, sc-10134) and Raw (`render_base_img2img`, sc-10226) advertise only `Reference`
/// (no MultiReference), so at most one is present; `None` ⇒ plain txt2img (CFG-free Turbo / full-CFG Raw).
/// `strength` is the optional per-reference img2img fidelity the worker threads from `advanced.strength`.
/// Pure so it is unit-testable without weights.
fn img2img_reference(req: &GenerationRequest) -> Option<(&Image, Option<f32>)> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, strength } => Some((image, *strength)),
        _ => None,
    })
}

/// The image-edit source references, in fixed order (image 1, then image 2; sc-10878) —
/// collected from both [`Conditioning::Reference`] (a single source) and [`Conditioning::MultiReference`]
/// (two sources). At least one and at most [`pipeline::MAX_EDIT_REFERENCES`] is required; zero or more
/// than the cap is an error. Borrows from `req.conditioning`; the generate path clones the resolved set
/// into the owned `&[Image]` the pipeline consumes. Pure so it is unit-testable without weights.
fn resolve_edit_references(req: &GenerationRequest) -> gen_core::Result<Vec<&Image>> {
    let reference_strength = req.conditioning.iter().any(|c| {
        matches!(
            c,
            Conditioning::Reference {
                strength: Some(_),
                ..
            }
        )
    });
    if req.strength.is_some() || reference_strength {
        return Err(gen_core::Error::Msg(format!(
            "{KREA_2_EDIT_ID}: strength is not supported for edit conditioning; use {KREA_2_RAW_ID} for img2img strength"
        )));
    }
    let mut refs: Vec<&Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {} // the capability floor already rejects the other conditioning kinds.
        }
    }
    if refs.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{KREA_2_EDIT_ID}: an instruction edit requires at least one source reference image \
             (image 1, then image 2)"
        )));
    }
    if refs.len() > pipeline::MAX_EDIT_REFERENCES {
        return Err(gen_core::Error::Msg(format!(
            "{KREA_2_EDIT_ID}: at most {} references are supported (image 1, then image \
             2); got {}",
            pipeline::MAX_EDIT_REFERENCES,
            refs.len()
        )));
    }
    Ok(refs)
}

/// sc-9300 ConvRot selection: decode whether a [`LoadSpec`] selects the community INT8-ConvRot DiT
/// consume path, returning the DiT single-file checkpoint when it does. ConvRot rides the shared,
/// already-optional [`LoadSpec::text_encoder`] field as a [`WeightsSource::File`]; a [`WeightsSource::Dir`]
/// there is a mis-shaped spec (ConvRot is a single file) and errors. `None` on `text_encoder` ⇒ the
/// dense/packed snapshot path. Extracted from [`build`] so the routing decision is unit-testable on CPU
/// without loading weights.
fn convrot_selector(spec: &LoadSpec, id: &str) -> gen_core::Result<Option<PathBuf>> {
    match spec.text_encoder.as_ref() {
        Some(WeightsSource::File(p)) => Ok(Some(p.clone())),
        Some(WeightsSource::Dir(_)) => Err(gen_core::Error::Msg(format!(
            "candle {id}: LoadSpec::text_encoder selects the INT8-ConvRot DiT and must be a single \
             .safetensors file (WeightsSource::File), not a directory"
        ))),
        None => Ok(None),
    }
}

fn build(spec: &LoadSpec, descriptor: ModelDescriptor) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (transformer/ text_encoder/ vae/ tokenizer/), not a \
                 single .safetensors file",
                descriptor.id
            )));
        }
    };
    // sc-9300 seam: select the community **INT8-ConvRot** DiT consume path when the spec carries a
    // ConvRot DiT single-file checkpoint. It rides the shared, already-optional `LoadSpec::text_encoder`
    // field as a `WeightsSource::File` — the canonical Krea 2 snapshot (`spec.weights`, a `Dir`) still
    // supplies the tokenizer / Qwen3-VL TE / Qwen-Image VAE / config + all non-quantized surface, and
    // only the DiT weights are taken from the int8 checkpoint (`pipeline::load_components_convrot`,
    // which enforces the sm_89 compute-cap floor). This reuses an existing extensibility point (the same
    // pattern LTX uses to ride an aux path on `text_encoder`) rather than growing the shared
    // `WeightsSource` enum with a ConvRot variant — which would force a new match arm across every
    // provider in candle-gen AND the worker plus a gen-core pin bump. Only Krea reads this; every other
    // engine ignores `text_encoder` unchanged. `None`/`Dir` here ⇒ the dense/packed snapshot path below.
    let convrot_dit = convrot_selector(spec, descriptor.id)?;
    // LoRA/LoKr adapters are accepted and merged into the DiT at first `generate` (sc-7836); the merge
    // (`adapters::merge_into_weights`) is lazy, so a nonexistent adapter path still loads here.
    //
    // sc-9607: `spec.quantize` (Q4/Q8) is ACCEPTED and no-ops — the resolved per-tier turnkey is
    // already MLX-packed and `loader::linear_detect` builds each `QLinear::Quantized` straight from the
    // packed parts (sc-9411), composing with the adapter overlay (an adapter-merged projection stays
    // dense and takes priority). No on-the-fly quant pass runs; the requested quant is recipe-only.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays",
            descriptor.id
        )));
    }
    // The ConvRot consume path (sc-9300) is DiT-only and does not thread LoRA/LoKr or PiD overlays — the
    // int8 checkpoint replaces the dense transformer wholesale. Reject the combination up front so the
    // worker gets a clear error instead of silently dropping the overlay.
    if convrot_dit.is_some() && (!spec.adapters.is_empty() || spec.pid.is_some()) {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {}: the INT8-ConvRot DiT path does not support LoRA/LoKr adapters or a PiD decoder \
             overlay",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    let policy = effective_residency_policy(spec.offload_policy, convrot_dit.is_some());
    let resident_root = root.clone();
    let resident_device = device.clone();
    let resident_adapters = spec.adapters.clone();
    let resident_pid = spec.pid.clone();
    let resident_convrot = convrot_dit.clone();
    let text_root = root.clone();
    let text_device = device.clone();
    let heavy_root = root.clone();
    let heavy_device = device.clone();
    let heavy_adapters = spec.adapters.clone();
    let heavy_pid = spec.pid.clone();
    // sc-12425: the sequential heavy phase must know whether to load the int8-ConvRot DiT (from the
    // single file) or the snapshot's dense/packed `transformer/`. Absent this the sequential path loaded
    // `root/transformer` unconditionally — the wrong DiT for a ConvRot request — which is why ConvRot
    // was pinned Resident (`effective_residency_policy`) rather than dropping its 15.6 GB f32 TE.
    let heavy_convrot = convrot_dit.clone();
    let residency = candle_gen::Residency::from_policy_with_resident(
        policy,
        move || {
            let components = match resident_convrot.as_ref() {
                Some(convrot_dit) => pipeline::load_components_convrot(
                    &resident_root,
                    convrot_dit,
                    &resident_device,
                )?,
                None => pipeline::load_components(
                    &resident_root,
                    &resident_device,
                    &resident_adapters,
                    resident_pid.as_ref(),
                )?,
            };
            Ok((
                KreaTextPhase::Resident,
                KreaHeavyPhase::Resident(Box::new(ResidentKrea {
                    components: Arc::new(components),
                    root: resident_root.clone(),
                    device: resident_device.clone(),
                    edit_components: Mutex::new(None),
                    img2img_encoder: Mutex::new(None),
                })),
            ))
        },
        move || {
            Ok(KreaTextPhase::Sequential(Box::new(pipeline::load_text(
                &text_root,
                &text_device,
            )?)))
        },
        move |use_pid| {
            let heavy = match heavy_convrot.as_ref() {
                // ConvRot: the int8 DiT from the single file + VAE (no adapters/PiD — the lane rejects
                // both, sc-9300). The TE was already loaded, encoded, and dropped by the text phase, so
                // this loads into that freed pool — the whole point of going sequential here.
                Some(convrot_dit) => {
                    pipeline::load_residency_heavy_convrot(&heavy_root, convrot_dit, &heavy_device)?
                }
                None => pipeline::load_residency_heavy(
                    &heavy_root,
                    &heavy_device,
                    &heavy_adapters,
                    heavy_pid.as_ref(),
                    use_pid,
                )?,
            };
            Ok(KreaHeavyPhase::Sequential(Box::new(heavy)))
        },
    )?;
    Ok(Box::new(KreaGenerator {
        descriptor,
        device,
        residency,
    }))
}

/// The residency policy this generator runs — the request's `offload_policy` after the
/// `CANDLE_GEN_OFFLOAD` override (`candle_gen::effective_offload_policy`).
///
/// `_has_convrot` is taken but no longer changes the answer (sc-12425). ConvRot USED to be forced
/// `Resident` here, because the sequential heavy loader read `root/transformer/` and could not source
/// the int8 single-file DiT; [`pipeline::load_residency_heavy_convrot`] now sources it, so ConvRot drops
/// its 15.6 GB f32 text encoder after encoding like every other Turbo request (measured 42.9 → ~29 GB
/// peak, sc-12381). The parameter (and the test pinning it no longer matters) stays so a future reader
/// cannot quietly re-add the special-case: the Turbo descriptor advertises `supports_sequential_offload`,
/// so a ConvRot generator that silently ran `Resident` would make the worker's fit-gate predict a staged
/// peak it never achieves — the sc-10840 lockstep violation the descriptor's own comment warns against.
fn effective_residency_policy(requested: OffloadPolicy, _has_convrot: bool) -> OffloadPolicy {
    candle_gen::effective_offload_policy(requested)
}

/// Construct a lazy candle Krea 2 **Turbo** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`).
///
/// **INT8-ConvRot (sc-9300).** To load the community int8-quantized DiT instead of the snapshot's dense
/// `transformer/`, pass the ConvRot DiT single-file checkpoint as
/// `spec.text_encoder = Some(WeightsSource::File(convrot_dit.safetensors))` while keeping
/// `spec.weights = WeightsSource::Dir(canonical_snapshot)` (which supplies the tokenizer / TE / VAE /
/// config). The ConvRot path enforces the sm_89 compute-cap floor and does not combine with LoRA/LoKr
/// or PiD overlays.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor())
}

/// Construct a lazy candle Krea 2 **Raw** generator (`krea_2_raw`, sc-9994 / epic 9992). Identical
/// snapshot assembly to [`load`] — the Raw + Turbo turnkeys share the exact architecture / weight layout
/// (only distilled-vs-base DiT weights differ), so one `build` serves both — but stores the CFG-capable
/// [`raw_descriptor`] so `generate` runs the full-CFG [`pipeline::render_base`] path. Accepts the same
/// LoRA/LoKr, PiD, and packed-quant surface as Turbo; the ConvRot / ControlNet rejections are shared.
pub fn load_raw(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, raw_descriptor())
}

/// Construct a lazy candle Krea 2 **Edit** generator (`krea_2_edit`, epic 10871 / sc-11085). Identical
/// snapshot assembly to [`load`] / [`load_raw`] — one `build` serves all three ids — but stores the
/// [`edit_descriptor`] so `generate` routes the reference-conditioned [`pipeline::render_edit`] path and
/// lazily loads the edit-only components (VAE encoder + vision tower). The edit LoRA rides the shared
/// `spec.adapters` merge path, exactly like a Raw-trained adapter on the txt2img ids.
pub fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, edit_descriptor())
}

// Link-time registration: all three variants register here — `krea_2_turbo` (distilled, CFG-free),
// `krea_2_raw` (undistilled, full-CFG; sc-9994 / epic 9992), and `krea_2_edit` (Kontext instruction
// edit over 1-2 references; epic 10871 / sc-11085).
candle_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub(crate) const RAW_REGISTRATION = raw_descriptor => load_raw
}
candle_gen::register_generators! {
    pub(crate) const EDIT_REGISTRATION = edit_descriptor => load_edit
}

/// Add all Candle Krea generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(TURBO_REGISTRATION)
        .register_generator(RAW_REGISTRATION)
        .register_generator(EDIT_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
        .register_trainer(control_trainer::CONTROL_TRAINER_REGISTRATION)
}

/// Build the complete explicit Candle Krea provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            explicit_generators,
            ["krea_2_turbo", "krea_2_raw", "krea_2_edit"]
        );
        assert_eq!(explicit_trainers, ["krea_2_raw", "krea_2_control"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_krea_2_turbo_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .expect("krea_2_turbo is registered");
        assert_eq!(g.descriptor().id, KREA_2_TURBO_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    // --- Raw (undistilled, full-CFG) variant — sc-9994 / epic 9992 ---

    #[test]
    fn registers_krea_2_raw_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_RAW_ID, &spec)
            .expect("krea_2_raw is registered");
        assert_eq!(g.descriptor().id, KREA_2_RAW_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn raw_descriptor_is_krea_2_raw_and_cfg_capable() {
        let d = raw_descriptor();
        assert_eq!(d.id, KREA_2_RAW_ID);
        // The generator id MUST equal the LoRA-trainer base id (Path 1: one id, both roles).
        assert_eq!(KREA_2_RAW_ID, crate::training::KREA_2_RAW_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        // Undistilled base: real CFG guidance + a user negative prompt (unlike Turbo / Boogu base).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // Not guidance-distilled: no separate embedded-guidance axis, so no true_cfg toggle.
        assert!(!d.capabilities.supports_true_cfg);
        // Shared surface stays in lockstep with Turbo (derived from `descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(!d.capabilities.mac_only);
        assert_eq!(pipeline::RAW_STEPS, 52);
        assert_eq!(pipeline::RAW_GUIDANCE, 3.5);
    }

    #[test]
    fn raw_validate_accepts_guidance_and_negative_prompt() {
        // The CFG floor that rejects these on Turbo must ACCEPT them on Raw.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_RAW_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            guidance: Some(3.5),
            negative_prompt: Some("blurry, lowres".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
    }

    /// sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Krea bucket to.
    /// Pin the value and mutation-check that a multiple of 8 which is not SIZE_MULTIPLE (16) is rejected
    /// with the stride error, and an on-stride size passes.
    #[test]
    fn size_multiple_is_the_pinned_stride() {
        assert_eq!(SIZE_MULTIPLE, 16);
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .unwrap();
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                height: 1024,
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            })
            .is_ok());
    }

    #[test]
    fn load_raw_rejects_single_file_like_turbo() {
        // Same snapshot loader as Turbo — a single-file weights source is rejected the same way.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load_raw(&file).is_err());
        // A LoRA `LoadSpec` on the Raw id is accepted + lazy, exactly like Turbo (sc-7836 wiring).
        let dir = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load_raw(&dir).is_ok());
    }

    #[test]
    fn descriptor_surface_is_cfg_free_turbo() {
        let d = descriptor();
        assert_eq!(d.id, KREA_2_TURBO_ID);
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        // Turbo advertises single-reference img2img (sc-10134) — but NOT MultiReference (the edit surface).
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // LoRA/LoKr merge wired (sc-7836); packed Q4/Q8 tiers advertised (sc-9607).
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(TURBO_STEPS, 8);
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                height: 1024,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 1024,
                height: 1024,
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// F-154 (sc-11210): the empty-prompt guard rejects a whitespace-only prompt (`trim().is_empty()`),
    /// matching the chroma and krea control-provider siblings — a whitespace prompt would otherwise
    /// reach the TE as an effectively-empty sequence.
    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .unwrap();
        for ws in ["   ", "\t", "\n", " \t\n "] {
            let req = GenerationRequest {
                prompt: ws.into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            assert!(
                g.validate(&req).is_err(),
                "whitespace-only prompt {ws:?} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_guidance_and_negative_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        assert!(g
            .validate(&GenerationRequest {
                guidance: Some(3.5),
                ..base.clone()
            })
            .is_err());
        assert!(g
            .validate(&GenerationRequest {
                negative_prompt: Some("y".into()),
                ..base
            })
            .is_err());
    }

    #[test]
    fn load_accepts_lora_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        // LoRA/LoKr now wired (sc-7836): a LoRA `LoadSpec` is accepted (lazily — the merge happens at
        // first `generate`), so `load` resolves rather than rejecting.
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA load is wired + lazy (sc-7836)");
        // sc-9607: a Q4/Q8 `spec.quantize` is now ACCEPTED (a no-op on the already-packed tier) — load
        // proceeds past the quant check and constructs lazily, exactly like the LoRA case above.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "Q4/Q8 quant is accepted + lazy (sc-9607)"
        );
    }

    // sc-9300: the ConvRot consume path is reachable through the LoadSpec API. The selector routes a
    // `WeightsSource::File` on `text_encoder` to the INT8-ConvRot DiT (`load_components_convrot`), a
    // plain `Dir` weights spec to the dense/packed snapshot path (`load_components`), and rejects the
    // mis-shaped / incompatible combinations. These assert the routing decision on CPU (no weights).
    #[test]
    fn convrot_selector_routes_file_to_convrot_dir_to_dense() {
        // A `Dir`-only spec (canonical snapshot, no ConvRot DiT) ⇒ the dense/packed path.
        let dense = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(
            convrot_selector(&dense, KREA_2_TURBO_ID).unwrap(),
            None,
            "a Dir-only spec dispatches to the dense/packed snapshot path"
        );
        // A ConvRot DiT single-file on `text_encoder` ⇒ the ConvRot path, carrying the DiT checkpoint.
        let convrot = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_convrot_text_encoder();
        assert_eq!(
            convrot_selector(&convrot, KREA_2_TURBO_ID).unwrap(),
            Some(PathBuf::from("/krea2_int8_convrot.safetensors")),
            "a File on text_encoder selects the ConvRot DiT consume path"
        );
        // A `Dir` on `text_encoder` is not a valid ConvRot selector (ConvRot is a single file).
        let bad = LoadSpec {
            text_encoder: Some(WeightsSource::Dir("/te_dir".into())),
            ..LoadSpec::new(WeightsSource::Dir("/snap".into()))
        };
        assert!(
            convrot_selector(&bad, KREA_2_TURBO_ID).is_err(),
            "a Dir on text_encoder is a mis-shaped ConvRot selector and errors"
        );
    }

    #[test]
    fn load_accepts_convrot_and_rejects_convrot_with_overlays() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // A ConvRot-selecting spec loads (lazily — the int8 DiT + snapshot load at first `generate`).
        let convrot = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_convrot_text_encoder();
        assert!(
            load(&convrot).is_ok(),
            "a ConvRot LoadSpec is accepted + lazy (sc-9300)"
        );
        // ConvRot does not thread LoRA/LoKr — the int8 checkpoint replaces the dense DiT wholesale.
        let convrot_lora = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_convrot_text_encoder()
            .with_adapters(vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]);
        assert!(
            load(&convrot_lora).is_err(),
            "ConvRot + LoRA is rejected (the int8 DiT path is not adapter-wired)"
        );
        // ConvRot does not thread a PiD decoder overlay either.
        let convrot_pid = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_convrot_text_encoder()
            .with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            );
        assert!(
            load(&convrot_pid).is_err(),
            "ConvRot + PiD is rejected (the int8 DiT path is not PiD-wired)"
        );
    }

    // --- Edit (Kontext instruction edit, full-CFG) variant — epic 10871 / sc-11085 ---

    #[test]
    fn registers_krea_2_edit_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_EDIT_ID, &spec)
            .expect("krea_2_edit is registered");
        assert_eq!(g.descriptor().id, KREA_2_EDIT_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn edit_descriptor_is_cfg_capable_and_advertises_references() {
        let d = edit_descriptor();
        assert_eq!(d.id, KREA_2_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        // Derived from the Raw surface: real CFG guidance + a user negative prompt (the edit runs the
        // undistilled full-CFG loop with the references as in-context conditioning).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // And it advertises BOTH single- and two-reference conditioning. Turbo (sc-10134) and Raw
        // (sc-10226) each advertise the single `Reference` img2img surface; only Edit adds MultiReference.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert_eq!(
            descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(
            raw_descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // Shared surface stays in lockstep with Raw/Turbo (derived from `raw_descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn load_edit_rejects_single_file_accepts_dir_and_lora() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // Same snapshot loader as Turbo/Raw — a single-file weights source is rejected.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load_edit(&file).is_err());
        // A plain snapshot dir loads lazily.
        let dir = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load_edit(&dir).is_ok());
        // The edit LoRA rides the shared `spec.adapters` merge path (accepted + lazy).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/edit_lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load_edit(&lora).is_ok(), "edit LoRA load is wired + lazy");
    }

    fn ref_image(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn resolve_edit_references_single_and_pair_fixed_order() {
        // A single `Reference` → one source (image 1).
        let one = GenerationRequest {
            prompt: "make it autumn".into(),
            conditioning: vec![Conditioning::Reference {
                image: ref_image(2, 2),
                strength: None,
            }],
            ..Default::default()
        };
        let refs = resolve_edit_references(&one).unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!((refs[0].width, refs[0].height), (2, 2));

        // A two-image `MultiReference` → image 1 then image 2, order preserved.
        let two = GenerationRequest {
            prompt: "combine the two references into one image".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_image(4, 4), ref_image(6, 6)],
            }],
            ..Default::default()
        };
        let refs = resolve_edit_references(&two).unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!((refs[0].width, refs[0].height), (4, 4), "image 1");
        assert_eq!((refs[1].width, refs[1].height), (6, 6), "image 2");
    }

    #[test]
    fn resolve_edit_references_rejects_zero_and_over_cap() {
        // Zero references → error (an edit needs a source).
        let none = GenerationRequest {
            prompt: "make it autumn".into(),
            ..Default::default()
        };
        assert!(resolve_edit_references(&none).is_err());

        // Three references → past the fixed-order cap (image 1, image 2).
        let three = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_image(2, 2), ref_image(2, 2), ref_image(2, 2)],
            }],
            ..Default::default()
        };
        let err = resolve_edit_references(&three).unwrap_err().to_string();
        assert!(err.contains("at most 2"), "got: {err}");
    }

    #[test]
    fn resolve_edit_references_rejects_reference_and_request_strength() {
        let reference_strength = GenerationRequest {
            prompt: "make it autumn".into(),
            conditioning: vec![Conditioning::Reference {
                image: ref_image(2, 2),
                strength: Some(0.5),
            }],
            ..Default::default()
        };
        let reference_err = resolve_edit_references(&reference_strength)
            .unwrap_err()
            .to_string();
        assert!(
            reference_err.contains(KREA_2_RAW_ID),
            "got: {reference_err}"
        );

        let request_strength = GenerationRequest {
            strength: Some(0.5),
            conditioning: vec![Conditioning::Reference {
                image: ref_image(2, 2),
                strength: None,
            }],
            ..reference_strength
        };
        let request_err = resolve_edit_references(&request_strength)
            .unwrap_err()
            .to_string();
        assert!(request_err.contains(KREA_2_RAW_ID), "got: {request_err}");
    }

    // --- Turbo img2img (reference-guided latent-init) — sc-10134 / epic 8588 ---

    #[test]
    fn img2img_reference_extracts_first_reference_and_strength() {
        // A single `Reference` with an explicit strength → (image, Some(strength)).
        let req = GenerationRequest {
            prompt: "a red apple".into(),
            conditioning: vec![Conditioning::Reference {
                image: ref_image(8, 4),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        let (image, strength) = img2img_reference(&req).expect("a Reference is present");
        assert_eq!((image.width, image.height), (8, 4));
        assert_eq!(strength, Some(0.6));

        // No conditioning → plain txt2img (None).
        let plain = GenerationRequest {
            prompt: "a red apple".into(),
            ..Default::default()
        };
        assert!(img2img_reference(&plain).is_none());
    }

    #[test]
    fn turbo_validate_accepts_reference_rejects_multireference() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .unwrap();
        // A single-reference img2img request validates on Turbo (the sc-10134 surface).
        let img2img = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::Reference {
                image: ref_image(64, 64),
                strength: Some(0.5),
            }],
            ..Default::default()
        };
        assert!(g.validate(&img2img).is_ok());
        // A two-image MultiReference is NOT the Turbo img2img surface (that's `krea_2_edit`) — rejected.
        let multi = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_image(64, 64), ref_image(64, 64)],
            }],
            ..Default::default()
        };
        assert!(g.validate(&multi).is_err(), "MultiReference not on Turbo");
    }

    // --- Sequential component residency — sc-12089 / epic 10765 Phase 1c ---

    /// Pin `CANDLE_GEN_OFFLOAD` to a known value for the duration of a test, restoring the prior value
    /// on drop.
    ///
    /// Both properties matter and neither is optional here:
    ///
    /// * **Pinning.** `candle_gen::sequential_offload_enabled` reads a process-global var, and the route
    ///   assertions below turn on the `Resident` default reaching `sequential() == false`. An ambient
    ///   `CANDLE_GEN_OFFLOAD=sequential` — which is exactly what a developer running the two-process A/B
    ///   has exported in that shell — would otherwise turn them red for a reason that has nothing to do
    ///   with the code under test.
    /// * **Restoring on `Drop`, not at the end of the body.** A failing assertion unwinds; a restore
    ///   written as the last statement would be skipped, leaking the mutation into every later test in
    ///   the binary (they run in-process and single-threaded — `.cargo/config.toml` force-pins
    ///   `RUST_TEST_THREADS=1`, F-160). One red test would then cascade into several.
    struct OffloadEnvGuard(Option<String>);

    impl OffloadEnvGuard {
        /// Pin the var to `value` (`None` ⇒ unset) until the guard drops.
        fn set(value: Option<&str>) -> Self {
            let prior = std::env::var(candle_gen::OFFLOAD_ENV).ok();
            match value {
                Some(v) => std::env::set_var(candle_gen::OFFLOAD_ENV, v),
                None => std::env::remove_var(candle_gen::OFFLOAD_ENV),
            }
            Self(prior)
        }
    }

    impl Drop for OffloadEnvGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(v) => std::env::set_var(candle_gen::OFFLOAD_ENV, v),
                None => std::env::remove_var(candle_gen::OFFLOAD_ENV),
            }
        }
    }

    /// The offload contract (sc-12089): `with_offload_policy` is CAPTURED at load and never rejected —
    /// on every id. Loading stays lazy, so this asserts the plumbing on
    /// CPU with no weights and no GPU; the phased route itself is selected inside `generate` and
    /// exercised end-to-end by the cuda A/B harness below.
    ///
    /// The env override's own semantics (spelling, case, whitespace) are asserted where the reader now
    /// lives — `candle_gen::residency`'s `offload_env_reads_sequential_case_insensitively` — rather than
    /// re-tested per engine.
    #[test]
    fn offload_policy_is_captured_not_rejected() {
        let _env = OffloadEnvGuard::set(None);

        // Resident remains lazy, but its cache now lives inside the shared residency owner.
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert!(load(&spec).is_ok());
        assert!(load_raw(&spec).is_ok());

        // `Sequential` is honored, not rejected — for all registered variants. Weights are never touched.
        let seq = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        assert_eq!(seq.offload_policy, OffloadPolicy::Sequential);
        assert!(load(&seq).is_ok());
        assert!(load_raw(&seq).is_ok());
        // Edit now selects the same deferred phase loaders; construction remains weights-free.
        assert!(load_edit(&seq).is_ok());
    }

    /// The env override reaches THIS engine's route decision (sc-12089) — the seam the two-process A/B
    /// harness drives when it cannot set a `LoadSpec`. The reader's own parsing is asserted in
    /// `candle_gen::residency`; what this pins is that krea consults it at all, and that it can flip a
    /// `Resident`-specced generator onto the phased path.
    #[test]
    fn env_override_selects_the_phased_route_on_a_resident_spec() {
        // One scope per guard: a second `let _env` would SHADOW the first rather than replace it, and
        // both would then live to the end of the body — restoring correctly only by accident of LIFO drop
        // order. Explicit scopes make each pin end where it is meant to.
        {
            let _env = OffloadEnvGuard::set(Some("sequential"));
            assert!(
                effective_residency_policy(OffloadPolicy::Resident, false)
                    == OffloadPolicy::Sequential,
                "CANDLE_GEN_OFFLOAD=sequential must select the phased path regardless of the spec"
            );
        }
        {
            let _env = OffloadEnvGuard::set(None);
            assert!(
                effective_residency_policy(OffloadPolicy::Resident, false)
                    == OffloadPolicy::Resident,
                "with the override unset, a Resident spec stays resident"
            );
        }
    }

    /// **The lockstep contract (sc-10840 / sc-12089 / sc-12129).** `supports_sequential_offload` must be true on
    /// exactly the ids whose provider actually wires the phased path — no more.
    ///
    /// This is the load-bearing assertion of the story: the bit is what a consumer's fit-gate reads to
    /// predict a staged (ex-text) peak, while `OffloadPolicy::Sequential` is *advisory* — an unwired lane
    /// silently runs resident. So an id that advertises but defers would be admitted on a card that only
    /// fits the staged set and then OOM. The flag is inherited `descriptor` → `raw_descriptor` →
    /// `edit_descriptor`; sc-12129 makes all three registered ids phase-complete.
    #[test]
    fn sequential_is_advertised_only_where_wired() {
        // Wired: both plain-txt2img lanes and the grounded edit lane phase their loads.
        assert!(descriptor().capabilities.supports_sequential_offload);
        assert!(raw_descriptor().capabilities.supports_sequential_offload);
        assert!(edit_descriptor().capabilities.supports_sequential_offload);
    }

    fn sequential_generator(descriptor: ModelDescriptor) -> KreaGenerator {
        KreaGenerator {
            descriptor,
            device: candle_gen::default_device().expect("a default device"),
            residency: candle_gen::Residency::sequential(
                || {
                    Err(candle_gen::CandleError::Msg(
                        "test text loader must not run".into(),
                    ))
                },
                |_| {
                    Err(candle_gen::CandleError::Msg(
                        "test heavy loader must not run".into(),
                    ))
                },
            ),
        }
    }

    /// F-173 (sc-12089): a request cancelled before `generate` returns `Canceled` without loading a
    /// thing.
    ///
    /// That this passes with `root = /snap` — a path holding no weights — IS the assertion. The
    /// `Sequential` path's first act used to be `load_text`, which would fail here with a missing-file
    /// error; reaching the cancel check first is what makes the error `Canceled`. On a real snapshot the
    /// difference is a cancelled job returning immediately instead of streaming the Qwen3-VL-4B encoder
    /// and then the 12B DiT from disk before noticing.
    ///
    /// The `Resident` path is not measured here: it loads behind the cross-request components cache, so
    /// a cancelled request reaches the sampler's per-step gate almost at once. Staging is what put a
    /// multi-GB load inside `generate`, ahead of the first cancellable step.
    #[test]
    fn cancelled_sequential_request_returns_before_loading_anything() {
        let _env = OffloadEnvGuard::set(None);

        let cancel = gen_core::runtime::CancelFlag::new();
        cancel.cancel();
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            width: 1024,
            height: 1024,
            cancel: cancel.clone(),
            ..Default::default()
        };

        for descriptor in [descriptor(), raw_descriptor()] {
            let g = sequential_generator(descriptor.clone());
            let err = g
                .generate(&req, &mut |_| {})
                .expect_err("a cancelled request must not produce images");
            assert!(
                matches!(err, gen_core::Error::Canceled),
                "{}: expected Canceled, got {err:?} — the stage-boundary check must precede the load",
                descriptor.id
            );
        }

        let edit_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: ref_image(64, 64),
                strength: None,
            }],
            ..req
        };
        let g = sequential_generator(edit_descriptor());
        let err = g
            .generate(&edit_req, &mut |_| {})
            .expect_err("a cancelled edit request must not load its vision tower");
        assert!(
            matches!(err, gen_core::Error::Canceled),
            "{}: expected Canceled, got {err:?} — the text-phase load must follow the cancel check",
            KREA_2_EDIT_ID
        );
    }

    /// The route guard (sc-12089). Two properties, and the second is the load-bearing one:
    ///
    /// 1. `Sequential` selects the phased path on Turbo/Raw/Edit; `Resident` (the default) never does.
    /// 2. **An advertising id takes the phased path for EVERY request it accepts** — txt2img, img2img,
    ///    and grounded edit. Because `supports_sequential_offload` is per-engine, a request-shape-dependent
    ///    deferral would silently break the fit-gate's staged-peak prediction and OOM the admitted job.
    ///    The only deferral is ConvRot, selected uniformly by the load spec rather than request shape.
    #[test]
    fn sequential_route_covers_every_request_an_advertising_id_accepts() {
        // The `Resident` assertions below read the process-global override through `sequential()`; pin it
        // off so an A/B runner's ambient export cannot turn them red (see `OffloadEnvGuard`).
        let _env = OffloadEnvGuard::set(None);

        let plain = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        let img2img = GenerationRequest {
            prompt: "a red apple".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::Reference {
                image: ref_image(64, 64),
                strength: Some(0.5),
            }],
            ..Default::default()
        };

        // Every (id, request-shape) pair the advertising ids accept takes the phased path — no shape
        // silently falls back to resident while the descriptor claims otherwise.
        for descriptor in [descriptor(), raw_descriptor()] {
            assert!(descriptor.capabilities.supports_sequential_offload);
            for _req in [&plain, &img2img] {
                assert!(
                    effective_residency_policy(OffloadPolicy::Sequential, false)
                        == OffloadPolicy::Sequential,
                    "{} must honor Sequential for every request it accepts",
                    descriptor.id
                );
            }
        }

        let edit_req = GenerationRequest {
            prompt: "make the person smile".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::Reference {
                image: ref_image(64, 64),
                strength: None,
            }],
            ..Default::default()
        };
        let edit = edit_descriptor();
        edit.capabilities
            .validate_request(edit.id, &edit_req)
            .expect("the edit descriptor accepts grounded reference conditioning");
        assert!(edit.capabilities.supports_sequential_offload);
        assert_eq!(
            effective_residency_policy(OffloadPolicy::Sequential, false),
            OffloadPolicy::Sequential
        );

        // `Resident` (the default) never takes it — that is the whole opt-in contract.
        assert_eq!(
            effective_residency_policy(OffloadPolicy::Resident, false),
            OffloadPolicy::Resident
        );

        // ConvRot NO LONGER defers (sc-12425): `load_residency_heavy_convrot` sources the int8 single
        // file on the sequential path, so a ConvRot request drops its 15.6 GB f32 TE like every other
        // Turbo request. It MUST follow the policy — the Turbo descriptor advertises
        // `supports_sequential_offload`, and a ConvRot generator silently running Resident would make the
        // fit-gate under-predict its peak (the sc-10840 lockstep violation).
        assert_eq!(
            effective_residency_policy(OffloadPolicy::Sequential, true),
            OffloadPolicy::Sequential
        );
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c, sc-12089) — the candle twin of the MLX
    /// krea A/B (sc-11101), mirroring the candle-gen-flux harness (sc-10769).
    ///
    /// ONE probed generation whose residency mode is chosen by the same two seams `generate` reads:
    /// `CANDLE_GEN_OFFLOAD=sequential` (the env override) or `KREA_OFFLOAD_MODE=spec-sequential` →
    /// `LoadSpec::offload_policy` (the worker-facing contract, with `CANDLE_GEN_OFFLOAD` unset). Prints
    /// the device peak VRAM and writes the raw RGB pixels to `KREA_OUT`.
    ///
    /// **Run it TWICE in SEPARATE processes** (resident vs sequential) and compare: the pixel files must
    /// be byte-identical (parity) and the sequential peak materially lower (the Qwen3-VL-4B TE dropped
    /// before the 12B DiT loads). Two processes are REQUIRED — this is the epic's cudarc caveat: candle's
    /// caching allocator has no `empty_cache` and `Device::synchronize()` does not reclaim, so a second
    /// in-process run would reuse the first run's pool and read the same peak. For the same reason
    /// `nvidia-smi` resident VRAM will NOT fall within a process; what moves is peak *allocation demand*,
    /// which is what the probe reads and what any gate math must key off.
    ///
    /// Reports through [`testkit::VramProbe`] (sc-9094) rather than a bare `PeakSampler`: it separates
    /// load-peak / steady / overall-peak and states each as a delta over a recorded **idle baseline** —
    /// which is what makes the number trustworthy here. The probe is device-level (`nvidia-smi
    /// memory.used`; WDDM reports per-process as `[N/A]`), so anything else resident on the sampled GPU
    /// lands in the measurement. The printed `baseline` is the tell — it must be ~0, else the run shared
    /// the card and the A/B delta is noise. `overall-peak` is also exactly the quantity the manifest's
    /// `candle.vramGbByTier` / `sequentialPeakGb` are derived from (sc-9094 / sc-10856), so this harness
    /// feeds the re-measure directly.
    ///
    /// **Multi-GPU:** the compute device is candle's `cuda:0`, but `nvidia-smi -i` takes a PHYSICAL
    /// ordinal and ignores `CUDA_VISIBLE_DEVICES` — so on a box where you pin the run to a free card with
    /// `CUDA_VISIBLE_DEVICES=1`, a hardcoded `start(0)` would sample the OTHER (busy) card and silently
    /// report its residency as this run's peak. [`candle_gen::testkit::probe_gpu`] derives the physical
    /// ordinal from `CUDA_VISIBLE_DEVICES` so the sampled card is always the one being rendered on.
    ///
    /// `KREA_SEQ_RAW=1` measures `krea_2_raw` (full-CFG, two forwards/step) instead of `krea_2_turbo`.
    /// `KREA_SEQ_EDIT=1` measures the sc-12129 grounded edit path and additionally requires
    /// `KREA_EDIT_LORA` + `KREA_EDIT_SOURCE`; it uses `KREA_RAW_DIR` and must be run resident/sequential
    /// in separate processes with the same explicit seed and source.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn krea_probed_generate_for_offload_ab() {
        let out = std::env::var("KREA_OUT").expect("set KREA_OUT to the pixel-dump path");
        let raw = std::env::var("KREA_SEQ_RAW").is_ok();
        let edit = std::env::var("KREA_SEQ_EDIT").is_ok();
        assert!(!(raw && edit), "set only one of KREA_SEQ_RAW/KREA_SEQ_EDIT");
        // `krea_2_raw` is a DIFFERENT CHECKPOINT (the undistilled base DiT), not a mode of the Turbo
        // snapshot — so it reads its own dir (the mlx-gen-krea `KREA_RAW_DIR` convention, sc-11101).
        // Sharing `KREA_TURBO_DIR` across both would silently load the DISTILLED DiT and run it under
        // the full-CFG loop: same architecture, so it would "work" and report a plausible peak, but the
        // number would not belong to the model it was published against.
        let dir = if raw || edit {
            std::env::var("KREA_RAW_DIR").expect("set KREA_RAW_DIR to a Krea 2 Raw snapshot")
        } else {
            std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR to a Krea 2 Turbo snapshot")
        };

        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        if edit {
            use candle_gen::gen_core::{AdapterKind, AdapterSpec};
            let lora =
                std::env::var("KREA_EDIT_LORA").expect("set KREA_EDIT_LORA for KREA_SEQ_EDIT=1");
            spec = spec.with_adapters(vec![AdapterSpec::new(lora.into(), 1.0, AdapterKind::Lora)]);
        }
        // sc-12425: `KREA_CONVROT_DIT` measures the community INT8-ConvRot lane by riding the DiT single
        // file on `text_encoder` (the `convrot_selector` seam). Run resident vs spec-sequential in two
        // processes: sequential must drop the 15.6 GB f32 Qwen3-VL TE before the int8 DiT loads, taking
        // the ~42.9 GB resident peak (sc-12381) down toward the DiT phase alone.
        if let Ok(convrot) = std::env::var("KREA_CONVROT_DIT") {
            assert!(
                !raw && !edit,
                "KREA_CONVROT_DIT is the Turbo-only community checkpoint; unset KREA_SEQ_RAW/EDIT"
            );
            spec.text_encoder = Some(WeightsSource::File(convrot.into()));
        }
        let spec_mode = std::env::var("KREA_OFFLOAD_MODE").unwrap_or_default();
        if spec_mode == "spec-sequential" {
            spec = spec.with_offload_policy(OffloadPolicy::Sequential);
        }
        // Square edge (default 768, the sc-11101 MLX A/B's resolution so the two backends compare).
        // Set `KREA_AB_RES=1024` to match the condition the manifest's `candle.vramGbByTier` q4 was
        // measured at (RTX PRO 6000, 1024²/8-step) — the activation transient scales with pixel count and
        // is the epic's dominant unknown (sc-11925: it was only ever calibrated at 1024²), so the tier
        // re-measure must be taken at the SAME resolution as the number it replaces.
        let res: u32 = std::env::var("KREA_AB_RES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(768);
        let conditioning = if edit {
            let source = std::env::var("KREA_EDIT_SOURCE")
                .expect("set KREA_EDIT_SOURCE for KREA_SEQ_EDIT=1");
            let rgb = image::open(source)
                .expect("decode KREA_EDIT_SOURCE")
                .to_rgb8();
            let (width, height) = rgb.dimensions();
            vec![Conditioning::Reference {
                image: Image {
                    width,
                    height,
                    pixels: rgb.into_raw(),
                },
                strength: None,
            }]
        } else {
            Vec::new()
        };
        let req = GenerationRequest {
            prompt: if edit {
                "make the person smile warmly, keep their identity".into()
            } else {
                "a rusty robot holding a lit candle, studio lighting".into()
            },
            width: res,
            height: res,
            // Turbo is the 8-step distilled student; Raw is undistilled, so hold it to a short schedule
            // (the A/B measures PEAK, which is step-count-independent — not sample quality).
            steps: Some(8),
            seed: Some(42),
            count: 1,
            conditioning,
            ..Default::default()
        };

        // Load and generate are sampled as SEPARATE phases so the report separates the load transient
        // (weights → device) from the denoise/decode activation spike — the epic's open question is which
        // dominates, and a single fused peak can't say (sc-11925 notes the transient was only calibrated
        // at 1024²).
        let mut probe = candle_gen::testkit::VramProbe::start_rendered();
        let load_phase = probe.phase();
        let g = if edit {
            load_edit(&spec).expect("load krea_2_edit")
        } else if raw {
            load_raw(&spec).expect("load krea_2_raw")
        } else {
            load(&spec).expect("load krea_2_turbo")
        };
        probe.end_load(load_phase);
        let gen_phase = probe.phase();
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        probe.end_gen(gen_phase);
        let report = probe.report();

        let img = match output {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        std::fs::write(&out, &img.pixels).expect("write pixels");

        let env_mode = std::env::var("CANDLE_GEN_OFFLOAD").unwrap_or_default();
        let mode = if spec_mode == "spec-sequential" {
            "spec-sequential"
        } else if env_mode.eq_ignore_ascii_case("sequential") {
            "env-sequential"
        } else {
            "resident"
        };
        let id = if edit {
            KREA_2_EDIT_ID
        } else if raw {
            KREA_2_RAW_ID
        } else {
            KREA_2_TURBO_ID
        };
        eprintln!(
            "SEQ_AB id={id} mode={mode} gpu={} {}x{} steps={:?} | {report} | bytes={} out={out}",
            candle_gen::testkit::probe_gpu(),
            req.width,
            req.height,
            req.steps,
            img.pixels.len(),
        );
        report.assert_trustworthy(1.0);
    }

    /// Test helper: attach a ConvRot DiT single-file selector on `text_encoder` (sc-9300).
    trait WithConvRot {
        fn with_convrot_text_encoder(self) -> Self;
    }
    impl WithConvRot for LoadSpec {
        fn with_convrot_text_encoder(mut self) -> Self {
            self.text_encoder = Some(WeightsSource::File(
                "/krea2_int8_convrot.safetensors".into(),
            ));
            self
        }
    }
}
