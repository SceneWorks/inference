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
// The resident aggregate plus the two residency phases it splits into (epic 10765 Phase 1c, sc-12089):
// `KreaText` (tokenizer + Qwen3-VL-4B TE) is dropped before `KreaHeavy` (DiT + VAE + optional PiD)
// loads on the `Sequential` path. The MLX twins are mlx-gen-krea's `KreaText` / `KreaHeavy` (sc-11101).
pub use pipeline::{Components, KreaHeavy, KreaText};
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
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, OffloadPolicy,
    PidWeights, Progress, Quant, WeightsSource,
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

/// patch_size(2)·vae_downsample(8) = 16 — patchify requires latent dims divisible by this.
const SIZE_MULTIPLE: u32 = 16;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Max images per request (the image-model standard, shared with the other families).
const MAX_COUNT: u32 = 8;

/// A lazily-loaded Krea 2 Turbo generator. The components (tokenizer + Qwen3-VL-4B TE + single-stream
/// DiT + Qwen-Image VAE) load on the first `generate` and are cached.
pub struct KreaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-7836). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// The community **INT8-ConvRot** DiT single-file checkpoint (sc-9300), captured at load when the
    /// spec selected the ConvRot consume path (a `WeightsSource::File` on `LoadSpec::text_encoder` — see
    /// [`build`]). `Some` ⇒ the lazy component build takes the DiT from this int8 checkpoint
    /// ([`pipeline::load_components_convrot`]) and the tokenizer / Qwen3-VL TE / Qwen-Image VAE from the
    /// canonical `root` snapshot; `None` ⇒ the dense/packed `transformer/` snapshot path (unchanged).
    convrot_dit: Option<PathBuf>,
    /// Component-residency policy captured from `LoadSpec::offload_policy` (epic 10765 Phase 1c,
    /// sc-12089). `Sequential` routes a plain **txt2img** `generate` (Turbo or Raw) through the phased
    /// load→encode→drop path ([`pipeline::render_sequential`] / [`pipeline::render_base_sequential`]),
    /// bounding peak allocation demand at the cost of the components cache; `Resident` (default) keeps
    /// the cached path. The worker's fit-gate sets this when it predicts the resident TE+DiT+VAE sum
    /// won't fit but the DiT+VAE working set will.
    ///
    /// Inert on the img2img / edit / ConvRot lanes — see [`Self::sequential`] for why each defers.
    offload_policy: OffloadPolicy,
    components: Mutex<Option<Arc<Components>>>,
    /// The image-edit-only components (Qwen-Image VAE **encoder** + Qwen3-VL **vision tower**), loaded
    /// lazily on the first edit so the txt2img (Turbo/Raw) paths keep their footprint (epic 10871).
    /// Only ever populated for the `krea_2_edit` id.
    edit_components: Mutex<Option<Arc<pipeline::EditComponents>>>,
    /// The Qwen-Image VAE **encoder**, loaded lazily on the first Turbo img2img request (sc-10134) — the
    /// only extra wire reference-guided latent-init needs (it VAE-encodes the reference into the clean
    /// init latent). Kept separate from `edit_components` so a plain img2img never pulls the Qwen3-VL
    /// vision tower the Kontext edit path also loads. Only ever populated for the `krea_2_turbo` id.
    img2img_encoder: Mutex<Option<Arc<QwenVaeEncoder>>>,
}

impl KreaGenerator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        candle_gen::cached(&self.components, || {
            // ConvRot consume path (sc-9300): when a ConvRot DiT was selected, the DiT is taken from
            // the int8 single-file checkpoint while everything else loads from the canonical snapshot.
            // The sm_89 compute-capability floor is enforced inside `load_components_convrot` (locked
            // decision 7). LoRA/LoKr and PiD overlays are not wired through the ConvRot variant
            // (rejected at `build`).
            Ok(match self.convrot_dit.as_ref() {
                Some(convrot_dit) => Arc::new(pipeline::load_components_convrot(
                    &self.root,
                    convrot_dit,
                    &self.device,
                )?),
                None => Arc::new(pipeline::load_components(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                )?),
            })
        })
    }

    /// The image-edit-only components (VAE encoder + Qwen3-VL vision tower), loaded once on the first
    /// edit and cached (epic 10871). Both read weights that already ship in the Krea 2 snapshot.
    fn edit_components(&self) -> gen_core::Result<Arc<pipeline::EditComponents>> {
        candle_gen::cached(&self.edit_components, || {
            Ok(Arc::new(pipeline::load_edit_components(
                &self.root,
                &self.device,
            )?))
        })
    }

    /// The Qwen-Image VAE encoder, loaded once on the first Turbo img2img request and cached (sc-10134).
    /// Reads the `vae/` encoder weights that already ship in the Krea 2 snapshot — no vision tower, unlike
    /// [`Self::edit_components`].
    fn img2img_encoder(&self) -> gen_core::Result<Arc<QwenVaeEncoder>> {
        candle_gen::cached(&self.img2img_encoder, || {
            Ok(Arc::new(crate::vae::load_vae_encoder(
                &self.root,
                &self.device,
            )?))
        })
    }

    /// Whether THIS request runs the `Sequential` phased-residency path (epic 10765 Phase 1c, sc-12089).
    ///
    /// Selected by `LoadSpec::offload_policy` (the worker fit-gate) or the `CANDLE_GEN_OFFLOAD=sequential`
    /// env override (the two-process A/B harness). Both txt2img AND img2img are wired on the Turbo/Raw
    /// ids; the two lanes that defer to resident are:
    ///
    /// * **ConvRot** (`convrot_dit`) — the DiT is a single int8 file, not `root/transformer/`, so
    ///   [`pipeline::load_heavy`] cannot source it. (A deferred non-shipping variant anyway, and it is
    ///   selected by the load spec rather than the request, so it cannot vary per request under one id.)
    /// * **Edit** (`krea_2_edit`) — the Kontext path interleaves the Qwen3-VL vision tower with the text
    ///   encode (`Grounding::condition`), so the text phase is not droppable before the DiT without
    ///   restructuring the grounded encode. The qwen precedent staged this as a follow-up
    ///   (sc-10867 → sc-10968); `edit_descriptor` therefore does NOT advertise the capability.
    ///
    /// **Why img2img is wired rather than deferred.** `supports_sequential_offload` is a per-**engine**
    /// bit, but Turbo/Raw serve both txt2img and reference-guided img2img under ONE id (unlike
    /// flux/flux2/qwen-image, whose sequential engines take `conditioning: vec![]`). A consumer's
    /// fit-gate reads the bit per engine id, so it predicts the staged peak for EVERY request the id
    /// accepts. Deferring img2img to resident while advertising the id sequential would make that
    /// prediction wrong exactly where it is load-bearing — the gate admits on a card that fits the staged
    /// set, the job runs resident, and it OOMs (the sc-10840 contract). So every request an advertising
    /// id accepts must honor the policy; the only deferrals left are load-spec-level (ConvRot) or
    /// non-advertising (Edit).
    fn sequential(&self, _req: &GenerationRequest) -> bool {
        self.convrot_dit.is_none()
            && self.descriptor.id != KREA_2_EDIT_ID
            && (self.offload_policy == OffloadPolicy::Sequential || sequential_offload_enabled())
    }
}

/// Whether the sequential-residency path is force-enabled by env (epic 10765 Phase 1c, sc-12089). Reads
/// `CANDLE_GEN_OFFLOAD`: `sequential` (case-insensitive) selects the phased load→encode→drop path
/// regardless of `LoadSpec::offload_policy`; unset or any other value defers to the spec (the worker
/// fit-gate sets the policy in production). Kept as the override the two-process GPU A/B harness drives,
/// matching the candle-gen-flux (sc-10769/sc-10821) and candle-gen-qwen-image (sc-10867) toggles — the
/// env var name is shared family-wide on purpose, so one A/B runner drives every candle engine.
fn sequential_offload_enabled() -> bool {
    std::env::var("CANDLE_GEN_OFFLOAD")
        .map(|value| value.trim().eq_ignore_ascii_case("sequential"))
        .unwrap_or(false)
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

        // Sequential-residency offload (epic 10765 Phase 1c, sc-12089): load→encode→DROP the Qwen3-VL-4B
        // text phase before the 12B DiT loads, so peak allocation demand is max(TE, DiT+VAE+activations)
        // rather than their sum — letting a card that OOMs the resident path render. Output is
        // byte-identical (the phased path runs the same encode + denoise/decode bodies). Taken BEFORE
        // `self.components()` on purpose: that cache load is exactly the co-resident TE+DiT+VAE set this
        // path exists to avoid. Driven by `LoadSpec::offload_policy` (the worker fit-gate sets
        // `Sequential`); `CANDLE_GEN_OFFLOAD=sequential` is the A/B override. See `Self::sequential` for
        // the lanes that defer to resident.
        if self.sequential(req) {
            // F-132 (the qwen-image sc-11190 fix): evict any resident component set a PRIOR resident
            // request populated. The policy + env are re-read every `generate`, so `self.components` can
            // be holding a live TE+DiT+VAE Arc set; without this take, a sequential request would phase-
            // load its copies on top of that and peak at resident + sequential — the opposite of the
            // flag's purpose. Dropping the cached Arc frees the resident residency first. Poison-tolerant.
            *candle_gen::lock_recover(&self.components) = None;
            // The same Raw-vs-Turbo × txt2img-vs-img2img fork the resident path below takes, against the
            // phased twins. Every arm the resident path can reach for an advertising id has one here —
            // that total coverage is the point (see `Self::sequential`).
            let raw = self.descriptor.id == KREA_2_RAW_ID;
            let images = match (raw, img2img_reference(req)) {
                (true, Some((reference, strength))) => pipeline::render_base_img2img_sequential(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                    req,
                    reference,
                    strength,
                    on_progress,
                )?,
                (true, None) => pipeline::render_base_sequential(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                    req,
                    on_progress,
                )?,
                (false, Some((reference, strength))) => pipeline::render_img2img_sequential(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                    req,
                    reference,
                    strength,
                    on_progress,
                )?,
                (false, None) => pipeline::render_sequential(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    self.pid_spec.as_ref(),
                    req,
                    on_progress,
                )?,
            };
            return Ok(GenerationOutput::Images(images));
        }

        let comps = self.components()?;
        // Variant read off the descriptor id (the mlx-gen-krea `generate_impl` branch): Edit = Kontext
        // instruction edit over 1-2 references on the full-CFG base; Raw = full-CFG undistilled txt2img
        // (52-step, dynamic-mu, two forwards/step); Turbo = CFG-free distilled (8-step, one forward).
        // One generator struct, three render paths.
        let images = if self.descriptor.id == KREA_2_EDIT_ID {
            // Extract the fixed-order source set (image 1, then image 2) from the request
            // conditioning, load the edit-only components lazily, and run the Kontext edit path.
            let references: Vec<Image> =
                resolve_edit_references(req)?.into_iter().cloned().collect();
            let edit = self.edit_components()?;
            // The registered `krea_2_edit` seam is the undistilled full-CFG edit (`distilled = false`);
            // the CFG-free distilled Turbo edit (sc-11640) is driven through the worker's bespoke
            // `generate_candle_krea_edit_stream` lane (which calls `render_edit(distilled = true)`
            // directly), matching how every candle edit lane bypasses the registered seam.
            pipeline::render_edit(
                &comps,
                &edit,
                req,
                &references,
                false,
                &self.device,
                on_progress,
            )?
        } else if self.descriptor.id == KREA_2_RAW_ID {
            if let Some((reference, strength)) = img2img_reference(req) {
                // Raw img2img (reference-guided latent-init under full CFG, sc-10226): a single
                // `Conditioning::Reference` seeds the undistilled two-forward CFG denoise from the
                // VAE-encoded reference — the full-CFG sibling of the Turbo `render_img2img` below. Reuses
                // the same lazily-loaded VAE encoder (no vision tower, unlike the edit path). Raw advertises
                // `Reference` in `raw_descriptor`, so the capability floor accepts it (a MultiReference is
                // still the `krea_2_edit` surface).
                let vae_encoder = self.img2img_encoder()?;
                pipeline::render_base_img2img(
                    &comps,
                    &vae_encoder,
                    req,
                    reference,
                    strength,
                    &self.device,
                    on_progress,
                )?
            } else {
                pipeline::render_base(&comps, req, &self.device, on_progress)?
            }
        } else if let Some((reference, strength)) = img2img_reference(req) {
            // Turbo img2img (reference-guided latent-init, sc-10134): a single `Conditioning::Reference`
            // seeds the CFG-free denoise from the VAE-encoded reference. The capability floor accepts only
            // `Reference` on `krea_2_turbo` (a MultiReference is already rejected); the VAE encoder loads
            // lazily on this first img2img request (no vision tower, unlike the edit path).
            let vae_encoder = self.img2img_encoder()?;
            pipeline::render_img2img(
                &comps,
                &vae_encoder,
                req,
                reference,
                strength,
                &self.device,
                on_progress,
            )?
        } else {
            pipeline::render(&comps, req, &self.device, on_progress)?
        };
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
            // the worker's fit-gate reads. `raw_descriptor` INHERITS this (Raw wires the CFG twin,
            // `render_base_sequential`); `edit_descriptor` explicitly clears it — see there.
            //
            // Provider + advertisement move in LOCKSTEP (the sc-10840 correctness contract): this bit
            // going true is what lets a consumer predict the staged peak, and `OffloadPolicy::Sequential`
            // is advisory (an unwired engine silently stays resident). Advertising a lane that would
            // actually run resident makes the gate under-predict its real peak — an admitted job that
            // then OOMs. Never flip this on ahead of the wiring.
            supports_sequential_offload: true,
        },
    }
}

/// Krea 2 **Raw** identity + capabilities (sc-9994 / epic 9992) — the undistilled 12B DiT run with
/// **true classifier-free guidance** (two DiT forwards/step: cond vs uncond) at 52 steps, unlike the
/// CFG-free distilled Turbo. Same architecture / snapshot layout as Turbo (only the DiT weights differ,
/// distilled vs base), so it shares [`build`] + the whole [`pipeline`]. Exposes a real guidance scale
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
/// [`build`] adapter path) — and additionally advertises the source-reference conditioning:
/// [`ConditioningKind::Reference`] for a single source and [`ConditioningKind::MultiReference`] for two
/// (image 1, then image 2; [`pipeline::MAX_EDIT_REFERENCES`]).
pub fn edit_descriptor() -> ModelDescriptor {
    let mut d = raw_descriptor();
    d.id = KREA_2_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    // Clear the `Sequential` residency bit inherited from `raw_descriptor` (sc-12089). The Kontext edit
    // interleaves the Qwen3-VL vision tower with the text encode (`Grounding::condition`), so the text
    // phase is not droppable before the DiT without restructuring the grounded encode — `KreaGenerator::
    // sequential` therefore defers this id to the resident path, and the advertisement must say so.
    //
    // This is the LOCKSTEP contract, and the reason this line exists at all: the flag is inherited down
    // `descriptor` → `raw_descriptor` → `edit_descriptor`, so wiring Turbo/Raw silently turns it on for
    // Edit too. Left true, the worker's fit-gate would predict Edit's staged (ex-text) peak, admit the
    // job on a card that only fits the staged set, and then run it RESIDENT — an OOM/SIGKILL. Staging
    // the edit lane (the qwen sc-10867 → sc-10968 precedent) means wiring it here AND flipping this.
    d.capabilities.supports_sequential_offload = false;
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
    Ok(Box::new(KreaGenerator {
        descriptor,
        root,
        device,
        adapters: spec.adapters.clone(),
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if any)
        // so the lazy component build loads the engine once. `None` keeps the byte-exact native path.
        pid_spec: spec.pid.clone(),
        convrot_dit,
        // Component residency (epic 10765 Phase 1c, sc-12089): captured, never rejected. `Sequential` is
        // advisory by contract — the lanes that wire it (Turbo/Raw txt2img) phase their loads; the rest
        // (edit / img2img / ConvRot) silently stay resident. `Resident` is the default.
        offload_policy: spec.offload_policy,
        components: Mutex::new(None),
        edit_components: Mutex::new(None),
        img2img_encoder: Mutex::new(None),
    }))
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
/// (only distilled-vs-base DiT weights differ), so one [`build`] serves both — but stores the CFG-capable
/// [`raw_descriptor`] so `generate` runs the full-CFG [`pipeline::render_base`] path. Accepts the same
/// LoRA/LoKr, PiD, and packed-quant surface as Turbo; the ConvRot / ControlNet rejections are shared.
pub fn load_raw(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, raw_descriptor())
}

/// Construct a lazy candle Krea 2 **Edit** generator (`krea_2_edit`, epic 10871 / sc-11085). Identical
/// snapshot assembly to [`load`] / [`load_raw`] — one [`build`] serves all three ids — but stores the
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

    /// The offload contract (sc-12089): `with_offload_policy` is CAPTURED at load, never rejected, and
    /// the `CANDLE_GEN_OFFLOAD` env override is read independently of the spec. Loading stays lazy, so
    /// this asserts the plumbing on CPU with no weights and no GPU — the phased route itself is selected
    /// inside `generate` and exercised end-to-end by the cuda A/B harness below.
    #[test]
    fn offload_policy_is_captured_not_rejected() {
        // Default (no policy set) → Resident: the generator builds and keeps the cached `render` path.
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert!(load(&spec).is_ok());
        assert!(load_raw(&spec).is_ok());

        // `Sequential` is honored, not rejected — for both txt2img variants. Weights are never touched.
        let seq = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        assert_eq!(seq.offload_policy, OffloadPolicy::Sequential);
        assert!(load(&seq).is_ok());
        assert!(load_raw(&seq).is_ok());
        // Edit accepts the spec too — it simply defers to resident (advisory contract), never errors.
        assert!(load_edit(&seq).is_ok());

        // The env override is read independently of the spec (the GPU A/B harness seam).
        //
        // `CANDLE_GEN_OFFLOAD` is process-global, and the sibling route tests assert the `Resident`
        // default through the same reader — so restore whatever was there rather than blindly removing
        // it, which would silently clobber an ambient `CANDLE_GEN_OFFLOAD=sequential` a developer (or an
        // A/B runner shell) had exported and turn a sibling assertion red for an unrelated reason. Safe
        // to mutate at all only because `.cargo/config.toml` force-pins `RUST_TEST_THREADS=1` (F-160).
        let prior = std::env::var("CANDLE_GEN_OFFLOAD").ok();
        std::env::set_var("CANDLE_GEN_OFFLOAD", "SeQuEnTiAl");
        assert!(super::sequential_offload_enabled());
        std::env::set_var("CANDLE_GEN_OFFLOAD", "resident");
        assert!(!super::sequential_offload_enabled());
        std::env::remove_var("CANDLE_GEN_OFFLOAD");
        assert!(!super::sequential_offload_enabled());
        if let Some(prior) = prior {
            std::env::set_var("CANDLE_GEN_OFFLOAD", prior);
        }
    }

    /// **The lockstep contract (sc-10840 / sc-12089).** `supports_sequential_offload` must be true on
    /// exactly the ids whose provider actually wires the phased path — no more.
    ///
    /// This is the load-bearing assertion of the story: the bit is what a consumer's fit-gate reads to
    /// predict a staged (ex-text) peak, while `OffloadPolicy::Sequential` is *advisory* — an unwired lane
    /// silently runs resident. So an id that advertises but defers would be admitted on a card that only
    /// fits the staged set and then OOM. The flag is inherited `descriptor` → `raw_descriptor` →
    /// `edit_descriptor`, so Edit's false is an EXPLICIT clear that a future refactor must not lose.
    #[test]
    fn sequential_is_advertised_only_where_wired() {
        // Wired: both plain-txt2img lanes phase their loads (`render_sequential` / `render_base_sequential`).
        assert!(descriptor().capabilities.supports_sequential_offload);
        assert!(raw_descriptor().capabilities.supports_sequential_offload);
        // NOT wired: the Kontext edit interleaves the vision tower with the text encode, so it defers to
        // resident — and must not advertise. (Staging it is the qwen sc-10867 → sc-10968 follow-up.)
        assert!(!edit_descriptor().capabilities.supports_sequential_offload);
    }

    /// Test helper: a lazily-built generator with an explicit residency policy + optional ConvRot DiT,
    /// so the `sequential` route guard can be asserted without weights or a GPU (the build is lazy).
    fn generator_with(
        descriptor: ModelDescriptor,
        offload_policy: OffloadPolicy,
        convrot_dit: Option<PathBuf>,
    ) -> KreaGenerator {
        KreaGenerator {
            descriptor,
            root: "/snap".into(),
            device: candle_gen::default_device().expect("a default device"),
            adapters: vec![],
            pid_spec: None,
            convrot_dit,
            offload_policy,
            components: Mutex::new(None),
            edit_components: Mutex::new(None),
            img2img_encoder: Mutex::new(None),
        }
    }

    /// The route guard (sc-12089). Two properties, and the second is the load-bearing one:
    ///
    /// 1. `Sequential` selects the phased path on Turbo/Raw; `Resident` (the default) never does.
    /// 2. **An advertising id takes the phased path for EVERY request it accepts** — txt2img *and*
    ///    img2img. Because `supports_sequential_offload` is per-engine and Turbo/Raw serve both surfaces
    ///    under one id, a request-shape-dependent deferral would silently break the fit-gate's staged-peak
    ///    prediction and OOM the job it admitted. Deferrals are only allowed where the id does not
    ///    advertise (Edit) or where the lane is load-spec-selected (ConvRot).
    #[test]
    fn sequential_route_covers_every_request_an_advertising_id_accepts() {
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
            for req in [&plain, &img2img] {
                let g = generator_with(descriptor.clone(), OffloadPolicy::Sequential, None);
                assert!(
                    g.sequential(req),
                    "{} must honor Sequential for every request it accepts",
                    descriptor.id
                );
            }
        }

        // `Resident` (the default) never takes it — that is the whole opt-in contract.
        assert!(!generator_with(descriptor(), OffloadPolicy::Resident, None).sequential(&plain));

        // Edit defers: the grounded encode interleaves the vision tower with the text phase. Safe only
        // because `edit_descriptor` does not advertise — the two must stay in lockstep.
        assert!(!edit_descriptor().capabilities.supports_sequential_offload);
        assert!(
            !generator_with(edit_descriptor(), OffloadPolicy::Sequential, None).sequential(&plain)
        );

        // ConvRot defers: the DiT is a single int8 file, not `root/transformer/`, so `load_heavy` can't
        // source it. Safe because it is selected by the LOAD SPEC, not the request — a ConvRot generator
        // defers uniformly, so the gate can't be fooled per-request.
        let convrot = Some(PathBuf::from("/krea2_int8_convrot.safetensors"));
        assert!(
            !generator_with(descriptor(), OffloadPolicy::Sequential, convrot).sequential(&plain)
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
    /// which is what `PeakSampler` reads and what any gate math must key off.
    ///
    /// `KREA_SEQ_RAW=1` measures `krea_2_raw` (full-CFG, two forwards/step) instead of `krea_2_turbo`.
    /// Ignored by default; needs a real-file (hardlink-staged, not raw-HF-symlink) Krea 2 snapshot in
    /// `KREA_TURBO_DIR` + a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn krea_probed_generate_for_offload_ab() {
        let dir = std::env::var("KREA_TURBO_DIR")
            .expect("set KREA_TURBO_DIR to a real-file (hardlink-staged) Krea 2 snapshot");
        let out = std::env::var("KREA_OUT").expect("set KREA_OUT to the pixel-dump path");
        let raw = std::env::var("KREA_SEQ_RAW").is_ok();

        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        let spec_mode = std::env::var("KREA_OFFLOAD_MODE").unwrap_or_default();
        if spec_mode == "spec-sequential" {
            spec = spec.with_offload_policy(OffloadPolicy::Sequential);
        }
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 768,
            height: 768,
            // Turbo is the 8-step distilled student; Raw is undistilled, so hold it to a short schedule
            // (the A/B measures PEAK, which is step-count-independent — not sample quality).
            steps: Some(8),
            seed: Some(42),
            count: 1,
            ..Default::default()
        };

        let sampler = candle_gen::testkit::PeakSampler::start(0);
        let g = if raw {
            load_raw(&spec).expect("load krea_2_raw")
        } else {
            load(&spec).expect("load krea_2_turbo")
        };
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        let peak_mib = sampler.stop();
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
        let id = if raw { KREA_2_RAW_ID } else { KREA_2_TURBO_ID };
        eprintln!(
            "SEQ_AB id={id} mode={mode} peak_mib={peak_mib} bytes={} {}x{} out={out}",
            img.pixels.len(),
            img.width,
            img.height
        );
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
