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
//! (sc-9607). Packed loads retain file-backed converted sidecars: writable snapshots cache beside the
//! component, while read-only snapshots use the configurable per-user external cache (sc-16587). A
//! complete valid warm cache is read without taking its preparation lock; operators should budget
//! roughly one additional packed-projection copy in whichever cache location is selected.

pub mod adapters;
pub mod config;
pub mod convert;
pub mod loader;
/// Multi-phase Krea denoise primitive (epic 13879, sc-13887 — the candle mirror of mlx-gen-krea's
/// sc-13884). Pure host-side decomposition of an ordered phase list over ONE shared sigma schedule.
pub mod multiphase;
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
    fold_diff_patch, install_additive, merge_adapters, merge_into_weights, AdditiveReport,
    MergeReport,
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
#[cfg(any(feature = "cuda", test))]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Device, Tensor};
#[cfg(test)]
use candle_gen::gen_core::OffloadPolicy;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    SizeFloor, WeightsSource,
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
    /// Multi-phase Raw contexts (epic 13879, sc-13887), encoded on the `Sequential` path before the text
    /// phase drops: the positive (conditional) context always, plus the unconditional (negative) context
    /// iff any phase uses CFG. The `Resident` path encodes these inside the heavy branch from the warm
    /// text phase, so it stays on the [`KreaEncoded::Resident`] marker instead.
    MultiPhase {
        context: Tensor,
        negative: Option<Tensor>,
    },
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
    loaded_quant: Option<Quant>,
    residency: candle_gen::Residency<KreaTextPhase, KreaHeavyPhase>,
    /// The snapshot root — retained so the multi-phase render (epic 13879, sc-13887) can load its
    /// **job-local** base DiT from `transformer/` regardless of residency mode (the shared resident DiT
    /// is never mutated for per-phase adapter toggling — the concurrency-safety invariant).
    root: PathBuf,
    /// The LoRA/LoKr adapters this model was loaded with (`LoadSpec::adapters`), retained so the
    /// multi-phase render can install each phase's named subset (by index, bounds-checked against
    /// `adapters.len()`, with an optional per-phase weight) on that phase's job-local DiT. Empty ⇒ a
    /// base (adapter-free) model, so only base-only phases are valid. The single-phase paths still bake
    /// these into the resident DiT at load, unchanged.
    adapters: Vec<AdapterSpec>,
    /// `true` if any load-time adapter is a ComfyUI/lightx2v **diff-patch** (`.diff`/`.diff_b`), detected
    /// from the adapter file keys at load ([`adapters::any_diff_patch`], sc-13887). A diff-patch delta
    /// folds IRREVERSIBLY into the dense base at load (`W += δ`); the job-local multi-phase DiT would
    /// inherit that mutated base and [`transformer::Krea2Transformer::clear_adapters`] (which only drops
    /// low-rank residuals) cannot undo it — so a "base-only" phase would silently carry the diff-patch.
    /// Multi-phase is therefore rejected loudly on such a model (low-rank LoRA/LoKr — including the
    /// rank-64 turbo LoRA — toggle cleanly and are unaffected).
    has_diff_patch: bool,
}

#[cfg(any(feature = "cuda", test))]
struct KreaMemoryScope {
    device: Device,
    memory: Option<gen_core::GenerationMemory>,
    requires_reference: bool,
    finished: bool,
}

#[cfg(test)]
fn krea_generation_memory(
    contract: &gen_core::MemoryProviderContract,
    selection: gen_core::MemorySelection,
) -> Option<gen_core::GenerationMemory> {
    contract.generation_memory(&selection)
}

#[cfg(any(feature = "cuda", test))]
impl KreaMemoryScope {
    fn ensure_active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(
                "krea memory-strategy request scope is already finished".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl gen_core::MemoryRequestScope for KreaMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.ensure_active()?;
        let has_reference = img2img_reference(request).is_some();
        if has_reference != self.requires_reference
            || request.phases.is_some()
            || request.use_pid
            || (!self.requires_reference && !request.conditioning.is_empty())
        {
            return Err(gen_core::Error::Unsupported(
                "krea: request conditioning does not match the admitted base/control memory route"
                    .to_owned(),
            ));
        }
        // The shared selection is authoritative and request-scoped. Overwrite any state left on a
        // reused warm request so a deeper prior rung cannot leak into the next run.
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        if tile_edge == 512 && overlap == 128 {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "krea_2_turbo: decode tiling is fixed at 512/128, got {tile_edge}/{overlap}"
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.ensure_active()?;
        if chunk_size == pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u32 {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "krea_2_turbo: attention chunk size is fixed at {}, got {chunk_size}",
                pipeline::CONSTRAINED_ATTN_SCORES_BUDGET
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        _first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        if SUPPORTED_TRANSFORMER_WINDOWS.contains(&block_count) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "krea_2_turbo: transformer residency window is fixed at \
                 {SUPPORTED_TRANSFORMER_WINDOWS:?}, got {block_count}"
            )))
        }
    }

    fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.ensure_active()?;
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        self.finished = true;
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
impl Drop for KreaMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            self.finished = true;
        }
    }
}

/// The shared ladder → this provider's per-request execution controls.
///
/// SC-15805: the cumulative default is **contract-owned and defeasible**, so ask
/// [`gen_core::MemoryProviderContract::engages`] which rungs the selection engages instead of
/// hardcoding the cost order in a `match`. A hardcoded `tile_vae_decode: true` on the rung-3 and
/// rung-4 arms is the same hazard as a `>=` comparison in different syntax — it turns a lever on
/// underneath a provider that has not declared that rung `Implemented`. Krea Turbo's contract
/// declares every rung `Implemented`, so the two agree exactly today; this is a consistency fix,
/// not a behavior change.
/// The rung-4 windows this provider will execute — the same list
/// `krea_turbo_memory_strategy_contract` publishes as `transformer_window_sizes`.
///
/// One entry, deliberately. SC-16154 re-measured the post-SC-16096 sidecar path on CUDA with 12
/// balanced interleaved samples per window (95.6 GiB RTX PRO 6000 Blackwell, device 0):
///
/// | window | median step | full min–max spread | paired mean delta vs 1 (95% CI) |
/// |---:|---:|---:|---:|
/// | 1 | 2.0423 s | 1.6311–2.3491 s | reference |
/// | 2 | 2.0027 s | 1.6234–2.4844 s | +0.0416 s (-0.0984–+0.1817) |
/// | 4 | 1.9914 s | 1.6957–2.4211 s | -0.0091 s (-0.1403–+0.1222) |
/// | 8 | 1.8468 s | 1.7265–2.2018 s | -0.0651 s (-0.2120–+0.0818) |
/// | 15 | 1.9286 s | 1.7662–2.4683 s | +0.0193 s (-0.1461–+0.1847) |
/// | 30 | 2.0813 s | 1.7697–2.6964 s | +0.1911 s (+0.0298–+0.3524) |
///
/// Paired time is flat through window 15 (every interval includes zero) and increasing at window 30
/// (+191 ms, interval excludes zero); the median-time fit is +2.0 ms/block. Even window 8's apparent
/// -9.6% median is unresolved by its paired interval. Peak is linear at
/// `107.9·window + 153.4 MiB`, so no wider window buys a demonstrated speedup while every one spends
/// more VRAM. Rung 4 is selected only for a caller already short of VRAM; publish only the
/// minimum-peak window. See
/// [`crate::transformer::DEFAULT_TRANSFORMER_WINDOW`].
///
/// NOT cfg-gated, unlike the contract that publishes it: `generate` re-validates the window on every
/// build, and the streamed trunk itself is not cuda-only (it runs on CPU in the parity tests).
const SUPPORTED_TRANSFORMER_WINDOWS: &[u32] = &[1];

impl Generator for KreaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        #[cfg(any(feature = "cuda", test))]
        {
            (self.descriptor.id == KREA_2_TURBO_ID).then(krea_turbo_memory_strategy_contract)
        }
        #[cfg(not(any(feature = "cuda", test)))]
        {
            None
        }
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy_contract() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        krea_memory_strategy_safety_check(contract, self.loaded_quant, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        #[cfg(feature = "cuda")]
        {
            if self.descriptor.id != KREA_2_TURBO_ID {
                return Ok(None);
            }
            if context.mode != gen_core::MemoryMode::TextToImage
                || context.has_reference
                || context.use_pid
                || context.has_phases
            {
                return Err(gen_core::Error::Unsupported(
                    "krea_2_turbo: optimized memory strategies cover ordinary text-to-image \
                     only (no reference, PiD, or multi-phase request)"
                        .to_owned(),
                ));
            }
            if let gen_core::MemorySafetyDecision::Reject { reason } =
                self.memory_strategy_safety_check(context)
            {
                return Err(gen_core::Error::Unsupported(reason));
            }
            Ok(Some(Box::new(KreaMemoryScope {
                device: self.device.clone(),
                memory: krea_turbo_memory_strategy_contract().generation_memory(&context.selection),
                requires_reference: false,
                finished: false,
            })))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = context;
            Ok(None)
        }
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
        // Multi-phase denoise (epic 13879, sc-13887): Raw-only, from pure noise, ≥1-step phases. The
        // per-phase adapter-index bounds are checked at generate (they need the loaded adapter count).
        validate_phases(id, req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // Loud-reject a multi-phase request on a diff-patch model (sc-13887): its baked `.diff` delta
        // can't be toggled off per phase, so it would silently corrupt "base-only" phases.
        ensure_multiphase_allowed_for(self.descriptor.id, self.has_diff_patch, req)?;

        let raw = self.descriptor.id == KREA_2_RAW_ID;
        let edit = self.descriptor.id == KREA_2_EDIT_ID;
        let edit_references: Vec<Image> = if edit {
            resolve_edit_references(req)?.into_iter().cloned().collect()
        } else {
            Vec::new()
        };
        let reference = img2img_reference(req);

        let has_higher_rung_controls = req.memory.as_ref().is_some_and(|memory| {
            gen_core::GenerationMemory {
                stage_residency: false,
                ..*memory
            } != gen_core::GenerationMemory::default()
        });
        if has_higher_rung_controls {
            let memory = req
                .memory
                .as_ref()
                .expect("higher-rung controls require a GenerationMemory block");
            // A request that is already cancelled must stop before capability checks or any
            // request-scoped component transition. This preserves the cancellation contract for
            // every descriptor, including variants that do not support the selected memory rung.
            candle_gen::check_cancel(&req.cancel)?;
            if self.descriptor.id != KREA_2_TURBO_ID
                || !self.descriptor.capabilities.supports_sequential_offload
                || reference.is_some()
                || req.phases.is_some()
                || req.use_pid
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: per-generation memory adaptation is supported only for native-VAE, \
                     ordinary Turbo text-to-image requests",
                    self.descriptor.id
                )));
            }
            // SC-15792: re-validate the rung-4 window here, not only in `MemoryRequestScope`.
            // `materialize_transformer_window` guards the calibration lifecycle, but this arm is
            // reachable with a hand-built `GenerationMemory` that never opens a scope — and since
            // the window is now genuinely honoured, an out-of-domain value would execute a schedule
            // the provider never declared. gen-core's rule for these fields is that an unsupported
            // value is "a typed rejection rather than a silently different execution than the
            // selector chose"; before the window was plumbed the same request silently ran 1.
            if let Some(window) = memory.transformer_window_size {
                if !SUPPORTED_TRANSFORMER_WINDOWS.contains(&window) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{}: transformer residency window is fixed at {:?}, got {window}",
                        self.descriptor.id, SUPPORTED_TRANSFORMER_WINDOWS
                    )));
                }
            }
            // Keep Krea's established text → DiT → VAE phase bodies disjoint. The shared owner
            // contributes the request-scoped warm-cache transition; this pipeline retains the
            // three-stage execution needed by every cumulative memory rung.
            let images = self.residency.run_exclusive_staged(&req.cancel, || {
                pipeline::render_three_stage(
                    &self.root,
                    &self.device,
                    &self.adapters,
                    req,
                    on_progress,
                )
            })?;
            return Ok(GenerationOutput::Images(images));
        }

        // Multi-phase denoise (epic 13879, sc-13887): resolve the phase list ONCE up-front so both the
        // text-encode phase (whether the unconditional context is needed) and the render phase drive from
        // the same plan. `validate_phases` has already gated this to the Raw t2i variant (no reference/PiD,
        // non-empty, ≥1-step); here we resolve the contiguous schedule slices, per-phase guidance, and
        // per-phase adapter sets (indices bounds-checked against `self.adapters`). `None` ⇒ the ordinary
        // single-phase render below.
        let mp_resolved: Option<Vec<multiphase::ResolvedPhase>> = match req.phases.as_deref() {
            Some(list) if raw && !list.is_empty() => {
                let default_guidance = req.guidance.unwrap_or(pipeline::RAW_GUIDANCE);
                Some(multiphase::resolve_phases(
                    list,
                    default_guidance,
                    self.adapters.len(),
                    self.descriptor.id,
                )?)
            }
            _ => None,
        };
        // The negative (unconditional) context is encoded once iff ANY phase uses CFG.
        let mp_need_neg = mp_resolved
            .as_ref()
            .map(|r| multiphase::any_phase_uses_cfg(r))
            .unwrap_or(false);

        let stage_residency = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stage_residency);
        let synchronize = |result| candle_gen::synchronize_result(&self.device, result);
        let images = self.residency.run_request_scoped(
            stage_residency,
            false,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text| match text {
                // Multi-phase: the `Resident` path encodes in the heavy branch from the warm text phase
                // (stays on the `Resident` marker); the `Sequential` path must encode here, before the
                // text phase drops.
                KreaTextPhase::Resident if mp_resolved.is_some() => Ok(KreaEncoded::Resident),
                KreaTextPhase::Sequential(text) if mp_resolved.is_some() => {
                    let (context, negative) =
                        pipeline::encode_multiphase_contexts(text, req, mp_need_neg)?;
                    Ok(KreaEncoded::MultiPhase { context, negative })
                }
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
            |_| Ok(self.device.synchronize()?),
            |heavy, encoded, on_progress| match (heavy, encoded) {
                // Multi-phase render (sc-13887): drive the resolved phases over the ONE global Raw schedule
                // through a job-local re-adapted DiT (the shared resident is never mutated). `Sequential`
                // carries the pre-encoded contexts; `Resident` encodes here from the warm text phase.
                (
                    KreaHeavyPhase::Sequential(heavy),
                    KreaEncoded::MultiPhase { context, negative },
                ) => {
                    let resolved = mp_resolved
                        .as_ref()
                        .expect("multi-phase encode implies a resolved plan");
                    synchronize(pipeline::render_multiphase(
                        heavy.vae(),
                        &self.root,
                        &self.device,
                        resolved,
                        &self.adapters,
                        &context,
                        negative.as_ref(),
                        req,
                        on_progress,
                    ))
                }
                (KreaHeavyPhase::Resident(resident), KreaEncoded::Resident)
                    if mp_resolved.is_some() =>
                {
                    let resolved = mp_resolved
                        .as_ref()
                        .expect("mp_resolved is Some in this arm");
                    let comps = &resident.components;
                    let (context, negative) =
                        pipeline::encode_multiphase_contexts(comps.text(), req, mp_need_neg)?;
                    synchronize(pipeline::render_multiphase(
                        comps.vae(),
                        &self.root,
                        &self.device,
                        resolved,
                        &self.adapters,
                        &context,
                        negative.as_ref(),
                        req,
                        on_progress,
                    ))
                }
                (KreaHeavyPhase::Sequential(heavy), KreaEncoded::Edit(context)) => {
                    synchronize(pipeline::render_edit_residency(
                        heavy,
                        context,
                        req,
                        &edit_references,
                        &self.device,
                        on_progress,
                    ))
                }
                (KreaHeavyPhase::Sequential(heavy), KreaEncoded::Sequential(context)) => {
                    synchronize(pipeline::render_residency(
                        heavy,
                        context,
                        req,
                        reference,
                        &self.device,
                        on_progress,
                    ))
                }
                (KreaHeavyPhase::Resident(resident), KreaEncoded::Resident) => {
                    let comps = &resident.components;
                    let result = if edit {
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
                    };
                    synchronize(result)
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
        control_kinds: None,
        required_components: &[],
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
            component_precision_floors: &[],
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
            supports_preview: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
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

/// Multi-phase request validation (epic 13879, sc-13887 — the candle mirror of mlx-gen-krea's sc-13884).
/// Multi-phase is the **Raw t2i** variant only: an ordered phase list run from pure noise over ONE global
/// schedule, each phase a contiguous step slice with its own guidance (per-phase CFG on/off) AND its own
/// adapter set (per-phase LoRA/LoKr toggling — the "N steps Raw + M steps Raw+turbo-LoRA" workflow).
/// Rejects, loudly:
/// - phases on any non-Raw variant (Turbo is CFG-free single-phase; edit is out of scope);
/// - phases combined with reference/edit conditioning or the PiD decoder (t2i-from-noise only in v1);
/// - an empty phase list or a 0-step phase (a malformed trajectory).
///
/// Per-phase **adapter index bounds** (against the model's loaded adapter set) are checked at `generate`
/// time by [`multiphase::resolve_phases`] (the count isn't on the descriptor). A no-op when `req.phases`
/// is `None` (the ordinary single-phase render).
fn validate_phases(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    let Some(phases) = req.phases.as_ref() else {
        return Ok(());
    };
    if id != KREA_2_RAW_ID {
        return Err(gen_core::Error::Msg(format!(
            "{id}: multi-phase denoise (phases) is supported on {KREA_2_RAW_ID} only"
        )));
    }
    if phases.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: phases must contain at least one phase"
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: multi-phase denoise renders from pure noise — reference/edit conditioning is not \
             supported (sc-13887 v1)"
        )));
    }
    if req.use_pid {
        return Err(gen_core::Error::Msg(format!(
            "{id}: multi-phase denoise does not support the PiD decoder yet (sc-13887 follow-on)"
        )));
    }
    for (i, ph) in phases.iter().enumerate() {
        if ph.steps == 0 {
            return Err(gen_core::Error::Msg(format!(
                "{id}: phase {i} must run at least one step"
            )));
        }
    }
    Ok(())
}

/// Reject a multi-phase (`phases`) request on a model loaded with a **diff-patch** adapter (sc-13887).
/// The per-phase adapter toggle clears + re-installs low-rank residuals on a job-local DiT, but a
/// `.diff`/`.diff_b` diff-patch delta folds irreversibly into the dense base at load (`W += δ`) — the
/// job-local DiT loaded from that snapshot inherits it and [`transformer::Krea2Transformer::clear_adapters`]
/// cannot undo it, so a "base-only" phase would silently carry the diff-patch (a wrong render, no error).
/// Turn that silent-wrong into a loud reject. Low-rank LoRA/LoKr adapters — including the epic's rank-64
/// turbo LoRA — toggle cleanly and are allowed. Factored as a free fn (id + the load-time flag) so the
/// reject is unit-testable without a loaded model. A no-op when `req.phases` is `None` or the model has
/// no diff-patch adapter.
fn ensure_multiphase_allowed_for(
    id: &str,
    has_diff_patch: bool,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    if req.phases.is_some() && has_diff_patch {
        return Err(gen_core::Error::Msg(format!(
            "{id}: multi-phase denoise is not supported on a model loaded with a diff-patch \
             (.diff/.diff_b) adapter — a diff-patch folds irreversibly into the base weights at load \
             and cannot be toggled off for a base-only phase; load a low-rank LoRA/LoKr adapter for \
             multi-phase"
        )));
    }
    Ok(())
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
    let loaded_quant = actual_quant_tier(spec, descriptor.id)?;
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
    // previously bypassed staged residency rather than dropping its 15.6 GB f32 TE.
    let heavy_convrot = convrot_dit.clone();
    let residency = candle_gen::Residency::request_scoped_with_resident(
        move |_| {
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
        move |_| {
            Ok(KreaTextPhase::Sequential(Box::new(pipeline::load_text(
                &text_root,
                &text_device,
            )?)))
        },
        move |use_pid, _| {
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
    );
    Ok(Box::new(KreaGenerator {
        descriptor,
        device,
        loaded_quant,
        residency,
        root,
        // The multi-phase diff-patch guard input (sc-13887): read the adapter file keys at load. The
        // ConvRot path already rejected adapters above, so `spec.adapters` is empty there ⇒ `false`.
        has_diff_patch: crate::adapters::any_diff_patch(&spec.adapters),
        adapters: spec.adapters.clone(),
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

/// Build a Krea 2 generator from a **community single-file DiT checkpoint** (sc-14022, epic 14015 S0b) —
/// the candle sibling of `mlx-gen-krea::load_from_native_dit_file`, and the candle out-of-registry pattern
/// z-image's `load_from_comfyui_components` established. `dit_file` is a ComfyUI-exported dense-bf16 or
/// descriptor-validated plain-int8 Krea 2 DiT stored under native-mmdit keys (typically namespaced
/// beneath `model.diffusion_model.`, e.g. `kreamania_variant5`/`variant4`); `base_snapshot_dir` is a
/// **resident turnkey** snapshot
/// (`transformer/ text_encoder/ vae/ tokenizer/`) supplying the shared Qwen3-VL text-encoder, Qwen-Image
/// VAE, tokenizer, and the DiT architecture config the single file omits.
///
/// The DiT is read from the single file and, through [`loader::Weights::from_native_file`], remapped
/// native→diffusers with `convrot` **OFF**. Dense weights pass through; plain int8 reconstructs
/// `W = codes.i8 * weight_scale` per row. Neither is corrupted by a rotation that was never applied. It is
/// coverage/bijection + shape validated ([`convert::validate_native_transformer`], fail-closed on any
/// unmapped/missing/foreign key) before assembly; the TE / VAE / tokenizer load from `base_snapshot_dir`
/// exactly as [`load`] does. The result is a warm-`Resident` generator that renders through the same
/// pipeline as a snapshot load. `descriptor` selects the surface — Turbo [`descriptor()`] is the natural
/// default (variant5 is a distilled-Turbo dense merge).
///
/// No load-time adapters (the community merge already baked its LoRAs into the weights). `Sequential`
/// offload is not threaded — the single-file DiT has no
/// snapshot dir to re-load from — so the generator is always `Resident`, mirroring the MLX entrypoint.
pub fn load_from_native_dit_file(
    dit_file: impl AsRef<std::path::Path>,
    base_snapshot_dir: impl AsRef<std::path::Path>,
    mut descriptor: ModelDescriptor,
) -> gen_core::Result<Box<dyn Generator>> {
    let root = base_snapshot_dir.as_ref().to_path_buf();
    let device = candle_gen::default_device()?;
    // Architecture config + TE/VAE/tokenizer come from the resident turnkey; only the DiT weights come
    // from the single file (dense or descriptor-validated plain int8 through the native remap).
    let components = pipeline::load_components_native(&root, dit_file.as_ref(), &device)?;
    let residency = candle_gen::Residency::resident(
        KreaTextPhase::Resident,
        KreaHeavyPhase::Resident(Box::new(ResidentKrea {
            components: Arc::new(components),
            root: root.clone(),
            device: device.clone(),
            edit_components: Mutex::new(None),
            img2img_encoder: Mutex::new(None),
        })),
    );
    // This source has no phase-local native-DiT reloader. Prevent the selector from choosing a
    // request-scoped staged strategy that would otherwise fall back to the snapshot's different DiT.
    descriptor.capabilities.supports_sequential_offload = false;
    Ok(Box::new(KreaGenerator {
        descriptor,
        device,
        loaded_quant: None,
        residency,
        root,
        // The single-file entrypoint threads no load-time adapters (S0b scope), so no diff-patch guard.
        adapters: Vec::new(),
        has_diff_patch: false,
    }))
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

/// Krea Turbo's provider-owned half of the shared memory-strategy handshake. The measured phase
/// coefficients and exact fit boundaries stay in SceneWorks generated evidence; this declaration
/// pins the executable structure that makes those measurements valid.
#[cfg(any(feature = "cuda", test))]
fn build_krea_turbo_memory_strategy_contract() -> gen_core::MemoryProviderContract {
    use gen_core::{
        LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
        MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
        MemoryPrerequisiteScope, MemoryProviderContract, MemoryRuntimeSemantics, MemoryStrategy,
        MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
        MemoryWindowMaterialization,
    };

    MemoryProviderContract {
        provider_id: KREA_2_TURBO_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            // SC-16096: component open content-addresses each MLX affine source triple and prepares a
            // GGML q4/q8 sidecar once. Each streamed window maps that artifact and transfers its bytes
            // directly to CUDA; there is no per-window conversion or device-to-host round trip.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: MemoryStrategySupport::Implemented,
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![
                            pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u32,
                        ],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => MemoryParameterRanges {
                        // One source for what this provider publishes, what its request scope
                        // accepts, and what `generate` re-validates — three sites that were three
                        // independent literals before SC-15792 and would have drifted silently.
                        transformer_window_sizes: SUPPORTED_TRANSFORMER_WINDOWS.to_vec(),
                        ..Default::default()
                    },
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                        MemoryParameterRanges::default()
                    }
                },
            })
            .collect(),
        pid_decode_routes: None,
        load_shape: LoadShape::DeferredMaterialization,
        // Every higher-rung Krea control is executed by `render_three_stage`: the provider reloads
        // text, DiT, and VAE in disjoint phases whenever decode tiling, attention chunking, or
        // transformer streaming is selected. Record that backend coupling on every affected rung so
        // selection/evidence identity cannot omit a mechanism that physically executes. This remains
        // provider-specific; MLX and other Candle providers keep the shared non-staged default.
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            variables: vec![
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::OverlayBytes,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            "krea-turbo-cuda-phase-curves-v1",
            LoadShape::DeferredMaterialization,
        )),
        // The Krea manifest phase curves already contain the measured resident floors. Asset facts
        // remain zero here rather than substituting on-disk shard sums for load-exact CUDA residency.
        asset_facts: gen_core::MemoryAssetFacts::default(),
        runtime: MemoryRuntimeSemantics::default(),
    }
}

#[cfg(any(feature = "cuda", test))]
fn krea_turbo_memory_strategy_contract() -> &'static gen_core::MemoryProviderContract {
    static CONTRACT: OnceLock<gen_core::MemoryProviderContract> = OnceLock::new();
    CONTRACT.get_or_init(build_krea_turbo_memory_strategy_contract)
}

#[cfg(feature = "cuda")]
fn registered_krea_turbo_memory_strategy_contract(
    _spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    Ok(krea_turbo_memory_strategy_contract().clone())
}

fn actual_quant_tier(spec: &LoadSpec, id: &str) -> gen_core::Result<Option<Quant>> {
    if convrot_selector(spec, id)?.is_some() {
        return Ok(Some(Quant::Q8));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id}: actual numeric tier requires a snapshot directory"
            )))
        }
    };
    loader::read_packed_config(&root.join("transformer"))
        .map_err(gen_core::Error::backend)?
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "{id}: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

#[cfg(any(feature = "cuda", test))]
fn registered_krea_safety_check(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    match actual_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => krea_memory_strategy_safety_check(contract, quant, context),
        Err(error) => gen_core::MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[cfg(any(feature = "cuda", test))]
fn registered_krea_valid_fixture(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    strategy: gen_core::MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let is_control = contract.provider_id.ends_with("_control");
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: actual_quant_tier(spec, &contract.provider_id)?,
            component_precision_floors: &[],
        },
        gen_core::MemoryBehaviorRoute {
            mode: if is_control {
                gen_core::MemoryMode::ImageToImage
            } else {
                gen_core::MemoryMode::TextToImage
            },
            reference_count: u32::from(is_control),
            use_pid: false,
            has_phases: false,
            overlay: is_control.then(|| "pose-control".to_owned()),
        },
    )?;
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

#[cfg(any(feature = "cuda", test))]
fn registered_krea_begin_request(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    if let gen_core::MemorySafetyDecision::Reject { reason } =
        registered_krea_safety_check(spec, contract, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    Ok(Some(Box::new(KreaMemoryScope {
        device: Device::Cpu,
        memory: contract.generation_memory(&context.selection),
        requires_reference: contract.provider_id.ends_with("_control"),
        finished: false,
    })))
}

#[cfg(test)]
mod weights_free_behavior_tests {
    use super::*;

    #[test]
    fn cpu_scope_executes_the_registered_base_and_control_behaviors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/krea".into()));
        for (contract, strategy) in [
            (
                krea_turbo_memory_strategy_contract().clone(),
                gen_core::MemoryStrategy::BoundedDecode,
            ),
            (
                build_krea_control_memory_strategy_contract(&spec).unwrap(),
                gen_core::MemoryStrategy::BoundedAttention,
            ),
        ] {
            let mut fixture = registered_krea_valid_fixture(&spec, &contract, strategy)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            let mut scope = registered_krea_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
            scope.configure_request(&mut fixture.request).unwrap();
            assert_eq!(
                fixture.request.memory,
                contract.generation_memory(&fixture.context.selection)
            );
            scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
        }
    }
}

fn krea_memory_strategy_safety_check(
    contract: &gen_core::MemoryProviderContract,
    loaded_quant: Option<Quant>,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    // Krea executes its dense tensors at the provider's BF16/default tier. `LoadSpec::precision`
    // is not wired into the loader, so it must not relabel the calibration evidence admitted here.
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: loaded_quant,
            component_precision_floors: &[],
        }),
        None,
    )
}

#[cfg(feature = "cuda")]
const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: KREA_2_TURBO_ID,
    contract: registered_krea_turbo_memory_strategy_contract,
    safety_check: registered_krea_safety_check,
};
#[cfg(feature = "cuda")]
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: KREA_2_TURBO_ID,
        valid_fixtures: registered_krea_valid_fixture,
        begin_request: registered_krea_begin_request,
    };

/// Provider-owned executable capabilities for SceneWorks' composed Krea Turbo + pose-ControlNet
/// route. The worker owns measured evidence and live-budget selection; this declaration owns which
/// controls the provider can actually execute.
#[cfg(any(feature = "cuda", test))]
fn build_krea_control_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    use gen_core::{
        LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
        MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
        MemoryPrerequisiteScope, MemoryProviderContract, MemoryStrategy,
        MemoryStrategyEngagementExclusion, MemoryStrategyPrerequisite, MemoryStrategySupport,
        MemoryWindowMaterialization,
    };

    let mut contract = MemoryProviderContract::compatibility_default(
        "krea_2_turbo_control",
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        "sc-16013-krea-control-direct-1024-v1",
        LoadShape::EagerMaterialization,
    ));
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode => {
                capability.parameters = MemoryParameterRanges {
                    decode_tile_edges: vec![512],
                    decode_overlaps: vec![128],
                    ..Default::default()
                };
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention => {
                capability.parameters = MemoryParameterRanges {
                    attention_chunk_sizes: vec![128 * 1024 * 1024],
                    ..Default::default()
                };
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "the Krea control provider has no transformer-window execution path"
                        .to_owned(),
                }
            }
        };
    }
    contract.additional_prerequisites = [
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
    ]
    .into_iter()
    .map(|strategy| {
        (
            strategy,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )
    })
    .collect();
    if actual_quant_tier(spec, "krea_2_turbo_control")? != Some(Quant::Q4) {
        // SC-16013's direct 1024² calibration found no decode-tail peak on q8, bf16, or
        // INT8-ConvRot. Attention chunking is independently executable there, so forcing tiled decode
        // underneath it adds a speed cost with no measured memory saving. Q4 retains the cumulative
        // composition because its staged 29.6 → 22.4 GiB decode saving is directly measured.
        contract
            .default_engagement_exclusions
            .push(MemoryStrategyEngagementExclusion {
            selection: MemoryStrategy::BoundedAttention,
            excluded_rung: MemoryStrategy::BoundedDecode,
            evidence:
                "sc-16013-krea-control-direct-1024-v1: non-q4 decode tail is not the measured peak"
                    .to_owned(),
        });
    }
    Ok(contract)
}

#[cfg(feature = "cuda")]
fn registered_krea_control_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    build_krea_control_memory_strategy_contract(spec)
}

#[cfg(feature = "cuda")]
const CONTROL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: "krea_2_turbo_control",
    contract: registered_krea_control_memory_strategy_contract,
    safety_check: registered_krea_safety_check,
};
#[cfg(feature = "cuda")]
const CONTROL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "krea_2_turbo_control",
        valid_fixtures: registered_krea_valid_fixture,
        begin_request: registered_krea_begin_request,
    };

/// Add all Candle Krea generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(TURBO_REGISTRATION)
        .register_generator(RAW_REGISTRATION)
        .register_generator(EDIT_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
        // The direct CUDA control runtime composes the registered Krea base with a native control
        // overlay in SceneWorks; it is a real route, but not a standalone gen-core Generator.
        .register_composed_memory_strategy(CONTROL_MEMORY_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry.register_memory_behavior(CONTROL_MEMORY_BEHAVIOR);
    registry
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

        let spec = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            "/nonexistent".into(),
        ));
        #[cfg(feature = "cuda")]
        {
            let contract = registry
                .memory_strategy_contract(super::KREA_2_TURBO_ID, &spec)
                .unwrap()
                .expect("Krea Turbo must register its CUDA memory-strategy contract");
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                "krea-turbo-cuda-phase-curves-v1"
            );
            assert_eq!(contract.strategies.len(), 5);
            assert!(contract.strategies.iter().all(|capability| matches!(
                capability.support,
                candle_gen::gen_core::MemoryStrategySupport::Implemented
            )));
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();

            let control_contract = registry
                .memory_strategy_contract("krea_2_turbo_control", &spec)
                .unwrap()
                .expect("Krea control must register its CUDA memory-strategy contract");
            assert_eq!(
                control_contract.calibration.as_ref().unwrap().fingerprint,
                "sc-16013-krea-control-direct-1024-v1"
            );
            assert!(matches!(
                control_contract
                    .capability(candle_gen::gen_core::MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                candle_gen::gen_core::MemoryStrategySupport::StructurallyNotApplicable { .. }
            ));
            gen_core_testkit::check_memory_strategy_contract(&control_contract).unwrap();

            let edit_default = candle_gen::gen_core::MemoryProviderContract::compatibility_default(
                super::KREA_2_EDIT_ID,
                contract.backend.clone(),
            );
            gen_core_testkit::check_memory_strategy_contract(&edit_default).unwrap();
        }
        #[cfg(not(feature = "cuda"))]
        assert!(registry
            .memory_strategy_contract(super::KREA_2_TURBO_ID, &spec)
            .unwrap()
            .is_none());

        for id in [super::KREA_2_RAW_ID, super::KREA_2_EDIT_ID] {
            assert!(registry
                .memory_strategy_contract(id, &spec)
                .unwrap()
                .is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident_memory_context(
        contract: &gen_core::MemoryProviderContract,
        quant: Option<Quant>,
    ) -> gen_core::MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        gen_core::MemoryRunContext {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 512,
            cache_state: gen_core::MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn prepacked_turnkeys_without_overrides_bind_registration_to_q4_and_q8() {
        for (bits, actual, wrong) in [
            (4, Quant::Q4, Some(Quant::Q8)),
            (8, Quant::Q8, Some(Quant::Q4)),
        ] {
            let root = std::env::temp_dir()
                .join(format!("candle-krea-tier-{bits}-{}", std::process::id()));
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            let contract = krea_turbo_memory_strategy_contract();
            let actual_context = resident_memory_context(contract, Some(actual));
            assert_eq!(
                registered_krea_safety_check(&spec, contract, &actual_context),
                gen_core::MemorySafetyDecision::Accept
            );
            let mut generator = sequential_generator(descriptor());
            generator.loaded_quant = Some(actual);
            assert_eq!(
                generator.memory_strategy_safety_check(&actual_context),
                gen_core::MemorySafetyDecision::Accept
            );
            for selected in [None, wrong] {
                assert!(matches!(
                    registered_krea_safety_check(
                        &spec,
                        contract,
                        &resident_memory_context(contract, selected),
                    ),
                    gen_core::MemorySafetyDecision::Reject { reason }
                        if reason.contains("does not match loaded tier")
                ));
                assert!(matches!(
                    generator.memory_strategy_safety_check(&resident_memory_context(
                        contract, selected,
                    )),
                    gen_core::MemorySafetyDecision::Reject { reason }
                        if reason.contains("does not match loaded tier")
                ));
            }
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn fp32_load_spec_cannot_relabel_bf16_loaded_or_registered_admission() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.precision = gen_core::Precision::Fp32;
        let contract = krea_turbo_memory_strategy_contract();
        let generator = sequential_generator(descriptor());

        let mut fp32_context = resident_memory_context(contract, None);
        fp32_context.selection.tier.precision = gen_core::Precision::Fp32;
        assert!(matches!(
            registered_krea_safety_check(&spec, contract, &fp32_context),
            gen_core::MemorySafetyDecision::Reject { reason }
                if reason.contains("does not match loaded tier")
        ));
        assert!(matches!(
            generator.memory_strategy_safety_check(&fp32_context),
            gen_core::MemorySafetyDecision::Reject { reason }
                if reason.contains("does not match loaded tier")
        ));

        let bf16_context = resident_memory_context(contract, None);
        assert_eq!(
            registered_krea_safety_check(&spec, contract, &bf16_context),
            gen_core::MemorySafetyDecision::Accept
        );
        assert_eq!(
            generator.memory_strategy_safety_check(&bf16_context),
            gen_core::MemorySafetyDecision::Accept
        );
    }

    #[test]
    fn krea_control_memory_contract_publishes_the_executable_surface() {
        let dense = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let contract = build_krea_control_memory_strategy_contract(&dense).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert!(matches!(
            contract
                .capability(gen_core::MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            gen_core::MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert!(matches!(
            contract
                .capability(gen_core::MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            gen_core::MemoryStrategySupport::Implemented
        ));
        assert!(!contract.engages(
            gen_core::MemoryStrategy::BoundedAttention,
            gen_core::MemoryStrategy::BoundedDecode
        ));

        let root =
            std::env::temp_dir().join(format!("krea-candle-q4-contract-{}", std::process::id()));
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let q4 = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let q4_contract = build_krea_control_memory_strategy_contract(&q4).unwrap();
        assert!(q4_contract.engages(
            gen_core::MemoryStrategy::BoundedAttention,
            gen_core::MemoryStrategy::BoundedDecode
        ));
        std::fs::remove_dir_all(root).ok();
    }

    /// **Krea keeps its own 128 Mi budget while consuming the shared rung-3 planner (SC-15796).**
    ///
    /// SC-15796 hoisted candle's chunk arithmetic onto `gen_core::attention_budget`, which also carries
    /// Z-Image's measured 64 Mi operating point. Krea's 128 Mi is a **legitimately different, measured**
    /// family operating point — the GPU-validated ControlNet chunk size — published through the same
    /// `attentionChunkSize` field, and `configure_attention` rejects anything else. Unifying the two
    /// numbers would silently re-calibrate this family, so this pins the split explicitly:
    ///
    /// 1. Krea declares exactly 128 Mi, and it is **not** the shared Z-Image constant.
    /// 2. The declared value is still interpreted by the *shared planner* — at the krea grounded-TE
    ///    geometry (B·H = 1·32, Sq = Sk = 8192, the inclusive token cap) 128 Mi plans 512 query rows
    ///    where 64 Mi would plan 256, so the number is load-bearing and the planner reads it.
    /// 3. `configure_attention` accepts 128 Mi and rejects the shared 64 Mi.
    #[test]
    fn krea_keeps_its_own_128_mi_budget_on_the_shared_planner() {
        use candle_gen::attention::{AttentionBudget, CONSTRAINED_ATTN_SCORES_BUDGET as SHARED};

        // (1) The declared family operating point, and that it is not the shared one.
        assert_eq!(pipeline::CONSTRAINED_ATTN_SCORES_BUDGET, 128 * 1024 * 1024);
        assert_eq!(SHARED, 64 * 1024 * 1024);
        assert_ne!(pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u64, SHARED);

        // (2) Shared planner, krea's budget. The grounded TE at its 8192-token cap: 32·8192 score
        // elements per query row.
        let rows_per_query = 32u64 * 8192;
        let krea = AttentionBudget::from_score_elements(
            pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u64,
            false,
        );
        assert_eq!(krea.query_block_rows(rows_per_query, 8192), 512);
        let z_image = AttentionBudget::from_score_elements(SHARED, false);
        assert_eq!(z_image.query_block_rows(rows_per_query, 8192), 256);

        // (3) The published contract admits krea's value and only krea's value.
        use gen_core::MemoryRequestScope;
        let mut scope = KreaMemoryScope {
            device: Device::Cpu,
            memory: Some(gen_core::GenerationMemory {
                chunk_attention: true,
                ..Default::default()
            }),
            requires_reference: false,
            finished: false,
        };
        scope
            .configure_attention(pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u32)
            .expect("krea must accept its own declared budget");
        let err = scope
            .configure_attention(SHARED as u32)
            .expect_err("krea must reject the shared z-image budget");
        assert!(
            err.to_string().contains("attention chunk size is fixed"),
            "{err}"
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn shared_memory_ladder_maps_to_cumulative_existing_controls() {
        let tier = gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let parameters = gen_core::MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            attention_chunk_size: Some(pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u32),
            transformer_window_size: Some(1),
            // Krea streams only the DiT (SC-15794 scoped the encoder for z-image); None is the
            // DiT-only default, so this declaration is unchanged in meaning.
            transformer_window_component: None,
        };
        let selected = |strategy| gen_core::MemorySelection {
            strategy,
            parameters,
            tier,
        };
        let contract = krea_turbo_memory_strategy_contract();
        assert!(
            !gen_core::MemoryStrategy::BoundedTransformerResidency
                .engages(gen_core::MemoryStrategy::StagedResidency),
            "the shared rung-4 contract must not imply phase release"
        );
        assert!(
            contract.engages(
                gen_core::MemoryStrategy::BoundedDecode,
                gen_core::MemoryStrategy::StagedResidency
            ),
            "Krea's additive backend prerequisite must preserve its current three-stage coupling"
        );
        assert_eq!(
            contract.engaged_composition(gen_core::MemoryStrategy::BoundedAttention),
            vec![
                gen_core::MemoryStrategy::Resident,
                gen_core::MemoryStrategy::StagedResidency,
                gen_core::MemoryStrategy::BoundedDecode,
                gen_core::MemoryStrategy::BoundedAttention,
            ]
        );

        assert_eq!(
            krea_generation_memory(contract, selected(gen_core::MemoryStrategy::Resident)),
            None
        );
        assert_eq!(
            krea_generation_memory(
                contract,
                selected(gen_core::MemoryStrategy::StagedResidency)
            ),
            Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
        assert_eq!(
            krea_generation_memory(contract, selected(gen_core::MemoryStrategy::BoundedDecode)),
            Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                ..Default::default()
            })
        );
        assert_eq!(
            krea_generation_memory(
                contract,
                selected(gen_core::MemoryStrategy::BoundedAttention)
            ),
            Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                chunk_attention: true,
                ..Default::default()
            })
        );
        assert_eq!(
            krea_generation_memory(
                contract,
                selected(gen_core::MemoryStrategy::BoundedTransformerResidency)
            ),
            Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                chunk_attention: true,
                stream_transformer_blocks: true,
                // SC-15792: the SELECTED window travels with the selection. Pinned as a non-default
                // value on purpose — `None` is `GenerationMemory::default()`, so asserting it would
                // pass with the propagation deleted, which is exactly how the field came to be
                // dropped on the floor in the first place.
                transformer_window_size: Some(1),
                transformer_window_component: Some(gen_core::TransformerComponent::Dit),
                ..Default::default()
            })
        );
    }

    /// **SC-15792 — the selected window must be the executed one.**
    ///
    /// `krea_generation_memory` used to build its `GenerationMemory` with `..Default::default()`,
    /// which silently discarded `transformer_window_size`: the pipeline then fell back to the
    /// provider constant no matter what the selector chose. That is invisible while the constant and
    /// the only published candidate are both 1 — so this drives a window the provider does NOT
    /// publish, which the shipped path can never produce, and asserts it survives the mapping.
    ///
    /// The companion half is that such a value is REJECTED at the request boundary rather than
    /// executed; `generate` checks it against `SUPPORTED_TRANSFORMER_WINDOWS`.
    #[test]
    fn the_selected_transformer_window_travels_to_the_request() {
        use gen_core::MemoryStrategy;

        let contract = krea_turbo_memory_strategy_contract();
        let select = |strategy, window| {
            krea_generation_memory(
                contract,
                gen_core::MemorySelection {
                    strategy,
                    parameters: gen_core::MemoryStrategyParameters {
                        transformer_window_size: window,
                        ..Default::default()
                    },
                    tier: gen_core::MemoryNumericTier {
                        precision: gen_core::Precision::Bf16,
                        quant: Some(Quant::Q4),
                        component_precision_floors: &[],
                    },
                },
            )
        };

        // A window the provider does not publish still arrives intact — the mapping's job is to
        // carry the selection faithfully, and rejecting it is `generate`'s job, not this one's.
        let engaged = select(MemoryStrategy::BoundedTransformerResidency, Some(4))
            .expect("rung 4 maps to a per-generation memory block");
        assert!(engaged.stream_transformer_blocks);
        assert_eq!(
            engaged.transformer_window_size,
            Some(4),
            "the selected window was dropped; the pipeline would silently run the provider default"
        );

        // A rung that does not stream blocks must not carry a window: it would be a parameter for a
        // lever that is off, and the pipeline reads the field without re-checking the flag.
        let shallower = select(MemoryStrategy::BoundedAttention, Some(4))
            .expect("rung 3 maps to a per-generation memory block");
        assert!(!shallower.stream_transformer_blocks);
        assert_eq!(shallower.transformer_window_size, None);
    }

    /// The three places this provider states its rung-4 window — what it publishes, what its request
    /// scope accepts, and what `generate` re-validates — must agree. They were three independent
    /// literals before SC-15792.
    #[test]
    fn the_published_window_candidates_are_the_ones_the_scope_accepts() {
        let contract = krea_turbo_memory_strategy_contract();
        let published = contract
            .capability(gen_core::MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4 is declared")
            .parameters
            .transformer_window_sizes
            .clone();
        assert_eq!(published, SUPPORTED_TRANSFORMER_WINDOWS.to_vec());
        assert!(
            !published.is_empty(),
            "an empty candidate list would make every window unsupported and rung 4 unselectable"
        );
        assert!(
            SUPPORTED_TRANSFORMER_WINDOWS
                .contains(&(crate::transformer::DEFAULT_TRANSFORMER_WINDOW as u32)),
            "the shipped default must be one of the windows the contract publishes"
        );
    }

    /// SC-15805: the cumulative default is DEFEASIBLE, and this provider now reads it from the
    /// contract rather than from the ladder's numeric order. Pin that with a contract that declares
    /// a cheaper rung unavailable: a deeper selection must leave that rung's lever OFF.
    ///
    /// Without this, reverting `krea_generation_memory` to its `match`-over-the-cost-order form is
    /// invisible — every other test uses the production contract, where all five rungs are
    /// `Implemented` and the two forms agree exactly.
    #[test]
    fn a_rung_the_provider_does_not_implement_is_not_engaged_by_a_deeper_selection() {
        use gen_core::{MemoryStrategy, MemoryStrategySupport};

        let mut contract = build_krea_turbo_memory_strategy_contract();
        for capability in &mut contract.strategies {
            if capability.strategy == MemoryStrategy::BoundedDecode {
                capability.support = MemoryStrategySupport::Missing;
            }
        }

        let memory = krea_generation_memory(
            &contract,
            gen_core::MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: gen_core::MemoryStrategyParameters {
                    transformer_window_size: Some(1),
                    ..Default::default()
                },
                tier: gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
        )
        .expect("an optimized rung maps to a control set");

        assert!(
            !memory.tile_vae_decode,
            "rung 2 is declared Missing, so a rung-4 selection must not tile the decode; the \
             cost order is not a dependency"
        );
        // ...while the rungs the provider DOES declare stay on, so this is not a vacuous all-false.
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
    }

    /// **SC-16090/SC-16096.** The shipped contract must be conformance-clean, asserted here rather than only in
    /// gen-core's own tests, because `conformance_errors` has no caller in this repo: the consumer is
    /// SceneWorks' selector, which bails on ANY non-empty result and drops this provider to
    /// resident-only gating. The assertion also pins SC-16096's device-format transfer declaration.
    #[test]
    fn the_shipped_contract_is_conformance_clean_and_declares_its_window_realization() {
        use gen_core::{MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization};

        let contract = build_krea_turbo_memory_strategy_contract();
        assert_eq!(
            contract.conformance_errors(),
            Vec::<String>::new(),
            "the shipped Krea contract must be conformance-clean; a non-empty result makes the \
             shared selector drop every rung, not just rung 4"
        );

        // Rung 4 is declared Implemented, so the window-realization rule is genuinely engaged above
        // rather than passing because nothing streams.
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        ));

        assert_eq!(
            contract.window_materialization(),
            Some(&MemoryWindowMaterialization::DeviceFormatTransfer),
            "Krea's streamed trunk maps a content-addressed GGML sidecar and transfers those bytes; \
             the shipped contract must not retain the transitional conversion escape hatch"
        );
    }

    #[test]
    fn request_scope_reapplies_warm_state_rejects_non_t2i_and_finishes_once() {
        use gen_core::MemoryRequestScope;

        let attention_memory = gen_core::GenerationMemory {
            tile_vae_decode: true,
            chunk_attention: true,
            ..Default::default()
        };
        let mut scope = KreaMemoryScope {
            device: Device::Cpu,
            memory: Some(attention_memory),
            requires_reference: false,
            finished: false,
        };
        let mut request = GenerationRequest {
            prompt: "test".to_owned(),
            memory: Some(gen_core::GenerationMemory {
                stream_transformer_blocks: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(request.memory, Some(attention_memory));
        request
            .memory
            .as_mut()
            .unwrap()
            .authorize_calibration_fault(gen_core::MemoryPhase::Denoise);
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            request.memory,
            Some(attention_memory),
            "a warm follow-up request must not inherit a prior calibration fault"
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(gen_core::MemoryRunOutcome::Complete).is_err());

        let mut rejected = KreaMemoryScope {
            device: Device::Cpu,
            memory: Some(attention_memory),
            requires_reference: false,
            finished: false,
        };
        let mut img2img = GenerationRequest {
            prompt: "test".to_owned(),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0; 16 * 16 * 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(matches!(
            rejected.configure_request(&mut img2img),
            Err(gen_core::Error::Unsupported(_))
        ));
        rejected
            .finish(gen_core::MemoryRunOutcome::Canceled)
            .unwrap();
    }

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

    /// Image construction stays lazy and ignores the legacy load-time policy; the request owns the
    /// decision. This is weight-free because no loader runs until `generate`.
    #[test]
    fn image_load_policy_is_not_a_residency_authority() {
        let resident = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        let legacy_staged = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        for spec in [&resident, &legacy_staged] {
            assert!(load(spec).is_ok());
            assert!(load_raw(spec).is_ok());
            assert!(load_edit(spec).is_ok());
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
            loaded_quant: None,
            residency: candle_gen::Residency::request_scoped(
                |_| {
                    Err(candle_gen::CandleError::Msg(
                        "test text loader must not run".into(),
                    ))
                },
                |_, _| {
                    Err(candle_gen::CandleError::Msg(
                        "test heavy loader must not run".into(),
                    ))
                },
            ),
            root: "/snap".into(),
            adapters: Vec::new(),
            has_diff_patch: false,
        }
    }

    #[test]
    fn constrained_memory_route_honors_a_pre_cancel_before_loading() {
        let generator = sequential_generator(descriptor());
        let cancel = gen_core::CancelFlag::default();
        cancel.cancel();
        let req = GenerationRequest {
            prompt: "test".into(),
            memory: Some(gen_core::GenerationMemory {
                tile_vae_decode: true,
                chunk_attention: true,
                stream_transformer_blocks: true,
                ..Default::default()
            }),
            cancel,
            ..Default::default()
        };
        assert!(matches!(
            generator.generate(&req, &mut |_| {}),
            Err(gen_core::Error::Canceled)
        ));
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
        let cancel = gen_core::runtime::CancelFlag::new();
        cancel.cancel();
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            width: 1024,
            height: 1024,
            cancel: cancel.clone(),
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
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

    #[test]
    fn rung_one_reaches_the_shared_staged_loader_for_every_advertised_krea_shape() {
        let staged = || GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut cases = vec![
            (descriptor(), staged()),
            (raw_descriptor(), staged()),
            (
                descriptor(),
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: ref_image(64, 64),
                        strength: Some(0.5),
                    }],
                    ..staged()
                },
            ),
            (
                edit_descriptor(),
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: ref_image(64, 64),
                        strength: None,
                    }],
                    ..staged()
                },
            ),
            (
                descriptor(),
                GenerationRequest {
                    use_pid: true,
                    ..staged()
                },
            ),
            (
                raw_descriptor(),
                GenerationRequest {
                    phases: Some(vec![gen_core::GenerationPhase {
                        steps: 1,
                        ..Default::default()
                    }]),
                    ..staged()
                },
            ),
        ];

        for (descriptor, request) in cases.drain(..) {
            let generator = sequential_generator(descriptor.clone());
            let error = generator
                .generate(&request, &mut |_| {})
                .expect_err("the fake text loader must fail");
            assert!(
                error.to_string().contains("test text loader must not run"),
                "{} must reach rung-one staging for this request shape, got {error:?}",
                descriptor.id
            );
        }
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c, sc-12089) — the candle twin of the MLX
    /// krea A/B (sc-11101), mirroring the candle-gen-flux harness (sc-10769).
    ///
    /// ONE probed generation whose residency mode is carried by the request memory contract and
    /// calibrated with `KREA_OFFLOAD_MODE=request-staged`. Prints
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
        // file on `text_encoder` (the `convrot_selector` seam). Run resident vs request-staged in two
        // processes: sequential must drop the 15.6 GB f32 Qwen3-VL TE before the int8 DiT loads, taking
        // the ~42.9 GB resident peak (sc-12381) down toward the DiT phase alone.
        if let Ok(convrot) = std::env::var("KREA_CONVROT_DIT") {
            assert!(
                !raw && !edit,
                "KREA_CONVROT_DIT is the Turbo-only community checkpoint; unset KREA_SEQ_RAW/EDIT"
            );
            spec.text_encoder = Some(WeightsSource::File(convrot.into()));
        }
        let stage_residency =
            std::env::var("KREA_OFFLOAD_MODE").is_ok_and(|mode| mode == "request-staged");
        let memory_mode = std::env::var("KREA_MEMORY_RUNG").unwrap_or_default();
        // Square edge (default 768, the sc-11101 MLX A/B's resolution so the two backends compare).
        // Set `KREA_AB_RES=1024` to match the condition the manifest's `candle.vramGbByTier` q4 was
        // measured at (RTX PRO 6000, 1024²/8-step) — the activation transient scales with pixel count and
        // is the epic's dominant unknown (sc-11925: it was only ever calibrated at 1024²), so the tier
        // re-measure must be taken at the SAME resolution as the number it replaces.
        let res: u32 = std::env::var("KREA_AB_RES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(768);
        let measured_steps: u32 = std::env::var("KREA_AB_STEPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(8);
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
        let memory = match memory_mode.as_str() {
            "" if stage_residency => Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            "" => None,
            "three-stage" => Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            "tiled-vae" => Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                ..Default::default()
            }),
            "chunked-attention" => Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                ..Default::default()
            }),
            "streamed-blocks" => Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                stream_transformer_blocks: true,
                ..Default::default()
            }),
            other => panic!(
                "unknown KREA_MEMORY_RUNG={other}; use three-stage/tiled-vae/chunked-attention/streamed-blocks"
            ),
        };
        assert!(
            memory.is_none() || (!raw && !edit),
            "KREA_MEMORY_RUNG measures ordinary Turbo text-to-image only"
        );
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
            steps: Some(measured_steps),
            seed: Some(42),
            count: 1,
            conditioning,
            memory,
            ..Default::default()
        };

        let max_baseline_gb = std::env::var("KREA_PROBE_MAX_BASELINE_GB")
            .ok()
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(1.0);
        let mut probe = candle_gen::testkit::VramProbe::start_rendered();

        // Optional same-process model-swap probe (sc-15205). Exercise another complete Turbo
        // snapshot first, drop it, then perform the measured target-tier run below. This catches
        // allocator/host-map retention that isolated one-model processes cannot expose. The swap uses
        // the same constrained route and fixed seed, but one step is sufficient because the working
        // set is step-count invariant.
        if let Ok(swap_dir) = std::env::var("KREA_SWAP_DIR") {
            assert!(
                !raw && !edit && memory.is_some(),
                "KREA_SWAP_DIR is supported only by the constrained ordinary Turbo probe"
            );
            let swap_spec = LoadSpec::new(WeightsSource::Dir(swap_dir.into()));
            let swap = load(&swap_spec).expect("load KREA_SWAP_DIR");
            let mut swap_req = req.clone();
            swap_req.steps = Some(1);
            let swap_output = swap
                .generate(&swap_req, &mut |_| {})
                .expect("generate KREA_SWAP_DIR before measured target");
            let swap_bytes = match swap_output {
                GenerationOutput::Images(mut images) => images.remove(0).pixels.len(),
                other => panic!("expected swap images, got {other:?}"),
            };
            drop(swap);
            eprintln!(
                "KREA_SWAP completed before target tier: {}x{} bytes={swap_bytes}",
                swap_req.width, swap_req.height
            );
        }

        // Load and generate are sampled as SEPARATE phases so the report separates the load transient
        // (weights → device) from the denoise/decode activation spike — the epic's open question is which
        // dominates, and a single fused peak can't say (sc-11925 notes the transient was only calibrated
        // at 1024²).
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
        if let Ok(delay_ms) = std::env::var("KREA_CANCEL_AFTER_MS") {
            let delay_ms = delay_ms
                .trim()
                .parse::<u64>()
                .expect("KREA_CANCEL_AFTER_MS must be an integer");
            let cancel = req.cancel.clone();
            let timer = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                cancel.cancel();
            });
            let cancel_started = std::time::Instant::now();
            let error = match g.generate(&req, &mut |_| {}) {
                Ok(_) => panic!("delayed cancellation unexpectedly produced an image"),
                Err(error) => error,
            };
            timer.join().expect("join delayed cancellation timer");
            let cancel_elapsed_s = cancel_started.elapsed().as_secs_f64();
            assert!(
                matches!(error, gen_core::Error::Canceled),
                "expected delayed cancellation, got {error:?}"
            );
            probe.end_gen(gen_phase);
            let report = probe.report();
            eprintln!("KREA_CANCEL delay_ms={delay_ms} elapsed_s={cancel_elapsed_s:.3} | {report}");
            report.assert_trustworthy(max_baseline_gb);
            return;
        }
        let repeats: usize = std::env::var("KREA_AB_REPEATS")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(1);
        assert!(repeats > 0, "KREA_AB_REPEATS must be positive");
        let mut phase_peaks = Vec::new();
        let mut repeat_elapsed = Vec::with_capacity(repeats);
        let mut first_pixels = None;
        let mut img = None;
        let started = std::time::Instant::now();
        for repeat in 0..repeats {
            let mut observed = None;
            let mut phase_index = 0usize;
            let repeat_started = std::time::Instant::now();
            let output = g
                .generate(&req, &mut |progress| {
                    if matches!(progress, Progress::Loading(_)) {
                        if let Some((name, phase)) = observed.take() {
                            phase_peaks.push((repeat, name, probe.end_observed(phase)));
                        }
                        let name = match phase_index {
                            0 => "text",
                            1 => "denoise",
                            _ => "decode",
                        };
                        phase_index += 1;
                        observed = Some((name, probe.phase()));
                    }
                })
                .expect("generate");
            repeat_elapsed.push(repeat_started.elapsed().as_secs_f64());
            if let Some((name, phase)) = observed.take() {
                phase_peaks.push((repeat, name, probe.end_observed(phase)));
            }
            let current = match output {
                GenerationOutput::Images(mut images) => images.remove(0),
                other => panic!("expected images, got {other:?}"),
            };
            if let Some(expected) = &first_pixels {
                assert_eq!(
                    &current.pixels, expected,
                    "repeat {repeat} changed the fixed-seed RGB output"
                );
            } else {
                first_pixels = Some(current.pixels.clone());
            }
            img = Some(current);
        }
        let elapsed_s = started.elapsed().as_secs_f64();
        probe.end_gen(gen_phase);
        let report = probe.report();

        let img = img.expect("at least one repeated image");
        std::fs::write(&out, &img.pixels).expect("write pixels");

        let mode = if !memory_mode.is_empty() {
            memory_mode.as_str()
        } else if stage_residency {
            "request-staged"
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
            "SEQ_AB id={id} mode={mode} gpu={} {}x{} steps={:?} repeats={repeats} \
             elapsed_s={elapsed_s:.3} repeat_elapsed_s={repeat_elapsed:?} | {report} | bytes={} \
             out={out}",
            candle_gen::testkit::probe_gpu(),
            req.width,
            req.height,
            req.steps,
            img.pixels.len(),
        );
        for (repeat, phase, peak_gb) in phase_peaks {
            eprintln!(
                "KREA_PHASE rung={memory_mode} repeat={repeat} phase={phase} peak_gb={peak_gb:.3}"
            );
        }
        report.assert_trustworthy(max_baseline_gb);
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

    // --- Multi-phase Raw denoise (epic 13879, sc-13887) — the candle mirror of mlx-gen-krea's sc-13884.
    // These pin the request-validation gates (Raw-only / from-noise / diff-patch reject); the schedule
    // decomposition + per-phase adapter resolution are pinned in `crate::multiphase`'s own unit tests, and
    // the shared-core per-phase adapter toggle in `candle_gen::quant::AdaptLinear`'s tests
    // (`clear_adapters_reverts_to_bare_base_and_repush_reinstalls`).

    fn phase(steps: u32, guidance: Option<f32>) -> gen_core::GenerationPhase {
        gen_core::GenerationPhase {
            steps,
            guidance,
            adapters: vec![],
        }
    }

    fn phase_req(phases: Vec<gen_core::GenerationPhase>) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            phases: Some(phases),
            ..Default::default()
        }
    }

    /// The canonical Raw multi-phase split (CFG-on then CFG-off, base-only) validates on `krea_2_raw`.
    #[test]
    fn multiphase_base_only_guidance_split_validates_on_raw() {
        let r = phase_req(vec![phase(20, Some(3.5)), phase(8, Some(0.0))]);
        assert!(validate_phases(KREA_2_RAW_ID, &r).is_ok());
    }

    /// Multi-phase is Raw-only in v1: Turbo/edit reject it (Turbo is CFG-free single-phase; edit is out
    /// of scope). Exercised through the full generator `validate`, so the whole request-gate chain runs.
    #[test]
    fn multiphase_rejected_on_non_raw_variants() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let r = phase_req(vec![phase(8, None)]);
        for id in [KREA_2_TURBO_ID, KREA_2_EDIT_ID] {
            // Free-fn gate.
            let err = validate_phases(id, &r).unwrap_err().to_string();
            assert!(err.contains("supported on krea_2_raw"), "{id}: {err}");
            // And through the generator's `validate` (Turbo reaches the phase gate; edit resolves refs
            // first, but a phase request with no conditioning trips the Raw-only gate before that matters).
            if id == KREA_2_TURBO_ID {
                let g = crate::provider_registry().unwrap().load(id, &spec).unwrap();
                assert!(g
                    .validate(&r)
                    .unwrap_err()
                    .to_string()
                    .contains("supported on krea_2_raw"));
            }
        }
    }

    /// An empty phase list and a 0-step phase are malformed trajectories.
    #[test]
    fn multiphase_rejects_empty_list_and_zero_step_phase() {
        let empty = validate_phases(KREA_2_RAW_ID, &phase_req(vec![]))
            .unwrap_err()
            .to_string();
        assert!(empty.contains("at least one phase"), "{empty}");
        let zero = validate_phases(
            KREA_2_RAW_ID,
            &phase_req(vec![phase(4, None), phase(0, None)]),
        )
        .unwrap_err()
        .to_string();
        assert!(
            zero.contains("phase 1 must run at least one step"),
            "{zero}"
        );
    }

    /// Per-phase adapters are WIRED (sc-13887): the flagship "N steps Raw (CFG on) + M steps
    /// Raw+turbo-LoRA (CFG off)" request VALIDATES (no longer rejected). The adapter-index bounds check
    /// runs at `generate` time (`multiphase::resolve_phases`), not here — see the multiphase resolver
    /// tests.
    #[test]
    fn multiphase_accepts_per_phase_adapters() {
        let with_adapter = gen_core::GenerationPhase {
            steps: 8,
            guidance: Some(0.0),
            adapters: vec![gen_core::PhaseAdapter {
                adapter: 0,
                weight: Some(1.0),
            }],
        };
        assert!(validate_phases(
            KREA_2_RAW_ID,
            &phase_req(vec![phase(20, Some(3.5)), with_adapter])
        )
        .is_ok());
    }

    /// sc-13887 diff-patch guard: a multi-phase request on a model that had a `.diff`/`.diff_b`
    /// diff-patch folded at load is REJECTED loudly (the baked delta can't be toggled off per phase); a
    /// multi-phase request on a low-rank-only model (the epic's turbo-LoRA case) is ACCEPTED; and a
    /// single-phase request on a diff-patch model is unaffected.
    #[test]
    fn multiphase_rejected_on_diff_patch_model_but_allowed_low_rank() {
        let mp = phase_req(vec![phase(20, Some(3.5)), phase(8, Some(0.0))]);
        // diff-patch folded at load → multi-phase rejected with the diff-patch error.
        let err = ensure_multiphase_allowed_for(KREA_2_RAW_ID, true, &mp)
            .unwrap_err()
            .to_string();
        assert!(err.contains("diff-patch"), "{err}");
        assert!(err.contains(".diff/.diff_b"), "{err}");
        // Low-rank-only model (no diff-patch) → the same multi-phase request is allowed.
        assert!(ensure_multiphase_allowed_for(KREA_2_RAW_ID, false, &mp).is_ok());
        // A single-phase request on a diff-patch model is unaffected (the guard is phases-only).
        let single = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(ensure_multiphase_allowed_for(KREA_2_RAW_ID, true, &single).is_ok());
    }

    /// Multi-phase renders from pure noise — reference/edit conditioning and the PiD decoder are
    /// rejected (t2i-from-noise only in v1).
    #[test]
    fn multiphase_rejects_reference_and_pid() {
        let mut with_ref = phase_req(vec![phase(8, None)]);
        with_ref.conditioning = vec![Conditioning::Reference {
            image: ref_image(64, 64),
            strength: None,
        }];
        assert!(validate_phases(KREA_2_RAW_ID, &with_ref)
            .unwrap_err()
            .to_string()
            .contains("renders from pure noise"));

        let mut with_pid = phase_req(vec![phase(8, None)]);
        with_pid.use_pid = true;
        assert!(validate_phases(KREA_2_RAW_ID, &with_pid)
            .unwrap_err()
            .to_string()
            .contains("PiD decoder"));
    }

    /// `phases: None` is the ordinary single-phase render — validation is unchanged, both through the
    /// free gate and the full generator `validate`.
    #[test]
    fn single_phase_request_is_unaffected() {
        let single = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_phases(KREA_2_RAW_ID, &single).is_ok());
        assert_eq!(single.phases, None);
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(KREA_2_RAW_ID, &spec)
            .unwrap();
        assert!(g.validate(&single).is_ok());
    }
}
