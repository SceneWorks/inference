//! # candle-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sdxl`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and exposes the candle SDXL generator through its explicit
//! family catalog.
//!
//! **txt2img (sc-3675 + sc-3673):** [`SdxlGenerator::generate`] runs the GO-validated epic-3494
//! prototype (`pipeline`) through the contract: dual CLIP → UNet (real CFG) → f16 VAE, emitting
//! `Progress` and honoring `req.cancel`, with **deterministic CPU-seeded noise + the non-ancestral
//! DDIM sampler** (sc-3673) so output is launch-portable per seed.
//!
//! **LoRA/LoKr (sc-5165):** [`load`] accepts `spec.adapters` and merges a trained adapter's delta into
//! the UNet weights at component load (`adapters` + `pipeline`) — the inference half of the native
//! candle trainer, closing the train→infer loop. The descriptor advertises the wired surface
//! (txt2img, negative prompt, guidance, `ddim`, the few-step `lightning` sampler, **LoRA/LoKr**) — NOT
//! the full mlx-gen-sdxl conditioning / accel-sampler surface — so the worker routes the rest to the
//! Python fallback (sc-3678) rather than the candle backend silently dropping a control. The
//! descriptor's `backend` is `"candle"` and `mac_only` is `false` (Windows/CUDA target).
//!
//! **Lightning (sc-6128):** a `req.sampler == "lightning"` request runs the few-step Euler-trailing
//! denoise (`pipeline`) — diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`, ε-pred,
//! `final_sigmas_type="zero"`, **CFG-off** — reusing the backend-neutral `gen_core::sampling`
//! `LightningPolicy` (the same schedule `mlx-gen-sdxl`'s `LightningSampler` drives). This makes the
//! `realvisxl_lightning` model id renderable on the candle (Windows) lane at ~5 steps; base SDXL is
//! unaffected (it keeps the DDIM default).
//!
//! Perf (sc-3674): CLIP loads f16 and the UNet attention runs through fused **flash-attention** when
//! built `--features flash-attn` and the runtime toggle ([`set_flash_attn`], default on) is set.
//!
//! Peak VRAM (sc-4987): the dual CLIP is loaded/run/freed before the UNet+VAE load (staged
//! sequential load), and the VAE decode tiles + blends above 512² output ([`set_vae_tiling`], default
//! on) — together targeting torch-parity peak VRAM at 1024².
//!
//! Component caching across `generate` calls (sc-5037 — a latency win, in tension with sc-4987's
//! mid-call frees) is wired. **RealVisXL + parity (sc-3677):** RealVisXL_V5.0 ships the standard
//! diffusers tree with the *same* `.fp16.safetensors` component filenames as SDXL-base, so it loads
//! through this identical path unmodified (no single-file loader needed); parity with the Python
//! `SdxlDiffusersAdapter` is locked by the CPU parity tests here + `tests/conformance.rs`
//! (`sdxl_conformance` / `realvisxl_conformance` on the CUDA lane). See `pipeline` for the layout
//! finding and the one accepted sampler difference (DDIM vs euler_ancestral, sc-3673).

mod control_provider;
mod memory_strategy;
mod pipeline;
// The PiD backbone (latent-space) tag (epic 7840 / sc-8373), re-exported so `candle-gen-instantid`
// loads the same `sdxl` student through its own `with_pid` (it composes the SDXL VAE).
pub use pipeline::{SdxlLatentDecoder, PID_BACKBONE};

// The vendored, packed-detecting SDXL CLIP text-encoder tower (sc-9527, sc-9089j follow-up to the
// sc-9416 UNet packed-load): its Linear surface (attn q/k/v/out_proj, MLP fc1/fc2, bigG
// text_projection) routes through `candle_gen::quant` so the packed MLX tier's dual CLIP loads
// straight from the packed parts. A pure superset — a dense snapshot takes the stock path unchanged.
pub mod clip;

// Inference-side LoRA/LoKr adapter merge (sc-5165) — folds a trained adapter's delta into the dense
// UNet weights before the stock UNet is built (`pipeline` calls this on the adapter path). The candle
// twin of `mlx-gen-sdxl::adapters`; closes the train→infer loop with the trainer below.
mod adapters;
// The merge entry point + its report are public: the worker can introspect what merged (the candle
// analog of `mlx-gen-sdxl::apply_sdxl_adapters`'s report), and the trainer-verify lane
// (`tests/trainer_e2e.rs`) asserts a trained adapter merges into every target with nothing skipped.
pub use adapters::{merge_adapters, AdditiveReport, MergeReport};
// Adapters on a **packed** (pre-quantized MLX tier) UNet (sc-11103, epic 10765): the distill LoRA rides
// the packed Linears **additively** (`adapters::install_additive`, no dequant — the q4/q8 footprint
// survives) and any conv LoRA folds into the dense convs (`adapters::fold_conv_adapters`). This retired
// the sc-9528 dequant→fold→keep-dense path (`packed_adapters.rs`), which dequantized the FF (the bulk of
// the UNet) and defeated the point of the packed tier for SDXL-Lightning / RealVisXL-Lightning.

// IP-Adapter (sc-5491, epic 5480): the perceiver `Resampler` (`image_proj.*` → image/identity tokens)
// + the decoupled cross-attn K/V pairs (`ip_adapter.*`), the candle twin of `mlx-gen-sdxl::ip_adapter`.
// Built here (not in the InstantID glue crate) so the SDXL IP-Adapter-Plus path (sc-5488) reuses them.
// sc-5488 adds the CLIP-ViT image preprocessing + the `IpImageEncoder` (CLIP image encoder → Resampler).
pub mod ip_adapter;
// CLIP ViT vision tower (sc-5488) — the IP-Adapter image encoder (ViT-H/14 for SDXL, ViT-L/14-336 for
// Kolors), the one net-new model the general IP-Adapter port needs (candle-gen had only the text CLIP).
pub mod vision_encoder;
pub use vision_encoder::{ClipVisionEncoder, VisionConfig};
// The safetensors key→Tensor map for the IP-Adapter / ControlNet loads (non-VarBuilder weights).
// Hoisted to the `candle-gen` core crate (F-060, sc-9044); re-exported here for source compatibility.
pub mod weights;

// Euler / Euler-ancestral sampler (sc-5491) — the InstantID/diffusers-SDXL solver the InstantID
// denoise loop runs (the txt2img `pipeline` runs the unified curated framework: DDIM eta=0 by
// default, sc-10826). Port of `mlx-gen-sdxl::sampler::EulerSampler`, with the ancestral noise
// injected (the loop owns the seeded RNG, sc-3673) and the schedule scalars in host f64 (no Python
// parity to hold ULP-for-ULP).
pub mod sampler;
pub use sampler::EulerAncestralSampler;

// The 4-channel SDXL/Kolors per-step latent preview seam (epic 16948, sc-16954) — the reused epic
// 16624 RGB fit plus the VE→fit-domain renormalization the ε/DDPM cohort needs. `candle-gen-kolors`
// projects through this module rather than restating the coefficients.
pub mod preview;

// InstantID denoise loop + the SDXL conditioning/prior/control/decode helpers (sc-5491) — the candle
// twin of `mlx-gen-sdxl::pipeline`'s `denoise_ip_control` family, composing the IP-Adapter UNet, the
// IdentityNet ControlNet, and the euler-ancestral sampler. Driven by the `candle-gen-instantid` glue.
pub mod denoise;
pub use denoise::{
    decode_image, decode_image_with_tiling, denoise_curated, denoise_ip_control,
    denoise_ip_multi_control, preprocess_control_image, seeded_prior, seeded_sigma_prior,
    text_time_ids, ControlContext, Denoiser,
};

// Long-prompt chunk-and-concatenate for the CLIP encoders (sc-20528) — the A1111/compel "long prompt
// weighting" shape that lets every CLIP-conditioned SDXL family accept a prompt past CLIP's
// architectural 77-token window instead of hard-erroring. Shared by the txt2img `pipeline`, the
// InstantID `conditioning` conditioner, and the `training` caption encoder so the three cannot drift.
mod long_prompt;

// SDXL dual-CLIP conditioning (sc-5491) — penultimate hidden (cross-attn) + pooled text-embeds
// (add_embedding), the micro-conditioning the txt2img `pipeline` skips but `forward_instantid` needs.
pub mod conditioning;
pub use conditioning::SdxlConditioner;

// SDXL component loaders for InstantID (sc-5491) — the vendored UNet (+ add_embedding), the fp16-fix
// VAE, and a diffusers ControlNet, built from an SDXL snapshot. The candle twins of mlx-gen-sdxl's
// load_unet_dtype/load_vae/load_controlnet.
pub mod ldm;
pub mod loaders;
pub use loaders::{
    load_instantid_unet, load_instantid_unet_with_adapters, load_sdxl_controlnet, load_sdxl_vae,
    load_sdxl_vae_encoder, load_vendored_unet_with_adapters,
};

// The SDXL VAE decode path (sc-19753). Ported natively rather than reusing candle-transformers'
// `AutoEncoderKL`, whose decoder internals are private and therefore cannot host the layer-wise,
// normalization-correct bounded decode. Re-exported so the `candle-gen-instantid` glue can hold one
// as a field + pass it to `decode_image` without depending on candle-transformers directly.
pub mod vae_decoder;
pub use vae_decoder::SdxlVaeDecoder;

// The vendored UNet itself, re-exported so the `candle-gen-instantid` glue can hold one + drive its
// InstantID surface (install_ip_adapter / set_ip_context / forward_instantid via the denoise loop).
// `sdxl_unet_config` + `UNet2DConditionModelConfig`/`BlockConfig` are re-exported too so the Kolors
// IP-Adapter provider (sc-5488) can build the same vendored stack from the SDXL-family Kolors UNet.
pub use unet::{
    sdxl_unet_config, BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
    VaeMomentsEncoder,
};

// SDXL IP-Adapter-Plus reference-image provider (sc-5488, epic 5480) — the [`ip_adapter`] +
// [`denoise`] stack composed without a face embedder / ControlNet: CLIP ViT-H image tokens → pure-IP
// denoise. The reference-conditioning sibling of the InstantID glue crate, but for plain SDXL/RealVisXL.
pub mod ip_provider;
pub use ip_provider::{
    IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest, DEFAULT_IP_ADAPTER_SCALE,
};

/// SDXL IP-Adapter-Plus real-weight GPU validation (sc-5488) — env-driven, `#[ignore]`d integration
/// test (the analog of the InstantID Phase-5 harness).
#[cfg(test)]
mod ip_validate;

// SDXL img2img / inpaint / outpaint edit provider (sc-6037, epic 5480) — pixel-conditioned editing,
// the candle twin of the `mlx-gen-sdxl` edit path and the provider half that unblocks the worker
// img2img/edit/inpaint routing (sc-5487). Reuses the IP/InstantID denoise stack with the IP branch
// inert (no install) + the deterministic VAE moments-encoder init + a per-step inpaint mask blend.
pub mod edit_provider;
pub use edit_provider::{
    SdxlEdit, SdxlEditPaths, SdxlEditRequest, DEFAULT_EDIT_STRENGTH, DEFAULT_INPAINT_STRENGTH,
};

// SDXL-family tile-ControlNet img2img provider for the image-detail utility (sc-18480). Kept as an
// explicit route instead of widening the generic `sdxl` descriptor to arbitrary ControlNet shapes.
pub mod detail_provider;
pub use detail_provider::{SdxlDetail, SdxlDetailPaths, SdxlDetailRequest};

/// SDXL edit (img2img / inpaint / outpaint) real-weight GPU validation (sc-6037) — env-driven,
/// `#[ignore]`d integration test (the analog of the IP-Adapter Phase-5 harness).
#[cfg(test)]
mod edit_validate;

// Vendored, training-adapted SDXL UNet + VAE-encode stack (sc-5165) — used by the native LoRA/LoKr
// trainer below. Inference continues to use the stock candle-transformers UNet via `pipeline`; the
// vendored copy retains some unused upstream surface (decoder blocks, the additional-residuals
// forward), hence `allow(dead_code)`.
#[allow(dead_code)]
mod unet;

// The InstantID inference surface built on the vendored UNet (sc-5491, epic 5480): the SDXL ControlNet
// branch (the IdentityNet) — an encoder copy + conditioning embedding + zero-conv heads producing the
// scaled down/mid residuals the InstantID UNet adds in. Re-exported so the `candle-gen-instantid` glue
// crate composes it; also the SDXL ControlNet building block sc-5489 reuses.
pub use unet::{ControlNet, ControlNetConfig, ControlResiduals};

// The native candle SDXL LoRA/LoKr trainer (sc-5165) implements `gen_core::Trainer` and is included
// in the explicit family catalog.
mod training;
pub use training::{load_trainer, trainer_descriptor, SdxlTrainer};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, PidWeights, Progress, Quant, WeightsSource,
};

use control_provider::SdxlControlGenerator;
use pipeline::{Components, Pipeline, SdxlComponents};

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["sdxl"]`). The
/// worker maps both `sdxl` and `realvisxl` onto engine id `"sdxl"`, so — exactly like
/// `mlx-gen-sdxl` — this crate registers a SINGLE descriptor under `"sdxl"`.
pub const MODEL_ID: &str = "sdxl";

pub use memory_strategy::{
    provider_contract_for_spec,
    resolved_numeric_tier,
    // Exported so the sibling providers that load *this* crate's vendored SDXL UNet + VAE
    // (candle-gen-instantid, sc-5491) publish the identical architecture axes at their own pinned
    // activation width rather than duplicating the geometry constants.
    sdxl_unet_family_architecture_facts,
    SdxlArtifactSeal,
    SdxlSurface,
    REQUEST_EVIDENCE_REVISION,
    SDXL_ROUTES,
};

/// SDXL works in latent space at /8: both dims must be multiples of 8. Exposed as the pinned-engine
/// stride SceneWorks ties each advertised SDXL image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`. `validate` enforces exactly this value, so the const cannot drift
/// from the check; the bespoke edit provider imports this same crate-root const.
pub const SIZE_MULTIPLE: u32 = 8;

/// Process-global flash-attention runtime toggle (sc-3674). This switch was designed to decide
/// whether a flash-attn-capable build actually *uses* the fused kernels, so the SceneWorks UI can
/// expose it (defaulted on) and the worker flips it from settings — without recompiling. Mirrors
/// `mlx-gen-sdxl::set_compile_glue`. **sc-9032:** the `flash-attn` cargo feature it was ANDed with
/// was a no-op alias (`= ["cuda"]`, no `candle-flash-attn` wired) and was removed; `components` now
/// hard-codes the flash path off, so this toggle is retained as public worker API but is inert until
/// the fused-kernel slice lands and re-gates the load on it. Default **on**.
static FLASH_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable flash-attention for subsequently-loaded pipelines (sc-3674). Process-global; the
/// worker calls this from its `backend_candle`/flash setting at startup. Inert since sc-9032 removed
/// the no-op `flash-attn` feature — no fused kernels are compiled in (retained as worker API).
pub fn set_flash_attn(on: bool) {
    FLASH_ATTN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether flash-attention is currently enabled (the runtime toggle, [`set_flash_attn`]). Since
/// sc-9032 the pipeline hard-codes the flash path off (the no-op `flash-attn` feature was removed),
/// so this returning `true` does not enable anything.
pub fn flash_attn_enabled() -> bool {
    FLASH_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-global VAE-tiling runtime toggle (sc-4987). When on, the VAE decode tiles the latent into
/// overlapping 64²-latent (512²-output) tiles and trapezoidally blends the seams — bounding the
/// decode's peak VRAM to one tile (the tallest single allocation at 1024², for torch-parity). Unlike
/// flash-attn there is no build feature: it is pure candle, so the switch alone decides. It only
/// *fires* above 512² output (smaller renders stay monolithic), so leaving it on is free at/below
/// 512². The SceneWorks worker/UI drives it; default **on** to hit the <12 GiB target out of the box.
static VAE_TILING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable VAE tiling for subsequent decodes (sc-4987). Process-global; the worker drives it
/// from its backend setting. Off restores the monolithic single-pass decode (higher peak VRAM).
pub fn set_vae_tiling(on: bool) {
    VAE_TILING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether VAE tiling is currently enabled (the runtime toggle, [`set_vae_tiling`]). The pipeline
/// additionally only tiles when the output exceeds the 512² threshold, so this returning `true` does
/// not change ≤512² output.
pub fn vae_tiling_enabled() -> bool {
    VAE_TILING.load(std::sync::atomic::Ordering::Relaxed)
}

/// A loaded candle SDXL generator. Loading is **lazy**: `load` does no file I/O (registry
/// introspection against a missing path still resolves), and the heavy UNet/VAE are built on the
/// first [`generate`](Generator::generate) call. sc-5037: those `Components` are then **cached** in
/// `components` and reused across subsequent calls (keyed by the flash-attn setting), so back-to-back
/// requests skip the ~7 GiB UNet/VAE disk re-read. CLIP is intentionally not cached — it stays
/// load-on-demand-and-free (the sc-4987 peak-VRAM lever), so the cache is a latency win that does not
/// raise the ~8.7 GiB high-water mark.
pub struct SdxlGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters to merge into the UNet weights at first-`generate` component load (sc-5165).
    /// Fixed for the generator's lifetime (from the `LoadSpec`), so they sit outside the component
    /// cache key. Empty ⇒ the stock no-adapter UNet load.
    adapters: Vec<AdapterSpec>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the UNet/VAE. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// The three caller-staged SDXL components (epic 13657, sc-13663): the CLIP-L/bigG tokenizers +
    /// the fp16-fix VAE, validated at [`load`] and threaded into the lazy tokenizer/UNet+VAE build in
    /// place of the deleted render-path self-fetch.
    component_paths: SdxlComponents,
    /// Cached UNet+VAE + the flash-attn flag they were built with. `Mutex` because `Generator` is
    /// shared and `generate` takes `&self`; the lock is held only to read/populate the cache (a
    /// cheap `Arc` clone or a one-time load), never across the denoise.
    components: Mutex<Option<(bool, Components)>>,
    /// Cached dual CLIP tokenizers, loaded+parsed once and reused across `generate` calls (sc-8991 /
    /// F-011) rather than re-reading `tokenizer.json` from the staged component on every text encode. Shared
    /// behind an `Arc` (model-agnostic); `lock_recover` mirrors the components-cache poison recovery.
    tokenizers: Mutex<Option<Arc<pipeline::SdxlTokenizers>>>,
    ldm: Option<Arc<ldm::LdmComponents>>,
    /// Exact source token retained for the lifetime of an imported fused checkpoint. The checkpoint
    /// is split eagerly, but retaining the token keeps deferred component/adaptor materialization
    /// tied to the same caller-prepared identity.
    _ldm_source_pin: Option<gen_core::PinnedWeightsFile>,
    /// Complete prepared-file set, reused when the lazy UNet/VAE + adapter component cache fills.
    file_pin_spec: LoadSpec,
    /// Real load-time affine fold for an imported fused checkpoint. Snapshot directories select
    /// their packed tier on disk and leave this unset.
    quant: Option<Quant>,
    memory_contract: Option<gen_core::MemoryProviderContract>,
    artifact_seal: Option<SdxlArtifactSeal>,
}

impl SdxlGenerator {
    /// Get the cached UNet/VAE, loading (and caching) them on a miss. Keyed by the effective
    /// flash-attn setting (`build_unet` bakes it in, sc-3674), so flipping [`set_flash_attn`] between
    /// calls rebuilds rather than serving a stale UNet. The lock is held over the cache-miss load
    /// (concurrent first-callers serialize on it) but released before the caller's denoise.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // sc-9032: the `flash-attn` cargo feature (a no-op alias for `cuda`) was removed. No fused
        // candle-flash-attn kernel is wired, so the flash path is never taken — `false` here is
        // byte-identical to the old `cfg!(feature = "flash-attn") && flash_attn_enabled()`, which
        // always resolved false in every buildable config. The `set_flash_attn` runtime toggle stays
        // (public worker API) but is inert until the fused-kernel slice lands and re-gates this.
        let flash = false;
        // sc-9015 / F-031: recover from a poisoned lock (overwrite-on-miss cache; a prior panic
        // while locked must not turn every later `generate` into a panic).
        let mut guard = candle_gen::lock_recover(&self.components);
        if let Some((cached_flash, comps)) = guard.as_ref() {
            if *cached_flash == flash {
                return Ok(comps.clone());
            }
        }
        let comps = self
            .file_pin_spec
            .read_prepared_files_unchanged(|| pipe.load_components(flash))?;
        *guard = Some((flash, comps.clone()));
        Ok(comps)
    }

    /// Get the cached dual CLIP tokenizers, loading (and caching) them on a miss (sc-8991 / F-011). The
    /// tokenizers are model-agnostic (fixed upstream repos) so a single pair serves every request; parsing
    /// them once here removes the tens-of-ms `tokenizer.json` re-parse the per-encode load did. The
    /// shared [`candle_gen::cached`] read-through recovers a poisoned lock internally (the F-031 idiom).
    fn tokenizers(&self) -> gen_core::Result<Arc<pipeline::SdxlTokenizers>> {
        self.file_pin_spec.read_prepared_files_unchanged(|| {
            candle_gen::cached(&self.tokenizers, || {
                Ok(Arc::new(pipeline::SdxlTokenizers::load(
                    &self.component_paths.tokenizer_clip_l,
                    &self.component_paths.tokenizer_clip_bigg,
                )?))
            })
        })
    }
}

impl Generator for SdxlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_contract.as_ref() else {
            return gen_core::default_memory_strategy_safety_check(
                &gen_core::MemoryProviderContract::compatibility_default(
                    MODEL_ID,
                    memory_strategy::backend(),
                ),
                context,
            );
        };
        memory_strategy::safety_check(contract, context, self.artifact_seal.as_ref())
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_contract.as_ref() else {
            return Ok(None);
        };
        let seal = self.artifact_seal.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported(
                "sdxl: authoritative memory contract has no artifact seal".into(),
            )
        })?;
        seal.ensure_unchanged()?;
        memory_strategy::validate_context(contract, context, seal)?;
        Ok(Some(Box::new(memory_strategy::request_scope(
            self.device.clone(),
            contract,
            context,
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size range/guidance/negative/sampler/conditioning):
        // since the descriptor advertises NO conditioning, any conditioning entry is rejected here.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific floor on top of the shared one (mirrors mlx-gen-sdxl::validate_request).
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl: prompt must not be empty".into(),
            ));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure scaled noise — reject loudly (a derived
        // 0 from img2img strength would be a legitimate no-op, but this is txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "sdxl: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // F-116: `lightning` (sc-6128) runs its own fixed Euler-trailing schedule and never consults
        // `req.scheduler` — so a caller-selected NON-DEFAULT curated scheduler (karras / sgm_uniform /
        // …) would be silently dropped (the F-004 shape). Reject that combination loudly instead of
        // misleading. Only a recognized non-default scheduler is rejected; the default `normal`, `None`,
        // and any unrecognized value (e.g. the native-fallback `discrete` alias) all pass — a worker
        // that always populates `scheduler:"normal"` alongside `lightning` must not hard-error, and
        // every one of these resolves to lightning's own trailing schedule anyway (nothing dropped).
        if req.sampler.as_deref() == Some(pipeline::LIGHTNING_SAMPLER) {
            if let Some(sched) = req.scheduler.as_deref() {
                let recognized = gen_core::sampling::Scheduler::from_name(sched);
                if recognized.is_some_and(|s| s != gen_core::sampling::Scheduler::Normal) {
                    return Err(gen_core::Error::Msg(format!(
                        "sdxl: the `lightning` sampler uses its own fixed trailing schedule and \
                         ignores the `scheduler` axis (got `{sched}`) — omit `scheduler` (or use \
                         `normal`) for lightning"
                    )));
                }
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if let Some(seal) = &self.artifact_seal {
            seal.ensure_unchanged()?;
        }
        let staged = req.memory.is_some_and(|memory| memory.stage_residency);
        let _staged_guard = staged.then(|| StagedCacheGuard::new(&self.components));
        // The rich-`CandleError` tail — including the typed `Canceled` — bridges into
        // `gen_core::Error` via `?` (the From bridge). The light `Pipeline` handle carries this
        // request's latent dims; the heavy UNet/VAE come from the cache.
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.dtype,
            req.width,
            req.height,
            &self.adapters,
            self.pid_spec.clone(),
            self.component_paths.vae_fp16_fix.clone(),
            self.ldm.clone(),
            self.quant,
        )?;
        // Encode text FIRST (loads + frees CLIP) so the cold-call ordering — CLIP gone before the
        // UNet/VAE are resident — is preserved (sc-4987); only then acquire the cached UNet/VAE
        // (sc-5037). On a warm call the UNet/VAE are already resident, but CLIP loads one encoder at a
        // time, so the footprint stays under the denoise-time peak.
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        if let Some(seal) = &self.artifact_seal {
            seal.ensure_unchanged()?;
        }
        let tokenizers = self.tokenizers()?;
        let text_embeddings = pipe.text_embeddings(&tokenizers, &req.prompt, negative)?;
        if let Some(seal) = &self.artifact_seal {
            seal.ensure_unchanged()?;
        }
        let components = self.components(&pipe)?;
        let images = with_request_attention_budget(req, || {
            pipe.render(
                req,
                &text_embeddings,
                &components.unet,
                &components.vae,
                components.pid.as_deref(),
                on_progress,
            )
        })?;
        Ok(GenerationOutput::Images(images))
    }
}

struct StagedCacheGuard<'a> {
    components: &'a Mutex<Option<(bool, Components)>>,
}

impl<'a> StagedCacheGuard<'a> {
    fn new(components: &'a Mutex<Option<(bool, Components)>>) -> Self {
        candle_gen::lock_recover(components).take();
        Self { components }
    }
}

impl Drop for StagedCacheGuard<'_> {
    fn drop(&mut self) {
        candle_gen::lock_recover(self.components).take();
    }
}

thread_local! {
    static REQUEST_ATTENTION_BUDGET: std::cell::Cell<usize> =
        const { std::cell::Cell::new(candle_gen::ATTN_SCORES_BUDGET) };
}

pub(crate) fn request_attention_budget() -> usize {
    REQUEST_ATTENTION_BUDGET.get()
}

fn with_request_attention_budget<T>(request: &GenerationRequest, f: impl FnOnce() -> T) -> T {
    with_attention_memory(request.memory, f)
}

pub(crate) fn with_attention_memory<T>(
    memory: Option<gen_core::GenerationMemory>,
    f: impl FnOnce() -> T,
) -> T {
    let _guard = enter_attention_memory(memory);
    f()
}

pub(crate) struct AttentionMemoryGuard(usize);

impl Drop for AttentionMemoryGuard {
    fn drop(&mut self) {
        REQUEST_ATTENTION_BUDGET.set(self.0);
    }
}

pub(crate) fn enter_attention_memory(
    memory: Option<gen_core::GenerationMemory>,
) -> AttentionMemoryGuard {
    let previous = REQUEST_ATTENTION_BUDGET.get();
    let selected = memory
        .filter(|memory| memory.chunk_attention)
        .and_then(|memory| memory.attention_chunk_size)
        .map(|budget| budget as usize)
        .unwrap_or(candle_gen::ATTN_SCORES_BUDGET);
    REQUEST_ATTENTION_BUDGET.set(selected);
    AttentionMemoryGuard(previous)
}

/// SDXL's identity + the surface candle wires: real classifier-free guidance (negative prompt + CFG
/// scale), txt2img, `ddim`, the few-step **`lightning`** sampler (sc-6128 — Euler-trailing, CFG-off,
/// for distilled Lightning checkpoints), and **LoRA/LoKr** (sc-5165 — load-time merge of a trained
/// adapter into the UNet weights, see [`load`] + `pipeline`). The registry declares the optional
/// ControlNet overlay so callers can stage it; a base load without [`LoadSpec::control`] keeps a
/// narrower descriptor. Other acceleration samplers (lcm/hyper) remain the Python fallback's job
/// (sc-3678). Two backend-correct deviations from `mlx-gen-sdxl`: `backend = "candle"` and
/// `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    let mut descriptor = plain_descriptor();
    descriptor.control_kinds = Some(gen_core::AcceptedControlKinds::Any);
    descriptor.capabilities.conditioning = vec![gen_core::ConditioningKind::Control];
    descriptor
}

/// The narrower descriptor installed on an ordinary base load without a ControlNet checkpoint.
/// Keeping it separate prevents a request from being admitted and then rendered without its control
/// residuals.
fn plain_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::SDXL_LATENT_SPACE),
        control_kinds: None,
        // epic 13657 (sc-13663): SDXL requires three caller-staged components — the two model-agnostic
        // CLIP tokenizers + the fp16-fix VAE — that used to be self-fetched from pinned upstream repos
        // on the render path. Advertised so SceneWorks stages them, and `load` fails fast if it doesn't.
        required_components: pipeline::REQUIRED_COMPONENTS,
        id: MODEL_ID,
        family: "sdxl",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            //
            // sc-14195 — the `guidance ≤ 1.0` contract, stated once here because it governs BOTH of
            // these flags. `guidance > 1.0` engages CFG; anything at or below 1.0 is **accepted and
            // runs the CFG-off single-branch path** (one conditioned forward per step) rather than
            // being rejected. That is diffusers' own rule (`do_classifier_free_guidance =
            // guidance_scale > 1`), and at exactly 1.0 it is provably the same result — the combine
            // `uncond + 1.0·(cond − uncond)` reduces to `cond` — for half the compute. Sub-1.0 and
            // non-positive values fold into the same branch rather than interpolating toward the
            // uncond prediction; they are not rejected, since the request is still satisfiable.
            //
            // Consequence, deliberate: with CFG off there is no uncond branch, so a supplied
            // `negative_prompt` has nothing to act on and is dropped. This matches diffusers and the
            // `lightning` lane (also CFG-free, also drops it). It is NOT the F-004 "reject a
            // silently-dropped axis" shape — that rule fires when a request names an axis the engine
            // cannot honor at all (e.g. `lightning` + a curated scheduler); here the user chose the
            // guidance value that switches the negative off, so honoring it literally IS the drop.
            supports_negative_prompt: true,
            supports_guidance: true,
            // The base load is txt2img only. The registry-level descriptor widens this to Control
            // when the caller stages `LoadSpec::control`; an unloaded base must reject every
            // conditioning shape rather than silently drop one.
            conditioning: vec![],
            // sc-5165: a trained LoRA/LoKr adapter is merged into the UNet weights at load (`load` +
            // `pipeline::Pipeline::load_components`). Advertised so the worker routes adapter jobs here
            // rather than to the Python fallback.
            supports_lora: true,
            supports_lokr: true,
            // DDIM (eta=0) is the deterministic, launch-portable DEFAULT (sc-3673); `lightning`
            // (sc-6128) is the few-step Euler-trailing path for distilled checkpoints. epic 7114 P4
            // (sc-7124) added the curated ε/DDPM menu (euler / euler_ancestral / heun / dpmpp_2m /
            // dpmpp_sde / uni_pc / lcm + ddim) over `DiscreteModelSampling`, plus the curated σ-schedule
            // axis (normal / karras / sgm_uniform / …). sc-10826: every non-`lightning` render — the
            // omitted-sampler default (→ curated `ddim`) AND every named solver — now routes the curated
            // EPS path; the native candle-transformers DDIM inference loop (which ghosted on the default)
            // is gone, so `ddim` and the default take the clean curated `ddim` solver. `lightning` keeps
            // its native few-step path. The legacy `discrete` scheduler alias falls back to the native schedule.
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &["lightning"],
            ),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["discrete"],
            ),
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // Packed q4/q8 MLX-tier inference (sc-9416 UNet + sc-9527 dual-CLIP + sc-9528 adapter fold)
            // is wired end-to-end, so advertise Q4/Q8 (sc-10767, epic 9083 full-catalog parity). The
            // tier is packed-detected from disk (`detect_packed_unet` / `detect_packed_clip`); the
            // LoadSpec `quant` overlay is an advisory no-op on an already-packed tier (as with
            // boogu/flux2-dev). bf16 tiers stay dense (Quant::None), verbatim.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Per-step latent previews (epic 16948, sc-16954): every shipped SDXL render lane
            // emits -- the curated driver lane and the bespoke Lightning loop on this registered
            // route, plus the name-driven edit / IP-Adapter providers. `crate::preview` reuses the
            // epic-16624 four-channel fit; the trainer's sample render stays deliberately dark.
            supports_preview: true,
            ..Default::default()
        },
    }
}

/// Construct the (lazy) candle SDXL generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-xl-base-1.0`-layout snapshot
/// (the diffusers multi-component tree: `text_encoder/`, `text_encoder_2/`, `unet/`, …).
///
/// `spec.adapters` (sc-5165) are LoRA/LoKr adapters to **merge into the UNet weights** — folded in at
/// the first `generate`'s component load (this `load` stays lazy: no file I/O here), via
/// `adapters::merge_adapters`. PEFT (the candle trainer's format) + kohya LoRA and
/// PEFT/kohya LoKr are supported; an adapter that matches no UNet target errors at that first
/// `generate` rather than rendering an unadapted image silently.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    // Validate the model-agnostic staged components before opening a potentially multi-gigabyte
    // fused checkpoint.
    let component_paths = SdxlComponents::from_spec(spec, MODEL_ID)?;
    let authoritative_seal = if spec.resolved_route.is_some() {
        Some(SdxlArtifactSeal::capture(spec)?)
    } else {
        None
    };
    // A ControlNet changes the graph: route it through the vendored SDXL stack that carries the
    // add-embedding and residual-injection seams.  Do not let the historical txt2img pipeline
    // accept an overlay it would never read.
    if spec.control.is_some() {
        return Ok(Box::new(SdxlControlGenerator::new(
            spec,
            component_paths,
            candle_gen::default_device()?,
            authoritative_seal,
        )?));
    }
    if !spec.extra_controls.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "sdxl: extra ControlNet checkpoints require LoadSpec::control as the first branch"
                .into(),
        ));
    }
    if spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "sdxl: generic SDXL does not accept an IP-Adapter overlay; use its dedicated provider"
                .into(),
        ));
    }
    if matches!(spec.weights, WeightsSource::File(_)) && matches!(spec.quantize, Some(Quant::Nvfp4))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: imported fused checkpoints support Q4/Q8 affine tiers, not NVFP4"
        )));
    }
    let (memory_contract, artifact_seal) = if let Some(seal) = authoritative_seal {
        (Some(seal.contract().clone()), Some(seal))
    } else {
        (None, None)
    };
    let (root, ldm, ldm_source_pin) = match &spec.weights {
        WeightsSource::Dir(p) => (p.clone(), None, None),
        WeightsSource::File(file) => {
            let pin = spec
                .weights_file_pin()?
                .expect("File weights must resolve to a pin");
            let components = pin.read_unchanged(ldm::split_ldm_checkpoint)?;
            (
                file.parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf(),
                Some(Arc::new(components)),
                Some(pin),
            )
        }
    };
    // epic 13657 (sc-13663) load gate: resolve + validate the three passed-in components
    // (`tokenizer_clip_l` / `tokenizer_clip_bigg` / `vae_fp16_fix`) before any work — a missing one is
    // a caller-actionable load-time error, and an unrecognized key is rejected. `load` stays lazy (the
    // component paths are read at the first `generate`), but the contract is checked up front here.
    // SDXL is fp16 (the production reference dtype) regardless of the CPU-default dtype; the device
    // is the backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    Ok(Box::new(SdxlGenerator {
        descriptor: plain_descriptor(),
        root,
        device,
        dtype: DType::F16,
        adapters: spec.adapters.clone(),
        component_paths,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters, it is not rejected
        // — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
        tokenizers: Mutex::new(None),
        quant: ldm.as_ref().and(spec.quantize),
        ldm,
        _ldm_source_pin: ldm_source_pin,
        file_pin_spec: spec.clone(),
        memory_contract,
        artifact_seal,
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// the explicit family and platform catalogs resolve the candle generator.
candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

/// Add the Candle SDXL generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry.register_generator(REGISTRATION);
    register_memory_contract_surfaces(registry)
        .register_checkpoint_adapter(gen_core::CheckpointAdapterRegistration {
            backend_bindings: &[
                gen_core::CheckpointBackendBindingRegistration {
                    backend: gen_core::CheckpointBackend::Candle,
                    source: gen_core::ImportedModelSource::FusedCheckpoint,
                    operation: gen_core::ImportedModelOperation::Generate,
                    provider_id: MODEL_ID,
                    required_components: Some(pipeline::LDM_REQUIRED_COMPONENTS),
                    inherit_adapters: true,
                },
                // **No `ImportedModelOperation::Edit` binding on Candle** (sc-20651 review, item 8).
                //
                // One was carried across from `main` when this epic replaced
                // `register_imported_model` with the checkpoint adapter, on the reasoning that a
                // capability `main` had added should not be silently dropped by the port. That
                // reasoning was wrong about what `main` had added: the binding named
                // `provider_id: MODEL_ID`, and `MODEL_ID`'s registered generator is
                // [`SdxlGenerator`], whose descriptor advertises `conditioning: [Control]` on the
                // registry route and `[]` on a plain load. `Capabilities::validate_request` rejects
                // every conditioning kind a descriptor does not list, so an Edit import resolving
                // through that binding got a generator that refuses `Conditioning::Reference` —
                // i.e. the operation was **declared and unreachable**, which is worse than absent:
                // SceneWorks stages components and routes a request that can only end in a refusal.
                //
                // It is not a wiring oversight; the fused route cannot reach the edit stack as
                // either side is currently shaped. Naming the exact blockers, since "it's hard" is
                // not evidence:
                //
                //  1. `registry.rs`'s builder requires a binding's `provider_id` to name a
                //     **registered `Generator`**. `edit_provider::SdxlEdit` does not implement
                //     `gen_core::Generator` at all — it has a bespoke `generate(&SdxlEditRequest,
                //     &Image, ..)` / `generate_masked(.., &Image, &Image, ..)` surface that takes
                //     the source image as a positional argument, not as request conditioning.
                //  2. `SdxlEdit::load` takes `SdxlEditPaths { sdxl_base: PathBuf, .. }` and hands
                //     it to `SdxlConditioner::load(root: &Path, ..)`, which reads the diffusers
                //     `text_encoder/` + `text_encoder_2/` subtrees (`load_clip_tower_with_vb`, for
                //     the bigG `text_projection` head) and to
                //     `loaders::load_instantid_unet_with_adapters(root, ..)`, which reads `unet/`.
                //     Both are directory-shaped. There is no `LdmComponents`-fed constructor for
                //     either: `ldm::split_ldm_checkpoint` has exactly one caller, `load` below,
                //     which threads its tensor maps into `pipeline::Pipeline` — a type `SdxlEdit`
                //     does not use and cannot borrow a VAE **encoder** from (`Pipeline` builds only
                //     `SdxlVaeDecoder` on the fused path; `VaeMomentsEncoder` has no fused caller).
                //  3. `SdxlEditPaths.vae_fp16_fix` is required, but this route's
                //     `LDM_REQUIRED_COMPONENTS` is `[tokenizer_clip_l, tokenizer_clip_bigg]` —
                //     deliberately no VAE, because a fused checkpoint carries its own. Nothing on
                //     the fused route can supply that field.
                //  4. `memory_strategy::validate_bespoke_spec_common` (via
                //     `SdxlEdit::load_admitted`) hard-requires
                //     `matches!(spec.weights, WeightsSource::Dir(p) if p == base)` and refuses a
                //     `WeightsSource::File` outright.
                //
                // Dropping the binding is structurally permitted: `SDXL_CHECKPOINT_ADAPTER`'s
                // `eligible_backends` is `[Mlx, Candle]`, so the builder's per-operation
                // completeness check (which fires only for a *sole* eligible backend) does not
                // require Candle to implement every operation — the same way Candle Krea
                // truthfully omits the MLX-only Pose route. MLX keeps its Edit binding, and its
                // descriptor really does declare `Reference`/`Mask` and really does consume them.
                //
                // `SdxlEdit` itself is untouched and still reachable as the name-driven worker
                // provider it has always been (a diffusers **snapshot dir** + a staged fp16-fix
                // VAE). What is gone is only the claim that a *fused single-file import* can edit.
                // Making that claim true is a feature, not a review fix: it needs
                // `LdmComponents`-fed CLIP/UNet/VAE-encoder constructors plus an edit-capable
                // registered generator id, mirroring `candle-gen-krea`'s `krea_2_edit`.
            ],
            ..gen_core::SDXL_CHECKPOINT_ADAPTER
        })
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Register SDXL's exhaustive weights-free q4/q8/bf16 contract and behavior surface.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::memory_registration())
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: MODEL_ID,
            contract: memory_strategy::weights_free_contract,
        })
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
}

/// Build the complete explicit Candle SDXL provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    use super::{gen_core, pipeline};

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

        assert_eq!(explicit_generators, ["sdxl"]);
        assert_eq!(explicit_trainers, ["sdxl"]);
        let imported = registry
            .imported_model_descriptor(
                "sdxl",
                gen_core::ImportedModelSource::FusedCheckpoint,
                gen_core::ImportedModelOperation::Generate,
            )
            .expect("fused SDXL route");
        assert_eq!(
            imported.required_components,
            pipeline::LDM_REQUIRED_COMPONENTS
        );
    }

    /// sc-20651 review (8). **Every checkpoint-import operation this crate binds must resolve to a
    /// descriptor that can actually accept the conditioning that operation is made of.**
    ///
    /// This is deliberately not `assert_eq!(operations, [Generate])` — that would only pin today's
    /// list and would go green again the moment someone re-adds an `Edit` binding pointing at a
    /// txt2img generator, which is exactly the defect. Instead it asks the question the registry
    /// consumer asks: resolve the binding the way SceneWorks does
    /// (`imported_model_descriptor(family, source, operation)`), then check the resolved descriptor
    /// against `Capabilities::accepts` for the kind that operation *requires*. An `Edit` route whose
    /// descriptor cannot take a `Reference` is a route that can only ever refuse.
    ///
    /// `Generate` requires no conditioning, so it is checked for resolvability only.
    #[test]
    fn every_bound_import_operation_resolves_to_a_descriptor_that_accepts_its_conditioning() {
        use gen_core::{ConditioningKind, ImportedModelOperation};

        let registry = super::provider_registry().unwrap();
        // The operation → conditioning kind an import of that operation cannot work without.
        let required_kind = |operation: ImportedModelOperation| match operation {
            ImportedModelOperation::Generate => None,
            _ => Some(ConditioningKind::Reference),
        };

        let mut checked = 0;
        for operation in [
            ImportedModelOperation::Generate,
            ImportedModelOperation::Edit,
        ] {
            let Some(descriptor) = registry.imported_model_descriptor(
                "sdxl",
                gen_core::ImportedModelSource::FusedCheckpoint,
                operation,
            ) else {
                // Not bound at all is fine and honest — the adapter's `eligible_backends` is
                // `[Mlx, Candle]`, so Candle is not obliged to implement every operation.
                continue;
            };
            checked += 1;
            if let Some(kind) = required_kind(operation) {
                assert!(
                    descriptor.capabilities.accepts(kind),
                    "fused-checkpoint {operation:?} binds provider '{}', whose descriptor does not \
                     accept {kind:?} — the operation is declared but every request through it is \
                     refused by the capability floor",
                    descriptor.id
                );
            }
        }
        // A guard that resolved nothing asks no question.
        assert!(
            checked > 0,
            "no fused-checkpoint route resolved; this crate must bind at least Generate"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// A lazy-loadable SDXL spec: the given `weights` plus the three staged components (epic 13657,
    /// sc-13663) the load gate now requires. The component paths are never read until `generate`, so
    /// nonexistent placeholders are fine for the lazy-resolution + validate tests here.
    fn sdxl_spec(weights: WeightsSource) -> LoadSpec {
        LoadSpec::new(weights)
            .with_component(
                "tokenizer_clip_l",
                WeightsSource::File("/nonexistent/clip_l/tokenizer.json".into()),
            )
            .with_component(
                "tokenizer_clip_bigg",
                WeightsSource::File("/nonexistent/clip_bigg/tokenizer.json".into()),
            )
            .with_component(
                "vae_fp16_fix",
                WeightsSource::File("/nonexistent/vae.safetensors".into()),
            )
    }

    /// The seam under test: resolving `"sdxl"` through the family registry returns this candle
    /// generator. `load` is
    /// lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn sdxl_registers_and_resolves_as_candle() {
        let spec = sdxl_spec(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("sdxl", &spec)
            .expect("candle sdxl is registered");
        assert_eq!(g.descriptor().id, "sdxl");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        // The registry declares ControlNet so orchestration stages the overlay. A bare base load has
        // a narrower dynamic descriptor and rejects it; that is asserted below.
        assert_eq!(d.capabilities.conditioning, vec![ConditioningKind::Control]);
        assert_eq!(d.control_kinds, Some(gen_core::AcceptedControlKinds::Any));
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        // sc-10767 (epic 9083): the packed q4/q8 MLX-tier inference path (sc-9416/9527/9528) is now
        // advertised, so the worker keeps quant tier-selects on the candle lane rather than deferring.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // sc-7124: the curated ε/DDPM sampler menu + the native `lightning` alias; `ddim` is part of
        // the curated vocabulary and (sc-10826) is now the omitted-sampler default's solver too. The
        // curated scheduler axis + `discrete`.
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::menu_with_aliases(candle_gen::curated_sampler_names(), &["lightning"])
        );
        assert!(d.capabilities.samplers.contains(&"ddim"));
        assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
        assert!(d.capabilities.samplers.contains(&"lightning"));
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::menu_with_aliases(candle_gen::curated_scheduler_names(), &["discrete"])
        );
        // epic 13657 (sc-13663): SDXL advertises its three passed-in components (the CLIP tokenizers +
        // the fp16-fix VAE) that used to be self-fetched from pinned upstream repos on the render path.
        assert_eq!(
            d.required_components,
            &["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"]
        );
    }

    /// epic 13657 (sc-13663): the load gate. Every declared `required_components` id, removed in turn,
    /// makes `load` fail with a caller-actionable error that names the missing id (not a mid-render
    /// fetch), and an unrecognized component key is rejected at load. Weights-free — the testkit drives
    /// the real `load`, which stays lazy, so the placeholder component/weights paths are never read.
    #[test]
    fn missing_or_unknown_component_fails_at_load() {
        let base = sdxl_spec(WeightsSource::Dir("/nonexistent".into()));
        gen_core_testkit::check_component_load_gate(load, &base, descriptor().required_components)
            .expect("SDXL load must gate on every declared required component");
    }

    /// sc-3677 parity: the worker maps BOTH `sdxl` and `realvisxl` onto this single descriptor, so
    /// the contract surface it reads (capability advertisement + request validation) is identical for
    /// the two model ids. This pins the parity-relevant shape the Python `SdxlDiffusersAdapter` path
    /// is reconciled against — dims policy (min/max size, the latent-/8 size multiple), batch ceiling,
    /// and the deterministic `ddim` sampler. The accepted *differences* (DDIM vs the adapter's
    /// euler_ancestral default, sc-3673; the txt2img-only surface routing conditioning/LoRA to the
    /// Python fallback, sc-3678) are documented in the crate docs + tests/conformance.rs.
    #[test]
    fn realvisxl_shares_the_sdxl_contract_surface() {
        let d = descriptor();
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.capabilities.min_size, 512);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.max_count, 8);
        // sc-7124: the curated sampler menu (incl. `ddim`) + the native `lightning` alias.
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::menu_with_aliases(candle_gen::curated_sampler_names(), &["lightning"])
        );
        // SDXL works in latent space at /8 — the size policy both ids share (validate rejects
        // non-multiples). Anchored here so a change to the alignment is a parity-visible diff.
        assert_eq!(SIZE_MULTIPLE, 8);
    }

    /// sc-3674: the flash-attn runtime toggle defaults on and round-trips (what the worker/UI drive).
    #[test]
    fn flash_attn_toggle_roundtrips() {
        assert!(
            flash_attn_enabled(),
            "flash-attn runtime toggle defaults on"
        );
        set_flash_attn(false);
        assert!(!flash_attn_enabled());
        set_flash_attn(true);
        assert!(flash_attn_enabled());
    }

    /// sc-4987: the VAE-tiling runtime toggle defaults on (to hit the <12 GiB target out of the box)
    /// and round-trips — what the worker/UI drive.
    #[test]
    fn vae_tiling_toggle_roundtrips() {
        assert!(
            vae_tiling_enabled(),
            "vae-tiling runtime toggle defaults on"
        );
        set_vae_tiling(false);
        assert!(!vae_tiling_enabled());
        set_vae_tiling(true);
        assert!(vae_tiling_enabled());
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly (not silently
    /// served). Uses the lazy generator so no weights are needed.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = sdxl_spec(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("sdxl", &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        // Empty prompt, non-multiple-of-8 size, explicit 0 steps, and any conditioning are rejected
        // by this base (no-overlay) load.
        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1020,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                }],
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![Conditioning::Control {
                    image: Image::default(),
                    kind: gen_core::ControlKind::Pose,
                    scale: None,
                }],
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
        // The registry advertises ControlNet, but this *base load* deliberately does not: the
        // capability and the loaded graph stay aligned, so an overlay can never be dropped.
        assert!(!g
            .descriptor()
            .capabilities
            .accepts(ConditioningKind::Control));

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SDXL bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 4 but not
        // SIZE_MULTIPLE (8) is rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 8);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1020, // 255×4 — a multiple of 4 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 8"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 128×8 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    /// sc-6128: `validate` accepts the advertised `lightning` sampler (the worker forces it for
    /// `realvisxl_lightning`) and still rejects an unadvertised one — the shared `validate_request`
    /// only passes a named sampler that is in `descriptor().samplers`. GPU-free (lazy generator).
    #[test]
    fn validate_accepts_lightning_sampler() {
        let spec = sdxl_spec(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("sdxl", &spec)
            .unwrap();

        let lightning = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            sampler: Some("lightning".into()),
            steps: Some(5),
            guidance: Some(1.0),
            ..Default::default()
        };
        assert!(g.validate(&lightning).is_ok());

        // sc-7124: a curated ε/DDPM sampler is now advertised and accepted.
        let curated = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        assert!(g.validate(&curated).is_ok());

        // An unadvertised sampler is still rejected (not silently downgraded).
        let bogus = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("not_a_sampler".into()),
            ..Default::default()
        };
        assert!(g.validate(&bogus).is_err());

        // F-116: `lightning` + a NON-DEFAULT curated scheduler (`karras`) is rejected — lightning
        // ignores the scheduler axis, so honoring the selection is impossible and silently dropping it
        // would mislead.
        let lightning_sched = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_sched).is_err());

        // …but `lightning` with the DEFAULT `normal` scheduler PASSES — a worker that always populates
        // `scheduler:"normal"` alongside lightning must not hard-error, and `normal` resolves to
        // lightning's own trailing schedule anyway (nothing dropped).
        let lightning_normal = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("normal".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_normal).is_ok());

        // …as does `lightning` with no scheduler at all, or the unrecognized native-fallback `discrete`
        // alias — both likewise resolve to lightning's own trailing schedule.
        let lightning_none = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: None,
            ..Default::default()
        };
        assert!(g.validate(&lightning_none).is_ok());
        let lightning_discrete = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("lightning".into()),
            scheduler: Some("discrete".into()),
            ..Default::default()
        };
        assert!(g.validate(&lightning_discrete).is_ok());
    }

    /// sc-5165: `load` now ACCEPTS LoRA/LoKr adapters — it carries them for a load-time merge into the
    /// UNet weights at the first `generate`. `load` stays lazy (no file I/O), so a nonexistent adapter
    /// path still resolves here; an unresolvable / mis-formatted adapter errors only when `generate`
    /// first builds the UNet. (The merge math + format routing are covered in `adapters`'s tests.)
    #[test]
    fn load_accepts_lora_adapters() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let spec =
            sdxl_spec(WeightsSource::Dir("/snap".into())).with_adapters(vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]);
        let g = load(&spec).expect("load accepts adapters (lazy; merge defers to generate)");
        assert!(g.descriptor().capabilities.supports_lora);
        assert!(g.descriptor().capabilities.supports_lokr);
    }

    /// sc-14195, the **request** half of the seam: `validate` must ACCEPT `guidance = 1.0` (and the
    /// rest of the CFG-off range) rather than reject it with a 4xx. The story allowed either
    /// outcome — accept-and-run-CFG-off or reject — and this pins which one shipped, so a later
    /// "just validate it away" change has to break a test that says so out loud.
    ///
    /// The engine half (that accepting it actually renders, on the cond row) is
    /// `pipeline::tests::render_at_guidance_one_runs_cfg_off_without_batch_mismatch`; together they
    /// span request → engine. GPU-free (lazy generator, no weights touched).
    #[test]
    fn validate_accepts_cfg_off_guidance() {
        let spec = sdxl_spec(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("sdxl", &spec)
            .unwrap();

        // The story's exact value, plus the rest of the range that resolves to the same CFG-off
        // branch (`use_guide = guidance > 1.0`). None of these is an error.
        for guidance in [1.0f32, 0.5, 0.0, -1.0] {
            let req = GenerationRequest {
                prompt: "a rusty robot holding a lit candle".into(),
                guidance: Some(guidance),
                ..Default::default()
            };
            assert!(
                g.validate(&req).is_ok(),
                "guidance {guidance} must be accepted and run CFG-off, not rejected"
            );
        }

        // A negative prompt alongside CFG-off is accepted too (it is dropped, by the contract
        // documented on `Capabilities` above — not a rejection, and not a hard error).
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                negative_prompt: Some("blurry".into()),
                guidance: Some(1.0),
                ..Default::default()
            })
            .is_ok());

        // Mutation guard: a NON-finite guidance is still rejected (the F-053 check), so this test
        // cannot pass by `validate` having become a blanket accept-everything.
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                guidance: Some(f32::NAN),
                ..Default::default()
            })
            .is_err());
    }

    #[test]
    fn load_reports_missing_single_file_source() {
        let spec = sdxl_spec(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        // The message is the OS's, so it is platform-specific: POSIX says "No such file or
        // directory", Windows says "The system cannot find the file specified. (os error 2)".
        // Accept either phrasing — what is under test is that the missing file is *reported*, not
        // how libc spells it. (Drive-by: the POSIX-only assertion made this test fail on every
        // Windows run of `cargo test -p candle-gen-sdxl`.)
        assert!(
            err.contains("No such file")
                || err.contains("not found")
                || err.contains("cannot find the file"),
            "got: {err}"
        );
    }
}
