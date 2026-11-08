//! PuLID-FLUX end-to-end provider (sc-5492) — the candle (Windows/CUDA) twin of `mlx-gen-pulid`'s
//! `pulid_flux.rs`. Assembles the full face-identity path on top of the candle FLUX.1-dev backbone:
//!
//!   1. **Face analysis** (native, `candle-gen-face`): the reference face → `FaceAnalysis::analyze` →
//!      largest face's ArcFace embedding (512-d) + `face_features_image` (512² aligned, bg-whitened
//!      grayscale via BiSeNet). No Python/onnx.
//!   2. **EVA-CLIP** ([`crate::eva_clip`]): `face_features_image` → resize/normalize → `id_cond_vit`
//!      (768-d, L2-normalized) + 5 hidden states.
//!   3. **IDFormer** ([`crate::idformer`]): `id_cond = cat(arcface 512, id_cond_vit 768)` + hidden →
//!      `id_embedding` `[1,32,2048]`.
//!   4. **CA injection** ([`crate::ca`]): build a [`PulidCa`] bound to the id_embedding and run the FLUX
//!      flow-match denoise through `IpFlux::forward_injected` → AutoEncoder decode.
//!
//! The conditioning path (EVA tower + IDFormer + the 20 CA modules) runs in **f32** for identity
//! fidelity; the candle FLUX DiT image stream is bf16, so the CA residual is cast to the image dtype at
//! injection (the `r.to_dtype(img.dtype())` in `IpFlux::forward_injected`). FLUX.1-dev is the only PuLID
//! backbone (guidance-distilled, single distilled forward per step — real-CFG / uncond-id is a later
//! slice, matching the candle `supports_true_cfg: false` stance).
//!
//! Like the candle InstantID / IP-Adapter providers, [`PulidFlux`] is a plain struct the worker drives
//! **directly** (a bespoke reference stream), NOT a gen-core-registered generator.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::flux::sampling::State;
use rand::{rngs::StdRng, SeedableRng};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    Conditioning, ConditioningKind, GenerationMemory, GenerationRequest, IdentityWeights, Image,
    LoadSpec, MemoryProviderContract, MemoryRunContext, MemoryRunOutcome, ModelDescriptor,
    PidWeights, PreviewSink, Progress, RenderLoadBinding, RenderPlan, RenderRun, WeightsSource,
};
use candle_gen::preview::PreviewHook;
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
#[cfg(test)]
use candle_gen_flux::{flow_mu, BASE_SHIFT, MAX_SHIFT};
use candle_gen_flux::{DitImageInjector, FluxRefBackbone, FluxRefHeavy, Variant};
use candle_gen_pid::{PidDecoder, PidEngine};

use crate::ca::PulidCa;
use crate::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use crate::idformer::{IdFormer, IdFormerConfig};

/// FLUX.1-dev DiT block counts (the PuLID injection schedule is defined over these).
const NUM_DOUBLE_BLOCKS: usize = 19;
const NUM_SINGLE_BLOCKS: usize = 38;
/// FLUX runs at bf16; the conditioning path runs at f32 (identity fidelity).
const DTYPE: DType = DType::BF16;
const COND_DTYPE: DType = DType::F32;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
pub(crate) const LATENT_CHANNELS: usize = 16;
// FLUX dev's flow-match time-shift endpoints (`BASE_SHIFT`/`MAX_SHIFT`) and the `flow_mu` linear map
// are shared from `candle-gen-flux` (sc-11249 / F-140) — PuLID (always dev) reuses the exact
// parity-critical schedule constants rather than maintaining a third copy.

/// FLUX packs the /8 VAE latent 2×2, so both render dims must be multiples of 16 (the flux1 txt2img /
/// IP-Adapter / control size floor). Exposed as the pinned-engine stride SceneWorks ties each
/// advertised PuLID-FLUX image bucket to (sc-12612), mirroring `wan::config::SIZE_MULTIPLE_14B`.
/// `reject_below_floor` enforces exactly this value, so the const cannot drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// Reject a below-floor request loudly before any tensor work. Without it `get_schedule(0, …)` returns
/// `[NaN]` — zero sampler steps, so the pure seeded noise is decoded and returned as a "success",
/// burning GPU time for garbage. A fast typed error mirrors the sibling bespoke lanes (`reject_zero_steps`
/// in sdxl-IP / scail2 / instantid, sc-9016, F-032); this worker-driven PuLID path has no gen-core
/// capability floor upstream of it, and (like flux1-IP) previously had no size floor either
/// (sc-11182, F-102).
fn reject_below_floor(req: &PulidFluxRequest) -> Result<()> {
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "pulid_flux: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg(
            "pulid_flux: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

/// Poll cancellation at every boundary of the expensive identity preparation stack. A cancellation
/// that arrives while a synchronous face/EVA/IDFormer stage is running is observed before the next
/// stage or any generation setup begins.
fn identity_stage<T>(cancel: &CancelFlag, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    let output = work()?;
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    Ok(output)
}

/// Default PuLID `id_weight` (the reference-face strength; 0–3, upstream default 1.0).
pub const DEFAULT_ID_WEIGHT: f32 = 1.0;
/// Default dev guidance for the PuLID photoreal recipe.
pub const DEFAULT_GUIDANCE: f32 = 4.0;

/// Paths to the PuLID-FLUX checkpoints.
#[derive(Clone)]
pub struct PulidFluxPaths {
    /// The FLUX.1-dev backbone snapshot dir — auto-detected (sc-10103): either a dense black-forest-labs
    /// `FLUX.1-dev` snapshot (`flux1-dev.safetensors`, `ae.safetensors`, `text_encoder{,_2}/`,
    /// `tokenizer_2/`) OR a packed/dense `SceneWorks/flux1-dev-mlx` turnkey **tier subdir**
    /// (`…/q4`, `…/q8`, `…/bf16`: `transformer/` + `text_encoder{,_2}/` + `vae/` + `tokenizer_2/`). The
    /// [`FluxRefBackbone`] loader picks the right layout — the worker resolves the tier subdir.
    pub flux_base: PathBuf,
    /// `guozinan/PuLID` `pulid_flux_v0.9.1.safetensors` (holds both `pulid_encoder.*` = the IDFormer and
    /// `pulid_ca.*` = the 20 cross-attn modules).
    pub pulid_weights: PathBuf,
    /// The converted EVA02-CLIP-L-14-336 safetensors (the `convert_eva_clip.py` output; bare key names).
    pub eva_weights: PathBuf,
    /// The native face-stack dir (`scrfd_10g` / `arcface_iresnet100` / `bisenet_parsing`).
    pub face_dir: PathBuf,
    pub adapters: Vec<candle_gen::gen_core::AdapterSpec>,
}

/// One PuLID-FLUX generation request.
#[derive(Clone)]
pub struct PulidFluxRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale — embedded by the dev DiT.
    pub guidance: f32,
    /// PuLID id_weight (reference-face strength; `0.0` ⇒ the no-id ablation = plain FLUX).
    pub id_weight: f32,
    /// Curated unified-sampler selection (epic 7114, sc-7297). `None` (or `flow_match` / `euler`) keeps
    /// the native flow-match Euler default; a curated [`Solver`](candle_gen::gen_core::sampling::Solver)
    /// name routes the PuLID-injected flow denoise through that integrator (the candle PuLID runs its
    /// OWN flow loop, vs the mlx PuLID which delegates to the FLUX backbone — so the knob is threaded
    /// here directly through [`candle_gen::run_flow_sampler`]).
    pub sampler: Option<String>,
    /// Curated σ-schedule selection (epic 7114). `None` (or a native alias) ⇒ FLUX's verbatim
    /// time-shifted `get_schedule`; a curated scheduler name re-strides σ over the dev time-shift `mu`.
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](PulidFlux::with_pid), the final latent is decoded by the `flux` PiD student
    /// (4× SR → 2K/4K) instead of the native FLUX.1 VAE. PiD is a *generative* decoder, so face likeness
    /// may shift — the user judges per-generation. `false` (default) keeps the byte-exact VAE decode.
    pub use_pid: bool,
    /// Per-step latent-preview sink (epic 16948, sc-16956) - the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which PuLID cannot carry because it registers no descriptor at all (a `BESPOKE_UTILITY_CRATES`
    /// member the worker drives by name). Frames are the *image* latent alone; the identity embedding
    /// is injected inside the DiT forward and never joins it (see [`crate::preview`]). Default is
    /// inert, and an inert sink makes a seeded render byte-identical to one with no preview at all.
    pub preview: PreviewSink,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for PulidFluxRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: DEFAULT_GUIDANCE,
            id_weight: DEFAULT_ID_WEIGHT,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            seed: 0,
            use_pid: false,
            preview: PreviewSink::default(),
            cancel: CancelFlag::default(),
        }
    }
}

/// L2-normalize each row of `[B, D]` over the feature axis (the PuLID `id_cond_vit` normalization),
/// clamping the norm to a tiny epsilon so a degenerate zero-norm row yields a zero vector, not NaN.
fn l2_normalize_rows(x: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12f32, f32::INFINITY)?;
    x.broadcast_div(&norm)
}

/// The loaded PuLID-FLUX model: the tier-detected FLUX backbone (text encoders, DiT, VAE, tokenizers),
/// the EVA tower, the IDFormer, the kept PuLID checkpoint (for the per-generate [`PulidCa`]), and the
/// native face stack.
pub struct PulidFlux {
    device: Device,
    dtype: DType,
    /// The FLUX.1-dev backbone — CLIP + T5 + DiT + VAE + both tokenizers, tier-detected at load
    /// (sc-10103): a dense BFL snapshot or a packed/dense `SceneWorks/flux1-dev-mlx` turnkey tier. The
    /// PuLID CA identity injection drives its post-block [`DitImageInjector`]
    /// [`forward_injected`](FluxRefBackbone::forward_injected) seam. Reuses the base FLUX txt2img pipeline
    /// load path, so PuLID and `flux_dev` never drift on tokenization / tier detection / VAE decode.
    backbone: FluxRefBackbone,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint (f32) — kept to build a per-generate [`PulidCa`] from `pulid_ca.*`
    /// (`pulid_encoder.*` is already consumed by `idformer`).
    pulid: Weights,
    face: candle_gen_face::CandleFaceAnalysis,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// PuLID composes the FLUX.1-dev VAE, so it loads the `flux` student (same tag as the base FLUX provider).
    pid: Option<PidEngine>,
    /// Exact request-scoped contract/context selected for the bespoke route. Legacy callers that use
    /// `load`/`load_with_memory` leave these empty; the SceneWorks route uses
    /// `load_with_memory_context` and must then call `generate_with_memory_context`.
    memory_contract: Option<MemoryProviderContract>,
    admitted_context: Option<MemoryRunContext>,
    loaded_paths: PulidFluxPaths,
    loaded_spec: LoadSpec,
}

fn default_spec(paths: &PulidFluxPaths) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
        .with_adapters(paths.adapters.clone());
    spec.identity = Some(IdentityWeights {
        encoder: Some(WeightsSource::File(paths.pulid_weights.clone())),
        eva: Some(WeightsSource::File(paths.eva_weights.clone())),
        face_dir: Some(WeightsSource::Dir(paths.face_dir.clone())),
    });
    spec
}

fn validate_pulid_spec(
    paths: &PulidFluxPaths,
    spec: &LoadSpec,
) -> candle_gen::gen_core::Result<()> {
    let expected = default_spec(paths);
    let slots = |identity: Option<&IdentityWeights>| {
        identity.map(|identity| {
            (
                identity.encoder.clone(),
                identity.eva.clone(),
                identity.face_dir.clone(),
            )
        })
    };
    if spec.weights != expected.weights
        || slots(spec.identity.as_ref()) != slots(expected.identity.as_ref())
        || spec.adapters != paths.adapters
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || !spec.components.is_empty()
        || spec.text_encoder.is_some()
        || spec.quantize.is_some()
        || spec.precision != candle_gen::gen_core::Precision::Bf16
    {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "PuLID-FLUX load contract differs from resolved paths".into(),
        ));
    }
    Ok(())
}

fn plan_descriptor() -> ModelDescriptor {
    let mut descriptor = candle_gen_flux::descriptor_dev();
    candle_gen_flux::add_conditioned_native_aliases(&mut descriptor);
    // PuLID's legacy size gate is the FLUX pack stride, not the base txt2img
    // descriptor's 256..=2048 range.
    descriptor.capabilities.min_size = SIZE_MULTIPLE;
    descriptor.capabilities.max_size = u32::MAX;
    descriptor.capabilities.size_floor = candle_gen::gen_core::SizeFloor::RangeCheckedOnGridArea {
        multiple: SIZE_MULTIPLE,
        max_cells: u32::MAX,
    };
    descriptor.id = "pulid_flux";
    descriptor.capabilities.conditioning = vec![ConditioningKind::Reference];
    descriptor.capabilities.max_count = 1;
    descriptor
}

fn typed_pulid_request(
    req: &PulidFluxRequest,
    reference: &Image,
) -> candle_gen::gen_core::Result<GenerationRequest> {
    let steps = u32::try_from(req.steps).map_err(|_| {
        candle_gen::gen_core::Error::Unsupported("PuLID step count exceeds u32".into())
    })?;
    Ok(GenerationRequest {
        prompt: req.prompt.clone(),
        width: req.width,
        height: req.height,
        count: 1,
        steps: Some(steps),
        seed: Some(req.seed),
        guidance: Some(req.guidance),
        sampler: req.sampler.clone(),
        scheduler: req.scheduler.clone(),
        scheduler_shift: req.scheduler_shift,
        conditioning: vec![Conditioning::Reference {
            image: reference.clone(),
            strength: Some(req.id_weight),
        }],
        use_pid: req.use_pid,
        preview: req.preview.clone(),
        cancel: req.cancel.clone(),
        ..Default::default()
    })
}

/// Freeze the face source, identity stack, optional PiD decoder and trajectory before load.
pub fn compile_pulid_plan(
    paths: &PulidFluxPaths,
    spec: &LoadSpec,
    req: &PulidFluxRequest,
    reference: &Image,
) -> candle_gen::gen_core::Result<RenderPlan> {
    validate_pulid_spec(paths, spec)?;
    reject_below_floor(req)?;
    if req.prompt.trim().is_empty() {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "PuLID needs a prompt".into(),
        ));
    }
    if req.use_pid && spec.pid.is_none() {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "PuLID PiD decode needs a loaded overlay".into(),
        ));
    }
    if req.cancel.is_cancelled() {
        return Err(candle_gen::gen_core::Error::Canceled);
    }
    let typed = typed_pulid_request(req, reference)?;
    let descriptor = plan_descriptor();
    descriptor
        .capabilities
        .validate_request(descriptor.id, &typed)?;
    candle_gen_flux::compile_conditioned_plan(Variant::Dev, &descriptor, spec, &typed)
}

impl PulidFlux {
    /// Load the FLUX.1-dev backbone + the EVA tower + the IDFormer + the PuLID CA weights + the native
    /// face stack (with the BiSeNet parser) from the [`PulidFluxPaths`].
    pub fn load(paths: &PulidFluxPaths) -> Result<Self> {
        Self::load_with_memory(paths, GenerationMemory::default())
    }

    /// Load PuLID with the shared FLUX request memory plan. Identity modules remain resident while the
    /// base backbone may stage text→DiT/VAE, stream its two block stacks, and CPU-decode through the
    /// calibrated bounded-host path.
    pub fn load_with_memory(paths: &PulidFluxPaths, memory: GenerationMemory) -> Result<Self> {
        Self::load_internal(paths, memory, None, None, &default_spec(paths))
    }

    pub fn load_with_memory_spec(
        paths: &PulidFluxPaths,
        memory: GenerationMemory,
        spec: &LoadSpec,
    ) -> Result<Self> {
        Self::load_internal(paths, memory, None, None, spec)
    }

    /// Load under an exact shared-ladder admission context. The provider rebuilds and validates the
    /// contract from the actual resolved paths, derives `GenerationMemory` from the admitted
    /// selection, and retains the handshake for every subsequent bespoke request.
    pub fn load_with_memory_context(
        paths: &PulidFluxPaths,
        context: MemoryRunContext,
    ) -> Result<Self> {
        let contract = crate::memory_strategy::provider_contract(paths)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(paths, &contract, &context)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        Self::load_internal(
            paths,
            memory,
            Some(contract),
            Some(context),
            &default_spec(paths),
        )
    }

    pub fn load_with_memory_context_spec(
        paths: &PulidFluxPaths,
        context: MemoryRunContext,
        spec: &LoadSpec,
    ) -> Result<Self> {
        let contract = crate::memory_strategy::provider_contract(paths)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(paths, &contract, &context)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        Self::load_internal(paths, memory, Some(contract), Some(context), spec)
    }

    fn load_internal(
        paths: &PulidFluxPaths,
        memory: GenerationMemory,
        memory_contract: Option<MemoryProviderContract>,
        admitted_context: Option<MemoryRunContext>,
        spec: &LoadSpec,
    ) -> Result<Self> {
        validate_pulid_spec(paths, spec)?;
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;

        // FLUX.1-dev backbone — tier-detected (sc-10103): a dense BFL snapshot or a packed/dense
        // `SceneWorks/flux1-dev-mlx` turnkey tier subdir. Reuses the base FLUX txt2img pipeline's
        // detect-and-load, so PuLID consumes the SAME q4/q8/bf16 tiers `flux_dev` does. The PuLID CA
        // injection runs through the backbone's post-block `forward_injected` seam (on the BFL `IpFlux`
        // or the diffusers `PackedFluxDit`, whichever tier loaded).
        let backbone = FluxRefBackbone::load_with_memory(
            &paths.flux_base,
            Variant::Dev,
            &device,
            dtype,
            memory,
            paths.adapters.clone(),
        )?;

        // EVA-CLIP tower (f32 conditioning path).
        let eva_w = Weights::from_file(&paths.eva_weights, &device, COND_DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "pulid_flux: load EVA weights {:?}: {e}",
                paths.eva_weights
            ))
        })?;
        let eva = EvaVisionTransformer::from_weights(&eva_w, "", EvaConfig::default())?;

        // PuLID encoder (IDFormer) + CA weights (f32).
        let pulid = Weights::from_file(&paths.pulid_weights, &device, COND_DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "pulid_flux: load PuLID weights {:?}: {e}",
                paths.pulid_weights
            ))
        })?;
        let idformer = IdFormer::from_weights(&pulid, "pulid_encoder", IdFormerConfig::default())?;

        // Native face stack + BiSeNet parser (the `face_features_image` path).
        let face = candle_gen_face::load_with_parser_on(&paths.face_dir, &device)?;

        Ok(Self {
            device,
            dtype,
            backbone,
            eva,
            idformer,
            pulid,
            face,
            pid: None,
            memory_contract,
            admitted_context,
            loaded_paths: paths.clone(),
            loaded_spec: spec.clone(),
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry FLUX.1 provider; PuLID composes the FLUX.1-dev VAE, so it loads the `flux` student.
    /// A `use_pid = true` request then decodes through it (4× SR) instead of the native VAE; without it,
    /// `use_pid` errors loudly. Face likeness may shift under the generative decode — the user's per-gen
    /// call. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        if self.loaded_spec.pid.as_ref().is_some_and(|expected| {
            expected.checkpoint != pid.checkpoint || expected.gemma != pid.gemma
        }) {
            return Err(CandleError::Unsupported(
                "PuLID PiD decoder differs from the pre-load plan".into(),
            ));
        }
        self.pid = Some(PidEngine::from_spec(pid, "flux", &self.device)?);
        self.loaded_spec.pid = Some(pid.clone());
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &PulidFluxRequest) -> Result<Option<PidDecoder>> {
        // Route through the shared guarded seam (sc-11242 / F-091) so the SR decode is budgeted
        // (F-013 sc-9095) and spatially tiled (sc-10087). Clean-latent σ=0 decode, single image.
        candle_gen_pid::resolve_pid_decoder_for_fields(
            self.pid.as_ref(),
            req.use_pid,
            &req.prompt,
            1,
            req.width,
            req.height,
            &req.cancel,
            req.seed,
            "pulid",
            0.0,
        )
    }

    /// Reference face (RGB [`Image`]) → `id_embedding` `[1,32,2048]` (f32). Mirrors PuLID's
    /// `get_id_embedding` (the conditional side). Direct callers retain the historical non-cancelable
    /// API; generation uses the cancellation-aware internal path below.
    pub fn compute_id_embedding(&self, reference: &Image) -> Result<Tensor> {
        self.compute_id_embedding_with_cancel(reference, &CancelFlag::new())
    }

    /// Cancellation-aware identity preparation used by generation. Polls `cancel` between each
    /// synchronous face/EVA/IDFormer stage before any denoise setup can begin.
    fn compute_id_embedding_with_cancel(
        &self,
        reference: &Image,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        let inner = self.face.inner();
        let (h, w) = (reference.height as usize, reference.width as usize);
        // Detect-then-embed-the-largest (sc-11249 / F-138), matching the `candle-gen-face` /
        // InstantID pattern: `detect` returns detections largest-first (no ArcFace forward), and
        // `embed` runs the single `[1,3,112,112]` iresnet100 forward on the largest one only. This
        // avoids `analyze`'s N−1 wasted host norm-crops + N-row batched forward for a group-photo
        // reference; iresnet100 has no cross-batch ops, so the kept embedding is bit-identical to the
        // old `analyze(..).first()`.
        let dets = identity_stage(cancel, || inner.detect(&reference.pixels, h, w))?;
        let det = dets.first().ok_or_else(|| {
            CandleError::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        let face = identity_stage(cancel, || inner.embed(&reference.pixels, h, w, det))?;
        // ArcFace 512-d (raw, un-normalized) → [1, 512] f32.
        let dim = face.embedding.len();
        let arcface = identity_stage(cancel, || {
            Ok(Tensor::from_vec(
                face.embedding.clone(),
                (1, dim),
                &self.device,
            )?)
        })?;
        // face_features_image (512² NCHW) → EVA 336² transform → tower.
        let ffi = identity_stage(cancel, || {
            inner.face_features_image(&reference.pixels, h, w, &face)
        })?;
        let eva_in = identity_stage(cancel, || {
            transform::eva_transform(&ffi, self.eva.config().image_size)
        })?;
        let eva_out = identity_stage(cancel, || Ok(self.eva.forward(&eva_in)?))?;
        let id_cond_vit = identity_stage(cancel, || Ok(l2_normalize_rows(&eva_out.id_cond_vit)?))?; // [1,768]
        let id_cond = identity_stage(cancel, || Ok(Tensor::cat(&[&arcface, &id_cond_vit], 1)?))?; // [1,1280]
        identity_stage(cancel, || self.idformer.forward(&id_cond, &eva_out.hidden))
    }

    /// Reference-image identity T2I: condition the FLUX.1-dev generation on `reference`'s PuLID
    /// id_embedding at `req.id_weight` (a single distilled forward per step — no true-CFG).
    pub fn generate(
        &self,
        req: &PulidFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = compile_pulid_plan(&self.loaded_paths, &self.loaded_spec, req, reference)?;
        self.generate_planned(req, reference, &plan, on_progress)
    }

    pub fn generate_planned(
        &self,
        req: &PulidFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if self.admitted_context.is_some() {
            return Err(CandleError::Msg(
                "pulid_flux: admitted model requires generate_with_memory_context".into(),
            ));
        }
        self.check_plan(req, reference, plan)?;
        let run = RenderRun::new(plan, &req.cancel, None);
        self.generate_inner(req, reference, plan, &run, on_progress)
    }

    /// Execute one bespoke PuLID request under the exact admission handshake retained at load. The
    /// route and geometry are revalidated at the last responsible moment and the CUDA device is
    /// synchronized on both success and failure so request-scoped materializations cannot leak into
    /// the next job's budget snapshot.
    pub fn generate_with_memory_context(
        &self,
        context: &MemoryRunContext,
        req: &PulidFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = compile_pulid_plan(&self.loaded_paths, &self.loaded_spec, req, reference)?;
        self.generate_with_memory_context_planned(context, req, reference, &plan, on_progress)
    }

    pub fn generate_with_memory_context_planned(
        &self,
        context: &MemoryRunContext,
        req: &PulidFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let admitted = self.admitted_context.as_ref().ok_or_else(|| {
            CandleError::Msg("pulid_flux: model was not loaded with a memory context".into())
        })?;
        let contract = self.memory_contract.as_ref().ok_or_else(|| {
            CandleError::Msg("pulid_flux: admitted model lost its memory contract".into())
        })?;
        if admitted != context {
            return Err(CandleError::Msg(
                "pulid_flux: request memory context changed after provider load".into(),
            ));
        }
        validate_request_geometry(
            &context.geometry,
            context.has_reference,
            context.use_pid,
            req,
        )?;
        crate::memory_strategy::validate_context_from_loaded(contract, context)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        self.check_plan(req, reference, plan)?;
        let mut scope = candle_gen_flux::memory_strategy::reference_backbone_request_scope(
            "pulid_flux",
            self.device.clone(),
            contract,
            context,
        );
        let result = {
            let run = RenderRun::new(plan, &req.cancel, Some(scope.as_mut()));
            self.generate_inner(req, reference, plan, &run, on_progress)
        };
        let outcome = match &result {
            Ok(_) => MemoryRunOutcome::Complete,
            Err(CandleError::Canceled) => MemoryRunOutcome::Canceled,
            Err(error) => MemoryRunOutcome::Error {
                message: error.to_string(),
            },
        };
        let synchronized = scope
            .finish(outcome)
            .map_err(|error| CandleError::Msg(error.to_string()));
        match (result, synchronized) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(image), Ok(())) => Ok(image),
        }
    }

    fn check_plan(
        &self,
        req: &PulidFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
    ) -> Result<()> {
        if req.use_pid && self.pid.is_none() {
            return Err(CandleError::Unsupported(
                "PuLID PiD decode was planned but the decoder was not attached".into(),
            ));
        }
        let typed = typed_pulid_request(req, reference)?;
        plan.check_request(&plan_descriptor(), &typed)?;
        plan.check_load(&RenderLoadBinding::from_spec(&self.loaded_spec))?;
        Ok(())
    }

    fn generate_inner(
        &self,
        req: &PulidFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        run: &RenderRun<'_, '_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        reject_below_floor(req)?;
        self.backbone
            .validate_native_vae_request(req.use_pid, &req.cancel)?;

        let (pulid_ca, t5_emb, clip_emb) = run.encode(|| {
            let id_embedding = self.compute_id_embedding_with_cancel(reference, &req.cancel)?;
            let pulid_ca = identity_stage(&req.cancel, || {
                PulidCa::from_weights(
                    &self.pulid,
                    "pulid_ca",
                    id_embedding,
                    req.id_weight as f64,
                    NUM_DOUBLE_BLOCKS,
                    NUM_SINGLE_BLOCKS,
                )
            })?;
            let (t5_emb, clip_emb) = self.backbone.encode_text_with_memory_and_progress(
                &req.prompt,
                &req.cancel,
                on_progress,
            )?;
            Ok::<_, CandleError>((pulid_ca, t5_emb, clip_emb))
        })?;
        let heavy = self.backbone.load_heavy(&req.cancel)?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let pid_decoder = self.pid_decoder_for(req)?;
        let images = run.images_with_denoise_phases(
            on_progress,
            |seed| {
                let n = LATENT_CHANNELS * lat_h * lat_w;
                let mut rng = StdRng::seed_from_u64(seed);
                let noise = candle_gen::seeded_normal_vec(&mut rng, n);
                Ok::<_, CandleError>(
                    Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
                        .to_device(&self.device)?
                        .to_dtype(self.dtype)?,
                )
            },
            |seed, noise, on_progress| {
                run.begin_denoise_phase(0)?;
                let state = State::new(&t5_emb, &clip_emb, &noise)?;
                if state.img.dim(1)? != plan.packed_tokens() {
                    return Err(CandleError::Msg(
                        "PuLID packed geometry changed after planning".into(),
                    ));
                }
                let preview = crate::preview::hook(&req.preview, req.width, req.height);
                self.denoise(
                    &state,
                    plan,
                    run,
                    &pulid_ca,
                    heavy.as_ref(),
                    seed,
                    &preview,
                    &req.cancel,
                    on_progress,
                )
            },
            |_, latents, on_progress| {
                on_progress(Progress::Decoding);
                self.backbone.decode_with_memory(
                    heavy.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                    pid_decoder.as_ref(),
                    &req.cancel,
                )
            },
        )?;
        images
            .into_iter()
            .next()
            .ok_or_else(|| CandleError::Msg("PuLID plan produced no image".into()))
    }

    /// The flow-match denoise with the PuLID CA injector, routed through the unified curated
    /// sampler/scheduler driver (epic 7114, sc-7297). The `scheduler` axis re-strides FLUX's native
    /// `get_schedule(..)` over the dev time-shift `mu`; the `sampler` axis picks the integrator. The
    /// forked [`IpFlux::forward_injected`] (`Some(injector)`) is the model forward, and the PuLID CA
    /// identity injection stays INSIDE the `predict` closure so a multi-eval solver (heun / dpmpp) re-runs
    /// the whole step. The DEFAULT (`sampler`/`scheduler` unset ⇒ euler over the native schedule) is the
    /// N1 path for the legacy inline flow-match Euler loop `img += pred·(σ_{i+1} − σ_i)`. FLUX feeds the
    /// raw timestep (`Sigma` convention: `t == σ`); guidance is a per-batch tensor the dev DiT embeds.
    /// Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        state: &State,
        render_plan: &RenderPlan,
        run: &RenderRun<'_, '_>,
        injector: &PulidCa,
        heavy: Option<&FluxRefHeavy>,
        seed: u64,
        preview: &PreviewHook<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(render_plan.guidance(), b_sz, &self.device)?;
        candle_gen::run_flow_sampler_spans(
            Some(render_plan.sampler()),
            TimestepConvention::Sigma,
            render_plan.denoise_sigmas(),
            &render_plan.denoise_spans(),
            Some(run),
            true,
            state.img.clone(),
            seed,
            cancel,
            on_progress,
            Some(preview),
            |img, t| -> Result<Tensor> {
                // The backbone dispatches to the loaded tier's DiT (BFL `IpFlux` or packed
                // `PackedFluxDit`) `forward_injected`; the PuLID CA identity injection lives inside this
                // closure so a multi-eval solver re-runs the whole step.
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                self.backbone.forward_injected_with_memory(
                    heavy,
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    Some(injector as &dyn DitImageInjector),
                    cancel,
                )
            },
        )
    }
}

fn validate_request_geometry(
    geometry: &candle_gen::gen_core::MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    req: &PulidFluxRequest,
) -> Result<()> {
    if geometry.width != req.width
        || geometry.height != req.height
        || geometry.batch != 1
        || geometry.frames != 1
        || geometry.reference_count != 1
        || has_reference != (geometry.reference_count > 0)
        || use_pid != req.use_pid
    {
        return Err(CandleError::Msg(
            "pulid_flux: request route or geometry changed after memory admission".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct PhaseTrace(Vec<(bool, candle_gen::gen_core::MemoryPhase)>);

    impl candle_gen::gen_core::MemoryRequestScope for PhaseTrace {
        fn configure_request(
            &mut self,
            _: &mut GenerationRequest,
        ) -> candle_gen::gen_core::Result<()> {
            unreachable!()
        }
        fn enter_phase(
            &mut self,
            _: candle_gen::gen_core::MemoryPhase,
        ) -> candle_gen::gen_core::Result<()> {
            unreachable!()
        }
        fn enter_render_phase(
            &mut self,
            phase: candle_gen::gen_core::MemoryPhase,
            identity: &candle_gen::gen_core::render::RenderPhaseIdentity,
        ) -> candle_gen::gen_core::Result<()> {
            assert_eq!((identity.pass, identity.phase), (0, 0));
            assert_eq!(
                (
                    identity.geometry.batch,
                    identity.geometry.frames,
                    identity.geometry.reference_count
                ),
                (1, 1, 1)
            );
            self.0.push((true, phase));
            Ok(())
        }
        fn leave_phase(
            &mut self,
            phase: candle_gen::gen_core::MemoryPhase,
        ) -> candle_gen::gen_core::Result<()> {
            self.0.push((false, phase));
            Ok(())
        }
        fn configure_decode(
            &mut self,
            _: u32,
            _: u32,
            _: candle_gen::gen_core::MemoryGeometry,
        ) -> candle_gen::gen_core::Result<()> {
            unreachable!()
        }
        fn configure_attention(&mut self, _: u32) -> candle_gen::gen_core::Result<()> {
            unreachable!()
        }
        fn materialize_transformer_window(
            &mut self,
            _: u32,
            _: u32,
        ) -> candle_gen::gen_core::Result<()> {
            unreachable!()
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> candle_gen::gen_core::Result<()> {
            Ok(())
        }
    }

    #[test]
    #[allow(clippy::drop_non_drop)]
    fn pulid_plan_binds_face_identity_and_pid_before_weights() {
        let paths = PulidFluxPaths {
            flux_base: "/missing/flux-dev".into(),
            pulid_weights: "/missing/pulid".into(),
            eva_weights: "/missing/eva".into(),
            face_dir: "/missing/face".into(),
            adapters: Vec::new(),
        };
        let spec = default_spec(&paths);
        let req = PulidFluxRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: 17,
            ..Default::default()
        };
        let mut reference = Image {
            width: 16,
            height: 16,
            pixels: vec![5; 16 * 16 * 3],
        };
        let plan = compile_pulid_plan(&paths, &spec, &req, &reference).unwrap();
        let large = PulidFluxRequest {
            width: 2304,
            height: 1024,
            ..req.clone()
        };
        compile_pulid_plan(&paths, &spec, &large, &reference).unwrap();
        assert_eq!(plan.base_seed(), 17);
        for name in ["flow_match", "flow_match_euler"] {
            let mut alias = req.clone();
            alias.sampler = Some(name.into());
            assert_eq!(
                compile_pulid_plan(&paths, &spec, &alias, &reference)
                    .unwrap()
                    .sampler(),
                name
            );
        }
        for name in ["native", "flow_match", "flow_match_euler"] {
            let mut alias = req.clone();
            alias.scheduler = Some(name.into());
            assert_eq!(
                compile_pulid_plan(&paths, &spec, &alias, &reference)
                    .unwrap()
                    .denoise_sigmas(),
                plan.denoise_sigmas()
            );
        }
        for (sampler, scheduler) in [(Some("unknown"), None), (None, Some("unknown"))] {
            let mut invalid = req.clone();
            invalid.sampler = sampler.map(str::to_owned);
            invalid.scheduler = scheduler.map(str::to_owned);
            assert!(compile_pulid_plan(&paths, &spec, &invalid, &reference).is_err());
        }
        let mut trace = PhaseTrace::default();
        let run = RenderRun::new(&plan, &req.cancel, Some(&mut trace));
        run.encode(|| Ok::<_, candle_gen::gen_core::Error>(()))
            .unwrap();
        run.images_with_denoise_phases(
            &mut |_| {},
            |_| Ok::<_, candle_gen::gen_core::Error>(()),
            |_, _, _| {
                run.begin_denoise_phase(0)?;
                run.end_denoise_phase(0)?;
                Ok::<_, candle_gen::gen_core::Error>(())
            },
            |_, _, _| Ok::<_, candle_gen::gen_core::Error>(()),
        )
        .unwrap();
        drop(run);
        use candle_gen::gen_core::MemoryPhase;
        assert_eq!(
            trace.0,
            vec![
                (true, MemoryPhase::Conditioning),
                (false, MemoryPhase::Conditioning),
                (true, MemoryPhase::Denoise),
                (false, MemoryPhase::Denoise),
                (true, MemoryPhase::Decode),
                (false, MemoryPhase::Decode),
            ]
        );
        reference.pixels[0] ^= 1;
        assert!(plan
            .check_request(
                &plan_descriptor(),
                &typed_pulid_request(&req, &reference).unwrap()
            )
            .is_err());
        let mut changed = spec.clone();
        changed.identity.as_mut().unwrap().eva = Some(WeightsSource::File("/other/eva".into()));
        assert!(plan
            .check_load(&RenderLoadBinding::from_spec(&changed))
            .is_err());
        let pid_req = PulidFluxRequest {
            use_pid: true,
            ..req.clone()
        };
        assert!(compile_pulid_plan(&paths, &spec, &pid_req, &reference).is_err());
        let mut missing_overlay = spec.clone();
        missing_overlay.identity = None;
        assert!(compile_pulid_plan(&paths, &missing_overlay, &req, &reference).is_err());
        let mut invalid = req.clone();
        invalid.steps = 0;
        assert!(compile_pulid_plan(&paths, &spec, &invalid, &reference).is_err());
        invalid.steps = req.steps;
        invalid.width = 513;
        assert!(compile_pulid_plan(&paths, &spec, &invalid, &reference).is_err());
    }

    /// `steps == 0` and a non-multiple-of-16 size are fast typed errors (never a decoded pure-noise
    /// "success"); the defaults pass (sc-11182, F-102).
    #[test]
    fn reject_below_floor_floors_steps_and_size() {
        let base = PulidFluxRequest::default();

        let zero_steps = PulidFluxRequest {
            steps: 0,
            ..base.clone()
        };
        let err = reject_below_floor(&zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");

        let bad_size = PulidFluxRequest {
            height: 1000, // not a multiple of 16
            ..base.clone()
        };
        let err = reject_below_floor(&bad_size).unwrap_err();
        assert!(err.to_string().contains("multiples of 16"), "{err}");

        assert!(reject_below_floor(&base).is_ok());

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised PuLID-FLUX
        // bucket to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE
        // scale) but not SIZE_MULTIPLE (16) is still rejected with the stride error, and on-stride passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = reject_below_floor(&PulidFluxRequest {
            width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
            ..base.clone()
        })
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(reject_below_floor(&PulidFluxRequest {
            width: 1024, // 64×16 — on-stride
            ..base.clone()
        })
        .is_ok());
    }

    /// The request defaults match the PuLID-FLUX dev knobs (1024², 25 steps, guidance 4.0, id 1.0).
    #[test]
    fn request_defaults() {
        let r = PulidFluxRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE);
        assert_eq!(r.id_weight, DEFAULT_ID_WEIGHT);
        assert!(!r.cancel.is_cancelled());
    }

    /// F-108: a request canceled before identity preparation starts returns the shared cancellation
    /// error without invoking the first identity stage.
    #[test]
    fn identity_stage_pre_cancelled_skips_work() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let mut calls = 0;
        let result = identity_stage(&cancel, || {
            calls += 1;
            Ok(())
        });

        assert!(matches!(result, Err(CandleError::Canceled)));
        assert_eq!(calls, 0);
    }

    /// F-108: cancellation observed after an identity operation prevents every later identity or
    /// generation stage from starting, and preserves the shared `CandleError::Canceled` result.
    #[test]
    fn identity_stage_cancellation_checkpoint_stops_before_next_stage() {
        let cancel = CancelFlag::new();
        let mut first_calls = 0;
        let mut next_calls = 0;
        let result: Result<()> = (|| {
            identity_stage(&cancel, || {
                first_calls += 1;
                cancel.cancel();
                Ok(())
            })?;
            identity_stage(&cancel, || {
                next_calls += 1;
                Ok(())
            })
        })();

        assert!(matches!(result, Err(CandleError::Canceled)));
        assert_eq!(first_calls, 1);
        assert_eq!(
            next_calls, 0,
            "generation setup must not follow a canceled identity stage"
        );
    }

    #[test]
    fn admitted_generation_geometry_rejects_zero_or_multi_image_batches() {
        let req = PulidFluxRequest::default();
        let mut geometry = candle_gen::gen_core::MemoryGeometry {
            width: req.width,
            height: req.height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        };

        assert!(validate_request_geometry(&geometry, true, false, &req).is_ok());
        for batch in [0, 2] {
            geometry.batch = batch;
            let error = validate_request_geometry(&geometry, true, false, &req).unwrap_err();
            assert!(error.to_string().contains("geometry changed"), "{error}");
        }
    }

    /// `l2_normalize_rows` returns unit-norm rows (and the FLUX block counts the schedule is built over
    /// are the canonical 19 / 38).
    #[test]
    fn l2_normalize_and_block_counts() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4.0, 0.0, 0.0], (2, 2), &dev).unwrap();
        let n = l2_normalize_rows(&x).unwrap();
        let rows = n.to_vec2::<f32>().unwrap();
        assert!((rows[0][0] - 0.6).abs() < 1e-5 && (rows[0][1] - 0.8).abs() < 1e-5);
        // Second row is all-zero → stays zero (epsilon-clamped, no NaN).
        assert!(rows[1].iter().all(|&v| v == 0.0));
        assert_eq!((NUM_DOUBLE_BLOCKS, NUM_SINGLE_BLOCKS), (19, 38));
    }

    /// The shared `candle-gen-flux` `flow_mu`/shift constants (sc-11249 / F-140) reproduce the deleted
    /// PuLID-local copy bit-for-bit: `BASE_SHIFT`/`MAX_SHIFT` are the dev endpoints and
    /// `flow_mu(Variant::Dev, ..)` equals the old inlined linear map at every seq_len.
    #[test]
    fn shared_flow_mu_matches_deleted_pulid_copy() {
        assert_eq!((BASE_SHIFT, MAX_SHIFT), (0.5, 1.15));
        // The exact formula the PuLID copy used before the dedup.
        let old = |seq_len: usize| -> f32 {
            let m = (MAX_SHIFT - BASE_SHIFT) / (4096.0 - 256.0);
            let b = BASE_SHIFT - m * 256.0;
            (m * seq_len as f64 + b) as f32
        };
        for &seq_len in &[256usize, 512, 1024, 4096, 4224] {
            assert_eq!(
                flow_mu(Variant::Dev, seq_len),
                old(seq_len),
                "seq_len={seq_len}"
            );
        }
        // Endpoints land on the shift constants (m·256+b = BASE_SHIFT, m·4096+b = MAX_SHIFT).
        assert!((flow_mu(Variant::Dev, 256) as f64 - BASE_SHIFT).abs() < 1e-6);
        assert!((flow_mu(Variant::Dev, 4096) as f64 - MAX_SHIFT).abs() < 1e-6);
    }
}
