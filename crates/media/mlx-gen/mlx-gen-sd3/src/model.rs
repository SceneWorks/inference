//! `Sd3Large` — the SD3.5-Large / Large-Turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and explicit registration constants
//! (E5 sc-7864 = Large; E6 sc-7865 = Large-Turbo).
//!
//! [`load`] / [`load_turbo`] assemble the full model from a `stabilityai/stable-diffusion-3.5-large`
//! (resp. `-large-turbo`) snapshot directory (see [`crate::loader`]) — CLIP BPE + T5 tokenizers, the
//! triple text encoder, the MMDiT transformer, the 16-ch VAE — and [`Sd3Large::generate`] runs the
//! complete prompt→image pipeline: tokenize → triple-TE conditioning → seeded noise → flow-match Euler
//! denoise over the MMDiT → VAE decode → RGB8 (see [`crate::pipeline`]).
//!
//! ## Large vs Large-Turbo
//!
//! Both variants share **one MMDiT backbone arch** ([`Sd3Arch::large`](crate::config::Sd3Arch::large))
//! and one snapshot layout — the Turbo checkpoint is ADD-distilled to a few-step, guidance-baked
//! schedule, so it differs ONLY in the *sampling recipe*, not the tensor layout:
//!
//! * **Large** — true-CFG: default **28 steps**, **guidance 3.5**, negative prompt supported. Each
//!   denoise step runs the MMDiT TWICE (cond + uncond).
//! * **Large-Turbo** — distilled: default **4 steps**, **guidance 1.0 → CFG off**, no negative prompt.
//!   Guidance is baked into the distilled weights, so each step runs the MMDiT ONCE (cond only) — both
//!   faster and required (a CFG forward on a distilled model is wrong). The flow-match shift is the
//!   same `3.0` (verified against the Turbo `scheduler_config.json`).
//!
//! The variant-aware [`pipeline::denoise_cfg`] already skips the uncond forward when guidance is `1.0`
//! (and `uncond` is `None`), so the Turbo path reuses E5's pipeline unchanged — only the per-variant
//! step/guidance defaults + descriptor differ.
//!
//! Registered engine ids: **`sd3_5_large`** + **`sd3_5_large_turbo`** + **`sd3_5_medium`** (the
//! SceneWorks worker's `payload.model`). Medium (M3, sc-7869) reuses ALL of this scaffolding — the
//! same [`Sd3Large`] generator struct, loader, and pipeline — but is driven by the MMDiT-X arch
//! ([`Sd3Variant::Medium`] →
//! [`Sd3Arch::medium`](crate::config::Sd3Arch::medium): 24 blocks,
//! hidden 1536, `pos_embed_max_size` 384, dual-attention blocks `0..=12`) and the Medium
//! true-CFG recipe (40 steps / guidance 5.0). The struct keeps the historical `Sd3Large` name; it is
//! variant-parameterized and serves all three variants.

use std::path::Path;

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Conditioning, Error, FlowMatchEuler, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant,
    Residency, Result, WeightsSource,
};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;
use mlx_rs::Array;

use crate::config::Sd3Variant;
use crate::loader;
use crate::pipeline;
use crate::text::{Sd3Conditioning, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;

/// Registry id for SD3.5-Large (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = crate::config::SD3_5_LARGE_ID;
/// Registry id for SD3.5-Large-Turbo (the distilled few-step / CFG-off variant on the same backbone).
pub const TURBO_MODEL_ID: &str = crate::config::SD3_5_LARGE_TURBO_ID;
/// Registry id for SD3.5-Medium (the MMDiT-X variant — 24 blocks, hidden 1536, dual-attention in the
/// first 13 — true-CFG, on the same triple-TE + 16-ch VAE + flow-match-Euler scaffolding).
pub const MEDIUM_MODEL_ID: &str = crate::config::SD3_5_MEDIUM_ID;

/// SD3.5-Large's identity + capabilities — constructible without loading weights (registry
/// introspection). The full capability surface lives on [`Sd3Variant::Large`].
pub fn descriptor() -> ModelDescriptor {
    Sd3Variant::Large.descriptor()
}

/// SD3.5-Large-Turbo's identity + capabilities — the distilled few-step variant (no CFG / negative
/// prompt). The capability surface lives on [`Sd3Variant::LargeTurbo`].
pub fn turbo_descriptor() -> ModelDescriptor {
    Sd3Variant::LargeTurbo.descriptor()
}

/// SD3.5-Medium's identity + capabilities (M3, sc-7869) — the MMDiT-X variant: true-CFG (negative
/// prompt + guidance), default 40 steps / guidance 5.0, on the same triple-TE + 16-ch VAE +
/// flow-match-Euler (shift 3.0) pipeline. The capability surface lives on [`Sd3Variant::Medium`];
/// its `max_size` (1440) is the higher-res ceiling Medium's `pos_embed_max_size` (384) can span while
/// staying inside the Mac activation budget (see [`load_medium`]).
pub fn medium_descriptor() -> ModelDescriptor {
    Sd3Variant::Medium.descriptor()
}

/// A loaded SD3.5-Large / Large-Turbo generator: the tokenizers + three text encoders + MMDiT + VAE
/// assembled from a snapshot directory, plus the cached descriptor and the [`Sd3Variant`] that
/// selects the sampling recipe (step/guidance defaults, CFG on/off).
pub struct Sd3Large {
    variant: Sd3Variant,
    descriptor: ModelDescriptor,
    clip_tokenizer: ClipBpeTokenizer,
    /// Per-encoder CLIP pad ids (sc-9581): CLIP-L pads with eos (49407), bigG with `!` (0).
    clip_pad: crate::loader::Sd3ClipPad,
    t5_tokenizer: TextTokenizer,
    /// Component-residency strategy (epic 10834; the shared seam sc-11125), selected from
    /// [`LoadSpec::offload_policy`] at [`load_variant`]. `Resident` (default) holds the triple text
    /// encoder (CLIP-L + CLIP-G + T5-XXL) + MMDiT + VAE warm for the whole job and across jobs;
    /// `Sequential` holds only the per-phase loader closures and re-loads each per generation in phase
    /// order (encode → **drop all three encoders** → denoise/decode), bounding peak unified memory to
    /// `max(triple-TE, MMDiT+VAE)` instead of the sum — the biggest TE-drop win in the group (T5-XXL is
    /// large). The [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel
    /// checks, and the error-safe cache flush once for all providers. The tokenizers stay always-warm
    /// (cheap) on the struct above.
    residency: Residency<Sd3TextEncoders, Sd3Heavy>,
    memory_strategy: Option<mlx_gen::gen_core::MemoryProviderContract>,
    loaded_spec: LoadSpec,
}

/// The heavy render-phase components (everything but the triple text encoder): the MMDiT transformer
/// and the 16-ch VAE. Owned by the `Resident` components (held for the whole job) or by a `Sequential`
/// generate (loaded after the encoders are dropped, freed when the job ends). SD3 has no PiD/control
/// overlay, so the heavy bundle is just the transformer + VAE.
pub(crate) struct Sd3Heavy {
    transformer: Sd3Transformer,
    vae: Vae,
}

/// Seed-independent reference work for one SD3 request. The production implementation owns both
/// the VAE encode and the eager eval, while tests inject a tensor-free counter through this exact
/// boundary. [`Sd3Variant::map_seeded_outputs`] calls `prepare` before entering its output loop.
trait Sd3ReferencePreparer<R, P, E> {
    fn prepare(&mut self, route: Sd3Variant, reference: &R) -> std::result::Result<P, E>;
}

/// Production reference preparer shared by Large, Turbo, and Medium. Keeping the only reference
/// encode/eval pair in this adapter makes its placement independently mutation-checkable without
/// loading MLX tensors or weights.
struct MlxReferencePreparer<'a> {
    vae: &'a Vae,
    width: u32,
    height: u32,
}

impl Sd3ReferencePreparer<Image, Array, Error> for MlxReferencePreparer<'_> {
    fn prepare(&mut self, _route: Sd3Variant, init: &Image) -> Result<Array> {
        let clean = pipeline::encode_reference(self.vae, init, self.width, self.height)?;
        // MLX is lazy: force the clean latent while the one reference-encode graph is current so
        // every seed reuses data, not a deferred VAE graph.
        mlx_rs::transforms::eval([&clean])?;
        Ok(clean)
    }
}

impl Sd3Variant {
    /// Tensor-free request scheduler shared by the production Large, Turbo, and Medium generator
    /// path. `preparer` owns the one seed-independent reference encode/materialization; `output`
    /// owns each seed-specific noise, denoise, and decode. Keeping this seam on the route itself
    /// makes the production call injectable without constructing MLX arrays in its
    /// mutation-sensitive tests.
    fn map_seeded_outputs<R, P, O, E, A>(
        self,
        reference: Option<&R>,
        base_seed: u64,
        count: u32,
        preparer: &mut A,
        mut output: impl FnMut(Self, u64, Option<&P>) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E>
    where
        A: Sd3ReferencePreparer<R, P, E>,
    {
        mlx_gen::gen_core::map_sd3_seeded_outputs(
            reference,
            base_seed,
            count,
            |reference| preparer.prepare(self, reference),
            |seed, prepared| output(self, seed, prepared),
        )
    }
}

/// Construct a SD3.5-**Large** generator from a [`LoadSpec`]. See [`load_variant`] for the shared body.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::Large)
}

/// Construct a SD3.5-**Large-Turbo** generator from a [`LoadSpec`]. Identical backbone/load path to
/// [`load`] (same `Sd3Arch::large` arch + snapshot layout); only the sampling recipe (4 steps,
/// guidance-baked CFG-off) and the advertised capabilities differ. See [`load_variant`].
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::LargeTurbo)
}

/// Construct a SD3.5-**Medium** generator from a [`LoadSpec`] (M3, sc-7869). Same load path as
/// [`load`] but driven by the Medium **MMDiT-X** arch
/// ([`Sd3Arch::medium`](crate::config::Sd3Arch::medium): 24 blocks, hidden 1536,
/// `pos_embed_max_size` 384, dual-attention in blocks `0..=12`) and the
/// `stabilityai/stable-diffusion-3.5-medium` snapshot. The triple-TE, the 16-ch VAE, the snapshot
/// layout, and the flow-match-Euler (shift 3.0 — verified against Medium's `scheduler_config.json`)
/// pipeline are all shared with Large; only the transformer arch + the true-CFG sampling recipe
/// (40 steps / guidance 5.0) differ. See [`load_variant`].
pub fn load_medium(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::Medium)
}

/// Construct a [`Sd3Large`] for `variant` from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at the matching
/// `stabilityai/stable-diffusion-3.5-{large,large-turbo}` snapshot (the diffusers multi-component
/// tree; both variants share the same layout). `spec.quantize` (Q4/Q8) quantizes the WHOLE model —
/// transformer + the three text encoders (group_size 64) — after the dense load, matching the fork's
/// `nn.quantize` over every quantizable Linear so a Q4/Q8 consumer gets the full memory saving. The
/// VAE stays dense (its decode quality dominates the final image; matches the other DiT families'
/// quantize-the-heavy-parts convention). An fp32 precision override is rejected (the validated dense
/// path is bf16/f32 internal).
pub fn load_variant(spec: &LoadSpec, variant: Sd3Variant) -> Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (the CLIP encoders run f32 and the T5 \
             promotes internally; drop the precision override)"
        )));
    }
    // Resolve the snapshot dir up front — a fail-fast for BOTH residencies (Sequential defers the
    // heavy component build to each generate, but a single-file source is still wrong here).
    let root = resolve_root(spec, id)?;
    // The tokenizers stay always-warm (cheap) on the struct; the heavy component dispatch is the
    // shared [`build_residency`] seam. F-181: a `Sequential` + `quantize` load re-quantizes the whole
    // model on EVERY generate (SD3 quantizes dense at load — there is no packed-detection path), so warn
    // for that combination (the dense transient also shrinks the memory win).
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(id, q.bits());
        }
    }
    let residency = build_residency(spec, variant)?;
    let memory_strategy = Some(
        crate::memory_strategy::contract_for(id, spec)
            .map_err(|error| Error::Msg(error.to_string()))?,
    );
    Ok(Box::new(Sd3Large {
        variant,
        descriptor: variant.descriptor(),
        clip_tokenizer: loader::load_clip_tokenizer(root)?,
        clip_pad: loader::load_clip_pad_ids(root)?,
        t5_tokenizer: loader::load_t5_tokenizer(root)?,
        residency,
        memory_strategy,
        loaded_spec: spec.clone(),
    }))
}

/// The policy→[`Residency`] dispatch every SD3 variant shares (sc-11126, F-180), routed through the
/// single [`Residency::from_policy`] seam so no variant re-derives the `match offload_policy`.
/// `Resident` eager-loads the triple text encoder + heavy bundle now (byte-identical to the pre-seam
/// composition — the same per-phase loaders over independent weight files, deterministic RNG-free
/// quant/adapter merge); `Sequential` captures the two loader closures and loads nothing now, deferring
/// each to [`Residency::run`]. SD3 has no PiD overlay, so the heavy loader's `use_pid` flag is ignored.
/// The deferral is weight-free-testable: under `Sequential` this touches no component weights, so a
/// dispatch that ignored `offload_policy` would eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    variant: Sd3Variant,
) -> Result<Residency<Sd3TextEncoders, Sd3Heavy>> {
    let id = variant.id();
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoders_component(resolve_root(&spec_text, id)?, spec_text.quantize),
        move |_use_pid| load_heavy(&spec_heavy, resolve_root(&spec_heavy, id)?, variant),
    )
}

/// Resolve the snapshot directory from the load spec, rejecting a single-file source (SD3 needs the
/// diffusers multi-component tree). Shared by [`load_variant`] and the `Sequential` per-phase loaders.
fn resolve_root<'a>(spec: &'a LoadSpec, id: &str) -> Result<&'a Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{id} expects a snapshot directory (transformer/ text_encoder{{,_2,_3}}/ \
             tokenizer{{,_2,_3}}/ vae/), not a single .safetensors file"
        ))),
    }
}

/// Load the triple text encoder (CLIP-L + CLIP-G + T5-XXL) — the phase-A component dropped first under
/// `Sequential`. Optional whole-encoder Q4/Q8 (group_size 64) matches the pre-seam load. Factored out
/// so the `Resident` and `Sequential` paths build byte-identical encoders (the encoders load from their
/// own weight files, and quantization is deterministic).
fn load_text_encoders_component(root: &Path, quant: Option<Quant>) -> Result<Sd3TextEncoders> {
    let mut encoders = loader::load_text_encoders(root)?;
    if let Some(q) = quant {
        encoders.quantize(q.bits())?;
    }
    Ok(encoders)
}

/// Load the heavy render-phase components — the MMDiT transformer (+ Q4/Q8 + LoRA/LoKr residuals) and
/// the VAE (dense) — everything but the triple text encoder. Factored out so the `Sequential` path can
/// load these AFTER the encoders are dropped (bounding peak to `max(triple-TE, MMDiT+VAE)`), and the
/// `Resident` path builds the same bundle up front. The quantize-then-adapters order matches the
/// pre-seam `load_variant`; the components are independent of the text encoders (separate weight files,
/// deterministic RNG-free quant/merge), so both residencies are byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, variant: Sd3Variant) -> Result<Sd3Heavy> {
    let arch = variant.arch();
    let mut transformer = loader::load_transformer(root, &arch)?;
    let vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        transformer.quantize(q.bits())?;
    }
    // T2 (sc-7883) — apply any trained LoRA/LoKr adapters over the (possibly quantized) base. The
    // base is never fused/mutated, so adapters compose with Q4/Q8 for free. The trainer's PEFT save
    // (bare diffusers paths + `.alpha`/`__metadata__`) round-trips through this exact loader. No-op
    // when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_sd3_adapters(&mut transformer, &spec.adapters)?;
    }
    // SC-18606: the one-block window is variant-generic. `stream_inventory` is the SAME seam
    // `memory_strategy::contract_for` consults to decide whether to publish
    // `BoundedTransformerResidency` as `Implemented`, so a route can never advertise a rung this
    // loader would then decline to attach — nor the reverse. `None` (an unstreamable selector or a
    // snapshot with no readable `transformer/` subtree) leaves the resident block stack in place,
    // exactly matching the rung-4-`Missing` contract the same spec produces.
    if let Some(inventory) = crate::artifact_inventory::stream_inventory(variant.id(), spec)? {
        transformer = transformer.with_block_stream(inventory, arch);
        transformer.finalize_block_stream()?;
    }
    Ok(Sd3Heavy { transformer, vae })
}

impl Generator for Sd3Large {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
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
        self.memory_strategy.as_ref().map_or_else(
            || mlx_gen::gen_core::MemorySafetyDecision::Reject {
                reason: format!("{} has no memory-strategy contract", self.descriptor.id),
            },
            |contract| crate::memory_strategy::safety_check(&self.loaded_spec, contract, context),
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        crate::memory_strategy::begin_request(&self.loaded_spec, contract, context)
    }
}

impl Sd3Large {
    /// The rich-`Result` body behind [`Generator::generate`]. The staged residency lifecycle (triple-TE
    /// encode → **drop the three encoders** under `Sequential` → load the MMDiT/VAE → denoise/decode →
    /// free the heavy bundle) is driven by the shared [`Residency::run`] seam (sc-11125), which owns the
    /// eval/drop/clear discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Per-variant sampling recipe: Large = 28 steps / guidance 3.5 (true-CFG); Large-Turbo = 4
        // steps / guidance 1.0 (distilled, guidance-baked → CFG off). The pipeline below skips the
        // uncond forward whenever guidance == 1.0, so Turbo runs ONE forward per step.
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps()) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let guidance = req
            .guidance
            .unwrap_or_else(|| self.variant.default_guidance());
        let cfg_on = guidance != 1.0;

        // img2img (epic 8588 slice A4, sc-10189): a single `Conditioning::Reference` seeds the denoise
        // from the VAE-encoded reference at `strength` (the reference's own strength, else the
        // request-level `strength`, else the 0.5 mid default — the full-range slider default the worker
        // sends). Plain txt2img otherwise. All three variants advertise `Reference` (16-ch VAE encoder
        // is loaded), so this routes for Large/Medium (true-CFG) and the distilled Large-Turbo alike.
        // Seed-independent; resolved above the residency lifecycle.
        let reference = single_reference(req)?;
        let strength = img2img_strength(reference.and_then(|(_, s)| s), req.strength);
        let sampler_name = req.sampler.as_deref();

        self.residency.run(
            &req.cancel,
            // SD3 has no PiD overlay; the heavy loader ignores `use_pid`.
            false,
            on_progress,
            // ── Phase A: triple-TE conditioning. Seed-independent (no RNG) — cond = the prompt; uncond =
            // the negative prompt (empty string when unset), used only when CFG is active (guidance !=
            // 1.0). Under `Sequential` the shared seam LOADS all three encoders, encodes, materializes,
            // then DROPS them + `clear_cache()` before the MMDiT/VAE load — bounding peak to
            // `max(triple-TE, MMDiT+VAE)`. Under `Resident` it borrows the warm encoders (byte-identical
            // to the pre-seam `encode_prompt`).
            |encoders: &Sd3TextEncoders| {
                let cond = pipeline::encode_prompt(
                    encoders,
                    &self.clip_tokenizer,
                    self.clip_pad,
                    &self.t5_tokenizer,
                    &req.prompt,
                )?;
                let uncond = if cfg_on {
                    let neg = req.negative_prompt.as_deref().unwrap_or("");
                    Some(pipeline::encode_prompt(
                        encoders,
                        &self.clip_tokenizer,
                        self.clip_pad,
                        &self.t5_tokenizer,
                        neg,
                    )?)
                } else {
                    None
                };
                Ok((cond, uncond))
            },
            // Materialize the pooled CLIP embeddings + T5 sequence context (and their uncond twins)
            // while the encoders are still alive (Sequential only) — MLX is lazy, so an un-evaluated
            // output keeps all three encoders referenced through the graph and the drop would free
            // nothing. All of `context`/`pooled` (cond + optional uncond) are forced before the drop.
            |encoded: Option<&(Sd3Conditioning, Option<Sd3Conditioning>)>| {
                let Some((cond, uncond)) = encoded else {
                    return Ok(());
                };
                let mut arrays = vec![&cond.context, &cond.pooled];
                if let Some(uc) = uncond {
                    arrays.push(&uc.context);
                    arrays.push(&uc.pooled);
                }
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            // ── Phase B: denoise/decode from the heavy bundle (MMDiT + VAE). Runs identically for both
            // residencies. `on_progress` is threaded through the seam (F-179) and shadows the outer sink.
            |heavy: &Sd3Heavy, (cond, uncond), on_progress: &mut dyn FnMut(Progress)| {
                // Static shift=3.0 schedule (scheduler_config.json), resolution-independent — build
                // once. An unset req.scheduler keeps it byte-exact; a curated name re-shapes σ over the
                // same mu=ln(3).
                let scheduler = FlowMatchEuler::from_sigmas(pipeline::resolve_sd3_sigmas(
                    req.scheduler.as_deref(),
                    steps,
                ))?;

                let attention = crate::memory_strategy::attention_plan(req);
                let transformer_window = crate::memory_strategy::transformer_window(req)?;
                let decode_tiling = crate::memory_strategy::decode_tiling(req)?;
                let mut reference_preparer = MlxReferencePreparer {
                    vae: &heavy.vae,
                    width: req.width,
                    height: req.height,
                };
                let images = self.variant.map_seeded_outputs(
                    reference.map(|(image, _)| image),
                    base_seed,
                    req.count,
                    &mut reference_preparer,
                    |_route, seed, clean| {
                        let latents = if let Some(clean) = clean {
                            pipeline::denoise_img2img_from_clean_cfg_with_memory(
                                &heavy.transformer,
                                &scheduler,
                                sampler_name,
                                seed,
                                clean,
                                strength,
                                req.width,
                                req.height,
                                steps,
                                &cond,
                                uncond.as_ref(),
                                guidance,
                                &req.cancel,
                                on_progress,
                                &req.preview,
                                attention,
                                transformer_window,
                            )?
                        } else {
                            let latents = pipeline::create_noise(seed, req.width, req.height)?;
                            pipeline::denoise_cfg_with_memory(
                                &heavy.transformer,
                                &scheduler,
                                sampler_name,
                                seed,
                                latents,
                                &cond,
                                uncond.as_ref(),
                                guidance,
                                &req.cancel,
                                on_progress,
                                &req.preview,
                                attention,
                                transformer_window,
                            )?
                        };
                        on_progress(Progress::Decoding);
                        pipeline::decode_to_image_tiled(
                            &heavy.vae,
                            &latents,
                            decode_tiling.as_ref(),
                            &req.cancel,
                        )
                    },
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Required divisor for requested image dims: VAE downsample (8) × transformer patch (2) = 16. Exposed
/// as the pinned-engine stride SceneWorks ties each advertised SD3.5 image bucket to (sc-12612),
/// mirroring `wan::config::SIZE_MULTIPLE_14B`. `validate_request` enforces exactly this value, so the
/// const cannot drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// Capability-driven request validation, factored out so it can be unit-tested without loaded weights.
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let caps = &desc.capabilities;
    let id = desc.id;
    // Empty-prompt first so it wins over the shared floor for a bare default request.
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    // The shared capability floor (F-018): count, steps range (0 and the F-004 ceiling),
    // size range, negative/guidance/true_cfg support gating + the F-053/F-001 finiteness guard,
    // sampler/scheduler/guidance_method membership, and accepted conditioning kinds. SD3 previously
    // hand-rolled a subset that missed non-finite guidance (NaN-poisons every step), sampler/scheduler
    // membership (unknown names silently fell back to the native path), and true_cfg/guidance_method
    // rejection — sibling SANA already delegates here. `?` keeps the typed `Unsupported` for gaps.
    caps.validate_request(id, req)?;
    // SD3-specific extras layered on top of the shared floor:
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{}x{} must be a multiple of {SIZE_MULTIPLE} (VAE 8 × patch 2)",
            req.width, req.height
        )));
    }
    // F-094a: a negative prompt is only honored when CFG is active (guidance != 1.0, which runs the
    // uncond forward). An explicit `guidance <= 1.0` alongside a negative prompt would silently
    // discard the negative — reject the combination with a typed error instead of the silent no-op
    // (Turbo already rejects `guidance`/`negative_prompt` wholesale via the shared floor's capability
    // checks; this is the true-CFG analogue for a guidance value that disables CFG).
    if req.negative_prompt.is_some() && req.guidance.is_some_and(|g| g <= 1.0) {
        return Err(Error::Msg(format!(
            "{id}: a negative prompt requires guidance > 1.0 (guidance <= 1.0 disables CFG, so the \
             negative prompt would be ignored); raise guidance or drop the negative prompt"
        )));
    }
    Ok(())
}

/// Extract the single reference image + its optional `strength` for img2img (epic 8588 slice A4), or
/// `None` for plain txt2img. SD3.5 conditions on exactly one reference image; more than one `Reference`
/// (or a `MultiReference`) errors. Mirrors the Krea / FLUX single-reference idiom; the descriptor now
/// advertises `Reference` on all three variants, so `validate_request` admits it before we get here.
fn single_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    match req.conditioning.as_slice() {
        [] => Ok(None),
        [Conditioning::Reference { image, strength }] => Ok(Some((image, *strength))),
        _ => Err(Error::Msg(
            "sd3: img2img supports exactly one Reference image".into(),
        )),
    }
}

/// Resolve SD3.5's reference strength with the explicit product default shared by Candle.
fn img2img_strength(reference: Option<f32>, request: Option<f32>) -> f32 {
    reference
        .or(request)
        .unwrap_or(mlx_gen::img2img::DEFAULT_IMG2IMG_STRENGTH)
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`. Both the true-CFG
// Large (E5) and the distilled CFG-off Large-Turbo (E6) register here on the shared backbone.
/// Per-component footprint (sc-10894) for the MLX fit-gate's staged-residency split — the THREE
/// text encoders (CLIP-L `text_encoder/`, CLIP-G `text_encoder_2/`, T5-XXL `text_encoder_3/`), the
/// MMDiT (`transformer/`), and the VAE (`vae/`).
///
/// The conditioning field is **projected**, not summed off disk (SC-22667). A directory sum was
/// wrong in both directions at once and the two errors do not cancel, because they land on
/// different files:
///
/// * `load_clip_l` and `load_clip_g` each do `w.cast_all(Dtype::Float32)` unconditionally, and the
///   pinned SD3.5-Large snapshot stores both at 2 bytes per element — so CLIP-L (123.65M params)
///   plus CLIP-G (694.66M) were under-declared by their own stored size, about 1.6 GB.
/// * `text_encoder_3/` ships f32 and fp16 shards side by side and
///   `resolve_sd3_text_encoder_artifacts` deliberately selects only the master set, so
///   the directory sum counted shards no load ever opens.
///
/// The projection therefore prices exactly the artifacts the resolver hands `load_text_encoders`:
/// the two CLIP files at their **materialized** f32 width, and the selected T5 shards at their
/// stored width (`T5TextEncoder::from_weights` performs no `cast_all`; the T5 promotes internally
/// per activation).
///
/// When those artifacts cannot be resolved — a snapshot with no `text_encoder*` subtree at all,
/// which the structural admission path deliberately accepts — the previous directory sum stands.
/// This builder runs at contract time, ahead of any deferred component load, so an unresolvable
/// encoder set must not become a contract-time refusal.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mut footprint = mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder", "text_encoder_2", "text_encoder_3"],
        &["transformer"],
        &["vae"],
    )?;
    if let mlx_gen::WeightsSource::Dir(root) = &spec.weights {
        if let Some(projected) = projected_conditioning_bytes(root) {
            footprint.text_encoder = projected;
        }
    }
    Ok(footprint)
}

/// Bytes the three text encoders occupy on device, at the width each is materialized in.
///
/// `None` means the exact artifact set could not be resolved or read; see [`component_footprint`]
/// for why that keeps the directory sum rather than failing the contract.
fn projected_conditioning_bytes(root: &std::path::Path) -> Option<u64> {
    use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};

    let artifacts = mlx_gen::gen_core::resolve_sd3_text_encoder_artifacts(root).ok()?;
    let clip_l =
        projected_safetensors_bytes(&artifacts.clip_l, |_| ResidentProjection::Float32).ok()?;
    let clip_g =
        projected_safetensors_bytes(&artifacts.clip_g, |_| ResidentProjection::Float32).ok()?;
    let mut total = clip_l.saturating_add(clip_g);
    for shard in &artifacts.t5_shards {
        total = total.saturating_add(
            projected_safetensors_bytes(shard, |_| ResidentProjection::Stored).ok()?,
        );
    }
    Some(total)
}

mlx_gen::register_generators! {
    pub(crate) const LARGE_REGISTRATION = descriptor => load;
    footprint = component_footprint
}

macro_rules! memory_registration {
    ($registration:ident, $behavior:ident, $provider_id:expr) => {
        pub(crate) const $registration: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $provider_id,
                contract: |spec| crate::memory_strategy::contract_for($provider_id, spec),
                safety_check: crate::memory_strategy::safety_check,
            };
        pub(crate) const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $provider_id,
                valid_fixtures: crate::memory_strategy::registered_fixture,
                begin_request: crate::memory_strategy::registered_begin_request,
            };
    };
}

memory_registration!(LARGE_MEMORY_REGISTRATION, LARGE_MEMORY_BEHAVIOR, MODEL_ID);
memory_registration!(
    TURBO_MEMORY_REGISTRATION,
    TURBO_MEMORY_BEHAVIOR,
    TURBO_MODEL_ID
);
memory_registration!(
    MEDIUM_MEMORY_REGISTRATION,
    MEDIUM_MEMORY_BEHAVIOR,
    MEDIUM_MODEL_ID
);
mlx_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = turbo_descriptor => load_turbo;
    footprint = component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const MEDIUM_REGISTRATION = medium_descriptor => load_medium;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{ConditioningKind, Modality};
    use std::cell::Cell;

    /// Feature-end review (SC-22667, E1): the conditioning field must price what the loader
    /// materializes, at the width it materializes it in. `load_clip_l` and `load_clip_g` both
    /// `cast_all(Dtype::Float32)` unconditionally over half-precision masters, and
    /// `resolve_sd3_text_encoder_artifacts` selects only the master T5 shard family while
    /// `text_encoder_3/` also ships an fp16 family. A directory sum was therefore wrong in both
    /// directions at once, on different files, so the two errors could not cancel.
    ///
    /// Mutation that fails this: dropping the `projected_conditioning_bytes` override in
    /// `component_footprint` — `text_encoder` falls back to the stored directory sum, which counts
    /// the two CLIP masters at half their materialized size and adds the fp16 shard nobody loads.
    #[test]
    fn conditioning_bytes_price_the_resolved_artifacts_at_their_materialized_width() {
        /// One-tensor safetensors file: `dtype` must agree with `width`, because the shared header
        /// reader refuses a payload length that is not `elements * dtype.size()`.
        fn write_tensor(path: &std::path::Path, dtype: &str, elements: usize, width: usize) {
            let data_bytes = elements * width;
            let mut header = format!(
                "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,{data_bytes}]}}}}"
            )
            .into_bytes();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.resize(bytes.len() + data_bytes, 0);
            std::fs::write(path, bytes).unwrap();
        }

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for component in [
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "transformer",
        ] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        // CLIP-L and CLIP-G ship at half precision and are upcast to f32 on every load.
        write_tensor(&root.join("text_encoder/model.safetensors"), "F16", 100, 2);
        write_tensor(&root.join("text_encoder_2/model.safetensors"), "F16", 50, 2);
        // The T5 master family, plus the fp16 family the resolver deliberately does not select.
        write_tensor(
            &root.join("text_encoder_3/model-00001-of-00001.safetensors"),
            "BF16",
            40,
            2,
        );
        write_tensor(
            &root.join("text_encoder_3/model.fp16-00001-of-00001.safetensors"),
            "F16",
            40,
            2,
        );
        std::fs::write(
            root.join("text_encoder_3/model.safetensors.index.json"),
            br#"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        // The resolver refuses an fp16 family without its own authoritative index, so the fixture
        // ships the complete side-by-side pair the real snapshot has.
        std::fs::write(
            root.join("text_encoder_3/model.safetensors.index.fp16.json"),
            br#"{"weight_map":{"weight":"model.fp16-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.to_path_buf()));
        let footprint = component_footprint(&spec).unwrap();
        assert_eq!(
            footprint.text_encoder,
            // CLIP-L 100 x f32 + CLIP-G 50 x f32 + the selected T5 shard at its stored width.
            400 + 200 + 80,
            "the two upcast CLIP towers must be priced at 4 bytes per element, and only the master \
             T5 shard family may be counted"
        );

        // A snapshot with no encoder subtree at all is the structural-admission shape: the resolver
        // declines, and the previous directory sum (zero here) stands rather than the contract
        // refusing at build time.
        let bare = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(bare.path().join("transformer")).unwrap();
        let bare_spec =
            mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(bare.path().to_path_buf()));
        assert_eq!(component_footprint(&bare_spec).unwrap().text_encoder, 0);
    }

    struct CountingReferencePreparer<'a> {
        expected_route: Sd3Variant,
        expected_reference: u8,
        request_id: u32,
        encode_count: &'a Cell<u32>,
        eval_count: &'a Cell<u32>,
    }

    impl Sd3ReferencePreparer<u8, Box<u32>, ()> for CountingReferencePreparer<'_> {
        fn prepare(&mut self, route: Sd3Variant, source: &u8) -> std::result::Result<Box<u32>, ()> {
            assert_eq!(route, self.expected_route);
            assert_eq!(*source, self.expected_reference);
            self.encode_count.set(self.encode_count.get() + 1);
            // The production adapter evaluates the lazy VAE result immediately after encoding.
            // Keep a separate counter so moving either operation into the output loop fails.
            self.eval_count.set(self.eval_count.get() + 1);
            Ok(Box::new(self.request_id))
        }
    }

    fn production_reference_placement_is_valid(source: &str) -> bool {
        let Some((_, adapter_tail)) = source.split_once(
            "impl Sd3ReferencePreparer<Image, Array, Error> for MlxReferencePreparer<'_> {",
        ) else {
            return false;
        };
        let Some((adapter_body, _)) = adapter_tail.split_once("\n}\n\nimpl Sd3Variant {") else {
            return false;
        };

        let Some((_, generate_tail)) = source.split_once("    fn generate_impl(") else {
            return false;
        };
        let Some((generate_body, _)) =
            generate_tail.split_once("\n/// Required divisor for requested image dims")
        else {
            return false;
        };

        let encode = "pipeline::encode_reference(";
        let eval = "mlx_rs::transforms::eval([&clean])";
        let adapter_order_is_encode_then_eval = adapter_body
            .find(encode)
            .zip(adapter_body.find(eval))
            .is_some_and(|(encode_at, eval_at)| encode_at < eval_at);

        adapter_body.matches(encode).count() == 1
            && adapter_body.matches(eval).count() == 1
            && adapter_order_is_encode_then_eval
            && generate_body
                .matches("let mut reference_preparer = MlxReferencePreparer {")
                .count()
                == 1
            && generate_body
                .matches("self.variant.map_seeded_outputs(")
                .count()
                == 1
            && generate_body.matches("&mut reference_preparer,").count() == 1
            && !generate_body.contains(encode)
            && !generate_body.contains(eval)
    }

    fn move_prepare_operation_into_output(
        source: &str,
        operation: &str,
        replacement: &str,
        output_statement: &str,
    ) -> String {
        let without_adapter_operation = source.replacen(operation, replacement, 1);
        without_adapter_operation.replacen(
            "                    |_route, seed, clean| {",
            &format!("                    |_route, seed, clean| {{\n{output_statement}"),
            1,
        )
    }

    fn tiny_image() -> Image {
        Image {
            width: 16,
            height: 16,
            pixels: vec![0u8; 16 * 16 * 3],
        }
    }

    /// A valid request carrying `refs` `Reference` conditionings (each with `strength`).
    fn ref_req(refs: usize, strength: Option<f32>) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            conditioning: (0..refs)
                .map(|_| Conditioning::Reference {
                    image: tiny_image(),
                    strength,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn every_variant_schedules_one_request_local_reference_encode_and_distinct_seeds() {
        for route in [
            Sd3Variant::Large,
            Sd3Variant::LargeTurbo,
            Sd3Variant::Medium,
        ] {
            let encode_count = Cell::new(0_u32);
            let eval_count = Cell::new(0_u32);
            let reference = 17_u8;

            let run_request = |request_id: u32, base_seed: u64, count: u32| {
                let mut preparer = CountingReferencePreparer {
                    expected_route: route,
                    expected_reference: reference,
                    request_id,
                    encode_count: &encode_count,
                    eval_count: &eval_count,
                };
                route
                    .map_seeded_outputs(
                        Some(&reference),
                        base_seed,
                        count,
                        &mut preparer,
                        |output_route, seed, clean| {
                            assert_eq!(output_route, route);
                            let clean = clean.expect("reference request carries a clean latent");
                            Ok::<_, ()>((
                                seed,
                                **clean,
                                std::ptr::from_ref(clean.as_ref()) as usize,
                            ))
                        },
                    )
                    .unwrap()
            };

            let first = run_request(1, u64::MAX - 1, 3);
            assert_eq!(
                first.iter().map(|(seed, _, _)| *seed).collect::<Vec<_>>(),
                [u64::MAX - 1, u64::MAX, 0]
            );
            assert!(first.iter().all(|(_, request_id, _)| *request_id == 1));
            assert!(
                first.windows(2).all(|pair| pair[0].2 == pair[1].2),
                "every output must borrow the same prepared clean latent"
            );
            assert_eq!(encode_count.get(), 1);
            assert_eq!(eval_count.get(), 1);

            let second = run_request(2, 41, 2);
            assert_eq!(
                second.iter().map(|(seed, _, _)| *seed).collect::<Vec<_>>(),
                [41, 42]
            );
            assert!(second.iter().all(|(_, request_id, _)| *request_id == 2));
            assert_eq!(encode_count.get(), 2, "a second request must encode afresh");
            assert_eq!(eval_count.get(), 2, "a second request must evaluate afresh");
        }
    }

    #[test]
    fn production_reference_work_is_bound_to_the_tested_adapter_and_scheduler() {
        let source = include_str!("model.rs");
        assert!(production_reference_placement_is_valid(source));

        let mutations = [
            (
                "production scheduler bypass",
                source.replacen(
                    "self.variant.map_seeded_outputs(",
                    "self.variant.map_seeded_outputs_bypassed(",
                    1,
                ),
            ),
            (
                "reference encode moved into the per-output closure",
                move_prepare_operation_into_output(
                    source,
                    "pipeline::encode_reference(",
                    "pipeline::reference_encode_moved_for_mutation(",
                    "                        let _mutation = \
                         pipeline::encode_reference(/* moved per output */);\n",
                ),
            ),
            (
                "reference eval moved into the per-output closure",
                move_prepare_operation_into_output(
                    source,
                    "mlx_rs::transforms::eval([&clean])",
                    "mlx_rs::transforms::eval_moved_for_mutation([&clean])",
                    "                        let _mutation = \
                         mlx_rs::transforms::eval([&clean]);\n",
                ),
            ),
            (
                "production adapter disconnected",
                source.replacen("&mut reference_preparer,", "&mut bypass_preparer,", 1),
            ),
        ];
        for (mutation, mutated_source) in mutations {
            assert!(
                !production_reference_placement_is_valid(&mutated_source),
                "placement contract must reject mutation: {mutation}"
            );
        }
    }

    #[test]
    fn all_variants_advertise_reference_img2img() {
        // A4 (sc-10189): every SD3.5 variant advertises `Reference` (16-ch VAE encoder is loaded on all
        // three; the flow-match schedule is sliceable) — Large + Medium (true-CFG) + Large-Turbo (distilled).
        for d in [descriptor(), turbo_descriptor(), medium_descriptor()] {
            assert_eq!(
                d.capabilities.conditioning,
                vec![ConditioningKind::Reference],
                "{} must advertise Reference img2img",
                d.id
            );
        }
    }

    #[test]
    fn validate_accepts_a_single_reference() {
        // The descriptor advertises Reference, so validation admits one reference conditioning (the
        // img2img path). This holds on the distilled Turbo too (img2img needs no guidance/CFG).
        assert!(validate_request(&descriptor(), &ref_req(1, Some(0.5))).is_ok());
        assert!(validate_request(&turbo_descriptor(), &ref_req(1, Some(0.4))).is_ok());
        assert!(validate_request(&medium_descriptor(), &ref_req(1, None)).is_ok());
    }

    /// sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SD3.5 bucket to.
    /// Pin the value and mutation-check that a multiple of 8 which is not SIZE_MULTIPLE (16) is rejected
    /// with the stride error, and an on-stride in-range size passes.
    #[test]
    fn size_multiple_is_the_pinned_stride() {
        assert_eq!(SIZE_MULTIPLE, 16);
        let off = validate_request(
            &descriptor(),
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                height: 1024,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(off.contains("multiple of 16"), "got: {off}");
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            }
        )
        .is_ok());
    }

    #[test]
    fn single_reference_extracts_one_or_errors() {
        // No conditioning → plain txt2img.
        let plain = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(single_reference(&plain).unwrap().is_none());
        // Exactly one → the image + its strength.
        let r1 = ref_req(1, Some(0.4));
        let one = single_reference(&r1).unwrap();
        assert_eq!(one.map(|(_, s)| s), Some(Some(0.4)));
        // More than one → error (SD3.5 conditions on a single reference).
        assert!(single_reference(&ref_req(2, None)).is_err());
    }

    #[test]
    fn missing_img2img_strength_defaults_to_midpoint() {
        assert_eq!(
            img2img_strength(None, None),
            mlx_gen::img2img::DEFAULT_IMG2IMG_STRENGTH
        );
        assert_eq!(img2img_strength(Some(0.75), Some(0.2)), 0.75);
        assert_eq!(img2img_strength(None, Some(0.2)), 0.2);
    }

    #[test]
    fn descriptor_is_sd3_5_large() {
        let d = descriptor();
        assert_eq!(d.id, "sd3_5_large");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(
            !d.capabilities.supports_true_cfg,
            "the MLX SD3.5 render path does not consume request.true_cfg"
        );
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let d = descriptor();
        let req = GenerationRequest::default();
        let err = validate_request(&d, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_bad_size_and_count() {
        let d = descriptor();
        // non-multiple-of-16
        let req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1000,
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // out-of-range count
        let req = GenerationRequest {
            prompt: "a fox".into(),
            count: 99,
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // zero steps
        let req = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // a plain valid request passes.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
    }

    #[test]
    fn validate_accepts_guidance_and_negative_prompt() {
        // Large uses classifier-free guidance: guidance + negative prompt are supported (unlike a
        // distilled Turbo), while the separate request.true_cfg knob is rejected.
        let d = descriptor();
        let req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.5),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
        let with_true_cfg = GenerationRequest {
            true_cfg: Some(4.5),
            ..req
        };
        let err = validate_request(&d, &with_true_cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("true_cfg is not supported"), "got: {err}");
    }

    #[test]
    fn validate_rejects_negative_prompt_with_cfg_disabling_guidance() {
        // F-094a: on a true-CFG variant, a negative prompt is only honored when guidance > 1.0. An
        // explicit guidance <= 1.0 disables CFG, so the negative would be silently dropped — the
        // combination is now a typed error rather than a silent no-op.
        let d = descriptor();
        let base = GenerationRequest {
            prompt: "a fox".into(),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        // guidance == 1.0 (CFG off) + a negative prompt → rejected.
        let g1 = GenerationRequest {
            guidance: Some(1.0),
            ..base.clone()
        };
        let err = validate_request(&d, &g1).unwrap_err().to_string();
        assert!(err.contains("guidance > 1.0"), "got: {err}");
        // guidance < 1.0 is likewise rejected with a negative prompt.
        let g0 = GenerationRequest {
            guidance: Some(0.5),
            ..base.clone()
        };
        assert!(validate_request(&d, &g0).is_err());
        // guidance > 1.0 + a negative prompt is fine (CFG on, negative honored).
        let g2 = GenerationRequest {
            guidance: Some(4.5),
            ..base.clone()
        };
        assert!(validate_request(&d, &g2).is_ok());
        // A negative prompt with UNSET guidance is fine — the variant's default guidance (3.5) is
        // > 1.0, so CFG stays on.
        assert!(validate_request(&d, &base).is_ok());
    }

    #[test]
    fn turbo_descriptor_is_distilled_cfg_off() {
        // E6: Large-Turbo registers its own id, is NOT true-CFG, and rejects guidance / negative
        // prompt (guidance is baked into the distilled weights — CFG off).
        let d = turbo_descriptor();
        assert_eq!(d.id, "sd3_5_large_turbo");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn turbo_recipe_defaults() {
        // The distilled few-step / guidance-baked recipe: 4 steps, guidance 1.0 (CFG off).
        assert_eq!(Sd3Variant::LargeTurbo.default_steps(), 4);
        assert_eq!(Sd3Variant::LargeTurbo.default_guidance(), 1.0);
        // And Large stays true-CFG at 28 steps / guidance 3.5.
        assert_eq!(Sd3Variant::Large.default_steps(), 28);
        assert_eq!(Sd3Variant::Large.default_guidance(), 3.5);
    }

    #[test]
    fn turbo_rejects_guidance_and_negative_prompt() {
        // On Turbo, supplying guidance or a negative prompt is a validation error (CFG-off variant).
        let d = turbo_descriptor();
        let with_guidance = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(validate_request(&d, &with_guidance).is_err());
        let with_neg = GenerationRequest {
            prompt: "a fox".into(),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &with_neg).is_err());
        // A plain txt2img request (no guidance/neg) passes.
        let plain = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&d, &plain).is_ok());
    }

    #[test]
    fn turbo_load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load_turbo(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
        // The error is namespaced to the Turbo id (not the Large id).
        assert!(err.contains("sd3_5_large_turbo"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_quant_spec() {
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }

    // --- M3 (sc-7869): SD3.5-Medium vertical -----------------------------------------------------

    #[test]
    fn medium_descriptor_is_sd3_5_medium_cfg() {
        // Medium registers its own engine id and uses classifier-free guidance (negative prompt +
        // guidance), like Large (NOT a distilled Turbo). It does not consume request.true_cfg.
        let d = medium_descriptor();
        assert_eq!(d.id, "sd3_5_medium");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // The higher-res ceiling: Medium's pos_embed_max_size (384) spans up to 1440² (patch grid
        // 90×90 ≤ 384²); the descriptor caps there to stay inside the Mac activation budget (2K is
        // activation-bound — the SCAIL-2 / SeedVR2 lesson).
        assert_eq!(d.capabilities.max_size, 1440);
    }

    #[test]
    fn medium_recipe_defaults() {
        // Medium's reference recipe: 40 steps / guidance 5.0 (true-CFG, more guidance-sensitive than
        // Large per Stability's model card). Shift stays 3.0 (Medium scheduler_config.json).
        assert_eq!(Sd3Variant::Medium.default_steps(), 40);
        assert_eq!(Sd3Variant::Medium.default_guidance(), 5.0);
    }

    #[test]
    fn medium_uses_mmdit_x_arch() {
        // The Medium generator is driven by the MMDiT-X arch (24 blocks, hidden 1536, dual-attention
        // in the first 13) — distinct from Large's plain MMDiT.
        let arch = Sd3Variant::Medium.arch();
        assert_eq!(arch.num_layers, 24);
        assert_eq!(arch.hidden(), 1536);
        assert_eq!(arch.dual_attention_layers, 13);
        assert_eq!(arch.pos_embed_max_size, 384);
        assert!(arch.is_dual_attention_block(12));
        assert!(!arch.is_dual_attention_block(13));
    }

    #[test]
    fn medium_validate_accepts_guidance_and_negative_prompt() {
        let d = medium_descriptor();
        let req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
        assert!(
            validate_request(
                &d,
                &GenerationRequest {
                    true_cfg: Some(5.0),
                    ..req
                },
            )
            .is_err(),
            "Medium must reject request.true_cfg because the render path never consumes it"
        );
    }

    #[test]
    fn medium_validate_guards_resolution_ceiling() {
        // 1440² is allowed (the validated Medium ceiling); anything above max_size is rejected by the
        // shared guard — the activation-budget cap (2K would be activation-bound).
        let d = medium_descriptor();
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1440,
            height: 1440,
            ..Default::default()
        };
        assert!(validate_request(&d, &ok).is_ok());
        let too_big = GenerationRequest {
            prompt: "a fox".into(),
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        // F-018: SD3 now delegates the size-range check to the shared floor ("outside supported range").
        let err = validate_request(&d, &too_big).unwrap_err().to_string();
        assert!(err.contains("outside supported range"), "got: {err}");
    }

    #[test]
    fn medium_load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load_medium(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
        assert!(err.contains("sd3_5_medium"), "got: {err}");
    }

    #[test]
    fn medium_load_accepts_quant_spec() {
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load_medium(&spec)
                .err()
                .expect("expected an error")
                .to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that SD3's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the up-front single-file guard
    // passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the triple text encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The `sequential_residency_real_weights.rs` A/B is
    // `#[ignore]`d (needs a snapshot); this runs by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/sd3-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        // The default run covers all three variants share the one dispatch — assert on Large.
        let res = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Sequential),
            Sd3Variant::Large,
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Resident),
            Sd3Variant::Large,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file"),
            "expected an eager-load failure, not the up-front single-file guard: {msg}"
        );
    }

    #[test]
    fn descriptor_advertises_sequential_offload() {
        // All three SD3 ids honor the shared Residency seam (the descriptor bit consumers read).
        for d in [descriptor(), turbo_descriptor(), medium_descriptor()] {
            assert!(
                d.capabilities.supports_sequential_offload,
                "{} must advertise supports_sequential_offload",
                d.id
            );
        }
    }
}
