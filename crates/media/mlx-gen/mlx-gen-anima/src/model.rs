//! The three Anima generators (`anima_base`, `anima_aesthetic`, `anima_turbo`) — [`Generator`]
//! implementations + descriptors + [`load_base`], [`load_aesthetic`], and [`load_turbo`] entry points,
//! plus explicit registration constants. All three share the same
//! architecture (Cosmos-Predict2 DiT + `AnimaTextConditioner` + Qwen3-0.6B TE + Qwen-Image VAE) and
//! differ only in the DiT weights file + default steps/CFG.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor,
    OffloadPolicy, Precision, Progress, Quant, Residency, Result, SizeFloor,
};

use crate::config::{Variant, RES_MULTIPLE};
use crate::pipeline::{AnimaCondInputs, AnimaDecodeView, AnimaHeavy, AnimaText};

const MAX_COUNT: u32 = 8;
const RES_MIN: u32 = 512;
/// Above ~1920 px/side the Cosmos RoPE would index out of its trained range; `rope.rs` **rejects**
/// (errors on) such a request rather than clamping. The shipped ceiling is 1536² (post-patch 96, well
/// within the 120-position max_size), so the guard is unreachable via the normal path. See [`crate::rope`].
const RES_MAX: u32 = 1536;

/// Build the descriptor for a variant. Turbo is the merged CFG-free student (no guidance / negative
/// prompt); Base/Aesthetic run true classifier-free guidance.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let cfg_capable = variant.uses_cfg();
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: variant.id(),
        family: "anima",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: cfg_capable,
            supports_guidance: cfg_capable,
            supports_true_cfg: false,
            conditioning: vec![],
            // LoRA/LoKr injection is sc-10521; every projection is an adapter-ready `AdaptableLinear`.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow over the unified curated-sampler framework (epic 7114). The native default
            // (req.sampler == None) is the recommended er_sde solver; the full menu is advertised.
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // Q4/Q8 quant tiers (sc-10517). Anima is convert-at-install: the SceneWorks worker packs
            // the Cosmos DiT on-device (the conditioner + Qwen3 TE + VAE stay dense bf16), and this
            // crate's loader packed-detects the tier off the on-disk `{base}.scales` — so `load`
            // ACCEPTS any `spec.quantize` (it is advisory; the resolved tier dir dictates precision,
            // like SANA). The worker reads `supported_quants` for its capability advertisement
            // (gen-core sc-3723); every advertised tier actually loads, so this is honest.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: true,
            // Wired onto the shared `Residency` seam (epic 10834, sc-10840); honors Sequential offload
            // (F-176). Under `Sequential` the Qwen3-0.6B text encoder is encoded, materialized, then
            // dropped before the DiT + bundled conditioner + VAE load — bounding peak unified memory to
            // `max(Qwen3-TE, DiT+conditioner+VAE)`. Q4/Q8 are packed convert-at-install tiers (no
            // load-time re-quant), so no F-181 dense-requant advisory is needed (mirrors SANA).
            supports_sequential_offload: true,
            supports_preview: true,
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

pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(Variant::Base)
}
pub fn descriptor_aesthetic() -> ModelDescriptor {
    descriptor_for(Variant::Aesthetic)
}
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::Turbo)
}

/// A loaded Anima generator: the cached descriptor + variant + the component-residency strategy
/// (epic 10834, sc-10840). Holds ONLY the [`Residency`] (no direct encoder/DiT/VAE fields — a retained
/// component would defeat the `Sequential` drop): `Resident` (default) holds the Qwen3 TE + DiT +
/// bundled conditioner + VAE warm for the whole job and across jobs; `Sequential` holds only the
/// per-phase loader closures and re-loads each per generation in phase order (encode → **drop the
/// Qwen3 TE** → conditioner/DiT/VAE), bounding peak unified memory to
/// `max(Qwen3-TE, DiT+conditioner+VAE)`.
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    residency: Residency<AnimaText, AnimaHeavy>,
    /// The spec this generator was loaded from — the memory contract is per-`LoadSpec` (rung 4 is a
    /// per-LOAD declaration), and `safety_check` must reproduce the loaded generator's tier.
    loaded_spec: LoadSpec,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    /// Ladder rung 1's default when a request carries no `GenerationMemory` — the legacy load-time
    /// `OffloadPolicy`. A shared-contract request overrides it per request.
    default_stage_residency: bool,
    /// Whether this load can rebuild transformer blocks per window (ladder rung 4).
    streamable: bool,
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Base)
}
pub fn load_aesthetic(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Aesthetic)
}
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Turbo)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense bf16 precision is wired (drop the precision override)"
        )));
    }
    // Q4/Q8 tiers (sc-10517) are NOT quantized at load. Anima is convert-at-install: the SceneWorks
    // worker packs the Cosmos DiT on-device (`convert::quantize_anima_dit`, conditioner + Qwen3 TE +
    // VAE kept dense bf16), and the DiT's `AdaptableLinear`s packed-detect the tier off the on-disk
    // `{base}.scales` inside `CosmosDiT::from_weights`. So a `spec.quantize` value is ADVISORY — the
    // resolved tier directory dictates the actual precision — and we accept any tier without a
    // load-time `.quantize()` (mirrors SANA, the Group-B packed-detect convert-at-install path;
    // Kolors/sd3 by contrast load-time-quantize, so SANA is the true precedent here).
    //
    // Quant + LoRA/LoKr together IS supported (sc-10578). No guard is needed here: the DiT's
    // `AdaptableLinear`s already carry a `LinearBase::Quantized` on a packed tier, and `AdaptableLinear`
    // evaluates `base(x) + Σ adapter.residual(x)` — i.e. the additive branch `y = xW_packed + scale·(xA)B`
    // (epic 10043) — leaving the packed codes untouched. A LoKr on a packed base installs as the
    // structured `Adapter::LokrStructured` (the Kronecker vec-trick), so it never materializes an
    // `[out,in]` delta; the shared `install_lycoris_groups` picks that form off `is_quantized()`.
    // (A LoHa has no deferred form, so it falls back to the materialized delta there — correct, but
    // memory-hungry. Whether a packed base should refuse that is sc-10678, not a load-gate concern.)
    let _ = spec.quantize;
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        residency: build_residency(spec, variant)?,
        memory_strategy: crate::memory_strategy::memory_strategy_contract(id, spec)
            .map_err(|error| Error::Msg(error.to_string()))?,
        default_stage_residency: spec.offload_policy == OffloadPolicy::Sequential,
        streamable: crate::memory_strategy::streamable(spec),
        loaded_spec: spec.clone(),
    }))
}

/// The policy→[`Residency`] dispatch every Anima variant shares (sc-10840), routed through the single
/// [`Residency::from_policy`] seam (F-180) so no variant re-derives the `match offload_policy`.
/// `Resident` eager-loads the Qwen3 TE phase + heavy bundle now; `Sequential` captures the two loader
/// closures and loads nothing now, deferring each to [`Residency::run`]. Both use the same
/// [`AnimaText::load`] / [`AnimaHeavy::load`], so the `Resident` composition is byte-identical to the
/// pre-seam `AnimaComponents` (independent files, RNG-free load + adapter merge). Anima has no PiD
/// overlay, so the heavy loader's `use_pid` flag is ignored. Adapters are baked in the heavy loader
/// (the DiT + bundled conditioner both live there), stacked/mixed and strict — an unmatched target
/// errors rather than loading a partial distillation (sc-10521 / sc-10274). The deferral is
/// weight-free-testable: under `Sequential` this touches no component weights, so a dispatch that
/// ignored `offload_policy` would eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    variant: Variant,
) -> Result<Residency<AnimaText, AnimaHeavy>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::request_scoped_from_policy(
        spec.offload_policy,
        // The Qwen3 TE has no re-materializable form here: rung 4's only implemented component scope
        // is the DiT, and rung 1 already sheds the whole tower before the heavy phase loads.
        move |_streamable| AnimaText::load(&spec_text.weights, variant),
        move |_use_pid, streamable| {
            let mut heavy = AnimaHeavy::load_with_stream(&spec_heavy.weights, variant, streamable)?;
            if !spec_heavy.adapters.is_empty() {
                // Strict-applies DiT + conditioner in one pass AND captures the per-block adapters
                // rung 4 replays, so a windowed render carries the same LoRA the resident one does.
                heavy.apply_adapters(&spec_heavy.adapters)?;
            }
            Ok(heavy)
        },
    )
}

impl Generator for Anima {
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
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::begin_request(
            self.descriptor.id,
            &self.loaded_spec,
            &self.memory_strategy,
            context,
        )
    }
}

impl Anima {
    /// One request, driven by the shared three-phase [`Residency::run_staged_request_scoped`] seam
    /// (SC-15524). It owns the eval/drop/`clear_cache` discipline, the stage-boundary cancel checks
    /// and the error-safe flush; this body only supplies Anima's phase tensors.
    ///
    /// **Rung 1 is request-scoped.** `stage_residency` comes from the request's
    /// [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory) when the shared contract admitted
    /// one, and falls back to the legacy load-time `OffloadPolicy` otherwise — so one cached
    /// generator serves warm → staged → warm without reconstruction, and a request that carries no
    /// memory block is byte-for-byte unaffected.
    ///
    /// **Staging is a real three-phase schedule here, not two.** The historical seam released only
    /// the Qwen3 TE and then held the DiT + conditioner + VAE through the decode. Every latent is now
    /// materialized while the DiT is alive, the 4.18 GB DiT + conditioner are shed, and only the
    /// ~250 MB VAE survives into the decode — which is also what makes rung 4's window worth
    /// selecting, and why rung 4 declares rung 1 as a same-request prerequisite.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let variant = self.variant;
        let guidance = if variant.uses_cfg() {
            req.guidance.unwrap_or(variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();
        // Epic 7114 sampler/scheduler axis: `None` ⇒ the native er_sde default / native σ schedule.
        let sampler = req
            .sampler
            .clone()
            .unwrap_or_else(|| crate::pipeline::DEFAULT_SAMPLER.to_string());
        let scheduler = req.scheduler.clone();

        // Resolve the whole ladder up front so an unimplementable request fails BEFORE any component
        // loads, rather than part-way through a staged schedule.
        let stage_residency =
            crate::memory_strategy::stage_residency(req, self.default_stage_residency);
        let attention = crate::memory_strategy::attention_plan(req);
        let window_size = crate::memory_strategy::transformer_window_size(req)?;
        let decode_tiling = crate::memory_strategy::decode_tiling(req)?;
        if window_size.is_some() && !self.streamable {
            return Err(Error::Unsupported(format!(
                "{}: bounded transformer residency requires a DeferredMaterialization load",
                self.descriptor.id
            )));
        }
        if window_size.is_some() && !stage_residency {
            return Err(Error::Unsupported(format!(
                "{}: bounded transformer residency requires staged residency engaged in the same \
                 request — bounding the DiT while the conditioner and VAE stay resident does not \
                 move the request peak",
                self.descriptor.id
            )));
        }

        self.residency.run_staged_request_scoped(
            stage_residency,
            // Only build the re-materializable DiT form when this request will actually window it:
            // arming it otherwise would evict a warm pair for nothing.
            window_size.is_some(),
            &req.cancel,
            // Anima has no PiD overlay; the heavy loader ignores `use_pid`.
            false,
            on_progress,
            // ── Phase A: encode the conditioner INPUTS (Qwen3 forward + mask-multiply). Seed-independent
            // (no RNG). `uncond` is encoded iff the variant uses CFG (NOT gated on the guidance value —
            // preserving the pre-seam behavior of running the uncond forward even at guidance 1.0). When
            // staged, the shared seam materializes these + DROPS the Qwen3 TE before the heavy load.
            |text: &AnimaText| {
                let cond = text.encode_inputs(&req.prompt)?;
                let uncond = if variant.uses_cfg() {
                    Some(text.encode_inputs(&negative)?)
                } else {
                    None
                };
                Ok((cond, uncond))
            },
            // Materialize the masked Qwen3 states + T5 ids (cond + optional uncond) while the TE is still
            // alive (staged only) — MLX is lazy, so an un-evaluated `source` keeps the TE referenced
            // and dropping it would free nothing. The T5 weights are host data (no eval).
            |encoded: Option<&(AnimaCondInputs, Option<AnimaCondInputs>)>| {
                let Some((cond, uncond)) = encoded else {
                    return Ok(());
                };
                let mut arrays = vec![&cond.source, &cond.t5_ids];
                if let Some(u) = uncond {
                    arrays.push(&u.source);
                    arrays.push(&u.t5_ids);
                }
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            // ── Phase B: conditioner forward (once per cond/uncond — seed-independent) then the count
            // loop of denoise. Identical body for every residency and rung, so a staged or windowed job
            // is numerically identical to a warm unbounded one.
            |heavy: &AnimaHeavy, (cond, uncond), on_progress: &mut dyn FnMut(Progress)| {
                let window = heavy.block_window(window_size, &req.cancel)?;
                let cond_enc = heavy.conditioner_forward(&cond)?;
                let uncond_enc = match &uncond {
                    Some(u) => Some(heavy.conditioner_forward(u)?),
                    None => None,
                };
                let mut latents = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    // Release the MLX cache between images so a batch doesn't accumulate to a SIGKILL
                    // (sc-5567).
                    if n > 0 {
                        mlx_rs::memory::clear_cache();
                    }
                    let seed = base_seed.wrapping_add(n as u64);
                    latents.push(heavy.denoise_one(
                        &cond_enc,
                        uncond_enc.as_ref(),
                        req.width,
                        req.height,
                        steps,
                        guidance,
                        &sampler,
                        scheduler.as_deref(),
                        seed,
                        &req.preview,
                        &req.cancel,
                        on_progress,
                        attention,
                        window,
                    )?);
                }
                Ok(latents)
            },
            // Force every latent before the DiT + conditioner are shed: an unevaluated latent still
            // references the DiT graph, so dropping it would free nothing (the SC-15750 MLX trap, one
            // level up from the block window).
            |latents: &Vec<mlx_rs::Array>| Ok(mlx_rs::transforms::eval(latents.iter())?),
            // ── Phase C: decode from whichever VAE survived — the warm bundle's or the shed one's.
            |view: AnimaDecodeView<'_>, latents, on_progress: &mut dyn FnMut(Progress)| {
                on_progress(Progress::Decoding);
                let mut images = Vec::with_capacity(latents.len());
                for latent in &latents {
                    images.push(view.decode_one(latent, &req.cancel, decode_tiling.as_ref())?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Capability-driven request validation (testable without loaded weights): non-empty prompt, size a
/// multiple of 16, steps ≥ 1, on top of the shared [`Capabilities::validate_request`] floor.
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Explicit registration constants for all three variants.
mlx_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base;
    footprint = crate::loader::component_footprint
}

/// The shared-ladder registration pair every Anima catalog entry publishes. All three route to the
/// SAME contract builder and safety check — the implementation is shared by construction — while the
/// contract itself is still built per `LoadSpec`, so rung 4's per-LOAD declaration is preserved.
macro_rules! memory_registration {
    ($registration:ident, $behavior:ident, $provider_id:expr) => {
        pub(crate) const $registration: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $provider_id,
                contract: |spec| {
                    crate::memory_strategy::memory_strategy_contract($provider_id, spec)
                },
                safety_check: crate::memory_strategy::safety_check,
            };
        pub(crate) const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $provider_id,
                valid_fixtures: crate::memory_strategy::registered_valid_fixture,
                begin_request: |spec, contract, context| {
                    crate::memory_strategy::registered_begin_request(
                        $provider_id,
                        spec,
                        contract,
                        context,
                    )
                },
            };
    };
}

memory_registration!(BASE_MEMORY_REGISTRATION, BASE_MEMORY_BEHAVIOR, "anima_base");
memory_registration!(
    AESTHETIC_MEMORY_REGISTRATION,
    AESTHETIC_MEMORY_BEHAVIOR,
    "anima_aesthetic"
);
memory_registration!(
    TURBO_MEMORY_REGISTRATION,
    TURBO_MEMORY_BEHAVIOR,
    "anima_turbo"
);
mlx_gen::register_generators! {
    pub(crate) const AESTHETIC_REGISTRATION = descriptor_aesthetic => load_aesthetic;
    footprint = crate::loader::component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo;
    footprint = crate::loader::component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "an anime girl with silver hair, detailed".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn three_variants_registered() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            assert!(
                crate::provider_registry()
                    .unwrap()
                    .generators()
                    .copied()
                    .any(|r| (r.descriptor)().id == id),
                "id {id} not registered"
            );
        }
    }

    #[test]
    fn descriptors_surface() {
        let b = descriptor_base();
        assert_eq!(b.id, "anima_base");
        assert_eq!(b.family, "anima");
        assert_eq!(b.backend, "mlx");
        assert_eq!(b.modality, Modality::Image);
        assert!(b.capabilities.supports_guidance);
        assert!(b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.requires_sigma_shift);
        assert!(b.capabilities.supports_lora && b.capabilities.supports_lokr);
        assert!(b.capabilities.supports_preview);
        assert!(b.capabilities.mac_only);
        // Q4/Q8 tiers advertised (sc-10517): convert-at-install packs the DiT on-device and the loader
        // packed-detects each tier, so every advertised tier actually loads — an honest advertisement.
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(b.capabilities.min_size, 512);
        assert_eq!(b.capabilities.max_size, 1536);
        // Turbo is the CFG-free merged student.
        let t = descriptor_turbo();
        assert!(descriptor_aesthetic().capabilities.supports_preview);
        assert!(t.capabilities.supports_preview);
        assert!(!t.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_negative_prompt);
        // er_sde is advertised in the curated menu.
        assert!(
            b.capabilities.samplers.contains(&"er_sde"),
            "er_sde not advertised"
        );
    }

    #[test]
    fn validate_rejects_bad_requests() {
        assert!(validate_request(&descriptor_base(), &GenerationRequest::default()).is_err()); // empty prompt
        assert!(validate_request(&descriptor_base(), &req(1000, 1024)).is_err()); // not mult of 16
        assert!(validate_request(&descriptor_base(), &req(256, 256)).is_err()); // below min
        assert!(validate_request(&descriptor_base(), &req(2048, 2048)).is_err()); // above max
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor_base(), &req(1536, 1536)).is_ok());

        // sc-12612: `RES_MULTIPLE` is the pinned stride SceneWorks ties every advertised Anima bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE scale) but
        // not RES_MULTIPLE (16) — 1000 = 125×8, in range [512, 1536] — is rejected with the stride
        // error, and an on-stride in-range size passes.
        assert_eq!(RES_MULTIPLE, 16);
        let off_stride = validate_request(&descriptor_base(), &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiple of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());

        // Turbo rejects guidance / negative (CFG-free).
        assert!(validate_request(
            &descriptor_turbo(),
            &GenerationRequest {
                guidance: Some(4.5),
                ..req(1024, 1024)
            }
        )
        .is_err());
    }

    #[test]
    fn load_accepts_quant_spec() {
        // Q4/Q8 are wired (sc-10517) as packed-detected tiers: a quantize request must get PAST the
        // load gate (no "unsupported"/defer rejection) and fail later on the missing snapshot instead —
        // proving `spec.quantize` is accepted as advisory, not rejected.
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-anima".into())).with_quant(q);
            let e = load_base(&spec).err().expect("error").to_string();
            assert!(
                !e.contains("quant") && !e.contains("sc-10517"),
                "quant spec must be accepted, got a quant-rejection: {e}"
            );
        }
    }

    #[test]
    fn load_accepts_quant_plus_adapter_sc10578() {
        // The inverse of the guard this story removed. A packed tier requested WITH an adapter must no
        // longer be rejected on CAPABILITY grounds: `AdaptableLinear` runs `base(x) + Σ residual(x)`
        // over a `LinearBase::Quantized` (the epic-10043 additive branch), and a packed LoKr installs as
        // the structured Kronecker form. The pair is supported.
        //
        // A nonexistent weights dir still errors — but it must now fail on WEIGHTS/IO, not on the pair.
        // Asserting the absence of the old rejection is what keeps a future "narrow the guard back"
        // refactor from silently re-breaking q4+LoRA, which is the single most common Anima workflow.
        //
        // This test only guards the load GATE. The numeric proof that the residual actually rides on
        // the packed codes lives in the real-weights `tests/packed_adapters.rs` (`#[ignore]`d, not run
        // in CI); the CI-covered proof of the install math is in the shared core unit tests,
        // `mlx-gen/src/adapters/loader.rs::lokr_on_packed_base_installs_structured_and_matches_dense`.
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        for variant_load in [load_base, load_aesthetic, load_turbo] {
            let adapter = AdapterSpec::new(
                "/nonexistent-anima-lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            );
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-anima".into()))
                .with_quant(Quant::Q8)
                .with_adapters(vec![adapter]);
            let e = variant_load(&spec)
                .err()
                .expect("a nonexistent weights dir still errors")
                .to_string();
            assert!(
                !e.contains("sc-10578") && !e.contains("not supported"),
                "packed tier + adapter must NOT be rejected as unsupported, got: {e}"
            );
        }
    }

    // ── Sequential residency (epic 10834, sc-10840): weight-free proof the dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot dir; the discriminator is
    // deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Qwen3 TE from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weights A/B is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: mlx_gen::OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/anima-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        // All three variants share the one dispatch — assert on Base.
        let res = build_residency(
            &missing_snapshot_spec(mlx_gen::OffloadPolicy::Sequential),
            Variant::Base,
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
            &missing_snapshot_spec(mlx_gen::OffloadPolicy::Resident),
            Variant::Base,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        // An eager-load failure (missing split_files / TE file), not a policy/precision rejection.
        assert!(
            !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front precision guard: {msg}"
        );
    }

    #[test]
    fn descriptors_advertise_sequential_offload() {
        // All three anima ids honor the shared Residency seam (the descriptor bit consumers read).
        for d in [
            descriptor_base(),
            descriptor_aesthetic(),
            descriptor_turbo(),
        ] {
            assert!(
                d.capabilities.supports_sequential_offload,
                "{} must advertise supports_sequential_offload",
                d.id
            );
        }
    }
}
