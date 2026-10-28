//! Epic 16774: additive SDXL modes require complete source-owned native certificates over an
//! actually captured text anchor. Dense/q8/q4, both literal load endpoints, whole-run and every
//! measured phase must satisfy `derive_certified` without empirical noise margins. These tiny CPU
//! fixtures establish native-allocator growth only, never process growth or physical-device proof.
//!
//! Each run is the production request lifecycle on the fixture's components: the shared
//! `Residency::run_staged_request_scoped` seam (rung 0: the warm pair; rung 1: text → drop → heavy →
//! shed → decode) over the production loaders, conditioning, encode, denoise and decode functions,
//! a fresh load per run, measured by `mlx_gen::ios_anchor_law::measure_phases`.

use super::*;
use crate::ios_pricing::ios_non_additive_modes;
use mlx_gen::gen_core::{GenerationMemory, MemoryGeometry, MemoryMode, MemorySelection};
use mlx_gen::ios_anchor_law::{area, derive_certified, measure_phases, Derivation, MeasuredRun};
use mlx_gen::ios_pricing::{IosPricing, IosPricingRequest};

/// Output edge of every fixture request (pixels).
const PX: i32 = LATENT * VAE_SCALE;

/// The derived modes and their pricing routes.
fn modes() -> [(MemoryMode, Route, u32); 4] {
    [
        (MemoryMode::ImageToImage, Route::Reference, 1),
        (MemoryMode::Edit, Route::Mask, 1),
        (MemoryMode::Other("control".into()), Route::Control, 0),
        (MemoryMode::Other("ip_adapter".into()), Route::IpAdapter, 1),
    ]
}

/// Endpoint load ownership over the fixture: optimized uses sequential/deferred loading;
/// the fully resident endpoint uses resident/eager loading. A full-cover streamed sub-stack is
/// NOT a fully resident UNet. Both retain the same declared VAE and mode components.
fn spec(s: &Snapshot, route: Route, which: Selection) -> mlx_gen::LoadSpec {
    let overlay = match route {
        Route::Control => "control",
        Route::IpAdapter => "ip_adapter",
        _ => "text",
    };
    super::super::reachable::ios_spec(
        "sdxl",
        &s.root,
        s.bits.map(|b| b as u64),
        &s.root.join("vae_fp16_fix"),
        (
            overlay,
            &s.file("controlnet/diffusion_pytorch_model.safetensors"),
            &s.root.join("ip"),
        ),
    )
    .with_load_shape(match which {
        Selection::Optimized => mlx_gen::LoadShape::DeferredMaterialization,
        Selection::Rung0 => mlx_gen::LoadShape::EagerMaterialization,
    })
    .with_offload_policy(match which {
        Selection::Optimized => mlx_gen::OffloadPolicy::Sequential,
        Selection::Rung0 => mlx_gen::OffloadPolicy::Resident,
    })
}

/// Which selection of the spec's contract a proof runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    /// The deepest `Implemented` rung at the LAST (cheapest) selection of its `selection_space`.
    Optimized,
    /// Rung 0, the resident selection.
    Rung0,
}

fn selection(spec: &mlx_gen::LoadSpec, which: Selection) -> (MemorySelection, GenerationMemory) {
    let (optimized, rung0) = super::super::reachable::selections(spec);
    match which {
        Selection::Optimized => optimized,
        Selection::Rung0 => rung0,
    }
}

/// The request's regime, through the production `Regime::from_request` (the fixture's CLIP window
/// is its 8-token table, not SDXL's 77).
fn regime(
    spec: &mlx_gen::LoadSpec,
    mode: &MemoryMode,
    selection: &MemorySelection,
    memory: &GenerationMemory,
) -> Regime {
    regime_with_count(spec, mode, selection, memory, 1)
}

fn regime_with_count(
    spec: &mlx_gen::LoadSpec,
    mode: &MemoryMode,
    selection: &MemorySelection,
    memory: &GenerationMemory,
    count: u32,
) -> Regime {
    let route = Route::from_mode(mode).unwrap();
    let mut generation = generation_request(route, PX as u32, PX as u32, memory);
    generation.count = count;
    let mut regime = Regime::from_request(&IosPricingRequest {
        provider: "sdxl",
        spec,
        mode,
        geometry: MemoryGeometry {
            width: PX as u32,
            height: PX as u32,
            batch: count,
            frames: 1,
            reference_count: generation.memory_reference_count(),
        },
        selection,
        execution: memory,
        generation: Some(&generation),
        transform: None,
        depth: None,
        token_window: 0,
    })
    .unwrap();
    regime.context_tokens = TOKENS;
    assert_eq!(
        spec.load_shape,
        if regime.staged {
            mlx_gen::LoadShape::DeferredMaterialization
        } else {
            mlx_gen::LoadShape::EagerMaterialization
        },
        "endpoint load ownership must match its declared specification"
    );
    assert_eq!(
        spec.offload_policy,
        if regime.staged {
            mlx_gen::OffloadPolicy::Sequential
        } else {
            mlx_gen::OffloadPolicy::Resident
        },
        "resident endpoint must retain every phase component"
    );
    regime
}

fn endpoint_inputs(s: &Snapshot, regime: &Regime) -> Inputs {
    let mut inputs = s.inputs(regime.route);
    inputs.streamable = regime.staged;
    inputs
}

/// `ios_pricing` of the request on the fixture (its architecture, the CPU kernel transients).
fn pricing(s: &Snapshot, regime: &Regime) -> IosPricing {
    crate::quantum::with_backend(Backend::Cpu, || {
        Priced::build(&endpoint_inputs(s, regime))
            .and_then(|priced| priced.price(regime))
            .unwrap()
            .0
    })
}

/// One request of `regime.route` on `files`: the production lifecycle (see the module docs).
fn run(files: &Files, regime: &Regime, memory: &GenerationMemory) -> Result<()> {
    (
        crate::test_device::record_model_inference_device(),
        run_image(files, regime, memory),
    )
        .1
        .map(|_| ())
}

#[derive(Clone, Default)]
struct StepOwnershipTrace {
    start_active: u64,
    whole_demand: u64,
    schedule: Option<(usize, f32)>,
    current: usize,
    inputs: Vec<(usize, Dtype, Dtype)>,
    outputs: Vec<(usize, Dtype)>,
    init: Option<(Dtype, u64)>,
    fixed_inputs: u64,
    peaks: Vec<(usize, String, u64)>,
    checks: Vec<(usize, String, u64)>,
}

fn checked_native_active_bytes() -> u64 {
    let mut value = 0usize;
    assert_eq!(
        unsafe { mlx_sys::mlx_get_active_memory(&mut value) },
        0,
        "native baseline/active-counter query failed; initialized zero is not evidence"
    );
    value as u64
}

fn checked_native_peak_bytes() -> u64 {
    let mut value = 0usize;
    assert_eq!(
        unsafe { mlx_sys::mlx_get_peak_memory(&mut value) },
        0,
        "native peak-counter query failed; initialized zero is not evidence"
    );
    value as u64
}

fn checked_reset_native_peak() {
    assert_eq!(
        unsafe { mlx_sys::mlx_reset_peak_memory() },
        0,
        "native peak reset failed; an older lifetime peak is not this interval's evidence"
    );
}

struct ObservedSampler<'a> {
    inner: &'a dyn mlx_gen::DiffusionSampler,
    trace: &'a Rc<RefCell<StepOwnershipTrace>>,
}

impl mlx_gen::DiffusionSampler for ObservedSampler<'_> {
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
    fn timestep(&self, i: usize) -> f32 {
        self.inner.timestep(i)
    }
    fn scale_initial_noise(&self, x: &Array) -> Result<Array> {
        self.inner.scale_initial_noise(x)
    }
    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        let scaled = self.inner.scale_model_input(x, i)?;
        let mut trace = self.trace.borrow_mut();
        trace.current = i;
        trace.inputs.push((i, x.dtype(), scaled.dtype()));
        Ok(scaled)
    }
    fn step(&self, eps: &Array, x: &Array, i: usize) -> Result<Array> {
        let out = self.inner.step(eps, x, i)?;
        self.trace.borrow_mut().outputs.push((i, out.dtype()));
        Ok(out)
    }
}

/// The exact same lifecycle, retaining RGB output for the boundary-cast parity proof.
fn run_image(
    files: &Files,
    regime: &Regime,
    memory: &GenerationMemory,
) -> Result<mlx_gen::media::Image> {
    (
        crate::test_device::record_model_inference_device(),
        run_image_traced(files, regime, memory, None),
    )
        .1
}

fn run_image_traced(
    files: &Files,
    regime: &Regime,
    memory: &GenerationMemory,
    trace: Option<&Rc<RefCell<StepOwnershipTrace>>>,
) -> Result<mlx_gen::media::Image> {
    use crate::text_encoder::ClipTextEncoder;
    // Match production: typed scope setup precedes even RNG initialization/Array construction.
    let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
    mlx_rs::random::seed(16774)?;
    let route = regime.route;
    let (text_files, mut heavy_files) = (files.clone(), files.clone());
    let streamed = regime.staged;
    heavy_files.streamed = streamed;
    let residency = mlx_gen::Residency::<
        (ClipTextEncoder, ClipTextEncoder),
        crate::model::SdxlHeavyOwned,
    >::request_scoped(
        move |_| text_files.text(),
        move |_, _| {
            let (unet, controls, vae, ip) = heavy_files.heavy(route)?;
            assert_eq!(
                unet.can_stream_blocks(),
                streamed,
                "resident endpoint must own the full UNet, not an armed full-cover stream"
            );
            Ok(crate::model::SdxlHeavyOwned::with_controls_for_test(
                unet, controls, vae, ip,
            ))
        },
    );
    let staged = memory.stage_residency;
    let cancel = mlx_gen::CancelFlag::default();
    let window = memory
        .transformer_window_size
        .map(|size| crate::plan::SdxlBlockWindow {
            size: size as usize,
            cancel: &cancel,
        });
    let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
        .with_window(window)
        .with_cfg_batching(regime.cfg_batching);
    let tiling = regime.tiling;
    residency.run_staged_request_scoped(
        staged,
        false,
        &cancel,
        false,
        &mut |_| {},
        |text: &(ClipTextEncoder, ClipTextEncoder)| {
            use mlx_rs::ops::indexing::IndexOp;
            let ids = ids();
            let ids = if regime.sampling.guidance() > 1.0 {
                ids
            } else {
                ids.index((0..1, ..))
            };
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::encode_conditioning(&text.0, &text.1, &ids),
            )
                .1
        },
        |enc| match enc {
            Some(enc) => Ok(mlx_rs::transforms::eval([&enc.0, &enc.1])?),
            None => Ok(()),
        },
        |heavy: &crate::model::SdxlHeavyOwned, (cond, pooled), _| {
            crate::model::collect_serial_latents(regime.count, 16774, |seed, finals| {
                let latents = (
                    crate::test_device::record_model_inference_device(),
                    denoise(
                        heavy,
                        route,
                        &regime.sampling,
                        staged,
                        plan,
                        tiling.as_ref(),
                        &cond,
                        &pooled,
                        trace,
                        seed,
                    ),
                )
                    .1?;
                mlx_gen::sampler::diagnostic_sampling_tensor(
                    "sdxl",
                    "serial-final-latent",
                    finals.len() as u64,
                    1,
                    &latents,
                );
                finals.push(latents);
                Ok(())
            })
        },
        |latents| Ok(mlx_rs::transforms::eval(latents.iter())?),
        |view, latents, _| {
            let mut images = Vec::with_capacity(latents.len());
            for latent in &latents {
                images.push(
                    (
                        crate::test_device::record_model_inference_device(),
                        crate::pipeline::decode_image_tiled(
                            view.vae_for_test(),
                            latent,
                            None,
                            tiling.as_ref(),
                            None,
                        ),
                    )
                        .1?,
                );
            }
            Ok(images.remove(0))
        },
    )
}

/// The heavy phase of one request, in `Sdxl::generate`'s order: the ControlNet conditioning
/// embedding, the IP image prompt (shed on a staged request), the init-latent encode, then a
/// guided ancestral denoise. The optional metadata observer never retains Arrays or evaluates them.
#[allow(clippy::too_many_arguments)]
fn denoise(
    heavy: &crate::model::SdxlHeavyOwned,
    route: Route,
    sampling: &crate::sampling::ResolvedSampling,
    staged: bool,
    plan: crate::plan::SdxlForwardPlan<'_>,
    tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cond: &Array,
    pooled: &Array,
    trace: Option<&Rc<RefCell<StepOwnershipTrace>>>,
    seed: u64,
) -> Result<Array> {
    let (unet, controls, vae) = heavy.parts_for_test();
    let cfg = sampling.guidance();
    let rows = if cfg > 1.0 { 2 } else { 1 };
    let ids = crate::pipeline::text_time_ids(rows);
    let cancel = mlx_gen::CancelFlag::default();
    let control = match route {
        Route::Control => {
            let edge = LATENT * CONTROL_SCALE;
            let image = crate::pipeline::preprocess_control_image(
                &image(edge, edge, 7),
                edge as u32,
                edge as u32,
            )?;
            let image = if rows == 2 {
                mlx_rs::ops::concatenate_axis(&[&image, &image], 0)?
            } else {
                image
            };
            Some(crate::pipeline::ControlContext {
                controlnet: &controls[0],
                cond_embed: controls[0].embed_cond(&image)?,
                scale: 1.0,
            })
        }
        _ => None,
    };
    let ip_tokens = match route {
        Route::IpAdapter => Some(heavy.ip_tokens_for_test(
            &image(20, 18, 1),
            cfg > 1.0,
            crate::model::sheds_ip_image_encoder(staged),
        )?),
        _ => None,
    };
    let euler = crate::sampler::EulerSampler::new_with_dtype(
        &crate::config::DiffusionConfig::sdxl_base(),
        true,
        crate::model::DTYPE,
    )?;
    let true_init = matches!(route, Route::Reference | Route::Mask);
    let config = crate::config::DiffusionConfig::sdxl_base();
    let alpha = mlx_gen::AlphaSchedule::scaled_linear(
        config.num_train_steps,
        config.beta_start,
        config.beta_end,
    );
    if sampling.route().is_curated() {
        let init = if true_init {
            Some(
                (
                    crate::test_device::record_model_inference_device(),
                    crate::pipeline::encode_init_latents_tiled(
                        vae,
                        &image(PX, PX, 3),
                        PX as u32,
                        PX as u32,
                        tiling,
                        None,
                    ),
                )
                    .1?,
            )
        } else {
            None
        };
        let sigmas =
            sampling.curated_sigmas(&alpha, sampling.effective_init_strength(true_init, false))?;
        let noise = mlx_rs::random::normal::<f32>(&[1, LATENT, LATENT, 4], None, None, None)?;
        let latents = match &init {
            Some(init) => mlx_rs::ops::add(
                init,
                mlx_rs::ops::multiply(&noise, mlx_gen::array::scalar(sigmas[0]))?,
            )?,
            None => mlx_rs::ops::multiply(&noise, mlx_gen::array::scalar(sigmas[0]))?,
        };
        let ms = mlx_gen::DiscreteModelSampling::sdxl(&alpha);
        let ip = ip_tokens.as_ref().map(|tokens| (tokens, 0.6));
        let control = control.as_slice();
        let preview = mlx_gen::PreviewSink::default();
        return if matches!(sampling.route(), crate::sampling::SamplingRoute::CfgPp(_)) {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise_cfgpp_with_preview(
                    unet,
                    Some(sampling.sampler_name()),
                    &ms,
                    &sigmas,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    &cancel,
                    &mut |_| {},
                    &preview,
                    control,
                    ip,
                    None,
                    plan,
                ),
            )
                .1
        } else {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise_curated_with_preview(
                    unet,
                    Some(sampling.sampler_name()),
                    &ms,
                    &sigmas,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    seed,
                    &cancel,
                    &mut |_| {},
                    &preview,
                    control,
                    ip,
                    None,
                    plan,
                ),
            )
                .1
        };
    }
    let acceleration = if sampling.route().is_acceleration() {
        Some(sampling.build_acceleration_sampler(&alpha, seed)?)
    } else {
        None
    };
    let (steps, start_time) = if acceleration.is_some() {
        (sampling.steps(), euler.max_time())
    } else {
        match trace.and_then(|t| t.borrow().schedule) {
            Some((requested, strength)) if true_init => {
                crate::model::ancestral_strength_schedule(requested, euler.max_time(), strength)
            }
            Some((requested, _)) => (requested, euler.max_time()),
            None => sampling.ancestral_window(
                euler.max_time(),
                sampling.effective_init_strength(true_init, route == Route::Mask),
            )?,
        }
    };
    // `Sdxl::generate` keeps the encoded init latent for the whole denoise (seed-independent).
    let mut init_latent = None;
    let mut blend = None;
    let latents = if let Some(sampler) = &acceleration {
        let noise = mlx_rs::random::normal::<f32>(&[1, LATENT, LATENT, 4], None, None, None)?;
        sampler.scale_initial_noise(&noise)?
    } else if matches!(route, Route::Reference | Route::Mask) {
        let init = (
            crate::test_device::record_model_inference_device(),
            crate::pipeline::encode_init_latents_tiled(
                vae,
                &image(PX, PX, 3),
                PX as u32,
                PX as u32,
                tiling,
                None,
            ),
        )
            .1?;
        let noise = rnd(init.shape(), 5, 1.0);
        let x_t = euler.add_noise_with(&init, &noise, start_time)?;
        if route == Route::Mask {
            let edge = LATENT * 8;
            let mask =
                crate::inpaint::preprocess_mask(&image(edge, edge, 9), edge as u32, edge as u32)?;
            let t_prev = euler
                .timesteps(steps, start_time)?
                .into_iter()
                .map(|(_, t)| t)
                .collect();
            if let Some(trace) = trace {
                trace.borrow_mut().fixed_inputs =
                    crate::quantum::bytes(&noise) + crate::quantum::bytes(&mask);
            }
            blend = Some(crate::inpaint::InpaintBlend::new(
                &euler,
                mask,
                init.clone(),
                noise,
                t_prev,
            ));
        }
        init_latent = Some(init);
        x_t
    } else {
        euler.sample_prior(&[1, LATENT, LATENT, 4])?
    };
    if let Some(trace) = trace {
        let mut trace = trace.borrow_mut();
        trace.init = init_latent
            .as_ref()
            .map(|x| (x.dtype(), crate::quantum::bytes(x)));
        let init_bytes = trace.init.map_or(0, |(_, bytes)| bytes);
        trace.fixed_inputs += init_bytes;
    }
    let ancestral = if acceleration.is_none() {
        Some(crate::sampler::AncestralEuler::new(
            &euler, steps, start_time,
        )?)
    } else {
        None
    };
    let sampler: &dyn mlx_gen::DiffusionSampler = match acceleration.as_deref() {
        Some(sampler) => sampler,
        None => ancestral.as_ref().expect("actual ancestral sampler"),
    };
    let observed = trace.map(|trace| ObservedSampler {
        inner: sampler,
        trace,
    });
    let driver: &dyn mlx_gen::DiffusionSampler = match &observed {
        Some(observed) => observed,
        None => sampler,
    };
    let d = crate::pipeline::Denoiser::with_plan(unet, driver, plan);
    let denoised = match route {
        Route::Control => {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise_control(
                    &d,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    &cancel,
                    &mut |_| {},
                    control.as_ref().expect("control context"),
                ),
            )
                .1
        }
        Route::IpAdapter => {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise_ip(
                    &d,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    &cancel,
                    &mut |_| {},
                    ip_tokens.as_ref().expect("IP tokens"),
                    0.6,
                ),
            )
                .1
        }
        Route::Mask => {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise_inpaint(
                    &d,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    &cancel,
                    &mut |_| {},
                    blend.as_ref().expect("inpaint blend"),
                ),
            )
                .1
        }
        _ => {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise(
                    &d,
                    latents,
                    cond,
                    pooled,
                    &ids,
                    cfg,
                    &cancel,
                    &mut |_| {},
                ),
            )
                .1
        }
    };
    drop(init_latent);
    denoised
}

/// Repeated actual native captures per source-profile fixture.
const REPEATS: usize = 4;

/// Production source pricing, with observations added ONLY by actual loaded-fixture execution.
fn captured_source_profile(
    snapshot: &Snapshot,
    regime: &Regime,
    memory: &GenerationMemory,
) -> (MeasuredRun, IosPricing) {
    let mut inputs = endpoint_inputs(snapshot, regime);
    let mut profile = crate::ios_pricing::source_pricing_for_test(&mut inputs, regime).unwrap();
    let files = Files::of(snapshot);
    with_cpu_diagnostic_headroom(PLENTY, || run(&files, regime, memory)).unwrap();
    let mut worst = MeasuredRun {
        demand: 0,
        phases: Default::default(),
    };
    for _ in 0..REPEATS {
        let mut baseline = None;
        let (result, mut observation) = observe_actual_marks(|| {
            with_cpu_diagnostic_headroom(PLENTY, || {
                measure_phases(ios_quantum_phase, || {
                    baseline = Some(checked_native_active_bytes());
                    run(&files, regime, memory)
                })
            })
        });
        let (_, measured) = result.unwrap();
        let baseline = baseline.expect("actual pre-load native baseline");
        assert!(measured.demand > 0 && !measured.phases.is_empty());
        assert!(!observation.checks.is_empty() && !observation.metadata.is_empty());
        observation.baseline_bytes = baseline;
        profile.observations.push(observation);
        profile.request_baseline = Some(mlx_gen::ios_pricing::IosMemoryBaseline {
            domain: mlx_gen::ios_pricing::IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
            bytes: baseline,
            source: "checked mlx_get_active_memory before native fixture load/generate".into(),
        });
        worst.demand = worst.demand.max(measured.demand);
        for (phase, bytes) in measured.phases {
            let slot = worst.phases.entry(phase).or_insert(0);
            *slot = (*slot).max(bytes);
        }
    }
    (worst, profile)
}

/// No spread, empirical margin or missing-phase default may conceal an underprediction.
fn assert_certified_native_run(measured: &MeasuredRun, derived: &Derivation, label: &str) {
    assert_eq!(
        derived.domain,
        mlx_gen::ios_pricing::IosMemoryDomain::NativeAllocatorGrowth
    );
    assert_eq!(
        derived.coordinate,
        mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad
    );
    assert!(
        measured.demand <= derived.demand,
        "{label}: whole-run native growth {} exceeds certified {}",
        measured.demand,
        derived.demand
    );
    assert_eq!(derived.phases.len(), IosPhase::ALL.len());
    for (&phase, &bytes) in &measured.phases {
        let upper = derived
            .phases
            .iter()
            .find(|candidate| candidate.phase == phase)
            .expect("every measured phase requires a certified bound");
        assert!(
            bytes <= upper.bytes,
            "{label}: {phase:?} native growth {bytes} exceeds certified {}",
            upper.bytes
        );
    }
}

fn run_all_once(
    files: &Files,
    text: &Regime,
    text_memory: &GenerationMemory,
    requests: &[(MemoryMode, Regime, GenerationMemory)],
) {
    with_cpu_diagnostic_headroom(PLENTY, || {
        run(files, text, text_memory)?;
        for (_, r, memory) in requests {
            run(files, r, memory)?;
        }
        Ok(())
    })
    .unwrap();
    mlx_gen::residency::drain_allocator_cache();
}

/// Record the SAME marks the production capture listener consumes. This fixture collector owns
/// only scalar metadata and check references, never Arrays or planned/replayed observations.
fn observe_actual_marks<T>(
    body: impl FnOnce() -> Result<T>,
) -> (Result<T>, mlx_gen::ios_pricing::IosAnchorObservation) {
    use mlx_gen::ios_pricing::{
        IosAnchorObservation, IosMemoryDomain, IosObservedCheck, IosObservedMetadata,
    };
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
    let recorded = std::sync::Arc::new(std::sync::Mutex::new((
        IosAnchorObservation {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
            ..Default::default()
        },
        std::collections::BTreeMap::<(String, String), u32>::new(),
    )));
    let sink = std::sync::Arc::clone(&recorded);
    struct RemoveListener;
    impl Drop for RemoveListener {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    let previous = set_quantum_listener(Some(Box::new(move |mark| {
        let mut recorded = sink.lock().expect("actual mark collector lock");
        let (observation, latest) = &mut *recorded;
        match mark {
            QuantumMark::Metadata {
                provider,
                kind,
                fields,
            } => {
                let index = u32::try_from(observation.metadata.len())
                    .expect("bounded native fixture metadata");
                observation.metadata.push(IosObservedMetadata {
                    provider: provider.into(),
                    kind: kind.into(),
                    fields: fields
                        .iter()
                        .map(|&(key, value)| (key.into(), value))
                        .collect(),
                });
                latest.insert((provider.into(), kind.into()), index);
            }
            QuantumMark::Checked { label, .. } => observation.checks.push(IosObservedCheck {
                phase: ios_quantum_phase(label),
                label: label.into(),
                context: latest.values().copied().collect(),
            }),
            QuantumMark::Synchronized { .. } => {}
        }
    })));
    let guard = RemoveListener;
    assert!(
        !previous,
        "isolated native child must not replace another capture listener"
    );
    let result = body();
    drop(guard);
    let observation = recorded
        .lock()
        .expect("actual mark collector lock")
        .0
        .clone();
    // Full process/phase telemetry is not claimed by this native-only single-window fixture.
    (result, observation)
}

/// Actual Control merge/cast barriers and IP token owners matched to their coincident LOGICAL
/// ledger. No measured/process subtraction, fresh-backing certificate or empirical offset is used.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn control_ip_coincident_logical_windows_follow_actual_barriers() {
    if !cpu_child(
        "pricing::additivity::control_ip_coincident_logical_windows_follow_actual_barriers",
    ) {
        return;
    }
    fn group<'a>(
        observation: &'a mlx_gen::ios_pricing::IosAnchorObservation,
        kind: &str,
    ) -> Vec<&'a mlx_gen::ios_pricing::IosObservedMetadata> {
        let first = observation
            .metadata
            .iter()
            .position(|m| m.provider == "sdxl" && m.kind == kind)
            .expect("actual source metadata missing");
        observation.metadata[first..]
            .iter()
            .take_while(|m| m.provider == "sdxl" && m.kind == kind)
            .collect()
    }
    let snap = Snapshot::new(2);
    for batching in [CfgBatching::Sequential, CfgBatching::Batched] {
        let mut sp = spec(&snap, Route::Control, Selection::Rung0);
        sp.extra_controls.push(mlx_gen::WeightsSource::File(
            snap.file("controlnet/diffusion_pytorch_model.safetensors"),
        ));
        let (sel, mut memory) = selection(&sp, Selection::Rung0);
        memory.cfg_batching = Some(batching);
        let mut generation = generation_request(Route::Control, PX as u32, PX as u32, &memory);
        generation
            .conditioning
            .push(mlx_gen::Conditioning::Control {
                image: mlx_gen::Image::default(),
                kind: mlx_gen::gen_core::ControlKind::Canny,
                scale: Some(0.0),
            });
        let mut r = Regime::from_request(&IosPricingRequest {
            provider: "sdxl",
            spec: &sp,
            mode: &MemoryMode::Other("control".into()),
            geometry: MemoryGeometry {
                width: PX as u32,
                height: PX as u32,
                batch: 1,
                frames: 1,
                reference_count: generation.conditioning.len() as u32,
            },
            selection: &sel,
            execution: &memory,
            generation: Some(&generation),
            transform: None,
            depth: None,
            token_window: 0,
        })
        .unwrap();
        r.context_tokens = TOKENS;
        let mut inputs = snap.inputs(Route::Control);
        inputs
            .controls
            .push(snap.file("controlnet/diffusion_pytorch_model.safetensors"));
        inputs.streamable = false;
        let profile = crate::quantum::with_backend(Backend::Cpu, || {
            Priced::build(&inputs).and_then(|p| p.price(&r)).unwrap().0
        });
        let (result, observation) = observe_actual_marks(|| {
            with_cpu_diagnostic_headroom(PLENTY, || {
                let mut files = Files::of(&snap);
                files.streamed = false;
                let text = files.text()?;
                let (conditioning, pooled) = (
                    crate::test_device::record_model_inference_device(),
                    crate::pipeline::encode_conditioning(&text.0, &text.1, &ids()),
                )
                    .1?;
                let (unet, controls, _vae, _ip) = files.heavy(Route::Control)?;
                let extra = crate::loader::load_controlnet_with(
                    &mlx_gen::WeightsSource::File(
                        snap.file("controlnet/diffusion_pytorch_model.safetensors"),
                    ),
                    crate::model::DTYPE,
                    &snap.unet,
                )?;
                let edge = LATENT * CONTROL_SCALE;
                let image = crate::pipeline::preprocess_control_image(
                    &image(edge, edge, 47),
                    edge as u32,
                    edge as u32,
                )?;
                let pair = mlx_rs::ops::concatenate_axis(&[&image, &image], 0)?;
                let contexts = vec![
                    crate::pipeline::ControlContext {
                        controlnet: &controls[0],
                        cond_embed: controls[0].embed_cond(&pair)?,
                        scale: 1.0,
                    },
                    crate::pipeline::ControlContext {
                        controlnet: &extra,
                        cond_embed: extra.embed_cond(&pair)?,
                        scale: 0.0,
                    },
                ];
                let state =
                    rnd(&[1, LATENT, LATENT, 4], 53, 1.0).as_dtype(r.initial_state_dtype())?;
                let time_ids = crate::pipeline::text_time_ids(conditioning.shape()[0]);
                let checked = Rc::new(RefCell::new(Vec::new()));
                let sink = Rc::clone(&checked);
                with_cpu_diagnostic_observer(
                    move |event| {
                        if let QuantumEvent::Checked { label, scratch } = event {
                            if ios_quantum_phase(label) == Some(IosPhase::Denoise) {
                                sink.borrow_mut().push((label.clone(), *scratch));
                            }
                        }
                    },
                    || {
                        (
                            crate::test_device::record_model_inference_device(),
                            crate::pipeline::guided_eps_pair(
                                &unet,
                                &state,
                                1.0,
                                &conditioning,
                                &pooled,
                                &time_ids,
                                &contexts,
                                &conditioning,
                                None,
                                crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(batching),
                            ),
                        )
                            .1?;
                        Ok(())
                    },
                )?;
                let output = checked.borrow().clone();
                Ok(output)
            })
        });
        let checked = result.unwrap();
        let expected: Vec<_> = profile
            .windows
            .iter()
            .filter(|w| w.phase == IosPhase::Denoise)
            .take_while(|w| w.label != crate::quantum::SAMPLER_STEP)
            .map(|w| (w.label.clone(), w.scratch_bytes))
            .collect();
        assert_eq!(
            checked, expected,
            "{batching:?}: actual Control guard workspace must match corresponding raw window"
        );
        assert!(
            profile
                .windows
                .iter()
                .filter(|w| w.phase == IosPhase::Denoise)
                .all(|w| w.class.is_none()),
            "{batching:?}: Control raw windows must remain unclassified"
        );
        assert!(
            profile.phase(IosPhase::Encode).is_some()
                && !profile.absent_phases.contains_key(&IosPhase::Encode),
            "{batching:?}: active Control encode cannot disappear"
        );
        let stems: Vec<_> = profile
            .windows
            .iter()
            .filter(|w| w.phase == IosPhase::Denoise && w.label == crate::quantum::CONTROL_STEM)
            .take(2)
            .collect();
        let returned = group(&observation, "control-branch-return")
            .iter()
            .map(|m| m.fields["logical_bytes"])
            .sum::<u64>();
        let selected = if batching.is_batched() {
            0
        } else {
            group(&observation, "control-selected-row-materialized")[0].fields
                ["available_logical_holder_bytes"]
        };
        assert_eq!(
            stems[1]
                .held_bytes
                .checked_sub(stems[0].held_bytes)
                .unwrap(),
            returned + selected,
            "{batching:?}: future final residuals must not be projected onto the first branch"
        );
        let pending = group(&observation, "control-merge-pending");
        let expected_down = pending
            .iter()
            .filter(|m| m.fields["mid"] == 0)
            .map(|m| mlx_gen::ios_quantum::allocated_bytes(m.fields["logical_bytes"]))
            .sum::<u64>();
        let expected_mid = pending
            .iter()
            .filter(|m| m.fields["mid"] == 1)
            .map(|m| mlx_gen::ios_quantum::allocated_bytes(m.fields["logical_bytes"]))
            .sum::<u64>();
        let work = group(&observation, "control-pending-merge-work")[0];
        assert_eq!(
            (
                work.fields["down_logical_upper_bytes"],
                work.fields["mid_logical_upper_bytes"]
            ),
            (expected_down, expected_mid),
            "{batching:?}: actual pending add outputs must be counted at their consuming barriers"
        );
        assert!(
            group(&observation, "control-cast-pending")
                .iter()
                .all(|m| m.fields["available"] == 0),
            "{batching:?}: cast construction is not materialization"
        );
        assert!(
            group(&observation, "control-after-down-injection")
                .iter()
                .all(|m| m.fields["available"] == u64::from(m.fields["mid"] == 0)),
            "{batching:?}: down barrier must not claim the future mid payload"
        );
        assert!(
            group(&observation, "control-after-mid-injection")
                .iter()
                .all(|m| m.fields["available"] == 1),
            "{batching:?}: mid barrier must record actual completed residual materialization"
        );
    }
    let sp = spec(&snap, Route::IpAdapter, Selection::Rung0);
    let (sel, memory) = selection(&sp, Selection::Rung0);
    let ip = regime(&sp, &MemoryMode::Other("ip_adapter".into()), &sel, &memory);
    let text = regime(&sp, &MemoryMode::TextToImage, &sel, &memory);
    let mut inputs = snap.inputs(Route::IpAdapter);
    inputs.streamable = false;
    let (ip_profile, text_profile) = crate::quantum::with_backend(Backend::Cpu, || {
        let p = Priced::build(&inputs).unwrap();
        (p.price(&ip).unwrap().0, p.price(&text).unwrap().0)
    });
    let (result, observation) = observe_actual_marks(|| {
        with_cpu_diagnostic_headroom(PLENTY, || {
            (
                crate::test_device::record_model_inference_device(),
                run_image(&Files::of(&snap), &ip, &memory),
            )
                .1
        })
    });
    result.unwrap();
    assert!(
        observation
            .checks
            .iter()
            .any(|c| c.label == crate::quantum::IP_CFG_TOKENS
                && c.phase == Some(IosPhase::Conditioning)),
        "IP token work belongs to Conditioning, not Encode"
    );
    assert!(
        ip_profile.absent_phases.contains_key(&IosPhase::Encode)
            && ip_profile
                .windows
                .iter()
                .filter(|w| w.phase == IosPhase::Denoise)
                .all(|w| w.class.is_none()),
        "IP raw windows remain required/unclassified without inventing an init encode"
    );
    let token_owner = group(&observation, "ip-persistent-token-owner")[0];
    let token_bytes = token_owner.fields["available_logical_holder_bytes"];
    assert!(
        token_bytes > 0 && token_owner.fields["pending_logical_holder_bytes"] == 0,
        "IP token owner must be observed after its actual conditioning barrier"
    );
    let stem = |p: &IosPricing| {
        p.windows
            .iter()
            .find(|w| w.phase == IosPhase::Denoise && w.label == crate::quantum::UNET_STEM)
            .unwrap()
            .held_bytes
    };
    // Pure logical ledger inclusion with IDENTICAL configured component ownership. This is not a
    // measured body cancellation, native lower bound, or ProcessGrowth subtraction certificate.
    assert_eq!(
        stem(&ip_profile).checked_sub(stem(&text_profile)).unwrap(),
        token_bytes,
        "persistent IP tokens must not vanish from forward ownership"
    );

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::control_ip_coincident_logical_windows_follow_actual_barriers");
}

/// Exercise every served factory through the existing loaded tiny-model lifecycle, not a
/// metadata replay or a sigma-only substitute. This checks logical owner/guard bindings only:
/// allocator/process conversion, physical disjointness and anchor transfer remain unproved.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn served_sampler_factories_bind_actual_model_callbacks_and_owners() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::served_sampler_factories_bind_actual_model_callbacks_and_owners",
    ) else {
        return;
    };
    use crate::sampling::{Evaluation, SamplingRoute};
    use mlx_gen::ios_pricing::IosObservedMetadata;
    let dense = Snapshot::new(1);
    let q8 = Snapshot::packed(8);
    let q4 = Snapshot::packed(4);
    // Independent actual native model-output dtype for each loaded tier. No universal F16 epsilon
    // assumption: packed projections can promote their actual output even with a compute-F16 input.
    let epsilon_dtype = |snapshot: &Snapshot| {
        with_cpu_diagnostic_headroom(PLENTY, || {
            use mlx_rs::ops::indexing::IndexOp;
            let mut files = Files::of(snapshot);
            files.streamed = false;
            let (unet, _, _, _) = files.heavy(Route::Text)?;
            let (te1, te2) = files.text()?;
            let (cond, pooled) = (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::encode_conditioning(&te1, &te2, &ids().index((0..1, ..))),
            )
                .1?;
            let x = rnd(&[1, LATENT, LATENT, 4], 7, 1.0).as_dtype(crate::model::DTYPE)?;
            let eps = (
                crate::test_device::record_model_inference_device(),
                unet.forward_planned(
                    &x,
                    1.0,
                    &cond,
                    &pooled,
                    &crate::pipeline::text_time_ids(1),
                    None,
                    None,
                    crate::plan::SdxlForwardPlan::UNBOUNDED,
                ),
            )
                .1?;
            eps.eval()?;
            Ok(mlx_gen::sampler::sampling_dtype_code(eps.dtype()))
        })
        .unwrap()
    };
    let epsilons = [
        epsilon_dtype(&dense),
        epsilon_dtype(&q8),
        epsilon_dtype(&q4),
    ];
    // Mechanism representatives, NOT a sampler×scheduler×CFG×tier matrix. Extra cases exercise
    // CFG++'s two returned predictions, nondefault ancestral dispatch, packed epsilon promotion,
    // init/mask and the actual shared serial owner. Multi-output overlay/init hoisting is NOT
    // established by this tiny lifecycle (those cases deliberately keep count=1).
    let mut cases = crate::model::descriptor()
        .capabilities
        .samplers
        .iter()
        .map(|&sampler| {
            let accel = matches!(sampler, "lcm" | "lightning" | "hyper");
            (
                sampler,
                "discrete",
                "cfg",
                if accel { 1.0 } else { 5.0 },
                CfgBatching::Sequential,
                Route::Text,
                0usize,
                1u32,
            )
        })
        .collect::<Vec<_>>();
    cases.extend([
        (
            "euler",
            "discrete",
            "cfg_pp",
            5.0,
            CfgBatching::Sequential,
            Route::Text,
            0,
            1,
        ),
        (
            "ddim",
            "discrete",
            "cfg_pp",
            5.0,
            CfgBatching::Batched,
            Route::Text,
            0,
            1,
        ),
        (
            "dpmpp_2m",
            "discrete",
            "cfg_pp",
            5.0,
            CfgBatching::Sequential,
            Route::Text,
            0,
            1,
        ),
        (
            "euler_ancestral",
            "karras",
            "cfg",
            5.0,
            CfgBatching::Batched,
            Route::Text,
            0,
            1,
        ),
        (
            "euler",
            "discrete",
            "cfg",
            1.0,
            CfgBatching::Batched,
            Route::Text,
            0,
            2,
        ),
        (
            "dpmpp_2m",
            "discrete",
            "cfg",
            5.0,
            CfgBatching::Sequential,
            Route::Reference,
            1,
            1,
        ),
        (
            "hyper",
            "discrete",
            "cfg",
            5.0,
            CfgBatching::Batched,
            Route::Text,
            2,
            1,
        ),
        (
            "euler_ancestral",
            "discrete",
            "cfg",
            5.0,
            CfgBatching::Batched,
            Route::Mask,
            1,
            1,
        ),
    ]);
    let snapshots = [&dense, &q8, &q4];
    let config = crate::config::DiffusionConfig::sdxl_base();
    let alpha = mlx_gen::AlphaSchedule::scaled_linear(
        config.num_train_steps,
        config.beta_start,
        config.beta_end,
    );
    for (sampler, scheduler, method, guidance, batching, route, tier, count) in cases {
        let snapshot = snapshots[tier];
        let sp = spec(snapshot, route, Selection::Rung0);
        let (sel, mut memory) = selection(&sp, Selection::Rung0);
        memory.cfg_batching = Some(batching);
        let mode = match route {
            Route::Reference => MemoryMode::ImageToImage,
            Route::Mask => MemoryMode::Edit,
            _ => MemoryMode::TextToImage,
        };
        let mut r = regime(&sp, &mode, &sel, &memory);
        let mut request = generation_request(route, PX as u32, PX as u32, &memory);
        request.sampler = Some(sampler.into());
        request.scheduler = Some(scheduler.into());
        request.guidance_method = Some(method.into());
        request.guidance = Some(guidance);
        request.steps = Some(3);
        request.strength = Some(1.0);
        request.count = count;
        r.sampling = crate::sampling::ResolvedSampling::resolve(&request).unwrap();
        r.count = count;
        let events = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&events);
        let trace = Rc::new(RefCell::new(StepOwnershipTrace::default()));
        let (result, observation) = observe_actual_marks(|| {
            with_cpu_diagnostic_headroom(PLENTY, || {
                with_cpu_diagnostic_observer(
                    move |event| sink.borrow_mut().push(event.clone()),
                    || {
                        (
                            crate::test_device::record_model_inference_device(),
                            run_image_traced(&Files::of(snapshot), &r, &memory, Some(&trace)),
                        )
                            .1
                    },
                )
            })
        });
        result.unwrap();
        // Source-bound standalone floors: these exact loaders call before(), not finish().
        // Resolve the same real fixture files and header producers after capture, not a guessed
        // floor constant. All load records stay in the complete event/metadata capture.
        let component_floor = |subdir, stem, dtype| {
            let file =
                crate::loader::resolve_weight_file(&snapshot.root, subdir, stem, dtype).unwrap();
            crate::loader::cast_load_floor_bytes(&[file], dtype, true, |_| true).unwrap()
        };
        let unet_file = crate::loader::resolve_weight_file(
            &snapshot.root,
            "unet",
            "diffusion_pytorch_model",
            crate::model::DTYPE,
        )
        .unwrap();
        let expected_floors = [
            (
                TEXT_ENCODER_1_LOAD,
                component_floor("text_encoder", "model", crate::model::DTYPE),
            ),
            (
                TEXT_ENCODER_2_LOAD,
                component_floor("text_encoder_2", "model", crate::model::DTYPE),
            ),
            (
                UNET_LOAD,
                crate::loader::unet_load_parts(
                    &[unet_file],
                    crate::model::DTYPE,
                    true,
                    false,
                    &snapshot.unet,
                )
                .unwrap()
                .1,
            ),
            (
                VAE_FP16_FIX_LOAD,
                component_floor(
                    "vae_fp16_fix",
                    "diffusion_pytorch_model",
                    crate::loader::VAE_FP16_FIX_DTYPE,
                ),
            ),
        ];
        let all_events = norm(&events.borrow());
        let floors = all_events
            .iter()
            .filter_map(|event| match event {
                Ev::Checked(label, bytes)
                    if expected_floors
                        .iter()
                        .any(|(floor, _)| label.as_str() == *floor) =>
                {
                    Some((label.as_str(), *bytes))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let floor_syncs = all_events.iter().filter(|event| matches!(event,Ev::Synced(label) if expected_floors.iter().any(|(floor,_)| label.as_str()==*floor))).count();
        assert_eq!((floors,floor_syncs),(expected_floors.to_vec(),0),"served {sampler} actual standalone floor checks match source-resolved files/header producers");
        // Paired-order claim is explicitly Conditioning/Denoise/Decode only. Other Load
        // materialization (including check-only LOAD_QUANTIZE) is NOT certified by this assertion.
        let paired = norm(&events.borrow())
            .into_iter()
            .filter(|event| {
                let label = match event {
                    Ev::Checked(label, _) | Ev::Synced(label) => label,
                };
                matches!(
                    ios_quantum_phase(label),
                    Some(IosPhase::Conditioning | IosPhase::Denoise | IosPhase::Decode)
                )
            })
            .collect::<Vec<_>>();
        assert_well_ordered(&paired);
        let marks = |provider: &str, kind: &str| {
            observation
                .metadata
                .iter()
                .filter(|mark| mark.provider == provider && mark.kind == kind)
                .collect::<Vec<_>>()
        };
        let field = |mark: &IosObservedMetadata, key: &str| *mark.fields.get(key).unwrap();
        let forwards = marks("sdxl", "forward-input-before-cast");
        let rows = if guidance > 1.0 && batching.is_batched() {
            2
        } else {
            1
        };
        assert!(
            !forwards.is_empty() && forwards.iter().all(|m| field(m, "rows") == rows),
            "served {sampler}/{method} actual CFG forward rows"
        );
        let starts = marks("sdxl", "serial-output-start");
        let finals = marks("sdxl", "serial-final-latent");
        let mut completed = 0;
        let expected_starts = finals
            .iter()
            .enumerate()
            .map(|(i, mark)| {
                let row = (i as u64, i as u64, completed);
                completed += field(mark, "logical_bytes");
                row
            })
            .collect::<Vec<_>>();
        assert_eq!(
            starts
                .iter()
                .map(|m| (
                    field(m, "output"),
                    field(m, "completed_latents"),
                    field(m, "completed_latent_bytes")
                ))
                .collect::<Vec<_>>(),
            expected_starts,
            "served {sampler} actual serial owners"
        );
        if r.sampling.route().is_curated() {
            let sigmas = r
                .sampling
                .curated_sigmas(
                    &alpha,
                    r.sampling.effective_init_strength(r.has_init, false),
                )
                .unwrap();
            let epochs = r.sampling.logical_epochs(&sigmas).unwrap();
            let (solver, cfgpp) = match r.sampling.route() {
                SamplingRoute::Curated(solver) => (solver, false),
                // The actual CFG++ factory maps BOTH Euler and DDIM to EulerCfgPp.
                SamplingRoute::CfgPp(mlx_gen::Solver::Ddim) => (mlx_gen::Solver::Euler, true),
                SamplingRoute::CfgPp(solver) => (solver, true),
                _ => unreachable!("curated native case"),
            };
            let expected = epochs
                .iter()
                .map(|epoch| {
                    let kind = match epoch.evaluation {
                        Evaluation::Primary => "solver_primary",
                        Evaluation::HeunCorrector => "solver_heun_corrector",
                        Evaluation::SdeMidpoint => "solver_sde_midpoint",
                    };
                    (
                        solver as u64,
                        u64::from(cfgpp),
                        epoch.step as u64,
                        kind,
                        epoch.sigma.to_bits() as u64,
                        epoch.next_sigma.to_bits() as u64,
                        u64::from(epoch.history_owners),
                        u64::from(epoch.outer_owners),
                    )
                })
                .collect::<Vec<_>>()
                .repeat(count as usize);
            let actual = observation
                .metadata
                .iter()
                .filter(|m| {
                    m.provider == "curated"
                        && matches!(
                            m.kind.as_str(),
                            "solver_primary" | "solver_heun_corrector" | "solver_sde_midpoint"
                        )
                })
                .map(|m| {
                    (
                        field(m, "solver_code"),
                        field(m, "cfg_pp"),
                        field(m, "step"),
                        m.kind.as_str(),
                        field(m, "sigma_bits"),
                        field(m, "next_sigma_bits"),
                        field(m, "history_denoised")
                            + field(m, "history_derivative")
                            + field(m, "history_sample"),
                        u64::from(field(m, "outer_x_logical_bytes") != 0)
                            + u64::from(field(m, "outer_x0_logical_bytes") != 0)
                            + u64::from(field(m, "outer_d_logical_bytes") != 0),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "served {sampler}/{method} actual callback/history/secondary owners"
            );
            let callbacks = marks("sdxl", "curated-callback");
            let epsilon = marks("sdxl", "curated-epsilon");
            let raw = marks("sdxl", "curated-model-input");
            let compute = marks("sdxl", "curated-compute-input");
            let uncond = marks("sdxl", "cfgpp-unconditional-epsilon");
            assert_eq!(
                (
                    callbacks.len(),
                    epsilon.len(),
                    raw.len(),
                    compute.len(),
                    uncond.len(),
                    marks("curated", "solver_correction")
                        .iter()
                        .map(|m| (
                            field(m, "step"),
                            field(m, "update_order"),
                            field(m, "cfg_pp")
                        ))
                        .collect::<Vec<_>>()
                ),
                (
                    expected.len(),
                    expected.len(),
                    expected.len(),
                    expected.len(),
                    if cfgpp { expected.len() } else { 0 },
                    epochs
                        .iter()
                        .filter(|e| e.corrector_after_callback)
                        .map(|e| (e.step as u64, 2u64, u64::from(cfgpp)))
                        .collect::<Vec<_>>()
                        .repeat(count as usize)
                ),
                "served {sampler}/{method} actual model callback/corrector bindings"
            );
            assert!(
                callbacks
                    .iter()
                    .all(|m| field(m, "cfg_on") == u64::from(guidance > 1.0)
                        && field(m, "cfg_pp") == u64::from(cfgpp)
                        && field(m, "cfg_batched") == u64::from(batching.is_batched()))
                    && epsilon
                        .iter()
                        .chain(uncond.iter())
                        .all(|m| field(m, "dtype") == epsilons[tier])
                    && raw.iter().all(|m| field(m, "dtype") == 3)
                    && compute.iter().all(|m| field(m, "dtype") == 1),
                "served {sampler}/{method} actual model dtype/guidance binding"
            );
            let updates = marks("curated", "solver_update");
            assert!(
                updates.len() == sigmas.len().saturating_sub(1) * count as usize
                    && updates.last().is_some_and(|m| field(m, "zero_next") == 1)
                    && checked(
                        &norm(&events.borrow()),
                        mlx_gen::sampler::SAMPLER_STEP_QUANTUM
                    )
                    .len()
                        == expected.len() + count as usize,
                "served {sampler}/{method} terminal update and existing barriers"
            );
        } else {
            let states = marks("sdxl", "discrete-state");
            let epsilon = marks("sdxl", "discrete-epsilon");
            let pending = marks("sdxl", "discrete-update-pending");
            let finished = marks("sdxl", "discrete-update-after-barrier");
            let blends = marks("sdxl", "discrete-blend-pending");
            let n = r.sampling.steps() * count as usize;
            assert_eq!(
                (
                    states.len(),
                    epsilon.len(),
                    pending.len(),
                    finished.len(),
                    blends.len()
                ),
                (n, n, n, n, if route == Route::Mask { n } else { 0 }),
                "served {sampler} actual first/steady/terminal/update/mask bindings"
            );
            assert!(
                epsilon.iter().all(|m| field(m, "dtype") == epsilons[tier])
                    && pending
                        .iter()
                        .zip(&finished)
                        .all(|(a, b)| field(a, "dtype") == field(b, "dtype"))
                    && checked(&norm(&events.borrow()), SAMPLER_STEP).len() == n,
                "served {sampler} actual epsilon/update dtype and barriers"
            );
        }
        eprintln!("SERVED ACTUAL {sampler}/{scheduler}/{method} tier={tier} guidance={guidance} batching={batching:?} route={route:?} count={count} metadata={} checks={}",observation.metadata.len(),observation.checks.len());
    }
    completion.body_completed = true;

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::served_sampler_factories_bind_actual_model_callbacks_and_owners");
}

/// Actual native acceleration factories versus the old literal affine update, using epsilon
/// from the loaded tiny model. Three 3-step trajectories plus ONE Hyper repeated transition:
/// no 51-step model campaign, requested-MAX schedule or physical alias claim from c_in==1.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn acceleration_transitions_bind_actual_model_epsilon_and_factory() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::acceleration_transitions_bind_actual_model_epsilon_and_factory",
    ) else {
        return;
    };
    with_cpu_diagnostic_headroom(PLENTY, || {
        use mlx_rs::ops::indexing::IndexOp;
        use mlx_gen::gen_core::sampling::SamplerPolicy;
        let snapshot = Snapshot::new(1);
        let mut files = Files::of(&snapshot);
        files.streamed = false;
        let (unet, _, _, _) = files.heavy(Route::Text)?;
        let (te1, te2) = files.text()?;
        let (cond, pooled) = (crate::test_device::record_model_inference_device(), crate::pipeline::encode_conditioning(&te1,&te2,&ids().index((0..1,..)))).1?;
        let memory = GenerationMemory::default();
        let config = crate::config::DiffusionConfig::sdxl_base();
        let alpha = mlx_gen::AlphaSchedule::scaled_linear(config.num_train_steps,config.beta_start,config.beta_end);
        for (name, steps, transitions) in [("lcm",3,3),("lightning",3,3),("hyper",3,3),("hyper",51,1)] {
            let mut req = generation_request(Route::Text,PX as u32,PX as u32,&memory);
            req.sampler = Some(name.into()); req.steps = Some(steps); req.guidance = Some(1.0);
            let sampling = crate::sampling::ResolvedSampling::resolve(&req)?;
            let sampler = sampling.build_acceleration_sampler(&alpha,7)?;
            let epochs = sampling.acceleration_epochs(&alpha)?;
            let noise = rnd(&[1,LATENT,LATENT,4],9,1.0);
            let mut x = sampler.scale_initial_noise(&noise)?;
            let scale = if name=="lightning" {
                mlx_gen::gen_core::sampling::LightningPolicy::new(&alpha,config.num_train_steps,steps as usize).init_noise_scale()
            } else { 1.0 };
            for (step, epoch) in epochs[..transitions].iter().enumerate() {
                let c = epoch.coefficients;
                let scaled = sampler.scale_model_input(&x,step)?;
                let epsilon = (crate::test_device::record_model_inference_device(), unet.forward_planned(&scaled,sampler.timestep(step),&cond,&pooled,&crate::pipeline::text_time_ids(1),None,None,crate::plan::SdxlForwardPlan::UNBOUNDED)).1?;
                let actual = sampler.step(&epsilon,&x,step)?;
                // Complete the actual model/update first. This separate numerical test is NOT a
                // liveness capture; build/read every oracle only after that actual completion.
                actual.eval()?;
                if step==0 {
                    let expected_prior = if scale==1.0 {noise.as_dtype(Dtype::Float32)?} else {mlx_rs::ops::multiply(&noise,mlx_gen::array::scalar(scale))?};
                    assert_eq!(bits(&x),bits(&expected_prior),"actual {name} native prior scaling steps={steps}");
                }
                let expected_input = if c.c_in==1.0 {x.as_dtype(crate::model::DTYPE)?} else {mlx_rs::ops::multiply(&x,mlx_gen::array::scalar(c.c_in))?.as_dtype(crate::model::DTYPE)?};
                assert_eq!((sampler.timestep(step).to_bits(),bits(&scaled)),(c.timestep.to_bits(),bits(&expected_input)),"actual {name} native timestep/input scaling steps={steps} step={step}");
                let state = x.as_dtype(Dtype::Float32)?;
                let out = epsilon.as_dtype(Dtype::Float32)?;
                // Independent pre-extraction MLX expression and exact seed-derived key. No
                // branch-predicate helper or native sampler is used to construct this oracle.
                let mut expected = if c.a_x==1.0 && c.a_noise==0.0 {
                    mlx_rs::ops::add(&state,mlx_rs::ops::multiply(&out,mlx_gen::array::scalar(c.a_out))?)?
                } else {
                    mlx_rs::ops::add(mlx_rs::ops::multiply(&state,mlx_gen::array::scalar(c.a_x))?,mlx_rs::ops::multiply(&out,mlx_gen::array::scalar(c.a_out))?)?
                };
                if c.a_noise!=0.0 {
                    let seed = 7u64.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64+1));
                    let key = mlx_rs::random::key(seed)?;
                    let noise = mlx_rs::random::normal::<f32>(expected.shape(),None,None,Some(&key))?;
                    expected = mlx_rs::ops::add(&expected,mlx_rs::ops::multiply(&noise,mlx_gen::array::scalar(c.a_noise))?)?;
                }
                expected.eval()?;
                assert_eq!(bits(&actual),bits(&expected),"actual {name} native old-expression update with actual model epsilon steps={steps} step={step}");
                if steps==51 {
                    assert!(epochs[step].additive_update&&!epochs[step].draws_noise&&sampler.timestep(0)==sampler.timestep(1)&&scaled.dtype()==crate::model::DTYPE&&x.dtype()==Dtype::Float32,"actual repeated Hyper factory/input branch, not whole-input physical alias");
                }
                x = actual;
            }
        }
        // Short repeated/leading-terminal production-driver mechanism case, not a claimed
        // full request-factory schedule or a ProcessGrowth witness.
        let ms = mlx_gen::DiscreteModelSampling::sdxl(&alpha);
        let sigma = mlx_gen::gen_core::sampling::ModelSampling::sigma_max(&ms);
        let sigmas = [sigma,sigma,0.0,0.0];
        let cancel = mlx_gen::CancelFlag::default();
        let x = rnd(&[1,LATENT,LATENT,4],9,1.0);
        let (result,observation) = observe_actual_marks(|| mlx_gen::run_curated_sampler(Some("euler"),&ms,&sigmas,x,7,&cancel,&mut |_|{},|xin,_| Ok(xin.as_dtype(crate::model::DTYPE)?)));
        let result = result?;
        let updates = observation.metadata.iter().filter(|m|m.provider=="curated"&&m.kind=="solver_update").collect::<Vec<_>>();
        assert!(updates.len()==3&&updates.last().is_some_and(|m|m.fields.get("terminal_alias")==Some(&1))&&result.shape()==[1,LATENT,LATENT,4],"actual repeated and leading-terminal native driver branches");
        Ok(())
    }).unwrap();
    completion.body_completed = true;

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::acceleration_transitions_bind_actual_model_epsilon_and_factory");
}

/// The shared production serial owner versus its frozen old inline seed/Vec loop, with native
/// arrays and real samplers. This establishes seed/count/Rust-scope/error behavior only, never
/// physical backing from a drop or equivalence of the whole Sdxl::generate on tiny components.
#[test]
fn actual_serial_owner_preserves_seed_retention_and_error_scopes() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::actual_serial_owner_preserves_seed_retention_and_error_scopes",
    ) else {
        return;
    };
    with_cpu_diagnostic_headroom(PLENTY, || {
        let config=crate::config::DiffusionConfig::sdxl_base();
        let alpha=mlx_gen::AlphaSchedule::scaled_linear(config.num_train_steps,config.beta_start,config.beta_end);
        let ms=mlx_gen::DiscreteModelSampling::sdxl(&alpha);
        let draw=|| mlx_rs::random::normal::<f32>(&[1,2,2,4],None,None,None).map_err(mlx_gen::Error::from);
        let predictions=Rc::new(RefCell::new(0usize));
        let run=|cancel:&mlx_gen::CancelFlag| {
            let x=draw()?;
            mlx_gen::run_curated_sampler(Some("euler"),&ms,&[2.0,1.0,0.0],x,7,cancel,&mut |_|{},|xin,_| {
                *predictions.borrow_mut()+=1;
                Ok(xin.as_dtype(crate::model::DTYPE)?)
            })
        };
        let base=u64::MAX;
        let seeds=Rc::new(RefCell::new(Vec::new()));
        let scope=Rc::new(RefCell::new(Vec::new()));
        struct DropMark(Rc<RefCell<Vec<&'static str>>>);
        impl Drop for DropMark { fn drop(&mut self) { self.0.borrow_mut().push("drop"); } }
        let cancel=mlx_gen::CancelFlag::default();
        let finals=crate::model::collect_serial_latents(2,base,|seed,finals| {
            let _local=DropMark(Rc::clone(&scope));
            seeds.borrow_mut().push(seed);
            let latent=run(&cancel)?;
            finals.push(latent);
            scope.borrow_mut().push("push");
            Ok(())
        })?;
        let mut old=Vec::with_capacity(2);
        for i in 0..2u32 { let seed=base.wrapping_add(i as u64); mlx_rs::random::seed(seed)?; old.push(run(&cancel)?); }
        mlx_rs::transforms::eval(finals.iter().chain(old.iter()))?;
        assert_eq!((seeds.borrow().clone(),scope.borrow().clone(),finals.iter().map(bits).collect::<Vec<_>>()),(vec![u64::MAX,0],vec!["push","drop","push","drop"],old.iter().map(bits).collect::<Vec<_>>()),"actual serial seed/wrapping/count/retained native outputs and append-before-local-drop parity");
        let failure_seeds=Rc::new(RefCell::new(Vec::new()));
        let error_scope=scope.borrow().len();
        let error=crate::model::collect_serial_latents(2,base,|seed,finals| {
            let _local=DropMark(Rc::clone(&scope));
            failure_seeds.borrow_mut().push(seed);
            if finals.len()==1 { return Err(mlx_gen::Error::Msg("intentional serial callback error".into())); }
            finals.push(run(&cancel)?); Ok(())
        });
        assert!(matches!(error,Err(mlx_gen::Error::Msg(ref s)) if s=="intentional serial callback error")&&*failure_seeds.borrow()==[u64::MAX,0]&&scope.borrow()[error_scope..]==["drop","drop"],"actual serial error propagation stops at the failing output and unwinds callback locals");
        let cancel=mlx_gen::CancelFlag::default();
        let cancel_scope=scope.borrow().len();
        let before_predictions=*predictions.borrow();
        let cancelled=crate::model::collect_serial_latents(2,base,|_,finals| {
            let _local=DropMark(Rc::clone(&scope));
            let latent=run(&cancel)?;
            finals.push(latent); cancel.cancel(); Ok(())
        });
        assert!(matches!(cancelled,Err(mlx_gen::Error::Canceled))&&*predictions.borrow()-before_predictions==2&&scope.borrow()[cancel_scope..]==["drop","drop"],"actual serial sampler observes cooperative cancellation without later predictions and unwinds callback locals");
        Ok(())
    }).unwrap();
    completion.body_completed = true;
}

/// Pure-header CLIP pack points versus actual shared-policy owners/guards, including native
/// per-projection transient checks. Logical slot replacement is NEVER subtracted from a physical
/// demand; native growth is compared only against its own checked actual pre-projection baseline.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn actual_text_finalize_points_match_headers_and_native_transients() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::actual_text_finalize_points_match_headers_and_native_transients",
    ) else {
        return;
    };
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
    #[derive(Default)]
    struct Capture {
        encoder: Option<usize>,
        slots: Vec<(usize, bool, u64)>,
        budgets: Vec<u64>,
        open: Option<(u64, u64)>,
        native: Vec<Option<(u64, u64)>>,
    }
    struct ClearListener;
    impl Drop for ClearListener {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    for (bits, stored) in [
        (None, false),
        (Some(8), false),
        (Some(4), false),
        (Some(8), true),
        (Some(4), true),
    ] {
        with_cpu_diagnostic_headroom(PLENTY, || {
            let snapshot=Snapshot::groupable_text(bits,stored);
            let files=Files::of(&snapshot);
            let (mut te1,mut te2)=files.text_unfinalized()?;
            let before=[te1.payload_layouts_for_test(),te2.payload_layouts_for_test()];
            let state=std::sync::Arc::new(std::sync::Mutex::new(Capture::default()));
            let sink=std::sync::Arc::clone(&state);
            let replaced=set_quantum_listener(Some(Box::new(move |mark| {
                let mut state=sink.lock().unwrap();
                match mark {
                    QuantumMark::Metadata {provider:"sdxl",kind,fields} => {
                        let field=|key| fields.iter().find(|(name,_)|*name==key).map(|(_,value)|*value);
                        if kind=="clip-pack-encoder" {
                            state.encoder=if field("active")==Some(1) {field("encoder").map(|n|n as usize)} else {None};
                        } else if let Some(encoder)=state.encoder {
                            if kind=="quantize-linear-before" {
                                if let Some(bytes)=field("logical_bytes") {state.slots.push((encoder,false,bytes));}
                            } else if kind=="quantize-linear-after" {
                                if let Some(bytes)=field("logical_bytes") {state.slots.push((encoder,true,bytes));}
                                let native=state.open.take().and_then(|(budget,start)| {
                                    checked_native_peak_bytes().max(checked_native_active_bytes()).checked_sub(start).map(|growth|(budget,growth))
                                });
                                state.native.push(native);
                            }
                        }
                    }
                    QuantumMark::Checked {label,scratch} if label==LOAD_QUANTIZE && state.encoder.is_some() => {
                        state.budgets.push(scratch);
                        state.open=Some((scratch,checked_native_active_bytes()));
                        checked_reset_native_peak();
                    }
                    _ => {}
                }
            })));
            let listener=ClearListener;
            if replaced { return Err(mlx_gen::Error::Msg("native pack fixture replaced an existing listener".into())); }
            if let Some(bits)=bits { crate::text_encoder::quantize_pair(&mut te1,&mut te2,bits)?; }
            drop(listener); // Close the actual region BEFORE any pricing/numeric oracle work.
            let after=[te1.payload_layouts_for_test(),te2.payload_layouts_for_test()];
            let headers=|path:&std::path::Path| mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path).map_err(|e|mlx_gen::Error::Msg(e.to_string()));
            let plans=[crate::text_encoder::text_pack_facts(&headers(&snapshot.file("text_encoder/model.safetensors"))?,&snapshot.te1,bits)?,
                crate::text_encoder::text_pack_facts(&headers(&snapshot.file("text_encoder_2/model.safetensors"))?,&snapshot.te2,bits)?];
            let state=state.lock().unwrap();
            let expected_budgets=plans.iter().flat_map(|plan|plan.points.iter().map(|point|point.scratch)).collect::<Vec<_>>();
            assert_eq!(state.budgets,expected_budgets,"actual CLIP pair pack guards follow exact selected header projections bits={bits:?} stored={stored}");
            let expected_before=plans[0].points.iter().map(|point|point.before_resident+before[1].1)
                .chain(plans[1].points.iter().map(|point|point.before_resident+plans[0].retained)).collect::<Vec<_>>();
            let mut current=[before[0].1,before[1].1]; let mut pending=[None,None]; let mut observed_before=Vec::new(); let mut valid=true;
            for &(encoder,after,bytes) in &state.slots {
                if encoder>1 {valid=false;continue;}
                if !after {observed_before.push(current[0]+current[1]);pending[encoder]=Some(bytes);}
                else if let Some(previous)=pending[encoder].take() {
                    match current[encoder].checked_sub(previous).and_then(|n|n.checked_add(bytes)) {Some(next)=>current[encoder]=next,None=>valid=false}
                } else {valid=false;}
            }
            assert!(valid&&pending==[None,None]&&observed_before==expected_before&&current==[after[0].1,after[1].1]&&current==[plans[0].retained,plans[1].retained],"actual CLIP coincident logical owners match each pack point, not a future maximum bits={bits:?} stored={stored}");
            assert!(state.native.len()==expected_budgets.len()&&state.native.iter().all(|interval|interval.is_some_and(|(budget,growth)|growth<=budget)),"actual native CLIP projection transient uses checked native baseline, not a logical subtraction bits={bits:?} stored={stored}");
            for index in 0..2 {
                let exceptions=before[index].0.iter().filter(|(key,_)|key.contains("embeddings.")||key.contains("layer_norm"));
                assert!(exceptions.into_iter().all(|(key,layout)|after[index].0.get(key)==Some(layout))&&after[index].0.iter().filter(|(key,_)|key.ends_with(".bias")&&!key.ends_with(".biases")).all(|(_,layout)|layout.0.len()==1&&layout.1!=Dtype::Uint32),"actual CLIP norms/embeddings/biases are not blindly packed bits={bits:?} stored={stored}");
            }
            drop(state);
            let mut quanta=Vec::new();
            let sp=spec(&snapshot,Route::Text,Selection::Rung0);
            let (sel,memory)=selection(&sp,Selection::Rung0);
            let regime=regime(&sp,&MemoryMode::TextToImage,&sel,&memory);
            let mut inputs=Inputs::from_spec_for_test(&sp,&regime)?;
            inputs.arch=snapshot.arch(); // Existing private pricing-fixture config seam only.
            let (priced,dry_capture)=observe_actual_marks(||crate::quantum::with_backend(Backend::Cpu,||Priced::build(&inputs)));
            let priced=priced?;
            assert!(dry_capture.checks.is_empty()&&dry_capture.metadata.iter().all(|mark|mark.kind!="quantize-linear-before"&&mark.kind!="quantize-linear-after"),"CLIP raw pricing stays in the lazy header/shape-only path, not actual pack/materialization bits={bits:?} stored={stored}");
            // Drive the ACTUAL complete price ledger, not a parallel reimplementation of its
            // control flow. Expected owners come from checked actual pre/post payloads and the
            // independently compared per-projection facts above, never physical subtraction.
            let (ledger_result, price_capture)=observe_actual_marks(||crate::quantum::with_backend(Backend::Cpu,||priced.price(&regime)));
            let (_, ledger) = ledger_result?;
            assert!(price_capture.checks.is_empty()&&price_capture.metadata.iter().all(|mark|mark.kind!="quantize-linear-before"&&mark.kind!="quantize-linear-after"),"complete CLIP pricing ledger stays shape/header-only bits={bits:?} stored={stored}");
            let floors=[
                crate::loader::cast_load_parts(&[snapshot.file("text_encoder/model.safetensors")],crate::model::DTYPE,true,|_|true)?,
                crate::loader::cast_load_parts(&[snapshot.file("text_encoder_2/model.safetensors")],crate::model::DTYPE,true,|_|true)?,
            ];
            assert_eq!([floors[0].0,floors[1].0],[before[0].1,before[1].1],"actual text pre-pack owners match independently selected load headers");
            let mut expected_load=vec![(TEXT_ENCODER_1_LOAD,0,floors[0].1),(TEXT_ENCODER_2_LOAD,before[0].1,floors[1].1)];
            expected_load.extend(expected_before.iter().zip(&expected_budgets).map(|(resident,scratch)|(LOAD_QUANTIZE,*resident,*scratch)));
            expected_load.push((UNET_LOAD,after[0].1+after[1].1,priced.unet_bytes.floor));
            let actual_load=ledger.iter().filter(|point|point.phase==IosPhase::Load).take(expected_load.len()).map(|point|(point.label,point.resident,point.scratch)).collect::<Vec<_>>();
            assert_eq!(actual_load,expected_load,"actual pricing load ledger retains raw TE1 for TE2, each coincident pack owner, and finalized text beside heavy load bits={bits:?} stored={stored}");
            let predicted=priced.price_text_conditioning(2,&mut quanta)?;
            let (conditioning,pooled)=(crate::test_device::record_model_inference_device(), crate::pipeline::encode_conditioning(&te1,&te2,&ids())).1?;
            mlx_rs::transforms::eval([&conditioning,&pooled])?;
            assert_eq!((conditioning.dtype(),pooled.dtype(),conditioning.shape().to_vec(),pooled.shape().to_vec()),(predicted.0,predicted.1,vec![2,TOKENS,snapshot.context],vec![2,snapshot.te2.projection_dim.unwrap()]),"actual post-pack CLIP conditioning dtype/layout follows actual models, never tier-fixed bits={bits:?} stored={stored}");
            // Negative admission must occur before the refused projection is changed. A packed
            // source has no pack quantum and is legitimately a no-op, not a skipped refusal.
            if let (Some(bits),Some(first))=(bits,plans[0].points.first()) {
                let (mut one,mut two)=files.text_unfinalized()?;
                let layouts=[one.payload_layouts_for_test().0,two.payload_layouts_for_test().0];
                let result=with_cpu_diagnostic_headroom(first.scratch-1,||crate::text_encoder::quantize_pair(&mut one,&mut two,bits));
                let refusal=take_refusal();
                assert!(result.is_err()&&refusal.is_some_and(|r|r.label==LOAD_QUANTIZE)&&layouts==[one.payload_layouts_for_test().0,two.payload_layouts_for_test().0],"actual CLIP pack refuses before changing the projection or later encoder bits={bits} stored={stored}");
            }
            // These factories are consumed by the REAL Dir/LDM builders. Only tiny encoder
            // configuration replaces the hardcoded production dimensions; tier validation,
            // pair-pack ordering and fused final materialization stay actual production flow.
            // Component maps below are already split: this is NOT proof of full-file remapping,
            // outer immutable-file guards, tokenizers or Sdxl::generate.
            #[derive(Default)]
            struct LoadTrace { events: Vec<String>, budgets: Vec<u64>, raw: Vec<u64>, materialized: Vec<u64> }
            for fused in [false,true] {
                let trace=std::sync::Arc::new(std::sync::Mutex::new(LoadTrace::default()));
                let sink=std::sync::Arc::clone(&trace);
                assert!(!set_quantum_listener(Some(Box::new(move |mark| {
                    let mut trace=sink.lock().unwrap();
                    match mark {
                        QuantumMark::Metadata {provider:"sdxl",kind:"clip-pack-encoder",fields} => {
                            let field=|key|fields.iter().find(|(name,_)|*name==key).map(|(_,value)|*value);
                            if field("active")==Some(1) {trace.events.push(format!("pack{}",field("encoder").unwrap()));}
                        }
                        QuantumMark::Checked {label:LOAD_QUANTIZE,scratch} => trace.budgets.push(scratch),
                        _=>{}
                    }
                }))));
                let listener=ClearListener;
                let quant=bits.map(|bits|if bits==4 {mlx_gen::gen_core::Quant::Q4}else{mlx_gen::gen_core::Quant::Q8});
                let loaded=if fused {
                    crate::model::build_ldm_text_using(
                        Weights::from_file(snapshot.file("text_encoder/model.safetensors"))?,
                        Weights::from_file(snapshot.file("text_encoder_2/model.safetensors"))?, quant,
                        |weights,dtype| {let encoder=crate::loader::load_clip_from_weights(weights,&snapshot.te1,dtype)?;let mut t=trace.lock().unwrap();t.events.push("load0".into());t.raw.push(encoder.payload_layouts_for_test().1);Ok(encoder)},
                        |weights,dtype| {let encoder=crate::loader::load_clip_from_weights(weights,&snapshot.te2,dtype)?;let mut t=trace.lock().unwrap();t.events.push("load1".into());t.raw.push(encoder.payload_layouts_for_test().1);Ok(encoder)},
                        |encoder| {encoder.materialize_weights()?;let mut t=trace.lock().unwrap();let index=t.materialized.len();t.events.push(format!("materialize{index}"));t.materialized.push(encoder.payload_layouts_for_test().1);Ok(())},
                    )?
                } else {
                    crate::model::load_text_encoders_using(&snapshot.root,quant,
                        |root,dtype| {let encoder=crate::loader::load_clip_dtype(root,"text_encoder",&snapshot.te1,dtype)?;let mut t=trace.lock().unwrap();t.events.push("load0".into());t.raw.push(encoder.payload_layouts_for_test().1);Ok(encoder)},
                        |root,dtype| {let encoder=crate::loader::load_clip_dtype(root,"text_encoder_2",&snapshot.te2,dtype)?;let mut t=trace.lock().unwrap();t.events.push("load1".into());t.raw.push(encoder.payload_layouts_for_test().1);Ok(encoder)},
                    )?
                };
                drop(listener);
                let mut events=vec!["load0".to_owned(),"load1".to_owned()];
                if bits.is_some() {events.extend(["pack0".to_owned(),"pack1".to_owned()]);}
                if fused {events.extend(["materialize0".to_owned(),"materialize1".to_owned()]);}
                let t=trace.lock().unwrap();
                assert!(t.events==events&&t.raw==[before[0].1,before[1].1]&&t.budgets==expected_budgets&&(!fused||t.materialized==[after[0].1,after[1].1]),"actual Dir/LDM load-pack-materialize order and owners preserve both raw encoders before requested packing bits={bits:?} stored={stored} fused={fused}");
                assert_eq!([loaded.0.payload_layouts_for_test(),loaded.1.payload_layouts_for_test()],after,"actual Dir/LDM builder delegates requested pair policy without changing packed no-op or norm/embedding/bias layouts bits={bits:?} stored={stored} fused={fused}");
            }
            // Actual Dir tier mismatch is checked AFTER construction and BEFORE any pair pack.
            // Only packed U-Net fixtures have a mismatched tier to reject; no invented fused rule.
            if let Some(bits)=bits {
                let wrong=if bits==4 {mlx_gen::gen_core::Quant::Q8}else{mlx_gen::gen_core::Quant::Q4};
                let (mismatch,marks)=observe_actual_marks(||crate::model::load_text_encoders_using(&snapshot.root,Some(wrong),
                    |root,dtype|crate::loader::load_clip_dtype(root,"text_encoder",&snapshot.te1,dtype),
                    |root,dtype|crate::loader::load_clip_dtype(root,"text_encoder_2",&snapshot.te2,dtype)));
                assert!(mismatch.is_err_and(|e|e.to_string().contains("pre-quantized"))&&marks.metadata.iter().all(|m|m.kind!="clip-pack-encoder"),"actual Dir packed-tier mismatch refuses before pair pack bits={bits} stored={stored}");
            }
            eprintln!("ACTUAL CLIP FINALIZE bits={bits:?} stored={stored} points={} conditioning={:?} pooled={:?}",expected_budgets.len(),conditioning.dtype(),pooled.dtype());
            Ok(())
        }).unwrap();
    }
    completion.body_completed = true;

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::actual_text_finalize_points_match_headers_and_native_transients");
}

/// The actual shared finalize policy versus the frozen old inline pair walk, throughout loaded
/// tiny CLIP→U-Net→VAE execution. No tolerance or sampler-precision change hides promotion; both
/// runs use the SAME iOS boundary policy and only the finalize wrapper differs.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn actual_text_finalize_preserves_pipeline_rgb_and_dtype() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::actual_text_finalize_preserves_pipeline_rgb_and_dtype",
    ) else {
        return;
    };
    for (bits, stored) in [
        (None, false),
        (Some(8), false),
        (Some(4), false),
        (Some(8), true),
        (Some(4), true),
    ] {
        let snapshot = Snapshot::groupable_text(bits, stored);
        let sp = spec(&snapshot, Route::Text, Selection::Rung0);
        let (selection, memory) = selection(&sp, Selection::Rung0);
        let r = regime(&sp, &MemoryMode::TextToImage, &selection, &memory);
        let files = Files::of(&snapshot);
        for fused in [false, true] {
            let mut actual_files = files.clone();
            actual_files.text_ldm = fused;
            let mut reference = actual_files.clone();
            reference.text_finalize_reference = true;
            let expected = with_cpu_diagnostic_headroom(PLENTY, || {
                (
                    crate::test_device::record_model_inference_device(),
                    run_image(&reference, &r, &memory),
                )
                    .1
            })
            .unwrap();
            let actual = with_cpu_diagnostic_headroom(PLENTY, || {
                (
                    crate::test_device::record_model_inference_device(),
                    run_image(&actual_files, &r, &memory),
                )
                    .1
            })
            .unwrap();
            assert_eq!((actual.width,actual.height,actual.pixels),(expected.width,expected.height,expected.pixels),"actual shared CLIP finalize preserves full tiny-pipeline RGB without relaxed dtype/tolerance bits={bits:?} stored={stored} fused={fused}");
        }
    }
    completion.body_completed = true;

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::actual_text_finalize_preserves_pipeline_rgb_and_dtype");
}

/// Record actual first/steady/zero-step state types and Denoise growth bindings from a real
/// pre-load native baseline. Scalar observation cannot make an Array non-donatable.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn actual_sampler_epochs_and_denoise_peak_bindings() {
    let Some(mut completion) =
        completed_cpu_child("pricing::additivity::actual_sampler_epochs_and_denoise_peak_bindings")
    else {
        return;
    };
    let dense = Snapshot::new(2);
    let q8 = Snapshot::packed(8);
    let q4 = Snapshot::packed(4);
    let cases = [
        ("q8", &q8, CfgBatching::Sequential, None, true),
        ("q8", &q8, CfgBatching::Batched, Some((2, 1.0)), false),
        ("q8", &q8, CfgBatching::Sequential, Some((3, 0.8)), false),
        ("q8", &q8, CfgBatching::Sequential, Some((1, 0.8)), false),
        ("q4", &q4, CfgBatching::Sequential, None, true),
        (
            "bf16",
            &dense,
            CfgBatching::Sequential,
            Some((3, 0.8)),
            false,
        ),
    ];
    for (tier, snap, batching, schedule, mask) in cases {
        let mut baseline: Option<StepOwnershipTrace> = None;
        let mut baseline_profile: Option<(Regime, IosPricing)> = None;
        for route in [Route::Text, Route::Reference]
            .into_iter()
            .chain(mask.then_some(Route::Mask))
        {
            let mode = match route {
                Route::Text => MemoryMode::TextToImage,
                Route::Reference => MemoryMode::ImageToImage,
                _ => MemoryMode::Edit,
            };
            let sp = spec(snap, route, Selection::Rung0);
            let (sel, mut memory) = selection(&sp, Selection::Rung0);
            memory.cfg_batching = Some(batching);
            let mut r = regime(&sp, &mode, &sel, &memory);
            if let Some((requested, strength)) = schedule {
                let mut generation = generation_request(route, PX as u32, PX as u32, &memory);
                generation.steps = Some(requested as u32);
                generation.strength = Some(strength);
                r.sampling = crate::sampling::ResolvedSampling::resolve(&generation).unwrap();
            }
            let trace = Rc::new(RefCell::new(StepOwnershipTrace {
                schedule,
                ..Default::default()
            }));
            let sink = Rc::clone(&trace);
            let mut open = None;
            let (result, mut observation) = observe_actual_marks(|| {
                with_cpu_diagnostic_headroom(PLENTY, || {
                    trace.borrow_mut().start_active = checked_native_active_bytes();
                    checked_reset_native_peak();
                    with_cpu_diagnostic_observer(
                        move |event| match event {
                            QuantumEvent::Checked { label, scratch }
                                if ios_quantum_phase(label) == Some(IosPhase::Denoise) =>
                            {
                                let mut trace = sink.borrow_mut();
                                let step = trace.current;
                                trace.checks.push((step, label.clone(), *scratch));
                                let growth = checked_native_peak_bytes()
                                    .max(checked_native_active_bytes())
                                    .checked_sub(trace.start_active)
                                    .expect("native lifetime baseline");
                                trace.whole_demand = trace.whole_demand.max(growth);
                                drop(trace);
                                open = Some(label.clone());
                                checked_reset_native_peak();
                            }
                            QuantumEvent::Synchronized { label, .. }
                                if ios_quantum_phase(label) == Some(IosPhase::Denoise) =>
                            {
                                assert_eq!(open.take().as_deref(), Some(label.as_str()));
                                let peak =
                                    checked_native_peak_bytes().max(checked_native_active_bytes());
                                let mut trace = sink.borrow_mut();
                                let growth = peak.checked_sub(trace.start_active)
                                .expect("Denoise native peak must include the actual pre-load native baseline");
                                let step = trace.current;
                                trace.whole_demand = trace.whole_demand.max(growth);
                                trace.peaks.push((step, label.clone(), growth));
                            }
                            _ => {}
                        },
                        || {
                            (
                                crate::test_device::record_model_inference_device(),
                                run_image_traced(&Files::of(snap), &r, &memory, Some(&trace)),
                            )
                                .1
                        },
                    )
                })
            });
            result.unwrap();
            {
                let mut trace = trace.borrow_mut();
                let growth = checked_native_peak_bytes()
                    .max(checked_native_active_bytes())
                    .checked_sub(trace.start_active)
                    .expect("native lifetime peak includes its actual starting baseline");
                trace.whole_demand = trace.whole_demand.max(growth);
            }
            let trace = trace.borrow();
            observation.baseline_bytes = trace.start_active;
            let steps = match schedule {
                Some((requested, strength)) if route != Route::Text => {
                    crate::model::ancestral_strength_schedule(requested, 1000.0, strength).0
                }
                Some((requested, _)) => requested,
                None => {
                    r.sampling
                        .ancestral_window(
                            1000.0,
                            r.sampling
                                .effective_init_strength(r.has_init, route == Route::Mask),
                        )
                        .unwrap()
                        .0
                }
            };
            assert_eq!(trace.inputs.len(), steps);
            assert_eq!(trace.outputs.len(), steps);
            let output_dtype = if route != Route::Text || tier != "bf16" {
                Dtype::Float32
            } else {
                crate::model::DTYPE
            };
            for (step, dtype) in &trace.outputs {
                assert_eq!(
                    *dtype, output_dtype,
                    "{tier} {route:?} actual step{step} output, including final epoch"
                );
            }
            if route == Route::Text {
                assert!(trace.init.is_none());
            } else {
                assert_eq!(
                    trace.init,
                    Some((Dtype::Float32, (LATENT * LATENT * 4) as u64 * 4))
                );
            }
            for (i, state, scaled) in &trace.inputs {
                let expected = if route != Route::Text || (tier != "bf16" && *i > 0) {
                    Dtype::Float32
                } else {
                    crate::model::DTYPE
                };
                assert_eq!(
                    *state, expected,
                    "{tier} {route:?} step{i}: actual caller state"
                );
                assert_eq!(*scaled, *state, "ancestral model-input scaling is identity");
                let peak = trace
                    .peaks
                    .iter()
                    .filter(|(step, _, _)| step == i)
                    .max_by_key(|(_, _, peak)| peak)
                    .expect("every executed denoise epoch has a checked peak binding");
                eprintln!("OWNERSHIP TRACE {tier} {route:?} {batching:?} schedule={schedule:?} step={i} state={state:?} scaled={scaled:?} init={:?} peak={} binding={}", trace.init, peak.2, peak.1);
            }
            if steps == 0 {
                let profile = pricing(snap, &r);
                assert!(profile.phase(IosPhase::Denoise).is_none());
                assert!(profile
                    .windows
                    .iter()
                    .all(|window| window.phase != IosPhase::Denoise));
                assert!(
                    profile
                        .absent_phases
                        .get(&IosPhase::Denoise)
                        .is_some_and(|reason| !reason.is_empty()),
                    "zero-step public pricing requires an explicit actual execution fact"
                );
                assert!(
                    observation
                        .metadata
                        .iter()
                        .any(|mark| mark.provider == "sdxl"
                            && mark.kind == "discrete-zero-denoise"
                            && mark.fields.get("steps") == Some(&0)),
                    "the real loop must observe the zero-step branch, not an invented epoch"
                );
                eprintln!("OWNERSHIP TRACE {tier} {route:?}: zero denoise epochs (legal strength tail), no model/step peak claimed");
            }
            if route == Route::Text {
                baseline = Some((*trace).clone()); // scalar metadata only, never an Array
                let mut profile = pricing(snap, &r);
                assert!(
                    !observation.checks.is_empty(),
                    "anchor requires actual checked windows"
                );
                assert!(
                    !observation.metadata.is_empty(),
                    "anchor requires actual scalar metadata"
                );
                profile.observations.push(observation);
                baseline_profile = Some((r.clone(), profile));
            } else if steps > 0 {
                let base = baseline.as_ref().expect("text baseline ran first");
                let signature = |trace: &StepOwnershipTrace, step| {
                    trace
                        .checks
                        .iter()
                        .filter(|(i, label, _)| {
                            *i == step
                                && label != SAMPLER_STEP
                                && label != mlx_gen::sampler::SAMPLER_STEP_QUANTUM
                        })
                        .map(|(_, label, scratch)| (label.clone(), *scratch))
                        .collect::<Vec<_>>()
                };
                for step in 0..steps {
                    let input_dtype = trace.inputs[step].1;
                    if let Some((base_step, _, _)) = base
                        .inputs
                        .iter()
                        .find(|(_, dtype, _)| *dtype == input_dtype)
                    {
                        assert_eq!(signature(&trace, step), signature(base, *base_step),
                            "{tier} {route:?}: same actual operand dtype must have the same model quantum bindings at epoch {step}");
                    } else {
                        assert_eq!(
                            tier, "bf16",
                            "every packed request body needs an actual same-dtype anchor epoch"
                        );
                        assert_ne!(signature(&trace, step), signature(base, 0),
                            "different dense forward precision must not claim the old common-body certificate");
                    }
                }
                let flows = |trace: &StepOwnershipTrace| {
                    trace
                        .inputs
                        .iter()
                        .map(|(_, state, _)| ForwardInputOwnership {
                            state: *state,
                            scaled: None,
                            cast_before_cfg: false,
                        })
                        .collect::<Vec<_>>()
                };
                let excess = forward_input_excess(
                    (LATENT * LATENT * 4) as u64,
                    crate::model::DTYPE,
                    if batching.is_batched() { 2 } else { 1 },
                    &flows(base),
                    &flows(&trace),
                );
                // Packed resident bindings have no recapture spread in the motivating failure.
                // Dense allocation overlap varies; its hard operator-storage law is checked by
                // forward_input_owners_match_native_alias_cast_and_cfg_storage instead.
                if tier != "bf16" {
                    let base_peak = base
                        .peaks
                        .iter()
                        .map(|(_, _, p)| *p)
                        .max()
                        .expect("baseline model peak");
                    let request_peak = trace
                        .peaks
                        .iter()
                        .map(|(_, _, p)| *p)
                        .max()
                        .expect("request model peak");
                    let measured_excess = request_peak.saturating_sub(base_peak);
                    let (anchor_regime, _) = baseline_profile.as_ref().unwrap();
                    let page = observation
                        .metadata
                        .iter()
                        .find(|mark| mark.provider == "mlx" && mark.kind == "allocation-scope")
                        .and_then(|mark| mark.fields.get("page_bytes"))
                        .copied()
                        .expect("actual allocation-scope page witness");
                    let source_prefix = if crate::ios_pricing::forward_prefix_differs(
                        (
                            anchor_regime.route,
                            anchor_regime.has_init,
                            anchor_regime.initial_state_dtype(),
                        ),
                        (r.route, r.has_init, r.initial_state_dtype()),
                    ) {
                        crate::ios_pricing::uncancelled_forward_prefix_upper(
                            &mut endpoint_inputs(snap, &r),
                            &r,
                            page,
                        )
                        .unwrap()
                    } else {
                        0
                    };
                    assert!(
                        source_prefix > 0,
                        "image initialization requires actual source-prefix allocation work"
                    );
                    let bound = source_prefix
                        + excess
                        + trace.fixed_inputs.saturating_sub(base.fixed_inputs);
                    assert!(bound >= measured_excess,
                        "{tier} {route:?} {batching:?}: source prefix + matched-epoch owners {bound} below actual binding excess {measured_excess}");
                    eprintln!("MATCHED INPUT BOUND {tier} {route:?} {batching:?} schedule={schedule:?}: forward_excess={excess} fixed={} measured_excess={measured_excess}", trace.fixed_inputs);
                    // Exercise the SAME source callback the public provider bounder uses, over
                    // real public pricing windows. No ad hoc input formula is substituted for
                    // the consumer: all candidates retain the first/steady binding identity.
                    let (anchor_regime, anchor_profile) = baseline_profile.as_ref().unwrap();
                    let mut request_profile = pricing(snap, &r);
                    request_profile.request_baseline =
                        Some(mlx_gen::ios_pricing::IosMemoryBaseline {
                            domain: mlx_gen::ios_pricing::IosMemoryDomain::NativeAllocatorGrowth,
                            coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
                            bytes: trace.start_active,
                            source:
                                "checked mlx_get_active_memory before native fixture load/generate"
                                    .into(),
                        });
                    // The production premise-mode states: steady = first state joined with eps.
                    let states = [anchor_regime, &r].map(|regime| {
                        crate::quantum::with_backend(Backend::Cpu, || {
                            Priced::build(&endpoint_inputs(snap, regime))
                                .and_then(|priced| priced.ancestral_states(regime))
                        })
                        .unwrap()
                    });
                    let bounds = crate::ios_pricing::bound_ancestral_forward_windows(
                        anchor_regime,
                        anchor_profile,
                        &r,
                        &request_profile,
                        VAE_SCALE,
                        states,
                        &mut endpoint_inputs(snap, &r),
                    )
                    .unwrap();
                    let model_windows = request_profile
                        .windows
                        .iter()
                        .filter(|window| {
                            window.owner_domain == "sdxl/ancestral/forward-input-native-v1"
                        })
                        .count();
                    assert!(
                        model_windows > 0,
                        "executed native forwards require public windows"
                    );
                    assert_eq!(
                        bounds.len(),
                        model_windows,
                        "every real first/steady/CFG model window needs its matched certificate"
                    );
                    let production_excess = bounds.iter().flat_map(|bound| match bound {
                        mlx_gen::ios_pricing::IosWindowBound::Transfer { anchor_candidates, .. } =>
                            anchor_candidates.iter().map(|candidate| candidate.delta_growth_upper),
                        mlx_gen::ios_pricing::IosWindowBound::RawPhysical { .. } =>
                            panic!("this native common-body proof must not invent a full raw process bound"),
                    }).max().expect("actual forward candidates").max(0) as u64;
                    assert!(production_excess >= measured_excess,
                        "{tier} {route:?} {batching:?}: public matched-window bound {production_excess} below real Denoise excess {measured_excess}");
                    let measured_anchor = MeasuredRun {
                        demand: base.whole_demand,
                        phases: std::collections::BTreeMap::from([(IosPhase::Denoise, base_peak)]),
                    }
                    .anchor(area(PX as u32, PX as u32, 1));
                    let mut public_upper = None;
                    for bound in &bounds {
                        let mlx_gen::ios_pricing::IosWindowBound::Transfer {
                            request_class, ..
                        } = bound
                        else {
                            unreachable!("native forward transfer only")
                        };
                        let upper = mlx_gen::ios_anchor_law::derive_window_certified(
                            &measured_anchor, anchor_profile, &request_profile, &bounds, request_class,
                        ).expect("actual source witness must reach the real public native window consumer");
                        public_upper =
                            Some(public_upper.map_or(upper, |prior: u64| prior.max(upper)));
                    }
                    let public_upper = public_upper.expect("actual native window bounds");
                    assert_eq!(public_upper.checked_sub(measured_anchor.demand), Some(production_excess),
                        "public law must preserve the source-matched differential, without calibration or extra margins");
                    assert!(
                        public_upper >= request_peak,
                        "public native window consumer underpredicted real Denoise growth"
                    );
                    assert!(mlx_gen::ios_anchor_law::derive_phase_certified(
                        &measured_anchor, anchor_profile, &request_profile, &bounds, IosPhase::Denoise,
                    ).is_err(), "unproved sampler/host windows must not disappear into a forward-only certificate");
                    assert!(
                        mlx_gen::ios_anchor_law::derive_certified(
                            &measured_anchor,
                            anchor_profile,
                            &request_profile,
                            &bounds,
                        )
                        .is_err(),
                        "native forward proof is not complete phase/process admission"
                    );
                }
            }
        }
    }
    completion.body_completed = true;

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::actual_sampler_epochs_and_denoise_peak_bindings");
}

/// Prove endpoint ownership on the actual loader, while the loaded UNet is still held. An
/// unarmed but unevaluated placeholder must not pass merely because its stream flag is false.
#[test]
fn endpoints_use_literal_resident_and_streamed_load_ownership() {
    if !cpu_child("pricing::additivity::endpoints_use_literal_resident_and_streamed_load_ownership")
    {
        return;
    }
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        for which in [Selection::Optimized, Selection::Rung0] {
            let expected_streamed = which == Selection::Optimized;
            let sp = spec(&snap, Route::Text, which);
            assert_eq!(
                crate::memory_strategy::streamable(&sp),
                expected_streamed,
                "{tier} {which:?}: actual spec, not the rung name, decides materialization"
            );
            let (sel, memory) = selection(&sp, which);
            let r = regime(&sp, &MemoryMode::TextToImage, &sel, &memory);
            assert_eq!(memory.transformer_window_size.is_some(), expected_streamed);
            assert_eq!(r.staged, expected_streamed);
            let file = snap.file("unet/diffusion_pytorch_model.safetensors");
            let (full_weights, _) =
                crate::loader::cast_load_parts(&[file], crate::model::DTYPE, true, |_| true)
                    .unwrap();
            let inputs = endpoint_inputs(&snap, &r);
            assert_eq!(
                inputs.streamable, expected_streamed,
                "pricing input must describe actual endpoint load ownership"
            );
            if !expected_streamed {
                let priced = Priced::build(&inputs).unwrap();
                assert!(
                    priced.unet_bytes.resident >= full_weights,
                    "resident pricing must include the complete UNet weight inventory"
                );
            }
            with_cpu_diagnostic_headroom(PLENTY, || {
                let before = mlx_rs::memory::get_active_memory() as u64;
                let mut files = Files::of(&snap);
                files.streamed = expected_streamed;
                let (unet, controls, vae, ip) = files.heavy(Route::Text).unwrap();
                assert_eq!(unet.can_stream_blocks(), expected_streamed);
                // Isolate the exact owner under test. VAE/optional-component bytes cannot
                // conceal missing UNet storage; retain the UNet and diagnostic guard until AFTER
                // the native allocator assertion.
                drop((controls, vae, ip));
                let held = (mlx_rs::memory::get_active_memory() as u64).saturating_sub(before);
                if !expected_streamed {
                    assert!(held >= full_weights,
                        "{tier}: caller-owned resident weights {held} below full UNet header inventory {full_weights}");
                }
                eprintln!("ENDPOINT OWNERSHIP {tier} {which:?}: streamed={} held={held} full_unet_weights={full_weights}", unet.can_stream_blocks());
                drop(unet);
                Ok(())
            }).unwrap();
        }
    }
}

/// Complete certified derivation over real text-anchor marks, every declared additive mode,
/// both literal load endpoints and all served storage tiers. No uncertified fallback is accepted.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn every_additive_mode_is_derived_from_the_text_anchor() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::every_additive_mode_is_derived_from_the_text_anchor",
    ) else {
        return;
    };
    for (tier, snapshot) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        for which in [Selection::Optimized, Selection::Rung0] {
            let text_spec = spec(&snapshot, Route::Text, which);
            let (sel, memory) = selection(&text_spec, which);
            let text = regime(&text_spec, &MemoryMode::TextToImage, &sel, &memory);
            let (measured_text, at_anchor) = captured_source_profile(&snapshot, &text, &memory);
            let anchor = measured_text.anchor(area(PX as u32, PX as u32, 1));
            let text_inputs = endpoint_inputs(&snapshot, &text);
            for (mode, route, _) in modes() {
                if ios_non_additive_modes(tier).contains(&mode.as_key()) {
                    continue;
                }
                let mode_spec = spec(&snapshot, route, which);
                let (mode_sel, mode_memory) = selection(&mode_spec, which);
                assert_eq!((mode_sel.strategy, mode_memory), (sel.strategy, memory));
                let request = regime(&mode_spec, &mode, &mode_sel, &mode_memory);
                let (measured, profile) =
                    captured_source_profile(&snapshot, &request, &mode_memory);
                let request_inputs = endpoint_inputs(&snapshot, &request);
                let bounds = crate::ios_pricing::certified_request_bounds_for_test(
                    &text_inputs,
                    &text,
                    &at_anchor,
                    &request_inputs,
                    &request,
                    &profile,
                )
                .unwrap_or_else(|error| panic!("{tier} {which:?} {}: {error}", mode.as_key()));
                let derived = derive_certified(&anchor, &at_anchor, &profile, &bounds)
                    .unwrap_or_else(|error| panic!("{tier} {which:?} {}: {error}", mode.as_key()));
                assert_certified_native_run(
                    &measured,
                    &derived,
                    &format!("{tier} {which:?} {}", mode.as_key()),
                );
                let mut unwitnessed = at_anchor.clone();
                unwitnessed.observations.clear();
                assert!(
                    crate::ios_pricing::certified_request_bounds_for_test(
                        &text_inputs,
                        &text,
                        &unwitnessed,
                        &request_inputs,
                        &request,
                        &profile,
                    )
                    .is_err(),
                    "{tier} {which:?} {}: planned windows without actual marks are not witnesses",
                    mode.as_key()
                );
                let mut forged = profile.clone();
                forged.windows[0].scratch_bytes += 1;
                assert!(
                    crate::ios_pricing::certified_request_bounds_for_test(
                        &text_inputs,
                        &text,
                        &at_anchor,
                        &request_inputs,
                        &request,
                        &forged,
                    )
                    .is_err(),
                    "fixture adapter must retain production source-profile equality"
                );
            }
        }
    }
    completion.body_completed = true;

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::every_additive_mode_is_derived_from_the_text_anchor");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::every_additive_mode_is_derived_from_the_text_anchor");
}

/// Count two retains the real first latent while executing the second output, not a batch-one
/// assertion relabeled as batching. Production request reconstruction rejects mismatched geometry.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn serial_count_two_uses_real_geometry_and_complete_native_bounds() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::serial_count_two_uses_real_geometry_and_complete_native_bounds",
    ) else {
        return;
    };
    let snapshot = Snapshot::packed(4);
    let sp = spec(&snapshot, Route::Text, Selection::Rung0);
    let (sel, memory) = selection(&sp, Selection::Rung0);
    let text = regime_with_count(&sp, &MemoryMode::TextToImage, &sel, &memory, 1);
    let request = regime_with_count(&sp, &MemoryMode::TextToImage, &sel, &memory, 2);
    assert_eq!((text.count, request.count), (1, 2));
    let (measured_text, at_anchor) = captured_source_profile(&snapshot, &text, &memory);
    let (measured, profile) = captured_source_profile(&snapshot, &request, &memory);
    let starts = |profile: &IosPricing| {
        profile.observations[0]
            .metadata
            .iter()
            .filter(|mark| mark.provider == "sdxl" && mark.kind == "serial-output-start")
            .map(|mark| {
                (
                    mark.fields["output"],
                    mark.fields["completed_latents"],
                    mark.fields["completed_latent_bytes"],
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(starts(&at_anchor), [(0, 0, 0)]);
    let serial = starts(&profile);
    assert_eq!(serial.len(), 2);
    assert_eq!(serial[0], (0, 0, 0));
    assert_eq!((serial[1].0, serial[1].1), (1, 1));
    assert!(
        serial[1].2 > 0,
        "second output must retain an actual first latent owner"
    );
    let forwards = |profile: &IosPricing| {
        profile.observations[0]
            .metadata
            .iter()
            .filter(|mark| mark.provider == "sdxl" && mark.kind == "forward-input-before-cast")
            .count()
    };
    assert!(forwards(&at_anchor) > 0);
    assert_eq!(forwards(&profile), 2 * forwards(&at_anchor));
    assert_ne!(
        profile.windows, at_anchor.windows,
        "count two must change source work/ownership"
    );
    let anchor_inputs = endpoint_inputs(&snapshot, &text);
    let request_inputs = endpoint_inputs(&snapshot, &request);
    let bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &anchor_inputs,
        &text,
        &at_anchor,
        &request_inputs,
        &request,
        &profile,
    )
    .unwrap();
    let derived = derive_certified(
        &measured_text.anchor(area(PX as u32, PX as u32, 1)),
        &at_anchor,
        &profile,
        &bounds,
    )
    .unwrap();
    assert_certified_native_run(&measured, &derived, "q4 resident serial count two");
    let single_bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &anchor_inputs,
        &text,
        &at_anchor,
        &anchor_inputs,
        &text,
        &at_anchor,
    )
    .unwrap();
    let single = derive_certified(
        &measured_text.anchor(area(PX as u32, PX as u32, 1)),
        &at_anchor,
        &at_anchor,
        &single_bounds,
    )
    .unwrap();
    assert_certified_native_run(&measured_text, &single, "q4 resident serial count one");
    assert!(
        derived.demand > single.demand,
        "count-two certified cumulative upper {} must exceed count-one {}",
        derived.demand,
        single.demand
    );
    let mut generation = generation_request(Route::Text, PX as u32, PX as u32, &memory);
    generation.count = 2;
    assert!(
        Regime::from_request(&IosPricingRequest {
            provider: "sdxl",
            spec: &sp,
            mode: &MemoryMode::TextToImage,
            geometry: MemoryGeometry {
                width: PX as u32,
                height: PX as u32,
                batch: 1,
                frames: 1,
                reference_count: generation.memory_reference_count(),
            },
            selection: &sel,
            execution: &memory,
            generation: Some(&generation),
            transform: None,
            depth: None,
            token_window: 0,
        })
        .is_err(),
        "count-two execution cannot be admitted under batch-one geometry"
    );
    completion.body_completed = true;

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::serial_count_two_uses_real_geometry_and_complete_native_bounds");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::serial_count_two_uses_real_geometry_and_complete_native_bounds");
}

/// Payload-free source replay resolves the same work as the actual text request, but never
/// manufactures observations. Actual captured marks are retained unchanged for certification.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn payload_free_text_anchor_replay_preserves_native_certificates() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::payload_free_text_anchor_replay_preserves_native_certificates",
    ) else {
        return;
    };
    let snapshot = Snapshot::packed(4);
    let sp = spec(&snapshot, Route::Text, Selection::Rung0);
    let (sel, memory) = selection(&sp, Selection::Rung0);
    let generation = generation_request(Route::Text, PX as u32, PX as u32, &memory);
    let mode = MemoryMode::TextToImage;
    let mut req = IosPricingRequest {
        provider: "sdxl",
        spec: &sp,
        mode: &mode,
        geometry: MemoryGeometry {
            width: PX as u32,
            height: PX as u32,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        selection: &sel,
        execution: &memory,
        generation: Some(&generation),
        transform: None,
        depth: None,
        token_window: 0,
    };
    let mut actual = Regime::from_request(&req).unwrap();
    let identity = |regime: &Regime| {
        (
            regime.route,
            regime.sampling.route(),
            regime.count,
            regime.has_init,
            regime.width,
            regime.height,
            regime.staged,
            regime.cfg_batching,
            regime.context_tokens,
            regime.ip_scale.to_bits(),
            regime.reference_image.is_none(),
            regime.tiling.is_none(),
        )
    };
    let token = actual.sampling.recipe().canonical_token();
    let precedence = crate::ios_pricing::resolve_bound_side_for_test(
        &req,
        Some(crate::ios_pricing::SdxlReplay {
            token: "not-an-sdxl-token",
            true_reference_count: usize::MAX,
            mask_count: usize::MAX,
            control_count: usize::MAX,
            source: None,
        }),
    )
    .unwrap();
    assert_eq!(identity(&precedence), identity(&actual));
    assert_eq!(precedence.sampling.recipe().canonical_token(), token);
    req.generation = None;
    let missing = crate::ios_pricing::resolve_bound_side_for_test(&req, None);
    assert!(
        matches!(missing, Err(mlx_gen::Error::Unsupported(ref message))
            if message == "SDXL native certificate: payload-free side requires an SDXL sampling replay token"),
        "payload-free anchor without replay must return the exact source refusal"
    );
    let mut replayed = crate::ios_pricing::resolve_bound_side_for_test(
        &req,
        Some(crate::ios_pricing::SdxlReplay {
            token: &token,
            true_reference_count: 0,
            mask_count: 0,
            control_count: 0,
            source: None,
        }),
    )
    .unwrap();
    assert_eq!(identity(&replayed), identity(&actual));
    assert_eq!(replayed.sampling.recipe().canonical_token(), token);
    assert!(actual.reference_image.is_none() && actual.tiling.is_none());
    // Both production resolutions above use the ordinary CLIP window. Only the tiny fixture's
    // loaded text architecture below substitutes its real eight-token table, identically on both.
    actual.context_tokens = TOKENS;
    replayed.context_tokens = TOKENS;
    let (measured, at_anchor) = captured_source_profile(&snapshot, &actual, &memory);
    let actual_inputs = endpoint_inputs(&snapshot, &actual);
    let mut replayed_inputs = endpoint_inputs(&snapshot, &replayed);
    let mut replayed_profile =
        crate::ios_pricing::source_pricing_for_test(&mut replayed_inputs, &replayed).unwrap();
    assert_eq!(replayed_profile.phases, at_anchor.phases);
    assert_eq!(replayed_profile.windows, at_anchor.windows);
    assert_eq!(replayed_profile.absent_phases, at_anchor.absent_phases);
    assert!(replayed_profile.observations.is_empty());
    replayed_profile.observations = at_anchor.observations.clone();
    replayed_profile.request_baseline = at_anchor.request_baseline.clone();
    let actual_bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &actual_inputs,
        &actual,
        &at_anchor,
        &actual_inputs,
        &actual,
        &at_anchor,
    )
    .unwrap();
    let replayed_bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &replayed_inputs,
        &replayed,
        &replayed_profile,
        &actual_inputs,
        &actual,
        &at_anchor,
    )
    .unwrap();
    assert_eq!(replayed_bounds, actual_bounds);
    let derived = derive_certified(
        &measured.anchor(area(PX as u32, PX as u32, 1)),
        &replayed_profile,
        &at_anchor,
        &replayed_bounds,
    )
    .unwrap();
    assert_certified_native_run(&measured, &derived, "q4 resident payload-free text anchor");
    completion.body_completed = true;

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::payload_free_text_anchor_replay_preserves_native_certificates");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::payload_free_text_anchor_replay_preserves_native_certificates");
}

/// Every listed exception is a mode key this crate prices.
#[test]
fn every_listed_non_additive_mode_is_a_priced_mode() {
    for tier in ["bf16", "q8", "q4"] {
        for listed in ios_non_additive_modes(tier) {
            assert!(
                modes().iter().any(|(mode, _, _)| mode.as_key() == *listed),
                "{listed} is not a derived SDXL mode"
            );
        }
    }
}

/// Honest Control certifies; deleting its actual resident component from C_req must fail
/// source-profile validation rather than manufacture a lower certified demand.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn dropping_the_controlnet_from_c_req_fails_the_proof() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::dropping_the_controlnet_from_c_req_fails_the_proof",
    ) else {
        return;
    };
    let snapshot = Snapshot::packed(4);
    for which in [Selection::Optimized, Selection::Rung0] {
        let text_spec = spec(&snapshot, Route::Text, which);
        let (sel, memory) = selection(&text_spec, which);
        let text = regime(&text_spec, &MemoryMode::TextToImage, &sel, &memory);
        let control_spec = spec(&snapshot, Route::Control, which);
        let (control_sel, control_memory) = selection(&control_spec, which);
        let control = regime(
            &control_spec,
            &MemoryMode::Other("control".into()),
            &control_sel,
            &control_memory,
        );
        let (measured_text, at_anchor) = captured_source_profile(&snapshot, &text, &memory);
        let (measured, profile) = captured_source_profile(&snapshot, &control, &control_memory);
        let anchor_inputs = endpoint_inputs(&snapshot, &text);
        let request_inputs = endpoint_inputs(&snapshot, &control);
        let bounds = crate::ios_pricing::certified_request_bounds_for_test(
            &anchor_inputs,
            &text,
            &at_anchor,
            &request_inputs,
            &control,
            &profile,
        )
        .unwrap();
        let derived = derive_certified(
            &measured_text.anchor(area(PX as u32, PX as u32, 1)),
            &at_anchor,
            &profile,
            &bounds,
        )
        .unwrap();
        assert_certified_native_run(&measured, &derived, &format!("q4 {which:?} Control"));
        let mutated = crate::quantum::with_backend(Backend::Metal, || {
            let mut priced = Priced::build(&request_inputs).unwrap();
            assert!(!priced.control_bytes.is_empty());
            for (load, resident) in &mut priced.control_bytes {
                load.resident = 0;
                *resident = 0;
            }
            priced.price(&control).unwrap().0
        });
        assert_ne!(
            mutated.windows, profile.windows,
            "ControlNet removal must change the ledger"
        );
        assert!(
            crate::ios_pricing::certified_request_bounds_for_test(
                &anchor_inputs,
                &text,
                &at_anchor,
                &request_inputs,
                &control,
                &mutated,
            )
            .is_err(),
            "{which:?}: dropping the ControlNet from C_req must fail source validation"
        );
    }
    completion.body_completed = true;

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::dropping_the_controlnet_from_c_req_fails_the_proof");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::dropping_the_controlnet_from_c_req_fails_the_proof");
}

// ---------------------------------------------------------------------------------------------
// The five SDXL-family catalog routes share ONE architecture.
// ---------------------------------------------------------------------------------------------

/// `mlx-gen-ios-catalog::sdxl::SDXL_FAMILY_ROUTES`.
const ROUTES: [&str; 5] = [
    "sdxl",
    "illustrious_xl_v1",
    "illustrious_xl_v2",
    "realvisxl",
    "realvisxl_lightning",
];

/// Every safetensors tensor under `root` as `(component directory, name) → (dtype, shape)` — the
/// content `mlx-gen-ios-catalog::layout::architecture_digest` hashes for each role.
fn tensor_layout(
    root: &std::path::Path,
) -> std::collections::BTreeMap<(String, String), (String, Vec<usize>)> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "safetensors") {
                let component = path
                    .parent()
                    .unwrap()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                for h in
                    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path).unwrap()
                {
                    out.insert(
                        (component.clone(), h.name),
                        (h.dtype.as_str().to_owned(), h.shape),
                    );
                }
            }
        }
    }
    out
}

/// A copy of `s` whose every tensor holds DIFFERENT values at the same name, dtype and shape — a
/// sibling route's checkpoint (fine-tunes of one architecture differ only in their values).
fn sibling(s: &Snapshot) -> Snapshot {
    let dir = tempfile::tempdir().unwrap();
    let mut stack = vec![s.root.clone()];
    while let Some(src) = stack.pop() {
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let to = dir.path().join(path.strip_prefix(&s.root).unwrap());
            if path.is_dir() {
                std::fs::create_dir_all(&to).unwrap();
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "safetensors") {
                let w = Weights::from_file(&path).unwrap();
                let keys: Vec<String> = w.keys().map(str::to_owned).collect();
                let map: HashMap<String, Array> = keys
                    .into_iter()
                    .map(|k| {
                        let a = w.get(&k).unwrap();
                        let shifted = if a.dtype() == Dtype::Uint32 {
                            a.add(Array::from_slice(&[1u32], &[1])).unwrap()
                        } else {
                            a.multiply(Array::from_f32(0.5))
                                .unwrap()
                                .as_dtype(a.dtype())
                                .unwrap()
                        };
                        shifted.eval().unwrap();
                        (k, shifted)
                    })
                    .collect();
                save(&Weights::from_map(map), &to);
            } else {
                std::fs::copy(&path, &to).unwrap();
            }
        }
    }
    Snapshot {
        root: dir.path().to_path_buf(),
        _dir: dir,
        bits: s.bits,
        text_quantize: s.text_quantize,
        context: s.context,
        unet: s.unet.clone(),
        te1: s.te1.clone(),
        te2: s.te2.clone(),
        vae: s.vae.clone(),
        vision: s.vision.clone(),
        resampler: s.resampler.clone(),
    }
}

/// The five routes differ only by checkpoint VALUES, which nothing on SDXL's iOS memory path reads:
/// two snapshots of one architecture with different values have the identical tensor layout (what
/// the catalog's architecture digest hashes), and every route's iOS load spec over either selects
/// the identical execution and prices identically, for every mode at both selections and tiers.
/// (The route enters the spec only as `resolved_route`, which this crate never reads; Lightning's
/// few-step sampler is a request choice — CFG-free and txt2img/control-only — not a load.)
#[test]
fn the_five_sdxl_routes_share_one_architecture() {
    if !cpu_child("pricing::additivity::the_five_sdxl_routes_share_one_architecture") {
        return;
    }
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        let other = sibling(&snap);
        assert_eq!(
            tensor_layout(&snap.root),
            tensor_layout(&other.root),
            "{tier}"
        );
        let unet = "unet/diffusion_pytorch_model.safetensors";
        assert_ne!(
            std::fs::read(snap.file(unet)).unwrap(),
            std::fs::read(other.file(unet)).unwrap(),
            "{tier}: the sibling must hold different values"
        );
        let all = std::iter::once((MemoryMode::TextToImage, Route::Text, 0)).chain(modes());
        for (mode, route, _) in all {
            for which in [Selection::Optimized, Selection::Rung0] {
                let mut first = None;
                for s in [&snap, &other] {
                    for name in ROUTES {
                        let mut spec = spec(s, route, which);
                        spec.resolved_route = Some(name.into());
                        let (selection, memory) = selection(&spec, which);
                        let r = regime(&spec, &mode, &selection, &memory);
                        let cell = (selection.strategy, memory, pricing(s, &r));
                        match &first {
                            None => first = Some(cell),
                            Some(first) => assert_eq!(
                                first,
                                &cell,
                                "{tier} {} {which:?}: route {name} differs",
                                mode.as_key()
                            ),
                        }
                    }
                }
            }
        }
    }
}

/// SOURCE-ONLY WIP: iOS boundary casts must preserve the VAE RGB contract: max 3/255,
/// mean 0.25/255. The corresponding pre-RGB VAE tensor thresholds are 6/255 and 0.5/255,
/// because RGB = (decoded + 1)/2. There is NO universal latent-space equivalent: the decoder
/// is nonlinear, so testing only latent rounding error cannot prove this requirement.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn ios_init_cast_preserves_decoded_rgb() {
    if !cpu_child("pricing::additivity::ios_init_cast_preserves_decoded_rgb") {
        return;
    }
    let snap = Snapshot::new(2);
    let files = Files::of(&snap);
    let (_, _, vae, _) = files.heavy(Route::Text).unwrap();
    let img = image(PX, PX, 3);
    let original = (
        crate::test_device::record_model_inference_device(),
        crate::pipeline::encode_init_latents_tiled(&vae, &img, PX as u32, PX as u32, None, None),
    )
        .1
        .unwrap();
    let cast = with_cpu_diagnostic_headroom(PLENTY, || crate::pipeline::ios_unet_input(&original))
        .unwrap();
    assert_eq!(cast.dtype(), original.dtype());
    assert_eq!(
        bits(&cast),
        bits(&original),
        "boundary preserves actual init values, not just RGB tolerance"
    );
    let decode = |z: &Array| {
        (
            crate::test_device::record_model_inference_device(),
            crate::pipeline::decode_image_tiled(&vae, z, None, None, None),
        )
            .1
            .unwrap()
    };
    let (a, b) = (decode(&original), decode(&cast));
    let differences: Vec<f64> = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(a, b)| a.abs_diff(*b) as f64 / 255.0)
        .collect();
    let max = differences.iter().copied().fold(0.0, f64::max);
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    eprintln!("iOS init cast decoded RGB: max {max} mean {mean}");
    assert!(max <= 3.0 / 255.0 && mean <= 0.25 / 255.0);

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::ios_init_cast_preserves_decoded_rgb");
}

/// Real-request coverage of every checked execution quantum, including upstream lazy allocations
/// it materializes. The stalled isolated proofs remain the stronger bound on individual kernels.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn every_guard_covers_its_real_request_quantum() {
    if !cpu_child("pricing::additivity::every_guard_covers_its_real_request_quantum") {
        return;
    }
    let under = Rc::new(RefCell::new(
        std::collections::BTreeMap::<String, (u64, u64)>::new(),
    ));
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        for (mode, route, _) in
            std::iter::once((MemoryMode::TextToImage, Route::Text, 0)).chain(modes())
        {
            for which in [Selection::Optimized, Selection::Rung0] {
                let sp = spec(&snap, route, which);
                let (sel, memory) = selection(&sp, which);
                let r = regime(&sp, &mode, &sel, &memory);
                let files = Files::of(&snap);
                run_all_once(&files, &r, &memory, &[]);
                let sink = Rc::clone(&under);
                let mut open: Option<(String, u64, u64)> = None;
                let key = format!("{tier} {which:?} {}", mode.as_key());
                eprintln!("GUARD CELL {key}");
                with_cpu_diagnostic_headroom(PLENTY, || {
                    with_cpu_diagnostic_observer(
                        move |event| match event {
                            QuantumEvent::Checked { label, scratch } => {
                                open = Some((
                                    label.clone(),
                                    *scratch,
                                    mlx_rs::memory::get_active_memory() as u64,
                                ));
                                mlx_rs::memory::reset_peak_memory();
                            }
                            QuantumEvent::Synchronized { label, .. } => {
                                if let Some((checked, scratch, base)) = open.take() {
                                    if checked == *label {
                                        let growth = (mlx_rs::memory::get_peak_memory() as u64)
                                            .saturating_sub(base);
                                        eprintln!("GUARD PEAK {key} {label}: measured {growth} checked {scratch}");
                                        if growth > scratch {
                                            let name = format!("{key} {label}");
                                            let mut sink = sink.borrow_mut();
                                            let entry = sink.entry(name).or_insert((0, scratch));
                                            if growth > entry.0 {
                                                *entry = (growth, scratch);
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        || run(&files, &r, &memory),
                    )
                })
                .unwrap();
            }
        }
    }
    for (name, (growth, scratch)) in under.borrow().iter() {
        eprintln!("GUARD UNDER {name}: measured {growth} > checked {scratch}");
    }
    assert!(
        under.borrow().is_empty(),
        "under-priced real-request quantum"
    );

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::every_guard_covers_its_real_request_quantum");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::every_guard_covers_its_real_request_quantum");
}

/// The exact native CI regression: bf16 / optimized / img2img. Keep this single production
/// lifecycle comparison ordinary even though the full 48-lifecycle matrix is selected-only.
/// It exercises real init encode, sampler, U-Net forward and RGB decode, not a cast-array proxy.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn optimized_bf16_img2img_preserves_full_request_rgb() {
    if !cpu_child("pricing::additivity::optimized_bf16_img2img_preserves_full_request_rgb") {
        return;
    }
    let snap = Snapshot::new(2);
    let sp = spec(&snap, Route::Reference, Selection::Optimized);
    let (sel, memory) = selection(&sp, Selection::Optimized);
    let r = regime(&sp, &MemoryMode::ImageToImage, &sel, &memory);
    let files = Files::of(&snap);
    let host = (
        crate::test_device::record_model_inference_device(),
        run_image(&files, &r, &memory),
    )
        .1
        .unwrap();
    let ios = with_cpu_diagnostic_headroom(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            run_image(&files, &r, &memory),
        )
            .1
    })
    .unwrap();
    assert_eq!(
        (host.width, host.height, host.pixels.len()),
        (ios.width, ios.height, ios.pixels.len())
    );
    assert!(!host.pixels.is_empty());
    let diffs: Vec<f64> = host
        .pixels
        .iter()
        .zip(&ios.pixels)
        .map(|(a, b)| a.abs_diff(*b) as f64 / 255.0)
        .collect();
    let max = diffs.iter().copied().fold(0.0, f64::max);
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    assert!(
        max <= 3.0 / 255.0 && mean <= 0.25 / 255.0,
        "bf16 Optimized image_to_image: max {max}, mean {mean}"
    );

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::optimized_bf16_img2img_preserves_full_request_rgb");
}

/// Full request output parity, not just cast-array error: init/noise/inpaint and ControlNet residual
/// casts must keep the decoded RGB image within max 3/255 and mean .25/255 of the unchanged host
/// path. The nonlinear VAE has no universal latent-error equivalent of an RGB error bound.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn ios_boundary_casts_preserve_full_request_rgb() {
    if !cpu_child("pricing::additivity::ios_boundary_casts_preserve_full_request_rgb") {
        return;
    }
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        for (mode, route, _) in modes() {
            for which in [Selection::Optimized, Selection::Rung0] {
                let sp = spec(&snap, route, which);
                let (sel, memory) = selection(&sp, which);
                let r = regime(&sp, &mode, &sel, &memory);
                let files = Files::of(&snap);
                let host = (
                    crate::test_device::record_model_inference_device(),
                    run_image(&files, &r, &memory),
                )
                    .1
                    .unwrap();
                let ios = with_cpu_diagnostic_headroom(PLENTY, || {
                    (
                        crate::test_device::record_model_inference_device(),
                        run_image(&files, &r, &memory),
                    )
                        .1
                })
                .unwrap();
                let diffs: Vec<f64> = host
                    .pixels
                    .iter()
                    .zip(&ios.pixels)
                    .map(|(a, b)| a.abs_diff(*b) as f64 / 255.0)
                    .collect();
                let max = diffs.iter().copied().fold(0.0, f64::max);
                let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
                eprintln!(
                    "RGB parity {tier} {which:?} {}: max {max:.8} mean {mean:.8}",
                    mode.as_key()
                );
                assert!(
                    max <= 3.0 / 255.0 && mean <= 0.25 / 255.0,
                    "{tier} {which:?} {}: max {max}, mean {mean}",
                    mode.as_key()
                );
            }
        }
    }

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::ios_boundary_casts_preserve_full_request_rgb");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::ios_boundary_casts_preserve_full_request_rgb");
}

/// Recorded IP source facts travel with payload-free catalog anchors. The same real
/// capture is used on both sides; replay never synthesizes execution observations.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn payload_free_ip_anchor_replay_preserves_native_certificates() {
    let Some(mut completion) = completed_cpu_child(
        "pricing::additivity::payload_free_ip_anchor_replay_preserves_native_certificates",
    ) else {
        return;
    };
    use crate::sampling::{SamplingSourceFacts, SamplingSourceView, SourceRequestBudget};
    let snapshot = Snapshot::packed(4);
    let sp = spec(&snapshot, Route::IpAdapter, Selection::Rung0);
    let (sel, memory) = selection(&sp, Selection::Rung0);
    let generation = generation_request(Route::IpAdapter, PX as u32, PX as u32, &memory);
    let mut budget = SourceRequestBudget::bounded(1 << 20);
    let source =
        SamplingSourceFacts::capture(SamplingSourceView::from_request(&generation), &mut budget)
            .unwrap();
    source.validate_roles().unwrap();
    let work = source.work();
    assert_eq!(
        (work.reference_count, work.mask_count, work.control_count),
        (1, 0, 0)
    );
    assert!(work.reference_image.is_some());
    let mode = MemoryMode::Other("ip_adapter".into());
    let mut req = IosPricingRequest {
        provider: "sdxl",
        spec: &sp,
        mode: &mode,
        geometry: MemoryGeometry {
            width: PX as u32,
            height: PX as u32,
            batch: 1,
            frames: 1,
            reference_count: generation.memory_reference_count(),
        },
        selection: &sel,
        execution: &memory,
        generation: Some(&generation),
        transform: None,
        depth: None,
        token_window: 0,
    };
    let mut actual = Regime::from_request(&req).unwrap();
    let token = actual.sampling.recipe().canonical_token();
    req.generation = None;
    let missing_source = crate::ios_pricing::resolve_bound_side_for_test(
        &req,
        Some(crate::ios_pricing::SdxlReplay {
            token: &token,
            true_reference_count: 1,
            mask_count: 0,
            control_count: 0,
            source: None,
        }),
    );
    assert!(
        matches!(missing_source, Err(mlx_gen::Error::Unsupported(ref message))
        if message == "sdxl IP pricing needs actual ReferenceImageFacts; missing legacy source facts are not crop geometry")
    );
    let mut replayed = crate::ios_pricing::resolve_bound_side_for_test(
        &req,
        Some(crate::ios_pricing::SdxlReplay {
            token: &token,
            true_reference_count: 1,
            mask_count: 0,
            control_count: 0,
            source: Some(&source),
        }),
    )
    .unwrap();
    assert_eq!(replayed.route, Route::IpAdapter);
    assert_eq!(replayed.reference_image, actual.reference_image);
    assert_eq!(replayed.sampling.recipe().canonical_token(), token);
    assert_eq!(replayed.cfg_batching, actual.cfg_batching);
    assert_eq!(
        (replayed.width, replayed.height, replayed.count),
        (actual.width, actual.height, actual.count)
    );
    actual.context_tokens = TOKENS;
    replayed.context_tokens = TOKENS;
    let (measured, actual_profile) = captured_source_profile(&snapshot, &actual, &memory);
    let actual_inputs = endpoint_inputs(&snapshot, &actual);
    let mut replayed_inputs = endpoint_inputs(&snapshot, &replayed);
    let mut replayed_profile =
        crate::ios_pricing::source_pricing_for_test(&mut replayed_inputs, &replayed).unwrap();
    assert_eq!(replayed_profile.phases, actual_profile.phases);
    assert_eq!(replayed_profile.windows, actual_profile.windows);
    assert_eq!(replayed_profile.absent_phases, actual_profile.absent_phases);
    assert!(replayed_profile.observations.is_empty());
    replayed_profile.observations = actual_profile.observations.clone();
    replayed_profile.request_baseline = actual_profile.request_baseline.clone();
    let actual_bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &actual_inputs,
        &actual,
        &actual_profile,
        &actual_inputs,
        &actual,
        &actual_profile,
    )
    .unwrap();
    let replayed_bounds = crate::ios_pricing::certified_request_bounds_for_test(
        &replayed_inputs,
        &replayed,
        &replayed_profile,
        &actual_inputs,
        &actual,
        &actual_profile,
    )
    .unwrap();
    assert_eq!(replayed_bounds, actual_bounds);
    let derived = derive_certified(
        &measured.anchor(area(PX as u32, PX as u32, 1)),
        &replayed_profile,
        &actual_profile,
        &replayed_bounds,
    )
    .unwrap();
    assert_certified_native_run(&measured, &derived, "q4 resident recorded-source IP replay");
    completion.body_completed = true;

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::pricing::additivity::payload_free_ip_anchor_replay_preserves_native_certificates");
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::pricing::additivity::payload_free_ip_anchor_replay_preserves_native_certificates");
}
