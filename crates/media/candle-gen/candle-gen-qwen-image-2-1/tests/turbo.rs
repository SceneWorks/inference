//! Qwen-Image 2.1 **Turbo** (sc-25901) on a Turbo-shaped miniature snapshot — the committed tiny
//! snapshot rewritten in a temp dir by [`turbo_snapshot`] into what `Qwen/Qwen-Image-2.1-Turbo`
//! ships differently: `sample_sigmas` in `model_index.json`, a static scheduler config, a bf16
//! VAE and a single-file text tower.
//!
//! What is claimed, and on what:
//!
//! * **registration** — `qwen_image_2_1_turbo` sits beside `qwen_image_2_1` in the provider
//!   catalog with its own memory registrations and no trainer, and loads and renders through the
//!   same `provider_registry().load(id, spec)` path the worker uses;
//! * **the schedule the sampler receives** — asserted on the *render*, not on a helper: the
//!   generator's pixels equal a render assembled here from the public pipeline pieces over a
//!   **literal** sigma list, and differ from the same assembly over any other list;
//! * **the VAE's dtype** — asserted on a weight tensor of the loaded autoencoder
//!   ([`QwenImage21Vae::weight_dtype`]), against a file whose stored dtype is checked to be bf16;
//! * **the mismatch refusals** — each snapshot under the other id, through `load` and through the
//!   memory registration.
//!
//! Every device here is `candle_gen::default_device()`, the one `load` itself uses: the CPU (f32)
//! in the default lane, the GPU (bf16) under `--features cuda` / `--features metal`.

use candle_core::{DType, Device};
use candle_gen::gen_core::{
    CancelFlag, Error as CoreError, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    OffloadPolicy, Progress, WeightsSource,
};
use candle_gen_qwen_image_2_1::{
    create_noise, decode_rgb, denoise, load_scheduler_config, load_text_encoder, load_tokenizer,
    load_transformer, load_turbo_schedule, load_vae, loader, scheduler, system_prompt_drop_count,
    DenoiseInputs,
};

use crate::common::{
    assert_close, read_json, safetensors_files, stored_dtypes, tiny_snapshot, turbo_snapshot,
    turbo_snapshot_with, TURBO_SAMPLE_SIGMAS,
};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const PROMPT: &str = "a red fox in the forest";
const SEED: u64 = 42;
const SIZE: u32 = 32;

/// The eight released sigmas and the trailing `0`, as f32 literals — what an 8-step Turbo render
/// must denoise over.
const SAVED_PLUS_ZERO: [f32; 9] = [
    1.0, 0.978_453, 0.954_18, 0.926_626, 0.895_08, 0.845_148, 0.704_534, 0.414_568, 0.0,
];

/// The 12-step resample of the eight released sigmas (knots `s_0..s_7, 0` at `u_j = j/8`,
/// piecewise-linear at `u = i/12`; f64, rounded to f32), as literals computed outside this crate.
/// The MLX twin pins the same thirteen values.
const TWELVE_STEP_RESAMPLE: [f32; 13] = [
    1.0,
    0.985_635_34,
    0.970_362,
    0.954_18,
    0.935_810_7,
    0.916_110_7,
    0.895_08,
    0.861_792,
    0.798_276_66,
    0.704_534,
    0.511_223_3,
    0.276_378_66,
    0.0,
];

fn spec(root: &std::path::Path, policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy)
}

fn load(id: &str, root: &std::path::Path, policy: OffloadPolicy) -> Box<dyn Generator> {
    candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(id, &spec(root, policy))
        .unwrap_or_else(|e| panic!("{id} loads through the catalog path: {e}"))
}

fn request(steps: Option<u32>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: SIZE,
        height: SIZE,
        steps,
        seed: Some(SEED),
        ..Default::default()
    }
}

fn pixels(generator: &dyn Generator, req: &GenerationRequest) -> Vec<u8> {
    match generator.generate(req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.remove(0).pixels
        }
        other => panic!("expected images, got {other:?}"),
    }
}

/// One text-to-image render of [`PROMPT`] at [`SEED`] assembled from the crate's public pipeline
/// pieces over an **explicit** sigma list — the independent side of the schedule assertions. The
/// generator is free to choose its schedule; this is not.
fn render_over(root: &std::path::Path, sigmas: &[f32]) -> Vec<u8> {
    let device = candle_gen::default_device().unwrap();
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root, &device).unwrap();
    let transformer = load_transformer(root, &device).unwrap();
    let vae = load_vae(root, &device).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = encoder.encode_prompt(&tokenizer, PROMPT, drop).unwrap();
    let seed = candle_gen::image_seed(SEED, 0);
    let noise = create_noise(seed, SIZE, SIZE, transformer.config().in_channels, &device).unwrap();
    let cancel = CancelFlag::new();
    let latents = denoise(
        DenoiseInputs {
            transformer: &transformer,
            sigmas,
            latents: noise,
            prompt_embeds: &prompt,
            negative_embeds: None,
            true_cfg_scale: 1.0,
            width: SIZE,
            height: SIZE,
            sampler: None,
            seed,
            cancel: &cancel,
            references: None,
        },
        &mut |_| {},
    )
    .unwrap();
    decode_rgb(&vae, &latents, SIZE, SIZE, None, Some(&cancel))
        .unwrap()
        .pixels
}

/// The fixture is what it says it is — otherwise every claim below is about a base snapshot.
#[test]
fn the_fixture_is_turbo_shaped_where_the_base_snapshot_is_not() {
    let turbo = turbo_snapshot();
    let (turbo, base) = (turbo.path(), tiny_snapshot());
    let vae = "vae/diffusion_pytorch_model.safetensors";
    assert_eq!(stored_dtypes(&turbo.join(vae)), ["BF16"]);
    assert_eq!(stored_dtypes(&base.join(vae)), ["F32"]);
    assert_eq!(
        safetensors_files(&turbo.join("text_encoder")),
        ["model.safetensors"]
    );
    assert_eq!(
        read_json(&turbo.join("model_index.json"))["sample_sigmas"],
        serde_json::json!(TURBO_SAMPLE_SIGMAS)
    );
    assert!(read_json(&base.join("model_index.json"))
        .get("sample_sigmas")
        .is_none());
    let scheduler = read_json(&turbo.join("scheduler/scheduler_config.json"));
    assert_eq!(scheduler["use_dynamic_shifting"], serde_json::json!(false));
    assert_eq!(scheduler["shift"], serde_json::json!(1.0));
    assert!(scheduler["shift_terminal"].is_null());
    // The transformer is the base one, byte for byte: one architecture, one config.
    for file in [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
    ] {
        assert_eq!(
            std::fs::read(turbo.join(file)).unwrap(),
            std::fs::read(base.join(file)).unwrap(),
            "{file}"
        );
    }
}

/// Both ids are registered, in order, each with its generator and all three memory surfaces; the
/// trainer exists under the base id only.
///
/// *Mutation that reds this:* dropping `TURBO_REGISTRATION`, or any one of the three Turbo memory
/// registrations, from the crate's `register_providers`.
#[test]
fn turbo_is_registered_beside_base_with_its_own_memory_surfaces_and_no_trainer() {
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generators: Vec<_> = registry.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(generators, [BASE, TURBO]);
    let trainers: Vec<_> = registry.trainers().map(|r| (r.descriptor)().id).collect();
    assert_eq!(trainers, [BASE], "no trainer under the Turbo id");
    let strategies: Vec<_> = registry
        .memory_strategy_registrations()
        .map(|r| r.provider_id)
        .collect();
    assert_eq!(strategies, [BASE, TURBO]);
    let fixtures: Vec<_> = registry
        .memory_contract_fixture_registrations()
        .map(|r| r.provider_id)
        .collect();
    assert_eq!(fixtures, [BASE, TURBO]);
    let behaviors: Vec<_> = registry
        .memory_behavior_registrations()
        .map(|r| r.provider_id)
        .collect();
    assert_eq!(behaviors, [BASE, TURBO]);
    assert!(registry.descriptor_conformance_errors().is_empty());

    // The loaded generator publishes a contract under ITS id, priced from ITS snapshot, and the
    // registry's own contract seam agrees with it.
    let turbo = turbo_snapshot();
    let generator = load(TURBO, turbo.path(), OffloadPolicy::Resident);
    assert_eq!(generator.descriptor().id, TURBO);
    let loaded = generator.memory_strategy_contract().expect("a contract");
    assert_eq!(loaded.provider_id, TURBO);
    let registered = registry
        .memory_strategy_contract(TURBO, &spec(turbo.path(), OffloadPolicy::Resident))
        .unwrap()
        .expect("the Turbo id adopted the shared memory contract");
    assert_eq!(registered.provider_id, TURBO);
    assert_eq!(
        registered.calibration.as_ref().map(|c| &c.fingerprint),
        loaded.calibration.as_ref().map(|c| &c.fingerprint)
    );
    let base = registry
        .memory_strategy_contract(BASE, &spec(&tiny_snapshot(), OffloadPolicy::Resident))
        .unwrap()
        .expect("the base contract");
    assert_ne!(
        registered.calibration.as_ref().map(|c| &c.fingerprint),
        base.calibration.as_ref().map(|c| &c.fingerprint),
        "a record built against one id must not satisfy the other's handshake"
    );
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: PROMPT.to_owned(),
        width: SIZE,
        height: SIZE,
        steps: 3,
        seed: SEED,
        cancel_steps: 6,
    }
}

/// The whole gen-core generator contract — validate honesty, progress, typed cancellation, seeded
/// determinism — on the Turbo fixture, both residency policies, through the catalog load path.
#[test]
fn gen_core_conformance_on_the_turbo_fixture_resident() {
    let turbo = turbo_snapshot();
    gen_core_testkit::conformance(
        || load(TURBO, turbo.path(), OffloadPolicy::Resident),
        &profile(),
    );
}

#[test]
fn gen_core_conformance_on_the_turbo_fixture_sequential() {
    let turbo = turbo_snapshot();
    gen_core_testkit::conformance(
        || load(TURBO, turbo.path(), OffloadPolicy::Sequential),
        &profile(),
    );
    let generator = load(TURBO, turbo.path(), OffloadPolicy::Sequential);
    gen_core_testkit::check_alpha_output_honesty(generator.as_ref(), &profile()).unwrap();
}

/// A request that omits `steps` runs one step per sigma **the snapshot saves** — eight for the
/// released list, five for a fixture that saves five — and emits RGB8 at the requested size.
///
/// *Mutation that reds this:* defaulting the Turbo step count to a constant (8, or the base 40)
/// instead of the saved list's length.
#[test]
fn an_unset_step_count_is_the_length_of_the_snapshots_saved_list() {
    for (sigmas, want) in [
        (TURBO_SAMPLE_SIGMAS.to_vec(), 8u32),
        (vec![1.0, 0.9, 0.7, 0.4, 0.1], 5),
    ] {
        let turbo = turbo_snapshot_with(&sigmas);
        let generator = load(TURBO, turbo.path(), OffloadPolicy::Resident);
        let mut events = Vec::new();
        let out = generator
            .generate(&request(None), &mut |p| events.push(p))
            .unwrap();
        let steps: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|p| match p {
                Progress::Step { current, total } => Some((*current, *total)),
                _ => None,
            })
            .collect();
        assert_eq!(
            steps,
            (1..=want).map(|c| (c, want)).collect::<Vec<_>>(),
            "{sigmas:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|p| matches!(p, Progress::Decoding))
                .count(),
            1
        );
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        assert_eq!((images[0].width, images[0].height), (SIZE, SIZE));
        assert_eq!(images[0].pixels.len(), (SIZE * SIZE * 3) as usize);
    }
}

/// **Eight steps denoise over exactly the snapshot's `sample_sigmas` plus 0.** The generator's
/// render — steps unset, and steps = 8 — is byte-identical to the render assembled here over the
/// literal list, and is NOT the render over the base id's 8-step resolution-shifted schedule, nor
/// over the saved list with one sigma nudged: so the equality is a statement about the list the
/// sampler received, not about a render that ignores its schedule.
///
/// *Mutations that red this:* resolving the Turbo id through the base `resolve_run_params`;
/// applying a `mu` shift or the terminal stretch to the saved list; reading a hardcoded list
/// instead of the snapshot's (the second fixture below saves a different one).
#[test]
fn eight_steps_denoise_over_exactly_the_saved_sigmas_then_zero() {
    let turbo = turbo_snapshot();
    let root = turbo.path();
    let generator = load(TURBO, root, OffloadPolicy::Resident);
    let default_steps = pixels(generator.as_ref(), &request(None));
    let eight_steps = pixels(generator.as_ref(), &request(Some(8)));
    let literal = render_over(root, &SAVED_PLUS_ZERO);
    assert_eq!(default_steps, literal, "steps unset");
    assert_eq!(eight_steps, literal, "steps = 8");

    // The controls: the same assembly over a different list is a different picture.
    let base_schedule = load_scheduler_config(&tiny_snapshot()).unwrap();
    let shifted = scheduler::sigmas_for_image(&base_schedule, 8, SIZE, SIZE).unwrap();
    assert_ne!(
        render_over(root, &shifted),
        literal,
        "the base id's resolution-shifted 8-step schedule"
    );
    let mut nudged = SAVED_PLUS_ZERO;
    nudged[6] = 0.75;
    assert_ne!(render_over(root, &nudged), literal, "one sigma moved");

    // Read from the snapshot: a fixture that saves another eight-sigma list renders over THAT.
    let other = [1.0, 0.95, 0.9, 0.8, 0.7, 0.55, 0.4, 0.2];
    let other_snapshot = turbo_snapshot_with(&other);
    let mut other_plus_zero: Vec<f32> = other.iter().map(|s| *s as f32).collect();
    other_plus_zero.push(0.0);
    let other_render = pixels(
        load(TURBO, other_snapshot.path(), OffloadPolicy::Resident).as_ref(),
        &request(None),
    );
    assert_eq!(other_render, render_over(root, &other_plus_zero));
    assert_ne!(other_render, literal);

    // And the loader hands back the saved list unrounded.
    assert_eq!(
        load_turbo_schedule(root).unwrap().sample_sigmas,
        TURBO_SAMPLE_SIGMAS
    );
}

/// **Twelve steps denoise over the 12-point resample**, pinned to thirteen literals. The same
/// render-level equality as the 8-step claim, plus the function-level pin the MLX twin shares.
///
/// *Mutations that red this:* resampling in `log σ` or by nearest knot; interpolating over the
/// eight saved sigmas without the trailing-0 knot (`u_j = j/7`); falling back to the saved list
/// for an off-count request.
#[test]
fn twelve_steps_denoise_over_the_pinned_resample() {
    assert_eq!(
        scheduler::saved_sigmas(&TURBO_SAMPLE_SIGMAS, 12).unwrap(),
        TWELVE_STEP_RESAMPLE
    );
    assert_eq!(
        scheduler::saved_sigmas(&TURBO_SAMPLE_SIGMAS, 8).unwrap(),
        SAVED_PLUS_ZERO
    );

    let turbo = turbo_snapshot();
    let root = turbo.path();
    let generator = load(TURBO, root, OffloadPolicy::Resident);
    let twelve = pixels(generator.as_ref(), &request(Some(12)));
    let literal = render_over(root, &TWELVE_STEP_RESAMPLE);
    assert_eq!(twelve, literal);

    let mut nudged = TWELVE_STEP_RESAMPLE;
    nudged[10] = 0.6;
    assert_ne!(render_over(root, &nudged), literal, "one sigma moved");
    assert_ne!(
        twelve,
        pixels(generator.as_ref(), &request(None)),
        "the step knob is live"
    );
    // One step is legal on Turbo (no terminal stretch to divide by) and renders.
    assert_eq!(
        pixels(generator.as_ref(), &request(Some(1))),
        render_over(root, &[1.0, 0.0])
    );
}

/// The curated `scheduler` axis on Turbo, at the render: a curated name replaces the saved list
/// with that scheduler's own **unshifted** (`mu = 0`) ramp at the same step count — not the base
/// id's resolution-shifted one — while a curated *sampler* keeps the saved list and changes only
/// the integrator. (The names this id advertises are the curated menu; a native alias is not on
/// it, so the saved list is reached by leaving `scheduler` unset.)
///
/// *Mutation that reds this:* passing the base `mu` for the request's resolution to
/// `resolve_flow_schedule` on the Turbo id.
#[test]
fn a_curated_scheduler_name_replaces_the_saved_list_with_its_unshifted_ramp() {
    let turbo = turbo_snapshot();
    let root = turbo.path();
    let generator = load(TURBO, root, OffloadPolicy::Resident);
    let literal = render_over(root, &SAVED_PLUS_ZERO);

    let mut curated = request(None);
    curated.scheduler = Some("karras".to_owned());
    generator.validate(&curated).unwrap();
    let karras = candle_gen::resolve_flow_schedule(Some("karras"), 0.0, 8, &SAVED_PLUS_ZERO);
    assert_eq!(karras.len(), 9);
    assert_ne!(karras, SAVED_PLUS_ZERO);
    assert_eq!(
        pixels(generator.as_ref(), &curated),
        render_over(root, &karras)
    );

    // A curated SAMPLER rides the saved list: same sigmas, a different integrator.
    let mut sampler = request(None);
    sampler.sampler = Some("heun".to_owned());
    generator.validate(&sampler).unwrap();
    assert_ne!(pixels(generator.as_ref(), &sampler), literal);
}

/// True CFG is off by default and still engages with a negative prompt and a scale above 1.
#[test]
fn true_cfg_is_off_by_default_and_still_accepted_on_turbo() {
    let turbo = turbo_snapshot();
    let generator = load(TURBO, turbo.path(), OffloadPolicy::Resident);
    let plain = pixels(generator.as_ref(), &request(None));
    let mut req = request(None);
    req.negative_prompt = Some("blurry low quality photo".to_owned());
    assert_eq!(
        pixels(generator.as_ref(), &req),
        plain,
        "a negative prompt alone is inert: the default scale is 1"
    );
    req.true_cfg = Some(2.5);
    assert_ne!(pixels(generator.as_ref(), &req), plain);
}

/// Each snapshot under the other id is a typed refusal that names the mismatch — through `load`
/// and through the memory registration — and nothing renders over the wrong schedule.
///
/// *Mutation that reds this:* removing `check_snapshot_variant` from `load_variant` (the base id
/// would then fail later with the generic "only `use_dynamic_shifting: true`" message, untyped)
/// or from `memory_strategy_contract` (a contract would be priced for a refused load).
#[test]
fn a_snapshot_under_the_other_id_is_a_typed_refusal_naming_the_mismatch() {
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let turbo = turbo_snapshot();
    let base = tiny_snapshot();

    let refused = |id: &str, root: &std::path::Path| -> Vec<String> {
        let spec = spec(root, OffloadPolicy::Resident);
        let by_load = match registry.load(id, &spec) {
            Ok(_) => panic!("{id} must refuse {}", root.display()),
            Err(err) => err,
        };
        let by_contract = registry
            .memory_strategy_contract(id, &spec)
            .expect_err("no contract is priced for a refused load");
        [by_load, by_contract]
            .into_iter()
            .map(|err| {
                assert!(matches!(err, CoreError::Unsupported(_)), "{id}: {err:?}");
                err.to_string()
            })
            .collect()
    };

    for message in refused(BASE, turbo.path()) {
        assert!(message.contains("qwen_image_2_1:"), "{message}");
        assert!(message.contains("Turbo snapshot"), "{message}");
        assert!(message.contains("`sample_sigmas`"), "{message}");
        assert!(message.contains("`qwen_image_2_1_turbo`"), "{message}");
    }
    for message in refused(TURBO, &base) {
        assert!(message.contains("qwen_image_2_1_turbo:"), "{message}");
        assert!(message.contains("no `sample_sigmas`"), "{message}");
        assert!(message.contains("`qwen_image_2_1`"), "{message}");
    }

    // The base trainer is not a Turbo trainer either: the schedule reader it shares with the base
    // generator refuses the static config.
    let err = load_scheduler_config(turbo.path()).unwrap_err().to_string();
    assert!(err.contains("use_dynamic_shifting"), "{err}");

    // Each snapshot under its OWN id still loads.
    assert!(registry
        .load(BASE, &spec(&base, OffloadPolicy::Resident))
        .is_ok());
    assert!(registry
        .load(TURBO, &spec(turbo.path(), OffloadPolicy::Resident))
        .is_ok());
}

/// **The bf16 VAE loads at the lane's compute dtype and decodes.** The file stores bf16 (checked
/// from its header); the loaded autoencoder's decoder weight is held at
/// `loader::compute_dtype_on(device)` — f32 on the CPU lane, bf16 on CUDA and Metal — which is the
/// same dtype the base snapshot's f32 file lands on; and the decode of one latent agrees with the
/// base VAE's decode of it to within the bf16 rounding of the weights.
///
/// Tolerance: 2e-2 absolute (`assert_close` floors the peak at 1; the decode is clamped to
/// `[-1, 1]`). Measured at `peak = 0.50`: CPU lane (`f32(bf16(w))` weights against `w`, f32
/// compute) `max|Δ| = 7.1e-3`, `mean|Δ| = 5.7e-4`; Metal lane (both in bf16 compute)
/// `max|Δ| = 5.9e-3`, `mean|Δ| = 7.6e-4` — so the bound carries ~2.8× headroom over the worst.
/// A decode of misread weights is O(1) off.
///
/// *Mutation that reds this:* loading the VAE at the file's stored dtype instead of the lane's
/// (`DType::BF16` on the CPU lane fails the dtype assertion, and candle's CPU backend then has no
/// bf16 conv to decode with).
#[test]
fn the_bf16_vae_loads_at_the_lane_compute_dtype_and_decodes() {
    let turbo = turbo_snapshot();
    let device = candle_gen::default_device().unwrap();
    let lane = loader::compute_dtype_on(&device);
    if device.is_cpu() {
        assert_eq!(lane, DType::F32, "the CPU lane computes in f32");
    } else {
        assert_eq!(lane, DType::BF16, "a GPU lane computes in bf16");
    }
    assert_eq!(
        stored_dtypes(&turbo.path().join("vae/diffusion_pytorch_model.safetensors")),
        ["BF16"]
    );

    let turbo_vae = load_vae(turbo.path(), &device).unwrap();
    let base_vae = load_vae(&tiny_snapshot(), &device).unwrap();
    assert_eq!(turbo_vae.weight_dtype(), lane, "the bf16 file");
    assert_eq!(base_vae.weight_dtype(), lane, "the f32 file");
    assert_eq!(turbo_vae.compute_dtype(), lane);

    let z = turbo_vae.config().z_dim;
    let values: Vec<f32> = (0..z * 2 * 2)
        .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let latents = candle_core::Tensor::from_vec(values, (1, z, 2, 2), &Device::Cpu)
        .unwrap()
        .to_device(&device)
        .unwrap();
    let from_bf16 = turbo_vae.decode_rgba(&latents).unwrap();
    let from_f32 = base_vae.decode_rgba(&latents).unwrap();
    assert_eq!(from_bf16.dims(), &[1, 4, 32, 32]);
    assert_eq!(from_bf16.dtype(), DType::F32);
    assert_close("turbo bf16 VAE decode", &from_bf16, &from_f32, 2e-2);
    let spread = crate::common::host_f32(&from_bf16);
    let (lo, hi) = spread
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert!(hi - lo > 1e-2, "the decode is a picture, not a constant");
}
