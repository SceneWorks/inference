//! The distilled **Qwen-Image 2.1 Turbo** engine id (`qwen_image_2_1_turbo`, sc-25900) on a
//! Turbo-shaped copy of the committed miniature snapshot ([`crate::common::turbo_snapshot`]: saved
//! `sample_sigmas`, static-shift scheduler config, bf16 VAE, single-file text encoder), through
//! the explicit provider catalog's production load path.
//!
//! The schedule assertions are made on the **render**: the generator's pixels are compared with a
//! manual run of the public pipeline pieces over an explicitly written sigma list, so they hold
//! only if the sampler really received that list. The lists themselves are pinned value for value
//! in the crate's `scheduler` and `model` unit tests.

use std::path::Path;

use mlx_gen::gen_core::{Error as CoreError, Progress};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};
use mlx_gen_qwen_image_2_1::pipeline::decode_tiling;
use mlx_gen_qwen_image_2_1::{
    create_noise, decode_rgb, denoise_reported, load_text_encoder, load_tokenizer,
    load_transformer, load_vae, system_prompt_drop_count, DenoiseInputs, DenoiseReport,
    EvaluationPath, KvCacheMode, KvCachePath, PromptBranch,
};

use crate::common::{tiny_snapshot, turbo_snapshot, TURBO_SAMPLE_SIGMAS};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";

fn load(
    id: &str,
    root: &Path,
    policy: OffloadPolicy,
) -> mlx_gen::gen_core::Result<Box<dyn mlx_gen::Generator>> {
    mlx_gen_qwen_image_2_1::provider_registry().unwrap().load(
        id,
        &LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy),
    )
}

fn request(steps: Option<u32>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        steps,
        seed: Some(42),
        ..Default::default()
    }
}

/// The generator's RGB8 pixels and the `Step` events it reported.
fn render(
    generator: &dyn mlx_gen::Generator,
    req: &GenerationRequest,
) -> (Vec<u8>, Vec<(u32, u32)>) {
    let mut steps = Vec::new();
    let out = generator
        .generate(req, &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        })
        .unwrap();
    let GenerationOutput::Images(mut images) = out else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    (images.remove(0).pixels, steps)
}

/// `req` rendered by hand from the snapshot's components over exactly `sigmas` — the reference a
/// generator render must equal if (and only if) its sampler ran that schedule.
fn render_over(root: &Path, req: &GenerationRequest, sigmas: &[f32]) -> Vec<u8> {
    render_over_reported(root, req, sigmas).0
}

/// [`render_over`] plus what the denoise loop reported. The prefix KV cache is **on**, as it is on
/// the generator's render path (`generate_impl` passes `KvCacheMode::On` for both engine ids), so
/// the two sides of every pixel comparison here run the same cache mode.
fn render_over_reported(
    root: &Path,
    req: &GenerationRequest,
    sigmas: &[f32],
) -> (Vec<u8>, DenoiseReport) {
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root).unwrap();
    let transformer = load_transformer(root).unwrap();
    let vae = load_vae(root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = encoder
        .encode_prompt(&tokenizer, &req.prompt, drop)
        .unwrap();
    let seed = req.seed.unwrap();
    let cancel = CancelFlag::new();
    let mut report = DenoiseReport::default();
    let latents = denoise_reported(
        DenoiseInputs {
            transformer: &transformer,
            sigmas,
            latents: create_noise(
                seed,
                req.width,
                req.height,
                transformer.config().in_channels,
            )
            .unwrap(),
            prompt_embeds: &prompt,
            negative_embeds: None,
            true_cfg_scale: 1.0,
            width: req.width,
            height: req.height,
            sampler: None,
            seed,
            cancel: &cancel,
            references: None,
            kv_cache: KvCacheMode::On,
        },
        &mut report,
        &mut |_| {},
    )
    .unwrap();
    let pixels = decode_rgb(
        &vae,
        &latents,
        req.width,
        req.height,
        decode_tiling(req).as_ref(),
        None,
    )
    .unwrap()
    .pixels;
    (pixels, report)
}

fn saved_then_zero() -> Vec<f32> {
    TURBO_SAMPLE_SIGMAS
        .iter()
        .map(|sigma| *sigma as f32)
        .chain([0.0])
        .collect()
}

fn every_step(total: u32) -> Vec<(u32, u32)> {
    (1..=total).map(|current| (current, total)).collect()
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    }
}

/// The Turbo-shaped fixture loads under the Turbo id and passes the shared generator contract
/// (validate honesty, progress, cancellation, seeded determinism) under both residency policies.
#[test]
fn turbo_fixture_passes_the_generator_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    // The fixture really is Turbo-shaped where it matters to the loader: a bf16 VAE.
    assert_eq!(
        load_vae(&root).unwrap().compute_dtype(),
        mlx_rs::Dtype::Bfloat16
    );
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let generator = load(TURBO, &root, policy).expect("the Turbo fixture loads");
        assert_eq!(generator.descriptor().id, TURBO);
        gen_core_testkit::conformance(|| load(TURBO, &root, policy).unwrap(), &profile());
    }
}

/// No step count, and the saved length named explicitly, both denoise over exactly the
/// snapshot's eight saved sigmas plus a trailing 0 — no `mu` shift, no terminal stretch.
///
/// *Mutations that red this:* a 40-step default on the Turbo id; running the saved sigmas through
/// the base shift/stretch; reading a hardcoded list instead of the snapshot's (the second half
/// rewrites the snapshot's `sample_sigmas` and the render must follow).
#[test]
fn turbo_default_and_eight_steps_render_over_the_snapshots_saved_sigmas() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = load(TURBO, &root, OffloadPolicy::Resident).unwrap();

    let want = render_over(&root, &request(None), &saved_then_zero());
    let (default, steps) = render(generator.as_ref(), &request(None));
    assert_eq!(steps, every_step(8));
    assert_eq!(default, want, "steps unset: the saved sigmas + 0");
    let (explicit, steps) = render(generator.as_ref(), &request(Some(8)));
    assert_eq!(steps, every_step(8));
    assert_eq!(explicit, want, "steps = 8: the saved sigmas + 0");

    // The schedule is the SNAPSHOT's: a different saved list is a different render, of that
    // list's own length.
    let index = root.join("model_index.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
    value["sample_sigmas"] = serde_json::json!([1.0, 0.9, 0.6, 0.3]);
    std::fs::write(&index, serde_json::to_string(&value).unwrap()).unwrap();
    let generator = load(TURBO, &root, OffloadPolicy::Resident).unwrap();
    let (rewritten, steps) = render(generator.as_ref(), &request(None));
    assert_eq!(steps, every_step(4));
    assert_eq!(
        rewritten,
        render_over(&root, &request(None), &[1.0, 0.9, 0.6, 0.3, 0.0])
    );
    assert_ne!(rewritten, want);
}

/// The prefix KV cache (sc-25903) on the Turbo schedule: over the snapshot's eight saved sigmas the
/// cached loop prefills once and decodes the other seven evaluations from the cache, and that
/// render is the one the `qwen_image_2_1_turbo` generator produces.
///
/// `Generator::generate` does not surface the `DenoiseReport`, so the evaluation paths are
/// asserted at the closest public seam — `denoise_reported` over the Turbo fixture's components
/// and schedule with the cache mode the generator's one shared call site passes — and tied to the
/// generator by pixel equality.
///
/// *Mutation that reds this:* the denoise loop not keeping the extracted cache (every evaluation
/// then re-extracts instead of decoding from it).
#[test]
fn turbo_saved_sigma_render_extracts_once_then_decodes_from_the_prefix_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let req = request(None);
    let (manual, report) = render_over_reported(&root, &req, &saved_then_zero());

    assert_eq!(report.kv_cache, KvCachePath::Cached);
    assert!(
        report.kv_cache_bytes > 0,
        "the positive branch held a cache"
    );
    let paths: Vec<_> = report
        .evaluations
        .iter()
        .map(|evaluation| {
            assert_eq!(evaluation.branch, PromptBranch::Positive, "true CFG is off");
            evaluation.path
        })
        .collect();
    let mut want = vec![EvaluationPath::Extract];
    want.extend([EvaluationPath::Cached; 7]);
    assert_eq!(paths, want, "one evaluation per saved sigma");
    // A cached evaluation runs the blocks over the target tokens only.
    let (extract, cached) = (report.evaluations[0], report.evaluations[1]);
    assert!(cached.block_tokens < extract.block_tokens);
    assert_eq!(
        cached.block_tokens,
        2 * 2,
        "a 32x32 target is a 2x2 latent grid"
    );

    let generator = load(TURBO, &root, OffloadPolicy::Resident).unwrap();
    let (pixels, steps) = render(generator.as_ref(), &req);
    assert_eq!(steps, every_step(8));
    assert_eq!(pixels, manual);
}

/// Twelve steps on the Turbo id denoise over the 12-point resample of the saved curve
/// (`scheduler::resample_saved_sigmas`), and the same request on the base id is unchanged: it
/// still renders the base snapshot's own resolution-shifted 12-step schedule.
///
/// *Mutation that reds this:* ignoring the requested step count on the Turbo id (the render
/// would equal the 8-step one and report 8 steps).
#[test]
fn turbo_twelve_steps_render_over_the_resample_and_the_base_id_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let req = request(Some(12));

    let resampled =
        mlx_gen_qwen_image_2_1::scheduler::resample_saved_sigmas(&TURBO_SAMPLE_SIGMAS, 12).unwrap();
    assert_eq!(resampled.len(), 13);
    let generator = load(TURBO, &root, OffloadPolicy::Resident).unwrap();
    let (turbo, steps) = render(generator.as_ref(), &req);
    assert_eq!(steps, every_step(12));
    assert_eq!(turbo, render_over(&root, &req, &resampled));
    let (eight, _) = render(generator.as_ref(), &request(Some(8)));
    assert_ne!(turbo, eight, "twelve steps is not the saved eight");

    // The base id, on the base snapshot, for the same request.
    let base_root = tiny_snapshot();
    let base_sigmas = mlx_gen_qwen_image_2_1::scheduler::sigmas_for_image(
        &mlx_gen_qwen_image_2_1::load_scheduler_config(&base_root).unwrap(),
        12,
        req.width,
        req.height,
    )
    .unwrap();
    let base = load(BASE, &base_root, OffloadPolicy::Resident).unwrap();
    let (base_pixels, steps) = render(base.as_ref(), &req);
    assert_eq!(steps, every_step(12));
    assert_eq!(base_pixels, render_over(&base_root, &req, &base_sigmas));
    assert_ne!(base_sigmas, resampled);
}

/// Each id refuses the other checkpoint's snapshot with a typed error that names the mismatch
/// and the id to use, instead of silently running the wrong schedule over it.
///
/// *Mutations that red this:* dropping `refuse_saved_sigma_snapshot` from the base load (the
/// hybrid below would then render 40 base steps over a distilled checkpoint); accepting a missing
/// `sample_sigmas` under the Turbo id.
#[test]
fn each_id_refuses_the_other_checkpoints_snapshot_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let turbo_root = turbo_snapshot(tmp.path());

    let refusal = |id: &str, root: &Path| match load(id, root, OffloadPolicy::Resident) {
        Err(CoreError::Unsupported(message)) => message,
        Err(other) => panic!("{id}: expected a typed Unsupported refusal, got {other:?}"),
        Ok(_) => panic!("{id}: the other checkpoint's snapshot must not load"),
    };

    let err = refusal(BASE, &turbo_root);
    assert!(
        err.starts_with("qwen_image_2_1: model_index.json saves `sample_sigmas`"),
        "{err}"
    );
    assert!(err.contains("load it as `qwen_image_2_1_turbo`"), "{err}");

    let err = refusal(TURBO, &tiny_snapshot());
    assert!(
        err.starts_with("qwen_image_2_1_turbo: model_index.json carries no `sample_sigmas`"),
        "{err}"
    );
    assert!(err.contains("as `qwen_image_2_1`"), "{err}");

    // A base load of a snapshot whose `sample_sigmas` is present but malformed is the same typed
    // base refusal — never the Turbo reader's validation error under the base id.
    let malformed = tmp.path().join("malformed");
    mlx_gen::quant::copy_dir(&tiny_snapshot(), &malformed).unwrap();
    for bad in [serde_json::json!([]), serde_json::json!("1.0, 0.5")] {
        let index = malformed.join("model_index.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
        value["sample_sigmas"] = bad.clone();
        std::fs::write(&index, serde_json::to_string(&value).unwrap()).unwrap();
        let err = refusal(BASE, &malformed);
        assert!(
            err.starts_with("qwen_image_2_1: model_index.json saves `sample_sigmas`"),
            "{bad}: {err}"
        );
        assert!(
            err.contains("load it as `qwen_image_2_1_turbo`"),
            "{bad}: {err}"
        );
    }
    // An explicit null is "no saved schedule": the base snapshot still loads under the base id.
    let index = malformed.join("model_index.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
    value["sample_sigmas"] = serde_json::Value::Null;
    std::fs::write(&index, serde_json::to_string(&value).unwrap()).unwrap();
    assert!(load(BASE, &malformed, OffloadPolicy::Resident).is_ok());

    // A Turbo-id load with a single-file source fails in the shared memory contract, which must
    // name the id that was asked for.
    let single_file = mlx_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract(
        TURBO,
        &LoadSpec::new(WeightsSource::File(tmp.path().join("model.safetensors"))),
    )
    .unwrap_err()
    .to_string();
    assert!(
        single_file.starts_with("qwen_image_2_1_turbo: memory facts require a snapshot directory"),
        "{single_file}"
    );

    // A hybrid — saved sigmas beside the base's dynamic-shift scheduler config — is refused under
    // BOTH ids: the base id because the sigmas are saved, the Turbo id because the scheduler
    // would shift them.
    let hybrid = tmp.path().join("hybrid");
    mlx_gen::quant::copy_dir(&turbo_root, &hybrid).unwrap();
    std::fs::copy(
        tiny_snapshot().join("scheduler/scheduler_config.json"),
        hybrid.join("scheduler/scheduler_config.json"),
    )
    .unwrap();
    assert!(refusal(BASE, &hybrid).contains("load it as `qwen_image_2_1_turbo`"));
    let err = refusal(TURBO, &hybrid);
    assert!(err.contains("`use_dynamic_shifting: true`"), "{err}");
}

/// The curated `scheduler` and `sampler` axes keep working on the Turbo id through the same
/// shared seam the base id uses: a curated scheduler is built over an unshifted flow model
/// (`mu = 0`) and **replaces** the saved sigmas, a name outside the menu is refused, and a curated
/// sampler integrates the saved sigmas.
#[test]
fn turbo_honours_the_curated_scheduler_and_sampler_axes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = load(TURBO, &root, OffloadPolicy::Resident).unwrap();
    let (native, _) = render(generator.as_ref(), &request(None));

    // A name outside the advertised menu never reaches the schedule: it is refused at validate,
    // exactly as on the base id.
    let mut unknown = request(None);
    unknown.scheduler = Some("flow_match_euler".to_owned());
    let err = generator.validate(&unknown).unwrap_err().to_string();
    assert!(err.contains("unsupported scheduler"), "{err}");

    let mut curated = request(None);
    curated.scheduler = Some("simple".to_owned());
    generator.validate(&curated).unwrap();
    let unshifted = mlx_gen::resolve_flow_schedule(Some("simple"), 0.0, 8, &saved_then_zero());
    let (pixels, steps) = render(generator.as_ref(), &curated);
    assert_eq!(steps.len(), unshifted.len() - 1);
    assert_eq!(pixels, render_over(&root, &curated, &unshifted));
    assert_ne!(pixels, native);

    let mut sampler = request(None);
    sampler.sampler = Some("heun".to_owned());
    generator.validate(&sampler).unwrap();
    // A second-order solver reports its stages too; what matters is that it integrates the saved
    // eight-interval schedule and lands somewhere other than Euler does.
    let (heun, steps) = render(generator.as_ref(), &sampler);
    assert_eq!(steps.last(), Some(&(8, 8)));
    assert_ne!(heun, native);
}
