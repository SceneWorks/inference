//! Prefix-KV-cache admission (sc-25905) — the Candle twin of
//! `mlx-gen-qwen-image-2-1/tests/kv_admission.rs`: the cache is a request-selected execution
//! domain, priced per request, and every render says which path it ran.
//!
//! Everything here runs through the explicit provider catalog's production load path and the
//! public [`Generator`] trait, under **both** engine ids, on the committed miniature snapshot.
//! What is observed is what the denoise loop did — `GenerationReport::prefix_kv_cache` is written
//! from the loop's own report — not what the request asked for.
//!
//! The byte figures are derived in the tests from the tensors the cache stores (blocks × prefix
//! tokens × joint width × element width), never copied from a run. The element width is the
//! loaded model's own: f32 on the CPU lane, bf16 under `--features cuda` / `--features metal`.
//!
//! Numerics: a generator render is compared with a hand-assembled run of the **same** path, so
//! the comparison is exact. Nothing here compares a cached forward with a prefill one — on x86-64
//! the CPU GEMM's kernel choice follows the row count, so those two agree to rounding only
//! (`tests/kv_cache.rs` gates that at `1e-5 + 1e-5·|want|`).

use std::path::Path;

use candle_gen::gen_core::{
    CancelFlag, Conditioning, Error as CoreError, GenerationMemory, GenerationOutput,
    GenerationReport, GenerationRequest, Generator, Image, LoadSpec, MemoryBehaviorRoute,
    MemoryBudget, MemoryMode, MemoryNumericTier, MemoryRunOutcome, MemoryStrategy, OffloadPolicy,
    PrefixKvCacheMode, PrefixKvCacheReport, WeightsSource,
};
use candle_gen_qwen_image_2_1::memory_strategy::{
    image_tokens, prefix_kv_cache_demand_bytes, prefix_kv_cache_tensor_bytes, request_demand_bytes,
    TABLE_CONDITIONING_TOKENS,
};
use candle_gen_qwen_image_2_1::{
    create_noise, decode_rgb, decode_tiling, denoise_reported, load_scheduler_config,
    load_text_encoder, load_tokenizer, load_transformer, load_turbo_schedule, load_vae,
    load_vision_config, reference_fit, scheduler, system_prompt_drop_count, DenoiseInputs,
    DenoiseReport, EvaluationPath, KvCacheMode, KvCachePath, PromptBranch,
};

use crate::common::{tiny_snapshot, turbo_snapshot};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const STEPS: usize = 4;
const SEED: u64 = 42;

fn spec(root: &Path) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Resident)
}

fn load(id: &str, root: &Path) -> Box<dyn Generator> {
    candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(id, &spec(root))
        .unwrap_or_else(|e| panic!("{id} loads through the catalog path: {e}"))
}

/// A 32×32 text-to-image request; `negative` adds a true-CFG negative branch.
fn request(negative: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        negative_prompt: negative.then(|| "blurry low quality photo".to_owned()),
        true_cfg: negative.then_some(2.5),
        width: 32,
        height: 32,
        steps: Some(STEPS as u32),
        seed: Some(SEED),
        ..Default::default()
    }
}

fn selecting(mut req: GenerationRequest, mode: PrefixKvCacheMode) -> GenerationRequest {
    req.memory = Some(GenerationMemory {
        prefix_kv_cache: Some(mode),
        ..req.memory.unwrap_or_default()
    });
    req
}

/// One render through `generate_with_report`: its pixels and what it says it ran.
fn render(generator: &dyn Generator, req: &GenerationRequest) -> (Vec<u8>, PrefixKvCacheReport) {
    let report: GenerationReport = generator
        .generate_with_report(req, &mut |_| {})
        .expect("render");
    let Some(GenerationOutput::Images(mut images)) = report.output else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    (
        images.remove(0).pixels,
        report
            .prefix_kv_cache
            .expect("this provider reports the path on every render"),
    )
}

/// `req` rendered by hand from the snapshot's components with the cache `mode`, over the schedule
/// the generator resolves for it, plus what the denoise loop reported and the derived bytes of
/// each branch's cache.
struct Manual {
    pixels: Vec<u8>,
    report: DenoiseReport,
    /// Derived bytes of ONE branch's cache for this request: K and V for every block, each
    /// `[1, heads, prefix_tokens, head_dim]`, at the model's compute dtype.
    positive_branch_bytes: u64,
    negative_branch_bytes: u64,
}

fn manual(id: &str, root: &Path, req: &GenerationRequest, mode: KvCacheMode) -> Manual {
    let device = candle_gen::default_device().unwrap();
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root, &device).unwrap();
    let transformer = load_transformer(root, &device).unwrap();
    let vae = load_vae(root, &device).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = encoder
        .encode_prompt(&tokenizer, &req.prompt, drop)
        .unwrap();
    let negative = req
        .negative_prompt
        .as_deref()
        .map(|text| encoder.encode_prompt(&tokenizer, text, drop).unwrap());
    let sigmas = if id == TURBO {
        scheduler::saved_sigmas(&load_turbo_schedule(root).unwrap().sample_sigmas, STEPS).unwrap()
    } else {
        scheduler::sigmas_for_image(
            &load_scheduler_config(root).unwrap(),
            STEPS,
            req.width,
            req.height,
        )
        .unwrap()
    };
    let seed = candle_gen::image_seed(SEED, 0);
    let cancel = CancelFlag::new();
    let mut report = DenoiseReport::default();
    let latents = denoise_reported(
        DenoiseInputs {
            transformer: &transformer,
            sigmas: &sigmas,
            latents: create_noise(
                seed,
                req.width,
                req.height,
                transformer.config().in_channels,
                &device,
            )
            .unwrap(),
            prompt_embeds: &prompt,
            negative_embeds: negative.as_ref(),
            true_cfg_scale: req.true_cfg.unwrap_or(1.0),
            width: req.width,
            height: req.height,
            sampler: None,
            seed,
            cancel: &cancel,
            references: None,
            kv_cache: mode,
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
        Some(&cancel),
    )
    .unwrap()
    .pixels;

    let cfg = transformer.config();
    let element_width = transformer.compute_dtype().size_in_bytes() as u64;
    let branch_bytes = |text: &candle_core::Tensor| {
        prefix_kv_cache_tensor_bytes(
            cfg.num_layers as u64,
            (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
            text.dims()[1] as u64,
            element_width,
        )
    };
    Manual {
        pixels,
        positive_branch_bytes: branch_bytes(&prompt),
        negative_branch_bytes: negative.as_ref().map_or(0, branch_bytes),
        report,
    }
}

fn paths(report: &DenoiseReport, branch: PromptBranch) -> Vec<EvaluationPath> {
    report
        .evaluations
        .iter()
        .filter(|evaluation| evaluation.branch == branch)
        .map(|evaluation| evaluation.path)
        .collect()
}

/// **A cached selection renders from the cache and says so** — under both ids, with and without
/// a negative branch. The generator's report is `cached`, its cache bytes are the derived figure
/// for that request (one cache per prompt branch, each over that branch's own prefix, at the
/// lane's element width), and its pixels are the hand-assembled cache-on pipeline's, whose loop
/// extracted once per branch and decoded every later evaluation from the cache.
///
/// *Mutations that red this:* flipping the generator's switch to `KvCacheMode::Off` (it reports
/// `prefill` and zero bytes); hard-coding the reported path or bytes; reporting only the positive
/// branch's bytes under true CFG; deriving the bytes at bf16 on the f32 CPU lane.
#[test]
fn a_cached_selection_renders_from_the_cache_and_reports_its_derived_bytes() {
    let turbo = turbo_snapshot();
    for (id, root) in [(BASE, tiny_snapshot()), (TURBO, turbo.path().to_path_buf())] {
        let generator = load(id, &root);
        for negative in [false, true] {
            let tag = format!("{id} negative={negative}");
            let req = request(negative);
            let want = manual(id, &root, &req, KvCacheMode::On);
            assert_eq!(want.report.kv_cache, KvCachePath::Cached, "{tag}");
            let mut sequence = vec![EvaluationPath::Extract];
            sequence.extend([EvaluationPath::Cached; STEPS - 1]);
            assert_eq!(
                paths(&want.report, PromptBranch::Positive),
                sequence,
                "{tag}"
            );
            assert_eq!(
                paths(&want.report, PromptBranch::Negative),
                if negative { sequence } else { Vec::new() },
                "{tag}"
            );
            assert!(want.positive_branch_bytes > 0, "{tag}");
            assert_eq!(negative, want.negative_branch_bytes > 0, "{tag}");
            let derived_bytes = want.positive_branch_bytes + want.negative_branch_bytes;
            assert_eq!(
                want.report.kv_cache_bytes as u64, derived_bytes,
                "{tag}: the loop's caches hold exactly the derived bytes"
            );

            let (pixels, ran) = render(
                generator.as_ref(),
                &selecting(req, PrefixKvCacheMode::Cached),
            );
            assert_eq!(
                ran,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Cached,
                    cache_bytes: derived_bytes,
                },
                "{tag}"
            );
            assert_eq!(pixels, want.pixels, "{tag}");
        }
    }
}

/// **A shed selection is exactly the prefill path, and so is a request that selects nothing** —
/// under both ids, with and without a negative branch. The report is `prefill` with zero cache
/// bytes and the pixels equal the hand-assembled cache-off pipeline's (the same path run twice).
///
/// *Mutations that red this:* ignoring a `Prefill` selection (it reports `cached`); defaulting an
/// unset request to the cache.
#[test]
fn a_shed_selection_and_an_unset_request_are_the_prefill_path() {
    let turbo = turbo_snapshot();
    for (id, root) in [(BASE, tiny_snapshot()), (TURBO, turbo.path().to_path_buf())] {
        let generator = load(id, &root);
        for negative in [false, true] {
            let tag = format!("{id} negative={negative}");
            let req = request(negative);
            let want = manual(id, &root, &req, KvCacheMode::Off);
            assert_eq!(want.report.kv_cache, KvCachePath::Prefill, "{tag}");
            assert_eq!(want.report.kv_cache_bytes, 0, "{tag}");
            assert!(
                want.report
                    .evaluations
                    .iter()
                    .all(|evaluation| evaluation.path == EvaluationPath::Prefill),
                "{tag}"
            );
            let prefill = PrefixKvCacheReport {
                path: PrefixKvCacheMode::Prefill,
                cache_bytes: 0,
            };
            let (pixels, ran) = render(
                generator.as_ref(),
                &selecting(req.clone(), PrefixKvCacheMode::Prefill),
            );
            assert_eq!(ran, prefill, "{tag}: shed selection");
            assert_eq!(pixels, want.pixels, "{tag}: shed selection");
            // No memory block at all, and a memory block that selects nothing.
            let mut blank = req.clone();
            blank.memory = Some(GenerationMemory::default());
            for (label, unset) in [("absent block", req.clone()), ("empty block", blank)] {
                let (pixels, ran) = render(generator.as_ref(), &unset);
                assert_eq!(ran, prefill, "{tag}: {label}");
                assert_eq!(pixels, want.pixels, "{tag}: {label}");
            }
        }
    }
}

/// What the SceneWorks worker does for one request whose cache-shed demand it prices at `shed`,
/// with `total_bytes` of budget: price the cache from the provider's derived model, pick the mode
/// ([`PrefixKvCacheMode::select`]), open the request scope through the `Generator` trait with the
/// demand of the mode it picked (the shed demand when neither fits — that is the request it would
/// have submitted before the cache existed), let the scope configure the request, then write the
/// selection onto it and render.
fn admit_and_render(
    generator: &dyn Generator,
    req: &GenerationRequest,
    shed: u64,
    total_bytes: u64,
) -> Result<(PrefixKvCacheMode, Vec<u8>, PrefixKvCacheReport), CoreError> {
    let references = req.memory_reference_count();
    let negative = req.negative_prompt.is_some() && req.true_cfg.is_some_and(|scale| scale > 1.0);
    let cache = prefix_kv_cache_demand_bytes(
        req.width,
        req.height,
        TABLE_CONDITIONING_TOKENS,
        references,
        negative,
    );
    let budget = MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let selected = PrefixKvCacheMode::select(budget, shed, cache);

    let contract = generator.memory_strategy_contract().expect("a contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision: candle_gen::gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: references,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )?;
    context.optimization_authority = Default::default();
    context.geometry.width = req.width;
    context.geometry.height = req.height;
    context.budget = budget;
    context.predicted_peak_bytes = request_demand_bytes(
        selected.unwrap_or(PrefixKvCacheMode::Prefill),
        shed,
        req.width,
        req.height,
        TABLE_CONDITIONING_TOKENS,
        references,
        negative,
    );
    let mut scope = generator
        .begin_memory_strategy_request(&context)?
        .expect("an adopting provider opens a scope");
    let mode = selected.expect("an admitted request has a mode");
    let mut req = req.clone();
    scope.configure_request(&mut req)?;
    let req = selecting(req, mode);
    let (pixels, ran) = render(generator, &req);
    scope.finish(MemoryRunOutcome::Complete)?;
    Ok((mode, pixels, ran))
}

/// **Admission, end to end** — the story's second and third acceptance criteria, under both ids,
/// with and without a negative branch, and with a reference image:
///
/// * a budget that fits the cached demand runs **cached**;
/// * a budget one byte short of it — above the prefill demand — is **admitted and runs prefill**,
///   with zero cache bytes;
/// * a budget one byte short of the prefill demand is **refused**, with the shared safety check's
///   own refusal — the one this request got before the cache existed.
///
/// The path is read off the render's report, so a decision that never reached the generator
/// fails here.
///
/// *Mutations that red this:* ignoring the cache term in the selection (the between-budget
/// request runs cached); ignoring a `Prefill` selection in the generator (it reports `cached`);
/// ignoring a `Cached` selection (the roomy request reports `prefill`); admitting the
/// fits-neither request.
#[test]
fn a_request_between_the_two_budgets_is_admitted_and_runs_prefill() {
    // The caller's own cache-shed demand for these requests — any number; this crate publishes
    // no activation table, so it is not derived here.
    const SHED: u64 = 8 << 30;
    let turbo = turbo_snapshot();
    let edge = 64_u32;
    let reference = Conditioning::Reference {
        image: Image {
            width: edge,
            height: edge,
            pixels: (0..edge * edge * 3)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect(),
        },
        strength: None,
    };
    for (id, root) in [(BASE, tiny_snapshot()), (TURBO, turbo.path().to_path_buf())] {
        let generator = load(id, &root);
        let mut referenced = request(false);
        referenced.conditioning = vec![reference.clone()];
        for (label, req) in [
            ("text-to-image", request(false)),
            ("true CFG", request(true)),
            ("one reference", referenced),
        ] {
            let tag = format!("{id} {label}");
            let references = req.memory_reference_count();
            let cache = prefix_kv_cache_demand_bytes(
                req.width,
                req.height,
                TABLE_CONDITIONING_TOKENS,
                references,
                req.negative_prompt.is_some(),
            );
            assert!(cache > 0, "{tag}: the cache is never free");

            // 1. Room for the cache: cached.
            let (mode, _, roomy) = admit_and_render(generator.as_ref(), &req, SHED, SHED + cache)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Cached, "{tag}");
            assert_eq!(roomy.path, PrefixKvCacheMode::Cached, "{tag}");
            assert!(roomy.cache_bytes > 0, "{tag}");

            // 2. One byte short of the cache: still admitted, and it runs prefill.
            let (mode, tight_pixels, tight) =
                admit_and_render(generator.as_ref(), &req, SHED, SHED + cache - 1)
                    .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(
                tight,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Prefill,
                    cache_bytes: 0,
                },
                "{tag}"
            );
            let (mode, exact_pixels, _) = admit_and_render(generator.as_ref(), &req, SHED, SHED)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(exact_pixels, tight_pixels, "{tag}");

            // 3. One byte short of the prefill demand: refused, as before the cache existed.
            let refusal = admit_and_render(generator.as_ref(), &req, SHED, SHED - 1)
                .err()
                .unwrap_or_else(|| panic!("{tag}: a request that fits neither must be refused"));
            assert!(
                matches!(&refusal, CoreError::Unsupported(reason)
                if *reason == format!(
                    "{id}: incremental live demand {SHED} exceeds effective budget {}",
                    SHED - 1
                )),
                "{tag}: {refusal:?}"
            );

            if references > 0 {
                // The cache is over the PREFIX: the text rows plus the reference's own latent
                // grid at the snapshot's fitted resolution (1024² → 4096 tokens on the released
                // checkpoints) — not the 32×32 target's four tokens.
                let device = candle_gen::default_device().unwrap();
                let transformer = load_transformer(&root, &device).unwrap();
                let cfg = transformer.config();
                let per_token = prefix_kv_cache_tensor_bytes(
                    cfg.num_layers as u64,
                    (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
                    1,
                    transformer.compute_dtype().size_in_bytes() as u64,
                );
                let vision = load_vision_config(&root)
                    .unwrap()
                    .expect("the miniature snapshot carries a vision tower");
                let (fit_w, fit_h) = reference_fit((edge, edge), 0, &vision).unwrap();
                let reference_tokens = image_tokens(fit_w, fit_h);
                let target_tokens = image_tokens(req.width, req.height);
                assert_ne!(
                    reference_tokens, target_tokens,
                    "{tag}: the case is vacuous"
                );
                assert_eq!(roomy.cache_bytes % per_token, 0, "{tag}");
                let prefix_tokens = roomy.cache_bytes / per_token;
                let text_tokens = prefix_tokens
                    .checked_sub(reference_tokens)
                    .unwrap_or_else(|| panic!("{tag}: {prefix_tokens} prefix tokens"));
                // The text rows of a one-reference prompt: at least the template, and nowhere
                // near another image's worth.
                assert!(
                    (1..=64).contains(&text_tokens),
                    "{tag}: {prefix_tokens} prefix tokens is not text + one fitted reference \
                     ({reference_tokens})"
                );
            }
        }
    }
}

/// **A selection the checkpoint cannot honour is refused, never run as the other mode.** The
/// cache is only valid on a `causal_condition` model (both released checkpoints are one); on a
/// snapshot that is not, a `Cached` selection is a typed refusal naming the field — a caller that
/// priced a cached render is not handed a prefill one in silence — while the same snapshot still
/// renders a request that selects nothing, and reports `prefill`.
///
/// *Mutation that reds this:* dropping the refusal in `generate_impl` (the denoise loop falls
/// back to prefill, as upstream's `use_kv_cache and causal_condition` does, and the render
/// succeeds).
#[test]
fn a_cached_selection_on_a_checkpoint_with_no_causal_prefix_is_refused_by_name() {
    let snapshot = crate::common::copy_of_tiny_snapshot();
    let root = snapshot.path();
    let config = root.join("transformer/config.json");
    let mut value = crate::common::read_json(&config);
    assert_eq!(value["causal_condition"], serde_json::json!(true));
    value["causal_condition"] = serde_json::json!(false);
    std::fs::write(&config, serde_json::to_string_pretty(&value).unwrap()).unwrap();

    let generator = load(BASE, root);
    let refusal = generator
        .generate_with_report(
            &selecting(request(false), PrefixKvCacheMode::Cached),
            &mut |_| {},
        )
        .expect_err("a cached selection cannot be honoured here");
    assert!(
        matches!(&refusal, CoreError::Unsupported(reason)
            if reason.contains("qwen_image_2_1: prefix_kv_cache=cached")
                && reason.contains("causal_condition")),
        "{refusal:?}"
    );
    let (_, unset) = render(generator.as_ref(), &request(false));
    assert_eq!(
        unset,
        PrefixKvCacheReport {
            path: PrefixKvCacheMode::Prefill,
            cache_bytes: 0,
        }
    );
}
