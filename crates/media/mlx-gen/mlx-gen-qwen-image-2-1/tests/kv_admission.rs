//! Prefix-KV-cache admission (sc-25905): the cache is a request-selected execution domain, priced
//! per request, and every render says which path it ran.
//!
//! Everything here runs through the explicit provider catalog's production load path and the
//! public [`mlx_gen::Generator`] trait, under **both** engine ids, on the committed miniature
//! snapshot. What is observed is what the denoise loop did — [`GenerationReport::prefix_kv_cache`]
//! is written from the loop's own report — and what the allocator was bound to while it ran, not
//! what the request asked for.
//!
//! The byte figures are derived in the tests from the tensors the cache stores (blocks × prefix
//! tokens × joint width × element width), never copied from a run.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationReport, MemoryBehaviorRoute, MemoryBudget,
    MemoryMode, MemoryNumericTier, MemoryRunOutcome, MemoryStrategy, PrefixKvCacheMode,
    PrefixKvCacheReport,
};
use mlx_gen::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    WeightsSource,
};
use mlx_gen_qwen_image_2_1::memory_strategy::{derived, AllocatorBounds};
use mlx_gen_qwen_image_2_1::model::{Schedule, Variant};
use mlx_gen_qwen_image_2_1::pipeline::decode_tiling;
use mlx_gen_qwen_image_2_1::{
    create_noise, decode_rgb, denoise_reported, load_text_encoder, load_tokenizer,
    load_transformer, load_vae, system_prompt_drop_count, DenoiseInputs, DenoiseReport,
    EvaluationPath, KvCacheMode, KvCachePath, PromptBranch,
};

use crate::common::{tiny_snapshot, turbo_snapshot};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";

/// Both engine ids over their own miniature snapshot.
fn ids(tmp: &Path) -> [(&'static str, Variant, PathBuf); 2] {
    [
        (BASE, Variant::Base, tiny_snapshot()),
        (TURBO, Variant::Turbo, turbo_snapshot(tmp)),
    ]
}

fn spec(root: &Path) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Resident)
}

fn load(id: &str, root: &Path) -> Box<dyn mlx_gen::Generator> {
    mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(id, &spec(root))
        .expect("the miniature snapshot loads through the catalog")
}

/// A 32×32 text-to-image request; `negative` adds a true-CFG negative branch.
fn request(negative: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        negative_prompt: negative.then(|| "blurry low quality photo".to_owned()),
        true_cfg: negative.then_some(2.5),
        width: 32,
        height: 32,
        steps: Some(4),
        seed: Some(42),
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

/// One render through `generate_with_report`: its pixels, its report, and the allocator's cache
/// limit while the denoise loop was running (read inside a `Step` callback).
struct Rendered {
    pixels: Vec<u8>,
    ran: PrefixKvCacheReport,
    cache_limit_during: usize,
}

fn render(generator: &dyn mlx_gen::Generator, req: &GenerationRequest) -> Rendered {
    let mut cache_limit_during = None;
    let report: GenerationReport = generator
        .generate_with_report(req, &mut |_| {
            cache_limit_during.get_or_insert_with(|| AllocatorBounds::current().1);
        })
        .expect("render");
    let Some(GenerationOutput::Images(mut images)) = report.output else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    Rendered {
        pixels: images.remove(0).pixels,
        ran: report
            .prefix_kv_cache
            .expect("this provider reports the path on every render"),
        cache_limit_during: cache_limit_during.expect("the render reported progress"),
    }
}

/// `req` rendered by hand from the snapshot's components with the cache `mode`, over the schedule
/// the generator resolves for it — the reference a generator render is compared with — plus what
/// the denoise loop reported and the element width / prefix length the cache was built over.
struct Manual {
    pixels: Vec<u8>,
    report: DenoiseReport,
    /// Derived bytes of ONE branch's cache for this request: K and V for every block, each
    /// `[1, prefix_tokens, heads, head_dim]`, at the model's compute dtype.
    positive_branch_bytes: u64,
    negative_branch_bytes: u64,
}

fn manual(variant: Variant, root: &Path, req: &GenerationRequest, mode: KvCacheMode) -> Manual {
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root).unwrap();
    let transformer = load_transformer(root).unwrap();
    let vae = load_vae(root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = encoder
        .encode_prompt(&tokenizer, &req.prompt, drop)
        .unwrap();
    let negative = req
        .negative_prompt
        .as_deref()
        .map(|text| encoder.encode_prompt(&tokenizer, text, drop).unwrap());
    let steps = req.steps.unwrap() as usize;
    let (native, mu) = Schedule::load(variant, root)
        .unwrap()
        .native_sigmas(steps, req.width, req.height)
        .unwrap();
    let sigmas = mlx_gen::resolve_flow_schedule(None, mu, steps, &native);
    let seed = req.seed.unwrap();
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
        None,
    )
    .unwrap()
    .pixels;

    let cfg = transformer.config();
    let element_width = match transformer.compute_dtype() {
        mlx_rs::Dtype::Float32 => 4,
        mlx_rs::Dtype::Bfloat16 | mlx_rs::Dtype::Float16 => 2,
        other => panic!("unexpected compute dtype {other:?}"),
    };
    let branch_bytes = |text: &mlx_rs::Array| {
        derived::prefix_kv_cache_tensor_bytes(
            cfg.num_layers as u64,
            (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
            text.shape()[1] as u64,
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
/// for that request (one cache per prompt branch, each over that branch's own prefix), and its
/// pixels are the hand-assembled cache-on pipeline's, whose loop extracted once per branch and
/// decoded every later evaluation from the cache.
///
/// *Mutations that red this:* flipping the generator's switch to `KvCacheMode::Off` (it reports
/// `prefill` and zero bytes); hard-coding the reported path or bytes; reporting only the positive
/// branch's bytes under true CFG.
#[test]
fn a_cached_selection_renders_from_the_cache_and_reports_its_derived_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    for (id, variant, root) in ids(tmp.path()) {
        let generator = load(id, &root);
        for negative in [false, true] {
            let tag = format!("{id} negative={negative}");
            let req = request(negative);
            let want = manual(variant, &root, &req, KvCacheMode::On);
            assert_eq!(want.report.kv_cache, KvCachePath::Cached, "{tag}");
            let mut sequence = vec![EvaluationPath::Extract];
            sequence.extend([EvaluationPath::Cached; 3]);
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

            let got = render(
                generator.as_ref(),
                &selecting(req, PrefixKvCacheMode::Cached),
            );
            assert_eq!(
                got.ran,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Cached,
                    cache_bytes: derived_bytes,
                },
                "{tag}"
            );
            assert_eq!(got.pixels, want.pixels, "{tag}");
        }
    }
}

/// **A shed selection is exactly the prefill path, and so is a request that selects nothing** —
/// under both ids, with and without a negative branch. The report is `prefill` with zero cache
/// bytes and the pixels are bit-identical to the hand-assembled cache-off pipeline (the same path
/// run twice, so bit equality is the right gate).
///
/// *Mutations that red this:* ignoring a `Prefill` selection (it reports `cached`); defaulting an
/// unset request to the cache.
#[test]
fn a_shed_selection_and_an_unset_request_are_the_prefill_path() {
    let tmp = tempfile::tempdir().unwrap();
    for (id, variant, root) in ids(tmp.path()) {
        let generator = load(id, &root);
        for negative in [false, true] {
            let tag = format!("{id} negative={negative}");
            let req = request(negative);
            let want = manual(variant, &root, &req, KvCacheMode::Off);
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
            let shed = render(
                generator.as_ref(),
                &selecting(req.clone(), PrefixKvCacheMode::Prefill),
            );
            assert_eq!(shed.ran, prefill, "{tag}: shed selection");
            assert_eq!(shed.pixels, want.pixels, "{tag}: shed selection");
            // No memory block at all, and a memory block that selects nothing.
            let mut blank = req.clone();
            blank.memory = Some(GenerationMemory::default());
            for (label, unset) in [("absent block", req.clone()), ("empty block", blank)] {
                let unset = render(generator.as_ref(), &unset);
                assert_eq!(unset.ran, prefill, "{tag}: {label}");
                assert_eq!(unset.pixels, want.pixels, "{tag}: {label}");
            }
        }
    }
}

/// What the SceneWorks worker does for one request, with `total_bytes` of budget: price the
/// request shed and cached from the provider's own derived model, pick the mode
/// ([`PrefixKvCacheMode::select`]), open the request scope through the `Generator` trait with the
/// demand of the mode it picked (the shed demand when neither fits — that is the request it would
/// have submitted before the cache existed), let the scope configure the request, then write the
/// selection onto it and render.
fn admit_and_render(
    generator: &dyn mlx_gen::Generator,
    req: &GenerationRequest,
    total_bytes: u64,
) -> Result<(PrefixKvCacheMode, Rendered), CoreError> {
    let params = (req.width, req.height, derived::TABLE_CONDITIONING_TOKENS);
    let references = req.memory_reference_count();
    let negative = req.negative_prompt.is_some() && req.true_cfg.is_some_and(|scale| scale > 1.0);
    let shed = derived::request_transient_budget_bytes(
        params.0, params.1, params.2, references, negative, None,
    );
    let cache =
        derived::prefix_kv_cache_demand_bytes(params.0, params.1, params.2, references, negative);
    let budget = MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let selected = PrefixKvCacheMode::select(budget, shed, cache);

    let contract = generator.memory_strategy_contract().expect("a contract");
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision: mlx_gen::gen_core::Precision::Bf16,
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
    context.predicted_peak_bytes = match selected {
        Some(PrefixKvCacheMode::Cached) => shed + cache,
        Some(PrefixKvCacheMode::Prefill) | None => shed,
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)?
        .expect("an adopting provider opens a scope");
    let mode = selected.expect("an admitted request has a mode");
    let mut req = req.clone();
    scope.configure_request(&mut req)?;
    let req = selecting(req, mode);
    let rendered = render(generator, &req);
    scope.finish(MemoryRunOutcome::Complete)?;
    Ok((mode, rendered))
}

/// **Admission, end to end** — the story's second and third acceptance criteria, under both ids,
/// with and without a negative branch, and with a reference image (whose fitted 1024² grid is
/// 4096 prefix tokens whatever the target size):
///
/// * a budget that fits the cached demand runs **cached**, under the cached allocator budget;
/// * a budget one byte short of it — above the prefill demand — is **admitted and runs prefill**,
///   under the prefill allocator budget, with zero cache bytes;
/// * a budget one byte short of the prefill demand is **refused**, with the shared safety check's
///   own refusal — the one this request got before the cache existed.
///
/// The path is read off the render's report and the allocator bound off the live allocator inside
/// the denoise loop, so a decision that never reached the generator fails here.
///
/// *Mutations that red this:* ignoring the cache term in the selection (the between-budget
/// request runs cached); ignoring a `Prefill` selection in the generator (it reports `cached`
/// under a prefill budget); ignoring a `Cached` selection (the roomy request reports `prefill`);
/// installing the shed budget for a cached render; admitting the fits-neither request.
#[test]
fn a_request_between_the_two_budgets_is_admitted_and_runs_prefill() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, cache_limit_before) = AllocatorBounds::current();
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
    for (id, _, root) in ids(tmp.path()) {
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
            let negative = req.negative_prompt.is_some();
            let shed = derived::request_transient_budget_bytes(
                req.width,
                req.height,
                derived::TABLE_CONDITIONING_TOKENS,
                references,
                negative,
                None,
            );
            let cache = derived::prefix_kv_cache_demand_bytes(
                req.width,
                req.height,
                derived::TABLE_CONDITIONING_TOKENS,
                references,
                negative,
            );
            assert!(cache > 0, "{tag}: the cache is never free");
            let bound = |transient: u64| cache_limit_before.min(transient as usize);

            // 1. Room for the cache: cached, under the cached budget.
            let (mode, roomy) = admit_and_render(generator.as_ref(), &req, shed + cache)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Cached, "{tag}");
            assert_eq!(roomy.ran.path, PrefixKvCacheMode::Cached, "{tag}");
            assert!(roomy.ran.cache_bytes > 0, "{tag}");
            assert_eq!(
                roomy.cache_limit_during,
                bound(shed + cache),
                "{tag}: a cached render runs under the cached budget"
            );

            // 2. One byte short of the cache: still admitted, and it runs prefill.
            let (mode, tight) = admit_and_render(generator.as_ref(), &req, shed + cache - 1)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(
                tight.ran,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Prefill,
                    cache_bytes: 0,
                },
                "{tag}"
            );
            assert_eq!(
                tight.cache_limit_during,
                bound(shed),
                "{tag}: a prefill render runs under the prefill budget"
            );
            let (mode, exact) = admit_and_render(generator.as_ref(), &req, shed)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(exact.pixels, tight.pixels, "{tag}");

            // 3. One byte short of the prefill demand: refused, as before the cache existed.
            let refusal = admit_and_render(generator.as_ref(), &req, shed - 1)
                .err()
                .unwrap_or_else(|| panic!("{tag}: a request that fits neither must be refused"));
            assert!(
                matches!(&refusal, CoreError::Unsupported(reason)
                if *reason == format!(
                    "{id}: incremental live demand {shed} exceeds effective budget {}",
                    shed - 1
                )),
                "{tag}: {refusal:?}"
            );

            if references > 0 {
                // The cache is over the PREFIX: the text rows plus the reference's own latent
                // grid at the snapshot's fitted resolution (1024² → 4096 tokens on the released
                // checkpoints) — not the 32×32 target's four tokens.
                let transformer = load_transformer(&root).unwrap();
                let cfg = transformer.config();
                assert_eq!(transformer.compute_dtype(), mlx_rs::Dtype::Float32);
                let per_token = derived::prefix_kv_cache_tensor_bytes(
                    cfg.num_layers as u64,
                    (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
                    1,
                    4,
                );
                let vision = mlx_gen_qwen_image_2_1::load_vision_config(&root)
                    .unwrap()
                    .expect("the miniature snapshot carries a vision tower");
                let (fit_w, fit_h) =
                    mlx_gen_qwen_image_2_1::reference::reference_fit((edge, edge), 0, &vision)
                        .unwrap();
                let reference_tokens = derived::image_tokens(fit_w, fit_h);
                let target_tokens = derived::image_tokens(req.width, req.height);
                assert_ne!(
                    reference_tokens, target_tokens,
                    "{tag}: the case is vacuous"
                );
                assert_eq!(roomy.ran.cache_bytes % per_token, 0, "{tag}");
                let prefix_tokens = roomy.ran.cache_bytes / per_token;
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
    assert_eq!(
        AllocatorBounds::current().1,
        cache_limit_before,
        "every request's bounds were restored"
    );
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
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("non-causal-snapshot");
    mlx_gen::quant::copy_dir(&tiny_snapshot(), &root).expect("copy the tiny snapshot");
    let config = root.join("transformer/config.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    assert_eq!(value["causal_condition"], serde_json::json!(true));
    value["causal_condition"] = serde_json::json!(false);
    std::fs::write(&config, serde_json::to_string_pretty(&value).unwrap()).unwrap();

    let generator = load(BASE, &root);
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
    let unset = render(generator.as_ref(), &request(false));
    assert_eq!(
        unset.ran,
        PrefixKvCacheReport {
            path: PrefixKvCacheMode::Prefill,
            cache_bytes: 0,
        }
    );
}
