//! Prefix-KV-cache admission (sc-25905) — the Candle twin of
//! `mlx-gen-qwen-image-2-1/tests/kv_admission.rs`: the cache is a request-selected execution
//! domain, priced per request, held to its price, and every render says which path it ran.
//!
//! Everything here runs through the explicit provider catalog's production load path and the
//! public [`Generator`] trait, under **both** engine ids, on the committed miniature snapshot and
//! on a packed tier of it. What is observed is what the denoise loop did —
//! `GenerationReport::prefix_kv_cache` is written from the loop's own report — not what the
//! request asked for.
//!
//! The byte figures are derived in the tests from the tensors the cache stores (blocks × prefix
//! tokens × joint width × element width), never copied from a run. The element width is the
//! loaded model's own: f32 on the CPU lane, bf16 under `--features cuda` / `--features metal` —
//! on **every tier**, because a packed Linear on this backend dequantizes to the activation's
//! dtype.
//!
//! Numerics: a generator render is compared with a hand-assembled run of the **same** path, so
//! the comparison is exact. Nothing here compares a cached forward with a prefill one — on x86-64
//! the CPU GEMM's kernel choice follows the row count, so those two agree to rounding only
//! (`tests/kv_cache.rs` gates that at `1e-5 + 1e-5·|want|`).

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    CancelFlag, Conditioning, Error as CoreError, GenerationMemory, GenerationOutput,
    GenerationReport, GenerationRequest, Generator, Image, LoadSpec, MemoryBehaviorRoute,
    MemoryBudget, MemoryMode, MemoryNumericTier, MemoryRunContext, MemoryRunOutcome,
    MemoryStrategy, OffloadPolicy, PrefixKvCacheMode, PrefixKvCacheOverflow, PrefixKvCacheReport,
    Quant, WeightsSource,
};
use candle_gen_qwen_image_2_1::memory_strategy::{
    compute_width, image_tokens, prefix_kv_cache_demand_bytes, prefix_kv_cache_model_bytes,
    prefix_kv_cache_priced_prefix_tokens, request_demand_bytes,
    PREFIX_KV_CACHE_CONDITIONING_TOKENS, PREFIX_KV_CACHE_OVERFLOW_WARNING,
};
use candle_gen_qwen_image_2_1::{
    create_noise, decode_rgb, decode_tiling, denoise_reported, load_saved_sigma_schedule,
    load_scheduler_config, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    load_vision_config, prompt_template, reference_fit, scheduler, system_prompt_drop_count,
    DenoiseInputs, DenoiseReport, EvaluationPath, KvCacheMode, KvCachePath, PromptBranch, Tier,
};

use crate::common::{tiny_snapshot, turbo_snapshot};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const PROMPT: &str = "a red fox in the forest";
const NEGATIVE: &str = "blurry low quality photo";
const STEPS: usize = 4;
const SEED: u64 = 42;

/// One loadable fixture: an engine id over a snapshot, dense or packed.
struct Fixture {
    id: &'static str,
    quant: Option<Quant>,
    root: PathBuf,
}

impl Fixture {
    fn tag(&self) -> String {
        format!("{} {:?}", self.id, self.quant)
    }

    fn spec(&self) -> LoadSpec {
        let spec = LoadSpec::new(WeightsSource::Dir(self.root.clone()))
            .with_offload_policy(OffloadPolicy::Resident);
        match self.quant {
            Some(quant) => spec.with_quant(quant),
            None => spec,
        }
    }

    fn load(&self) -> Box<dyn Generator> {
        candle_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(self.id, &self.spec())
            .unwrap_or_else(|e| panic!("{}: loads through the catalog path: {e}", self.tag()))
    }
}

/// Both engine ids over their own dense miniature snapshot.
fn dense(turbo: &tempfile::TempDir) -> [Fixture; 2] {
    [
        Fixture {
            id: BASE,
            quant: None,
            root: tiny_snapshot(),
        },
        Fixture {
            id: TURBO,
            quant: None,
            root: turbo.path().to_path_buf(),
        },
    ]
}

/// The base id over the committed packed `tier` of the miniature snapshot.
fn packed(tmp: &Path, tier: Tier) -> Fixture {
    Fixture {
        id: BASE,
        quant: Some(match tier {
            Tier::Q8 => Quant::Q8,
            Tier::Q4 => Quant::Q4,
            Tier::Bf16 => panic!("the bf16 tier is the dense snapshot"),
        }),
        root: crate::tiers::compose_tier(tmp, tier),
    }
}

/// A 32×32 request for `prompt`; `negative` adds a true-CFG negative branch.
fn request_for(prompt: &str, negative: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt: negative.then(|| NEGATIVE.to_owned()),
        true_cfg: negative.then_some(2.5),
        width: 32,
        height: 32,
        steps: Some(STEPS as u32),
        seed: Some(SEED),
        ..Default::default()
    }
}

fn request(negative: bool) -> GenerationRequest {
    request_for(PROMPT, negative)
}

fn selecting(mut req: GenerationRequest, mode: PrefixKvCacheMode) -> GenerationRequest {
    req.memory = Some(GenerationMemory {
        prefix_kv_cache: Some(mode),
        ..req.memory.unwrap_or_default()
    });
    req
}

fn uses_negative(req: &GenerationRequest) -> bool {
    req.negative_prompt.is_some() && req.true_cfg.is_some_and(|scale| scale > 1.0)
}

/// One render through `generate_with_report`: its images and what it says it ran.
struct Rendered {
    images: Vec<Vec<u8>>,
    ran: PrefixKvCacheReport,
    warnings: Vec<String>,
}

impl Rendered {
    fn pixels(&self) -> &[u8] {
        assert_eq!(self.images.len(), 1);
        &self.images[0]
    }
}

fn render(generator: &dyn Generator, req: &GenerationRequest) -> Rendered {
    let report: GenerationReport = generator
        .generate_with_report(req, &mut |_| {})
        .expect("render");
    let Some(GenerationOutput::Images(images)) = report.output else {
        panic!("images expected");
    };
    Rendered {
        images: images.into_iter().map(|image| image.pixels).collect(),
        ran: report
            .prefix_kv_cache
            .expect("this provider reports the path on every render"),
        warnings: report.warnings.into_iter().map(|w| w.code).collect(),
    }
}

const PREFILL_BY_SELECTION: PrefixKvCacheReport = PrefixKvCacheReport {
    path: PrefixKvCacheMode::Prefill,
    cache_bytes: 0,
    overflow: None,
};

/// The loaded transformer's own shape and compute width, read from the snapshot on the device
/// `load` uses.
struct Shape {
    blocks: u64,
    inner: u64,
    head_dim: u64,
    /// Bytes per element the transformer computes in — and so of every cached K and V.
    element_width: u64,
}

fn shape(root: &Path) -> Shape {
    let device = candle_gen::default_device().unwrap();
    let transformer = load_transformer(root, &device).unwrap();
    let cfg = transformer.config();
    Shape {
        blocks: cfg.num_layers as u64,
        inner: (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
        head_dim: cfg.attention_head_dim as u64,
        element_width: transformer.compute_dtype().size_in_bytes() as u64,
    }
}

impl Shape {
    /// Derived K/V bytes of one branch's cache over `prefix_tokens`: K and V for every block,
    /// each `[1, heads, prefix_tokens, head_dim]`, at the transformer's compute width.
    fn kv_bytes(&self, prefix_tokens: u64) -> u64 {
        2 * self.blocks * prefix_tokens * self.inner * self.element_width
    }

    /// What the generator prices this request's caches at, in this model's own terms: the priced
    /// prefix per branch at the **build's** priced width, plus each branch's target RoPE rows.
    fn priced_bytes(&self, req: &GenerationRequest) -> u64 {
        let branches = if uses_negative(req) { 2 } else { 1 };
        prefix_kv_cache_model_bytes(
            self.blocks,
            self.inner,
            self.head_dim,
            compute_width(),
            &vec![prefix_kv_cache_priced_prefix_tokens(req.memory_reference_count()); branches],
            image_tokens(req.width, req.height),
        )
    }

    fn rope_bytes(&self, req: &GenerationRequest) -> u64 {
        2 * image_tokens(req.width, req.height) * (self.head_dim / 2) * 4
    }
}

/// The invariant every cached render must hold: it never holds more than it was priced at.
fn assert_within_price(tag: &str, fixture: &Fixture, req: &GenerationRequest, ran: &Rendered) {
    assert_eq!(ran.ran.path, PrefixKvCacheMode::Cached, "{tag}");
    assert_eq!(ran.ran.overflow, None, "{tag}");
    assert!(ran.warnings.is_empty(), "{tag}: {:?}", ran.warnings);
    let shape = shape(&fixture.root);
    let branches = if uses_negative(req) { 2 } else { 1 };
    let held = ran.ran.cache_bytes + branches * shape.rope_bytes(req);
    let priced = shape.priced_bytes(req);
    assert!(
        held <= priced,
        "{tag}: a cached render holds {held} bytes against {priced} priced"
    );
}

/// `req` rendered by hand from the snapshot's components with the cache `mode`, over the schedule
/// the generator resolves for it, plus what the denoise loop reported and the text rows of each
/// branch.
struct Manual {
    pixels: Vec<u8>,
    report: DenoiseReport,
    positive_text_rows: u64,
    negative_text_rows: u64,
}

fn manual(fixture: &Fixture, req: &GenerationRequest, mode: KvCacheMode) -> Manual {
    let root = &fixture.root;
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
    let sigmas = if fixture.id == TURBO {
        scheduler::resample_saved_sigmas(
            &load_saved_sigma_schedule(root).unwrap().sample_sigmas,
            STEPS,
        )
        .unwrap()
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
    Manual {
        pixels,
        positive_text_rows: prompt.dims()[1] as u64,
        negative_text_rows: negative.as_ref().map_or(0, |n| n.dims()[1] as u64),
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

/// **A cached selection renders from the cache and says so** — under both ids and on a packed
/// tier, with and without a negative branch. The generator's report is `cached`, its cache bytes
/// are the derived figure for that request (one cache per prompt branch, each over that branch's
/// own prefix, at the loaded model's compute width), it stays within its price, and its pixels
/// are the hand-assembled cache-on pipeline's, whose loop extracted once per branch and decoded
/// every later evaluation from the cache.
///
/// **The packed tier's cache is as wide as the dense one's.** On this backend a packed Linear
/// dequantizes its weight to the activation's dtype (`QLinear::forward`,
/// `candle-gen/src/quant/mod.rs`), so the DiT computes at the dtype it was loaded at on every
/// tier: the q8 fixture's `compute_dtype` equals the dense fixture's, and its reported cache
/// bytes are `2 · blocks · prefix · inner · that width`.
///
/// *Mutations that red this:* flipping the generator's switch to `KvCacheMode::Off` (it reports
/// `prefill` and zero bytes); hard-coding the reported path or bytes; reporting only the positive
/// branch's bytes under true CFG; deriving the bytes at bf16 on the f32 CPU lane.
#[test]
fn a_cached_selection_renders_from_the_cache_and_reports_its_derived_bytes() {
    let turbo = turbo_snapshot();
    let tmp = tempfile::tempdir().unwrap();
    let [base, turbo] = dense(&turbo);
    let dense_width = shape(&base.root).element_width;
    for fixture in [base, turbo, packed(tmp.path(), Tier::Q8)] {
        let generator = fixture.load();
        let shape = shape(&fixture.root);
        assert_eq!(
            shape.element_width,
            dense_width,
            "{}: the compute width follows the device, not the tier",
            fixture.tag()
        );
        assert_eq!(
            shape.element_width,
            candle_gen_qwen_image_2_1::memory_strategy::compute_width_on(
                &candle_gen::default_device().unwrap()
            ),
            "{}: and it is the width this device is priced at",
            fixture.tag()
        );
        for negative in [false, true] {
            let tag = format!("{} negative={negative}", fixture.tag());
            let req = request(negative);
            let want = manual(&fixture, &req, KvCacheMode::On);
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
            assert!(want.positive_text_rows > 0, "{tag}");
            assert_eq!(negative, want.negative_text_rows > 0, "{tag}");
            let derived_bytes =
                shape.kv_bytes(want.positive_text_rows) + shape.kv_bytes(want.negative_text_rows);
            assert_eq!(
                want.report.kv_cache_bytes as u64, derived_bytes,
                "{tag}: the loop's caches hold exactly the derived bytes"
            );

            let req = selecting(req, PrefixKvCacheMode::Cached);
            let got = render(generator.as_ref(), &req);
            assert_eq!(
                got.ran,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Cached,
                    cache_bytes: derived_bytes,
                    overflow: None,
                },
                "{tag}"
            );
            assert_eq!(got.pixels(), want.pixels, "{tag}");
            assert_within_price(&tag, &fixture, &req, &got);
        }
    }
}

/// **A shed selection is exactly the prefill path, and so is a request that selects nothing** —
/// under both ids, with and without a negative branch. The report is `prefill` by selection (no
/// overflow), with zero cache bytes, and the pixels equal the hand-assembled cache-off pipeline's
/// (the same path run twice).
///
/// *Mutations that red this:* ignoring a `Prefill` selection (it reports `cached`); defaulting an
/// unset request to the cache.
#[test]
fn a_shed_selection_and_an_unset_request_are_the_prefill_path() {
    let turbo = turbo_snapshot();
    for fixture in dense(&turbo) {
        let generator = fixture.load();
        for negative in [false, true] {
            let tag = format!("{} negative={negative}", fixture.tag());
            let req = request(negative);
            let want = manual(&fixture, &req, KvCacheMode::Off);
            assert_eq!(want.report.kv_cache, KvCachePath::Prefill, "{tag}");
            assert_eq!(want.report.kv_cache_bytes, 0, "{tag}");
            assert!(
                want.report
                    .evaluations
                    .iter()
                    .all(|evaluation| evaluation.path == EvaluationPath::Prefill),
                "{tag}"
            );
            let shed = render(
                generator.as_ref(),
                &selecting(req.clone(), PrefixKvCacheMode::Prefill),
            );
            assert_eq!(shed.ran, PREFILL_BY_SELECTION, "{tag}: shed selection");
            assert_eq!(shed.pixels(), want.pixels, "{tag}: shed selection");
            assert!(shed.warnings.is_empty(), "{tag}");
            // No memory block at all, and a memory block that selects nothing.
            let mut blank = req.clone();
            blank.memory = Some(GenerationMemory::default());
            for (label, unset) in [("absent block", req.clone()), ("empty block", blank)] {
                let unset = render(generator.as_ref(), &unset);
                assert_eq!(unset.ran, PREFILL_BY_SELECTION, "{tag}: {label}");
                assert_eq!(unset.pixels(), want.pixels, "{tag}: {label}");
            }
        }
    }
}

/// **Several images hold one image's caches at a time.** A `count: 2` cached request renders two
/// images, each with its own caches built and dropped inside its own denoise loop, and reports
/// the bytes one image holds — not twice that.
///
/// *Mutation that reds this:* accumulating the reported bytes across images.
#[test]
fn a_two_image_cached_request_reports_one_images_cache_bytes() {
    let turbo = turbo_snapshot();
    for fixture in dense(&turbo) {
        let generator = fixture.load();
        let tag = fixture.tag();
        let single = selecting(request(true), PrefixKvCacheMode::Cached);
        let one = render(generator.as_ref(), &single);
        let mut double = single.clone();
        double.count = 2;
        let two = render(generator.as_ref(), &double);
        assert_eq!(two.images.len(), 2, "{tag}");
        assert_eq!(two.images[0], one.pixels(), "{tag}: the first image");
        assert_ne!(two.images[1], two.images[0], "{tag}: a second seed");
        assert!(one.ran.cache_bytes > 0, "{tag}");
        assert_eq!(two.ran, one.ran, "{tag}: one image's caches, not two");
        assert_within_price(&tag, &fixture, &double, &two);
    }
}

/// A prompt whose positive branch has exactly `rows` text rows on `fixture`'s tokenizer, found
/// by growing a filler prompt one word at a time around an estimate.
fn prompt_with_text_rows(fixture: &Fixture, rows: u64) -> String {
    let tokenizer = load_tokenizer(&fixture.root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let text_rows = |prompt: &str| -> u64 {
        let ids = tokenizer
            .tokenize_preformatted(&prompt_template(prompt))
            .unwrap()
            .ids;
        (ids.len() - drop) as u64
    };
    let prompt_of = |words: usize| vec!["fox"; words].join(" ");
    let (one, two) = (text_rows(&prompt_of(1)), text_rows(&prompt_of(2)));
    let per_word = two - one;
    assert!(per_word > 0, "the filler word adds text rows");
    let mut words = ((rows - one) / per_word + 1) as usize;
    while text_rows(&prompt_of(words)) > rows && words > 1 {
        words -= 1;
    }
    while text_rows(&prompt_of(words)) < rows {
        words += 1;
    }
    let prompt = prompt_of(words);
    assert_eq!(
        text_rows(&prompt),
        rows,
        "no filler prompt has exactly {rows} text rows on this tokenizer"
    );
    prompt
}

/// **A cached request is held to its price: at the bound it runs cached, one text row past it
/// it runs prefill and says why** — never cached above its price, never a failed render.
///
/// The bound is the priced text rows, in bytes: where the loaded model computes at the width the
/// build prices (the CPU lane: f32 priced, f32 run; a GPU lane on its GPU: bf16, bf16) that is
/// exactly [`PREFIX_KV_CACHE_CONDITIONING_TOKENS`] rows. A prompt at the bound renders cached
/// holding exactly its priced K/V; one row more renders **prefill** with
/// `overflow = Some { actual, priced }`, a `prefix_kv_cache_shed_over_priced_bound` warning, zero
/// cache bytes, and pixels equal to the cache-off pipeline. Dense and packed alike.
///
/// *Mutations that red this:* dropping the check (the long prompt runs cached above its price);
/// comparing with `>=` (the at-bound prompt sheds); failing the render instead of shedding;
/// reporting the shed as prefill-by-selection.
#[test]
fn a_prompt_past_the_priced_text_bound_runs_prefill_and_reports_the_overflow() {
    let tmp = tempfile::tempdir().unwrap();
    for fixture in [
        Fixture {
            id: BASE,
            quant: None,
            root: tiny_snapshot(),
        },
        packed(tmp.path(), Tier::Q4),
    ] {
        let tag = fixture.tag();
        let generator = fixture.load();
        let shape = shape(&fixture.root);
        // Rows the priced bytes cover at the width this model actually computes in.
        let bound = PREFIX_KV_CACHE_CONDITIONING_TOKENS * compute_width() / shape.element_width;

        let at_bound = selecting(
            request_for(&prompt_with_text_rows(&fixture, bound), false),
            PrefixKvCacheMode::Cached,
        );
        let cached = render(generator.as_ref(), &at_bound);
        assert_eq!(
            cached.ran,
            PrefixKvCacheReport {
                path: PrefixKvCacheMode::Cached,
                cache_bytes: shape.kv_bytes(bound),
                overflow: None,
            },
            "{tag}: at the bound"
        );
        assert_within_price(&tag, &fixture, &at_bound, &cached);
        // At the bound the request holds exactly what it was priced at.
        assert_eq!(
            cached.ran.cache_bytes + shape.rope_bytes(&at_bound),
            shape.priced_bytes(&at_bound),
            "{tag}"
        );

        let over_prompt = prompt_with_text_rows(&fixture, bound + 1);
        let over = selecting(request_for(&over_prompt, false), PrefixKvCacheMode::Cached);
        let shed = render(generator.as_ref(), &over);
        assert_eq!(
            shed.ran,
            PrefixKvCacheReport {
                path: PrefixKvCacheMode::Prefill,
                cache_bytes: 0,
                overflow: Some(PrefixKvCacheOverflow {
                    actual_bytes: shape.kv_bytes(bound + 1) + shape.rope_bytes(&over),
                    priced_bytes: shape.priced_bytes(&over),
                }),
            },
            "{tag}: one row past the bound"
        );
        assert_eq!(
            shed.warnings,
            [PREFIX_KV_CACHE_OVERFLOW_WARNING],
            "{tag}: the shed is visible"
        );
        let want = manual(
            &fixture,
            &request_for(&over_prompt, false),
            KvCacheMode::Off,
        );
        assert_eq!(shed.pixels(), want.pixels, "{tag}: the prefill output");
        // The same prompt with no selection is prefill by selection: no overflow, no warning.
        let unset = render(generator.as_ref(), &request_for(&over_prompt, false));
        assert_eq!(unset.ran, PREFILL_BY_SELECTION, "{tag}");
        assert!(unset.warnings.is_empty(), "{tag}");
    }
}

/// The worker's admitted `Resident` context for `req`, with an unlimited budget.
fn context(
    fixture: &Fixture,
    generator: &dyn Generator,
    req: &GenerationRequest,
) -> MemoryRunContext {
    let contract = generator.memory_strategy_contract().expect("a contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision: candle_gen::gen_core::Precision::Bf16,
            quant: fixture.quant,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: req.memory_reference_count(),
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )
    .expect("a valid behaviour context");
    context.optimization_authority = Default::default();
    context.geometry.width = req.width;
    context.geometry.height = req.height;
    context.budget.total_bytes = u64::MAX;
    context.predicted_peak_bytes = 0;
    context
}

/// What the SceneWorks worker does for one request whose cache-shed demand it prices at `shed`,
/// with `total_bytes` of budget: price the cache from the provider's derived model, pick the mode
/// ([`PrefixKvCacheMode::select`]), open the request scope through the `Generator` trait with the
/// demand of the mode it picked (the shed demand when neither fits — that is the request it would
/// have submitted before the cache existed), let the scope configure the request, and only
/// **then** write the selection onto it and render.
fn admit_and_render(
    fixture: &Fixture,
    generator: &dyn Generator,
    req: &GenerationRequest,
    shed: u64,
    total_bytes: u64,
) -> Result<(PrefixKvCacheMode, GenerationRequest, Rendered), CoreError> {
    let references = req.memory_reference_count();
    let negative = uses_negative(req);
    let cache = prefix_kv_cache_demand_bytes(req.width, req.height, references, negative);
    let budget = MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let selected = PrefixKvCacheMode::select(budget, shed, cache);
    let mut context = context(fixture, generator, req);
    context.budget = budget;
    context.predicted_peak_bytes = request_demand_bytes(
        selected.unwrap_or(PrefixKvCacheMode::Prefill),
        shed,
        req.width,
        req.height,
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
    let rendered = render(generator, &req);
    scope.finish(MemoryRunOutcome::Complete)?;
    Ok((mode, req, rendered))
}

/// **Admission, end to end** — the story's second and third acceptance criteria, under both ids
/// and on a packed tier, with and without a negative branch, with a square reference and with a
/// 3:1 one (whose fitted grid is not the square's):
///
/// * a budget that fits the cached demand runs **cached**, within its price;
/// * a budget one byte short of it — above the prefill demand — is **admitted and runs prefill**
///   by selection, with zero cache bytes;
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
    let tmp = tempfile::tempdir().unwrap();
    let reference = |width: u32, height: u32| Conditioning::Reference {
        image: Image {
            width,
            height,
            pixels: (0..width * height * 3)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect(),
        },
        strength: None,
    };
    let [base, turbo] = dense(&turbo);
    for fixture in [base, turbo, packed(tmp.path(), Tier::Q4)] {
        let generator = fixture.load();
        let with = |conditioning: Conditioning| GenerationRequest {
            conditioning: vec![conditioning],
            ..request(false)
        };
        for (label, req) in [
            ("text-to-image", request(false)),
            ("true CFG", request(true)),
            ("one square reference", with(reference(64, 64))),
            ("one 3:1 reference", with(reference(96, 32))),
        ] {
            let tag = format!("{} {label}", fixture.tag());
            let cache = prefix_kv_cache_demand_bytes(
                req.width,
                req.height,
                req.memory_reference_count(),
                uses_negative(&req),
            );
            assert!(cache > 0, "{tag}: the cache is never free");

            // 1. Room for the cache: cached, within its price.
            let (mode, admitted, roomy) =
                admit_and_render(&fixture, generator.as_ref(), &req, SHED, SHED + cache)
                    .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Cached, "{tag}");
            assert!(roomy.ran.cache_bytes > 0, "{tag}");
            assert_within_price(&tag, &fixture, &admitted, &roomy);

            // 2. One byte short of the cache: still admitted, and it runs prefill.
            let (mode, _, tight) =
                admit_and_render(&fixture, generator.as_ref(), &req, SHED, SHED + cache - 1)
                    .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(tight.ran, PREFILL_BY_SELECTION, "{tag}");
            let (mode, _, exact) = admit_and_render(&fixture, generator.as_ref(), &req, SHED, SHED)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(exact.pixels(), tight.pixels(), "{tag}");

            // 3. One byte short of the prefill demand: refused, as before the cache existed.
            let refusal = admit_and_render(&fixture, generator.as_ref(), &req, SHED, SHED - 1)
                .err()
                .unwrap_or_else(|| panic!("{tag}: a request that fits neither must be refused"));
            assert!(
                matches!(&refusal, CoreError::Unsupported(reason)
                if *reason == format!(
                    "{}: incremental live demand {SHED} exceeds effective budget {}",
                    fixture.id,
                    SHED - 1
                )),
                "{tag}: {refusal:?}"
            );

            if let Some(Conditioning::Reference { image, .. }) = req.conditioning.first() {
                // The cache is over the PREFIX: the text rows plus the reference's own latent
                // grid at the snapshot's fitted resolution — not the 32×32 target's four tokens.
                let shape = shape(&fixture.root);
                let per_token = shape.kv_bytes(1);
                let vision = load_vision_config(&fixture.root)
                    .unwrap()
                    .expect("the miniature snapshot carries a vision tower");
                let (fit_w, fit_h) =
                    reference_fit((image.width, image.height), 0, &vision).unwrap();
                let reference_tokens = image_tokens(fit_w, fit_h);
                let target_tokens = image_tokens(req.width, req.height);
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
}

/// **The ordering trap, pinned.** The request scope's `configure_request` *replaces*
/// `GenerationRequest::memory` — with `None` for a `Resident` selection on this provider — so a
/// `Cached` selection written **before** it is lost: the render runs prefill, by "selection", and
/// nothing but the report says so. Written **after** it, the same request runs cached.
///
/// This is the documented caller contract, not a defect fixed here: the scope owns the block.
///
/// *Mutation that reds this:* the scope merging instead of replacing the block (the early
/// selection would survive) — at which point the documented ordering rule is stale and must be
/// rewritten with it.
#[test]
fn a_selection_written_before_the_scope_configures_the_request_is_lost() {
    let turbo = turbo_snapshot();
    for fixture in dense(&turbo) {
        let tag = fixture.tag();
        let generator = fixture.load();
        let context = context(&fixture, generator.as_ref(), &request(false));

        let mut early = selecting(request(false), PrefixKvCacheMode::Cached);
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .unwrap()
            .expect("a scope");
        scope.configure_request(&mut early).unwrap();
        assert_eq!(early.memory, None, "{tag}: the scope replaced the block");
        let lost = render(generator.as_ref(), &early);
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert_eq!(
            lost.ran, PREFILL_BY_SELECTION,
            "{tag}: the selection was lost"
        );

        let mut late = request(false);
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .unwrap()
            .expect("a scope");
        scope.configure_request(&mut late).unwrap();
        let late = selecting(late, PrefixKvCacheMode::Cached);
        let kept = render(generator.as_ref(), &late);
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert_eq!(kept.ran.path, PrefixKvCacheMode::Cached, "{tag}");
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

    let fixture = Fixture {
        id: BASE,
        quant: None,
        root: root.to_path_buf(),
    };
    let generator = fixture.load();
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
    assert_eq!(unset.ran, PREFILL_BY_SELECTION);
}
