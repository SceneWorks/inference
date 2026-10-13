//! Prefix-KV-cache admission (sc-25905): the cache is a request-selected execution domain, priced
//! per request, held to its price, and every render says which path it ran.
//!
//! Everything here runs through the explicit provider catalog's production load path and the
//! public [`mlx_gen::Generator`] trait, under **both** engine ids, on the committed miniature
//! snapshot and on packed tiers converted from it. What is observed is what the denoise loop did
//! — `GenerationReport::prefix_kv_cache` is written from the loop's own report — and what the
//! allocator was bound to while it ran, not what the request asked for.
//!
//! The byte figures are derived in the tests from the tensors the cache stores (blocks × prefix
//! tokens × joint width × element width), never copied from a run.
//!
//! **The miniature dense snapshot is f32** where the released one is bf16, so on the dense
//! fixture a cache element is twice as wide as the bf16 tier prices it. That is deliberate
//! coverage, not an accident: the generator holds a cached render to its priced bytes, and the
//! dense fixture is the checkpoint that "computes wider than its tier". The packed (q8 / q4)
//! fixtures compute in f32 and are priced at f32, like the released packed tiers.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationReport, MemoryBehaviorRoute, MemoryBudget,
    MemoryMode, MemoryNumericTier, MemoryRunOutcome, MemoryStrategy, PrefixKvCacheMode,
    PrefixKvCacheOverflow, PrefixKvCacheReport, Quant,
};
use mlx_gen::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    WeightsSource,
};
use mlx_gen_qwen_image_2_1::memory_strategy::{
    derived, AllocatorBounds, PREFIX_KV_CACHE_OVERFLOW_WARNING,
};
use mlx_gen_qwen_image_2_1::model::{Schedule, Variant};
use mlx_gen_qwen_image_2_1::pipeline::decode_tiling;
use mlx_gen_qwen_image_2_1::{
    create_noise, decode_rgb, denoise_reported, load_text_encoder, load_tokenizer,
    load_transformer, load_vae, prequantize_turnkey, system_prompt_drop_count, DenoiseInputs,
    DenoiseReport, EvaluationPath, KvCacheMode, KvCachePath, PromptBranch, Tier,
};

use crate::common::{tiny_snapshot, turbo_snapshot};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const PROMPT: &str = "a red fox in the forest";
const NEGATIVE: &str = "blurry low quality photo";

/// One loadable fixture: an engine id over a snapshot at a tier.
struct Fixture {
    id: &'static str,
    variant: Variant,
    tier: Tier,
    root: PathBuf,
}

impl Fixture {
    fn tag(&self) -> String {
        format!("{} {:?}", self.id, self.tier)
    }

    fn quant(&self) -> Option<Quant> {
        match self.tier {
            Tier::Bf16 => None,
            Tier::Q8 => Some(Quant::Q8),
            Tier::Q4 => Some(Quant::Q4),
        }
    }

    fn spec(&self) -> LoadSpec {
        let spec = LoadSpec::new(WeightsSource::Dir(self.root.clone()))
            .with_offload_policy(OffloadPolicy::Resident);
        match self.quant() {
            Some(quant) => spec.with_quant(quant),
            None => spec,
        }
    }

    fn load(&self) -> Box<dyn mlx_gen::Generator> {
        mlx_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(self.id, &self.spec())
            .unwrap_or_else(|e| panic!("{}: loads through the catalog: {e}", self.tag()))
    }
}

/// Both engine ids over their own dense miniature snapshot.
fn dense(tmp: &Path) -> [Fixture; 2] {
    [
        Fixture {
            id: BASE,
            variant: Variant::Base,
            tier: Tier::Bf16,
            root: tiny_snapshot(),
        },
        Fixture {
            id: TURBO,
            variant: Variant::Turbo,
            tier: Tier::Bf16,
            root: turbo_snapshot(tmp),
        },
    ]
}

/// A packed tier converted from an engine id's miniature snapshot, under its own directory of
/// `tmp`.
fn packed(tmp: &Path, id: &'static str, tier: Tier) -> Fixture {
    let scratch = tmp.join(format!("{id}-{}", tier.dir_name()));
    std::fs::create_dir_all(&scratch).unwrap();
    let (variant, source) = match id {
        BASE => (Variant::Base, tiny_snapshot()),
        _ => (Variant::Turbo, turbo_snapshot(&scratch)),
    };
    let root = scratch.join("tier");
    prequantize_turnkey(&source, &root, tier).expect("the miniature snapshot converts");
    Fixture {
        id,
        variant,
        tier,
        root,
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
        steps: Some(4),
        seed: Some(42),
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

/// One render through `generate_with_report`: its images, its report, and the allocator's limits
/// while the denoise loop was running (read inside the first progress callback).
struct Rendered {
    images: Vec<Vec<u8>>,
    ran: PrefixKvCacheReport,
    warnings: Vec<String>,
    /// `(memory limit, cache limit)` during the denoise loop.
    limits_during: (usize, usize),
}

impl Rendered {
    fn pixels(&self) -> &[u8] {
        assert_eq!(self.images.len(), 1);
        &self.images[0]
    }
}

fn render(generator: &dyn mlx_gen::Generator, req: &GenerationRequest) -> Rendered {
    let mut limits_during = None;
    let report: GenerationReport = generator
        .generate_with_report(req, &mut |_| {
            limits_during.get_or_insert_with(AllocatorBounds::current);
        })
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
        limits_during: limits_during.expect("the render reported progress"),
    }
}

const PREFILL_BY_SELECTION: PrefixKvCacheReport = PrefixKvCacheReport {
    path: PrefixKvCacheMode::Prefill,
    cache_bytes: 0,
    overflow: None,
};

/// The loaded transformer's own shape and compute width, read from the snapshot.
struct Shape {
    blocks: u64,
    inner: u64,
    head_dim: u64,
    /// Bytes per element the transformer computes in — and so of every cached K and V.
    element_width: u64,
}

fn shape(root: &Path) -> Shape {
    let transformer = load_transformer(root).unwrap();
    let cfg = transformer.config();
    Shape {
        blocks: cfg.num_layers as u64,
        inner: (cfg.num_attention_heads * cfg.attention_head_dim) as u64,
        head_dim: cfg.attention_head_dim as u64,
        element_width: match transformer.compute_dtype() {
            mlx_rs::Dtype::Float32 => 4,
            mlx_rs::Dtype::Bfloat16 | mlx_rs::Dtype::Float16 => 2,
            other => panic!("unexpected compute dtype {other:?}"),
        },
    }
}

impl Shape {
    /// Derived K/V bytes of one branch's cache over `prefix_tokens`: K and V for every block,
    /// each `[1, prefix_tokens, heads, head_dim]`, at the transformer's compute width.
    fn kv_bytes(&self, prefix_tokens: u64) -> u64 {
        2 * self.blocks * prefix_tokens * self.inner * self.element_width
    }

    /// What the generator prices this request's caches at, in this model's own terms: the priced
    /// prefix per branch at the **tier's** width, plus each branch's target RoPE rows.
    fn priced_bytes(&self, tier: Tier, req: &GenerationRequest) -> u64 {
        let branches = if uses_negative(req) { 2 } else { 1 };
        derived::prefix_kv_cache_model_bytes(
            self.blocks,
            self.inner,
            self.head_dim,
            derived::dit_compute_width(tier),
            &vec![
                derived::prefix_kv_cache_priced_prefix_tokens(req.memory_reference_count());
                branches
            ],
            derived::image_tokens(req.width, req.height),
        )
    }

    fn rope_bytes(&self, req: &GenerationRequest) -> u64 {
        2 * derived::image_tokens(req.width, req.height) * (self.head_dim / 2) * 4
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
    let priced = shape.priced_bytes(fixture.tier, req);
    assert!(
        held <= priced,
        "{tag}: a cached render holds {held} bytes against {priced} priced"
    );
}

/// `req` rendered by hand from the snapshot's components with the cache `mode`, over the schedule
/// the generator resolves for it — the reference a generator render is compared with — plus what
/// the denoise loop reported and the text rows of each branch.
struct Manual {
    pixels: Vec<u8>,
    report: DenoiseReport,
    positive_text_rows: u64,
    negative_text_rows: u64,
}

fn manual(fixture: &Fixture, req: &GenerationRequest, mode: KvCacheMode) -> Manual {
    let root = &fixture.root;
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
    let (native, mu) = Schedule::load(fixture.variant, root)
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
    Manual {
        pixels,
        positive_text_rows: prompt.shape()[1] as u64,
        negative_text_rows: negative.as_ref().map_or(0, |n| n.shape()[1] as u64),
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
/// for that request (one cache per prompt branch, each over that branch's own prefix), it stays
/// within its price, and its pixels are the hand-assembled cache-on pipeline's, whose loop
/// extracted once per branch and decoded every later evaluation from the cache.
///
/// *Mutations that red this:* flipping the generator's switch to `KvCacheMode::Off` (it reports
/// `prefill` and zero bytes); hard-coding the reported path or bytes; reporting only the positive
/// branch's bytes under true CFG.
#[test]
fn a_cached_selection_renders_from_the_cache_and_reports_its_derived_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    for fixture in dense(tmp.path()) {
        let generator = fixture.load();
        let shape = shape(&fixture.root);
        for negative in [false, true] {
            let tag = format!("{} negative={negative}", fixture.tag());
            let req = request(negative);
            let want = manual(&fixture, &req, KvCacheMode::On);
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

/// **A packed tier's cache is f32, and it is priced at f32** — the review finding this story's
/// fix pass settled by observation. On a q8 and a q4 tier converted from the miniature snapshot,
/// under both ids: the loaded transformer computes in f32 (its Linears are packed, so there is no
/// dense weight dtype to follow), a cached render's reported bytes are exactly
/// `2 · blocks · prefix · inner · 4`, and the width the tier is priced at
/// ([`derived::dit_compute_width`]) is that same 4 — twice the bf16 tier's.
///
/// The bytes come from the cache's own arrays (`PrefixKvCache::nbytes` sums `k.nbytes()` and
/// `v.nbytes()`), so a cache that held bf16 would report half of this.
///
/// *Mutations that red this:* pricing a packed tier at bf16 (`dit_compute_width` returning 2);
/// the transformer casting its packed-tier activations to bf16.
#[test]
fn a_packed_tiers_cache_is_f32_and_is_priced_at_f32() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(derived::dit_compute_width(Tier::Bf16), 2);
    for (id, tier) in [(BASE, Tier::Q8), (BASE, Tier::Q4), (TURBO, Tier::Q8)] {
        let fixture = packed(tmp.path(), id, tier);
        let tag = fixture.tag();
        let shape = shape(&fixture.root);
        assert_eq!(
            load_transformer(&fixture.root).unwrap().compute_dtype(),
            mlx_rs::Dtype::Float32,
            "{tag}: a packed DiT computes in f32"
        );
        assert_eq!(shape.element_width, 4, "{tag}");
        assert_eq!(
            derived::dit_compute_width(tier),
            shape.element_width,
            "{tag}: the tier is priced at the width it computes in"
        );

        let generator = fixture.load();
        for negative in [false, true] {
            let req = request(negative);
            let want = manual(&fixture, &req, KvCacheMode::On);
            let derived_bytes = 2
                * shape.blocks
                * (want.positive_text_rows + want.negative_text_rows)
                * shape.inner
                * 4;
            let req = selecting(req, PrefixKvCacheMode::Cached);
            let got = render(generator.as_ref(), &req);
            assert_eq!(
                got.ran,
                PrefixKvCacheReport {
                    path: PrefixKvCacheMode::Cached,
                    cache_bytes: derived_bytes,
                    overflow: None,
                },
                "{tag} negative={negative}"
            );
            assert_eq!(got.pixels(), want.pixels, "{tag} negative={negative}");
            assert_within_price(&tag, &fixture, &req, &got);
        }
    }
}

/// **A shed selection is exactly the prefill path, and so is a request that selects nothing** —
/// under both ids, with and without a negative branch. The report is `prefill` by selection
/// (no overflow), with zero cache bytes, and the pixels are bit-identical to the hand-assembled
/// cache-off pipeline (the same path run twice, so bit equality is the right gate).
///
/// *Mutations that red this:* ignoring a `Prefill` selection (it reports `cached`); defaulting an
/// unset request to the cache.
#[test]
fn a_shed_selection_and_an_unset_request_are_the_prefill_path() {
    let tmp = tempfile::tempdir().unwrap();
    for fixture in dense(tmp.path()) {
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
    let tmp = tempfile::tempdir().unwrap();
    for fixture in dense(tmp.path()) {
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
            .tokenize_preformatted(&mlx_gen_qwen_image_2_1::prompt_template(prompt))
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
/// On a packed (q8) tier, where the cache is f32 and priced at f32, the bound is the priced 1024
/// text rows: a 1024-row prompt renders cached holding exactly its priced K/V, and a 1025-row
/// prompt renders **prefill** with `overflow = Some { actual, priced }`, a
/// `prefix_kv_cache_shed_over_priced_bound` warning, zero cache bytes, and pixels equal to the
/// cache-off pipeline.
///
/// On the dense miniature snapshot — f32 where the bf16 tier is priced at 2 bytes — the same
/// rule sheds at half the rows: 512 rows is within price, 513 is not. That is the "checkpoint
/// computes wider than its tier is priced at" case, end to end.
///
/// *Mutations that red this:* dropping the check (the long prompt runs cached above its price);
/// comparing with `>=` (the at-bound prompt sheds); pricing the comparison at the model's own
/// width (the dense fixture is never caught); failing the render instead of shedding; reporting
/// the shed as prefill-by-selection.
#[test]
fn a_prompt_past_the_priced_text_bound_runs_prefill_and_reports_the_overflow() {
    let tmp = tempfile::tempdir().unwrap();
    for (fixture, bound) in [
        (
            packed(tmp.path(), BASE, Tier::Q8),
            derived::PREFIX_KV_CACHE_CONDITIONING_TOKENS,
        ),
        (
            Fixture {
                id: BASE,
                variant: Variant::Base,
                tier: Tier::Bf16,
                root: tiny_snapshot(),
            },
            // An f32 cache under a 2-byte price: half the priced rows.
            derived::PREFIX_KV_CACHE_CONDITIONING_TOKENS / 2,
        ),
    ] {
        let tag = fixture.tag();
        let generator = fixture.load();
        let shape = shape(&fixture.root);

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
            shape.priced_bytes(fixture.tier, &at_bound),
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
                    priced_bytes: shape.priced_bytes(fixture.tier, &over),
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

/// `(cache-shed transient, cache surcharge)` for `req` on `fixture`'s tier — what the consumer
/// prices from the provider's own derived model **at the tier it loaded**.
fn demands(fixture: &Fixture, req: &GenerationRequest) -> (u64, u64) {
    let references = req.memory_reference_count();
    let negative = uses_negative(req);
    (
        derived::request_transient_budget_bytes(
            req.width,
            req.height,
            derived::TABLE_CONDITIONING_TOKENS,
            references,
            negative,
            None,
        ),
        derived::prefix_kv_cache_demand_bytes(
            fixture.tier,
            req.width,
            req.height,
            references,
            negative,
        ),
    )
}

/// The worker's admitted `Resident` context for `req`, with an unlimited budget.
fn context(
    fixture: &Fixture,
    generator: &dyn mlx_gen::Generator,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::MemoryRunContext {
    let contract = generator.memory_strategy_contract().expect("a contract");
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision: mlx_gen::gen_core::Precision::Bf16,
            quant: fixture.quant(),
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

/// What the SceneWorks worker does for one request, with `total_bytes` of budget: price it
/// ([`demands`]), pick the mode ([`PrefixKvCacheMode::select`]), open the request scope through
/// the `Generator` trait with the demand of the mode it picked (the shed demand when neither
/// fits — that is the request it would have submitted before the cache existed), let the scope
/// configure the request, and only **then** write the selection onto it and render.
fn admit_and_render(
    fixture: &Fixture,
    generator: &dyn mlx_gen::Generator,
    req: &GenerationRequest,
    total_bytes: u64,
) -> Result<(PrefixKvCacheMode, GenerationRequest, Rendered), CoreError> {
    let (shed, cache) = demands(fixture, req);
    let budget = MemoryBudget {
        total_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let selected = PrefixKvCacheMode::select(budget, shed, cache);
    let mut context = context(fixture, generator, req);
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
    Ok((mode, req, rendered))
}

/// **Admission, end to end** — the story's second and third acceptance criteria, under both ids
/// and on a packed tier, with and without a negative branch, with a square reference and with a
/// 3:1 one (whose fitted grid is not the square's):
///
/// * a budget that fits the cached demand runs **cached**, within its price, under a memory
///   limit that includes the priced cache;
/// * a budget one byte short of it — above the prefill demand — is **admitted and runs prefill**
///   by selection, under the prefill memory limit, with zero cache bytes;
/// * a budget one byte short of the prefill demand is **refused**, with the shared safety check's
///   own refusal — the one this request got before the cache existed.
///
/// The path is read off the render's report and the allocator's limits off the live allocator
/// inside the denoise loop, so a decision that never reached the generator fails here. The
/// buffer-cache limit is the cache-shed transient on **both** paths (the caches are live arrays,
/// not pooled buffers); the memory limit is what separates them — and where this host's own MLX
/// memory limit is too low to separate them, the case says so on stderr instead of passing in
/// silence.
///
/// *Mutations that red this:* ignoring the cache term in the selection (the between-budget
/// request runs cached); ignoring a `Prefill` selection in the generator (it reports `cached`
/// under a prefill limit); ignoring a `Cached` selection (the roomy request reports `prefill`);
/// leaving the priced cache out of a cached render's memory limit; admitting the fits-neither
/// request.
#[test]
fn a_request_between_the_two_budgets_is_admitted_and_runs_prefill() {
    let tmp = tempfile::tempdir().unwrap();
    let (memory_limit_before, cache_limit_before) = AllocatorBounds::current();
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
    let [base, turbo] = dense(tmp.path());
    let (mut cases, mut separated) = (0, 0);
    for fixture in [base, turbo, packed(tmp.path(), BASE, Tier::Q4)] {
        let generator = fixture.load();
        let resident =
            derived::resident_weights_for(fixture.variant, fixture.tier).resident_total();
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
            cases += 1;
            let (shed, cache) = demands(&fixture, &req);
            assert!(cache > 0, "{tag}: the cache is never free");
            let memory_limit =
                |transient: u64| memory_limit_before.min((resident + transient) as usize);

            // 1. Room for the cache: cached, within its price.
            let (mode, admitted, roomy) =
                admit_and_render(&fixture, generator.as_ref(), &req, shed + cache)
                    .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Cached, "{tag}");
            assert!(roomy.ran.cache_bytes > 0, "{tag}");
            assert_within_price(&tag, &fixture, &admitted, &roomy);

            // 2. One byte short of the cache: still admitted, and it runs prefill.
            let (mode, _, tight) =
                admit_and_render(&fixture, generator.as_ref(), &req, shed + cache - 1)
                    .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(tight.ran, PREFILL_BY_SELECTION, "{tag}");
            let (mode, _, exact) = admit_and_render(&fixture, generator.as_ref(), &req, shed)
                .unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(mode, PrefixKvCacheMode::Prefill, "{tag}");
            assert_eq!(exact.pixels(), tight.pixels(), "{tag}");

            // The allocator bounds of the two paths. The buffer-cache limit is the cache-shed
            // transient either way; the memory limit carries the priced cache when cached.
            for (path, rendered) in [("cached", &roomy), ("prefill", &tight)] {
                assert_eq!(
                    rendered.limits_during.1,
                    cache_limit_before.min(shed as usize),
                    "{tag}: the {path} render's buffer-cache limit is the cache-shed transient"
                );
            }
            assert_eq!(
                roomy.limits_during.0,
                memory_limit(shed + cache),
                "{tag}: a cached render's memory limit includes its priced cache"
            );
            assert_eq!(
                tight.limits_during.0,
                memory_limit(shed),
                "{tag}: a prefill render's memory limit does not"
            );
            if memory_limit(shed + cache) > memory_limit(shed) {
                separated += 1;
                assert!(
                    roomy.limits_during.0 > tight.limits_during.0,
                    "{tag}: the two paths run under different memory limits"
                );
            } else {
                eprintln!(
                    "{tag}: NOT SEPARATED — this host's MLX memory limit ({memory_limit_before} \
                     B) is at or below resident + shed ({} B), so the cached and prefill memory \
                     limits both clamp to it and this case cannot tell them apart",
                    resident + shed
                );
            }

            // 3. One byte short of the prefill demand: refused, as before the cache existed.
            let refusal = admit_and_render(&fixture, generator.as_ref(), &req, shed - 1)
                .err()
                .unwrap_or_else(|| panic!("{tag}: a request that fits neither must be refused"));
            assert!(
                matches!(&refusal, CoreError::Unsupported(reason)
                if *reason == format!(
                    "{}: incremental live demand {shed} exceeds effective budget {}",
                    fixture.id,
                    shed - 1
                )),
                "{tag}: {refusal:?}"
            );

            if let Some(Conditioning::Reference { image, .. }) = req.conditioning.first() {
                // The cache is over the PREFIX: the text rows plus the reference's own latent
                // grid at the snapshot's fitted resolution — not the 32×32 target's four tokens.
                let shape = shape(&fixture.root);
                let per_token = shape.kv_bytes(1);
                let vision = mlx_gen_qwen_image_2_1::load_vision_config(&fixture.root)
                    .unwrap()
                    .expect("the miniature snapshot carries a vision tower");
                let (fit_w, fit_h) = mlx_gen_qwen_image_2_1::reference::reference_fit(
                    (image.width, image.height),
                    0,
                    &vision,
                )
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
        AllocatorBounds::current(),
        (memory_limit_before, cache_limit_before),
        "every request's bounds were restored"
    );
    eprintln!(
        "cached and prefill memory limits were separated in {separated} of {cases} cases on this \
         host"
    );
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
    let tmp = tempfile::tempdir().unwrap();
    for fixture in dense(tmp.path()) {
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
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("non-causal-snapshot");
    mlx_gen::quant::copy_dir(&tiny_snapshot(), &root).expect("copy the tiny snapshot");
    let config = root.join("transformer/config.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    assert_eq!(value["causal_condition"], serde_json::json!(true));
    value["causal_condition"] = serde_json::json!(false);
    std::fs::write(&config, serde_json::to_string_pretty(&value).unwrap()).unwrap();

    let fixture = Fixture {
        id: BASE,
        variant: Variant::Base,
        tier: Tier::Bf16,
        root,
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
