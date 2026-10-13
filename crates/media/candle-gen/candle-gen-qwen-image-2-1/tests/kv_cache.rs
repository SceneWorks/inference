//! The prefix KV cache (sc-25904): upstream's `QwenImage21KVCache` / `use_kv_cache=True` path, the
//! candle twin of `mlx-gen-qwen-image-2-1`'s `tests/kv_cache.rs`.
//!
//! Three questions, each asked of the tensors the code under test actually runs on:
//!
//! 1. **Is the cached path the prefill path?** At every model evaluation of every case — text only,
//!    one, two and three references, with and without a true-CFG negative branch — the velocity the
//!    sampler consumes with the cache on equals the full-sequence (prefill) velocity for the same
//!    latents and timestep. Both sides are this port on the same device, so the bound is an
//!    absolute + relative f32 bound, not a peak-relative parity tolerance.
//! 2. **Is it upstream's cached path?** The same evaluations against the fixture the MLX twin reads
//!    (`mlx-gen-qwen-image-2-1/tests/fixtures/qwen21_kv_cache.safetensors`, written by
//!    `mlx-gen/tools/dump_qwen21_kv_cache.py`), which records what the frozen
//!    `QwenImage21Pipeline(use_kv_cache=True)` fed its transformer and got back, plus the K/V
//!    `QwenImage21KVLayerCache` held — and the `use_kv_cache=False` latents, the independent oracle
//!    for the off path.
//! 3. **Does the loop use it correctly?** Extract on the first model *evaluation*, cached on every
//!    later one (also under a solver that evaluates twice per step), one independent cache per
//!    prompt branch, target tokens only through the blocks, nothing cached when the switch is off
//!    or the model is not `causal_condition`, nothing decoded after a cancellation, and nothing of
//!    a cache left alive once the loop returns.
//!
//! Everything here runs on candle's **CPU** backend in f32. The released checkpoint's bf16 runs
//! only on a GPU backend (candle's CPU backend has no bf16 matmul), which no test in this file
//! reaches.
//!
//! **Memory.** candle's CPU backend has no peak-memory probe, so this binary installs a counting
//! global allocator ([`heap`]): CPU tensor storage is plain heap memory allocated on the calling
//! thread, and the counters are thread-local, so a measurement sees its own test's tensors and
//! nothing of the tests running beside it. Every bound is derived from tensor sizes.

use std::collections::HashMap;

use candle_core::{DType, Storage, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen::run_flow_sampler;
use candle_gen_qwen_image_2_1::{
    denoise_reported, joint_images, load_scheduler_config, load_transformer, scheduler, text_rows,
    DenoiseInputs, DenoiseReport, EvaluationPath, JointLayout, KvCacheMode, KvCachePath,
    ModelEvaluation, PrefixKvCache, PromptBranch, QwenImage21Transformer, ReferenceConditioning,
    Segment, TransformerConfig,
};

use crate::common::{assert_close, device, host_f32, tiny_snapshot, Fixture};

/// A counting global allocator with **thread-local** counters: bytes live and peak bytes live on
/// the calling thread while a [`heap::measure`] is open. Outside a measurement it only forwards
/// to the system allocator.
mod heap {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ON: Cell<bool> = const { Cell::new(false) };
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    fn account(delta: isize) {
        // `try_with`: an allocation during thread teardown must not panic inside the allocator.
        let _ = ON.try_with(|on| {
            if on.get() {
                let _ = LIVE.try_with(|live| {
                    let now = live.get() + delta;
                    live.set(now);
                    let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
                });
            }
        });
    }

    struct Counting;

    // SAFETY: every method forwards to `System` unchanged; the accounting touches only
    // const-initialised, destructor-free thread-locals and never allocates.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = System.alloc(layout);
            if !p.is_null() {
                account(layout.size() as isize);
            }
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let p = System.alloc_zeroed(layout);
            if !p.is_null() {
                account(layout.size() as isize);
            }
            p
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            account(-(layout.size() as isize));
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let p = System.realloc(ptr, layout, new_size);
            if !p.is_null() {
                account(new_size as isize - layout.size() as isize);
            }
            p
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    /// Heap bytes of one closure, on the calling thread, relative to its entry.
    pub struct Measured {
        /// The most that was live at once while it ran.
        pub peak: usize,
        /// What was still live when it returned (whatever it handed back or leaked).
        pub held: usize,
    }

    /// Run `f` and report its peak and retained heap bytes on this thread.
    pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Measured) {
        LIVE.with(|live| live.set(0));
        PEAK.with(|peak| peak.set(0));
        ON.with(|on| on.set(true));
        let out = f();
        ON.with(|on| on.set(false));
        let measured = Measured {
            peak: PEAK.with(Cell::get).max(0) as usize,
            held: LIVE.with(Cell::get).max(0) as usize,
        };
        (out, measured)
    }
}

/// Cached vs prefill, same port, same device, f32: `|Δ| ≤ ATOL + RTOL·|want|` per element.
///
/// Measured on candle CPU (arm64) over all 31 comparisons this file makes (every case, branch and
/// evaluation of the fixture, the whole denoise under euler / heun / dpmpp_sde, the synthetic
/// geometries): worst `max|Δ|` 4.1e-6 at a peak of 3.9 (1.0e-6 × peak); the worst element sits at
/// 0.28× this bound. The two paths are not bit-identical, and are not meant to be: a cached step's
/// target tokens read the sampled-timestep modulation row exactly (as upstream's `where` does),
/// while the prefill selects it through a blend that rounds in the last place (see UPSTREAM.md,
/// *Deliberate divergences*). The mutations this file names move the velocity by 1e-2 … 1e0.
const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-5;

/// Port (candle CPU, f32) vs upstream (torch CPU, f32), `max|Δ| ≤ tol · max(1, peak)`:
///
/// * the cached K/V themselves — [`KV_TOL`]; a `k` cached before RoPE is off by O(1).
/// * velocities and the final latents — [`UPSTREAM_TOL`], the bar `pipeline_parity.rs` and
///   `edit_parity.rs` already hold this crate's denoise to against the same snapshot (2e-4 × peak).
///
/// Both carry the measured numbers in the test output (`assert_close` prints them).
const KV_TOL: f32 = 1e-5;
const UPSTREAM_TOL: f32 = 2e-4;

const CASES: [&str; 6] = ["t2i", "t2i_cfg", "ref1", "ref2", "ref2_cfg", "aspect3"];

fn kv_fixture() -> Fixture {
    Fixture::open("qwen21_kv_cache.safetensors")
}

/// One prompt branch of a fixture case, as the transformer consumes it.
struct Branch {
    name: &'static str,
    text: Tensor,
    layout: JointLayout,
    prefix_tokens: usize,
}

struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    steps: usize,
    true_cfg: f32,
    blocks: usize,
    reference_latents: Vec<Tensor>,
    branches: Vec<Branch>,
}

impl Case {
    fn target_tokens(&self) -> usize {
        self.branches[0].layout.target_tokens()
    }

    fn latents(&self, w: &Fixture, eval: usize) -> Tensor {
        w.tensor(&format!("{}/eval_{eval}/latents", self.name))
    }

    fn sigma(&self, w: &Fixture, eval: usize) -> f32 {
        host_f32(&w.tensor(&format!("{}/eval_{eval}/timestep", self.name)))[0]
    }
}

fn load_case(w: &Fixture, name: &'static str) -> Case {
    let v: serde_json::Value = serde_json::from_str(w.meta(name)).expect("case metadata is JSON");
    let references = v["references"].as_u64().expect("references") as usize;
    let branches = ["pos", "neg"]
        .into_iter()
        .filter(|b| !v["branches"][b].is_null())
        .map(|branch| {
            let meta = &v["branches"][branch];
            let segments = meta["segments"]
                .as_array()
                .expect("segments")
                .iter()
                .map(|s| match s[0].as_str().expect("segment kind") {
                    "text" => Segment::Text {
                        len: s[1].as_u64().unwrap() as usize,
                    },
                    "image" => Segment::Image {
                        height: s[1].as_u64().unwrap() as usize,
                        width: s[2].as_u64().unwrap() as usize,
                    },
                    other => panic!("unknown segment kind {other}"),
                })
                .collect();
            let hidden = w.tensor(&format!("{name}/{branch}/encoder_hidden_states"));
            // The transformer's `img_mask` spans the joint sequence (target slots appended); the
            // text rows are selected by its leading VLM-length part.
            let mask: Vec<bool> = host_f32(&w.tensor(&format!("{name}/{branch}/img_mask")))
                [..hidden.dim(1).unwrap()]
                .iter()
                .map(|m| *m != 0.0)
                .collect();
            Branch {
                name: branch,
                text: text_rows(&hidden, &mask).unwrap(),
                layout: JointLayout { segments },
                prefix_tokens: meta["prefix_tokens"].as_u64().unwrap() as usize,
            }
        })
        .collect();
    Case {
        name,
        width: v["width"].as_u64().unwrap() as u32,
        height: v["height"].as_u64().unwrap() as u32,
        steps: v["steps"].as_u64().unwrap() as usize,
        true_cfg: v["true_cfg_scale"].as_f64().unwrap() as f32,
        blocks: v["blocks"].as_u64().unwrap() as usize,
        reference_latents: (0..references)
            .map(|i| w.tensor(&format!("{name}/ref_latents_{i}")))
            .collect(),
        branches,
    }
}

/// `(max |Δ|, max |want|, worst |Δ| / (atol + rtol·|want|))`.
fn mixed_error(got: &Tensor, want: &Tensor, atol: f32, rtol: f32) -> (f32, f32, f32) {
    assert_eq!(got.dims(), want.dims(), "shape mismatch");
    let (g, w) = (host_f32(got), host_f32(want));
    let (mut max_abs, mut peak, mut worst) = (0f32, 0f32, 0f32);
    for (a, b) in g.iter().zip(&w) {
        let d = (a - b).abs();
        assert!(d.is_finite(), "non-finite difference ({a} vs {b})");
        max_abs = max_abs.max(d);
        peak = peak.max(b.abs());
        worst = worst.max(d / (atol + rtol * b.abs()));
    }
    (max_abs, peak, worst)
}

/// Assert `|got − want| ≤ atol + rtol·|want|` for every element, printing the measured numbers.
fn assert_agrees(name: &str, got: &Tensor, want: &Tensor, atol: f32, rtol: f32) {
    let (max_abs, peak, worst) = mixed_error(got, want, atol, rtol);
    eprintln!("{name}: max|Δ|={max_abs:.3e} peak={peak:.3e} worst/bound={worst:.3e}");
    assert!(
        peak > 0.0,
        "{name}: the reference is all zeros — nothing was compared"
    );
    assert!(
        worst <= 1.0,
        "{name}: max|Δ|={max_abs:.3e} at peak {peak:.3e} is {worst:.2}× the bound \
         (atol {atol:.0e} + rtol {rtol:.0e}·|want|)"
    );
}

fn bits(t: &Tensor) -> Vec<u32> {
    host_f32(t).iter().map(|v| v.to_bits()).collect()
}

fn prefill(
    model: &QwenImage21Transformer,
    case: &Case,
    branch: &Branch,
    x: &Tensor,
    t: f32,
) -> Tensor {
    model
        .forward_joint(
            &branch.text,
            &joint_images(&case.reference_latents, x),
            t,
            &branch.layout,
        )
        .unwrap()
}

fn extract(
    model: &QwenImage21Transformer,
    case: &Case,
    branch: &Branch,
    x: &Tensor,
    t: f32,
) -> (Tensor, PrefixKvCache) {
    model
        .forward_joint_extract(
            &branch.text,
            &joint_images(&case.reference_latents, x),
            t,
            &branch.layout,
        )
        .unwrap()
}

/// Question 1 at the transformer: for every case, branch and evaluation the fixture records, the
/// extract evaluation and every cached evaluation return the prefill velocity for the same inputs;
/// and the cached forward's blocks run on the target tokens only.
///
/// *Mutations that red this:* storing the cached `k` **before** RoPE; skipping the cache concat in
/// `Attention::forward_cached` (target attends to itself only); reading the `t = 0` modulation row
/// for the cached target tokens; the wrong RoPE rows for the target; the cached arm calling the
/// prefill forward (the block stages are then joint-length).
#[test]
fn cached_velocity_is_the_prefill_velocity_at_every_evaluation() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    let inner = model.config().inner_dim();
    for name in CASES {
        let case = load_case(&w, name);
        assert!(case.steps >= 3, "{name}: two cached evaluations are needed");
        for branch in &case.branches {
            let tag = format!("{name}/{}", branch.name);
            let (x0, t0) = (case.latents(&w, 0), case.sigma(&w, 0));
            let (velocity, cache) = extract(&model, &case, branch, &x0, t0);
            // The extract IS the prefill forward plus copies: bit for bit.
            assert_eq!(
                bits(&velocity),
                bits(&prefill(&model, &case, branch, &x0, t0)),
                "{tag}/eval_0: the extract evaluation must be the prefill forward"
            );
            assert_eq!(cache.prefix_len(), branch.layout.prefix_len(), "{tag}");
            assert_eq!(cache.prefix_len(), branch.prefix_tokens, "{tag}");
            assert_eq!(cache.target_tokens(), case.target_tokens(), "{tag}");
            assert_eq!(cache.num_blocks(), model.num_blocks(), "{tag}");
            assert_eq!(
                cache.nbytes(),
                2 * model.num_blocks() * branch.prefix_tokens * inner * 4,
                "{tag}: the cache holds exactly the prefix K/V"
            );
            for eval in 1..case.steps {
                let (x, t) = (case.latents(&w, eval), case.sigma(&w, eval));
                assert!(t != t0, "{tag}: a later evaluation is at another timestep");
                let (got, trace) = model.forward_joint_cached_traced(&x, t, &cache).unwrap();
                assert_agrees(
                    &format!("{tag}/eval_{eval} cached vs prefill"),
                    &got,
                    &prefill(&model, &case, branch, &x, t),
                    ATOL,
                    RTOL,
                );
                // What the blocks actually ran on: every block stage is target-length.
                let block_stages: Vec<_> = trace
                    .iter()
                    .filter(|(stage, _)| stage.starts_with("block_"))
                    .collect();
                assert_eq!(block_stages.len(), 3 * model.num_blocks(), "{tag}");
                for (stage, value) in block_stages {
                    assert_eq!(
                        value.dim(1).unwrap(),
                        case.target_tokens(),
                        "{tag}/eval_{eval}/{stage}: the blocks must see target tokens only \
                         (joint sequence is {})",
                        branch.layout.total_len()
                    );
                }
            }
        }
    }
}

/// Question 2 at the transformer: the port's extract velocity, the K/V it caches for every block,
/// and every cached-step velocity equal what upstream's `kv_cache_mode="extract"` / `"cached"`
/// calls produced.
///
/// The cache is heads-first (`[1, heads, prefix, head_dim]`); upstream's is
/// `[1, prefix, heads, head_dim]`, so each entry is transposed for the comparison and nothing else.
///
/// *Mutations that red this:* caching `k` before RoPE (the fixture's `k` is post-RoPE, as
/// `QwenImage21KVLayerCache` documents); caching `v` after the norm; caching the wrong rows (e.g.
/// the target's); any cached-step defect listed on the test above.
#[test]
fn extract_cache_and_cached_velocity_match_upstream() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    let cfg = model.config().clone();
    for name in CASES {
        let case = load_case(&w, name);
        assert_eq!(case.blocks, model.num_blocks(), "{name}: block count");
        for branch in &case.branches {
            let tag = format!("{name}/{}", branch.name);
            let (velocity, cache) = extract(
                &model,
                &case,
                branch,
                &case.latents(&w, 0),
                case.sigma(&w, 0),
            );
            assert_close(
                &format!("{tag}/eval_0 velocity"),
                &velocity,
                &w.tensor(&format!("{name}/eval_0/{}/velocity", branch.name)),
                UPSTREAM_TOL,
            );
            for block in 0..case.blocks {
                let (k, v) = cache.layer(block).unwrap();
                assert_eq!(
                    k.dims(),
                    &[
                        1,
                        cfg.num_attention_heads,
                        branch.prefix_tokens,
                        cfg.attention_head_dim
                    ],
                    "{tag}: cached k layout"
                );
                assert_eq!(k.dtype(), model.compute_dtype(), "{tag}: cache dtype");
                for (kind, got) in [("k", k), ("v", v)] {
                    assert_close(
                        &format!("{tag}/{kind}_{block}"),
                        &got.transpose(1, 2).unwrap(),
                        &w.tensor(&format!("{name}/{}/{kind}_{block}", branch.name)),
                        KV_TOL,
                    );
                }
            }
            for eval in 1..case.steps {
                let got = model
                    .forward_joint_cached(&case.latents(&w, eval), case.sigma(&w, eval), &cache)
                    .unwrap();
                assert_close(
                    &format!("{tag}/eval_{eval} velocity"),
                    &got,
                    &w.tensor(&format!("{name}/eval_{eval}/{}/velocity", branch.name)),
                    UPSTREAM_TOL,
                );
            }
        }
    }
}

/// Everything one `denoise_reported` call needs for a fixture case.
struct Run<'a> {
    model: &'a QwenImage21Transformer,
    case: &'a Case,
    sigmas: &'a [f32],
    init: &'a Tensor,
    sampler: Option<&'a str>,
    kv_cache: KvCacheMode,
}

impl Run<'_> {
    fn go(
        &self,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> (candle_gen::Result<Tensor>, DenoiseReport) {
        let case = self.case;
        let pos = &case.branches[0];
        let neg = case.branches.get(1).filter(|_| case.true_cfg > 1.0);
        let mut report = DenoiseReport::default();
        let out = denoise_reported(
            DenoiseInputs {
                transformer: self.model,
                sigmas: self.sigmas,
                latents: self.init.clone(),
                prompt_embeds: &pos.text,
                negative_embeds: neg.map(|n| &n.text),
                true_cfg_scale: case.true_cfg,
                width: case.width,
                height: case.height,
                sampler: self.sampler,
                seed: 0,
                cancel,
                references: (!case.reference_latents.is_empty()).then(|| ReferenceConditioning {
                    latents: &case.reference_latents,
                    layout: &pos.layout,
                    negative_layout: neg.map(|n| &n.layout),
                }),
                kv_cache: self.kv_cache,
            },
            &mut report,
            on_progress,
        );
        (out, report)
    }

    fn ok(&self) -> (Tensor, DenoiseReport) {
        let (out, report) = self.go(&CancelFlag::new(), &mut |_| {});
        (out.unwrap(), report)
    }
}

fn case_sigmas(case: &Case) -> Vec<f32> {
    let scheduler_config = load_scheduler_config(&tiny_snapshot()).unwrap();
    scheduler::sigmas_for_image(&scheduler_config, case.steps, case.width, case.height).unwrap()
}

/// The per-branch evaluation sequence of a report.
fn by_branch(report: &DenoiseReport) -> HashMap<&'static str, Vec<ModelEvaluation>> {
    let mut out: HashMap<&'static str, Vec<ModelEvaluation>> = HashMap::new();
    for evaluation in &report.evaluations {
        let key = match evaluation.branch {
            PromptBranch::Positive => "pos",
            PromptBranch::Negative => "neg",
        };
        out.entry(key).or_default().push(*evaluation);
    }
    out
}

/// Assert the cached loop's shape: per branch, the first model evaluation extracted over the whole
/// joint sequence and every later one decoded over the target tokens only.
fn assert_cached_shape(
    tag: &str,
    model: &QwenImage21Transformer,
    case: &Case,
    report: &DenoiseReport,
    evaluations: usize,
) {
    assert_eq!(report.kv_cache, KvCachePath::Cached, "{tag}");
    let per_branch = by_branch(report);
    assert_eq!(per_branch.len(), case.branches.len(), "{tag}: branches run");
    let inner = model.config().inner_dim();
    let mut bytes = 0;
    for branch in &case.branches {
        let seen = &per_branch[branch.name];
        assert_eq!(
            seen.len(),
            evaluations,
            "{tag}/{}: evaluations",
            branch.name
        );
        assert_eq!(
            (seen[0].path, seen[0].block_tokens),
            (EvaluationPath::Extract, branch.layout.total_len()),
            "{tag}/{}: the first model evaluation prefills over the joint sequence",
            branch.name
        );
        for (i, later) in seen.iter().enumerate().skip(1) {
            assert_eq!(
                (later.path, later.block_tokens),
                (EvaluationPath::Cached, case.target_tokens()),
                "{tag}/{}: evaluation {i} must decode target tokens only",
                branch.name
            );
        }
        bytes += 2 * case.blocks * branch.prefix_tokens * inner * 4;
    }
    assert_eq!(report.kv_cache_bytes, bytes, "{tag}: one cache per branch");
}

/// The whole loop, default sampler, every case: the cache-on latents equal the cache-off latents,
/// and the report shows extract-then-cached per branch with target-only block inputs.
///
/// *Mutations that red this:* a cached step silently recomputing the full sequence (the
/// `block_tokens` assertion: 16 expected, the joint length observed); sharing one cache across the
/// two CFG branches (the `*_cfg` cases diverge by ~1e0 and `kv_cache_bytes` halves); ignoring the
/// switch (`Off` reporting a cache, or `On` reporting none).
#[test]
fn cached_denoise_matches_prefill_denoise_and_reports_its_path() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.tensor(&format!("{name}/latents_init"));
        let run = |kv_cache| Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init: &init,
            sampler: None,
            kv_cache,
        };
        let (on, on_report) = run(KvCacheMode::On).ok();
        let (off, off_report) = run(KvCacheMode::Off).ok();
        assert_agrees(&format!("{name}: cache on vs off"), &on, &off, ATOL, RTOL);
        assert_cached_shape(name, &model, &case, &on_report, case.steps);

        assert_eq!(off_report.kv_cache, KvCachePath::Prefill, "{name}");
        assert_eq!(off_report.kv_cache_bytes, 0, "{name}");
        assert_eq!(
            off_report.evaluations.len(),
            case.steps * case.branches.len(),
            "{name}"
        );
        for evaluation in &off_report.evaluations {
            let branch = match evaluation.branch {
                PromptBranch::Positive => &case.branches[0],
                PromptBranch::Negative => &case.branches[1],
            };
            assert_eq!(
                (evaluation.path, evaluation.block_tokens),
                (EvaluationPath::Prefill, branch.layout.total_len()),
                "{name}: the off path never touches a cache"
            );
        }
    }
}

/// Question 2 for the whole loop: three Euler steps with the cache on land on upstream's
/// `use_kv_cache=True` latents — and with it off on upstream's `use_kv_cache=False` latents for the
/// same case (`latents_final_prefill`). The second is the independent oracle for the off path: the
/// prefill forward this story refactored, against something that is not this port's own code.
///
/// *Mutation that reds the off-path row:* a numeric change to the prefill loop or forward — e.g.
/// the true-CFG combine applied with the wrong scale, or `Attention::project` normalising `k` with
/// `norm_q`.
#[test]
fn cached_denoise_matches_upstream_end_to_end() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.tensor(&format!("{name}/latents_init"));
        let run = |kv_cache| {
            Run {
                model: &model,
                case: &case,
                sigmas: &sigmas,
                init: &init,
                sampler: None,
                kv_cache,
            }
            .ok()
        };
        let (off, off_report) = run(KvCacheMode::Off);
        assert_eq!(off_report.kv_cache, KvCachePath::Prefill, "{name}");
        assert_close(
            &format!("{name}/latents_final_prefill (cache off)"),
            &off,
            &w.tensor(&format!("{name}/latents_final_prefill")),
            UPSTREAM_TOL,
        );
        let (got, report) = run(KvCacheMode::On);
        assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
        assert_close(
            &format!("{name}/latents_final"),
            &got,
            &w.tensor(&format!("{name}/latents_final")),
            UPSTREAM_TOL,
        );
    }
}

/// The rule is per model **evaluation**, not per sampler step. Heun evaluates the model twice on
/// every step but the last, the second time at the *next* sigma; DPM++ SDE evaluates at an
/// off-schedule midpoint. Either way: one extract per branch, everything else cached, and the
/// result equals the prefill loop under the same solver.
///
/// *Mutation that reds this:* any per-step or per-sigma lifetime for the cache (e.g. dropping it
/// after a decode so the next evaluation prefills again) — the report then shows more than one
/// `Extract` for the branch.
#[test]
fn a_multi_evaluation_sampler_extracts_once_and_decodes_every_other_evaluation() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for (name, sampler) in [("t2i", "heun"), ("ref2_cfg", "heun"), ("ref1", "dpmpp_sde")] {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.tensor(&format!("{name}/latents_init"));
        let run = |kv_cache| Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init: &init,
            sampler: Some(sampler),
            kv_cache,
        };
        let (on, on_report) = run(KvCacheMode::On).ok();
        let (off, off_report) = run(KvCacheMode::Off).ok();
        let evaluations = off_report.evaluations.len() / case.branches.len();
        assert!(
            evaluations > case.steps,
            "{name}/{sampler}: {evaluations} evaluations over {} steps is not a multi-evaluation \
             solver — the test would not ask its question",
            case.steps
        );
        assert_cached_shape(
            &format!("{name}/{sampler}"),
            &model,
            &case,
            &on_report,
            evaluations,
        );
        assert_agrees(
            &format!("{name}/{sampler}: cache on vs off"),
            &on,
            &off,
            ATOL,
            RTOL,
        );
    }
}

/// True CFG holds two independent caches. The prompts differ in length, so the two prefixes do too
/// — and decoding the negative branch against the **positive** branch's cache is far from the
/// negative branch's own velocity, which is what makes the on-vs-off agreement of the `*_cfg` cases
/// a test of isolation rather than of two interchangeable caches.
///
/// *Mutation that reds this (and the `*_cfg` rows of the tests above):* the negative branch
/// reading the positive branch's cache.
#[test]
fn the_negative_branch_has_its_own_cache() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for name in ["t2i_cfg", "ref2_cfg"] {
        let case = load_case(&w, name);
        let (pos, neg) = (&case.branches[0], &case.branches[1]);
        assert_ne!(
            pos.prefix_tokens, neg.prefix_tokens,
            "{name}: prefixes differ"
        );
        let (x0, t0) = (case.latents(&w, 0), case.sigma(&w, 0));
        let (_, pos_cache) = extract(&model, &case, pos, &x0, t0);
        let (_, neg_cache) = extract(&model, &case, neg, &x0, t0);
        assert_eq!(pos_cache.prefix_len(), pos.prefix_tokens);
        assert_eq!(neg_cache.prefix_len(), neg.prefix_tokens);

        let (x, t) = (case.latents(&w, 1), case.sigma(&w, 1));
        let want = prefill(&model, &case, neg, &x, t);
        assert_agrees(
            &format!("{name}: negative branch, own cache"),
            &model.forward_joint_cached(&x, t, &neg_cache).unwrap(),
            &want,
            ATOL,
            RTOL,
        );
        let crossed = model.forward_joint_cached(&x, t, &pos_cache).unwrap();
        let (_, _, worst) = mixed_error(&crossed, &want, ATOL, RTOL);
        eprintln!("{name}: negative branch on the positive cache is {worst:.3e}× the bound");
        assert!(
            worst > 100.0,
            "{name}: the positive branch's cache reproduces the negative branch to {worst:.2}× \
             the bound — the two branches are not distinguishable here, so nothing above tests \
             their isolation"
        );
    }
}

/// With the switch off the loop is the plain prefill loop: the latents are bit-for-bit what the
/// pre-story closure (`forward_joint` / `forward` per branch, then the true-CFG combine, under the
/// shared sampler) produces. This pins the **loop** — no extra evaluation boundary, no cache, the
/// same combine — against this port's own forward; it cannot see a numeric change inside that
/// forward. The independent oracle for the off path is upstream's `use_kv_cache=False` output,
/// `{case}/latents_final_prefill`, gated in `cached_denoise_matches_upstream_end_to_end`.
///
/// *Mutations that red this:* any numeric change to the off path's loop — e.g. the true-CFG
/// combine rewritten as `pos + (s − 1)·(pos − neg)` — and a switch that is ignored (always on):
/// the cached path's latents differ from the prefill's in the last bits.
#[test]
fn cache_off_is_bit_identical_to_the_prefill_loop() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.tensor(&format!("{name}/latents_init"));
        let run = |kv_cache| {
            Run {
                model: &model,
                case: &case,
                sigmas: &sigmas,
                init: &init,
                sampler: None,
                kv_cache,
            }
            .ok()
            .0
        };
        let got = run(KvCacheMode::Off);

        let pos = &case.branches[0];
        let neg = case.branches.get(1).filter(|_| case.true_cfg > 1.0);
        let conditioned = !case.reference_latents.is_empty();
        let (h, w_tokens) = match pos.layout.segments.last() {
            Some(Segment::Image { height, width }) => (*height, *width),
            other => panic!("{name}: no target block ({other:?})"),
        };
        let branch_velocity = |branch: &Branch, latents: &Tensor, sigma: f32| -> Tensor {
            if conditioned {
                prefill(&model, &case, branch, latents, sigma)
            } else {
                model
                    .forward(latents, &branch.text, sigma, h, w_tokens)
                    .unwrap()
            }
        };
        let cancel = CancelFlag::new();
        let want = run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            0,
            &cancel,
            &mut |_| {},
            None,
            |latents: &Tensor, sigma: f32| -> candle_gen::Result<Tensor> {
                let velocity = branch_velocity(pos, latents, sigma);
                match neg {
                    Some(neg) => {
                        let uncond = branch_velocity(neg, latents, sigma);
                        let diff = ((&velocity - &uncond)? * case.true_cfg as f64)?;
                        Ok((&uncond + diff)?)
                    }
                    None => Ok(velocity),
                }
            },
        )
        .unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "{name}: cache off must be the prefill loop, bit for bit"
        );
    }
}

/// The shared sampler gates every model evaluation on the cancel flag *before* announcing its
/// step, so a cancellation requested while step 1 is announced lets that step's evaluation — the
/// one that prefills — finish and stops the loop at the next gate: no evaluation decodes
/// afterwards, and what did run is still in the report. A request cancelled before it starts runs
/// no evaluation and builds no cache at all.
///
/// *Mutation that reds this:* a report only written back when the loop completes (a cancelled
/// run would then report nothing), or a loop that keeps evaluating after the cancellation (the
/// report would carry `Cached` entries).
#[test]
fn cancelling_after_the_prefill_step_runs_no_cached_evaluation() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    for name in ["ref1", "ref2_cfg"] {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.tensor(&format!("{name}/latents_init"));
        let run = Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init: &init,
            sampler: None,
            kv_cache: KvCacheMode::On,
        };
        // `Some(1)`: cancel while step 1 is announced. `None`: cancelled before the loop starts.
        for (cancel_at, steps_run) in [(Some(1u32), 1usize), (None, 0)] {
            let cancel = CancelFlag::new();
            if cancel_at.is_none() {
                cancel.cancel();
            }
            let mut announced = Vec::new();
            let (out, report) = run.go(&cancel, &mut |p| {
                if let Progress::Step { current, .. } = p {
                    announced.push(current);
                    if Some(current) == cancel_at {
                        cancel.cancel();
                    }
                }
            });
            assert!(
                matches!(out, Err(candle_gen::CandleError::Canceled)),
                "{name}: cancelling at {cancel_at:?} must end the denoise as Canceled"
            );
            assert_eq!(
                announced,
                (1..=steps_run as u32).collect::<Vec<_>>(),
                "{name}"
            );
            assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
            assert_eq!(
                report
                    .evaluations
                    .iter()
                    .map(|e| e.path)
                    .collect::<Vec<_>>(),
                vec![EvaluationPath::Extract; steps_run * case.branches.len()],
                "{name}: cancelled at {cancel_at:?}"
            );
            assert_eq!(
                report.kv_cache_bytes > 0,
                steps_run > 0,
                "{name}: a cache exists exactly when an extract completed"
            );
        }
        // The cancelled runs left nothing behind: a fresh run prefills again and completes.
        let (_, report) = run.ok();
        assert_cached_shape(name, &model, &case, &report, case.steps);
    }
}

/// The caches live exactly as long as the denoise loop: once `denoise_reported` has returned —
/// completed or cancelled right after the prefill step — the thread holds no more heap than the
/// same run with the cache off (the final latents, the report's entries), although the run itself
/// peaked at least a cache higher. So nothing of a cache can reach the VAE decode.
///
/// *Mutation that reds this:* anything that lets a cache outlive the loop — e.g. the report
/// keeping the `PrefixKvCache` it accounts for, or the branches living in a caller-owned slot.
#[test]
fn the_caches_are_dropped_when_the_loop_returns() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    let case = load_case(&w, "ref2_cfg");
    let sigmas = case_sigmas(&case);
    let init = w.tensor("ref2_cfg/latents_init");
    let run = |kv_cache| Run {
        model: &model,
        case: &case,
        sigmas: &sigmas,
        init: &init,
        sampler: None,
        kv_cache,
    };
    // The first forward on a thread allocates the GEMM's thread-local workspace, which stays for
    // the thread's life; run once unmeasured so no measurement below is charged for it.
    run(KvCacheMode::On).ok();
    let ((_, on_report), on) = heap::measure(|| run(KvCacheMode::On).ok());
    let (_, off) = heap::measure(|| run(KvCacheMode::Off).ok());
    let cache_bytes = on_report.kv_cache_bytes;
    assert!(cache_bytes > 0, "the run held caches");
    // A returned run keeps its latents and its report: a few hundred bytes, the same either way
    // up to the report's `Vec` growth. One cached K tensor of one block is the smallest thing a
    // leaked cache could be.
    let one_tensor = cache_bytes / (4 * case.blocks);
    assert!(one_tensor >= 4096, "the cached tensors must dwarf a report");
    eprintln!(
        "completed: on held {} peak {} · off held {} peak {} · caches {cache_bytes} · one cached \
         tensor ≥ {one_tensor}",
        on.held, on.peak, off.held, off.peak
    );
    assert!(
        on.peak >= cache_bytes,
        "the cached run peaked at {} bytes, below the {cache_bytes} its caches hold — the probe \
         is not seeing the tensors",
        on.peak
    );
    assert!(
        on.held < off.held + one_tensor,
        "after the loop returned the cached run still holds {} bytes against the prefill run's {} \
         — a cache ({cache_bytes} bytes) outlived the loop",
        on.held,
        off.held
    );

    // Cancelled right after the step that prefills: both branches' caches existed, and are gone.
    let cancel = CancelFlag::new();
    let (report, cancelled) = heap::measure(|| {
        let (out, report) = run(KvCacheMode::On).go(&cancel, &mut |p| {
            if matches!(p, Progress::Step { current: 1, .. }) {
                cancel.cancel();
            }
        });
        assert!(matches!(out, Err(candle_gen::CandleError::Canceled)));
        report
    });
    assert_eq!(report.kv_cache_bytes, cache_bytes, "both caches were built");
    eprintln!("cancelled: held {} peak {}", cancelled.held, cancelled.peak);
    assert!(cancelled.peak >= cache_bytes);
    assert!(
        cancelled.held < one_tensor,
        "a cancelled run still holds {} bytes — a cache ({cache_bytes} bytes) outlived the loop",
        cancelled.held
    );
}

/// The same tiny weights as a model that is **not** `causal_condition`.
fn non_causal_model() -> QwenImage21Transformer {
    let root = tiny_snapshot();
    let cfg = TransformerConfig {
        causal_condition: false,
        ..TransformerConfig::from_json_file(&root.join("transformer").join("config.json")).unwrap()
    };
    let vb =
        candle_gen::loader::component_vb(&root, "transformer", DType::F32, &device(), "kv_cache")
            .unwrap();
    QwenImage21Transformer::new(&cfg, vb).unwrap()
}

/// The cache is only valid under `causal_condition` (the prefix must modulate from `t = 0`). As
/// upstream: the transformer refuses a cache outright, and the loop — `use_kv_cache and
/// causal_condition` — falls back to the prefill path and says so.
///
/// *Mutation that reds this:* dropping the `causal_condition` term from the loop's decision (the
/// run fails with the transformer's refusal) or the refusal from the transformer (a step-dependent
/// prefix would be cached silently).
#[test]
fn without_causal_condition_the_cache_is_refused_and_the_loop_runs_prefill() {
    let w = kv_fixture();
    let model = non_causal_model();
    let case = load_case(&w, "ref1");
    let branch = &case.branches[0];
    let (x0, t0) = (case.latents(&w, 0), case.sigma(&w, 0));
    let refused = model.forward_joint_extract(
        &branch.text,
        &joint_images(&case.reference_latents, &x0),
        t0,
        &branch.layout,
    );
    match refused {
        Err(candle_gen::CandleError::Unsupported(message)) => {
            assert!(message.contains("causal_condition"), "{message}")
        }
        Err(other) => panic!("expected the typed refusal, got {other}"),
        Ok(_) => panic!("a non-causal_condition model must refuse the prefix cache"),
    }
    // A cache from the causal model is refused just the same.
    let (_, cache) = extract(
        &load_transformer(&tiny_snapshot(), &device()).unwrap(),
        &case,
        branch,
        &x0,
        t0,
    );
    assert!(matches!(
        model.forward_joint_cached(&x0, t0, &cache),
        Err(candle_gen::CandleError::Unsupported(_))
    ));

    let sigmas = case_sigmas(&case);
    let init = w.tensor("ref1/latents_init");
    let run = |kv_cache| Run {
        model: &model,
        case: &case,
        sigmas: &sigmas,
        init: &init,
        sampler: None,
        kv_cache,
    };
    let (on, on_report) = run(KvCacheMode::On).ok();
    let (off, off_report) = run(KvCacheMode::Off).ok();
    assert_eq!(on_report.kv_cache, KvCachePath::Prefill);
    assert_eq!(on_report, off_report);
    assert_eq!(on_report.evaluations.len(), case.steps);
    assert!(on_report
        .evaluations
        .iter()
        .all(|e| e.path == EvaluationPath::Prefill));
    assert_eq!(bits(&on), bits(&off));
}

/// A cache cannot be built from a failed prefill, and it decodes only the target it was extracted
/// for.
#[test]
fn a_cache_is_only_ever_complete_and_only_fits_its_own_target() {
    let w = kv_fixture();
    let model = load_transformer(&tiny_snapshot(), &device()).unwrap();
    let case = load_case(&w, "ref2");
    let branch = &case.branches[0];
    let (x0, t0) = (case.latents(&w, 0), case.sigma(&w, 0));
    // One condition latent short: the prefill fails, and no cache comes back with the error.
    assert!(model
        .forward_joint_extract(
            &branch.text,
            &joint_images(&case.reference_latents[..1], &x0),
            t0,
            &branch.layout
        )
        .is_err());
    let (_, cache) = extract(&model, &case, branch, &x0, t0);
    let wrong_target = Tensor::zeros((1, case.target_tokens() * 2, 8), DType::F32, &device())
        .expect("a zero target");
    let err = model
        .forward_joint_cached(&wrong_target, t0, &cache)
        .expect_err("a cache must refuse a target of another size");
    assert_eq!(
        err.to_string(),
        "qwen_image_2_1: the prefix KV cache was extracted for a 16-token target, got 32"
    );
}

/// A seeded random f32 model at an arbitrary geometry — for the production head dim, and for a
/// target much longer than the prefix, neither of which the committed tiny snapshot has.
fn synthetic_model(cfg: &TransformerConfig, seed: u64) -> QwenImage21Transformer {
    use rand::SeedableRng;
    let dev = device();
    let inner = cfg.inner_dim();
    let mlp = inner * cfg.mlp_ratio;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut linear = |name: &str, out: usize, input: usize| {
        // Variance-preserving scale, so a deep stack of a wide model stays well conditioned.
        let std = 1.0 / (input as f32).sqrt();
        let values: Vec<f32> = candle_gen::seeded_normal_vec(&mut rng, out * input)
            .into_iter()
            .map(|v| v * std)
            .collect();
        tensors.insert(
            format!("{name}.weight"),
            Tensor::from_vec(values, (out, input), &dev).unwrap(),
        );
    };
    linear("img_in", inner, cfg.in_channels);
    linear("txt_in.in_layer", inner, cfg.context_in_dim);
    linear("txt_in.out_layer", inner, inner);
    linear("time_text_embed.timestep_embedder.linear_1", inner, 256);
    linear("time_text_embed.timestep_embedder.linear_2", inner, inner);
    linear("modulation.1", 4 * inner, inner);
    linear("norm_out.linear", inner, inner);
    linear("proj_out", cfg.out_channels, inner);
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            linear(&format!("{p}.attn.{name}"), inner, inner);
        }
        linear(&format!("{p}.img_mlp.gate_layer"), mlp, inner);
        linear(&format!("{p}.img_mlp.proj"), mlp, inner);
        linear(&format!("{p}.img_mlp.out"), inner, mlp);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut norm = |name: String, width: usize, centre: f32| {
        let values: Vec<f32> = candle_gen::seeded_normal_vec(&mut rng, width)
            .into_iter()
            .map(|v| centre + 0.2 * v)
            .collect();
        tensors.insert(name, Tensor::from_vec(values, width, &dev).unwrap());
    };
    norm("txt_in.text_norm.weight".into(), cfg.context_in_dim, 0.0);
    for i in 0..cfg.num_layers {
        for which in ["norm_q", "norm_k"] {
            norm(
                format!("transformer_blocks.{i}.attn.{which}.weight"),
                cfg.attention_head_dim,
                1.0,
            );
        }
    }
    QwenImage21Transformer::new(cfg, VarBuilder::from_tensors(tensors, DType::F32, &dev)).unwrap()
}

/// The production attention geometry: `head_dim = 128`, the released rope split.
fn production_head_config() -> TransformerConfig {
    TransformerConfig {
        in_channels: 8,
        out_channels: 8,
        num_layers: 2,
        attention_head_dim: 128,
        num_attention_heads: 2,
        context_in_dim: 32,
        mlp_ratio: 2,
        axes_dims_rope: [16, 56, 56],
        eps: 1e-6,
        causal_condition: true,
    }
}

fn seeded(shape: (usize, usize, usize), seed: u64) -> Tensor {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let values = candle_gen::seeded_normal_vec(&mut rng, shape.0 * shape.1 * shape.2);
    Tensor::from_vec(values, shape, &device()).unwrap()
}

/// Question 1 at the production attention geometry (`head_dim = 128`, the released
/// `axes_dims_rope`), which the tiny snapshot (`head_dim = 16`, rope `[4, 6, 6]`) does not have: a
/// two-reference interleaved layout with heterogeneous blocks, cached against prefill, in f32.
///
/// *Mutations that red this:* the same cached-step defects as the tiny-fixture test, at this
/// geometry (e.g. `k` cached before RoPE, the prefix concat skipped).
#[test]
fn cached_velocity_is_the_prefill_velocity_at_the_production_head_dim() {
    let cfg = production_head_config();
    let layout = JointLayout {
        segments: vec![
            Segment::Text { len: 7 },
            Segment::Image {
                height: 4,
                width: 2,
            },
            Segment::Image {
                height: 2,
                width: 6,
            },
            Segment::Text { len: 3 },
            Segment::Image {
                height: 4,
                width: 6,
            },
        ],
    };
    let target = layout.target_tokens();
    let text = seeded((1, 10, cfg.context_in_dim), 101);
    let refs = [
        seeded((1, 8, cfg.in_channels), 102),
        seeded((1, 12, cfg.in_channels), 103),
    ];
    let model = synthetic_model(&cfg, 40);
    let x0 = seeded((1, target, cfg.in_channels), 104);
    let (velocity, cache) = model
        .forward_joint_extract(&text, &joint_images(&refs, &x0), 0.9, &layout)
        .unwrap();
    assert_eq!(
        cache.nbytes(),
        2 * cfg.num_layers * layout.prefix_len() * cfg.inner_dim() * 4
    );
    assert_eq!(
        bits(&velocity),
        bits(
            &model
                .forward_joint(&text, &joint_images(&refs, &x0), 0.9, &layout)
                .unwrap()
        ),
        "the extract evaluation must be the prefill forward"
    );
    for (seed, sigma) in [(105u64, 0.6f32), (106, 0.15)] {
        let x = seeded((1, target, cfg.in_channels), seed);
        let want = model
            .forward_joint(&text, &joint_images(&refs, &x), sigma, &layout)
            .unwrap();
        let got = model.forward_joint_cached(&x, sigma, &cache).unwrap();
        assert_agrees(
            &format!("head_dim 128: cached vs prefill at sigma {sigma}"),
            &got,
            &want,
            ATOL,
            RTOL,
        );
        // The comparison has teeth at this geometry: the same decode against a cache extracted
        // for *other* reference latents is far outside the bound.
        let other_refs = [
            seeded((1, 8, cfg.in_channels), 202),
            seeded((1, 12, cfg.in_channels), 203),
        ];
        let (_, other) = model
            .forward_joint_extract(&text, &joint_images(&other_refs, &x0), 0.9, &layout)
            .unwrap();
        let (_, _, worst) = mixed_error(
            &model.forward_joint_cached(&x, sigma, &other).unwrap(),
            &want,
            ATOL,
            RTOL,
        );
        assert!(
            worst > 10.0,
            "a cache of other references is within {worst:.2}× the bound"
        );
    }
}

/// Elements of the storage a CPU tensor shares, and where in it the tensor starts.
fn cpu_storage_elements(t: &Tensor) -> (usize, usize) {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => (
            cpu.as_slice::<f32>().expect("an f32 tensor").len(),
            layout.start_offset(),
        ),
        _ => panic!("the parity lane is the CPU device"),
    }
}

/// The cache owns its storage. A candle `narrow` is a view of the tensor it was cut from, so a
/// cache that kept each block's prefix rows as a view would pin that block's whole prefill K/V —
/// target rows included — for the length of the denoise (upstream's "`clone()`, not
/// `contiguous()`"). Asked two ways, both derived from tensor sizes:
///
/// * **structurally** — every cached tensor's storage holds exactly `heads · prefix · head_dim`
///   elements and the tensor starts at its first one. A view's storage is the block's full K or V.
/// * **by the heap** — right after the extract, with the velocity dropped, the thread holds the
///   cache's K/V plus the target's RoPE rows, and not so much as half of one block's full K (or V)
///   more. Here the target is 64× the prefix, so a single pinned tensor is 65× the cache entry
///   that pins it.
///
/// The second geometry has **one head**: there the prefix view counts as contiguous, which is
/// exactly where `contiguous()` returns the view itself.
///
/// *Mutation that reds this:* caching the views (`owned_prefix_rows` returning the `narrow`, or
/// its `contiguous()`).
#[test]
fn the_cache_does_not_pin_the_prefill_keys_and_values() {
    for heads in [2usize, 1] {
        let cfg = TransformerConfig {
            num_attention_heads: heads,
            ..production_head_config()
        };
        let model = synthetic_model(&cfg, 60);
        let (height, width) = (32usize, 32usize);
        let layout = JointLayout::text_to_image(16, height, width);
        let text = seeded((1, 16, cfg.context_in_dim), 301);
        let x0 = seeded((1, height * width, cfg.in_channels), 302);
        // Warm the thread's GEMM workspace (see `the_caches_are_dropped_when_the_loop_returns`).
        drop(model.forward_joint(&text, &[&x0], 0.8, &layout).unwrap());

        let (cache, measured) = heap::measure(|| {
            let (velocity, cache) = model
                .forward_joint_extract(&text, &[&x0], 0.8, &layout)
                .unwrap();
            drop(velocity);
            cache
        });
        let prefix_elements = heads * layout.prefix_len() * cfg.attention_head_dim;
        for block in 0..cfg.num_layers {
            let (k, v) = cache.layer(block).unwrap();
            for (kind, tensor) in [("k", k), ("v", v)] {
                assert_eq!(
                    cpu_storage_elements(tensor),
                    (prefix_elements, 0),
                    "heads {heads}: cached {kind}_{block} must own a storage of exactly its \
                     {prefix_elements} prefix elements (the block's full tensor has {})",
                    heads * layout.total_len() * cfg.attention_head_dim
                );
            }
        }
        // cos + sin rows of the target block, f32.
        let rope = 2 * layout.target_tokens() * (cfg.attention_head_dim / 2) * 4;
        let owned = cache.nbytes() + rope;
        // One block's full K (or V): the least a single pinned view keeps alive.
        let full_tensor = layout.total_len() * cfg.inner_dim() * 4;
        eprintln!(
            "heads {heads}: cache K/V {} + rope {rope} · held {} · one block's full K {full_tensor}",
            cache.nbytes(),
            measured.held
        );
        assert!(
            measured.held <= owned + full_tensor / 2,
            "heads {heads}: holding the cache keeps {} bytes live against the {owned} it owns — \
             it is pinning a prefill tensor it was cut from (one block's full K is {full_tensor})",
            measured.held
        );
        assert!(
            measured.held >= owned,
            "heads {heads}: only {} bytes are live for a cache of {owned} — the probe is not \
             seeing the tensors",
            measured.held
        );
        // And the storage it owns is the right one: the cache still decodes.
        let x = seeded((1, height * width, cfg.in_channels), 303);
        assert_agrees(
            &format!("heads {heads}: owned cache, cached vs prefill"),
            &model.forward_joint_cached(&x, 0.4, &cache).unwrap(),
            &model.forward_joint(&text, &[&x], 0.4, &layout).unwrap(),
            ATOL,
            RTOL,
        );
    }
}

/// The extract step does not hold every block's K/V at once. Each block's prefix rows are cut from
/// that block's full `[1, heads, prefix + target, head_dim]` K and V; if the views were kept across
/// blocks and copied at the end of the forward, every block's full K/V would stay alive until then
/// — `blocks · 2 · (prefix + target) · inner · itemsize` on top of the prefill. Copying inside each
/// block, the extract costs the prefill forward plus the cache it is building:
///
/// `extract peak ≤ prefill peak + 2 · blocks · prefix · inner · itemsize + target RoPE rows`
///
/// The bound is derived from tensor sizes, not from this machine: the extract's peak may exceed
/// the plain prefill forward's peak (same inputs, measured here) by the finished cache, the
/// target's RoPE rows and **one** block's full K/V of slack. A forward that keeps all twelve
/// blocks' K/V alive exceeds the prefill by about eleven.
///
/// *Mutation that reds this:* keeping the captured views alive across blocks and copying them only
/// when the forward ends.
#[test]
fn the_extract_step_peaks_at_the_prefill_plus_the_cache() {
    let cfg = TransformerConfig {
        num_layers: 12,
        ..production_head_config()
    };
    let model = synthetic_model(&cfg, 80);
    let (height, width) = (32usize, 32usize);
    let layout = JointLayout::text_to_image(16, height, width);
    let text = seeded((1, 16, cfg.context_in_dim), 401);
    let x0 = seeded((1, height * width, cfg.in_channels), 402);
    // Warm the thread's GEMM workspace (see `the_caches_are_dropped_when_the_loop_returns`).
    drop(model.forward_joint(&text, &[&x0], 0.8, &layout).unwrap());

    let (_, prefill_heap) =
        heap::measure(|| drop(model.forward_joint(&text, &[&x0], 0.8, &layout).unwrap()));
    let (cache_bytes, extract_heap) = heap::measure(|| {
        let (_, cache) = model
            .forward_joint_extract(&text, &[&x0], 0.8, &layout)
            .unwrap();
        cache.nbytes()
    });

    let block_full_kv = 2 * layout.total_len() * cfg.inner_dim() * 4;
    let rope = 2 * layout.target_tokens() * (cfg.attention_head_dim / 2) * 4;
    assert_eq!(
        cache_bytes,
        2 * cfg.num_layers * layout.prefix_len() * cfg.inner_dim() * 4
    );
    let allowed = prefill_heap.peak + cache_bytes + rope + block_full_kv;
    let pinned = cache_bytes + rope + (cfg.num_layers - 1) * block_full_kv;
    eprintln!(
        "prefill peak {} · extract peak {} (Δ {}) · cache {cache_bytes} · rope {rope} · one \
         block's full K/V {block_full_kv} · allowed Δ {} · all-blocks-pinned Δ ≈ {pinned}",
        prefill_heap.peak,
        extract_heap.peak,
        extract_heap.peak as i64 - prefill_heap.peak as i64,
        cache_bytes + rope + block_full_kv,
    );
    assert!(
        prefill_heap.peak > block_full_kv,
        "the prefill's own peak ({}) must at least hold one block's K/V — nothing was measured",
        prefill_heap.peak
    );
    assert!(
        extract_heap.peak <= allowed,
        "the extract step peaked {} bytes above the prefill forward; the cache it builds is \
         {cache_bytes} and one block's full K/V is {block_full_kv} — it is holding several blocks' \
         K/V at once (all {} pinned would be ≈ {pinned})",
        extract_heap.peak.saturating_sub(prefill_heap.peak),
        cfg.num_layers
    );
}
