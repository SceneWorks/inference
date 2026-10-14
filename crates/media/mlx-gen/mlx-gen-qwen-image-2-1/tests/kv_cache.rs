//! The prefix KV cache (sc-25903): upstream's `QwenImage21KVCache` / `use_kv_cache=True` path.
//!
//! Three questions, each asked of the tensors the code under test actually runs on:
//!
//! 1. **Is the cached path the prefill path?** At every model evaluation of every case — text only,
//!    one, two and three references, with and without a true-CFG negative branch — the velocity the
//!    sampler consumes with the cache on equals the full-sequence (prefill) velocity for the same
//!    latents and timestep. Both sides are this port on the same device, so the bound is an
//!    absolute + relative f32 bound, not a peak-relative parity tolerance.
//! 2. **Is it upstream's cached path?** The same evaluations against
//!    `tests/fixtures/qwen21_kv_cache.safetensors` (`tools/dump_qwen21_kv_cache.py`), which records
//!    what the frozen `QwenImage21Pipeline(use_kv_cache=True)` fed its transformer and got back,
//!    plus the K/V `QwenImage21KVLayerCache` held. These run on the MLX **CPU** device: Metal's
//!    reduced-precision f32 GEMM (~1e-3 per op) would need a tolerance that hides a wrong cache.
//! 3. **Does the loop use it correctly?** Extract on the first model *evaluation*, cached on every
//!    later one (also under a solver that evaluates twice per step), one independent cache per
//!    prompt branch, target tokens only through the blocks, nothing cached when the switch is off
//!    or the model is not `causal_condition`, and nothing decoded after a cancellation.
//!
//! The tiny snapshot's `head_dim = 16` is not a geometry MLX has a fused attention kernel for; the
//! production head dim (128) is. A synthetic `head_dim = 128` model (f32 and the production bf16)
//! repeats question 1 at that geometry. Which kernel MLX dispatches is not asserted here.

use std::collections::HashMap;

use mlx_gen::gen_core::sampling::TimestepConvention;
use mlx_gen::gen_core::Progress;
use mlx_gen::sampler::run_flow_sampler;
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_qwen_image_2_1::config::TransformerConfig;
use mlx_gen_qwen_image_2_1::{
    denoise_reported, joint_images, load_scheduler_config, load_transformer, scheduler, text_rows,
    DenoiseInputs, DenoiseReport, EvaluationPath, JointLayout, KvCacheMode, KvCachePath,
    ModelEvaluation, PrefixKvCache, PromptBranch, QwenImage21Transformer, ReferenceConditioning,
    Segment,
};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{Array, Dtype};

use crate::common::{fixture, host_f32, tiny_snapshot};

/// Cached vs prefill, same port, same device, f32: `|Δ| ≤ ATOL + RTOL·|want|` per element.
///
/// Measured on Metal over every case and evaluation of the fixture: the two paths are
/// **bit-identical** per evaluation and after the whole denoise (max |Δ| = 0). The bound leaves
/// room for a backend that orders the target call's accumulation differently, and nothing more:
/// the mutations in this file's tests move the velocity by 1e-2 … 1e0.
const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-5;

/// The fused-kernel cases in the released checkpoint's bf16: also measured bit-identical. The
/// bound is under one bf16 ULP at the velocity's peak (~1.3e-2 at 3.4), and a cache extracted for
/// other reference latents sits ~50× outside it.
const BF16_ATOL: f32 = 2e-3;
const BF16_RTOL: f32 = 2e-3;

/// Port (MLX CPU, f32) vs upstream (torch CPU, f32), `max|Δ| ≤ tol · max(1, peak)`, sized from the
/// measured drift over every case, branch and evaluation of the fixture:
///
/// * the cached K/V themselves — worst `k` 7.7e-7 at peak 1.02, worst `v` 2.8e-6 at peak 4.90
///   (5.7e-7 × peak): [`KV_TOL`] is ~13× above. A `k` cached before RoPE is off by O(1).
/// * velocities and the final latents — worst 1.65e-4 at peak 4.21 (3.9e-5 × peak, the output
///   head's amplification of the same f32 noise; the extract evaluation, i.e. the prefill, is the
///   worst row): [`UPSTREAM_TOL`] is ~10× above, and 60× tighter than the Metal parity bar
///   (2.5e-2 × peak) the other gates of this crate have to use.
const KV_TOL: f32 = 1e-5;
const UPSTREAM_TOL: f32 = 4e-4;

const CASES: [&str; 6] = ["t2i", "t2i_cfg", "ref1", "ref2", "ref2_cfg", "aspect3"];

/// Run on the MLX CPU device for the guard's lifetime (restores the GPU default on drop).
struct CpuDevice;
impl CpuDevice {
    fn new() -> Self {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        CpuDevice
    }
}
impl Drop for CpuDevice {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    }
}

/// One prompt branch of a fixture case, as the transformer consumes it.
struct Branch {
    name: &'static str,
    text: Array,
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
    reference_latents: Vec<Array>,
    branches: Vec<Branch>,
}

impl Case {
    fn target_tokens(&self) -> usize {
        self.branches[0].layout.target_tokens()
    }

    fn latents(&self, w: &Weights, eval: usize) -> Array {
        w.require(&format!("{}/eval_{eval}/latents", self.name))
            .unwrap()
            .clone()
    }

    fn sigma(&self, w: &Weights, eval: usize) -> f32 {
        host_f32(
            w.require(&format!("{}/eval_{eval}/timestep", self.name))
                .unwrap(),
        )[0]
    }
}

fn load_case(w: &Weights, name: &'static str) -> Case {
    let raw = w
        .metadata(name)
        .unwrap_or_else(|| panic!("fixture metadata for case {name}"));
    let v: serde_json::Value = serde_json::from_str(raw).expect("case metadata is JSON");
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
            let hidden = w
                .require(&format!("{name}/{branch}/encoder_hidden_states"))
                .unwrap();
            // The transformer's `img_mask` spans the joint sequence (target slots appended); the
            // text rows are selected by its leading VLM-length part.
            let mask: Vec<bool> =
                host_f32(w.require(&format!("{name}/{branch}/img_mask")).unwrap())
                    [..hidden.shape()[1] as usize]
                    .iter()
                    .map(|m| *m != 0.0)
                    .collect();
            Branch {
                name: branch,
                text: text_rows(hidden, &mask).unwrap(),
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
            .map(|i| {
                w.require(&format!("{name}/ref_latents_{i}"))
                    .unwrap()
                    .clone()
            })
            .collect(),
        branches,
    }
}

/// `(max |Δ|, max |want|, worst |Δ| / (atol + rtol·|want|))`.
fn mixed_error(got: &Array, want: &Array, atol: f32, rtol: f32) -> (f32, f32, f32) {
    assert_eq!(got.shape(), want.shape(), "shape mismatch");
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
fn assert_agrees(name: &str, got: &Array, want: &Array, atol: f32, rtol: f32) {
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

/// Peak-relative gate against the upstream fixture, printing the measured numbers.
fn assert_upstream(name: &str, got: &Array, want: &Array, tol: f32) {
    crate::common::assert_close(name, got, want, tol);
}

fn prefill(
    model: &QwenImage21Transformer,
    case: &Case,
    branch: &Branch,
    x: &Array,
    t: f32,
) -> Array {
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
    x: &Array,
    t: f32,
) -> (Array, PrefixKvCache) {
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
/// *Mutations that red this:* storing the cached `k` **before** RoPE (`project` returning the
/// un-rotated key to the capture); skipping the cache concat in `Attention::forward_cached`
/// (target attends to itself only); reading the `t = 0` modulation row for the cached target
/// tokens; slicing the wrong RoPE rows for the target.
#[test]
fn cached_velocity_is_the_prefill_velocity_at_every_evaluation() {
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    let inner = model.config().inner_dim();
    for name in CASES {
        let case = load_case(&w, name);
        assert!(case.steps >= 3, "{name}: two cached evaluations are needed");
        for branch in &case.branches {
            let tag = format!("{name}/{}", branch.name);
            let (x0, t0) = (case.latents(&w, 0), case.sigma(&w, 0));
            let (velocity, cache) = extract(&model, &case, branch, &x0, t0);
            assert_agrees(
                &format!("{tag}/eval_0 extract vs prefill"),
                &velocity,
                &prefill(&model, &case, branch, &x0, t0),
                ATOL,
                RTOL,
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
                        value.shape()[1] as usize,
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

/// Question 2 at the transformer, on the CPU device: the port's extract velocity, the K/V it
/// caches for every block, and every cached-step velocity equal what upstream's
/// `kv_cache_mode="extract"` / `"cached"` calls produced.
///
/// *Mutations that red this:* caching `k` before RoPE (the fixture's `k` is post-RoPE, as
/// `QwenImage21KVLayerCache` documents); caching `v` after a norm or rotation; caching the wrong
/// rows (e.g. the target's); any cached-step defect listed on the test above.
#[test]
fn extract_cache_and_cached_velocity_match_upstream() {
    let _cpu = CpuDevice::new();
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
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
            assert_upstream(
                &format!("{tag}/eval_0 velocity"),
                &velocity,
                w.require(&format!("{name}/eval_0/{}/velocity", branch.name))
                    .unwrap(),
                UPSTREAM_TOL,
            );
            for block in 0..case.blocks {
                let (k, v) = cache.layer(block).unwrap();
                assert_eq!(
                    k.shape(),
                    &[
                        1,
                        branch.prefix_tokens as i32,
                        cfg.num_attention_heads as i32,
                        cfg.attention_head_dim as i32
                    ],
                    "{tag}: cached k layout"
                );
                for (kind, got) in [("k", k), ("v", v)] {
                    assert_upstream(
                        &format!("{tag}/{kind}_{block}"),
                        got,
                        w.require(&format!("{name}/{}/{kind}_{block}", branch.name))
                            .unwrap(),
                        KV_TOL,
                    );
                }
            }
            for eval in 1..case.steps {
                let got = model
                    .forward_joint_cached(&case.latents(&w, eval), case.sigma(&w, eval), &cache)
                    .unwrap();
                assert_upstream(
                    &format!("{tag}/eval_{eval} velocity"),
                    &got,
                    w.require(&format!("{name}/eval_{eval}/{}/velocity", branch.name))
                        .unwrap(),
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
    init: &'a Array,
    sampler: Option<&'a str>,
    kv_cache: KvCacheMode,
}

impl Run<'_> {
    fn go(
        &self,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> (mlx_gen::Result<Array>, DenoiseReport) {
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

    fn ok(&self) -> (Array, DenoiseReport) {
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
fn assert_cached_shape(tag: &str, case: &Case, report: &DenoiseReport, evaluations: usize) {
    assert_eq!(report.kv_cache, KvCachePath::Cached, "{tag}");
    let per_branch = by_branch(report);
    assert_eq!(per_branch.len(), case.branches.len(), "{tag}: branches run");
    let inner = 32;
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
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.require(&format!("{name}/latents_init")).unwrap();
        let run = |kv_cache| Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init,
            sampler: None,
            kv_cache,
        };
        let (on, on_report) = run(KvCacheMode::On).ok();
        let (off, off_report) = run(KvCacheMode::Off).ok();
        assert_agrees(&format!("{name}: cache on vs off"), &on, &off, ATOL, RTOL);
        assert_cached_shape(name, &case, &on_report, case.steps);

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

/// Question 2 for the whole loop, on the CPU device: three Euler steps with the cache on land on
/// upstream's `use_kv_cache=True` latents — and with it off on upstream's `use_kv_cache=False`
/// latents for the same case (`latents_final_prefill`). The second is the independent oracle for
/// the off path: the prefill forward this story refactored, against something that is not this
/// port's own prefill code.
///
/// *Mutation that reds the off-path row:* a numeric change to the prefill loop or forward — e.g.
/// the off path's true-CFG combine rewritten as `pos + (s − 1)·(pos − neg)` with the wrong scale,
/// or `Attention::project` normalising `k` with `norm_q`.
#[test]
fn cached_denoise_matches_upstream_end_to_end() {
    let _cpu = CpuDevice::new();
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let run = |kv_cache| {
            Run {
                model: &model,
                case: &case,
                sigmas: &sigmas,
                init: w.require(&format!("{name}/latents_init")).unwrap(),
                sampler: None,
                kv_cache,
            }
            .ok()
        };
        let (off, off_report) = run(KvCacheMode::Off);
        assert_eq!(off_report.kv_cache, KvCachePath::Prefill, "{name}");
        assert_upstream(
            &format!("{name}/latents_final_prefill (cache off)"),
            &off,
            w.require(&format!("{name}/latents_final_prefill")).unwrap(),
            UPSTREAM_TOL,
        );
        let (got, report) = run(KvCacheMode::On);
        assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
        assert_upstream(
            &format!("{name}/latents_final"),
            &got,
            w.require(&format!("{name}/latents_final")).unwrap(),
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
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for (name, sampler) in [("t2i", "heun"), ("ref2_cfg", "heun"), ("ref1", "dpmpp_sde")] {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.require(&format!("{name}/latents_init")).unwrap();
        let run = |kv_cache| Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init,
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
        assert_cached_shape(&format!("{name}/{sampler}"), &case, &on_report, evaluations);
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
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
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

/// With the switch off the loop is the plain prefill loop: the latents are bit-for-bit what an
/// equivalent closure (`forward_joint` per branch, then the true-CFG combine, under the shared
/// sampler) produces. This pins the **loop** — no extra evaluation boundary, no cache, the same
/// combine — against this port's own `forward_joint`; it cannot see a numeric change inside that
/// forward. The independent oracle for the off path is upstream's `use_kv_cache=False` output,
/// `{case}/latents_final_prefill`, gated in `cached_denoise_matches_upstream_end_to_end`.
///
/// *Mutation that reds this:* any numeric change to the off path — e.g. the true-CFG combine
/// rewritten as `pos + (s − 1)·(pos − neg)`. (An ignored switch is caught by the report assertions
/// of `cached_denoise_matches_prefill_denoise_and_reports_its_path`: on Metal the cached path's
/// latents are bit-identical to the prefill's, so bits alone could not tell.)
#[test]
fn cache_off_is_bit_identical_to_the_prefill_loop() {
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for name in CASES {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let init = w.require(&format!("{name}/latents_init")).unwrap();
        let (got, _) = Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init,
            sampler: None,
            kv_cache: KvCacheMode::Off,
        }
        .ok();

        let pos = &case.branches[0];
        let neg = case.branches.get(1).filter(|_| case.true_cfg > 1.0);
        let cancel = CancelFlag::new();
        let want = run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            0,
            &cancel,
            &mut |_| {},
            |latents: &Array, sigma: f32| -> mlx_gen::Result<Array> {
                let velocity = prefill(&model, &case, pos, latents, sigma);
                match neg {
                    Some(neg) => {
                        let uncond = prefill(&model, &case, neg, latents, sigma);
                        let diff = subtract(&velocity, &uncond)?;
                        Ok(add(
                            &uncond,
                            &multiply(&diff, Array::from_f32(case.true_cfg))?,
                        )?)
                    }
                    None => Ok(velocity),
                }
            },
        )
        .unwrap();
        assert_eq!(
            host_f32(&got)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            host_f32(&want)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{name}: cache off must be the prefill loop, bit for bit"
        );
    }
}

/// A cancellation requested when step 2 is announced — i.e. right after the step that prefills —
/// ends the loop: no evaluation decodes afterwards, and what did run is still in the report. A
/// cancellation before the first evaluation leaves nothing behind at all.
///
/// *Mutation that reds this:* a report only written back when the loop completes (a cancelled
/// run would then report nothing), or a loop that keeps evaluating after the cancellation (the
/// report would carry `Cached` entries).
#[test]
fn cancelling_after_the_prefill_step_runs_no_cached_evaluation() {
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for name in ["ref1", "ref2_cfg"] {
        let case = load_case(&w, name);
        let sigmas = case_sigmas(&case);
        let run = Run {
            model: &model,
            case: &case,
            sigmas: &sigmas,
            init: w.require(&format!("{name}/latents_init")).unwrap(),
            sampler: None,
            kv_cache: KvCacheMode::On,
        };
        for (cancel_at, extracts) in [(2u32, case.branches.len()), (1, 0)] {
            let cancel = CancelFlag::new();
            let mut announced = Vec::new();
            let (out, report) = run.go(&cancel, &mut |p| {
                if let Progress::Step { current, .. } = p {
                    announced.push(current);
                    if current == cancel_at {
                        cancel.cancel();
                    }
                }
            });
            assert!(
                matches!(out, Err(mlx_gen::Error::Canceled)),
                "{name}: cancelling at step {cancel_at} must end the denoise as Canceled"
            );
            assert_eq!(announced, (1..=cancel_at).collect::<Vec<_>>(), "{name}");
            assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
            assert_eq!(
                report
                    .evaluations
                    .iter()
                    .map(|e| e.path)
                    .collect::<Vec<_>>(),
                vec![EvaluationPath::Extract; extracts],
                "{name}: cancelled at step {cancel_at}"
            );
        }
        // The cancelled runs left nothing behind: a fresh run prefills again and completes.
        let (_, report) = run.ok();
        assert_cached_shape(name, &case, &report, case.steps);
    }
}

/// The same tiny weights as a model that is **not** `causal_condition`.
fn non_causal_model() -> QwenImage21Transformer {
    let dir = tiny_snapshot().join("transformer");
    let cfg = TransformerConfig {
        causal_condition: false,
        ..TransformerConfig::from_json_file(&dir.join("config.json")).unwrap()
    };
    QwenImage21Transformer::from_weights(&Weights::from_dir(&dir).unwrap(), &cfg).unwrap()
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
    let w = fixture("qwen21_kv_cache.safetensors");
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
        Err(mlx_gen::Error::Unsupported(message)) => {
            assert!(message.contains("causal_condition"), "{message}")
        }
        Err(other) => panic!("expected the typed refusal, got {other}"),
        Ok(_) => panic!("a non-causal_condition model must refuse the prefix cache"),
    }
    // A cache from the causal model is refused just the same.
    let (_, cache) = extract(
        &load_transformer(&tiny_snapshot()).unwrap(),
        &case,
        branch,
        &x0,
        t0,
    );
    assert!(matches!(
        model.forward_joint_cached(&x0, t0, &cache),
        Err(mlx_gen::Error::Unsupported(_))
    ));

    let sigmas = case_sigmas(&case);
    let init = w.require("ref1/latents_init").unwrap();
    let run = |kv_cache| Run {
        model: &model,
        case: &case,
        sigmas: &sigmas,
        init,
        sampler: None,
        kv_cache,
    };
    let (on, on_report) = run(KvCacheMode::On).ok();
    let (off, off_report) = run(KvCacheMode::Off).ok();
    assert_eq!(on_report.kv_cache, KvCachePath::Prefill);
    assert_eq!(on_report, off_report);
    assert!(on_report
        .evaluations
        .iter()
        .all(|e| e.path == EvaluationPath::Prefill));
    assert_eq!(
        host_f32(&on)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        host_f32(&off)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
    );
}

/// A cache cannot be built from a failed prefill, and it decodes only the target it was extracted
/// for.
#[test]
fn a_cache_is_only_ever_complete_and_only_fits_its_own_target() {
    let w = fixture("qwen21_kv_cache.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
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
    let wrong_target = Array::zeros::<f32>(&[1, case.target_tokens() as i32 * 2, 8]).unwrap();
    let err = model
        .forward_joint_cached(&wrong_target, t0, &cache)
        .expect_err("a cache must refuse a target of another size");
    assert!(
        err.to_string()
            .contains("prefix KV cache was extracted for a 16-token target"),
        "{err}"
    );
}

/// A seeded random model at an arbitrary geometry — for the production head dim, which the
/// committed tiny snapshot (`head_dim = 16`) does not have.
fn synthetic_model(cfg: &TransformerConfig, seed: u64, dtype: Dtype) -> QwenImage21Transformer {
    let inner = cfg.inner_dim() as i32;
    let mlp = inner * cfg.mlp_ratio as i32;
    let head = cfg.attention_head_dim as i32;
    let mut tensors: HashMap<String, Array> = HashMap::new();
    let mut counter = seed;
    let mut next_key = || {
        counter += 1;
        mlx_rs::random::key(counter).unwrap()
    };
    let mut linear = |name: &str, out: i32, input: i32| {
        // Variance-preserving scale, so two blocks of a wide model stay well conditioned.
        let std = 1.0 / (input as f32).sqrt();
        let weight =
            mlx_rs::random::normal::<f32>(&[out, input][..], None, std, Some(&next_key())).unwrap();
        tensors.insert(format!("{name}.weight"), weight.as_dtype(dtype).unwrap());
    };
    linear("img_in", inner, cfg.in_channels as i32);
    linear("txt_in.in_layer", inner, cfg.context_in_dim as i32);
    linear("txt_in.out_layer", inner, inner);
    linear("time_text_embed.timestep_embedder.linear_1", inner, 256);
    linear("time_text_embed.timestep_embedder.linear_2", inner, inner);
    linear("modulation.1", 4 * inner, inner);
    linear("norm_out.linear", inner, inner);
    linear("proj_out", cfg.out_channels as i32, inner);
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            linear(&format!("{p}.attn.{name}"), inner, inner);
        }
        linear(&format!("{p}.img_mlp.gate_layer"), mlp, inner);
        linear(&format!("{p}.img_mlp.proj"), mlp, inner);
        linear(&format!("{p}.img_mlp.out"), inner, mlp);
    }
    let mut norm = |name: String, width: i32, centre: f32| {
        let weight =
            mlx_rs::random::normal::<f32>(&[width][..], centre, 0.2, Some(&next_key())).unwrap();
        tensors.insert(name, weight.as_dtype(dtype).unwrap());
    };
    norm(
        "txt_in.text_norm.weight".into(),
        cfg.context_in_dim as i32,
        0.0,
    );
    for i in 0..cfg.num_layers {
        norm(
            format!("transformer_blocks.{i}.attn.norm_q.weight"),
            head,
            1.0,
        );
        norm(
            format!("transformer_blocks.{i}.attn.norm_k.weight"),
            head,
            1.0,
        );
    }
    let weights = Weights::from_map(tensors);
    weights.materialize().unwrap();
    QwenImage21Transformer::from_weights(&weights, cfg).unwrap()
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

fn seeded(shape: &[i32], seed: u64) -> Array {
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
        .unwrap()
}

/// Question 1 at the production attention geometry. This is the geometry at which MLX selects its
/// fused attention kernel on Metal — in the pinned MLX build,
/// `ScaledDotProductAttention::use_fallback` (`mlx/backend/metal/scaled_dot_product_attention.cpp`)
/// takes the fused `sdpa_full` path for `head_dim` 64 / 80 / 128 with more than eight queries and
/// no mask, which is what the cached step's only attention call is (target queries over
/// `[cached prefix, target]` keys). **The dispatch is not asserted here**: the test constructs the
/// geometry and compares cached against prefill on whatever kernel the backend picks. The tiny
/// snapshot (`head_dim = 16`) cannot reach that path at all, so this repeats cached-vs-prefill on
/// a `head_dim = 128` model with a two-reference interleaved layout, in f32 and in the released
/// checkpoint's bf16.
///
/// *Mutations that red this:* the same cached-step defects as the tiny-fixture test, at this
/// geometry (e.g. `k` cached before RoPE, the prefix concat skipped).
#[test]
fn cached_velocity_is_the_prefill_velocity_through_the_fused_attention_kernel() {
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
    assert!(
        target > 8,
        "the fused full kernel needs more than 8 queries"
    );
    let text = seeded(&[1, 10, cfg.context_in_dim as i32], 101);
    let refs = [
        seeded(&[1, 8, cfg.in_channels as i32], 102),
        seeded(&[1, 12, cfg.in_channels as i32], 103),
    ];
    for (dtype, atol, rtol) in [
        (Dtype::Float32, ATOL, RTOL),
        (Dtype::Bfloat16, BF16_ATOL, BF16_RTOL),
    ] {
        let model = synthetic_model(&cfg, 40, dtype);
        assert_eq!(model.compute_dtype(), dtype);
        let x0 = seeded(&[1, target as i32, cfg.in_channels as i32], 104);
        let (velocity, cache) = model
            .forward_joint_extract(&text, &joint_images(&refs, &x0), 0.9, &layout)
            .unwrap();
        assert_eq!(cache.layer(0).unwrap().0.dtype(), dtype, "cache dtype");
        assert_eq!(
            cache.nbytes(),
            2 * cfg.num_layers
                * layout.prefix_len()
                * cfg.inner_dim()
                * if dtype == Dtype::Float32 { 4 } else { 2 }
        );
        assert_agrees(
            &format!("{dtype:?}: extract vs prefill"),
            &velocity,
            &model
                .forward_joint(&text, &joint_images(&refs, &x0), 0.9, &layout)
                .unwrap(),
            atol,
            rtol,
        );
        for (seed, sigma) in [(105u64, 0.6f32), (106, 0.15)] {
            let x = seeded(&[1, target as i32, cfg.in_channels as i32], seed);
            let want = model
                .forward_joint(&text, &joint_images(&refs, &x), sigma, &layout)
                .unwrap();
            let got = model.forward_joint_cached(&x, sigma, &cache).unwrap();
            assert_agrees(
                &format!("{dtype:?}: cached vs prefill at sigma {sigma}"),
                &got,
                &want,
                atol,
                rtol,
            );
            // The comparison has teeth at this geometry: the same decode against a cache
            // extracted for *other* reference latents is far outside the bound.
            let other_refs = [
                seeded(&[1, 8, cfg.in_channels as i32], 202),
                seeded(&[1, 12, cfg.in_channels as i32], 203),
            ];
            let (_, other) = model
                .forward_joint_extract(&text, &joint_images(&other_refs, &x0), 0.9, &layout)
                .unwrap();
            let (_, _, worst) = mixed_error(
                &model.forward_joint_cached(&x, sigma, &other).unwrap(),
                &want,
                atol,
                rtol,
            );
            assert!(
                worst > 10.0,
                "{dtype:?}: a cache of other references is within {worst:.2}× the bound"
            );
        }
    }
}

/// The cache owns its buffers. An MLX slice is a view of the array it was cut from, so a cache
/// that kept each block's prefix rows as a slice would pin that block's whole prefill K/V — target
/// rows included — for the length of the denoise (upstream's "`clone()`, not `contiguous()`"). Here
/// the target is 64× the prefix: an owning cache holds its K/V plus the target's RoPE rows
/// (measured: exactly that, 589 824 bytes), a pinning one at least one prefill's full K/V (7×
/// more). The bound is twice the owned amount.
///
/// *Mutations that red this:* caching the slices themselves (`owned_copy` returning its input);
/// evaluating only each block's output and never its cache entry (the lazy cache then keeps every
/// block's full K/V alive).
#[test]
fn the_cache_does_not_pin_the_prefill_keys_and_values() {
    let cfg = production_head_config();
    let model = synthetic_model(&cfg, 60, Dtype::Float32);
    let (height, width) = (32usize, 32usize);
    let layout = JointLayout::text_to_image(16, height, width);
    let text = seeded(&[1, 16, cfg.context_in_dim as i32], 301);
    let x0 = seeded(&[1, (height * width) as i32, cfg.in_channels as i32], 302);
    mlx_rs::transforms::eval([&text, &x0]).unwrap();

    let before = mlx_rs::memory::get_active_memory();
    let (velocity, cache) = model
        .forward_joint_extract(&text, &[&x0], 0.8, &layout)
        .unwrap();
    drop(velocity);
    let held = mlx_rs::memory::get_active_memory().saturating_sub(before);
    // cos + sin rows of the target block, f32.
    let rope = 2 * layout.target_tokens() * (cfg.attention_head_dim / 2) * 4;
    let owned = cache.nbytes() + rope;
    let full_kv = 2 * cfg.num_layers * layout.total_len() * cfg.inner_dim() * 4;
    eprintln!(
        "cache K/V {} + rope {rope} · active held {held} · one prefill's full K/V {full_kv}",
        cache.nbytes()
    );
    assert!(4 * owned < full_kv, "the geometry must separate the two");
    assert!(
        held <= 2 * owned,
        "holding the cache keeps {held} bytes active against the {owned} it owns — it is pinning \
         the prefill buffers it was sliced from (one prefill's K/V is {full_kv})"
    );
    // ...and it IS holding them: the extract returns a materialized cache, not a lazy graph that
    // every later step would re-evaluate (an unevaluated cache holds no buffer at all).
    assert!(
        held >= owned,
        "right after the extract only {held} bytes are active for a cache of {owned}: its K/V \
         were not evaluated"
    );
    // And the buffers it owns are the right ones: the cache still decodes.
    let x = seeded(&[1, (height * width) as i32, cfg.in_channels as i32], 303);
    assert_agrees(
        "owned cache: cached vs prefill",
        &model.forward_joint_cached(&x, 0.4, &cache).unwrap(),
        &model.forward_joint(&text, &[&x], 0.4, &layout).unwrap(),
        ATOL,
        RTOL,
    );
}

/// The extract step does not hold every block's K/V at once. Each block's prefix rows are cut from
/// that block's full `[1, prefix + target, heads, head_dim]` K and V; if the copies were only
/// evaluated at the end of the forward, every block's full K/V would stay alive until then —
/// `blocks · 2 · (prefix + target) · inner · itemsize` on top of the prefill, ~1.5× the finished
/// cache at ten references and far more for a small prefix. Materializing block by block, the
/// extract costs the prefill forward plus the cache it is building.
///
/// The bound is derived from tensor sizes, not from this machine: the extract's peak may exceed the
/// plain prefill forward's peak (same inputs, measured here) by the finished cache, the target's
/// RoPE rows and **two** blocks' full K/V of slack. A forward that pins all twelve blocks' K/V
/// exceeds it by about eleven.
///
/// *Mutation that reds this:* deferring the copies' evaluation to the end of the forward (keeping
/// the captured slices alive across blocks).
#[test]
fn the_extract_step_peaks_at_the_prefill_plus_the_cache() {
    let _cpu = CpuDevice::new();
    let cfg = TransformerConfig {
        num_layers: 12,
        ..production_head_config()
    };
    let model = synthetic_model(&cfg, 80, Dtype::Float32);
    let (height, width) = (32usize, 32usize);
    let layout = JointLayout::text_to_image(16, height, width);
    let text = seeded(&[1, 16, cfg.context_in_dim as i32], 401);
    let x0 = seeded(&[1, (height * width) as i32, cfg.in_channels as i32], 402);
    mlx_rs::transforms::eval([&text, &x0]).unwrap();

    // Peak active memory above the resident baseline while `run` executes.
    let peak_of = |run: &mut dyn FnMut()| -> usize {
        mlx_rs::memory::clear_cache();
        let base = mlx_rs::memory::get_active_memory();
        mlx_rs::memory::reset_peak_memory();
        run();
        mlx_rs::memory::get_peak_memory().saturating_sub(base)
    };
    let prefill_peak = peak_of(&mut || {
        let v = model.forward_joint(&text, &[&x0], 0.8, &layout).unwrap();
        mlx_rs::transforms::eval([&v]).unwrap();
    });
    let mut cache_bytes = 0;
    let extract_peak = peak_of(&mut || {
        let (_, cache) = model
            .forward_joint_extract(&text, &[&x0], 0.8, &layout)
            .unwrap();
        cache_bytes = cache.nbytes();
    });

    let block_full_kv = 2 * layout.total_len() * cfg.inner_dim() * 4;
    let rope = 2 * layout.target_tokens() * (cfg.attention_head_dim / 2) * 4;
    assert_eq!(
        cache_bytes,
        2 * cfg.num_layers * layout.prefix_len() * cfg.inner_dim() * 4
    );
    let allowed = prefill_peak + cache_bytes + rope + 2 * block_full_kv;
    let pinned = cache_bytes + rope + (cfg.num_layers - 1) * block_full_kv;
    eprintln!(
        "prefill peak {prefill_peak} · extract peak {extract_peak} (Δ {}) · cache {cache_bytes} · \
         rope {rope} · one block's full K/V {block_full_kv} · allowed Δ {} · all-blocks-pinned Δ \
         ≈ {pinned}",
        extract_peak as i64 - prefill_peak as i64,
        cache_bytes + rope + 2 * block_full_kv,
    );
    assert!(
        prefill_peak > block_full_kv,
        "the prefill's own peak ({prefill_peak}) must at least hold one block's K/V — nothing was \
         measured"
    );
    assert!(
        extract_peak <= allowed,
        "the extract step peaked {} bytes above the prefill forward; the cache it builds is \
         {cache_bytes} and one block's full K/V is {block_full_kv} — it is holding several blocks' \
         K/V at once (all {} pinned would be ≈ {pinned})",
        extract_peak.saturating_sub(prefill_peak),
        cfg.num_layers
    );
}

/// AT4's committed half (sc-25909): **the base id's render with the cache shed is the prefill
/// loop the upstream fixture gates**, through `Generator::generate` — not only through the
/// pipeline. `generate` seeds its own noise, so it cannot be started from the fixture's
/// `latents_init`; the claim is made in two links, both on the CPU device in f32 and both in this
/// test:
///
/// 1. the base generator's pixels for a request that **selects nothing** (and for one that
///    selects `prefill`) equal, byte for byte, the render assembled here from the public pieces
///    with `KvCacheMode::Off` — the same `denoise_reported` and decode, the same seed and sigmas —
///    and the generator reports the prefill path with no cache held;
/// 2. that same `denoise_reported(KvCacheMode::Off)`, started from the fixture's `t2i` inputs,
///    lands on upstream's `use_kv_cache=False` latents (`t2i/latents_final_prefill`) within
///    [`UPSTREAM_TOL`].
///
/// Machine-independent: nothing here is a committed render, only the upstream fixture.
///
/// *Mutations that red this:* the base generator running cached when nothing is selected (link
/// 1's report, and its pixels off the `Off` assembly); any numeric change to the prefill forward
/// or loop (link 2).
#[test]
fn the_base_ids_unselected_render_is_the_prefill_loop_the_upstream_fixture_gates() {
    use mlx_gen::gen_core::{GenerationMemory, GenerationOutput, PrefixKvCacheMode};
    use mlx_gen::{GenerationRequest, LoadSpec, WeightsSource};

    let _cpu = CpuDevice::new();
    let root = tiny_snapshot();
    let (width, height, steps, seed) = (64_u32, 64_u32, 3_u32, 7_u64);
    let prompt = "a red fox in the forest";

    // Link 1: the generator against the explicit `Off` assembly.
    let generator = mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(
            mlx_gen_qwen_image_2_1::MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(root.clone())),
        )
        .unwrap();
    let tokenizer = mlx_gen_qwen_image_2_1::load_tokenizer(&root).unwrap();
    let encoder = mlx_gen_qwen_image_2_1::load_text_encoder(&root).unwrap();
    let model = load_transformer(&root).unwrap();
    let vae = mlx_gen_qwen_image_2_1::load_vae(&root).unwrap();
    let drop = mlx_gen_qwen_image_2_1::system_prompt_drop_count(&tokenizer).unwrap();
    let prepared = mlx_gen_qwen_image_2_1::prepare_conditioning_references(&encoder, &[]).unwrap();
    let conditioning = encoder
        .encode_conditioning(&tokenizer, prompt, drop, &prepared)
        .unwrap();
    let branch =
        mlx_gen_qwen_image_2_1::joint_branch(&conditioning, &prepared, width, height).unwrap();
    let scheduler_config = load_scheduler_config(&root).unwrap();
    let sigmas =
        scheduler::sigmas_for_image(&scheduler_config, steps as usize, width, height).unwrap();
    let mut report = DenoiseReport::default();
    let latents = denoise_reported(
        DenoiseInputs {
            transformer: &model,
            sigmas: &sigmas,
            latents: mlx_gen_qwen_image_2_1::create_noise(
                seed,
                width,
                height,
                model.config().in_channels,
            )
            .unwrap(),
            prompt_embeds: &branch.text,
            negative_embeds: None,
            true_cfg_scale: 1.0,
            width,
            height,
            sampler: None,
            seed,
            cancel: &CancelFlag::new(),
            references: None,
            kv_cache: KvCacheMode::Off,
        },
        &mut report,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(report.kv_cache, KvCachePath::Prefill);
    let want = mlx_gen_qwen_image_2_1::decode_rgb(&vae, &latents, width, height, None, None)
        .unwrap()
        .pixels;
    for memory in [
        None,
        Some(GenerationMemory {
            prefix_kv_cache: Some(PrefixKvCacheMode::Prefill),
            ..Default::default()
        }),
    ] {
        let req = GenerationRequest {
            prompt: prompt.to_owned(),
            width,
            height,
            steps: Some(steps),
            seed: Some(seed),
            memory,
            ..Default::default()
        };
        let rendered = generator.generate_with_report(&req, &mut |_| {}).unwrap();
        let ran = rendered.prefix_kv_cache.expect("the path is reported");
        assert_eq!(
            (ran.path, ran.cache_bytes, ran.overflow),
            (PrefixKvCacheMode::Prefill, 0, None),
            "{memory:?}: the base id sheds the cache unless it is selected"
        );
        let GenerationOutput::Images(mut images) = rendered.output.unwrap() else {
            panic!("images expected");
        };
        assert!(
            images.remove(0).pixels == want,
            "{memory:?}: the generator's render is not the explicit `Off` assembly"
        );
    }

    // Link 2: that denoise against upstream's cache-off latents for the fixture's `t2i` case.
    let w = fixture("qwen21_kv_cache.safetensors");
    let case = load_case(&w, "t2i");
    let (off, off_report) = Run {
        model: &model,
        case: &case,
        sigmas: &case_sigmas(&case),
        init: w.require("t2i/latents_init").unwrap(),
        sampler: None,
        kv_cache: KvCacheMode::Off,
    }
    .ok();
    assert_eq!(off_report.kv_cache, KvCachePath::Prefill);
    assert_upstream(
        "t2i/latents_final_prefill (the base id's shed path)",
        &off,
        w.require("t2i/latents_final_prefill").unwrap(),
        UPSTREAM_TOL,
    );
}
