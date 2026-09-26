//! Parity of the production decode path against the pinned upstream (sc-22991).
//!
//! Every check here drives [`crate::generate::decode_tokens`] — the loop the stages run — and
//! compares what its [`DecodeObserver`] sees (the exact logits row each step samples from) and the
//! tokens it emits with fixtures produced by `scripts/reference/yue2/ar_fixtures.py` from the
//! pinned upstream `generate_tokens` / `distribution` (see `tests/fixtures/README.md`).
//!
//! * `synthetic_decodes_match_upstream` (CI): the tiny-width real-architecture model of
//!   [`crate::model::synthetic`] — greedy, guidance, ABC, `cot = off` legacy, injected-draw
//!   stochastic, natural-stop and longer-than-one-prefill-chunk decodes.
//! * `sampler_rows_match_upstream` (CI): `distribution`, the final softmax and the CFG
//!   mix on fixed rows, F32 and BF16-legacy, compared by SHA-256 of the exact output bytes.
//! * `rms_norm_and_rope_match_upstream` (CI): the RMSNorm and RoPE leaves, BF16 bit for bit.
//! * `real_weight_decodes_match_upstream` (`#[ignore]`, `YUE2_HF_HUB`): YuE2-3B in F32 on the
//!   CPU, every mode, from the upstream tokenizer's exact prompt ids.
//!
//! Tolerances were measured before they were asserted; the measured spreads and the reasoning
//! behind each bound are recorded next to the constants.

use std::collections::VecDeque;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::generate::{decode_tokens, DecodeObserver, DecodeRequest, Guidance, Hooks};
use crate::model::{synthetic, MotPaths, Yue2Lm};

/// The vocabulary width a logits row has.
const VOCAB: usize = crate::protocol::VOCAB_SIZE as usize;
use crate::sampling::{cfg_mix, distribution, probabilities, Arith, Phase, Sampling, TokenRng};

/// Draws from the fixture's recorded uniforms; a greedy decode gets an empty queue, so any draw
/// fails the test.
struct Injected(VecDeque<f32>);

impl TokenRng for Injected {
    fn next_f32(&mut self) -> f32 {
        self.0
            .pop_front()
            .expect("decode drew more uniforms than the reference")
    }
}

#[derive(Default)]
struct Steps {
    rows: Vec<Vec<f32>>,
    emitted: Vec<u32>,
}

impl DecodeObserver for Steps {
    fn on_logits(&mut self, _step: usize, logits: &[f32]) {
        self.rows.push(logits.to_vec());
    }
    fn on_token(&mut self, _phase: Phase, token: u32) {
        self.emitted.push(token);
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn phase_of(v: &Value) -> Phase {
    match v.as_str() {
        Some("abc") => Phase::Abc,
        Some("semantic") => Phase::Semantic,
        other => panic!("unknown phase {other:?}"),
    }
}

/// A fixture's seven sampling values through the protocol's own validation.
fn sampling_of(v: &Value) -> Sampling {
    Sampling::semantic_default()
        .with_json_overrides(v)
        .unwrap_or_else(|e| panic!("fixture sampling {v}: {e}"))
}

/// The largest deviations between native rows and the reference summaries of a set of steps.
#[derive(Clone, Copy, Debug, Default)]
struct Spread {
    /// Max |Δ| over the top values and probe values.
    max_abs: f64,
    /// Max |Δ| / max(1, |ref|) over the same values.
    max_rel: f64,
    /// Max |Δ logsumexp| over the phase's allowed row.
    lse_abs: f64,
    /// Max of |ΔΣx| / n (the mean per-id error) and |ΔΣx²| / Σx² over the allowed row.
    moment_rel: f64,
    /// Steps whose top-k ids disagree beyond noise (see [`top_ids_mismatch`]).
    top_id_mismatches: usize,
    /// Steps compared.
    steps: usize,
}

impl Spread {
    fn merge(&mut self, o: Spread) {
        self.max_abs = self.max_abs.max(o.max_abs);
        self.max_rel = self.max_rel.max(o.max_rel);
        self.lse_abs = self.lse_abs.max(o.lse_abs);
        self.moment_rel = self.moment_rel.max(o.moment_rel);
        self.top_id_mismatches += o.top_id_mismatches;
        self.steps += o.steps;
    }
}

/// Whether native top-k ids `got` disagree with the reference's `want` (values `want_vals`,
/// descending) beyond logit noise of `tol`: the top-1 id must match, the top-k **set** must match
/// exactly, and two entries may appear in swapped order only when the reference gap between them
/// is within `2 · tol` — two logits each within `tol` of the reference can cross only then.
fn top_ids_mismatch(got: &[u32], want: &[u32], want_vals: &[f64], tol: f64) -> bool {
    if got.first() != want.first() || got.len() != want.len() {
        return true;
    }
    let position = |id: u32| got.iter().position(|&g| g == id);
    let Some(rank): Option<Vec<usize>> = want.iter().map(|&id| position(id)).collect() else {
        return true; // a reference id is missing from the native top-k set
    };
    for i in 0..want.len() {
        for j in i + 1..want.len() {
            if rank[j] < rank[i] && want_vals[i] - want_vals[j] > 2.0 * tol {
                return true;
            }
        }
    }
    false
}

/// Compare one native logits row with a reference `summarize_row` record.
fn compare_row(row: &[f32], reference: &Value, phase: Phase, probes: &[u32], tol: f64) -> Spread {
    assert_eq!(row.len(), VOCAB, "logits row width");
    let want_ids = u32s(&reference["top_ids"]);
    let want_vals = f64s(&reference["top_values"]);
    let mut allowed: Vec<(u32, f32)> = (0..VOCAB as u32)
        .filter(|&i| phase.allows(i))
        .map(|i| (i, row[i as usize]))
        .collect();
    let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
    for &(_, x) in &allowed {
        sum += x as f64;
        sumsq += (x as f64) * (x as f64);
    }
    let max = allowed
        .iter()
        .map(|&(_, x)| x as f64)
        .fold(f64::MIN, f64::max);
    let lse = max
        + allowed
            .iter()
            .map(|&(_, x)| (x as f64 - max).exp())
            .sum::<f64>()
            .ln();
    allowed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let got_ids: Vec<u32> = allowed.iter().take(want_ids.len()).map(|x| x.0).collect();
    let mut s = Spread {
        steps: 1,
        top_id_mismatches: usize::from(top_ids_mismatch(&got_ids, &want_ids, &want_vals, tol)),
        ..Spread::default()
    };
    let mut pairs: Vec<(f64, f64)> = allowed
        .iter()
        .take(want_vals.len())
        .map(|x| x.1 as f64)
        .zip(want_vals.iter().copied())
        .collect();
    pairs.extend(
        probes
            .iter()
            .map(|&p| row[p as usize] as f64)
            .zip(f64s(&reference["probe_values"])),
    );
    for (got, want) in pairs {
        let d = (got - want).abs();
        s.max_abs = s.max_abs.max(d);
        s.max_rel = s.max_rel.max(d / want.abs().max(1.0));
    }
    s.lse_abs = (lse - reference["logsumexp"].as_f64().unwrap()).abs();
    // Σx over n ids is compared as a mean (its per-id error is the logit error); Σx² relatively.
    let n = allowed.len() as f64;
    let want_sumsq = reference["sumsq"].as_f64().unwrap();
    s.moment_rel = ((sum - reference["sum"].as_f64().unwrap()).abs() / n)
        .max((sumsq - want_sumsq).abs() / want_sumsq.max(1.0));
    s
}

/// Run one fixture decode through the production loop; return the observed rows and outputs.
fn replay(lm: &Yue2Lm, case: &Value, prefix_key: &str, sampling: &Sampling) -> (Steps, Value) {
    let prefix = u32s(&case[prefix_key]);
    let negative = case
        .get("negative")
        .or_else(|| case.get("negative_prefix"))
        .filter(|v| !v.is_null())
        .map(u32s);
    let cfg = case.get("cfg_scale").and_then(Value::as_f64).unwrap_or(1.0);
    let request = DecodeRequest {
        phase: phase_of(&case["phase"]),
        prefix: &prefix,
        sampling,
        guidance: negative.as_deref().map(|negative| Guidance {
            negative,
            scale: cfg,
        }),
        legacy_off: case
            .get("legacy_off")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let uniforms: VecDeque<f32> = f64s(&case["uniforms"]).iter().map(|&u| u as f32).collect();
    let used = uniforms.len();
    let mut rng = Injected(uniforms);
    let mut steps = Steps::default();
    let decoded = decode_tokens(
        lm,
        &request,
        &mut rng,
        Hooks {
            cancelled: &|| false,
            observer: &mut steps,
        },
    )
    .expect("decode");
    assert_eq!(decoded.tokens, u32s(&case["tokens"]), "generated tokens");
    assert_eq!(
        steps.emitted,
        u32s(&case["emitted"]),
        "emitted tokens (end id included)"
    );
    assert_eq!(
        decoded.truncated,
        case["truncated"].as_bool().unwrap(),
        "truncation flag"
    );
    assert_eq!(decoded.ended, !decoded.truncated);
    assert!(
        rng.0.is_empty(),
        "drew {} of {used} uniforms",
        used - rng.0.len()
    );
    let summary = serde_json::json!({"cfg_branches": decoded.cfg_branches});
    (steps, summary)
}

fn spread_of(steps: &Steps, case: &Value, phase: Phase, probes: &[u32], tol: f64) -> Spread {
    let reference = case["steps"].as_array().expect("steps");
    assert_eq!(steps.rows.len(), reference.len(), "decode steps");
    let mut total = Spread::default();
    for (row, want) in steps.rows.iter().zip(reference) {
        total.merge(compare_row(row, want, phase, probes, tol));
    }
    total
}

fn probes_of(fixture: &Value, phase: Phase) -> Vec<u32> {
    u32s(
        &fixture["probes"][match phase {
            Phase::Abc => "abc",
            Phase::Semantic => "semantic",
        }],
    )
}

/// Synthetic-model tolerance (also the order-swap scale of [`top_ids_mismatch`] there). Measured (F32, both sides on the CPU, 93 decode steps over 10
/// cases): max |Δ| 3.3e-6 over every compared logit (the largest under guidance scale 2, which
/// doubles the branch difference), |Δ logsumexp| ≤ 6.3e-7, mean per-id |ΔΣx| ≤ 6.7e-7 — pure
/// reduction-order noise of O(1) logits. Bounded at 2e-5 (6× the measured maximum, headroom for
/// another CPU's GEMM blocking) and still 300× below the smallest reference top-1 margin (6e-3),
/// so a structural error (a RoPE position off by one, a wrong mask, a dropped norm) fails while no
/// rounding-order change can.
const SYNTHETIC_ABS: f64 = 2e-5;

#[test]
fn synthetic_decodes_match_upstream() {
    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/ar_synthetic.json")).unwrap();
    let lm = synthetic::model(MotPaths::Ar);
    let mut all = Spread::default();
    for (name, case) in fixture["cases"].as_object().unwrap() {
        let sampling = sampling_of(&case["sampling"]);
        let phase = phase_of(&case["phase"]);
        let (steps, _) = replay(&lm, case, "prefix", &sampling);
        let s = spread_of(
            &steps,
            case,
            phase,
            &probes_of(&fixture, phase),
            SYNTHETIC_ABS,
        );
        println!("{name}: {s:?}");
        assert_eq!(s.top_id_mismatches, 0, "{name}: top ids differ");
        assert!(
            s.max_abs <= SYNTHETIC_ABS && s.lse_abs <= SYNTHETIC_ABS,
            "{name}: {s:?}"
        );
        assert!(s.moment_rel <= SYNTHETIC_ABS, "{name}: {s:?}");
        all.merge(s);
    }
    println!("all synthetic cases: {all:?}");
    assert!(all.steps >= 80, "fixture cases went missing: {all:?}");
}

fn sha(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn sampler_rows(history: &[u32]) -> Vec<Vec<f32>> {
    (0..3)
        .map(|k| {
            let mut row = synthetic::row_values(&format!("sampler/row{k}"), VOCAB, 8.0);
            for &t in history {
                row[t as usize] = 7.99;
            }
            row
        })
        .collect()
}

/// F32 softmax tolerance. PyTorch's CPU softmax uses a vectorized `expf` and a vectorized sum, so
/// the last bits of an F32 probability differ from the scalar ones here; measured over every
/// sampled fixture row: ≤ 1.3e-6 relative. Bounded at 1e-5: a probability error of that size moves
/// an inverse-CDF boundary by far less than the smallest recorded draw margin of the stochastic
/// fixtures, while any shaping error (a wrong survivor, temperature, penalty) is ≥ 1e-3.
const F32_PROBABILITY_REL: f64 = 1e-5;

/// `distribution`, the final softmax and the CFG mix are **bit-identical** to upstream's on fixed
/// rows — F32, and the BF16 `legacy_off` arithmetic (every intermediate rounding reproduced) —
/// except the F32 softmax, which is within [`F32_PROBABILITY_REL`].
#[test]
fn sampler_rows_match_upstream() {
    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/sampler.json")).unwrap();
    let history = u32s(&fixture["history"]);
    let rows = sampler_rows(&history);
    let mut failures = Vec::new();
    let mut worst_f32_prob = 0.0f64;
    for case in fixture["distribution"].as_array().unwrap() {
        let k = case["row"].as_u64().unwrap() as usize;
        let arith = match case["dtype"].as_str().unwrap() {
            "bf16" => Arith::Bf16,
            _ => Arith::F32,
        };
        let row: Vec<f32> = rows[k].iter().map(|&x| arith.round(x)).collect();
        let sampling = sampling_of(&case["sampling"]);
        let legacy = case["legacy_off"].as_bool().unwrap();
        let scores = distribution(
            &row,
            &sampling,
            &history,
            case["step"].as_u64().unwrap() as usize,
            phase_of(&case["phase"]),
            arith,
            legacy,
        );
        let finite: Vec<u32> = (0..scores.len() as u32)
            .filter(|&i| scores[i as usize].is_finite())
            .collect();
        let label = format!(
            "row {k} {} {}",
            case["case"].as_str().unwrap(),
            case["dtype"].as_str().unwrap()
        );
        if finite.len() != case["finite_count"].as_u64().unwrap() as usize
            || finite.iter().take(64).copied().collect::<Vec<_>>() != u32s(&case["finite_ids"])
        {
            let want = u32s(&case["finite_ids"]);
            let got: Vec<u32> = finite.iter().take(64).copied().collect();
            failures.push(format!(
                "{label}: surviving ids differ: {} vs {} ({:?} vs {:?})",
                finite.len(),
                case["finite_count"],
                got.iter().filter(|g| !want.contains(g)).collect::<Vec<_>>(),
                want.iter().filter(|w| !got.contains(w)).collect::<Vec<_>>()
            ));
            continue;
        }
        if sha(&scores) != case["scores_sha256"].as_str().unwrap() {
            failures.push(format!("{label}: scores differ"));
        }
        // The draw's distribution: BF16 bit for bit; F32 within the expf/summation-order
        // tolerance (a greedy row's softmax is never used, so it is not compared).
        let probs = probabilities(&scores, arith);
        match arith {
            Arith::Bf16 => {
                if sha(&probs) != case["probabilities_sha256"].as_str().unwrap() {
                    failures.push(format!("{label}: probabilities differ"));
                }
            }
            Arith::F32 if sampling.temperature() != 0.0 => {
                let rel = finite
                    .iter()
                    .zip(f64s(&case["finite_probabilities"]))
                    .map(|(&i, want)| (probs[i as usize] as f64 - want).abs() / want)
                    .fold(0.0, f64::max);
                worst_f32_prob = worst_f32_prob.max(rel);
                if rel > F32_PROBABILITY_REL {
                    failures.push(format!("{label}: probability rel diff {rel:e}"));
                }
            }
            Arith::F32 => {}
        }
    }
    for case in fixture["cfg"].as_array().unwrap() {
        let arith = match case["dtype"].as_str().unwrap() {
            "bf16" => Arith::Bf16,
            _ => Arith::F32,
        };
        let c: Vec<f32> = rows[0].iter().map(|&x| arith.round(x)).collect();
        let u: Vec<f32> = rows[1].iter().map(|&x| arith.round(x)).collect();
        let scale = case["scale"].as_f64().unwrap();
        if sha(&cfg_mix(&c, &u, scale, arith)) != case["sha256"].as_str().unwrap() {
            failures.push(format!("cfg {} scale {scale}: differs", case["dtype"]));
        }
    }
    println!("worst F32 probability relative difference: {worst_f32_prob:e}");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// F32 leaf tolerance: outputs are O(5); last-ulp libm differences are ≤ 5e-7 absolute. Measured
/// in `rms_norm_and_rope_match_upstream`; bounded at 4e-6 (≈ 8 ulp at magnitude 4).
const F32_OP_ABS: f64 = 4e-6;

/// The two leaves ported for their rounding sequence — RMSNorm and RoPE — equal upstream's
/// `RMSNorm` / `_apply_rotary` bit for bit in BF16 (CPU Candle runs BF16 elementwise ops, so this
/// is the part of the BF16 path testable without a GPU), and within [`F32_OP_ABS`] in F32.
#[test]
fn rms_norm_and_rope_match_upstream() {
    use candle_audio::candle_core::{DType, Device, Tensor};
    use candle_llm::primitives::{apply_rope, Rope};

    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/sampler.json")).unwrap();
    let shape = (1, 7, 16, 128);
    let x = Tensor::from_vec(
        synthetic::row_values("ops/x", 7 * 16 * 128, 4.0),
        shape,
        &Device::Cpu,
    )
    .unwrap();
    let w = Tensor::from_vec(
        synthetic::values("ops/w", 128, 0.25, 1.0),
        128,
        &Device::Cpu,
    )
    .unwrap();
    let flat = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let mut failures = Vec::new();
    let mut worst = 0.0f64;
    for case in fixture["ops"].as_array().unwrap() {
        let dtype = match case["dtype"].as_str().unwrap() {
            "bf16" => DType::BF16,
            _ => DType::F32,
        };
        let xd = x.to_dtype(dtype).unwrap();
        let got = match case["op"].as_str().unwrap() {
            "rms_norm" => crate::model::rms_norm(&xd, &w.to_dtype(dtype).unwrap(), 1e-6).unwrap(),
            "rope" => {
                let start = case["start"].as_u64().unwrap() as i32;
                let (cos, sin) = Rope::standard(128, 1_000_000.0)
                    .cos_sin(7, start, dtype, &Device::Cpu)
                    .unwrap();
                apply_rope(&xd, &cos, &sin, false).unwrap()
            }
            other => panic!("unknown op {other}"),
        };
        let got = flat(&got);
        let label = format!("{} {}", case["op"], case["dtype"]);
        if dtype == DType::BF16 {
            if sha(&got) != case["sha256"].as_str().unwrap() {
                failures.push(format!("{label}: differs"));
            }
            continue;
        }
        // F32: `rsqrt` vs `1/sqrt` and the platform `cosf`/`sinf`/`powf` differ in the last ulp.
        let d = got
            .iter()
            .zip(f64s(&case["head"]))
            .map(|(&g, w)| (g as f64 - w).abs())
            .fold(0.0, f64::max);
        worst = worst.max(d);
        if d > F32_OP_ABS {
            failures.push(format!("{label}: max |Δ| {d:e}"));
        }
    }
    println!("worst F32 op |Δ|: {worst:e}");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Real-weight logit tolerance (YuE2-3B, F32 on the CPU, native Candle vs PyTorch 2.10). Measured
/// over 258 decode steps (every mode, planner and semantic, guidance 1 / 1.01 / 1.5 / 2, greedy
/// and injected-draw): max |Δ| 3.6e-5 over the top-8 and probe logits, |Δ logsumexp| 3.6e-5,
/// mean per-id |ΔΣx| and relative |ΔΣx²| ≤ 1.2e-5 — 28 layers of F32 reduction-order noise on
/// logits of magnitude ~10. Bounded at 2e-4 (5.5× the measured maximum) and still 60× below the
/// smallest reference top-1 margin in the fixture (1.2e-2): a layer, position or mask error moves
/// logits by ≥ 1e-2 and fails, while GEMM blocking on another CPU cannot. Greedy and
/// injected-draw token sequences are compared exactly, not within a tolerance.
///
/// Top-8 ids: the smallest **adjacent** reference gap in the fixture is 6.9e-5, below twice the
/// measured noise, so an exact top-8 order would gate on noise. [`top_ids_mismatch`] keeps the
/// top-1 id and the top-8 set exact and counts an order swap only when the reference gap between
/// the swapped entries exceeds `2 · REAL_ABS` (two logits each within `REAL_ABS` cannot cross a
/// wider gap).
const REAL_ABS: f64 = 2e-4;
/// |Δ| between a cached decode step's logits and a native full recompute of the same sequence.
/// Measured 1.6e-5 (24 cached steps after 120–357-token prefills); bounded at 1e-4.
const CACHE_ABS: f64 = 1e-4;

fn hub_dirs() -> crate::SnapshotDirs {
    let hub = std::path::PathBuf::from(std::env::var_os("YUE2_HF_HUB").unwrap_or_else(|| {
        panic!(
            "real-weight test run without YUE2_HF_HUB (a hub directory holding the pinned repos)"
        )
    }));
    crate::inventory::REPOS
        .iter()
        .fold(crate::SnapshotDirs::new(), |dirs, repo| {
            let dir = hub
                .join(format!("models--{}", repo.id.replace('/', "--")))
                .join("snapshots")
                .join(repo.revision);
            dirs.with(repo.id, dir)
        })
}

fn opt_u32s(v: &Value) -> Option<Vec<u32>> {
    (!v.is_null()).then(|| u32s(v))
}

/// Every mode (`cot` full / melody / off, a supplied full score, a supplied melody score) through
/// the production path end to end: the request rebuilt natively ([`crate::protocol::SongRequest`]),
/// its prefixes built by the native tokenizer and protocol ([`crate::plan::SymbolicPlan`], checked
/// id for id against the upstream tokenizer's), [`crate::generate::plan_score`] and
/// [`crate::generate::generate_semantic`] on the verified YuE2-3B loaded by [`Yue2Lm::load`].
/// Greedy token sequences must match exactly; every step's logits row within [`REAL_ABS`]; a
/// cached decode must equal a full recompute; stochastic decodes replay the reference's draws.
///
/// ```text
/// YUE2_HF_HUB=/path/to/hub cargo test --release -p candle-audio-yue2 --lib \
///   parity::real_weight -- --ignored --nocapture
/// ```
///
/// CPU F32: peak RSS measured 9.5 GB (the AR path's 2.2 B parameters in F32); ~210 s on an Apple
/// M-series CPU, 16 s of it verifying and loading the checkpoint.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the doc comment)"]
fn real_weight_decodes_match_upstream() {
    use std::time::Instant;

    use candle_audio::candle_core::{DType, Device};

    use crate::generate::{generate_semantic, plan_score};
    use crate::plan::{PlanStep, SymbolicPlan};
    use crate::protocol::{GenerationConfig, SongRequest, CODEC_OFFSET};
    use crate::tokenizer::Yue2TextTokenizer;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ar_real_weights.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dirs = hub_dirs();
    let start = Instant::now();
    let tok = Yue2TextTokenizer::load(&dirs).unwrap();
    let lm = Yue2Lm::load(&dirs, MotPaths::Ar, DType::F32, &Device::Cpu).unwrap();
    println!(
        "verified + loaded the tokenizer and YuE2-3B (AR path, F32) in {:.1?}",
        start.elapsed()
    );
    let never = || false;
    let mut all = Spread::default();
    let mut cache_worst = 0.0f64;
    let report = |name: &str, s: Spread, all: &mut Spread| {
        println!("{name}: {s:?}");
        assert_eq!(s.top_id_mismatches, 0, "{name}: top ids differ: {s:?}");
        all.merge(s);
    };
    for (mode, rec) in fixture["modes"].as_object().unwrap() {
        let t0 = Instant::now();
        let request = SongRequest::from_json(&rec["request"]).unwrap();
        let sem = &rec["semantic"];
        let abc_sampling = rec
            .get("abc")
            .map_or(Sampling::abc_default(), |abc| sampling_of(&abc["sampling"]));
        let config =
            GenerationConfig::new(abc_sampling, sampling_of(&sem["sampling"]), 32).unwrap();
        let plan = match SymbolicPlan::prepare(request, &tok).unwrap() {
            PlanStep::GenerateAbc(planning) => {
                let abc = &rec["abc"];
                assert_eq!(
                    planning.prefix(),
                    &u32s(&rec["planner_prefix"])[..],
                    "{mode}: native planner prefix vs upstream's"
                );
                let mut steps = Steps::default();
                let (plan, decoded) = plan_score(
                    &lm,
                    planning,
                    &tok,
                    &config,
                    &mut Injected(VecDeque::new()),
                    Hooks {
                        cancelled: &never,
                        observer: &mut steps,
                    },
                )
                .unwrap();
                assert_eq!(decoded.tokens, u32s(&abc["tokens"]), "{mode}: planned ABC");
                assert_eq!(steps.emitted, u32s(&abc["emitted"]), "{mode}: ABC emitted");
                assert_eq!(decoded.truncated, abc["truncated"].as_bool().unwrap());
                assert_eq!(plan.truncated(), decoded.truncated);
                let probes = probes_of(&fixture, Phase::Abc);
                let s = spread_of(&steps, abc, Phase::Abc, &probes, REAL_ABS);
                report(&format!("{mode} abc"), s, &mut all);
                plan
            }
            PlanStep::Ready(plan) => plan,
        };
        assert_eq!(
            plan.abc_ids(),
            &u32s(&rec["abc_ids"])[..],
            "{mode}: plan ids"
        );
        let cond = plan.semantic_conditioning(&tok, config.semantic()).unwrap();
        let prefix = u32s(&rec["semantic_prefix"]);
        let negative = opt_u32s(&rec["negative_prefix"]);
        assert_eq!(
            cond.positive, prefix,
            "{mode}: native semantic prefix vs upstream's"
        );
        assert_eq!(
            cond.negative, negative,
            "{mode}: native negative prefix vs upstream's"
        );
        assert_eq!(cond.cfg_scale, rec["cfg_scale"].as_f64().unwrap());
        let mut steps = Steps::default();
        let out = generate_semantic(
            &lm,
            &plan,
            &cond,
            &config,
            &mut Injected(VecDeque::new()),
            Hooks {
                cancelled: &never,
                observer: &mut steps,
            },
        )
        .unwrap();
        let want = u32s(&sem["tokens"]);
        assert_eq!(out.decoded.tokens, want, "{mode}: semantic tokens");
        let codes: Vec<u32> = want.iter().map(|t| t - CODEC_OFFSET).collect();
        assert_eq!(out.codes, codes);
        assert_eq!(steps.emitted, u32s(&sem["emitted"]), "{mode}: emitted");
        assert_eq!(out.decoded.truncated, sem["truncated"].as_bool().unwrap());
        let branches = if negative.is_some() { 2 } else { 1 };
        assert_eq!(out.decoded.cfg_branches, branches);
        let probes = probes_of(&fixture, Phase::Semantic);
        let s = spread_of(&steps, sem, Phase::Semantic, &probes, REAL_ABS);
        report(&format!("{mode} semantic"), s, &mut all);
        // Cache semantics: a fresh prefill of prefix + tokens[..n-1] (a full recompute) against
        // the reference's uncached forward, and — without guidance, where the observed row is the
        // conditional row — against this decode's own last cached step.
        let mut seq = prefix.clone();
        seq.extend_from_slice(&want[..want.len() - 1]);
        let mut cache = lm.new_cache(seq.len()).unwrap();
        let full: Vec<f32> = lm
            .prefill(&seq, &mut cache, || Ok(()))
            .unwrap()
            .to_vec1()
            .unwrap();
        let reference = &sem["full_recompute_last"];
        let s = compare_row(&full, reference, Phase::Semantic, &probes, REAL_ABS);
        report(&format!("{mode} full recompute"), s, &mut all);
        if negative.is_none() {
            let last = steps.rows.last().unwrap();
            let d = full
                .iter()
                .zip(last)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0, f64::max);
            println!("{mode}: cached decode vs full recompute max |Δ| {d:e}");
            cache_worst = cache_worst.max(d);
        }
        println!("{mode}: {:.1?}", t0.elapsed());
    }
    for name in ["abc_natural_end", "abc_stochastic", "off_stochastic"] {
        let case = &fixture[name];
        let t0 = Instant::now();
        let phase = phase_of(&case["phase"]);
        let (steps, _) = replay(&lm, case, "prefix", &sampling_of(&case["sampling"]));
        let s = spread_of(&steps, case, phase, &probes_of(&fixture, phase), REAL_ABS);
        report(name, s, &mut all);
        println!("{name}: {:.1?}", t0.elapsed());
    }
    assert!(
        !fixture["abc_natural_end"]["truncated"].as_bool().unwrap(),
        "the natural-end case must end on ABC_END"
    );
    println!("ALL real-weight steps: {all:?}; cache worst {cache_worst:e}");
    assert!(
        all.max_abs <= REAL_ABS && all.lse_abs <= REAL_ABS && all.moment_rel <= REAL_ABS,
        "{all:?}"
    );
    assert!(
        cache_worst <= CACHE_ABS,
        "cache vs recompute {cache_worst:e}"
    );
}

/// The top-k comparison: top-1 and set membership are exact; an order swap counts only when the
/// reference gap between the swapped entries exceeds twice the logit tolerance.
#[test]
fn top_id_order_swaps_count_only_beyond_twice_the_tolerance() {
    let want = [7, 3, 9, 1];
    let vals = [10.0, 9.99995, 9.5, 9.0];
    let tol = 5e-5;
    assert!(!top_ids_mismatch(&want, &want, &vals, tol), "identical");
    // 3 and 7 are 5e-5 apart (< 2·tol) — but 7 is the top-1, which must match exactly.
    assert!(
        top_ids_mismatch(&[3, 7, 9, 1], &want, &vals, tol),
        "top-1 swap"
    );
    // 9 and 1 are 0.5 apart: a real reordering.
    assert!(
        top_ids_mismatch(&[7, 3, 1, 9], &want, &vals, tol),
        "large-gap swap"
    );
    // A reference id missing from the native top-k.
    assert!(
        top_ids_mismatch(&[7, 3, 9, 2], &want, &vals, tol),
        "set differs"
    );
    // A sub-tolerance swap below the top-1 is noise.
    let vals = [10.0, 9.5, 9.49995, 9.0];
    assert!(
        !top_ids_mismatch(&[7, 9, 3, 1], &want, &vals, tol),
        "noise swap"
    );
}
