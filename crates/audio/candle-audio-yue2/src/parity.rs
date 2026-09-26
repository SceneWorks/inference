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
use crate::sampling::{
    cfg_mix, distribution, probabilities, Arith, Phase, Sampling, TokenRng, VOCAB_SIZE,
};

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

fn sampling_of(v: &Value) -> Sampling {
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("sampling.{k}"));
    let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("sampling.{k}")) as usize;
    Sampling {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: u("top_k"),
        repetition_penalty: f("repetition_penalty"),
        penalty_window: u("penalty_window"),
        min_tokens: u("min_tokens"),
        max_tokens: u("max_tokens"),
    }
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
    /// Steps whose top-k ids (in order) differ.
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

/// Compare one native logits row with a reference `summarize_row` record.
fn compare_row(row: &[f32], reference: &Value, phase: Phase, probes: &[u32]) -> Spread {
    assert_eq!(row.len(), VOCAB_SIZE, "logits row width");
    let want_ids = u32s(&reference["top_ids"]);
    let want_vals = f64s(&reference["top_values"]);
    let mut allowed: Vec<(u32, f32)> = (0..VOCAB_SIZE as u32)
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
        top_id_mismatches: usize::from(got_ids != want_ids),
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

fn spread_of(steps: &Steps, case: &Value, phase: Phase, probes: &[u32]) -> Spread {
    let reference = case["steps"].as_array().expect("steps");
    assert_eq!(steps.rows.len(), reference.len(), "decode steps");
    let mut total = Spread::default();
    for (row, want) in steps.rows.iter().zip(reference) {
        total.merge(compare_row(row, want, phase, probes));
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

/// Synthetic-model tolerance. Measured (F32, both sides on the CPU, 93 decode steps over 10
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
        let s = spread_of(&steps, case, phase, &probes_of(&fixture, phase));
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
            let mut row = synthetic::row_values(&format!("sampler/row{k}"), VOCAB_SIZE, 8.0);
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
            Arith::F32 if sampling.temperature != 0.0 => {
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
