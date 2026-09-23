//! Story sc-24133 (epic sc-24128): the shared on-device sampler.
//!
//! * **AC1** — the device sampler draws from the host reference's distribution: a chi-square
//!   goodness-of-fit test over 100k seeded draws, `p > 0.01`, for temperature, top-k, top-p and
//!   their combinations, against the probabilities the **host implementation** produces for the
//!   same fixed logits ([`shaped_candidates`], the reference's own shaping), plus the reference
//!   sampler run through the same test so the test itself is calibrated. CUDA rows run the kernel;
//!   the CPU rows run the same statistics over the portable selection rule.
//! * **AC2** — a temperature + top-p request copies **zero** logits rows to the host and syncs
//!   exactly once per token (the sampled id): unit-level on a tiny Qwen3.6 config on CUDA, and on
//!   real Qwen3.8-27B weights (`#[ignore]`d; writes the evidence row).
//! * **AC3** — penalties and constraint masks still take the host path, and the record says so.
//!
//! ```text
//! cargo test --locked --features cuda -p candle-llm --test device_sampler
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//! SAMPLER_EVIDENCE_OUTPUT=docs/migration/evidence/sc-24133/ac2-qwen38-27b.json \
//!   cargo test --release --locked --features cuda -p candle-llm --test device_sampler \
//!   -- --ignored --nocapture ac2_qwen38
//! ```

mod common;

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_llm::decode::{
    generate_step, CancelFlag, ConstraintMask, DecodeRecord, GenerationConfig,
};
use candle_llm::models::{Qwen35Config, Qwen35Model};
use candle_llm::primitives::{
    sample_device, sample_host, sampler_path, shaped_candidates, with_reference_sampler,
    HostSampleReason, SamplerPath, SamplingParams, SplitMix64, Weights,
};
use serde_json::json;

// ---------------------------------------------------------------------------------------------
// Chi-square machinery.
// ---------------------------------------------------------------------------------------------

/// `ln Γ(x)` (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + 7.5;
        for (i, c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Upper regularized incomplete gamma `Q(a, x)`.
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Series for P, then Q = 1 - P.
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        // Lentz continued fraction for Q.
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}

/// Chi-square p-value of `counts` against `probs` (bins with expected count < 5 pooled). Panics
/// if any draw landed on a token the reference gives zero probability.
fn chi_square_p(counts: &HashMap<i32, u64>, probs: &HashMap<i32, f64>, draws: u64) -> (f64, usize) {
    for (tok, n) in counts {
        assert!(
            probs.contains_key(tok),
            "{n} draw(s) of token {tok}, which the host reference never samples"
        );
    }
    let mut ordered: Vec<(i32, f64)> = probs.iter().map(|(t, p)| (*t, *p)).collect();
    ordered.sort_by_key(|x| x.0);
    let (mut stat, mut bins) = (0.0, 0usize);
    let (mut pooled_obs, mut pooled_exp) = (0.0, 0.0);
    for (tok, p) in ordered {
        let expected = p * draws as f64;
        let observed = *counts.get(&tok).unwrap_or(&0) as f64;
        if expected < 5.0 {
            pooled_obs += observed;
            pooled_exp += expected;
        } else {
            stat += (observed - expected).powi(2) / expected;
            bins += 1;
        }
    }
    if pooled_exp >= 5.0 {
        stat += (pooled_obs - pooled_exp).powi(2) / pooled_exp;
        bins += 1;
    } else if pooled_exp > 0.0 && bins > 0 {
        // Too little mass to stand alone: fold the pooled bin's deviation in loosely.
        stat += (pooled_obs - pooled_exp).powi(2) / pooled_exp.max(1.0);
    }
    if bins <= 1 {
        return (1.0, bins); // a single possible outcome: nothing to test beyond the support
    }
    (gamma_q((bins - 1) as f64 / 2.0, stat / 2.0), bins)
}

/// The host reference's normalized distribution for `logits` under `params`.
fn reference_probs(logits: &[f32], params: &SamplingParams) -> HashMap<i32, f64> {
    let t = Tensor::from_vec(logits.to_vec(), (1, logits.len()), &Device::Cpu).unwrap();
    let shaped = shaped_candidates(&t, &[], params, None).unwrap();
    let total: f64 = shaped.iter().map(|x| f64::from(x.1)).sum();
    shaped
        .into_iter()
        .filter(|x| x.1 > 0.0)
        .map(|(t, w)| (t, f64::from(w) / total))
        .collect()
}

fn tally(ids: impl IntoIterator<Item = i32>) -> HashMap<i32, u64> {
    let mut m = HashMap::new();
    for id in ids {
        *m.entry(id).or_insert(0) += 1;
    }
    m
}

/// A fixed, structured logits vector: a spread of magnitudes, an exact tie straddling typical
/// top-k boundaries, and a long flat tail.
fn fixed_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| {
            let i = i as f32;
            match i as usize {
                3 | 11 => 2.5, // an exact tie
                _ => 3.0 * (i * 0.61).sin() - 0.004 * i,
            }
        })
        .collect()
}

fn configs() -> Vec<(&'static str, SamplingParams)> {
    let base = SamplingParams::default();
    vec![
        (
            "temperature",
            SamplingParams {
                temperature: 0.8,
                ..base
            },
        ),
        (
            "temperature_low",
            SamplingParams {
                temperature: 0.35,
                ..base
            },
        ),
        (
            "top_k",
            SamplingParams {
                temperature: 1.0,
                top_k: 12,
                ..base
            },
        ),
        (
            "top_p",
            SamplingParams {
                temperature: 1.0,
                top_p: 0.8,
                ..base
            },
        ),
        (
            "temperature_top_p",
            SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                ..base
            },
        ),
        (
            "temperature_top_k_top_p",
            SamplingParams {
                temperature: 1.3,
                top_k: 20,
                top_p: 0.85,
                ..base
            },
        ),
    ]
}

/// The `top_k` that cuts the exact tie of `fixed_logits` in half: index 3 is the k-th largest
/// weight and index 11 (equal weight, higher index) the (k+1)-th, so the kernel's index cutoff
/// among equal keys decides — a draw of token 11 fails the support check.
fn tie_cut_k(vocab: usize) -> usize {
    match vocab {
        64 => 10,
        5_000 => 16,
        _ => panic!("no tie-cut k for vocab {vocab}"),
    }
}

/// The eight indices of `tie_run_logits`' run of equal logits, spread over the row so different
/// threads' index segments own them.
fn tie_run_indices(vocab: usize) -> Vec<usize> {
    (0..8).map(|j| 1 + j * (vocab - 2) / 8).collect()
}

/// One dominant token (the last index, weight 1), a run of eight exactly equal logits two below it
/// (weight e^-2 each), and a tail ~12 below. Top-p 0.7 puts the nucleus threshold inside the run
/// (the prefix needs 4 of the 8 ties, by index); top-k 5 likewise keeps 4 of them.
fn tie_run_logits(vocab: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..vocab)
        .map(|i| -8.0 - 0.5 * ((i as f32) * 0.37).sin())
        .collect();
    for i in tie_run_indices(vocab) {
        v[i] = 2.0;
    }
    v[vocab - 1] = 4.0;
    v
}

/// Every AC1 case at `vocab`: the shared configs on `fixed_logits`, the top-k tie cut, and the
/// top-p / top-k thresholds landing inside a run of equal weights.
fn cases(vocab: usize) -> Vec<(String, Vec<f32>, SamplingParams)> {
    let logits = fixed_logits(vocab);
    let mut out: Vec<(String, Vec<f32>, SamplingParams)> = configs()
        .into_iter()
        .map(|(name, params)| (name.to_string(), logits.clone(), params))
        .collect();
    let base = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    out.push((
        "top_k_tie_cut".into(),
        logits,
        SamplingParams {
            top_k: tie_cut_k(vocab),
            ..base
        },
    ));
    let run = tie_run_logits(vocab);
    out.push((
        "top_p_inside_tie_run".into(),
        run.clone(),
        SamplingParams { top_p: 0.7, ..base },
    ));
    out.push((
        "top_k_inside_tie_run".into(),
        run,
        SamplingParams { top_k: 5, ..base },
    ));
    out
}

const DRAWS: u64 = 100_000;
const P_FLOOR: f64 = 0.01;

/// The tie cases test what they claim: the host reference keeps index 3 and drops index 11 at the
/// tie-cut k, and keeps exactly the four lowest-index members of the tie run (plus the dominant
/// token) under both the top-p and the top-k threshold.
#[test]
fn tie_cases_split_their_ties_in_the_reference() {
    for vocab in [64usize, 5_000] {
        let cut = SamplingParams {
            temperature: 1.0,
            top_k: tie_cut_k(vocab),
            ..Default::default()
        };
        let probs = reference_probs(&fixed_logits(vocab), &cut);
        assert!(probs.contains_key(&3), "V={vocab}: tie index 3 kept");
        assert!(!probs.contains_key(&11), "V={vocab}: tie index 11 cut");

        let run = tie_run_indices(vocab);
        let mut want: Vec<i32> = run[..4].iter().map(|&i| i as i32).collect();
        want.push(vocab as i32 - 1);
        for params in [
            SamplingParams {
                temperature: 1.0,
                top_p: 0.7,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_k: 5,
                ..Default::default()
            },
        ] {
            let mut kept: Vec<i32> = reference_probs(&tie_run_logits(vocab), &params)
                .into_keys()
                .collect();
            kept.sort_unstable();
            assert_eq!(kept, want, "V={vocab} {params:?}");
        }
    }
}

/// The reference sampler through the same test: proves the statistic and the expected
/// distribution are right, so a device pass means something.
#[test]
fn ac1_host_reference_passes_its_own_chi_square() {
    for (name, logits, params) in cases(64) {
        let t = Tensor::from_vec(logits.clone(), (1, logits.len()), &Device::Cpu).unwrap();
        let probs = reference_probs(&logits, &params);
        let mut rng = SplitMix64::new(0x5eed);
        let draws = (0..DRAWS).map(|_| sample_host(&t, &[], &params, &mut rng, None).unwrap());
        let (p, bins) = chi_square_p(&tally(draws), &probs, DRAWS);
        eprintln!("[ac1 host] {name}: p = {p:.4} over {bins} bins");
        assert!(p > P_FLOOR, "{name}: host reference p = {p}");
    }
}

/// The portable selection rule `sample_device` falls back to off-CUDA (index-order walk).
#[test]
fn ac1_portable_selection_rule_matches_the_host_distribution() {
    for (name, logits, params) in cases(64) {
        let t = Tensor::from_vec(logits.clone(), (1, logits.len()), &Device::Cpu).unwrap();
        let probs = reference_probs(&logits, &params);
        let mut rng = SplitMix64::new(0xc0ffee);
        let draws = (0..DRAWS).map(|_| {
            let id = sample_device(&t, &params, &mut rng).unwrap();
            id.to_vec1::<u32>().unwrap()[0] as i32
        });
        let (p, bins) = chi_square_p(&tally(draws), &probs, DRAWS);
        eprintln!("[ac1 portable] {name}: p = {p:.4} over {bins} bins");
        assert!(p > P_FLOOR, "{name}: portable rule p = {p}");
    }
}

#[test]
fn chi_square_rejects_a_wrong_distribution() {
    // Guard against a vacuous test: temperature 0.8 draws judged against temperature 1.0.
    let logits = fixed_logits(64);
    let t = Tensor::from_vec(logits.clone(), (1, logits.len()), &Device::Cpu).unwrap();
    let drawn = SamplingParams {
        temperature: 0.8,
        ..Default::default()
    };
    let judged = SamplingParams {
        temperature: 1.0,
        ..Default::default()
    };
    let mut rng = SplitMix64::new(1);
    let draws = (0..DRAWS).map(|_| sample_host(&t, &[], &drawn, &mut rng, None).unwrap());
    let (p, _) = chi_square_p(&tally(draws), &reference_probs(&logits, &judged), DRAWS);
    assert!(p < 1e-6, "a mismatched distribution must fail, p = {p}");
}

// ---------------------------------------------------------------------------------------------
// A tiny Qwen3.6 config (4 layers: 3 GatedDeltaNet + 1 full attention) built on any device.
// ---------------------------------------------------------------------------------------------

const TINY_VOCAB: usize = 1500; // > one 1024-thread chunk, so the kernel's chunked scans run

fn tiny_qwen35(device: &Device) -> Qwen35Model {
    let cfg = Qwen35Config::from_json(&json!({
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": TINY_VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5, "max_position_embeddings": 256,
            "tie_word_embeddings": false, "full_attention_interval": 4,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4
        }
    }))
    .unwrap();
    let h = cfg.hidden_size as usize;
    let key_dim = (cfg.linear_key_head_dim * cfg.linear_num_key_heads) as usize;
    let value_dim = (cfg.linear_value_head_dim * cfg.linear_num_value_heads) as usize;
    let conv_dim = key_dim * 2 + value_dim;
    let (nh, nkv, hd) = (
        cfg.num_heads as usize,
        cfg.num_kv_heads as usize,
        cfg.head_dim as usize,
    );
    let hv = cfg.linear_num_value_heads as usize;
    let inter = cfg.intermediate_size as usize;
    let v = cfg.vocab_size as usize;
    let mut m = HashMap::new();
    let mut seed = 0usize;
    let mut t = |key: String, dims: &[usize]| {
        let n: usize = dims.iter().product();
        seed += 1;
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 7 + seed * 13) % 29) as f32 - 14.0) * 0.03)
            .collect();
        m.insert(key, Tensor::from_vec(data, dims.to_vec(), device).unwrap());
    };
    let p = "model.language_model";
    t(format!("{p}.embed_tokens.weight"), &[v, h]);
    t(format!("{p}.norm.weight"), &[h]);
    t("lm_head.weight".into(), &[v, h]);
    for i in 0..cfg.num_layers {
        let lp = |s: &str| format!("{p}.layers.{i}.{s}");
        t(lp("input_layernorm.weight"), &[h]);
        t(lp("post_attention_layernorm.weight"), &[h]);
        t(lp("mlp.gate_proj.weight"), &[inter, h]);
        t(lp("mlp.up_proj.weight"), &[inter, h]);
        t(lp("mlp.down_proj.weight"), &[h, inter]);
        if cfg.is_linear(i) {
            t(lp("linear_attn.in_proj_qkv.weight"), &[conv_dim, h]);
            t(lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
            t(lp("linear_attn.in_proj_a.weight"), &[hv, h]);
            t(lp("linear_attn.in_proj_b.weight"), &[hv, h]);
            t(lp("linear_attn.conv1d.weight"), &[conv_dim, 1, 4]);
            t(lp("linear_attn.A_log"), &[hv]);
            t(lp("linear_attn.dt_bias"), &[hv]);
            t(
                lp("linear_attn.norm.weight"),
                &[cfg.linear_value_head_dim as usize],
            );
            t(lp("linear_attn.out_proj.weight"), &[h, value_dim]);
        } else {
            t(lp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
            t(lp("self_attn.k_proj.weight"), &[nkv * hd, h]);
            t(lp("self_attn.v_proj.weight"), &[nkv * hd, h]);
            t(lp("self_attn.o_proj.weight"), &[h, nh * hd]);
            t(lp("self_attn.q_norm.weight"), &[hd]);
            t(lp("self_attn.k_norm.weight"), &[hd]);
        }
    }
    let weights = Weights::from_map(m, device.clone());
    Qwen35Model::from_weights(&weights, p, cfg).unwrap()
}

fn stochastic(max_new_tokens: usize) -> GenerationConfig {
    let mut config = GenerationConfig {
        max_new_tokens,
        seed: Some(11),
        stop_tokens: Vec::new(),
        ..Default::default()
    };
    config.sampling.temperature = 0.8;
    config.sampling.top_p = 0.9;
    config
}

/// A mask that allows only even token ids.
struct EvenOnly(Vec<bool>);

impl ConstraintMask for EvenOnly {
    fn allowed(&mut self) -> &[bool] {
        &self.0
    }
    fn accept(&mut self, _token: i32) {}
}

const PROMPT: [i32; 5] = [1, 7, 42, 300, 9];
const TOKENS: usize = 24;

fn run_step(model: &Qwen35Model, config: &GenerationConfig) -> (Vec<i32>, DecodeRecord) {
    let (out, record) = generate_step(
        model,
        &PROMPT,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    (out.tokens, record)
}

/// AC3 on any device: penalties and constraints take the host path, and the record says why.
fn assert_host_routes_are_reported(model: &Qwen35Model) {
    let mut penalized = stochastic(TOKENS);
    penalized.sampling.repetition_penalty = 1.3;
    penalized.sampling.repetition_context = 16;
    let (tokens, record) = run_step(model, &penalized);
    assert_eq!(tokens.len(), TOKENS);
    assert_eq!(
        record.sampler.path,
        Some(SamplerPath::Host(HostSampleReason::Penalty))
    );
    assert_eq!(record.sampler.label(), "host:penalty");
    assert_eq!(record.sampler.logits_to_host, TOKENS as u64);
    assert_eq!(record.sampler.host_draws, TOKENS as u64);

    let mut presence = stochastic(TOKENS);
    presence.sampling.presence_penalty = 0.4;
    let (_, record) = run_step(model, &presence);
    assert_eq!(record.sampler.label(), "host:penalty");

    let mut mask = EvenOnly((0..TINY_VOCAB).map(|i| i % 2 == 0).collect());
    let (out, record) = generate_step(
        model,
        &PROMPT,
        &stochastic(TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        Some(&mut mask),
    )
    .unwrap();
    assert!(out.tokens.iter().all(|t| t % 2 == 0), "mask honoured");
    assert_eq!(
        record.sampler.path,
        Some(SamplerPath::Host(HostSampleReason::Constraint))
    );
    assert_eq!(record.sampler.logits_to_host, TOKENS as u64);
}

/// E1: greedy is the device argmax on every device, one sync per token, no logits copy, and
/// token-identical with the reference sampler forced.
fn assert_greedy_unchanged(model: &Qwen35Model) {
    let mut greedy = stochastic(TOKENS);
    greedy.sampling.temperature = 0.0;
    let (tokens, record) = run_step(model, &greedy);
    let (reference, _) = with_reference_sampler(|| run_step(model, &greedy));
    assert_eq!(tokens, reference);
    assert_eq!(record.sampler.path, Some(SamplerPath::Device));
    assert_eq!(record.sampler.logits_to_host, 0);
    assert_eq!(record.host_syncs, TOKENS as u64);
}

#[test]
fn cpu_routes_are_reported_and_never_silent() {
    let model = tiny_qwen35(&Device::Cpu);
    // No device kernel on CPU: a temperature + top-p request is a host request, and says so.
    let (_, record) = run_step(&model, &stochastic(TOKENS));
    assert_eq!(
        record.sampler.path,
        Some(SamplerPath::Host(HostSampleReason::DeviceUnavailable))
    );
    assert_eq!(record.sampler.logits_to_host, TOKENS as u64);
    let (_, record) = with_reference_sampler(|| run_step(&model, &stochastic(TOKENS)));
    assert_eq!(record.sampler.label(), "host:reference");
    assert_host_routes_are_reported(&model);
    assert_greedy_unchanged(&model);
}

#[test]
fn routing_policy_is_the_core_llm_policy_plus_device_availability() {
    let cpu = Device::Cpu;
    let top_p = stochastic(1).sampling;
    assert_eq!(
        sampler_path(&cpu, 32, &top_p, false),
        SamplerPath::Host(HostSampleReason::DeviceUnavailable)
    );
    assert_eq!(
        sampler_path(&cpu, 32, &top_p, true),
        SamplerPath::Host(HostSampleReason::Constraint)
    );
    assert_eq!(
        sampler_path(&cpu, 32, &SamplingParams::default(), false),
        SamplerPath::Device,
        "greedy argmax is on-device everywhere"
    );
    // A greedy request with a penalty was, and stays, a host request.
    let greedy_penalized = SamplingParams {
        presence_penalty: 0.5,
        ..Default::default()
    };
    assert_eq!(
        sampler_path(&cpu, 32, &greedy_penalized, false),
        SamplerPath::Host(HostSampleReason::Penalty)
    );
}

// ---------------------------------------------------------------------------------------------
// CUDA: the kernel itself.
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::DType;
    use candle_llm::decode::{generate_with, DecodePath};
    use candle_llm::primitives::{sample, uniform_device, TokenRng};

    fn gpu() -> Device {
        Device::new_cuda(0).expect("the cuda lane runs on a CUDA device")
    }

    fn device_draws_batched(logits: &[f32], params: &SamplingParams, seed: u64) -> Vec<i32> {
        let device = gpu();
        let row = Tensor::from_vec(logits.to_vec(), (1, logits.len()), &device).unwrap();
        let rows = row.broadcast_as((DRAWS as usize, logits.len())).unwrap();
        let mut rng = SplitMix64::new(seed);
        sample_device(&rows.contiguous().unwrap(), params, &mut rng)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .into_iter()
            .map(|x| x as i32)
            .collect()
    }

    #[test]
    fn ac1_device_sampler_matches_the_host_distribution() {
        for vocab in [64usize, 5_000] {
            for (name, logits, params) in cases(vocab) {
                let probs = reference_probs(&logits, &params);
                let draws = device_draws_batched(&logits, &params, 0xace1);
                let (p, bins) = chi_square_p(&tally(draws), &probs, DRAWS);
                eprintln!("[ac1 cuda] V={vocab} {name}: p = {p:.4} over {bins} bins");
                assert!(p > P_FLOOR, "V={vocab} {name}: device p = {p}");
            }
        }
    }

    /// The per-token route `sample` takes, on the real vocabulary width (248 320 — every
    /// chunked scan in the kernel runs hundreds of iterations).
    #[test]
    fn ac1_device_sampler_matches_on_the_qwen38_vocabulary_width() {
        let vocab = 248_320usize;
        // Qwen-like logits: most of the vocabulary far below a few dozen plausible tokens.
        let logits: Vec<f32> = (0..vocab)
            .map(|i| {
                let x = i as f32;
                if i % 9_973 == 17 {
                    6.0 + (x * 0.37).sin() * 2.0
                } else {
                    -4.0 + (x * 0.013).sin() * 3.0
                }
            })
            .collect();
        let device = gpu();
        let t = Tensor::from_vec(logits.clone(), (1, vocab), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 20,
            ..Default::default()
        };
        let probs = reference_probs(&logits, &params);
        let mut rng = SplitMix64::new(0x248);
        let draws: Vec<i32> = (0..DRAWS)
            .map(|_| sample(&t, &[], &params, &mut rng, None).unwrap())
            .collect();
        let (p, bins) = chi_square_p(&tally(draws), &probs, DRAWS);
        eprintln!(
            "[ac1 cuda] V={vocab} temperature_top_k_top_p (per-token): p = {p:.4} over {bins} bins"
        );
        assert!(p > P_FLOOR, "p = {p}");

        // Temperature only, over the whole vocabulary (no threshold passes).
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let probs = reference_probs(&logits, &params);
        let draws = (0..DRAWS).map(|_| sample(&t, &[], &params, &mut rng, None).unwrap());
        let (p, bins) = chi_square_p(&tally(draws), &probs, DRAWS);
        eprintln!("[ac1 cuda] V={vocab} temperature (per-token): p = {p:.4} over {bins} bins");
        assert!(p > P_FLOOR, "p = {p}");
    }

    #[test]
    fn seeded_device_draws_are_deterministic_and_track_the_portable_rule() {
        let logits = fixed_logits(5_000);
        let (_, params) = configs().pop().unwrap();
        let a = device_draws_batched(&logits, &params, 99);
        let b = device_draws_batched(&logits, &params, 99);
        assert_eq!(a, b, "same seed, same draws");
        // Same uniforms, same index-order rule: the CPU rule lands on the same token except where
        // fixed-point and f64 rounding straddle a boundary.
        let t = Tensor::from_vec(logits.clone(), (1, logits.len()), &Device::Cpu).unwrap();
        let mut rng = SplitMix64::new(99);
        let agree = a
            .iter()
            .filter(|&&d| {
                let id = sample_device(&t, &params, &mut rng).unwrap();
                id.to_vec1::<u32>().unwrap()[0] as i32 == d
            })
            .count();
        assert!(
            agree as f64 >= 0.999 * a.len() as f64,
            "{agree}/{} agree",
            a.len()
        );
    }

    #[test]
    fn uniform_device_is_the_host_splitmix_stream() {
        let device = gpu();
        let mut dev_rng = SplitMix64::new(0xdead_beef);
        let mut host_rng = SplitMix64::new(0xdead_beef);
        let u = uniform_device(&mut dev_rng, 4_099, &device)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected: Vec<f32> = (0..4_099).map(|_| host_rng.next_f32()).collect();
        assert_eq!(u, expected, "bit-identical to next_f32");
        assert_eq!(
            dev_rng.state(),
            host_rng.state(),
            "the stream advanced past them"
        );
    }

    #[test]
    fn degenerate_rows_fall_back_like_the_host() {
        let device = gpu();
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };
        let cases: Vec<(Vec<f32>, i32)> = vec![
            (vec![f32::NEG_INFINITY; 2_000], 0),
            (
                {
                    let mut v = vec![0.0f32; 2_000];
                    v[1_500] = f32::INFINITY;
                    v
                },
                1_500,
            ),
            (
                {
                    let mut v = vec![0.0f32; 2_000];
                    v[7] = 3.0;
                    v[1_999] = f32::NAN;
                    v
                },
                7,
            ),
        ];
        // Separate, identically seeded streams: every degenerate row consumes exactly one draw
        // on the device path and on the host reference, so the streams stay in lockstep.
        let mut dev_rng = SplitMix64::new(3);
        let mut host_rng = SplitMix64::new(3);
        for (row, want) in cases {
            let t = Tensor::from_vec(row.clone(), (1, row.len()), &device).unwrap();
            let host = Tensor::from_vec(row.clone(), (1, row.len()), &Device::Cpu).unwrap();
            let before = dev_rng.state();
            let got = sample(&t, &[], &params, &mut dev_rng, None).unwrap();
            let reference = sample_host(&host, &[], &params, &mut host_rng, None).unwrap();
            assert_eq!(got, want);
            assert_eq!(reference, want, "host reference agrees");
            assert_eq!(
                dev_rng.state(),
                before.wrapping_add(SplitMix64::INCREMENT),
                "the device path consumed one draw"
            );
            assert_eq!(
                host_rng.state(),
                dev_rng.state(),
                "the host reference consumed the same draw"
            );
        }
        let mut rng = dev_rng;
        // top_k = 1 and top_p = 0 are the argmax; ties go to the lower index.
        let mut row = fixed_logits(3_000);
        row[2_900] = 50.0;
        row[2_950] = 50.0;
        let t = Tensor::from_vec(row, (1, 3_000), &device).unwrap();
        for params in [
            SamplingParams {
                temperature: 1.0,
                top_k: 1,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_p: 0.0,
                ..Default::default()
            },
        ] {
            for _ in 0..50 {
                assert_eq!(sample(&t, &[], &params, &mut rng, None).unwrap(), 2_900);
            }
        }
    }

    /// A temperature with no finite, non-zero reciprocal never reaches the kernel (which cannot
    /// shape NaN weights): it takes the host reference, labelled `host:degenerate_temperature`.
    #[test]
    fn degenerate_temperatures_take_the_host_reference_on_cuda() {
        let device = gpu();
        let mut row = vec![f32::NEG_INFINITY; 2_000];
        for (i, x) in [(5usize, 1.0f32), (700, 3.0), (1_500, 2.0)] {
            row[i] = x;
        }
        let t = Tensor::from_vec(row, (1, 2_000), &device).unwrap();
        for temperature in [1e-39_f32, f32::INFINITY, f32::NAN] {
            let params = SamplingParams {
                temperature,
                top_p: 0.9,
                ..Default::default()
            };
            assert_eq!(
                sampler_path(&device, 2_000, &params, false),
                SamplerPath::Host(HostSampleReason::DegenerateTemperature),
                "T = {temperature:e}"
            );
            let mut rng = SplitMix64::new(8);
            let id = sample(&t, &[], &params, &mut rng, None).unwrap();
            let batched = sample_device(&t, &params, &mut rng).unwrap();
            let batched = batched.to_vec1::<u32>().unwrap()[0] as i32;
            // 1e-39 is the T -> 0 limit; +inf and NaN weights meet a -inf logit (-inf * 0 is
            // NaN), which the host reference resolves to the argmax as it always has.
            assert_eq!(id, 700, "T = {temperature:e}");
            assert_eq!(batched, 700, "T = {temperature:e}");
            assert_eq!(
                candle_llm::primitives::last_host_reason(),
                Some(HostSampleReason::DegenerateTemperature)
            );
        }
    }

    /// Top-p boundaries placed exactly on, and at the rounding edge of, the nucleus threshold:
    /// the device kept-set (every token a batch of seeded draws lands on) against the host
    /// reference's (`shaped_candidates`). Exact boundaries must agree exactly; rounding-edge ones
    /// may differ by the documented one token (`nucleus_select`, `sampler_cuda.cu` header).
    #[test]
    fn nucleus_boundary_divergence_is_at_most_one_token() {
        const V: usize = 2_000;
        const N: usize = 20_000;
        let device = gpu();
        let kept_on_device = |logits: &[f32], params: &SamplingParams| -> usize {
            let row = Tensor::from_vec(logits.to_vec(), (1, V), &device).unwrap();
            let rows = row.broadcast_as((N, V)).unwrap().contiguous().unwrap();
            let mut ids = sample_device(&rows, params, &mut SplitMix64::new(0xb0))
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        let kept_on_host = |logits: &[f32], params: &SamplingParams| -> usize {
            let t = Tensor::from_vec(logits.to_vec(), (1, V), &Device::Cpu).unwrap();
            shaped_candidates(&t, &[], params, None).unwrap().len()
        };
        let spread = |count: usize| (0..count).map(move |j| 3 + j * (V - 7) / count);

        // Exact in both arithmetics: `count` weights of exactly 1.0 (logit 0, the rest -inf) and
        // top_p * count an integer, so both sides stop at exactly `j` tokens.
        for count in [4usize, 8] {
            let mut logits = vec![f32::NEG_INFINITY; V];
            for i in spread(count) {
                logits[i] = 0.0;
            }
            for j in 1..count {
                let params = SamplingParams {
                    temperature: 1.0,
                    top_p: j as f32 / count as f32,
                    ..Default::default()
                };
                let host = kept_on_host(&logits, &params);
                let dev = kept_on_device(&logits, &params);
                assert_eq!(host, j, "host, {j}/{count}");
                assert_eq!(
                    dev, host,
                    "exact boundary {j}/{count}: device {dev}, host {host}"
                );
            }
        }

        // At the rounding edge: top_p set to each descending prefix's exact share of the host
        // mass, and one f32 ulp either side of it.
        let mut logits = vec![f32::NEG_INFINITY; V];
        let values = [0.0f32, -0.3, -0.7, -1.1, -1.6, -2.2];
        for (i, x) in spread(values.len()).zip(values) {
            logits[i] = x;
        }
        let weights: Vec<f64> = values.iter().map(|&x| f64::from(x.exp())).collect();
        let total: f64 = weights.iter().sum();
        let (mut edges, mut equal) = (0usize, 0usize);
        for j in 1..values.len() {
            let share = (weights[..j].iter().sum::<f64>() / total) as f32;
            for top_p in [
                f32::from_bits(share.to_bits() - 1),
                share,
                f32::from_bits(share.to_bits() + 1),
            ] {
                let params = SamplingParams {
                    temperature: 1.0,
                    top_p,
                    ..Default::default()
                };
                let host = kept_on_host(&logits, &params);
                let dev = kept_on_device(&logits, &params);
                assert!(
                    host.abs_diff(dev) <= 1,
                    "prefix {j}, top_p {top_p:e}: device kept {dev}, host kept {host}"
                );
                edges += 1;
                equal += usize::from(host == dev);
            }
        }
        eprintln!("[nucleus boundary] rounding-edge cases: {equal}/{edges} identical, rest +-1");
    }

    #[test]
    fn bf16_logits_sample_on_device() {
        let device = gpu();
        let logits = fixed_logits(64);
        let t = Tensor::from_vec(logits, (1, 64), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let params = stochastic(1).sampling;
        let mut rng = SplitMix64::new(5);
        let id = sample(&t, &[], &params, &mut rng, None).unwrap();
        assert!((0..64).contains(&id));
    }

    /// AC2, unit level: a temperature + top-p request on a tiny config copies zero logits rows and
    /// syncs exactly once per token — the same as greedy — on both decode drivers.
    #[test]
    fn ac2_tiny_config_temperature_top_p_has_zero_logits_copies_and_one_sync_per_token() {
        let model = tiny_qwen35(&gpu());
        let (tokens, record) = run_step(&model, &stochastic(TOKENS));
        assert_eq!(tokens.len(), TOKENS);
        assert_eq!(record.path, DecodePath::StepModel);
        assert_eq!(record.sampler.path, Some(SamplerPath::Device));
        assert_eq!(record.sampler.label(), "device");
        assert_eq!(
            record.sampler.logits_to_host, 0,
            "no logits row reached the host"
        );
        assert_eq!(record.sampler.device_draws, TOKENS as u64);
        assert_eq!(record.sampler.host_draws, 0);
        assert_eq!(
            record.host_syncs, TOKENS as u64,
            "exactly one sync per token"
        );
        assert_eq!(record.logits_to_host_per_token(), Some(0.0));

        // The same request on the reference `Decode` loop (the provider's driver).
        let span = candle_llm::decode::RequestSpan::begin();
        let out = generate_with(
            &model,
            &PROMPT,
            &stochastic(TOKENS),
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let counters = span.counters();
        assert_eq!(
            out.tokens, tokens,
            "both drivers draw the same seeded tokens"
        );
        assert_eq!(counters.sampler.logits_to_host, 0);
        assert_eq!(counters.host_syncs, TOKENS as u64);

        // The forced host reference: same request, one logits row per token, and it says so.
        let (_, reference) = with_reference_sampler(|| run_step(&model, &stochastic(TOKENS)));
        assert_eq!(reference.sampler.label(), "host:reference");
        assert_eq!(reference.sampler.logits_to_host, TOKENS as u64);
    }

    #[test]
    fn ac3_cuda_penalties_and_constraints_take_the_host_path() {
        let model = tiny_qwen35(&gpu());
        assert_host_routes_are_reported(&model);
        assert_greedy_unchanged(&model);
    }

    /// The sampler's own per-token cost on the Qwen3.8 vocabulary width (no weights): device
    /// sampler vs host reference vs greedy argmax, each a full `sample` call including its sync.
    /// `#[ignore]`d measurement; run in release for the evidence row.
    #[test]
    #[ignore = "measurement; run in release with --ignored --nocapture"]
    fn sampler_cost_per_token_on_the_qwen38_vocabulary_width() {
        use std::time::Instant;
        let device = gpu();
        let vocab = 248_320usize;
        // `flat`: every token within ~8 logits of the max (the worst case for the thresholds).
        // `peaked`: an LLM-shaped row - a few dozen plausible tokens, the vocabulary 15-25
        // logits below them.
        let flat: Vec<f32> = (0..vocab)
            .map(|i| ((i as f32) * 0.37).sin() * 4.0 + ((i % 97) as f32) * 0.01)
            .collect();
        let peaked: Vec<f32> = (0..vocab)
            .map(|i| {
                let x = i as f32;
                if i % 5_003 == 11 {
                    22.0 + (x * 0.41).sin() * 3.0
                } else {
                    2.0 + (x * 0.017).sin() * 4.0
                }
            })
            .collect();
        let greedy = SamplingParams::default();
        let mut rng = SplitMix64::new(1);
        let time = |f: &mut dyn FnMut() -> i32| {
            for _ in 0..20 {
                f();
            }
            device.synchronize().unwrap();
            let n = 300;
            let start = Instant::now();
            for _ in 0..n {
                f();
            }
            start.elapsed().as_secs_f64() * 1e6 / n as f64
        };
        let configs = [
            (
                "temperature",
                SamplingParams {
                    temperature: 0.8,
                    ..greedy
                },
            ),
            ("temperature_top_p", stochastic(1).sampling),
            (
                "temperature_top_k",
                SamplingParams {
                    temperature: 0.8,
                    top_k: 20,
                    ..greedy
                },
            ),
            (
                "temperature_top_k_top_p",
                SamplingParams {
                    temperature: 0.7,
                    top_k: 20,
                    top_p: 0.8,
                    ..greedy
                },
            ),
        ];
        let mut rows = Vec::new();
        for (shape, logits) in [("flat", flat), ("peaked", peaked)] {
            let t = Tensor::from_vec(logits, (1, vocab), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            for (name, params) in configs {
                let device_us = time(&mut || sample(&t, &[], &params, &mut rng, None).unwrap());
                let host_us = with_reference_sampler(|| {
                    time(&mut || sample(&t, &[], &params, &mut SplitMix64::new(2), None).unwrap())
                });
                rows.push(json!({
                    "logits": shape,
                    "sampling": name,
                    "device_sampler_us_per_token": device_us,
                    "host_reference_us_per_token": host_us,
                }));
            }
            let greedy_us = time(&mut || sample(&t, &[], &greedy, &mut rng, None).unwrap());
            rows.push(json!({
                "logits": shape,
                "sampling": "greedy_argmax",
                "device_sampler_us_per_token": greedy_us,
            }));
        }
        eprintln!(
            "{}",
            json!({
                "hardware": "RTX Pro 6000 / sm_120",
                "vocab": vocab,
                "logits_dtype": "bf16",
                "note": "one full `sample` call per token, including the id sync",
                "rows": rows,
            })
        );
    }
}

// ---------------------------------------------------------------------------------------------
// AC2 on real Qwen3.8-27B weights (evidence row).
// ---------------------------------------------------------------------------------------------

const SNAPSHOT_VAR: &str = "BONSAI_QWEN38_SNAPSHOT";
const EVIDENCE_VAR: &str = "SAMPLER_EVIDENCE_OUTPUT";
const REAL_TOKENS: usize = 128;
const REAL_PROMPT: &str = "Write a short story about a lighthouse keeper who discovers that the \
    light has been signalling to someone across the sea. Use vivid, varied language.";

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU with ~54 GB free"]
fn ac2_qwen38_temperature_top_p_decodes_without_logits_copies() {
    use candle_llm::decode::generate_step_timed;
    use candle_llm::device::select_device;
    use std::time::Instant;

    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, REAL_PROMPT);
    let mut config = GenerationConfig {
        max_new_tokens: REAL_TOKENS,
        seed: Some(20_240_923),
        stop_tokens: Vec::new(),
        ..Default::default()
    };
    config.sampling.temperature = 0.7;
    config.sampling.top_p = 0.9;

    let run = |config: &GenerationConfig| {
        let mut decode_started = None;
        let mut boundary = || -> candle_llm::Result<()> {
            device.synchronize()?;
            decode_started = Some(Instant::now());
            Ok(())
        };
        let (out, record, _) = generate_step_timed(
            &model,
            &prompt,
            config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            Some(&mut boundary),
        )
        .unwrap();
        device.synchronize().unwrap();
        let decode_s = decode_started.unwrap().elapsed().as_secs_f64();
        (out.tokens, record, decode_s)
    };

    // Warm both paths (kernel compile, allocator) before timing.
    let mut warm = config.clone();
    warm.max_new_tokens = 8;
    run(&warm);
    with_reference_sampler(|| run(&warm));

    let (device_tokens, device_record, device_s) = run(&config);
    let (host_tokens, host_record, host_s) = with_reference_sampler(|| run(&config));

    assert_eq!(device_tokens.len(), REAL_TOKENS);
    assert_eq!(host_tokens.len(), REAL_TOKENS);
    assert_eq!(device_record.sampler.path, Some(SamplerPath::Device));
    assert_eq!(
        device_record.sampler.logits_to_host, 0,
        "zero logits copies"
    );
    assert_eq!(
        device_record.host_syncs, REAL_TOKENS as u64,
        "one sync per token"
    );
    assert_eq!(
        host_record.sampler.path,
        Some(SamplerPath::Host(HostSampleReason::Reference))
    );

    // The decode phase emits REAL_TOKENS - 1 tokens after the prefill boundary (the first token
    // is sampled from the prefill logits, inside the timed window too), so tok/s counts all.
    let row = |record: &DecodeRecord, secs: f64| {
        json!({
            "sampler_path": record.sampler.label(),
            "generated_tokens": record.generated_tokens,
            "host_syncs": record.host_syncs,
            "host_syncs_per_token": record.host_syncs_per_token(),
            "logits_to_host": record.sampler.logits_to_host,
            "logits_to_host_per_token": record.logits_to_host_per_token(),
            "device_draws": record.sampler.device_draws,
            "host_draws": record.sampler.host_draws,
            "decode_seconds": secs,
            "decode_tokens_per_second": record.generated_tokens as f64 / secs,
        })
    };
    // The code the row was measured at: HEAD, and whether tracked files had uncommitted changes.
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]).expect("git rev-parse HEAD");
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(true);
    let evidence = json!({
        "story": "sc-24133",
        "commit": commit,
        "worktree_dirty": dirty,
        "hardware": "RTX Pro 6000 / sm_120",
        "model": "Qwen3.8-27B",
        "snapshot": snapshot.file_name().map(|s| s.to_string_lossy().into_owned()),
        "driver": "generate_step (StepModel)",
        "prompt_tokens": prompt.len(),
        "sampling": { "temperature": 0.7, "top_p": 0.9, "top_k": 0, "seed": 20_240_923u64 },
        "device": row(&device_record, device_s),
        "host_reference": row(&host_record, host_s),
    });
    let text = serde_json::to_string_pretty(&evidence).unwrap();
    eprintln!("{text}");
    if let Some(path) = std::env::var_os(EVIDENCE_VAR).filter(|v| !v.is_empty()) {
        std::fs::write(path, text + "\n").unwrap();
    }
}
