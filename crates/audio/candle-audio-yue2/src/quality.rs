//! Measured output quality of every precision tier against the F32 reference, on the real YuE2-3B
//! and through the production engine loader (sc-22995, epic E8/E9).
//!
//! `#[ignore]`d (CI has no weights); under `--ignored` a missing input panics. Each configuration is
//! loaded with [`Yue2Engine::load_with_precision`] — the `q8` / `q4` tiers from snapshots derived by
//! [`crate::tier::convert`] from the verified original, the FP8 mode prepared by the engine — and
//! measured at every stage that matters:
//!
//! * **AR** — teacher-forced on the committed real-weight fixture's exact token sequences
//!   (`tests/fixtures/ar_real_weights.json`: every mode's score and semantic tokens and the three
//!   stochastic decodes), through the model the engine's AR stages run (FP8 prepared). Per step,
//!   over the phase's allowed ids: top-1 agreement, top-8 set overlap, KL(reference ‖ tier) of the
//!   temperature-1 distributions, and the logits' relative L2.
//! * **Acoustic** — [`Yue2Engine::synthesize`] (which restores BF16 before the NAR in the FP8 mode)
//!   on the fixture's `supplied_full` plan with the semantic codes of `nar_real_reference.json`'s two
//!   cases, the song noise drawn from the request seed: latent relative L2, cosine and SNR.
//! * **Decode** — [`Yue2Engine::decode`] of those latents with the standard decoder (FP32 at every
//!   tier): waveform SNR and relative L2.
//!
//! The reference is the released checkpoint in F32 (`f32`, the CPU unless
//! `YUE2_QUALITY_REFERENCE_DEVICE=cuda`); its outputs are written to `YUE2_QUALITY_OUT` (default
//! `~/.cache/sceneworks-yue2-fixtures/quality` — they are derived from CC BY-NC 4.0 weights, so
//! never inside the repository) and reused by later runs. An F32 reference is itself checked
//! against the pinned upstream's F32 logits in the committed fixture (top-k values), so a reference
//! computed on another device is shown to be the same reference.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/hub YUE2_QUALITY_CONFIGS=f32,q8,q4 \
//!   cargo test --release -p candle-audio-yue2 --lib quality:: -- --ignored --nocapture
//! ```
//!
//! Configurations (`YUE2_QUALITY_CONFIGS`, comma-separated, run in order, each engine dropped before
//! the next loads): `f32` (the reference), `f32dev` (F32 on `YUE2_QUALITY_DEVICE`, compared with the
//! reference: the device's own F32 parity), `q8`, `q4` (CPU F32 compute, or CUDA BF16 with
//! `YUE2_QUALITY_DEVICE=cuda`), `bf16` and `fp8` (CUDA only). Derived tiers are read from, or
//! written to, `YUE2_TIER_DIR/<tier>` (default `YUE2_QUALITY_OUT/tiers`). Results are written as
//! `quality-<config>-<device>.json` beside the reference.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_audio::candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};

use crate::engine::{EngineHooks, EngineOptions, ModelPrecision, SemanticResult, Yue2Engine};
use crate::fp8::ArPrecision;
use crate::inventory::{self, ComponentId, VaeVariant};
use crate::plan::{PlanStep, SymbolicPlan};
use crate::precision::Tier;
use crate::protocol::{GenerationConfig, Sampling, SongRequest};
use crate::sampling::Phase;
use crate::snapshot::SnapshotDirs;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn hub() -> SnapshotDirs {
    let hub = PathBuf::from(env("YUE2_HF_HUB").expect("YUE2_HF_HUB (a hub holding the pins)"));
    inventory::REPOS
        .iter()
        .fold(SnapshotDirs::new(), |dirs, repo| {
            let dir = hub
                .join(format!("models--{}", repo.id.replace('/', "--")))
                .join("snapshots")
                .join(repo.revision);
            dirs.with(repo.id, dir)
        })
}

fn out_dir() -> PathBuf {
    std::env::var_os("YUE2_QUALITY_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/sceneworks-yue2-fixtures/quality")
        })
}

fn device_named(name: &str) -> Device {
    match name {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0).expect("a CUDA device"),
        other => panic!("unknown device {other} (cpu | cuda)"),
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

/// One teacher-forced sequence: `prefix` then `tokens`, one logits row per token.
struct Sequence {
    name: String,
    phase: Phase,
    prefix: Vec<u32>,
    tokens: Vec<u32>,
    /// The upstream F32 reference's per-step records (`top_ids` / `top_values`), for checking an
    /// F32 reference.
    upstream: Vec<Value>,
}

fn sequences(fixture: &Value) -> Vec<Sequence> {
    let mut out = Vec::new();
    for (mode, rec) in fixture["modes"].as_object().unwrap() {
        if let Some(abc) = rec.get("abc").filter(|a| !a.is_null()) {
            out.push(Sequence {
                name: format!("{mode}/abc"),
                phase: Phase::Abc,
                prefix: u32s(&rec["planner_prefix"]),
                tokens: u32s(&abc["emitted"]),
                upstream: abc["steps"].as_array().unwrap().clone(),
            });
        }
        let sem = &rec["semantic"];
        // A guided decode's observed rows are CFG mixes; the conditional row is what a model's
        // fidelity is, so only unguided semantic decodes carry upstream rows to check against.
        let guided = !rec["negative_prefix"].is_null();
        out.push(Sequence {
            name: format!("{mode}/semantic"),
            phase: Phase::Semantic,
            prefix: u32s(&rec["semantic_prefix"]),
            tokens: u32s(&sem["emitted"]),
            upstream: if guided {
                Vec::new()
            } else {
                sem["steps"].as_array().unwrap().clone()
            },
        });
    }
    out
}

/// Teacher-forced rows (phase-allowed ids only, F32) of `seq` on the engine's AR model.
fn teacher_forced(engine: &Yue2Engine, seq: &Sequence) -> Vec<Vec<f32>> {
    let nar = engine.lock_nar_for_ar().unwrap();
    let lm = nar.lm();
    let mut cache = lm.new_cache(seq.prefix.len() + seq.tokens.len()).unwrap();
    let allowed: Vec<usize> = (0..lm.config().vocab_size as u32)
        .filter(|&i| seq.phase.allows(i))
        .map(|i| i as usize)
        .collect();
    let row = |t: Tensor| -> Vec<f32> {
        let full: Vec<f32> = t.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        allowed.iter().map(|&i| full[i]).collect()
    };
    let mut rows = vec![row(lm.prefill(&seq.prefix, &mut cache, || Ok(())).unwrap())];
    for &t in &seq.tokens[..seq.tokens.len() - 1] {
        rows.push(row(lm.decode(t, &mut cache).unwrap()));
    }
    rows
}

/// The acoustic cases: `(name, codes, ode steps)`.
fn nar_cases() -> Vec<(String, Vec<u32>, i64)> {
    let meta: Value =
        serde_json::from_str(include_str!("../tests/fixtures/nar_real_reference.json")).unwrap();
    meta["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["name"].as_str().unwrap().to_string(),
                u32s(&c["codes"]),
                c["steps"].as_i64().unwrap(),
            )
        })
        .collect()
}

/// Everything one configuration produced.
#[derive(Default)]
struct Outputs {
    ar: BTreeMap<String, Vec<Vec<f32>>>,
    latents: BTreeMap<String, Vec<f32>>,
    audio: BTreeMap<String, Vec<f32>>,
    seconds: BTreeMap<String, f64>,
}

fn run(engine: &Yue2Engine, fixture: &Value) -> Outputs {
    let mut o = Outputs::default();
    let t = Instant::now();
    for seq in sequences(fixture) {
        o.ar.insert(seq.name.clone(), teacher_forced(engine, &seq));
    }
    o.seconds.insert("ar".into(), t.elapsed().as_secs_f64());
    let supplied = &fixture["modes"]["supplied_full"];
    let request = SongRequest::from_json(&supplied["request"]).unwrap();
    let plan = match SymbolicPlan::prepare(request, engine.tokenizer()).unwrap() {
        PlanStep::Ready(p) => p,
        PlanStep::GenerateAbc(_) => panic!("supplied_full supplies its score"),
    };
    assert_eq!(plan.prefix(), &u32s(&supplied["semantic_prefix"])[..]);
    let mut hooks = EngineHooks {
        cancelled: &|| false,
        observer: &mut (),
    };
    for (name, codes, steps) in nar_cases() {
        let generation =
            GenerationConfig::new(Sampling::abc_default(), Sampling::semantic_default(), steps)
                .unwrap();
        let semantic = SemanticResult {
            plan: plan.clone(),
            codes,
            truncated: false,
            timing: Default::default(),
        };
        let t = Instant::now();
        let synthesis = engine
            .synthesize(&semantic, &generation, &mut hooks)
            .unwrap();
        o.seconds
            .insert(format!("nar/{name}"), t.elapsed().as_secs_f64());
        let t = Instant::now();
        let audio = engine
            .decode(&synthesis.latents, VaeVariant::Standard, &mut hooks)
            .unwrap();
        o.seconds
            .insert(format!("decode/{name}"), t.elapsed().as_secs_f64());
        o.latents
            .insert(name.clone(), synthesis.latents.values().to_vec());
        o.audio.insert(name, audio.samples().to_vec());
    }
    o
}

fn save(o: &Outputs, path: &std::path::Path) {
    let mut t = std::collections::HashMap::new();
    for (k, rows) in &o.ar {
        let (n, w) = (rows.len(), rows[0].len());
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        t.insert(
            format!("ar/{k}"),
            Tensor::from_vec(flat, (n, w), &Device::Cpu).unwrap(),
        );
    }
    for (prefix, map) in [("latents", &o.latents), ("audio", &o.audio)] {
        for (k, v) in map {
            t.insert(
                format!("{prefix}/{k}"),
                Tensor::from_vec(v.clone(), v.len(), &Device::Cpu).unwrap(),
            );
        }
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    candle_audio::candle_core::safetensors::save(&t, path).unwrap();
}

fn load_reference(path: &std::path::Path) -> Outputs {
    let t = candle_audio::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    let mut o = Outputs::default();
    for (k, v) in t {
        if let Some(name) = k.strip_prefix("ar/") {
            o.ar.insert(name.to_string(), v.to_vec2().unwrap());
        } else if let Some(name) = k.strip_prefix("latents/") {
            o.latents.insert(name.to_string(), v.to_vec1().unwrap());
        } else if let Some(name) = k.strip_prefix("audio/") {
            o.audio.insert(name.to_string(), v.to_vec1().unwrap());
        }
    }
    o
}

/// Per-step AR fidelity of `got` against `want` rows.
#[derive(Clone, Copy, Debug, Default)]
struct ArStats {
    steps: usize,
    top1_agree: usize,
    top8_overlap: f64,
    kl_sum: f64,
    kl_max: f64,
    rel_l2_max: f64,
    max_abs: f64,
}

fn softmax_log(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
    let lse = m + row.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
    row.iter().map(|&x| x as f64 - lse).collect()
}

fn topk(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

fn ar_stats(got: &[Vec<f32>], want: &[Vec<f32>]) -> ArStats {
    let mut s = ArStats::default();
    for (g, w) in got.iter().zip(want) {
        s.steps += 1;
        let (tg, tw) = (topk(g, 8), topk(w, 8));
        s.top1_agree += usize::from(tg[0] == tw[0]);
        s.top8_overlap += tg.iter().filter(|i| tw.contains(i)).count() as f64 / 8.0;
        let (lg, lw) = (softmax_log(g), softmax_log(w));
        let kl: f64 = lw
            .iter()
            .zip(&lg)
            .map(|(a, b)| a.exp() * (a - b))
            .sum::<f64>()
            .max(0.0);
        s.kl_sum += kl;
        s.kl_max = s.kl_max.max(kl);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in g.iter().zip(w) {
            let d = (*a - *b) as f64;
            num += d * d;
            den += (*b as f64).powi(2);
            s.max_abs = s.max_abs.max(d.abs());
        }
        s.rel_l2_max = s.rel_l2_max.max((num / den).sqrt());
    }
    s
}

fn signal_stats(got: &[f32], want: &[f32]) -> Value {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den, mut dot, mut gg) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output");
        let d = (g - w) as f64;
        num += d * d;
        den += (w as f64).powi(2);
        dot += g as f64 * w as f64;
        gg += (g as f64).powi(2);
    }
    json!({
        "rel_l2": (num / den).sqrt(),
        "snr_db": 10.0 * (den / num.max(1e-300)).log10(),
        "cosine": dot / (gg.sqrt() * den.sqrt()).max(1e-300),
    })
}

/// An F32 reference must be the upstream F32 reference: the top-1 id of every unguided step and
/// the top-8 logit values within 1e-3 of the committed upstream values (the CPU-vs-torch parity
/// test holds them to 2e-4; this only has to show another device computes the same reference).
fn check_reference_against_upstream(o: &Outputs, fixture: &Value) -> f64 {
    let mut worst = 0f64;
    for seq in sequences(fixture) {
        let allowed: Vec<u32> = (0..crate::protocol::VOCAB_SIZE)
            .filter(|&i| seq.phase.allows(i))
            .collect();
        for (row, up) in o.ar[&seq.name].iter().zip(&seq.upstream) {
            let want_ids = u32s(&up["top_ids"]);
            let want_vals: Vec<f64> = up["top_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let top = topk(row, want_ids.len());
            assert_eq!(
                allowed[top[0]], want_ids[0],
                "{}: top-1 vs upstream",
                seq.name
            );
            for (i, v) in top.iter().zip(&want_vals) {
                worst = worst.max((row[*i] as f64 - v).abs());
            }
        }
    }
    assert!(
        worst < 1e-3,
        "F32 reference vs upstream top-8 values: {worst:e}"
    );
    worst
}

fn config(name: &str, device: &Device) -> (ModelPrecision, DType) {
    let accel = !device.is_cpu();
    let dtype = if accel { DType::BF16 } else { DType::F32 };
    match name {
        "f32" | "f32dev" => (ModelPrecision::default(), DType::F32),
        "bf16" => {
            assert!(
                accel,
                "bf16 compute needs an accelerator (Candle CPU has no BF16 matmul)"
            );
            (
                ModelPrecision {
                    tier: Some(Tier::Bf16),
                    ar: ArPrecision::Native,
                },
                DType::BF16,
            )
        }
        "fp8" => (
            ModelPrecision {
                tier: Some(Tier::Bf16),
                ar: ArPrecision::Fp8,
            },
            DType::BF16,
        ),
        "q8" | "q4" => (
            ModelPrecision {
                tier: Tier::parse(name),
                ar: ArPrecision::Native,
            },
            dtype,
        ),
        other => panic!("unknown configuration {other}"),
    }
}

#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn tier_quality_against_the_f32_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/ar_real_weights.json")).unwrap();
    let hub = hub();
    let out = out_dir();
    let tier_dir = env("YUE2_TIER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| out.join("tiers"));
    let device_name = env("YUE2_QUALITY_DEVICE").unwrap_or_else(|| "cpu".into());
    let ref_device_name = env("YUE2_QUALITY_REFERENCE_DEVICE").unwrap_or_else(|| "cpu".into());
    let reference_path = out.join(format!("reference-f32-{ref_device_name}.safetensors"));
    let configs = env("YUE2_QUALITY_CONFIGS").unwrap_or_else(|| "f32,q8,q4".into());
    let mut reference: Option<Outputs> = reference_path
        .is_file()
        .then(|| load_reference(&reference_path));
    for name in configs.split(',') {
        let device = device_named(if name == "f32" {
            &ref_device_name
        } else {
            &device_name
        });
        let (precision, dtype) = config(name, &device);
        let mut dirs = hub.clone();
        if let Some(tier @ (Tier::Q8 | Tier::Q4)) = precision.tier {
            let dir = tier_dir.join(tier.name());
            if !dir.exists() {
                let t = Instant::now();
                crate::tier::convert(&hub, tier, &dir).unwrap();
                println!("derived {tier} in {:.1} s", t.elapsed().as_secs_f64());
            }
            let manifest = crate::tier::read_manifest(&dir).unwrap();
            println!(
                "{tier} tier: {} = {}",
                crate::tier::WEIGHTS_FILE,
                manifest["files"][crate::tier::WEIGHTS_FILE]
            );
            dirs = dirs.with(ComponentId::Lm.component().repo.id, dir);
        }
        let t = Instant::now();
        let engine = Yue2Engine::load_with_precision(
            &dirs,
            dtype,
            &device,
            precision,
            GenerationConfig::default(),
            EngineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let load_seconds = t.elapsed().as_secs_f64();
        let residency = engine.weight_residency().unwrap();
        let fp8 = engine.fp8_status().unwrap();
        println!(
            "{name} on {device_name}: loaded in {load_seconds:.1} s; weights {:.2} GB on the \
             device + {:.2} GB host originals",
            residency.device_bytes as f64 / 1e9,
            residency.host_bytes as f64 / 1e9
        );
        let outputs = run(&engine, &fixture);
        drop(engine);
        if name == "f32" {
            let worst = check_reference_against_upstream(&outputs, &fixture);
            println!("f32 reference on {ref_device_name}: top-8 values vs upstream ≤ {worst:e}");
            save(&outputs, &reference_path);
            reference = Some(outputs);
            continue;
        }
        let r = reference
            .as_ref()
            .expect("run the f32 reference first (or keep its saved outputs)");
        let mut total = ArStats::default();
        let mut per = serde_json::Map::new();
        for (k, rows) in &outputs.ar {
            let s = ar_stats(rows, &r.ar[k]);
            per.insert(
                k.clone(),
                json!({
                    "steps": s.steps,
                    "top1_agreement": s.top1_agree as f64 / s.steps as f64,
                    "top8_overlap": s.top8_overlap / s.steps as f64,
                    "kl_mean": s.kl_sum / s.steps as f64,
                    "kl_max": s.kl_max,
                    "logit_rel_l2_max": s.rel_l2_max,
                }),
            );
            total.steps += s.steps;
            total.top1_agree += s.top1_agree;
            total.top8_overlap += s.top8_overlap;
            total.kl_sum += s.kl_sum;
            total.kl_max = total.kl_max.max(s.kl_max);
            total.rel_l2_max = total.rel_l2_max.max(s.rel_l2_max);
            total.max_abs = total.max_abs.max(s.max_abs);
        }
        let mut nar = serde_json::Map::new();
        for (k, v) in &outputs.latents {
            nar.insert(
                k.clone(),
                json!({
                    "latents": signal_stats(v, &r.latents[k]),
                    "audio": signal_stats(&outputs.audio[k], &r.audio[k]),
                }),
            );
        }
        let result = json!({
            "config": name,
            "device": device_name,
            "reference": format!("f32 on {ref_device_name}"),
            "load_seconds": load_seconds,
            "seconds": outputs.seconds,
            "residency": {"device_bytes": residency.device_bytes, "host_bytes": residency.host_bytes},
            "fp8": fp8.to_json(),
            "ar": {
                "steps": total.steps,
                "top1_agreement": total.top1_agree as f64 / total.steps as f64,
                "top8_overlap": total.top8_overlap / total.steps as f64,
                "kl_mean": total.kl_sum / total.steps as f64,
                "kl_max": total.kl_max,
                "logit_rel_l2_max": total.rel_l2_max,
                "logit_max_abs": total.max_abs,
                "per_sequence": per,
            },
            "acoustic": nar,
        });
        let summary: serde_json::Map<String, Value> = [
            "top1_agreement",
            "top8_overlap",
            "kl_mean",
            "kl_max",
            "logit_rel_l2_max",
        ]
        .iter()
        .map(|k| (k.to_string(), result["ar"][k].clone()))
        .collect();
        println!(
            "QUALITY {name} {device_name}: ar {} acoustic {}",
            Value::Object(summary),
            result["acoustic"]
        );
        std::fs::write(
            out.join(format!("quality-{name}-{device_name}.json")),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        // Sanity bounds only — the measured numbers are the evidence (see the crate docs). Every
        // output finite (`signal_stats` asserts it), and no configuration so broken that it
        // disagrees with the reference on most steps.
        assert!(
            total.top1_agree * 2 > total.steps,
            "{name}: top-1 agreement {} of {}",
            total.top1_agree,
            total.steps
        );
    }
}
