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
//! `YUE2_QUALITY_REFERENCE_DEVICE` names another device); its outputs are written to
//! `YUE2_QUALITY_OUT` (default `~/.cache/sceneworks-yue2-fixtures/quality` — they are derived from
//! CC BY-NC 4.0 weights, so never inside the repository) and reused by later runs. An F32
//! reference is itself checked against the pinned upstream's F32 logits in the committed fixture
//! (top-k values), so a reference computed on another device is shown to be the same reference.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/hub YUE2_QUALITY_CONFIGS=f32,q8,q4 \
//!   cargo test --release -p candle-audio-yue2 --lib quality:: -- --ignored --nocapture
//! ```
//!
//! Devices (`YUE2_QUALITY_DEVICE`, `YUE2_QUALITY_REFERENCE_DEVICE`): `cpu` (the default), `cuda`
//! (a `--features cuda` build) and `metal` (a `--features metal` build; the process-wide Metal
//! device the registered provider loads on, [`candle_audio::default_device`]).
//!
//! Configurations (`YUE2_QUALITY_CONFIGS`, comma-separated, run in order, each engine dropped before
//! the next loads): `f32` (the reference), `f32dev` (F32 on `YUE2_QUALITY_DEVICE`, compared with the
//! reference: the device's own F32 parity), `q8`, `q4` (CPU F32 compute, or BF16 on CUDA / Metal),
//! `bf16` (CUDA or Metal: Candle's CPU has no BF16 matmul) and `fp8` (CUDA only — upstream's
//! `torch._scaled_mm`, the cuBLASLt E4M3 GEMM; refused on CPU and Metal). Every configuration is
//! checked against its device before anything loads, so a refused one fails the run at once rather
//! than after the configurations before it. Derived tiers are read from, or written to,
//! `YUE2_TIER_DIR/<tier>` (default `YUE2_QUALITY_OUT/tiers`). Results are written as
//! `quality-<config>-<device>.json` beside the reference, each recording the backend, device,
//! compute dtype and random streams (epic E9: the acoustic stage's noise is the request seed's own
//! stream, so a result reproduces on the same backend and dtype only), and each configuration
//! prints one [`crate::evidence`] summary line.
//!
//! On the owner's Mac (sc-23002; the F32 CPU reference already saved in `YUE2_QUALITY_OUT`):
//!
//! ```text
//! YUE2_HF_HUB=/path/to/hub YUE2_QUALITY_DEVICE=metal YUE2_QUALITY_CONFIGS=f32dev,bf16,q8,q4 \
//!   cargo test --release -p candle-audio-yue2 --features metal --lib \
//!   quality::tier_quality_against_the_f32_reference -- --ignored --exact --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_audio::candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};

use crate::engine::{
    EngineHooks, EngineOptions, ModelPrecision, SemanticResult, SongSettings, Yue2Engine,
};
use crate::evidence;
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

/// A device a configuration runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QualityDevice {
    Cpu,
    Cuda,
    Metal,
}

impl QualityDevice {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            other => Err(format!("unknown device {other} (cpu | cuda | metal)")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }

    fn open(self) -> Device {
        match self {
            Self::Cpu => Device::Cpu,
            Self::Cuda => Device::new_cuda(0).expect("a CUDA device"),
            Self::Metal if !cfg!(feature = "metal") => {
                panic!("the metal device needs a `--features metal` build")
            }
            Self::Metal => {
                // The process-wide instance the registered provider loads on (a second
                // `Device::new_metal(0)` would be a different, non-equal device).
                let device = candle_audio::default_device().expect("a Metal device");
                assert!(device.is_metal(), "{device:?} is not Metal");
                device
            }
        }
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
    /// The acoustic input's truncation flags: the supplied plan's score and the fixture's semantic
    /// codes (passed as an untruncated [`SemanticResult`]).
    truncated: Value,
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
    o.truncated = json!({"abc": plan.truncated(), "semantic": false});
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

/// The precision and compute dtype configuration `name` loads on `device`, or why `device` cannot
/// run it.
fn config(name: &str, device: QualityDevice) -> Result<(ModelPrecision, DType), String> {
    let accel = device != QualityDevice::Cpu;
    let dtype = if accel { DType::BF16 } else { DType::F32 };
    match name {
        "f32" | "f32dev" => Ok((ModelPrecision::default(), DType::F32)),
        "bf16" if !accel => {
            Err("bf16 compute needs an accelerator (Candle CPU has no BF16 matmul)".into())
        }
        "bf16" => Ok((
            ModelPrecision {
                tier: Some(Tier::Bf16),
                ar: ArPrecision::Native,
            },
            DType::BF16,
        )),
        "fp8" if device != QualityDevice::Cuda => Err(format!(
            "fp8 is CUDA only (the cuBLASLt E4M3 GEMM, compute capability >= 8.9), not {}",
            device.name()
        )),
        "fp8" => Ok((
            ModelPrecision {
                tier: Some(Tier::Bf16),
                ar: ArPrecision::Fp8,
            },
            DType::BF16,
        )),
        "q8" | "q4" => Ok((
            ModelPrecision {
                tier: Tier::parse(name),
                ar: ArPrecision::Native,
            },
            dtype,
        )),
        other => Err(format!(
            "unknown configuration {other} (f32 | f32dev | bf16 | fp8 | q8 | q4)"
        )),
    }
}

/// `YUE2_QUALITY_CONFIGS` resolved against the devices, every entry checked before anything loads:
/// `(config, device, precision, compute dtype)`. `f32` runs on the reference device, every other
/// configuration on `device`.
fn plan_configs(
    configs: &str,
    device: &str,
    reference_device: &str,
) -> Result<Vec<(String, QualityDevice, ModelPrecision, DType)>, String> {
    let device = QualityDevice::parse(device)?;
    let reference_device = QualityDevice::parse(reference_device)?;
    configs
        .split(',')
        .map(|name| {
            let on = if name == "f32" {
                reference_device
            } else {
                device
            };
            let (precision, dtype) =
                config(name, on).map_err(|e| format!("{name} on {}: {e}", on.name()))?;
            Ok((name.to_string(), on, precision, dtype))
        })
        .collect()
}

/// What the loaded engine declares it runs with — its effective configuration for the supplied
/// request: backend, device, compute dtype, weight tier, AR mode and the random streams (epic E9).
fn engine_record(engine: &Yue2Engine, fixture: &Value) -> Value {
    let request = SongRequest::from_json(&fixture["modes"]["supplied_full"]["request"]).unwrap();
    let settings = SongSettings {
        generation: GenerationConfig::default(),
        decoder: VaeVariant::Standard,
        options: None,
    };
    let c = engine.effective_config(&request, &settings);
    json!({
        "backend": c["backend"],
        "device": c["device"],
        "dtype": c["model_dtype"],
        "weight_tier": c["weight_tier"],
        "quantization": c["quantization"],
        "rng": {"token_rng": c["token_rng"], "noise": c["noise"], "seed": request.seed()},
    })
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
    // The `bf16` run's latents, when it ran: the FP8 mode's acoustic stage must reproduce them bit
    // for bit (it runs on the restored BF16 originals).
    let mut bf16_latents: Option<BTreeMap<String, Vec<f32>>> = None;
    let planned = plan_configs(&configs, &device_name, &ref_device_name)
        .unwrap_or_else(|e| panic!("YUE2_QUALITY_CONFIGS: {e}"));
    for (name, on, precision, dtype) in planned {
        let name = name.as_str();
        let device = on.open();
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
        let record = engine_record(&engine, &fixture);
        let bounds = bounds_record(name, on);
        let on = on.name();
        println!(
            "{name} on {on}: loaded in {load_seconds:.1} s; weights {:.2} GB on the \
             device + {:.2} GB host originals",
            residency.device_bytes as f64 / 1e9,
            residency.host_bytes as f64 / 1e9
        );
        let outputs = run(&engine, &fixture);
        drop(engine);
        let mut seconds = serde_json::Map::new();
        seconds.insert("load".into(), json!(load_seconds));
        for (k, v) in &outputs.seconds {
            seconds.insert(k.clone(), json!(v));
        }
        println!(
            "{}",
            evidence::summary_line(json!({
                "test": "quality::tier_quality_against_the_f32_reference",
                "config": name,
                "backend": record["backend"],
                "device": record["device"],
                "dtype": record["dtype"],
                "weight_tier": record["weight_tier"],
                "quantization": record["quantization"],
                "rng": record["rng"],
                "seconds": seconds,
                "peak_rss_bytes": evidence::peak_rss_bytes(),
                "residency": {
                    "device_bytes": residency.device_bytes,
                    "host_bytes": residency.host_bytes,
                },
                "outputs": outputs
                    .audio
                    .iter()
                    .map(|(k, v)| (k.clone(), evidence::audio_record(v)))
                    .collect::<serde_json::Map<_, _>>(),
                "truncated": outputs.truncated,
                "bounds": bounds,
            }))
        );
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
            "backend": record["backend"],
            "dtype": record["dtype"],
            "rng": record["rng"],
            "reference": format!("f32 on {ref_device_name}"),
            "bounds": bounds,
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
        // The bounds: see `BOUNDS`. Every output is finite (`signal_stats` asserts it).
        let (_, top1_min, kl_mean_max, snr_min, _) = BOUNDS
            .iter()
            .copied()
            .find(|b| b.0 == name)
            .expect("every configuration has bounds");
        let top1 = total.top1_agree as f64 / total.steps as f64;
        let kl_mean = total.kl_sum / total.steps as f64;
        assert!(
            top1 >= top1_min,
            "{name}: top-1 agreement {top1} < {top1_min}"
        );
        assert!(
            kl_mean <= kl_mean_max,
            "{name}: mean KL {kl_mean} > {kl_mean_max}"
        );
        for (case, stats) in &nar {
            let snr = stats["latents"]["snr_db"].as_f64().unwrap();
            assert!(
                snr >= snr_min,
                "{name} {case}: latents SNR {snr} dB < {snr_min}"
            );
        }
        match name {
            "bf16" => bf16_latents = Some(outputs.latents.clone()),
            "fp8" => {
                if let Some(bf16) = &bf16_latents {
                    assert_eq!(
                        &outputs.latents, bf16,
                        "the FP8 mode's acoustic stage must run the restored BF16 originals"
                    );
                    println!("fp8: acoustic latents are bit-identical to bf16's");
                }
            }
            _ => {}
        }
    }
}

/// `(config, top-1 agreement ≥, mean KL ≤, latents SNR ≥ dB, backends the bound was measured on)`
/// against the F32 reference.
///
/// Measured 2026-09-26 — Candle CPU (Apple M-series, F32 activations): q8 99.5 % / 1.5e-4 /
/// 37.0 dB, q4 93.5 % / 9.0e-3 / 18.3 dB; Candle CUDA (RTX PRO 6000 Blackwell, BF16 activations,
/// run 36255139284): f32dev 100 % / 1.3e-12 / 119.5 dB, bf16 98.4 % / 3.1e-4 / 37.1 dB, fp8 97.8 %
/// / 1.9e-3 / 37.1 dB, q8 98.9 % / 4.8e-4 / 34.1 dB, q4 94.0 % / 9.1e-3 / 17.9 dB (the worst of the
/// two acoustic cases). Bounds leave ≈3× on KL and ≈3–5 dB on SNR for another machine's
/// reduction order; each is far outside the next-coarser configuration's measurement (a q8 run
/// that loaded q4 weights, an FP8 mode left active in the acoustic stage — the bit-identity check
/// — or a BF16 run on the wrong weights fails).
///
/// **Metal reuses these CPU/CUDA-derived bounds and is unmeasured until the first Metal run**
/// (sc-23002): a Metal pass means "within the CPU/CUDA envelope", not "Metal-calibrated". Every
/// result records the bounds it was judged against and where they were measured
/// ([`bounds_record`]).
const BOUNDS: [(&str, f64, f64, f64, &[&str]); 5] = [
    ("f32dev", 0.999, 1e-8, 90.0, &["cuda"]),
    ("bf16", 0.95, 1e-3, 32.0, &["cuda"]),
    ("fp8", 0.93, 6e-3, 32.0, &["cuda"]),
    ("q8", 0.95, 1.5e-3, 30.0, &["cpu", "cuda"]),
    ("q4", 0.85, 3e-2, 14.0, &["cpu", "cuda"]),
];

/// The bounds configuration `name` is judged against on `device`, with their provenance:
/// `measured_on` (the backends the [`BOUNDS`] were derived from) and whether `device` is one of
/// them. `null` for the `f32` reference, which is checked against upstream instead.
fn bounds_record(name: &str, device: QualityDevice) -> Value {
    match BOUNDS.iter().find(|b| b.0 == name) {
        None => Value::Null,
        Some(&(_, top1_min, kl_mean_max, snr_min, measured_on)) => json!({
            "top1_agreement_min": top1_min,
            "kl_mean_max": kl_mean_max,
            "latents_snr_min_db": snr_min,
            "measured_on": measured_on,
            "measured_on_this_device": measured_on.contains(&device.name()),
        }),
    }
}

/// Weights-free: every result names where its bounds were measured, and a Metal result says its
/// bounds were not measured on Metal.
#[test]
fn bounds_carry_their_provenance() {
    use QualityDevice::{Cpu, Cuda, Metal};

    assert_eq!(bounds_record("f32", Cpu), Value::Null);
    for &(name, top1_min, ..) in &BOUNDS {
        let metal = bounds_record(name, Metal);
        assert_eq!(metal["top1_agreement_min"], top1_min, "{name}");
        let on: Vec<&str> = metal["measured_on"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no measured_on"))
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(!on.is_empty(), "{name}");
        assert!(
            on.iter().all(|b| ["cpu", "cuda"].contains(b)),
            "{name}: {on:?}"
        );
        assert_eq!(metal["measured_on_this_device"], false, "{name} on Metal");
    }
    for name in ["q8", "q4"] {
        let record = bounds_record(name, Metal);
        assert_eq!(record["measured_on"], json!(["cpu", "cuda"]), "{name}");
        assert_eq!(bounds_record(name, Cpu)["measured_on_this_device"], true);
    }
    assert_eq!(bounds_record("bf16", Cuda)["measured_on_this_device"], true);
    assert_eq!(bounds_record("bf16", Cpu)["measured_on"], json!(["cuda"]));
}

/// Weights-free (runs on the CPU): the device and configuration parsing the Metal run depends on.
#[test]
fn metal_is_a_quality_device_and_fp8_is_cuda_only() {
    use QualityDevice::{Cpu, Cuda, Metal};

    for (name, device) in [("cpu", Cpu), ("cuda", Cuda), ("metal", Metal)] {
        assert_eq!(QualityDevice::parse(name), Ok(device));
        assert_eq!(device.name(), name);
    }
    assert!(QualityDevice::parse("mps").is_err());

    // Every configuration but FP8 runs on Metal at CUDA's precision: BF16 compute, the tier's
    // weights, the native AR path.
    for name in ["f32", "f32dev", "bf16", "q8", "q4"] {
        assert_eq!(
            config(name, Metal),
            config(name, Cuda),
            "{name} on Metal matches CUDA"
        );
    }
    assert_eq!(
        config("q4", Metal),
        Ok((
            ModelPrecision {
                tier: Some(Tier::Q4),
                ar: ArPrecision::Native,
            },
            DType::BF16,
        ))
    );
    assert_eq!(
        config("f32dev", Metal),
        Ok((ModelPrecision::default(), DType::F32))
    );
    let fp8 = config("fp8", Metal).unwrap_err();
    assert!(fp8.contains("CUDA only") && fp8.contains("metal"), "{fp8}");
    assert!(config("fp8", Cpu).is_err());
    assert_eq!(config("fp8", Cuda).unwrap().0.ar, ArPrecision::Fp8);
    assert!(config("bf16", Cpu).is_err());

    // The whole list is checked before anything loads: one refused entry refuses the run, and
    // `f32` resolves on the reference device.
    let planned = plan_configs("f32dev,bf16,q8,q4", "metal", "cpu").unwrap();
    let devices: Vec<_> = planned.iter().map(|p| (p.0.as_str(), p.1)).collect();
    assert_eq!(
        devices,
        [
            ("f32dev", Metal),
            ("bf16", Metal),
            ("q8", Metal),
            ("q4", Metal)
        ]
    );
    assert_eq!(plan_configs("f32", "metal", "cpu").unwrap()[0].1, Cpu);
    let refused = plan_configs("f32dev,bf16,fp8,q8", "metal", "cpu").unwrap_err();
    assert!(refused.starts_with("fp8 on metal:"), "{refused}");
}

/// Weights-free: a build without the `metal` feature refuses the Metal device by name instead of
/// silently measuring on the CPU (the audio lane's default device in such a build).
#[cfg(not(feature = "metal"))]
#[test]
#[should_panic(expected = "needs a `--features metal` build")]
fn the_metal_device_needs_a_metal_build() {
    QualityDevice::Metal.open();
}
