//! Real-weight parity of the native acoustic stage against the pinned upstream (sc-22992).
//!
//! `#[ignore]`d in ordinary runs (CI has no weights); under `--ignored` a missing `YUE2_HF_HUB` or
//! reference file panics rather than silently passing. Needs:
//!
//! * `YUE2_HF_HUB` — a hub directory holding the pinned `m-a-p/YuE2-3B` revision (see
//!   `scripts/reference/yue2/README.md`);
//! * the upstream reference produced by `scripts/reference/yue2/nar_fixtures.py real` (written
//!   outside the repository because latents computed by CC BY-NC 4.0 weights are derived from them;
//!   its SHA-256 is committed in `tests/fixtures/nar_real_reference.json` and checked here).
//!   Location: `YUE2_NAR_REFERENCE_DIR`, default `~/.cache/sceneworks-yue2-fixtures/nar`.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/huggingface/hub cargo test --release -p candle-audio-yue2 \
//!   --test nar_real_weights -- --ignored --nocapture --test-threads 1
//! ```
//!
//! CPU only, F32 (the BF16 checkpoint upcast; BF16 device tiers are sc-22995). Run cost is in
//! `tests/fixtures/README.md`; never run it beside the upstream `real` generator.
//!
//! Every latent compared here is produced by the production entry points ([`Yue2Nar::load`] from a
//! verified snapshot, then [`synthesize`]) from the reference's own injected noise.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_yue2::inventory;
use candle_audio_yue2::nar::{
    synthesize, Evaluation, NarOptions, QueryTile, SongNoise, SynthesisHooks, SynthesisObserver,
    SynthesisRequest, Yue2Nar,
};
use candle_audio_yue2::SnapshotDirs;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Native vs upstream on YuE2-3B (Candle CPU F32 vs torch 2.10.0 CPU F32): max |Δ| over every
/// evaluation's input state and velocity and the final latents of both cases (202 evaluations: 3 chunks × 64 + 10).
/// Measured 2026-09-26 (Apple M-series CPU): 4.4e-5 (final latents 1.9e-5) on values of magnitude
/// O(1) — reduction-order noise through 28 layers of 2048 channels over up to 32 midpoint steps.
/// The bound is ~11× that; the synthetic mutations (`tests/fixtures/README.md`) move latents by
/// ≥ 7e-3.
const REAL_MAX_ABS: f32 = 5e-4;
/// Relative L2 over the same tensors. Measured 5.9e-6; bound ~10×.
const REAL_REL_L2: f64 = 6e-5;
/// Native `Rows(7)` + AR offload vs native default on the real weights (same process, same model).
/// Bound ≤ 1e-5; measured 0 on macOS CPU (the synthetic equivalent measured 7.2e-7 on the Linux CI
/// CPU, whose GEMM blocking depends on the row count). Values may differ in their last bits, so only
/// the stage identity is compared, not the value hash.
const REAL_TILING_MAX_ABS: f32 = 1e-5;

fn hub() -> SnapshotDirs {
    let hub = PathBuf::from(std::env::var_os("YUE2_HF_HUB").unwrap_or_else(|| {
        panic!(
            "real-weight test run without YUE2_HF_HUB (a hub directory holding the pinned repos)"
        )
    }));
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

fn meta() -> Value {
    serde_json::from_str(include_str!("fixtures/nar_real_reference.json")).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The upstream reference tensors, refused unless their SHA-256 equals the committed record.
fn reference() -> HashMap<String, Tensor> {
    let dir = std::env::var_os("YUE2_NAR_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/sceneworks-yue2-fixtures/nar")
        });
    let meta = meta();
    let file = &meta["reference_file"];
    let path = dir.join(file["name"].as_str().unwrap());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — run scripts/reference/yue2/nar_fixtures.py real first",
            path.display()
        )
    });
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        file["sha256"].as_str().unwrap(),
        "{} is not the committed reference (regenerate it and the JSON together)",
        path.display()
    );
    candle_audio::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct Spread {
    max_abs: f32,
    rel_l2: f64,
}

impl Spread {
    fn of(got: &[f32], want: &[f32]) -> Self {
        assert_eq!(got.len(), want.len());
        let (mut num, mut den, mut max_abs) = (0.0f64, 0.0f64, 0.0f32);
        for (&g, &w) in got.iter().zip(want) {
            let d = g - w;
            assert!(d.is_finite(), "non-finite comparison");
            max_abs = max_abs.max(d.abs());
            num += (d as f64).powi(2);
            den += (w as f64).powi(2);
        }
        Self {
            max_abs,
            rel_l2: num.sqrt() / den.sqrt().max(1e-30),
        }
    }

    fn merge(&mut self, o: Spread) {
        self.max_abs = self.max_abs.max(o.max_abs);
        self.rel_l2 = self.rel_l2.max(o.rel_l2);
    }
}

/// Compares every evaluation with the reference as it happens (nothing large is retained).
struct Compare<'a> {
    reference: &'a HashMap<String, Tensor>,
    case: &'a str,
    index: HashMap<usize, usize>,
    spread: Spread,
    evaluations: usize,
}

impl SynthesisObserver for Compare<'_> {
    fn on_evaluation(&mut self, e: &Evaluation<'_>) {
        let i = self.index.entry(e.chunk).or_insert(0);
        let key = |what: &str| format!("{}/c{}/{what}", self.case, e.chunk);
        let raw: Vec<f64> = self.reference[&key("raw")].to_vec1().unwrap();
        assert_eq!(
            e.raw_t, raw[*i] as f32,
            "{} chunk {} evaluation {i}: raw t",
            self.case, e.chunk
        );
        let input = self.reference[&key("input")].get(*i).unwrap();
        let velocity = self.reference[&key("velocity")].get(*i).unwrap();
        self.spread.merge(Spread::of(&host(e.input), &host(&input)));
        self.spread
            .merge(Spread::of(&host(e.velocity), &host(&velocity)));
        *i += 1;
        self.evaluations += 1;
    }
}

fn never() -> bool {
    false
}

#[test]
#[ignore = "real weights: set YUE2_HF_HUB and generate the reference (see the module docs)"]
fn real_weight_synthesis_matches_upstream_and_memory_controls_are_invariant() {
    let r = reference();
    let meta = meta();
    let start = Instant::now();
    let mut nar = Yue2Nar::load(&hub(), DType::F32, &Device::Cpu).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "YuE2-3B verified + loaded (F32, both paths) in {:.1?}",
        start.elapsed()
    );
    let mut worst = Spread::default();
    for case in meta["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prefix = u32s(&case["prefix"]);
        let codes = u32s(&case["codes"]);
        let steps = case["steps"].as_u64().unwrap() as usize;
        let context = case["context"].as_u64().unwrap() as usize;
        let noise_t = &r[&format!("{name}/noise")];
        let noise = SongNoise::injected(host(noise_t), codes.len()).unwrap();
        let request = SynthesisRequest {
            prefix: &prefix,
            codes: &codes,
            noise: &noise,
            steps,
            context,
        };
        let mut compare = Compare {
            reference: &r,
            case: name,
            index: HashMap::new(),
            spread: Spread::default(),
            evaluations: 0,
        };
        let t0 = Instant::now();
        let out = synthesize(
            &mut nar,
            &request,
            &NarOptions::default(),
            SynthesisHooks {
                cancelled: &never,
                observer: &mut compare,
            },
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let seconds = t0.elapsed().as_secs_f64();
        let chunks: Vec<(usize, usize)> = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c[0].as_u64().unwrap() as usize,
                    c[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(out.chunks, chunks, "{name}: chunks");
        assert_eq!(compare.evaluations, chunks.len() * 2 * steps);
        let fin = Spread::of(out.latents.values(), &host(&r[&format!("{name}/final")]));
        let mut spread = compare.spread;
        spread.merge(fin);
        println!(
            "{name}: {} chunks × {steps} steps in {seconds:.1} s; evaluations+final max |Δ| {:.3e} \
             rel L2 {:.3e}; final max |Δ| {:.3e} rel L2 {:.3e}",
            chunks.len(),
            spread.max_abs,
            spread.rel_l2,
            fin.max_abs,
            fin.rel_l2
        );
        worst.merge(spread);

        if name == "single_chunk_5" {
            // The memory controls on the real weights: tiny query tiles plus AR offload.
            let options = NarOptions {
                query_tile: QueryTile::Rows(7),
                offload_ar: true,
            };
            let hooks = SynthesisHooks {
                cancelled: &never,
                observer: &mut (),
            };
            let tiled = synthesize(&mut nar, &request, &options, hooks).unwrap();
            let d = Spread::of(tiled.latents.values(), out.latents.values());
            println!(
                "{name}: Rows(7) + offload vs default: max |Δ| {:.3e}",
                d.max_abs
            );
            assert!(
                d.max_abs <= REAL_TILING_MAX_ABS,
                "tiling changed the latents: {d:?}"
            );
            assert_eq!(
                tiled.latents.identity().source,
                out.latents.identity().source
            );
            assert!(!nar.lm().ar_offloaded());
        }
    }
    assert!(
        worst.max_abs <= REAL_MAX_ABS && worst.rel_l2 <= REAL_REL_L2,
        "real weights: max |Δ| {} (bound {REAL_MAX_ABS}), rel L2 {} (bound {REAL_REL_L2})",
        worst.max_abs,
        worst.rel_l2
    );
}
