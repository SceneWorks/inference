//! CI parity and property tests of the acoustic stage on the synthetic MoT.
//!
//! Every latent compared with the upstream reference is produced by the production path
//! ([`synthesize`] / [`solve_chunk`], and for the joint-forward check the production
//! `ChunkSolver::velocity`) — never a test-only forward. The reference is
//! `tests/fixtures/nar_synthetic.{json,safetensors}`, written by
//! `scripts/reference/yue2/nar_fixtures.py synthetic` from the pinned upstream `yue2.nar` with the
//! same injected noise (see `tests/fixtures/README.md`).

use std::cell::Cell;
use std::collections::HashMap;

use super::*;

/// Native vs upstream on the synthetic model (Candle CPU F32 vs torch 2.10.0 CPU F32), max |Δ|
/// over every evaluation's input state, every velocity and the final latents of every case.
/// Measured 2026-09-26: 2.1e-6 (cases), 2.1e-6 (joint/cached velocities), 2.0e-6 (`cond_end`), on
/// values of magnitude O(1) (N(0, 1) noise) — a few F32 ulps of reduction-order noise, which the
/// ODE does not amplify at this width. The bound is ~10× that; every structural defect it guards
/// (Euler instead of midpoint, a shifted timestep, an off-by-one chunk, a dropped key, an unclamped
/// or shifted position) moves values by ≥ 1e-3 (see `tests/fixtures/README.md`).
const SYNTH_MAX_ABS: f32 = 2e-5;
/// Relative L2 (‖Δ‖ / ‖ref‖) over the same tensors — scale-sensitive, unlike cosine similarity.
/// Measured 4.8e-7; bound ~10×.
const SYNTH_REL_L2: f64 = 5e-6;

fn meta() -> Value {
    serde_json::from_str(include_str!("../../tests/fixtures/nar_synthetic.json")).unwrap()
}

fn reference() -> HashMap<String, Tensor> {
    candle_audio::candle_core::safetensors::load_buffer(
        include_bytes!("../../tests/fixtures/nar_synthetic.safetensors"),
        &Device::Cpu,
    )
    .unwrap()
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

/// The largest deviations of native from reference values.
#[derive(Clone, Copy, Debug, Default)]
struct Spread {
    max_abs: f32,
    rel_l2: f64,
}

impl Spread {
    fn of(got: &[f32], want: &[f32]) -> Self {
        assert_eq!(got.len(), want.len(), "compared tensors differ in size");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut max_abs = 0.0f32;
        for (&g, &w) in got.iter().zip(want) {
            let d = g - w;
            assert!(d.is_finite(), "non-finite comparison");
            max_abs = max_abs.max(d.abs());
            num += (d as f64) * (d as f64);
            den += (w as f64) * (w as f64);
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

    fn assert_within(&self, max_abs: f32, rel_l2: f64, what: &str) {
        assert!(
            self.max_abs <= max_abs && self.rel_l2 <= rel_l2,
            "{what}: max |Δ| {} (bound {max_abs}), rel L2 {} (bound {rel_l2})",
            self.max_abs,
            self.rel_l2
        );
    }
}

/// One recorded evaluation.
struct Eval {
    chunk: usize,
    step: usize,
    stage: Stage,
    raw_t: f32,
    input: Vec<f32>,
    velocity: Vec<f32>,
}

#[derive(Default)]
struct Trace {
    evals: Vec<Eval>,
    steps: Vec<(usize, usize)>,
    progress: Vec<(usize, usize)>,
}

impl SynthesisObserver for Trace {
    fn on_progress(&mut self, completed: usize, total: usize) {
        self.progress.push((completed, total));
    }
    fn on_evaluation(&mut self, e: &Evaluation<'_>) {
        self.evals.push(Eval {
            chunk: e.chunk,
            step: e.step,
            stage: e.stage,
            raw_t: e.raw_t,
            input: host(e.input),
            velocity: host(e.velocity),
        });
    }
    fn on_step(&mut self, chunk: usize, step: usize, _state: &Tensor) {
        self.steps.push((chunk, step));
    }
}

fn never() -> bool {
    false
}

fn injected(r: &HashMap<String, Tensor>, name: &str) -> SongNoise {
    let t = &r[&format!("{name}/noise")];
    SongNoise::injected(host(t), t.dim(0).unwrap()).unwrap()
}

fn run(
    nar: &mut Yue2Nar,
    prefix: &[u32],
    codes: &[u32],
    noise: &SongNoise,
    steps: usize,
    context: usize,
    options: &NarOptions,
) -> (Synthesis, Trace) {
    let mut trace = Trace::default();
    let request = SynthesisRequest {
        prefix,
        codes,
        noise,
        steps,
        context,
    };
    let out = synthesize(
        nar,
        &request,
        options,
        SynthesisHooks {
            cancelled: &never,
            observer: &mut trace,
        },
    )
    .unwrap();
    (out, trace)
}

/// Compare a native trace with the reference evaluations of `name` (all chunks).
fn compare_evaluations(
    trace: &Trace,
    r: &HashMap<String, Tensor>,
    name: &str,
    chunks: usize,
    steps: usize,
) -> Spread {
    let mut spread = Spread::default();
    assert_eq!(trace.evals.len(), chunks * 2 * steps, "{name}: evaluations");
    for c in 0..chunks {
        let raw: Vec<f64> = r[&format!("{name}/c{c}/raw")].to_vec1().unwrap();
        let inputs = &r[&format!("{name}/c{c}/input")];
        let velocities = &r[&format!("{name}/c{c}/velocity")];
        let native: Vec<&Eval> = trace.evals.iter().filter(|e| e.chunk == c).collect();
        assert_eq!(native.len(), raw.len(), "{name} chunk {c}: evaluations");
        for (i, e) in native.iter().enumerate() {
            assert_eq!(e.step, i / 2);
            let stage = if i % 2 == 0 {
                Stage::First
            } else {
                Stage::Midpoint
            };
            assert_eq!(e.stage, stage);
            // The schedule is exact: the same F64 logit, rounded to the F32 the model sees.
            assert_eq!(
                e.raw_t, raw[i] as f32,
                "{name} chunk {c} evaluation {i}: raw t"
            );
            spread.merge(Spread::of(&e.input, &host(&inputs.get(i).unwrap())));
            spread.merge(Spread::of(&e.velocity, &host(&velocities.get(i).unwrap())));
        }
    }
    spread
}

/// Every recorded upstream synthesis — the default 32-step solver, other step counts, several
/// original chunks, chunk edges (exactly two full chunks; one frame over), a chunk longer than the
/// latent-position table, a shifted timestep — reproduced by [`synthesize`] from the same injected
/// noise: every evaluation's input state and velocity, the exact timestep schedule, the chunks,
/// each chunk's AR sequence, progress, and the final latents.
#[test]
fn synthetic_synthesis_matches_upstream() {
    let meta = meta();
    let r = reference();
    let mut total = Spread::default();
    let mut models: HashMap<u64, Yue2Nar> = HashMap::new();
    let mut compared = 0;
    for case in meta["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if case.get("chunks").is_none() || !r.contains_key(&format!("{name}/c0/raw")) {
            continue; // cond_end (not a song-level synthesis) and the tiled final-only case
        }
        let shift = case["timestep_shift"].as_f64().unwrap();
        let nar = models
            .entry(shift.to_bits())
            .or_insert_with(|| synthetic::model(shift));
        let prefix = u32s(&case["prefix"]);
        let codes = u32s(&case["codes"]);
        let steps = case["steps"].as_u64().unwrap() as usize;
        let context = case["context"].as_u64().unwrap() as usize;
        let noise = injected(&r, name);
        let (out, trace) = run(
            nar,
            &prefix,
            &codes,
            &noise,
            steps,
            context,
            &NarOptions::default(),
        );
        let pairs = |v: &Value| -> Vec<(usize, usize)> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        p[0].as_u64().unwrap() as usize,
                        p[1].as_u64().unwrap() as usize,
                    )
                })
                .collect()
        };
        assert_eq!(out.chunks, pairs(&case["chunks"]), "{name}: chunks");
        for (k, &(a, b)) in out.chunks.iter().enumerate() {
            assert_eq!(
                chunk_ar_tokens(&prefix, &codes[a..b]),
                u32s(&case["ar_tokens"][k]),
                "{name} chunk {k}: AR sequence"
            );
        }
        assert_eq!(trace.progress, pairs(&case["progress"]), "{name}: progress");
        assert_eq!(trace.steps.len(), out.chunks.len() * steps);
        let mut spread = compare_evaluations(&trace, &r, name, out.chunks.len(), steps);
        let fin = Spread::of(out.latents.values(), &host(&r[&format!("{name}/final")]));
        spread.merge(fin);
        assert_eq!(
            out.latents.frames(),
            codes.len(),
            "{name}: every frame solved"
        );
        eprintln!(
            "{name}: evaluations+final max |Δ| {:.3e} rel L2 {:.3e}; final max |Δ| {:.3e}",
            spread.max_abs, spread.rel_l2, fin.max_abs
        );
        spread.assert_within(SYNTH_MAX_ABS, SYNTH_REL_L2, name);
        total.merge(spread);
        compared += 1;
    }
    assert_eq!(compared, 8, "every recorded song-level case compared");
    eprintln!(
        "all synthetic cases: max |Δ| {:.3e}, rel L2 {:.3e}",
        total.max_abs, total.rel_l2
    );
}

/// Tiled vs untiled on the same model: the tile only splits the query rows of an otherwise
/// identical score/softmax/value product, so any drift is reduction-order noise. Bound ≤ 1e-5;
/// measured 0 on macOS CPU and 7.2e-7 on the Linux CI CPU (whose GEMM blocking depends on the row
/// count), while a dropped key tile moves latents by ≥ 1e-2 (see the mutation record). The values
/// may therefore differ in their last bits, so only the **stage** identity — not the value hash —
/// is compared across settings.
const TILING_MAX_ABS: f32 = 1e-5;

/// Query tiling (every setting, including a score-byte budget that admits exactly one row per
/// call) and AR offload leave the latents (within [`TILING_MAX_ABS`]) and the stage identity
/// unchanged, and match the upstream run that tiled by 5 rows with `offload_ar=True`.
#[test]
fn tiling_budget_and_offload_do_not_change_the_latents() {
    let meta = meta();
    let r = reference();
    let case = meta["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "default_32")
        .unwrap()
        .clone();
    let (prefix, codes) = (u32s(&case["prefix"]), u32s(&case["codes"]));
    let noise = injected(&r, "default_32");
    let mut nar = synthetic::model(1.0);
    let base = run(
        &mut nar,
        &prefix,
        &codes,
        &noise,
        32,
        protocol::CONTEXT,
        &NarOptions::default(),
    )
    .0;
    let upstream_tiled = host(&r["tiled_q5_offload/final"]);
    const ONE_ROW: usize = SCORE_TILES_LIVE * 4 * 26 * 4;
    let settings = [
        QueryTile::Whole,
        QueryTile::Rows(1),
        QueryTile::Rows(3),
        QueryTile::Rows(5),
        QueryTile::Rows(7),
        QueryTile::Rows(10_000),
        // 26 keys (18 visible AR + 8 NAR), 4 heads, F32: one row costs 3 · 4 · 26 · 4 bytes.
        QueryTile::ScoreBytes(ONE_ROW),
        QueryTile::ScoreBytes(3 * ONE_ROW + 5),
        QueryTile::ScoreBytes(usize::MAX / 2),
    ];
    for query_tile in settings {
        for offload_ar in [false, true] {
            let options = NarOptions {
                query_tile,
                offload_ar,
            };
            let out = run(
                &mut nar,
                &prefix,
                &codes,
                &noise,
                32,
                protocol::CONTEXT,
                &options,
            )
            .0;
            let d = Spread::of(out.latents.values(), base.latents.values());
            eprintln!("{options:?}: vs default tiling max |Δ| {:.3e}", d.max_abs);
            d.assert_within(TILING_MAX_ABS, 1e-5, &format!("{options:?}"));
            Spread::of(out.latents.values(), &upstream_tiled).assert_within(
                SYNTH_MAX_ABS,
                SYNTH_REL_L2,
                &format!("{options:?} vs upstream tiled"),
            );
            assert_eq!(
                out.latents.identity().source,
                base.latents.identity().source
            );
            assert_eq!(out.latents.frames(), base.latents.frames());
            assert!(!nar.lm().ar_offloaded(), "restored after the synthesis");
        }
    }
}

/// Reference attention in F64 over every key (grouped-query: head `h` reads KV head `h / groups`).
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let (heads, kv_heads, n, keys, d) = dims;
    let groups = heads / kv_heads;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f32; heads * n * d];
    for h in 0..heads {
        let kh = h / groups;
        for i in 0..n {
            let qi = &q[(h * n + i) * d..(h * n + i + 1) * d];
            let scores: Vec<f64> = (0..keys)
                .map(|j| {
                    let kj = &k[(kh * keys + j) * d..(kh * keys + j + 1) * d];
                    qi.iter()
                        .zip(kj)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let z: f64 = w.iter().sum();
            for c in 0..d {
                let acc: f64 = (0..keys)
                    .map(|j| w[j] * v[(kh * keys + j) * d + c] as f64)
                    .sum();
                out[(h * n + i) * d + c] = (acc / z) as f32;
            }
        }
    }
    out
}

/// Every query tile attends the whole key set: `attend` with any row count equals an F64 softmax
/// over all keys, for tiles that divide the rows, leave a remainder, are a single row, or exceed
/// the rows.
#[test]
fn every_query_tile_attends_every_key() {
    let (heads, kv_heads, n, keys, d) = (4, 2, 11, 29, 16);
    let values = |name: &str, len: usize| crate::model::synthetic::values(name, len, 2.0, 0.0);
    let q = values("attend.q", heads * n * d);
    let k = values("attend.k", kv_heads * keys * d);
    let v = values("attend.v", kv_heads * keys * d);
    let want = naive_attention(&q, &k, &v, (heads, kv_heads, n, keys, d));
    let qt = Tensor::from_vec(q, (1, heads, n, d), &Device::Cpu).unwrap();
    let kt = Tensor::from_vec(k, (1, kv_heads, keys, d), &Device::Cpu).unwrap();
    let vt = Tensor::from_vec(v, (1, kv_heads, keys, d), &Device::Cpu).unwrap();
    for rows in [1, 2, 3, 4, 5, 10, 11, 12, 256] {
        let got = attend(&qt, &kt, &vt, 1.0 / (d as f32).sqrt(), rows).unwrap();
        assert_eq!(got.dims(), [1, heads, n, d]);
        // F32 scores/softmax vs F64: a few ulps of O(1) values.
        let s = Spread::of(&host(&got), &want);
        assert!(s.max_abs < 1e-5, "rows {rows}: max |Δ| {}", s.max_abs);
    }
}

/// The cached formulation — the AR prefix prefilled once, the NAR attending its visible keys —
/// equals upstream's **joint** forward (`nar_velocity`: one sequence through both MoT paths under
/// the hybrid mask), for the full prefix and for text-only visibility (`nar_cond_end`), and a
/// cache reused by later evaluations still yields the same velocity (no evaluation sees another's
/// keys).
#[test]
fn cached_velocity_equals_upstream_joint_forward() {
    let meta = meta();
    let r = reference();
    let nar = synthetic::model(1.0);
    let mut worst = Spread::default();
    let joint = meta["joint"].as_array().unwrap();
    assert_eq!(joint.len(), 4);
    for case in joint {
        let name = case["name"].as_str().unwrap();
        let ar_tokens = u32s(&case["ar_tokens"]);
        let state = &r[&format!("{name}/state")];
        let input = ChunkInput {
            ar_tokens: &ar_tokens,
            noise: state,
            nar_cond_end: case["nar_cond_end"].as_u64().unwrap() as usize,
        };
        let mut solver = ChunkSolver::prefill(&nar, &input, QueryTile::Upstream, &never).unwrap();
        let raw = case["raw_t"].as_f64().unwrap() as f32;
        let velocity = |solver: &mut ChunkSolver, x: &Tensor, t: f32, tile| {
            host(&solver.velocity(&nar, x, t, tile).unwrap())
        };
        let first = velocity(&mut solver, state, raw, QueryTile::Upstream);
        // Another evaluation in between rewrites the NAR slots of the cache.
        let other = (state * 0.25).unwrap();
        velocity(&mut solver, &other, -3.0, QueryTile::Rows(2));
        let again = velocity(&mut solver, state, raw, QueryTile::Upstream);
        assert_eq!(first, again, "{name}: cache reuse changed the velocity");
        for key in ["joint", "cached"] {
            let s = Spread::of(&first, &host(&r[&format!("{name}/{key}")]));
            eprintln!("{name} vs upstream {key}: max |Δ| {:.3e}", s.max_abs);
            worst.merge(s);
        }
    }
    worst.assert_within(SYNTH_MAX_ABS, SYNTH_REL_L2, "cached vs joint");
}

/// Upstream `CachedNAR` with `nar_cond_end = 5`: only the first five AR keys are visible.
#[test]
fn text_only_visibility_matches_upstream() {
    let meta = meta();
    let r = reference();
    let case = meta["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "cond_end_5")
        .unwrap()
        .clone();
    let mut nar = synthetic::model(1.0);
    let ar_tokens = u32s(&case["ar_tokens"][0]);
    let noise = &r["cond_end_5/noise"];
    let steps = case["steps"].as_u64().unwrap() as usize;
    let mut trace = Trace::default();
    let (latents, moved) = solve_chunk(
        &mut nar,
        &ChunkInput {
            ar_tokens: &ar_tokens,
            noise,
            nar_cond_end: 5,
        },
        steps,
        &NarOptions::default(),
        0,
        1,
        &mut SynthesisHooks {
            cancelled: &never,
            observer: &mut trace,
        },
    )
    .unwrap();
    assert_eq!(moved, 0);
    let mut spread = compare_evaluations(&trace, &r, "cond_end_5", 1, steps);
    spread.merge(Spread::of(&host(&latents), &host(&r["cond_end_5/final"])));
    eprintln!("cond_end_5: max |Δ| {:.3e}", spread.max_abs);
    spread.assert_within(SYNTH_MAX_ABS, SYNTH_REL_L2, "cond_end_5");
}

/// The song's noise is one tensor: each chunk starts from exactly its rows, whatever the chunking;
/// the seeded draw is a standard normal, prefix-stable in the frame count, and seed-dependent.
#[test]
fn song_noise_is_drawn_once_and_sliced_per_chunk() {
    let mut nar = synthetic::model(1.0);
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let codes: Vec<u32> = (0..21).map(|i| (i * 1543) % protocol::CODEC_SIZE).collect();
    let noise = SongNoise::seeded(7, codes.len());
    for context in [
        prefix.len() + 3 + 2 * 8,
        prefix.len() + 3 + 2 * 5,
        protocol::CONTEXT,
    ] {
        let (out, trace) = run(
            &mut nar,
            &prefix,
            &codes,
            &noise,
            1,
            context,
            &NarOptions::default(),
        );
        for (k, &(a, b)) in out.chunks.iter().enumerate() {
            let start = trace
                .evals
                .iter()
                .find(|e| e.chunk == k && e.stage == Stage::First)
                .unwrap();
            assert_eq!(
                start.input,
                noise.values()[a * LATENT_CHANNELS..b * LATENT_CHANNELS],
                "context {context}, chunk {k} [{a}, {b}) did not start from its song rows"
            );
        }
    }
    let long = SongNoise::seeded(7, 400);
    assert_eq!(noise.values(), &long.values()[..noise.values().len()]);
    assert_ne!(long.values(), SongNoise::seeded(8, 400).values());
    let n = long.values().len() as f64;
    let mean = long.values().iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = long
        .values()
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    // 25 600 draws: the sample mean's standard error is 1/160 and the variance's ~1/113.
    assert!(
        mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
        "mean {mean}, var {var}"
    );
    assert!(long.values().iter().all(|v| v.is_finite()));
    assert!(SongNoise::injected(vec![0.0; 64 * 3 - 1], 3).is_err());
    assert!(SongNoise::injected(Vec::new(), 0).is_err());
    let mut bad = vec![0.0; 64 * 2];
    bad[70] = f32::NAN;
    assert!(SongNoise::injected(bad, 2).is_err());
}

/// A long song over many original chunks is exactly the concatenation of its chunks solved one
/// by one ([`solve_chunk`] with each chunk's AR sequence and noise rows), every frame is produced,
/// and the chunk ranges tile the song with no gap or overlap.
#[test]
fn long_song_is_the_concatenation_of_its_original_chunks() {
    let mut nar = synthetic::model(1.0);
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let codes: Vec<u32> = (0..45)
        .map(|i| (i * 7919 + 13) % protocol::CODEC_SIZE)
        .collect();
    let noise = SongNoise::seeded(11, codes.len());
    let context = prefix.len() + 3 + 2 * 7;
    let steps = 2;
    let (out, trace) = run(
        &mut nar,
        &prefix,
        &codes,
        &noise,
        steps,
        context,
        &NarOptions::default(),
    );
    assert_eq!(out.chunks.len(), 7, "45 frames in 7-frame chunks");
    assert_eq!(out.chunks.first().unwrap().0, 0);
    assert_eq!(out.chunks.last().unwrap().1, codes.len());
    assert!(out.chunks.windows(2).all(|w| w[0].1 == w[1].0));
    assert_eq!(out.latents.frames(), codes.len());
    assert_eq!(trace.progress.last(), Some(&(7 * steps, 7 * steps)));
    let mut pieces = Vec::new();
    for &(a, b) in &out.chunks {
        let tokens = chunk_ar_tokens(&prefix, &codes[a..b]);
        let rows = noise.rows(a, b).unwrap();
        let (latents, _) = solve_chunk(
            &mut nar,
            &ChunkInput {
                ar_tokens: &tokens,
                noise: &rows,
                nar_cond_end: 0,
            },
            steps,
            &NarOptions::default(),
            0,
            1,
            &mut SynthesisHooks {
                cancelled: &never,
                observer: &mut (),
            },
        )
        .unwrap();
        pieces.extend(host(&latents));
    }
    assert_eq!(out.latents.values(), pieces.as_slice());
}

fn try_synthesize(
    nar: &mut Yue2Nar,
    prefix: &[u32],
    codes: &[u32],
    noise: &SongNoise,
    steps: usize,
    context: usize,
) -> gen_core::Result<()> {
    let request = SynthesisRequest {
        prefix,
        codes,
        noise,
        steps,
        context,
    };
    let hooks = SynthesisHooks {
        cancelled: &never,
        observer: &mut (),
    };
    synthesize(nar, &request, &NarOptions::default(), hooks).map(|_| ())
}

/// Explicit refusals instead of truncation or a silent default.
#[test]
fn invalid_requests_are_refused() {
    let mut nar = synthetic::model(1.0);
    let nar = &mut nar;
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let codes = vec![1u32, 2, 3];
    let noise = SongNoise::seeded(1, 3);
    let ctx = protocol::CONTEXT;
    assert!(try_synthesize(nar, &prefix, &codes, &noise, 1, ctx).is_ok());
    assert!(try_synthesize(nar, &prefix, &codes, &noise, 0, ctx).is_err());
    assert!(try_synthesize(nar, &[], &codes, &noise, 1, ctx).is_err());
    let empty = SongNoise::seeded(1, 0);
    assert!(try_synthesize(nar, &prefix, &[], &empty, 1, ctx).is_err());
    let past_codebook = [1, protocol::CODEC_SIZE, 3];
    assert!(try_synthesize(nar, &prefix, &past_codebook, &noise, 1, ctx).is_err());
    let four = SongNoise::seeded(1, 4);
    assert!(try_synthesize(nar, &prefix, &codes, &four, 1, ctx).is_err());
    let past_vocab = [protocol::VOCAB_SIZE];
    assert!(try_synthesize(nar, &past_vocab, &codes, &noise, 1, ctx).is_err());
    // (context − prefix − 3) // 2 = 0 frames per chunk: no acoustic context.
    assert!(try_synthesize(nar, &prefix, &codes, &noise, 1, prefix.len() + 4).is_err());
    assert!(try_synthesize(nar, &prefix, &codes, &noise, 1, ctx + 1).is_err());
    // One frame per chunk is the smallest context that works.
    assert!(try_synthesize(nar, &prefix, &codes, &noise, 1, prefix.len() + 5).is_ok());
    // A chunk handed to `solve_chunk` directly must still fit the model's positions.
    let too_long = vec![5u32; protocol::CONTEXT - 3];
    let rows = SongNoise::seeded(1, 2).rows(0, 2).unwrap();
    let err = solve_chunk(
        nar,
        &ChunkInput {
            ar_tokens: &too_long,
            noise: &rows,
            nar_cond_end: 0,
        },
        1,
        &NarOptions::default(),
        0,
        1,
        &mut SynthesisHooks {
            cancelled: &never,
            observer: &mut (),
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("positions"), "{err}");
    // An unusable query tile is refused before any work, not rounded to one row.
    for query_tile in [QueryTile::Rows(0), QueryTile::ScoreBytes(1)] {
        let request = SynthesisRequest {
            prefix: &prefix,
            codes: &codes,
            noise: &noise,
            steps: 1,
            context: ctx,
        };
        let options = NarOptions {
            query_tile,
            offload_ar: true,
        };
        let polls = Cell::new(0usize);
        let cancelled = || {
            polls.set(polls.get() + 1);
            false
        };
        let hooks = SynthesisHooks {
            cancelled: &cancelled,
            observer: &mut (),
        };
        let err = synthesize(nar, &request, &options, hooks).unwrap_err();
        assert!(err.to_string().contains("query"), "{query_tile:?}: {err}");
        assert_eq!(polls.get(), 2, "{query_tile:?}: refused before the prefill");
        assert!(!nar.lm().ar_offloaded());
    }
}

/// Cancellation at every bounded boundary — before a chunk's prefill, between prefill forwards,
/// before the first and the midpoint evaluation, between chunks — returns `Canceled`, leaves the
/// AR path restored (offload on), and leaves the model producing exactly what it produced before.
#[test]
fn cancellation_releases_resources_and_leaves_the_model_intact() {
    let mut nar = synthetic::model(1.0);
    // A prefix longer than one prefill forward, so a poll happens between prefill forwards.
    let mut prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let tail = prefix.split_off(prefix.len() - 1);
    prefix.extend((0..crate::model::PREFILL_CHUNK as u32 + 20).map(|i| (i * 31) % 151_000));
    prefix.extend(tail);
    let codes: Vec<u32> = (0..10).map(|i| i * 3001 % protocol::CODEC_SIZE).collect();
    let noise = SongNoise::seeded(3, codes.len());
    let context = prefix.len() + 3 + 2 * 5; // two chunks of five frames
    let options = NarOptions {
        query_tile: QueryTile::Upstream,
        offload_ar: true,
    };
    let reference = run(&mut nar, &prefix, &codes, &noise, 2, context, &options).0;
    // Polls per chunk: the chunk loop, `solve_chunk`, two prefill forwards, two per step.
    let polls_per_chunk = 4 + 2 * 2;
    for cancel_at in 0..2 * polls_per_chunk {
        let calls = Cell::new(0usize);
        let cancelled = || {
            let n = calls.get();
            calls.set(n + 1);
            n >= cancel_at
        };
        let request = SynthesisRequest {
            prefix: &prefix,
            codes: &codes,
            noise: &noise,
            steps: 2,
            context,
        };
        let hooks = SynthesisHooks {
            cancelled: &cancelled,
            observer: &mut (),
        };
        let out = synthesize(&mut nar, &request, &options, hooks);
        assert!(
            matches!(out, Err(gen_core::Error::Canceled)),
            "cancel at poll {cancel_at}: {out:?}"
        );
        assert_eq!(
            calls.get(),
            cancel_at + 1,
            "no poll after the cancelling one"
        );
        assert!(
            !nar.lm().ar_offloaded(),
            "cancel at poll {cancel_at}: AR path restored"
        );
    }
    // The model is intact: a full run after all the cancellations equals the first one, and the
    // AR path still prefills.
    let after = run(&mut nar, &prefix, &codes, &noise, 2, context, &options).0;
    assert_eq!(after.latents, reference.latents);
    let mut cache = nar.lm().new_cache(4).unwrap();
    nar.lm().prefill(&[1, 2, 3], &mut cache, || Ok(())).unwrap();
}

/// The offload transition: while offloaded every AR entry point refuses (never silently computes
/// on host copies), a synthesis refuses to start on a model left offloaded, and a restore brings
/// back exactly the same AR function.
#[test]
fn ar_offload_transitions_are_explicit_and_reversible() {
    let mut nar = synthetic::model(1.0);
    let ids = [151_643u32, 40, 1234, 99];
    let prefill = |lm: &Yue2Lm| {
        let mut cache = lm.new_cache(ids.len()).unwrap();
        lm.prefill(&ids, &mut cache, || Ok(())).map(|t| host(&t))
    };
    let before = prefill(nar.lm()).unwrap();
    // A host model has nothing to move; the AR path is still marked unavailable.
    assert_eq!(nar.lm.offload_ar().unwrap(), 0);
    assert!(nar.lm().ar_offloaded());
    let err = prefill(nar.lm()).unwrap_err();
    assert!(err.to_string().contains("offloaded"), "{err}");
    let rows = SongNoise::seeded(1, 2).rows(0, 2).unwrap();
    let err = solve_chunk(
        &mut nar,
        &ChunkInput {
            ar_tokens: &ids,
            noise: &rows,
            nar_cond_end: 0,
        },
        1,
        &NarOptions::default(),
        0,
        1,
        &mut SynthesisHooks {
            cancelled: &never,
            observer: &mut (),
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("restore"), "{err}");
    nar.restore_ar().unwrap();
    nar.restore_ar().unwrap(); // idempotent
    assert_eq!(prefill(nar.lm()).unwrap(), before);
}

/// The stage identity names everything the latents are a function of — and nothing else.
#[test]
fn stage_identity_covers_the_inputs_not_the_memory_controls() {
    let prefix = [1u32, 2, 3];
    let codes = [4u32, 5];
    let noise = SongNoise::seeded(1, 2);
    let base = SynthesisRequest {
        prefix: &prefix,
        codes: &codes,
        noise: &noise,
        steps: 32,
        context: protocol::CONTEXT,
    };
    let id = base.stage_identity("w", DType::F32);
    assert_eq!(id.len(), 64);
    assert_eq!(id, base.stage_identity("w", DType::F32));
    let other_noise = SongNoise::seeded(2, 2);
    let variants = [
        SynthesisRequest {
            prefix: &[1, 2, 4],
            ..base
        },
        SynthesisRequest {
            codes: &[4, 6],
            ..base
        },
        SynthesisRequest {
            noise: &other_noise,
            ..base
        },
        SynthesisRequest { steps: 31, ..base },
        SynthesisRequest {
            context: protocol::CONTEXT - 1,
            ..base
        },
    ];
    for v in variants {
        assert_ne!(v.stage_identity("w", DType::F32), id);
    }
    assert_ne!(base.stage_identity("v", DType::F32), id);
    assert_ne!(
        base.stage_identity("w", DType::BF16),
        id,
        "the compute dtype"
    );
}

/// The released acoustic config parses; a different latent width or a bad shift is refused.
#[test]
fn acoustic_config_parses_and_refuses_other_architectures() {
    let released = r#"{"latent_type":"vae","latent_dim":64,"max_latent_frames":24576,
        "timestep_shift":1.0}"#;
    let cfg = NarConfig::from_json(released).unwrap();
    assert_eq!(cfg.max_latent_frames, 24_576);
    assert_eq!(cfg.timestep_shift, 1.0);
    for (from, to) in [
        (r#""latent_dim":64"#, r#""latent_dim":32"#),
        (r#""timestep_shift":1.0"#, r#""timestep_shift":0.0"#),
        (r#""timestep_shift":1.0"#, r#""timestep_shift":-2.0"#),
        (r#""max_latent_frames":24576"#, r#""max_latent_frames":0"#),
        (r#""max_latent_frames":24576,"#, ""),
    ] {
        let bad = released.replace(from, to);
        assert!(NarConfig::from_json(&bad).is_err(), "accepted {to}");
    }
}

/// [`QueryTile::rows`]: exact row counts, the budget covers all [`SCORE_TILES_LIVE`] live score
/// tiles whenever a row fits, and unusable settings are refused rather than rounded.
#[test]
fn query_tile_rows_are_bounded_by_the_budget() {
    let f32_row = SCORE_TILES_LIVE * 16 * 1000 * 4; // 16 heads, 1000 keys, F32
    let bf16_row = SCORE_TILES_LIVE * 16 * 1000 * 2;
    let cases = [
        (QueryTile::ScoreBytes(f32_row), 300, DType::F32, 1),
        (QueryTile::ScoreBytes(5 * f32_row + 7), 300, DType::F32, 5),
        (QueryTile::ScoreBytes(2 * f32_row - 1), 300, DType::F32, 1),
        (QueryTile::ScoreBytes(10 * bf16_row), 300, DType::BF16, 10),
        (QueryTile::ScoreBytes(usize::MAX / 2), 300, DType::F32, 300),
        (QueryTile::Rows(7), 300, DType::F32, 7),
        (QueryTile::Rows(7), 3, DType::F32, 3),
        (QueryTile::Upstream, 1000, DType::F32, UPSTREAM_QUERY_TILE),
        (QueryTile::Upstream, 9, DType::F32, 9),
        (QueryTile::Whole, 1000, DType::F32, 1000),
    ];
    for (tile, queries, dtype, want) in cases {
        let rows = tile.rows(queries, 16, 1000, dtype).unwrap();
        assert_eq!(rows, want, "{tile:?} over {queries} queries in {dtype:?}");
        if let QueryTile::ScoreBytes(bytes) = tile {
            let peak = SCORE_TILES_LIVE * rows * 16 * 1000 * dtype.size_in_bytes();
            assert!(peak <= bytes, "{tile:?}: {rows} rows need {peak} bytes");
        }
    }
    assert!(QueryTile::Rows(0).rows(10, 16, 1000, DType::F32).is_err());
    assert!(QueryTile::ScoreBytes(f32_row - 1)
        .rows(10, 16, 1000, DType::F32)
        .is_err());
    assert!(QueryTile::ScoreBytes(0)
        .rows(10, 16, 1000, DType::F32)
        .is_err());
}

/// Panics in an observer mid-solve: the panic reaches the caller, and the offloaded AR path is
/// restored on the way (the model is not left refusing its AR path, and still synthesizes the
/// same latents).
#[test]
fn a_panic_mid_solve_still_restores_the_ar_path() {
    struct Boom;
    impl SynthesisObserver for Boom {
        fn on_evaluation(&mut self, e: &Evaluation<'_>) {
            if e.stage == Stage::Midpoint {
                panic!("observer failure");
            }
        }
    }
    let mut nar = synthetic::model(1.0);
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let codes = [3u32, 1, 4, 1, 5];
    let noise = SongNoise::seeded(9, codes.len());
    let options = NarOptions {
        query_tile: QueryTile::Upstream,
        offload_ar: true,
    };
    let before = run(
        &mut nar,
        &prefix,
        &codes,
        &noise,
        2,
        protocol::CONTEXT,
        &options,
    )
    .0;
    let request = SynthesisRequest {
        prefix: &prefix,
        codes: &codes,
        noise: &noise,
        steps: 2,
        context: protocol::CONTEXT,
    };
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let hooks = SynthesisHooks {
            cancelled: &never,
            observer: &mut Boom,
        };
        synthesize(&mut nar, &request, &options, hooks)
    }));
    let payload = panicked.expect_err("the observer's panic reaches the caller");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"observer failure"));
    assert!(
        !nar.lm().ar_offloaded(),
        "AR path restored during the unwind"
    );
    let after = run(
        &mut nar,
        &prefix,
        &codes,
        &noise,
        2,
        protocol::CONTEXT,
        &options,
    )
    .0;
    assert_eq!(after.latents, before.latents);
}

/// A restore failure is appended to the error that caused it, never substituted for it.
#[test]
fn a_failed_restore_keeps_the_original_error() {
    assert!(matches!(
        with_restore(gen_core::Error::Canceled, Ok(())),
        gen_core::Error::Canceled
    ));
    let both = with_restore(
        gen_core::Error::Msg("offload: device lost".into()),
        Err(gen_core::Error::Msg("restore: out of memory".into())),
    )
    .to_string();
    let (original, restore) = (
        both.find("offload: device lost").expect("original kept"),
        both.find("restore: out of memory")
            .expect("restore failure kept"),
    );
    assert!(original < restore, "original first: {both}");
}

/// Chunks at the real protocol boundary: with the released [`protocol::CONTEXT`], a song of
/// `size + 1` frames (`size = (CONTEXT − prefix − 3) / 2`) is one chunk whose AR + NAR positions
/// are **exactly** `CONTEXT` — the model's whole position range — and a one-frame tail chunk. A
/// chunk one position longer is refused rather than truncated.
#[test]
fn chunks_at_the_released_context_limit() {
    let mut nar = synthetic::model(1.0);
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let size = (protocol::CONTEXT - prefix.len() - 3) / 2;
    assert_eq!(
        prefix.len() + 2 * size + 3,
        protocol::CONTEXT,
        "this prefix length makes the first chunk fill the context exactly"
    );
    assert_eq!(nar.lm().config().max_position_embeddings, protocol::CONTEXT);
    let frames = size + 1;
    let codes: Vec<u32> = (0..frames as u32)
        .map(|i| (i * 7919 + 5) % protocol::CODEC_SIZE)
        .collect();
    let noise = SongNoise::seeded(21, frames);
    let options = NarOptions::default();
    let (out, trace) = run(
        &mut nar,
        &prefix,
        &codes,
        &noise,
        1,
        protocol::CONTEXT,
        &options,
    );
    assert_eq!(out.chunks, vec![(0, size), (size, frames)]);
    assert_eq!(out.latents.frames(), frames, "every frame solved");
    assert_eq!(trace.evals.len(), 4);
    // The one-frame tail solved alone is the song's last frame (composition at the boundary).
    let tail_tokens = chunk_ar_tokens(&prefix, &codes[size..]);
    let tail_noise = noise.rows(size, frames).unwrap();
    let hooks = &mut SynthesisHooks {
        cancelled: &never,
        observer: &mut (),
    };
    let chunk = ChunkInput {
        ar_tokens: &tail_tokens,
        noise: &tail_noise,
        nar_cond_end: 0,
    };
    let (tail, _) = solve_chunk(&mut nar, &chunk, 1, &options, 0, 1, hooks).unwrap();
    assert_eq!(
        host(&tail),
        out.latents.values()[size * LATENT_CHANNELS..].to_vec()
    );
    // One position past the model's range: the full chunk's AR sequence with one more frame.
    let over_tokens = chunk_ar_tokens(&prefix, &codes[..size]);
    let over_noise = noise.rows(0, size + 1).unwrap();
    let chunk = ChunkInput {
        ar_tokens: &over_tokens,
        noise: &over_noise,
        nar_cond_end: 0,
    };
    let err = solve_chunk(&mut nar, &chunk, 1, &options, 0, 1, hooks).unwrap_err();
    assert!(err.to_string().contains("positions"), "{err}");
}

/// The offload transition on a real accelerator, where it actually moves memory (on the CPU it
/// only marks the AR path unavailable). Runs on the manual CUDA lane (`--features cuda`) and any
/// `--features metal` test run; compiled out of the CPU lanes.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[test]
fn ar_offload_moves_exactly_the_ar_weights_on_a_gpu() {
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).expect("a CUDA device");
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::new_metal(0).expect("a Metal device");
    let on = |devices: &[Device]| devices.iter().all(|d| d.same_device(&device));
    let host_only = |devices: &[Device]| devices.iter().all(Device::is_cpu);
    // Every AR-only tensor of the synthetic checkpoint (F32), counted from its state dict.
    let cfg = crate::model::synthetic::config();
    let expected: usize = crate::model::synthetic::state_dict(&cfg)
        .iter()
        .filter(|(name, ..)| !name.contains(".nar_") && name != "model.norm.weight")
        .map(|(_, shape, ..)| shape.iter().product::<usize>() * 4)
        .sum();
    let mut nar = synthetic::model_on(1.0, &device);
    let (ar, fixed) = nar.lm().placement();
    assert!(on(&ar) && on(&fixed), "loaded on the device");

    // The transition itself.
    assert_eq!(nar.lm.offload_ar().unwrap(), expected, "bytes moved");
    let (ar, fixed) = nar.lm().placement();
    assert!(
        host_only(&ar),
        "every AR-only tensor on the host while offloaded"
    );
    assert!(on(&fixed), "NAR twins and the final norm never move");
    nar.restore_ar().unwrap();
    let (ar, _) = nar.lm().placement();
    assert!(on(&ar), "restored to the device");

    // Completed synthesis, offload off vs on.
    let prefix: Vec<u32> = u32s(&meta()["cases"][0]["prefix"]);
    let codes: Vec<u32> = (0..12).map(|i| (i * 977) % protocol::CODEC_SIZE).collect();
    let noise = SongNoise::seeded(5, codes.len());
    let context = prefix.len() + 3 + 2 * 5; // three chunks: 5, 5, 2
    let plain = NarOptions::default();
    let offload = NarOptions {
        offload_ar: true,
        ..plain
    };
    let base = run(&mut nar, &prefix, &codes, &noise, 2, context, &plain).0;
    assert_eq!(base.offloaded_bytes, 0);
    let moved = run(&mut nar, &prefix, &codes, &noise, 2, context, &offload).0;
    assert_eq!(
        moved.offloaded_bytes, expected,
        "the offload moved exactly the AR weights"
    );
    Spread::of(moved.latents.values(), base.latents.values()).assert_within(
        TILING_MAX_ABS,
        1e-5,
        "offload vs no offload on the device",
    );
    let (ar, fixed) = nar.lm().placement();
    assert!(on(&ar) && on(&fixed) && !nar.lm().ar_offloaded());

    // Cancelled mid-solve with offload on: 7 polls per chunk (chunk loop, `solve_chunk`, one
    // prefill forward, two per step), so poll 14 is the second chunk's last midpoint evaluation.
    let polls = Cell::new(0usize);
    let cancelled = || {
        polls.set(polls.get() + 1);
        polls.get() > 8 + 5
    };
    let request = SynthesisRequest {
        prefix: &prefix,
        codes: &codes,
        noise: &noise,
        steps: 2,
        context,
    };
    let hooks = SynthesisHooks {
        cancelled: &cancelled,
        observer: &mut (),
    };
    let out = synthesize(&mut nar, &request, &offload, hooks);
    assert!(matches!(out, Err(gen_core::Error::Canceled)), "{out:?}");
    let (ar, fixed) = nar.lm().placement();
    assert!(
        on(&ar) && on(&fixed) && !nar.lm().ar_offloaded(),
        "restored after a cancel"
    );
    let again = run(&mut nar, &prefix, &codes, &noise, 2, context, &offload).0;
    assert_eq!(again.latents.values(), moved.latents.values());
}
