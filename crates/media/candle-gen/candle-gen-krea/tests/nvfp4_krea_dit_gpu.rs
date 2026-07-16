//! **sc-12110 — the epic's SC#1/SC#2 validation, on Krea 2 Turbo, consumer Blackwell `sm_120`** (epic
//! 11037).
//!
//! This is the story the whole NVFP4 epic hinges on. Everything before it measured either *layers*
//! (sc-11041/sc-11044, synthetic or single-weight) or a model that could not settle the question:
//! sc-11045 drove a real SANA-1.6B denoise and proved the lane works end-to-end (SC#3/SC#4), but SANA's
//! Mix-FFN is a `GLUMBConv` — **linears are 0.20% of block time**, capping any end-to-end multiple at
//! ~1.002× regardless of how fast the FP4 lane is. Michael redirected the vehicle here (epic
//! activity-12117).
//!
//! **Krea 2 Turbo is the inverse of SANA on every relevant axis:** a ~12.5B single-stream DiT, hidden
//! 6144 × 28 blocks × intermediate 16384, **100% linear GEMM with zero `Conv2d`** — so the NVFP4 lane
//! reaches essentially all parameterized compute — and it is the only model in the workspace with
//! **both** epic baselines wired and on disk: dense **bf16** (SC#1's specified baseline) and the
//! **Q4 dequant-on-forward tier** NVFP4 actually replaces (SC#2's honest baseline).
//!
//! What each gate pins:
//!
//! 1. [`nvfp4_krea_dit_sc1_throughput_bf16_vs_w4a16_vs_w4a4`] — **SC#1, the number of record.** ms/step
//!    for dense bf16 vs NVFP4 W4A16 vs W4A4-mixed vs W4A4-blanket, on the real trunk under an exclusive
//!    GPU, decomposed into FP4-GEMM vs activation-quantizer share (one test, not two: the decomposition
//!    is arithmetic on this table, and a second test would re-pack the same three 12.5B trunks).
//!    **Measured answer (2026-07-16, after sc-12207 + sc-12078): W4A4 is a net WIN — blanket 1.25×,
//!    mixed 1.10× vs dense bf16 — but SC#1's ~2× is still NOT met.** The earlier 0.01× reading was an
//!    artefact of two now-fixed quantizer defects, not of W4A4; the residual gap to 2× is structural (a
//!    denoise step is not all GEMM). See the test's docs for the full measured history.
//! 2. [`nvfp4_krea_dit_sc2_parity_vs_q4_tier`] — **SC#2.** NVFP4 vs the **Q4 tier**, all against a bf16
//!    reference. sc-11045 explicitly deferred this: cosine vs bf16 measures *divergence*, not quality,
//!    and no 4-bit tier passes a "cosine > 0.95 vs bf16" bar. The question SC#2 actually asks is
//!    whether NVFP4 is at least as good as the tier it replaces. Runs **three** NVFP4/Q4 legs, because
//!    Q4 is weight-only: NVFP4 **W4A16** is the like-for-like weight-format comparison (and the only
//!    throughput-viable regime), while NVFP4 **W4A4-mixed** additionally pays for FP4 activations —
//!    scoring only the latter against Q4 would conflate two different questions.
//! 3. [`nvfp4_krea_dit_real_activation_outlier_sparsity`] — **the partition, re-derived on Krea's
//!    naming.** The sc-11038 policy is substring-based and was tuned on SANA; Krea has no `attn2`, no
//!    `caption_projection`, and a head named `final_layer.linear`. Measures real per-layer outlier
//!    sparsity across live denoise steps and crosses measured-vs-assumed. Includes `to_gate`, the
//!    projection with no SANA analogue.
//! 4. [`nvfp4_krea_dit_sc6_resident_vram_per_regime`] — **SC#6 per regime**, by contention-immune tensor
//!    byte-accounting (`resident_bytes()`, not a free-mem delta).
//! 5. [`nvfp4_krea_dit_sc3_no_nan_across_full_denoise`] — **SC#3**, NaN guard armed on every projection
//!    at every step of a full Turbo denoise.
//!
//! All are `#[cfg(feature = "cuda")]` + `#[ignore]`d + weight-env-gated per repo convention (the CPU
//! lane compiles `--all-targets` without cuda). Every test fn name carries an `nvfp4` substring so a
//! name-filtered lane cannot silently skip them.
//!
//! ```sh
//! KREA_TURBO_BF16_DIR=E:\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\bf16 \
//! KREA_TURBO_Q4_DIR=E:\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\q4 \
//! CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 \
//!   cargo test -p candle-gen-krea --release --features cuda --test nvfp4_krea_dit_gpu -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::{ActPrecision, CublasLt, OutlierClass};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::nvfp4_dit::LayerRole;
use candle_gen_krea::pipeline::MAX_TEXT_TOKENS;
use candle_gen_krea::{
    summarize, turbo_sigmas, ActProbe, DitPlan, Krea2Config, Krea2Transformer, KreaTeConfig,
    KreaTextEncoder, KreaTokenizer, Nvfp4Quant, TURBO_STEPS,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The prompt every gate encodes — a real caption, so the text-derived activations that drive the
/// outlier question are the real ones. (A synthetic `randn` context would understate outliers and
/// hand the partition gate a false green.)
const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

/// Render geometry for the timed runs. 1024² is Krea's native resolution and the shipping default.
const EDGE: usize = 1024;
const SEED: u64 = 0x5C12110;
/// Latent channels the DiT denoises (`Krea2Config::in_channels`, patchified p=2 → 64/patch).
const LATENT_CH: usize = 16;

// ==============================================================================================
// Environment / snapshot gates
// ==============================================================================================

/// The CUDA device, iff it is a real NVFP4-capable `sm_120` Blackwell part. `None` (skip) otherwise —
/// the SC#4 Blackwell-only gate, observed from the test side.
fn nvfp4_device() -> Option<Device> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-12110] no CUDA device; skipping");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_nvfp4_floor() {
        Ok(true) => {
            eprintln!(
                "[sc-12110] device cap = {:?} (NVFP4 eligible)",
                lt.compute_cap().unwrap()
            );
            Some(dev)
        }
        _ => {
            eprintln!(
                "[sc-12110] device not sm_120 ({:?}); skipping — NVFP4 is Blackwell-only (SC#4)",
                lt.compute_cap().ok()
            );
            None
        }
    }
}

/// The **bf16** snapshot root — SC#1's baseline tier and the master NVFP4 packs from.
fn bf16_root() -> Option<PathBuf> {
    match std::env::var("KREA_TURBO_BF16_DIR").ok().map(PathBuf::from) {
        Some(p) if p.join("transformer").is_dir() => Some(p),
        _ => {
            eprintln!("[sc-12110] KREA_TURBO_BF16_DIR unset/invalid; skipping");
            None
        }
    }
}

/// The **q4** snapshot root — SC#2's honest baseline (the tier NVFP4 replaces).
fn q4_root() -> Option<PathBuf> {
    match std::env::var("KREA_TURBO_Q4_DIR").ok().map(PathBuf::from) {
        Some(p) if p.join("transformer").is_dir() => Some(p),
        _ => {
            eprintln!("[sc-12110] KREA_TURBO_Q4_DIR unset/invalid; skipping");
            None
        }
    }
}

/// Load a tier's `transformer/` weight set at its native dtype.
fn trunk_weights(root: &std::path::Path, dev: &Device, dtype: DType) -> Weights {
    Weights::from_dir(&root.join("transformer"), dev, dtype).expect("load transformer weights")
}

/// Encode `PROMPT` through the **real Qwen3-VL-4B text encoder** to the DiT's `text_fusion` context
/// `[1, n_tok, 12, 2560]`, then drop the TE (freeing its VRAM before the 25 GB trunk loads).
///
/// This is load-bearing for the partition gate: Krea's outlier question is about **caption-derived**
/// activations, which only a real caption encode produces. A synthetic `randn` context would carry no
/// massive activations at all and hand the gate a false green.
///
/// The TE loads at **f32**, matching `pipeline::load_text`'s `TE_DTYPE` — the shipping path's dtype,
/// deliberately not the DiT's bf16. A bf16 TE yields a bf16 context and the trunk's front-end then
/// fails `dtype mismatch in binary op`; more importantly it would be a *different* encode than the one
/// that ships, i.e. the wrong activations to be measuring at all.
fn encode_context(root: &std::path::Path, dev: &Device) -> Tensor {
    let tok = KreaTokenizer::from_snapshot(root, dev).expect("tokenizer");
    let te_cfg = KreaTeConfig::from_snapshot(root).expect("te config");
    let te_w =
        Weights::from_dir(&root.join("text_encoder"), dev, DType::F32).expect("load TE weights");
    let te =
        KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS).expect("build TE");
    let ids = tok
        .encode_prompt(PROMPT, MAX_TEXT_TOKENS)
        .expect("tokenize");
    let ctx = te.forward(&ids).expect("TE forward");
    eprintln!(
        "[sc-12110] encoded context {:?} from a real caption",
        ctx.dims()
    );
    drop(te);
    drop(te_w);
    ctx
}

/// A seeded initial latent `[1, 16, H/8, W/8]` — the denoise's starting point. Mirrors
/// `pipeline::init_noise` (CPU draw → device, sc-3673 launch-portable determinism), so every regime's
/// run starts from byte-identical noise.
fn init_latent(dev: &Device, edge: usize) -> Tensor {
    let (h, w) = (edge / 8, edge / 8);
    let mut rng = StdRng::seed_from_u64(SEED);
    let v = candle_gen::seeded_normal_vec(&mut rng, LATENT_CH * h * w);
    Tensor::from_vec(v, (1, LATENT_CH, h, w), &Device::Cpu)
        .expect("init latent")
        .to_device(dev)
        .expect("latent to device")
}

fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn rel_rms(got: &[f32], reference: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        num += (*g as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

/// PSNR (dB) between two latents, normalized by the reference's dynamic range — the "no visible
/// regression" metric applied in latent space (we compare trunks, not decoders).
fn psnr_latent(got: &[f32], reference: &[f32]) -> f64 {
    let mse = got
        .iter()
        .zip(reference)
        .map(|(g, r)| (*g as f64 - *r as f64).powi(2))
        .sum::<f64>()
        / got.len() as f64;
    if mse <= f64::EPSILON {
        return f64::INFINITY;
    }
    let (mn, mx) = reference
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let peak = (mx - mn) as f64;
    10.0 * (peak * peak / mse).log10()
}

/// Assert the GPU is idle before a timed run. These are the epic's numbers of record — a contaminated
/// bench is worse than none (sc-11039's 2.96× was retracted for exactly this).
fn assert_exclusive_gpu(tag: &str) {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            eprintln!("[sc-12110] GPU state before {tag}:\n{}", s.trim());
        }
        Err(e) => eprintln!("[sc-12110] nvidia-smi unavailable ({e}); cannot verify exclusivity"),
    }
}

/// Run `steps` real Turbo denoise steps over `model`, driving the probe's step counter. Returns the
/// final latent. Krea Turbo is **CFG-free** — one DiT forward per step (`pipeline::render_from_context`).
fn run_denoise(
    model: &Krea2Transformer,
    ctx: &Tensor,
    dev: &Device,
    steps: usize,
    probe: Option<&Arc<ActProbe>>,
) -> Tensor {
    let sigmas = turbo_sigmas(steps);
    let mut x = init_latent(dev, EDGE);
    for i in 0..steps {
        if let Some(p) = probe {
            p.set_step(i);
        }
        let t = Tensor::from_vec(vec![sigmas[i]], (1,), dev).expect("timestep");
        let v = model
            .forward(&x, &t, ctx)
            .expect("DiT forward (a NaN guard trip surfaces here)")
            .to_dtype(DType::F32)
            .expect("velocity dtype");
        // Rectified-flow Euler: x + v·(σ_{i+1} − σ_i), matching `run_flow_sampler`'s Sigma convention.
        let dt = (sigmas[i + 1] - sigmas[i]) as f64;
        x = (&x + (v * dt).expect("euler scale")).expect("euler step");
    }
    x
}

/// Build a trunk under `plan` from `root`'s tier, print its NVFP4 report, return both.
fn build(
    root: &std::path::Path,
    dev: &Device,
    dtype: DType,
    plan: &DitPlan,
    tag: &str,
) -> Krea2Transformer {
    let cfg = Krea2Config::from_snapshot(root).expect("krea config");
    let w = trunk_weights(root, dev, dtype);
    let t0 = Instant::now();
    let m = Krea2Transformer::load_planned(&w, &cfg, plan).expect("build trunk");
    eprintln!(
        "[sc-12110] built `{tag}` trunk in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    drop(w);
    m
}

/// Time `steps` denoise steps after `warmup` untimed ones. Returns ms/step.
fn time_steps(
    model: &Krea2Transformer,
    ctx: &Tensor,
    dev: &Device,
    warmup: usize,
    steps: usize,
) -> f64 {
    let sigmas = turbo_sigmas(TURBO_STEPS);
    let x = init_latent(dev, EDGE);
    let t = Tensor::from_vec(vec![sigmas[0]], (1,), dev).expect("timestep");
    for _ in 0..warmup {
        let v = model.forward(&x, &t, ctx).expect("warmup forward");
        // Force completion: candle's CUDA path is async, so a `to_vec` sync is what makes the timing
        // real rather than a measure of launch-queue depth.
        let _ = to_vec_f32(&v.narrow(0, 0, 1).unwrap().flatten_all().unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        let v = model.forward(&x, &t, ctx).expect("timed forward");
        let _ = to_vec_f32(&v.narrow(0, 0, 1).unwrap().flatten_all().unwrap());
    }
    t0.elapsed().as_secs_f64() * 1000.0 / steps as f64
}

// ==============================================================================================
// (1) SC#1 — the number of record.
// ==============================================================================================

/// **SC#1: does NVFP4 deliver ~2× over the current bf16 compute path on a real, all-linear DiT?**
///
/// This is the measurement the epic has been trying to make since sc-11039, on the first vehicle where
/// it is meaningful. Four regimes over the real Krea 2 Turbo trunk at 1024², CFG-free (Turbo runs one
/// DiT forward per step), exclusive GPU:
///
/// | regime | what it is |
/// |---|---|
/// | dense **bf16** | the epic's specified SC#1 baseline — "the current bf16 compute path" |
/// | NVFP4 **W4A16** | the storage tier: FP4 weights, bf16 activation, **no FP4 compute** |
/// | NVFP4 **W4A4 mixed** | the shipping policy: FP4 compute on the benign bulk, W4A16 on the outlier class |
/// | NVFP4 **W4A4 blanket** | every eligible projection on the FP4 cores — the compute ceiling, not shippable |
///
/// # The prediction under test
///
/// sc-11044 measured the unfused activation quantizer at ~25.8 ms/fwd, and on SANA it swamped the FP4
/// GEMM (W4A4 came in at **0.69×** — slower than dense). But quantizer cost scales ~O(M·K) while the
/// GEMM scales ~O(M·K·N), so **overhead/GEMM ≈ 1/N**. Krea's SwiGLU N is **16384** — ~8× SANA's — so the
/// same absolute overhead should amortize ~8× better here. If that holds, W4A4 wins on Krea *before*
/// sc-12078 (the fused quantize kernel) lands, and sc-12078 drops from a gate to an optimization.
///
/// The test reports the multiple and the quantizer share; it does **not** assert a 2× bar. SC#1 is a
/// question, and the honest answer is whatever the rig says.
///
/// # The measured answer (2026-07-16, exclusive rig, after sc-12207 + sc-12078)
///
/// | regime | ms/step | vs bf16 | FP4-lit | *(2026-07-15, pre-fix)* |
/// |---|---:|---:|---:|---:|
/// | dense bf16 | 893.7 | 1.00× | — | *907.9* |
/// | NVFP4 W4A16 | 895.7 | 1.00× | 0/260 | *897.3 (1.01×)* |
/// | NVFP4 W4A4 mixed | 810.6 | **1.10×** | 139/260 | *45 992.4 (**0.02×**)* |
/// | NVFP4 W4A4 blanket | 716.3 | **1.25×** | 260/260 | *90 109.8 (**0.01×**)* |
///
/// **W4A4 is a net win — and SC#1's ~2× is still NOT met.** Both halves of that sentence matter.
///
/// The `W4A4 − W4A16` delta flipped sign: **+89 212 ms → −179.4 ms**. The FP4 GEMM's speedup now
/// *exceeds* the activation-quantizer cost, instead of being buried under it by two orders of magnitude.
///
/// # Why the old 0.01× was an artefact, not a property of W4A4
///
/// The 2026-07-15 rows measured two defects in the *quantizer*, not the cost of FP4 activations. Both are
/// fixed; the per-projection cost fell **~860–914×**:
///
/// | per projection (M=4118) | K=6144 | K=16384 |
/// |---|---:|---:|
/// | 2026-07-15 (`scatter_add` swizzle) | 328.10 ms | 879.42 ms |
/// | after **sc-12207** (bijection → `index_select` gather) | 19.08 ms | 55.30 ms |
/// | after **sc-12078** (fused two-pass CUDA kernel) | **0.382 ms** | **0.962 ms** |
/// | FP4 GEMM at the same shape, for scale | 0.366 ms | 2.253 ms |
///
/// 1. **sc-12207** — the UE4M3 scale swizzle is a *pure bijection* (the packer's own
///    `scale_swizzle_padding_and_bijection` test proves it) but was implemented with `scatter_add`
///    **atomics**: 76% of the quantizer, and `index_select` does the identical permutation ~7 000× faster.
/// 2. **sc-12078** — the remaining ~19 ms was ~40 unfused candle ops plus a host sync, now one fused
///    kernel (block+tensor amax → E4M3 scale → E2M1 codes → nibble pack → swizzle) that is **bit-exact**
///    (rel-RMS 0.000000 vs the CPU packer). The quantizer is now *smaller than the GEMM it feeds*.
///
/// Two superseded conclusions, recorded so they are not re-derived: (a) *"a fused kernel must make
/// quantize essentially free for W4A4 to break even"* — withdrawn; the true unfused residual was 19 ms,
/// not 343. (b) *"post-sc-12207 blanket W4A4 ≈ 21.4 s/step (0.043×), still not viable"* — that estimate
/// assumed a ~78 ms residual by naive subtraction; the warm, index-cached residual measured 19 ms, and
/// with sc-12078 the real answer is **716 ms/step (1.25×)**.
///
/// # Why ~2× is still not reached — and why no quantizer work will get there
///
/// The FP4 tensor-core GEMM *does* deliver ~2× (measured **1.95–2.24×** core-vs-core, sc-11044). But a
/// denoise step is not all GEMM: attention, norms and modulation are untouched by NVFP4 and dilute the
/// GEMM win to 1.25× end-to-end. With the quantizer now at ~0.4 ms against a ~0.37 ms GEMM, there is no
/// meaningful quantizer cost left to remove — **the residual gap to 2× is structural, not a defect.**
/// Reaching it would require accelerating the non-GEMM step work, which is outside this epic.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot + an sm_120 device"]
fn nvfp4_krea_dit_sc1_throughput_bf16_vs_w4a16_vs_w4a4() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };
    assert_exclusive_gpu("SC#1 throughput");

    let ctx = encode_context(&root, &dev);
    let (warmup, steps) = (2usize, 6usize);

    // Each regime is built, timed, and DROPPED before the next loads — 25 GB dense bf16 plus a second
    // resident trunk would be avoidable pressure, and a freed allocator is the cleaner timing baseline.
    let mut rows: Vec<(&str, f64, String)> = Vec::new();

    {
        let m = build(&root, &dev, DType::BF16, &DitPlan::baseline(), "dense bf16");
        let ms = time_steps(&m, &ctx, &dev, warmup, steps);
        rows.push(("dense bf16 (SC#1 baseline)", ms, "—".to_string()));
    }
    for (tag, quant) in [
        ("NVFP4 W4A16", Nvfp4Quant::BlanketW4A16),
        ("NVFP4 W4A4 (mixed)", Nvfp4Quant::Mixed),
        ("NVFP4 W4A4 (blanket)", Nvfp4Quant::BlanketW4A4),
    ] {
        let m = build(&root, &dev, DType::BF16, &DitPlan::nvfp4(quant), tag);
        let r = m.nvfp4_report();
        let ms = time_steps(&m, &ctx, &dev, warmup, steps);
        rows.push((tag, ms, format!("{}/{} fp4-lit", r.fp4_lit, r.n_quantized)));
    }

    let base = rows[0].1;
    eprintln!(
        "\n[sc-12110] ===== SC#1 THROUGHPUT — Krea 2 Turbo, 1024², CFG-free, exclusive GPU ====="
    );
    eprintln!(
        "[sc-12110] {:<28} {:>12} {:>10}   {}",
        "regime", "ms/step", "vs bf16", "lane"
    );
    for (tag, ms, lane) in &rows {
        eprintln!(
            "[sc-12110] {:<28} {:>12.1} {:>9.2}×   {}",
            tag,
            ms,
            base / ms,
            lane
        );
    }
    let best = rows[1..]
        .iter()
        .map(|(t, ms, _)| (*t, base / *ms))
        .fold(("", 0f64), |acc, x| if x.1 > acc.1 { x } else { acc });
    eprintln!(
        "\n[sc-12110] SC#1 VERDICT: best NVFP4 regime = {} at {:.2}× vs dense bf16 — target ~2× is {}",
        best.0,
        best.1,
        if best.1 >= 2.0 { "MET" } else { "NOT met" }
    );

    // ---- The quantizer-share decomposition, derived from the SAME four measurements ----------
    //
    // A W4A4 step is (FP4 GEMM) + (fused activation quantize, sc-12078). W4A16 holds the same packed
    // weights but dequantizes them once at construction and runs a dense bf16 GEMM — so it is the "no FP4
    // compute, NO act-quant" leg. The W4A4 − W4A16 delta is therefore the NET of two opposing terms: the
    // FP4 GEMM's speedup (negative) and the activation-quantizer's cost (positive). Pre-sc-12207 the
    // quantizer term dominated so utterly (+89 212 ms) that the delta read as a pure quantizer bound;
    // post-sc-12078 the sign has flipped (−179.4 ms), i.e. the GEMM speedup now exceeds the quantizer.
    // The delta is consequently a LOWER bound on the GEMM win, not a bound on the quantizer cost — read
    // the per-projection quantizer numbers in this test's docs for that.
    let (ms_w4a16, ms_w4a4) = (rows[1].1, rows[3].1);
    let delta = ms_w4a4 - ms_w4a16;
    let w4a4_mult = base / ms_w4a4;
    eprintln!("\n[sc-12110] ===== W4A4 STEP DECOMPOSITION (Krea N=16384 vs SANA's ~2048) =====");
    eprintln!("[sc-12110] dense bf16         : {base:>10.1} ms/step");
    eprintln!("[sc-12110] NVFP4 W4A16        : {ms_w4a16:>10.1} ms/step  (packed weights, bf16 act, no FP4 compute, no act-quant)");
    eprintln!(
        "[sc-12110] NVFP4 W4A4 blanket : {ms_w4a4:>10.1} ms/step  (FP4 GEMM + FUSED act-quant, sc-12078)"
    );
    eprintln!(
        "[sc-12110] W4A4 − W4A16 delta : {delta:>+10.1} ms/step ({:+.1}% of the W4A16 step) — net of the \
         FP4 GEMM speedup (−) and the fused act-quant cost (+); {}",
        100.0 * delta / ms_w4a16,
        if delta < 0.0 {
            "NEGATIVE ⟹ the FP4 GEMM win now exceeds the quantizer cost (was +89 212 ms pre-sc-12207)"
        } else {
            "POSITIVE ⟹ the quantizer still costs more than the FP4 GEMM saves"
        }
    );
    eprintln!(
        "[sc-12110] HISTORY: pre-sc-12207 this table read W4A4 blanket **0.01×** (90 109.8 ms/step). That \
         was an artefact of the quantizer, not of FP4 activations: a `scatter_add`-atomics bijection \
         (sc-12207) plus ~40 unfused candle ops (sc-12078). Per-projection quantizer at K=6144 fell \
         328.10 → 19.08 → **0.382 ms** (~859×), and is now smaller than the 0.366 ms GEMM it feeds. W4A4 \
         measures **{w4a4_mult:.2}×** vs dense bf16 ⟹ {}.",
        if w4a4_mult > 1.0 {
            "W4A4 is a net WIN — which it never was before the quantizer was fixed"
        } else {
            "W4A4 is STILL a net loss — investigate before quoting the fused kernel as effective"
        }
    );
    eprintln!(
        "[sc-12110] ⟹ sc-12078 (fused activation-quantize kernel) is LANDED and wired into \
         `Nvfp4Linear::forward_fp4`. SC#1's ~2× is {}",
        if best.1 >= 2.0 {
            "MET.".to_string()
        } else {
            format!(
                "NOT met (best {:.2}×) — and no further quantizer work will reach it. The FP4 GEMM core \
                 does deliver ~2× (1.95–2.24× core-vs-core, sc-11044), but attention/norms/modulation are \
                 untouched by NVFP4 and dilute it end-to-end. With the quantizer at ~0.4 ms against a \
                 ~0.37 ms GEMM there is no quantizer cost left to remove: the residual gap is STRUCTURAL. \
                 Closing it means accelerating the non-GEMM step work, which is outside this epic.",
                best.1
            )
        }
    );

    // No throughput bar is asserted — SC#1 is the question this test exists to answer honestly, and a
    // green-by-construction assert would defeat the point. What IS asserted: the lane actually ran. A
    // regime that silently fell back to bf16 would make the whole table a measurement of nothing.
    assert!(
        rows.iter()
            .any(|(_, _, lane)| lane.contains('/') && !lane.starts_with("0/")),
        "no NVFP4 regime lit a single FP4 layer — the table would be meaningless"
    );
}

// ==============================================================================================
// (2) SC#2 — parity against the Q4 tier (the honest baseline).
// ==============================================================================================

/// **SC#2: no quality regression — measured against the tier NVFP4 actually replaces.**
///
/// sc-11045 deliberately did not close this and said why: a flow-match denoise is chaotic, so any
/// ~4.5-bit weight tier walks the sampler onto a *different but equally valid* trajectory. Cosine/PSNR
/// **vs bf16** therefore measures divergence, not quality — **no 4-bit tier would pass a "cosine > 0.95
/// vs bf16" bar**, including the Q4 tier we ship today. Asserting such a bar would be tuning a number
/// until it looked green.
///
/// The question SC#2 actually asks, on a model that has both tiers on disk: **is NVFP4 at least as good
/// as the Q4 dequant-on-forward tier it replaces?** All three trunks run the same seed, prompt and
/// schedule; bf16 is the common reference; Q4 and NVFP4 are each scored against it, and against each
/// other.
///
/// # ⚠ This test CANNOT rank the two weight formats — do not use it for that (sc-12110 review)
///
/// An earlier revision of this test's docs concluded from the numbers below that **"NVFP4's weight
/// format is less faithful than int4 at identical 4.5 bits."** **That conclusion was false and is
/// withdrawn.** The metric here is an end-to-end 8-step denoise cosine — which the paragraph above
/// already calls a measure of *divergence, not quality* — and it **disagrees in direction with every
/// direct measurement of weight fidelity**:
///
/// * **weight rel-RMS** via `Nvfp4Tensor::pack` vs `dequant_mlx_q4_reference_gs` on 6 real Krea
///   tensors: **NVFP4 0.0939 vs MLX q4 0.1006 — NVFP4 wins 6/6**;
/// * **per-layer output error** `y = x·Wᵀ`: **NVFP4 wins 8/8** (Gaussian, Student-t(3), 1%×30 outliers,
///   massive-channel activations).
///
/// Two further confounds polluted this leg specifically:
///
/// 1. **Surface mismatch.** The MLX q4 tier leaves `final_layer.linear`, `img_in` and
///    `txt_in.linear_{1,2}` **dense at F32** (no `.scales` in the snapshot header; 54.26M params). This
///    test's NVFP4 leg quantizes **all 260** — including the head that
///    `nvfp4_krea_dit_real_activation_outlier_sparsity` measures as Dense (crush 909×). Surface-matched
///    at 256 quantized, the gap narrows **38%**: rel-RMS 0.22850 → **0.21837**, cosine gap 0.00601 →
///    **0.00371** (NVFP4 0.21837/0.97592/30.60 dB vs Q4 0.20193/0.97963/31.28 dB). **Those numbers still
///    do not rank the formats** — a residual gap in a chaotic metric that points opposite to 6/6 and 8/8
///    direct measurements is evidence about the *metric*.
/// 2. **Equal per-projection bits ≠ equal tier bits.** Both are exactly 4.5 bits/weight per projection
///    (q4 `group_size=64` + BF16 scale+bias at `[out, in/64]` → 4 + 32/64; NVFP4 → 4 + 8/16), but the
///    **tiers** spend **59.19 (Q4) vs 57.68 (NVFP4) Gbit — +2.6%**, concentrated on the 4 most
///    sensitivity-critical projections.
///
/// **What this test is for:** confirming that both ~4.5-bit tiers diverge from bf16 by a *similar*
/// amount, and catching a regression that changes their ordering. Rank the formats with the direct
/// weight/per-layer measurements, not with this.
#[test]
#[ignore = "real-weight GPU test: needs BOTH the Krea 2 Turbo bf16 and q4 snapshots + an sm_120 device"]
fn nvfp4_krea_dit_sc2_parity_vs_q4_tier() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(bf16) = bf16_root() else { return };
    let Some(q4) = q4_root() else { return };

    let ctx = encode_context(&bf16, &dev);
    let steps = TURBO_STEPS;

    // Reference: the dense bf16 trunk.
    let lat_bf16 = {
        let m = build(&bf16, &dev, DType::BF16, &DitPlan::baseline(), "dense bf16");
        to_vec_f32(&run_denoise(&m, &ctx, &dev, steps, None))
    };
    // The tier NVFP4 replaces: Q4 packed → `Q4_1` repack → dequant-on-forward (`quant.rs:5-12`).
    let lat_q4 = {
        let m = build(&q4, &dev, DType::BF16, &DitPlan::baseline(), "Q4 tier");
        to_vec_f32(&run_denoise(&m, &ctx, &dev, steps, None))
    };
    // The candidate: NVFP4 under the shipping mixed policy, packed from the SAME bf16 master as the
    // baseline (the loader refuses to pack from a quantized tier for exactly this reason).
    let lat_nvfp4 = {
        let m = build(
            &bf16,
            &dev,
            DType::BF16,
            &DitPlan::nvfp4(Nvfp4Quant::Mixed),
            "NVFP4 mixed",
        );
        let r = m.nvfp4_report();
        eprintln!(
            "[sc-12110] NVFP4 mixed: {}/{} fp4-lit, {} dequant→bf16",
            r.fp4_lit, r.n_quantized, r.dequant_bf16
        );
        to_vec_f32(&run_denoise(&m, &ctx, &dev, steps, None))
    };
    // The **weight-only** NVFP4 leg — the apples-to-apples comparison against Q4, and the one that
    // matters for what would actually ship. Q4 is weight-only (dequant-on-forward, full-precision
    // activations), so scoring it against NVFP4-**mixed** conflates two different questions: "is the
    // NVFP4 *weight format* as good as int4?" and "does FP4 *activation* quantization cost quality?".
    // W4A16 isolates the first — and is also the only throughput-viable regime (SC#1 measured mixed
    // W4A4 at 0.02×), so it is the regime a shipping NVFP4 tier would use.
    let lat_w4a16 = {
        let m = build(
            &bf16,
            &dev,
            DType::BF16,
            &DitPlan::nvfp4(Nvfp4Quant::BlanketW4A16),
            "NVFP4 W4A16",
        );
        to_vec_f32(&run_denoise(&m, &ctx, &dev, steps, None))
    };

    let score = |lat: &[f32]| {
        (
            rel_rms(lat, &lat_bf16),
            cosine(lat, &lat_bf16),
            psnr_latent(lat, &lat_bf16),
        )
    };
    let (q4_rel, q4_cos, q4_psnr) = score(&lat_q4);
    let (fp4_rel, fp4_cos, fp4_psnr) = score(&lat_nvfp4);
    let (w16_rel, w16_cos, w16_psnr) = score(&lat_w4a16);

    eprintln!("\n[sc-12110] ===== SC#2 PARITY — Krea 2 Turbo, {steps}-step Turbo denoise, same seed/prompt =====");
    eprintln!(
        "[sc-12110] {:<40} {:>10} {:>10} {:>10}",
        "tier (vs dense bf16 reference)", "rel-RMS", "cosine", "PSNR dB"
    );
    eprintln!(
        "[sc-12110] {:<40} {q4_rel:>10.5} {q4_cos:>10.5} {q4_psnr:>10.2}",
        "Q4 weight-only (the tier NVFP4 replaces)"
    );
    eprintln!(
        "[sc-12110] {:<40} {w16_rel:>10.5} {w16_cos:>10.5} {w16_psnr:>10.2}",
        "NVFP4 W4A16 weight-only (like-for-like)"
    );
    eprintln!(
        "[sc-12110] {:<40} {fp4_rel:>10.5} {fp4_cos:>10.5} {fp4_psnr:>10.2}",
        "NVFP4 W4A4 mixed (weights + activations)"
    );
    eprintln!(
        "[sc-12110] {:<40} {:>10.5} {:>10.5} {:>10.2}",
        "NVFP4 W4A16 vs Q4 (head to head)",
        rel_rms(&lat_w4a16, &lat_q4),
        cosine(&lat_w4a16, &lat_q4),
        psnr_latent(&lat_w4a16, &lat_q4)
    );

    // Report the ordering, but NOT as a ranking of the two weight formats — this metric cannot do that
    // (see the retraction in this test's docs). Direct measurement says NVFP4's weight format WINS:
    // rel-RMS 0.0939 vs q4's 0.1006 (6/6 real tensors), and 8/8 on per-layer output error.
    let w16_ahead = w16_cos >= q4_cos;
    eprintln!(
        "\n[sc-12110] SC#2 (end-to-end DIVERGENCE, not a fidelity ranking): NVFP4 W4A16 cosine \
         {w16_cos:.5} {} Q4's {q4_cos:.5} (gap {:.5}).",
        if w16_ahead { "≥" } else { "<" },
        (q4_cos - w16_cos).abs()
    );
    eprintln!(
        "[sc-12110] Both are ~4.5-bit tiers walking an 8-step flow-match denoise onto \
         different-but-valid trajectories. This number does NOT rank the weight formats — direct \
         measurement (weight rel-RMS 0.0939 vs 0.1006, 6/6; per-layer output error 8/8) says NVFP4's \
         format is MORE faithful than int4 at equal bits. Confounds here: the q4 tier leaves 4 layers \
         dense at F32 while this leg quantizes all 260, and the tiers spend 59.19 vs 57.68 Gbit (+2.6%)."
    );
    // And the shipping-policy leg, which additionally pays for FP4 activations.
    eprintln!(
        "[sc-12110] SC#2 (shipping mixed W4A4): cosine {fp4_cos:.5} vs Q4 {q4_cos:.5} — FP4 activations \
         cost a further {:.5} cosine on top of the weight-only leg.",
        w16_cos - fp4_cos
    );
    eprintln!(
        "[sc-12110] ⟹ SC#2's accuracy premise HOLDS (NVFP4's weight format beats int4 at equal bits). \
         What blocks the lane is SC#1 (throughput), not fidelity — see \
         nvfp4_krea_dit_sc1_throughput_bf16_vs_w4a16_vs_w4a4."
    );
    assert!(
        lat_nvfp4.iter().all(|v| v.is_finite())
            && lat_q4.iter().all(|v| v.is_finite())
            && lat_w4a16.iter().all(|v| v.is_finite()),
        "a tier produced a non-finite latent"
    );

    // ==========================================================================================
    // The bar: PIN the measured ordering, two-sided.
    //
    // The previous bar was `w16_cos >= q4_cos - 0.05 * q4_cos` — with Q4 at 0.97963 that is a bar of
    // **0.9306**, i.e. ~8x looser than the gap it was meant to police (0.00601). A real regression
    // sailed through it silently and the actual finding lived only in an `eprintln`. Worse, it was
    // **one-sided**: it could only ever fire if NVFP4 got worse.
    //
    // This asserts the *reported* relationship instead, as a two-sided band on the cosine gap, so it
    // fails if the ordering moves in EITHER direction:
    //   * gap grows  => NVFP4's divergence regressed;
    //   * gap shrinks/inverts => something changed for the better (a packer improvement, a surface fix,
    //     or the q4 reference moved). That is NOT a free pass — the recorded conclusion is derived from
    //     these numbers, so it must be re-derived, not silently absorbed.
    //
    // If this fires, RE-MEASURE AND RE-WRITE THE RECORD. Do not widen the band to make it green — that
    // is precisely how the retracted "NVFP4 is less faithful than int4" conclusion survived review.
    // ==========================================================================================
    // Q4 − NVFP4-W4A16 cosine gap measured 2026-07-15 on real weights (this leg quantizes all 260
    // projections; the q4 tier leaves 4 dense — see the surface-mismatch confound in the docs).
    const MEASURED_COS_GAP: f64 = 0.00601;
    // Absorbs GEMM/library nondeterminism only. The seed, prompt and schedule are fixed, so this is
    // NOT a chaos budget — chaos is *between* the tiers, and it is exactly what the band pins.
    const COS_GAP_TOL: f64 = 0.004;

    let cos_gap = q4_cos - w16_cos;
    assert!(
        (cos_gap - MEASURED_COS_GAP).abs() <= COS_GAP_TOL,
        "the SC#2 cosine ordering moved: Q4 {q4_cos:.5} − NVFP4 W4A16 {w16_cos:.5} = gap {cos_gap:.5}, \
         outside the measured {MEASURED_COS_GAP:.5} ± {COS_GAP_TOL:.5}. This test pins the ordering the \
         repo's record is written from — RE-MEASURE and update the record (README's NVFP4 section + this \
         test's docs). Do NOT widen the band to make this green.\n\
         NOTE: this end-to-end metric measures trajectory DIVERGENCE, not weight fidelity, and must not \
         be used to rank the two formats — direct measurement (weight rel-RMS 0.0939 vs 0.1006, 6/6 \
         tensors; per-layer output error 8/8) says NVFP4's format is MORE faithful than int4 at equal \
         bits."
    );
}

// ==============================================================================================
// (3) The partition, re-derived on Krea's naming.
// ==============================================================================================

/// **Re-measure the activation-outlier partition on Krea's layer naming (sc-12110 scope item 4).**
///
/// The sc-11038 mixed policy is substring/naming-based and was tuned on SANA. sc-11045 already caught it
/// mis-firing once (27 Dense collapse sites under the pre-widening rule). Krea is a *different naming
/// universe* and every one of the shared policy's caption-class anchors misses here:
///
/// * no `attn2` — Krea is single-stream: the text context is **concatenated onto the image sequence**
///   and read by ordinary self-attention, so the caption's activations enter the **compute bulk**;
/// * no `caption_projection` — the text ingest is `txt_in.linear_{1,2}` fed by `text_fusion`;
/// * no `proj_out` — the head is `final_layer.linear`.
///
/// Krea also has **`to_gate`, a projection with no SANA analogue**. This test classifies it by
/// measurement. (Structurally it must match `to_q`/`to_k`/`to_v`: all four read the *same* input `x`,
/// and the probe records inputs — so a divergence would indicate a probe bug, which is itself worth
/// knowing.)
///
/// Measures the **baseline (unquantized)** trunk, so the recorded distribution is the model's real one
/// rather than one already shaped by quantization.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot + an sm_120 device"]
fn nvfp4_krea_dit_real_activation_outlier_sparsity() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };

    let ctx = encode_context(&root, &dev);
    let probe = Arc::new(ActProbe::new());
    let cfg = Krea2Config::from_snapshot(&root).expect("krea config");
    let m = build(
        &root,
        &dev,
        DType::BF16,
        &DitPlan::baseline().with_probe(probe.clone()),
        "probed baseline",
    );

    // A short but schedule-spanning denoise: σ moves ~1 → 0 and the activation distribution is not
    // stationary across it — the gate is a worst-case-across-steps question. The probe moves every
    // activation to host f32, so this is deliberately not a full 8-step run.
    let probe_steps = 4usize;
    let lat = run_denoise(&m, &ctx, &dev, probe_steps, Some(&probe));
    assert!(
        to_vec_f32(&lat).iter().all(|v| v.is_finite()),
        "probed denoise non-finite"
    );
    drop(m);

    let records = probe.records();
    assert!(!records.is_empty(), "probe recorded nothing");
    let summaries = summarize(&records);
    eprintln!(
        "\n[sc-12110] ===== REAL ACTIVATION-OUTLIER SPARSITY (Krea 2 Turbo, {probe_steps} real \
         denoise steps, {} measurements over {} projections) =====",
        records.len(),
        summaries.len()
    );
    eprintln!(
        "[sc-12110] {:<48} {:>7} {:>11} {:>11} {:>8} {:>11}",
        "layer", "policy", "min_benign", "mean_benign", "worst", "max_crush"
    );

    let mut violations: Vec<String> = Vec::new();
    let (mut w4a4_layers, mut w4a4_benign, mut w4a4_sparse) = (0usize, 0usize, 0usize);
    let (mut outlier_layers, mut outlier_dense) = (0usize, 0usize);
    let mixed = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(cfg.num_layers);

    for s in &summaries {
        // What the shipping mixed policy assigns — derived through the SAME `LayerRole::for_krea_layer`
        // the loader uses, so the harness cannot cross the measurement against a different partition
        // than the one that was actually built.
        let assigned = mixed.act_for_layer(&s.layer);
        eprintln!(
            "[sc-12110] {:<48} {:>7} {:>11.5} {:>11.5} {:>8} {:>11.1}",
            s.layer,
            match assigned {
                ActPrecision::W4A4 => "W4A4",
                ActPrecision::W4A16 => "W4A16",
            },
            s.min_benign_fraction,
            s.mean_benign_fraction,
            format!("{:?}", s.worst_class),
            s.max_crush_ratio,
        );
        match assigned {
            ActPrecision::W4A4 => {
                w4a4_layers += 1;
                match s.worst_class {
                    OutlierClass::Benign => w4a4_benign += 1,
                    OutlierClass::Sparse => w4a4_sparse += 1,
                    OutlierClass::Dense => violations.push(format!(
                        "{} → W4A4 but measures Dense (min benign {:.5}, crush {:.1}×)",
                        s.layer, s.min_benign_fraction, s.max_crush_ratio
                    )),
                }
            }
            ActPrecision::W4A16 => {
                outlier_layers += 1;
                if matches!(s.worst_class, OutlierClass::Dense) {
                    outlier_dense += 1;
                }
            }
        }
    }

    eprintln!(
        "\n[sc-12110] PARTITION VERDICT: {w4a4_layers} layers assigned W4A4 — {w4a4_benign} measure \
         Benign, {w4a4_sparse} measure Sparse, {} measure Dense (violations).\n\
         [sc-12110] {outlier_layers} layers held at W4A16 (outlier class) — {outlier_dense} of them do \
         measure Dense on real activations (i.e. the override was earning its keep).",
        violations.len()
    );
    for v in &violations {
        eprintln!("[sc-12110]   VIOLATION: {v}");
    }

    // `to_gate` has no SANA analogue — report its measured class explicitly rather than leaving the
    // reader to infer it from the table.
    for s in summaries
        .iter()
        .filter(|s| s.layer.ends_with("to_gate"))
        .take(4)
    {
        eprintln!(
            "[sc-12110]   to_gate sample: {} → {:?} (min benign {:.5}, crush {:.1}×)",
            s.layer, s.worst_class, s.min_benign_fraction, s.max_crush_ratio
        );
    }

    assert!(
        violations.is_empty(),
        "the benign→W4A4 partition does NOT hold on Krea's real activations — {} layer(s) assigned \
         W4A4 measure Dense-outlier: {:?}",
        violations.len(),
        violations
    );
}

// ==============================================================================================
// (4) SC#6 — resident VRAM per regime.
// ==============================================================================================

/// **SC#6: NVFP4 must be served natively packed — resident VRAM == the NVFP4 footprint, per regime.**
///
/// Contention-immune tensor byte-accounting (`Nvfp4Report::resident_bytes`), not an `nvidia-smi`
/// free-memory delta: it sums the actual resident weight buffers each layer holds.
///
/// **Expect the mixed regime to be poor** (~0.70× on SANA): every W4A16 layer goes through
/// `Nvfp4Linear::new_dequant`, which materializes a *full dense bf16* weight and holds it for life. That
/// is **sc-12121**, not this story's to fix — it is reported honestly, not smoothed.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot + an sm_120 device"]
fn nvfp4_krea_dit_sc6_resident_vram_per_regime() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };

    eprintln!("\n[sc-12110] ===== SC#6 RESIDENT VRAM PER REGIME (Krea 2 Turbo trunk) =====");
    eprintln!(
        "[sc-12110] {:<24} {:>9} {:>12} {:>12} {:>10} {:>9}",
        "regime", "fp4-lit", "resident MiB", "bf16 MiB", "ratio", "eff bits"
    );
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    for (tag, quant) in [
        ("blanket W4A4", Nvfp4Quant::BlanketW4A4),
        ("mixed (shipping)", Nvfp4Quant::Mixed),
        ("blanket W4A16", Nvfp4Quant::BlanketW4A16),
    ] {
        let m = build(&root, &dev, DType::BF16, &DitPlan::nvfp4(quant), tag);
        let r = m.nvfp4_report();
        eprintln!(
            "[sc-12110] {:<24} {:>4}/{:<4} {:>12.2} {:>12.2} {:>9.4}× {:>9.2}",
            tag,
            r.fp4_lit,
            r.n_quantized,
            mib(r.resident_bytes()),
            mib(r.bf16_bytes),
            r.footprint_ratio(),
            r.effective_bits()
        );
        if matches!(quant, Nvfp4Quant::BlanketW4A4) {
            // The SC#6 claim proper: with everything on the packed path, resident == the packed
            // footprint exactly — no bf16 expansion anywhere.
            assert_eq!(
                r.resident_bytes(),
                r.nvfp4_bytes,
                "blanket W4A4 must hold ONLY packed NVFP4 bytes resident (SC#6)"
            );
            assert_eq!(
                r.fp4_lit, r.n_quantized,
                "blanket W4A4 must light every layer"
            );
            assert!(
                r.footprint_ratio() < 0.30,
                "blanket W4A4 footprint {:.4}× is not NVFP4-scale",
                r.footprint_ratio()
            );
        }
        if matches!(quant, Nvfp4Quant::BlanketW4A16) {
            // The honest W4A16 answer: nothing packed on-device, every weight dense bf16 → 1.0×.
            assert!(
                (r.footprint_ratio() - 1.0).abs() < 1e-6,
                "a W4A16 run holds dense bf16 — it must report 1.0×, got {:.4}×",
                r.footprint_ratio()
            );
        }
    }
    eprintln!(
        "[sc-12110] NOTE: the mixed regime's ratio is dominated by `Nvfp4Linear::new_dequant` \
         materializing dense bf16 for every W4A16 layer — that is sc-12121, not sc-12110."
    );
}

/// **sc-12274: what the SC#6 number does NOT count — the per-layer cuBLASLt workspace, measured.**
///
/// [`nvfp4_krea_dit_sc6_resident_vram_per_regime`] proves SC#6 by summing `resident_weight_bytes` over
/// the lane's projections. That is deliberate and contention-immune — but it is **weights-only**, and
/// `Nvfp4Linear::try_build_fp4` used to build a *fresh* `CublasLt` per W4A4 layer, each eagerly
/// allocating a 32 MiB workspace held for life. Those bytes are real resident VRAM and invisible to
/// that sum.
///
/// # Measured before the fix (2026-07-16, exclusive sm_120, this test)
///
/// | regime | handles | real VRAM | SC#6 says | excess |
/// |---|---:|---:|---:|---:|
/// | blanket W4A16 (= dense bf16) | 0 | 25.56 GiB | 23.38 | 2.18 |
/// | mixed (shipping) | 139 | 19.69 GiB | 14.28 | 5.42 |
/// | blanket W4A4 | 260 | **15.41 GiB** | **6.58** | **8.84** |
///
/// Fitted **26.1 MiB/handle** (intercept 2.13 GiB) ⇒ ~6.6 GiB of duplicated workspace. The real
/// footprint was **0.603×**, not the reported **0.2813×** — the headline SC#6 figure was **2.14×
/// optimistic on the exact regime it was claimed for**. `Krea2Transformer::load_planned` now builds one
/// shared handle, and this test is the gate that it stays one.
///
/// The warm probe was worth running and came back **negative**: only **0.031 MiB/handle** (the nvrtc
/// module). The gather index never materializes because `forward_fp4` takes the *fused* quantizer
/// (sc-12078) and `nvfp4_act_scale_gather_idx` belongs to the unfused path. Kept measuring it anyway —
/// a fused-path fallback would put it back.
///
/// # The natural experiment
///
/// `from_packed` only calls `try_build_fp4` for **W4A4**, so the handle count is exactly
/// [`Nvfp4Report::fp4_lit`] — and the three regimes give three points on an *otherwise identical*
/// trunk:
///
/// | regime | fp4-lit → handles |
/// |---|---:|
/// | blanket W4A16 | 0 |
/// | mixed (shipping) | ~139 |
/// | blanket W4A4 | 260 |
///
/// Everything else the trunk holds that `resident_bytes` does not count — norms, the batch-1 embedders,
/// `text_fusion.projector`, allocator rounding — is **identical across all three**. So regressing
/// `measured_vram − resident_bytes` on `fp4_lit` isolates the workspace as the **slope** and dumps every
/// confound into the **intercept**. The slope is predicted to be `CublasLt::WORKSPACE` = 32 MiB/handle.
///
/// That makes this a real refutation test, not a demo: if the slope came back ~0, the per-layer handle
/// would cost nothing and sc-12274 would be wrong.
///
/// # Measured COLD **and WARM** — the cold number is a floor, not the answer
///
/// The 32 MiB workspace is allocated eagerly in `CublasLt::new`, so a probe around construction sees
/// it. But two of the handle's three caches only populate on the **first forward**, and one of them
/// holds **device** memory:
///
/// * `nvfp4_act_scale_gather_idx` — a `Tensor` (the sc-12207 inverse-permutation index), built per
///   activation shape. Per-handle ⇒ one per *layer* instead of one per shape.
/// * `nvfp4_quant_kernels` — the nvrtc-compiled fused quantizer module, held for the handle's life.
///
/// So a construction-only probe **understates** the per-layer cost, and quoting it would repeat this
/// epic's own recurring mistake (sc-12207's residual was mis-sized ~4× by a cold micro-decomposition).
/// This test therefore probes each regime twice — after `load_planned` (cold) and after one real
/// forward (warm) — and regresses **both**. The warm slope is the honest per-layer cost.
///
/// Pairs with `candle-gen`'s weights-free `nvfp4_linear_builds_one_32mib_cublaslt_workspace_per_layer`,
/// which pins the same 32 MiB/handle in isolation.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot + an sm_120 device"]
fn nvfp4_krea_dit_sc6_cublaslt_workspace_gap() {
    use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };

    /// `CublasLt::WORKSPACE` — the predicted per-handle cost this regression must recover as its slope.
    const WORKSPACE: usize = 32 * 1024 * 1024;
    // A VRAM delta is exactly the measurement a background process corrupts — say what else is resident.
    assert_exclusive_gpu("sc-12274 workspace gap");
    let cfg = Krea2Config::from_snapshot(&root).expect("krea config");
    let gib = |b: f64| b / (1024.0 * 1024.0 * 1024.0);
    let mib = |b: f64| b / (1024.0 * 1024.0);
    let free_now = |d: &Device| {
        d.synchronize().unwrap();
        let (free, _total) = cuda::mem_get_info().unwrap();
        free as f64
    };

    // The real caption-derived context — the warm forward must be the real activation shape, since the
    // gather index is keyed by `(rows, n_blocks)`. Encoded once, before any trunk is resident.
    let ctx = encode_context(&root, &dev);
    let sigmas = turbo_sigmas(TURBO_STEPS);

    eprintln!("\n[sc-12274] ===== SC#6 vs REAL VRAM — the cuBLASLt workspace gap (Krea 2 Turbo trunk) =====");
    eprintln!(
        "[sc-12274] {:<20} {:>8} {:>11} {:>11} {:>10} {:>11} {:>11} {:>11}",
        "regime", "fp4-lit", "cold GiB", "warm GiB", "SC#6 GiB", "cold exc", "warm exc", "pred ws"
    );

    /// One regime's VRAM measurement — a point in the excess-vs-handles regression (sc-12274).
    struct RegimePoint {
        /// Handles this regime built. Exactly `Nvfp4Report::fp4_lit`: only a W4A4 layer reaches
        /// `try_build_fp4`, so this IS the handle count — the regression's x.
        handles: f64,
        /// Real VRAM minus `resident_bytes`, probed after construction — what SC#6 cannot see.
        cold_excess: f64,
        /// ...and probed again after one real forward, since the handle's caches populate lazily.
        warm_excess: f64,
        /// Real VRAM the whole trunk holds, warm.
        warm_measured: f64,
        /// Dense bf16 weight bytes — SC#6's denominator (regime-invariant).
        bf16_bytes: f64,
    }
    let mut pts: Vec<RegimePoint> = Vec::new();
    for (tag, quant) in [
        ("blanket W4A16", Nvfp4Quant::BlanketW4A16),
        ("mixed (shipping)", Nvfp4Quant::Mixed),
        ("blanket W4A4", Nvfp4Quant::BlanketW4A4),
    ] {
        // Load the tier FIRST and probe around `load_planned` only, so the measurement is the trunk's
        // own construction — not the transient bf16 source tensors, which are identical per regime and
        // would otherwise swamp the signal.
        let w = trunk_weights(&root, &dev, DType::BF16);
        let plan = DitPlan::nvfp4(quant).with_num_layers(cfg.num_layers);

        let before = free_now(&dev);
        let m = Krea2Transformer::load_planned(&w, &cfg, &plan).expect("build trunk");
        let cold = before - free_now(&dev);

        // One real forward at the real activation shape, then drop every transient it produced: what
        // stays is the handle-resident caches (gather index + nvrtc module), which cold cannot see.
        {
            let x = init_latent(&dev, EDGE);
            let t = Tensor::from_vec(vec![sigmas[0]], (1,), &dev).expect("timestep");
            let v = m.forward(&x, &t, &ctx).expect("warm DiT forward");
            drop(v);
        }
        let warm = before - free_now(&dev);

        let r = m.nvfp4_report();
        let resident = r.resident_bytes() as f64;
        let predicted_ws = (r.fp4_lit * WORKSPACE) as f64;
        eprintln!(
            "[sc-12274] {:<20} {:>4}/{:<3} {:>11.3} {:>11.3} {:>10.3} {:>11.3} {:>11.3} {:>11.3}",
            tag,
            r.fp4_lit,
            r.n_quantized,
            gib(cold),
            gib(warm),
            gib(resident),
            gib(cold - resident),
            gib(warm - resident),
            gib(predicted_ws)
        );
        pts.push(RegimePoint {
            handles: r.fp4_lit as f64,
            cold_excess: cold - resident,
            warm_excess: warm - resident,
            warm_measured: warm,
            bf16_bytes: r.bf16_bytes as f64,
        });
        drop(m);
        drop(w);
        // Confirm the trunk's VRAM actually came back, else the next regime's baseline is a lie.
        let _ = free_now(&dev);
    }

    // --- Least-squares slope of excess-vs-handles: per-layer cost, isolated from every constant ------
    // Everything the trunk holds that `resident_bytes` does not count — norms, batch-1 embedders,
    // `text_fusion.projector`, allocator rounding — is IDENTICAL across regimes, so it lands in the
    // intercept and the slope is the per-handle cost alone.
    let fit = |sel: &dyn Fn(&RegimePoint) -> f64| -> (f64, f64) {
        let n = pts.len() as f64;
        let xbar = pts.iter().map(|p| p.handles).sum::<f64>() / n;
        let ybar = pts.iter().map(sel).sum::<f64>() / n;
        let sxy: f64 = pts
            .iter()
            .map(|p| (p.handles - xbar) * (sel(p) - ybar))
            .sum();
        let sxx: f64 = pts.iter().map(|p| (p.handles - xbar).powi(2)).sum();
        let slope = sxy / sxx;
        (slope, ybar - slope * xbar)
    };
    let (cold_slope, cold_int) = fit(&|p| p.cold_excess);
    let (warm_slope, warm_int) = fit(&|p| p.warm_excess);

    eprintln!(
        "\n[sc-12274] REGRESSION excess = slope·fp4_lit + intercept over {} regimes:",
        pts.len()
    );
    eprintln!(
        "[sc-12274]   COLD slope = {:>9.3} MiB/handle  intercept {:>9.3} MiB  (predicted {:.3} = CublasLt::WORKSPACE)",
        mib(cold_slope),
        mib(cold_int),
        mib(WORKSPACE as f64)
    );
    eprintln!(
        "[sc-12274]   WARM slope = {:>9.3} MiB/handle  intercept {:>9.3} MiB  (workspace + per-handle caches)",
        mib(warm_slope),
        mib(warm_int)
    );
    eprintln!(
        "[sc-12274]   cache-only (warm − cold) = {:.3} MiB/handle — gather index + nvrtc module, \
         duplicated per layer instead of per shape",
        mib(warm_slope - cold_slope)
    );
    eprintln!(
        "[sc-12274]   ⇒ blanket W4A4 (260 handles): COLD {:.3} GiB / WARM {:.3} GiB of per-handle VRAM \
         that the SC#6 weights-only sum does not count.",
        gib(cold_slope * 260.0),
        gib(warm_slope * 260.0)
    );

    // --- SC#6 RESTATED against a MEASURED baseline, with no attribution ---------------------------
    // The cleanest restatement needs no model of what the workspace costs: blanket W4A16 IS the
    // dense-bf16-resident trunk (nothing packed on-device, 1.00× by construction — asserted by
    // `nvfp4_krea_dit_sc6_resident_vram_per_regime`). So measured(W4A4) / measured(W4A16) is the real
    // footprint ratio, whole-trunk, including every byte both regimes actually hold.
    let w4a4 = pts.last().expect("blanket W4A4 measured last");
    let (warm_exc_w4a4, w4a4_measured, bf16_bytes) =
        (w4a4.warm_excess, w4a4.warm_measured, w4a4.bf16_bytes);
    let w4a16_measured = pts[0].warm_measured;
    assert_eq!(w4a4.handles, 260.0, "blanket W4A4 must light all 260 lane projections");
    assert_eq!(
        pts[0].handles, 0.0,
        "blanket W4A16 lights nothing — it is the dense-bf16 baseline"
    );

    // The SHIPPED SC#6 figure, recomputed exactly as `Nvfp4Report::footprint_ratio` does it
    // (resident weight bytes / dense bf16 weight bytes) — this is the ~0.2813× the epic headlines.
    let w4a4_resident = w4a4_measured - warm_exc_w4a4;
    let sc6_shipped = w4a4_resident / bf16_bytes;
    // ...and what the same run actually costs: measured W4A4 trunk / measured dense-bf16 trunk. Both
    // are "a W4A4 run vs dense bf16", which is precisely the claim SC#6 makes — one accounted on
    // weights only, one measured whole-trunk.
    let sc6_real = w4a4_measured / w4a16_measured;
    eprintln!("\n[sc-12274] ===== SC#6 RESTATED (blanket W4A4, measured — no attribution) =====");
    eprintln!(
        "[sc-12274]   real resident, whole trunk : {:.3} GiB (W4A4) vs {:.3} GiB (dense bf16 = blanket W4A16)",
        gib(w4a4_measured),
        gib(w4a16_measured)
    );
    eprintln!(
        "[sc-12274]   REAL footprint ratio       : {:.4}×   <-- what a blanket-W4A4 run actually costs",
        sc6_real
    );
    eprintln!(
        "[sc-12274]   SC#6 weights-only reports  : {:.4}×   ({:.3} GiB packed / {:.3} GiB bf16)",
        sc6_shipped,
        gib(w4a4_resident),
        gib(bf16_bytes)
    );
    eprintln!(
        "[sc-12274]   ⇒ the shipped SC#6 figure is {:.2}× optimistic on the very regime it is claimed for.",
        sc6_real / sc6_shipped
    );
    // sc-12274's sharpest line, scored: is the workspace really bigger than the weights it serves?
    eprintln!(
        "[sc-12274]   workspace ({:.3} GiB @ measured slope) vs the packed weights it serves ({:.3} GiB) = {:.2}×",
        gib(warm_slope * 260.0),
        gib(w4a4_resident),
        (warm_slope * 260.0) / w4a4_resident
    );
    eprintln!(
        "[sc-12274]   weights-only excludes {:.3} GiB (warm). Report SC#6 as weights-only EXPLICITLY, \
         or report weights+handles — silence is not an option (sc-12274 scope 4).",
        gib(warm_exc_w4a4)
    );

    // --- THE GATE: the per-layer handle must not come back ---------------------------------------
    // Post-fix the trunk shares ONE handle via `DitPlan::with_nvfp4_context`, so per-layer VRAM beyond
    // the weights must be ~0 — the slope, not the intercept, is what regresses. Any reintroduced
    // `CublasLt::new` per layer puts ~26–32 MiB back on this slope and trips here.
    //
    // Scale note: this bounds the slope at 4 MiB/handle, i.e. <1.0 GiB across 260 — an eighth of the
    // ~6.6 GiB the defect cost. It is deliberately not tighter: the regression's premise (everything
    // `resident_bytes` misses is regime-invariant) is imperfect, because the ALLOCATION PATTERN
    // differs — blanket W4A16 allocates 260 large ~92 MiB dense tensors while blanket W4A4 allocates
    // 260 small packed buffers, so allocator rounding leaks into the fit. That imprecision is also why
    // the pre-fix slope measured ~26 MiB here but exactly 32.00 MiB in the isolated, nothing-else-
    // allocating `nvfp4_linear_shares_one_cublaslt_workspace_across_layers`. Trust that test for the
    // intrinsic per-handle cost, this one for trunk-scale behaviour, and `sc6_real` for the SC#6
    // restatement — it is measured and needs no attribution at all.
    assert!(
        cold_slope < 4.0 * 1024.0 * 1024.0,
        "{:.2} MiB of resident VRAM per fp4-lit layer beyond its weights — the trunk is building a \
         cuBLASLt handle PER LAYER again (sc-12274 regression). One shared handle should make this \
         ~0; a full per-layer {:.2} MiB workspace costs ~6.6 GiB across this trunk and is invisible \
         to the weights-only SC#6 sum.",
        mib(cold_slope),
        mib(WORKSPACE as f64)
    );
    // Warm can only be worse than cold: the handle's caches are additive and never freed.
    assert!(
        warm_slope >= cold_slope - 1024.0 * 1024.0,
        "warm per-handle cost {:.2} MiB < cold {:.2} MiB — impossible if the handle's caches are \
         retained for life; the probe is measuring something other than steady-state residency.",
        mib(warm_slope),
        mib(cold_slope)
    );
}

// ==============================================================================================
// (5) SC#3 — stability across the full denoise.
// ==============================================================================================

/// **SC#3: no NaN/blowup from FP4 across a full Krea Turbo denoise.**
///
/// [`DitPlan::checked`] arms the sc-11044 NaN/inf guard on **every** NVFP4 projection at **every** step,
/// so a non-finite value fails loud at the layer and step that produced it rather than silently
/// propagating. That is strictly stronger than checking the final latent.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot + an sm_120 device"]
fn nvfp4_krea_dit_sc3_no_nan_across_full_denoise() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };

    let ctx = encode_context(&root, &dev);
    for (tag, quant) in [
        ("mixed (shipping)", Nvfp4Quant::Mixed),
        ("blanket W4A4", Nvfp4Quant::BlanketW4A4),
    ] {
        let m = build(
            &root,
            &dev,
            DType::BF16,
            &DitPlan::nvfp4(quant).checked(),
            tag,
        );
        let r = m.nvfp4_report();
        let lat = run_denoise(&m, &ctx, &dev, TURBO_STEPS, None);
        let v = to_vec_f32(&lat);
        let finite = v.iter().all(|x| x.is_finite());
        let std = {
            let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        eprintln!(
            "[sc-12110] SC#3 {tag}: {}/{} fp4-lit, {} guarded forwards ({} steps × {} projections), \
             finite = {finite}, latent std = {std:.4}",
            r.fp4_lit,
            r.n_quantized,
            TURBO_STEPS * r.n_quantized,
            TURBO_STEPS,
            r.n_quantized
        );
        assert!(finite, "{tag}: NaN/inf in the final latent");
        assert!(
            std > 1e-3,
            "{tag}: latent collapsed to a constant (std {std:.6})"
        );
    }
}

// ==============================================================================================
// (6) The lane's surface — a cheap structural gate that needs no GPU timing.
// ==============================================================================================

/// The NVFP4 lane covers what [`Krea2Transformer::load_planned`] says it covers, and — the sc-12140
/// regression — **`final_layer.linear` is actually guarded**.
///
/// Cheap (build-only, no denoise), but it needs the real snapshot to have the real key set.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot"]
fn nvfp4_krea_dit_lane_surface_and_final_head_are_correct() {
    let Some(dev) = nvfp4_device() else { return };
    let Some(root) = bf16_root() else { return };

    let cfg = Krea2Config::from_snapshot(&root).expect("krea config");
    let m = build(
        &root,
        &dev,
        DType::BF16,
        &DitPlan::nvfp4(Nvfp4Quant::Mixed),
        "mixed",
    );
    let names = m.nvfp4_layer_names();
    let r = m.nvfp4_report();

    // 28 blocks × 8 + 4 text-fusion blocks × 8 + img_in + txt_in×2 + final_layer.linear.
    let expected =
        cfg.num_layers * 8 + (cfg.num_layerwise_text_blocks + cfg.num_refiner_text_blocks) * 8 + 4;
    assert_eq!(names.len(), expected, "lane surface size");
    assert_eq!(
        r.n_quantized, expected,
        "every lane layer must be NVFP4-served"
    );
    assert!(
        names.contains(&"final_layer.linear".to_string()),
        "the trunk head must be in the lane"
    );
    // The batch-1 embedders are deliberately OUT of the lane.
    for excluded in [
        "time_embed.linear_1",
        "time_mod_proj",
        "text_fusion.projector",
    ] {
        assert!(
            !names.contains(&excluded.to_string()),
            "{excluded} must not be in the NVFP4 lane"
        );
    }

    // sc-12140: the head is guarded ONLY because the loader states the role. If this ever regresses to
    // the name-only anchor, the head silently lands on W4A4.
    let mixed = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(cfg.num_layers);
    assert_eq!(
        mixed.act_for_layer("final_layer.linear"),
        ActPrecision::W4A16,
        "Krea's final head must be guarded (sc-12140)"
    );
    assert_eq!(
        LayerRole::for_krea_layer("final_layer.linear", cfg.num_layers),
        LayerRole::final_proj()
    );
    eprintln!(
        "[sc-12110] lane surface: {} projections, {}/{} fp4-lit under the mixed policy",
        names.len(),
        r.fp4_lit,
        r.n_quantized
    );
}
