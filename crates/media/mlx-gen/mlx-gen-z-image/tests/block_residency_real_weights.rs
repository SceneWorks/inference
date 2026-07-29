//! SC-15754 / SC-15510 — the real-weight measurements for ladder rungs 4 (bounded transformer
//! residency) and 2 (the widened decode ladder) on the MLX Z-Image lane.
//!
//! `#[ignore]`d: needs a real Z-Image-Turbo snapshot (`ZIMAGE_SNAPSHOT`, a pre-quantized tier dir for
//! q4/q8) and an Apple/Metal GPU. Run:
//!
//! ```text
//! ZIMAGE_SNAPSHOT=<snapshot or tier dir> \
//!   cargo test -p mlx-gen-z-image --release --test block_residency_real_weights -- --ignored --nocapture
//! ```
//!
//! Constrained-host arm: add `SCENEWORKS_MLX_MEMORY_CAP_GB=6` (see
//! [`mlx_gen::memory::apply_memory_cap_env`] for exactly what that does and does not emulate).
//! Tier: `ZIMAGE_TIER=q4|q8|bf16`.
//!
//! ## What this file is for
//!
//! `block_window_dit.rs` proves the windowed forward is *correct* and really streams; it cannot prove
//! it *saves*, because the fixture stack is 2 blocks of 96 dims. The saving is a property of the real
//! 30-block, ~97 MiB/block DiT and can only be measured here.
//!
//! Every printed number is a measurement. Where a figure is a bound rather than an estimate — the
//! constrained-host re-materialization cost in particular — it says so.

mod common;

use common::snapshot;
use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The tier under test. Unlike rung 3 (whose saving is activation-side and tier-agnostic), rung 4
/// bounds *weights*, so it is worth exactly `(1 - window/30) × transformer bytes` and therefore
/// roughly twice as much on q8 as on q4. q4 is the default because it is the tier the 8 GB question
/// has always been about; `ZIMAGE_TIER=q8` is the one the story asks to answer.
fn tier() -> Option<Quant> {
    match std::env::var("ZIMAGE_TIER").as_deref() {
        Ok("q8") => Some(Quant::Q8),
        Ok("bf16") => None,
        _ => Some(Quant::Q4),
    }
}

fn tier_label() -> String {
    match tier() {
        Some(q) => format!("{q:?}").to_lowercase(),
        None => "bf16".to_owned(),
    }
}

fn spec_with(policy: OffloadPolicy) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.offload_policy = policy;
    if let Some(q) = tier() {
        spec = spec.with_quant(q);
    }
    spec
}

fn spec() -> LoadSpec {
    // Rung 4 composes INSIDE rung 1: `run_staged` sheds the encoder and then the DiT between phases,
    // and the window bounds residency *within* the denoise phase. Both are active together, which is
    // also the only configuration in which a window saves anything (see `pipeline::resolve_block_window`).
    spec_with(OffloadPolicy::Sequential)
}

/// The ladder as a request, at an explicit window. `None` window = rung 3 (the previous ladder top).
fn memory_at(window: Option<u32>) -> GenerationMemory {
    GenerationMemory {
        tile_vae_decode: true,
        chunk_attention: true,
        stream_transformer_blocks: window.is_some(),
        transformer_window_size: window,
        ..Default::default()
    }
}

fn request(memory: Option<GenerationMemory>) -> GenerationRequest {
    let size = env_u32("ZIMAGE_SIZE", 1024);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_STEPS", 4)),
        memory,
        ..Default::default()
    }
}

/// Per-phase peaks plus wall-clock, for one generation.
///
/// The phase sampling mirrors `bounded_attention_real_weights.rs`: MLX's `get_peak_memory` is a
/// high-water mark of ACTIVE bytes, so each boundary is read *before* the counter is reset, and the
/// request peak is the max over all three phases rather than accidentally denoise+decode.
struct Run {
    conditioning_bytes: usize,
    denoise_bytes: usize,
    decode_bytes: usize,
    denoise_secs: f64,
    steps: u32,
    image: Image,
}

impl Run {
    fn request_bytes(&self) -> usize {
        self.conditioning_bytes
            .max(self.denoise_bytes)
            .max(self.decode_bytes)
    }
    fn secs_per_step(&self) -> f64 {
        self.denoise_secs / f64::from(self.steps.max(1))
    }
}

fn run_with(spec: &LoadSpec, memory: Option<GenerationMemory>) -> Run {
    let generator = mlx_gen_z_image::load(spec).expect("load z_image_turbo");
    let req = request(memory);

    clear_cache();
    reset_peak_memory();

    let mut conditioning_bytes = 0usize;
    let mut denoise_bytes = 0usize;
    let mut denoise_start: Option<std::time::Instant> = None;
    let mut denoise_secs = 0.0;
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current: 1, .. } => {
            conditioning_bytes = get_peak_memory();
            reset_peak_memory();
            denoise_start = Some(std::time::Instant::now());
        }
        Progress::Decoding if denoise_bytes == 0 => {
            denoise_secs = denoise_start
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            denoise_bytes = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };

    let out = generator
        .generate(&req, &mut on_progress)
        .expect("generation");
    let decode_bytes = get_peak_memory();
    let image = match out {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    assert!(
        conditioning_bytes > 0 && denoise_bytes > 0 && decode_bytes > 0,
        "a phase boundary was not sampled"
    );
    Run {
        conditioning_bytes,
        denoise_bytes,
        decode_bytes,
        denoise_secs,
        steps: req.steps.unwrap_or(4),
        image,
    }
}

fn run(memory: Option<GenerationMemory>) -> Run {
    run_with(&spec(), memory)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

/// `(max|Δ|, mean|Δ|)` over the RGB8 channels.
fn image_delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height), "image dims");
    let mut max = 0u8;
    let mut sum = 0u64;
    for (x, y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

/// Announce whether a constrained-host cap is in force, so a printed table can never be mistaken for
/// the wrong host.
fn host_banner() -> String {
    match mlx_gen::memory::apply_memory_cap_env() {
        Some(cap) => format!("CONSTRAINED host emulation: MLX limit {cap:.1} GiB"),
        None => "UNCONSTRAINED host (no MLX memory cap)".to_owned(),
    }
}

// ────────────────────────────────────────────────────────────────────────────────────────────────

/// **AC: window sweep per tier, peak and per-step cost, on both host classes.**
///
/// ## What this sweep concluded, against its own premise
///
/// It was written to pick a production window off a `1/2/4/8/15` ladder. The attribution control —
/// a window of 30, which walks the identical driver and bounds **nothing** — landed on exactly the
/// same peak as window 8. Every window does. The window size is inert on this lane, because the
/// family's block loop materializes one block, runs it, and drops it before advancing, and that drop
/// is a real release: rung 3's per-chunk `eval` has already evaluated the carried activation inside
/// the block's own attention, so nothing in the lazy graph is still holding the block's weights.
///
/// So the rung is real (4.653 → 1.795 GiB against the resident trunk) but the *parameter* is not, and
/// [`TRANSFORMER_WINDOW_SIZES`](mlx_gen_z_image::memory_strategy::TRANSFORMER_WINDOW_SIZES) publishes the
/// single exact value rather than four rows that differ by noise.
///
/// This test therefore has two jobs now: prove the saving against the resident baseline, and prove the
/// declaration by showing the peak is flat across every cadence including the never-bounds control. A
/// future change that made the window matter would fail the flatness check — which is the right
/// failure, because the published domain would then need to widen.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn the_window_sweep_bounds_the_denoise_peak() {
    /// The declaration under test: at most this many blocks are ever held. Anything the sweep shows
    /// to be indistinguishable from it is not a separate candidate.
    const DECLARED: u32 = mlx_gen_z_image::memory_strategy::TRANSFORMER_WINDOW_SIZE;
    const N_BLOCKS: u32 = 30;

    println!(
        "\n== rung-4 window sweep — tier {} — {}",
        tier_label(),
        host_banner()
    );
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>10}",
        "window", "denoise GiB", "request GiB", "s/step"
    );

    // Baseline: the previous ladder top (rungs 1-3), i.e. the resident 30-block trunk.
    let baseline = run(Some(memory_at(None)));
    println!(
        "  {:>10}  {:>12.3}  {:>12.3}  {:>10.3}",
        "resident",
        gib(baseline.denoise_bytes),
        gib(baseline.request_bytes()),
        baseline.secs_per_step()
    );

    // The declared window, every intermediate cadence, and the never-bounds control. `N_BLOCKS` is
    // reachable only through the raw request knob: the contract does not advertise it, precisely
    // because one all-covering window bounds nothing.
    let mut rows: Vec<(u32, Run)> = Vec::new();
    for window in [N_BLOCKS, 15, 8, 4, 2, DECLARED] {
        let r = run(Some(memory_at(Some(window))));
        println!(
            "  {:>10}  {:>12.3}  {:>12.3}  {:>10.3}{}",
            window,
            gib(r.denoise_bytes),
            gib(r.request_bytes()),
            r.secs_per_step(),
            if window == N_BLOCKS {
                "   <- control: one all-covering window (bounds nothing)"
            } else if window == DECLARED {
                "   <- the declared candidate"
            } else {
                ""
            }
        );
        // The image must be unchanged at every window — a saving that costs quality is not a rung.
        let (max, mean) = image_delta(&r.image, &baseline.image);
        assert!(
            max <= 1,
            "window {window} changed the image (max channel delta {max}, mean {mean:.4})"
        );
        rows.push((window, r));
    }

    // 1. The rung is real: every setting, including the never-bounds control, cuts the resident peak.
    for (window, r) in &rows {
        assert!(
            r.denoise_bytes < baseline.denoise_bytes,
            "window {window} ({:.3} GiB) does not cut the resident denoise peak ({:.3} GiB)",
            gib(r.denoise_bytes),
            gib(baseline.denoise_bytes)
        );
    }
    let declared = rows
        .iter()
        .find(|(w, _)| *w == DECLARED)
        .map(|(_, r)| r)
        .expect("the declared window is swept");
    println!(
        "\n  rung 4 vs resident at the declared window {DECLARED}: {:.3} -> {:.3} GiB ({:.0}%)",
        gib(baseline.denoise_bytes),
        gib(declared.denoise_bytes),
        (declared.denoise_bytes as f64 / baseline.denoise_bytes as f64 - 1.0) * 100.0
    );

    // 2. The DECLARATION is exact: no cadence — not even the one that bounds nothing — holds
    //    materially more than the declared window. If this fails, residency really does depend on the
    //    window and `TRANSFORMER_WINDOW_SIZES` must widen to a real ladder again.
    let control = rows
        .iter()
        .find(|(w, _)| *w == N_BLOCKS)
        .map(|(_, r)| r)
        .expect("the control is swept");
    // The granularity of this rung is ONE BLOCK, so the noise floor for "indistinguishable" has to be
    // a fraction of a block — and a block is tier-dependent (q4 ~97 MiB, q8 ~190 MiB, bf16 larger
    // still). A hardcoded floor is therefore a tier-specific test that silently passes on q4 and fails
    // on q8 for no real reason, which is exactly what the first version of this assertion did.
    //
    // Derive it from THIS run instead: going from the declared 1-block window to the resident stack
    // adds `n_blocks - 1` blocks, so the difference over 29 is the per-block size at this tier. It
    // self-checks — the derivation lands within a few percent of the header-computed 97.1 MiB (q4) and
    // ~190 MiB (q8), which is why it is trustworthy as a scale rather than merely convenient.
    let per_block = (baseline
        .denoise_bytes
        .saturating_sub(declared.denoise_bytes)) as f64
        / f64::from(N_BLOCKS - 1);
    let slack = (per_block * 0.75) as i64;
    println!(
        "  derived per-block size at this tier: {:.1} MiB (slack {:.1} MiB)",
        per_block / (1024.0 * 1024.0),
        slack as f64 / (1024.0 * 1024.0)
    );
    for (window, r) in &rows {
        let excess = r.denoise_bytes as i64 - declared.denoise_bytes as i64;
        assert!(
            excess < slack,
            "window {window} holds {:.3} GiB against the declared window {DECLARED}'s {:.3} GiB — a \
             difference of {:.1} MiB, which is more than {:.0}% of one block ({:.1} MiB). Residency \
             DOES depend on the window at this tier, so the published domain must widen from \
             [{DECLARED}] back to a measured ladder.",
            gib(r.denoise_bytes),
            gib(declared.denoise_bytes),
            excess as f64 / (1024.0 * 1024.0),
            75.0,
            per_block / (1024.0 * 1024.0)
        );
    }
    println!(
        "  declaration check: the never-bounds control holds {:.3} GiB vs the declared window's \
         {:.3} GiB ({:.1} MiB apart, under one block) — residency is per-BLOCK by construction, so \
         [{DECLARED}] is the exact domain",
        gib(control.denoise_bytes),
        gib(declared.denoise_bytes),
        (control.denoise_bytes as i64 - declared.denoise_bytes as i64) as f64 / (1024.0 * 1024.0)
    );

    // 3. Where the per-block release comes from. Rung 4 is cumulative over rung 3, so production
    //    always has the per-chunk `eval` that cuts the graph. Running rung 4 WITHOUT it attributes the
    //    release: if the peak climbs here, the window would matter on a hypothetical rung-3-less path,
    //    which is worth knowing and is why the declaration is scoped to the cumulative ladder.
    let no_rung3 = |window: Option<u32>| {
        run(Some(GenerationMemory {
            tile_vae_decode: true,
            chunk_attention: false,
            stream_transformer_blocks: window.is_some(),
            transformer_window_size: window,
            ..Default::default()
        }))
    };
    let bare_resident = no_rung3(None);
    let bare_declared = no_rung3(Some(DECLARED));
    let bare_control = no_rung3(Some(N_BLOCKS));
    println!(
        "\n  without rung 3 (attribution only, not a selectable ladder): resident {:.3}, window \
         {DECLARED} {:.3}, never-bounds control {:.3} GiB",
        gib(bare_resident.denoise_bytes),
        gib(bare_declared.denoise_bytes),
        gib(bare_control.denoise_bytes)
    );
    assert!(
        bare_declared.denoise_bytes < bare_resident.denoise_bytes,
        "rung 4 must bound the trunk even without rung 3"
    );

    // Which phase binds after rung 4 is a *result*, not a constant — it is the decode on q4 and the
    // conditioning phase on q8 — so it is derived rather than asserted into the message.
    let binding = [
        ("conditioning", declared.conditioning_bytes),
        ("denoise", declared.denoise_bytes),
        ("decode", declared.decode_bytes),
    ]
    .into_iter()
    .max_by_key(|(_, bytes)| *bytes)
    .expect("three phases");
    assert_ne!(
        binding.0, "denoise",
        "rung 4 bounds the denoise phase, so it must no longer be the staged maximum"
    );
    println!(
        "\n  After rung 4 the staged maximum moves: conditioning {:.3} / denoise {:.3} / decode \
         {:.3} GiB — the binding phase is now the {}. Report to SC-15611 and SC-15614.",
        gib(declared.conditioning_bytes),
        gib(declared.denoise_bytes),
        gib(declared.decode_bytes),
        binding.0.to_uppercase(),
    );
}

/// **AC: output numerically identical to the resident path on real weights.**
///
/// Whole-request identity at the production window, at the same seed. The image is the contract's
/// output; the assertion is on it, not on an intermediate.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn the_windowed_render_matches_the_resident_render() {
    println!(
        "\n== rung-4 identity — tier {} — {}",
        tier_label(),
        host_banner()
    );
    let resident = run(Some(memory_at(None)));
    for window in [1u32, 4] {
        let windowed = run(Some(memory_at(Some(window))));
        let (max, mean) = image_delta(&windowed.image, &resident.image);
        println!("  window {window}: max channel delta {max}/255, mean {mean:.5}");
        // BIT-IDENTICAL, not "close". A streamed block runs the same ops on the same weights, so the
        // measured delta is 0/255 at every window and tier — and asserting the measurement rather than
        // a tolerance is what makes this test able to notice a real divergence.
        assert_eq!(
            max, 0,
            "window {window} changed the image (max {max}, mean {mean:.5}) — a streamed block must be \
             byte-identical to its resident twin"
        );
    }
}

/// **AC: cancellation mid-window and an injected error leave no residual cache growth across
/// repeats.**
///
/// ## Why this measures RETAINED bytes and not a later run's peak
///
/// The obvious instrument — "run again and compare the peak" — is the wrong one here, and this test
/// originally used it and produced a false positive: a fault at `Conditioning` read as **+9.4%
/// residual**. It was not residue. Under `Sequential` the conditioning phase covers the text-encoder
/// load, the encode, the encoder drop AND the heavy (DiT + VAE) load, and its peak legitimately varies
/// run to run with allocator reuse — 3.602 to 4.291 GiB was observed across otherwise identical clean
/// runs. Any threshold tight enough to catch a real leak is inside that noise.
///
/// So the assertion is on **retained active bytes after the failed run is dropped and the cache
/// cleared**, which is the thing "leaves no residual cache growth" actually means. MLX's
/// `get_active_memory` is live bytes, not a high-water mark, so a request that failed to release
/// something shows up directly rather than as a statistical shift in a later run. The two stable
/// phases (denoise, decode) are still compared as a cross-check, and the noisy conditioning phase is
/// printed but not asserted on — with its measured spread, so the choice is visible rather than
/// tuned-until-green.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn a_canceled_or_errored_windowed_run_leaves_no_residual_cache() {
    use mlx_gen::gen_core::MemoryPhase;
    use mlx_rs::memory::get_active_memory;
    println!(
        "\n== rung-4 cleanup — tier {} — {}",
        tier_label(),
        host_banner()
    );
    let window = Some(1u32);

    /// Bytes still live after a generator is gone and the allocator cache is flushed. Anything a
    /// failed request failed to release is in here.
    fn retained_after(run_it: impl FnOnce()) -> usize {
        run_it();
        clear_cache();
        get_active_memory()
    }

    // Control: a clean windowed run, twice, to establish both the baseline retention and the
    // run-to-run spread of the phases we will compare.
    let warm_a = run(Some(memory_at(window)));
    let retained_success = retained_after(|| {
        let _ = run(Some(memory_at(window)));
    });
    let warm_b = run(Some(memory_at(window)));
    let cond_spread = (warm_a.conditioning_bytes as f64 - warm_b.conditioning_bytes as f64).abs()
        / warm_b.conditioning_bytes as f64;
    println!(
        "  clean control: retained {:.1} MiB after drop+clear; denoise {:.3}/{:.3} decode \
         {:.3}/{:.3} conditioning {:.3}/{:.3} GiB (conditioning spread {:.1}% — why it is not \
         asserted on)",
        retained_success as f64 / (1024.0 * 1024.0),
        gib(warm_a.denoise_bytes),
        gib(warm_b.denoise_bytes),
        gib(warm_a.decode_bytes),
        gib(warm_b.decode_bytes),
        gib(warm_a.conditioning_bytes),
        gib(warm_b.conditioning_bytes),
        cond_spread * 100.0
    );

    // Retention after a clean run is the floor everything else is compared to; allow a small absolute
    // slack rather than a ratio, because the floor itself may legitimately be ~0.
    let slack = (retained_success as f64 * 0.05).max(64.0 * 1024.0 * 1024.0) as usize;

    // ── Cancel mid-denoise: the flag trips at the first progress step, so the abort lands inside a
    // window rather than between steps.
    let retained_cancel = retained_after(|| {
        let generator = mlx_gen_z_image::load(&spec()).expect("load");
        let cancel = CancelFlag::default();
        let req = GenerationRequest {
            cancel: cancel.clone(),
            ..request(Some(memory_at(window)))
        };
        let mut on_progress = |p: Progress| {
            if matches!(p, Progress::Step { .. }) {
                cancel.cancel();
            }
        };
        // Typed, not merely errored: a cancelled job must not be reported as a failed one.
        match generator.generate(&req, &mut on_progress) {
            Err(e) if format!("{e:?}").contains("Canceled") => {}
            other => panic!(
                "a cancel during the windowed denoise must surface as a typed cancellation, got \
                 {:?}",
                other.map(|_| "images")
            ),
        }
        drop(generator);
    });
    println!(
        "  after cancel:  retained {:.1} MiB",
        retained_cancel as f64 / (1024.0 * 1024.0)
    );
    assert!(
        retained_cancel <= retained_success + slack,
        "a canceled windowed run retained {} bytes vs {} after a clean run",
        retained_cancel,
        retained_success
    );

    // ── Injected error at each physical phase boundary, through the shared calibration hook.
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let retained_fault = retained_after(|| {
            let generator = mlx_gen_z_image::load(&spec()).expect("load");
            let req = request(Some(GenerationMemory {
                calibration_error_phase: Some(phase),
                ..memory_at(window)
            }));
            assert!(
                generator.generate(&req, &mut |_| {}).is_err(),
                "the injected fault at {phase:?} must fail the run"
            );
            drop(generator);
        });
        println!(
            "  after {phase:?} fault: retained {:.1} MiB",
            retained_fault as f64 / (1024.0 * 1024.0)
        );
        assert!(
            retained_fault <= retained_success + slack,
            "a fault at {phase:?} retained {retained_fault} bytes vs {retained_success} after a \
             clean run"
        );

        // Cross-check on the two phases that are stable run-to-run: a warm follow-up must not have
        // become materially more expensive.
        let after = run(Some(memory_at(window)));
        for (label, got, control) in [
            ("denoise", after.denoise_bytes, warm_b.denoise_bytes),
            ("decode", after.decode_bytes, warm_b.decode_bytes),
        ] {
            let drift = got as f64 / control as f64 - 1.0;
            assert!(
                drift < 0.05,
                "after a {phase:?} fault the warm {label} peak drifted {:+.1}%",
                drift * 100.0
            );
        }
    }
}

/// **AC: composition with `Residency::run_staged` pinned by a test.**
///
/// Both are active together: rung 1 sheds the encoder before the DiT phase and the DiT before decode;
/// rung 4 bounds residency *within* the DiT phase. The observable consequence is that the staged
/// maximum stops being the denoise phase.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn rung_four_composes_inside_staged_residency_and_moves_the_binding_phase() {
    println!(
        "\n== rung 4 x rung 1 — tier {} — {}",
        tier_label(),
        host_banner()
    );
    let staged_only = run(Some(memory_at(None)));
    let staged_windowed = run(Some(memory_at(Some(1))));

    for (label, r) in [
        ("rungs 1-3", &staged_only),
        ("+ rung 4 (w=1)", &staged_windowed),
    ] {
        println!(
            "  {label:<16} conditioning {:.3}  denoise {:.3}  decode {:.3}  request {:.3} GiB",
            gib(r.conditioning_bytes),
            gib(r.denoise_bytes),
            gib(r.decode_bytes),
            gib(r.request_bytes())
        );
    }
    assert!(
        staged_windowed.denoise_bytes < staged_only.denoise_bytes,
        "the window must bound the staged denoise phase"
    );
    // Rung 1 is still doing its job: the decode phase excludes the DiT either way, so it is unchanged.
    let decode_drift = staged_windowed.decode_bytes as f64 / staged_only.decode_bytes as f64 - 1.0;
    assert!(
        decode_drift.abs() < 0.05,
        "the decode phase should be unaffected by the trunk window ({:+.1}%)",
        decode_drift * 100.0
    );

    // A rung-4 selection on a Resident load is refused rather than silently degraded.
    let generator = mlx_gen_z_image::load(&spec_with(OffloadPolicy::Resident)).expect("load");
    let err = generator
        .generate(&request(Some(memory_at(Some(1)))), &mut |_| {})
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Sequential"), "{err}");
    println!("  Resident load correctly refuses rung 4: {err}");
}

/// **AC: whether q8 fits an 8 GB budget is answered with measurement, not inference.**
///
/// Runs the whole ladder at this tier and prints the verdict against the 8 GiB budget the generic
/// gate applies its 2 GiB reserve to. Deliberately prints rather than asserts a fit: the answer is
/// evidence for SC-15611 / SC-15614, and a hard assertion here would encode a policy this story does
/// not own.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn the_eight_gb_verdict_with_the_full_ladder() {
    const BUDGET_GIB: f64 = 8.0;
    const RESERVE_GIB: f64 = 2.0;
    println!(
        "\n== 8 GB verdict, full ladder — tier {} — {}",
        tier_label(),
        host_banner()
    );
    println!(
        "  {:>14}  {:>12}  {:>12}  {:>12}  {:>12}",
        "arm", "cond GiB", "denoise", "decode", "request"
    );

    let mut best = f64::MAX;
    let mut first = f64::MAX;
    for (label, memory) in [
        ("rungs 1-3", memory_at(None)),
        ("+ rung 4 w=4", memory_at(Some(4))),
        ("+ rung 4 w=1", memory_at(Some(1))),
    ] {
        let r = run(Some(memory));
        println!(
            "  {label:>14}  {:>12.3}  {:>12.3}  {:>12.3}  {:>12.3}",
            gib(r.conditioning_bytes),
            gib(r.denoise_bytes),
            gib(r.decode_bytes),
            gib(r.request_bytes())
        );
        best = best.min(gib(r.request_bytes()));
        if first == f64::MAX {
            first = gib(r.request_bytes());
        }
    }
    // The verdict itself is deliberately NOT asserted — whether a peak "fits" is SC-15611's admission
    // policy, not this provider's to encode. What IS asserted is the provider-owned fact the verdict
    // rests on: adding rung 4 must strictly improve the request peak over the rungs-1-3 ladder. That
    // is a real regression gate; a printed table alone would not notice the rung going inert.
    assert!(
        best < first,
        "the full ladder's best request peak ({best:.3} GiB) must beat the rungs-1-3 ladder \
         ({first:.3} GiB) — rung 4 has stopped contributing"
    );
    let usable = BUDGET_GIB - RESERVE_GIB;
    println!(
        "\n  best request peak {best:.3} GiB vs {usable:.1} GiB usable ({BUDGET_GIB:.0} GiB budget \
         − {RESERVE_GIB:.0} GiB reserve): {}",
        if best <= usable { "FITS" } else { "DOES NOT FIT" }
    );
    println!(
        "  Disclosure: measured on this machine, not a physical 8 GB Mac. MLX's counter is ACTIVE \
         bytes; on a real 8 GB Mac the pool is unified and shared with macOS/WindowServer/SceneWorks, \
         so what decides the render is the reserve policy and admission arithmetic (SC-15611 / \
         SC-15614)."
    );
}

/// **SC-15510 AC: production tile/chunk/block-window candidate ranges are exposed to the harness** —
/// swept against the exact decode, with the rejected edges re-checked so the exclusion cannot rot.
///
/// ## The reference is the UNTILED decode, not the default tile
///
/// This test first compared every candidate against the 512 px default and failed on 768 with a
/// 46/255 delta — which proves nothing about 768, because both arms were tiled and neither is ground
/// truth. Tiling this GroupNorm VAE perturbs per-tile norm statistics, so *any* two tile edges differ;
/// the question a candidate has to answer is how far it is from the **untiled** decode, which is the
/// exact output. So the reference here is a `Resident`, untiled render at the same seed.
///
/// ## Both halves matter
///
/// The published set must be seam-free, and the rejected set must still be rejected. Asserting only
/// the first would let the ladder silently re-widen; asserting only the second would let it silently
/// narrow to a single point, which is the state SC-15508 refuses to call Verified. See
/// [`DECODE_TILE_EDGES`] for the measured table this bound comes from.
#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn every_declared_decode_candidate_runs_and_preserves_the_image() {
    use mlx_gen_z_image::memory_strategy::{
        DECODE_OVERLAP, DECODE_TILE_EDGE, DECODE_TILE_EDGES, DECODE_TILE_EDGES_REJECTED,
    };

    /// The seam-free bound, in max RGB8 channel distance from the exact decode. Set from the measured
    /// gap between the two groups — the admitted set tops out at 48/255 and the rejected set starts at
    /// 64/255 — so it sits in a 33% void rather than being tuned against one candidate.
    const SEAM_FREE_MAX_DELTA: u8 = 56;

    println!(
        "\n== rung-2 decode ladder — tier {} — {}",
        tier_label(),
        host_banner()
    );

    // Ground truth: the exact, untiled decode. `decode_tiling` returns `None` for a `Resident` load at
    // or below 2048 px with no bounded-decode selection, so this is the single-pass `Vae::decode`.
    let exact = run_with(&spec_with(OffloadPolicy::Resident), None);
    println!(
        "  untiled reference: decode {:.3} GiB, request {:.3} GiB",
        gib(exact.decode_bytes),
        gib(exact.request_bytes())
    );
    println!(
        "  {:>10}  {:>9}  {:>12}  {:>12}  {:>8}  {:>10}",
        "tile edge", "published", "decode GiB", "request GiB", "max Δ", "mean Δ"
    );

    let mut published: Vec<(u32, usize, u8)> = Vec::new();
    let mut rejected: Vec<(u32, u8)> = Vec::new();
    for (&edge, is_published) in DECODE_TILE_EDGES
        .iter()
        .map(|e| (e, true))
        .chain(DECODE_TILE_EDGES_REJECTED.iter().map(|e| (e, false)))
    {
        let r = run(Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(edge),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        }));
        let (max, mean) = image_delta(&r.image, &exact.image);
        println!(
            "  {edge:>10}  {:>9}  {:>12.3}  {:>12.3}  {max:>8}  {mean:>10.4}",
            if is_published { "yes" } else { "rejected" },
            gib(r.decode_bytes),
            gib(r.request_bytes())
        );
        if is_published {
            published.push((edge, r.decode_bytes, max));
        } else {
            rejected.push((edge, max));
        }
    }

    // Every published candidate is seam-free against the exact decode...
    for (edge, _, max) in &published {
        assert!(
            *max <= SEAM_FREE_MAX_DELTA,
            "published tile edge {edge} is {max}/255 from the exact decode, beyond the \
             {SEAM_FREE_MAX_DELTA}/255 seam-free bound — it does not belong in the ladder"
        );
    }
    // ...and every rejected one is still genuinely worse, so the exclusion is a measured fact rather
    // than a stale comment. If this half fails, the ladder should WIDEN, not the bound.
    for (edge, max) in &rejected {
        assert!(
            *max > SEAM_FREE_MAX_DELTA,
            "tile edge {edge} now measures {max}/255 — it is seam-free and should be promoted into \
             DECODE_TILE_EDGES rather than left rejected"
        );
    }
    assert!(
        published
            .iter()
            .any(|(edge, _, _)| *edge == DECODE_TILE_EDGE),
        "the default must be one of the published candidates"
    );
    assert!(
        published.len() > 1,
        "a single-point domain cannot be swept (SC-15508)"
    );

    // Across the published set the decode peak is monotone in the tile edge — that is the whole reason
    // a ladder is a ladder, and it is what lets a selector trade peak for forward count. (It does NOT
    // hold below 512: 448 measures 4.645 GiB against 512's 4.363. Another reason those are out.)
    for pair in published.windows(2) {
        let ((wide, wide_bytes, _), (narrow, narrow_bytes, _)) = (pair[0], pair[1]);
        assert!(
            narrow_bytes <= wide_bytes,
            "tile {narrow} ({:.3} GiB) should not exceed tile {wide} ({:.3} GiB)",
            gib(narrow_bytes),
            gib(wide_bytes)
        );
    }
    // And the point of the rung: even the largest published tile bounds the untiled decode.
    let (_, largest, _) = published[0];
    assert!(
        largest < exact.decode_bytes,
        "the largest published tile ({:.3} GiB) must still bound the untiled decode ({:.3} GiB)",
        gib(largest),
        gib(exact.decode_bytes)
    );
}
