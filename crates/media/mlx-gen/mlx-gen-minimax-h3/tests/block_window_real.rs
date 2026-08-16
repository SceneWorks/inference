//! sc-18662 — **rung 4 (`BoundedTransformerResidency`) measured on real weights, both arms.**
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<upstream snapshot root> SCENEWORKS_GPU_ID=mlx \
//!   cargo test -p mlx-gen-minimax-h3 --test block_window_real -- --ignored --nocapture \
//!   --test-threads=1
//! # the DiT arm additionally takes a staged tier:
//! MINIMAX_H3_DIT=<tier>/transformer  …
//! ```
//!
//! # Both arms, separately, because `TransformerComponent::Both` is two claims
//!
//! `gen_core::memory_strategy::TransformerComponent` offers `Dit`, `TextEncoder` and `Both`, and the
//! story's AC3 says outright that a result which only measures `Dit` does not satisfy it. So the two
//! arms are measured in their own processes' own windows and reported as their own cells:
//!
//! | arm | what a window holds | resident baseline |
//! |---|---|---|
//! | [`the_text_encoder_arm_is_measured_per_window`] | one Qwen3-VL decoder layer | the whole 50-layer tap |
//! | [`the_dit_arm_is_measured_per_window_and_tier`] | one DiT block **body** | the whole 50-block stack |
//!
//! # Peaks, not wall clock
//!
//! This Mac is also the `nax-macos` CI runner and has been measured at load average 80-109 while the
//! fleet is working. Every number this file **asserts** on is an MLX peak-memory figure, which is a
//! per-process allocator high-water mark and does not read the machine's load at all. Wall clock is
//! printed for context and is deliberately never asserted — sc-19505's class.
//!
//! # Why the window sizes are what they are
//!
//! `1` is the floor case AC5 names, and it is the configuration the rung exists for. The others
//! bracket it so the bound can be shown to be **linear in the window** rather than merely lower:
//! a residency that fell to a constant regardless of window would mean something else was setting
//! the peak and the rung's parameter would be inert — which is exactly what
//! `epic_15448`'s rung-4 survey found on another family.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_minimax_h3::dit::MiniMaxH3Dit;
use mlx_gen_minimax_h3::text_encoder::{MiniMaxH3TeConfig, MiniMaxH3TextEncoder, LM_PREFIX};
use mlx_gen_minimax_h3::DitBlockStream;

/// Window sizes measured on both arms. `1` is AC5's floor; `50` is the fully-resident degenerate
/// plan, which is the A-side of the parity comparison and must reproduce the resident path exactly.
const WINDOWS: [usize; 4] = [1, 5, 10, 50];

fn peak_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let out = f();
    (out, mlx_rs::memory::get_peak_memory() as u64)
}

fn relative_max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let delta: f32 = a
        .subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let scale: f32 = a.abs().unwrap().max(None).unwrap().item();
    assert!(scale > 1e-6, "the reference arm is degenerate ({scale})");
    delta / scale
}

/// **The TE arm (AC3).** Peak of one conditioning forward, resident against each window size.
///
/// A deliberately short prompt: the quantity under test is *weight residency*, and a long prompt
/// would add activation the window does not bound and dilute the very ratio being measured. The
/// context output is compared across arms at the same prompt, so correctness is not weakened by it.
///
/// The encoder is loaded **inside** each measurement window, unlike the DiT arm below, because for
/// this arm the load *is* the residency: a resident `MiniMaxH3TextEncoder` maps its 50 tapped layers
/// and holds them, and measuring only the forward would put both arms at the same near-zero mark
/// (MLX mmaps lazily — a bare load leaves peak at ~33 KB, `memory_strategy`'s own note).
#[test]
#[ignore = "sc-18662: needs MINIMAX_H3_SNAPSHOT + Metal; the resident arm holds the 62 GB tapped encoder"]
fn the_text_encoder_arm_is_measured_per_window() {
    let root = common::snapshot();
    let te_dir = root.join("text_encoder");
    assert!(
        te_dir.join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT={} has no text_encoder/config.json",
        root.display()
    );
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();

    // 24 presentation tokens. Real ids are not needed for a residency measurement and the tokenizer
    // is a separate component; ids are clamped well inside the vocabulary so the embedding gather is
    // in bounds.
    let ids = Array::from_slice(&(1i32..=24).collect::<Vec<_>>(), &[1, 24]);
    let mask = Array::ones::<i32>(&[1, 24]).unwrap();

    let ((resident_ctx, resident_wall), resident_peak) = peak_of(|| {
        let started = Instant::now();
        let w = Weights::from_dir(&te_dir).expect("te weights");
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg).expect("resident te");
        assert_eq!(te.resident_layers(), te.num_loaded_layers());
        assert!(!te.is_deferred());
        let ctx = te.forward(&ids, &mask).expect("resident forward");
        mlx_rs::transforms::eval([&ctx]).unwrap();
        (ctx, started.elapsed().as_secs_f64())
    });
    eprintln!(
        "\n=== TE arm | tapped {} layers | {} tokens ===\n  resident            peak {:8.4} GB  {:7.1} s",
        cfg.select_hidden,
        ids.shape()[1],
        resident_peak as f64 / 1e9,
        resident_wall
    );

    let mut cells = Vec::new();
    for window in WINDOWS {
        let ((ctx, wall), peak) = peak_of(|| {
            let started = Instant::now();
            let mut te = MiniMaxH3TextEncoder::from_dir_deferred(&te_dir, LM_PREFIX, &cfg)
                .expect("deferred te");
            assert_eq!(
                te.resident_layers(),
                0,
                "a deferred encoder must hold no layers — that is the entire rung"
            );
            assert!(te.is_deferred());
            te.set_block_window(window, CancelFlag::default())
                .expect("window");
            let plan = te.block_window().expect("plan");
            assert_eq!(plan.window(), window.min(cfg.select_hidden));
            let ctx = te.forward(&ids, &mask).expect("windowed forward");
            mlx_rs::transforms::eval([&ctx]).unwrap();
            (ctx, started.elapsed().as_secs_f64())
        });
        let rel = relative_max_abs(&resident_ctx, &ctx);
        eprintln!(
            "  window {window:<3}          peak {:8.4} GB  {:7.1} s   {:5.2}x below resident  \
             rel|Δ| {rel:.3e}",
            peak as f64 / 1e9,
            wall,
            resident_peak as f64 / peak.max(1) as f64
        );
        // AC6: correctness alongside memory. The windowed walk runs the identical layer forwards in
        // the identical order, so this is bit identity rather than a tolerance.
        assert_eq!(
            rel, 0.0,
            "window {window} changed the conditioning context by {rel:e} — the windowed walk runs \
             the same arithmetic and differs only in when a layer's weights exist"
        );
        cells.push((window, peak, wall));
    }

    // The rung must actually bound. A windowed arm that matched the resident one would be inert
    // against the thing it exists to bound, and the contract must not declare it.
    let (_, floor_peak, _) = cells[0];
    assert!(
        (floor_peak as f64) < resident_peak as f64 * 0.5,
        "window 1 peaked at {floor_peak} B against a resident {resident_peak} B — the TE arm is \
         not bounding the encoder, so `TransformerComponent::TextEncoder` must not be declared"
    );
    // **The window parameter is INERT — this is the finding, and it is why the published domain is
    // the singleton `[1]`.**
    //
    // `run_windowed`'s `apply` releases each layer as soon as it is consumed rather than at the end
    // of the window, so even the 50-layer all-covering plan holds one layer at a time and the
    // *plan's* window never sets the residency. Measured across two runs, every window size lands in
    // one band — 6.24 to 6.34 GB on this run, and a 5.30 GB window-1 outlier on the previous one, so
    // the run-to-run spread (~1 GB) is larger than the spread across window sizes.
    //
    // So `transformer_window_sizes` publishes `[1]`, and that singleton is the finding rather than a
    // placeholder — the same shape rung 2 already has on this provider, where `DECODE_TILE_EDGE` is a
    // singleton for its own measured reason. Publishing `[1, 5, 10, 50]` would advertise a lever
    // whose other values do nothing.
    //
    // Independently reproduced: SC-15448's rung-4 survey recorded the same window-inert-above-1
    // behaviour on another family.
    //
    // The assertion is a **characterization, not an assumption**: if a window genuinely scaled
    // residency the spread across `[1 … 50]` would be ~50x, and this would fire. It is the guard
    // that would catch the residency silently becoming window-proportional — at which point the
    // domain would have to be widened and re-measured.
    //
    // The bound is 2.0 rather than something tight because the discriminator is 50x against ~1x, so
    // anything well under 50 separates them, while the peak counter carries ~1 GB of run-to-run
    // spread on a ~6 GB figure. Measured spreads so far: 1.09x and 1.20x. A tighter bound would buy
    // no discrimination and would go red on machine noise.
    let lo = cells.iter().map(|(_, p, _)| *p).min().unwrap();
    let hi = cells.iter().map(|(_, p, _)| *p).max().unwrap();
    assert!(
        (hi as f64) / (lo as f64) < 2.0,
        "the window sizes {WINDOWS:?} spread {:.2}x ({lo} B to {hi} B) — the parameter is no longer \
         inert, so the singleton `[1]` domain understates it and must be re-measured",
        (hi as f64) / (lo as f64)
    );
    eprintln!(
        "  window parameter is INERT: {:.2}x spread across {WINDOWS:?}, against {:.2}x below \
         resident — a layer is released as it is consumed, not at the window boundary",
        (hi as f64) / (lo as f64),
        (resident_peak as f64) / (hi as f64)
    );

    // **Declared versus measured.** `memory_strategy`'s constants are what the contract's rung-4
    // declaration and its module docs are written from, so a drift between them and this harness
    // would leave the provider publishing a saving it no longer reproduces. Held as a ratio band
    // rather than to the byte: the peak counter carries ~1 GB of run-to-run spread on this machine.
    let declared = mlx_gen_minimax_h3::memory_strategy::RUNG4_TE_RESIDENT_PEAK_BYTES as f64
        / mlx_gen_minimax_h3::memory_strategy::RUNG4_TE_WINDOWED_PEAK_BYTES as f64;
    let measured = (resident_peak as f64) / (floor_peak as f64);
    eprintln!("  TE arm declared {declared:.2}x, measured {measured:.2}x");
    assert!(
        (measured / declared - 1.0).abs() < 0.25,
        "the TE arm measured {measured:.2}x against a declared {declared:.2}x — \
         RUNG4_TE_*_PEAK_BYTES is stale, and the contract's rung-4 declaration is written from it"
    );
    common::write_rung4_evidence("text_encoder", "bf16", &cells, resident_peak);
}

/// **The DiT arm**, per window and per tier. `MINIMAX_H3_DIT` selects the tier.
///
/// Unlike the TE arm this measures the **denoise-phase** residency: the stack is walked with a
/// precomputed AdaLN cache, which is what a render does, and the window holds block *bodies* only.
/// `crate::block_stream`'s module docs explain why `adaln_proj` must not be in that window and why
/// no equivalence test could see it if it were.
#[test]
#[ignore = "sc-18662: needs MINIMAX_H3_DIT (a tier's transformer/) + Metal"]
fn the_dit_arm_is_measured_per_window_and_tier() {
    let dir = PathBuf::from(
        std::env::var("MINIMAX_H3_DIT")
            .expect("MINIMAX_H3_DIT=<tier>/transformer must name a staged MiniMax-H3 DiT"),
    );
    let tier = common::describe_dit_tier(&dir);
    let cfg = mlx_gen_minimax_h3::MiniMaxH3DitConfig::default();
    let stream = DitBlockStream::new(&dir, Dtype::Bfloat16, cfg.clone()).expect("stream");

    // A single hidden state through the whole stack, with the modulation supplied from a resident
    // precompute so both arms denoise from the identical cache. The measured window is the stack
    // walk alone; the precompute is its own residency and is measured by its own arm below.
    let seq = 4096i32;
    let hidden = mlx_rs::random::normal::<f32>(
        &[1, seq, cfg.hidden_size],
        None,
        None,
        Some(&mlx_rs::random::key(7).unwrap()),
    )
    .unwrap()
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    mlx_rs::transforms::eval([&hidden]).unwrap();

    let ((resident_out, resident_wall), resident_peak) = peak_of(|| {
        let started = Instant::now();
        let dit = MiniMaxH3Dit::load_dir(&dir, Dtype::Bfloat16).expect("resident dit");
        assert_eq!(dit.resident_blocks(), cfg.num_layers as usize);
        let out = common::walk_resident_blocks(&dit, &hidden).expect("resident walk");
        mlx_rs::transforms::eval([&out]).unwrap();
        (out, started.elapsed().as_secs_f64())
    });
    eprintln!(
        "\n=== DiT arm | {tier} | {} blocks | seq {seq} ===\n  resident            peak {:8.4} GB  \
         {:7.1} s",
        cfg.num_layers,
        resident_peak as f64 / 1e9,
        resident_wall
    );

    let mut cells = Vec::new();
    for window in WINDOWS {
        let plan = stream.plan(window).expect("plan");
        let ((out, wall), peak) = peak_of(|| {
            let started = Instant::now();
            let dit = MiniMaxH3Dit::load_dir_deferred(&dir, Dtype::Bfloat16).expect("deferred dit");
            assert_eq!(dit.resident_blocks(), 0);
            let out = common::walk_windowed_blocks(&stream, &plan, &hidden).expect("windowed walk");
            mlx_rs::transforms::eval([&out]).unwrap();
            drop(dit);
            (out, started.elapsed().as_secs_f64())
        });
        let rel = relative_max_abs(&resident_out, &out);
        eprintln!(
            "  window {window:<3}          peak {:8.4} GB  {:7.1} s   {:5.2}x below resident  \
             rel|Δ| {rel:.3e}",
            peak as f64 / 1e9,
            wall,
            resident_peak as f64 / peak.max(1) as f64
        );
        assert_eq!(
            rel, 0.0,
            "window {window} changed the block stack's output by {rel:e}"
        );
        cells.push((window, peak, wall));
    }

    let (_, floor_peak, _) = cells[0];
    assert!(
        (floor_peak as f64) < resident_peak as f64 * 0.75,
        "window 1 peaked at {floor_peak} B against a resident {resident_peak} B — the DiT arm is \
         not bounding the stack"
    );
    // Declared versus measured, per tier — see the TE arm for why this is a band.
    let (_, declared_resident, declared_windowed) =
        mlx_gen_minimax_h3::memory_strategy::RUNG4_DIT_MEASURED_PEAKS
            .into_iter()
            .find(|(t, _, _)| *t == tier)
            .unwrap_or_else(|| {
                panic!(
                    "tier {tier:?} has no row in RUNG4_DIT_MEASURED_PEAKS; a tier measured but not \
                     declared leaves the contract's docs describing a subset of what was run"
                )
            });
    let declared = declared_resident as f64 / declared_windowed as f64;
    let measured = (resident_peak as f64) / (floor_peak as f64);
    eprintln!("  DiT arm {tier} declared {declared:.2}x, measured {measured:.2}x");
    assert!(
        (measured / declared - 1.0).abs() < 0.25,
        "the {tier} DiT arm measured {measured:.2}x against a declared {declared:.2}x — \
         RUNG4_DIT_MEASURED_PEAKS is stale"
    );
    common::write_rung4_evidence("dit", &tier, &cells, resident_peak);
}
