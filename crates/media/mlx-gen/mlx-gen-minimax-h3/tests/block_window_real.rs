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

/// One TE arm end to end: the resident baseline, then every window size in [`WINDOWS`], comparing
/// **output** as well as peak. Returns `(tier_label, resident_peak, cells)`.
///
/// A deliberately short prompt: the quantity under test is *weight residency*, and a long prompt
/// would add activation the window does not bound and dilute the very ratio being measured. The
/// context output is compared across arms at the same prompt, so correctness is not weakened by it.
///
/// The encoder is loaded **inside** each measurement window, unlike the DiT arm below, because for
/// this arm the load *is* the residency: a resident `MiniMaxH3TextEncoder` maps its 50 tapped layers
/// and holds them, and measuring only the forward would put both arms at the same near-zero mark
/// (MLX mmaps lazily — a bare load leaves peak at ~33 KB, `memory_strategy`'s own note).
///
/// # Why this is a function and not one test (sc-17153)
///
/// Until sc-17153 the TE arm ran against **only** the dense upstream `text_encoder/` reachable from
/// `MINIMAX_H3_SNAPSHOT`. A shipped `q4` render does not use that encoder — it uses the packed tier
/// `MINIMAX_H3_TE` names ([`crate::convert::quantize_minimax_h3_text_encoder`]'s artifact, the one
/// `tests/streamed_generate_real.rs` and `tests/te_tier_real_weights.rs` already take) — so the
/// encoder a real windowed request actually conditions through had its windowed output compared
/// against nothing at all. The two arms below are the same measurement over the two encoders.
fn measure_te_arm(te_dir: &std::path::Path, arm: &str) -> (String, u64, Vec<(usize, u64, f64)>) {
    assert!(
        te_dir.join("config.json").is_file(),
        "{arm}: {} has no config.json",
        te_dir.display()
    );
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();

    // 24 presentation tokens. Real ids are not needed for a residency measurement and the tokenizer
    // is a separate component; ids are clamped well inside the vocabulary so the embedding gather is
    // in bounds.
    let ids = Array::from_slice(&(1i32..=24).collect::<Vec<_>>(), &[1, 24]);
    let mask = Array::ones::<i32>(&[1, 24]).unwrap();

    let ((resident_ctx, tier, resident_wall), resident_peak) = peak_of(|| {
        let started = Instant::now();
        let w = Weights::from_dir(te_dir).expect("te weights");
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg).expect("resident te");
        assert_eq!(te.resident_layers(), te.num_loaded_layers());
        assert!(!te.is_deferred());
        // The tier comes from the loaded projections rather than from the directory name, which a
        // caller chooses freely — an unlabelled tier makes every measured cell unattributable.
        let tier = match te.packed_bits().expect("uniform packed width") {
            Some(bits) => format!("q{bits}"),
            None => "bf16".to_owned(),
        };
        let ctx = te.forward(&ids, &mask).expect("resident forward");
        mlx_rs::transforms::eval([&ctx]).unwrap();
        (ctx, tier, started.elapsed().as_secs_f64())
    });
    eprintln!(
        "\n=== TE arm ({arm}) | {tier} | tapped {} layers | {} tokens ===\n  resident            \
         peak {:8.4} GB  {:7.1} s",
        cfg.select_hidden,
        ids.shape()[1],
        resident_peak as f64 / 1e9,
        resident_wall
    );

    let mut cells = Vec::new();
    for window in WINDOWS {
        let ((ctx, wall), peak) = peak_of(|| {
            let started = Instant::now();
            let mut te = MiniMaxH3TextEncoder::from_dir_deferred(te_dir, LM_PREFIX, &cfg)
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
        // **AC6, and the assertion that catches a silently-empty conditioning stage.** The windowed
        // walk runs the identical layer forwards in the identical order, so this is bit identity
        // rather than a tolerance — anything but `0.0` is a defect, not a numeric difference.
        //
        // It is stated in terms of `rel` rather than a norm, a cosine or a checksum deliberately:
        // this model's defects have slipped past cosine 0.73–0.98 seven times. `relative_max_abs`
        // additionally refuses a degenerate *reference*, so a run where the resident arm is the
        // broken one cannot read as agreement either.
        //
        // A windowed forward returning an all-zero context was observed twice on this hardware
        // (sc-17153, 2 of 25 windowed runs) — same shapes, clean run, no Metal diagnostic. Nothing
        // but a value comparison sees it: the peak counter reads *lower* when the work does not
        // happen, so every ratio in this file is happy. See `memory_strategy`'s rung-4 note.
        assert_eq!(
            rel, 0.0,
            "{arm}/window {window} changed the conditioning context by {rel:e} — the windowed walk \
             runs the same arithmetic and differs only in when a layer's weights exist. A context \
             that differs at all (and `0.0` against a max|resident| of the right order is the only \
             passing answer) means the streamed conditioning stage is not the resident one"
        );
        cells.push((window, peak, wall));
    }

    // The rung must actually bound. A windowed arm that matched the resident one would be inert
    // against the thing it exists to bound, and the contract must not declare it.
    let (_, floor_peak, _) = cells[0];
    assert!(
        (floor_peak as f64) < resident_peak as f64 * 0.5,
        "{arm}: window 1 peaked at {floor_peak} B against a resident {resident_peak} B — the TE arm \
         is not bounding the encoder, so `TransformerComponent::TextEncoder` must not be declared"
    );

    // **Window 1 is the measured FLOOR — that, not inertness, is why the published domain is the
    // singleton `[1]` (sc-17153).**
    //
    // The retired claim here was that the parameter is *inert* above 1, bounded by a `< 2.0` spread
    // across `[1 … 50]`. That was measured on the dense encoder alone, where the spread is small
    // (1.41x on the re-measurement) because a single dense Qwen3 layer's f32 upcast dominates the
    // peak at every window size. It does not hold for the encoder a shipped `q4` render actually
    // uses: the packed tier spreads **2.80x** (0.7958 GB at window 1 against 2.2255 GB at window
    // 50), and the campaign that found it independently measured 3.12x. The window parameter is a
    // real lever on the packed encoder, and a `< 2.0` bound asserted over both arms is simply false.
    //
    // What IS true on both, with a wide margin, is that **window 1 is the minimum**: every larger
    // window holds more, monotonically on the packed tier and by 39 % on the dense one. That is the
    // property the singleton domain rests on — `[1]` is the cheapest point, not an arbitrary one —
    // and it is what this now asserts. The spread stays *reported*, because it is this arm's
    // characterization output; it is simply no longer claimed to be ~1.
    let lo = cells.iter().map(|(_, p, _)| *p).min().unwrap();
    let hi = cells.iter().map(|(_, p, _)| *p).max().unwrap();
    assert_eq!(
        floor_peak, lo,
        "{arm}: window 1 peaked at {floor_peak} B but some larger window reached only {lo} B — \
         `TRANSFORMER_WINDOW_SIZE = 1` is published because 1 is the residency floor, so a larger \
         window that costs less means the singleton domain is advertising the wrong point"
    );
    eprintln!(
        "  window 1 is the FLOOR: {:.2}x spread across {WINDOWS:?} ({lo} B to {hi} B), against \
         {:.2}x below resident — larger windows hold more, they are not inert",
        (hi as f64) / (lo as f64),
        (resident_peak as f64) / (hi as f64)
    );
    (tier, resident_peak, cells)
}

/// **The dense TE arm (AC3)** — the upstream 62 GB `text_encoder/` under `MINIMAX_H3_SNAPSHOT`.
#[test]
#[ignore = "sc-18662: needs MINIMAX_H3_SNAPSHOT + Metal; the resident arm holds the 62 GB tapped encoder"]
fn the_text_encoder_arm_is_measured_per_window() {
    let root = common::snapshot();
    let te_dir = root.join("text_encoder");
    let (tier, resident_peak, cells) = measure_te_arm(&te_dir, "dense");
    assert_eq!(
        tier, "bf16",
        "MINIMAX_H3_SNAPSHOT's text_encoder/ must be the dense upstream component; a packed tier \
         there would send packed peaks to RUNG4_TE_*_PEAK_BYTES, which were measured on the dense \
         encoder"
    );
    let (_, floor_peak, _) = cells[0];

    // **Declared versus measured.** `memory_strategy`'s constants are what the contract's rung-4
    // declaration and its module docs are written from, so a drift between them and this harness
    // would leave the provider publishing a saving it no longer reproduces.
    //
    // **sc-17153 (activity 20066, rider 1): each constant is bounded on its own**, replacing the
    // `|measured/declared − 1| < 0.25` band on their *quotient*. That band was asymmetric (it
    // admitted a declared value up to ~1.333x measured) and — the reason it had to go — a quotient
    // cancels *pair-proportional* drift exactly, so the two constants could both drift by the same
    // factor and nothing would red. The retired comment claimed the request-peak bound in
    // `tests/streamed_generate_real.rs` covered that class; it does not, it reads neither of these
    // constants, so the class was covered by nothing. See `common::assert_declared_peak_constant`.
    //
    // The ratio is still *reported*, because the window characterization is what this arm exists to
    // publish — it is simply no longer the thing asserted.
    let declared_resident = mlx_gen_minimax_h3::memory_strategy::RUNG4_TE_RESIDENT_PEAK_BYTES;
    let declared_windowed = mlx_gen_minimax_h3::memory_strategy::RUNG4_TE_WINDOWED_PEAK_BYTES;
    let declared = declared_resident as f64 / declared_windowed as f64;
    let measured = (resident_peak as f64) / (floor_peak as f64);
    eprintln!("  TE arm declared {declared:.2}x, measured {measured:.2}x");
    common::assert_declared_peak_constant(
        "RUNG4_TE_RESIDENT_PEAK_BYTES",
        declared_resident,
        resident_peak,
    );
    common::assert_declared_peak_constant(
        "RUNG4_TE_WINDOWED_PEAK_BYTES",
        declared_windowed,
        floor_peak,
    );
    common::write_rung4_evidence("text_encoder", &tier, &cells, resident_peak);
}

/// **The packed TE arm (sc-17153)** — the tier `MINIMAX_H3_TE` names, which is the encoder a
/// shipped `q4` / `q8` render actually conditions through.
///
/// This arm exists because the dense one above cannot stand in for it. `TransformerComponent::Both`
/// windows the encoder on **every** tier, but until sc-17153 the only encoder whose windowed output
/// was ever compared to anything was the 62 GB dense component — which no tiered render loads. The
/// packed path differs from the dense one in the loader that reconstructs each windowed layer
/// (`mlx_gen::quant::lin`'s packed triple against a dense `Dense` matmul) and in the token table's
/// representation, so "the dense arm is bit-identical" was never evidence about it.
///
/// It publishes **no byte constants of its own**. The packed conditioning stage's resident peak is
/// already declared, and independently reproduced, by
/// [`CONDITIONING_STAGE_PEAK_Q4_BYTES`](mlx_gen_minimax_h3::memory_strategy::CONDITIONING_STAGE_PEAK_Q4_BYTES)
/// through `tests/te_tier_real_weights.rs`; what was missing was the windowed-versus-resident
/// **output** comparison and the window characterization, which are what this asserts.
#[test]
#[ignore = "sc-17153: needs MINIMAX_H3_TE (a packed text_encoder/) + Metal"]
fn the_packed_text_encoder_arm_is_measured_per_window() {
    let raw = std::env::var("MINIMAX_H3_TE").unwrap_or_default();
    assert!(
        !raw.trim().is_empty(),
        "MINIMAX_H3_TE must point at a packed MiniMax-H3 `text_encoder/` (the same variable \
         `tests/streamed_generate_real.rs` and `tests/te_tier_real_weights.rs` take). This test is \
         #[ignore]d and asserts rather than skips so that a missing tier cannot be mistaken for a \
         pass — which is exactly how the packed encoder went unmeasured under windowing."
    );
    let te_dir = PathBuf::from(raw.trim());
    let (tier, resident_peak, cells) = measure_te_arm(&te_dir, "packed");
    assert_ne!(
        tier, "bf16",
        "MINIMAX_H3_TE={} loaded DENSE projections — this arm exists to cover the packed tier, and \
         pointing it at the dense component would silently re-run the arm above",
        te_dir.display()
    );
    common::write_rung4_evidence("text_encoder", &tier, &cells, resident_peak);
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
    // sc-17153 (activity 20066, rider 1): bounded per constant, per tier, for the same reason as
    // the TE arm above — a band on the quotient cancels pair-proportional drift of a
    // `RUNG4_DIT_MEASURED_PEAKS` row exactly, and nothing else reads these two numbers. The ratio
    // remains reported as this arm's characterization output.
    common::assert_declared_peak_constant(
        &format!("RUNG4_DIT_MEASURED_PEAKS[{tier}].resident"),
        declared_resident,
        resident_peak,
    );
    common::assert_declared_peak_constant(
        &format!("RUNG4_DIT_MEASURED_PEAKS[{tier}].windowed"),
        declared_windowed,
        floor_peak,
    );
    common::write_rung4_evidence("dit", &tier, &cells, resident_peak);
}
