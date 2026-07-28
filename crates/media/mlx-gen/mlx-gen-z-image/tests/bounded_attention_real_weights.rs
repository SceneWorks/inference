//! SC-15615 — the real-weight bounded-attention (ladder rung 3) A/B on the MLX Z-Image lane.
//!
//! `#[ignore]`d: needs a real Z-Image-Turbo snapshot (`ZIMAGE_SNAPSHOT`, a pre-quantized tier dir for
//! q4/q8) and an Apple/Metal GPU. Run:
//!
//! ```text
//! ZIMAGE_SNAPSHOT=~/.cache/huggingface/hub/models--SceneWorks--z-image-turbo-mlx/snapshots/<rev>/q4 \
//!   cargo test -p mlx-gen-z-image --release --test bounded_attention_real_weights -- --ignored --nocapture
//! ```
//!
//! ## What this pins
//!
//! The story asked for the Candle rung-3 saving to be ported to MLX and for the MLX staged denoise
//! peak to be recorded **with and without** the budget "so the saving is evidenced rather than
//! assumed". Measured here on Apple M5 Max, real `z_image_turbo` q4, 1024², 4 steps, staged
//! residency, count 1:
//!
//! | Arm | Staged denoise peak | vs unbounded |
//! |---|---:|---:|
//! | DiT denoise loop, unbounded | 4.7746 GiB | — |
//! | DiT denoise loop, 64 Mi chunk, **lazy** | 4.7708 GiB | −0.08% (noise) |
//! | DiT denoise loop, 64 Mi chunk + per-chunk eval | 4.6944 GiB | **−1.7%** |
//! | never-chunks budget + eval flag (control) | 4.7747 GiB | +0.00% |
//! | Full staged generate, no budget | 4.898 GiB | — |
//! | Full staged generate, 64 Mi budget | 4.653 GiB | **−5.0%** |
//!
//! The saving is real and the output is bit-identical, but it comes from the lazy-graph cut the
//! per-chunk `eval` forces — **not** from bounding a materialized score matrix, which MLX's fused
//! `scaled_dot_product_attention` never builds (see `mlx_gen::attention`). Hence the magnitude is
//! nothing like CUDA's 32%.
//!
//! The tests:
//!
//! 1. [`bounded_attention_reduces_the_staged_denoise_peak`] — the A/B, asserting the measured
//!    direction so a regression that erases the saving fails loudly.
//! 2. [`the_saving_comes_from_the_per_chunk_eval_barrier`] — the attribution, including the
//!    never-chunks control that proves the eval flag alone is inert.
//! 3. [`bounded_attention_preserves_the_image`] — numerical equivalence: the two arms produce the
//!    **same** image at the same seed.
//! 4. [`the_eight_gb_mac_verdict`] — the measured answer to "does `z_image_turbo` q4 fit an 8 GB
//!    Mac", with its host disclosure.
//! 5. [`a_canceled_run_leaves_no_resident_cache`] — a cancel mid-denoise with the budget engaged must
//!    not leave the next request worse off.
//!
//! Every number printed here is what the story's evidence quotes; nothing is asserted as a saving
//! that was not measured.

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

/// The tier under test. The A/B is tier-agnostic (attention scratch does not depend on the weight
/// tier), but q4 is the tier the 8 GB question is about, so it is the default.
fn quant() -> Option<Quant> {
    match std::env::var("ZIMAGE_ATTN_TIER").as_deref() {
        Ok("q8") => Some(Quant::Q8),
        Ok("bf16") => None,
        // A pre-quantized turnkey tier dir loads packed; the requested tier must match what is on
        // disk or the loader's tier guard hard-errors (F-009).
        _ => Some(Quant::Q4),
    }
}

fn spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    // The constrained-Mac path: staged residency (encode → drop encoder → denoise → drop DiT →
    // tiled decode) is exactly the configuration whose denoise peak the story asks about.
    spec.offload_policy = OffloadPolicy::Sequential;
    if let Some(q) = quant() {
        spec = spec.with_quant(q);
    }
    spec
}

fn request(memory: Option<GenerationMemory>) -> GenerationRequest {
    let size = env_u32("ZIMAGE_ATTN_SIZE", 1024);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_ATTN_STEPS", 4)),
        memory,
        ..Default::default()
    }
}

/// Peaks for one generation, in bytes: `(denoise-phase peak, whole-request peak)`.
///
/// MLX's `get_peak_memory` is peak **active** bytes, not cache, so it is the right counter for a
/// residency budget. The denoise-phase peak is isolated by resetting the counter at the first
/// `Progress::Step` and reading it at `Progress::Decoding` — i.e. exactly the staged denoise window
/// the Candle twin reported as 8.394 → 5.709 GB.
struct Peaks {
    denoise_bytes: usize,
    total_bytes: usize,
    image: Image,
}

fn run(memory: Option<GenerationMemory>) -> Peaks {
    let generator = mlx_gen_z_image::load(&spec()).expect("load z_image_turbo");
    let req = request(memory);

    clear_cache();
    reset_peak_memory();

    let mut denoise_bytes = 0usize;
    let mut saw_step = false;
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current: 1, .. } => {
            // Start of the denoise phase: the encoder is already dropped and the DiT is resident.
            reset_peak_memory();
            saw_step = true;
        }
        Progress::Decoding if saw_step && denoise_bytes == 0 => {
            denoise_bytes = get_peak_memory();
        }
        _ => {}
    };

    let out = generator
        .generate(&req, &mut on_progress)
        .expect("generation");
    let total_bytes = get_peak_memory().max(denoise_bytes);
    let image = match out {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    Peaks {
        denoise_bytes,
        total_bytes,
        image,
    }
}

/// `(max|Δ|, mean|Δ|)` over the RGB8 channels — the two arms must agree pixel-for-pixel or within the
/// documented MLX Metal reduced-precision class.
fn image_delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height), "image dims");
    assert_eq!(a.pixels.len(), b.pixels.len(), "pixel count");
    let mut max = 0u8;
    let mut sum = 0u64;
    for (x, y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

fn bounded() -> Option<GenerationMemory> {
    Some(GenerationMemory {
        // Rung 2 is on in both arms (it is what the constrained-Mac path already does), so the ONLY
        // difference between the arms is the attention budget.
        tile_vae_decode: true,
        chunk_attention: true,
        ..Default::default()
    })
}

fn staged_only() -> Option<GenerationMemory> {
    Some(GenerationMemory {
        tile_vae_decode: true,
        ..Default::default()
    })
}

/// The A/B the story asked for: staged denoise peak WITH and WITHOUT the 64 Mi attention budget.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn bounded_attention_reduces_the_staged_denoise_peak() {
    let without = run(staged_only());
    let with = run(bounded());

    let (w_den, b_den) = (without.denoise_bytes as f64, with.denoise_bytes as f64);
    let (w_tot, b_tot) = (without.total_bytes as f64, with.total_bytes as f64);
    println!(
        "SC-15615 MLX Z-Image bounded-attention A/B ({:?} tier)",
        quant()
    );
    println!(
        "  WITHOUT budget: denoise peak {:.3} GiB, request peak {:.3} GiB",
        w_den / GIB,
        w_tot / GIB
    );
    println!(
        "  WITH 64Mi budget: denoise peak {:.3} GiB, request peak {:.3} GiB",
        b_den / GIB,
        b_tot / GIB
    );
    println!(
        "  delta: denoise {:+.1}%, request {:+.1}%",
        100.0 * (b_den - w_den) / w_den,
        100.0 * (b_tot - w_tot) / w_tot
    );

    assert!(
        without.denoise_bytes > 0,
        "denoise-phase peak was not sampled"
    );
    assert!(with.denoise_bytes > 0, "denoise-phase peak was not sampled");

    // Measured -5.0% end to end. Assert only that the rung is a genuine (non-noise) improvement, so
    // this fails if a future change erases the saving that justifies declaring rung 3 Implemented —
    // without over-fitting to this host's exact figure.
    assert!(
        b_den <= w_den * 0.99,
        "bounded attention no longer reduces the MLX staged denoise peak ({:.3} -> {:.3} GiB, \
         {:+.1}%). Rung 3's Implemented declaration in mlx_gen_z_image::image_memory rests on that \
         saving — re-measure SC-15615 before shipping.",
        w_den / GIB,
        b_den / GIB,
        100.0 * (b_den - w_den) / w_den
    );
    assert!(b_tot <= w_tot * 0.99, "request peak did not improve either");
}

/// Attribution: the saving is the per-chunk `eval` barrier cutting the lazy graph, not a bounded
/// score matrix. Drives the denoise loop directly so an arbitrary [`AttentionBudget`] can be applied,
/// and includes the **control** that pins the claim — a budget that never chunks but still sets
/// `eval_per_chunk` must be inert.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_saving_comes_from_the_per_chunk_eval_barrier() {
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::FlowMatchEuler;
    use mlx_rs::Dtype;

    let size = env_u32("ZIMAGE_ATTN_SIZE", 1024);
    let steps = env_u32("ZIMAGE_ATTN_STEPS", 4) as usize;
    let root = snapshot();
    let mut transformer = mlx_gen_z_image::load_transformer(&root).expect("transformer");
    if let Some(q) = quant() {
        transformer.quantize(q.bits()).expect("quantize");
    }
    // Synthetic caption conditioning: the DiT's memory behaviour depends on the token count and
    // dtype, not the encoder's values, so this avoids loading the 2.1 GB text encoder.
    let cap = mlx_rs::random::normal::<f32>(&[32, 2560], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    mlx_rs::transforms::eval([&cap]).unwrap();
    let scheduler = FlowMatchEuler::for_static_shift(steps, 3.0);

    let peak_for = |label: &str, budget: AttentionBudget| -> (f64, f32) {
        let latents = mlx_gen_z_image::create_noise(1234, size, size)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        mlx_rs::transforms::eval([&latents]).unwrap();
        clear_cache();
        reset_peak_memory();
        let out = mlx_gen_z_image::denoise_with_progress(
            &transformer,
            &scheduler,
            None,
            1234,
            latents,
            &cap,
            0,
            budget,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .expect("denoise");
        mlx_rs::transforms::eval([&out]).unwrap();
        let peak = get_peak_memory() as f64;
        let checksum: f32 = out
            .as_dtype(Dtype::Float32)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item();
        println!(
            "  {label:<34} denoise peak {:.4} GiB  sum|out| {checksum:.6}",
            peak / GIB
        );
        drop(out);
        clear_cache();
        (peak, checksum)
    };

    println!(
        "SC-15615 attribution ({:?} tier, {size}², {steps} steps)",
        quant()
    );
    let (unbounded, sum_unbounded) = peak_for("unbounded", AttentionBudget::UNBOUNDED);
    let (lazy, sum_lazy) = peak_for(
        "64Mi chunk, lazy",
        AttentionBudget::from_score_elements(
            u64::from(mlx_gen_z_image::image_memory::ATTENTION_CHUNK_SIZE),
            false,
        ),
    );
    let (eager, sum_eager) = peak_for("64Mi chunk + per-chunk eval", AttentionBudget::CONSTRAINED);
    // Control: `u64::MAX - 1` is not the UNBOUNDED sentinel, so the eval flag is live, but the budget
    // is large enough that `query_block` returns the whole sequence — the single-call fast path.
    let (control, sum_control) = peak_for(
        "never-chunks budget + eval flag",
        AttentionBudget::from_score_elements(u64::MAX - 1, true),
    );

    // Every arm computes the same velocity, exactly.
    for (label, sum) in [
        ("lazy", sum_lazy),
        ("eager", sum_eager),
        ("control", sum_control),
    ] {
        assert_eq!(
            sum, sum_unbounded,
            "{label} arm changed the DiT output (sum|out| {sum} vs {sum_unbounded})"
        );
    }

    // The control is inert: the eval flag alone, without chunking, does nothing.
    assert!(
        (control - unbounded).abs() <= unbounded * 0.005,
        "the eval flag alone moved the peak ({:.4} -> {:.4} GiB); the attribution below is invalid",
        unbounded / GIB,
        control / GIB
    );
    // Lazy chunking is ~noise; the per-chunk eval is where the saving is.
    assert!(
        eager < lazy,
        "the per-chunk eval barrier is supposed to be the lever: lazy {:.4} GiB vs eager {:.4} GiB",
        lazy / GIB,
        eager / GIB
    );
    println!(
        "  => lazy chunking {:+.2}%, chunking+eval {:+.2}% vs unbounded",
        100.0 * (lazy - unbounded) / unbounded,
        100.0 * (eager - unbounded) / unbounded
    );
}

/// The numerical half: the budget must not change the image.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn bounded_attention_preserves_the_image() {
    let without = run(staged_only());
    let with = run(bounded());
    let (max, mean) = image_delta(&without.image, &with.image);
    println!("SC-15615 image equivalence: max|Δ| {max}/255, mean|Δ| {mean:.4}/255");
    // Measured EXACT at the production 64 Mi budget: 0/255 on every channel. The unit-level
    // `chunking_is_exact_at_the_production_shape_and_budget` pins the same thing at the tensor level.
    // (Query-row chunking can change which Metal kernel specialization runs, and MLX's Metal matmul
    // is reduced-precision — the sc-2338 class — so a ±1 LSB result would still be acceptable; this
    // asserts the stronger fact that was actually observed.)
    assert_eq!(
        max, 0,
        "bounded attention changed the image (max channel delta {max}/255)"
    );
    assert_eq!(
        mean, 0.0,
        "bounded attention changed the image (mean delta {mean})"
    );
}

/// The measured answer to GitHub #1932's question, printed with its assumptions stated. This does not
/// assert a fit — it records the number the truthful-rejection path (SC-15611) needs.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_eight_gb_mac_verdict() {
    let staged = run(staged_only());
    let with_rung3 = run(bounded());
    let peak = staged.total_bytes as f64;
    let denoise = staged.denoise_bytes as f64;
    // The generic MLX gate's operational reserve today. On a Mac this is the *shared* pool macOS,
    // WindowServer and SceneWorks itself live in, which SC-15614 is separately re-deriving.
    let reserve_gib = 2.0;
    let budget_gib = 8.0;
    println!(
        "SC-15615 8 GB Mac verdict for z_image_turbo {:?} at {}²:",
        quant(),
        staged.image.width
    );
    println!("  staged denoise peak : {:.3} GiB", denoise / GIB);
    println!("  staged request peak : {:.3} GiB", peak / GIB);
    println!(
        "  + {reserve_gib:.1} GiB reserve = {:.3} GiB against an {budget_gib:.1} GiB budget -> {}",
        peak / GIB + reserve_gib,
        if peak / GIB + reserve_gib <= budget_gib {
            "FITS"
        } else {
            "DOES NOT FIT"
        }
    );
    println!(
        "  with rung 3 (64 Mi bounded attention): {:.3} GiB, + reserve = {:.3} GiB",
        with_rung3.total_bytes as f64 / GIB,
        with_rung3.total_bytes as f64 / GIB + reserve_gib
    );
    println!(
        "  DISCLOSURE: measured on this host, which is NOT a physical 8 GB Mac. MLX's peak counter \
         is ACTIVE bytes (not cache). Staged residency + tiled decode, count 1. On a real 8 GB Mac \
         the pool is UNIFIED and shared with macOS/WindowServer/SceneWorks, so the 2 GiB reserve \
         borrowed from the dedicated-VRAM lane is the open question (SC-15611 / SC-15614), not this \
         rung."
    );
    assert!(peak > 0.0, "no peak was sampled");
}

/// Cancelling mid-denoise with the budget engaged must not leave a partially resident cache that
/// poisons the next request (the story's cancellation criterion, and the MLX analog of the Candle
/// twin's per-tile cancellation work).
///
/// The comparison that matters is **canceled-then-warm vs succeeded-then-warm**, not vs a cold run:
/// a warm second generation on the same generator legitimately peaks a little higher than a cold one
/// (the allocator has grown), and comparing against cold would blame that on the cancel. Both arms
/// here are second-runs on a warm generator; the only difference is whether the first run was
/// canceled.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn a_canceled_run_leaves_no_resident_cache() {
    /// Peak of a second generation on a warm generator whose first generation either completed or
    /// was canceled at the first step boundary.
    fn warm_second_run_peak(cancel_first: bool) -> usize {
        let generator = mlx_gen_z_image::load(&spec()).expect("load z_image_turbo");

        let cancel = CancelFlag::default();
        let mut first = request(bounded());
        first.cancel = cancel.clone();
        let mut trip = |p: Progress| {
            if cancel_first && matches!(p, Progress::Step { .. }) {
                cancel.cancel();
            }
        };
        let outcome = generator.generate(&first, &mut trip);
        if cancel_first {
            let err = outcome.expect_err("a canceled generation must return an error");
            println!("  cancel arm first run: {err}");
        } else {
            outcome.expect("the control arm's first run must succeed");
        }

        clear_cache();
        reset_peak_memory();
        let out = generator
            .generate(&request(bounded()), &mut |_| {})
            .expect("the warm follow-up must succeed");
        let peak = get_peak_memory();
        assert!(matches!(out, GenerationOutput::Images(ref v) if v.len() == 1));
        drop(generator);
        clear_cache();
        peak
    }

    let after_success = warm_second_run_peak(false);
    let after_cancel = warm_second_run_peak(true);
    println!(
        "SC-15615 warm second-run peak: after success {:.3} GiB, after cancel {:.3} GiB ({:+.1}%)",
        after_success as f64 / GIB,
        after_cancel as f64 / GIB,
        100.0 * (after_cancel as f64 - after_success as f64) / after_success as f64
    );

    // A cancel must cost nothing measurable on the NEXT request. 2% is measurement noise; anything
    // above it means the canceled run left resident state behind.
    assert!(
        after_cancel as f64 <= after_success as f64 * 1.02,
        "a request following a CANCELED run peaked {:.3} GiB vs {:.3} GiB following a SUCCESSFUL \
         one — the canceled run left resident state behind",
        after_cancel as f64 / GIB,
        after_success as f64 / GIB
    );
}
