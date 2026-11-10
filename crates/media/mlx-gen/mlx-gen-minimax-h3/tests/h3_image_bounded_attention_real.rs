//! SW-7094 — **rung 3 (`BoundedAttention`) re-measured at image geometry** on the real
//! MiniMax-H3-Image DiT.
//!
//! The image family declares rung 3 `StructurallyNotApplicable` on MLX
//! (`mlx_gen_minimax_h3::h3_image::memory`) on two grounds: MLX's fused SDPA never builds the
//! `[B, H, S, S]` scores, and H3 measured every chunked arm costing denoise peak on the same block
//! stack at video geometry (sc-18661, `tests/bounded_attention_real.rs`). The requirement is that
//! the verdict stands for this family **only** if a measurement at image geometry confirms it.
//! This is that measurement: the shipped 4 MP canvas and the 16 MP canvas (16,384 image rows), each
//! with a 512-token prompt, through the registered image DiT's own `ImageDenoiser` on the q4 tier
//! (the tier where an activation-side rung has its best chance — see H3's harness).
//!
//! Four arms per cell — unbounded, heads at the shipped 64 Mi budget, query rows at 64 Mi, and one
//! head per chunk (`B · S²`) — with the engagement premise asserted from the always-on
//! `attention_probe` before any peak is compared. **It fails if any chunked arm saves more than
//! 0.5 % of the unbounded denoise peak**, which is the condition under which the declaration must
//! become an implementation.
//!
//! The weights are real; the text rows and latents are synthetic (peak memory does not read
//! values, and every arm sees identical inputs). Nothing measured here is recorded as a memory
//! anchor — E13's two anchors per tier are taken once at the end of the feature.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis};
use mlx_gen::testing::{max_abs, max_abs_diff_f32};
use mlx_gen_minimax_h3::denoise::schedule::SigmaSchedule;
use mlx_gen_minimax_h3::dit::config::MiniMaxH3ImageDitConfig;
use mlx_gen_minimax_h3::dit::layers::attention_probe;
use mlx_gen_minimax_h3::h3_image::{
    ImageDenoiser, ImageLayout, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT,
};

/// Prompt rows of every cell.
const NUM_TEXT_TOKENS: i32 = 512;

/// Steps per arm: the first-call allocator artifact against the steady state.
const EVALS: usize = 2;

/// A chunked arm may not save more than this fraction of the unbounded peak.
const SAVING_BAND: f64 = 0.005;

fn q4_dit_dir() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_TIERS_ROOT").unwrap_or_default();
    assert!(
        !raw.trim().is_empty(),
        "MINIMAX_H3_IMAGE_TIERS_ROOT must hold the converted q4 tier (q4/{IMAGE_DIT_COMPONENT}/). \
         This test is #[ignore]d and asserts rather than skips."
    );
    let dir = PathBuf::from(raw.trim())
        .join("q4")
        .join(IMAGE_DIT_COMPONENT);
    assert!(
        dir.join("config.json").is_file(),
        "{} has no config.json",
        dir.display()
    );
    dir
}

fn drive(shape: &[i32], seed: u64, dtype: Dtype) -> Array {
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
        .unwrap()
        .as_dtype(dtype)
        .unwrap()
}

struct Arm {
    peak: u64,
    velocity: Array,
    head_chunks: usize,
    query_chunks: usize,
    seq: usize,
}

struct Cell {
    layout: ImageLayout,
    context: Array,
    rows: Array,
}

impl Cell {
    fn new(edge: u32, cfg: &MiniMaxH3ImageDitConfig) -> Self {
        let latent = (edge / 16) as i32;
        let layout =
            ImageLayout::t2i(NUM_TEXT_TOKENS, latent, latent, cfg.base.patch_size).expect("layout");
        let context = drive(
            &[1, NUM_TEXT_TOKENS, cfg.base.text_dim],
            103,
            Dtype::Bfloat16,
        );
        let rows = drive(
            &[
                1,
                layout.num_packed_image_rows(),
                cfg.base.video_patch_dim(),
            ],
            101,
            Dtype::Float32,
        );
        mlx_rs::transforms::eval([&context, &rows]).unwrap();
        Self {
            layout,
            context,
            rows,
        }
    }

    /// One arm: the DiT is reloaded per arm (the denoiser consumes it and evicts `adaln_proj`), and
    /// the load and the precompute sit outside the measured window.
    fn arm(&self, dir: &std::path::Path, budget: AttentionBudget, axis: AttentionChunkAxis) -> Arm {
        let dit = MiniMaxH3ImageDit::load_dir(dir, Dtype::Bfloat16).expect("image dit loads");
        let mut denoiser = ImageDenoiser::new(
            dit,
            self.layout.clone(),
            &self.context,
            SigmaSchedule::h3_image_distilled().unwrap(),
        )
        .expect("denoiser");
        denoiser.set_bounded_attention(budget, axis);
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let mut last = None;
        for step in 0..EVALS {
            attention_probe::reset();
            let v = denoiser.velocity(step, &self.rows).expect("velocity");
            mlx_rs::transforms::eval([&v]).unwrap();
            last = Some(v);
        }
        let peak = mlx_rs::memory::get_peak_memory() as u64;
        Arm {
            peak,
            velocity: last.unwrap(),
            head_chunks: attention_probe::last_head_chunks(),
            query_chunks: attention_probe::last_query_chunks(),
            seq: attention_probe::last_seq(),
        }
    }
}

fn relative_max_abs(a: &Array, b: &Array) -> f32 {
    let scale = max_abs(a);
    assert!(scale > 1e-3, "the reference arm is degenerate ({scale})");
    max_abs_diff_f32(a, b) / scale
}

/// The verdict input for the image family's rung-3 declaration.
#[test]
#[ignore = "SW-7094: needs MINIMAX_H3_IMAGE_TIERS_ROOT (the converted q4 image DiT) + Metal; loads the 50-block DiT once per arm"]
fn h3_image_rung3_verdict_holds_at_image_geometry() {
    let dir = q4_dit_dir();
    let cfg = MiniMaxH3ImageDitConfig::default();
    let heads = cfg.base.num_attention_heads as usize;
    for edge in [2048u32, 4096] {
        let cell = Cell::new(edge, &cfg);
        let seq = cell.layout.seq_len();
        eprintln!("\n=== q4 | {edge}x{edge} | packed seq {seq} | {heads} heads ===");
        let unbounded = cell.arm(&dir, AttentionBudget::UNBOUNDED, AttentionChunkAxis::Heads);
        assert_eq!((unbounded.head_chunks, unbounded.query_chunks), (1, 1));
        assert_eq!(unbounded.seq as i32, seq, "the probe saw another sequence");
        let one_head = AttentionBudget::from_score_elements((seq as u64).pow(2), true);
        let by_heads = cell.arm(
            &dir,
            AttentionBudget::CONSTRAINED,
            AttentionChunkAxis::Heads,
        );
        let by_rows = cell.arm(
            &dir,
            AttentionBudget::CONSTRAINED,
            AttentionChunkAxis::QueryRows,
        );
        let exact = cell.arm(&dir, one_head, AttentionChunkAxis::Heads);
        // The premise: every bounded arm split on its own axis, or the comparison is vacuous.
        assert!(
            by_heads.head_chunks > 1,
            "the head axis did not split at seq {seq}"
        );
        assert!(
            by_rows.query_chunks > 1,
            "the query-row axis did not split at seq {seq}"
        );
        assert_eq!(by_rows.head_chunks, 1);
        assert_eq!((exact.head_chunks, exact.query_chunks), (heads, 1));

        eprintln!(
            "  unbounded           peak {:8.4} GB",
            unbounded.peak as f64 / 1e9
        );
        for (label, arm, rel_bound) in [
            ("heads @ 64Mi", &by_heads, 1e-2f32),
            ("query rows @ 64Mi", &by_rows, 1e-2),
            ("heads @ one-head", &exact, 0.0),
        ] {
            let rel = relative_max_abs(&unbounded.velocity, &arm.velocity);
            let delta = arm.peak as f64 / unbounded.peak as f64 - 1.0;
            eprintln!(
                "  {label:<19} peak {:8.4} GB  {:>3} head x {:>3} row  {:+6.2}% peak  rel|Δ| {rel:.3e}",
                arm.peak as f64 / 1e9,
                arm.head_chunks,
                arm.query_chunks,
                delta * 100.0
            );
            assert!(
                rel <= rel_bound,
                "{label} at {edge}² moved the velocity by {rel:e} relative — a wiring error"
            );
            assert!(
                delta > -SAVING_BAND,
                "{label} at {edge}² SAVED {:.3}% of denoise peak — rung 3 is declared \
                 StructurallyNotApplicable for minimax_h3_image on the grounds that it cannot. \
                 Implement the rung instead.",
                -delta * 100.0
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_bounded_attention_real::h3_image_rung3_verdict_holds_at_image_geometry"
    );
}
