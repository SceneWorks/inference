//! SW-7094 — **rung 3 (`BoundedAttention`) at image geometry** on the real MiniMax-H3-Image DiT.
//!
//! The image family implements rung 3 on MLX (`mlx_gen_minimax_h3::h3_image::memory`): every
//! block's attention runs as whole-head chunks at the declared 64 Mi budget, floored at one whole
//! head. An earlier verdict declared it not applicable; this gate then measured the head axis at
//! 64 Mi saving 1.05 % of the q4 denoise peak at 2048² with a bit-identical velocity (run
//! 43042251357). It now asserts the implemented rung on the shipped 4 MP canvas and the 16 MP
//! canvas (16,384 image rows), each with a 512-token prompt, through the registered image DiT's own
//! `ImageDenoiser` on the q4 tier:
//!
//! * the plan is the one the generator runs — derived from the contract's own rung-3 ladder
//!   selection through `generation_memory` and `requested_bounded_attention`, not hand-built;
//! * the premise: the unbounded arm ran one call, and the rung-3 arm split the head axis into the
//!   planned whole-head chunks and never fell back to query rows — so the comparison is not
//!   vacuous;
//! * the rung **lowers** the steady-state denoise peak;
//! * the velocity is **bit-identical** to the unbounded one (whole-head chunks keep every call's
//!   full query axis — the head axis's exact contract).
//!
//! The weights are real; the text rows and latents are synthetic (peak memory does not read
//! values, and both arms see identical inputs). Nothing measured here is recorded as a memory
//! anchor — E13's two anchors per tier are taken once at the end of the feature.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis};
use mlx_gen::gen_core::{LoadSpec, MemoryNumericTier, MemoryStrategy, Quant, WeightsSource};
use mlx_gen::testing::{max_abs, max_abs_diff_f32};
use mlx_gen_minimax_h3::denoise::schedule::SigmaSchedule;
use mlx_gen_minimax_h3::dit::config::MiniMaxH3ImageDitConfig;
use mlx_gen_minimax_h3::dit::layers::attention_probe;
use mlx_gen_minimax_h3::h3_image::memory::{
    image_attention_heads_per_chunk, requested_bounded_attention, weights_free_contract,
};
use mlx_gen_minimax_h3::h3_image::{
    admission, ImageDenoiser, ImageLayout, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT,
};

/// Prompt rows of every cell.
const NUM_TEXT_TOKENS: i32 = 512;

/// Steps per arm: the first-call allocator artifact against the steady state.
const EVALS: usize = 2;

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
    /// the load and the precompute sit outside the measured window. `plan` is `None` for the
    /// unbounded forward the generator runs when rung 3 is not engaged.
    fn arm(
        &self,
        dir: &std::path::Path,
        plan: Option<(AttentionBudget, AttentionChunkAxis)>,
    ) -> Arm {
        let dit = MiniMaxH3ImageDit::load_dir(dir, Dtype::Bfloat16).expect("image dit loads");
        let mut denoiser = ImageDenoiser::new(
            dit,
            self.layout.clone(),
            &self.context,
            SigmaSchedule::h3_image_distilled().unwrap(),
        )
        .expect("denoiser");
        if let Some((budget, axis)) = plan {
            denoiser.set_bounded_attention(budget, axis);
        }
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

/// The rung-3 plan the generator runs for a request of `seq` packed rows: the contract's own
/// rung-3 ladder selection, through `generation_memory`, through the generator's resolver.
fn production_plan(seq: u64) -> (AttentionBudget, AttentionChunkAxis) {
    let contract =
        weights_free_contract(&LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))).unwrap();
    let tier = MemoryNumericTier {
        precision: mlx_gen::Precision::Bf16,
        quant: Some(Quant::Q4),
        component_precision_floors: &[],
    };
    let selection = admission::ladder(&contract, tier)
        .into_iter()
        .find(|s| s.strategy == MemoryStrategy::BoundedAttention)
        .expect("rung 3 is on the ladder");
    let memory = contract.generation_memory(&selection);
    assert!(memory.is_some_and(|m| m.chunk_attention), "{memory:?}");
    requested_bounded_attention(memory, seq)
        .unwrap()
        .expect("the rung-3 selection chunks attention")
}

/// Rung 3 lowers the denoise peak at image geometry and leaves the velocity bit-identical.
#[test]
#[ignore = "SW-7094: needs MINIMAX_H3_IMAGE_TIERS_ROOT (the converted q4 image DiT) + Metal; loads the 50-block DiT once per arm"]
fn h3_image_rung3_lowers_the_denoise_peak_at_image_geometry() {
    let dir = q4_dit_dir();
    let cfg = MiniMaxH3ImageDitConfig::default();
    let heads = cfg.base.num_attention_heads as usize;
    for edge in [2048u32, 4096] {
        let cell = Cell::new(edge, &cfg);
        let seq = cell.layout.seq_len();
        let plan = production_plan(seq as u64);
        assert_eq!(plan.1, AttentionChunkAxis::Heads);
        let per_chunk = image_attention_heads_per_chunk(plan.0, heads as u64, seq as u64) as usize;
        let chunks = heads.div_ceil(per_chunk);
        eprintln!(
            "\n=== q4 | {edge}x{edge} | packed seq {seq} | {heads} heads | {per_chunk} per chunk ==="
        );
        let unbounded = cell.arm(&dir, None);
        assert_eq!((unbounded.head_chunks, unbounded.query_chunks), (1, 1));
        assert_eq!(unbounded.seq as i32, seq, "the probe saw another sequence");
        let rung3 = cell.arm(&dir, Some(plan));
        // The premise: the rung split the head axis as planned and never fell back to query rows.
        assert!(chunks > 1, "the plan does not chunk at seq {seq}");
        assert_eq!(
            (rung3.head_chunks, rung3.query_chunks),
            (chunks, 1),
            "the rung-3 arm did not run the planned whole-head chunks at seq {seq}"
        );
        assert_eq!(rung3.seq as i32, seq, "the probe saw another sequence");

        let rel = relative_max_abs(&unbounded.velocity, &rung3.velocity);
        let delta = rung3.peak as f64 / unbounded.peak as f64 - 1.0;
        eprintln!(
            "  unbounded  peak {:8.4} GB\n  rung 3     peak {:8.4} GB  {:+6.2}% peak  rel|Δ| {rel:.3e}",
            unbounded.peak as f64 / 1e9,
            rung3.peak as f64 / 1e9,
            delta * 100.0
        );
        assert!(
            rel == 0.0,
            "rung 3 at {edge}² moved the velocity by {rel:e} relative; whole-head chunks are exact"
        );
        assert!(
            rung3.peak < unbounded.peak,
            "rung 3 at {edge}² did not lower the denoise peak ({:+.3}%)",
            delta * 100.0
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_bounded_attention_real::h3_image_rung3_lowers_the_denoise_peak_at_image_geometry"
    );
}
