//! **The DiT's 17 input/output tensors, against the reference** (sc-17147).
//!
//! `tests/fixtures/dit_block.safetensors` was dumped from the official diffusers
//! `MiniMaxH3Transformer3DModel` (sc-17144), and it carries the whole `state_dict()` — including the
//! 17 tensors sc-17144 deliberately did not port — plus two goldens computed *through* them:
//!
//! | golden | covers |
//! |---|---|
//! | `in.temb` from `in.timestep` | `time_proj` (the flip, the `−0` exponent) **and** `time_embedder.linear_{1,2}` |
//! | `in.refiner.hidden` from `in.refiner.context` | `context_embedder` |
//!
//! `proj_in` / `audio_proj_in` / `norm_out` / `proj_out` / `audio_proj_out` have no golden *output*
//! in the fixture, so they are covered here by loading the reference's own weights and asserting the
//! properties a wrong implementation would fail: exact shapes against the published geometry, the
//! `shift`-then-`scale` half order of `norm_out.linear`, and that the `norm_out` row addressing is
//! the bare timestep index rather than the blocks' `timestep · MODALITY_NUM + tag`. Those are the
//! distinctions that are shape-identical and therefore silent — the class this crate's `layout`
//! module exists for.

mod common;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::heads::{
    timestep_sincos, AdaLayerNormOut, DitProjections, TimestepEmbedder,
    TIME_PROJ_DOWNSCALE_FREQ_SHIFT, TIME_PROJ_FLIP_SIN_TO_COS,
};
use mlx_gen_minimax_h3::dit::model::{BlockModulation, PackedForward};
use mlx_gen_minimax_h3::{
    MiniMaxH3Dit, MiniMaxH3DitConfig, MmRope, TimestepSchedule, MODALITY_NUM, PUBLISHED_DIT_TENSORS,
};

use common::{assert_parity, cosine, dit_fixture_config, rel, DIT_FIXTURE};

/// The same `1e-2` `tests/dit_parity.rs` gates the block stack at, and for the same reason:
/// **MLX's Metal matmul runs at reduced precision**, so an f32-vs-f32 comparison against a torch
/// reference floors at ~1e-3 whatever the port does. Measured here: 9.9e-4 for the timestep MLP and
/// 8.0e-4 for `context_embedder`, both at cosine 1.000000.
///
/// That floor is what the gate can resolve, and it bounds what these tests can claim — a defect
/// smaller than 1e-3 relative is invisible to them. The defects this file exists to catch are not
/// of that size: the sin/cos flip moves the embedding by O(1) and is asserted separately as a
/// measured mutation rather than assumed to be catchable.
const TOL: f32 = 1e-2;

fn fixture() -> Weights {
    Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture")
}

/// Load the 17 projections out of the fixture at the tiny geometry.
fn projections(w: &mut Weights, cfg: &MiniMaxH3DitConfig) -> DitProjections {
    DitProjections::from_weights(w, cfg).expect("the 17 input/output tensors")
}

/// **`time_proj` + `time_embedder` reproduce the reference's `temb`.**
///
/// This is the golden that pins the two invisible `Timesteps` settings at once: `flip_sin_to_cos`
/// (cos first) and `downscale_freq_shift = 0`. Both are shape-invisible, and the whole model's
/// modulation — every AdaLN projection in all 50 blocks plus `norm_out` — is a function of this one
/// tensor, so an error here is a uniform bias applied at every step.
#[test]
fn the_timestep_mlp_reproduces_the_reference_embedding() {
    let mut w = fixture();
    let cfg = dit_fixture_config();
    let embedder = TimestepEmbedder::from_weights(&mut w, "time_embedder", &cfg)
        .expect("time_embedder.linear_1 / linear_2");

    let timesteps: Vec<f32> = w
        .require("in.timestep")
        .expect("in.timestep")
        .as_slice::<f32>()
        .to_vec();
    let want = w.require("in.temb").expect("in.temb").clone();
    assert_eq!(want.shape(), &[timesteps.len() as i32, cfg.time_embed_dim]);

    let got = embedder.forward(&timesteps).expect("timestep embedding");
    let (peak, mean) = rel(&got, &want);
    println!(
        "  time_proj + time_embedder over {timesteps:?}: rel-max-abs {peak:.3e} (mean {mean:.3e}, \
         cosine {:.6})",
        cosine(&got, &want)
    );
    assert_parity(&got, &want, TOL, "time_embedder(time_proj(t))");

    // ...and the settings the golden pins are the ones this port declares.
    const { assert!(TIME_PROJ_FLIP_SIN_TO_COS) };
    assert_eq!(TIME_PROJ_DOWNSCALE_FREQ_SHIFT, 0.0);

    // **The mutation.** Flipping the halves back to diffusers' default `[sin | cos]` is
    // shape-identical; it must move the embedding far past the tolerance. Gated on relative
    // max-abs-diff, because the two orders share every value and differ only in position — a norm
    // or a checksum is exactly blind to it.
    let sincos = timestep_sincos(&timesteps, cfg.freq_dim).unwrap();
    let half = cfg.freq_dim / 2;
    let cos_half = mlx_gen_minimax_h3::tensor::slice_axis(&sincos, 1, 0, half).unwrap();
    let sin_half = mlx_gen_minimax_h3::tensor::slice_axis(&sincos, 1, half, cfg.freq_dim).unwrap();
    let unflipped = mlx_rs::ops::concatenate_axis(&[sin_half, cos_half], -1).unwrap();
    let (mutant_peak, _) = rel(&unflipped, &sincos);
    println!("  unflipped [sin | cos] mutation: rel-max-abs {mutant_peak:.3e}");
    assert!(
        mutant_peak > 0.1,
        "the sin/cos flip must be observable, got {mutant_peak:.3e}"
    );
    let norm = |a: &Array| -> f32 {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt()
    };
    assert!(
        (norm(&unflipped) - norm(&sincos)).abs() / norm(&sincos) < 1e-6,
        "…and the two have the SAME norm, which is why a magnitude gate cannot see the flip"
    );
}

/// **`context_embedder` reproduces the reference's refiner input.**
///
/// The fixture's `in.refiner.hidden` is `model.context_embedder(in.refiner.context)`, so this covers
/// the one input projection that is bfloat16 in the published checkpoint rather than float32.
#[test]
fn the_context_embedder_reproduces_the_reference_projection() {
    let mut w = fixture();
    let cfg = dit_fixture_config();
    let p = projections(&mut w, &cfg);

    let context = w.require("in.refiner.context").expect("context").clone();
    let want = w.require("in.refiner.hidden").expect("embedded").clone();
    let got = p.context_embedder.forward(&context).expect("embed");
    assert_eq!(got.shape(), want.shape());
    assert_parity(&got, &want, TOL, "context_embedder");
    println!(
        "  context_embedder {:?} -> {:?}: rel-max-abs {:.3e}",
        context.shape(),
        got.shape(),
        rel(&got, &want).0
    );
}

/// The whole model reproduces the refiner chain the reference dumped — `context_embedder` **then**
/// `token_refiner`, which is the order the reference's `forward` applies and the order a port that
/// refined before embedding would get wrong at the same shapes.
#[test]
fn the_model_embeds_the_context_before_refining_it() {
    let mut w = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&mut w, &cfg, Dtype::Float32).expect("the whole DiT");

    let context = w.require("in.refiner.context").expect("context").clone();
    let want = w.require("out.refiner.hidden").expect("refined").clone();
    let got = dit.embed_context(&context).expect("embed + refine");
    assert_parity(&got, &want, TOL, "context_embedder then token_refiner");

    // A context of the wrong width is a typed error, not a silent broadcast.
    assert!(dit
        .embed_context(&Array::from_slice(&[0.0f32; 8], &[1, 2, 4]))
        .is_err());
    assert_eq!(dit.num_layers(), cfg.num_layers as usize);
    assert!(dit.holds_adaln(), "nothing is evicted at load");
}

/// **`norm_out.linear` is `shift` then `scale`**, and it is addressed by the **bare timestep
/// index**.
///
/// Both are shape-identical mistakes. Reversing the halves swaps an additive term for a
/// multiplicative one; feeding it the blocks' `adaln_indices` reads a row `MODALITY_NUM` times too
/// far into the table and, when the table happens to be long enough, gathers silently.
#[test]
fn the_output_norm_is_shift_then_scale_and_keyed_on_the_timestep_alone() {
    let mut w = fixture();
    let cfg = dit_fixture_config();
    let p = projections(&mut w, &cfg);

    let temb = w.require("in.temb").expect("in.temb").clone();
    let m = p.norm_out.modulation(&temb).expect("norm_out modulation");
    assert_eq!(m.shift.shape(), &[temb.shape()[0], cfg.hidden_size]);
    assert_eq!(m.scale.shape(), m.shift.shape());

    // The two halves are genuinely different tensors, so a reversed reading is a different model.
    let (peak, _) = rel(&m.shift, &m.scale);
    println!("  norm_out shift vs scale: rel-max-abs {peak:.3e}");
    assert!(
        peak > 0.1,
        "shift and scale must differ, or their order would be untestable"
    );

    // The row addressing. `adaln_indices` from the fixture is `timestep_indices · 3 + tag`; the
    // recovered timestep index must equal the fixture's own `layout.timestep_indices`.
    let adaln = w
        .require("layout.adaln_indices")
        .expect("layout.adaln_indices")
        .as_dtype(Dtype::Int32)
        .unwrap();
    let want_ts = w
        .require("layout.timestep_indices")
        .expect("layout.timestep_indices")
        .as_dtype(Dtype::Int32)
        .unwrap();
    let got_ts = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
    assert_eq!(
        got_ts.as_slice::<i32>(),
        want_ts.as_slice::<i32>(),
        "the recovered timestep index must be the reference's own"
    );

    // ...and it really is a different tensor from the one the blocks use: the fixture's sequence
    // carries all three modality tags, so the two disagree on most rows.
    let differing = adaln
        .as_slice::<i32>()
        .iter()
        .zip(got_ts.as_slice::<i32>())
        .filter(|(a, t)| **a != **t)
        .count();
    assert!(
        differing > 0,
        "adaln_indices and timestep_indices must not coincide, or feeding the wrong one would be \
         undetectable"
    );
    println!(
        "  {differing} of {} rows address a different table row under the two index tensors",
        adaln.size()
    );

    // Applying it with the blocks' index instead is rejected rather than gathering out of bounds.
    let x = Array::from_slice(
        &vec![0.5f32; (adaln.size() as i32 * cfg.hidden_size) as usize],
        &[1, adaln.size() as i32, cfg.hidden_size],
    );
    p.norm_out.apply(&x, &m, &got_ts).expect("the right index");
    assert!(
        p.norm_out.apply(&x, &m, &adaln).is_err(),
        "the blocks' AdaLN index addresses {MODALITY_NUM}x too far into a {}-row table and must be \
         refused",
        m.shift.shape()[0]
    );
}

/// The declared timestep index really is what an independent derivation from the schedule produces
/// — the shortcut in `timestep_indices_from_adaln` is pinned, not assumed.
#[test]
fn the_derived_timestep_index_agrees_with_the_schedule() {
    let steps: Vec<Vec<f32>> = (0..6)
        .map(|i| {
            let t = i as f32 / 6.0;
            vec![t, t * 0.5, t.max(0.999), 1.0]
        })
        .collect();
    let schedule = TimestepSchedule::new(steps).unwrap();
    // A sequence with all four classes and all three tags.
    let classes = Array::from_slice(&[0i32, 1, 2, 3, 0, 1], &[6]);
    let tags = Array::from_slice(&[0i32, 2, 0, 2, 1, 2], &[6]);

    for step in 0..schedule.num_steps() {
        let adaln = schedule.adaln_indices(step, &classes, &tags).unwrap();
        let derived = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
        let independent: Vec<i32> = classes
            .as_slice::<i32>()
            .iter()
            .map(|&c| schedule.global_timestep_index(step, c).unwrap())
            .collect();
        assert_eq!(
            derived.as_slice::<i32>(),
            &independent[..],
            "step {step}: the quotient must equal the schedule's own class -> row map"
        );
    }
}

/// The 17 load at the published geometry with the right shapes, and the set really is 17 of the 638.
#[test]
fn the_seventeen_load_at_the_published_geometry() {
    let cfg = MiniMaxH3DitConfig::default();
    let names = DitProjections::names();
    assert_eq!(names.len(), 17);
    assert_eq!(MiniMaxH3Dit::names(&cfg).len(), PUBLISHED_DIT_TENSORS);

    // The tiny fixture's own shapes, which are the published ones scaled down.
    let mut w = fixture();
    let tiny = dit_fixture_config();
    let p = projections(&mut w, &tiny);
    assert_eq!(
        p.time_embedder.time_embed_dim(),
        tiny.time_embed_dim,
        "the AdaLN projections consume this width"
    );
    // Twelve of the seventeen are float32 in the published checkpoint; the fixture is all-f32, so
    // what this can check is that the loader takes the STORED dtype rather than forcing one.
    assert_eq!(p.proj_in.dtype(), Dtype::Float32);
    assert!(p.nbytes() > 0);

    // A projection whose shape disagrees with the config is refused rather than reshaped.
    let mut broken = Weights::empty();
    for n in &names {
        let t = w.require(n).unwrap().clone();
        broken.insert(n.clone(), t);
    }
    broken.insert("proj_in.weight", Array::from_slice(&[0.0f32; 4], &[2, 2]));
    assert!(DitProjections::from_weights(&mut broken, &tiny).is_err());
}

/// **The whole-model golden** (sc-17147): the reference's own
/// `MiniMaxH3Transformer3DModel.forward` over the fixture's packed layout, reproduced by
/// [`MiniMaxH3Dit::forward_packed`] — which is the very function `JointDit` calls per step.
///
/// This is the gate that covers the five projections with no golden of their own (`proj_in`,
/// `audio_proj_in`, `norm_out`, `proj_out`, `audio_proj_out`) **and** their composition: the
/// scatter into the packed sequence, the row-major patch order, the two distinct index tensors, and
/// the output heads' row selection. Each of those is a shape-identical mistake on its own.
#[test]
fn the_whole_model_reproduces_the_reference_velocity() {
    let mut w = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&mut w, &cfg, Dtype::Float32).expect("the whole DiT");

    let video_rows = w.require("in.model.video_rows").unwrap().clone();
    let audio_rows = w.require("in.model.audio_rows").unwrap().clone();
    let context = w.require("in.refiner.context").unwrap().clone();
    let temb = w.require("in.temb").unwrap().clone();
    let adaln = w
        .require("layout.adaln_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let timestep_indices = w
        .require("layout.timestep_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let position_ids = w.require("layout.position_ids").unwrap().clone();
    let idx = |k: &str| -> Vec<i32> {
        w.require(k)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice::<i32>()
            .to_vec()
    };
    let (text_indices, video_indices, audio_indices) = (
        idx("layout.text_indices"),
        idx("layout.video_indices"),
        idx("layout.audio_indices"),
    );

    let text_rows = dit.embed_context(&context).expect("context rows");
    let tables = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)
        .unwrap()
        .tables(&position_ids, Dtype::Float32)
        .unwrap();
    let norm_out = dit
        .projections()
        .norm_out
        .modulation(&temb)
        .expect("norm_out modulation");
    let packed = PackedForward {
        video_rows: &video_rows,
        audio_rows: &audio_rows,
        text_rows: &text_rows,
        adaln_indices: &adaln,
        timestep_indices: &timestep_indices,
        tables: &tables,
        text_indices: &text_indices,
        video_indices: &video_indices,
        audio_indices: &audio_indices,
    };
    let (video, audio) = dit
        .forward_packed(&packed, BlockModulation::Temb(&temb), &norm_out)
        .expect("whole-model forward");

    let want_video = w.require("out.model.video_velocity").unwrap().clone();
    let want_audio = w.require("out.model.audio_velocity").unwrap().clone();
    println!(
        "  whole model {:?} + {:?} -> video rel-max-abs {:.3e} (cosine {:.6}), audio {:.3e} \
         (cosine {:.6})",
        video_rows.shape(),
        audio_rows.shape(),
        rel(&video, &want_video).0,
        cosine(&video, &want_video),
        rel(&audio, &want_audio).0,
        cosine(&audio, &want_audio),
    );
    assert_parity(&video, &want_video, TOL, "whole-model video velocity");
    assert_parity(&audio, &want_audio, TOL, "whole-model audio velocity");

    // **The mutation, measured by the generator itself.** `tools/dump_minimax_h3_dit.py` swaps the
    // two halves of `norm_out.linear` — shape-identical, and diffusers' own `AdaLayerNormContinuous`
    // reads them the other way round in some models — and records how far the output moves. Read it
    // back here so this test's tolerance is known to be far below the defect it must catch.
    let swap = w
        .metadata("norm_out_swap_rel")
        .expect("the generator records the norm_out half-swap negative control")
        .parse::<f32>()
        .expect("a float");
    println!("  norm_out shift/scale swap negative control: {swap:.3e} (tolerance {TOL:.1e})");
    assert!(
        swap > TOL * 10.0,
        "the norm_out half-swap moves the output by {swap:.3e}, which this test's {TOL:.1e} \
         tolerance could not distinguish from round-off"
    );
    assert_eq!(
        w.metadata("provenance"),
        Some("converted-checkpoint"),
        "rule 3: the golden must come from the CONVERTED layout production loads"
    );
}
