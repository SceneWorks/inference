//! **The DiT's 17 input/output tensors, and the whole-model velocity**, against the reference.
//!
//! `tests/fixtures/dit_block.safetensors` carries the whole `state_dict()` plus two goldens
//! computed *through* the projections:
//!
//! | golden | covers |
//! |---|---|
//! | `in.temb` from `in.timestep` | `time_proj` (the flip, the `−0` exponent) **and** `time_embedder.linear_{1,2}` |
//! | `in.refiner.hidden` from `in.refiner.context` | `context_embedder` |
//!
//! …and `out.model.{video,audio}_velocity`, the reference's own whole-model forward, which is what
//! covers the five projections with no golden of their own (`proj_in`, `audio_proj_in`, `norm_out`,
//! `proj_out`, `audio_proj_out`) **and their composition**: the scatter into the packed sequence,
//! the row-major patch order, the two distinct index tensors, and the output heads' row selection.
//! Each of those is a shape-identical mistake on its own.

use crate::common;

use std::collections::BTreeSet;

use candle_gen::candle_core::{DType, Device, Tensor};

use candle_gen_minimax_h3::dit::heads::{
    timestep_sincos, AdaLayerNormOut, DitProjections, TimestepEmbedder,
    TIME_PROJ_DOWNSCALE_FREQ_SHIFT, TIME_PROJ_FLIP_SIN_TO_COS,
};
use candle_gen_minimax_h3::dit::model::{BlockModulation, PackedForward};
use candle_gen_minimax_h3::{
    MiniMaxH3Dit, MiniMaxH3DitConfig, MmRope, TimestepSchedule, MODALITY_NUM, PUBLISHED_DIT_TENSORS,
};

use common::{assert_parity, cosine, dit_fixture_config, flat, rel, weights, Golden, DIT_FIXTURE};

/// **This lane's** bound, as in `dit_parity.rs` — set from the residual this suite measures rather
/// than inherited from the MLX lane's Metal-limited 1e-2.
const TOL: f32 = 1e-4;

fn dev() -> Device {
    Device::Cpu
}

fn fixture() -> Golden {
    Golden::load(DIT_FIXTURE)
}

/// The whole fixture as a weight map — the 17 projections live outside the block/refiner prefixes,
/// so the model needs everything except the reference-side extras.
fn model_weights(f: &Golden) -> candle_gen::Weights {
    weights(f.model_map(&["src.", "in.", "out.", "layout."]))
}

/// **`time_proj` + `time_embedder` reproduce the reference's `temb`.**
///
/// This is the golden that pins the two invisible `Timesteps` settings at once: `flip_sin_to_cos`
/// (cos first) and `downscale_freq_shift = 0`. Both are shape-invisible, and the whole model's
/// modulation — every AdaLN projection in all 50 blocks plus `norm_out` — is a function of this one
/// tensor, so an error here is a uniform bias applied at every step.
#[test]
fn the_timestep_mlp_reproduces_the_reference_embedding() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let w = model_weights(&f);
    let embedder = TimestepEmbedder::from_weights(&w, "time_embedder", &cfg).unwrap();

    let timesteps = flat(&f.tensor("in.timestep"));
    let want = f.tensor("in.temb");
    assert_eq!(want.dims(), &[timesteps.len(), cfg.time_embed_dim]);

    let got = embedder.forward(&timesteps, &dev()).unwrap();
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
    let sincos = timestep_sincos(&timesteps, cfg.freq_dim, &dev()).unwrap();
    let half = cfg.freq_dim / 2;
    let cos_half = sincos.narrow(1, 0, half).unwrap();
    let sin_half = sincos.narrow(1, half, half).unwrap();
    let unflipped = Tensor::cat(&[&sin_half, &cos_half], 1)
        .unwrap()
        .contiguous()
        .unwrap();
    let (mutant_peak, _) = rel(&unflipped, &sincos);
    println!("  unflipped [sin | cos] mutation: rel-max-abs {mutant_peak:.3e}");
    assert!(
        mutant_peak > 0.1,
        "the sin/cos flip must be observable, got {mutant_peak:.3e}"
    );
    let norm = |a: &Tensor| -> f32 { flat(a).iter().map(|v| v * v).sum::<f32>().sqrt() };
    assert!(
        (norm(&unflipped) - norm(&sincos)).abs() / norm(&sincos) < 1e-6,
        "…and the two have the SAME norm, which is why a magnitude gate cannot see the flip"
    );
}

/// **`context_embedder` reproduces the reference's refiner input.**
#[test]
fn the_context_embedder_reproduces_the_reference_projection() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let p = DitProjections::from_weights(&model_weights(&f), &cfg).unwrap();

    let context = f.tensor("in.refiner.context");
    let want = f.tensor("in.refiner.hidden");
    let got = p.context_embedder.forward(&context).unwrap();
    assert_eq!(got.dims(), want.dims());
    assert_parity(&got, &want, TOL, "context_embedder");
    println!(
        "  context_embedder {:?} -> {:?}: rel-max-abs {:.3e}",
        context.dims(),
        got.dims(),
        rel(&got, &want).0
    );
}

/// The whole model reproduces the refiner chain the reference dumped — `context_embedder` **then**
/// `token_refiner`, which is the order the reference's `forward` applies and the order a port that
/// refined before embedding would get wrong at the same shapes.
#[test]
fn the_model_embeds_the_context_before_refining_it() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&model_weights(&f), &cfg, &dev(), DType::F32).unwrap();

    let context = f.tensor("in.refiner.context");
    let want = f.tensor("out.refiner.hidden");
    let got = dit.embed_context(&context).unwrap();
    assert_parity(&got, &want, TOL, "context_embedder then token_refiner");

    // A context of the wrong width is a typed error, not a silent broadcast.
    assert!(dit
        .embed_context(&Tensor::zeros((1, 2, 4), DType::F32, &dev()).unwrap())
        .is_err());
    assert_eq!(dit.num_layers(), cfg.num_layers);
    assert!(dit.holds_adaln(), "nothing is evicted at load");
    assert!(dit.adaln_nbytes() > 0);
}

/// **`norm_out.linear` is `shift` then `scale`**, and it is addressed by the **bare timestep
/// index**.
///
/// Both are shape-identical mistakes. Reversing the halves swaps an additive term for a
/// multiplicative one; feeding it the blocks' `adaln_indices` reads a row `MODALITY_NUM` times too
/// far into the table.
#[test]
fn the_output_norm_is_shift_then_scale_and_keyed_on_the_timestep_alone() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let p = DitProjections::from_weights(&model_weights(&f), &cfg).unwrap();

    let temb = f.tensor("in.temb");
    let m = p.norm_out.modulation(&temb).unwrap();
    assert_eq!(m.shift.dims(), &[temb.dims()[0], cfg.hidden_size]);
    assert_eq!(m.scale.dims(), m.shift.dims());

    // The two halves are genuinely different tensors, so a reversed reading is a different model.
    let (peak, _) = rel(&m.shift, &m.scale);
    println!("  norm_out shift vs scale: rel-max-abs {peak:.3e}");
    assert!(
        peak > 0.1,
        "shift and scale must differ, or their order would be untestable"
    );

    // The row addressing. `adaln_indices` from the fixture is `timestep_indices · 3 + tag`; the
    // recovered timestep index must equal the fixture's own `layout.timestep_indices`.
    let adaln = f.indices("layout.adaln_indices");
    let want_ts = f.u32_vec("layout.timestep_indices");
    let got_ts = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
    assert_eq!(
        got_ts.to_vec1::<u32>().unwrap(),
        want_ts,
        "the recovered timestep index must be the reference's own"
    );

    // ...and it really is a different tensor from the one the blocks use.
    let adaln_v = adaln.to_vec1::<u32>().unwrap();
    let differing = adaln_v
        .iter()
        .zip(&want_ts)
        .filter(|(a, t)| **a != **t)
        .count();
    assert!(
        differing > 0,
        "adaln_indices and timestep_indices must not coincide, or feeding the wrong one would be \
         undetectable"
    );
    println!(
        "  {differing} of {} rows address a different table row under the two index tensors",
        adaln_v.len()
    );

    // Applying it with the blocks' index instead is rejected rather than gathering out of bounds.
    let seq = adaln_v.len();
    let x = Tensor::full(0.5f32, (1, seq, cfg.hidden_size), &dev()).unwrap();
    p.norm_out.apply(&x, &m, &got_ts).expect("the right index");
    let e = p.norm_out.apply(&x, &m, &adaln).unwrap_err().to_string();
    assert!(
        e.contains("BARE timestep index"),
        "the blocks' AdaLN index addresses {MODALITY_NUM}x too far into a {}-row table and must be \
         refused: {e}",
        m.shift.dims()[0]
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
    let classes = [0u32, 1, 2, 3, 0, 1];
    let tags = [0u32, 2, 0, 2, 1, 2];

    for step in 0..schedule.num_steps() {
        let adaln = schedule
            .adaln_indices(step, &classes, &tags, &dev())
            .unwrap();
        let derived = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
        let independent: Vec<u32> = classes
            .iter()
            .map(|&c| schedule.global_timestep_index(step, c as usize).unwrap())
            .collect();
        assert_eq!(
            derived.to_vec1::<u32>().unwrap(),
            independent,
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

    let f = fixture();
    let tiny = dit_fixture_config();
    let p = DitProjections::from_weights(&model_weights(&f), &tiny).unwrap();
    assert_eq!(
        p.time_embedder.time_embed_dim(),
        tiny.time_embed_dim,
        "the AdaLN projections consume this width"
    );
    // Twelve of the seventeen are float32 in the published checkpoint; the fixture is all-f32, so
    // what this can check is that the loader takes the STORED dtype rather than forcing one.
    assert_eq!(p.proj_in.dtype(), DType::F32);
    assert!(p.nbytes() > 0);

    // A projection whose shape disagrees with the config is refused rather than reshaped.
    let mut broken = f.model_map(&["src.", "in.", "out.", "layout."]);
    broken.insert(
        "proj_in.weight".into(),
        Tensor::zeros((2, 2), DType::F32, &dev()).unwrap(),
    );
    assert!(DitProjections::from_weights(&weights(broken), &tiny).is_err());
}

/// The three declared name groups partition the fixture's own published key set exactly — nothing
/// declared twice, nothing in the checkpoint left unread.
#[test]
fn the_declared_names_partition_the_fixture() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let declared: BTreeSet<String> = MiniMaxH3Dit::names(&cfg).into_iter().collect();
    let published: BTreeSet<String> = f
        .keys()
        .into_iter()
        .filter(|k| {
            !k.starts_with("src.")
                && !k.starts_with("in.")
                && !k.starts_with("out.")
                && !k.starts_with("layout.")
        })
        .collect();
    assert_eq!(
        declared, published,
        "the declared tensor set must be exactly the fixture's published one"
    );
    assert_eq!(declared.len(), cfg.num_layers * 12 + 21 + 17);
}

/// **The whole-model golden**: the reference's own `MiniMaxH3Transformer3DModel.forward` over the
/// fixture's packed layout, reproduced by `MiniMaxH3Dit::forward_packed` — which is the very
/// function `JointDit` calls per step.
#[test]
fn the_whole_model_reproduces_the_reference_velocity() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&model_weights(&f), &cfg, &dev(), DType::F32).unwrap();

    let video_rows = f.tensor("in.model.video_rows");
    let audio_rows = f.tensor("in.model.audio_rows");
    let context = f.tensor("in.refiner.context");
    let temb = f.tensor("in.temb");
    let adaln = f.indices("layout.adaln_indices");
    let timestep_indices = f.indices("layout.timestep_indices");
    let position_ids = f.tensor("layout.position_ids");
    let text_indices = f.u32_vec("layout.text_indices");
    let video_indices = f.u32_vec("layout.video_indices");
    let audio_indices = f.u32_vec("layout.audio_indices");

    let text_rows = dit.embed_context(&context).unwrap();
    let tables = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)
        .unwrap()
        .tables(&position_ids)
        .unwrap();
    let norm_out = dit.projections().norm_out.modulation(&temb).unwrap();
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
        .unwrap();

    let want_video = f.tensor("out.model.video_velocity");
    let want_audio = f.tensor("out.model.audio_velocity");
    println!(
        "  whole model {:?} + {:?} -> video rel-max-abs {:.3e} (cosine {:.6}), audio {:.3e} \
         (cosine {:.6})",
        video_rows.dims(),
        audio_rows.dims(),
        rel(&video, &want_video).0,
        cosine(&video, &want_video),
        rel(&audio, &want_audio).0,
        cosine(&audio, &want_audio),
    );
    assert_parity(&video, &want_video, TOL, "whole-model video velocity");
    assert_parity(&audio, &want_audio, TOL, "whole-model audio velocity");

    // **The mutation, measured by the generator itself.** `dump_minimax_h3_dit.py` swaps the two
    // halves of `norm_out.linear` — shape-identical, and diffusers' own `AdaLayerNormContinuous`
    // reads them the other way round in some models — and records how far the output moves. Read it
    // back here so this test's tolerance is known to be far below the defect it must catch.
    let swap: f32 = f
        .meta("norm_out_swap_rel")
        .expect("the generator records the norm_out half-swap negative control")
        .parse()
        .expect("a float");
    println!("  norm_out shift/scale swap negative control: {swap:.3e} (tolerance {TOL:.1e})");
    assert!(
        swap > TOL * 10.0,
        "the norm_out half-swap moves the output by {swap:.3e}, which this test's {TOL:.1e} \
         tolerance could not distinguish from round-off"
    );
    assert_eq!(
        f.meta("provenance"),
        Some("converted-checkpoint"),
        "rule 3: the golden must come from the CONVERTED layout production loads"
    );
}

/// **The AdaLN bounds check is on the production path**, not only in `block.rs`'s unit test of the
/// helper.
///
/// `forward_packed` hoists `check_adaln_indices` out of the 50-block loop — it is a blocking D2H
/// readback and all the blocks gather the same index tensor — which leaves the call as a single
/// deletable line. Deleting it does not make a bad index *pass*: candle's own `index_select` still
/// refuses the gather on the backends that bounds-check. It downgrades the failure from an error
/// naming the modulation table to an opaque kernel one, which is exactly the regression the
/// hoisting invited, so the rejection is asserted here through `forward_packed` itself.
#[test]
fn forward_packed_rejects_an_out_of_range_adaln_index() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&model_weights(&f), &cfg, &dev(), DType::F32).unwrap();

    let temb = f.tensor("in.temb");
    let video_rows = f.tensor("in.model.video_rows");
    let audio_rows = f.tensor("in.model.audio_rows");
    let text_rows = dit.embed_context(&f.tensor("in.refiner.context")).unwrap();
    let timestep_indices = f.indices("layout.timestep_indices");
    let tables = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)
        .unwrap()
        .tables(&f.tensor("layout.position_ids"))
        .unwrap();
    let norm_out = dit.projections().norm_out.modulation(&temb).unwrap();
    let text_indices = f.u32_vec("layout.text_indices");
    let video_indices = f.u32_vec("layout.video_indices");
    let audio_indices = f.u32_vec("layout.audio_indices");

    // The table `forward_packed` derives for the resident path: one row per timestep per modality.
    let rows = temb.dims()[0] * MODALITY_NUM;
    // The reachable caller mistake: `adaln_indices` is `timestep_indices · MODALITY_NUM + tag`, so
    // one stale timestep index lands exactly one row past the end.
    let mut bad = f.u32_vec("layout.adaln_indices");
    assert!(
        bad.iter().all(|&i| (i as usize) < rows),
        "the fixture's own indices must be in range before the perturbation"
    );
    bad[0] = rows as u32;
    let adaln = Tensor::from_vec(bad, (timestep_indices.dims()[0],), &dev()).unwrap();

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

    let e = dit
        .forward_packed(&packed, BlockModulation::Temb(&temb), &norm_out)
        .expect_err("an out-of-range adaln index must not reach the blocks")
        .to_string();
    println!("  out-of-range adaln index (row {rows} of {rows}) -> {e}");
    assert!(
        e.contains("outside the modulation table") && e.contains(&rows.to_string()),
        "the rejection must name the modulation table and its row count, got: {e}"
    );
}

/// The **cached** modulation path reproduces the same whole-model velocity as the resident one.
///
/// This is the numeric-identity half of the AdaLN acceptance criterion, at whole-model scale: an
/// `AdaLnCache` built from the fixture's own `temb` and then driven through `forward_packed` must
/// land on the reference velocity exactly as the un-cached path does. `adaln_evict_memory.rs` is
/// the memory half.
#[test]
fn the_cached_modulation_path_reproduces_the_resident_one() {
    use candle_gen_minimax_h3::AdaLnCache;

    let f = fixture();
    let cfg = dit_fixture_config();
    let dit = MiniMaxH3Dit::from_weights(&model_weights(&f), &cfg, &dev(), DType::F32).unwrap();

    let temb = f.tensor("in.temb");
    let steps = temb.dims()[0];
    // A schedule whose distinct timesteps are exactly the fixture's two, so the cache's table is
    // row-for-row what the resident path projects.
    let ts = flat(&f.tensor("in.timestep"));
    assert_eq!(ts.len(), steps);
    let schedule = TimestepSchedule::new(vec![ts.clone()]).unwrap();
    assert_eq!(schedule.num_distinct_timesteps(), steps);

    let blocks: Vec<_> = (0..cfg.num_layers)
        .map(|i| {
            candle_gen_minimax_h3::DitBlock::from_weights(
                &model_weights(&f),
                &format!("transformer_blocks.{i}"),
                &cfg,
                DType::F32,
            )
            .unwrap()
        })
        .collect();
    let captured = temb.clone();
    let cache = AdaLnCache::precompute(&blocks, schedule, |_| Ok(captured.clone())).unwrap();
    assert_eq!(cache.num_layers(), cfg.num_layers);

    let context = f.tensor("in.refiner.context");
    let text_rows = dit.embed_context(&context).unwrap();
    let tables = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)
        .unwrap()
        .tables(&f.tensor("layout.position_ids"))
        .unwrap();
    let norm_out = dit.projections().norm_out.modulation(&temb).unwrap();
    let adaln = f.indices("layout.adaln_indices");
    let timestep_indices = f.indices("layout.timestep_indices");
    let video_rows = f.tensor("in.model.video_rows");
    let audio_rows = f.tensor("in.model.audio_rows");
    let text_indices = f.u32_vec("layout.text_indices");
    let video_indices = f.u32_vec("layout.video_indices");
    let audio_indices = f.u32_vec("layout.audio_indices");
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

    let (cached_v, cached_a) = dit
        .forward_packed(&packed, BlockModulation::Cached(&cache), &norm_out)
        .unwrap();
    let (resident_v, resident_a) = dit
        .forward_packed(&packed, BlockModulation::Temb(&temb), &norm_out)
        .unwrap();

    // Bit-identical, not merely close: both paths run the same `AdaLnProjection` over the same
    // `temb`, so any difference at all would mean the cache is not the thing it claims to be.
    assert_eq!(
        flat(&cached_v),
        flat(&resident_v),
        "the cached path must be bitwise the resident one"
    );
    assert_eq!(flat(&cached_a), flat(&resident_a));
    // ...and it still hits the reference.
    assert_parity(
        &cached_v,
        &f.tensor("out.model.video_velocity"),
        TOL,
        "cached whole-model video velocity",
    );
    assert_parity(
        &cached_a,
        &f.tensor("out.model.audio_velocity"),
        TOL,
        "cached whole-model audio velocity",
    );
    println!(
        "  cached vs resident: bitwise identical over {} video + {} audio rows; cache {} B for {} \
         layers",
        cached_v.dims()[1],
        cached_a.dims()[1],
        cache.bytes(),
        cache.num_layers()
    );
}
