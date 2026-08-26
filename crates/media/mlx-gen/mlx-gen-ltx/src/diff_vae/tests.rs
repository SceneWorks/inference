//! Unit tests for the LTX-2.5 `NADiffusionDecoder` port (sc-18766).
//!
//! Two layers. The **arithmetic** tests pin the geometry algebra — window bounds, tile floors,
//! halos, interval propagation, blend masks — against values derived by hand from the released
//! checkpoint's own config, so a transcription slip in the ladder is a red test rather than a
//! quietly mis-sized volume. The **numeric** tests run a miniature decoder on deterministic
//! synthetic weights: [`na3d`] against a brute-force masked softmax over the whole grid, and a
//! tiled decode against an untiled one where the two must agree exactly (interior) rather than
//! approximately.

use super::*;
use mlx_rs::ops::{abs, max as max_op, softmax_axes, sum_axes};
use std::collections::HashMap;

/// The released `vae` block, verbatim from `vae/ltx-2.5-video-vae-bf16.safetensors`
/// (`docs/reference/sc-18756-headers/vae/`), trimmed to the decoder-relevant keys.
const RELEASED_VAE: &str = r#"{
  "_class_name": "CausalDiffusionVAE",
  "dims": 3,
  "encoder": { "_class_name": "Encoder", "out_channels": 128, "patch_size": 4 },
  "decoder": {
    "_class_name": "NADiffusionDecoder",
    "in_channels": 128,
    "out_channels": 3,
    "patch_size": 4,
    "head_dim": 64,
    "stage_channels": [2048, 1024, 512, 512, 256],
    "stage_depths": [4, 6, 4, 2, 8],
    "stage_kernels": [[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]],
    "upsamples": [[[1,2,2],2],[[2,1,1],2],[[2,2,2],1],[[2,2,2],2]],
    "spatial_padding_mode": "zeros",
    "resampler_kind": "linear",
    "stage5_kernel": [11,11,11],
    "timestep_scale_multiplier": 1000.0,
    "default_num_inference_steps": 1
  },
  "model_output_type": "x0",
  "spatial_padding_mode": "zeros"
}"#;

fn released() -> NaDiffusionDecoderConfig {
    NaDiffusionDecoderConfig::from_embedded_vae(&serde_json::from_str(RELEASED_VAE).unwrap())
        .expect("the released vae block must parse")
}

fn max_abs(a: &Array) -> f32 {
    max_op(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn mean_abs(a: &Array) -> f32 {
    mlx_rs::ops::mean(abs(a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

// ---------------------------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------------------------

#[test]
fn released_config_parses_to_the_measured_structure() {
    let cfg = released();
    assert_eq!(cfg.in_channels, 128);
    assert_eq!(cfg.out_channels, 3);
    assert_eq!(cfg.patch_size, 4);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.stage_channels, vec![2048, 1024, 512, 512, 256]);
    assert_eq!(cfg.stage_depths, vec![4, 6, 4, 2, 8]);
    assert_eq!(cfg.stage5_kernel, [11, 11, 11]);
    assert_eq!(cfg.upsamples.len(), 4);
    assert_eq!(cfg.upsamples[1], ([2, 1, 1], 2));
    assert_eq!(cfg.model_output_type, ModelOutputType::X0);
    assert_eq!(cfg.default_num_inference_steps, 1);
    assert_eq!(cfg.timestep_scale_multiplier, 1000.0);
    assert_eq!(cfg.t_emb_dim, 384);
    assert_eq!(cfg.stage5_width(), 256);
}

#[test]
fn a_conv_vae_block_is_refused_rather_than_defaulted() {
    let conv = serde_json::json!({
        "_class_name": "CausalVideoAutoencoder",
        "latent_channels": 128,
        "decoder_blocks": [["res_x", {"num_layers": 4}]]
    });
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&conv)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no `decoder` block"), "got: {err}");

    // A `decoder` block that is not an NADiffusionDecoder must not be run as one either.
    let wrong_class = serde_json::json!({ "decoder": { "_class_name": "ConvVideoDecoder" } });
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&wrong_class)
        .unwrap_err()
        .to_string();
    assert!(err.contains("NADiffusionDecoder"), "got: {err}");
}

#[test]
fn structural_fields_are_required_not_defaulted() {
    // There has never been a default NADiffusionDecoder ladder, so a missing one is an error and
    // not a guess. Drop each required key in turn.
    for key in [
        "stage_channels",
        "stage_depths",
        "stage_kernels",
        "upsamples",
        "stage5_kernel",
        // Not shape, but sampler: defaulting this to 1.0 against the released 1000.0 embeds every
        // timestep at the wrong frequency and decodes silently wrongly (sc-18767).
        "timestep_scale_multiplier",
    ] {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
        v["decoder"].as_object_mut().unwrap().remove(key);
        let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
            .expect_err(&format!("dropping {key} must be an error"))
            .to_string();
        assert!(
            err.contains(key),
            "refusal for {key} must name it, got: {err}"
        );
    }

    // `model_output_type` sits on the `vae` block rather than on `decoder`. Defaulting it to `v`
    // against the released `x0` checkpoint changes what the stage-5 blocks are taken to predict.
    let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    v.as_object_mut().unwrap().remove("model_output_type");
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("dropping model_output_type must be an error")
        .to_string();
    assert!(
        err.contains("model_output_type"),
        "refusal must name it, got: {err}"
    );
}

#[test]
fn inconsistent_ladders_are_refused() {
    let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    v["decoder"]["stage_depths"] = serde_json::json!([4, 6, 4, 2]);
    assert!(NaDiffusionDecoderConfig::from_embedded_vae(&v).is_err());

    let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    v["decoder"]["upsamples"] = serde_json::json!([[[1, 2, 2], 2]]);
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .unwrap_err()
        .to_string();
    assert!(err.contains("upsample hops"), "got: {err}");

    // Every stage width must be a whole number of heads.
    let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    v["decoder"]["stage_channels"] = serde_json::json!([2048, 1024, 512, 512, 200]);
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .unwrap_err()
        .to_string();
    assert!(err.contains("head_dim"), "got: {err}");

    // `v` and `x0` are the only parameterisations; anything else changes the sampler silently.
    let mut v: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    v["model_output_type"] = serde_json::json!("epsilon");
    assert!(NaDiffusionDecoderConfig::from_embedded_vae(&v).is_err());
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

#[test]
fn latent_floor_is_the_widest_window_seen_through_the_upsample_ladder() {
    // Cumulative strides are (1,1,1) (1,2,2) (2,2,2) (4,4,4) (8,8,8); the binding constraints are
    // stage 1's 3x7x7 at the latent grid itself and stage 5's 11x11x11 seen through 8x.
    assert_eq!(released().min_latent_shape(), [3, 7, 7]);
}

#[test]
fn stage4_and_stage5_geometry_match_the_reference_arithmetic() {
    let cfg = released();
    // 768x512x25 -> latent 4x16x24. Hops: x(1,2,2), then x(2,1,1) less the duplicate frame, then
    // x(2,2,2) less the duplicate frame.
    assert_eq!(cfg.stage4_shape(4, 16, 24), [13, 64, 96]);
    assert_eq!(
        cfg.stage5_pixel_shape([13, 64, 96], true, true),
        [25, 512, 768]
    );
    assert_eq!(cfg.noise_shape(4, 16, 24), [25, 512, 768]);
    // A non-origin tile keeps its duplicate leading frame (it has none of its own to drop).
    assert_eq!(cfg.stage5_pixel_shape([6, 8, 8], false, false)[0], 12);
    // 1280x704x25 -> latent 4x22x40.
    assert_eq!(cfg.noise_shape(4, 22, 40), [25, 704, 1280]);
    // A below-floor latent reports the geometry it will actually run at, not the one asked for.
    assert_eq!(cfg.noise_shape(1, 2, 2), cfg.noise_shape(3, 7, 7));
    // The ladder's own hops multiply out to the pinned 8/32/32 the rest of LTX assumes.
    assert_eq!(cfg.pixel_scale(), VIDEO_SCALE_FACTORS);
}

#[test]
fn tile_floors_and_halos_come_from_the_stage_4_and_5_windows() {
    let cfg = released();
    // Stage 4's own 3x5x5, and stage 5's 11x11x11 divided by the last hop's 2x2x2 stride.
    assert_eq!(cfg.min_tile_shape(), [6, 6, 6]);
    // halo4 = depth 2 * (5/2) = 4 spatially, 2 temporally; halo5 = ceil(8 * 5 / 2) = 20.
    assert_eq!(cfg.tile_halo(), [20, 20, 20]);
    assert_eq!(cfg.ghost_latent_frames(), 2);
}

#[test]
fn window_starts_shift_inward_at_the_border() {
    // NATTEN keeps the window full-size and slides it in; it does not clip and renormalise.
    assert_eq!(window_starts(8, 3), vec![0, 0, 1, 2, 3, 4, 5, 5]);
    assert_eq!(window_starts(5, 5), vec![0, 0, 0, 0, 0]);
    // A window wider than the axis collapses to the whole axis.
    assert_eq!(window_starts(3, 11), vec![0, 0, 0]);
    // Odd/even kernels both centre with a floor.
    assert_eq!(window_starts(6, 4), vec![0, 0, 0, 1, 2, 2]);
}

#[test]
fn query_tiles_stay_under_the_budget_and_never_collapse_to_nothing() {
    for (dims, kernel) in [
        ([25, 128, 192], [11, 11, 11]),
        ([6, 16, 24], [3, 7, 7]),
        ([21, 64, 96], [3, 5, 5]),
        ([3, 7, 7], [3, 7, 7]),
    ] {
        let mut kernels = [0i32; 3];
        for a in 0..3 {
            kernels[a] = kernel[a].min(dims[a]);
        }
        let tiles = pick_tiles(dims, kernels);
        assert!(tiles.iter().all(|&t| t >= 1), "{dims:?} -> {tiles:?}");
        let nq: i64 = tiles.iter().map(|&x| x as i64).product();
        let nk: i64 = (0..3)
            .map(|a| dims[a].min(tiles[a] + kernels[a] - 1) as i64)
            .product();
        let all_singleton = tiles.iter().all(|&t| t == 1);
        assert!(
            nq * nk <= NA_TILE_BUDGET || all_singleton,
            "{dims:?} kernel {kernel:?} -> tiles {tiles:?} cost {}",
            nq * nk
        );
    }
}

#[test]
fn split_and_propagate_reproduce_the_reference_intervals() {
    let split = split_by_size(13, 6, 3, 4);
    assert_eq!(split.len(), 4);
    assert_eq!((split[0].start, split[0].end), (0, 6));
    assert_eq!((split[0].left_ramp, split[0].right_ramp), (0, 3));
    assert_eq!(
        (split.last().unwrap().start, split.last().unwrap().end),
        (9, 13)
    );
    assert_eq!(split.last().unwrap().right_ramp, 0);
    // Contiguous cover with the declared overlap.
    for pair in split.windows(2) {
        assert_eq!(pair[0].end - pair[1].start, 3);
    }
    validate_split(&split, 13, 4, 0).unwrap();
    // A dimension that fits in one tile is one untiled interval with no ramps.
    let single = split_by_size(5, 6, 3, 4);
    assert_eq!(single.len(), 1);
    assert_eq!((single[0].left_ramp, single[0].right_ramp), (0, 0));

    // A short trailing tile is grown leftward rather than left as a sliver stage 4/5 cannot
    // attend over — and its neighbour's ramp widens to keep the blend complementary.
    let grown = split_by_size(32, 31, 1, 6);
    assert_eq!(grown.len(), 2);
    assert_eq!((grown[1].start, grown[1].end), (26, 32));
    assert_eq!(grown[0].right_ramp, 5);
    assert_eq!(grown[1].left_ramp, 5);
    validate_split(&grown, 32, 6, 2).unwrap();

    // Temporal propagation carries the pixel-shuffle duplicate-frame drop; the origin tile keeps
    // its start at 0 while later tiles shift back by one.
    let origin = propagate(split[0], 2, true);
    assert_eq!((origin.start, origin.end), (0, 11));
    let later = propagate(split[1], 2, true);
    assert_eq!((later.start, later.end), (5, 17));
    // Spatial propagation is exact scaling.
    assert_eq!(
        (
            propagate(split[1], 2, false).start,
            propagate(split[1], 2, false).end
        ),
        (6, 18)
    );
}

#[test]
fn trapezoid_ramps_of_neighbouring_tiles_sum_to_one() {
    let left = trapezoid(10, 0, 4);
    let right = trapezoid(10, 4, 0);
    // The last 4 of `left` overlap the first 4 of `right`.
    for i in 0..4 {
        let sum = left[6 + i] + right[i];
        assert!((sum - 1.0).abs() < 1e-6, "overlap slot {i} sums to {sum}");
    }
    assert!(left[..6].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    assert!(right[4..].iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn rope_split_matches_the_reference_default() {
    assert_eq!(rope_dim_split(64).unwrap(), [16, 24, 24]);
    assert_eq!(rope_dim_split(16).unwrap(), [4, 6, 6]);
    assert!(rope_dim_split(12).is_err());
}

#[test]
fn resize_axis_pads_and_crops_with_the_declared_policy() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
    let (grown, pad) = resize_axis(&x, 1, 5, false).unwrap();
    assert_eq!(pad, (0, 2));
    assert_eq!(grown.as_slice::<f32>(), &[1.0, 2.0, 3.0, 3.0, 3.0]);

    let (grown, pad) = resize_axis(&x, 1, 6, true).unwrap();
    assert_eq!(pad, (1, 2));
    assert_eq!(grown.as_slice::<f32>(), &[1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);

    let (cropped, pad) = resize_axis(&grown, 1, 3, true).unwrap();
    assert_eq!(pad, (1, 2));
    assert_eq!(cropped.as_slice::<f32>(), &[1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------------------------
// Neighborhood attention
// ---------------------------------------------------------------------------------------------

/// Brute-force NA: full `[N, N]` attention over the whole grid with the window as an additive
/// mask. Only usable on tiny grids, which is the point — it shares no code with [`na3d`]'s tiling.
fn na3d_reference(q: &Array, k: &Array, v: &Array, kernel: [i32; 3]) -> Array {
    let sh = q.shape().to_vec();
    let (b, t, h, w, nh, hd) = (sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]);
    let dims = [t, h, w];
    let starts: Vec<Vec<i32>> = (0..3)
        .map(|a| window_starts(dims[a], kernel[a].min(dims[a])))
        .collect();
    let kernels: Vec<i32> = (0..3).map(|a| kernel[a].min(dims[a])).collect();

    let n = (t * h * w) as usize;
    let mut mask = vec![0.0f32; n * n];
    let mut qi = 0usize;
    for qt in 0..t {
        for qh in 0..h {
            for qw in 0..w {
                let mut ki = 0usize;
                for kt in 0..t {
                    for kh in 0..h {
                        for kw in 0..w {
                            let inside = (0..3).all(|a| {
                                let (qidx, kidx) = match a {
                                    0 => (qt, kt),
                                    1 => (qh, kh),
                                    _ => (qw, kw),
                                };
                                let lo = starts[a][qidx as usize];
                                kidx >= lo && kidx < lo + kernels[a]
                            });
                            if !inside {
                                mask[qi * n + ki] = MASK_NEG;
                            }
                            ki += 1;
                        }
                    }
                }
                qi += 1;
            }
        }
    }
    let mask = Array::from_slice(&mask, &[1, 1, n as i32, n as i32]);
    let flat = |x: &Array| {
        x.reshape(&[b, n as i32, nh, hd])
            .unwrap()
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap()
    };
    let (qf, kf, vf) = (flat(q), flat(k), flat(v));
    let scores = add(
        matmul(&qf, kf.transpose_axes(&[0, 1, 3, 2]).unwrap()).unwrap(),
        &mask,
    )
    .unwrap();
    let probs = softmax_axes(&scores, &[-1], None).unwrap();
    let out = matmul(&probs, &vf).unwrap();
    out.transpose_axes(&[0, 2, 1, 3])
        .unwrap()
        .reshape(&[b, t, h, w, nh * hd])
        .unwrap()
}

/// Deterministic pseudo-random f32s — a 64-bit mix, so the fixtures are identical on every host.
fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x =
                (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed.wrapping_mul(0x1000_0193);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let u = (x >> 40) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn probe(shape: &[i32], seed: u64, scale: f32) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_slice(&noise(n as usize, seed, scale), shape)
}

#[test]
fn na3d_matches_a_brute_force_windowed_softmax() {
    // Grids chosen so the tiler has to split at least one axis and so every axis exercises the
    // border shift: a 5-long axis with a 3-window has two clamped positions at each end.
    for (dims, kernel) in [
        ([5i32, 5, 5], [3i32, 3, 3]),
        ([4, 6, 5], [3, 5, 3]),
        ([3, 3, 3], [3, 3, 3]),
        ([7, 4, 4], [5, 3, 3]),
    ] {
        let (nh, hd) = (2i32, 8i32);
        let shape = [1, dims[0], dims[1], dims[2], nh, hd];
        let q = probe(&shape, 1, 0.7);
        let k = probe(&shape, 2, 0.7);
        let v = probe(&shape, 3, 0.7);
        let got = na3d(&q, &k, &v, kernel).unwrap();
        let want = na3d_reference(&q, &k, &v, kernel);
        assert_eq!(got.shape(), want.shape());
        let delta = max_abs(&subtract(&got, &want).unwrap());
        assert!(
            delta < 2e-6,
            "{dims:?} kernel {kernel:?}: max|delta| = {delta:.3e}"
        );
    }
}

#[test]
fn na3d_refuses_a_grid_narrower_than_its_window() {
    // Upstream raises here too. Silently shrinking the window would change the operator.
    let q = probe(&[1, 2, 5, 5, 1, 8], 1, 0.5);
    let err = na3d(&q, &q, &q, [3, 3, 3]).unwrap_err().to_string();
    assert!(err.contains("dim >= its window"), "got: {err}");
}

#[test]
fn na3d_tiling_is_invisible_to_the_result() {
    // Force the tiler to split by shrinking the budget is not possible from here, so instead check
    // that a grid large enough to be split agrees with the brute force reference — the tiling only
    // partitions queries, so any disagreement is a halo bug.
    let (nh, hd) = (1i32, 8i32);
    let shape = [1, 6, 9, 9, nh, hd];
    let q = probe(&shape, 11, 0.6);
    let k = probe(&shape, 12, 0.6);
    let v = probe(&shape, 13, 0.6);
    let delta = max_abs(
        &subtract(
            na3d(&q, &k, &v, [3, 5, 5]).unwrap(),
            na3d_reference(&q, &k, &v, [3, 5, 5]),
        )
        .unwrap(),
    );
    assert!(delta < 2e-6, "max|delta| = {delta:.3e}");
}

// ---------------------------------------------------------------------------------------------
// A miniature decoder on synthetic weights
// ---------------------------------------------------------------------------------------------

/// A tiny but structurally faithful decoder: five stages, four upsample hops with the released
/// strides, a diffusion stage with its own wider window. Small enough to run in a unit test,
/// shaped enough that every code path is exercised.
fn tiny_config() -> NaDiffusionDecoderConfig {
    NaDiffusionDecoderConfig::from_embedded_vae(&serde_json::json!({
        "model_output_type": "x0",
        "decoder": {
            "_class_name": "NADiffusionDecoder",
            "in_channels": 8,
            "out_channels": 3,
            "patch_size": 2,
            "head_dim": 16,
            "t_emb_dim": 32,
            "stage_channels": [32, 32, 16, 16, 16],
            "stage_depths": [1, 1, 1, 1, 2],
            "stage_kernels": [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
            "upsamples": [[[1,2,2],1],[[2,1,1],2],[[2,2,2],1],[[2,2,2],1]],
            "stage5_kernel": [3,3,3],
            "timestep_scale_multiplier": 1000.0,
            "default_num_inference_steps": 1
        }
    }))
    .expect("tiny config")
}

fn tiny_weights(cfg: &NaDiffusionDecoderConfig) -> Weights {
    let mut map: HashMap<String, Array> = HashMap::new();
    let mut seed = 1u64;
    let put =
        |map: &mut HashMap<String, Array>, key: &str, shape: &[i32], scale: f32, seed: &mut u64| {
            *seed += 1;
            map.insert(key.to_string(), probe(shape, *seed, scale));
        };
    let c = &cfg.stage_channels;
    put(
        &mut map,
        "per_channel_statistics.mean",
        &[cfg.in_channels],
        0.05,
        &mut seed,
    );
    put(
        &mut map,
        "per_channel_statistics.std",
        &[cfg.in_channels],
        0.05,
        &mut seed,
    );
    // std is a scale: keep it away from zero.
    map.insert(
        "per_channel_statistics.std".into(),
        add(&map["per_channel_statistics.std"], Array::from_f32(1.0)).unwrap(),
    );
    put(
        &mut map,
        "conv_in.weight",
        &[c[0], cfg.in_channels],
        0.2,
        &mut seed,
    );
    put(&mut map, "conv_in.bias", &[c[0]], 0.05, &mut seed);

    let block = |map: &mut HashMap<String, Array>,
                 prefix: &str,
                 dim: i32,
                 ctx: Option<i32>,
                 seed: &mut u64| {
        let hidden = mlp_hidden(dim);
        put(map, &format!("{prefix}.norm1.weight"), &[dim], 0.1, seed);
        put(map, &format!("{prefix}.norm2.weight"), &[dim], 0.1, seed);
        put(
            map,
            &format!("{prefix}.attn.qkv.weight"),
            &[3 * dim, dim],
            0.2,
            seed,
        );
        put(
            map,
            &format!("{prefix}.attn.qkv.bias"),
            &[3 * dim],
            0.05,
            seed,
        );
        put(
            map,
            &format!("{prefix}.attn.proj.weight"),
            &[dim, dim],
            0.2,
            seed,
        );
        put(map, &format!("{prefix}.attn.proj.bias"), &[dim], 0.05, seed);
        put(
            map,
            &format!("{prefix}.attn.q_norm.weight"),
            &[cfg.head_dim],
            0.1,
            seed,
        );
        put(
            map,
            &format!("{prefix}.attn.k_norm.weight"),
            &[cfg.head_dim],
            0.1,
            seed,
        );
        put(
            map,
            &format!("{prefix}.mlp.w_gate.weight"),
            &[hidden, dim],
            0.2,
            seed,
        );
        put(
            map,
            &format!("{prefix}.mlp.w_up.weight"),
            &[hidden, dim],
            0.2,
            seed,
        );
        put(
            map,
            &format!("{prefix}.mlp.w_down.weight"),
            &[dim, hidden],
            0.2,
            seed,
        );
        if let Some(ctx) = ctx {
            put(
                map,
                &format!("{prefix}.context_proj.weight"),
                &[dim, ctx],
                0.2,
                seed,
            );
            put(
                map,
                &format!("{prefix}.context_proj.bias"),
                &[dim],
                0.05,
                seed,
            );
            put(
                map,
                &format!("{prefix}.scale_shift_table"),
                &[ADALN_CHUNKS, dim],
                0.1,
                seed,
            );
        }
    };

    for stage in 0..cfg.upsamples.len() {
        for i in 0..cfg.stage_depths[stage] {
            block(
                &mut map,
                &format!("det_stages.{stage}.{i}"),
                c[stage],
                None,
                &mut seed,
            );
        }
        let (stride, _) = cfg.upsamples[stage];
        let out = stride[0] * stride[1] * stride[2] * c[stage + 1];
        put(
            &mut map,
            &format!("upsamples.{stage}.proj.weight"),
            &[out, c[stage]],
            0.2,
            &mut seed,
        );
        put(
            &mut map,
            &format!("upsamples.{stage}.proj.bias"),
            &[out],
            0.05,
            &mut seed,
        );
    }
    let c5 = cfg.stage5_width();
    for i in 0..*cfg.stage_depths.last().unwrap() {
        block(
            &mut map,
            &format!("diff_blocks.{i}"),
            c5,
            Some(*c.last().unwrap()),
            &mut seed,
        );
    }
    let patched = cfg.out_channels * cfg.patch_size * cfg.patch_size;
    put(
        &mut map,
        "conv_in_x_t.weight",
        &[c5, patched],
        0.2,
        &mut seed,
    );
    put(&mut map, "conv_in_x_t.bias", &[c5], 0.05, &mut seed);
    put(&mut map, "conv_out.weight", &[patched, c5], 0.2, &mut seed);
    put(&mut map, "conv_out.bias", &[patched], 0.05, &mut seed);
    put(&mut map, "norm_out.weight", &[c5], 0.1, &mut seed);
    put(
        &mut map,
        "t_embedder.mlp.0.weight",
        &[cfg.t_emb_dim, 256],
        0.1,
        &mut seed,
    );
    put(
        &mut map,
        "t_embedder.mlp.0.bias",
        &[cfg.t_emb_dim],
        0.05,
        &mut seed,
    );
    put(
        &mut map,
        "t_embedder.mlp.2.weight",
        &[cfg.t_emb_dim, cfg.t_emb_dim],
        0.1,
        &mut seed,
    );
    put(
        &mut map,
        "t_embedder.mlp.2.bias",
        &[cfg.t_emb_dim],
        0.05,
        &mut seed,
    );
    put(
        &mut map,
        "shared_adaln.proj.weight",
        &[ADALN_CHUNKS * c5, cfg.t_emb_dim],
        0.05,
        &mut seed,
    );
    put(
        &mut map,
        "shared_adaln.proj.bias",
        &[ADALN_CHUNKS * c5],
        0.02,
        &mut seed,
    );
    // Carried by the real checkpoint, consumed by nothing — the loader must tolerate it.
    put(&mut map, "type_emb", &[cfg.in_channels], 0.1, &mut seed);
    Weights::from_map(map)
}

fn tiny_decoder() -> (NaDiffusionDecoder, NaDiffusionDecoderConfig) {
    let cfg = tiny_config();
    let w = tiny_weights(&cfg);
    (
        NaDiffusionDecoder::from_weights(&w, &cfg, None).expect("build the tiny decoder"),
        cfg,
    )
}

/// **A tier directory carries both video VAEs**, so the diffusion decoder's config must be read from
/// the section that actually holds it.
///
/// `crate::tiers` writes the conv VAE's block under `vae` and the DiffVAE's under `diffusion_vae` —
/// one key cannot name two autoencoders. Reading `vae` there hands the diffusion decoder the *conv*
/// VAE's config: a block that parses as JSON, is not corrupt, and is simply the wrong autoencoder.
/// Nothing before this test read `diffusion_vae` at all, so a tier's DiffVAE could not be configured
/// by the shipped loader (sc-18775).
#[test]
fn the_decoder_config_comes_from_the_diffusion_section_when_a_tier_carries_both() {
    let tmp = tempfile::tempdir().unwrap();
    let diffusion: serde_json::Value = serde_json::from_str(RELEASED_VAE).unwrap();
    // A conv `CausalVideoAutoencoder` — no `decoder._class_name: NADiffusionDecoder` anywhere in it.
    let conv = serde_json::json!({
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3, "latent_channels": 128, "patch_size": 4,
    });

    let tier = tmp.path().join("tier");
    std::fs::create_dir_all(&tier).unwrap();
    std::fs::write(
        tier.join("embedded_config.json"),
        serde_json::json!({"vae": conv, "diffusion_vae": diffusion}).to_string(),
    )
    .unwrap();
    let from_tier = NaDiffusionDecoderConfig::from_model_dir(&tier)
        .expect("a tier must resolve through its `diffusion_vae` section");
    assert_eq!(
        from_tier,
        NaDiffusionDecoderConfig::from_embedded_vae(&serde_json::from_str(RELEASED_VAE).unwrap())
            .unwrap(),
        "the tier must yield the DiffVAE's own geometry, not the conv VAE's"
    );

    // The single-VAE shape `crate::convert` emits still resolves through `vae`.
    let single = tmp.path().join("single");
    std::fs::create_dir_all(&single).unwrap();
    std::fs::write(
        single.join("embedded_config.json"),
        serde_json::json!({"vae": serde_json::from_str::<serde_json::Value>(RELEASED_VAE).unwrap()})
            .to_string(),
    )
    .unwrap();
    assert_eq!(
        NaDiffusionDecoderConfig::from_model_dir(&single).unwrap(),
        from_tier,
        "both directory shapes must describe the same decoder"
    );

    // A tier whose diffusion section is absent must say so rather than silently reading the conv
    // block — the failure this ordering exists to prevent, made visible.
    let conv_only = tmp.path().join("conv-only");
    std::fs::create_dir_all(&conv_only).unwrap();
    std::fs::write(
        conv_only.join("embedded_config.json"),
        serde_json::json!({"vae": conv}).to_string(),
    )
    .unwrap();
    let err = NaDiffusionDecoderConfig::from_model_dir(&conv_only)
        .expect_err("a conv-only bundle has no diffusion decoder to configure")
        .to_string();
    assert!(
        err.contains("decoder"),
        "the refusal must name what is missing, got: {err}"
    );
}

/// The shipped tiers' group width, and the one used here. MLX's `quantize` implements exactly three
/// group sizes — 32, 64, 128 — so this is not a free fixture parameter: [`packable_config`] is sized
/// around it rather than the other way round.
const TINY_GROUP: i32 = 64;
/// 8-bit: the packed/dense agreement below is a numeric claim, and q8's error is small enough that a
/// *mis-bound* projection is unmistakable against it rather than lost in q4's noise floor.
const TINY_BITS: i32 = 8;

/// Pack every Linear a `q4`/`q8` tier packs, using the converter's **own** predicate rather than a
/// second copy of it — the two must not be able to disagree about which keys become triples.
/// A miniature decoder whose every quantizable input axis is a multiple of [`TINY_GROUP`].
///
/// [`tiny_config`]'s 8–32-wide stages cannot be packed at all — MLX supports no group below 32 — so
/// the packed-tier tests need their own geometry. Everything else about it is [`tiny_config`]: the
/// same stage count, depths, and upsample ladder.
fn packable_config() -> NaDiffusionDecoderConfig {
    NaDiffusionDecoderConfig::from_embedded_vae(&serde_json::json!({
        "model_output_type": "x0",
        "decoder": {
            "_class_name": "NADiffusionDecoder",
            "in_channels": 64,
            "out_channels": 3,
            "patch_size": 2,
            "head_dim": 16,
            "t_emb_dim": 64,
            "stage_channels": [64, 64, 64, 64, 64],
            "stage_depths": [1, 1, 1, 1, 2],
            "stage_kernels": [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
            "upsamples": [[[1,2,2],1],[[2,1,1],2],[[2,2,2],1],[[2,2,2],1]],
            "stage5_kernel": [3,3,3],
            "timestep_scale_multiplier": 1000.0,
            "default_num_inference_steps": 1
        }
    }))
    .expect("packable config")
}

fn tiny_weights_packed(cfg: &NaDiffusionDecoderConfig) -> Weights {
    let dense = tiny_weights(cfg);
    let mut map: HashMap<String, Array> = HashMap::new();
    let mut packed = 0usize;
    for key in dense.keys() {
        let value = dense.require(key).unwrap();
        if !crate::tiers::is_diff_vae_decoder_quantizable(key) {
            map.insert(key.to_string(), value.clone());
            continue;
        }
        let shape = value.shape();
        assert_eq!(
            shape.len(),
            2,
            "{key} is selected for quantization but is not rank-2"
        );
        assert_eq!(
            shape[1] % TINY_GROUP,
            0,
            "{key}'s input axis {} does not divide the fixture group {TINY_GROUP}",
            shape[1]
        );
        let base = key.strip_suffix(".weight").unwrap().to_string();
        let (q, scales, biases) =
            mlx_rs::ops::quantize(value, TINY_GROUP, TINY_BITS).expect("quantize");
        map.insert(format!("{base}.weight"), q);
        map.insert(format!("{base}.scales"), scales);
        map.insert(format!("{base}.biases"), biases);
        packed += 1;
    }
    assert!(
        packed > 10,
        "the fixture must pack a real population, got {packed}"
    );
    Weights::from_map(map)
}

/// **A packed decoder loads and decodes**, and lands where 8-bit affine quantization puts it — not
/// at zero (which would mean the packed weights were never bound) and not adrift (which would mean
/// they were bound wrong).
///
/// This is the load path sc-18775's q4/q8 tiers depend on: before it existed the whole component was
/// emitted dense under a `no-mlx-port` exemption, and the tier's 834 MB stayed bf16 inside a "q4".
#[test]
fn a_packed_decoder_binds_its_triples_and_tracks_the_dense_one() {
    let cfg = packable_config();
    let dense = NaDiffusionDecoder::from_weights(&tiny_weights(&cfg), &cfg, None).unwrap();
    let quant = Some(DiffVaeQuant {
        bits: TINY_BITS,
        group: TINY_GROUP,
    });
    let packed = NaDiffusionDecoder::from_weights(&tiny_weights_packed(&cfg), &cfg, quant).unwrap();

    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 42, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 4);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        43,
        1.0,
    );
    let a = dense.decode(&latent, &noise).unwrap();
    let b = packed.decode(&latent, &noise).unwrap();
    assert_eq!(a.shape(), b.shape());

    let delta = mean_abs(&subtract(&a, &b).unwrap());
    let scale = mean_abs(&a);
    assert!(
        delta > 0.0,
        "identical output means the packed triples were never bound — the loader read the dense \
         `.weight` and ignored `.scales`"
    );
    assert!(
        delta < 0.25 * scale,
        "8-bit quantization moved the decode by {delta} against a mean magnitude of {scale} — that \
         is a mis-bound projection, not quantization error"
    );
}

/// Declaring the file dense while it carries `.scales` is refused, by name.
///
/// The alternative — reading the `U32` payload as a float weight — produces a decoder that builds,
/// runs, and returns noise. Nothing downstream could tell that apart from a bad checkpoint.
#[test]
fn a_packed_file_loaded_as_dense_is_refused_rather_than_misread() {
    let cfg = packable_config();
    let err = match NaDiffusionDecoder::from_weights(&tiny_weights_packed(&cfg), &cfg, None) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a packed checkpoint must not load as dense"),
    };
    assert!(
        err.contains(".scales is present") && err.contains("DiffVaeQuant"),
        "the refusal must name the packed sibling and what to pass, got: {err}"
    );
}

#[test]
fn expected_weight_keys_covers_exactly_what_the_loader_reads() {
    let cfg = tiny_config();
    let w = tiny_weights(&cfg);
    let present: std::collections::BTreeSet<String> = w.keys().map(str::to_string).collect();
    let expected: std::collections::BTreeSet<String> =
        expected_weight_keys(&cfg).into_iter().collect();
    let unused: Vec<&String> = present.difference(&expected).collect();
    assert_eq!(
        unused,
        vec![&"type_emb".to_string()],
        "the only tensor the decoder ignores is the checkpoint's dead type_emb"
    );
    assert!(
        expected.difference(&present).next().is_none(),
        "expected_weight_keys names a key the fixture does not carry"
    );
    assert!(looks_like_diffusion_decoder(&w));
    assert_eq!(UNUSED_DECODER_KEYS, ["type_emb"]);
}

#[test]
fn a_width_disagreement_between_config_and_weights_is_a_load_error() {
    let cfg = tiny_config();
    let mut wrong = cfg.clone();
    wrong.stage_channels[0] = 64;
    let w = tiny_weights(&cfg);
    let err = match NaDiffusionDecoder::from_weights(&w, &wrong, None) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a config that disagrees with the weights must not load"),
    };
    assert!(
        err.contains("conv_in output width"),
        "the refusal must name the mismatched tensor, got: {err}"
    );
}

#[test]
fn decode_returns_the_pixel_geometry_the_latent_implies() {
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 42, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 4);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        43,
        1.0,
    );
    let out = decoder.decode(&latent, &noise).unwrap();
    let scale = cfg.pixel_scale();
    assert_eq!(
        scale,
        [8, 16, 16],
        "this ladder's own hops, not LTX-2.3's 8/32/32"
    );
    assert_eq!(out.shape(), &[1, 3, 17, 4 * scale[1], 4 * scale[2]]);
    assert!(max_abs(&out).is_finite());
}

#[test]
fn a_mis_shaped_noise_canvas_is_refused_with_the_shape_it_wanted() {
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 42, 0.8);
    let wrong = probe(&[1, 3, 16, 48, 48], 43, 1.0);
    let err = decoder.decode(&latent, &wrong).unwrap_err().to_string();
    assert!(err.contains("noise_shape"), "got: {err}");
}

#[test]
fn a_below_floor_latent_is_padded_up_and_cropped_back() {
    // The latent floor is 3x3x3 for this ladder; a 1x2x2 latent must still decode, at the geometry
    // its own extent implies rather than the padded one.
    let (decoder, cfg) = tiny_decoder();
    assert_eq!(cfg.min_latent_shape(), [3, 3, 3]);
    let latent = probe(&[1, cfg.in_channels, 1, 2, 2], 7, 0.8);
    let shape5 = cfg.noise_shape(1, 2, 2);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        8,
        1.0,
    );
    let out = decoder.decode(&latent, &noise).unwrap();
    let scale = cfg.pixel_scale();
    assert_eq!(out.shape(), &[1, 3, 1, 2 * scale[1], 2 * scale[2]]);
    assert!(max_abs(&out).is_finite());
}

#[test]
fn seeded_decode_is_reproducible() {
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 3, 3], 5, 0.8);
    let a = decoder.decode_seeded(&latent, 18766, None).unwrap();
    let b = decoder.decode_seeded(&latent, 18766, None).unwrap();
    assert_eq!(max_abs(&subtract(&a, &b).unwrap()), 0.0);
    let c = decoder.decode_seeded(&latent, 18767, None).unwrap();
    assert!(
        max_abs(&subtract(&a, &c).unwrap()) > 1e-6,
        "a new seed must move the noise"
    );
}

#[test]
fn an_under_haloed_tiling_is_refused_rather_than_smeared() {
    let (decoder, cfg) = tiny_decoder();
    let halo = cfg.tile_halo();
    let min_tile = cfg.min_tile_shape();
    let legal = DiffVaeTiling {
        tile: [halo[0] * 3, halo[1] * 3, halo[2] * 3],
        overlap: halo,
    };
    assert!(legal.validated(&cfg).is_ok());

    for axis in 0..3 {
        let mut starved = legal;
        starved.overlap[axis] = halo[axis] - 1;
        let err = starved.validated(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("halo") && err.contains(&format!("axis {axis}")),
            "axis {axis} refusal must name the axis and the halo, got: {err}"
        );
        // A starved tile is likewise refused — stage 4 or 5 would attend outside its own extent.
        let mut small = legal;
        small.tile[axis] = min_tile[axis] - 1;
        assert!(small.validated(&cfg).is_err(), "axis {axis} tile floor");
    }
    // The temporal axis is not exempt. This is the axis the conv decoder's tiler starved
    // (`tests/vae_decode_tiling_parity.rs`): a temporally under-haloed decode smears rather than
    // erroring, so the refusal has to be the thing that catches it.
    let mut temporal = legal;
    temporal.overlap[0] = 0;
    assert!(temporal.validated(&cfg).is_err());

    // An overlap wider than its own tile is refused only where the axis actually splits — a clip
    // too short to tile in time must still be tileable in space, which is the common case once the
    // stage-5 halo is 20 cells wide.
    let latent = probe(&[1, cfg.in_channels, 3, 4, 8], 91, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 8);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        92,
        1.0,
    );
    let stage4 = cfg.stage4_shape(3, 4, 8);
    let spatial_only = DiffVaeTiling {
        // Temporal tile covers the whole grid with an overlap wider than it — never split.
        tile: [stage4[0], stage4[1], 20],
        overlap: [stage4[0] + 5, halo[1], halo[2]],
    };
    decoder
        .decode_tiled(&latent, &noise, &spatial_only)
        .expect("an unsplit axis' overlap is irrelevant");
    let split_too_far = DiffVaeTiling {
        tile: [stage4[0], stage4[1], 20],
        overlap: [halo[0], halo[1], 20],
    };
    let err = match decoder.decode_tiled(&latent, &noise, &split_too_far) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a split axis' overlap must be smaller than its tile"),
    };
    assert!(err.contains("axis 2"), "got: {err}");
}

#[test]
fn a_single_covering_tile_is_bit_identical_to_an_untiled_decode() {
    // The tiled driver must be the untiled one plus a blend — not a second implementation. With a
    // tile that covers the whole stage-4 grid, the blend is the identity and the two paths must
    // agree exactly.
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 9, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 4);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        10,
        1.0,
    );
    let stage4 = cfg.stage4_shape(3, 4, 4);
    let tiling = DiffVaeTiling {
        tile: stage4,
        overlap: cfg.tile_halo(),
    };
    let untiled = decoder.decode(&latent, &noise).unwrap();
    let tiled = decoder.decode_tiled(&latent, &noise, &tiling).unwrap();
    assert_eq!(tiled.shape(), untiled.shape());
    let delta = max_abs(&subtract(&tiled, &untiled).unwrap());
    assert!(delta < 1e-5, "single-tile decode differs by {delta:.3e}");
}

#[test]
fn a_tiled_interior_matches_the_untiled_decode() {
    // Split the width axis only, then compare the region of the first tile whose whole stage-4/5
    // receptive field lies inside that tile. There the tiled decode sees exactly what the untiled
    // one saw, so agreement is a property, not a tolerance — and it is precisely what a starved
    // halo destroys.
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 4, 8], 21, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 8);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        22,
        1.0,
    );
    let stage4 = cfg.stage4_shape(3, 4, 8);
    assert_eq!(stage4, [9, 16, 32]);
    let halo = cfg.tile_halo();
    let min_tile = cfg.min_tile_shape();
    let tile_w = 20;
    let tiling = DiffVaeTiling {
        tile: [stage4[0], stage4[1], tile_w],
        overlap: halo,
    };
    // Two tiles along W, one along T and H.
    let split = split_by_size(stage4[2], tile_w, halo[2], min_tile[2]);
    assert_eq!(split.len(), 2, "the test needs a real seam");

    let untiled = decoder.decode(&latent, &noise).unwrap();
    let tiled = decoder.decode_tiled(&latent, &noise, &tiling).unwrap();
    assert_eq!(tiled.shape(), untiled.shape());

    // The combined stage-4 + stage-5 receptive field is `halo4 + halo5` stage-4 cells wide on each
    // side; `tile_halo` reports the larger of the two, so back off by twice it plus one cell to
    // stay strictly inside. Pixels below that column see the same neighbourhood in both decodes.
    let last = cfg.upsamples.len() - 1;
    let per_cell = cfg.upsamples[last].0[2] * cfg.patch_size;
    let interior = (tile_w - 2 * halo[2] - 1) * per_cell;
    assert!(
        interior > 4 * per_cell,
        "the test needs a real interior, got {interior}"
    );
    let cut = |x: &Array| slice_axis(x, 4, 0, interior).unwrap();
    let delta = max_abs(&subtract(cut(&tiled), cut(&untiled)).unwrap());
    assert!(
        delta < 1e-4,
        "tiled interior differs from untiled by {delta:.3e} — the halo is not covering the \
         stage-4/5 receptive field"
    );
    // And the blend really is a partition of unity: nothing may be scaled away.
    let ratio = max_abs(&tiled) / max_abs(&untiled);
    assert!(
        (0.5..2.0).contains(&ratio),
        "tiled output magnitude drifted by {ratio:.3}, which is what an unnormalised blend does"
    );
}

#[test]
fn a_heavy_overlap_tiling_is_still_normalised_to_the_right_level() {
    // With `2 * overlap > tile` the trapezoid ramps meet inside a tile and multiply, so the masks
    // stop being a partition of unity and every seam region is scaled by whatever they happen to
    // sum to. The driver normalises by the accumulated weight profile precisely so that regime
    // stays correct — a dimmed seam is the silent-corruption failure mode, not an error.
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 3, 4, 8], 51, 0.8);
    let shape5 = cfg.noise_shape(3, 4, 8);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        52,
        1.0,
    );
    let stage4 = cfg.stage4_shape(3, 4, 8);
    let tiling = DiffVaeTiling {
        tile: [stage4[0], stage4[1], 3],
        overlap: [1, 1, 2],
    };
    // The width split really is non-complementary: interior tiles are 3 long with 2-long ramps at
    // both ends, so the ramps overlap and multiply.
    let split = split_by_size(stage4[2], 3, 2, cfg.min_tile_shape()[2]);
    assert!(split.len() > 2);
    assert!(split[1].left_ramp + split[1].right_ramp > split[1].end - split[1].start);

    let untiled = decoder.decode(&latent, &noise).unwrap();
    let tiled = decoder.decode_tiled(&latent, &noise, &tiling).unwrap();
    assert_eq!(tiled.shape(), untiled.shape());
    // The multiplied ramps sum to ~1.11 here, so an unnormalised blend brightens the whole picture
    // by ~11%. Normalised, the level must track the untiled decode's.
    let energy = |x: &Array| mean_abs(x);
    let ratio = energy(&tiled) / energy(&untiled);
    assert!(
        (ratio - 1.0).abs() < 0.03,
        "heavy-overlap blend rescaled the picture by {ratio:.4} — the weight normalisation is not \
         doing its job"
    );
}

#[test]
fn the_blend_weight_profile_sums_to_one_everywhere() {
    // The tiled driver divides by the outer product of three 1-D weight profiles. If the profiles
    // did not cover the canvas exactly, the seams would be dimmed rather than blended — visible as
    // a dark band, and invisible to any test that only looks at shapes.
    let cfg = tiny_config();
    let stage4 = cfg.stage4_shape(3, 4, 8);
    let halo = cfg.tile_halo();
    let min_tile = cfg.min_tile_shape();
    let tile_w = 20;
    let last = cfg.upsamples.len() - 1;
    let canvas_w = stage4[2] * cfg.upsamples[last].0[2] * cfg.patch_size;
    let mut profile = vec![0.0f32; canvas_w as usize];
    for iv in split_by_size(stage4[2], tile_w, halo[2], min_tile[2]) {
        let px = propagate(
            propagate(iv, cfg.upsamples[last].0[2], false),
            cfg.patch_size,
            false,
        );
        for (i, value) in trapezoid(px.end - px.start, px.left_ramp, px.right_ramp)
            .iter()
            .enumerate()
        {
            profile[px.start as usize + i] += value;
        }
    }
    for (i, value) in profile.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < 1e-5,
            "blend weight at {i} is {value}, not 1"
        );
    }
}

#[test]
fn the_euler_loop_runs_for_a_multi_step_velocity_checkpoint() {
    // The released checkpoint is single-step x0, but the sampler is not hardcoded to it: a `v`
    // checkpoint with more than one step must run its whole schedule.
    let mut cfg = tiny_config();
    cfg.model_output_type = ModelOutputType::Velocity;
    cfg.default_num_inference_steps = 3;
    let w = tiny_weights(&cfg);
    let decoder = NaDiffusionDecoder::from_weights(&w, &cfg, None).unwrap();
    let schedule = decoder.timesteps();
    assert_eq!(schedule.len(), 3);
    for (got, want) in schedule.iter().zip([1.0f32, 2.0 / 3.0, 1.0 / 3.0]) {
        assert!((got - want).abs() < 1e-6, "schedule {schedule:?}");
    }
    let latent = probe(&[1, cfg.in_channels, 3, 3, 3], 31, 0.8);
    let shape5 = cfg.noise_shape(3, 3, 3);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        32,
        1.0,
    );
    let out = decoder.decode(&latent, &noise).unwrap();
    let scale = cfg.pixel_scale();
    assert_eq!(out.shape(), &[1, 3, 17, 3 * scale[1], 3 * scale[2]]);
    assert!(max_abs(&out).is_finite());

    // The x0 schedule of the same length must differ — the two parameterisations are not aliases.
    let mut x0_cfg = cfg.clone();
    x0_cfg.model_output_type = ModelOutputType::X0;
    let x0 = NaDiffusionDecoder::from_weights(&w, &x0_cfg, None).unwrap();
    let other = x0.decode(&latent, &noise).unwrap();
    assert!(max_abs(&subtract(&out, &other).unwrap()) > 1e-6);
}

#[test]
fn the_single_step_schedule_is_one_timestep_at_one() {
    let (decoder, _) = tiny_decoder();
    assert_eq!(decoder.timesteps(), vec![1.0]);
}

#[test]
fn the_timestep_embedding_is_a_function_of_the_scaled_timestep() {
    // `timestep_scale_multiplier` is 1000 in the released config; folding it in at the wrong place
    // is a silent, plausible-looking error, so pin that the embedder sees the scaled value.
    let (decoder, cfg) = tiny_decoder();
    assert_eq!(cfg.timestep_scale_multiplier, 1000.0);
    let a = decoder.timestep_embedding(1.0).unwrap();
    let b = decoder.timestep_embedding(0.5).unwrap();
    assert_eq!(a.shape(), &[1, cfg.t_emb_dim]);
    assert!(max_abs(&subtract(&a, &b).unwrap()) > 1e-3);
    // Chunking of the shared AdaLN output must be contiguous and in order.
    let modulation = decoder.modulation(1.0).unwrap();
    assert_eq!(modulation.len(), ADALN_CHUNKS as usize);
    for chunk in &modulation {
        assert_eq!(chunk.shape(), &[1, 1, 1, 1, cfg.stage5_width()]);
    }
    let joined = concatenate_axis(&modulation.iter().collect::<Vec<_>>(), 4).unwrap();
    let direct = decoder
        .shared_adaln
        .forward(&silu(&decoder.timestep_embedding(1.0).unwrap()).unwrap())
        .unwrap();
    let delta = max_abs(
        &subtract(
            joined.reshape(&[1, -1]).unwrap(),
            direct.reshape(&[1, -1]).unwrap(),
        )
        .unwrap(),
    );
    assert_eq!(delta, 0.0);
}

#[test]
fn swiglu_token_tiling_changes_nothing() {
    // The hidden buffer is tiled to bound peak memory; each token is independent, so the tiled and
    // untiled results must be identical, not merely close.
    let dim = 16;
    let hidden = 64;
    let mlp = SwiGlu {
        w_gate: ProjWeight::Dense(probe(&[hidden, dim], 61, 0.3)),
        w_up: ProjWeight::Dense(probe(&[hidden, dim], 62, 0.3)),
        w_down: ProjWeight::Dense(probe(&[dim, hidden], 63, 0.3)),
    };
    let tokens = SWIGLU_TILE_TOKENS + 37;
    let x = probe(&[1, tokens, dim], 64, 0.5);
    let tiled = mlp.forward(&x).unwrap();
    let whole = mlp.tile(&x.reshape(&[tokens, dim]).unwrap()).unwrap();
    assert_eq!(tiled.shape(), &[1, tokens, dim]);
    let delta = max_abs(&subtract(tiled.reshape(&[tokens, dim]).unwrap(), &whole).unwrap());
    assert_eq!(delta, 0.0);
}

#[test]
fn the_per_channel_statistics_are_undone_before_the_first_projection() {
    // The encoder normalises its latent; a decoder that forgets to undo that produces a washed-out
    // picture rather than an error. Check the un-normalisation directly.
    let (decoder, cfg) = tiny_decoder();
    let latent = probe(&[1, cfg.in_channels, 2, 2, 2], 71, 0.6);
    let got = decoder.un_normalize(&latent).unwrap();
    assert_eq!(got.shape(), &[1, 2, 2, 2, cfg.in_channels]);
    let want = add(
        multiply(&latent, &decoder.stat_std).unwrap(),
        &decoder.stat_mean,
    )
    .unwrap()
    .transpose_axes(&[0, 2, 3, 4, 1])
    .unwrap();
    assert_eq!(max_abs(&subtract(&got, &want).unwrap()), 0.0);
    // And it is not the identity, or the check above would be vacuous.
    let identity = latent.transpose_axes(&[0, 2, 3, 4, 1]).unwrap();
    assert!(max_abs(&subtract(&got, &identity).unwrap()) > 1e-3);
}

#[test]
fn the_pixel_shuffle_upsample_drops_the_duplicate_frame_only_at_the_origin() {
    let w = {
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("up.proj.weight".into(), probe(&[8 * 4, 4], 81, 0.3));
        map.insert("up.proj.bias".into(), probe(&[8 * 4], 82, 0.1));
        Weights::from_map(map)
    };
    let up = PixelShuffleUpsample::load(&w, "up", [2, 2, 2], None).unwrap();
    assert_eq!(up.out_channels(), 4);
    let x = probe(&[1, 3, 2, 2, 4], 83, 0.5);
    assert_eq!(up.forward(&x, true).unwrap().shape(), &[1, 5, 4, 4, 4]);
    assert_eq!(up.forward(&x, false).unwrap().shape(), &[1, 6, 4, 4, 4]);
    // The kept frames are the same ones; only the leading duplicate differs.
    let dropped = up.forward(&x, true).unwrap();
    let kept = slice_axis(&up.forward(&x, false).unwrap(), 1, 1, 6).unwrap();
    assert_eq!(max_abs(&subtract(&dropped, &kept).unwrap()), 0.0);

    // A spatial-only hop never drops anything.
    let w2 = {
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("up.proj.weight".into(), probe(&[4 * 4, 4], 84, 0.3));
        map.insert("up.proj.bias".into(), probe(&[4 * 4], 85, 0.1));
        Weights::from_map(map)
    };
    let spatial = PixelShuffleUpsample::load(&w2, "up", [1, 2, 2], None).unwrap();
    assert_eq!(spatial.forward(&x, true).unwrap().shape(), &[1, 3, 4, 4, 4]);
}

#[test]
fn keys_outside_the_window_contribute_nothing() {
    // The additive mask is assembled by summing three per-axis masks, so a wrong sign or a value
    // that is merely "very negative" rather than saturating would leak a little of every key into
    // every query — a global blur that no shape check sees. Load V with a large value everywhere
    // except the corner query's own window, and the corner output must stay at zero.
    let dims = [4i32, 4, 4];
    let kernel = [3i32, 3, 3];
    let (nh, hd) = (1i32, 8i32);
    let shape = [1, dims[0], dims[1], dims[2], nh, hd];
    let q = probe(&shape, 91, 0.9);
    let k = probe(&shape, 92, 0.9);

    let mut v = vec![0.0f32; (dims[0] * dims[1] * dims[2] * nh * hd) as usize];
    let mut i = 0usize;
    for t in 0..dims[0] {
        for h in 0..dims[1] {
            for w in 0..dims[2] {
                let outside = t >= kernel[0] || h >= kernel[1] || w >= kernel[2];
                for _ in 0..(nh * hd) {
                    v[i] = if outside { 1_000.0 } else { 0.0 };
                    i += 1;
                }
            }
        }
    }
    let v = Array::from_slice(&v, &shape);
    let out = na3d(&q, &k, &v, kernel).unwrap();
    let corner = slice_axis(
        &slice_axis(&slice_axis(&out, 1, 0, 1).unwrap(), 2, 0, 1).unwrap(),
        3,
        0,
        1,
    )
    .unwrap();
    let leak = max_abs(&corner);
    assert!(
        leak < 1e-2,
        "keys outside the corner query's 3x3x3 window leaked {leak:.3e} of a 1000-valued signal"
    );
    // Sanity: a query whose window does reach the loaded region must see it, or the test above
    // would pass on an all-zero output.
    let far = slice_axis(
        &slice_axis(&slice_axis(&out, 1, 3, 4).unwrap(), 2, 3, 4).unwrap(),
        3,
        3,
        4,
    )
    .unwrap();
    assert!(
        max_abs(&far) > 100.0,
        "the far corner must see the loaded keys"
    );
    let _ = sum_axes(&out, &[-1], false).unwrap();
}
