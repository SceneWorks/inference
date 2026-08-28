//! Weights-free unit tests for the candle `NADiffusionDecoder` (sc-18767).
//!
//! No GPU, no gated weights, ordinary CI on every platform — these cover the parts a real-weight
//! golden cannot localise: the config refusals, the geometry algebra pinned against the released
//! ladder, [`na3d`] against a brute-force masked softmax over the whole grid, and the tiling
//! properties on a miniature decoder built from deterministic synthetic weights.
//!
//! The real-weight absolute-error goldens — the same fixture the MLX port asserts against — live in
//! `tests/ltx_2_5_diffvae_parity.rs`.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;

use super::*;

/// The released LTX-2.5 `vae` config block, verbatim from
/// `docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json` (`ltx_2_5_video_vae_diffusion`).
/// Hand-typed here so a config-parsing regression is caught without reading a fixture file.
fn released_vae_json() -> Value {
    serde_json::json!({
        "_class_name": "CausalDiffusionVAE",
        "dims": 3,
        "model_output_type": "x0",
        "decoder": {
            "_class_name": "NADiffusionDecoder",
            "in_channels": 128,
            "out_channels": 3,
            "patch_size": 4,
            "head_dim": 64,
            "stage_channels": [2048, 1024, 512, 512, 256],
            "stage_depths": [4, 6, 4, 2, 8],
            "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
            "upsamples": [
                [[1, 2, 2], 2],
                [[2, 1, 1], 2],
                [[2, 2, 2], 1],
                [[2, 2, 2], 1]
            ],
            "stage5_kernel": [11, 11, 11],
            "t_emb_dim": 384,
            "default_num_inference_steps": 1,
            "timestep_scale_multiplier": 1000.0,
            "spatial_padding_mode": "zeros"
        }
    })
}

fn released() -> NaDiffusionDecoderConfig {
    NaDiffusionDecoderConfig::from_embedded_vae(&released_vae_json()).expect("released config")
}

// ---------------------------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------------------------

#[test]
fn the_released_config_parses_into_the_shipped_ladder() {
    let cfg = released();
    assert_eq!(cfg.stage_channels, vec![2048, 1024, 512, 512, 256]);
    assert_eq!(cfg.stage_depths, vec![4, 6, 4, 2, 8]);
    assert_eq!(cfg.stage5_kernel, [11, 11, 11]);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.model_output_type, ModelOutputType::X0);
    assert_eq!(cfg.default_num_inference_steps, 1);
    assert_eq!(cfg.timestep_scale_multiplier, 1000.0);
    assert_eq!(cfg.stage5_width(), 256);
}

#[test]
fn the_ladders_own_hops_multiply_out_to_the_declared_scale_factors() {
    // The decoder derives its pixel scale from the upsample strides rather than reading
    // VIDEO_SCALE_FACTORS, so this is the assertion that the constant still describes the ladder.
    assert_eq!(released().pixel_scale(), VIDEO_SCALE_FACTORS);
}

#[test]
fn a_decoder_that_is_not_an_na_diffusion_decoder_is_refused() {
    let mut v = released_vae_json();
    v["decoder"]["_class_name"] = Value::String("ConvVideoDecoder".into());
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("a conv decoder must not parse as an NADiffusionDecoder")
        .to_string();
    assert!(err.contains("NADiffusionDecoder"), "{err}");
}

#[test]
fn an_architecture_field_is_never_defaulted() {
    for key in [
        "stage_channels",
        "stage_depths",
        "stage_kernels",
        "upsamples",
        "stage5_kernel",
        // Not shape, but sampler: defaulting this to 1.0 against the released 1000.0 embeds every
        // timestep at the wrong frequency and decodes silently wrongly.
        "timestep_scale_multiplier",
    ] {
        let mut v = released_vae_json();
        v["decoder"]
            .as_object_mut()
            .expect("decoder object")
            .remove(key);
        let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
            .expect_err("a missing architecture field must be an error, never a default")
            .to_string();
        assert!(err.contains(key), "{key}: {err}");
    }

    // `model_output_type` sits on the `vae` block rather than on `decoder`. Defaulting it to `v`
    // against the released `x0` checkpoint changes what the stage-5 blocks are taken to predict.
    let mut v = released_vae_json();
    v.as_object_mut()
        .expect("vae object")
        .remove("model_output_type");
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("a missing model_output_type must be an error, never a default")
        .to_string();
    assert!(err.contains("model_output_type"), "{err}");
}

#[test]
fn a_stage_width_that_is_not_a_multiple_of_head_dim_is_refused() {
    let mut v = released_vae_json();
    v["decoder"]["stage_channels"] = serde_json::json!([2048, 1024, 512, 512, 260]);
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("260 is not a multiple of head_dim 64")
        .to_string();
    assert!(err.contains("head_dim"), "{err}");
}

#[test]
fn a_non_zero_spatial_padding_mode_is_refused() {
    let mut v = released_vae_json();
    v["decoder"]["spatial_padding_mode"] = Value::String("replicate".into());
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("only `zeros` is meaningful")
        .to_string();
    assert!(err.contains("spatial_padding_mode"), "{err}");
}

#[test]
fn an_unknown_model_output_type_is_refused() {
    let mut v = released_vae_json();
    v["model_output_type"] = Value::String("epsilon".into());
    let err = NaDiffusionDecoderConfig::from_embedded_vae(&v)
        .expect_err("only x0 | v are implemented")
        .to_string();
    assert!(err.contains("model_output_type"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// Geometry algebra, pinned against the released ladder
// ---------------------------------------------------------------------------------------------

#[test]
fn the_released_ladders_tiling_geometry_is_pinned() {
    let cfg = released();
    assert_eq!(cfg.ghost_latent_frames(), 2, "(3 / 2) * 2");
    // stage 4 kernel 3x5x5, depth 2 -> halo4 = [2, 4, 4]; stage 5 kernel 11 with depth 8 over
    // stride 2 -> halo5 = ceil(8 * 5 / 2) = 20 on every axis.
    assert_eq!(cfg.tile_halo(), [20, 20, 20]);
    // stage-4 window 3x5x5 vs ceil(11 / 2) = 6 -> the stage-5 window dominates on every axis.
    assert_eq!(cfg.min_tile_shape(), [6, 6, 6]);
    assert_eq!(cfg.min_latent_shape(), [3, 7, 7]);
    // Three hops reach the stage-4 input: (1,2,2) -> 7x14x14, (2,1,1) -> (2*7-1)=13, then
    // (2,2,2) -> (2*13-1)=25 frames and 28x28 spatially.
    assert_eq!(cfg.stage4_shape(7, 7, 7), [25, 28, 28]);
    // The final (2,2,2) hop plus the patch-4 unpatchify: 2*25-1 = 49 frames, 28*2*4 = 224 px.
    assert_eq!(cfg.noise_shape(7, 7, 7), [49, 224, 224]);
}

#[test]
fn the_noise_shape_applies_the_latent_floor_before_reporting() {
    let cfg = released();
    // A 1x1x1 latent is grown to the [3, 7, 7] floor first, so the reported canvas is the floor's.
    assert_eq!(cfg.noise_shape(1, 1, 1), cfg.noise_shape(3, 7, 7));
}

// ---------------------------------------------------------------------------------------------
// Neighborhood attention
// ---------------------------------------------------------------------------------------------

#[test]
fn window_starts_slide_inward_at_the_border_instead_of_clipping() {
    // NATTEN keeps the window full-size and slides it in; it does not clip and renormalise.
    assert_eq!(window_starts(8, 3), vec![0, 0, 1, 2, 3, 4, 5, 5]);
    // A window wider than the axis collapses to the whole axis, once.
    assert_eq!(window_starts(3, 7), vec![0, 0, 0]);
}

/// Brute-force `na3d` over the whole grid: every query attends every key, masked by the window.
/// Deliberately not tiled and not separable — it is the independent statement of the operator the
/// tiled implementation has to agree with.
fn na3d_reference(q: &Tensor, k: &Tensor, v: &Tensor, kernel: [usize; 3]) -> Tensor {
    let d = q.dims().to_vec();
    let (b, t, h, w, nh, hd) = (d[0], d[1], d[2], d[3], d[4], d[5]);
    assert_eq!(b, 1, "the reference is written for one batch item");
    let starts: Vec<Vec<usize>> = (0..3)
        .map(|a| window_starts([t, h, w][a], kernel[a].min([t, h, w][a])))
        .collect();
    let flat = |x: &Tensor| -> Vec<f32> {
        x.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let (qv, kv, vv) = (flat(q), flat(k), flat(v));
    let index =
        |ti: usize, hi: usize, wi: usize, head: usize| (((ti * h + hi) * w + wi) * nh + head) * hd;
    let mut out = vec![0f32; t * h * w * nh * hd];
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                for head in 0..nh {
                    let qo = index(ti, hi, wi, head);
                    let mut keys: Vec<usize> = Vec::new();
                    for kt in starts[0][ti]..starts[0][ti] + kernel[0].min(t) {
                        for kh in starts[1][hi]..starts[1][hi] + kernel[1].min(h) {
                            for kw in starts[2][wi]..starts[2][wi] + kernel[2].min(w) {
                                keys.push(index(kt, kh, kw, head));
                            }
                        }
                    }
                    let scores: Vec<f32> = keys
                        .iter()
                        .map(|&ko| (0..hd).map(|i| qv[qo + i] * kv[ko + i]).sum::<f32>())
                        .collect();
                    let peak = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
                    let total: f32 = exp.iter().sum();
                    for (weight, &ko) in exp.iter().zip(&keys) {
                        for i in 0..hd {
                            out[qo + i] += weight / total * vv[ko + i];
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, (b, t, h, w, nh * hd), &Device::Cpu).unwrap()
}

/// Deterministic, smooth values in `[-1, 1]` — the shape of signal the decoder actually carries.
fn probe(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64;
            let s = seed as f64;
            ((x * 0.013_1 + s * 1.7).sin() * (x * 0.007_3 - s * 0.31).cos() * 0.9
                + 0.1 * (x * 0.000_37 + s).sin()) as f32
        })
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).expect("probe")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "shape mismatch before comparison");
    let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn na3d_matches_a_brute_force_masked_softmax_over_the_whole_grid() {
    let (t, h, w, nh, hd) = (5usize, 6, 7, 2, 8);
    let shape = [1usize, t, h, w, nh, hd];
    let q = probe(&shape, 1);
    let k = probe(&shape, 2);
    let v = probe(&shape, 3);
    for kernel in [[3usize, 3, 3], [1, 5, 5], [5, 3, 7]] {
        let got = na3d(&q, &k, &v, kernel).expect("na3d");
        let want = na3d_reference(&q, &k, &v, kernel);
        let err = max_abs_diff(&got, &want);
        assert!(err < 2e-6, "kernel {kernel:?}: max|delta| = {err:.3e}");
    }
}

#[test]
fn na3d_refuses_a_window_wider_than_its_grid() {
    let shape = [1usize, 2, 4, 4, 1, 8];
    let q = probe(&shape, 1);
    let err = na3d(&q, &q, &q, [3, 3, 3])
        .expect_err("a 3-frame window over a 2-frame grid must be refused")
        .to_string();
    assert!(err.contains("window"), "{err}");
}

#[test]
fn na3d_is_tiling_invariant() {
    // `pick_tiles` picks the tiling from a budget; forcing a small budget must not change the
    // answer, only the schedule. Exercised by comparing against the brute-force reference at a
    // geometry the default budget covers in one tile.
    let (t, h, w, nh, hd) = (4usize, 5, 5, 1, 8);
    let shape = [1usize, t, h, w, nh, hd];
    let q = probe(&shape, 4);
    let k = probe(&shape, 5);
    let v = probe(&shape, 6);
    let tiles = pick_tiles([t, h, w], [3, 3, 3], NA_TILE_BUDGET);
    assert_eq!(tiles, [t, h, w], "this geometry fits one tile");
    let got = na3d(&q, &k, &v, [3, 3, 3]).unwrap();
    let want = na3d_reference(&q, &k, &v, [3, 3, 3]);
    assert!(max_abs_diff(&got, &want) < 2e-6);
}

#[test]
fn the_query_tiled_head_chunked_schedule_computes_the_same_operator() {
    // Production geometry always splits: a released stage-5 tile is far past `NA_TILE_BUDGET`, and
    // its score matrix is far past `NA_SCORE_BUDGET`. Nothing at a size a CPU test can brute-force
    // reaches either budget, so the split schedule is forced here instead — the *only* committed
    // test that runs it, and the schedule is the one documented candle-vs-MLX divergence.
    const SMALL_TILE: usize = 1 << 10;
    const SMALL_SCORE: usize = 1;

    let (t, h, w, nh, hd) = (4usize, 5, 5, 2, 8);
    let kernel = [3usize, 3, 3];
    let shape = [1usize, t, h, w, nh, hd];
    let q = probe(&shape, 4);
    let k = probe(&shape, 5);
    let v = probe(&shape, 6);

    // Precondition 1: the query axis really tiles.
    let tiles = pick_tiles([t, h, w], kernel, SMALL_TILE);
    assert_ne!(
        tiles,
        [t, h, w],
        "the forced budget must actually split the query grid"
    );

    // Precondition 2: the head axis really chunks. `head_chunk` is `score_budget / per_head`
    // clamped into `[1, heads]`, so at `SMALL_SCORE = 1` it is 1 for *every* per-head score count
    // — one head per matmul, and `1 < b * nh = 2` chunks of them.
    let heads = shape[0] * nh;
    assert!(heads > 1, "the head axis needs >= 2 rows to chunk");
    for per_head in [1usize, 27, 1 << 20] {
        assert_eq!(
            head_chunk(per_head, SMALL_SCORE, heads),
            1,
            "per_head {per_head}: the forced score budget must chunk one head at a time"
        );
    }

    let split = na3d_with_budgets(&q, &k, &v, kernel, SMALL_TILE, SMALL_SCORE).expect("split na3d");
    let whole = na3d(&q, &k, &v, kernel).expect("unsplit na3d");
    let want = na3d_reference(&q, &k, &v, kernel);

    let err_ref = max_abs_diff(&split, &want);
    assert!(
        err_ref < 2e-6,
        "the split schedule disagrees with the brute-force reference: max|delta| = {err_ref:.3e}"
    );
    let err_whole = max_abs_diff(&split, &whole);
    assert!(
        err_whole < 2e-6,
        "the split schedule disagrees with the unsplit one: max|delta| = {err_whole:.3e}"
    );
}

// ---------------------------------------------------------------------------------------------
// Rotary positions and the pixel shuffle — the two named scale-preserving failure modes
// ---------------------------------------------------------------------------------------------

#[test]
fn the_rope_split_partitions_head_dim_into_even_chunks() {
    for head_dim in [16usize, 24, 32, 64, 128] {
        let split = rope_dim_split(head_dim).expect("split");
        assert_eq!(
            split.iter().sum::<usize>(),
            head_dim,
            "head_dim {head_dim}: the three chunks must tile the head"
        );
        assert!(
            split.iter().all(|d| d % 2 == 0),
            "head_dim {head_dim}: every chunk is rotated in pairs, so each must be even"
        );
        assert_eq!(split[1], split[2], "H and W share a width");
    }
    assert_eq!(rope_dim_split(64).expect("released"), [16, 24, 24]);
    rope_dim_split(12).expect_err("a head_dim that is not a multiple of 8 has no default split");
}

#[test]
fn rotating_by_position_is_a_plain_rotation_at_the_axis_index() {
    // `rope_inv_freqs`'s first entry is `base^0 = 1`, so the first pair of an axis chunk rotates by
    // exactly `position` radians. That is a closed-form check on all three of the things a wrong
    // rotary implementation gets wrong: which pairs are paired, which axis supplies the position,
    // and whether the position starts at 0.
    let half = 2usize; // a 4-wide chunk -> two pairs
    let d = half * 2;
    let inv = rope_inv_freqs(d, ROPE_BASE, &Device::Cpu).expect("inv");
    let inv_host: Vec<f32> = inv.to_vec1().unwrap();
    assert_eq!(inv_host[0], 1.0, "the first inverse frequency is base^0");

    // (B, T, H, W, NH, D) with the pair `(XE, XO)` at every site. BOTH components are non-zero on
    // purpose: with `(1, 0)` the rotation degenerates and a sign flip in either term is invisible.
    const XE: f32 = 1.0;
    const XO: f32 = 0.5;
    let (t, h, w) = (3usize, 2, 2);
    let mut data = vec![0f32; t * h * w * d];
    for site in data.chunks_mut(d) {
        for pair in 0..half {
            site[2 * pair] = XE;
            site[2 * pair + 1] = XO;
        }
    }
    let x = Tensor::from_vec(data, (1, t, h, w, 1, d), &Device::Cpu).unwrap();

    for (axis, len) in [(1usize, t), (2, h), (3, w)] {
        let out = rotate_axis(&x, &inv, axis).expect("rotate");
        let host: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for position in 0..len {
            // Any site at `position` along `axis` will do; take the first.
            let index = match axis {
                1 => position * h * w,
                2 => position * w,
                _ => position,
            } * d;
            for pair in 0..half {
                let angle = position as f32 * inv_host[pair];
                let (want_e, want_o) = (
                    XE * angle.cos() - XO * angle.sin(),
                    XE * angle.sin() + XO * angle.cos(),
                );
                let (got_e, got_o) = (host[index + 2 * pair], host[index + 2 * pair + 1]);
                assert!(
                    (got_e - want_e).abs() < 1e-6 && (got_o - want_o).abs() < 1e-6,
                    "axis {axis} position {position} pair {pair}: got ({got_e}, {got_o}), want \
                     ({want_e}, {want_o})"
                );
            }
        }
    }
}

#[test]
fn the_pixel_shuffle_places_each_sub_cell_where_the_reference_does() {
    // The upsample projects to `C_out * p1 * p2 * p3` and unshuffles. Upstream's ordering is
    // `(b, t, h, w, c, p1, p2, p3)` -> `(b, t, p1, h, p2, w, p3, c)`, i.e. the *slowest* varying
    // index of the packed channel axis is the output channel and the fastest is the W sub-cell.
    // A transposed shuffle preserves every norm and every moment, so only an index check finds it.
    let (p1, p2, p3, c) = (2usize, 2, 2, 3);
    let fan = p1 * p2 * p3;
    let (t, h, w) = (2usize, 2, 2);
    let packed = c * fan;
    // An identity projection, so the test reads the shuffle rather than the GEMM.
    let mut eye = vec![0f32; packed * packed];
    for i in 0..packed {
        eye[i * packed + i] = 1.0;
    }
    let up = PixelShuffleUpsample {
        proj: Linear {
            w: Weight::Dense(Tensor::from_vec(eye, (packed, packed), &Device::Cpu).unwrap()),
            b: Tensor::zeros(packed, DType::F32, &Device::Cpu).unwrap(),
        },
        stride: [p1, p2, p3],
    };
    assert_eq!(up.out_channels(), c);

    let n = t * h * w * packed;
    let input = Tensor::from_vec(
        (0..n).map(|i| i as f32).collect::<Vec<_>>(),
        (1, t, h, w, packed),
        &Device::Cpu,
    )
    .unwrap();
    // `drop_leading_frame = false`: the frame drop is a separate claim, checked below.
    let out = up.forward(&input, false).expect("shuffle");
    assert_eq!(out.dims(), [1, t * p1, h * p2, w * p3, c]);
    let host: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    for i1 in 0..p1 {
                        for i2 in 0..p2 {
                            for i3 in 0..p3 {
                                let src = (((ti * h + hi) * w + wi) * packed)
                                    + ((ci * p1 + i1) * p2 + i2) * p3
                                    + i3;
                                let dst = ((((ti * p1 + i1) * (h * p2) + hi * p2 + i2) * (w * p3)
                                    + wi * p3
                                    + i3)
                                    * c)
                                    + ci;
                                assert_eq!(
                                    host[dst], src as f32,
                                    "sub-cell ({i1},{i2},{i3}) of ({ti},{hi},{wi}) channel {ci}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // The temporal stride of 2 duplicates the leading frame; the origin chunk drops it.
    let dropped = up.forward(&input, true).expect("shuffle, origin chunk");
    assert_eq!(dropped.dims()[1], t * p1 - 1);
    let kept: Vec<f32> = dropped.flatten_all().unwrap().to_vec1().unwrap();
    let stride = (h * p2) * (w * p3) * c;
    assert_eq!(
        &kept[..stride],
        &host[stride..2 * stride],
        "the origin chunk must drop frame 0, not frame 1"
    );
}

// ---------------------------------------------------------------------------------------------
// Split / blend arithmetic
// ---------------------------------------------------------------------------------------------

#[test]
fn a_split_that_covers_its_axis_validates_and_one_that_does_not_is_refused() {
    let split = split_by_size(40, 16, 6, 6);
    validate_split(&split, 40, 6, 0).expect("a covering split");
    assert_eq!(split[0].start, 0);
    assert_eq!(split.last().unwrap().end, 40);

    let mut broken = split.clone();
    broken.pop();
    let err = validate_split(&broken, 40, 6, 0)
        .expect_err("a split that stops short must be refused")
        .to_string();
    assert!(err.contains("cover"), "{err}");
}

#[test]
fn a_short_trailing_tile_is_grown_to_the_window_floor() {
    // 33 cells, 16-long tiles overlapping by 6: the natural last tile is a 3-cell sliver.
    let split = split_by_size(33, 16, 6, 6);
    let last = *split.last().unwrap();
    assert!(
        last.end - last.start >= 6,
        "the trailing tile is {} long, below the floor",
        last.end - last.start
    );
    validate_split(&split, 33, 6, 0).expect("the grown split is still consistent");
}

#[test]
fn neighbouring_trapezoids_sum_to_one_over_their_overlap() {
    let overlap = 5usize;
    let left = trapezoid(20, 0, overlap);
    let right = trapezoid(20, overlap, 0);
    for i in 0..overlap {
        let total = left[20 - overlap + i] + right[i];
        assert!(
            (total - 1.0).abs() < 1e-6,
            "overlap slot {i} sums to {total}"
        );
    }
}

#[test]
fn a_causal_propagation_drops_the_duplicate_frame_only_off_the_origin_tile() {
    let origin = Interval {
        start: 0,
        end: 4,
        left_ramp: 0,
        right_ramp: 2,
    };
    let later = Interval {
        start: 4,
        end: 8,
        left_ramp: 2,
        right_ramp: 0,
    };
    let up_origin = propagate(origin, 2, true);
    let up_later = propagate(later, 2, true);
    assert_eq!((up_origin.start, up_origin.end), (0, 7));
    assert_eq!((up_later.start, up_later.end), (7, 15));
    // Contiguity is the property the blend rests on: the later tile starts exactly where the
    // origin one ended, with no gap and no double-counted frame.
    assert_eq!(up_origin.end, up_later.start);
}

// ---------------------------------------------------------------------------------------------
// A miniature decoder, end to end
// ---------------------------------------------------------------------------------------------

/// A decoder small enough to run on a CPU in a unit test but structurally identical to the released
/// one: two deterministic stages, one diffusion stage, one upsample hop.
fn tiny_config() -> NaDiffusionDecoderConfig {
    NaDiffusionDecoderConfig::from_embedded_vae(&serde_json::json!({
        "model_output_type": "x0",
        "decoder": {
            "_class_name": "NADiffusionDecoder",
            "in_channels": 4,
            "out_channels": 3,
            "patch_size": 2,
            "head_dim": 16,
            "stage_channels": [32, 16],
            "stage_depths": [1, 1],
            "stage_kernels": [[3, 3, 3], [3, 3, 3]],
            "upsamples": [[[2, 2, 2], 2]],
            "stage5_kernel": [3, 3, 3],
            "t_emb_dim": 32,
            "default_num_inference_steps": 1,
            "timestep_scale_multiplier": 1000.0
        }
    }))
    .expect("tiny config")
}

/// Deterministic synthetic weights for `cfg`, keyed exactly as the checkpoint keys them.
///
/// Shapes are derived here rather than read from a file, so a loader that started reading a
/// differently-shaped tensor fails this rather than silently reshaping.
fn tiny_weights(cfg: &NaDiffusionDecoderConfig) -> HashMap<String, Tensor> {
    /// Deterministic small values: a random 16-wide GEMM stack saturates a softmax otherwise, and
    /// this is a structural test, not a numeric one.
    fn put(out: &mut HashMap<String, Tensor>, seed: &mut usize, name: String, shape: &[usize]) {
        *seed += 1;
        let t = (probe(shape, *seed) * 0.2).expect("scaled probe");
        out.insert(name, t);
    }
    fn linear(out: &mut HashMap<String, Tensor>, seed: &mut usize, name: &str, o: usize, i: usize) {
        put(out, seed, format!("{name}.weight"), &[o, i]);
        put(out, seed, format!("{name}.bias"), &[o]);
    }
    fn block(
        out: &mut HashMap<String, Tensor>,
        seed: &mut usize,
        cfg: &NaDiffusionDecoderConfig,
        prefix: &str,
        dim: usize,
        diffusion: bool,
    ) {
        put(out, seed, format!("{prefix}.norm1.weight"), &[dim]);
        put(out, seed, format!("{prefix}.norm2.weight"), &[dim]);
        linear(out, seed, &format!("{prefix}.attn.qkv"), 3 * dim, dim);
        linear(out, seed, &format!("{prefix}.attn.proj"), dim, dim);
        put(
            out,
            seed,
            format!("{prefix}.attn.q_norm.weight"),
            &[cfg.head_dim],
        );
        put(
            out,
            seed,
            format!("{prefix}.attn.k_norm.weight"),
            &[cfg.head_dim],
        );
        let hidden = mlp_hidden(dim);
        put(
            out,
            seed,
            format!("{prefix}.mlp.w_gate.weight"),
            &[hidden, dim],
        );
        put(
            out,
            seed,
            format!("{prefix}.mlp.w_up.weight"),
            &[hidden, dim],
        );
        put(
            out,
            seed,
            format!("{prefix}.mlp.w_down.weight"),
            &[dim, hidden],
        );
        if diffusion {
            linear(out, seed, &format!("{prefix}.context_proj"), dim, dim);
            put(
                out,
                seed,
                format!("{prefix}.scale_shift_table"),
                &[ADALN_CHUNKS, dim],
            );
        }
    }

    let out = &mut HashMap::new();
    let seed = &mut 0usize;
    let c5 = cfg.stage5_width();
    let patched = cfg.out_channels * cfg.patch_size * cfg.patch_size;
    linear(out, seed, "conv_in", cfg.stage_channels[0], cfg.in_channels);
    linear(out, seed, "conv_in_x_t", c5, patched);
    linear(out, seed, "conv_out", patched, c5);
    linear(
        out,
        seed,
        "shared_adaln.proj",
        ADALN_CHUNKS * c5,
        cfg.t_emb_dim,
    );
    linear(out, seed, "t_embedder.mlp.0", cfg.t_emb_dim, 256);
    linear(out, seed, "t_embedder.mlp.2", cfg.t_emb_dim, cfg.t_emb_dim);
    put(out, seed, "norm_out.weight".to_string(), &[c5]);

    for stage in 0..cfg.upsamples.len() {
        for i in 0..cfg.stage_depths[stage] {
            block(
                out,
                seed,
                cfg,
                &format!("det_stages.{stage}.{i}"),
                cfg.stage_channels[stage],
                false,
            );
        }
        let (stride, _) = cfg.upsamples[stage];
        let fan = stride[0] * stride[1] * stride[2];
        linear(
            out,
            seed,
            &format!("upsamples.{stage}.proj"),
            cfg.stage_channels[stage + 1] * fan,
            cfg.stage_channels[stage],
        );
    }
    for i in 0..*cfg.stage_depths.last().expect("validated") {
        block(out, seed, cfg, &format!("diff_blocks.{i}"), c5, true);
    }
    std::mem::take(out)
}

fn tiny_decoder(cfg: &NaDiffusionDecoderConfig) -> NaDiffusionDecoder {
    let mut tensors: HashMap<String, Tensor> = tiny_weights(cfg)
        .into_iter()
        .map(|(k, v)| (format!("{DECODER_PREFIX}.{k}"), v))
        .collect();
    let stat = |v: f32| Tensor::full(v, cfg.in_channels, &Device::Cpu).unwrap();
    tensors.insert(STAT_MEAN_KEY.to_string(), stat(0.0));
    tensors.insert(STAT_STD_KEY.to_string(), stat(1.0));
    let root = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    NaDiffusionDecoder::load(root.pp(DECODER_PREFIX), root, cfg).expect("tiny decoder")
}

#[test]
fn every_key_the_loader_reads_is_a_key_the_checkpoint_ships() {
    // `expected_weight_keys` is what loader audits are written against, so it has to be the same
    // set `load` actually asks for. `tiny_weights` builds exactly `expected_weight_keys`; a loader
    // that read one more key would fail to build below.
    let cfg = tiny_config();
    let built: std::collections::BTreeSet<String> = tiny_weights(&cfg).into_keys().collect();
    let declared: std::collections::BTreeSet<String> =
        expected_weight_keys(&cfg).into_iter().collect();
    assert_eq!(built, declared, "the audit list and the loader disagree");
    let _ = tiny_decoder(&cfg);
}

#[test]
fn the_released_key_audit_covers_every_decoder_tensor_the_checkpoint_carries() {
    // 310 `decoder.*` tensors in the released file (sc-18765 keyset), of which `type_emb` is the
    // one the reference itself drops.
    let cfg = released();
    assert_eq!(
        expected_weight_keys(&cfg).len() + UNUSED_DECODER_KEYS.len(),
        310,
        "the audit list no longer accounts for every decoder tensor in the released checkpoint"
    );
}

#[test]
fn a_config_that_disagrees_with_the_weights_is_refused_at_load() {
    let cfg = tiny_config();
    let mut tensors: HashMap<String, Tensor> = tiny_weights(&cfg)
        .into_iter()
        .map(|(k, v)| (format!("{DECODER_PREFIX}.{k}"), v))
        .collect();
    let stat = |v: f32| Tensor::full(v, cfg.in_channels, &Device::Cpu).unwrap();
    tensors.insert(STAT_MEAN_KEY.to_string(), stat(0.0));
    tensors.insert(STAT_STD_KEY.to_string(), stat(1.0));
    let root = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // The weights are 16-wide at stage 1; claim 8 and the cross-check must say so rather than
    // failing several stages deep in a reshape.
    let mut wrong = cfg.clone();
    wrong.stage_channels[0] = 8;
    let err = NaDiffusionDecoder::load(root.pp(DECODER_PREFIX), root, &wrong)
        .expect_err("a config/weight width disagreement must be a load error")
        .to_string();
    assert!(err.contains("in the config"), "{err}");
}

#[test]
fn the_decoder_returns_the_geometry_its_latent_asks_for() {
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let (lt, lh, lw) = (3usize, 4, 4);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 11);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 12);
    let pixels = decoder.decode(&latent, &noise).expect("decode");
    let scale = cfg.pixel_scale();
    assert_eq!(
        pixels.dims(),
        [
            1,
            cfg.out_channels,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2]
        ]
    );
    let values: Vec<f32> = pixels.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "the decode produced non-finite pixels"
    );
}

#[test]
fn a_noise_canvas_of_the_wrong_shape_is_refused() {
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 11);
    let shape5 = cfg.noise_shape(3, 4, 4);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1] + 2, shape5[2]],
        12,
    );
    let err = decoder
        .decode(&latent, &noise)
        .expect_err("a mis-shaped stage-5 canvas must be refused, not broadcast")
        .to_string();
    assert!(err.contains("noise must be"), "{err}");
}

#[test]
fn a_single_covering_tile_reproduces_the_untiled_decode() {
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let (lt, lh, lw) = (3usize, 4, 4);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 21);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 22);

    let untiled = decoder.decode(&latent, &noise).expect("untiled");
    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let tiling = DiffVaeTiling {
        tile: stage4,
        overlap: cfg.tile_halo(),
    };
    let tiled = decoder
        .decode_tiled(&latent, &noise, &tiling)
        .expect("one covering tile");
    let err = max_abs_diff(&tiled, &untiled);
    assert!(
        err < 1e-5,
        "one covering tile must reproduce the untiled decode, max|delta| = {err:.3e}"
    );
}

#[test]
fn a_heavy_overlap_tiling_is_still_normalised_to_the_right_level() {
    // Ramps that meet multiply, so the blend is not a partition of unity and only the accumulated
    // weight profile makes it one. This is the regime that catches an unnormalised blend; a
    // complementary tiling would not.
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let (lt, lh, lw) = (3usize, 8, 8);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 31);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 32);
    let untiled = decoder.decode(&latent, &noise).expect("untiled");

    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let halo = cfg.tile_halo();
    let tile = [stage4[0], cfg.min_tile_shape()[1] + 1, stage4[2]];
    let overlap = [halo[0], tile[1] - 1, halo[2]];
    assert!(tile[1] < stage4[1], "the height axis must actually split");
    assert!(
        2 * overlap[1] > tile[1],
        "the ramps must meet for this test to ask its question"
    );
    let tiled = decoder
        .decode_tiled(&latent, &noise, &DiffVaeTiling { tile, overlap })
        .expect("heavy-overlap tiling");
    assert_eq!(tiled.dims(), untiled.dims());

    // The scale-sensitive statistic, not the signed mean: an unnormalised blend multiplies the
    // overlap region by its accumulated weight, which moves |x| whatever the picture's level is.
    // A signed mean near zero would hide exactly that.
    let mean_abs = |x: &Tensor| -> f32 {
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        v.iter().map(|y| y.abs()).sum::<f32>() / v.len() as f32
    };
    let (a, b) = (mean_abs(&tiled), mean_abs(&untiled));
    assert!(
        (a / b - 1.0).abs() < 0.05,
        "the blend rescaled the picture by {:.4}x: mean|x| {a:.5} tiled vs {b:.5} untiled",
        a / b
    );
}

#[test]
fn an_under_haloed_tiling_is_refused_on_every_axis() {
    let cfg = released();
    let halo = cfg.tile_halo();
    for axis in 0..3 {
        let mut starved = DiffVaeTiling {
            tile: [halo[0] * 3, halo[1] * 3, halo[2] * 3],
            overlap: halo,
        };
        starved.overlap[axis] = halo[axis] - 1;
        let err = starved
            .validated(&cfg)
            .expect_err("an under-haloed tiling must be refused, not smeared")
            .to_string();
        assert!(err.contains("halo"), "axis {axis}: {err}");
    }
}

#[test]
fn a_tile_below_the_window_floor_is_refused() {
    let cfg = released();
    let floor = cfg.min_tile_shape();
    let err = DiffVaeTiling {
        tile: [floor[0] - 1, 64, 64],
        overlap: cfg.tile_halo(),
    }
    .validated(&cfg)
    .expect_err("a tile narrower than the stage-4/5 window must be refused")
    .to_string();
    assert!(err.contains("window floor"), "{err}");
}

#[test]
fn a_split_axis_whose_overlap_swallows_its_tile_is_refused_at_decode() {
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let (lt, lh, lw) = (3usize, 8, 8);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 41);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 42);
    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let floor = cfg.min_tile_shape();
    let tile = [stage4[0], floor[1], stage4[2]];
    let overlap = [cfg.tile_halo()[0], floor[1], cfg.tile_halo()[2]];
    assert!(tile[1] < stage4[1], "the height axis must actually split");
    let err = decoder
        .decode_tiled(&latent, &noise, &DiffVaeTiling { tile, overlap })
        .expect_err("a split axis needs overlap < tile")
        .to_string();
    assert!(err.contains("smaller than the tile"), "{err}");
}

#[test]
fn a_diffusion_checkpoints_keys_are_never_mistaken_for_a_conv_decoders() {
    let cfg = released();
    let diffusion: Vec<String> = expected_weight_keys(&cfg)
        .into_iter()
        .map(|k| format!("decoder.{k}"))
        .collect();
    assert!(looks_like_diffusion_decoder(
        diffusion.iter().map(String::as_str)
    ));
    // The conv decoder's own key shape (sc-18765 keyset): `up_blocks.*` / `conv_in.*`, no
    // `det_stages` and no `diff_blocks`.
    let conv = [
        "decoder.conv_in.weight",
        "decoder.up_blocks.0.res_blocks.0.conv1.weight",
        "decoder.conv_out.weight",
    ];
    assert!(!looks_like_diffusion_decoder(conv));
}

#[test]
fn the_schedule_is_torch_linspace_from_one_down_to_one_over_n() {
    let mut cfg = tiny_config();
    assert_eq!(cfg.model_output_type, ModelOutputType::X0);
    let decoder = tiny_decoder(&cfg);
    assert_eq!(decoder.timesteps(), vec![1.0]);

    cfg.model_output_type = ModelOutputType::Velocity;
    cfg.default_num_inference_steps = 3;
    let decoder = tiny_decoder(&cfg);
    // torch.linspace(1, 1/3, 3).
    let ts = decoder.timesteps();
    assert_eq!(ts.len(), 3);
    assert!((ts[0] - 1.0).abs() < 1e-9);
    assert!((ts[1] - 2.0 / 3.0).abs() < 1e-6, "{ts:?}");
    assert!((ts[2] - 1.0 / 3.0).abs() < 1e-6, "{ts:?}");
}

#[test]
fn the_euler_update_is_the_literal_arithmetic_of_each_objective() {
    // Closed form on literal tensors, at `t_now != 1` and `t_next != 0` so the two arms are
    // distinguishable and neither `(t_now - t_next)` nor the `/ t_now` divide degenerates:
    //   Velocity: x_t - v * (t_now - t_next)
    //   X0:       x_t - ((x_t - out) / t_now) * (t_now - t_next)
    const X: f32 = 2.0;
    const OUT: f32 = 0.5;
    const T_NOW: f64 = 0.6;
    const T_NEXT: f64 = 0.25;

    let x_t = Tensor::full(X, (1usize, 2, 2), &Device::Cpu).expect("x_t");
    let model_out = Tensor::full(OUT, (1usize, 2, 2), &Device::Cpu).expect("model_out");
    let host = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };

    let mut cfg = tiny_config();
    cfg.model_output_type = ModelOutputType::Velocity;
    let v_dec = tiny_decoder(&cfg);
    let want_v = X - OUT * (T_NOW - T_NEXT) as f32;
    for got in host(
        &v_dec
            .euler_step(&x_t, &model_out, T_NOW, T_NEXT)
            .expect("v step"),
    ) {
        assert!((got - want_v).abs() < 1e-6, "v: got {got}, want {want_v}");
    }

    cfg.model_output_type = ModelOutputType::X0;
    let x0_dec = tiny_decoder(&cfg);
    let want_x0 = X - ((X - OUT) / T_NOW as f32) * (T_NOW - T_NEXT) as f32;
    for got in host(
        &x0_dec
            .euler_step(&x_t, &model_out, T_NOW, T_NEXT)
            .expect("x0 step"),
    ) {
        assert!(
            (got - want_x0).abs() < 1e-6,
            "x0: got {got}, want {want_x0}"
        );
    }
    assert!(
        (want_v - want_x0).abs() > 0.5,
        "the two objectives must land somewhere different for this test to ask its question"
    );
}

#[test]
fn the_x0_schedule_returns_the_prediction_and_v_takes_a_final_euler_step() {
    // At `t_next = 0` the X0 Euler step collapses to `out` itself, so the terminal `match` is only
    // observable through the Velocity arm: `x_T - out * t_last`. With `N = 1` (`t_last = 1`) the
    // two decoders share every weight and every intermediate, so the X0 decode *is* `crop(out)`
    // and the Velocity decode must be `crop(noise) - crop(out)` exactly. Swapping the arms makes
    // the Velocity decode `crop(out)`; flipping the step's sign makes it `crop(noise) + crop(out)`.
    let mut cfg = tiny_config();
    assert_eq!(cfg.model_output_type, ModelOutputType::X0);
    assert_eq!(cfg.default_num_inference_steps, 1);

    let (lt, lh, lw) = (3usize, 4, 4);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 51);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 52);

    let x0_dec = tiny_decoder(&cfg);
    let x0_pixels = x0_dec.decode(&latent, &noise).expect("x0 decode");

    cfg.model_output_type = ModelOutputType::Velocity;
    let v_dec = tiny_decoder(&cfg);
    let v_pixels = v_dec.decode(&latent, &noise).expect("v decode");

    // The same crop the decode applies, so the comparison is in the returned geometry.
    let scale = cfg.pixel_scale();
    let (_, h_pad, w_pad) = x0_dec.prepare_latent(&latent).expect("prepare");
    let cropped_noise = x0_dec
        .crop_to_content(
            &noise,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2],
            h_pad,
            w_pad,
        )
        .expect("crop the noise the same way");

    let want = (&cropped_noise - &x0_pixels).expect("x_T - out");
    let err = max_abs_diff(&v_pixels, &want);
    assert!(
        err < 1e-5,
        "the velocity decode is not `x_T - out * t_last`: max|delta| = {err:.3e}"
    );
    // And the two objectives really do land somewhere different, so the equality above is not
    // being satisfied by a degenerate `out`.
    assert!(
        max_abs_diff(&v_pixels, &x0_pixels) > 1e-3,
        "the two objectives decoded to the same picture"
    );
}

#[test]
fn a_multi_step_velocity_schedule_takes_every_intermediate_euler_step() {
    // `N = 2` runs the in-loop Euler update once before the terminal one; `N = 1` runs none. A
    // decoder that ignored the loop would return the same picture for both.
    let mut cfg = tiny_config();
    cfg.model_output_type = ModelOutputType::Velocity;
    let one_step = tiny_decoder(&cfg);
    cfg.default_num_inference_steps = 2;
    let two_step = tiny_decoder(&cfg);
    assert_eq!(two_step.timesteps().len(), 2);

    let (lt, lh, lw) = (3usize, 4, 4);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 61);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 62);

    let a = one_step.decode(&latent, &noise).expect("1-step decode");
    let b = two_step.decode(&latent, &noise).expect("2-step decode");
    assert!(
        max_abs_diff(&a, &b) > 1e-3,
        "the second sampling step changed nothing"
    );
}

#[test]
fn a_multiaxis_tiled_decode_crosses_real_boundaries_and_is_mutation_sensitive() {
    // This is deliberately an end-to-end CPU test, rather than an assertion on a `DiffVaeTiling`
    // value.  A test which only inspected the selected shape could pass if `decode_tiled` quietly
    // called `decode`; force temporal plus spatial splits, prove their result differs from the
    // single-pass path, then perturb one input value and require that the tiled result responds.
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let (lt, lh, lw) = (5usize, 5, 5);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 71);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 72);
    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let tiling = DiffVaeTiling {
        tile: [stage4[0] - 2, stage4[1] - 2, stage4[2]],
        overlap: cfg.tile_halo(),
    };
    let split_axes: Vec<usize> = (0..3)
        .filter(|&axis| tiling.tile[axis] < stage4[axis])
        .collect();
    assert!(
        split_axes.contains(&0) && split_axes.len() >= 2,
        "the test must cross a temporal and spatial boundary, got {split_axes:?}"
    );

    let untiled = decoder.decode(&latent, &noise).expect("untiled decode");
    let tiled = decoder
        .decode_tiled(&latent, &noise, &tiling)
        .expect("multiaxis tiled decode");
    assert!(
        max_abs_diff(&tiled, &untiled) > 1e-6,
        "the selected multiaxis tiling was bypassed by the single-pass path"
    );

    let mut altered: Vec<f32> = latent.flatten_all().unwrap().to_vec1().unwrap();
    let altered_index = altered.len() / 2;
    altered[altered_index] += 0.25;
    let altered = Tensor::from_vec(altered, latent.dims(), &Device::Cpu).expect("altered latent");
    let altered_tiled = decoder
        .decode_tiled(&altered, &noise, &tiling)
        .expect("tiled decode after latent mutation");
    assert!(
        max_abs_diff(&altered_tiled, &tiled) > 1e-6,
        "the boundary-crossing tiled decode ignored a non-degenerate latent mutation"
    );
}

#[test]
fn budgeted_seeded_decode_preserves_the_seeded_canvas_and_reaches_the_selector() {
    // This is the executable counterpart to the provider-route guard: the ordinary provider uses
    // `decode_budgeted_seeded`, so it must draw the same full canvas as `decode_seeded` and then
    // enter `decode_budgeted` rather than silently selecting the unbounded decode path.
    let cfg = tiny_config();
    let decoder = tiny_decoder(&cfg);
    let latent = probe(&[1, cfg.in_channels, 3, 4, 4], 81);
    let noise = decoder.seeded_noise(&latent, 82).expect("seeded canvas");
    let explicit = decoder
        .decode_budgeted(&latent, &noise, DiffVaeMode::ChunkedEager)
        .expect("explicit budgeted decode");
    let seeded = decoder
        .decode_budgeted_seeded(&latent, 82, DiffVaeMode::ChunkedEager)
        .expect("seeded budgeted decode");
    assert!(
        max_abs_diff(&seeded, &explicit) < 1e-5,
        "budgeted seeded decode changed its full noise canvas or bypassed the selector"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-18799 — the budgeted DiffVAE selector
// ---------------------------------------------------------------------------------------------

use super::budget::{
    auto_diffvae_tiling_budgeted_ltx, compute_cap_is_datacenter_blackwell,
    estimated_diffvae_decode_peak_bytes, plan_diffvae_tiling, DecodeGeometry, DecodePlan,
    DiffVaeMode, HostNaSupport, NaKind, ResolvedDiffVaeMode,
};

/// The resolved mode every plan on a host without NATTEN runs under — i.e. every host this crate
/// builds for.
fn eager() -> ResolvedDiffVaeMode {
    DiffVaeMode::ChunkedEager
        .resolve_for_host(HostNaSupport::detect(&Device::Cpu))
        .expect("chunked_eager is the mode this backend serves")
}

#[test]
fn the_four_upstream_modes_declare_their_own_coefficients_and_withholds() {
    // Upstream `_MEM_COEF_BY_MODE` / `_BUDGET_SAFETY_BYTES_*`, verbatim, and byte-for-byte the same
    // table the MLX twin declares — the two backends must not drift on the mode contract.
    let declared: Vec<(&str, f64, u64)> = DiffVaeMode::ALL_MODES
        .iter()
        .map(|m| {
            (
                m.as_str(),
                m.declared_stage5_coef(),
                m.declared_budget_safety_bytes(),
            )
        })
        .collect();
    assert_eq!(
        declared,
        vec![
            ("chunked_eager", 5.0, 1 << 30),
            ("chunked_compile", 7.0, 2 << 30),
            ("combined_compile", 11.0, 2 << 30),
            ("blackwell_dsl", 2.5, 2 << 30),
        ]
    );
    for mode in DiffVaeMode::ALL_MODES {
        assert_eq!(DiffVaeMode::parse(mode.as_str()).unwrap(), mode);
    }
    let err = DiffVaeMode::parse("combined_eager")
        .unwrap_err()
        .to_string();
    assert!(err.contains("chunked_eager"), "{err}");
}

#[test]
fn the_host_resolve_reproduces_upstreams_natten_and_blackwell_rules() {
    let natten = HostNaSupport::with(true, false, false);
    for mode in [
        DiffVaeMode::ChunkedEager,
        DiffVaeMode::ChunkedCompile,
        DiffVaeMode::CombinedCompile,
    ] {
        let r = mode.resolve_for_host(natten).unwrap();
        assert_eq!(r.attention, NaKind::Natten);
        assert_eq!(r.stage5_coef, mode.declared_stage5_coef());
        assert_eq!(r.budget_safety_bytes, mode.declared_budget_safety_bytes());
    }
    let fallback = HostNaSupport::with(false, false, false);
    for mode in [DiffVaeMode::ChunkedEager, DiffVaeMode::ChunkedCompile] {
        let r = mode.resolve_for_host(fallback).unwrap();
        assert_eq!(r.attention, NaKind::EagerSdpa);
        assert_eq!(r.stage5_coef, 5.0, "{}", mode.as_str());
        assert_eq!(r.budget_safety_bytes, 1 << 30, "{}", mode.as_str());
    }
    let err = DiffVaeMode::CombinedCompile
        .resolve_for_host(fallback)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires NATTEN"), "{err}");
    let r = DiffVaeMode::BlackwellDsl
        .resolve_for_host(HostNaSupport::with(false, false, true))
        .unwrap();
    assert_eq!(r.attention, NaKind::BlackwellDsl);
    assert_eq!(r.stage5_coef, 2.5);
    assert_eq!(r.budget_safety_bytes, 2 << 30);
}

#[test]
fn the_blackwell_gate_fails_closed_on_a_device_it_cannot_prove() {
    // The hardware gate on the device this test actually binds. A CPU device — and, on a build
    // without the `cuda` feature, any device — cannot be a datacenter Blackwell part, and the
    // refusal must say which of those it is rather than "unsupported".
    let host = HostNaSupport::detect(&Device::Cpu);
    assert!(!host.natten && !host.triton && !host.blackwell_dsl);
    eprintln!("[gate] blackwell_dsl reason: {}", host.blackwell_reason);
    assert!(
        host.blackwell_reason.contains("CUDA"),
        "the reason must name the CUDA requirement: {}",
        host.blackwell_reason
    );
    let err = DiffVaeMode::BlackwellDsl
        .resolve_for_host(host)
        .unwrap_err()
        .to_string();
    assert!(err.contains("datacenter Blackwell"), "{err}");
    assert!(
        err.contains("chunked_eager"),
        "the refusal must offer a way out: {err}"
    );
    // The production entry point refuses too, rather than silently falling back to a mode the
    // caller did not ask for.
    let cfg = released();
    let err =
        auto_diffvae_tiling_budgeted_ltx(&cfg, &Device::Cpu, 4, 22, 40, DiffVaeMode::BlackwellDsl)
            .unwrap_err()
            .to_string();
    assert!(err.contains("blackwell_dsl"), "{err}");
}

#[test]
fn datacenter_blackwell_is_major_ten_and_consumer_sm_120_is_not() {
    assert!(compute_cap_is_datacenter_blackwell((10, 0)), "B200 sm_100");
    assert!(compute_cap_is_datacenter_blackwell((10, 3)), "B300 sm_103");
    // The scar: consumer Blackwell is sm_120 — real Blackwell silicon, wrong Blackwell for this
    // kernel. A `>=` floor would accept it. This predicate is deliberately not a floor.
    assert!(
        !compute_cap_is_datacenter_blackwell((12, 0)),
        "consumer Blackwell sm_120 must NOT satisfy the datacenter-Blackwell gate"
    );
    assert!(!compute_cap_is_datacenter_blackwell((9, 0)), "Hopper");
    assert!(!compute_cap_is_datacenter_blackwell((8, 9)), "Ada");
}

#[test]
fn a_single_pass_that_fits_selects_no_tiling_and_one_that_does_not_selects_a_tile() {
    let cfg = released();
    let generous = plan_diffvae_tiling(&cfg, 4, 22, 40, 160.0, &eager()).unwrap();
    assert!(
        generous.is_none(),
        "a 160 GiB budget must take the single-pass decode, got {generous:?}"
    );
    let tight = plan_diffvae_tiling(&cfg, 4, 22, 40, 24.0, &eager())
        .unwrap()
        .expect("24 GiB must not fit the single-pass decode at this geometry");
    let stage4 = cfg.stage4_shape(4, 22, 40);
    assert!(
        (0..3).any(|a| tight.tile[a] < stage4[a]),
        "a tiling that splits nothing is not a tiling: {tight:?} vs stage-4 grid {stage4:?}"
    );
    assert_eq!(
        tight.overlap,
        cfg.tile_halo(),
        "the selector must use the stage-4/5 halo as its overlap"
    );
}

#[test]
fn every_selected_tiling_is_one_decode_tiled_will_accept() {
    let cfg = released();
    let mut planned = 0usize;
    for &(t, h, w) in &[(4usize, 22usize, 40usize), (7, 16, 24), (4, 34, 60)] {
        for &safe_gib in &[16.0_f64, 24.0, 32.0, 48.0, 64.0] {
            let Ok(Some(tiling)) = plan_diffvae_tiling(&cfg, t, h, w, safe_gib, &eager()) else {
                continue;
            };
            planned += 1;
            tiling
                .validated(&cfg)
                .unwrap_or_else(|e| panic!("{t}x{h}x{w} @ {safe_gib} GiB → {tiling:?}: {e}"));
            let geometry = DecodeGeometry::new(&cfg, t, h, w);
            let bytes =
                estimated_diffvae_decode_peak_bytes(&geometry, DecodePlan::Tiled(tiling), &eager());
            let usable = (safe_gib * 1024.0 * 1024.0 * 1024.0) as u64 - eager().budget_safety_bytes;
            assert!(
                bytes <= usable,
                "{t}x{h}x{w} @ {safe_gib} GiB: selected {tiling:?} costs {bytes} > usable {usable}"
            );
        }
    }
    assert!(
        planned >= 6,
        "only {planned} plans exercised — the sweep is not covering the tiled arm"
    );
}

#[test]
fn the_estimate_is_monotone_in_the_tile_and_in_the_coefficient() {
    let cfg = released();
    let geometry = DecodeGeometry::new(&cfg, 4, 22, 40);
    let overlap = cfg.tile_halo();
    let small = DecodePlan::Tiled(DiffVaeTiling {
        tile: [geometry.stage4[0], 40, 40],
        overlap,
    });
    let large = DecodePlan::Tiled(DiffVaeTiling {
        tile: geometry.stage4,
        overlap,
    });
    let r = eager();
    assert!(
        estimated_diffvae_decode_peak_bytes(&geometry, small, &r)
            < estimated_diffvae_decode_peak_bytes(&geometry, large, &r),
        "a bigger tile must never cost less"
    );
    let combined = DiffVaeMode::CombinedCompile
        .resolve_for_host(HostNaSupport::with(true, false, false))
        .unwrap();
    let dsl = DiffVaeMode::BlackwellDsl
        .resolve_for_host(HostNaSupport::with(false, false, true))
        .unwrap();
    assert!(
        estimated_diffvae_decode_peak_bytes(&geometry, large, &dsl)
            < estimated_diffvae_decode_peak_bytes(&geometry, large, &r)
            && estimated_diffvae_decode_peak_bytes(&geometry, large, &r)
                < estimated_diffvae_decode_peak_bytes(&geometry, large, &combined),
        "the per-mode coefficients must order the estimates 2.5 < 5 < 11"
    );
}

#[test]
fn an_unplannable_geometry_is_a_catchable_error_naming_the_budget() {
    let cfg = released();
    let err = plan_diffvae_tiling(&cfg, 31, 68, 120, 4.0, &eager())
        .expect_err("a 4K/241-frame clip cannot be planned at 4 GiB")
        .to_string();
    assert!(err.contains("ltx diffvae decode"), "{err}");
    assert!(err.contains("safe budget"), "{err}");
    assert!(
        err.contains("chunked_eager"),
        "the error must name the mode: {err}"
    );
}
