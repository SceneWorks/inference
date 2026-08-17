//! Weights-free Mage-VAE decode parity (sc-14039).
//!
//! Everything here runs in a default `cargo test` on a fresh clone: the oracle is
//! `tests/fixtures/mage_vae_tiny.safetensors`, a tiny randomly-initialised decode path captured
//! straight from the vendored PyTorch reference by `tools/dump_mage_vae_fixture.py`. It exercises
//! the same code the published checkpoint runs — the CoD decoder (including the `AttnBlock`'s
//! replicate padding), the DiCo stack, both 8192-channel orderings, the DCT position code,
//! `SimpleMLPAdaLN`, and the unfold/fold round trip.
//!
//! The real-weights gate lives in `vae_decode_real_weights.rs` and is `#[ignore]`d.

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mage::vae::denoiser::{dct_position_table, MageVaeShape};
use mlx_gen_mage::vae::MageVae;

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mage_vae_tiny.safetensors"
    );
    Weights::from_file(path).expect("tiny Mage-VAE fixture")
}

/// The shape the fixture was dumped with, read out of the fixture itself rather than restated —
/// a mismatch fails loudly instead of silently mis-loading.
fn fixture_shape(w: &Weights) -> (MageVaeShape, i32) {
    let raw = w
        .require("fixture.shape")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let v: Vec<i32> = raw.as_slice::<i32>().to_vec();
    assert_eq!(v.len(), 12, "fixture.shape layout changed");
    (
        MageVaeShape {
            patch: v[0],
            hidden: v[1],
            hidden_x: v[2],
            in_channels: v[3],
            bottleneck: v[4],
            num_cond_blocks: v[5] as usize,
            num_mlp_blocks: v[6] as usize,
            max_freqs: v[7] as usize,
            attn_tile: v[8],
        },
        v[9],
    )
}

/// `a[i]` along axis 0, with that axis dropped.
fn row(a: &Array, i: i32) -> Array {
    a.take_axis(Array::from_int(i), 0).unwrap()
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    max(abs(subtract(&a, &b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn build(w: &Weights, shape: MageVaeShape, fold: bool) -> MageVae {
    MageVae::from_weights_with_shape(w, Dtype::Float32, fold, shape, "pipeline")
        .expect("build tiny Mage-VAE")
}

/// The end-to-end gate: a full decode of the tiny model must match the reference's.
#[test]
fn tiny_decode_matches_the_torch_reference() {
    let w = fixture();
    let (shape, latent_hw) = fixture_shape(&w);
    let vae = build(&w, shape, true);

    let latent = w.require("fixture.latent").unwrap().clone();
    assert_eq!(
        latent.shape(),
        &[1, shape.bottleneck, latent_hw, latent_hw],
        "fixture latent geometry"
    );

    let got = vae.decode(&latent).unwrap();
    let want = w.require("fixture.decoded").unwrap();
    assert_eq!(got.shape(), want.shape(), "decoded geometry");

    let err = max_abs(&got, want);
    // f32 throughout on both sides; the residual is Metal's reduced-precision matmul/conv
    // accumulation over ~30 layers. A structural bug (transposed layout, wrong ordering, missing
    // padding) diverges by orders of magnitude — see the discrimination test below.
    assert!(
        err < 2e-3,
        "tiny decode max_abs {err} vs the torch reference"
    );
}

/// The CoD decoder alone, so an `AttnBlock` / `ResnetBlock` / GroupNorm fault localises there
/// rather than showing up only as a wrong image.
///
/// The tiny latent is 6×6 against a 4×4 attention tile, so this is also the **replicate-padding**
/// gate: a zero-padded or unpadded implementation fails here.
#[test]
fn tiny_cod_decoder_matches_the_torch_reference() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let cod = mlx_gen_mage::vae::cod_decoder::CodDecoder::from_weights(
        &w,
        "pipeline.y_embedder.decoder",
        shape.attn_tile,
    )
    .unwrap();

    let latent = w
        .require("fixture.latent")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    let got = cod
        .forward(&latent)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap();
    let want = w.require("fixture.cond").unwrap();

    assert_eq!(got.shape(), want.shape(), "cond geometry");
    let err = max_abs(&got, want);
    assert!(err < 5e-4, "CoD decoder max_abs {err}");
}

/// A wrong attention tile changes the result — proof the padding/tiling assertion above
/// discriminates rather than passing on any implementation.
#[test]
fn a_wrong_attention_tile_is_rejected() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let want = w.require("fixture.cond").unwrap();
    let latent = w
        .require("fixture.latent")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();

    // 8 >= the padded 6->8 extent, so every tile becomes one whole tile: a different attention
    // neighbourhood, hence a different answer.
    let cod = mlx_gen_mage::vae::cod_decoder::CodDecoder::from_weights(
        &w,
        "pipeline.y_embedder.decoder",
        8,
    )
    .unwrap();
    let got = cod
        .forward(&latent)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap();
    let err = max_abs(&got, want);
    assert!(
        err > 1e-2,
        "tile {} vs {}: max_abs {err} — the tiling assertion does not discriminate",
        8,
        shape.attn_tile
    );
}

/// Constant-folding adaLN at `t = 0` must not change the answer.
///
/// This is the load-bearing check on the fold: it is not enough that the folded model runs, it
/// must agree with the live-MLP model built from the same weights. A fold that mis-ordered the six
/// chunks, applied one block's modulation to all of them, or folded at the wrong timestep would
/// still produce a plausible image.
#[test]
fn adaln_constant_folding_is_numerically_identical() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let latent = w.require("fixture.latent").unwrap().clone();

    let folded = build(&w, shape, true);
    let unfolded = build(&w, shape, false);
    assert!(folded.is_adaln_folded(), "fold=true must fold");
    assert!(
        folded.is_decode_folded(),
        "fold=true must fold DCT/zero RGB"
    );
    assert!(!unfolded.is_adaln_folded(), "fold=false must not fold");
    assert!(
        !unfolded.is_decode_folded(),
        "fold=false must retain the original projection"
    );

    let a = folded.decode(&latent).unwrap();
    let b = unfolded.decode(&latent).unwrap();
    let err = max_abs(&a, &b);
    assert!(
        err < 1e-5,
        "algebraically folded and unfolded decode differ by max_abs {err}"
    );
}

/// The `t = 0` timestep embedding and the six adaLN vectors each block folds, against the
/// reference's own values — so a fold that is self-consistent but wrong is still caught.
#[test]
fn adaln_folded_values_match_the_torch_reference() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let denoiser =
        mlx_gen_mage::vae::denoiser::DConvDenoiser::from_weights(&w, "pipeline", shape).unwrap();

    let t = mlx_rs::ops::zeros_dtype(&[1], Dtype::Float32).unwrap();
    let c = denoiser.conditioning(&t, Dtype::Float32).unwrap();
    let want_c = w.require("fixture.t_embed_zero").unwrap();
    let err = max_abs(&c, want_c);
    assert!(err < 1e-5, "t_embedder(0) max_abs {err}");

    // `fixture.adaln_zero` is [num_cond_blocks, 6 * hidden] — the value each block folds.
    let want = w.require("fixture.adaln_zero").unwrap();
    assert_eq!(
        want.shape(),
        &[shape.num_cond_blocks as i32, 6 * shape.hidden],
        "adaln fixture geometry"
    );

    let got = denoiser.adaln_packed(&c).unwrap();
    assert_eq!(got.len(), shape.num_cond_blocks, "one modulation per block");
    for (i, g) in got.iter().enumerate() {
        let want_row = row(want, i as i32).reshape(&[1, 6 * shape.hidden]).unwrap();
        let err = max_abs(g, &want_row);
        assert!(err < 1e-5, "block {i} adaLN modulation max_abs {err}");
    }

    // Discrimination: the six chunks are distinct, so a mis-ordered split would not match. If the
    // reference's own vectors were near-identical this test would pass on a wrong chunk order.
    let row0 = row(want, 0).reshape(&[6, shape.hidden]).unwrap();
    let spread = max_abs(&row(&row0, 0), &row(&row0, 1));
    assert!(
        spread > 1e-2,
        "shift_msa and scale_msa differ by only {spread} — the chunk-order check is vacuous"
    );
}

/// The DCT position code — pinned against the reference at both the tiny and the **published**
/// (`patch = 16`) geometry, so the real decode path's table is covered by a committed fixture.
#[test]
fn dct_position_table_matches_the_torch_reference() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);

    for (patch, key) in [
        (shape.patch, "fixture.dct_tiny"),
        (16, "fixture.dct_published"),
    ] {
        let got = dct_position_table(patch, shape.max_freqs).unwrap();
        let want = w.require(key).unwrap();
        assert_eq!(got.shape(), want.shape(), "{key} geometry");
        let err = max_abs(&got, want);
        assert!(err < 1e-5, "{key} max_abs {err}");
    }
}

/// `freqs = linspace(0, max_freqs, max_freqs)` spans 0…8 **inclusive** with a step of 8/7
/// (`mage_vae.py:196`). Reading it as `arange(8)` is the natural misreading; this proves the
/// fixture check above would catch it rather than both being wrong the same way.
#[test]
fn an_integer_frequency_ramp_would_not_match() {
    let w = fixture();
    let want = w.require("fixture.dct_published").unwrap();
    let max_freqs = 8usize;

    // Rebuild with `arange(max_freqs)` instead of `linspace(0, max_freqs, max_freqs)`.
    let p = 16usize;
    let denom = (p - 1) as f32;
    let pos: Vec<f32> = (0..p).map(|i| i as f32 / denom).collect();
    let freqs: Vec<f32> = (0..max_freqs).map(|i| i as f32).collect();
    let mut data = Vec::with_capacity(p * p * max_freqs * max_freqs);
    for i in 0..p {
        for j in 0..p {
            for &fx in freqs.iter() {
                for &fy in freqs.iter() {
                    let coeff = 1.0 / (1.0 + fx * fy);
                    data.push(
                        (pos[j] * fx * std::f32::consts::PI).cos()
                            * (pos[i] * fy * std::f32::consts::PI).cos()
                            * coeff,
                    );
                }
            }
        }
    }
    let wrong = Array::from_slice(&data, &[1, (p * p) as i32, (max_freqs * max_freqs) as i32]);
    let err = max_abs(&wrong, want);
    assert!(
        err > 0.1,
        "an integer frequency ramp differs by only {err} — the DCT check does not discriminate"
    );
}

/// The published shape must be exactly what the epic's ground truth records.
#[test]
fn published_shape_is_the_documented_one() {
    let s = MageVaeShape::PUBLISHED;
    assert_eq!(s.patch, 16);
    assert_eq!(s.hidden, 384);
    assert_eq!(s.hidden_x, 32);
    assert_eq!(s.in_channels, 3);
    assert_eq!(s.bottleneck, 128);
    assert_eq!(s.num_cond_blocks, 21);
    assert_eq!(s.num_mlp_blocks, 3);
    assert_eq!(s.max_freqs, 8);
    assert_eq!(s.attn_tile, 32);
}

/// A latent whose channel count or rank is wrong is a typed error, not a panic or a wrong image.
#[test]
fn decode_rejects_a_malformed_latent() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let vae = build(&w, shape, true);

    let wrong_channels =
        mlx_rs::ops::zeros_dtype(&[1, shape.bottleneck + 1, 4, 4], Dtype::Float32).unwrap();
    assert!(vae.decode(&wrong_channels).is_err(), "wrong channel count");

    let wrong_rank = mlx_rs::ops::zeros_dtype(&[shape.bottleneck, 4, 4], Dtype::Float32).unwrap();
    assert!(vae.decode(&wrong_rank).is_err(), "wrong rank");
}

/// **The non-square + asymmetric-padding gate.** Two distinct faults hide behind square inputs:
///
/// 1. a transposed height/width — `[b, hl, wl, ...]` read as `[b, wl, hl, ...]`;
/// 2. a swapped `pad_h`/`pad_w` in the `AttnBlock` tiling.
///
/// The fixture's second latent is `6 × 9`, which the 4×4 tiling pads to `8 × 12` — `pad_h = 2`
/// but `pad_w = 3`. Both faults fail here. **Non-squareness alone is not enough for (2):** the
/// first version of this test used `6 × 10`, which pads `(2, 2)`, and a `pad_h`/`pad_w` swap
/// passed it — as it passed every real-weights geometry too, since 256² and 768×1280 pad `(16,16)`
/// and 1024²/2048/512×2048 pad nothing. `768x1152` is the only real-weights geometry that catches
/// it. The epic's native-resolution range admits aspects up to 4:1, so this is supported surface.
#[test]
fn non_square_decode_matches_the_torch_reference() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let vae = build(&w, shape, true);

    let latent = w.require("fixture.latent_ns").unwrap().clone();
    let sh = latent.shape();
    assert_ne!(sh[2], sh[3], "the non-square fixture latent is square");

    let got = vae.decode(&latent).unwrap();
    let want = w.require("fixture.decoded_ns").unwrap();
    assert_eq!(got.shape(), want.shape(), "non-square decoded geometry");
    assert_eq!(
        got.shape(),
        &[
            1,
            shape.in_channels,
            sh[2] * shape.patch,
            sh[3] * shape.patch
        ],
        "decode must scale height and width independently"
    );

    let err = max_abs(&got, want);
    assert!(err < 2e-3, "non-square decode max_abs {err}");
}

/// The CoD decoder alone on the non-square latent, so an asymmetric-padding fault in `AttnBlock`
/// localises there rather than surfacing only as a wrong image.
#[test]
fn non_square_cod_decoder_matches_the_torch_reference() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let cod = mlx_gen_mage::vae::cod_decoder::CodDecoder::from_weights(
        &w,
        "pipeline.y_embedder.decoder",
        shape.attn_tile,
    )
    .unwrap();

    let latent = w
        .require("fixture.latent_ns")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    let got = cod
        .forward(&latent)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap();
    let want = w.require("fixture.cond_ns").unwrap();

    assert_eq!(got.shape(), want.shape(), "non-square cond geometry");
    let err = max_abs(&got, want);
    assert!(err < 5e-4, "non-square CoD decoder max_abs {err}");
}

/// sc-19753 — the **bounded** decode must reproduce the dense decode.
///
/// Mage's codec splits cleanly in two: everything with a cross-position dependence (the CoD
/// decoder's `GroupNorm(32)`s and its 32×32-window attention, and each DiCo block's
/// whole-extent squeeze-and-excite pool) is latent-resolution, and the per-latent-pixel tail is
/// the output-resolution half. `decode_tiled` now runs the first half whole and tiles only the
/// second, so the bounded result is the dense result — not an approximation of it.
///
/// Verified by mutation, not assumed: tiling the whole `decode` (this method's previous shape)
/// moves the same comparison to max_abs **1.655e-1** against the 2e-4 bound below, because every
/// crop then normalizes, attends and gates against itself.
#[test]
fn tiled_decode_reproduces_the_dense_decode() {
    let w = fixture();
    let (shape, latent_hw) = fixture_shape(&w);
    let latent = w.require("fixture.latent").unwrap().clone();

    // Four latent pixels per tile: `needs_tiling` compares the latent extent against
    // `tile_px / patch`, so this genuinely splits the fixture's 6×6 grid on both axes.
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(4 * shape.patch, 0);
    let geometry = mlx_gen::tiling::VaeTiling {
        spatial_scale: shape.patch,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: shape.hidden_x,
    };
    assert!(
        cfg.needs_tiling(geometry, 1, latent_hw, latent_hw),
        "the fixture geometry must actually tile, else this proves nothing"
    );
    let plan = cfg.plan(geometry, 1, latent_hw, latent_hw);
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "the bounded plan must split both spatial axes, got {}x{}",
        plan.h.len(),
        plan.w.len()
    );

    // Both fold states: folded is production, unfolded exercises the zero-RGB per-tile path.
    for fold in [true, false] {
        let vae = build(&w, shape, fold);
        let dense = vae.decode(&latent).unwrap();
        let bounded = vae.decode_tiled(&latent, &cfg, None).unwrap();
        assert_eq!(bounded.shape(), dense.shape(), "fold={fold} geometry");
        let err = max_abs(&dense, &bounded);
        assert!(
            err < 2e-4,
            "fold={fold}: bounded decode diverged from dense by max_abs {err}"
        );
    }
}

/// A tiling request too wide to split this latent must fall through to the exact single-pass
/// decode rather than assembling a one-tile plan.
#[test]
fn an_untiled_request_falls_through_to_the_dense_decode() {
    let w = fixture();
    let (shape, latent_hw) = fixture_shape(&w);
    let latent = w.require("fixture.latent").unwrap().clone();
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(64 * shape.patch, 0);
    let geometry = mlx_gen::tiling::VaeTiling {
        spatial_scale: shape.patch,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: shape.hidden_x,
    };
    assert!(!cfg.needs_tiling(geometry, 1, latent_hw, latent_hw));

    let vae = build(&w, shape, true);
    let dense = vae.decode(&latent).unwrap();
    let passthrough = vae.decode_tiled(&latent, &cfg, None).unwrap();
    assert_eq!(max_abs(&dense, &passthrough), 0.0);
}

/// A pre-tripped cancel is observed before any tensor work.
#[test]
fn tiled_decode_honors_a_pretripped_cancel() {
    let w = fixture();
    let (shape, _) = fixture_shape(&w);
    let latent = w.require("fixture.latent").unwrap().clone();
    let vae = build(&w, shape, true);
    let cancel = mlx_gen::CancelFlag::new();
    cancel.cancel();
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(4 * shape.patch, 0);
    let result = vae.decode_tiled(&latent, &cfg, Some(&cancel));
    assert!(matches!(result, Err(mlx_gen::Error::Canceled)));
}
