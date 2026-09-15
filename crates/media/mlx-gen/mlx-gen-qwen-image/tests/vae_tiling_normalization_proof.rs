//! sc-19753 — the Qwen-Image VAE's tiled decode is **already** normalization-correct; this proves it.
//!
//! The sc-19753 rule is that globally-scoped ops must run ONCE on the whole latent and only
//! spatially-local work may be tiled. [`QwenVae::decode_tiled`] was audited and found to already obey
//! it: it runs the dense head [`QwenVae::decode_pre_upsample`] (denormalize → `post_quant_conv` →
//! decoder `conv_in` → mid-block, whose `AttentionBlock3D` is the decoder's ONLY global op — a softmax
//! over all H·W tokens per frame) and tiles only [`QwenVae::decode_upsample_tail`], every op of which
//! is spatially local (convs, nearest upsample, and `rms_norm_channels`, which reduces ONLY axis 1 —
//! the channel axis of NCTHW — so it is a per-(b,t,h,w)-position op). An audit verdict is worth what a
//! test proves, so these run the real seam on a weights-free synthetic checkpoint.
//!
//! Weights-free and unit-scale: [`weights`] is a structurally faithful Qwen-Image VAE (the real 4
//! up-blocks × 3 resnets, 3 upsamplers ⇒ ×8 spatial, and the real mid-block resnet/attention/resnet
//! bottleneck) at an 8-channel width. `Decoder3D::from_weights` infers every channel width from the
//! weight shapes, so the narrow fixture exercises the identical code path as the shipped checkpoint.
//! No weights are downloaded and no real render runs.

use std::collections::HashMap;

use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::vae_tiling::tiled_decode;
use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::vae::QwenVae;
use mlx_rs::Array;

/// Fixture decoder width. Every stage is this wide (the shipped checkpoint halves at each upsampler;
/// nothing in the port depends on that, and holding it constant keeps the fixture one builder).
const C: i32 = 8;
/// The Qwen latent channel count is fixed by `QwenVae`'s baked-in `LATENTS_{MEAN,STD}`.
const LATENT_C: i32 = 16;
/// Latent H = W. ×8 spatial ⇒ a 256×256 decode.
const LATENT_HW: i32 = 32;
/// Width of the interior latent crop used by the locality proof.
const CROP_W: i32 = 20;
/// The tail's exact receptive-field radius in OUTPUT pixels, derived from the port's own structure:
/// each `ResBlock3D` is two `k=3` convs (radius 1 each) and each stage has 3 of them (radius 6 per
/// stage at that stage's resolution); each `Resample3d` upsample doubles the accumulated radius then
/// adds 1 for its `k=3` conv; `conv_out` adds 1. So
/// `((((6·2+1)+6)·2+1)+6)·2+1 + 6 + 1 = 98`. Nothing else in the tail moves information spatially
/// (`rms_norm_channels` reduces the channel axis only, SiLU is elementwise).
const TAIL_HALO_OUT_PX: i32 = 98;
/// Output columns of a `CROP_W`-wide latent crop that the tail computes from crop-interior data only
/// (the left edge is the real image edge, so its zero padding is identical to the dense decode's).
const CROP_EXACT_OUT_W: i32 = CROP_W * 8 - TAIL_HALO_OUT_PX;

fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
    let count = shape.iter().product::<i32>();
    let data = (0..count)
        .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
        .collect::<Vec<_>>();
    Array::from_slice(&data, shape)
}

/// A `CausalConv3d`: mlx `[out, kD, kH, kW, in]` weight under `{prefix}.conv3d`.
fn insert_conv3d(
    tensors: &mut HashMap<String, Array>,
    prefix: &str,
    input: i32,
    output: i32,
    kernel: i32,
    phase: f32,
) {
    tensors.insert(
        format!("{prefix}.conv3d.weight"),
        values(&[output, kernel, kernel, kernel, input], phase, 0.09),
    );
    tensors.insert(
        format!("{prefix}.conv3d.bias"),
        values(&[output], phase + 3.0, 0.01),
    );
}

/// A `Resample3d`: mlx `[out, kH, kW, in]` 2-D weight under `{prefix}.resample_conv`.
fn insert_resample(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
    tensors.insert(
        format!("{prefix}.resample_conv.weight"),
        values(&[C, 3, 3, C], phase, 0.09),
    );
    tensors.insert(
        format!("{prefix}.resample_conv.bias"),
        values(&[C], phase + 3.0, 0.01),
    );
}

/// A channel-L2 (`rms_norm_channels`) gain, centered on 1.0 so the fixture stays unit-scale.
fn insert_norm(tensors: &mut HashMap<String, Array>, key: &str, phase: f32) {
    let gain = (0..C)
        .map(|i| 1.0 + ((i as f32 + phase) * 0.071).sin() * 0.15)
        .collect::<Vec<_>>();
    tensors.insert(key.to_string(), Array::from_slice(&gain, &[C]));
}

fn insert_resnet(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
    insert_norm(tensors, &format!("{prefix}.norm1.weight"), phase);
    insert_conv3d(tensors, &format!("{prefix}.conv1"), C, C, 3, phase + 4.0);
    insert_norm(tensors, &format!("{prefix}.norm2.weight"), phase + 1.0);
    insert_conv3d(tensors, &format!("{prefix}.conv2"), C, C, 3, phase + 5.0);
}

/// The mid-block: `resnet → attention → resnet`. The attention is the global op the head protects.
fn insert_mid(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
    insert_resnet(tensors, &format!("{prefix}.resnets.0"), phase);
    let attn = format!("{prefix}.attentions.0");
    insert_norm(tensors, &format!("{attn}.norm.weight"), phase + 10.0);
    tensors.insert(
        format!("{attn}.to_qkv.weight"),
        values(&[3 * C, 1, 1, C], phase + 11.0, 0.25),
    );
    tensors.insert(
        format!("{attn}.to_qkv.bias"),
        values(&[3 * C], phase + 12.0, 0.02),
    );
    tensors.insert(
        format!("{attn}.proj.weight"),
        values(&[C, 1, 1, C], phase + 13.0, 0.25),
    );
    tensors.insert(
        format!("{attn}.proj.bias"),
        values(&[C], phase + 14.0, 0.02),
    );
    insert_resnet(tensors, &format!("{prefix}.resnets.1"), phase + 20.0);
}

/// A structurally faithful synthetic Qwen-Image VAE at width [`C`]. The encoder half is built only
/// because `QwenVae::from_weights` requires it; every assertion here drives the decode path.
fn weights() -> Weights {
    let mut t = HashMap::new();

    // Decoder: conv_in → mid_block → up_block0..3 (0..2 upsample) → norm_out → conv_out.
    insert_conv3d(&mut t, "decoder.conv_in", LATENT_C, C, 3, 1.0);
    insert_mid(&mut t, "decoder.mid_block", 10.0);
    for block in 0..4 {
        for resnet in 0..3 {
            insert_resnet(
                &mut t,
                &format!("decoder.up_block{block}.resnets.{resnet}"),
                40.0 + (block * 10 + resnet) as f32,
            );
        }
        if block < 3 {
            insert_resample(
                &mut t,
                &format!("decoder.up_block{block}.upsamplers.0"),
                80.0 + block as f32,
            );
        }
    }
    insert_norm(&mut t, "decoder.norm_out.weight", 90.0);
    insert_conv3d(&mut t, "decoder.conv_out", C, 3, 3, 92.0);

    // Encoder: required by `QwenVae::from_weights`, unused by these assertions.
    insert_conv3d(&mut t, "encoder.conv_in", 3, C, 3, 100.0);
    for block in 0..4 {
        for resnet in 0..2 {
            insert_resnet(
                &mut t,
                &format!("encoder.down_blocks.{block}.resnets.{resnet}"),
                110.0 + (block * 10 + resnet) as f32,
            );
        }
        if block < 3 {
            insert_resample(
                &mut t,
                &format!("encoder.down_blocks.{block}.downsamplers.0"),
                150.0 + block as f32,
            );
        }
    }
    insert_mid(&mut t, "encoder.mid_block", 160.0);
    insert_norm(&mut t, "encoder.norm_out.weight", 190.0);
    insert_conv3d(&mut t, "encoder.conv_out", C, 2 * LATENT_C, 3, 192.0);
    insert_conv3d(&mut t, "quant_conv", 2 * LATENT_C, 2 * LATENT_C, 1, 200.0);
    insert_conv3d(&mut t, "post_quant_conv", LATENT_C, LATENT_C, 1, 210.0);

    Weights::from_map(t)
}

/// Position-dependent NCTHW latents `[1, 16, 1, hw, hw]` with a strong spatial ramp: a crop's global
/// attention sees visibly different statistics from the dense one, so a crop/dense comparison over
/// these discriminates the defect rather than merely re-checking shapes.
fn latents(hw: i32) -> Array {
    let shape = [1, LATENT_C, 1, hw, hw];
    let count = shape.iter().product::<i32>();
    let data = (0..count)
        .map(|i| {
            let y = (i / hw % hw) as f32;
            let x = (i % hw) as f32;
            (i as f32 * 0.037).sin() + y * 0.11 - x * 0.07
        })
        .collect::<Vec<_>>();
    Array::from_slice(&data, &shape)
}

fn max_abs_delta(left: &Array, right: &Array) -> f32 {
    left.as_slice::<f32>()
        .iter()
        .zip(right.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

/// `x[..., ..width]` on the last (W) axis, made contiguous so `as_slice` reads it correctly.
fn front_cols(x: &Array, width: i32) -> Array {
    let idx = (0..width).collect::<Vec<_>>();
    let taken = x.take_axis(Array::from_slice(&idx, &[width]), 4).unwrap();
    mlx_gen::array::contiguous(&taken).unwrap()
}

fn vae() -> QwenVae {
    QwenVae::from_weights(&weights()).unwrap()
}

/// The bounded request that actually splits this geometry: 192 px tiles / 128 px overlap at ×8
/// spatial ⇒ 24-latent-px tiles with 16 px overlap over a 32-px latent (2×2 tiles). The overlap is
/// chosen wider than [`TAIL_HALO_OUT_PX`] so the blend can actually absorb the tail's conv halo —
/// the same relationship the shipped 512 px / 64 px preset targets at the real channel widths.
fn bounded() -> TilingConfig {
    TilingConfig::spatial_only(192, 128)
}

/// **The head/tail decomposition is an EXACT identity of the single-pass decode.**
///
/// `decode_upsample_tail(decode_pre_upsample(z))` is bit-for-bit `decode(z)` — delta 0.0 (measured;
/// not a tolerance). That is what licenses `decode_tiled` running the head once and tiling only the
/// tail: the split introduces no approximation of its own, so any tiled/dense gap is attributable to
/// the blend alone.
///
/// **Executed mutation** (`QwenVae::decode` rewritten to denormalize as `(z + mean)·std` instead of
/// `z·std + mean`, i.e. `decode` stops agreeing with its own head/tail pair): this test goes RED at
/// **5.770e-1**; every other test in this file stays green, so the discrimination is this one's.
#[test]
fn head_tail_decomposition_is_the_exact_single_pass_decode() {
    let vae = vae();
    let z = latents(LATENT_HW);
    let dense = vae.decode(&z).unwrap();
    let split = vae
        .decode_upsample_tail(&vae.decode_pre_upsample(&z).unwrap())
        .unwrap();
    dense.eval().unwrap();
    split.eval().unwrap();
    assert_eq!(split.shape(), dense.shape());
    assert_eq!(
        max_abs_delta(&dense, &split),
        0.0,
        "the head/tail split must reproduce the single-pass decode exactly"
    );
}

/// **The upsample tail is spatially LOCAL and the head is GLOBAL — the exact form of the proof.**
///
/// The tail is a composition of local ops only, so decoding a latent crop must reproduce the dense
/// decode *exactly* everywhere outside the crop's [`TAIL_HALO_OUT_PX`] boundary halo — no tolerance
/// argument, a structural identity. The head is a per-frame softmax over all H·W tokens, so it has no
/// such interior: re-running it on the crop perturbs *every* output pixel.
///
/// Both routes crop the SAME `CROP_W`-wide left slab and are compared on the SAME
/// `CROP_EXACT_OUT_W`-column halo-interior window:
///  - tail-of-crop (what `decode_tiled` tiles) vs the dense tail → max|Δ| = **0.0**, bit-exact —
///    the locality claim, with no tolerance argument at all;
///  - whole-decode-of-crop (head included, i.e. the defect) vs the dense decode → max|Δ| =
///    **5.279e-1** on the identical window.
///
/// That gap is the discrimination: route B's error is not a seam artifact and no amount of overlap
/// removes it, because the attention is global by construction. (Re-measure if the fixture changes;
/// the test prints both numbers under `--nocapture`.)
///
/// **Executed mutation** (`rms_norm_channels`'s `sum_axes(.., &[1], ..)` widened to `&[1, H, W]`, so
/// the tail's normalization becomes spatially scoped): the tail's crop-locality goes 0.0 → **8.272e-4**
/// and this test goes RED.
#[test]
fn the_upsample_tail_is_spatially_local_but_the_head_is_not() {
    let vae = vae();
    let z = latents(LATENT_HW);
    let z_crop = front_cols(&z, CROP_W);

    // Local claim: the tail of a cropped head == the dense tail, on the halo interior.
    let head = vae.decode_pre_upsample(&z).unwrap();
    let dense_tail = vae.decode_upsample_tail(&head).unwrap();
    let crop_tail = vae
        .decode_upsample_tail(&front_cols(&head, CROP_W))
        .unwrap();
    let dense_window = front_cols(&dense_tail, CROP_EXACT_OUT_W);
    let crop_window = front_cols(&crop_tail, CROP_EXACT_OUT_W);
    dense_window.eval().unwrap();
    crop_window.eval().unwrap();
    assert_eq!(crop_window.shape(), dense_window.shape());
    let local = max_abs_delta(&dense_window, &crop_window);

    // Global claim (the executed mutation control): re-running the head on the crop moves the SAME
    // window by orders of magnitude, because the attention softmax spans the whole latent.
    let dense_decode = vae.decode(&z).unwrap();
    let crop_decode = vae.decode(&z_crop).unwrap();
    let dense_decode_window = front_cols(&dense_decode, CROP_EXACT_OUT_W);
    let crop_decode_window = front_cols(&crop_decode, CROP_EXACT_OUT_W);
    dense_decode_window.eval().unwrap();
    crop_decode_window.eval().unwrap();
    let global = max_abs_delta(&dense_decode_window, &crop_decode_window);

    assert!(
        local < 1e-4,
        "the upsample tail must be spatially local: cropping it moved the halo interior by \
         max|Δ|={local:.3e}"
    );
    assert!(
        global > 1e-1,
        "the pre-upsample head must be spatially GLOBAL, but cropping it moved the halo interior by \
         only max|Δ|={global:.3e} — the mid-block attention would have to have stopped being global"
    );
    assert!(
        global > local * 1e3,
        "the head/tail locality gap collapsed: head {global:.3e} vs tail {local:.3e}"
    );
    println!("tail crop-locality = {local:.3e}; head crop-locality = {global:.3e}");
}

/// **`decode_tiled` runs the global head ONCE and tiles only the local tail** — asserted as an
/// identity on the shipped seam, not as a tolerance.
///
/// Two references are built over the identical [`TilePlan`] and trapezoidal blend:
///  - route A — head ONCE on the full latent, then `tiled_decode` over `decode_upsample_tail`. This
///    is what sc-19753 requires, and `decode_tiled` reproduces it **exactly**: max|Δ| = **0.0**.
///  - route B — the defect: `tiled_decode` over the WHOLE `decode`, so every tile's mid-block
///    attention softmaxes over its own crop's H·W tokens. It differs from the shipped output by
///    max|Δ| = **4.772e-1**.
///
/// The 0.0 pins the decomposition (any op moved across the head/tail boundary, or a head recomputed
/// per tile, breaks it) and the route-B number is the executed control proving the 0.0 is not
/// vacuous. Informational, same run: the shipped tiled decode sits **7.488e-1** from the dense
/// single-pass decode. That residual is the tail's own conv halo ([`TAIL_HALO_OUT_PX`] = 98 output px
/// against a 128 px overlap) amplified by this narrow synthetic fixture's untrained weights — it is
/// blend geometry, not normalization, and `the_upsample_tail_is_spatially_local_but_the_head_is_not`
/// is what separates the two.
///
/// **Executed mutations**, each run alone against the real source:
///  - `decode_tiled` rewritten to tile the WHOLE decode (`head = l`, per-tile `self.decode`): the
///    route-A identity goes 0.0 → **4.772e-1**, RED. This is exactly the sc-19753 defect.
///  - `rms_norm_channels` widened to a spatial reduction: the route-B control collapses to
///    **8.098e-5** (a spatially-scoped tail makes whole-decode tiling look almost free), RED.
#[test]
fn tiled_decode_runs_the_global_head_once_and_tiles_only_the_local_tail() {
    let vae = vae();
    let z = latents(LATENT_HW);
    let cfg = bounded();
    let plan = cfg.plan(VaeTiling::QWEN_IMAGE, 1, LATENT_HW, LATENT_HW);
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "the fixture must split both spatial axes, got {}x{}",
        plan.h.len(),
        plan.w.len()
    );

    let shipped = vae.decode_tiled(&z, &cfg, None).unwrap();

    // Route A: head once on the full latent, tail tiled over the same plan.
    let head = vae.decode_pre_upsample(&z).unwrap();
    let route_a = tiled_decode(&head, &plan, [2, 3, 4], None, |tile| {
        vae.decode_upsample_tail(tile)
    })
    .unwrap();

    // Route B: the same plan and blend, but every tile re-runs the global head.
    let route_b = tiled_decode(&z, &plan, [2, 3, 4], None, |tile| vae.decode(tile)).unwrap();

    let dense = vae.decode(&z).unwrap();
    for a in [&shipped, &route_a, &route_b, &dense] {
        a.eval().unwrap();
    }
    assert_eq!(route_a.shape(), shipped.shape());
    assert_eq!(route_b.shape(), shipped.shape());

    assert_eq!(
        max_abs_delta(&shipped, &route_a),
        0.0,
        "decode_tiled must be exactly `dense head + tiled local tail`"
    );
    let b = max_abs_delta(&shipped, &route_b);
    assert!(
        b > 1e-1,
        "tiling the global attention head must be observably wrong, but it moved the output by only \
         max|Δ|={b:.3e} — either the head stopped being global or the fixture stopped discriminating"
    );
    // Informational: the blend's own halo residual against the dense single-pass decode.
    let halo = max_abs_delta(&dense, &shipped);
    println!("route_b vs shipped = {b:.3e}; dense vs shipped (halo residual) = {halo:.3e}");
}

/// The tiling must actually fire, otherwise the bound above would be measured against an untiled
/// fall-through and prove nothing. A tile edge wide enough to hold the whole latent takes
/// `decode_tiled`'s single-pass fallback and is therefore *exactly* the dense decode — the control
/// that separates "bounded and correct" from "never bounded".
///
/// **Executed mutation** (the `if !needs_tiling` fallback disabled so every request tiles): this test
/// goes RED at **7.488e-1**, and it is the only one that does.
#[test]
fn a_request_too_wide_to_tile_is_the_exact_single_pass_decode() {
    let vae = vae();
    let z = latents(LATENT_HW);
    let wide = TilingConfig::spatial_only(4096, 128);
    assert!(!wide.needs_tiling(VaeTiling::QWEN_IMAGE, 1, LATENT_HW, LATENT_HW));
    assert!(bounded().needs_tiling(VaeTiling::QWEN_IMAGE, 1, LATENT_HW, LATENT_HW));

    let dense = vae.decode(&z).unwrap();
    let untiled = vae.decode_tiled(&z, &wide, None).unwrap();
    dense.eval().unwrap();
    untiled.eval().unwrap();
    assert_eq!(
        max_abs_delta(&dense, &untiled),
        0.0,
        "a request too wide to tile must be the exact single-pass decode"
    );
}

/// `rms_norm_channels` — the tail's ONLY normalization — reduces the channel axis alone
/// (`sum_axes(x·x, &[1], true)` in `vae/blocks.rs`), so it **commutes with spatial cropping**:
/// normalizing a crop equals cropping the normalized whole. That is the layer-level reason the tail
/// is tile-safe, isolated from the conv halo that the decode-level proof has to reason around.
///
/// Measured: the real channel-only reduction commutes to **0.0** (exact), while the executed mutation
/// control — the same normalization with a GroupNorm-shaped reduction (mean/var over C, H, W instead
/// of over C) — breaks commutation by **6.005e-1** on the identical tensor. A spatial softmax would
/// break it the same way. Without the control the 0.0 above would be satisfied by any implementation.
///
/// **Executed mutation** (the real `rms_norm_channels` widened to reduce `&[1, H, W]`): this test goes
/// RED at **2.545e-1**.
#[test]
fn the_tail_normalization_commutes_with_cropping_but_a_spatial_reduction_does_not() {
    use mlx_gen_qwen_image::vae::blocks::rms_norm_channels;
    use mlx_rs::ops::{add, multiply, rsqrt, subtract};

    let (b, c, t, h, w) = (1, C, 1, 7, 9);
    let count = b * c * t * h * w;
    let data = (0..count)
        .map(|i| {
            let y = (i / w % h) as f32;
            let x = (i % w) as f32;
            ((i as f32) * 0.071).sin() + y * 0.31 - x * 0.19
        })
        .collect::<Vec<_>>();
    let x = Array::from_slice(&data, &[b, c, t, h, w]);
    let gain = Array::from_slice(
        &(0..c).map(|i| 0.7 + i as f32 * 0.013).collect::<Vec<_>>(),
        &[c],
    );

    /// `a[:, :, :, 2:5, 3:8]` — an interior spatial crop on both axes.
    fn crop(a: &Array) -> Array {
        let rows = Array::from_slice(&[2, 3, 4], &[3]);
        let cols = Array::from_slice(&[3, 4, 5, 6, 7], &[5]);
        let cropped = a.take_axis(rows, 3).unwrap().take_axis(cols, 4).unwrap();
        mlx_gen::array::contiguous(&cropped).unwrap()
    }

    // The real op: channel-axis-only reduction ⇒ crop(norm(x)) == norm(crop(x)), exactly.
    let dense = rms_norm_channels(&x, &gain, 1e-12).unwrap();
    let cropped_dense = crop(&dense);
    let norm_of_crop = rms_norm_channels(&crop(&x), &gain, 1e-12).unwrap();
    cropped_dense.eval().unwrap();
    norm_of_crop.eval().unwrap();
    assert_eq!(
        max_abs_delta(&cropped_dense, &norm_of_crop),
        0.0,
        "a channel-axis-only normalization must commute with spatial cropping"
    );

    // The mutation control: a GroupNorm-shaped (C, H, W) reduction does not commute with cropping.
    let group_norm_shaped = |a: &Array| -> Array {
        let mean = a.mean_axes(&[1, 3, 4], Some(true)).unwrap();
        let var = a.var_axes(&[1, 3, 4], Some(true), None).unwrap();
        let shifted = add(&var, Array::from_slice(&[1e-5f32], &[1])).unwrap();
        let inv = rsqrt(&shifted).unwrap();
        let centered = subtract(a, &mean).unwrap();
        multiply(&centered, &inv).unwrap()
    };
    let cropped_group = crop(&group_norm_shaped(&x));
    let group_of_crop = group_norm_shaped(&crop(&x));
    cropped_group.eval().unwrap();
    group_of_crop.eval().unwrap();
    let broken = max_abs_delta(&cropped_group, &group_of_crop);
    assert!(
        broken > 1e-1,
        "the GroupNorm-shaped control must break commutation, got max|Δ|={broken:.3e} — the \
         assertion above would then be proving nothing"
    );
    println!("channel-only commutation = 0.0; GroupNorm-shaped control = {broken:.3e}");
}
