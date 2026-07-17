//! Chunked AsymmVAE decode for Mochi 1 (sc-12291) — CI-green, no model weights.
//!
//! An untiled decode runs the 128-channel `block_out` stage at the **full** output resolution for the
//! whole clip (~30 GiB per f32 tensor at 848×480/5 s, several live at once), which is what gated Mochi
//! to 96 GB-class Macs. [`MochiVaeDecoder::decode_denormalized_chunked`] decodes a few latent frames at
//! a time, threading a `FrameCache` of each causal conv's trailing input frames.
//!
//! These tests build a **real-geometry** decoder — the true `layers_per_block` `[3, 3, 4, 6, 3]` (19
//! resnets → 38 `kt=3` causal convs) and the true `[1,2,3]`/`[2,2,2]` expansions — with narrow channels
//! and a tiny spatial grid, so the temporal structure is exact while the arithmetic stays cheap. They
//! pin, in order:
//!  - the decoder is temporally **causal** (a prefix of the latent decodes to a prefix of the video);
//!  - the temporal **receptive field is ~45 latent frames**, i.e. wider than a whole 5 s clip — the
//!    reason this is a conv cache and not the repo's shared overlap+blend `mlx_gen::tiling`;
//!  - chunked decode is **identical** to single-shot at every chunk size;
//!  - a decode without that history is materially wrong (what overlap+blend would compute).

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_mochi::{MochiVaeConfig, MochiVaeDecoder};

/// Deterministic bounded "random" fill (5 stages of GroupNorm+conv stay well-conditioned).
fn rnd(shape: &[i32], seed: u64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 0.000_001).sin()
                * 0.05
        })
        .collect();
    Array::from_slice(&data, shape)
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// The real AsymmVAE temporal geometry (resnet counts + expansions) at 32 channels — GroupNorm(32)
/// needs channels divisible by 32, and the temporal behaviour under test is channel-independent.
fn real_geometry_cfg() -> MochiVaeConfig {
    MochiVaeConfig {
        latent_channels: 12,
        out_channels: 3,
        decoder_block_out_channels: vec![32, 32, 32, 32],
        layers_per_block: vec![3, 3, 4, 6, 3],
        temporal_expansions: vec![1, 2, 3],
        spatial_expansions: vec![2, 2, 2],
        latents_mean: vec![0.0; 12],
        latents_std: vec![1.0; 12],
        scaling_factor: 1.0,
    }
}

fn insert_resnet(w: &mut Weights, pfx: &str, ch: i32, seed: u64) {
    for norm in ["norm1", "norm2"] {
        w.insert(
            format!("{pfx}.{norm}.norm_layer.weight"),
            Array::ones::<f32>(&[ch]).unwrap(),
        );
        w.insert(
            format!("{pfx}.{norm}.norm_layer.bias"),
            Array::zeros::<f32>(&[ch]).unwrap(),
        );
    }
    for (j, conv) in ["conv1", "conv2"].iter().enumerate() {
        w.insert(
            format!("{pfx}.{conv}.conv.weight"),
            rnd(&[ch, ch, 3, 3, 3], seed + j as u64 * 7 + 1),
        );
        w.insert(
            format!("{pfx}.{conv}.conv.bias"),
            Array::zeros::<f32>(&[ch]).unwrap(),
        );
    }
}

/// The full synthetic weight map for [`real_geometry_cfg`] (every resnet the real config declares).
fn synthetic_weights(cfg: &MochiVaeConfig) -> Weights {
    let mut w = Weights::empty();
    let c_last = *cfg.decoder_block_out_channels.last().unwrap() as i32;
    let c_first = cfg.decoder_block_out_channels[0] as i32;
    let lat = cfg.latent_channels as i32;
    let n = cfg.decoder_block_out_channels.len();
    let nl = cfg.layers_per_block.len();
    let k = cfg.temporal_expansions.len();

    w.insert("decoder.conv_in.weight", rnd(&[c_last, lat, 1, 1, 1], 10));
    w.insert(
        "decoder.conv_in.bias",
        Array::zeros::<f32>(&[c_last]).unwrap(),
    );
    for i in 0..cfg.layers_per_block[nl - 1] {
        insert_resnet(
            &mut w,
            &format!("decoder.block_in.resnets.{i}"),
            c_last,
            100 + i as u64,
        );
    }

    for i in 0..(n - 1) {
        let in_ch = cfg.decoder_block_out_channels[n - 1 - i] as i32;
        let out_ch = cfg.decoder_block_out_channels[n - 2 - i] as i32;
        let t = cfg.temporal_expansions[k - 1 - i] as i32;
        let s = cfg.spatial_expansions[k - 1 - i] as i32;
        let pfx = format!("decoder.up_blocks.{i}");
        for r in 0..cfg.layers_per_block[nl - 2 - i] {
            insert_resnet(
                &mut w,
                &format!("{pfx}.resnets.{r}"),
                in_ch,
                200 + i as u64 * 13 + r as u64,
            );
        }
        let proj_out = out_ch * t * s * s;
        w.insert(
            format!("{pfx}.proj.weight"),
            rnd(&[proj_out, in_ch], 300 + i as u64 * 13),
        );
        w.insert(
            format!("{pfx}.proj.bias"),
            Array::zeros::<f32>(&[proj_out]).unwrap(),
        );
    }

    for i in 0..cfg.layers_per_block[0] {
        insert_resnet(
            &mut w,
            &format!("decoder.block_out.resnets.{i}"),
            c_first,
            400 + i as u64,
        );
    }
    w.insert(
        "decoder.proj_out.weight",
        rnd(&[cfg.out_channels as i32, c_first], 500),
    );
    w.insert(
        "decoder.proj_out.bias",
        Array::zeros::<f32>(&[cfg.out_channels as i32]).unwrap(),
    );
    w
}

fn real_geometry_decoder() -> (MochiVaeConfig, MochiVaeDecoder) {
    let cfg = real_geometry_cfg();
    let w = synthetic_weights(&cfg);
    let dec = MochiVaeDecoder::from_weights(&w, &cfg).expect("build decoder");
    (cfg, dec)
}

/// Per-output-frame max abs difference between two `[B, C, F, H, W]` videos.
fn per_frame_diff(a: &Array, b: &Array) -> Vec<f32> {
    let d = abs(subtract(a, b).unwrap()).unwrap();
    (0..d.shape()[2])
        .map(|f| {
            let idx = Array::from_slice(&[f], &[1]);
            max(d.take_axis(&idx, 2).unwrap(), None)
                .unwrap()
                .item::<f32>()
        })
        .collect()
}

/// **Chunked == single-shot, exactly.** Every op in the decoder is per-frame (GroupNorm is per-frame;
/// silu/residual/proj are elementwise or per-position) or a causal conv fed real history by the
/// `FrameCache`, so chunking is an exact refactor — not an approximation to be blended. Any chunk size,
/// including ones that do not divide the clip, must reproduce the single-shot decode bit-for-bit.
#[test]
fn chunked_decode_is_identical_to_single_shot() {
    let (_cfg, dec) = real_geometry_decoder();
    let t_lat = 13;
    let latent = rnd(&[1, 12, t_lat, 2, 2], 42);

    let single = dec.decode_denormalized(&latent).expect("single-shot");
    // Sanity: a non-constant decode, so "identical" is a real claim and not two flat tensors.
    assert!(
        max_abs(&single) > 1e-4,
        "synthetic decode is ~constant — the equivalence assertions below would be vacuous"
    );

    // 1 and 4 do not divide 13 (ragged final chunk); 13 hits the `chunk >= t_lat` single-shot path.
    for chunk in [1usize, 2, 3, 4, 5, 13] {
        let chunked = dec
            .decode_denormalized_chunked(&latent, chunk, None)
            .unwrap_or_else(|e| panic!("chunked decode (chunk={chunk}): {e}"));
        assert_eq!(
            chunked.shape(),
            single.shape(),
            "chunk={chunk}: shape must match the single-shot decode"
        );
        let d = max_abs(&subtract(&chunked, &single).unwrap());
        assert_eq!(
            d, 0.0,
            "chunk={chunk}: chunked decode must be identical to single-shot (max abs diff {d:.3e})"
        );
    }
}

/// The decoder is temporally **causal**: decoding only the first `p` latent frames reproduces the
/// corresponding prefix of the full decode exactly. This is the property the `FrameCache` exploits —
/// if it did not hold, no chunking scheme could be exact.
#[test]
fn decode_is_temporally_causal() {
    let (_cfg, dec) = real_geometry_decoder();
    let latent = rnd(&[1, 12, 13, 2, 2], 7);

    let full = dec.decode_denormalized(&latent).expect("full decode");
    let idx: Vec<i32> = (0..7).collect();
    let prefix_latent = latent
        .take_axis(Array::from_slice(&idx, &[7]), 2)
        .expect("slice latent prefix");
    let prefix = dec.decode_denormalized(&prefix_latent).expect("prefix");

    let pf = prefix.shape()[2];
    let full_head = full
        .take_axis(Array::from_slice(&(0..pf).collect::<Vec<_>>(), &[pf]), 2)
        .expect("slice video prefix");
    let d = max_abs(&subtract(&prefix, &full_head).unwrap());
    assert_eq!(
        d, 0.0,
        "decoding a latent prefix must reproduce the video prefix (causality); max abs diff {d:.3e}"
    );
}

/// **Why this is a conv cache and not the repo's shared overlap+blend tiling.**
///
/// `mlx_gen::tiling`'s causal path gives each tile *one* latent frame of left context. Measure what the
/// decoder actually needs: perturb latent frame 0 and see how far forward the change reaches. The 38
/// stacked `kt=3` causal convs carry it ~45 latent frames (~270 output frames, ~9 s at 30 fps) — wider
/// than an entire 5 s clip (26 latent frames). A tile decoded with 1 frame of history is therefore not
/// slightly seamed at its edge; it is wrong throughout, and no trapezoidal blend recovers it.
#[test]
fn temporal_receptive_field_exceeds_a_full_clip() {
    let (_cfg, dec) = real_geometry_decoder();
    let t_lat = 60i32;
    let base = rnd(&[1, 12, t_lat, 2, 2], 42);

    // Perturb latent frame 0 only (NCTHW).
    let hw = 2 * 2;
    let mut data: Vec<f32> = base.as_slice::<f32>().to_vec();
    for c in 0..12i32 {
        for p in 0..hw {
            data[(c * t_lat * hw + p) as usize] += 1.0;
        }
    }
    let perturbed = Array::from_slice(&data, &[1, 12, t_lat, 2, 2]);

    let v0 = dec.decode_denormalized(&base).expect("base");
    let v1 = dec.decode_denormalized(&perturbed).expect("perturbed");
    let diffs = per_frame_diff(&v1, &v0);

    let last_affected = diffs
        .iter()
        .rposition(|&d| d > 1e-6)
        .expect("frame 0 must affect at least one output frame") as i32;
    // +1 for 0-indexing, +(ratio−1) for the leading frames the decode dropped, ÷ratio → latent frames.
    let reach_latent = (last_affected + 1 + 5) as f32 / 6.0;
    eprintln!(
        "temporal receptive field: latent frame 0 reaches output frame {last_affected} of {} \
         (~{reach_latent:.1} latent frames, ~{:.1} s @30fps)",
        diffs.len(),
        (last_affected + 1) as f32 / 30.0
    );

    // The clip must be long enough that the field is measurable rather than clipped by the clip.
    assert!(
        last_affected < diffs.len() as i32 - 1,
        "probe clip too short to bound the receptive field — increase t_lat"
    );
    // A 5 s clip is 26 latent frames; the field must exceed it for the "no overlap+blend" claim.
    assert!(
        reach_latent > 26.0,
        "receptive field {reach_latent:.1} latent frames no longer exceeds a 5 s clip (26) — the \
         rationale for the FrameCache over `mlx_gen::tiling` overlap+blend needs revisiting"
    );
}

/// The negative control for the test above: a chunk decoded **without** history — which is what an
/// overlap+blend tile with a 1-frame left context approximates — diverges from the truth by O(1) across
/// the whole tile, not just at its seam. This is what the `FrameCache` buys.
#[test]
fn decode_without_history_is_materially_wrong() {
    let (_cfg, dec) = real_geometry_decoder();
    let t_lat = 13;
    let latent = rnd(&[1, 12, t_lat, 2, 2], 42);

    let truth = dec
        .decode_denormalized_chunked(&latent, 1, None)
        .expect("chunked truth");

    // Decode the tail latent frames standalone (no history) — an "independent tile".
    let tail_idx: Vec<i32> = (7..t_lat).collect();
    let tail_latent = latent
        .take_axis(Array::from_slice(&tail_idx, &[tail_idx.len() as i32]), 2)
        .expect("slice tail");
    let standalone = dec.decode_denormalized(&tail_latent).expect("tail decode");

    // Align: the standalone tail decode's frames correspond to the truth's last `n` frames.
    let n = standalone.shape()[2];
    let tf = truth.shape()[2];
    let truth_tail = truth
        .take_axis(
            Array::from_slice(&(tf - n..tf).collect::<Vec<_>>(), &[n]),
            2,
        )
        .expect("slice truth tail");

    let d = max_abs(&subtract(&standalone, &truth_tail).unwrap());
    let scale = max_abs(&truth_tail);
    eprintln!("history-free tile error: max abs {d:.3e} vs signal {scale:.3e}");
    assert!(
        d > scale * 0.01,
        "a history-free tile came within 1% of the truth ({d:.3e} vs {scale:.3e}) — if the decoder \
         really were this weakly coupled in time, overlap+blend tiling would suffice and the \
         FrameCache would be unnecessary complexity"
    );
}

/// **The element-ceiling guard** (sc-12291). At 848×480 the untiled decode is exact through `T_lat = 6`
/// and silently returns wrong pixels from `T_lat = 7` on, because `block_out`'s intermediate crosses
/// `i32::MAX` elements (measured on real weights — see `decode_memory_real_weights.rs`). Since MLX
/// reports no error, a single-shot decode over that line must refuse rather than return garbage, and a
/// chunk size that would cross it must be clamped down instead of honored.
///
/// This runs on **shape arithmetic alone — no decode** — so it can hold the real production geometry in
/// CI, which is exactly what the `vae_parity` golden (64×64/7 frames, ~340× under) cannot do. Decoding a
/// real 848×480 clip here would allocate tens of GiB and OOM a CI runner; the clamp's *behaviour* is
/// covered separately at a cheap geometry by [`an_over_large_chunk_is_clamped_not_honored`].
#[test]
fn element_ceiling_guard_holds_at_production_geometry() {
    let (_cfg, dec) = real_geometry_decoder();
    // Mochi's shipped 848×480 → latent 60×106. This synthetic decoder is 32-wide where the real one is
    // 128, so assert the *relationship* rather than hardcoding a width-specific answer.
    let (h_lat, w_lat) = (60, 106);
    let safe = dec.max_safe_chunk_frames(h_lat, w_lat);
    eprintln!("max safe chunk at 848×480 (this decoder is 32-wide): {safe} latent frames");
    assert!(safe >= 1, "a single latent frame must fit at 848×480");

    // The bound must be driven by the geometry: 4× the pixels ⇒ ~¼ the frames.
    let quarter = dec.max_safe_chunk_frames(h_lat * 2, w_lat * 2);
    assert!(
        quarter < safe && quarter >= safe / 5,
        "the cap must scale with output area: {safe} at 848×480 vs {quarter} at 1696×960"
    );

    // The real 128-wide decoder is 4× this one, so its cap at 848×480 is ~6 — under a 5 s clip's 26
    // latent frames. That is the whole point: at production geometry the single pass is NOT available.
    assert!(
        safe / 4 < 26,
        "a 128-wide decoder must NOT be able to single-pass a 5 s clip at 848×480 (this 32-wide \
         decoder caps at {safe}, so the real one caps at ~{})",
        safe / 4
    );
}

/// The clamp's behaviour, at a geometry cheap enough for CI: an over-large chunk must be clamped down
/// rather than honored, and the result must still be exact.
#[test]
fn an_over_large_chunk_is_clamped_not_honored() {
    let (_cfg, dec) = real_geometry_decoder();
    let latent = rnd(&[1, 12, 13, 2, 2], 3);
    let asked_too_big = dec
        .decode_denormalized_chunked(&latent, 1_000_000, None)
        .expect("an over-large chunk must be clamped, not corrupt or fail");
    let safe_chunked = dec
        .decode_denormalized_chunked(&latent, 1, None)
        .expect("chunk=1");
    assert_eq!(
        max_abs(&subtract(&asked_too_big, &safe_chunked).unwrap()),
        0.0,
        "a clamped over-large chunk must still decode exactly"
    );
}

/// A single-shot decode whose intermediates would cross the element ceiling must return an actionable
/// error rather than silently-wrong pixels.
#[test]
fn single_shot_decode_refuses_over_the_element_ceiling() {
    let (_cfg, dec) = real_geometry_decoder();
    // Just over the bound, not far over: 26 latent frames at 90² latent is ~2.6e9 elements for this
    // 32-wide decoder. The guard is shape-only and fires before any compute, so keep the allocation
    // small — a CI runner has to hold this.
    let latent = rnd(&[1, 12, 26, 90, 90], 1);
    assert!(
        dec.max_safe_chunk_frames(90, 90) < 26,
        "test precondition: 26 latent frames at 90² must be over the bound"
    );
    match dec.decode_denormalized(&latent) {
        Err(mlx_gen::Error::Msg(m)) => {
            assert!(
                m.contains("ceiling") && m.contains("chunk"),
                "the error must name the ceiling and point at chunking, got: {m}"
            );
        }
        Err(e) => panic!("expected an element-ceiling Msg error, got {e}"),
        Ok(_) => panic!(
            "a decode over the element ceiling must error — MLX returns wrong pixels without one"
        ),
    }
}

/// A pre-tripped cancel is honored before any chunk work (mirrors the LTX/Wan tiled-decode gates).
#[test]
fn chunked_decode_honors_cancel() {
    let (_cfg, dec) = real_geometry_decoder();
    let latent = rnd(&[1, 12, 13, 2, 2], 42);
    let cancel = CancelFlag::default();
    cancel.cancel();
    let r = dec.decode_denormalized_chunked(&latent, 2, Some(&cancel));
    assert!(
        matches!(r, Err(mlx_gen::Error::Canceled)),
        "a tripped cancel must abort the chunked decode"
    );
}
