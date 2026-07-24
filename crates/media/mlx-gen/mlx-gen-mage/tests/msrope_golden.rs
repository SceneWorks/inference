//! sc-14040: msrope parity against the frozen torch reference — **weights-free**.
//!
//! `MageFlowEmbedRope` has no parameters, so the whole table is determined by `img_shapes`,
//! `axes_dim`, `theta` and `scale_rope`. That makes it the one part of the DiT that can be gated
//! against the real reference without the 8.2 GB checkpoint: the block golden ships the exact
//! complex table the reference handed to `transformer_blocks[0]`, split into `_re` / `_im`.
//!
//! `#[ignore]`d only because the goldens are gitignored (they need the licensed weights to
//! produce), not because they need weights to *check*.

mod common;

use common::{error, ints, require_golden, BLOCK_GOLDEN, STACK_GOLDEN};

use mlx_rs::ops::concatenate_axis;

use mlx_gen_mage::config::MageFlowConfig;
use mlx_gen_mage::{ImgShape, MsRope, PackLayout, RopeTable};

/// The golden's `img_shapes`, `[segments, 3]` of `(frames, height, width)`.
fn golden_shapes() -> Vec<ImgShape> {
    let stack = require_golden(STACK_GOLDEN);
    let flat = ints(&stack, "img_shapes");
    assert_eq!(flat.len() % 3, 0, "img_shapes must be [segments, 3]");
    flat.chunks(3)
        .map(|s| ImgShape::new(s[0], s[1], s[2]))
        .collect()
}

fn golden_table() -> RopeTable {
    let block = require_golden(BLOCK_GOLDEN);
    RopeTable {
        cos: block
            .require("block_in.image_rotary_emb_re")
            .unwrap()
            .clone(),
        sin: block
            .require("block_in.image_rotary_emb_im")
            .unwrap()
            .clone(),
    }
}

/// The ported table must reproduce the reference's, including the **fused-CFG frame shift**: the
/// golden was dumped at `cfg = 5.0` with `batch_cfg` on, so its `img_shapes` carries the segment
/// list duplicated and the second copy rotates at frame index 1.
///
/// Tolerance is f32 trig, not a model tolerance: both sides evaluate `cos/sin` of the same angles
/// in f32, so this is a near-bit-exact check (measured max_abs ~1e-7).
#[test]
#[ignore = "needs crates/media/mlx-gen/tools/golden/mage_flow_dit_{block_,}golden.safetensors"]
fn msrope_matches_the_reference_table() {
    let shapes = golden_shapes();
    assert_eq!(
        shapes.len(),
        2,
        "the golden is a fused-CFG pack — two segments, both 1×16×16"
    );
    let rope = MsRope::from_config(&MageFlowConfig::mage_flow()).unwrap();
    let got = rope.forward(&shapes).unwrap();
    let want = golden_table();

    let (cos_abs, cos_rel, cos_mean) = error(&got.cos, &want.cos);
    let (sin_abs, sin_rel, sin_mean) = error(&got.sin, &want.sin);
    println!(
        "msrope cos: max_abs {cos_abs:.3e} max_rel {cos_rel:.3e} mean_rel {cos_mean:.3e}\n\
         msrope sin: max_abs {sin_abs:.3e} max_rel {sin_rel:.3e} mean_rel {sin_mean:.3e}"
    );
    assert!(cos_abs < 1e-5, "msrope cos diverged: max_abs {cos_abs}");
    assert!(sin_abs < 1e-5, "msrope sin diverged: max_abs {sin_abs}");
}

/// The counter-probe: rotating the duplicated **unconditional** branch at frame **0** — the
/// reading the reference's own docstring at `pipeline.py:136-140` implies ("numerically identical
/// to two separate forwards") — does NOT reproduce the golden, and the disagreement is confined
/// entirely to the frame lanes.
///
/// Without this, `msrope_matches_the_reference_table` would be a test that any plausible msrope
/// passes: the spatial lanes dominate 56 of the 64 columns.
#[test]
#[ignore = "needs crates/media/mlx-gen/tools/golden/mage_flow_dit_{block_,}golden.safetensors"]
fn rotating_the_uncond_branch_at_frame_zero_does_not_match_the_golden() {
    let shapes = golden_shapes();
    let rope = MsRope::from_config(&MageFlowConfig::mage_flow()).unwrap();
    // Build each segment's table as if it were segment 0 and stack them: every branch at frame 0.
    let per_segment: Vec<_> = shapes
        .iter()
        .map(|s| rope.forward(std::slice::from_ref(s)).unwrap())
        .collect();
    let cos: Vec<_> = per_segment.iter().map(|t| &t.cos).collect();
    let sin: Vec<_> = per_segment.iter().map(|t| &t.sin).collect();
    let wrong_cos = concatenate_axis(&cos, 0).unwrap();
    let wrong_sin = concatenate_axis(&sin, 0).unwrap();

    let want = golden_table();
    let (cos_abs, ..) = error(&wrong_cos, &want.cos);
    let (sin_abs, ..) = error(&wrong_sin, &want.sin);
    println!("frame-0 uncond: cos max_abs {cos_abs:.4} sin max_abs {sin_abs:.4}");
    assert!(
        cos_abs > 0.4 && sin_abs > 0.4,
        "the frame-index shift must be observable in the table (cos {cos_abs}, sin {sin_abs}); \
         if this ever passes, the golden was dumped without batch_cfg and every downstream \
         parity claim about the fused path is void"
    );

    // ...and the damage is confined to the frame lanes: `axes_dim = [16, 56, 56]` ⇒ columns 0..8.
    let half = want.cos.shape()[1];
    let (rows, frame_lanes) = (want.cos.shape()[0], 8);
    let spatial = |a: &mlx_rs::Array| {
        a.reshape(&[rows, half])
            .unwrap()
            .take_axis(
                mlx_rs::Array::from_slice(
                    &(frame_lanes..half).collect::<Vec<i32>>(),
                    &[half - frame_lanes],
                ),
                1,
            )
            .unwrap()
    };
    let (spatial_abs, ..) = error(&spatial(&wrong_cos), &spatial(&want.cos));
    assert!(
        spatial_abs < 1e-5,
        "the frame index must not touch the height/width lanes (max_abs {spatial_abs})"
    );
}

/// The layout this crate builds for a fused-CFG forward must reproduce the reference's packing
/// byte-for-byte in its offsets, not just in its shapes.
#[test]
#[ignore = "needs crates/media/mlx-gen/tools/golden/mage_flow_dit_golden.safetensors"]
fn fused_cfg_layout_reproduces_the_goldens_cu_seqlens() {
    let stack = require_golden(STACK_GOLDEN);
    let img_cu = ints(&stack, "dit_in.img_cu_seqlens");
    let txt_cu = ints(&stack, "dit_in.txt_cu_seqlens");
    let shapes = golden_shapes();
    assert_eq!(
        shapes[0], shapes[1],
        "fused CFG duplicates the image segment"
    );

    let pos = txt_cu[1] - txt_cu[0];
    let neg = txt_cu[2] - txt_cu[1];
    let layout = PackLayout::generation(shapes[..1].to_vec(), vec![pos])
        .unwrap()
        .fused_cfg(&[neg])
        .unwrap();
    assert_eq!(layout.img_cu(), img_cu);
    assert_eq!(layout.txt_cu(), txt_cu);
    assert_eq!(layout.img_shapes(), shapes.as_slice());
}
