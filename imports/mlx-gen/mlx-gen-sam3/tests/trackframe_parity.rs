//! SAM3-F2.5a no-prompt tracking-frame decode parity (sc-4924): load the real `facebook/sam3`
//! weights, feed the captured memory-conditioned features + high-res features into
//! `Sam3Tracker::decode_tracked_frame`, and check the low-res/high-res masks + object pointer + score
//! against the torch oracle (`scripts/spikes/sam3_oracle/dump_trackframe_fixture.py`, captured by
//! wrapping `_single_frame_forward` on the first no-prompt call of a real 2-frame `Sam3VideoModel`
//! PCS run).
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_TRACKFRAME_FIXTURE=$PWD/scripts/spikes/sam3_oracle/trackframe_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test trackframe_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::Sam3Tracker;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

fn scalar(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let dot = scalar(&sum(multiply(&a, &b).unwrap(), None).unwrap());
    let na = scalar(&sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap()).unwrap());
    let nb = scalar(&sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

/// NCHW [1,C,H,W] → NHWC [1,H,W,C].
fn nhwc(a: &Array) -> Array {
    a.transpose_axes(&[0, 2, 3, 1]).unwrap()
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_TRACKFRAME_FIXTURE"]
fn tracked_frame_decode_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_TRACKFRAME_FIXTURE").unwrap_or_else(|_| {
        "scripts/spikes/sam3_oracle/trackframe_fixture.safetensors".to_string()
    });

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load trackframe fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    // Captured decoder inputs (NCHW → NHWC).
    let pix_feat = nhwc(fx.require("pix_feat").unwrap()); // [1,72,72,256]
    let feat_s0 = nhwc(fx.require("feat_s0").unwrap()); // [1,288,288,32]
    let feat_s1 = nhwc(fx.require("feat_s1").unwrap()); // [1,144,144,64]

    let out = tracker
        .decode_tracked_frame(&pix_feat, &[feat_s0, feat_s1])
        .expect("decode_tracked_frame");

    let pred_want = fx.require("pred_masks").unwrap().clone(); // [1,1,288,288]
    let high_want = fx.require("high_res_masks").unwrap().clone(); // [1,1,1008,1008]
    let ptr_want = fx.require("object_pointer").unwrap().clone(); // [1,1,256]
    let score_want = scalar(fx.require("object_score_logits").unwrap());

    let c_low = cosine(&out.low_res, &pred_want);
    let c_high = cosine(&out.high_res, &high_want);
    let c_ptr = cosine(&out.object_pointer, &ptr_want);
    println!(
        "low_res cosine={c_low:.7}  high_res cosine={c_high:.7}  object_pointer cosine={c_ptr:.7}  \
         object_score got={:.4} want={score_want:.4} |Δ|={:.4}",
        out.object_score,
        (out.object_score - score_want).abs()
    );
    assert!(c_low > 0.9999, "low_res mask cosine {c_low}");
    assert!(c_high > 0.9999, "high_res mask cosine {c_high}");
    assert!(c_ptr > 0.9999, "object_pointer cosine {c_ptr}");
    assert!(
        (out.object_score - score_want).abs() < 0.05,
        "object_score |Δ| too large"
    );
}
