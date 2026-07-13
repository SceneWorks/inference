//! SAM3-F2 dynamic-multimask-via-stability parity (sc-4924): run the `Sam3Tracker` single-frame
//! decode with `multimask=false` (the no-prompt video-frame policy → `dynamic_multimask_via_stability`)
//! and check it against the torch oracle (`scripts/spikes/sam3_oracle/dump_dynmask_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_DYNMASK_FIXTURE=scripts/spikes/sam3_oracle/dynmask_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test dynmask_parity -- --ignored --nocapture

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

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_DYNMASK_FIXTURE"]
fn dynamic_multimask_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_DYNMASK_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/dynmask_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load dynmask fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pixel_values = fx.require("pixel_values").unwrap().clone();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    let (emb, high_res) = tracker.encode_frame(&pixel_values).expect("encode_frame");
    let out = tracker
        .segment_encoded_multimask(&emb, &high_res, box_xyxy, false)
        .expect("segment multimask=false");

    let want_mask = fx.require("dyn_mask").unwrap().clone(); // [mg, mg]
    let want_iou = scalar(fx.require("dyn_iou").unwrap());
    assert_eq!(out.low_res.shape(), want_mask.shape(), "dyn mask shape");
    let c = cosine(&out.low_res, &want_mask);
    println!(
        "dyn_mask: cosine={c:.7}  iou ours={:.4} oracle={want_iou:.4}",
        out.iou
    );
    // The dynamic-stability branch must pick the SAME candidate as the reference (else cosine tanks).
    assert!(c > 0.999, "dyn mask cosine {c}");
    // iou is a small MLP logit → relative-tolerance compare (MLX Metal matmul is reduced-precision);
    // the cosine above is the real guarantee that the SAME candidate was selected.
    assert!(
        (out.iou - want_iou).abs() / want_iou.abs().max(1.0) < 0.01,
        "dyn iou ours {} vs {want_iou}",
        out.iou
    );
}
