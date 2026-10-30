//! SAM3 tracker single-frame (box-prompt PVS) parity (sc-6245): load the real `facebook/sam3`
//! weights, run the candle `Sam3Tracker` single-frame box path, and check it against the SAME torch
//! oracle the MLX twin uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_tracker_fixture.py`).
//!
//! `#[ignore]` until `facebook/sam3` (gated) + the fixture are staged on the box (sc-6248). Run:
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot> SAM3_TRACKER_FIXTURE=<tracker_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test integration tracker_parity:: -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::default_device;
use candle_gen::testkit::{tensor_cosine_device_sums, Denom};
use candle_gen_sam3::{Sam3Tracker, Weights};

fn scalar(t: &Tensor) -> f32 {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
}

fn load_weights(path: &str, device: &Device) -> Weights {
    let wp = Path::new(path);
    if wp.is_dir() {
        Weights::from_dir(wp, device)
    } else {
        Weights::from_file(wp, device)
    }
    .expect("load sam3 weights")
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_TRACKER_FIXTURE — sc-6248"]
fn tracker_single_frame_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_TRACKER_FIXTURE").expect("set SAM3_TRACKER_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load tracker fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pixel_values = fx.require("pixel_values").unwrap();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    // --- Stage: tracker neck (shared backbone → FPN → conv_s0/s1). ours NHWC → NCHW to match.
    let (_emb, high_res) = tracker.encode_frame(&pixel_values).expect("encode_frame");
    for (i, key) in ["high_res_s0", "high_res_s1"].iter().enumerate() {
        let got = high_res[i]
            .permute([0, 3, 1, 2])
            .unwrap()
            .contiguous()
            .unwrap();
        let want = fx.require(key).unwrap();
        assert_eq!(got.dims(), want.dims(), "{key} shape");
        let c = tensor_cosine_device_sums(&got, &want, Denom::Exact);
        println!("{key}: cosine={c:.7}");
        assert!(c > 0.9999, "{key} cosine {c}");
    }

    // --- End-to-end: box-prompt → best low-res mask + iou + object score.
    let out = tracker.segment(&pixel_values, box_xyxy).expect("segment");
    let want_mask = fx.require("best_low_res").unwrap(); // [288, 288]
    assert_eq!(out.low_res.dims(), want_mask.dims(), "mask shape");
    let c = tensor_cosine_device_sums(&out.low_res, &want_mask, Denom::Exact);
    let want_obj = scalar(&fx.require("object_score").unwrap());
    println!(
        "mask: cosine={c:.7}  iou ours={:.4}  obj ours={:.3} oracle={:.3}",
        out.iou, out.object_score, want_obj
    );
    assert!(c > 0.999, "mask cosine {c}");
    assert!(
        (out.object_score - want_obj).abs() / want_obj.abs().max(1.0) < 0.01,
        "object score ours {} vs {want_obj}",
        out.object_score
    );

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED tracker_parity::tracker_single_frame_matches_oracle"
    );
}
