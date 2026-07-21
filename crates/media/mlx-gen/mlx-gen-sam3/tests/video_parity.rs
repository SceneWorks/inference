//! SAM3-F2.6 end-to-end multi-object video PCS parity (sc-4924): run `Sam3VideoModel::propagate` on
//! the captured 2-frame clip and check per-frame `obj_id` sets + per-object 288² mask logits against
//! the torch oracle (`scripts/spikes/sam3_oracle/dump_video_fixture.py`, full `Sam3VideoModel` run).
//!
//! Run:
//!   SAM3_WEIGHTS=/path/to/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_VIDEO_FIXTURE=$PWD/scripts/spikes/sam3_oracle/video_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test video_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::Sam3VideoModel;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

fn scalar(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let aa = Array::from_slice(a, &[a.len() as i32]);
    let bb = Array::from_slice(b, &[b.len() as i32]);
    let dot = scalar(&sum(multiply(&aa, &bb).unwrap(), None).unwrap());
    let na = scalar(&sqrt(sum(multiply(&aa, &aa).unwrap(), None).unwrap()).unwrap());
    let nb = scalar(&sqrt(sum(multiply(&bb, &bb).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

fn ints(a: &Array) -> Vec<i32> {
    a.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec()
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_VIDEO_FIXTURE"]
fn video_pcs_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_VIDEO_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/video_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load video fixture");
    let mut model = Sam3VideoModel::from_weights(&w).expect("build video model");

    let num_frames = ints(fx.require("num_frames").unwrap())[0];
    let frames: Vec<Array> = (0..num_frames)
        .map(|f| fx.require(&format!("frame_{f}")).unwrap().clone())
        .collect();
    let input_ids = fx.require("input_ids").unwrap().clone();
    let text_mask = ints(fx.require("attention_mask").unwrap());

    let outputs = model
        .propagate(&frames, &input_ids, &text_mask, None, None)
        .expect("propagate");

    let mut worst = 1.0f32;
    for f in 0..num_frames {
        let want_ids = ints(fx.require(&format!("obj_ids_{f}")).unwrap());
        let want_masks = fx.require(&format!("masks_{f}")).unwrap().clone(); // [num_obj,288,288]
        let out = &outputs[f as usize];
        println!("frame {f}: got obj_ids={:?} want={want_ids:?}", out.obj_ids);
        assert_eq!(out.obj_ids, want_ids, "frame {f} obj_id mismatch");
        let want_flat = want_masks
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let per = (288 * 288) as usize;
        let mut frame_worst = 1.0f32;
        for (oi, mask) in out.masks.iter().enumerate() {
            let want = &want_flat[oi * per..(oi + 1) * per];
            let c = cosine(mask, want);
            frame_worst = frame_worst.min(c);
            worst = worst.min(c);
        }
        println!("frame {f}: worst obj cosine={frame_worst:.7}");
    }
    println!("worst per-object mask cosine = {worst:.7}");
    assert!(worst > 0.999, "video PCS mask cosine {worst}");
}
