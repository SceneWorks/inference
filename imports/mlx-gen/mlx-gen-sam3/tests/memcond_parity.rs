//! SAM3-F2.4 per-object memory-conditioning parity (sc-4924): load the real `facebook/sam3` weights,
//! assemble the per-object memory bank + object pointers and run the full
//! `Sam3Tracker::prepare_memory_conditioned_features`, and check the conditioned feature map against
//! the torch oracle (`scripts/spikes/sam3_oracle/dump_memcond_fixture.py`, captured by wrapping
//! `_gather_memory_frame_outputs` / `_get_object_pointers` / `_prepare_memory_conditioned_features`
//! on a real 2-frame `Sam3VideoModel` PCS run).
//!
//! This validates the bank-assembly math (`_build_memory_attention_inputs` temporal-pos add +
//! `_process_object_pointers` sine-PE/project/4×64-split) end-to-end through memory attention; the
//! attention itself is separately gated by `memattn_parity`.
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_MEMCOND_FIXTURE=scripts/spikes/sam3_oracle/memcond_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test memcond_parity -- --ignored --nocapture

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

fn ints(a: &Array) -> Vec<i32> {
    a.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec()
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_MEMCOND_FIXTURE"]
fn memory_conditioning_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_MEMCOND_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/memcond_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load memcond fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let cvf = fx.require("current_vision_features").unwrap().clone(); // [5184,1,256]
    let cvp = fx
        .require("current_vision_position_embeddings")
        .unwrap()
        .clone();

    // Spatial memory frames: [M,5184,1,64] features/pos + [M] offsets.
    let mem_feats = fx.require("memory_features").unwrap().clone();
    let mem_pos = fx.require("memory_pos_enc").unwrap().clone();
    let mem_offsets = ints(fx.require("memory_offsets").unwrap());
    let m = mem_feats.shape()[0];
    let seq = mem_feats.shape()[1];
    let mut spatial: Vec<(i32, Array, Array)> = Vec::new();
    for i in 0..m {
        let feat = mem_feats
            .take_axis(Array::from_int(i), 0)
            .unwrap()
            .reshape(&[seq, 1, 64])
            .unwrap();
        let pos = mem_pos
            .take_axis(Array::from_int(i), 0)
            .unwrap()
            .reshape(&[seq, 1, 64])
            .unwrap();
        spatial.push((mem_offsets[i as usize], feat, pos));
    }

    // Object pointers: [P,1,1,256] + [P] offsets.
    let ptrs = fx.require("object_pointers").unwrap().clone();
    let ptr_offsets = ints(fx.require("object_pointer_offsets").unwrap());
    let max_optr = ints(fx.require("max_object_pointers_to_use").unwrap())[0];
    let n_optr_want = ints(fx.require("num_object_pointer_tokens").unwrap())[0];
    let p = ptrs.shape()[0];
    let mut object_pointers: Vec<(i32, Array)> = Vec::new();
    for j in 0..p {
        let t = ptrs
            .take_axis(Array::from_int(j), 0)
            .unwrap()
            .reshape(&[1, 256])
            .unwrap();
        object_pointers.push((ptr_offsets[j as usize], t));
    }
    println!(
        "M={m} mem_offsets={mem_offsets:?} P={p} ptr_offsets={ptr_offsets:?} max_optr={max_optr} \
         num_obj_ptr_tokens(want)={n_optr_want}",
    );

    let got = tracker
        .prepare_memory_conditioned_features(&cvf, &cvp, &spatial, &object_pointers, max_optr)
        .expect("prepare_memory_conditioned_features"); // NHWC [1,72,72,256]

    // Reference output is NCHW [1,256,72,72] → permute to NHWC for an aligned cosine.
    let want_nchw = fx.require("output").unwrap().clone();
    let want = want_nchw.transpose_axes(&[0, 2, 3, 1]).unwrap(); // [1,72,72,256]
    assert_eq!(got.shape(), want.shape(), "conditioned feature map shape");
    let c = cosine(&got, &want);
    println!("conditioned feature map: cosine={c:.7}");
    // Full bank assembly + 4 stacked memory-attention layers over 5184 tokens; MLX Metal matmul is
    // reduced-precision.
    assert!(c > 0.999, "memory-conditioned feature map cosine {c}");
}
