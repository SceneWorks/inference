//! SAM3-F2 tracker memory-attention parity (sc-4924): load the real `facebook/sam3` weights, run the
//! `Sam3Tracker` memory attention (`condition_with_memory` + RoPE tables), and check it against the
//! torch oracle (`scripts/spikes/sam3_oracle/dump_memattn_fixture.py`, captured via a forward hook on
//! a real 2-frame `Sam3VideoModel` PCS run).
//!
//! Run:
//!   SAM3_WEIGHTS=/path/to/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_MEMATTN_FIXTURE=scripts/spikes/sam3_oracle/memattn_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test memattn_parity -- --ignored --nocapture

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
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_MEMATTN_FIXTURE"]
fn memory_attention_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_MEMATTN_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/memattn_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load memattn fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    // --- Stage 1: RoPE tables (host-built create_inv_freq) vs the reference rotary_emb().
    let (cos, sin) = tracker.memory_attention_rope_tables();
    for (key, got) in [("rope_cos", &cos), ("rope_sin", &sin)] {
        let want = fx.require(key).unwrap().clone();
        assert_eq!(got.shape(), want.shape(), "{key} shape");
        let c = cosine(got, &want);
        println!("{key}: cosine={c:.7}");
        assert!(c > 0.999999, "{key} cosine {c}");
    }

    // --- Stage 2: full memory attention (self-attn + cross-attn over the bank + FFN, 4 layers).
    let cvf = fx.require("current_vision_features").unwrap().clone(); // [5184,1,256]
    let cvp = fx
        .require("current_vision_position_embeddings")
        .unwrap()
        .clone();
    let mem = fx.require("memory").unwrap().clone(); // [seq_k,1,64]
    let mem_pos = fx.require("memory_pos").unwrap().clone();
    let n_optr = fx
        .require("num_object_pointer_tokens")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()[0];

    let got = tracker
        .condition_with_memory(&cvf, &cvp, &mem, &mem_pos, n_optr)
        .expect("condition_with_memory");
    let want = fx.require("output").unwrap().clone(); // [1,1,5184,256] — flatten-compatible with [1,5184,256]
    assert_eq!(
        got.reshape(&[-1]).unwrap().shape(),
        want.reshape(&[-1]).unwrap().shape(),
        "output numel"
    );
    let c = cosine(&got, &want);
    println!("memory_attention output (num_obj_ptr={n_optr}): cosine={c:.7}");
    // 4 stacked attention layers over 5184 tokens; MLX Metal matmul is reduced-precision.
    assert!(c > 0.999, "memory_attention cosine {c}");
}
