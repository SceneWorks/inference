//! sc-2528: Qwen adapter key→module routing for the targets whose trained-file (diffusers) naming
//! differs from the crate's internal fields — joint attention (`to_out.0`, the text-stream
//! `add_{q,k,v}_proj` → `add_{q,k,v}`) and the stream feed-forwards (`net.0.proj`/`net.2`). The
//! full 60-block routing is gated locally against real weights; this locks the translations in CI
//! with synthetic temp fixtures (no real weights).

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::adapters::{install_adapter, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::transformer::{FeedForward, QwenJointAttention};
use mlx_rs::Array;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("mlx_gen_qwen_routing_test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn dummy() -> Adapter {
    Adapter::Lora {
        a: Array::from_slice(&[0.0f32], &[1, 1]),
        b: Array::from_slice(&[0.0f32], &[1, 1]),
        scale: 0.0,
    }
}

fn write(path: &PathBuf, arrays: Vec<(&str, &Array)>) {
    Array::save_safetensors(arrays, None as Option<&HashMap<String, String>>, path).unwrap();
}

#[test]
fn attention_routes_diffusers_names() {
    // inner = num_heads*head_dim = 2*4 = 8; all 8 projections [8,8]+bias[8], norms [4].
    let w8 = Array::from_slice(&vec![0.1f32; 64], &[8, 8]);
    let b8 = Array::from_slice(&[0.0f32; 8], &[8]);
    let n4 = Array::from_slice(&[1.0f32; 4], &[4]);
    let path = tmp("attn.safetensors");
    let mut t: Vec<(&str, &Array)> = Vec::new();
    for p in [
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "attn_to_out.0",
        "to_add_out",
    ] {
        t.push((Box::leak(format!("{p}.weight").into_boxed_str()), &w8));
        t.push((Box::leak(format!("{p}.bias").into_boxed_str()), &b8));
    }
    for p in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
        t.push((Box::leak(format!("{p}.weight").into_boxed_str()), &n4));
    }
    write(&path, t);
    let w = Weights::from_file(&path).unwrap();
    let mut attn = QwenJointAttention::from_weights(&w, "", 2, 4).unwrap();

    // Trained-file (diffusers) naming resolves.
    for p in [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ] {
        assert!(
            install_adapter(&mut attn, p, dummy()).is_ok(),
            "{p} should resolve"
        );
    }
    // Off-surface / internal names must not.
    for p in ["to_out", "add_q", "to_q.0", "to_add_out.0"] {
        assert!(
            install_adapter(&mut attn, p, dummy()).is_err(),
            "{p} must not resolve"
        );
    }
}

#[test]
fn feed_forward_routes_net_indices() {
    // mlp_in [16,8], mlp_out [8,16] + biases.
    let win = Array::from_slice(&vec![0.1f32; 128], &[16, 8]);
    let bin = Array::from_slice(&[0.0f32; 16], &[16]);
    let wout = Array::from_slice(&vec![0.1f32; 128], &[8, 16]);
    let bout = Array::from_slice(&[0.0f32; 8], &[8]);
    let path = tmp("ff.safetensors");
    write(
        &path,
        vec![
            ("mlp_in.weight", &win),
            ("mlp_in.bias", &bin),
            ("mlp_out.weight", &wout),
            ("mlp_out.bias", &bout),
        ],
    );
    let w = Weights::from_file(&path).unwrap();
    let mut ff = FeedForward::from_weights(&w, "").unwrap();

    // diffusers file naming: `net.0.proj` (in) / `net.2` (out).
    assert!(install_adapter(&mut ff, "net.0.proj", dummy()).is_ok());
    assert!(install_adapter(&mut ff, "net.2", dummy()).is_ok());
    // Internal field names + other indices must not resolve.
    assert!(install_adapter(&mut ff, "mlp_in", dummy()).is_err());
    assert!(install_adapter(&mut ff, "net.1", dummy()).is_err());
    assert!(install_adapter(&mut ff, "net.0", dummy()).is_err());
}
