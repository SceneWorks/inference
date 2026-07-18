//! Shared CPU-only synthetic fixtures for Bernini unit tests.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_wan::config::TransformerConfig;
use candle_gen_wan::transformer::WanTransformer;

/// A tiny dense DiT config (dim 16 = 2 heads × head_dim 8, z16 in/out, patch (1,2,2)).
pub(crate) fn tiny_cfg() -> TransformerConfig {
    TransformerConfig {
        in_channels: 16,
        out_channels: 16,
        num_layers: 2,
        num_heads: 2,
        head_dim: 8,
        dim: 16,
        ffn_dim: 32,
        freq_dim: 16,
        text_dim: 16,
        patch: (1, 2, 2),
        eps: 1e-6,
        rope_theta: 10000.0,
        rope_max_seq_len: 64,
    }
}

/// A randomly initialized tiny DiT that exercises packed geometry without real weights.
pub(crate) fn tiny_dit(cfg: &TransformerConfig, dev: &Device) -> WanTransformer {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut put = |key: &str, shape: &[usize]| {
        tensors.insert(
            key.to_string(),
            Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
        );
    };
    let (pt, ph, pw) = cfg.patch;
    let d = cfg.dim;
    put("patch_embedding.weight", &[d, cfg.in_channels, pt, ph, pw]);
    put("patch_embedding.bias", &[d]);
    put(
        "condition_embedder.text_embedder.linear_1.weight",
        &[d, cfg.text_dim],
    );
    put("condition_embedder.text_embedder.linear_1.bias", &[d]);
    put("condition_embedder.text_embedder.linear_2.weight", &[d, d]);
    put("condition_embedder.text_embedder.linear_2.bias", &[d]);
    put(
        "condition_embedder.time_embedder.linear_1.weight",
        &[d, cfg.freq_dim],
    );
    put("condition_embedder.time_embedder.linear_1.bias", &[d]);
    put("condition_embedder.time_embedder.linear_2.weight", &[d, d]);
    put("condition_embedder.time_embedder.linear_2.bias", &[d]);
    put("condition_embedder.time_proj.weight", &[6 * d, d]);
    put("condition_embedder.time_proj.bias", &[6 * d]);
    for i in 0..cfg.num_layers {
        let block = format!("blocks.{i}");
        put(&format!("{block}.scale_shift_table"), &[1, 6, d]);
        for attn in ["attn1", "attn2"] {
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                put(&format!("{block}.{attn}.{leaf}.weight"), &[d, d]);
                put(&format!("{block}.{attn}.{leaf}.bias"), &[d]);
            }
            put(&format!("{block}.{attn}.norm_q.weight"), &[d]);
            put(&format!("{block}.{attn}.norm_k.weight"), &[d]);
        }
        put(&format!("{block}.norm2.weight"), &[d]);
        put(&format!("{block}.norm2.bias"), &[d]);
        put(&format!("{block}.ffn.net.0.proj.weight"), &[cfg.ffn_dim, d]);
        put(&format!("{block}.ffn.net.0.proj.bias"), &[cfg.ffn_dim]);
        put(&format!("{block}.ffn.net.2.weight"), &[d, cfg.ffn_dim]);
        put(&format!("{block}.ffn.net.2.bias"), &[d]);
    }
    put("proj_out.weight", &[cfg.out_channels * pt * ph * pw, d]);
    put("proj_out.bias", &[cfg.out_channels * pt * ph * pw]);
    put("scale_shift_table", &[1, 2, d]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    WanTransformer::new(cfg, vb).unwrap()
}
