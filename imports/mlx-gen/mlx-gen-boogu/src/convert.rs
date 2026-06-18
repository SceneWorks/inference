//! Boogu transformer "conversion" = **architecture validation**.
//!
//! The published diffusers checkpoint uses dotted keys that map 1:1 onto the
//! `BooguImageTransformer2DModel` module tree, so [`mlx_gen::weights::Weights::from_dir`] loads
//! them directly — there is no fork-style key remap (unlike FLUX.2's `to_out.0`→`to_out`). What we
//! *do* need is to prove the on-disk tensor set exactly matches the architecture implied by
//! [`BooguConfig`] before the DiT forward (E3) trusts it: every expected key present, no stray
//! extras, and the shape-bearing entry-points (patch embedders, caption embedder, FFN, out proj)
//! sized as the config says. This catches a wrong variant / truncated download / config-weight
//! mismatch loudly at load instead of as garbage latents.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::BooguConfig;

/// A non-modulated transformer block (context refiner): plain `norm1`/`norm2` RMSNorm.
fn block_keys_no_mod(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ffn_keys(&format!("{prefix}.feed_forward")));
    for n in ["ffn_norm1", "ffn_norm2", "norm1", "norm2"] {
        k.push(format!("{prefix}.{n}.weight"));
    }
    k
}

/// A modulated single-stream / refiner block: `norm1` is a `LuminaRMSNormZero`
/// (`linear.{weight,bias}` + `norm.weight`), `norm2` a plain RMSNorm.
fn block_keys_mod(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ffn_keys(&format!("{prefix}.feed_forward")));
    k.push(format!("{prefix}.ffn_norm1.weight"));
    k.push(format!("{prefix}.ffn_norm2.weight"));
    k.extend(lumina_rms_zero_keys(&format!("{prefix}.norm1")));
    k.push(format!("{prefix}.norm2.weight"));
    k
}

/// A double-stream (dual-stream) block: a joint instruct↔img attention whose QKV lives on the
/// processor, an image self-attention, two FFNs, three img modulations + two instruct modulations,
/// and the per-sublayer output RMSNorms.
fn double_block_keys(prefix: &str) -> Vec<String> {
    let mut k = Vec::new();
    // Joint attention: per-head q/k norm + the processor's own projections + the shared output.
    k.push(format!("{prefix}.img_instruct_attn.norm_q.weight"));
    k.push(format!("{prefix}.img_instruct_attn.norm_k.weight"));
    for side in ["img", "instruct"] {
        for p in ["to_q", "to_k", "to_v", "out"] {
            k.push(format!("{prefix}.img_instruct_attn.processor.{side}_{p}.weight"));
        }
    }
    k.push(format!("{prefix}.img_instruct_attn.to_out.0.weight"));
    // Image self-attention.
    k.extend(attn_keys(&format!("{prefix}.img_self_attn")));
    // FFNs.
    k.extend(ffn_keys(&format!("{prefix}.img_feed_forward")));
    k.extend(ffn_keys(&format!("{prefix}.instruct_feed_forward")));
    // Modulations.
    for n in ["img_norm1", "img_norm2", "img_norm3", "instruct_norm1", "instruct_norm2"] {
        k.extend(lumina_rms_zero_keys(&format!("{prefix}.{n}")));
    }
    // Output RMSNorms.
    for n in [
        "img_attn_norm",
        "img_self_attn_norm",
        "img_ffn_norm1",
        "img_ffn_norm2",
        "instruct_attn_norm",
        "instruct_ffn_norm1",
        "instruct_ffn_norm2",
    ] {
        k.push(format!("{prefix}.{n}.weight"));
    }
    k
}

/// GQA attention with per-head q/k RMSNorm and a `to_out` Sequential (`to_out.0`).
fn attn_keys(prefix: &str) -> Vec<String> {
    ["norm_q", "norm_k", "to_q", "to_k", "to_v", "to_out.0"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// SwiGLU feed-forward (`linear_1`/`linear_3` in, `linear_2` out), all bias-free.
fn ffn_keys(prefix: &str) -> Vec<String> {
    ["linear_1", "linear_2", "linear_3"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// `LuminaRMSNormZero`: a SiLU→Linear modulation (`linear.weight`+`linear.bias`) plus the RMSNorm
/// (`norm.weight`).
fn lumina_rms_zero_keys(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}.linear.weight"),
        format!("{prefix}.linear.bias"),
        format!("{prefix}.norm.weight"),
    ]
}

/// The complete set of transformer tensor keys implied by `cfg`.
pub fn expected_transformer_keys(cfg: &BooguConfig) -> Vec<String> {
    let mut keys = Vec::new();

    // Embedders.
    for e in ["x_embedder", "ref_image_patch_embedder"] {
        keys.push(format!("{e}.weight"));
        keys.push(format!("{e}.bias"));
    }
    keys.push("image_index_embedding".to_string());

    // Time + caption embedding.
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("time_caption_embed.timestep_embedder.{n}.weight"));
        keys.push(format!("time_caption_embed.timestep_embedder.{n}.bias"));
    }
    keys.push("time_caption_embed.caption_embedder.0.weight".to_string()); // RMSNorm
    keys.push("time_caption_embed.caption_embedder.1.weight".to_string()); // Linear
    keys.push("time_caption_embed.caption_embedder.1.bias".to_string());

    // Refiners (context = no modulation; noise + ref-image = modulated).
    for i in 0..cfg.num_refiner_layers {
        keys.extend(block_keys_no_mod(&format!("context_refiner.{i}")));
        keys.extend(block_keys_mod(&format!("noise_refiner.{i}")));
        keys.extend(block_keys_mod(&format!("ref_image_refiner.{i}")));
    }

    // Dual-stream then single-stream stacks.
    for i in 0..cfg.num_double_stream_layers {
        keys.extend(double_block_keys(&format!("double_stream_layers.{i}")));
    }
    for i in 0..cfg.num_single_stream_layers() {
        keys.extend(block_keys_mod(&format!("single_stream_layers.{i}")));
    }

    // Continuous-AdaLN output projection (LuminaLayerNormContinuous).
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("norm_out.{n}.weight"));
        keys.push(format!("norm_out.{n}.bias"));
    }

    keys
}

/// Validate a loaded transformer against `cfg`: exact key coverage (no missing, no extra) and the
/// shapes of the dimension-bearing entry points.
pub fn validate_transformer(w: &Weights, cfg: &BooguConfig) -> Result<()> {
    use std::collections::BTreeSet;

    let expected: BTreeSet<String> = expected_transformer_keys(cfg).into_iter().collect();
    let actual: BTreeSet<String> = w.keys().map(str::to_string).collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    if !missing.is_empty() || !extra.is_empty() {
        let head = |v: &[&String]| {
            v.iter().take(8).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        };
        return Err(Error::Msg(format!(
            "boogu transformer key mismatch vs config: {} missing [{}], {} extra [{}]",
            missing.len(),
            head(&missing),
            extra.len(),
            head(&extra),
        )));
    }

    // Shape checks on the dimension-bearing tensors (Linear weight = [out, in]).
    let h = cfg.hidden_size as i32;
    check_shape(w, "x_embedder.weight", &[h, cfg.patch_in_dim() as i32])?;
    check_shape(w, "norm_out.linear_2.weight", &[cfg.patch_out_dim() as i32, h])?;
    check_shape(
        w,
        "time_caption_embed.caption_embedder.1.weight",
        &[h, cfg.preprocessed_instruction_feat_dim() as i32],
    )?;
    check_shape(
        w,
        "time_caption_embed.timestep_embedder.linear_1.weight",
        &[cfg.modulation_dim() as i32, 256],
    )?;
    // A representative SwiGLU FFN: validates the multiple_of rounding.
    check_shape(
        w,
        "single_stream_layers.0.feed_forward.linear_1.weight",
        &[cfg.ffn_inner_dim() as i32, h],
    )?;
    // GQA: q projects to all heads, k/v to kv heads.
    let head_dim = cfg.head_dim() as i32;
    check_shape(w, "single_stream_layers.0.attn.to_q.weight", &[cfg.num_attention_heads as i32 * head_dim, h])?;
    check_shape(w, "single_stream_layers.0.attn.to_k.weight", &[cfg.num_kv_heads as i32 * head_dim, h])?;
    Ok(())
}

fn check_shape(w: &Weights, key: &str, expected: &[i32]) -> Result<()> {
    let t = w.require(key)?;
    if t.shape() != expected {
        return Err(Error::Msg(format!(
            "boogu: {key} shape {:?}, expected {:?}",
            t.shape(),
            expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_key_count_matches_published_base() {
        let cfg = BooguConfig::base();
        let keys = expected_transformer_keys(&cfg);
        let unique: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "no duplicate expected keys");
        // 26 (context×2) + 30 (noise×2) + 30 (ref×2) + 360 (double×8) + 480 (single×32) + 16 top-level.
        assert_eq!(keys.len(), 942);
    }

    /// Real-weight architecture validation against the published Base snapshot.
    /// `BOOGU_BASE_DIR=<snapshot root>` (the dir containing `transformer/`).
    #[test]
    #[ignore = "needs real weights: set BOOGU_BASE_DIR"]
    fn validate_real_base_snapshot() {
        let root = std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR");
        let cfg = BooguConfig::from_snapshot(&root).unwrap();
        assert_eq!(cfg, BooguConfig::base());
        let w = Weights::from_dir(format!("{root}/transformer")).unwrap();
        validate_transformer(&w, &cfg).unwrap();
        assert_eq!(w.len(), 942);
    }
}
