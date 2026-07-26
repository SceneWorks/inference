//! Krea Realtime 14B transformer **load path** (sc-8435 S2, non-gated).
//!
//! Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight, so once
//! [`sanitize_krea_realtime_transformer`] has collapsed either on-disk layout to the internal Wan DiT
//! key names it loads straight into the reused [`mlx_gen_wan::WanTransformer`] via its
//! [`from_weights`](mlx_gen_wan::WanTransformer::from_weights). This module wraps that with an
//! **explicit completeness + shape check** ([`verify_transformer_tensors`]) so a truncated shard, a
//! stray extra tensor, or a wrong-shape weight fails loudly here — with a diff — instead of surfacing
//! as an opaque `require` error deep inside `from_weights` (or, worse, a silent mis-load).
//!
//! The expected internal tensor set + shapes are derived **purely from the config**
//! ([`expected_transformer_tensors`]), so the check is exact for any Wan geometry and needs no real
//! checkpoint. This is the non-gated S2 surface: `tests/` validate it against the S1 inventory with
//! synthesized fixtures; real-weight byte parity is the gated remainder on sc-8435.

use std::collections::{HashMap, HashSet};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::WanTransformer;
use mlx_rs::Array;

use crate::config::KreaRealtimeConfig;
use crate::convert::sanitize_krea_realtime_transformer;

/// One expected **internal** (post-[`sanitize_krea_realtime_transformer`]) transformer tensor: the key
/// [`mlx_gen_wan::WanTransformer::from_weights`] reads and the config-derived shape it must carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// The internal tensor key (e.g. `blocks.0.self_attn.q.weight`, `patch_embedding_proj.weight`).
    pub name: String,
    /// The exact shape the tensor must have, derived from the [`WanModelConfig`].
    pub shape: Vec<i32>,
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: &[i32]) -> Self {
        Self {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }
}

/// Every internal transformer tensor Krea Realtime's reused Wan DiT expects, with its config-derived
/// shape. This is the **post-sanitize** layout (`patch_embedding_proj`, `text_embedding_{0,1}`,
/// `time_projection`, `ffn.fc1`/`fc2`, `head.head`) — exactly what
/// [`mlx_gen_wan::WanTransformer::from_weights`] reads — not the native on-disk names.
///
/// Shapes for the shipped `wan21_t2v_14b` geometry (`dim=5120`, `ffn=13824`, `text_dim=4096`,
/// `freq_dim=256`, `in/out=16`, `patch=(1,2,2)`, `40` layers): `self_attn.q [5120,5120]`,
/// `ffn.fc1 [13824,5120]`, `patch_embedding_proj [5120,64]`, `head.head [64,5120]`,
/// `time_projection [30720,5120]`, `text_embedding_0 [5120,4096]`.
pub fn expected_transformer_tensors(cfg: &WanModelConfig) -> Vec<TensorSpec> {
    let dim = cfg.dim as i32;
    let ffn = cfg.ffn_dim as i32;
    let text_dim = cfg.text_dim as i32;
    let freq = cfg.freq_dim as i32;
    let (pt, ph, pw) = cfg.patch_size;
    // The patch-embed conv `[dim, in, pt, ph, pw]` flattens to a Linear `[dim, in·∏patch]`; the head
    // projects `dim → out·∏patch`.
    let patch_cols = (cfg.in_dim * pt * ph * pw) as i32;
    let head_out = (cfg.out_dim * pt * ph * pw) as i32;

    let mut specs = Vec::with_capacity(15 + cfg.num_layers * 27);

    // Patch embedding (conv→Linear reshape).
    specs.push(TensorSpec::new(
        "patch_embedding_proj.weight",
        &[dim, patch_cols],
    ));
    specs.push(TensorSpec::new("patch_embedding_proj.bias", &[dim]));

    // Text embedding Sequential: Linear(text_dim→dim), GELU, Linear(dim→dim).
    specs.push(TensorSpec::new("text_embedding_0.weight", &[dim, text_dim]));
    specs.push(TensorSpec::new("text_embedding_0.bias", &[dim]));
    specs.push(TensorSpec::new("text_embedding_1.weight", &[dim, dim]));
    specs.push(TensorSpec::new("text_embedding_1.bias", &[dim]));

    // Time embedding Sequential: Linear(freq_dim→dim), SiLU, Linear(dim→dim).
    specs.push(TensorSpec::new("time_embedding_0.weight", &[dim, freq]));
    specs.push(TensorSpec::new("time_embedding_0.bias", &[dim]));
    specs.push(TensorSpec::new("time_embedding_1.weight", &[dim, dim]));
    specs.push(TensorSpec::new("time_embedding_1.bias", &[dim]));

    // Time projection: Linear(dim→6·dim) (the six modulation vectors).
    specs.push(TensorSpec::new("time_projection.weight", &[6 * dim, dim]));
    specs.push(TensorSpec::new("time_projection.bias", &[6 * dim]));

    // Output head: modulated LayerNorm table + projection dim→out·∏patch.
    specs.push(TensorSpec::new("head.modulation", &[1, 2, dim]));
    specs.push(TensorSpec::new("head.head.weight", &[head_out, dim]));
    specs.push(TensorSpec::new("head.head.bias", &[head_out]));

    for i in 0..cfg.num_layers {
        let p = format!("blocks.{i}");
        // Per-block 6-vector modulation table.
        specs.push(TensorSpec::new(format!("{p}.modulation"), &[1, 6, dim]));
        // Self- and cross-attention: q/k/v/o Linears (dim→dim) + full-dim qk-RMSNorm weights.
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                specs.push(TensorSpec::new(
                    format!("{p}.{attn}.{proj}.weight"),
                    &[dim, dim],
                ));
                specs.push(TensorSpec::new(format!("{p}.{attn}.{proj}.bias"), &[dim]));
            }
            specs.push(TensorSpec::new(format!("{p}.{attn}.norm_q.weight"), &[dim]));
            specs.push(TensorSpec::new(format!("{p}.{attn}.norm_k.weight"), &[dim]));
        }
        // Cross-attention pre-norm (affine LayerNorm).
        specs.push(TensorSpec::new(format!("{p}.norm3.weight"), &[dim]));
        specs.push(TensorSpec::new(format!("{p}.norm3.bias"), &[dim]));
        // FFN: Linear(dim→ffn), GELU, Linear(ffn→dim).
        specs.push(TensorSpec::new(format!("{p}.ffn.fc1.weight"), &[ffn, dim]));
        specs.push(TensorSpec::new(format!("{p}.ffn.fc1.bias"), &[ffn]));
        specs.push(TensorSpec::new(format!("{p}.ffn.fc2.weight"), &[dim, ffn]));
        specs.push(TensorSpec::new(format!("{p}.ffn.fc2.bias"), &[dim]));
    }

    specs
}

/// Cap a long list in an error message so a fully-missing map does not print thousands of lines.
fn preview(items: &[String]) -> String {
    const MAX: usize = 12;
    if items.len() <= MAX {
        items.join(", ")
    } else {
        format!(
            "{} … (+{} more)",
            items[..MAX].join(", "),
            items.len() - MAX
        )
    }
}

/// Assert `map` (an already-[`sanitize_krea_realtime_transformer`]d internal weight map) contains
/// **exactly** the tensors [`expected_transformer_tensors`] derives from `cfg`, each at its exact
/// shape — no missing, no extra, no wrong shape. On any discrepancy returns a single [`Error::Msg`]
/// summarizing the (capped) missing / extra / mis-shaped keys. Shape checks read only tensor metadata,
/// so this never forces MLX to materialize the (lazy) buffers.
pub fn verify_transformer_tensors(
    map: &HashMap<String, Array>,
    cfg: &WanModelConfig,
) -> Result<()> {
    let expected = expected_transformer_tensors(cfg);
    let expected_names: HashSet<&str> = expected.iter().map(|s| s.name.as_str()).collect();

    let mut missing = Vec::new();
    let mut mis_shape = Vec::new();
    for spec in &expected {
        match map.get(&spec.name) {
            None => missing.push(spec.name.clone()),
            Some(tensor) => {
                if tensor.shape() != spec.shape.as_slice() {
                    mis_shape.push(format!(
                        "{} (want {:?}, got {:?})",
                        spec.name,
                        spec.shape,
                        tensor.shape()
                    ));
                }
            }
        }
    }

    let mut extra: Vec<String> = map
        .keys()
        .filter(|k| !expected_names.contains(k.as_str()))
        .cloned()
        .collect();

    if missing.is_empty() && extra.is_empty() && mis_shape.is_empty() {
        return Ok(());
    }

    missing.sort();
    extra.sort();
    mis_shape.sort();
    Err(Error::Msg(format!(
        "krea-realtime: transformer tensor set does not match the wan21_t2v_14b inventory \
         (expected {} tensors): {} missing [{}], {} extra [{}], {} wrong-shape [{}]",
        expected.len(),
        missing.len(),
        preview(&missing),
        extra.len(),
        preview(&extra),
        mis_shape.len(),
        preview(&mis_shape),
    )))
}

/// Load a native Krea Realtime 14B transformer weight map (either on-disk layout — single-file
/// `model.`-prefixed or sharded `transformer/` bare) into the reused [`mlx_gen_wan::WanTransformer`].
///
/// The pipeline is: [`sanitize_krea_realtime_transformer`] (normalize the layout, map onto the
/// internal Wan DiT names, cast F16 → bf16) → [`verify_transformer_tensors`] (assert the full,
/// exactly-shaped inventory is present) → [`mlx_gen_wan::WanTransformer::from_weights`]. The TE / VAE /
/// tokenizer are stock Wan and provisioned separately (Krea Realtime ships transformer-only), so this
/// loads the DiT only. No inference, causal attention, KV cache, or scheduler — those AR pieces are
/// S3–S5.
pub fn load_krea_realtime_transformer(
    raw: HashMap<String, Array>,
    cfg: &KreaRealtimeConfig,
) -> Result<WanTransformer> {
    let sanitized = sanitize_krea_realtime_transformer(raw)?;
    verify_transformer_tensors(&sanitized, &cfg.wan)?;
    let weights = Weights::from_map(sanitized);
    WanTransformer::from_weights(&weights, &cfg.wan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_has_1095_tensors_at_audit_shapes() {
        let cfg = WanModelConfig::wan21_t2v_14b();
        let specs = expected_transformer_tensors(&cfg);
        // S1 audit: 1095 parameter tensors (the `freqs` RoPE buffer is dropped on sanitize).
        assert_eq!(specs.len(), 1095, "Wan2.1-14B transformer parameter count");

        let by_name: HashMap<&str, &[i32]> = specs
            .iter()
            .map(|s| (s.name.as_str(), s.shape.as_slice()))
            .collect();
        // The representative shapes from the S1 audit.
        assert_eq!(by_name["blocks.0.self_attn.q.weight"], &[5120, 5120]);
        assert_eq!(by_name["blocks.0.ffn.fc1.weight"], &[13824, 5120]);
        assert_eq!(by_name["patch_embedding_proj.weight"], &[5120, 64]);
        assert_eq!(by_name["head.head.weight"], &[64, 5120]);
        assert_eq!(by_name["time_projection.weight"], &[30720, 5120]);
        assert_eq!(by_name["text_embedding_0.weight"], &[5120, 4096]);
        assert_eq!(by_name["blocks.39.cross_attn.o.weight"], &[5120, 5120]);
    }

    #[test]
    fn inventory_scales_with_layer_count() {
        // Discriminating: the per-layer block (27 tensors) must actually be emitted per layer.
        let mut two = WanModelConfig::wan21_t2v_14b();
        two.num_layers = 2;
        let mut three = WanModelConfig::wan21_t2v_14b();
        three.num_layers = 3;
        let d =
            expected_transformer_tensors(&three).len() - expected_transformer_tensors(&two).len();
        assert_eq!(d, 27, "each transformer block contributes 27 tensors");
    }
}
