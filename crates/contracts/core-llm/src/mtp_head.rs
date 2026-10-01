//! Companion multi-token-prediction heads (epic sc-24432, story sc-24444): the backend-neutral
//! contract a standalone Qwen3.8 MTP proposal head is checked against before a backend builds it.
//!
//! A companion head (e.g. `EigenLabs/Qwen3.8-27B-MTP-4bit`) is a directory holding a
//! `config.json` with `model_type` [`COMPANION_MTP_MODEL_TYPE`] and the predictor layer's
//! safetensors — no embeddings and no LM head: it borrows the target's. Both backends read the
//! head's config through [`read_companion_mtp_config`], project it and the target's config onto
//! [`CompanionMtpGeometry`] and refuse the head when [`CompanionMtpGeometry::mismatches`] is
//! non-empty, then check every stored tensor against
//! [`matrices`](CompanionMtpGeometry::matrices) / [`norms`](CompanionMtpGeometry::norms) — one
//! check, so the two backends accept and refuse exactly the same heads (E8).

use std::path::Path;

use crate::error::{Error, Result};

/// The `model_type` of a standalone Qwen3.8 MTP proposal head: one predictor layer, no
/// embeddings, no LM head.
pub const COMPANION_MTP_MODEL_TYPE: &str = "qwen3_5_mtp";

/// Read a companion head's `config.json` from `dir` and require `model_type`
/// [`COMPANION_MTP_MODEL_TYPE`]. The returned value is the backend's to parse (its Qwen3.5
/// config reader resolves `text_config` nesting and the partial-rotary width).
pub fn read_companion_mtp_config(dir: &Path) -> Result<serde_json::Value> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Load(format!("read {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Load(format!("parse {}: {e}", path.display())))?;
    let model_type = value.get("model_type").and_then(|v| v.as_str());
    if model_type != Some(COMPANION_MTP_MODEL_TYPE) {
        return Err(Error::Load(format!(
            "companion MTP head `model_type` must be `{COMPANION_MTP_MODEL_TYPE}`, got {}",
            model_type.unwrap_or("none")
        )));
    }
    Ok(value)
}

/// Everything the one-layer Qwen3.8 predictor computes with, as a head's or a target's config
/// declares it. The predictor runs inside the target's residual stream, RoPE and vocabulary, so
/// a head is usable only when its geometry equals the target's.
#[derive(Clone, Debug, PartialEq)]
pub struct CompanionMtpGeometry {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    /// The only guard against a head trained for another tokenizer.
    pub vocab_size: i32,
    /// The partial-rotary width (`head_dim · partial_rotary_factor`).
    pub rotary_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    /// The head's own predictor-layer count (a target's is irrelevant: it may have none).
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    /// Whether the config declares a sparse-MoE FFN.
    pub moe: bool,
}

impl CompanionMtpGeometry {
    /// Every disagreement between this head and `target`, each named with both values — not just
    /// the first. Empty when the head fits.
    pub fn mismatches(&self, target: &Self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, head, target) in [
            ("hidden_size", self.hidden_size, target.hidden_size),
            (
                "num_attention_heads",
                self.num_attention_heads,
                target.num_attention_heads,
            ),
            (
                "num_key_value_heads",
                self.num_key_value_heads,
                target.num_key_value_heads,
            ),
            ("head_dim", self.head_dim, target.head_dim),
            (
                "intermediate_size",
                self.intermediate_size,
                target.intermediate_size,
            ),
            ("vocab_size", self.vocab_size, target.vocab_size),
            ("rotary_dim", self.rotary_dim, target.rotary_dim),
        ] {
            if head != target {
                out.push(format!("{name} {head} != target {target}"));
            }
        }
        if self.mtp_num_hidden_layers != 1 {
            out.push(format!(
                "mtp_num_hidden_layers {} (this runtime runs exactly one predictor layer)",
                self.mtp_num_hidden_layers
            ));
        }
        for (name, head, target) in [
            ("rms_norm_eps", self.rms_norm_eps, target.rms_norm_eps),
            ("rope_theta", self.rope_theta, target.rope_theta),
        ] {
            if head != target {
                out.push(format!("{name} {head} != target {target}"));
            }
        }
        if self.mrope_section != target.mrope_section {
            out.push(format!(
                "mrope_section {:?} != target {:?}",
                self.mrope_section, target.mrope_section
            ));
        }
        if self.mtp_use_dedicated_embeddings {
            out.push(
                "mtp_use_dedicated_embeddings is true (the head must share the target's)".into(),
            );
        }
        if self.moe {
            out.push("the head declares a MoE predictor".into());
        }
        out
    }

    /// The `[out, in]` of every projection a dense predictor layer stores, by bare tensor stem
    /// (a backend reads `{prefix}{stem}.weight`, plus its `.scales` / `.biases` when quantized).
    pub fn matrices(&self) -> [(&'static str, [usize; 2]); 8] {
        let d = |v: i32| v.max(0) as usize;
        let (h, heads, kv, hd, inter) = (
            d(self.hidden_size),
            d(self.num_attention_heads),
            d(self.num_key_value_heads),
            d(self.head_dim),
            d(self.intermediate_size),
        );
        [
            ("fc", [h, 2 * h]),
            // The gated attention's query projection carries the output gate too (2×).
            ("layers.0.self_attn.q_proj", [2 * heads * hd, h]),
            ("layers.0.self_attn.k_proj", [kv * hd, h]),
            ("layers.0.self_attn.v_proj", [kv * hd, h]),
            ("layers.0.self_attn.o_proj", [h, heads * hd]),
            ("layers.0.mlp.gate_proj", [inter, h]),
            ("layers.0.mlp.up_proj", [inter, h]),
            ("layers.0.mlp.down_proj", [h, inter]),
        ]
    }

    /// The width of every RMSNorm vector the predictor stores, by bare tensor stem. Their values
    /// follow the zero-centred Qwen3.8 checkpoint convention: the runtime applies `1 + w`.
    pub fn norms(&self) -> [(&'static str, usize); 7] {
        let (h, hd) = (
            self.hidden_size.max(0) as usize,
            self.head_dim.max(0) as usize,
        );
        [
            ("pre_fc_norm_embedding", h),
            ("pre_fc_norm_hidden", h),
            ("norm", h),
            ("layers.0.input_layernorm", h),
            ("layers.0.post_attention_layernorm", h),
            ("layers.0.self_attn.q_norm", hd),
            ("layers.0.self_attn.k_norm", hd),
        ]
    }
}

/// The tensor-name prefix of a head's stored tensors: bare (`fc.weight`, the published layout) or
/// `mtp.`-prefixed. `contains` answers whether the head stores a tensor name.
pub fn companion_mtp_prefix(contains: impl Fn(&str) -> bool) -> Result<&'static str> {
    match (contains("fc.weight"), contains("mtp.fc.weight")) {
        (true, false) => Ok(""),
        (false, true) => Ok("mtp."),
        (true, true) => Err(Error::Load(
            "companion MTP head has both bare and `mtp.`-prefixed tensors".into(),
        )),
        (false, false) => Err(Error::Load(
            "companion MTP head stores no `fc.weight`".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> CompanionMtpGeometry {
        CompanionMtpGeometry {
            hidden_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 64,
            intermediate_size: 256,
            vocab_size: 64,
            rotary_dim: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10],
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            moe: false,
        }
    }

    fn head() -> CompanionMtpGeometry {
        CompanionMtpGeometry {
            mtp_num_hidden_layers: 1,
            ..target()
        }
    }

    /// Every field the predictor computes with is checked and named with both values; a head
    /// matching the target (its own layer count aside) has no mismatch.
    #[test]
    fn every_geometry_field_is_checked_and_named() {
        assert!(head().mismatches(&target()).is_empty());
        type Edit = fn(&mut CompanionMtpGeometry);
        let cases: [(Edit, &str); 13] = [
            (|g| g.hidden_size = 256, "hidden_size 256 != target 128"),
            (
                |g| g.num_attention_heads = 4,
                "num_attention_heads 4 != target 2",
            ),
            (
                |g| g.num_key_value_heads = 2,
                "num_key_value_heads 2 != target 1",
            ),
            (|g| g.head_dim = 32, "head_dim 32 != target 64"),
            (
                |g| g.intermediate_size = 128,
                "intermediate_size 128 != target 256",
            ),
            (|g| g.vocab_size = 65, "vocab_size 65 != target 64"),
            (|g| g.rotary_dim = 64, "rotary_dim 64 != target 32"),
            (
                |g| g.rms_norm_eps = 1e-5,
                "rms_norm_eps 0.00001 != target 0.000001",
            ),
            (
                |g| g.rope_theta = 1e6,
                "rope_theta 1000000 != target 10000000",
            ),
            (
                |g| g.mrope_section = [8, 12, 12],
                "mrope_section [8, 12, 12] != target [11, 11, 10]",
            ),
            (|g| g.mtp_num_hidden_layers = 2, "mtp_num_hidden_layers 2"),
            (
                |g| g.mtp_use_dedicated_embeddings = true,
                "mtp_use_dedicated_embeddings is true",
            ),
            (|g| g.moe = true, "the head declares a MoE predictor"),
        ];
        for (mutate, named) in cases {
            let mut h = head();
            mutate(&mut h);
            let found = h.mismatches(&target());
            assert_eq!(found.len(), 1, "{found:?}");
            assert!(found[0].contains(named), "{found:?} lacks {named}");
        }
        // Several at once are all named.
        let mut h = head();
        h.vocab_size = 65;
        h.rope_theta = 1e6;
        assert_eq!(h.mismatches(&target()).len(), 2);
    }

    #[test]
    fn a_head_config_must_declare_the_companion_model_type() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_mtp"}"#).unwrap();
        assert!(read_companion_mtp_config(dir).is_ok());
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
        let err = read_companion_mtp_config(dir).unwrap_err().to_string();
        assert!(err.contains("must be `qwen3_5_mtp`, got qwen3_5"), "{err}");
        assert!(read_companion_mtp_config(&dir.join("absent")).is_err());
    }

    #[test]
    fn the_tensor_prefix_is_bare_or_mtp_never_both() {
        assert_eq!(companion_mtp_prefix(|k| k == "fc.weight").unwrap(), "");
        assert_eq!(
            companion_mtp_prefix(|k| k == "mtp.fc.weight").unwrap(),
            "mtp."
        );
        assert!(companion_mtp_prefix(|_| true).is_err());
        assert!(companion_mtp_prefix(|_| false).is_err());
    }
}
