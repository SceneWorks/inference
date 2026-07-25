//! Loading the Qwen3-VL text encoder from a published `microsoft/Mage-Flow*` snapshot.
//!
//! There is deliberately **no crate-level `loader.rs`** (see the decision note in
//! [`crate`]): each component loads from inside the module that owns it, so the concurrent VAE and
//! DiT ports never touch a file this story owns.
//!
//! ```text
//! <root>/text_encoder/config.json              ← verified against the pinned shapes
//! <root>/text_encoder/tokenizer.json           ← Qwen2 fast tokenizer
//! <root>/text_encoder/model-0000{1,2}-of-00002.safetensors
//! ```
//!
//! The `text_encoder/` directory is **bit-identical across all six repos** (8.875 GB), so one
//! resident encoder serves every variant — the co-requisite modelling sc-14047 relies on.
//!
//! The directory also holds the Qwen3-VL **vision** tower (`model.visual.*`, ≈0.9 GB of the total).
//! [`load_lm`] does not reference those tensors, and the `Weights` map that held them is dropped
//! before it returns, so a text-to-image load does not retain them. sc-14048 loads them separately
//! for the edit path.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_boogu::{VisionConfig, VisionTower};

use crate::config::{
    max_prompt_tokens, QwenVlTextConfig, DROP_IDX_EDIT, TE_HIDDEN_ACT, TE_RMS_NORM_EPS,
    TE_ROPE_THETA,
};

use super::encode::MageTextEncoder;
use super::encoder::Qwen3VlTextEncoder;

/// Component sub-directory of a snapshot root. There is **no `tokenizer/` folder** in these repos —
/// the processor files live beside the weights (epic ground truth).
pub const COMPONENT_DIR: &str = "text_encoder";

/// Weight-key prefix of the language model inside `text_encoder/`
/// (`model.language_model.embed_tokens.weight`, …). The vision tower is `model.visual.*`.
pub const LM_PREFIX: &str = "model.language_model";

/// Qwen2 pad token id, `<|endoftext|>` (`text_encoder/tokenizer_config.json`).
const PAD_TOKEN_ID: i32 = 151_643;

/// The only `rope_scaling.rope_type` this port implements. `"default"` selects the plain
/// `1/θ^(2i/head_dim)` inverse-frequency table with `attention_scaling == 1.0`; every other
/// `ROPE_INIT_FUNCTIONS` entry (`"yarn"`, `"linear"`, `"dynamic"`, `"longrope"`, …) rewrites those
/// on identical weights.
const ROPE_TYPE_DEFAULT: &str = "default";

/// Group size for group-wise-affine Q4/Q8 packing. Every quantizable projection in this LM has an
/// input dim divisible by 64 (2560 hidden, 4096 q, 1024 kv, 9728 FFN), so one group size covers the
/// whole encoder — the codebase default, matching the `mlx-gen-krea` / `mlx-gen-z-image` Qwen3
/// encoders. sc-14046 owns the packer and must pack at the same size.
pub(crate) const QUANT_GROUP_SIZE: i32 = 64;

/// Load the tokenizer from `<root>/text_encoder/tokenizer.json`.
///
/// [`ChatTemplate::None`]: this crate renders the Mage-Flow ChatML wrapper itself from
/// [`PROMPT_TEMPLATE_GEN`](crate::config::PROMPT_TEMPLATE_GEN) /
/// [`PROMPT_TEMPLATE_EDIT`](crate::config::PROMPT_TEMPLATE_EDIT), which are pinned against the
/// vendored `utils.py`. (The core [`ChatTemplate::QwenImage`] variant happens to spell the same
/// system prefix, because Qwen-Image ships the identical template — but letting the tokenizer wrap
/// the prompt would move the source of truth off the pinned constants and silently break the edit
/// template, which has no core equivalent.)
///
/// `pad_to_max_length` is off: the reference never pads, it packs
/// (`_vendor/mage_flow/models/modules/text_encoder.py:496-508`). Truncation is applied by
/// [`token_ids`](MageTextEncoder::token_ids) at the per-kind budget, so `max_length` here is only
/// the ceiling for the unused padded path and is set to the larger of the two.
pub fn load_tokenizer(root: impl AsRef<Path>) -> Result<TextTokenizer> {
    TextTokenizer::from_file(
        root.as_ref().join(COMPONENT_DIR).join("tokenizer.json"),
        TokenizerConfig {
            max_length: max_prompt_tokens(DROP_IDX_EDIT),
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(Into::into)
}

/// Load the language model from `<root>/text_encoder/*.safetensors`, after verifying the
/// snapshot's `config.json` describes the model this port implements.
pub fn load_lm(root: impl AsRef<Path>) -> Result<Qwen3VlTextEncoder> {
    let dir = root.as_ref().join(COMPONENT_DIR);
    let config_path = dir.join("config.json");
    let published = std::fs::read_to_string(&config_path).map_err(|e| {
        Error::Msg(format!(
            "mage_flow text encoder: read {}: {e}",
            config_path.display()
        ))
    })?;
    let cfg = verify_text_config(&published)?;
    let mut w = Weights::from_dir(&dir)?;
    let model = Qwen3VlTextEncoder::from_weights_draining(
        &mut w,
        LM_PREFIX,
        &cfg,
        TE_RMS_NORM_EPS,
        TE_ROPE_THETA,
    )?;
    let remaining_lm = w
        .keys()
        .filter(|key| key.starts_with(&format!("{LM_PREFIX}.")))
        .count();
    if remaining_lm != 0 {
        return Err(Error::Msg(format!(
            "mage_flow text encoder: drain left {remaining_lm} language-model source tensors resident"
        )));
    }
    Ok(model)
}

/// Load tokenizer + LM together.
pub fn load(root: impl AsRef<Path>) -> Result<MageTextEncoder> {
    let root = root.as_ref();
    Ok(MageTextEncoder::new(load_tokenizer(root)?, load_lm(root)?))
}

pub fn mage_vision_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 1024,
        num_heads: 16,
        intermediate_size: 4096,
        depth: 24,
        out_hidden_size: 2560,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        num_position_embeddings: 2304,
        deepstack_visual_indexes: vec![5, 11, 17],
    }
}

/// Load tokenizer, language model, and the Qwen3-VL vision tower needed by Mage-Flow-Edit.
pub fn load_multimodal(root: impl AsRef<Path>) -> Result<MageTextEncoder> {
    let root = root.as_ref();
    let dir = root.join(COMPONENT_DIR);
    let weights = Weights::from_dir(&dir)?;
    let vision = VisionTower::from_weights(&weights, mage_vision_config(), "model.visual")?;
    Ok(MageTextEncoder::new_multimodal(
        load_tokenizer(root)?,
        load_lm(root)?,
        vision,
    ))
}

/// Check a `text_encoder/config.json` body against the pinned Qwen3-VL-4B shapes and return them.
///
/// The shapes are **verified, not read**. Reading them would let a 8B `text_config` load "fine" and
/// then fail deep inside `Weights::require` with an opaque key error — or worse, load a variant
/// with the same key names and a different `head_dim` and produce silently wrong conditioning.
/// Every field this port depends on is checked, including the ones with no weight-shape
/// consequence (`rope_scaling.rope_type`, `rope_scaling.mrope_interleaved`, `mrope_section`,
/// `attention_bias`, `hidden_act`, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`), which are
/// exactly the ones a shape mismatch would *not* catch.
pub fn verify_text_config(json: &str) -> Result<QwenVlTextConfig> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        Error::Msg(format!(
            "mage_flow text encoder: config.json is invalid: {e}"
        ))
    })?;
    let tc = v.get("text_config").ok_or_else(|| {
        Error::Msg("mage_flow text encoder: config.json has no 'text_config'".into())
    })?;
    let want = QwenVlTextConfig::mage_flow();

    let check_i = |key: &str, expected: i64| -> Result<()> {
        let found = tc.get(key).and_then(serde_json::Value::as_i64);
        if found == Some(expected) {
            Ok(())
        } else {
            Err(mismatch(key, &expected.to_string(), tc.get(key)))
        }
    };
    check_i("hidden_size", want.hidden_size as i64)?;
    check_i("num_hidden_layers", want.num_layers as i64)?;
    check_i("num_attention_heads", want.num_attention_heads as i64)?;
    check_i("num_key_value_heads", want.num_key_value_heads as i64)?;
    check_i("head_dim", want.head_dim as i64)?;
    check_i("intermediate_size", want.intermediate_size as i64)?;
    check_i("vocab_size", want.vocab_size as i64)?;

    let bool_of = |key: &str| tc.get(key).and_then(serde_json::Value::as_bool);
    if bool_of("attention_bias") != Some(want.attention_bias) {
        return Err(mismatch(
            "attention_bias",
            &want.attention_bias.to_string(),
            tc.get("attention_bias"),
        ));
    }
    if bool_of("tie_word_embeddings") != Some(want.tie_word_embeddings) {
        return Err(mismatch(
            "tie_word_embeddings",
            &want.tie_word_embeddings.to_string(),
            tc.get("tie_word_embeddings"),
        ));
    }
    if tc.get("hidden_act").and_then(serde_json::Value::as_str) != Some(TE_HIDDEN_ACT) {
        return Err(mismatch("hidden_act", TE_HIDDEN_ACT, tc.get("hidden_act")));
    }
    let f64_of = |key: &str| tc.get(key).and_then(serde_json::Value::as_f64);
    // Compare at f32: the constant is an `f32` and `1e-6f32 as f64` is 9.9999999747e-7, which is
    // NOT the f64 the JSON `1e-06` parses to. Widening the constant instead of narrowing the file
    // value would reject the published config.
    if f64_of("rms_norm_eps").map(|found| found as f32) != Some(TE_RMS_NORM_EPS) {
        return Err(mismatch(
            "rms_norm_eps",
            &format!("{TE_RMS_NORM_EPS:e}"),
            tc.get("rms_norm_eps"),
        ));
    }
    if f64_of("rope_theta") != Some(TE_ROPE_THETA) {
        return Err(mismatch(
            "rope_theta",
            &TE_ROPE_THETA.to_string(),
            tc.get("rope_theta"),
        ));
    }

    // `rope_scaling` selects the rotary scheme; every entry here changes the frequency table on
    // otherwise identical weights, with no shape consequence at all.
    let scaling = tc.get("rope_scaling").ok_or_else(|| {
        Error::Msg("mage_flow text encoder: text_config has no 'rope_scaling'".into())
    })?;
    // `rope_type` picks the `ROPE_INIT_FUNCTIONS` entry that builds `inv_freq` **and** the
    // `attention_scaling` factor. A non-`"default"` value (`"yarn"`, `"linear"`, `"longrope"`, …)
    // rescales every rotation angle while loading perfectly cleanly — the same silent class as
    // `mrope_interleaved`, and worth extra care because this guard is the only control behind the
    // QK-norm epsilon the parity golden provably cannot separate.
    if scaling.get("rope_type").and_then(serde_json::Value::as_str) != Some(ROPE_TYPE_DEFAULT) {
        return Err(mismatch(
            "rope_scaling.rope_type",
            ROPE_TYPE_DEFAULT,
            scaling.get("rope_type"),
        ));
    }
    if scaling
        .get("mrope_interleaved")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(mismatch(
            "rope_scaling.mrope_interleaved",
            "true",
            scaling.get("mrope_interleaved"),
        ));
    }
    let section: Vec<i64> = scaling
        .get("mrope_section")
        .and_then(serde_json::Value::as_array)
        .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    let want_section: Vec<i64> = want.mrope_section.iter().map(|&n| n as i64).collect();
    if section != want_section {
        return Err(mismatch(
            "rope_scaling.mrope_section",
            &format!("{want_section:?}"),
            scaling.get("mrope_section"),
        ));
    }

    Ok(want)
}

fn mismatch(key: &str, expected: &str, found: Option<&serde_json::Value>) -> Error {
    Error::Msg(format!(
        "mage_flow text encoder: text_config '{key}' is {} but this port implements {expected} \
         (Qwen3-VL-4B as published by every microsoft/Mage-Flow* repo); loading a different \
         Qwen3-VL variant would produce silently wrong conditioning",
        found.map_or_else(|| "absent".to_string(), ToString::to_string),
    ))
}

/// Load a bias-less projection, auto-detecting a pre-quantized (packed) snapshot.
pub(crate) fn lin(w: &Weights, base: &str) -> Result<mlx_gen::adapters::AdaptableLinear> {
    mlx_gen::quant::lin(w, base, false, QUANT_GROUP_SIZE)
}

/// Load the token-embedding table, auto-detecting a packed snapshot.
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<mlx_gen::nn::TokenEmbedding> {
    mlx_gen::quant::embedding(w, base, QUANT_GROUP_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUBLISHED: &str = include_str!("../../tests/fixtures/text_encoder_config.json");

    #[test]
    fn the_published_config_verifies() {
        let cfg = verify_text_config(PUBLISHED).unwrap();
        assert_eq!(cfg, QwenVlTextConfig::mage_flow());
    }

    /// Mutate one key inside `text_config` (or a nested path under it) and re-serialise. Mutating
    /// the parsed tree rather than the text avoids the ambiguity that makes a string edit a false
    /// green here: `"hidden_size"` is a substring of `vision_config`'s `"out_hidden_size"`, and
    /// `"tie_word_embeddings"` appears both at the top level *and* inside `text_config`.
    fn mutate(path: &[&str], value: serde_json::Value) -> String {
        let mut v: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
        let mut node = v.get_mut("text_config").expect("text_config");
        for key in &path[..path.len() - 1] {
            node = node.get_mut(key).unwrap_or_else(|| panic!("no {key}"));
        }
        let leaf = path[path.len() - 1];
        let slot = node
            .get_mut(leaf)
            .unwrap_or_else(|| panic!("text_config has no {leaf}"));
        assert_ne!(*slot, value, "the mutation for {leaf} is a no-op");
        *slot = value;
        v.to_string()
    }

    /// Every checked key must be load-bearing: a drifting value must be rejected. A guard that
    /// silently accepted one would be the whole point of the function failing to fire.
    #[test]
    fn every_verified_key_rejects_a_drifting_value() {
        let cases: [(&[&str], serde_json::Value); 14] = [
            (&["hidden_size"], 4096.into()),
            (&["num_hidden_layers"], 48.into()),
            (&["num_attention_heads"], 40.into()),
            (&["num_key_value_heads"], 4.into()),
            (&["head_dim"], 80.into()),
            (&["intermediate_size"], 8192.into()),
            (&["vocab_size"], 152_064.into()),
            (&["attention_bias"], true.into()),
            (&["tie_word_embeddings"], false.into()),
            (&["hidden_act"], "gelu".into()),
            (&["rms_norm_eps"], 1e-5.into()),
            (&["rope_theta"], 1_000_000.into()),
            (&["rope_scaling", "mrope_interleaved"], false.into()),
            (&["rope_scaling", "rope_type"], "yarn".into()),
        ];
        for (path, value) in cases {
            let mutated = mutate(path, value.clone());
            assert!(
                verify_text_config(&mutated).is_err(),
                "a config declaring {path:?} = {value} was accepted"
            );
        }
        // The section widths must be checked as an ordered triple, not just a sum.
        assert!(verify_text_config(&mutate(
            &["rope_scaling", "mrope_section"],
            serde_json::json!([20, 24, 20])
        ))
        .is_err());
        assert!(verify_text_config(&mutate(
            &["rope_scaling", "mrope_section"],
            serde_json::json!([24, 20])
        ))
        .is_err());
    }

    /// `tie_word_embeddings` appears twice — top level and inside `text_config`. The guard must
    /// read the `text_config` one, so flipping ONLY the top-level copy must still verify.
    #[test]
    fn tie_word_embeddings_is_read_from_text_config_not_the_top_level() {
        let mut v: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
        assert_eq!(v["tie_word_embeddings"], serde_json::json!(true));
        assert_eq!(
            v["text_config"]["tie_word_embeddings"],
            serde_json::json!(true)
        );
        v["tie_word_embeddings"] = false.into();
        verify_text_config(&v.to_string())
            .expect("the top-level copy is not the one this port depends on");
    }

    /// The exact counter-example review used: a YaRN-scaled `rope_scaling` block. It changes every
    /// rotation angle and the `attention_scaling` factor with **zero** shape consequence, so
    /// without this guard it loaded cleanly and only failed later — as a confusing
    /// "no .safetensors files" error from `Weights::from_dir` rather than a config mismatch.
    #[test]
    fn a_yarn_scaled_rope_config_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
        v["text_config"]["rope_scaling"] = serde_json::json!({
            "rope_type": "yarn",
            "factor": 4.0,
            "mrope_interleaved": true,
            "mrope_section": [24, 20, 20],
        });
        let err = match verify_text_config(&v.to_string()) {
            Err(e) => e,
            Ok(_) => panic!("a YaRN-scaled rope_scaling block was accepted"),
        };
        assert!(
            format!("{err}").contains("rope_type"),
            "the error must name rope_type: {err}"
        );

        // …and `rope_type` is checked on its own, not merely as part of a whole-block swap.
        assert!(
            verify_text_config(&mutate(&["rope_scaling", "rope_type"], "linear".into())).is_err()
        );
        assert!(
            verify_text_config(&mutate(&["rope_scaling", "rope_type"], "longrope".into())).is_err()
        );
        // A missing or wrong-typed `rope_type` is a mismatch, not a skip.
        let mut absent: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
        absent["text_config"]["rope_scaling"]
            .as_object_mut()
            .unwrap()
            .remove("rope_type");
        assert!(verify_text_config(&absent.to_string()).is_err());
        assert!(verify_text_config(&mutate(&["rope_scaling", "rope_type"], 0.into())).is_err());
    }

    /// A key that is present but wrong-typed is a mismatch, not a skip — `"head_dim": "128"` must
    /// not slip past an `as_i64()` that quietly returns `None`.
    #[test]
    fn a_present_but_wrong_typed_value_is_rejected() {
        assert!(verify_text_config(&mutate(&["head_dim"], "128".into())).is_err());
        assert!(verify_text_config(&mutate(&["attention_bias"], 0.into())).is_err());
        assert!(verify_text_config(&mutate(&["hidden_act"], 1.into())).is_err());
    }

    #[test]
    fn a_config_without_text_config_is_rejected() {
        assert!(verify_text_config("{}").is_err());
        assert!(verify_text_config("not json").is_err());
    }
}
