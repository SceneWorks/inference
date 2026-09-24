//! Real-weight Qwen3.6/3.8 helpers shared by the seam-parity suites (sc-24129): load a snapshot
//! directory into a `Qwen35Model` (+ its MTP head when complete) and render one chat turn to ids.
//! Every path is passed in by the caller; nothing is derived from a cache location.

#![allow(dead_code)]

use std::path::Path;

use candle_core::Device;
use candle_llm::models::{Qwen35Config, Qwen35Model, Qwen35Mtp};
use candle_llm::primitives::Weights;
use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
use serde_json::Value;

/// The snapshot directory named by `var`, or `None` (the caller skips or panics as its gate says).
pub fn snapshot_from_env(var: &str) -> Option<std::path::PathBuf> {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(Into::into)
}

/// Load the decoder (dense, device compute dtype) and, when the checkpoint carries the complete
/// native MTP layout, the predictor.
pub fn load(snapshot: &Path, device: &Device) -> (Qwen35Model, Option<Qwen35Mtp>) {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let cfg = Qwen35Config::from_json(&config).expect("qwen3_5 config");
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    let prefix = if weights.contains("model.language_model.embed_tokens.weight") {
        "model.language_model"
    } else {
        "model"
    };
    let model = Qwen35Model::from_weights(&weights, prefix, cfg.clone()).expect("build model");
    let mtp = (cfg.mtp_num_hidden_layers > 0 && Qwen35Mtp::complete_in(&weights, &cfg))
        .then(|| Qwen35Mtp::from_weights_with(&weights, &model, None).expect("build mtp"));
    (model, mtp)
}

/// Render one user turn through the snapshot's chat template (the sidecar `chat_template.jinja`,
/// else the key embedded in `tokenizer_config.json`) and tokenize it.
pub fn render_chat_prompt(snapshot: &Path, user: &str) -> Vec<i32> {
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer.json");
    let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
        Ok(source) if !source.trim().is_empty() => JinjaChatTemplate::new(source),
        _ => JinjaChatTemplate::from_tokenizer_config_file(snapshot.join("tokenizer_config.json"))
            .expect("tokenizer_config.json chat template"),
    };
    let rendered = template
        .render_with(&[Message::user(user)], &RenderOptions::generation())
        .expect("render prompt");
    tokenizer
        .encode(&rendered, false)
        .expect("encode prompt")
        .into_iter()
        .map(|id| id as i32)
        .collect()
}
