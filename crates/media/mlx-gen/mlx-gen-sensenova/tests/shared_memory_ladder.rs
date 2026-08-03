//! Synthetic parity and cancellation proof for the SenseNova Gen-path memory seams.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, WeightsSource};
use mlx_gen_sensenova::{NeoChatConfig, Path, Qwen3Backbone};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/backbone_golden.safetensors"
);

fn config_from_meta(w: &Weights) -> NeoChatConfig {
    let m = |key: &str| w.metadata(key).unwrap();
    let llm = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": m("hidden_size").parse::<u64>().unwrap(),
        "intermediate_size": m("intermediate_size").parse::<u64>().unwrap(),
        "num_hidden_layers": m("num_hidden_layers").parse::<u64>().unwrap(),
        "num_attention_heads": m("num_attention_heads").parse::<u64>().unwrap(),
        "num_key_value_heads": m("num_key_value_heads").parse::<u64>().unwrap(),
        "head_dim": m("head_dim").parse::<u64>().unwrap(),
        "rms_norm_eps": m("rms_norm_eps").parse::<f64>().unwrap(),
        "rope_theta": m("rope_theta").parse::<f64>().unwrap(),
        "rope_theta_hw": m("rope_theta_hw").parse::<f64>().unwrap(),
        "vocab_size": m("vocab_size").parse::<u64>().unwrap(),
        "attention_bias": false,
    });
    NeoChatConfig::from_config_json(&serde_json::json!({
        "model_type": "neo_chat",
        "tie_word_embeddings": false,
        "llm_config": llm,
        "vision_config": {}
    }))
    .unwrap()
}

fn index_rows(w: &Weights) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let indexes = w.require("gen.indexes").unwrap().clone();
    indexes.eval().unwrap();
    let length = indexes.shape()[1] as usize;
    let flat = indexes.as_slice::<i32>();
    let row = |index: usize| flat[index * length..(index + 1) * length].to_vec();
    (row(0), row(1), row(2))
}

#[test]
fn cached_attention_and_one_block_gen_windows_match_resident() {
    let weights = Weights::from_file(FIXTURE).unwrap();
    let cfg = config_from_meta(&weights);
    let resident = Qwen3Backbone::from_weights(&weights, &cfg, "language_model").unwrap();
    let deferred = Qwen3Backbone::from_weights_deferred(
        &weights,
        &cfg,
        "language_model",
        WeightsSource::File(FIXTURE.into()),
        None,
    )
    .unwrap();
    let embeds = weights.require("input.embeds").unwrap();
    let (t, h, w) = index_rows(&weights);
    let mut resident_cache = resident.new_cache();
    let mut deferred_cache = deferred.new_cache();
    let resident_mask = resident.prepare_rope_mask(&t, &h, &w, 0).unwrap();
    let deferred_mask = deferred.prepare_rope_mask(&t, &h, &w, 0).unwrap();
    let cancel = CancelFlag::default();
    let bounded =
        AttentionPlan::budgeted(AttentionBudget::from_score_elements(4, true)).with_cancel(&cancel);
    let expected = resident
        .forward_prepared_memory(
            embeds,
            &resident_mask,
            Path::Gen,
            &mut resident_cache,
            false,
            bounded,
            None,
            false,
        )
        .unwrap();
    let actual = deferred
        .forward_prepared_memory(
            embeds,
            &deferred_mask,
            Path::Gen,
            &mut deferred_cache,
            false,
            bounded,
            Some(1),
            false,
        )
        .unwrap();
    expected.eval().unwrap();
    actual.eval().unwrap();
    assert_eq!(expected.shape(), actual.shape());
    let expected = expected.as_slice::<f32>();
    let actual = actual.as_slice::<f32>();
    let max = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 1e-5, "streamed Gen path drifted by {max}");
}

#[test]
fn canceled_windowed_forward_returns_typed_cancellation() {
    let weights = Weights::from_file(FIXTURE).unwrap();
    let cfg = config_from_meta(&weights);
    let deferred = Qwen3Backbone::from_weights_deferred(
        &weights,
        &cfg,
        "language_model",
        WeightsSource::File(FIXTURE.into()),
        None,
    )
    .unwrap();
    let embeds = weights.require("input.embeds").unwrap();
    let (t, h, w) = index_rows(&weights);
    let mask = deferred.prepare_rope_mask(&t, &h, &w, 0).unwrap();
    let mut cache = deferred.new_cache();
    let cancel = CancelFlag::default();
    cancel.cancel();
    let error = deferred
        .forward_prepared_memory(
            embeds,
            &mask,
            Path::Gen,
            &mut cache,
            false,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(4, true))
                .with_cancel(&cancel),
            Some(1),
            false,
        )
        .unwrap_err();
    assert!(matches!(error, mlx_gen::Error::Canceled));
}
