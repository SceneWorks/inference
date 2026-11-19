//! Tiny DSpine fixtures (a random draft over a random `qwen3_5` target) and the draft's
//! load-time checks. The block-draft compute proofs live with the proposer
//! (`decode::block_draft::tests`).

use std::collections::HashMap;

use super::*;
use crate::models::qwen35::tests::{cfg_json, synthetic_weights};
use crate::models::{Qwen35Config, Qwen35Model};
use crate::primitives::sampler::{SplitMix64, TokenRng};

/// The tiny target's vocabulary.
pub(crate) const TINY_VOCAB: usize = 50;

/// The shared synthetic `qwen3_5` hybrid (three DeltaNet layers, one attention layer) at width
/// 64, so a draft over it quantizes at the 64-wide group.
pub(crate) fn tiny_target() -> Qwen35Model {
    let mut v = cfg_json();
    v["text_config"]["hidden_size"] = 64.into();
    v["text_config"]["intermediate_size"] = 128.into();
    let cfg = Qwen35Config::from_json(&v).unwrap();
    Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap()
}

/// [`tiny_target`] with its embedding and LM head redrawn at random: the shared synthetic weights
/// are periodic over the vocabulary, which leaves every next-token distribution on a few evenly
/// spaced ids — a sampled proof needs distributions a draft's candidates can overlap.
pub(crate) fn tiny_random_target() -> Qwen35Model {
    let mut v = cfg_json();
    v["text_config"]["hidden_size"] = 64.into();
    v["text_config"]["intermediate_size"] = 128.into();
    let cfg = Qwen35Config::from_json(&v).unwrap();
    let mut tensors = synthetic_weights(&cfg).into_map();
    let mut rng = SplitMix64::new(0x7193_7A26);
    for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
        let data: Vec<f32> = (0..TINY_VOCAB * 64)
            .map(|_| (rng.next_f32() - 0.5) * 0.8)
            .collect();
        tensors.insert(
            name.into(),
            Array::from_slice(&data, &[TINY_VOCAB as i32, 64]),
        );
    }
    Qwen35Model::from_weights(&Weights::from_map(tensors), "model.language_model", cfg).unwrap()
}

/// The ids [`tiny_target_biased`] boosts, most first.
pub(crate) const BIASED_IDS: [usize; 6] = [1, 9, 3, 14, 22, 30];

/// [`tiny_target`] with a random embedding and head, every embedding row carrying one shared
/// direction and the head rows of [`BIASED_IDS`] aligned with it, by decreasing strength — the
/// Candle fixture of the same name (SW-7198), ported for SW-7209. The direction rides the
/// residual stream of the target and of a block draft (which embeds its block with the target's
/// embedding and scores its rows with the target's head) alike, so both put most of their mass
/// on those ids whatever the context: a draft that agrees with the target often, as a trained one
/// does — what a sampled acceptance proof needs to be sensitive to the proposal distribution.
pub(crate) fn tiny_target_biased() -> Qwen35Model {
    let mut v = cfg_json();
    v["text_config"]["hidden_size"] = 64.into();
    v["text_config"]["intermediate_size"] = 128.into();
    let cfg = Qwen35Config::from_json(&v).unwrap();
    let hidden = 64usize;
    let mut tensors = synthetic_weights(&cfg).into_map();
    let mut rng = SplitMix64::new(0x7198_B1A5);
    let direction = |i: usize| if i.is_multiple_of(2) { 0.25f32 } else { -0.25 };
    let embed: Vec<f32> = (0..TINY_VOCAB * hidden)
        .map(|i| (rng.next_f32() - 0.5) * 0.8 + 6.0 * direction(i % hidden))
        .collect();
    let head: Vec<f32> = (0..TINY_VOCAB * hidden)
        .map(|i| {
            let (row, col) = (i / hidden, i % hidden);
            let boost = BIASED_IDS
                .iter()
                .position(|&id| id == row)
                .map_or(0.0, |rank| 0.3 - 0.04 * rank as f32);
            (rng.next_f32() - 0.5) * 0.8 + boost * direction(col)
        })
        .collect();
    for (name, data) in [
        ("model.language_model.embed_tokens.weight", embed),
        ("lm_head.weight", head),
    ] {
        tensors.insert(
            name.into(),
            Array::from_slice(&data, &[TINY_VOCAB as i32, hidden as i32]),
        );
    }
    Qwen35Model::from_weights(&Weights::from_map(tensors), "model.language_model", cfg).unwrap()
}

/// A DSpine config shaped like `Qwen3.8-27B-DSpine-b16` at toy size over [`tiny_target`]: two
/// layers reading target layers 0 and 2, block 6, top-4 refinement and an 8-row sliding window,
/// so a 12-token prompt already exercises the window (E5).
pub(crate) fn tiny_config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DSpineDraftModel"],
        "block_size": 6,
        "code_bridge_second_pass": {"first_exact": 1, "head_topk": 4, "mode": "serial_table"},
        "dflash_config": {
            "draft_block_causal": true,
            "mask_token_id": TINY_VOCAB - 1,
            "selected_prefix_code_bridge": {"code_size": 16, "eps": 1e-06, "rank": 8},
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": [1, 2],
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": true,
            "selected_prefix_last_write_refine": {"head_topk": 4, "kind": "exact_sender", "version": 1, "weight": 1.0},
            "selected_prefix_terminal": false,
            "target_layer_ids": [0, 2]
        },
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "max_window_layers": 2,
        "model_type": "qwen3",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_target_layers": 4,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 8,
        "tie_word_embeddings": false,
        "use_sliding_window": true,
        "vocab_size": TINY_VOCAB
    })
}

/// [`tiny_config_json`] parsed.
pub(crate) fn tiny_config() -> BlockDraftConfig {
    BlockDraftConfig::from_json(&tiny_config_json()).unwrap()
}

/// Every tensor of a `cfg` checkpoint, drawn from one fixed seed: random matrices, unit norms,
/// version markers 1 and the disabled (zero) terminal projection.
pub(crate) fn tiny_tensors(cfg: &BlockDraftConfig) -> HashMap<String, Array> {
    let mut rng = SplitMix64::new(0x5EED_7193);
    let mut out = HashMap::new();
    for (name, shape) in cfg.tensors() {
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let n: usize = shape.iter().product::<usize>().max(1);
        let array = if shape.is_empty() {
            Array::from_slice(&[1i64], &[])
        } else if name.ends_with("out.weight") {
            Array::zeros::<f32>(&dims).unwrap()
        } else if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            Array::ones::<f32>(&dims).unwrap()
        } else {
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            Array::from_slice(&data, &dims)
        };
        out.insert(name, array);
    }
    out
}

/// The tiny draft at `precision` (`None`: dense BF16).
pub(crate) fn tiny_draft(precision: Option<BlockDraftPrecision>) -> DSpineDraft {
    tiny_draft_with(tiny_config(), precision)
}

/// A tiny draft of `cfg` at `precision`.
pub(crate) fn tiny_draft_with(
    cfg: BlockDraftConfig,
    precision: Option<BlockDraftPrecision>,
) -> DSpineDraft {
    DSpineDraft::from_weights(&mut Weights::from_map(tiny_tensors(&cfg)), cfg, precision).unwrap()
}

/// The tiny draft fits the tiny target by the shared check.
#[test]
fn the_tiny_draft_fits_the_tiny_target() {
    let target = tiny_target();
    let cfg = tiny_config();
    assert!(cfg
        .mismatches(&core_llm::BlockDraftTarget {
            hidden_size: target.config().hidden_size as usize,
            num_layers: target.num_layers(),
            vocab_size: target.config().vocab_size as usize,
            tokenizer_vocab: TINY_VOCAB,
        })
        .is_empty());
    assert_eq!(cfg.context_rows(), Some(7));
}

/// A build quantizes the backbone at the requested precision and leaves the bridge BF16.
#[test]
fn a_build_quantizes_the_backbone_only() {
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        let draft = tiny_draft(Some(precision));
        assert_eq!(draft.precision(), Some(precision));
        assert!(draft.fc.is_quantized());
        assert!(draft
            .layers
            .iter()
            .all(|l| l.q_proj.is_quantized() && l.down_proj.is_quantized()));
        assert_eq!(draft.bridge.codebook.dtype(), Dtype::Bfloat16);
    }
    let dense = tiny_draft(None);
    assert!(!dense.fc.is_quantized());
}

/// The draft is built from the groups its load is priced by
/// ([`BlockDraftConfig::load_groups`]): one materialization group per layer plus the head, each
/// reading only its own group's tensors, and together every tensor but the checks' (read and
/// dropped first). A build that reads outside its group is refused by name. Lazy: nothing is
/// evaluated.
#[test]
fn the_draft_builds_from_the_load_groups_its_load_is_priced_by() {
    let cfg = tiny_config();
    let groups = cfg.load_groups();
    let w = Weights::from_map(tiny_tensors(&cfg));
    let q8 = Some(BlockDraftPrecision::Q8);
    let draft = DSpineDraft::build(&w, cfg.clone(), q8, &groups[1..]).unwrap();
    assert_eq!(draft.param_groups().len(), groups.len() - 1);
    let mut unread = w.unused_keys();
    unread.sort_unstable();
    let mut checks: Vec<&str> = groups[0].tensors.iter().map(String::as_str).collect();
    checks.sort_unstable();
    assert_eq!(
        unread, checks,
        "the build reads every tensor but the checks'"
    );
    let mut short = groups[1..].to_vec();
    short[0].tensors.retain(|t| !t.ends_with("q_proj.weight"));
    let why = DSpineDraft::build(&w, cfg, q8, &short)
        .unwrap_err()
        .to_string();
    assert!(why.contains("outside its priced load group"), "{why}");
}

/// A draft loaded from a BF16 checkpoint on disk — lazy safetensors loads, the production path —
/// releases every source as its group materializes and builds, at each precision, exactly the
/// arrays the resident build of the same weights does.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_draft_loaded_from_disk_releases_its_sources_and_matches_the_resident_build() {
    let cfg = tiny_config();
    let dir = tempfile::tempdir().unwrap();
    let stored: Vec<(String, Array)> = tiny_tensors(&cfg)
        .into_iter()
        .map(|(name, a)| {
            let a = if a.dtype() == Dtype::Float32 {
                a.as_dtype(Dtype::Bfloat16).unwrap()
            } else {
                a
            };
            (name, a)
        })
        .collect();
    let refs: Vec<_> = stored.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        let mut w = Weights::from_dir(dir.path()).unwrap();
        let loaded = DSpineDraft::from_weights(&mut w, cfg.clone(), Some(precision)).unwrap();
        assert_eq!(w.len(), stored.len(), "released keys stay contained");
        assert!(
            w.into_map().is_empty(),
            "{precision:?}: every source released"
        );
        let resident = tiny_draft(Some(precision));
        let (got, want) = (loaded.param_groups(), resident.param_groups());
        assert_eq!(got.len(), want.len());
        for (g, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(want).enumerate() {
                assert_eq!(a.dtype(), b.dtype(), "{precision:?} group {g} array {i}");
                assert!(
                    a.array_eq(b, None).unwrap().item::<bool>(),
                    "{precision:?} group {g} array {i}"
                );
            }
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::dspine::tests::a_draft_loaded_from_disk_releases_its_sources_and_matches_the_resident_build"
    );
}

/// A checkpoint whose version markers, terminal projection or tensors break the DSpine contract
/// is refused by name before anything is built.
#[test]
fn a_checkpoint_outside_the_contract_is_refused_by_name() {
    let cfg = tiny_config();
    let refuse = |edit: &dyn Fn(&mut HashMap<String, Array>)| {
        let mut tensors = tiny_tensors(&cfg);
        edit(&mut tensors);
        DSpineDraft::from_weights(&mut Weights::from_map(tensors), cfg.clone(), None)
            .unwrap_err()
            .to_string()
    };
    assert!(refuse(&|t| {
        t.insert(
            "draft_block_causal_version".into(),
            Array::from_slice(&[2i64], &[]),
        );
    })
    .contains("`draft_block_causal_version` is not version 1"));
    assert!(refuse(&|t| {
        t.insert(
            "selected_prefix_bridge.out.weight".into(),
            Array::ones::<f32>(&[16, 8]).unwrap(),
        );
    })
    .contains("nonzero terminal projection"));
    let missing = refuse(&|t| {
        t.remove("layers.1.mlp.up_proj.weight");
        t.insert(
            "lm_head.weight".into(),
            Array::zeros::<f32>(&[4, 4]).unwrap(),
        );
    });
    assert!(missing.starts_with("block draft: tensors:"), "{missing}");
    assert!(missing.contains("`layers.1.mlp.up_proj.weight` is missing"));
    assert!(missing.contains("unused tensors lm_head.weight"));
}
