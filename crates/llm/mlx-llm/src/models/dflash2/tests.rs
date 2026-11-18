//! Tiny DFlash2 fixtures (a random draft over the tiny `qwen3_5` target of the DSpine fixtures)
//! and the draft's load-time checks. The block-draft compute proofs live with the proposer
//! (`decode::block_draft::tests`).

use std::collections::HashMap;

use super::*;
use crate::models::dspine::tests::TINY_VOCAB;
use crate::primitives::sampler::{SplitMix64, TokenRng};

/// A DFlash2 config shaped like `Qwen3.8-27B-DFlash2` at toy size over the tiny target: two
/// non-causal sliding layers reading target layers 0 and 2, two-tap convolutions over 16-channel
/// groups, a rank-8 selector over the top 4 candidates, block 6 and an 8-row window, so a
/// 12-token prompt already exercises the window (E5).
pub(crate) fn tiny_config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": false,
        "is_causal": false,
        "dflash_config": {
            "block_size": 6,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": TINY_VOCAB - 1,
            "selector_rank": 8,
            "selector_top_k": 4,
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
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
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

/// Every tensor of a `cfg` checkpoint, drawn from one fixed seed: random matrices and kernels,
/// unit norms.
pub(crate) fn tiny_tensors(cfg: &BlockDraftConfig) -> HashMap<String, Array> {
    let mut rng = SplitMix64::new(0x5EED_7194);
    let mut out = HashMap::new();
    for (name, shape) in cfg.tensors() {
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let n: usize = shape.iter().product::<usize>().max(1);
        let array = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            Array::ones::<f32>(&dims).unwrap()
        } else {
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            Array::from_slice(&data, &dims)
        };
        out.insert(name, array);
    }
    out
}

/// A tiny DFlash2 draft of `cfg` at `precision` (`None`: dense BF16).
pub(crate) fn tiny_draft_with(
    cfg: BlockDraftConfig,
    precision: Option<BlockDraftPrecision>,
) -> DFlash2Draft {
    DFlash2Draft::from_weights(&mut Weights::from_map(tiny_tensors(&cfg)), cfg, precision).unwrap()
}

/// The tiny DFlash2 draft at `precision`, as the proposer takes it.
pub(crate) fn tiny_draft(precision: Option<BlockDraftPrecision>) -> crate::models::BlockDraft {
    crate::models::BlockDraft::DFlash2(tiny_draft_with(tiny_config(), precision))
}

/// The tiny draft fits the tiny target by the shared check, and parses to DFlash2's head.
#[test]
fn the_tiny_dflash2_draft_fits_the_tiny_target() {
    let cfg = tiny_config();
    assert_eq!(cfg.format, core_llm::BlockDraftFormat::DFlash2);
    assert!(cfg
        .mismatches(&core_llm::BlockDraftTarget {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: TINY_VOCAB,
            tokenizer_vocab: TINY_VOCAB,
        })
        .is_empty());
    assert_eq!(cfg.context_rows(), Some(7));
    assert!(matches!(
        cfg.head,
        BlockDraftHead::DFlash2 {
            conv_kernel_size: 2,
            conv_group_size: 16,
            selector_rank: 8,
            causal: false,
            ..
        }
    ));
}

/// A build quantizes the projections — `fc`, the layers', the convolutions' kernel projections
/// and the selector's — at the requested precision, and keeps the codebooks and base kernels BF16.
#[test]
fn a_build_quantizes_the_projections_and_keeps_the_codebooks_bf16() {
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        let draft = tiny_draft_with(tiny_config(), Some(precision));
        assert_eq!(draft.precision(), Some(precision));
        assert!(draft.fc.is_quantized() && draft.selector.is_quantized());
        assert!(draft.layers.iter().all(|l| l.body.q_proj.is_quantized()
            && l.attention_conv.kernel_projection.is_quantized()
            && l.mlp_conv.kernel_projection.is_quantized()));
        assert_eq!(draft.predecessor.dtype(), Dtype::Bfloat16);
        assert_eq!(
            draft.layers[0].mlp_conv.base_kernel.dtype(),
            Dtype::Bfloat16
        );
    }
    assert!(!tiny_draft_with(tiny_config(), None).fc.is_quantized());
}

/// The draft is built from the groups its load is priced by: one materialization group per layer
/// plus the head, each reading only its own group's tensors, together every stored tensor. A
/// build that reads outside its group is refused by name. Lazy: nothing is evaluated.
#[test]
fn the_dflash2_draft_builds_from_the_load_groups_its_load_is_priced_by() {
    let cfg = tiny_config();
    let groups = cfg.load_groups();
    let w = Weights::from_map(tiny_tensors(&cfg));
    let q8 = Some(BlockDraftPrecision::Q8);
    let draft = DFlash2Draft::build(&w, cfg.clone(), q8, &groups).unwrap();
    assert_eq!(draft.param_groups().len(), groups.len());
    assert!(w.unused_keys().is_empty(), "the build reads every tensor");
    let mut short = groups.clone();
    short[1]
        .tensors
        .retain(|t| !t.ends_with("mlp_conv.base_kernel"));
    let why = DFlash2Draft::build(&w, cfg, q8, &short)
        .unwrap_err()
        .to_string();
    assert!(why.contains("outside its priced load group"), "{why}");
}

/// A checkpoint missing a tensor, carrying one the draft does not read, or of a DSpine config, is
/// refused by name before anything is built.
#[test]
fn a_dflash2_checkpoint_outside_the_contract_is_refused_by_name() {
    let cfg = tiny_config();
    let mut tensors = tiny_tensors(&cfg);
    tensors.remove("layers.1.attention_conv.kernel_projection.weight");
    tensors.insert("mask_embedding".into(), Array::zeros::<f32>(&[64]).unwrap());
    let why = DFlash2Draft::from_weights(&mut Weights::from_map(tensors), cfg, None)
        .unwrap_err()
        .to_string();
    assert!(why.starts_with("block draft: tensors:"), "{why}");
    assert!(why.contains("`layers.1.attention_conv.kernel_projection.weight` is missing"));
    assert!(why.contains("unused tensors mask_embedding"));
    let dspine = crate::models::dspine::tests::tiny_config();
    let w = Weights::from_map(crate::models::dspine::tests::tiny_tensors(&dspine));
    let groups = dspine.load_groups();
    let why = DFlash2Draft::build(&w, dspine, None, &groups[1..])
        .unwrap_err()
        .to_string();
    assert!(why.contains("not a DFlash2 draft"), "{why}");
}

/// The block's attention mask is the reference's `context | block`: non-causal over the block,
/// and a context row at `c` visible to a block row at `p` while `p - c` is inside the window.
#[test]
fn the_block_mask_is_non_causal_over_the_block_and_windowed_over_the_context() {
    let draft = tiny_draft_with(tiny_config(), None);
    let mut ctx = BlockDraftContext::new(7);
    // No context: every block row sees the whole block.
    assert!(draft.mask(&ctx, 6).unwrap().is_none());
    // A full 7-row context (window 8): block row i no longer sees the first i context rows.
    let row = Array::zeros::<f32>(&[1, 1, 7, 32]).unwrap();
    ctx.keys = vec![row.clone(), row.clone()];
    ctx.values = vec![row.clone(), row];
    let mask = draft.mask(&ctx, 3).unwrap().unwrap();
    assert_eq!(mask.shape(), [1, 1, 3, 10]);
    let m = mask.as_slice::<f32>();
    for i in 0..3usize {
        for j in 0..10usize {
            let blocked = m[i * 10 + j] < -1.0;
            let want = j < 7 && j < i;
            assert_eq!(blocked, want, "row {i} key {j}");
        }
    }
}

/// A draft loaded from a BF16 checkpoint on disk — lazy safetensors loads, the production path —
/// releases every source as its group materializes and builds, at each precision, exactly the
/// arrays the resident build of the same weights does.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_dflash2_draft_loaded_from_disk_releases_its_sources_and_matches_the_resident_build() {
    let cfg = tiny_config();
    let dir = tempfile::tempdir().unwrap();
    let stored: Vec<(String, Array)> = tiny_tensors(&cfg)
        .into_iter()
        .map(|(name, a)| (name, a.as_dtype(Dtype::Bfloat16).unwrap()))
        .collect();
    let refs: Vec<_> = stored.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        tiny_config_json().to_string(),
    )
    .unwrap();
    for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
        let mut w = Weights::from_dir(dir.path()).unwrap();
        let loaded = DFlash2Draft::from_weights(&mut w, cfg.clone(), Some(precision)).unwrap();
        assert_eq!(w.len(), stored.len(), "released keys stay contained");
        assert!(
            w.into_map().is_empty(),
            "{precision:?}: every source released"
        );
        let resident = tiny_draft_with(tiny_config(), Some(precision));
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
        // The production entry point dispatches on the checkpoint's `architectures`.
        let cfg = BlockDraftConfig::read(dir.path()).unwrap();
        let draft = crate::models::BlockDraft::load(dir.path(), cfg, precision).unwrap();
        assert!(matches!(draft, crate::models::BlockDraft::DFlash2(_)));
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::dflash2::tests::a_dflash2_draft_loaded_from_disk_releases_its_sources_and_matches_the_resident_build"
    );
}

/// The selector's table against a host recomputation of the reference's edge formula: row `i`'s
/// candidates are the top-`K` of the row's logits, and `table[i, s, j] = logit_i(c_j) +
/// Σ_r pred[sender_s]_r · h_i,r · succ[c_j]_r` with row 0's senders the anchor.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_selector_table_is_the_reference_edge_score() {
    let draft = tiny_draft_with(tiny_config(), None);
    let mut rng = SplitMix64::new(0x7194);
    let n = 3usize;
    let data: Vec<f32> = (0..n * 64).map(|_| rng.next_f32() - 0.5).collect();
    let rows = bf16(&Array::from_slice(&data, &[n as i32, 64])).unwrap();
    let head_w: Vec<f32> = (0..TINY_VOCAB * 64).map(|_| rng.next_f32() - 0.5).collect();
    let head_w = Array::from_slice(&head_w, &[TINY_VOCAB as i32, 64]);
    let head = |x: &Array| -> Result<Array> { Ok(matmul(&f32(x)?, head_w.t())?) };
    let anchor = 7;
    let (cand, table) = draft.table(&rows, anchor, &head).unwrap();
    let k = 4usize;
    assert_eq!(cand.shape(), [n as i32, k as i32]);
    assert_eq!(table.shape(), [n as i32, k as i32, k as i32]);
    let host = |a: &Array| -> Vec<f32> {
        f32(a)
            .unwrap()
            .reshape(&[-1])
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    };
    let logits = host(&head(&rows).unwrap());
    let hidden = host(&draft.selector.forward(&rows).unwrap());
    let (pred, succ) = (host(&draft.predecessor), host(&draft.successor));
    let cand: Vec<i32> = cand.as_slice::<i32>().to_vec();
    let table = host(&table);
    let r = 8usize;
    let v = TINY_VOCAB;
    for i in 0..n {
        let row = &logits[i * v..(i + 1) * v];
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let mut top: Vec<i32> = order[..k].iter().map(|&t| t as i32).collect();
        top.sort_unstable();
        assert_eq!(&cand[i * k..(i + 1) * k], &top[..], "row {i} candidates");
        for s in 0..k {
            let sender = if i == 0 {
                anchor
            } else {
                cand[(i - 1) * k + s]
            } as usize;
            for j in 0..k {
                let c = cand[i * k + j] as usize;
                let edge: f32 = (0..r)
                    .map(|q| pred[sender * r + q] * hidden[i * r + q] * succ[c * r + q])
                    .sum();
                let want = row[c] + edge;
                let got = table[(i * k + s) * k + j];
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "{i} {s} {j}: {got} vs {want}"
                );
            }
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::dflash2::tests::the_selector_table_is_the_reference_edge_score"
    );
}

/// The grouped dynamic causal convolution against a host loop of the reference's
/// `_grouped_dynamic_convolve`: output row `t`, channel `c` of group `g` is
/// `Σ_o (base[o, c] + dynamic[t, o, g]) · hidden[t - o, c]`, zero before the block — including a
/// block shorter than the taps.
#[test]
fn the_dynamic_convolution_is_the_reference_grouped_causal_sum() {
    let mut rng = SplitMix64::new(0xC0A7);
    let (h, gs, taps) = (32usize, 16usize, 2usize);
    let groups = h / gs;
    for t in [1usize, 5] {
        let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
        let (x, d, b) = (draw(t * h), draw(t * taps * groups), draw(taps * h));
        let got = convolve(
            &Array::from_slice(&x, &[1, t as i32, h as i32]),
            &Array::from_slice(&d, &[1, t as i32, taps as i32, groups as i32]),
            &Array::from_slice(&b, &[taps as i32, h as i32]),
            gs,
        )
        .unwrap();
        let got = got.as_slice::<f32>();
        for row in 0..t {
            for c in 0..h {
                let want: f32 = (0..taps.min(row + 1))
                    .map(|o| {
                        (b[o * h + c] + d[(row * taps + o) * groups + c / gs])
                            * x[(row - o) * h + c]
                    })
                    .sum();
                let at = got[row * h + c];
                assert!(
                    (at - want).abs() < 1e-5,
                    "t {t} row {row} ch {c}: {at} vs {want}"
                );
            }
        }
    }
}
