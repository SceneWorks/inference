//! Tiny block-draft fixtures (a random draft of each format over a tiny `qwen3_5` target, the MLX
//! fixtures' shapes) and the drafts' load-time checks — construction only, no forward. The
//! block-draft compute proofs live with the proposer (`decode::block_draft::tests`).

use super::*;
use crate::models::qwen35::tests::{cfg_json, synthetic_weights};
use crate::models::{Qwen35Config, Qwen35Model};
use crate::primitives::sampler::{SplitMix64, TokenRng};
use crate::primitives::Weights;

/// The proof device: the CUDA device the production selector opens with the graph runner on (its
/// own capturable stream) in a `cuda` build, else the CPU. The guard holds the graph switch.
pub(crate) fn proof_device() -> (Option<crate::decode::graph::CudaGraphsPolicyGuard>, Device) {
    #[cfg(feature = "cuda")]
    {
        let guard = crate::decode::graph::cuda_graphs_policy_guard(Some(true));
        let device = crate::device::select_device().expect("the CUDA proof device");
        assert!(device.is_cuda(), "a cuda build proves on the CUDA device");
        (Some(guard), device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        (None, Device::Cpu)
    }
}

/// The tiny target's vocabulary.
pub(crate) const TINY_VOCAB: usize = 50;

fn tiny_target_config() -> Qwen35Config {
    let mut v = cfg_json();
    v["text_config"]["hidden_size"] = 64.into();
    v["text_config"]["intermediate_size"] = 128.into();
    Qwen35Config::from_json(&v).unwrap()
}

/// The shared synthetic `qwen3_5` hybrid (three DeltaNet layers, one attention layer) at width
/// 64 — the MLX fixtures' target — on the CPU.
pub(crate) fn tiny_target() -> Qwen35Model {
    tiny_target_on(&Device::Cpu, false)
}

/// The tiny target on `device` (its compute dtype there). With `random`, its embedding and LM
/// head are redrawn at random: the synthetic weights are periodic over the vocabulary, which
/// leaves every next-token distribution on a few evenly spaced ids — a sampled proof needs
/// distributions a draft's candidates can overlap.
pub(crate) fn tiny_target_on(device: &Device, random: bool) -> Qwen35Model {
    let cfg = tiny_target_config();
    let mut tensors = synthetic_weights(&cfg).into_map();
    let mut rng = SplitMix64::new(0x7193_7A26);
    for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
        if random {
            let data: Vec<f32> = (0..TINY_VOCAB * 64)
                .map(|_| (rng.next_f32() - 0.5) * 0.8)
                .collect();
            let t = Tensor::from_vec(data, (TINY_VOCAB, 64), &Device::Cpu).unwrap();
            tensors.insert(name.into(), t);
        }
    }
    let tensors = tensors
        .into_iter()
        .map(|(k, t)| (k, t.to_device(device).unwrap()))
        .collect();
    let w = Weights::from_map(tensors, device.clone());
    Qwen35Model::from_weights(&w, "model.language_model", cfg).unwrap()
}

/// The tiny target loaded as a Prism snapshot whose LM head (and every other matrix) stayed
/// dense: its registry packs one tensor no projection reads, so only the load is Prism.
pub(crate) fn tiny_prism_target() -> Qwen35Model {
    let cfg = tiny_target_config();
    let w = Weights::from_map(synthetic_weights(&cfg).into_map(), Device::Cpu);
    let name = "model.visual.unused.weight";
    let registry = crate::primitives::PrismRegistry::new(HashMap::from([(
        name.to_string(),
        std::sync::Arc::new(crate::primitives::prism::tests::one_row_weight(name)),
    )]))
    .unwrap();
    let dtype = crate::device::compute_dtype(&Device::Cpu);
    Qwen35Model::from_prism_weights(&w, "model.language_model", cfg, &registry, dtype).unwrap()
}

/// A DSpine config shaped like `Qwen3.8-27B-DSpine-b16` at toy size over [`tiny_target`]: two
/// layers reading target layers 0 and 2, block 6, top-4 refinement and an 8-row sliding window,
/// so a 12-token prompt already exercises the window (E5). The MLX fixture's config.
pub(crate) fn dspine_config_json() -> serde_json::Value {
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

/// A DFlash2 config shaped like `Qwen3.8-27B-DFlash2` at toy size: two non-causal sliding layers
/// reading target layers 0 and 2, two-tap convolutions over 16-channel groups, a rank-8 selector
/// over the top 4 candidates, block 6 and an 8-row window. The MLX fixture's config.
pub(crate) fn dflash2_config_json() -> serde_json::Value {
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

/// The tiny config of `format`, parsed.
pub(crate) fn tiny_config(format: BlockDraftFormat) -> BlockDraftConfig {
    BlockDraftConfig::from_json(&match format {
        BlockDraftFormat::DSpine => dspine_config_json(),
        BlockDraftFormat::DFlash2 => dflash2_config_json(),
    })
    .unwrap()
}

/// Every tensor of a `cfg` checkpoint, drawn from one fixed seed: random matrices, unit norms,
/// DSpine's version markers (I64 scalars, 1) and its disabled (zero) terminal projection.
pub(crate) fn tiny_tensors(cfg: &BlockDraftConfig) -> HashMap<String, Tensor> {
    let mut rng = SplitMix64::new(0x5EED_7197);
    let mut out = HashMap::new();
    for (name, shape) in cfg.tensors() {
        let n: usize = shape.iter().product::<usize>().max(1);
        let t = if shape.is_empty() {
            Tensor::new(1i64, &Device::Cpu).unwrap()
        } else if name.ends_with("out.weight") {
            Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
        } else if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
        } else {
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
        };
        out.insert(name, t);
    }
    out
}

/// A tiny draft of `cfg` at `precision` (`None`: dense) on the CPU in f32 (the CPU's compute
/// dtype), or on `device` in its compute dtype.
pub(crate) fn tiny_draft_on(
    cfg: BlockDraftConfig,
    precision: Option<BlockDraftPrecision>,
    device: &Device,
) -> BlockDraft {
    let source = DraftSource::Map(tiny_tensors(&cfg));
    BlockDraft::from_source(
        &source,
        cfg,
        precision,
        device,
        crate::device::compute_dtype(device),
    )
    .unwrap()
}

/// The tiny draft of `format` at `precision` on the CPU.
pub(crate) fn tiny_draft(
    format: BlockDraftFormat,
    precision: Option<BlockDraftPrecision>,
) -> BlockDraft {
    tiny_draft_on(tiny_config(format), precision, &Device::Cpu)
}

/// Both tiny drafts fit the tiny target by the shared check, and keep a 7-row context.
#[test]
fn the_tiny_drafts_fit_the_tiny_target() {
    let target = tiny_target();
    for format in BlockDraftFormat::ALL {
        let cfg = tiny_config(format);
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
}

/// A build quantizes `fc` and every layer projection at the requested precision (GGML blocks),
/// and keeps them dense without one.
#[test]
fn a_build_quantizes_the_projections_at_the_requested_precision() {
    for format in BlockDraftFormat::ALL {
        for precision in [BlockDraftPrecision::Q8, BlockDraftPrecision::Q4] {
            let draft = tiny_draft(format, Some(precision));
            assert_eq!(draft.precision(), Some(precision));
            assert!(draft.is_quantized(), "{format:?} {precision:?}");
            assert_eq!(draft.config().format, format);
        }
        let dense = tiny_draft(format, None);
        assert!(!dense.is_quantized());
        assert_eq!(dense.causal(), format == BlockDraftFormat::DSpine);
    }
}

/// A build reads each tensor inside the load group it was priced in: a group that omits one is
/// refused by name.
#[test]
fn a_read_outside_its_priced_load_group_is_refused() {
    for format in BlockDraftFormat::ALL {
        let cfg = tiny_config(format);
        let source = DraftSource::Map(tiny_tensors(&cfg));
        let mut groups = cfg.load_groups();
        let first_layer = usize::from(format == BlockDraftFormat::DSpine);
        groups[first_layer]
            .tensors
            .retain(|t| !t.ends_with("q_proj.weight"));
        let why = match format {
            BlockDraftFormat::DSpine => {
                DSpineDraft::build(&source, cfg, None, &groups[1..], &Device::Cpu, DType::F32).err()
            }
            BlockDraftFormat::DFlash2 => {
                DFlash2Draft::build(&source, cfg, None, &groups, &Device::Cpu, DType::F32).err()
            }
        }
        .unwrap()
        .to_string();
        assert!(why.contains("outside its priced load group"), "{why}");
    }
}

/// DSpine's checks group refuses a version marker that is not 1 and a nonzero terminal
/// projection by name, and every format refuses a missing or unused tensor — the shared
/// [`BlockDraftConfig::check_tensors`].
#[test]
fn a_checkpoint_outside_the_contract_is_refused_by_name() {
    let cfg = tiny_config(BlockDraftFormat::DSpine);
    let load = |tensors: HashMap<String, Tensor>| {
        BlockDraft::from_source(
            &DraftSource::Map(tensors),
            cfg.clone(),
            None,
            &Device::Cpu,
            DType::F32,
        )
        .err()
        .unwrap()
    };
    let mut tensors = tiny_tensors(&cfg);
    tensors.insert(
        "last_write_refine_version".into(),
        Tensor::new(2i64, &Device::Cpu).unwrap(),
    );
    assert!(load(tensors).to_string().contains("is not version 1"));
    let mut tensors = tiny_tensors(&cfg);
    tensors.insert(
        "selected_prefix_bridge.out.weight".into(),
        Tensor::ones((16, 8), DType::F32, &Device::Cpu).unwrap(),
    );
    assert!(load(tensors)
        .to_string()
        .contains("nonzero terminal projection"));
    for format in BlockDraftFormat::ALL {
        let cfg = tiny_config(format);
        let mut tensors = tiny_tensors(&cfg);
        tensors.remove("fc.weight");
        tensors.insert(
            "stray.weight".into(),
            Tensor::ones(2, DType::F32, &Device::Cpu).unwrap(),
        );
        let why = BlockDraft::from_source(
            &DraftSource::Map(tensors),
            cfg,
            None,
            &Device::Cpu,
            DType::F32,
        )
        .err()
        .unwrap();
        assert_eq!(why.kind(), "tensors");
        let why = why.to_string();
        assert!(
            why.contains("`fc.weight` is missing") && why.contains("unused tensors stray.weight")
        );
    }
}

/// The block mask: a block row sees the context rows its window covers, and the block rows
/// before it (causal) or every block row (non-causal); a non-causal block whose window covers the
/// whole context masks nothing.
#[test]
fn the_block_mask_is_causal_or_not_over_the_block_and_windowed_over_the_context() {
    let draft = tiny_draft(BlockDraftFormat::DFlash2, None);
    let body = draft.body();
    let mut ctx = BlockDraftContext::new(0);
    let kv = Tensor::zeros((1, 1, 5, 32), DType::F32, &Device::Cpu).unwrap();
    ctx.keys = vec![kv.clone(); 2];
    ctx.values = vec![kv; 2];
    // 5 context rows, 3 block rows: row 2 sits 7 past context row 0 — outside the 8-row window
    // only for row 2 at context row 0 (2 + 5 - 0 = 7 < 8 is inside): nothing masked.
    assert!(body.mask(&ctx, 3, false).unwrap().is_none());
    let rows = |mask: Tensor| -> Vec<Vec<bool>> {
        mask.squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .map(|r| r.into_iter().map(|x| x == 0.0).collect())
            .collect()
    };
    let seen = rows(body.mask(&ctx, 4, false).unwrap().unwrap());
    assert_eq!(seen[3][..5], [false, true, true, true, true], "the window");
    assert!(seen.iter().all(|r| r[5..].iter().all(|&s| s)), "non-causal");
    let seen = rows(body.mask(&ctx, 4, true).unwrap().unwrap());
    for (i, r) in seen.iter().enumerate() {
        for (j, &s) in r[5..].iter().enumerate() {
            assert_eq!(s, j <= i, "causal row {i} col {j}");
        }
    }
}

/// A bare `[vocab, hidden]` matrix as the target head (its logits are `x · Wᵀ`).
struct MatrixHead<'a>(&'a Tensor);

impl TargetHead for MatrixHead<'_> {
    fn logits(&self, rows: &Tensor) -> Result<Tensor> {
        mm(rows, self.0)
    }

    fn scores(&self, rows: &Tensor, ids: &Tensor) -> Result<Tensor> {
        crate::primitives::head_rows::HeadRows::Dense(self.0.clone()).scores(
            &Projection::Dense(candle_nn::Linear::new(self.0.clone(), None)),
            rows,
            ids,
        )
    }
}

/// The selector's table against a host recomputation of the reference's edge formula: row `i`'s
/// candidates are the top-`K` of the row's logits, and `table[i, s, j] = logit_i(c_j) +
/// Σ_r pred[sender_s]_r · h_i,r · succ[c_j]_r` with row 0's senders the anchor.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_selector_table_is_the_reference_edge_score() {
    let (_guard, device) = proof_device();
    let BlockDraft::DFlash2(draft) =
        tiny_draft_on(tiny_config(BlockDraftFormat::DFlash2), None, &device)
    else {
        unreachable!()
    };
    let dtype = crate::device::compute_dtype(&device);
    let on_device = |data: Vec<f32>, shape: (usize, usize)| {
        Tensor::from_vec(data, shape, &Device::Cpu)
            .and_then(|t| t.to_device(&device)?.to_dtype(dtype))
            .unwrap()
    };
    let mut rng = SplitMix64::new(0x7194);
    let n = 3usize;
    let rows = on_device((0..n * 64).map(|_| rng.next_f32() - 0.5).collect(), (n, 64));
    let head_w = on_device(
        (0..TINY_VOCAB * 64).map(|_| rng.next_f32() - 0.5).collect(),
        (TINY_VOCAB, 64),
    );
    let head = |x: &Tensor| -> Result<Tensor> { mm(x, &head_w) };
    let anchor = 7;
    let (cand, table) = draft.table(&rows, anchor, &MatrixHead(&head_w)).unwrap();
    let k = 4usize;
    assert!(cand.len() == n && cand.iter().all(|c| c.len() == k));
    assert_eq!(table.len(), n * k * k);
    let host =
        |t: &Tensor| -> Vec<f32> { f32(t).unwrap().flatten_all().unwrap().to_vec1().unwrap() };
    let logits = host(&head(&rows).unwrap());
    let hidden = host(&draft.selector.forward(&rows).unwrap());
    let (pred, succ) = (host(&draft.predecessor), host(&draft.successor));
    let (r, v) = (8usize, TINY_VOCAB);
    for i in 0..n {
        let row = &logits[i * v..(i + 1) * v];
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let mut top: Vec<i32> = order[..k].iter().map(|&t| t as i32).collect();
        top.sort_unstable();
        assert_eq!(cand[i], top, "row {i} candidates");
        for s in 0..k {
            let sender = if i == 0 { anchor } else { cand[i - 1][s] } as usize;
            for j in 0..k {
                let c = cand[i][j] as usize;
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED models::block_draft::tests::the_selector_table_is_the_reference_edge_score");
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
        let cpu = &Device::Cpu;
        let got = convolve(
            &Tensor::from_vec(x.clone(), (1, t, h), cpu).unwrap(),
            &Tensor::from_vec(d.clone(), (1, t, taps, groups), cpu).unwrap(),
            &Tensor::from_vec(b.clone(), (taps, h), cpu).unwrap(),
            gs,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
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
