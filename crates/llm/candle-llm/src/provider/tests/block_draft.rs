//! The block draft through the Candle provider (SW-F360, SW-7197): the shared typed refusals
//! (AC2), the load and request pricing at the draft's precision (AC3), and — in the explicit
//! model-proof workflow — a `block_draft` request that decodes `off`'s greedy stream with its
//! report naming the draft (AC1, AC3).

use super::super::*;
use super::{synthetic_tokenizer_json, tiny_causal_from, tiny_llama_parts};
use crate::models::block_draft::tests::{
    dflash2_config_json, dspine_config_json, proof_device, tiny_prism_target, tiny_target,
    tiny_target_on, tiny_tensors,
};
use crate::primitives::head_rows::HeadRows;
use core_llm::{BlockDraftFormat, Speculative};

fn hybrid_provider(model: Qwen35Model) -> LlamaProvider {
    LlamaProvider {
        descriptor: descriptor_for_qwen35(model.config(), model.device()),
        model: Decoder::Qwen35(model),
        ..LlamaProvider::from_parts(
            tiny_causal_from(tiny_llama_parts().0, tiny_llama_parts().1),
            Tokenizer::from_json(&synthetic_tokenizer_json(50)).unwrap(),
            vec![],
        )
    }
}

/// A draft directory holding `config` and, with `weights`, its BF16 tensors.
fn draft_dir(config: &serde_json::Value, weights: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    if weights {
        let cfg = core_llm::BlockDraftConfig::from_json(config).unwrap();
        let tensors: std::collections::HashMap<String, Tensor> = tiny_tensors(&cfg)
            .into_iter()
            .map(|(k, t)| match t.dtype() {
                DType::F32 => (k, t.to_dtype(DType::BF16).unwrap()),
                _ => (k, t),
            })
            .collect();
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
    }
    dir
}

fn spec(dir: &tempfile::TempDir, precision: Option<Quantize>) -> LoadSpec {
    let mut spec = LoadSpec::dense(dir.path().to_string_lossy());
    spec.draft_quantize = precision;
    spec
}

fn format_of(config: &serde_json::Value) -> BlockDraftFormat {
    BlockDraftFormat::from_config(config).unwrap()
}

/// AC2: Candle refuses exactly the drafts MLX refuses, with the shared typed reasons — the
/// reason is `core_llm::block_draft_refusal_reason` over `BlockDraftConfig::mismatches`, the one
/// rule, verbatim — before any weight is read; the target loads alone and `block_draft` is not
/// advertised. A precision other than 8 or 4 bits and a target that is not the hybrid are refused
/// by kind, and a fitting draft of either format is resident at its precision and advertised up
/// to its trained block.
#[test]
fn block_drafts_are_accepted_and_refused_by_the_shared_typed_rule() {
    let narrow = || hybrid_provider(crate::models::qwen35::tests::text_model().1);
    for config in [dspine_config_json(), dflash2_config_json()] {
        let format = format_of(&config);
        // The tiny drafts are 64 wide over a 50-word vocabulary; the narrow hybrid is 32 wide.
        let mut wrong = config.clone();
        wrong["vocab_size"] = 64.into();
        wrong["dflash_config"]["mask_token_id"] = 60.into();
        wrong["dflash_config"]["target_layer_ids"] = serde_json::json!([0, 4]);
        let dir = draft_dir(&wrong, false);
        let mut refused = narrow();
        refused.attach_block_draft(&spec(&dir, None), format);
        let Decoder::Qwen35(target) = &refused.model else {
            unreachable!()
        };
        let expected = core_llm::block_draft_refusal_reason(
            &core_llm::BlockDraftConfig::from_json(&wrong)
                .unwrap()
                .mismatches(&core_llm::BlockDraftTarget {
                    hidden_size: 32,
                    num_layers: target.num_layers(),
                    vocab_size: 50,
                    tokenizer_vocab: 50,
                }),
        );
        let report = refused.draft_report.clone().unwrap();
        assert_eq!(report.refusal.as_deref(), Some(expected.as_str()));
        for kind in ["hidden_size", "vocabulary", "mask_token", "target_layers"] {
            assert!(
                expected.contains(&format!("block draft: {kind}:")),
                "{kind}"
            );
        }
        assert!(refused.block_draft.is_none());
        assert!(refused.load_record.fallbacks.contains(&expected));
        assert!(refused
            .descriptor
            .capabilities
            .proposer(SpeculativeProposer::BlockDraft)
            .is_none());

        let dir = draft_dir(&config, true);
        let mut nvfp4 = hybrid_provider(tiny_target());
        nvfp4.attach_block_draft(&spec(&dir, Some(Quantize::Nvfp4)), format);
        let why = nvfp4.draft_report.unwrap().refusal.unwrap();
        assert!(why.starts_with("block draft: precision:"), "{why}");
        let mut causal = LlamaProvider::from_parts(
            tiny_causal_from(tiny_llama_parts().0, tiny_llama_parts().1),
            Tokenizer::from_json(&synthetic_tokenizer_json(50)).unwrap(),
            vec![],
        );
        causal.attach_block_draft(&spec(&dir, None), format);
        let why = causal.draft_report.unwrap().refusal.unwrap();
        assert!(why.starts_with("block draft: target_family:"), "{why}");

        for (precision, label) in [(None, "q8"), (Some(Quantize::Q4), "q4")] {
            let mut resident = hybrid_provider(tiny_target());
            resident.attach_draft(&spec(&dir, precision), &Device::Cpu, None);
            let report = resident.draft_report.clone().unwrap();
            assert!(report.is_resident(), "{:?}", report.refusal);
            let block = resident.block_draft.as_ref().unwrap();
            assert_eq!(block.report.format, format.label());
            assert_eq!(block.report.precision, label);
            assert!(block.model.is_quantized());
            let caps = resident
                .descriptor
                .capabilities
                .proposer(SpeculativeProposer::BlockDraft)
                .unwrap();
            assert_eq!(caps.max_depth, 5, "the trained block of 6");
        }
    }
}

/// AC2: a Prism target is refused for DSpine by the MLX rule — any Prism load, whatever its LM
/// head — even when the snapshot kept its head dense; DFlash2, whose selector reads only the
/// head's logits, is accepted over the same target, as MLX accepts it.
#[test]
fn a_prism_load_with_a_dense_head_refuses_dspine_and_accepts_dflash2() {
    let target = tiny_prism_target();
    assert!(target.is_prism());
    assert_eq!(
        target.head_rows().unwrap().label(),
        "dense",
        "the head stayed dense"
    );
    assert!(!tiny_target().is_prism());

    let dir = draft_dir(&dspine_config_json(), true);
    let mut dspine = hybrid_provider(tiny_prism_target());
    dspine.attach_draft(&spec(&dir, None), &Device::Cpu, None);
    let why = dspine.draft_report.unwrap().refusal.unwrap();
    assert!(why.starts_with("block draft: target_family:"), "{why}");
    assert!(why.contains("Prism"), "{why}");
    assert!(dspine.block_draft.is_none());
    assert!(dspine.load_record.fallbacks.contains(&why));

    let dir = draft_dir(&dflash2_config_json(), true);
    let mut dflash2 = hybrid_provider(tiny_prism_target());
    dflash2.attach_draft(&spec(&dir, None), &Device::Cpu, None);
    let report = dflash2.draft_report.unwrap();
    assert!(report.is_resident(), "{:?}", report.refusal);
    assert_eq!(
        dflash2.block_draft.unwrap().report.format,
        BlockDraftFormat::DFlash2.label()
    );
}

/// A block-draft route's captured verify steps stage the tapped layers' outputs (`layers ×
/// hidden` per row), so the graph workspace prices that width, not one hidden row; with the graph
/// runner off nothing changes.
#[test]
fn a_block_draft_route_prices_its_tapped_rows_in_the_graph_workspace() {
    let decoder = Decoder::Qwen35(tiny_target());
    let route = DecodeRoute::Engine {
        proposer: ProposerKind::BlockDraft,
        drafts: 3,
    };
    let priced = |graphs, taps| {
        priced_request_bytes_tapped(&decoder, route, 12, 16, 0, graphs, taps).unwrap()
    };
    let geometry = decoder.step_memory_geometry(3);
    let (hidden, tap) = (geometry.hidden_size, 3 * geometry.hidden_size);
    // Σ_{M=1}^{K+1} M token-rows in two scopes, each staging `tap - hidden` more elements.
    let rows = 4 * 5 / 2 * 2;
    assert_eq!(
        priced(true, Some(tap)) - priced(true, Some(hidden)),
        rows * (tap - hidden) * geometry.element_bytes
    );
    assert_eq!(priced(true, None), priced(true, Some(hidden)));
    assert_eq!(priced(false, Some(tap)), priced(false, None));
    let reach = 12 + 16 + 3;
    assert_eq!(
        crate::decode::graph_workspace_admission_bytes_with_hidden_row(&geometry, reach, 4, tap)
            .unwrap()
            - crate::decode::graph_workspace_admission_bytes(&geometry, reach, 4).unwrap(),
        rows * (tap - hidden) * geometry.element_bytes
    );
}

/// AC3: a block draft's load is priced by the shared rule at its requested precision — its
/// resident arrays and twice its largest group's window, plus each GGML projection's CUDA row
/// padding — and a request on its route adds the shared request pricing.
#[test]
fn a_block_draft_is_priced_at_its_precision() {
    for config in [dspine_config_json(), dflash2_config_json()] {
        let dir = draft_dir(&config, false);
        let cfg = core_llm::BlockDraftConfig::from_json(&config).unwrap();
        let device = |p: Option<Quantize>| {
            block_draft_load_estimate(&spec(&dir, p), true)
                .unwrap()
                .unwrap()
                .device_required_bytes
                .unwrap()
        };
        let (q8, q4) = (device(None), device(Some(Quantize::Q4)));
        // DSpine adds the head-row address table: one 8-byte entry per vocabulary row.
        let table = match cfg.format {
            BlockDraftFormat::DSpine => 50 * 8,
            BlockDraftFormat::DFlash2 => 0,
        };
        let padded = |p, bytes: u64| {
            cfg.load_bytes(p) + bytes * cfg.quantized_matrices().len() as u64 + table
        };
        assert_eq!(q8, padded(core_llm::BlockDraftPrecision::Q8, 544));
        assert_eq!(q4, padded(core_llm::BlockDraftPrecision::Q4, 288));
        assert!(q4 < q8);
        assert!(
            block_draft_load_estimate(&spec(&dir, Some(Quantize::Nvfp4)), true)
                .unwrap()
                .is_err()
        );
        let draft = crate::models::block_draft::tests::tiny_draft(format_of(&config), None);
        let shared = cfg.request_bytes(12, 16, 3, 2).unwrap();
        let target = tiny_target();
        let head = target.head_rows().unwrap();
        assert_eq!(head.label(), "dense");
        let priced = block_draft_request_bytes(&draft, Some(head), 2, 12, 16, 3).unwrap();
        match cfg.format {
            // The refinement reads each draft row's 4 candidate rows of the 64-wide head and
            // writes 3 × 4 × 4 scores — never the 3 × 4 × 50 whole-head logits.
            BlockDraftFormat::DSpine => {
                assert_eq!(priced, shared + 3 * 4 * 64 * 4 + 3 * 4 * 4 * 6);
                let whole = HeadRows::Full("test");
                let fallback = block_draft_request_bytes(&draft, Some(&whole), 2, 12, 16, 3);
                assert_eq!(fallback, Some(shared + 3 * 4 * 50 * 6 + 3 * 4 * 4 * 6));
                assert_eq!(block_draft_request_bytes(&draft, None, 2, 12, 16, 3), None);
            }
            BlockDraftFormat::DFlash2 => assert_eq!(priced, shared),
        }
    }
    let target = LoadSpec::dense("/nonexistent");
    assert!(block_draft_load_estimate(&target, true).is_none());
}

fn request(speculative: Speculative, sampling: Sampling) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user("t3 t9 t4 t11 t3 t9 t4 t11 t3")],
        sampling,
        max_new_tokens: 16,
        seed: Some(3),
        speculative: Some(speculative),
        ..Default::default()
    }
}

fn run(provider: &LlamaProvider, req: &TextLlmRequest) -> (TextLlmOutput, Vec<u32>) {
    let mut ids = Vec::new();
    let out = provider
        .generate(req, &mut |ev| {
            if let CoreEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .unwrap();
    (out, ids)
}

/// AC1/AC3 through the provider: a resident draft of either format, loaded from its directory,
/// runs a greedy `block_draft` request that decodes exactly `off`'s stream, its report naming
/// the proposer, the depth, the draft, its precision, the verify passes and the accepted tokens;
/// a seeded sampled request runs it too and reproduces. On a provider whose draft was refused the
/// same request decodes without a proposer, by name.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_resident_block_draft_runs_block_draft_and_is_named() {
    let (_guard, device) = proof_device();
    for config in [dspine_config_json(), dflash2_config_json()] {
        let format = format_of(&config);
        let dir = draft_dir(&config, true);
        let mut provider = hybrid_provider(tiny_target_on(&device, true));
        provider.attach_draft(&spec(&dir, None), &device, None);
        let block = Speculative::proposer(SpeculativeProposer::BlockDraft, 3);
        let (out, ids) = run(&provider, &request(block, Sampling::greedy()));
        let (off, off_ids) = run(&provider, &request(Speculative::Off, Sampling::greedy()));
        assert_eq!(ids, off_ids, "{format:?}");
        assert_eq!(out.text, off.text);
        let report = out.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::BlockDraft);
        assert_eq!(report.path, "block_draft");
        assert_eq!(report.draft_tokens, Some(3));
        assert_eq!(
            report.draft,
            Some(core_llm::DecodeDraft {
                source: dir.path().to_string_lossy().into_owned(),
                format: format.label().into(),
                precision: "q8".into(),
            })
        );
        assert!(report.verify_steps > 0 && report.proposed_tokens > 0);
        assert!(off.decode.unwrap().draft.is_none());

        let sampled = Sampling {
            temperature: 0.8,
            top_p: 0.95,
            ..Sampling::greedy()
        };
        let (a, a_ids) = run(&provider, &request(block, sampled));
        let (_, b_ids) = run(&provider, &request(block, sampled));
        assert_eq!(a_ids, b_ids, "{format:?}: a seeded sampled run reproduces");
        let report = a.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::BlockDraft);
        assert!(report.draft.is_some());

        let mut wrong = config.clone();
        wrong["vocab_size"] = 64.into();
        let refused_dir = draft_dir(&wrong, false);
        let mut refused = hybrid_provider(tiny_target_on(&device, true));
        refused.attach_block_draft(&spec(&refused_dir, None), format);
        let (out, ids) = run(&refused, &request(block, Sampling::greedy()));
        let report = out.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::None);
        assert!(report.draft.is_none());
        assert!(!report.fallbacks.is_empty(), "the downgrade is named");
        assert_eq!(ids.len(), 16);
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::block_draft::a_resident_block_draft_runs_block_draft_and_is_named");
}

/// With the graph runner off, a `block_draft` route's eager verify steps still stage their tapped
/// rows, and they are priced (SW-7198 review): the request beside the target
/// ([`block_draft_request_bytes`] over [`core_llm::BlockDraftConfig::request_bytes`]) holds the
/// prompt's tapped rows (capped at the window) and a verify step's `K + 1`, `tap_width` wide, on
/// top of the target's own price.
#[test]
fn an_eager_block_draft_route_prices_its_tapped_rows() {
    let dir = draft_dir(&dspine_config_json(), true);
    let mut provider = hybrid_provider(tiny_target());
    provider.attach_draft(&spec(&dir, None), &Device::Cpu, None);
    let block = provider.block_draft.as_ref().unwrap();
    let tap = block.model.config().tap_width() as u64;
    let window = block.model.config().context_rows().unwrap() as u64;
    let route = DecodeRoute::Engine {
        proposer: ProposerKind::BlockDraft,
        drafts: 3,
    };
    let beside = provider.draft_bytes(route, 12, 16).unwrap();
    assert!(beside >= (12u64.min(window) + 4) * tap * 2, "{beside}");
    let target =
        priced_request_bytes_tapped(&provider.model, route, 12, 16, 0, false, Some(tap)).unwrap();
    let _graphs = crate::decode::cuda_graphs_scope(Some(false));
    assert_eq!(
        provider.request_bytes(route, 12, 16, 0),
        Some(target + beside)
    );
}

/// The draft's request and a block-draft tree are priced at the target's real compute dtype: an
/// f32 target (Candle's CPU) holds exactly twice a bf16 target's tapped rows — the eager verify
/// and prompt-window taps beside the target ([`block_draft_request_bytes`]) and every tree row's
/// taps and their gathered copy ([`tree_request_bytes`]).
#[test]
fn an_f32_target_prices_twice_a_bf16_targets_tapped_rows() {
    let config = dflash2_config_json();
    let cfg = core_llm::BlockDraftConfig::from_json(&config).unwrap();
    let draft = crate::models::block_draft::tests::tiny_draft(format_of(&config), None);
    let tap = cfg.tap_width() as u64;
    let f32_target = tiny_target();
    assert_eq!(
        f32_target.compute_dtype(),
        DType::F32,
        "the CPU's compute dtype"
    );
    let act = f32_target.compute_dtype().size_in_bytes() as u64;
    let at = |element: u64| block_draft_request_bytes(&draft, None, element, 12, 16, 3).unwrap();
    let fixed = cfg.request_bytes(12, 16, 3, 0).unwrap();
    assert_eq!(at(act) - fixed, 2 * (at(2) - fixed));
    // The 8-row window of the 12-token prompt, then a verify step of 4 rows, at 2 bytes.
    assert!(at(2) - fixed >= (8 + 4) * tap * 2);

    let bf16_target = {
        let json = {
            let mut v = crate::models::qwen35::tests::cfg_json();
            v["text_config"]["hidden_size"] = 64.into();
            v["text_config"]["intermediate_size"] = 128.into();
            v
        };
        let qcfg = crate::models::Qwen35Config::from_json(&json).unwrap();
        let w = crate::models::qwen35::tests::synthetic_weights(&qcfg);
        Qwen35Model::from_weights_dtype(&w, "model.language_model", qcfg, None, DType::BF16)
            .unwrap()
    };
    assert_eq!(bf16_target.compute_dtype(), DType::BF16);
    let tree_taps = |model: Qwen35Model| {
        let decoder = Decoder::Qwen35(model);
        let block = ProposerKind::BlockDraft;
        tree_request_bytes(&decoder, block, 4, 16, 100, 50, tap).unwrap()
            - tree_request_bytes(&decoder, block, 4, 16, 100, 50, 0).unwrap()
    };
    let (f32_taps, bf16_taps) = (tree_taps(f32_target), tree_taps(bf16_target));
    assert_eq!(bf16_taps, 17 * tap * 2 * 2);
    assert_eq!(f32_taps, 2 * bf16_taps);
}

/// The tiny target ([`tiny_target_on`]'s random-head hybrid, 64 wide over 50 ids) with a
/// [`synthetic_vision_tower`](super::synthetic_vision_tower) written as a Qwen3.5 VLM snapshot
/// (keep the directory alive): the image fixture the tiny drafts fit.
fn tiny_vl_target_snapshot() -> tempfile::TempDir {
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    let mut json = crate::models::qwen35::tests::cfg_json();
    json["text_config"]["hidden_size"] = 64.into();
    json["text_config"]["intermediate_size"] = 128.into();
    json["text_config"]["mtp_num_hidden_layers"] = 0.into();
    // Room for the 64 image tokens beside the prompt and the continuation.
    json["text_config"]["max_position_embeddings"] = 512.into();
    let cfg = crate::models::Qwen35Config::from_json(&json).unwrap();
    let vocab = cfg.vocab_size as usize;
    let mut tensors = crate::models::qwen35::tests::synthetic_weights(&cfg).into_map();
    let mut rng = SplitMix64::new(0x7193_7A26);
    for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
        let data: Vec<f32> = (0..vocab * 64)
            .map(|_| (rng.next_f32() - 0.5) * 0.8)
            .collect();
        tensors.insert(
            name.into(),
            Tensor::from_vec(data, (vocab, 64), &Device::Cpu).unwrap(),
        );
    }
    let (vision_config, vision) = super::synthetic_vision_tower(64);
    tensors.extend(vision);
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": super::vl_special_ids(vocab).1,
        "text_config": json["text_config"],
        "vision_config": vision_config,
    });
    super::write_vl_snapshot(&config, &tensors, vocab)
}

/// SW-7198 AC1 through the provider: a Qwen3.5 VLM with a resident draft of either format runs a
/// greedy image + text `block_draft` request — the image at the head of the prompt and mid-prompt
/// — that decodes exactly `off`'s stream: the provider prefills the fused prompt tapped at the
/// draft's layers, warms the draft from those rows and shifts its positions by the prompt's
/// M-RoPE delta, and the report names the draft with no `block_draft` fallback. The same request
/// under a tree bound verifies linearly, by name (the multimodal rule), and is still `off`'s.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_block_draft_decodes_an_image_request_as_off_does() {
    use core_llm::SpeculativeVerify;
    let (_guard, _device) = proof_device();
    let target = tiny_vl_target_snapshot();
    for config in [dspine_config_json(), dflash2_config_json()] {
        let format = format_of(&config);
        let draft = draft_dir(&config, true);
        let spec = LoadSpec::dense(target.path().to_string_lossy())
            .with_draft(draft.path().to_string_lossy());
        let provider = LlamaProvider::load(&spec).unwrap();
        assert!(provider.descriptor.capabilities.supports_vision);
        assert!(
            provider.block_draft.is_some(),
            "{:?}",
            provider.draft_report
        );
        let block = Speculative::proposer(SpeculativeProposer::BlockDraft, 3);
        let (mut proposed, mut accepted) = (0, 0);
        for prompt in super::image_fixture_prompts() {
            let label = format!("{format:?} {}", prompt.id);
            let req = |speculative| {
                core_llm_testkit::bench_request(&prompt, speculative, &Sampling::greedy(), 24)
            };
            let (on, out) = super::token_events(&provider, &req(block));
            let (off, _) = super::token_events(&provider, &req(Speculative::Off));
            assert_eq!(on, off, "{label}: the off stream");
            assert!(
                out.usage.prompt_tokens > 64,
                "{label}: the image tokens are in the prompt"
            );
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, ProposerKind::BlockDraft, "{label}");
            assert_eq!(report.draft.unwrap().format, format.label(), "{label}");
            assert!(
                report.fallbacks.iter().all(|f| !f.contains("block_draft")),
                "{label}: {:?}",
                report.fallbacks
            );
            assert!(report.verify_steps > 0, "{label}");
            proposed += report.proposed_tokens;
            accepted += report.accepted_tokens;

            let tree = TextLlmRequest {
                speculative_verify: SpeculativeVerify::Tree { max_rows: 16 },
                ..req(block)
            };
            let (treed, out) = super::token_events(&provider, &tree);
            assert_eq!(treed, off, "{label}: under a tree bound");
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, ProposerKind::BlockDraft, "{label}");
            assert!(
                report
                    .fallbacks
                    .iter()
                    .any(|f| f == super::super::TREE_MULTIMODAL_LINEAR),
                "{label}: {:?}",
                report.fallbacks
            );
        }
        assert!(proposed > 0, "{format:?}: the draft proposed");
        println!("{format:?}: {accepted} of {proposed} drafts accepted on the image prompts");
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED provider::tests::block_draft::a_block_draft_decodes_an_image_request_as_off_does");
}
