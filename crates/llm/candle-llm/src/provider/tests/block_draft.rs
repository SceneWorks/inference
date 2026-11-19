//! The block draft through the Candle provider (SW-F360, SW-7197): the shared typed refusals
//! (AC2), the load and request pricing at the draft's precision (AC3), and — in the explicit
//! model-proof workflow — a `block_draft` request that decodes `off`'s greedy stream with its
//! report naming the draft (AC1, AC3).

use super::super::*;
use super::{synthetic_tokenizer_json, tiny_causal_from, tiny_llama_parts};
use crate::models::block_draft::tests::{
    dflash2_config_json, dspine_config_json, proof_device, tiny_target, tiny_target_on,
    tiny_tensors,
};
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
        let padded =
            |p, bytes: u64| cfg.load_bytes(p) + bytes * cfg.quantized_matrices().len() as u64;
        assert_eq!(q8, padded(core_llm::BlockDraftPrecision::Q8, 544));
        assert_eq!(q4, padded(core_llm::BlockDraftPrecision::Q4, 288));
        assert!(q4 < q8);
        assert!(
            block_draft_load_estimate(&spec(&dir, Some(Quantize::Nvfp4)), true)
                .unwrap()
                .is_err()
        );
        let draft = crate::models::block_draft::tests::tiny_draft(format_of(&config), None);
        let shared = cfg.request_bytes(12, 16, 3).unwrap();
        let priced = block_draft_request_bytes(&draft, 12, 16, 3).unwrap();
        match cfg.format {
            BlockDraftFormat::DSpine => assert_eq!(priced, shared + 3 * 4 * 50 * 6),
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
