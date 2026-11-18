//! Real-weight block-draft smoke (epic SW-F360, story SW-7193), `#[ignore]`d: the Qwen3.8-27B
//! target with the DSpine draft resident beside it, through the provider.
//!
//! ```text
//! MLX_LLM_BLOCK_DRAFT_TARGET=/path/to/qwen3.8-27b-abliterated-mlx/q4 \
//! MLX_LLM_BLOCK_DRAFT_SOURCE=/path/to/Qwen3.8-27B-DSpine-b16 \
//!   cargo test -p mlx-llm --test integration -- block_draft:: --ignored --nocapture
//! ```
//!
//! `MLX_LLM_BLOCK_DRAFT_PRECISION=q4` loads the draft at 4-bit (default 8-bit). The run asserts
//! the draft is resident and named at its precision, and that a greedy `block_draft` request
//! accepts more than one draft token per target pass; it reports the first position where the
//! greedy stream differs from the same engine with speculation off (a multi-token verify rounds
//! a few BF16 ULP differently from the single-token step, so real weights track rather than
//! bit-match where a near-tie flips — exact agreement is proven on the tiny fixtures).

use core_llm::{
    LoadSpec, Message, ProposerKind, Quantize, Sampling, Speculative, SpeculativeProposer,
    StreamEvent, TextLlm, TextLlmRequest, ThinkingMode,
};

fn generate(
    provider: &mlx_llm::LlamaProvider,
    speculative: Speculative,
) -> (Vec<u32>, core_llm::TextLlmOutput) {
    let req = TextLlmRequest {
        messages: vec![Message::user(
            "Describe, in four short paragraphs, how a lighthouse keeper's day changes from \
             summer to winter.",
        )],
        sampling: Sampling::greedy(),
        max_new_tokens: 192,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        speculative: Some(speculative),
        ..Default::default()
    };
    let mut ids = Vec::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .unwrap();
    (ids, out)
}

#[test]
#[ignore = "needs the Qwen3.8-27B target and its DSpine draft via MLX_LLM_BLOCK_DRAFT_TARGET/SOURCE"]
fn qwen38_27b_dspine_accepts_more_than_one_token_per_pass() {
    let (Ok(target), Ok(draft)) = (
        std::env::var("MLX_LLM_BLOCK_DRAFT_TARGET"),
        std::env::var("MLX_LLM_BLOCK_DRAFT_SOURCE"),
    ) else {
        crate::model_inference_device::record_model_inference_prerequisite_missing(
            "MLX_LLM_BLOCK_DRAFT_TARGET and MLX_LLM_BLOCK_DRAFT_SOURCE are required",
        );
        panic!("set MLX_LLM_BLOCK_DRAFT_TARGET and MLX_LLM_BLOCK_DRAFT_SOURCE");
    };
    let precision = match std::env::var("MLX_LLM_BLOCK_DRAFT_PRECISION").as_deref() {
        Ok("q4") => Some(Quantize::Q4),
        _ => None,
    };
    let mut spec = LoadSpec::dense(target).with_draft(draft.clone());
    spec.draft_quantize = precision;
    let provider = mlx_llm::LlamaProvider::load(&spec).unwrap();
    let report = provider.load_report().unwrap().draft.unwrap();
    assert!(report.is_resident(), "{:?}", report.refusal);
    assert!(provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::BlockDraft)
        .is_some());

    let (ids, out) = generate(
        &provider,
        Speculative::proposer(SpeculativeProposer::BlockDraft, 4),
    );
    let decode = out.decode.unwrap();
    let (off_ids, _) = generate(&provider, Speculative::Off);
    let first_difference = ids.iter().zip(&off_ids).position(|(a, b)| a != b);
    let per_pass = decode.mean_accepted_length().unwrap_or(0.0);
    println!(
        "BLOCK_DRAFT proposer={} draft={:?} target_passes={} accepted={} accepted_per_pass={per_pass:.2} \
         first_difference={first_difference:?} tokens={} off_tokens={}",
        decode.proposer.label(),
        decode.draft,
        decode.verify_steps,
        decode.accepted_tokens,
        ids.len(),
        off_ids.len(),
    );
    assert_eq!(decode.proposer, ProposerKind::BlockDraft);
    let named = decode.draft.unwrap();
    assert_eq!(named.source, draft);
    assert_eq!(named.format, "dspine");
    assert_eq!(
        named.precision,
        if precision == Some(Quantize::Q4) {
            "q4"
        } else {
            "q8"
        }
    );
    assert!(
        per_pass > 1.0,
        "accepted tokens per target pass {per_pass:.2} must exceed 1"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dspine_accepts_more_than_one_token_per_pass"
    );
}
