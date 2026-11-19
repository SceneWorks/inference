//! Real-weight block-draft smokes on Candle (epic SW-F360, story SW-7197), `#[ignore]`d: the
//! Qwen3.8-27B Candle describer tier with the DSpine or the DFlash2 draft resident beside it,
//! through the provider on the CUDA device, greedy and sampled.
//!
//! The model-proof workflow stages the pinned snapshots and binds their paths:
//!
//! * target: `SceneWorks/qwen3.8-27b-abliterated-candle` @
//!   `accaccc109912b9b194f7d5a47f2123fdc70672e`, its `q4/` tier (`CANDLE_LLM_BLOCK_DRAFT_TARGET`);
//! * DSpine draft: `SceneWorks/qwen3.8-27b-dspine-b16` @
//!   `8e450d7faf38238b64580163a7d3dd7e44af759d` (`CANDLE_LLM_DSPINE_SOURCE`), the SceneWorks
//!   re-host of `tencent/Qwen3.8-27B-DSpine-b16`;
//! * DFlash2 draft: `SceneWorks/qwen3.8-27b-dflash2` @
//!   `9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9` (`CANDLE_LLM_DFLASH2_SOURCE`), the re-host of
//!   `incoai/Qwen3.8-27B-DFlash2`.
//!
//! ```text
//! CANDLE_LLM_BLOCK_DRAFT_TARGET=… CANDLE_LLM_DSPINE_SOURCE=… CANDLE_LLM_DFLASH2_SOURCE=… \
//!   cargo test -p candle-llm --features cuda --test block_draft -- --ignored --nocapture
//! ```
//!
//! `CANDLE_LLM_BLOCK_DRAFT_PRECISION=q4` loads the draft at 4-bit (default 8-bit). Each run
//! asserts the draft is resident, advertised and named at its precision, that a greedy
//! `block_draft` request commits more than one token per target pass, and that a seeded sampled
//! request runs the draft and reproduces; it prints the tokens per target pass and reports the
//! first position where the greedy stream differs from the same provider with speculation off (a
//! multi-token verify rounds a few BF16 ULP differently from the single-token step, so real
//! weights track rather than bit-match where a near-tie flips — exact agreement is proven on the
//! tiny fixtures).

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_core::Tensor, ()>::Err(()));
};

use candle_llm::LlamaProvider;
use core_llm::{
    BlockDraftFormat, LoadSpec, Message, ProposerKind, Quantize, Sampling, Speculative,
    SpeculativeProposer, StreamEvent, TextLlm, TextLlmOutput, TextLlmRequest, ThinkingMode,
};

const TARGET: &str = "CANDLE_LLM_BLOCK_DRAFT_TARGET";

fn request(speculative: Speculative, sampling: Sampling) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(
            "Describe, in four short paragraphs, how a lighthouse keeper's day changes from \
             summer to winter.",
        )],
        sampling,
        max_new_tokens: 192,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        speculative: Some(speculative),
        ..Default::default()
    }
}

fn generate(provider: &LlamaProvider, req: &TextLlmRequest) -> (Vec<u32>, TextLlmOutput) {
    let mut ids = Vec::new();
    let out = provider
        .generate(req, &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .unwrap();
    (ids, out)
}

fn path(var: &str) -> String {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| panic!("set {var}"))
}

/// Load the target with the `format` draft in `source_var` and run the greedy and sampled
/// `block_draft` requests.
fn block_draft_on_the_27b(format: BlockDraftFormat, source_var: &str) {
    let draft = path(source_var);
    let precision = match std::env::var("CANDLE_LLM_BLOCK_DRAFT_PRECISION").as_deref() {
        Ok("q4") => Some(Quantize::Q4),
        _ => None,
    };
    let mut spec = LoadSpec::dense(path(TARGET)).with_draft(draft.clone());
    spec.draft_quantize = precision;
    let provider = LlamaProvider::load(&spec).unwrap();
    let report = provider.load_report().unwrap().draft.unwrap();
    assert!(report.is_resident(), "{:?}", report.refusal);
    let depth = provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::BlockDraft)
        .expect("block_draft advertised")
        .recommended_depth;
    let label = if precision == Some(Quantize::Q4) {
        "q4"
    } else {
        "q8"
    };

    let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
    let (ids, out) = generate(&provider, &request(block, Sampling::greedy()));
    let decode = out.decode.unwrap();
    assert_eq!(decode.proposer, ProposerKind::BlockDraft);
    assert_eq!(decode.draft_tokens, Some(depth));
    let named = decode.draft.clone().expect("the draft is named");
    assert_eq!(
        (named.format.as_str(), named.precision.as_str()),
        (format.label(), label)
    );
    let per_pass =
        (decode.accepted_tokens + decode.verify_steps) as f64 / decode.verify_steps as f64;
    println!(
        "{} {label} depth {depth}: {} tokens, {} target passes, {per_pass:.2} tokens per target \
         pass, {} accepted of {} proposed, graphs {:?}",
        format.label(),
        ids.len(),
        decode.verify_steps,
        decode.accepted_tokens,
        decode.proposed_tokens,
        decode.cuda_graphs,
    );
    assert!(per_pass > 1.0, "the draft is accepted: {per_pass:.2}");

    let (off, _) = generate(&provider, &request(Speculative::Off, Sampling::greedy()));
    let first = ids.iter().zip(&off).position(|(a, b)| a != b);
    println!("{}: first difference from off at {first:?}", format.label());

    let sampled = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        ..Sampling::greedy()
    };
    let (a, out) = generate(&provider, &request(block, sampled));
    let (b, _) = generate(&provider, &request(block, sampled));
    assert_eq!(a, b, "a seeded sampled run reproduces");
    let decode = out.decode.unwrap();
    assert_eq!(decode.proposer, ProposerKind::BlockDraft);
    assert!(decode.draft.is_some());
    println!(
        "{} sampled: {} tokens, {:.2} tokens per target pass",
        format.label(),
        a.len(),
        (decode.accepted_tokens + decode.verify_steps) as f64 / decode.verify_steps as f64,
    );
}

/// The 27B with the DSpine draft on CUDA (AC1, AC3).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dspine_accepts_more_than_one_token_per_pass() {
    block_draft_on_the_27b(BlockDraftFormat::DSpine, "CANDLE_LLM_DSPINE_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dspine_accepts_more_than_one_token_per_pass"
    );
}

/// The 27B with the DFlash2 draft on CUDA (AC1, AC3).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dflash2_accepts_more_than_one_token_per_pass() {
    block_draft_on_the_27b(BlockDraftFormat::DFlash2, "CANDLE_LLM_DFLASH2_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dflash2_accepts_more_than_one_token_per_pass"
    );
}
