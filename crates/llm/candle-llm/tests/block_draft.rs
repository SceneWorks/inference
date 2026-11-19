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
//! asserts the draft is resident, advertised and named at its precision; that a greedy
//! `block_draft` request on a cold prompt, the same request again (its prompt prefix restored
//! from the cross-turn cache) and a seeded sampled request each reach the acceptance floors
//! ([`MIN_TOKENS_PER_PASS`], [`MIN_ACCEPTED_FRACTION`], set from the healthy MLX and Candle
//! measurements); and that the sampled request reproduces. It prints each run's acceptance and
//! reports the first position where the greedy stream differs from the same provider with
//! speculation off (a
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
    // The first request prefills the prompt cold, so the draft's context starts with the prompt's
    // tapped rows. The repeat restores the prompt's prefix from the cross-turn cache (when the
    // load kept one) and the draft reads only the rows past it. Both are held to the floor: a
    // context built from a cold prompt prefill must draft as well as one built from commits.
    let (ids, out) = generate(&provider, &request(block, Sampling::greedy()));
    let decode = out.decode.unwrap();
    assert_eq!(decode.proposer, ProposerKind::BlockDraft);
    assert_eq!(decode.draft_tokens, Some(depth));
    let named = decode.draft.clone().expect("the draft is named");
    assert_eq!(
        (named.format.as_str(), named.precision.as_str()),
        (format.label(), label)
    );
    println!(
        "{} {label} depth {depth}, graphs {:?}",
        format.label(),
        decode.cuda_graphs
    );
    assert_accepts(&format!("{} greedy", format.label()), ids.len(), &decode);

    let (off, _) = generate(&provider, &request(Speculative::Off, Sampling::greedy()));
    let first = ids.iter().zip(&off).position(|(a, b)| a != b);
    println!("{}: first difference from off at {first:?}", format.label());

    let (again, out) = generate(&provider, &request(block, Sampling::greedy()));
    let decode = out.decode.unwrap();
    assert_accepts(
        &format!("{} greedy repeat", format.label()),
        again.len(),
        &decode,
    );

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
    assert_accepts(&format!("{} sampled", format.label()), a.len(), &decode);
}

/// The least tokens a greedy or sampled `block_draft` pass commits on this prompt at depth 4,
/// draft and accepted together. The healthy measurements on these weights: MLX DSpine 2.09
/// tokens per pass (99 of about 368 drafts accepted, 27%), MLX DFlash2 2.59, Candle DFlash2 2.36
/// (110 of 317, 35%) and Candle DSpine sampled 2.10. 1.6 keeps 0.6 accepted drafts per pass, a
/// little over half of the lowest healthy run's 1.08, as headroom for the backends' rounding
/// (CUDA Q4_K against MLX affine 4-bit, BF16 reduction order); a draft that is right only by
/// chance (the SW-7197 defect: 1.04, 7 of 726) fails it by fifteen times.
const MIN_TOKENS_PER_PASS: f64 = 1.6;

/// The least fraction of proposed drafts accepted: 15%, a little over half of the lowest healthy
/// run's 27% (MLX DSpine), the same headroom as [`MIN_TOKENS_PER_PASS`].
const MIN_ACCEPTED_FRACTION: f64 = 0.15;

/// Print and check one run's acceptance against the floors.
fn assert_accepts(what: &str, tokens: usize, decode: &core_llm::DecodeReport) {
    let per_pass =
        (decode.accepted_tokens + decode.verify_steps) as f64 / decode.verify_steps.max(1) as f64;
    let fraction = decode.accepted_tokens as f64 / decode.proposed_tokens.max(1) as f64;
    println!(
        "{what}: {tokens} tokens, {} target passes, {per_pass:.2} tokens per target pass, {} \
         accepted of {} proposed ({:.1}%), prefix {} ({} restored tokens)",
        decode.verify_steps,
        decode.accepted_tokens,
        decode.proposed_tokens,
        fraction * 100.0,
        decode.prefix_cache.path,
        decode.prefix_hit_tokens,
    );
    assert!(
        per_pass >= MIN_TOKENS_PER_PASS,
        "{what}: {per_pass:.2} tokens per target pass, below {MIN_TOKENS_PER_PASS}"
    );
    assert!(
        fraction >= MIN_ACCEPTED_FRACTION,
        "{what}: {:.1}% of drafts accepted, below {:.0}%",
        fraction * 100.0,
        MIN_ACCEPTED_FRACTION * 100.0
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
