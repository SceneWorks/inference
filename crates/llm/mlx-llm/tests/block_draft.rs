//! Real-weight block-draft smoke (epic SW-F360, story SW-7193), `#[ignore]`d: the Qwen3.8-27B
//! target with the DSpine draft resident beside it, through the provider.
//!
//! It reads the pinned snapshots under the explicit models root `MLX_GEN_MODELS_ROOT` (a Hugging
//! Face Hub-layout root holding `models--*/snapshots`; inference never derives a cache location,
//! epic 13657):
//!
//! * target: `SceneWorks/qwen3.8-27b-abliterated-mlx` @ `76f3c445cd5347fdb85fc24066350724fe272017`,
//!   its `q4/` tier;
//! * draft: `SceneWorks/qwen3.8-27b-dspine-b16` @ `8e450d7faf38238b64580163a7d3dd7e44af759d`, the
//!   SceneWorks re-host of `tencent/Qwen3.8-27B-DSpine-b16` @ `260a6c23` (the same
//!   `model.safetensors` and `config.json` bytes).
//!
//! `MLX_LLM_BLOCK_DRAFT_TARGET` / `MLX_LLM_BLOCK_DRAFT_SOURCE` override either path:
//!
//! ```text
//! MLX_GEN_MODELS_ROOT=/path/to/hub \
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

/// The pinned target: the SceneWorks Qwen3.8-27B at its 4-bit tier, under `MLX_GEN_MODELS_ROOT`.
const TARGET_SNAPSHOT: &str = "models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/\
                               76f3c445cd5347fdb85fc24066350724fe272017/q4";

/// The pinned draft: the SceneWorks re-host of `tencent/Qwen3.8-27B-DSpine-b16`, under
/// `MLX_GEN_MODELS_ROOT`.
const DRAFT_SNAPSHOT: &str = "models--SceneWorks--qwen3.8-27b-dspine-b16/snapshots/\
                              8e450d7faf38238b64580163a7d3dd7e44af759d";

/// `override_var` when set, else the pinned `snapshot` under `MLX_GEN_MODELS_ROOT` when that
/// directory exists.
fn snapshot(override_var: &str, snapshot: &str) -> Result<String, String> {
    if let Ok(path) = std::env::var(override_var) {
        return Ok(path);
    }
    let root = std::env::var("MLX_GEN_MODELS_ROOT")
        .map_err(|_| format!("set {override_var} or MLX_GEN_MODELS_ROOT"))?;
    let path = std::path::Path::new(&root).join(snapshot);
    if path.is_dir() {
        Ok(path.to_string_lossy().into_owned())
    } else {
        Err(format!("{} is not provisioned", path.display()))
    }
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DSpine draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dspine_accepts_more_than_one_token_per_pass() {
    let (target, draft) = match (
        snapshot("MLX_LLM_BLOCK_DRAFT_TARGET", TARGET_SNAPSHOT),
        snapshot("MLX_LLM_BLOCK_DRAFT_SOURCE", DRAFT_SNAPSHOT),
    ) {
        (Ok(target), Ok(draft)) => (target, draft),
        (target, draft) => {
            let missing = [target.err(), draft.err()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("; ");
            crate::model_inference_device::record_model_inference_prerequisite_missing(&missing);
            panic!("{missing}");
        }
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
