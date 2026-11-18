//! Real-weight block-draft smokes (epic SW-F360, stories SW-7193, SW-7194, SW-7195 and SW-7196),
//! `#[ignore]`d: the Qwen3.8-27B target with the DSpine or the DFlash2 draft resident beside it,
//! through the provider, on a text request and on an image + text request (the image generated
//! in the test).
//!
//! It reads the pinned snapshots under the explicit models root `MLX_GEN_MODELS_ROOT` (a Hugging
//! Face Hub-layout root holding `models--*/snapshots`; inference never derives a cache location,
//! epic 13657):
//!
//! * target: `SceneWorks/qwen3.8-27b-abliterated-mlx` @ `76f3c445cd5347fdb85fc24066350724fe272017`,
//!   its `q4/` tier;
//! * draft: `SceneWorks/qwen3.8-27b-dspine-b16` @ `8e450d7faf38238b64580163a7d3dd7e44af759d`, the
//!   SceneWorks re-host of `tencent/Qwen3.8-27B-DSpine-b16` @ `260a6c23` (the same
//!   `model.safetensors` and `config.json` bytes);
//! * DFlash2 draft: `SceneWorks/qwen3.8-27b-dflash2` @ `9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9`,
//!   the SceneWorks re-host of `incoai/Qwen3.8-27B-DFlash2` (the same weights).
//!
//! `MLX_LLM_BLOCK_DRAFT_TARGET` / `MLX_LLM_BLOCK_DRAFT_SOURCE` override either DSpine path, and
//! `MLX_LLM_DFLASH2_TARGET` / `MLX_LLM_DFLASH2_SOURCE` either DFlash2 path:
//!
//! ```text
//! MLX_GEN_MODELS_ROOT=/path/to/hub \
//!   cargo test -p mlx-llm --test integration -- block_draft:: --ignored --nocapture
//! ```
//!
//! `MLX_LLM_BLOCK_DRAFT_PRECISION=q4` (DFlash2: `MLX_LLM_DFLASH2_PRECISION=q4`) loads the draft
//! at 4-bit (default 8-bit). Each run asserts
//! the draft is resident and named at its precision, and that a greedy `block_draft` request
//! accepts more than one draft token per target pass; it reports the first position where the
//! greedy stream differs from the same engine with speculation off (a multi-token verify rounds
//! a few BF16 ULP differently from the single-token step, so real weights track rather than
//! bit-match where a near-tie flips — exact agreement is proven on the tiny fixtures).

use core_llm::{
    Content, ImageRef, LoadSpec, Message, ProposerKind, Quantize, Role, Sampling, Speculative,
    SpeculativeProposer, SpeculativeVerify, StreamEvent, TextLlm, TextLlmRequest, ThinkingMode,
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

/// The pinned DFlash2 draft: the SceneWorks re-host of `incoai/Qwen3.8-27B-DFlash2`, under
/// `MLX_GEN_MODELS_ROOT`.
const DFLASH2_SNAPSHOT: &str = "models--SceneWorks--qwen3.8-27b-dflash2/snapshots/\
                                9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9";

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

/// Load the pinned target (`target_var`) with the block draft (`source_var`, pinned at
/// `draft_snapshot`) at the precision `precision_var` names, and assert the draft is resident and
/// `block_draft` advertised. Returns the provider, the draft's path and its requested precision.
fn load_with_block_draft(
    target_var: &str,
    source_var: &str,
    precision_var: &str,
    draft_snapshot: &str,
) -> (mlx_llm::LlamaProvider, String, Option<Quantize>) {
    let (target, draft) = match (
        snapshot(target_var, TARGET_SNAPSHOT),
        snapshot(source_var, draft_snapshot),
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
    let precision = match std::env::var(precision_var).as_deref() {
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
    (provider, draft, precision)
}

/// Load the pinned target with the block draft ([`load_with_block_draft`]) and assert the draft
/// is named as `format` at its precision, and that a greedy `block_draft` request of `depth`
/// drafts per step accepts more than one draft token per target pass.
fn accepts_more_than_one_token_per_pass(
    target_var: &str,
    source_var: &str,
    precision_var: &str,
    draft_snapshot: &str,
    format: &str,
    depth: u32,
) {
    let (provider, draft, precision) =
        load_with_block_draft(target_var, source_var, precision_var, draft_snapshot);

    let (ids, out) = generate(
        &provider,
        Speculative::proposer(SpeculativeProposer::BlockDraft, depth),
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
    assert_eq!(named.format, format);
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
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DSpine draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dspine_accepts_more_than_one_token_per_pass() {
    accepts_more_than_one_token_per_pass(
        "MLX_LLM_BLOCK_DRAFT_TARGET",
        "MLX_LLM_BLOCK_DRAFT_SOURCE",
        "MLX_LLM_BLOCK_DRAFT_PRECISION",
        DRAFT_SNAPSHOT,
        "dspine",
        4,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dspine_accepts_more_than_one_token_per_pass"
    );
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DFlash2 draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dflash2_accepts_more_than_one_token_per_pass() {
    accepts_more_than_one_token_per_pass(
        "MLX_LLM_DFLASH2_TARGET",
        "MLX_LLM_DFLASH2_SOURCE",
        "MLX_LLM_DFLASH2_PRECISION",
        DFLASH2_SNAPSHOT,
        "dflash2",
        // The trained block: the reference and Splash draft all 8 rows (anchor + 7) per step,
        // and a non-causal block's rows read every mask row of it.
        7,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dflash2_accepts_more_than_one_token_per_pass"
    );
}

/// The image request (SW-7195, E10): a 448 × 448 picture generated here — the left half orange,
/// the right half deep blue — and a describe instruction, greedy, thinking off, as the describer
/// asks.
fn generate_image(
    provider: &mlx_llm::LlamaProvider,
    speculative: Speculative,
) -> (Vec<u32>, core_llm::TextLlmOutput) {
    let (w, h) = (448u32, 448u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(if x < w / 2 {
                &[240, 140, 30]
            } else {
                &[20, 40, 140]
            });
        }
    }
    let req = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![
                Content::Image(ImageRef::new(w, h, pixels).unwrap()),
                Content::text(
                    "Describe this image in two short paragraphs: its colours, how the picture \
                     is divided, and what it might be used for.",
                ),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 160,
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

/// Load the pinned describer target with the block draft and assert that a greedy image + text
/// `block_draft` request runs the draft — no fallback, the draft named as `format` — and accepts
/// more than one draft token per target pass; it reports where the stream first differs from the
/// same engine with speculation off.
fn image_request_runs_the_block_draft(
    target_var: &str,
    source_var: &str,
    precision_var: &str,
    draft_snapshot: &str,
    format: &str,
    depth: u32,
) {
    let (provider, draft, _) =
        load_with_block_draft(target_var, source_var, precision_var, draft_snapshot);
    let (ids, out) = generate_image(
        &provider,
        Speculative::proposer(SpeculativeProposer::BlockDraft, depth),
    );
    let decode = out.decode.unwrap();
    let (off_ids, _) = generate_image(&provider, Speculative::Off);
    let first_difference = ids.iter().zip(&off_ids).position(|(a, b)| a != b);
    let per_pass = decode.mean_accepted_length().unwrap_or(0.0);
    println!(
        "BLOCK_DRAFT_IMAGE proposer={} draft={:?} fallbacks={:?} target_passes={} accepted={} \
         accepted_per_pass={per_pass:.2} first_difference={first_difference:?} tokens={} \
         off_tokens={} text={:?}",
        decode.proposer.label(),
        decode.draft,
        decode.fallbacks,
        decode.verify_steps,
        decode.accepted_tokens,
        ids.len(),
        off_ids.len(),
        out.text,
    );
    assert_eq!(decode.proposer, ProposerKind::BlockDraft);
    assert!(
        decode.fallbacks.iter().all(|f| !f.contains("block_draft")),
        "{:?}",
        decode.fallbacks
    );
    let named = decode.draft.unwrap();
    assert_eq!(named.source, draft);
    assert_eq!(named.format, format);
    assert!(
        per_pass > 1.0,
        "accepted tokens per target pass {per_pass:.2} must exceed 1 on the image prompt"
    );
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DSpine draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dspine_runs_on_an_image_request() {
    image_request_runs_the_block_draft(
        "MLX_LLM_BLOCK_DRAFT_TARGET",
        "MLX_LLM_BLOCK_DRAFT_SOURCE",
        "MLX_LLM_BLOCK_DRAFT_PRECISION",
        DRAFT_SNAPSHOT,
        "dspine",
        4,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dspine_runs_on_an_image_request"
    );
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DFlash2 draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dflash2_runs_on_an_image_request() {
    image_request_runs_the_block_draft(
        "MLX_LLM_DFLASH2_TARGET",
        "MLX_LLM_DFLASH2_SOURCE",
        "MLX_LLM_DFLASH2_PRECISION",
        DFLASH2_SNAPSHOT,
        "dflash2",
        7,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dflash2_runs_on_an_image_request"
    );
}

/// The text request of [`generate`] under the verify layout `verify` (SW-7196).
fn generate_verified(
    provider: &mlx_llm::LlamaProvider,
    speculative: Speculative,
    verify: SpeculativeVerify,
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
        speculative_verify: verify,
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

/// Load the pinned target with the block draft and run the same greedy `block_draft` request of
/// `depth` drafts linearly and under a 16-row tree (SW-7196). Prints committed tokens per target
/// pass for both, and the tree run's rows verified and tokens accepted per pass; asserts the tree
/// run verified branching trees with no linear fallback, reports its rows and acceptance per pass,
/// and commits more than one token per target pass.
fn tree_verifies_block_draft_candidates(
    target_var: &str,
    source_var: &str,
    precision_var: &str,
    draft_snapshot: &str,
    format: &str,
    depth: u32,
) {
    let (provider, _, _) =
        load_with_block_draft(target_var, source_var, precision_var, draft_snapshot);
    let speculative = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
    let (linear_ids, linear) = generate_verified(&provider, speculative, SpeculativeVerify::Linear);
    let (tree_ids, tree) = generate_verified(
        &provider,
        speculative,
        SpeculativeVerify::Tree { max_rows: 16 },
    );
    let linear = linear.decode.unwrap();
    let tree = tree.decode.unwrap();
    let per_pass = |tokens: usize, passes: u64| tokens as f64 / passes.max(1) as f64;
    let linear_per_pass = per_pass(linear_ids.len(), linear.verify_steps);
    let tree_per_pass = per_pass(tree_ids.len(), tree.verify_steps);
    let report = tree.tree_verify.expect("the tree request ran a tree");
    let first_difference = tree_ids.iter().zip(&linear_ids).position(|(a, b)| a != b);
    println!(
        "BLOCK_DRAFT_TREE format={format} depth={depth} linear_tokens={} linear_passes={} \
         linear_tokens_per_pass={linear_per_pass:.2} tree_tokens={} tree_passes={} \
         tree_tokens_per_pass={tree_per_pass:.2} tree_branching_passes={} \
         rows_verified_per_pass={:.2} accepted_per_pass={:.2} rows_per_tree_pass={:.2} \
         accepted_per_tree_pass={:.2} max_rows_per_pass={} \
         peak_workspace_bytes={} first_difference={first_difference:?} fallbacks={:?}",
        linear_ids.len(),
        linear.verify_steps,
        tree_ids.len(),
        tree.verify_steps,
        report.tree_passes,
        report.rows_per_pass().unwrap_or(0.0),
        report.accepted_per_pass().unwrap_or(0.0),
        report.rows_per_tree_pass().unwrap_or(0.0),
        report.accepted_per_tree_pass().unwrap_or(0.0),
        report.max_rows_per_pass,
        report.peak_workspace_bytes,
        tree.fallbacks,
    );
    assert_eq!(tree.proposer, ProposerKind::BlockDraft);
    assert_eq!(tree.draft.unwrap().format, format);
    assert!(
        tree.fallbacks.iter().all(|f| !f.contains("tree")),
        "{:?}",
        tree.fallbacks
    );
    assert_eq!(report.max_rows, 16);
    assert!(
        report.tree_passes > 0,
        "the tree run verified branching trees"
    );
    assert_eq!(report.passes, tree.verify_steps);
    assert_eq!(report.accepted_tokens, tree.accepted_tokens);
    assert!(report.tree_rows_verified > report.tree_passes);
    assert!(report.tree_accepted_tokens <= report.accepted_tokens);
    assert!(report.rows_per_tree_pass().unwrap() > 1.0);
    assert!(
        tree_per_pass > 1.0,
        "committed tokens per target pass {tree_per_pass:.2} must exceed 1 under a tree"
    );
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DSpine draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dspine_tree_verifies_its_candidates() {
    tree_verifies_block_draft_candidates(
        "MLX_LLM_BLOCK_DRAFT_TARGET",
        "MLX_LLM_BLOCK_DRAFT_SOURCE",
        "MLX_LLM_BLOCK_DRAFT_PRECISION",
        DRAFT_SNAPSHOT,
        "dspine",
        4,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dspine_tree_verifies_its_candidates"
    );
}

#[test]
#[ignore = "needs the pinned Qwen3.8-27B target and DFlash2 draft under MLX_GEN_MODELS_ROOT"]
fn qwen38_27b_dflash2_tree_verifies_its_candidates() {
    tree_verifies_block_draft_candidates(
        "MLX_LLM_DFLASH2_TARGET",
        "MLX_LLM_DFLASH2_SOURCE",
        "MLX_LLM_DFLASH2_PRECISION",
        DFLASH2_SNAPSHOT,
        "dflash2",
        7,
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED block_draft::qwen38_27b_dflash2_tree_verifies_its_candidates"
    );
}
