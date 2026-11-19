//! Real-weight block-draft smokes on Candle (epic SW-F360, stories SW-7197 and SW-7198; SW-7209),
//! `#[ignore]`d: the Qwen3.8-27B Candle describer tier with the DSpine or the DFlash2 draft
//! resident beside it, through the provider on the CUDA device, greedy and sampled — on a text
//! request, on an image + text request (the image generated in the test), and with tree
//! verification against the linear verifier on the same request and on the image request.
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
//! `CANDLE_LLM_BLOCK_DRAFT_PRECISION=q4` loads the draft at 4-bit (default 8-bit). Each test
//! asserts the draft is resident, advertised and named at its precision, and holds four arms to
//! the acceptance floors ([`MIN_TOKENS_PER_PASS`], [`MIN_ACCEPTED_FRACTION`], set from the healthy
//! MLX and Candle measurements): a greedy request on a cold prompt, the same request again (its
//! prompt prefix restored from the cross-turn cache), a seeded sampled request (which must
//! reproduce), and the cold greedy request on a second load with the CUDA-graph runner off. Every
//! arm runs and prints its acceptance, prefix-cache path and CUDA-graph report before any floor
//! fails. It also reports the first position where the greedy stream differs from the same
//! provider with speculation off (a
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
    BlockDraftFormat, Content, ImageRef, LoadSpec, Message, ProposerKind, Quantize, Role, Sampling,
    Speculative, SpeculativeProposer, SpeculativeVerify, StreamEvent, TextLlm, TextLlmOutput,
    TextLlmRequest, ThinkingMode,
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

/// The 27B with the `format` draft in `source_var`, its load's CUDA-graph policy `cuda_graphs`
/// (`None`: the device default, on for CUDA), the depth it advertises and the draft's precision
/// label.
fn load(source_var: &str, cuda_graphs: Option<bool>) -> (LlamaProvider, u32, &'static str) {
    let precision = match std::env::var("CANDLE_LLM_BLOCK_DRAFT_PRECISION").as_deref() {
        Ok("q4") => Some(Quantize::Q4),
        _ => None,
    };
    let mut spec = LoadSpec::dense(path(TARGET)).with_draft(path(source_var));
    spec.draft_quantize = precision;
    spec.cuda_graphs = cuda_graphs;
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
    (provider, depth, label)
}

/// Run the `format` draft's arms and hold each to the acceptance floors — every arm runs and
/// prints before any floor fails, so one run tells the hypotheses apart (SW-7197):
///
/// * **cold greedy**: a fresh load's first request, its draft context starting with the prompt's
///   prefill rows; CUDA graphs on (the default); the draft named at its precision;
/// * **warm greedy**: the same request again, its prompt prefix restored from the cross-turn
///   cache, the draft reading only the rows past it;
/// * **sampled**: seeded, run twice (it must reproduce);
/// * **graphs-off cold greedy**: a second load of the same weights with the CUDA-graph runner off
///   (`LoadSpec::cuda_graphs = Some(false)`), the cold request again — every verify step eager.
///
/// Each arm prints its acceptance, its prefix-cache path and its CUDA-graph report.
fn block_draft_on_the_27b(format: BlockDraftFormat, source_var: &str) {
    let name = format.label();
    let sampled = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        ..Sampling::greedy()
    };
    let mut failures = Vec::new();
    {
        let (provider, depth, label) = load(source_var, None);
        let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
        let (ids, out) = generate(&provider, &request(block, Sampling::greedy()));
        let decode = out.decode.as_ref().expect("a decode report");
        assert_eq!(decode.draft_tokens, Some(depth));
        let named = decode.draft.clone().expect("the draft is named");
        assert_eq!(
            (named.format.as_str(), named.precision.as_str()),
            (format.label(), label)
        );
        failures.extend(check(&format!("{name} cold greedy"), ids.len(), out));

        let (off, _) = generate(&provider, &request(Speculative::Off, Sampling::greedy()));
        let first = ids.iter().zip(&off).position(|(a, b)| a != b);
        println!("{name}: first difference from off at {first:?}");

        let (again, out) = generate(&provider, &request(block, Sampling::greedy()));
        failures.extend(check(&format!("{name} warm greedy"), again.len(), out));

        let (a, out) = generate(&provider, &request(block, sampled));
        let (b, _) = generate(&provider, &request(block, sampled));
        failures.extend(check(&format!("{name} sampled"), a.len(), out));
        if a != b {
            failures.push(format!(
                "{name} sampled: a seeded sampled run does not reproduce"
            ));
        }
    }
    {
        let (provider, depth, _) = load(source_var, Some(false));
        let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
        let (ids, out) = generate(&provider, &request(block, Sampling::greedy()));
        let graphs = out.decode.as_ref().map(|d| d.cuda_graphs.replayed);
        failures.extend(check(
            &format!("{name} graphs-off cold greedy"),
            ids.len(),
            out,
        ));
        if graphs != Some(0) {
            failures.push(format!(
                "{name} graphs-off cold greedy: {graphs:?} graph replays with the runner off"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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

/// Print one arm's acceptance, prefix-cache path and CUDA-graph report, and return why it falls
/// below the floors (`None`: it clears them).
fn check(what: &str, tokens: usize, out: TextLlmOutput) -> Option<String> {
    let Some(decode) = out.decode else {
        return Some(format!("{what}: no decode report"));
    };
    let per_pass =
        (decode.accepted_tokens + decode.verify_steps) as f64 / decode.verify_steps.max(1) as f64;
    let fraction = decode.accepted_tokens as f64 / decode.proposed_tokens.max(1) as f64;
    println!(
        "{what}: {tokens} tokens, {} target passes, {per_pass:.2} tokens per target pass, {} \
         accepted of {} proposed ({:.1}%), prefix {} ({} restored tokens), graphs {:?}",
        decode.verify_steps,
        decode.accepted_tokens,
        decode.proposed_tokens,
        fraction * 100.0,
        decode.prefix_cache.path,
        decode.prefix_hit_tokens,
        decode.cuda_graphs,
    );
    if decode.proposer != ProposerKind::BlockDraft || decode.draft.is_none() {
        return Some(format!(
            "{what}: ran {:?} without the named draft",
            decode.proposer
        ));
    }
    if per_pass < MIN_TOKENS_PER_PASS || fraction < MIN_ACCEPTED_FRACTION {
        return Some(format!(
            "{what}: {per_pass:.2} tokens per target pass and {:.1}% of drafts accepted, below \
             {MIN_TOKENS_PER_PASS} and {:.0}%",
            fraction * 100.0,
            MIN_ACCEPTED_FRACTION * 100.0
        ));
    }
    None
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

/// The image request (SW-7198, E7; MLX's SW-7195 request): a 448 × 448 picture generated here —
/// the left half orange, the right half deep blue — and a describe instruction, thinking off, as
/// the describer asks.
fn image_request(speculative: Speculative, sampling: Sampling) -> TextLlmRequest {
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
    TextLlmRequest {
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
        sampling,
        max_new_tokens: 160,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        speculative: Some(speculative),
        ..Default::default()
    }
}

/// Why a `block_draft` run's report shows it did not run as asked — another proposer, an unnamed
/// draft, or a `block_draft` fallback (the depth clamp among them). `None`: it ran as asked.
fn ran_as_asked(what: &str, out: &TextLlmOutput) -> Option<String> {
    let decode = out.decode.as_ref()?;
    if decode.proposer != ProposerKind::BlockDraft || decode.draft.is_none() {
        return Some(format!(
            "{what}: ran {:?} without the named draft",
            decode.proposer
        ));
    }
    decode
        .fallbacks
        .iter()
        .find(|f| f.contains("block_draft"))
        .map(|f| format!("{what}: fell back: {f}"))
}

/// The greedy tokens a `block_draft` stream must share with the same provider's speculation-off
/// stream before they may part. Greedy speculative decoding commits exactly the target's own
/// greedy tokens, so the two streams agree until a multi-token verify rounds a few BF16 ULPs
/// differently from the single-token step on a near-tie (the drift the module docs describe;
/// exact agreement is proven on the tiny fixtures, through the provider too). A commit of a draft
/// the target did not verify parts them at the first wrong draft — with a quarter or more of the
/// drafts accepted, within the first few passes — so an early split fails. 16 tokens is a judgment,
/// not a measurement: a near-tie flip inside it would fail a correct run, so a failure here
/// prints both streams for that call.
const MIN_GREEDY_AGREEMENT: usize = 16;

/// Why a greedy `block_draft` stream `ids` parts from the speculation-off stream `off` too early
/// ([`MIN_GREEDY_AGREEMENT`]), after printing where they first differ. `None`: they agree that far.
fn agrees_with_off(what: &str, ids: &[u32], off: &[u32]) -> Option<String> {
    let first = ids.iter().zip(off).position(|(a, b)| a != b);
    let shorter = ids.len().min(off.len());
    println!(
        "{what}: first difference from off at {first:?} ({} against {} tokens)",
        ids.len(),
        off.len()
    );
    let agreed = first.unwrap_or(shorter);
    (agreed < MIN_GREEDY_AGREEMENT.min(shorter)).then(|| {
        format!(
            "{what}: parts from off at token {agreed}, inside the first {MIN_GREEDY_AGREEMENT}: \
             {ids:?} against {off:?}"
        )
    })
}

/// Load the 27B with the `format` draft and run an image + text request with it (SW-7198 AC1):
/// greedy and seeded sampled (which must reproduce), each with no `block_draft` fallback, the
/// draft named, and the acceptance floors held; the greedy stream agrees with the same provider's
/// speculation-off stream for at least [`MIN_GREEDY_AGREEMENT`] tokens (the target's own greedy
/// tokens) and reports where it first differs. Every arm runs and prints before any check fails.
fn image_request_runs_the_block_draft(format: BlockDraftFormat, source_var: &str) {
    let name = format.label();
    let (provider, depth, label) = load(source_var, None);
    assert!(
        provider.descriptor().capabilities.supports_vision,
        "the describer tier loads its vision tower"
    );
    let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
    let sampled = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        ..Sampling::greedy()
    };
    let mut failures = Vec::new();
    let (ids, out) = generate(&provider, &image_request(block, Sampling::greedy()));
    let (off, _) = generate(
        &provider,
        &image_request(Speculative::Off, Sampling::greedy()),
    );
    println!(
        "BLOCK_DRAFT_IMAGE {name} ({label}) depth {depth}: fallbacks {:?}, text {:?}",
        out.decode.as_ref().map(|d| &d.fallbacks),
        out.text,
    );
    failures.extend(agrees_with_off(
        &format!("BLOCK_DRAFT_IMAGE {name} greedy"),
        &ids,
        &off,
    ));
    let named = out.decode.as_ref().and_then(|d| d.draft.clone());
    if named
        .as_ref()
        .map(|d| (d.format.as_str(), d.precision.as_str()))
        != Some((name, label))
    {
        failures.push(format!("{name} image: the draft is named {named:?}"));
    }
    failures.extend(ran_as_asked(&format!("{name} image greedy"), &out));
    failures.extend(check(&format!("{name} image greedy"), ids.len(), out));
    let (a, out) = generate(&provider, &image_request(block, sampled));
    let (b, _) = generate(&provider, &image_request(block, sampled));
    failures.extend(ran_as_asked(&format!("{name} image sampled"), &out));
    failures.extend(check(&format!("{name} image sampled"), a.len(), out));
    if a != b {
        failures.push(format!(
            "{name} image sampled: a seeded sampled run does not reproduce"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The 27B with the DSpine draft on an image + text request on CUDA (SW-7198 AC1).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dspine_runs_on_an_image_request() {
    image_request_runs_the_block_draft(BlockDraftFormat::DSpine, "CANDLE_LLM_DSPINE_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dspine_runs_on_an_image_request");
}

/// The 27B with the DFlash2 draft on an image + text request on CUDA (SW-7198 AC1).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dflash2_runs_on_an_image_request() {
    image_request_runs_the_block_draft(BlockDraftFormat::DFlash2, "CANDLE_LLM_DFLASH2_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dflash2_runs_on_an_image_request");
}

/// Why a tree run's report shows it did not verify the block draft's candidate trees as asked:
/// a tree or `block_draft` fallback, no branching pass, a bound other than `max_rows`, pass
/// figures that disagree with the run's, or fewer committed tokens per target pass than
/// [`MIN_TOKENS_PER_PASS`]. Prints the run's rows checked and tokens accepted per pass and per
/// tree pass first. `None`: it verified as asked.
fn tree_ran_as_asked(
    what: &str,
    tokens: usize,
    out: &TextLlmOutput,
    max_rows: u32,
) -> Option<String> {
    let Some(decode) = out.decode.as_ref() else {
        return Some(format!("{what}: no decode report"));
    };
    let Some(tree) = decode.tree_verify else {
        return Some(format!("{what}: the tree request ran no tree"));
    };
    let per_pass = tokens as f64 / decode.verify_steps.max(1) as f64;
    println!(
        "BLOCK_DRAFT_TREE {what}: {tokens} tokens, {} target passes, {per_pass:.2} tokens per \
         pass, tree_passes={} rows_verified_per_pass={:.2} accepted_per_pass={:.2} \
         rows_per_tree_pass={:.2} accepted_per_tree_pass={:.2} max_rows_per_pass={} \
         peak_workspace_bytes={} graphs {:?} fallbacks {:?}",
        decode.verify_steps,
        tree.tree_passes,
        tree.rows_per_pass().unwrap_or(0.0),
        tree.accepted_per_pass().unwrap_or(0.0),
        tree.rows_per_tree_pass().unwrap_or(0.0),
        tree.accepted_per_tree_pass().unwrap_or(0.0),
        tree.max_rows_per_pass,
        tree.peak_workspace_bytes,
        decode.cuda_graphs,
        decode.fallbacks,
    );
    if let Some(f) = decode.fallbacks.iter().find(|f| f.contains("tree")) {
        return Some(format!("{what}: verified linearly: {f}"));
    }
    if tree.max_rows != max_rows || tree.tree_passes == 0 {
        return Some(format!(
            "{what}: bound {} and {} branching passes",
            tree.max_rows, tree.tree_passes
        ));
    }
    // The report's own consistency (bookkeeping, not proof of the committed tokens — the greedy
    // agreement with `off` is that): every pass and every accepted candidate counted once, each
    // tree pass feeding its anchor and at least one candidate.
    if tree.passes != decode.verify_steps
        || tree.accepted_tokens != decode.accepted_tokens
        || tree.tree_rows_verified <= tree.tree_passes
        || tree.tree_accepted_tokens > tree.accepted_tokens
        || tree.max_rows_per_pass > 1 + u64::from(max_rows)
    {
        return Some(format!(
            "{what}: the tree report disagrees with the run: {tree:?}"
        ));
    }
    // Each tree pass ran eager through the CUDA-graph runner.
    if decode.cuda_graphs.eager < tree.tree_passes {
        return Some(format!(
            "{what}: {} eager steps for {} tree passes",
            decode.cuda_graphs.eager, tree.tree_passes
        ));
    }
    (per_pass < MIN_TOKENS_PER_PASS).then(|| {
        format!("{what}: {per_pass:.2} tokens per target pass, below {MIN_TOKENS_PER_PASS}")
    })
}

/// Load the 27B with the `format` draft and run the same text request under a 16-row tree and
/// then linearly, greedy and seeded sampled (SW-7198 AC2): each tree run verifies branching trees
/// with no tree or `block_draft` fallback, its tree report consistent with the run's counts,
/// runs each tree pass eager through the graph runner and holds [`MIN_TOKENS_PER_PASS`]; the
/// greedy tree stream agrees with speculation off for at least [`MIN_GREEDY_AGREEMENT`] tokens
/// (the target's own greedy tokens: the accepted branch only); the linear runs after it hold
/// both floors and replay their captured graphs again; a seeded sampled tree run reproduces.
/// Prints both runs' tokens per pass, the tree's rows checked and tokens accepted per pass, and
/// where the greedy tree stream first differs from off and from the linear one. Every arm runs
/// and prints before any check fails.
fn tree_verifies_block_draft_candidates(format: BlockDraftFormat, source_var: &str) {
    let name = format.label();
    let (provider, depth, _) = load(source_var, None);
    let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
    let tree = SpeculativeVerify::Tree { max_rows: 16 };
    let sampled = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        ..Sampling::greedy()
    };
    let verified = |sampling: Sampling, verify: SpeculativeVerify| TextLlmRequest {
        speculative_verify: verify,
        ..request(block, sampling)
    };
    let mut failures = Vec::new();
    let (off, _) = generate(&provider, &request(Speculative::Off, Sampling::greedy()));
    for (mode, sampling) in [("greedy", Sampling::greedy()), ("sampled", sampled)] {
        let (tree_ids, out) = generate(&provider, &verified(sampling, tree));
        let what = format!("{name} {mode} tree depth {depth}");
        failures.extend(ran_as_asked(&what, &out));
        if mode == "greedy" {
            failures.extend(agrees_with_off(&what, &tree_ids, &off));
        }
        failures.extend(tree_ran_as_asked(&what, tree_ids.len(), &out, 16));
        if mode == "sampled" {
            let (again, _) = generate(&provider, &verified(sampling, tree));
            if again != tree_ids {
                failures.push(format!(
                    "{what}: a seeded sampled tree run does not reproduce"
                ));
            }
        }
        let (linear_ids, out) = generate(&provider, &verified(sampling, SpeculativeVerify::Linear));
        let what = format!("{name} {mode} linear depth {depth}");
        let replayed = out.decode.as_ref().map_or(0, |d| d.cuda_graphs.replayed);
        failures.extend(ran_as_asked(&what, &out));
        failures.extend(check(&what, linear_ids.len(), out));
        if replayed == 0 {
            failures.push(format!(
                "{what}: no graph replay on the linear steps after the tree run"
            ));
        }
        let first = tree_ids.iter().zip(&linear_ids).position(|(a, b)| a != b);
        println!("{name} {mode}: the tree stream first differs from the linear one at {first:?}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The 27B with the DSpine draft under tree verification on CUDA (SW-7198 AC2).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dspine_tree_verifies_its_candidates() {
    tree_verifies_block_draft_candidates(BlockDraftFormat::DSpine, "CANDLE_LLM_DSPINE_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dspine_tree_verifies_its_candidates");
}

/// The 27B with the DFlash2 draft under tree verification on CUDA (SW-7198 AC2).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dflash2_tree_verifies_its_candidates() {
    tree_verifies_block_draft_candidates(BlockDraftFormat::DFlash2, "CANDLE_LLM_DFLASH2_SOURCE");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dflash2_tree_verifies_its_candidates");
}

/// Why a tree run fills less than [`MIN_ACCEPTED_FRACTION`] of its draft depth per pass (SW-7209):
/// a tree pass proposes up to 16 rows for at most `depth` accepted tokens, so the accepted-draft
/// floor is taken against the `depth` a linear pass proposes, not the tree's rows. `None`: it
/// clears the floor.
fn tree_depth_filled(what: &str, out: &TextLlmOutput, depth: u32) -> Option<String> {
    let decode = out.decode.as_ref()?;
    let filled =
        decode.accepted_tokens as f64 / (decode.verify_steps.max(1) as f64 * f64::from(depth));
    println!(
        "{what}: {:.1}% of the depth accepted per pass",
        filled * 100.0
    );
    (filled < MIN_ACCEPTED_FRACTION).then(|| {
        format!(
            "{what}: {:.1}% of the depth accepted per pass, below {:.0}%",
            filled * 100.0,
            MIN_ACCEPTED_FRACTION * 100.0
        )
    })
}

/// Load the 27B with the `format` draft and run the image + text request under a 16-row tree,
/// greedy and seeded sampled (SW-7209 AC1/AC2): each run verifies branching trees with no tree or
/// `block_draft` fallback, its tree report consistent with the run's counts and its rows checked
/// and tokens accepted per pass and per tree pass printed, runs each tree pass eager through the
/// graph runner and holds [`MIN_TOKENS_PER_PASS`] and [`MIN_ACCEPTED_FRACTION`] of its depth; the
/// greedy tree stream agrees with speculation off for at least [`MIN_GREEDY_AGREEMENT`] tokens;
/// the sampled run reproduces. Every arm runs and prints before any check fails.
fn image_tree_verifies_block_draft_candidates(format: BlockDraftFormat, source_var: &str) {
    let name = format.label();
    let (provider, depth, _) = load(source_var, None);
    assert!(
        provider.descriptor().capabilities.supports_vision,
        "the describer tier loads its vision tower"
    );
    let block = Speculative::proposer(SpeculativeProposer::BlockDraft, depth);
    let tree = SpeculativeVerify::Tree { max_rows: 16 };
    let sampled = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        ..Sampling::greedy()
    };
    let verified = |sampling: Sampling| TextLlmRequest {
        speculative_verify: tree,
        ..image_request(block, sampling)
    };
    let mut failures = Vec::new();
    let (off, _) = generate(
        &provider,
        &image_request(Speculative::Off, Sampling::greedy()),
    );
    for (mode, sampling) in [("greedy", Sampling::greedy()), ("sampled", sampled)] {
        let what = format!("{name} image {mode} tree depth {depth}");
        let (ids, out) = generate(&provider, &verified(sampling));
        println!("{what}: text {:?}", out.text);
        failures.extend(ran_as_asked(&what, &out));
        if mode == "greedy" {
            failures.extend(agrees_with_off(&what, &ids, &off));
        } else {
            let (again, _) = generate(&provider, &verified(sampling));
            if again != ids {
                failures.push(format!(
                    "{what}: a seeded sampled tree run does not reproduce"
                ));
            }
        }
        failures.extend(tree_depth_filled(&what, &out, depth));
        failures.extend(tree_ran_as_asked(&what, ids.len(), &out, 16));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The 27B with the DSpine draft under tree verification on an image + text request on CUDA
/// (SW-7209).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dspine_tree_verifies_an_image_request() {
    image_tree_verifies_block_draft_candidates(
        BlockDraftFormat::DSpine,
        "CANDLE_LLM_DSPINE_SOURCE",
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dspine_tree_verifies_an_image_request");
}

/// The 27B with the DFlash2 draft under tree verification on an image + text request on CUDA
/// (SW-7209).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_27b_dflash2_tree_verifies_an_image_request() {
    image_tree_verifies_block_draft_candidates(
        BlockDraftFormat::DFlash2,
        "CANDLE_LLM_DFLASH2_SOURCE",
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_27b_dflash2_tree_verifies_an_image_request");
}
