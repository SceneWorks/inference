//! Draft-model speculation through the provider (epic sc-24432, story sc-24436), on the shared
//! core-llm-testkit fixture: a tiny Qwen3 target loaded with a smaller tiny Qwen3 draft over the
//! same tokenizer, and with a draft whose tokenizer is not the target's. The checks are the
//! backend-neutral ones MLX runs too.

use candle_llm::LlamaProvider;
use core_llm::{LoadSpec, TextLlm};
use core_llm_testkit::{
    check_draft_model_refused, check_draft_model_resident, check_draft_model_short_context,
    check_draft_model_stop_token, write_draft_model_fixture,
};

mod common;
use common::Fixture;

/// AC1: `{proposer: draft_model}` emits exactly `off`'s greedy stream at every depth with a
/// report naming `draft_model`, drafts are accepted and rejected, and a seeded stochastic run
/// reproduces.
#[test]
fn a_resident_draft_proposes_and_greedy_output_is_off() {
    let root = Fixture::new("candle-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    let provider = LlamaProvider::load(&fixture.spec_with_draft()).unwrap();
    check_draft_model_resident(&provider, &fixture.draft.to_string_lossy())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// AC2: a draft whose tokenizer is not the target's is refused at load by name; the target
/// still loads, does not advertise `draft_model`, and decodes exactly as it does alone.
#[test]
fn a_mismatched_tokenizer_draft_is_refused_and_the_target_still_loads() {
    let root = Fixture::new("candle-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    let provider = LlamaProvider::load(&fixture.spec_with_foreign_draft()).unwrap();
    let alone = LlamaProvider::load(&LoadSpec::dense(fixture.target.to_string_lossy())).unwrap();
    check_draft_model_refused(&provider, &alone, &fixture.foreign_draft.to_string_lossy())
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(alone.load_report().unwrap().draft.is_none());
}

fn load(spec: &LoadSpec) -> Result<Box<dyn TextLlm>, String> {
    LlamaProvider::load(spec)
        .map(|p| Box::new(p) as Box<dyn TextLlm>)
        .map_err(|e| e.to_string())
}

/// A real stop token the drafts propose and the target accepts ends every `draft_model` run —
/// the host-sampled paths (a penalty, a near-zero temperature) included — where `off` stops.
#[test]
fn a_draft_proposed_stop_token_ends_the_run_where_off_does() {
    let root = Fixture::new("candle-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    check_draft_model_stop_token(&fixture, &load).unwrap_or_else(|e| panic!("{e}"));
}

/// E2: a request past the draft's own context window runs `auto` by name instead of driving the
/// draft past it; within the window the draft runs.
#[test]
fn a_request_past_the_draft_context_falls_back_by_name() {
    let root = Fixture::new("candle-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    check_draft_model_short_context(&fixture, &load).unwrap_or_else(|e| panic!("{e}"));
}
