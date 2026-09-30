//! Draft-model speculation through the provider (epic sc-24432, story sc-24436), on the shared
//! core-llm-testkit fixture: a tiny Qwen3 target loaded with a smaller tiny Qwen3 draft over the
//! same tokenizer, and with a draft whose tokenizer is not the target's. The checks are the
//! backend-neutral ones Candle runs too.

use core_llm::{LoadSpec, TextLlm};
use core_llm_testkit::{
    check_draft_model_refused, check_draft_model_resident, write_draft_model_fixture,
};
use mlx_llm::LlamaProvider;

use crate::common::Fixture;

/// AC1: `{proposer: draft_model}` emits exactly `off`'s greedy stream at every depth with a
/// report naming `draft_model`, drafts are accepted and rejected, and a seeded stochastic run
/// reproduces.
#[test]
fn a_resident_draft_proposes_and_greedy_output_is_off() {
    let root = Fixture::new("mlx-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    let provider = LlamaProvider::load(&fixture.spec_with_draft()).unwrap();
    check_draft_model_resident(&provider, &fixture.draft.to_string_lossy())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// AC2: a draft whose tokenizer is not the target's is refused at load by name; the target
/// still loads, does not advertise `draft_model`, and decodes exactly as it does alone.
#[test]
fn a_mismatched_tokenizer_draft_is_refused_and_the_target_still_loads() {
    let root = Fixture::new("mlx-llm-draft-model-", None);
    let fixture = write_draft_model_fixture(&root).unwrap();
    let provider = LlamaProvider::load(&fixture.spec_with_foreign_draft()).unwrap();
    let alone = LlamaProvider::load(&LoadSpec::dense(fixture.target.to_string_lossy())).unwrap();
    check_draft_model_refused(&provider, &alone, &fixture.foreign_draft.to_string_lossy())
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(alone.load_report().unwrap().draft.is_none());
}
