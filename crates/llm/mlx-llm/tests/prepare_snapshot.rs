use core_llm::{LoadSpec, Message, PrepareSpec, Quantize, Sampling, TextLlmRequest};
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::{load_for_model, prepare_snapshot};

use crate::common;
use common::{assert_fixture_is_a_guarded_entry, Fixture};

const PROMPT: &str = "The capital of France is";

/// A guarded path for the preparer's source / output dirs (sc-17768) — see [`Fixture`]. Points
/// inside a `TempDir` root because the preparer creates the output directory itself (and, on a
/// passthrough, must deliberately leave it absent); the guard takes the tree on `Drop`, including
/// out of a panicking test, which the trailing `remove_dir_all` lines never covered.
fn tmp_out(label: &str) -> Fixture {
    Fixture::new(&format!("mlx-llm-prepare-e2e-{label}-"), Some("out"))
}

fn greedy_request() -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        max_new_tokens: 256,
        seed: Some(0),
        ..Default::default()
    }
}

fn assert_loads_and_generates(out_dir: &std::path::Path) {
    let llm = load_for_model(&LoadSpec::dense(out_dir.to_string_lossy().to_string())).unwrap();
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    let out = llm.complete(&greedy_request()).unwrap();
    assert!(out.usage.generated_tokens > 0);
    let thinking = out.thinking.unwrap_or_default();
    assert!(!out.text.trim().is_empty() || !thinking.trim().is_empty());
}

#[test]
fn unknown_input_is_unsupported() {
    let src = tmp_out("unknown-src");
    let out = tmp_out("unknown-out");
    std::fs::create_dir_all(&src).unwrap();
    match prepare_snapshot(&PrepareSpec::dense(&src, &out)) {
        Err(core_llm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_q4_prepare_loads_and_generates() {
    let source = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let out = tmp_out("hf-q4");
    let report = prepare_snapshot(&PrepareSpec::quantized(&source, &out, Quantize::Q4)).unwrap();
    assert_eq!(report.quantized, Some(Quantize::Q4));
    assert!(!report.passthrough);
    assert_eq!(report.out_dir, *out);
    assert_loads_and_generates(&report.out_dir);
}

#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_dense_passthrough_returns_source_without_rewrite() {
    let source = std::path::PathBuf::from(
        std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL"),
    );
    let out = tmp_out("hf-passthrough-out");
    let report = prepare_snapshot(&PrepareSpec::dense(&source, &out)).unwrap();
    assert!(report.passthrough);
    assert_eq!(report.quantized, None);
    assert_eq!(report.out_dir, source);
    assert!(!out.exists());
    assert_loads_and_generates(&report.out_dir);
}

#[test]
#[ignore = "needs a GGUF file or dir via MLX_LLM_GGUF_SOURCE"]
fn gguf_dense_and_q4_prepare_load_and_generate() {
    let source = std::env::var("MLX_LLM_GGUF_SOURCE").expect("set MLX_LLM_GGUF_SOURCE");
    for (label, quantize) in [("dense", None), ("q4", Some(Quantize::Q4))] {
        let out = tmp_out(&format!("gguf-{label}"));
        let spec = PrepareSpec {
            source: source.clone().into(),
            out_dir: out.to_path_buf(),
            quantize,
        };
        let report = prepare_snapshot(&spec).unwrap();
        assert_eq!(report.quantized, quantize);
        assert!(!report.passthrough);
        assert_eq!(report.out_dir, *out);
        assert_loads_and_generates(&report.out_dir);
    }
}

/// Drop-regression for this suite's fixture helper: the guarded root leaves with the value. Flip
/// [`Fixture::new`]'s builder to `disable_cleanup(true)` and this goes RED.
#[test]
fn prepare_e2e_fixture_is_self_removing() {
    assert_fixture_is_a_guarded_entry(tmp_out("guard"));
}
