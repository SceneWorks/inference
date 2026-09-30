//! The speculative-decoding benchmark harness on real weights (epic sc-24432, story sc-24433):
//! the backend-neutral [`core_llm_testkit::run_speculative_bench`] over the
//! [`core_llm_testkit::speculative_prompt_set`] — predictable (code edit, RAG answer, summary) and
//! open-ended (chat, creative) — through a loaded `candle-llama` provider, written as one
//! baseline-format JSON document (schema [`core_llm_testkit::BENCH_SCHEMA`]). The weights-free run
//! of the same harness on a fixture model is `provider::tests::the_benchmark_harness_writes_a_baseline_document_for_the_fixture_model`;
//! this `#[ignore]`d entry point is what the epic's terminal campaign drives, once, against the
//! pre-epic and post-epic revisions. Every input is passed in; nothing is derived from a cache:
//!
//! | variable                        | meaning                                                      |
//! |---------------------------------|--------------------------------------------------------------|
//! | `SPECULATIVE_BENCH_SNAPSHOT`    | snapshot directory (config.json, tokenizer*.json, shards)    |
//! | `SPECULATIVE_BENCH_OUTPUT`      | JSON path to write (must not exist)                          |
//! | `SPECULATIVE_BENCH_OPTIONS`     | JSON array of speculative options (default `["off","auto"]`), e.g. `["off",{"proposer":"prompt_lookup","depth":4}]` |
//! | `SPECULATIVE_BENCH_NEW_TOKENS`  | tokens generated per row (default 256)                       |
//! | `SPECULATIVE_BENCH_MODEL`       | model label recorded verbatim (default: the snapshot's directory name) |
//! | `SPECULATIVE_BENCH_BACKEND`     | backend label recorded verbatim (default `candle`)           |
//! | `SPECULATIVE_BENCH_WARMUP`      | `0` skips the untimed warm-up run per row (default on)       |
//!
//! ```text
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/baseline.json \
//!   cargo test --release --features cuda -p candle-llm --test speculative_bench -- --ignored --nocapture
//! ```

use core_llm::{LoadSpec, Speculative};
use core_llm_testkit::{run_speculative_bench, speculative_prompt_set, BenchConfig};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The options a run measures: `SPECULATIVE_BENCH_OPTIONS` as JSON (legacy `mtp` shapes
/// included), or `off` and `auto`.
fn options() -> Vec<Speculative> {
    match env("SPECULATIVE_BENCH_OPTIONS") {
        Some(json) => serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("SPECULATIVE_BENCH_OPTIONS is not a JSON option list: {e}")),
        None => vec![Speculative::Off, Speculative::Auto],
    }
}

#[test]
fn options_default_to_off_and_auto_and_parse_the_wire_form() {
    // Read through the same parser the harness uses, without touching the process environment.
    let parsed: Vec<Speculative> =
        serde_json::from_str(r#"["off", {"proposer": "prompt_lookup", "depth": 4}]"#).unwrap();
    assert_eq!(
        parsed,
        [
            Speculative::Off,
            Speculative::proposer(core_llm::SpeculativeProposer::PromptLookup, 4)
        ]
    );
    if env("SPECULATIVE_BENCH_OPTIONS").is_none() {
        assert_eq!(options(), [Speculative::Off, Speculative::Auto]);
    }
}

#[test]
#[ignore = "needs a snapshot via SPECULATIVE_BENCH_SNAPSHOT and an output path via SPECULATIVE_BENCH_OUTPUT"]
fn speculative_bench_writes_the_baseline_document() {
    let snapshot = env("SPECULATIVE_BENCH_SNAPSHOT").expect("set SPECULATIVE_BENCH_SNAPSHOT");
    let output = std::path::PathBuf::from(
        env("SPECULATIVE_BENCH_OUTPUT").expect("set SPECULATIVE_BENCH_OUTPUT"),
    );
    assert!(
        !output.exists(),
        "{} exists; a baseline is never overwritten",
        output.display()
    );
    let provider = candle_llm::LlamaProvider::load(&LoadSpec::dense(snapshot.clone()))
        .unwrap_or_else(|e| panic!("load {snapshot}: {e}"));
    let config = BenchConfig {
        model: env("SPECULATIVE_BENCH_MODEL").unwrap_or_else(|| {
            std::path::Path::new(&snapshot)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or(snapshot.clone())
        }),
        backend: env("SPECULATIVE_BENCH_BACKEND").unwrap_or_else(|| "candle".into()),
        max_new_tokens: env("SPECULATIVE_BENCH_NEW_TOKENS")
            .map(|v| {
                v.parse()
                    .expect("SPECULATIVE_BENCH_NEW_TOKENS is a token count")
            })
            .unwrap_or(256),
        options: options(),
        warmup: env("SPECULATIVE_BENCH_WARMUP").as_deref() != Some("0"),
    };
    let doc = run_speculative_bench(&provider, &speculative_prompt_set(), &config)
        .unwrap_or_else(|e| panic!("{e}"));
    doc.write_new(&output)
        .unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    println!("wrote {} ({} rows)", output.display(), doc.rows.len());
}
