//! The speculative-decoding benchmark (epic sc-24432 acceptance test 3 / E6) on Candle: the
//! `#[ignore]`d real-weight entry point the terminal campaign drives, the weights-free run of the
//! same harness on the shared fixture, and the pre-epic baseline driver compiled against this tree.
//!
//! The entry point is [`core_llm_testkit::run_speculative_bench_from_env`] over a `candle-llama`
//! provider; its doc lists every `SPECULATIVE_BENCH_*` knob, and [`core_llm_testkit::BenchRow`]
//! documents the JSON schema the MLX entry point (`mlx-llm`'s `speculative_bench` module) writes
//! too. The pre-epic revision predates the harness, so its rows come from
//! `core-llm-testkit/baseline/speculative_bench_baseline.rs` (its header has the copy-and-run
//! steps); it is compiled here as `baseline`, and the tests below hold it to the harness's prompt
//! set, budgets and schema.
//!
//! ```text
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/bench.json \
//!   cargo test --release --features cuda -p candle-llm --test speculative_bench -- \
//!   speculative_bench_writes_the_baseline_document --ignored --nocapture
//! ```

use core_llm::{LoadSpec, TextLlm};

#[path = "../../../contracts/core-llm/core-llm-testkit/baseline/speculative_bench_baseline.rs"]
mod baseline;

fn load(spec: &LoadSpec) -> Result<Box<dyn TextLlm>, String> {
    candle_llm::LlamaProvider::load(spec)
        .map(|p| Box::new(p) as Box<dyn TextLlm>)
        .map_err(|e| e.to_string())
}

#[test]
#[ignore = "needs a snapshot via SPECULATIVE_BENCH_SNAPSHOT and an output path via SPECULATIVE_BENCH_OUTPUT"]
fn speculative_bench_writes_the_baseline_document() {
    let (output, doc) = core_llm_testkit::run_speculative_bench_from_env("candle", &load)
        .unwrap_or_else(|e| panic!("{e}"));
    println!("wrote {} ({} rows)", output.display(), doc.rows.len());
}

/// The harness end to end on the shared fixture: every schema field over two repeats with the
/// prefix cache off by default, and a warm-up that never lends the measured prompt to its row.
#[test]
fn the_bench_runs_on_the_fixture_with_isolated_warmups() {
    let root = tempfile::tempdir().unwrap();
    core_llm_testkit::check_speculative_bench_on_fixture(root.path(), "candle-cpu", &load)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// The baseline driver copies the harness's prompt set and budgets verbatim.
#[test]
fn the_baseline_driver_runs_the_harness_prompt_set_and_budgets() {
    let harness: Vec<_> = core_llm_testkit::speculative_prompt_set()
        .into_iter()
        .map(|p| {
            (
                p.id,
                p.class.label().to_string(),
                p.messages[0].text_content(),
            )
        })
        .collect();
    let copied: Vec<_> = baseline::PROMPTS
        .iter()
        .map(|(id, class, text)| (id.to_string(), class.to_string(), text.to_string()))
        .collect();
    assert_eq!(copied, harness);
    assert_eq!(baseline::BENCH_SCHEMA, core_llm_testkit::BENCH_SCHEMA);
    assert_eq!(
        baseline::DEFAULT_NEW_TOKENS,
        core_llm_testkit::BENCH_DEFAULT_NEW_TOKENS
    );
    assert_eq!(
        baseline::DEFAULT_REPEATS,
        core_llm_testkit::BENCH_DEFAULT_REPEATS
    );
}

/// The keys of a JSON object, and of every nested object, as dotted paths, sorted.
fn keys(value: &serde_json::Value) -> Vec<String> {
    fn walk(value: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
        for (k, v) in value.as_object().into_iter().flatten() {
            let path = format!("{prefix}{k}");
            walk(v, &format!("{path}."), out);
            out.push(path);
        }
    }
    let mut out = Vec::new();
    walk(value, "", &mut out);
    out.sort();
    out
}

/// The baseline driver writes the harness's schema: the same document, row, statistics and sample
/// keys, the same statistics over the same repeats, and the same option wire form.
#[test]
fn the_baseline_driver_writes_the_harness_schema() {
    use core_llm::{DecodeReport, LoadReport, MtpMode, Sampling, Speculative};
    use core_llm_testkit::{BenchConfig, BenchDocument, BenchRow, BenchSample, PromptClass};

    let report = DecodeReport::default();
    let repeats = [(10.0, 50.0), (12.0, 54.0), (14.0, 52.0)];
    let harness_row = BenchRow {
        prompt_id: "chat".into(),
        class: PromptClass::OpenEnded,
        requested: Speculative::Off,
        prompt_tokens: 9,
        generated_tokens: 4,
        timing_source: "backend",
        report: Some(report.clone()),
        samples: repeats
            .iter()
            .map(|&(ttft, tok_s)| BenchSample {
                ttft_ms: Some(ttft),
                prefill_ms: ttft,
                decode_ms: 4e3 / tok_s,
                decode_tok_s: Some(tok_s),
                generated_tokens: 4,
                prefix_hit_tokens: Some(0),
            })
            .collect(),
    }
    .to_json();
    let measured: Vec<_> = repeats
        .iter()
        .map(|&(ttft, tok_s)| baseline::Measured {
            ttft_ms: Some(ttft),
            prefill_ms: ttft,
            decode_ms: 4e3 / tok_s,
            decode_tok_s: Some(tok_s),
            timing_source: "backend",
            prompt_tokens: 9,
            generated_tokens: 4,
            report: Some(report.clone()),
            mtp: None,
        })
        .collect();
    let baseline_row = baseline::row_json("chat", "open_ended", &MtpMode::Off, &measured);
    // The harness's populated `prefix_cache` object; pre-epic reports have none (`null`).
    let mut want = keys(&harness_row);
    want.retain(|k| !k.starts_with("prefix_cache."));
    assert_eq!(keys(&baseline_row), want);
    for key in [
        "prompt_id",
        "class",
        "requested",
        "ttft_ms",
        "decode_tok_s",
        "prefill_ms",
        "decode_ms",
        "repeats",
        "fused",
    ] {
        assert_eq!(baseline_row[key], harness_row[key], "{key}");
    }
    for key in [
        "prefix_cache",
        "prefix_hit_tokens",
        "graph_path",
        "verify_steps",
    ] {
        assert!(baseline_row[key].is_null(), "pre-epic `{key}` is null");
    }

    let harness_doc = BenchDocument {
        config: BenchConfig {
            model: "m".into(),
            backend: "candle".into(),
            max_new_tokens: 4,
            sampling: Sampling::greedy(),
            options: vec![Speculative::Off, Speculative::Auto],
            warmup: true,
            repeats: 3,
        },
        load: Some(LoadReport::default()),
        rows: Vec::new(),
    }
    .to_json();
    let baseline_config = baseline::Config {
        model: "m".into(),
        backend: "candle".into(),
        max_new_tokens: 4,
        sampling: Sampling::greedy(),
        warmup: true,
        repeats: 3,
        options: vec![MtpMode::Off, MtpMode::Auto],
    };
    let baseline_doc =
        baseline::document_json(&baseline_config, Some(&LoadReport::default()), Vec::new());
    assert_eq!(keys(&baseline_doc), keys(&harness_doc));
    for key in [
        "schema",
        "model",
        "backend",
        "max_new_tokens",
        "sampling",
        "options",
        "repeats",
        "warmup",
    ] {
        assert_eq!(baseline_doc[key], harness_doc[key], "{key}");
    }
    // The legacy `enabled` mode's wire form is the harness's `{"proposer": "mtp", "depth": N}`.
    let enabled = MtpMode::Enabled { draft_tokens: 3 };
    assert_eq!(
        baseline::mode_json(&enabled),
        serde_json::to_value(Speculative::from(enabled)).unwrap()
    );
}
