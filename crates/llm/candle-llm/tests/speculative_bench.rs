//! The speculative-decoding benchmark harness on real weights (epic sc-24432, story sc-24433):
//! the backend-neutral [`core_llm_testkit::run_speculative_bench`] over the
//! [`core_llm_testkit::speculative_prompt_set`] — predictable (code edit, RAG answer, summary) and
//! open-ended (chat, creative) — through a loaded `candle-llama` provider, written as one
//! baseline-format JSON document (schema [`core_llm_testkit::BENCH_SCHEMA`]). The weights-free run
//! of the same harness on a fixture model is `provider::tests::the_benchmark_harness_writes_a_baseline_document_for_the_fixture_model`;
//! this `#[ignore]`d entry point is what the epic's terminal campaign drives. The harness first
//! exists at this story's revision, so the baseline is this harness at this revision with `off`
//! (and `{"proposer": "mtp", ...}` where the checkpoint has a head) as the pre-epic-equivalent
//! options, compared against the same harness at the epic's final revision. Every input is passed
//! in; nothing is derived from a cache:
//!
//! | variable                        | meaning                                                      |
//! |---------------------------------|--------------------------------------------------------------|
//! | `SPECULATIVE_BENCH_SNAPSHOT`    | snapshot directory (config.json, tokenizer*.json, shards)    |
//! | `SPECULATIVE_BENCH_OUTPUT`      | JSON path to write (must not exist)                          |
//! | `SPECULATIVE_BENCH_OPTIONS`     | JSON array of speculative options (default `["off","auto"]`), e.g. `["off",{"proposer":"prompt_lookup","depth":4}]` |
//! | `SPECULATIVE_BENCH_SAMPLING`    | JSON sampling spec (default `"greedy"`): `"greedy"` or an object of `temperature`, `top_p`, `top_k`, `presence_penalty`, `repetition_penalty`, `repetition_context` over greedy, e.g. `{"temperature":0.7,"top_p":0.9}` (seed pinned to 0) |
//! | `SPECULATIVE_BENCH_FORMAT`      | projection format quantized at load (`LoadSpec::quantize`): `bf16` (default, dense), `q8` / `q4`, `nvfp4` (CUDA sm_120+) |
//! | `SPECULATIVE_BENCH_NEW_TOKENS`  | tokens generated per row (default 256)                       |
//! | `SPECULATIVE_BENCH_MODEL`       | model label recorded verbatim (default: the snapshot's directory name, `@<format>` appended unless bf16) |
//! | `SPECULATIVE_BENCH_BACKEND`     | backend label recorded verbatim (default `candle`)           |
//! | `SPECULATIVE_BENCH_WARMUP`      | `0` skips the untimed warm-up run per row (default on)       |
//!
//! ```text
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/baseline.json \
//!   cargo test --release --features cuda -p candle-llm --test speculative_bench -- --ignored --nocapture
//! ```

use core_llm::{LoadSpec, Quantize, Sampling, Speculative};
use core_llm_testkit::{
    parse_bench_sampling, run_speculative_bench, speculative_prompt_set, BenchConfig,
};

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

/// The sampling every row runs under: `SPECULATIVE_BENCH_SAMPLING`, or greedy.
fn sampling() -> Sampling {
    env("SPECULATIVE_BENCH_SAMPLING").map_or_else(Sampling::greedy, |spec| {
        parse_bench_sampling(&spec).unwrap_or_else(|e| panic!("SPECULATIVE_BENCH_SAMPLING: {e}"))
    })
}

/// `SPECULATIVE_BENCH_FORMAT` as the load's `quantize`: `bf16` is the dense load, `q8` / `q4`
/// the GGML Q8_0 / Q4_K load path, `nvfp4` the at-load NVFP4 quantization (refused off CUDA
/// sm_120+ with the typed capability error, never a fallback).
fn quantize(format: &str) -> Option<Quantize> {
    match format {
        "bf16" => None,
        "q8" => Some(Quantize::Q8),
        "q4" => Some(Quantize::Q4),
        "nvfp4" => Some(Quantize::Nvfp4),
        other => panic!("SPECULATIVE_BENCH_FORMAT must be bf16, q8, q4 or nvfp4, got {other}"),
    }
}

#[test]
fn format_names_bf16_q8_q4_and_nvfp4() {
    assert_eq!(quantize("bf16"), None);
    assert_eq!(quantize("q8"), Some(Quantize::Q8));
    assert_eq!(quantize("q4"), Some(Quantize::Q4));
    assert_eq!(quantize("nvfp4"), Some(Quantize::Nvfp4));
    assert!(std::panic::catch_unwind(|| quantize("fp8")).is_err());
}

#[test]
fn sampling_defaults_to_greedy_and_parses_a_stochastic_spec() {
    let parsed = parse_bench_sampling(r#"{"temperature": 0.7, "top_p": 0.9}"#).unwrap();
    assert_eq!((parsed.temperature, parsed.top_p), (0.7, 0.9));
    if env("SPECULATIVE_BENCH_SAMPLING").is_none() {
        assert_eq!(sampling(), Sampling::greedy());
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
    let format = env("SPECULATIVE_BENCH_FORMAT").unwrap_or_else(|| "bf16".into());
    let provider = candle_llm::LlamaProvider::load(&LoadSpec {
        quantize: quantize(&format),
        ..LoadSpec::dense(snapshot.clone())
    })
    .unwrap_or_else(|e| panic!("load {snapshot} ({format}): {e}"));
    let config = BenchConfig {
        model: env("SPECULATIVE_BENCH_MODEL").unwrap_or_else(|| {
            let name = std::path::Path::new(&snapshot)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or(snapshot.clone());
            if format == "bf16" {
                name
            } else {
                format!("{name}@{format}")
            }
        }),
        backend: env("SPECULATIVE_BENCH_BACKEND").unwrap_or_else(|| "candle".into()),
        max_new_tokens: env("SPECULATIVE_BENCH_NEW_TOKENS")
            .map(|v| {
                v.parse()
                    .expect("SPECULATIVE_BENCH_NEW_TOKENS is a token count")
            })
            .unwrap_or(256),
        options: options(),
        sampling: sampling(),
        warmup: env("SPECULATIVE_BENCH_WARMUP").as_deref() != Some("0"),
    };
    let doc = run_speculative_bench(&provider, &speculative_prompt_set(), &config)
        .unwrap_or_else(|e| panic!("{e}"));
    doc.write_new(&output)
        .unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    println!("wrote {} ({} rows)", output.display(), doc.rows.len());
}
