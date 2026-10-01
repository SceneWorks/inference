//! The pre-epic baseline driver for the decode-speedups benchmark (epic sc-24432 acceptance
//! test 3 / E6, story sc-24446).
//!
//! The benchmark harness (`core_llm_testkit::run_speculative_bench`) cannot compile at the
//! pre-epic revision `c1e8f8e02` — the speculative option, the prefix cache and most of the
//! `DecodeReport` it records do not exist there. This file is a self-contained driver that uses
//! only APIs that exist both there and on the epic's tree (the legacy `mtp` request field, plain
//! `generate`, the four-field `LoadSpec`), runs the harness's prompt set under the harness's
//! budgets, warm-up rule and repeat statistics, and writes **the harness's JSON schema**
//! (`core_llm_testkit::BenchRow` documents it): every field the pre-epic revision cannot report is
//! `null` — `graph_path`, `prefix_cache`, `prefix_hit_tokens`, `prefill_forwards`, `verify_steps`,
//! `mean_accepted_length` (its denominator is `verify_steps`), `fallbacks`, and the `load` block's
//! `prefix_cache_bytes`, `draft` and `fallbacks`. Where a pre-epic MLX provider returns no
//! `DecodeReport`, `proposer` is `mtp` when its legacy `MtpStats` say MTP ran (else `null`) and the
//! draft counts come from those stats.
//!
//! The prompt set and budgets are copied verbatim from `core-llm-testkit/src/speculative.rs`
//! (`speculative_prompt_set`, `BENCH_DEFAULT_NEW_TOKENS`, `BENCH_DEFAULT_REPEATS`,
//! `BENCH_SCHEMA`), and the timing and statistics code from its `observe` / `bench_sample` /
//! `BenchStats`: the testkit could not be imported at `c1e8f8e02`. On the epic's tree this file is
//! compiled as a module of `candle-llm`'s `speculative_bench` test, whose tests fail if a copy
//! drifts from the harness. Measure the epic's tree with that test (and `mlx-llm`'s), never with
//! this driver: here the cross-turn prefix cache is on by default and this driver cannot turn it
//! off.
//!
//! Options: `SPECULATIVE_BENCH_OPTIONS` takes the harness's wire form restricted to what the
//! pre-epic request could carry — `"off"`, `"auto"` and `{"proposer": "mtp", "depth": N}` (or the
//! legacy `{"mode": …}` form). Pre-epic `auto` means MTP where the model has a head, else plain;
//! the epic's `auto` falls back to prompt lookup instead, so compare `off` and explicit `mtp` rows.
//! The knobs `SPECULATIVE_BENCH_DRAFT` and `SPECULATIVE_BENCH_MTP_HEAD` and a non-zero
//! `SPECULATIVE_BENCH_PREFIX_CACHE_BYTES` are refused: the pre-epic revision has no draft models,
//! companion heads or cross-turn prefix cache. Every other knob is the harness's
//! (`SNAPSHOT`, `OUTPUT`, `SAMPLING`, `FORMAT`, `NEW_TOKENS`, `REPEATS`, `MODEL`, `BACKEND` —
//! default: the provider descriptor's backend, `candle` / `mlx` as the harness's entry points
//! label them — and `WARMUP`). With no prefix cache the warm-up runs the measured request, as the
//! harness does with its cache off.
//!
//! # Run it at the pre-epic revision
//!
//! From a checkout of this tree (`$EPIC`) on the campaign host:
//!
//! ```text
//! git -C "$EPIC" worktree add ../inference-pre-epic c1e8f8e02 && cd "$EPIC/../inference-pre-epic"
//! SRC="$EPIC/crates/contracts/core-llm/core-llm-testkit/baseline/speculative_bench_baseline.rs"
//!
//! # Candle (the CUDA campaign host): a per-file test target.
//! cp "$SRC" crates/llm/candle-llm/tests/speculative_bench_baseline.rs
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/pre-epic.json \
//!   cargo test --release --features cuda -p candle-llm --test speculative_bench_baseline -- \
//!   --ignored --nocapture
//!
//! # MLX (Apple M-series): one integration binary, so register the module, and load mlx-llm.
//! sed 's/^use candle_llm::LlamaProvider as Provider;$/use mlx_llm::LlamaProvider as Provider;/' \
//!   "$SRC" > crates/llm/mlx-llm/tests/speculative_bench_baseline.rs
//! printf '\n#[path = "speculative_bench_baseline.rs"]\nmod speculative_bench_baseline;\n' \
//!   >> crates/llm/mlx-llm/tests/main.rs
//! eval "$(scripts/fetch-prebuilt-mlx.sh)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/pre-epic.json \
//!   cargo test --release -p mlx-llm --test integration -- speculative_bench_baseline:: \
//!   --ignored --nocapture
//! ```
//!
//! Then run the epic's entry point with the same knobs (its prefix cache defaults to off) and
//! compare the two documents row for row.

// The one line the MLX copy rewrites (see above).
use candle_llm::LlamaProvider as Provider;

use core_llm::{
    DecodeReport, LoadReport, LoadSpec, Message, MtpMode, MtpStats, Quantize, Sampling,
    StreamEvent, TextLlm, TextLlmRequest,
};
use serde_json::{json, Value};
use std::time::Instant;

/// `core_llm_testkit::BENCH_SCHEMA`.
pub const BENCH_SCHEMA: &str = "sceneworks.decode-speedups.baseline/2";
/// `core_llm_testkit::BENCH_DEFAULT_NEW_TOKENS`.
pub const DEFAULT_NEW_TOKENS: u32 = 256;
/// `core_llm_testkit::BENCH_DEFAULT_REPEATS`.
pub const DEFAULT_REPEATS: u32 = 3;

const CODE_EDIT: &str = "Rename the function `total` to `sum_prices` everywhere in this Rust \
module and return the whole module, unchanged otherwise:\n\n```rust\npub struct Item {\n    pub \
name: String,\n    pub price: u32,\n}\n\npub fn total(items: &[Item]) -> u32 {\n    \
items.iter().map(|item| item.price).sum()\n}\n\npub fn average(items: &[Item]) -> Option<u32> {\n    \
if items.is_empty() {\n        return None;\n    }\n    Some(total(items) / items.len() as u32)\n}\n\n\
#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn total_adds_prices() {\n        \
let items = vec![Item { name: \"a\".into(), price: 2 }, Item { name: \"b\".into(), price: 3 }];\n        \
assert_eq!(total(&items), 5);\n    }\n}\n```";

const RAG_ANSWER: &str = "Answer the question using only the context, quoting it where you can.\n\n\
Context:\nThe Lindqvist Bridge opened in 1932 and carries two road lanes and one rail track across \
the Vessel River. It was designed by the engineer Maren Lindqvist, who also designed the Harbor \
Street viaduct. The bridge closed for eighteen months in 1987 for a deck replacement, and a \
pedestrian walkway was added on the east side in 2004. The rail track is used by regional freight \
trains; passenger services were withdrawn in 1961.\n\nQuestion: When did the Lindqvist Bridge open, \
who designed it, and what was added in 2004?";

const SUMMARY: &str = "Summarize the following meeting notes in five bullet points, keeping the \
names and dates exactly as written.\n\nNotes: The release review met on March 3. Priya Raman \
reported that the installer build is green on all three platforms. Tomas Okafor said the crash on \
resume is fixed in build 412 and asked for one more day of soak testing. The team agreed to move \
the release date from March 10 to March 12. Priya Raman will update the release notes by March 8. \
Tomas Okafor will send the soak-test results by March 9. The next review is on March 11.";

const CHAT: &str = "I have a free Saturday afternoon in a new city. What are a few good ways to \
spend it if I like walking and trying local food?";

const CREATIVE: &str = "Write a short story, about three paragraphs, about a lighthouse keeper \
who finds a message in a bottle that seems to be addressed to them.";

/// `core_llm_testkit::speculative_prompt_set()` as `(id, class label, text)`.
pub const PROMPTS: [(&str, &str, &str); 5] = [
    ("code_edit", "predictable", CODE_EDIT),
    ("rag_answer", "predictable", RAG_ANSWER),
    ("summary", "predictable", SUMMARY),
    ("chat", "open_ended", CHAT),
    ("creative", "open_ended", CREATIVE),
];

/// What a run measures and labels (`core_llm_testkit::BenchConfig`, options as legacy modes).
pub struct Config {
    pub model: String,
    pub backend: String,
    pub max_new_tokens: u32,
    pub sampling: Sampling,
    pub warmup: bool,
    pub repeats: u32,
    pub options: Vec<MtpMode>,
}

/// One measured generation.
pub struct Measured {
    pub ttft_ms: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tok_s: Option<f64>,
    pub timing_source: &'static str,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub report: Option<DecodeReport>,
    pub mtp: Option<MtpStats>,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The harness's wire form of a legacy mode (`Speculative::from(mode)` serialized).
pub fn mode_json(mode: &MtpMode) -> Value {
    match mode {
        MtpMode::Off => json!("off"),
        MtpMode::Auto => json!("auto"),
        MtpMode::Enabled { draft_tokens } => json!({"proposer": "mtp", "depth": draft_tokens}),
    }
}

/// One option in the harness's wire form, as far as the pre-epic request can carry it.
fn parse_mode(value: &Value) -> Result<MtpMode, String> {
    let depth = |v: &Value| {
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("option {value}: the depth must be an unsigned integer"))
    };
    match value {
        Value::String(s) if s == "off" => Ok(MtpMode::Off),
        Value::String(s) if s == "auto" => Ok(MtpMode::Auto),
        Value::Object(map) if map.get("proposer") == Some(&json!("mtp")) => Ok(MtpMode::Enabled {
            draft_tokens: depth(&map["depth"])?,
        }),
        Value::Object(map) if map.get("mode") == Some(&json!("enabled")) => Ok(MtpMode::Enabled {
            draft_tokens: depth(&map["draft_tokens"])?,
        }),
        Value::Object(map) if map.get("mode").is_some() => parse_mode(&map["mode"]),
        other => Err(format!(
            "option {other} is not available at the pre-epic revision (only \"off\", \"auto\" and \
             {{\"proposer\": \"mtp\", \"depth\": N}})"
        )),
    }
}

/// `core_llm_testkit::sampling_json`.
fn sampling_json(sampling: &Sampling) -> Value {
    let f32_json = |x: f32| {
        x.to_string()
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map_or(Value::Null, Value::Number)
    };
    json!({
        "temperature": f32_json(sampling.temperature),
        "top_p": f32_json(sampling.top_p),
        "top_k": sampling.top_k,
        "presence_penalty": f32_json(sampling.presence_penalty),
        "repetition_penalty": f32_json(sampling.repetition_penalty),
        "repetition_context": sampling.repetition_context,
    })
}

/// `core_llm_testkit::parse_bench_sampling`.
fn parse_sampling(spec: &str) -> Result<Sampling, String> {
    let value: Value =
        serde_json::from_str(spec).map_err(|e| format!("sampling `{spec}` is not JSON: {e}"))?;
    let mut sampling = Sampling::greedy();
    if value == json!("greedy") {
        return Ok(sampling);
    }
    let object = value
        .as_object()
        .ok_or_else(|| format!("sampling `{spec}` is neither \"greedy\" nor an object"))?;
    for (key, v) in object {
        let float = || {
            v.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("sampling `{key}` must be a number, got {v}"))
        };
        let count = || {
            v.as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| format!("sampling `{key}` must be a non-negative integer, got {v}"))
        };
        match key.as_str() {
            "temperature" => sampling.temperature = float()?,
            "top_p" => sampling.top_p = float()?,
            "top_k" => sampling.top_k = count()?,
            "presence_penalty" => sampling.presence_penalty = float()?,
            "repetition_penalty" => sampling.repetition_penalty = float()?,
            "repetition_context" => sampling.repetition_context = count()?,
            other => return Err(format!("sampling has no knob `{other}`")),
        }
    }
    Ok(sampling)
}

/// `core_llm_testkit::BenchStats::json_of`.
fn stats_json(values: &[f64]) -> Value {
    let n = values.len();
    if n == 0 {
        return Value::Null;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let stddev = (n > 1)
        .then(|| (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt());
    json!({
        "n": n,
        "mean": mean,
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "stddev": stddev,
    })
}

/// The request the harness's `bench_request` builds, through the legacy `mtp` field.
// `.into()`: the field is `MtpMode` at the pre-epic revision and may become `Option<MtpMode>`.
#[allow(clippy::useless_conversion)]
fn request(text: &str, mode: MtpMode, config: &Config) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(text)],
        sampling: config.sampling,
        max_new_tokens: config.max_new_tokens,
        seed: Some(0),
        mtp: mode.into(),
        ..Default::default()
    }
}

/// One generation, timed as the harness's `observe` + `bench_sample` time it.
fn measure(provider: &dyn TextLlm, req: &TextLlmRequest) -> Result<Measured, String> {
    provider
        .validate(req)
        .map_err(|e| format!("validate refused the request: {e}"))?;
    let mut ttft = None;
    let started = Instant::now();
    let out = provider
        .generate(req, &mut |event| {
            if let StreamEvent::Token { .. } = event {
                ttft.get_or_insert_with(|| started.elapsed());
            }
        })
        .map_err(|e| format!("generate failed: {e}"))?;
    let wall = started.elapsed();
    let (prefill, decode, timing_source) = match out.timings {
        Some(t) => (t.prefill, t.decode, "backend"),
        None => {
            let ttft = ttft.unwrap_or(wall);
            (ttft, wall.saturating_sub(ttft), "wall")
        }
    };
    let decode_s = decode.as_secs_f64();
    Ok(Measured {
        ttft_ms: ttft.map(|d| d.as_secs_f64() * 1e3),
        prefill_ms: prefill.as_secs_f64() * 1e3,
        decode_ms: decode_s * 1e3,
        decode_tok_s: (decode_s > 0.0).then(|| f64::from(out.usage.generated_tokens) / decode_s),
        timing_source,
        prompt_tokens: out.usage.prompt_tokens,
        generated_tokens: out.usage.generated_tokens,
        report: out.decode,
        mtp: out.mtp,
    })
}

/// One row in the harness's schema (`core_llm_testkit::BenchRow::to_json`) over `runs`, the
/// first measured repeat's telemetry first.
pub fn row_json(prompt_id: &str, class: &str, mode: &MtpMode, runs: &[Measured]) -> Value {
    let first = &runs[0];
    let report = first.report.as_ref();
    let mtp = first.mtp.as_ref();
    let series = |f: fn(&Measured) -> Option<f64>| {
        stats_json(&runs.iter().filter_map(f).collect::<Vec<_>>())
    };
    let count = |from_report: fn(&DecodeReport) -> u64, from_mtp: fn(&MtpStats) -> u32| {
        report
            .map(from_report)
            .or_else(|| mtp.map(|m| u64::from(from_mtp(m))))
    };
    let proposer = match (report, mtp) {
        (Some(r), _) => Some(r.proposer.label()),
        (None, Some(_)) => Some("mtp"),
        (None, None) => None,
    };
    json!({
        "prompt_id": prompt_id,
        "class": class,
        "requested": mode_json(mode),
        "proposer": proposer,
        "path": report.map(|r| r.path.clone()),
        "draft_tokens": report.and_then(|r| r.draft_tokens),
        "prompt_tokens": first.prompt_tokens,
        "generated_tokens": first.generated_tokens,
        "repeats": runs.len(),
        "ttft_ms": series(|m| m.ttft_ms),
        "decode_tok_s": series(|m| m.decode_tok_s),
        "prefill_ms": series(|m| Some(m.prefill_ms)),
        "decode_ms": series(|m| Some(m.decode_ms)),
        "timing_source": first.timing_source,
        "target_forwards": count(|r| r.target_forwards, |m| m.target_forwards),
        "prefill_forwards": null,
        "verify_steps": null,
        "replay_forwards": report.map(|r| r.replay_forwards),
        "proposed_tokens": count(|r| r.proposed_tokens, |m| m.proposed_tokens),
        "accepted_tokens": count(|r| r.accepted_tokens, |m| m.accepted_tokens),
        "mean_accepted_length": null,
        "prefix_cache": null,
        "prefix_hit_tokens": null,
        "sampler": report.map(|r| r.sampler.clone()),
        "kv_cache": report.map(|r| r.kv_cache.clone()),
        "attention": report.map(|r| r.attention.clone()),
        "cuda_graphs": report.map(|r| r.cuda_graphs.path.clone()),
        "graph_path": null,
        "fused": report.map(|r| {
            json!({"path": r.fused_primitives.path, "reason": r.fused_primitives.reason})
        }),
        "fallbacks": null,
        "samples": runs
            .iter()
            .map(|m| {
                json!({
                    "ttft_ms": m.ttft_ms,
                    "decode_tok_s": m.decode_tok_s,
                    "prefill_ms": m.prefill_ms,
                    "decode_ms": m.decode_ms,
                    "generated_tokens": m.generated_tokens,
                    "prefix_hit_tokens": null,
                })
            })
            .collect::<Vec<_>>(),
    })
}

/// The document in the harness's schema (`core_llm_testkit::BenchDocument::to_json`).
pub fn document_json(config: &Config, load: Option<&LoadReport>, rows: Vec<Value>) -> Value {
    json!({
        "schema": BENCH_SCHEMA,
        "model": config.model,
        "backend": config.backend,
        "max_new_tokens": config.max_new_tokens,
        "sampling": sampling_json(&config.sampling),
        "warmup": config.warmup,
        "repeats": config.repeats,
        "options": config.options.iter().map(mode_json).collect::<Vec<_>>(),
        "load": load.map(|r| json!({
            "prefix_cache_bytes": null,
            "draft": null,
            "cuda_graphs": r.cuda_graphs,
            "fallbacks": null,
        })),
        "rows": rows,
    })
}

/// Every prompt under every option: an untimed warm-up of the measured request (when
/// `config.warmup`), then `config.repeats` measured runs.
pub fn run(provider: &dyn TextLlm, config: &Config) -> Result<Vec<Value>, String> {
    if config.repeats == 0 {
        return Err("a benchmark row needs at least one measured repeat".into());
    }
    let mut rows = Vec::new();
    for (id, class, text) in PROMPTS {
        for mode in &config.options {
            let tag = format!("[{id}] {}", mode_json(mode));
            let req = request(text, *mode, config);
            if config.warmup {
                measure(provider, &req).map_err(|e| format!("{tag} warm-up: {e}"))?;
            }
            let runs = (0..config.repeats)
                .map(|r| measure(provider, &req).map_err(|e| format!("{tag} repeat {r}: {e}")))
                .collect::<Result<Vec<_>, _>>()?;
            rows.push(row_json(id, class, mode, &runs));
        }
    }
    Ok(rows)
}

#[test]
#[ignore = "needs a snapshot via SPECULATIVE_BENCH_SNAPSHOT and an output path via SPECULATIVE_BENCH_OUTPUT"]
fn speculative_bench_baseline_writes_the_document() {
    use std::io::Write;
    let snapshot = env("SPECULATIVE_BENCH_SNAPSHOT").expect("set SPECULATIVE_BENCH_SNAPSHOT");
    let output = std::path::PathBuf::from(
        env("SPECULATIVE_BENCH_OUTPUT").expect("set SPECULATIVE_BENCH_OUTPUT"),
    );
    assert!(
        !output.exists(),
        "{} exists; a baseline is never overwritten",
        output.display()
    );
    for knob in ["SPECULATIVE_BENCH_DRAFT", "SPECULATIVE_BENCH_MTP_HEAD"] {
        assert!(env(knob).is_none(), "{knob} is not available pre-epic");
    }
    assert!(
        env("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES").is_none_or(|v| v == "0"),
        "the pre-epic revision has no cross-turn prefix cache"
    );
    let format = env("SPECULATIVE_BENCH_FORMAT").unwrap_or_else(|| "bf16".into());
    let quantize = match format.as_str() {
        "bf16" => None,
        "q8" => Some(Quantize::Q8),
        "q4" => Some(Quantize::Q4),
        "nvfp4" => Some(Quantize::Nvfp4),
        other => panic!("SPECULATIVE_BENCH_FORMAT must be bf16, q8, q4 or nvfp4, got {other}"),
    };
    let count = |name: &str, default: u32| {
        env(name).map_or(default, |v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be a non-negative integer, got {v}"))
        })
    };
    let options = match env("SPECULATIVE_BENCH_OPTIONS") {
        Some(list) => serde_json::from_str::<Vec<Value>>(&list)
            .unwrap_or_else(|e| panic!("SPECULATIVE_BENCH_OPTIONS is not a JSON list: {e}"))
            .iter()
            .map(|v| parse_mode(v).unwrap_or_else(|e| panic!("{e}")))
            .collect(),
        None => vec![MtpMode::Off, MtpMode::Auto],
    };
    let provider = Provider::load(&LoadSpec {
        quantize,
        ..LoadSpec::dense(snapshot.clone())
    })
    .unwrap_or_else(|e| panic!("load {snapshot} ({format}): {e}"));
    let config = Config {
        model: env("SPECULATIVE_BENCH_MODEL").unwrap_or_else(|| {
            let name = std::path::Path::new(&snapshot)
                .file_name()
                .map_or_else(|| snapshot.clone(), |n| n.to_string_lossy().into_owned());
            if format == "bf16" {
                name
            } else {
                format!("{name}@{format}")
            }
        }),
        backend: env("SPECULATIVE_BENCH_BACKEND")
            .unwrap_or_else(|| provider.descriptor().backend.clone()),
        max_new_tokens: count("SPECULATIVE_BENCH_NEW_TOKENS", DEFAULT_NEW_TOKENS),
        sampling: env("SPECULATIVE_BENCH_SAMPLING").map_or_else(Sampling::greedy, |spec| {
            parse_sampling(&spec).unwrap_or_else(|e| panic!("SPECULATIVE_BENCH_SAMPLING: {e}"))
        }),
        warmup: env("SPECULATIVE_BENCH_WARMUP").as_deref() != Some("0"),
        repeats: count("SPECULATIVE_BENCH_REPEATS", DEFAULT_REPEATS),
        options,
    };
    let rows = run(&provider, &config).unwrap_or_else(|e| panic!("{e}"));
    let n = rows.len();
    let doc = document_json(&config, provider.load_report().as_ref(), rows);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    let text = serde_json::to_string_pretty(&doc).expect("a JSON value serializes");
    file.write_all(text.as_bytes())
        .and_then(|()| file.write_all(b"\n"))
        .unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    println!("wrote {} ({n} rows)", output.display());
}
