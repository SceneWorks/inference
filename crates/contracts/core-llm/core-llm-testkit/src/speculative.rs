//! The speculative-decoding parity suite and decode benchmark harness (epic sc-24432, story
//! sc-24433) — backend-neutral, driven purely through [`TextLlm`], so the MLX and Candle
//! providers run the same checks and write the same evidence rows.
//!
//! * **Parity** ([`check_speculative_greedy_parity`]): every speculative option in a case table,
//!   on every prompt, emits the same greedy stream as the same provider with speculation `off`
//!   (epic E1), and its [`DecodeReport`] names the proposer that actually ran (E3). A backend adds
//!   rows (a proposer, a depth, a decoder type) by extending its case table, not this code.
//! * **Benchmark** ([`run_speculative_bench`]): the [`speculative_prompt_set`] — predictable
//!   (code edit, RAG answer, summary) and open-ended (chat, creative) — under each option, one
//!   [`BenchRow`] per (prompt, option) with decode tok/s and TTFT statistics over measured
//!   repeats, the realized mean accepted length and the report's telemetry, written as one
//!   baseline-format JSON document ([`BenchDocument::write_new`], schema [`BENCH_SCHEMA`]). Both
//!   backends' entry points share [`run_speculative_bench_from_env`]. The pre-epic revision
//!   predates this harness, so its rows come from the standalone driver in this crate's
//!   `baseline/` directory, which emits the same schema; the pre- and post-epic campaign rows
//!   compare field for field.

use core_llm::{
    DecodeReport, FinishReason, LoadReport, LoadSpec, Message, ProposerKind, Quantize, Sampling,
    Speculative, SpeculativeProposer, StreamEvent, TextLlm, TextLlmRequest,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::draft_model::{draft_model_prompts, write_draft_model_fixture, DraftLoader};

/// The benchmark document's schema identifier; bump it when a field changes meaning.
pub const BENCH_SCHEMA: &str = "sceneworks.decode-speedups.baseline/2";

/// Whether a prompt's answer largely re-uses its context (where prompt lookup pays) or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptClass {
    /// The answer copies spans of the prompt (code edit, RAG answer, summary).
    Predictable,
    /// The answer is new text (chat, creative).
    OpenEnded,
}

impl PromptClass {
    /// `predictable` / `open_ended`.
    pub fn label(self) -> &'static str {
        match self {
            PromptClass::Predictable => "predictable",
            PromptClass::OpenEnded => "open_ended",
        }
    }
}

/// One benchmark / parity prompt.
#[derive(Clone, Debug)]
pub struct BenchPrompt {
    /// Stable row identity (`code_edit`, `rag_answer`, …).
    pub id: String,
    /// Predictable or open-ended.
    pub class: PromptClass,
    /// The conversation sent.
    pub messages: Vec<Message>,
}

impl BenchPrompt {
    /// A single-user-turn prompt.
    pub fn user(id: impl Into<String>, class: PromptClass, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            class,
            messages: vec![Message::user(text)],
        }
    }
}

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

/// The benchmark prompt set (epic sc-24432 acceptance test 3): three predictable prompts whose
/// answers copy their context — a code edit, a RAG answer, a summary — and two open-ended ones —
/// a chat turn and a creative piece. The same set feeds the pre-epic baseline and every later
/// campaign row, so the rows compare prompt for prompt.
pub fn speculative_prompt_set() -> Vec<BenchPrompt> {
    vec![
        BenchPrompt::user("code_edit", PromptClass::Predictable, CODE_EDIT),
        BenchPrompt::user("rag_answer", PromptClass::Predictable, RAG_ANSWER),
        BenchPrompt::user("summary", PromptClass::Predictable, SUMMARY),
        BenchPrompt::user("chat", PromptClass::OpenEnded, CHAT),
        BenchPrompt::user("creative", PromptClass::OpenEnded, CREATIVE),
    ]
}

/// A request over `prompt` under `sampling` with `speculative` set explicitly (never the legacy
/// field), seed pinned — the request the parity suite (greedy) and the benchmark (its configured
/// [`BenchConfig::sampling`]) send.
pub fn bench_request(
    prompt: &BenchPrompt,
    speculative: Speculative,
    sampling: &Sampling,
    max_new_tokens: u32,
) -> TextLlmRequest {
    TextLlmRequest {
        messages: prompt.messages.clone(),
        sampling: *sampling,
        max_new_tokens,
        seed: Some(0),
        speculative: Some(speculative),
        ..Default::default()
    }
}

/// An `f32` knob as the JSON number of its shortest decimal form (`0.7`, not `0.699999988…`).
fn f32_json(x: f32) -> Value {
    x.to_string()
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// The benchmark document's record of a [`Sampling`]: every knob, by its field name.
pub fn sampling_json(sampling: &Sampling) -> Value {
    json!({
        "temperature": f32_json(sampling.temperature),
        "top_p": f32_json(sampling.top_p),
        "top_k": sampling.top_k,
        "presence_penalty": f32_json(sampling.presence_penalty),
        "repetition_penalty": f32_json(sampling.repetition_penalty),
        "repetition_context": sampling.repetition_context,
    })
}

/// Parse a benchmark sampling spec: `"greedy"`, or a JSON object of [`sampling_json`]'s keys, each
/// optional over [`Sampling::greedy`] (so `{"temperature": 0.7, "top_p": 0.9}` is a seeded
/// stochastic run). An unknown key or a mistyped value is an error, never ignored.
pub fn parse_bench_sampling(spec: &str) -> Result<Sampling, String> {
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

/// The observable result of one generation: the streamed token events, the output text, the
/// generated-token count, the finish reason and the decode report.
struct Observed {
    tokens: Vec<(u32, String)>,
    text: String,
    generated: u32,
    finish: Option<FinishReason>,
    report: Option<DecodeReport>,
    ttft: Option<Duration>,
    wall: Duration,
    prompt_tokens: u32,
    timings: Option<core_llm::GenerationTimings>,
}

fn observe(provider: &dyn TextLlm, req: &TextLlmRequest) -> Result<Observed, String> {
    provider
        .validate(req)
        .map_err(|e| format!("validate refused the request: {e}"))?;
    let mut tokens = Vec::new();
    let mut ttft = None;
    let started = Instant::now();
    let out = provider
        .generate(req, &mut |event| {
            if let StreamEvent::Token { id, text, .. } = event {
                ttft.get_or_insert_with(|| started.elapsed());
                tokens.push((id, text));
            }
        })
        .map_err(|e| format!("generate failed: {e}"))?;
    let wall = started.elapsed();
    Ok(Observed {
        tokens,
        text: out.text,
        generated: out.usage.generated_tokens,
        finish: out.finish_reason,
        report: out.decode,
        ttft,
        wall,
        prompt_tokens: out.usage.prompt_tokens,
        timings: out.timings,
    })
}

/// One row of a parity case table: a speculative option and the proposer its report must name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParityCase {
    /// The option under test.
    pub speculative: Speculative,
    /// The proposer [`DecodeReport::proposer`] must name for it on this provider.
    pub expect_proposer: ProposerKind,
}

/// One passing parity row (for the caller's own assertions: that drafting actually happened).
#[derive(Clone, Debug, PartialEq)]
pub struct ParityRow {
    /// The prompt's id.
    pub prompt_id: String,
    /// The option run.
    pub speculative: Speculative,
    /// The report of the speculative run.
    pub report: DecodeReport,
    /// Tokens generated (identical to the `off` run's).
    pub generated_tokens: u32,
}

/// The greedy parity check (epic sc-24432 E1/E3): each `case` on each prompt emits exactly the
/// `off` run's stream — every token event, the text, the generated count and the finish reason —
/// and its report names `case.expect_proposer`; the `off` run's report names `none`. Returns every
/// row when all hold; otherwise one message listing every failure.
pub fn check_speculative_greedy_parity(
    provider: &dyn TextLlm,
    prompts: &[BenchPrompt],
    cases: &[ParityCase],
    max_new_tokens: u32,
) -> Result<Vec<ParityRow>, String> {
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for prompt in prompts {
        let off = match observe(
            provider,
            &bench_request(
                prompt,
                Speculative::Off,
                &Sampling::greedy(),
                max_new_tokens,
            ),
        ) {
            Ok(off) => off,
            Err(e) => {
                failures.push(format!("[{}] off: {e}", prompt.id));
                continue;
            }
        };
        let off_proposer = off.report.as_ref().map(|r| r.proposer);
        if off_proposer.is_some_and(|p| p != ProposerKind::None) {
            failures.push(format!(
                "[{}] off: the report names proposer {off_proposer:?}, not none",
                prompt.id
            ));
        }
        for case in cases {
            let tag = format!(
                "[{}] {}",
                prompt.id,
                serde_json::to_string(&case.speculative).unwrap_or_default()
            );
            let run = match observe(
                provider,
                &bench_request(
                    prompt,
                    case.speculative,
                    &Sampling::greedy(),
                    max_new_tokens,
                ),
            ) {
                Ok(run) => run,
                Err(e) => {
                    failures.push(format!("{tag}: {e}"));
                    continue;
                }
            };
            let Some(report) = run.report.clone() else {
                failures.push(format!("{tag}: the provider returned no decode report"));
                continue;
            };
            if report.proposer != case.expect_proposer {
                failures.push(format!(
                    "{tag}: the report names proposer `{}`, expected `{}`",
                    report.proposer.label(),
                    case.expect_proposer.label()
                ));
            }
            if run.tokens != off.tokens
                || run.text != off.text
                || run.generated != off.generated
                || run.finish != off.finish
            {
                let first = run
                    .tokens
                    .iter()
                    .zip(&off.tokens)
                    .position(|(a, b)| a != b)
                    .unwrap_or(run.tokens.len().min(off.tokens.len()));
                failures.push(format!(
                    "{tag}: greedy output differs from off (first differing token event @{first}; \
                     generated {} vs {}, finish {:?} vs {:?})",
                    run.generated, off.generated, run.finish, off.finish
                ));
                continue;
            }
            rows.push(ParityRow {
                prompt_id: prompt.id.clone(),
                speculative: case.speculative,
                report,
                generated_tokens: run.generated,
            });
        }
    }
    if failures.is_empty() {
        Ok(rows)
    } else {
        Err(failures.join("\n"))
    }
}

/// Tokens generated per row when a run does not say (`SPECULATIVE_BENCH_NEW_TOKENS`).
pub const BENCH_DEFAULT_NEW_TOKENS: u32 = 256;

/// Measured repeats per row when a run does not say (`SPECULATIVE_BENCH_REPEATS`): enough for a
/// standard deviation, so the run-to-run noise E6 compares against is measured, not assumed.
pub const BENCH_DEFAULT_REPEATS: u32 = 3;

/// The prompt the untimed warm-up runs when the loaded provider holds a cross-turn prefix cache
/// (its [`LoadReport::prefix_cache_bytes`] is non-zero). Its content shares no leading token with
/// any [`speculative_prompt_set`] prompt, so the warm-up's cached entry can lend a measured row at
/// most the chat template's fixed lead-in (none on a hybrid decoder, whose entries are only reused
/// whole), never the measured prompt itself. With the cache off the warm-up runs the measured
/// request itself — the closest warm-up, and nothing for it to leak.
pub const BENCH_WARMUP_PROMPT: &str = "Count from one to ten in words, one number per line.";

/// What a benchmark run measures and labels.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// The model label recorded verbatim (a snapshot name, a fixture name).
    pub model: String,
    /// The backend label recorded verbatim (`candle-cpu`, `candle-cuda`, `mlx`).
    pub backend: String,
    /// Tokens generated per row.
    pub max_new_tokens: u32,
    /// The sampling every row runs under (seed pinned to 0): [`Sampling::greedy`] for the
    /// greedy baseline, or a seeded stochastic setting — the product's `auto` + temperature path.
    pub sampling: Sampling,
    /// The options each prompt runs under, in row order.
    pub options: Vec<Speculative>,
    /// Run each (prompt, option) once untimed before its measured repeats (see
    /// [`BENCH_WARMUP_PROMPT`] for which request the warm-up sends).
    pub warmup: bool,
    /// Measured repeats per (prompt, option), at least one.
    pub repeats: u32,
}

/// One measured repeat of a [`BenchRow`].
#[derive(Clone, Debug, PartialEq)]
pub struct BenchSample {
    /// Wall time from the `generate` call to the first streamed token.
    pub ttft_ms: Option<f64>,
    /// Prefill time.
    pub prefill_ms: f64,
    /// Decode time.
    pub decode_ms: f64,
    /// `generated_tokens / decode` seconds.
    pub decode_tok_s: Option<f64>,
    /// Tokens generated.
    pub generated_tokens: u32,
    /// Prompt tokens this repeat restored from the cross-turn prefix cache (`None` without a
    /// decode report).
    pub prefix_hit_tokens: Option<u64>,
}

impl BenchSample {
    /// The sample as its JSON object (keys as in the [`BenchRow`] schema's `samples`).
    pub fn to_json(&self) -> Value {
        json!({
            "ttft_ms": self.ttft_ms,
            "decode_tok_s": self.decode_tok_s,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "generated_tokens": self.generated_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
        })
    }
}

/// Summary of one measured series across a row's repeats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchStats {
    /// Values summarized (repeats that produced the value).
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Smallest value.
    pub min: f64,
    /// Largest value.
    pub max: f64,
    /// Sample standard deviation (Bessel-corrected, `n - 1`); `None` for a single value.
    pub stddev: Option<f64>,
}

impl BenchStats {
    /// The summary of `values`, or `None` when there are none.
    pub fn of(values: &[f64]) -> Option<Self> {
        let n = values.len();
        if n == 0 {
            return None;
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = |n: usize| values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        Some(Self {
            n,
            mean,
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            stddev: (n > 1).then(|| variance(n - 1).sqrt()),
        })
    }

    /// `{"n", "mean", "min", "max", "stddev"}` for `values`, or `null` when there are none.
    pub fn json_of(values: &[f64]) -> Value {
        Self::of(values).map_or(
            Value::Null,
            |s| json!({"n": s.n, "mean": s.mean, "min": s.min, "max": s.max, "stddev": s.stddev}),
        )
    }
}

/// One benchmarked (prompt, option): its telemetry and every measured repeat. Timings come from
/// the provider's own [`GenerationTimings`](core_llm::GenerationTimings) when it reports them
/// (`timing_source = backend`), else from the wall clock around `generate` (`wall`).
///
/// # JSON schema ([`BENCH_SCHEMA`])
///
/// This is the one place the row schema is defined; [`BenchDocument::to_json`] wraps the rows, and
/// both backends' entry points (and the pre-epic baseline driver, which copies it) emit exactly
/// these keys. A field the provider does not report is `null`.
///
/// | key | meaning |
/// |-----|---------|
/// | `prompt_id`, `class` | the prompt and `predictable` / `open_ended` |
/// | `requested` | the option sent (`"off"`, `"auto"`, `{"proposer", "depth"}`) |
/// | `proposer`, `path`, `draft_tokens` | [`DecodeReport`]'s proposer label, decode path and depth |
/// | `prompt_tokens`, `generated_tokens` | token counts |
/// | `repeats` | measured repeats (the length of `samples`) |
/// | `ttft_ms`, `decode_tok_s`, `prefill_ms`, `decode_ms` | [`BenchStats`] over the repeats: `{"n", "mean", "min", "max", "stddev"}` (`stddev` `null` for one repeat; the whole object `null` when no repeat produced the value) |
/// | `timing_source` | `backend` or `wall` |
/// | `target_forwards`, `prefill_forwards`, `verify_steps`, `replay_forwards` | [`DecodeReport`] forward counts |
/// | `proposed_tokens`, `accepted_tokens`, `mean_accepted_length` | draft accounting ([`DecodeReport::mean_accepted_length`]) |
/// | `prefix_cache` | `{"path", "reason"}` — [`DecodeReport::prefix_cache`] |
/// | `prefix_hit_tokens` | prompt tokens restored from the prefix cache |
/// | `sampler`, `kv_cache`, `attention` | [`DecodeReport`] labels |
/// | `cuda_graphs` | [`CudaGraphsReport::path`](core_llm::CudaGraphsReport::path) |
/// | `graph_path` | [`DecodeReport::graph_path`] |
/// | `fused` | `{"path", "reason"}` — [`DecodeReport::fused_primitives`] |
/// | `fallbacks` | [`DecodeReport::fallbacks`] |
/// | `samples` | one object per repeat: `ttft_ms`, `decode_tok_s`, `prefill_ms`, `decode_ms`, `generated_tokens`, `prefix_hit_tokens` |
///
/// Every key above `samples` but the timing statistics is the **first** measured repeat's (the
/// repeat the warm-up isolation covers, [`BENCH_WARMUP_PROMPT`]). With a prefix cache on, later
/// repeats restore the earlier repeats' prompt — each sample's `prefix_hit_tokens` shows it — so a
/// cold-prefill TTFT comes from a run with `prefix_cache_bytes` 0 (the entry points' default).
#[derive(Clone, Debug, PartialEq)]
pub struct BenchRow {
    /// The prompt's id.
    pub prompt_id: String,
    /// Its class.
    pub class: PromptClass,
    /// The option requested.
    pub requested: Speculative,
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Tokens the first repeat generated.
    pub generated_tokens: u32,
    /// `backend` or `wall` (the first repeat's).
    pub timing_source: &'static str,
    /// The first repeat's decode report (`None` when the provider reports none).
    pub report: Option<DecodeReport>,
    /// Every measured repeat, in run order.
    pub samples: Vec<BenchSample>,
}

impl BenchRow {
    /// The row as its JSON object (the schema above).
    pub fn to_json(&self) -> Value {
        let report = self.report.as_ref();
        let series = |f: fn(&BenchSample) -> Option<f64>| {
            BenchStats::json_of(&self.samples.iter().filter_map(f).collect::<Vec<_>>())
        };
        let path_json = |p: &core_llm::PathReport| json!({"path": p.path, "reason": p.reason});
        json!({
            "prompt_id": self.prompt_id,
            "class": self.class.label(),
            "requested": self.requested,
            "proposer": report.map(|r| r.proposer.label()),
            "path": report.map(|r| r.path.clone()),
            "draft_tokens": report.and_then(|r| r.draft_tokens),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "repeats": self.samples.len(),
            "ttft_ms": series(|s| s.ttft_ms),
            "decode_tok_s": series(|s| s.decode_tok_s),
            "prefill_ms": series(|s| Some(s.prefill_ms)),
            "decode_ms": series(|s| Some(s.decode_ms)),
            "timing_source": self.timing_source,
            "target_forwards": report.map(|r| r.target_forwards),
            "prefill_forwards": report.map(|r| r.prefill_forwards),
            "verify_steps": report.map(|r| r.verify_steps),
            "replay_forwards": report.map(|r| r.replay_forwards),
            "proposed_tokens": report.map(|r| r.proposed_tokens),
            "accepted_tokens": report.map(|r| r.accepted_tokens),
            "mean_accepted_length": report.and_then(DecodeReport::mean_accepted_length),
            "prefix_cache": report.map(|r| path_json(&r.prefix_cache)),
            "prefix_hit_tokens": report.map(|r| r.prefix_hit_tokens),
            "sampler": report.map(|r| r.sampler.clone()),
            "kv_cache": report.map(|r| r.kv_cache.clone()),
            "attention": report.map(|r| r.attention.clone()),
            "cuda_graphs": report.map(|r| r.cuda_graphs.path.clone()),
            "graph_path": report.map(|r| r.graph_path.clone()),
            "fused": report.map(|r| path_json(&r.fused_primitives)),
            "fallbacks": report.map(|r| r.fallbacks.clone()),
            "samples": self.samples.iter().map(BenchSample::to_json).collect::<Vec<_>>(),
        })
    }
}

/// A finished benchmark run: the configuration, what the load settled, and every row.
#[derive(Clone, Debug)]
pub struct BenchDocument {
    /// What was measured.
    pub config: BenchConfig,
    /// The provider's [`TextLlm::load_report`] (`None` when it reports none).
    pub load: Option<LoadReport>,
    /// One row per (prompt, option), prompt-major.
    pub rows: Vec<BenchRow>,
}

impl BenchDocument {
    /// The baseline-format JSON document ([`BENCH_SCHEMA`]): `schema`, `model`, `backend`,
    /// `max_new_tokens`, `sampling` ([`sampling_json`]), `warmup`, `repeats`, `options`, `load`
    /// (`{"prefix_cache_bytes", "draft": {"source", "refusal"} | null, "cuda_graphs",
    /// "fallbacks"}` from the load report — the settled prefix-cache budget and what became of a
    /// named draft / MTP head — or `null`), and `rows` ([`BenchRow`]'s schema).
    pub fn to_json(&self) -> Value {
        let load = self.load.as_ref().map(|r| {
            json!({
                "prefix_cache_bytes": r.prefix_cache_bytes,
                "draft": r.draft.as_ref().map(|d| json!({"source": d.source, "refusal": d.refusal})),
                "cuda_graphs": r.cuda_graphs,
                "fallbacks": r.fallbacks,
            })
        });
        json!({
            "schema": BENCH_SCHEMA,
            "model": self.config.model,
            "backend": self.config.backend,
            "max_new_tokens": self.config.max_new_tokens,
            "sampling": sampling_json(&self.config.sampling),
            "warmup": self.config.warmup,
            "repeats": self.config.repeats,
            "options": self.config.options,
            "load": load,
            "rows": self.rows.iter().map(BenchRow::to_json).collect::<Vec<_>>(),
        })
    }

    /// Write the document to `path`, refusing to overwrite an existing file (a sealed baseline is
    /// never replaced by accident).
    pub fn write_new(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let text = serde_json::to_string_pretty(&self.to_json()).map_err(std::io::Error::other)?;
        file.write_all(text.as_bytes())?;
        file.write_all(b"\n")
    }
}

/// One measured generation as a [`BenchSample`], with its timing source.
fn bench_sample(run: &Observed) -> (BenchSample, &'static str) {
    let (prefill, decode, timing_source) = match run.timings {
        Some(t) => (t.prefill, t.decode, "backend"),
        None => {
            let ttft = run.ttft.unwrap_or(run.wall);
            (ttft, run.wall.saturating_sub(ttft), "wall")
        }
    };
    let decode_s = decode.as_secs_f64();
    let sample = BenchSample {
        ttft_ms: run.ttft.map(|d| d.as_secs_f64() * 1e3),
        prefill_ms: prefill.as_secs_f64() * 1e3,
        decode_ms: decode_s * 1e3,
        decode_tok_s: (decode_s > 0.0).then(|| f64::from(run.generated) / decode_s),
        generated_tokens: run.generated,
        prefix_hit_tokens: run.report.as_ref().map(|r| r.prefix_hit_tokens),
    };
    (sample, timing_source)
}

/// Run the benchmark: every prompt under every option in `config.options`, under
/// `config.sampling` — an untimed warm-up (when `config.warmup`; [`BENCH_WARMUP_PROMPT`] says
/// which request) and then `config.repeats` measured runs — one [`BenchRow`] each. Fails on the
/// first request the provider refuses or cannot generate — a benchmark row that silently went
/// missing would read as coverage.
pub fn run_speculative_bench(
    provider: &dyn TextLlm,
    prompts: &[BenchPrompt],
    config: &BenchConfig,
) -> Result<BenchDocument, String> {
    if config.repeats == 0 {
        return Err("a benchmark row needs at least one measured repeat".into());
    }
    let load = provider.load_report();
    let isolated_warmup = load
        .as_ref()
        .and_then(|r| r.prefix_cache_bytes)
        .is_some_and(|bytes| bytes > 0);
    let warmup_prompt = BenchPrompt::user("warmup", PromptClass::OpenEnded, BENCH_WARMUP_PROMPT);
    let mut rows = Vec::with_capacity(prompts.len() * config.options.len());
    for prompt in prompts {
        for &option in &config.options {
            let req = bench_request(prompt, option, &config.sampling, config.max_new_tokens);
            let tag = format!(
                "[{}] {}",
                prompt.id,
                serde_json::to_string(&option).unwrap_or_default()
            );
            if config.warmup {
                let warm = if isolated_warmup {
                    bench_request(
                        &warmup_prompt,
                        option,
                        &config.sampling,
                        config.max_new_tokens,
                    )
                } else {
                    req.clone()
                };
                observe(provider, &warm).map_err(|e| format!("{tag} warm-up: {e}"))?;
            }
            let mut first = None;
            let mut samples = Vec::with_capacity(config.repeats as usize);
            for repeat in 0..config.repeats {
                let run =
                    observe(provider, &req).map_err(|e| format!("{tag} repeat {repeat}: {e}"))?;
                let (sample, timing_source) = bench_sample(&run);
                samples.push(sample);
                first.get_or_insert((run, timing_source));
            }
            let (run, timing_source) = first.expect("at least one repeat ran");
            rows.push(BenchRow {
                prompt_id: prompt.id.clone(),
                class: prompt.class,
                requested: option,
                prompt_tokens: run.prompt_tokens,
                generated_tokens: run.generated,
                timing_source,
                report: run.report,
                samples,
            });
        }
    }
    Ok(BenchDocument {
        config: config.clone(),
        load,
        rows,
    })
}

/// One benchmark knob from the process environment: `None` when unset or blank.
pub fn bench_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// A `SPECULATIVE_BENCH_FORMAT` as the load's `quantize`: `bf16` is the dense load, `q8` / `q4`
/// the backend's 8- / 4-bit load path, `nvfp4` the at-load NVFP4 quantization (refused off CUDA
/// sm_120+ with the typed capability error, never a fallback).
pub fn parse_bench_format(format: &str) -> Result<Option<Quantize>, String> {
    match format {
        "bf16" => Ok(None),
        "q8" => Ok(Some(Quantize::Q8)),
        "q4" => Ok(Some(Quantize::Q4)),
        "nvfp4" => Ok(Some(Quantize::Nvfp4)),
        other => Err(format!(
            "SPECULATIVE_BENCH_FORMAT must be bf16, q8, q4 or nvfp4, got {other}"
        )),
    }
}

/// The load an entry point benchmarks: `snapshot` with the `SPECULATIVE_BENCH_FORMAT`,
/// `SPECULATIVE_BENCH_DRAFT`, `SPECULATIVE_BENCH_MTP_HEAD` and
/// `SPECULATIVE_BENCH_PREFIX_CACHE_BYTES` knobs read through `var` (see
/// [`run_speculative_bench_from_env`]). The prefix cache defaults to **off** (`Some(0)`), so a row's
/// TTFT is a cold prefill comparable with the pre-epic baseline; `default` asks for the backend
/// default budget (`None`).
pub fn bench_load_spec(
    snapshot: &str,
    var: &dyn Fn(&str) -> Option<String>,
) -> Result<LoadSpec, String> {
    let format = var("SPECULATIVE_BENCH_FORMAT").unwrap_or_else(|| "bf16".into());
    let prefix_cache_bytes = match var("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES").as_deref() {
        None => Some(0),
        Some("default") => None,
        Some(bytes) => Some(bytes.parse().map_err(|_| {
            format!("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES must be a byte count or `default`, got {bytes}")
        })?),
    };
    Ok(LoadSpec {
        quantize: parse_bench_format(&format)?,
        mtp_head_source: var("SPECULATIVE_BENCH_MTP_HEAD"),
        draft_source: var("SPECULATIVE_BENCH_DRAFT"),
        prefix_cache_bytes,
        ..LoadSpec::dense(snapshot)
    })
}

/// What an entry point measures over `snapshot`, from the knobs read through `var` (see
/// [`run_speculative_bench_from_env`]); `default_backend` labels the rows unless
/// `SPECULATIVE_BENCH_BACKEND` does.
pub fn bench_config(
    snapshot: &str,
    default_backend: &str,
    var: &dyn Fn(&str) -> Option<String>,
) -> Result<BenchConfig, String> {
    let count = |name: &str, default: u32| -> Result<u32, String> {
        var(name).map_or(Ok(default), |v| {
            v.parse()
                .map_err(|_| format!("{name} must be a non-negative integer, got {v}"))
        })
    };
    let format = var("SPECULATIVE_BENCH_FORMAT").unwrap_or_else(|| "bf16".into());
    let model = var("SPECULATIVE_BENCH_MODEL").unwrap_or_else(|| {
        let name = std::path::Path::new(snapshot).file_name().map_or_else(
            || snapshot.to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        if format == "bf16" {
            name
        } else {
            format!("{name}@{format}")
        }
    });
    let options = match var("SPECULATIVE_BENCH_OPTIONS") {
        Some(json) => serde_json::from_str(&json)
            .map_err(|e| format!("SPECULATIVE_BENCH_OPTIONS is not a JSON option list: {e}"))?,
        None => vec![Speculative::Off, Speculative::Auto],
    };
    let sampling = match var("SPECULATIVE_BENCH_SAMPLING") {
        Some(spec) => {
            parse_bench_sampling(&spec).map_err(|e| format!("SPECULATIVE_BENCH_SAMPLING: {e}"))?
        }
        None => Sampling::greedy(),
    };
    Ok(BenchConfig {
        model,
        backend: var("SPECULATIVE_BENCH_BACKEND").unwrap_or_else(|| default_backend.into()),
        max_new_tokens: count("SPECULATIVE_BENCH_NEW_TOKENS", BENCH_DEFAULT_NEW_TOKENS)?,
        sampling,
        options,
        warmup: var("SPECULATIVE_BENCH_WARMUP").as_deref() != Some("0"),
        repeats: count("SPECULATIVE_BENCH_REPEATS", BENCH_DEFAULT_REPEATS)?,
    })
}

/// The real-weight benchmark entry both backends' `#[ignore]`d `speculative_bench` tests call
/// (epic sc-24432 acceptance test 3 / E6): load `SPECULATIVE_BENCH_SNAPSHOT` through `load`, run
/// [`run_speculative_bench`] over [`speculative_prompt_set`], and write the document to
/// `SPECULATIVE_BENCH_OUTPUT` (never over an existing file). Returns the path and the document.
/// Every input is passed in; nothing is derived from a cache:
///
/// | variable | meaning |
/// |----------|---------|
/// | `SPECULATIVE_BENCH_SNAPSHOT` | snapshot directory (config.json, tokenizer*.json, shards) |
/// | `SPECULATIVE_BENCH_OUTPUT` | JSON path to write (must not exist) |
/// | `SPECULATIVE_BENCH_OPTIONS` | JSON array of speculative options (default `["off","auto"]`), e.g. `["off",{"proposer":"prompt_lookup","depth":4}]` |
/// | `SPECULATIVE_BENCH_SAMPLING` | JSON sampling spec (default `"greedy"`; [`parse_bench_sampling`]), e.g. `{"temperature":0.7,"top_p":0.9}` (seed pinned to 0) |
/// | `SPECULATIVE_BENCH_FORMAT` | projection format quantized at load ([`parse_bench_format`]): `bf16` (default), `q8`, `q4`, `nvfp4` |
/// | `SPECULATIVE_BENCH_DRAFT` | draft model snapshot (`LoadSpec::draft_source`; default none) |
/// | `SPECULATIVE_BENCH_MTP_HEAD` | companion MTP head artifact (`LoadSpec::mtp_head_source`; default none) |
/// | `SPECULATIVE_BENCH_PREFIX_CACHE_BYTES` | prefix-cache budget (`LoadSpec::prefix_cache_bytes`): default `0` (off — a cold-prefill TTFT), a byte count, or `default` for the backend default |
/// | `SPECULATIVE_BENCH_NEW_TOKENS` | tokens generated per run (default [`BENCH_DEFAULT_NEW_TOKENS`]) |
/// | `SPECULATIVE_BENCH_REPEATS` | measured repeats per row (default [`BENCH_DEFAULT_REPEATS`]) |
/// | `SPECULATIVE_BENCH_MODEL` | model label recorded verbatim (default: the snapshot's directory name, `@<format>` appended unless bf16) |
/// | `SPECULATIVE_BENCH_BACKEND` | backend label recorded verbatim (default: the entry point's) |
/// | `SPECULATIVE_BENCH_WARMUP` | `0` skips the untimed warm-up per row (default on) |
pub fn run_speculative_bench_from_env(
    default_backend: &str,
    load: DraftLoader<'_>,
) -> Result<(PathBuf, BenchDocument), String> {
    let snapshot =
        bench_env("SPECULATIVE_BENCH_SNAPSHOT").ok_or("set SPECULATIVE_BENCH_SNAPSHOT")?;
    let output =
        PathBuf::from(bench_env("SPECULATIVE_BENCH_OUTPUT").ok_or("set SPECULATIVE_BENCH_OUTPUT")?);
    if output.exists() {
        return Err(format!(
            "{} exists; a baseline is never overwritten",
            output.display()
        ));
    }
    let spec = bench_load_spec(&snapshot, &bench_env)?;
    let config = bench_config(&snapshot, default_backend, &bench_env)?;
    let provider = load(&spec).map_err(|e| format!("load {snapshot}: {e}"))?;
    let doc = run_speculative_bench(provider.as_ref(), &speculative_prompt_set(), &config)?;
    doc.write_new(&output)
        .map_err(|e| format!("write {}: {e}", output.display()))?;
    Ok((output, doc))
}

/// The weights-free run of the benchmark both backends' `speculative_bench` tests make, on the
/// shared draft-model fixture's target ([`write_draft_model_fixture`], written under `root`) loaded
/// through `load`:
///
/// * with the entry points' default load ([`bench_load_spec`] with no knobs: the prefix cache off),
///   two repeats of `off`, prompt lookup and `auto` over the fixture prompts give one row per
///   (prompt, option) with every schema field, the timing statistics over both repeats, and no
///   prefix-cache hit anywhere;
/// * with a prefix-cache budget, the warm-up does not lend the measured prompt to its first
///   repeat: that repeat restores exactly the template lead-in a different prompt run first would
///   (measured on a separately loaded control), and the second repeat's sample records its own,
///   larger hit.
pub fn check_speculative_bench_on_fixture(
    root: &Path,
    backend: &str,
    load: DraftLoader<'_>,
) -> Result<(), String> {
    let fixture = write_draft_model_fixture(root).map_err(|e| format!("write the fixture: {e}"))?;
    let target = fixture.target.to_string_lossy().into_owned();
    let prompts = draft_model_prompts();
    let config = BenchConfig {
        model: "draft-fixture-target".into(),
        backend: backend.into(),
        max_new_tokens: 8,
        sampling: Sampling::greedy(),
        options: vec![
            Speculative::Off,
            Speculative::proposer(SpeculativeProposer::PromptLookup, 3),
            Speculative::Auto,
        ],
        warmup: true,
        repeats: 2,
    };
    let mut failures = Vec::new();

    let cold = load(&bench_load_spec(&target, &|_| None)?)?;
    let json = run_speculative_bench(cold.as_ref(), &prompts, &config)?.to_json();
    if json["load"]["prefix_cache_bytes"] != json!(0) {
        failures.push(format!("the default load settled {}", json["load"]));
    }
    let rows = json["rows"].as_array().cloned().unwrap_or_default();
    if rows.len() != prompts.len() * config.options.len() {
        failures.push(format!("{} rows", rows.len()));
    }
    for row in &rows {
        let tag = format!("[{}] {}", row["prompt_id"], row["requested"]);
        for key in [
            "graph_path",
            "attention",
            "fused",
            "prefix_cache",
            "target_forwards",
            "prefill_forwards",
            "verify_steps",
            "replay_forwards",
        ] {
            if row[key].is_null() {
                failures.push(format!("{tag}: `{key}` is null"));
            }
        }
        if row["repeats"] != 2 || row["samples"].as_array().map(Vec::len) != Some(2) {
            failures.push(format!("{tag}: not two repeats: {}", row["samples"]));
        }
        for key in ["ttft_ms", "decode_tok_s"] {
            let s = &row[key];
            let ordered =
                s["min"].as_f64() <= s["mean"].as_f64() && s["mean"].as_f64() <= s["max"].as_f64();
            if s["n"] != 2 || !s["stddev"].is_number() || !ordered {
                failures.push(format!("{tag}: `{key}` is not a two-repeat summary: {s}"));
            }
        }
        let hits: Vec<_> = row["samples"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|s| s["prefix_hit_tokens"].clone())
            .chain([row["prefix_hit_tokens"].clone()])
            .collect();
        if hits.iter().any(|h| h != &json!(0)) || row["prefix_cache"]["path"] != "off" {
            failures.push(format!(
                "{tag}: the cache-off run reused a prefix: {hits:?}, {}",
                row["prefix_cache"]
            ));
        }
    }
    let proposers: Vec<_> = rows.iter().take(2).map(|r| r["proposer"].clone()).collect();
    if proposers != [json!("none"), json!("prompt_lookup")] {
        failures.push(format!("proposers {proposers:?}"));
    }

    // The warm-up isolation, with the cache on.
    let cache_on = |name: &str| {
        (name == "SPECULATIVE_BENCH_PREFIX_CACHE_BYTES").then(|| (64u64 << 20).to_string())
    };
    let spec = bench_load_spec(&target, &cache_on)?;
    let control = load(&spec)?;
    let off = |p: &BenchPrompt| bench_request(p, Speculative::Off, &Sampling::greedy(), 8);
    observe(control.as_ref(), &off(&prompts[1]))?;
    let lead_in = observe(control.as_ref(), &off(&prompts[0]))?
        .report
        .map(|r| r.prefix_hit_tokens)
        .ok_or("the control run reported no decode report")?;
    let warm = load(&spec)?;
    if warm
        .load_report()
        .and_then(|r| r.prefix_cache_bytes)
        .is_none_or(|b| b == 0)
    {
        return Err("a 64 MiB prefix-cache load settled no cache; the check needs one".into());
    }
    let doc = run_speculative_bench(
        warm.as_ref(),
        &prompts[..1],
        &BenchConfig {
            options: vec![Speculative::Off],
            ..config
        },
    )?;
    let row = &doc.rows[0];
    if lead_in + 1 >= u64::from(row.prompt_tokens) {
        failures.push(format!(
            "the control's lead-in ({lead_in}) spans the {} prompt tokens; the fixture prompts \
             must diverge after the template",
            row.prompt_tokens
        ));
    }
    let first = row.report.as_ref().map(|r| r.prefix_hit_tokens);
    let second = row.samples.get(1).and_then(|s| s.prefix_hit_tokens);
    if first != Some(lead_in) || row.samples[0].prefix_hit_tokens != first {
        failures.push(format!(
            "the first measured repeat restored {first:?} tokens; a prompt other than the \
             measured one, run first, lends only the template lead-in ({lead_in}) of the {} \
             prompt tokens",
            row.prompt_tokens
        ));
    }
    if second.is_none_or(|s| s <= lead_in) {
        failures.push(format!(
            "the second repeat's sample records {second:?} restored tokens; it re-runs the \
             measured prompt, so it restores more than the lead-in ({lead_in})"
        ));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        Channel, ProposerCapabilities, Result as CoreResult, TextLlmCapabilities,
        TextLlmDescriptor, TextLlmOutput, Usage,
    };
    use std::cell::RefCell;

    /// A deterministic provider that resolves the speculative option like a backend does and
    /// reports it; `diverge` corrupts the stream whenever a proposer runs, `mislabel` reports the
    /// wrong proposer. It records each request's prompt text (`seen`) and reports the prefix-cache
    /// budget `prefix_cache_bytes` in its load report.
    struct SpeculativeStub {
        descriptor: TextLlmDescriptor,
        diverge: bool,
        mislabel: bool,
        prefix_cache_bytes: Option<u64>,
        seen: RefCell<Vec<String>>,
    }

    fn stub(diverge: bool, mislabel: bool) -> SpeculativeStub {
        SpeculativeStub {
            descriptor: TextLlmDescriptor {
                id: "spec-stub".into(),
                family: "stub".into(),
                backend: "test".into(),
                capabilities: TextLlmCapabilities {
                    // The backends' finite ceiling (sc-24438: 8 verify rows = 7 drafts).
                    speculative: vec![ProposerCapabilities {
                        proposer: SpeculativeProposer::PromptLookup,
                        max_depth: 7,
                        recommended_depth: 4,
                    }],
                    ..Default::default()
                },
            },
            diverge,
            mislabel,
            prefix_cache_bytes: Some(0),
            seen: RefCell::default(),
        }
    }

    impl TextLlm for SpeculativeStub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn load_report(&self) -> Option<LoadReport> {
            Some(LoadReport {
                prefix_cache_bytes: self.prefix_cache_bytes,
                ..LoadReport::default()
            })
        }

        fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
            self.descriptor
                .capabilities
                .validate_request("spec-stub", req)
        }

        fn generate(
            &self,
            req: &TextLlmRequest,
            on_event: &mut dyn FnMut(StreamEvent),
        ) -> CoreResult<TextLlmOutput> {
            let plan = core_llm::resolve_speculative(
                req.speculative_mode(),
                &self.descriptor.capabilities,
            )
            .plan;
            self.seen.borrow_mut().push(req.messages[0].text_content());
            let speculating = plan.proposer() != ProposerKind::None;
            let mut text = String::new();
            for i in 0..req.max_new_tokens as usize {
                let id = if self.diverge && speculating && i == 2 {
                    99
                } else {
                    (i % 3) as u32
                };
                let piece = format!("t{id} ");
                text.push_str(&piece);
                on_event(StreamEvent::Token {
                    id,
                    text: piece,
                    index: i,
                    channel: Channel::Content,
                });
            }
            let usage = Usage {
                prompt_tokens: 4,
                generated_tokens: req.max_new_tokens,
            };
            on_event(StreamEvent::Done {
                finish_reason: FinishReason::Length,
                usage,
            });
            let proposer = if self.mislabel && speculating {
                ProposerKind::Mtp
            } else {
                plan.proposer()
            };
            Ok(TextLlmOutput {
                text,
                usage,
                decode: Some(DecodeReport {
                    path: if speculating {
                        "prompt_lookup"
                    } else {
                        "step_model"
                    }
                    .into(),
                    proposer,
                    draft_tokens: plan.depth(),
                    // What the stub was asked to sample with, so the bench's threading of
                    // `BenchConfig::sampling` into the request is observable.
                    sampler: if req.sampling.is_greedy() {
                        "none"
                    } else {
                        "host:stub"
                    }
                    .into(),
                    verify_steps: if speculating { 4 } else { 0 },
                    proposed_tokens: if speculating { 8 } else { 0 },
                    accepted_tokens: if speculating { 6 } else { 0 },
                    ..Default::default()
                }),
                finish_reason: Some(FinishReason::Length),
                ..Default::default()
            })
        }
    }

    fn cases() -> Vec<ParityCase> {
        vec![
            ParityCase {
                speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
                expect_proposer: ProposerKind::PromptLookup,
            },
            ParityCase {
                speculative: Speculative::Auto,
                expect_proposer: ProposerKind::PromptLookup,
            },
        ]
    }

    #[test]
    fn the_parity_suite_passes_an_exact_provider_and_names_every_failure_otherwise() {
        let prompts = speculative_prompt_set();
        let rows = check_speculative_greedy_parity(&stub(false, false), &prompts, &cases(), 6)
            .expect("an exact provider passes");
        assert_eq!(rows.len(), prompts.len() * 2);
        assert!(rows
            .iter()
            .all(|r| r.report.proposer == ProposerKind::PromptLookup));

        let err =
            check_speculative_greedy_parity(&stub(true, false), &prompts, &cases(), 6).unwrap_err();
        assert_eq!(err.lines().count(), prompts.len() * 2, "{err}");
        assert!(err.contains("first differing token event @2"), "{err}");
        let err = check_speculative_greedy_parity(&stub(false, true), &prompts[..1], &cases(), 6)
            .unwrap_err();
        assert!(err.contains("expected `prompt_lookup`"), "{err}");
        // A refused option is a failure, never a skipped row.
        let refused = [ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::DraftModel, 2),
            expect_proposer: ProposerKind::DraftModel,
        }];
        let err = check_speculative_greedy_parity(&stub(false, false), &prompts[..1], &refused, 6)
            .unwrap_err();
        assert!(err.contains("validate refused"), "{err}");
    }

    #[test]
    fn the_prompt_set_is_three_predictable_and_two_open_ended_prompts() {
        let set = speculative_prompt_set();
        let ids: Vec<_> = set.iter().map(|p| (p.id.as_str(), p.class)).collect();
        assert_eq!(
            ids,
            [
                ("code_edit", PromptClass::Predictable),
                ("rag_answer", PromptClass::Predictable),
                ("summary", PromptClass::Predictable),
                ("chat", PromptClass::OpenEnded),
                ("creative", PromptClass::OpenEnded),
            ]
        );
    }

    fn stub_config() -> BenchConfig {
        BenchConfig {
            model: "spec-stub".into(),
            backend: "test".into(),
            max_new_tokens: 5,
            options: vec![
                Speculative::Off,
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
            ],
            warmup: true,
            sampling: Sampling::greedy(),
            repeats: 3,
        }
    }

    #[test]
    fn the_bench_writes_one_baseline_row_per_prompt_and_option_and_never_overwrites() {
        let config = stub_config();
        let prompts = speculative_prompt_set();
        let doc = run_speculative_bench(&stub(false, false), &prompts, &config).unwrap();
        assert_eq!(doc.rows.len(), 10);
        let json = doc.to_json();
        assert_eq!(json["schema"], BENCH_SCHEMA);
        assert_eq!(json["repeats"], 3);
        assert_eq!(json["load"]["prefix_cache_bytes"], 0);
        assert_eq!(
            json["options"][1],
            json!({"proposer": "prompt_lookup", "depth": 4})
        );
        assert_eq!(json["sampling"], sampling_json(&Sampling::greedy()));
        assert_eq!(json["sampling"]["temperature"], 0.0);
        let off = &json["rows"][0];
        assert_eq!(off["prompt_id"], "code_edit");
        assert_eq!(off["sampler"], "none");
        assert_eq!(off["class"], "predictable");
        assert_eq!(off["requested"], "off");
        assert_eq!(off["proposer"], "none");
        assert_eq!(off["mean_accepted_length"], Value::Null);
        let lookup = &json["rows"][1];
        assert_eq!(lookup["proposer"], "prompt_lookup");
        assert_eq!(lookup["draft_tokens"], 4);
        assert_eq!(lookup["mean_accepted_length"], 1.5);
        assert_eq!(lookup["verify_steps"], 4);
        assert_eq!(lookup["generated_tokens"], 5);
        assert_eq!(lookup["repeats"], 3);
        assert_eq!(lookup["samples"].as_array().unwrap().len(), 3);
        assert_eq!(lookup["decode_tok_s"]["n"], 3);
        assert_eq!(lookup["prefix_cache"], json!({"path": "", "reason": null}));
        assert_eq!(lookup["fused"], json!({"path": "", "reason": null}));
        for key in [
            "decode_tok_s",
            "ttft_ms",
            "prefill_ms",
            "decode_ms",
            "fallbacks",
            "sampler",
            "graph_path",
            "attention",
            "prefill_forwards",
            "replay_forwards",
            "prefix_hit_tokens",
        ] {
            assert!(!lookup[key].is_null(), "row lacks `{key}`: {lookup}");
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("baseline.json");
        doc.write_new(&path).unwrap();
        let back: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // Field for field (the float timings round-trip to within an ULP, not bit for bit).
        assert_eq!(back["schema"], json["schema"]);
        assert_eq!(back["options"], json["options"]);
        assert_eq!(back["rows"].as_array().unwrap().len(), 10);
        assert_eq!(back["rows"][1]["proposer"], "prompt_lookup");
        assert!(
            doc.write_new(&path).is_err(),
            "a sealed baseline is never overwritten"
        );

        // A seeded stochastic run: the document records the knobs, and every request carried them.
        let stochastic = parse_bench_sampling(r#"{"temperature": 0.7, "top_p": 0.9}"#).unwrap();
        let doc = run_speculative_bench(
            &stub(false, false),
            &prompts[..1],
            &BenchConfig {
                sampling: stochastic,
                ..config.clone()
            },
        )
        .unwrap();
        let json = doc.to_json();
        assert_eq!(
            json["sampling"],
            json!({"temperature": 0.7, "top_p": 0.9, "top_k": 0, "presence_penalty": 0.0,
                   "repetition_penalty": 1.0, "repetition_context": 0})
        );
        for row in json["rows"].as_array().unwrap() {
            assert_eq!(row["sampler"], "host:stub", "{row}");
        }
        let zero = BenchConfig {
            repeats: 0,
            ..config
        };
        assert!(run_speculative_bench(&stub(false, false), &prompts, &zero).is_err());
    }

    /// E6 noise: the row's statistics are the mean, extremes and Bessel-corrected standard
    /// deviation of its repeats.
    #[test]
    fn bench_stats_summarize_a_series() {
        let s = BenchStats::of(&[10.0, 12.0, 14.0]).unwrap();
        assert_eq!((s.n, s.mean, s.min, s.max), (3, 12.0, 10.0, 14.0));
        assert_eq!(s.stddev, Some(2.0));
        let s = BenchStats::of(&[3.0, 1.0, 2.0, 6.0]).unwrap();
        assert_eq!((s.mean, s.min, s.max), (3.0, 1.0, 6.0));
        assert!((s.stddev.unwrap() - (14.0f64 / 3.0).sqrt()).abs() < 1e-12);
        let one = BenchStats::of(&[7.5]).unwrap();
        assert_eq!(
            (one.mean, one.min, one.max, one.stddev),
            (7.5, 7.5, 7.5, None)
        );
        assert_eq!(BenchStats::of(&[]), None);
        assert_eq!(
            BenchStats::json_of(&[10.0, 12.0, 14.0]),
            json!({"n": 3, "mean": 12.0, "min": 10.0, "max": 14.0, "stddev": 2.0})
        );
        assert_eq!(BenchStats::json_of(&[]), Value::Null);
    }

    /// The warm-up sends the measured request when the load holds no prefix cache, and the
    /// separate warm-up prompt when it does, so it never lends a measured row its own prompt.
    #[test]
    fn the_warmup_runs_the_measured_request_unless_a_prefix_cache_could_carry_it_over() {
        let prompts = speculative_prompt_set();
        let config = BenchConfig {
            options: vec![Speculative::Off],
            repeats: 2,
            ..stub_config()
        };
        for (budget, isolated) in [(Some(0), false), (None, false), (Some(1 << 20), true)] {
            let provider = SpeculativeStub {
                prefix_cache_bytes: budget,
                ..stub(false, false)
            };
            run_speculative_bench(&provider, &prompts[..2], &config).unwrap();
            let warm = if isolated {
                BENCH_WARMUP_PROMPT
            } else {
                CODE_EDIT
            };
            let warm2 = if isolated {
                BENCH_WARMUP_PROMPT
            } else {
                RAG_ANSWER
            };
            assert_eq!(
                *provider.seen.borrow(),
                [warm, CODE_EDIT, CODE_EDIT, warm2, RAG_ANSWER, RAG_ANSWER],
                "budget {budget:?}"
            );
        }
        // No measured prompt starts with the warm-up prompt's first word.
        let first = |s: &str| s.split_whitespace().next().unwrap().to_string();
        for prompt in &prompts {
            assert_ne!(
                first(&prompt.messages[0].text_content()),
                first(BENCH_WARMUP_PROMPT)
            );
        }
    }

    #[test]
    fn the_entry_knobs_build_the_load_and_config_with_the_prefix_cache_off_by_default() {
        let none = |_: &str| None;
        let spec = bench_load_spec("/snap/Model-7B", &none).unwrap();
        assert_eq!(spec.source, "/snap/Model-7B");
        assert_eq!(
            spec.prefix_cache_bytes,
            Some(0),
            "a cold prefill by default"
        );
        assert_eq!(
            (spec.quantize, spec.draft_source, spec.mtp_head_source),
            (None, None, None)
        );
        let config = bench_config("/snap/Model-7B", "candle", &none).unwrap();
        assert_eq!(config.model, "Model-7B");
        assert_eq!(config.backend, "candle");
        assert_eq!(config.max_new_tokens, BENCH_DEFAULT_NEW_TOKENS);
        assert_eq!(config.repeats, BENCH_DEFAULT_REPEATS);
        assert_eq!(config.options, [Speculative::Off, Speculative::Auto]);
        assert_eq!(config.sampling, Sampling::greedy());
        assert!(config.warmup);

        let set = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        let knobs = set(&[
            ("SPECULATIVE_BENCH_DRAFT", "/snap/draft"),
            ("SPECULATIVE_BENCH_MTP_HEAD", "/snap/head"),
            ("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES", "1048576"),
            ("SPECULATIVE_BENCH_FORMAT", "q4"),
            ("SPECULATIVE_BENCH_REPEATS", "5"),
            ("SPECULATIVE_BENCH_NEW_TOKENS", "64"),
            ("SPECULATIVE_BENCH_WARMUP", "0"),
            ("SPECULATIVE_BENCH_BACKEND", "candle-cuda"),
            (
                "SPECULATIVE_BENCH_OPTIONS",
                r#"["off", {"proposer": "draft_model", "depth": 3}]"#,
            ),
            ("SPECULATIVE_BENCH_SAMPLING", r#"{"temperature": 0.7}"#),
        ]);
        let spec = bench_load_spec("/snap/Model-7B", &knobs).unwrap();
        assert_eq!(spec.draft_source.as_deref(), Some("/snap/draft"));
        assert_eq!(spec.mtp_head_source.as_deref(), Some("/snap/head"));
        assert_eq!(spec.prefix_cache_bytes, Some(1 << 20));
        assert_eq!(spec.quantize, Some(Quantize::Q4));
        let config = bench_config("/snap/Model-7B", "candle", &knobs).unwrap();
        assert_eq!(
            (config.model.as_str(), config.backend.as_str()),
            ("Model-7B@q4", "candle-cuda")
        );
        assert_eq!((config.repeats, config.max_new_tokens), (5, 64));
        assert!(!config.warmup);
        assert_eq!(
            config.options,
            [
                Speculative::Off,
                Speculative::proposer(SpeculativeProposer::DraftModel, 3)
            ]
        );
        assert_eq!(config.sampling.temperature, 0.7);

        let backend_default = set(&[("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES", "default")]);
        assert_eq!(
            bench_load_spec("m", &backend_default)
                .unwrap()
                .prefix_cache_bytes,
            None
        );
        for bad in [
            set(&[("SPECULATIVE_BENCH_PREFIX_CACHE_BYTES", "1GiB")]),
            set(&[("SPECULATIVE_BENCH_FORMAT", "fp8")]),
        ] {
            assert!(bench_load_spec("m", &bad).is_err());
        }
        for bad in [
            set(&[("SPECULATIVE_BENCH_REPEATS", "three")]),
            set(&[("SPECULATIVE_BENCH_OPTIONS", r#"["sometimes"]"#)]),
        ] {
            assert!(bench_config("m", "candle", &bad).is_err());
        }
        assert_eq!(parse_bench_format("nvfp4").unwrap(), Some(Quantize::Nvfp4));
        assert_eq!(parse_bench_format("q8").unwrap(), Some(Quantize::Q8));
    }

    #[test]
    fn a_bench_sampling_spec_parses_over_greedy_and_refuses_what_it_cannot_read() {
        assert_eq!(
            parse_bench_sampling("\"greedy\"").unwrap(),
            Sampling::greedy()
        );
        let s = parse_bench_sampling(r#"{"temperature": 0.7, "top_p": 0.9, "top_k": 20}"#).unwrap();
        assert_eq!((s.temperature, s.top_p, s.top_k), (0.7, 0.9, 20));
        assert_eq!(s.repetition_penalty, 1.0, "unset knobs stay greedy's");
        for bad in [
            "greedy",
            r#""sample""#,
            r#"{"temp": 0.7}"#,
            r#"{"temperature": "hot"}"#,
            r#"{"top_k": -1}"#,
        ] {
            assert!(parse_bench_sampling(bad).is_err(), "{bad}");
        }
    }
}
