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
//!   [`BenchRow`] per (prompt, option) with decode tok/s, TTFT, the realized mean accepted length
//!   and the report's telemetry, written as one baseline-format JSON document
//!   ([`BenchDocument::write_new`], schema [`BENCH_SCHEMA`]). The pre- and post-epic campaign rows
//!   are this document, so they compare field for field.

use core_llm::{
    DecodeReport, FinishReason, Message, ProposerKind, Sampling, Speculative, StreamEvent, TextLlm,
    TextLlmRequest,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// The benchmark document's schema identifier; bump it when a field changes meaning.
pub const BENCH_SCHEMA: &str = "sceneworks.decode-speedups.baseline/1";

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
    /// Run each (prompt, option) once untimed before the measured run.
    pub warmup: bool,
}

/// One measured (prompt, option) row. Timings come from the provider's own
/// [`GenerationTimings`](core_llm::GenerationTimings) when it reports them (`timing_source =
/// backend`), else from the wall clock around `generate` (`wall`).
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
    /// Tokens generated.
    pub generated_tokens: u32,
    /// Wall time from the `generate` call to the first streamed token.
    pub ttft_ms: Option<f64>,
    /// Prefill time.
    pub prefill_ms: f64,
    /// Decode time.
    pub decode_ms: f64,
    /// `generated_tokens / decode` seconds.
    pub decode_tok_s: Option<f64>,
    /// `backend` or `wall`.
    pub timing_source: &'static str,
    /// The provider's decode report (`None` when it reports none).
    pub report: Option<DecodeReport>,
}

impl BenchRow {
    /// The row as its JSON object.
    pub fn to_json(&self) -> Value {
        let report = self.report.as_ref();
        json!({
            "prompt_id": self.prompt_id,
            "class": self.class.label(),
            "requested": self.requested,
            "proposer": report.map(|r| r.proposer.label()),
            "path": report.map(|r| r.path.clone()),
            "draft_tokens": report.and_then(|r| r.draft_tokens),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": self.ttft_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "decode_tok_s": self.decode_tok_s,
            "timing_source": self.timing_source,
            "target_forwards": report.map(|r| r.target_forwards),
            "proposed_tokens": report.map(|r| r.proposed_tokens),
            "accepted_tokens": report.map(|r| r.accepted_tokens),
            "verify_steps": report.map(|r| r.verify_steps),
            "mean_accepted_length": report.and_then(DecodeReport::mean_accepted_length),
            "sampler": report.map(|r| r.sampler.clone()),
            "kv_cache": report.map(|r| r.kv_cache.clone()),
            "cuda_graphs": report.map(|r| r.cuda_graphs.path.clone()),
            "fallbacks": report.map(|r| r.fallbacks.clone()),
        })
    }
}

/// A finished benchmark run: the configuration and every row.
#[derive(Clone, Debug)]
pub struct BenchDocument {
    /// What was measured.
    pub config: BenchConfig,
    /// One row per (prompt, option), prompt-major.
    pub rows: Vec<BenchRow>,
}

impl BenchDocument {
    /// The baseline-format JSON document ([`BENCH_SCHEMA`]).
    pub fn to_json(&self) -> Value {
        json!({
            "schema": BENCH_SCHEMA,
            "model": self.config.model,
            "backend": self.config.backend,
            "max_new_tokens": self.config.max_new_tokens,
            "sampling": sampling_json(&self.config.sampling),
            "warmup": self.config.warmup,
            "options": self.config.options,
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

/// Run the benchmark: every prompt under every option in `config.options`, under
/// `config.sampling`, one [`BenchRow`] each. Fails on the first request the provider refuses or
/// cannot generate — a benchmark row that silently went missing would read as coverage.
pub fn run_speculative_bench(
    provider: &dyn TextLlm,
    prompts: &[BenchPrompt],
    config: &BenchConfig,
) -> Result<BenchDocument, String> {
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
                observe(provider, &req).map_err(|e| format!("{tag} warm-up: {e}"))?;
            }
            let run = observe(provider, &req).map_err(|e| format!("{tag}: {e}"))?;
            let ttft_ms = run.ttft.map(|d| d.as_secs_f64() * 1e3);
            let (prefill, decode, timing_source) = match run.timings {
                Some(t) => (t.prefill, t.decode, "backend"),
                None => {
                    let ttft = run.ttft.unwrap_or(run.wall);
                    (ttft, run.wall.saturating_sub(ttft), "wall")
                }
            };
            let decode_s = decode.as_secs_f64();
            rows.push(BenchRow {
                prompt_id: prompt.id.clone(),
                class: prompt.class,
                requested: option,
                prompt_tokens: run.prompt_tokens,
                generated_tokens: run.generated,
                ttft_ms,
                prefill_ms: prefill.as_secs_f64() * 1e3,
                decode_ms: decode_s * 1e3,
                decode_tok_s: (decode_s > 0.0).then(|| f64::from(run.generated) / decode_s),
                timing_source,
                report: run.report,
            });
        }
    }
    Ok(BenchDocument {
        config: config.clone(),
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        Channel, ProposerCapabilities, Result as CoreResult, SpeculativeProposer,
        TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, Usage,
    };

    /// A deterministic provider that resolves the speculative option like a backend does and
    /// reports it; `diverge` corrupts the stream whenever a proposer runs, `mislabel` reports the
    /// wrong proposer.
    struct SpeculativeStub {
        descriptor: TextLlmDescriptor,
        diverge: bool,
        mislabel: bool,
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
        }
    }

    impl TextLlm for SpeculativeStub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
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
        // An option that cannot run (an unadvertised proposer decodes plainly, E2) is a
        // failure, never a skipped row.
        let refused = [ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::DraftModel, 2),
            expect_proposer: ProposerKind::DraftModel,
        }];
        let err = check_speculative_greedy_parity(&stub(false, false), &prompts[..1], &refused, 6)
            .unwrap_err();
        assert!(err.contains("expected `draft_model`"), "{err}");
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

    #[test]
    fn the_bench_writes_one_baseline_row_per_prompt_and_option_and_never_overwrites() {
        let config = BenchConfig {
            model: "spec-stub".into(),
            backend: "test".into(),
            max_new_tokens: 5,
            options: vec![
                Speculative::Off,
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
            ],
            warmup: true,
            sampling: Sampling::greedy(),
        };
        let prompts = speculative_prompt_set();
        let doc = run_speculative_bench(&stub(false, false), &prompts, &config).unwrap();
        assert_eq!(doc.rows.len(), 10);
        let json = doc.to_json();
        assert_eq!(json["schema"], BENCH_SCHEMA);
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
        assert_eq!(lookup["generated_tokens"], 5);
        for key in [
            "decode_tok_s",
            "prefill_ms",
            "decode_ms",
            "fallbacks",
            "sampler",
        ] {
            assert!(lookup.get(key).is_some(), "row lacks `{key}`");
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
                ..config
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
