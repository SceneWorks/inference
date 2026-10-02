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
//! * **Provenance** ([`BenchProvenance`]): every document records the checkout it was built from
//!   — the SHA and dirty flag stamped into the binary at compile time ([`BENCH_BUILD_GIT_SHA_ENV`],
//!   [`BENCH_BUILD_GIT_DIRTY_ENV`]) beside `git rev-parse HEAD` / `git status` at run time, a
//!   mismatch refused — the effective state
//!   of every runtime decode switch ([`BENCH_SWITCHES`], read from the backend's switch objects by
//!   the entry point) and the device-selection environment ([`BENCH_ENV`]), and the reasoning
//!   setting the requests carried ([`BenchThinking`], `SPECULATIVE_BENCH_THINKING`).

use core_llm::{
    DecodeReport, FinishReason, LoadReport, LoadSpec, Message, ProposerKind, Quantize,
    ReasoningEffort, Sampling, Speculative, SpeculativeProposer, StreamEvent, TextLlm,
    TextLlmCapabilities, TextLlmRequest, ThinkingMode,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::draft_model::{draft_model_prompts, write_draft_model_fixture, DraftLoader};

/// The benchmark document's schema identifier; bump it when a field changes meaning.
pub const BENCH_SCHEMA: &str = "sceneworks.decode-speedups.baseline/4";

/// The reasoning setting every benchmark request carries (`SPECULATIVE_BENCH_THINKING`): which
/// `enable_thinking` / `reasoning_effort` chat-template kwargs the requests send. A reasoning model
/// decodes a different stream (and, under thinking, a much longer one) per setting, so a row is only
/// comparable with a row of the same setting — the document and every row record it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BenchThinking {
    /// `default`: the template's own default — [`ThinkingMode::Auto`], no `reasoning_effort`, so
    /// neither kwarg is rendered.
    #[default]
    Default,
    /// `off`: [`ThinkingMode::Disabled`] (`enable_thinking=false`).
    Off,
    /// `on`: [`ThinkingMode::Enabled`] (`enable_thinking=true`) at the template's default effort.
    On,
    /// `xhigh` / `medium` / `low`: [`ThinkingMode::Enabled`] with Qwen's `reasoning_effort` at that
    /// level (refused by a provider that does not advertise `supports_reasoning_effort`).
    Effort(ReasoningEffort),
}

impl BenchThinking {
    /// Parse a `SPECULATIVE_BENCH_THINKING` value: `default`, `off`, `on`, or a `reasoning_effort`
    /// level (`xhigh`, `medium`, `low`). Anything else is an error, never the default.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "default" => Ok(Self::Default),
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            level => level
                .parse::<ReasoningEffort>()
                .map(Self::Effort)
                .map_err(|_| {
                    format!(
                    "SPECULATIVE_BENCH_THINKING must be default, off, on, xhigh, medium or low, \
                     got {value}"
                )
                }),
        }
    }

    /// The setting's spelling (`default`, `off`, `on`, `xhigh`, `medium`, `low`).
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Off => "off",
            Self::On => "on",
            Self::Effort(effort) => effort.as_str(),
        }
    }

    /// The request's [`TextLlmRequest::thinking`].
    pub fn mode(self) -> ThinkingMode {
        match self {
            Self::Default => ThinkingMode::Auto,
            Self::Off => ThinkingMode::Disabled,
            Self::On | Self::Effort(_) => ThinkingMode::Enabled,
        }
    }

    /// The request's [`TextLlmRequest::reasoning_effort`].
    pub fn reasoning_effort(self) -> Option<ReasoningEffort> {
        match self {
            Self::Effort(effort) => Some(effort),
            _ => None,
        }
    }

    /// Set `req`'s reasoning controls to this setting.
    pub fn apply(self, req: &mut TextLlmRequest) {
        req.thinking = self.mode();
        req.reasoning_effort = self.reasoning_effort();
    }

    /// The document's `thinking` block: the setting, the kwargs it renders, and whether `caps`
    /// (the loaded provider's) honor them — a setting the model does not support is either
    /// refused (`on`, an effort level) or rendered to no effect (`off`), and the reader needs to
    /// know which run that was.
    pub fn to_json(self, caps: &TextLlmCapabilities) -> Value {
        json!({
            "setting": self.label(),
            "enable_thinking": self.mode().enable_thinking_kwarg(),
            "reasoning_effort": self.reasoning_effort().map(ReasoningEffort::as_str),
            "supports_thinking": caps.supports_thinking,
            "supports_reasoning_effort": caps.supports_reasoning_effort,
        })
    }
}

/// The runtime decode switches every document records (`provenance.switches`), by their
/// environment variable — Candle's and MLX's alike, so documents from both backends (and the
/// pre-epic baseline driver's) have one key set. A backend's entry point reports the **effective**
/// state of its own switches from the switch objects ([`BenchSwitches`]); a switch the backend
/// (or revision) does not have is recorded with `effective: null`.
pub const BENCH_SWITCHES: [&str; 9] = [
    "CANDLE_LLM_CUDA_GRAPHS",
    "CANDLE_LLM_CUDA_STREAM",
    "CANDLE_LLM_DEVICE_POSITIONS",
    "CANDLE_LLM_FUSED_KERNELS",
    "CANDLE_LLM_NVFP4_GEMV",
    "MLX_LLM_PIPELINING",
    "MLX_LLM_DEVICE_SAMPLER",
    "MLX_LLM_FUSED_ROTATION",
    "MLX_LLM_GDN_KERNEL",
];

/// The device-selection variables every document records verbatim (`provenance.env`): which GPU
/// the process saw and whether Candle was forced onto the CPU.
pub const BENCH_ENV: [&str; 2] = ["CUDA_VISIBLE_DEVICES", "CANDLE_LLM_DEVICE"];

/// The operator's SHA override (`source` `SPECULATIVE_BENCH_GIT_SHA`): accepted only when the
/// binary carries no compile-time SHA ([`BENCH_BUILD_GIT_SHA_ENV`]) **and**
/// [`BENCH_ALLOW_SHA_OVERRIDE_ENV`] is `1` (the campaign script's `--allow-sha-override`); then it
/// is the recorded SHA when `git rev-parse HEAD` cannot answer, and must equal `HEAD` when it can.
/// Set in any other case, the run is refused ([`reconcile_git_provenance`]).
pub const BENCH_GIT_SHA_ENV: &str = "SPECULATIVE_BENCH_GIT_SHA";

/// `1` permits [`BENCH_GIT_SHA_ENV`] (see there).
pub const BENCH_ALLOW_SHA_OVERRIDE_ENV: &str = "SPECULATIVE_BENCH_ALLOW_SHA_OVERRIDE";

/// The **compile-time** variable carrying the checkout's 40-hex `HEAD` into the benchmark binary
/// (`option_env!`, so cargo rebuilds the crate when it changes): the campaign script
/// (`scripts/release/speculative_bench_campaign.py`) sets it, after checking `HEAD`, for every
/// build and run. A binary built without it records `build_sha: null`, which the campaign refuses.
pub const BENCH_BUILD_GIT_SHA_ENV: &str = "SPECULATIVE_BENCH_BUILD_GIT_SHA";

/// The compile-time dirty flag (`0` / `1`, `git status --porcelain` non-empty at build time) set
/// beside [`BENCH_BUILD_GIT_SHA_ENV`].
pub const BENCH_BUILD_GIT_DIRTY_ENV: &str = "SPECULATIVE_BENCH_BUILD_GIT_DIRTY";

/// What this binary was compiled from: [`BENCH_BUILD_GIT_SHA_ENV`] and
/// [`BENCH_BUILD_GIT_DIRTY_ENV`] as `option_env!` saw them (each `None` when unset).
pub fn bench_build_stamp() -> (Option<&'static str>, Option<&'static str>) {
    (
        option_env!("SPECULATIVE_BENCH_BUILD_GIT_SHA"),
        option_env!("SPECULATIVE_BENCH_BUILD_GIT_DIRTY"),
    )
}

/// What an entry point reports about its backend's runtime switches: `(variable, effective state)`
/// for each switch the backend has, read from the switch objects (a bool, or a label for a
/// non-boolean switch such as the CUDA stream). Every variable must be one of [`BENCH_SWITCHES`].
pub type BenchSwitches<'a> = &'a dyn Fn() -> Vec<(&'static str, Value)>;

/// The checkout a benchmark binary was built from, and the runtime switches it ran under.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchProvenance {
    /// `HEAD` of the checkout at run time (40 hex), or [`BENCH_GIT_SHA_ENV`]'s value; `None` when
    /// neither answered.
    pub git_sha: Option<String>,
    /// The SHA stamped at compile time ([`BENCH_BUILD_GIT_SHA_ENV`]); `None` when the build set
    /// none.
    pub build_sha: Option<String>,
    /// The dirty flag stamped at compile time ([`BENCH_BUILD_GIT_DIRTY_ENV`]).
    pub build_dirty: Option<bool>,
    /// Whether the checkout differed from `HEAD` (`git status --porcelain`, untracked files
    /// included); `None` when `git` did not answer.
    pub git_dirty: Option<bool>,
    /// The `git status --porcelain` lines, at most [`BenchProvenance::MAX_CHANGES`].
    pub git_changes: Vec<String>,
    /// `git`, [`BENCH_GIT_SHA_ENV`], or `None`.
    pub git_source: Option<&'static str>,
    /// The backend's effective switch states ([`BenchSwitches`]).
    pub switches: Vec<(&'static str, Value)>,
    /// Each [`BENCH_SWITCHES`] and [`BENCH_ENV`] variable's raw value at collection (`None` when
    /// unset).
    pub env: Vec<(&'static str, Option<String>)>,
}

impl BenchProvenance {
    /// The most `git status --porcelain` lines a document records.
    pub const MAX_CHANGES: usize = 50;

    /// The provenance of this process: the checkout this crate was compiled from (its manifest
    /// directory — the benchmark binary is built from the same checkout) reconciled with the
    /// compile-time stamp ([`bench_build_stamp`], [`reconcile_git_provenance`]), the raw
    /// environment read through `var`, and `switches` — refused when it names a variable outside
    /// [`BENCH_SWITCHES`] or names one twice, so a renamed switch cannot drop out of the record,
    /// and when the binary's compile-time SHA or dirty flag is not the checkout's now.
    pub fn collect(
        switches: Vec<(&'static str, Value)>,
        var: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        for (i, (name, _)) in switches.iter().enumerate() {
            if !BENCH_SWITCHES.contains(name) {
                return Err(format!(
                    "switch {name} is not one of the recorded BENCH_SWITCHES"
                ));
            }
            if switches[..i].iter().any(|(other, _)| other == name) {
                return Err(format!("switch {name} is reported twice"));
            }
        }
        let mut provenance = reconcile_git_provenance(
            git_provenance(Path::new(env!("CARGO_MANIFEST_DIR")), var),
            bench_build_stamp(),
            var,
        )?;
        provenance.switches = switches;
        provenance.env = BENCH_SWITCHES
            .iter()
            .chain(&BENCH_ENV)
            .map(|&name| (name, var(name)))
            .collect();
        Ok(provenance)
    }

    /// The document's `provenance` block: `git` (`{"sha", "dirty", "changes", "source",
    /// "build_sha", "build_dirty"}` — run time, then compile time),
    /// `switches` (one `{"env", "effective"}` per [`BENCH_SWITCHES`] variable: the raw value and
    /// the backend's effective state, each `null` when absent) and `env` (each [`BENCH_ENV`]
    /// variable's raw value).
    pub fn to_json(&self) -> Value {
        let raw = |name: &str| {
            self.env
                .iter()
                .find(|(n, _)| *n == name)
                .and_then(|(_, v)| v.clone())
        };
        let switches: serde_json::Map<String, Value> = BENCH_SWITCHES
            .iter()
            .map(|&name| {
                let effective = self
                    .switches
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map_or(Value::Null, |(_, v)| v.clone());
                (
                    name.to_string(),
                    json!({"env": raw(name), "effective": effective}),
                )
            })
            .collect();
        let env: serde_json::Map<String, Value> = BENCH_ENV
            .iter()
            .map(|&name| (name.to_string(), json!(raw(name))))
            .collect();
        json!({
            "git": {
                "sha": self.git_sha,
                "dirty": self.git_dirty,
                "changes": self.git_changes,
                "source": self.git_source,
                "build_sha": self.build_sha,
                "build_dirty": self.build_dirty,
            },
            "switches": switches,
            "env": env,
        })
    }
}

/// The git state of the checkout at `repo`: `git rev-parse HEAD` and `git status --porcelain`,
/// run now; when `git` cannot answer, the SHA from [`BENCH_GIT_SHA_ENV`] (read through `var`)
/// with the tree state unknown; else nothing. [`reconcile_git_provenance`] decides whether the
/// operator's SHA may stand.
pub fn git_provenance(repo: &Path, var: &dyn Fn(&str) -> Option<String>) -> BenchProvenance {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8(out.stdout).ok())
    };
    let sha = git(&["rev-parse", "HEAD"])
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()));
    if let Some(sha) = sha {
        let status = git(&["status", "--porcelain"]);
        return BenchProvenance {
            git_sha: Some(sha),
            git_dirty: status.as_ref().map(|s| !s.trim().is_empty()),
            git_changes: status
                .iter()
                .flat_map(|s| s.lines())
                .filter(|l| !l.trim().is_empty())
                .take(BenchProvenance::MAX_CHANGES)
                .map(str::to_string)
                .collect(),
            git_source: Some("git"),
            ..BenchProvenance::default()
        };
    }
    match var(BENCH_GIT_SHA_ENV) {
        Some(sha) => BenchProvenance {
            git_sha: Some(sha.trim().to_string()),
            git_source: Some(BENCH_GIT_SHA_ENV),
            ..BenchProvenance::default()
        },
        None => BenchProvenance::default(),
    }
}

/// Reconcile the run-time git state (`runtime`, [`git_provenance`]) with the compile-time stamp
/// `build` (`(sha, dirty)`, [`bench_build_stamp`]) and record the stamp. Refused:
///
/// * a stamped SHA that is not 40 hex, or a dirty flag that is not `0` / `1`;
/// * a stamped SHA (or dirty flag) that differs from what `git` reports for the checkout now —
///   the binary was not built from this tree (a stale test binary, a moved checkout);
/// * [`BENCH_GIT_SHA_ENV`] set while the binary carries a stamped SHA, or without
///   [`BENCH_ALLOW_SHA_OVERRIDE_ENV`] `=1`, or naming a SHA other than the one `git` reports.
pub fn reconcile_git_provenance(
    mut runtime: BenchProvenance,
    build: (Option<&str>, Option<&str>),
    var: &dyn Fn(&str) -> Option<String>,
) -> Result<BenchProvenance, String> {
    let (build_sha, build_dirty) = build;
    let build_sha = build_sha.map(str::trim).filter(|s| !s.is_empty());
    if let Some(sha) = build_sha {
        if sha.len() != 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!(
                "{BENCH_BUILD_GIT_SHA_ENV} is {sha:?}, not a 40-hex commit"
            ));
        }
    }
    let build_dirty = match build_dirty.map(str::trim).filter(|s| !s.is_empty()) {
        None => None,
        Some("0") => Some(false),
        Some("1") => Some(true),
        Some(other) => {
            return Err(format!(
                "{BENCH_BUILD_GIT_DIRTY_ENV} is {other:?}, not 0 or 1"
            ));
        }
    };
    let from_git = runtime.git_source == Some("git");
    if runtime.git_source == Some(BENCH_GIT_SHA_ENV) || var(BENCH_GIT_SHA_ENV).is_some() {
        let given = var(BENCH_GIT_SHA_ENV).unwrap_or_default();
        if build_sha.is_some() {
            return Err(format!(
                "{BENCH_GIT_SHA_ENV}={given} refused: this binary carries its compile-time SHA"
            ));
        }
        if var(BENCH_ALLOW_SHA_OVERRIDE_ENV).as_deref() != Some("1") {
            return Err(format!(
                "{BENCH_GIT_SHA_ENV}={given} refused without {BENCH_ALLOW_SHA_OVERRIDE_ENV}=1"
            ));
        }
        if from_git && runtime.git_sha.as_deref() != Some(given.trim()) {
            return Err(format!(
                "{BENCH_GIT_SHA_ENV}={given} is not the checkout's HEAD {:?}",
                runtime.git_sha
            ));
        }
    }
    if from_git {
        if let Some(sha) = build_sha {
            if runtime.git_sha.as_deref() != Some(sha) {
                return Err(format!(
                    "this binary was compiled from {sha} but the checkout is at {:?}: rebuild it",
                    runtime.git_sha
                ));
            }
        }
        if let (Some(built), Some(now)) = (build_dirty, runtime.git_dirty) {
            if built != now {
                return Err(format!(
                    "this binary was compiled from a {} tree but the checkout is {} now: rebuild it",
                    if built { "dirty" } else { "clean" },
                    if now { "dirty" } else { "clean" },
                ));
            }
        }
    }
    runtime.build_sha = build_sha.map(str::to_string);
    runtime.build_dirty = build_dirty;
    Ok(runtime)
}

/// `SPECULATIVE_BENCH_THINKING` read through `var` ([`BenchThinking::parse`]); unset is
/// [`BenchThinking::Default`].
pub fn bench_thinking(var: &dyn Fn(&str) -> Option<String>) -> Result<BenchThinking, String> {
    var("SPECULATIVE_BENCH_THINKING").map_or(Ok(BenchThinking::Default), |v| {
        BenchThinking::parse(v.trim())
    })
}

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

/// The parity rows for every proposer `caps` advertises (epic sc-24432 AT1): depths 1, 3 (at most
/// the maximum) and the advertised maximum — read from the capabilities, never assumed — each
/// expected to run that proposer.
pub fn advertised_parity_cases(caps: &TextLlmCapabilities) -> Vec<ParityCase> {
    SpeculativeProposer::ALL
        .into_iter()
        .filter_map(|proposer| caps.proposer(proposer))
        .flat_map(|advertised| {
            let mut depths = vec![1, 3.min(advertised.max_depth), advertised.max_depth];
            depths.dedup();
            depths.into_iter().map(move |depth| ParityCase {
                speculative: Speculative::proposer(advertised.proposer, depth),
                expect_proposer: advertised.proposer.into(),
            })
        })
        .collect()
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
    /// The reasoning setting every request (warm-ups included) carries.
    pub thinking: BenchThinking,
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
/// | `thinking` | the reasoning setting the request carried ([`BenchThinking::label`]) |
/// | `proposer`, `path`, `draft_tokens` | [`DecodeReport`]'s proposer label, decode path and depth |
/// | `prompt_tokens`, `generated_tokens` | token counts |
/// | `repeats` | measured repeats (the length of `samples`) |
/// | `ttft_ms`, `decode_tok_s`, `prefill_ms`, `decode_ms` | [`BenchStats`] over the repeats: `{"n", "mean", "min", "max", "stddev"}` (`stddev` `null` for one repeat; the whole object `null` when no repeat produced the value) |
/// | `timing_source` | `backend` or `wall` |
/// | `target_forwards`, `prefill_forwards`, `verify_steps`, `replay_forwards`, `discarded_forwards` | [`DecodeReport`] forward counts |
/// | `proposed_tokens`, `accepted_tokens`, `mean_accepted_length` | draft accounting ([`DecodeReport::mean_accepted_length`]) |
/// | `speculative_demoted_at` | tokens generated when `auto` demoted the proposer ([`DecodeReport::speculative_demoted_at`]); `null` when it was not |
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
    /// The reasoning setting the request carried.
    pub thinking: BenchThinking,
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
            "thinking": self.thinking.label(),
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
            "discarded_forwards": report.map(|r| r.discarded_forwards),
            "speculative_demoted_at": report.and_then(|r| r.speculative_demoted_at),
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
    /// The provider's capabilities (the `thinking` block records its reasoning support).
    pub capabilities: TextLlmCapabilities,
    /// The checkout and runtime switches the run measured ([`BenchProvenance`]).
    pub provenance: BenchProvenance,
    /// One row per (prompt, option), prompt-major.
    pub rows: Vec<BenchRow>,
}

impl BenchDocument {
    /// The baseline-format JSON document ([`BENCH_SCHEMA`]): `schema`, `model`, `backend`,
    /// `max_new_tokens`, `sampling` ([`sampling_json`]), `thinking` ([`BenchThinking::to_json`]),
    /// `warmup`, `repeats`, `options`, `load` (`{"prefix_cache_bytes", "draft": {"source",
    /// "refusal"} | null, "cuda_graphs", "fallbacks"}` from the load report — the settled
    /// prefix-cache budget and what became of a named draft / MTP head — or `null`), `provenance`
    /// ([`BenchProvenance::to_json`]), and `rows` ([`BenchRow`]'s schema).
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
            "thinking": self.config.thinking.to_json(&self.capabilities),
            "warmup": self.config.warmup,
            "repeats": self.config.repeats,
            "options": self.config.options,
            "load": load,
            "provenance": self.provenance.to_json(),
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
/// `config.sampling` and `config.thinking` — an untimed warm-up (when `config.warmup`;
/// [`BENCH_WARMUP_PROMPT`] says which request) and then `config.repeats` measured runs — one
/// [`BenchRow`] each. Fails on the first request the provider refuses or cannot generate — a
/// benchmark row that silently went missing would read as coverage. The document's provenance
/// records the checkout and environment with no backend switches; an entry point replaces it with
/// its backend's ([`run_speculative_bench_from_env`]).
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
    let request = |prompt: &BenchPrompt, option: Speculative| {
        let mut req = bench_request(prompt, option, &config.sampling, config.max_new_tokens);
        config.thinking.apply(&mut req);
        req
    };
    let mut rows = Vec::with_capacity(prompts.len() * config.options.len());
    for prompt in prompts {
        for &option in &config.options {
            let req = request(prompt, option);
            let tag = format!(
                "[{}] {}",
                prompt.id,
                serde_json::to_string(&option).unwrap_or_default()
            );
            if config.warmup {
                let warm = if isolated_warmup {
                    request(&warmup_prompt, option)
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
                thinking: config.thinking,
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
        capabilities: provider.descriptor().capabilities.clone(),
        provenance: BenchProvenance::collect(Vec::new(), &bench_env)?,
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
        thinking: bench_thinking(var)?,
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
/// | `SPECULATIVE_BENCH_THINKING` | reasoning setting of every request ([`BenchThinking::parse`]): `default` (the template's; the default), `off`, `on`, or a `reasoning_effort` level `xhigh` / `medium` / `low` (thinking on at that effort) |
/// | `SPECULATIVE_BENCH_GIT_SHA` | the operator's SHA, only with `SPECULATIVE_BENCH_ALLOW_SHA_OVERRIDE=1` and no compile-time SHA ([`BENCH_GIT_SHA_ENV`]) |
///
/// At **compile time** `SPECULATIVE_BENCH_BUILD_GIT_SHA` / `SPECULATIVE_BENCH_BUILD_GIT_DIRTY`
/// stamp the binary ([`bench_build_stamp`]). The document's `provenance` ([`BenchProvenance`])
/// records the checkout's `HEAD` and tree state now and the stamp, the effective state of
/// `switches` (the backend's runtime switches, read after the run) and the raw
/// [`BENCH_SWITCHES`] / [`BENCH_ENV`] variables. The provenance is collected once before the load
/// too, so a binary that is not the checkout's ([`reconcile_git_provenance`]) is refused before it
/// measures anything.
pub fn run_speculative_bench_from_env(
    default_backend: &str,
    load: DraftLoader<'_>,
    switches: BenchSwitches<'_>,
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
    BenchProvenance::collect(Vec::new(), &bench_env)?;
    let provider = load(&spec).map_err(|e| format!("load {snapshot}: {e}"))?;
    let mut doc = run_speculative_bench(provider.as_ref(), &speculative_prompt_set(), &config)?;
    doc.provenance = BenchProvenance::collect(switches(), &bench_env)?;
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
///   larger hit;
/// * the document records the `default` reasoning setting on every row, and `switches` — the
///   entry point's switch reader — reports at least one switch, each a recorded
///   [`BENCH_SWITCHES`] variable with a non-null effective state, which the provenance block
///   carries; a request for thinking the fixture does not advertise is refused, never run as the
///   default.
pub fn check_speculative_bench_on_fixture(
    root: &Path,
    backend: &str,
    load: DraftLoader<'_>,
    switches: BenchSwitches<'_>,
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
        thinking: bench_thinking(&|_| None)?,
    };
    let mut failures = Vec::new();

    let cold = load(&bench_load_spec(&target, &|_| None)?)?;
    let mut doc = run_speculative_bench(cold.as_ref(), &prompts, &config)?;
    let reported = switches();
    if reported.is_empty() || reported.iter().any(|(_, v)| v.is_null()) {
        failures.push(format!("the switch reader reported {reported:?}"));
    }
    doc.provenance = BenchProvenance::collect(reported.clone(), &bench_env)?;
    let json = doc.to_json();
    if json["load"]["prefix_cache_bytes"] != json!(0) {
        failures.push(format!("the default load settled {}", json["load"]));
    }
    if json["thinking"]["setting"] != "default"
        || json["thinking"]["enable_thinking"] != Value::Null
    {
        failures.push(format!(
            "the default reasoning setting is {}",
            json["thinking"]
        ));
    }
    let recorded = json["provenance"]["switches"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    if recorded.len() != BENCH_SWITCHES.len() {
        failures.push(format!(
            "the provenance records {} switches",
            recorded.len()
        ));
    }
    for (name, value) in &reported {
        if recorded.get(*name).map(|s| &s["effective"]) != Some(value) {
            failures.push(format!("the provenance lost switch {name}: {recorded:?}"));
        }
    }
    let refused = run_speculative_bench(
        cold.as_ref(),
        &prompts[..1],
        &BenchConfig {
            thinking: BenchThinking::On,
            ..config.clone()
        },
    );
    if !cold.descriptor().capabilities.supports_thinking && refused.is_ok() {
        failures.push("thinking `on` ran on a provider without a thinking mode".into());
    }
    let rows = json["rows"].as_array().cloned().unwrap_or_default();
    if rows.len() != prompts.len() * config.options.len() {
        failures.push(format!("{} rows", rows.len()));
    }
    for row in &rows {
        let tag = format!("[{}] {}", row["prompt_id"], row["requested"]);
        if row["thinking"] != "default" {
            failures.push(format!("{tag}: thinking {}", row["thinking"]));
        }
        for key in [
            "graph_path",
            "attention",
            "fused",
            "prefix_cache",
            "target_forwards",
            "prefill_forwards",
            "verify_steps",
            "replay_forwards",
            "discarded_forwards",
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
        /// Each request's reasoning controls, in request order.
        seen_thinking: RefCell<Vec<(ThinkingMode, Option<ReasoningEffort>)>>,
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
            seen_thinking: RefCell::default(),
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
            self.seen_thinking
                .borrow_mut()
                .push((req.thinking, req.reasoning_effort));
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

    /// An object's keys, sorted (independent of `serde_json`'s map ordering feature).
    fn sorted_keys(value: &Value) -> Vec<&str> {
        let mut keys: Vec<_> = value
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, _)| k.as_str())
            .collect();
        keys.sort_unstable();
        keys
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
            thinking: BenchThinking::Default,
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
        assert_eq!(
            json["thinking"],
            json!({"setting": "default", "enable_thinking": null, "reasoning_effort": null,
                   "supports_thinking": false, "supports_reasoning_effort": false})
        );
        assert_eq!(
            sorted_keys(&json),
            [
                "backend",
                "load",
                "max_new_tokens",
                "model",
                "options",
                "provenance",
                "repeats",
                "rows",
                "sampling",
                "schema",
                "thinking",
                "warmup"
            ]
        );
        let switches = json["provenance"]["switches"].as_object().unwrap();
        assert_eq!(sorted_keys(&json["provenance"]["switches"]), {
            let mut names = BENCH_SWITCHES.to_vec();
            names.sort_unstable();
            names
        });
        // `run_speculative_bench` alone knows no backend: every effective state is unreported.
        assert!(switches.values().all(|s| s["effective"].is_null()));
        assert_eq!(
            sorted_keys(&json["provenance"]["env"]),
            ["CANDLE_LLM_DEVICE", "CUDA_VISIBLE_DEVICES"]
        );
        let off = &json["rows"][0];
        assert_eq!(off["thinking"], "default");
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
            "discarded_forwards",
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

    #[test]
    fn a_thinking_setting_parses_to_its_request_controls_and_refuses_anything_else() {
        let cases = [
            ("default", ThinkingMode::Auto, None, None),
            ("off", ThinkingMode::Disabled, None, Some(false)),
            ("on", ThinkingMode::Enabled, None, Some(true)),
            (
                "xhigh",
                ThinkingMode::Enabled,
                Some(ReasoningEffort::XHigh),
                Some(true),
            ),
            (
                "medium",
                ThinkingMode::Enabled,
                Some(ReasoningEffort::Medium),
                Some(true),
            ),
            (
                "low",
                ThinkingMode::Enabled,
                Some(ReasoningEffort::Low),
                Some(true),
            ),
        ];
        for (spelling, mode, effort, kwarg) in cases {
            let thinking = BenchThinking::parse(spelling).unwrap();
            assert_eq!(thinking.label(), spelling);
            assert_eq!(
                (thinking.mode(), thinking.reasoning_effort()),
                (mode, effort)
            );
            let mut req = TextLlmRequest::default();
            thinking.apply(&mut req);
            assert_eq!((req.thinking, req.reasoning_effort), (mode, effort));
            let json = thinking.to_json(&TextLlmCapabilities::default());
            assert_eq!(json["setting"], spelling);
            assert_eq!(json["enable_thinking"], json!(kwarg));
            assert_eq!(
                json["reasoning_effort"],
                json!(effort.map(ReasoningEffort::as_str))
            );
        }
        for bad in ["", "ON", "true", "high", "auto", "think"] {
            let err = BenchThinking::parse(bad).unwrap_err();
            assert!(err.contains("SPECULATIVE_BENCH_THINKING"), "{bad}: {err}");
        }
        let env = |value: &'static str| {
            move |name: &str| (name == "SPECULATIVE_BENCH_THINKING").then(|| value.to_string())
        };
        assert_eq!(bench_thinking(&|_| None).unwrap(), BenchThinking::Default);
        assert_eq!(
            bench_config("m", "candle", &env("low")).unwrap().thinking,
            BenchThinking::Effort(ReasoningEffort::Low)
        );
        assert!(bench_config("m", "candle", &env("maximum")).is_err());
    }

    /// Every request the bench sends — warm-up and measured — carries the configured reasoning
    /// setting, every row records it, and a provider that cannot honor it refuses the run.
    #[test]
    fn the_bench_sends_and_records_the_thinking_setting_and_a_refusal_fails_the_run() {
        let prompts = speculative_prompt_set();
        let mut provider = stub(false, false);
        provider.descriptor.capabilities.supports_thinking = true;
        provider.descriptor.capabilities.supports_reasoning_effort = true;
        let config = BenchConfig {
            options: vec![Speculative::Off],
            repeats: 2,
            thinking: BenchThinking::Effort(ReasoningEffort::Medium),
            ..stub_config()
        };
        let json = run_speculative_bench(&provider, &prompts[..2], &config)
            .unwrap()
            .to_json();
        assert_eq!(
            *provider.seen_thinking.borrow(),
            [(ThinkingMode::Enabled, Some(ReasoningEffort::Medium)); 6],
            "two warm-ups and four measured repeats"
        );
        assert_eq!(
            json["thinking"],
            json!({"setting": "medium", "enable_thinking": true, "reasoning_effort": "medium",
                   "supports_thinking": true, "supports_reasoning_effort": true})
        );
        for row in json["rows"].as_array().unwrap() {
            assert_eq!(row["thinking"], "medium", "{row}");
        }

        // No thinking mode at all; a thinking mode without Qwen's effort control.
        let mut thinking_only = stub(false, false);
        thinking_only.descriptor.capabilities.supports_thinking = true;
        for (provider, thinking, needle) in [
            (&stub(false, false), BenchThinking::On, "thinking"),
            (
                &thinking_only,
                BenchThinking::Effort(ReasoningEffort::Low),
                "reasoning_effort",
            ),
        ] {
            let err = run_speculative_bench(
                provider,
                &prompts[..1],
                &BenchConfig {
                    thinking,
                    ..config.clone()
                },
            )
            .unwrap_err();
            assert!(err.contains(needle), "{thinking:?}: {err}");
        }
    }

    #[test]
    fn the_provenance_records_every_switch_and_refuses_an_unrecorded_one() {
        let var = |name: &str| match name {
            "CUDA_VISIBLE_DEVICES" => Some("1".to_string()),
            "CANDLE_LLM_CUDA_GRAPHS" => Some("0".to_string()),
            _ => None,
        };
        let provenance = BenchProvenance::collect(
            vec![
                ("CANDLE_LLM_CUDA_GRAPHS", json!(false)),
                ("CANDLE_LLM_CUDA_STREAM", json!("legacy")),
            ],
            &var,
        )
        .unwrap();
        let json = provenance.to_json();
        assert_eq!(
            json["switches"]["CANDLE_LLM_CUDA_GRAPHS"],
            json!({"env": "0", "effective": false})
        );
        assert_eq!(
            json["switches"]["CANDLE_LLM_CUDA_STREAM"],
            json!({"env": null, "effective": "legacy"})
        );
        assert_eq!(
            json["switches"]["MLX_LLM_PIPELINING"],
            json!({"env": null, "effective": null})
        );
        assert_eq!(
            json["env"],
            json!({"CUDA_VISIBLE_DEVICES": "1", "CANDLE_LLM_DEVICE": null})
        );
        assert_eq!(
            sorted_keys(&json["git"]),
            [
                "build_dirty",
                "build_sha",
                "changes",
                "dirty",
                "sha",
                "source"
            ]
        );
        // This crate is compiled from a checkout: `git` answers with the 40-hex `HEAD` when it is
        // installed (the fallback below covers a host without it).
        if provenance.git_source == Some("git") {
            let sha = provenance.git_sha.as_deref().unwrap();
            assert!(sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(provenance.git_dirty.is_some());
        }

        for bad in [
            vec![("CANDLE_LLM_CUDA_GRAPH", json!(true))],
            vec![
                ("MLX_LLM_PIPELINING", json!(true)),
                ("MLX_LLM_PIPELINING", json!(false)),
            ],
        ] {
            assert!(BenchProvenance::collect(bad, &var).is_err());
        }
    }

    #[test]
    fn git_provenance_falls_back_to_the_operator_sha_outside_a_checkout() {
        let outside = tempfile::tempdir().unwrap();
        let none = git_provenance(outside.path(), &|_| None);
        assert_eq!(none, BenchProvenance::default());
        let sha = "c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6";
        let given = git_provenance(outside.path(), &|name: &str| {
            (name == BENCH_GIT_SHA_ENV).then(|| sha.to_string())
        });
        assert_eq!(given.git_sha.as_deref(), Some(sha));
        assert_eq!(given.git_source, Some(BENCH_GIT_SHA_ENV));
        assert_eq!(
            given.git_dirty, None,
            "the operator's SHA says nothing of the tree"
        );
    }

    /// The compile-time stamp and the run-time checkout must agree, and the operator's SHA stands
    /// only without a stamp and with the explicit permission.
    #[test]
    fn the_build_stamp_must_be_the_checkout_and_the_operator_sha_needs_permission() {
        let head = "c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6";
        let other = "99f0717948b35b28b3cf8b8491c3297121c6508c";
        let checkout = |dirty: bool| BenchProvenance {
            git_sha: Some(head.into()),
            git_dirty: Some(dirty),
            git_source: Some("git"),
            ..BenchProvenance::default()
        };
        let none = |_: &str| None;
        let ok = reconcile_git_provenance(checkout(false), (Some(head), Some("0")), &none).unwrap();
        assert_eq!(
            (ok.build_sha.as_deref(), ok.build_dirty),
            (Some(head), Some(false))
        );
        assert_eq!(ok.to_json()["git"]["build_sha"], head);
        // A stale binary (built at another commit, or from a tree that has since changed).
        for (built, dirty, now_dirty) in [
            (other, "0", false),
            (head, "0", true),
            (head, "1", false),
            ("not-a-sha", "0", false),
            (head, "yes", false),
        ] {
            let err =
                reconcile_git_provenance(checkout(now_dirty), (Some(built), Some(dirty)), &none);
            assert!(err.is_err(), "{built} {dirty} {now_dirty}: {err:?}");
        }
        // No stamp: the run-time state stands, recorded as unstamped.
        let bare = reconcile_git_provenance(checkout(true), (None, None), &none).unwrap();
        assert_eq!((bare.build_sha, bare.build_dirty), (None, None));

        // The operator's SHA.
        let outside = BenchProvenance {
            git_sha: Some(head.into()),
            git_source: Some(BENCH_GIT_SHA_ENV),
            ..BenchProvenance::default()
        };
        let allowed = |name: &str| match name {
            BENCH_GIT_SHA_ENV => Some(head.to_string()),
            BENCH_ALLOW_SHA_OVERRIDE_ENV => Some("1".to_string()),
            _ => None,
        };
        let unpermitted = |name: &str| (name == BENCH_GIT_SHA_ENV).then(|| head.to_string());
        let accepted = reconcile_git_provenance(outside.clone(), (None, None), &allowed).unwrap();
        assert_eq!(accepted.git_source, Some(BENCH_GIT_SHA_ENV));
        assert!(reconcile_git_provenance(outside.clone(), (None, None), &unpermitted).is_err());
        assert!(
            reconcile_git_provenance(outside, (Some(head), None), &allowed).is_err(),
            "a stamped binary never takes the operator's SHA"
        );
        // Set beside a checkout that answers, it must name that checkout's HEAD.
        assert!(reconcile_git_provenance(checkout(false), (None, None), &allowed).is_ok());
        let wrong = |name: &str| match name {
            BENCH_GIT_SHA_ENV => Some(other.to_string()),
            BENCH_ALLOW_SHA_OVERRIDE_ENV => Some("1".to_string()),
            _ => None,
        };
        assert!(reconcile_git_provenance(checkout(false), (None, None), &wrong).is_err());
        assert!(reconcile_git_provenance(checkout(false), (None, None), &unpermitted).is_err());
    }
}
