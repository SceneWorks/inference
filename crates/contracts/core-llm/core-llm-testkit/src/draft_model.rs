//! The draft-model speculative checks (epic sc-24432, story sc-24436) — backend-neutral, driven
//! purely through [`TextLlm`] over one shared on-disk fixture, so the MLX and Candle providers load
//! the same snapshots and pass the same checks (E8).
//!
//! * **Fixture** ([`write_draft_model_fixture`]): a tiny Qwen3-architecture **target** and a
//!   smaller tiny Qwen3 **draft** sharing one tokenizer vocabulary, plus a **foreign** draft whose
//!   tokenizer is the same size over different tokens. Target and draft share their embedding and
//!   output projection and differ in depth, width and layer weights, so the draft agrees with the
//!   target often but not always: a greedy run accepts drafts *and* rejects some. Written as plain
//!   safetensors from this crate, so the fixture itself needs no tensor library. No real weights.
//! * **Resident** ([`check_draft_model_resident`]): a target loaded with the draft advertises
//!   `draft_model`, its load report names the draft resident, `{proposer: draft_model}` at every
//!   depth emits exactly the greedy stream of `off` with a report naming `draft_model`, drafts
//!   were proposed, accepted and rejected, and a seeded stochastic run is reproducible.
//! * **Refused** ([`check_draft_model_refused`]): a target loaded with the foreign draft still
//!   loads, its load report names the tokenizer refusal, `draft_model` is not advertised (so a
//!   request for it is refused up front), and the target decodes as it would alone.
//! * **Stop token** ([`check_draft_model_stop_token`]): the fixture's target with a real stop
//!   token its greedy stream reaches — drafts propose it and the target accepts it — ends every
//!   `draft_model` run (greedy, penalized, near-zero temperature, every depth) at that stop, the
//!   greedy decisions exactly where `off` does.
//! * **Short draft context** ([`check_draft_model_short_context`]): a draft whose context window
//!   is shorter than a request's reach is not driven past it — the request runs `auto` with the
//!   reason named — while a request within it runs the draft.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use core_llm::{
    FinishReason, LoadSpec, ProposerCapabilities, ProposerKind, Sampling, Speculative,
    SpeculativeProposer, StreamEvent, TextLlm, TextLlmRequest,
};
use serde_json::{json, Map, Value};

use crate::speculative::{
    bench_request, check_speculative_greedy_parity, BenchPrompt, ParityCase, PromptClass,
};

/// The fixture's vocabulary size (the tokenizer's and both models' `vocab_size`).
pub const DRAFT_FIXTURE_VOCAB: usize = 32;

/// The three snapshots [`write_draft_model_fixture`] writes.
#[derive(Clone, Debug)]
pub struct DraftModelFixture {
    /// The tiny Qwen3 target (2 layers, hidden 16).
    pub target: PathBuf,
    /// The smaller tiny Qwen3 draft (1 layer, narrower MLP, fewer heads) over the target's
    /// tokenizer.
    pub draft: PathBuf,
    /// A draft identical in shape to [`draft`](Self::draft) whose tokenizer is the same size over
    /// different tokens — the mismatched-tokenizer case.
    pub foreign_draft: PathBuf,
}

impl DraftModelFixture {
    /// The target load naming the compatible draft.
    pub fn spec_with_draft(&self) -> LoadSpec {
        LoadSpec::dense(self.target.to_string_lossy()).with_draft(self.draft.to_string_lossy())
    }

    /// The target load naming the foreign (mismatched-tokenizer) draft.
    pub fn spec_with_foreign_draft(&self) -> LoadSpec {
        LoadSpec::dense(self.target.to_string_lossy())
            .with_draft(self.foreign_draft.to_string_lossy())
    }
}

/// The deterministic SplitMix64 stream the fixture weights come from.
struct Stream(u64);

impl Stream {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// `n` values uniform in `[-scale, scale)`.
    fn uniform(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (self.next_f32() * 2.0 - 1.0) * scale)
            .collect()
    }
}

/// One tensor: its name, shape and `f32` data.
type Tensor = (String, Vec<usize>, Vec<f32>);

/// Write `tensors` as an `F32` safetensors file (8-byte little-endian header length, the JSON
/// header padded to 8 bytes, then the raw little-endian data).
fn write_safetensors(path: &Path, tensors: &[Tensor]) -> io::Result<()> {
    let mut header = Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let len = data.len() * 4;
        header.insert(
            name.clone(),
            json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len] }),
        );
        offset += len;
    }
    let mut header = serde_json::to_vec(&Value::Object(header))?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = Vec::with_capacity(8 + header.len() + offset);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    for (_, _, data) in tensors {
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(path, bytes)
}

/// A whitespace WordLevel `tokenizer.json` over `{prefix}0 .. {prefix}{vocab-1}`.
fn tokenizer_json(prefix: &str) -> String {
    let vocab: Map<String, Value> = (0..DRAFT_FIXTURE_VOCAB)
        .map(|i| (format!("{prefix}{i}"), json!(i)))
        .collect();
    json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": format!("{prefix}0") },
    })
    .to_string()
}

/// One tiny Qwen3 decoder's geometry and layer-weight scale.
struct Shape {
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    layer_scale: f32,
    seed: u64,
}

const HIDDEN: usize = 16;
const HEAD_DIM: usize = 4;

fn write_snapshot(
    dir: &Path,
    shape: &Shape,
    shared: &[Tensor],
    tokenizer_prefix: &str,
) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    // `eos_token_id` outside the vocabulary: every run decodes to its token budget.
    let config = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": shape.intermediate,
        "num_hidden_layers": shape.layers,
        "num_attention_heads": shape.heads,
        "num_key_value_heads": shape.kv_heads,
        "head_dim": HEAD_DIM,
        "vocab_size": DRAFT_FIXTURE_VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "max_position_embeddings": 512,
        "eos_token_id": 999,
    });
    fs::write(dir.join("config.json"), config.to_string())?;
    fs::write(dir.join("tokenizer.json"), tokenizer_json(tokenizer_prefix))?;

    let mut rng = Stream(shape.seed);
    let mut tensors: Vec<Tensor> = shared.to_vec();
    let (qd, kvd, inter, s) = (
        shape.heads * HEAD_DIM,
        shape.kv_heads * HEAD_DIM,
        shape.intermediate,
        shape.layer_scale,
    );
    for i in 0..shape.layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let mut push = |name: String, dims: Vec<usize>, data: Vec<f32>| {
            tensors.push((name, dims, data));
        };
        push(p("input_layernorm.weight"), vec![HIDDEN], vec![1.0; HIDDEN]);
        push(
            p("post_attention_layernorm.weight"),
            vec![HIDDEN],
            vec![1.0; HIDDEN],
        );
        push(
            p("self_attn.q_norm.weight"),
            vec![HEAD_DIM],
            vec![1.0; HEAD_DIM],
        );
        push(
            p("self_attn.k_norm.weight"),
            vec![HEAD_DIM],
            vec![1.0; HEAD_DIM],
        );
        push(
            p("self_attn.q_proj.weight"),
            vec![qd, HIDDEN],
            rng.uniform(qd * HIDDEN, s),
        );
        push(
            p("self_attn.k_proj.weight"),
            vec![kvd, HIDDEN],
            rng.uniform(kvd * HIDDEN, s),
        );
        push(
            p("self_attn.v_proj.weight"),
            vec![kvd, HIDDEN],
            rng.uniform(kvd * HIDDEN, s),
        );
        push(
            p("self_attn.o_proj.weight"),
            vec![HIDDEN, qd],
            rng.uniform(HIDDEN * qd, s),
        );
        push(
            p("mlp.gate_proj.weight"),
            vec![inter, HIDDEN],
            rng.uniform(inter * HIDDEN, s),
        );
        push(
            p("mlp.up_proj.weight"),
            vec![inter, HIDDEN],
            rng.uniform(inter * HIDDEN, s),
        );
        push(
            p("mlp.down_proj.weight"),
            vec![HIDDEN, inter],
            rng.uniform(HIDDEN * inter, s),
        );
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors)
}

/// Write the draft-model fixture under `root` (which must exist): `target/`, `draft/` and
/// `foreign_draft/` snapshot directories, each `config.json` + `tokenizer.json` +
/// `model.safetensors`, all `F32`. Deterministic: the same bytes on every call.
pub fn write_draft_model_fixture(root: &Path) -> io::Result<DraftModelFixture> {
    // Embedding and output projection are shared, so both models see the same token geometry and
    // their argmax agrees where the layers do not overturn it; the draft's single, weaker layer
    // overturns it less often than the target's two, which is where drafts get rejected.
    let mut rng = Stream(0x5EED_2443_6000);
    let shared: Vec<Tensor> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![DRAFT_FIXTURE_VOCAB, HIDDEN],
            rng.uniform(DRAFT_FIXTURE_VOCAB * HIDDEN, 1.0),
        ),
        (
            "lm_head.weight".into(),
            vec![DRAFT_FIXTURE_VOCAB, HIDDEN],
            rng.uniform(DRAFT_FIXTURE_VOCAB * HIDDEN, 1.0),
        ),
        ("model.norm.weight".into(), vec![HIDDEN], vec![1.0; HIDDEN]),
    ];
    let target = Shape {
        layers: 2,
        heads: 4,
        kv_heads: 2,
        intermediate: 32,
        layer_scale: 0.25,
        seed: 0x7A26_E700,
    };
    let draft = Shape {
        layers: 1,
        heads: 2,
        kv_heads: 1,
        intermediate: 16,
        layer_scale: 0.1,
        seed: 0xD2AF_7000,
    };
    let fixture = DraftModelFixture {
        target: root.join("target"),
        draft: root.join("draft"),
        foreign_draft: root.join("foreign_draft"),
    };
    write_snapshot(&fixture.target, &target, &shared, "t")?;
    write_snapshot(&fixture.draft, &draft, &shared, "t")?;
    write_snapshot(&fixture.foreign_draft, &draft, &shared, "w")?;
    Ok(fixture)
}

/// Prompts in the fixture's vocabulary: a repetitive one and a sparse one.
pub fn draft_model_prompts() -> Vec<BenchPrompt> {
    vec![
        BenchPrompt::user(
            "repeat",
            PromptClass::Predictable,
            "t3 t9 t4 t11 t3 t9 t4 t11 t3 t9 t4 t11",
        ),
        BenchPrompt::user("sparse", PromptClass::OpenEnded, "t5 t8 t1 t20 t13 t27 t2"),
    ]
}

/// The `{proposer: draft_model}` parity rows for an advertisement: depths 1, 3 (at most the
/// maximum), the recommended depth and the maximum the provider advertises.
pub fn draft_model_parity_cases(advertised: &ProposerCapabilities) -> Vec<ParityCase> {
    let mut depths = vec![
        1,
        3.min(advertised.max_depth),
        advertised.recommended_depth,
        advertised.max_depth,
    ];
    depths.dedup();
    depths
        .into_iter()
        .map(|depth| ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::DraftModel, depth),
            expect_proposer: ProposerKind::DraftModel,
        })
        .collect()
}

/// Tokens generated per parity row.
const MAX_NEW_TOKENS: u32 = 24;

/// A provider loaded with the compatible draft ([`DraftModelFixture::spec_with_draft`]): it
/// advertises `draft_model` (recommended depth within `1..=max_depth`, and `max_depth` the model's
/// verify bound — prompt lookup's), its load report names the draft resident, every
/// [`draft_model_parity_cases`] row of its advertisement emits exactly `off`'s greedy stream with
/// a report naming `draft_model` and no fallback, drafts were proposed, accepted and rejected,
/// and a seeded stochastic `draft_model` run is reproducible and reports `draft_model`.
pub fn check_draft_model_resident(provider: &dyn TextLlm, source: &str) -> Result<(), String> {
    let caps = &provider.descriptor().capabilities;
    let advertised = caps
        .proposer(SpeculativeProposer::DraftModel)
        .ok_or("a resident draft does not advertise `draft_model`")?;
    if advertised.proposer != SpeculativeProposer::DraftModel
        || advertised.recommended_depth < 1
        || advertised.recommended_depth > advertised.max_depth
    {
        return Err(format!(
            "`draft_model` advertised as {advertised:?}: needs 1 <= recommended <= max"
        ));
    }
    // Every draft is one more verify row whichever proposer drew it: the bound is the model's
    // own, the one prompt lookup advertises.
    if let Some(lookup) = caps.proposer(SpeculativeProposer::PromptLookup) {
        if lookup.max_depth != advertised.max_depth {
            return Err(format!(
                "`draft_model` depth bound {} is not the model's verify bound {} (prompt lookup)",
                advertised.max_depth, lookup.max_depth
            ));
        }
    }
    let draft = provider
        .load_report()
        .and_then(|r| r.draft)
        .ok_or("the load report does not name the draft")?;
    if !draft.is_resident() || draft.source != source {
        return Err(format!(
            "the load report names {draft:?}, not {source} resident"
        ));
    }

    let prompts = draft_model_prompts();
    let cases = draft_model_parity_cases(&advertised);
    let rows = check_speculative_greedy_parity(provider, &prompts, &cases, MAX_NEW_TOKENS)?;
    if rows.len() != prompts.len() * cases.len() {
        return Err(format!(
            "{} parity rows, expected {}",
            rows.len(),
            prompts.len() * cases.len()
        ));
    }
    if let Some(row) = rows.iter().find(|r| !r.report.fallbacks.is_empty()) {
        return Err(format!(
            "{}: fell back: {:?}",
            row.prompt_id, row.report.fallbacks
        ));
    }
    let (proposed, accepted) = rows.iter().fold((0, 0), |(p, a), r| {
        (p + r.report.proposed_tokens, a + r.report.accepted_tokens)
    });
    if accepted == 0 || accepted >= proposed {
        return Err(format!(
            "the draft must be both accepted and rejected across the rows: {accepted} of \
             {proposed} drafts accepted"
        ));
    }
    if let Some(row) = rows.iter().find(|r| {
        r.report.draft_tokens
            != match r.speculative {
                Speculative::Proposer { depth, .. } => Some(depth),
                _ => None,
            }
    }) {
        return Err(format!(
            "{}: the report names depth {:?} for {:?}",
            row.prompt_id, row.report.draft_tokens, row.speculative
        ));
    }

    // A seeded stochastic run (exact rejection sampling against the draft's own q) reproduces.
    let stochastic = Sampling {
        temperature: 0.9,
        top_p: 0.95,
        top_k: 12,
        ..Sampling::greedy()
    };
    let request = || -> TextLlmRequest {
        let mut req = bench_request(
            &prompts[1],
            Speculative::proposer(SpeculativeProposer::DraftModel, 3.min(advertised.max_depth)),
            &stochastic,
            MAX_NEW_TOKENS,
        );
        req.seed = Some(11);
        req
    };
    let a = provider
        .generate(&request(), &mut |_| {})
        .map_err(|e| e.to_string())?;
    let b = provider
        .generate(&request(), &mut |_| {})
        .map_err(|e| e.to_string())?;
    if a.text != b.text {
        return Err("a seeded stochastic draft_model run is not reproducible".into());
    }
    let report = a
        .decode
        .ok_or("the stochastic run returned no decode report")?;
    if report.proposer != ProposerKind::DraftModel || report.proposed_tokens == 0 {
        return Err(format!(
            "the stochastic run reports proposer `{}` with {} drafts",
            report.proposer.label(),
            report.proposed_tokens
        ));
    }
    Ok(())
}

/// A provider loaded with the foreign draft ([`DraftModelFixture::spec_with_foreign_draft`]):
/// the target loaded, its load report names the draft refused for its tokenizer vocabulary (in
/// `draft` and in `fallbacks`), `draft_model` is not advertised (a request for it decodes plainly
/// with the reason named, E2), and `off` decodes exactly as `alone` — the same target loaded with
/// no draft — does.
pub fn check_draft_model_refused(
    provider: &dyn TextLlm,
    alone: &dyn TextLlm,
    source: &str,
) -> Result<(), String> {
    let report = provider
        .load_report()
        .ok_or("the provider reports no load")?;
    let draft = report
        .draft
        .clone()
        .ok_or("the load report does not name the refused draft")?;
    let refusal = draft.refusal.clone().unwrap_or_default();
    if draft.source != source || !refusal.contains("tokenizer vocabulary is not the target's") {
        return Err(format!(
            "the load report names {draft:?}, not {source} refused for its tokenizer"
        ));
    }
    // E2: every load fallback is named in one place — the refusal is in `fallbacks` too.
    if !report.fallbacks.contains(&refusal) {
        return Err(format!(
            "the load's fallbacks {:?} do not name the refused draft ({refusal})",
            report.fallbacks
        ));
    }
    let caps = &provider.descriptor().capabilities;
    if caps.proposer(SpeculativeProposer::DraftModel).is_some() {
        return Err("a refused draft still advertises `draft_model`".into());
    }
    let prompts = draft_model_prompts();
    let ask = bench_request(
        &prompts[0],
        Speculative::proposer(SpeculativeProposer::DraftModel, 2),
        &Sampling::greedy(),
        MAX_NEW_TOKENS,
    );
    // E2: an explicit `draft_model` the model does not advertise is not refused: it decodes
    // plainly — exactly `off`'s tokens — with the reason named.
    provider.validate(&ask).map_err(|e| {
        format!("a request for an unadvertised `draft_model` was refused, not run plain: {e}")
    })?;
    let plain = provider
        .generate(&ask, &mut |_| {})
        .map_err(|e| e.to_string())?;
    let plain_report = plain.decode.clone().ok_or("no decode report")?;
    if plain_report.proposer != ProposerKind::None
        || !plain_report
            .fallbacks
            .iter()
            .any(|f| f.starts_with("speculative: `draft_model` is not available"))
    {
        return Err(format!(
            "an unadvertised `draft_model` request reports {:?} / {:?}, not plain with the \
             reason named",
            plain_report.proposer, plain_report.fallbacks
        ));
    }
    let off = bench_request(
        &prompts[0],
        Speculative::Off,
        &Sampling::greedy(),
        MAX_NEW_TOKENS,
    );
    let with = provider
        .generate(&off, &mut |_| {})
        .map_err(|e| e.to_string())?;
    let without = alone
        .generate(&off, &mut |_| {})
        .map_err(|e| e.to_string())?;
    if with.text != without.text || with.usage != without.usage || plain.text != with.text {
        return Err(
            "the target with a refused draft decodes differently from the target alone".into(),
        );
    }
    if alone.load_report().and_then(|r| r.draft).is_some() {
        return Err("a load naming no draft reports one".into());
    }
    Ok(())
}

/// Loads a provider for a [`LoadSpec`] — the backend-specific half of the fixture-editing checks.
pub type DraftLoader<'a> = &'a dyn Fn(&LoadSpec) -> Result<Box<dyn TextLlm>, String>;

/// Copy the snapshot directory `from` to `to` (created), applying `edit` to its `config.json`.
fn derive_snapshot(from: &Path, to: &Path, edit: impl FnOnce(&mut Value)) -> Result<(), String> {
    let io = |e: io::Error| format!("deriving {}: {e}", to.display());
    fs::create_dir_all(to).map_err(io)?;
    for entry in fs::read_dir(from).map_err(io)? {
        let entry = entry.map_err(io)?;
        fs::copy(entry.path(), to.join(entry.file_name())).map_err(io)?;
    }
    let path = to.join("config.json");
    let mut config: Value =
        serde_json::from_slice(&fs::read(&path).map_err(io)?).map_err(|e| e.to_string())?;
    edit(&mut config);
    fs::write(&path, config.to_string()).map_err(io)
}

/// The ids a generation streams, its text and finish reason, and its decode report's proposer.
struct Streamed {
    ids: Vec<u32>,
    text: String,
    finish: Option<FinishReason>,
    proposer: Option<ProposerKind>,
    fallbacks: Vec<String>,
}

fn stream(provider: &dyn TextLlm, request: &TextLlmRequest) -> Result<Streamed, String> {
    let mut ids = Vec::new();
    let out = provider
        .generate(request, &mut |event| {
            if let StreamEvent::Token { id, .. } = event {
                ids.push(id);
            }
        })
        .map_err(|e| format!("generate failed: {e}"))?;
    let report = out.decode;
    Ok(Streamed {
        ids,
        text: out.text,
        finish: out.finish_reason,
        proposer: report.as_ref().map(|r| r.proposer),
        fallbacks: report.map(|r| r.fallbacks).unwrap_or_default(),
    })
}

/// The fixture's target with a **real stop token**, loaded beside its draft through `load`: the
/// stop is the first token of the target's greedy `off` stream (after its first four) not seen
/// before in it, written as the derived target's `eos_token_id`, so a greedy draft run reaches it
/// inside a run of drafts. Every `draft_model` depth under greedy, a light presence penalty and a
/// near-zero temperature (the host-sampled draft paths, which stop drafting at a stop token
/// without feeding it) completes and reports `draft_model`; the greedy and penalized runs stop
/// exactly where `off` does, with the same stream.
pub fn check_draft_model_stop_token(
    fixture: &DraftModelFixture,
    load: DraftLoader<'_>,
) -> Result<(), String> {
    let prompts = draft_model_prompts();
    let plain = load(&LoadSpec::dense(fixture.target.to_string_lossy()))?;
    let greedy_stream = stream(
        plain.as_ref(),
        &bench_request(
            &prompts[0],
            Speculative::Off,
            &Sampling::greedy(),
            MAX_NEW_TOKENS,
        ),
    )?
    .ids;
    let end = (4..greedy_stream.len())
        .find(|&i| !greedy_stream[..i].contains(&greedy_stream[i]))
        .ok_or("the greedy stream repeats itself from its fifth token: no stop to pick")?;
    let stop = greedy_stream[end];
    let target = fixture.target.with_file_name("stop_target");
    derive_snapshot(&fixture.target, &target, |config| {
        config["eos_token_id"] = serde_json::json!(stop);
    })?;
    let provider = load(
        &LoadSpec::dense(target.to_string_lossy()).with_draft(fixture.draft.to_string_lossy()),
    )?;
    let advertised = provider
        .descriptor()
        .capabilities
        .proposer(SpeculativeProposer::DraftModel)
        .ok_or("a resident draft does not advertise `draft_model`")?;
    let penalized = Sampling {
        presence_penalty: 0.01,
        ..Sampling::greedy()
    };
    let cold = Sampling {
        temperature: 0.01,
        ..Sampling::greedy()
    };
    let mut failures = Vec::new();
    for (name, sampling, decided) in [
        ("greedy", Sampling::greedy(), true),
        ("penalized", penalized, true),
        ("cold", cold, false),
    ] {
        let off = stream(
            provider.as_ref(),
            &bench_request(&prompts[0], Speculative::Off, &sampling, MAX_NEW_TOKENS),
        )?;
        if decided && off.finish != Some(FinishReason::Stop) {
            return Err(format!(
                "{name}: `off` did not stop at token {stop} ({:?})",
                off.finish
            ));
        }
        for depth in draft_model_parity_cases(&advertised)
            .into_iter()
            .filter_map(|case| match case.speculative {
                Speculative::Proposer { depth, .. } => Some(depth),
                _ => None,
            })
        {
            let tag = format!("{name} depth {depth}");
            let run = match stream(
                provider.as_ref(),
                &bench_request(
                    &prompts[0],
                    Speculative::proposer(SpeculativeProposer::DraftModel, depth),
                    &sampling,
                    MAX_NEW_TOKENS,
                ),
            ) {
                Ok(run) => run,
                Err(e) => {
                    failures.push(format!("{tag}: {e}"));
                    continue;
                }
            };
            if run.proposer != Some(ProposerKind::DraftModel) {
                failures.push(format!("{tag}: ran {:?}", run.proposer));
            }
            if decided && (run.ids != off.ids || run.text != off.text || run.finish != off.finish) {
                failures.push(format!(
                    "{tag}: stopped as {:?} after {} tokens, `off` as {:?} after {}",
                    run.finish,
                    run.ids.len(),
                    off.finish,
                    off.ids.len()
                ));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

/// A draft whose context window is shorter than the target's (sc-24436, E2), loaded beside the
/// fixture's target through `load`: the derived draft's `max_position_embeddings` is the
/// repetitive prompt's length plus 8 generated tokens plus a depth-3 step's 4 positions. A
/// depth-3 `draft_model` request for 8 tokens runs the draft with no fallback; for 9 it runs
/// what `auto` resolves to instead — never
/// `draft_model` — names the draft's context window in its fallbacks, and emits `off`'s greedy
/// stream.
pub fn check_draft_model_short_context(
    fixture: &DraftModelFixture,
    load: DraftLoader<'_>,
) -> Result<(), String> {
    let prompts = draft_model_prompts();
    let plain = load(&LoadSpec::dense(fixture.target.to_string_lossy()))?;
    let prompt_tokens = plain
        .generate(
            &bench_request(&prompts[0], Speculative::Off, &Sampling::greedy(), 1),
            &mut |_| {},
        )
        .map_err(|e| e.to_string())?
        .usage
        .prompt_tokens;
    let draft = fixture.draft.with_file_name("short_draft");
    derive_snapshot(&fixture.draft, &draft, |config| {
        config["max_position_embeddings"] = serde_json::json!(prompt_tokens + 8 + 3 + 1);
    })?;
    let provider = load(
        &LoadSpec::dense(fixture.target.to_string_lossy()).with_draft(draft.to_string_lossy()),
    )?;
    let ask = |max_new_tokens| {
        bench_request(
            &prompts[0],
            Speculative::proposer(SpeculativeProposer::DraftModel, 3),
            &Sampling::greedy(),
            max_new_tokens,
        )
    };
    let within = stream(provider.as_ref(), &ask(8))?;
    if within.proposer != Some(ProposerKind::DraftModel) || !within.fallbacks.is_empty() {
        return Err(format!(
            "a request within the draft's context ran {:?} with fallbacks {:?}",
            within.proposer, within.fallbacks
        ));
    }
    let beyond = stream(provider.as_ref(), &ask(9))?;
    if beyond.proposer == Some(ProposerKind::DraftModel) {
        return Err("a request past the draft's context window still ran the draft".into());
    }
    if !beyond
        .fallbacks
        .iter()
        .any(|f| f.contains("exceeds the draft model's context window"))
    {
        return Err(format!(
            "the fallback does not name the draft's context window: {:?}",
            beyond.fallbacks
        ));
    }
    let off = stream(
        provider.as_ref(),
        &bench_request(&prompts[0], Speculative::Off, &Sampling::greedy(), 9),
    )?;
    if beyond.ids != off.ids || beyond.text != off.text {
        return Err("the fallen-back request's greedy stream differs from `off`".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_writes_three_readable_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let fixture = write_draft_model_fixture(root.path()).unwrap();
        for dir in [&fixture.target, &fixture.draft, &fixture.foreign_draft] {
            let bytes = fs::read(dir.join("model.safetensors")).unwrap();
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            assert_eq!(header_len % 8, 0);
            let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
            let data = bytes.len() - 8 - header_len;
            let end = header
                .as_object()
                .unwrap()
                .values()
                .map(|t| t["data_offsets"][1].as_u64().unwrap() as usize)
                .max()
                .unwrap();
            assert_eq!(end, data, "{}", dir.display());
            let tokenizer = core_llm::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
            assert_eq!(tokenizer.vocab_size(), DRAFT_FIXTURE_VOCAB);
        }
        let tok = |d: &PathBuf| core_llm::Tokenizer::from_file(d.join("tokenizer.json")).unwrap();
        assert_eq!(
            tok(&fixture.target).vocabulary_mismatch(&tok(&fixture.draft)),
            None
        );
        assert!(tok(&fixture.target)
            .vocabulary_mismatch(&tok(&fixture.foreign_draft))
            .is_some());
        // The draft is the smaller model.
        let size = |d: &PathBuf| fs::metadata(d.join("model.safetensors")).unwrap().len();
        assert!(size(&fixture.draft) < size(&fixture.target));
        // Deterministic bytes.
        let again = tempfile::tempdir().unwrap();
        let second = write_draft_model_fixture(again.path()).unwrap();
        assert_eq!(
            fs::read(fixture.target.join("model.safetensors")).unwrap(),
            fs::read(second.target.join("model.safetensors")).unwrap()
        );
    }
}
