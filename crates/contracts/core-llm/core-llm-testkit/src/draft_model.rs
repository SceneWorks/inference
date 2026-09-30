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

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use core_llm::{
    LoadSpec, ProposerKind, Sampling, Speculative, SpeculativeProposer, TextLlm, TextLlmRequest,
    DRAFT_MODEL_MAX_DEPTH, DRAFT_MODEL_RECOMMENDED_DEPTH,
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

/// The `{proposer: draft_model}` parity rows: depths 1, 3, the recommended depth and the maximum.
pub fn draft_model_parity_cases() -> Vec<ParityCase> {
    [1, 3, DRAFT_MODEL_RECOMMENDED_DEPTH, DRAFT_MODEL_MAX_DEPTH]
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
/// advertises `draft_model` at the shared depth bound, its load report names the draft
/// resident, every [`draft_model_parity_cases`] row emits exactly `off`'s greedy stream with a
/// report naming `draft_model` and no fallback, drafts were proposed, accepted and rejected, and a
/// seeded stochastic `draft_model` run is reproducible and reports `draft_model`.
pub fn check_draft_model_resident(provider: &dyn TextLlm, source: &str) -> Result<(), String> {
    let caps = &provider.descriptor().capabilities;
    let advertised = caps
        .proposer(SpeculativeProposer::DraftModel)
        .ok_or("a resident draft does not advertise `draft_model`")?;
    if advertised != core_llm::draft_model_capabilities() {
        return Err(format!(
            "`draft_model` advertised as {advertised:?}, not the shared bound"
        ));
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
    let cases = draft_model_parity_cases();
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
            Speculative::proposer(SpeculativeProposer::DraftModel, 3),
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
/// the target loaded, its load report names the draft refused for its tokenizer vocabulary,
/// `draft_model` is not advertised (a request for it is refused up front), and `off` decodes
/// exactly as `alone` — the same target loaded with no draft — does.
pub fn check_draft_model_refused(
    provider: &dyn TextLlm,
    alone: &dyn TextLlm,
    source: &str,
) -> Result<(), String> {
    let draft = provider
        .load_report()
        .and_then(|r| r.draft)
        .ok_or("the load report does not name the refused draft")?;
    let refusal = draft.refusal.clone().unwrap_or_default();
    if draft.source != source || !refusal.contains("tokenizer vocabulary is not the target's") {
        return Err(format!(
            "the load report names {draft:?}, not {source} refused for its tokenizer"
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
    if provider.validate(&ask).is_ok() {
        return Err("a request for an unadvertised `draft_model` validated".into());
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
    if with.text != without.text || with.usage != without.usage {
        return Err(
            "the target with a refused draft decodes differently from the target alone".into(),
        );
    }
    if alone.load_report().and_then(|r| r.draft).is_some() {
        return Err("a load naming no draft reports one".into());
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
