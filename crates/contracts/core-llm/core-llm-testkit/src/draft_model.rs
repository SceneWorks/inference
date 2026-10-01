//! The draft-model speculative checks (epic sc-24432, story sc-24436) — backend-neutral, driven
//! purely through [`TextLlm`] over one shared on-disk fixture, so the MLX and Candle providers load
//! the same snapshots and pass the same checks (E8).
//!
//! * **Fixture** ([`write_draft_model_fixture`]): a tiny Qwen3-architecture **target** and a
//!   smaller tiny Qwen3 **draft** sharing one tokenizer vocabulary, plus a **foreign** draft whose
//!   tokenizer is the same size over different tokens. Target and draft share their embedding and
//!   output projection and differ in depth, width and layer weights, so the draft agrees with the
//!   target often but not always: a greedy run accepts drafts *and* rejects some. Beside them, a
//!   tiny Qwen3.5 **hybrid** target with an MTP head, its sparse-MoE variant, a Qwen2-MoE Causal
//!   target and a smaller hybrid draft, all over the same tokenizer and shared embedding. Written
//!   as plain safetensors from this crate, so the fixture itself needs no tensor library. No real
//!   weights.
//! * **Resident** ([`check_draft_model_resident`]): a target loaded with the draft advertises
//!   `draft_model`, its load report names the draft resident, `{proposer: draft_model}` at every
//!   depth emits exactly the greedy stream of `off` with a report naming `draft_model`, drafts
//!   were proposed, accepted and rejected, and a seeded stochastic run is reproducible.
//! * **Other targets** ([`check_draft_model_targets`]): the resident check on the hybrid, hybrid
//!   MoE and Causal MoE targets beside the hybrid and the Causal draft, and every other proposer
//!   those targets advertise (MTP, prompt lookup) at depths 1, 3 and max, greedy-identical to
//!   `off` (epic AT1).
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
    advertised_parity_cases, bench_request, check_speculative_greedy_parity, BenchPrompt,
    ParityCase, PromptClass,
};

/// The fixture's vocabulary size (the tokenizer's and every model's `vocab_size`).
pub const DRAFT_FIXTURE_VOCAB: usize = 32;

/// The snapshots [`write_draft_model_fixture`] writes.
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
    /// A tiny Qwen3.5 hybrid target (a Gated DeltaNet layer then a gated full-attention layer,
    /// 6 query heads over 1 KV head at head dim 64) with an MTP head, over the target's
    /// tokenizer.
    pub hybrid_target: PathBuf,
    /// [`hybrid_target`](Self::hybrid_target) with every FFN — the MTP predictor layer's too — a
    /// sparse-MoE block (`qwen3_5_moe`, fused expert layout).
    pub hybrid_moe_target: PathBuf,
    /// A tiny Qwen2-MoE Causal target (2 layers, every FFN a sparse-MoE block) over the target's
    /// tokenizer.
    pub moe_target: PathBuf,
    /// A smaller tiny Qwen3.5 hybrid draft (no MTP head) over the target's tokenizer.
    pub hybrid_draft: PathBuf,
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

    /// Every (target, draft) pairing [`check_draft_model_targets`] runs beyond the Causal
    /// target's own: each hybrid target beside the hybrid draft and the Causal draft, and the
    /// Causal MoE target beside both, labelled.
    pub fn target_pairings(&self) -> Vec<(&'static str, &Path, &Path)> {
        vec![
            ("hybrid/hybrid", &self.hybrid_target, &self.hybrid_draft),
            ("hybrid/causal", &self.hybrid_target, &self.draft),
            (
                "hybrid-moe/hybrid",
                &self.hybrid_moe_target,
                &self.hybrid_draft,
            ),
            ("hybrid-moe/causal", &self.hybrid_moe_target, &self.draft),
            ("moe/causal", &self.moe_target, &self.draft),
            ("moe/hybrid", &self.moe_target, &self.hybrid_draft),
        ]
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

/// The decoder a fixture snapshot is written as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arch {
    /// `Qwen3ForCausalLM`: the Causal decoder.
    Qwen3,
    /// `Qwen2MoeForCausalLM`: the Causal decoder with every FFN a sparse MoE block (per-expert
    /// layout, sigmoid-gated shared expert) and biased q/k/v projections.
    Qwen2Moe,
    /// The Qwen3.5 hybrid (`qwen3_5` / `qwen3_5_moe`): Gated DeltaNet and gated full-attention
    /// layers alternating (`full_attention_interval` 2), every FFN dense or — `moe` — a sparse MoE
    /// block (fused Qwen3.6 expert layout), with an MTP head (`mtp`) whose predictor layer's FFN
    /// follows the body's.
    Qwen35 { moe: bool, mtp: bool },
}

/// One tiny decoder's geometry and layer-weight scale.
struct Shape {
    arch: Arch,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    layer_scale: f32,
    seed: u64,
}

const HIDDEN: usize = 16;
/// The Causal decoders' attention head dim.
const HEAD_DIM: usize = 4;
/// The hybrid targets' attention: 6 query heads over 1 KV head (a GQA group of 6) at head dim 64
/// — a geometry MLX's fused SDPA kernels serve (the vector kernel up to `32 / 6 = 5` query rows,
/// the full kernel past 8), so a verify there runs the production kernels rather than the
/// score-materializing fallback the narrow head dims take, and MLX's geometry-derived maximum
/// depth (4) differs from its 8-row ceiling.
const VECTOR_HEADS: usize = 6;
const VECTOR_HEAD_DIM: usize = 64;
/// The sparse-MoE blocks: experts, experts per token, expert and shared-expert widths.
const EXPERTS: usize = 4;
const EXPERTS_PER_TOKEN: usize = 2;
const MOE_INTERMEDIATE: usize = 16;
const SHARED_INTERMEDIATE: usize = 16;
/// The Gated DeltaNet layers: key heads, value heads, per-head dim and conv kernel width.
const LINEAR_KEY_HEADS: usize = 2;
const LINEAR_VALUE_HEADS: usize = 4;
const LINEAR_HEAD_DIM: usize = 4;
const CONV_KERNEL: usize = 4;

impl Shape {
    /// The weight-name root the decoder's layers, embedding and final norm live under.
    fn root(&self) -> &'static str {
        match self.arch {
            Arch::Qwen35 { .. } => "model.language_model",
            _ => "model",
        }
    }

    fn config(&self) -> Value {
        // `eos_token_id` outside the vocabulary: every run decodes to its token budget.
        let mut config = json!({
            "hidden_size": HIDDEN,
            "intermediate_size": self.intermediate,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": DRAFT_FIXTURE_VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "max_position_embeddings": 512,
            "eos_token_id": 999,
        });
        let moe = json!({
            "num_experts": EXPERTS,
            "num_experts_per_tok": EXPERTS_PER_TOKEN,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": SHARED_INTERMEDIATE,
            "norm_topk_prob": true,
        });
        let extend = |config: &mut Value, extra: &Value| {
            for (k, v) in extra.as_object().unwrap() {
                config[k] = v.clone();
            }
        };
        match self.arch {
            Arch::Qwen3 => {
                extend(
                    &mut config,
                    &json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
                );
                config
            }
            Arch::Qwen2Moe => {
                extend(
                    &mut config,
                    &json!({"architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe"}),
                );
                extend(&mut config, &moe);
                config
            }
            Arch::Qwen35 { moe: sparse, mtp } => {
                extend(
                    &mut config,
                    &json!({
                        "model_type": if sparse { "qwen3_5_moe_text" } else { "qwen3_5_text" },
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.5,
                        "full_attention_interval": 2,
                        "linear_num_key_heads": LINEAR_KEY_HEADS,
                        "linear_num_value_heads": LINEAR_VALUE_HEADS,
                        "linear_key_head_dim": LINEAR_HEAD_DIM,
                        "linear_value_head_dim": LINEAR_HEAD_DIM,
                        "linear_conv_kernel_dim": CONV_KERNEL,
                        "mtp_num_hidden_layers": u32::from(mtp),
                        "mtp_use_dedicated_embeddings": false,
                    }),
                );
                if sparse {
                    extend(&mut config, &moe);
                }
                json!({
                    "architectures": [if sparse {
                        "Qwen3_5MoeForConditionalGeneration"
                    } else {
                        "Qwen3_5ForConditionalGeneration"
                    }],
                    "model_type": if sparse { "qwen3_5_moe" } else { "qwen3_5" },
                    "eos_token_id": 999,
                    "text_config": config,
                })
            }
        }
    }

    /// Layer `prefix`'s full-attention projections (and the per-head q/k norms where the family
    /// has them); the hybrid's query projection carries its output gate.
    fn attention(&self, prefix: &str, rng: &mut Stream, tensors: &mut Vec<Tensor>) {
        let gate = if matches!(self.arch, Arch::Qwen35 { .. }) {
            2
        } else {
            1
        };
        let (qd, kvd, s) = (
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.layer_scale,
        );
        let name = |s: &str| format!("{prefix}.self_attn.{s}");
        if self.arch != Arch::Qwen2Moe {
            for norm in ["q_norm", "k_norm"] {
                tensors.push((
                    name(&format!("{norm}.weight")),
                    vec![self.head_dim],
                    vec![1.0; self.head_dim],
                ));
            }
        }
        for (proj, rows, cols) in [
            ("q_proj", qd * gate, HIDDEN),
            ("k_proj", kvd, HIDDEN),
            ("v_proj", kvd, HIDDEN),
            ("o_proj", HIDDEN, qd),
        ] {
            tensors.push((
                name(&format!("{proj}.weight")),
                vec![rows, cols],
                rng.uniform(rows * cols, s),
            ));
        }
        if self.arch == Arch::Qwen2Moe {
            for (proj, rows) in [("q_proj", qd), ("k_proj", kvd), ("v_proj", kvd)] {
                tensors.push((
                    name(&format!("{proj}.bias")),
                    vec![rows],
                    rng.uniform(rows, s),
                ));
            }
        }
    }

    /// Layer `prefix`'s Gated DeltaNet mixer.
    fn linear_attention(&self, prefix: &str, rng: &mut Stream, tensors: &mut Vec<Tensor>) {
        let key = LINEAR_KEY_HEADS * LINEAR_HEAD_DIM;
        let value = LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM;
        let conv = 2 * key + value;
        let s = self.layer_scale;
        let name = |s: &str| format!("{prefix}.linear_attn.{s}");
        for (t, dims) in [
            ("in_proj_qkv.weight", vec![conv, HIDDEN]),
            ("in_proj_z.weight", vec![value, HIDDEN]),
            ("in_proj_a.weight", vec![LINEAR_VALUE_HEADS, HIDDEN]),
            ("in_proj_b.weight", vec![LINEAR_VALUE_HEADS, HIDDEN]),
            ("conv1d.weight", vec![conv, 1, CONV_KERNEL]),
            ("A_log", vec![LINEAR_VALUE_HEADS]),
            ("dt_bias", vec![LINEAR_VALUE_HEADS]),
            ("out_proj.weight", vec![HIDDEN, value]),
        ] {
            let n = dims.iter().product();
            tensors.push((name(t), dims, rng.uniform(n, s)));
        }
        tensors.push((
            name("norm.weight"),
            vec![LINEAR_HEAD_DIM],
            vec![1.0; LINEAR_HEAD_DIM],
        ));
    }

    /// Layer `prefix`'s FFN: the dense MLP, or the family's sparse-MoE block.
    fn ffn(&self, prefix: &str, rng: &mut Stream, tensors: &mut Vec<Tensor>) {
        let s = self.layer_scale;
        let mut push = |name: String, dims: Vec<usize>| {
            let n = dims.iter().product();
            tensors.push((name, dims, rng.uniform(n, s)));
        };
        let mlp = |s: &str| format!("{prefix}.mlp.{s}");
        let triple = |push: &mut dyn FnMut(String, Vec<usize>), at: &str, inter: usize| {
            push(format!("{at}.gate_proj.weight"), vec![inter, HIDDEN]);
            push(format!("{at}.up_proj.weight"), vec![inter, HIDDEN]);
            push(format!("{at}.down_proj.weight"), vec![HIDDEN, inter]);
        };
        let sparse = match self.arch {
            Arch::Qwen3 | Arch::Qwen35 { moe: false, .. } => {
                return triple(&mut push, &format!("{prefix}.mlp"), self.intermediate);
            }
            Arch::Qwen2Moe => false,
            Arch::Qwen35 { moe: true, .. } => true,
        };
        push(mlp("gate.weight"), vec![EXPERTS, HIDDEN]);
        if sparse {
            push(
                mlp("experts.gate_up_proj"),
                vec![EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN],
            );
            push(
                mlp("experts.down_proj"),
                vec![EXPERTS, HIDDEN, MOE_INTERMEDIATE],
            );
        } else {
            for e in 0..EXPERTS {
                triple(&mut push, &mlp(&format!("experts.{e}")), MOE_INTERMEDIATE);
            }
        }
        triple(&mut push, &mlp("shared_expert"), SHARED_INTERMEDIATE);
        push(mlp("shared_expert_gate.weight"), vec![1, HIDDEN]);
    }
}

fn write_snapshot(
    dir: &Path,
    shape: &Shape,
    shared: &[Tensor],
    tokenizer_prefix: &str,
) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("config.json"), shape.config().to_string())?;
    fs::write(dir.join("tokenizer.json"), tokenizer_json(tokenizer_prefix))?;

    let mut rng = Stream(shape.seed);
    let root = shape.root();
    // The shared tensors are written by name under this decoder's root (`lm_head` stays at the
    // checkpoint root in every layout).
    let mut tensors: Vec<Tensor> = shared
        .iter()
        .map(|(name, dims, data)| {
            let name = match name.strip_prefix("model.") {
                Some(rest) => format!("{root}.{rest}"),
                None => name.clone(),
            };
            (name, dims.clone(), data.clone())
        })
        .collect();
    let ones = |name: String| (name, vec![HIDDEN], vec![1.0; HIDDEN]);
    for i in 0..shape.layers {
        let prefix = format!("{root}.layers.{i}");
        tensors.push(ones(format!("{prefix}.input_layernorm.weight")));
        tensors.push(ones(format!("{prefix}.post_attention_layernorm.weight")));
        // The hybrid alternates (interval 2): even layers are Gated DeltaNet, odd ones attention.
        if matches!(shape.arch, Arch::Qwen35 { .. }) && i % 2 == 0 {
            shape.linear_attention(&prefix, &mut rng, &mut tensors);
        } else {
            shape.attention(&prefix, &mut rng, &mut tensors);
        }
        shape.ffn(&prefix, &mut rng, &mut tensors);
    }
    if let Arch::Qwen35 { mtp: true, .. } = shape.arch {
        tensors.push((
            "mtp.fc.weight".into(),
            vec![HIDDEN, 2 * HIDDEN],
            rng.uniform(2 * HIDDEN * HIDDEN, shape.layer_scale),
        ));
        for norm in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"] {
            tensors.push(ones(format!("mtp.{norm}.weight")));
        }
        let prefix = "mtp.layers.0";
        tensors.push(ones(format!("{prefix}.input_layernorm.weight")));
        tensors.push(ones(format!("{prefix}.post_attention_layernorm.weight")));
        shape.attention(prefix, &mut rng, &mut tensors);
        shape.ffn(prefix, &mut rng, &mut tensors);
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors)
}

/// Write the draft-model fixture under `root` (which must exist): one snapshot directory per
/// [`DraftModelFixture`] field, each `config.json` + `tokenizer.json` + `model.safetensors`, all
/// `F32`. Deterministic: the same bytes on every call.
pub fn write_draft_model_fixture(root: &Path) -> io::Result<DraftModelFixture> {
    // Embedding and output projection are shared, so every model sees the same token geometry and
    // their argmax agrees where the layers do not overturn it; a draft's fewer, weaker layers
    // overturn it less often than a target's, which is where drafts get rejected.
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
        arch: Arch::Qwen3,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: HEAD_DIM,
        intermediate: 32,
        layer_scale: 0.25,
        seed: 0x7A26_E700,
    };
    let draft = Shape {
        arch: Arch::Qwen3,
        layers: 1,
        heads: 2,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        intermediate: 16,
        layer_scale: 0.1,
        seed: 0xD2AF_7000,
    };
    let hybrid = |moe, seed| Shape {
        arch: Arch::Qwen35 { moe, mtp: true },
        layers: 2,
        heads: VECTOR_HEADS,
        kv_heads: 1,
        head_dim: VECTOR_HEAD_DIM,
        intermediate: 32,
        layer_scale: 0.25,
        seed,
    };
    let hybrid_draft = Shape {
        arch: Arch::Qwen35 {
            moe: false,
            mtp: false,
        },
        layers: 2,
        heads: 2,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        intermediate: 16,
        layer_scale: 0.1,
        seed: 0xD2AF_7035,
    };
    let moe_target = Shape {
        arch: Arch::Qwen2Moe,
        intermediate: 2 * HIDDEN,
        seed: 0x7A26_E702,
        ..target
    };
    let fixture = DraftModelFixture {
        target: root.join("target"),
        draft: root.join("draft"),
        foreign_draft: root.join("foreign_draft"),
        hybrid_target: root.join("hybrid_target"),
        hybrid_moe_target: root.join("hybrid_moe_target"),
        moe_target: root.join("moe_target"),
        hybrid_draft: root.join("hybrid_draft"),
    };
    write_snapshot(&fixture.target, &target, &shared, "t")?;
    write_snapshot(&fixture.draft, &draft, &shared, "t")?;
    write_snapshot(&fixture.foreign_draft, &draft, &shared, "w")?;
    write_snapshot(
        &fixture.hybrid_target,
        &hybrid(false, 0x7A26_E735),
        &shared,
        "t",
    )?;
    write_snapshot(
        &fixture.hybrid_moe_target,
        &hybrid(true, 0x7A26_E736),
        &shared,
        "t",
    )?;
    write_snapshot(&fixture.moe_target, &moe_target, &shared, "t")?;
    write_snapshot(&fixture.hybrid_draft, &hybrid_draft, &shared, "t")?;
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

/// The fixture's other targets (epic sc-24432 AT1, E1), each loaded through `load` beside each
/// draft ([`DraftModelFixture::target_pairings`] — the Qwen3.5 hybrid and its MoE variant, and
/// the Qwen2-MoE Causal target, beside the hybrid draft and the Causal draft): every pairing
/// passes [`check_draft_model_resident`] (`draft_model` at depths 1, 3, the recommended depth and
/// the advertised max emits `off`'s greedy stream, drafts accepted and rejected), and on each
/// target every other proposer it must advertise — the hybrids' MTP head and prompt lookup,
/// prompt lookup on the Causal MoE target — at depths 1, 3 and its advertised max
/// ([`advertised_parity_cases`]) emits `off`'s greedy stream, with no fallback, the report naming
/// the depth, and each proposer drafting somewhere.
pub fn check_draft_model_targets(
    fixture: &DraftModelFixture,
    load: DraftLoader<'_>,
) -> Result<(), String> {
    let prompts = draft_model_prompts();
    let mut failures = Vec::new();
    let mut checked: Vec<&Path> = Vec::new();
    for (label, target, draft) in fixture.target_pairings() {
        let source = draft.to_string_lossy();
        let spec = LoadSpec::dense(target.to_string_lossy()).with_draft(source.clone());
        let provider = match load(&spec) {
            Ok(provider) => provider,
            Err(e) => {
                failures.push(format!("{label}: load failed: {e}"));
                continue;
            }
        };
        if let Err(e) = check_draft_model_resident(provider.as_ref(), &source) {
            failures.push(format!("{label}: {e}"));
        }
        if checked.contains(&target) {
            continue;
        }
        checked.push(target);
        let expected: &[SpeculativeProposer] = if target == fixture.moe_target {
            &[SpeculativeProposer::PromptLookup]
        } else {
            &[SpeculativeProposer::Mtp, SpeculativeProposer::PromptLookup]
        };
        let caps = &provider.descriptor().capabilities;
        for proposer in expected {
            if caps.proposer(*proposer).is_none() {
                failures.push(format!("{label}: `{}` is not advertised", proposer.label()));
            }
        }
        let cases: Vec<ParityCase> = advertised_parity_cases(caps)
            .into_iter()
            .filter(|case| case.expect_proposer != ProposerKind::DraftModel)
            .collect();
        let rows = match check_speculative_greedy_parity(
            provider.as_ref(),
            &prompts,
            &cases,
            MAX_NEW_TOKENS,
        ) {
            Ok(rows) => rows,
            Err(e) => {
                failures.push(format!("{label}: {e}"));
                continue;
            }
        };
        for row in &rows {
            let depth = match row.speculative {
                Speculative::Proposer { depth, .. } => Some(depth),
                _ => None,
            };
            if !row.report.fallbacks.is_empty() || row.report.draft_tokens != depth {
                failures.push(format!(
                    "{label} [{}] {:?}: depth {:?}, fallbacks {:?}",
                    row.prompt_id, row.speculative, row.report.draft_tokens, row.report.fallbacks
                ));
            }
        }
        for proposer in expected {
            let kind = ProposerKind::from(*proposer);
            if !rows
                .iter()
                .any(|r| r.report.proposer == kind && r.report.proposed_tokens > 0)
            {
                failures.push(format!("{label}: `{}` never drafted", proposer.label()));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
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
    fn the_fixture_writes_every_snapshot_readable() {
        let root = tempfile::tempdir().unwrap();
        let fixture = write_draft_model_fixture(root.path()).unwrap();
        for dir in [
            &fixture.target,
            &fixture.draft,
            &fixture.foreign_draft,
            &fixture.hybrid_target,
            &fixture.hybrid_moe_target,
            &fixture.moe_target,
            &fixture.hybrid_draft,
        ] {
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
        // Each draft is smaller than the targets it drafts for.
        let size = |d: &PathBuf| fs::metadata(d.join("model.safetensors")).unwrap().len();
        assert!(size(&fixture.draft) < size(&fixture.target));
        for (_, target, draft) in fixture.target_pairings() {
            assert!(size(&draft.to_path_buf()) < size(&target.to_path_buf()));
        }
        // The hybrids are written as the qwen3_5 family, the MoE variants with expert banks.
        let config = |d: &PathBuf| -> Value {
            serde_json::from_slice(&fs::read(d.join("config.json")).unwrap()).unwrap()
        };
        assert_eq!(config(&fixture.hybrid_target)["model_type"], "qwen3_5");
        assert_eq!(
            config(&fixture.hybrid_moe_target)["model_type"],
            "qwen3_5_moe"
        );
        assert_eq!(
            config(&fixture.hybrid_target)["text_config"]["mtp_num_hidden_layers"],
            1
        );
        assert_eq!(config(&fixture.moe_target)["model_type"], "qwen2_moe");
        // Deterministic bytes.
        let again = tempfile::tempdir().unwrap();
        let second = write_draft_model_fixture(again.path()).unwrap();
        assert_eq!(
            fs::read(fixture.target.join("model.safetensors")).unwrap(),
            fs::read(second.target.join("model.safetensors")).unwrap()
        );
    }
}
