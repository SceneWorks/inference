//! Persisted MLX snapshot writer, shared by the GGUF and HF-safetensors ingest paths (epic 7153,
//! story 7660).
//!
//! A *snapshot* is the `{config.json, model.safetensors, tokenizer.json, tokenizer_config.json}`
//! directory the engine loads ([`crate::models::CausalLm::from_weights`]). Two producers feed it:
//! the GGUF converter ([`mod@crate::gguf::convert`], story 7165) and — added here — a Hugging Face
//! safetensors directory. Both funnel their dense tensor set through one sink, [`write_snapshot`],
//! so the requant + write logic lives in a single place.
//!
//! [`write_snapshot`] optionally re-quantizes the attention/MLP **projection** weights to MLX
//! group-wise Q4/Q8 ([`QuantizedLinear::quantize`]) and stores them as packed
//! `weight`/`scales`/`biases`, keeping embeddings, the LM head, and norms dense — the engine's
//! quant invariant. It writes a `config.json` carrying a matching `quantization` block (so the
//! loader reads the projections through its existing pre-quantized branch, llama.rs:69, with no
//! loader change) and drops the tokenizer files through verbatim.
//!
//! [`write_hf_snapshot`] is the HF leaf: load a dense HF model directory via [`Weights`] and persist
//! it as such a snapshot. With `quantize: None` the weights are written through unchanged, so the
//! snapshot reloads bit-identically to loading the source directly. Quantization selects projection
//! weights by name and mirrors the llama-family loader's coverage exactly ([`crate::models::CausalLm::from_weights_with`]):
//! split GQA `q/k/v/o_proj`, the packed Phi-3 / GLM-4 `qkv_proj` / `gate_up_proj` (split into the
//! standard projections **before** quantizing, exactly as the loader splits before load-time quant,
//! so the snapshot reads back through the loader's split-key pre-quantized branch), dense and
//! per-expert / shared-expert MoE MLPs, and the DeepSeek-V2 MLA low-rank projections. VLM vision
//! towers stay dense (their loaders are dense-only, matching load-time behavior). Anything else that
//! looks like an attention/FFN projection but is not recognized is a **loud [`Error::Unsupported`]**,
//! never a silent dense fallback — including the Qwen3.5/3.6 hybrid linear-attention family
//! (`linear_attn.*`), whose loader has no pre-quantized snapshot branch (prepare it dense and use
//! load-time quantization instead).

use std::path::{Path, PathBuf};

use mlx_rs::ops::split_sections;
use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use crate::config::{Architecture, ModelConfig};
use crate::error::{Error, Result};
use crate::primitives::quant::QuantizedLinear;
use crate::primitives::QuantSpec;
use crate::primitives::Weights;

/// The engine's bf16 compute/storage dtype. Projection weights are cast to it before requant so a
/// quantized snapshot matches the loader's compute path (and the GGUF converter's behavior).
const STORE_DTYPE: Dtype = Dtype::Bfloat16;

/// The attention/MLP projection weight suffixes quantization targets, mirroring the set the
/// llama-family loader quantizes on load ([`crate::models::CausalLm::from_weights_with`]): split
/// GQA attention, the DeepSeek-V2 MLA low-rank query/KV projections, and the gated-MLP triplet.
/// The MLP suffixes are bare (no `mlp.` prefix) so per-expert (`mlp.experts.{e}.gate_proj.weight`)
/// and shared-expert (`mlp.shared_expert(s).gate_proj.weight`) keys match alongside the dense
/// `mlp.gate_proj.weight` — exactly the keys the loader routes through its quantizing projection
/// loader. Embeddings (`model.embed_tokens.weight`), the LM head (`lm_head.weight`), norms, the MoE
/// router (`mlp.gate.weight`), and the shared-expert gate never match and so stay dense — the
/// engine's quant invariant. Packed Phi-3 / GLM-4 tensors are handled separately (split first, see
/// [`write_snapshot`]), and keys under a VLM vision tower are excluded entirely
/// (vision loaders are dense-only).
pub const PROJECTION_SUFFIXES: [&str; 11] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    // DeepSeek-V2 Multi-head Latent Attention low-rank projections.
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    // Gated MLP — bare so dense, per-expert, and shared-expert keys all match.
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
];

/// Packed Phi-3 q‖k‖v attention projection, split into `q/k/v_proj` before quantizing.
const PACKED_QKV_SUFFIX: &str = "self_attn.qkv_proj.weight";
/// Packed Phi-3 / GLM-4 gate‖up MLP projection, split into `gate/up_proj` before quantizing.
const PACKED_GATE_UP_SUFFIX: &str = "mlp.gate_up_proj.weight";

/// Weight-key roots of VLM vision towers / projectors (JoyCaption's SigLIP + LLaVA projector,
/// the Qwen3-VL ViT). Vision weights are never quantized at load time — their loaders are
/// dense-only — so the snapshot writer keeps them dense too (a SigLIP `self_attn.q_proj.weight`
/// would otherwise match the decoder suffixes).
const VISION_ROOTS: [&str; 4] = [
    "vision_tower.",
    "model.visual.",
    "visual.",
    "multi_modal_projector.",
];

/// Whether a weight key lives under a VLM vision tower / projector (kept dense).
fn under_vision_root(key: &str) -> bool {
    VISION_ROOTS.iter().any(|r| key.starts_with(r))
}

/// Whether a weight key is a quantization-eligible attention/MLP projection (packed Phi-3 / GLM-4
/// tensors are handled separately by [`write_snapshot`] — they are split first).
pub fn is_projection(key: &str) -> bool {
    !under_vision_root(key)
        && !is_packed_projection(key)
        && PROJECTION_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// Whether a weight key is a packed projection that must be split before quantizing.
fn is_packed_projection(key: &str) -> bool {
    !under_vision_root(key)
        && (key.ends_with(PACKED_QKV_SUFFIX) || key.ends_with(PACKED_GATE_UP_SUFFIX))
}

/// Whether a key that stayed dense under a quantize request looks like an attention/FFN projection
/// the writer does not know how to cover. Used as the loud-refusal net: matching keys abort the
/// write instead of silently producing a mixed-tier snapshot. Tensors the loader deliberately keeps
/// dense — q/k norms, the `q_a`/`kv_a` layernorms, the MoE router `mlp.gate.weight`, the
/// shared-expert gate, biases, and everything under a vision tower — are exempt.
fn is_unrecognized_projection(key: &str, arr: &Array) -> bool {
    if under_vision_root(key) {
        return false;
    }
    if ![".self_attn.", ".mlp.", ".linear_attn."]
        .iter()
        .any(|b| key.contains(b))
    {
        return false;
    }
    if key.ends_with("norm.weight")
        || key.ends_with(".mlp.gate.weight")
        || key.ends_with("shared_expert_gate.weight")
    {
        return false;
    }
    // A matrix-or-stacked-matrices `.weight` operand, or a projection tensor without the `.weight`
    // suffix (e.g. Qwen3.5-MoE's stacked `mlp.experts.gate_up_proj`). Refuse ndim > 2 too: a future
    // stacked-expert layout must get an explicit quantization path rather than reaching the 2-D
    // `QuantizedLinear` implementation and failing with an opaque shape error.
    (key.ends_with(".weight") && arr.shape().len() >= 2) || key.ends_with("_proj")
}

/// Tokenizer files to drop into a snapshot, written verbatim. The GGUF path supplies its
/// reconstructed `tokenizer.json` / `tokenizer_config.json` (serialized); the HF path supplies the
/// source files read through byte-for-byte. Either may be `None` (no file written).
#[derive(Clone, Debug, Default)]
pub struct SnapshotTokenizer {
    /// `tokenizer.json` contents.
    pub tokenizer_json: Option<String>,
    /// `tokenizer_config.json` contents.
    pub tokenizer_config_json: Option<String>,
}

/// What writing a snapshot produced.
#[derive(Clone, Debug)]
pub struct SnapshotReport {
    /// Number of weight tensors written (a quantized projection contributes three:
    /// `weight`/`scales`/`biases`).
    pub num_tensors: usize,
    /// The requant scheme applied to the projections, if any (`None` ⇒ dense). Only reported when
    /// every recognized projection was actually quantized — a request the writer cannot cover fully
    /// fails loudly instead of returning a report with silent dense fallbacks.
    pub quantized: Option<QuantSpec>,
    /// Number of projection matrices quantized (`0` for a dense write). A packed tensor split into
    /// its standard projections counts each split part.
    pub quantized_projections: usize,
    /// Directory the snapshot was written to.
    pub out_dir: PathBuf,
}

/// Write a loadable MLX snapshot to `out_dir` from a dense, HF-keyed tensor set.
///
/// When `quantize` is `Some`, each attention/MLP projection weight ([`is_projection`]) is cast to
/// bf16 and re-quantized to MLX group-wise Q4/Q8, stored as packed `weight`/`scales`/`biases`;
/// packed Phi-3 / GLM-4 `qkv_proj` / `gate_up_proj` tensors are cast to bf16 and **split into the
/// standard projections first** (the same split the loader performs before load-time quant), then
/// quantized and stored under the split keys so the loader reads them through its pre-quantized
/// branch. Every other tensor (embeddings, LM head, norms, biases, VLM vision towers, anything
/// else) is written through unchanged, and a matching `quantization` block is added to `config`.
/// When `quantize` is `None` every tensor is written through unchanged — a dense snapshot.
///
/// # Errors
///
/// A quantize request fails with [`Error::Unsupported`] — writing nothing rather than a snapshot
/// with silent dense fallbacks — when:
/// - the config names the Qwen3.5/3.6 hybrid linear-attention family (its loader has no
///   pre-quantized snapshot branch; prepare dense and quantize on load instead),
/// - the tensor set contains attention/FFN projection-like keys the writer does not recognize
///   (e.g. `linear_attn.*`), or
/// - no projection matched at all (the "quantized" snapshot would be entirely dense).
pub fn write_snapshot(
    out_dir: &Path,
    tensors: impl IntoIterator<Item = (String, Array)>,
    mut config: Value,
    tokenizer: &SnapshotTokenizer,
    quantize: Option<QuantSpec>,
) -> Result<SnapshotReport> {
    let tensors: Vec<(String, Array)> = tensors.into_iter().collect();

    // Quantization is only writable for families whose loader reads stored packed projections back
    // (the llama-family pre-quantized branch). Refuse the Qwen3.5/3.6 hybrid loudly: its loader
    // quantizes on load but has no pre-quantized snapshot branch, so a quantized snapshot would be
    // unloadable garbage.
    let mut split_dims: Option<(i32, i32, i32)> = None; // (q_dim, kv_dim, intermediate)
    if quantize.is_some() {
        let arch = Architecture::from_config(&config)
            .map_err(|e| Error::Unsupported(format!("cannot quantize snapshot: {e}")))?;
        if arch == Architecture::Qwen35 {
            return Err(Error::Unsupported(
                "cannot quantize snapshot: the Qwen3.5/3.6 hybrid linear-attention family \
                 (`linear_attn.*` Gated DeltaNet layers) has no pre-quantized snapshot loader \
                 branch; prepare the snapshot dense (quantize: None) and use load-time \
                 quantization instead"
                    .to_string(),
            ));
        }
        // Both packed and ordinary recognized projections feed rank-2 quantization operations.
        // Validate the common invariant before dispatch so neither branch can reach MLX with an
        // opaque shape failure (and before any output directory is created).
        if let Some((key, arr)) = tensors
            .iter()
            .find(|(key, arr)| (is_packed_projection(key) || is_projection(key)) && arr.ndim() != 2)
        {
            return Err(Error::Unsupported(format!(
                "cannot quantize projection `{key}` with shape {:?} (rank {}): expected a \
                 rank-2 weight matrix",
                arr.shape(),
                arr.ndim()
            )));
        }
        // Splitting packed Phi-3 / GLM-4 tensors needs the config's head/intermediate dims, derived
        // through the same `ModelConfig` parse the loader uses so the split points are identical.
        if tensors.iter().any(|(k, _)| is_packed_projection(k)) {
            let cfg = ModelConfig::from_json(&config)?;
            split_dims = Some((
                cfg.num_heads * cfg.head_dim,
                cfg.num_kv_heads * cfg.head_dim,
                cfg.intermediate_size,
            ));
        }
    }

    // Build the safetensors set: projections optionally requantized (cast to bf16 first, packed
    // tensors split first), the rest written through unchanged. Projection-like keys a quantize
    // request leaves dense are collected and refused after the pass.
    let mut out: Vec<(String, Array)> = Vec::new();
    let mut quantized_projections = 0usize;
    let mut uncovered: Vec<String> = Vec::new();
    for (key, arr) in tensors {
        match quantize {
            Some(spec) if is_packed_projection(&key) => {
                let (q_dim, kv_dim, inter) = split_dims.expect("packed key implies parsed dims");
                let w = arr.as_dtype(STORE_DTYPE)?;
                let rows = w.shape()[0];
                let (stem, names, points): (&str, &[&str], Vec<i32>) =
                    if key.ends_with(PACKED_QKV_SUFFIX) {
                        if rows != q_dim + 2 * kv_dim {
                            return Err(Error::Config(format!(
                                "packed `{key}` has {rows} rows; expected q+2·kv = {} from config",
                                q_dim + 2 * kv_dim
                            )));
                        }
                        (
                            key.strip_suffix("qkv_proj.weight").unwrap(),
                            &["q_proj", "k_proj", "v_proj"],
                            vec![q_dim, q_dim + kv_dim],
                        )
                    } else {
                        if rows != 2 * inter {
                            return Err(Error::Config(format!(
                                "packed `{key}` has {rows} rows; expected 2·intermediate = {} \
                                 from config",
                                2 * inter
                            )));
                        }
                        (
                            key.strip_suffix("gate_up_proj.weight").unwrap(),
                            &["gate_proj", "up_proj"],
                            vec![inter],
                        )
                    };
                let parts = split_sections(&w, &points, 0)?;
                for (name, part) in names.iter().zip(parts.iter()) {
                    push_quantized(&mut out, &format!("{stem}{name}"), part, spec)?;
                    quantized_projections += 1;
                }
            }
            Some(spec) if is_projection(&key) => {
                let w = arr.as_dtype(STORE_DTYPE)?;
                let base = key.strip_suffix(".weight").unwrap_or(&key);
                push_quantized(&mut out, base, &w, spec)?;
                quantized_projections += 1;
            }
            Some(_) if is_unrecognized_projection(&key, &arr) => {
                uncovered.push(key.clone());
                out.push((key, arr));
            }
            _ => out.push((key, arr)),
        }
    }
    if quantize.is_some() {
        if !uncovered.is_empty() {
            uncovered.sort();
            return Err(Error::Unsupported(format!(
                "cannot quantize snapshot: {} projection-like tensor(s) are not covered by the \
                 quantizer and would be silently written dense: {}",
                uncovered.len(),
                uncovered.join(", ")
            )));
        }
        if quantized_projections == 0 {
            return Err(Error::Unsupported(
                "cannot quantize snapshot: no attention/MLP projection weights matched — the \
                 \"quantized\" snapshot would be entirely dense"
                    .to_string(),
            ));
        }
    }
    let num_tensors = out.len();

    std::fs::create_dir_all(out_dir)?;

    // A `quantization` block marks the snapshot pre-quantized so the loader reads the stored
    // projections as-is (its `stored_quant` branch) rather than re-quantizing on load.
    if let Some(spec) = quantize {
        if let Value::Object(map) = &mut config {
            map.insert(
                "quantization".into(),
                json!({ "group_size": spec.group_size, "bits": spec.bits }),
            );
        } else {
            return Err(Error::Config(
                "snapshot config.json is not a JSON object".into(),
            ));
        }
    }
    write_json_string(&out_dir.join("config.json"), &config)?;

    Array::save_safetensors(
        out.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        out_dir.join("model.safetensors"),
    )
    .map_err(|e| Error::Msg(format!("write model.safetensors: {e}")))?;

    if let Some(t) = &tokenizer.tokenizer_json {
        std::fs::write(out_dir.join("tokenizer.json"), t)?;
    }
    if let Some(t) = &tokenizer.tokenizer_config_json {
        std::fs::write(out_dir.join("tokenizer_config.json"), t)?;
    }

    Ok(SnapshotReport {
        num_tensors,
        quantized: quantize,
        quantized_projections,
        out_dir: out_dir.to_path_buf(),
    })
}

/// Quantize a dense (bf16) projection weight and push its packed `weight`/`scales`/`biases` under
/// `base` (the key with the `.weight` suffix stripped).
fn push_quantized(
    out: &mut Vec<(String, Array)>,
    base: &str,
    w: &Array,
    spec: QuantSpec,
) -> Result<()> {
    let q = QuantizedLinear::quantize(w, spec.group_size, spec.bits, None)?;
    out.push((format!("{base}.weight"), q.weight));
    out.push((format!("{base}.scales"), q.scales));
    out.push((format!("{base}.biases"), q.biases));
    Ok(())
}

/// Persist a dense Hugging Face safetensors model directory as an MLX snapshot, optionally
/// quantizing the projections to Q4/Q8.
///
/// The dense tensor set is loaded via [`Weights`] (single file or sharded) and handed to
/// [`write_snapshot`]; `config.json` is read through (with a `quantization` block added when
/// quantizing — every other key preserved) and `tokenizer.json` / `tokenizer_config.json` are
/// copied verbatim when present. With `quantize: None` the weights are written unchanged, so the
/// snapshot reloads bit-identically to loading the source directly.
pub fn write_hf_snapshot(
    source_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<QuantSpec>,
) -> Result<SnapshotReport> {
    let source = source_dir.as_ref();
    let out_dir = out_dir.as_ref();

    // config.json is required — it carries the architecture + shapes the loader dispatches on. Read
    // it as a Value so the writer can add the quantization block; all other keys pass through.
    let config_path = source.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Config(format!("read {}: {e}", config_path.display())))?;
    let config: Value = serde_json::from_str(&config_text)
        .map_err(|e| Error::Config(format!("parse {}: {e}", config_path.display())))?;

    // Tokenizer files pass through verbatim (byte-identical) when present.
    let tokenizer = SnapshotTokenizer {
        tokenizer_json: read_to_string_if_exists(&source.join("tokenizer.json"))?,
        tokenizer_config_json: read_to_string_if_exists(&source.join("tokenizer_config.json"))?,
    };

    let weights = Weights::from_dir(source)?;
    write_snapshot(out_dir, weights.into_map(), config, &tokenizer, quantize)
}

/// Write a JSON value to `path`, pretty-printed (the snapshot's `config.json`).
fn write_json_string(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(path, text).map_err(|e| Error::Msg(format!("write {}: {e}", path.display())))?;
    Ok(())
}

/// Read a file to a string, returning `None` if it does not exist (other IO errors propagate).
fn read_to_string_if_exists(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Msg(format!("read {}: {e}", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::models::CausalLm;
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    use std::collections::HashMap;

    fn unique_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mlx-llm-snapshot-{label}-{}", std::process::id()))
    }

    fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    }

    /// A complete tiny Llama tensor set + matching config.json. Widths are multiples of the Q4/Q8
    /// group size (64) so the projections' input dim is quantizable: hidden 64 (head_dim 32 ×
    /// 2 heads), intermediate 128.
    fn tiny_model() -> (Vec<(String, Array)>, Value) {
        let (h, v, inter, qd, kvd, layers) = (64i32, 4i32, 128i32, 64i32, 32i32, 2usize);
        let mut rng = SplitMix64::new(0xABCDEF);
        let mut t: Vec<(String, Array)> = Vec::new();
        t.push(("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng)));
        t.push((
            "model.norm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        t.push(("lm_head.weight".into(), randn(&[v, h], &mut rng)));
        for i in 0..layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            t.push((
                p("input_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            ));
            t.push((
                p("post_attention_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            ));
            t.push((p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng)));
            t.push((p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng)));
            t.push((p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng)));
            t.push((p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng)));
            t.push((p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng)));
            t.push((p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng)));
            t.push((p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng)));
        }
        let config = json!({
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": v,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 99
        });
        (t, config)
    }

    #[test]
    fn projection_predicate_selects_only_attn_mlp_projections() {
        for k in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.7.self_attn.o_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
            // DeepSeek-V2 MLA low-rank projections
            "model.layers.1.self_attn.q_a_proj.weight",
            "model.layers.1.self_attn.q_b_proj.weight",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.1.self_attn.kv_b_proj.weight",
            // MoE per-expert and shared-expert MLPs
            "model.layers.2.mlp.experts.13.gate_proj.weight",
            "model.layers.2.mlp.experts.0.down_proj.weight",
            "model.layers.2.mlp.shared_experts.up_proj.weight", // DeepSeek plural
            "model.layers.2.mlp.shared_expert.gate_proj.weight", // Qwen2-MoE singular
            // Qwen3-VL nested decoder
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ] {
            assert!(is_projection(k), "{k} should be a projection");
        }
        for k in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight", // Qwen3 q/k norm stays dense
            "model.layers.0.self_attn.q_a_layernorm.weight", // MLA norms stay dense
            "model.layers.0.mlp.gate.weight",         // MoE router stays dense
            "model.layers.0.mlp.shared_expert_gate.weight", // Qwen2-MoE gate stays dense
            "model.layers.0.linear_attn.out_proj.weight", // Qwen3.6 hybrid: refused, not quantized
            "model.layers.0.mlp.gate_up_proj.weight", // packed: split before quantizing
            // VLM vision towers stay dense (their loaders are dense-only)
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "model.visual.blocks.0.attn.proj.weight",
        ] {
            assert!(!is_projection(k), "{k} should NOT be a projection");
        }
    }

    /// Everything the llama-family loader would quantize on load but this writer leaves dense must
    /// be flagged; everything the loader deliberately keeps dense must not be.
    #[test]
    fn unrecognized_projection_net_flags_uncovered_keys_only() {
        let m = Array::zeros::<f32>(&[64, 64]).unwrap(); // 2-D matmul operand
        let v = Array::zeros::<f32>(&[64]).unwrap(); // 1-D vector
        let stacked = Array::zeros::<f32>(&[4, 128, 64]).unwrap(); // stacked experts
        for (k, a) in [
            ("model.layers.0.linear_attn.in_proj_qkv.weight", &m),
            ("model.layers.0.linear_attn.out_proj.weight", &m),
            ("model.layers.0.mlp.experts.gate_up_proj", &stacked), // Qwen3.5-MoE stacked
            ("model.layers.0.self_attn.w_qkv.weight", &m),         // unknown layout
            ("model.layers.0.mlp.experts.w1.weight", &stacked),    // future stacked-expert layout
        ] {
            assert!(is_unrecognized_projection(k, a), "{k} must be refused");
        }
        for (k, a) in [
            ("model.layers.0.self_attn.q_norm.weight", &v),
            ("model.layers.0.self_attn.q_a_layernorm.weight", &v),
            ("model.layers.0.mlp.gate.weight", &m), // MoE router
            ("model.layers.0.mlp.shared_expert_gate.weight", &m),
            ("model.layers.0.self_attn.q_proj.bias", &v),
            ("model.layers.0.input_layernorm.weight", &v),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight",
                &m,
            ),
            ("model.visual.blocks.0.mlp.linear_fc1.weight", &m),
        ] {
            assert!(!is_unrecognized_projection(k, a), "{k} must not be flagged");
        }
    }

    /// Dense write: every tensor reloads bit-identical and config carries no quantization block.
    #[test]
    fn dense_write_round_trips_bit_identical() {
        let dir = unique_dir("dense");
        let (tensors, config) = tiny_model();
        let original: HashMap<String, Vec<u8>> = tensors
            .iter()
            .map(|(k, a)| (k.clone(), bytes_of(a)))
            .collect();

        let report =
            write_snapshot(&dir, tensors, config, &SnapshotTokenizer::default(), None).unwrap();
        assert_eq!(report.quantized, None);

        let reloaded = Weights::from_dir(&dir).unwrap();
        assert_eq!(reloaded.len(), original.len(), "tensor count preserved");
        for (k, want) in &original {
            let got = bytes_of(reloaded.require(k).unwrap());
            assert_eq!(&got, want, "tensor {k} must reload bit-identical");
        }
        // No quantization block was added.
        assert_eq!(ModelConfig::from_dir(&dir).unwrap().quantization, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Quantized write: projections expand to weight/scales/biases, config gains the quantization
    /// block, and the snapshot loads through the loader's pre-quantized branch and runs a forward.
    #[test]
    fn quantized_write_loads_through_prequantized_branch() {
        let dir = unique_dir("q8");
        let (tensors, config) = tiny_model();
        let spec = QuantSpec::q8();

        let report = write_snapshot(
            &dir,
            tensors,
            config,
            &SnapshotTokenizer::default(),
            Some(spec),
        )
        .unwrap();
        assert_eq!(report.quantized, Some(spec));
        assert_eq!(report.quantized_projections, 14, "2 layers × 7 projections");

        let w = Weights::from_dir(&dir).unwrap();
        // A projection was stored as packed parts; a dense tensor was not.
        let base = "model.layers.0.self_attn.q_proj";
        assert!(w.contains(&format!("{base}.weight")));
        assert!(w.contains(&format!("{base}.scales")));
        assert!(w.contains(&format!("{base}.biases")));
        assert!(
            !w.contains("model.embed_tokens.scales"),
            "embeddings stay dense"
        );

        let cfg = ModelConfig::from_dir(&dir).unwrap();
        assert_eq!(
            cfg.quantization,
            Some(spec),
            "config carries quantization block"
        );

        // Loads through `from_weights` (no load-time quant) as a quantized model and runs.
        let model = CausalLm::from_weights(&w, "", cfg).unwrap();
        assert!(model.is_quantized(), "snapshot must load as quantized");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The HF leaf: a dense source dir written through with `None` reloads bit-identical, and the
    /// tokenizer files pass through verbatim.
    #[test]
    fn hf_dense_passthrough_round_trips() {
        let src = unique_dir("hf-src");
        let out = unique_dir("hf-out");
        std::fs::create_dir_all(&src).unwrap();

        let (tensors, config) = tiny_model();
        let original: HashMap<String, Vec<u8>> = tensors
            .iter()
            .map(|(k, a)| (k.clone(), bytes_of(a)))
            .collect();
        std::fs::write(
            src.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(src.join("tokenizer.json"), "{\"tok\":true}").unwrap();
        std::fs::write(src.join("tokenizer_config.json"), "{\"cfg\":true}").unwrap();
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, src.join("model.safetensors")).unwrap();

        let report = write_hf_snapshot(&src, &out, None).unwrap();
        assert_eq!(report.quantized, None);

        let reloaded = Weights::from_dir(&out).unwrap();
        for (k, want) in &original {
            assert_eq!(
                &bytes_of(reloaded.require(k).unwrap()),
                want,
                "tensor {k} bit-identical"
            );
        }
        // Tokenizer files copied verbatim.
        assert_eq!(
            std::fs::read_to_string(out.join("tokenizer.json")).unwrap(),
            "{\"tok\":true}"
        );
        assert_eq!(
            std::fs::read_to_string(out.join("tokenizer_config.json")).unwrap(),
            "{\"cfg\":true}"
        );
        assert_eq!(ModelConfig::from_dir(&out).unwrap().quantization, None);

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    /// The HF leaf with quantization: source dir → quantized snapshot loads as quantized and runs.
    #[test]
    fn hf_quantized_loads_as_quantized() {
        let src = unique_dir("hfq-src");
        let out = unique_dir("hfq-out");
        std::fs::create_dir_all(&src).unwrap();

        let (tensors, config) = tiny_model();
        std::fs::write(
            src.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, src.join("model.safetensors")).unwrap();

        let report = write_hf_snapshot(&src, &out, Some(QuantSpec::q4())).unwrap();
        assert_eq!(report.quantized, Some(QuantSpec::q4()));

        let cfg = ModelConfig::from_dir(&out).unwrap();
        assert_eq!(cfg.quantization, Some(QuantSpec::q4()));
        let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();
        assert!(model.is_quantized());

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    /// A complete tiny Phi-3 tensor set (packed `qkv_proj` + `gate_up_proj`) with matching config:
    /// hidden 64 (head_dim 32 × 2 heads, 1 kv head ⇒ qkv rows 64 + 2·32 = 128), intermediate 128
    /// (gate_up rows 256).
    fn tiny_phi3() -> (Vec<(String, Array)>, Value) {
        let (h, v, inter) = (64i32, 4i32, 128i32);
        let mut rng = SplitMix64::new(0x5EED);
        let mut t: Vec<(String, Array)> = Vec::new();
        t.push(("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng)));
        t.push((
            "model.norm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        t.push(("lm_head.weight".into(), randn(&[v, h], &mut rng)));
        let p = |s: &str| format!("model.layers.0.{s}");
        t.push((
            p("input_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        t.push((
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        t.push((p("self_attn.qkv_proj.weight"), randn(&[128, h], &mut rng)));
        t.push((p("self_attn.o_proj.weight"), randn(&[h, 64], &mut rng)));
        t.push((
            p("mlp.gate_up_proj.weight"),
            randn(&[2 * inter, h], &mut rng),
        ));
        t.push((p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng)));
        let config = json!({
            "architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": v,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 99
        });
        (t, config)
    }

    /// Packed Phi-3: `qkv_proj` / `gate_up_proj` are split into the standard projections and
    /// quantized — bit-identically to the loader's own split-then-quantize load-time path — and the
    /// snapshot loads back through the loader's split-key pre-quantized branch as quantized.
    #[test]
    fn phi3_packed_projections_split_quantize_and_load() {
        let dir = unique_dir("phi3");
        let (tensors, config) = tiny_phi3();
        let qkv = tensors
            .iter()
            .find(|(k, _)| k.ends_with("self_attn.qkv_proj.weight"))
            .map(|(_, a)| a.clone())
            .unwrap();
        let spec = QuantSpec::q8();

        let report = write_snapshot(
            &dir,
            tensors,
            config,
            &SnapshotTokenizer::default(),
            Some(spec),
        )
        .unwrap();
        assert_eq!(report.quantized, Some(spec));
        assert_eq!(report.quantized_projections, 7, "q,k,v,o + gate,up,down");

        let w = Weights::from_dir(&dir).unwrap();
        // The packed keys are gone; the split projections are stored as packed quantized parts.
        assert!(!w.contains("model.layers.0.self_attn.qkv_proj.weight"));
        assert!(!w.contains("model.layers.0.mlp.gate_up_proj.weight"));
        for base in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ] {
            assert!(w.contains(&format!("{base}.weight")), "{base}.weight");
            assert!(w.contains(&format!("{base}.scales")), "{base}.scales");
            assert!(w.contains(&format!("{base}.biases")), "{base}.biases");
        }

        // Mirror-the-loader check: the stored q_proj equals quantizing the loader's own split of
        // the bf16 packed tensor (identical math ⇒ snapshot-quant ≡ load-time-quant).
        let qkv_bf16 = qkv.as_dtype(STORE_DTYPE).unwrap();
        let parts = split_sections(&qkv_bf16, &[64, 96], 0).unwrap();
        let expected =
            QuantizedLinear::quantize(&parts[0], spec.group_size, spec.bits, None).unwrap();
        let stored_w = w.require("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(
            stored_w.as_slice::<u32>(),
            expected.weight.as_slice::<u32>(),
            "stored packed q_proj must equal the loader-equivalent split-then-quantize"
        );
        let f32s = |a: &Array| -> Vec<f32> {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        let stored_s = w.require("model.layers.0.self_attn.q_proj.scales").unwrap();
        assert_eq!(f32s(stored_s), f32s(&expected.scales), "scales match");

        // Loads through the loader's split-key pre-quantized branch as a quantized model.
        let cfg = ModelConfig::from_dir(&dir).unwrap();
        assert_eq!(cfg.quantization, Some(spec));
        let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap();
        assert!(
            model.is_quantized(),
            "packed snapshot must load as quantized"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// MoE expert / shared-expert and DeepSeek MLA projections are quantized (they previously fell
    /// through as silent dense), while the router and MLA norms stay dense without tripping the
    /// refusal net.
    #[test]
    fn moe_and_mla_projections_are_quantized() {
        let dir = unique_dir("moe-mla");
        let (h, inter) = (64i32, 128i32);
        let mut rng = SplitMix64::new(0xD5EE);
        let p = |s: &str| format!("model.layers.0.{s}");
        let mut t: Vec<(String, Array)> = Vec::new();
        t.push(("model.embed_tokens.weight".into(), randn(&[4, h], &mut rng)));
        for key in [
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.experts.0.gate_proj.weight",
            "mlp.experts.0.up_proj.weight",
            "mlp.experts.1.gate_proj.weight",
            "mlp.shared_experts.gate_proj.weight",
        ] {
            t.push((p(key), randn(&[inter, h], &mut rng)));
        }
        t.push((
            p("mlp.experts.0.down_proj.weight"),
            randn(&[h, inter], &mut rng),
        ));
        t.push((
            p("self_attn.q_a_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        t.push((p("mlp.gate.weight"), randn(&[4, h], &mut rng))); // MoE router — dense
        let config = json!({ "model_type": "deepseek_v2" });

        let report = write_snapshot(
            &dir,
            t,
            config,
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q4()),
        )
        .unwrap();
        assert_eq!(report.quantized_projections, 10);

        let w = Weights::from_dir(&dir).unwrap();
        for base in [
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.q_b_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "model.layers.0.self_attn.kv_b_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.0.mlp.experts.1.gate_proj",
            "model.layers.0.mlp.shared_experts.gate_proj",
        ] {
            assert!(
                w.contains(&format!("{base}.scales")),
                "{base} must be quantized"
            );
        }
        // Router and MLA norm stay dense.
        assert!(w.contains("model.layers.0.mlp.gate.weight"));
        assert!(!w.contains("model.layers.0.mlp.gate.scales"));
        assert!(w.contains("model.layers.0.self_attn.q_a_layernorm.weight"));
        assert!(!w.contains("model.layers.0.self_attn.q_a_layernorm.scales"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The Qwen3.5/3.6 hybrid linear-attention family is refused loudly (its loader has no
    /// pre-quantized snapshot branch), and nothing is written.
    #[test]
    fn quantize_refuses_qwen35_hybrid() {
        let dir = unique_dir("refuse-qwen35");
        std::fs::remove_dir_all(&dir).ok();
        let t = vec![(
            "model.embed_tokens.weight".to_string(),
            Array::zeros::<f32>(&[4, 64]).unwrap(),
        )];
        let config = json!({ "model_type": "qwen3_5" });
        match write_snapshot(
            &dir,
            t,
            config,
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q4()),
        ) {
            Err(Error::Unsupported(msg)) => {
                assert!(
                    msg.contains("linear-attention"),
                    "message names the family: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(!dir.exists(), "a refused quantize must write nothing");
    }

    /// Projection-like keys the writer does not cover abort the write (loud refusal, never a silent
    /// mixed-tier snapshot). Dense writes of the same set still pass through fine.
    #[test]
    fn quantize_refuses_unrecognized_projection_keys() {
        let dir = unique_dir("refuse-unknown");
        std::fs::remove_dir_all(&dir).ok();
        let make = || {
            vec![
                (
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    randn(&[64, 64], &mut SplitMix64::new(2)),
                ),
                (
                    "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
                    randn(&[128, 64], &mut SplitMix64::new(3)),
                ),
            ]
        };
        let config = json!({ "model_type": "llama" }); // stripped config: arch gate alone won't catch it
        match write_snapshot(
            &dir,
            make(),
            config.clone(),
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q8()),
        ) {
            Err(Error::Unsupported(msg)) => {
                assert!(
                    msg.contains("model.layers.0.linear_attn.in_proj_qkv.weight"),
                    "message lists the uncovered key: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(!dir.exists(), "a refused quantize must write nothing");

        // The same tensor set writes fine dense.
        let report =
            write_snapshot(&dir, make(), config, &SnapshotTokenizer::default(), None).unwrap();
        assert_eq!(report.quantized, None);
        assert_eq!(report.quantized_projections, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A recognized projection suffix with a future stacked-expert layout must fail before the
    /// rank-2 quantizer, naming the offending key and shape, and must not create a partial snapshot.
    #[test]
    fn quantize_refuses_rank_three_recognized_projection() {
        let dir = unique_dir("refuse-rank-three-projection");
        std::fs::remove_dir_all(&dir).ok();
        let key = "model.layers.0.mlp.experts.gate_proj.weight";
        let tensors = vec![(key.to_string(), Array::zeros::<f32>(&[4, 128, 64]).unwrap())];
        let config = json!({ "model_type": "llama" });

        match write_snapshot(
            &dir,
            tensors,
            config,
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q4()),
        ) {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains(key), "message names the key: {msg}");
                assert!(
                    msg.contains("[4, 128, 64]"),
                    "message gives the shape: {msg}"
                );
                assert!(
                    msg.contains("rank 3"),
                    "message gives the actual rank: {msg}"
                );
                assert!(
                    msg.contains("rank-2"),
                    "message gives the expected rank: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(!dir.exists(), "a refused quantize must write nothing");
    }

    /// Packed projections share the same rank guard; malformed packed shapes must be rejected
    /// before the split logic indexes rows or invokes the rank-2 quantizer.
    #[test]
    fn quantize_refuses_rank_three_packed_projection() {
        let dir = unique_dir("refuse-rank-three-packed");
        std::fs::remove_dir_all(&dir).ok();
        let key = "model.layers.0.mlp.gate_up_proj.weight";
        let tensors = vec![(key.to_string(), Array::zeros::<f32>(&[2, 128, 64]).unwrap())];
        let (_, config) = tiny_phi3();

        match write_snapshot(
            &dir,
            tensors,
            config,
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q4()),
        ) {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains(key), "message names the key: {msg}");
                assert!(
                    msg.contains("[2, 128, 64]"),
                    "message gives the shape: {msg}"
                );
                assert!(
                    msg.contains("rank 3"),
                    "message gives the actual rank: {msg}"
                );
                assert!(
                    msg.contains("rank-2"),
                    "message gives the expected rank: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(!dir.exists(), "a refused quantize must write nothing");
    }

    /// A quantize request that matches no projections at all is refused — the report must never
    /// claim `quantized` for a snapshot that is entirely dense.
    #[test]
    fn quantize_refuses_zero_projection_coverage() {
        let dir = unique_dir("refuse-zero");
        std::fs::remove_dir_all(&dir).ok();
        let t = vec![(
            "model.embed_tokens.weight".to_string(),
            Array::zeros::<f32>(&[4, 64]).unwrap(),
        )];
        let config = json!({ "model_type": "llama" });
        match write_snapshot(
            &dir,
            t,
            config,
            &SnapshotTokenizer::default(),
            Some(QuantSpec::q4()),
        ) {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains("no attention/MLP projection"), "{msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(!dir.exists(), "a refused quantize must write nothing");
    }

    /// Raw little-endian bytes of an array's values, for bit-identity checks (dtype preserved).
    fn bytes_of(a: &Array) -> Vec<u8> {
        // Compare in the stored dtype without converting: read the f32 view is lossy for bf16, so
        // round-trip through the array's own element bytes via safetensors-equivalent f32 cast only
        // when float — here all tiny-model tensors are f32, so a direct f32 slice is exact.
        a.as_slice::<f32>()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect()
    }
}
