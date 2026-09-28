//! Persisted, backend-neutral model-snapshot preparation (story 7662): the candle backend's
//! [`core_llm::SnapshotPreparerRegistration`].
//!
//! `core-llm` owns format detection, the explicit preparer registry, and dispatch (story 7659); a
//! backend supplies the *tensor work* — turn a downloaded model (an HF-safetensors snapshot directory
//! or a `*.gguf` container) into a persisted, loadable snapshot, optionally re-quantizing the
//! projections on the way out. This is candle's peer of the mlx-llm impl: candle already passes
//! `core-llm` conformance, so a working preparer here de-provisionalizes the *convert+quantize* seam
//! across a second backend.
//!
//! # What it writes
//! A prepared snapshot is the HF shape [`TextLlmRegistry::load_for_model`](core_llm::TextLlmRegistry::load_for_model) already consumes:
//! `config.json` + `model.safetensors` (via [`candle_core::safetensors::save`]) + `tokenizer.json`
//! (and `tokenizer_config.json` when there is a chat template). Reading the dense tensors reuses the
//! loaders candle already has — HF via [`Weights::from_dir`], GGUF via [`GgufCheckpoint::open`]
//! (Candle's native reader dequantizes every GGML block type for free), so the writer is uniform
//! across both inputs.
//!
//! # How quantization is persisted
//! A Q4 / Q8 snapshot stores its layer projections **already quantized** (sc-19375), so a tier
//! directory holds only the bytes of the tier a user picked — never a dense bundle re-quantized at
//! load. Every tensor the loader would quantize at load — for a llama-family [`CausalLm`](crate::CausalLm)
//! the layer projections `llama::loads_quantized` selects (Phi-3's fused `qkv_proj` /
//! `gate_up_proj` included), for the qwen3_5 hybrid the decoder and MTP projections
//! `qwen35::loads_quantized` selects (its stacked MoE experts included) — is quantized once by
//! Candle's quantizer and written as a **stored GGML block tensor**: a `U8` `[rows,
//! blocks_per_row, block_bytes]` tensor holding the raw GGML blocks (a stacked `[experts, rows,
//! cols]` weight keeps its leading dimension), see
//! [`to_ggml_block_tensor`](crate::primitives::quant::to_ggml_block_tensor); the block type is
//! recovered from `block_bytes`, see `ggml_block_storage`. The loader rebuilds each `QTensor`
//! straight from those bytes onto the target device — no dequantize → re-quantize — carving a
//! fused part or an expert out by rows (each row is whole blocks), so a prepared snapshot loads
//! to exactly the model a dense load quantized at load time would build. `config.json` carries
//! `quantization: { bits, storage: "ggml" }`. The selection is the one load admission prices the
//! quantized copy by, so a tier never stores a tensor admission does not price.
//!
//! Every other tensor — embeddings, norms, the LM head (which the qwen3_5 loader quantizes at load
//! from its dense weight, per the contract's tensor-level invariant), and any tensor the loader
//! does not build — is written exactly as read. Snapshots prepared before sc-19375 carry dense
//! weights with the quantization rounding and a bare `quantization` block; they load exactly as
//! before (a llama-family load re-quantizes them; the qwen3_5 loader honours only a `storage:
//! "ggml"` block, so it loads them dense).
//!
//! Q4 is Q4_K (256-weight blocks), falling back to Q4_0 (32-weight blocks, the same 4.5
//! bits/weight) for a projection whose input dimension is 32- but not 256-aligned; Q8 is Q8_0
//! (32-weight blocks). A projection aligned to neither is a clear error, not a silent dense
//! fallback, matching quantize-on-load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use serde_json::{Map, Value as Json};

use core_llm::{
    detect_format, ModelFormat, PrepareReport, PrepareSpec, Quantize, Result as CoreResult,
    SnapshotPreparerRegistration,
};

use crate::config::{Architecture, ModelConfig};
use crate::error::{Error, Result};
use crate::gguf::GgufCheckpoint;
use crate::models::{llama, qwen35, Qwen35Config};
use crate::primitives::projection::QuantSpec;
use crate::primitives::Weights;
use crate::provider::to_core;

/// The backend tag this preparer registers under (matches the provider's `backend` field).
const BACKEND: &str = "candle";

fn backend() -> &'static str {
    BACKEND
}

/// Weightless probe: can the candle backend prepare `spec.source`? A `*.gguf` container is accepted;
/// an HF source must be a directory holding `config.json` and must not be a multimodal snapshot (a
/// `vision_config` block belongs to the vision provider's preparer). Reads only `config.json`, never a
/// weight shard — mirrors [`provider::can_load`](crate::provider::can_load).
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    match detect_format(&spec.source) {
        Ok(ModelFormat::Gguf) => true,
        Ok(ModelFormat::Safetensors) => {
            spec.source.is_dir()
                && spec.source.join("config.json").is_file()
                && !has_vision_config(&spec.source.join("config.json"))
        }
        Err(_) => false,
    }
}

/// Materialize a persisted, loadable snapshot per `spec`. The detected format selects the reader; the
/// `quantize` knob selects dense vs. projection re-quantization. A dense HF source that is already a
/// loadable snapshot is returned as-is ([`PrepareReport::passthrough`]); everything else is written
/// to `spec.out_dir`.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    match detect_format(&spec.source)? {
        ModelFormat::Gguf => prepare_gguf(spec).map_err(to_core),
        ModelFormat::Safetensors => prepare_hf(spec).map_err(to_core),
    }
}

/// Prepare from an HF-safetensors snapshot directory. Dense + already-loadable ⇒ passthrough;
/// quantized ⇒ load the dense weights, round the projections, and write a fresh snapshot.
fn prepare_hf(spec: &PrepareSpec) -> Result<PrepareReport> {
    let src = &spec.source;
    let config_path = src.join("config.json");
    let tokenizer_path = src.join("tokenizer.json");
    if !config_path.is_file() {
        return Err(Error::Unsupported(format!(
            "prepare: HF source {} has no config.json",
            src.display()
        )));
    }
    if !tokenizer_path.is_file() {
        return Err(Error::Unsupported(format!(
            "prepare: HF source {} has no tokenizer.json (cannot build a self-contained snapshot)",
            src.display()
        )));
    }

    let Some(q) = spec.quantize.map(quant_spec).transpose()? else {
        // Dense: the source is already a loadable snapshot, so return it untouched (write nothing).
        let num_tensors = count_safetensors_tensors(src)?;
        if num_tensors == 0 {
            return Err(Error::Msg(format!(
                "prepare: HF source {} has no safetensors tensors",
                src.display()
            )));
        }
        return Ok(PrepareReport {
            input_format: ModelFormat::Safetensors,
            quantized: None,
            out_dir: src.clone(),
            num_tensors,
            passthrough: true,
        });
    };

    // Quantized: load dense weights, round the projections to the requested scheme, write the
    // snapshot. The source dtype is preserved (the loader casts to its compute dtype anyway).
    let mut config = read_json(&config_path)?;
    let mut tensors = Weights::from_dir(src, &Device::Cpu)?.into_map();
    let stored = quantize_projections(&mut tensors, q, &config)?;

    std::fs::create_dir_all(&spec.out_dir)?;
    stamp_quantization(&mut config, q, stored);
    write_json(&spec.out_dir.join("config.json"), &config)?;
    save_safetensors(&tensors, &spec.out_dir.join("model.safetensors"))?;
    std::fs::copy(&tokenizer_path, spec.out_dir.join("tokenizer.json"))?;
    copy_optional(src, &spec.out_dir, "tokenizer_config.json")?;
    copy_optional(src, &spec.out_dir, "special_tokens_map.json")?;

    Ok(PrepareReport {
        input_format: ModelFormat::Safetensors,
        quantized: spec.quantize,
        out_dir: spec.out_dir.clone(),
        num_tensors: tensors.len(),
        passthrough: false,
    })
}

/// Prepare from a `*.gguf` container: Candle's native reader dequantizes every block type, then this
/// writes an HF-shaped snapshot (reconstructed `config.json`, a `tokenizer.json` rebuilt from the
/// GGUF metadata, and `model.safetensors`). Always writes — a GGUF is not itself an HF snapshot — so
/// it is never a passthrough. The dequantized tensors are stored as f16 (a GGUF dequantizes to f32,
/// and f16 keeps the snapshot from bloating).
fn prepare_gguf(spec: &PrepareSpec) -> Result<PrepareReport> {
    let gguf_path = resolve_gguf_path(&spec.source)?;
    let ck = GgufCheckpoint::open(&gguf_path, &Device::Cpu)?;

    // Pull everything needed off the checkpoint before consuming its weights.
    let tokenizer_json = ck.tokenizer_json_from_metadata()?;
    let mut config = ck.config_json.clone();
    let stop_tokens = ck.stop_tokens.clone();
    let chat_template = ck.chat_template.clone();
    let bos_token = ck.bos_token.clone();
    let eos_token = ck.eos_token.clone();
    let mut tensors = ck.weights.into_map();

    // Store dense tensors as f16 (GGUF dequantizes to f32).
    for t in tensors.values_mut() {
        *t = t.to_dtype(DType::F16)?;
    }
    let quant = spec.quantize.map(quant_spec).transpose()?;
    if let Some(q) = quant {
        let stored = quantize_projections(&mut tensors, q, &config)?;
        stamp_quantization(&mut config, q, stored);
    }

    // The GGUF reconstructed config carries no stop-token ids; stamp them from the GGUF metadata so
    // the converted snapshot stops correctly without the original GGUF.
    if let Some(obj) = config.as_object_mut() {
        if !obj.contains_key("eos_token_id") && !stop_tokens.is_empty() {
            let ids: Vec<Json> = stop_tokens.iter().map(|&i| Json::from(i)).collect();
            obj.insert("eos_token_id".into(), Json::Array(ids));
        }
    }

    std::fs::create_dir_all(&spec.out_dir)?;
    write_json(&spec.out_dir.join("config.json"), &config)?;
    save_safetensors(&tensors, &spec.out_dir.join("model.safetensors"))?;
    std::fs::write(spec.out_dir.join("tokenizer.json"), &tokenizer_json)?;
    if let Some(template) = chat_template {
        let mut tc = Map::new();
        tc.insert("chat_template".into(), Json::String(template));
        if let Some(b) = bos_token {
            tc.insert("bos_token".into(), Json::String(b));
        }
        if let Some(e) = eos_token {
            tc.insert("eos_token".into(), Json::String(e));
        }
        write_json(
            &spec.out_dir.join("tokenizer_config.json"),
            &Json::Object(tc),
        )?;
    }

    Ok(PrepareReport {
        input_format: ModelFormat::Gguf,
        quantized: spec.quantize,
        out_dir: spec.out_dir.clone(),
        num_tensors: tensors.len(),
        passthrough: false,
    })
}

/// Map the contract's [`Quantize`] knob to the engine's [`QuantSpec`].
fn quant_spec(q: Quantize) -> Result<QuantSpec> {
    match q {
        Quantize::Q4 => Ok(QuantSpec::q4()),
        Quantize::Q8 => Ok(QuantSpec::q8()),
        // NVFP4 is a load-time CUDA capability (sc-24135): quantized on the device at load, never
        // persisted into a prepared snapshot.
        Quantize::Nvfp4 => Err(Error::Unsupported(
            "nvfp4: NVFP4 is quantized at load on a CUDA sm_120 device and is never persisted by \
             snapshot preparation; prepare a dense snapshot and load it with Quantize::Nvfp4"
                .into(),
        )),
    }
}

/// The tensors a load of a snapshot quantizes at load time — the decoder family's own selection,
/// the one load admission prices the quantized copy by ([`llama::loads_quantized`],
/// [`qwen35::loads_quantized`]).
enum QuantizedTensors {
    /// A llama-family [`CausalLm`](crate::CausalLm), its decoder under `root`.
    Causal { cfg: ModelConfig, root: String },
    /// The qwen3_5 hybrid, its decoder under `prefix`.
    Qwen35 { cfg: Qwen35Config, prefix: String },
}

impl QuantizedTensors {
    /// Dispatch `config` as the loader does. A quantized tier exists only for a decoder the loader
    /// quantizes: Prism/Bonsai is already packed, and a config neither decoder reads would write a
    /// snapshot no loader opens — both refused.
    fn from_config(config: &Json, has_key: impl Fn(&str) -> bool) -> Result<Self> {
        if config.get("model_type").and_then(Json::as_str) == Some("prism_hadamard_qwen35") {
            return Err(Error::Unsupported(
                "prepare: a Prism/Bonsai snapshot is already packed affine-2; it has no Q4 / Q8 \
                 tier"
                    .into(),
            ));
        }
        match Architecture::from_config(config)? {
            Architecture::Qwen35 => {
                let cfg = Qwen35Config::from_json(config)?;
                let prefix = crate::provider::qwen35_dense_prefix(has_key)
                    .map_err(|e| Error::Unsupported(format!("prepare: {e}")))?;
                Ok(Self::Qwen35 {
                    cfg,
                    prefix: prefix.to_string(),
                })
            }
            _ => {
                let cfg = ModelConfig::from_json(config)?;
                let root = llama::decoder_root(&cfg, "", has_key);
                Ok(Self::Causal { cfg, root })
            }
        }
    }

    /// Whether the loader stores `key` in the requested projection format.
    fn contains(&self, key: &str) -> bool {
        match self {
            Self::Causal { cfg, root } => llama::loads_quantized(cfg, root, key),
            Self::Qwen35 { cfg, prefix } => qwen35::loads_quantized(cfg, prefix, key),
        }
    }
}

/// Replace every tensor the loader of this `config` quantizes at load ([`QuantizedTensors`]) by
/// its stored GGML block tensor at `q` — the representation a load-time quantization builds:
/// `q`'s type for the tensor's input dimension ([`QuantSpec::dtype_for_in_dim`]), each row of a
/// fused or stacked tensor quantized on its own. Everything else is left exactly as read. Returns
/// how many tensors were stored as blocks.
fn quantize_projections(
    tensors: &mut HashMap<String, Tensor>,
    q: QuantSpec,
    config: &Json,
) -> Result<usize> {
    let selection = QuantizedTensors::from_config(config, |key| tensors.contains_key(key))?;
    let mut keys: Vec<String> = tensors
        .keys()
        .filter(|k| selection.contains(k))
        .cloned()
        .collect();
    keys.sort(); // deterministic order so an error names the first offender stably
    for key in &keys {
        let w = &tensors[key];
        let dims = w.dims().to_vec();
        let [lead @ .., cols] = dims.as_slice() else {
            return Err(Error::Unsupported(format!(
                "prepare: `{key}` is a scalar, not a projection"
            )));
        };
        if lead.is_empty() {
            return Err(Error::Unsupported(format!(
                "prepare: `{key}` {dims:?} is not a projection matrix"
            )));
        }
        // A stacked `[experts, rows, cols]` weight quantizes as its `[experts * rows, cols]` rows.
        let rows: usize = lead.iter().product();
        let ggml = q.dtype_for_in_dim(*cols);
        let flat = w.to_dtype(DType::F32)?.reshape((rows, *cols))?;
        let qt = QTensor::quantize(&flat, ggml).map_err(|e| {
            Error::Unsupported(format!(
                "prepare: cannot quantize `{key}` {dims:?} to {ggml:?}: {e} — the input dimension \
                 must be a multiple of the block size (Q4_K=256, falling back to Q4_0=32; Q8_0=32)"
            ))
        })?;
        let blocks = crate::primitives::quant::to_ggml_block_tensor(&qt)?;
        let mut shape = lead.to_vec();
        shape.extend_from_slice(&blocks.dims()[1..]);
        tensors.insert(key.clone(), blocks.reshape(shape)?);
    }
    Ok(keys.len())
}

/// Stamp a `quantization` block (`{ "bits": 4 | 8 }`) into a `config.json` value so the loader
/// builds quantized projections; `"storage": "ggml"` is added when `stored` projections were
/// persisted as GGML block tensors (a snapshot that needs a block-aware loader).
fn stamp_quantization(config: &mut Json, q: QuantSpec, stored: usize) {
    if let Some(obj) = config.as_object_mut() {
        let mut block = Map::new();
        block.insert("bits".into(), Json::from(q.bits()));
        if stored > 0 {
            block.insert("storage".into(), Json::from(GGML_STORAGE));
        }
        obj.insert("quantization".into(), Json::Object(block));
    }
}

/// The `quantization.storage` tag of a snapshot whose projections are stored GGML blocks.
pub const GGML_STORAGE: &str = "ggml";

/// The format a prepared tier's `config.json` persists for its stored GGML blocks: the
/// `quantization` block (top level, or under `text_config`) when it says `storage: "ggml"` and
/// `bits` 4 or 8. `None` for any other config — a dense snapshot, or one prepared before sc-19375
/// (a bare block over dense-rounded weights). The qwen3_5 loader and its load admission honour
/// only this block, so an older qwen3_5 tier keeps loading dense exactly as it did.
pub(crate) fn persisted_ggml_blocks(config: &Json) -> Option<QuantSpec> {
    let block = config.get("quantization").or_else(|| {
        config
            .get("text_config")
            .and_then(|text| text.get("quantization"))
    })?;
    if block.get("storage").and_then(Json::as_str) != Some(GGML_STORAGE) {
        return None;
    }
    match block.get("bits").and_then(Json::as_u64)? {
        4 => Some(QuantSpec::q4()),
        8 => Some(QuantSpec::q8()),
        _ => None,
    }
}

/// Resolve a GGUF source to the `*.gguf` file: a file path is used directly; a directory is searched
/// for the first `*.gguf`.
fn resolve_gguf_path(source: &Path) -> Result<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    std::fs::read_dir(source)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
        .ok_or_else(|| Error::Unsupported(format!("prepare: no *.gguf in {}", source.display())))
}

/// Whether a `config.json` declares a `vision_config` (a multimodal snapshot the text preparer
/// declines). Any read/parse failure ⇒ `false` (the format probe handles non-snapshots).
fn has_vision_config(config_json: &Path) -> bool {
    std::fs::read_to_string(config_json)
        .ok()
        .and_then(|t| serde_json::from_str::<Json>(&t).ok())
        .map(|v| v.get("vision_config").is_some())
        .unwrap_or(false)
}

/// Count weight tensors across every `*.safetensors` shard in `dir` by reading only the headers (no
/// tensor data) — cheap enough for the dense passthrough's `num_tensors`.
fn count_safetensors_tensors(dir: &Path) -> Result<usize> {
    let mut total = 0;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            total += count_header_tensors(&path)?;
        }
    }
    Ok(total)
}

/// Count tensors in one safetensors file from its header (an 8-byte little-endian length prefix
/// followed by that many bytes of JSON), excluding the reserved `__metadata__` key.
fn count_header_tensors(path: &Path) -> Result<usize> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let v: Json = serde_json::from_slice(&header).map_err(|e| {
        Error::Msg(format!(
            "prepare: safetensors header {}: {e}",
            path.display()
        ))
    })?;
    let obj = v.as_object().ok_or_else(|| {
        Error::Msg(format!(
            "prepare: safetensors header {} is not an object",
            path.display()
        ))
    })?;
    Ok(obj.keys().filter(|k| k.as_str() != "__metadata__").count())
}

fn save_safetensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    candle_core::safetensors::save(tensors, path)
        .map_err(|e| Error::Msg(format!("prepare: write {}: {e}", path.display())))
}

fn read_json(path: &Path) -> Result<Json> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map_err(|e| Error::Config(format!("prepare: parse {}: {e}", path.display())))
}

fn write_json(path: &Path, value: &Json) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("prepare: serialize {}: {e}", path.display())))?;
    std::fs::write(path, text)?;
    Ok(())
}

/// Copy `name` from `src` to `out` when it exists (an optional sidecar like `tokenizer_config.json`).
fn copy_optional(src: &Path, out: &Path, name: &str) -> Result<()> {
    let from = src.join(name);
    if from.is_file() {
        std::fs::copy(&from, out.join(name))?;
    }
    Ok(())
}

pub const REGISTRATION: SnapshotPreparerRegistration = SnapshotPreparerRegistration {
    backend,
    can_prepare,
    prepare,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-19375: the preparer stores blocks for exactly the tensors the loader quantizes and load
    /// admission prices — the shared `loads_quantized` rule — so a projection-named tensor outside
    /// the decoder the loader builds (a layer past `num_hidden_layers`, another root) and every
    /// dense tensor (embeddings, head, norms, router) are left exactly as read.
    #[test]
    fn projection_selection_is_the_loaders() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 8,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0
        });
        let selected = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.1.mlp.gate_up_proj.weight",
            "model.layers.1.mlp.experts.7.up_proj.weight",
        ];
        let untouched = [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.gate.weight", // MoE router stays dense
            "model.layers.2.self_attn.q_proj.weight", // past num_hidden_layers
            "mtp.layers.0.self_attn.q_proj.weight", // another root
            "model.visual.blocks.0.attn.o_proj.weight",
        ];
        let mut tensors: HashMap<String, Tensor> = selected
            .iter()
            .chain(&untouched)
            .map(|k| {
                let t = Tensor::ones((2, 32), DType::BF16, &Device::Cpu).unwrap();
                (k.to_string(), t)
            })
            .collect();
        let stored = quantize_projections(&mut tensors, QuantSpec::q8(), &config).unwrap();
        assert_eq!(stored, selected.len());
        for k in selected {
            assert_eq!(tensors[k].dtype(), DType::U8, "{k} is stored as blocks");
            assert_eq!(tensors[k].dims(), &[2, 1, 34], "{k}");
        }
        for k in untouched {
            assert_eq!(tensors[k].dtype(), DType::BF16, "{k} is left as read");
        }
    }

    /// Only a `storage: "ggml"` block is a persisted block-storage format; a bare (pre-sc-19375)
    /// block is not.
    #[test]
    fn persisted_ggml_blocks_needs_the_storage_tag() {
        let with = |block: Json| serde_json::json!({ "quantization": block });
        assert_eq!(
            persisted_ggml_blocks(&with(serde_json::json!({"bits": 4, "storage": "ggml"}))),
            Some(QuantSpec::q4())
        );
        assert_eq!(
            persisted_ggml_blocks(&serde_json::json!({
                "text_config": {"quantization": {"bits": 8, "storage": "ggml"}}
            })),
            Some(QuantSpec::q8())
        );
        assert_eq!(
            persisted_ggml_blocks(&with(serde_json::json!({"bits": 8}))),
            None
        );
        assert_eq!(
            persisted_ggml_blocks(&with(serde_json::json!({"bits": 3, "storage": "ggml"}))),
            None
        );
    }

    #[test]
    fn stamps_quantization_block() {
        let mut cfg = serde_json::json!({ "hidden_size": 8 });
        stamp_quantization(&mut cfg, QuantSpec::q4(), 0);
        assert_eq!(cfg["quantization"], serde_json::json!({ "bits": 4 }));
        stamp_quantization(&mut cfg, QuantSpec::q8(), 3);
        assert_eq!(
            cfg["quantization"],
            serde_json::json!({ "bits": 8, "storage": "ggml" })
        );
    }
}
