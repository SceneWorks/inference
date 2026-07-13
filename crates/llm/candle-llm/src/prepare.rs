//! Persisted, backend-neutral model-snapshot preparation (story 7662): the candle backend's
//! [`core_llm::SnapshotPreparerRegistration`].
//!
//! `core-llm` owns format detection, the link-time preparer registry, and dispatch (story 7659); a
//! backend supplies the *tensor work* — turn a downloaded model (an HF-safetensors snapshot directory
//! or a `*.gguf` container) into a persisted, loadable snapshot, optionally re-quantizing the
//! projections on the way out. This is candle's peer of the mlx-llm impl: candle already passes
//! `core-llm` conformance, so a working preparer here de-provisionalizes the *convert+quantize* seam
//! across a second backend.
//!
//! # What it writes
//! A prepared snapshot is the HF shape [`load_for_model`](core_llm::load_for_model) already consumes:
//! `config.json` + `model.safetensors` (via [`candle_core::safetensors::save`]) + `tokenizer.json`
//! (and `tokenizer_config.json` when there is a chat template). Reading the dense tensors reuses the
//! loaders candle already has — HF via [`Weights::from_dir`], GGUF via [`GgufCheckpoint::open`]
//! (Candle's native reader dequantizes every GGML block type for free), so the writer is uniform
//! across both inputs.
//!
//! # How quantization is persisted
//! mlx-llm stores genuinely quantized tensors (packed `weight`/`scales`/`biases`). Candle's quantized
//! tensor (`QTensor`, GGML blocks) has **no safetensors representation**, so candle persists a Q4/Q8
//! snapshot as **dense weights carrying the quantization rounding** plus a `quantization` block in
//! `config.json`. The writer runs each attention/MLP **projection** through Candle's quantizer
//! ([`primitives::quant`](crate::primitives::quant)) and stores the dequantized (rounded) result;
//! embeddings, the LM head, and norms stay dense (the contract's tensor-level invariant). On load the
//! provider honors the `quantization` block and re-quantizes the projections via `QTensor`, so a
//! `LoadSpec::dense` of a prepared Q4/Q8 snapshot yields a genuinely quantized model — the same
//! observable result as mlx-llm, in candle's storage shape.
//!
//! Q4_K's block size is 256 and Q8_0's is 32, so a projection's input dimension must be a multiple of
//! that to quantize (Qwen3's 1024 is 256-aligned; SmolLM2's 576 is not — use Q8 there). A misaligned
//! projection is a clear error, not a silent dense fallback, matching quantize-on-load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use serde_json::{Map, Value as Json};

use core_llm::{
    detect_format, ModelFormat, PrepareReport, PrepareSpec, Quantize, Result as CoreResult,
    SnapshotPreparerRegistration,
};

use crate::error::{Error, Result};
use crate::gguf::GgufCheckpoint;
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

    let Some(q) = spec.quantize.map(quant_spec) else {
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
    let mut tensors = Weights::from_dir(src, &Device::Cpu)?.into_map();
    requant_projections(&mut tensors, q)?;

    std::fs::create_dir_all(&spec.out_dir)?;
    let mut config = read_json(&config_path)?;
    stamp_quantization(&mut config, q);
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
    let quant = spec.quantize.map(quant_spec);
    if let Some(q) = quant {
        requant_projections(&mut tensors, q)?;
        stamp_quantization(&mut config, q);
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
fn quant_spec(q: Quantize) -> QuantSpec {
    match q {
        Quantize::Q4 => QuantSpec::q4(),
        Quantize::Q8 => QuantSpec::q8(),
    }
}

/// Round each attention/MLP **projection** weight in place to `q`, by quantizing then dequantizing
/// via Candle's `QTensor` — the persisted weights then carry the quantization error, and the loader
/// re-quantizes them losslessly via the `quantization` config block. Embeddings, the LM head, and
/// norms (anything not ending `_proj.weight`) stay dense, per the contract's quant invariant.
fn requant_projections(tensors: &mut HashMap<String, Tensor>, q: QuantSpec) -> Result<()> {
    let mut keys: Vec<String> = tensors
        .keys()
        .filter(|k| is_quantizable_projection(k))
        .cloned()
        .collect();
    keys.sort(); // deterministic order so an error names the first offender stably
    for key in keys {
        let w = &tensors[&key];
        if w.rank() != 2 {
            continue;
        }
        let dtype = w.dtype();
        let qt = QTensor::quantize(&w.to_dtype(DType::F32)?, q.dtype).map_err(|e| {
            Error::Unsupported(format!(
                "prepare: cannot quantize `{key}` {:?} to {:?}: {e} — the input dimension must be a \
                 multiple of the block size (Q4_K=256, Q8_0=32); use Q8 for non-256-aligned models",
                w.dims(),
                q.dtype
            ))
        })?;
        let rounded = qt.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
        tensors.insert(key, rounded);
    }
    Ok(())
}

/// Whether a weight key is a quantizable projection: the attention/MLP projection matrices
/// (`q/k/v/o_proj`, `gate/up/down_proj`, packed `qkv_proj`/`gate_up_proj`, and MoE expert
/// projections) all end `_proj.weight`. Embeddings (`embed_tokens.weight`), the LM head
/// (`lm_head.weight`), norms, and the MoE router (`mlp.gate.weight`) do not, so they stay dense —
/// matching the decoder's quantize-on-load decisions for the dense families.
fn is_quantizable_projection(key: &str) -> bool {
    key.ends_with("_proj.weight")
}

/// Stamp a `quantization` block (`{ "bits": 4 | 8 }`) into a `config.json` value so the loader
/// re-quantizes the projections on load.
fn stamp_quantization(config: &mut Json, q: QuantSpec) {
    if let Some(obj) = config.as_object_mut() {
        let mut block = Map::new();
        block.insert("bits".into(), Json::from(q.bits()));
        obj.insert("quantization".into(), Json::Object(block));
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

// Compatibility registration for consumers that have not adopted an explicit runtime bundle.
inventory::submit! { REGISTRATION }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_classification() {
        for k in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.2.self_attn.qkv_proj.weight",
            "model.layers.5.mlp.experts.7.up_proj.weight",
        ] {
            assert!(is_quantizable_projection(k), "{k} should be quantizable");
        }
        for k in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.mlp.gate.weight", // MoE router stays dense
        ] {
            assert!(!is_quantizable_projection(k), "{k} should stay dense");
        }
    }

    #[test]
    fn stamps_quantization_block() {
        let mut cfg = serde_json::json!({ "hidden_size": 8 });
        stamp_quantization(&mut cfg, QuantSpec::q4());
        assert_eq!(cfg["quantization"]["bits"], serde_json::json!(4));
        stamp_quantization(&mut cfg, QuantSpec::q8());
        assert_eq!(cfg["quantization"]["bits"], serde_json::json!(8));
    }
}
