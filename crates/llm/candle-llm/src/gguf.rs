//! GGUF checkpoint loading (story 7254): load a llama.cpp `*.gguf` directly into the Candle decoder.
//!
//! Where `mlx-llm` had to hand-roll the GGUF block dequantizers and emit a converted snapshot
//! (story 7165), Candle ships a native reader — [`candle_core::quantized::gguf_file`] — that parses
//! the container and dequantizes every GGML block type (legacy `Q*_0/_1`, the `Q*_K` k-quants, and
//! the `IQ*` i-quants) for free. So this module is a *direct loader*, not a converter: it
//!
//! 1. reads the container ([`candle_core::quantized::gguf_file::Content`]),
//! 2. dequantizes each tensor to dense and **remaps** its llama.cpp name (`blk.0.attn_q.weight`) to
//!    the transformer key the decoder loads (`model.layers.0.self_attn.q_proj.weight`),
//! 3. **un-permutes** the Llama/Mistral q/k projections (llama.cpp interleaves their rows for its
//!    `NORM` RoPE; the engine uses HF's half-split RoPE — without this the model emits garbage),
//! 4. reconstructs a [`ModelConfig`] from the GGUF metadata table, and
//! 5. optionally re-quantizes the projections on load (the same quantize-on-load path as a
//!    safetensors load, story 7163).
//!
//! The result is a [`Weights`] map + [`ModelConfig`] the existing
//! [`crate::CausalLm::from_weights_with`]
//! consumes unchanged, so a GGUF load and a safetensors load converge on one decoder. The tokenizer
//! is taken from a sibling `tokenizer.json` when present, else reconstructed from the GGUF's embedded
//! tokenizer metadata ([`GgufCheckpoint::tokenizer_from_metadata`]).

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{Device, Tensor};
use serde_json::{json, Map, Value as Json};

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::primitives::Weights;

/// A GGUF checkpoint parsed into the engine's native shapes: a dense [`Weights`] map + a
/// [`ModelConfig`], plus the tokenizer/template/stop metadata the provider needs to run it.
pub struct GgufCheckpoint {
    /// Model config reconstructed from the GGUF metadata table.
    pub config: ModelConfig,
    /// The HF-shaped `config.json` value the [`config`](Self::config) was parsed from — retained so
    /// the snapshot writer (story 7662) can serialize it back out for a converted snapshot.
    pub config_json: Json,
    /// Dense, transformer-keyed weights (q/k already un-permuted to the HF layout).
    pub weights: Weights,
    /// Stop token ids from `tokenizer.ggml.eos_token_id` (+ EOT when distinct); empty if absent.
    pub stop_tokens: Vec<i32>,
    /// The model's own Jinja `chat_template` metadata string, if the GGUF carries one.
    pub chat_template: Option<String>,
    /// The BOS / EOS token *strings* (for rendering a Jinja template that references them).
    pub bos_token: Option<String>,
    /// See [`GgufCheckpoint::bos_token`].
    pub eos_token: Option<String>,
    // --- raw tokenizer metadata, retained for `tokenizer_from_metadata` ---
    tok_model: Option<String>,
    tok_tokens: Vec<String>,
    tok_merges: Vec<String>,
    tok_types: Vec<i32>,
}

impl GgufCheckpoint {
    /// Open and load a `.gguf` file onto `device`: dequantize every tensor to dense, remap keys, and
    /// un-permute the Llama/Mistral q/k projections. Load-time re-quantization (`spec.quantize`) is
    /// applied by the caller via [`crate::CausalLm::from_weights_with`], exactly as for a
    /// safetensors load.
    pub fn open(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)
            .map_err(|e| Error::Msg(format!("gguf: open {}: {e}", path.display())))?;
        let content = Content::read(&mut file)
            .map_err(|e| Error::Msg(format!("gguf: parse {}: {e}", path.display())))?;
        let meta = &content.metadata;

        let arch = meta_str(meta, "general.architecture")
            .ok_or_else(|| Error::Config("gguf: missing general.architecture".into()))?
            .to_string();
        // llama.cpp interleaves q/k rows for the Llama/Mistral RoPE (`rope_type=NORM`); Qwen3 keeps
        // the HF half-split layout (`rope_type=NEOX`) and so is not permuted.
        let permute_qk = arch == "llama";
        let mkey = |s: &str| format!("{arch}.{s}");
        let num_heads = meta_u64(meta, &mkey("attention.head_count")).ok_or_else(|| {
            Error::Config(format!(
                "gguf: missing metadata {}",
                mkey("attention.head_count")
            ))
        })? as usize;
        let num_kv_heads =
            meta_u64(meta, &mkey("attention.head_count_kv")).unwrap_or(num_heads as u64) as usize;

        // --- remap + dequantize every tensor to a dense, transformer-keyed Weights map ---
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut unmapped: Vec<String> = Vec::new();
        // Deterministic order so an error message (and any future logging) is stable.
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in names {
            let Some(hf_key) = remap_key(name) else {
                if !is_ignorable(name) {
                    unmapped.push(name.clone());
                }
                continue;
            };
            let q = content
                .tensor(&mut file, name, device)
                .map_err(|e| Error::Msg(format!("gguf: read tensor {name}: {e}")))?;
            let mut dense = q.dequantize(device)?;
            if permute_qk && hf_key.ends_with("self_attn.q_proj.weight") {
                dense = unpermute_qk(&dense, num_heads)?;
            } else if permute_qk && hf_key.ends_with("self_attn.k_proj.weight") {
                dense = unpermute_qk(&dense, num_kv_heads)?;
            }
            tensors.insert(hf_key, dense);
        }
        if !unmapped.is_empty() {
            return Err(Error::Unsupported(format!(
                "gguf: {} tensor(s) with no transformer-key mapping: {}",
                unmapped.len(),
                unmapped.join(", ")
            )));
        }

        // vocab from the embedding rows ([vocab, hidden]) — the most reliable source.
        let vocab = tensors
            .get("model.embed_tokens.weight")
            .ok_or_else(|| Error::Config("gguf: no token embedding tensor".into()))?
            .dims()
            .first()
            .copied()
            .ok_or_else(|| Error::Config("gguf: token embedding has no rows".into()))?
            as i64;
        // lm_head tied iff the GGUF has no separate output projection.
        let tied = !tensors.contains_key("lm_head.weight");

        let (config, config_json) =
            reconstruct_config(meta, &arch, num_heads, num_kv_heads, vocab, tied)?;
        let weights = Weights::from_map(tensors, device.clone());

        // --- tokenizer / template / stop-token metadata ---
        let stop_tokens = stop_tokens_from_meta(meta, &arch);
        let chat_template = meta_str(meta, "tokenizer.chat_template").map(str::to_string);
        let tok_tokens = meta_str_array(meta, "tokenizer.ggml.tokens").unwrap_or_default();
        let bos_token = meta_u64(meta, "tokenizer.ggml.bos_token_id")
            .and_then(|i| tok_tokens.get(i as usize).cloned());
        let eos_token = meta_u64(meta, "tokenizer.ggml.eos_token_id")
            .and_then(|i| tok_tokens.get(i as usize).cloned());

        Ok(Self {
            config,
            config_json,
            weights,
            stop_tokens,
            chat_template,
            bos_token,
            eos_token,
            tok_model: meta_str(meta, "tokenizer.ggml.model").map(str::to_string),
            tok_tokens,
            tok_merges: meta_str_array(meta, "tokenizer.ggml.merges").unwrap_or_default(),
            tok_types: meta_i32_array(meta, "tokenizer.ggml.token_type").unwrap_or_default(),
        })
    }

    /// Reconstruct the tokenizer from the GGUF's embedded metadata (the `else` branch when there is
    /// no sibling `tokenizer.json`). Supports byte-level **BPE** tokenizers (`tokenizer.ggml.model`
    /// `gpt2` / `llama-bpe` — the form Llama-3 / Qwen / SmolLM2 GGUFs ship); a SentencePiece/unigram
    /// GGUF (`model = "llama"`, scores-only, no merges) is rejected with a clear message asking for a
    /// sibling `tokenizer.json`.
    pub fn tokenizer_from_metadata(&self) -> Result<core_llm::Tokenizer> {
        let json = self.tokenizer_json_from_metadata()?;
        core_llm::Tokenizer::from_json(&json).map_err(|e| Error::Msg(format!("gguf: {e}")))
    }

    /// Build the HF `tokenizer.json` string from the GGUF's embedded tokenizer metadata (the source
    /// for [`tokenizer_from_metadata`](Self::tokenizer_from_metadata), exposed so the snapshot writer
    /// (story 7662) can persist a self-contained `tokenizer.json` when converting a GGUF). Errors on a
    /// tokenizer kind that can't be rebuilt from GGUF (no tokens, or SentencePiece/unigram with no
    /// merges) — the caller should supply a sibling `tokenizer.json` in that case.
    pub fn tokenizer_json_from_metadata(&self) -> Result<String> {
        if self.tok_tokens.is_empty() {
            return Err(Error::Unsupported(
                "gguf: no tokenizer.ggml.tokens metadata; provide a sibling tokenizer.json".into(),
            ));
        }
        if self.tok_merges.is_empty() {
            return Err(Error::Unsupported(format!(
                "gguf: tokenizer model {:?} has no BPE merges (likely SentencePiece/unigram); \
                 provide a sibling tokenizer.json",
                self.tok_model.as_deref().unwrap_or("?")
            )));
        }
        Ok(self.build_bpe_tokenizer_json())
    }

    /// Build a HF `tokenizer.json` (byte-level BPE) string from the retained GGUF tokenizer metadata.
    fn build_bpe_tokenizer_json(&self) -> String {
        // GGML token-type tags: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 5=UNUSED 6=BYTE.
        // CONTROL / USER_DEFINED ids are the special/added tokens (e.g. `<|im_start|>`), surfaced as
        // `added_tokens` with `special:true` so encode maps them whole and decode can skip them.
        let mut added = Vec::new();
        let mut vocab = Map::new();
        for (id, tok) in self.tok_tokens.iter().enumerate() {
            vocab.insert(tok.clone(), json!(id));
            let ty = self.tok_types.get(id).copied().unwrap_or(1);
            if ty == 3 || ty == 4 {
                added.push(json!({
                    "id": id,
                    "content": tok,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                }));
            }
        }
        let merges: Vec<Json> = self.tok_merges.iter().map(|m| json!(m)).collect();
        let doc = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false,
                "use_regex": true
            },
            "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges
            }
        });
        doc.to_string()
    }
}

/// Whether `source` should be loaded as a single GGUF file (vs an HF snapshot directory): a path
/// whose extension is `gguf` (case-insensitive).
pub fn is_gguf_path(source: &str) -> bool {
    Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

/// Read **only** the GGUF header/metadata table — never a tensor block — and return the
/// `general.architecture` string, or `None` if `path` can't be opened/parsed or omits the key.
///
/// Candle's `Content::read` parses the magic, the metadata KV table, and the tensor-info table
/// (names/shapes/offsets) and then stops at `tensor_data_offset`; it never reads a tensor block. So
/// this probe is **weightless**: a GGUF truncated right after its metadata still resolves. The text
/// provider's `can_load` uses it to confirm a `.gguf`'s architecture before claiming the file,
/// mirroring the safetensors path's `config.json` probe (story 7420).
pub fn gguf_architecture(path: impl AsRef<Path>) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let content = Content::read(&mut file).ok()?;
    meta_str(&content.metadata, "general.architecture").map(str::to_string)
}

/// Map a GGUF `general.architecture` to the `(model_type, architectures[0])` HF identifiers the
/// engine's decoder dispatch understands, or `None` if this provider's GGUF loader can't reconstruct
/// it. llama.cpp labels both Llama and Mistral `"llama"`; Qwen3 is `"qwen3"`. Every other GGUF
/// architecture (`qwen2`, `gemma2`, `phi3`, a non-LLM `bert`/`clip`, …) has no GGUF reconstruction
/// path and is declined.
///
/// `reconstruct_config` (the loader) and the provider's `can_load` (the weightless resolve probe)
/// both consult this one mapping, so the "can this load?" and "does this load?" decisions can never
/// drift apart.
pub fn gguf_arch_to_hf(arch: &str) -> Option<(&'static str, &'static str)> {
    match arch {
        "llama" => Some(("llama", "LlamaForCausalLM")),
        "qwen3" => Some(("qwen3", "Qwen3ForCausalLM")),
        _ => None,
    }
}

/// Map a GGML tensor name to the transformer (HF) key the decoder loads, or `None` if it is not a
/// weight the engine consumes.
pub fn remap_key(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<usize>().ok()?;
    let hf_suffix = match suffix {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight", // Qwen3
        "attn_k_norm.weight" => "self_attn.k_norm.weight", // Qwen3
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{hf_suffix}"))
}

/// GGML tensors the engine recomputes itself and so ignores rather than rejecting as unmapped.
fn is_ignorable(name: &str) -> bool {
    // Llama-3 ships precomputed RoPE frequencies; the engine derives RoPE from theta/scaling.
    name == "rope_freqs.weight" || name.ends_with(".rope_freqs.weight")
}

/// Un-permute a llama.cpp q/k projection weight back to the HF half-split RoPE layout.
///
/// `convert_hf_to_gguf.py` reorders each head's `head_dim` rows
/// (`reshape(n_head, 2, hd/2).swapaxes(1, 2)`) so llama.cpp's interleaved RoPE matches HF's
/// half-split RoPE. The inverse gather restores the HF order: for HF row `r = k·(hd/2) + j` in a
/// head, the source GGUF row is `2·j + k`. Applied per head to a `[out, in]` weight where
/// `out = n_head · head_dim`, via an on-device row gather.
fn unpermute_qk(weight: &Tensor, n_head: usize) -> Result<Tensor> {
    let out = weight.dim(0)?;
    if n_head == 0 || !out.is_multiple_of(n_head) {
        return Err(Error::Msg(format!(
            "gguf: q/k permute: out {out} not divisible by n_head {n_head}"
        )));
    }
    let head_dim = out / n_head;
    if !head_dim.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "gguf: q/k permute: odd head_dim {head_dim}"
        )));
    }
    let half = head_dim / 2;
    let mut idx = Vec::with_capacity(out);
    for h in 0..n_head {
        for r in 0..head_dim {
            let (k, j) = (r / half, r % half);
            idx.push((h * head_dim + 2 * j + k) as u32);
        }
    }
    let idx = Tensor::from_vec(idx, (out,), weight.device())?;
    Ok(weight.index_select(&idx, 0)?)
}

/// Rebuild a [`ModelConfig`] from the GGUF metadata table by assembling the HF `config.json` fields
/// it implies and reusing [`ModelConfig::from_json`] (so arch dispatch / RoPE scaling parsing match a
/// safetensors load exactly).
fn reconstruct_config(
    meta: &HashMap<String, Value>,
    arch: &str,
    num_heads: usize,
    num_kv_heads: usize,
    vocab: i64,
    tied: bool,
) -> Result<(ModelConfig, Json)> {
    let key = |s: &str| format!("{arch}.{s}");
    let req_u64 = |s: &str| -> Result<u64> {
        meta_u64(meta, &key(s))
            .ok_or_else(|| Error::Config(format!("gguf: missing metadata {}", key(s))))
    };
    let (model_type, hf_arch) = gguf_arch_to_hf(arch).ok_or_else(|| {
        Error::Unsupported(format!(
            "gguf architecture {arch:?} (engine supports llama/mistral and qwen3; \
             mistral GGUFs are labelled \"llama\")"
        ))
    })?;

    let hidden = req_u64("embedding_length")? as i64;
    let blocks = req_u64("block_count")? as i64;
    let ffn = req_u64("feed_forward_length")? as i64;
    let head_dim = meta_u64(meta, &key("attention.key_length"))
        .map(|v| v as i64)
        .unwrap_or(hidden / num_heads as i64);
    let rms_eps = meta_f64(meta, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
    let rope_theta = meta_f64(meta, &key("rope.freq_base")).unwrap_or(10000.0);
    let context = meta_u64(meta, &key("context_length")).unwrap_or(0) as i64;

    let mut cfg = Map::new();
    cfg.insert("architectures".into(), json!([hf_arch]));
    cfg.insert("model_type".into(), json!(model_type));
    cfg.insert("hidden_size".into(), json!(hidden));
    cfg.insert("intermediate_size".into(), json!(ffn));
    cfg.insert("num_hidden_layers".into(), json!(blocks));
    cfg.insert("num_attention_heads".into(), json!(num_heads));
    cfg.insert("num_key_value_heads".into(), json!(num_kv_heads));
    cfg.insert("head_dim".into(), json!(head_dim));
    cfg.insert("vocab_size".into(), json!(vocab));
    cfg.insert("rms_norm_eps".into(), json!(rms_eps));
    cfg.insert("rope_theta".into(), json!(rope_theta));
    cfg.insert("tie_word_embeddings".into(), json!(tied));
    if context > 0 {
        cfg.insert("max_position_embeddings".into(), json!(context));
    }
    if let Some(scaling) = reconstruct_rope_scaling(meta, arch) {
        cfg.insert("rope_scaling".into(), scaling);
    }
    let json = Json::Object(cfg);
    let config = ModelConfig::from_json(&json)?;
    Ok((config, json))
}

/// Best-effort llama3 RoPE-scaling reconstruction from GGUF metadata; absent keys ⇒ standard RoPE.
fn reconstruct_rope_scaling(meta: &HashMap<String, Value>, arch: &str) -> Option<Json> {
    let key = |s: &str| format!("{arch}.{s}");
    let low = meta_f64(meta, &key("rope.scaling.low_freq_factor"));
    let high = meta_f64(meta, &key("rope.scaling.high_freq_factor"));
    let scaling_type = meta_str(meta, &key("rope.scaling.type"));
    if scaling_type != Some("llama3") && low.is_none() && high.is_none() {
        return None;
    }
    let factor = meta_f64(meta, &key("rope.scaling.factor")).unwrap_or(8.0);
    let orig = meta_u64(meta, &key("rope.scaling.original_context_length")).unwrap_or(8192) as f64;
    Some(json!({
        "rope_type": "llama3",
        "factor": factor,
        "low_freq_factor": low.unwrap_or(1.0),
        "high_freq_factor": high.unwrap_or(4.0),
        "original_max_position_embeddings": orig,
    }))
}

/// Stop ids from the GGUF tokenizer metadata: `eos_token_id` plus a distinct EOT id if present.
fn stop_tokens_from_meta(meta: &HashMap<String, Value>, _arch: &str) -> Vec<i32> {
    let mut stop = Vec::new();
    if let Some(eos) = meta_u64(meta, "tokenizer.ggml.eos_token_id") {
        stop.push(eos as i32);
    }
    if let Some(eot) = meta_u64(meta, "tokenizer.ggml.eot_token_id") {
        let eot = eot as i32;
        if !stop.contains(&eot) {
            stop.push(eot);
        }
    }
    stop
}

// --- GGUF metadata accessors (coerce across the integer/float width variants) ---

fn meta_u64(meta: &HashMap<String, Value>, key: &str) -> Option<u64> {
    Some(match meta.get(key)? {
        Value::U8(v) => *v as u64,
        Value::U16(v) => *v as u64,
        Value::U32(v) => *v as u64,
        Value::U64(v) => *v,
        Value::I8(v) if *v >= 0 => *v as u64,
        Value::I16(v) if *v >= 0 => *v as u64,
        Value::I32(v) if *v >= 0 => *v as u64,
        Value::I64(v) if *v >= 0 => *v as u64,
        Value::Bool(b) => *b as u64,
        _ => return None,
    })
}

fn meta_f64(meta: &HashMap<String, Value>, key: &str) -> Option<f64> {
    match meta.get(key)? {
        Value::F32(v) => Some(*v as f64),
        Value::F64(v) => Some(*v),
        _ => None,
    }
}

fn meta_str<'a>(meta: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    match meta.get(key)? {
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn meta_str_array(meta: &HashMap<String, Value>, key: &str) -> Option<Vec<String>> {
    match meta.get(key)? {
        Value::Array(a) => Some(
            a.iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

fn meta_i32_array(meta: &HashMap<String, Value>, key: &str) -> Option<Vec<i32>> {
    match meta.get(key)? {
        Value::Array(a) => Some(
            a.iter()
                .map(|v| match v {
                    Value::I8(x) => *x as i32,
                    Value::I16(x) => *x as i32,
                    Value::I32(x) => *x,
                    Value::U8(x) => *x as i32,
                    Value::U16(x) => *x as i32,
                    Value::U32(x) => *x as i32,
                    _ => 0,
                })
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn detects_gguf_paths() {
        assert!(is_gguf_path("model.gguf"));
        assert!(is_gguf_path("/x/y/Model-Q4_K_M.GGUF"));
        assert!(!is_gguf_path("/x/snapshot-dir"));
        assert!(!is_gguf_path("model.safetensors"));
    }

    #[test]
    fn remaps_non_layer_keys() {
        assert_eq!(
            remap_key("token_embd.weight").unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            remap_key("output_norm.weight").unwrap(),
            "model.norm.weight"
        );
        assert_eq!(remap_key("output.weight").unwrap(), "lm_head.weight");
    }

    #[test]
    fn remaps_layer_keys() {
        assert_eq!(
            remap_key("blk.0.attn_q.weight").unwrap(),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            remap_key("blk.13.ffn_down.weight").unwrap(),
            "model.layers.13.mlp.down_proj.weight"
        );
        assert_eq!(
            remap_key("blk.5.attn_norm.weight").unwrap(),
            "model.layers.5.input_layernorm.weight"
        );
        assert_eq!(
            remap_key("blk.2.attn_q_norm.weight").unwrap(),
            "model.layers.2.self_attn.q_norm.weight"
        );
    }

    #[test]
    fn unknown_keys_and_ignorables() {
        assert!(remap_key("blk.0.some_future_tensor.weight").is_none());
        assert!(remap_key("rope_freqs.weight").is_none());
        assert!(is_ignorable("rope_freqs.weight"));
        assert!(is_ignorable("blk.0.rope_freqs.weight"));
        assert!(!is_ignorable("blk.0.attn_q.weight"));
    }

    /// The q/k un-permute must invert llama.cpp's forward row permutation exactly (so a GGUF Llama
    /// projection lands back in the HF half-split layout the decoder's RoPE expects).
    #[test]
    fn unpermute_qk_inverts_llama_cpp_forward() {
        // Forward (HF -> GGUF) per head: dst row (2j+k) <- src row (k*half + j).
        fn forward(data: &[f32], out: usize, in_dim: usize, n_head: usize) -> Vec<f32> {
            let head_dim = out / n_head;
            let half = head_dim / 2;
            let mut res = vec![0f32; data.len()];
            for h in 0..n_head {
                for k in 0..2 {
                    for j in 0..half {
                        let dst = h * head_dim + (2 * j + k);
                        let src = h * head_dim + (k * half + j);
                        res[dst * in_dim..dst * in_dim + in_dim]
                            .copy_from_slice(&data[src * in_dim..src * in_dim + in_dim]);
                    }
                }
            }
            res
        }
        let (n_head, head_dim, in_dim) = (2usize, 4usize, 3usize);
        let out = n_head * head_dim;
        let hf: Vec<f32> = (0..(out * in_dim) as i32).map(|x| x as f32).collect();
        let gguf = forward(&hf, out, in_dim, n_head);
        assert_ne!(gguf, hf, "forward permute should be non-trivial");

        let gguf_t = Tensor::from_vec(gguf, (out, in_dim), &Device::Cpu).unwrap();
        let back = unpermute_qk(&gguf_t, n_head)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(back, hf, "inverse permute must recover the HF layout");
    }
}
