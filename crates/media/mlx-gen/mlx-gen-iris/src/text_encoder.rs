//! Qwen3-VL text conditioning — the port of `iris3b/text/qwen3_vl.py` `Qwen3VLTextEncoder`.
//!
//! * The chat-template prefix, the caption and the suffix are tokenized **separately**
//!   (`add_special_tokens=False`) and concatenated; the caption alone is cut to
//!   `max_length − len(suffix)` so the assistant-turn marker always survives
//!   ([`gen_core::iris::assemble_window`]).
//! * The language tower is the shared generic decoder [`mlx_llm::CausalLm`] (the `qwen3_vl`
//!   architecture: GQA, per-head q/k RMSNorm, SwiGLU, θ = 5e6; interleaved mRoPE collapses to 1-D
//!   for a text-only prompt). It runs in its own compute dtype (bf16), matching the release's
//!   `text_encoder.dtype: bfloat16`.
//! * The conditioning is the stack of the 1-based post-block hidden states `hidden_layers`
//!   (HF `output_hidden_states` indexing) over the window `[len(prefix), len(prefix) + max_length)`,
//!   with every pad row zeroed — `[1, max_length, len(hidden_layers), dim]`.
//!
//! Upstream right-pads the batch with the pad token and runs the padded sequence; with a causal
//! tower and trailing pads the real rows never see a pad and the pad rows are zeroed by the mask,
//! so this port runs only the real tokens and writes zeros — bit-for-bit the same conditioning
//! without the pad compute.

use std::collections::HashMap;
use std::path::Path;

use gen_core::iris::{
    assemble_window, TextEncoderConfig, TextWindow, PROMPT_PREFIX, PROMPT_SUFFIX,
};
use gen_core::safetensors_shards::{resolve_indexed_safetensors_shards, snapshot_shard_roots};
use gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::gen_core;
use mlx_gen::{Error, Quant, Result};
use mlx_llm::primitives::projection::{QuantGrid, QuantSpec};
use mlx_llm::{CausalLm, ModelConfig};
use mlx_rs::ops::{concatenate_axis, stack_axis, zeros};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Labels of the text encoder's iOS execution quanta.
pub(crate) mod quanta {
    pub const LOAD: &str = "Iris text encoder load";
    pub const TOWER: &str = "Iris text tower";
    pub const EMBEDDING: &str = "Iris text embedding";
    pub const LAYER: &str = "Iris text layer";
    /// Checked from inside the shared decoder stack before each layer's graph is built.
    pub const DECODER_LAYER: &str = "Iris text decoder layer";
    pub const CONDITIONING: &str = "Iris text conditioning";
    /// A streamed tower's window read (rung 4): the window's decoder layers read and built.
    pub const LAYER_READ: &str = "Iris text layer read";
    /// One streamed decoder layer's forward (rung 4).
    pub const STREAMED_LAYER: &str = "Iris text streamed layer";
    /// Every label above, for the phase attribution's coverage test.
    pub const ALL: [&str; 8] = [
        LOAD,
        TOWER,
        EMBEDDING,
        LAYER,
        DECODER_LAYER,
        CONDITIONING,
        LAYER_READ,
        STREAMED_LAYER,
    ];
}

/// Activation width of the shared decoder (`mlx_llm` computes in bf16 whatever the file stores).
const DECODER_COMPUTE_BYTES: u64 = 2;

/// One prompt's conditioning, as `TextEncoding` upstream.
pub struct TextConditioning {
    /// `[1, max_length, layers, dim]` f32, pad rows zero.
    pub states: Array,
    /// `[max_length]` 0/1 flags.
    pub mask: Vec<i32>,
    /// Caption tokens dropped by the budget (warned at encode under `on_caption_overflow: warn`).
    pub truncated_tokens: usize,
}

/// The Qwen3-VL language tower: the shared decoder holding every layer, or (rung 4) the layers
/// streamed from the snapshot one window at a time.
enum Tower {
    Resident(Box<CausalLm>),
    Streamed(Box<StreamedTower>),
}

/// The loaded Qwen3-VL language tower + tokenizer + the backbone's conditioning contract.
pub struct IrisTextEncoder {
    tower: Tower,
    tokenizer: TextTokenizer,
    prefix_ids: Vec<i32>,
    suffix_ids: Vec<i32>,
    hidden_layers: Vec<usize>,
    max_length: usize,
    dim: usize,
    /// The checkpoint's `text_encoder.on_caption_overflow` policy (the release's is `warn`).
    overflow: CaptionOverflowPolicy,
}

use gen_core::iris::apply_caption_overflow_policy;
/// The `on_caption_overflow: warn` message for one window — backend-neutral, shared with the
/// Candle twin through [`gen_core::iris`].
pub use gen_core::iris::caption_overflow_warning;
use gen_core::CaptionOverflowPolicy;

fn from_llm(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(m) => Error::Unsupported(m),
        mlx_llm::Error::MissingTensor(k) => Error::MissingTensor(k),
        mlx_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(format!("iris text encoder: {other}")),
    }
}

/// Key prefix of the Qwen3-VL language tower inside the VLM checkpoint.
pub(crate) const LANGUAGE_TOWER_PREFIX: &str = "model.language_model.";
/// The untied output head, at the checkpoint root.
pub(crate) const LANGUAGE_HEAD_KEY: &str = "lm_head.weight";

/// Whether the text-only encoder reads tensor `name`: the language tower, and the output head when
/// it is untied. The vision tower (`model.visual.*`) is never read on any Iris route.
pub(crate) fn is_text_tower_tensor(name: &str, untied_head: bool) -> bool {
    name.starts_with(LANGUAGE_TOWER_PREFIX) || (untied_head && name == LANGUAGE_HEAD_KEY)
}

/// Every safetensors file of the snapshot at `dir`: the shards its `model.safetensors.index.json`
/// names, else the single `model.safetensors`. Each named path is validated (inside the snapshot,
/// a regular file) before any is returned.
fn snapshot_shards(dir: &Path) -> Result<Option<Vec<std::path::PathBuf>>> {
    let roots = snapshot_shard_roots(dir).map_err(Error::from)?;
    Ok(
        resolve_indexed_safetensors_shards(dir, "model.safetensors.index.json", &roots)
            .map_err(Error::from)?
            .map(|shards| shards.into_iter().map(|s| s.loader_path).collect()),
    )
}

/// The safetensors files [`IrisTextEncoder::load`] opens for the snapshot at `dir`: of the shards
/// the index names, only those holding a tensor [`is_text_tower_tensor`] reads (a vision-only
/// shard is never opened); without an index, the single `model.safetensors`. Shared with the
/// memory contract's asset pricing ([`crate::memory_strategy`]) so both read one file list.
pub(crate) fn weight_shards(dir: &Path, untied_head: bool) -> Result<Vec<std::path::PathBuf>> {
    let Some(all) = snapshot_shards(dir)? else {
        return Ok(vec![dir.join("model.safetensors")]);
    };
    let index = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .map_err(|e| Error::Msg(format!("iris: reading {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("iris: parsing {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::Msg(format!("iris: {} has no weight_map", index.display())))?;
    let used: std::collections::BTreeSet<std::path::PathBuf> = map
        .iter()
        .filter(|(name, _)| is_text_tower_tensor(name, untied_head))
        .filter_map(|(_, shard)| shard.as_str().map(|shard| dir.join(shard)))
        .collect();
    Ok(all.into_iter().filter(|path| used.contains(path)).collect())
}

/// The group size of a quantized text tower (sc-26266): the codebase default 64 when every
/// projection input width (`hidden_size` for q/k/v/gate/up, `heads · head_dim` for o, the MLP width
/// for down) divides it, else 32 — the shared decoder packs every projection at ONE group size.
/// `None` when not even 32 divides them all: such a tower has no affine tier.
pub fn text_encoder_quant_group(cfg: &ModelConfig) -> Option<i32> {
    let widths = [
        cfg.hidden_size,
        cfg.num_heads * cfg.head_dim,
        cfg.intermediate_size,
    ];
    [64, 32]
        .into_iter()
        .find(|group| widths.iter().all(|width| *width > 0 && width % group == 0))
}

/// The load-time quantization of the text tower at `quant` (`None` is the dense tier).
pub fn text_encoder_quant_spec(
    cfg: &ModelConfig,
    quant: Option<Quant>,
) -> Result<Option<QuantSpec>> {
    let bits = match quant {
        None => return Ok(None),
        Some(Quant::Q4) => 4,
        Some(Quant::Q8) => 8,
        Some(other) => {
            return Err(Error::Unsupported(format!(
            "iris: the Qwen3-VL text tower has the q8 and q4 affine tiers; {other:?} is not one \
                 of them"
        )))
        }
    };
    let group_size = text_encoder_quant_group(cfg).ok_or_else(|| {
        Error::Unsupported(format!(
            "iris: the text tower's projection widths (hidden {}, attention {}, MLP {}) share no \
             affine group size (64 or 32), so it has no quantized tier",
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            cfg.intermediate_size
        ))
    })?;
    // Codes on the stored bf16 grid (`QuantGrid::Stored`): the same packed bytes as MLX's
    // `quantize`, 0.64 x its weight error at q8 on this tower (see `nn::pack_projection`).
    Ok(Some(QuantSpec {
        group_size,
        bits,
        grid: QuantGrid::Stored,
    }))
}

/// [`IrisTextEncoder::conditioning_scratch`] from the conditioning contract alone.
pub(crate) fn conditioning_bytes(layers: u64, max_length: u64, dim: u64, real: i32) -> u64 {
    use crate::retention::{Kind, Retention};
    let real = real.max(0) as u64;
    let pad = max_length.saturating_sub(real);
    let mut r = Retention::on(4, false);
    r.alloc(Kind::Temp, layers, 4, real * dim)
        .temp(4, real * layers * dim);
    if pad > 0 {
        r.temp(4, pad * layers * dim)
            .temp(4, max_length * layers * dim);
    }
    r.price()
}

/// [`IrisTextEncoder::layer_scratch_parts`] from the tower's config alone, on the current device.
pub(crate) fn decoder_layer_parts(cfg: &ModelConfig, n: i32, k: usize) -> (u64, u64) {
    decoder_layer_parts_on(cfg, n, k, crate::retention::unfused())
}

/// [`decoder_layer_parts`] pricing the unfused fallbacks when `unfused`.
///
/// One `mlx_llm` decoder layer is one graph (no barrier inside), so every buffer it allocates is
/// counted under MLX's retention ([`crate::retention`]), at the decoder's bf16 compute width over
/// `n` tokens:
/// * each RMSNorm (input, post-attention, QK): fused, its result (a QK-norm's operand is a
///   single-consumer projection, which it reuses); unfused, the f32 cast, `square`, the f32
///   product (its operand is still read by the `square` in flight), the per-row mean and the bf16
///   result;
/// * the q / k / v projections, and per RoPE (q and k, NeoX halves) the negated half, the rotated
///   concatenation and `x·cos` (the sine product and the sum donate);
/// * the layer's K/V cache block — `mlx_llm`'s contiguous cache allocates
///   [`mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS`]-position zero blocks the slice-update
///   writes into;
/// * the attention: fused, its `[B, N, H, D]` output (the merge is a view); unfused, the scaled
///   query, its head-split copy (it keeps the roped query's transposed layout), the key and value
///   copies (broadcast over the query groups), the scores (the causal `where` and the softmax
///   donate) with the causal mask's comparison and its two position ranges, the output and its
///   merged copy;
/// * the o-projection (both residual sums reuse their operand: the layer's bindings are gone when
///   its graph runs), the gate and up projections (SiLU and their product donate) and the down
///   projection.
///
/// Held: the `k + 1` hidden states already returned (`[n, hidden]` each) and `k` layers' K/V cache
/// blocks.
pub(crate) fn decoder_layer_parts_on(
    cfg: &ModelConfig,
    n: i32,
    k: usize,
    unfused: bool,
) -> (u64, u64) {
    use crate::retention::allocated;
    use mlx_gen::ios_quantum::tensor_bytes;
    let cw = DECODER_COMPUTE_BYTES;
    let n = n.max(0) as u64;
    let hidden = n * cfg.hidden_size.max(0) as u64;
    let kv_heads = cfg.num_kv_heads.max(0) as u64;
    let head_dim = cfg.head_dim.max(0) as u64;
    let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
    let cache_block = 2 * allocated(cw * n.div_ceil(block) * block * kv_heads * head_dim);
    let layer = decoder_layer_bytes(cfg, n, unfused, cache_block);
    let held = (k as u64 + 1)
        .saturating_mul(tensor_bytes(cw, &[hidden]))
        .saturating_add((k as u64).saturating_mul(cache_block));
    (layer, held)
}

/// One decoder layer over `n` tokens (see [`decoder_layer_parts_on`]), its K/V landing in a cache
/// block of `cache_block` bytes (`0`: the streamed tower's cache-free layer).
fn decoder_layer_bytes(cfg: &ModelConfig, n: u64, unfused: bool, cache_block: u64) -> u64 {
    use crate::retention::{Kind, Retention};
    let cw = DECODER_COMPUTE_BYTES;
    let hidden = n * cfg.hidden_size.max(0) as u64;
    let heads = cfg.num_heads.max(0) as u64;
    let kv_heads = cfg.num_kv_heads.max(0) as u64;
    let head_dim = cfg.head_dim.max(0) as u64;
    let q = n * heads * head_dim;
    let kv = n * kv_heads * head_dim;
    let ff = n * cfg.intermediate_size.max(0) as u64;
    let mut r = Retention::on(cw, unfused);
    decoder_norm(&mut r, hidden, n, false);
    r.temp(cw, q).temp(cw, kv).temp(cw, kv);
    decoder_norm(&mut r, q, n * heads, true);
    decoder_norm(&mut r, kv, n * kv_heads, true);
    decoder_rope(&mut r, q);
    decoder_rope(&mut r, kv);
    r.temp_bytes(cache_block);
    decoder_attention(&mut r, n, heads, head_dim);
    r.temp(cw, hidden);
    decoder_norm(&mut r, hidden, n, false);
    r.alloc(Kind::Temp, 2, cw, ff).temp(cw, hidden);
    r.price()
}

/// One streamed decoder layer's quantum (rung 4) over `n` tokens on the current device: the
/// shared decoder's layer with no K/V cache block (the K/V feed the attention directly).
pub(crate) fn streamed_layer_bytes(cfg: &ModelConfig, n: u64) -> u64 {
    decoder_layer_bytes(cfg, n, crate::retention::unfused(), 0)
}

/// The streamed tower's embedding quantum over `n` tokens: the gathered `[1, n, hidden]` bf16
/// rows, and the RoPE tables `mlx_llm` builds (`Rope::tables`): the i32 positions and their f32
/// copy, the f32 angle table and its cosine and sine, and their bf16 casts (held).
pub(crate) fn streamed_embedding_bytes(cfg: &ModelConfig, n: u64) -> u64 {
    use crate::retention::{Kind, Retention};
    let dim = cfg.head_dim.max(0) as u64;
    let mut r = Retention::on(DECODER_COMPUTE_BYTES, false);
    r.named(DECODER_COMPUTE_BYTES, n * cfg.hidden_size.max(0) as u64)
        .alloc(Kind::Temp, 2, 4, n)
        .alloc(Kind::Temp, 3, 4, n * dim)
        .alloc(Kind::Named, 2, DECODER_COMPUTE_BYTES, n * dim);
    r.price()
}

/// One `mlx_llm` RMSNorm over a bf16 `elements` in `rows` rows (see [`decoder_layer_parts_on`]);
/// `donor`: its operand is a single-consumer temporary (the fused kernel reuses it).
pub(crate) fn decoder_norm(
    r: &mut crate::retention::Retention,
    elements: u64,
    rows: u64,
    donor: bool,
) {
    use crate::retention::Kind;
    let cw = r.cw;
    if r.unfused {
        r.alloc(Kind::Temp, 3, 4, elements)
            .temp(4, rows)
            .temp(cw, elements);
    } else if !donor {
        r.temp(cw, elements);
    }
}

/// One `mlx_llm` NeoX RoPE over a bf16 `elements` (see [`decoder_layer_parts_on`]).
pub(crate) fn decoder_rope(r: &mut crate::retention::Retention, elements: u64) {
    let cw = r.cw;
    r.temp(cw, elements / 2)
        .alloc(crate::retention::Kind::Temp, 2, cw, elements);
}

/// One `mlx_llm` causal GQA attention over `n` tokens and its merged output (see
/// [`decoder_layer_parts_on`]).
pub(crate) fn decoder_attention(
    r: &mut crate::retention::Retention,
    n: u64,
    heads: u64,
    head_dim: u64,
) {
    use crate::retention::Kind;
    let cw = r.cw;
    let q = n * heads * head_dim;
    if r.unfused {
        r.alloc(Kind::Temp, 5, cw, q)
            .temp(cw, heads * n * n)
            .temp(1, n * n)
            .alloc(Kind::Temp, 2, 4, n);
    }
    r.temp(cw, q);
}

/// Check a Qwen3-VL snapshot against the backbone's `text_encoder` section; returns its config.
fn checked_config(dir: &Path, cfg: &TextEncoderConfig) -> Result<ModelConfig> {
    let model_cfg = ModelConfig::from_dir(dir).map_err(from_llm)?;
    if !model_cfg.architecture.is_qwen3_vl() {
        return Err(Error::Unsupported(format!(
            "iris: {} is not a Qwen3-VL checkpoint (the backbone was trained on {})",
            dir.display(),
            cfg.pretrained
        )));
    }
    if model_cfg.hidden_size as usize != cfg.dim {
        return Err(Error::Msg(format!(
            "iris: the text encoder at {} is {} wide, but the backbone's text_encoder.dim is {}",
            dir.display(),
            model_cfg.hidden_size,
            cfg.dim
        )));
    }
    let deepest = *cfg.hidden_layers.last().unwrap_or(&0);
    if deepest > model_cfg.num_layers {
        return Err(Error::Msg(format!(
            "iris: text_encoder.hidden_layers needs layer {deepest}, but the text encoder at {} \
             has {} layers",
            dir.display(),
            model_cfg.num_layers
        )));
    }
    Ok(model_cfg)
}

/// The shard list of the tower, every one present.
fn present_shards(dir: &Path, untied_head: bool) -> Result<Vec<std::path::PathBuf>> {
    let files = weight_shards(dir, untied_head)?;
    if let Some(missing) = files.iter().find(|file| !file.is_file()) {
        return Err(Error::Msg(format!(
            "iris: the text encoder resource is incomplete — shard {} is missing",
            missing.display()
        )));
    }
    Ok(files)
}

/// Key stem of decoder layer `i` of the tower.
fn layer_stem(i: usize) -> String {
    format!("{LANGUAGE_TOWER_PREFIX}layers.{i}.")
}

/// The decoder layer a tower key belongs to, if any.
pub(crate) fn tower_layer_of(name: &str) -> Option<usize> {
    let rest = name
        .strip_prefix(LANGUAGE_TOWER_PREFIX)?
        .strip_prefix("layers.")?;
    let (index, tail) = rest.split_once('.')?;
    (!tail.is_empty()).then(|| index.parse().ok()).flatten()
}

/// The keys and read price of every decoder layer of the tower, from its shards' headers (shared
/// with [`crate::ios_pricing`]).
/// Per decoder layer, its `(file key, built key)` pairs; and its read price.
pub(crate) type TowerInventory = (Vec<Vec<(String, String)>>, Vec<crate::stream::BlockRead>);

pub(crate) fn tower_layer_inventory(
    files: &[std::path::PathBuf],
    layers: usize,
) -> Result<TowerInventory> {
    let mut headers = Vec::new();
    for file in files {
        headers.extend(crate::stream::headers(file)?);
    }
    let mut grouped: Vec<Vec<&gen_core::weightsmeta::SafetensorsTensorHeader>> =
        vec![Vec::new(); layers];
    for header in &headers {
        if let Some(i) = tower_layer_of(&header.name) {
            grouped
                .get_mut(i)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "iris: the text encoder carries layer {i} beyond its {layers} layers"
                    ))
                })?
                .push(header);
        }
    }
    if let Some(i) = grouped.iter().position(Vec::is_empty) {
        return Err(Error::Msg(format!(
            "iris: the text encoder has no decoder layer {i}"
        )));
    }
    let keys = grouped
        .iter()
        .map(|layer| {
            let mut keys: Vec<(String, String)> = layer
                .iter()
                .map(|h| (h.name.clone(), h.name.clone()))
                .collect();
            keys.sort();
            keys
        })
        .collect();
    let reads = grouped
        .iter()
        .map(|layer| crate::stream::block_read(layer, crate::stream::CastPolicy::Tower))
        .collect();
    Ok((keys, reads))
}

/// Rung 4 of the text tower: the embedding and the final norm resident, every decoder layer read
/// from the snapshot per window and released after it (sc-18798's discipline, applied by this
/// crate because the shared decoder's own stream reads a single file and the Qwen3-VL release is
/// sharded). The layer is the shared decoder's Qwen3 layer — the same `mlx_llm` primitives
/// (projections, per-head q/k RMSNorm, NeoX RoPE, causal GQA SDPA, SwiGLU) in the same order; with
/// no K/V cache, since an encoder forward never decodes.
pub(crate) struct StreamedTower {
    cfg: ModelConfig,
    files: Vec<std::path::PathBuf>,
    keys: Vec<Vec<(String, String)>>,
    reads: Vec<crate::stream::BlockRead>,
    /// `[vocab, hidden]` bf16.
    embed: Array,
    /// The final norm, applied only to the last state (HF's `output_hidden_states` layout).
    norm: Array,
    /// The packed artifact's stored quantization (`None`: the dense tier).
    stored_quant: Option<QuantSpec>,
    rope: mlx_llm::primitives::rope::Rope,
}

/// A test-build hook run as each streamed tower window opens.
#[cfg(test)]
pub(crate) type TowerWindowHook = Box<dyn FnMut(usize)>;

#[cfg(test)]
thread_local! {
    /// Called as each streamed tower window opens, before its read (test builds only).
    pub(crate) static TOWER_WINDOW_HOOK: std::cell::RefCell<Option<TowerWindowHook>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
impl StreamedTower {
    /// Read and build layer `i` from a fresh view (the read proof's body), returning its arrays.
    pub(crate) fn read_layer(&self, i: usize) -> Result<Vec<Array>> {
        let mut view = mlx_gen::weights::Weights::from_paths(&self.files, Path::new(""))?;
        let map =
            crate::stream::materialize(&mut view, &self.keys[i], crate::stream::CastPolicy::Tower)?;
        let layer = StreamedLayer::build(&map, i, self.stored_quant)?;
        drop(map);
        Ok(layer.arrays())
    }

    /// The read price of layer `i`.
    pub(crate) fn read_price(&self, i: usize) -> crate::stream::BlockRead {
        self.reads[i]
    }

    /// Read and build layer `i` (the streamed-layer proof runs its forward).
    pub(crate) fn build_layer(&self, i: usize) -> Result<StreamedLayer> {
        let mut view = mlx_gen::weights::Weights::from_paths(&self.files, Path::new(""))?;
        let map =
            crate::stream::materialize(&mut view, &self.keys[i], crate::stream::CastPolicy::Tower)?;
        StreamedLayer::build(&map, i, self.stored_quant)
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub(crate) fn embed_table(&self) -> &Array {
        &self.embed
    }

    pub(crate) fn rope(&self) -> &mlx_llm::primitives::rope::Rope {
        &self.rope
    }
}

/// One streamed Qwen3 decoder layer.
pub(crate) struct StreamedLayer {
    input_ln: Array,
    post_ln: Array,
    q: mlx_llm::primitives::projection::Projection,
    k: mlx_llm::primitives::projection::Projection,
    v: mlx_llm::primitives::projection::Projection,
    o: mlx_llm::primitives::projection::Projection,
    q_norm: Array,
    k_norm: Array,
    gate: mlx_llm::primitives::projection::Projection,
    up: mlx_llm::primitives::projection::Projection,
    down: mlx_llm::primitives::projection::Projection,
}

impl StreamedLayer {
    #[cfg(test)]
    fn arrays(&self) -> Vec<Array> {
        use mlx_llm::primitives::projection::Projection;
        let mut out = vec![
            self.input_ln.clone(),
            self.post_ln.clone(),
            self.q_norm.clone(),
            self.k_norm.clone(),
        ];
        for p in [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ] {
            match p {
                Projection::Dense { weight, bias } => {
                    out.push(weight.clone());
                    out.extend(bias.iter().cloned());
                }
                Projection::Quantized(q) => {
                    out.extend([q.weight.clone(), q.scales.clone(), q.biases.clone()]);
                    out.extend(q.bias.iter().cloned());
                }
                Projection::Prism(_) => unreachable!("the Iris tower has no Prism projection"),
            }
        }
        out
    }

    fn build(w: &mlx_gen::weights::Weights, i: usize, stored: Option<QuantSpec>) -> Result<Self> {
        use mlx_llm::primitives::projection::Projection;
        let stem = layer_stem(i);
        let key = |suffix: &str| format!("{stem}{suffix}");
        let get = |suffix: &str| -> Result<Array> { Ok(w.require(&key(suffix))?.clone()) };
        let proj = |name: &str| -> Result<Projection> {
            let scales = key(&format!("{name}.scales"));
            match (w.get(&scales), stored) {
                (Some(scales), Some(spec)) => Projection::from_quantized(
                    get(&format!("{name}.weight"))?,
                    scales.clone(),
                    get(&format!("{name}.biases"))?,
                    spec,
                    Dtype::Bfloat16,
                )
                .map_err(from_llm),
                (Some(_), None) => Err(Error::Msg(format!(
                    "iris: {stem}{name} is stored packed but the text encoder config has no \
                     quantization block"
                ))),
                (None, _) => {
                    Projection::load(get(&format!("{name}.weight"))?, None).map_err(from_llm)
                }
            }
        };
        Ok(Self {
            input_ln: get("input_layernorm.weight")?,
            post_ln: get("post_attention_layernorm.weight")?,
            q: proj("self_attn.q_proj")?,
            k: proj("self_attn.k_proj")?,
            v: proj("self_attn.v_proj")?,
            o: proj("self_attn.o_proj")?,
            q_norm: get("self_attn.q_norm.weight")?,
            k_norm: get("self_attn.k_norm.weight")?,
            gate: proj("mlp.gate_proj")?,
            up: proj("mlp.up_proj")?,
            down: proj("mlp.down_proj")?,
        })
    }

    /// The shared decoder's Qwen3 layer forward (`LlamaLayer::forward` without a K/V cache).
    pub(crate) fn forward(
        &self,
        cfg: &ModelConfig,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        use mlx_llm::primitives::attention::{sdpa_capped, AttnMask};
        use mlx_llm::primitives::nn::{rms_norm, silu};
        use mlx_llm::primitives::rope::apply_rope;
        let llm = from_llm;
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let eps = cfg.rms_norm_eps;
        let (heads, kv_heads, head_dim) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let normed = rms_norm(x, &self.input_ln, eps).map_err(llm)?;
        let q = self
            .q
            .forward(&normed)
            .map_err(llm)?
            .reshape(&[b, s, heads, head_dim])?;
        let q = rms_norm(&q, &self.q_norm, eps).map_err(llm)?;
        let q = apply_rope(&q, cos, sin, false)
            .map_err(llm)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let shape = [b, s, kv_heads, head_dim];
        let k = self.k.forward(&normed).map_err(llm)?.reshape(&shape)?;
        let v = self.v.forward(&normed).map_err(llm)?.reshape(&shape)?;
        let k = rms_norm(&k, &self.k_norm, eps).map_err(llm)?;
        let k = apply_rope(&k, cos, sin, false)
            .map_err(llm)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let attn = sdpa_capped(
            &q,
            &k,
            &v,
            cfg.attn_scale(),
            cfg.attn_logit_softcap,
            AttnMask::Causal,
        )
        .map_err(llm)?;
        let merged = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, heads * head_dim])?;
        let h = x.add(&self.o.forward(&merged).map_err(llm)?)?;
        let n = rms_norm(&h, &self.post_ln, eps).map_err(llm)?;
        let g = silu(&self.gate.forward(&n).map_err(llm)?).map_err(llm)?;
        let up = self.up.forward(&n).map_err(llm)?;
        Ok(h.add(&self.down.forward(&g.multiply(&up)?).map_err(llm)?)?)
    }
}

impl IrisTextEncoder {
    /// Load from a `Qwen/Qwen3-VL-4B-Instruct` snapshot directory under the backbone's
    /// `text_encoder` config section. Only the language tower's tensors are read.
    pub fn load(dir: &Path, cfg: &TextEncoderConfig) -> Result<Self> {
        Self::load_quantized(dir, cfg, None)
    }

    /// [`load`](Self::load) at a quantization tier (sc-26266): `quant` packs every attention and
    /// MLP projection of the language tower to Q4/Q8 at load (group size
    /// [`text_encoder_quant_group`]); the token embedding (tied to the head) and the norms stay
    /// bf16, as the shared decoder keeps them. The quantized build materializes group by group and
    /// releases each group's bf16 sources as it goes, so the load never holds the dense tower
    /// beside the packed one.
    pub fn load_quantized(
        dir: &Path,
        cfg: &TextEncoderConfig,
        quant: Option<Quant>,
    ) -> Result<Self> {
        // Fresh-buffer accounting requires cache-zero BEFORE materialization, not only later
        // generation. Keep this existing enabled-only policy alive through all load evaluations.
        let _load_quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;

        #[cfg(all(test, target_os = "macos"))]
        crate::anchor_tests::record_generation_loader_entry("tower");
        let model_cfg = checked_config(dir, cfg)?;
        let untied_head = !model_cfg.tie_word_embeddings;
        let files = present_shards(dir, untied_head)?;
        // Only the tower's tensors are priced and read: a shard's other tensors (the vision tower
        // rides in the last shard of the release) are lazy handles dropped unevaluated.
        mlx_gen::ios_quantum::before_load_where(quanta::LOAD, &files, |name| {
            is_text_tower_tensor(name, untied_head)
        })?;
        let mut tensors = HashMap::new();
        for file in &files {
            let part = Array::load_safetensors(file).map_err(|e| {
                Error::Msg(format!(
                    "iris: loading text-encoder shard {}: {e}",
                    file.display()
                ))
            })?;
            tensors.extend(
                part.into_iter()
                    .filter(|(name, _)| is_text_tower_tensor(name, untied_head)),
            );
        }
        // A pre-quantized tower (`crate::packed`) is read as stored — only at its own tier — and
        // the decoder builds it like any dense one; a dense tower is packed at load when a tier is
        // asked for.
        let quant = match model_cfg.quantization {
            // `text_encoder_tier` validates the stored pack: the recorded bits and group of the
            // config and every weight file agree, the group is the tower's, every projection packed.
            Some(_) => {
                let tier = crate::packed::text_encoder_tier(dir)?;
                crate::packed::check_requested("the text encoder", tier, quant)?;
                None
            }
            None => text_encoder_quant_spec(&model_cfg, quant)?,
        };
        let mut weights = mlx_llm::primitives::weights::Weights::from_map(tensors);
        let lm = match quant {
            // `from_weights` evaluates every parameter group it built (the load boundary), so no
            // forward ever reads a weight from disk.
            None => CausalLm::from_weights(&weights, "", model_cfg).map_err(from_llm)?,
            // Read, verify, quantize and release one parameter group at a time (the shared
            // decoder's bounded load, `mlx_llm::LlamaProvider`'s path), so the packed tower never
            // sits beside the whole bf16 one.
            Some(spec) => {
                let lm =
                    CausalLm::build_lazy(&weights, "", model_cfg, Some(spec)).map_err(from_llm)?;
                let report = weights
                    .materialize_groups(&lm.param_groups())
                    .map_err(from_llm)?;
                if report.leftover > 0 {
                    return Err(Error::Msg(format!(
                        "iris: {} text-encoder source tensor(s) were read but consumed by no \
                         parameter group",
                        report.leftover
                    )));
                }
                lm
            }
        };
        drop(weights);
        Self::assemble(dir, cfg, Tower::Resident(Box::new(lm)))
    }

    /// Rung 4 (bounded transformer residency): the tower with its embedding and final norm
    /// resident and every decoder layer read from the snapshot per window at encode time
    /// (`StreamedTower`). A packed tower (`crate::packed`) streams at its own tier; the dense
    /// release streams at the dense tier only — asked for at `q8`/`q4` it is refused by name
    /// (`refuse_stream_requantization`).
    pub fn load_streamed(
        dir: &Path,
        cfg: &TextEncoderConfig,
        quant: Option<Quant>,
    ) -> Result<Self> {
        // Fresh-buffer accounting requires cache-zero BEFORE materialization, not only later
        // generation. Keep this existing enabled-only policy alive through all load evaluations.
        let _load_quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;

        #[cfg(all(test, target_os = "macos"))]
        crate::anchor_tests::record_generation_loader_entry("tower");
        let model_cfg = checked_config(dir, cfg)?;
        let files = present_shards(dir, !model_cfg.tie_word_embeddings)?;
        let stored_quant = match model_cfg.quantization {
            Some(spec) => {
                let tier = crate::packed::text_encoder_tier(dir)?;
                crate::packed::check_requested("the text encoder", tier, quant)?;
                Some(spec)
            }
            None => {
                crate::stream::refuse_stream_requantization(
                    crate::model::MODEL_ID,
                    "the text encoder",
                    None,
                    quant,
                )?;
                None
            }
        };
        let embed_key = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
        let norm_key = format!("{LANGUAGE_TOWER_PREFIX}norm.weight");
        let resident = |name: &str| name == embed_key || name == norm_key;
        // iOS: only the embedding and the final norm are read here; every decoder layer is read
        // inside its own window's priced read quantum.
        mlx_gen::ios_quantum::before_load_where(quanta::LOAD, &files, resident)?;
        let (keys, reads) = tower_layer_inventory(&files, model_cfg.num_layers)?;
        let (mut embed, mut norm) = (None, None);
        for file in &files {
            let part = Array::load_safetensors(file).map_err(|e| {
                Error::Msg(format!(
                    "iris: loading text-encoder shard {}: {e}",
                    file.display()
                ))
            })?;
            for (name, array) in part {
                if name == format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.scales") {
                    return Err(Error::Unsupported(format!(
                        "iris: {} stores a packed token embedding; the streamed tower reads it \
                         dense (the Iris packed tower keeps it bf16)",
                        dir.display()
                    )));
                }
                if !resident(&name) {
                    continue;
                }
                mlx_rs::transforms::eval_pending_loads([&array])?;
                let array = array.as_dtype(Dtype::Bfloat16)?;
                eval([&array])?;
                if name == embed_key {
                    embed = Some(array);
                } else {
                    norm = Some(array);
                }
            }
        }
        let missing = |key: &str| Error::MissingTensor(key.to_owned());
        let tower = StreamedTower {
            rope: model_cfg.build_rope(),
            embed: embed.ok_or_else(|| missing(&embed_key))?,
            norm: norm.ok_or_else(|| missing(&norm_key))?,
            cfg: model_cfg,
            files,
            keys,
            reads,
            stored_quant,
        };
        Self::assemble(dir, cfg, Tower::Streamed(Box::new(tower)))
    }

    fn assemble(dir: &Path, cfg: &TextEncoderConfig, tower: Tower) -> Result<Self> {
        let tokenizer = TextTokenizer::from_file(
            dir.join("tokenizer.json"),
            TokenizerConfig {
                max_length: usize::MAX,
                pad_token_id: 0,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(Error::from)?;
        let prefix_ids = tokenizer
            .encode_ids(PROMPT_PREFIX, false)
            .map_err(Error::from)?;
        let suffix_ids = tokenizer
            .encode_ids(PROMPT_SUFFIX, false)
            .map_err(Error::from)?;
        Ok(Self {
            tower,
            tokenizer,
            prefix_ids,
            suffix_ids,
            hidden_layers: cfg.hidden_layers.clone(),
            max_length: cfg.max_length,
            dim: cfg.dim,
            overflow: CaptionOverflowPolicy::from_name(&cfg.on_caption_overflow).ok_or_else(
                || {
                    Error::Unsupported(format!(
                        "iris: text_encoder.on_caption_overflow = {} is not warn, error or silent",
                        cfg.on_caption_overflow
                    ))
                },
            )?,
        })
    }

    /// The tower's decoder config.
    pub(crate) fn model_config(&self) -> &ModelConfig {
        match &self.tower {
            Tower::Resident(lm) => lm.config(),
            Tower::Streamed(t) => &t.cfg,
        }
    }

    /// Whether the decoder layers are streamed by window (rung 4) rather than resident.
    pub fn is_streamed(&self) -> bool {
        matches!(self.tower, Tower::Streamed(_))
    }

    /// Inspect actual decoder parameter ownership, including every resident layer group.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn assert_endpoint_ownership(&self, streamed: bool) {
        assert_eq!(self.is_streamed(), streamed, "tower stream state");
        match &self.tower {
            Tower::Resident(lm) => {
                let groups = lm.param_groups();
                assert_eq!(
                    groups.len(),
                    lm.config().num_layers + 1,
                    "resident tower must own outside parameters and every decoder layer"
                );
                assert!(
                    groups
                        .iter()
                        .all(|group| !group.is_empty()
                            && group.iter().all(|array| array.nbytes() > 0)),
                    "resident tower contains an empty parameter group"
                );
            }
            Tower::Streamed(tower) => {
                assert_eq!(tower.reads.len(), tower.cfg.num_layers);
                assert!(
                    tower.embed.nbytes() > 0 && tower.norm.nbytes() > 0,
                    "streamed tower must own its embedding and final norm"
                );
            }
        }
        let _addresses = self.endpoint_parameter_addresses();
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn endpoint_parameter_addresses(&self) -> Vec<usize> {
        match &self.tower {
            Tower::Resident(lm) => {
                let groups = lm.param_groups();
                crate::anchor_tests::materialized_addresses(groups.iter().flatten())
            }
            Tower::Streamed(tower) => {
                crate::anchor_tests::materialized_addresses([&tower.embed, &tower.norm])
            }
        }
    }

    /// The streamed tower (test builds: the read proofs read layers from it).
    #[cfg(test)]
    pub(crate) fn streamed(&self) -> Option<&StreamedTower> {
        match &self.tower {
            Tower::Streamed(t) => Some(t),
            Tower::Resident(_) => None,
        }
    }

    /// Bytes the loaded tower holds — every parameter group of the decoder, at the width it is
    /// resident in (a streamed tower: its embedding and final norm). Measured from the model, not
    /// priced from the checkpoint.
    ///
    /// The decoder lists a tied head beside the embedding it shares, so arrays are deduplicated by
    /// the buffer they hold ([`mlx_llm::primitives::weights::data_address`], the identity the
    /// decoder's own group materialization uses): a buffer two handles share is counted once,
    /// wherever in the groups the second handle sits.
    pub fn resident_weight_bytes(&self) -> u64 {
        let lm = match &self.tower {
            Tower::Resident(lm) => lm,
            Tower::Streamed(t) => return (t.embed.nbytes() + t.norm.nbytes()) as u64,
        };
        let mut seen = std::collections::BTreeSet::new();
        lm.param_groups()
            .iter()
            .flatten()
            .filter(|a| {
                // An array with no data yet (address 0) is not a shared buffer: count it.
                let address = mlx_llm::primitives::weights::data_address(a);
                address == 0 || seen.insert(address)
            })
            .map(|a| a.nbytes() as u64)
            .sum()
    }

    /// Whether the tower's projections are affine-packed.
    pub fn is_quantized(&self) -> bool {
        match &self.tower {
            Tower::Resident(lm) => lm.is_quantized(),
            Tower::Streamed(t) => t.stored_quant.is_some(),
        }
    }

    pub fn prefix_ids(&self) -> &[i32] {
        &self.prefix_ids
    }

    pub fn suffix_ids(&self) -> &[i32] {
        &self.suffix_ids
    }

    /// Tokenize and assemble one prompt's window.
    pub fn window(&self, prompt: &str) -> Result<TextWindow> {
        let caption = self
            .tokenizer
            .encode_ids(prompt, false)
            .map_err(Error::from)?;
        assemble_window(
            &self.prefix_ids,
            &caption,
            &self.suffix_ids,
            self.max_length,
        )
        .map_err(Error::from)
    }

    /// `encode([prompt])` (and `null(negative_prompt)`, which is the same computation) under the
    /// checkpoint's own caption-overflow policy; a `warn` is logged.
    pub fn encode(&self, prompt: &str) -> Result<TextConditioning> {
        Ok(self.encode_with_policy(prompt, self.overflow)?.0)
    }

    /// [`encode`](Self::encode) under an explicit caption-overflow `policy` (upstream
    /// `_tokenize_captions`): `error` refuses an overflowing caption before the tower runs, `warn`
    /// logs and returns the warning, `silent` truncates without a word.
    pub fn encode_with_policy(
        &self,
        prompt: &str,
        policy: CaptionOverflowPolicy,
    ) -> Result<(TextConditioning, Option<String>)> {
        self.encode_with_policy_in(prompt, policy, None, &mlx_gen::CancelFlag::default())
    }

    /// [`Self::encode_with_policy`] with a streamed tower's layer window (rung 4; see
    /// [`Self::encode_window_with`]) and the request's cancel flag, checked between windows.
    pub fn encode_with_policy_in(
        &self,
        prompt: &str,
        policy: CaptionOverflowPolicy,
        layer_window: Option<usize>,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<(TextConditioning, Option<String>)> {
        let window = self.window(prompt)?;
        let warning =
            apply_caption_overflow_policy(&window, self.max_length, self.suffix_ids.len(), policy)
                .map_err(Error::from)?;
        if let Some(msg) = &warning {
            eprintln!("{msg}");
        }
        Ok((
            self.encode_window_with(&window, layer_window, cancel)?,
            warning,
        ))
    }

    /// iOS floor of evaluating hidden state `k` (0 = the embedding) of an `n`-token window: one
    /// decoder layer of the LOADED tower's geometry plus what the walk holds by then.
    pub(crate) fn layer_scratch(&self, n: i32, k: usize) -> u64 {
        let (layer, held) = self.layer_scratch_parts(n, k);
        layer.saturating_add(held)
    }

    /// [`Self::layer_scratch`] as `(one decoder layer, what the walk holds)` — counted from
    /// `mlx_llm`'s decoder layer over the LOADED tower's geometry ([`decoder_layer_parts_on`]).
    pub(crate) fn layer_scratch_parts(&self, n: i32, k: usize) -> (u64, u64) {
        decoder_layer_parts(self.model_config(), n, k)
    }

    /// The conditioning quantum over a window of `real` tokens: each picked state's f32 slice
    /// (`[1, real, dim]`), their `[1, real, layers, dim]` stack, and — when the window is short —
    /// the zero pad and the padded `[1, max_length, layers, dim]` copy.
    pub(crate) fn conditioning_scratch(&self, real: i32) -> u64 {
        conditioning_bytes(
            self.hidden_layers.len() as u64,
            self.max_length as u64,
            self.dim as u64,
            real,
        )
    }

    /// The conditioning quantum's guard: [`Self::conditioning_scratch`] at its worst caption
    /// length (the count peaks one token short of the budget, where the pad is one row).
    pub(crate) fn conditioning_guard(&self) -> u64 {
        (0..=self.max_length as i32)
            .map(|real| self.conditioning_scratch(real))
            .max()
            .unwrap_or(0)
    }

    /// The body of the conditioning quantum: the hidden states `hidden_layers` picks, sliced to the
    /// window from `start`, cast to f32, stacked and zero-padded to `max_length`.
    pub(crate) fn conditioning_body(
        &self,
        states: &[Option<Array>],
        start: i32,
        real: i32,
    ) -> Result<Array> {
        let pad = self.max_length as i32 - real;
        let picked: Vec<Array> =
            self.hidden_layers
                .iter()
                .map(|&k| -> Result<Array> {
                    let state = states.get(k).and_then(Option::as_ref).ok_or_else(|| {
                        Error::Msg(format!("iris: hidden state {k} not returned"))
                    })?;
                    Ok(state
                        .split_axis(&[start], 1)?
                        .swap_remove(1)
                        .as_dtype(Dtype::Float32)?)
                })
                .collect::<Result<_>>()?;
        let stacked = stack_axis(&picked, 2)?; // [1, real, layers, dim]
        Ok(if pad > 0 {
            let zeros = zeros::<f32>(&[1, pad, self.hidden_layers.len() as i32, self.dim as i32])?;
            concatenate_axis(&[&stacked, &zeros], 1)?
        } else {
            stacked
        })
    }

    /// The loaded decoder (the CPU proofs drive one layer of it). A streamed tower has none.
    #[cfg(test)]
    pub(crate) fn lm(&self) -> &CausalLm {
        match &self.tower {
            Tower::Resident(lm) => lm,
            Tower::Streamed(_) => panic!("a streamed tower holds no resident decoder"),
        }
    }

    /// Decoder layers in the loaded tower (each one gated before construction on iOS).
    #[cfg(test)]
    pub(crate) fn lm_config_layers(&self) -> usize {
        self.model_config().num_layers
    }

    /// The resident tower's hidden states `0..=deepest` (each evaluated in its own quantum).
    fn resident_states(&self, lm: &CausalLm, window: &TextWindow) -> Result<Vec<Option<Array>>> {
        let n = window.input_ids.len() as i32;
        let ids = Array::from_slice(&window.input_ids, &[1, n]);
        let deepest = *self.hidden_layers.last().unwrap_or(&0);
        // The whole walk's floor (one layer plus every state it holds), checked before the shared
        // decoder builds anything: a window longer than the decoder's prefill checkpoint
        // (`mlx_llm`'s 256 rows) is evaluated layer by layer INSIDE `hidden_states`, before the
        // per-layer checks below can run.
        mlx_gen::ios_quantum::before(quanta::TOWER, self.layer_scratch(n, deepest))?;
        let mut cache = lm.new_cache();
        // Every decoder layer is ALSO checked against live headroom before its graph is built, from
        // inside the shared decoder stack (which evaluates long windows layer by layer itself).
        let states = if mlx_gen::ios_quantum::enabled() {
            let scratch: Vec<u64> = (0..=lm.config().num_layers)
                .map(|k| self.layer_scratch(n, k + 1))
                .collect();
            mlx_llm::layer_gate::with_layer_gate(
                move |layer| {
                    mlx_gen::ios_quantum::before(
                        quanta::DECODER_LAYER,
                        scratch.get(layer).copied().unwrap_or(u64::MAX),
                    )
                    .map_err(|e| e.to_string())
                },
                || lm.hidden_states(&ids, &mut cache, 0),
            )
        } else {
            lm.hidden_states(&ids, &mut cache, 0)
        }
        .map_err(from_llm)?;
        // One command buffer per layer: the returned handles are a single lazy graph. On a
        // physical iOS device each layer is also a checked, synchronized and drained quantum.
        for (k, state) in states.iter().take(deepest + 1).enumerate() {
            let label = if k == 0 {
                quanta::EMBEDDING
            } else {
                quanta::LAYER
            };
            mlx_gen::ios_quantum::before(label, self.layer_scratch(n, k))?;
            eval([state])?;
            mlx_gen::ios_quantum::finish(label, &[state])?;
        }
        mlx_gen::ios_quantum::finish(quanta::TOWER, &[])?;
        Ok(states.into_iter().map(Some).collect())
    }

    /// The streamed tower's picked hidden states (rung 4): the embedding (and the RoPE tables) in
    /// one quantum, then the decoder layers `0..deepest` window by window through the shared
    /// [`mlx_gen::block_residency`] driver — each window read inside its priced
    /// [`quanta::LAYER_READ`] quantum, each layer one [`quanta::STREAMED_LAYER`] quantum — keeping
    /// only the states `hidden_layers` picks and the carry.
    fn streamed_states(
        &self,
        tower: &StreamedTower,
        window: &TextWindow,
        layer_window: Option<usize>,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<Vec<Option<Array>>> {
        let cfg = &tower.cfg;
        let n = window.input_ids.len() as i32;
        let deepest = *self.hidden_layers.last().unwrap_or(&0);
        let keep = |k: usize| self.hidden_layers.contains(&k);
        mlx_gen::ios_quantum::before(
            quanta::EMBEDDING,
            streamed_embedding_bytes(cfg, n.max(0) as u64),
        )?;
        let ids = Array::from_slice(&window.input_ids, &[1, n]);
        let h = mlx_llm::primitives::nn::embed(&tower.embed, &ids).map_err(from_llm)?;
        let (cos, sin) = tower
            .rope
            .cos_sin(n, 0, Dtype::Bfloat16)
            .map_err(from_llm)?;
        eval([&h, &cos, &sin])?;
        mlx_gen::ios_quantum::finish(quanta::EMBEDDING, &[&h, &cos, &sin])?;
        let mut states: Vec<Option<Array>> = vec![None; deepest + 1];
        if keep(0) {
            states[0] = Some(h.clone());
        }
        // Layers 0..deepest produce states 1..=deepest (the last one final-normed when it is the
        // tower's last layer, HF's `output_hidden_states` layout).
        let layers = deepest.min(cfg.num_layers);
        if layers == 0 {
            return Ok(states);
        }
        let size = layer_window.unwrap_or(layers).clamp(1, layers);
        let plan = mlx_gen::block_residency::BlockPlan::new(layers, size)
            .map_err(|e| Error::Msg(format!("iris text stream: {e}")))?;
        let layer_scratch = streamed_layer_bytes(cfg, n.max(0) as u64);
        let h = mlx_gen::block_residency::run_windowed(
            &plan,
            cancel,
            h,
            || mlx_gen::weights::Weights::from_paths(&tower.files, Path::new("")),
            |mut h, view, range| {
                #[cfg(test)]
                TOWER_WINDOW_HOOK.with(|hook| {
                    if let Some(hook) = hook.borrow_mut().as_mut() {
                        hook(range.start);
                    }
                });
                mlx_gen::ios_quantum::before(
                    quanta::LAYER_READ,
                    crate::stream::window_read_bytes(&tower.reads[range.clone()]),
                )?;
                let mut built = Vec::with_capacity(range.len());
                for i in range.clone() {
                    let map = crate::stream::materialize(
                        view,
                        &tower.keys[i],
                        crate::stream::CastPolicy::Tower,
                    )?;
                    built.push(StreamedLayer::build(&map, i, tower.stored_quant)?);
                    let unused = map.unused_keys();
                    if let Some(first) = unused.first() {
                        return Err(Error::Msg(format!(
                            "iris text stream: layer {i} carries `{first}`, which the layer does \
                             not consume"
                        )));
                    }
                }
                mlx_gen::ios_quantum::finish(quanta::LAYER_READ, &[])?;
                for (i, layer) in range.zip(&built) {
                    let last = i + 1 == cfg.num_layers;
                    h = mlx_gen::ios_quantum::run(quanta::STREAMED_LAYER, layer_scratch, || {
                        let out = layer.forward(cfg, &h, &cos, &sin)?;
                        Ok(if last {
                            mlx_llm::primitives::nn::rms_norm(&out, &tower.norm, cfg.rms_norm_eps)
                                .map_err(from_llm)?
                        } else {
                            out
                        })
                    })?;
                    eval([&h])?;
                    if keep(i + 1) {
                        states[i + 1] = Some(h.clone());
                    }
                }
                Ok(h)
            },
            |h| Ok(eval([h])?),
        )?;
        drop(h);
        Ok(states)
    }

    /// Run the tower over an assembled window (a streamed tower at its whole-stack window).
    pub fn encode_window(&self, window: &TextWindow) -> Result<TextConditioning> {
        self.encode_window_with(window, None, &mlx_gen::CancelFlag::default())
    }

    /// [`Self::encode_window`] with a streamed tower's window size (`None`: the whole stack in one
    /// window — a deferred load that did not select rung 4 still never keeps the stack resident).
    /// A resident tower ignores `layer_window`.
    pub fn encode_window_with(
        &self,
        window: &TextWindow,
        layer_window: Option<usize>,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<TextConditioning> {
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let states = match &self.tower {
            Tower::Resident(lm) => self.resident_states(lm, window)?,
            Tower::Streamed(tower) => self.streamed_states(tower, window, layer_window, cancel)?,
        };
        let start = window.prefix_len as i32;
        let real = window.window_tokens() as i32;
        // Guarded at the worst caption length the window admits: the price is then one figure
        // per tower, which the request-independent iOS pricing (`crate::ios_pricing`) states.
        let states =
            mlx_gen::ios_quantum::run(quanta::CONDITIONING, self.conditioning_guard(), || {
                self.conditioning_body(&states, start, real)
            })?;
        eval([&states])?;
        Ok(TextConditioning {
            states,
            mask: window.mask.clone(),
            truncated_tokens: window.truncated_tokens,
        })
    }
}
