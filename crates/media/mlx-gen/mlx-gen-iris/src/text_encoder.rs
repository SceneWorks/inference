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

/// The loaded Qwen3-VL language tower + tokenizer + the backbone's conditioning contract.
pub struct IrisTextEncoder {
    lm: CausalLm,
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
    use crate::retention::{allocated, Kind, Retention};
    use mlx_gen::ios_quantum::tensor_bytes;
    let cw = DECODER_COMPUTE_BYTES;
    let n = n.max(0) as u64;
    let hidden = n * cfg.hidden_size.max(0) as u64;
    let heads = cfg.num_heads.max(0) as u64;
    let kv_heads = cfg.num_kv_heads.max(0) as u64;
    let head_dim = cfg.head_dim.max(0) as u64;
    let q = n * heads * head_dim;
    let kv = n * kv_heads * head_dim;
    let ff = n * cfg.intermediate_size.max(0) as u64;
    let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
    let cache_block = 2 * allocated(cw * n.div_ceil(block) * block * kv_heads * head_dim);
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
    let held = (k as u64 + 1)
        .saturating_mul(tensor_bytes(cw, &[hidden]))
        .saturating_add((k as u64).saturating_mul(cache_block));
    (r.price(), held)
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

        let untied_head = !model_cfg.tie_word_embeddings;
        let files = weight_shards(dir, untied_head)?;
        if let Some(missing) = files.iter().find(|file| !file.is_file()) {
            return Err(Error::Msg(format!(
                "iris: the text encoder resource is incomplete — shard {} is missing",
                missing.display()
            )));
        }
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
            lm,
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

    /// Bytes the loaded tower holds — every parameter group of the decoder, at the width it is
    /// resident in. Measured from the model, not priced from the checkpoint.
    ///
    /// The decoder lists a tied head beside the embedding it shares, so arrays are deduplicated by
    /// the buffer they hold ([`mlx_llm::primitives::weights::data_address`], the identity the
    /// decoder's own group materialization uses): a buffer two handles share is counted once,
    /// wherever in the groups the second handle sits.
    pub fn resident_weight_bytes(&self) -> u64 {
        let mut seen = std::collections::BTreeSet::new();
        self.lm
            .param_groups()
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
        self.lm.is_quantized()
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
        let window = self.window(prompt)?;
        let warning =
            apply_caption_overflow_policy(&window, self.max_length, self.suffix_ids.len(), policy)
                .map_err(Error::from)?;
        if let Some(msg) = &warning {
            eprintln!("{msg}");
        }
        Ok((self.encode_window(&window)?, warning))
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
        decoder_layer_parts(self.lm.config(), n, k)
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

    /// The body of the conditioning quantum: the hidden states `hidden_layers` picks, sliced to the
    /// window from `start`, cast to f32, stacked and zero-padded to `max_length`.
    pub(crate) fn conditioning_body(
        &self,
        states: &[Array],
        start: i32,
        real: i32,
    ) -> Result<Array> {
        let pad = self.max_length as i32 - real;
        let picked: Vec<Array> = self
            .hidden_layers
            .iter()
            .map(|&k| -> Result<Array> {
                let state = states
                    .get(k)
                    .ok_or_else(|| Error::Msg(format!("iris: hidden state {k} not returned")))?;
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

    /// The loaded decoder (the CPU proofs drive one layer of it).
    #[cfg(test)]
    pub(crate) fn lm(&self) -> &CausalLm {
        &self.lm
    }

    /// Decoder layers in the loaded tower (each one gated before construction on iOS).
    #[cfg(test)]
    pub(crate) fn lm_config_layers(&self) -> usize {
        self.lm.config().num_layers
    }

    /// Run the tower over an assembled window.
    pub fn encode_window(&self, window: &TextWindow) -> Result<TextConditioning> {
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        let n = window.input_ids.len() as i32;
        let ids = Array::from_slice(&window.input_ids, &[1, n]);
        let deepest = *self.hidden_layers.last().unwrap_or(&0);
        // The whole walk's floor (one layer plus every state it holds), checked before the shared
        // decoder builds anything: a window longer than the decoder's prefill checkpoint
        // (`mlx_llm`'s 256 rows) is evaluated layer by layer INSIDE `hidden_states`, before the
        // per-layer checks below can run.
        mlx_gen::ios_quantum::before(quanta::TOWER, self.layer_scratch(n, deepest))?;
        let mut cache = self.lm.new_cache();
        // Every decoder layer is ALSO checked against live headroom before its graph is built, from
        // inside the shared decoder stack (which evaluates long windows layer by layer itself).
        let states = if mlx_gen::ios_quantum::enabled() {
            let scratch: Vec<u64> = (0..=self.lm.config().num_layers)
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
                || self.lm.hidden_states(&ids, &mut cache, 0),
            )
        } else {
            self.lm.hidden_states(&ids, &mut cache, 0)
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
        let start = window.prefix_len as i32;
        let real = window.window_tokens() as i32;
        let states = mlx_gen::ios_quantum::run(
            quanta::CONDITIONING,
            self.conditioning_scratch(real),
            || self.conditioning_body(&states, start, real),
        )?;
        eval([&states])?;
        Ok(TextConditioning {
            states,
            mask: window.mask.clone(),
            truncated_tokens: window.truncated_tokens,
        })
    }
}
