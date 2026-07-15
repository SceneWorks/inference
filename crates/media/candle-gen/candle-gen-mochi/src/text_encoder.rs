//! Mochi text conditioning — the reused T5-XXL **masked** encode.
//!
//! Mochi's `MochiPipeline._get_t5_prompt_embeds` runs the T5 encoder **with** the tokenizer padding
//! mask (`self.text_encoder(input_ids, attention_mask=prompt_attention_mask)`), so padded tokens don't
//! pollute the real-token embeddings — the same masked path Chroma uses (and unlike FLUX, which runs T5
//! unmasked). We reuse [`candle_gen_flux::packed_te::PackedT5Encoder`] (t5-v1.1-xxl: 24 blocks, 64×64
//! heads, gated-GELU FFN) through its `forward_masked` seam, feeding an additive key-padding mask built
//! from the tokenizer's 0/1 mask.
//!
//! **Dtype regime.** The reference loads the pipeline at `torch_dtype=bfloat16`, and `text_encoder/`
//! ships no bf16 variant — so diffusers reads the fp32 shards and rounds them to bf16. Those shards
//! are already bf16-trained (verified: 0 of 806k sampled fp32 weights carry sub-bf16 mantissa bits),
//! so the *weights* are the same either way; only the **activation** precision is a choice. We run
//! **bf16 weights + f32 activations** ([`MochiT5::load_reference_regime`]) — MLX's T5 regime and this
//! crate's own DiT regime. The residual `te_parity` gap reflects cross-backend kernel noise against a
//! Metal-blessed golden, not a code delta.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::{CandleError, Result};
use candle_gen_flux::packed_te::{PackedT5Encoder, T5Config};
use tokenizers::Tokenizer;

use crate::tokenizer::{MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};

/// Large negative added to padded keys in the T5 self-attention (softmax → ~0 weight). Matches the
/// MLX port's `T5_MASK_NEG`; the HF reference uses `finfo(dtype).min`, but any value that drives the
/// softmax weight to 0 is equivalent.
const T5_MASK_NEG: f32 = -1e9;

/// One prompt's T5 conditioning: the last hidden state and the 0/1 per-token attention mask — the pair
/// `_get_t5_prompt_embeds` returns (both threaded into the Mochi DiT downstream).
pub struct MochiTextConditioning {
    /// `[1, L, 4096]` T5-XXL last hidden state, computed with the padding mask (at [`crate::DIT_DTYPE`]).
    pub prompt_embeds: Tensor,
    /// `[1, L]` per-token attention mask, f32 (`1` = real/EOS token, `0` = pad).
    pub prompt_attention_mask: Tensor,
}

/// The reused T5-XXL encoder loaded at a fixed compute dtype (bf16 for Mochi).
pub struct MochiT5 {
    enc: PackedT5Encoder,
    dtype: DType,
}

impl MochiT5 {
    /// Load the T5-XXL encoder from the snapshot's `text_encoder/` shards (index-filtered — see
    /// [`load_indexed_var_builder`]) at `dtype` — weights **and** activations at `dtype`.
    pub fn load(dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let vb = load_indexed_var_builder(dir, dtype, device)?;
        let enc = PackedT5Encoder::new(&T5Config::xxl(), vb)?;
        Ok(Self { enc, dtype })
    }

    /// Load in the **reference regime**: weights resident at **bf16**, activations computed at
    /// **f32**. This is what the `te_parity` golden was blessed in and what `mlx-gen-mochi` runs
    /// (bf16 weights, f32 activations), and it matches this crate's own DiT (bf16 storage / f32
    /// compute, [`crate::DIT_DTYPE`] + the `nn::linear_*` helpers).
    ///
    /// Both halves matter, and neither naive load gets it:
    ///   - all-**bf16** computes activations in bf16 too, stacking our own independent bf16 rounding
    ///     on the reference's — two independent bf16 impls diverge from each other by *more* than
    ///     either diverges from exact f32 (measured `te_parity` `peak_rel`: 1.444e-1 vs 6.044e-2);
    ///   - all-**f32** is numerically right (the fp32 shards are bf16-trained, so an f32 load carries
    ///     the reference's exact weights) but doubles the resident T5 from ~9.5 GB to ~19 GB.
    ///
    /// Loading bf16 and upcasting per matmul (`candle_gen::QLinear::forward_upcast`) is numerically
    /// identical to the f32 load at bf16's footprint — only the one weight in flight is transient.
    pub fn load_reference_regime(dir: &Path, device: &Device) -> Result<Self> {
        let vb = load_indexed_var_builder(dir, DType::BF16, device)?;
        let enc = PackedT5Encoder::new(&T5Config::xxl(), vb)?;
        Ok(Self {
            enc,
            dtype: DType::F32,
        })
    }

    /// Encode `input_ids` `[1, L]` with the additive key-padding `mask` `[1, 1, 1, L]` → `[1, L, 4096]`.
    pub fn forward_masked(&self, input_ids: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        Ok(self.enc.forward_masked(input_ids, self.dtype, mask)?)
    }
}

/// mmap a [`VarBuilder`] over **only** the safetensors shards referenced by
/// `<dir>/model.safetensors.index.json`'s `weight_map` (the canonical shard set), at `dtype`/`device`.
///
/// Mochi's `text_encoder/` ships **two overlapping shard sets** (`*-of-00002` and `*-of-00004`); only
/// the 4-shard set is referenced by the index. The shared [`candle_gen::sorted_safetensors`] globs
/// *all* `.safetensors` in the dir → duplicate keys, so we read the index's `weight_map`, dedupe to the
/// referenced shards, and mmap exactly those. Falls back to the plain sorted load when no index is
/// present (a single-file / unambiguous checkpoint).
pub fn load_indexed_var_builder(
    dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return candle_gen::load_sorted_mmap(dir, dtype, device, "mochi t5");
    }
    let text = std::fs::read_to_string(&index)
        .map_err(|e| CandleError::Msg(format!("mochi t5 index {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| CandleError::Msg(format!("mochi t5 index {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| {
            CandleError::Msg(format!("mochi t5 index {}: no weight_map", index.display()))
        })?;

    // Unique shard filenames (BTreeSet → sorted + deduped), resolved under `dir`.
    let shard_files: BTreeSet<String> = map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    if shard_files.is_empty() {
        return Err(CandleError::Msg(format!(
            "mochi t5 index {}: weight_map references no shards",
            index.display()
        )));
    }
    let files: Vec<PathBuf> = shard_files.into_iter().map(|f| dir.join(f)).collect();
    candle_gen::mmap_var_builder(&files, dtype, device)
}

/// Tokenize `prompt` (`padding="max_length"`, `max_length=256`, `truncation`, `add_special_tokens`) →
/// `(input_ids [1, 256] u32, mask01 [256] 0/1)`. Real tokens (content + the appended EOS `</s>`) are a
/// contiguous prefix; the tail is right-padded with pad id `0` and `mask01 = 0`.
pub fn tokenize(
    tokenizer: &Tokenizer,
    prompt: &str,
    device: &Device,
) -> Result<(Tensor, Vec<u32>)> {
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("mochi: T5 tokenize: {e}")))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    let max = MAX_SEQUENCE_LENGTH;
    if ids.len() > max {
        ids.truncate(max);
    }
    let nv = ids.len();
    let mut mask = vec![1u32; nv];
    ids.resize(max, PAD_TOKEN_ID);
    mask.resize(max, 0);
    let input_ids = Tensor::from_vec(ids, (1, max), device)?;
    Ok((input_ids, mask))
}

/// The additive T5 key-padding mask `[1, 1, 1, L]` = `0` where `mask01 == 1` else [`T5_MASK_NEG`],
/// broadcastable to the T5 attention scores `[1, heads, L, L]`.
pub(crate) fn t5_key_mask(mask01: &[u32], device: &Device) -> Result<Tensor> {
    let l = mask01.len();
    let data: Vec<f32> = mask01
        .iter()
        .map(|&m| if m != 0 { 0.0 } else { T5_MASK_NEG })
        .collect();
    Ok(Tensor::from_vec(data, (1, 1, 1, l), device)?)
}

/// The per-token attention mask `[1, L]` f32 (`1` real, `0` pad) — the tokenizer `attention_mask` Mochi
/// returns alongside the embeds (it drives the DiT joint-attention mask).
pub(crate) fn attention_mask(mask01: &[u32], device: &Device) -> Result<Tensor> {
    let l = mask01.len();
    let data: Vec<f32> = mask01.iter().map(|&m| m as f32).collect();
    Ok(Tensor::from_vec(data, (1, l), device)?)
}

/// Encode one prompt → [`MochiTextConditioning`] (`_get_t5_prompt_embeds`): tokenize at `max_length`
/// (pad-to-max), run the T5 encoder **with** the additive key-padding mask, and return the
/// `[1, L, 4096]` embeds + the `[1, L]` 0/1 attention mask.
pub fn encode_prompt(
    tokenizer: &Tokenizer,
    t5: &MochiT5,
    prompt: &str,
    device: &Device,
) -> Result<MochiTextConditioning> {
    let (input_ids, mask01) = tokenize(tokenizer, prompt, device)?;
    let key_mask = t5_key_mask(&mask01, device)?;
    let prompt_embeds = t5.forward_masked(&input_ids, Some(&key_mask))?;
    let prompt_attention_mask = attention_mask(&mask01, device)?;
    Ok(MochiTextConditioning {
        prompt_embeds,
        prompt_attention_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The additive key mask is `0` for real ids and [`T5_MASK_NEG`] for pad, shaped `[1, 1, 1, L]`.
    #[test]
    fn key_mask_is_zero_for_content_and_neg_for_pad() {
        let dev = Device::Cpu;
        let mask01 = [1u32, 1, 1, 1, 0, 0];
        let m = t5_key_mask(&mask01, &dev).unwrap();
        assert_eq!(m.dims(), &[1, 1, 1, 6]);
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0, T5_MASK_NEG, T5_MASK_NEG]);
    }

    /// The 0/1 attention mask marks content+EOS as `1` and pad as `0`, shaped `[1, L]`.
    #[test]
    fn attention_mask_marks_real_tokens() {
        let dev = Device::Cpu;
        let mask01 = [1u32, 1, 1, 1, 0, 0];
        let m = attention_mask(&mask01, &dev).unwrap();
        assert_eq!(m.dims(), &[1, 6]);
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
