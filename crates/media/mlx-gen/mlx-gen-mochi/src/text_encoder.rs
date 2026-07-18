//! Mochi text conditioning — the reused T5-XXL **masked** encode.
//!
//! Mochi's `MochiPipeline._get_t5_prompt_embeds` runs the T5 encoder **with** the tokenizer padding
//! mask (`self.text_encoder(input_ids, attention_mask=prompt_attention_mask)`), so padded tokens don't
//! pollute the real-token embeddings — the same masked path Chroma uses (and unlike FLUX, which runs
//! T5 unmasked). We reuse [`mlx_gen_flux::T5TextEncoder`] (t5-v1.1-xxl: 24 blocks, 64×64 heads,
//! gated-GELU FFN) and build the additive key-padding mask from `input_ids != pad`.
//!
//! The reference loads the whole pipeline at `torch_dtype=bfloat16`, so the T5 runs bf16 weights; we
//! mirror that by casting the fp32 snapshot shards to bf16 at load (the embedding still upcasts to f32,
//! so activations are f32 — strictly more precise than the reference's bf16 activations, which is why
//! the parity residual reflects the reference's accumulated bf16 rounding, not a code delta).

use std::path::Path;

use mlx_gen::tokenizer::to_arrays;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_flux::T5TextEncoder;
use mlx_rs::{Array, Dtype};

use crate::tokenizer::{MochiTokenizer, PAD_TOKEN_ID};

/// Large negative added to padded keys in the T5 self-attention (softmax → exactly 0 weight in f32).
/// Matches Chroma's `T5_MASK_NEG`; the HF reference uses `finfo(dtype).min`, but any value that drives
/// the softmax weight to 0 is equivalent.
const T5_MASK_NEG: f32 = -1e9;

/// One prompt's T5 conditioning: the last hidden state and the 0/1 per-token attention mask — the pair
/// `_get_t5_prompt_embeds` returns (both threaded into the Mochi DiT downstream).
pub struct MochiTextConditioning {
    /// `[1, L, 4096]` T5-XXL last hidden state, computed with the padding mask.
    pub prompt_embeds: Array,
    /// `[1, L]` per-token attention mask (`1` = real/EOS token, `0` = pad).
    pub prompt_attention_mask: Array,
}

/// Load the T5-XXL encoder from the snapshot's `text_encoder/` shards.
///
/// The `text_encoder/` dir ships **two overlapping shard sets** (`*-of-00002` and `*-of-00004`); only
/// the 4-shard set is referenced by `model.safetensors.index.json`. A plain
/// [`Weights::from_dir`](mlx_gen::weights::Weights::from_dir) would load both and abort on the
/// duplicate keys, so we load exactly the shards the index references. Weights are cast to bf16 to
/// mirror the reference's `torch_dtype=bfloat16` load (and to halve resident memory).
pub fn load_t5_encoder(root: &Path) -> Result<T5TextEncoder> {
    let dir = root.join("text_encoder");
    let mut w = load_indexed_shards(&dir)?;
    w.cast_all(Dtype::Bfloat16)?;
    T5TextEncoder::from_weights(&w, "")
}

/// Load only the safetensors shards referenced by `<dir>/model.safetensors.index.json` (the canonical
/// shard set), merged into one [`Weights`]. Falls back to [`Weights::from_dir`] when no index is
/// present (a single-file or unambiguous checkpoint).
fn load_indexed_shards(dir: &Path) -> Result<Weights> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Weights::from_dir(dir);
    }
    let text = std::fs::read_to_string(&index)
        .map_err(|e| Error::Msg(format!("mochi t5 index {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("mochi t5 index {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::Msg(format!("mochi t5 index {}: no weight_map", index.display())))?;

    // Unique shard filenames, sorted for determinism.
    let mut shard_files: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let mut combined = Weights::empty();
    for f in shard_files {
        let shard = Weights::from_file(dir.join(&f))?;
        let keys: Vec<String> = shard.keys().map(String::from).collect();
        for k in keys {
            // `get` is guaranteed to hit — we're iterating this shard's own keys.
            if let Some(t) = shard.get(&k) {
                combined.insert(k, t.clone());
            }
        }
    }
    Ok(combined)
}

/// The additive T5 key-padding mask `[1, 1, 1, L]` = `0` where `id != pad` else [`T5_MASK_NEG`],
/// broadcastable to the T5 attention scores `[1, heads, L, L]`. Real tokens (content + EOS) are a
/// contiguous non-pad prefix, so this is derived from `input_ids != pad`.
pub(crate) fn t5_key_mask(input_ids: &Array) -> Result<Array> {
    let ids = input_ids.as_dtype(Dtype::Int32)?;
    let ids: Vec<i32> = ids.as_slice::<i32>().to_vec();
    let l = *input_ids.shape().last().unwrap();
    let data: Vec<f32> = ids
        .iter()
        .map(|&id| if id != PAD_TOKEN_ID { 0.0 } else { T5_MASK_NEG })
        .collect();
    Ok(Array::from_slice(&data, &[1, 1, 1, l]))
}

/// The per-token attention mask `[1, L]` f32 (`1` where `id != pad` else `0`) — the tokenizer
/// `attention_mask` Mochi returns alongside the embeds (its `.bool()` form drives the DiT mask).
pub(crate) fn attention_mask(input_ids: &Array) -> Result<Array> {
    let ids = input_ids.as_dtype(Dtype::Int32)?;
    let ids: Vec<i32> = ids.as_slice::<i32>().to_vec();
    let l = *input_ids.shape().last().unwrap();
    let data: Vec<f32> = ids
        .iter()
        .map(|&id| if id != PAD_TOKEN_ID { 1.0 } else { 0.0 })
        .collect();
    Ok(Array::from_slice(&data, &[1, l]))
}

/// Encode one prompt → [`MochiTextConditioning`] (`_get_t5_prompt_embeds`).
///
/// Tokenizes at `max_length` (pad-to-max), runs the T5 encoder **with** the additive key-padding mask,
/// and returns the `[1, L, 4096]` embeds + the `[1, L]` 0/1 attention mask.
pub fn encode_prompt(
    tokenizer: &MochiTokenizer,
    t5: &T5TextEncoder,
    prompt: &str,
) -> Result<MochiTextConditioning> {
    let tok = tokenizer.tokenize(prompt)?;
    let (input_ids, _) = to_arrays(&tok);
    let key_mask = t5_key_mask(&input_ids)?;
    let prompt_embeds = t5.forward_masked(&input_ids, Some(&key_mask))?;
    let prompt_attention_mask = attention_mask(&input_ids)?;
    Ok(MochiTextConditioning {
        prompt_embeds,
        prompt_attention_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The additive key mask is `0` for non-pad ids and [`T5_MASK_NEG`] for pad (id 0), shaped
    /// `[1, 1, 1, L]` for broadcast over the T5 attention scores.
    #[test]
    fn key_mask_is_zero_for_content_and_neg_for_pad() {
        // ids: 3 content tokens + EOS(1) + 2 pad(0).
        let ids = Array::from_slice(&[10, 20, 30, 1, 0, 0], &[1, 6]);
        let m = t5_key_mask(&ids).unwrap();
        assert_eq!(m.shape(), &[1, 1, 1, 6]);
        let v: Vec<f32> = m.as_slice::<f32>().to_vec();
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0, T5_MASK_NEG, T5_MASK_NEG]);
    }

    /// The 0/1 attention mask marks content+EOS as `1` and pad as `0`, shaped `[1, L]`.
    #[test]
    fn attention_mask_marks_real_tokens() {
        let ids = Array::from_slice(&[10, 20, 30, 1, 0, 0], &[1, 6]);
        let m = attention_mask(&ids).unwrap();
        assert_eq!(m.shape(), &[1, 6]);
        let v: Vec<f32> = m.as_slice::<f32>().to_vec();
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
