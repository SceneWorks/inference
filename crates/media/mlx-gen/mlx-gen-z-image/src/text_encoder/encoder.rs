//! Full Z-Image text encoder: token embedding → N pre-norm decoder layers → the **second-to-
//! last** layer's hidden states (no final norm), cast to f32. Port of the fork's `TextEncoder`.
//! These hidden states are the DiT's `cap_feats` conditioning (after slicing to valid tokens).

use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, EncoderLayer, TextRope};

/// Z-Image text-encoder dimensions (Qwen3-style decoder LM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZTextEncoderConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl ZTextEncoderConfig {
    /// The production Z-Image-turbo text encoder (`Tongyi-MAI/Z-Image` `text_encoder/`).
    pub fn z_image() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

pub struct TextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<EncoderLayer>,
    rope: TextRope,
    /// Rung-4 text-encoder scope (SC-15794). `Some` ⇒ [`TextEncoder::forward_windowed`] may
    /// materialize the stack a window at a time **instead of** touching `layers`.
    ///
    /// `layers` stays populated either way, exactly as the DiT stream leaves its resident stack
    /// populated. Under `Sequential` those are unevaluated lazy handles costing ~0 bytes
    /// (SC-15744 measured 1073 DiT handles at 0.0 MiB), and a windowed forward never touches them, so
    /// they stay that way — while a plain [`forward`](Self::forward) on the same encoder still runs the
    /// full resident stack correctly. Emptying `layers` instead would make an unscoped `forward` return
    /// the bare embedding: a silent, catastrophic conditioning bug rather than a memory saving.
    stream: Option<super::stream::TextEncoderBlockStream>,
}

impl TextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ZTextEncoderConfig) -> Result<Self> {
        // Packed-detect (sc-8670): the token embedding loads packed from a pre-quantized snapshot
        // (the table is bf16-native, so the converter's bf16-cast pack is byte-equal to the
        // load-time `quantize(bits, cast_to_bf16=false)`) or dense otherwise.
        let embed_tokens = crate::quant::embedding(w, &join(prefix, "embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lp = join(prefix, &format!("layers.{i}"));
            layers.push(EncoderLayer::from_weights(
                w,
                &lp,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        // The fork also has a final `norm`, but it is never applied (the returned [-2] layer is
        // un-normed), so we don't load it.
        Ok(Self {
            embed_tokens,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            stream: None,
        })
    }

    /// Rung 4, text-encoder scope (SC-15794): load **only** the token embedding and keep the 36 layers
    /// in the re-openable snapshot, materializing them a window at a time in [`Self::forward_windowed`].
    ///
    /// `source` must be re-openable (a directory or a single `.safetensors`); an in-memory load has no
    /// such source and must keep using [`Self::from_weights`].
    ///
    /// The embedding stays resident because it is not block-shaped and is a small share of the encoder
    /// — measured 0.724 GiB of bf16's 7.440 GiB, 0.204 of q4's 2.132 (SC-15794). Streaming the layers
    /// is what moves the number.
    ///
    /// The resident `layers` are deliberately **not** built. Holding them as lazy handles alongside
    /// the stream was measured and costs most of the saving — q4 conditioning bounded to −38.3% with
    /// them versus −88.7% without (SC-15794) — because the constructor clones refcounted handles out
    /// of the view rather than leaving the tensors untouched.
    ///
    /// A plain [`forward`](Self::forward) on such an encoder is still correct: it detects the empty
    /// stack and runs the stream as a single full-width window, which is the same arithmetic at the
    /// same cost as resident. So this is a drop-in, and there is no state in which the encoder
    /// silently conditions on the bare embedding.
    /// Construct a streamed encoder whose source has already passed the provider's exact architecture
    /// contract. The validator is retained by the block stream and rechecked before every window opens
    /// its lazy view, so deferred materialization cannot outlive the validation boundary.
    pub fn from_validated_streamable_source(
        w: &Weights,
        source: mlx_gen::gen_core::ValidatedEncoderSource,
        prefix: &str,
        cfg: &ZTextEncoderConfig,
    ) -> Result<Self> {
        let embed_tokens = crate::quant::embedding(w, &join(prefix, "embed_tokens"))?;
        Ok(Self {
            embed_tokens,
            layers: Vec::new(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            stream: Some(super::stream::TextEncoderBlockStream::new_validated(
                source, prefix, *cfg,
            )),
        })
    }

    /// `true` when this encoder can run the rung-4 text-encoder scope.
    pub fn is_streamable(&self) -> bool {
        self.stream.is_some()
    }

    /// Quantize the encoder to Q4/Q8 (group_size 64): the token embedding + every layer's Linears
    /// — the full set the fork's `nn.quantize(text_encoder, …)` hits. Layer norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        // `cast_to_bf16 = false`: pack the weight as-is — the Z-Image behaviour preserved by the
        // shared `TokenEmbedding` (equivalent to the bf16-cast path for this bf16-native table) (F-083).
        self.embed_tokens.quantize(bits, false)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        // A streamed stack has no resident layers to quantize now, so record the bits and replay them
        // on every materialized layer instead. Without this a streamed q4/q8 encoder would run dense
        // layers against a packed embedding — correct-looking output at the wrong precision and the
        // wrong memory, which is precisely the silent failure rung 4 keeps producing when the streamed
        // and resident paths are not held byte-identical.
        if let Some(stream) = self.stream.as_mut() {
            stream.set_quant_bits(bits);
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `[b, s, hidden]` (f32) — the
    /// second-to-last layer's hidden states, matching the fork's `all_hidden_states[-2]`.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        // A streamable encoder holds no resident layers (that is where its saving comes from), so an
        // unscoped `forward` on one must run the stream rather than an empty stack — otherwise it
        // would silently return the bare token embedding as `cap_feats`, which is a catastrophic
        // conditioning bug that produces plausible-looking images. One full-width window is the same
        // arithmetic at the same cost as the resident stack, so this stays a drop-in.
        // `the_unscoped_forward_on_a_streamable_encoder_still_runs_every_layer` pins it.
        if self.layers.is_empty() {
            if let Some(stream) = self.stream.as_ref() {
                return self.forward_windowed(
                    input_ids,
                    attention_mask,
                    stream.n_blocks(),
                    &mlx_gen::CancelFlag::default(),
                );
            }
        }
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);

        let embed = self.embed_tokens.forward(input_ids)?; // [b, s, hidden] f32
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        // all_hidden_states = [embed, out(L0), out(L1), ...]; return the second-to-last. Track only
        // the previous + current layer output instead of retaining all 37 (~180 MB f32 at the real
        // hidden size) — F-089. `second_to_last` ends as all_hidden_states[len-2].
        let mut prev = embed; // all_hidden_states[0] = embed
        let mut second_to_last = prev.clone();
        for layer in &self.layers {
            let cur = layer.forward(&prev, &cos, &sin, &mask)?;
            // After this push all_hidden_states would be [..., prev, cur]; second-to-last is `prev`.
            second_to_last = std::mem::replace(&mut prev, cur);
        }
        Ok(second_to_last)
    }

    /// [`forward`](Self::forward) with rung 4's text-encoder scope applied (SC-15794): the layer stack
    /// is materialized `window` layers at a time from the snapshot rather than held resident.
    ///
    /// Everything outside the stack — the token embedding, RoPE, the mask build, and the
    /// second-to-last selection — is byte-for-byte the resident path, and each streamed layer is
    /// quantized identically to its resident twin. So the two forwards compute the same arithmetic in
    /// the same order; only *when the weights exist* differs.
    ///
    /// Errors when this encoder has no re-openable source, rather than silently running resident: a
    /// caller that selected the rung must not be told it ran when it did not.
    pub fn forward_windowed(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        window: usize,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<Array> {
        let stream = self.stream.as_ref().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "z-image: the rung-4 text-encoder scope was requested but this encoder has no \
                 re-openable weights source (an in-memory / ComfyUI load); the contract declares the \
                 text-encoder component unavailable for such a load"
                    .to_owned(),
            )
        })?;
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let embed = self.embed_tokens.forward(input_ids)?;
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;
        let plan = mlx_gen::block_residency::BlockPlan::new(stream.n_blocks(), window)?;
        super::stream::run_windowed_layers(stream, &plan, cancel, embed, &cos, &sin, &mask)
    }
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and**
/// not padding), `-inf` otherwise — the fork's causal ⊕ padding combination.
///
/// Built host-side (a one-time `O(b·s²)` fill per prompt encode, **not** per denoise step).
/// Deliberately kept on the host rather than constructed with on-device broadcast ops: at realistic
/// prompt lengths this is negligible against the denoise loop, and a plain fill is the simplest way
/// to stay bit-exact with the fork (sc-2583). Revisit only if profiling ever flags it.
fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = host_i32(attention_mask)?;
    let (b, s) = (b as usize, s as usize);
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_causal_and_masks_padding() {
        // b=1, s=3, last token padded.
        let am = Array::from_slice(&[1i32, 1, 0], &[1, 3]);
        let m = build_mask(&am, 1, 3).unwrap();
        let v = m.as_slice::<f32>(); // [1,1,3,3] -> 9 values, row-major [query][key]
        let neg = f32::NEG_INFINITY;
        // query 0: key0 allowed, key1 future, key2 future+pad
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], neg);
        assert_eq!(v[2], neg);
        // query 1: key0,key1 allowed, key2 future
        assert_eq!(v[3], 0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], neg);
        // query 2: key0,key1 allowed (causal), key2 padded -> masked
        assert_eq!(v[6], 0.0);
        assert_eq!(v[7], 0.0);
        assert_eq!(v[8], neg);
    }
}
