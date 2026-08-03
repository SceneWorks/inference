//! FLUX.1 prompt text path: CLIP pooled prompt embedding + T5 sequence prompt embedding.
//! Ports the fork's `flux_text_encoder` modules directly.

use crate::quant::GROUP_SIZE;
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::{host_i32, scalar};
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::{join, Weights};
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{
    add, dequantize, matmul, multiply, power, quantize, sigmoid, softmax_axis, subtract,
};
use mlx_rs::{Array, Dtype};

pub struct FluxTextEncoders {
    pub t5: T5TextEncoder,
    pub clip: ClipTextEncoder,
}

impl FluxTextEncoders {
    pub fn encode(&self, t5_ids: &Array, clip_ids: &Array) -> Result<(Array, Array)> {
        Ok((self.t5.forward(t5_ids)?, self.clip.forward(clip_ids)?))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.t5.quantize(bits)?;
        self.clip.quantize(bits)?;
        Ok(())
    }
}

enum TokenEmbedding {
    Dense(Array),
    Quantized {
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
        residual: Option<PackedTerm>,
        residual2: Option<PackedTerm>,
    },
}

struct PackedTerm {
    wq: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

fn validate_t5_group_size(group_size: i32) -> Result<()> {
    // T5's relative-attention-bias table has logical width 64. Group 128 is valid for MLX affine
    // quantization generally, but cannot pack that table and would violate complete-surface parity.
    if matches!(group_size, 32 | 64) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "T5 quantization group size must be 32 or 64, got {group_size}"
        )))
    }
}

impl TokenEmbedding {
    /// Load `{base}` — **packed** ([`Self::Quantized`]) when `{base}.scales` is present (a
    /// pre-quantized snapshot; bit-width inferred from the packed shapes at `group_size`), else
    /// **dense** ([`Self::Dense`]). The embedding analogue of [`crate::quant::lin`] for this local
    /// enum (sc-8669), so a published Q4/Q8 text encoder loads its token/position/relative-bias
    /// tables packed with no dense bf16 transient.
    fn from_weights(w: &Weights, base: &str, group_size: i32) -> Result<Self> {
        if let Some(scales) = w.get(&format!("{base}.scales")) {
            let wq = w.require(&format!("{base}.weight"))?.clone();
            // F-011: reuse the shared, validated bit-width derivation (mlx_gen::quant::packed_bits)
            // instead of a local copy that panic'd on 1-D shapes / divided by zero on [vocab,0].
            let bits = mlx_gen::quant::packed_bits(&wq, scales, group_size)?;
            return Ok(Self::Quantized {
                wq,
                scales: scales.clone(),
                biases: w.require(&format!("{base}.biases"))?.clone(),
                group_size,
                bits,
                residual: load_packed_term(w, &format!("{base}.residual"), group_size)?,
                // The second term is currently an in-memory calibration seam only. Chroma's
                // fail-closed packed-surface validator must learn its exact provenance before any
                // `.residual2.*` artifact is accepted on disk.
                residual2: None,
            });
        }
        Ok(Self::Dense(w.require(&format!("{base}.weight"))?.clone()))
    }

    fn forward(&self, ids: &Array) -> Result<Array> {
        let out = match self {
            Self::Dense(w) => w.take_axis(ids, 0)?,
            Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
                residual,
                residual2,
            } => {
                let pw = wq.take_axis(ids, 0)?;
                let sc = scales.take_axis(ids, 0)?;
                let bi = biases.take_axis(ids, 0)?;
                let primary = dequantize(&pw, &sc, &bi, *group_size, *bits)?;
                let with_residual = match residual {
                    Some(residual) => {
                        let rw = residual.wq.take_axis(ids, 0)?;
                        let rs = residual.scales.take_axis(ids, 0)?;
                        let rb = residual.biases.take_axis(ids, 0)?;
                        add(
                            &primary,
                            &dequantize(&rw, &rs, &rb, residual.group_size, residual.bits)?,
                        )?
                    }
                    None => primary,
                };
                match residual2 {
                    Some(residual2) => {
                        let rw = residual2.wq.take_axis(ids, 0)?;
                        let rs = residual2.scales.take_axis(ids, 0)?;
                        let rb = residual2.biases.take_axis(ids, 0)?;
                        add(
                            &with_residual,
                            &dequantize(&rw, &rs, &rb, residual2.group_size, residual2.bits)?,
                        )?
                    }
                    None => with_residual,
                }
            }
        };
        // Return the native (bf16) embedding to match the mflux reference (sc-2787). CLIP genuinely
        // runs bf16 (its `nn.LayerNorm` fast kernel returns bf16); T5 immediately upcasts to f32 in
        // its `T5LayerNorm` (variance `astype(f32)`, which MLX promotion propagates through the whole
        // encoder), so T5 stays f32-internally either way — the FLUX checkpoint is bf16-native, so the
        // bf16↔f32 cast is lossless here. The old MANDATORY-f32 comment was bug-forced: T5/CLIP
        // attention is bf16×bf16 K≤512 (the [[pmetal-mlx-bf16-matmul-bug]] dense 16-bit GEMM), now
        // fixed by sc-2772 (NAX metal target ≥26.2) — so bf16 is correct AND the parity dtype.
        Ok(out)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.quantize_with_group_size(bits, GROUP_SIZE)
    }

    fn quantize_with_group_size(&mut self, bits: i32, group_size: i32) -> Result<()> {
        if let Self::Dense(w) = self {
            let (wq, scales, biases) = quantize(&w.as_dtype(Dtype::Bfloat16)?, group_size, bits)?;
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
                residual: None,
                residual2: None,
            };
        }
        Ok(())
    }

    fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        if let Self::Dense(w) = self {
            let wbf16 = w.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            let restored = dequantize(&wq, &scales, &biases, group_size, bits)?;
            let residual = subtract(&wbf16, &restored)?;
            let (residual_wq, residual_scales, residual_biases) =
                quantize(&residual, group_size, residual_bits)?;
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
                residual: Some(PackedTerm {
                    wq: residual_wq,
                    scales: residual_scales,
                    biases: residual_biases,
                    group_size,
                    bits: residual_bits,
                }),
                residual2: None,
            };
        }
        Ok(())
    }

    fn quantize_progressive_with_secondary(
        &mut self,
        bits: i32,
        residual_bits: i32,
        secondary_bits: Option<i32>,
        group_size: i32,
    ) -> Result<()> {
        if let Self::Dense(w) = self {
            let wbf16 = w.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            let restored = dequantize(&wq, &scales, &biases, group_size, bits)?;
            let residual = subtract(&wbf16, &restored)?;
            let (residual_wq, residual_scales, residual_biases) =
                quantize(&residual, group_size, residual_bits)?;
            let residual2 = if let Some(secondary_bits) = secondary_bits {
                let restored_residual = dequantize(
                    &residual_wq,
                    &residual_scales,
                    &residual_biases,
                    group_size,
                    residual_bits,
                )?;
                let secondary = subtract(&residual, &restored_residual)?;
                let (wq, scales, biases) = quantize(&secondary, group_size, secondary_bits)?;
                Some(PackedTerm {
                    wq,
                    scales,
                    biases,
                    group_size,
                    bits: secondary_bits,
                })
            } else {
                None
            };
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
                residual: Some(PackedTerm {
                    wq: residual_wq,
                    scales: residual_scales,
                    biases: residual_biases,
                    group_size,
                    bits: residual_bits,
                }),
                residual2,
            };
        }
        Ok(())
    }
}

fn load_packed_term(w: &Weights, base: &str, group_size: i32) -> Result<Option<PackedTerm>> {
    let Some(scales) = w.get(&format!("{base}.scales")) else {
        return Ok(None);
    };
    let wq = w.require(&format!("{base}.weight"))?.clone();
    let bits = mlx_gen::quant::packed_bits(&wq, scales, group_size)?;
    Ok(Some(PackedTerm {
        wq,
        scales: scales.clone(),
        biases: w.require(&format!("{base}.biases"))?.clone(),
        group_size,
        bits,
    }))
}

pub struct ClipTextEncoder {
    token_embedding: TokenEmbedding,
    position_embedding: TokenEmbedding,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
}

impl ClipTextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |suffix: &str| join(prefix, suffix);
        let mut layers = Vec::with_capacity(12);
        for i in 0..12 {
            layers.push(ClipEncoderLayer::from_weights(
                w,
                &p(&format!("text_model.encoder.layers.{i}")),
            )?);
        }
        Ok(Self {
            token_embedding: TokenEmbedding::from_weights(
                w,
                &p("text_model.embeddings.token_embedding"),
                GROUP_SIZE,
            )?,
            position_embedding: TokenEmbedding::from_weights(
                w,
                &p("text_model.embeddings.position_embedding"),
                GROUP_SIZE,
            )?,
            layers,
            final_ln_w: w.require(&p("text_model.final_layer_norm.weight"))?.clone(),
            final_ln_b: w.require(&p("text_model.final_layer_norm.bias"))?.clone(),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.token_embedding.quantize(bits)?;
        self.position_embedding.quantize(bits)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `tokens`: `[1, 77]` int32. Returns pooled CLIP embedding `[1, 768]`, selected at the
    /// highest token id (the fork's `mx.argmax(tokens, axis=-1)`).
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        let s = tokens.shape()[1];
        let token = self.token_embedding.forward(tokens)?;
        let pos_ids: Vec<i32> = (0..s).collect();
        let pos_ids = Array::from_slice(&pos_ids, &[1, s]);
        let pos = self.position_embedding.forward(&pos_ids)?;
        let mut hidden = add(&token, &pos)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        let hidden = layer_norm(
            &hidden,
            Some(&self.final_ln_w),
            Some(&self.final_ln_b),
            1e-5,
        )?;
        let token_ids = host_i32(tokens)?;
        // Pooled output is the hidden state at the *first* argmax of the token ids — the fork's
        // `mx.argmax(tokens, axis=-1)` (first occurrence on ties). CLIP pads to 77 with the EOS id
        // (49407), so the EOS and every pad token tie; `Iterator::max_by_key` would return the
        // LAST tie (a pad position) instead of the EOS, picking the wrong pooled vector.
        let max_id = token_ids.iter().copied().max().unwrap_or(0);
        let idx = token_ids.iter().position(|&id| id == max_id).unwrap_or(0) as i32;
        let flat = hidden.reshape(&[s, 768])?;
        let idx = Array::from_slice(&[idx], &[1]);
        Ok(flat.take_axis(&idx, 0)?)
    }
}

struct ClipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    attn: ClipAttention,
    ln2_w: Array,
    ln2_b: Array,
    mlp: ClipMlp,
}

impl ClipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?.clone(),
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?.clone(),
            attn: ClipAttention::from_weights(w, &join(prefix, "self_attn"))?,
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?.clone(),
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?.clone(),
            mlp: ClipMlp::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let residual = hidden;
        let normed = layer_norm(hidden, Some(&self.ln1_w), Some(&self.ln1_b), 1e-5)?;
        let hidden = add(residual, &self.attn.forward(&normed)?)?;
        let residual = hidden.clone();
        let normed = layer_norm(&hidden, Some(&self.ln2_w), Some(&self.ln2_b), 1e-5)?;
        Ok(add(&residual, &self.mlp.forward(&normed)?)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }
}

struct ClipAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
}

impl ClipAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `{name}.scales` is present, else dense.
        let linear = |name: &str| crate::quant::lin(w, &join(prefix, name), true);
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            out: linear("out_proj")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let b = hidden.shape()[0];
        let s = hidden.shape()[1];
        // Read the batch from the input instead of hardcoding 1, so a B>1 CLIP encode reshapes
        // correctly rather than shape-erroring / mis-shaping (F-061).
        let q = self
            .q
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        // CLIP text attention is purely causal (no key-padding term — pads are attended causally),
        // so use the implicit causal mode instead of materializing an `s·s` additive mask host-side
        // each encode (F-040). q_len == k_len here, so the modes are equivalent.
        let y = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (64.0_f32).powf(-0.5),
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let y = y.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, 768])?;
        self.out.forward(&y)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.out.quantize(bits, None)?;
        Ok(())
    }
}

struct ClipMlp {
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl ClipMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `{name}.scales` is present, else dense.
        let linear = |name: &str| crate::quant::lin(w, &join(prefix, name), true);
        Ok(Self {
            fc1: linear("fc1")?,
            fc2: linear("fc2")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let hidden = self.fc1.forward(hidden)?;
        let hidden = quick_gelu(&hidden)?;
        self.fc2.forward(&hidden)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.fc1.quantize(bits, None)?;
        self.fc2.quantize(bits, None)?;
        Ok(())
    }
}

struct T5Linear {
    primary: AdaptableLinear,
    residual: Option<AdaptableLinear>,
    residual2: Option<AdaptableLinear>,
}

impl T5Linear {
    fn from_weights(w: &Weights, base: &str, group_size: i32) -> Result<Self> {
        let primary = mlx_gen::quant::lin(w, base, false, group_size)?;
        let residual = load_packed_term(w, &format!("{base}.residual"), group_size)?.map(|term| {
            AdaptableLinear::from_quantized_parts(
                term.wq,
                term.scales,
                term.biases,
                None,
                term.group_size,
                term.bits,
            )
        });
        Ok(Self {
            primary,
            residual,
            // In-memory calibration only until the provider's exact-set artifact validator and
            // provenance schema support a second packed residual term.
            residual2: None,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let primary = self.primary.forward(hidden)?;
        let with_residual = match &self.residual {
            Some(residual) => add(&primary, &residual.forward(hidden)?)?,
            None => primary,
        };
        match &self.residual2 {
            Some(residual2) => Ok(add(&with_residual, &residual2.forward(hidden)?)?),
            None => Ok(with_residual),
        }
    }

    fn quantize(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.primary.quantize(bits, Some(group_size))
    }

    fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        if self.residual.is_some() {
            return Ok(());
        }
        let Some((weight, bias)) = self.primary.dense_weight() else {
            return Err(Error::Msg(
                "T5 progressive quantization requires a dense source weight".into(),
            ));
        };
        if bias.is_some() {
            return Err(Error::Msg(
                "T5 progressive quantization does not support biased linears".into(),
            ));
        }
        let wbf16 = weight.as_dtype(Dtype::Bfloat16)?;
        let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
        let restored = dequantize(&wq, &scales, &biases, group_size, bits)?;
        let residual = subtract(&wbf16, &restored)?;
        let (residual_wq, residual_scales, residual_biases) =
            quantize(&residual, group_size, residual_bits)?;
        self.primary =
            AdaptableLinear::from_quantized_parts(wq, scales, biases, None, group_size, bits);
        self.residual = Some(AdaptableLinear::from_quantized_parts(
            residual_wq,
            residual_scales,
            residual_biases,
            None,
            group_size,
            residual_bits,
        ));
        Ok(())
    }

    fn quantize_progressive_with_secondary(
        &mut self,
        bits: i32,
        residual_bits: i32,
        secondary_bits: Option<i32>,
        group_size: i32,
    ) -> Result<()> {
        if self.residual.is_some() || self.residual2.is_some() {
            return Ok(());
        }
        let Some((weight, bias)) = self.primary.dense_weight() else {
            return Err(Error::Msg(
                "T5 progressive quantization requires a dense source weight".into(),
            ));
        };
        if bias.is_some() {
            return Err(Error::Msg(
                "T5 progressive quantization does not support biased linears".into(),
            ));
        }
        let wbf16 = weight.as_dtype(Dtype::Bfloat16)?;
        let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
        let restored = dequantize(&wq, &scales, &biases, group_size, bits)?;
        let residual = subtract(&wbf16, &restored)?;
        let (residual_wq, residual_scales, residual_biases) =
            quantize(&residual, group_size, residual_bits)?;
        self.residual2 = if let Some(secondary_bits) = secondary_bits {
            let restored_residual = dequantize(
                &residual_wq,
                &residual_scales,
                &residual_biases,
                group_size,
                residual_bits,
            )?;
            let secondary = subtract(&residual, &restored_residual)?;
            let (wq, scales, biases) = quantize(&secondary, group_size, secondary_bits)?;
            Some(AdaptableLinear::from_quantized_parts(
                wq,
                scales,
                biases,
                None,
                group_size,
                secondary_bits,
            ))
        } else {
            None
        };
        self.primary =
            AdaptableLinear::from_quantized_parts(wq, scales, biases, None, group_size, bits);
        self.residual = Some(AdaptableLinear::from_quantized_parts(
            residual_wq,
            residual_scales,
            residual_biases,
            None,
            group_size,
            residual_bits,
        ));
        Ok(())
    }
}

pub struct T5TextEncoder {
    shared: TokenEmbedding,
    blocks: Vec<T5Block>,
    final_ln_w: Array,
}

/// A residual sublayer that may remain at source precision while the rest of T5's Linear surface
/// is quantized. Chroma uses this to calibrate the smallest quality-preserving packed policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum T5Sublayer {
    Attention,
    FeedForward,
}

impl T5TextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights_with_group_size(w, prefix, GROUP_SIZE)
    }

    /// Load T5 with the group size declared by its packed component manifest. Dense FLUX callers
    /// retain the default group size; Chroma uses this seam for its independently calibrated T5
    /// artifacts.
    pub fn from_weights_with_group_size(
        w: &Weights,
        prefix: &str,
        group_size: i32,
    ) -> Result<Self> {
        validate_t5_group_size(group_size)?;
        let p = |suffix: &str| join(prefix, suffix);
        let mut blocks = Vec::with_capacity(24);
        for i in 0..24 {
            blocks.push(T5Block::from_weights(
                w,
                &p(&format!("encoder.block.{i}")),
                group_size,
            )?);
        }
        Ok(Self {
            shared: TokenEmbedding::from_weights(w, &p("shared"), group_size)?,
            blocks,
            final_ln_w: w.require(&p("encoder.final_layer_norm.weight"))?.clone(),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.quantize_with_group_size(bits, GROUP_SIZE)
    }

    /// Quantize the complete packable T5 surface at an explicit MLX affine group size.
    pub fn quantize_with_group_size(&mut self, bits: i32, group_size: i32) -> Result<()> {
        validate_t5_group_size(group_size)?;
        self.shared.quantize_with_group_size(bits, group_size)?;
        for block in &mut self.blocks {
            block.quantize(bits, group_size)?;
        }
        Ok(())
    }

    /// Quantize the complete packable T5 surface as a primary affine pack plus a packed residual.
    /// Both terms remain quantized at runtime; the second term improves fidelity without restoring
    /// any dense T5 weight or load-time dense transient.
    pub fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        self.quantize_progressive_with_sensitive_residuals(
            bits,
            residual_bits,
            residual_bits,
            group_size,
        )
    }

    /// Progressively quantize the complete packable T5 surface while giving the shared token
    /// embedding and relative-position bias an independently selected residual width. Those two
    /// boundary tables are reused across the full residual stream, so providers can spend a small
    /// Q8 correction there while retaining Q4 residuals for the large attention/FFN projections.
    pub fn quantize_progressive_with_sensitive_residuals(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        self.quantize_progressive_with_sensitive_sublayer_residuals(
            bits,
            residual_bits,
            sensitive_residual_bits,
            group_size,
            None,
        )
    }

    /// Progressively quantize T5 while additionally giving one attention or feed-forward block
    /// the sensitive residual width. This keeps every source weight packed and provides a narrow
    /// calibration seam for providers whose strict end-to-end quality gate needs more precision
    /// than the shared embedding and relative-position bias alone.
    pub fn quantize_progressive_with_sensitive_sublayer_residuals(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        group_size: i32,
        sensitive_sublayer: Option<(usize, T5Sublayer)>,
    ) -> Result<()> {
        match sensitive_sublayer {
            Some(selected) => self.quantize_progressive_with_sensitive_sublayers_residuals(
                bits,
                residual_bits,
                sensitive_residual_bits,
                group_size,
                &[selected],
            ),
            None => self.quantize_progressive_with_sensitive_sublayers_residuals(
                bits,
                residual_bits,
                sensitive_residual_bits,
                group_size,
                &[],
            ),
        }
    }

    /// Progressively quantize T5 while giving a calibrated set of attention or feed-forward
    /// sublayers the sensitive residual width. Selections change only packed residual widths; they
    /// never retain a dense source projection.
    pub fn quantize_progressive_with_sensitive_sublayers_residuals(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        group_size: i32,
        sensitive_sublayers: &[(usize, T5Sublayer)],
    ) -> Result<()> {
        validate_t5_group_size(group_size)?;
        if !matches!(bits, 4 | 8)
            || !matches!(residual_bits, 4 | 8)
            || !matches!(sensitive_residual_bits, 4 | 8)
        {
            return Err(Error::Msg(format!(
                "T5 progressive quantization widths must be Q4 or Q8, got Q{bits} + Q{residual_bits}/Q{sensitive_residual_bits} residuals"
            )));
        }
        for &(block, _) in sensitive_sublayers {
            if block >= self.blocks.len() {
                return Err(Error::Msg(format!(
                    "T5 sensitive-residual block {block} is outside 0..{}",
                    self.blocks.len()
                )));
            }
        }
        self.shared
            .quantize_progressive(bits, sensitive_residual_bits, group_size)?;
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let attention_is_sensitive =
                sensitive_sublayers.contains(&(index, T5Sublayer::Attention));
            let feed_forward_is_sensitive =
                sensitive_sublayers.contains(&(index, T5Sublayer::FeedForward));
            block.quantize_progressive(
                bits,
                residual_bits,
                sensitive_residual_bits,
                group_size,
                attention_is_sensitive,
                feed_forward_is_sensitive,
            )?;
        }
        Ok(())
    }

    /// In-memory calibration seam for a third packed affine term on a narrowly selected T5 surface.
    /// The primary and first residual retain the normal progressive policy; selected boundaries and
    /// sublayers receive another Q4/Q8 correction. Every runtime term remains packed. This method
    /// deliberately does not define an on-disk format; provider provenance validation must be
    /// extended separately after hosted calibration selects an exact surface.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_progressive_with_secondary_residuals(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        secondary_bits: i32,
        group_size: i32,
        sensitive_sublayers: &[(usize, T5Sublayer)],
        secondary_boundaries: bool,
        secondary_sublayers: &[(usize, T5Sublayer)],
    ) -> Result<()> {
        validate_t5_group_size(group_size)?;
        if !matches!(bits, 4 | 8)
            || !matches!(residual_bits, 4 | 8)
            || !matches!(sensitive_residual_bits, 4 | 8)
            || !matches!(secondary_bits, 4 | 8)
        {
            return Err(Error::Msg(format!(
                "T5 progressive quantization widths must be Q4 or Q8, got Q{bits} + Q{residual_bits}/Q{sensitive_residual_bits} + Q{secondary_bits}"
            )));
        }
        for &(block, _) in sensitive_sublayers.iter().chain(secondary_sublayers) {
            if block >= self.blocks.len() {
                return Err(Error::Msg(format!(
                    "T5 sensitive-residual block {block} is outside 0..{}",
                    self.blocks.len()
                )));
            }
        }
        self.shared.quantize_progressive_with_secondary(
            bits,
            sensitive_residual_bits,
            secondary_boundaries.then_some(secondary_bits),
            group_size,
        )?;
        for (index, block) in self.blocks.iter_mut().enumerate() {
            block.quantize_progressive_with_secondary(
                bits,
                residual_bits,
                sensitive_residual_bits,
                secondary_bits,
                group_size,
                sensitive_sublayers.contains(&(index, T5Sublayer::Attention)),
                sensitive_sublayers.contains(&(index, T5Sublayer::FeedForward)),
                secondary_boundaries,
                secondary_sublayers.contains(&(index, T5Sublayer::Attention)),
                secondary_sublayers.contains(&(index, T5Sublayer::FeedForward)),
            )?;
        }
        Ok(())
    }

    /// Quantize the large attention/FFN Linear surface while retaining the token-embedding and
    /// relative-position-bias tables at their source precision. Chroma uses this sensitivity-
    /// calibration policy for its packed auxiliary artifacts; hosted evidence determines whether
    /// retaining those two boundary tables sufficiently reduces perturbation through the 24-layer
    /// residual stream while the packed Linear surface removes the full dense-T5 residency.
    pub fn quantize_linears(&mut self, bits: i32) -> Result<()> {
        for block in &mut self.blocks {
            block.quantize_linears(bits)?;
        }
        Ok(())
    }

    /// Quantize every attention/FFN Linear except one explicitly selected residual sublayer.
    ///
    /// This is the narrow calibration seam used by Chroma's real-weight sensitivity sweep. The
    /// caller must mirror the selected dense sublayer in any packed artifact predicate before the
    /// policy can ship.
    pub fn quantize_linears_except(
        &mut self,
        bits: i32,
        dense_block: usize,
        dense_sublayer: T5Sublayer,
    ) -> Result<()> {
        if dense_block >= self.blocks.len() {
            return Err(Error::Msg(format!(
                "T5 dense carve-out block {dense_block} is outside 0..{}",
                self.blocks.len()
            )));
        }
        for (index, block) in self.blocks.iter_mut().enumerate() {
            block
                .quantize_linears_except(bits, (index == dense_block).then_some(dense_sublayer))?;
        }
        Ok(())
    }

    /// `tokens`: `[1, L]` int32. Returns T5 sequence embeddings `[1, L, 4096]`.
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        self.forward_masked(tokens, None)
    }

    /// As [`forward`](Self::forward), but with an optional **additive** key-padding mask (broadcastable
    /// to the attention scores `[1, heads, L, L]`, e.g. `[1, 1, 1, L]` with a large negative at padded
    /// keys). Chroma (epic 3531) runs T5 with the tokenizer padding mask — unlike FLUX, which runs T5
    /// unmasked. `mask = None` is **byte-identical** to [`forward`](Self::forward).
    pub fn forward_masked(&self, tokens: &Array, mask: Option<&Array>) -> Result<Array> {
        let mut hidden = self.shared.forward(tokens)?;
        // The relative-position bias depends only on seq_len and is identical across all blocks (only
        // block 0 carries the table; every other block clones it), so compute it once here and share
        // it instead of rebuilding the O(L²) gather inside each of the 24 blocks (F-099).
        if let Some(block0) = self.blocks.first() {
            let bias = block0.attn.position_bias(hidden.shape()[1])?;
            for block in &self.blocks {
                hidden = block.forward(&hidden, mask, &bias)?;
            }
        }
        t5_rms_norm(&hidden, &self.final_ln_w, 1e-6)
    }
}

struct T5Block {
    attn: T5Attention,
    ff: T5FeedForward,
}

impl T5Block {
    fn from_weights(w: &Weights, prefix: &str, group_size: i32) -> Result<Self> {
        Ok(Self {
            attn: T5Attention::from_weights(w, &join(prefix, "layer.0"), group_size)?,
            ff: T5FeedForward::from_weights(w, &join(prefix, "layer.1"), group_size)?,
        })
    }

    fn forward(&self, hidden: &Array, mask: Option<&Array>, bias: &Array) -> Result<Array> {
        let hidden = self.attn.forward(hidden, mask, bias)?;
        self.ff.forward(&hidden)
    }

    fn quantize(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.attn.quantize_with_group_size(bits, group_size)?;
        self.ff.quantize_with_group_size(bits, group_size)?;
        Ok(())
    }

    fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        group_size: i32,
        attention_is_sensitive: bool,
        feed_forward_is_sensitive: bool,
    ) -> Result<()> {
        let attention_residual_bits = if attention_is_sensitive {
            sensitive_residual_bits
        } else {
            residual_bits
        };
        let feed_forward_residual_bits = if feed_forward_is_sensitive {
            sensitive_residual_bits
        } else {
            residual_bits
        };
        self.attn.quantize_progressive(
            bits,
            attention_residual_bits,
            sensitive_residual_bits,
            group_size,
        )?;
        self.ff
            .quantize_progressive(bits, feed_forward_residual_bits, group_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize_progressive_with_secondary(
        &mut self,
        bits: i32,
        residual_bits: i32,
        sensitive_residual_bits: i32,
        secondary_bits: i32,
        group_size: i32,
        attention_is_sensitive: bool,
        feed_forward_is_sensitive: bool,
        secondary_boundaries: bool,
        secondary_attention: bool,
        secondary_feed_forward: bool,
    ) -> Result<()> {
        let attention_residual_bits = if attention_is_sensitive {
            sensitive_residual_bits
        } else {
            residual_bits
        };
        let feed_forward_residual_bits = if feed_forward_is_sensitive {
            sensitive_residual_bits
        } else {
            residual_bits
        };
        self.attn.quantize_progressive_with_secondary(
            bits,
            attention_residual_bits,
            sensitive_residual_bits,
            secondary_bits,
            group_size,
            secondary_boundaries,
            secondary_attention,
        )?;
        self.ff.quantize_progressive_with_secondary(
            bits,
            feed_forward_residual_bits,
            secondary_feed_forward.then_some(secondary_bits),
            group_size,
        )
    }

    fn quantize_linears(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize_linears(bits)?;
        self.ff.quantize(bits)?;
        Ok(())
    }

    fn quantize_linears_except(
        &mut self,
        bits: i32,
        dense_sublayer: Option<T5Sublayer>,
    ) -> Result<()> {
        if dense_sublayer != Some(T5Sublayer::Attention) {
            self.attn.quantize_linears(bits)?;
        }
        if dense_sublayer != Some(T5Sublayer::FeedForward) {
            self.ff.quantize(bits)?;
        }
        Ok(())
    }
}

struct T5Attention {
    ln_w: Array,
    q: T5Linear,
    k: T5Linear,
    v: T5Linear,
    o: T5Linear,
    rel_bias: TokenEmbedding,
}

impl T5Attention {
    fn from_weights(w: &Weights, prefix: &str, group_size: i32) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `.scales` is present, else dense.
        let linear = |name: &str| {
            T5Linear::from_weights(
                w,
                &join(prefix, &format!("SelfAttention.{name}")),
                group_size,
            )
        };
        Ok(Self {
            ln_w: w.require(&join(prefix, "layer_norm.weight"))?.clone(),
            q: linear("q")?,
            k: linear("k")?,
            v: linear("v")?,
            o: linear("o")?,
            rel_bias: {
                // The relative-attention bias lives only on block 0 and is shared across all blocks,
                // so blocks 1+ fall back to the block-0 key. Packed-detect via `from_weights` picks
                // up `.scales` on whichever key is present (sc-8669).
                let own = join(prefix, "SelfAttention.relative_attention_bias");
                let base = if w.get(&format!("{own}.weight")).is_some() {
                    own
                } else {
                    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias".to_string()
                };
                TokenEmbedding::from_weights(w, &base, group_size)?
            },
        })
    }

    /// `bias` is the shared relative-position bias for this seq_len, precomputed once per forward in
    /// [`T5TextEncoder::forward_masked`] (it is identical across all blocks — F-099).
    fn forward(&self, hidden: &Array, mask: Option<&Array>, bias: &Array) -> Result<Array> {
        let normed = t5_rms_norm(hidden, &self.ln_w, 1e-6)?;
        let q = shape_t5(&self.q.forward(&normed)?)?;
        let k = shape_t5(&self.k.forward(&normed)?)?;
        let v = shape_t5(&self.v.forward(&normed)?)?;
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        // Chroma key-padding mask (epic 3531): additive, broadcast over query/heads. `None` for FLUX.
        let biased = match mask {
            Some(m) => add(&add(&scores, bias)?, m)?,
            None => add(&scores, bias)?,
        };
        let weights = softmax_axis(&biased, -1, false)?;
        let attn = unshape_t5(&matmul(&weights, &v)?)?;
        Ok(add(hidden, &self.o.forward(&attn)?)?)
    }

    fn position_bias(&self, seq_len: i32) -> Result<Array> {
        let buckets = relative_position_buckets(seq_len);
        let idx = Array::from_slice(&buckets, &[seq_len, seq_len]);
        let values = self.rel_bias.forward(&idx)?;
        Ok(values.transpose_axes(&[2, 0, 1])?.expand_dims(0)?)
    }

    fn quantize_with_group_size(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.q.quantize(bits, group_size)?;
        self.k.quantize(bits, group_size)?;
        self.v.quantize(bits, group_size)?;
        self.o.quantize(bits, group_size)?;
        self.rel_bias.quantize_with_group_size(bits, group_size)?;
        Ok(())
    }

    fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        relative_bias_residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        self.q
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.k
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.v
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.o
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.rel_bias
            .quantize_progressive(bits, relative_bias_residual_bits, group_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize_progressive_with_secondary(
        &mut self,
        bits: i32,
        residual_bits: i32,
        relative_bias_residual_bits: i32,
        secondary_bits: i32,
        group_size: i32,
        secondary_boundary: bool,
        secondary_attention: bool,
    ) -> Result<()> {
        let linear_secondary = secondary_attention.then_some(secondary_bits);
        self.q.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            linear_secondary,
            group_size,
        )?;
        self.k.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            linear_secondary,
            group_size,
        )?;
        self.v.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            linear_secondary,
            group_size,
        )?;
        self.o.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            linear_secondary,
            group_size,
        )?;
        self.rel_bias.quantize_progressive_with_secondary(
            bits,
            relative_bias_residual_bits,
            secondary_boundary.then_some(secondary_bits),
            group_size,
        )
    }

    fn quantize_linears(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, GROUP_SIZE)?;
        self.k.quantize(bits, GROUP_SIZE)?;
        self.v.quantize(bits, GROUP_SIZE)?;
        self.o.quantize(bits, GROUP_SIZE)?;
        Ok(())
    }
}

struct T5FeedForward {
    ln_w: Array,
    wi0: T5Linear,
    wi1: T5Linear,
    wo: T5Linear,
}

impl T5FeedForward {
    fn from_weights(w: &Weights, prefix: &str, group_size: i32) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `.scales` is present, else dense.
        let linear = |name: &str| {
            T5Linear::from_weights(
                w,
                &join(prefix, &format!("DenseReluDense.{name}")),
                group_size,
            )
        };
        Ok(Self {
            ln_w: w.require(&join(prefix, "layer_norm.weight"))?.clone(),
            wi0: linear("wi_0")?,
            wi1: linear("wi_1")?,
            wo: linear("wo")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let normed = t5_rms_norm(hidden, &self.ln_w, 1e-6)?;
        // Shared dtype-preserving tanh-GELU (sc-2779). Replaces the local `new_gelu`, whose f32
        // `√(2/π)` constant was 1 ULP off the fork's f64-host value (see [[mlx-rs-gelu-approx-f64-constant]]);
        // `gelu_tanh` computes the constant in f64 and preserves the input dtype.
        let gelu = gelu_tanh(&self.wi0.forward(&normed)?)?;
        let linear = self.wi1.forward(&normed)?;
        let ff = self.wo.forward(&multiply(&gelu, &linear)?)?;
        Ok(add(hidden, &ff)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.quantize_with_group_size(bits, GROUP_SIZE)
    }

    fn quantize_with_group_size(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.wi0.quantize(bits, group_size)?;
        self.wi1.quantize(bits, group_size)?;
        self.wo.quantize(bits, group_size)?;
        Ok(())
    }

    fn quantize_progressive(
        &mut self,
        bits: i32,
        residual_bits: i32,
        group_size: i32,
    ) -> Result<()> {
        self.wi0
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.wi1
            .quantize_progressive(bits, residual_bits, group_size)?;
        self.wo
            .quantize_progressive(bits, residual_bits, group_size)
    }

    fn quantize_progressive_with_secondary(
        &mut self,
        bits: i32,
        residual_bits: i32,
        secondary_bits: Option<i32>,
        group_size: i32,
    ) -> Result<()> {
        self.wi0.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            secondary_bits,
            group_size,
        )?;
        self.wi1.quantize_progressive_with_secondary(
            bits,
            residual_bits,
            secondary_bits,
            group_size,
        )?;
        self.wo
            .quantize_progressive_with_secondary(bits, residual_bits, secondary_bits, group_size)
    }
}

fn quick_gelu(x: &Array) -> Result<Array> {
    // Dtype-preserving (sc-2787): the fork's `1.702 * input_array` is a weak python scalar, so a bf16
    // input stays bf16. A strong f32 `scalar(1.702)` would promote bf16→f32 and break CLIP bf16 parity.
    let c = scalar(1.702).as_dtype(x.dtype())?;
    Ok(multiply(x, &sigmoid(&multiply(x, &c)?)?)?)
}

/// T5's `T5LayerNorm` — RMS-normalize over the last axis with NO mean subtraction.
///
/// This is deliberately the fork's hand-rolled primitive sequence (`weight * x *
/// rsqrt(mean(x^2) + eps)`), NOT `mlx_rs::fast::rms_norm`. The fused kernel differs from the fork's
/// primitives by ~1e-7 per call; T5-xxl applies it 49×, so on the wheel that grows to ~3e-3 in
/// `prompt_embeds` (this exact form is BIT-EXACT to the fork on the wheel — verified sc-2345 review,
/// 2026-06-02). On the pinned NAX build it removes the fast-vs-manual share of the T5 drift
/// (dev@512²: 2.66e-3 → 1.87e-3 mean_rel); the rest is irreducible NAX-vs-wheel f32 accumulation over
/// the 24 layers (block-0 bit-exact, grows monotonically with depth — not a code bug, the deferred
/// cross-build delta). CLIP is unaffected because it uses `LayerNorm`, whose fused kernel DOES match
/// the fork. `power(x, 2)` (not `square`) matches the fork's `mx.power(_, 2)` — they differ by 1 ULP.
fn t5_rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let var = power(x, Array::from_slice(&[2.0_f32], &[1]))?.mean_axis(-1, true)?;
    let normed = multiply(x, &add(&var, scalar(eps))?.rsqrt()?)?;
    Ok(multiply(weight, &normed)?)
}

fn shape_t5(x: &Array) -> Result<Array> {
    Ok(x.reshape(&[1, -1, 64, 64])?.transpose_axes(&[0, 2, 1, 3])?)
}

fn unshape_t5(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, -1, 4096])?)
}

fn relative_position_buckets(seq_len: i32) -> Vec<i32> {
    let mut buckets = Vec::with_capacity((seq_len * seq_len) as usize);
    for context in 0..seq_len {
        for memory in 0..seq_len {
            let relative = memory - context;
            buckets.push(relative_position_bucket(relative));
        }
    }
    buckets
}

fn relative_position_bucket(relative_position: i32) -> i32 {
    let num_buckets = 32;
    let max_distance = 128.0_f32;
    let mut bucket = 0;
    let mut n = relative_position;
    let half = num_buckets / 2;
    if n > 0 {
        bucket += half;
    }
    n = n.abs();
    let max_exact = half / 2;
    let val = if n < max_exact {
        n
    } else {
        let n_float = n as f32;
        let log_ratio = (n_float / max_exact as f32).ln() / (max_distance / max_exact as f32).ln();
        let large = max_exact + (log_ratio * (half - max_exact) as f32).floor() as i32;
        large.min(half - 1)
    };
    bucket + val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_relative_position_buckets_match_known_edges() {
        assert_eq!(relative_position_bucket(0), 0);
        assert_eq!(relative_position_bucket(1), 17);
        assert_eq!(relative_position_bucket(-1), 1);
        assert_eq!(relative_position_bucket(128), 31);
        assert_eq!(relative_position_bucket(-128), 15);
    }

    #[test]
    fn t5_group_size_matches_mlx_affine_quantization_contract() {
        for group_size in [32, 64] {
            assert!(validate_t5_group_size(group_size).is_ok());
        }
        for group_size in [0, 16, 128, 256] {
            assert!(validate_t5_group_size(group_size).is_err());
        }
    }
}
