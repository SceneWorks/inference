//! FLUX.1 prompt text path: CLIP pooled prompt embedding + T5 sequence prompt embedding.
//! Ports the fork's `flux_text_encoder` modules directly.

use crate::quant::GROUP_SIZE;
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::{host_i32, scalar};
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::{join, Weights};
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, dequantize, matmul, multiply, power, quantize, sigmoid, softmax_axis};
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
    },
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
            } => {
                let pw = wq.take_axis(ids, 0)?;
                let sc = scales.take_axis(ids, 0)?;
                let bi = biases.take_axis(ids, 0)?;
                dequantize(&pw, &sc, &bi, *group_size, *bits)?
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
            };
        }
        Ok(())
    }
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

pub struct T5TextEncoder {
    shared: TokenEmbedding,
    blocks: Vec<T5Block>,
    final_ln_w: Array,
    /// Ladder rung 4, `TransformerComponent::TextEncoder` scope (SC-15520). `None` on every ordinary
    /// load, which keeps the resident path byte-for-byte unchanged.
    block_stream: Option<T5BlockStream>,
}

/// T5-XXL's encoder depth. Fixed by the architecture, and the same number
/// [`T5TextEncoder::from_weights_with_group_size`] loads.
pub const T5_BLOCKS: usize = 24;

/// A reopenable description of a T5 encoder's 24 blocks, for rung 4's **text-encoder** scope.
///
/// The encoder is the peak-bearing phase for a T5-only-conditioned family whose auxiliary components
/// ship dense: measured on Chroma1-Base q4 at 1024² (SC-15520) the conditioning phase peaks at
/// **10.15 GiB** against a 7.14 GiB denoise and a 4.37 GiB bounded decode, so a DiT-scoped window is
/// inert on the request peak and this is the scope that is not. Kolors reached the same conclusion
/// first, for the same reason and a different encoder (SC-15521).
#[derive(Clone)]
pub struct T5BlockStream {
    /// The reopenable `text_encoder/` component — never the caller's snapshot root.
    source: mlx_gen::WeightsSource,
    prefix: String,
    group_size: i32,
    n_blocks: usize,
}

impl T5BlockStream {
    pub fn new(source: mlx_gen::WeightsSource, prefix: &str, group_size: i32) -> Result<Self> {
        validate_t5_group_size(group_size)?;
        Ok(Self {
            source,
            prefix: prefix.to_owned(),
            group_size,
            n_blocks: T5_BLOCKS,
        })
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Open a fresh lazy view of the encoder component. Called once per window.
    fn open(&self) -> Result<Weights> {
        match &self.source {
            mlx_gen::WeightsSource::Dir(dir) => Weights::from_dir(dir),
            mlx_gen::WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn materialize(&self, view: &mut Weights, index: usize) -> Result<T5Block> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "t5 block stream: block {index} is outside the {}-block encoder",
                self.n_blocks
            )));
        }
        let prefix = join(&self.prefix, &format!("encoder.block.{index}"));
        let block = T5Block::from_weights(view, &prefix, self.group_size)?;
        // LOAD-BEARING: `Array` is refcounted and the constructor cloned out of the view, so
        // draining exactly the accessed keys is what makes the window's drop a real release.
        view.remove_accessed();
        Ok(block)
    }
}

/// T5's relative-attention-bias table has logical width 64, so group 128 — valid for MLX affine
/// quantization in general — cannot pack it and would leave a hole in the packed surface. Only 32
/// and 64 are admissible here.
pub(crate) fn validate_t5_group_size(group_size: i32) -> Result<()> {
    if matches!(group_size, 32 | 64) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "T5 quantization group size must be 32 or 64, got {group_size}"
        )))
    }
}

impl T5TextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights_with_group_size(w, prefix, GROUP_SIZE)
    }

    /// Load T5 at the affine group size its packed component declares. FLUX keeps the codebase
    /// default; Chroma publishes group 32, which measurably halves packed-T5 render error versus 64
    /// at the same Q8 width, so its artifacts are self-describing rather than assumed.
    pub fn from_weights_with_group_size(
        w: &Weights,
        prefix: &str,
        group_size: i32,
    ) -> Result<Self> {
        validate_t5_group_size(group_size)?;
        let p = |suffix: &str| join(prefix, suffix);
        let mut blocks = Vec::with_capacity(T5_BLOCKS);
        for i in 0..T5_BLOCKS {
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
            block_stream: None,
        })
    }

    /// Arm snapshot-backed reconstruction of the 24 encoder blocks (rung 4, text-encoder scope), then
    /// evict the resident stack. The token embedding and the final layer norm remain resident: they
    /// are not block weights and run once per prompt.
    ///
    /// Must be called **after** any load-time quantization, so a streamed block is rebuilt from the
    /// same on-disk state the resident block ended in. A dense source the caller intends to
    /// `quantize` in place is therefore not streamable — the window would re-pack every
    /// materialization, which is a host-format conversion rather than the device-format transfer the
    /// contract declares.
    pub fn with_block_stream(mut self, stream: T5BlockStream) -> Result<Self> {
        if stream.n_blocks() != self.blocks.len() {
            return Err(Error::Msg(format!(
                "t5 block stream: declared {} blocks against a {}-block encoder",
                stream.n_blocks(),
                self.blocks.len()
            )));
        }
        self.blocks.clear();
        self.block_stream = Some(stream);
        Ok(self)
    }

    /// `true` once [`Self::with_block_stream`] has armed the encoder.
    pub fn is_streamable(&self) -> bool {
        self.block_stream.is_some()
    }

    /// Resident block count — `0` once a stream is armed.
    pub fn resident_block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.quantize_with_group_size(bits, GROUP_SIZE)
    }

    /// Pack the complete quantizable T5 surface at an explicit affine group size. Byte-identical to
    /// what the offline converter writes at the same `(bits, group_size)`.
    pub fn quantize_with_group_size(&mut self, bits: i32, group_size: i32) -> Result<()> {
        validate_t5_group_size(group_size)?;
        self.shared.quantize_with_group_size(bits, group_size)?;
        for block in &mut self.blocks {
            block.quantize(bits, group_size)?;
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
        self.forward_masked_windowed(tokens, mask, None, &mlx_gen::CancelFlag::new())
    }

    /// [`Self::forward_masked`] with an optional rung-4 block window (SC-15520).
    ///
    /// `window = None` is byte-for-byte the historical resident path. `Some(size)` requires an armed
    /// [`T5BlockStream`] and runs the 24 blocks through the shared
    /// [`mlx_gen::block_residency::run_windowed`] driver, holding `size` blocks materialized at a
    /// time.
    pub fn forward_masked_windowed(
        &self,
        tokens: &Array,
        mask: Option<&Array>,
        window: Option<usize>,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<Array> {
        let hidden = self.shared.forward(tokens)?;
        let seq_len = hidden.shape()[1];
        let hidden = match window {
            None => {
                if self.block_stream.is_some() && self.blocks.is_empty() {
                    return Err(Error::Unsupported(
                        "t5: a deferred encoder requires an explicit block window".to_owned(),
                    ));
                }
                let mut hidden = hidden;
                // The relative-position bias depends only on seq_len and is identical across all
                // blocks (only block 0 carries the table; every other block clones it), so compute it
                // once here and share it instead of rebuilding the O(L²) gather inside each of the 24
                // blocks (F-099).
                if let Some(block0) = self.blocks.first() {
                    let bias = block0.attn.position_bias(seq_len)?;
                    for block in &self.blocks {
                        hidden = block.forward(&hidden, mask, &bias)?;
                    }
                }
                hidden
            }
            Some(size) => {
                let stream = self.block_stream.as_ref().ok_or_else(|| {
                    Error::Unsupported(
                        "t5: a bounded encoder window needs a snapshot-backed block stream"
                            .to_owned(),
                    )
                })?;
                if !self.blocks.is_empty() {
                    return Err(Error::Msg(
                        "t5: a windowed encode ran against a stack that still holds resident blocks \
                         — the bound would not hold"
                            .to_owned(),
                    ));
                }
                // The position bias lives on block 0 only. Materialize that block alone, force the
                // bias, and drop it: leaving the bias as a lazy node would keep block 0's weights
                // referenced for the whole encode and the window would bound nothing.
                let bias = {
                    let mut view = stream.open()?;
                    let block0 = stream.materialize(&mut view, 0)?;
                    let bias = block0.attn.position_bias(seq_len)?;
                    mlx_rs::transforms::eval([&bias])?;
                    drop(block0);
                    drop(view);
                    mlx_rs::memory::clear_cache();
                    bias
                };
                let plan = mlx_gen::block_residency::BlockPlan::new(stream.n_blocks(), size)?;
                mlx_gen::block_residency::run_windowed(
                    &plan,
                    cancel,
                    hidden,
                    || stream.open(),
                    |mut hidden, view, range| {
                        for i in range {
                            let block = stream.materialize(view, i)?;
                            hidden = block.forward(&hidden, mask, &bias).map_err(|error| {
                                Error::Msg(format!("t5 block stream: block {i} forward: {error}"))
                            })?;
                        }
                        Ok(hidden)
                    },
                    // LOAD-BEARING: MLX is lazy, so the carried activation still references the
                    // window's weights until it is forced.
                    |hidden: &Array| mlx_rs::transforms::eval([hidden]).map_err(Into::into),
                )?
            }
        };
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
        self.attn.quantize(bits, group_size)?;
        self.ff.quantize(bits, group_size)?;
        Ok(())
    }
}

struct T5Attention {
    ln_w: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    rel_bias: TokenEmbedding,
}

impl T5Attention {
    fn from_weights(w: &Weights, prefix: &str, group_size: i32) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `.scales` is present, else dense.
        let linear = |name: &str| {
            mlx_gen::quant::lin(
                w,
                &join(prefix, &format!("SelfAttention.{name}")),
                false,
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

    fn quantize(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.q.quantize(bits, Some(group_size))?;
        self.k.quantize(bits, Some(group_size))?;
        self.v.quantize(bits, Some(group_size))?;
        self.o.quantize(bits, Some(group_size))?;
        self.rel_bias.quantize_with_group_size(bits, group_size)?;
        Ok(())
    }
}

struct T5FeedForward {
    ln_w: Array,
    wi0: AdaptableLinear,
    wi1: AdaptableLinear,
    wo: AdaptableLinear,
}

impl T5FeedForward {
    fn from_weights(w: &Weights, prefix: &str, group_size: i32) -> Result<Self> {
        // Packed-detect (sc-8669): loads Q4/Q8 packed when `.scales` is present, else dense.
        let linear = |name: &str| {
            mlx_gen::quant::lin(
                w,
                &join(prefix, &format!("DenseReluDense.{name}")),
                false,
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

    fn quantize(&mut self, bits: i32, group_size: i32) -> Result<()> {
        self.wi0.quantize(bits, Some(group_size))?;
        self.wi1.quantize(bits, Some(group_size))?;
        self.wo.quantize(bits, Some(group_size))?;
        Ok(())
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
}
