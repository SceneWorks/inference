//! Text-encoder decoder block (pre-norm residual): `h += attn(input_ln(h))`, then
//! `h += mlp(post_ln(h))`. Port of the fork's `EncoderLayer`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, TextAttention, TextMlp};

pub struct EncoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: TextAttention,
    mlp: TextMlp,
    eps: f32,
}

impl EncoderLayer {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w.require(&join(prefix, "input_layernorm.weight"))?.clone(),
            post_ln: w
                .require(&join(prefix, "post_attention_layernorm.weight"))?
                .clone(),
            attn: TextAttention::from_weights(
                w,
                &join(prefix, "self_attn"),
                num_heads,
                num_kv_heads,
                head_dim,
            )?,
            mlp: TextMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps,
        })
    }

    /// Quantize the block's Linears (attention QKV/out + MLP) to Q4/Q8. The two layer-norm scales
    /// stay dense (not Linears).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }

    /// The layer's counted iOS quantum geometry over `[batch, tokens, hidden]` at `element_bytes`
    /// (sc-26265 for Anima's Qwen3-0.6B tower, shared with Z-Image's Qwen3-4B one by sc-26433), read
    /// off the LOADED projections. The attention is one unchunked
    /// `mlx_rs::fast::scaled_dot_product_attention` with the additive causal/padding mask over the
    /// GQA-repeated k/v; its score treatment on `backend` is
    /// [`sdpa_scores`](mlx_gen::ios_quantum::sdpa_scores) of the head dim (128 in Qwen3, which the
    /// fused Metal kernel serves), the width and the token count. The f32 RoPE tables promote the
    /// rotated q/k and everything after them, so a bf16 stream runs f32 after the first layer.
    ///
    /// One layer's buffers are held together (MLX holds an op's buffers until its stream catches
    /// up, and nothing in a layer synchronizes it), so the attention and MLP buffers form one phase.
    /// Counted per projection WIDTH and expressed in `[B, S, hidden]` units, rounded up (Qwen3's q is
    /// wider than its hidden; its MLP is ~3-4x); an RMSNorm is counted with the temporaries of MLX's
    /// composed non-Metal fallback, so the CPU proof bounds the device's fused kernel from above:
    ///
    /// * hidden width (10): the input and post-attention RMSNorms (2 x 3), `o_proj`, the attention
    ///   residual sum, `down_proj` and the MLP residual sum;
    /// * q width (8): q, its per-head RMSNorm, its RoPE rotate-half and product (2), the
    ///   GQA-repeated k and v, the attention output and its merged copy;
    /// * k/v width (5): k, v, the k RMSNorm and the k rotation (2);
    /// * intermediate width (3): gate, up and the gated SiLU product.
    ///
    /// Weight copies: every dense projection narrower than the stream is promoted to a stream-width
    /// copy for its matmul; all of the layer's are priced. A packed projection is consumed in place.
    pub fn quantum_geometry(
        &self,
        batch: i32,
        tokens: i32,
        element_bytes: u64,
        backend: mlx_gen::ios_quantum::Backend,
    ) -> mlx_gen::ios_quantum::AttentionBlockGeometry {
        let rows = |lin: &mlx_gen::adapters::AdaptableLinear| {
            lin.base_shape().first().map_or(0, |&d| d.max(0) as u64)
        };
        let [q, k, _, _] = self.attn.projections();
        let [gate, ..] = self.mlp.projections();
        let hidden = self.input_ln.size() as u64;
        let head_dim = (self.attn.head_dim().max(1)) as u64;
        let (q_width, kv_width, intermediate) = (rows(q), rows(k), rows(gate));
        let units = |width: u64| width.div_ceil(hidden.max(1));
        let dense_elements: u64 = self
            .attn
            .projections()
            .into_iter()
            .chain(self.mlp.projections())
            .filter(|lin| !lin.is_quantized())
            .map(|lin| {
                lin.base_shape()
                    .iter()
                    .fold(1u64, |a, &d| a.saturating_mul(d.max(0) as u64))
            })
            .sum();
        let weight_width = q
            .weight_dtype()
            .map_or(element_bytes, mlx_gen::ios_quantum::dtype_bytes);
        let n = tokens.max(0) as u64;
        mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: batch.max(0) as u64,
            tokens: n,
            dim: hidden,
            heads: q_width / head_dim,
            context_tokens: 0,
            ff_hidden: intermediate,
            query_rows: n,
            element_bytes,
            weight_bytes: if weight_width < element_bytes {
                dense_elements.saturating_mul(element_bytes)
            } else {
                0
            },
            attn_hidden_live: 10
                + 8 * units(q_width)
                + 5 * units(kv_width)
                + 3 * units(intermediate),
            context_live: 0,
            scores: mlx_gen::ios_quantum::sdpa_scores(
                backend,
                head_dim,
                element_bytes,
                &[(n, n, n)],
            ),
            mlp_hidden_live: 0,
            ff_live: 0,
        }
    }

    /// [`Self::quantum_geometry`]'s price.
    pub fn quantum_scratch(
        &self,
        batch: i32,
        tokens: i32,
        element_bytes: u64,
        backend: mlx_gen::ios_quantum::Backend,
    ) -> u64 {
        mlx_gen::ios_quantum::attention_block_bytes(self.quantum_geometry(
            batch,
            tokens,
            element_bytes,
            backend,
        ))
    }

    /// The packed bytes [`Self::quantize`] builds from the layer's dense projections (0 when packed).
    pub(crate) fn quantize_bytes(&self, bits: i32) -> u64 {
        self.attn
            .projections()
            .into_iter()
            .chain(self.mlp.projections())
            .map(|lin| crate::quantum::quantize_bytes(lin, bits))
            .sum()
    }

    /// The per-head width the layer's RoPE tables are built at.
    pub fn head_dim(&self) -> i32 {
        self.attn.head_dim()
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}
