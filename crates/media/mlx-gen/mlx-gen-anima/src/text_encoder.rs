//! Anima's source text encoder — **Qwen3-0.6B base** (`Qwen3Model`), reusing z-image's Qwen3 decoder
//! block (`EncoderLayer`) + the shared HF RoPE. Unlike z-image (which returns the second-to-last
//! layer un-normed for its DiT `cap_feats`), Anima consumes the model's **`last_hidden_state`** — the
//! last layer output AFTER the final `norm` — matching `Qwen3Model(...).last_hidden_state`. The hidden
//! states are then mask-multiplied and fed to the `AnimaTextConditioner` as `source_hidden_states`.

use mlx_rs::fast::rms_norm;
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;
use mlx_gen_z_image::text_encoder::EncoderLayer;

use crate::config::Qwen3Config;

/// iOS execution-quantum labels (see [`mlx_gen::ios_quantum`]).
pub(crate) const TE_EMBED_QUANTUM: &str = "Anima Qwen3 embed";
pub(crate) const TE_LAYER_QUANTUM: &str = "Anima Qwen3 layer";
pub(crate) const TE_NORM_QUANTUM: &str = "Anima Qwen3 final norm";

/// One decoder layer's loaded geometry for its iOS quantum floor, read off the layer's own tensors
/// (z-image's `EncoderLayer` does not expose them): `(hidden, intermediate, heads, head_dim)`, and
/// the q and k/v projection widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LayerGeometry {
    hidden: u64,
    intermediate: u64,
    heads: u64,
    head_dim: u64,
    q_width: u64,
    kv_width: u64,
    /// Logical elements of the layer's DENSE projection weights (`0` when packed) and their storage
    /// width in bytes.
    dense_weight_elements: u64,
    weight_width: u64,
}

impl LayerGeometry {
    /// Row counts are the logical out-features for a dense AND a packed (`.scales`) projection.
    fn read(w: &Weights, prefix: &str) -> Result<Self> {
        let rows = |name: &str| -> Result<u64> {
            Ok(w.require(&join(prefix, name))?.shape()[0].max(0) as u64)
        };
        let head_dim = rows("self_attn.q_norm.weight")?.max(1);
        let q_width = rows("self_attn.q_proj.weight")?;
        let (hidden, intermediate) = (
            rows("input_layernorm.weight")?,
            rows("mlp.gate_proj.weight")?,
        );
        let kv_width = rows("self_attn.k_proj.weight")?;
        let packed = w.get(&join(prefix, "self_attn.q_proj.scales")).is_some();
        let q_weight = w.require(&join(prefix, "self_attn.q_proj.weight"))?;
        Ok(Self {
            hidden,
            intermediate,
            heads: q_width / head_dim,
            head_dim,
            q_width,
            kv_width,
            // q, k, v, o and gate, up, down — the logical [out, in] of each.
            dense_weight_elements: if packed {
                0
            } else {
                2 * q_width * hidden + 2 * kv_width * hidden + 3 * intermediate * hidden
            },
            weight_width: q_weight.item_size() as u64,
        })
    }

    /// The layer's counted geometry. The attention is one unchunked
    /// `mlx_rs::fast::scaled_dot_product_attention` with the additive causal/padding mask over the
    /// GQA-repeated k/v; its score treatment on `backend` is [`crate::transformer::sdpa_scores`] of
    /// the loaded head dim (128 in Qwen3-0.6B, which the fused Metal kernel serves), the width and
    /// the token count. The width is f32: the f32 RoPE tables
    /// promote the rotated q/k and everything after them, so every layer after the first runs f32
    /// (see [`layer_element_bytes`]).
    ///
    /// One layer's buffers are held together (MLX holds an op's buffers until its stream catches
    /// up, and nothing in a layer synchronizes it), so the attention and MLP buffers form one phase.
    /// Counted per projection WIDTH and expressed in `[B, S, hidden]` units, rounded up (Qwen3's q,
    /// heads x head_dim = 2048, is twice its hidden 1024; its MLP is 3x); an RMSNorm is counted
    /// with the temporaries of MLX's composed non-Metal fallback, so the CPU proof bounds the
    /// device's fused kernel from above:
    ///
    /// * hidden width (10): the input and post-attention RMSNorms (2 x 3), `o_proj`, the attention
    ///   residual sum, `down_proj` and the MLP residual sum;
    /// * q width (8): q, its per-head RMSNorm, its RoPE rotate-half and product (2), the
    ///   GQA-repeated k and v, the attention output and its merged copy;
    /// * k/v width (5): k, v, the k RMSNorm and the k rotation (2);
    /// * intermediate width (3): gate, up and the gated SiLU product.
    ///
    /// Weight copies: every dense bf16 projection fed the f32 stream is promoted to an f32 copy for
    /// its matmul; all of the layer's are priced (measured, 31 of the 40 f32-copy units at the
    /// proof's shape are live at once). A packed projection is consumed in place: no copy.
    ///
    /// Proven, with f32 and with bf16 weights, in
    /// `quantum_tests::text_encoder_formulas_track_the_measured_peak`.
    fn geometry(
        self,
        batch: i32,
        tokens: i32,
        element_bytes: u64,
        backend: crate::transformer::Backend,
    ) -> mlx_gen::ios_quantum::AttentionBlockGeometry {
        let n = tokens.max(0) as u64;
        let units = |width: u64| width.div_ceil(self.hidden.max(1));
        let (q, kv, inter) = (
            units(self.q_width),
            units(self.kv_width),
            units(self.intermediate),
        );
        mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: batch.max(0) as u64,
            tokens: tokens.max(0) as u64,
            dim: self.hidden,
            heads: self.heads,
            context_tokens: 0,
            ff_hidden: self.intermediate,
            query_rows: tokens.max(0) as u64,
            element_bytes,
            weight_bytes: if self.weight_width < element_bytes {
                self.dense_weight_elements.saturating_mul(element_bytes)
            } else {
                0
            },
            attn_hidden_live: 10 + 8 * q + 5 * kv + 3 * inter,
            context_live: 0,
            scores: crate::transformer::sdpa_scores(
                backend,
                self.head_dim,
                element_bytes,
                &[(n, n, n)],
            ),
            mlp_hidden_live: 0,
            ff_live: 0,
        }
    }

    fn scratch(
        self,
        batch: i32,
        tokens: i32,
        element_bytes: u64,
        backend: crate::transformer::Backend,
    ) -> u64 {
        mlx_gen::ios_quantum::attention_block_bytes(self.geometry(
            batch,
            tokens,
            element_bytes,
            backend,
        ))
    }
}

/// The activation width a Qwen3 layer quantum runs at: the f32 RoPE tables promote the rotated q/k
/// and everything after them to f32, whatever the incoming hidden state's width.
pub(crate) fn layer_element_bytes(hidden: &Array, rope_table: &Array) -> u64 {
    hidden.item_size().max(rope_table.item_size()) as u64
}

/// The Qwen3-0.6B text tower (token embed → 28 pre-norm decoder layers → final RMSNorm).
pub struct AnimaQwen3 {
    embed: TokenEmbedding,
    layers: Vec<EncoderLayer>,
    /// Per-layer iOS quantum geometry, parallel to `layers`.
    layer_geometry: Vec<LayerGeometry>,
    norm: Array,
    rope: TextRope,
    eps: f32,
    /// The tower's compute dtype. `Bfloat16` for the shipped path (Anima's on-disk Qwen3 weights are
    /// bf16, so this is a no-op cast that matches the reference `text_encoder.dtype`). `Float32` builds
    /// the fp32-TE **reference** variant that isolates the bf16-conditioning parity offset (sc-10577).
    compute_dtype: Dtype,
}

impl AnimaQwen3 {
    /// `prefix` is the checkpoint root of the Qwen3 model (`"model"` for `model.embed_tokens.*`). The
    /// tower runs in **bf16** — byte-identical to the shipped path (Anima's Qwen3 weights are bf16 on
    /// disk). Use [`from_weights_dtype`](Self::from_weights_dtype) for the sc-10577 fp32 reference.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        Self::from_weights_dtype(w, prefix, cfg, Dtype::Bfloat16)
    }

    /// Build the tower recording an explicit **compute dtype** (sc-10577). `Bfloat16` reproduces the
    /// shipped bf16 path exactly. `Float32` runs the whole tower in fp32 to build the fp32-TE reference
    /// variant that isolates the bf16-conditioning parity offset — the caller MUST supply fp32-upcast
    /// weights (MLX `matmul` requires the activation and weight dtypes to match), which
    /// [`crate::loader::load_conditioning_at_dtype`] does via `Weights::cast_all`.
    pub fn from_weights_dtype(
        w: &Weights,
        prefix: &str,
        cfg: &Qwen3Config,
        compute_dtype: Dtype,
    ) -> Result<Self> {
        let embed = mlx_gen::quant::embedding(
            w,
            &join(prefix, "embed_tokens"),
            mlx_gen::quant::DEFAULT_GROUP_SIZE,
        )?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut layer_geometry = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let layer_prefix = join(prefix, &format!("layers.{i}"));
            layers.push(EncoderLayer::from_weights(
                w,
                &layer_prefix,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
            layer_geometry.push(LayerGeometry::read(w, &layer_prefix)?);
        }
        Ok(Self {
            embed,
            layers,
            layer_geometry,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            eps: cfg.rms_norm_eps,
            compute_dtype,
        })
    }

    /// Layer `i` and its counted iOS quantum scratch at `(batch, tokens, element_bytes)` (the CPU
    /// proofs run the layer's quantum body directly).
    #[cfg(test)]
    pub(crate) fn layer(
        &self,
        i: usize,
        batch: i32,
        tokens: i32,
        element_bytes: u64,
        backend: crate::transformer::Backend,
    ) -> (&EncoderLayer, u64) {
        (
            &self.layers[i],
            self.layer_geometry[i].scratch(batch, tokens, element_bytes, backend),
        )
    }

    /// The final norm's quantum body.
    #[cfg(test)]
    pub(crate) fn final_norm(&self, h: &Array) -> Result<Array> {
        Ok(rms_norm(h, &self.norm, self.eps)?)
    }

    /// The tower's compute dtype (`Bfloat16` shipped; `Float32` for the sc-10577 fp32-TE reference).
    pub fn compute_dtype(&self) -> Dtype {
        self.compute_dtype
    }

    /// The embed quantum's body: gather the token rows and cast them to the compute dtype, build
    /// the RoPE tables and the additive causal/padding mask. Returns `(h, cos, sin, mask)`.
    pub(crate) fn embed(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array, Array, Array)> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        // `compute_dtype` tower (bf16 shipped — matches the reference `text_encoder.dtype`; fp32 for the
        // sc-10577 reference variant): embed gathers weight-dtype rows, cast to the compute dtype.
        let h = self
            .embed
            .forward(input_ids)?
            .as_dtype(self.compute_dtype)?;
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?.as_dtype(self.compute_dtype)?;
        Ok((h, cos, sin, mask))
    }

    /// The embed quantum's counted body ([`Self::embed`]): the gathered rows (priced at f32, the
    /// width a packed embedding dequantizes to), `TokenEmbedding`'s f32 cast and, when the compute
    /// dtype is narrower, the cast to it (`[B, S, hidden]` each), the shared
    /// [`mlx_gen::ios_quantum::rope_table_bytes`] allowance, and the `[B, 1, S, S]` mask built in f32
    /// and its cast. Proven in `quantum_tests::text_encoder_formulas_track_the_measured_peak`.
    pub(crate) fn embed_body_bytes(&self, batch: i32, tokens: i32) -> u64 {
        let (b, s) = (batch.max(0) as u64, tokens.max(0) as u64);
        let eb = crate::transformer::dtype_bytes(self.compute_dtype);
        let hidden = self.norm.size() as u64;
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        // The compute-dtype cast allocates only when it narrows the f32 rows.
        let cast = if eb < 4 { tb(eb, &[b, s, hidden]) } else { 0 };
        (2 * tb(4, &[b, s, hidden]))
            .saturating_add(cast)
            .saturating_add(mlx_gen::ios_quantum::rope_table_bytes(
                s,
                self.rope_head_dim(),
            ))
            .saturating_add(tb(4, &[b, s, s]))
            .saturating_add(tb(eb, &[b, s, s]))
    }

    /// The final-norm quantum over the last layer's f32 output: the RMSNorm's square, normalized
    /// product and weighted product (3 `[B, S, hidden]`; MLX's composed fallback — the fused Metal
    /// kernel allocates only the last).
    pub(crate) fn norm_bytes(&self, batch: i32, tokens: i32, element_bytes: u64) -> u64 {
        3 * mlx_gen::ios_quantum::tensor_bytes(
            element_bytes,
            &[
                batch.max(0) as u64,
                tokens.max(0) as u64,
                self.norm.size() as u64,
            ],
        )
    }

    fn rope_head_dim(&self) -> u64 {
        self.layer_geometry.first().map_or(0, |g| g.head_dim)
    }

    /// `input_ids` / `attention_mask`: `[B, S]` int32. Returns the **last_hidden_state** `[B, S, hidden]`
    /// in [`compute_dtype`](Self::compute_dtype) — the last decoder layer AFTER the final norm (a causal
    /// LM with padding masking).
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        mlx_gen::ios_quantum::before(TE_EMBED_QUANTUM, self.embed_body_bytes(b, s))?;
        let (mut h, cos, sin, mask) = self.embed(input_ids, attention_mask)?;
        mlx_gen::ios_quantum::finish(TE_EMBED_QUANTUM, &[&h, &cos, &sin, &mask])?;
        for (layer, geometry) in self.layers.iter().zip(&self.layer_geometry) {
            let eb = layer_element_bytes(&h, &cos);
            h = mlx_gen::ios_quantum::run(
                TE_LAYER_QUANTUM,
                geometry.scratch(b, s, eb, crate::transformer::Backend::current()),
                || layer.forward(&h, &cos, &sin, &mask),
            )?;
        }
        // last_hidden_state = final RMSNorm applied to the last layer (unlike z-image's [-2] un-normed).
        let eb = h.item_size() as u64;
        mlx_gen::ios_quantum::run(TE_NORM_QUANTUM, self.norm_bytes(b, s, eb), || {
            Ok(rms_norm(&h, &self.norm, self.eps)?)
        })
    }
}
