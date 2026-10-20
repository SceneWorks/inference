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

/// The activation width a Qwen3 layer quantum runs at: the f32 RoPE tables promote the rotated q/k
/// and everything after them to f32, whatever the incoming hidden state's width.
pub(crate) fn layer_element_bytes(hidden: &Array, rope_table: &Array) -> u64 {
    hidden.item_size().max(rope_table.item_size()) as u64
}

/// The Qwen3-0.6B text tower (token embed → 28 pre-norm decoder layers → final RMSNorm).
pub struct AnimaQwen3 {
    embed: TokenEmbedding,
    layers: Vec<EncoderLayer>,
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
        }
        Ok(Self {
            embed,
            layers,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            eps: cfg.rms_norm_eps,
            compute_dtype,
        })
    }

    /// Layer `i` and its counted iOS quantum scratch at `(batch, tokens, element_bytes)` (the CPU
    /// proofs run the layer's quantum body directly; the iOS pricing prices each layer's quantum).
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
            self.layers[i].quantum_scratch(batch, tokens, element_bytes, backend),
        )
    }

    /// The final norm's quantum body.
    #[cfg(test)]
    pub(crate) fn final_norm(&self, h: &Array) -> Result<Array> {
        Ok(rms_norm(h, &self.norm, self.eps)?)
    }

    /// The number of decoder layers.
    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
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
        self.layers
            .first()
            .map_or(0, |l| l.head_dim().max(0) as u64)
    }

    /// `input_ids` / `attention_mask`: `[B, S]` int32. Returns the **last_hidden_state** `[B, S, hidden]`
    /// in [`compute_dtype`](Self::compute_dtype) — the last decoder layer AFTER the final norm (a causal
    /// LM with padding masking).
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        mlx_gen::ios_quantum::before(TE_EMBED_QUANTUM, self.embed_body_bytes(b, s))?;
        let (mut h, cos, sin, mask) = self.embed(input_ids, attention_mask)?;
        mlx_gen::ios_quantum::finish(TE_EMBED_QUANTUM, &[&h, &cos, &sin, &mask])?;
        for layer in &self.layers {
            let eb = layer_element_bytes(&h, &cos);
            h = mlx_gen::ios_quantum::run(
                TE_LAYER_QUANTUM,
                layer.quantum_scratch(b, s, eb, crate::transformer::Backend::current()),
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
