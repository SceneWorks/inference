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
/// (z-image's `EncoderLayer` does not expose them): `(hidden, intermediate, heads, head_dim)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LayerGeometry {
    hidden: u64,
    intermediate: u64,
    heads: u64,
    head_dim: u64,
}

impl LayerGeometry {
    /// Row counts are the logical out-features for a dense AND a packed (`.scales`) projection.
    fn read(w: &Weights, prefix: &str) -> Result<Self> {
        let rows = |name: &str| -> Result<u64> {
            Ok(w.require(&join(prefix, name))?.shape()[0].max(0) as u64)
        };
        let head_dim = rows("self_attn.q_norm.weight")?.max(1);
        Ok(Self {
            hidden: rows("input_layernorm.weight")?,
            intermediate: rows("mlp.gate_proj.weight")?,
            heads: rows("self_attn.q_proj.weight")? / head_dim,
            head_dim,
        })
    }

    fn scratch(self, batch: i32, tokens: i32) -> u64 {
        mlx_gen::ios_quantum::transformer_layer_bytes(
            batch.max(0) as u64,
            tokens.max(0) as u64,
            self.hidden,
            self.intermediate,
            self.heads,
            self.head_dim,
        )
    }
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

    /// The tower's compute dtype (`Bfloat16` shipped; `Float32` for the sc-10577 fp32-TE reference).
    pub fn compute_dtype(&self) -> Dtype {
        self.compute_dtype
    }

    /// `input_ids` / `attention_mask`: `[B, S]` int32. Returns the **last_hidden_state** `[B, S, hidden]`
    /// in [`compute_dtype`](Self::compute_dtype) — the last decoder layer AFTER the final norm (a causal
    /// LM with padding masking).
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        // Pre/post quanta are checked against the largest layer floor.
        let scratch = self
            .layer_geometry
            .iter()
            .map(|g| g.scratch(b, s))
            .max()
            .unwrap_or(0);
        mlx_gen::ios_quantum::before(TE_EMBED_QUANTUM, scratch)?;
        // `compute_dtype` tower (bf16 shipped — matches the reference `text_encoder.dtype`; fp32 for the
        // sc-10577 reference variant): embed gathers weight-dtype rows, cast to the compute dtype.
        let mut h = self
            .embed
            .forward(input_ids)?
            .as_dtype(self.compute_dtype)?;
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?.as_dtype(self.compute_dtype)?;
        mlx_gen::ios_quantum::finish(TE_EMBED_QUANTUM, &[&h, &cos, &sin, &mask])?;
        for (layer, geometry) in self.layers.iter().zip(&self.layer_geometry) {
            h = mlx_gen::ios_quantum::run(TE_LAYER_QUANTUM, geometry.scratch(b, s), || {
                layer.forward(&h, &cos, &sin, &mask)
            })?;
        }
        // last_hidden_state = final RMSNorm applied to the last layer (unlike z-image's [-2] un-normed).
        mlx_gen::ios_quantum::run(TE_NORM_QUANTUM, scratch, || {
            Ok(rms_norm(&h, &self.norm, self.eps)?)
        })
    }
}
