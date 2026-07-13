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
            layers.push(EncoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
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
        // `compute_dtype` tower (bf16 shipped — matches the reference `text_encoder.dtype`; fp32 for the
        // sc-10577 reference variant): embed gathers weight-dtype rows, cast to the compute dtype.
        let mut h = self
            .embed
            .forward(input_ids)?
            .as_dtype(self.compute_dtype)?;
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?.as_dtype(self.compute_dtype)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        // last_hidden_state = final RMSNorm applied to the last layer (unlike z-image's [-2] un-normed).
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }
}
