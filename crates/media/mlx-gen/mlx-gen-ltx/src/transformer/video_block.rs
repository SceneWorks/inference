//! The video-only transformer block, the adaLN-single timestep embedding and the RoPE memo.

use super::*;

/// One video transformer block (`BasicAVTransformerBlock`, video-only / gated 2.3 path).
#[derive(Clone)]
pub struct VideoBlock {
    pub(super) attn1: Attention,
    pub(super) attn2: Attention,
    pub(super) ff: FeedForward,
    pub(super) scale_shift_table: Array,        // (9, inner)
    pub(super) prompt_scale_shift_table: Array, // (2, inner)
    pub(super) eps: f32,
}

impl VideoBlock {
    /// Load a block (`prefix` e.g. `transformer_blocks.0`) at the given [`Precision`].
    pub fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let (h, dh, eps) = (
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.norm_eps as f32,
        );
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), h, dh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), h, dh, eps, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec, cfg.ff_bias)?,
            scale_shift_table: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            prompt_scale_shift_table: param(
                w,
                &format!("{prefix}.prompt_scale_shift_table"),
                prec,
            )?,
            eps: cfg.norm_eps as f32,
        })
    }

    /// Forward (gated, 9-row table): MSA(self, RoPE) → text cross-attn (prompt-modulated context) →
    /// FeedForward, each adaLN-modulated and gated.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        timesteps: &Array,
        prompt_timestep: Option<&Array>,
        context: &Array,
        mask: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // --- MSA (self-attention) ---
        let msa = ada_values(&self.scale_shift_table, timesteps, 0, 3)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = self
            .attn1
            .forward(&norm, None, None, Some((cos, sin)), None)?;
        let mut x = gated(x, &attn, &msa[2])?;

        // --- prompt-adaLN on the text context ---
        let v_context = {
            let (p_shift, p_scale) = match prompt_timestep {
                Some(pt) => {
                    let p = ada_values(&self.prompt_scale_shift_table, pt, 0, 2)?;
                    (p[0].clone(), p[1].clone())
                }
                None => (
                    self.prompt_scale_shift_table.index_axis(0, 0)?,
                    self.prompt_scale_shift_table.index_axis(1, 0)?,
                ),
            };
            modulate(context, &p_scale, &p_shift)?
        };

        // --- text cross-attention (adaLN rows 6..9) ---
        let ca = ada_values(&self.scale_shift_table, timesteps, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = self
            .attn2
            .forward(&norm_ca, Some(&v_context), mask, None, None)?;
        x = gated(&x, &cross, &ca[2])?;

        // --- FeedForward (adaLN rows 3..6) ---
        let mlp = ada_values(&self.scale_shift_table, timesteps, 3, 6)?;
        let norm_mlp = modulate(&rms_norm_noweight(&x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm_mlp)?;
        x = gated(&x, &ff, &mlp[2])?;

        Ok(x)
    }

    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            _ => None,
        }
    }

    pub(crate) fn set_lora_pass(&self, pass: usize) {
        self.attn1.set_lora_pass(pass);
        self.attn2.set_lora_pass(pass);
        self.ff.set_lora_pass(pass);
    }

    /// Toggle SDPA-segment checkpointing on both attentions (sc-4942). See [`Attention::ckpt_sdpa`].
    pub(super) fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn1.set_sdpa_checkpoint(on);
        self.attn2.set_sdpa_checkpoint(on);
    }

    /// Cast the block's frozen weights — both attentions, the FFN, and the adaLN scale-shift tables —
    /// to the training compute `dtype` in place (sc-4942 bf16 lever).
    pub(super) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.attn1.cast_weights(dtype)?;
        self.attn2.cast_weights(dtype)?;
        self.ff.cast_weights(dtype)?;
        self.scale_shift_table = to_dtype(&self.scale_shift_table, dtype)?;
        self.prompt_scale_shift_table = to_dtype(&self.prompt_scale_shift_table, dtype)?;
        Ok(())
    }
}

/// PixArt sinusoidal timestep embedding (`flip_sin_to_cos`, `downscale_freq_shift = 0`, max_period
/// 10000): `concat([cos(t·f), sin(t·f)])` with `f[i] = exp(−ln(10000)·i/half)`. `timesteps` is `(N,)`
/// f32; returns `(N, TIME_PROJ_DIM)` f32.
///
/// The log-spaced freqs are computed in **MLX float32** (`arange → ×(−ln θ) → ÷half → exp`), mirroring
/// the reference `get_timestep_embedding` op-for-op. A host-f64 table (the obvious shortcut) diverges
/// ~1e-7 per element from the MLX-f32 kernels (88/128 freqs differ; the projection differs up to
/// ~5e-5 after ×1000 + cos/sin) — invisible in bf16 but, in the F32Q8 path, this adaLN timestep
/// embedding modulates **every** block and the sub-ULP seed compounds across the 48-layer residual
/// stream into a percent-level velocity divergence that the distilled stage-1 sampler then amplifies.
/// (RoPE, by contrast, follows the reference's own numpy-f64 path — see [`crate::rope`].)
pub(super) fn timestep_embedding(timesteps: &Array) -> Result<Array> {
    let half = TIME_PROJ_DIM / 2; // 128
    let neg_ln = -(10000f64).ln() as f32;
    let exponent = divide(
        &multiply(&Array::arange::<_, f32>(None, half, None)?, scalar(neg_ln))?,
        scalar(half as f32),
    )?;
    let freq = exponent.exp()?.reshape(&[1, half])?; // (1, half)
    let emb = multiply(&timesteps.reshape(&[-1, 1])?, &freq)?; // (N, half)
    Ok(concatenate_axis(&[&emb.cos()?, &emb.sin()?], 1)?) // (N, dim), cos first
}

/// adaLN-single (`AdaLayerNormSingle`): `timestep → sinusoidal(256) → MLP(silu) → embedded`, then
/// `linear(silu(embedded)) → coeff·dim` scale-shift parameters.
pub(super) struct AdaLayerNormSingle {
    pub(super) ts_lin1: Linear, // 256 → dim
    pub(super) ts_lin2: Linear, // dim → dim
    pub(super) linear: Linear,  // dim → coeff·dim
}

impl AdaLayerNormSingle {
    pub(super) fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Ok(Self {
            ts_lin1: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear1"), prec)?,
            ts_lin2: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear2"), prec)?,
            linear: Linear::load(w, &format!("{prefix}.linear"), prec)?,
        })
    }

    /// `timestep` is the already-scaled `(N,)` f32. Returns `(scale_shift (N, coeff·dim), embedded
    /// (N, dim))` in `dt`.
    pub(super) fn forward(&self, timestep: &Array, dt: Dtype) -> Result<(Array, Array)> {
        let proj = timestep_embedding(timestep)?.as_dtype(dt)?;
        let h = mlx_gen::nn::silu(&self.ts_lin1.forward(&proj)?)?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&mlx_gen::nn::silu(&embedded)?)?;
        Ok((scale_shift, embedded))
    }

    /// LoRA targets in the converted MLX naming: `emb.timestep_embedder.linear1/linear2` and
    /// `linear`. The LTX-2.3 loader leaves PixArt-spelled `linear_1/linear_2` targets unresolved to
    /// preserve its reference compatibility behavior; the strict LTX-2.5 loader aliases the
    /// official published spellings to these base names before routing.
    pub(super) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["emb", "timestep_embedder", "linear1"] => Some(&mut self.ts_lin1),
            ["emb", "timestep_embedder", "linear2"] => Some(&mut self.ts_lin2),
            ["linear"] => Some(&mut self.linear),
            _ => None,
        }
    }

    pub(super) fn set_lora_pass(&self, pass: usize) {
        self.ts_lin1.set_lora_pass(pass);
        self.ts_lin2.set_lora_pass(pass);
        self.linear.set_lora_pass(pass);
    }

    /// Cast the three projection weights to the training compute `dtype` (sc-4942 bf16 lever).
    pub(super) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.ts_lin1.cast_weights(dtype)?;
        self.ts_lin2.cast_weights(dtype)?;
        self.linear.cast_weights(dtype)
    }
}

/// Per-stage cache for a SPLIT-RoPE `(cos, sin)` table pair (F-048). The tables depend only on
/// `positions` (+ the model's fixed dims), which are **constant across every denoise step of a
/// stage**, so the expensive single-threaded f64 host trig in [`precompute_split_freqs_cis`] runs
/// once per stage instead of once per step. A cache hit returns byte-equal tables (exactly what a
/// recompute would produce), so the memo stays bit-identical.
///
/// Cache key (sc-7141): the caller's per-stage **epoch** token when supplied
/// ([`LtxDiT::next_rope_epoch`] / [`AvDiT::next_rope_epoch`], bumped once per denoise loop) — an O(1),
/// collision-free identity that replaces the old per-step `positions.as_slice().to_vec()` (a ~1.4 MB
/// host copy + element compare *every* step, ×4 on the AV path). A fresh epoch is never reused for a
/// different `positions`, so a stale hit is impossible — no lossy hash. With no epoch (`None`: the
/// trainer / direct callers) it falls back to the old positions-content compare, byte-for-byte
/// unchanged. Inference is one job per thread (like the LoRA-pass [`Cell`]), so a [`RefCell`] suffices.
#[derive(Default)]
pub(super) struct RopeMemo {
    pub(super) cached: RefCell<Option<(RopeKey, Array, Array)>>,
}

/// [`RopeMemo`] cache key: the per-stage epoch (fast path) or the `positions` content (fallback). The
/// two variants never compare equal, so a memo shared between keyed and fallback callers stays correct.
#[derive(PartialEq)]
pub(super) enum RopeKey {
    Epoch(u64),
    Content(Vec<f32>),
}

impl RopeMemo {
    /// Return the cached `(cos, sin)` if the key is unchanged since the last call, else run `compute`,
    /// cache, and return it. `epoch` is the caller's per-stage token: `Some(e)` keys on the epoch (the
    /// O(1) fast path, no host copy); `None` keys on the `positions` content (the unchanged fallback).
    pub(super) fn get_or_compute(
        &self,
        positions: &Array,
        epoch: Option<u64>,
        compute: impl FnOnce() -> Result<(Array, Array)>,
    ) -> Result<(Array, Array)> {
        let key = match epoch {
            Some(e) => RopeKey::Epoch(e),
            None => RopeKey::Content(positions.as_slice::<f32>().to_vec()),
        };
        if let Some((k, cos, sin)) = self.cached.borrow().as_ref() {
            if *k == key {
                return Ok((cos.clone(), sin.clone()));
            }
        }
        let (cos, sin) = compute()?;
        *self.cached.borrow_mut() = Some((key, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }
}
