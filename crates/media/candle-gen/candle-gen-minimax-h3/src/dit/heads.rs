//! **The DiT's 17 input/output tensors** — everything in `transformer/` that is not a
//! `transformer_blocks.*` or `token_refiner.*` entry.
//!
//! The published index carries exactly **638** tensors: 600 for the 50-block stack (12 each), 21 for
//! the token refiner, and the **17** here. [`DitProjections::names`] enumerates them, and
//! `tests/dit_parity.rs` asserts the three sets partition the fixture's own key set with nothing
//! left over — the "a tensor the loader never reads still decodes to something plausible" gate this
//! crate applies to every component.
//!
//! | | tensors | what |
//! |---|---|---|
//! | `proj_in` | 2 | patchified video rows `in_channels · prod(patch)` → `hidden_size` |
//! | `audio_proj_in` | 2 | audio latents `audio_in_channels` → `hidden_size` |
//! | `context_embedder` | 2 | text context `text_dim` → `hidden_size`, **before** the token refiner |
//! | `time_embedder.linear_{1,2}` | 4 | the timestep MLP every AdaLN projection consumes |
//! | `norm_out.{norm,linear}` | 3 | the packed sequence's final shift/scale-modulated norm |
//! | `proj_out` | 2 | `hidden_size` → the video patch dim |
//! | `audio_proj_out` | 2 | `hidden_size` → `audio_in_channels` |
//!
//! # Three things here are silently wrong if guessed
//!
//! **1. `norm_out.linear` emits `shift` then `scale`, and it is addressed by `timestep_indices` —
//! not by `adaln_indices`.** The block stack's AdaLN table is keyed on
//! `timestep_index · MODALITY_NUM + token_tag`; `norm_out`'s is keyed on the **bare timestep
//! index**, because its projection emits `2 · hidden_size` rather than `6 · MODALITY_NUM ·
//! hidden_size` — the final norm is per-timestep and modality-blind. Feeding it `adaln_indices`
//! reads a row three times too far into the table and is shape-identical.
//! [`AdaLayerNormOut::timestep_indices_from_adaln`] derives the right tensor **from the very tensor
//! the blocks use**, so the two can never disagree.
//!
//! The two halves are `shift, scale = linear(silu(temb)).chunk(2, -1)`, applied as
//! `norm(x)·(1 + scale) + shift`. The reference states the order explicitly (it matches
//! `LTX2Transformer3DModel` and `WanTransformer3DModel`); the reversed reading is shape-identical
//! and produces a plausible, wrong model.
//!
//! **2. `time_proj` is `flip_sin_to_cos = True, downscale_freq_shift = 0`** — so the embedding is
//! `[cos | sin]`, not diffusers' default `[sin | cos]`, and the frequency exponent divides by
//! `half_dim` rather than `half_dim − 1`. Both are invisible in the tensor index (the projection
//! shapes are unchanged) and both change every modulation parameter at every step.
//! [`timestep_sincos`] reproduces the reference's arithmetic and `tests/dit_io.rs` pins it against
//! the fixture's `in.temb`, which is the reference's own `time_embedder(time_proj(t))`.
//!
//! **3. The checkpoint is mixed precision, and these tensors are where it lives.** Of the 17,
//! twelve ship **float32** (`proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`,
//! `audio_proj_out`) and five ship bfloat16 (`context_embedder`, `norm_out`). That is the
//! reference's `_keep_in_fp32_modules` as bytes on disk, so this module loads **at the published
//! dtype** rather than casting everything to the block stack's, and each projection casts its input
//! to its own weight dtype the way the reference's explicit `.to(get_parameter_dtype(...))` casts
//! do. Narrowing `time_embedder` to bf16 would round the one tensor every AdaLN projection in the
//! model reads, biasing all 50 blocks identically at every step — the accumulating-coherently
//! failure the reference's own comment warns about.
//!
//! Note this makes `Weights::from_file`'s `dtype` argument the wrong tool for loading
//! `transformer/` on this lane: it casts uniformly. [`crate::dit::MiniMaxH3Dit::load`] therefore
//! reads the shards at their stored dtype and casts only the block stack.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM};
use crate::dit::layers::RmsNorm;
use crate::nn::{linear, silu};

/// `max_period` of the sinusoidal timestep embedding — `get_timestep_embedding`'s default, which
/// `Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)` does not
/// override.
pub const TIME_PROJ_MAX_PERIOD: f64 = 10_000.0;

/// `Timesteps.flip_sin_to_cos` for MiniMax-H3: the embedding is `[cos | sin]`.
pub const TIME_PROJ_FLIP_SIN_TO_COS: bool = true;

/// `Timesteps.downscale_freq_shift` for MiniMax-H3: **0**, so the exponent divides by `half_dim`
/// and not by `half_dim − 1`.
pub const TIME_PROJ_DOWNSCALE_FREQ_SHIFT: f64 = 0.0;

/// An `nn.Linear(in, out, bias=True)` over a stored `[out, in]` weight.
///
/// Distinct from [`crate::dit::layers::LinearNoBias`] because **every** projection in this module
/// carries a bias while every projection inside a block carries none — which is exactly how the
/// published index reads.
#[derive(Debug, Clone)]
pub struct LinearBias {
    weight: Tensor,
    bias: Tensor,
}

impl LinearBias {
    /// Load `{prefix}.weight` / `{prefix}.bias` **at their stored dtype**, checking the shape.
    ///
    /// The dtype is deliberately not a parameter: these tensors are the mixed-precision half of the
    /// checkpoint (see the module docs), so the bytes on disk are the contract.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?;
        let bias = w.require(&format!("{prefix}.bias"))?;
        if weight.dims() != [out_features, in_features] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit {prefix}.weight: expected [{out_features}, {in_features}], got {:?}",
                weight.dims()
            )));
        }
        if bias.dims() != [out_features] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit {prefix}.bias: expected [{out_features}], got {:?}",
                bias.dims()
            )));
        }
        Ok(Self { weight, bias })
    }

    /// The two tensor names this projection consumes.
    pub fn names(prefix: &str) -> [String; 2] {
        [format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// The dtype the stored bytes carry — f32 for the fp32-kept heads, bf16 otherwise.
    pub fn dtype(&self) -> DType {
        self.weight.dtype()
    }

    /// Device bytes this projection holds.
    pub fn nbytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
            + self.bias.elem_count() * self.bias.dtype().size_in_bytes()
    }

    /// The loaded bias, as read off disk.
    ///
    /// Exposed for `tests/ref2va_checkpoint.rs` (sc-17157): `transformer/` and `transformer_ref/`
    /// are structurally identical and differ only in their **values**, so the only thing that can
    /// tell a `ref2va` load from a base one is a tensor read back off the loaded model.
    /// `proj_in.bias` is the probe — float32, `[inner_dim]`, and different between the two.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// `y = x · Wᵀ + b`, with `x` cast to the weight's own dtype first.
    ///
    /// The cast mirrors the reference's explicit `x.to(get_parameter_dtype(module))` and is what
    /// keeps a float32 head from silently running its matmul at the block stack's bfloat16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(self.weight.dtype())?;
        linear(&x, &self.weight, &self.bias)
    }
}

/// `Timesteps(num_channels = freq_dim, flip_sin_to_cos = True, downscale_freq_shift = 0)` —
/// diffusers' `get_timestep_embedding` for MiniMax-H3's settings, at float32.
///
/// ```text
/// exponent[i] = -ln(max_period) · i / half_dim          (i in [0, half_dim))
/// angle[n, i] = t[n] · exp(exponent[i])
/// emb[n]      = [cos(angle[n]) | sin(angle[n])]         <- flipped: cos FIRST
/// ```
///
/// Two departures from the diffusers default that no shape can reveal: the flip, and the `−0`
/// rather than `−1` in the exponent's denominator. `freq_dim` is even in every shipped and fixture
/// config, so the odd-width zero pad the reference carries is rejected here rather than reproduced
/// — an odd `freq_dim` would silently change the width the timestep MLP consumes.
pub fn timestep_sincos(
    timesteps: &[f32],
    freq_dim: usize,
    device: &candle_gen::candle_core::Device,
) -> Result<Tensor> {
    if timesteps.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 dit time_proj: cannot embed an empty timestep list".into(),
        ));
    }
    if freq_dim == 0 || !freq_dim.is_multiple_of(2) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit time_proj: freq_dim must be positive and even, got {freq_dim}"
        )));
    }
    if let Some(bad) = timesteps.iter().find(|t| !t.is_finite()) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit time_proj: timestep {bad} is not finite"
        )));
    }
    let half = freq_dim / 2;
    // `exponent / (half_dim - downscale_freq_shift)` with the shift at 0.
    let denom = half as f64 - TIME_PROJ_DOWNSCALE_FREQ_SHIFT;
    let inv: Vec<f32> = (0..half)
        .map(|i| (-TIME_PROJ_MAX_PERIOD.ln() * i as f64 / denom).exp() as f32)
        .collect();

    let n = timesteps.len();
    let mut out = Vec::with_capacity(n * freq_dim);
    for &t in timesteps {
        let angles: Vec<f32> = inv.iter().map(|f| t * f).collect();
        if TIME_PROJ_FLIP_SIN_TO_COS {
            out.extend(angles.iter().map(|a| a.cos()));
            out.extend(angles.iter().map(|a| a.sin()));
        } else {
            out.extend(angles.iter().map(|a| a.sin()));
            out.extend(angles.iter().map(|a| a.cos()));
        }
    }
    Ok(Tensor::from_vec(out, (n, freq_dim), device)?)
}

/// `time_proj` + `time_embedder` — the timestep MLP whose output every AdaLN projection in the
/// model consumes.
///
/// `TimestepEmbedding(in_channels = freq_dim, time_embed_dim = time_embed_hidden_dim,
/// out_dim = time_embed_dim)` with the default `silu` activation and **no** post-activation:
/// `linear_1 → silu → linear_2`.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear_1: LinearBias,
    linear_2: LinearBias,
    freq_dim: usize,
    time_embed_dim: usize,
}

impl TimestepEmbedder {
    /// Load `{prefix}.linear_1` / `{prefix}.linear_2` (e.g. `time_embedder`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: LinearBias::from_weights(
                w,
                &format!("{prefix}.linear_1"),
                cfg.freq_dim,
                cfg.time_embed_hidden_dim,
            )?,
            linear_2: LinearBias::from_weights(
                w,
                &format!("{prefix}.linear_2"),
                cfg.time_embed_hidden_dim,
                cfg.time_embed_dim,
            )?,
            freq_dim: cfg.freq_dim,
            time_embed_dim: cfg.time_embed_dim,
        })
    }

    /// The four tensor names the timestep MLP consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = LinearBias::names(&format!("{prefix}.linear_1")).to_vec();
        v.extend(LinearBias::names(&format!("{prefix}.linear_2")));
        v
    }

    /// Width of the embedding this produces — `time_embed_dim`, and the input of every AdaLN
    /// projection.
    pub fn time_embed_dim(&self) -> usize {
        self.time_embed_dim
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.linear_1.nbytes() + self.linear_2.nbytes()
    }

    /// `[num_timesteps]` unscaled timesteps in `[0, 1]` → `[num_timesteps, time_embed_dim]`.
    ///
    /// This is the closure [`crate::dit::adaln::AdaLnCache::precompute`] takes: it is applied to the
    /// schedule's own [`crate::dit::adaln::TimestepSchedule::distinct_timesteps`], so the embedding
    /// cannot be mis-bound to a different timestep list.
    pub fn forward(
        &self,
        timesteps: &[f32],
        device: &candle_gen::candle_core::Device,
    ) -> Result<Tensor> {
        let projected = timestep_sincos(timesteps, self.freq_dim, device)?;
        let h = self.linear_1.forward(&projected)?;
        self.linear_2.forward(&silu(&h)?)
    }
}

/// One evaluated `norm_out` modulation table: `[num_timesteps, hidden_size]` each.
///
/// Produced once per run alongside the block stack's [`crate::dit::adaln::AdaLnCache`], because it
/// is the same kind of quantity — a function of the timestep embedding only.
#[derive(Debug, Clone)]
pub struct NormOutModulation {
    /// Per-timestep additive shift.
    pub shift: Tensor,
    /// Per-timestep multiplicative scale, applied as `1 + scale`.
    pub scale: Tensor,
}

/// The packed sequence's final norm: an RMSNorm whose shift and scale are gathered **per row** from
/// a per-timestep table (`MiniMaxH3AdaLayerNormOut`).
///
/// Same checkpoint keys as diffusers' `AdaLayerNormContinuous` (`norm` + `linear`), with the two
/// MiniMax-H3 specifics stated in the module docs: the table is addressed by the row's **timestep
/// index alone**, and the projection's halves are `shift` then `scale`.
#[derive(Debug, Clone)]
pub struct AdaLayerNormOut {
    norm: RmsNorm,
    linear: LinearBias,
    hidden_size: usize,
}

impl AdaLayerNormOut {
    /// Load `{prefix}.norm` / `{prefix}.linear` (e.g. `norm_out`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
        let dtype = w.require(&format!("{prefix}.norm.weight"))?.dtype();
        Ok(Self {
            norm: RmsNorm::from_weights(w, &format!("{prefix}.norm"), cfg.final_norm_eps, dtype)?,
            linear: LinearBias::from_weights(
                w,
                &format!("{prefix}.linear"),
                cfg.time_embed_dim,
                2 * cfg.hidden_size,
            )?,
            hidden_size: cfg.hidden_size,
        })
    }

    /// The three tensor names the output norm consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = RmsNorm::names(&format!("{prefix}.norm")).to_vec();
        v.extend(LinearBias::names(&format!("{prefix}.linear")));
        v
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.linear.nbytes()
    }

    /// Project the timestep embedding into `(shift, scale)` — **shift is the FIRST half**.
    ///
    /// As in [`crate::dit::block::AdaLnProjection::forward`], the activation runs at `temb`'s own
    /// (float32) precision and only its result is cast to the projection's dtype.
    pub fn modulation(&self, temb: &Tensor) -> Result<NormOutModulation> {
        let s = temb.dims();
        if s.len() != 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit norm_out: expected temb as [num_timesteps, time_embed_dim], got \
                 {s:?}"
            )));
        }
        let projected = self.linear.forward(&silu(temb)?)?;
        Ok(NormOutModulation {
            shift: projected.narrow(1, 0, self.hidden_size)?.contiguous()?,
            scale: projected
                .narrow(1, self.hidden_size, self.hidden_size)?
                .contiguous()?,
        })
    }

    /// `norm(x) · (1 + scale[timestep_indices]) + shift[timestep_indices]`.
    ///
    /// `timestep_indices` is one entry per row of the packed sequence — the row's index into the
    /// **distinct timestep** table, with no modality factor. Bounds-checked here so an index that
    /// is `MODALITY_NUM`× too large names the mistake rather than failing inside a kernel.
    pub fn apply(
        &self,
        x: &Tensor,
        modulation: &NormOutModulation,
        timestep_indices: &Tensor,
    ) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit norm_out: expected [B, S, hidden], got {s:?}"
            )));
        }
        if timestep_indices.dims() != [s[1]] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit norm_out: timestep_indices must be [seq_len={}], got {:?}",
                s[1],
                timestep_indices.dims()
            )));
        }
        let rows = modulation.shift.dims()[0];
        let idx = timestep_indices
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        if let Some(&bad) = idx.iter().find(|&&i| i as usize >= rows) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit norm_out: timestep index {bad} is outside the modulation table's \
                 {rows} timesteps. NOTE norm_out is addressed by the BARE timestep index, not by \
                 the blocks' `timestep_index · {MODALITY_NUM} + tag`"
            )));
        }
        let idx = timestep_indices.to_dtype(DType::U32)?;

        let normed = self.norm.forward(x)?;
        let seq = s[1];
        let scale = modulation
            .scale
            .index_select(&idx, 0)?
            .to_dtype(normed.dtype())?
            .reshape((1, seq, self.hidden_size))?;
        let shift = modulation
            .shift
            .index_select(&idx, 0)?
            .to_dtype(normed.dtype())?
            .reshape((1, seq, self.hidden_size))?;
        Ok(normed
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?)
    }

    /// Recover the per-row **timestep** index from the per-row **AdaLN** index the block stack uses.
    ///
    /// `adaln_indices = timestep_index · MODALITY_NUM + token_tag` with `0 ≤ tag < MODALITY_NUM`, so
    /// the integer quotient is exactly the timestep index. Deriving it rather than rebuilding it
    /// from the schedule is deliberate: the two tensors then cannot disagree, whatever a caller
    /// passes. `tests/dit_io.rs` additionally checks this against an independent derivation straight
    /// from [`crate::dit::adaln::TimestepSchedule`], so the shortcut is pinned rather than assumed.
    pub fn timestep_indices_from_adaln(adaln_indices: &Tensor) -> Result<Tensor> {
        let idx = adaln_indices
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let quotient: Vec<u32> = idx.iter().map(|i| i / MODALITY_NUM as u32).collect();
        Ok(Tensor::from_vec(
            quotient,
            adaln_indices.dims().to_vec(),
            adaln_indices.device(),
        )?)
    }
}

/// **The 17 input/output tensors of `transformer/`**, loaded together so the set is one auditable
/// unit rather than seven scattered fields.
#[derive(Debug, Clone)]
pub struct DitProjections {
    /// Patchified video rows → the residual stream. **float32** in the published checkpoint.
    pub proj_in: LinearBias,
    /// Audio latent rows → the residual stream. **float32**.
    pub audio_proj_in: LinearBias,
    /// The text encoder's 5120-wide context → the residual stream. bfloat16, and it runs **before**
    /// the token refiner.
    pub context_embedder: LinearBias,
    /// `time_proj` + `time_embedder`. **float32**.
    pub time_embedder: TimestepEmbedder,
    /// The packed sequence's final modulated norm. bfloat16.
    pub norm_out: AdaLayerNormOut,
    /// The residual stream → the video patch dim. **float32**.
    pub proj_out: LinearBias,
    /// The residual stream → `audio_in_channels`. **float32**.
    pub audio_proj_out: LinearBias,
}

impl DitProjections {
    /// Tensors this module owns — exactly 17 in the published checkpoint.
    pub const TENSOR_COUNT: usize = 17;

    /// Load all 17, at their stored dtypes.
    pub fn from_weights(w: &Weights, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            proj_in: LinearBias::from_weights(
                w,
                "proj_in",
                cfg.video_patch_dim(),
                cfg.hidden_size,
            )?,
            audio_proj_in: LinearBias::from_weights(
                w,
                "audio_proj_in",
                cfg.audio_in_channels,
                cfg.hidden_size,
            )?,
            context_embedder: LinearBias::from_weights(
                w,
                "context_embedder",
                cfg.text_dim,
                cfg.hidden_size,
            )?,
            time_embedder: TimestepEmbedder::from_weights(w, "time_embedder", cfg)?,
            norm_out: AdaLayerNormOut::from_weights(w, "norm_out", cfg)?,
            proj_out: LinearBias::from_weights(
                w,
                "proj_out",
                cfg.hidden_size,
                cfg.video_patch_dim(),
            )?,
            audio_proj_out: LinearBias::from_weights(
                w,
                "audio_proj_out",
                cfg.hidden_size,
                cfg.audio_in_channels,
            )?,
        })
    }

    /// Every tensor name these projections consume — the 17.
    pub fn names() -> Vec<String> {
        let mut v = LinearBias::names("proj_in").to_vec();
        v.extend(LinearBias::names("audio_proj_in"));
        v.extend(LinearBias::names("context_embedder"));
        v.extend(TimestepEmbedder::names("time_embedder"));
        v.extend(AdaLayerNormOut::names("norm_out"));
        v.extend(LinearBias::names("proj_out"));
        v.extend(LinearBias::names("audio_proj_out"));
        v
    }

    /// Device bytes the 17 hold — small next to the stack, and the only float32 weights in the DiT.
    pub fn nbytes(&self) -> usize {
        self.proj_in.nbytes()
            + self.audio_proj_in.nbytes()
            + self.context_embedder.nbytes()
            + self.time_embedder.nbytes()
            + self.norm_out.nbytes()
            + self.proj_out.nbytes()
            + self.audio_proj_out.nbytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The set is exactly 17 names, all distinct, and none of them belongs to a block or the
    /// refiner. 638 published tensors = 50·12 + 21 + 17.
    #[test]
    fn the_seventeen_names_are_the_published_top_level_set() {
        let names = DitProjections::names();
        assert_eq!(names.len(), DitProjections::TENSOR_COUNT, "{names:?}");
        assert_eq!(names.len(), 17);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate names");
        assert!(
            !names
                .iter()
                .any(|n| n.starts_with("transformer_blocks.") || n.starts_with("token_refiner.")),
            "these are the TOP-LEVEL tensors only"
        );
        let cfg = MiniMaxH3DitConfig::default();
        assert_eq!(
            cfg.num_layers * 12 + 21 + names.len(),
            638,
            "the three sets must partition the published index"
        );
        for expect in [
            "proj_in.weight",
            "audio_proj_in.bias",
            "context_embedder.weight",
            "time_embedder.linear_1.weight",
            "time_embedder.linear_2.bias",
            "norm_out.norm.weight",
            "norm_out.linear.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
        ] {
            assert!(names.contains(&expect.to_string()), "missing {expect}");
        }
    }

    /// **The flip.** `flip_sin_to_cos = True` puts COSINE first. The unflipped order is the
    /// diffusers default and is shape-identical.
    #[test]
    fn the_sinusoidal_embedding_is_cosine_first() {
        let emb = timestep_sincos(&[1.0], 4, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[1, 4]);
        let v = flat(&emb);
        // half_dim 2; inv_freq[0] = 1, inv_freq[1] = 10000^(-1/2) = 0.01.
        assert!(
            (v[0] - 1.0f32.cos()).abs() < 1e-6,
            "channel 0 must be cos: {v:?}"
        );
        assert!((v[1] - 0.01f32.cos()).abs() < 1e-6, "{v:?}");
        assert!(
            (v[2] - 1.0f32.sin()).abs() < 1e-6,
            "channel 2 must be sin: {v:?}"
        );
        assert!((v[3] - 0.01f32.sin()).abs() < 1e-6, "{v:?}");
        // ...and the two orders are genuinely different at this timestep.
        assert!(
            (v[0] - v[2]).abs() > 0.3,
            "cos and sin must differ here or the flip is untestable"
        );
    }

    /// **The `downscale_freq_shift = 0`.** Dividing by `half_dim − 1` instead — diffusers' own
    /// default — moves every frequency but the first.
    #[test]
    fn the_frequency_exponent_divides_by_half_dim_not_half_dim_minus_one() {
        // Probed at a large timestep on purpose: at `t = 1` the two candidate frequencies are 1e-3
        // and 1e-4, whose cosines agree to 5e-7 — the difference is real but below the f32 noise a
        // test can gate on, so a small probe would pass against BOTH denominators.
        let t = 1000.0f32;
        let emb = timestep_sincos(&[t], 8, &Device::Cpu).unwrap();
        let v = flat(&emb);
        // half_dim 4; shift 0 => inv_freq[3] = 10000^(-3/4). The `-1` variant would give
        // 10000^(-3/3) = 1e-4.
        let want = t * (10_000f32).powf(-3.0 / 4.0);
        let wrong = t * (10_000f32).powf(-1.0);
        assert!((v[3] - want.cos()).abs() < 1e-5, "{v:?}");
        assert!(
            (want.cos() - wrong.cos()).abs() > 0.4,
            "the two denominators must be distinguishable ({} vs {})",
            want.cos(),
            wrong.cos()
        );
        assert!(
            timestep_sincos(&[1.0], 5, &Device::Cpu).is_err(),
            "odd freq_dim"
        );
        assert!(
            timestep_sincos(&[], 8, &Device::Cpu).is_err(),
            "no timesteps"
        );
        assert!(
            timestep_sincos(&[f32::NAN], 8, &Device::Cpu).is_err(),
            "non-finite"
        );
    }

    /// The AdaLN index carries the modality; the `norm_out` index must not. The quotient recovers
    /// it exactly.
    #[test]
    fn the_norm_out_index_is_the_adaln_index_without_the_modality() {
        let dev = Device::Cpu;
        let m = MODALITY_NUM as u32;
        // timestep 0 tags 0/1/2, then timestep 3 tag 1, then timestep 2 tag 2.
        let adaln = Tensor::from_vec(vec![0u32, 1, 2, 3 * m + 1, 2 * m + 2], (5,), &dev).unwrap();
        let t = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
        assert_eq!(t.to_vec1::<u32>().unwrap(), vec![0, 0, 0, 3, 2]);
    }

    /// `shift` is the FIRST half and `scale` the second, and the scale is applied as `1 + scale`.
    /// Built over a projection whose two halves are trivially distinguishable, so a reversed
    /// reading cannot pass.
    #[test]
    fn norm_out_modulation_is_shift_then_scale() {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "n.norm.weight".into(),
            Tensor::from_vec(vec![1.0f32, 1.0], (2,), &dev).unwrap(),
        );
        // hidden_size 2 => out 4 = [shift(2) | scale(2)].
        map.insert(
            "n.linear.weight".into(),
            Tensor::from_vec(vec![5.0f32, 6.0, 0.5, 0.25], (4, 1), &dev).unwrap(),
        );
        map.insert(
            "n.linear.bias".into(),
            Tensor::from_vec(vec![0.0f32; 4], (4,), &dev).unwrap(),
        );
        let w = Weights::from_map(map);
        let cfg = MiniMaxH3DitConfig {
            hidden_size: 2,
            time_embed_dim: 1,
            ..MiniMaxH3DitConfig::default()
        };
        let n = AdaLayerNormOut::from_weights(&w, "n", &cfg).unwrap();
        let temb = Tensor::from_vec(vec![2.0f32], (1, 1), &dev).unwrap();
        let m = n.modulation(&temb).unwrap();
        let act = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert_eq!(m.shift.dims(), &[1, 2]);
        assert!((flat(&m.shift)[0] - 5.0 * act).abs() < 1e-5);
        assert!((flat(&m.scale)[0] - 0.5 * act).abs() < 1e-5);
        assert!(
            flat(&m.shift)[0] > 4.0 * flat(&m.scale)[0],
            "the two halves must not be interchangeable"
        );

        // An out-of-range timestep index is a typed error rather than a kernel failure.
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 1, 2), &dev).unwrap();
        let idx = |v: u32| Tensor::from_vec(vec![v], (1,), &dev).unwrap();
        n.apply(&x, &m, &idx(0)).unwrap();
        let e = n.apply(&x, &m, &idx(1)).unwrap_err().to_string();
        assert!(e.contains("BARE timestep index"), "{e}");
        assert!(n
            .modulation(&Tensor::ones(3, DType::F32, &dev).unwrap())
            .is_err());
    }
}
