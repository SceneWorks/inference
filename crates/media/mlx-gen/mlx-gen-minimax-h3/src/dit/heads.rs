//! **The DiT's 17 input/output tensors** (sc-17147) — everything in `transformer/` that is not a
//! `transformer_blocks.*` or `token_refiner.*` entry.
//!
//! The published index carries exactly **638** tensors: 600 for the 50-block stack (12 each), 21 for
//! the token refiner, and the **17** here. [`DitProjections::names`] enumerates them, and
//! `tests/real_weights.rs` asserts the three sets partition the published index with nothing left
//! over — the "a tensor the loader never reads still decodes to something plausible" gate this crate
//! applies to every component.
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

use mlx_rs::ops::{add, concatenate_axis, cos, multiply, sin};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM};
use crate::dit::layers::RmsNorm;
use crate::tensor::slice_axis;

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
    weight: Array,
    bias: Array,
    /// Forward-time LoRA/LoKr residuals on this projection (SW-7096). Empty unless an install
    /// routes here — only the image DiT does.
    adapters: ResidualSlot,
}

/// The adapter residuals of a projection whose base is not an
/// [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear) (SW-7096) — the raw weight/bias
/// [`LinearBias`] the mixed-precision I/O heads use.
///
/// The slot is an `AdaptableLinear` over a **shape-only** base (a lazy zero array that is never
/// evaluated), so the shared install can push LoRA and LoKr residuals onto it exactly as onto a
/// block leaf and read the logical `[out, in]` shape a LoKr needs. Only its residuals run: the
/// owning projection computes its base itself, then [`Self::apply`] adds `Σ residual(x)` under the
/// same rules the shared seam uses (a zero-scale adapter is skipped; each residual is narrowed to
/// the base output's dtype before the add).
#[derive(Clone, Default)]
pub struct ResidualSlot(Option<Box<mlx_gen::adapters::AdaptableLinear>>);

impl std::fmt::Debug for ResidualSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ResidualSlot")
            .field(&self.0.as_ref().map_or(0, |slot| slot.adapters().len()))
            .finish()
    }
}

impl ResidualSlot {
    /// The slot for an `[out, in]` projection, created on first use.
    pub fn slot_mut(
        &mut self,
        out_features: i32,
        in_features: i32,
    ) -> Option<&mut mlx_gen::adapters::AdaptableLinear> {
        if self.0.is_none() {
            let base = Array::zeros::<f32>(&[out_features, in_features]).ok()?;
            self.0 = Some(Box::new(mlx_gen::adapters::AdaptableLinear::dense(
                base, None,
            )));
        }
        self.0.as_deref_mut()
    }

    /// Residuals held.
    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |slot| slot.adapters().len())
    }

    /// Whether no residual is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `y + Σ residual(x)`; `y` untouched when the slot holds nothing.
    pub fn apply(&self, x: &Array, mut y: Array) -> Result<Array> {
        let Some(slot) = &self.0 else {
            return Ok(y);
        };
        for adapter in slot.adapters() {
            if adapter.is_disabled() {
                continue;
            }
            let residual = adapter.residual(x)?;
            let residual = if residual.dtype() == y.dtype() {
                residual
            } else {
                residual.as_dtype(y.dtype())?
            };
            y = mlx_rs::ops::add(&y, &residual)?;
        }
        Ok(y)
    }
}

impl LinearBias {
    /// The loaded weight, at its published dtype.
    ///
    /// Read-only, and exposed so a test can assert **which checkpoint** a loaded model carries.
    /// `transformer/` and `transformer_ref/` are structurally identical — byte-identical configs,
    /// the same 638 tensor names — so the only thing that separates them is a value, and the only
    /// honest place to read that value is off the loaded model rather than off a file the loader
    /// may never have opened. See `tests/ref2va_checkpoint.rs`.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// The loaded bias, at its published dtype. See [`Self::weight`] for why this is readable.
    pub fn bias(&self) -> &Array {
        &self.bias
    }

    /// Load `{prefix}.weight` / `{prefix}.bias` **at their published dtype**, checking the shape.
    ///
    /// The dtype is deliberately not a parameter: these tensors are the mixed-precision half of the
    /// checkpoint (see the module docs), so the bytes on disk are the contract.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        in_features: i32,
        out_features: i32,
    ) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let bias = w.require(&format!("{prefix}.bias"))?.clone();
        if weight.shape() != [out_features, in_features] {
            return Err(Error::Msg(format!(
                "minimax-h3 dit {prefix}.weight: expected [{out_features}, {in_features}], got {:?}",
                weight.shape()
            )));
        }
        if bias.shape() != [out_features] {
            return Err(Error::Msg(format!(
                "minimax-h3 dit {prefix}.bias: expected [{out_features}], got {:?}",
                bias.shape()
            )));
        }
        Ok(Self {
            weight,
            bias,
            adapters: ResidualSlot::default(),
        })
    }

    /// This projection's adapter slot (SW-7096), created on first use at its `[out, in]` shape.
    pub fn adaptable_mut(&mut self) -> Option<&mut mlx_gen::adapters::AdaptableLinear> {
        let (out, inp) = (self.weight.shape()[0], self.weight.shape()[1]);
        self.adapters.slot_mut(out, inp)
    }

    /// Residuals installed on this projection.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// The two tensor names this projection consumes.
    pub fn names(prefix: &str) -> [String; 2] {
        [format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// The dtype the published bytes carry — f32 for the fp32-kept heads, bf16 otherwise.
    pub fn dtype(&self) -> Dtype {
        self.weight.dtype()
    }

    /// Device bytes this projection holds.
    pub fn nbytes(&self) -> usize {
        self.weight.nbytes() + self.bias.nbytes()
    }

    /// `y = x · Wᵀ + b`, with `x` cast to the weight's own dtype first.
    ///
    /// The cast mirrors the reference's explicit `x.to(get_parameter_dtype(module))` and is what
    /// keeps a float32 head from silently running its matmul at the block stack's bfloat16.
    ///
    /// A float32 projection (the reference's `_keep_in_fp32_modules`: `proj_in`, `time_embedder`,
    /// the output heads) runs through [`mlx_gen::nn::linear_precise`]: an ordinary F32 `addmm`
    /// runs as TF32 on NAX, which rounded `time_embedder`'s output to 2.8e-3 rel-L2 (SW-7092) —
    /// the tensor every AdaLN projection reads. See [`uses_precise_dispatch`].
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(self.weight.dtype())?;
        let y = if uses_precise_dispatch(self.weight.dtype()) {
            mlx_gen::nn::linear_precise(&x, &self.weight, Some(&self.bias))?
        } else {
            mlx_gen::nn::linear(&x, &self.weight, &self.bias)?
        };
        // SW-7096: the residuals read the same cast input the base does — what a PEFT LoRA on
        // this module sees in the reference, which casts before calling the module.
        self.adapters.apply(&x, y)
    }
}

/// Whether a [`LinearBias`] over a weight of `weight_dtype` takes the full-precision N=1 dispatch
/// ([`mlx_gen::nn::linear_precise`]) rather than the fused `addmm`: exactly the float32 weights,
/// which are the modules the reference keeps in float32. A bfloat16 projection keeps the fused
/// `addmm`, whose single rounding of the bias add is the bf16 contract.
pub fn uses_precise_dispatch(weight_dtype: Dtype) -> bool {
    weight_dtype == Dtype::Float32
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
pub fn timestep_sincos(timesteps: &[f32], freq_dim: i32) -> Result<Array> {
    if timesteps.is_empty() {
        return Err(Error::Msg(
            "minimax-h3 dit time_proj: cannot embed an empty timestep list".into(),
        ));
    }
    if freq_dim <= 0 || freq_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 dit time_proj: freq_dim must be positive and even, got {freq_dim}"
        )));
    }
    if let Some(bad) = timesteps.iter().find(|t| !t.is_finite()) {
        return Err(Error::Msg(format!(
            "minimax-h3 dit time_proj: timestep {bad} is not finite"
        )));
    }
    let half = (freq_dim / 2) as usize;
    // `exponent / (half_dim - downscale_freq_shift)` with the shift at 0.
    let denom = half as f64 - TIME_PROJ_DOWNSCALE_FREQ_SHIFT;
    let inv: Vec<f32> = (0..half)
        .map(|i| (-TIME_PROJ_MAX_PERIOD.ln() * i as f64 / denom).exp() as f32)
        .collect();

    let n = timesteps.len() as i32;
    let t = Array::from_slice(timesteps, &[n, 1]);
    let inv = Array::from_slice(&inv, &[1, half as i32]);
    let angle = multiply(&t, &inv)?;
    let (c, s) = (cos(&angle)?, sin(&angle)?);
    Ok(if TIME_PROJ_FLIP_SIN_TO_COS {
        concatenate_axis(&[c, s], -1)?
    } else {
        concatenate_axis(&[s, c], -1)?
    })
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
    freq_dim: i32,
    time_embed_dim: i32,
}

impl TimestepEmbedder {
    /// `linear_1` / `linear_2` for an adapter install (SW-7096).
    pub fn linear_mut(&mut self, name: &str) -> Option<&mut LinearBias> {
        match name {
            "linear_1" => Some(&mut self.linear_1),
            "linear_2" => Some(&mut self.linear_2),
            _ => None,
        }
    }

    /// Load `{prefix}.linear_1` / `{prefix}.linear_2` (e.g. `time_embedder`).
    pub fn from_weights(w: &mut Weights, prefix: &str, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
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
    pub fn time_embed_dim(&self) -> i32 {
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
    pub fn forward(&self, timesteps: &[f32]) -> Result<Array> {
        let projected = timestep_sincos(timesteps, self.freq_dim)?;
        let h = self.linear_1.forward(&projected)?;
        self.linear_2.forward(&silu(&h)?)
    }
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
    /// `None` on a pruned topology, whose final projection is the reduced
    /// [`crate::dit::pruned_adaln::PrunedFinalProjection`] instead.
    linear: Option<LinearBias>,
    hidden_size: i32,
}

/// One evaluated `norm_out` modulation table: `[num_timesteps, hidden_size]` each.
///
/// Produced once per run alongside the block stack's [`crate::dit::adaln::AdaLnCache`], because it
/// is the same kind of quantity — a function of the timestep embedding only.
#[derive(Debug, Clone)]
pub struct NormOutModulation {
    /// Per-timestep additive shift.
    pub shift: Array,
    /// Per-timestep multiplicative scale, applied as `1 + scale`.
    pub scale: Array,
}

impl AdaLayerNormOut {
    /// `linear` for an adapter install (SW-7096); `None` on a pruned topology.
    pub fn linear_mut(&mut self) -> Option<&mut LinearBias> {
        self.linear.as_mut()
    }

    /// Load `{prefix}.norm` / `{prefix}.linear` (e.g. `norm_out`).
    pub fn from_weights(w: &mut Weights, prefix: &str, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
        let norm_weight = w.require(&format!("{prefix}.norm.weight"))?.clone();
        let dtype = norm_weight.dtype();
        Ok(Self {
            norm: RmsNorm::from_weights(w, &format!("{prefix}.norm"), cfg.final_norm_eps, dtype)?,
            linear: Some(LinearBias::from_weights(
                w,
                &format!("{prefix}.linear"),
                cfg.time_embed_dim,
                2 * cfg.hidden_size,
            )?),
            hidden_size: cfg.hidden_size,
        })
    }

    /// Load only `{prefix}.norm` — the shared RMSNorm a **pruned** model keeps — leaving `linear`
    /// `None`. The reduced final projection is owned by
    /// [`crate::dit::pruned_adaln::PrunedFinalProjection`].
    pub fn from_weights_norm_only(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
    ) -> Result<Self> {
        let norm_weight = w.require(&format!("{prefix}.norm.weight"))?.clone();
        let dtype = norm_weight.dtype();
        Ok(Self {
            norm: RmsNorm::from_weights(w, &format!("{prefix}.norm"), cfg.final_norm_eps, dtype)?,
            linear: None,
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
        self.linear.as_ref().map_or(0, LinearBias::nbytes)
    }

    /// Project the timestep embedding into `(shift, scale)` — **shift is the FIRST half**.
    ///
    /// As in [`crate::dit::block::AdaLnProjection::forward`], the activation runs at `temb`'s own
    /// (float32) precision and only its result is cast to the projection's dtype.
    ///
    /// A pruned topology has no full `linear`; its reduced projection is
    /// [`crate::dit::pruned_adaln::PrunedFinalProjection`] and this method refuses rather than
    /// falling back.
    pub fn modulation(&self, temb: &Array) -> Result<NormOutModulation> {
        let linear = self.linear.as_ref().ok_or_else(|| {
            Error::Msg(
                "minimax-h3 dit norm_out: this is a pruned topology; modulate through \
                 PrunedFinalProjection, not the full norm_out.linear"
                    .into(),
            )
        })?;
        let s = temb.shape();
        if s.len() != 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit norm_out: expected temb as [num_timesteps, time_embed_dim], got \
                 {s:?}"
            )));
        }
        let projected = linear.forward(&silu(temb)?)?;
        Ok(NormOutModulation {
            shift: slice_axis(&projected, 1, 0, self.hidden_size)?,
            scale: slice_axis(&projected, 1, self.hidden_size, 2 * self.hidden_size)?,
        })
    }

    /// `norm(x) · (1 + scale[timestep_indices]) + shift[timestep_indices]`.
    ///
    /// `timestep_indices` is one entry per row of the packed sequence — the row's index into the
    /// **distinct timestep** table, with no modality factor. Bounds-checked here because MLX
    pub fn apply(
        &self,
        x: &Array,
        modulation: &NormOutModulation,
        timestep_indices: &Array,
    ) -> Result<Array> {
        let s = x.shape();
        if s.len() != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit norm_out: expected [B, S, hidden], got {s:?}"
            )));
        }
        if timestep_indices.shape() != [s[1]] {
            return Err(Error::Msg(format!(
                "minimax-h3 dit norm_out: timestep_indices must be [seq_len={}], got {:?}",
                s[1],
                timestep_indices.shape()
            )));
        }
        let rows = modulation.shift.shape()[0];
        let idx = timestep_indices.as_dtype(Dtype::Int32)?;
        let min: i32 = idx.min(None)?.item();
        let max: i32 = idx.max(None)?.item();
        if min < 0 || max >= rows {
            return Err(Error::Msg(format!(
                "minimax-h3 dit norm_out: timestep index range [{min}, {max}] is outside the \
                 modulation table's {rows} timesteps; MLX gathers out of bounds silently rather \
                 than failing. NOTE norm_out is addressed by the BARE timestep index, not by the \
                 blocks' `timestep_index · {MODALITY_NUM} + tag`"
            )));
        }
        let normed = self.norm.forward(x)?;
        let scale = modulation
            .scale
            .take_axis(&idx, 0)?
            .as_dtype(normed.dtype())?;
        let shift = modulation
            .shift
            .take_axis(&idx, 0)?
            .as_dtype(normed.dtype())?;
        let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
        Ok(add(&multiply(&normed, &add(&one, &scale)?)?, &shift)?)
    }

    /// Recover the per-row **timestep** index from the per-row **AdaLN** index the block stack uses.
    ///
    /// `adaln_indices = timestep_index · MODALITY_NUM + token_tag` with `0 ≤ tag < MODALITY_NUM`, so
    /// the integer quotient is exactly the timestep index. Deriving it rather than rebuilding it
    /// from the schedule is deliberate: the two tensors then cannot disagree, whatever a caller
    /// passes. `tests/dit_io.rs` additionally checks this against an independent derivation straight
    /// from [`crate::dit::adaln::TimestepSchedule`], so the shortcut is pinned rather than assumed.
    pub fn timestep_indices_from_adaln(adaln_indices: &Array) -> Result<Array> {
        let idx = adaln_indices.as_dtype(Dtype::Int32)?;
        let min: i32 = idx.min(None)?.item();
        if min < 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit norm_out: adaln index {min} is negative; the timestep index cannot \
                 be recovered from it"
            )));
        }
        Ok(mlx_rs::ops::floor_divide(
            &idx,
            Array::from_int(MODALITY_NUM),
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
    /// `time_proj` + `time_embedder`. **float32**. `None` on a pruned topology, whose coordinates
    /// come straight from the `adaln_t_table`.
    pub time_embedder: Option<TimestepEmbedder>,
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

    /// Load all 17, at their published dtypes.
    pub fn from_weights(w: &mut Weights, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
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
            time_embedder: Some(TimestepEmbedder::from_weights(w, "time_embedder", cfg)?),
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

    /// Load the **pruned** surface: the five unchanged input/output projections and `norm_out.norm`,
    /// with `time_embedder` left `None` and `norm_out.linear` reduced. The table and reduced
    /// projections themselves are owned by [`crate::dit::pruned_adaln::PrunedAdaLnRuntime`].
    pub fn from_weights_pruned(w: &mut Weights, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
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
            time_embedder: None,
            norm_out: AdaLayerNormOut::from_weights_norm_only(w, "norm_out", cfg)?,
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
            + self
                .time_embedder
                .as_ref()
                .map_or(0, TimestepEmbedder::nbytes)
            + self.norm_out.nbytes()
            + self.proj_out.nbytes()
            + self.audio_proj_out.nbytes()
    }
}

/// MiniMax-H3-**Image**'s per-step output heads (`pdd_proj_out.{i}`): one
/// `hidden_size → in_channels · prod(patch)` projection per distilled sampling step, replacing
/// H3's single `proj_out`. **float32** in the published checkpoint (`_keep_in_fp32_modules`).
///
/// The checkpoint is distilled with Parallel Decoding Distillation, which keeps one head per step,
/// so the forward of step `i` must project with head `i`: [`Self::select`] is the only way to
/// reach a head, and it refuses an index outside the table instead of clamping it.
#[derive(Debug, Clone)]
pub struct PddHeads {
    heads: Vec<LinearBias>,
}

impl PddHeads {
    /// Load `pdd_proj_out.{0..count}`, each at its published dtype and shape-checked.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        count: i32,
        hidden_size: i32,
        patch_dim: i32,
    ) -> Result<Self> {
        if count <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3-image dit {prefix}: needs at least one head, got {count}"
            )));
        }
        let heads = (0..count)
            .map(|i| LinearBias::from_weights(w, &format!("{prefix}.{i}"), hidden_size, patch_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { heads })
    }

    /// The `2 · count` tensor names.
    pub fn names(prefix: &str, count: i32) -> Vec<String> {
        (0..count)
            .flat_map(|i| LinearBias::names(&format!("{prefix}.{i}")))
            .collect()
    }

    /// Heads in the table — the number of steps the sampler runs.
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Always `false` for a loaded table ([`Self::from_weights`] refuses zero heads).
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The head sampling step `step` projects with.
    /// Head `index` for an adapter install (SW-7096) — `pdd_proj_out.{index}`, one PEFT module per
    /// head as in the reference's `nn.ModuleList`.
    pub fn head_mut(&mut self, index: usize) -> Option<&mut LinearBias> {
        self.heads.get_mut(index)
    }

    pub fn select(&self, step: usize) -> Result<&LinearBias> {
        pdd_head_index(step, self.heads.len()).map(|i| &self.heads[i])
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.heads.iter().map(LinearBias::nbytes).sum()
    }
}

/// Which PDD head step `step` uses: head `step`, exactly — never a clamp or a wrap, because a step
/// past the table means the sampler and the checkpoint disagree on how many steps there are.
pub fn pdd_head_index(step: usize, heads: usize) -> Result<usize> {
    if step < heads {
        Ok(step)
    } else {
        Err(Error::Msg(format!(
            "minimax-h3-image dit: sampling step {step} has no output head; the checkpoint was \
             distilled to {heads} steps, one head each"
        )))
    }
}

/// **MiniMax-H3-Image's input/output tensors** — H3's [`DitProjections`] without the audio pair,
/// with [`PddHeads`] in place of the single `proj_out`. 11 + `2 · num_pdd_heads` tensors (23 at the
/// shipped six heads); with the 600 block and 21 refiner tensors that is the published 644.
///
/// The shared pieces are H3's own types — [`LinearBias`], [`TimestepEmbedder`],
/// [`AdaLayerNormOut`] — so the image family carries no copy of any layer, and H3's projection set
/// (which requires its audio heads at load) is untouched.
#[derive(Debug, Clone)]
pub struct ImageDitProjections {
    /// Patchified image rows → the residual stream. **float32**.
    pub proj_in: LinearBias,
    /// The text encoder's 5120-wide context → the residual stream. bfloat16.
    pub context_embedder: LinearBias,
    /// `time_proj` + `time_embedder`. **float32**.
    pub time_embedder: TimestepEmbedder,
    /// The packed sequence's final modulated norm. bfloat16.
    pub norm_out: AdaLayerNormOut,
    /// One output head per distilled sampling step. **float32**.
    pub pdd_heads: PddHeads,
}

impl ImageDitProjections {
    /// The checkpoint prefix of the per-step heads.
    pub const PDD_PREFIX: &'static str = "pdd_proj_out";

    /// Tensors besides the heads: `proj_in` 2, `context_embedder` 2, `time_embedder` 4,
    /// `norm_out` 3.
    pub const SHARED_TENSOR_COUNT: usize = 11;

    /// Load every input/output tensor at its published dtype.
    pub fn from_weights(
        w: &mut Weights,
        cfg: &crate::dit::config::MiniMaxH3ImageDitConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        let base = &cfg.base;
        Ok(Self {
            proj_in: LinearBias::from_weights(
                w,
                "proj_in",
                base.video_patch_dim(),
                base.hidden_size,
            )?,
            context_embedder: LinearBias::from_weights(
                w,
                "context_embedder",
                base.text_dim,
                base.hidden_size,
            )?,
            time_embedder: TimestepEmbedder::from_weights(w, "time_embedder", base)?,
            norm_out: AdaLayerNormOut::from_weights(w, "norm_out", base)?,
            pdd_heads: PddHeads::from_weights(
                w,
                Self::PDD_PREFIX,
                cfg.num_pdd_heads,
                base.hidden_size,
                base.video_patch_dim(),
            )?,
        })
    }

    /// Every tensor name these projections consume.
    pub fn names(cfg: &crate::dit::config::MiniMaxH3ImageDitConfig) -> Vec<String> {
        let mut v = LinearBias::names("proj_in").to_vec();
        v.extend(LinearBias::names("context_embedder"));
        v.extend(TimestepEmbedder::names("time_embedder"));
        v.extend(AdaLayerNormOut::names("norm_out"));
        v.extend(PddHeads::names(Self::PDD_PREFIX, cfg.num_pdd_heads));
        v
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.proj_in.nbytes()
            + self.context_embedder.nbytes()
            + self.time_embedder.nbytes()
            + self.norm_out.nbytes()
            + self.pdd_heads.nbytes()
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::dit::config::MiniMaxH3ImageDitConfig;

    /// 11 shared + 12 head tensors, no audio, and no H3 `proj_out`: with 50·12 block and 21 refiner
    /// tensors that is the 644 the published `transformer_diffusers` index carries.
    #[test]
    fn the_image_projection_names_are_the_published_top_level_set() {
        let cfg = MiniMaxH3ImageDitConfig::default();
        let names = ImageDitProjections::names(&cfg);
        assert_eq!(names.len(), ImageDitProjections::SHARED_TENSOR_COUNT + 12);
        assert_eq!(cfg.base.num_layers * 12 + 21 + names.len() as i32, 644);
        assert!(!names.iter().any(|n| n.contains("audio")), "{names:?}");
        assert!(
            !names.iter().any(|n| n.starts_with("proj_out")),
            "{names:?}"
        );
        for expect in [
            "pdd_proj_out.0.weight",
            "pdd_proj_out.5.bias",
            "norm_out.linear.weight",
            "time_embedder.linear_2.bias",
        ] {
            assert!(names.iter().any(|n| n == expect), "{expect} missing");
        }
    }

    /// Step `i` projects with head `i`; a step past the table is an error, not a clamp.
    #[test]
    fn step_i_selects_head_i_and_nothing_else() {
        for step in 0..6 {
            assert_eq!(pdd_head_index(step, 6).unwrap(), step);
        }
        let e = pdd_head_index(6, 6).unwrap_err().to_string();
        assert!(e.contains("6 steps"), "{e}");
    }

    /// Each head is a distinct tensor and `select` returns the one at its own index.
    #[test]
    fn select_returns_the_head_loaded_at_that_index() {
        let mut w = Weights::empty();
        for i in 0..3 {
            let v = i as f32 + 1.0;
            w.insert(
                format!("pdd_proj_out.{i}.weight"),
                Array::from_slice(&[v; 8], &[2, 4]),
            );
            w.insert(
                format!("pdd_proj_out.{i}.bias"),
                Array::from_slice(&[v; 2], &[2]),
            );
        }
        let heads = PddHeads::from_weights(&mut w, "pdd_proj_out", 3, 4, 2).unwrap();
        assert_eq!(heads.len(), 3);
        for i in 0..3 {
            let b: Vec<f32> = heads.select(i).unwrap().bias().as_slice::<f32>().to_vec();
            assert_eq!(b, vec![i as f32 + 1.0; 2]);
        }
        assert!(heads.select(3).is_err());
        assert!(PddHeads::from_weights(&mut w, "pdd_proj_out", 0, 4, 2).is_err());
        // A wrong head width is refused at load.
        assert!(PddHeads::from_weights(&mut w, "pdd_proj_out", 3, 4, 3).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(shape: &[i32], seed: u32) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n as u32)
            .map(|i| {
                ((i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 997) as f32 / 997.0
                    - 0.5
            })
            .collect();
        Array::from_slice(&v, shape)
    }

    /// SW-7092: exactly the float32 projections take the N=1 full-precision dispatch.
    #[test]
    fn only_float32_heads_take_the_precise_dispatch() {
        assert!(uses_precise_dispatch(Dtype::Float32));
        for dtype in [Dtype::Bfloat16, Dtype::Float16] {
            assert!(!uses_precise_dispatch(dtype), "{dtype:?}");
        }
    }

    /// SW-7092: a float32 head over many rows (the shape that NAX runs as a TF32 GEMM through
    /// `addmm`) matches an f64 host projection to F32 accuracy. TF32 keeps 10 fraction bits, so a
    /// TF32 product misses this by well over 1e-5 of the peak.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn a_float32_head_projects_at_full_float32_precision() {
        let (rows, inp, out) = (64, 256, 96);
        let w = seeded(&[out, inp], 11);
        let b = seeded(&[out], 12);
        let x = seeded(&[1, rows, inp], 13);
        let mut map = Weights::empty();
        map.insert("h.weight", w.clone());
        map.insert("h.bias", b.clone());
        let head = LinearBias::from_weights(&mut map, "h", inp, out).unwrap();
        let got = head.forward(&x).unwrap();
        assert_eq!(got.shape(), [1, rows, out]);
        assert_eq!(got.dtype(), Dtype::Float32);
        let (w, b, x, got) = (
            w.as_slice::<f32>().to_vec(),
            b.as_slice::<f32>().to_vec(),
            x.as_slice::<f32>().to_vec(),
            got.as_slice::<f32>().to_vec(),
        );
        let inp = inp as usize;
        let out = out as usize;
        let (mut peak, mut worst) = (0f64, 0f64);
        for r in 0..rows as usize {
            for o in 0..out {
                let want = f64::from(b[o])
                    + (0..inp)
                        .map(|i| f64::from(x[r * inp + i]) * f64::from(w[o * inp + i]))
                        .sum::<f64>();
                peak = peak.max(want.abs());
                worst = worst.max((f64::from(got[r * out + o]) - want).abs());
            }
        }
        assert!(
            worst / peak < 1e-5,
            "float32 head error {:.3e} of peak",
            worst / peak
        );
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED dit::heads::tests::a_float32_head_projects_at_full_float32_precision");
    }

    /// SW-7096: a LoRA on a bias-Linear head is `x·Wᵀ + b + scale·(alpha/rank)·x·Aᵀ·Bᵀ` — checked
    /// against the same projection with the delta MERGED into its weight, an independent route —
    /// and a zero-scale LoRA is byte-identical to the bare head.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn a_lora_on_a_bias_linear_head_matches_its_merged_weight() {
        let (inp, out, r) = (6, 5, 2);
        let w = seeded(&[out, inp], 1);
        let b = seeded(&[out], 2);
        let (down, up) = (seeded(&[r, inp], 3), seeded(&[out, r], 4));
        let scale = 0.75f32;
        let mut map = Weights::empty();
        map.insert("h.weight", w.clone());
        map.insert("h.bias", b.clone());
        let mut head = LinearBias::from_weights(&mut map, "h", inp, out).unwrap();
        let x = seeded(&[1, 3, inp], 5);
        let bare = head.forward(&x).unwrap();
        head.adaptable_mut()
            .unwrap()
            .push(mlx_gen::adapters::Adapter::Lora {
                a: down.t(),
                b: up.t(),
                scale,
            });
        assert_eq!(head.adapter_count(), 1);
        let got = head.forward(&x).unwrap();
        let delta = multiply(
            mlx_rs::ops::matmul(&up, &down).unwrap(),
            Array::from_f32(scale),
        )
        .unwrap();
        let mut merged_map = Weights::empty();
        merged_map.insert("h.weight", add(&w, &delta).unwrap());
        merged_map.insert("h.bias", b);
        let merged = LinearBias::from_weights(&mut merged_map, "h", inp, out).unwrap();
        let want = merged.forward(&x).unwrap();
        let drift = mlx_gen::testing::max_abs_diff_f32(&got, &want);
        // Metal's f32 GEMM accumulates in a different order for the two routes.
        assert!(drift < 1e-3, "merged drift {drift:e}");
        assert!(mlx_gen::testing::max_abs_diff_f32(&got, &bare) > 1e-2);

        let mut off = LinearBias::from_weights(&mut merged_map, "h", inp, out).unwrap();
        let before = off.forward(&x).unwrap();
        off.adaptable_mut()
            .unwrap()
            .push(mlx_gen::adapters::Adapter::Lora {
                a: down.t(),
                b: up.t(),
                scale: 0.0,
            });
        assert_eq!(
            mlx_gen::testing::max_abs_diff_f32(&off.forward(&x).unwrap(), &before),
            0.0
        );
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED dit::heads::tests::a_lora_on_a_bias_linear_head_matches_its_merged_weight");
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
            (cfg.num_layers * 12 + 21 + names.len() as i32),
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
        let emb = timestep_sincos(&[1.0], 4).unwrap();
        assert_eq!(emb.shape(), &[1, 4]);
        let v = emb.as_slice::<f32>();
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
        let emb = timestep_sincos(&[t], 8).unwrap();
        let v = emb.as_slice::<f32>();
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
        assert!(timestep_sincos(&[1.0], 5).is_err(), "odd freq_dim");
        assert!(timestep_sincos(&[], 8).is_err(), "no timesteps");
        assert!(timestep_sincos(&[f32::NAN], 8).is_err(), "non-finite");
    }

    /// The AdaLN index carries the modality; the `norm_out` index must not. The quotient recovers it
    /// exactly.
    #[test]
    fn the_norm_out_index_is_the_adaln_index_without_the_modality() {
        // timestep 0 tags 0/1/2, then timestep 3 tag 1, then timestep 2 tag 2.
        let adaln = Array::from_slice(
            &[0i32, 1, 2, 3 * MODALITY_NUM + 1, 2 * MODALITY_NUM + 2],
            &[5],
        );
        let t = AdaLayerNormOut::timestep_indices_from_adaln(&adaln).unwrap();
        assert_eq!(t.as_slice::<i32>(), &[0, 0, 0, 3, 2]);
        assert!(
            AdaLayerNormOut::timestep_indices_from_adaln(&Array::from_slice(&[-1i32], &[1]))
                .is_err()
        );
    }

    /// `shift` is the FIRST half and `scale` the second, and the scale is applied as `1 + scale`.
    /// Built over a projection whose two halves are trivially distinguishable, so a reversed
    /// reading cannot pass.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn norm_out_modulation_is_shift_then_scale() {
        // A 1-row temb of width 1 projected to [shift = 5, scale = 0.5] by an identity-ish linear.
        let mut w = Weights::empty();
        w.insert("n.norm.weight", Array::from_slice(&[1.0f32, 1.0], &[2]));
        // hidden_size 2 => out 4 = [shift(2) | scale(2)].
        w.insert(
            "n.linear.weight",
            Array::from_slice(&[5.0f32, 6.0, 0.5, 0.25], &[4, 1]),
        );
        w.insert(
            "n.linear.bias",
            Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4]),
        );
        let cfg = MiniMaxH3DitConfig {
            hidden_size: 2,
            time_embed_dim: 1,
            ..MiniMaxH3DitConfig::default()
        };
        let n = AdaLayerNormOut::from_weights(&mut w, "n", &cfg).unwrap();
        // silu(temb) at temb = a large value is ~= temb, but use the exact value below.
        let temb = Array::from_slice(&[2.0f32], &[1, 1]);
        let m = n.modulation(&temb).unwrap();
        let act = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert_eq!(m.shift.shape(), &[1, 2]);
        assert!((m.shift.as_slice::<f32>()[0] - 5.0 * act).abs() < 1e-5);
        assert!((m.scale.as_slice::<f32>()[0] - 0.5 * act).abs() < 1e-5);
        assert!(
            m.shift.as_slice::<f32>()[0] > 4.0 * m.scale.as_slice::<f32>()[0],
            "the two halves must not be interchangeable"
        );

        // An out-of-range timestep index is an error rather than a silent OOB gather.
        let x = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        n.apply(&x, &m, &Array::from_slice(&[0i32], &[1])).unwrap();
        assert!(n.apply(&x, &m, &Array::from_slice(&[1i32], &[1])).is_err());
        assert!(n.apply(&x, &m, &Array::from_slice(&[-1i32], &[1])).is_err());
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED dit::heads::tests::norm_out_modulation_is_shift_then_scale");
    }
}
