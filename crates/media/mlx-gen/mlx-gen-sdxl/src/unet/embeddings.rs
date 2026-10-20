//! UNet time/conditioning embeddings: the `SinusoidalPositionalEncoding` (used for both the
//! timestep and the SDXL `text_time` micro-conditioning) and the 2-layer `TimestepEmbedding` MLP.
//! Faithful ports of the vendored `unet.py` + mlx `nn.SinusoidalPositionalEncoding`.

use std::f32::consts::PI;

use mlx_rs::ops::{add, concatenate_axis, cos, exp, multiply, sin};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;

use crate::silu_glue;
use mlx_gen::Result;

/// Port of mlx `nn.SinusoidalPositionalEncoding`. Precomputes `sigmas` from `(dims, min_freq,
/// max_freq, full_turns)` and on call returns `concat([cos(x·σ), sin(x·σ)])` (when `cos_first`)
/// scaled by `scale`.
///
/// The `sigmas` table is computed with **MLX ops** (not host `f32::exp`) so it is bit-identical to
/// the reference `mx.exp(one_zero·(max−min)+min)` — a host-exp `sigmas` differs at the ULP level,
/// and that ~2e-4 timestep-embedding error chaotically amplifies under CFG=7 ancestral sampling
/// (sc-2400 S5). The `min_freq`/`max_freq` logs are taken in f64 (matching the reference's
/// `math.log`) before the f32 array math.
pub struct SinusoidalPositionalEncoding {
    sigmas: Array, // [dims/2]
    scale: f32,
    cos_first: bool,
}

impl SinusoidalPositionalEncoding {
    pub fn new(
        dims: i32,
        min_freq: f64,
        max_freq: f64,
        scale: f32,
        cos_first: bool,
        full_turns: bool,
    ) -> Result<Self> {
        let half = dims / 2;
        // one_zero = 1 - arange(half)/(half-1)  (f32 array, matching mx.arange int→float division).
        let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
        let arange = Array::from_slice(&arange, &[half]);
        let one_zero = mlx_rs::ops::subtract(
            scalar(1.0),
            &mlx_rs::ops::divide(&arange, scalar((half - 1) as f32))?,
        )?;
        // sigmas = exp(one_zero·(max_l − min_l) + min_l), logs in f64 then cast to f32 (as the
        // reference does: math.log is f64, the mlx scalar broadcast casts to f32).
        let min_l = min_freq.ln();
        let max_l = max_freq.ln();
        let scaled = add(
            &multiply(&one_zero, scalar((max_l - min_l) as f32))?,
            scalar(min_l as f32),
        )?;
        let mut sigmas = exp(&scaled)?;
        if full_turns {
            sigmas = multiply(&sigmas, scalar(2.0 * PI))?;
        }
        Ok(Self {
            sigmas,
            scale,
            cos_first,
        })
    }

    /// The SDXL timestep encoder: `SinusoidalPositionalEncoding(dim, min_freq=exp(-ln(10000) +
    /// 2·ln(10000)/dim), max_freq=1, scale=1, cos_first=true, full_turns=false)` — the vendored
    /// `UNetModel.timesteps` / `add_time_proj` construction.
    pub fn timestep(dim: i32) -> Result<Self> {
        let ln10000 = 10000f64.ln();
        let min_freq = (-ln10000 + 2.0 * ln10000 / dim as f64).exp();
        Self::new(dim, min_freq, 1.0, 1.0, true, false)
    }

    /// `x[..., None] * sigmas → concat([cos, sin], -1)` (cos-first) `· scale`. `x` is any shape;
    /// output appends a `dims` axis.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let axis = x.shape().len() as i32; // append a trailing axis
        let y = multiply(&x.expand_dims(axis)?, &self.sigmas)?;
        let cosy = cos(&y)?;
        let siny = sin(&y)?;
        let order: [&Array; 2] = if self.cos_first {
            [&cosy, &siny]
        } else {
            [&siny, &cosy]
        };
        let mut y = concatenate_axis(&order, -1)?;
        if self.scale != 1.0 {
            y = multiply(&y, scalar(self.scale))?;
        }
        Ok(y)
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval([&self.sigmas])?;
        Ok(())
    }

    /// The host bytes [`Self::timestep`]`(dims)` holds from construction (its lazy `sigmas` graph
    /// keeps the f32 `arange` table and its four f32 scalars): header-free pricing of the U-Net
    /// load (`loader::unet_load_parts`).
    pub(crate) fn timestep_table_bytes(dims: i32) -> u64 {
        4 * (dims / 2).max(0) as u64 + 4 * 4
    }
}

/// The 2-layer time-embedding MLP (`linear_1 → SiLU → linear_2`). Used both for the timestep
/// (`time_embedding`) and the SDXL added-conditioning (`add_embedding`).
pub struct TimestepEmbedding {
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
}

impl TimestepEmbedding {
    /// `prefix` is e.g. `time_embedding` or `add_embedding`; leaves are `linear_1`/`linear_2`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8746): both MLP Linears are quantized in [`Self::quantize`], so a
        // pre-quantized snapshot stores their packed triple — `crate::quant::lin` loads packed or
        // dense automatically.
        Ok(Self {
            linear1: crate::quant::lin(w, &format!("{prefix}.linear_1"), true)?,
            linear2: crate::quant::lin(w, &format!("{prefix}.linear_2"), true)?,
        })
    }

    pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut crate::quant::QuantizeLinear<'_>) -> Result<()> {
        visit(&crate::quant::child_path(prefix, "linear_1"), &mut self.linear1, bits)?;
        visit(&crate::quant::child_path(prefix, "linear_2"), &mut self.linear2, bits)?;
        Ok(())
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        self.linear1.materialize_weights()?;
        self.linear2.materialize_weights()
    }

    /// Cast both MLP Linears to `dtype` (sc-4941 bf16 training). The sinusoidal `sigmas` tables stay
    /// f32 (computed once, cast to the model dtype at the `text_time_temb` call site).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.linear1.cast_weights(dtype)?;
        self.linear2.cast_weights(dtype)?;
        Ok(())
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear1.forward(x)?;
        let x = silu_glue(&x)?;
        self.linear2.forward(&x)
    }

    pub(crate) fn output_dtype_with_effects(&self, x: Dtype, effects: &crate::quantum::EffectScope<'_>) -> Result<Dtype> {
        let first = effects.linear("linear_1", &self.linear1)?.output_dtype(x);
        let first = Dtype::from_promoting_types(x, first);
        Ok(Dtype::from_promoting_types(first,
            effects.linear("linear_2", &self.linear2)?.output_dtype(first)))
    }

    pub(crate) fn projection_scratch(&self, batch: i32, input: Dtype, effects: &crate::quantum::EffectScope<'_>) -> Result<u64> {
        let first = effects.linear("linear_1", &self.linear1)?;
        let second = effects.linear("linear_2", &self.linear2)?;
        let shape = [batch, *first.base.base_shape.last().unwrap_or(&0)];
        let hidden = first.output_dtype(input);
        let intermediate = [batch, first.output_width() as i32];
        let output = second.output_dtype(hidden);
        let plane = |dtype, width| mlx_gen::ios_quantum::tensor_bytes(
            crate::quantum::dtype_bytes(dtype), &[batch.max(0) as u64, width]);
        Ok(first.operand_bytes(&shape, input)
            .saturating_add(plane(hidden, first.output_width()).saturating_mul(3))
            .saturating_add(second.operand_bytes(&intermediate, hidden))
            .saturating_add(plane(output, second.output_width()))
            .saturating_add(crate::quantum::page_slack(4)))
    }

    pub(crate) fn out_features_with_effects(&self, effects: &crate::quantum::EffectScope<'_>) -> Result<i32> {
        Ok(effects.linear("linear_2", &self.linear2)?.output_width() as i32)
    }

}

/// Pure final projection facts for the existing time/add embedding flow.
pub(crate) fn text_time_temb_dtype_with_effects(
    time_embedding: &TimestepEmbedding,
    add_embedding: &TimestepEmbedding,
    text_emb_dtype: Dtype,
    dtype: Dtype,
    effects: &crate::quantum::EffectScope<'_>,
) -> Result<Dtype> {
    Ok(Dtype::from_promoting_types(
        time_embedding.output_dtype_with_effects(dtype, &effects.child("time_embedding"))?,
        add_embedding.output_dtype_with_effects(Dtype::from_promoting_types(text_emb_dtype, dtype),
            &effects.child("add_embedding"))?,
    ))
}

/// The actual stem's time/add MLP and optional context projection operands. Shared by the
/// runtime guard and header overlay; no output graph is built to derive these extents.
pub(crate) fn stem_projection_scratch(
    time: &TimestepEmbedding, add: &TimestepEmbedding, encoder: Option<&AdaptableLinear>,
    batch: i32, dtype: Dtype, text: Dtype, memory: &Array,
    effects: &crate::quantum::EffectScope<'_>,
) -> Result<u64> {
    let mut bytes = time.projection_scratch(batch, dtype, &effects.child("time_embedding"))?
        .saturating_add(add.projection_scratch(batch, Dtype::from_promoting_types(text, dtype),
            &effects.child("add_embedding"))?);
    if let Some(encoder) = encoder {
        let work = effects.linear("encoder_hid_proj", encoder)?;
        bytes = bytes.saturating_add(work.operand_bytes(memory.shape(), memory.dtype()))
            .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
                crate::quantum::dtype_bytes(work.output_dtype(memory.dtype())),
                &[memory.shape()[0].max(0) as u64, memory.shape()[1].max(0) as u64, work.output_width()]));
    }
    Ok(bytes)
}

/// The SDXL timestep + `text_time` micro-conditioning embedding, shared verbatim by
/// [`UNet2DConditionModel::forward_core`](crate::unet::UNet2DConditionModel) and
/// [`ControlNet::forward`](crate::unet::ControlNet) — the encoder-copy contract requires the two to
/// stay bit-identical, so the sequence lives in one place (F-070). Returns the summed `temb`.
///
/// Sequence: broadcast the scalar `timestep` to the batch → sinusoidal `timesteps` (f32) → cast to
/// `dtype` → `time_embedding` MLP; in parallel the `time_ids` sinusoidal (`add_time_proj`, f32) is
/// flattened, cast to `dtype`, concatenated with the (model-dtype) pooled `text_emb`, and run
/// through `add_embedding`; the two are summed.
#[allow(clippy::too_many_arguments)]
pub fn text_time_temb(
    timesteps: &SinusoidalPositionalEncoding,
    time_embedding: &TimestepEmbedding,
    add_time_proj: &SinusoidalPositionalEncoding,
    add_embedding: &TimestepEmbedding,
    timestep: f32,
    text_emb: &Array,
    time_ids: &Array,
    batch: i32,
    dtype: Dtype,
) -> Result<Array> {
    // Timestep embedding (broadcast the scalar time to the batch). The sinusoidal encoding runs in
    // f32 (its `sigmas` table is f32), then the reference casts to the model dtype *before* the
    // `time_embedding` MLP (`temb = self.timesteps(t).astype(x.dtype)`), so the MLP runs in the model
    // dtype. The cast is a no-op for the f32 path.
    let t = Array::from_slice(&vec![timestep; batch as usize], &[batch]);
    let temb = timesteps.forward(&t)?.as_dtype(dtype)?;
    let temb = time_embedding.forward(&temb)?;

    // SDXL `text_time` added conditioning: concat(pooled_text, flattened sinusoidal time_ids).
    // `time_ids` stays f32 through its sinusoidal (the reference builds it f32), then the flattened
    // result is cast to the model dtype before concat with the (model-dtype) pooled text
    // (`...flatten(1).astype(x.dtype)`).
    let emb = add_time_proj.forward(time_ids)?; // [B, 6, 256]
    let es = emb.shape();
    let emb = emb.reshape(&[es[0], es[1] * es[2]])?.as_dtype(dtype)?; // flatten(1) → [B, 1536]
    let emb = concatenate_axis(&[text_emb, &emb], -1)?; // [B, 2816]
    let emb = add_embedding.forward(&emb)?;
    Ok(add(&temb, &emb)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_shape_and_cos_first() {
        // dims=8 -> output appends an 8-wide axis; cos-first means first half is cos.
        let enc = SinusoidalPositionalEncoding::timestep(8).unwrap();
        let x = Array::from_slice(&[0.0f32, 1.0], &[2]);
        let y = enc.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
        // At x=0: cos(0)=1 for the first half, sin(0)=0 for the second.
        let row0 = y.reshape(&[2, 8]).unwrap();
        let s = row0.as_slice::<f32>();
        for v in &s[0..4] {
            assert!((v - 1.0).abs() < 1e-6, "cos(0) half should be 1");
        }
        for v in &s[4..8] {
            assert!(v.abs() < 1e-6, "sin(0) half should be 0");
        }
    }
}
