//! Timestep → conditioning embedding. Port of the fork's `QwenTimeTextEmbed`: sinusoidal
//! `time_proj` (256, scale 1000) → `timestep_embedder` (linear_1 → SiLU → linear_2). `[B] → [B, inner]`.

use mlx_rs::Array;

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;
use super::timesteps::timestep_proj;

const PROJ_DIM: i32 = 256;
const SCALE: f32 = 1000.0;

pub struct TimeTextEmbed {
    l1_w: Array,
    l1_b: Array,
    l2_w: Array,
    l2_b: Array,
}

impl TimeTextEmbed {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = join(prefix, "timestep_embedder");
        Ok(Self {
            l1_w: w.require(&join(&p, "linear_1.weight"))?.clone(),
            l1_b: w.require(&join(&p, "linear_1.bias"))?.clone(),
            l2_w: w.require(&join(&p, "linear_2.weight"))?.clone(),
            l2_b: w.require(&join(&p, "linear_2.bias"))?.clone(),
        })
    }

    /// `timestep`: `[B]` f32 → `[B, inner]`.
    pub fn forward(&self, timestep: &Array) -> Result<Array> {
        let proj = timestep_proj(timestep, PROJ_DIM, SCALE)?;
        let x = silu(&linear(&proj, &self.l1_w, &self.l1_b)?)?;
        linear(&x, &self.l2_w, &self.l2_b)
    }
}
