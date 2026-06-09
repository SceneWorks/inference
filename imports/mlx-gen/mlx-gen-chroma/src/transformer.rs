//! The Chroma DiT (`ChromaTransformer2DModel`).
//!
//! **sc-3836 (this slice):** the distilled-guidance modulation generator — the
//! `ChromaCombinedTimestepTextProjEmbeddings` (timestep + zero-guidance sinusoids + the `mod_proj`
//! index buffer) and the `ChromaApproximator` (`in_proj → 5× SiLU residual block with RMSNorm →
//! out_proj`) that together produce `pooled_temb [B, mod_index_len, inner_dim]`. The double/single
//! blocks, pruned-adaLN slicing, RoPE, MMDiT masking, and the forward pass land in sc-3837.

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{concatenate_axis, cos, exp, multiply, sin};
use mlx_rs::{Array, Dtype};

use crate::config::ChromaTransformerConfig;

/// Sinusoid frequency base (diffusers `get_timestep_embedding` `max_period`).
const MAX_PERIOD: f64 = 10000.0;
/// RMSNorm epsilon for the Approximator norms. The reference is torch `nn.RMSNorm(hidden)` with the
/// default `eps=None`, which `F.rms_norm` resolves to `torch.finfo(float32).eps` — the f32 machine
/// epsilon. The Approximator runs in f32 (its input is the f32 sinusoid `input_vec`).
const APPROX_RMS_EPS: f32 = 1.192_092_9e-7;

/// `get_timestep_embedding(timesteps, dim, flip_sin_to_cos, downscale_freq_shift)` (diffusers),
/// in f32. `dim` is assumed even (Chroma uses 32). `flip_sin_to_cos=True` ⇒ output order `[cos, sin]`.
fn timestep_embedding(timesteps: &Array, dim: usize, downscale_freq_shift: f64) -> Result<Array> {
    let half = (dim / 2) as i32;
    // exponent = -ln(max_period) * arange(half) / (half - downscale)
    let factor = -MAX_PERIOD.ln() / (half as f64 - downscale_freq_shift);
    let exponent: Vec<f32> = (0..half).map(|i| (i as f64 * factor) as f32).collect();
    let freqs = exp(Array::from_slice(&exponent, &[1, half]))?; // [1, half]
    let t = timesteps.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?; // [N, 1]
    let emb = multiply(&t, &freqs)?; // [N, half]
                                     // flip_sin_to_cos: [cos, sin]
    Ok(concatenate_axis(&[cos(&emb)?, sin(&emb)?], -1)?)
}

/// A dense `nn.Linear` (`[out, in]` weight + bias), PyTorch convention.
struct Lin {
    w: Array,
    b: Array,
}

impl Lin {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        linear(x, &self.w, &self.b)
    }
}

/// `ChromaCombinedTimestepTextProjEmbeddings` — builds the Approximator input vector. Parameter-free
/// (the `mod_proj` index sinusoid is a constant buffer); only the config matters.
struct TimestepTextProj {
    /// `approximator_num_channels // 4` (the per-projection sinusoid width; 16 for the real model).
    num_channels: usize,
    /// Precomputed `mod_proj` = `get_timestep_embedding(arange(out_dim)*1000, 2*num_channels)` —
    /// shape `[mod_index_len, 2*num_channels]`, f32.
    mod_proj: Array,
}

impl TimestepTextProj {
    fn new(cfg: &ChromaTransformerConfig) -> Result<Self> {
        let num_channels = cfg.approximator_num_channels / 4;
        let n = cfg.mod_index_len();
        let idx: Vec<f32> = (0..n).map(|i| (i as f32) * 1000.0).collect();
        let idx = Array::from_slice(&idx, &[n as i32]);
        let mod_proj = timestep_embedding(&idx, 2 * num_channels, 0.0)?;
        Ok(Self {
            num_channels,
            mod_proj,
        })
    }

    /// `timestep` is the **already-scaled** denoise timestep (the transformer forward passes `t*1000`),
    /// shape `[B]`. Returns `input_vec [B, mod_index_len, 4*num_channels]` (f32).
    fn forward(&self, timestep: &Array) -> Result<Array> {
        let b = timestep.shape()[0];
        let n = self.mod_proj.shape()[0];
        let time = timestep_embedding(timestep, self.num_channels, 0.0)?; // [B, nc]
                                                                          // guidance is always projected from 0 (Chroma has no guidance-distillation input).
        let zeros = Array::from_slice(&vec![0.0_f32; b as usize], &[b]);
        let guid = timestep_embedding(&zeros, self.num_channels, 0.0)?; // [B, nc]
        let tg = concatenate_axis(&[time, guid], -1)?; // [B, 2*nc]
        let tg = tg.reshape(&[b, 1, 2 * self.num_channels as i32])?;
        let tg = mlx_rs::ops::broadcast_to(&tg, &[b, n, 2 * self.num_channels as i32])?;
        let mp = self
            .mod_proj
            .reshape(&[1, n, 2 * self.num_channels as i32])?;
        let mp = mlx_rs::ops::broadcast_to(&mp, &[b, n, 2 * self.num_channels as i32])?;
        Ok(concatenate_axis(&[tg, mp], -1)?) // [B, n, 4*nc]
    }
}

/// `ChromaApproximator` — `in_proj` then `n_layers` residual blocks
/// `x = x + linear_2(silu(linear_1(rms_norm(x))))`, then `out_proj`.
struct Approximator {
    in_proj: Lin,
    layers: Vec<(Lin, Lin)>,
    norms: Vec<Array>,
    out_proj: Lin,
}

impl Approximator {
    fn load(w: &Weights, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = "distilled_guidance_layer";
        let mut layers = Vec::with_capacity(cfg.approximator_layers);
        let mut norms = Vec::with_capacity(cfg.approximator_layers);
        for i in 0..cfg.approximator_layers {
            layers.push((
                Lin::load(w, &format!("{p}.layers.{i}.linear_1"))?,
                Lin::load(w, &format!("{p}.layers.{i}.linear_2"))?,
            ));
            norms.push(w.require(&format!("{p}.norms.{i}.weight"))?.clone());
        }
        Ok(Self {
            in_proj: Lin::load(w, &format!("{p}.in_proj"))?,
            layers,
            norms,
            out_proj: Lin::load(w, &format!("{p}.out_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.in_proj.forward(x)?;
        for ((lin1, lin2), norm) in self.layers.iter().zip(self.norms.iter()) {
            let n = rms_norm(&x, norm, APPROX_RMS_EPS)?;
            let h = lin2.forward(&silu(&lin1.forward(&n)?)?)?;
            x = mlx_rs::ops::add(&x, &h)?;
        }
        self.out_proj.forward(&x)
    }
}

/// The Chroma transformer. The Approximator + embeddings are typed (sc-3836); the block stacks still
/// live in the raw `weights` map until sc-3837 materializes them.
pub struct ChromaTransformer {
    pub cfg: ChromaTransformerConfig,
    time_text_embed: TimestepTextProj,
    approximator: Approximator,
    /// Diffusers-named weights for the not-yet-typed modules (`x_embedder`, `context_embedder`, the
    /// block stacks, `norm_out`/`proj_out`). Materialized in sc-3837.
    pub weights: Weights,
}

impl ChromaTransformer {
    /// Load from a diffusers `transformer/` weight map. Validates the Chroma key surface (a structural
    /// sanity check) and the pruned-adaLN invariant before building the typed modules.
    pub fn from_weights(w: Weights, cfg: ChromaTransformerConfig) -> Result<Self> {
        let required = [
            "x_embedder.weight",
            "context_embedder.weight",
            "distilled_guidance_layer.in_proj.weight",
            "distilled_guidance_layer.out_proj.weight",
            "distilled_guidance_layer.layers.0.linear_1.weight",
            "distilled_guidance_layer.norms.0.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "single_transformer_blocks.0.proj_mlp.weight",
            "single_transformer_blocks.0.proj_out.weight",
            "proj_out.weight",
        ];
        for k in required {
            if w.get(k).is_none() {
                return Err(Error::Msg(format!(
                    "chroma transformer: missing expected key {k:?} (not a ChromaTransformer2DModel \
                     diffusers layout?)"
                )));
            }
        }

        // Pruned-adaLN invariant: Chroma blocks have NO `.norm*.linear` weights — modulation is sliced
        // out of the Approximator output, not projected per block.
        if let Some(k) = w
            .keys()
            .find(|k| k.contains(".norm1.linear") || k.contains(".norm.linear"))
        {
            return Err(Error::Msg(format!(
                "chroma transformer: unexpected per-block modulation linear {k:?} — Chroma uses \
                 pruned adaLN (modulation comes from distilled_guidance_layer)"
            )));
        }

        // Block-count sanity against the config.
        let n_double = (0..)
            .take_while(|i| {
                w.get(&format!("transformer_blocks.{i}.attn.to_q.weight"))
                    .is_some()
            })
            .count();
        let n_single = (0..)
            .take_while(|i| {
                w.get(&format!("single_transformer_blocks.{i}.proj_out.weight"))
                    .is_some()
            })
            .count();
        if n_double != cfg.num_layers || n_single != cfg.num_single_layers {
            return Err(Error::Msg(format!(
                "chroma transformer: block counts {n_double} double / {n_single} single != config \
                 {} / {}",
                cfg.num_layers, cfg.num_single_layers
            )));
        }

        let time_text_embed = TimestepTextProj::new(&cfg)?;
        let approximator = Approximator::load(&w, &cfg)?;
        Ok(Self {
            cfg,
            time_text_embed,
            approximator,
            weights: w,
        })
    }

    /// The distilled-guidance modulation tensor `pooled_temb [B, mod_index_len, inner_dim]` for a
    /// **raw** (unscaled) denoise timestep `[B]`. Mirrors the transformer forward's `timestep*1000`
    /// scaling before the embedding. Sliced per-block by the forward pass (sc-3837).
    pub fn pooled_temb(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        let input_vec = self.time_text_embed.forward(&scaled)?;
        self.approximator.forward(&input_vec)
    }

    /// Test hook: the Approximator input vector (sinusoid `time/guidance/mod_proj` build) for a raw
    /// timestep `[B]`, before the MLP. Pure elementwise — isolates the embedding from the matmul floor.
    #[doc(hidden)]
    pub fn input_vec_for_tests(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        self.time_text_embed.forward(&scaled)
    }
}
