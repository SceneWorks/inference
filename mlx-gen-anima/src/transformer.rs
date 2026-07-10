//! Cosmos-Predict2 DiT — the Anima image transformer, transcribed from diffusers
//! `transformer_cosmos.py::CosmosTransformer3DModel` (+ the `Cosmos-2.0-Diffusion-2B-Text2Image`
//! config). Weight keys are the **original Cosmos** names (`{prefix}.blocks.N.*`,
//! `{prefix}.x_embedder.proj.1`, `{prefix}.t_embedder.1.*`, `{prefix}.final_layer.*`) — the single-file
//! bf16 checkpoint loads unchanged, no diffusers rename applied. `prefix` is detected per file (`net`
//! for the base cut, `model.diffusion_model` for turbo/aesthetic; see [`crate::loader`]).
//!
//! Ported pieces: `CosmosPatchEmbed`, `CosmosTimestepEmbedding`/`CosmosEmbedding`,
//! `CosmosAdaLayerNorm(Zero)`, `CosmosAttention` (q/k RMSNorm + half-split RoPE on self-attn),
//! `CosmosTransformerBlock`, final layer. **Skipped** (config-off for Anima): learnable pos-embed
//! (`extra_pos_embed_type=null`), cross-attn projection, ControlNet hooks, img-context.

use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, zeros_dtype};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{apply_text_rope, gelu_exact, modulate, silu, timestep_sincos};
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;

use crate::config::DitConfig;
use crate::rope::cosmos_image_rope;

/// q/k RMSNorm eps in diffusers `Attention` (`qk_norm="rms_norm"`, default `eps=1e-5`).
const ATTN_QK_NORM_EPS: f32 = 1e-5;
/// LayerNorm / time-embed-norm eps (`elementwise_affine=false, eps=1e-6`).
const NORM_EPS: f32 = 1e-6;
/// Sinusoidal timestep-embedding `max_period` (`Timesteps` default).
const TIME_MAX_PERIOD: f64 = 10000.0;

/// Dense linear (no bias) — adapter-ready (`AdaptableLinear`); auto-detects a packed Q4/Q8 base.
fn lin(w: &Weights, name: &str) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, name, false, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// `x[:, :end]` along axis 1.
fn head_cols(x: &Array, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (0..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end]), 1)?)
}

/// `CosmosEmbedding`: sinusoidal time_proj → `CosmosTimestepEmbedding` (`temb`, 3·hidden) +
/// `RMSNorm` (`embedded_timestep`, hidden).
struct TimeEmbed {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
    norm: Array,
    hidden: usize,
}

impl TimeEmbed {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: lin(w, &join(prefix, "t_embedder.1.linear_1"))?,
            linear_2: lin(w, &join(prefix, "t_embedder.1.linear_2"))?,
            norm: w.require(&join(prefix, "t_embedding_norm.weight"))?.clone(),
            hidden: cfg.hidden_size(),
        })
    }

    /// `sigma`: `[B]`. Returns `(temb [B, 3·hidden], embedded [B, hidden])` in `dtype`.
    fn forward(&self, sigma: &Array, dtype: Dtype) -> Result<(Array, Array)> {
        let proj = timestep_sincos(sigma, self.hidden, TIME_MAX_PERIOD, 0.0)?.as_dtype(dtype)?;
        let temb = self
            .linear_2
            .forward(&silu(&self.linear_1.forward(&proj)?)?)?;
        let embedded = rms_norm(&proj, &self.norm, NORM_EPS)?;
        Ok((temb, embedded))
    }
}

/// `CosmosAdaLayerNormZero` (norm1/2/3): LayerNorm(no affine) then `(1+scale)·norm + shift`, plus a
/// `gate`. `linear_2` emits `3·hidden` (shift|scale|gate), added to `temb`.
struct AdaLayerNormZero {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
}

impl AdaLayerNormZero {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // e.g. prefix = "net.blocks.0.adaln_modulation_self_attn"
        Ok(Self {
            linear_1: lin(w, &join(prefix, "1"))?,
            linear_2: lin(w, &join(prefix, "2"))?,
        })
    }

    /// Returns `(modulated_norm [B,S,H], gate [B,1,H])`.
    fn forward(&self, hidden: &Array, embedded: &Array, temb: &Array) -> Result<(Array, Array)> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&silu(embedded)?)?)?;
        let e = add(&e, temb)?; // [B, 3H]
        let parts = split(&e, 3, 1)?; // shift, scale, gate
        let shift = parts[0].expand_dims(1)?;
        let scale = parts[1].expand_dims(1)?;
        let gate = parts[2].expand_dims(1)?;
        let normed = layer_norm(hidden, None, None, NORM_EPS)?;
        Ok((modulate(&normed, &scale, &shift, true)?, gate))
    }
}

/// `CosmosAdaLayerNorm` (final `norm_out`): LayerNorm(no affine) then `(1+scale)·norm + shift`.
/// `linear_2` emits `2·hidden` (shift|scale), added to `temb[..., :2·hidden]`.
struct AdaLayerNorm {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
    hidden: i32,
}

impl AdaLayerNorm {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            linear_1: lin(w, &join(prefix, "1"))?,
            linear_2: lin(w, &join(prefix, "2"))?,
            hidden: cfg.hidden_size() as i32,
        })
    }

    fn forward(&self, hidden: &Array, embedded: &Array, temb: &Array) -> Result<Array> {
        let e = self
            .linear_2
            .forward(&self.linear_1.forward(&silu(embedded)?)?)?;
        let e = add(&e, &head_cols(temb, 2 * self.hidden)?)?; // + temb[:, :2H]
        let parts = split(&e, 2, 1)?; // shift, scale
        let shift = parts[0].expand_dims(1)?;
        let scale = parts[1].expand_dims(1)?;
        let normed = layer_norm(hidden, None, None, NORM_EPS)?;
        modulate(&normed, &scale, &shift, true)
    }
}

/// `CosmosAttention` — self (attn1: q/k/v from hidden, RoPE) or cross (attn2: q from hidden, k/v from
/// text, no RoPE). Per-head q/k RMSNorm; heads == kv_heads (no GQA repeat for Anima).
struct Attention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        let hd = cfg.attention_head_dim as i32;
        Ok(Self {
            to_q: lin(w, &join(prefix, "q_proj"))?,
            to_k: lin(w, &join(prefix, "k_proj"))?,
            to_v: lin(w, &join(prefix, "v_proj"))?,
            to_out: lin(w, &join(prefix, "output_proj"))?,
            norm_q: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: hd,
            scale: (hd as f32).powf(-0.5),
        })
    }

    /// `hidden`: `[B, Sq, H]`. `encoder`: `Some([B, Sk, Ctx])` for cross-attn (else self-attn on
    /// `hidden`). `rope`: `Some((cos,sin))` applies half-split RoPE (self-attn only).
    fn forward(
        &self,
        hidden: &Array,
        encoder: Option<&Array>,
        rope: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let hsh = hidden.shape();
        let (b, sq) = (hsh[0], hsh[1]);
        let kv_src = encoder.unwrap_or(hidden);
        let sk = kv_src.shape()[1];

        let q = self
            .to_q
            .forward(hidden)?
            .reshape(&[b, sq, self.heads, self.head_dim])?;
        let k = self
            .to_k
            .forward(kv_src)?
            .reshape(&[b, sk, self.heads, self.head_dim])?;
        let v = self
            .to_v
            .forward(kv_src)?
            .reshape(&[b, sk, self.heads, self.head_dim])?;

        // per-head q/k RMSNorm (over the head_dim).
        let q = rms_norm(&q, &self.norm_q, ATTN_QK_NORM_EPS)?;
        let k = rms_norm(&k, &self.norm_k, ATTN_QK_NORM_EPS)?;

        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_text_rope(&q, cos, sin)?,
                apply_text_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        // [b,s,h,hd] -> [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, sq, self.heads * self.head_dim])?;
        self.to_out.forward(&o)
    }
}

/// `FeedForward(mult=4, activation="gelu")` — `net.2(gelu_exact(net.0.proj(x)))`, no bias.
struct FeedForward {
    proj_in: AdaptableLinear,  // mlp.layer1 -> ff.net.0.proj
    proj_out: AdaptableLinear, // mlp.layer2 -> ff.net.2
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj_in: lin(w, &join(prefix, "layer1"))?,
            proj_out: lin(w, &join(prefix, "layer2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out
            .forward(&gelu_exact(&self.proj_in.forward(x)?)?)
    }
}

/// `CosmosTransformerBlock`: gated self-attn → gated cross-attn → gated FF.
struct Block {
    norm1: AdaLayerNormZero,
    attn1: Attention,
    norm2: AdaLayerNormZero,
    attn2: Attention,
    norm3: AdaLayerNormZero,
    ff: FeedForward,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_self_attn"))?,
            attn1: Attention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            norm2: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_cross_attn"))?,
            attn2: Attention::from_weights(w, &join(prefix, "cross_attn"), cfg)?,
            norm3: AdaLayerNormZero::from_weights(w, &join(prefix, "adaln_modulation_mlp"))?,
            ff: FeedForward::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        embedded: &Array,
        temb: &Array,
        rope: (&Array, &Array),
    ) -> Result<Array> {
        // 1. self attention (RoPE)
        let (normed, gate) = self.norm1.forward(hidden, embedded, temb)?;
        let attn = self.attn1.forward(&normed, None, Some(rope))?;
        let hidden = add(hidden, &multiply(&gate, &attn)?)?;
        // 2. cross attention (no RoPE)
        let (normed, gate) = self.norm2.forward(&hidden, embedded, temb)?;
        let attn = self.attn2.forward(&normed, Some(encoder), None)?;
        let hidden = add(&hidden, &multiply(&gate, &attn)?)?;
        // 3. feed forward
        let (normed, gate) = self.norm3.forward(&hidden, embedded, temb)?;
        let ff = self.ff.forward(&normed)?;
        Ok(add(&hidden, &multiply(&gate, &ff)?)?)
    }
}

/// The full Cosmos-Predict2 DiT.
pub struct CosmosDiT {
    patch_embed: AdaptableLinear, // x_embedder.proj.1
    time_embed: TimeEmbed,
    blocks: Vec<Block>,
    norm_out: AdaLayerNorm,
    proj_out: AdaptableLinear, // final_layer.linear
    cfg: DitConfig,
}

impl CosmosDiT {
    /// `prefix` is the checkpoint's DiT root (`"net"`); keys are the original Cosmos names.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: DitConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::from_weights(
                w,
                &join(prefix, &format!("blocks.{i}")),
                &cfg,
            )?);
        }
        Ok(Self {
            patch_embed: lin(w, &join(prefix, "x_embedder.proj.1"))?,
            time_embed: TimeEmbed::from_weights(w, prefix, &cfg)?,
            blocks,
            norm_out: AdaLayerNorm::from_weights(
                w,
                &join(prefix, "final_layer.adaln_modulation"),
                &cfg,
            )?,
            proj_out: lin(w, &join(prefix, "final_layer.linear"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// Patchify a `[B, C, 1, Hl, Wl]` latent (`C=17` after mask concat) to `[B, seq, C·ph·pw]`.
    fn patchify(&self, x: &Array) -> Result<Array> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let sh = x.shape();
        let (b, c, t, hl, wl) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
        // reshape (B, C, T/pt, pt, Hl/ph, ph, Wl/pw, pw)
        let x = x.reshape(&[b, c, t / pt, pt, hl / ph, ph, wl / pw, pw])?;
        // permute (0,2,4,6,1,3,5,7) -> (B, T/pt, Hl/ph, Wl/pw, C, pt, ph, pw)
        let x = x.transpose_axes(&[0, 2, 4, 6, 1, 3, 5, 7])?;
        // flatten patch dims -> (B, seq, C*pt*ph*pw)
        let seq = (t / pt) * (hl / ph) * (wl / pw);
        Ok(x.reshape(&[b, seq, c * pt * ph * pw])?)
    }

    /// Inverse of [`patchify`] on the `proj_out` output `[B, seq, ph·pw·pt·out_ch]` → `[B, out_ch, 1, Hl, Wl]`.
    fn unpatchify(&self, x: &Array, pe_t: i32, pe_h: i32, pe_w: i32) -> Result<Array> {
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
        let oc = self.cfg.out_channels as i32;
        let b = x.shape()[0];
        // [B, seq, ph*pw*pt*oc] -> [B, pe_t, pe_h, pe_w, ph, pw, pt, oc]
        let x = x.reshape(&[b, pe_t, pe_h, pe_w, ph, pw, pt, oc])?;
        // permute (0,7,1,6,2,4,3,5) -> [B, oc, pe_t, pt, pe_h, ph, pe_w, pw]
        let x = x.transpose_axes(&[0, 7, 1, 6, 2, 4, 3, 5])?;
        // collapse the patch pairs -> [B, oc, pe_t*pt, pe_h*ph, pe_w*pw]
        Ok(x.reshape(&[b, oc, pe_t * pt, pe_h * ph, pe_w * pw])?)
    }

    /// Denoise forward. `latents`: `[B, 16, 1, Hl, Wl]` (any dtype — cast to `dtype`). `sigma`: `[B]`.
    /// `encoder`: `[B, 512, text_embed_dim]`. Returns the velocity `[B, 16, 1, Hl, Wl]` in `dtype`.
    pub fn forward(
        &self,
        latents: &Array,
        sigma: &Array,
        encoder: &Array,
        dtype: Dtype,
    ) -> Result<Array> {
        let latents = latents.as_dtype(dtype)?;
        let sh = latents.shape();
        let (b, _c, t, hl, wl) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (pt, ph, pw) = self.cfg.patch_size;
        let (pe_t, pe_h, pe_w) = (t / pt as i32, hl / ph as i32, wl / pw as i32);

        // 1. concat the (all-zeros, full-res-resized-to-latent) padding mask channel -> [B,17,1,Hl,Wl].
        let hidden = if self.cfg.concat_padding_mask {
            let pad = zeros_dtype(&[b, 1, t, hl, wl], dtype)?;
            concatenate_axis(&[&latents, &pad], 1)?
        } else {
            latents
        };

        // 2. RoPE for this latent grid (per-axis OOD-guarded).
        let rope = cosmos_image_rope(&self.cfg, pe_t as usize, pe_h as usize, pe_w as usize)?;

        // 3. patchify + patch-embed -> [B, seq, hidden].
        let hidden = self.patch_embed.forward(&self.patchify(&hidden)?)?;

        // 4. time embedding.
        let (temb, embedded) = self.time_embed.forward(sigma, dtype)?;

        // 5. transformer blocks.
        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block.forward(&hidden, encoder, &embedded, &temb, (&rope.cos, &rope.sin))?;
        }

        // 6. output norm + projection + unpatchify.
        let hidden = self.norm_out.forward(&hidden, &embedded, &temb)?;
        let hidden = self.proj_out.forward(&hidden)?;
        self.unpatchify(&hidden, pe_t, pe_h, pe_w)
    }
}
