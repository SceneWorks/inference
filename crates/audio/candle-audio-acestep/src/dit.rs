//! The ACE-Step 1.5 **`AceStepTransformer1DModel`** DiT (sc-12842) — a faithful port of the
//! diffusers v0.39.0 reference forward (`ace_step_transformer.py`). ~2B parameters: a Qwen3-derived
//! backbone (GQA, half-split RoPE, RMSNorm) with AdaLN-Zero timestep conditioning and
//! cross-attention to the packed condition-encoder context.
//!
//! ## Forward (per the reference)
//!
//! ```text
//!   x = cat([context_latents, hidden_states], dim=-1)           # [B, T, in_channels=192]
//!   x = pad_to(patch_size) ; x = proj_in_conv(xᵀ)ᵀ             # Conv1d k=s=patch → [B, T/2, dim]
//!   temb, tproj   = time_embed(t)   ⊕  time_embed_r(t − t_r)    # AdaLN-Zero conditioning
//!   ctx = condition_embedder(encoder_hidden_states)             # Linear 2048 → dim
//!   for block: x = AceStepTransformerBlock(x, tproj, ctx, rope, mask)
//!   shift,scale = (scale_shift_table + temb).chunk(2)
//!   x = norm_out(x)·(1+scale) + shift
//!   v = proj_out_conv(xᵀ)ᵀ  cropped to the original T                 # ConvTranspose1d → [B, T, 64]
//! ```
//!
//! ## Fidelity note (sc-12842)
//!
//! Weight names below are the diffusers export names, read directly. The sliding-window mask
//! geometry (symmetric band of `sliding_window`) and the RoPE unbind convention (half-split, the
//! `use_real_unbind_dim=-2` branch → candle `rope`) match the reference reading; these are the two
//! points that most need reference-activation validation before the acoustic output is certified
//! bit-faithful (see the crate docs and the real-weight conformance test).

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, linear, linear_b, rms_norm, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::TransformerConfig;

/// `x·(1+scale) + shift` with `[B, 1, dim]` modulation broadcast over the sequence.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> CandleResult<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// Sinusoidal timestep embedding with `flip_sin_to_cos=True` (cos first, then sin) over `dim`
/// lanes for a scalar timestep `t`. ACE-Step passes `t ∈ [0, 1]`.
fn sinusoidal(dim: usize, t: f64, device: &Device) -> CandleResult<Tensor> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for j in 0..half {
        let freq = (-(j as f64) * (10_000f64.ln()) / half as f64).exp();
        let angle = t * freq;
        out[j] = angle.cos() as f32;
        out[half + j] = angle.sin() as f32;
    }
    Tensor::from_vec(out, (1, dim), device)
}

/// Half-split RoPE tables `[len, head_dim/2]` (candle `rope`, the `use_real_unbind_dim=-2` branch).
fn rope_tables(
    head_dim: usize,
    len: usize,
    theta: f64,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for j in 0..half {
            let inv = 1.0 / theta.powf(2.0 * j as f64 / head_dim as f64);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

fn to_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> CandleResult<Tensor> {
    let (b, l, _) = x.dims3()?;
    x.reshape((b, l, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

fn from_heads(x: &Tensor) -> CandleResult<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, l, h * d))
}

fn repeat_kv(x: &Tensor, groups: usize) -> CandleResult<Tensor> {
    if groups == 1 {
        return x.contiguous();
    }
    let (b, h, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, groups, l, d))?
        .reshape((b, h * groups, l, d))?
        .contiguous()
}

/// The `AceStepAttention` module (self- or cross-attention). GQA with per-head q/k RMSNorm.
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_cross: bool,
}

impl Attention {
    fn new(cfg: &TransformerConfig, is_cross: bool, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        let hidden = cfg.hidden_size;
        Ok(Self {
            to_q: linear_b(
                hidden,
                cfg.num_attention_heads * d,
                cfg.attention_bias,
                vb.pp("to_q"),
            )?,
            to_k: linear_b(
                hidden,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("to_k"),
            )?,
            to_v: linear_b(
                hidden,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("to_v"),
            )?,
            to_out: linear_b(
                cfg.num_attention_heads * d,
                hidden,
                false,
                vb.pp("to_out.0"),
            )?,
            norm_q: rms_norm(d, cfg.rms_norm_eps, vb.pp("norm_q"))?,
            norm_k: rms_norm(d, cfg.rms_norm_eps, vb.pp("norm_k"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: d,
            is_cross,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        kv_src: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let q = self.norm_q.forward(&to_heads(
            &self.to_q.forward(x)?,
            self.num_heads,
            self.head_dim,
        )?)?;
        let k = self.norm_k.forward(&to_heads(
            &self.to_k.forward(kv_src)?,
            self.num_kv_heads,
            self.head_dim,
        )?)?;
        let v = to_heads(
            &self.to_v.forward(kv_src)?,
            self.num_kv_heads,
            self.head_dim,
        )?;
        let (q, k) = if let Some((cos, sin)) = rope {
            (
                candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?,
                candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?,
            )
        } else {
            (q, k)
        };
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v.contiguous()?)?;
        let _ = self.is_cross;
        self.to_out.forward(&from_heads(&out)?)
    }
}

struct Block {
    self_attn_norm: RmsNorm,
    self_attn: Attention,
    cross_attn_norm: RmsNorm,
    cross_attn: Attention,
    mlp_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    scale_shift_table: Tensor, // [1, 6, dim]
    sliding: bool,
}

impl Block {
    fn new(cfg: &TransformerConfig, sliding: bool, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            self_attn_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("self_attn_norm"))?,
            self_attn: Attention::new(cfg, false, vb.pp("self_attn"))?,
            cross_attn_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("cross_attn_norm"))?,
            cross_attn: Attention::new(cfg, true, vb.pp("cross_attn"))?,
            mlp_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("mlp_norm"))?,
            gate_proj: linear_b(d, cfg.intermediate_size, false, vb.pp("mlp.gate_proj"))?,
            up_proj: linear_b(d, cfg.intermediate_size, false, vb.pp("mlp.up_proj"))?,
            down_proj: linear_b(cfg.intermediate_size, d, false, vb.pp("mlp.down_proj"))?,
            scale_shift_table: vb.get((1, 6, d), "scale_shift_table")?,
            sliding,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        tproj: &Tensor, // [B, 6, dim]
        ctx: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        full_mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let m = self.scale_shift_table.broadcast_add(tproj)?; // [B, 6, dim]
        let chunk = |i: usize| m.narrow(1, i, 1); // [B, 1, dim]
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (c_shift, c_scale, c_gate) = (chunk(3)?, chunk(4)?, chunk(5)?);

        // Self-attention (AdaLN-Zero modulated, gated residual).
        let norm_x = modulate(&self.self_attn_norm.forward(x)?, &shift_msa, &scale_msa)?;
        let mask = if self.sliding {
            sliding_mask
        } else {
            full_mask
        };
        let attn = self
            .self_attn
            .forward(&norm_x, &norm_x, Some((cos, sin)), mask)?;
        let x = (x + attn.broadcast_mul(&gate_msa)?)?;

        // Cross-attention to the condition context (ungated residual, no RoPE).
        let norm_x = self.cross_attn_norm.forward(&x)?;
        let cross = self.cross_attn.forward(&norm_x, ctx, None, None)?;
        let x = (&x + cross)?;

        // SwiGLU MLP (AdaLN-Zero modulated, gated residual).
        let norm_x = modulate(&self.mlp_norm.forward(&x)?, &c_shift, &c_scale)?;
        let ff = self.down_proj.forward(
            &(self.gate_proj.forward(&norm_x)?.silu()? * self.up_proj.forward(&norm_x)?)?,
        )?;
        &x + ff.broadcast_mul(&c_gate)?
    }
}

/// The `AceStepTimestepEmbedding` (returns the `(temb, timestep_proj)` pair the AdaLN uses).
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
    time_proj: Linear,
    dim: usize,
    freq_dim: usize,
}

/// Sinusoidal timestep-embedding width (the pinned checkpoint's `time_embed.linear_1` input).
const FREQ_DIM: usize = 256;

impl TimestepEmbedding {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            linear_1: linear(FREQ_DIM, d, vb.pp("linear_1"))?,
            linear_2: linear(d, d, vb.pp("linear_2"))?,
            time_proj: linear(d, d * 6, vb.pp("time_proj"))?,
            dim: d,
            freq_dim: FREQ_DIM,
        })
    }

    /// `(temb [B, dim], timestep_proj [B, 6·dim])` for a scalar timestep, matching the diffusers
    /// v0.39.0 `AceStepTimestepEmbedding.forward` exactly:
    ///
    /// ```text
    ///   t_freq = time_sinusoid(t · self.scale)        # self.scale = 1000.0
    ///   temb   = linear_2(act1(linear_1(t_freq)))     # RAW linear_2 output — feeds the output AdaLN
    ///   tproj  = time_proj(act2(temb))                # act2 (SiLU) is applied ONLY into time_proj
    /// ```
    ///
    /// The `×1000` timestep scale is load-bearing: feeding `t ∈ [0, 1]` raw collapses the per-step
    /// conditioning (near-constant across the 8 sigmas) and flattens the output dynamics. `temb` is
    /// the raw `linear_2` output (no `act2`); the output-projection AdaLN consumes it unchanged.
    fn forward(&self, t: f64, device: &Device) -> CandleResult<(Tensor, Tensor)> {
        let s = sinusoidal(self.freq_dim, t * 1000.0, device)?;
        let temb = self.linear_2.forward(&self.linear_1.forward(&s)?.silu()?)?; // [1, dim]
        let tproj = self.time_proj.forward(&temb.silu()?)?; // [1, 6·dim] — act2 into time_proj only
        Ok((temb, tproj))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// One denoise run's precomputed cross-attention context (embedded once, reused per step is not
/// possible here since cross K/V are recomputed inside each block — kept simple/faithful).
pub struct DiT {
    proj_in_conv: Conv1d,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    condition_embedder: Linear,
    blocks: Vec<Block>,
    norm_out: RmsNorm,
    scale_shift_table_out: Tensor, // [1, 2, dim]
    proj_out_conv: ConvTranspose1d,
    cfg: TransformerConfig,
}

impl DiT {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.hidden_size;
        let proj_in_conv = conv1d(
            cfg.in_channels,
            d,
            cfg.patch_size,
            Conv1dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("proj_in_conv"),
        )?;
        let time_embed = TimestepEmbedding::new(cfg, vb.pp("time_embed"))?;
        let time_embed_r = TimestepEmbedding::new(cfg, vb.pp("time_embed_r"))?;
        let condition_embedder = linear(cfg.encoder_hidden_size, d, vb.pp("condition_embedder"))?;
        let vb_b = vb.pp("layers");
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(Block::new(cfg, cfg.is_sliding(i), vb_b.pp(i))?);
        }
        let norm_out = rms_norm(d, cfg.rms_norm_eps, vb.pp("norm_out"))?;
        let scale_shift_table_out = vb.get((1, 2, d), "scale_shift_table")?;
        let proj_out_conv = conv_transpose1d(
            d,
            cfg.audio_acoustic_hidden_dim,
            cfg.patch_size,
            ConvTranspose1dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("proj_out_conv"),
        )?;
        Ok(Self {
            proj_in_conv,
            time_embed,
            time_embed_r,
            condition_embedder,
            blocks,
            norm_out,
            scale_shift_table_out,
            proj_out_conv,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.cfg
    }

    /// Embed the condition context once per run: `Linear(encoder_hidden_size → dim)`.
    pub fn embed_context(&self, encoder_hidden_states: &Tensor) -> CandleResult<Tensor> {
        self.condition_embedder.forward(encoder_hidden_states)
    }

    /// Symmetric sliding-window band mask `[1, 1, T, T]` (additive `-inf` outside `|i−j| ≤ w`).
    fn sliding_mask(&self, len: usize, device: &Device) -> CandleResult<Tensor> {
        let w = self.cfg.sliding_window as i64;
        let data: Vec<f32> = (0..len)
            .flat_map(|i| {
                (0..len).map(move |j| {
                    if (i as i64 - j as i64).abs() <= w {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_vec(data, (1, 1, len, len), device)
    }

    /// One velocity prediction. `hidden` is `[B, T, acoustic]`; `context_latents` is
    /// `[B, T, 2·acoustic]`; `ctx` is the pre-embedded condition context `[B, S, dim]`.
    /// `cancel` is polled between blocks.
    pub fn forward(
        &self,
        hidden: &Tensor,
        context_latents: &Tensor,
        timestep: f64,
        ctx: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Tensor>> {
        let device = hidden.device();
        let (_b, orig_len, _c) = hidden.dims3()?;

        // Concatenate context + noisy latents, pad the sequence to a patch boundary.
        let mut x = Tensor::cat(&[context_latents, hidden], D::Minus1)?; // [B, T, in_channels]
        let pad = (self.cfg.patch_size - orig_len % self.cfg.patch_size) % self.cfg.patch_size;
        if pad > 0 {
            let (b, _l, c) = x.dims3()?;
            let z = Tensor::zeros((b, pad, c), x.dtype(), device)?;
            x = Tensor::cat(&[&x, &z], 1)?;
        }
        // Patchify: [B, T, C] → [B, C, T] → Conv1d → [B, dim, T/p] → [B, T/p, dim].
        let x = self
            .proj_in_conv
            .forward(&x.transpose(1, 2)?.contiguous()?)?;
        let mut x = x.transpose(1, 2)?.contiguous()?;
        let patched_len = x.dim(1)?;

        // AdaLN-Zero conditioning. For inference timestep_r == timestep, so time_embed_r sees 0.
        let (temb_t, tproj_t) = self.time_embed.forward(timestep, device)?;
        let (temb_r, tproj_r) = self.time_embed_r.forward(0.0, device)?;
        let temb = (temb_t + temb_r)?; // [1, dim]
        let tproj = (tproj_t + tproj_r)?.reshape((1, 6, self.time_embed.dim()))?;

        let (cos, sin) = rope_tables(self.cfg.head_dim, patched_len, self.cfg.rope_theta, device)?;
        let full_mask: Option<Tensor> = None;
        let sliding_mask = if self
            .cfg
            .layer_types
            .iter()
            .any(|t| t == "sliding_attention")
        {
            Some(self.sliding_mask(patched_len, device)?)
        } else {
            None
        };

        for (i, block) in self.blocks.iter().enumerate() {
            if cancel() {
                return Ok(None);
            }
            let _ = i;
            x = block.forward(
                &x,
                &tproj,
                ctx,
                &cos,
                &sin,
                full_mask.as_ref(),
                sliding_mask.as_ref(),
            )?;
        }

        // Output AdaLN (2-way) from the unprojected temb.
        let m = self
            .scale_shift_table_out
            .broadcast_add(&temb.unsqueeze(1)?)?; // [1, 2, dim]
        let shift = m.narrow(1, 0, 1)?;
        let scale = m.narrow(1, 1, 1)?;
        let x = modulate(&self.norm_out.forward(&x)?, &shift, &scale)?;

        // Depatchify: [B, T/p, dim] → [B, dim, T/p] → ConvTranspose1d → [B, acoustic, T] → [B, T, acoustic].
        let v = self
            .proj_out_conv
            .forward(&x.transpose(1, 2)?.contiguous()?)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let v = v.narrow(1, 0, orig_len)?; // crop the patch padding.
        Ok(Some(v))
    }
}

/// Informational: the pinned DiT checkpoint dtype (loading converts to the compute dtype).
pub const CHECKPOINT_DTYPE: DType = DType::F32;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn sinusoidal_flip_puts_cos_first() {
        let dev = Device::Cpu;
        let e = sinusoidal(8, 0.0, &dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(e, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rope_tables_start_at_identity() {
        let dev = Device::Cpu;
        let (cos, sin) = rope_tables(4, 3, 1_000_000.0, &dev).unwrap();
        assert_eq!(cos.dims(), &[3, 2]);
        let c = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&c[0..2], &[1.0, 1.0]);
        assert_eq!(&s[0..2], &[0.0, 0.0]);
    }
}
