//! The MOSS-SoundEffect **audio DiT** (sc-12841) — a faithful port of the reference
//! `WanAudioModel` (a Wan-2.1-style DiT specialized to 1-D audio latents) in the shipped `"dac"`
//! configuration: `patch_size=[1]` Conv1d patchify over `[B, 128, T]` latents, 30 blocks of
//! {self-attention with dim-wide q/k RMSNorm + interleaved 1-D RoPE, text cross-attention,
//! GELU-tanh FFN} under 6-way adaLN modulation, and a 2-way-modulated linear head back to the
//! 128-channel velocity field.
//!
//! Weight names are the **diffusers export** names of the pinned checkpoint
//! (`transformer/diffusion_pytorch_model.safetensors`) — `blocks.N.attn1/attn2/ffn.net/norm2/
//! scale_shift_table`, `condition_embedder.*`, `patch_embedding`, `proj_out` — read directly so
//! no rename table can drift (the reference maps these to its native module names; the math
//! below is the native forward).

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use crate::config::DitConfig;

/// LayerNorm without affine parameters (`elementwise_affine=False`), over the last dim.
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&(var + eps)?.sqrt()?)
}

/// RMSNorm over the last dim with a learned weight (`torch.nn.RMSNorm(dim, eps)` — the
/// dim-wide q/k norm the Wan attention stacks use, distinct from Qwen3's per-head norm).
struct RmsNormDim {
    weight: Tensor,
    eps: f64,
}

impl RmsNormDim {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(ms + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)
    }
}

/// `x·(1+scale) + shift` with `[B, 1, dim]` modulation broadcast over the sequence.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> CandleResult<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// The sinusoidal timestep embedding (`sinusoidal_embedding_1d(freq_dim, t)`): f64 host math,
/// `cat[cos, sin]` ordering (cos first).
pub fn sinusoidal_embedding(freq_dim: usize, t: f64, device: &Device) -> CandleResult<Tensor> {
    let half = freq_dim / 2;
    let mut out = vec![0f32; freq_dim];
    for j in 0..half {
        let freq = 10_000f64.powf(-(j as f64) / half as f64);
        let angle = t * freq;
        out[j] = angle.cos() as f32;
        out[half + j] = angle.sin() as f32;
    }
    Tensor::from_vec(out, (1, freq_dim), device)
}

/// Interleaved 1-D RoPE tables for positions `0..len` over `head_dim` — the `"dac"` branch's
/// `precompute_freqs_cis_1d(head_dim)` (θ=10 000; the chunk-into-3-then-concat in the reference
/// forward reassembles exactly this table). Returns `(cos, sin)` of shape `[len, head_dim/2]`.
pub fn rope_tables(head_dim: usize, len: usize, device: &Device) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for j in 0..half {
            let freq = 1.0 / 10_000f64.powf(2.0 * j as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> CandleResult<Tensor> {
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v.contiguous()?)
}

/// Split `[B, L, H·D]` into heads `[B, H, L, D]`.
fn to_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> CandleResult<Tensor> {
    let (b, l, _) = x.dims3()?;
    x.reshape((b, l, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

/// Merge `[B, H, L, D]` back to `[B, L, H·D]`.
fn from_heads(x: &Tensor) -> CandleResult<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, l, h * d))
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: RmsNormDim,
    norm_k: RmsNormDim,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.dim;
        Ok(Self {
            q: linear(d, d, vb.pp("to_q"))?,
            k: linear(d, d, vb.pp("to_k"))?,
            v: linear(d, d, vb.pp("to_v"))?,
            o: linear(d, d, vb.pp("to_out.0"))?,
            norm_q: RmsNormDim::new(d, cfg.eps, vb.pp("norm_q"))?,
            norm_k: RmsNormDim::new(d, cfg.eps, vb.pp("norm_k"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let q = self.norm_q.forward(&self.q.forward(x)?)?;
        let k = self.norm_k.forward(&self.k.forward(x)?)?;
        let v = self.v.forward(x)?;
        let q = to_heads(&q, self.num_heads, self.head_dim)?;
        let k = to_heads(&k, self.num_heads, self.head_dim)?;
        let v = to_heads(&v, self.num_heads, self.head_dim)?;
        // Interleaved-pair rotation (`rope_apply` — even/odd lanes), identical per head.
        let q = candle_nn::rotary_emb::rope_i(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, cos, sin)?;
        let out = attention(&q, &k, &v, self.head_dim)?;
        self.o.forward(&from_heads(&out)?)
    }
}

struct CrossAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: RmsNormDim,
    norm_k: RmsNormDim,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.dim;
        Ok(Self {
            q: linear(d, d, vb.pp("to_q"))?,
            k: linear(d, d, vb.pp("to_k"))?,
            v: linear(d, d, vb.pp("to_v"))?,
            o: linear(d, d, vb.pp("to_out.0"))?,
            norm_q: RmsNormDim::new(d, cfg.eps, vb.pp("norm_q"))?,
            norm_k: RmsNormDim::new(d, cfg.eps, vb.pp("norm_k"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `x` attends over the (pre-embedded) text context; the context K/V are precomputed once
    /// per denoise run by [`DiT::embed_context`] and reused across steps.
    fn forward(&self, x: &Tensor, ctx_k: &Tensor, ctx_v: &Tensor) -> CandleResult<Tensor> {
        let q = self.norm_q.forward(&self.q.forward(x)?)?;
        let q = to_heads(&q, self.num_heads, self.head_dim)?;
        let out = attention(&q, ctx_k, ctx_v, self.head_dim)?;
        self.o.forward(&from_heads(&out)?)
    }

    fn context_kv(&self, ctx: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let k = self.norm_k.forward(&self.k.forward(ctx)?)?;
        let v = self.v.forward(ctx)?;
        Ok((
            to_heads(&k, self.num_heads, self.head_dim)?,
            to_heads(&v, self.num_heads, self.head_dim)?,
        ))
    }
}

struct Block {
    self_attn: SelfAttention,
    cross_attn: CrossAttention,
    norm3: candle_nn::LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
    modulation: Tensor, // [1, 6, dim]
    eps: f64,
}

impl Block {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> CandleResult<Self> {
        // The affine LayerNorm between self- and cross-attention is the diffusers export's
        // `norm2` (native `norm3`); `norm1`/`norm2` (native) are affine-free and carry no keys.
        let norm3 = candle_nn::layer_norm(
            cfg.dim,
            candle_nn::LayerNormConfig {
                eps: cfg.eps,
                ..Default::default()
            },
            vb.pp("norm2"),
        )?;
        Ok(Self {
            self_attn: SelfAttention::new(cfg, vb.pp("attn1"))?,
            cross_attn: CrossAttention::new(cfg, vb.pp("attn2"))?,
            norm3,
            ffn_in: linear(cfg.dim, cfg.ffn_dim, vb.pp("ffn.net.0.proj"))?,
            ffn_out: linear(cfg.ffn_dim, cfg.dim, vb.pp("ffn.net.2"))?,
            modulation: vb.get((1, 6, cfg.dim), "scale_shift_table")?,
            eps: cfg.eps,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        ctx_k: &Tensor,
        ctx_v: &Tensor,
        t_mod: &Tensor, // [1, 6, dim]
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let m = self.modulation.broadcast_add(t_mod)?; // [1, 6, dim]
        let chunk = |i: usize| m.narrow(1, i, 1); // [1, 1, dim]
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);

        let input_x = modulate(&layer_norm_no_affine(x, self.eps)?, &shift_msa, &scale_msa)?;
        let x = (x + self
            .self_attn
            .forward(&input_x, cos, sin)?
            .broadcast_mul(&gate_msa)?)?;
        let x = (&x
            + self
                .cross_attn
                .forward(&self.norm3.forward(&x)?, ctx_k, ctx_v)?)?;
        let input_x = modulate(&layer_norm_no_affine(&x, self.eps)?, &shift_mlp, &scale_mlp)?;
        let ffn = self
            .ffn_out
            .forward(&self.ffn_in.forward(&input_x)?.gelu()?)?;
        &x + ffn.broadcast_mul(&gate_mlp)?
    }
}

/// One denoise-run's precomputed conditioning: the embedded text context's per-block K/V, the
/// RoPE tables, and the timestep-independent pieces are all owned by the caller so the per-step
/// forward is just the block stack.
pub struct ContextKv {
    per_block: Vec<(Tensor, Tensor)>,
}

/// The assembled audio DiT.
pub struct DiT {
    patch_embedding: Conv1d,
    text_emb_1: Linear,
    text_emb_2: Linear,
    time_emb_1: Linear,
    time_emb_2: Linear,
    time_proj: Linear,
    blocks: Vec<Block>,
    head: Linear,
    head_modulation: Tensor, // [1, 2, dim]
    cfg: DitConfig,
}

impl DiT {
    pub fn new(cfg: &DitConfig, vb: VarBuilder) -> CandleResult<Self> {
        let patch_embedding = conv1d(
            cfg.in_dim,
            cfg.dim,
            cfg.patch_size[0],
            Conv1dConfig {
                stride: cfg.patch_size[0],
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let ce = vb.pp("condition_embedder");
        let text_emb_1 = linear(cfg.text_dim, cfg.dim, ce.pp("text_embedder.linear_1"))?;
        let text_emb_2 = linear(cfg.dim, cfg.dim, ce.pp("text_embedder.linear_2"))?;
        let time_emb_1 = linear(cfg.freq_dim, cfg.dim, ce.pp("time_embedder.linear_1"))?;
        let time_emb_2 = linear(cfg.dim, cfg.dim, ce.pp("time_embedder.linear_2"))?;
        let time_proj = linear(cfg.dim, cfg.dim * 6, ce.pp("time_proj"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_b = vb.pp("blocks");
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb_b.pp(i))?);
        }
        let head = linear(cfg.dim, cfg.out_dim * cfg.patch_size[0], vb.pp("proj_out"))?;
        let head_modulation = vb.get((1, 2, cfg.dim), "scale_shift_table")?;
        Ok(Self {
            patch_embedding,
            text_emb_1,
            text_emb_2,
            time_emb_1,
            time_emb_2,
            time_proj,
            blocks,
            head,
            head_modulation,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// Embed a raw text context `[B, L, text_dim]` (`text_embedding`: Linear → GELU-tanh →
    /// Linear) and precompute every block's cross-attention K/V — timestep-independent, done
    /// once per prompt per run.
    pub fn embed_context(&self, context: &Tensor) -> CandleResult<ContextKv> {
        let ctx = self
            .text_emb_2
            .forward(&self.text_emb_1.forward(context)?.gelu()?)?;
        let mut per_block = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            per_block.push(block.cross_attn.context_kv(&ctx)?);
        }
        Ok(ContextKv { per_block })
    }

    /// RoPE tables for a `latent_len`-frame run (shared across steps).
    pub fn rope(&self, latent_len: usize, device: &Device) -> CandleResult<(Tensor, Tensor)> {
        rope_tables(self.cfg.head_dim(), latent_len, device)
    }

    /// One velocity prediction: noisy latents `[B, in_dim, T]` + timestep + embedded context →
    /// `[B, out_dim, T]`. `cancel` is polled between blocks so a mid-forward cancel lands
    /// without waiting for a full 30-block stack.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: f64,
        ctx: &ContextKv,
        cos: &Tensor,
        sin: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Tensor>> {
        let device = latents.device();
        // Timestep embedding (f64 host math, then the SiLU MLP) + 6-way projection.
        let t_emb = sinusoidal_embedding(self.cfg.freq_dim, timestep, device)?;
        let t = self
            .time_emb_2
            .forward(&self.time_emb_1.forward(&t_emb)?.silu()?)?; // [1, dim]
        let t_mod = self
            .time_proj
            .forward(&t.silu()?)?
            .reshape((1, 6, self.cfg.dim))?;

        // Patchify: Conv1d(k=s=patch) then [B, C, F] → [B, F, C].
        let mut x = self
            .patch_embedding
            .forward(latents)?
            .transpose(1, 2)?
            .contiguous()?;

        for (i, block) in self.blocks.iter().enumerate() {
            if cancel() {
                return Ok(None);
            }
            let (ctx_k, ctx_v) = &ctx.per_block[i];
            x = block.forward(&x, ctx_k, ctx_v, &t_mod, cos, sin)?;
        }

        // Head: 2-way modulation from the *unprojected* time embedding `t`.
        let m = self.head_modulation.broadcast_add(&t.unsqueeze(1)?)?; // [1, 2, dim]
        let shift = m.narrow(1, 0, 1)?;
        let scale = m.narrow(1, 1, 1)?;
        let x = modulate(&layer_norm_no_affine(&x, self.cfg.eps)?, &shift, &scale)?;
        // Head projection [B, F, out_dim·p], then unpatchify `b f (p c) -> b c (f p)`
        // (p = 1 for the shipped checkpoint).
        let x = self.head.forward(&x)?;
        let out = x.transpose(1, 2)?.contiguous()?;
        Ok(Some(out))
    }
}

/// Expected weight dtype of the pinned DiT checkpoint (informational; loading converts to the
/// compute dtype).
pub const CHECKPOINT_DTYPE: DType = DType::F32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_embedding_matches_the_reference_layout() {
        let dev = Device::Cpu;
        // t = 0 → cos half all 1, sin half all 0 (cos-first concat).
        let e = sinusoidal_embedding(8, 0.0, &dev).unwrap();
        assert_eq!(
            e.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        // t = 1000, j = 0 → angle 1000 rad: cos(1000) in slot 0, sin(1000) in slot half.
        let e = sinusoidal_embedding(8, 1000.0, &dev).unwrap();
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - (1000f64.cos() as f32)).abs() < 1e-6);
        assert!((v[4] - (1000f64.sin() as f32)).abs() < 1e-6);
    }

    #[test]
    fn rope_tables_are_position_frequency_ordered() {
        let dev = Device::Cpu;
        let (cos, sin) = rope_tables(4, 3, &dev).unwrap();
        assert_eq!(cos.dims(), &[3, 2]);
        let c = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Position 0 → identity rotation.
        assert_eq!(&c[0..2], &[1.0, 1.0]);
        assert_eq!(&s[0..2], &[0.0, 0.0]);
        // Position 1, j=0 → angle 1; j=1 → angle 10000^(-1/2).
        assert!((c[2] - (1f64.cos() as f32)).abs() < 1e-6);
        let f1 = 1.0 / 10_000f64.powf(0.5);
        assert!((s[3] - (f1.sin() as f32)).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_no_affine_normalizes_rows() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 4), &dev).unwrap();
        let y = layer_norm_no_affine(&x, 1e-6).unwrap();
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean: f32 = v.iter().sum::<f32>() / 4.0;
        let var: f32 = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-4);
    }
}
