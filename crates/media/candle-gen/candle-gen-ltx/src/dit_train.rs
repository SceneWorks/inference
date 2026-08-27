//! Vendored, trainable video-only LTX-2.3 DiT (sc-13866).
//!
//! This is the candle twin of `mlx-gen-ltx::LtxDiT`: it loads the video leaves from the same
//! `AVTransformer3DModel` checkpoint as [`crate::transformer::AvDiT`], but omits the audio and
//! bidirectional cross-modal branches used only by joint AV inference. The four projections in each
//! video `attn1`/`attn2` are [`LoraLinear`]s over the unchanged frozen [`QLinear`] bases.
//!
//! Two inference kernels are intentionally replaced on adapter-gradient paths. The fused
//! `softmax_last_dim` and `rotary_emb::rope_i` custom ops do not implement backward in candle, so
//! attention uses composable [`softmax`] and `apply_rope_diff`. Both implement the same forward math;
//! the zero-adapter parity tests compare this model with the inference AvDiT video reduction.

use candle_gen::candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen::candle_nn::{ops::sigmoid, ops::softmax, VarBuilder};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{LoraHost, LoraLinear};

use crate::config::TransformerConfig;
use crate::quant::{qlinear, QLinear};
use crate::rope::precompute_split_freqs_nd;

/// Default PEFT suffixes for every adapted attention projection.
pub const LTX_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

fn linear(vb: &VarBuilder, key: &str) -> Result<QLinear> {
    qlinear(vb, key, true)
}

fn lora_linear(
    vb: &VarBuilder,
    key: &str,
    in_features: usize,
    out_features: usize,
) -> Result<LoraLinear> {
    let path = vb.pp(key).prefix();
    Ok(linear(vb, key)?.into_lora(in_features, out_features, path))
}

fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

fn gated(x: &Tensor, out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x + out.broadcast_mul(gate)?
}

fn rms_noweight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    xf.broadcast_mul(&inv)?.to_dtype(x.dtype())
}

fn rms_weighted(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    rms_noweight(x, eps)?.broadcast_mul(weight)
}

fn layer_norm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(x.dtype())
}

fn timestep_embedding(ts: &Tensor, device: &Device) -> Result<Tensor> {
    const DIM: usize = 256;
    let half = DIM / 2;
    let neg_ln = -(10000f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f64 / half as f64).exp() as f32)
        .collect();
    let n = ts.dim(0)?;
    let freq = Tensor::from_vec(freqs, (1, half), device)?;
    let emb = ts.reshape((n, 1))?.broadcast_mul(&freq)?;
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1)
}

fn ada_values(table: &Tensor, ts_emb: &Tensor, lo: usize, hi: usize) -> Result<Vec<Tensor>> {
    let (num, inner) = table.dims2()?;
    let (b, s, _) = ts_emb.dims3()?;
    let ts4 = ts_emb.reshape((b, s, num, inner))?;
    let mut out = Vec::with_capacity(hi - lo);
    for row in lo..hi {
        let trow = table.narrow(0, row, 1)?.reshape((1, 1, inner))?;
        let tsrow = ts4.narrow(2, row, 1)?.squeeze(2)?;
        out.push(trow.broadcast_add(&tsrow)?);
    }
    Ok(out)
}

/// Differentiable GPT-NeoX/split RoPE, equivalent to `crate::rope::apply_split_rope`.
fn apply_rope_diff(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let half = x.dim(D::Minus1)? / 2;
    let xf = x.to_dtype(DType::F32)?;
    let a = xf.narrow(D::Minus1, 0, half)?;
    let b = xf.narrow(D::Minus1, half, half)?;
    let out_a = (a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?;
    let out_b = (b.broadcast_mul(cos)? + a.broadcast_mul(sin)?)?;
    Tensor::cat(&[&out_a, &out_b], D::Minus1)?.to_dtype(dtype)
}

struct TrainAttention {
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: QLinear,
    heads: usize,
    dim_head: usize,
    eps: f64,
}

impl TrainAttention {
    fn load(vb: VarBuilder, heads: usize, dim_head: usize, eps: f64) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_q: lora_linear(&vb, "to_q", inner, inner)?,
            to_k: lora_linear(&vb, "to_k", inner, inner)?,
            to_v: lora_linear(&vb, "to_v", inner, inner)?,
            to_out: lora_linear(&vb, "to_out.0", inner, inner)?,
            q_norm: vb.get_unchecked("q_norm.weight")?.to_dtype(vb.dtype())?,
            k_norm: vb.get_unchecked("k_norm.weight")?.to_dtype(vb.dtype())?,
            gate: linear(&vb, "to_gate_logits")?,
            heads,
            dim_head,
            eps,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.to_q)?;
        f(&mut self.to_k)?;
        f(&mut self.to_v)?;
        f(&mut self.to_out)
    }

    fn to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ctx = context.unwrap_or(x);
        let q = rms_weighted(&self.to_q.forward(x)?.contiguous()?, &self.q_norm, self.eps)?;
        let k = rms_weighted(
            &self.to_k.forward(ctx)?.contiguous()?,
            &self.k_norm,
            self.eps,
        )?;
        let v = self.to_v.forward(ctx)?;
        let mut q = self.to_heads(&q)?;
        let mut k = self.to_heads(&k)?;
        let v = self.to_heads(&v)?;
        if let Some((cos, sin)) = rope {
            q = apply_rope_diff(&q, cos, sin)?;
            k = apply_rope_diff(&k, cos, sin)?;
        }

        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let q = q.to_dtype(DType::F32)?.contiguous()?;
        let k = k.to_dtype(DType::F32)?.contiguous()?;
        let v = v.to_dtype(DType::F32)?.contiguous()?;
        let out = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            |scores| softmax(scores, D::Minus1),
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let (b, s, _) = x.dims3()?;
        let inner = self.heads * self.dim_head;
        let mut out = out
            .transpose(1, 2)?
            .reshape((b, s, inner))?
            .to_dtype(x.dtype())?;
        let gates = (sigmoid(&self.gate.forward(x)?)? * 2.0)?.reshape((b, s, self.heads, 1))?;
        out = out
            .reshape((b, s, self.heads, self.dim_head))?
            .broadcast_mul(&gates)?
            .reshape((b, s, inner))?;
        self.to_out.forward(&out)
    }
}

struct FeedForward {
    proj_in: QLinear,
    proj_out: QLinear,
}

impl FeedForward {
    /// `bias` is `TransformerConfig::ff_bias` (sc-18758) — see `crate::transformer::FeedForward::load`.
    fn load(vb: VarBuilder, bias: bool) -> Result<Self> {
        Ok(Self {
            proj_in: qlinear(&vb.pp("net.0"), "proj", bias)?,
            proj_out: qlinear(&vb.pp("net"), "2", bias)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }
}

struct AdaLayerNormSingle {
    ts_lin1: QLinear,
    ts_lin2: QLinear,
    linear: QLinear,
}

impl AdaLayerNormSingle {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ts_lin1: linear(&vb.pp("emb.timestep_embedder"), "linear_1")?,
            ts_lin2: linear(&vb.pp("emb.timestep_embedder"), "linear_2")?,
            linear: linear(&vb, "linear")?,
        })
    }

    fn forward(&self, ts_flat: &Tensor, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let proj = timestep_embedding(ts_flat, device)?.to_dtype(dtype)?;
        let h = self.ts_lin1.forward(&proj)?.silu()?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&embedded.silu()?)?;
        Ok((scale_shift, embedded))
    }
}

struct TrainBlock {
    attn1: TrainAttention,
    attn2: TrainAttention,
    ff: FeedForward,
    scale_shift_table: Tensor,
    prompt_scale_shift_table: Tensor,
    eps: f64,
}

impl TrainBlock {
    fn load(vb: VarBuilder, cfg: &TransformerConfig) -> Result<Self> {
        let table = |key: &str| vb.get_unchecked(key)?.to_dtype(vb.dtype());
        Ok(Self {
            attn1: TrainAttention::load(vb.pp("attn1"), cfg.num_heads, cfg.head_dim, cfg.norm_eps)?,
            attn2: TrainAttention::load(vb.pp("attn2"), cfg.num_heads, cfg.head_dim, cfg.norm_eps)?,
            ff: FeedForward::load(vb.pp("ff"), cfg.ff_bias)?,
            scale_shift_table: table("scale_shift_table")?,
            prompt_scale_shift_table: table("prompt_scale_shift_table")?,
            eps: cfg.norm_eps,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn1.visit_lora_mut(f)?;
        self.attn2.visit_lora_mut(f)
    }

    fn forward(&self, hidden: &Tensor, ctx: &MainCtx) -> Result<Tensor> {
        let msa = ada_values(&self.scale_shift_table, &ctx.ts_emb, 0, 3)?;
        let norm = modulate(&rms_noweight(hidden, self.eps)?, &msa[1], &msa[0])?;
        let attn = self
            .attn1
            .forward(&norm, None, Some((&ctx.cos, &ctx.sin)))?;
        let hidden = gated(hidden, &attn, &msa[2])?;

        let prompt = ada_values(&self.prompt_scale_shift_table, &ctx.prompt_ts, 0, 2)?;
        let context = modulate(&ctx.context, &prompt[1], &prompt[0])?;
        let cross_mod = ada_values(&self.scale_shift_table, &ctx.ts_emb, 6, 9)?;
        let norm = modulate(
            &rms_noweight(&hidden, self.eps)?,
            &cross_mod[1],
            &cross_mod[0],
        )?;
        let cross = self.attn2.forward(&norm, Some(&context), None)?;
        let hidden = gated(&hidden, &cross, &cross_mod[2])?;

        let mlp = ada_values(&self.scale_shift_table, &ctx.ts_emb, 3, 6)?;
        let norm = modulate(&rms_noweight(&hidden, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm)?;
        gated(&hidden, &ff, &mlp[2])
    }
}

/// Constant side tensors produced by [`LtxDiT::forward_pre_main`] and reused by every block segment.
#[derive(Clone)]
pub struct MainCtx {
    ts_emb: Tensor,
    emb_ts: Tensor,
    prompt_ts: Tensor,
    context: Tensor,
    cos: Tensor,
    sin: Tensor,
}

/// Trainable video-only reduction of the LTX-2.3 `AvDiT`.
pub struct LtxDiT {
    patchify: QLinear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    blocks: Vec<TrainBlock>,
    scale_shift_table: Tensor,
    proj_out: QLinear,
    cfg: TransformerConfig,
    device: Device,
    dtype: DType,
}

impl LtxDiT {
    /// Load video weights from a builder rooted at `model.diffusion_model`.
    pub fn new(vb: VarBuilder, cfg: &TransformerConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(TrainBlock::load(
                vb.pp(format!("transformer_blocks.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            patchify: linear(&vb, "patchify_proj")?,
            adaln: AdaLayerNormSingle::load(vb.pp("adaln_single"))?,
            prompt_adaln: AdaLayerNormSingle::load(vb.pp("prompt_adaln_single"))?,
            blocks,
            scale_shift_table: vb
                .get_unchecked("scale_shift_table")?
                .to_dtype(vb.dtype())?,
            proj_out: linear(&vb, "proj_out")?,
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Dense composition of pre-main, all block segments, and the velocity head.
    pub fn forward(
        &self,
        latent: &Tensor,
        sigma: f64,
        context: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (mut hidden, ctx) = self.forward_pre_main(latent, sigma, context, positions)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &ctx)?;
        }
        self.velocity_out(&hidden, &ctx)
    }

    /// Patchify, timestep/prompt modulation, context cast and split-RoPE table construction.
    pub fn forward_pre_main(
        &self,
        latent: &Tensor,
        sigma: f64,
        context: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, MainCtx)> {
        let b = latent.dim(0)?;
        let inner = self.cfg.inner_dim();
        let ts_flat = Tensor::from_vec(
            vec![(sigma * self.cfg.timestep_scale_multiplier) as f32; b],
            (b,),
            &self.device,
        )?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, &self.device, self.dtype)?;
        let (prompt_ts, _) = self
            .prompt_adaln
            .forward(&ts_flat, &self.device, self.dtype)?;
        let (cos, sin) = precompute_split_freqs_nd(
            positions,
            inner,
            self.cfg.rope_theta,
            &self.cfg.rope_max_pos,
            self.cfg.num_heads,
            &self.device,
        )?;
        Ok((
            self.patchify.forward(&latent.to_dtype(self.dtype)?)?,
            MainCtx {
                ts_emb: ts_emb.reshape((b, 1, 9 * inner))?,
                emb_ts: emb_ts.reshape((b, 1, inner))?,
                prompt_ts: prompt_ts.reshape((b, 1, 2 * inner))?,
                context: context.to_dtype(self.dtype)?,
                cos,
                sin,
            },
        ))
    }

    /// One checkpoint segment per transformer block (`[hidden] -> [hidden]`).
    pub fn main_block_segments<'a>(&'a self, ctx: &'a MainCtx) -> Vec<Segment<'a>> {
        self.blocks
            .iter()
            .map(|block| -> Segment<'a> {
                Box::new(move |state: &[Tensor]| Ok(vec![block.forward(&state[0], ctx)?]))
            })
            .collect()
    }

    /// Final affine-free LayerNorm, two-row adaLN modulation and video velocity projection.
    pub fn velocity_out(&self, hidden: &Tensor, ctx: &MainCtx) -> Result<Tensor> {
        let b = hidden.dim(0)?;
        let inner = self.cfg.inner_dim();
        let table = self.scale_shift_table.reshape((1, 1, 2, inner))?;
        let ss = table.broadcast_add(&ctx.emb_ts.reshape((b, 1, 1, inner))?)?;
        let shift = ss.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = ss.narrow(2, 1, 1)?.squeeze(2)?;
        let normed = layer_norm_noaffine(hidden, self.cfg.norm_eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }
}

impl LoraHost for LtxDiT {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for block in &mut self.blocks {
            block.visit_lora_mut(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AvConfig;
    use crate::transformer::AvDiT;
    use candle_gen::candle_core::Shape;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::train::gradient_checkpoint::checkpointed_backward;
    use candle_gen::train::lora::{build_lora_targets, save_lora_peft};
    use candle_gen::train::optim::TrainOptimizer;
    use std::collections::HashMap;

    fn tiny_cfg() -> AvConfig {
        AvConfig {
            video: TransformerConfig {
                num_layers: 1,
                num_heads: 1,
                head_dim: 12,
                norm_eps: 1e-6,
                rope_theta: 10000.0,
                rope_max_pos: [20, 64, 64],
                timestep_scale_multiplier: 1000.0,
                ff_bias: true,
                use_keyframes_abs_pos_embedding: false,
            },
            audio_heads: 1,
            audio_head_dim: 12,
            audio_max_pos: 20,
            cross_inner: 12,
            cross_max_pos: 20,
            caption_feature_version: AvConfig::ltx_2_3().caption_feature_version,
            audio_ff_bias: true,
        }
    }

    /// A deliberately tiny shape with the two LTX-2.5 DiT deltas enabled.  Keeping the lifecycle
    /// test on this config catches a trainer accidentally falling back to the 2.3 FF-bias layout.
    fn tiny_cfg_25() -> AvConfig {
        let mut cfg = tiny_cfg();
        cfg.video.ff_bias = false;
        cfg.video.use_keyframes_abs_pos_embedding = true;
        cfg
    }

    fn put<S: Into<Shape>>(
        map: &mut HashMap<String, Tensor>,
        key: impl Into<String>,
        shape: S,
        dev: &Device,
    ) {
        map.insert(
            key.into(),
            Tensor::randn(0f32, 0.08f32, shape, dev).unwrap(),
        );
    }

    fn put_linear(
        map: &mut HashMap<String, Tensor>,
        key: &str,
        out: usize,
        input: usize,
        dev: &Device,
    ) {
        put(map, format!("{key}.weight"), (out, input), dev);
        put(map, format!("{key}.bias"), out, dev);
    }

    fn put_adaln(
        map: &mut HashMap<String, Tensor>,
        key: &str,
        inner: usize,
        coeff: usize,
        dev: &Device,
    ) {
        put_linear(
            map,
            &format!("{key}.emb.timestep_embedder.linear_1"),
            inner,
            256,
            dev,
        );
        put_linear(
            map,
            &format!("{key}.emb.timestep_embedder.linear_2"),
            inner,
            inner,
            dev,
        );
        put_linear(map, &format!("{key}.linear"), coeff * inner, inner, dev);
    }

    fn put_attention(
        map: &mut HashMap<String, Tensor>,
        key: &str,
        inner: usize,
        heads: usize,
        dev: &Device,
    ) {
        for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
            put_linear(map, &format!("{key}.{leaf}"), inner, inner, dev);
        }
        put_linear(map, &format!("{key}.to_gate_logits"), heads, inner, dev);
        put(map, format!("{key}.q_norm.weight"), inner, dev);
        put(map, format!("{key}.k_norm.weight"), inner, dev);
    }

    fn put_ff(map: &mut HashMap<String, Tensor>, key: &str, inner: usize, dev: &Device) {
        put_linear(map, &format!("{key}.net.0.proj"), 2 * inner, inner, dev);
        put_linear(map, &format!("{key}.net.2"), inner, 2 * inner, dev);
    }

    fn weights(cfg: &AvConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let vi = cfg.video.inner_dim();
        let ai = cfg.audio_inner();
        put_linear(&mut map, "patchify_proj", vi, 8, dev);
        put_linear(&mut map, "proj_out", 8, vi, dev);
        put(&mut map, "scale_shift_table", (2, vi), dev);
        if cfg.video.use_keyframes_abs_pos_embedding {
            put(&mut map, "keyframes_abs_pos_embedding", (1, vi), dev);
        }
        put_adaln(&mut map, "adaln_single", vi, 9, dev);
        put_adaln(&mut map, "prompt_adaln_single", vi, 2, dev);
        put_adaln(&mut map, "av_ca_video_scale_shift_adaln_single", vi, 4, dev);
        put_adaln(&mut map, "av_ca_a2v_gate_adaln_single", vi, 1, dev);

        put_linear(&mut map, "audio_patchify_proj", ai, 8, dev);
        put_linear(&mut map, "audio_proj_out", 8, ai, dev);
        put(&mut map, "audio_scale_shift_table", (2, ai), dev);
        put_adaln(&mut map, "audio_adaln_single", ai, 9, dev);
        put_adaln(&mut map, "audio_prompt_adaln_single", ai, 2, dev);
        put_adaln(&mut map, "av_ca_audio_scale_shift_adaln_single", ai, 4, dev);
        put_adaln(&mut map, "av_ca_v2a_gate_adaln_single", ai, 1, dev);

        for i in 0..cfg.video.num_layers {
            let p = format!("transformer_blocks.{i}");
            put_attention(
                &mut map,
                &format!("{p}.attn1"),
                vi,
                cfg.video.num_heads,
                dev,
            );
            put_attention(
                &mut map,
                &format!("{p}.attn2"),
                vi,
                cfg.video.num_heads,
                dev,
            );
            put_ff(&mut map, &format!("{p}.ff"), vi, dev);
            put(&mut map, format!("{p}.scale_shift_table"), (9, vi), dev);
            put(
                &mut map,
                format!("{p}.prompt_scale_shift_table"),
                (2, vi),
                dev,
            );

            put_attention(
                &mut map,
                &format!("{p}.audio_attn1"),
                ai,
                cfg.audio_heads,
                dev,
            );
            put_attention(
                &mut map,
                &format!("{p}.audio_attn2"),
                ai,
                cfg.audio_heads,
                dev,
            );
            put_ff(&mut map, &format!("{p}.audio_ff"), ai, dev);
            put(
                &mut map,
                format!("{p}.audio_scale_shift_table"),
                (9, ai),
                dev,
            );
            put(
                &mut map,
                format!("{p}.audio_prompt_scale_shift_table"),
                (2, ai),
                dev,
            );
            put_attention(
                &mut map,
                &format!("{p}.audio_to_video_attn"),
                cfg.cross_inner,
                cfg.audio_heads,
                dev,
            );
            put_attention(
                &mut map,
                &format!("{p}.video_to_audio_attn"),
                cfg.cross_inner,
                cfg.audio_heads,
                dev,
            );
            put(
                &mut map,
                format!("{p}.scale_shift_table_a2v_ca_audio"),
                (5, ai),
                dev,
            );
            put(
                &mut map,
                format!("{p}.scale_shift_table_a2v_ca_video"),
                (5, vi),
                dev,
            );
        }
        map
    }

    fn inputs(dev: &Device) -> (Tensor, Tensor, Tensor) {
        let latent = Tensor::randn(0f32, 1f32, (1, 2, 8), dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, 12), dev).unwrap();
        let positions = Tensor::from_vec(
            vec![
                0f32, 1., 1., 2., // time
                0., 1., 0., 1., // height
                0., 1., 1., 2., // width
            ],
            (1, 3, 2, 2),
            dev,
        )
        .unwrap();
        (latent, context, positions)
    }

    #[test]
    fn zero_adapter_matches_inference_avdit_video_path() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let map = weights(&cfg, &dev);
        let mut train = LtxDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg.video,
        )
        .unwrap();
        // Install the real adapter surface at its defined zero-delta initialization (A random,
        // B zero). This is stronger than comparing an adapter-free model: every LoraLinear branch
        // executes while the residual remains mathematically zero.
        let suffixes = LTX_ATTN_TARGETS.map(str::to_string);
        let installed = build_lora_targets(&mut train, &suffixes, 2, 2.0, 7, &dev).unwrap();
        assert_eq!(installed.len(), 8 * cfg.video.num_layers);
        let inference = AvDiT::new(VarBuilder::from_tensors(map, DType::F32, &dev), &cfg).unwrap();
        let (latent, context, positions) = inputs(&dev);
        let a = train
            .forward(&latent, 0.37, &context, &positions)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let b = inference
            .forward_video_only(&latent, 0.37, &context, &positions)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 2e-3,
            "train/inference video velocity max diff {diff}"
        );
    }

    fn assert_trained_lora_inference_roundtrip(dev: Device, tag: &str) {
        let cfg = tiny_cfg_25();
        let map = weights(&cfg, &dev);
        let mut train = LtxDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg.video,
        )
        .unwrap();
        let suffixes = LTX_ATTN_TARGETS.map(str::to_string);
        let set = build_lora_targets(&mut train, &suffixes, 2, 2.0, 17, &dev).unwrap();
        let (latent, context, positions) = inputs(&dev);
        // One real optimizer update through the trainable DiT. LoRA starts with B=0, so this first
        // step updates B from a genuine velocity loss and makes the saved adapter nonzero.
        let loss = train
            .forward(&latent, 0.37, &context, &positions)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        opt.step(&grads).unwrap();

        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp
            .path()
            .join(format!("ltx_infer_lora_roundtrip_{tag}.safetensors"));
        let ltx25_metadata = HashMap::from([
            ("lora_rank".to_string(), "2".to_string()),
            ("lora_alpha".to_string(), "2".to_string()),
        ]);
        save_lora_peft(&set, "", &ltx25_metadata, &path).unwrap();

        // The declared header is authoritative on LTX-2.5: a rank mismatch or a missing rank is
        // rejected before a residual can be installed, rather than falling back to the factor shape.
        let wrong_rank_path = path_tmp
            .path()
            .join(format!("ltx_bad_rank_{tag}.safetensors"));
        save_lora_peft(
            &set,
            "",
            &HashMap::from([
                ("lora_rank".to_string(), "3".to_string()),
                ("lora_alpha".to_string(), "2".to_string()),
            ]),
            &wrong_rank_path,
        )
        .unwrap();
        let mut wrong_rank = AvDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg,
        )
        .unwrap();
        let wrong_rank_error = crate::adapters::install_ltx25_adapters(
            &mut wrong_rank,
            &[candle_gen::gen_core::AdapterSpec::new(
                wrong_rank_path.clone(),
                1.0,
                candle_gen::gen_core::AdapterKind::Lora,
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(
            wrong_rank_error.contains("declares lora_rank 3"),
            "{wrong_rank_error}"
        );

        let missing_rank_path = path_tmp
            .path()
            .join(format!("ltx_missing_rank_{tag}.safetensors"));
        save_lora_peft(
            &set,
            "",
            &HashMap::from([("lora_alpha".to_string(), "2".to_string())]),
            &missing_rank_path,
        )
        .unwrap();
        let mut missing_rank = AvDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg,
        )
        .unwrap();
        let missing_rank_error = crate::adapters::install_ltx25_adapters(
            &mut missing_rank,
            &[candle_gen::gen_core::AdapterSpec::new(
                missing_rank_path.clone(),
                1.0,
                candle_gen::gen_core::AdapterKind::Lora,
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(
            missing_rank_error.contains("missing required `lora_rank`"),
            "{missing_rank_error}"
        );
        let spec = candle_gen::gen_core::AdapterSpec::new(
            path.clone(),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        );

        let mut inference = AvDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg,
        )
        .unwrap();
        let base = inference
            .forward_video_only(&latent, 0.37, &context, &positions)
            .unwrap();
        let report =
            crate::adapters::install_ltx25_adapters(&mut inference, std::slice::from_ref(&spec))
                .unwrap();
        assert_eq!(report.applied, 8);
        assert_eq!(report.skipped_keys, 0);
        let adapted = inference
            .forward_video_only(&latent, 0.37, &context, &positions)
            .unwrap();
        let effect = (adapted.clone() - &base)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(effect > 1e-6, "nonzero trained factors must change output");

        // Saving and loading a second copy must reproduce the adapted prediction exactly.  This is
        // intentionally a separate DiT, so bypassing the adapter install cannot borrow the first
        // model's in-memory residuals.
        let mut reloaded = AvDiT::new(
            VarBuilder::from_tensors(map.clone(), DType::F32, &dev),
            &cfg,
        )
        .unwrap();
        crate::adapters::install_ltx25_adapters(&mut reloaded, std::slice::from_ref(&spec))
            .unwrap();
        let reproduced = reloaded
            .forward_video_only(&latent, 0.37, &context, &positions)
            .unwrap();
        let roundtrip_diff = (adapted - reproduced)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            roundtrip_diff < 1e-6,
            "saved LTX-2.5 LoRA must reproduce on reload"
        );

        let mut scale_zero =
            AvDiT::new(VarBuilder::from_tensors(map, DType::F32, &dev), &cfg).unwrap();
        let zero_spec = candle_gen::gen_core::AdapterSpec::new(
            path.clone(),
            0.0,
            candle_gen::gen_core::AdapterKind::Lora,
        );
        crate::adapters::install_ltx25_adapters(&mut scale_zero, &[zero_spec]).unwrap();
        let zero = scale_zero
            .forward_video_only(&latent, 0.37, &context, &positions)
            .unwrap();
        let zero_diff = (zero - base)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(zero_diff, 0.0, "scale=0 must be an exact base no-op");
        std::fs::remove_file(wrong_rank_path).ok();
        std::fs::remove_file(missing_rank_path).ok();
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn trained_lora_reloads_into_inference_and_changes_velocity() {
        assert_trained_lora_inference_roundtrip(Device::Cpu, "cpu");
    }

    /// A valid Eros/distill adapter must not hide a second selected adapter that contributes no LTX
    /// projection residuals. Before the per-spec accounting in `install_ltx_adapters`, the valid
    /// member made the aggregate `report.applied` nonzero and this mixed stack was silently accepted.
    #[test]
    fn mixed_valid_and_invalid_adapter_stack_rejects_zero_apply_member() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut inference = AvDiT::new(
            VarBuilder::from_tensors(weights(&cfg, &dev), DType::F32, &dev),
            &cfg,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let valid_path = dir.path().join("eros-distill.safetensors");
        let invalid_path = dir.path().join("selected-but-unrelated.safetensors");
        candle_gen::candle_core::safetensors::save(
            &HashMap::from([
                (
                    "transformer_blocks.0.attn1.to_q.lora_A.weight".to_owned(),
                    Tensor::ones((1, cfg.video.inner_dim()), DType::F32, &dev).unwrap(),
                ),
                (
                    "transformer_blocks.0.attn1.to_q.lora_B.weight".to_owned(),
                    Tensor::ones((cfg.video.inner_dim(), 1), DType::F32, &dev).unwrap(),
                ),
            ]),
            &valid_path,
        )
        .unwrap();
        candle_gen::candle_core::safetensors::save(
            &HashMap::from([(
                "unrelated_model.embedding.weight".to_owned(),
                Tensor::ones((2, 2), DType::F32, &dev).unwrap(),
            )]),
            &invalid_path,
        )
        .unwrap();

        let specs = [
            candle_gen::gen_core::AdapterSpec::new(
                valid_path,
                1.0,
                candle_gen::gen_core::AdapterKind::Lora,
            ),
            candle_gen::gen_core::AdapterSpec::new(
                invalid_path.clone(),
                1.0,
                candle_gen::gen_core::AdapterKind::Lora,
            ),
        ];
        let error = crate::adapters::install_ltx_adapters(&mut inference, &specs)
            .expect_err("every selected adapter must apply at least one projection")
            .to_string();
        assert!(error.contains("selected adapter #2"), "{error}");
        assert!(
            error.contains(invalid_path.file_name().unwrap().to_str().unwrap()),
            "{error}"
        );
        assert!(
            error.contains("applied zero projection residuals"),
            "{error}"
        );
    }

    /// CUDA execution gate for the actual train→save→inference-load path. Ignored by default because
    /// feature-enabled compile runners need not have a GPU; run explicitly on a CUDA host.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn trained_lora_reloads_into_inference_and_changes_velocity_cuda() {
        assert_trained_lora_inference_roundtrip(Device::new_cuda(0).unwrap(), "cuda");
    }

    #[test]
    fn lora_host_and_segments_cover_every_block() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut model = LtxDiT::new(
            VarBuilder::from_tensors(weights(&cfg, &dev), DType::F32, &dev),
            &cfg.video,
        )
        .unwrap();
        let mut paths = Vec::new();
        model
            .visit_lora_mut(&mut |linear| {
                paths.push(linear.path().to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(paths.len(), 8 * cfg.video.num_layers);
        for suffix in LTX_ATTN_TARGETS {
            assert!(paths.iter().any(|path| path.ends_with(suffix)));
        }

        let (latent, context, positions) = inputs(&dev);
        let (mut hidden, ctx) = model
            .forward_pre_main(&latent, 0.37, &context, &positions)
            .unwrap();
        let segments = model.main_block_segments(&ctx);
        assert_eq!(segments.len(), cfg.video.num_layers);
        for segment in segments {
            hidden = segment(&[hidden]).unwrap().remove(0);
        }
        let segmented = model.velocity_out(&hidden, &ctx).unwrap();
        let dense = model.forward(&latent, 0.37, &context, &positions).unwrap();
        let diff = (segmented - dense)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn backward_reaches_attention_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut model = LtxDiT::new(
            VarBuilder::from_tensors(weights(&cfg, &dev), DType::F32, &dev),
            &cfg.video,
        )
        .unwrap();
        let suffixes = LTX_ATTN_TARGETS.map(str::to_string);
        let set = build_lora_targets(&mut model, &suffixes, 2, 2.0, 7, &dev).unwrap();
        for var in &set.vars {
            var.set(&Tensor::randn(0f32, 0.03f32, var.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (latent, context, positions) = inputs(&dev);
        let (hidden, ctx) = model
            .forward_pre_main(&latent, 0.37, &context, &positions)
            .unwrap();
        let mut segments = model.main_block_segments(&ctx);
        segments.push(Box::new(|state: &[Tensor]| {
            let loss = model
                .velocity_out(&state[0], &ctx)?
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_all()?;
            Ok(vec![loss])
        }));
        let (_loss, grads) =
            checkpointed_backward(&segments, &[hidden.detach()], &set.vars).unwrap();
        let mut nonzero = 0usize;
        for (i, var) in set.vars.iter().enumerate() {
            let grad = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("LoRA factor {i} has no gradient"));
            let max = grad
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(max.is_finite(), "LoRA factor {i} gradient is not finite");
            if max > 1e-9 {
                nonzero += 1;
            }
        }
        assert_eq!(
            nonzero,
            set.vars.len(),
            "some LoRA factors had zero gradients"
        );
    }
}
