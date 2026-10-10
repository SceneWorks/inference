//! `IrisDiT` — the dual-level pixel-space diffusion transformer (`iris3b/models/dit.py` @
//! [`crate::UPSTREAM_CODE_REVISION`]): patch embed + timestep embed + layerwise text adapter →
//! hybrid dual-/single-stream trunk with shared-bias adaLN → timestep re-fusion → PiT pixel head →
//! `fold` back to an RGB velocity. No VAE.
//!
//! Every module mirrors its upstream class one-to-one under the upstream state-dict key names, so
//! the released `model.safetensors` loads unchanged. Precision: see [`crate::nn`].

use std::collections::HashMap;
use std::sync::Mutex;

use gen_core::iris::ModelConfig;
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::nn::{gelu, sigmoid, silu};
use mlx_rs::{Array, Dtype};

use crate::nn::{
    attention, cat, chunk_last, key_padding_mask, modulate, Linear, Loader, RmsNorm, Rope,
};

/// `SwiGLU`: `w2(silu(w1 x) · w3 x)`, bias-free.
struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    fn load(l: &Loader, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: l.linear(&format!("{prefix}.w1"), false)?,
            w2: l.linear(&format!("{prefix}.w2"), false)?,
            w3: l.linear(&format!("{prefix}.w3"), false)?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(self.w1.forward(x)?)?;
        self.w2.forward(&gate.multiply(&self.w3.forward(x)?)?)
    }
    fn arrays(&self) -> Vec<&Array> {
        [&self.w1, &self.w2, &self.w3]
            .into_iter()
            .flat_map(Linear::arrays)
            .collect()
    }
}

/// `SelfAttention`: fused qkv, optional per-head QK RMSNorm, optional RoPE, optional key mask.
struct SelfAttention {
    qkv: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    proj: Linear,
    heads: i32,
}

impl SelfAttention {
    fn load(l: &Loader, prefix: &str, heads: usize, qk_norm: bool, eps: f32) -> Result<Self> {
        let norm = |name: &str| -> Result<Option<RmsNorm>> {
            qk_norm
                .then(|| l.norm(&format!("{prefix}.{name}"), eps))
                .transpose()
        };
        Ok(Self {
            qkv: l.linear(&format!("{prefix}.qkv"), false)?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            proj: l.linear(&format!("{prefix}.proj"), true)?,
            heads: heads as i32,
        })
    }

    fn forward(
        &self,
        x: &Array,
        rope: Option<&Rope>,
        mask: Option<&Array>,
        compute: Dtype,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, n, dim) = (sh[0], sh[1], sh[2]);
        let head_dim = dim / self.heads;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape(&[b, n, 3, self.heads, head_dim])?;
        let parts = qkv.split(3, 2)?;
        let squeeze = |a: &Array| -> Result<Array> { Ok(a.squeeze_axes(&[2])?) };
        let (mut q, mut k, v) = (
            squeeze(&parts[0])?,
            squeeze(&parts[1])?,
            squeeze(&parts[2])?,
        );
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        if let Some(rope) = rope {
            q = rope.apply(&q)?;
            k = rope.apply(&k)?;
        }
        self.proj.forward(&attention(&q, &k, &v, compute, mask)?)
    }

    fn arrays(&self) -> Vec<&Array> {
        let mut out = self.qkv.arrays();
        out.extend(self.q_norm.iter().flat_map(RmsNorm::arrays));
        out.extend(self.k_norm.iter().flat_map(RmsNorm::arrays));
        out.extend(self.proj.arrays());
        out
    }
}

/// Q/K/V projections of one stream: fused under full MHA, separate under GQA.
enum Qkv {
    Fused(Linear),
    Split { q: Linear, k: Linear, v: Linear },
}

impl Qkv {
    fn load(l: &Loader, fused: &str, q: &str, k: &str, v: &str, gqa: bool) -> Result<Self> {
        Ok(if gqa {
            Qkv::Split {
                q: l.linear(q, false)?,
                k: l.linear(k, false)?,
                v: l.linear(v, false)?,
            }
        } else {
            Qkv::Fused(l.linear(fused, false)?)
        })
    }

    /// `[B, N, D]` → q `[B, N, H, hd]`, k/v `[B, N, KV, hd]`.
    fn project(&self, x: &Array, heads: i32, kv_heads: i32) -> Result<(Array, Array, Array)> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        match self {
            Qkv::Fused(qkv) => {
                let out = qkv.forward(x)?;
                let head_dim = out.shape()[2] / (3 * heads);
                let parts = out.reshape(&[b, n, 3, heads, head_dim])?.split(3, 2)?;
                Ok((
                    parts[0].squeeze_axes(&[2])?,
                    parts[1].squeeze_axes(&[2])?,
                    parts[2].squeeze_axes(&[2])?,
                ))
            }
            Qkv::Split { q, k, v } => {
                let qo = q.forward(x)?;
                let head_dim = qo.shape()[2] / heads;
                Ok((
                    qo.reshape(&[b, n, heads, head_dim])?,
                    k.forward(x)?.reshape(&[b, n, kv_heads, head_dim])?,
                    v.forward(x)?.reshape(&[b, n, kv_heads, head_dim])?,
                ))
            }
        }
    }

    fn arrays(&self) -> Vec<&Array> {
        match self {
            Qkv::Fused(l) => l.arrays(),
            Qkv::Split { q, k, v } => [q, k, v].into_iter().flat_map(Linear::arrays).collect(),
        }
    }
}

/// One stream's adaLN: the model-owned shared core output plus this block's learned bias
/// (`SharedCoreBias`), split as (shift1, scale1, gate1, shift2, scale2, gate2).
struct Modulation {
    bias: Array,
}

impl Modulation {
    fn chunks(&self, core_out: &Array) -> Result<Vec<Array>> {
        chunk_last(&core_out.add(&self.bias)?, 6)
    }
}

/// Text side of a dual-stream block (absent when the checkpoint dropped the final block's text path).
struct DualText {
    norm2: RmsNorm,
    proj: Linear,
    gate: Linear,
    attn_post_norm: RmsNorm,
    mlp: SwiGlu,
    mlp_post_norm: RmsNorm,
}

/// `MMDiTBlock`: separate image/text weights, text-first joint attention.
struct DualBlock {
    norm_x1: RmsNorm,
    norm_x2: RmsNorm,
    norm_y1: RmsNorm,
    qkv_x: Qkv,
    qkv_y: Qkv,
    q_norm_x: RmsNorm,
    k_norm_x: RmsNorm,
    q_norm_y: RmsNorm,
    k_norm_y: RmsNorm,
    proj_x: Linear,
    gate_x: Linear,
    attn_post_norm_x: RmsNorm,
    mlp_x: SwiGlu,
    mlp_post_norm_x: RmsNorm,
    text: Option<DualText>,
    mod_img: Modulation,
    mod_txt: Modulation,
}

/// `SingleStreamBlock`: one weight set over `cat([text, image])`.
struct SingleBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    qkv: Qkv,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attn_gate: Linear,
    attn_proj: Linear,
    attn_post_norm: RmsNorm,
    mlp: SwiGlu,
    mlp_post_norm: RmsNorm,
    modulation: Modulation,
    text_out: bool,
}

enum Block {
    Dual(Box<DualBlock>),
    Single(Box<SingleBlock>),
}

/// `PiTBlock` with `modulation="post"`.
struct PitBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    adaln: Linear,
    compress: Linear,
    expand: Linear,
    attn: SelfAttention,
    fc1: Linear,
    fc2: Linear,
}

/// `LayerwiseAttentionBlock` over one token's encoder-layer states.
struct LayerBlock {
    norm1: RmsNorm,
    attn: SelfAttention,
    norm2: RmsNorm,
    mlp0: Linear,
    mlp2: Linear,
}

/// `TextAdapterBlock` (masked keys).
struct AdapterBlock {
    norm1: RmsNorm,
    attn: SelfAttention,
    norm2: RmsNorm,
    mlp: SwiGlu,
}

/// `LayerwiseTextEmbedder` (`text_adapter="lap_blocks2"`).
struct TextAdapter {
    layer_blocks: Vec<LayerBlock>,
    layer_pool: Linear,
    proj: Linear,
    blocks: Vec<AdapterBlock>,
    norm: RmsNorm,
}

/// The loaded Iris backbone.
pub struct IrisDiT {
    cfg: ModelConfig,
    compute: Dtype,
    /// Channels of `x` (`model.in_channels`, plus the depth task's extra zero channel).
    input_channels: usize,
    s_embedder: Linear,
    t_mlp0: Linear,
    t_mlp2: Linear,
    y_embedder: TextAdapter,
    y_pos_embedding: Array,
    core_img: Linear,
    core_txt: Option<Linear>,
    blocks: Vec<Block>,
    pixel_proj: Linear,
    pixel_blocks: Vec<PitBlock>,
    final_norm: RmsNorm,
    final_linear: Linear,
    pos_cache: Mutex<HashMap<(usize, usize), Array>>,
}

/// Shape of one forward's text conditioning.
pub struct TextBatch<'a> {
    /// `[B, T, L, text_dim]` selected-layer states (pad rows zero).
    pub states: &'a Array,
    /// `B` rows of `T` 0/1 flags.
    pub mask: &'a [Vec<i32>],
}

impl IrisDiT {
    /// Build from the backbone's `model.safetensors` (upstream key names) at `compute` dtype.
    /// Every source key must be consumed; a leftover or missing key is a load error.
    pub fn from_weights(w: &Weights, cfg: &ModelConfig, compute: Dtype) -> Result<Self> {
        Self::from_weights_widened(w, cfg, compute, cfg.in_channels)
    }

    /// [`Self::from_weights`] for a backbone whose input projections were widened to
    /// `input_channels` (`iris3b/downstream/depth.py` `_widen`: the depth task concatenates a zero
    /// channel after RGB). The output stays `model.in_channels` wide.
    pub fn from_weights_widened(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        input_channels: usize,
    ) -> Result<Self> {
        let l = Loader {
            weights: w,
            compute,
        };
        let eps = cfg.norm_eps as f32;
        let gqa = cfg.num_kv_heads.is_some_and(|kv| kv != cfg.num_heads);
        let core_img = l.linear("modulation_cores.adaln_img", true)?;
        let core_txt = if cfg.dual_depth > 0 {
            Some(l.linear("modulation_cores.adaln_txt", true)?)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = |name: &str| format!("blocks.{i}.{name}");
            let text_out = cfg.final_block_text == "keep" || i + 1 < cfg.depth;
            let bias = |name: &str| -> Result<Modulation> {
                Ok(Modulation {
                    bias: l.f32(&p(name))?,
                })
            };
            if i < cfg.dual_depth {
                let text = if text_out {
                    Some(DualText {
                        norm2: l.norm(&p("norm_y2"), eps)?,
                        proj: l.linear(&p("attn.proj_y"), true)?,
                        gate: l.linear(&p("attn_gate_y"), false)?,
                        attn_post_norm: l.norm(&p("attn_post_norm_y"), eps)?,
                        mlp: SwiGlu::load(&l, &p("mlp_y"))?,
                        mlp_post_norm: l.norm(&p("mlp_post_norm_y"), eps)?,
                    })
                } else {
                    None
                };
                blocks.push(Block::Dual(Box::new(DualBlock {
                    norm_x1: l.norm(&p("norm_x1"), eps)?,
                    norm_x2: l.norm(&p("norm_x2"), eps)?,
                    norm_y1: l.norm(&p("norm_y1"), eps)?,
                    qkv_x: Qkv::load(
                        &l,
                        &p("attn.qkv_x"),
                        &p("attn.q_proj_x"),
                        &p("attn.k_proj_x"),
                        &p("attn.v_proj_x"),
                        gqa,
                    )?,
                    qkv_y: Qkv::load(
                        &l,
                        &p("attn.qkv_y"),
                        &p("attn.q_proj_y"),
                        &p("attn.k_proj_y"),
                        &p("attn.v_proj_y"),
                        gqa,
                    )?,
                    q_norm_x: l.norm(&p("attn.q_norm_x"), eps)?,
                    k_norm_x: l.norm(&p("attn.k_norm_x"), eps)?,
                    q_norm_y: l.norm(&p("attn.q_norm_y"), eps)?,
                    k_norm_y: l.norm(&p("attn.k_norm_y"), eps)?,
                    proj_x: l.linear(&p("attn.proj_x"), true)?,
                    gate_x: l.linear(&p("attn_gate_x"), false)?,
                    attn_post_norm_x: l.norm(&p("attn_post_norm_x"), eps)?,
                    mlp_x: SwiGlu::load(&l, &p("mlp_x"))?,
                    mlp_post_norm_x: l.norm(&p("mlp_post_norm_x"), eps)?,
                    text,
                    mod_img: bias("adaln_img.bias")?,
                    mod_txt: bias("adaln_txt.bias")?,
                })));
            } else {
                blocks.push(Block::Single(Box::new(SingleBlock {
                    norm1: l.norm(&p("norm1"), eps)?,
                    norm2: l.norm(&p("norm2"), eps)?,
                    qkv: Qkv::load(&l, &p("qkv"), &p("q_proj"), &p("k_proj"), &p("v_proj"), gqa)?,
                    q_norm: l.norm(&p("q_norm"), eps)?,
                    k_norm: l.norm(&p("k_norm"), eps)?,
                    attn_gate: l.linear(&p("attn_gate"), false)?,
                    attn_proj: l.linear(&p("attn_proj"), true)?,
                    attn_post_norm: l.norm(&p("attn_post_norm"), eps)?,
                    mlp: SwiGlu::load(&l, &p("mlp"))?,
                    mlp_post_norm: l.norm(&p("mlp_post_norm"), eps)?,
                    modulation: bias("adaln.bias")?,
                    text_out,
                })));
            }
        }
        let mut layer_blocks = Vec::with_capacity(2);
        for i in 0..2 {
            let p = |name: &str| format!("y_embedder.layer_blocks.{i}.{name}");
            layer_blocks.push(LayerBlock {
                norm1: l.norm(&p("norm1"), eps)?,
                attn: SelfAttention::load(&l, &p("attn"), cfg.text_lap_num_heads, false, eps)?,
                norm2: l.norm(&p("norm2"), eps)?,
                mlp0: l.linear(&p("mlp.0"), true)?,
                mlp2: l.linear(&p("mlp.2"), true)?,
            });
        }
        let mut adapter_blocks = Vec::with_capacity(2);
        for i in 0..2 {
            let p = |name: &str| format!("y_embedder.refiner.blocks.{i}.{name}");
            adapter_blocks.push(AdapterBlock {
                norm1: l.norm(&p("norm1"), eps)?,
                attn: SelfAttention::load(&l, &p("attn"), cfg.num_heads, cfg.qk_norm, eps)?,
                norm2: l.norm(&p("norm2"), eps)?,
                mlp: SwiGlu::load(&l, &p("mlp"))?,
            });
        }
        let y_embedder = TextAdapter {
            layer_blocks,
            layer_pool: l.linear("y_embedder.layer_pool", true)?,
            proj: l.linear("y_embedder.refiner.proj", true)?,
            blocks: adapter_blocks,
            norm: l.norm("y_embedder.refiner.norm", eps)?,
        };

        let mut pixel_blocks = Vec::with_capacity(cfg.pixel.depth);
        for i in 0..cfg.pixel.depth {
            let p = |name: &str| format!("pixel_blocks.{i}.{name}");
            pixel_blocks.push(PitBlock {
                norm1: l.norm(&p("norm1"), eps)?,
                norm2: l.norm(&p("norm2"), eps)?,
                adaln: l.linear(&p("adaln"), true)?,
                compress: l.linear(&p("compress"), true)?,
                expand: l.linear(&p("expand"), true)?,
                attn: SelfAttention::load(&l, &p("attn"), cfg.pixel.num_heads, cfg.qk_norm, eps)?,
                fc1: l.linear(&p("mlp.fc1"), true)?,
                fc2: l.linear(&p("mlp.fc2"), true)?,
            });
        }

        let dit = Self {
            cfg: cfg.clone(),
            compute,
            input_channels,
            s_embedder: l.linear("s_embedder.proj", true)?,
            t_mlp0: l.linear("t_embedder.mlp.0", true)?,
            t_mlp2: l.linear("t_embedder.mlp.2", true)?,
            y_embedder,
            y_pos_embedding: l.f32("y_pos_embedding")?,
            core_img,
            core_txt,
            blocks,
            pixel_proj: l.linear("pixel_embedder.proj", true)?,
            pixel_blocks,
            final_norm: l.norm("final_layer.norm", eps)?,
            final_linear: l.linear("final_layer.linear", true)?,
            pos_cache: Mutex::new(HashMap::new()),
        };
        let mut unused = w.unused_keys();
        unused.sort_unstable();
        if !unused.is_empty() {
            return Err(Error::Msg(format!(
                "iris: the backbone checkpoint carries {} key(s) the configured architecture does \
                 not consume (first: {}); config.yaml and model.safetensors disagree",
                unused.len(),
                unused[0]
            )));
        }
        let p = cfg.patch_size as i32;
        crate::nn::expect_shape(
            "s_embedder.proj.weight",
            &dit.s_embedder.arrays()[0].clone(),
            &[cfg.hidden_size as i32, p * p * input_channels as i32],
        )?;
        crate::nn::expect_shape(
            "pixel_embedder.proj.weight",
            &dit.pixel_proj.arrays()[0].clone(),
            &[cfg.pixel.hidden_size as i32, input_channels as i32],
        )?;
        Ok(dit)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.compute
    }

    /// Every parameter array, for an eager evaluation at load (never inside a forward).
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        out.extend(self.s_embedder.arrays());
        out.extend(self.t_mlp0.arrays());
        out.extend(self.t_mlp2.arrays());
        let y = &self.y_embedder;
        for b in &y.layer_blocks {
            out.extend(b.norm1.arrays());
            out.extend(b.attn.arrays());
            out.extend(b.norm2.arrays());
            out.extend(b.mlp0.arrays());
            out.extend(b.mlp2.arrays());
        }
        out.extend(y.layer_pool.arrays());
        out.extend(y.proj.arrays());
        for b in &y.blocks {
            out.extend(b.norm1.arrays());
            out.extend(b.attn.arrays());
            out.extend(b.norm2.arrays());
            out.extend(b.mlp.arrays());
        }
        out.extend(y.norm.arrays());
        out.push(&self.y_pos_embedding);
        out.extend(self.core_img.arrays());
        out.extend(self.core_txt.iter().flat_map(Linear::arrays));
        for block in &self.blocks {
            match block {
                Block::Dual(b) => {
                    for n in [
                        &b.norm_x1,
                        &b.norm_x2,
                        &b.norm_y1,
                        &b.q_norm_x,
                        &b.k_norm_x,
                        &b.q_norm_y,
                        &b.k_norm_y,
                        &b.attn_post_norm_x,
                        &b.mlp_post_norm_x,
                    ] {
                        out.extend(n.arrays());
                    }
                    out.extend(b.qkv_x.arrays());
                    out.extend(b.qkv_y.arrays());
                    out.extend(b.proj_x.arrays());
                    out.extend(b.gate_x.arrays());
                    out.extend(b.mlp_x.arrays());
                    if let Some(t) = &b.text {
                        out.extend(t.norm2.arrays());
                        out.extend(t.proj.arrays());
                        out.extend(t.gate.arrays());
                        out.extend(t.attn_post_norm.arrays());
                        out.extend(t.mlp.arrays());
                        out.extend(t.mlp_post_norm.arrays());
                    }
                    out.push(&b.mod_img.bias);
                    out.push(&b.mod_txt.bias);
                }
                Block::Single(b) => {
                    for n in [
                        &b.norm1,
                        &b.norm2,
                        &b.q_norm,
                        &b.k_norm,
                        &b.attn_post_norm,
                        &b.mlp_post_norm,
                    ] {
                        out.extend(n.arrays());
                    }
                    out.extend(b.qkv.arrays());
                    out.extend(b.attn_gate.arrays());
                    out.extend(b.attn_proj.arrays());
                    out.extend(b.mlp.arrays());
                    out.push(&b.modulation.bias);
                }
            }
        }
        out.extend(self.pixel_proj.arrays());
        for b in &self.pixel_blocks {
            out.extend(b.norm1.arrays());
            out.extend(b.norm2.arrays());
            out.extend(b.adaln.arrays());
            out.extend(b.compress.arrays());
            out.extend(b.expand.arrays());
            out.extend(b.attn.arrays());
            out.extend(b.fc1.arrays());
            out.extend(b.fc2.arrays());
        }
        out.extend(self.final_norm.arrays());
        out.extend(self.final_linear.arrays());
        out
    }

    /// `TimestepEmbedder`: sinusoid bank (period `timestep_max_period`, 256 frequencies, cos‖sin)
    /// → Linear → SiLU → Linear. `t` is model time `[B]` f32 → `[B, 1, D]`.
    fn timestep_embedding(&self, t: &Array) -> Result<Array> {
        const FREQ_DIM: usize = 256;
        let n = FREQ_DIM / 2;
        let neg_log = -(self.cfg.timestep_max_period.ln()) as f32;
        let freqs: Vec<f32> = (0..n)
            .map(|k| (neg_log * k as f32 / n as f32).exp())
            .collect();
        let b = t.shape()[0];
        let freqs = Array::from_slice(&freqs, &[1, n as i32]);
        let phase = t
            .as_dtype(Dtype::Float32)?
            .reshape(&[b, 1])?
            .multiply(&freqs)?;
        let emb = cat(&[&phase.cos()?, &phase.sin()?], -1)?;
        let h = self.t_mlp2.forward(&silu(self.t_mlp0.forward(&emb)?)?)?;
        Ok(h.reshape(&[b, 1, -1])?)
    }

    /// `LayerwiseTextEmbedder.forward(y, mask)` (`y` `[B, T, L, Dt]`) → `[B, T, D]` f32.
    pub fn text_adapter(&self, text: &TextBatch) -> Result<Array> {
        let sh = text.states.shape();
        let (b, t, layers, dt) = (sh[0], sh[1], sh[2], sh[3]);
        if text.mask.len() != b as usize || text.mask.iter().any(|m| m.len() != t as usize) {
            return Err(Error::Msg(format!(
                "iris: text mask must be [{b}, {t}] for states {sh:?}"
            )));
        }
        let y_ad = &self.y_embedder;
        let mut y = text
            .states
            .as_dtype(Dtype::Float32)?
            .reshape(&[b * t, layers, dt])?;
        for block in &y_ad.layer_blocks {
            let h = block
                .attn
                .forward(&block.norm1.forward(&y)?, None, None, self.compute)?;
            y = y.add(&h)?;
            let m = block
                .mlp2
                .forward(&silu(block.mlp0.forward(&block.norm2.forward(&y)?)?)?)?;
            y = y.add(&m)?;
        }
        // layer_pool: Linear(L → 1) over the layer axis.
        let pooled = y_ad
            .layer_pool
            .forward(&y.transpose_axes(&[0, 2, 1])?)?
            .reshape(&[b, t, dt])?;
        // refiner (TransformerTextEmbedder): keys masked, every query keeps its own position.
        let key_mask = key_padding_mask(text.mask, t as usize);
        let mut y = y_ad.proj.forward(&pooled)?;
        for block in &y_ad.blocks {
            let h = block.attn.forward(
                &block.norm1.forward(&y)?,
                None,
                Some(&key_mask),
                self.compute,
            )?;
            y = y.add(&h)?;
            y = y.add(&block.mlp.forward(&block.norm2.forward(&y)?)?)?;
        }
        y_ad.norm.forward(&y)
    }

    /// The fixed full-resolution 2D sincos table of `PixelEmbedder` (`[H, W, hidden]` f32), cached.
    fn pixel_pos(&self, height: usize, width: usize) -> Result<Array> {
        let mut cache = self
            .pos_cache
            .lock()
            .map_err(|_| Error::Msg("iris: pixel position cache poisoned".into()))?;
        if let Some(table) = cache.get(&(height, width)) {
            return Ok(table.clone());
        }
        let dim = self.cfg.pixel.hidden_size;
        let half = dim / 2; // per axis
        let quarter = half / 2; // frequencies per axis
        let omega: Vec<f64> = (0..quarter)
            .map(|k| 1.0 / 10_000f64.powf(k as f64 / (half as f64 / 2.0)))
            .collect();
        let mut data = vec![0f32; height * width * dim];
        for r in 0..height {
            for c in 0..width {
                let row = &mut data[(r * width + c) * dim..(r * width + c + 1) * dim];
                for (k, w) in omega.iter().enumerate() {
                    let (xc, yr) = (c as f64 * w, r as f64 * w);
                    row[k] = xc.sin() as f32;
                    row[quarter + k] = xc.cos() as f32;
                    row[half + k] = yr.sin() as f32;
                    row[half + quarter + k] = yr.cos() as f32;
                }
            }
        }
        let table = Array::from_slice(&data, &[height as i32, width as i32, dim as i32]);
        cache.insert((height, width), table.clone());
        Ok(table)
    }

    /// Predict the flow velocity `[B, C, H, W]` (f32) for noisy pixels `x` at model time `t` `[B]`.
    pub fn forward(&self, x: &Array, t: &Array, text: &TextBatch) -> Result<Array> {
        let cfg = &self.cfg;
        let sh = x.shape();
        let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let p = cfg.patch_size as i32;
        if c as usize != self.input_channels {
            return Err(Error::Msg(format!(
                "iris: input has {c} channels, the backbone embeds {}",
                self.input_channels
            )));
        }
        if h % p != 0 || w % p != 0 {
            return Err(Error::Msg(format!(
                "iris: input {h}x{w} is not divisible by patch_size {p}"
            )));
        }
        let (hp, wp) = (h / p, w / p);
        let n_patches = hp * wp;
        let ts = text.states.shape();
        if ts[0] != b || ts[1] as usize != cfg.text_len {
            return Err(Error::Msg(format!(
                "iris: text states {ts:?} do not match batch {b} / text_len {}",
                cfg.text_len
            )));
        }

        // F.unfold(x, p, stride=p).transpose(1, 2): channel-major patch vectors, row-major patches.
        let patches = x
            .reshape(&[b, c, hp, p, wp, p])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, n_patches, c * p * p])?;
        let mut s = self.s_embedder.forward(&patches)?;
        let t_emb = self.timestep_embedding(t)?;
        let cond = silu(&t_emb)?;

        let mut y = self.text_adapter(text)?;
        let n_txt = y.shape()[1];
        let pos = self
            .y_pos_embedding
            .reshape(&[1, -1, cfg.hidden_size as i32])?
            .split_axis(&[n_txt], 1)?
            .swap_remove(0);
        y = y.add(&pos)?;

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rope_img = Rope::grid_2d(
            head_dim,
            hp as usize,
            wp as usize,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
        );
        let rope_txt = Rope::line_1d(head_dim, n_txt as usize, cfg.text_rope_theta as f32);

        let core_img = self.core_img.forward(&cond)?;
        let core_txt = self
            .core_txt
            .as_ref()
            .map(|core| core.forward(&cond))
            .transpose()?;
        let heads = cfg.num_heads as i32;
        let kv_heads = cfg.num_kv_heads.unwrap_or(cfg.num_heads) as i32;
        for block in &self.blocks {
            (s, y) = match block {
                Block::Dual(blk) => self.dual(
                    blk,
                    &s,
                    &y,
                    &core_img,
                    core_txt.as_ref().expect("dual blocks carry a text core"),
                    &rope_img,
                    &rope_txt,
                    heads,
                    kv_heads,
                )?,
                Block::Single(blk) => self.single(
                    blk, &s, &y, &core_img, &rope_img, &rope_txt, heads, kv_heads,
                )?,
            };
        }

        // timestep re-fused into every patch token
        let s = silu(&t_emb.add(&s)?)?;
        let s_cond = s.reshape(&[b * n_patches, -1])?;

        // PixelEmbedder: per-pixel Linear + full-resolution sincos, grouped per patch.
        let tokens = self.pixel_proj.forward(&x.transpose_axes(&[0, 2, 3, 1])?)?;
        let tokens = tokens.add(
            &self
                .pixel_pos(h as usize, w as usize)?
                .as_dtype(tokens.dtype())?,
        )?;
        let pix = cfg.pixel.hidden_size as i32;
        let mut pixels = tokens
            .reshape(&[b, hp, p, wp, p, pix])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b * n_patches, p * p, pix])?;
        let pix_head = cfg.pixel.attn_hidden_size / cfg.pixel.num_heads;
        let rope_pix = Rope::grid_2d(
            pix_head,
            hp as usize,
            wp as usize,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
        );
        for block in &self.pixel_blocks {
            pixels = self.pit(block, &pixels, &s_cond, &rope_pix, b, n_patches)?;
        }
        let out = self
            .final_linear
            .forward(&self.final_norm.forward(&pixels)?)?;
        // fold: [B·L, p·p, C] → [B, C, H, W] (C = model.in_channels, whatever x carried)
        let c_out = cfg.in_channels as i32;
        Ok(out
            .reshape(&[b, hp, wp, p, p, c_out])?
            .transpose_axes(&[0, 5, 1, 3, 2, 4])?
            .reshape(&[b, c_out, h, w])?
            .as_dtype(Dtype::Float32)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn dual(
        &self,
        blk: &DualBlock,
        x: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: &Array,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: i32,
        kv_heads: i32,
    ) -> Result<(Array, Array)> {
        let mx = blk.mod_img.chunks(core_img)?;
        let my = blk.mod_txt.chunks(core_txt)?;
        let hx = modulate(&blk.norm_x1.forward(x)?, &mx[0], &mx[1])?;
        let hy = modulate(&blk.norm_y1.forward(y)?, &my[0], &my[1])?;
        let (qx, kx, vx) = blk.qkv_x.project(&hx, heads, kv_heads)?;
        let (qy, ky, vy) = blk.qkv_y.project(&hy, heads, kv_heads)?;
        let qx = rope_img.apply(&blk.q_norm_x.forward(&qx)?)?;
        let kx = rope_img.apply(&blk.k_norm_x.forward(&kx)?)?;
        let qy = rope_txt.apply(&blk.q_norm_y.forward(&qy)?)?;
        let ky = rope_txt.apply(&blk.k_norm_y.forward(&ky)?)?;
        let q = cat(&[&qy, &qx], 1)?;
        let k = cat(&[&ky, &kx], 1)?;
        let v = cat(&[&vy, &vx], 1)?;
        let out = attention(&q, &k, &v, self.compute, None)?;
        let n_txt = y.shape()[1];
        let mut halves = out.split_axis(&[n_txt], 1)?;
        let out_x = halves.swap_remove(1);
        let out_y = halves.swap_remove(0);

        let out_x = out_x.multiply(&sigmoid(blk.gate_x.forward(&hx)?)?)?;
        let attn_x = blk.proj_x.forward(&out_x)?;
        let mut x = x.add(&mx[2].multiply(&blk.attn_post_norm_x.forward(&attn_x)?)?)?;
        let mlp_in = modulate(&blk.norm_x2.forward(&x)?, &mx[3], &mx[4])?;
        x = x.add(&mx[5].multiply(&blk.mlp_post_norm_x.forward(&blk.mlp_x.forward(&mlp_in)?)?)?)?;
        let Some(text) = &blk.text else {
            return Ok((x, y.clone()));
        };
        let out_y = out_y.multiply(&sigmoid(text.gate.forward(&hy)?)?)?;
        let attn_y = text.proj.forward(&out_y)?;
        let mut y = y.add(&my[2].multiply(&text.attn_post_norm.forward(&attn_y)?)?)?;
        let mlp_in = modulate(&text.norm2.forward(&y)?, &my[3], &my[4])?;
        y = y.add(&my[5].multiply(&text.mlp_post_norm.forward(&text.mlp.forward(&mlp_in)?)?)?)?;
        Ok((x, y))
    }

    #[allow(clippy::too_many_arguments)]
    fn single(
        &self,
        blk: &SingleBlock,
        x: &Array,
        y: &Array,
        core: &Array,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: i32,
        kv_heads: i32,
    ) -> Result<(Array, Array)> {
        let m = blk.modulation.chunks(core)?;
        let n_txt = y.shape()[1];
        let tokens = cat(&[y, x], 1)?;
        let h = modulate(&blk.norm1.forward(&tokens)?, &m[0], &m[1])?;
        let (q, k, v) = blk.qkv.project(&h, heads, kv_heads)?;
        let q = blk.q_norm.forward(&q)?;
        let k = blk.k_norm.forward(&k)?;
        let mut qs = q.split_axis(&[n_txt], 1)?;
        let mut ks = k.split_axis(&[n_txt], 1)?;
        let (qx, qy) = (qs.swap_remove(1), qs.swap_remove(0));
        let (kx, ky) = (ks.swap_remove(1), ks.swap_remove(0));
        let q = cat(&[&rope_txt.apply(&qy)?, &rope_img.apply(&qx)?], 1)?;
        let k = cat(&[&rope_txt.apply(&ky)?, &rope_img.apply(&kx)?], 1)?;
        let mut attn = attention(&q, &k, &v, self.compute, None)?;
        let (mut h, mut tokens) = (h, tokens);
        if !blk.text_out {
            attn = attn.split_axis(&[n_txt], 1)?.swap_remove(1);
            h = h.split_axis(&[n_txt], 1)?.swap_remove(1);
            tokens = x.clone();
        }
        let attn = attn.multiply(&sigmoid(blk.attn_gate.forward(&h)?)?)?;
        let attn = blk.attn_post_norm.forward(&blk.attn_proj.forward(&attn)?)?;
        let tokens = tokens.add(&m[2].multiply(&attn)?)?;
        let mlp = blk
            .mlp
            .forward(&modulate(&blk.norm2.forward(&tokens)?, &m[3], &m[4])?)?;
        let tokens = tokens.add(&m[5].multiply(&blk.mlp_post_norm.forward(&mlp)?)?)?;
        if !blk.text_out {
            return Ok((tokens, y.clone()));
        }
        let mut parts = tokens.split_axis(&[n_txt], 1)?;
        let x = parts.swap_remove(1);
        let y = parts.swap_remove(0);
        Ok((x, y))
    }

    fn pit(
        &self,
        blk: &PitBlock,
        x: &Array,
        cond: &Array,
        rope: &Rope,
        batch: i32,
        n_patches: i32,
    ) -> Result<Array> {
        let sh = x.shape();
        let (rows, ppp, pix) = (sh[0], sh[1], sh[2]);
        let mods = blk.adaln.forward(cond)?.reshape(&[rows, ppp, -1])?;
        let m = chunk_last(&mods, 4)?; // post: (scale1, shift1, scale2, shift2)
                                       // global attention at patch granularity
        let compact = blk
            .compress
            .forward(&blk.norm1.forward(x)?.reshape(&[rows, ppp * pix])?)?;
        let attn = blk.attn.forward(
            &compact.reshape(&[batch, n_patches, -1])?,
            Some(rope),
            None,
            self.compute,
        )?;
        let expanded = blk
            .expand
            .forward(&attn.reshape(&[rows, -1])?)?
            .reshape(&[rows, ppp, pix])?;
        let x = x.add(&modulate(&expanded, &m[1], &m[0])?)?;
        let mlp = blk
            .fc2
            .forward(&gelu(blk.fc1.forward(&blk.norm2.forward(&x)?)?)?)?;
        Ok(x.add(&modulate(&mlp, &m[3], &m[2])?)?)
    }
}
