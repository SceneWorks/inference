//! `IrisDiT` — the dual-level pixel-space diffusion transformer (`iris3b/models/dit.py` @
//! [`crate::UPSTREAM_CODE_REVISION`]): patch embed + timestep embed + layerwise text adapter →
//! hybrid dual-/single-stream trunk with shared-bias adaLN → timestep re-fusion → PiT pixel head →
//! `fold` back to an RGB velocity. No VAE.
//!
//! Every module mirrors its upstream class one-to-one under the upstream state-dict key names (and
//! the MLX twin `mlx_gen_iris::dit` op for op), so the released `model.safetensors` loads unchanged.
//! Precision: see [`crate::nn`].

use std::collections::HashMap;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::gen_core::iris::{self, ModelConfig};
use candle_gen::{CandleError as Error, Result};

use crate::nn::{
    add, attention, cat, chunk_last, key_padding_mask, modulate, mul, split_at, Checkpoint, Linear,
    Loader, RmsNorm, Rope,
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
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        self.w2.forward(&(gate * self.w3.forward(x)?)?)
    }
}

/// `SelfAttention`: fused qkv, optional per-head QK RMSNorm, optional RoPE, optional key mask.
struct SelfAttention {
    qkv: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    proj: Linear,
    heads: usize,
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
            heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: Option<&Rope>,
        mask: Option<&Tensor>,
        compute: DType,
    ) -> Result<Tensor> {
        let (b, n, dim) = x.dims3()?;
        let head_dim = dim / self.heads;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, n, 3, self.heads, head_dim))?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(2, i, 1)?.squeeze(2)?) };
        let (mut q, mut k, v) = (part(0)?, part(1)?, part(2)?);
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
    fn project(
        &self,
        x: &Tensor,
        heads: usize,
        kv_heads: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, n, _) = x.dims3()?;
        match self {
            Qkv::Fused(qkv) => {
                let out = qkv.forward(x)?;
                let head_dim = out.dim(2)? / (3 * heads);
                let out = out.reshape((b, n, 3, heads, head_dim))?;
                let part = |i: usize| -> Result<Tensor> { Ok(out.narrow(2, i, 1)?.squeeze(2)?) };
                Ok((part(0)?, part(1)?, part(2)?))
            }
            Qkv::Split { q, k, v } => {
                let qo = q.forward(x)?;
                let head_dim = qo.dim(2)? / heads;
                Ok((
                    qo.reshape((b, n, heads, head_dim))?,
                    k.forward(x)?.reshape((b, n, kv_heads, head_dim))?,
                    v.forward(x)?.reshape((b, n, kv_heads, head_dim))?,
                ))
            }
        }
    }
}

/// One stream's adaLN: the model-owned shared core output plus this block's learned bias
/// (`SharedCoreBias`), split as (shift1, scale1, gate1, shift2, scale2, gate2).
struct Modulation {
    bias: Tensor,
}

impl Modulation {
    fn chunks(&self, core_out: &Tensor) -> Result<Vec<Tensor>> {
        chunk_last(&add(core_out, &self.bias)?, 6)
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
    compute: DType,
    device: Device,
    /// Channels of `x` (`model.in_channels`, plus the depth task's extra zero channel).
    input_channels: usize,
    s_embedder: Linear,
    t_mlp0: Linear,
    t_mlp2: Linear,
    y_embedder: TextAdapter,
    y_pos_embedding: Tensor,
    core_img: Linear,
    core_txt: Option<Linear>,
    blocks: Vec<Block>,
    pixel_proj: Linear,
    pixel_blocks: Vec<PitBlock>,
    final_norm: RmsNorm,
    final_linear: Linear,
    pos_cache: Mutex<HashMap<(usize, usize), Tensor>>,
}

/// Shape of one forward's text conditioning.
pub struct TextBatch<'a> {
    /// `[B, T, L, text_dim]` selected-layer states (pad rows zero).
    pub states: &'a Tensor,
    /// `B` rows of `T` 0/1 flags.
    pub mask: &'a [Vec<i32>],
}

impl IrisDiT {
    /// Build from the backbone's `model.safetensors` (upstream key names) at `compute` dtype on
    /// `device`. Every source key must be consumed; a leftover or missing key is a load error.
    pub fn from_checkpoint(
        w: &Checkpoint,
        cfg: &ModelConfig,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::from_checkpoint_widened(w, cfg, compute, device, cfg.in_channels)
    }

    /// [`Self::from_checkpoint`] for a backbone whose input projections were widened to
    /// `input_channels` (`iris3b/downstream/depth.py` `_widen`: the depth task concatenates a zero
    /// channel after RGB). The output stays `model.in_channels` wide.
    pub fn from_checkpoint_widened(
        w: &Checkpoint,
        cfg: &ModelConfig,
        compute: DType,
        device: &Device,
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
            device: device.clone(),
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
        let unused = w.unused_keys();
        if !unused.is_empty() {
            return Err(Error::Msg(format!(
                "iris: the backbone checkpoint carries {} key(s) the configured architecture does \
                 not consume (first: {}); config.yaml and model.safetensors disagree",
                unused.len(),
                unused[0]
            )));
        }
        let stray = w.unconsumed_residuals();
        if !stray.is_empty() {
            return Err(Error::Msg(format!(
                "iris: adapter target(s) {stray:?} are not projections of this backbone's graph"
            )));
        }
        let p = cfg.patch_size;
        crate::nn::expect_shape(
            "s_embedder.proj.weight",
            dit.s_embedder.weight(),
            &[cfg.hidden_size, p * p * input_channels],
        )?;
        crate::nn::expect_shape(
            "pixel_embedder.proj.weight",
            dit.pixel_proj.weight(),
            &[cfg.pixel.hidden_size, input_channels],
        )?;
        Ok(dit)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `TimestepEmbedder`: sinusoid bank (period `timestep_max_period`, 256 frequencies, cos‖sin)
    /// → Linear → SiLU → Linear. `t` is model time `[B]` f32 → `[B, 1, D]`.
    fn timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let freqs = iris::timestep_freqs(self.cfg.timestep_max_period);
        let n = freqs.len();
        let b = t.dim(0)?;
        let freqs = Tensor::from_vec(freqs, (1, n), &self.device)?;
        let phase = t
            .to_dtype(DType::F32)?
            .reshape((b, 1))?
            .broadcast_mul(&freqs)?;
        let emb = Tensor::cat(&[phase.cos()?, phase.sin()?], 1)?;
        let h = self.t_mlp2.forward(&self.t_mlp0.forward(&emb)?.silu()?)?;
        Ok(h.reshape((b, 1, ()))?)
    }

    /// `LayerwiseTextEmbedder.forward(y, mask)` (`y` `[B, T, L, Dt]`) → `[B, T, D]` f32.
    pub fn text_adapter(&self, text: &TextBatch) -> Result<Tensor> {
        let (b, t, layers, dt) = text.states.dims4()?;
        if text.mask.len() != b || text.mask.iter().any(|m| m.len() != t) {
            return Err(Error::Msg(format!(
                "iris: text mask must be [{b}, {t}] for states {:?}",
                text.states.dims()
            )));
        }
        let y_ad = &self.y_embedder;
        let mut y = text
            .states
            .to_dtype(DType::F32)?
            .reshape((b * t, layers, dt))?;
        for block in &y_ad.layer_blocks {
            let h = block
                .attn
                .forward(&block.norm1.forward(&y)?, None, None, self.compute)?;
            y = add(&y, &h)?;
            let m = block
                .mlp2
                .forward(&block.mlp0.forward(&block.norm2.forward(&y)?)?.silu()?)?;
            y = add(&y, &m)?;
        }
        // layer_pool: Linear(L → 1) over the layer axis.
        let pooled = y_ad
            .layer_pool
            .forward(&y.transpose(1, 2)?.contiguous()?)?
            .reshape((b, t, dt))?;
        // refiner (TransformerTextEmbedder): keys masked, every query keeps its own position.
        let key_mask = key_padding_mask(text.mask, t, &self.device)?;
        let mut y = y_ad.proj.forward(&pooled)?;
        for block in &y_ad.blocks {
            let h = block.attn.forward(
                &block.norm1.forward(&y)?,
                None,
                Some(&key_mask),
                self.compute,
            )?;
            y = add(&y, &h)?;
            y = add(&y, &block.mlp.forward(&block.norm2.forward(&y)?)?)?;
        }
        y_ad.norm.forward(&y)
    }

    /// The fixed full-resolution 2D sincos table of `PixelEmbedder` (`[H, W, hidden]` f32), cached.
    fn pixel_pos(&self, height: usize, width: usize) -> Result<Tensor> {
        let mut cache = candle_gen::lock_recover(&self.pos_cache);
        if let Some(table) = cache.get(&(height, width)) {
            return Ok(table.clone());
        }
        let dim = self.cfg.pixel.hidden_size;
        let data = iris::pixel_sincos_table(height, width, dim);
        let table = Tensor::from_vec(data, (height, width, dim), &self.device)?;
        cache.insert((height, width), table.clone());
        Ok(table)
    }

    /// Predict the flow velocity `[B, C, H, W]` (f32) for noisy pixels `x` at model time `t` `[B]`.
    pub fn forward(&self, x: &Tensor, t: &Tensor, text: &TextBatch) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (b, c, h, w) = x.dims4()?;
        let p = cfg.patch_size;
        if c != self.input_channels {
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
        let ts = text.states.dims();
        if ts[0] != b || ts[1] != cfg.text_len {
            return Err(Error::Msg(format!(
                "iris: text states {ts:?} do not match batch {b} / text_len {}",
                cfg.text_len
            )));
        }

        // F.unfold(x, p, stride=p).transpose(1, 2): channel-major patch vectors, row-major patches.
        let patches = x
            .reshape((b, c, hp, p, wp, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, n_patches, c * p * p))?;
        let mut s = self.s_embedder.forward(&patches)?;
        let t_emb = self.timestep_embedding(t)?;
        let cond = t_emb.silu()?;

        let y = self.text_adapter(text)?;
        let n_txt = y.dim(1)?;
        let pos = self
            .y_pos_embedding
            .reshape((1, (), cfg.hidden_size))?
            .narrow(1, 0, n_txt)?;
        let mut y = add(&y, &pos)?;

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rope_img = Rope::grid_2d(
            head_dim,
            hp,
            wp,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
            &self.device,
        )?;
        let rope_txt = Rope::line_1d(head_dim, n_txt, cfg.text_rope_theta as f32, &self.device)?;

        let core_img = self.core_img.forward(&cond)?;
        let core_txt = self
            .core_txt
            .as_ref()
            .map(|core| core.forward(&cond))
            .transpose()?;
        let heads = cfg.num_heads;
        let kv_heads = cfg.num_kv_heads.unwrap_or(cfg.num_heads);
        for block in &self.blocks {
            (s, y) = match block {
                Block::Dual(blk) => {
                    let core_txt = core_txt.as_ref().ok_or_else(|| {
                        Error::Msg(
                            "iris: a dual-stream block needs the text modulation core".into(),
                        )
                    })?;
                    self.dual(
                        blk, &s, &y, &core_img, core_txt, &rope_img, &rope_txt, heads, kv_heads,
                    )?
                }
                Block::Single(blk) => self.single(
                    blk, &s, &y, &core_img, &rope_img, &rope_txt, heads, kv_heads,
                )?,
            };
        }

        // timestep re-fused into every patch token
        let s = add(&t_emb, &s)?.silu()?;
        let s_cond = s.reshape((b * n_patches, ()))?;

        // PixelEmbedder: per-pixel Linear + full-resolution sincos, grouped per patch.
        let tokens = self
            .pixel_proj
            .forward(&x.permute((0, 2, 3, 1))?.contiguous()?)?;
        let pos = self.pixel_pos(h, w)?.to_dtype(tokens.dtype())?;
        let tokens = tokens.broadcast_add(&pos)?;
        let pix = cfg.pixel.hidden_size;
        let mut pixels = tokens
            .reshape((b, hp, p, wp, p, pix))?
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((b * n_patches, p * p, pix))?;
        let pix_head = cfg.pixel.attn_hidden_size / cfg.pixel.num_heads;
        let rope_pix = Rope::grid_2d(
            pix_head,
            hp,
            wp,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
            &self.device,
        )?;
        for block in &self.pixel_blocks {
            pixels = self.pit(block, &pixels, &s_cond, &rope_pix, b, n_patches)?;
        }
        let out = self
            .final_linear
            .forward(&self.final_norm.forward(&pixels)?)?;
        // fold: [B·L, p·p, C] → [B, C, H, W] (C = model.in_channels, whatever x carried)
        let c_out = cfg.in_channels;
        Ok(out
            .reshape((b, hp, wp, p, p, c_out))?
            .permute((0, 5, 1, 3, 2, 4))?
            .contiguous()?
            .reshape((b, c_out, h, w))?
            .to_dtype(DType::F32)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn dual(
        &self,
        blk: &DualBlock,
        x: &Tensor,
        y: &Tensor,
        core_img: &Tensor,
        core_txt: &Tensor,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: usize,
        kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
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
        let n_txt = y.dim(1)?;
        let (out_y, out_x) = split_at(&out, 1, n_txt)?;

        let out_x = mul(&out_x, &sigmoid(&blk.gate_x.forward(&hx)?)?)?;
        let attn_x = blk.proj_x.forward(&out_x)?;
        let mut x = add(x, &mul(&mx[2], &blk.attn_post_norm_x.forward(&attn_x)?)?)?;
        let mlp_in = modulate(&blk.norm_x2.forward(&x)?, &mx[3], &mx[4])?;
        x = add(
            &x,
            &mul(
                &mx[5],
                &blk.mlp_post_norm_x.forward(&blk.mlp_x.forward(&mlp_in)?)?,
            )?,
        )?;
        let Some(text) = &blk.text else {
            return Ok((x, y.clone()));
        };
        let out_y = mul(&out_y, &sigmoid(&text.gate.forward(&hy)?)?)?;
        let attn_y = text.proj.forward(&out_y)?;
        let mut y = add(y, &mul(&my[2], &text.attn_post_norm.forward(&attn_y)?)?)?;
        let mlp_in = modulate(&text.norm2.forward(&y)?, &my[3], &my[4])?;
        y = add(
            &y,
            &mul(
                &my[5],
                &text.mlp_post_norm.forward(&text.mlp.forward(&mlp_in)?)?,
            )?,
        )?;
        Ok((x, y))
    }

    #[allow(clippy::too_many_arguments)]
    fn single(
        &self,
        blk: &SingleBlock,
        x: &Tensor,
        y: &Tensor,
        core: &Tensor,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: usize,
        kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let m = blk.modulation.chunks(core)?;
        let n_txt = y.dim(1)?;
        let tokens = cat(&[y, x], 1)?;
        let h = modulate(&blk.norm1.forward(&tokens)?, &m[0], &m[1])?;
        let (q, k, v) = blk.qkv.project(&h, heads, kv_heads)?;
        let q = blk.q_norm.forward(&q)?;
        let k = blk.k_norm.forward(&k)?;
        let (qy, qx) = split_at(&q, 1, n_txt)?;
        let (ky, kx) = split_at(&k, 1, n_txt)?;
        let q = cat(&[&rope_txt.apply(&qy)?, &rope_img.apply(&qx)?], 1)?;
        let k = cat(&[&rope_txt.apply(&ky)?, &rope_img.apply(&kx)?], 1)?;
        let mut attn = attention(&q, &k, &v, self.compute, None)?;
        let (mut h, mut tokens) = (h, tokens);
        if !blk.text_out {
            attn = split_at(&attn, 1, n_txt)?.1;
            h = split_at(&h, 1, n_txt)?.1;
            tokens = x.clone();
        }
        let attn = mul(&attn, &sigmoid(&blk.attn_gate.forward(&h)?)?)?;
        let attn = blk.attn_post_norm.forward(&blk.attn_proj.forward(&attn)?)?;
        let tokens = add(&tokens, &mul(&m[2], &attn)?)?;
        let mlp = blk
            .mlp
            .forward(&modulate(&blk.norm2.forward(&tokens)?, &m[3], &m[4])?)?;
        let tokens = add(&tokens, &mul(&m[5], &blk.mlp_post_norm.forward(&mlp)?)?)?;
        if !blk.text_out {
            return Ok((tokens, y.clone()));
        }
        let (y, x) = split_at(&tokens, 1, n_txt)?;
        Ok((x, y))
    }

    fn pit(
        &self,
        blk: &PitBlock,
        x: &Tensor,
        cond: &Tensor,
        rope: &Rope,
        batch: usize,
        n_patches: usize,
    ) -> Result<Tensor> {
        let (rows, ppp, pix) = x.dims3()?;
        let mods = blk.adaln.forward(cond)?.reshape((rows, ppp, ()))?;
        let m = chunk_last(&mods, 4)?; // post: (scale1, shift1, scale2, shift2)
                                       // global attention at patch granularity
        let compact = blk
            .compress
            .forward(&blk.norm1.forward(x)?.reshape((rows, ppp * pix))?)?;
        let attn = blk.attn.forward(
            &compact.reshape((batch, n_patches, ()))?,
            Some(rope),
            None,
            self.compute,
        )?;
        let expanded = blk
            .expand
            .forward(&attn.reshape((rows, ()))?)?
            .reshape((rows, ppp, pix))?;
        let x = add(x, &modulate(&expanded, &m[1], &m[0])?)?;
        let mlp = blk
            .fc2
            .forward(&blk.fc1.forward(&blk.norm2.forward(&x)?)?.gelu_erf()?)?;
        add(&x, &modulate(&mlp, &m[3], &m[2])?)
    }
}
