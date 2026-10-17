//! `IrisDiT` — the dual-level pixel-space diffusion transformer (`iris3b/models/dit.py` @
//! [`crate::UPSTREAM_CODE_REVISION`]): patch embed + timestep embed + layerwise text adapter →
//! hybrid dual-/single-stream trunk with shared-bias adaLN → timestep re-fusion → PiT pixel head →
//! `fold` back to an RGB velocity. No VAE.
//!
//! Every module mirrors its upstream class one-to-one under the upstream state-dict key names, so
//! the released `model.safetensors` loads unchanged. Precision: see [`crate::nn`].

use std::collections::HashMap;
use std::sync::Mutex;

use gen_core::iris::{self, ModelConfig};
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::nn::{gelu, sigmoid, silu};
use mlx_rs::{Array, Dtype};

use crate::retention::{Attention, Kind, NormInput, Retention};

use crate::nn::{
    attention_at_site, barrier, cat, chunk_last, key_padding_mask, modulate, AdaptedLinears,
    BackboneQuant, Linear, Loader, RmsNorm, Rope,
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
    /// The loaded gated width (`w1`'s output).
    fn hidden(&self) -> u64 {
        self.w1.out_features()
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
        plan: mlx_gen::AttentionPlan<'_>,
        site: &'static str,
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
        barrier(&[&q, &k, &v])?;
        self.proj
            .forward(&attention_at_site(&q, &k, &v, compute, mask, plan, site)?)
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
// Built once per block at load and never moved on the forward path, so the variant size (three
// adaptable projections vs one) costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
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

impl Block {
    /// `(model width, widest MLP hidden)` of THIS block's loaded tensors — the iOS quantum
    /// accounting's inputs, never production constants.
    fn widths(&self) -> (u64, u64) {
        match self {
            Block::Dual(b) => {
                let ff_hidden = b
                    .text
                    .as_ref()
                    .map_or(b.mlp_x.hidden(), |t| b.mlp_x.hidden().max(t.mlp.hidden()));
                (b.norm_x1.width(), ff_hidden)
            }
            Block::Single(b) => (b.norm1.width(), b.mlp.hidden()),
        }
    }

    /// `(separate q / k / v projections, text path, down-projection input padding)` of THIS
    /// block's loaded modules — the iOS quantum accounting's layout inputs.
    fn layout(&self) -> (bool, bool, u64) {
        match self {
            Block::Dual(b) => {
                let pad = b.text.as_ref().map_or(b.mlp_x.w2.input_pad(), |t| {
                    b.mlp_x.w2.input_pad().max(t.mlp.w2.input_pad())
                });
                (matches!(b.qkv_x, Qkv::Split { .. }), b.text.is_some(), pad)
            }
            Block::Single(b) => (
                matches!(b.qkv, Qkv::Split { .. }),
                b.text_out,
                b.mlp.w2.input_pad(),
            ),
        }
    }
}

/// Labels of the iOS execution quanta of one backbone forward, in forward order.
pub(crate) mod quanta {
    pub const EMBED: &str = "Iris patch/timestep embed";
    pub const TEXT_LAYER_BLOCK: &str = "Iris text adapter layer block";
    pub const TEXT_POOL: &str = "Iris text adapter pool";
    pub const TEXT_REFINER_BLOCK: &str = "Iris text refiner block";
    pub const TEXT_OUTPUT: &str = "Iris text adapter output";
    pub const TRUNK_BLOCK: &str = "Iris trunk block";
    pub const PIXEL_EMBED: &str = "Iris pixel embed";
    pub const PIXEL_BLOCK: &str = "Iris pixel block";
    pub const OUTPUT: &str = "Iris output projection";
}

/// The geometry of one trunk block quantum (see [`IrisDiT::trunk_block_scratch`]): the request's
/// token counts and plan over the loaded block's widths.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TrunkGeometry {
    pub compute_bytes: u64,
    pub batch: u64,
    /// Image (patch) tokens per batch row.
    pub img_tokens: u64,
    /// Text tokens per batch row.
    pub txt_tokens: u64,
    pub dim: u64,
    pub heads: u64,
    pub kv_heads: u64,
    /// The block's widest SwiGLU hidden width.
    pub ff: u64,
    /// Query rows per SDPA call (the plan's block; all tokens when unbounded).
    pub query_rows: u64,
    pub eval_per_chunk: bool,
    /// Grouped-query attention's separate q / k / v projections (else one fused qkv).
    pub split_qkv: bool,
    /// Dual-stream: the block has a text path. Single-stream: the block returns updated text.
    pub text: bool,
    /// Zero columns appended to the SwiGLU down projection's input (an unaligned packed base).
    pub down_pad: u64,
    /// Bytes per element of the image stream entering the block: the compute width at the first
    /// block (the patch embedding's output), f32 after it (every residual sum promotes to f32).
    pub img_stream_bytes: u64,
}

/// The widths of the patch / timestep embedding quantum (see [`IrisDiT::embed_scratch`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbedGeometry {
    pub compute_bytes: u64,
    pub batch: u64,
    pub patches: u64,
    /// `p² · C`, the patch embedding's input width.
    pub patch_in: u64,
    pub hidden: u64,
    /// The timestep sinusoid width (`cos ‖ sin`, the first timestep projection's input).
    pub sinusoid: u64,
    /// The first timestep projection's output width.
    pub t_hidden: u64,
    /// The second timestep projection's output width (`t_emb`).
    pub t_out: u64,
    /// Both modulation cores' output widths summed.
    pub cores: u64,
}

/// The widths of one PiT block quantum (see [`IrisDiT::pixel_block_scratch`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelBlockGeometry {
    pub compute_bytes: u64,
    pub batch: u64,
    pub patches: u64,
    /// `p²` pixels per patch.
    pub patch_pixels: u64,
    pub pixel_width: u64,
    pub compact_width: u64,
    /// `adaln`'s output width (`4 · p² · pixel width`).
    pub modulation_width: u64,
    /// `adaln`'s input width (the per-patch conditioning).
    pub condition_width: u64,
    pub fc1_width: u64,
    pub heads: u64,
    pub query_rows: u64,
    pub eval_per_chunk: bool,
}

/// The geometry of one text-adapter block quantum: a layerwise block (`masked == false`: attention
/// over each token's `tokens` encoder layers, no QK-norm) or a refiner block (`masked`: masked-key
/// attention over the text tokens, QK-norm, SwiGLU).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextBlockGeometry {
    pub compute_bytes: u64,
    /// Attention batch rows (`B · T` for a layerwise block, `B` for a refiner block).
    pub batch: u64,
    /// Tokens per row (encoder layers, or text tokens).
    pub tokens: u64,
    pub dim: u64,
    pub heads: u64,
    pub ff: u64,
    pub query_rows: u64,
    pub eval_per_chunk: bool,
    /// Refiner block: QK-norm, an additive key mask and a SwiGLU MLP (else the layerwise block's
    /// plain attention and `mlp.0 → SiLU → mlp.2`).
    pub refiner: bool,
    pub down_pad: u64,
    /// Bytes per element of the residual stream: f32 through the layerwise blocks (the f32
    /// encoder states), the compute width through the refiner (the pool's projection output).
    pub stream_bytes: u64,
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

/// What the embed quantum produces: the patch tokens, the timestep embedding and both modulation
/// cores' outputs.
pub(crate) struct Embedded {
    pub s: Array,
    pub t_emb: Array,
    pub core_img: Array,
    pub core_txt: Option<Array>,
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
        Self::from_weights_adapted(w, cfg, compute, None)
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
        Self::build(w, cfg, compute, input_channels, None, None)
    }

    /// [`from_weights`](Self::from_weights) with the projections of an adapter install
    /// ([`AdaptedLinears`]): every adapted projection must be one the graph reads, or the build
    /// fails naming it.
    pub fn from_weights_adapted(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        adapted: Option<&AdaptedLinears>,
    ) -> Result<Self> {
        Self::from_weights_quantized(w, cfg, compute, adapted, None)
    }

    /// [`from_weights_adapted`](Self::from_weights_adapted) at a quantized tier (sc-26266): every
    /// projection is held as [`crate::nn::ProjectionLayout::of`] says under `quant` (`None` is
    /// the dense tier). Adapters are installed on the dense projections first and stay
    /// forward-time residuals over the packed base.
    pub fn from_weights_quantized(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        adapted: Option<&AdaptedLinears>,
        quant: Option<BackboneQuant>,
    ) -> Result<Self> {
        let dit = Self::build(w, cfg, compute, cfg.in_channels, adapted, quant)?;
        if let Some(adapted) = adapted {
            let orphans = adapted.unconsumed();
            if !orphans.is_empty() {
                return Err(Error::Msg(format!(
                    "iris: adapter targets {orphans:?} are not projections of the backbone graph"
                )));
            }
        }
        Ok(dit)
    }

    fn build(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        input_channels: usize,
        adapted: Option<&AdaptedLinears>,
        quant: Option<BackboneQuant>,
    ) -> Result<Self> {
        let l = Loader {
            weights: w,
            compute,
            adapted,
            quant,
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
            blocks.push(Self::load_block(&l, cfg, i, eps, gqa)?);
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
            pixel_blocks.push(Self::load_pit(&l, cfg, i, eps)?);
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
        crate::nn::expect_dims(
            "s_embedder.proj.weight",
            &dit.s_embedder.logical_shape(),
            &[cfg.hidden_size as i32, p * p * input_channels as i32],
        )?;
        crate::nn::expect_dims(
            "pixel_embedder.proj.weight",
            &dit.pixel_proj.logical_shape(),
            &[cfg.pixel.hidden_size as i32, input_channels as i32],
        )?;
        Ok(dit)
    }

    /// Load trunk block `i` (`blocks.{i}.*`) from `l` — the backbone build and a checkpointed
    /// region's recompute (sc-25689) read one definition.
    fn load_block(l: &Loader, cfg: &ModelConfig, i: usize, eps: f32, gqa: bool) -> Result<Block> {
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
                    mlp: SwiGlu::load(l, &p("mlp_y"))?,
                    mlp_post_norm: l.norm(&p("mlp_post_norm_y"), eps)?,
                })
            } else {
                None
            };
            Ok(Block::Dual(Box::new(DualBlock {
                norm_x1: l.norm(&p("norm_x1"), eps)?,
                norm_x2: l.norm(&p("norm_x2"), eps)?,
                norm_y1: l.norm(&p("norm_y1"), eps)?,
                qkv_x: Qkv::load(
                    l,
                    &p("attn.qkv_x"),
                    &p("attn.q_proj_x"),
                    &p("attn.k_proj_x"),
                    &p("attn.v_proj_x"),
                    gqa,
                )?,
                qkv_y: Qkv::load(
                    l,
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
                mlp_x: SwiGlu::load(l, &p("mlp_x"))?,
                mlp_post_norm_x: l.norm(&p("mlp_post_norm_x"), eps)?,
                text,
                mod_img: bias("adaln_img.bias")?,
                mod_txt: bias("adaln_txt.bias")?,
            })))
        } else {
            Ok(Block::Single(Box::new(SingleBlock {
                norm1: l.norm(&p("norm1"), eps)?,
                norm2: l.norm(&p("norm2"), eps)?,
                qkv: Qkv::load(l, &p("qkv"), &p("q_proj"), &p("k_proj"), &p("v_proj"), gqa)?,
                q_norm: l.norm(&p("q_norm"), eps)?,
                k_norm: l.norm(&p("k_norm"), eps)?,
                attn_gate: l.linear(&p("attn_gate"), false)?,
                attn_proj: l.linear(&p("attn_proj"), true)?,
                attn_post_norm: l.norm(&p("attn_post_norm"), eps)?,
                mlp: SwiGlu::load(l, &p("mlp"))?,
                mlp_post_norm: l.norm(&p("mlp_post_norm"), eps)?,
                modulation: bias("adaln.bias")?,
                text_out,
            })))
        }
    }

    /// Load PiT block `i` (`pixel_blocks.{i}.*`) from `l`.
    fn load_pit(l: &Loader, cfg: &ModelConfig, i: usize, eps: f32) -> Result<PitBlock> {
        let p = |name: &str| format!("pixel_blocks.{i}.{name}");
        Ok(PitBlock {
            norm1: l.norm(&p("norm1"), eps)?,
            norm2: l.norm(&p("norm2"), eps)?,
            adaln: l.linear(&p("adaln"), true)?,
            compress: l.linear(&p("compress"), true)?,
            expand: l.linear(&p("expand"), true)?,
            attn: SelfAttention::load(l, &p("attn"), cfg.pixel.num_heads, cfg.qk_norm, eps)?,
            fc1: l.linear(&p("mlp.fc1"), true)?,
            fc2: l.linear(&p("mlp.fc2"), true)?,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.compute
    }

    /// Bytes the loaded parameters hold — every array [`Self::arrays`] names, at the width it is
    /// resident in (packed codes, scales and biases on a quantized tier). Measured from the model,
    /// not priced from the checkpoint.
    pub fn resident_weight_bytes(&self) -> u64 {
        self.arrays().iter().map(|a| a.nbytes() as u64).sum()
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

    // ---- iOS execution quanta (pass-through everywhere but a physical iOS device) ----------
    //
    // Every figure below is COUNTED under MLX's retention ([`crate::retention`]): every buffer a
    // quantum's body allocates, phase by phase between its sub-layer barriers
    // ([`crate::nn::barrier`]), over the loaded tensors' widths (projection shapes, gain widths,
    // head counts of the built modules) and the actual request geometry. Weights are consumed in
    // place (dense in the compute dtype, or packed and read by `quantized_matmul`), so no quantum
    // prices a weight copy. Every count is proven against MLX's measured peak at small shapes in
    // `crate::quantum_proofs`.

    fn query_rows(plan: mlx_gen::AttentionPlan<'_>, batch: i32, heads: i32, tokens: i32) -> u64 {
        plan.budget.query_block(batch, heads, tokens, tokens).max(0) as u64
    }

    fn compute_bytes(&self) -> u64 {
        crate::nn::dtype_bytes(self.compute)
    }

    /// Trunk block `i` over `n_patches` image and `n_txt` text tokens.
    pub(crate) fn trunk_block_scratch(
        &self,
        i: usize,
        batch: i32,
        n_patches: i32,
        n_txt: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let (dim, ff) = self.blocks[i].widths();
        let tokens = (n_patches + n_txt).max(0);
        let heads = self.cfg.num_heads as i32;
        let kv_heads = self.cfg.num_kv_heads.unwrap_or(self.cfg.num_heads) as u64;
        let (split_qkv, text, down_pad) = self.blocks[i].layout();
        let g = TrunkGeometry {
            compute_bytes: self.compute_bytes(),
            batch: batch.max(0) as u64,
            img_tokens: n_patches.max(0) as u64,
            txt_tokens: n_txt.max(0) as u64,
            dim,
            heads: heads as u64,
            kv_heads,
            ff,
            query_rows: Self::query_rows(plan, batch, heads, tokens),
            eval_per_chunk: plan.budget.eval_per_chunk(),
            split_qkv,
            text,
            down_pad,
            img_stream_bytes: if i == 0 { self.compute_bytes() } else { 4 },
        };
        match self.blocks[i] {
            Block::Dual(_) => Self::dual_block_bytes(&g, crate::retention::unfused()),
            Block::Single(_) => Self::single_block_bytes(&g, crate::retention::unfused()),
        }
    }

    /// One grouped-query (or fused) q / k / v projection of `rows` tokens from an f32 input,
    /// each output held.
    fn project_qkv(r: &mut Retention, g: &TrunkGeometry, rows: u64) {
        let (d, kvw) = (g.dim, g.dim / g.heads.max(1) * g.kv_heads);
        let narrow = r.narrow();
        if g.split_qkv {
            r.linear(rows, d, d, narrow, 0, Kind::Named)
                .linear(rows, d, kvw, narrow, 0, Kind::Named)
                .linear(rows, d, kvw, narrow, 0, Kind::Named);
        } else {
            r.linear(rows, d, d + 2 * kvw, narrow, 0, Kind::Named);
        }
    }

    /// A SwiGLU MLP over `rows` tokens of an f32 input (`w1` / `w3` read their own compute-width
    /// casts; SiLU and the gate product donate), and its output norm when `post_norm`. `bound`: the
    /// block binds the MLP output (held to the end; its norm reads a held input), else the output is
    /// a temporary the norm's result can reuse.
    fn swiglu(
        r: &mut Retention,
        rows: u64,
        d: u64,
        ff: u64,
        pad: u64,
        post_norm: bool,
        bound: bool,
    ) {
        let narrow = r.narrow();
        let out_kind = if bound { Kind::Named } else { Kind::Temp };
        r.linear(rows, d, ff, narrow, 0, Kind::Temp)
            .linear(rows, d, ff, narrow, 0, Kind::Temp)
            .linear(rows, ff, d, false, pad, out_kind);
        if post_norm {
            let input = match (narrow, bound) {
                (true, _) => NormInput::Cast,
                (false, true) => NormInput::Held,
                (false, false) => NormInput::Donor,
            };
            r.norm(rows * d, rows, input, Kind::Temp);
        }
    }

    /// The attention half of a trunk block's joint attention over `tokens` per row: the SDPA call
    /// on the f32 roped q / k and the compute-width v, its result held.
    fn trunk_attention(r: &mut Retention, g: &TrunkGeometry) {
        let narrow = r.narrow();
        let tokens = g.img_tokens + g.txt_tokens;
        r.attention(&Attention {
            batch: g.batch,
            tokens,
            heads: g.heads,
            kv_heads: g.kv_heads,
            head_dim: g.dim / g.heads.max(1),
            query_rows: g.query_rows,
            eval_per_chunk: g.eval_per_chunk,
            q_dense: true,
            q_cast: narrow,
            k_cast: narrow,
            v_cast: false,
            mask: None,
            out_kind: Kind::Named,
        });
    }

    /// `single`: one weight set over `cat([text, image])`. Phase 1 (to the q / k / v barrier): the
    /// modulation core's f32 sum, the token concatenation, the modulated norm `h`, the q / k / v
    /// projections, the QK-norms and RoPE (four applications), the q and k concatenations. Phase 2
    /// (to the residual barrier): the attention, the gate projection (its sigmoid and the gated
    /// product donate), the output projection and its post-norm, the gated residual. Phase 3: the
    /// modulated norm, the SwiGLU and its post-norm (the residual sum donates). Every bound handle
    /// is held to the end.
    pub(crate) fn single_block_bytes(g: &TrunkGeometry, unfused: bool) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused);
        let narrow = r.narrow();
        let (b, d) = (g.batch, g.dim);
        let kvw = d / g.heads.max(1) * g.kv_heads;
        let rows = b * (g.img_tokens + g.txt_tokens);
        let h = rows * d;
        let kv = rows * kvw;
        let qk_in = if narrow {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        // Phase 1 (a compute-width image stream is cast to f32 for the concatenation).
        if g.img_stream_bytes < 4 {
            r.temp(4, b * g.img_tokens * d);
        }
        r.named(4, b * 6 * d)
            .named(4, h)
            .norm(h, rows, NormInput::Held, Kind::Named)
            .temp(4, b * d);
        Self::project_qkv(&mut r, g, rows);
        r.norm(h, rows * g.heads, qk_in, Kind::Named)
            .norm(kv, rows * g.kv_heads, qk_in, Kind::Named)
            .rope(b * g.txt_tokens * d, Kind::Temp)
            .rope(b * g.img_tokens * d, Kind::Temp)
            .rope(b * g.txt_tokens * kvw, Kind::Temp)
            .rope(b * g.img_tokens * kvw, Kind::Temp)
            .named(4, h)
            .named(4, kv)
            .barrier();
        // Phase 2 — over the image tokens alone when the block drops the text stream.
        Self::trunk_attention(&mut r, g);
        let rows_o = if g.text { rows } else { b * g.img_tokens };
        let h_o = rows_o * d;
        let proj_kind = if narrow {
            Kind::Temp
        } else {
            r.donor(Kind::Named)
        };
        let post_in = if narrow {
            NormInput::Cast
        } else {
            NormInput::Donor
        };
        r.linear(rows_o, d, d, narrow, 0, Kind::Named)
            .linear(rows_o, d, d, false, 0, proj_kind)
            .norm(h_o, rows_o, post_in, Kind::Named)
            .named(4, h_o)
            .barrier();
        // Phase 3: the returned graph, run after the block's bindings are gone — it reads the
        // modulation sum and the residual stream.
        r.last_phase()
            .hold(4, b * 6 * d)
            .hold(4, h_o)
            .norm(h_o, rows_o, NormInput::Held, Kind::Temp)
            .temp(4, b * d);
        Self::swiglu(&mut r, rows_o, d, g.ff, g.down_pad, true, true);
        r.price()
    }

    /// `dual`: separate image / text weights, text-first joint attention. Phase 1: both streams'
    /// modulation sums and modulated norms, their q / k / v projections, the four QK-norms and RoPE
    /// applications, the q / k / v concatenations. Phase 2: the attention and the image half's gate,
    /// output projection, post-norm and residual. Phase 3: the image MLP. Phases 4 and 5: the text
    /// half's, when the block has a text path. Every bound handle is held to the end.
    pub(crate) fn dual_block_bytes(g: &TrunkGeometry, unfused: bool) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused);
        let narrow = r.narrow();
        let (b, d) = (g.batch, g.dim);
        let kvw = d / g.heads.max(1) * g.kv_heads;
        let (rows_x, rows_y) = (b * g.img_tokens, b * g.txt_tokens);
        let rows = rows_x + rows_y;
        let qk_in = if narrow {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        let post_in = qk_in;
        let img_narrow = g.img_stream_bytes < 4;
        // Phase 1.
        r.named(4, b * 6 * d).named(4, b * 6 * d);
        for (stream, narrow_in) in [(rows_x, img_narrow), (rows_y, false)] {
            let input = if narrow_in {
                NormInput::Cast
            } else {
                NormInput::Held
            };
            r.norm(stream * d, stream, input, Kind::Named)
                .temp(4, b * d);
            Self::project_qkv(&mut r, g, stream);
        }
        for stream in [rows_x, rows_y] {
            r.norm(stream * d, stream * g.heads, qk_in, Kind::Temp)
                .rope(stream * d, Kind::Named)
                .norm(stream * kvw, stream * g.kv_heads, qk_in, Kind::Temp)
                .rope(stream * kvw, Kind::Named);
        }
        let cw = r.cw;
        r.named(4, rows * d)
            .named(4, rows * kvw)
            .named(cw, rows * kvw)
            .barrier();
        // Phase 2: the attention, then the image half's post-attention.
        Self::trunk_attention(&mut r, g);
        let halves: &[u64] = if g.text { &[rows_x, rows_y] } else { &[rows_x] };
        for (k, &stream) in halves.iter().enumerate() {
            if k > 0 {
                // Phase 4: the text half's post-attention.
                r.barrier();
            }
            r.linear(stream, d, d, narrow, 0, Kind::Named)
                .linear(stream, d, d, false, 0, Kind::Named)
                .norm(stream * d, stream, post_in, Kind::Named);
            if k == 0 && img_narrow {
                // The compute-width image stream's f32 cast for the residual sum.
                r.temp(4, stream * d);
            }
            if k == 1 {
                // Phase 5: the text MLP is the returned graph, run after the block's bindings are
                // gone — it reads the text modulation sum, the text residual stream and (returned
                // with it) the image stream.
                r.last_phase()
                    .hold(4, b * 6 * d)
                    .hold(4, rows_y * d)
                    .hold(4, rows_x * d);
            } else {
                // Phase 3: the image MLP, evaluated by the block.
                r.barrier();
            }
            r.norm(stream * d, stream, NormInput::Held, Kind::Named)
                .temp(4, b * d);
            Self::swiglu(&mut r, stream, d, g.ff, g.down_pad, true, false);
        }
        r.price()
    }

    /// PiT block `i` over `n_patches` patches of `p²` pixels.
    pub(crate) fn pixel_block_scratch(
        &self,
        i: usize,
        batch: i32,
        n_patches: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let blk = &self.pixel_blocks[i];
        let heads = blk.attn.heads;
        Self::pixel_block_bytes(
            PixelBlockGeometry {
                compute_bytes: self.compute_bytes(),
                batch: batch.max(0) as u64,
                patches: n_patches.max(0) as u64,
                patch_pixels: (self.cfg.patch_size * self.cfg.patch_size) as u64,
                pixel_width: blk.norm1.width(),
                compact_width: blk.compress.out_features(),
                modulation_width: blk.adaln.out_features(),
                condition_width: blk.adaln.logical_shape()[1] as u64,
                fc1_width: blk.fc1.out_features(),
                heads: heads as u64,
                query_rows: Self::query_rows(plan, batch, heads, n_patches),
                eval_per_chunk: plan.budget.eval_per_chunk(),
            },
            crate::retention::unfused(),
        )
    }

    /// [`Self::pixel_block_scratch`] from the block's widths alone. Phase 1 (to the attention's
    /// q / k / v barrier): the adaLN modulation (held), the norm of the compute-width pixel plane,
    /// the compressed tokens (held), the fused qkv, the QK-norms and RoPE (q and k held into the
    /// attention). Phase 2 (to the residual barrier): the attention, its projection, the expanded
    /// plane (held) and the modulated residual. Phase 3: the norm, `fc1` (GELU donates; bf16: its
    /// result is f32, which `fc2` casts back), `fc2` (held) and the modulated residual.
    pub(crate) fn pixel_block_bytes(g: PixelBlockGeometry, unfused: bool) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused);
        let narrow = r.narrow();
        let cw = g.compute_bytes;
        let rows = g.batch * g.patches;
        let plane = rows * g.patch_pixels * g.pixel_width;
        let compact = rows * g.compact_width;
        let plane_in = if narrow {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        // Phase 1.
        r.linear(
            rows,
            g.condition_width,
            g.modulation_width,
            narrow,
            0,
            Kind::Named,
        )
        .norm(plane, rows * g.patch_pixels, plane_in, Kind::Temp)
        .linear(
            rows,
            g.patch_pixels * g.pixel_width,
            g.compact_width,
            narrow,
            0,
            Kind::Named,
        )
        .linear(
            rows,
            g.compact_width,
            3 * g.compact_width,
            false,
            0,
            Kind::Named,
        )
        .norm(compact, rows * g.heads, plane_in, Kind::Temp)
        .rope(compact, Kind::Named)
        .norm(compact, rows * g.heads, plane_in, Kind::Temp)
        .rope(compact, Kind::Named)
        .barrier();
        // Phase 2.
        r.attention(&Attention {
            batch: g.batch,
            tokens: g.patches,
            heads: g.heads,
            kv_heads: g.heads,
            head_dim: g.compact_width / g.heads.max(1),
            query_rows: g.query_rows,
            eval_per_chunk: g.eval_per_chunk,
            q_dense: true,
            q_cast: narrow,
            k_cast: narrow,
            v_cast: false,
            mask: None,
            out_kind: Kind::Temp,
        })
        .linear(
            rows,
            g.compact_width,
            g.compact_width,
            false,
            0,
            Kind::Named,
        )
        .linear(
            rows,
            g.compact_width,
            g.patch_pixels * g.pixel_width,
            false,
            0,
            Kind::Named,
        )
        .named(cw, plane);
        // Phase 3: the returned graph, run after the block's bindings are gone — it reads the
        // modulation and the residual stream.
        let mlp_rows = rows * g.patch_pixels;
        r.last_phase()
            .hold(cw, rows * g.modulation_width)
            .hold(cw, plane)
            .norm(plane, mlp_rows, plane_in, Kind::Temp)
            .linear(mlp_rows, g.pixel_width, g.fc1_width, narrow, 0, Kind::Temp);
        if narrow {
            // MLX's GELU divides by f32 constants: a bf16 input's result is a new f32 buffer.
            r.temp(4, mlp_rows * g.fc1_width);
        }
        r.linear(mlp_rows, g.fc1_width, g.pixel_width, narrow, 0, Kind::Named)
            .temp(cw, plane);
        r.price()
    }

    /// Text-adapter layer block `i`: attention over the `layers` encoder states of each of the
    /// `rows = B · T` tokens.
    pub(crate) fn text_layer_block_scratch(
        &self,
        i: usize,
        rows: i32,
        layers: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let blk = &self.y_embedder.layer_blocks[i];
        Self::text_block_bytes(
            TextBlockGeometry {
                compute_bytes: self.compute_bytes(),
                batch: rows.max(0) as u64,
                tokens: layers.max(0) as u64,
                dim: blk.norm1.width(),
                heads: blk.attn.heads as u64,
                ff: blk.mlp0.out_features(),
                query_rows: Self::query_rows(plan, rows, blk.attn.heads, layers),
                eval_per_chunk: plan.budget.eval_per_chunk(),
                refiner: false,
                down_pad: blk.mlp2.input_pad(),
                stream_bytes: 4,
            },
            crate::retention::unfused(),
        )
    }

    /// Text-refiner block `i` over `t` masked-key text tokens.
    pub(crate) fn text_refiner_block_scratch(
        &self,
        i: usize,
        batch: i32,
        t: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let blk = &self.y_embedder.blocks[i];
        Self::text_block_bytes(
            TextBlockGeometry {
                compute_bytes: self.compute_bytes(),
                batch: batch.max(0) as u64,
                tokens: t.max(0) as u64,
                dim: blk.norm1.width(),
                heads: blk.attn.heads as u64,
                ff: blk.mlp.hidden(),
                query_rows: Self::query_rows(plan, batch, blk.attn.heads, t),
                eval_per_chunk: plan.budget.eval_per_chunk(),
                refiner: true,
                down_pad: blk.mlp.w2.input_pad(),
                stream_bytes: self.compute_bytes(),
            },
            crate::retention::unfused(),
        )
    }

    /// One text-adapter block from its widths. Phase 1 (to the attention's q / k / v barrier): the
    /// input norm, the fused qkv (held into the attention) and, in a refiner block, the QK-norms
    /// (held). Phase 2 (to the residual barrier): the attention (refiner: under the compute-width
    /// key mask), its projection (bound) and the residual sum (a new buffer at the stream's width:
    /// the projection is bound, or its f32 cast is the sum). Phase 3: the norm, the MLP
    /// (`mlp.0 → SiLU → mlp.2`, or the SwiGLU) and the residual.
    pub(crate) fn text_block_bytes(g: TextBlockGeometry, unfused: bool) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused);
        let narrow = r.narrow();
        let rows = g.batch * g.tokens;
        let h = rows * g.dim;
        let narrow_in = |narrow: bool| {
            if narrow {
                NormInput::Cast
            } else {
                NormInput::Held
            }
        };
        let stream_in = narrow_in(g.stream_bytes < 4);
        // Phase 1. The normed input is the attention call's argument: a chunking plan evaluates its
        // chunks inside that call, while the argument is still held.
        let input_kind = if g.query_rows < g.tokens {
            Kind::Named
        } else {
            Kind::Temp
        };
        r.norm(h, rows, stream_in, input_kind).linear(
            rows,
            g.dim,
            3 * g.dim,
            narrow,
            0,
            Kind::Named,
        );
        if g.refiner {
            r.norm(h, rows * g.heads, narrow_in(narrow), Kind::Named)
                .norm(h, rows * g.heads, narrow_in(narrow), Kind::Named);
        }
        r.barrier();
        // Phase 2.
        r.attention(&Attention {
            batch: g.batch,
            tokens: g.tokens,
            heads: g.heads,
            kv_heads: g.heads,
            head_dim: g.dim / g.heads.max(1),
            query_rows: g.query_rows,
            eval_per_chunk: g.eval_per_chunk,
            q_dense: g.refiner,
            q_cast: g.refiner && narrow,
            k_cast: g.refiner && narrow,
            v_cast: false,
            mask: g.refiner.then_some((g.batch * g.tokens * g.tokens, narrow)),
            out_kind: Kind::Temp,
        })
        .linear(rows, g.dim, g.dim, false, 0, Kind::Named)
        .named(g.stream_bytes, h);
        // Phase 3: the returned graph, run after the block's bindings are gone — it reads the
        // residual stream.
        r.last_phase()
            .hold(g.stream_bytes, h)
            .norm(h, rows, stream_in, Kind::Temp);
        if g.refiner {
            // The SwiGLU output is a temporary: the residual sum reuses it.
            Self::swiglu(&mut r, rows, g.dim, g.ff, g.down_pad, false, false);
        } else {
            // `mlp.2`'s output is bound, so the residual sum is a new buffer (bf16: its operand's
            // f32 cast, which the sum reuses).
            r.linear(rows, g.dim, g.ff, narrow, 0, Kind::Temp)
                .linear(rows, g.ff, g.dim, false, g.down_pad, Kind::Named)
                .temp(g.stream_bytes, h);
        }
        r.price()
    }

    /// What the first trunk block materializes besides itself: the image and text RoPE tables and
    /// the text positional add (states, sliced table, sum).
    pub(crate) fn trunk_entry_allowance(&self, batch: i32, n_patches: i32, n_txt: i32) -> u64 {
        let cfg = &self.cfg;
        Self::trunk_entry_bytes(
            batch.max(0) as u64,
            n_patches.max(0) as u64,
            n_txt.max(0) as u64,
            (cfg.hidden_size / cfg.num_heads) as u64,
            cfg.hidden_size as u64,
        )
    }

    /// [`Self::trunk_entry_allowance`] from the widths alone.
    pub(crate) fn trunk_entry_bytes(b: u64, n: u64, n_txt: u64, head_dim: u64, hidden: u64) -> u64 {
        mlx_gen::ios_quantum::rope_table_bytes(n, head_dim)
            .saturating_add(mlx_gen::ios_quantum::rope_table_bytes(n_txt, head_dim))
            .saturating_add(mlx_gen::ios_quantum::elementwise_bytes(
                b.saturating_mul(n_txt).saturating_mul(hidden),
                3,
            ))
    }

    /// The patch/timestep embedding and both modulation cores.
    pub(crate) fn embed_scratch(&self, batch: i32, n_patches: i32) -> u64 {
        let [hidden, patch_in] = self.s_embedder.logical_shape();
        let [_, sinusoid] = self.t_mlp0.logical_shape();
        Self::embed_bytes(EmbedGeometry {
            compute_bytes: self.compute_bytes(),
            batch: batch.max(0) as u64,
            patches: n_patches.max(0) as u64,
            patch_in: patch_in as u64,
            hidden: hidden as u64,
            sinusoid: sinusoid as u64,
            t_hidden: self.t_mlp0.out_features(),
            t_out: self.t_mlp2.out_features(),
            cores: self.core_img.out_features()
                + self.core_txt.as_ref().map_or(0, Linear::out_features),
        })
    }

    /// [`Self::embed_scratch`] from the widths alone (one phase): the patch vectors' re-layout copy
    /// (f32) and the patch projection (held: the embedded tokens); the timestep path — the
    /// frequency table, the f32 phase, its cosine and sine and their concatenation, the first
    /// timestep projection (SiLU donates) and the second (held: `t_emb`); the conditioning SiLU of
    /// the held `t_emb`, and both cores' outputs (held).
    pub(crate) fn embed_bytes(g: EmbedGeometry) -> u64 {
        let mut r = Retention::on(g.compute_bytes, false);
        let narrow = r.narrow();
        let (b, freqs) = (g.batch, g.sinusoid / 2);
        r.temp(4, b * g.patches * g.patch_in)
            .linear(b * g.patches, g.patch_in, g.hidden, narrow, 0, Kind::Named)
            .temp(4, freqs)
            .alloc(Kind::Temp, 3, 4, b * freqs)
            .temp(4, b * g.sinusoid)
            .linear(b, g.sinusoid, g.t_hidden, narrow, 0, Kind::Temp)
            .linear(b, g.t_hidden, g.t_out, false, 0, Kind::Named)
            .temp(g.compute_bytes, b * g.t_out)
            .named(g.compute_bytes, b * g.cores);
        r.price()
    }

    /// The layer pool and the refiner's input projection over `[B·T, layers, Dt]` states.
    pub(crate) fn text_pool_scratch(&self, batch: i32, t: i32, layers: i32, text_dim: i32) -> u64 {
        Self::text_pool_bytes(
            self.compute_bytes(),
            batch.max(0) as u64,
            t.max(0) as u64,
            layers.max(0) as u64,
            text_dim.max(0) as u64,
            self.y_embedder.proj.out_features(),
        )
    }

    /// [`Self::text_pool_scratch`] from the widths alone (one phase): the pool reads the
    /// layer-transposed states through a row-contiguous copy (bf16: of its compute-width cast, which
    /// keeps the transposed layout), the pooled `[B·T, Dt]` and the projection `[B, T, D]`.
    pub(crate) fn text_pool_bytes(
        cw: u64,
        b: u64,
        t: u64,
        layers: u64,
        text_dim: u64,
        out: u64,
    ) -> u64 {
        let mut r = Retention::on(cw, false);
        let copies = if r.narrow() { 2 } else { 1 };
        r.alloc(Kind::Temp, copies, cw, b * t * layers * text_dim)
            .temp(cw, b * t * text_dim)
            .named(cw, b * t * out);
        r.price()
    }

    /// The refiner's final norm over `[B, T, D]` (one phase).
    pub(crate) fn text_output_scratch(&self, batch: i32, t: i32) -> u64 {
        Self::text_output_bytes(
            self.compute_bytes(),
            batch.max(0) as u64,
            t.max(0) as u64,
            self.y_embedder.norm.width(),
            crate::retention::unfused(),
        )
    }

    /// [`Self::text_output_scratch`] from the widths alone: the refiner's stream (compute width)
    /// is held, so its norm's result is a new buffer (bf16: of its f32 cast).
    pub(crate) fn text_output_bytes(cw: u64, b: u64, t: u64, width: u64, unfused: bool) -> u64 {
        let mut r = Retention::on(cw, unfused);
        let input = if r.narrow() {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        r.norm(b * t * width, b * t, input, Kind::Named);
        r.price()
    }

    /// The timestep re-fusion and the per-pixel embedding over an `h × w` input.
    pub(crate) fn pixel_embed_scratch(&self, batch: i32, h: i32, w: i32) -> u64 {
        let [pixel_width, channels] = self.pixel_proj.logical_shape();
        Self::pixel_embed_bytes(
            self.compute_bytes(),
            batch.max(0) as u64,
            h.max(0) as u64,
            w.max(0) as u64,
            self.cfg.patch_size as u64,
            self.core_img.logical_shape()[1] as u64,
            pixel_width as u64,
            channels as u64,
        )
    }

    /// [`Self::pixel_embed_scratch`] from the widths alone (one phase): the f32 `t_emb + s` (SiLU
    /// donates; it is the held per-patch conditioning; bf16: plus the broadcast `t_emb`'s f32 cast);
    /// the projection's row-contiguous copy of the transposed `[B, H, W, C]` input (bf16: and the
    /// compute-width cast it copies); the projection over the `[B, H, W, pixel]` plane (the
    /// position-table sum donates); the f32 position table (built once per size and cached; bf16:
    /// plus its per-call compute-width cast); and the per-patch re-layout copy (held: the pixel
    /// stream).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pixel_embed_bytes(
        cw: u64,
        b: u64,
        h: u64,
        w: u64,
        patch: u64,
        hidden: u64,
        pixel_width: u64,
        channels: u64,
    ) -> u64 {
        let mut r = Retention::on(cw, false);
        let narrow = r.narrow();
        let n = (h / patch.max(1)) * (w / patch.max(1));
        let pixels = b * h * w;
        if narrow {
            r.temp(4, b * hidden).temp(cw, pixels * channels);
        }
        r.named(4, b * n * hidden)
            .temp(cw, pixels * channels)
            .temp(cw, pixels * pixel_width)
            .temp(4, h * w * pixel_width);
        if narrow {
            r.temp(cw, h * w * pixel_width);
        }
        r.named(cw, pixels * pixel_width);
        r.price()
    }

    /// The final norm, the final projection and the fold back to `[B, C, H, W]` f32.
    pub(crate) fn output_scratch(&self, batch: i32, n_patches: i32) -> u64 {
        let [out, input] = self.final_linear.logical_shape();
        Self::output_bytes(
            self.compute_bytes(),
            batch.max(0) as u64,
            n_patches.max(0) as u64,
            (self.cfg.patch_size * self.cfg.patch_size) as u64,
            input as u64,
            out as u64,
            crate::retention::unfused(),
        )
    }

    /// [`Self::output_scratch`] from the widths alone (one phase): the norm of the compute-width
    /// pixel stream, the projection to `[B·N, p², C]` (bf16: of its compute-width cast), the fold's
    /// re-layout copy and (bf16) its f32 cast, the output.
    pub(crate) fn output_bytes(
        cw: u64,
        b: u64,
        n: u64,
        ppp: u64,
        input: u64,
        out: u64,
        unfused: bool,
    ) -> u64 {
        let mut r = Retention::on(cw, unfused);
        let narrow = r.narrow();
        let rows = b * n * ppp;
        let input_kind = if narrow {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        r.norm(rows * input, rows, input_kind, Kind::Temp)
            .linear(rows, input, out, narrow, 0, Kind::Temp)
            .named(cw, rows * out);
        if narrow {
            r.named(4, rows * out);
        }
        r.price()
    }

    /// `TimestepEmbedder`: sinusoid bank (period `timestep_max_period`, 256 frequencies, cos‖sin)
    /// → Linear → SiLU → Linear. `t` is model time `[B]` f32 → `[B, 1, D]`.
    fn timestep_embedding(&self, t: &Array) -> Result<Array> {
        let freqs = iris::timestep_freqs(self.cfg.timestep_max_period);
        let n = freqs.len();
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
        self.text_adapter_budgeted(text, mlx_gen::AttentionPlan::UNBOUNDED)
    }

    fn text_adapter_budgeted(
        &self,
        text: &TextBatch,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        let sh = text.states.shape();
        let (b, t, layers, dt) = (sh[0], sh[1], sh[2], sh[3]);
        if text.mask.len() != b as usize || text.mask.iter().any(|m| m.len() != t as usize) {
            return Err(Error::Msg(format!(
                "iris: text mask must be [{b}, {t}] for states {sh:?}"
            )));
        }
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        let mut y = text
            .states
            .as_dtype(Dtype::Float32)?
            .reshape(&[b * t, layers, dt])?;
        for i in 0..self.y_embedder.layer_blocks.len() {
            let scratch = self.text_layer_block_scratch(i, b * t, layers, plan);
            y = mlx_gen::ios_quantum::run(quanta::TEXT_LAYER_BLOCK, scratch, || {
                self.text_layer_body(i, &y, plan)
            })?;
        }
        // layer_pool: Linear(L → 1) over the layer axis, then the refiner's input projection.
        let mut y = mlx_gen::ios_quantum::run(
            quanta::TEXT_POOL,
            self.text_pool_scratch(b, t, layers, dt),
            || self.text_pool_body(&y, b, t, dt),
        )?;
        // refiner (TransformerTextEmbedder): keys masked, every query keeps its own position.
        let key_mask = key_padding_mask(text.mask, t as usize);
        for i in 0..self.y_embedder.blocks.len() {
            let scratch = self.text_refiner_block_scratch(i, b, t, plan);
            y = mlx_gen::ios_quantum::run(quanta::TEXT_REFINER_BLOCK, scratch, || {
                self.text_refiner_body(i, &y, &key_mask, plan)
            })?;
        }
        mlx_gen::ios_quantum::run(quanta::TEXT_OUTPUT, self.text_output_scratch(b, t), || {
            self.text_output_body(&y)
        })
    }

    /// Body of text-adapter layer block `i` over `y` `[B·T, layers, Dt]` (one quantum).
    pub(crate) fn text_layer_body(
        &self,
        i: usize,
        y: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        let block = &self.y_embedder.layer_blocks[i];
        let h = block.attn.forward(
            &block.norm1.forward(y)?,
            None,
            None,
            self.compute,
            plan,
            "adapter",
        )?;
        let y = y.add(&h)?;
        barrier(&[&y])?;
        let m = block
            .mlp2
            .forward(&silu(block.mlp0.forward(&block.norm2.forward(&y)?)?)?)?;
        Ok(y.add(&m)?)
    }

    /// Body of the layer pool and the refiner's input projection (one quantum).
    pub(crate) fn text_pool_body(&self, y: &Array, b: i32, t: i32, dt: i32) -> Result<Array> {
        let y_ad = &self.y_embedder;
        let pooled = y_ad
            .layer_pool
            .forward(&y.transpose_axes(&[0, 2, 1])?)?
            .reshape(&[b, t, dt])?;
        y_ad.proj.forward(&pooled)
    }

    /// Body of text-refiner block `i` over `y` `[B, T, D]` under the additive key mask (one quantum).
    pub(crate) fn text_refiner_body(
        &self,
        i: usize,
        y: &Array,
        key_mask: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        let block = &self.y_embedder.blocks[i];
        let h = block.attn.forward(
            &block.norm1.forward(y)?,
            None,
            Some(key_mask),
            self.compute,
            plan,
            "refiner",
        )?;
        let y = y.add(&h)?;
        barrier(&[&y])?;
        Ok(y.add(&block.mlp.forward(&block.norm2.forward(&y)?)?)?)
    }

    /// Body of the refiner's final norm (one quantum).
    pub(crate) fn text_output_body(&self, y: &Array) -> Result<Array> {
        self.y_embedder.norm.forward(y)
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
        let data = iris::pixel_sincos_table(height, width, dim);
        let table = Array::from_slice(&data, &[height as i32, width as i32, dim as i32]);
        cache.insert((height, width), table.clone());
        Ok(table)
    }

    /// Predict the flow velocity `[B, C, H, W]` (f32) for noisy pixels `x` at model time `t` `[B]`.
    pub fn forward(&self, x: &Array, t: &Array, text: &TextBatch) -> Result<Array> {
        self.forward_hooked(x, t, text, &mut NoHooks)
    }

    /// [`Self::forward`] with `hooks` around every trunk and pixel block — the training forward's
    /// seam for activation-checkpointing regions and the REPA feature capture (sc-25689). With
    /// [`NoHooks`] it is exactly the inference forward.
    pub fn forward_hooked(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        hooks: &mut dyn ForwardHooks,
    ) -> Result<Array> {
        self.forward_budgeted_hooked(x, t, text, hooks, mlx_gen::AttentionPlan::UNBOUNDED)
    }

    /// Convenience wrapper for callers without cancellation (legacy/training stays unbounded).
    pub fn forward_budgeted(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        budget: mlx_gen::AttentionBudget,
    ) -> Result<Array> {
        self.forward_plan(x, t, text, mlx_gen::AttentionPlan::budgeted(budget))
    }

    /// Request-scoped inference plan, including cancellation checked between attention chunks.
    pub fn forward_plan(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        self.forward_budgeted_hooked(x, t, text, &mut NoHooks, plan)
    }

    /// Body of the patch/timestep embedding and both modulation cores over `x` `[B, C, H, W]`
    /// (one quantum).
    pub(crate) fn embed_body(&self, x: &Array, t: &Array) -> Result<Embedded> {
        let sh = x.shape();
        let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let p = self.cfg.patch_size as i32;
        let (hp, wp) = (h / p, w / p);
        // F.unfold(x, p, stride=p).transpose(1, 2): channel-major patch vectors, row-major patches.
        let patches = x
            .reshape(&[b, c, hp, p, wp, p])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, hp * wp, c * p * p])?;
        let s = self.s_embedder.forward(&patches)?;
        let t_emb = self.timestep_embedding(t)?;
        let cond = silu(&t_emb)?;
        let core_img = self.core_img.forward(&cond)?;
        let core_txt = self
            .core_txt
            .as_ref()
            .map(|core| core.forward(&cond))
            .transpose()?;
        Ok(Embedded {
            s,
            t_emb,
            core_img,
            core_txt,
        })
    }

    /// The text tokens `[B, T, D]` plus the absolute text position table (lazy: it materializes
    /// inside the first trunk block's quantum, whose entry allowance carries it).
    pub(crate) fn text_positions(&self, y: &Array) -> Result<Array> {
        let n_txt = y.shape()[1];
        let pos = self
            .y_pos_embedding
            .reshape(&[1, -1, self.cfg.hidden_size as i32])?
            .split_axis(&[n_txt], 1)?
            .swap_remove(0);
        Ok(y.add(&pos)?)
    }

    /// What every trunk block of one forward reads besides its weights.
    pub(crate) fn trunk_ctx<'a>(
        &self,
        hp: i32,
        wp: i32,
        n_txt: i32,
        plan: mlx_gen::AttentionPlan<'a>,
    ) -> TrunkCtx<'a> {
        let cfg = &self.cfg;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let square = cfg.rope_aspect == "square";
        let rope_img = Rope::grid_2d_aspect(
            head_dim,
            hp as usize,
            wp as usize,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
            square,
        );
        let rope_txt = Rope::line_1d(head_dim, n_txt as usize, cfg.text_rope_theta as f32);
        TrunkCtx {
            compute: self.compute,
            plan,
            rope_img,
            rope_txt,
            heads: cfg.num_heads as i32,
            kv_heads: cfg.num_kv_heads.unwrap_or(cfg.num_heads) as i32,
        }
    }

    /// Body of the timestep re-fusion and the per-pixel embedding (one quantum): returns the
    /// per-patch conditioning `[B·N, D]` and the pixel stream `[B·N, p², pixel width]`.
    pub(crate) fn pixel_embed_body(
        &self,
        x: &Array,
        t_emb: &Array,
        s: &Array,
    ) -> Result<(Array, Array)> {
        let cfg = &self.cfg;
        let sh = x.shape();
        let (b, h, w) = (sh[0], sh[2], sh[3]);
        let p = cfg.patch_size as i32;
        let (hp, wp) = (h / p, w / p);
        let n_patches = hp * wp;
        // timestep re-fused into every patch token
        let s = silu(&t_emb.add(s)?)?;
        let s_cond = s.reshape(&[b * n_patches, -1])?;

        // PixelEmbedder: per-pixel Linear + full-resolution sincos, grouped per patch.
        let tokens = self.pixel_proj.forward(&x.transpose_axes(&[0, 2, 3, 1])?)?;
        let tokens = tokens.add(
            &self
                .pixel_pos(h as usize, w as usize)?
                .as_dtype(tokens.dtype())?,
        )?;
        let pix = cfg.pixel.hidden_size as i32;
        let pixels = tokens
            .reshape(&[b, hp, p, wp, p, pix])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b * n_patches, p * p, pix])?;
        Ok((s_cond, pixels))
    }

    /// What every PiT block of one forward reads besides its weights.
    pub(crate) fn pixel_ctx<'a>(
        &self,
        batch: i32,
        hp: i32,
        wp: i32,
        plan: mlx_gen::AttentionPlan<'a>,
    ) -> PixelCtx<'a> {
        let cfg = &self.cfg;
        let pix_head = cfg.pixel.attn_hidden_size / cfg.pixel.num_heads;
        let rope_pix = Rope::grid_2d_aspect(
            pix_head,
            hp as usize,
            wp as usize,
            cfg.rope_theta as f32,
            cfg.rope_scale as f32,
            cfg.rope_aspect == "square",
        );
        PixelCtx {
            compute: self.compute,
            plan,
            rope_pix,
            batch,
            n_patches: hp * wp,
        }
    }

    /// Body of the final norm, the final projection and the fold back to `[B, C, H, W]` f32 (one
    /// quantum).
    pub(crate) fn output_body(&self, pixels: &Array, b: i32, h: i32, w: i32) -> Result<Array> {
        let cfg = &self.cfg;
        let p = cfg.patch_size as i32;
        let (hp, wp) = (h / p, w / p);
        let out = self
            .final_linear
            .forward(&self.final_norm.forward(pixels)?)?;
        // fold: [B·L, p·p, C] → [B, C, H, W] (C = model.in_channels, whatever x carried)
        let c_out = cfg.in_channels as i32;
        Ok(out
            .reshape(&[b, hp, wp, p, p, c_out])?
            .transpose_axes(&[0, 5, 1, 3, 2, 4])?
            .reshape(&[b, c_out, h, w])?
            .as_dtype(Dtype::Float32)?)
    }

    fn forward_budgeted_hooked(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        hooks: &mut dyn ForwardHooks,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
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

        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        // The previous step's guidance combine + solver arithmetic is lazy and materializes here.
        mlx_gen::ios_quantum::before(
            quanta::EMBED,
            self.embed_scratch(b, n_patches)
                .saturating_add(mlx_gen::ios_quantum::sampler_step_bytes(x.size() as u64)),
        )?;
        let Embedded {
            mut s,
            t_emb,
            core_img,
            core_txt,
        } = self.embed_body(x, t)?;
        let mut embedded = vec![&s, &t_emb, &core_img];
        embedded.extend(core_txt.as_ref());
        mlx_gen::ios_quantum::finish(quanta::EMBED, &embedded)?;

        let mut y = self.text_adapter_budgeted(text, plan)?;
        let n_txt = y.shape()[1];
        y = self.text_positions(&y)?;

        let ctx = self.trunk_ctx(hp, wp, n_txt, plan);
        // The text positional add and both RoPE tables are lazy and materialize inside the FIRST
        // trunk block's quantum, whose scratch carries them.
        let entry_extra = self.trunk_entry_allowance(b, n_patches, n_txt);
        for i in 0..self.blocks.len() {
            let scratch = self
                .trunk_block_scratch(i, b, n_patches, n_txt, plan)
                .saturating_add(if i == 0 { entry_extra } else { 0 });
            mlx_gen::ios_quantum::before(quanta::TRUNK_BLOCK, scratch)?;
            (s, y) = hooks.trunk_block(self, i, &ctx, &s, &y, &core_img, core_txt.as_ref())?;
            mlx_gen::ios_quantum::finish(quanta::TRUNK_BLOCK, &[&s, &y])?;
        }

        mlx_gen::ios_quantum::before(quanta::PIXEL_EMBED, self.pixel_embed_scratch(b, h, w))?;
        let (s_cond, mut pixels) = self.pixel_embed_body(x, &t_emb, &s)?;
        let pctx = self.pixel_ctx(b, hp, wp, plan);
        mlx_gen::ios_quantum::finish(quanta::PIXEL_EMBED, &[&s_cond, &pixels])?;
        for i in 0..self.pixel_blocks.len() {
            let scratch = self.pixel_block_scratch(i, b, n_patches, plan);
            pixels = mlx_gen::ios_quantum::run(quanta::PIXEL_BLOCK, scratch, || {
                hooks.pixel_block(self, i, &pctx, &pixels, &s_cond)
            })?;
        }
        mlx_gen::ios_quantum::run(quanta::OUTPUT, self.output_scratch(b, n_patches), || {
            self.output_body(&pixels, b, h, w)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dual(
        compute: Dtype,
        blk: &DualBlock,
        x: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: &Array,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: i32,
        kv_heads: i32,
        plan: mlx_gen::AttentionPlan<'_>,
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
        barrier(&[&q, &k, &v])?;
        let out = attention_at_site(&q, &k, &v, compute, None, plan, "dual")?;
        let n_txt = y.shape()[1];
        let mut halves = out.split_axis(&[n_txt], 1)?;
        let out_x = halves.swap_remove(1);
        let out_y = halves.swap_remove(0);

        let out_x = out_x.multiply(&sigmoid(blk.gate_x.forward(&hx)?)?)?;
        let attn_x = blk.proj_x.forward(&out_x)?;
        let mut x = x.add(&mx[2].multiply(&blk.attn_post_norm_x.forward(&attn_x)?)?)?;
        barrier(&[&x])?;
        let mlp_in = modulate(&blk.norm_x2.forward(&x)?, &mx[3], &mx[4])?;
        x = x.add(&mx[5].multiply(&blk.mlp_post_norm_x.forward(&blk.mlp_x.forward(&mlp_in)?)?)?)?;
        barrier(&[&x])?;
        let Some(text) = &blk.text else {
            return Ok((x, y.clone()));
        };
        let out_y = out_y.multiply(&sigmoid(text.gate.forward(&hy)?)?)?;
        let attn_y = text.proj.forward(&out_y)?;
        let mut y = y.add(&my[2].multiply(&text.attn_post_norm.forward(&attn_y)?)?)?;
        barrier(&[&y])?;
        let mlp_in = modulate(&text.norm2.forward(&y)?, &my[3], &my[4])?;
        y = y.add(&my[5].multiply(&text.mlp_post_norm.forward(&text.mlp.forward(&mlp_in)?)?)?)?;
        Ok((x, y))
    }

    #[allow(clippy::too_many_arguments)]
    fn single(
        compute: Dtype,
        blk: &SingleBlock,
        x: &Array,
        y: &Array,
        core: &Array,
        rope_img: &Rope,
        rope_txt: &Rope,
        heads: i32,
        kv_heads: i32,
        plan: mlx_gen::AttentionPlan<'_>,
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
        barrier(&[&q, &k, &v])?;
        let mut attn = attention_at_site(&q, &k, &v, compute, None, plan, "single")?;
        let (mut h, mut tokens) = (h, tokens);
        if !blk.text_out {
            attn = attn.split_axis(&[n_txt], 1)?.swap_remove(1);
            h = h.split_axis(&[n_txt], 1)?.swap_remove(1);
            tokens = x.clone();
        }
        let attn = attn.multiply(&sigmoid(blk.attn_gate.forward(&h)?)?)?;
        let attn = blk.attn_post_norm.forward(&blk.attn_proj.forward(&attn)?)?;
        let tokens = tokens.add(&m[2].multiply(&attn)?)?;
        barrier(&[&tokens])?;
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

    #[allow(clippy::too_many_arguments)]
    fn pit(
        compute: Dtype,
        blk: &PitBlock,
        x: &Array,
        cond: &Array,
        rope: &Rope,
        batch: i32,
        n_patches: i32,
        plan: mlx_gen::AttentionPlan<'_>,
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
            compute,
            plan,
            "pixel",
        )?;
        let expanded = blk
            .expand
            .forward(&attn.reshape(&[rows, -1])?)?
            .reshape(&[rows, ppp, pix])?;
        let x = x.add(&modulate(&expanded, &m[1], &m[0])?)?;
        barrier(&[&x])?;
        let mlp = blk
            .fc2
            .forward(&gelu(blk.fc1.forward(&blk.norm2.forward(&x)?)?)?)?;
        Ok(x.add(&modulate(&mlp, &m[3], &m[2])?)?)
    }
}

// =============================================================================================
// Training seams (sc-25689): per-block hooks, externally built blocks
// =============================================================================================

/// What a trunk block reads besides its own weights in one forward: the image / text RoPE tables
/// and the head counts.
#[derive(Clone)]
pub struct TrunkCtx<'a> {
    compute: Dtype,
    plan: mlx_gen::AttentionPlan<'a>,
    rope_img: Rope,
    rope_txt: Rope,
    heads: i32,
    kv_heads: i32,
}

/// What a PiT block reads besides its own weights in one forward.
#[derive(Clone)]
pub struct PixelCtx<'a> {
    compute: Dtype,
    plan: mlx_gen::AttentionPlan<'a>,
    rope_pix: Rope,
    batch: i32,
    n_patches: i32,
}

/// A trunk block built outside a loaded backbone (an activation-checkpointing region's
/// recompute builds one from the region's own input arrays).
pub struct TrunkBlock(Block);

/// A PiT block built outside a loaded backbone.
pub struct PixelBlock(PitBlock);

/// Hooks around every block of [`IrisDiT::forward_hooked`]. The defaults run the backbone's own
/// block, which is the inference forward.
pub trait ForwardHooks {
    #[allow(clippy::too_many_arguments)]
    fn trunk_block(
        &mut self,
        dit: &IrisDiT,
        i: usize,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        dit.run_trunk_block(i, ctx, s, y, core_img, core_txt)
    }

    fn pixel_block(
        &mut self,
        dit: &IrisDiT,
        i: usize,
        ctx: &PixelCtx,
        pixels: &Array,
        s_cond: &Array,
    ) -> Result<Array> {
        dit.run_pixel_block(i, ctx, pixels, s_cond)
    }
}

/// The inference forward's hooks: none.
pub struct NoHooks;

impl ForwardHooks for NoHooks {}

impl IrisDiT {
    /// Trunk blocks.
    pub fn trunk_depth(&self) -> usize {
        self.blocks.len()
    }

    /// PiT blocks.
    pub fn pixel_depth(&self) -> usize {
        self.pixel_blocks.len()
    }

    /// Run the backbone's own trunk block `i` on `(s, y)`.
    pub fn run_trunk_block(
        &self,
        i: usize,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        Self::trunk_block_with(&self.blocks[i], ctx, s, y, core_img, core_txt)
    }

    /// Run an externally built trunk block on `(s, y)` (same graph as [`Self::run_trunk_block`]).
    pub fn run_external_trunk_block(
        &self,
        block: &TrunkBlock,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        Self::trunk_block_with(&block.0, ctx, s, y, core_img, core_txt)
    }

    fn trunk_block_with(
        block: &Block,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        match block {
            Block::Dual(blk) => Self::dual(
                ctx.compute,
                blk,
                s,
                y,
                core_img,
                core_txt.ok_or_else(|| {
                    Error::Msg("iris: a dual-stream block needs the text modulation core".into())
                })?,
                &ctx.rope_img,
                &ctx.rope_txt,
                ctx.heads,
                ctx.kv_heads,
                ctx.plan,
            ),
            Block::Single(blk) => Self::single(
                ctx.compute,
                blk,
                s,
                y,
                core_img,
                &ctx.rope_img,
                &ctx.rope_txt,
                ctx.heads,
                ctx.kv_heads,
                ctx.plan,
            ),
        }
    }

    /// Run the backbone's own PiT block `i`.
    pub fn run_pixel_block(
        &self,
        i: usize,
        ctx: &PixelCtx,
        pixels: &Array,
        s_cond: &Array,
    ) -> Result<Array> {
        Self::pit(
            ctx.compute,
            &self.pixel_blocks[i],
            pixels,
            s_cond,
            &ctx.rope_pix,
            ctx.batch,
            ctx.n_patches,
            ctx.plan,
        )
    }

    /// Run an externally built PiT block.
    pub fn run_external_pixel_block(
        &self,
        block: &PixelBlock,
        ctx: &PixelCtx,
        pixels: &Array,
        s_cond: &Array,
    ) -> Result<Array> {
        Self::pit(
            ctx.compute,
            &block.0,
            pixels,
            s_cond,
            &ctx.rope_pix,
            ctx.batch,
            ctx.n_patches,
            ctx.plan,
        )
    }

    /// Build trunk block `i` from `w` exactly as the backbone build does (`adapted` installs the
    /// adapter residuals of its projections).
    pub fn load_trunk_block(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        i: usize,
        adapted: Option<&AdaptedLinears>,
    ) -> Result<TrunkBlock> {
        let l = Loader {
            weights: w,
            compute,
            adapted,
            quant: None,
        };
        let gqa = cfg.num_kv_heads.is_some_and(|kv| kv != cfg.num_heads);
        Ok(TrunkBlock(Self::load_block(
            &l,
            cfg,
            i,
            cfg.norm_eps as f32,
            gqa,
        )?))
    }

    /// Build PiT block `i` from `w` exactly as the backbone build does.
    pub fn load_pixel_block(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        i: usize,
        adapted: Option<&AdaptedLinears>,
    ) -> Result<PixelBlock> {
        let l = Loader {
            weights: w,
            compute,
            adapted,
            quant: None,
        };
        Ok(PixelBlock(Self::load_pit(&l, cfg, i, cfg.norm_eps as f32)?))
    }
}

impl TrunkBlock {
    /// Run this block on `(s, y)` under `ctx` (owns nothing of a backbone: a checkpointed region's
    /// recompute runs it after the forward that built `ctx` returned).
    pub fn run(
        &self,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        IrisDiT::trunk_block_with(&self.0, ctx, s, y, core_img, core_txt)
    }
}

impl PixelBlock {
    /// Run this PiT block under `ctx`.
    pub fn run(&self, ctx: &PixelCtx, pixels: &Array, s_cond: &Array) -> Result<Array> {
        IrisDiT::pit(
            ctx.compute,
            &self.0,
            pixels,
            s_cond,
            &ctx.rope_pix,
            ctx.batch,
            ctx.n_patches,
            ctx.plan,
        )
    }
}
