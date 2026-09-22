//! The Qwen-Image 2.1 single-stream DiT — a faithful port of diffusers'
//! `QwenImage21Transformer2DModel` (`transformer_qwenimage21.py` @
//! [`crate::UPSTREAM_DIFFUSERS_REVISION`]), prefill path (`QwenImage21AttnProcessor`, no KV cache).
//!
//! Text and image latents share **one joint sequence** ([`JointLayout`]): the projected text tokens
//! in order, condition-image blocks substituted at the vision slots the encoder reserved for them
//! (a later story), and the target image's tokens appended. Two behaviours distinguish 2.1:
//!
//! * **Block-causal attention** — `(q ≥ kv) or same_image_block`: the joint sequence is causal,
//!   every image block is internally bidirectional, and the target block (last) therefore sees
//!   everything. Realised exactly as upstream's SDPA processor does: one attention call per prefix
//!   segment (text segments with a causal mask over keys `[0, end)`, image segments unmasked) plus
//!   one unmasked call for the target over all keys.
//! * **`causal_condition`** — the shared modulation is computed for two timestep rows, the sampled
//!   `t` and `t = 0`; text/condition tokens read the `t = 0` row, target tokens their own.
//!
//! Numerics: the model runs in the dtype of its weights (bf16 for the released checkpoint, f32 for
//! the parity fixtures); the zero-centred text norm, RoPE rotation and timestep sinusoids are
//! computed in f32 and rounded back, as upstream does. The 3-axis RoPE
//! (`axes_dims_rope = [16, 56, 56]`, `θ = 10000`) rotates **adjacent** channel pairs
//! (`view_as_complex`, `use_real=False`), unlike the text tower's half-split RoPE.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_tanh, rope_rotate, rope_sincos_from_ids, silu, timestep_sincos};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{
    layer_norm, rms_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, mean_axis, multiply, r#where, split, stack_axis, tanh,
};
use mlx_rs::{Array, Dtype};

use crate::config::TransformerConfig;

/// Rotary base of the joint-sequence RoPE (`QwenImage21Rope(theta=10000)`).
const ROPE_THETA: f32 = 10_000.0;
/// Sinusoidal timestep projection width (`QwenImage21TemporalTimesteps(timestep_dim=256)`).
const TIMESTEP_DIM: usize = 256;
/// `time_factor`: the transformer receives `sigma` and scales it by 1000 inside.
const TIME_FACTOR: f32 = 1000.0;
/// Group size for a pre-quantized (packed) DiT — the codebase-wide default (64).
const GROUP_SIZE: i32 = 64;

/// One run of the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `len` prompt tokens (strictly causal among themselves).
    Text { len: usize },
    /// One image block of `height × width` latent tokens (row-major), internally bidirectional.
    /// The **last** image segment of a layout is the target being denoised; any earlier one is a
    /// condition image (edit path).
    Image { height: usize, width: usize },
}

/// The joint text/image token layout the transformer attends over. Built by the pipeline; for
/// text-to-image it is `[Text { len }, Image { target }]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointLayout {
    pub segments: Vec<Segment>,
}

impl JointLayout {
    /// The text-to-image layout: the prompt then the target image.
    pub fn text_to_image(text_len: usize, height: usize, width: usize) -> Self {
        Self {
            segments: vec![
                Segment::Text { len: text_len },
                Segment::Image { height, width },
            ],
        }
    }

    fn validate(&self) -> Result<()> {
        match self.segments.last() {
            Some(Segment::Image { height, width }) if *height > 0 && *width > 0 => Ok(()),
            _ => Err(Error::Msg(
                "qwen_image_2_1: the joint layout must end with the non-empty target image block"
                    .into(),
            )),
        }
    }

    fn segment_len(segment: &Segment) -> usize {
        match segment {
            Segment::Text { len } => *len,
            Segment::Image { height, width } => height * width,
        }
    }

    pub fn total_len(&self) -> usize {
        self.segments.iter().map(Self::segment_len).sum()
    }

    /// Tokens of the target image block.
    pub fn target_tokens(&self) -> usize {
        self.segments.last().map_or(0, Self::segment_len)
    }

    /// Everything before the target block.
    pub fn prefix_len(&self) -> usize {
        self.total_len() - self.target_tokens()
    }

    /// `(start, end, is_text)` runs over the prefix — `_qwenimage21_prefix_segments`.
    pub fn prefix_segments(&self) -> Vec<(usize, usize, bool)> {
        let mut out = Vec::new();
        let mut cursor = 0;
        for segment in &self.segments[..self.segments.len().saturating_sub(1)] {
            let len = Self::segment_len(segment);
            if len > 0 {
                out.push((
                    cursor,
                    cursor + len,
                    matches!(segment, Segment::Text { .. }),
                ));
            }
            cursor += len;
        }
        out
    }

    /// `(frame, height, width)` RoPE position per token — `QwenImage21Rope.forward`: text tokens
    /// advance one shared position on all three axes; each image block freezes the frame axis at
    /// the position reached so far and lays its tokens on a zero-centred grid, then advances the
    /// shared position by `max(height, width)`.
    pub fn position_ids(&self) -> Vec<[i32; 3]> {
        let mut ids = Vec::with_capacity(self.total_len());
        let mut position: i32 = 0;
        for segment in &self.segments {
            match *segment {
                Segment::Text { len } => {
                    for _ in 0..len {
                        ids.push([position, position, position]);
                        position += 1;
                    }
                }
                Segment::Image { height, width } => {
                    let (h, w) = (height as i32, width as i32);
                    for y in -(h - h / 2)..(h / 2) {
                        for x in -(w - w / 2)..(w / 2) {
                            ids.push([position, y, x]);
                        }
                    }
                    position += h.max(w);
                }
            }
        }
        ids
    }

    /// `true` at the target image's tokens.
    pub fn target_mask(&self) -> Vec<bool> {
        let total = self.total_len();
        let target = self.target_tokens();
        (0..total).map(|i| i >= total - target).collect()
    }
}

/// Zero-centred RMSNorm (`QwenImage21ZeroCenterRMSNorm`): effective scale `weight + 1`, f32.
fn zero_center_rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let dtype = x.dtype();
    let x32 = x.as_dtype(Dtype::Float32)?;
    let ms = mean_axis(&multiply(&x32, &x32)?, -1, true)?;
    let rrms = add(&ms, scalar(eps))?.rsqrt()?;
    let scale = add(&weight.as_dtype(Dtype::Float32)?, scalar(1.0))?;
    Ok(multiply(&multiply(&x32, &rrms)?, &scale)?.as_dtype(dtype)?)
}

/// `x · (1 + scale)` with the literal `1` in `scale`'s dtype (upstream's bf16 `1 + scale`).
fn scale_residual(x: &Array, scale: &Array) -> Result<Array> {
    let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
    Ok(multiply(x, &add(&one, scale)?)?)
}

/// Adjacent-pair complex rotation of `x` `[B, S, H, D]` by `cos`/`sin` `[S, D/2]`, in f32.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let (b, s, h, d) = (shape[0], shape[1], shape[2], shape[3]);
    let pairs = x.as_dtype(Dtype::Float32)?.reshape(&[b, s, h, d / 2, 2])?;
    let parts = split(&pairs, 2, 4)?;
    let real = parts[0].squeeze_axes(&[4])?;
    let imag = parts[1].squeeze_axes(&[4])?;
    let cos = cos.reshape(&[1, s, 1, d / 2])?;
    let sin = sin.reshape(&[1, s, 1, d / 2])?;
    let (out_real, out_imag) = rope_rotate(&real, &imag, &cos, &sin)?;
    let out = stack_axis(&[out_real, out_imag], 4)?.reshape(&[b, s, h, d])?;
    Ok(out.as_dtype(dtype)?)
}

struct Attention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    eps: f32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        let lin =
            |name: &str| mlx_gen::quant::lin(w, &format!("{prefix}.{name}"), false, GROUP_SIZE);
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out.0")?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: cfg.attention_head_dim as i32,
            eps: cfg.eps,
        })
    }

    fn linears(&mut self) -> [&mut AdaptableLinear; 4] {
        [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
        ]
    }

    /// Block-causal attention over the joint sequence `x` `[B, S, inner]`.
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, s) = (shape[0], shape[1]);
        let heads = |y: Array| y.reshape(&[b, s, self.heads, self.head_dim]);
        let q = heads(self.to_q.forward(x)?)?;
        let k = heads(self.to_k.forward(x)?)?;
        let v = heads(self.to_v.forward(x)?)?;
        let q = rms_norm(&q, &self.norm_q, self.eps)?.as_dtype(v.dtype())?;
        let k = rms_norm(&k, &self.norm_k, self.eps)?.as_dtype(v.dtype())?;
        let q = apply_rope(&q, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let scale = (self.head_dim as f32).powf(-0.5);

        let mut outputs = Vec::with_capacity(prefix_segments.len() + 1);
        for &(start, end, is_text) in prefix_segments {
            let (start, end) = (start as i32, end as i32);
            let qs = q.index((.., .., start..end, ..));
            let ks = k.index((.., .., 0..end, ..));
            let vs = v.index((.., .., 0..end, ..));
            // A text run attends causally within itself and to everything before it; MLX's causal
            // mask aligns the diagonal to the last key, which is exactly that when the keys are
            // `[0, end)` and the queries `[start, end)`. An image block sees all of `[0, end)`.
            let mask = is_text.then_some(ScaledDotProductAttentionMask::Causal);
            outputs.push(scaled_dot_product_attention(
                &qs, &ks, &vs, scale, mask, None,
            )?);
        }
        let prefix_len = prefix_len as i32;
        let qt = q.index((.., .., prefix_len..s, ..));
        outputs.push(scaled_dot_product_attention(
            &qt, &k, &v, scale, None, None,
        )?);
        let refs: Vec<&Array> = outputs.iter().collect();
        let out = concatenate_axis(&refs, 2)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.heads * self.head_dim])?;
        self.to_out.forward(&out)
    }
}

struct FeedForward {
    gate_layer: AdaptableLinear,
    proj: AdaptableLinear,
    out: AdaptableLinear,
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let lin =
            |name: &str| mlx_gen::quant::lin(w, &format!("{prefix}.{name}"), false, GROUP_SIZE);
        Ok(Self {
            gate_layer: lin("gate_layer")?,
            proj: lin("proj")?,
            out: lin("out")?,
        })
    }

    /// `out(silu(gate(x)) · proj(x))`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate_layer.forward(x)?)?, &self.proj.forward(x)?)?;
        self.out.forward(&gated)
    }
}

/// Optional per-stage capture for the parity tests: every named intermediate, cast to f32, in
/// forward order. `None` costs nothing on the production path.
pub struct Trace<'a>(Option<&'a mut Vec<(String, Array)>>);

impl Trace<'_> {
    fn push(&mut self, name: impl Into<String>, value: &Array) -> Result<()> {
        if let Some(sink) = self.0.as_deref_mut() {
            sink.push((name.into(), value.as_dtype(Dtype::Float32)?));
        }
        Ok(())
    }
}

/// Per-token modulation rows for one forward: `[1, S, inner]` each, already selected between the
/// sampled-timestep row (target tokens) and the `t = 0` row (prefix tokens).
struct Modulation {
    scale1: Array,
    gate1: Array,
    scale2: Array,
    gate2: Array,
}

struct Block {
    attn: Attention,
    mlp: FeedForward,
    eps: f32,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            attn: Attention::from_weights(w, &format!("{prefix}.attn"), cfg)?,
            mlp: FeedForward::from_weights(w, &format!("{prefix}.img_mlp"))?,
            eps: cfg.eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        index: usize,
        x: &Array,
        m: &Modulation,
        cos: &Array,
        sin: &Array,
        prefix_segments: &[(usize, usize, bool)],
        prefix_len: usize,
        trace: &mut Trace<'_>,
    ) -> Result<Array> {
        let h = scale_residual(&layer_norm(x, None, None, self.eps)?, &m.scale1)?;
        let a = self
            .attn
            .forward(&h, cos, sin, prefix_segments, prefix_len)?;
        trace.push(format!("block_{index}_attn"), &a)?;
        let x = add(x, &multiply(&tanh(&m.gate1)?, &a)?)?;
        let h = scale_residual(&layer_norm(&x, None, None, self.eps)?, &m.scale2)?;
        let f = self.mlp.forward(&h)?;
        trace.push(format!("block_{index}_mlp"), &f)?;
        let out = add(&x, &multiply(&tanh(&m.gate2)?, &f)?)?;
        trace.push(format!("block_{index}_out"), &out)?;
        Ok(out)
    }
}

/// The single-stream Qwen-Image 2.1 transformer.
pub struct QwenImage21Transformer {
    cfg: TransformerConfig,
    img_in: AdaptableLinear,
    txt_norm: Array,
    txt_in: AdaptableLinear,
    txt_out: AdaptableLinear,
    time_in: AdaptableLinear,
    time_out: AdaptableLinear,
    modulation: AdaptableLinear,
    blocks: Vec<Block>,
    norm_out: AdaptableLinear,
    proj_out: AdaptableLinear,
}

impl QwenImage21Transformer {
    /// Build from diffusers-keyed weights (`img_in.weight`, `transformer_blocks.{i}.…`, …).
    pub fn from_weights(w: &Weights, cfg: &TransformerConfig) -> Result<Self> {
        let lin = |name: &str| mlx_gen::quant::lin(w, name, false, GROUP_SIZE);
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::from_weights(
                w,
                &format!("transformer_blocks.{i}"),
                cfg,
            )?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            img_in: lin("img_in")?,
            txt_norm: w.require("txt_in.text_norm.weight")?.clone(),
            txt_in: lin("txt_in.in_layer")?,
            txt_out: lin("txt_in.out_layer")?,
            time_in: lin("time_text_embed.timestep_embedder.linear_1")?,
            time_out: lin("time_text_embed.timestep_embedder.linear_2")?,
            modulation: lin("modulation.1")?,
            blocks,
            norm_out: lin("norm_out.linear")?,
            proj_out: lin("proj_out")?,
        })
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.cfg
    }

    /// The dtype the model computes in — its weight dtype (bf16 released, f32 fixtures).
    pub fn compute_dtype(&self) -> Dtype {
        self.img_in.weight_dtype().unwrap_or(Dtype::Float32)
    }

    /// Quantize every Linear to Q4/Q8; the two per-head norms and the text norm stay dense. The
    /// released DiT's input widths (64 / 4096 / 12288) all take the codebase-wide group size 64;
    /// a Linear whose input width is not a multiple of 64 (only the miniature parity snapshot has
    /// one) takes 32, and a width below 32 stays dense rather than failing the load.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        fn quantize_one(lin: &mut AdaptableLinear, bits: i32) -> Result<()> {
            let width = lin.base_shape()[1];
            let group = if width % GROUP_SIZE == 0 {
                GROUP_SIZE
            } else if width % 32 == 0 {
                32
            } else {
                return Ok(());
            };
            lin.quantize(bits, Some(group))
        }
        for lin in [
            &mut self.img_in,
            &mut self.txt_in,
            &mut self.txt_out,
            &mut self.time_in,
            &mut self.time_out,
            &mut self.modulation,
            &mut self.norm_out,
            &mut self.proj_out,
        ] {
            quantize_one(lin, bits)?;
        }
        for block in &mut self.blocks {
            for lin in block.attn.linears() {
                quantize_one(lin, bits)?;
            }
            for lin in [
                &mut block.mlp.gate_layer,
                &mut block.mlp.proj,
                &mut block.mlp.out,
            ] {
                quantize_one(lin, bits)?;
            }
        }
        Ok(())
    }

    /// The joint RoPE table `(cos, sin)`, each `[S, head_dim/2]` f32, for `layout`.
    pub fn rope(&self, layout: &JointLayout) -> Result<(Array, Array)> {
        let ids = layout.position_ids();
        let flat: Vec<f32> = ids.iter().flatten().map(|&p| p as f32).collect();
        let ids = Array::from_slice(&flat, &[ids.len() as i32, 3]);
        let axes: Vec<i32> = self.cfg.axes_dims_rope.iter().map(|&d| d as i32).collect();
        rope_sincos_from_ids(&ids, &axes, ROPE_THETA)
    }

    /// `[1, target_tokens, in_channels]` latents + `[1, text_len, context_in_dim]` text → the
    /// target's velocity `[1, target_tokens, out_channels]` (f32), for the text-to-image layout.
    pub fn forward(
        &self,
        latents: &Array,
        encoder_hidden_states: &Array,
        timestep: f32,
        height: usize,
        width: usize,
    ) -> Result<Array> {
        let layout =
            JointLayout::text_to_image(encoder_hidden_states.shape()[1] as usize, height, width);
        self.forward_joint(encoder_hidden_states, &[latents], timestep, &layout)
    }

    /// The general joint forward: `text` `[1, L, context_in_dim]` and one `[1, h·w, in_channels]`
    /// array per [`Segment::Image`] of `layout` in order (condition images first, target last).
    /// Returns the **target block's** velocity `[1, target_tokens, out_channels]` in f32.
    pub fn forward_joint(
        &self,
        text: &Array,
        images: &[&Array],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<Array> {
        self.run_joint(text, images, timestep, layout, Trace(None))
    }

    /// [`Self::forward_joint`] that also captures every named intermediate (`img_in`, `txt_in`,
    /// `temb`, `modulation`, `block_{i}_attn` / `_mlp` / `_out`, `norm_out`, `proj_out`) in
    /// forward order — the localisation seam the parity tests read. Returns the velocity and the
    /// trace.
    pub fn forward_joint_traced(
        &self,
        text: &Array,
        images: &[&Array],
        timestep: f32,
        layout: &JointLayout,
    ) -> Result<(Array, Vec<(String, Array)>)> {
        let mut trace = Vec::new();
        let velocity = self.run_joint(text, images, timestep, layout, Trace(Some(&mut trace)))?;
        Ok((velocity, trace))
    }

    fn run_joint(
        &self,
        text: &Array,
        images: &[&Array],
        timestep: f32,
        layout: &JointLayout,
        mut trace: Trace<'_>,
    ) -> Result<Array> {
        layout.validate()?;
        let dtype = self.compute_dtype();
        let image_segments = layout
            .segments
            .iter()
            .filter(|s| matches!(s, Segment::Image { .. }))
            .count();
        if images.len() != image_segments {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: layout has {image_segments} image blocks but {} latent arrays were given",
                images.len()
            )));
        }
        let text_len: usize = layout
            .segments
            .iter()
            .map(|s| match s {
                Segment::Text { len } => *len,
                Segment::Image { .. } => 0,
            })
            .sum();
        if text.shape()[1] as usize != text_len {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: layout expects {text_len} text tokens, got {}",
                text.shape()[1]
            )));
        }

        // Projections into the joint width.
        let txt = zero_center_rms_norm(&text.as_dtype(dtype)?, &self.txt_norm, self.cfg.eps)?;
        let txt = self
            .txt_out
            .forward(&gelu_tanh(&self.txt_in.forward(&txt)?)?)?;
        trace.push("txt_in", &txt)?;
        let mut pieces: Vec<Array> = Vec::with_capacity(layout.segments.len());
        let (mut text_cursor, mut image_cursor) = (0i32, 0usize);
        for segment in &layout.segments {
            match *segment {
                Segment::Text { len } => {
                    let len = len as i32;
                    pieces.push(txt.index((.., text_cursor..text_cursor + len, ..)));
                    text_cursor += len;
                }
                Segment::Image { height, width } => {
                    let img = images[image_cursor];
                    if img.shape()[1] as usize != height * width {
                        return Err(Error::Msg(format!(
                            "qwen_image_2_1: image block {image_cursor} expects {} tokens, got {}",
                            height * width,
                            img.shape()[1]
                        )));
                    }
                    let projected = self.img_in.forward(&img.as_dtype(dtype)?)?;
                    trace.push("img_in", &projected)?;
                    pieces.push(projected);
                    image_cursor += 1;
                }
            }
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        let mut x = concatenate_axis(&refs, 1)?;
        let s = x.shape()[1];

        let (cos, sin) = self.rope(layout)?;

        // Timestep rows: the sampled `t` and, under `causal_condition`, an extra `t = 0` row.
        let ts: Vec<f32> = if self.cfg.causal_condition {
            vec![timestep * TIME_FACTOR, 0.0]
        } else {
            vec![timestep * TIME_FACTOR]
        };
        let proj = timestep_sincos(
            &Array::from_slice(&ts, &[ts.len() as i32]),
            TIMESTEP_DIM,
            10_000.0,
            0.0,
        )?
        .as_dtype(dtype)?;
        let temb = self
            .time_out
            .forward(&silu(&self.time_in.forward(&proj)?)?)?;
        trace.push("temb", &temb)?;
        let silu_temb = silu(&temb)?;
        let modulation = self.modulation.forward(&silu_temb)?; // [rows, 4·inner]
        trace.push("modulation", &modulation)?;
        let inner = self.cfg.inner_dim() as i32;
        let target_mask: Vec<bool> = layout.target_mask();
        let select = |rows: &Array| -> Result<Array> {
            // `_select_modulation_rows`: target tokens take row 0 (their sample's timestep), every
            // other token the trailing `t = 0` row.
            let real = rows.index((0..1, ..)).reshape(&[1, 1, inner])?;
            if !self.cfg.causal_condition {
                return Ok(broadcast_to(&real, &[1, s, inner])?);
            }
            let zero = rows.index((1..2, ..)).reshape(&[1, 1, inner])?;
            let mask = Array::from_slice(&target_mask, &[1, s, 1]);
            Ok(r#where(&mask, &real, &zero)?)
        };
        let chunks = split(&modulation, 4, 1)?;
        let m = Modulation {
            scale1: select(&chunks[0])?,
            gate1: select(&chunks[1])?,
            scale2: select(&chunks[2])?,
            gate2: select(&chunks[3])?,
        };
        let out_scale = select(&self.norm_out.forward(&silu_temb)?)?;

        let prefix_segments = layout.prefix_segments();
        let prefix_len = layout.prefix_len();
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                index,
                &x,
                &m,
                &cos,
                &sin,
                &prefix_segments,
                prefix_len,
                &mut trace,
            )?;
        }
        let x = scale_residual(&layer_norm(&x, None, None, self.cfg.eps)?, &out_scale)?;
        trace.push("norm_out", &x)?;
        let out = self.proj_out.forward(&x)?;
        trace.push("proj_out", &out)?;
        let target = layout.target_tokens() as i32;
        Ok(out
            .index((.., s - target..s, ..))
            .as_dtype(Dtype::Float32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t2i_layout_positions_follow_upstream_rope() {
        // text 3, target 2x4: text positions 0,1,2 on every axis; image frame 3, height in
        // range(-1, 1), width in range(-2, 2), row-major.
        let layout = JointLayout::text_to_image(3, 2, 4);
        let ids = layout.position_ids();
        assert_eq!(ids.len(), 11);
        assert_eq!(&ids[..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(ids[3], [3, -1, -2]);
        assert_eq!(ids[6], [3, -1, 1]);
        assert_eq!(ids[7], [3, 0, -2]);
        assert_eq!(ids[10], [3, 0, 1]);
        assert_eq!(layout.prefix_len(), 3);
        assert_eq!(layout.target_tokens(), 8);
        assert_eq!(layout.prefix_segments(), vec![(0, 3, true)]);
        let mask = layout.target_mask();
        assert!(!mask[2] && mask[3] && mask[10]);
    }

    #[test]
    fn condition_image_blocks_keep_their_own_segments() {
        let layout = JointLayout {
            segments: vec![
                Segment::Text { len: 2 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
            ],
        };
        assert_eq!(
            layout.prefix_segments(),
            vec![(0, 2, true), (2, 6, false), (6, 7, true)]
        );
        let ids = layout.position_ids();
        // After the first block the shared position advanced by max(2, 2) = 2 → the trailing text
        // token sits at 4 and the target block freezes its frame axis at 5.
        assert_eq!(ids[6], [4, 4, 4]);
        assert_eq!(ids[7], [5, -1, -1]);
    }
}
