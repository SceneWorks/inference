//! `IrisDiT` — the dual-level pixel-space diffusion transformer (`iris3b/models/dit.py` @
//! [`crate::UPSTREAM_CODE_REVISION`]): patch embed + timestep embed + layerwise text adapter →
//! hybrid dual-/single-stream trunk with shared-bias adaLN → timestep re-fusion → PiT pixel head →
//! `fold` back to an RGB velocity. No VAE.
//!
//! Every module mirrors its upstream class one-to-one under the upstream state-dict key names, so
//! the released `model.safetensors` loads unchanged. Precision: see [`crate::nn`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use gen_core::iris::{self, ModelConfig};
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::nn::{gelu, sigmoid, silu};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::retention::{Attention, Kind, LinearShape, NormInput, Retention};
use crate::stream::{BlockRead, CastPolicy};

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
pub(crate) struct DualBlock {
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
pub(crate) struct SingleBlock {
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

pub(crate) enum Block {
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

impl Block {
    fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        match self {
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
        out
    }
}

impl PitBlock {
    fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        out.extend(self.norm1.arrays());
        out.extend(self.norm2.arrays());
        out.extend(self.adaln.arrays());
        out.extend(self.compress.arrays());
        out.extend(self.expand.arrays());
        out.extend(self.attn.arrays());
        out.extend(self.fc1.arrays());
        out.extend(self.fc2.arrays());
        out
    }
}

impl LayerBlock {
    fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        out.extend(self.norm1.arrays());
        out.extend(self.attn.arrays());
        out.extend(self.norm2.arrays());
        out.extend(self.mlp0.arrays());
        out.extend(self.mlp2.arrays());
        out
    }
}

impl AdapterBlock {
    fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        out.extend(self.norm1.arrays());
        out.extend(self.attn.arrays());
        out.extend(self.norm2.arrays());
        out.extend(self.mlp.arrays());
        out
    }
}

/// Labels of the iOS execution quanta of one backbone forward, in forward order.
pub(crate) mod quanta {
    pub const EMBED: &str = "Iris patch/timestep embed";
    pub const TEXT_LAYER_BLOCK: &str = "Iris text adapter layer block";
    pub const TEXT_POOL: &str = "Iris text adapter pool";
    pub const TEXT_MASK: &str = "Iris text key mask";
    pub const TEXT_REFINER_BLOCK: &str = "Iris text refiner block";
    pub const TEXT_OUTPUT: &str = "Iris text adapter output";
    pub const TRUNK_BLOCK: &str = "Iris trunk block";
    pub const TRUNK_POSITIONS: &str = "Iris trunk positions";
    pub const PIXEL_EMBED: &str = "Iris pixel embed";
    pub const PIXEL_POSITIONS: &str = "Iris pixel positions";
    pub const PIXEL_BLOCK: &str = "Iris pixel block";
    pub const OUTPUT: &str = "Iris output projection";
    /// A streamed backbone's window read (rung 4): the window's blocks read, cast and built.
    pub const BLOCK_READ: &str = "Iris backbone block read";
    /// The per-row PiT stage's output rows concatenated back into the batch (rung 3).
    pub const OUTPUT_ROWS: &str = "Iris output rows";
    /// Every label above, for the phase attribution's coverage test.
    pub const ALL: [&str; 14] = [
        EMBED,
        TEXT_LAYER_BLOCK,
        TEXT_POOL,
        TEXT_MASK,
        TEXT_REFINER_BLOCK,
        TEXT_OUTPUT,
        TRUNK_POSITIONS,
        TRUNK_BLOCK,
        PIXEL_EMBED,
        PIXEL_POSITIONS,
        PIXEL_BLOCK,
        OUTPUT,
        BLOCK_READ,
        OUTPUT_ROWS,
    ];
}

/// The four block stacks of the backbone, in forward order: the text adapter's layerwise blocks
/// and refiner blocks, the trunk, the PiT pixel head. Rung 4 streams each of them by window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Stack {
    Layer,
    Refiner,
    Trunk,
    Pixel,
}

impl Stack {
    pub(crate) const ALL: [Stack; 4] = [Stack::Layer, Stack::Refiner, Stack::Trunk, Stack::Pixel];

    /// The checkpoint key stem of the stack's blocks (`{stem}.{i}.*`).
    pub(crate) const fn stem(self) -> &'static str {
        match self {
            Stack::Layer => "y_embedder.layer_blocks",
            Stack::Refiner => "y_embedder.refiner.blocks",
            Stack::Trunk => "blocks",
            Stack::Pixel => "pixel_blocks",
        }
    }

    const fn index(self) -> usize {
        match self {
            Stack::Layer => 0,
            Stack::Refiner => 1,
            Stack::Trunk => 2,
            Stack::Pixel => 3,
        }
    }

    /// The stack and block index a backbone key (export prefix stripped) belongs to, if any.
    pub(crate) fn of(name: &str) -> Option<(Stack, usize)> {
        Stack::ALL.into_iter().find_map(|stack| {
            let rest = name.strip_prefix(stack.stem())?.strip_prefix('.')?;
            let (index, tail) = rest.split_once('.')?;
            (!tail.is_empty())
                .then(|| index.parse().ok())
                .flatten()
                .map(|i| (stack, i))
        })
    }

    /// Blocks of this stack under `cfg` (the text adapter has two layerwise and two refiner blocks).
    pub(crate) fn depth(self, cfg: &ModelConfig) -> usize {
        match self {
            Stack::Layer | Stack::Refiner => TEXT_ADAPTER_BLOCKS,
            Stack::Trunk => cfg.depth,
            Stack::Pixel => cfg.pixel.depth,
        }
    }
}

/// A test-build hook run as each streamed window opens.
#[cfg(test)]
pub(crate) type WindowHook = Box<dyn FnMut(Stack, usize)>;

#[cfg(test)]
thread_local! {
    /// Called as each streamed window opens, before its read (test builds only): the residency
    /// tests read MLX's active memory there.
    pub(crate) static WINDOW_HOOK: std::cell::RefCell<Option<WindowHook>> =
        const { std::cell::RefCell::new(None) };
}

/// `lap_blocks2`: two layerwise blocks, two refiner blocks.
const TEXT_ADAPTER_BLOCKS: usize = 2;

/// Per stack, per block: the checkpoint's `(file key, built key)` pairs and the block's read price,
/// from the safetensors headers alone (`file_prefix` is the export's key prefix, `pixel.` for
/// depth). Shared by the streamed loader and the iOS pricing, so both read one inventory.
pub(crate) struct StackInventory {
    pub keys: [Vec<Vec<(String, String)>>; 4],
    pub reads: [Vec<BlockRead>; 4],
}

impl StackInventory {
    pub(crate) fn from_headers(
        headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
        file_prefix: &str,
        cfg: &ModelConfig,
        policy: CastPolicy,
    ) -> Result<Self> {
        let mut grouped: [Vec<Vec<&mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>>; 4] =
            Stack::ALL.map(|stack| vec![Vec::new(); stack.depth(cfg)]);
        let mut keys: [Vec<Vec<(String, String)>>; 4] =
            Stack::ALL.map(|stack| vec![Vec::new(); stack.depth(cfg)]);
        for header in headers {
            let Some(name) = header.name.strip_prefix(file_prefix) else {
                continue;
            };
            let Some((stack, i)) = Stack::of(name) else {
                continue;
            };
            let slot = grouped[stack.index()].get_mut(i).ok_or_else(|| {
                Error::Msg(format!(
                    "iris: the checkpoint carries {} block {i}, but the config has {}",
                    stack.stem(),
                    stack.depth(cfg)
                ))
            })?;
            slot.push(header);
            keys[stack.index()][i].push((header.name.clone(), name.to_owned()));
        }
        for stack in Stack::ALL {
            if let Some(i) = keys[stack.index()].iter().position(Vec::is_empty) {
                return Err(Error::Msg(format!(
                    "iris: the checkpoint has no {} block {i}",
                    stack.stem()
                )));
            }
        }
        for stack_keys in &mut keys {
            for block in stack_keys {
                block.sort();
            }
        }
        let reads = Stack::ALL.map(|stack| {
            grouped[stack.index()]
                .iter()
                .map(|block| crate::stream::block_read(block, policy))
                .collect()
        });
        Ok(Self { keys, reads })
    }

    pub(crate) fn reads(&self, stack: Stack) -> &[BlockRead] {
        &self.reads[stack.index()]
    }
}

/// Rung 4 of a backbone: the checkpoint its block stacks are read from per window, and what one
/// window of each stack reads and holds.
pub(crate) struct BlockStream {
    path: PathBuf,
    compute: Dtype,
    inventory: StackInventory,
    /// The window a forward that names none runs (the encoder-free routes' published window; the
    /// generation route's whole stack — a deferred load that did not select rung 4 still never
    /// keeps a stack resident across forwards).
    default_window: Option<usize>,
}

impl BlockStream {
    /// The stream over `path`'s block stacks (export key prefix `file_prefix`), at `compute`.
    pub(crate) fn open(
        path: &Path,
        file_prefix: &str,
        cfg: &ModelConfig,
        compute: Dtype,
        default_window: Option<usize>,
    ) -> Result<Self> {
        let headers = crate::stream::headers(path)?;
        let inventory = StackInventory::from_headers(
            &headers,
            file_prefix,
            cfg,
            CastPolicy::Backbone { compute },
        )?;
        Ok(Self {
            path: path.to_path_buf(),
            compute,
            inventory,
            default_window,
        })
    }

    pub(crate) fn len(&self, stack: Stack) -> usize {
        self.inventory.keys[stack.index()].len()
    }

    /// The read price of block `i` of `stack`.
    #[cfg(test)]
    pub(crate) fn read(&self, stack: Stack, i: usize) -> BlockRead {
        self.inventory.reads(stack)[i]
    }

    /// The checkpoint the stacks are read from.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Read block `i` of `stack` out of `view` and build it with the resident constructors.
    pub(crate) fn build(
        &self,
        view: &mut Weights,
        cfg: &ModelConfig,
        stack: Stack,
        i: usize,
    ) -> Result<Built> {
        let map = crate::stream::materialize(
            view,
            &self.inventory.keys[stack.index()][i],
            CastPolicy::Backbone {
                compute: self.compute,
            },
        )?;
        let l = Loader {
            weights: &map,
            compute: self.compute,
            adapted: None,
            quant: None,
        };
        let eps = cfg.norm_eps as f32;
        let gqa = cfg.num_kv_heads.is_some_and(|kv| kv != cfg.num_heads);
        let built = match stack {
            Stack::Trunk => Built::Trunk(Box::new(IrisDiT::load_block(&l, cfg, i, eps, gqa)?)),
            Stack::Pixel => Built::Pixel(Box::new(IrisDiT::load_pit(&l, cfg, i, eps)?)),
            Stack::Layer => Built::Layer(Box::new(IrisDiT::load_layer_block(&l, cfg, i, eps)?)),
            Stack::Refiner => {
                Built::Refiner(Box::new(IrisDiT::load_refiner_block(&l, cfg, i, eps)?))
            }
        };
        let unused = map.unused_keys();
        if let Some(first) = unused.first() {
            return Err(Error::Msg(format!(
                "iris stream: {} block {i} carries `{first}`, which the block does not consume",
                stack.stem()
            )));
        }
        crate::stream::eval_all(&built.arrays())?;
        Ok(built)
    }
}

/// One block a streamed window built (boxed: a window holds a handful at most).
pub(crate) enum Built {
    Trunk(Box<Block>),
    Pixel(Box<PitBlock>),
    Layer(Box<LayerBlock>),
    Refiner(Box<AdapterBlock>),
}

impl Built {
    pub(crate) fn arrays(&self) -> Vec<&Array> {
        match self {
            Built::Trunk(b) => b.arrays(),
            Built::Pixel(b) => b.arrays(),
            Built::Layer(b) => b.arrays(),
            Built::Refiner(b) => b.arrays(),
        }
    }

    fn as_ref(&self) -> BlockRef<'_> {
        match self {
            Built::Trunk(b) => BlockRef::Trunk(b),
            Built::Pixel(b) => BlockRef::Pixel(b),
            Built::Layer(b) => BlockRef::Layer(b),
            Built::Refiner(b) => BlockRef::Refiner(b),
        }
    }
}

/// A block of one of the four stacks, resident or just built by a window.
#[derive(Clone, Copy)]
enum BlockRef<'a> {
    Trunk(&'a Block),
    Pixel(&'a PitBlock),
    Layer(&'a LayerBlock),
    Refiner(&'a AdapterBlock),
}

/// How one backbone forward executes: the attention plan (rung 3's query-row bound and the
/// request's cancel flag), the streamed window (rung 4), and whether the PiT stage runs one batch
/// row at a time (rung 3's pixel-row bound). [`Execution::resident`] is the historical forward.
#[derive(Clone, Copy)]
pub struct Execution<'a> {
    pub plan: mlx_gen::AttentionPlan<'a>,
    /// Blocks per streamed window; `None` runs the stream's default. A resident backbone ignores it.
    pub window: Option<usize>,
    /// Run the PiT blocks and the output projection one batch row at a time. The PiT attention
    /// mixes patches within a row only, so the rows are independent: equal to the batched stage up
    /// to matmul blocking, with the stage's activation bounded by one row instead of the batch.
    pub pixel_rows: bool,
}

impl<'a> Execution<'a> {
    /// `plan`, no window, batched pixel stage — the forward every caller ran before rung 4.
    pub const fn resident(plan: mlx_gen::AttentionPlan<'a>) -> Self {
        Self {
            plan,
            window: None,
            pixel_rows: false,
        }
    }
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
pub(crate) struct PitBlock {
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
pub(crate) struct LayerBlock {
    norm1: RmsNorm,
    attn: SelfAttention,
    norm2: RmsNorm,
    mlp0: Linear,
    mlp2: Linear,
}

/// `TextAdapterBlock` (masked keys).
pub(crate) struct AdapterBlock {
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
    /// Actual attached payload facts, keyed identically to header-only pricing.
    adapter_facts: crate::adapters::ProjectionAdapterFacts,
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
    /// Rung 4: the block stacks are read per window from this checkpoint (`None`: resident).
    stream: Option<BlockStream>,
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

/// One SwiGLU MLP request: `rows` tokens of width `d` through hidden `ff` (down-projection input
/// pad `pad`), with an optional post-norm; `bound` when the block binds the MLP output.
struct SwigluMlp<'a> {
    rows: u64,
    d: u64,
    ff: u64,
    pad: u64,
    post_norm: bool,
    bound: bool,
    path: &'a str,
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
        Self::build(w, cfg, compute, input_channels, None, None, None)
    }

    /// The skeleton of a streamed backbone (rung 4): `w` holds every tensor but the four block
    /// stacks, which `stream` reads per window (`BlockStream`). The skeleton is built by the
    /// same constructors as the resident backbone; the stacks by the same per-block ones.
    pub(crate) fn from_weights_streamed(
        w: &Weights,
        cfg: &ModelConfig,
        compute: Dtype,
        input_channels: usize,
        stream: BlockStream,
    ) -> Result<Self> {
        Self::build(w, cfg, compute, input_channels, None, None, Some(stream))
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
        let dit = Self::build(w, cfg, compute, cfg.in_channels, adapted, quant, None)?;
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
        stream: Option<BlockStream>,
    ) -> Result<Self> {
        let adapter_facts = adapted
            .map(AdaptedLinears::adapter_facts)
            .unwrap_or_default();
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
        // A streamed backbone builds no stack here: each block is built per window by
        // `BlockStream::build`, through these same constructors.
        let resident = stream.is_none();
        let depth = |stack: Stack| if resident { stack.depth(cfg) } else { 0 };
        let mut blocks = Vec::with_capacity(depth(Stack::Trunk));
        for i in 0..depth(Stack::Trunk) {
            blocks.push(Self::load_block(&l, cfg, i, eps, gqa)?);
        }
        let mut layer_blocks = Vec::with_capacity(depth(Stack::Layer));
        for i in 0..depth(Stack::Layer) {
            layer_blocks.push(Self::load_layer_block(&l, cfg, i, eps)?);
        }
        let mut adapter_blocks = Vec::with_capacity(depth(Stack::Refiner));
        for i in 0..depth(Stack::Refiner) {
            adapter_blocks.push(Self::load_refiner_block(&l, cfg, i, eps)?);
        }
        let y_embedder = TextAdapter {
            layer_blocks,
            layer_pool: l.linear("y_embedder.layer_pool", true)?,
            proj: l.linear("y_embedder.refiner.proj", true)?,
            blocks: adapter_blocks,
            norm: l.norm("y_embedder.refiner.norm", eps)?,
        };

        let mut pixel_blocks = Vec::with_capacity(depth(Stack::Pixel));
        for i in 0..depth(Stack::Pixel) {
            pixel_blocks.push(Self::load_pit(&l, cfg, i, eps)?);
        }

        let dit = Self {
            cfg: cfg.clone(),
            compute,
            adapter_facts,
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
            stream,
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

    /// Load text-adapter layerwise block `i` (`y_embedder.layer_blocks.{i}.*`) from `l`.
    fn load_layer_block(l: &Loader, cfg: &ModelConfig, i: usize, eps: f32) -> Result<LayerBlock> {
        let p = |name: &str| format!("y_embedder.layer_blocks.{i}.{name}");
        Ok(LayerBlock {
            norm1: l.norm(&p("norm1"), eps)?,
            attn: SelfAttention::load(l, &p("attn"), cfg.text_lap_num_heads, false, eps)?,
            norm2: l.norm(&p("norm2"), eps)?,
            mlp0: l.linear(&p("mlp.0"), true)?,
            mlp2: l.linear(&p("mlp.2"), true)?,
        })
    }

    /// Load text-refiner block `i` (`y_embedder.refiner.blocks.{i}.*`) from `l`.
    fn load_refiner_block(
        l: &Loader,
        cfg: &ModelConfig,
        i: usize,
        eps: f32,
    ) -> Result<AdapterBlock> {
        let p = |name: &str| format!("y_embedder.refiner.blocks.{i}.{name}");
        Ok(AdapterBlock {
            norm1: l.norm(&p("norm1"), eps)?,
            attn: SelfAttention::load(l, &p("attn"), cfg.num_heads, cfg.qk_norm, eps)?,
            norm2: l.norm(&p("norm2"), eps)?,
            mlp: SwiGlu::load(l, &p("mlp"))?,
        })
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

    /// Every parameter array, for an eager evaluation at load (never inside a forward). A streamed
    /// backbone (rung 4) holds only its skeleton; its block stacks are read per window.
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        out.extend(self.s_embedder.arrays());
        out.extend(self.t_mlp0.arrays());
        out.extend(self.t_mlp2.arrays());
        let y = &self.y_embedder;
        for b in &y.layer_blocks {
            out.extend(b.arrays());
        }
        out.extend(y.layer_pool.arrays());
        out.extend(y.proj.arrays());
        for b in &y.blocks {
            out.extend(b.arrays());
        }
        out.extend(y.norm.arrays());
        out.push(&self.y_pos_embedding);
        out.extend(self.core_img.arrays());
        out.extend(self.core_txt.iter().flat_map(Linear::arrays));
        for block in &self.blocks {
            out.extend(block.arrays());
        }
        out.extend(self.pixel_proj.arrays());
        for b in &self.pixel_blocks {
            out.extend(b.arrays());
        }
        out.extend(self.final_norm.arrays());
        out.extend(self.final_linear.arrays());
        out
    }

    /// Whether the block stacks are streamed by window (rung 4) rather than resident.
    pub fn is_streamed(&self) -> bool {
        self.stream.is_some()
    }

    /// Inspect owned stacks, not a selection label: a resident endpoint owns every block and no
    /// window source; a streamed endpoint owns the skeleton and a window source, not the stacks.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn assert_endpoint_ownership(&self, streamed: bool) {
        assert_eq!(self.is_streamed(), streamed, "backbone stream source");
        for (stack, owned) in [
            (Stack::Layer, self.y_embedder.layer_blocks.len()),
            (Stack::Refiner, self.y_embedder.blocks.len()),
            (Stack::Trunk, self.blocks.len()),
            (Stack::Pixel, self.pixel_blocks.len()),
        ] {
            assert_eq!(
                owned,
                if streamed { 0 } else { stack.depth(&self.cfg) },
                "{stack:?}: endpoint owns the wrong number of blocks"
            );
        }
        assert!(
            self.resident_weight_bytes() > 0,
            "the endpoint owns no skeleton/weights"
        );
        let _addresses = self.endpoint_parameter_addresses();
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn endpoint_parameter_addresses(&self) -> Vec<usize> {
        crate::anchor_tests::materialized_addresses(self.arrays())
    }

    /// The streamed backbone's window source (test builds: the read proofs build blocks from it).
    #[cfg(test)]
    pub(crate) fn block_stream(&self) -> Option<&BlockStream> {
        self.stream.as_ref()
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

    pub(crate) fn query_rows(
        plan: mlx_gen::AttentionPlan<'_>,
        batch: i32,
        heads: i32,
        tokens: i32,
    ) -> u64 {
        plan.budget.query_block(batch, heads, tokens, tokens).max(0) as u64
    }

    fn compute_bytes(&self) -> u64 {
        crate::nn::dtype_bytes(self.compute)
    }

    /// Trunk block `i` over `n_patches` image and `n_txt` text tokens.
    #[cfg(test)]
    pub(crate) fn trunk_block_scratch(
        &self,
        i: usize,
        batch: i32,
        n_patches: i32,
        n_txt: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        self.trunk_block_scratch_of(&self.blocks[i], i, batch, n_patches, n_txt, plan)
    }

    /// [`Self::trunk_block_scratch`] of a block built outside the resident stack (a streamed
    /// window's), at its index `i`.
    fn trunk_block_scratch_of(
        &self,
        block: &Block,
        i: usize,
        batch: i32,
        n_patches: i32,
        n_txt: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let (dim, ff) = block.widths();
        let tokens = (n_patches + n_txt).max(0);
        let heads = self.cfg.num_heads as i32;
        let kv_heads = self.cfg.num_kv_heads.unwrap_or(self.cfg.num_heads) as u64;
        let (split_qkv, text, down_pad) = block.layout();
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
        let prefix = format!("blocks.{i}");
        match block {
            Block::Dual(_) => Self::dual_block_bytes_with_adapters(
                &g,
                crate::retention::unfused(),
                &self.adapter_facts,
                &prefix,
            ),
            Block::Single(_) => Self::single_block_bytes_with_adapters(
                &g,
                crate::retention::unfused(),
                &self.adapter_facts,
                &prefix,
            ),
        }
    }

    /// One grouped-query (or fused) q / k / v projection of `rows` tokens from an f32 input,
    /// each output held.
    fn project_qkv(r: &mut Retention, g: &TrunkGeometry, rows: u64, stem: &str, side: &str) {
        let (d, kvw) = (g.dim, g.dim / g.heads.max(1) * g.kv_heads);
        let narrow = r.narrow();
        let path = |role| format!("{stem}{role}{side}");
        if g.split_qkv {
            r.linear_at(
                &path("q_proj"),
                LinearShape {
                    rows,
                    input: d,
                    output: d,
                },
                narrow,
                0,
                Kind::Named,
            )
            .linear_at(
                &path("k_proj"),
                LinearShape {
                    rows,
                    input: d,
                    output: kvw,
                },
                narrow,
                0,
                Kind::Named,
            )
            .linear_at(
                &path("v_proj"),
                LinearShape {
                    rows,
                    input: d,
                    output: kvw,
                },
                narrow,
                0,
                Kind::Named,
            );
        } else {
            r.linear_at(
                &path("qkv"),
                LinearShape {
                    rows,
                    input: d,
                    output: d + 2 * kvw,
                },
                narrow,
                0,
                Kind::Named,
            );
        }
    }

    /// A SwiGLU MLP over `rows` tokens of an f32 input (`w1` / `w3` read their own compute-width
    /// casts; SiLU and the gate product donate), and its output norm when `post_norm`. `bound`: the
    /// block binds the MLP output (held to the end; its norm reads a held input), else the output is
    /// a temporary the norm's result can reuse.
    fn swiglu(r: &mut Retention, mlp: SwigluMlp<'_>) {
        let SwigluMlp {
            rows,
            d,
            ff,
            pad,
            post_norm,
            bound,
            path,
        } = mlp;
        let narrow = r.narrow();
        let out_kind = if bound { Kind::Named } else { Kind::Temp };
        r.linear_at(
            &format!("{path}.w1"),
            LinearShape {
                rows,
                input: d,
                output: ff,
            },
            narrow,
            0,
            Kind::Temp,
        )
        .linear_at(
            &format!("{path}.w3"),
            LinearShape {
                rows,
                input: d,
                output: ff,
            },
            narrow,
            0,
            Kind::Temp,
        )
        .linear_at(
            &format!("{path}.w2"),
            LinearShape {
                rows,
                input: ff,
                output: d,
            },
            false,
            pad,
            out_kind,
        );
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
    #[cfg(test)]
    pub(crate) fn single_block_bytes(g: &TrunkGeometry, unfused: bool) -> u64 {
        Self::single_block_bytes_with_adapters(g, unfused, &Default::default(), "")
    }

    pub(crate) fn single_block_bytes_with_adapters(
        g: &TrunkGeometry,
        unfused: bool,
        facts: &crate::adapters::ProjectionAdapterFacts,
        prefix: &str,
    ) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused).with_adapters(facts, prefix);
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
        Self::project_qkv(&mut r, g, rows, "", "");
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
        r.linear_at(
            "attn_gate",
            LinearShape {
                rows: rows_o,
                input: d,
                output: d,
            },
            narrow,
            0,
            Kind::Named,
        )
        .linear_at(
            "attn_proj",
            LinearShape {
                rows: rows_o,
                input: d,
                output: d,
            },
            false,
            0,
            proj_kind,
        )
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
        Self::swiglu(
            &mut r,
            SwigluMlp {
                rows: rows_o,
                d,
                ff: g.ff,
                pad: g.down_pad,
                post_norm: true,
                bound: true,
                path: "mlp",
            },
        );
        r.price()
    }

    /// `dual`: separate image / text weights, text-first joint attention. Phase 1: both streams'
    /// modulation sums and modulated norms, their q / k / v projections, the four QK-norms and RoPE
    /// applications, the q / k / v concatenations. Phase 2: the attention and the image half's gate,
    /// output projection, post-norm and residual. Phase 3: the image MLP. Phases 4 and 5: the text
    /// half's, when the block has a text path. Every bound handle is held to the end.
    #[cfg(test)]
    pub(crate) fn dual_block_bytes(g: &TrunkGeometry, unfused: bool) -> u64 {
        Self::dual_block_bytes_with_adapters(g, unfused, &Default::default(), "")
    }

    pub(crate) fn dual_block_bytes_with_adapters(
        g: &TrunkGeometry,
        unfused: bool,
        facts: &crate::adapters::ProjectionAdapterFacts,
        prefix: &str,
    ) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused).with_adapters(facts, prefix);
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
        for (stream, narrow_in, side) in [(rows_x, img_narrow, "_x"), (rows_y, false, "_y")] {
            let input = if narrow_in {
                NormInput::Cast
            } else {
                NormInput::Held
            };
            r.norm(stream * d, stream, input, Kind::Named)
                .temp(4, b * d);
            Self::project_qkv(&mut r, g, stream, "attn.", side);
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
            let side = if k == 0 { "x" } else { "y" };
            r.linear_at(
                &format!("attn_gate_{side}"),
                LinearShape {
                    rows: stream,
                    input: d,
                    output: d,
                },
                narrow,
                0,
                Kind::Named,
            )
            .linear_at(
                &format!("attn.proj_{side}"),
                LinearShape {
                    rows: stream,
                    input: d,
                    output: d,
                },
                false,
                0,
                Kind::Named,
            )
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
            Self::swiglu(
                &mut r,
                SwigluMlp {
                    rows: stream,
                    d,
                    ff: g.ff,
                    pad: g.down_pad,
                    post_norm: true,
                    bound: false,
                    path: &format!("mlp_{side}"),
                },
            );
        }
        r.price()
    }

    /// PiT block `i` over `n_patches` patches of `p²` pixels.
    #[cfg(test)]
    pub(crate) fn pixel_block_scratch(
        &self,
        i: usize,
        batch: i32,
        n_patches: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        self.pixel_block_scratch_of(&self.pixel_blocks[i], i, batch, n_patches, plan)
    }

    fn pixel_block_scratch_of(
        &self,
        blk: &PitBlock,
        i: usize,
        batch: i32,
        n_patches: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        let heads = blk.attn.heads;
        Self::pixel_block_bytes_with_adapters(
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
            &self.adapter_facts,
            &format!("pixel_blocks.{i}"),
        )
    }

    /// [`Self::pixel_block_scratch`] from the block's widths alone. Phase 1 (to the attention's
    /// q / k / v barrier): the adaLN modulation (held), the norm of the compute-width pixel plane,
    /// the compressed tokens (held), the fused qkv, the QK-norms and RoPE (q and k held into the
    /// attention). Phase 2 (to the residual barrier): the attention, its projection, the expanded
    /// plane (held) and the modulated residual. Phase 3: the norm, `fc1` (GELU donates; bf16: its
    /// result is f32, which `fc2` casts back), `fc2` (held) and the modulated residual.
    pub(crate) fn pixel_block_bytes_with_adapters(
        g: PixelBlockGeometry,
        unfused: bool,
        facts: &crate::adapters::ProjectionAdapterFacts,
        prefix: &str,
    ) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused).with_adapters(facts, prefix);
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
        r.linear_at(
            "adaln",
            LinearShape {
                rows,
                input: g.condition_width,
                output: g.modulation_width,
            },
            narrow,
            0,
            Kind::Named,
        )
        .norm(plane, rows * g.patch_pixels, plane_in, Kind::Temp)
        .linear_at(
            "compress",
            LinearShape {
                rows,
                input: g.patch_pixels * g.pixel_width,
                output: g.compact_width,
            },
            narrow,
            0,
            Kind::Named,
        )
        .linear_at(
            "attn.qkv",
            LinearShape {
                rows,
                input: g.compact_width,
                output: 3 * g.compact_width,
            },
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
        .linear_at(
            "attn.proj",
            LinearShape {
                rows,
                input: g.compact_width,
                output: g.compact_width,
            },
            false,
            0,
            Kind::Named,
        )
        .linear_at(
            "expand",
            LinearShape {
                rows,
                input: g.compact_width,
                output: g.patch_pixels * g.pixel_width,
            },
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
            .linear_at(
                "mlp.fc1",
                LinearShape {
                    rows: mlp_rows,
                    input: g.pixel_width,
                    output: g.fc1_width,
                },
                narrow,
                0,
                Kind::Temp,
            );
        if narrow {
            // MLX's GELU divides by f32 constants: a bf16 input's result is a new f32 buffer.
            r.temp(4, mlp_rows * g.fc1_width);
        }
        r.linear_at(
            "mlp.fc2",
            LinearShape {
                rows: mlp_rows,
                input: g.fc1_width,
                output: g.pixel_width,
            },
            narrow,
            0,
            Kind::Named,
        )
        .temp(cw, plane);
        r.price()
    }

    /// Text-adapter layer block `i`: attention over the `layers` encoder states of each of the
    /// `rows = B · T` tokens.
    #[cfg(test)]
    pub(crate) fn text_layer_block_scratch(
        &self,
        i: usize,
        rows: i32,
        layers: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        self.text_layer_block_scratch_of(&self.y_embedder.layer_blocks[i], i, rows, layers, plan)
    }

    fn text_layer_block_scratch_of(
        &self,
        blk: &LayerBlock,
        i: usize,
        rows: i32,
        layers: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        Self::text_block_bytes_with_adapters(
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
            &self.adapter_facts,
            &format!("y_embedder.layer_blocks.{i}"),
        )
    }

    /// Text-refiner block `i` over `t` masked-key text tokens.
    #[cfg(test)]
    pub(crate) fn text_refiner_block_scratch(
        &self,
        i: usize,
        batch: i32,
        t: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        self.text_refiner_block_scratch_of(&self.y_embedder.blocks[i], i, batch, t, plan)
    }

    fn text_refiner_block_scratch_of(
        &self,
        blk: &AdapterBlock,
        i: usize,
        batch: i32,
        t: i32,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> u64 {
        Self::text_block_bytes_with_adapters(
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
            &self.adapter_facts,
            &format!("y_embedder.refiner.blocks.{i}"),
        )
    }

    /// One text-adapter block from its widths. Phase 1 (to the attention's q / k / v barrier): the
    /// input norm, the fused qkv (held into the attention) and, in a refiner block, the QK-norms
    /// (held). Phase 2 (to the residual barrier): the attention (refiner: under the compute-width
    /// key mask), its projection (bound) and the residual sum (a new buffer at the stream's width:
    /// the projection is bound, or its f32 cast is the sum). Phase 3: the norm, the MLP
    /// (`mlp.0 → SiLU → mlp.2`, or the SwiGLU) and the residual.
    #[cfg(test)]
    pub(crate) fn text_block_bytes(g: TextBlockGeometry, unfused: bool) -> u64 {
        Self::text_block_bytes_with_adapters(g, unfused, &Default::default(), "")
    }

    pub(crate) fn text_block_bytes_with_adapters(
        g: TextBlockGeometry,
        unfused: bool,
        facts: &crate::adapters::ProjectionAdapterFacts,
        prefix: &str,
    ) -> u64 {
        let mut r = Retention::on(g.compute_bytes, unfused).with_adapters(facts, prefix);
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
        r.norm(h, rows, stream_in, input_kind).linear_at(
            "attn.qkv",
            LinearShape {
                rows,
                input: g.dim,
                output: 3 * g.dim,
            },
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
        .linear_at(
            "attn.proj",
            LinearShape {
                rows,
                input: g.dim,
                output: g.dim,
            },
            false,
            0,
            Kind::Named,
        )
        .named(g.stream_bytes, h);
        // Phase 3: the returned graph, run after the block's bindings are gone — it reads the
        // residual stream.
        r.last_phase()
            .hold(g.stream_bytes, h)
            .norm(h, rows, stream_in, Kind::Temp);
        if g.refiner {
            // The SwiGLU output is a temporary: the residual sum reuses it.
            Self::swiglu(
                &mut r,
                SwigluMlp {
                    rows,
                    d: g.dim,
                    ff: g.ff,
                    pad: g.down_pad,
                    post_norm: false,
                    bound: false,
                    path: "mlp",
                },
            );
        } else {
            // `mlp.2`'s output is bound, so the residual sum is a new buffer (bf16: its operand's
            // f32 cast, which the sum reuses).
            r.linear_at(
                "mlp.0",
                LinearShape {
                    rows,
                    input: g.dim,
                    output: g.ff,
                },
                narrow,
                0,
                Kind::Temp,
            )
            .linear_at(
                "mlp.2",
                LinearShape {
                    rows,
                    input: g.ff,
                    output: g.dim,
                },
                false,
                g.down_pad,
                Kind::Named,
            )
            .temp(g.stream_bytes, h);
        }
        r.price()
    }

    /// The positional-construction quantum before the first trunk block: image/text RoPE tables
    /// and the text positional add. RoPE from_slice allocates immediately, so this guard is early.
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
        Self::embed_bytes_with_adapters(
            EmbedGeometry {
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
            },
            &self.adapter_facts,
        )
    }

    /// [`Self::embed_scratch`] from the widths alone (one phase): the patch vectors' re-layout copy
    /// (f32) and the patch projection (held: the embedded tokens); the timestep path — the
    /// frequency table, the f32 phase, its cosine and sine and their concatenation, the first
    /// timestep projection (SiLU donates) and the second (held: `t_emb`); the conditioning SiLU of
    /// the held `t_emb`, and both cores' outputs (held).
    pub(crate) fn embed_bytes_with_adapters(
        g: EmbedGeometry,
        facts: &crate::adapters::ProjectionAdapterFacts,
    ) -> u64 {
        let mut r = Retention::on(g.compute_bytes, false).with_adapters(facts, "");
        let narrow = r.narrow();
        let (b, freqs) = (g.batch, g.sinusoid / 2);
        r.temp(4, b * g.patches * g.patch_in)
            .linear_at(
                "s_embedder.proj",
                LinearShape {
                    rows: b * g.patches,
                    input: g.patch_in,
                    output: g.hidden,
                },
                narrow,
                0,
                Kind::Named,
            )
            .temp(4, freqs)
            .alloc(Kind::Temp, 3, 4, b * freqs)
            .temp(4, b * g.sinusoid)
            .linear_at(
                "t_embedder.mlp.0",
                LinearShape {
                    rows: b,
                    input: g.sinusoid,
                    output: g.t_hidden,
                },
                narrow,
                0,
                Kind::Temp,
            )
            .linear_at(
                "t_embedder.mlp.2",
                LinearShape {
                    rows: b,
                    input: g.t_hidden,
                    output: g.t_out,
                },
                false,
                0,
                Kind::Named,
            )
            .temp(g.compute_bytes, b * g.t_out)
            .named(g.compute_bytes, b * g.cores)
            .adapter_extras_at("modulation_cores.adaln_img", b, false)
            .adapter_extras_at("modulation_cores.adaln_txt", b, false);
        r.price()
    }

    /// The layer pool and the refiner's input projection over `[B·T, layers, Dt]` states.
    pub(crate) fn text_pool_scratch(&self, batch: i32, t: i32, layers: i32, text_dim: i32) -> u64 {
        Self::text_pool_bytes_with_adapters(
            self.compute_bytes(),
            batch.max(0) as u64,
            t.max(0) as u64,
            layers.max(0) as u64,
            text_dim.max(0) as u64,
            self.y_embedder.proj.out_features(),
            &self.adapter_facts,
        )
    }

    /// [`Self::text_pool_scratch`] from the widths alone (one phase): the pool reads the
    /// layer-transposed states through a row-contiguous copy (bf16: of its compute-width cast, which
    /// keeps the transposed layout), the pooled `[B·T, Dt]` and the projection `[B, T, D]`.
    pub(crate) fn text_pool_bytes_with_adapters(
        cw: u64,
        b: u64,
        t: u64,
        layers: u64,
        text_dim: u64,
        out: u64,
        facts: &crate::adapters::ProjectionAdapterFacts,
    ) -> u64 {
        let mut r = Retention::on(cw, false).with_adapters(facts, "");
        let copies = if r.narrow() { 2 } else { 1 };
        r.alloc(Kind::Temp, copies, cw, b * t * layers * text_dim)
            .temp(cw, b * t * text_dim)
            .named(cw, b * t * out)
            .adapter_extras_at("y_embedder.layer_pool", b * t * text_dim, true)
            .adapter_extras_at("y_embedder.refiner.proj", b * t, false);
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
        Self::pixel_embed_bytes_with_adapters(
            self.compute_bytes(),
            batch.max(0) as u64,
            h.max(0) as u64,
            w.max(0) as u64,
            self.cfg.patch_size as u64,
            self.core_img.logical_shape()[1] as u64,
            pixel_width as u64,
            channels as u64,
            &self.adapter_facts,
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
    pub(crate) fn pixel_embed_bytes_with_adapters(
        cw: u64,
        b: u64,
        h: u64,
        w: u64,
        patch: u64,
        hidden: u64,
        pixel_width: u64,
        channels: u64,
        facts: &crate::adapters::ProjectionAdapterFacts,
    ) -> u64 {
        let mut r = Retention::on(cw, false).with_adapters(facts, "");
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
        r.named(cw, pixels * pixel_width)
            .adapter_extras_at("pixel_embedder.proj", pixels, true);
        r.price()
    }

    /// The final norm, the final projection and the fold back to `[B, C, H, W]` f32.
    pub(crate) fn output_scratch(&self, batch: i32, n_patches: i32) -> u64 {
        let [out, input] = self.final_linear.logical_shape();
        Self::output_bytes_with_adapters(
            self.compute_bytes(),
            batch.max(0) as u64,
            n_patches.max(0) as u64,
            (self.cfg.patch_size * self.cfg.patch_size) as u64,
            input as u64,
            out as u64,
            crate::retention::unfused(),
            &self.adapter_facts,
        )
    }

    /// [`Self::output_scratch`] from the widths alone (one phase): the norm of the compute-width
    /// pixel stream, the projection to `[B·N, p², C]` (bf16: of its compute-width cast), the fold's
    /// re-layout copy and (bf16) its f32 cast, the output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn output_bytes_with_adapters(
        cw: u64,
        b: u64,
        n: u64,
        ppp: u64,
        input: u64,
        out: u64,
        unfused: bool,
        facts: &crate::adapters::ProjectionAdapterFacts,
    ) -> u64 {
        let mut r = Retention::on(cw, unfused).with_adapters(facts, "");
        let narrow = r.narrow();
        let rows = b * n * ppp;
        let input_kind = if narrow {
            NormInput::Cast
        } else {
            NormInput::Held
        };
        r.norm(rows * input, rows, input_kind, Kind::Temp)
            .linear_at(
                "final_layer.linear",
                LinearShape {
                    rows,
                    input,
                    output: out,
                },
                narrow,
                0,
                Kind::Temp,
            )
            .named(cw, rows * out);
        if narrow {
            r.named(4, rows * out);
        }
        r.price()
    }

    /// `TimestepEmbedder`: sinusoid bank (period `timestep_max_period`, 256 frequencies, cos‖sin)
    /// → Linear → SiLU → Linear. `t` is model time `[B]` f32 → `[B, 1, D]`.
    ///
    /// Deliberately NOT [`mlx_gen::nn::timestep_sincos`]: the frequencies are built on the host, and the
    /// shared helper's on-device exponent differs by ~1e-7 (enough to flip bf16 ULPs that this crate's
    /// single-host parity goldens pin), so migrating would perturb pinned numerics (sc-27334).
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
        self.text_adapter_exec(
            text,
            &Execution::resident(mlx_gen::AttentionPlan::UNBOUNDED),
        )
    }

    /// Host f32 additive mask plus its immediate MLX copy, before either constructor runs.
    pub(crate) fn text_mask_bytes(batch: u64, tokens: u64) -> u64 {
        let data = mlx_gen::ios_quantum::tensor_bytes(4, &[batch, tokens, tokens]);
        data.saturating_add(crate::retention::allocated(data))
    }

    fn text_adapter_exec(&self, text: &TextBatch, exec: &Execution<'_>) -> Result<Array> {
        let plan = exec.plan;
        let sh = text.states.shape();
        let (b, t, layers, dt) = (sh[0], sh[1], sh[2], sh[3]);
        if text.mask.len() != b as usize || text.mask.iter().any(|m| m.len() != t as usize) {
            return Err(Error::Msg(format!(
                "iris: text mask must be [{b}, {t}] for states {sh:?}"
            )));
        }
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let y = text
            .states
            .as_dtype(Dtype::Float32)?
            .reshape(&[b * t, layers, dt])?;
        let y = self.run_stack(
            Stack::Layer,
            exec,
            y,
            |y, index, block| {
                mlx_gen::ios_quantum::diagnostic_metadata(
                    crate::model::MODEL_ID,
                    "dit_block",
                    &[
                        ("stack", 0),
                        ("index", index as u64),
                        ("row", 0),
                        ("rows", b as u64),
                    ],
                );
                let BlockRef::Layer(blk) = block else {
                    unreachable!("the layerwise stack holds layerwise blocks")
                };
                let scratch = self.text_layer_block_scratch_of(blk, index, b * t, layers, plan);
                mlx_gen::ios_quantum::run(quanta::TEXT_LAYER_BLOCK, scratch, || {
                    Self::text_layer_body_with(blk, self.compute, &y, plan)
                })
            },
            |y| Ok(eval([y])?),
        )?;
        // layer_pool: Linear(L → 1) over the layer axis, then the refiner's input projection.
        let y = mlx_gen::ios_quantum::run(
            quanta::TEXT_POOL,
            self.text_pool_scratch(b, t, layers, dt),
            || self.text_pool_body(&y, b, t, dt),
        )?;
        // refiner (TransformerTextEmbedder): keys masked, every query keeps its own position.
        mlx_gen::ios_quantum::diagnostic_metadata(
            crate::model::MODEL_ID,
            "allocation_region",
            &[("role", 0), ("rows", b as u64), ("positions", t as u64)],
        );
        mlx_gen::ios_quantum::before(quanta::TEXT_MASK, Self::text_mask_bytes(b as u64, t as u64))?;
        let key_mask = key_padding_mask(text.mask, t as usize);
        mlx_gen::ios_quantum::finish(quanta::TEXT_MASK, &[&key_mask])?;
        let y = self.run_stack(
            Stack::Refiner,
            exec,
            y,
            |y, index, block| {
                mlx_gen::ios_quantum::diagnostic_metadata(
                    crate::model::MODEL_ID,
                    "dit_block",
                    &[
                        ("stack", 1),
                        ("index", index as u64),
                        ("row", 0),
                        ("rows", b as u64),
                    ],
                );
                let BlockRef::Refiner(blk) = block else {
                    unreachable!("the refiner stack holds refiner blocks")
                };
                let scratch = self.text_refiner_block_scratch_of(blk, index, b, t, plan);
                mlx_gen::ios_quantum::run(quanta::TEXT_REFINER_BLOCK, scratch, || {
                    Self::text_refiner_body_with(blk, self.compute, &y, &key_mask, plan)
                })
            },
            |y| Ok(eval([y])?),
        )?;
        mlx_gen::ios_quantum::run(quanta::TEXT_OUTPUT, self.text_output_scratch(b, t), || {
            self.text_output_body(&y)
        })
    }

    /// Run `step` over every block of `stack`: the resident stack in order, or — on a streamed
    /// backbone (rung 4) — window by window through the shared [`mlx_gen::block_residency`] driver:
    /// each window's blocks are read inside one priced [`quanta::BLOCK_READ`] quantum, run, the
    /// carry `materialize`d, and the window released before the next is read.
    fn run_stack<S>(
        &self,
        stack: Stack,
        exec: &Execution<'_>,
        init: S,
        mut step: impl FnMut(S, usize, BlockRef<'_>) -> Result<S>,
        materialize: impl Fn(&S) -> Result<()>,
    ) -> Result<S> {
        let Some(stream) = &self.stream else {
            let mut state = init;
            for i in 0..self.resident_len(stack) {
                state = step(state, i, self.resident_block(stack, i))?;
            }
            return Ok(state);
        };
        let n = stream.len(stack);
        if n == 0 {
            return Ok(init);
        }
        let window = exec
            .window
            .or(stream.default_window)
            .unwrap_or(n)
            .clamp(1, n);
        let windows = mlx_gen::block_residency::BlockPlan::new(n, window)
            .map_err(|e| Error::Msg(format!("iris stream: {e}")))?;
        let uncancellable = mlx_gen::CancelFlag::default();
        let cancel = exec.plan.cancel.unwrap_or(&uncancellable);
        mlx_gen::block_residency::run_windowed(
            &windows,
            cancel,
            init,
            || Weights::from_file(&stream.path),
            |mut state, view, range| {
                #[cfg(test)]
                WINDOW_HOOK.with(|hook| {
                    if let Some(hook) = hook.borrow_mut().as_mut() {
                        hook(stack, range.start);
                    }
                });
                let reads = &stream.inventory.reads(stack)[range.clone()];
                mlx_gen::ios_quantum::diagnostic_metadata(
                    crate::model::MODEL_ID,
                    "dit_window",
                    &[
                        (
                            "stack",
                            match stack {
                                Stack::Layer => 0,
                                Stack::Refiner => 1,
                                Stack::Trunk => 2,
                                Stack::Pixel => 3,
                            },
                        ),
                        ("first", range.start as u64),
                        ("count", range.len() as u64),
                    ],
                );
                mlx_gen::ios_quantum::before(
                    quanta::BLOCK_READ,
                    crate::stream::window_read_bytes(reads),
                )?;
                let mut built = Vec::with_capacity(range.len());
                for i in range.clone() {
                    built.push(stream.build(view, &self.cfg, stack, i)?);
                }
                let arrays: Vec<&Array> = built.iter().flat_map(Built::arrays).collect();
                mlx_gen::ios_quantum::finish(quanta::BLOCK_READ, &arrays)?;
                drop(arrays);
                for (i, block) in range.zip(&built) {
                    state = step(state, i, block.as_ref())?;
                }
                Ok(state)
            },
            materialize,
        )
    }

    fn resident_len(&self, stack: Stack) -> usize {
        match stack {
            Stack::Layer => self.y_embedder.layer_blocks.len(),
            Stack::Refiner => self.y_embedder.blocks.len(),
            Stack::Trunk => self.blocks.len(),
            Stack::Pixel => self.pixel_blocks.len(),
        }
    }

    fn resident_block(&self, stack: Stack, i: usize) -> BlockRef<'_> {
        match stack {
            Stack::Layer => BlockRef::Layer(&self.y_embedder.layer_blocks[i]),
            Stack::Refiner => BlockRef::Refiner(&self.y_embedder.blocks[i]),
            Stack::Trunk => BlockRef::Trunk(&self.blocks[i]),
            Stack::Pixel => BlockRef::Pixel(&self.pixel_blocks[i]),
        }
    }

    /// Body of text-adapter layer block `i` over `y` `[B·T, layers, Dt]` (one quantum).
    #[cfg(test)]
    pub(crate) fn text_layer_body(
        &self,
        i: usize,
        y: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        Self::text_layer_body_with(&self.y_embedder.layer_blocks[i], self.compute, y, plan)
    }

    fn text_layer_body_with(
        block: &LayerBlock,
        compute: Dtype,
        y: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        let h = block.attn.forward(
            &block.norm1.forward(y)?,
            None,
            None,
            compute,
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
    #[cfg(test)]
    pub(crate) fn text_refiner_body(
        &self,
        i: usize,
        y: &Array,
        key_mask: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        Self::text_refiner_body_with(&self.y_embedder.blocks[i], self.compute, y, key_mask, plan)
    }

    fn text_refiner_body_with(
        block: &AdapterBlock,
        compute: Dtype,
        y: &Array,
        key_mask: &Array,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<Array> {
        let h = block.attn.forward(
            &block.norm1.forward(y)?,
            None,
            Some(key_mask),
            compute,
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
        self.forward_exec(
            x,
            t,
            text,
            Execution::resident(mlx_gen::AttentionPlan::UNBOUNDED),
        )
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
        self.forward_impl(
            x,
            t,
            text,
            Some(hooks),
            Execution::resident(mlx_gen::AttentionPlan::UNBOUNDED),
        )
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
        self.forward_exec(x, t, text, Execution::resident(plan))
    }

    /// The inference forward under an explicit [`Execution`]: the attention plan, the streamed
    /// window (rung 4) and the per-row pixel stage (rung 3).
    pub fn forward_exec(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        exec: Execution<'_>,
    ) -> Result<Array> {
        self.forward_impl(x, t, text, None, exec)
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

    fn forward_impl(
        &self,
        x: &Array,
        t: &Array,
        text: &TextBatch,
        mut hooks: Option<&mut dyn ForwardHooks>,
        exec: Execution<'_>,
    ) -> Result<Array> {
        #[cfg(all(test, target_os = "macos"))]
        crate::anchor_tests::observe_generation_forward(x, text.states);
        let cfg = &self.cfg;
        let plan = exec.plan;
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
        if hooks.is_some() && self.stream.is_some() {
            return Err(Error::Unsupported(
                "iris: forward hooks run the resident block stacks; a streamed backbone (rung 4) \
                 builds each block per window"
                    .into(),
            ));
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

        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        // The previous step's guidance combine + solver arithmetic is lazy and materializes here.
        mlx_gen::ios_quantum::before(
            quanta::EMBED,
            self.embed_scratch(b, n_patches)
                .saturating_add(mlx_gen::ios_quantum::sampler_step_bytes(x.size() as u64)),
        )?;
        let Embedded {
            s,
            t_emb,
            core_img,
            core_txt,
        } = self.embed_body(x, t)?;
        let mut embedded = vec![&s, &t_emb, &core_img];
        embedded.extend(core_txt.as_ref());
        mlx_gen::ios_quantum::finish(quanta::EMBED, &embedded)?;

        let text_base = self.text_adapter_exec(text, &exec)?;
        let n_txt = text_base.shape()[1];
        // RoPE uses from_slice: it allocates during construction, not at the later block barrier.
        mlx_gen::ios_quantum::diagnostic_metadata(
            crate::model::MODEL_ID,
            "allocation_region",
            &[
                ("role", 1),
                ("rows", b as u64),
                ("positions", n_patches as u64),
            ],
        );
        mlx_gen::ios_quantum::before(
            quanta::TRUNK_POSITIONS,
            self.trunk_entry_allowance(b, n_patches, n_txt),
        )?;
        let y = self.text_positions(&text_base)?;
        let ctx = self.trunk_ctx(hp, wp, n_txt, plan);
        mlx_gen::ios_quantum::finish(
            quanta::TRUNK_POSITIONS,
            &[
                &y,
                &ctx.rope_img.cos,
                &ctx.rope_img.sin,
                &ctx.rope_txt.cos,
                &ctx.rope_txt.sin,
            ],
        )?;
        let (s, _y) = self.run_stack(
            Stack::Trunk,
            &exec,
            (s, y),
            |(s, y), i, block| {
                mlx_gen::ios_quantum::diagnostic_metadata(
                    crate::model::MODEL_ID,
                    "dit_block",
                    &[
                        ("stack", 2),
                        ("index", i as u64),
                        ("row", 0),
                        ("rows", b as u64),
                    ],
                );
                let BlockRef::Trunk(blk) = block else {
                    unreachable!("the trunk holds trunk blocks")
                };
                let scratch = self.trunk_block_scratch_of(blk, i, b, n_patches, n_txt, plan);
                mlx_gen::ios_quantum::before(quanta::TRUNK_BLOCK, scratch)?;
                let (s, y) = match hooks.as_deref_mut() {
                    Some(hooks) => {
                        hooks.trunk_block(self, i, &ctx, &s, &y, &core_img, core_txt.as_ref())?
                    }
                    None => {
                        Self::trunk_block_with(blk, &ctx, &s, &y, &core_img, core_txt.as_ref())?
                    }
                };
                mlx_gen::ios_quantum::finish(quanta::TRUNK_BLOCK, &[&s, &y])?;
                Ok((s, y))
            },
            |(s, y)| Ok(eval([s, y])?),
        )?;

        mlx_gen::ios_quantum::before(quanta::PIXEL_EMBED, self.pixel_embed_scratch(b, h, w))?;
        let (s_cond, pixels) = self.pixel_embed_body(x, &t_emb, &s)?;
        mlx_gen::ios_quantum::finish(quanta::PIXEL_EMBED, &[&s_cond, &pixels])?;
        drop(s);
        // Rung 3's pixel-row bound: the PiT stage one batch row at a time (each row evaluated
        // before the next is built, on every device), so its activation is one row's.
        let by_row = exec.pixel_rows && b > 1;
        let (row_batch, rows) = if by_row { (1, b) } else { (b, 1) };
        mlx_gen::ios_quantum::diagnostic_metadata(
            crate::model::MODEL_ID,
            "allocation_region",
            &[
                ("role", 2),
                ("rows", row_batch as u64),
                ("positions", n_patches as u64),
            ],
        );
        mlx_gen::ios_quantum::before(
            quanta::PIXEL_POSITIONS,
            mlx_gen::ios_quantum::rope_table_bytes(
                n_patches as u64,
                (cfg.pixel.attn_hidden_size / cfg.pixel.num_heads) as u64,
            ),
        )?;
        let pctx = self.pixel_ctx(row_batch, hp, wp, plan);
        mlx_gen::ios_quantum::finish(
            quanta::PIXEL_POSITIONS,
            &[&pctx.rope_pix.cos, &pctx.rope_pix.sin],
        )?;
        let conds = if by_row {
            s_cond.split(b, 0)?
        } else {
            vec![s_cond]
        };
        let parts = if by_row {
            pixels.split(b, 0)?
        } else {
            vec![pixels]
        };
        let parts = self.run_stack(
            Stack::Pixel,
            &exec,
            parts,
            |mut parts, i, block| {
                let BlockRef::Pixel(blk) = block else {
                    unreachable!("the PiT stack holds PiT blocks")
                };
                let scratch = self.pixel_block_scratch_of(blk, i, row_batch, n_patches, plan);
                for (row, (part, cond)) in parts.iter_mut().zip(&conds).enumerate() {
                    mlx_gen::ios_quantum::diagnostic_metadata(
                        crate::model::MODEL_ID,
                        "dit_block",
                        &[
                            ("stack", 3),
                            ("index", i as u64),
                            ("row", row as u64),
                            ("rows", row_batch as u64),
                        ],
                    );
                    let next = mlx_gen::ios_quantum::run(quanta::PIXEL_BLOCK, scratch, || {
                        match hooks.as_deref_mut() {
                            Some(hooks) => hooks.pixel_block(self, i, &pctx, part, cond),
                            None => Self::pit(
                                pctx.compute,
                                blk,
                                part,
                                cond,
                                &pctx.rope_pix,
                                pctx.batch,
                                pctx.n_patches,
                                pctx.plan,
                            ),
                        }
                    })?;
                    if by_row {
                        eval([&next])?;
                    }
                    *part = next;
                }
                Ok(parts)
            },
            |parts| Ok(eval(parts.iter())?),
        )?;
        if !by_row {
            mlx_gen::ios_quantum::diagnostic_metadata(
                crate::model::MODEL_ID,
                "dit_output",
                &[("row", 0), ("rows", b as u64)],
            );
            return mlx_gen::ios_quantum::run(
                quanta::OUTPUT,
                self.output_scratch(b, n_patches),
                || self.output_body(&parts[0], b, h, w),
            );
        }
        let mut outs = Vec::with_capacity(rows as usize);
        for (row, part) in parts.iter().enumerate() {
            mlx_gen::ios_quantum::diagnostic_metadata(
                crate::model::MODEL_ID,
                "dit_output",
                &[("row", row as u64), ("rows", 1)],
            );
            let out = mlx_gen::ios_quantum::run(
                quanta::OUTPUT,
                self.output_scratch(1, n_patches),
                || self.output_body(part, 1, h, w),
            )?;
            eval([&out])?;
            outs.push(out);
        }
        drop(parts);
        mlx_gen::ios_quantum::run(
            quanta::OUTPUT_ROWS,
            Self::output_rows_bytes(b, cfg.in_channels as i32, h, w),
            || {
                let refs: Vec<&Array> = outs.iter().collect();
                Ok(mlx_rs::ops::concatenate_axis(&refs, 0)?)
            },
        )
    }

    /// The per-row PiT stage's concatenation of its `b` output rows: the `[B, C, H, W]` f32 batch.
    pub(crate) fn output_rows_bytes(b: i32, channels: i32, h: i32, w: i32) -> u64 {
        let d = |v: i32| v.max(0) as u64;
        crate::retention::allocated(mlx_gen::ios_quantum::tensor_bytes(
            4,
            &[d(b), d(channels), d(h), d(w)],
        ))
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
