//! MMAudio's **DFN5B-CLIP ViT-H/14-384** visual + text conditioner (sc-13437), ported natively onto
//! the workspace's pinned candle revision from the **open_clip** checkpoint
//! `apple/DFN5B-CLIP-ViT-H-14-384`.
//!
//! ## What MMAudio actually consumes (verified against `features_utils.py`)
//!
//! MMAudio builds this model from the open_clip `apple/DFN5B-CLIP-ViT-H-14-384` checkpoint (upstream
//! `create_model_from_pretrained`) and conditions on two features:
//!
//! - **Visual** ([`DfnClipEncoder::encode_image`]): the *standard* open_clip
//!   `encode_image(x, normalize=True)` — RGB frames resized to **384×384**, normalized with the
//!   **OpenAI-CLIP** mean/std (`[0.481,0.458,0.408]` / `[0.269,0.261,0.276]`, **not** 0.5), through
//!   the ViT, CLS-pooled, `ln_post`, then projected `1280 → 1024` by `visual.proj`, then
//!   **L2-normalized** → one 1024-d vector per frame. MMAudio runs this per frame at 8 fps and stacks
//!   to `(B, T, 1024)`; the fps sampling is the downstream generator's concern — this module is the
//!   per-frame encoder.
//! - **Text** ([`DfnClipEncoder::encode_text`]): MMAudio *monkey-patches* `encode_text` (see
//!   `patch_clip`) to return the **full last hidden state** — `ln_final(transformer(tok + pos))`,
//!   **L2-normalized along the feature dim**, `(B, 77, 1024)` per-token. Crucially the patch **drops
//!   the `text_projection`** and the EOT-pooling of stock open_clip: it is the raw 1024-wide
//!   per-token sequence. Text conditioning is optional at inference.
//!
//! ## Architecture (`open_clip_config.json`, ViT-H-14 quickgelu)
//!
//! - **Vision:** `image 378` native (MMAudio feeds 384; a `k=stride=14` conv yields a `27×27=729`
//!   patch grid at **both** 378 and 384, so `positional_embedding` `[730, 1280]` is compatible),
//!   `patch 14`, `width 1280`, `depth 32`, `heads 16` (`head_dim 80`), `mlp 5120`. CLS token + a
//!   single learned `positional_embedding`; pre-norm `ln_pre`; 32 `ResidualAttentionBlock`s;
//!   `ln_post`; `proj [1280, 1024]`.
//! - **Text:** `vocab 49408`, `width 1024`, `depth 24`, `heads 16` (`head_dim 64`), `mlp 4096`,
//!   `context 77`, **causal** attention mask; `token_embedding` + learned `positional_embedding`;
//!   `ln_final`. (`text_projection [1024,1024]` exists in the checkpoint but the MMAudio patch does
//!   not use it.)
//! - **Activation is `quickgelu`** everywhere (`x · sigmoid(1.702 x)`) — DFN5B is the `-quickgelu`
//!   variant (the tokenizer is `ViT-H-14-378-quickgelu`), **not** the erf-GELU MetaCLIP/OpenAI-HF
//!   variant. `LayerNorm` eps is `1e-5`. Attention is `nn.MultiheadAttention`: a single fused
//!   `in_proj [3·width, width]`, scale `1/√head_dim`.
//!
//! ## Weight naming
//!
//! Read **directly** from the open_clip state dict (`open_clip_pytorch_model.bin`): `visual.conv1`,
//! `visual.class_embedding`, `visual.positional_embedding`, `visual.ln_pre/ln_post`, `visual.proj`,
//! `visual.transformer.resblocks.{i}.{ln_1,attn.in_proj_weight,attn.in_proj_bias,attn.out_proj,ln_2,
//! mlp.c_fc,mlp.c_proj}`, and the text tower at the root (`token_embedding`, `positional_embedding`,
//! `transformer.resblocks.{i}.*`, `ln_final`). No open_clip→HF remap is performed — matching the
//! checkpoint's own names is both faithful to what MMAudio loads and avoids a whole class of
//! q/k/v-split remap bugs.
//!
//! ## Faithfulness
//!
//! Numerically parity-checked against the `open_clip` reference on a fixed synthetic 384² image and
//! two text strings: pooled image feature cosine and per-token text feature cosine both `> 0.999`
//! (see the crate's `tests/clip_conformance.rs` doc and the PR for sc-13437). Preprocessing uses
//! `CatmullRom` as a bicubic approximation (the `image` crate has no torchvision bicubic), so
//! features from *non-384* source frames can differ slightly from the reference resize — the
//! encoder math itself is exact.

use candle_audio::candle_core::{DType, Device, IndexOp, Result as CResult, Tensor, D};
use candle_nn::{
    conv2d_no_bias, layer_norm, ops::softmax_last_dim, Conv2d, Conv2dConfig, Embedding, LayerNorm,
    Linear, Module, VarBuilder,
};

// ---- config (open_clip_config.json, ViT-H-14 quickgelu) --------------------------------------

/// Joint CLIP embedding dim — the width of the visual feature MMAudio consumes.
pub const EMBED_DIM: usize = 1024;
/// Input resolution MMAudio feeds the visual tower (px). The native config is 378, but a
/// `k=stride=14` conv produces the same `27×27` grid at 384, so the pinned `positional_embedding`
/// applies unchanged (see module doc).
pub const IMAGE_SIZE: usize = 384;
/// Spatial patch size (px).
pub const PATCH_SIZE: usize = 14;
/// RGB channels.
pub const IN_CHANS: usize = 3;
/// Patches per spatial axis: `floor((384 - 14) / 14) + 1 = 27` (the conv drops the 6px remainder).
pub const GRID: usize = (IMAGE_SIZE - PATCH_SIZE) / PATCH_SIZE + 1;
/// Patch tokens per frame (`27² = 729`).
pub const NUM_PATCHES: usize = GRID * GRID;
/// Position count including the prepended CLS token (`730`).
pub const NUM_POSITIONS: usize = NUM_PATCHES + 1;

/// Vision transformer hidden width.
pub const VISION_WIDTH: usize = 1280;
/// Vision transformer depth.
pub const VISION_LAYERS: usize = 32;
/// Vision attention heads.
pub const VISION_HEADS: usize = 16;
/// Vision MLP hidden width (`width · 4`).
pub const VISION_MLP: usize = VISION_WIDTH * 4;

/// Text transformer hidden width (= [`EMBED_DIM`], but conceptually distinct).
pub const TEXT_WIDTH: usize = 1024;
/// Text transformer depth.
pub const TEXT_LAYERS: usize = 24;
/// Text attention heads.
pub const TEXT_HEADS: usize = 16;
/// Text MLP hidden width (`width · 4`).
pub const TEXT_MLP: usize = TEXT_WIDTH * 4;
/// Token vocabulary size.
pub const VOCAB_SIZE: usize = 49408;
/// Text context length (fixed 77 tokens).
pub const CONTEXT_LENGTH: usize = 77;

/// CLIP start-of-text / end-of-text token ids and the pad id (open_clip pads with `0`).
pub const SOT_TOKEN: u32 = 49406;
pub const EOT_TOKEN: u32 = 49407;
pub const PAD_TOKEN: u32 = 0;

/// LayerNorm epsilon (open_clip's `LayerNorm` default).
pub const LN_EPS: f64 = 1e-5;

/// OpenAI-CLIP per-channel normalization mean (`preprocess_cfg.mean`) — **not** 0.5.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// OpenAI-CLIP per-channel normalization std (`preprocess_cfg.std`).
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

// ---- primitives ------------------------------------------------------------------------------

/// `quickgelu`: `x · sigmoid(1.702 · x)` (open_clip `-quickgelu` activation).
fn quick_gelu(xs: &Tensor) -> CResult<Tensor> {
    xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?
}

/// open_clip `nn.MultiheadAttention`: one fused `in_proj [3·d, d]`, per-head scaling `1/√head_dim`,
/// optional additive attention mask (used causal by the text tower).
struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiheadAttention {
    fn load(vb: VarBuilder, width: usize, num_heads: usize) -> CResult<Self> {
        let head_dim = width / num_heads;
        let in_w = vb.get((3 * width, width), "in_proj_weight")?;
        let in_b = vb.get(3 * width, "in_proj_bias")?;
        let in_proj = Linear::new(in_w, Some(in_b));
        let out_w = vb.get((width, width), "out_proj.weight")?;
        let out_b = vb.get(width, "out_proj.bias")?;
        let out_proj = Linear::new(out_w, Some(out_b));
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Split each `(B, L, H, hd)` head layout out of the fused projection and put heads first.
    fn heads(&self, x: &Tensor, b: usize, l: usize) -> CResult<Tensor> {
        x.reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CResult<Tensor> {
        let (b, l, width) = x.dims3()?;
        let qkv = self.in_proj.forward(x)?; // (B, L, 3·width)
        let q = qkv.narrow(D::Minus1, 0, width)?;
        let k = qkv.narrow(D::Minus1, width, width)?;
        let v = qkv.narrow(D::Minus1, 2 * width, width)?;
        let q = (self.heads(&q, b, l)? * self.scale)?; // (B, H, L, hd)
        let k = self.heads(&k, b, l)?;
        let v = self.heads(&v, b, l)?;
        let mut attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // (B, H, L, L)
        if let Some(m) = mask {
            attn = attn.broadcast_add(m)?; // additive (L, L) causal mask
        }
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // (B, H, L, hd)
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, l, width))?;
        self.out_proj.forward(&out)
    }
}

/// open_clip `ResidualAttentionBlock`: pre-norm attention then pre-norm quickgelu MLP, each residual.
struct ResidualAttentionBlock {
    ln_1: LayerNorm,
    attn: MultiheadAttention,
    ln_2: LayerNorm,
    c_fc: Linear,
    c_proj: Linear,
}

impl ResidualAttentionBlock {
    fn load(vb: VarBuilder, width: usize, heads: usize, mlp: usize) -> CResult<Self> {
        let ln_1 = layer_norm(width, LN_EPS, vb.pp("ln_1"))?;
        let attn = MultiheadAttention::load(vb.pp("attn"), width, heads)?;
        let ln_2 = layer_norm(width, LN_EPS, vb.pp("ln_2"))?;
        let c_fc = linear(width, mlp, vb.pp("mlp").pp("c_fc"))?;
        let c_proj = linear(mlp, width, vb.pp("mlp").pp("c_proj"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            c_fc,
            c_proj,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CResult<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?, mask)?)?;
        let h = self
            .c_proj
            .forward(&quick_gelu(&self.c_fc.forward(&self.ln_2.forward(&x)?)?)?)?;
        x + h
    }
}

/// A `candle_nn::Linear` with explicit `weight`/`bias` tensors (open_clip stores both).
fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CResult<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = vb.get(out_dim, "bias")?;
    Ok(Linear::new(w, Some(b)))
}

// ---- towers ----------------------------------------------------------------------------------

/// The ViT-H/14 visual tower ending at the projected, poolable representation.
struct VisionTower {
    conv1: Conv2d,
    class_embedding: Tensor,      // (width,)
    positional_embedding: Tensor, // (730, width)
    ln_pre: LayerNorm,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    proj: Tensor, // (width, embed_dim)
}

impl VisionTower {
    fn load(vb: VarBuilder) -> CResult<Self> {
        let cfg = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(IN_CHANS, VISION_WIDTH, PATCH_SIZE, cfg, vb.pp("conv1"))?;
        let class_embedding = vb.get(VISION_WIDTH, "class_embedding")?;
        let positional_embedding = vb.get((NUM_POSITIONS, VISION_WIDTH), "positional_embedding")?;
        let ln_pre = layer_norm(VISION_WIDTH, LN_EPS, vb.pp("ln_pre"))?;
        let mut blocks = Vec::with_capacity(VISION_LAYERS);
        let rvb = vb.pp("transformer").pp("resblocks");
        for i in 0..VISION_LAYERS {
            blocks.push(ResidualAttentionBlock::load(
                rvb.pp(i),
                VISION_WIDTH,
                VISION_HEADS,
                VISION_MLP,
            )?);
        }
        let ln_post = layer_norm(VISION_WIDTH, LN_EPS, vb.pp("ln_post"))?;
        let proj = vb.get((VISION_WIDTH, EMBED_DIM), "proj")?;
        Ok(Self {
            conv1,
            class_embedding,
            positional_embedding,
            ln_pre,
            blocks,
            ln_post,
            proj,
        })
    }

    /// `_embeds`: conv patchify → prepend CLS → add positional → `ln_pre`, then the 32 blocks and
    /// `ln_post`. Returns the post-`ln_post` sequence `(B, 730, width)`.
    fn hidden_states(&self, pixels: &Tensor) -> CResult<Tensor> {
        let b = pixels.dim(0)?;
        let x = self.conv1.forward(pixels)?; // (B, width, 27, 27)
        let x = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // (B, 729, width)
        let cls = self
            .class_embedding
            .reshape((1, 1, VISION_WIDTH))?
            .broadcast_as((b, 1, VISION_WIDTH))?;
        let x = Tensor::cat(&[&cls, &x], 1)?; // (B, 730, width)
        let x = x.broadcast_add(&self.positional_embedding.unsqueeze(0)?)?;
        let mut x = self.ln_pre.forward(&x)?;
        for blk in &self.blocks {
            x = blk.forward(&x, None)?;
        }
        self.ln_post.forward(&x)
    }
}

/// The 77-token CLIP text tower ending at `ln_final` (the MMAudio-patched output; no projection).
struct TextTower {
    token_embedding: Embedding,
    positional_embedding: Tensor, // (77, width)
    blocks: Vec<ResidualAttentionBlock>,
    ln_final: LayerNorm,
}

impl TextTower {
    fn load(vb: VarBuilder) -> CResult<Self> {
        let te = vb.get((VOCAB_SIZE, TEXT_WIDTH), "token_embedding.weight")?;
        let token_embedding = Embedding::new(te, TEXT_WIDTH);
        let positional_embedding = vb.get((CONTEXT_LENGTH, TEXT_WIDTH), "positional_embedding")?;
        let mut blocks = Vec::with_capacity(TEXT_LAYERS);
        let rvb = vb.pp("transformer").pp("resblocks");
        for i in 0..TEXT_LAYERS {
            blocks.push(ResidualAttentionBlock::load(
                rvb.pp(i),
                TEXT_WIDTH,
                TEXT_HEADS,
                TEXT_MLP,
            )?);
        }
        let ln_final = layer_norm(TEXT_WIDTH, LN_EPS, vb.pp("ln_final"))?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_final,
        })
    }
}

/// Additive causal mask `(L, L)`: `0` on/below the diagonal, `f32::MIN` above it.
fn causal_mask(l: usize, device: &Device) -> CResult<Tensor> {
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in (i + 1)..l {
            data[i * l + j] = f32::MIN;
        }
    }
    Tensor::from_vec(data, (l, l), device)
}

// ---- assembled encoder -----------------------------------------------------------------------

/// The assembled DFN5B-CLIP ViT-H/14 encoder (both towers), weights resolved onto `device`.
pub struct DfnClipEncoder {
    visual: VisionTower,
    text: TextTower,
    device: Device,
}

impl DfnClipEncoder {
    /// Load both towers from a `VarBuilder` rooted at the open_clip state dict's top level.
    pub fn load(vb: VarBuilder, device: Device) -> CResult<Self> {
        let visual = VisionTower::load(vb.pp("visual"))?;
        let text = TextTower::load(vb.clone())?;
        Ok(Self {
            visual,
            text,
            device,
        })
    }

    /// The compute device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// **The visual feature MMAudio consumes.** `(B, 3, 384, 384)` → CLS-pool → `visual.proj`
    /// (`1280 → 1024`) → **L2-normalize** → `(B, 1024)`. Equivalent to open_clip
    /// `encode_image(x, normalize=True)`.
    pub fn encode_image(&self, pixels: &Tensor) -> CResult<Tensor> {
        let hs = self.visual.hidden_states(pixels)?; // (B, 730, width)
        let pooled = cls_pool(&hs)?; // CLS token, (B, width) — contiguous for the visual.proj matmul
        let projected = pooled.matmul(&self.visual.proj)?; // (B, 1024)
        l2_normalize(&projected)
    }

    /// The full post-`ln_post` patch-token sequence `(B, 729, 1280)` (auxiliary — **not** consumed
    /// by MMAudio's `features_utils`, exposed for completeness / future joint-attention use). These
    /// are un-projected 1280-wide tokens, matching open_clip's `output_tokens` return.
    pub fn encode_image_tokens(&self, pixels: &Tensor) -> CResult<Tensor> {
        let hs = self.visual.hidden_states(pixels)?; // (B, 730, width)
        hs.i((.., 1.., ..))?.contiguous()
    }

    /// **The text feature MMAudio consumes.** `(B, 77)` token ids → `token+positional` embeddings →
    /// 24 causal blocks → `ln_final` → **L2-normalize along the feature dim** → `(B, 77, 1024)`
    /// per-token. Matches MMAudio's patched `encode_text(tokens, normalize=True)` (no
    /// `text_projection`, no EOT pooling). Ids must be `CONTEXT_LENGTH`-wide (see [`tokenize`]).
    pub fn encode_text(&self, token_ids: &Tensor) -> CResult<Tensor> {
        let l = token_ids.dim(D::Minus1)?;
        let x = self.text.token_embedding.forward(token_ids)?; // (B, L, width)
        let pos = self
            .text
            .positional_embedding
            .narrow(0, 0, l)?
            .unsqueeze(0)?;
        let mut x = x.broadcast_add(&pos)?;
        let mask = causal_mask(l, x.device())?;
        for blk in &self.text.blocks {
            x = blk.forward(&x, Some(&mask))?;
        }
        let x = self.text.ln_final.forward(&x)?;
        l2_normalize(&x)
    }
}

/// CLS-pool the ViT hidden states `(B, L, width)` → the CLS token `(B, width)`.
///
/// The `.contiguous()` is load-bearing: the CLS-token slice `hs[:, 0, :]` is a strided view (its row
/// stride is the full `L·width` token span), and candle's CUDA `matmul` rejects a non-contiguous
/// **lhs** — which is exactly how [`DfnClipEncoder::encode_image`] then uses this tensor against the
/// `visual.proj` weight (sc-13888). It is a no-op on already-contiguous tensors and bit-identical
/// across CPU/Metal/CUDA. `is_contiguous()` is a device-independent layout property, so the returned
/// tensor's contiguity is regression-tested on the default CPU lane (see `tests::cls_pool_*`).
fn cls_pool(hs: &Tensor) -> CResult<Tensor> {
    hs.i((.., 0, ..))?.contiguous()
}

/// L2-normalize along the last dim (open_clip `F.normalize(x, dim=-1)`).
fn l2_normalize(x: &Tensor) -> CResult<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Wrap raw BPE token ids into an open_clip 77-wide row: `[SOT, ...ids..., EOT]` truncated so EOT is
/// always present, then padded with [`PAD_TOKEN`] (`0`) to [`CONTEXT_LENGTH`]. This mirrors
/// open_clip's `SimpleTokenizer.__call__`. The BPE step itself (lowercasing + merges) is the
/// caller's — pass the already-BPE'd content ids (no SOT/EOT).
pub fn wrap_tokens(content_ids: &[u32]) -> Vec<u32> {
    let mut row = vec![PAD_TOKEN; CONTEXT_LENGTH];
    row[0] = SOT_TOKEN;
    // Reserve the last slot for EOT: content may fill positions 1..=CONTEXT_LENGTH-2.
    let max_content = CONTEXT_LENGTH - 2;
    let n = content_ids.len().min(max_content);
    row[1..1 + n].copy_from_slice(&content_ids[..n]);
    row[1 + n] = EOT_TOKEN;
    row
}

/// Build a `(B, 77)` `u32` token-id tensor from pre-wrapped 77-wide rows (see [`wrap_tokens`]).
pub fn tokenize(rows: &[Vec<u32>], device: &Device) -> CResult<Tensor> {
    let b = rows.len();
    let mut flat = Vec::with_capacity(b * CONTEXT_LENGTH);
    for r in rows {
        assert_eq!(r.len(), CONTEXT_LENGTH, "each row must be 77-wide");
        flat.extend_from_slice(r);
    }
    Tensor::from_vec(flat, (b, CONTEXT_LENGTH), device)
}

// ---- string -> BPE token ids (open_clip `SimpleTokenizer`) ------------------------------------
//
// Reproduces open_clip's `SimpleTokenizer` (the generic CLIP BPE, `bpe_simple_vocab_16e6`) as used
// for the DFN5B / ViT-H-14 model — the string->id front end for [`DfnClipEncoder::encode_text`].
// Faithfulness is byte-match-verified against `open_clip.get_tokenizer('ViT-H-14-378-quickgelu')`
// (see `tests/clip_conformance.rs`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// The pinned CLIP BPE merge table, vendored verbatim from the pinned DFN5B repo's `merges.txt`
/// (`apple/DFN5B-CLIP-ViT-H-14-384` @ [`CLIP_HUB_REVISION`]). This is the same merge list as
/// open_clip's bundled `bpe_simple_vocab_16e6.txt.gz`; the encoder rebuilt from it reproduces the
/// repo's `vocab.json` exactly (all 49408 ids) — asserted in tests.
const CLIP_BPE_MERGES: &str = include_str!("dfn5b_clip_merges.txt");

/// Merges open_clip keeps: `merges[1 : 49152-256-2+1]` → `49152 - 256 - 2 = 48894`.
const NUM_BPE_MERGES: usize = 48894;

/// open_clip `bytes_to_unicode()`: a reversible byte↔printable-unicode map. Built in open_clip's
/// exact *insertion order* (printable byte ranges first, then the remaining bytes mapped to `256+n`)
/// — that order defines the base-vocab ids, so it must be preserved, not sorted.
fn byte_to_unicode() -> Vec<(u8, char)> {
    let mut bytes: Vec<u32> = Vec::with_capacity(256);
    for range in [0x21u32..=0x7e, 0xa1..=0xac, 0xae..=0xff] {
        bytes.extend(range);
    }
    let mut codes: Vec<u32> = bytes.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            codes.push(256 + n);
            n += 1;
        }
    }
    bytes
        .into_iter()
        .zip(codes)
        .map(|(b, c)| (b as u8, char::from_u32(c).expect("bytes_to_unicode scalar")))
        .collect()
}

/// The assembled CLIP byte-level BPE tokenizer: byte encoder, vocab, merge ranks, and the
/// pre-tokenization pattern. Built once (via [`clip_bpe`]) from the vendored merges.
struct ClipBpe {
    /// byte value → its printable-unicode char (open_clip `byte_encoder`).
    byte_encoder: [char; 256],
    /// vocab token string → id (open_clip `encoder`).
    encoder: HashMap<String, u32>,
    /// ordered merge pair → rank (lower = merged earlier; open_clip `bpe_ranks`).
    ranks: HashMap<(String, String), usize>,
    /// open_clip `self.pat` pre-tokenization regex (special tokens, contractions, letters, single
    /// digit, other non-space run). `(?i)` mirrors open_clip's `re.IGNORECASE`.
    pat: regex::Regex,
}

/// The process-wide CLIP tokenizer, built on first use from the vendored merge table.
fn clip_bpe() -> &'static ClipBpe {
    static BPE: OnceLock<ClipBpe> = OnceLock::new();
    BPE.get_or_init(ClipBpe::build)
}

impl ClipBpe {
    fn build() -> Self {
        let b2u = byte_to_unicode();
        let mut byte_encoder = ['\0'; 256];
        for &(b, c) in &b2u {
            byte_encoder[b as usize] = c;
        }

        // merges[1 : 49152-256-2+1] — skip the `#version` header line, take 48894 pairs.
        let merges: Vec<(String, String)> = CLIP_BPE_MERGES
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .take(NUM_BPE_MERGES)
            .map(|line| {
                let mut it = line.split_whitespace();
                let first = it.next().expect("merge first symbol").to_string();
                let second = it.next().expect("merge second symbol").to_string();
                (first, second)
            })
            .collect();
        assert_eq!(
            merges.len(),
            NUM_BPE_MERGES,
            "vendored CLIP merge table is truncated"
        );

        // vocab = base byte chars ++ base+'</w>' ++ joined merges ++ [SOT, EOT] (open_clip order).
        let mut vocab: Vec<String> = Vec::with_capacity(2 * b2u.len() + merges.len() + 2);
        for &(_, c) in &b2u {
            vocab.push(c.to_string());
        }
        for &(_, c) in &b2u {
            vocab.push(format!("{c}</w>"));
        }
        for (a, b) in &merges {
            vocab.push(format!("{a}{b}"));
        }
        vocab.push("<start_of_text>".to_string());
        vocab.push("<end_of_text>".to_string());
        let encoder: HashMap<String, u32> = vocab
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();

        let ranks: HashMap<(String, String), usize> = merges
            .into_iter()
            .enumerate()
            .map(|(i, pair)| (pair, i))
            .collect();

        let pat = regex::Regex::new(
            r"(?i)<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+",
        )
        .expect("static CLIP pre-tokenization pattern compiles");

        Self {
            byte_encoder,
            encoder,
            ranks,
            pat,
        }
    }

    /// open_clip `SimpleTokenizer.bpe`: greedily merge the lowest-ranked adjacent pair until no
    /// ranked pair remains. `token` is the byte-encoded piece; returns the final symbol sequence.
    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        // word = [c0, c1, ..., c_{n-2}, (c_{n-1} + "</w>")]
        let last = chars.len() - 1;
        let mut word: Vec<String> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i == last {
                    format!("{c}</w>")
                } else {
                    c.to_string()
                }
            })
            .collect();

        while word.len() > 1 {
            // lowest-rank adjacent pair (ranks are unique, so no tie-break is needed).
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..word.len() - 1 {
                if let Some(&r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, idx)) = best else { break };
            let (first, second) = (word[idx].clone(), word[idx + 1].clone());

            // merge every non-overlapping occurrence of (first, second).
            let mut merged: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if word[i] == first && i + 1 < word.len() && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    /// open_clip `SimpleTokenizer.encode`: clean → pre-tokenize → byte-level encode each piece →
    /// BPE → map symbols to ids. Returns the raw content ids (no SOT/EOT).
    fn encode(&self, text: &str) -> Vec<u32> {
        let cleaned = clip_clean(text);
        let mut ids: Vec<u32> = Vec::new();
        for m in self.pat.find_iter(&cleaned) {
            // byte-level: UTF-8 bytes of the piece → byte_encoder chars.
            let mut piece = String::new();
            for &b in m.as_str().as_bytes() {
                piece.push(self.byte_encoder[b as usize]);
            }
            for sym in self.bpe(&piece) {
                if let Some(&id) = self.encoder.get(&sym) {
                    ids.push(id);
                }
            }
        }
        ids
    }
}

/// open_clip `_clean_lower` = `whitespace_clean(basic_clean(text)).lower()`.
///
/// `basic_clean` is `ftfy.fix_text` + `html.unescape` (applied twice) + strip; `whitespace_clean`
/// collapses every run of unicode whitespace to a single space and trims. This reproduces the
/// html-unescape (numeric + the core named entities) + whitespace-collapse + lowercase steps. It does
/// **not** run ftfy's mojibake repair: ftfy is a no-op on well-formed UTF-8 (verified: the final
/// cleaned text is byte-identical with and without ftfy across the conformance prompt set), so for
/// the already-decoded text a request carries the output matches open_clip byte-for-byte. The full
/// HTML5 named-entity set is likewise not reproduced (only `amp/lt/gt/quot/apos/nbsp` + numeric).
fn clip_clean(text: &str) -> String {
    // basic_clean: html.unescape twice (ftfy is identity on well-formed text — see doc above).
    let unescaped = html_unescape(&html_unescape(text));
    // whitespace_clean: `" ".join(text.split())` — collapse unicode-whitespace runs, trim.
    let collapsed = unescaped.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.to_lowercase()
}

/// Minimal, faithful `html.unescape`: decodes `&#NN;` / `&#xHH;` numeric references and the core
/// named entities. Non-entity `&` and unknown names pass through unchanged (as Python's does for
/// text with no matching entity). open_clip applies this twice; [`clip_clean`] does the same.
fn html_unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '&' {
            // look for a closing ';' within a bounded window
            let end = (i + 12).min(chars.len());
            if let Some(semi) = (i + 1..end).find(|&j| chars[j] == ';') {
                let name: String = chars[i + 1..semi].iter().collect();
                if let Some(rep) = decode_html_entity(&name) {
                    out.push_str(&rep);
                    i = semi + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Decode a single HTML entity body (the text between `&` and `;`). Returns `None` for unknown
/// names so the original text is preserved verbatim.
fn decode_html_entity(name: &str) -> Option<String> {
    if let Some(num) = name.strip_prefix('#') {
        let cp = match num.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => num.parse::<u32>().ok()?,
        };
        return char::from_u32(cp).map(|c| c.to_string());
    }
    let c = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{a0}',
        _ => return None,
    };
    Some(c.to_string())
}

/// Tokenize a raw prompt string into an open_clip 77-wide CLIP token-id row, reproducing open_clip's
/// `SimpleTokenizer` for the DFN5B / ViT-H-14 model: lowercase + whitespace/html cleanup, byte-level
/// BPE with the pinned merge ranks, wrapped `[SOT …ids… EOT]`, truncated so EOT survives, padded
/// with [`PAD_TOKEN`] (`0`) to [`CONTEXT_LENGTH`].
///
/// This is the string→id front end for [`DfnClipEncoder::encode_text`]: the shipping MMAudio
/// assembly (sc-12843) calls it to turn a request's text prompt into the `(1, 77)` id row the text
/// tower consumes (build the tensor with [`tokenize`] from `&[tokenize_str(text).to_vec()]`).
///
/// Byte-identical to `open_clip.get_tokenizer('ViT-H-14-378-quickgelu')(text)` — validated on a
/// varied prompt set (empty, punctuation, digits, contractions, accents, CJK, emoji, html entities,
/// >77-token truncation) in `tests/clip_conformance.rs`.
pub fn tokenize_str(text: &str) -> [u32; CONTEXT_LENGTH] {
    let ids = clip_bpe().encode(text);
    let row = wrap_tokens(&ids);
    let mut out = [PAD_TOKEN; CONTEXT_LENGTH];
    out.copy_from_slice(&row);
    out
}

// ---- preprocessing ---------------------------------------------------------------------------

/// Preprocess RGB frames into the visual tower's input tensor `(N, 3, 384, 384)`.
///
/// Per frame: resize to `384×384` (squash — non-aspect-preserving, matching the model's
/// `resize_mode: squash`), scale to `[0,1]`, normalize with the OpenAI-CLIP [`CLIP_MEAN`]/[`CLIP_STD`].
/// Uses `CatmullRom` as a bicubic approximation (see module doc): faithful for already-384 input,
/// slightly approximate otherwise.
pub fn frames_to_clip_input(frames: &[image::RgbImage], device: &Device) -> CResult<Tensor> {
    use image::imageops::FilterType;
    let sz = IMAGE_SIZE as u32;
    let hw = IMAGE_SIZE * IMAGE_SIZE;
    let mut buf: Vec<f32> = Vec::with_capacity(frames.len() * IN_CHANS * hw);
    for frame in frames {
        let resized = if frame.dimensions() == (sz, sz) {
            frame.clone()
        } else {
            image::imageops::resize(frame, sz, sz, FilterType::CatmullRom)
        };
        // channel-major (C, H, W)
        for c in 0..IN_CHANS {
            for px in resized.pixels() {
                let v = (px[c] as f32) / 255.0;
                buf.push((v - CLIP_MEAN[c]) / CLIP_STD[c]);
            }
        }
    }
    Tensor::from_vec(
        buf,
        (frames.len(), IN_CHANS, IMAGE_SIZE, IMAGE_SIZE),
        device,
    )
}

// ---- pinned checkpoint + weight license + load entry points ----------------------------------

use std::path::{Path, PathBuf};

use candle_audio::gen_core::WeightsSource;
use candle_audio::{AudioError, Result};

/// Stable identity of this encoder (weight-license key). Not a shipping provider id — this crate
/// registers nothing this slice.
pub const MODEL_ID: &str = "dfn5b_clip_vit_h14_384";

/// Hub pin: Apple's DFN5B-CLIP ViT-H/14-384 open_clip repo, immutable commit SHA (F-029).
pub const CLIP_HUB_REPO: &str = "apple/DFN5B-CLIP-ViT-H-14-384";
pub const CLIP_HUB_REVISION: &str = "01b771ed0d1395ca5ffdd279897d665ebe00dfd2";
/// The open_clip-format state dict (fused `in_proj`, `visual.*`/root text tower) MMAudio loads via
/// `create_model_from_pretrained`. (The repo also ships an HF-`CLIPModel` `pytorch_model.bin`;
/// we deliberately load the open_clip one to match what MMAudio uses.)
pub const CLIP_WEIGHTS_PATH: &str = "open_clip_pytorch_model.bin";

/// The license of the pinned DFN5B-CLIP weights (sc-13332 framework), surfaced for SceneWorks'
/// end-product licenses page.
///
/// **Apple Machine Learning Research Model License Agreement** — verified against the repo `LICENSE`
/// file. It is **research/non-commercial only**: use is limited to scientific research and academic
/// development, and "Research Purposes" explicitly excludes "any commercial exploitation, product
/// development or use in any commercial product or service." SceneWorks is a non-commercial product
/// (per the sc-13332 discipline a `commercial_use = false` checkpoint is admissible **iff** its
/// terms are recorded), so the restriction is surfaced here rather than buried. A legal read is
/// warranted before any use that could be construed as commercial.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "LicenseRef-Apple-MLR",
        name: "Apple Machine Learning Research Model License Agreement",
        source_url: "https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384",
        attribution: Some(
            "DFN5B-CLIP ViT-H/14-384 © Apple Inc. — Apple Machine Learning Research Model License \
             Agreement (research / non-commercial use only)",
        ),
        commercial_use: false,
        restriction: Some(
            "Apple ML Research Model License: research and academic use only; 'Research Purposes' \
             excludes any commercial exploitation, product development, or use in a commercial \
             product or service. Trained on the DFN-5B image-text dataset. A legal read is \
             warranted before any commercial use.",
        ),
    };

/// This encoder's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation once a
/// shipping MMAudio generator registers it.
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: None,
        license: WEIGHT_LICENSE,
    };

/// Load the encoder from an `open_clip_pytorch_model.bin` file path.
pub fn load_from_pth(weights: &Path, device: &Device) -> Result<DfnClipEncoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID}: weights file {} not found (pass {CLIP_WEIGHTS_PATH} in via the LoadSpec)",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth(weights, DType::F32, device).map_err(AudioError::from)?;
    DfnClipEncoder::load(vb, device.clone()).map_err(AudioError::from)
}

/// Load from a [`WeightsSource`] (a `File` path to the `.bin`, or a `Dir` containing it).
pub fn load(source: &WeightsSource, device: &Device) -> Result<DfnClipEncoder> {
    let path: PathBuf = match source {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => d.join(CLIP_WEIGHTS_PATH),
    };
    load_from_pth(&path, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_license_is_well_formed_non_commercial() {
        assert!(WEIGHT_LICENSE.is_well_formed());
        assert!(
            !WEIGHT_LICENSE.is_permissive(),
            "Apple MLR is non-commercial"
        );
        assert!(WEIGHT_LICENSE.restriction.is_some());
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, MODEL_ID);
    }

    #[test]
    fn hub_revision_is_a_full_commit_sha() {
        assert_eq!(CLIP_HUB_REVISION.len(), 40);
        assert!(CLIP_HUB_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn missing_weights_file_errors_clearly() {
        let dev = Device::Cpu;
        let err = match load_from_pth(Path::new("/nonexistent/open_clip_pytorch_model.bin"), &dev) {
            Ok(_) => panic!("loading a nonexistent path must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("not found"));
    }

    /// sc-13932 (guards the sc-13888 fix): [`cls_pool`] must return a **contiguous** tensor — it is
    /// the lhs of the `visual.proj` matmul in [`DfnClipEncoder::encode_image`], and candle's CUDA
    /// `matmul` rejects a non-contiguous lhs. `is_contiguous()` is a device-independent layout
    /// property, so this runs on the default CPU lane (no ViT forward / weights needed) and flips
    /// **RED** if the `.contiguous()` is dropped from `cls_pool`.
    #[test]
    fn cls_pool_is_contiguous_for_proj_matmul() {
        let dev = Device::Cpu;
        // A contiguous `(B, L, width)` parent; its CLS slice `hs[:, 0, :]` is therefore strided
        // (row stride = L*width != width), which is precisely the operand candle's CUDA matmul rejects.
        let hs = Tensor::zeros((2, NUM_POSITIONS, VISION_WIDTH), DType::F32, &dev).unwrap();
        assert!(
            !hs.i((.., 0, ..)).unwrap().is_contiguous(),
            "precondition: the raw CLS slice must be strided (else the test proves nothing)"
        );
        let pooled = cls_pool(&hs).unwrap();
        assert_eq!(
            pooled.dims(),
            &[2, VISION_WIDTH],
            "CLS-pooled to (B, width)"
        );
        assert!(
            pooled.is_contiguous(),
            "sc-13888/sc-13932: cls_pool must return a contiguous tensor (the visual.proj matmul lhs)"
        );
    }

    #[test]
    fn grid_and_positions_match_reference() {
        assert_eq!(GRID, 27, "27x27 patch grid at 384 with k=stride=14");
        assert_eq!(NUM_PATCHES, 729);
        assert_eq!(
            NUM_POSITIONS, 730,
            "matches positional_embedding [730, 1280]"
        );
        assert_eq!(VISION_WIDTH / VISION_HEADS, 80, "vision head_dim");
        assert_eq!(TEXT_WIDTH / TEXT_HEADS, 64, "text head_dim");
        assert_eq!(VISION_MLP, 5120);
        assert_eq!(TEXT_MLP, 4096);
    }

    #[test]
    fn quick_gelu_matches_formula() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], 4, &dev).unwrap();
        let y = quick_gelu(&x).unwrap().to_vec1::<f32>().unwrap();
        for (i, &xi) in [-1.0f32, 0.0, 1.0, 2.0].iter().enumerate() {
            let expect = xi * (1.0 / (1.0 + (-1.702 * xi).exp()));
            assert!((y[i] - expect).abs() < 1e-6, "quickgelu[{i}]");
        }
    }

    #[test]
    fn wrap_tokens_sot_eot_pad() {
        let row = wrap_tokens(&[10, 11, 12]);
        assert_eq!(row.len(), CONTEXT_LENGTH);
        assert_eq!(row[0], SOT_TOKEN);
        assert_eq!(&row[1..4], &[10, 11, 12]);
        assert_eq!(row[4], EOT_TOKEN);
        assert!(row[5..].iter().all(|&t| t == PAD_TOKEN));
    }

    #[test]
    fn wrap_tokens_truncates_keeping_eot() {
        let long: Vec<u32> = (0..200).collect();
        let row = wrap_tokens(&long);
        assert_eq!(row.len(), CONTEXT_LENGTH);
        assert_eq!(row[0], SOT_TOKEN);
        assert_eq!(
            row[CONTEXT_LENGTH - 1],
            EOT_TOKEN,
            "EOT survives truncation"
        );
    }

    #[test]
    fn causal_mask_is_upper_triangular_neg_inf() {
        let dev = Device::Cpu;
        let m = causal_mask(3, &dev).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(m[0][0], 0.0);
        assert_eq!(m[1][0], 0.0);
        assert_eq!(m[0][1], f32::MIN, "j>i masked");
        assert_eq!(m[2][1], 0.0);
    }

    #[test]
    fn byte_to_unicode_is_a_256_bijection() {
        let b2u = byte_to_unicode();
        assert_eq!(b2u.len(), 256, "one mapping per byte");
        let bytes: std::collections::HashSet<u8> = b2u.iter().map(|&(b, _)| b).collect();
        let chars: std::collections::HashSet<char> = b2u.iter().map(|&(_, c)| c).collect();
        assert_eq!(bytes.len(), 256, "all bytes distinct");
        assert_eq!(chars.len(), 256, "all chars distinct");
    }

    #[test]
    fn clip_bpe_rebuilds_the_full_vocab_and_specials() {
        let bpe = clip_bpe();
        assert_eq!(
            bpe.encoder.len(),
            VOCAB_SIZE,
            "rebuilt encoder is the full 49408-token vocab"
        );
        assert_eq!(bpe.ranks.len(), NUM_BPE_MERGES, "48894 merge ranks");
        assert_eq!(
            bpe.encoder.get("<start_of_text>").copied(),
            Some(SOT_TOKEN),
            "SOT id"
        );
        assert_eq!(
            bpe.encoder.get("<end_of_text>").copied(),
            Some(EOT_TOKEN),
            "EOT id"
        );
    }

    #[test]
    fn tokenize_str_wraps_pads_and_matches_known_ids() {
        // "dog barking" -> open_clip ids [SOT, 1929, 32676, EOT, 0...] (see fixture).
        let row = tokenize_str("dog barking");
        assert_eq!(row.len(), CONTEXT_LENGTH);
        assert_eq!(&row[..4], &[SOT_TOKEN, 1929, 32676, EOT_TOKEN]);
        assert!(row[4..].iter().all(|&t| t == PAD_TOKEN), "0-padded tail");

        // Empty prompt -> just [SOT, EOT, 0...].
        let empty = tokenize_str("");
        assert_eq!(&empty[..2], &[SOT_TOKEN, EOT_TOKEN]);
        assert!(empty[2..].iter().all(|&t| t == PAD_TOKEN));
    }

    #[test]
    fn tokenize_str_truncates_keeping_eot() {
        // A prompt far longer than 77 tokens must fill the row and keep EOT in the last slot.
        let long = "dog ".repeat(200);
        let row = tokenize_str(&long);
        assert_eq!(row[0], SOT_TOKEN);
        assert_eq!(
            row[CONTEXT_LENGTH - 1],
            EOT_TOKEN,
            "EOT survives truncation"
        );
        assert!(row.iter().all(|&t| t != PAD_TOKEN), "row is full (no pad)");
    }

    #[test]
    fn clip_clean_html_unescape_and_whitespace() {
        // html.unescape (amp/lt/gt + numeric) then whitespace-collapse + lowercase.
        assert_eq!(clip_clean("Cats &amp; Dogs"), "cats & dogs");
        assert_eq!(clip_clean("a &lt;b&gt; c"), "a <b> c");
        assert_eq!(clip_clean("100&#37; DONE"), "100% done");
        assert_eq!(clip_clean("  a\t b \n c  "), "a b c");
        // a bare, non-entity ampersand is preserved verbatim.
        assert_eq!(clip_clean("rock & roll"), "rock & roll");
    }

    #[test]
    fn clip_preprocess_shape_and_range() {
        let dev = Device::Cpu;
        let frames: Vec<image::RgbImage> = vec![image::RgbImage::from_pixel(
            400,
            300,
            image::Rgb([120, 60, 200]),
        )];
        let t = frames_to_clip_input(&frames, &dev).unwrap();
        assert_eq!(t.dims(), &[1, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
