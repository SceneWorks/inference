//! SCAIL-2's CLIP image encoder — the open-CLIP **XLM-RoBERTa ViT-H/14** *visual tower* (upstream
//! `wan/modules/clip.py` `VisionTransformer`, the "onlyvisual" checkpoint).
//!
//! The reference image is encoded to the `[1, 257, 1280]` features the DiT's `img_emb` consumes
//! (`Scail2Dit`'s `clip_fea`). Crucially this is the **`use_31_block=True`** path: patch-embed → prepend
//! cls → add pos → pre-norm → run only the **first 31 of 32** transformer blocks, returning the
//! **penultimate** hidden state — *no* `post_norm`, *no* `head` projection, *no* pooling.
//!
//! A standard pre-norm ViT: Conv2d(3→1280, 14×14, stride 14, no bias) read as an `[out, 3·14·14]`
//! Linear (stride==kernel, like the DiT patch stems); 32 blocks with `x = x + attn(norm1(x))` then
//! `x = x + mlp(norm2(x))`; **fused `to_qkv`** (`[3·dim, dim]`); **exact GELU** (`activation='gelu'`);
//! LayerNorm eps 1e-5. We reuse only mlx primitives — the SDXL `ClipVisionEncoder` is the same
//! architecture but uses separate q/k/v and the HF `vision_model.*` key scheme, incompatible with the
//! open-CLIP fused-qkv `visual.transformer.*` weights.
//!
//! Image preprocessing (224² bicubic resize, `[-1,1]→[0,1]`, CLIP mean/std normalize) is the caller's
//! concern (the pipeline / preprocessing slice); [`ScailClip::encode`] takes a preprocessed
//! `[B, 3, 224, 224]` pixel tensor.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::gelu_exact;
use mlx_gen::weights::{to_f32, Weights};
use mlx_gen::Result;
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, split};
use mlx_rs::{Array, Dtype};

/// open-CLIP ViT visual-tower geometry. The shipped SCAIL-2 CLIP is always [`ClipVisionConfig::vit_h_14`].
#[derive(Clone, Debug)]
pub struct ClipVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    /// Total transformer blocks (32 for ViT-H/14). The `use_31_block` path runs `num_layers - 1`.
    pub num_layers: usize,
    pub eps: f32,
}

impl ClipVisionConfig {
    /// open-CLIP XLM-RoBERTa ViT-H/14 (the SCAIL-2 / Wan-I2V image encoder).
    pub fn vit_h_14() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            dim: 1280,
            num_heads: 16,
            num_layers: 32,
            eps: 1e-5,
        }
    }

    /// Blocks actually executed by the `use_31_block` penultimate path.
    pub fn run_layers(&self) -> usize {
        self.num_layers - 1
    }
}

fn load_lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(
        w.require(&format!("{prefix}.weight"))?.clone(),
        Some(w.require(&format!("{prefix}.bias"))?.clone()),
    ))
}

/// One pre-norm ViT block: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
struct ClipBlock {
    norm1_w: Array,
    norm1_b: Array,
    to_qkv: AdaptableLinear,
    proj: AdaptableLinear,
    norm2_w: Array,
    norm2_b: Array,
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
    n: i32,
    d: i32,
    scale: f32,
    eps: f32,
}

impl ClipBlock {
    fn load(w: &Weights, i: usize, cfg: &ClipVisionConfig) -> Result<Self> {
        let p = format!("transformer.{i}");
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            norm1_w: to_f32(w.require(&format!("{p}.norm1.weight"))?)?,
            norm1_b: to_f32(w.require(&format!("{p}.norm1.bias"))?)?,
            to_qkv: load_lin(w, &format!("{p}.attn.to_qkv"))?,
            proj: load_lin(w, &format!("{p}.attn.proj"))?,
            norm2_w: to_f32(w.require(&format!("{p}.norm2.weight"))?)?,
            norm2_b: to_f32(w.require(&format!("{p}.norm2.bias"))?)?,
            mlp0: load_lin(w, &format!("{p}.mlp.0"))?,
            mlp2: load_lin(w, &format!("{p}.mlp.2"))?,
            n: cfg.num_heads as i32,
            d: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.eps,
        })
    }

    fn attn(&self, x: &Array, cdt: Dtype) -> Result<Array> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (n, d) = (self.n, self.d);
        // Fused qkv: [b, s, 3·dim] → [b, s, 3, n, d], split the "3" axis.
        let qkv = self
            .to_qkv
            .forward(&x.as_dtype(cdt)?)?
            .reshape(&[b, s, 3, n, d])?;
        let parts = split(&qkv, 3, 2)?;
        let head = |t: &Array| -> Result<Array> {
            Ok(t.reshape(&[b, s, n, d])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = head(&parts[0])?;
        let k = head(&parts[1])?;
        let v = head(&parts[2])?;
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.proj.forward(&out)
    }

    fn mlp(&self, x: &Array, cdt: Dtype) -> Result<Array> {
        let h = gelu_exact(&self.mlp0.forward(&x.as_dtype(cdt)?)?)?;
        self.mlp2.forward(&h)
    }

    /// `x`: `[B, L, dim]` (f32).
    fn forward(&self, x: &Array, cdt: Dtype) -> Result<Array> {
        let h = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), self.eps)?;
        let x = add(x, &to_f32(&self.attn(&h, cdt)?)?)?;
        let h = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), self.eps)?;
        add(&x, &to_f32(&self.mlp(&h, cdt)?)?).map_err(Into::into)
    }
}

/// The SCAIL-2 CLIP visual tower (penultimate-feature extractor).
pub struct ScailClip {
    patch_embedding: AdaptableLinear, // Conv2d(3→dim, p×p) as [dim, 3·p·p]
    cls: Array,                       // [1, 1, dim]
    pos: Array,                       // [1, num_patches+1, dim]
    pre_norm_w: Array,
    pre_norm_b: Array,
    blocks: Vec<ClipBlock>, // run_layers (31 for ViT-H/14)
    cfg: ClipVisionConfig,
    compute_dtype: Dtype,
}

impl ScailClip {
    /// Load the visual tower from a `Weights` view (the de-prefixed `onlyvisual` state dict:
    /// `patch_embedding.weight`, `cls_embedding`, `pos_embedding`, `pre_norm.*`, `transformer.{i}.*`).
    /// Only the `run_layers` blocks the penultimate path needs are loaded — `post_norm` and `head` are
    /// skipped.
    pub fn from_weights(w: &Weights, cfg: &ClipVisionConfig) -> Result<Self> {
        let pe = w.require("patch_embedding.weight")?; // [dim, 3, p, p]
        let s = pe.shape();
        let pe = pe.reshape(&[s[0], s[1] * s[2] * s[3]])?; // [dim, 3·p·p]
        let mut blocks = Vec::with_capacity(cfg.run_layers());
        for i in 0..cfg.run_layers() {
            blocks.push(ClipBlock::load(w, i, cfg)?);
        }
        Ok(Self {
            patch_embedding: AdaptableLinear::dense(pe, None),
            cls: to_f32(w.require("cls_embedding")?)?,
            pos: to_f32(w.require("pos_embedding")?)?,
            pre_norm_w: to_f32(w.require("pre_norm.weight")?)?,
            pre_norm_b: to_f32(w.require("pre_norm.bias")?)?,
            blocks,
            cfg: cfg.clone(),
            compute_dtype: Dtype::Float32,
        })
    }

    /// Set the matmul compute dtype (f32 default; bf16 for production).
    pub fn set_compute_dtype(&mut self, dt: Dtype) {
        self.compute_dtype = dt;
    }

    /// Encode a preprocessed pixel tensor `[B, 3, image_size, image_size]` → penultimate CLIP features
    /// `[B, num_patches + 1, dim]` (e.g. `[1, 257, 1280]` for ViT-H/14 at 224²).
    pub fn encode(&self, pixel: &Array) -> Result<Array> {
        let cdt = self.compute_dtype;
        let p = self.cfg.patch_size as i32;
        let b = pixel.shape()[0];
        let h = pixel.shape()[2];
        let wd = pixel.shape()[3];
        let (nh, nw) = (h / p, wd / p);
        let dim = self.cfg.dim as i32;

        // Patchify [B, 3, H, W] → [B, nh·nw, 3·p·p] (feature order (c, kh, kw) matches the Conv flatten),
        // then the Conv-as-Linear patch embed.
        let tokens = pixel
            .reshape(&[b, 3, nh, p, nw, p])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, nh * nw, 3 * p * p])?;
        let x = self.patch_embedding.forward(&tokens.as_dtype(cdt)?)?;
        let x = to_f32(&x)?;

        // Prepend cls, add positional, pre-norm.
        let cls = self.cls.reshape(&[1, 1, dim])?;
        let cls = mlx_rs::ops::broadcast_to(&cls, &[b, 1, dim])?;
        let x = concatenate_axis(&[&cls, &x], 1)?; // [B, nh·nw+1, dim]
        let x = add(&x, &self.pos)?;
        let mut x = layer_norm(
            &x,
            Some(&self.pre_norm_w),
            Some(&self.pre_norm_b),
            self.cfg.eps,
        )?;

        for block in &self.blocks {
            x = block.forward(&x, cdt)?;
        }
        Ok(x)
    }
}
