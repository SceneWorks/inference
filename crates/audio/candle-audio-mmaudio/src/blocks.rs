//! The MotionFormer backbone primitives: the 3D patch embedding and the divided space-time
//! transformer block (`DividedSpaceTimeBlock` + `DividedAttention` from MMAudio's `vit_helper.py`).
//!
//! Divided attention factorizes full space-time self-attention into a **temporal** pass (each
//! spatial location attends across its `T` temporal tokens) followed by a **spatial** pass (each
//! frame attends across its `196` spatial tokens), with the CLS token attending to — and attended
//! by — every token in both passes. This is the exact two-pass scheme, CLS handling included.

use candle_audio::candle_core::{Result as CResult, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::config;
use crate::preprocess::softmax_last;

/// 3D patch embedding — MMAudio's `patch_embed_3d`, a single non-overlapping `Conv3d` with
/// kernel = stride = `(z=2, 16, 16)`, `3 → 768`. Because stride equals kernel, the convolution is
/// exactly a patchify + linear projection: we gather each `(c, dz, dh, dw)` patch in the Conv3d
/// weight's own memory order and matmul with the flattened weight `(768, 3·2·16·16=1536)`. This
/// avoids a candle Conv3d (which the pinned revision does not expose) while remaining bit-faithful.
pub struct PatchEmbed3d {
    weight: Tensor, // (EMBED_DIM, 1536)  — Conv3d weight (768,3,2,16,16) flattened
    bias: Tensor,   // (EMBED_DIM,)
}

impl PatchEmbed3d {
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        let z = config::PATCH_SIZE_TEMP;
        let p = config::PATCH_SIZE;
        let patch_elems = config::IN_CHANS * z * p * p;
        // Conv3d weight arrives as (out=768, in=3, kz=2, kh=16, kw=16); flatten trailing dims.
        let w = vb.get(
            (config::EMBED_DIM, config::IN_CHANS, z, p, p),
            "proj.weight",
        )?;
        let weight = w.reshape((config::EMBED_DIM, patch_elems))?;
        let bias = vb.get(config::EMBED_DIM, "proj.bias")?;
        Ok(Self { weight, bias })
    }

    /// `(BS, C=3, T=16, H=224, W=224)` → `(BS, 1568, 768)` patch tokens, temporal-major
    /// (`token = t·196 + h·14 + w`), matching Conv3d output `flatten(2).transpose(1,2)`.
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (bs, c, t, h, w) = x.dims5()?;
        let z = config::PATCH_SIZE_TEMP;
        let p = config::PATCH_SIZE;
        let (tt, hh, ww) = (t / z, h / p, w / p);
        // (BS, C, tt, z, hh, p, ww, p) — rank 8, so use the slice Shape form.
        let x = x.reshape(&[bs, c, tt, z, hh, p, ww, p])?;
        // → (BS, tt, hh, ww, C, z, p, p): token dims (tt,hh,ww) outer; patch content (C,z,p,p) inner
        //   in the SAME order as the Conv3d weight's (in=C, kz=z, kh=p, kw=p).
        let x = x.permute([0usize, 2, 4, 6, 1, 3, 5, 7])?.contiguous()?;
        let ntok = tt * hh * ww;
        let patch_elems = c * z * p * p;
        let x = x.reshape((bs, ntok, patch_elems))?;
        // (BS, ntok, patch_elems) · (patch_elems, EMBED_DIM) + bias
        let wt = self.weight.t()?; // (patch_elems, EMBED_DIM)
        let x = x.broadcast_matmul(&wt)?;
        x.broadcast_add(&self.bias)
    }
}

/// Scaled-dot-product attention over the leading `(batch, seq, head_dim)` layout, no mask
/// (`qkv_attn` in `vit_helper.py`, mask path unused because every video token is valid). `q` is
/// pre-scaled by the caller, matching the reference (`q *= self.scale` before this call).
fn qkv_attn(q: &Tensor, k: &Tensor, v: &Tensor) -> CResult<Tensor> {
    let sim = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
    let attn = softmax_last(&sim).map_err(candle_err)?;
    attn.matmul(v)
}

fn candle_err(e: crate::AudioError) -> candle_audio::candle_core::Error {
    candle_audio::candle_core::Error::Msg(e.to_string())
}

/// `DividedAttention` — one factorized-attention pass. `mode` selects the temporal vs spatial
/// rearrangement; the CLS token (index 0) is split out, attends to the full sequence, and is
/// prepended back to every attention group so all tokens can attend to it.
pub struct DividedAttention {
    qkv: Linear,
    proj: Linear,
}

/// Which factorization a [`DividedAttention`] call performs.
#[derive(Clone, Copy)]
pub enum Mode {
    /// `b (f n) d -> (b n) f d`: attend across the `f=T` temporal tokens per spatial location.
    Time,
    /// `b (f n) d -> (b f) n d`: attend across the `n=196` spatial tokens per frame.
    Space,
}

impl DividedAttention {
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        // QKV bias is enabled (`VIT.QKV_BIAS=True`).
        let qkv = linear(config::EMBED_DIM, config::EMBED_DIM * 3, vb.pp("qkv"))?;
        let proj = linear(config::EMBED_DIM, config::EMBED_DIM, vb.pp("proj"))?;
        Ok(Self { qkv, proj })
    }

    /// `x`: `(B, N=1+F·n, D)`. `f` = temporal frames (`T`), `n` = spatial tokens (196).
    pub fn forward(&self, x: &Tensor, mode: Mode, f: usize, n: usize) -> CResult<Tensor> {
        let (b, ntot, _d) = x.dims3()?;
        let h = config::NUM_HEADS;
        let hd = config::HEAD_DIM;
        let scale = (hd as f64).powf(-0.5);

        // qkv → 3×(B, N, D); reshape each to (B·h, N, hd): 'b n (h d) -> (b h) n d'.
        let qkv = self.qkv.forward(x)?; // (B, N, 3D)
        let qkv = qkv.reshape((b, ntot, 3, h, hd))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // (B, N, h, hd)
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        let to_bh = |t: &Tensor| -> CResult<Tensor> {
            t.transpose(1, 2)?.contiguous()?.reshape((b * h, ntot, hd))
        };
        let q = (to_bh(&q)? * scale)?; // pre-scale q (reference does `q *= scale`)
        let k = to_bh(&k)?;
        let v = to_bh(&v)?;

        let bh = b * h;
        // Split CLS (index 0) from the F·n content tokens.
        let cls_q = q.narrow(1, 0, 1)?; // (bh, 1, hd)
        let cls_k = k.narrow(1, 0, 1)?;
        let cls_v = v.narrow(1, 0, 1)?;
        let q_ = q.narrow(1, 1, ntot - 1)?.contiguous()?; // (bh, F·n, hd)
        let k_ = k.narrow(1, 1, ntot - 1)?.contiguous()?;
        let v_ = v.narrow(1, 1, ntot - 1)?.contiguous()?;

        // CLS attends over the FULL (cls + content) sequence.
        let cls_out = qkv_attn(&cls_q, &k, &v)?; // (bh, 1, hd)

        // Rearrange content tokens into per-group sequences.
        // Layout of the F·n tokens is temporal-major: token = fi·n + ni.
        let regroup = |t: &Tensor| -> CResult<Tensor> {
            // (bh, F, n, hd)
            let t = t.reshape((bh, f, n, hd))?;
            match mode {
                // '(b n) f d': group by spatial n, sequence over f.
                Mode::Time => t.transpose(1, 2)?.contiguous()?.reshape((bh * n, f, hd)),
                // '(b f) n d': group by frame f, sequence over n.
                Mode::Space => t.contiguous()?.reshape((bh * f, n, hd)),
            }
        };
        let q_g = regroup(&q_)?;
        let mut k_g = regroup(&k_)?;
        let mut v_g = regroup(&v_)?;
        let groups = q_g.dim(0)?; // bh·n (time) or bh·f (space)
        let r = groups / bh; // n (time) or f (space)

        // Repeat CLS across the r groups per (b·h) and prepend so every token attends to CLS:
        // 'b () d -> (b r) () d'.
        let repeat_cls = |c: &Tensor| -> CResult<Tensor> {
            // (bh, 1, hd) -> (bh, 1, 1, hd) -> (bh, r, 1, hd) -> (bh·r, 1, hd)
            c.reshape((bh, 1, 1, hd))?
                .broadcast_as((bh, r, 1, hd))?
                .contiguous()?
                .reshape((bh * r, 1, hd))
        };
        let cls_k_r = repeat_cls(&cls_k)?;
        let cls_v_r = repeat_cls(&cls_v)?;
        k_g = Tensor::cat(&[&cls_k_r, &k_g], 1)?; // (groups, 1+seq, hd)
        v_g = Tensor::cat(&[&cls_v_r, &v_g], 1)?;

        let out = qkv_attn(&q_g, &k_g, &v_g)?; // (groups, seq, hd)

        // Rearrange back to (bh, F·n, hd): inverse of `regroup`.
        let out = match mode {
            // '(b n) f d -> b (f n) d'
            Mode::Time => out
                .reshape((bh, n, f, hd))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((bh, f * n, hd))?,
            // '(b f) n d -> b (f n) d'
            Mode::Space => out.reshape((bh, f * n, hd))?,
        };
        // Reattach CLS at index 0.
        let out = Tensor::cat(&[&cls_out, &out], 1)?; // (bh, N, hd)
                                                      // '(b h) n d -> b n (h d)'
        let out = out
            .reshape((b, h, ntot, hd))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, ntot, config::EMBED_DIM))?;
        self.proj.forward(&out)
    }
}

/// MotionFormer MLP: `fc1(768→3072) → GELU(erf) → fc2(3072→768)`.
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            fc1: linear(config::EMBED_DIM, config::MLP_HIDDEN, vb.pp("fc1"))?,
            fc2: linear(config::MLP_HIDDEN, config::EMBED_DIM, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // nn.GELU() default is the exact erf gelu.
        let x = self.fc1.forward(x)?.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

/// `DividedSpaceTimeBlock`: temporal attention (`norm3`→`timeattn`) → spatial attention
/// (`norm1`→`attn`) → MLP (`norm2`→`mlp`), each a residual add, in that exact order.
pub struct DividedSpaceTimeBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    attn: DividedAttention,     // spatial
    timeattn: DividedAttention, // temporal
    mlp: Mlp,
}

impl DividedSpaceTimeBlock {
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            norm1: layer_norm(config::EMBED_DIM, config::LN_EPS, vb.pp("norm1"))?,
            norm2: layer_norm(config::EMBED_DIM, config::LN_EPS, vb.pp("norm2"))?,
            norm3: layer_norm(config::EMBED_DIM, config::LN_EPS, vb.pp("norm3"))?,
            attn: DividedAttention::load(vb.pp("attn"))?,
            timeattn: DividedAttention::load(vb.pp("timeattn"))?,
            mlp: Mlp::load(vb.pp("mlp"))?,
        })
    }

    /// `x`: `(B, 1+F·n, D)`; `f` = T temporal tokens, `n` = 196 spatial tokens.
    pub fn forward(&self, x: &Tensor, f: usize, n: usize) -> CResult<Tensor> {
        let time_out = self
            .timeattn
            .forward(&self.norm3.forward(x)?, Mode::Time, f, n)?;
        let time_res = (x + time_out)?;
        let space_out = self
            .attn
            .forward(&self.norm1.forward(&time_res)?, Mode::Space, f, n)?;
        let space_res = (time_res + space_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&space_res)?)?;
        &space_res + mlp_out
    }
}
