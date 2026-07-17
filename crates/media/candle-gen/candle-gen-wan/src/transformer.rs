//! The **`WanTransformer3DModel`** DiT (TI2V-5B, dense) — a port of diffusers `transformer_wan.py`.
//! 30 blocks, each: AdaLN-modulated self-attention (3-axis interleaved RoPE, full-dim qk-RMSNorm) →
//! ungated cross-attention to the UMT5 context → AdaLN-modulated gated GELU FFN. The per-block
//! 6-vector modulation is `scale_shift_table + time_proj`; the head uses a separate 2-vector.
//!
//! Runs in **bf16** (the 5B checkpoint's native dtype) with norms / modulation / RoPE upcast to f32,
//! mirroring diffusers' `FP32LayerNorm` + `.float()` modulation.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, VarBuilder};

use crate::config::TransformerConfig;
use crate::quant::QLinear;
use crate::rope::apply_rope;

/// Dense Linear loader — retained for the VACE model (`vace.rs`) and the training DiT (`dit_train.rs`),
/// whose tiers are not packed. The inference DiT Linears route through [`qlinear`] (packed-detect).
pub(crate) fn linear(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((out_c, in_c), "weight")?,
        Some(vb.get(out_c, "bias")?),
    ))
}

/// LayerNorm over the last dim with no learnable affine, in f32.
pub(crate) fn ln_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// RMSNorm over the last dim (qk-norm "across heads") with affine weight, in f32.
pub(crate) fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dt)
}

/// Scaled-dot-product attention. `q,k,v`: `[B, H, S*, d]`; softmax upcast to f32.
///
/// Delegates to the shared i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] — the sc-6217 /
/// sc-9116 query-row chunking Qwen and Krea already carry, ported to Wan here (sc-12434). It chunks
/// over the query rows once the `[B, H, Sq, Sk]` score block would exceed
/// [`candle_gen::ATTN_SCORES_BUDGET`], so the full `[B, H, S, S]` matrix is never materialized. Both
/// A14B experts and the 5B share this one attention; at every advertised A14B geometry the un-chunked
/// score block (S ≈ 33k tokens at 480p, up to ≈ 76k at the 720p `MAX_AREA_14B` ceiling; 40 heads) is
/// hundreds of GiB and OOMs a 96 GB card before the first denoise step — chunking caps each block's
/// transient near the budget instead. Each query row's softmax is over all keys and independent of the
/// others, so the chunked result equals the single pass; the chunking engages only on the over-budget
/// denoise self-attention and stays a no-op single pass for the small cross-attention (S_kv = text
/// tokens) and every in-budget size.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let dtype = q.dtype();
    sdpa_budgeted(
        q,
        k,
        v,
        scale,
        candle_gen::ATTN_SCORES_BUDGET,
        |scores: &Tensor| softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype),
    )
}

/// [`sdpa`] with an explicit scores-element `budget` (the query-row chunk threshold) and `softmax`
/// closure — the shared budgeted attention both production and the tests route through, so the test's
/// call-counting proof exercises the same chunking the render uses. Production fixes the budget at
/// [`candle_gen::ATTN_SCORES_BUDGET`] and passes the f32-upcast softmax; the test drives a tiny budget
/// and a counting wrapper of that same softmax. Wan self- and cross-attention carry no mask, so `mask`
/// is always `None`.
fn sdpa_budgeted(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    budget: usize,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    candle_gen::sdpa_budgeted_bhsd(q, k, v, scale, None, softmax, budget)
}

struct Attention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.dim;
        Ok(Self {
            to_q: QLinear::linear_detect(cfg.dim, inner, &vb, "to_q", true)?,
            to_k: QLinear::linear_detect(cfg.dim, inner, &vb, "to_k", true)?,
            to_v: QLinear::linear_detect(cfg.dim, inner, &vb, "to_v", true)?,
            to_out: QLinear::linear_detect(inner, cfg.dim, &vb, "to_out.0", true)?,
            norm_q: vb.pp("norm_q").get(inner, "weight")?,
            norm_k: vb.pp("norm_k").get(inner, "weight")?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            eps: cfg.eps,
        })
    }

    /// Visit this attention's four adaptable projections (`{prefix}.{to_q,to_k,to_v,to_out.0}`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.to_q"), &mut self.to_q)?;
        f(&format!("{prefix}.to_k"), &mut self.to_k)?;
        f(&format!("{prefix}.to_v"), &mut self.to_v)?;
        f(&format!("{prefix}.to_out.0"), &mut self.to_out)?;
        Ok(())
    }

    /// `hidden`: `[B, S, dim]`; `context`: cross-attn K/V source (= hidden for self-attn). RoPE is
    /// applied only when `cos`/`sin` are given (self-attn).
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let s_kv = context.dim(1)?;
        let q = rms(&self.to_q.forward(hidden)?, &self.norm_q, self.eps)?;
        let k = rms(&self.to_k.forward(context)?, &self.norm_k, self.eps)?;
        let v = self.to_v.forward(context)?;
        let to_heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut q = to_heads(&q, s)?; // [B,H,S,d]
        let mut k = to_heads(&k, s_kv)?;
        let v = to_heads(&v, s_kv)?;
        if let Some((cos, sin)) = rope {
            q = apply_rope(&q, cos, sin)?;
            k = apply_rope(&k, cos, sin)?;
        }
        let scale = (self.head_dim as f64).powf(-0.5);
        let out = sdpa(&q, &k, &v, scale)?; // [B,H,S,d]
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

struct Ffn {
    proj: QLinear, // net.0.proj
    out: QLinear,  // net.2
}

impl Ffn {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: QLinear::linear_detect(cfg.dim, cfg.ffn_dim, &vb, "net.0.proj", true)?,
            out: QLinear::linear_detect(cfg.ffn_dim, cfg.dim, &vb, "net.2", true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }

    /// Visit the FFN's two adaptable projections (`{prefix}.net.0.proj`, `{prefix}.net.2`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.net.0.proj"), &mut self.proj)?;
        f(&format!("{prefix}.net.2"), &mut self.out)?;
        Ok(())
    }
}

pub(crate) struct Block {
    scale_shift_table: Tensor, // [1,6,dim] f32
    attn1: Attention,
    norm2_w: Tensor, // affine cross-attn norm
    norm2_b: Tensor,
    attn2: Attention,
    ffn: Ffn,
    eps: f64,
}

impl Block {
    pub(crate) fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale_shift_table: vb
                .get((1, 6, cfg.dim), "scale_shift_table")?
                .to_dtype(DType::F32)?,
            attn1: Attention::new(cfg, vb.pp("attn1"))?,
            norm2_w: vb
                .pp("norm2")
                .get(cfg.dim, "weight")?
                .to_dtype(DType::F32)?,
            norm2_b: vb.pp("norm2").get(cfg.dim, "bias")?.to_dtype(DType::F32)?,
            attn2: Attention::new(cfg, vb.pp("attn2"))?,
            ffn: Ffn::new(cfg, vb.pp("ffn"))?,
            eps: cfg.eps,
        })
    }

    /// `hidden`: `[B,S,dim]` (bf16); `temb6`: `[B,6,dim]` (f32); `context`: `[B,S_ctx,dim]` (bf16).
    pub(crate) fn forward(
        &self,
        hidden: &Tensor,
        temb6: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dt = hidden.dtype();
        // mods: scale_shift_table[1,6,dim] + temb6[B,6,dim] → 6 × [B,1,dim] (f32).
        let mods = self.scale_shift_table.broadcast_add(temb6)?;
        let m = |i: usize| -> Result<Tensor> { mods.narrow(1, i, 1) };
        let (shift_msa, scale_msa, gate_msa) = (m(0)?, m(1)?, m(2)?);
        let (c_shift, c_scale, c_gate) = (m(3)?, m(4)?, m(5)?);

        let hf = hidden.to_dtype(DType::F32)?;
        // 1. self-attention
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?
            .to_dtype(dt)?;
        let a = self.attn1.forward(&n, &n, Some((cos, sin)))?;
        let hf = (hf + a.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?;

        // 2. cross-attention (affine norm2, ungated)
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&self.norm2_w)?
            .broadcast_add(&self.norm2_b)?
            .to_dtype(dt)?;
        let a = self.attn2.forward(&n, context, None)?;
        let hf = (hf + a.to_dtype(DType::F32)?)?;

        // 3. feed-forward
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(c_scale + 1.0)?)?
            .broadcast_add(&c_shift)?
            .to_dtype(dt)?;
        let f = self.ffn.forward(&n)?;
        let hf = (hf + f.to_dtype(DType::F32)?.broadcast_mul(&c_gate)?)?;
        hf.to_dtype(dt)
    }

    /// Visit this block's adaptable projections (`{prefix}.attn1/attn2.*`, `{prefix}.ffn.*`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        self.attn1
            .visit_adaptable_mut(&format!("{prefix}.attn1"), f)?;
        self.attn2
            .visit_adaptable_mut(&format!("{prefix}.attn2"), f)?;
        self.ffn.visit_adaptable_mut(&format!("{prefix}.ffn"), f)?;
        Ok(())
    }
}

/// Build the `[B, freq_dim]` sinusoidal timestep embedding (diffusers `Timesteps`,
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`): `[cos(t·ω) | sin(t·ω)]`.
pub(crate) fn timestep_sinusoid(t: f64, freq_dim: usize, b: usize, dev: &Device) -> Result<Tensor> {
    let half = freq_dim / 2;
    let mut row = vec![0f32; freq_dim];
    for i in 0..half {
        let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
        let ang = t * freq;
        row[i] = ang.cos() as f32;
        row[half + i] = ang.sin() as f32;
    }
    let one = Tensor::from_vec(row, (1, freq_dim), dev)?;
    if b == 1 {
        Ok(one)
    } else {
        Ok(one.broadcast_as((b, freq_dim))?.contiguous()?)
    }
}

pub struct WanTransformer {
    patch_w: Tensor, // [dim,48,p_h,p_w]
    patch_b: Tensor, // [1,dim,1,1]
    text_l1: QLinear,
    text_l2: QLinear,
    time_l1: QLinear,
    time_l2: QLinear,
    time_proj: QLinear,
    blocks: Vec<Block>,
    norm_out_eps: f64,
    proj_out: QLinear,
    scale_shift_table: Tensor, // [1,2,dim] f32
    cfg: TransformerConfig,
    device: Device,
    dtype: DType,
}

impl WanTransformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let (pt, ph, pw) = cfg.patch;
        // patch_embedding is a Conv3d (1,2,2); temporal kernel 1 → squeeze to a per-frame conv2d.
        let pw_full = vb.get(
            (cfg.dim, cfg.in_channels, pt, ph, pw),
            "patch_embedding.weight",
        )?;
        let patch_w = pw_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [dim,48,ph,pw]
        let patch_b = vb
            .get(cfg.dim, "patch_embedding.bias")?
            .reshape((1, cfg.dim, 1, 1))?;

        let ce = vb.pp("condition_embedder");
        let text_l1 =
            QLinear::linear_detect(cfg.text_dim, cfg.dim, &ce, "text_embedder.linear_1", true)?;
        let text_l2 =
            QLinear::linear_detect(cfg.dim, cfg.dim, &ce, "text_embedder.linear_2", true)?;
        let time_l1 =
            QLinear::linear_detect(cfg.freq_dim, cfg.dim, &ce, "time_embedder.linear_1", true)?;
        let time_l2 =
            QLinear::linear_detect(cfg.dim, cfg.dim, &ce, "time_embedder.linear_2", true)?;
        let time_proj = QLinear::linear_detect(cfg.dim, 6 * cfg.dim, &ce, "time_proj", true)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let proj_out = QLinear::linear_detect(
            cfg.dim,
            cfg.out_channels * pt * ph * pw,
            &vb,
            "proj_out",
            true,
        )?;
        let scale_shift_table = vb
            .get((1, 2, cfg.dim), "scale_shift_table")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            patch_w,
            patch_b,
            text_l1,
            text_l2,
            time_l1,
            time_l2,
            time_proj,
            blocks,
            norm_out_eps: cfg.eps,
            proj_out,
            scale_shift_table,
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Project UMT5 prompt embeds `[B,S,4096]` → cross-attn context `[B,S,dim]` (constant across the
    /// denoise loop). `gelu_tanh` between the two linears (PixArtAlphaTextProjection).
    pub fn embed_text(&self, prompt_embeds: &Tensor) -> Result<Tensor> {
        let x = prompt_embeds.to_dtype(self.dtype)?;
        self.text_l2.forward(&self.text_l1.forward(&x)?.gelu()?)
    }

    /// One DiT forward: `latents [B,in_c,F,Hl,Wl]`, projected `context [B,S,dim]`, scalar `t`,
    /// RoPE `cos`/`sin [L,64]` → predicted velocity `[B,out_c,F,Hl,Wl]`.
    ///
    /// Composed from the three seams below (patch-embed → block-stack/head → unpatchify), byte-identical
    /// to the previous monolithic body. The seams are exposed additively for the Bernini renderer's
    /// token-axis packed conditioning (sc-11004), which patch-embeds the target + each source separately
    /// and runs one packed [`forward_packed`](Self::forward_packed) over the concatenated token axis.
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        t: f64,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (tokens, grid) = self.patch_embed_tokens(latents)?;
        let out = self.forward_packed(&tokens, t, context, cos, sin)?;
        self.unpatchify_tokens(&out, grid)
    }

    /// Patch-embed `latents [B, in_channels, F, Hl, Wl]` into the DiT token stream `[B, L, dim]` (bf16)
    /// plus the patch grid `(ppf, pph, ppw)` — the embedding half of [`forward`](Self::forward), exposed
    /// as a seam for the Bernini renderer (sc-11004), which patch-embeds the noisy target **and** each
    /// conditioning source separately (each with its own source-id RoPE) and concatenates them on the
    /// token axis before a single packed forward. `L = ppf·pph·ppw`.
    pub fn patch_embed_tokens(&self, latents: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
        let (b, _c, f, hl, wl) = latents.dims5()?;
        let (pt, ph, pw) = self.cfg.patch;
        let (ppf, pph, ppw) = (f / pt, hl / ph, wl / pw);

        // Patch embed: per-frame strided conv2d, then flatten to tokens (f outer, then h, w).
        let merged = latents
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * f, self.cfg.in_channels, hl, wl))?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let y = merged.conv2d(&self.patch_w, 0, ph, 1, 1)?; // [B*F,dim,pph,ppw]
        let y = y.broadcast_add(&self.patch_b)?;
        let hidden = y
            .reshape((b, f, self.cfg.dim, pph, ppw))?
            .permute((0, 1, 3, 4, 2))? // [B,F,pph,ppw,dim]
            .reshape((b, ppf * pph * ppw, self.cfg.dim))?
            .contiguous()?;
        Ok((hidden, (ppf, pph, ppw)))
    }

    /// Run the block stack + output head over a **pre-embedded, pre-packed** token sequence
    /// `tokens [B, L, dim]` (bf16) with caller-supplied RoPE `cos`/`sin [L, head_dim/2]` and the
    /// projected cross-attention `context [B, S, dim]` — returning the per-token velocity
    /// `[B, L, out_channels·∏patch]` (this DiT's dtype) **without** unpatchifying. This is
    /// [`forward`](Self::forward)'s body minus the patch-embed in / unpatchify out, the seam the Bernini
    /// renderer (sc-11004) uses: at batch 1 the packed `[sources…, target]` sequence is plain full
    /// self-attention (the reference's varlen attention with a single `cu_seqlens` segment), so the
    /// caller assembles the token + RoPE concat, calls this once, then slices the target tokens and
    /// [`unpatchify_tokens`](Self::unpatchify_tokens) them.
    pub fn forward_packed(
        &self,
        tokens: &Tensor,
        t: f64,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, _l, _dim) = tokens.dims3()?;
        // Time embedding → temb [B,dim], and the per-block 6-vector temb6 [B,6,dim] (f32).
        let sinus =
            timestep_sinusoid(t, self.cfg.freq_dim, b, &self.device)?.to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?; // [B,dim]
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, 6, self.cfg.dim))?
            .to_dtype(DType::F32)?;

        let mut hidden = tokens.clone();
        for blk in &self.blocks {
            hidden = blk.forward(&hidden, &temb6, context, cos, sin)?;
        }

        // Head: norm_out (non-affine) modulated by scale_shift_table + temb.
        let head_mod = self
            .scale_shift_table
            .broadcast_add(&temb.unsqueeze(1)?.to_dtype(DType::F32)?)?;
        let shift = head_mod.narrow(1, 0, 1)?;
        let scale = head_mod.narrow(1, 1, 1)?;
        let hf = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hf, self.norm_out_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        self.proj_out.forward(&normed) // [B,L,out_c*patch]
    }

    /// Unpatchify a per-token velocity `[B, L, out_channels·∏patch]` (with `L = ppf·pph·ppw`) back to a
    /// spatial latent `[B, out_channels, F, Hl, Wl]` (f32) — the tail of [`forward`](Self::forward),
    /// exposed so the Bernini renderer can unpatchify the **target-sliced** packed output (sc-11004).
    pub fn unpatchify_tokens(&self, out: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (ppf, pph, ppw) = grid;
        let (b, _l, _op) = out.dims3()?;
        let (pt, ph, pw) = self.cfg.patch;
        let oc = self.cfg.out_channels;
        out.reshape(&[b, ppf, pph, ppw, pt, ph, pw, oc][..])?
            .permute(&[0usize, 7, 1, 4, 2, 5, 3, 6][..])?
            .reshape((b, oc, ppf * pt, pph * ph, ppw * pw))?
            .to_dtype(DType::F32)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this DiT loaded from a **packed** MLX tier (its projections are quantized) — the additive
    /// router uses this to reject LoKr/LoHa on a packed base (sc-10094). Probed on `proj_out` (every
    /// projection in a tier packs together; a dense checkpoint packs none).
    pub fn is_packed(&self) -> bool {
        self.proj_out.is_packed()
    }

    /// The canonical dotted paths of every adaptable projection (attention q/k/v/out, FFN, the
    /// condition-embedder projections, `time_proj`, `proj_out`) — the LoRA merge surface, in the diffusers
    /// key namespace. Drives the additive-adapter kohya `flat→dotted` table (sc-10094).
    pub fn adaptable_paths(&self) -> Vec<String> {
        let mut paths = vec![
            "condition_embedder.text_embedder.linear_1".to_string(),
            "condition_embedder.text_embedder.linear_2".to_string(),
            "condition_embedder.time_embedder.linear_1".to_string(),
            "condition_embedder.time_embedder.linear_2".to_string(),
            "condition_embedder.time_proj".to_string(),
        ];
        for i in 0..self.blocks.len() {
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    paths.push(format!("blocks.{i}.{attn}.{leaf}"));
                }
            }
            paths.push(format!("blocks.{i}.ffn.net.0.proj"));
            paths.push(format!("blocks.{i}.ffn.net.2"));
        }
        paths.push("proj_out".to_string());
        paths
    }

    /// Walk every adaptable projection, invoking `f(path, &mut QLinear)` once each with the projection's
    /// canonical dotted path — the host visitor the additive-adapter installer routes residuals through
    /// (sc-10094; the candle analog of mlx-gen's `AdaptableHost`). The order matches
    /// [`Self::adaptable_paths`].
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(
            "condition_embedder.text_embedder.linear_1",
            &mut self.text_l1,
        )?;
        f(
            "condition_embedder.text_embedder.linear_2",
            &mut self.text_l2,
        )?;
        f(
            "condition_embedder.time_embedder.linear_1",
            &mut self.time_l1,
        )?;
        f(
            "condition_embedder.time_embedder.linear_2",
            &mut self.time_l2,
        )?;
        f("condition_embedder.time_proj", &mut self.time_proj)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("blocks.{i}"), f)?;
        }
        f("proj_out", &mut self.proj_out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::{apply_source_id, WanRope};
    use std::collections::HashMap;

    /// A tiny dense config the CPU synthetic weights below fill (dim 16 = 2 heads × head_dim 8, z16
    /// in/out, patch (1,2,2)). Keeps the packed-forward geometry (`ppf·pph·ppw` tokens, 3-axis RoPE) but
    /// small enough to run on CPU without weights.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            dim: 16,
            ffn_dim: 32,
            freq_dim: 16,
            text_dim: 16,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    /// Build a synthetic `WanTransformer` (all dense) from randn weights — every tensor key
    /// [`WanTransformer::new`] reads, at [`DType::F32`] so the whole forward runs on CPU.
    fn tiny_dit(cfg: &TransformerConfig, dev: &Device) -> WanTransformer {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: &str, shape: &[usize]| {
            m.insert(
                k.to_string(),
                Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
            );
        };
        let (pt, ph, pw) = cfg.patch;
        let d = cfg.dim;
        put("patch_embedding.weight", &[d, cfg.in_channels, pt, ph, pw]);
        put("patch_embedding.bias", &[d]);
        put(
            "condition_embedder.text_embedder.linear_1.weight",
            &[d, cfg.text_dim],
        );
        put("condition_embedder.text_embedder.linear_1.bias", &[d]);
        put("condition_embedder.text_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.text_embedder.linear_2.bias", &[d]);
        put(
            "condition_embedder.time_embedder.linear_1.weight",
            &[d, cfg.freq_dim],
        );
        put("condition_embedder.time_embedder.linear_1.bias", &[d]);
        put("condition_embedder.time_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.time_embedder.linear_2.bias", &[d]);
        put("condition_embedder.time_proj.weight", &[6 * d, d]);
        put("condition_embedder.time_proj.bias", &[6 * d]);
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            put(&format!("{b}.scale_shift_table"), &[1, 6, d]);
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    put(&format!("{b}.{attn}.{leaf}.weight"), &[d, d]);
                    put(&format!("{b}.{attn}.{leaf}.bias"), &[d]);
                }
                put(&format!("{b}.{attn}.norm_q.weight"), &[d]);
                put(&format!("{b}.{attn}.norm_k.weight"), &[d]);
            }
            put(&format!("{b}.norm2.weight"), &[d]);
            put(&format!("{b}.norm2.bias"), &[d]);
            put(&format!("{b}.ffn.net.0.proj.weight"), &[cfg.ffn_dim, d]);
            put(&format!("{b}.ffn.net.0.proj.bias"), &[cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.weight"), &[d, cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.bias"), &[d]);
        }
        put("proj_out.weight", &[cfg.out_channels * pt * ph * pw, d]);
        put("proj_out.bias", &[cfg.out_channels * pt * ph * pw]);
        put("scale_shift_table", &[1, 2, d]);
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        WanTransformer::new(cfg, vb).unwrap()
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The refactored [`WanTransformer::forward`] is byte-identical to the explicit
    /// `patch_embed_tokens → forward_packed → unpatchify_tokens` composition — pins the additive seams
    /// to the validated monolithic forward (the many-crates-depend-on-it invariant).
    #[test]
    fn forward_equals_seam_composition() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let latents = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &dev).unwrap(); // L = 8
        let t = 833.0;
        let want = dit.forward(&latents, &context, t, &cos, &sin).unwrap();

        let (tokens, grid) = dit.patch_embed_tokens(&latents).unwrap();
        assert_eq!(grid, (2, 2, 2));
        assert_eq!(tokens.dims(), &[1, 8, cfg.dim]);
        let out = dit
            .forward_packed(&tokens, t, &context, &cos, &sin)
            .unwrap();
        assert_eq!(out.dims(), &[1, 8, cfg.out_channels * 4]);
        let got = dit.unpatchify_tokens(&out, grid).unwrap();
        assert_eq!(
            max_abs(&got, &want),
            0.0,
            "seam composition must equal forward"
        );
    }

    /// A conditioning source concatenated on the token axis extends the packed sequence, but the sliced
    /// target velocity keeps the target's shape — and the source actually couples into the target through
    /// self-attention (the packed target-slice differs from the target-only forward), with the source-id
    /// RoPE shifting the result. Mirrors the mlx `conditioning_source_preserves_target_shape` intent.
    #[test]
    fn packed_source_preserves_target_shape_and_couples() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let hd = cfg.head_dim;
        let rope = WanRope::new(&cfg);
        let t = 700.0;

        let target = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let source = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();

        let (tok_t, grid_t) = dit.patch_embed_tokens(&target).unwrap();
        let (cos_t, sin_t) = rope.cos_sin(grid_t.0, grid_t.1, grid_t.2, &dev).unwrap();
        let (tok_s, grid_s) = dit.patch_embed_tokens(&source).unwrap();
        let (cos_s0, sin_s0) = rope.cos_sin(grid_s.0, grid_s.1, grid_s.2, &dev).unwrap();
        // source id 1 shifts the source segment's RoPE; the target stays id 0.
        let (cos_s, sin_s) = apply_source_id(&cos_s0, &sin_s0, 1.0, hd).unwrap();

        let l_t = grid_t.0 * grid_t.1 * grid_t.2;
        let tokens = Tensor::cat(&[&tok_s, &tok_t], 1).unwrap();
        let cos = Tensor::cat(&[&cos_s, &cos_t], 0).unwrap();
        let sin = Tensor::cat(&[&sin_s, &sin_t], 0).unwrap();
        let out = dit
            .forward_packed(&tokens, t, &context, &cos, &sin)
            .unwrap();
        let total = out.dim(1).unwrap();
        let target_tokens = out.narrow(1, total - l_t, l_t).unwrap();
        let vel = dit.unpatchify_tokens(&target_tokens, grid_t).unwrap();
        assert_eq!(
            vel.dims(),
            target.dims(),
            "target velocity keeps target shape"
        );

        // Coupling: the packed target-slice differs from the target-only forward (the source tokens
        // entered the target through self-attention).
        let solo = dit.forward(&target, &context, t, &cos_t, &sin_t).unwrap();
        assert!(
            max_abs(&vel, &solo) > 1e-5,
            "a conditioning source must couple into the target velocity"
        );

        // The source-id RoPE matters: id 0 on the source segment yields a different target velocity.
        let cos0 = Tensor::cat(&[&cos_s0, &cos_t], 0).unwrap();
        let sin0 = Tensor::cat(&[&sin_s0, &sin_t], 0).unwrap();
        let out0 = dit
            .forward_packed(&tokens, t, &context, &cos0, &sin0)
            .unwrap();
        let vel0 = dit
            .unpatchify_tokens(&out0.narrow(1, total - l_t, l_t).unwrap(), grid_t)
            .unwrap();
        assert!(
            max_abs(&vel, &vel0) > 1e-6,
            "source-id RoPE (id 1 vs id 0) must change the coupled velocity"
        );
    }

    /// The ported sc-6217 query-row chunking (sc-12434): forcing a tiny scores budget must split the
    /// query rows yet reproduce the single un-chunked pass — byte-for-byte (exact `0.0`), since each
    /// query row's softmax is independent. This is the guarantee that stops the A14B self-attention
    /// from materializing the whole `[B,H,S,S]` block. Counting softmax invocations **through the
    /// production `sdpa_budgeted`** proves the render's own path chunks (one call per query block), so
    /// a regression back to a single materialized pass fails here, not just a silently-slower one.
    #[test]
    fn sdpa_chunks_query_rows_and_matches_single_pass() {
        use std::cell::Cell;
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let dtype = q.dtype();

        // Production default budget (ATTN_SCORES_BUDGET ≫ this size) is a single un-chunked pass.
        let single = sdpa(&q, &k, &v, scale).unwrap();

        // Drive the PRODUCTION `sdpa_budgeted` with a call-counting wrapper of the exact f32-upcast
        // softmax and tiny budgets. budget 42 → block = 42/(b·h·sk) = 42/14 = 3 → blocks 3,3,1 over
        // S=7 (3 calls); budget 1 → 7 single-row blocks (7 more calls). A regression that stopped
        // chunking would report 1 and fail. Each chunked result is byte-identical to the single pass.
        let calls = Cell::new(0usize);
        let counting = |scores: &Tensor| {
            calls.set(calls.get() + 1);
            softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)
        };
        let chunked = sdpa_budgeted(&q, &k, &v, scale, 42, counting).unwrap();
        assert_eq!(
            calls.get(),
            3,
            "budget 42 must split S=7 into 3 query-row blocks (3,3,1)"
        );
        assert_eq!(
            max_abs(&single, &chunked),
            0.0,
            "chunked attention must be byte-identical to the single pass"
        );
        let block1 = sdpa_budgeted(&q, &k, &v, scale, 1, counting).unwrap();
        assert_eq!(calls.get(), 10, "budget 1 adds 7 single-row blocks (3 + 7)");
        assert_eq!(
            max_abs(&single, &block1),
            0.0,
            "single-row chunks must be byte-identical to the single pass"
        );

        // The production budget genuinely engages at the story's 832x480 A14B proof geometry
        // (h = 40, S ≈ 32,760, b = 1 under Lightning CFG-off): one query row's score contribution
        // already exceeds the budget, so `sdpa`'s fixed ATTN_SCORES_BUDGET forces chunking there, and
        // each resulting block stays under candle's CUDA i32 element ceiling.
        let rows_per_query = 40 * 32_760usize;
        assert!(
            rows_per_query * 32_760 > candle_gen::ATTN_SCORES_BUDGET,
            "A14B 480p self-attention must be over budget (chunking engages)"
        );
        let block = candle_gen::ATTN_SCORES_BUDGET / rows_per_query;
        assert!(
            rows_per_query * block <= i32::MAX as usize,
            "each chunk's score block must stay under the CUDA i32 element ceiling"
        );
    }
}
