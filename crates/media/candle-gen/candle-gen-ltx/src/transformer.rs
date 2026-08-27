//! LTX-2.3 **AudioVideo DiT** (`AVTransformer3DModel`) — port of mlx-gen-ltx `transformer.rs`
//! (`AvDiT`). Each stream: patchify_proj (128→4096) → adaLN-single (timestep→9·dim) + prompt-adaLN
//! (→2·dim) → 48 gated blocks → affine-false LayerNorm output head + 2-row scale-shift → proj_out
//! (→128) velocity.
//!
//! Per-block (gated 9-row `scale_shift_table` + adaLN-single timestep; rows `shift,scale,gate` ×
//! {MSA 0:3, FF 3:6, text-cross-attn 6:9}): MSA self-attn (q/k RMSNorm over full inner, split 3-D
//! RoPE, **2·sigmoid** per-head gate) → prompt-modulated text cross-attn (no RoPE) → tanh-gelu FFN,
//! each adaLN-modulated (`x·(1+scale)+shift`) and gated (`x + out·gate`). Our checkpoint is dense
//! bf16; the whole forward runs bf16, with attention/norms/layernorm computed in f32 for fidelity.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_gen::candle_core::{DType, Device, DeviceLocation, Result, Tensor, TensorId, D};
use candle_gen::candle_nn::{ops::rms_norm, ops::sigmoid, ops::softmax_last_dim, VarBuilder};

use crate::config::AvConfig;
use crate::quant::{qlinear, QLinear};
use crate::rope::{apply_split_rope, precompute_split_freqs_nd, time_axis};

static NEXT_AVDIT_LOAD_ID: AtomicU64 = AtomicU64::new(1);

fn next_avdit_load_id() -> u64 {
    NEXT_AVDIT_LOAD_ID.fetch_add(1, Ordering::Relaxed)
}

/// Packed-detecting biased Linear (sc-9417): loads the MLX-packed AvDiT projection triple when a
/// `{key}.scales` sibling is present (attn `to_{q,k,v,out}` + `ff.proj_in/out` are packed in the
/// `SceneWorks/ltx-2.3-mlx` q4/q8 tiers), else the dense bf16 weight [+ bias] unchanged. Every AvDiT
/// projection carries a bias in the checkpoint.
fn linear(vb: &VarBuilder, key: &str) -> Result<QLinear> {
    qlinear(vb, key, true)
}

/// `x·(1+scale)+shift`; scale/shift `[B,1,inner]` broadcast over the token axis.
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `x + out·gate`; gate `[B,1,inner]` broadcasts over `out [B,S,inner]`.
fn gated(x: &Tensor, out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x + out.broadcast_mul(gate)?
}

/// Add the learned keyframe absolute-position marker to single-pixel generated-keyframe tokens
/// (sc-18758; reference `apply_keyframes_absolute_embedding`, ported from mlx-gen-ltx's twin). Applied
/// to the patchified video hidden states immediately after `patchify_proj`. `embedding` is the
/// stream's `(1, inner)` `keyframes_abs_pos_embedding` (`None` for a model built without
/// `use_keyframes_abs_pos_embedding`, or for the audio stream, which never carries one);
/// `keyframes_mask` is `(B, T, 1)`, `> 0` marking a keyframe token (`None` = no token marked). Either
/// `None` makes this an exact no-op. The DFR token loops (sc-18789, [`crate::dfr`] / the
/// conditioned forwards) thread a real mask marking generated-keyframe slot tokens; paths with no
/// slots pass `None`.
fn apply_keyframes_embedding(
    x: &Tensor,
    embedding: Option<&Tensor>,
    keyframes_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (Some(embedding), Some(mask)) = (embedding, keyframes_mask) else {
        return Ok(x.clone());
    };
    let gate = mask.gt(0f32)?.to_dtype(x.dtype())?;
    let marker = gate.broadcast_mul(&embedding.to_dtype(x.dtype())?)?;
    x + marker
}

/// Weightless RMSNorm (unit weight) over the last axis, in f32.
fn rms_noweight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    xf.broadcast_mul(&inv)?.to_dtype(x.dtype())
}

/// PixArt sinusoidal timestep embedding (flip_sin_to_cos, cos first), `[N,256]` f32. `ts` is `[N]`
/// f32 (already × timestep_scale_multiplier).
fn timestep_embedding(ts: &Tensor, device: &Device) -> Result<Tensor> {
    const TIME_PROJ_DIM: usize = 256;
    let half = TIME_PROJ_DIM / 2;
    let neg_ln = -(10000f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f64 / half as f64).exp() as f32)
        .collect();
    let n = ts.dim(0)?;
    let freq = Tensor::from_vec(freqs, (1, half), device)?;
    let emb = ts.reshape((n, 1))?.broadcast_mul(&freq)?; // (N, half)
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1) // (N, 256)
}

/// `table[row] + ts4[:,:,row,:]` for `row in [lo,hi)`; each result `[B,1,inner]`.
fn ada_values(table: &Tensor, ts_emb: &Tensor, lo: usize, hi: usize) -> Result<Vec<Tensor>> {
    let (num, inner) = table.dims2()?;
    let (b, s, _) = ts_emb.dims3()?;
    let ts4 = ts_emb.reshape((b, s, num, inner))?;
    let mut out = Vec::with_capacity(hi - lo);
    for row in lo..hi {
        let trow = table.narrow(0, row, 1)?.reshape((1, 1, inner))?;
        let tsrow = ts4.narrow(2, row, 1)?.squeeze(2)?; // (b,s,inner)
        out.push(trow.broadcast_add(&tsrow)?);
    }
    Ok(out)
}

struct Attention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: QLinear,
    heads: usize,
    dim_head: usize,
    eps: f64,
}

impl Attention {
    /// Load with explicit head dims — the cross-modal + audio attns run at the audio inner dim
    /// (heads 32 × head_dim 64 = 2048), and the q/kv input dims ride on the loaded weight shapes.
    fn load_with_dims(vb: VarBuilder, heads: usize, dim_head: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            to_q: linear(&vb, "to_q")?,
            to_k: linear(&vb, "to_k")?,
            to_v: linear(&vb, "to_v")?,
            to_out: linear(&vb, "to_out.0")?,
            q_norm: vb.get_unchecked("q_norm.weight")?.to_dtype(vb.dtype())?,
            k_norm: vb.get_unchecked("k_norm.weight")?.to_dtype(vb.dtype())?,
            gate: linear(&vb, "to_gate_logits")?,
            heads,
            dim_head,
            eps,
        })
    }

    fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        for linear in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.gate,
        ] {
            let path = linear.path().to_string();
            f(&path, linear)?;
        }
        Ok(())
    }

    fn set_adapter_pass(&self, pass: usize) {
        for linear in [&self.to_q, &self.to_k, &self.to_v, &self.to_out, &self.gate] {
            linear.set_additive_pass(pass);
        }
    }

    fn to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)
    }

    /// `rope` rotates the query (and the key when `k_rope` is `None`); `k_rope` rotates the key
    /// separately (cross-modal: video-positioned q, audio-positioned k, or vice-versa). `rope ==
    /// None` ⇒ no RoPE on either (text cross-attention). Self-attn when `context` is `None`.
    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        k_rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ctx = context.unwrap_or(x);
        // q/k RMSNorm over the full inner dim (pre-head), then head reshape.
        let q = rms_norm(
            &self.to_q.forward(x)?.contiguous()?,
            &self.q_norm,
            self.eps as f32,
        )?;
        let k = rms_norm(
            &self.to_k.forward(ctx)?.contiguous()?,
            &self.k_norm,
            self.eps as f32,
        )?;
        let v = self.to_v.forward(ctx)?;
        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = rope {
            qh = apply_split_rope(&qh, cos, sin)?;
            let (kc, ks) = k_rope.unwrap_or((cos, sin));
            kh = apply_split_rope(&kh, kc, ks)?;
        }
        // Attention in f32. i32-overflow guard (sc-9116): the video-DiT self-attn scores `[b,h,s,s]`
        // reach `i32::MAX` at max_size 1280 / long clips (49 frames → 40·40·7 = 11200 tokens →
        // `32·11200² ≈ 4.0e9 > i32::MAX`, growing with clip length), silently corrupting the tail rows
        // on the candle CUDA kernels. The shared budgeted helper chunks over the query rows
        // (byte-identical for common sizes; cross-attn to the fixed text context is a single un-chunked
        // pass). Softmax closure preserves the exact fused `softmax_last_dim`.
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let qf = qh.to_dtype(DType::F32)?.contiguous()?;
        let kf = kh.to_dtype(DType::F32)?.contiguous()?;
        let vf = vh.to_dtype(DType::F32)?.contiguous()?;
        let out = candle_gen::sdpa_budgeted_bhsd(
            &qf,
            &kf,
            &vf,
            scale,
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // (b,h,s,d)
        let (b, s, _) = x.dims3()?;
        let inner = self.heads * self.dim_head;
        let mut out = out
            .transpose(1, 2)?
            .reshape((b, s, inner))?
            .to_dtype(x.dtype())?;
        // Per-head gate: 2·sigmoid(logits) (zero-init → identity).
        let logits = self.gate.forward(x)?;
        let gates = (sigmoid(&logits)? * 2.0)?.reshape((b, s, self.heads, 1))?;
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
    /// `bias` is the caller's `AvConfig::video.ff_bias` / `audio_ff_bias` (sc-18758) — `true` keeps
    /// both Linears biased (byte-identical to pre-sc-18758; this is 2.3 for both fields, and 2.5 for
    /// `audio_ff_bias` — the real 2.5 header carries no `audio_ff_bias` key, so it stays the
    /// reference absent-key default `True`). `false` (2.5's **video** `ff_bias` only) means neither
    /// `net.0.proj` nor `net.2` carries a bias; reference `FeedForward.__init__` threads a single
    /// `bias` flag to both.
    fn load(vb: VarBuilder, bias: bool) -> Result<Self> {
        Ok(Self {
            proj_in: qlinear(&vb.pp("net.0"), "proj", bias)?,
            proj_out: qlinear(&vb.pp("net"), "2", bias)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // tanh-approx gelu.
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }

    fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        for linear in [&mut self.proj_in, &mut self.proj_out] {
            let path = linear.path().to_string();
            f(&path, linear)?;
        }
        Ok(())
    }

    fn set_adapter_pass(&self, pass: usize) {
        self.proj_in.set_additive_pass(pass);
        self.proj_out.set_additive_pass(pass);
    }
}

struct AdaLayerNormSingle {
    ts_lin1: QLinear,
    ts_lin2: QLinear,
    linear: QLinear,
    dtype: DType,
}

impl AdaLayerNormSingle {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ts_lin1: linear(&vb.pp("emb.timestep_embedder"), "linear_1")?,
            ts_lin2: linear(&vb.pp("emb.timestep_embedder"), "linear_2")?,
            linear: linear(&vb, "linear")?,
            dtype: vb.dtype(),
        })
    }

    /// `ts_flat` is `[N]` f32 (already scaled). Returns `(scale_shift [N, coeff·inner], embedded
    /// [N, inner])`, bf16.
    fn forward(&self, ts_flat: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
        let proj = timestep_embedding(ts_flat, device)?.to_dtype(self.dtype)?;
        let h = self.ts_lin1.forward(&proj)?.silu()?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&embedded.silu()?)?;
        Ok((scale_shift, embedded))
    }

    fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        for linear in [&mut self.ts_lin1, &mut self.ts_lin2, &mut self.linear] {
            let path = linear.path().to_string();
            f(&path, linear)?;
        }
        Ok(())
    }

    fn set_adapter_pass(&self, pass: usize) {
        self.ts_lin1.set_additive_pass(pass);
        self.ts_lin2.set_additive_pass(pass);
        self.linear.set_additive_pass(pass);
    }
}

/// Affine-false LayerNorm over the last axis (computed in f32, cast back).
fn layer_norm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(x.dtype())
}

// =================================================================================================
// AvDiT — the dual-modal AudioVideo DiT (sc-5495). The video stack ([`AvStream`] + gated video
// attns) + an audio stack at the audio inner dim (2048) + bidirectional cross-modal attention. Per
// block: video self+text-CA → audio self+text-CA → cross-modal (a2v updates video, v2a updates audio)
// → video FF → audio FF. Predicts `(video_velocity, audio_velocity)`. Mirrors mlx-gen-ltx `AvDiT`.
// =================================================================================================

/// Precomputed per-stream adaLN timestep tensors (`Stream::prepare`).
struct AvTs {
    ts_emb: Tensor,        // (b,1,9·inner)
    emb_ts: Tensor,        // (b,1,inner)
    prompt_ts: Tensor,     // (b,1,2·inner)
    cross_ss_ts: Tensor,   // (b,1,4·inner)
    cross_gate_ts: Tensor, // (b,1,inner)
}

/// One modality's non-block modules + dims (the video or audio half of the AV DiT).
struct AvStream {
    patchify: QLinear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    cross_ss_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    scale_shift_table: Tensor, // (2, inner) bf16
    proj_out: QLinear,
    /// `(1, inner)` keyframe absolute-position marker (sc-18758) — the **video** stream only (the
    /// reference `_init_video` never builds this for audio), `Some` only when
    /// `cfg.use_keyframes_abs_pos_embedding`.
    keyframes_embedding: Option<Tensor>,
    inner: usize,
    coeff: usize, // adaLN row count (9 gated)
    eps: f64,
    dtype: DType,
}

impl AvStream {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &VarBuilder,
        patchify: &str,
        adaln: &str,
        prompt: &str,
        cross_ss: &str,
        cross_gate: &str,
        sst: &str,
        proj_out: &str,
        inner: usize,
        eps: f64,
        keyframes_embedding_key: Option<&str>,
    ) -> Result<Self> {
        // sc-18758: `require`d (not an existence probe) so a config that sets the flag but a
        // checkpoint that omits the tensor is a load error, not a silent None.
        let keyframes_embedding = keyframes_embedding_key
            .map(|k| -> Result<Tensor> { vb.get_unchecked(k)?.to_dtype(vb.dtype()) })
            .transpose()?;
        Ok(Self {
            patchify: linear(vb, patchify)?,
            adaln: AdaLayerNormSingle::load(vb.pp(adaln))?,
            prompt_adaln: AdaLayerNormSingle::load(vb.pp(prompt))?,
            cross_ss_adaln: AdaLayerNormSingle::load(vb.pp(cross_ss))?,
            cross_gate_adaln: AdaLayerNormSingle::load(vb.pp(cross_gate))?,
            scale_shift_table: vb.get_unchecked(sst)?.to_dtype(vb.dtype())?,
            proj_out: linear(vb, proj_out)?,
            keyframes_embedding,
            inner,
            coeff: 9,
            eps,
            dtype: vb.dtype(),
        })
    }

    fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        for linear in [&mut self.patchify, &mut self.proj_out] {
            let path = linear.path().to_string();
            f(&path, linear)?;
        }
        self.adaln.visit_adaptable_mut(f)?;
        self.prompt_adaln.visit_adaptable_mut(f)?;
        self.cross_ss_adaln.visit_adaptable_mut(f)?;
        self.cross_gate_adaln.visit_adaptable_mut(f)
    }

    fn set_adapter_pass(&self, pass: usize) {
        self.patchify.set_additive_pass(pass);
        self.proj_out.set_additive_pass(pass);
        self.adaln.set_adapter_pass(pass);
        self.prompt_adaln.set_adapter_pass(pass);
        self.cross_ss_adaln.set_adapter_pass(pass);
        self.cross_gate_adaln.set_adapter_pass(pass);
    }

    /// adaLN-single + prompt/cross controllers for a uniform T2V timestep `sigma`.
    fn ts_embeds(&self, sigma: f64, ts_mult: f64, b: usize, device: &Device) -> Result<AvTs> {
        let ts_scaled = (sigma * ts_mult) as f32;
        let ts_flat = Tensor::from_vec(vec![ts_scaled; b], (b,), device)?;
        self.ts_embeds_flat(&ts_flat, &ts_flat, b, 1, device)
    }

    /// adaLN-single controllers for a per-token video timestep table `[B, S]`.
    ///
    /// LTX image/keyframe and IC-LoRA conditioning feeds `sigma * denoise_mask` per token. Keeping
    /// that table here (instead of collapsing it to a scalar) is load-bearing: fully pinned tokens
    /// receive timestep zero while generation tokens receive the current schedule sigma.
    fn ts_embeds_tokens(&self, timesteps: &Tensor, ts_mult: f64, device: &Device) -> Result<AvTs> {
        let (b, s) = timesteps.dims2()?;
        let ts_flat = (timesteps.to_dtype(DType::F32)? * ts_mult)?
            .flatten_all()?
            .contiguous()?;
        // Prompt adaLN uses one shared modulation per sample (`timestep[:, :1]`), not one per latent
        // token; otherwise its sequence length would not broadcast over the text context.
        let prompt_flat = (timesteps.narrow(1, 0, 1)?.to_dtype(DType::F32)? * ts_mult)?
            .flatten_all()?
            .contiguous()?;
        self.ts_embeds_flat(&ts_flat, &prompt_flat, b, s, device)
    }

    fn ts_embeds_flat(
        &self,
        ts_flat: &Tensor,
        prompt_flat: &Tensor,
        b: usize,
        s: usize,
        device: &Device,
    ) -> Result<AvTs> {
        let inner = self.inner;
        let (ss, emb) = self.adaln.forward(ts_flat, device)?;
        let (pss, _) = self.prompt_adaln.forward(prompt_flat, device)?;
        let (css, _) = self.cross_ss_adaln.forward(ts_flat, device)?;
        let (cgs, _) = self.cross_gate_adaln.forward(ts_flat, device)?;
        Ok(AvTs {
            ts_emb: ss.reshape((b, s, self.coeff * inner))?,
            emb_ts: emb.reshape((b, s, inner))?,
            prompt_ts: pss.reshape((b, 1, 2 * inner))?,
            cross_ss_ts: css.reshape((b, s, 4 * inner))?,
            cross_gate_ts: cgs.reshape((b, s, inner))?,
        })
    }

    fn output_head(&self, h: &Tensor, emb_ts: &Tensor) -> Result<Tensor> {
        let (scale, shift) = output_scale_shift(&self.scale_shift_table, emb_ts, self.inner)?;
        let normed = layer_norm_noaffine(h, self.eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }
}

/// Broadcast the output adaLN table across the full token-wise timestep embedding `[B,S,D]`.
/// Uniform T2V has `S=1`; conditioned I2V/keyframe/clip paths carry one embedding per video token.
fn output_scale_shift(table: &Tensor, emb_ts: &Tensor, inner: usize) -> Result<(Tensor, Tensor)> {
    let (b, s, d) = emb_ts.dims3()?;
    if d != inner {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "LTX output timestep width {d} does not match stream width {inner}"
        )));
    }
    let table = table.reshape((1, 1, 2, inner))?;
    let ss = table.broadcast_add(&emb_ts.reshape((b, s, 1, inner))?)?;
    let shift = ss.narrow(2, 0, 1)?.squeeze(2)?;
    let scale = ss.narrow(2, 1, 1)?.squeeze(2)?;
    Ok((scale, shift))
}

/// Borrowed per-stream args threaded into an [`AvBlock`].
struct AvStreamArgs<'a> {
    ts_emb: &'a Tensor,
    prompt_ts: &'a Tensor,
    context: &'a Tensor,
    cos: &'a Tensor,
    sin: &'a Tensor,
    cross_cos: &'a Tensor,
    cross_sin: &'a Tensor,
    cross_ss_ts: &'a Tensor,
    cross_gate_ts: &'a Tensor,
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn conditioned_output_head_preserves_every_token_timestep() -> Result<()> {
        let device = Device::Cpu;
        let table = Tensor::zeros((2, 4), DType::F32, &device)?;
        let emb = Tensor::arange(0f32, 24f32, &device)?.reshape((2, 3, 4))?;
        let (scale, shift) = output_scale_shift(&table, &emb, 4)?;
        assert_eq!(scale.dims(), &[2, 3, 4]);
        assert_eq!(shift.dims(), &[2, 3, 4]);
        assert_eq!(
            scale.flatten_all()?.to_vec1::<f32>()?,
            emb.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            shift.flatten_all()?.to_vec1::<f32>()?,
            emb.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn apply_keyframes_embedding_is_a_no_op_without_embedding_or_mask() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device)?;
        let embedding = Tensor::from_vec(vec![10.0f32, 20.0], (1, 2), &device)?;
        let mask = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2, 1), &device)?;

        // No embedding configured (a pre-sc-18758 / 2.3 model, or the audio stream) → passthrough.
        let got = apply_keyframes_embedding(&x, None, Some(&mask))?;
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );

        // Embedding configured but no mask supplied (a path with no generated slots) → passthrough.
        let got2 = apply_keyframes_embedding(&x, Some(&embedding), None)?;
        assert_eq!(
            got2.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn apply_keyframes_embedding_marks_only_gated_tokens() -> Result<()> {
        let device = Device::Cpu;
        // (1, 2, 2): token 0 marked (mask > 0), token 1 not.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device)?;
        let embedding = Tensor::from_vec(vec![10.0f32, 20.0], (1, 2), &device)?;
        let mask = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2, 1), &device)?;

        let got = apply_keyframes_embedding(&x, Some(&embedding), Some(&mask))?;
        let out = got.flatten_all()?.to_vec1::<f32>()?;
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 22.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn prepared_av_request_reuses_once_and_rejects_stale_identity() -> Result<()> {
        let device = Device::Cpu;
        let video = Tensor::zeros((1, 2, 128), DType::F32, &device)?;
        let audio = Tensor::zeros((1, 3, 128), DType::F32, &device)?;
        let video_grid = Tensor::zeros((1, 3, 2, 2), DType::F32, &device)?;
        let audio_grid = Tensor::zeros((1, 1, 3, 2), DType::F32, &device)?;
        let prepared = PreparedAvRequest::new(17, &video, &audio, &video_grid, &audio_grid)?;

        // One preparation remains valid for every same-spec denoise latent in the request.
        for _ in 0..2 {
            let next_video = Tensor::ones((1, 2, 128), DType::F32, &device)?;
            let next_audio = Tensor::ones((1, 3, 128), DType::F32, &device)?;
            prepared.validate(17, &next_video, &next_audio, &video_grid, &audio_grid)?;
        }

        // A same-shape replacement grid can carry different positions and must be rebuilt.
        let changed_grid = Tensor::ones((1, 3, 2, 2), DType::F32, &device)?;
        assert!(prepared
            .validate(17, &video, &audio, &changed_grid, &audio_grid)
            .is_err());
        let rebuilt = PreparedAvRequest::new(17, &video, &audio, &changed_grid, &audio_grid)?;
        rebuilt.validate(17, &video, &audio, &changed_grid, &audio_grid)?;

        let wrong_geometry = Tensor::zeros((1, 4, 128), DType::F32, &device)?;
        assert!(prepared
            .validate(17, &video, &wrong_geometry, &video_grid, &audio_grid)
            .is_err());
        let wrong_dtype = video.to_dtype(DType::F16)?;
        assert!(prepared
            .validate(17, &wrong_dtype, &audio, &video_grid, &audio_grid)
            .is_err());
        let mut wrong_device = prepared.clone();
        wrong_device.video_latent.device = DeviceLocation::Metal { gpu_id: 7 };
        assert!(wrong_device
            .validate(17, &video, &audio, &video_grid, &audio_grid)
            .is_err());
        assert!(prepared
            .validate(18, &video, &audio, &video_grid, &audio_grid)
            .is_err());
        Ok(())
    }
}

/// `4·scale-shift + 1·gate` cross-modal adaLN values from the pre-split tables → `(scale_a2v,
/// shift_a2v, scale_v2a, shift_v2a, gate)`.
fn av_ca_ada(
    ss_table: &Tensor,
    gate_table: &Tensor,
    ss_ts: &Tensor,
    gate_ts: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let ss = ada_values(ss_table, ss_ts, 0, 4)?;
    let g = ada_values(gate_table, gate_ts, 0, 1)?;
    Ok((
        ss[0].clone(),
        ss[1].clone(),
        ss[2].clone(),
        ss[3].clone(),
        g[0].clone(),
    ))
}

/// One AudioVideo transformer block (`BasicAVTransformerBlock`).
struct AvBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    v_sst: Tensor, // (9, 4096)
    v_pst: Tensor, // (2, 4096)
    a_attn1: Attention,
    a_attn2: Attention,
    a_ff: FeedForward,
    a_sst: Tensor, // (9, 2048)
    a_pst: Tensor, // (2, 2048)
    a2v: Attention,
    v2a: Attention,
    ca_audio_ss: Tensor,   // (4, 2048)
    ca_audio_gate: Tensor, // (1, 2048)
    ca_video_ss: Tensor,   // (4, 4096)
    ca_video_gate: Tensor, // (1, 4096)
    eps: f64,
}

impl AvBlock {
    fn load(vb: VarBuilder, cfg: &AvConfig) -> Result<Self> {
        let eps = cfg.video.norm_eps;
        let (vh, vdh) = (cfg.video.num_heads, cfg.video.head_dim);
        let (ah, adh) = (cfg.audio_heads, cfg.audio_head_dim);
        let bf = |k: &str| -> Result<Tensor> { vb.get_unchecked(k)?.to_dtype(vb.dtype()) };
        // Split a (5, dim) cross table → 4-row scale-shift + 1-row gate.
        let split = |key: &str| -> Result<(Tensor, Tensor)> {
            let t = bf(key)?;
            Ok((t.narrow(0, 0, 4)?, t.narrow(0, 4, 1)?))
        };
        let (ca_audio_ss, ca_audio_gate) = split("scale_shift_table_a2v_ca_audio")?;
        let (ca_video_ss, ca_video_gate) = split("scale_shift_table_a2v_ca_video")?;
        Ok(Self {
            attn1: Attention::load_with_dims(vb.pp("attn1"), vh, vdh, eps)?,
            attn2: Attention::load_with_dims(vb.pp("attn2"), vh, vdh, eps)?,
            ff: FeedForward::load(vb.pp("ff"), cfg.video.ff_bias)?,
            v_sst: bf("scale_shift_table")?,
            v_pst: bf("prompt_scale_shift_table")?,
            a_attn1: Attention::load_with_dims(vb.pp("audio_attn1"), ah, adh, eps)?,
            a_attn2: Attention::load_with_dims(vb.pp("audio_attn2"), ah, adh, eps)?,
            a_ff: FeedForward::load(vb.pp("audio_ff"), cfg.audio_ff_bias)?,
            a_sst: bf("audio_scale_shift_table")?,
            a_pst: bf("audio_prompt_scale_shift_table")?,
            a2v: Attention::load_with_dims(vb.pp("audio_to_video_attn"), ah, adh, eps)?,
            v2a: Attention::load_with_dims(vb.pp("video_to_audio_attn"), ah, adh, eps)?,
            ca_audio_ss,
            ca_audio_gate,
            ca_video_ss,
            ca_video_gate,
            eps,
        })
    }

    /// Visit the complete LTX-2.3 AudioVideo block adapter surface. The official Eros distill LoRA
    /// includes feed-forward, gated attention, audio, and cross-modal projections in addition to
    /// the video attention leaves emitted by SceneWorks' native trainer.
    fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        self.attn1.visit_adaptable_mut(f)?;
        self.attn2.visit_adaptable_mut(f)?;
        self.ff.visit_adaptable_mut(f)?;
        self.a_attn1.visit_adaptable_mut(f)?;
        self.a_attn2.visit_adaptable_mut(f)?;
        self.a_ff.visit_adaptable_mut(f)?;
        self.a2v.visit_adaptable_mut(f)?;
        self.v2a.visit_adaptable_mut(f)
    }

    fn set_adapter_pass(&self, pass: usize) {
        self.attn1.set_adapter_pass(pass);
        self.attn2.set_adapter_pass(pass);
        self.ff.set_adapter_pass(pass);
        self.a_attn1.set_adapter_pass(pass);
        self.a_attn2.set_adapter_pass(pass);
        self.a_ff.set_adapter_pass(pass);
        self.a2v.set_adapter_pass(pass);
        self.v2a.set_adapter_pass(pass);
    }

    /// Self-attn (RoPE) → prompt-modulated text cross-attention (no RoPE), for one modality.
    fn self_and_text(
        &self,
        x: &Tensor,
        attn1: &Attention,
        attn2: &Attention,
        sst: &Tensor,
        pst: &Tensor,
        a: &AvStreamArgs,
    ) -> Result<Tensor> {
        let msa = ada_values(sst, a.ts_emb, 0, 3)?;
        let norm = modulate(&rms_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = attn1.forward(&norm, None, Some((a.cos, a.sin)), None)?;
        let x = gated(x, &attn, &msa[2])?;

        let p = ada_values(pst, a.prompt_ts, 0, 2)?;
        let context = modulate(a.context, &p[1], &p[0])?;

        let ca = ada_values(sst, a.ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = attn2.forward(&norm_ca, Some(&context), None, None)?;
        gated(&x, &cross, &ca[2])
    }

    fn feed_forward(
        &self,
        x: &Tensor,
        ff: &FeedForward,
        sst: &Tensor,
        ts_emb: &Tensor,
    ) -> Result<Tensor> {
        let mlp = ada_values(sst, ts_emb, 3, 6)?;
        let norm = modulate(&rms_noweight(x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff_out = ff.forward(&norm)?;
        gated(x, &ff_out, &mlp[2])
    }

    /// Joint forward `(vx, ax)` → `(vx, ax)`.
    fn forward(
        &self,
        vx: &Tensor,
        ax: &Tensor,
        v: &AvStreamArgs,
        a: &AvStreamArgs,
    ) -> Result<(Tensor, Tensor)> {
        let mut vx =
            self.self_and_text(vx, &self.attn1, &self.attn2, &self.v_sst, &self.v_pst, v)?;
        let mut ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
        )?;

        // Cross-modal — both directions read the pre-update rms_norm snapshots.
        let vx_n3 = rms_noweight(&vx, self.eps)?;
        let ax_n3 = rms_noweight(&ax, self.eps)?;
        let (sca_a2v, sha_a2v, sca_v2a, sha_v2a, gate_v2a) = av_ca_ada(
            &self.ca_audio_ss,
            &self.ca_audio_gate,
            a.cross_ss_ts,
            a.cross_gate_ts,
        )?;
        let (scv_a2v, shv_a2v, scv_v2a, shv_v2a, gate_a2v) = av_ca_ada(
            &self.ca_video_ss,
            &self.ca_video_gate,
            v.cross_ss_ts,
            v.cross_gate_ts,
        )?;

        // Audio-to-Video: Q from video (video cross-PE), K/V from audio (audio cross-PE).
        let a2v = self.a2v.forward(
            &modulate(&vx_n3, &scv_a2v, &shv_a2v)?,
            Some(&modulate(&ax_n3, &sca_a2v, &sha_a2v)?),
            Some((v.cross_cos, v.cross_sin)),
            Some((a.cross_cos, a.cross_sin)),
        )?;
        vx = gated(&vx, &a2v, &gate_a2v)?;

        // Video-to-Audio: Q from audio (audio cross-PE), K/V from video (video cross-PE).
        let v2a = self.v2a.forward(
            &modulate(&ax_n3, &sca_v2a, &sha_v2a)?,
            Some(&modulate(&vx_n3, &scv_v2a, &shv_v2a)?),
            Some((a.cross_cos, a.cross_sin)),
            Some((v.cross_cos, v.cross_sin)),
        )?;
        ax = gated(&ax, &v2a, &gate_v2a)?;

        vx = self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)?;
        ax = self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)?;
        Ok((vx, ax))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorSpec {
    shape: Vec<usize>,
    dtype: DType,
    device: DeviceLocation,
}

impl TensorSpec {
    fn new(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: tensor.device().location(),
        }
    }

    fn matches(&self, tensor: &Tensor) -> bool {
        tensor.dims() == self.shape
            && tensor.dtype() == self.dtype
            && tensor.device().location() == self.device
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorIdentity {
    id: TensorId,
    spec: TensorSpec,
}

impl TensorIdentity {
    fn new(tensor: &Tensor) -> Self {
        Self {
            id: tensor.id(),
            spec: TensorSpec::new(tensor),
        }
    }

    fn matches(&self, tensor: &Tensor) -> bool {
        tensor.id() == self.id && self.spec.matches(tensor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedAvRequest {
    model_load_id: u64,
    video_latent: TensorSpec,
    audio_latent: TensorSpec,
    video_grid: TensorIdentity,
    audio_grid: TensorIdentity,
}

impl PreparedAvRequest {
    fn new(
        model_load_id: u64,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<Self> {
        let (_, video_tokens, _) = video_latent.dims3()?;
        let (_, audio_tokens, _) = audio_latent.dims3()?;
        let (_, _, video_grid_tokens, _) = video_grid.dims4()?;
        let (_, _, audio_grid_tokens, _) = audio_grid.dims4()?;
        if video_tokens != video_grid_tokens || audio_tokens != audio_grid_tokens {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx: latent token counts ({video_tokens}, {audio_tokens}) do not match position-grid token counts ({video_grid_tokens}, {audio_grid_tokens})"
            )));
        }
        Ok(Self {
            model_load_id,
            video_latent: TensorSpec::new(video_latent),
            audio_latent: TensorSpec::new(audio_latent),
            video_grid: TensorIdentity::new(video_grid),
            audio_grid: TensorIdentity::new(audio_grid),
        })
    }

    fn validate(
        &self,
        model_load_id: u64,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<()> {
        if model_load_id != self.model_load_id
            || !self.video_latent.matches(video_latent)
            || !self.audio_latent.matches(audio_latent)
            || !self.video_grid.matches(video_grid)
            || !self.audio_grid.matches(audio_grid)
        {
            return Err(candle_gen::candle_core::Error::Msg(
                "ltx: prepared RoPE request identity does not match model, geometry, position grid, dtype, or device".into(),
            ));
        }
        Ok(())
    }
}

/// Request-scoped split-RoPE tables bound to the exact model load, position grids, latent geometry,
/// dtype, and device that created them. The denoise latent value may change while its spec stays fixed.
pub(crate) struct PreparedAvRope {
    request: PreparedAvRequest,
    v_cos: Tensor,
    v_sin: Tensor,
    vc_cos: Tensor,
    vc_sin: Tensor,
    a_cos: Tensor,
    a_sin: Tensor,
    ac_cos: Tensor,
    ac_sin: Tensor,
}

/// The LTX-2.3 **AudioVideo** DiT. Predicts `(video_velocity, audio_velocity)` from the two latent
/// token streams + shared text conditioning.
pub struct AvDiT {
    video: AvStream,
    audio: AvStream,
    blocks: Vec<AvBlock>,
    cfg: AvConfig,
    device: Device,
    load_id: u64,
}

impl AvDiT {
    /// Build from a VarBuilder rooted at `model.diffusion_model.`.
    pub fn new(vb: VarBuilder, cfg: &AvConfig) -> Result<Self> {
        let device = vb.device().clone();
        let video = AvStream::load(
            &vb,
            "patchify_proj",
            "adaln_single",
            "prompt_adaln_single",
            "av_ca_video_scale_shift_adaln_single",
            "av_ca_a2v_gate_adaln_single",
            "scale_shift_table",
            "proj_out",
            cfg.video.inner_dim(),
            cfg.video.norm_eps,
            cfg.video
                .use_keyframes_abs_pos_embedding
                .then_some("keyframes_abs_pos_embedding"),
        )?;
        let audio = AvStream::load(
            &vb,
            "audio_patchify_proj",
            "audio_adaln_single",
            "audio_prompt_adaln_single",
            "av_ca_audio_scale_shift_adaln_single",
            "av_ca_v2a_gate_adaln_single",
            "audio_scale_shift_table",
            "audio_proj_out",
            cfg.audio_inner(),
            cfg.video.norm_eps,
            // The reference never builds a keyframe marker for the audio stream (`_init_video` only).
            None,
        )?;
        let mut blocks = Vec::with_capacity(cfg.video.num_layers);
        for i in 0..cfg.video.num_layers {
            blocks.push(AvBlock::load(
                vb.pp(format!("transformer_blocks.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            video,
            audio,
            blocks,
            cfg: cfg.clone(),
            device,
            load_id: next_avdit_load_id(),
        })
    }

    /// Walk the complete LTX-2.3 AudioVideo inference adapter surface. This is a superset of the
    /// native video-attention training surface and is required by the official Eros distill LoRA.
    pub(crate) fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        self.video.visit_adaptable_mut(f)?;
        self.audio.visit_adaptable_mut(f)?;
        for block in &mut self.blocks {
            block.visit_adaptable_mut(f)?;
        }
        Ok(())
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Select the active distilled denoise pass for every installed adapter residual. This takes
    /// `&self`: projections carry an atomic pass selector so a shared render-time `Arc<AvDiT>` can
    /// switch stages without rebuilding or mutating the frozen model.
    pub(crate) fn set_adapter_pass(&self, pass: usize) {
        self.video.set_adapter_pass(pass);
        self.audio.set_adapter_pass(pass);
        for block in &self.blocks {
            block.set_adapter_pass(pass);
        }
    }

    /// Build the four split-RoPE tables once for a request. This deliberately has no model-owned cache:
    /// callers retain the prepared object for the render, avoiding host key extraction on each forward.
    pub(crate) fn prepare_rope(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<PreparedAvRope> {
        let request = PreparedAvRequest::new(
            self.load_id,
            video_latent,
            audio_latent,
            video_grid,
            audio_grid,
        )?;
        let device = &self.device;
        let theta = self.cfg.video.rope_theta;
        // Self RoPE (video 3-axis @4096, audio 1-axis @2048) + cross RoPE (time axis @2048 both).
        let (v_cos, v_sin) = precompute_split_freqs_nd(
            video_grid,
            self.cfg.video.inner_dim(),
            theta,
            &self.cfg.video.rope_max_pos,
            self.cfg.video.num_heads,
            device,
        )?;
        let (vc_cos, vc_sin) = precompute_split_freqs_nd(
            &time_axis(video_grid)?,
            self.cfg.cross_inner,
            theta,
            &[self.cfg.cross_max_pos],
            self.cfg.video.num_heads,
            device,
        )?;
        let (a_cos, a_sin) = precompute_split_freqs_nd(
            audio_grid,
            self.cfg.audio_inner(),
            theta,
            &[self.cfg.audio_max_pos],
            self.cfg.audio_heads,
            device,
        )?;
        let (ac_cos, ac_sin) = precompute_split_freqs_nd(
            &time_axis(audio_grid)?,
            self.cfg.cross_inner,
            theta,
            &[self.cfg.cross_max_pos],
            self.cfg.audio_heads,
            device,
        )?;
        Ok(PreparedAvRope {
            request,
            v_cos: v_cos.clone(),
            v_sin: v_sin.clone(),
            vc_cos: vc_cos.clone(),
            vc_sin: vc_sin.clone(),
            a_cos: a_cos.clone(),
            a_sin: a_sin.clone(),
            ac_cos: ac_cos.clone(),
            ac_sin: ac_sin.clone(),
        })
    }

    /// Joint velocity forward.
    ///
    /// * `*_latent` — `[B, S, 128]` patchified tokens.
    /// * `sigma` — scalar σ (uniform T2V timestep, shared by both streams).
    /// * `*_context` — text embeddings (video 4096, audio 2048).
    /// * `*_grid` — position grids (video `[1,3,Tv,2]`, audio `[1,1,Ta,2]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        sigma: f64,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let prepared = self.prepare_rope(video_latent, audio_latent, video_grid, audio_grid)?;
        self.forward_prepared(
            video_latent,
            audio_latent,
            sigma,
            video_context,
            audio_context,
            video_grid,
            audio_grid,
            &prepared,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prepared(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        sigma: f64,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
        prepared: &PreparedAvRope,
    ) -> Result<(Tensor, Tensor)> {
        prepared.request.validate(
            self.load_id,
            video_latent,
            audio_latent,
            video_grid,
            audio_grid,
        )?;
        let device = &self.device;
        let b = video_latent.dim(0)?;
        let ts_mult = self.cfg.video.timestep_scale_multiplier;

        let v_ts = self.video.ts_embeds(sigma, ts_mult, b, device)?;
        let a_ts = self.audio.ts_embeds(sigma, ts_mult, b, device)?;

        self.forward_with_ts(
            video_latent,
            audio_latent,
            video_context,
            audio_context,
            None,
            prepared,
            &v_ts,
            &a_ts,
        )
    }

    /// Joint velocity forward with per-token video timesteps `[B, Sv]` and a uniform audio sigma.
    /// Used by every LTX image/keyframe/clip-conditioned lane.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_conditioned(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_timesteps: &Tensor,
        audio_sigma: f64,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
        video_keyframes_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let prepared = self.prepare_rope(video_latent, audio_latent, video_grid, audio_grid)?;
        self.forward_conditioned_prepared(
            video_latent,
            audio_latent,
            video_timesteps,
            audio_sigma,
            video_context,
            audio_context,
            video_grid,
            audio_grid,
            video_keyframes_mask,
            &prepared,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_conditioned_prepared(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_timesteps: &Tensor,
        audio_sigma: f64,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_grid: &Tensor,
        audio_grid: &Tensor,
        video_keyframes_mask: Option<&Tensor>,
        prepared: &PreparedAvRope,
    ) -> Result<(Tensor, Tensor)> {
        prepared.request.validate(
            self.load_id,
            video_latent,
            audio_latent,
            video_grid,
            audio_grid,
        )?;
        let b = video_latent.dim(0)?;
        if video_timesteps.dims2()? != (b, video_latent.dim(1)?) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx: video timestep shape {:?} must be [batch={}, tokens={}]",
                video_timesteps.dims(),
                b,
                video_latent.dim(1)?
            )));
        }
        let ts_mult = self.cfg.video.timestep_scale_multiplier;
        let v_ts = self
            .video
            .ts_embeds_tokens(video_timesteps, ts_mult, &self.device)?;
        let a_ts = self
            .audio
            .ts_embeds(audio_sigma, ts_mult, b, &self.device)?;
        self.forward_with_ts(
            video_latent,
            audio_latent,
            video_context,
            audio_context,
            video_keyframes_mask,
            prepared,
            &v_ts,
            &a_ts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_ts(
        &self,
        video_latent: &Tensor,
        audio_latent: &Tensor,
        video_context: &Tensor,
        audio_context: &Tensor,
        video_keyframes_mask: Option<&Tensor>,
        prepared: &PreparedAvRope,
        v_ts: &AvTs,
        a_ts: &AvTs,
    ) -> Result<(Tensor, Tensor)> {
        let mut vx = self
            .video
            .patchify
            .forward(&video_latent.to_dtype(self.video.dtype)?)?;
        // sc-18758/sc-18789: the DFR keyframe-slot marker (video stream only). The DFR token loops
        // thread a real `(B, S, 1)` mask marking generated-keyframe slot tokens; every other path
        // passes `None`, keeping this an exact no-op.
        vx = apply_keyframes_embedding(
            &vx,
            self.video.keyframes_embedding.as_ref(),
            video_keyframes_mask,
        )?;
        let mut ax = self
            .audio
            .patchify
            .forward(&audio_latent.to_dtype(self.audio.dtype)?)?;
        let v_ctx = video_context.to_dtype(self.video.dtype)?;
        let a_ctx = audio_context.to_dtype(self.audio.dtype)?;

        let va = AvStreamArgs {
            ts_emb: &v_ts.ts_emb,
            prompt_ts: &v_ts.prompt_ts,
            context: &v_ctx,
            cos: &prepared.v_cos,
            sin: &prepared.v_sin,
            cross_cos: &prepared.vc_cos,
            cross_sin: &prepared.vc_sin,
            cross_ss_ts: &v_ts.cross_ss_ts,
            cross_gate_ts: &v_ts.cross_gate_ts,
        };
        let aa = AvStreamArgs {
            ts_emb: &a_ts.ts_emb,
            prompt_ts: &a_ts.prompt_ts,
            context: &a_ctx,
            cos: &prepared.a_cos,
            sin: &prepared.a_sin,
            cross_cos: &prepared.ac_cos,
            cross_sin: &prepared.ac_sin,
            cross_ss_ts: &a_ts.cross_ss_ts,
            cross_gate_ts: &a_ts.cross_gate_ts,
        };

        for block in &self.blocks {
            let (nv, na) = block.forward(&vx, &ax, &va, &aa)?;
            vx = nv;
            ax = na;
        }
        let v_vel = self.video.output_head(&vx, &v_ts.emb_ts)?;
        let a_vel = self.audio.output_head(&ax, &a_ts.emb_ts)?;
        Ok((v_vel, a_vel))
    }

    /// The inference `AvDiT`'s video-only reduction used by LoRA training: video patch/timestep
    /// preparation, then each block's video self-attention, text cross-attention and feed-forward
    /// branches, followed by the video velocity head. Audio and bidirectional cross-modal branches
    /// are deliberately absent, matching the reference trainer's `audio=None` path.
    pub fn forward_video_only(
        &self,
        video_latent: &Tensor,
        sigma: f64,
        video_context: &Tensor,
        video_grid: &Tensor,
    ) -> Result<Tensor> {
        let b = video_latent.dim(0)?;
        let v_ts = self.video.ts_embeds(
            sigma,
            self.cfg.video.timestep_scale_multiplier,
            b,
            &self.device,
        )?;
        let (v_cos, v_sin) = precompute_split_freqs_nd(
            video_grid,
            self.cfg.video.inner_dim(),
            self.cfg.video.rope_theta,
            &self.cfg.video.rope_max_pos,
            self.cfg.video.num_heads,
            &self.device,
        )?;
        let mut vx = self
            .video
            .patchify
            .forward(&video_latent.to_dtype(self.video.dtype)?)?;
        vx = apply_keyframes_embedding(&vx, self.video.keyframes_embedding.as_ref(), None)?;
        let v_ctx = video_context.to_dtype(self.video.dtype)?;
        // The video-only path never consumes the cross-modal fields. Reuse the self-RoPE/timestep
        // tensors to keep this borrowed argument bundle allocation-free.
        let va = AvStreamArgs {
            ts_emb: &v_ts.ts_emb,
            prompt_ts: &v_ts.prompt_ts,
            context: &v_ctx,
            cos: &v_cos,
            sin: &v_sin,
            cross_cos: &v_cos,
            cross_sin: &v_sin,
            cross_ss_ts: &v_ts.cross_ss_ts,
            cross_gate_ts: &v_ts.cross_gate_ts,
        };
        for block in &self.blocks {
            vx = block.self_and_text(
                &vx,
                &block.attn1,
                &block.attn2,
                &block.v_sst,
                &block.v_pst,
                &va,
            )?;
            vx = block.feed_forward(&vx, &block.ff, &block.v_sst, &v_ts.ts_emb)?;
        }
        self.video.output_head(&vx, &v_ts.emb_ts)
    }

    /// **Video-only** forward with per-token video timesteps and the DFR keyframes mask — the
    /// reference `LTXModel` called with `audio=None` on a conditioned token state (the DFR
    /// temporal-round tile denoise): the audio stream is skipped and the cross-modal attentions do
    /// not run; each block is video self-attention + text cross-attention + feed-forward
    /// (sc-18789; the uniform-sigma sibling above serves LoRA training).
    pub fn forward_video_only_conditioned(
        &self,
        video_latent: &Tensor,
        video_timesteps: &Tensor,
        video_context: &Tensor,
        video_grid: &Tensor,
        video_keyframes_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = video_latent.dim(0)?;
        if video_timesteps.dims2()? != (b, video_latent.dim(1)?) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx: video timestep shape {:?} must be [batch={}, tokens={}]",
                video_timesteps.dims(),
                b,
                video_latent.dim(1)?
            )));
        }
        let v_ts = self.video.ts_embeds_tokens(
            video_timesteps,
            self.cfg.video.timestep_scale_multiplier,
            &self.device,
        )?;
        let (v_cos, v_sin) = precompute_split_freqs_nd(
            video_grid,
            self.cfg.video.inner_dim(),
            self.cfg.video.rope_theta,
            &self.cfg.video.rope_max_pos,
            self.cfg.video.num_heads,
            &self.device,
        )?;
        let mut vx = self
            .video
            .patchify
            .forward(&video_latent.to_dtype(self.video.dtype)?)?;
        vx = apply_keyframes_embedding(
            &vx,
            self.video.keyframes_embedding.as_ref(),
            video_keyframes_mask,
        )?;
        let v_ctx = video_context.to_dtype(self.video.dtype)?;
        // The video-only path never consumes the cross-modal fields; reuse the self-RoPE tensors.
        let va = AvStreamArgs {
            ts_emb: &v_ts.ts_emb,
            prompt_ts: &v_ts.prompt_ts,
            context: &v_ctx,
            cos: &v_cos,
            sin: &v_sin,
            cross_cos: &v_cos,
            cross_sin: &v_sin,
            cross_ss_ts: &v_ts.cross_ss_ts,
            cross_gate_ts: &v_ts.cross_gate_ts,
        };
        for block in &self.blocks {
            vx = block.self_and_text(
                &vx,
                &block.attn1,
                &block.attn2,
                &block.v_sst,
                &block.v_pst,
                &va,
            )?;
            vx = block.feed_forward(&vx, &block.ff, &block.v_sst, &v_ts.ts_emb)?;
        }
        self.video.output_head(&vx, &v_ts.emb_ts)
    }
}
