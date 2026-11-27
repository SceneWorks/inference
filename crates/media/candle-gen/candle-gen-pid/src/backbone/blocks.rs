//! Attention + transformer blocks: `RotaryAttention` (pixel stream), `MMDiTJointAttention` +
//! `MMDiTBlockT2I` (dual-stream patch blocks), and `PiTBlock` (per-pixel block). Faithful port of the
//! corresponding classes in `pixeldit_official.py`.
//!
//! candle simplification vs the MLX port: `candle_gen::sdpa_budgeted_bhsd` materializes (query-chunked)
//! scores, so it works at any `head_dim` — the pixel stream's `head_dim = 72` needs no flash-pad hack
//! (the MLX port padded to 80 for MLX's fused kernel). Runs f32 throughout.

use candle_gen::candle_core::{DType, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::modulate::{adaln, AdaLn};
use candle_gen::{Result, Weights};

use super::layers::{FeedForward, Mlp, RMS_EPS};
use super::rope::apply_rope;
use crate::nn::{linear, rms};

/// `rms(x, weight) · (scale + 1) + shift` — [`crate::nn::rms`] (candle `ops::rms_norm` on an f32
/// copy with the f32 weight, cast back) then `norm · (scale + 1) + shift` (mirrors
/// `mlx_gen::nn::modulate`), as the shared fused AdaLN (SW-6129,
/// [`AdaLn::rms`] `.norm_in_f32()`): one launch on CUDA, bit-identical to that chain, which it runs
/// everywhere else. A rank-2 `[B, D]` modulation is the right-aligned `[1, B, D]` broadcast it was.
fn rms_modulate(x: &Tensor, weight: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let w = weight.to_dtype(DType::F32)?;
    let spec = AdaLn::rms(f64::from(RMS_EPS)).with_weight(&w).norm_in_f32();
    Ok(adaln(x, &spec, &rank3(scale)?, &rank3(shift)?)?)
}

/// `t` with leading unit axes up to rank 3 — the same right-aligned broadcast candle applies.
fn rank3(t: &Tensor) -> Result<Tensor> {
    let mut t = t.clone();
    while t.rank() < 3 {
        t = t.unsqueeze(0)?;
    }
    Ok(t)
}

/// `x + gate · y` (mirrors `mlx_gen::nn::gated`). `gate` broadcasts over the token axis.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_add(&y.broadcast_mul(gate)?)?)
}

/// `[B,S,3·H·Dh]` → q,k,v each `[B,S,H,Dh]` via `reshape(B,S,3,H,Dh)` then slice the `3` axis.
fn split_qkv(qkv: &Tensor, heads: usize, head_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, s, _) = qkv.dims3()?;
    let q5 = qkv.reshape((b, s, 3, heads, head_dim))?;
    let take = |i: usize| -> Result<Tensor> {
        Ok(q5
            .narrow(2, i, 1)?
            .contiguous()?
            .reshape((b, s, heads, head_dim))?)
    };
    Ok((take(0)?, take(1)?, take(2)?))
}

/// `[B,H,S,Dh]` → `[B,S,H·Dh]`.
fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))?)
}

/// `[B,S,H,Dh]` → `[B,H,S,Dh]`.
fn to_bhsd(x: &Tensor) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?)
}

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    Ok(candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?)
}

/// `RotaryAttention` — single-stream qk-normed rotary attention (the PiT pixel block's attention).
pub struct RotaryAttention {
    qkv: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl RotaryAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv: linear(w, &format!("{prefix}.qkv"))?,
            q_norm: w.require(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.require(&format!("{prefix}.k_norm.weight"))?,
            proj: linear(w, &format!("{prefix}.proj"))?,
            heads: heads as usize,
            head_dim: (dim / heads) as usize,
        })
    }

    /// `x`: `[B, N, dim]`; `cos`/`sin`: `[N, head_dim/2]`.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (q, k, v) = split_qkv(
            &__model_proof_output!(self.qkv.forward(x))?,
            self.heads,
            self.head_dim,
        )?;
        let q = rms(&q, &self.q_norm, RMS_EPS)?;
        let k = rms(&k, &self.k_norm, RMS_EPS)?;
        let (q, k, v) = (to_bhsd(&q)?, to_bhsd(&k)?, to_bhsd(&v)?);
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let o = sdpa(&q, &k, &v, scale)?;
        Ok(__model_proof_output!(self.proj.forward(&merge_heads(&o)?))?)
    }
}

/// `MMDiTJointAttention` — separate img/txt QKV with per-stream qk-norm, RoPE (2-D on img, 1-D on
/// txt), a single joint SDPA over `[txt, img]`, then per-stream output projections.
pub struct MMDiTJointAttention {
    qkv_x: Linear,
    qkv_y: Linear,
    q_norm_x: Tensor,
    k_norm_x: Tensor,
    q_norm_y: Tensor,
    k_norm_y: Tensor,
    proj_x: Linear,
    proj_y: Linear,
    heads: usize,
    head_dim: usize,
}

impl MMDiTJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv_x: linear(w, &format!("{prefix}.qkv_x"))?,
            qkv_y: linear(w, &format!("{prefix}.qkv_y"))?,
            q_norm_x: w.require(&format!("{prefix}.q_norm_x.weight"))?,
            k_norm_x: w.require(&format!("{prefix}.k_norm_x.weight"))?,
            q_norm_y: w.require(&format!("{prefix}.q_norm_y.weight"))?,
            k_norm_y: w.require(&format!("{prefix}.k_norm_y.weight"))?,
            proj_x: linear(w, &format!("{prefix}.proj_x"))?,
            proj_y: linear(w, &format!("{prefix}.proj_y"))?,
            heads: heads as usize,
            head_dim: (dim / heads) as usize,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        y: &Tensor,
        cos_img: &Tensor,
        sin_img: &Tensor,
        cos_txt: &Tensor,
        sin_txt: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ny = y.dim(1)?;
        let nx = x.dim(1)?;

        let (qx, kx, vx) = split_qkv(
            &__model_proof_output!(self.qkv_x.forward(x))?,
            self.heads,
            self.head_dim,
        )?;
        let qx = rms(&qx, &self.q_norm_x, RMS_EPS)?;
        let kx = rms(&kx, &self.k_norm_x, RMS_EPS)?;
        let (qy, ky, vy) = split_qkv(
            &__model_proof_output!(self.qkv_y.forward(y))?,
            self.heads,
            self.head_dim,
        )?;
        let qy = rms(&qy, &self.q_norm_y, RMS_EPS)?;
        let ky = rms(&ky, &self.k_norm_y, RMS_EPS)?;

        let (qx, kx, vx) = (to_bhsd(&qx)?, to_bhsd(&kx)?, to_bhsd(&vx)?);
        let (qy, ky, vy) = (to_bhsd(&qy)?, to_bhsd(&ky)?, to_bhsd(&vy)?);
        let (qx, kx) = apply_rope(&qx, &kx, cos_img, sin_img)?;
        let (qy, ky) = apply_rope(&qy, &ky, cos_txt, sin_txt)?;

        // joint sequence [txt, img] along the token axis (axis 2 of [B,H,S,Dh])
        let q = Tensor::cat(&[&qy, &qx], 2)?;
        let k = Tensor::cat(&[&ky, &kx], 2)?;
        let v = Tensor::cat(&[&vy, &vx], 2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let out = sdpa(&q, &k, &v, scale)?;

        let out_y = merge_heads(&out.narrow(2, 0, ny)?.contiguous()?)?;
        let out_x = merge_heads(&out.narrow(2, ny, nx)?.contiguous()?)?;
        Ok((
            __model_proof_output!(self.proj_x.forward(&out_x))?,
            __model_proof_output!(self.proj_y.forward(&out_y))?,
        ))
    }
}

/// `MMDiTBlockT2I` — dual-stream block: joint attention + per-stream SwiGLU FFN, each gated by an
/// AdaLN modulation of the shared (already-SiLU'd) condition.
pub struct MMDiTBlockT2I {
    norm_x1: Tensor,
    norm_y1: Tensor,
    attn: MMDiTJointAttention,
    norm_x2: Tensor,
    norm_y2: Tensor,
    mlp_x: FeedForward,
    mlp_y: FeedForward,
    adaln_img: Linear,
    adaln_txt: Linear,
}

impl MMDiTBlockT2I {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            norm_x1: w.require(&format!("{prefix}.norm_x1.weight"))?,
            norm_y1: w.require(&format!("{prefix}.norm_y1.weight"))?,
            attn: MMDiTJointAttention::from_weights(w, &format!("{prefix}.attn"), dim, heads)?,
            norm_x2: w.require(&format!("{prefix}.norm_x2.weight"))?,
            norm_y2: w.require(&format!("{prefix}.norm_y2.weight"))?,
            mlp_x: FeedForward::from_weights(w, &format!("{prefix}.mlp_x"))?,
            mlp_y: FeedForward::from_weights(w, &format!("{prefix}.mlp_y"))?,
            adaln_img: linear(w, &format!("{prefix}.adaLN_modulation_img.0"))?,
            adaln_txt: linear(w, &format!("{prefix}.adaLN_modulation_txt.0"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        y: &Tensor,
        c: &Tensor,
        cos_img: &Tensor,
        sin_img: &Tensor,
        cos_txt: &Tensor,
        sin_txt: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mx = __model_proof_output!(self.adaln_img.forward(c))?.chunk(6, D::Minus1)?;
        let my = __model_proof_output!(self.adaln_txt.forward(c))?.chunk(6, D::Minus1)?;

        let x_norm = rms_modulate(x, &self.norm_x1, &mx[1], &mx[0])?;
        let y_norm = rms_modulate(y, &self.norm_y1, &my[1], &my[0])?;
        let (attn_x, attn_y) = self
            .attn
            .forward(&x_norm, &y_norm, cos_img, sin_img, cos_txt, sin_txt)?;
        let x = gated(x, &mx[2], &attn_x)?;
        let y = gated(y, &my[2], &attn_y)?;

        let x_mlp = self
            .mlp_x
            .forward(&rms_modulate(&x, &self.norm_x2, &mx[4], &mx[3])?)?;
        let x = gated(&x, &mx[5], &x_mlp)?;
        let y_mlp = self
            .mlp_y
            .forward(&rms_modulate(&y, &self.norm_y2, &my[4], &my[3])?)?;
        let y = gated(&y, &my[5], &y_mlp)?;
        Ok((x, y))
    }
}

/// `PiTBlock` — per-pixel block: compress the per-patch pixels to one attention token, rotary
/// attention across patch tokens, expand back, GELU MLP; both stages AdaLN-gated per pixel.
pub struct PiTBlock {
    compress_to_attn: Linear,
    expand_from_attn: Linear,
    norm1: Tensor,
    attn: RotaryAttention,
    norm2: Tensor,
    mlp: Mlp,
    adaln: Linear,
    pixel_dim: usize,
    attn_dim: usize,
}

impl PiTBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        pixel_dim: i32,
        attn_dim: i32,
        attn_heads: i32,
    ) -> Result<Self> {
        Ok(Self {
            compress_to_attn: linear(w, &format!("{prefix}.compress_to_attn"))?,
            expand_from_attn: linear(w, &format!("{prefix}.expand_from_attn"))?,
            norm1: w.require(&format!("{prefix}.norm1.weight"))?,
            attn: RotaryAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                attn_dim,
                attn_heads,
            )?,
            norm2: w.require(&format!("{prefix}.norm2.weight"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.0"))?,
            pixel_dim: pixel_dim as usize,
            attn_dim: attn_dim as usize,
        })
    }

    /// `x`: `[B·L, P², pixel_dim]`; `s_cond`: `[B·L, context_dim]`; `cos`/`sin`: pixel-stream
    /// 2-D RoPE for the `(Hs, Ws)` patch grid. `b`/`l` are the batch and patch-grid token counts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        s_cond: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let (bl, p2, _) = x.dims3()?;
        let cond = __model_proof_output!(self.adaln.forward(s_cond))?.reshape((
            bl,
            p2,
            6 * self.pixel_dim,
        ))?;
        let m = cond.chunk(6, D::Minus1)?; // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

        let x_norm = rms_modulate(x, &self.norm1, &m[1], &m[0])?;
        let x_flat = x_norm.reshape((bl, p2 * self.pixel_dim))?;
        let x_comp = __model_proof_output!(self.compress_to_attn.forward(&x_flat))?.reshape((
            b,
            l,
            self.attn_dim,
        ))?;
        let attn_out = self.attn.forward(&x_comp, cos, sin)?;
        let attn_exp = __model_proof_output!(self
            .expand_from_attn
            .forward(&attn_out.reshape((b * l, self.attn_dim))?))?
        .reshape((bl, p2, self.pixel_dim))?;
        let x = gated(x, &m[2], &attn_exp)?;

        let mlp_out = self
            .mlp
            .forward(&rms_modulate(&x, &self.norm2, &m[4], &m[3])?)?;
        gated(&x, &m[5], &mlp_out)
    }
}

/// SW-6129: the backbone AdaLN sites are the shared fused AdaLN, bit-identical to
/// `rms` + `modulate`.
#[cfg(test)]
mod adaln_route_tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn values(dims: &[usize], seed: u64, amp: f32, bias: f32, dev: &Device) -> Tensor {
        let mut s = seed | 1;
        let v: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp + bias
            })
            .collect();
        Tensor::from_vec(v, dims, dev).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The former site: `rms` then the composable modulate.
    fn former(x: &Tensor, w: &Tensor, scale: &Tensor, shift: &Tensor) -> Tensor {
        candle_gen::modulate::modulate(&rms(x, w, RMS_EPS).unwrap(), scale, shift).unwrap()
    }

    /// `(x, weight, scale, shift)` sites at the real widths: the MMDiT patch stream (hidden 1536,
    /// `[1, D]` and `[1, 1, D]` modulation) and the PiT pixel stream (pixel dim 16, per-pixel
    /// `[B·L, P², 16]` modulation chunks of a `[B·L, P², 96]` table).
    fn sites(dtype: DType, dev: &Device) -> Vec<(Tensor, Tensor, Tensor, Tensor)> {
        let x = values(&[1, 40, 1536], 1, 3.0, 0.3, dev)
            .to_dtype(dtype)
            .unwrap();
        let w = values(&[1536], 2, 0.4, 1.0, dev).to_dtype(dtype).unwrap();
        let m = values(&[1, 6 * 1536], 3, 0.5, 0.0, dev)
            .to_dtype(dtype)
            .unwrap();
        let c = m.chunk(6, D::Minus1).unwrap();
        let px = values(&[12, 16, 16], 4, 3.0, 0.2, dev)
            .to_dtype(dtype)
            .unwrap();
        let pw = values(&[16], 5, 0.4, 1.0, dev).to_dtype(dtype).unwrap();
        let pm = values(&[12, 16, 96], 6, 0.5, 0.0, dev)
            .to_dtype(dtype)
            .unwrap();
        let pc = pm.chunk(6, D::Minus1).unwrap();
        vec![
            (x.clone(), w.clone(), c[1].clone(), c[0].clone()),
            (x, w, c[4].unsqueeze(1).unwrap(), c[3].unsqueeze(1).unwrap()),
            (px, pw, pc[1].clone(), pc[0].clone()),
        ]
    }

    #[test]
    fn fused_adaln_sites_match_the_former_chain() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for (i, (x, w, scale, shift)) in sites(dtype, &Device::Cpu).iter().enumerate() {
                let got = rms_modulate(x, w, scale, shift).unwrap();
                assert_eq!(
                    bits(&got),
                    bits(&former(x, w, scale, shift)),
                    "site {i} {dtype:?}"
                );
            }
        }
    }

    /// CUDA bf16: the three site shapes take the fused AdaLN and are bit-identical to the same
    /// calls with the kernel switch off (and to the former chain).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_adaln_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::modulate::ADALN_PATH;
        const CALLS: u64 = 3;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));
        let sites = sites(DType::BF16, &dev);
        let run = || {
            let outs: Vec<Tensor> = sites
                .iter()
                .map(|(x, w, sc, sh)| rms_modulate(x, w, sc, sh).unwrap().flatten_all().unwrap())
                .collect();
            Tensor::cat(&outs, 0).unwrap()
        };
        ADALN_PATH.set_enabled(Some(true));
        let start = ADALN_PATH.thread_counts();
        let on = run();
        let c_on = ADALN_PATH.thread_counts().since(&start);
        assert_eq!((c_on.fused, c_on.reference_total()), (CALLS, 0), "{c_on}");
        ADALN_PATH.set_enabled(Some(false));
        let start = ADALN_PATH.thread_counts();
        let off = run();
        let c_off = ADALN_PATH.thread_counts().since(&start);
        assert_eq!(c_off.reference_for("disabled"), CALLS, "{c_off}");
        ADALN_PATH.set_enabled(Some(true));
        let (a, b) = (bits(&on), bits(&off));
        let mismatches = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        println!(
            "pid adaln (bf16, D=1536 / 16): {mismatches} of {} differ; on {c_on}, off {c_off}",
            a.len()
        );
        assert_eq!(mismatches, 0);
    }
}
