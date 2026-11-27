//! SW-6128 rows: the fused strided RoPE and QK-RMSNorm + RoPE against the shared reference op
//! chains, at Krea 2's attention shapes — the joint sequence of 512 caption + image tokens
//! (`S = 4608` at 1024², `1536` at 512²), 48 q heads / 12 k heads x 128, bf16, interleaved pairs in
//! f32 math with the `[1, S, 64]` f32 table (exactly the Krea call site).
//!
//! - `rope`: `apply_rope_reference` vs the fused out-of-place and in-place forms;
//! - `rope (bhsd view)`: rotating the permuted `[B, H, S, D]` view of BSHD memory — the reference
//!   needs the view materialized, the fused kernel reads it through its strides;
//! - `rms_norm_rope`: Krea's `rms_scale` → RoPE chain vs the fused in-place QK-norm + RoPE (what
//!   the Krea attention runs per q and k).

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::fused_path::media_fused_policy_guard;
use candle_gen::rope::{
    apply_rope, apply_rope_in_place, apply_rope_reference, rms_norm_rope_in_place,
    rms_norm_rope_reference, RopeLayout, RopeSpec, RMS_NORM_ROPE_PATH, ROPE_PATH,
};

use crate::harness::{Bench, Row};

/// `(label, S)`: Krea's joint sequence at 1024² and 512².
const SEQS: &[(&str, usize)] = &[("1024^2", 512 + 64 * 64), ("512^2", 512 + 32 * 32)];
const HEAD_DIM: usize = 128;

fn tables(s: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let pos = Tensor::arange(0u32, s as u32, dev)?.to_dtype(DType::F32)?;
    let inv = Tensor::arange(0u32, (HEAD_DIM / 2) as u32, dev)?
        .to_dtype(DType::F32)?
        .affine(-(1000f64.ln()) * 2.0 / HEAD_DIM as f64, 0.0)?
        .exp()?;
    let angles = pos
        .unsqueeze(1)?
        .broadcast_mul(&inv.unsqueeze(0)?)?
        .unsqueeze(0)?;
    Ok((angles.cos()?, angles.sin()?))
}

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let mut rows = Vec::new();
    let _media = media_fused_policy_guard(Some(true));
    let _rope = ROPE_PATH.guard(Some(true));
    let _rms = RMS_NORM_ROPE_PATH.guard(Some(true));
    let spec = RopeSpec::interleaved(RopeLayout::Bshd);
    let bhsd = RopeSpec::interleaved(RopeLayout::Bhsd);
    for &(label, s) in SEQS {
        let (cos, sin) = tables(s, dev)?;
        let w = (Tensor::randn(0f32, 0.3, HEAD_DIM, dev)? + 1.0)?;
        for (which, heads) in [("q", 48usize), ("k", 12)] {
            let x =
                Tensor::randn(0f32, 1.0, (1, s, heads, HEAD_DIM), dev)?.to_dtype(DType::BF16)?;
            let shape = format!("{which} [1,{s},{heads},{HEAD_DIM}] bf16 ({label})");
            rows.push(bench.compare(
                "rope (out of place)",
                &shape,
                || apply_rope_reference(&x, &cos, &sin, &spec).map(drop),
                || apply_rope(&x, &cos, &sin, &spec).map(drop),
            )?);
            let owned = x.copy()?;
            rows.push(bench.compare(
                "rope (in place)",
                &shape,
                || apply_rope_reference(&x, &cos, &sin, &spec).map(drop),
                || apply_rope_in_place(&owned, &cos, &sin, &spec).map(drop),
            )?);
            let view = x.transpose(1, 2)?;
            rows.push(bench.compare(
                "rope (bhsd view, out of place)",
                &shape,
                || apply_rope_reference(&view, &cos, &sin, &bhsd).map(drop),
                || apply_rope(&view, &cos, &sin, &bhsd).map(drop),
            )?);
            let owned = x.copy()?;
            rows.push(bench.compare(
                "rms_norm_rope (krea qk-norm, in place)",
                &shape,
                || rms_norm_rope_reference(&x, &w, 1e-5, &cos, &sin, &spec).map(drop),
                || rms_norm_rope_in_place(&owned, &w, 1e-5, &cos, &sin, &spec).map(drop),
            )?);
        }
    }
    let (rope, rms) = (ROPE_PATH.counts(), RMS_NORM_ROPE_PATH.counts());
    eprintln!("rope paths: rope {rope}; rms_norm_rope {rms}");
    Ok(rows)
}
