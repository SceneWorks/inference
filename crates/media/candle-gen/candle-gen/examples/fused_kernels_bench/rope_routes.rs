//! SW-6129 rows: the RoPE routes S4 moved onto the shared fused kernel, each against the op chain
//! the crate ran before, at the crate's production attention shape (bf16 activations, f32 tables).
//!
//! - `apply_interleaved_rope`: candle's `rope_i` on an f32 contiguous copy cast back (the old body
//!   of `candle_gen::rope::apply_interleaved_rope`, used by Lens / Qwen-Image / Wan / SCAIL-2) vs the fused
//!   strided kernel it now calls — Qwen-Image 1024² (24 heads, 4096 + 256 tokens) and Wan 2.x 14B
//!   480p x 81 frames (40 heads, 21·30·52 tokens);
//! - `split-half native`: the HF `x·cos + rotate_half(x)·sin` chain in bf16 (Anima, Ideogram,
//!   SenseNova, PiD Gemma-2, MiniMax-H3) vs the fused kernel, at Anima's 1024² DiT shape;
//! - `flux freqs_cis (native)`: FLUX.1's `fr0·x0 + fr1·x1` over the `[B, 1, S, 64, 2, 2]` rotation
//!   matrix in bf16 vs the fused kernel reading the matrix's cos / sin as strided views.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::rotary_emb::rope_i;
use candle_gen::fused_path::media_fused_policy_guard;
use candle_gen::rope::{
    apply_rope, apply_rope_reference, freqs_cis_cos_sin, FreqsCisLayout, RopeLayout, RopeMath,
    RopeSpec, ROPE_PATH,
};

use crate::harness::{Bench, Row};

const HEAD_DIM: usize = 128;

fn angles(s: usize, w: usize, dev: &Device) -> Result<Tensor> {
    let pos = Tensor::arange(0u32, s as u32, dev)?.to_dtype(DType::F32)?;
    let inv = Tensor::arange(0u32, w as u32, dev)?
        .to_dtype(DType::F32)?
        .affine(-(10000f64.ln()) / w as f64, 0.0)?
        .exp()?;
    pos.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)
}

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let mut rows = Vec::new();
    let _media = media_fused_policy_guard(Some(true));
    let _rope = ROPE_PATH.guard(Some(true));

    // `apply_interleaved_rope`: `[B, H, S, D]`, `[S, D/2]` f32 tables.
    for (label, heads, s) in [
        ("qwen-image 1024^2", 24usize, 4096 + 256),
        ("wan 14b 480p x81", 40, 21 * 30 * 52),
    ] {
        let a = angles(s, HEAD_DIM / 2, dev)?;
        let (cos, sin) = (a.cos()?, a.sin()?);
        let x = Tensor::randn(0f32, 1.0, (1, heads, s, HEAD_DIM), dev)?.to_dtype(DType::BF16)?;
        let shape = format!("q [1,{heads},{s},{HEAD_DIM}] bf16 ({label})");
        rows.push(bench.compare(
            "apply_interleaved_rope (was rope_i)",
            &shape,
            || {
                let xf = x.to_dtype(DType::F32)?.contiguous()?;
                rope_i(&xf, &cos, &sin)?.to_dtype(DType::BF16).map(drop)
            },
            || candle_gen::rope::apply_interleaved_rope(&x, &cos, &sin).map(drop),
        )?);
    }

    // Split-half native math, `[B, S, H, D]`, full-width `[1, S, D]` tables in the activation dtype.
    let (heads, s) = (16usize, 4096usize);
    let a = angles(s, HEAD_DIM / 2, dev)?;
    let full = |t: Tensor| -> Result<Tensor> { Tensor::cat(&[&t, &t], D::Minus1)?.unsqueeze(0) };
    let (cos, sin) = (full(a.cos()?)?, full(a.sin()?)?);
    let x = Tensor::randn(0f32, 1.0, (1, s, heads, HEAD_DIM), dev)?.to_dtype(DType::BF16)?;
    let spec = RopeSpec::split_half(RopeLayout::Bshd).with_math(RopeMath::Native);
    rows.push(bench.compare(
        "split-half native (anima/ideogram)",
        &format!("q [1,{s},{heads},{HEAD_DIM}] bf16 (anima 1024^2)"),
        || apply_rope_reference(&x, &cos, &sin, &spec).map(drop),
        || apply_rope(&x, &cos, &sin, &spec).map(drop),
    )?);

    // FLUX.1 freqs_cis: `[B, H, S, D]` with the `[1, 1, S, D/2, 2, 2]` rotation matrix, native math.
    let (heads, s) = (24usize, 512 + 4096);
    let a = angles(s, HEAD_DIM / 2, dev)?.to_dtype(DType::BF16)?;
    let (c, sn) = (a.cos()?, a.sin()?);
    let matrix =
        Tensor::stack(&[&c, &sn.neg()?, &sn, &c], 2)?.reshape((1, 1, s, HEAD_DIM / 2, 2, 2))?;
    let (cos, sin) = freqs_cis_cos_sin(&matrix, FreqsCisLayout::Matrix)?;
    let x = Tensor::randn(0f32, 1.0, (1, heads, s, HEAD_DIM), dev)?.to_dtype(DType::BF16)?;
    let spec = RopeSpec::interleaved(RopeLayout::Bhsd).with_math(RopeMath::Native);
    rows.push(bench.compare(
        "flux freqs_cis (native)",
        &format!("q [1,{heads},{s},{HEAD_DIM}] bf16 (flux.1 1024^2)"),
        || apply_rope_reference(&x, &cos, &sin, &spec).map(drop),
        || apply_rope(&x, &cos, &sin, &spec).map(drop),
    )?);
    Ok(rows)
}
