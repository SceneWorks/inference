//! SW-6133 rows: the fused video-VAE norm + SiLU + temporal pad against the op chain it replaces,
//! at representative Wan decode shapes.
//!
//! - `vae_norm_silu_pad` — building one causal conv's padded input: the chain (`ChanNorm` →
//!   `.silu()` → `cat(cache, x)` / `pad_with_zeros`, channel-major, as `CausalConv3d::forward`
//!   builds it) vs the kernel writing the frame-major conv input directly. Streaming-decode chunk
//!   shapes of the z16 (A14B, bf16) decoder at 1280x720 — mid block, the up stages and the
//!   full-resolution last stage, first chunk (zero pad) and cached chunks — and one z48 (5B, f32)
//!   last-stage chunk at 1280x704.
//! - `vae resblock conv` — the same plus the 3x3x3 causal conv it feeds (three conv2d taps; the
//!   chain permutes each tap's frames into the merged batch, the fused path reads them as views),
//!   at shapes whose per-frame im2col stays under the VAE's 128M-element conv budget.
//!
//! - `vae_norm_silu_pad (LTX conv input)` — LTX's single-pass decoder: pixel-RMS norm + SiLU +
//!   first / last frame replication over every frame at once (f32, 768x768x25 output).
//!
//! Synthetic activations and weights; the decode-level numbers (real z16 weights, PSNR and decode
//! peak) come from candle-gen-wan's `fused_norm_silu_pad_gate` model proof.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::fused_path::Attempt;
use candle_gen::vae_norm_silu_pad::{
    norm_silu_pad, norm_silu_pad_reference, ConvInputLayout, TemporalPad, VaeNorm,
    VAE_NORM_SILU_PAD_PATH,
};

use crate::harness::{Bench, Row};

/// `(label, C, H, W, T, cached frames, dtype)`; the cache plus zero pad always total 2 frames.
const INPUT_SHAPES: &[(&str, usize, usize, usize, usize, usize, DType)] = &[
    (
        "z16 mid 90x160 first chunk",
        384,
        90,
        160,
        1,
        0,
        DType::BF16,
    ),
    ("z16 up1 180x320 cached", 384, 180, 320, 2, 2, DType::BF16),
    ("z16 up2 360x640 cached", 192, 360, 640, 4, 2, DType::BF16),
    ("z16 up3 720x1280 cached", 96, 720, 1280, 4, 2, DType::BF16),
    ("z48 up3 352x640 cached", 256, 352, 640, 4, 2, DType::F32),
];

const CONV_SHAPES: &[(&str, usize, usize, usize, usize, usize, DType)] = &[
    (
        "z16 mid 90x160 first chunk",
        384,
        90,
        160,
        1,
        0,
        DType::BF16,
    ),
    ("z16 192ch 180x320 cached", 192, 180, 320, 4, 2, DType::BF16),
];

struct Inputs {
    x: Tensor,
    gamma: Tensor,
    cache: Option<Tensor>,
    pad: TemporalPad,
    weight: Tensor,
    bias: Tensor,
}

fn inputs(
    c: usize,
    h: usize,
    w: usize,
    t: usize,
    cached: usize,
    dt: DType,
    dev: &Device,
) -> Result<Inputs> {
    let x = Tensor::randn(0f32, 2.0, (1, c, t, h, w), dev)?.to_dtype(dt)?;
    let gamma = Tensor::randn(1f32, 0.2, c, dev)?.to_dtype(dt)?;
    let cache = match cached {
        0 => None,
        n => Some(Tensor::randn(0f32, 1.0, (1, c, n, h, w), dev)?.to_dtype(dt)?),
    };
    let weight = Tensor::randn(0f32, 0.02, (c, c, 3, 3, 3), dev)?.to_dtype(dt)?;
    let bias = Tensor::zeros((1, c, 1, 1, 1), dt, dev)?;
    Ok(Inputs {
        x,
        gamma,
        cache,
        pad: TemporalPad {
            zero_front: 2 - cached,
            ..TemporalPad::default()
        },
        weight,
        bias,
    })
}

fn norm(i: &Inputs) -> VaeNorm<'_> {
    VaeNorm::ChannelL2 {
        gamma: &i.gamma,
        scale: (i.gamma.dim(0).unwrap_or(1) as f64).sqrt(),
        eps: 1e-12,
    }
}

fn fused_input(i: &Inputs) -> Result<Tensor> {
    match norm_silu_pad(
        &i.x,
        &norm(i),
        i.cache.as_ref(),
        i.pad,
        ConvInputLayout::FrameMajor,
    )? {
        Attempt::Fused(t) => Ok(t),
        Attempt::Reference(r) => candle_gen::candle_core::bail!("the fused kernel declined: {r}"),
    }
}

fn reference_input(i: &Inputs) -> Result<Tensor> {
    norm_silu_pad_reference(
        &i.x,
        &norm(i),
        i.cache.as_ref(),
        i.pad,
        ConvInputLayout::ChannelMajor,
    )
}

/// The three causal taps over a padded input, one conv2d per frame (the per-frame im2col stays
/// under the VAE conv budget); `tap(kd)` yields tap `kd`'s merged `[T, C, H, W]` frames.
fn taps(i: &Inputs, t: usize, mut tap: impl FnMut(usize) -> Result<Tensor>) -> Result<Tensor> {
    let mut acc: Option<Tensor> = None;
    for kd in 0..3 {
        let wk = i.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
        let merged = tap(kd)?;
        let frames = (0..t)
            .map(|f| merged.narrow(0, f, 1)?.conv2d(&wk, 1, 1, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        let y = Tensor::cat(&frames, 0)?;
        acc = Some(match acc {
            Some(a) => (a + y)?,
            None => y,
        });
    }
    let y = acc.expect("three taps");
    let (_, o, h, w) = y.dims4()?;
    y.reshape((1, t, o, h, w))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .broadcast_add(&i.bias)
}

/// `(label, C, H, W, T)` of LTX decoder resblock inputs (f32, all frames in one pass).
const LTX_SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("ltx dec 512ch 48x48", 512, 48, 48, 13),
    ("ltx dec 128ch 192x192", 128, 192, 192, 25),
];

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let _kernel = VAE_NORM_SILU_PAD_PATH.guard(Some(true));
    let mut rows = Vec::new();
    for &(label, c, h, w, t, cached, dt) in INPUT_SHAPES {
        let i = inputs(c, h, w, t, cached, dt, dev)?;
        rows.push(bench.compare(
            "vae_norm_silu_pad (Wan conv input)",
            &format!("{label}: {c}ch T{t}+2 {dt:?}"),
            || reference_input(&i).map(drop),
            || fused_input(&i).map(drop),
        )?);
    }
    for &(label, c, h, w, t) in LTX_SHAPES {
        let x = Tensor::randn(0f32, 2.0, (1, c, t, h, w), dev)?;
        let norm = VaeNorm::PixelRms { eps: 1e-8 };
        let pad = TemporalPad {
            replicate_front: 1,
            replicate_back: 1,
            ..TemporalPad::default()
        };
        rows.push(bench.compare(
            "vae_norm_silu_pad (LTX conv input)",
            &format!("{label}: {c}ch T{t}+2 F32"),
            || {
                norm_silu_pad_reference(&x, &norm, None, pad, ConvInputLayout::ChannelMajor)
                    .map(drop)
            },
            || match norm_silu_pad(&x, &norm, None, pad, ConvInputLayout::FrameMajor)? {
                Attempt::Fused(_) => Ok(()),
                Attempt::Reference(r) => candle_gen::candle_core::bail!("declined: {r}"),
            },
        )?);
    }
    for &(label, c, h, w, t, cached, dt) in CONV_SHAPES {
        let i = inputs(c, h, w, t, cached, dt, dev)?;
        let (_, _, _, hh, ww) = i.x.dims5()?;
        rows.push(bench.compare(
            "vae resblock conv (norm+silu+pad+3x3x3)",
            &format!("{label}: {c}->{c}ch T{t} {dt:?}"),
            || {
                let xpad = reference_input(&i)?;
                taps(&i, t, |kd| {
                    xpad.narrow(2, kd, t)?
                        .permute((0, 2, 1, 3, 4))?
                        .reshape((t, c, hh, ww))?
                        .contiguous()
                })
                .map(drop)
            },
            || {
                let xpad = fused_input(&i)?;
                taps(&i, t, |kd| xpad.narrow(1, kd, t)?.reshape((t, c, hh, ww))).map(drop)
            },
        )?);
    }
    Ok(rows)
}
