//! SW-6127 rows: the fused AdaLN (`candle_gen::modulate::adaln`) against its reference op chain
//! (`adaln_reference`: candle norm + `(scale + 1)` + `broadcast_mul` + `broadcast_add`).
//!
//! Krea 2's single-stream block modulation is the POC shape: a bf16 `[B, S, 6144]` stream, the
//! `+1`-folded f32 RMSNorm weight with the norm computed on an f32 copy (`norm_in_f32`), and the
//! scale / shift as chunks of the block's `[B, 1, 6·6144]` modulation table. `S = 4118` is the
//! real-caption 1024² joint sequence, `4608` the 512-token caption bucket, `B = 2` the CFG batch,
//! `16896` the 2048² sequence. One LayerNorm row with per-token modulation covers the other variant.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::modulate::{adaln, adaln_reference, AdaLn, ADALN_PATH};

use crate::harness::{Bench, Row};

/// `(B, S, D)` of the Krea rows.
const KREA_SHAPES: &[(usize, usize, usize)] = &[
    (1, 4118, 6144),
    (1, 4608, 6144),
    (2, 4608, 6144),
    (1, 16896, 6144),
];

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let mut rows = Vec::new();
    let _kernel = ADALN_PATH.guard(Some(true));
    for &(b, s, n) in KREA_SHAPES {
        let x = Tensor::randn(0f32, 1.0, (b, s, n), dev)?.to_dtype(DType::BF16)?;
        let w = (Tensor::randn(0f32, 0.3, n, dev)? + 1.0)?;
        let table = Tensor::randn(0f32, 0.3, (b, 1, 6 * n), dev)?.to_dtype(DType::BF16)?;
        let chunks = table.chunk(6, D::Minus1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);
        let spec = AdaLn::rms(1e-6).with_weight(&w).norm_in_f32();
        rows.push(bench.compare(
            "adaln rms +1 (Krea block, f32 norm)",
            &format!("{b}x{s}x{n} bf16, [B,1,D] mod"),
            || adaln_reference(&x, &spec, scale, shift).map(drop),
            || adaln(&x, &spec, scale, shift).map(drop),
        )?);
    }
    let (b, s, n) = (1, 4096, 3072);
    let x = Tensor::randn(0f32, 1.0, (b, s, n), dev)?.to_dtype(DType::BF16)?;
    let scale = Tensor::randn(0f32, 0.3, (b, s, n), dev)?.to_dtype(DType::BF16)?;
    let shift = Tensor::randn(0f32, 0.3, (b, s, n), dev)?.to_dtype(DType::BF16)?;
    let spec = AdaLn::layer(1e-6);
    rows.push(bench.compare(
        "adaln layer +1 (no affine)",
        &format!("{b}x{s}x{n} bf16, [B,S,D] mod"),
        || adaln_reference(&x, &spec, &scale, &shift).map(drop),
        || adaln(&x, &spec, &scale, &shift).map(drop),
    )?);
    Ok(rows)
}
