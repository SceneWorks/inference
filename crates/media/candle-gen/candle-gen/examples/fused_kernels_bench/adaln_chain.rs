//! SW-6129 rows: the composable-chain fused AdaLN (`candle_gen::modulate::adaln_chain`) against
//! the providers' own op chain ([`adaln_chain_reference`]), at the DiT shapes that run it.
//!
//! - Qwen-Image / Lens / SD3 / SANA / Mage / FLUX.1 form: f32 LayerNorm chain cast back, bf16
//!   modulation (`[1, 4096 + 256, 3072]`);
//! - Wan 2.x 14B 480p x 81: the f32 hidden stream's LayerNorm, modulated in f32, cast to bf16
//!   (`[1, 32760, 5120]`);
//! - LTX-2: the f32 `x * recip(sqrt(mean(x²) + eps))` RMS cast back, per-token modulation;
//! - FLUX.2 / Chroma: f32 in, f32 LayerNorm, f32 modulation.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::modulate::{adaln_chain, adaln_chain_reference, AdaLnChain};

use crate::harness::{Bench, Row};

/// `(kernel label, shape label, [B, S, D], per-token modulation, x dtype, spec, modulation dtype)`.
type Case = (
    &'static str,
    &'static str,
    [usize; 3],
    bool,
    DType,
    AdaLnChain<'static>,
    DType,
);

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let mut rows = Vec::new();
    let randn = |dims: &[usize], std: f32, dtype: DType| -> Result<Tensor> {
        Tensor::randn(0f32, std, dims, dev)?.to_dtype(dtype)
    };

    let cases: [Case; 4] = [
        (
            "layer f32 cast back (qwen-image)",
            "1x4352x3072 bf16",
            [1, 4352, 3072],
            false,
            DType::BF16,
            AdaLnChain::layer(1e-6),
            DType::BF16,
        ),
        (
            "layer f32, f32 modulation, bf16 out (wan 14b)",
            "1x32760x5120 f32 -> bf16",
            [1, 32760, 5120],
            false,
            DType::F32,
            AdaLnChain::layer(1e-6)
                .modulate_in(DType::F32)
                .output(DType::BF16),
            DType::F32,
        ),
        (
            "rms recip f32 cast back, per-token (ltx-2)",
            "1x6144x4096 bf16",
            [1, 6144, 4096],
            true,
            DType::BF16,
            AdaLnChain::rms_recip(1e-6),
            DType::BF16,
        ),
        (
            "layer f32, f32 throughout (flux.2)",
            "1x4608x6144 f32",
            [1, 4608, 6144],
            false,
            DType::F32,
            AdaLnChain::layer(1e-6).modulate_in(DType::F32),
            DType::F32,
        ),
    ];
    for (kernel, shape, [b, s, d], per_token, xd, spec, md) in cases {
        let x = randn(&[b, s, d], 1.0, xd)?;
        let ms = if per_token { s } else { 1 };
        let scale = randn(&[b, ms, d], 0.3, md)?;
        let shift = randn(&[b, ms, d], 0.3, md)?;
        rows.push(bench.compare(
            kernel,
            shape,
            || adaln_chain_reference(&x, &spec, &scale, &shift).map(drop),
            || adaln_chain(&x, &spec, &scale, &shift).map(drop),
        )?);
    }
    Ok(rows)
}
