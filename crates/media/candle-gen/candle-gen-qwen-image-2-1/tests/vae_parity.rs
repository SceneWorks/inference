//! RGBA VAE parity vs the frozen `AutoencoderKLQwenImage21` on the committed miniature snapshot
//! (`tools/dump_qwen21_vae.py`): every encoder stage, the moments and posterior mode, every
//! decoder stage and the decoded RGBA image, for a square and a non-square geometry.
//!
//! Same fixture and same oracle as `mlx-gen-qwen-image-2-1`'s own `tests/vae_parity.rs`, reached
//! across the backend boundary by relative path. The MLX twin budgets **1e-3 × peak** on the
//! encoder and **1e-2 × peak** on the decoder for Metal's reduced-precision f32 matmul; candle's
//! CPU f32 runs the same arithmetic torch did, so the bars here are the MEASURED maxima rounded up
//! (see the constants). Every stage prints its own numbers.

use candle_core::Device;
use candle_gen_qwen_image_2_1::{load_vae, rgba_to_rgb_over_white};

use crate::common::{assert_close, fixture, tiny_snapshot};

/// Encoder bar. Measured worst `max|Δ| / max(1, peak)` = **3.6e-7** over both cases and every
/// stage (largest absolute 3.8e-6, at `down_block_4`); the MLX twin budgets 1e-3 for Metal.
const ENCODER_TOL: f32 = 1e-5;
/// Decoder bar. Measured worst `max|Δ| / max(1, peak)` = **9.3e-7** over both cases and every
/// stage (largest absolute 7.6e-6, at `up_block_3`, whose 16× stack peaks near 18); the MLX twin
/// budgets 1e-2 for Metal's upsampling stack.
const DECODER_TOL: f32 = 1e-5;

#[test]
fn every_stage_of_encode_and_decode_matches_upstream() {
    let w = fixture("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot(), &Device::Cpu).unwrap();
    let want = |key: &str| {
        // Fixtures are 5-D `[B, C, T=1, H, W]`; the port is single-frame NCHW.
        w.tensor(key).squeeze(2).unwrap()
    };
    for case in ["square", "tall"] {
        let image = want(&format!("{case}/image"));
        let (moments, trace) = vae.encode_moments_traced(&image).unwrap();
        assert_eq!(trace.len(), 4 + vae.config().dim_mult.len());
        for (stage, got) in &trace {
            assert_close(
                &format!("{case}/{stage}"),
                got,
                &want(&format!("{case}/trace/{stage}")),
                ENCODER_TOL,
            );
        }
        assert_close(
            &format!("{case}/moments"),
            &moments,
            &want(&format!("{case}/moments")),
            ENCODER_TOL,
        );
        let mode = vae.encode_mode(&image).unwrap();
        assert_close(
            &format!("{case}/mode"),
            &mode,
            &want(&format!("{case}/mode")),
            ENCODER_TOL,
        );

        let z = want(&format!("{case}/z"));
        let (decoded, trace) = vae.decode_rgba_traced(&z).unwrap();
        assert_eq!(trace.len(), 4 + vae.config().dim_mult.len());
        for (stage, got) in &trace {
            assert_close(
                &format!("{case}/{stage}"),
                got,
                &want(&format!("{case}/trace/{stage}")),
                DECODER_TOL,
            );
        }
        assert_eq!(decoded.dims()[1], 4, "four RGBA channels");
        assert_close(
            &format!("{case}/decoded"),
            &decoded,
            &want(&format!("{case}/decoded")),
            DECODER_TOL,
        );

        // Composited RGB keeps three channels and the spatial size.
        let rgb = rgba_to_rgb_over_white(&decoded).unwrap();
        assert_eq!(rgb.dims()[1], 3);
        assert_eq!(rgb.dims()[2], decoded.dims()[2]);
        assert_eq!(rgb.dims()[3], decoded.dims()[3]);
    }
}
