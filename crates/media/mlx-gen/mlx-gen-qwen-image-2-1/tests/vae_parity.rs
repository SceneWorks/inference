//! RGBA VAE parity vs the frozen `AutoencoderKLQwenImage21` on the committed miniature snapshot
//! (`tools/dump_qwen21_vae.py`): every encoder stage, the moments and posterior mode, every
//! decoder stage and the decoded RGBA image, for a square and a non-square geometry.
//!
//! Tolerances: **1e-3 × peak** on the encoder (measured ≤ 3e-6: the convolutions agree almost
//! exactly) and **1e-2 × peak** on the decoder stages and the clamped image (measured ≤ 3e-3: the
//! decoder's upsampling stack runs at 16× the resolution and compounds the reduced-precision f32
//! Metal matmul the repository documents at ~1e-3 per op). Every stage prints its own numbers.

use mlx_gen_qwen_image_2_1::load_vae;
use mlx_rs::ops::indexing::IndexOp;

use crate::common::{assert_close, fixture, tiny_snapshot};

const ENCODER_TOL: f32 = 1e-3;
const DECODER_TOL: f32 = 1e-2;

#[test]
fn every_stage_of_encode_and_decode_matches_upstream() {
    let w = fixture("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot()).unwrap();
    let want = |key: &str| {
        // Fixtures are 5-D `[B, C, T=1, H, W]`; the port is single-frame NCHW.
        w.require(key)
            .unwrap_or_else(|_| panic!("fixture lacks {key}"))
            .squeeze_axes(&[2])
            .unwrap()
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
        assert_eq!(decoded.shape()[1], 4, "four RGBA channels");
        assert_close(
            &format!("{case}/decoded"),
            &decoded,
            &want(&format!("{case}/decoded")),
            DECODER_TOL,
        );

        // Composited RGB keeps three channels and the spatial size.
        let rgb = mlx_gen_qwen_image_2_1::rgba_to_rgb_over_white(&decoded).unwrap();
        assert_eq!(rgb.shape()[1], 3);
        assert_eq!(rgb.index((.., 0..1, .., ..)).shape()[2], decoded.shape()[2]);
    }
}
