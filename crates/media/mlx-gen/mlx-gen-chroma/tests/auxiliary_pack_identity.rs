//! sc-16462: the offline auxiliary pack must be BIT-IDENTICAL to the in-app load-time pack.
//!
//! Chroma has two ways to reach a packed T5/VAE: the offline converter
//! ([`mlx_gen_chroma::convert::repack_auxiliaries`], which is what gets published) and the load-time
//! seam ([`mlx_gen_chroma::loader::quantize_t5_for_dense_source`], which packs a dense snapshot in
//! process when `spec.quantize` is set). If those two ever diverge, a user rendering from a dense
//! source gets different pixels than a user on the shipped tier, at the same declared quant tier —
//! and every calibration number measured on one is invalid for the other.
//!
//! This is the conversion-faithfulness gate, and it is the one gate that a pixel comparison against a
//! bf16 reference can never substitute for: it asserts the two packed paths agree with each other,
//! not that either agrees with bf16. Both produce the same `mx.quantize(w.astype(bf16), group, bits)`
//! call over the same surface, so the outputs are compared for EXACT equality, not a tolerance.
//!
//! Run:
//!   SC16462_BASELINE=<shipped q4/q8 tier dir> SC16462_OUT=<scratch dir> \
//!     cargo test -p mlx-gen-chroma --release --test auxiliary_pack_identity \
//!       -- --ignored --nocapture packed_auxiliaries_match_load_time_quantization

use mlx_rs::Array;
use std::path::PathBuf;

/// Deterministic token ids over T5's vocabulary — content is irrelevant, only that both paths see
/// exactly the same input.
fn probe_tokens(len: i32) -> Array {
    let ids: Vec<i32> = (0..len).map(|i| (i * 7919) % 32000).collect();
    Array::from_slice(&ids, &[1, len])
}

fn probe_latents() -> Array {
    let data: Vec<f32> = (0..16 * 8 * 8)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    Array::from_slice(&data, &[1, 16, 8, 8])
}

fn host_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(mlx_rs::Dtype::Float32)
        .expect("cast to f32")
        .as_slice::<f32>()
        .to_vec()
}

/// The width the baseline tier's transformer declares — the width the auxiliaries must match.
fn tier_bits(root: &std::path::Path) -> i32 {
    mlx_gen::quant::packed_quant_bits_at(&root.join("transformer"))
        .expect("read transformer width")
        .expect("baseline tier transformer is packed")
}

#[test]
#[ignore = "needs the shipped Chroma tier + Apple Silicon MLX; set SC16462_BASELINE/SC16462_OUT"]
fn packed_auxiliaries_match_load_time_quantization() {
    let baseline = PathBuf::from(
        std::env::var("SC16462_BASELINE").expect("SC16462_BASELINE = shipped q4/q8 tier dir"),
    );
    let out = PathBuf::from(std::env::var("SC16462_OUT").expect("SC16462_OUT = scratch dir"));
    let bits = tier_bits(&baseline);
    let packed_root = out.join(format!("identity-q{bits}"));
    if packed_root.exists() {
        std::fs::remove_dir_all(&packed_root).expect("clear stale");
    }
    std::fs::create_dir_all(&out).expect("create scratch");
    mlx_gen_chroma::convert::repack_auxiliaries(&baseline, &packed_root).expect("repack");

    // The offline artifact must declare exactly the tier width — never a wider auxiliary.
    let declared = mlx_gen::quant::packed_quant_bits_at(&packed_root.join("text_encoder"))
        .expect("read T5 width")
        .expect("T5 is packed");
    assert_eq!(
        declared, bits,
        "packed T5 declares Q{declared} on a Q{bits} tier; the auxiliary width must follow the tier"
    );

    // ---- T5: offline-packed vs dense-source packed in process ----
    let tokens = probe_tokens(mlx_gen_chroma::MAX_SEQUENCE_LENGTH as i32);
    let offline = mlx_gen_chroma::loader::load_t5_encoder(&packed_root).expect("load packed T5");
    let offline_t5 = host_f32(&offline.forward(&tokens).expect("packed T5 forward"));
    drop(offline);

    let mut load_time =
        mlx_gen_chroma::loader::load_t5_encoder(&baseline).expect("load dense bf16 T5");
    mlx_gen_chroma::loader::quantize_t5_for_dense_source(&mut load_time, bits)
        .expect("load-time T5 pack");
    let load_time_t5 = host_f32(&load_time.forward(&tokens).expect("load-time T5 forward"));
    drop(load_time);

    assert_eq!(
        offline_t5.len(),
        load_time_t5.len(),
        "T5 output geometry differs between the packed paths"
    );
    let t5_mismatches = offline_t5
        .iter()
        .zip(&load_time_t5)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        t5_mismatches, 0,
        "Q{bits} offline-packed T5 differs from the load-time pack in {t5_mismatches}/{} positions; \
         the published artifact would not reproduce the in-app dense path",
        offline_t5.len()
    );

    // ---- VAE: same comparison over decode and encode ----
    let latents = probe_latents();
    let offline_vae = mlx_gen_chroma::loader::load_vae(&packed_root).expect("load packed VAE");
    let offline_decode = host_f32(&offline_vae.decode(&latents).expect("packed decode"));
    let offline_encode = host_f32(
        &offline_vae
            .encode(&offline_vae.decode(&latents).expect("packed decode"))
            .expect("packed encode"),
    );
    drop(offline_vae);

    let mut load_time_vae = mlx_gen_chroma::loader::load_vae(&baseline).expect("load dense VAE");
    mlx_gen_chroma::loader::quantize_vae_for_dense_source(&mut load_time_vae, bits)
        .expect("load-time VAE pack");
    let load_time_decode = host_f32(&load_time_vae.decode(&latents).expect("load-time decode"));
    let load_time_encode = host_f32(
        &load_time_vae
            .encode(&load_time_vae.decode(&latents).expect("load-time decode"))
            .expect("load-time encode"),
    );
    drop(load_time_vae);

    for (label, offline, load_time) in [
        ("decode", &offline_decode, &load_time_decode),
        ("encode", &offline_encode, &load_time_encode),
    ] {
        let mismatches = offline
            .iter()
            .zip(load_time.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches,
            0,
            "Q{bits} offline-packed VAE {label} differs from the load-time pack in {mismatches}/{} \
             positions",
            offline.len()
        );
    }

    println!(
        "SC16462_IDENTITY {}",
        serde_json::json!({
            "model": std::env::var("SC16462_MODEL").unwrap_or_else(|_| "chroma1_base".into()),
            "tier": format!("q{bits}"),
            "auxiliaryBits": declared,
            "t5GroupSize": mlx_gen_chroma::convert::T5_GROUP_SIZE,
            "t5Positions": offline_t5.len(),
            "vaeDecodePositions": offline_decode.len(),
            "exactMatch": true,
        })
    );
    println!("✓ Q{bits} offline pack ≡ load-time pack (exact, T5 + VAE decode/encode)");
}
