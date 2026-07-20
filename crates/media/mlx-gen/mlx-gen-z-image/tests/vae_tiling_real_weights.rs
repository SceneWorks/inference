//! sc-13571 / GitHub SceneWorks#1658: the memory-bounded tiled VAE decode ([`Vae::decode_tiled`])
//! matches the single-pass [`Vae::decode`] within blend tolerance on real weights. This is the
//! consume-side proof for the fix that lets an 8 GB Mac render z-image at 1024² (the untiled decode
//! materializes a ~14 GiB transient in one shot — OOM/violet on 8 GB).
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` snapshot in the HF cache. Run with:
//!   cargo test -p mlx-gen-z-image --release --test vae_tiling_real_weights -- --ignored --nocapture
//!
//! The Z-Image VAE is a diffusers AutoencoderKL whose up-blocks + `conv_norm_out` use **GroupNorm**
//! (spatially global statistics), so per-tile stats drift slightly from the whole-image ones and the
//! parity is APPROXIMATE — visually seam-free (verified by eye), not bit-exact like an RMSNorm VAE. The
//! assertion therefore guards only against gross seams/corruption; the measured 512 px residual is
//! ~1.1% of pixels differing by >8.

use mlx_gen::tiling::TilingConfig;
use mlx_gen::FlowMatchEuler;
use mlx_gen_z_image::{
    create_noise, decoded_to_image, denoise, load_text_encoder, load_tokenizer, load_transformer,
    load_vae, slice_valid, unpack_latents,
};
use mlx_rs::{Array, Dtype};

mod common;
use common::snapshot_opt;

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image-Turbo snapshot"]
fn tiled_decode_matches_single_pass() {
    let Some(snap) = snapshot_opt() else {
        eprintln!(
            "skip tiled_decode_matches_single_pass: no Z-Image-Turbo snapshot in the HF cache"
        );
        return;
    };

    // A real (natural) final latent via a 4-step denoise, so the GroupNorm parity is measured on a
    // coherent image (with smooth gradients — the worst case for tile-seam drift) rather than on noise.
    let (w, h) = (1024u32, 1024u32);
    let tok = load_tokenizer(&snap).unwrap();
    let te = load_text_encoder(&snap).unwrap();
    let prompt = "a red fox sitting in a snowy forest, photorealistic, sharp focus";
    let t = tok.tokenize(prompt).unwrap();
    let (ids, mask) = mlx_gen::tokenizer::to_arrays(&t);
    mask.eval().unwrap();
    let num_valid: i32 = mask.as_slice::<i32>().iter().sum();
    let cap = bf16(&slice_valid(&te.forward(&ids, &mask).unwrap(), num_valid).unwrap());
    let transformer = load_transformer(&snap).unwrap();
    let scheduler = FlowMatchEuler::for_static_shift(4, 3.0);
    let latents = denoise(
        &transformer,
        &scheduler,
        bf16(&create_noise(42, w, h).unwrap()),
        &cap,
    )
    .unwrap();
    let unpacked = unpack_latents(&latents).unwrap();
    let s = unpacked.shape();
    let latent5 = unpacked
        .reshape(&[s[0], s[1], 1, s[2], s[3]])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let vae = load_vae(&snap).unwrap();
    let full = decoded_to_image(
        &vae.decode(&latent5)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap(),
    )
    .unwrap();
    let cfg = TilingConfig::spatial_only(512, 64); // the production small-Mac tile (pipeline::decode_tiling)
    let tiled = decoded_to_image(
        &vae.decode_tiled(&latent5, &cfg, None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap(),
    )
    .unwrap();

    assert_eq!((full.width, full.height), (tiled.width, tiled.height));
    let (mut max_d, mut n8) = (0i32, 0usize);
    for (a, b) in full.pixels.iter().zip(&tiled.pixels) {
        let d = (*a as i32 - *b as i32).abs();
        max_d = max_d.max(d);
        if d > 8 {
            n8 += 1;
        }
    }
    let frac8 = n8 as f64 / full.pixels.len() as f64;
    println!(
        "tiled (512px) vs single-pass: max|Δ|={max_d}  px>8={:.4}%",
        100.0 * frac8
    );
    assert!(
        frac8 < 0.03,
        "tiled decode grossly diverges from single-pass: {:.3}% of pixels differ by >8",
        100.0 * frac8
    );
}
