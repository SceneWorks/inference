//! sc-5147 task 1 acceptance gate: the FLUX.2 VAE **encoder** must match diffusers `AutoencoderKLFlux2`.
//!
//! Loads a real `vae/` checkpoint (a diffusers `AutoencoderKLFlux2`) **as f32** into
//! [`candle_gen_flux2::vae::Flux2Vae::new_with_encoder`], encodes a synthetic image, and compares both
//! the posterior **mean** (`Flux2Vae::encode`) and the **packed, bn-normalized** transformer latent
//! (`Flux2Vae::encode_packed`) against `scripts/dump_flux2_vae_encode_golden.py`. f32 both sides → a
//! tight correctness gate for the conv encoder + 2×2 patchify + bn-normalize.
//!
//! Heavy + machine-specific (needs the VAE weights + GPU), so it is **gated** on env vars and skips
//! cleanly when they are unset:
//!   FLUX2_VAE_DIR             — a FLUX.2 / Lens `vae` snapshot dir (config.json + .safetensors)
//!   FLUX2_VAE_ENCODE_GOLDENS  — flux2_vae_encode_golden.safetensors
//!                               (default: .scratch/flux2-vae-encode-goldens/…)
//! Run with the `cuda` feature (absolute goldens path — cargo test cwd is the crate dir):
//!   cargo test -p candle-gen-flux2 --features cuda --test integration vae_encode_parity:: -- --nocapture

use candle_gen::candle_core::{DType, Result};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::testkit::{tensor_cosine, tensor_peak_rel_host_f64};
use candle_gen_flux2::vae::Flux2Vae;

#[test]
#[ignore = "needs the FLUX.2 vae snapshot (FLUX2_VAE_DIR) + goldens (FLUX2_VAE_ENCODE_GOLDENS)"]
fn flux2_vae_encode_matches_reference() -> Result<()> {
    let vae_dir = match std::env::var("FLUX2_VAE_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set FLUX2_VAE_DIR to a FLUX.2 / Lens vae snapshot dir");
            #[cfg(feature = "model-inference-proof")]
            crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-flux2/tests/vae_encode_parity.rs:55 (flux2_vae_encode_matches_reference): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("FLUX2_VAE_ENCODE_GOLDENS").unwrap_or_else(|_| {
        ".scratch/flux2-vae-encode-goldens/flux2_vae_encode_golden.safetensors".to_string()
    });
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_flux2_vae_encode_golden.py)"
        );
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-flux2/tests/vae_encode_parity.rs:61 (flux2_vae_encode_matches_reference): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&vae_dir)
        .expect("read vae dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {vae_dir}");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };
    let vae = Flux2Vae::new_with_encoder(vb)?;

    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let image = g["image"].to_dtype(DType::F32)?;
    let mean_golden = g["mean"].to_dtype(DType::F32)?;
    let packed_golden = g["packed"].to_dtype(DType::F32)?;
    eprintln!(
        "image={:?} mean={:?} packed={:?}",
        image.dims(),
        mean_golden.dims(),
        packed_golden.dims()
    );

    // 1) Posterior mean (the conv encoder + quant_conv).
    let mean = vae.encode(&image)?;
    assert_eq!(
        mean.dims(),
        mean_golden.dims(),
        "encode mean shape mismatch"
    );
    let mean_pr = tensor_peak_rel_host_f64(&mean, &mean_golden, Some(1e-12)) as f32;
    let mean_cos = tensor_cosine(&mean, &mean_golden) as f32;
    eprintln!("encode mean: peak_rel={mean_pr:.3e} cosine={mean_cos:.7}");

    // 2) Packed bn-normalized transformer latent (mean → patchify → bn-normalize).
    let packed = vae.encode_packed(&image)?;
    assert_eq!(
        packed.dims(),
        packed_golden.dims(),
        "encode_packed shape mismatch"
    );
    let packed_pr = tensor_peak_rel_host_f64(&packed, &packed_golden, Some(1e-12)) as f32;
    let packed_cos = tensor_cosine(&packed, &packed_golden) as f32;
    eprintln!("encode_packed: peak_rel={packed_pr:.3e} cosine={packed_cos:.7}");

    // Single conv-encode pass; CUDA/CPU-vs-torch f32 should be tight.
    assert!(mean_cos > 0.999, "encode mean cosine {mean_cos:.7} ≤ 0.999");
    assert!(mean_pr < 2e-2, "encode mean peak_rel {mean_pr:.3e} ≥ 2e-2");
    assert!(
        packed_cos > 0.999,
        "encode_packed cosine {packed_cos:.7} ≤ 0.999"
    );
    assert!(
        packed_pr < 2e-2,
        "encode_packed peak_rel {packed_pr:.3e} ≥ 2e-2"
    );
    eprintln!("ALL PASS");
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED vae_encode_parity::flux2_vae_encode_matches_reference"
    );
    Ok(())
}
