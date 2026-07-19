//! In-place ComfyUI Qwen-Image **VAE** real-weight GPU validation (sc-10830, epic 10451 Phase 2b) —
//! an env-driven, `#[ignore]`d integration test that decodes the SAME latent through two `QwenVae`
//! builds and asserts the pixels are byte-identical:
//!
//! * **A (in place):** the tree's `vae/qwen_image_vae.safetensors` (native WAN-VAE keys) loaded via
//!   [`crate::comfyui::remap_vae_wan_to_diffusers`] + `VarBuilder::from_tensors` — the sc-10830 path.
//! * **B (snapshot):** a diffusers-format Qwen-Image `vae/` dir mmapped the way the registry lane
//!   loads it (`sorted_safetensors` → `mmap_var_builder`).
//!
//! Both bf16 sources upcast to the same f32 VAE compute dtype at `VarBuilder` build, so when the two
//! files carry the same VAE weights (only the key layout differs) the decode is **byte-identical** —
//! the direct, DiT-free proof that the remap is correct (a full txt2img render would confound the VAE
//! swap with the DiT). The offline header/value check already shows the remapped tree VAE is
//! bit-identical to the `Qwen/Qwen-Image-2512` diffusers VAE (all 194 tensors, 0.0 max abs diff); this
//! runs the same claim through the real `QwenVae` decode on the deployed GPU.
//!
//! Run (point NATIVE at the ComfyUI tree VAE and DIFFUSERS_DIR at a diffusers Qwen-Image `vae/`; use a
//! ≤ 1536² output so the decode stays the monolithic bit-exact path):
//! ```text
//! set QWEN_VAE_NATIVE=...\ComfyUI\models\vae\qwen_image_vae.safetensors
//! set QWEN_VAE_DIFFUSERS_DIR=...\models--Qwen--Qwen-Image-2512\snapshots\<hash>\vae
//! set QWEN_VAE_OUT=...\out   # optional; writes the two decodes as PPM for the eyeball check
//! cargo test -p candle-gen-qwen-image --features cuda --release comfyui_vae_validate::real_weight -- --ignored --nocapture
//! ```

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::Image;
use candle_gen::testkit::{env_path, env_path_opt, write_ppm};

use crate::vae::QwenVae;
use crate::ENC_DTYPE;

/// Build a [`QwenVae`] from the tree's native-WAN-key VAE file, in place (the sc-10830 loader path).
fn load_inplace(native_file: &std::path::Path, device: &Device) -> QwenVae {
    let map = candle_gen::candle_core::safetensors::load(native_file, &Device::Cpu)
        .expect("load native WAN-VAE safetensors");
    let map = crate::comfyui::remap_vae_wan_to_diffusers(map).expect("remap native WAN-VAE keys");
    let vb = VarBuilder::from_tensors(map, ENC_DTYPE, device);
    QwenVae::new(vb).expect("build QwenVae from in-place remapped VAE")
}

/// Build a [`QwenVae`] from a diffusers `vae/` dir the way the registry lane loads it.
fn load_snapshot(dir: &std::path::Path, device: &Device) -> QwenVae {
    let files = candle_gen::sorted_safetensors(dir, "qwen-image").expect("sorted vae safetensors");
    let vb = candle_gen::mmap_var_builder(&files, ENC_DTYPE, device).expect("mmap vae var builder");
    QwenVae::new(vb).expect("build QwenVae from diffusers vae dir")
}

/// `[1, 3, H, W]` decode in `[-1, 1]` → an RGB [`Image`] (the same clamp/quantize `to_image` uses).
fn to_image(decoded: &Tensor) -> Image {
    let scaled = ((decoded.clamp(-1f32, 1f32).unwrap() + 1.0).unwrap() * 127.5).unwrap();
    let img = candle_gen::round_rgb8(&scaled).unwrap();
    let img = img.i(0).unwrap().to_device(&Device::Cpu).unwrap();
    let (_c, h, w) = img.dims3().unwrap();
    let pixels = img
        .permute((1, 2, 0))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

#[test]
#[ignore = "real-weight GPU validation; set QWEN_VAE_NATIVE/QWEN_VAE_DIFFUSERS_DIR (QWEN_VAE_OUT optional)"]
fn real_weight_vae_inplace_matches_snapshot() {
    let device = candle_gen::default_device().expect("cuda device");
    let native = env_path("QWEN_VAE_NATIVE");
    let diffusers_dir = env_path("QWEN_VAE_DIFFUSERS_DIR");
    println!("in-place VAE:  {}", native.display());
    println!("snapshot VAE:  {}", diffusers_dir.display());

    let t0 = std::time::Instant::now();
    let vae_a = load_inplace(&native, &device);
    let vae_b = load_snapshot(&diffusers_dir, &device);
    println!("loaded both VAEs in {:?}", t0.elapsed());

    // A fixed latent [1, 16, H/8, W/8] → 1024² output (monolithic bit-exact decode path). Seeded via a
    // deterministic ramp (no RNG dep) so the two decodes see the identical input.
    let (lat_c, lat_h, lat_w) = (16usize, 128usize, 128usize);
    let n = lat_c * lat_h * lat_w;
    let vals: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let latent = Tensor::from_vec(vals, (1, lat_c, lat_h, lat_w), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let dec_a = vae_a.decode(&latent).expect("decode in-place VAE");
    let dec_b = vae_b.decode(&latent).expect("decode snapshot VAE");

    // Max abs diff on the raw [-1,1] decode (before the u8 quantize) — the strongest comparison.
    let max_abs = (&dec_a - &dec_b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    println!("=== sc-10830 in-place VAE parity ===");
    println!("  max abs decode diff (in-place vs snapshot): {max_abs:.3e}");

    if let Some(out_dir) = env_path_opt("QWEN_VAE_OUT") {
        std::fs::create_dir_all(&out_dir).ok();
        write_ppm(&out_dir.join("qwen_vae_inplace.ppm"), &to_image(&dec_a));
        write_ppm(&out_dir.join("qwen_vae_snapshot.ppm"), &to_image(&dec_b));
        println!("  wrote decodes to {}", out_dir.display());
    }

    // Byte-identical: same VAE weights, only the key layout differs, same f32 upcast + decode path.
    assert!(
        max_abs == 0.0,
        "in-place VAE decode differs from snapshot (max abs {max_abs:.3e}) — remap or dtype-handling \
         bug (expected byte-identical: same weights, different key layout)"
    );
    println!("sc-10830 in-place VAE parity PASS ✅ (byte-identical decode)");
}
