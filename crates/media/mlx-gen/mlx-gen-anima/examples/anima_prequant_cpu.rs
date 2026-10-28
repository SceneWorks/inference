//! CPU-only Anima Turbo q8/q4 export using the existing install-time quantizer.
//! The caller supplies an external RSS, disk, and wall guard.
#[cfg(target_os = "macos")]
#[path = "../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod metal_guard;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: anima_prequant_cpu SPLIT_FILES_SOURCE DESTINATION (external RSS guard required)".into());
    }
    metal_guard::forbid_metal();
    eprintln!(
        "Metal submission trap armed: {}",
        metal_guard::metal_forbidden()
    );
    let src = Path::new(&args[0]);
    let dst = Path::new(&args[1]);
    let dit = src.join("diffusion_models/anima-turbo-v1.0.safetensors");
    let te = src.join("text_encoders/qwen_3_06b_base.safetensors");
    let vae = src.join("vae/qwen_image_vae.safetensors");
    for path in [&dit, &te, &vae] {
        if !path.is_file() {
            return Err(format!("Anima source missing: {}", path.display()).into());
        }
    }
    struct RestoreDevice(mlx_rs::Device);
    impl Drop for RestoreDevice {
        fn drop(&mut self) {
            mlx_rs::Device::set_default(&self.0);
        }
    }
    let _restore = RestoreDevice(mlx_rs::Device::try_default()?);
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    mlx_rs::with_new_default_stream(
        mlx_rs::Stream::cpu(),
        || -> Result<(), Box<dyn std::error::Error>> {
            for (tier, bits) in [("bf16", None), ("q8", Some(8)), ("q4", Some(4))] {
                let tier_dir = dst.join(tier);
                for sub in ["diffusion_models", "text_encoders", "vae"] {
                    std::fs::create_dir_all(tier_dir.join(sub))?;
                }
                let dit_out = tier_dir.join("diffusion_models/anima-turbo-v1.0.safetensors");
                if let Some(bits) = bits {
                    mlx_gen_anima::quantize_anima_dit(&dit, &dit_out, bits, 64)?;
                } else {
                    std::os::unix::fs::symlink(std::fs::canonicalize(&dit)?, &dit_out)?;
                }
                for (source, sub, name) in [
                    (&te, "text_encoders", "qwen_3_06b_base.safetensors"),
                    (&vae, "vae", "qwen_image_vae.safetensors"),
                ] {
                    std::os::unix::fs::symlink(
                        std::fs::canonicalize(source)?,
                        tier_dir.join(sub).join(name),
                    )?;
                }
            }
            std::fs::write(
                dst.join("model_index.json"),
                "{\n  \"_class_name\": \"AnimaPipeline\"\n}\n",
            )?;
            Ok(())
        },
    )
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Anima CPU export requires macOS");
    std::process::exit(1);
}
