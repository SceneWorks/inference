//! Offline decode probe for saved prototype boundaries; no generation route.
use candle_gen::candle_core::{DType, Device};
use candle_gen_minimax_h3 as h3;
use std::io::Write;
fn arg(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    Ok(args
        .windows(2)
        .find(|a| a[0] == name)
        .ok_or_else(|| format!("missing {name}"))?[1]
        .clone())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok(),
        "explicit device assignment required"
    );
    let device = Device::new_cuda(0)?;
    let data = candle_gen::candle_core::safetensors::load(arg("--intermediates")?, &device)?;
    let vae = h3::MiniMaxH3VideoVae::load_decode_only(
        std::path::Path::new(&arg("--root")?),
        &device,
        DType::BF16,
    )?;
    for key in ["source.raw", "upscale.normalized"] {
        let latent = data.get(key).ok_or("missing saved latent")?;
        let normalized = if key == "source.raw" {
            h3::upscale_prototype::normalize_vae_raw(latent)?
        } else {
            latent.clone()
        };
        let pixels = h3::revert_pixel_normalization(&vae.decode(&normalized)?)?.narrow(2, 0, 39)?;
        let mut file = std::fs::File::create(format!("{}-{key}.rgb", arg("--out-prefix")?))?;
        for image in h3::frames_to_images(&pixels)? {
            file.write_all(&image.pixels)?;
        }
        device.synchronize()?;
        println!("decoded {key}: {:?}", pixels.dims());
    }
    Ok(())
}
