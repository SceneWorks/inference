//! Offline fixed 39-frame 512x288@24 ->1024x576 experiment. No product route.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_minimax_h3 as h3;
use h3::upscale_prototype::{self as prototype, LatentUpscaler};

fn argument(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    Ok(args
        .windows(2)
        .find(|a| a[0] == name)
        .ok_or_else(|| format!("missing {name}"))?[1]
        .clone())
}
fn pixel_normalize(rgb: &Tensor) -> candle_gen::Result<Tensor> {
    let mean = Tensor::from_slice(&h3::PIXEL_MEAN, (1, 3, 1, 1, 1), rgb.device())?;
    let std = Tensor::from_slice(&h3::PIXEL_STD, (1, 3, 1, 1, 1), rgb.device())?;
    Ok(rgb.broadcast_sub(&mean)?.broadcast_div(&std)?)
}
fn metadata(path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len > 16 * 1024 * 1024 {
        return Err("oversize safetensors header".into());
    }
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(header["__metadata__"].clone())
}
fn mark(
    stages: &mut Vec<serde_json::Value>,
    name: &str,
    since: Instant,
    device: &Device,
) -> candle_gen::Result<()> {
    device.synchronize()?;
    let seconds = since.elapsed().as_secs_f64();
    println!("{name}: {seconds:.3}s");
    stages.push(serde_json::json!({"stage":name,"seconds":seconds}));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let allowed = [
        "--rgb24",
        "--root",
        "--upscaler",
        "--lora",
        "--out",
        "--denoise",
        "--guide",
        "--noise-fixture",
        "--capture-only",
    ];
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut seen = std::collections::HashSet::new();
    if !args.len().is_multiple_of(2) {
        return Err("settings must be flag/value pairs".into());
    }
    for pair in args.chunks_exact(2) {
        if !allowed.contains(&pair[0].as_str()) || !seen.insert(&pair[0]) {
            return Err(format!("unsupported or repeated setting {}", pair[0]).into());
        }
    }
    if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
        return Err("assign CUDA_VISIBLE_DEVICES explicitly before this offline experiment".into());
    }
    if !cfg!(feature = "cuda") {
        return Err("build this experimental executable with --features cuda".into());
    }
    let source = PathBuf::from(argument("--rgb24")?);
    let root = PathBuf::from(argument("--root")?);
    let upscaler = PathBuf::from(argument("--upscaler")?);
    let lora = PathBuf::from(argument("--lora")?);
    let output = PathBuf::from(argument("--out")?);
    let noise_path = PathBuf::from(argument("--noise-fixture")?);
    let denoise: f32 = argument("--denoise")?.parse()?;
    let guided: bool = argument("--guide")?.parse()?;
    let sigma = prototype::recipe_sigma(denoise)?;
    let capture_only: bool = argument("--capture-only")
        .unwrap_or_else(|_| "false".into())
        .parse()?;
    // All installed files preflight before a CUDA context or heavyweight load.
    for path in [&source, &upscaler, &lora, &noise_path] {
        if !path.is_file() {
            return Err(format!("missing installed {}", path.display()).into());
        }
    }
    for component in ["vae", "text_encoder", "transformer_ref"] {
        if !root.join(component).is_dir() {
            return Err(format!("missing installed H3 component {component}").into());
        }
    }
    prototype::validate_lora_metadata(&metadata(&lora)?)?;
    let bytes = std::fs::read(source)?;
    if bytes.len() != 39 * 512 * 288 * 3 {
        return Err("input must contain exactly39 frames512x288RGB24".into());
    }
    let device = Device::new_cuda(0)?;
    let mut stages = Vec::new();
    let pixels = Tensor::from_vec(
        bytes.iter().map(|&v| v as f32 / 255.).collect::<Vec<_>>(),
        (1, 39, 288, 512, 3),
        &device,
    )?
    .permute((0, 4, 1, 2, 3))?
    .contiguous()?;
    let mut phase = Instant::now();
    // Published VAE weights and the pinned Comfy reference use F32. Keep
    // this precision choice confined to the offline experiment.
    let mut vae = h3::MiniMaxH3VideoVae::load(&root, &device, DType::F32)?;
    let source_raw = vae.encode(&pixel_normalize(&pixels)?)?.mean().clone();
    let source_normalized = prototype::normalize_vae_raw(&source_raw)?;
    let mut intermediates = std::collections::HashMap::new();
    intermediates.insert("source.raw", source_raw.to_device(&Device::Cpu)?);
    intermediates.insert(
        "source.normalized",
        source_normalized.to_device(&Device::Cpu)?,
    );
    if source_raw.dims() != [1, 24, 12, 18, 32] {
        return Err(format!("source encode shape {:?}", source_raw.dims()).into());
    }
    mark(&mut stages, "source_encode", phase, &device)?;
    phase = Instant::now();
    // Source guide is decoded and re-encoded before learned enlargement. It is
    // never the enlarged latent or an ordinary reference video presentation.
    let guide = if guided && sigma.is_some() {
        let decoded = h3::revert_pixel_normalization(&prototype::decode_pinned_vae(
            &vae,
            &source_normalized,
        )?)?;
        intermediates.insert(
            "source.decoded.rgb",
            decoded.narrow(2, 0, 39)?.to_device(&Device::Cpu)?,
        );
        let guide_pixels = prototype::source_guide_pixels(&decoded.narrow(2, 0, 39)?, 576, 1024)?;
        intermediates.insert("guide.pixels.rgb", guide_pixels.to_device(&Device::Cpu)?);
        Some(
            prototype::normalize_vae_raw(vae.encode(&pixel_normalize(&guide_pixels)?)?.mean())?
                .to_dtype(DType::F32)?,
        )
    } else {
        None
    };
    mark(
        &mut stages,
        "source_guide_decode_resize_encode",
        phase,
        &device,
    )?;
    if let Some(guide) = &guide {
        intermediates.insert("guide.normalized", guide.to_device(&Device::Cpu)?);
    }
    if capture_only {
        candle_gen::candle_core::safetensors::save(
            &intermediates,
            output.with_extension("intermediates.safetensors"),
        )?;
        std::fs::write(
            output.with_extension("stages.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"status":"source_guide_capture_only", "vae_dtype":"fp32", "stages":stages}),
            )?,
        )?;
        return Ok(());
    }
    drop(vae);
    h3::release_device_memory(&device)?;
    phase = Instant::now();
    let net = LatentUpscaler::load(&upscaler, &device, DType::F16)?;
    let enlarged = net
        .upscale_vae_raw(&source_raw, 36, 64)?
        .to_dtype(DType::F32)?;
    intermediates.insert("upscale.normalized", enlarged.to_device(&Device::Cpu)?);
    candle_gen::candle_core::safetensors::save(
        &intermediates,
        output.with_extension("intermediates.safetensors"),
    )?;
    drop(net);
    drop(source_raw);
    h3::release_device_memory(&device)?;
    mark(&mut stages, "learned_upscale", phase, &device)?;
    phase = Instant::now();
    // Learned output already has the normalized Comfy/DiT representation.
    let result = if let Some(sigma) = sigma {
        let tok = h3::MiniMaxH3Tokenizer::from_snapshot(&root)?;
        let (ids, mask, _) = tok.encode_ref2va(prototype::CAPTION, &[], &device)?;
        let te_root = root.join("text_encoder");
        let cfg = h3::MiniMaxH3TeConfig::from_component_dir(&te_root)?;
        let files = candle_gen::loader::sorted_safetensors(&te_root, "h3-upscale conditioner")?;
        let prefixes = h3::lm_prefixes(h3::LM_PREFIX, &cfg);
        let refs: Vec<_> = prefixes.iter().map(String::as_str).collect();
        let weights =
            candle_gen::Weights::from_files_filtered(&files, &device, DType::BF16, &refs)?;
        let te =
            h3::MiniMaxH3TextEncoder::from_weights(&weights, h3::LM_PREFIX, &cfg, DType::BF16)?;
        drop(weights);
        let context = te.forward(&ids, &mask)?;
        drop(te);
        h3::release_device_memory(&device)?;
        mark(&mut stages, "caption_ref2va_conditioner", phase, &device)?;
        phase = Instant::now();
        let mut dit = h3::MiniMaxH3Dit::load(&root, "transformer_ref", &device, DType::BF16)?;
        let specs = [AdapterSpec {
            path: lora.clone(),
            scale: 1.,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let report = h3::apply_minimax_h3_adapters(&mut dit, &specs)?;
        println!("LoRA applied targets: {}", report.applied);
        mark(
            &mut stages,
            "unaccelerated_ref2va_dit_lora_load",
            phase,
            &device,
        )?;
        phase = Instant::now();
        let noises = candle_gen::candle_core::safetensors::load(&noise_path, &device)?;
        let noise = noises.get("video").ok_or("missing video noise")?;
        let result = prototype::refine_once(
            &dit,
            &context,
            &enlarged,
            guide.as_ref(),
            noise,
            noises.get("guide"),
            sigma,
        )?;
        drop(dit);
        drop(context);
        h3::release_device_memory(&device)?;
        mark(
            &mut stages,
            "one_euler_refinement_frozen_audio",
            phase,
            &device,
        )?;
        result
    } else {
        enlarged
    };
    phase = Instant::now();
    vae = h3::MiniMaxH3VideoVae::load_decode_only(&root, &device, DType::F32)?;
    let decoded = h3::revert_pixel_normalization(&prototype::decode_pinned_vae(&vae, &result)?)?
        .narrow(2, 0, 39)?;
    let images = h3::frames_to_images(&decoded)?;
    let mut file = std::fs::File::create(&output)?;
    for image in images {
        file.write_all(&image.pixels)?;
    }
    mark(&mut stages, "decode_rgb24", phase, &device)?;
    std::fs::write(
        output.with_extension("stages.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "status":"native_output_emitted_unreviewed","upstream_commit":prototype::REFERENCE_COMMIT,
            "backend":"candle/cuda","tier":"bf16","vae_dtype":"fp32","upscaler_dtype":"fp16","frames":39,"width":1024,"height":576,"denoise":denoise,"sigma":sigma,"guide":guided,
            "seed":444,"stages":stages,"internal_audio":"clean_zero_frozen","delivery_audio":"mux original source soundtrack externally under AAC policy"
        }))?,
    )?;
    Ok(())
}
