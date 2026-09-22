//! The bounded real-weight validation render (sc-24108) — `#[ignore]`d, needs the pinned
//! `Qwen/Qwen-Image-2.1` snapshot at `MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT` (inference never
//! self-fetches or derives a cache location, epic 13657) and the Metal GPU.
//!
//! Loads the released bf16 weights through the explicit catalog's production load path and
//! renders one image at the upstream default preset (1:1 2048×2048, 40 steps, seed 42, no
//! guidance), writing a PNG to `QWEN_IMAGE_2_1_RENDER_OUT` (default: the current directory). It
//! also pins the released tokenizer's system-prefix drop count (14).
//!
//! Run detached with an external RSS guard (the host has been kernel-panicked by an unguarded
//! MLX run before):
//!
//! ```sh
//! MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1/snapshots/790c9263… \
//! QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-24108 \
//!   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
//!   e2e_real_weights:: -- --ignored --nocapture
//! ```
//!
//! `QWEN_IMAGE_2_1_RENDER_SIZE=WxH` and `QWEN_IMAGE_2_1_RENDER_STEPS=N` override the preset for a
//! quicker smoke.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::Progress;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, PRESETS};

fn snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the pinned snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// Tokenizer only — no weights are opened. The released `processor/chat_template.jinja` renders a
/// lone system message as exactly the literal prefix the port tokenizes, so the derived drop count
/// is upstream's `_drop_idx` (14); `tools/_qwen21_common.py` re-proves the template/literal
/// agreement token-for-token through `Qwen3VLProcessor.apply_chat_template` whenever the snapshot
/// is present.
#[test]
#[ignore]
fn released_tokenizer_drops_fourteen_system_tokens() {
    let tokenizer = load_tokenizer(&snapshot()).unwrap();
    let count = system_prompt_drop_count(&tokenizer).unwrap();
    let ids = tokenizer
        .encode_ids(&mlx_gen_qwen_image_2_1::system_prefix(), true)
        .unwrap();
    eprintln!("released tokenizer: system prefix = {count} tokens {ids:?}");
    assert_eq!(count, 14);
    assert_eq!(
        ids,
        [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]
    );
}

/// Config only — no weights are opened. The 128 transcribed `QWEN_IMAGE_2_1_Z64_MEAN` / `_STD`
/// floats that identify the latent space must be the released `vae/config.json`'s
/// `latents_mean` / `latents_std`, bit for bit after the f32 round both sides make.
#[test]
#[ignore]
fn latent_space_statistics_match_the_released_vae_config() {
    use mlx_gen::gen_core::{QWEN_IMAGE_2_1_Z64_MEAN, QWEN_IMAGE_2_1_Z64_STD};
    let cfg =
        mlx_gen_qwen_image_2_1::VaeConfig::from_json_file(&snapshot().join("vae/config.json"))
            .unwrap();
    assert_eq!(cfg.z_dim, 64);
    assert_eq!(cfg.scale_factor_spatial, 16);
    assert_eq!(cfg.latents_mean, QWEN_IMAGE_2_1_Z64_MEAN.to_vec());
    assert_eq!(cfg.latents_std, QWEN_IMAGE_2_1_Z64_STD.to_vec());
    eprintln!("released vae/config.json latents_mean/std == QWEN_IMAGE_2_1_Z64_MEAN/STD (64 + 64)");
}

#[test]
#[ignore]
fn validation_render_default_preset() {
    let root = snapshot();
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(mlx_gen_qwen_image_2_1::DEFAULT_STEPS);
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();

    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
            .to_owned(),
        width,
        height,
        steps: Some(steps),
        seed: Some(42),
        ..Default::default()
    };
    let render_started = Instant::now();
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                eprintln!(
                    "step {current}/{total} ({:.1}s)",
                    render_started.elapsed().as_secs_f32()
                );
            } else {
                eprintln!("{p:?} ({:.1}s)", render_started.elapsed().as_secs_f32());
            }
        })
        .unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("images expected");
    };
    let image = &images[0];
    assert_eq!((image.width, image.height), (width, height));
    let path = out_dir.join(format!(
        "qwen_image_2_1_{width}x{height}_{steps}steps_seed42.png"
    ));
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    eprintln!(
        "wrote {} after {:.1}s total ({:.1}s render)",
        path.display(),
        started.elapsed().as_secs_f32(),
        render_started.elapsed().as_secs_f32()
    );
}
