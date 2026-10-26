//! Real-weight checks against the pinned `black-forest-labs/flux-3-image-klein` package.
//! `#[ignore]`d: they need ~20 GB of weights and the GPU. Inputs come from the environment (no
//! machine path is baked in):
//!
//! * `FLUX3_PACKAGE_DIR` — the pinned package snapshot root (`config.json`, `model.safetensors`,
//!   `ae.safetensors`, `text_encoder/`).
//! * `FLUX3_REAL_GOLDEN` — the upstream CPU reference from `tools/dump_flux3_realweight.py`
//!   (default: the committed `tests/fixtures/flux3_real_golden_64.safetensors`).
//! * `FLUX3_OUT` — a directory the render writes its PNG + record into (render case only).
//! * `FLUX3_REAL_SIZE` / `FLUX3_REAL_STEPS` / `FLUX3_REAL_PROMPT` — the render geometry
//!   (defaults: the release's 1024², 16 steps, the golden's prompt).
//! * `FLUX3_REAL_COMPUTE=fp32` — run the DiT in FP32 instead of the release's bf16 policy.
//!
//! Run each under an external memory guard:
//! `cargo test -p mlx-gen-flux3 --release --test integration -- --ignored real_weights:: --test-threads 1`

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::flux3::{
    text_ids, Flux3Config, GenerationParams, DISTILLED_AUTOENCODER_SHA256, DISTILLED_DIT_SHA256,
};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_flux3::{denoise_flow_map, load_autoencoder, load_dit, Flux3TextEncoder, SampleTokens};
use mlx_rs::{Array, Dtype};

use crate::common::{assert_close, errors, fixtures, host_f32, host_i32, nchw_to_nhwc};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn package() -> PathBuf {
    env_dir("FLUX3_PACKAGE_DIR")
}

/// Lower-case hex SHA-256 of a file, streamed.
fn sha256_hex(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn golden() -> Weights {
    let path = std::env::var("FLUX3_REAL_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixtures().join("flux3_real_golden_64.safetensors"));
    Weights::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn meta_f32(g: &Weights, key: &str) -> f32 {
    g.metadata(key)
        .unwrap_or_else(|| panic!("golden metadata {key}"))
        .parse()
        .unwrap()
}

fn compute() -> Dtype {
    if std::env::var("FLUX3_REAL_COMPUTE").is_ok_and(|v| v == "fp32") {
        Dtype::Float32
    } else {
        Dtype::Bfloat16
    }
}

fn ids_of(a: &Array) -> Vec<[f32; 4]> {
    host_f32(a)
        .chunks(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// The pinned package parses, validates and agrees with the source constants, and its two
/// weight files ARE the pinned revision's (the sha256 pins of `gen_core::flux3`).
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR)"]
fn real_package_config_is_the_release() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    config.validate_supported().unwrap();
    assert_eq!(config.model, "flux-3-image-klein");
    assert_eq!((config.dit.hidden, config.dit.heads), (3072, 24));
    assert_eq!((config.dit.depth, config.dit.depth_single), (5, 28));
    assert!(config.dit.flow_map);
    assert_eq!(config.text_encoder.window_max_length(), 4096);
    assert_eq!(config.sampler.steps, 16);
    assert_eq!(
        sha256_hex(&package().join("model.safetensors")),
        DISTILLED_DIT_SHA256,
        "model.safetensors is not the pinned revision's"
    );
    assert_eq!(
        sha256_hex(&package().join("ae.safetensors")),
        DISTILLED_AUTOENCODER_SHA256,
        "ae.safetensors is not the pinned revision's"
    );
}

/// The pinned tokenizer + the bf16 tower reproduce upstream's window and conditioning. The window
/// is exact; the tower is bf16 on both sides (MLX GPU vs torch CPU): bf16-ulp scale, 4e-2 of peak.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_text_encoder_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let te = Flux3TextEncoder::load(
        &package().join(&config.text_encoder.path),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_owned();
    let out = te.encode(&prompt).unwrap();
    assert_eq!(
        out.window.input_ids,
        host_i32(g.require("text/input_ids").unwrap())
    );
    assert_eq!(out.window.mask, host_i32(g.require("text/mask").unwrap()));
    assert_eq!(
        out.window.positions,
        host_i32(g.require("text/positions").unwrap())
    );
    assert_eq!(out.window.length() % 80, 0);
    assert_close(
        "real text/tokens",
        &out.tokens,
        g.require("text/tokens").unwrap(),
        4e-2,
    );
}

/// One flow-map forward of the real DiT on the golden's inputs. Under bf16 the bound is 1.5× the
/// distance upstream's OWN bf16 autocast lands from its f32 forward on the same input (recorded
/// in the golden); under `FLUX3_REAL_COMPUTE=fp32` it is 1e-2 of peak (Metal's f32 GEMM is itself
/// reduced precision, ~1e-3 relative, over 33 blocks).
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_dit_forward_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let compute = compute();
    let dit = load_dit(&package().join("model.safetensors"), &config, compute).unwrap();
    let tokens = g
        .require("text/tokens")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let text_ids = text_ids(tokens.shape()[0] as usize);
    let image_ids = ids_of(g.require("dit/image_ids").unwrap());
    let out = dit
        .forward(&[SampleTokens {
            text: &tokens,
            text_ids: &text_ids,
            image: g.require("dit/x").unwrap(),
            image_ids: &image_ids,
            image_cond: None,
            timestep: host_f32(g.require("dit/t").unwrap())[0],
            target_timestep: Some(host_f32(g.require("dit/t_ref").unwrap())[0]),
        }])
        .unwrap();
    if compute == Dtype::Float32 {
        assert_close(
            "real dit/velocity (fp32)",
            &out,
            g.require("dit/velocity").unwrap(),
            1e-2,
        );
    } else {
        let bound = (1.5 * meta_f32(&g, "dit_bf16_distance")).max(2e-2);
        let (max_abs, peak, mean) = errors(&out, g.require("dit/velocity").unwrap());
        eprintln!("real dit/velocity (bf16): max|d|={max_abs:.3e} mean|d|={mean:.3e} peak={peak:.3e} bound={bound:.3e} of peak");
        assert!(
            max_abs <= bound * peak.max(1.0),
            "bf16 forward outside upstream's own bf16 distance"
        );
        assert_close(
            "real dit/velocity_bf16",
            &out,
            g.require("dit/velocity_bf16").unwrap(),
            bound,
        );
    }
}

/// The real autoencoder decodes the golden's random tokens (4×4) to upstream's f32 image.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_autoencoder_decode_matches_upstream() {
    let g = golden();
    let ae = load_autoencoder(&package().join("ae.safetensors")).unwrap();
    assert_eq!(ae.latent_channels(), 128);
    let decoded = ae
        .decode(g.require("ae/tokens_rand").unwrap(), 4, 4)
        .unwrap();
    assert_close(
        "real ae/decoded",
        &decoded,
        &nchw_to_nhwc(g.require("ae/decoded").unwrap()),
        5e-3,
    );
}

/// The 16-step flow-map loop at 64×64 from the golden's noise, with the real text encoder and
/// DiT, against upstream's f32 loop: under bf16 the bound is 1.5× upstream's own bf16-loop
/// distance; the decoded image follows within the same bound.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_e2e_64_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let compute = compute();
    let te = Flux3TextEncoder::load(
        &package().join(&config.text_encoder.path),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let text = te.encode(g.metadata("prompt").unwrap()).unwrap();
    drop(te);
    let dit = load_dit(&package().join("model.safetensors"), &config, compute).unwrap();
    let schedule = host_f32(g.require("e2e/schedule").unwrap());
    let ids = mlx_gen_flux3::image_ids(4, 4, config.dit.image_rope_scale as f32);
    let tokens = denoise_flow_map(
        &dit,
        &text,
        g.require("e2e/noise").unwrap(),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |_| {},
    )
    .unwrap();
    let bound = if compute == Dtype::Float32 {
        2e-2
    } else {
        (1.5 * meta_f32(&g, "e2e_bf16_distance")).max(5e-2)
    };
    assert_close(
        "real e2e/tokens",
        &tokens,
        g.require("e2e/tokens").unwrap(),
        bound,
    );
    drop(dit);
    let ae = load_autoencoder(&package().join("ae.safetensors")).unwrap();
    let image = ae.decode(&tokens, 4, 4).unwrap();
    assert_close(
        "real e2e/image",
        &image,
        &nchw_to_nhwc(g.require("e2e/image").unwrap()),
        bound,
    );
}

/// The ordinary catalog route on the real package: `provider_registry().load("flux3_klein")` →
/// one text-to-image render at the release geometry → a PNG and a JSON record in `FLUX3_OUT`.
/// Asserts the progress contract and a non-degenerate image (no constant frame).
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR), FLUX3_OUT and the GPU"]
fn real_distilled_text_to_image_renders() {
    let out_dir = env_dir("FLUX3_OUT");
    std::fs::create_dir_all(&out_dir).unwrap();
    let size: u32 = std::env::var("FLUX3_REAL_SIZE")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(1024);
    let steps: u32 = std::env::var("FLUX3_REAL_STEPS")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(16);
    let prompt = std::env::var("FLUX3_REAL_PROMPT").unwrap_or_else(|_| {
        "a red fox sitting in fresh snow at golden hour, soft light, photograph".to_owned()
    });
    let mut spec = LoadSpec::new(WeightsSource::Dir(package()));
    if compute() == Dtype::Float32 {
        spec.precision = mlx_gen::Precision::Fp32;
    }
    let config = Flux3Config::from_dir(&package()).unwrap();
    let request = GenerationRequest {
        prompt: prompt.clone(),
        width: size,
        height: size,
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&request, 0, &config).unwrap();
    let t_load = Instant::now();
    let g = mlx_gen_flux3::provider_registry()
        .unwrap()
        .load("flux3_klein", &spec)
        .unwrap();
    let load_seconds = t_load.elapsed().as_secs_f64();
    let mut events = Vec::new();
    let t_render = Instant::now();
    let out = g
        .generate(&request, &mut |p| events.push(format!("{p:?}")))
        .unwrap();
    let render_seconds = t_render.elapsed().as_secs_f64();
    let expected: Vec<String> = std::iter::once("ConditioningComplete".to_owned())
        .chain((1..=steps).map(|i| format!("Step {{ current: {i}, total: {steps} }}")))
        .chain(std::iter::once("Decoding".to_owned()))
        .collect();
    assert_eq!(events, expected);
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    assert_eq!(images.len(), 1);
    let image = &images[0];
    assert_eq!((image.width, image.height), (size, size));
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!(
        var.sqrt() > 10.0,
        "a constant frame is not a render (std {})",
        var.sqrt()
    );
    let png = out_dir.join(format!("flux3_klein_{size}_{steps}steps_seed0.png"));
    image::save_buffer(
        &png,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    let record = serde_json::json!({
        "model": "flux3_klein",
        "package_revision": mlx_gen::gen_core::flux3::DISTILLED_WEIGHTS_REVISION,
        "upstream_commit": mlx_gen::gen_core::flux3::UPSTREAM_CODE_REVISION,
        "prompt": prompt,
        "width": size,
        "height": size,
        "steps": params.steps,
        "alpha": params.alpha,
        "seed": params.seed,
        "guidance": serde_json::Value::Null,
        "compute": format!("{:?}", compute()),
        "load_seconds": load_seconds,
        "render_seconds": render_seconds,
        "pixel_std": var.sqrt(),
        "png": png.file_name().unwrap().to_string_lossy(),
    });
    std::fs::write(
        out_dir.join("flux3_klein_render.json"),
        serde_json::to_string_pretty(&record).unwrap() + "\n",
    )
    .unwrap();
    eprintln!(
        "rendered {} in {render_seconds:.1}s (load {load_seconds:.1}s)",
        png.display()
    );
    let _ = Progress::Decoding;
}

/// sc-26238: the pinned prompt-upsampler package (`FLUX3_PUP_DIR`) is the pinned revision
/// (weights sha256), loads through the shared MLX engine, renders the oracle's chat text and
/// token counts byte-for-byte with the REAL tokenizer (the gen-core golden), and rewrites the
/// README request and a locked-box request for real: a non-empty caption with no angle-bracket
/// ids in text mode, and in the box mode a caption citing both locked ids whose table keeps both
/// locked boxes verbatim. The captions are kept in the evidence artifact (`FLUX3_OUT`).
#[test]
#[ignore = "needs the pinned upsampler package (FLUX3_PUP_DIR)"]
fn real_prompt_upsampler_rewrites_the_readme_request() {
    use mlx_gen::gen_core::flux3::pup::{
        self, upsample, BoxSpec, PromptUpsampler, UpsampleMode, UpsampleOptions, UpsamplePlan,
        UpsamplerInput, UPSAMPLER_WEIGHTS_SHA256,
    };
    use mlx_gen::gen_core::PromptEnhancementOutcome;

    let dir = env_dir("FLUX3_PUP_DIR");
    assert_eq!(
        sha256_hex(&dir.join("model.safetensors")),
        UPSAMPLER_WEIGHTS_SHA256,
        "model.safetensors is not the pinned upsampler revision's"
    );
    let started = Instant::now();
    let up = mlx_gen_flux3::Flux3PromptUpsampler::load(&dir).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();

    // The oracle's text goldens with the REAL tokenizer: chat text and token counts.
    let golden_path = std::env::var("FLUX3_PUP_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../contracts/gen-core/tests/fixtures/flux3_pup/golden.json")
        });
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).unwrap()).unwrap();
    for case in golden["plan"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prompt = case["prompt"].as_str().unwrap();
        let boxes = case["boxes_json"]
            .as_str()
            .map(|b| BoxSpec::parse_list(b).unwrap());
        let n = case["n_images"].as_u64().unwrap() as usize;
        let (mode, user_text, locked) = pup::plan_text(prompt, boxes.as_deref(), n).unwrap();
        let user_text = if mode.uses_references() {
            pup::i2i_scaffold(&user_text, locked.is_some())
        } else {
            user_text
        };
        let plan = UpsamplePlan {
            mode,
            original_prompt: prompt.to_string(),
            system_prompt: up.package().prompts().get(mode).to_string(),
            user_text,
            locked,
            references: (0..n)
                .map(|_| mlx_gen::gen_core::core_llm::ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
                .collect(),
            max_new_tokens: 1,
        };
        let chat = up.package().render(&plan).unwrap();
        assert_eq!(
            chat,
            case["chat_text"].as_str().unwrap(),
            "{name}: chat text"
        );
        assert_eq!(
            up.package().tokenizer().encode(&chat, false).unwrap().len() as u64,
            case["chat_tokens_real"].as_u64().unwrap(),
            "{name}: real tokenizer count"
        );
    }

    let out_dir = std::env::var("FLUX3_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut record = serde_json::Map::new();
    record.insert("package_revision".into(), pup::UPSAMPLER_REVISION.into());
    record.insert("load_seconds".into(), load_seconds.into());

    // Text mode: the README request.
    let started = Instant::now();
    let input = UpsamplerInput {
        prompt: "A man walking with his dog, afternoon sun".into(),
        references: Vec::new(),
        boxes: None,
        mode: UpsampleMode::Text,
        bypass: false,
    };
    let mut tokens = 0u32;
    let out = upsample(
        Some(&up as &dyn PromptUpsampler),
        &input,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |p| {
            if let Progress::Upsampling { tokens: t, .. } = p {
                tokens = t;
            }
        },
    )
    .unwrap();
    let t2i_seconds = started.elapsed().as_secs_f64();
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Enhanced);
    assert_eq!(out.report.mode.as_deref(), Some("t2i"));
    assert_eq!(out.report.effective_prompt, out.caption);
    let words = out.caption.split_whitespace().count();
    assert!(
        words >= 60,
        "a t2i caption is 100-350 words upstream, got {words}: {}",
        out.caption
    );
    let text_caption = &out.caption;
    assert!(
        text_caption.chars().all(|c| !"<[".contains(c)),
        "t2i captions carry no ids or tables: {text_caption}"
    );
    assert!(tokens > 0 && tokens <= pup::MAX_TOKENS);
    record.insert("t2i_prompt".into(), input.prompt.clone().into());
    record.insert("t2i_caption".into(), out.caption.clone().into());
    record.insert("t2i_tokens".into(), tokens.into());
    record.insert("t2i_seconds".into(), t2i_seconds.into());

    // Text with locked boxes: the README balloons.
    let started = Instant::now();
    let boxes = BoxSpec::parse_list(
        r#"[{"id": "obj_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]}, {"id": "obj_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]}]"#,
    )
    .unwrap();
    let input = UpsamplerInput {
        prompt: "two hot-air balloons over the sea".into(),
        references: Vec::new(),
        boxes: Some(boxes),
        mode: UpsampleMode::TextWithBoxes,
        bypass: false,
    };
    let out = upsample(
        Some(&up as &dyn PromptUpsampler),
        &input,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    let bbox_seconds = started.elapsed().as_secs_f64();
    assert_eq!(out.report.mode.as_deref(), Some("t2i_bbox"));
    let locked = out
        .report
        .locked_annotations
        .clone()
        .expect("locked rows recorded");
    assert!(locked.contains("[80, 60, 480, 380]") && locked.contains("[200, 620, 420, 800]"));
    assert!(
        out.caption.contains("<obj_1>") && out.caption.contains("<obj_2>"),
        "{}",
        out.caption
    );
    assert!(
        out.caption.contains("\"bbox\": [80, 60, 480, 380]")
            && out.caption.contains("\"bbox\": [200, 620, 420, 800]"),
        "the locked boxes survive reconcile verbatim: {}",
        out.caption
    );
    assert!(
        pup::is_finished(&out.caption),
        "a finished answer: its table fully cited"
    );
    record.insert("t2i_bbox_prompt".into(), input.prompt.clone().into());
    record.insert("t2i_bbox_caption".into(), out.caption.clone().into());
    record.insert("t2i_bbox_locked".into(), locked.into());
    record.insert("t2i_bbox_seconds".into(), bbox_seconds.into());
    std::fs::write(
        out_dir.join("flux3_pup_upsample.json"),
        serde_json::to_string_pretty(&serde_json::Value::Object(record)).unwrap() + "\n",
    )
    .unwrap();
}
