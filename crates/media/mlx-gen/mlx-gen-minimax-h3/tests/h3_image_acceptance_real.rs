//! SW-7104: feature SW-F363's acceptance renders on MLX (AT1, AT3, AT6, AT8, AT9, AT11).
//!
//! * AT1 — text-to-image at 2048², 4096² and 6272x2688, each from a fresh process (one test per
//!   size, one process per test), at the strategy admission selects against this device's memory
//!   limit ([`device_budget`]).
//! * AT3 — a 3-reference edit whose references are three flat-coloured shapes the prompt never
//!   names by colour; each reference's hue must appear in the output.
//! * AT6 — steps ≠ 6, guidance ≠ 1, a negative prompt and 17 references are refused by the
//!   registered generator, and a LoRA with an unmatched key is refused by name.
//! * AT8 — a 16 MP render with two community LoRAs, admitted under a budget the staged rung cannot
//!   hold, through bounded transformer residency and bounded decode.
//! * AT9 — the short and the PE-expanded prompt rendered at one seed; the expanded render carries
//!   more high-frequency detail.
//! * AT11 — every community LoRA under `MINIMAX_H3_IMAGE_COMMUNITY_LORAS` applies and renders.
//!
//! Every render prints one `RUN RECORD` line: size, tier, budget, the selected rung, the rungs
//! admission refused first, the request's memory switches, the estimate and the MLX peak.
//!
//! All `#[ignore]`d. Inputs: `MINIMAX_H3_IMAGE_SNAPSHOT` (`transformer_diffusers/`,
//! `vae_diffusers/`, `tokenizer/`) and `MINIMAX_H3_SNAPSHOT` (MiniMax-H3's byte-identical
//! `text_encoder/`, shards 1-12 and, for the edit, 14). Each asserts on its inputs rather than
//! skipping. PNGs go to `MINIMAX_H3_IMAGE_ACCEPTANCE_OUT` when set.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationMemory, GenerationOutput, GenerationRequest,
    Generator, LoadShape, LoadSpec, MemoryBudget, MemoryStrategy, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen_minimax_h3::h3_image::admission::{
    admit, estimate, ladder, ImageAdmission, ImageMemoryEstimate, ImageMemoryFacts,
    ImageRequestShape,
};
use mlx_gen_minimax_h3::h3_image::IMAGE_DIT_COMPONENT;
use mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT;
use mlx_gen_minimax_h3::training::memory::device_budget;

const PROMPT: &str = "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves \
                      breaking on the rocks, detailed photograph";

fn image_snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root. This \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join(IMAGE_DIT_COMPONENT).join("config.json").is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no {IMAGE_DIT_COMPONENT}/config.json",
        path.display()
    );
    path
}

/// MiniMax-H3's `text_encoder/`; `vision` also requires shard 14 (the vision tower an edit reads).
fn h3_text_encoder(vision: bool) -> PathBuf {
    let te = PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default())
        .join("text_encoder");
    assert!(
        te.join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT must hold MiniMax-H3's text_encoder/ (found no {})",
        te.join("config.json").display()
    );
    if vision {
        assert!(
            te.join("model-00014-of-00014.safetensors").is_file(),
            "an edit needs MiniMax-H3's text_encoder/ shard 14 (the vision tower)"
        );
    }
    te
}

/// The dense bf16 load, deferred so every rung (rung 4 included) is on the ladder.
fn bf16_spec(vision: bool, adapters: Vec<AdapterSpec>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(image_snapshot()));
    spec.load_shape = LoadShape::DeferredMaterialization;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.to_owned(),
        WeightsSource::Dir(h3_text_encoder(vision)),
    );
    spec.adapters = adapters;
    spec
}

fn load(spec: &LoadSpec) -> Box<dyn Generator> {
    let registry = mlx_gen_minimax_h3::h3_image::register_providers(
        mlx_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    let registration = registry
        .generators()
        .find(|g| (g.descriptor)().id == "minimax_h3_image")
        .expect("minimax_h3_image is registered");
    (registration.load)(spec).unwrap()
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

fn describe(e: &ImageMemoryEstimate) -> String {
    format!(
        "{:?} peak {:.2} GB ({:?} binds)",
        e.selection.strategy,
        gb(e.peak_bytes()),
        e.binding_phase()
    )
}

fn switches(memory: Option<GenerationMemory>) -> String {
    match memory {
        None => "none (resident load defaults)".into(),
        Some(m) => format!(
            "stage_residency={} tile_vae_decode={} chunk_attention={} \
             stream_transformer_blocks={} transformer_window={:?} decode_tile={:?}/{:?}",
            m.stage_residency,
            m.tile_vae_decode,
            m.chunk_attention,
            m.stream_transformer_blocks,
            m.transformer_window_size,
            m.decode_tile_edge,
            m.decode_overlap
        ),
    }
}

fn pixel_stats(image: &Image) -> (f64, f64) {
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

/// Writes `image` as `<MINIMAX_H3_IMAGE_ACCEPTANCE_OUT>/<name>.png` when the directory is set.
fn save(image: &Image, name: &str) {
    let Ok(dir) = std::env::var("MINIMAX_H3_IMAGE_ACCEPTANCE_OUT") else {
        return;
    };
    if dir.is_empty() {
        return;
    }
    std::fs::create_dir_all(&dir).unwrap();
    let path = Path::new(&dir).join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    println!("wrote {}", path.display());
}

/// One admitted render and what admission chose for it.
struct Admitted {
    image: Image,
    admission: ImageAdmission,
    memory: Option<GenerationMemory>,
}

/// Admit `req` against `budget` on the loaded generator's own contract, render it at the admitted
/// selection, and print its run record. `label` names the record and the PNG.
fn render_admitted(
    label: &str,
    spec: &LoadSpec,
    mut req: GenerationRequest,
    budget: MemoryBudget,
) -> Admitted {
    let generator = load(spec);
    let contract = generator.memory_strategy_contract().unwrap().clone();
    let tier = mlx_gen::request_scope::loaded_tier(spec);
    let facts = ImageMemoryFacts::resolve(spec).unwrap();
    let references = mlx_gen::gen_core::h3_image::edit_reference_images(&req.conditioning)
        .unwrap()
        .iter()
        .map(|r| (r.width, r.height))
        .collect();
    let shape = ImageRequestShape {
        width: req.width,
        height: req.height,
        count: 1,
        prompt_tokens: None,
        references,
    };
    let admission = admit(&contract, &facts, &shape, tier, budget)
        .unwrap_or_else(|e| panic!("{label}: admission refused the request: {e}"));
    req.memory = contract.generation_memory(&admission.estimate.selection);
    generator.validate(&req).unwrap();
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = std::time::Instant::now();
    let mut steps = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let mlx_gen::gen_core::Progress::Step { current, .. } = p {
                steps = current;
            }
        })
        .unwrap_or_else(|e| panic!("{label}: the render failed: {e}"));
    let seconds = started.elapsed().as_secs_f64();
    let peak = mlx_rs::memory::get_peak_memory() as u64;
    let GenerationOutput::Images(mut images) = out else {
        panic!("minimax_h3_image must return images");
    };
    assert_eq!(images.len(), 1);
    let image = images.remove(0);
    assert_eq!((image.width, image.height), (req.width, req.height));
    assert_eq!(steps, 6, "the fixed sampler runs six steps");
    let (mean, std) = pixel_stats(&image);
    println!(
        "RUN RECORD {label}: {}x{} backend mlx tier {:?} adapters {} budget {:.2} GB (effective \
         {:.2} GB) | admitted {} after refusing [{}] | memory {} | MLX peak {:.2} GB | {seconds:.1} \
         s | pixel mean {mean:.2} std {std:.2}",
        req.width,
        req.height,
        spec.quantize,
        spec.adapters.len(),
        gb(budget.total_bytes),
        gb(budget.effective_bytes()),
        describe(&admission.estimate),
        admission
            .rejected
            .iter()
            .map(describe)
            .collect::<Vec<_>>()
            .join("; "),
        switches(req.memory),
        gb(peak)
    );
    assert!(
        std > 10.0,
        "{label}: the render is near-constant (std {std:.2})"
    );
    assert!(
        peak <= budget.total_bytes,
        "{label}: the render peaked at {peak} B, above the {} B budget it was admitted under",
        budget.total_bytes
    );
    save(&image, label);
    Admitted {
        image,
        admission,
        memory: req.memory,
    }
}

fn t2i(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        ..Default::default()
    }
}

/// AT1 at one size: dense bf16, admitted against the device's memory limit.
fn at1(width: u32, height: u32) {
    let label = format!("at1-t2i-{width}x{height}");
    let budget = device_budget();
    let spec = bf16_spec(false, Vec::new());
    render_admitted(&label, &spec, t2i(width, height, 7104), budget);
    crate::model_inference_device::record_model_inference_device();
}

/// AT1: 2048² from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); full 2048² render"]
fn h3_image_t2i_2048_at_the_admitted_strategy() {
    at1(2048, 2048);
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_2048_at_the_admitted_strategy"
    );
}

/// AT1: 4096² from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); full 4096² render"]
fn h3_image_t2i_4096_at_the_admitted_strategy() {
    at1(4096, 4096);
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_4096_at_the_admitted_strategy"
    );
}

/// AT1: 6272x2688 (the published 21:9 row) from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); full 6272x2688 render"]
fn h3_image_t2i_6272x2688_at_the_admitted_strategy() {
    at1(6272, 2688);
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_6272x2688_at_the_admitted_strategy"
    );
}

// ── AT3: a 3-reference edit draws on each reference ──────────────────────────────────────────────

/// The three references: flat red disc, blue square and green triangle on white, 512².
fn shape_reference(kind: usize) -> Image {
    let (w, h) = (512u32, 512u32);
    let colour = [[220u8, 30, 30], [30, 60, 220], [30, 170, 40]][kind];
    let inside = |x: f32, y: f32| match kind {
        0 => (x - 256.0).powi(2) + (y - 256.0).powi(2) < 150.0f32.powi(2),
        1 => (106.0..406.0).contains(&x) && (106.0..406.0).contains(&y),
        _ => y > 106.0 && y < 406.0 && (x - 256.0).abs() < (y - 106.0) * 0.6,
    };
    let pixels = (0..w * h)
        .flat_map(|p| {
            let (x, y) = ((p % w) as f32, (p / w) as f32);
            if inside(x, y) {
                colour
            } else {
                [245, 245, 245]
            }
        })
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Fraction of saturated, lit pixels whose hue falls in each reference's band: red [345°, 15°],
/// green [90°, 150°], blue [200°, 260°]. Saturation > 0.5 and value > 0.25 keep greys, whites and
/// shadows out.
pub(crate) fn hue_fractions(pixels: &[u8]) -> [f64; 3] {
    let mut counts = [0u64; 3];
    let n = pixels.len() / 3;
    for px in pixels.chunks_exact(3) {
        let (r, g, b) = (
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        );
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        if max <= 0.25 || delta / max <= 0.5 {
            continue;
        }
        let hue = if max == r {
            60.0 * ((g - b) / delta).rem_euclid(6.0)
        } else if max == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };
        if !(15.0..345.0).contains(&hue) {
            counts[0] += 1;
        } else if (200.0..=260.0).contains(&hue) {
            counts[1] += 1;
        } else if (90.0..=150.0).contains(&hue) {
            counts[2] += 1;
        }
    }
    counts.map(|c| c as f64 / n as f64)
}

/// The least share of the canvas each reference's hue must cover: 0.25 %. One of three objects in
/// a still life covers several percent of a 1024² frame; a reference the render ignored leaves its
/// hue at the background's near-zero share.
pub(crate) const AT3_MIN_HUE_FRACTION: f64 = 0.0025;

/// AT3: `<Picture 1>`-`<Picture 3>` are a red disc, a blue square and a green triangle; the prompt
/// names their positions but no colour, so each hue in the output came from its reference.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/ shards 1-12 and 14); full edit render"]
fn h3_image_edit_draws_on_each_of_three_references() {
    let spec = bf16_spec(true, Vec::new());
    let req = GenerationRequest {
        prompt: "A studio photograph of a wooden table. On the left stands the object from \
                 <Picture 1>, in the centre the object from <Picture 2>, and on the right the \
                 object from <Picture 3>."
            .into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(7104),
        conditioning: vec![Conditioning::MultiReference {
            images: (0..3).map(shape_reference).collect(),
        }],
        ..Default::default()
    };
    let rendered = render_admitted("at3-edit-three-references", &spec, req, device_budget());
    let [red, blue, green] = hue_fractions(&rendered.image.pixels);
    println!(
        "AT3 hue shares of the 1024² edit: red (Picture 1) {:.3} %, blue (Picture 2) {:.3} %, \
         green (Picture 3) {:.3} % (floor {:.2} %)",
        red * 100.0,
        blue * 100.0,
        green * 100.0,
        AT3_MIN_HUE_FRACTION * 100.0
    );
    for (name, share) in [
        ("red disc", red),
        ("blue square", blue),
        ("green triangle", green),
    ] {
        assert!(
            share >= AT3_MIN_HUE_FRACTION,
            "the edit does not draw on the {name} reference: its hue covers {:.3} % of the canvas",
            share * 100.0
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_edit_draws_on_each_of_three_references"
    );
}

// ── AT6: refusals ────────────────────────────────────────────────────────────────────────────────

/// AT6: the registered generator refuses steps ≠ 6, guidance ≠ 1, a negative prompt and 17
/// references before any weight is read, and a LoRA with an unmatched key is refused by name when
/// it is installed on the real DiT — the install `generate` runs.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); one deferred DiT load"]
fn h3_image_refuses_the_unsupported_requests() {
    let spec = bf16_spec(false, Vec::new());
    let generator = load(&spec);
    let ok = t2i(1024, 1024, 7104);
    generator.validate(&ok).expect("the plain request is valid");
    let tiny = Image {
        width: 64,
        height: 64,
        pixels: vec![128; 64 * 64 * 3],
    };
    let cases: Vec<(&str, GenerationRequest, &str)> = vec![
        (
            "steps 8",
            GenerationRequest {
                steps: Some(8),
                ..ok.clone()
            },
            "steps 8 refused",
        ),
        (
            "guidance 4",
            GenerationRequest {
                guidance: Some(4.0),
                ..ok.clone()
            },
            "guidance 4 refused",
        ),
        (
            "a negative prompt",
            GenerationRequest {
                negative_prompt: Some("blurry".into()),
                ..ok.clone()
            },
            "negative prompt refused",
        ),
        (
            "17 references",
            GenerationRequest {
                conditioning: vec![Conditioning::MultiReference {
                    images: vec![tiny; 17],
                }],
                ..ok.clone()
            },
            "17 reference images refused",
        ),
    ];
    for (what, req, expected) in cases {
        let message = generator
            .validate(&req)
            .expect_err(&format!("{what} must be refused"))
            .to_string();
        println!("AT6 {what}: refused with {message:?}");
        assert!(
            message.contains(expected),
            "{what}: the refusal must say {expected:?}, got {message:?}"
        );
    }

    // A file with one matched target and one the image DiT does not have: the strict install
    // refuses the whole file and names the unmatched key, never applying the matched half.
    let dir = match std::env::var("RUNNER_TEMP") {
        Ok(d) if !d.is_empty() => tempfile::tempdir_in(d).unwrap(),
        _ => tempfile::tempdir().unwrap(),
    };
    let mut dit = mlx_gen_minimax_h3::h3_image::MiniMaxH3ImageDit::load_dir_deferred(
        image_snapshot().join(IMAGE_DIT_COMPONENT),
        mlx_rs::Dtype::Bfloat16,
    )
    .expect("deferred image DiT");
    let cfg = dit.config().clone();
    let b = &cfg.base;
    let factor = |shape: &[i32]| {
        mlx_rs::Array::zeros::<f32>(shape)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
    };
    let tensors = [
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
            factor(&[4, b.hidden_size]),
        ),
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
            factor(&[b.inner_dim(), 4]),
        ),
        (
            "transformer.proj_out.lora_A.weight",
            factor(&[4, b.hidden_size]),
        ),
        (
            "transformer.proj_out.lora_B.weight",
            factor(&[b.video_patch_dim(), 4]),
        ),
    ];
    let path = dir.path().join("unmatched.safetensors");
    mlx_rs::Array::save_safetensors(
        tensors.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>(),
        Some(&std::collections::HashMap::<String, String>::new()),
        &path,
    )
    .unwrap();
    let message = mlx_gen_minimax_h3::adapters::apply_minimax_h3_image_adapters(
        &mut dit,
        &[AdapterSpec::new(path, 1.0, AdapterKind::Lora)],
    )
    .expect_err("a LoRA with an unmatched key must be refused")
    .to_string();
    println!("AT6 a LoRA with an unmatched key: refused with {message:?}");
    assert!(
        message.contains("proj_out"),
        "the refusal must name proj_out: {message}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_refuses_the_unsupported_requests"
    );
}

// ── community LoRAs (AT8, AT11) ──────────────────────────────────────────────────────────────────

/// `MINIMAX_H3_IMAGE_COMMUNITY_LORAS`: `;`-separated `name=path` entries, each a community
/// H3-Image LoRA file from the Hub (the `minimax-h3-image-community-lora-*` rows).
fn community_loras() -> Vec<(String, PathBuf)> {
    let raw = std::env::var("MINIMAX_H3_IMAGE_COMMUNITY_LORAS").unwrap_or_default();
    let loras: Vec<(String, PathBuf)> = raw
        .split(';')
        .filter(|e| !e.trim().is_empty())
        .map(|e| {
            let (name, path) = e
                .split_once('=')
                .unwrap_or_else(|| panic!("{e:?} is not name=path"));
            let path = PathBuf::from(path.trim());
            assert!(
                path.is_file(),
                "community LoRA {name}: no file at {}",
                path.display()
            );
            (name.trim().to_owned(), path)
        })
        .collect();
    assert!(
        !loras.is_empty(),
        "MINIMAX_H3_IMAGE_COMMUNITY_LORAS must name the community LoRA files (name=path;...)"
    );
    loras
}

fn lora(path: &Path) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), 1.0, AdapterKind::Lora)
}

/// The two community LoRAs AT8 stacks: `MINIMAX_H3_IMAGE_AT8_LORAS`, two names from
/// [`community_loras`].
fn at8_loras() -> Vec<(String, PathBuf)> {
    let all = community_loras();
    let names = std::env::var("MINIMAX_H3_IMAGE_AT8_LORAS").unwrap_or_default();
    let picked: Vec<(String, PathBuf)> = names
        .split(',')
        .map(|n| {
            all.iter()
                .find(|(name, _)| name == n.trim())
                .unwrap_or_else(|| {
                    panic!("MINIMAX_H3_IMAGE_AT8_LORAS names {n:?}, not a community LoRA")
                })
                .clone()
        })
        .collect();
    assert_eq!(picked.len(), 2, "AT8 applies exactly two LoRAs");
    picked
}

/// AT8: 4096² (16 MP) dense bf16 with two community LoRAs at strength 1, admitted against a
/// budget between the staged and the streamed estimates (capped at the device's memory limit), so
/// the request does not fit resident and admission must walk to bounded transformer residency;
/// the request's switches must engage bounded decode with it.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + MINIMAX_H3_IMAGE_COMMUNITY_LORAS; 16 MP streamed render"]
fn h3_image_16mp_with_two_loras_streams_and_tiles() {
    let loras = at8_loras();
    let spec = bf16_spec(false, loras.iter().map(|(_, p)| lora(p)).collect());
    let generator = load(&spec);
    let contract = generator.memory_strategy_contract().unwrap().clone();
    let tier = mlx_gen::request_scope::loaded_tier(&spec);
    let facts = ImageMemoryFacts::resolve(&spec).unwrap();
    let shape = ImageRequestShape::text_to_image(4096, 4096);
    let at = |strategy| {
        let selection = ladder(&contract, tier)
            .into_iter()
            .find(|s| s.strategy == strategy)
            .unwrap_or_else(|| panic!("{strategy:?} is not on the ladder with two LoRAs"));
        estimate(&contract, &facts, &shape, &selection).unwrap()
    };
    let staged = at(MemoryStrategy::StagedResidency);
    let streamed = at(MemoryStrategy::BoundedTransformerResidency);
    let device = device_budget();
    let total = ((staged.peak_bytes() + streamed.peak_bytes()) / 2).min(device.total_bytes);
    let budget = MemoryBudget {
        total_bytes: total,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    assert!(
        !budget.fits(staged.peak_bytes()) && budget.fits(streamed.peak_bytes()),
        "the {total} B budget must refuse the staged rung ({}) and hold the streamed one ({})",
        staged.peak_bytes(),
        streamed.peak_bytes()
    );
    drop(generator);
    let label = format!(
        "at8-16mp-loras-{}",
        loras
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join("+")
    );
    let rendered = render_admitted(&label, &spec, t2i(4096, 4096, 7104), budget);
    assert_eq!(
        rendered.admission.estimate.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    let memory = rendered.memory.expect("rung 4 sets the request's memory");
    assert!(
        memory.stream_transformer_blocks && memory.tile_vae_decode,
        "rung 4 must stream the transformer and bound the decode: {}",
        switches(Some(memory))
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_16mp_with_two_loras_streams_and_tiles"
    );
}

/// Mean absolute difference of two equal-size 8-bit images, in 8-bit levels.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.pixels.len() as f64
}

/// AT11: each community LoRA, at strength 1, installs strictly on the dense bf16 DiT and renders
/// 1024²; its render must differ from the bare render at the same seed.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + MINIMAX_H3_IMAGE_COMMUNITY_LORAS; one 1024² render per LoRA plus a bare one"]
fn h3_image_community_loras_apply() {
    let loras = community_loras();
    let budget = device_budget();
    let bare = render_admitted(
        "at11-bare",
        &bf16_spec(false, Vec::new()),
        t2i(1024, 1024, 7104),
        budget,
    );
    for (name, path) in &loras {
        let rendered = render_admitted(
            &format!("at11-lora-{name}"),
            &bf16_spec(false, vec![lora(path)]),
            t2i(1024, 1024, 7104),
            budget,
        );
        let moved = mean_abs_diff(&rendered.image, &bare.image);
        println!(
            "AT11 {name} ({}): applied; moves the render by {moved:.2} levels",
            path.display()
        );
        assert!(
            moved > 0.5,
            "community LoRA {name} installed but changed the render by only {moved:.3} levels"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_community_loras_apply"
    );
}

// ── AT9: the PE-expanded prompt renders more detail ──────────────────────────────────────────────

/// Mean absolute 4-neighbour Laplacian of the luma, in 8-bit levels: the image's high-frequency
/// energy. More rendered detail (texture, small objects, edges) raises it.
pub(crate) fn laplacian_energy(image: &Image) -> f64 {
    let (w, h) = (image.width as usize, image.height as usize);
    let luma: Vec<f64> = image
        .pixels
        .chunks_exact(3)
        .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
        .collect();
    let mut sum = 0.0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = luma[y * w + x];
            let l = luma[y * w + x - 1]
                + luma[y * w + x + 1]
                + luma[(y - 1) * w + x]
                + luma[(y + 1) * w + x]
                - 4.0 * c;
            sum += l.abs();
        }
    }
    sum / ((w - 2) * (h - 2)) as f64
}

/// The PE run under test: `MINIMAX_H3_IMAGE_PE_LABEL` (its tier), `MINIMAX_H3_IMAGE_PE_SHORT_PROMPT`
/// (the short t2i prompt it expanded) and `MINIMAX_H3_IMAGE_PE_EXPANDED_FILE` (a UTF-8 file holding
/// its `expanded_caption`, which the profile's step reads out of the PE evidence document).
fn pe_expansion() -> (String, String, String) {
    let label = std::env::var("MINIMAX_H3_IMAGE_PE_LABEL").unwrap_or_default();
    let short = std::env::var("MINIMAX_H3_IMAGE_PE_SHORT_PROMPT").unwrap_or_default();
    let file = std::env::var("MINIMAX_H3_IMAGE_PE_EXPANDED_FILE").unwrap_or_default();
    assert!(
        !label.is_empty() && !short.trim().is_empty() && !file.is_empty(),
        "MINIMAX_H3_IMAGE_PE_LABEL, MINIMAX_H3_IMAGE_PE_SHORT_PROMPT and \
         MINIMAX_H3_IMAGE_PE_EXPANDED_FILE must be set"
    );
    let expanded = std::fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("read the PE expansion {file}: {e}"))
        .trim()
        .to_owned();
    assert!(
        expanded.len() > short.len(),
        "the expansion is not longer than the short prompt"
    );
    (label, short, expanded)
}

/// AT9: the short prompt and the PE expansion of it, rendered at 2048² with one seed; the
/// expanded render must carry more high-frequency detail ([`laplacian_energy`]).
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + the PE evidence (MINIMAX_H3_IMAGE_PE_*); two 2048² renders"]
fn h3_image_pe_expanded_prompt_renders_more_detail() {
    let (label, short, expanded) = pe_expansion();
    println!("AT9 PE tier {label}\n  short:    {short}\n  expanded: {expanded}");
    let spec = bf16_spec(false, Vec::new());
    let budget = device_budget();
    let request = |prompt: &str| GenerationRequest {
        prompt: prompt.to_owned(),
        ..t2i(2048, 2048, 7104)
    };
    let plain = render_admitted("at9-short-prompt", &spec, request(&short), budget);
    let rich = render_admitted("at9-expanded-prompt", &spec, request(&expanded), budget);
    let (plain_detail, rich_detail) = (
        laplacian_energy(&plain.image),
        laplacian_energy(&rich.image),
    );
    println!(
        "AT9 detail (mean |Laplacian| of luma): short {plain_detail:.3}, expanded {rich_detail:.3}"
    );
    assert!(
        rich_detail > plain_detail,
        "the expanded prompt's render ({rich_detail:.3}) is not more detailed than the short \
         prompt's ({plain_detail:.3})"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail"
    );
}
