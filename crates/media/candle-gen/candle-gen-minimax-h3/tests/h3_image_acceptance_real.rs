//! SW-7104: feature SW-F363's acceptance renders on candle/CUDA (AT2, AT3, AT6, AT8, AT9, AT11) —
//! the twin of mlx-gen-minimax-h3's `h3_image_acceptance_real`.
//!
//! * AT2 — text-to-image at 2048², 4096² and 6272x2688, each from a fresh process (one test per
//!   size, one process per test), at the strategy admission selects against the card's
//!   actually-free memory (`cuMemGetInfo` after a synchronize).
//! * AT3 — a 3-reference edit whose references are three flat-coloured shapes the prompt never
//!   names by colour; each reference's hue must appear in the output.
//! * AT6 — steps ≠ 6, guidance ≠ 1, a negative prompt and 17 references are refused by the loaded
//!   generator, and a LoRA with an unmatched key is refused by name.
//! * AT8 — a 16 MP render with two community LoRAs, admitted under a budget the staged rung cannot
//!   hold, through bounded transformer residency and bounded decode.
//! * AT9 — the short and the PE-expanded prompt rendered at one seed; the expansion must be longer
//!   than the short prompt and both renders must complete at the requested size. Michael's visual
//!   review of the two PNGs (2026-11-16) is the recorded evidence that the expansion adds detail.
//! * AT11 — every community LoRA under `MINIMAX_H3_IMAGE_COMMUNITY_LORAS` applies and renders.
//!
//! Every render runs through the request scope admission returns and prints one `RUN RECORD`
//! line: size, tier, budget, the selected rung, the rungs refused first and the request's memory
//! switches.
//!
//! All `#[ignore]`d. Inputs: `MINIMAX_H3_IMAGE_SNAPSHOT` (`transformer_diffusers/`,
//! `vae_diffusers/`, `tokenizer/`) and `MINIMAX_H3_SNAPSHOT` (MiniMax-H3's byte-identical
//! `text_encoder/`, shards 1-12 and, for the edit, 14). Each asserts on its inputs rather than
//! skipping. PNGs go to `MINIMAX_H3_IMAGE_ACCEPTANCE_OUT` when set.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationMemory, GenerationOutput, GenerationRequest,
    Generator, Image, LoadShape, LoadSpec, MemoryBudget, MemoryStrategy, WeightsSource,
};
use candle_gen_minimax_h3::image_memory_strategy::admission::{
    ImageAdmission, ImageMemoryEstimate,
};

const PROMPT: &str = "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves \
                      breaking on the rocks, detailed photograph";

fn image_snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root; this \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join("transformer_diffusers")
            .join("config.json")
            .is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no transformer_diffusers/config.json",
        path.display()
    );
    path
}

/// The dense bf16 load, deferred so every rung (rung 4 included) is on the ladder, with
/// MiniMax-H3's `text_encoder/` as the conditioner; `vision` also requires shard 14.
fn bf16_spec(vision: bool, adapters: Vec<AdapterSpec>) -> LoadSpec {
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
    let mut spec = LoadSpec::new(WeightsSource::Dir(image_snapshot()))
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.components
        .insert("text_encoder".into(), WeightsSource::Dir(te));
    spec.adapters = adapters;
    spec
}

fn load(spec: &LoadSpec) -> Box<dyn Generator> {
    let generator = candle_gen_minimax_h3::image_model::load_image(spec)
        .expect("load the minimax_h3_image generator");
    assert_eq!(generator.descriptor().id, "minimax_h3_image");
    generator
}

fn budget_of(bytes: u64) -> MemoryBudget {
    MemoryBudget {
        total_bytes: bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// The selected card's actually-free memory, after any pending frees complete.
#[cfg(feature = "cuda")]
fn free_device_bytes() -> u64 {
    let device = candle_gen::candle_core::Device::new_cuda(0).expect("CUDA device 0");
    device.synchronize().expect("synchronize");
    let (free, _total) = device
        .as_cuda_device()
        .expect("a CUDA device")
        .cuda_stream()
        .context()
        .mem_get_info()
        .expect("cuMemGetInfo");
    free as u64
}

fn gb(bytes: u64) -> f64 {
    candle_gen_minimax_h3::image_memory_strategy::admission::decimal_gb(bytes)
}

fn short(e: &ImageMemoryEstimate) -> String {
    format!(
        "{:?}{} GPU peak {:.2} GB host peak {:.2} GB ({:?} binds)",
        e.selection.strategy,
        if e.streamed { " (streamed)" } else { "" },
        gb(e.peak_device_bytes()),
        gb(e.peak_host_bytes()),
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
    let mean = image.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
    let var = image
        .pixels
        .iter()
        .map(|&p| (f64::from(p) - mean).powi(2))
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
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .expect("rgb")
        .save(&path)
        .expect("write the render");
    println!("wrote {}", path.display());
}

/// One admitted render and what admission chose for it.
struct Admitted {
    image: Image,
    admission: ImageAdmission,
    memory: Option<GenerationMemory>,
}

/// Admit `req` against `budget` (the free memory, or less) on the loaded generator's contract,
/// open the request scope at the admitted selection, render through it, and print the run record.
#[cfg(feature = "cuda")]
fn render_admitted(label: &str, spec: &LoadSpec, req: GenerationRequest, budget: u64) -> Admitted {
    use candle_gen::gen_core::{
        MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier,
        MemoryOptimizationAuthority, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision,
    };
    use candle_gen_minimax_h3::image_memory_strategy::admission::{admit, ImageRequestShape};
    use candle_gen_minimax_h3::image_memory_strategy::facts_for;

    let generator = load(spec);
    let contract = generator
        .memory_strategy_contract()
        .expect("the image generator publishes a memory contract")
        .clone();
    let facts = facts_for(spec).expect("header-only facts of the real sources");
    let shape = ImageRequestShape::from_request(&req).expect("shape");
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let admission = admit(&contract, &facts, &shape, tier, budget_of(budget))
        .unwrap_or_else(|e| panic!("{label}: admission refused the request: {e}"));
    let references = shape.references.len() as u32;
    let calibration = contract.calibration.clone().expect("calibration identity");
    let context = MemoryRunContext {
        reference_resources: None,
        selection: admission.estimate.selection,
        optimization_authority: MemoryOptimizationAuthority::Estimated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: spec.load_shape,
        mode: if references == 0 {
            MemoryMode::TextToImage
        } else {
            MemoryMode::Edit
        },
        has_reference: references > 0,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: req.width,
            height: req.height,
            batch: 1,
            frames: 1,
            reference_count: references,
        },
        overlay: None,
        budget: budget_of(budget),
        predicted_peak_bytes: admission.estimate.peak_device_bytes(),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "sw-7104-acceptance".into(),
    };
    assert_eq!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept,
        "{label}: the loaded generator must accept the admitted selection"
    );
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .expect("open the admitted request scope")
        .expect("a request scope");
    let mut request = req.clone();
    scope.configure_request(&mut request).expect("configure");
    let started = std::time::Instant::now();
    let mut steps = Vec::new();
    let output = generator
        .generate(&request, &mut |p| {
            if let candle_gen::gen_core::Progress::Step { current, .. } = p {
                steps.push(current)
            }
        })
        .unwrap_or_else(|e| panic!("{label}: the render failed: {e}"));
    scope
        .finish(MemoryRunOutcome::Complete)
        .expect("finish the request scope");
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(steps, vec![1, 2, 3, 4, 5, 6], "one forward per fixed step");
    let GenerationOutput::Images(mut images) = output else {
        panic!("an image generator must return images")
    };
    assert_eq!(images.len(), 1);
    let image = images.remove(0);
    assert_eq!((image.width, image.height), (req.width, req.height));
    let (mean, std) = pixel_stats(&image);
    println!(
        "RUN RECORD {label}: {}x{} backend candle-cuda tier {:?} adapters {} budget {:.2} GB | \
         admitted {} after refusing [{}] | memory {} | {seconds:.1} s | pixel mean {mean:.2} std \
         {std:.2}",
        req.width,
        req.height,
        spec.quantize,
        spec.adapters.len(),
        gb(budget),
        short(&admission.estimate),
        admission
            .rejected
            .iter()
            .map(short)
            .collect::<Vec<_>>()
            .join("; "),
        switches(request.memory)
    );
    assert!(
        std > 10.0 && (5.0..250.0).contains(&mean),
        "{label}: degenerate render: mean {mean:.1}, std {std:.1}"
    );
    save(&image, label);
    Admitted {
        image,
        admission,
        memory: request.memory,
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

/// AT2 at one size: dense bf16, admitted against the card's free memory.
#[cfg(feature = "cuda")]
fn at2(width: u32, height: u32) {
    let label = format!("at2-t2i-{width}x{height}");
    let spec = bf16_spec(false, Vec::new());
    render_admitted(&label, &spec, t2i(width, height, 7104), free_device_bytes());
}

/// AT2: 2048² from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/) and CUDA; full 2048² render"]
fn h3_image_t2i_2048_at_the_admitted_strategy_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        at2(2048, 2048);
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_2048_at_the_admitted_strategy_cuda"
        );
    }
}

/// AT2: 4096² from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/) and CUDA; full 4096² render"]
fn h3_image_t2i_4096_at_the_admitted_strategy_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        at2(4096, 4096);
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_4096_at_the_admitted_strategy_cuda"
        );
    }
}

/// AT2: 6272x2688 (the published 21:9 row) from a fresh process at the admitted strategy.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/) and CUDA; full 6272x2688 render"]
fn h3_image_t2i_6272x2688_at_the_admitted_strategy_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        at2(6272, 2688);
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_t2i_6272x2688_at_the_admitted_strategy_cuda"
        );
    }
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
/// blue [200°, 260°], green [90°, 150°]. Saturation > 0.5 and value > 0.25 keep greys, whites and
/// shadows out.
fn hue_fractions(pixels: &[u8]) -> [f64; 3] {
    let mut counts = [0u64; 3];
    let n = pixels.len() / 3;
    for px in pixels.chunks_exact(3) {
        let (r, g, b) = (
            f32::from(px[0]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[2]) / 255.0,
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

/// The least share of the canvas each reference's hue must cover: 0.25 % (the MLX twin's bound).
const AT3_MIN_HUE_FRACTION: f64 = 0.0025;

/// AT3: `<Picture 1>`-`<Picture 3>` are a red disc, a blue square and a green triangle; the prompt
/// names their positions but no colour, so each hue in the output came from its reference.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/ shards 1-12 and 14) and CUDA; full edit render"]
fn h3_image_edit_draws_on_each_of_three_references_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
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
        let rendered =
            render_admitted("at3-edit-three-references", &spec, req, free_device_bytes());
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
                "the edit does not draw on the {name} reference: its hue covers {:.3} % of the \
                 canvas",
                share * 100.0
            );
        }
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_edit_draws_on_each_of_three_references_cuda"
        );
    }
}

// ── AT6: refusals ────────────────────────────────────────────────────────────────────────────────

/// AT6: the loaded generator refuses steps ≠ 6, guidance ≠ 1, a negative prompt and 17
/// references before any weight is read, and a LoRA with an unmatched key is refused by name when
/// it is installed on the real DiT (loaded on the host) — the install `generate` runs. No learned
/// tensor runs.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); one host DiT load"]
fn h3_image_refuses_the_unsupported_requests_cuda() {
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
    let mut dit = candle_gen_minimax_h3::MiniMaxH3Dit::load_from_dir(
        image_snapshot().join("transformer_diffusers"),
        &candle_gen::candle_core::Device::Cpu,
        candle_gen::candle_core::DType::BF16,
    )
    .expect("load the image DiT on the host");
    let cfg = dit.config().clone();
    let inner = cfg.num_attention_heads * cfg.attention_head_dim;
    let patch_dim = cfg.in_channels * cfg.patch_size.iter().product::<usize>();
    let factors: Vec<(&str, Vec<usize>)> = vec![
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
            vec![4, cfg.hidden_size],
        ),
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
            vec![inner, 4],
        ),
        (
            "transformer.proj_out.lora_A.weight",
            vec![4, cfg.hidden_size],
        ),
        ("transformer.proj_out.lora_B.weight", vec![patch_dim, 4]),
    ];
    let raw: Vec<Vec<u8>> = factors
        .iter()
        .map(|(_, shape)| vec![0u8; 4 * shape.iter().product::<usize>()])
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = factors
        .iter()
        .zip(&raw)
        .map(|((name, shape), bytes)| {
            (
                (*name).to_owned(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let path = dir.path().join("unmatched.safetensors");
    safetensors::serialize_to_file(views, &None, &path).expect("write the LoRA");
    let message = candle_gen_minimax_h3::adapters::apply_minimax_h3_image_adapters(
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
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_refuses_the_unsupported_requests_cuda"
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

/// AT8: 4096² (16 MP) dense bf16 with two community LoRAs at strength 1, admitted against the
/// lower of the card's free memory and a budget between the staged and the streamed estimates, so
/// the request does not fit resident and admission must walk to bounded transformer residency;
/// the request's switches must engage bounded decode with it.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + MINIMAX_H3_IMAGE_COMMUNITY_LORAS and CUDA; 16 MP streamed render"]
fn h3_image_16mp_with_two_loras_streams_and_tiles_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen::gen_core::MemoryNumericTier;
        use candle_gen_minimax_h3::image_memory_strategy::admission::{
            estimate, ladder, ImageRequestShape,
        };
        use candle_gen_minimax_h3::image_memory_strategy::facts_for;

        let loras = at8_loras();
        let spec = bf16_spec(false, loras.iter().map(|(_, p)| lora(p)).collect());
        let generator = load(&spec);
        let contract = generator.memory_strategy_contract().unwrap().clone();
        drop(generator);
        let facts = facts_for(&spec).expect("header-only facts of the real sources");
        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: None,
            component_precision_floors: &[],
        };
        let rungs = ladder(&contract, tier);
        let at = |strategy| {
            let selection = rungs
                .iter()
                .find(|s| s.strategy == strategy)
                .unwrap_or_else(|| panic!("{strategy:?} is not on the ladder with two LoRAs"));
            estimate(&contract, &facts, &shape, selection).expect("estimate")
        };
        let staged = at(MemoryStrategy::StagedResidency);
        let streamed = at(MemoryStrategy::BoundedTransformerResidency);
        let budget = ((staged.peak_device_bytes() + streamed.peak_device_bytes()) / 2)
            .min(free_device_bytes());
        assert!(
            budget < staged.peak_device_bytes() && budget >= streamed.peak_device_bytes(),
            "the {budget} B budget must refuse the staged rung ({staged}) and hold the streamed \
             one ({streamed})"
        );
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
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_16mp_with_two_loras_streams_and_tiles_cuda"
        );
    }
}

/// Mean absolute difference of two equal-size 8-bit images, in 8-bit levels.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / a.pixels.len() as f64
}

/// AT11: each community LoRA, at strength 1, installs strictly on the dense bf16 DiT and renders
/// 1024²; its render must differ from the bare render at the same seed.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + MINIMAX_H3_IMAGE_COMMUNITY_LORAS and CUDA; one 1024² render per LoRA plus a bare one"]
fn h3_image_community_loras_apply_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        let loras = community_loras();
        let bare = render_admitted(
            "at11-bare",
            &bf16_spec(false, Vec::new()),
            t2i(1024, 1024, 7104),
            free_device_bytes(),
        );
        for (name, path) in &loras {
            let rendered = render_admitted(
                &format!("at11-lora-{name}"),
                &bf16_spec(false, vec![lora(path)]),
                t2i(1024, 1024, 7104),
                free_device_bytes(),
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
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_community_loras_apply_cuda"
        );
    }
}

// ── AT9: the PE expansion is longer and both prompts render ──────────────────────────────────────

/// Whether the PE's `expanded` prompt is longer (in characters, ignoring surrounding whitespace)
/// than the `short` prompt it expanded. AT9's machine check on the expansion itself.
fn expansion_is_longer(short: &str, expanded: &str) -> bool {
    expanded.trim().chars().count() > short.trim().chars().count()
}

/// Mean absolute per-channel difference between two equally sized 8-bit RGB buffers. AT9 uses it
/// to prove the expanded conditioning reached the render: identical images (a generator ignoring
/// the prompt) score 0.
fn mean_abs_pixel_difference(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "renders must have the same size to compare"
    );
    if a.is_empty() {
        return 0.0;
    }
    let total: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.len() as f64
}

/// The smallest mean per-channel difference AT9 accepts between the short- and expanded-prompt
/// renders. Two different prompts at one seed differ by tens of levels; 2.0 only rejects a render
/// that ignored the expansion.
const AT9_MIN_RENDER_DIFFERENCE: f64 = 2.0;

#[test]
fn at9_identical_renders_fail_the_difference_check() {
    let a = vec![10u8, 20, 30, 40, 50, 60];
    assert_eq!(mean_abs_pixel_difference(&a, &a), 0.0);
    assert!(mean_abs_pixel_difference(&a, &a) < AT9_MIN_RENDER_DIFFERENCE);
    let b = vec![35u8, 45, 55, 15, 25, 85];
    assert_eq!(mean_abs_pixel_difference(&a, &b), 25.0);
    assert!(mean_abs_pixel_difference(&a, &b) >= AT9_MIN_RENDER_DIFFERENCE);
}

#[test]
fn at9_expansion_must_be_longer_than_the_short_prompt() {
    let short = "a red bicycle leaning against a brick fence";
    assert!(expansion_is_longer(
        short,
        "A red bicycle with a worn leather saddle leans against a weathered brick fence."
    ));
    assert!(!expansion_is_longer(short, short));
    assert!(!expansion_is_longer(short, "  a red bicycle  "));
    assert!(!expansion_is_longer(short, &format!("   {short}   ")));
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
        expansion_is_longer(&short, &expanded),
        "the expansion is not longer than the short prompt"
    );
    (label, short, expanded)
}

/// AT9: the short prompt and the PE expansion of it, rendered at 2048² with one seed. The gate is
/// that the expansion parses and is longer than the short prompt ([`pe_expansion`]) and that both
/// renders complete at the requested size; both PNGs are written for human review. "The expanded
/// render carries more detail" is not machine-checked: an edge-energy comparison penalised the
/// shallow depth of field the expansion legitimately asks for, and Michael's visual review of the
/// two renders (2026-11-16) is the recorded evidence for it.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT + the PE evidence (MINIMAX_H3_IMAGE_PE_*) and CUDA; two 2048² renders"]
fn h3_image_pe_expanded_prompt_renders_more_detail_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        let (label, short, expanded) = pe_expansion();
        println!("AT9 PE tier {label}\n  short:    {short}\n  expanded: {expanded}");
        let spec = bf16_spec(false, Vec::new());
        let request = |prompt: &str| GenerationRequest {
            prompt: prompt.to_owned(),
            ..t2i(2048, 2048, 7104)
        };
        let plain = render_admitted(
            "at9-short-prompt",
            &spec,
            request(&short),
            free_device_bytes(),
        );
        let rich = render_admitted(
            "at9-expanded-prompt",
            &spec,
            request(&expanded),
            free_device_bytes(),
        );
        for render in [&plain, &rich] {
            assert_eq!((render.image.width, render.image.height), (2048, 2048));
        }
        let difference = mean_abs_pixel_difference(&plain.image.pixels, &rich.image.pixels);
        println!("AT9 mean per-channel difference short vs expanded: {difference:.2}");
        assert!(
            difference >= AT9_MIN_RENDER_DIFFERENCE,
            "AT9: the expanded-prompt render is (near-)identical to the short-prompt render \
             ({difference:.2} < {AT9_MIN_RENDER_DIFFERENCE}); the expansion did not reach the generator"
        );
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail_cuda"
        );
    }
}
