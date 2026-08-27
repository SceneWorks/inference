//! LTX-2.3 **packed-tier** real-weight GPU video-render validation (sc-9545, sc-9089 umbrella).
//!
//! Loads the pre-quantized MLX-packed `SceneWorks/ltx-2.3-mlx` q4 (and, when present, q8) tier
//! **directly from the split packed parts** (no dense bf16 staging) through the registered
//! `ltx_2_3_distilled` generator, and asserts it renders a **coherent, non-degenerate** short video —
//! the end-to-end proof that the sc-9417 packed-detect seam fired on the REAL remapped tier keys (a
//! silent dense fall-back would fail to load the u32-packed transformer weights, and a broken packed
//! forward would decode to solid-black / NaN frames).
//!
//! This is the story sc-9545 render AC that sc-9417 could not satisfy: the tier ships split
//! per-component safetensors (`transformer` / `connector` / `vae_decoder` / gemma shards) with the DiT
//! keys remapped (`to_out.0`↔`to_out`, `ff.net.0.proj`↔`ff.proj_in`, `ff.net.2`↔`ff.proj_out`,
//! `linear_1/2`↔`linear1/2`), ingested by `candle_gen_ltx::tier`.
//!
//! `#[ignore]`d (needs a real GPU + the cached packed tier). On the Windows/Blackwell box (v143 vcvars
//! with CUDA on PATH), point at the **q4 tier subdir** (the packed snapshot nests `gemma/`, `q4/`,
//! `q8/`; the gemma sibling is auto-resolved, or override via `LoadSpec::text_encoder` — sc-13749
//! deleted the `$LTX_GEMMA_DIR` env, which the tier path never read):
//!
//! ```text
//! set LTX_PACKED_Q4=D:\.cache\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\q4
//! set LTX_PACKED_Q8=...\q8    (optional)
//! cargo test -p candle-gen-ltx --features cuda --release --test integration packed_tier_validate:: -- --ignored --nocapture
//! ```
//! # LTX-**2.5** packed tiers (sc-18776)
//!
//! The LTX-2.3 render checks above are `--features cuda` and `#[ignore]`d, because rendering needs a
//! GPU and a cached tier. The LTX-2.5 checks below split into three groups by what they actually
//! need, so the ones that need nothing run on **every** CI lane:
//!
//! * [`ltx25_validation`] — always on, no weights, no GPU. It builds structurally-exact synthetic
//!   bundles in a tempdir and proves each way a bundle can be wrong is refused with the *typed*
//!   error for that fault. This is where the "fails loudly rather than loading and rendering noise"
//!   AC is actually enforced on every push.
//! * [`ltx25_real_bundle`] — `#[ignore]`d, `LTX25_TIER_DIR`. Header-only validation of the real
//!   shipped q4 / q8 / bf16 bundles. No GPU: it reads safetensors headers, never tensor data, so all
//!   three ~40 GB tiers validate in seconds on a CPU box.
//! * [`ltx25_real_packed_forward`] — `#[ignore]`d, `LTX25_TIER_DIR`. Loads the **real packed q4**
//!   DiffVAE decoder and runs a forward, proving the affine triples decode to finite, non-degenerate
//!   activations rather than to noise.

#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, MemoryBehaviorRoute, MemoryMode,
    MemoryNumericTier, MemoryRunOutcome, MemoryStrategy, Precision, Progress, Quant, WeightsSource,
};

/// Basic per-frame non-degeneracy: the frame is not solid-black / constant (a broken packed forward —
/// NaN or zeroed activations — decodes to a flat frame) and has some spread of pixel values.
#[cfg(feature = "cuda")]
fn assert_frame_coherent(img: &Image, tag: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{tag}: RGB buffer size mismatch"
    );
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / img.pixels.len() as f64;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / img.pixels.len() as f64;
    eprintln!(
        "[{tag}] {}x{} pixel min={min} max={max} mean={mean:.1} std={:.1}",
        img.width,
        img.height,
        var.sqrt()
    );
    assert!(
        max > min + 16,
        "{tag}: frame is (near-)constant [{min}, {max}] — packed forward likely degenerate (black)"
    );
    assert!(
        var.sqrt() > 8.0,
        "{tag}: pixel std {:.1} too low — degenerate frame",
        var.sqrt()
    );
}

#[cfg(feature = "cuda")]
fn render_tier(env: &str, tag: &str) {
    let Ok(dir) = std::env::var(env) else {
        eprintln!("SKIP {tag}: set {env} to the packed tier subdir (e.g. …/q4)");
        return;
    };
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)));
    let gen = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx_2_3_distilled registered");

    // Short + low-res + the baked distilled step schedule (8) to bound time + VRAM for the 22B model:
    // 9 frames (2 latent frames), 256×256, seed 42.
    let req = GenerationRequest {
        prompt: "a fluffy cat walking across a sunny garden, gentle camera pan, cinematic".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(42),
        frames: Some(9),
        sampler: Some("rectified-flow".into()),
        ..Default::default()
    };

    let t = Instant::now();
    let mut on_progress = |_p: Progress| {};
    let out = gen
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("{tag}: packed-tier generate failed: {e}"));
    let secs = t.elapsed().as_secs_f32();
    eprintln!("[{tag}] load+render wall-clock {secs:.1}s (cold: includes packed tier load)");

    let (frames, fps) = match out {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        _ => panic!("{tag}: expected video, got images"),
    };
    assert!(!frames.is_empty(), "{tag}: no frames rendered");
    eprintln!("[{tag}] {} frame(s) @ {fps}fps", frames.len());
    for (i, f) in frames.iter().enumerate() {
        assert_frame_coherent(f, &format!("{tag}#{i}"));
    }

    // Write the first + middle frames next to temp so they can be eyeballed.
    for &i in &[0usize, frames.len() / 2] {
        if let Some(buf) =
            image::RgbImage::from_raw(frames[i].width, frames[i].height, frames[i].pixels.clone())
        {
            let out_path = std::env::temp_dir().join(format!("ltx_packed_{tag}_frame{i:03}.png"));
            let _ = buf.save(&out_path);
            eprintln!("[{tag}] wrote {}", out_path.display());
        }
    }
}

/// The q4 packed tier renders a coherent short video straight from the split packed parts (the primary
/// sc-9545 deliverable — the sc-9417 render AC).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_renders_coherent_video() {
    render_tier("LTX_PACKED_Q4", "q4");
}

/// SC-20772's exact Candle q4 I2V memory cell. This is intentionally separate from the generic
/// packed render above: it proves the selected bounded-decode request control, fixed-strength
/// fitted reference, and the actual q4 loader execute together rather than only agreeing in a
/// weights-free fixture. It is Windows/CUDA-only and deliberately not a calibration run.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_i2v_memory_route_renders() {
    let dir = std::env::var("LTX_PACKED_Q4")
        .expect("set LTX_PACKED_Q4 to the packed q4 tier subdirectory");
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(dir)));
    spec.quantize = Some(Quant::Q4);
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx q4 generator");
    let contract = generator
        .memory_strategy_contract()
        .expect("the exact q4 split artifact publishes its I2V memory contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::BoundedDecode,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("image_to_video".into()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("reference:image:768x512:strength:3f800000".into()),
        },
    )
    .expect("q4 bounded decode selection");
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 97;

    let mut request = GenerationRequest {
        prompt: "a red fox walking through snowy pines, slow dolly shot".into(),
        width: 768,
        height: 512,
        count: 1,
        seed: Some(42),
        frames: Some(97),
        fps: Some(24),
        sampler: Some("euler".into()),
        conditioning: vec![candle_gen::gen_core::Conditioning::Reference {
            image: Image {
                width: 768,
                height: 512,
                pixels: vec![127; 768 * 512 * 3],
            },
            strength: Some(1.0),
        }],
        ..Default::default()
    };
    {
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("q4 I2V admission")
            .expect("memory scope");
        scope
            .configure_request(&mut request)
            .expect("bind request controls");
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("scope cleanup");
    }
    assert!(request
        .memory
        .as_ref()
        .is_some_and(|memory| memory.tile_vae_decode));
    let mut on_progress = |_p: Progress| {};
    let output = generator
        .generate(&request, &mut on_progress)
        .expect("q4 I2V render with selected bounded decode");
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(fps, 24);
    assert_eq!(frames.len(), 97);
    assert_frame_coherent(&frames[0], "q4-i2v#0");
}

/// SC-20773's ordered first/last-frame counterpart. The provider VAE-encodes both keyframes at
/// both LTX stages, so this proves the four-encode carrier and selected bounded decode run through
/// the actual packed q4 CUDA loader rather than only the weights-free admission fixture.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_first_last_memory_route_renders() {
    let dir = std::env::var("LTX_PACKED_Q4")
        .expect("set LTX_PACKED_Q4 to the packed q4 tier subdirectory");
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(dir)));
    spec.quantize = Some(Quant::Q4);
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx q4 generator");
    let contract = generator
        .memory_strategy_contract()
        .expect("the exact q4 split artifact publishes its first-last memory contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::BoundedDecode,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("first_last_frame".into()),
            reference_count: 2,
            use_pid: false,
            has_phases: false,
            overlay: Some(
                "keyframe:first:image:768x512:frame:0:strength:3f800000+keyframe:last:image:768x512:frame:-1:strength:3f800000".into(),
            ),
        },
    )
    .expect("q4 first-last bounded decode selection");
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 97;

    let first = Image {
        width: 768,
        height: 512,
        pixels: vec![96; 768 * 512 * 3],
    };
    let last = Image {
        width: 768,
        height: 512,
        pixels: vec![160; 768 * 512 * 3],
    };
    let mut request = GenerationRequest {
        prompt: "a red fox crossing a snowy clearing, slow dolly shot".into(),
        width: 768,
        height: 512,
        count: 1,
        seed: Some(42),
        frames: Some(97),
        fps: Some(24),
        sampler: Some("euler".into()),
        conditioning: vec![
            candle_gen::gen_core::Conditioning::Keyframe {
                image: first,
                frame_idx: 0,
                strength: 1.0,
            },
            candle_gen::gen_core::Conditioning::Keyframe {
                image: last,
                frame_idx: -1,
                strength: 1.0,
            },
        ],
        ..Default::default()
    };
    {
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("q4 first-last admission")
            .expect("memory scope");
        scope
            .configure_request(&mut request)
            .expect("bind request controls");
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("scope cleanup");
    }
    let mut on_progress = |_p: Progress| {};
    let output = generator
        .generate(&request, &mut on_progress)
        .expect("q4 first-last render with selected bounded decode");
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(fps, 24);
    assert_eq!(frames.len(), 97);
    assert_frame_coherent(&frames[0], "q4-first-last#0");
    assert_frame_coherent(frames.last().expect("last frame"), "q4-first-last#last");
}

/// SC-20775's two-clip bridge witness. It exercises the packed q4 CUDA loader with the two
/// ordered IC-LoRA clip endpoints and bounded-decode scope, so a receipt-only fixture cannot
/// accidentally advertise a route the physical provider no longer executes.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_video_bridge_memory_route_renders() {
    let dir = std::env::var("LTX_PACKED_Q4")
        .expect("set LTX_PACKED_Q4 to the packed q4 tier subdirectory");
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(dir)));
    spec.quantize = Some(Quant::Q4);
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx q4 generator");
    let contract = generator
        .memory_strategy_contract()
        .expect("the exact q4 split artifact publishes its bridge memory contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::BoundedDecode,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_bridge".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some(
                "clip:append:frames:97:image:768x512:frame:0:strength:3f800000+clip:append:frames:97:image:768x512:frame:-1:strength:3f800000".into(),
            ),
        },
    )
    .expect("q4 bridge bounded decode selection");
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 97;

    let left = Image {
        width: 768,
        height: 512,
        pixels: vec![96; 768 * 512 * 3],
    };
    let right = Image {
        width: 768,
        height: 512,
        pixels: vec![160; 768 * 512 * 3],
    };
    let mut request = GenerationRequest {
        prompt: "a red fox walks from one snowy clearing into another, continuous dolly shot"
            .into(),
        width: 768,
        height: 512,
        count: 1,
        seed: Some(42),
        frames: Some(97),
        fps: Some(24),
        sampler: Some("euler".into()),
        conditioning: vec![
            candle_gen::gen_core::Conditioning::VideoClip {
                frames: vec![left; 97],
                frame_idx: 0,
                strength: 1.0,
            },
            candle_gen::gen_core::Conditioning::VideoClip {
                frames: vec![right; 97],
                frame_idx: -1,
                strength: 1.0,
            },
        ],
        ..Default::default()
    };
    {
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("q4 bridge admission")
            .expect("memory scope");
        scope
            .configure_request(&mut request)
            .expect("bind ordered bridge controls");
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("scope cleanup");
    }
    assert!(request
        .memory
        .as_ref()
        .is_some_and(|memory| memory.tile_vae_decode));
    let mut on_progress = |_p: Progress| {};
    let output = generator
        .generate(&request, &mut on_progress)
        .expect("q4 bridge render with selected bounded decode");
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(fps, 24);
    assert_eq!(frames.len(), 97);
    assert_frame_coherent(&frames[0], "q4-bridge#0");
    assert_frame_coherent(frames.last().expect("last frame"), "q4-bridge#last");
}

/// SC-20775's single-clip extend witness — the bridge witness above covers two ordered IC-LoRA
/// endpoints, this one covers the one-clip append the extend route actually admits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_extend_clip_memory_route_renders() {
    let dir = std::env::var("LTX_PACKED_Q4")
        .expect("set LTX_PACKED_Q4 to the packed q4 tier subdirectory");
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(dir)));
    spec.quantize = Some(Quant::Q4);
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx q4 generator");
    let contract = generator
        .memory_strategy_contract()
        .expect("the exact q4 split artifact publishes its extend memory contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::BoundedDecode,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("extend_clip".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: Some("clip:append:frames:97:image:768x512:frame:0:strength:3f800000".into()),
        },
    )
    .expect("q4 extend bounded decode selection");
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 97;

    let source = Image {
        width: 768,
        height: 512,
        pixels: vec![96; 768 * 512 * 3],
    };
    let mut request = GenerationRequest {
        prompt: "a red fox keeps walking through the same snowy clearing, continuous dolly shot"
            .into(),
        width: 768,
        height: 512,
        count: 1,
        seed: Some(42),
        frames: Some(97),
        fps: Some(24),
        sampler: Some("euler".into()),
        conditioning: vec![candle_gen::gen_core::Conditioning::VideoClip {
            frames: vec![source; 97],
            frame_idx: 0,
            strength: 1.0,
        }],
        ..Default::default()
    };
    {
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("q4 extend admission")
            .expect("memory scope");
        scope
            .configure_request(&mut request)
            .expect("bind the appended source clip");
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("scope cleanup");
    }
    assert!(request
        .memory
        .as_ref()
        .is_some_and(|memory| memory.tile_vae_decode));
    let mut on_progress = |_p: Progress| {};
    let output = generator
        .generate(&request, &mut on_progress)
        .expect("q4 extend render with selected bounded decode");
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(fps, 24);
    assert_eq!(frames.len(), 97);
    assert_frame_coherent(&frames[0], "q4-extend#0");
    assert_frame_coherent(frames.last().expect("last frame"), "q4-extend#last");
}

/// sc-20799's replace-person witness. `replace_person` is the only admitted route whose carrier is
/// a masked `ControlClip` plus an ordered character `MultiReference`, and the only one whose
/// receipt carries axes the geometry cannot reconstruct (mask blend weight, replacement mode). It
/// was implemented and advertised but had no admission arm at all, so nothing physical proved the
/// packed q4 loader could execute a request the memory contract admits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q4 (packed q4 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q4_replace_person_memory_route_renders() {
    let dir = std::env::var("LTX_PACKED_Q4")
        .expect("set LTX_PACKED_Q4 to the packed q4 tier subdirectory");
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(dir)));
    spec.quantize = Some(Quant::Q4);
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load("ltx_2_3_distilled", &spec)
        .expect("ltx q4 generator");
    let contract = generator
        .memory_strategy_contract()
        .expect("the exact q4 split artifact publishes its replace-person memory contract");
    let mut context = candle_gen::gen_core::standard_memory_behavior_context(
        contract,
        MemoryStrategy::BoundedDecode,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("replace_person".into()),
            reference_count: 2,
            use_pid: false,
            has_phases: false,
            // mode:1 = FullPersonKeepOutfit; mask 3f800000 = 1.0.
            overlay: Some(
                "clip:replace:frames:97:image:768x512:frame:0:mode:1:mask:3f800000+reference:sheet:2:image:768x512".into(),
            ),
        },
    )
    .expect("q4 replace-person bounded decode selection");
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 97;

    let plate = Image {
        width: 768,
        height: 512,
        pixels: vec![96; 768 * 512 * 3],
    };
    let mask = Image {
        width: 768,
        height: 512,
        pixels: vec![255; 768 * 512 * 3],
    };
    let character = |pixel: u8| Image {
        width: 768,
        height: 512,
        pixels: vec![pixel; 768 * 512 * 3],
    };
    let mut request = GenerationRequest {
        prompt: "replace the walking figure with the referenced character, same framing".into(),
        width: 768,
        height: 512,
        count: 1,
        seed: Some(42),
        frames: Some(97),
        fps: Some(24),
        sampler: Some("euler".into()),
        conditioning: vec![
            candle_gen::gen_core::Conditioning::ControlClip {
                frames: vec![plate; 97],
                mask: vec![mask; 97],
                masking_strength: 1.0,
                start_frame: 0,
                mode: candle_gen::gen_core::ReplacementMode::FullPersonKeepOutfit,
            },
            candle_gen::gen_core::Conditioning::MultiReference {
                images: vec![character(48), character(200)],
            },
        ],
        ..Default::default()
    };
    {
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("q4 replace-person admission")
            .expect("memory scope");
        scope
            .configure_request(&mut request)
            .expect("bind the masked clip and ordered character sheet");
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("scope cleanup");
    }
    assert!(request
        .memory
        .as_ref()
        .is_some_and(|memory| memory.tile_vae_decode));
    let mut on_progress = |_p: Progress| {};
    let output = generator
        .generate(&request, &mut on_progress)
        .expect("q4 replace-person render with selected bounded decode");
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(fps, 24);
    assert_eq!(frames.len(), 97);
    assert_frame_coherent(&frames[0], "q4-replace-person#0");
    assert_frame_coherent(frames.last().expect("last frame"), "q4-replace-person#last");
}

/// The q8 packed tier renders a coherent short video (double-quant Q8_0 path); only runs when the q8
/// tier is present locally.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs LTX_PACKED_Q8 (packed q8 tier subdir) + a CUDA GPU; run with --features cuda --ignored"]
fn packed_q8_renders_coherent_video() {
    render_tier("LTX_PACKED_Q8", "q8");
}

// =================================================================================================
// LTX-2.5 packed tier fixtures (sc-18776)
// =================================================================================================

/// Builds **structurally exact** synthetic LTX-2.5 bundles in a tempdir.
///
/// "Structurally exact" is the whole point: the files are tiny, but every property
/// `Ltx25Tier::validate` reads is the property the real bundle has — the `sceneworks_tier` /
/// `model_version` stamps, the `split_model.json` schema, the `{base}.weight` / `.scales` /
/// `.biases` triple, the `U32` code payload, and the `[out, in·bits/32]` × `[out, in/group]` shape
/// pair. A fixture that only approximated those would let a check pass for the wrong reason.
///
/// The manifest is **generated from the same spec that writes the files**, so the default bundle is
/// valid by construction and every negative test perturbs exactly one thing. That ordering matters:
/// a hand-written manifest drifts from the fixtures it describes, and the drift then looks like a
/// caught fault.
mod ltx25_fixture {
    use std::path::{Path, PathBuf};

    /// The bit widths the fixtures pack at, mirroring the shipped tiers.
    pub const Q4: usize = 4;
    pub const Q8: usize = 8;
    /// The affine group width every shipped LTX tier uses.
    pub const GROUP: usize = 64;
    /// The version an LTX-2.5 bundle declares.
    pub const V25: &str = "2.5.0";

    /// One tensor in a fixture file.
    struct TensorSpec {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
    }

    impl TensorSpec {
        fn width(&self) -> usize {
            match self.dtype {
                "U32" => 4,
                "BF16" => 2,
                other => panic!("fixture: unhandled dtype {other}"),
            }
        }

        fn bytes(&self) -> usize {
            self.shape.iter().product::<usize>() * self.width()
        }
    }

    /// Write a valid safetensors file: 8-byte little-endian header length, the JSON header, then the
    /// (all-zero) data block. Validation reads headers only, so the payload's *content* is never
    /// consulted — but its **length** is, because a header whose offsets run past the file is a
    /// different fault than the ones under test.
    fn write_safetensors(path: &Path, metadata: &[(String, String)], tensors: &[TensorSpec]) {
        let mut header = serde_json::Map::new();
        let meta: serde_json::Map<String, serde_json::Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        header.insert("__metadata__".into(), serde_json::Value::Object(meta));
        let mut offset = 0_usize;
        for t in tensors {
            let end = offset + t.bytes();
            header.insert(
                t.name.clone(),
                serde_json::json!({
                    "dtype": t.dtype,
                    "shape": t.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let text = serde_json::to_string(&serde_json::Value::Object(header)).expect("header json");
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes.resize(bytes.len() + offset, 0_u8);
        std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture dir");
        std::fs::write(path, bytes).expect("write fixture");
    }

    /// How a component's weights are stored in the fixture.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Storage {
        /// `n` complete affine triples at the bundle's declared width.
        Packed(usize),
        /// `n` complete affine triples at an *overridden* width — the "component from another
        /// tier" fixture.
        PackedAt(usize, usize),
        /// No packed weights.
        Dense,
        /// One triple missing its `.biases` leg.
        MissingBiases,
        /// One triple whose codes are stored as `BF16` instead of `U32`.
        FloatCodes,
    }

    /// One component of the synthetic bundle.
    #[derive(Clone)]
    pub struct Comp {
        pub name: &'static str,
        pub storage: Storage,
        pub dense_reason: Option<(&'static str, &'static str)>,
        /// Extra dense tensors beyond the one every component carries — the knob the
        /// tensor-count fixture turns.
        pub extra_dense: usize,
        /// Overrides for this component's own stamps.
        pub tier_stamp: Option<Option<&'static str>>,
        pub version_stamp: Option<&'static str>,
    }

    impl Comp {
        fn dense(name: &'static str, reason: (&'static str, &'static str)) -> Self {
            Comp {
                name,
                storage: Storage::Dense,
                dense_reason: Some(reason),
                extra_dense: 0,
                tier_stamp: None,
                version_stamp: None,
            }
        }

        fn packed(name: &'static str, n: usize) -> Self {
            Comp {
                name,
                storage: Storage::Packed(n),
                dense_reason: None,
                extra_dense: 0,
                tier_stamp: None,
                version_stamp: None,
            }
        }
    }

    /// The whole bundle, as a spec the writer and the manifest are both generated from.
    #[derive(Clone)]
    pub struct BundleSpec {
        pub tier: &'static str,
        pub quantized: bool,
        pub bits: usize,
        pub group: usize,
        /// The group width written into the manifest, when it must differ from the one the fixtures
        /// were packed at.
        pub manifest_group: Option<usize>,
        pub model_version: &'static str,
        pub components: Vec<Comp>,
    }

    /// The reasons the shipped q4 tier actually declares, so the fixture exercises the same strings
    /// the real manifest carries rather than a placeholder.
    const NO_LINEARS: (&str, &str) = (
        "no-linear-weights",
        "no rank-2 Linear weights: every weight is a convolution kernel, a norm or a per-channel \
         statistic, and MLX has no quantized convolution",
    );
    const NO_PORT: (&str, &str) = (
        "no-mlx-port",
        "this crate has no MLX port that can run these weights yet",
    );
    const BELOW_BAR: (&str, &str) = (
        "below-quality-bar",
        "quantizing this component at this tier's width is structurally fine and measurably too \
         lossy to ship",
    );

    impl BundleSpec {
        /// A valid `q4` bundle: the same twelve components, the same packed/dense split, and the
        /// same declared reasons as the shipped q4 tier.
        pub fn q4() -> Self {
            BundleSpec {
                tier: "q4",
                quantized: true,
                bits: Q4,
                group: GROUP,
                manifest_group: None,
                model_version: V25,
                components: vec![
                    Comp::packed("transformer", 2),
                    // Dense at q4 by measurement, exactly as the shipped tier ships it.
                    Comp::dense("text_encoder", BELOW_BAR),
                    Comp::packed("connector", 1),
                    Comp::dense("vae_decoder", NO_LINEARS),
                    Comp::dense("vae_encoder", NO_LINEARS),
                    Comp::dense("diffusion_vae_encoder", NO_LINEARS),
                    Comp::packed("vae_diffusion_decoder", 1),
                    Comp::dense("audio_vae", NO_LINEARS),
                    Comp::dense("vocoder", NO_LINEARS),
                    Comp::dense("spatial_upsampler", NO_LINEARS),
                    Comp::dense("temporal_upsampler", NO_LINEARS),
                    Comp::dense("duration_head", NO_PORT),
                ],
            }
        }

        /// A valid dense `bf16` bundle — nothing packed, nothing to justify.
        pub fn bf16() -> Self {
            let mut spec = BundleSpec::q4();
            spec.tier = "bf16";
            spec.quantized = false;
            for c in &mut spec.components {
                c.storage = Storage::Dense;
                c.dense_reason = Some(("dense-tier", "the bf16 tier is dense by definition"));
            }
            spec
        }

        /// The component with this manifest name, for a test to perturb.
        pub fn comp(&mut self, name: &str) -> &mut Comp {
            self.components
                .iter_mut()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("fixture has no `{name}` component"))
        }

        /// Drop a component's manifest row **and** its file, as a bundle that never shipped it.
        pub fn without(mut self, name: &str) -> Self {
            self.components.retain(|c| c.name != name);
            self
        }

        /// Write the bundle under `root` and return the tier directory.
        pub fn build(&self, root: &Path) -> PathBuf {
            let dir = root.join(self.tier);
            std::fs::create_dir_all(&dir).expect("create tier dir");
            let mut rows = Vec::new();
            for comp in &self.components {
                let file = format!("{}.safetensors", comp.name);
                let (tensors, packed) = self.tensors_for(comp);
                let mut metadata = vec![(
                    "config".to_string(),
                    // Every 2.5 component carries its own config; the reader only needs it to be
                    // valid JSON, and a real one is an object.
                    serde_json::json!({ "component": comp.name }).to_string(),
                )];
                match comp.tier_stamp {
                    // Absent: the "component carries no stamp" fixture.
                    Some(None) => {}
                    Some(Some(stamp)) => {
                        metadata.push(("sceneworks_tier".to_string(), stamp.to_string()))
                    }
                    None => metadata.push(("sceneworks_tier".to_string(), self.tier.to_string())),
                }
                metadata.push((
                    "model_version".to_string(),
                    comp.version_stamp.unwrap_or(self.model_version).to_string(),
                ));
                write_safetensors(&dir.join(&file), &metadata, &tensors);
                let mut row = serde_json::Map::new();
                row.insert("name".into(), comp.name.into());
                row.insert("file".into(), file.into());
                row.insert("tensors".into(), tensors.len().into());
                row.insert("quantized_linears".into(), packed.into());
                if let Some((id, detail)) = comp.dense_reason {
                    row.insert("dense_reason".into(), id.into());
                    row.insert("dense_reason_detail".into(), detail.into());
                }
                rows.push(serde_json::Value::Object(row));
            }
            let manifest = serde_json::json!({
                "format": "split",
                "model_version": self.model_version,
                "variant": "distilled",
                "tier": self.tier,
                "quantized": self.quantized,
                "quantization_bits": self.bits,
                "quantization_group_size": self.manifest_group.unwrap_or(self.group),
                "components": self.components.iter().map(|c| c.name).collect::<Vec<_>>(),
                "component_detail": rows,
            });
            std::fs::write(
                dir.join("split_model.json"),
                serde_json::to_string_pretty(&manifest).expect("manifest json"),
            )
            .expect("write manifest");
            dir
        }

        /// `(tensors, packed_count)` for one component.
        fn tensors_for(&self, comp: &Comp) -> (Vec<TensorSpec>, usize) {
            // out=2, in=GROUP → scales/biases are [2, 1] and the codes are [2, in*bits/32].
            let (out, inp) = (2_usize, GROUP);
            let triple = |base: String, bits: usize, code_dtype: &'static str, biases: bool| {
                let cols = inp * bits / 32;
                let mut out_v = vec![
                    TensorSpec {
                        name: format!("{base}.weight"),
                        dtype: code_dtype,
                        // A BF16 "code" payload keeps the same element count, so only the dtype
                        // distinguishes it — which is the fault being fixtured.
                        shape: vec![out, cols],
                    },
                    TensorSpec {
                        name: format!("{base}.scales"),
                        dtype: "BF16",
                        shape: vec![out, inp / GROUP],
                    },
                ];
                if biases {
                    out_v.push(TensorSpec {
                        name: format!("{base}.biases"),
                        dtype: "BF16",
                        shape: vec![out, inp / GROUP],
                    });
                }
                out_v
            };
            let (mut tensors, packed) = match comp.storage {
                Storage::Dense => (Vec::new(), 0),
                Storage::Packed(n) => (
                    (0..n)
                        .flat_map(|i| triple(format!("blocks.{i}.proj"), self.bits, "U32", true))
                        .collect(),
                    n,
                ),
                Storage::PackedAt(n, bits) => (
                    (0..n)
                        .flat_map(|i| triple(format!("blocks.{i}.proj"), bits, "U32", true))
                        .collect(),
                    n,
                ),
                Storage::MissingBiases => {
                    (triple("blocks.0.proj".into(), self.bits, "U32", false), 1)
                }
                Storage::FloatCodes => (triple("blocks.0.proj".into(), self.bits, "BF16", true), 1),
            };
            // One dense tensor every component carries, plus any extras a fixture asks for.
            for i in 0..=comp.extra_dense {
                tensors.push(TensorSpec {
                    name: format!("norm.{i}.weight"),
                    dtype: "BF16",
                    shape: vec![out],
                });
            }
            (tensors, packed)
        }
    }
}

/// **The bullet-2 enforcement point.** Always on: no weights, no GPU, no env var.
///
/// Each test perturbs exactly one property of an otherwise-valid bundle and asserts the *typed*
/// error for that fault. Asserting the variant rather than a message substring is deliberate — a
/// substring assertion passes when a different check fires with similar wording, which is precisely
/// the failure mode that lets a validation gate rot into a tautology.
mod ltx25_validation {
    use super::ltx25_fixture::{BundleSpec, Storage, GROUP, Q8, V25};
    use candle_gen_ltx::tier::{Ltx25Component, Ltx25Tier, Ltx25TierError};

    /// Build `spec` in a fresh tempdir and validate it, returning the outcome and the guard that
    /// owns the tree (dropping it early would delete the bundle mid-test).
    fn validate(
        spec: &BundleSpec,
    ) -> (
        Result<candle_gen_ltx::tier::Ltx25TierReport, Ltx25TierError>,
        tempfile::TempDir,
    ) {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        let tier = Ltx25Tier::detect(&dir)
            .expect("a well-formed 2.5 manifest must parse")
            .expect("a 2.5 manifest must be detected as a 2.5 tier");
        (tier.validate(), root)
    }

    /// The positive control. Every negative test below is only meaningful because this one passes:
    /// it proves the fixture bundle is valid *before* one property is perturbed, so a red negative
    /// test is evidence about that property rather than about the fixture.
    #[test]
    fn a_well_formed_q4_bundle_validates() {
        let spec = BundleSpec::q4();
        let (result, _root) = validate(&spec);
        let report = result.expect("the fixture q4 bundle is valid by construction");
        assert_eq!(report.tier, "q4");
        assert_eq!(report.checked.len(), Ltx25Component::all().len());
        // 2 transformer + 1 connector + 1 diffusion decoder, matching the shipped split.
        assert_eq!(report.packed_total(), 4);
        // Every dense component names a reason — the whole-pipeline tier contract, observed rather
        // than assumed.
        for (name, reason) in report.dense_components() {
            assert!(
                reason.is_some(),
                "`{name}` is dense in a quantized tier and declares no reason"
            );
        }
        // The q4 text encoder is dense *by measurement*; that it reaches the report as a declared
        // exception (not as a silently-dense component) is the property under test.
        let te = report
            .checked
            .iter()
            .find(|c| c.name == "text_encoder")
            .expect("text_encoder is checked");
        assert_eq!(te.packed, 0);
        assert_eq!(te.dense_reason.as_deref(), Some("below-quality-bar"));
    }

    /// The dense tier validates too, with nothing packed anywhere.
    #[test]
    fn a_well_formed_bf16_bundle_validates_with_nothing_packed() {
        let spec = BundleSpec::bf16();
        let (result, _root) = validate(&spec);
        let report = result.expect("the fixture bf16 bundle is valid by construction");
        assert_eq!(report.packed_total(), 0);
    }

    // --- missing components -------------------------------------------------------------------

    /// The manifest names the component and the file is not there.
    #[test]
    fn a_missing_component_file_is_refused() {
        let spec = BundleSpec::q4();
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        std::fs::remove_file(dir.join("spatial_upsampler.safetensors")).expect("remove component");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest parses")
            .expect("2.5 tier");
        let err = tier
            .validate()
            .expect_err("a bundle missing a file is not loadable");
        assert!(
            matches!(&err, Ltx25TierError::MissingComponentFile { component, .. } if *component == "spatial_upsampler"),
            "{err}"
        );
    }

    /// The manifest never lists a component this engine needs. Distinct from the file being absent:
    /// the fix is a different one, so the error is a different one.
    #[test]
    fn a_component_absent_from_the_manifest_is_refused() {
        let spec = BundleSpec::q4().without("temporal_upsampler");
        let (result, _root) = validate(&spec);
        let err = result.expect_err("a bundle with no temporal upsampler is incomplete");
        assert!(
            matches!(&err, Ltx25TierError::MissingComponentEntry { component, .. } if *component == "temporal_upsampler"),
            "{err}"
        );
    }

    // --- wrong precision ----------------------------------------------------------------------

    /// **The core wrong-precision fixture.** A q8 component dropped into a q4 bundle keeps the same
    /// tensor count *and* the same packed-Linear count — the real q4 and q8 transformers both hold
    /// 6779 tensors and 1344 packed Linears — so every count-based check passes it. Only the
    /// per-triple shape identity catches it.
    #[test]
    fn a_q8_component_inside_a_q4_bundle_is_refused_on_geometry_alone() {
        let mut spec = BundleSpec::q4();
        spec.comp("transformer").storage = Storage::PackedAt(2, Q8);
        let (result, _root) = validate(&spec);
        let err = result.expect_err("a q8 file in a q4 tier is not a q4 tier");
        let Ltx25TierError::PackedGeometryMismatch {
            component,
            declared,
            shapes,
            ..
        } = &err
        else {
            panic!("expected a geometry mismatch, got {err}");
        };
        assert_eq!(*component, "transformer");
        assert_eq!(declared.bits, 4);
        assert_eq!(declared.group, GROUP);
        // The identity that fails: 32·weight_cols == bits·group·scales_cols. At q8 the left side is
        // exactly double, which is the whole signal.
        assert_eq!(
            32 * shapes.compared[1],
            2 * (declared.bits * declared.group * shapes.scales[1])
        );
    }

    /// A dense component substituted where the manifest declares packed Linears. This is what
    /// dropping the bf16 transformer into the q4 directory looks like: it would load, run, and
    /// silently need 38 GB instead of 11.
    #[test]
    fn a_dense_component_where_the_manifest_declares_packed_is_refused() {
        let mut spec = BundleSpec::q4();
        let comp = spec.comp("connector");
        comp.storage = Storage::Dense;
        // Keep the manifest's declaration honest about what it *expected*: the row is regenerated
        // from the spec, so re-declare the packed count the shipped tier has.
        let (result, _root) = {
            // Rebuild with a manifest that still claims one packed Linear.
            let root = tempfile::tempdir().expect("tempdir");
            let dir = spec.build(root.path());
            patch_manifest(&dir, "connector", "quantized_linears", 1.into());
            let tier = Ltx25Tier::detect(&dir)
                .expect("manifest parses")
                .expect("2.5 tier");
            (tier.validate(), root)
        };
        let err =
            result.expect_err("a dense connector is not the packed one the manifest declares");
        assert!(
            matches!(
                &err,
                Ltx25TierError::PackedCountMismatch { component, declared, actual, .. }
                    if *component == "connector" && *declared == 1 && *actual == 0
            ),
            "{err}"
        );
    }

    /// A packed tensor inside the tier that declares itself dense.
    #[test]
    fn a_packed_tensor_in_the_dense_tier_is_refused() {
        let mut spec = BundleSpec::bf16();
        spec.comp("transformer").storage = Storage::Packed(1);
        let (result, _root) = validate(&spec);
        let err = result.expect_err("the bf16 tier packs nothing");
        assert!(
            matches!(&err, Ltx25TierError::PackedTensorInDenseTier { component, .. } if *component == "transformer"),
            "{err}"
        );
    }

    /// Bit-packed codes stored at a float dtype. They *read* fine; every value is wrong.
    #[test]
    fn packed_codes_stored_as_a_float_are_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("transformer").storage = Storage::FloatCodes;
        let (result, _root) = validate(&spec);
        let err = result.expect_err("U32 codes read as BF16 decode to noise");
        assert!(
            matches!(&err, Ltx25TierError::PackedWeightDtype { dtype, .. } if *dtype == "BF16"),
            "{err}"
        );
    }

    /// An affine triple missing its third leg.
    #[test]
    fn an_incomplete_packed_triple_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("transformer").storage = Storage::MissingBiases;
        let (result, _root) = validate(&spec);
        let err = result.expect_err("an affine triple needs weight, scales and biases");
        assert!(
            matches!(
                &err,
                Ltx25TierError::IncompletePackedTriple { has_weight, has_biases, .. }
                    if *has_weight && !*has_biases
            ),
            "{err}"
        );
    }

    /// A group width the packed loaders cannot repack at. Left unchecked this is the quiet path to
    /// noise: at group 128 the bit-width identity resolves to a *different valid width*, so the
    /// repack would succeed and decode every weight wrongly.
    #[test]
    fn a_group_width_the_loaders_cannot_repack_at_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.manifest_group = Some(32);
        let (result, _root) = validate(&spec);
        let err = result.expect_err("the packed loaders repack at one fixed group");
        assert!(
            matches!(
                &err,
                Ltx25TierError::UnsupportedGroupSize { declared, supported, .. }
                    if *declared == 32 && *supported == GROUP
            ),
            "{err}"
        );
    }

    // --- provenance ---------------------------------------------------------------------------

    /// A component stamped for another tier.
    #[test]
    fn a_component_stamped_for_another_tier_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("vae_decoder").tier_stamp = Some(Some("q8"));
        let (result, _root) = validate(&spec);
        let err = result.expect_err("a q8-stamped file does not belong in the q4 tier");
        assert!(
            matches!(
                &err,
                Ltx25TierError::TierStampMismatch { component, expected, stamped, .. }
                    if *component == "vae_decoder" && expected == "q4" && stamped == "q8"
            ),
            "{err}"
        );
    }

    /// A component with no tier stamp at all.
    #[test]
    fn a_component_with_no_tier_stamp_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("audio_vae").tier_stamp = Some(None);
        let (result, _root) = validate(&spec);
        let err = result.expect_err("every shipped component is stamped with its tier");
        assert!(
            matches!(&err, Ltx25TierError::TierStampMissing { component, .. } if *component == "audio_vae"),
            "{err}"
        );
    }

    /// A component from a different release.
    #[test]
    fn a_component_from_another_release_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("vocoder").version_stamp = Some("2.4.0");
        let (result, _root) = validate(&spec);
        let err = result.expect_err("one bundle, one release");
        assert!(
            matches!(
                &err,
                Ltx25TierError::ModelVersionMismatch { component, expected, declared, .. }
                    if *component == "vocoder" && expected == V25 && declared == "2.4.0"
            ),
            "{err}"
        );
    }

    /// A file that holds more tensors than the manifest declares.
    #[test]
    fn a_tensor_count_that_disagrees_with_the_manifest_is_refused() {
        let spec = BundleSpec::q4();
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        // Shrink the *declaration* rather than the file: perturbing the manifest is the only way to
        // make the counts disagree without also changing something else the checks read.
        patch_manifest(&dir, "spatial_upsampler", "tensors", 99.into());
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest parses")
            .expect("2.5 tier");
        let err = tier
            .validate()
            .expect_err("the file is not the one the tier was built from");
        assert!(
            matches!(
                &err,
                Ltx25TierError::TensorCountMismatch { component, declared, actual, .. }
                    if *component == "spatial_upsampler" && *declared == 99 && *actual == 1
            ),
            "{err}"
        );
    }

    // --- the whole-pipeline tier contract (R5) -------------------------------------------------

    /// A component that is dense inside a quantized tier and says nothing about why. This is the
    /// check that stops "q4 transformer, bf16 quietly everything else" from shipping as `q4`.
    #[test]
    fn an_undeclared_dense_component_in_a_quantized_tier_is_refused() {
        let mut spec = BundleSpec::q4();
        spec.comp("vae_decoder").dense_reason = None;
        let (result, _root) = validate(&spec);
        let err = result.expect_err("a dense component inside q4 must justify itself");
        assert!(
            matches!(
                &err,
                Ltx25TierError::UndeclaredDenseComponent { component, tier }
                    if *component == "vae_decoder" && tier == "q4"
            ),
            "{err}"
        );
    }

    /// A `dense_reason` with no detail is not a justification. The detail is where the *evidence*
    /// lives (the q4 text encoder's exemption carries its measurement), so a bare id would let the
    /// contract be satisfied by a label.
    #[test]
    fn a_dense_reason_without_its_detail_is_refused() {
        let spec = BundleSpec::q4();
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        patch_manifest_remove(&dir, "text_encoder", "dense_reason_detail");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest parses")
            .expect("2.5 tier");
        let err = tier
            .validate()
            .expect_err("an exemption needs its evidence");
        assert!(
            matches!(&err, Ltx25TierError::UndeclaredDenseComponent { component, .. } if *component == "text_encoder"),
            "{err}"
        );
    }

    // --- layout selection (R1: 2.3 consumption unchanged) --------------------------------------

    /// A SceneWorks-converted **2.3** tree also ships a `split_model.json`. It must not be picked up
    /// as a 2.5 bundle, or the 2.3 path this story is required to leave alone would stop being
    /// reachable.
    ///
    /// Read from the **committed real** 2.3 manifest
    /// (`mlx-gen-ltx/tests/fixtures/ltx_2_3_split_model.json`), not from this file's own generator
    /// with the version flipped. That distinction is the whole test: the generator emits the 2.5
    /// schema, so a synthetic "2.3" manifest still carries `component_detail` and `tier` and would
    /// pass a `detect` that parsed the 2.5 schema before gating on the version. The real 2.3
    /// manifest carries **neither** — it is `format` / `model_version` / `components` / `source` /
    /// `variant` / `quantized` / `quantization_*` and nothing else — so it is the only fixture that
    /// can tell the two orderings apart.
    #[test]
    fn the_committed_real_2_3_manifest_is_not_detected_as_a_2_5_tier() {
        const REAL_2_3_MANIFEST: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_2_3_split_model.json"
        );
        let manifest = std::fs::read_to_string(REAL_2_3_MANIFEST)
            .unwrap_or_else(|e| panic!("read {REAL_2_3_MANIFEST}: {e}"));
        // Guard the fixture itself: if a future 2.3 manifest grows these keys, this test would
        // silently stop distinguishing the two parse orders.
        let value: serde_json::Value = serde_json::from_str(&manifest).expect("fixture is JSON");
        assert!(
            value.get("component_detail").is_none() && value.get("tier").is_none(),
            "the real 2.3 manifest must carry neither `component_detail` nor `tier`, or this test \
             no longer discriminates the parse order"
        );

        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("split_model.json"), &manifest).expect("write");
        assert!(
            Ltx25Tier::detect(root.path())
                .expect("a real 2.3 manifest is `not mine`, never a hard error")
                .is_none(),
            "a 2.3 tree must keep taking the 2.3 path"
        );
    }

    /// A manifest that declares no `model_version` at all — pre-`model_version` trees exist, and
    /// the shared resolver reads an undeclared version as the oldest layout. `Ok(None)`, not an
    /// error, and in particular not a demand for the 2.5 schema.
    #[test]
    fn a_manifest_without_a_model_version_is_not_a_2_5_tier() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            root.path().join("split_model.json"),
            r#"{"format":"split","components":["transformer"]}"#,
        )
        .expect("write");
        assert!(Ltx25Tier::detect(root.path())
            .expect("an undeclared version is `not mine`, never a hard error")
            .is_none());
    }

    /// A directory with no manifest is not a 2.5 tier, and that is not an error.
    #[test]
    fn a_directory_without_a_manifest_is_not_a_2_5_tier() {
        let root = tempfile::tempdir().expect("tempdir");
        assert!(Ltx25Tier::detect(root.path())
            .expect("no manifest is not a fault")
            .is_none());
    }

    /// A manifest that is present and broken is an **error**, never a quiet "not a 2.5 tier".
    /// Reporting `None` here would drop the tree onto a loader that picks files by name.
    #[test]
    fn a_malformed_manifest_is_an_error_not_a_silent_miss() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("split_model.json"), "{ not json").expect("write");
        let err = Ltx25Tier::detect(root.path()).expect_err("a broken manifest must be loud");
        assert!(err.to_string().contains("parse"), "{err}");
    }

    // --- defensive parsing ---------------------------------------------------------------------

    /// safetensors types every `__metadata__` value as a string, and converters that round-trip a
    /// manifest through that representation emit `"4"` where the schema says `4` (the lightx2v LoRA
    /// packs are the case on record). Such a bundle is well-formed and must load.
    #[test]
    fn string_encoded_manifest_numbers_and_flags_are_accepted() {
        let spec = BundleSpec::q4();
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        let path = dir.join("split_model.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        json["quantization_bits"] = "4".into();
        json["quantization_group_size"] = "64".into();
        json["quantized"] = "true".into();
        std::fs::write(&path, serde_json::to_string(&json).expect("json")).expect("write");
        let tier = Ltx25Tier::detect(&dir)
            .expect("string-encoded numbers parse")
            .expect("2.5 tier");
        let quant = tier
            .quant()
            .expect("a quantized tier declares its geometry");
        assert_eq!((quant.bits, quant.group), (4, GROUP));
        tier.validate()
            .expect("the bundle is otherwise unchanged and valid");
    }

    /// A manifest number that is neither a number nor a numeric string is an error, not a default.
    /// Defaulting here would pick a bit width, which is exactly the guess this reader refuses.
    #[test]
    fn an_unparseable_manifest_number_is_an_error_not_a_default() {
        let spec = BundleSpec::q4();
        let root = tempfile::tempdir().expect("tempdir");
        let dir = spec.build(root.path());
        let path = dir.join("split_model.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        json["quantization_bits"] = "four".into();
        std::fs::write(&path, serde_json::to_string(&json).expect("json")).expect("write");
        let err = Ltx25Tier::detect(&dir).expect_err("`four` is not a bit width");
        assert!(err.to_string().contains("quantization_bits"), "{err}");
    }

    // --- helpers --------------------------------------------------------------------------------

    /// Overwrite one field of one `component_detail` row.
    fn patch_manifest(dir: &std::path::Path, component: &str, key: &str, value: serde_json::Value) {
        edit_row(dir, component, |row| {
            row.insert(key.to_string(), value.clone());
        });
    }

    /// Remove one field from one `component_detail` row.
    fn patch_manifest_remove(dir: &std::path::Path, component: &str, key: &str) {
        edit_row(dir, component, |row| {
            row.remove(key);
        });
    }

    fn edit_row(
        dir: &std::path::Path,
        component: &str,
        mut edit: impl FnMut(&mut serde_json::Map<String, serde_json::Value>),
    ) {
        let path = dir.join("split_model.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read manifest"))
                .expect("parse manifest");
        let rows = json["component_detail"]
            .as_array_mut()
            .expect("component_detail array");
        let row = rows
            .iter_mut()
            .find(|r| r["name"] == component)
            .unwrap_or_else(|| panic!("manifest has no `{component}` row"));
        edit(row.as_object_mut().expect("row is an object"));
        std::fs::write(&path, serde_json::to_string_pretty(&json).expect("json"))
            .expect("write manifest");
    }
}

/// The **real** shipped bundles, validated header-only (`LTX25_TIER_DIR`).
///
/// `#[ignore]`d because it needs the built tiers, but it needs **no GPU and no tensor reads**: the
/// whole check walks safetensors headers, so all three ~40 GB bundles validate in seconds on a CPU
/// box. An unset env var is a hard failure, not a silent pass — `#[ignore]` is the only opt-out.
mod ltx25_real_bundle {
    use candle_gen_ltx::diff_vae::NaDiffusionDecoderConfig;
    use candle_gen_ltx::tier::{Ltx25Component, Ltx25Tier};

    /// The tier root (the directory holding `q4/`, `q8/`, `bf16/`).
    fn tier_root() -> std::path::PathBuf {
        match std::env::var_os("LTX25_TIER_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => panic!(
                "set LTX25_TIER_DIR to the built LTX-2.5 tier root (the directory holding \
                 q4/ q8/ bf16/)"
            ),
        }
    }

    fn validate_tier(name: &str) {
        let dir = tier_root().join(name);
        let tier = Ltx25Tier::detect(&dir)
            .unwrap_or_else(|e| panic!("{}: manifest: {e}", dir.display()))
            .unwrap_or_else(|| {
                panic!(
                    "{} is not a 2.5 tier (no split_model.json declaring a 2.5 model_version)",
                    dir.display()
                )
            });
        let report = tier
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        eprintln!(
            "[{name}] {} components, {} packed Linears total",
            report.checked.len(),
            report.packed_total()
        );
        for c in &report.checked {
            eprintln!(
                "  {:<22} tensors={:<5} packed={:<5} dense_reason={}",
                c.name,
                c.tensors,
                c.packed,
                c.dense_reason.as_deref().unwrap_or("-")
            );
        }
        // A skipped check is printed, never swallowed: the packed text encoder carries no
        // `model_version`, and a green run has to say so rather than imply it was checked.
        for skipped in &report.skipped {
            eprintln!("  SKIPPED: {skipped}");
        }
        assert_eq!(report.tier, name);
        assert_eq!(
            report.checked.len(),
            Ltx25Component::all().len(),
            "a complete 2.5 bundle carries every component"
        );
        // Every component this engine names must resolve to a file that is actually there.
        for component in Ltx25Component::all() {
            tier.file(*component)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    /// The q4 bundle validates, and its Gemma 4 text encoder is dense **by declared measurement** —
    /// the one exception to the whole-pipeline tier contract the shipped tier carries.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only, header-reads only"]
    fn the_real_q4_bundle_validates() {
        validate_tier("q4");
        let tier = Ltx25Tier::detect(&tier_root().join("q4"))
            .expect("manifest")
            .expect("2.5 tier");
        let report = tier.validate().expect("valid");
        let te = report
            .checked
            .iter()
            .find(|c| c.name == "text_encoder")
            .expect("the bundle carries a text encoder");
        assert_eq!(
            te.packed, 0,
            "sc-18775 measured q4 below the quality bar for this encoder and ships it dense"
        );
        assert_eq!(te.dense_reason.as_deref(), Some("below-quality-bar"));
        // The components the candle DiffVAE / DiT loaders now have to dequantize really are packed
        // here — asserted so a tier that quietly stopped packing them could not pass as q4.
        for name in ["transformer", "connector", "vae_diffusion_decoder"] {
            let c = report
                .checked
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} is checked"));
            assert!(c.packed > 0, "`{name}` must be packed in the q4 tier");
        }
    }

    /// The q8 bundle validates, and packs the text encoder the q4 tier leaves dense.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only, header-reads only"]
    fn the_real_q8_bundle_validates() {
        validate_tier("q8");
        let tier = Ltx25Tier::detect(&tier_root().join("q8"))
            .expect("manifest")
            .expect("2.5 tier");
        let report = tier.validate().expect("valid");
        let te = report
            .checked
            .iter()
            .find(|c| c.name == "text_encoder")
            .expect("the bundle carries a text encoder");
        assert!(
            te.packed > 0,
            "q8 passed the quality bar for this encoder, so the q8 tier packs it"
        );
    }

    /// The bf16 bundle validates and packs nothing.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only, header-reads only"]
    fn the_real_bf16_bundle_validates() {
        validate_tier("bf16");
        let tier = Ltx25Tier::detect(&tier_root().join("bf16"))
            .expect("manifest")
            .expect("2.5 tier");
        let report = tier.validate().expect("valid");
        assert_eq!(
            report.packed_total(),
            0,
            "the bf16 tier is dense by definition"
        );
    }

    /// **Per-component configs.** An LTX-2.5 bundle carries no single `config.json`: each component
    /// declares its own structure in its own `__metadata__`, and this is the tier reader resolving
    /// each one to the section the matching candle loader consumes.
    ///
    /// Asserted per component rather than as a blanket "every file carries a config", because that
    /// blanket claim is false and would have to be weakened into something that asserts nothing:
    /// the connector, both VAE encoders and the vocoder carry no `config` at all (they are weights
    /// the tier splits out of a component that does), and the text encoder carries `gemma_config`
    /// rather than `config`. Components with no config of their own are named here explicitly, so
    /// the set is a statement rather than an omission.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only, header-reads only"]
    fn every_real_component_resolves_the_config_its_loader_reads() {
        use candle_gen::gen_core::ltx_checkpoint::{
            CONV_VIDEO_VAE_CLASS, DIFFUSION_VIDEO_VAE_CLASS, LATENT_UPSAMPLER_CLASS,
            TRANSFORMER_CLASS,
        };

        let tier = Ltx25Tier::detect(&tier_root().join("q4"))
            .expect("manifest")
            .expect("2.5 tier");
        let meta = |c: Ltx25Component| {
            tier.component_metadata(c)
                .unwrap_or_else(|e| panic!("{}: {e}", c.id()))
        };
        let class = |value: &serde_json::Value| {
            value
                .get("_class_name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };

        // The DiT: `config.transformer`, the section `AvConfig::from_transformer_config` reads.
        let transformer = meta(Ltx25Component::Transformer);
        let section = transformer
            .section("transformer")
            .expect("the transformer declares `config.transformer`");
        assert_eq!(class(section).as_deref(), Some(TRANSFORMER_CLASS));

        // The two video VAEs are separate files with different decoders, and each says which it is.
        // Collapsing them into one slot would mean picking between two real, differently-shaped
        // decoders, so the distinguishing `_class_name` is the thing checked.
        let conv = meta(Ltx25Component::VaeDecoder);
        assert_eq!(
            class(conv.section("vae").expect("`config.vae`")).as_deref(),
            Some(CONV_VIDEO_VAE_CLASS)
        );
        let diffusion = meta(Ltx25Component::VaeDiffusionDecoder);
        assert_eq!(
            class(diffusion.section("vae").expect("`config.vae`")).as_deref(),
            Some(DIFFUSION_VIDEO_VAE_CLASS)
        );
        // And it parses into the config the decoder is actually built from.
        NaDiffusionDecoderConfig::from_checkpoint(
            tier.file(Ltx25Component::VaeDiffusionDecoder)
                .expect("diffusion decoder"),
        )
        .expect("the diffusion decoder's own config builds its loader config");

        // One file, both audio sections — the audio VAE and the vocoder ride together.
        let audio = meta(Ltx25Component::AudioVae);
        assert!(audio.section("audio_vae").is_some(), "`config.audio_vae`");
        assert!(audio.section("vocoder").is_some(), "`config.vocoder`");

        // The duration head re-declares the transformer dims it projects from, alongside its own.
        // Written as `expect` rather than `assert!` because `scripts/check_clock_assertions.py`
        // flags any `\w*duration\w*` identifier reaching an assert as a possible wall-clock
        // reading. There is no clock here — the component is simply called the duration head — and
        // contorting the check to dodge the regex would be worse than choosing the form that says
        // the same thing without one.
        let head = meta(Ltx25Component::DurationHead);
        head.section("duration_head")
            .expect("the head declares its own `config.duration_head`");
        head.section("transformer").expect(
            "the head projects from the transformer's cross-attention dims and re-declares them",
        );

        // Both upsamplers are the same class and differ only by which axis they scale — the one
        // field that decides which of the two a file is.
        for (component, temporal) in [
            (Ltx25Component::SpatialUpsampler, false),
            (Ltx25Component::TemporalUpsampler, true),
        ] {
            let m = meta(component);
            let cfg = m
                .config()
                .expect("an upsampler's config is bare at the root");
            assert_eq!(class(cfg).as_deref(), Some(LATENT_UPSAMPLER_CLASS));
            assert_eq!(
                cfg.get("temporal_upsample")
                    .and_then(serde_json::Value::as_bool),
                Some(temporal),
                "{} must declare temporal_upsample={temporal}",
                component.id()
            );
        }

        // The packed text encoder carries `gemma_config`, not `config` — the HF config it was
        // packed with.
        let te = meta(Ltx25Component::TextEncoder);
        assert!(te.gemma_config().is_some(), "`gemma_config`");
        assert!(
            te.config().is_none(),
            "a packed text encoder carries no LTX `config` section"
        );

        // The components the tier splits out of a parent, which therefore declare no config of
        // their own. Named so this set is a claim, not a gap.
        for component in [
            Ltx25Component::Connector,
            Ltx25Component::VaeEncoder,
            Ltx25Component::DiffusionVaeEncoder,
            Ltx25Component::Vocoder,
        ] {
            assert!(
                meta(component).config().is_none(),
                "`{}` is split out of a parent component and declares no config of its own",
                component.id()
            );
        }
    }

    /// **A real, perturbed component.** Copies the *actual* shipped q4 bundle into a scratch
    /// directory, then removes one component from the copy — the same fault the synthetic fixture
    /// covers, proved once against real bytes and a real manifest. Nothing under the tier root is
    /// ever written to: the small components are copied and the large ones hard-linked, so the
    /// scratch bundle reads as the real files without duplicating 41 GB.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only, header-reads only"]
    fn a_real_bundle_missing_a_real_component_is_refused() {
        let src = tier_root().join("q4");
        let scratch = tempfile::tempdir().expect("tempdir");
        let dir = scratch.path().join("q4");
        std::fs::create_dir_all(&dir).expect("scratch tier dir");
        // Copy the manifest plus the small components verbatim; hard-link the large ones so the
        // copy costs nothing and still reads as the real file.
        for entry in std::fs::read_dir(&src).expect("read tier") {
            let entry = entry.expect("dir entry");
            let target = dir.join(entry.file_name());
            let meta = entry.metadata().expect("entry metadata");
            if meta.len() < 8 * 1024 * 1024 {
                std::fs::copy(entry.path(), &target).expect("copy component");
            } else if std::fs::hard_link(entry.path(), &target).is_err() {
                std::fs::copy(entry.path(), &target).expect("copy large component");
            }
        }
        // The copy is valid before it is perturbed — otherwise the refusal below proves nothing.
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest")
            .expect("2.5 tier");
        tier.validate()
            .expect("the copied bundle is still the real, valid one");

        std::fs::remove_file(dir.join("spatial_upsampler.safetensors")).expect("perturb the copy");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest")
            .expect("2.5 tier");
        let err = tier
            .validate()
            .expect_err("a real bundle missing its duration head is not loadable");
        eprintln!("[q4-perturbed] {err}");
        assert!(err.to_string().contains("spatial_upsampler"), "{err}");
    }
}

/// **Real packed q4 weights through the real forward** (`LTX25_TIER_DIR`), on the CPU.
///
/// The DiffVAE decoder is the component this story made packed-capable: the shipped q4 and q8 tiers
/// store 137 of its Linears as MLX affine triples, and before this change the candle loader read
/// `{prefix}.weight` as a float — which *succeeds* on a bit-packed `U32` payload and decodes every
/// weight into noise. Header validation cannot catch that; only running the weights can.
///
/// Deliberately not a full text→video render: candle has no registered `ltx_2_5` generator yet
/// (sc-18778), so there is no 2.5 pipeline to render through. This exercises the seam that exists.
mod ltx25_real_packed_forward {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen_ltx::diff_vae::{NaDiffusionDecoder, NaDiffusionDecoderConfig};
    use candle_gen_ltx::tier::{Ltx25Component, Ltx25Tier};

    fn tier_dir(name: &str) -> std::path::PathBuf {
        match std::env::var_os("LTX25_TIER_DIR") {
            Some(dir) => std::path::PathBuf::from(dir).join(name),
            None => panic!(
                "set LTX25_TIER_DIR to the built LTX-2.5 tier root (the directory holding \
                 q4/ q8/ bf16/)"
            ),
        }
    }

    /// `max |x|` over a tensor, on the host.
    fn max_abs(t: &Tensor) -> f32 {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("host copy")
            .into_iter()
            .fold(0f32, |a, v| a.max(v.abs()))
    }

    /// Standard deviation over a tensor, on the host.
    fn std_dev(t: &Tensor) -> f32 {
        let v: Vec<f32> = t
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("host copy");
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// The real q4 DiffVAE decoder loads from its packed triples and its stage-4 features are
    /// finite and non-degenerate.
    ///
    /// `stage_features` is the decoder's first real pass over the latent: it runs `conv_in` (packed
    /// in this tier) and every `det_stages` block (packed attention and SwiGLU). If the affine
    /// triples were being repacked under the wrong geometry — or read as floats at all — this is
    /// where it shows, as `NaN`/`inf` or as an all-but-constant output.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only; no GPU, no render"]
    fn the_real_q4_diff_vae_decoder_loads_packed_and_forwards() {
        let dir = tier_dir("q4");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest")
            .expect("LTX25_TIER_DIR/q4 must be a 2.5 tier");
        // Validation first: the forward below is only evidence about the packed path if the bundle
        // it reads is the one the manifest describes.
        tier.validate().expect("the q4 bundle validates");
        let quant = tier
            .quant()
            .expect("the q4 tier declares its affine geometry");
        assert_eq!((quant.bits, quant.group), (4, 64));

        let path = tier
            .file(Ltx25Component::VaeDiffusionDecoder)
            .expect("the bundle ships a diffusion decoder");
        let cfg = NaDiffusionDecoderConfig::from_checkpoint(&path)
            .expect("the tier component carries its own `config.vae`");
        let device = Device::Cpu;
        let (body, stats) = tier
            .diff_vae_vb(DType::F32, &device)
            .expect("diffusion decoder builders");

        // The declaration is threaded in. Passing `None` here is the pre-fix behaviour and is now a
        // hard error, which the companion test below pins.
        let decoder = NaDiffusionDecoder::load_quantized(body, stats, &cfg, Some(quant))
            .expect("build the decoder from the REAL packed q4 triples");

        let [t, h, w] = decoder.config().min_latent_shape();
        let latent =
            Tensor::randn(0f32, 1f32, (1, cfg.in_channels, t, h, w), &device).expect("latent");
        let (features, _) = decoder
            .stage_features(&latent)
            .expect("forward the packed decoder");

        let hi = max_abs(&features);
        let sd = std_dev(&features);
        eprintln!(
            "[q4-diffvae] stage-4 features {:?} max|x|={hi:.4} std={sd:.4}",
            features.dims()
        );
        assert!(
            hi.is_finite(),
            "packed q4 stage-4 features contain a non-finite value — the affine triples did not \
             decode"
        );
        // Both bounds matter and they fail differently: an all-zero output is a stack that never
        // bound its weights, and a huge one is codes being read as floats.
        assert!(hi > 0.0, "stage-4 features are identically zero");
        assert!(
            sd > 1e-4,
            "stage-4 feature std {sd:.6} is degenerate — the packed weights decoded to a constant"
        );
        assert!(
            hi < 1.0e4,
            "stage-4 features reach {hi:.1}; U32 codes reinterpreted as floats blow up like this"
        );
    }

    /// The pre-fix behaviour, pinned as a refusal: loading the real packed q4 decoder **without**
    /// the tier's geometry must fail loudly rather than read the `U32` codes as floats.
    ///
    /// This is the exact fault the story names — "loading and rendering noise" — asserted against
    /// the real shipped weights rather than a fixture.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only; no GPU, no render"]
    fn the_real_q4_diff_vae_decoder_refuses_to_load_as_dense() {
        let dir = tier_dir("q4");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest")
            .expect("2.5 tier");
        let path = tier
            .file(Ltx25Component::VaeDiffusionDecoder)
            .expect("diffusion decoder");
        let cfg = NaDiffusionDecoderConfig::from_checkpoint(&path).expect("config");
        let device = Device::Cpu;
        let (body, stats) = tier.diff_vae_vb(DType::F32, &device).expect("builders");
        let err = NaDiffusionDecoder::load(body, stats, &cfg)
            .expect_err("a packed component must not load through the dense entry point");
        eprintln!("[q4-diffvae] dense-load refusal: {err}");
        let text = err.to_string();
        assert!(text.contains(".scales"), "{text}");
        assert!(text.contains("noise"), "{text}");
    }

    /// The **bf16** tier's diffusion decoder is dense, and must keep loading through the unchanged
    /// dense entry point. Without this the packed work could have made the dense path unreachable
    /// and nothing would have said so.
    #[test]
    #[ignore = "sc-18776: needs the built LTX-2.5 tiers (LTX25_TIER_DIR). CPU-only; no GPU, no render"]
    fn the_real_bf16_diff_vae_decoder_still_loads_dense() {
        let dir = tier_dir("bf16");
        let tier = Ltx25Tier::detect(&dir)
            .expect("manifest")
            .expect("2.5 tier");
        assert!(
            tier.quant().is_none(),
            "the bf16 tier declares no affine geometry"
        );
        let path = tier
            .file(Ltx25Component::VaeDiffusionDecoder)
            .expect("diffusion decoder");
        let cfg = NaDiffusionDecoderConfig::from_checkpoint(&path).expect("config");
        let device = Device::Cpu;
        let (body, stats) = tier.diff_vae_vb(DType::F32, &device).expect("builders");
        let decoder =
            NaDiffusionDecoder::load(body, stats, &cfg).expect("the dense path is unchanged");
        let [t, h, w] = decoder.config().min_latent_shape();
        let latent =
            Tensor::randn(0f32, 1f32, (1, cfg.in_channels, t, h, w), &device).expect("latent");
        let (features, _) = decoder.stage_features(&latent).expect("dense forward");
        assert!(max_abs(&features).is_finite());
    }
}
