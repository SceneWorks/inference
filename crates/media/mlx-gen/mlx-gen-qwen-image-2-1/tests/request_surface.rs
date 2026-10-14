//! The rest of the request surface on the distilled **Turbo** engine id (`qwen_image_2_1_turbo`,
//! sc-25902): everything the base id serves beyond plain text-to-image — ordered references, RGBA
//! in and out, true CFG, LoRA/LoKr, the size envelope and `count` — on the Turbo-shaped miniature
//! snapshot ([`crate::common::turbo_snapshot`]), through the explicit provider catalog's
//! production load path. `turbo.rs` covers the schedule and the curated sampler/scheduler axes.
//!
//! **Every generator here is loaded under the Turbo id from the Turbo-shaped fixture** ([`turbo`]).
//! The base id refuses that snapshot by name, so swapping the id string in any test below fails at
//! the load instead of quietly proving the claim about the base checkpoint.
//!
//! `Generator::generate` returns pixels only, so each claim is made in two halves:
//!
//! * the generator's pixels **equal** a render assembled here from the public pipeline pieces
//!   ([`manual`]) whose inputs are written out explicitly — the sigma list, the ordered reference
//!   list, the negative prompt and scale, the adapter stack, the cache mode. The equality holds
//!   only if the generator fed its denoiser the same things;
//! * "X changes the output" is then asserted on the **latents** of two such assemblies with a
//!   stated margin ([`assert_moved`]), never on a bare pixel inequality, and the evaluation paths
//!   are read off that assembly's [`DenoiseReport`].
//!
//! The two sides of a pixel equality always run the same cache mode: every request here
//! **selects the prefix KV cache** ([`request`] — sc-25905 made it a request selection, and an
//! unset request runs prefill), the assemblies run `KvCacheMode::On`, and [`generate`] asserts on
//! the generator's own report that the render really ran cached. Where cached is compared with
//! prefill, the gate is the mixed `atol + rtol·|want|` bound `kv_cache.rs` uses, not bits.

use std::path::Path;

use mlx_gen::gen_core::{
    Conditioning, GenerationMemory, GenerationOutput, Image, ModelDescriptor, OutputChannels,
    PrefixKvCacheMode, Progress,
};
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::{CancelFlag, GenerationRequest, Generator, LoadSpec, RgbaImage, WeightsSource};
use mlx_gen_qwen_image_2_1::{
    apply_qwen_image_2_1_adapters, create_noise, decode_rgb, decode_rgba, denoise_reported,
    encode_references, joint_branch, load_scheduler_config, load_text_encoder, load_tokenizer,
    load_transformer, load_vae, prepare_conditioning_references, scheduler,
    system_prompt_drop_count, DenoiseInputs, DenoiseReport, EvaluationPath, KvCacheMode,
    KvCachePath, PromptBranch, QwenImage21Vae, ReferenceConditioning,
};
use mlx_rs::Array;

use crate::adapters::{dense, lokr, lora, peft_lokr, render_lora};
use crate::common::{host_f32, tiny_snapshot, turbo_snapshot, TURBO_SAMPLE_SIGMAS};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const PROMPT: &str = "a red fox in the forest";
const NEGATIVE: &str = "blurry low quality photo";
const SEED: u64 = 42;
/// Target edge of the renders here: a 2×2 latent grid, four target tokens.
const EDGE: u32 = 32;
/// Edge of the synthetic references. Every reference is fitted to the miniature snapshot's own
/// `output_resolution` (64 px), i.e. a 4×4 latent grid — sixteen prefix tokens each.
const REFERENCE_EDGE: u32 = 64;

/// Cached vs prefill on the same port and device: `|Δ| ≤ ATOL + RTOL·|want|` per element — the
/// bound `kv_cache.rs` holds the two paths to. Measured on Metal through the adapted eight-step
/// loop: max |Δ| = 4.8e-7 at peak 8.9, 1.7 % of the bound. An adapter missing from the cached
/// attention moves the latents by 6.6e-1.
const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-5;

// ── the generator side ──────────────────────────────────────────────────────────────────────────

fn load_with(id: &str, root: &Path, adapters: Vec<AdapterSpec>) -> Box<dyn Generator> {
    let generator = mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(
            id,
            &LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_adapters(adapters),
        )
        .unwrap_or_else(|e| panic!("{id} loads through the catalog path: {e}"));
    assert_eq!(generator.descriptor().id, id);
    generator
}

/// The Turbo generator over the Turbo-shaped fixture at `root`.
fn turbo(root: &Path) -> Box<dyn Generator> {
    load_with(TURBO, root, Vec::new())
}

/// The base request of every test here. It **selects the prefix KV cache**: the assemblies it is
/// compared with run the cache, and a request that selects nothing runs prefill (sc-25905).
fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: EDGE,
        height: EDGE,
        seed: Some(SEED),
        memory: Some(GenerationMemory {
            prefix_kv_cache: Some(PrefixKvCacheMode::Cached),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn generate(
    generator: &dyn Generator,
    req: &GenerationRequest,
) -> (GenerationOutput, Vec<(u32, u32)>) {
    let mut steps = Vec::new();
    let report = generator
        .generate_with_report(req, &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        })
        .unwrap_or_else(|e| panic!("the render runs: {e}"));
    // Every render in this file is compared with a cache-on assembly, so it must have run the
    // cache — read off what the denoise loop did, not off the request.
    let ran = report
        .prefix_kv_cache
        .expect("the path is reported on every render");
    assert_eq!(
        (ran.path, ran.overflow),
        (PrefixKvCacheMode::Cached, None),
        "the request selected the cache"
    );
    assert!(ran.cache_bytes > 0, "a cached render holds a cache");
    (report.output.expect("a render has an output"), steps)
}

/// The single RGB8 image of a default-channel render, and the steps it reported.
fn rgb(generator: &dyn Generator, req: &GenerationRequest) -> (Vec<u8>, Vec<(u32, u32)>) {
    let (out, steps) = generate(generator, req);
    let GenerationOutput::Images(mut images) = out else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    (images.remove(0).pixels, steps)
}

fn every_step(total: u32) -> Vec<(u32, u32)> {
    (1..=total).map(|current| (current, total)).collect()
}

/// The snapshot's eight saved sigmas and the trailing 0 — what a Turbo render with no step count
/// denoises over (pinned value for value in `model::tests` and rendered against in `turbo.rs`).
fn saved_then_zero() -> Vec<f32> {
    TURBO_SAMPLE_SIGMAS
        .iter()
        .map(|sigma| *sigma as f32)
        .chain([0.0])
        .collect()
}

// ── the explicit assembly ───────────────────────────────────────────────────────────────────────

/// One render written out by hand: nothing here is derived from a `GenerationRequest`, so a
/// generator render that equals it was given exactly these inputs.
struct Manual<'a> {
    width: u32,
    height: u32,
    seed: u64,
    sigmas: &'a [f32],
    /// The condition images, in the order they are fed.
    references: &'a [RgbaImage],
    /// `Some((negative prompt, scale))` runs the true-CFG negative branch.
    cfg: Option<(&'a str, f32)>,
    adapters: &'a [AdapterSpec],
    kv_cache: KvCacheMode,
}

impl<'a> Manual<'a> {
    /// Text-to-image at [`EDGE`]², [`SEED`], over `sigmas`, cache on.
    fn over(sigmas: &'a [f32]) -> Self {
        Self {
            width: EDGE,
            height: EDGE,
            seed: SEED,
            sigmas,
            references: &[],
            cfg: None,
            adapters: &[],
            kv_cache: KvCacheMode::On,
        }
    }
}

struct Rendered {
    latents: Array,
    report: DenoiseReport,
    vae: QwenImage21Vae,
    width: u32,
    height: u32,
    /// Tokens of the positive branch's joint layout before the target block.
    prefix_tokens: usize,
    /// Latent tokens the condition images contribute to that prefix.
    reference_tokens: usize,
    target_tokens: usize,
}

impl Rendered {
    fn rgb(&self) -> Vec<u8> {
        decode_rgb(
            &self.vae,
            &self.latents,
            self.width,
            self.height,
            None,
            None,
        )
        .unwrap()
        .pixels
    }

    fn rgba(&self) -> RgbaImage {
        decode_rgba(
            &self.vae,
            &self.latents,
            self.width,
            self.height,
            None,
            None,
        )
        .unwrap()
    }
}

/// Assemble and run `run` from the components of the snapshot at `root`.
fn manual(root: &Path, run: &Manual<'_>) -> Rendered {
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root).unwrap();
    let mut transformer = load_transformer(root).unwrap();
    if !run.adapters.is_empty() {
        apply_qwen_image_2_1_adapters(&mut transformer, run.adapters).unwrap();
    }
    let vae = load_vae(root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();

    let prepared = prepare_conditioning_references(&encoder, run.references).unwrap();
    let branch = |prompt: &str| {
        let conditioning = encoder
            .encode_conditioning(&tokenizer, prompt, drop, &prepared)
            .unwrap();
        joint_branch(&conditioning, &prepared, run.width, run.height).unwrap()
    };
    let pos = branch(PROMPT);
    let neg = run.cfg.map(|(negative, _)| branch(negative));
    let reference_latents = encode_references(&vae, &prepared).unwrap();

    let cancel = CancelFlag::new();
    let mut report = DenoiseReport::default();
    let latents = denoise_reported(
        DenoiseInputs {
            transformer: &transformer,
            sigmas: run.sigmas,
            latents: create_noise(
                run.seed,
                run.width,
                run.height,
                transformer.config().in_channels,
            )
            .unwrap(),
            prompt_embeds: &pos.text,
            negative_embeds: neg.as_ref().map(|neg| &neg.text),
            true_cfg_scale: run.cfg.map_or(1.0, |(_, scale)| scale),
            width: run.width,
            height: run.height,
            sampler: None,
            seed: run.seed,
            cancel: &cancel,
            references: (!prepared.is_empty()).then(|| ReferenceConditioning {
                latents: &reference_latents,
                layout: &pos.layout,
                negative_layout: neg.as_ref().map(|neg| &neg.layout),
            }),
            kv_cache: run.kv_cache,
        },
        &mut report,
        &mut |_| {},
    )
    .unwrap();
    latents.eval().unwrap();
    Rendered {
        latents,
        report,
        vae,
        width: run.width,
        height: run.height,
        prefix_tokens: pos.layout.prefix_len(),
        reference_tokens: prepared.iter().map(|r| r.latent_tokens()).sum(),
        target_tokens: pos.layout.target_tokens(),
    }
}

// ── comparisons ─────────────────────────────────────────────────────────────────────────────────

/// `(max |a − b|, max |b|)`.
fn gap(a: &Array, b: &Array) -> (f32, f32) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let (a, b) = (host_f32(a), host_f32(b));
    let max_abs = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(max_abs.is_finite(), "non-finite difference");
    (max_abs, b.iter().fold(0f32, |m, v| m.max(v.abs())))
}

/// The two latents are different renders, by a margin no rounding produces:
/// `max |got − base| > 1e-3 · max(1, peak)` — the bar `adapters.rs` holds "the adapter moved the
/// velocity" to. The smallest difference asserted through it in this file is the reference-order
/// one, measured 5.2e-2 at peak 4.2 on Metal (12× the bar); every other is above 2.5e-1.
fn assert_moved(name: &str, got: &Array, base: &Array) {
    let (max_abs, peak) = gap(got, base);
    eprintln!("{name}: max|Δ|={max_abs:.3e} peak={peak:.3e}");
    assert!(
        max_abs > 1e-3 * peak.max(1.0),
        "{name}: the latents did not move (max|Δ|={max_abs:.3e}, peak {peak:.3e})"
    );
}

/// `|got − want| ≤ atol + rtol·|want|` for every element.
fn assert_agrees(name: &str, got: &Array, want: &Array) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape mismatch");
    let (g, w) = (host_f32(got), host_f32(want));
    let worst = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs() / (ATOL + RTOL * b.abs()))
        .fold(0f32, f32::max);
    let (max_abs, peak) = gap(got, want);
    eprintln!("{name}: max|Δ|={max_abs:.3e} peak={peak:.3e} worst/bound={worst:.3e}");
    assert!(
        worst <= 1.0,
        "{name}: cached and prefill disagree (max|Δ|={max_abs:.3e}, {worst:.1}× the bound)"
    );
}

/// One extract then cached-only evaluations for `branch`, over `evaluations` model evaluations.
fn assert_extract_then_cached(report: &DenoiseReport, branch: PromptBranch, evaluations: usize) {
    let paths: Vec<EvaluationPath> = report
        .evaluations
        .iter()
        .filter(|evaluation| evaluation.branch == branch)
        .map(|evaluation| evaluation.path)
        .collect();
    let mut want = vec![EvaluationPath::Cached; evaluations];
    want[0] = EvaluationPath::Extract;
    assert_eq!(paths, want, "{branch:?}");
}

// ── fixtures ────────────────────────────────────────────────────────────────────────────────────

/// A deterministic RGB reference; `kind` picks one of two unrelated pictures.
fn reference(kind: u32) -> Image {
    let edge = REFERENCE_EDGE;
    let mut pixels = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            if kind == 0 {
                pixels.extend([(x * 4) as u8, (y * 4) as u8, 32]);
            } else {
                let on = (x / 8 + y / 8) % 2 == 0;
                pixels.extend([if on { 230 } else { 20 }, 128, (255 - x * 3) as u8]);
            }
        }
    }
    Image {
        width: edge,
        height: edge,
        pixels,
    }
}

fn opaque(image: &Image) -> RgbaImage {
    RgbaImage::from_rgb(image).unwrap()
}

/// [`reference`]`(0)` with a real alpha: opaque inside a centred square, transparent outside.
fn transparent_reference() -> RgbaImage {
    let mut image = opaque(&reference(0));
    let edge = REFERENCE_EDGE;
    for (i, px) in image.pixels.chunks_exact_mut(4).enumerate() {
        let (x, y) = (i as u32 % edge, i as u32 / edge);
        let inside = (16..48).contains(&x) && (16..48).contains(&y);
        px[3] = if inside { 255 } else { 0 };
    }
    assert!(!image.is_opaque());
    image
}

fn with_references(images: Vec<Image>) -> GenerationRequest {
    GenerationRequest {
        conditioning: vec![Conditioning::MultiReference { images }],
        ..request()
    }
}

// ── (a) references ──────────────────────────────────────────────────────────────────────────────

/// One and two ordered references render on the Turbo id, over the snapshot's saved sigmas, and
/// the cached loop holds the reference tokens in its prefix: the first evaluation extracts over
/// the whole joint sequence (text + every reference block + target) and the other seven run the
/// blocks over the four target tokens only.
///
/// *Mutations that red this:* the Turbo generator dropping its reference conditioning (the render
/// equals the text-to-image one, not the assembly with references); denoising a reference request
/// over the base shifted schedule or 40 steps; the loop re-extracting instead of decoding from
/// the cache.
#[test]
fn turbo_renders_one_and_two_references_over_the_saved_sigmas_from_the_prefix_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let sigmas = saved_then_zero();
    let plain = manual(&root, &Manual::over(&sigmas));

    let one = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: reference(0),
            strength: None,
        }],
        ..request()
    };
    let two = with_references(vec![reference(0), reference(1)]);
    for (name, req, references) in [
        ("one reference", one, vec![opaque(&reference(0))]),
        (
            "two references",
            two,
            vec![opaque(&reference(0)), opaque(&reference(1))],
        ),
    ] {
        generator.validate(&req).unwrap();
        let (pixels, steps) = rgb(generator.as_ref(), &req);
        assert_eq!(steps, every_step(8), "{name}: one step per saved sigma");

        let want = manual(
            &root,
            &Manual {
                references: &references,
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(
            pixels,
            want.rgb(),
            "{name}: the saved sigmas, these references"
        );

        // The cached path, with the reference blocks inside the cached prefix.
        let report = &want.report;
        assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
        assert_eq!(report.evaluations.len(), 8, "{name}");
        assert_extract_then_cached(report, PromptBranch::Positive, 8);
        assert_eq!(want.reference_tokens, 16 * references.len(), "{name}");
        assert!(
            want.prefix_tokens > want.reference_tokens,
            "{name}: the prefix is the text plus every reference block"
        );
        assert_eq!(want.target_tokens, 4, "{name}: a 32x32 target");
        assert_eq!(
            report.evaluations[0].block_tokens,
            want.prefix_tokens + want.target_tokens,
            "{name}: the extract ran the whole joint sequence"
        );
        assert!(report.evaluations[1..]
            .iter()
            .all(|evaluation| evaluation.block_tokens == want.target_tokens));
        assert!(
            want.prefix_tokens >= plain.prefix_tokens + want.reference_tokens,
            "{name}: the prefix grew by every reference block (and the template's image markers)"
        );
        assert!(
            report.kv_cache_bytes > plain.report.kv_cache_bytes,
            "{name}: the cache holds the reference blocks' K/V"
        );

        // The references are load-bearing, and so is the schedule.
        assert_moved(&format!("{name} vs none"), &want.latents, &plain.latents);
        // …in the emitted bytes too: the equality above tells the two assemblies apart only if
        // their decodes differ, so that is asserted rather than assumed of the fixture.
        assert_ne!(pixels, plain.rgb(), "{name}: not the text-to-image render");
        let shifted = scheduler::sigmas_for_image(
            &load_scheduler_config(&tiny_snapshot()).unwrap(),
            8,
            EDGE,
            EDGE,
        )
        .unwrap();
        let over_base_schedule = manual(
            &root,
            &Manual {
                references: &references,
                ..Manual::over(&shifted)
            },
        );
        assert_moved(
            &format!("{name}: saved vs base-shifted sigmas"),
            &want.latents,
            &over_base_schedule.latents,
        );
        assert_ne!(
            want.rgb(),
            over_base_schedule.rgb(),
            "{name}: not the render over the base shifted schedule"
        );
    }
}

/// Reference ORDER reaches the Turbo render: each order renders its own assembly, and the two
/// assemblies are different latents.
///
/// *Mutation that reds this:* canonicalising the reference list (any order-insensitive feed) —
/// both orders then render the same picture.
#[test]
fn turbo_reference_order_changes_the_render() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let sigmas = saved_then_zero();
    let (a, b) = (reference(0), reference(1));

    let mut rendered = Vec::new();
    for order in [[&a, &b], [&b, &a]] {
        let (pixels, _) = rgb(
            generator.as_ref(),
            &with_references(order.iter().map(|image| (*image).clone()).collect()),
        );
        let references: Vec<RgbaImage> = order.iter().map(|image| opaque(image)).collect();
        let want = manual(
            &root,
            &Manual {
                references: &references,
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(pixels, want.rgb(), "each order renders its own assembly");
        rendered.push((pixels, want.latents));
    }
    assert_moved("[a, b] vs [b, a]", &rendered[0].1, &rendered[1].1);
    assert_ne!(rendered[0].0, rendered[1].0);
}

fn refusal(generator: &dyn Generator, req: &GenerationRequest) -> String {
    match generator.validate(req) {
        Err(err) => format!("{err:?}"),
        Ok(()) => panic!("{}: the request must be refused", generator.descriptor().id),
    }
}

/// `req` is refused on the Turbo id exactly as on the base id: the same error variant and the
/// same message, up to the engine id the shared floor interpolates.
fn assert_refused_as_on_base(
    turbo: &dyn Generator,
    base: &dyn Generator,
    req: &GenerationRequest,
    needle: &str,
) {
    let (on_turbo, on_base) = (refusal(turbo, req), refusal(base, req));
    assert!(on_base.contains(needle), "{on_base}");
    assert_eq!(on_turbo.replace(TURBO, BASE), on_base);
}

/// The one-to-ten window is the base id's: ten references validate, the eleventh is the same
/// refusal — and so are an empty list and a `Mask`.
#[test]
fn turbo_refuses_the_eleventh_reference_exactly_as_base_does() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let base = load_with(BASE, &tiny_snapshot(), Vec::new());

    let many = |count: u32| with_references((0..count).map(|i| reference(i % 2)).collect());
    generator
        .validate(&many(10))
        .expect("ten references is the documented boundary");
    assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &many(11), "at most 10");
    assert_refused_as_on_base(
        generator.as_ref(),
        base.as_ref(),
        &with_references(Vec::new()),
        "empty MultiReference",
    );
    let masked = GenerationRequest {
        conditioning: vec![Conditioning::Mask {
            image: reference(0),
        }],
        ..request()
    };
    assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &masked, "no mask input");
}

// ── (b) RGBA ────────────────────────────────────────────────────────────────────────────────────

/// `output_channels: Rgba` on the Turbo id emits the VAE's own four channels — the assembly's
/// `decode_rgba`, byte for byte, alpha included — and the default request still emits RGB, the
/// white composite of the same decode.
///
/// *Mutation that reds this:* the Turbo generator answering an `Rgba` request from the RGB decode
/// (a fabricated `A = 255`, or `Images`).
#[test]
fn turbo_emits_the_decoded_alpha_on_an_rgba_request() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let sigmas = saved_then_zero();
    let want = manual(&root, &Manual::over(&sigmas));

    let mut asked = request();
    asked.output_channels = OutputChannels::Rgba;
    generator.validate(&asked).unwrap();
    let (out, steps) = generate(generator.as_ref(), &asked);
    assert_eq!(steps, every_step(8));
    let GenerationOutput::ImagesRgba(images) = out else {
        panic!("an Rgba request must emit ImagesRgba");
    };
    assert_eq!(images.len(), 1);
    let image = &images[0];
    image.validate().unwrap();
    assert_eq!(image.channels(), 4);
    assert_eq!((image.width, image.height), (EDGE, EDGE));
    assert_eq!(image.pixels.len(), (EDGE * EDGE * 4) as usize);
    assert_eq!(
        image.pixels,
        want.rgba().pixels,
        "the four decoded channels"
    );
    assert!(
        !image.is_opaque(),
        "the miniature decode's alpha is not a constant 255, so a fabricated plane cannot pass"
    );

    let (default, _) = rgb(generator.as_ref(), &request());
    assert_eq!(default, want.rgb(), "the default is still the RGB emission");
    let composited = image.to_rgb_over_white().unwrap();
    let worst = default
        .iter()
        .zip(&composited.pixels)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 1, "RGB vs composited RGBA differ by {worst} LSB");
}

/// An RGBA reference is accepted on the Turbo id and its **alpha** conditions the render: the
/// transparent layer and the same colours made opaque are two different renders, each equal to its
/// own assembly.
///
/// *Mutation that reds this:* flattening a `ReferenceRgba` to opaque before it is encoded.
#[test]
fn turbo_accepts_an_rgba_reference_and_its_alpha_conditions_the_render() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let sigmas = saved_then_zero();

    let transparent = transparent_reference();
    let mut solid = transparent.clone();
    solid.pixels.chunks_exact_mut(4).for_each(|px| px[3] = 255);

    let mut rendered = Vec::new();
    let mut emitted = Vec::new();
    for layer in [&transparent, &solid] {
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceRgba {
                image: layer.clone(),
                strength: None,
            }],
            output_channels: OutputChannels::Rgba,
            ..request()
        };
        generator.validate(&req).unwrap();
        let (out, _) = generate(generator.as_ref(), &req);
        let GenerationOutput::ImagesRgba(images) = out else {
            panic!("an Rgba request must emit ImagesRgba");
        };
        let want = manual(
            &root,
            &Manual {
                references: std::slice::from_ref(layer),
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(images[0].pixels, want.rgba().pixels);
        rendered.push(want.latents);
        emitted.push(images[0].pixels.clone());
    }
    assert_moved("transparent vs opaque layer", &rendered[0], &rendered[1]);
    assert_ne!(
        emitted[0], emitted[1],
        "the transparent and the opaque layer are different emitted images"
    );
    let none = manual(&root, &Manual::over(&sigmas));
    assert_moved("transparent layer vs none", &rendered[0], &none.latents);
    assert_ne!(
        emitted[0],
        none.rgba().pixels,
        "the layer-conditioned render is not the text-to-image one"
    );
}

// ── (c) true CFG ────────────────────────────────────────────────────────────────────────────────

/// A negative prompt with a scale above 1 runs the negative branch on the Turbo id — its own
/// evaluations and its own prefix cache, on the text-to-image route and with a reference — and
/// changes the latents. A scale of 1 (explicit or default), and a scale without a negative prompt,
/// leave the render exactly what it was.
///
/// *Mutation that reds this:* the Turbo generator never engaging the negative branch (the guided
/// request renders the unguided assembly).
#[test]
fn turbo_true_cfg_runs_the_negative_branch_with_its_own_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let sigmas = saved_then_zero();
    let layer = [opaque(&reference(1))];

    for (name, base_req, references) in [
        ("t2i", request(), &[][..]),
        (
            "one reference",
            with_references(vec![reference(1)]),
            &layer[..],
        ),
    ] {
        let unguided = manual(
            &root,
            &Manual {
                references,
                ..Manual::over(&sigmas)
            },
        );
        let (plain, _) = rgb(generator.as_ref(), &base_req);
        assert_eq!(plain, unguided.rgb(), "{name}: unguided");

        let mut req = base_req.clone();
        req.negative_prompt = Some(NEGATIVE.to_owned());
        req.true_cfg = Some(2.5);
        generator.validate(&req).unwrap();
        let (guided_pixels, steps) = rgb(generator.as_ref(), &req);
        assert_eq!(steps, every_step(8), "{name}");
        let guided = manual(
            &root,
            &Manual {
                references,
                cfg: Some((NEGATIVE, 2.5)),
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(guided_pixels, guided.rgb(), "{name}: guided");
        assert_moved(
            &format!("{name}: guided vs unguided"),
            &guided.latents,
            &unguided.latents,
        );
        assert_ne!(guided_pixels, plain, "{name}: not the unguided render");

        // Two branches, each extracting once and decoding from its own cache afterwards.
        let report = &guided.report;
        assert_eq!(report.kv_cache, KvCachePath::Cached, "{name}");
        assert_eq!(
            report.evaluations.len(),
            16,
            "{name}: two evaluations per step"
        );
        assert_extract_then_cached(report, PromptBranch::Positive, 8);
        assert_extract_then_cached(report, PromptBranch::Negative, 8);
        assert!(
            report.kv_cache_bytes > unguided.report.kv_cache_bytes,
            "{name}: the negative branch holds a second cache"
        );
        assert_eq!(unguided.report.evaluations.len(), 8, "{name}");
        assert!(unguided
            .report
            .evaluations
            .iter()
            .all(|evaluation| evaluation.branch == PromptBranch::Positive));

        // Like with like: the same generator, the same request but for the inert knob.
        req.true_cfg = Some(1.0);
        assert_eq!(rgb(generator.as_ref(), &req).0, plain, "{name}: scale 1");
        req.true_cfg = None;
        assert_eq!(
            rgb(generator.as_ref(), &req).0,
            plain,
            "{name}: a negative prompt alone (the default scale is 1)"
        );
        let mut scale_only = base_req.clone();
        scale_only.true_cfg = Some(2.5);
        assert_eq!(
            rgb(generator.as_ref(), &scale_only).0,
            plain,
            "{name}: a scale with no negative prompt"
        );
    }
}

// ── (d) adapters ────────────────────────────────────────────────────────────────────────────────

/// A `qwen-image-2-1` LoRA and a LoKr install onto the Turbo transformer through the Turbo id's
/// load and change what it renders; a strength of zero is the unadapted render.
///
/// *Mutation that reds this:* the Turbo load skipping the adapter install.
#[test]
fn a_lora_and_a_lokr_apply_to_the_turbo_transformer() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let sigmas = saved_then_zero();
    let lora_file = render_lora(tmp.path());
    let lokr_file = peft_lokr(
        tmp.path(),
        &mut dense(),
        &[
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.1.attn.to_out.0",
        ],
        2.0,
    );

    let plain = manual(&root, &Manual::over(&sigmas));
    let (plain_pixels, _) = rgb(turbo(&root).as_ref(), &request());
    assert_eq!(plain_pixels, plain.rgb());

    for (name, spec) in [
        ("lora", lora(&lora_file, 1.0)),
        ("lokr", lokr(&lokr_file, 1.0)),
    ] {
        let stack = [spec];
        let generator = load_with(TURBO, &root, stack.to_vec());
        let (pixels, steps) = rgb(generator.as_ref(), &request());
        assert_eq!(steps, every_step(8), "{name}");
        let adapted = manual(
            &root,
            &Manual {
                adapters: &stack,
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(pixels, adapted.rgb(), "{name}: the adapted transformer");
        assert_moved(&format!("{name} vs none"), &adapted.latents, &plain.latents);
        assert_ne!(pixels, plain_pixels, "{name}");
    }

    let off = load_with(TURBO, &root, vec![lora(&lora_file, 0.0)]);
    assert_eq!(
        rgb(off.as_ref(), &request()).0,
        plain_pixels,
        "strength 0 is the unadapted render"
    );
}

/// The adapter is in effect on the **cached** path. In a two-reference Turbo render seven of the
/// eight evaluations decode the target tokens against the cached prefix; with a LoRA installed
/// that render agrees with the same adapted transformer run prefill-only (cache off), and is far
/// from the unadapted one.
///
/// *Mutation that reds this:* the cached attention projecting through the un-adapted base
/// Linears (cache on then departs from cache off by the adapter's whole effect).
#[test]
fn turbo_adapters_are_in_effect_on_the_cached_path() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let sigmas = saved_then_zero();
    let stack = [lora(&render_lora(tmp.path()), 1.0)];
    let references = [opaque(&reference(0)), opaque(&reference(1))];
    let run = |adapters, kv_cache| {
        manual(
            &root,
            &Manual {
                references: &references,
                adapters,
                kv_cache,
                ..Manual::over(&sigmas)
            },
        )
    };

    let cached = run(&stack[..], KvCacheMode::On);
    let prefill = run(&stack[..], KvCacheMode::Off);
    let unadapted = run(&[][..], KvCacheMode::On);

    assert_eq!(cached.report.kv_cache, KvCachePath::Cached);
    assert_extract_then_cached(&cached.report, PromptBranch::Positive, 8);
    assert_eq!(prefill.report.kv_cache, KvCachePath::Prefill);
    assert!(prefill
        .report
        .evaluations
        .iter()
        .all(|evaluation| evaluation.path == EvaluationPath::Prefill));

    assert_agrees(
        "adapted: cached vs prefill",
        &cached.latents,
        &prefill.latents,
    );
    assert_moved(
        "cached: adapted vs unadapted",
        &cached.latents,
        &unadapted.latents,
    );

    // …and that cached, adapted run is what the Turbo generator renders.
    let generator = load_with(TURBO, &root, stack.to_vec());
    let (pixels, steps) = rgb(
        generator.as_ref(),
        &with_references(vec![reference(0), reference(1)]),
    );
    assert_eq!(steps, every_step(8));
    assert_eq!(pixels, cached.rgb());
}

// ── (e) the curated menu's refusals ─────────────────────────────────────────────────────────────

/// A sampler or scheduler name outside the curated menu is refused on the Turbo id exactly as on
/// the base id. (The menu itself is compared in the descriptor test below; a non-default sampler
/// and a non-default scheduler are rendered in `turbo.rs`.)
#[test]
fn turbo_refuses_an_off_menu_sampler_or_scheduler_exactly_as_base_does() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let base = load_with(BASE, &tiny_snapshot(), Vec::new());

    let mut sampler = request();
    sampler.sampler = Some("not-a-sampler".to_owned());
    assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &sampler, "sampler");
    let mut scheduler = request();
    scheduler.scheduler = Some("flow_match_euler".to_owned());
    assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &scheduler, "scheduler");
}

// ── (f) size envelope and count ─────────────────────────────────────────────────────────────────

/// Non-square and non-preset on-grid sizes render on the Turbo id at the size asked for, an
/// off-grid or out-of-range size is the base id's refusal, and `count` returns that many images.
#[test]
fn turbo_renders_the_size_envelope_and_the_requested_count() {
    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let generator = turbo(&root);
    let base = load_with(BASE, &tiny_snapshot(), Vec::new());
    let sigmas = saved_then_zero();

    // None of these is one of the seven upstream presets; two are not square.
    for (width, height) in [(64, 32), (32, 96), (96, 96)] {
        let req = GenerationRequest {
            width,
            height,
            ..request()
        };
        generator.validate(&req).unwrap();
        let (out, steps) = generate(generator.as_ref(), &req);
        assert_eq!(steps, every_step(8), "{width}x{height}");
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        assert_eq!((images[0].width, images[0].height), (width, height));
        assert_eq!(images[0].pixels.len(), (width * height * 3) as usize);
        let want = manual(
            &root,
            &Manual {
                width,
                height,
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(images[0].pixels, want.rgb(), "{width}x{height}");
    }
    for (width, height, needle) in [(48, 32, "32"), (4096, 4096, "4096")] {
        let req = GenerationRequest {
            width,
            height,
            ..request()
        };
        assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &req, needle);
    }

    let batch = GenerationRequest {
        count: 3,
        ..request()
    };
    let (out, steps) = generate(generator.as_ref(), &batch);
    let GenerationOutput::Images(images) = out else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 3);
    assert_eq!(steps.len(), 3 * 8, "eight steps per image");
    for (i, image) in images.iter().enumerate() {
        let want = manual(
            &root,
            &Manual {
                seed: SEED + i as u64,
                ..Manual::over(&sigmas)
            },
        );
        assert_eq!(image.pixels, want.rgb(), "image {i} is seed + {i}");
    }
    assert_ne!(images[0].pixels, images[1].pixels);
    let over = GenerationRequest {
        count: 9,
        ..request()
    };
    assert_refused_as_on_base(generator.as_ref(), base.as_ref(), &over, "count");
}

// ── (g) the descriptor ──────────────────────────────────────────────────────────────────────────

/// The top-level fields of a `Capabilities`, by name, from its pretty `Debug` rendering — every
/// field the struct has, including ones added after this test was written.
fn capability_fields(descriptor: &ModelDescriptor) -> Vec<(String, String)> {
    let rendered = format!("{:#?}", descriptor.capabilities);
    let mut fields: Vec<(String, String)> = Vec::new();
    for line in rendered.lines() {
        let top_level = line.starts_with("    ") && !line.starts_with("     ");
        match line.trim_start().split_once(": ") {
            Some((name, value)) if top_level => fields.push((name.to_owned(), value.to_owned())),
            _ => {
                if let Some((_, value)) = fields.last_mut() {
                    value.push('\n');
                    value.push_str(line);
                }
            }
        }
    }
    fields
}

/// `turbo` is `base` field for field, except the id. The descriptor is destructured with no `..`,
/// so a field added to `ModelDescriptor` has to be placed here deliberately; `Capabilities` is
/// walked field by field, so a divergence names the capability that moved.
fn assert_mirrors_base(turbo: &ModelDescriptor, base: &ModelDescriptor) {
    let ModelDescriptor {
        encoder_contract,
        denoiser_output_latent_space,
        id,
        family,
        backend,
        modality,
        capabilities,
        required_components,
        control_kinds,
    } = turbo;
    assert_eq!(*id, TURBO);
    assert_eq!(base.id, BASE);
    assert_eq!(*family, base.family);
    assert_eq!(*backend, base.backend);
    assert_eq!(*modality, base.modality);
    assert_eq!(*required_components, base.required_components);
    assert_eq!(
        format!("{encoder_contract:?}"),
        format!("{:?}", base.encoder_contract)
    );
    assert_eq!(
        format!("{denoiser_output_latent_space:?}"),
        format!("{:?}", base.denoiser_output_latent_space)
    );
    assert_eq!(
        format!("{control_kinds:?}"),
        format!("{:?}", base.control_kinds)
    );

    // The capabilities this story is about, typed, so a failure reads as the capability.
    let caps = &base.capabilities;
    assert_eq!(capabilities.conditioning, caps.conditioning, "references");
    assert_eq!(
        capabilities.supports_alpha_output,
        caps.supports_alpha_output
    );
    assert_eq!(capabilities.supports_true_cfg, caps.supports_true_cfg);
    assert_eq!(
        capabilities.supports_negative_prompt,
        caps.supports_negative_prompt
    );
    assert_eq!(capabilities.supports_guidance, caps.supports_guidance);
    assert_eq!(capabilities.supports_lora, caps.supports_lora);
    assert_eq!(capabilities.supports_lokr, caps.supports_lokr);
    assert_eq!(capabilities.samplers, caps.samplers, "sampler menu");
    assert_eq!(capabilities.schedulers, caps.schedulers, "scheduler menu");
    assert_eq!(capabilities.min_size, caps.min_size);
    assert_eq!(capabilities.max_size, caps.max_size);
    assert_eq!(capabilities.max_count, caps.max_count);
    assert_eq!(capabilities.size_floor, caps.size_floor);
    assert_eq!(capabilities.supported_quants, caps.supported_quants);

    // …and every other one, whatever it is.
    let (turbo_fields, base_fields) = (capability_fields(turbo), capability_fields(base));
    let names: Vec<&str> = base_fields.iter().map(|(name, _)| name.as_str()).collect();
    for expected in [
        "conditioning",
        "supports_alpha_output",
        "supports_true_cfg",
        "supports_lora",
        "supports_lokr",
        "samplers",
        "schedulers",
        "min_size",
        "max_size",
        "size_floor",
    ] {
        assert!(
            names.contains(&expected),
            "the field walk lost `{expected}`: {names:?}"
        );
    }
    assert_eq!(turbo_fields.len(), base_fields.len());
    for ((turbo_name, turbo_value), (base_name, base_value)) in
        turbo_fields.iter().zip(&base_fields)
    {
        assert_eq!(turbo_name, base_name);
        assert_eq!(
            turbo_value, base_value,
            "capability `{base_name}` differs between the Turbo and the base descriptor"
        );
    }
}

/// The Turbo id advertises the base id's surface — references, alpha, true CFG, LoRA/LoKr, the
/// sampler and scheduler menus, the size bounds, and everything else — both as registered and as
/// the loaded generator reports it.
///
/// *Mutation that reds this:* flipping any one capability on the Turbo descriptor.
#[test]
fn the_turbo_descriptor_is_the_base_descriptor_field_for_field_except_the_id() {
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let registered = |id: &str| {
        registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == id)
            .unwrap_or_else(|| panic!("{id} is registered"))
    };
    assert_mirrors_base(&registered(TURBO), &registered(BASE));

    let tmp = tempfile::tempdir().unwrap();
    let root = turbo_snapshot(tmp.path());
    let base = load_with(BASE, &tiny_snapshot(), Vec::new());
    let loaded = turbo(&root);
    assert_mirrors_base(loaded.descriptor(), base.descriptor());
    // The capabilities the rest of this file renders through are the advertised ones.
    let caps = &loaded.descriptor().capabilities;
    assert!(caps.supports_alpha_output && caps.supports_true_cfg);
    assert!(caps.supports_lora && caps.supports_lokr);
    assert!(caps.accepts(mlx_gen::gen_core::ConditioningKind::MultiReference));
    assert!(caps.accepts(mlx_gen::gen_core::ConditioningKind::ReferenceRgba));
}
