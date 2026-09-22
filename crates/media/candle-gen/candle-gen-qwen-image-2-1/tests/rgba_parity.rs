//! Transparency / RGBA parity vs the frozen `QwenImage21Pipeline` on the committed miniature
//! snapshot (`crates/media/mlx-gen/tools/dump_qwen21_rgba.py`), plus the contract claims the alpha
//! surface has to hold (sc-24111). The candle twin of the MLX crate's `rgba_parity.rs`, reading the
//! **same** committed fixture (`qwen21_rgba.safetensors`).
//!
//! **Upstream has no transparency flag.** The 2.1 VAE is four-channel in and out, so the pipeline
//! always decodes four channels and `image_processor.postprocess(..., "pil")` always returns mode
//! `RGBA`; *whether* a render is transparent is decided by the prompt. `OutputChannels::Rgba` is
//! therefore an output-surface selector, not a render mode, and the gates below are split to say
//! exactly that:
//!
//! 1. **the quantisation rule is byte-exact** — feeding upstream's OWN float decode through the
//!    port's `RgbaImage` emission must reproduce upstream's OWN uint8 RGBA, byte for byte. This
//!    is the one claim sc-24111 newly introduces, and it is checkable without any numerical
//!    drift: it isolates "straight alpha, per-channel clamp, `(v·255).round()`" from the denoise.
//! 2. **the two emissions agree** — the RGB image a default request receives is the white
//!    composite of the RGBA image the opted-in request receives, to within the 1 LSB the two
//!    composite orders (float-then-quantise vs quantise-then-composite) can differ by.
//! 3. **an RGBA reference is fed to the two consumers the way upstream feeds it** — the VAE
//!    encode gets all four channels (a real alpha plane, not the constant `+1.0`), while the
//!    Qwen3-VL vision tower gets the reference composited over **white**. The oracle for the
//!    flatten is upstream's own bytes (`extract/vision_rgb_0`), not a re-derivation of the rule.
//! 4. **RGB is the opaque special case** — the `extract_opaque` case runs the identical geometry
//!    with `A = 255` and must land on a constant `+1.0` VAE alpha plane, which is what the RGB
//!    reference route always produced.
//!
//! Tolerances are **this lane's own**, not the MLX twin's: candle CPU f32 against torch CPU f32 is
//! two to three orders of magnitude tighter than Metal, so `edit_parity`'s bounds carry over
//! unchanged (host preprocessing `1e-6 × peak`, conditioning `2e-6 × peak`, reference latents
//! `1e-6`, end-to-end latents `3e-5 × peak`). The image gate is the one exception: it compares the
//! **uint8** emission against upstream's float decode, so it is floored by quantisation, not by
//! drift — see [`IMAGE_TOL`]. Measured on this fixture, on this lane:
//!
//! * `vae_input_0` — **exactly 0** on both cases (host `u8 → f32` arithmetic).
//! * `pixel_values_0` — `max|Δ| = 2.384e-7` at `peak = 2.146` on both cases, one f32 ULP.
//! * `prompt_embeds` — worst `extract`, `max|Δ| = 1.013e-6` at `peak = 3.367`.
//! * `ref_latents_0` — worst `extract`, `max|Δ| = 2.608e-7` against a bound floor of 1.0.
//! * `latents_final` — worst `extract`, `max|Δ| = 3.043e-5` at `peak = 5.086`.
//! * `image_rgba` — `max|Δ| = 1.961e-3` on both cases, i.e. **exactly** the `1/510` uint8
//!   half-step: the decode agrees with upstream to better than one output level everywhere.
//! * the byte-exact gates (`image_rgba_u8`) — **0 differing bytes** on all three cases.
//!
//! Every comparison prints its numbers, so no bound here is tighter than its evidence.

use candle_core::{DType, IndexOp, Tensor};
use candle_gen::gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, OutputChannels,
    Progress, RgbaImage, WeightsSource,
};
use candle_gen_qwen_image_2_1::{
    denoise, encode_references, joint_layout, load_scheduler_config, load_text_encoder,
    load_tokenizer, load_transformer, load_vae, load_vision_config, prepare_references, scheduler,
    system_prompt_drop_count, DenoiseInputs, ReferenceConditioning,
};

use crate::common::{assert_close, device, errors, host_f32, tiny_snapshot, Fixture};

/// `edit_parity`'s host-preprocessing bound; both gates below measure exactly 0.
const HOST_TOL: f32 = 1e-6;
/// `edit_parity`'s conditioning bound (ViT tower + DeepStack + interleaved M-RoPE + decoder).
const TEXT_TOL: f32 = 2e-6;
/// `edit_parity`'s reference-latent bound (VAE posterior mode, normalised, packed).
const REF_LATENT_TOL: f32 = 1e-6;
/// `edit_parity`'s end-to-end bound for the joint block-causal denoise.
const LATENT_TOL: f32 = 3e-5;
/// The image gate compares the port's **uint8** RGBA emission, divided by 255, against upstream's
/// float decode, so its floor is the quantisation half-step `1/510 = 1.96e-3` — not this lane's
/// numerical drift. `pipeline_parity`'s `2e-4` is a float-vs-float bound and does **not** carry
/// over here. Measured worst `1.961e-3` (i.e. exactly the half-step, meaning the decode agrees to
/// better than one uint8 level everywhere); ~2.5x headroom.
const IMAGE_TOL: f32 = 5e-3;
/// The white composite is applied in f32 before quantisation on the RGB path and in 8-bit integer
/// arithmetic on the readback path, so the two can disagree by one least-significant bit. They may
/// not disagree by more — that would mean a different compositing rule, not a rounding order.
const COMPOSITE_LSB: i32 = 1;

fn rgba_fixture() -> Fixture {
    Fixture::open("qwen21_rgba.safetensors")
}

fn meta(w: &Fixture, name: &str) -> serde_json::Value {
    serde_json::from_str(w.meta(name)).expect("case metadata is JSON")
}

fn u8_host(t: &Tensor) -> Vec<u8> {
    host_f32(t).into_iter().map(|v| v as u8).collect()
}

/// Upstream's `[H, W, 4]` uint8 golden as the `RgbaImage` a request would receive.
fn golden_rgba(w: &Fixture, key: &str) -> RgbaImage {
    let tensor = w.tensor(key);
    let shape = tensor.dims().to_vec();
    assert_eq!(shape.len(), 3, "the uint8 golden is HWC");
    assert_eq!(shape[2], 4, "the uint8 golden is RGBA");
    RgbaImage {
        height: shape[0] as u32,
        width: shape[1] as u32,
        pixels: u8_host(&tensor),
    }
}

/// Upstream's `[4, H, W]` float decode in `[0, 1]`, mapped back to the `[-1, 1]` the VAE emits and
/// the port's emission consumes.
fn golden_decode_in_vae_range(w: &Fixture, key: &str) -> Tensor {
    let x = w.tensor(key).to_dtype(DType::F32).unwrap();
    ((x * 2.0).unwrap() - 1.0).unwrap().unsqueeze(0).unwrap()
}

/// **Gate 1 — the quantisation rule, byte-exact.**
///
/// Upstream's own float decode, pushed through the port's `RgbaImage` emission, must reproduce
/// upstream's own uint8 RGBA exactly. No denoise and no VAE run here, so nothing numerical can
/// mask a wrong rule: this gate fails if the alpha is premultiplied, if the clamp is applied
/// across channels instead of per channel, if the rounding is a truncation, or if the channel
/// order is not `[R, G, B, A]`.
#[test]
fn the_rgba_quantisation_matches_upstreams_postprocess_byte_for_byte() {
    let w = rgba_fixture();
    for case in ["t2i_rgba", "extract", "extract_opaque"] {
        let decoded = golden_decode_in_vae_range(&w, &format!("{case}/image_rgba"));
        let got = candle_gen_qwen_image_2_1::pipeline::decoded_to_rgba_image(&decoded).unwrap();
        let want = golden_rgba(&w, &format!("{case}/image_rgba_u8"));
        assert_eq!(
            (got.width, got.height),
            (want.width, want.height),
            "{case}: geometry"
        );
        assert_eq!(got.pixels.len(), want.pixels.len(), "{case}: buffer length");
        let differing = got
            .pixels
            .iter()
            .zip(&want.pixels)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "{case}/image_rgba_u8: {differing} differing bytes of {}",
            want.pixels.len()
        );
        assert_eq!(
            got.pixels, want.pixels,
            "{case}: the RGBA8 emission must be upstream's postprocess byte for byte"
        );
        got.validate().expect("a well-formed RGBA image");
        assert_eq!(got.channels(), 4);
        assert_eq!(got.row_stride(), got.width as usize * 4);
    }
}

/// **Gate 1b — the alpha is genuinely carried, not fabricated or flattened.**
///
/// The `extract` golden's alpha plane must be non-constant (upstream painted a real matte) and the
/// port must reproduce it; a port that widened an RGB decode with `A = 255` would pass every
/// colour check and fail here.
#[test]
fn the_emitted_alpha_is_upstreams_alpha_not_a_constant() {
    let w = rgba_fixture();
    let decoded = golden_decode_in_vae_range(&w, "extract/image_rgba");
    let got = candle_gen_qwen_image_2_1::pipeline::decoded_to_rgba_image(&decoded).unwrap();
    let alpha: Vec<u8> = got.pixels.chunks_exact(4).map(|px| px[3]).collect();
    let (min, max) = (*alpha.iter().min().unwrap(), *alpha.iter().max().unwrap());
    eprintln!("extract alpha: min={min} max={max}");
    assert!(
        min != max,
        "the decoded alpha must vary; a constant plane means the port fabricated it"
    );
    assert!(
        !got.is_opaque(),
        "an RGBA render whose every pixel is opaque carries no transparency"
    );
}

/// **Gate 2 — the RGB default is the white composite of the RGBA output.**
///
/// Both emissions come from ONE decode, so they must agree: the RGB image an untouched request
/// receives is exactly what a viewer would show the RGBA image as on a white page. Checked on
/// upstream's own decode so the claim is about the emissions, not about denoise drift.
#[test]
fn the_rgb_emission_is_the_white_composite_of_the_rgba_emission() {
    let w = rgba_fixture();
    for case in ["t2i_rgba", "extract"] {
        let decoded = golden_decode_in_vae_range(&w, &format!("{case}/image_rgba"));
        // The RGB path: composite in f32, then quantise (what `decode_rgb` does).
        let rgb = candle_gen_qwen_image_2_1::pipeline::decoded_to_image(
            &candle_gen_qwen_image_2_1::rgba_to_rgb_over_white(&decoded).unwrap(),
        )
        .unwrap();
        // The RGBA path: quantise, then composite in 8-bit (what a consumer does).
        let composited = candle_gen_qwen_image_2_1::pipeline::decoded_to_rgba_image(&decoded)
            .unwrap()
            .to_rgb_over_white()
            .unwrap();
        assert_eq!(rgb.width, composited.width);
        assert_eq!(rgb.height, composited.height);
        let worst = rgb
            .pixels
            .iter()
            .zip(&composited.pixels)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap_or(0);
        eprintln!("{case}: RGB vs composited RGBA worst byte delta {worst}");
        assert!(
            worst <= COMPOSITE_LSB,
            "{case}: the two emissions disagree by {worst} > {COMPOSITE_LSB} LSB — that is a \
             different compositing rule, not a rounding order"
        );
    }
}

struct Snapshot {
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    encoder: candle_gen_qwen_image_2_1::QwenImage21TextEncoder,
    transformer: candle_gen_qwen_image_2_1::QwenImage21Transformer,
    vae: candle_gen_qwen_image_2_1::QwenImage21Vae,
    scheduler: candle_gen_qwen_image_2_1::SchedulerConfig,
    drop: usize,
    vision: candle_gen_qwen_image_2_1::VisionConfig,
}

fn snapshot() -> Snapshot {
    let root = tiny_snapshot();
    let dev = device();
    let tokenizer = load_tokenizer(&root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let vision = load_vision_config(&root)
        .unwrap()
        .expect("the tiny snapshot declares a Qwen3-VL vision tower");
    Snapshot {
        encoder: load_text_encoder(&root, &dev).unwrap(),
        transformer: load_transformer(&root, &dev).unwrap(),
        vae: load_vae(&root, &dev).unwrap(),
        scheduler: load_scheduler_config(&root).unwrap(),
        tokenizer,
        drop,
        vision,
    }
}

/// The case's RGBA source, as the `Conditioning::ReferenceRgba` a request would carry.
fn rgba_source(w: &Fixture, case: &str) -> RgbaImage {
    golden_rgba(w, &format!("{case}/source_rgba_0"))
}

/// **Gate 3 — an RGBA reference reaches both consumers the way upstream feeds it.**
///
/// The load-bearing asymmetry: all four channels to the VAE, composited-over-white to the vision
/// tower. Both halves are checked against upstream's own recorded bytes.
#[test]
fn an_rgba_reference_reaches_the_vae_whole_and_the_vision_tower_whitened() {
    let w = rgba_fixture();
    let snap = snapshot();
    let dev = device();
    for case in ["extract", "extract_opaque"] {
        let source = rgba_source(&w, case);
        let prepared =
            prepare_references(std::slice::from_ref(&source), &snap.vision, &dev).unwrap();
        assert_eq!(prepared.len(), 1);
        let reference = &prepared[0];

        // The vision grid, and the 4:1 slot-to-latent binding the whole layout rests on.
        let grid: Vec<i32> = host_f32(&w.tensor(&format!("{case}/grid_thw_0")))
            .into_iter()
            .map(|v| v as i32)
            .collect();
        assert_eq!(reference.grid_thw.to_vec(), grid, "{case}: vision grid");
        assert_eq!(reference.latent_tokens(), reference.vision_slots() * 4);

        // The VISION copy: upstream's white composite, its own bytes.
        assert_close(
            &format!("{case}/pixel_values_0"),
            &reference.pixel_values,
            &w.tensor(&format!("{case}/pixel_values_0")),
            HOST_TOL,
        );

        // The VAE copy: all four channels, alpha intact.
        assert_close(
            &format!("{case}/vae_input_0"),
            &reference.vae_input,
            &w.tensor(&format!("{case}/vae_input_0")),
            HOST_TOL,
        );

        // ...and the alpha plane is what distinguishes the two cases. A port that flattened the
        // reference before the VAE would produce a constant +1.0 plane for BOTH.
        let alpha = host_f32(&reference.vae_input.i((0, 3, .., ..)).unwrap());
        let min = alpha.iter().copied().fold(f32::INFINITY, f32::min);
        let max = alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("{case}: VAE alpha range [{min}, {max}]");
        if case == "extract" {
            assert!(
                min < -0.9 && max > 0.9,
                "{case}: the transparent reference's alpha must reach the VAE, got [{min}, {max}]"
            );
        } else {
            assert!(
                alpha.iter().all(|&a| a == 1.0),
                "{case}: an opaque reference's VAE alpha is the constant +1.0"
            );
        }
    }
}

/// The same asymmetry stated as a difference: the transparent and opaque references share their
/// colours and differ only in alpha, so a port that ignored alpha on either consumer would make
/// the two cases identical there.
#[test]
fn transparency_changes_both_consumers() {
    let w = rgba_fixture();
    let snap = snapshot();
    let dev = device();
    let transparent = prepare_references(
        std::slice::from_ref(&rgba_source(&w, "extract")),
        &snap.vision,
        &dev,
    )
    .unwrap();
    let opaque = prepare_references(
        std::slice::from_ref(&rgba_source(&w, "extract_opaque")),
        &snap.vision,
        &dev,
    )
    .unwrap();

    let (vae_max, _, _) = errors(&transparent[0].vae_input, &opaque[0].vae_input);
    let (vision_max, _, _) = errors(&transparent[0].pixel_values, &opaque[0].pixel_values);
    eprintln!("transparent vs opaque: vae {vae_max}, vision {vision_max}");
    assert!(
        vae_max > 0.5,
        "the VAE input must differ between a transparent and an opaque reference"
    );
    assert!(
        vision_max > 0.5,
        "the vision copy must differ too — upstream composites the alpha over white before the \
         processor, so transparency is visible there as well"
    );
}

/// **Gate 3b — the conditioning and the denoise, end to end, from a transparent reference.**
///
/// Subject extraction / transparent-layer editing is an ordinary reference call with an ordinary
/// prompt (upstream ships no dedicated mode and no matting model), so this is the end-to-end gate
/// run with an alpha-carrying reference.
#[test]
fn transparent_layer_editing_matches_upstream_end_to_end() {
    let w = rgba_fixture();
    let snap = snapshot();
    let dev = device();
    for case in ["extract", "extract_opaque"] {
        let m = meta(&w, case);
        let prompt = m["prompt"].as_str().unwrap();
        let (width, height) = (
            m["width"].as_u64().unwrap() as u32,
            m["height"].as_u64().unwrap() as u32,
        );
        let steps = m["steps"].as_u64().unwrap() as usize;

        let source = rgba_source(&w, case);
        let prepared =
            prepare_references(std::slice::from_ref(&source), &snap.vision, &dev).unwrap();
        let cond = snap
            .encoder
            .encode_conditioning(&snap.tokenizer, prompt, snap.drop, &prepared)
            .expect("image-conditioned encode");

        let want_mask: Vec<i32> = host_f32(&w.tensor(&format!("{case}/image_pad_mask")))
            .into_iter()
            .map(|v| v as i32)
            .collect();
        assert_eq!(
            cond.image_pad_mask
                .iter()
                .map(|&m| i32::from(m))
                .collect::<Vec<_>>(),
            want_mask,
            "{case}: the image-pad mask places the condition block in the joint sequence"
        );
        assert_close(
            &format!("{case}/prompt_embeds"),
            &cond.hidden,
            &w.tensor(&format!("{case}/prompt_embeds")),
            TEXT_TOL,
        );

        let reference_latents = encode_references(&snap.vae, &prepared).unwrap();
        assert_close(
            &format!("{case}/ref_latents_0"),
            &reference_latents[0],
            &w.tensor(&format!("{case}/ref_latents_0")),
            REF_LATENT_TOL,
        );

        let layout = joint_layout(&cond.image_pad_mask, &prepared, width, height).unwrap();
        let text =
            candle_gen_qwen_image_2_1::text_rows(&cond.hidden, &cond.image_pad_mask).unwrap();
        let sigmas = scheduler::sigmas_for_image(&snap.scheduler, steps, width, height).unwrap();
        let cancel = CancelFlag::new();
        let latents = denoise(
            DenoiseInputs {
                transformer: &snap.transformer,
                sigmas: &sigmas,
                latents: w.tensor(&format!("{case}/latents_init")),
                prompt_embeds: &text,
                negative_embeds: None,
                true_cfg_scale: 1.0,
                width,
                height,
                sampler: None,
                seed: 0,
                cancel: &cancel,
                references: Some(ReferenceConditioning {
                    latents: &reference_latents,
                    layout: &layout,
                    negative_layout: None,
                }),
            },
            &mut |_: Progress| {},
        )
        .unwrap();
        assert_close(
            &format!("{case}/latents_final"),
            &latents,
            &w.tensor(&format!("{case}/latents_final")),
            LATENT_TOL,
        );

        // The four-channel decode of those latents against upstream's float image.
        let rgba = candle_gen_qwen_image_2_1::decode_rgba(
            &snap.vae,
            &latents,
            width,
            height,
            None,
            Some(&cancel),
        )
        .unwrap();
        rgba.validate().unwrap();
        assert_eq!(rgba.channels(), 4);
        let as_float = Tensor::from_vec(
            rgba.pixels
                .iter()
                .map(|&v| f32::from(v) / 255.0)
                .collect::<Vec<_>>(),
            (rgba.height as usize, rgba.width as usize, 4),
            &dev,
        )
        .unwrap()
        .permute((2, 0, 1))
        .unwrap()
        .contiguous()
        .unwrap();
        assert_close(
            &format!("{case}/image_rgba"),
            &as_float,
            &w.tensor(&format!("{case}/image_rgba")),
            IMAGE_TOL,
        );
    }
}

/// **Gate 4 — the request seam.** The transparent reference travels as
/// `Conditioning::ReferenceRgba` and keeps its alpha through `collect_references`; an RGB
/// `Reference` of the same picture arrives widened with `A = 255`.
#[test]
fn the_request_seam_carries_the_reference_alpha() {
    let w = rgba_fixture();
    let source = rgba_source(&w, "extract");
    assert!(
        !source.is_opaque(),
        "the fixture source is genuinely transparent"
    );

    let req = GenerationRequest {
        prompt: "the red fox on a transparent background".into(),
        width: 64,
        height: 64,
        conditioning: vec![Conditioning::ReferenceRgba {
            image: source.clone(),
            strength: None,
        }],
        ..Default::default()
    };
    let refs = candle_gen_qwen_image_2_1::collect_references(&req).unwrap();
    assert_eq!(refs, vec![source.clone()], "the alpha survives the seam");

    let flattened = source.to_rgb_over_white().unwrap();
    let rgb_req = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: flattened.clone(),
            strength: None,
        }],
        ..req.clone()
    };
    let rgb_refs = candle_gen_qwen_image_2_1::collect_references(&rgb_req).unwrap();
    assert!(
        rgb_refs[0].is_opaque(),
        "an RGB reference widens to A = 255"
    );
    assert_ne!(
        rgb_refs[0], refs[0],
        "sending the flattened RGB is a DIFFERENT request from sending the transparent layer — \
         the VAE would receive white where the layer is transparent"
    );
}

/// **The output surface, through the registry-facing generator contract.** `Rgba` on this
/// provider yields `GenerationOutput::ImagesRgba`; the default still yields
/// `GenerationOutput::Images`, and the two agree under the white composite.
#[test]
fn the_generator_emits_rgba_on_request_and_rgb_by_default() {
    // Through the catalog path a consumer actually uses, so the descriptor's
    // `supports_alpha_output` and the shared request floor are both in the loop.
    let generator = candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(
            "qwen_image_2_1",
            &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())),
        )
        .expect("the tiny snapshot loads through the catalog path");

    let base = GenerationRequest {
        prompt: "a red fox in the forest".into(),
        width: 32,
        height: 32,
        steps: Some(2),
        seed: Some(3),
        ..Default::default()
    };

    let rgb = match generator.generate(&base, &mut |_| {}).unwrap() {
        GenerationOutput::Images(images) => images,
        other => panic!("the default request must emit RGB, got {other:?}"),
    };

    let mut asked = base.clone();
    asked.output_channels = OutputChannels::Rgba;
    generator
        .validate(&asked)
        .expect("this provider advertises alpha output");
    let rgba = match generator.generate(&asked, &mut |_| {}).unwrap() {
        GenerationOutput::ImagesRgba(images) => images,
        other => panic!("an Rgba request must emit ImagesRgba, got {other:?}"),
    };

    assert_eq!(rgb.len(), 1);
    assert_eq!(rgba.len(), 1);
    rgba[0].validate().unwrap();
    assert_eq!(rgba[0].pixels.len(), rgb[0].pixels.len() / 3 * 4);
    assert_eq!(
        (rgba[0].width, rgba[0].height),
        (rgb[0].width, rgb[0].height)
    );

    // Same seed, same decode: the RGB output is the RGBA output over white.
    let composited = rgba[0].to_rgb_over_white().unwrap();
    let worst = rgb[0]
        .pixels
        .iter()
        .zip(&composited.pixels)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap_or(0);
    eprintln!("generator: RGB vs composited RGBA worst byte delta {worst}");
    assert!(worst <= COMPOSITE_LSB, "worst delta {worst}");
}
