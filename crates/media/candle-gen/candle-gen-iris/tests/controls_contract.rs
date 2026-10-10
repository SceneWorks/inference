//! sc-25681: the full generation-control surface through the provider catalog's production load
//! path on the miniature snapshot — every advertised control changes the render (nothing is
//! accepted and ignored), the prompt batch, the caption-overflow policies, the per-step preview,
//! adapters with their provenance, and the typed refusals of everything the route does not have.
//! The MLX twin's `controls_contract`, case for case.

use std::sync::{Arc, Mutex};

use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CaptionOverflowPolicy, Error as CoreError, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Precision, PreviewSink, WeightsSource,
};

use crate::common::{fixtures, tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b";
/// 12 caption tokens against the miniature's 10-token window minus its suffix.
const OVERFLOW: &str = "a red fox , a red fox , golden hour , snow in the";

fn load_at(
    adapters: Vec<AdapterSpec>,
    precision: Precision,
) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    spec.adapters = adapters;
    spec.precision = precision;
    candle_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec)
}

fn load_with(adapters: Vec<AdapterSpec>) -> Result<Box<dyn Generator>, CoreError> {
    load_at(adapters, Precision::Bf16)
}

fn load() -> Box<dyn Generator> {
    load_with(Vec::new()).expect("the tiny snapshot loads")
}

fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the snow".to_owned(),
        width: 32,
        height: 16,
        steps: Some(4),
        seed: Some(42),
        ..Default::default()
    }
}

/// A low-contrast render (CFG off, 8 steps) whose output stays mostly inside the clamp, so the
/// preview comparison has unclamped patch cells to check.
fn quiet() -> GenerationRequest {
    GenerationRequest {
        guidance: Some(1.0),
        steps: Some(8),
        ..request()
    }
}

fn images(g: &dyn Generator, req: &GenerationRequest) -> Vec<Image> {
    g.validate(req).unwrap();
    match g.generate(req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(images) => images,
        other => panic!("expected images, got {other:?}"),
    }
}

#[test]
fn every_advertised_control_changes_the_render() {
    let g = load();
    let base = images(g.as_ref(), &request());
    let cases: Vec<(&str, GenerationRequest)> = vec![
        (
            "steps",
            GenerationRequest {
                steps: Some(5),
                ..request()
            },
        ),
        (
            "seed",
            GenerationRequest {
                seed: Some(43),
                ..request()
            },
        ),
        (
            "sampler euler (order 1)",
            GenerationRequest {
                sampler: Some("euler".into()),
                ..request()
            },
        ),
        (
            "scheduler_shift",
            GenerationRequest {
                scheduler_shift: Some(2.0),
                ..request()
            },
        ),
        (
            "guidance",
            GenerationRequest {
                guidance: Some(5.0),
                ..request()
            },
        ),
        (
            "cfg_interval",
            GenerationRequest {
                cfg_interval: Some((0.0, 0.7)),
                ..request()
            },
        ),
        (
            "negative_prompt",
            GenerationRequest {
                negative_prompt: Some("golden hour".into()),
                ..request()
            },
        ),
        (
            "prompt",
            GenerationRequest {
                prompt: "golden hour snow".into(),
                ..request()
            },
        ),
    ];
    for (name, req) in cases {
        let out = images(g.as_ref(), &req);
        assert!(
            out[0].pixels != base[0].pixels,
            "{name} did not change the render"
        );
    }
    // `dpmpp_2m` names the default solver: the same render as no sampler at all.
    let named = images(
        g.as_ref(),
        &GenerationRequest {
            sampler: Some("dpmpp_2m".into()),
            ..request()
        },
    );
    assert!(
        named[0].pixels == base[0].pixels,
        "dpmpp_2m is the default solver"
    );
    // Width/height/aspect: portrait and landscape canvases render at their own geometry.
    for (w, h) in [(16, 32), (48, 16)] {
        let out = images(
            g.as_ref(),
            &GenerationRequest {
                width: w,
                height: h,
                ..request()
            },
        );
        assert_eq!((out[0].width, out[0].height), (w, h));
    }
}

#[test]
fn a_prompt_batch_renders_one_image_per_prompt_per_count() {
    // FP32 compute, so the batched solve is compared with the single renders at f32 rounding rather
    // than at bf16's batch-size-dependent GEMM rounding.
    let g = load_at(Vec::new(), Precision::Fp32).unwrap();
    let prompts = ["a red fox in the snow", "golden hour snow"];
    let req = GenerationRequest {
        prompt: String::new(),
        prompt_batch: prompts.iter().map(|p| p.to_string()).collect(),
        count: 2,
        ..request()
    };
    let mut last = (0, 0);
    g.validate(&req).unwrap();
    let GenerationOutput::Images(batch) = g
        .generate(&req, &mut |p| {
            if let candle_gen::gen_core::Progress::Step { current, total } = p {
                last = (current, total);
            }
        })
        .unwrap()
    else {
        panic!("expected images")
    };
    assert_eq!(batch.len(), 4, "prompts × count");
    assert_eq!(last, (8, 8), "one progress bar over count × steps");
    // Image k = c·rows + j renders prompt j from seed + k; a batched solve matches the single
    // render of the same prompt and seed up to batched-GEMM rounding.
    for (k, img) in batch.iter().enumerate() {
        let single = images(
            g.as_ref(),
            &GenerationRequest {
                prompt: prompts[k % 2].into(),
                seed: Some(42 + k as u64),
                ..request()
            },
        );
        let max = img
            .pixels
            .iter()
            .zip(&single[0].pixels)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(
            max <= 1,
            "batch image {k} vs its single render: max pixel Δ {max}"
        );
    }
    assert!(batch[0].pixels != batch[1].pixels, "two prompts, one image");
    // A prompt batch and a prompt together are refused, as is a batch past the per-request cap.
    let both = GenerationRequest {
        prompt: "x".into(),
        ..req.clone()
    };
    assert!(g.validate(&both).is_err());
    let too_many = GenerationRequest { count: 5, ..req };
    assert!(g.validate(&too_many).is_err());
}

#[test]
fn caption_overflow_policies_are_honoured() {
    let g = load();
    let req = |policy: Option<CaptionOverflowPolicy>| GenerationRequest {
        prompt: OVERFLOW.into(),
        caption_overflow: policy,
        ..request()
    };
    // The checkpoint's own policy (`warn`): truncate and report.
    let report = g.generate_with_report(&req(None), &mut |_| {}).unwrap();
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    assert_eq!(report.warnings[0].code, "caption_truncated");
    let warned = report.output.unwrap();
    // `silent`: the same truncated render, no warning.
    let report = g
        .generate_with_report(&req(Some(CaptionOverflowPolicy::Silent)), &mut |_| {})
        .unwrap();
    assert!(report.warnings.is_empty());
    match (&warned, &report.output.unwrap()) {
        (GenerationOutput::Images(a), GenerationOutput::Images(b)) => {
            assert!(
                a[0].pixels == b[0].pixels,
                "silent truncates exactly like warn"
            )
        }
        _ => panic!("expected images"),
    }
    // `error`: refused before the tower runs.
    let err = g
        .generate(&req(Some(CaptionOverflowPolicy::Error)), &mut |_| {})
        .expect_err("an overflowing caption is an error under `error`");
    assert!(err.to_string().contains("tokenizes to"), "{err}");
    // A caption that fits is fine under `error`.
    let fits = GenerationRequest {
        caption_overflow: Some(CaptionOverflowPolicy::Error),
        ..request()
    };
    let report = g.generate_with_report(&fits, &mut |_| {}).unwrap();
    assert!(report.warnings.is_empty());
}

#[test]
fn previews_stream_the_patch_pooled_clean_image_every_step() {
    let g = load();
    assert!(g.descriptor().capabilities.supports_preview);
    let frames = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&frames);
    let req = GenerationRequest {
        preview: PreviewSink::new(move |frame| sink.lock().unwrap().push(frame)),
        ..quiet()
    };
    let out = images(g.as_ref(), &req);
    let frames = frames.lock().unwrap();
    let numbers: Vec<(u32, u32)> = frames.iter().map(|f| (f.current, f.total)).collect();
    assert_eq!(numbers, (1..=8).map(|i| (i, 8)).collect::<Vec<_>>());
    // The miniature's 4-px patch grid; the image develops across the steps.
    let last = &frames.last().unwrap().image;
    assert!(
        frames[0].image.pixels != last.pixels,
        "the preview does not develop"
    );
    assert_eq!((last.width, last.height), (8, 4));
    // The final step's x0 IS the output before its clamp (the terminal update is the exact x0
    // projection), and the frame is that x0 averaged over each patch cell, then decoded. So, per
    // cell and channel, against the mean `m` of the clamped output pixels: a cell the clamp never
    // touched matches `m` (to the two u8 roundings); one clipped only at 255 can only sit above `m`
    // (the clipped values were larger), one clipped only at 0 only below, an all-255 cell is 255
    // and an all-0 cell 0.
    let (w, cell) = (out[0].width as usize, 4usize);
    let mut exact = 0;
    for py in 0..4 {
        for px in 0..8 {
            for c in 0..3 {
                let values: Vec<f32> = (0..cell * cell)
                    .map(|i| {
                        let (y, x) = (py * cell + i / cell, px * cell + i % cell);
                        out[0].pixels[(y * w + x) * 3 + c] as f32
                    })
                    .collect();
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let got = last.pixels[(py * 8 + px) * 3 + c] as f32;
                let (high, low) = (values.contains(&255.0), values.contains(&0.0));
                let at = format!("frame ({px},{py},{c}): {got} vs cell mean {mean}");
                match (high, low) {
                    (false, false) => {
                        assert!((got - mean).abs() <= 1.0, "{at}");
                        exact += 1;
                    }
                    (true, false) => assert!(got >= mean - 1.0, "{at}"),
                    (false, true) => assert!(got <= mean + 1.0, "{at}"),
                    (true, true) => {}
                }
            }
        }
    }
    eprintln!("preview: {exact} unclamped cells match the pooled output exactly");
    // An inert sink costs nothing and changes nothing.
    assert!(
        images(g.as_ref(), &quiet())[0].pixels == out[0].pixels,
        "an inert sink changes nothing"
    );
}

#[test]
fn adapters_load_through_the_catalog_and_report_provenance() {
    let lora = AdapterSpec::new(
        fixtures().join("iris_lora.safetensors"),
        0.75,
        AdapterKind::Lora,
    );
    let lokr = AdapterSpec::new(
        fixtures().join("iris_lokr.safetensors"),
        1.0,
        AdapterKind::Lokr,
    );
    let g = load_with(vec![lora.clone(), lokr.clone()]).unwrap();
    let caps = &g.descriptor().capabilities;
    assert!(caps.supports_lora && caps.supports_lokr);
    let adapted = images(g.as_ref(), &request());
    let reports = g.adapter_apply_reports();
    assert_eq!(
        reports
            .iter()
            .map(|r| r.adapter_path.clone())
            .collect::<Vec<_>>(),
        [lora.path.clone(), lokr.path.clone()]
    );
    assert!(reports
        .iter()
        .all(|r| r.applied == 4 && r.skipped.is_empty()));
    let bare = images(load().as_ref(), &request());
    assert!(
        adapted[0].pixels != bare[0].pixels,
        "the adapters changed nothing"
    );
    // A scale-0 stack is the bare render.
    let off = load_with(vec![AdapterSpec { scale: 0.0, ..lora }]).unwrap();
    assert!(
        images(off.as_ref(), &request())[0].pixels == bare[0].pixels,
        "a scale-0 adapter is the bare render"
    );
    // Another task's adapter is refused at load, by name.
    let depth = AdapterSpec::new(
        fixtures().join("iris_lora_depth_task.safetensors"),
        1.0,
        AdapterKind::Lora,
    );
    let err = load_with(vec![depth]).err().expect("must be refused");
    assert!(
        matches!(&err, CoreError::Unsupported(m) if m.contains("depth")),
        "{err}"
    );
}

#[test]
fn controls_the_route_does_not_have_are_refused() {
    let g = load();
    for (name, req) in [
        (
            "sampler heun",
            GenerationRequest {
                sampler: Some("heun".into()),
                ..request()
            },
        ),
        (
            "scheduler",
            GenerationRequest {
                scheduler: Some("karras".into()),
                ..request()
            },
        ),
        (
            "timestep_to_start_cfg",
            GenerationRequest {
                timestep_to_start_cfg: Some(2),
                ..request()
            },
        ),
        (
            "true_cfg",
            GenerationRequest {
                true_cfg: Some(4.0),
                ..request()
            },
        ),
        (
            "guidance_method",
            GenerationRequest {
                guidance_method: Some("apg".into()),
                ..request()
            },
        ),
        (
            "interval with CFG off",
            GenerationRequest {
                guidance: Some(1.0),
                cfg_interval: Some((0.25, 0.75)),
                ..request()
            },
        ),
    ] {
        assert!(
            matches!(g.validate(&req), Err(CoreError::Unsupported(_))),
            "{name}: {:?}",
            g.validate(&req)
        );
    }
    for (name, req) in [
        (
            "shift 0",
            GenerationRequest {
                scheduler_shift: Some(0.0),
                ..request()
            },
        ),
        (
            "interval outside [0, 1]",
            GenerationRequest {
                cfg_interval: Some((0.5, 1.5)),
                ..request()
            },
        ),
        // CFG on, but no evaluation time of a 4-step plan falls inside the interval.
        (
            "empty interval",
            GenerationRequest {
                cfg_interval: Some((0.9999, 1.0)),
                ..request()
            },
        ),
        (
            "width off the 16-px grid",
            GenerationRequest {
                width: 40,
                ..request()
            },
        ),
        (
            "negative prompt with CFG off",
            GenerationRequest {
                guidance: Some(1.0),
                negative_prompt: Some("blurry".into()),
                ..request()
            },
        ),
    ] {
        assert!(g.validate(&req).is_err(), "{name} must be refused");
    }
}
