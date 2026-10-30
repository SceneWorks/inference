//! sc-26266: the `q8` / `q4` tiers on the committed miniature, on the Candle CPU device — the
//! backbone and the Qwen3-VL tower are quantized at load, hold GGML-packed weights (measured from
//! the loaded model, checked against the memory contract's pricing), track the dense tier within a
//! declared bound, and the whole pipeline renders through the production load path at each tier.
//!
//! The released miniature tower's MLP width (48) is not a multiple of the 32-wide GGML block, so
//! it has no quantized tier (a typed refusal, asserted below). The tower tests run on an
//! **aligned** derivative written into a temp dir: the same snapshot with `intermediate_size: 64`
//! and seeded uniform(±0.035) MLP weights of the widened shapes (σ ≈ 0.02, the MLX twin's), every
//! other tensor byte-for-byte the fixture.
//!
//! Tolerances are relative to the dense tier on the same inputs (not to upstream): quantization
//! error is a property of the tier, and these gates bound it rather than claim parity. The
//! real-weight render gate is `quant_real_weights`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, PreviewSink, Quant,
    WeightsSource,
};
use candle_gen_iris::memory_strategy::generation_contract;
use candle_gen_iris::{
    denoise, encode, load_backbone_quantized, noise, IrisDiT, IrisTextEncoder, MODEL_ID,
};
use gen_core_iris::{GenerationParams, IrisTask, TEXT_ENCODER_COMPONENT};

use crate::common::{cpu, errors, rel_and_cos, tiny_backbone, tiny_config};

const TIERS: [Option<Quant>; 3] = [None, Some(Quant::Q8), Some(Quant::Q4)];

/// A deterministic uniform(±0.035) fill (σ ≈ 0.02) — a fixed LCG, so the derivative is the same
/// on every host.
fn seeded(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit * 2.0 - 1.0) * 0.035
        })
        .collect()
}

/// Write the aligned miniature tower (see the module docs) under `root`; returns its directory.
fn aligned_tiny_text_encoder(root: &Path) -> PathBuf {
    let source = crate::common::tiny_text_encoder();
    let dir = root.join("text_encoder");
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_owned();
        if name != "model.safetensors" && name != "config.json" {
            std::fs::copy(&path, dir.join(name)).unwrap();
        }
    }
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    config["text_config"]["intermediate_size"] = 64.into();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut tensors: HashMap<String, Tensor> =
        candle_gen::candle_core::safetensors::load(source.join("model.safetensors"), &Device::Cpu)
            .unwrap();
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort();
    for (i, name) in names.iter().enumerate() {
        let shape = if name.contains(".mlp.gate_proj.") || name.contains(".mlp.up_proj.") {
            (64, 32)
        } else if name.contains(".mlp.down_proj.") {
            (32, 64)
        } else {
            continue;
        };
        let values = seeded(shape.0 * shape.1, 26266 + i as u64);
        tensors.insert(
            name.clone(),
            Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
        );
    }
    candle_gen::candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
    dir
}

fn spec(text_encoder: &Path, quant: Option<Quant>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.quantize = quant;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(text_encoder.to_path_buf()),
    );
    spec
}

fn backbone(quant: Option<Quant>) -> IrisDiT {
    load_backbone_quantized(
        &tiny_backbone(),
        &tiny_config(),
        DType::BF16,
        &cpu(),
        &[],
        IrisTask::Generation,
        MODEL_ID,
        quant,
    )
    .unwrap()
    .0
}

/// `measured` against the contract's `priced` bytes: equal on a CPU build; a CUDA build's contract
/// adds candle's per-tensor CUDA row padding, which a CPU-device load does not allocate.
fn assert_priced(what: &str, measured: u64, priced: u64) {
    eprintln!("{what}: resident {measured} B, priced {priced} B");
    if cfg!(feature = "cuda") {
        assert!(measured <= priced, "{what}: {measured} > priced {priced}");
    } else {
        assert_eq!(measured, priced, "{what}");
    }
}

/// The backbone at each tier: GGML-packed resident bytes that shrink q8 → q4 and equal the
/// contract's pricing; the solver's image within the tier's bound of the dense tier's from the
/// same conditioning and noise.
///
/// Bounds (on the [−1, 1] image, 3 CFG-3 steps over a random-weight miniature, which amplifies
/// weight error far more than a trained network does). Measured on the Candle CPU device (arm64
/// macOS): q8 mean |Δ| 1.19e-2 (cos 0.99966), q4 mean |Δ| 1.06e-1 (cos 0.9753). Gated at the MLX
/// twin's bounds: q8 ≤ 6e-2 / cos ≥ 0.99 and q4 ≤ 2.5e-1 / cos ≥ 0.90.
///
/// *Mutations that red this:* a tier that loads dense (bytes do not shrink, pricing mismatch);
/// forgetting the input padding in the forward (a shape error); packing at the wrong width or a
/// block the contract does not price (pricing mismatch). Leaving the unaligned projections dense
/// (adapted or not) is caught by `an_adapted_load_holds_the_packed_model_at_a_tier` and the
/// `memory_strategy` pricing unit test.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_backbone_holds_packed_projections_at_each_tier() {
    let scratch = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(scratch.path());
    let config = tiny_config();
    let te = IrisTextEncoder::load(&tower, &config.text_encoder, &cpu()).unwrap();
    let req = GenerationRequest {
        prompt: "a red fox in the snow".into(),
        width: 16,
        height: 16,
        steps: Some(3),
        guidance: Some(3.0),
        seed: Some(7),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&req, 0, &config).unwrap();
    let conditioning = encode(&te, &params).unwrap();
    let z = noise(7, config.model.in_channels, 16, 16, &cpu()).unwrap();

    let mut images = Vec::new();
    let mut bytes = Vec::new();
    for quant in TIERS {
        let dit = backbone(quant);
        assert_eq!(dit.is_quantized(), quant.is_some(), "{quant:?}");
        let resident = dit.resident_weight_bytes();
        let priced = generation_contract(&spec(&tower, quant))
            .unwrap()
            .asset_facts
            .transformer_bytes;
        assert_priced(&format!("backbone {quant:?}"), resident, priced);
        bytes.push(resident);
        images.push(
            denoise(
                &dit,
                &config.flow,
                &conditioning,
                &z,
                &params,
                &CancelFlag::new(),
                |_| {},
                &PreviewSink::default(),
            )
            .unwrap(),
        );
    }
    assert!(bytes[1] < bytes[0] && bytes[2] < bytes[1], "{bytes:?}");
    for (tier, image, mean_bound, min_cos) in [
        ("q8", &images[1], 6e-2_f32, 0.99),
        ("q4", &images[2], 2.5e-1, 0.90),
    ] {
        let (max_abs, _, mean) = errors(image, &images[0]);
        let (rel, cos) = rel_and_cos(image, &images[0]);
        eprintln!(
            "backbone {tier} vs bf16 image: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
             rel={rel:.3e} cos={cos:.5}"
        );
        assert!(mean <= mean_bound, "{tier}: mean |Δ| {mean} > {mean_bound}");
        assert!(cos >= min_cos, "{tier}: cos {cos} < {min_cos}");
        assert_ne!(
            crate::common::host_f32(image),
            crate::common::host_f32(&images[0]),
            "{tier}: the dense image"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED quant_tiers::the_backbone_holds_packed_projections_at_each_tier");
}

/// The committed LoRA fixture plus one more module on `blocks.2.mlp.w2` — a 170-wide SwiGLU down
/// projection, the one family a quantized tier packs over a zero-padded input — written under
/// `root` with the fixture's `__metadata__` (identity stamps) kept verbatim. Returns its path.
fn lora_on_an_unaligned_projection(root: &Path) -> PathBuf {
    let bytes = std::fs::read(crate::common::fixtures().join("iris_lora.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let mut data = bytes[8 + header_len..].to_vec();
    let module = "transformer.blocks.2.mlp.w2";
    // Factors of magnitude ~0.5 (the MLX twin's σ), so the residual is visible through q4 error.
    let factor = |len: usize, seed: u64| -> Vec<f32> {
        seeded(len, seed).into_iter().map(|v| v * 15.0).collect()
    };
    for (name, shape, values) in [
        (
            format!("{module}.lora_A.weight"),
            vec![2, 170],
            factor(2 * 170, 1),
        ),
        (
            format!("{module}.lora_B.weight"),
            vec![64, 2],
            factor(64 * 2, 2),
        ),
        (format!("{module}.alpha"), vec![1], vec![4.0_f32]),
    ] {
        let begin = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            name,
            serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()] }),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let path = root.join("iris_lora_w2.safetensors");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    std::fs::write(&path, out).unwrap();
    path
}

/// An adapted load at the q4 tier is the q4 model: every projection — the adapted ones, and the
/// 170-wide adapted down projection over its zero-padded input — is packed, so its resident bytes
/// are the unadapted q4 model's and equal the contract's pricing of the adapted spec (which is the
/// unadapted q4 pricing: the stack is an overlay); its image is within the q4 bound of the adapted
/// bf16 image, the residuals reach it (it differs from the unadapted q4 image), and the residual
/// on the padded projection is live (it differs from the same stack without that module).
///
/// Bounds are the backbone gate's q4 bounds (mean |Δ| ≤ 2.5e-1, cos ≥ 0.90), the MLX twin's.
///
/// *Mutations that red this:* keeping the unaligned adapted projection dense, at load or in the
/// pricing (bytes); dropping an adapted projection's residuals on the packed base (the image
/// equals the unadapted / without-w2 image); handing the residuals the padded input (a shape
/// error).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_adapted_load_holds_the_packed_model_at_a_tier() {
    let scratch = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(scratch.path());
    let w2_lora = lora_on_an_unaligned_projection(scratch.path());
    let lora = |path: &Path| {
        vec![candle_gen::gen_core::AdapterSpec::new(
            path.to_path_buf(),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        )]
    };
    let with_w2 = lora(&w2_lora);
    let without_w2 = lora(&crate::common::fixtures().join("iris_lora.safetensors"));
    let config = tiny_config();
    let te = IrisTextEncoder::load(&tower, &config.text_encoder, &cpu()).unwrap();
    let req = GenerationRequest {
        prompt: "a red fox in the snow".into(),
        width: 16,
        height: 16,
        steps: Some(3),
        guidance: Some(3.0),
        seed: Some(7),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&req, 0, &config).unwrap();
    let conditioning = encode(&te, &params).unwrap();
    let z = noise(7, config.model.in_channels, 16, 16, &cpu()).unwrap();
    let render = |dit: &IrisDiT| {
        denoise(
            dit,
            &config.flow,
            &conditioning,
            &z,
            &params,
            &CancelFlag::new(),
            |_| {},
            &PreviewSink::default(),
        )
        .unwrap()
    };
    let load = |adapters: &[candle_gen::gen_core::AdapterSpec], quant: Option<Quant>| {
        let (dit, reports) = load_backbone_quantized(
            &tiny_backbone(),
            &config,
            DType::BF16,
            &cpu(),
            adapters,
            IrisTask::Generation,
            MODEL_ID,
            quant,
        )
        .unwrap();
        assert_eq!(reports.len(), adapters.len());
        dit
    };
    let priced = |adapters: &[candle_gen::gen_core::AdapterSpec], quant: Option<Quant>| {
        let mut spec = spec(&tower, quant);
        spec.adapters = adapters.to_vec();
        generation_contract(&spec)
            .unwrap()
            .asset_facts
            .transformer_bytes
    };

    let plain_q4 = backbone(Some(Quant::Q4));
    let adapted_bf16 = load(&with_w2, None);
    let adapted_q4 = load(&with_w2, Some(Quant::Q4));
    assert!(adapted_q4.is_quantized());
    let resident = adapted_q4.resident_weight_bytes();
    let adapted_priced = priced(&with_w2, Some(Quant::Q4));
    assert_eq!(resident, plain_q4.resident_weight_bytes());
    assert_eq!(adapted_priced, priced(&[], Some(Quant::Q4)));
    assert_priced("adapted backbone Q4", resident, adapted_priced);

    let image = render(&adapted_q4);
    let reference = render(&adapted_bf16);
    let (max_abs, _, mean) = errors(&image, &reference);
    let (rel, cos) = rel_and_cos(&image, &reference);
    eprintln!(
        "adapted q4 vs adapted bf16 image: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
         rel={rel:.3e} cos={cos:.5}"
    );
    assert!(mean <= 2.5e-1, "mean |Δ| {mean} > 2.5e-1");
    assert!(cos >= 0.90, "cos {cos} < 0.90");
    let (_, _, from_plain) = errors(&image, &render(&plain_q4));
    let (_, _, from_without_w2) = errors(&image, &render(&load(&without_w2, Some(Quant::Q4))));
    eprintln!(
        "adapted q4 mean|Δ| vs unadapted q4 {from_plain:.3e}, vs without the w2 module \
         {from_without_w2:.3e}"
    );
    assert!(
        from_plain > 0.0,
        "the adapted q4 image is the unadapted one"
    );
    assert!(
        from_without_w2 > 0.0,
        "the padded projection's residual did not reach the q4 image"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED quant_tiers::an_adapted_load_holds_the_packed_model_at_a_tier");
}

/// The text tower at each tier: packed projections whose measured resident bytes equal the
/// contract's conditioning pricing (the tied embedding counted once), and conditioning within the
/// tier's bound of the dense tower's.
///
/// Bounds (relative L2 over the stacked hidden states the backbone reads, 36 random-weight
/// layers, f32 dense reference). Measured on the Candle CPU device (arm64 macOS): q8 rel 1.44e-2
/// (cos 0.99990), q4 rel 0.375 (cos 0.928). Gated at the MLX twin's bounds: q8 ≤ 6e-2 / cos ≥ 0.998 and q4 ≤ 0.5 /
/// cos ≥ 0.90.
///
/// *Mutations that red this:* loading dense at a quantized tier; counting the tied head twice;
/// quantizing the embedding (pricing mismatch); a tower spec without
/// `with_dequantized_forward` (every packed projection would fall back to `QMatMul`, which
/// quantizes the activation — the census below counts weight-only projections; the conditioning
/// bounds alone absorb it). That the weight-only forward is exact is held by `candle-llm`'s
/// `a_dequantized_forward_projection_never_quantizes_the_activation`.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_text_tower_holds_packed_projections_at_each_tier() {
    let scratch = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(scratch.path());
    let config = tiny_config();
    let mut states = Vec::new();
    let mut bytes = Vec::new();
    for quant in TIERS {
        let te =
            IrisTextEncoder::load_quantized(&tower, &config.text_encoder, &cpu(), quant).unwrap();
        assert_eq!(te.is_quantized(), quant.is_some(), "{quant:?}");
        let census = te.projection_census();
        if quant.is_some() {
            // Every attention / MLP projection packed and served weight-only; only the tied head
            // stays dense.
            assert!(census.ggml.count > 0, "{quant:?}: {census:?}");
            assert_eq!(
                census.ggml_weight_only, census.ggml.count,
                "{quant:?}: {census:?}"
            );
            assert_eq!(census.dense.count, 1, "{quant:?}: {census:?}");
        } else {
            assert_eq!(census.ggml.count, 0, "{census:?}");
        }
        let resident = te.resident_weight_bytes();
        let priced = generation_contract(&spec(&tower, quant))
            .unwrap()
            .asset_facts
            .conditioning_bytes;
        assert_priced(&format!("tower {quant:?}"), resident, priced);
        bytes.push(resident);
        states.push(te.encode("a red fox in the snow").unwrap().states);
    }
    assert!(bytes[1] < bytes[0] && bytes[2] < bytes[1], "{bytes:?}");
    for (tier, got, bound, min_cos) in [
        ("q8", &states[1], 6e-2_f64, 0.998),
        ("q4", &states[2], 0.5, 0.90),
    ] {
        let (rel, cos) = rel_and_cos(got, &states[0]);
        eprintln!("tower {tier} vs dense conditioning: rel={rel:.3e} cos={cos:.5}");
        assert!(rel <= bound, "{tier}: rel {rel} > {bound}");
        assert!(cos >= min_cos, "{tier}: cos {cos} < {min_cos}");
        assert!(rel > 0.0, "{tier}: the dense conditioning");
    }

    // The released miniature's MLP width is not GGML-aligned: a typed refusal by name.
    match IrisTextEncoder::load_quantized(
        &crate::common::tiny_text_encoder(),
        &config.text_encoder,
        &cpu(),
        Some(Quant::Q4),
    ) {
        Err(candle_gen::CandleError::Unsupported(message)) => {
            assert!(message.contains("block size"), "{message}")
        }
        other => panic!("expected a typed refusal, got {:?}", other.err()),
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED quant_tiers::the_text_tower_holds_packed_projections_at_each_tier");
}

/// The production path at each tier: `load` of a quantized spec renders a non-constant image
/// from a fixed seed, under both residency policies, within the tier's bound of the dense render,
/// and publishes a contract that names the tier.
///
/// Bounds (mean |Δ| on RGB8, 0–255; same miniature caveat as the backbone gate, now with the tower
/// quantized too). Measured on the Candle CPU device (arm64 macOS): q8 1.46, q4 12.4, identical
/// under both policies. Gated at the MLX twin's bounds: q8 ≤ 6 and q4 ≤ 24.
///
/// *Mutation that reds this:* `load` refusing a quantized spec, or dropping `spec.quantize` on the
/// way to both component loaders (the q4 render would equal the dense one byte for byte —
/// asserted unequal). Which component a tier reaches is gated by the two tests above.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_production_load_renders_at_each_tier() {
    let scratch = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(scratch.path());
    let req = GenerationRequest {
        prompt: "a red fox in the snow".into(),
        width: 32,
        height: 16,
        steps: Some(3),
        seed: Some(42),
        ..Default::default()
    };
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let mut renders = Vec::new();
        for quant in TIERS {
            let spec = spec(&tower, quant).with_offload_policy(policy);
            let generator = candle_gen_iris::load(&spec).unwrap();
            assert_eq!(
                generator
                    .memory_strategy_contract()
                    .unwrap()
                    .asset_facts
                    .base_bytes,
                generation_contract(&spec).unwrap().asset_facts.base_bytes
            );
            let GenerationOutput::Images(images) = generator.generate(&req, &mut |_| {}).unwrap()
            else {
                panic!("expected images");
            };
            let pixels = images[0].pixels.clone();
            let lo = *pixels.iter().min().unwrap();
            let hi = *pixels.iter().max().unwrap();
            assert!(hi > lo, "{policy:?} {quant:?}: a constant image");
            renders.push(pixels);
        }
        for (tier, pixels, bound) in [("q8", &renders[1], 6.0_f64), ("q4", &renders[2], 24.0)] {
            let mean = pixels
                .iter()
                .zip(&renders[0])
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .sum::<f64>()
                / pixels.len() as f64;
            eprintln!("load {policy:?} {tier} vs bf16: mean |Δ| = {mean:.3} / 255");
            assert!(
                mean <= bound,
                "{policy:?} {tier}: mean |Δ| {mean} > {bound}"
            );
            assert_ne!(
                pixels, &renders[0],
                "{policy:?} {tier} rendered the dense image"
            );
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quant_tiers::the_production_load_renders_at_each_tier"
    );
}
