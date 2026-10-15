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
use candle_gen::gen_core::iris::{GenerationParams, IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, PreviewSink, Quant,
    WeightsSource,
};
use candle_gen_iris::memory_strategy::generation_contract;
use candle_gen_iris::{
    denoise, encode, load_backbone_quantized, noise, IrisDiT, IrisTextEncoder, MODEL_ID,
};

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
/// on an unadapted load is caught by `an_adapted_quantized_backbone_keeps_the_unaligned_projections_dense`
/// and the `memory_strategy` pricing unit test.
#[test]
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
}

/// An adapted quantized load keeps the unaligned SwiGLU down projections dense (an adapter
/// residual reads the unpadded input), prices exactly that, and still renders.
///
/// *Mutation that reds this:* `BackboneQuant::for_load` padding under adapters (pricing mismatch).
#[test]
fn an_adapted_quantized_backbone_keeps_the_unaligned_projections_dense() {
    let scratch = tempfile::tempdir().unwrap();
    let tower = aligned_tiny_text_encoder(scratch.path());
    let adapters = vec![candle_gen::gen_core::AdapterSpec::new(
        crate::common::fixtures().join("iris_lora.safetensors"),
        1.0,
        candle_gen::gen_core::AdapterKind::Lora,
    )];
    let (dit, reports) = load_backbone_quantized(
        &tiny_backbone(),
        &tiny_config(),
        DType::BF16,
        &cpu(),
        &adapters,
        IrisTask::Generation,
        MODEL_ID,
        Some(Quant::Q4),
    )
    .unwrap();
    assert_eq!(reports.len(), 1);
    assert!(dit.is_quantized());
    let mut adapted = spec(&tower, Some(Quant::Q4));
    adapted.adapters = adapters;
    let priced = generation_contract(&adapted)
        .unwrap()
        .asset_facts
        .transformer_bytes;
    assert_priced("adapted backbone Q4", dit.resident_weight_bytes(), priced);
    assert!(dit.resident_weight_bytes() > backbone(Some(Quant::Q4)).resident_weight_bytes());
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
/// quantizing the embedding (pricing mismatch); a forward that quantizes the activation is not
/// caught here (CPU) — `candle-llm`'s `a_dequantized_forward_projection_never_quantizes_the_activation`
/// holds that.
#[test]
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
}
