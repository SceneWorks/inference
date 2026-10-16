//! sc-26266: the `q8` / `q4` tiers on the committed miniature, every body in a CPU child with Metal
//! trapped (`common::cpu_process`) — the backbone and the Qwen3-VL tower are quantized at load,
//! hold packed weights (measured from the loaded model, checked against the memory contract's
//! pricing), track the dense tier within a declared bound, and the whole pipeline renders through
//! the catalog's production load path at each tier.
//!
//! The released miniature tower's MLP width (48) is not a multiple of any affine group, so it has
//! no quantized tier (a typed refusal, asserted below). The tower tests run on an **aligned**
//! derivative written into a temp dir: the same snapshot with `intermediate_size: 64` and seeded
//! N(0, 0.02²) MLP weights of the widened shapes, everything else byte-for-byte the fixture.
//!
//! Tolerances are relative to the dense tier on the same inputs (not to upstream): affine
//! quantization error is a property of the tier, and these gates bound it rather than claim
//! parity. The real-weight 512² render gate is `quant_real_weights`.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::{GenerationParams, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, PreviewSink, Quant, WeightsSource,
};
use mlx_gen_iris::memory_strategy::memory_strategy_contract;
use mlx_gen_iris::{
    denoise, encode, load_backbone_quantized, noise, IrisDiT, IrisTextEncoder, MODEL_ID,
};
use mlx_rs::{Array, Dtype};

use crate::common::{cpu_process, errors, rel_and_cos, tiny_backbone, tiny_config};

const TIERS: [Option<Quant>; 3] = [None, Some(Quant::Q8), Some(Quant::Q4)];

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

    let mut tensors = Array::load_safetensors(source.join("model.safetensors")).unwrap();
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort();
    for (i, name) in names.iter().enumerate() {
        let shape = if name.contains(".mlp.gate_proj.") || name.contains(".mlp.up_proj.") {
            [64, 32]
        } else if name.contains(".mlp.down_proj.") {
            [32, 64]
        } else {
            continue;
        };
        let key = mlx_rs::random::key(26266 + i as u64).unwrap();
        let weight =
            mlx_rs::random::normal::<f32>(&shape[..], None, Some(0.02), Some(&key)).unwrap();
        tensors.insert(name.clone(), weight);
    }
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
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
        Dtype::Bfloat16,
        &[],
        IrisTask::Generation,
        MODEL_ID,
        quant,
    )
    .unwrap()
    .0
}

/// The backbone at each tier: packed resident bytes that shrink q8 → q4 and sit exactly the dense
/// tier's constant below the contract's pricing; the solver's image within the tier's bound of the
/// dense tier's from the same conditioning and noise.
///
/// Bounds (on the [−1, 1] image, 3 CFG-3 steps over a random-weight miniature, which amplifies
/// weight error far more than a trained network does). Measured on the MLX CPU stream: q8 mean
/// |Δ| 3.7e-2 (cos 0.9964), q4 mean |Δ| 1.53e-1 (cos 0.946). Gated at q8 ≤ 6e-2 / cos ≥ 0.99 and
/// q4 ≤ 2.5e-1 / cos ≥ 0.90.
///
/// The contract prices norm gains and biases at their stored f32 width while the loader holds the
/// biases in the compute dtype — a constant over-price no tier changes (the biases are never
/// packed). Every tier's `priced − resident` must equal the dense tier's, so a packed projection
/// held at any other width than priced reds in either direction.
///
/// *Mutations that red this:* a tier that loads dense (bytes do not shrink); quantizing without
/// padding the 170-wide projections (q4 resident below the padded pricing); forgetting the input
/// padding in the forward (a shape error); packing at the wrong width (bytes).
#[test]
fn the_backbone_holds_packed_projections_at_each_tier() {
    cpu_process(
        "quant_tiers::the_backbone_holds_packed_projections_at_each_tier",
        || {
            let scratch = tempfile::tempdir().unwrap();
            let tower = aligned_tiny_text_encoder(scratch.path());
            let config = tiny_config();
            let te = IrisTextEncoder::load(&tower, &config.text_encoder).unwrap();
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
            let z = noise(7, config.model.in_channels, 16, 16).unwrap();

            let mut images = Vec::new();
            let mut bytes = Vec::new();
            let mut slack = None;
            for quant in TIERS {
                let dit = backbone(quant);
                let resident = dit.resident_weight_bytes();
                let priced = memory_strategy_contract(MODEL_ID, &spec(&tower, quant))
                    .unwrap()
                    .asset_facts
                    .transformer_bytes;
                eprintln!("backbone {quant:?}: resident {resident} B, priced {priced} B");
                assert!(
                    resident <= priced,
                    "{quant:?}: {resident} > priced {priced}"
                );
                let gap = priced - resident;
                assert_eq!(
                    gap,
                    *slack.get_or_insert(gap),
                    "{quant:?}: priced − resident differs from the dense tier's"
                );
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
            for (tier, image, mean_bound) in
                [("q8", &images[1], 6e-2_f32), ("q4", &images[2], 2.5e-1)]
            {
                let (max_abs, _, mean) = errors(image, &images[0]);
                let (rel, cos) = rel_and_cos(image, &images[0]);
                eprintln!(
                    "backbone {tier} vs bf16 image: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
                     rel={rel:.3e} cos={cos:.5}"
                );
                assert!(mean <= mean_bound, "{tier}: mean |Δ| {mean} > {mean_bound}");
                let min_cos = if tier == "q8" { 0.99 } else { 0.90 };
                assert!(cos >= min_cos, "{tier}: cos {cos} < {min_cos}");
            }
        },
    );
}

/// The committed LoRA fixture plus one more module on `blocks.2.mlp.w2` — a 170-wide SwiGLU down
/// projection, the one family a quantized tier packs over a zero-padded input — written under
/// `root` with the fixture's identity stamps. Returns its path.
fn lora_on_an_unaligned_projection(root: &Path) -> PathBuf {
    let (mut tensors, metadata) = Array::load_safetensors_with_metadata(
        crate::common::fixtures().join("iris_lora.safetensors"),
    )
    .unwrap();
    let factor = |shape: [i32; 2], seed: u64| {
        let key = mlx_rs::random::key(seed).unwrap();
        mlx_rs::random::normal::<f32>(&shape[..], None, Some(0.5), Some(&key)).unwrap()
    };
    let module = "transformer.blocks.2.mlp.w2";
    tensors.insert(format!("{module}.lora_A.weight"), factor([2, 170], 1));
    tensors.insert(format!("{module}.lora_B.weight"), factor([64, 2], 2));
    tensors.insert(
        format!("{module}.alpha"),
        Array::from_slice(&[4.0_f32], &[1]),
    );
    let path = root.join("iris_lora_w2.safetensors");
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, &metadata, &path).unwrap();
    path
}

/// An adapted load at the q4 tier is the q4 model: every projection — the adapted ones, and the
/// 170-wide adapted down projection over its zero-padded input — is packed, so its resident bytes
/// are the unadapted q4 model's and sit the dense tier's constant below the contract's pricing of
/// the adapted spec; its image is within the q4 bound of the adapted bf16 image, the residuals
/// reach it (it differs from the unadapted q4 image), and the residual on the padded projection
/// is live (it differs from the same stack without that module).
///
/// Bounds are the backbone gate's q4 bounds (mean |Δ| ≤ 2.5e-1, cos ≥ 0.90). Measured on the MLX
/// CPU stream: adapted q4 vs adapted bf16 mean |Δ| 1.86e-1 (cos 0.919); vs unadapted q4 1.52e-1;
/// vs the stack without the down-projection module 1.49e-1.
///
/// *Mutations that red this:* dropping the adapted branch's load-time quantization (resident above
/// the pricing); keeping the unaligned adapted projection dense, at load or in the pricing (bytes);
/// dropping its adapter stack when the padded base is rebuilt (the image equals the stack without
/// that module); handing the residuals the padded input (a shape error).
#[test]
fn an_adapted_load_holds_the_packed_model_at_a_tier() {
    cpu_process(
        "quant_tiers::an_adapted_load_holds_the_packed_model_at_a_tier",
        || {
            let scratch = tempfile::tempdir().unwrap();
            let tower = aligned_tiny_text_encoder(scratch.path());
            let w2_lora = lora_on_an_unaligned_projection(scratch.path());
            let lora = |path: &Path| {
                vec![mlx_gen::AdapterSpec::new(
                    path.to_path_buf(),
                    1.0,
                    mlx_gen::AdapterKind::Lora,
                )]
            };
            let with_w2 = lora(&w2_lora);
            let without_w2 = lora(&crate::common::fixtures().join("iris_lora.safetensors"));
            let config = tiny_config();
            let te = IrisTextEncoder::load(&tower, &config.text_encoder).unwrap();
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
            let z = noise(7, config.model.in_channels, 16, 16).unwrap();
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
            let load = |adapters: &[mlx_gen::AdapterSpec], quant: Option<Quant>| {
                let (dit, reports) = load_backbone_quantized(
                    &tiny_backbone(),
                    &config,
                    Dtype::Bfloat16,
                    adapters,
                    IrisTask::Generation,
                    MODEL_ID,
                    quant,
                )
                .unwrap();
                for report in &reports {
                    assert!(report.skipped.is_empty(), "{:?}", report.skipped);
                }
                dit
            };
            let priced = |adapters: &[mlx_gen::AdapterSpec], quant: Option<Quant>| {
                let mut spec = spec(&tower, quant);
                spec.adapters = adapters.to_vec();
                memory_strategy_contract(MODEL_ID, &spec)
                    .unwrap()
                    .asset_facts
                    .transformer_bytes
            };

            let dense = backbone(None);
            let slack = priced(&[], None) - dense.resident_weight_bytes();
            let plain_q4 = backbone(Some(Quant::Q4));
            let adapted_bf16 = load(&with_w2, None);
            let adapted_q4 = load(&with_w2, Some(Quant::Q4));
            let resident = adapted_q4.resident_weight_bytes();
            let adapted_priced = priced(&with_w2, Some(Quant::Q4));
            eprintln!(
                "adapted q4: resident {resident} B, priced {adapted_priced} B, slack {slack}"
            );
            assert_eq!(resident, plain_q4.resident_weight_bytes());
            assert_eq!(adapted_priced, priced(&[], Some(Quant::Q4)));
            assert_eq!(adapted_priced - resident, slack);

            let image = render(&adapted_q4);
            let (max_abs, _, mean) = errors(&image, &render(&adapted_bf16));
            let (rel, cos) = rel_and_cos(&image, &render(&adapted_bf16));
            eprintln!(
                "adapted q4 vs adapted bf16 image: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
                 rel={rel:.3e} cos={cos:.5}"
            );
            assert!(mean <= 2.5e-1, "mean |Δ| {mean} > 2.5e-1");
            assert!(cos >= 0.90, "cos {cos} < 0.90");
            let (_, _, from_plain) = errors(&image, &render(&plain_q4));
            let (_, _, from_without_w2) =
                errors(&image, &render(&load(&without_w2, Some(Quant::Q4))));
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
        },
    );
}

/// The text tower at each tier: packed projections whose measured resident bytes equal the
/// contract's conditioning pricing exactly (the embedding counted once though the head is tied),
/// and conditioning within the tier's bound of the dense tower's.
///
/// Bounds (relative L2 over the stacked hidden states the backbone reads, 36 random-weight layers).
/// Measured on the MLX CPU stream: q8 rel 3.3e-2 (cos 0.9995), q4 rel 0.355 (cos 0.939) — about
/// what per-layer affine rounding (~0.6 % at 8 bits, ~10 % at 4 bits of a group's spread)
/// compounds to over the depth. Gated at q8 ≤ 6e-2 / cos ≥ 0.998 and q4 ≤ 0.5 / cos ≥ 0.90.
///
/// *Mutations that red this:* loading dense at a quantized tier; counting the tied head twice;
/// quantizing the embedding (pricing mismatch); a wrong group size (pricing mismatch).
#[test]
fn the_text_tower_holds_packed_projections_at_each_tier() {
    cpu_process(
        "quant_tiers::the_text_tower_holds_packed_projections_at_each_tier",
        || {
            let scratch = tempfile::tempdir().unwrap();
            let tower = aligned_tiny_text_encoder(scratch.path());
            let config = tiny_config();
            let mut states = Vec::new();
            let mut bytes = Vec::new();
            for quant in TIERS {
                let te =
                    IrisTextEncoder::load_quantized(&tower, &config.text_encoder, quant).unwrap();
                assert_eq!(te.is_quantized(), quant.is_some(), "{quant:?}");
                let resident = te.resident_weight_bytes();
                let priced = memory_strategy_contract(MODEL_ID, &spec(&tower, quant))
                    .unwrap()
                    .asset_facts
                    .conditioning_bytes;
                eprintln!("tower {quant:?}: resident {resident} B, priced {priced} B");
                assert_eq!(resident, priced, "{quant:?}");
                bytes.push(resident);
                states.push(te.encode("a red fox in the snow").unwrap().states);
            }
            assert!(bytes[1] < bytes[0] && bytes[2] < bytes[1], "{bytes:?}");
            for (tier, got, bound, min_cos) in [
                ("q8", &states[1], 6e-2_f64, 0.998),
                ("q4", &states[2], 0.5, 0.90),
            ] {
                let (rel, cos) = rel_and_cos(got, &states[0]);
                eprintln!("tower {tier} vs bf16 conditioning: rel={rel:.3e} cos={cos:.5}");
                assert!(rel <= bound, "{tier}: rel {rel} > {bound}");
                assert!(cos >= min_cos, "{tier}: cos {cos} < {min_cos}");
            }

            // The released miniature's MLP width has no affine group: a typed refusal by name.
            match IrisTextEncoder::load_quantized(
                &crate::common::tiny_text_encoder(),
                &config.text_encoder,
                Some(Quant::Q4),
            ) {
                Err(mlx_gen::Error::Unsupported(message)) => {
                    assert!(message.contains("group size"), "{message}")
                }
                other => panic!("expected a typed refusal, got {:?}", other.err()),
            }
        },
    );
}

/// The production path at each tier: the catalog load (`provider_registry().load`) of a quantized
/// spec renders a finite, non-constant image from a fixed seed, under both residency policies,
/// within the tier's bound of the dense render.
///
/// Bounds (mean |Δ| on RGB8, 0–255; same miniature caveat as the backbone gate, now with the tower
/// quantized too). Measured on the MLX CPU stream: q8 3.58, q4 14.6, identical under both
/// policies. Gated at q8 ≤ 6 and q4 ≤ 24.
///
/// *Mutation that reds this:* `load` refusing a quantized spec, or dropping `spec.quantize` on
/// the way to both component loaders (the q4 render would equal the dense one byte for byte —
/// asserted unequal). Which component a tier reaches is gated by the two tests above.
#[test]
fn the_catalog_renders_at_each_tier() {
    cpu_process("quant_tiers::the_catalog_renders_at_each_tier", || {
        let scratch = tempfile::tempdir().unwrap();
        let tower = aligned_tiny_text_encoder(scratch.path());
        let registry = mlx_gen_iris::provider_registry().unwrap();
        let req = GenerationRequest {
            prompt: "a red fox in the snow".into(),
            width: 32,
            height: 16,
            steps: Some(3),
            seed: Some(42),
            ..Default::default()
        };
        for policy in [
            mlx_gen::OffloadPolicy::Resident,
            mlx_gen::OffloadPolicy::Sequential,
        ] {
            let mut renders = Vec::new();
            for quant in TIERS {
                let generator = registry
                    .load(MODEL_ID, &spec(&tower, quant).with_offload_policy(policy))
                    .unwrap();
                let GenerationOutput::Images(images) =
                    generator.generate(&req, &mut |_| {}).unwrap()
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
                eprintln!("catalog {policy:?} {tier} vs bf16: mean |Δ| = {mean:.3} / 255");
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
    });
}
