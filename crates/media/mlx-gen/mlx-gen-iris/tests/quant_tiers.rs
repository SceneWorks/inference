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
/// |Δ| 2.5e-2 (cos 0.9984), q4 mean |Δ| 1.56e-1 (cos 0.944). Gated at q8 ≤ 6e-2 / cos ≥ 0.99 and
/// q4 ≤ 2.5e-1 / cos ≥ 0.90.
///
/// Header-role pricing must equal the actual parameter inventory with each allocation rounded
/// separately. Page slack changes with the tier's allocation sizes/count; it is not one constant
/// difference from the unrounded resident payload sum.
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
                // Round EACH actual parameter allocation, never the aggregate. Dense and
                // packed tiers have different page slack and a different number of allocations.
                let rounded = dit
                    .arrays()
                    .iter()
                    .map(|a| {
                        let bytes = a.nbytes() as u64;
                        if bytes > 16 * 1024 {
                            bytes.div_ceil(16 * 1024) * (16 * 1024)
                        } else {
                            bytes
                        }
                    })
                    .sum::<u64>();
                assert_eq!(
                    priced, rounded,
                    "{quant:?}: header-role price must match actual separately rounded storage"
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
/// 170-wide adapted down projection over its zero-padded input — is packed. Its complete arrays
/// include the residual factors, charged separately through overlay_bytes rather than base bytes.
/// Its image is within the q4 bound of the adapted bf16 image, the residuals
/// reach it (it differs from the unadapted q4 image), and the residual on the padded projection
/// is live (it differs from the same stack without that module).
///
/// Bounds are the backbone gate's q4 bounds (mean |Δ| ≤ 2.5e-1, cos ≥ 0.90). Measured on the MLX
/// CPU stream: adapted q4 vs adapted bf16 mean |Δ| 1.83e-1 (cos 0.923); vs unadapted q4 1.46e-1;
/// vs the stack without the down-projection module 1.46e-1.
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
            };
            let plain_q4 = backbone(Some(Quant::Q4));
            let adapted_bf16 = load(&with_w2, None);
            let adapted_q4 = load(&with_w2, Some(Quant::Q4));
            let resident = adapted_q4.resident_weight_bytes();
            let adapted_facts = priced(&with_w2, Some(Quant::Q4));
            let adapted_priced = adapted_facts.transformer_bytes + adapted_facts.overlay_bytes;
            let plain_facts = priced(&[], Some(Quant::Q4));
            let base_resident = plain_q4.resident_weight_bytes();
            assert!(
                resident > base_resident,
                "arrays must retain the adapter payloads"
            );
            assert_eq!(
                adapted_facts.transformer_bytes, plain_facts.transformer_bytes,
                "adapter factors are not base transformer bytes"
            );
            assert_eq!(plain_facts.overlay_bytes, 0);
            assert!(adapted_facts.overlay_bytes > 0);
            assert!(
                resident - base_resident <= adapted_facts.overlay_bytes,
                "the overlay must cover actual retained adapter storage"
            );
            assert!(resident <= adapted_priced);

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

/// Every packed backbone projection holds its codes on the bf16 grid it stores (sc-26266):
/// its dequantized weight is as close to the bf16 checkpoint weight as the f32-grid optimum —
/// at q8 too, where MLX's `quantize` (codes rounded to the f32 grid, scales then cast to bf16)
/// lands ~1.5 x above it. An adapted load packs the same base: its packed arrays equal the
/// unadapted load's, word for word.
///
/// Bound: error ≤ 1.05 x the f32-grid optimum per projection. Measured on the miniature (MLX CPU
/// stream), worst projection: q8 1.022 x, q4 1.001 x; with MLX's `quantize` q8 is
/// 1.395 x.
///
/// *Mutations that red this:* packing with MLX's `quantize` (`AdaptableLinear::quantize`) in
/// either loader branch — the bare one (q8 ratio), the adapted one (arrays differ from the
/// unadapted load's).
#[test]
fn packed_projections_sit_on_their_stored_grid() {
    cpu_process(
        "quant_tiers::packed_projections_sit_on_their_stored_grid",
        || {
            use mlx_gen_iris::nn::{BackboneQuant, Loader};
            let weights =
                mlx_gen::weights::Weights::from_file(tiny_backbone().join("model.safetensors"))
                    .unwrap();
            let mut keys: Vec<String> = weights
                .keys()
                .filter(|k| k.ends_with(".weight"))
                .map(str::to_owned)
                .collect();
            keys.sort();
            for bits in [8, 4] {
                let loader = Loader {
                    weights: &weights,
                    compute: Dtype::Bfloat16,
                    adapted: None,
                    quant: Some(BackboneQuant { bits }),
                };
                let (mut worst, mut worst_mlx, mut packed) = (0f64, 0f64, 0);
                for key in &keys {
                    let w = weights.get(key).unwrap();
                    if w.ndim() != 2 || w.shape()[1] < 64 {
                        continue;
                    }
                    let prefix = key.trim_end_matches(".weight");
                    let bias = weights.get(&format!("{prefix}.bias")).is_some();
                    let linear = loader.linear(prefix, bias).unwrap();
                    assert!(linear.is_quantized(), "{prefix} is not packed");
                    let arrays = linear.arrays();
                    let bf16 = w.as_dtype(Dtype::Bfloat16).unwrap();
                    let err =
                        crate::common::packing_error(&bf16, arrays[0], arrays[1], arrays[2], bits);
                    let ratio = err / crate::common::f32_grid_error(&bf16, bits);
                    // MLX's own packing of the same padded bf16 weight, for the record.
                    let mut mlx = mlx_gen::adapters::AdaptableLinear::dense(
                        mlx_rs::ops::pad(
                            &bf16,
                            &[
                                (0, 0),
                                (
                                    0,
                                    (w.shape()[1] as usize).div_ceil(64) as i32 * 64 - w.shape()[1],
                                ),
                            ][..],
                            None,
                            None,
                        )
                        .unwrap(),
                        None,
                    );
                    mlx.quantize(bits, Some(64)).unwrap();
                    let (c, s, b, ..) = mlx.quantized_params().unwrap();
                    worst_mlx = worst_mlx.max(
                        crate::common::packing_error(&bf16, c, s, b, bits)
                            / crate::common::f32_grid_error(&bf16, bits),
                    );
                    assert!(
                        ratio <= 1.05,
                        "q{bits} {prefix}: packing error {err:.4e} is {ratio:.3} x the f32-grid optimum"
                    );
                    worst = worst.max(ratio);
                    packed += 1;
                }
                eprintln!(
                    "q{bits}: {packed} packed projections, worst {worst:.4} x the f32-grid optimum \
                     (MLX quantize: {worst_mlx:.4} x)"
                );
                assert!(packed > 0);
            }

            // The adapted loader branch packs the same base (the LoRA stays a residual).
            let scratch = tempfile::tempdir().unwrap();
            let lora = vec![mlx_gen::AdapterSpec::new(
                lora_on_an_unaligned_projection(scratch.path()),
                1.0,
                mlx_gen::AdapterKind::Lora,
            )];
            let load = |adapters: &[mlx_gen::AdapterSpec]| {
                load_backbone_quantized(
                    &tiny_backbone(),
                    &tiny_config(),
                    Dtype::Bfloat16,
                    adapters,
                    IrisTask::Generation,
                    MODEL_ID,
                    Some(Quant::Q8),
                )
                .unwrap()
                .0
            };
            let (plain, adapted) = (load(&[]), load(&lora));
            let (a, b) = (plain.arrays(), adapted.arrays());
            let file = Array::load_safetensors(&lora[0].path).unwrap();
            let factors: Vec<_> = file
                .iter()
                .filter(|(name, _)| {
                    name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight")
                })
                .map(|(_, array)| array)
                .collect();
            assert!(!factors.is_empty());
            assert_eq!(
                b.len(),
                a.len() + factors.len(),
                "every LoRA factor is owned beside the unchanged base"
            );
            let mut next = 0;
            for (i, x) in a.iter().enumerate() {
                let mut found = false;
                while let Some(y) = b.get(next) {
                    next += 1;
                    if x.shape() != y.shape() || x.dtype() != y.dtype() {
                        continue;
                    }
                    let same = x
                        .as_dtype(Dtype::Float32)
                        .unwrap()
                        .eq(y.as_dtype(Dtype::Float32).unwrap())
                        .unwrap()
                        .all(None)
                        .unwrap()
                        .item::<bool>();
                    if same {
                        found = true;
                        break;
                    }
                }
                assert!(
                    found,
                    "base array {i} changed or disappeared from adapted q8 inventory"
                );
            }
            let base_bytes: usize = a.iter().map(|x| x.nbytes()).sum();
            let all_bytes: usize = b.iter().map(|x| x.nbytes()).sum();
            assert_eq!(all_bytes - base_bytes, factors.iter().map(|x| x.nbytes()).sum::<usize>(),
                "F32 fixture factors are retained exactly; alpha scaling introduces no extra retained scalar");
        },
    );
}

/// The text tower's load-time quantization rounds to the stored grid too (sc-26266) — the shared
/// decoder's `QuantGrid::Stored`, the same packing rule as the backbone's.
///
/// *Mutation that reds this:* `text_encoder_quant_spec` leaving the grid at MLX's default.
#[test]
fn the_text_tower_quantizes_on_the_stored_grid() {
    let tower = mlx_llm::ModelConfig::from_dir(crate::common::tiny_text_encoder()).unwrap();
    let mut aligned = tower.clone();
    aligned.intermediate_size = 64;
    for quant in [Quant::Q8, Quant::Q4] {
        let spec = mlx_gen_iris::text_encoder::text_encoder_quant_spec(&aligned, Some(quant))
            .unwrap()
            .unwrap();
        assert_eq!(
            spec.grid,
            mlx_llm::primitives::QuantGrid::Stored,
            "{quant:?}"
        );
    }
}

/// The text tower at each tier: packed projections whose measured resident bytes equal the
/// contract's conditioning pricing exactly (the embedding counted once though the head is tied),
/// and conditioning within the tier's bound of the dense tower's.
///
/// Bounds (relative L2 over the stacked hidden states the backbone reads, 36 random-weight layers).
/// Measured on the MLX CPU stream: q8 rel 3.5e-2 (cos 0.9994), q4 rel 0.358 (cos 0.938) — about
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
/// quantized too). Measured on the MLX CPU stream: q8 2.51, q4 14.0, identical under both
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
