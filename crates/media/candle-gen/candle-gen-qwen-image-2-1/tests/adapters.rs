//! LoRA / LoKr / LoHa on the Qwen-Image 2.1 DiT (sc-24157), on the committed miniature DiT (CPU).
//!
//! The dense host is `tiny-snapshot/transformer/`; the packed host is the committed q8 tier the MLX
//! converter writes (`tiers/q8/transformer/`), whose `img_mlp.out` projections are MLX-packed (the
//! only widths in the miniature geometry a group-64 tier can pack). Every test drives the DiT
//! forward, so "applied" means "the velocity moved", not "a residual was attached".

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};
use candle_gen::CandleError;
use candle_gen_qwen_image_2_1::adapters::{install, loha_on_packed_tier_refusal, preflight};
use candle_gen_qwen_image_2_1::quant::{Tier, GROUP_SIZE};
use candle_gen_qwen_image_2_1::{load_transformer, QwenImage21Transformer};

use crate::common::{fixtures, tiny_snapshot};

const DIM: usize = 32; // inner_dim of the miniature DiT
const HIDDEN: usize = 64; // inner_dim · mlp_ratio
const ATTN_TARGET: &str = "transformer_blocks.0.attn.to_q";
/// The projection the committed q8 fixture actually packs.
const PACKED_TARGET: &str = "transformer_blocks.0.img_mlp.out";

fn dense_dit() -> QwenImage21Transformer {
    load_transformer(&tiny_snapshot(), &Device::Cpu).expect("tiny DiT loads")
}

/// A transformer-only packed `tier` snapshot: the committed packed `model.safetensors` plus the
/// source config with the converter's `quantization` marker.
fn packed_root(dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    let component = out.join("transformer");
    std::fs::create_dir_all(&component).unwrap();
    std::fs::copy(
        fixtures()
            .join("tiers")
            .join(tier.dir_name())
            .join("transformer/model.safetensors"),
        component.join("model.safetensors"),
    )
    .unwrap();
    let source = tiny_snapshot().join("transformer/config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source).unwrap()).unwrap();
    config["quantization"] = serde_json::json!({
        "bits": tier.transformer_bits().expect("a packed tier"),
        "group_size": GROUP_SIZE,
    });
    std::fs::write(
        component.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    out
}

/// The packed DiT, with the fixture's packed projection asserted packed — the test is about a
/// residual over a quantized base, so that base must really be quantized.
fn packed_dit(dir: &Path, tier: Tier) -> QwenImage21Transformer {
    let mut dit =
        load_transformer(&packed_root(dir, tier), &Device::Cpu).expect("packed DiT loads");
    let mut packed = Vec::new();
    dit.visit_adaptable_mut(&mut |name, linear| {
        if linear.is_packed() {
            packed.push(name.to_string());
        }
        Ok(())
    })
    .unwrap();
    assert!(
        packed.iter().any(|name| name == PACKED_TARGET),
        "{tier:?}: {PACKED_TARGET} must load packed, packed set = {packed:?}"
    );
    dit
}

fn ramp(shape: (usize, usize, usize), seed: usize) -> Tensor {
    let n = shape.0 * shape.1 * shape.2;
    let v: Vec<f32> = (0..n)
        .map(|i| (((i * 31 + seed * 7) % 41) as f32 / 41.0) - 0.5)
        .collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

/// The target velocity for a fixed 3-token prompt and a 2x2 target block.
fn velocity(dit: &QwenImage21Transformer) -> Tensor {
    let latents = ramp((1, 4, 8), 1);
    let text = ramp((1, 3, DIM), 2);
    dit.forward(&latents, &text, 0.5, 2, 2).expect("forward")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn filled(shape: (usize, usize), seed: usize) -> Tensor {
    let n = shape.0 * shape.1;
    let v: Vec<f32> = (0..n)
        .map(|i| (((i * 13 + seed * 5) % 23) as f32 / 23.0) - 0.4)
        .collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn save(path: &Path, tensors: Vec<(String, Tensor)>, meta: Option<HashMap<String, String>>) {
    safetensors::serialize_to_file(tensors, meta, path).unwrap();
}

/// A PEFT LoRA (`lora_A` `[r, in]`, `lora_B` `[out, r]`) over `targets`.
fn write_lora(path: &Path, targets: &[(&str, usize, usize)], seed: usize) {
    let mut tensors = Vec::new();
    for (i, (target, in_dim, out_dim)) in targets.iter().enumerate() {
        tensors.push((
            format!("transformer.{target}.lora_A.weight"),
            filled((2, *in_dim), seed + i),
        ));
        tensors.push((
            format!("transformer.{target}.lora_B.weight"),
            filled((*out_dim, 2), seed + i + 1),
        ));
    }
    save(path, tensors, None);
}

/// A PEFT LoKr over `target`: `kron(w1, w2)` reconstructs `[w1.0·w2.0, w1.1·w2.1] = [out, in]`.
fn write_lokr(path: &Path, target: &str, w1: (usize, usize), w2: (usize, usize)) {
    save(
        path,
        vec![
            (format!("{target}.lokr_w1"), filled(w1, 3)),
            (format!("{target}.lokr_w2"), filled(w2, 4)),
        ],
        Some(HashMap::from([
            ("networkType".to_string(), "lokr".to_string()),
            ("rank".to_string(), "1".to_string()),
            ("alpha".to_string(), "1".to_string()),
        ])),
    );
}

/// A third-party LyCORIS LoHa over `targets` (`hada_w{1,2}_a` `[out, r]`, `_b` `[r, in]`).
fn write_loha(path: &Path, targets: &[(&str, usize, usize)]) {
    let mut tensors = Vec::new();
    for (i, (target, in_dim, out_dim)) in targets.iter().enumerate() {
        tensors.push((format!("{target}.hada_w1_a"), filled((*out_dim, 2), i)));
        tensors.push((format!("{target}.hada_w1_b"), filled((2, *in_dim), i + 1)));
        tensors.push((format!("{target}.hada_w2_a"), filled((*out_dim, 2), i + 2)));
        tensors.push((format!("{target}.hada_w2_b"), filled((2, *in_dim), i + 3)));
        tensors.push((
            format!("{target}.alpha"),
            Tensor::new(2f32, &Device::Cpu).unwrap(),
        ));
    }
    save(path, tensors, None);
}

fn lora(path: &Path, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), scale, AdapterKind::Lora)
}

/// The visitor walks exactly the dotted names the MLX host adapts — the attention and MLP
/// projections of every block plus the embedder / modulation / output projections — once each.
#[test]
fn the_visitor_names_every_projection_once_in_the_mlx_dotted_spelling() {
    let mut dit = dense_dit();
    let mut names = Vec::new();
    dit.visit_adaptable_mut(&mut |name, _| {
        names.push(name.to_string());
        Ok(())
    })
    .unwrap();
    let mut expected: Vec<String> = [
        "img_in",
        "txt_in.in_layer",
        "txt_in.out_layer",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "modulation.1",
    ]
    .map(String::from)
    .to_vec();
    for i in 0..2 {
        for leaf in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.gate_layer",
            "img_mlp.proj",
            "img_mlp.out",
        ] {
            expected.push(format!("transformer_blocks.{i}.{leaf}"));
        }
    }
    expected.push("norm_out.linear".into());
    expected.push("proj_out".into());
    assert_eq!(names, expected);
}

/// A dense LoRA moves the velocity; stacking a second adapter at its own weight moves it again;
/// a PEFT LoKr moves it too.
#[test]
fn lora_and_lokr_install_on_the_dense_dit_and_stack() {
    let temp = tempfile::tempdir().unwrap();
    let base = velocity(&dense_dit());

    let first = temp.path().join("first.safetensors");
    write_lora(&first, &[(ATTN_TARGET, DIM, DIM)], 0);
    let mut one = dense_dit();
    let report = install(&mut one, &[lora(&first, 1.0)], Tier::Bf16, &Device::Cpu).unwrap();
    assert_eq!(report.residuals, 1);
    let single = velocity(&one);
    assert!(
        max_abs_diff(&base, &single) > 1e-4,
        "the LoRA must move the output"
    );

    let second = temp.path().join("second.safetensors");
    write_lora(
        &second,
        &[("transformer_blocks.1.img_mlp.proj", DIM, HIDDEN)],
        5,
    );
    let mut stacked = dense_dit();
    let report = install(
        &mut stacked,
        &[lora(&first, 1.0), lora(&second, 0.5)],
        Tier::Bf16,
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(report.residuals, 2);
    assert!(
        max_abs_diff(&single, &velocity(&stacked)) > 1e-5,
        "the second adapter of the stack must apply"
    );

    // Per-adapter weight: the same file at 0 strength is inert.
    let mut zero = dense_dit();
    install(&mut zero, &[lora(&first, 0.0)], Tier::Bf16, &Device::Cpu).unwrap();
    assert!(max_abs_diff(&base, &velocity(&zero)) < 1e-6);

    let lokr = temp.path().join("lokr.safetensors");
    write_lokr(&lokr, ATTN_TARGET, (2, 2), (DIM / 2, DIM / 2));
    let mut kron = dense_dit();
    let report = install(
        &mut kron,
        &[AdapterSpec::new(lokr, 1.0, AdapterKind::Lokr)],
        Tier::Bf16,
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(report.residuals, 1);
    assert!(max_abs_diff(&base, &velocity(&kron)) > 1e-4);
}

/// On the packed q8 and q4 DiTs a LoRA rides as a residual over the still-packed projection, and a
/// PEFT LoKr does too.
#[test]
fn lora_and_lokr_install_as_residuals_over_packed_projections() {
    for tier in [Tier::Q8, Tier::Q4] {
        let temp = tempfile::tempdir().unwrap();
        let base = velocity(&packed_dit(temp.path(), tier));
        let adapter = temp.path().join("packed.safetensors");
        write_lora(&adapter, &[(PACKED_TARGET, HIDDEN, DIM)], 2);
        let mut dit = packed_dit(temp.path(), tier);
        let report = install(&mut dit, &[lora(&adapter, 1.0)], tier, &Device::Cpu).unwrap();
        assert_eq!(report.residuals, 1);
        let mut checked = false;
        dit.visit_adaptable_mut(&mut |name, linear| {
            if name == PACKED_TARGET {
                assert!(linear.is_packed(), "{tier:?}: the base must stay packed");
                assert!(linear.is_adapted());
                checked = true;
            }
            Ok(())
        })
        .unwrap();
        assert!(checked);
        assert!(
            max_abs_diff(&base, &velocity(&dit)) > 1e-4,
            "{tier:?}: the residual must move the output"
        );

        let lokr = temp.path().join("lokr.safetensors");
        write_lokr(&lokr, PACKED_TARGET, (2, 4), (DIM / 2, HIDDEN / 4));
        let mut kron = packed_dit(temp.path(), tier);
        let report = install(
            &mut kron,
            &[AdapterSpec::new(lokr, 1.0, AdapterKind::Lokr)],
            tier,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(report.residuals, 1);
        assert!(max_abs_diff(&base, &velocity(&kron)) > 1e-4, "{tier:?}");
    }
}

/// A LoHa folds into the dense bf16-tier weights and moves the output.
#[test]
fn loha_folds_into_the_dense_dit() {
    let temp = tempfile::tempdir().unwrap();
    let base = velocity(&dense_dit());
    let adapter = temp.path().join("loha.safetensors");
    write_loha(
        &adapter,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("transformer_blocks.1.img_mlp.out", HIDDEN, DIM),
        ],
    );
    preflight(&tiny_snapshot(), &[lora(&adapter, 1.0)], Tier::Bf16).unwrap();
    let mut dit = dense_dit();
    let report = install(&mut dit, &[lora(&adapter, 0.8)], Tier::Bf16, &Device::Cpu).unwrap();
    assert_eq!(report.loha_folds, 2);
    assert_eq!(report.residuals, 0);
    dit.visit_adaptable_mut(&mut |_, linear| {
        assert!(!linear.is_adapted(), "a fold attaches no residual");
        Ok(())
    })
    .unwrap();
    assert!(max_abs_diff(&base, &velocity(&dit)) > 1e-4);
}

/// A LoHa on a packed tier is a typed `Unsupported` naming LoHa and the tier — at preflight and at
/// install — even when its target happens to be one of the fixture's dense projections.
#[test]
fn loha_on_a_packed_tier_is_a_typed_refusal() {
    let temp = tempfile::tempdir().unwrap();
    let adapter = temp.path().join("loha.safetensors");
    write_loha(&adapter, &[(PACKED_TARGET, HIDDEN, DIM)]);
    let expected = loha_on_packed_tier_refusal(Tier::Q8, &adapter);
    assert!(
        expected.contains("LoHa") && expected.contains("q8"),
        "{expected}"
    );
    for tier in [Tier::Q8, Tier::Q4] {
        match preflight(&tiny_snapshot(), &[lora(&adapter, 1.0)], tier) {
            Err(CandleError::Unsupported(message)) => {
                assert_eq!(message, loha_on_packed_tier_refusal(tier, &adapter));
                assert!(message.contains(tier.dir_name()));
            }
            other => panic!("LoHa on {tier:?} must be a typed refusal, got {other:?}"),
        }
    }
    for tier in [Tier::Q8, Tier::Q4] {
        let mut dit = packed_dit(temp.path(), tier);
        match install(&mut dit, &[lora(&adapter, 1.0)], tier, &Device::Cpu) {
            Err(CandleError::Unsupported(message)) => {
                assert_eq!(message, loha_on_packed_tier_refusal(tier, &adapter))
            }
            other => panic!("install must refuse LoHa on {tier:?}, got {other:?}"),
        }
        dit.visit_adaptable_mut(&mut |_, linear| {
            assert!(!linear.is_adapted(), "a refused LoHa attaches nothing");
            Ok(())
        })
        .unwrap();
    }
}

/// Strict: a key that reaches no DiT projection refuses the whole stack — for LoRA (alongside a
/// valid target in the same file), for a file that matches nothing at all, and for LoHa.
#[test]
fn unmatched_adapter_keys_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let partial = temp.path().join("partial.safetensors");
    write_lora(
        &partial,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("transformer_blocks.9.attn.to_q", DIM, DIM),
        ],
        0,
    );
    let error = install(
        &mut dense_dit(),
        &[lora(&partial, 1.0)],
        Tier::Bf16,
        &Device::Cpu,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("transformer_blocks.9.attn.to_q"),
        "the refusal names the key: {error}"
    );

    let foreign = temp.path().join("foreign.safetensors");
    write_lora(
        &foreign,
        &[("single_transformer_blocks.0.proj_out", DIM, DIM)],
        0,
    );
    assert!(install(
        &mut dense_dit(),
        &[lora(&foreign, 1.0)],
        Tier::Bf16,
        &Device::Cpu
    )
    .is_err());

    // A factor whose shape does not match its projection is not silently skipped either.
    let misshaped = temp.path().join("misshaped.safetensors");
    write_lora(
        &misshaped,
        &[(ATTN_TARGET, DIM, DIM), ("proj_out", DIM, 3)],
        0,
    );
    assert!(install(
        &mut dense_dit(),
        &[lora(&misshaped, 1.0)],
        Tier::Bf16,
        &Device::Cpu
    )
    .is_err());

    let loha = temp.path().join("loha-partial.safetensors");
    write_loha(
        &loha,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("transformer_blocks.7.img_mlp.out", HIDDEN, DIM),
        ],
    );
    let error = install(
        &mut dense_dit(),
        &[lora(&loha, 1.0)],
        Tier::Bf16,
        &Device::Cpu,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("transformer_blocks.7.img_mlp.out"),
        "{error}"
    );
}

/// The weight-free projection table is exactly the visitor walk of a loaded DiT — same keys, same
/// order, same `[out, in]` — so the header-only preflight resolves against what install will see.
#[test]
fn the_weight_free_projection_table_is_the_visitor_walk() {
    let mut dit = dense_dit();
    let mut walked = Vec::new();
    dit.visit_adaptable_mut(&mut |name, linear| {
        walked.push((name.to_string(), linear.base_shape()));
        Ok(())
    })
    .unwrap();
    let cfg = dit.config().clone();
    assert_eq!(QwenImage21Transformer::adaptable_projections(&cfg), walked);
}

fn refusal<T: std::fmt::Debug>(result: candle_gen::Result<T>) -> String {
    result.expect_err("must be refused").to_string()
}

/// A LoHa whose factors have the right element count but the transposed orientation (`[64, 32]`
/// factors over the `[out=32, in=64]` `img_mlp.out`) is refused — at install, by the provider's own
/// orientation check, and at the header-only preflight — never folded as scrambled weights.
#[test]
fn a_transposed_loha_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let adapter = temp.path().join("transposed.safetensors");
    write_loha(
        &adapter,
        &[("transformer_blocks.1.img_mlp.out", DIM, HIDDEN)],
    );
    let mut dit = dense_dit();
    let base = velocity(&dit);
    let error = refusal(install(
        &mut dit,
        &[lora(&adapter, 1.0)],
        Tier::Bf16,
        &Device::Cpu,
    ));
    assert!(
        error.contains("transformer_blocks.1.img_mlp.out")
            && error.contains("are not oriented for the projection's [out=32, in=64]"),
        "{error}"
    );
    assert!(
        max_abs_diff(&base, &velocity(&dit)) < 1e-6,
        "nothing folded"
    );
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&adapter, 1.0)],
        Tier::Bf16,
    ));
    assert!(
        error.contains("transformer_blocks.1.img_mlp.out") && error.contains("not oriented"),
        "{error}"
    );
}

/// Two raw keys that normalize to one module (`transformer.X` beside `X`) are refused rather than
/// one silently winning, and two spellings of one projection (`X` beside `lora_unet_X`) are
/// refused rather than folded twice — at install and at preflight.
#[test]
fn duplicate_spellings_of_one_projection_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let namespaced = temp.path().join("namespaced.safetensors");
    write_loha(
        &namespaced,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("transformer.transformer_blocks.0.attn.to_q", DIM, DIM),
        ],
    );
    let error = refusal(install(
        &mut dense_dit(),
        &[lora(&namespaced, 1.0)],
        Tier::Bf16,
        &Device::Cpu,
    ));
    assert!(error.contains("name the same module"), "{error}");
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&namespaced, 1.0)],
        Tier::Bf16,
    ));
    assert!(error.contains("name the same module"), "{error}");

    let kohya = temp.path().join("kohya.safetensors");
    write_loha(
        &kohya,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("lora_unet_transformer_blocks_0_attn_to_q", DIM, DIM),
        ],
    );
    let mut dit = dense_dit();
    let base = velocity(&dit);
    let error = refusal(install(
        &mut dit,
        &[lora(&kohya, 1.0)],
        Tier::Bf16,
        &Device::Cpu,
    ));
    assert!(
        error.contains("target the same projection `transformer_blocks.0.attn.to_q`"),
        "{error}"
    );
    assert!(
        max_abs_diff(&base, &velocity(&dit)) < 1e-6,
        "nothing folded"
    );
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&kohya, 1.0)],
        Tier::Bf16,
    ));
    assert!(error.contains("target the same projection"), "{error}");

    // The same header-only check covers a LoRA, whose shared install would otherwise keep
    // whichever spelling it read last.
    let lora_dup = temp.path().join("lora-dup.safetensors");
    save(
        &lora_dup,
        vec![
            (format!("{ATTN_TARGET}.lora_A.weight"), filled((2, DIM), 0)),
            (format!("{ATTN_TARGET}.lora_B.weight"), filled((DIM, 2), 1)),
            (
                format!("transformer.{ATTN_TARGET}.lora_A.weight"),
                filled((2, DIM), 2),
            ),
            (
                format!("transformer.{ATTN_TARGET}.lora_B.weight"),
                filled((DIM, 2), 3),
            ),
        ],
        None,
    );
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&lora_dup, 1.0)],
        Tier::Bf16,
    ));
    assert!(error.contains("name the same module"), "{error}");
}

/// The header-only preflight refuses what install would — an unmatched LoRA target, a mis-shaped
/// LoRA factor, a non-factor key — and admits a matching LoRA and LoKr.
#[test]
fn preflight_key_matches_and_shape_checks_from_headers() {
    let temp = tempfile::tempdir().unwrap();
    let good = temp.path().join("good.safetensors");
    write_lora(&good, &[(ATTN_TARGET, DIM, DIM)], 0);
    preflight(&tiny_snapshot(), &[lora(&good, 1.0)], Tier::Bf16).unwrap();
    let lokr = temp.path().join("lokr.safetensors");
    write_lokr(&lokr, ATTN_TARGET, (2, 2), (DIM / 2, DIM / 2));
    preflight(
        &tiny_snapshot(),
        &[AdapterSpec::new(lokr, 1.0, AdapterKind::Lokr)],
        Tier::Bf16,
    )
    .unwrap();

    let unmatched = temp.path().join("unmatched.safetensors");
    write_lora(
        &unmatched,
        &[("transformer_blocks.9.attn.to_q", DIM, DIM)],
        0,
    );
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&unmatched, 1.0)],
        Tier::Bf16,
    ));
    assert!(
        error.contains("transformer_blocks.9.attn.to_q") && error.contains("matches no DiT"),
        "{error}"
    );

    let misshaped = temp.path().join("misshaped.safetensors");
    write_lora(&misshaped, &[("proj_out", DIM, 3)], 0);
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&misshaped, 1.0)],
        Tier::Bf16,
    ));
    assert!(
        error.contains("does not reconstruct the projection's"),
        "{error}"
    );

    let stray = temp.path().join("stray.safetensors");
    save(
        &stray,
        vec![
            (format!("{ATTN_TARGET}.lora_A.weight"), filled((2, DIM), 0)),
            (format!("{ATTN_TARGET}.lora_B.weight"), filled((DIM, 2), 1)),
            (format!("{ATTN_TARGET}.lora_B.bias"), filled((1, DIM), 2)),
        ],
        None,
    );
    let error = refusal(preflight(
        &tiny_snapshot(),
        &[lora(&stray, 1.0)],
        Tier::Bf16,
    ));
    assert!(error.contains("is not a LoRA factor"), "{error}");
}

/// `load` admits an adapter stack now (the old blanket refusal is gone) and renders with it; a
/// zero-match adapter fails the load.
#[test]
fn load_wires_adapters_through_the_generator() {
    let temp = tempfile::tempdir().unwrap();
    let adapter = temp.path().join("load.safetensors");
    write_lora(&adapter, &[(ATTN_TARGET, DIM, DIM)], 0);
    let spec =
        LoadSpec::new(WeightsSource::Dir(tiny_snapshot())).with_adapters(vec![lora(&adapter, 1.0)]);
    let generator = candle_gen_qwen_image_2_1::load(&spec).expect("an adapted load succeeds");
    let request = GenerationRequest {
        prompt: "a red fox".into(),
        width: 64,
        height: 64,
        steps: Some(2),
        seed: Some(7),
        ..Default::default()
    };
    generator
        .generate(&request, &mut |_| {})
        .expect("an adapted render runs");

    let missing = temp.path().join("missing.safetensors");
    write_lora(&missing, &[("transformer_blocks.9.attn.to_q", DIM, DIM)], 0);
    let spec =
        LoadSpec::new(WeightsSource::Dir(tiny_snapshot())).with_adapters(vec![lora(&missing, 1.0)]);
    assert!(candle_gen_qwen_image_2_1::load(&spec).is_err());
}

/// Under `Sequential` the DiT — and so the adapter install — is deferred to the first render, so
/// the weight-free preflight is what refuses a zero-match or mis-oriented adapter at `load`, not
/// mid-generate.
#[test]
fn a_sequential_load_refuses_a_bad_adapter_before_any_render() {
    let temp = tempfile::tempdir().unwrap();
    let sequential = |adapter: &Path| {
        LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_adapters(vec![lora(adapter, 1.0)])
    };
    let missing = temp.path().join("missing.safetensors");
    write_lora(&missing, &[("transformer_blocks.9.attn.to_q", DIM, DIM)], 0);
    let error = candle_gen_qwen_image_2_1::load(&sequential(&missing))
        .err()
        .expect("a zero-match adapter fails a Sequential load")
        .to_string();
    assert!(error.contains("transformer_blocks.9.attn.to_q"), "{error}");

    let transposed = temp.path().join("transposed.safetensors");
    write_loha(
        &transposed,
        &[("transformer_blocks.1.img_mlp.out", DIM, HIDDEN)],
    );
    assert!(
        candle_gen_qwen_image_2_1::load(&sequential(&transposed)).is_err(),
        "a mis-oriented LoHa fails a Sequential load"
    );

    let good = temp.path().join("good.safetensors");
    write_lora(&good, &[(ATTN_TARGET, DIM, DIM)], 0);
    candle_gen_qwen_image_2_1::load(&sequential(&good)).expect("a matching adapter loads");
}

// ── sc-24158: every adapter format, on the dense and the packed tiers ───────────────────────────

/// `[out, in]` of every adaptable projection of the tiny DiT, by dotted path.
fn projection_shapes() -> HashMap<String, (usize, usize)> {
    QwenImage21Transformer::adaptable_projections(dense_dit().config())
        .into_iter()
        .collect()
}

/// The projections that carry an attached residual, in visitor order.
fn adapted(dit: &mut QwenImage21Transformer) -> Vec<String> {
    let mut names = Vec::new();
    dit.visit_adaptable_mut(&mut |name, linear| {
        if linear.is_adapted() {
            names.push(name.to_string());
        }
        Ok(())
    })
    .unwrap();
    names
}

fn transformer_ns(path: &str) -> String {
    format!("transformer.{path}")
}

fn peft_wrapper(path: &str) -> String {
    format!("base_model.model.{path}")
}

fn diffusion_model_ns(path: &str) -> String {
    format!("diffusion_model.{path}")
}

fn kohya(path: &str) -> String {
    format!("lora_unet_{}", path.replace('.', "_"))
}

fn lycoris(path: &str) -> String {
    format!("lycoris_{}", path.replace('.', "_"))
}

fn bare(path: &str) -> String {
    path.to_string()
}

/// One LoRA spelling: how a dotted projection path becomes the file's module key, and the factor
/// suffixes the convention writes.
struct LoraSpelling {
    name: &'static str,
    module: fn(&str) -> String,
    down: &'static str,
    up: &'static str,
    /// Whether the convention ships a per-module `.alpha` scalar.
    alpha: bool,
}

const LORA_SPELLINGS: [LoraSpelling; 7] = [
    LoraSpelling {
        name: "diffusers transformer.",
        module: transformer_ns,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "raw PEFT base_model.model.",
        module: peft_wrapper,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "PEFT .default adapter",
        module: transformer_ns,
        down: "lora_A.default.weight",
        up: "lora_B.default.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "ComfyUI diffusion_model. lora_down/up",
        module: diffusion_model_ns,
        down: "lora_down.weight",
        up: "lora_up.weight",
        alpha: true,
    },
    LoraSpelling {
        name: "ai-toolkit diffusion_model. lora_A/B",
        module: diffusion_model_ns,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "kohya lora_unet_",
        module: kohya,
        down: "lora_down.weight",
        up: "lora_up.weight",
        alpha: true,
    },
    LoraSpelling {
        name: "bare (SceneWorks trainer)",
        module: bare,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: true,
    },
];

fn write_spelled_lora(
    path: &Path,
    spelling: &LoraSpelling,
    targets: &[&str],
    shapes: &HashMap<String, (usize, usize)>,
) {
    let mut tensors = Vec::new();
    for (i, target) in targets.iter().enumerate() {
        let (out_f, in_f) = shapes[*target];
        let module = (spelling.module)(target);
        tensors.push((format!("{module}.{}", spelling.down), filled((2, in_f), i)));
        tensors.push((
            format!("{module}.{}", spelling.up),
            filled((out_f, 2), i + 1),
        ));
        if spelling.alpha {
            tensors.push((
                format!("{module}.alpha"),
                Tensor::new(4f32, &Device::Cpu).unwrap(),
            ));
        }
    }
    save(path, tensors, None);
}

/// The globals and the block leaves whose names already carry `_` — the kohya flattening is
/// ambiguous for them unless resolved against the projection table.
const KOHYA_HARD_TARGETS: [&str; 8] = [
    "txt_in.in_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "modulation.1",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.1.img_mlp.gate_layer",
    "transformer_blocks.1.img_mlp.out",
    "norm_out.linear",
    "proj_out",
];

/// Every LoRA spelling the shared loaders accept applies on the dense DiT — over the globals and the
/// underscore-ambiguous leaves — passes the header-only preflight, and lands on exactly the
/// projections it names (never a neighbour a blind `_` → `.` split would pick).
#[test]
fn every_lora_spelling_applies_to_exactly_its_projections_on_the_dense_dit() {
    let temp = tempfile::tempdir().unwrap();
    let shapes = projection_shapes();
    let base = velocity(&dense_dit());
    let order: Vec<String> = QwenImage21Transformer::adaptable_projections(dense_dit().config())
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    let want: Vec<String> = order
        .iter()
        .filter(|path| KOHYA_HARD_TARGETS.contains(&path.as_str()))
        .cloned()
        .collect();
    assert_eq!(want.len(), KOHYA_HARD_TARGETS.len());
    for (i, spelling) in LORA_SPELLINGS.iter().enumerate() {
        let file = temp.path().join(format!("lora-{i}.safetensors"));
        write_spelled_lora(&file, spelling, &KOHYA_HARD_TARGETS, &shapes);
        preflight(&tiny_snapshot(), &[lora(&file, 1.0)], Tier::Bf16)
            .unwrap_or_else(|e| panic!("{}: preflight: {e}", spelling.name));
        let mut dit = dense_dit();
        let report = install(&mut dit, &[lora(&file, 1.0)], Tier::Bf16, &Device::Cpu)
            .unwrap_or_else(|e| panic!("{}: install: {e}", spelling.name));
        assert_eq!(
            report.residuals,
            KOHYA_HARD_TARGETS.len(),
            "{}",
            spelling.name
        );
        assert_eq!(adapted(&mut dit), want, "{}", spelling.name);
        assert!(
            max_abs_diff(&base, &velocity(&dit)) > 1e-4,
            "{}: the LoRA must move the output",
            spelling.name
        );
    }
}

/// The same spellings ride as residuals over the packed q8 / q4 projection, which stays packed.
#[test]
fn every_lora_spelling_applies_over_a_packed_projection() {
    let shapes = projection_shapes();
    for tier in [Tier::Q8, Tier::Q4] {
        let temp = tempfile::tempdir().unwrap();
        let base = velocity(&packed_dit(temp.path(), tier));
        for (i, spelling) in LORA_SPELLINGS.iter().enumerate() {
            let file = temp.path().join(format!("lora-{i}.safetensors"));
            write_spelled_lora(&file, spelling, &[PACKED_TARGET], &shapes);
            preflight(&tiny_snapshot(), &[lora(&file, 1.0)], tier)
                .unwrap_or_else(|e| panic!("{tier:?} {}: preflight: {e}", spelling.name));
            let mut dit = packed_dit(temp.path(), tier);
            let report = install(&mut dit, &[lora(&file, 1.0)], tier, &Device::Cpu)
                .unwrap_or_else(|e| panic!("{tier:?} {}: install: {e}", spelling.name));
            assert_eq!(report.residuals, 1, "{tier:?} {}", spelling.name);
            assert_eq!(
                adapted(&mut dit),
                [PACKED_TARGET],
                "{tier:?} {}",
                spelling.name
            );
            assert!(
                max_abs_diff(&base, &velocity(&dit)) > 1e-4,
                "{tier:?} {}: the residual must move the output",
                spelling.name
            );
        }
    }
}

/// One LoKr module's factors, by suffix.
type LokrModule = Vec<(&'static str, Tensor)>;

/// Full Kronecker factors over `[out, in]`: `w1 [a, c]`, `w2 [out/a, in/c]`.
fn full_lokr(out_f: usize, in_f: usize, (a, c): (usize, usize)) -> LokrModule {
    vec![
        ("lokr_w1", filled((a, c), 3)),
        ("lokr_w2", filled((out_f / a, in_f / c), 4)),
    ]
}

/// Full `w1`, low-rank `w2 = w2_a [out/a, 2] · w2_b [2, in/c]` with a per-module `alpha`.
fn low_rank_lokr(out_f: usize, in_f: usize, (a, c): (usize, usize), alpha: f32) -> LokrModule {
    vec![
        ("lokr_w1", filled((a, c), 3)),
        ("lokr_w2_a", filled((out_f / a, 2), 5)),
        ("lokr_w2_b", filled((2, in_f / c), 6)),
        ("alpha", Tensor::new(alpha, &Device::Cpu).unwrap()),
    ]
}

/// A LyCORIS tucker right factor in the Linear form: `lokr_t2 [2, 2, 1, 1]`, `lokr_w2_a [2, b]`,
/// `lokr_w2_b [2, d]`, rebuilt as `w2_aᵀ · t2 · w2_b`.
fn tucker_lokr(out_f: usize, in_f: usize, (a, c): (usize, usize), alpha: f32) -> LokrModule {
    vec![
        ("lokr_w1", filled((a, c), 3)),
        ("lokr_t2", filled((2, 2), 7).reshape((2, 2, 1, 1)).unwrap()),
        ("lokr_w2_a", filled((2, out_f / a), 8)),
        ("lokr_w2_b", filled((2, in_f / c), 9)),
        ("alpha", Tensor::new(alpha, &Device::Cpu).unwrap()),
    ]
}

fn write_lokr_modules(
    path: &Path,
    modules: &[(String, LokrModule)],
    meta: Option<HashMap<String, String>>,
) {
    let mut tensors = Vec::new();
    for (module, factors) in modules {
        for (factor, tensor) in factors {
            tensors.push((format!("{module}.{factor}"), tensor.clone()));
        }
    }
    save(path, tensors, meta);
}

/// The SceneWorks / PEFT `networkType=lokr` stamp with a global `rank`/`alpha`.
fn lokr_stamp(rank: &str, alpha: &str) -> HashMap<String, String> {
    HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("rank".to_string(), rank.to_string()),
        ("alpha".to_string(), alpha.to_string()),
    ])
}

fn low_rank_alpha_1(out_f: usize, in_f: usize, w1: (usize, usize)) -> LokrModule {
    low_rank_lokr(out_f, in_f, w1, 1.0)
}

fn low_rank_alpha_2(out_f: usize, in_f: usize, w1: (usize, usize)) -> LokrModule {
    low_rank_lokr(out_f, in_f, w1, 2.0)
}

fn tucker_alpha_1(out_f: usize, in_f: usize, w1: (usize, usize)) -> LokrModule {
    tucker_lokr(out_f, in_f, w1, 1.0)
}

/// One third-party LyCORIS LoKr layout: module spelling and the factor set it writes.
struct LycorisLayout {
    name: &'static str,
    module: fn(&str) -> String,
    factors: fn(usize, usize, (usize, usize)) -> LokrModule,
}

const LYCORIS_LOKR_LAYOUTS: [LycorisLayout; 4] = [
    LycorisLayout {
        name: "lycoris-lib lycoris_ flattened, low-rank w2",
        module: lycoris,
        factors: low_rank_alpha_1,
    },
    LycorisLayout {
        name: "kohya lora_unet_ flattened, both factors full",
        module: kohya,
        factors: full_lokr,
    },
    LycorisLayout {
        name: "ai-toolkit diffusion_model. dotted",
        module: diffusion_model_ns,
        factors: low_rank_alpha_2,
    },
    LycorisLayout {
        name: "lycoris_ flattened Linear tucker lokr_t2",
        module: lycoris,
        factors: tucker_alpha_1,
    },
];

/// Every unstamped LyCORIS LoKr layout (no `networkType`, per-module `.alpha`, flattened or
/// dotted keys, a Linear tucker `lokr_t2`) installs as a structured residual on the dense DiT and
/// over the packed q8 / q4 projection — the base staying packed — and passes the preflight, under
/// either declared kind (a third-party file carries no stamp a caller could read the kind from).
#[test]
fn every_lycoris_lokr_layout_applies_on_dense_and_packed() {
    let temp = tempfile::tempdir().unwrap();
    let dense_targets = [
        ATTN_TARGET,
        "txt_in.in_layer",
        "transformer_blocks.1.img_mlp.gate_layer",
        "norm_out.linear",
    ];
    let shapes = projection_shapes();
    let dense_base = velocity(&dense_dit());
    for (i, layout) in LYCORIS_LOKR_LAYOUTS.iter().enumerate() {
        let modules: Vec<(String, LokrModule)> = dense_targets
            .iter()
            .map(|target| {
                let (out_f, in_f) = shapes[*target];
                (
                    (layout.module)(target),
                    (layout.factors)(out_f, in_f, (2, 2)),
                )
            })
            .collect();
        let file = temp.path().join(format!("lycoris-dense-{i}.safetensors"));
        write_lokr_modules(&file, &modules, None);
        for kind in [AdapterKind::Lokr, AdapterKind::Lora] {
            let spec = AdapterSpec::new(file.clone(), 1.0, kind);
            preflight(&tiny_snapshot(), std::slice::from_ref(&spec), Tier::Bf16)
                .unwrap_or_else(|e| panic!("{}: preflight: {e}", layout.name));
            let mut dit = dense_dit();
            let report = install(&mut dit, &[spec], Tier::Bf16, &Device::Cpu)
                .unwrap_or_else(|e| panic!("{} ({kind:?}): install: {e}", layout.name));
            assert_eq!(report.residuals, dense_targets.len(), "{}", layout.name);
            // Exactly the named projections, in visitor order — never a neighbour.
            assert_eq!(
                adapted(&mut dit),
                [
                    "txt_in.in_layer",
                    ATTN_TARGET,
                    "transformer_blocks.1.img_mlp.gate_layer",
                    "norm_out.linear",
                ],
                "{}",
                layout.name
            );
            assert!(
                max_abs_diff(&dense_base, &velocity(&dit)) > 1e-4,
                "{}: the LoKr must move the output",
                layout.name
            );
        }

        for tier in [Tier::Q8, Tier::Q4] {
            let (out_f, in_f) = shapes[PACKED_TARGET];
            let file = temp
                .path()
                .join(format!("lycoris-{}-{i}.safetensors", tier.dir_name()));
            write_lokr_modules(
                &file,
                &[(
                    (layout.module)(PACKED_TARGET),
                    (layout.factors)(out_f, in_f, (2, 4)),
                )],
                None,
            );
            let spec = AdapterSpec::new(file, 1.0, AdapterKind::Lokr);
            preflight(&tiny_snapshot(), std::slice::from_ref(&spec), tier)
                .unwrap_or_else(|e| panic!("{tier:?} {}: preflight: {e}", layout.name));
            let base = velocity(&packed_dit(temp.path(), tier));
            let mut dit = packed_dit(temp.path(), tier);
            let report = install(&mut dit, &[spec], tier, &Device::Cpu)
                .unwrap_or_else(|e| panic!("{tier:?} {}: install: {e}", layout.name));
            assert_eq!(report.residuals, 1, "{tier:?} {}", layout.name);
            let mut packed = false;
            dit.visit_adaptable_mut(&mut |name, linear| {
                if name == PACKED_TARGET {
                    packed = linear.is_packed() && linear.is_adapted();
                }
                Ok(())
            })
            .unwrap();
            assert!(
                packed,
                "{tier:?} {}: residual over a packed base",
                layout.name
            );
            assert!(
                max_abs_diff(&base, &velocity(&dit)) > 1e-4,
                "{tier:?} {}",
                layout.name
            );
        }
    }
}

/// The LyCORIS per-module scale (`alpha / lora_dim`; forced 1 when both factors are full) and the
/// tucker rebuild are exact: each unstamped layout renders the same velocity as the stamped LoKr
/// carrying the equivalent full right factor at scale 1.
#[test]
fn lycoris_lokr_scale_and_tucker_match_the_equivalent_full_factor_lokr() {
    let temp = tempfile::tempdir().unwrap();
    let render =
        |modules: Vec<(String, LokrModule)>, meta: Option<HashMap<String, String>>, name: &str| {
            let file = temp.path().join(format!("{name}.safetensors"));
            write_lokr_modules(&file, &modules, meta);
            let mut dit = dense_dit();
            install(
                &mut dit,
                &[AdapterSpec::new(file, 1.0, AdapterKind::Lokr)],
                Tier::Bf16,
                &Device::Cpu,
            )
            .unwrap();
            velocity(&dit)
        };
    let factor = |module: &LokrModule, name: &str| {
        module
            .iter()
            .find(|(factor, _)| *factor == name)
            .map(|(_, t)| t.clone())
            .unwrap()
    };
    // Low-rank, alpha 1 over rank 2 ⇒ scale 0.5.
    let low = low_rank_lokr(DIM, DIM, (2, 2), 1.0);
    let low_w2 = (factor(&low, "lokr_w2_a")
        .matmul(&factor(&low, "lokr_w2_b"))
        .unwrap()
        * 0.5)
        .unwrap();
    // Both full ⇒ scale 1 whatever the alpha.
    let mut full = full_lokr(DIM, DIM, (2, 2));
    full.push(("alpha", Tensor::new(7f32, &Device::Cpu).unwrap()));
    let full_w2 = factor(&full, "lokr_w2");
    // Tucker, alpha 1 over rank 2 ⇒ scale 0.5 on w2_aᵀ · t2 · w2_b.
    let tucker = tucker_lokr(DIM, DIM, (2, 2), 1.0);
    let core = factor(&tucker, "lokr_t2").reshape((2, 2)).unwrap();
    let tucker_w2 = (factor(&tucker, "lokr_w2_a")
        .t()
        .unwrap()
        .matmul(&core)
        .unwrap()
        .matmul(&factor(&tucker, "lokr_w2_b"))
        .unwrap()
        * 0.5)
        .unwrap();
    let base = velocity(&dense_dit());
    for (name, module, w2) in [
        ("low-rank", low, low_w2),
        ("both-full", full, full_w2),
        ("tucker", tucker, tucker_w2),
    ] {
        let w1 = factor(&module, "lokr_w1");
        let thirdparty = render(
            vec![(lycoris(ATTN_TARGET), module)],
            None,
            &format!("lycoris-{name}"),
        );
        let stamped = render(
            vec![(
                ATTN_TARGET.to_string(),
                vec![("lokr_w1", w1), ("lokr_w2", w2)],
            )],
            Some(lokr_stamp("1", "1")),
            &format!("stamped-{name}"),
        );
        assert!(max_abs_diff(&base, &stamped) > 1e-4, "{name}: not a no-op");
        assert!(
            max_abs_diff(&thirdparty, &stamped) < 1e-5,
            "{name}: the LyCORIS scale / rebuild must equal the full-factor LoKr"
        );
    }
}

/// The projections whose output on a fixed probe differs between `before` and `after`, in visitor
/// order — a fold leaves no residual to look for, so compare what each projection computes.
fn changed_projections(
    before: &mut QwenImage21Transformer,
    after: &mut QwenImage21Transformer,
) -> Vec<String> {
    let probe = |dit: &mut QwenImage21Transformer| {
        let mut out = Vec::new();
        dit.visit_adaptable_mut(&mut |name, linear| {
            let (_, in_f) = linear.base_shape();
            let x = ramp((1, 1, in_f), 3);
            out.push((name.to_string(), linear.forward(&x)?));
            Ok(())
        })
        .unwrap();
        out
    };
    probe(before)
        .into_iter()
        .zip(probe(after))
        .filter(|((_, a), (_, b))| max_abs_diff(a, b) > 0.0)
        .map(|((name, _), _)| name)
        .collect()
}

/// A LoHa folds under its LyCORIS (`lycoris_…`), kohya (`lora_unet_…`), namespaced
/// (`diffusion_model.…`) and raw PEFT (`base_model.model.…`) spellings alike — the resolution the
/// MLX twin uses.
#[test]
fn loha_folds_under_every_key_spelling() {
    let temp = tempfile::tempdir().unwrap();
    let base = velocity(&dense_dit());
    for (i, module) in [
        "lycoris_transformer_blocks_0_attn_to_q",
        "lora_unet_transformer_blocks_0_attn_to_q",
        "diffusion_model.transformer_blocks.0.attn.to_q",
        "base_model.model.transformer_blocks.0.attn.to_q",
    ]
    .into_iter()
    .enumerate()
    {
        let file = temp.path().join(format!("loha-{i}.safetensors"));
        write_loha(&file, &[(module, DIM, DIM)]);
        preflight(&tiny_snapshot(), &[lora(&file, 1.0)], Tier::Bf16)
            .unwrap_or_else(|e| panic!("{module}: preflight: {e}"));
        let mut dit = dense_dit();
        let report = install(&mut dit, &[lora(&file, 1.0)], Tier::Bf16, &Device::Cpu)
            .unwrap_or_else(|e| panic!("{module}: install: {e}"));
        assert_eq!(report.loha_folds, 1, "{module}");
        assert_eq!(
            changed_projections(&mut dense_dit(), &mut dit),
            [ATTN_TARGET],
            "{module}: the fold lands on exactly to_q (txt_in.out_layer, img_mlp.proj, … bare)"
        );
        assert!(max_abs_diff(&base, &velocity(&dit)) > 1e-4, "{module}");
    }
}

/// The `__metadata__` the MLX trainer (sc-24159) stamps: the provenance block plus the
/// `networkType` / `rank` / `alpha` (+ LoKr `decomposeFactor`) reload contract.
fn trainer_metadata(network: &str) -> HashMap<String, String> {
    let mut meta = HashMap::from([
        ("family".to_string(), "qwen-image-2-1".to_string()),
        ("baseModel".to_string(), "qwen_image_2_1".to_string()),
        (
            "ss_base_model_version".to_string(),
            "qwen_image_2_1".to_string(),
        ),
        (
            "license".to_string(),
            "Qwen Research License Agreement".to_string(),
        ),
        ("networkType".to_string(), network.to_string()),
        ("rank".to_string(), "2".to_string()),
        ("alpha".to_string(), "2".to_string()),
    ]);
    if network == "lokr" {
        meta.insert("decomposeFactor".to_string(), "-1".to_string());
    }
    meta
}

/// The trainer's `factorization(dim, -1)`: the most balanced `m ≤ n` with `m · n = dim`.
fn balanced(dim: usize) -> (usize, usize) {
    (1..=dim)
        .filter(|m| dim.is_multiple_of(*m) && *m <= dim / m)
        .map(|m| (m, dim / m))
        .next_back()
        .unwrap()
}

/// What the MLX trainer writes, key layout for key layout: bare dotted keys; LoRA `lora_A` /
/// `lora_B` plus a `[1]` `.alpha`; LoKr `lokr_w1 [out_a, in_a]` plus low-rank
/// `lokr_w2_a [out_b, r]` / `lokr_w2_b [r, in_b]`.
fn write_trainer_adapter(path: &Path, network: &str, targets: &[&str]) {
    let shapes = projection_shapes();
    let mut tensors = Vec::new();
    for (i, target) in targets.iter().enumerate() {
        let (out_f, in_f) = shapes[*target];
        if network == "lora" {
            tensors.push((format!("{target}.lora_A.weight"), filled((2, in_f), i)));
            tensors.push((format!("{target}.lora_B.weight"), filled((out_f, 2), i + 1)));
            tensors.push((
                format!("{target}.alpha"),
                Tensor::new(&[2f32], &Device::Cpu).unwrap(),
            ));
        } else {
            let (out_a, out_b) = balanced(out_f);
            let (in_a, in_b) = balanced(in_f);
            tensors.push((format!("{target}.lokr_w1"), filled((out_a, in_a), i)));
            tensors.push((format!("{target}.lokr_w2_a"), filled((out_b, 2), i + 1)));
            tensors.push((format!("{target}.lokr_w2_b"), filled((2, in_b), i + 2)));
        }
    }
    save(path, tensors, Some(trainer_metadata(network)));
}

/// An adapter in the MLX trainer's output layout loads on candle with no conversion — LoRA and
/// LoKr, on the dense DiT and over the packed q8 / q4 projection.
#[test]
fn an_mlx_trainer_adapter_loads_on_candle_without_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let dense_targets = [
        ATTN_TARGET,
        "transformer_blocks.1.img_mlp.proj",
        PACKED_TARGET,
    ];
    let base = velocity(&dense_dit());
    for (network, kind) in [("lora", AdapterKind::Lora), ("lokr", AdapterKind::Lokr)] {
        let file = temp.path().join(format!("trainer-{network}.safetensors"));
        write_trainer_adapter(&file, network, &dense_targets);
        let spec = AdapterSpec::new(file, 1.0, kind);
        preflight(&tiny_snapshot(), std::slice::from_ref(&spec), Tier::Bf16).unwrap();
        let mut dit = dense_dit();
        let report = install(
            &mut dit,
            std::slice::from_ref(&spec),
            Tier::Bf16,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(report.residuals, dense_targets.len(), "{network}");
        assert!(max_abs_diff(&base, &velocity(&dit)) > 1e-4, "{network}");

        let packed_file = temp
            .path()
            .join(format!("trainer-{network}-packed.safetensors"));
        write_trainer_adapter(&packed_file, network, &[PACKED_TARGET]);
        let spec = AdapterSpec::new(packed_file, 1.0, kind);
        for tier in [Tier::Q8, Tier::Q4] {
            preflight(&tiny_snapshot(), std::slice::from_ref(&spec), tier).unwrap();
            let packed_base = velocity(&packed_dit(temp.path(), tier));
            let mut dit = packed_dit(temp.path(), tier);
            let report =
                install(&mut dit, std::slice::from_ref(&spec), tier, &Device::Cpu).unwrap();
            assert_eq!(report.residuals, 1, "{tier:?} {network}");
            assert!(
                max_abs_diff(&packed_base, &velocity(&dit)) > 1e-4,
                "{tier:?} {network}"
            );
        }
    }
}

/// Files the DiT genuinely cannot serve are typed, actionable refusals — at the header-only
/// preflight and at install — never a silent skip: a spatial (conv) tucker LoKr, a LyCORIS LoKr
/// whose module reaches no projection, a stray non-factor key beside LyCORIS factors, Kronecker
/// factors that do not reconstruct the projection, and a stamped LoKr declared as a LoRA.
#[test]
fn unservable_lokr_files_are_typed_refusals() {
    let temp = tempfile::tempdir().unwrap();
    let refuse = |name: &str,
                  modules: Vec<(String, LokrModule)>,
                  meta: Option<HashMap<String, String>>,
                  kind: AdapterKind,
                  at_preflight: &str,
                  at_install: &str| {
        let file = temp.path().join(format!("{name}.safetensors"));
        write_lokr_modules(&file, &modules, meta);
        let spec = AdapterSpec::new(file, 1.0, kind);
        if !at_preflight.is_empty() {
            let error = refusal(preflight(
                &tiny_snapshot(),
                std::slice::from_ref(&spec),
                Tier::Bf16,
            ));
            assert!(error.contains(at_preflight), "{name}: preflight: {error}");
        }
        let mut dit = dense_dit();
        let error = refusal(install(&mut dit, &[spec], Tier::Bf16, &Device::Cpu));
        assert!(error.contains(at_install), "{name}: install: {error}");
        assert!(adapted(&mut dit).is_empty(), "{name}: nothing attached");
    };

    let mut conv = tucker_lokr(DIM, DIM, (2, 2), 1.0);
    conv[1].1 = filled((2, 18), 7).reshape((2, 2, 3, 3)).unwrap();
    refuse(
        "conv-tucker",
        vec![(lycoris(ATTN_TARGET), conv)],
        None,
        AdapterKind::Lokr,
        "not the Linear form",
        "not the Linear form",
    );

    refuse(
        "unmatched",
        vec![(
            "lycoris_transformer_blocks_9_attn_to_q".to_string(),
            full_lokr(DIM, DIM, (2, 2)),
        )],
        None,
        AdapterKind::Lokr,
        "lycoris_transformer_blocks_9_attn_to_q",
        "lycoris_transformer_blocks_9_attn_to_q",
    );

    let mut stray = full_lokr(DIM, DIM, (2, 2));
    stray.push(("lokr_w1.bias", filled((1, 2), 1)));
    refuse(
        "stray",
        vec![(lycoris(ATTN_TARGET), stray)],
        None,
        AdapterKind::Lokr,
        "is not a LyCORIS LoKr factor",
        "is not a LoKr factor",
    );

    // `w1 [2, 2] ⊗ w2 [4, 4]` reconstructs [8, 8], not to_q's [32, 32]: refused at the
    // weight-free preflight (so a `Sequential` load never admits it) and at install — LyCORIS and
    // stamped alike.
    refuse(
        "misshaped",
        vec![(lycoris(ATTN_TARGET), full_lokr(8, 8, (2, 2)))],
        None,
        AdapterKind::Lokr,
        "does not reconstruct",
        "does not reconstruct",
    );
    refuse(
        "misshaped-stamped",
        vec![(ATTN_TARGET.to_string(), full_lokr(8, 8, (2, 2)))],
        Some(lokr_stamp("1", "1")),
        AdapterKind::Lokr,
        "does not reconstruct",
        "",
    );

    // A present-but-empty alpha is a malformed file, never a silent `alpha = rank`.
    let mut empty_alpha = low_rank_lokr(DIM, DIM, (2, 2), 1.0);
    empty_alpha[3].1 = Tensor::zeros(0, candle_core::DType::F32, &Device::Cpu).unwrap();
    refuse(
        "empty-alpha",
        vec![(lycoris(ATTN_TARGET), empty_alpha)],
        None,
        AdapterKind::Lokr,
        "present but empty",
        "present but empty",
    );

    // Over-specified / ambiguous factor sets.
    let mut w1_twice = low_rank_lokr(DIM, DIM, (2, 2), 1.0);
    w1_twice.push(("lokr_w1_a", filled((2, 1), 1)));
    w1_twice.push(("lokr_w1_b", filled((1, 2), 2)));
    refuse(
        "w1-full-and-low-rank",
        vec![(lycoris(ATTN_TARGET), w1_twice)],
        None,
        AdapterKind::Lokr,
        "carries both a full lokr_w1",
        "carries both a full lokr_w1",
    );
    let mut t2_and_w2 = tucker_lokr(DIM, DIM, (2, 2), 1.0);
    t2_and_w2.push(("lokr_w2", filled((DIM / 2, DIM / 2), 4)));
    refuse(
        "tucker-and-full-w2",
        vec![(lycoris(ATTN_TARGET), t2_and_w2)],
        None,
        AdapterKind::Lokr,
        "carries both a full lokr_w2",
        "carries both a full lokr_w2",
    );
    let mut half_tucker = tucker_lokr(DIM, DIM, (2, 2), 1.0);
    half_tucker.retain(|(factor, _)| *factor != "lokr_w2_b");
    refuse(
        "tucker-without-w2_b",
        vec![(lycoris(ATTN_TARGET), half_tucker)],
        None,
        AdapterKind::Lokr,
        "missing a Kronecker factor",
        "missing a Kronecker factor",
    );

    refuse(
        "declared-lora",
        vec![(ATTN_TARGET.to_string(), full_lokr(DIM, DIM, (2, 2)))],
        Some(lokr_stamp("1", "1")),
        AdapterKind::Lora,
        "declared LoRA",
        "declared LoRA",
    );
}

/// A LoKr prices what the install keeps on device — the two small Kronecker factors in f32 (a
/// low-rank leg materialized to its full `[b, d]`, a tucker leg collapsed to it) plus their
/// compute-dtype prepared copy — not its (much smaller) file bytes: the overlay is at least the f32
/// factors every installed `LokrFactors` retains, for stamped, low-rank LyCORIS and tucker files.
#[test]
fn a_lokr_overlay_prices_the_resident_kronecker_factors() {
    use candle_gen::quant::LokrFactors;
    use candle_gen_qwen_image_2_1::memory_strategy::adapter_overlay;

    let temp = tempfile::tempdir().unwrap();
    let get = |module: &LokrModule, name: &str| {
        module
            .iter()
            .find(|(factor, _)| *factor == name)
            .map(|(_, t)| t.clone())
    };
    let low = low_rank_lokr(DIM, DIM, (2, 2), 1.0);
    let tucker = tucker_lokr(DIM, DIM, (2, 2), 1.0);
    let stamped = low_rank_lokr(DIM, DIM, (2, 2), 1.0)
        .into_iter()
        .filter(|(factor, _)| *factor != "alpha")
        .collect::<LokrModule>();
    for (name, module, meta, resident_w2) in [
        (
            "lycoris-low-rank",
            low.clone(),
            None,
            get(&low, "lokr_w2_a")
                .unwrap()
                .matmul(&get(&low, "lokr_w2_b").unwrap())
                .unwrap(),
        ),
        (
            "lycoris-tucker",
            tucker.clone(),
            None,
            get(&tucker, "lokr_w2_a")
                .unwrap()
                .t()
                .unwrap()
                .matmul(&get(&tucker, "lokr_t2").unwrap().reshape((2, 2)).unwrap())
                .unwrap()
                .matmul(&get(&tucker, "lokr_w2_b").unwrap())
                .unwrap(),
        ),
        (
            "stamped-low-rank",
            stamped.clone(),
            Some(lokr_stamp("2", "2")),
            get(&stamped, "lokr_w2_a")
                .unwrap()
                .matmul(&get(&stamped, "lokr_w2_b").unwrap())
                .unwrap(),
        ),
    ] {
        let key = if meta.is_some() {
            ATTN_TARGET.to_string()
        } else {
            lycoris(ATTN_TARGET)
        };
        let w1 = get(&module, "lokr_w1").unwrap();
        let file = temp.path().join(format!("{name}.safetensors"));
        write_lokr_modules(&file, &[(key, module)], meta);
        let installed = LokrFactors::build(
            1.0,
            (DIM, DIM),
            Some(&w1),
            None,
            None,
            Some(&resident_w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("the factors reconstruct to_q")
        .resident_f32_bytes() as u64;
        let spec = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))
            .with_adapters(vec![AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lokr)]);
        let overlay = adapter_overlay(&spec, &tiny_snapshot(), Tier::Bf16).unwrap();
        let file_bytes = std::fs::metadata(&file).unwrap().len();
        assert!(
            overlay.residual_bytes >= installed,
            "{name}: overlay {} must cover the installed f32 factors {installed}",
            overlay.residual_bytes
        );
        assert!(
            overlay.residual_bytes > file_bytes,
            "{name}: a low-rank LoKr is resident above its file bytes ({file_bytes})"
        );
    }
}

// ── sc-24163: feature-end review — mixed stacks, per-adapter strength, the render seam ─────────

/// The output of the projection at `path` on a fixed probe — linear in the projection's weight and
/// in every residual it carries, so a strength's effect is exactly measurable here (the DiT
/// velocity is not linear in its weights).
fn projection_output(dit: &mut QwenImage21Transformer, path: &str) -> Tensor {
    let mut out = None;
    dit.visit_adaptable_mut(&mut |name, linear| {
        if name == path {
            let (_, in_f) = linear.base_shape();
            out = Some(linear.forward(&ramp((1, 2, in_f), 5))?);
        }
        Ok(())
    })
    .unwrap();
    out.unwrap_or_else(|| panic!("{path} is not an adaptable projection"))
}

/// A third-party (unstamped) LyCORIS LoKr over `target`: full `w1 [2, 2]`, low-rank `w2` with a
/// per-module alpha 1 (LyCORIS scale 0.5).
fn write_lycoris_lokr(path: &Path, target: &str) {
    let (out_f, in_f) = projection_shapes()[target];
    write_lokr_modules(
        path,
        &[(lycoris(target), low_rank_lokr(out_f, in_f, (2, 2), 1.0))],
        None,
    );
}

/// The four adapter routes, one file each, stacked on overlapping projections: a LoRA and a
/// stamped (PEFT) LoKr on `to_v`, a LoHa fold on `img_mlp.out`, and a LyCORIS LoKr beside the LoRA
/// on `to_q`.
struct MixedStack {
    _dir: tempfile::TempDir,
    files: [(PathBuf, AdapterKind); 4],
}

const MIXED_NAMES: [&str; 4] = ["LoRA", "stamped LoKr", "LoHa", "LyCORIS LoKr"];

fn mixed_stack() -> MixedStack {
    let dir = tempfile::tempdir().unwrap();
    let lora_file = dir.path().join("lora.safetensors");
    write_lora(
        &lora_file,
        &[
            (ATTN_TARGET, DIM, DIM),
            ("transformer_blocks.0.attn.to_v", DIM, DIM),
        ],
        0,
    );
    let lokr_file = dir.path().join("lokr.safetensors");
    write_lokr(
        &lokr_file,
        "transformer_blocks.0.attn.to_v",
        (2, 2),
        (DIM / 2, DIM / 2),
    );
    let loha_file = dir.path().join("loha.safetensors");
    write_loha(
        &loha_file,
        &[("transformer_blocks.1.img_mlp.out", HIDDEN, DIM)],
    );
    let lycoris_file = dir.path().join("lycoris.safetensors");
    write_lycoris_lokr(&lycoris_file, ATTN_TARGET);
    MixedStack {
        _dir: dir,
        files: [
            (lora_file, AdapterKind::Lora),
            (lokr_file, AdapterKind::Lokr),
            (loha_file, AdapterKind::Lora),
            (lycoris_file, AdapterKind::Lokr),
        ],
    }
}

impl MixedStack {
    /// The stack at `strengths` (one per file, in order); `None` leaves that file out.
    fn specs(&self, strengths: [Option<f32>; 4]) -> Vec<AdapterSpec> {
        self.files
            .iter()
            .zip(strengths)
            .filter_map(|((path, kind), strength)| {
                strength.map(|s| AdapterSpec::new(path.clone(), s, *kind))
            })
            .collect()
    }

    fn velocity(&self, strengths: [Option<f32>; 4]) -> Tensor {
        let mut dit = dense_dit();
        install(&mut dit, &self.specs(strengths), Tier::Bf16, &Device::Cpu).unwrap();
        velocity(&dit)
    }
}

/// A MIXED stack — LoRA + stamped LoKr on one projection, LoRA + LyCORIS LoKr on another, and a
/// LoHa fold — installs through one `adapters::install` (the candle twin of the MLX
/// `stacked_mixed_adapters_install_with_per_file_strengths`), and each file's strength is its own:
/// every file at strength 0 renders exactly the stack without that file, and moving any one file's
/// strength alone moves the velocity.
///
/// *Mutation that reds this:* any route ignoring its own `AdapterSpec::scale` (e.g. `spec.scale` →
/// `1.0` in `fold_loha` / `install_lycoris_lokr`, or the shared additive install taking the first
/// spec's scale for every file) — that file at 0 then still applies, so it no longer equals the
/// stack without it.
#[test]
fn a_mixed_adapter_stack_installs_with_independent_per_file_strengths() {
    let stack = mixed_stack();
    let full = [Some(0.75), Some(0.5), Some(1.0), Some(1.25)];
    preflight(&tiny_snapshot(), &stack.specs(full), Tier::Bf16).unwrap();
    let mut dit = dense_dit();
    let report = install(&mut dit, &stack.specs(full), Tier::Bf16, &Device::Cpu).unwrap();
    assert_eq!(
        report.residuals,
        2 + 1 + 1,
        "LoRA ×2, stamped LoKr, LyCORIS LoKr"
    );
    assert_eq!(report.loha_folds, 1);
    let stacked = velocity(&dit);
    let base = velocity(&dense_dit());
    assert!(max_abs_diff(&base, &stacked) > 1e-4, "the stack applies");

    for (i, name) in MIXED_NAMES.iter().enumerate() {
        let mut zeroed = full;
        zeroed[i] = Some(0.0);
        let mut without = full;
        without[i] = None;
        let mut moved = full;
        moved[i] = full[i].map(|s| s * 2.0);
        let at_zero = stack.velocity(zeroed);
        assert!(
            max_abs_diff(&at_zero, &stack.velocity(without)) < 1e-6,
            "{name} at strength 0 must equal the stack without it"
        );
        assert!(
            max_abs_diff(&stacked, &at_zero) > 1e-5,
            "{name}'s own strength must matter inside the stack"
        );
        assert!(
            max_abs_diff(&stacked, &stack.velocity(moved)) > 1e-5,
            "{name} at twice its strength must move the stack"
        );
    }
}

/// Per-adapter strength on the two routes this crate installs itself (the LoHa fold and the
/// third-party LyCORIS LoKr residual), measured where it is linear — the adapted projection's own
/// output: strength 0 is bit-identical to the bare base, and strength `s` moves the output by
/// exactly `s ×` the strength-1 delta.
///
/// *Mutation that reds this:* `spec.scale` → `1.0` at the `group.delta(…)` call in `fold_loha`, or
/// at the `group.factors(…)` call in `install_lycoris_lokr`.
#[test]
fn loha_and_lycoris_lokr_strength_scales_the_projection_delta() {
    let temp = tempfile::tempdir().unwrap();
    let loha = temp.path().join("loha.safetensors");
    let loha_target = "transformer_blocks.1.img_mlp.out";
    write_loha(&loha, &[(loha_target, HIDDEN, DIM)]);
    let lycoris_file = temp.path().join("lycoris.safetensors");
    write_lycoris_lokr(&lycoris_file, ATTN_TARGET);
    for (name, file, kind, target) in [
        ("LoHa", &loha, AdapterKind::Lora, loha_target),
        (
            "LyCORIS LoKr",
            &lycoris_file,
            AdapterKind::Lokr,
            ATTN_TARGET,
        ),
    ] {
        let at = |strength: f32| {
            let mut dit = dense_dit();
            install(
                &mut dit,
                &[AdapterSpec::new(file.clone(), strength, kind)],
                Tier::Bf16,
                &Device::Cpu,
            )
            .unwrap();
            projection_output(&mut dit, target)
        };
        let base = projection_output(&mut dense_dit(), target);
        assert_eq!(
            max_abs_diff(&base, &at(0.0)),
            0.0,
            "{name}: strength 0 is bit-identical to the base"
        );
        let unit = (at(1.0) - &base).unwrap();
        let unit_peak = max_abs_diff(&unit, &unit.zeros_like().unwrap());
        assert!(unit_peak > 1e-3, "{name}: the adapter moves its projection");
        for strength in [0.5f32, 2.0, -0.75] {
            let delta = (at(strength) - &base).unwrap();
            let want = (&unit * f64::from(strength)).unwrap();
            assert!(
                max_abs_diff(&delta, &want) <= 1e-4 * unit_peak,
                "{name}: strength {strength} must move the output by {strength}× the unit delta"
            );
        }
    }
}

/// A large-amplitude LoRA (the MLX twin's `render_lora`) over a block attention, a packable FFN
/// projection and the output projection, so its effect survives the VAE decode and u8 quantization
/// of a 2-step miniature render.
fn write_render_lora(path: &Path) {
    let shapes = projection_shapes();
    let mut tensors = Vec::new();
    for (i, target) in ["transformer_blocks.0.attn.to_v", PACKED_TARGET, "proj_out"]
        .iter()
        .enumerate()
    {
        let (out_f, in_f) = shapes[*target];
        tensors.push((
            format!("transformer.{target}.lora_A.weight"),
            (filled((2, in_f), i) * 2.0).unwrap(),
        ));
        tensors.push((
            format!("transformer.{target}.lora_B.weight"),
            (filled((out_f, 2), i + 1) * 2.0).unwrap(),
        ));
    }
    save(path, tensors, None);
}

fn render_pixels(spec: &LoadSpec, request: &GenerationRequest) -> Vec<u8> {
    let generator = candle_gen_qwen_image_2_1::load(spec).expect("the tiny snapshot loads");
    match generator
        .generate(request, &mut |_| {})
        .expect("the render runs")
    {
        candle_gen::gen_core::GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.remove(0).pixels
        }
        other => panic!("images expected, got {other:?}"),
    }
}

fn t2i_request(edge: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".into(),
        width: edge,
        height: edge,
        steps: Some(2),
        seed: Some(42),
        ..Default::default()
    }
}

/// At the generator seam: the adapter changes the T2I render against the bare base; a
/// `Sequential` load — which defers the DiT, and so the install, to the render — renders exactly
/// the `Resident` adapted pixels; and strength 0 is pixel-identical to no adapter.
///
/// *Mutation that reds this:* `load_heavy` skipping `adapters::install` (both adapted renders equal
/// the plain one), or installing only on the `Resident` path (Sequential ≠ Resident).
#[test]
fn adapters_reach_the_render_under_both_residencies() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("render.safetensors");
    write_render_lora(&file);
    let request = t2i_request(32);
    let bare = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
    let plain = render_pixels(&bare, &request);
    let adapted_spec = bare.clone().with_adapters(vec![lora(&file, 1.0)]);
    let adapted = render_pixels(&adapted_spec, &request);
    assert_ne!(adapted, plain, "the LoRA must change the T2I render");
    let sequential = render_pixels(
        &adapted_spec
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential),
        &request,
    );
    assert_eq!(
        sequential, adapted,
        "Sequential must render the same adapted pixels as Resident"
    );
    let off = render_pixels(&bare.with_adapters(vec![lora(&file, 0.0)]), &request);
    assert_eq!(off, plain, "strength 0 is pixel-identical to no adapter");
}

/// The documented boundary — ten ordered references — renders with an adapter installed, and the
/// adapter changes the 10-reference edit render (the reference route denoises through the same
/// adapted DiT).
///
/// *Mutation that reds this:* the reference/edit route denoising through an un-adapted DiT.
#[test]
fn an_adapter_reaches_a_ten_reference_edit_render() {
    use candle_gen::gen_core::{Conditioning, Image};

    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("render.safetensors");
    write_render_lora(&file);
    let edge = 64u32;
    let images: Vec<Image> = (0..10u32)
        .map(|r| Image {
            width: edge,
            height: edge,
            pixels: (0..edge * edge * 3)
                .map(|i| ((i * 37 + r * 53 + 11) % 251) as u8)
                .collect(),
        })
        .collect();
    let request = GenerationRequest {
        prompt: "combine every image".into(),
        conditioning: vec![Conditioning::MultiReference { images }],
        ..t2i_request(edge)
    };
    let bare = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
    let plain = render_pixels(&bare, &request);
    let adapted = render_pixels(&bare.with_adapters(vec![lora(&file, 1.0)]), &request);
    assert_eq!(adapted.len(), (edge * edge * 3) as usize);
    assert_ne!(
        adapted, plain,
        "the LoRA must change the 10-reference render"
    );
}
