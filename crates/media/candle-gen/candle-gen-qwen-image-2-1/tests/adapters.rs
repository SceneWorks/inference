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
