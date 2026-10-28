//! sc-25681: LoRA and LoKr on the Iris backbone vs upstream's own forward with the adapter delta
//! folded into the weights (`tools/dump_iris_controls.py`), on the `iris_dit_golden` inputs. The
//! synthetic adapters target four projections (text adapter, a dual-stream block, a single-stream
//! block, the pixel head); the LoKr's delta is the Kronecker product of a full `w1` and a low-rank
//! `w2_a·w2_b`. MLX installs both as forward-time residuals on the dense projection (the LoKr as
//! its reconstructed `[out, in]` delta), so the comparison is residual-vs-merged.
//!
//! Tolerance: the bare forward's 1e-4 of peak (`dit_parity`, FP32 on the MLX CPU stream). Under
//! `Precision::Fp32` the LoKr residual is built in f32 (the host's `lycoris_dtype`, sc-25690);
//! measured max|Δ| / peak: LoRA 1.9e-5, LoKr 2.5e-5, both 2.6e-5 (with the former bf16 LoKr
//! reconstruction: LoKr 6.5e-4, both 6.3e-4 — over this bound).

use mlx_gen::gen_core::iris::{IrisTask, ADAPTER_TASK_KEY};
use mlx_gen::{AdapterKind, AdapterSpec, Error};
use mlx_gen_iris::{load_backbone_with_adapters, TextBatch};
use mlx_rs::Dtype;

use crate::common::{
    assert_close, cpu_process, errors, fixture, fixtures, host_i32, on_cpu, tiny_backbone,
    tiny_config,
};

const TOL: f32 = 1e-4;

fn spec(file: &str, kind: AdapterKind, scale: f32) -> AdapterSpec {
    AdapterSpec::new(fixtures().join(file), scale, kind)
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_and_lokr_match_upstream_merged_weights() {
    cpu_process(
        "adapter_parity::lora_and_lokr_match_upstream_merged_weights",
        || {
            lora_and_lokr_match_upstream_merged_weights_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED adapter_parity::lora_and_lokr_match_upstream_merged_weights");
        },
    );
}

fn lora_and_lokr_match_upstream_merged_weights_body() {
    on_cpu(|| {
        let config = tiny_config();
        let golden = fixture("iris_adapter_golden.safetensors");
        let inputs = fixture("iris_dit_golden.safetensors");
        let targets: Vec<String> =
            serde_json::from_str(golden.metadata("targets").unwrap()).unwrap();
        let lora_scale: f32 = golden.metadata("lora_scale").unwrap().parse().unwrap();
        let lokr_scale: f32 = golden.metadata("lokr_scale").unwrap().parse().unwrap();
        let lora = spec("iris_lora.safetensors", AdapterKind::Lora, lora_scale);
        let lokr = spec("iris_lokr.safetensors", AdapterKind::Lokr, lokr_scale);
        let flat = host_i32(inputs.require("y_mask").unwrap());
        let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(<[i32]>::to_vec).collect();
        let text = TextBatch {
            states: inputs.require("y").unwrap(),
            mask: &mask,
        };
        for (name, specs) in [
            ("lora", vec![lora.clone()]),
            ("lokr", vec![lokr.clone()]),
            ("both", vec![lora.clone(), lokr.clone()]),
        ] {
            let (dit, reports) = load_backbone_with_adapters(
                &tiny_backbone(),
                &config,
                Dtype::Float32,
                &specs,
                IrisTask::Generation,
                "iris_3b",
            )
            .unwrap();
            assert_eq!(reports.len(), specs.len(), "{name}: one report per file");
            for (report, spec) in reports.iter().zip(&specs) {
                assert_eq!(report.adapter_path, spec.path);
                assert_eq!(report.applied, targets.len(), "{name}: every target lands");
                assert!(report.skipped.is_empty());
            }
            let out = dit
                .forward(
                    inputs.require("x").unwrap(),
                    inputs.require("t").unwrap(),
                    &text,
                )
                .unwrap();
            let want = golden.require(&format!("out/{name}")).unwrap();
            assert_close(&format!("adapter {name}"), &out, want, TOL);
            let (effect, _, _) = errors(want, inputs.require("out").unwrap());
            assert!(
                effect > 20.0 * TOL,
                "{name}: the adapter moves the output only {effect:.3e}"
            );
        }
    });
}

/// A backbone whose `config.yaml` declares one pixel block fewer than its checkpoint carries
/// (`pixel_blocks.1.*` unconsumed) — a config/checkpoint mismatch.
fn mismatched_backbone() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config = std::fs::read_to_string(tiny_backbone().join("config.yaml")).unwrap();
    assert_eq!(config.matches("\n    depth: 2\n").count(), 1);
    std::fs::write(
        dir.path().join("config.yaml"),
        config.replace("\n    depth: 2\n", "\n    depth: 1\n"),
    )
    .unwrap();
    std::fs::copy(
        tiny_backbone().join("model.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    dir
}

/// sc-25690 I8: the checkpoint's unused-key check holds with an adapter attached — the adapter
/// install must not mark every projection as consumed.
/// MUTATION: build the install's projection host through tracked `Weights::require` reads — RED.
#[test]
fn a_mismatched_checkpoint_is_refused_with_an_adapter_attached() {
    crate::common::cpu_process(
        "adapter_parity::a_mismatched_checkpoint_is_refused_with_an_adapter_attached",
        a_mismatched_checkpoint_is_refused_with_an_adapter_attached_body,
    );
}

fn a_mismatched_checkpoint_is_refused_with_an_adapter_attached_body() {
    let dir = mismatched_backbone();
    let config = mlx_gen::gen_core::iris::IrisConfig::from_dir(dir.path()).unwrap();
    assert_eq!(config.model.pixel.depth, 1);
    let lora = spec("iris_lora.safetensors", AdapterKind::Lora, 0.75);
    let lokr = spec("iris_lokr.safetensors", AdapterKind::Lokr, 1.0);
    for specs in [vec![], vec![lora], vec![lokr]] {
        let err = load_backbone_with_adapters(
            dir.path(),
            &config,
            Dtype::Float32,
            &specs,
            IrisTask::Generation,
            "iris_3b",
        )
        .err()
        .unwrap_or_else(|| panic!("{} adapter(s): the mismatch must be refused", specs.len()));
        let msg = err.to_string();
        assert!(
            msg.contains("does not consume"),
            "{} adapter(s): {msg}",
            specs.len()
        );
        assert!(msg.contains("pixel_blocks.1."), "{msg}");
    }
}

#[test]
fn an_adapter_for_another_task_or_family_is_refused() {
    cpu_process(
        "adapter_parity::an_adapter_for_another_task_or_family_is_refused",
        an_adapter_for_another_task_or_family_is_refused_body,
    );
}

fn an_adapter_for_another_task_or_family_is_refused_body() {
    let config = tiny_config();
    let load = |file: &str| {
        load_backbone_with_adapters(
            &tiny_backbone(),
            &config,
            Dtype::Float32,
            &[spec(file, AdapterKind::Lora, 1.0)],
            IrisTask::Generation,
            "iris_3b",
        )
        .err()
        .expect("must be refused")
    };
    match load("iris_lora_depth_task.safetensors") {
        Error::Unsupported(m) => assert!(m.contains("depth"), "{m}"),
        other => panic!("expected a typed refusal, got {other:?}"),
    }
    // The generation-stamped LoRA is refused on the depth task's backbone, by the same stamp.
    let err = load_backbone_with_adapters(
        &tiny_backbone(),
        &config,
        Dtype::Float32,
        &[spec("iris_lora.safetensors", AdapterKind::Lora, 1.0)],
        IrisTask::Depth,
        "iris_3b",
    )
    .err()
    .expect("must be refused");
    assert!(err.to_string().contains(ADAPTER_TASK_KEY) || err.to_string().contains("generation"));
    // An unstamped file (the text-encoder fixture is a safetensors without the Iris stamps).
    let err = load_backbone_with_adapters(
        &tiny_backbone(),
        &config,
        Dtype::Float32,
        &[AdapterSpec::new(
            crate::common::tiny_backbone().join("model.safetensors"),
            1.0,
            AdapterKind::Lora,
        )],
        IrisTask::Generation,
        "iris_3b",
    )
    .err()
    .expect("must be refused");
    assert!(err.to_string().contains("family"), "{err}");
}
