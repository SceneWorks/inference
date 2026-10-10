//! sc-25681: LoRA and LoKr on the Iris backbone vs upstream's own forward with the adapter delta
//! folded into the weights (`tools/dump_iris_controls.py`), on the `iris_dit_golden` inputs. The
//! synthetic adapters target four projections (text adapter, a dual-stream block, a single-stream
//! block, the pixel head); the LoKr's delta is the Kronecker product of a full `w1` and a low-rank
//! `w2_a·w2_b`. MLX installs both as forward-time residuals (LoKr through the structured
//! `w1·X·w2ᵀ`), so the comparison is residual-vs-merged.
//!
//! Tolerance: FP32 backbone on the MLX CPU stream (1e-4 for the bare forward, `dit_parity`); the
//! shared LoKr path reconstructs its factors in bf16 (PARITY-BF16, `mlx_gen::adapters::loader`), so
//! the adapted forwards are held to 2e-3 of peak — two orders below each adapter's own effect on
//! the output (asserted).

use mlx_gen::gen_core::iris::{IrisTask, ADAPTER_TASK_KEY};
use mlx_gen::{AdapterKind, AdapterSpec, Error};
use mlx_gen_iris::{load_backbone_with_adapters, TextBatch};
use mlx_rs::Dtype;

use crate::common::{
    assert_close, errors, fixture, fixtures, host_i32, on_cpu, tiny_backbone, tiny_config,
};

const TOL: f32 = 2e-3;

fn spec(file: &str, kind: AdapterKind, scale: f32) -> AdapterSpec {
    AdapterSpec::new(fixtures().join(file), scale, kind)
}

#[test]
fn lora_and_lokr_match_upstream_merged_weights() {
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

#[test]
fn an_adapter_for_another_task_or_family_is_refused() {
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
