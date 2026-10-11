//! sc-25681: LoRA and LoKr on the Iris backbone vs upstream (`tools/dump_iris_controls.py`) — the
//! MLX twin's `adapter_parity` fixtures. Candle installs every adapter as a forward-time residual
//! (sc-25686), so this checks the reconstructed (merged-view) deltas themselves against the oracle's
//! `delta/{lora,lokr}/<path>` (the LoKr's being `kron(w1, w2_a·w2_b)`), then upstream's merged-weight
//! forward on the `iris_dit_golden` inputs.
//!
//! Tolerance: the deltas are f32 reconstructions of the same f32 factors (1e-6 of peak); the
//! forwards are FP32 on Candle CPU (the `dit_parity` 1e-4).

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::IrisTask;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_iris::adapters::merge_adapters;
use candle_gen_iris::nn::safetensors_shapes;
use candle_gen_iris::{load_backbone_with_adapters, TextBatch};

use crate::common::{
    assert_close, cpu, errors, fixture, fixtures, host_i32, tiny_backbone, tiny_config,
};

fn spec(file: &str, kind: AdapterKind, scale: f32) -> AdapterSpec {
    AdapterSpec::new(fixtures().join(file), scale, kind)
}

#[test]
fn lora_and_lokr_deltas_and_forwards_match_upstream() {
    let config = tiny_config();
    let golden = fixture("iris_adapter_golden.safetensors");
    let inputs = fixture("iris_dit_golden.safetensors");
    let targets: Vec<String> = serde_json::from_str(golden.meta("targets")).unwrap();
    let lora = spec(
        "iris_lora.safetensors",
        AdapterKind::Lora,
        golden.meta("lora_scale").parse().unwrap(),
    );
    let lokr = spec(
        "iris_lokr.safetensors",
        AdapterKind::Lokr,
        golden.meta("lokr_scale").parse().unwrap(),
    );
    let shapes = safetensors_shapes(&tiny_backbone().join("model.safetensors")).unwrap();
    for (kind, file) in [("lora", &lora), ("lokr", &lokr)] {
        let merged = merge_adapters(
            &shapes,
            std::slice::from_ref(file),
            IrisTask::Generation,
            "iris_3b",
        )
        .unwrap();
        let deltas = merged.deltas().unwrap();
        assert_eq!(deltas.len(), targets.len(), "{kind}");
        for path in &targets {
            assert_close(
                &format!("delta/{kind}/{path}"),
                &deltas[&format!("{path}.weight")],
                golden.require(&format!("delta/{kind}/{path}")),
                1e-6,
            );
        }
    }
    let flat = host_i32(inputs.require("y_mask"));
    let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(<[i32]>::to_vec).collect();
    let text = TextBatch {
        states: inputs.require("y"),
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
            DType::F32,
            &cpu(),
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
            .forward(inputs.require("x"), inputs.require("t"), &text)
            .unwrap();
        let want = golden.require(&format!("out/{name}"));
        assert_close(&format!("adapter {name}"), &out, want, 1e-4);
        let (effect, _, _) = errors(want, inputs.require("out"));
        assert!(
            effect > 100.0 * 1e-4,
            "{name}: the adapter moves the output only {effect:.3e}"
        );
    }
}

#[test]
fn an_adapter_for_another_task_or_family_is_refused() {
    let config = tiny_config();
    let load = |specs: &[AdapterSpec], task: IrisTask| {
        load_backbone_with_adapters(
            &tiny_backbone(),
            &config,
            DType::F32,
            &cpu(),
            specs,
            task,
            "iris_3b",
        )
        .err()
        .expect("must be refused")
        .to_string()
    };
    let depth = spec("iris_lora_depth_task.safetensors", AdapterKind::Lora, 1.0);
    let err = load(&[depth], IrisTask::Generation);
    assert!(err.contains("depth"), "{err}");
    let generation = spec("iris_lora.safetensors", AdapterKind::Lora, 1.0);
    let err = load(&[generation], IrisTask::Depth);
    assert!(err.contains("generation"), "{err}");
    // An unstamped safetensors (the backbone itself) is not an Iris adapter.
    let unstamped = AdapterSpec::new(
        tiny_backbone().join("model.safetensors"),
        1.0,
        AdapterKind::Lora,
    );
    let err = load(&[unstamped], IrisTask::Generation);
    assert!(err.contains("family"), "{err}");
    // A LoRA file declared as LoKr carries no LoKr factors.
    let mislabeled = spec("iris_lora.safetensors", AdapterKind::Lokr, 1.0);
    let err = load(&[mislabeled], IrisTask::Generation);
    assert!(err.contains("LoKr"), "{err}");
}
