use super::*;
use crate::adapters::facts::AdapterFacts;
fn fixture(
    root: &std::path::Path,
    entries: &[(&str, &[usize], f32)],
    metadata: serde_json::Value,
) -> std::path::PathBuf {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), metadata);
    let mut data = Vec::new();
    for (name, shape, value) in entries {
        let start = data.len();
        let count: usize = shape.iter().product();
        for _ in 0..count {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    let path = root.join("tiny.safetensors");
    std::fs::write(&path, bytes).unwrap();
    path
}
fn spec(path: std::path::PathBuf, kind: AdapterKind, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path,
        kind,
        scale,
        pass_scales: None,
        moe_expert: None,
    }
}
fn facts(
    specs: &[AdapterSpec],
    packed: bool,
) -> Result<(serde_json::Value, Vec<(String, AdapterFacts)>)> {
    configured_adapter_facts(specs, &["layer.proj".into()], Dtype::Bfloat16, |path| {
        Ok((path == "layer.proj").then(|| {
            (
                path.into(),
                super::super::LinearFacts {
                    base_shape: vec![4, 4],
                    is_quantized: packed,
                    has_live_adapters: false,
                    adapter_count: 0,
                    bias_shape: None,
                },
            )
        }))
    })
}
#[test]
fn configured_lora_normalizes_aliases_partial_groups_and_continuous_scales() {
    let root = tempfile::tempdir().unwrap();
    let path = fixture(
        root.path(),
        &[
            ("layer.proj.lora_A.weight", &[1, 4], 1.0),
            ("layer.proj.lora_B.weight", &[4, 1], 2.0),
            ("ignored.lora_A.weight", &[1, 4], 3.0),
        ],
        serde_json::json!({}),
    );
    let a = spec(path.clone(), AdapterKind::Lora, 0.5);
    let b = spec(path.clone(), AdapterKind::Lora, 2.0);
    let result = facts(&[a], false);
    assert!(
        result.is_ok(),
        "valid LoRA plus skipped partial group must normalize"
    );
    let (identity, retained) = result.unwrap();
    assert_eq!(identity, facts(&[b], false).unwrap().0);
    assert_eq!(retained.len(), 1);
    assert!(
        matches!(&retained[0].1, AdapterFacts::Lora { a,b,disabled:false } if a.shape == [4,1] && b.shape == [1,4])
    );
    assert!(!identity.to_string().contains(path.to_str().unwrap()));
    let disabled = facts(&[spec(path, AdapterKind::Lora, 0.0)], false).unwrap();
    assert!(disabled.1[0].1.disabled());
    assert_ne!(identity, disabled.0);
    let duplicate = fixture(
        root.path(),
        &[
            ("layer.proj.lora_A.weight", &[1, 4], 1.0),
            ("layer.proj.lora_down.weight", &[1, 4], 1.0),
            ("layer.proj.lora_B.weight", &[4, 1], 1.0),
        ],
        serde_json::json!({}),
    );
    assert!(facts(&[spec(duplicate, AdapterKind::Lora, 1.0)], false).is_err());
}
#[test]
fn configured_packed_lokr_uses_alpha_zero_and_actual_t2_fallback() {
    let root = tempfile::tempdir().unwrap();
    let path = fixture(
        root.path(),
        &[
            ("layer.proj.lokr_w1_a", &[2, 1], 1.0),
            ("layer.proj.lokr_w1_b", &[1, 2], 1.0),
            ("layer.proj.lokr_w2", &[2, 2], 1.0),
            ("layer.proj.alpha", &[3], 0.0),
        ],
        serde_json::json!({}),
    );
    let s = spec(path, AdapterKind::Lokr, 1.0);
    let (_, packed) = facts(std::slice::from_ref(&s), true).unwrap();
    assert!(matches!(
        &packed[0].1,
        AdapterFacts::Structured { disabled: true, .. }
    ));
    let (_, dense) = facts(&[s], false).unwrap();
    assert!(matches!(
        &dense[0].1,
        AdapterFacts::Materialized {
            disabled: false,
            ..
        }
    ));
    let path = fixture(
        root.path(),
        &[
            ("layer.proj.lokr_w1", &[2, 2], 1.0),
            ("layer.proj.lokr_w2", &[2, 2], 1.0),
            ("layer.proj.lokr_t2", &[1, 1, 1, 1], 1.0),
        ],
        serde_json::json!({}),
    );
    let (_, fallback) = facts(&[spec(path, AdapterKind::Lokr, 1.0)], true).unwrap();
    assert!(
        matches!(&fallback[0].1, AdapterFacts::Materialized { disabled:false,delta } if delta.shape == [4,4])
    );
}
#[test]
fn configured_peft_rank_gate_depends_on_actual_install_path() {
    let root = tempfile::tempdir().unwrap();
    let path = fixture(
        root.path(),
        &[
            ("layer.proj.lokr_w1", &[2, 2], 1.0),
            ("layer.proj.lokr_w2", &[2, 2], 1.0),
        ],
        serde_json::json!({"networkType":"lokr","rank":"-1","alpha":"1"}),
    );
    let s = spec(path, AdapterKind::Lokr, 1.0);
    assert!(matches!(
        &facts(std::slice::from_ref(&s), true).unwrap().1[0].1,
        AdapterFacts::Structured {
            disabled: false,
            ..
        }
    ));
    let negative_result = facts(std::slice::from_ref(&s), false);
    assert!(
        negative_result.is_ok(),
        "actual rank<=0 parser normalization must precede the dense gate"
    );
    let negative_dense = negative_result.unwrap();
    assert!(matches!(
        &negative_dense.1[0].1,
        AdapterFacts::Materialized {
            disabled: false,
            ..
        }
    ));
    let default_path = fixture(
        root.path(),
        &[
            ("layer.proj.lokr_w1", &[2, 2], 1.0),
            ("layer.proj.lokr_w2", &[2, 2], 1.0),
        ],
        serde_json::json!({"networkType":"lokr","rank":"1","alpha":"1"}),
    );
    let default_spec = spec(default_path, AdapterKind::Lokr, 1.0);
    assert_eq!(negative_dense.0, facts(&[default_spec], false).unwrap().0);
    let infinite_path = fixture(
        root.path(),
        &[
            ("layer.proj.lokr_w1", &[2, 2], 1.0),
            ("layer.proj.lokr_w2", &[2, 2], 1.0),
        ],
        serde_json::json!({"networkType":"lokr","rank":"inf","alpha":"1"}),
    );
    let infinite = spec(infinite_path, AdapterKind::Lokr, 1.0);
    let infinite_packed = facts(std::slice::from_ref(&infinite), true);
    assert!(
        infinite_packed.is_ok(),
        "finite packed coefficient zero must remain accepted"
    );
    assert!(matches!(
        &infinite_packed.unwrap().1[0].1,
        AdapterFacts::Structured { disabled: true, .. }
    ));
    assert!(facts(&[infinite], false).is_err());
    assert!(facts(
        &[AdapterSpec {
            kind: AdapterKind::Lora,
            ..s
        }],
        true
    )
    .is_err());
}
