//! Tensor-free frozen inputs and provenance controls.
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::io::Read;
use std::path::Path;
pub(super) const BASE: &str = "d0775d4922f38e442d38485d412405997f9abc80";
pub(super) const PROTOCOL: &str =
    "09008e5a0ba6bc31dfb5f6f0ec95cb322741a7df3be6a9a09896896d29df709f";
pub(super) const DONOR: &str = "1a8535148c65d266d2b583d88a4c0b056d0e8e55eca9c9fa05378d67b5fde781";
pub(super) const TRAINING: &str =
    "f7abbffe6aa4f1a0117e549524e5cfcc3cde2fc1214d1eec2ed3fe49a2567c74";
pub(super) const DATASET: &str = "44a19887ab4c11293069c5ae2b2a5022ed53eec4bdbbe423a75cc82136b09ba4";
pub(super) const CAPTION: &str = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour; image 2 is only the RGB level palette; do not copy its layout";
pub(super) const SHAPE: [i32; 3] = [1, 2304, 64];
pub(super) const REFERENCE_HASHES: [&str; 2] = [
    "a25c4aa67d3eb7c9107d353acc5813c1e3d95aeeb355fb41f137ffd3edc44a60",
    "da3be3ca711ec3d0e89df8f1e91bc662365fea188fd513d0108782b6c726189f",
];
pub(super) fn validate_reference_order(hashes: &[String]) {
    assert_eq!(
        hashes, REFERENCE_HASHES,
        "two exact ordered native reference PNGs"
    );
}
const FACTOR_PATHS: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
];

pub(super) fn header(path: &Path) -> Value {
    let mut file = std::fs::File::open(path).unwrap();
    let mut n = [0_u8; 8];
    file.read_exact(&mut n).unwrap();
    let n = u64::from_le_bytes(n);
    assert!(n > 0 && n <= 1_048_576);
    let mut bytes = vec![0_u8; n as usize];
    file.read_exact(&mut bytes).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn expected_keys() -> std::collections::BTreeSet<String> {
    (0..32)
        .flat_map(|block| {
            FACTOR_PATHS.into_iter().flat_map(move |path| {
                ["lokr_w1", "lokr_w2_a", "lokr_w2_b"]
                    .map(|suffix| format!("transformer_blocks.{block}.{path}.{suffix}"))
            })
        })
        .collect()
}
pub(super) fn validate_header(h: &Value) {
    let object = h.as_object().unwrap();
    let keys = object
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        keys,
        expected_keys(),
        "exact trained key set, not loose count"
    );
    let m = &h["__metadata__"];
    for (key, value) in [
        ("rank", "16"),
        ("alpha", "16"),
        ("family", "qwen-image-2-1"),
        ("baseModel", "qwen_image_2_1"),
        ("trainingMode", "edit"),
        ("networkType", "lokr"),
        (
            "license",
            "Qwen Research License Agreement (research/evaluation only)",
        ),
    ] {
        assert_eq!(m[key], value, "diagnostic donor metadata {key}");
    }
    for key in keys {
        let row = &h[&key];
        assert_eq!(row["dtype"], "F32");
        let shape = row["shape"].as_array().unwrap();
        assert_eq!(shape.len(), 2);
        assert!(shape.iter().all(|n| n.as_u64().is_some_and(|n| n > 0)));
        if key.ends_with("lokr_w2_a") {
            assert_eq!(shape[1], 16);
        }
        if key.ends_with("lokr_w2_b") {
            assert_eq!(shape[0], 16);
        }
    }
}
pub(super) fn validate_manifest(m: &Value, training: &Value) {
    assert_eq!(m["kind"], "DIAGNOSTIC_ONLY");
    assert_eq!(m["purpose"], "DIAGNOSTIC_ONLY");
    assert_eq!(m["acceptanceEvidence"], false);
    let p = &m["trainingProvenance"];
    assert_eq!(p["sourceCandidate"], BASE);
    assert_eq!(p["runId"], 37_249_426_665_u64);
    assert_eq!(p["steps"], 120);
    assert_eq!(p["datasetSha256"], DATASET);
    let a = m["adapters"].as_array().unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(a[0]["name"], "mlx_failed_balanced_edit_lokr");
    assert_eq!(a[0]["kind"], "lokr");
    assert_eq!(a[0]["file"], "qwen21_edit_lokr_120step_1a853.safetensors");
    assert_eq!(a[0]["sha256"], DONOR);
    assert_eq!(a[0]["size"], 6_759_417);
    assert_eq!(
        m["trainingReceipt"]["file"],
        "qwen21_edit_lokr_120step_1a853_training.json"
    );
    assert_eq!(m["trainingReceipt"]["sha256"], TRAINING);
    assert_eq!(m["trainingReceipt"]["size"], 53_379);
    let t = &training["training"];
    assert_eq!(t["steps"], 120);
    assert_eq!(t["stepsRun"], 120);
    assert_eq!(t["rank"], 16);
    assert_eq!(t["networkType"], "Lokr");
    assert_eq!(t["adapterSha256"], DONOR);
    assert_eq!(t["dataset"]["sha256"], DATASET);
    assert_eq!(t["learningRate"], f64::from(1e-4_f32));
    assert_eq!(t["resolution"], 512);
    assert_eq!(t["gradientCheckpointing"], true);
    assert_eq!(t["editProtocol"]["trainingReferenceCount"], 1);
    assert_eq!(t["editProtocol"]["evaluationReferenceCount"], 2);
    assert_eq!(t["editProtocol"]["evaluationCaption"], CAPTION);
    assert_eq!(
        t["editProtocol"]["trainingDataRecipe"]["heldoutSource99UsedForTraining"],
        false
    );
    let losses = t["losses"].as_array().unwrap();
    assert_eq!(losses.len(), 120);
    assert!(losses
        .iter()
        .all(|v| v.as_f64().is_some_and(|n| n.is_finite())));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_parameter_keys_reject_drop_extra_swap_and_alpha() {
        let mut h = json!({"__metadata__":{"rank":"16","alpha":"16","family":"qwen-image-2-1","baseModel":"qwen_image_2_1",
            "trainingMode":"edit","networkType":"lokr","license":"Qwen Research License Agreement (research/evaluation only)"}});
        for key in expected_keys() {
            let shape = if key.ends_with("lokr_w2_a") {
                vec![64, 16]
            } else if key.ends_with("lokr_w2_b") {
                vec![16, 64]
            } else {
                vec![64, 64]
            };
            h[&key] = json!({"dtype":"F32","shape":shape});
        }
        validate_header(&h);
        for change in 0..5 {
            let mut bad = h.clone();
            let key = expected_keys().into_iter().next().unwrap();
            match change {
                0 => {
                    bad.as_object_mut().unwrap().remove(&key);
                }
                1 => {
                    bad["unexpected.lokr_w1"] = bad[&key].clone();
                }
                2 => bad[&key]["dtype"] = json!("BF16"),
                3 => bad["__metadata__"]["alpha"] = json!("8"),
                _ => {
                    let row = bad.as_object_mut().unwrap().remove(&key).unwrap();
                    bad["wrong_target.lokr_w1"] = row;
                }
            }
            assert!(std::panic::catch_unwind(|| validate_header(&bad)).is_err());
        }
    }

    #[test]
    fn ordered_references_reject_swap_drop_and_changed_source() {
        let correct = REFERENCE_HASHES.map(str::to_owned);
        validate_reference_order(&correct);
        for bad in [
            vec![correct[1].clone(), correct[0].clone()],
            vec![correct[0].clone()],
            vec![correct[0].clone(), "different-key".to_owned()],
        ] {
            assert!(std::panic::catch_unwind(|| validate_reference_order(&bad)).is_err());
        }
    }

    #[test]
    fn donor_provenance_rejects_acceptance_missing_steps_and_changed_recipe() {
        let m = json!({"kind":"DIAGNOSTIC_ONLY","purpose":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,
            "trainingProvenance":{"sourceCandidate":BASE,"runId":37249426665_u64,"steps":120,"datasetSha256":DATASET},
            "adapters":[{"name":"mlx_failed_balanced_edit_lokr","kind":"lokr","file":"qwen21_edit_lokr_120step_1a853.safetensors","sha256":DONOR,"size":6759417}],
            "trainingReceipt":{"file":"qwen21_edit_lokr_120step_1a853_training.json","sha256":TRAINING,"size":53379}});
        let t = json!({"training":{"steps":120,"stepsRun":120,"rank":16,"networkType":"Lokr","adapterSha256":DONOR,
            "dataset":{"sha256":DATASET},"learningRate":f64::from(1e-4_f32),"resolution":512,"gradientCheckpointing":true,
            "editProtocol":{"trainingReferenceCount":1,"evaluationReferenceCount":2,"evaluationCaption":CAPTION,
                "trainingDataRecipe":{"heldoutSource99UsedForTraining":false}},"losses":vec![0.01;120]}});
        validate_manifest(&m, &t);
        for change in 0..8 {
            let mut bad_m = m.clone();
            let mut bad_t = t.clone();
            match change {
                0 => bad_m["acceptanceEvidence"] = json!(true),
                1 => bad_m["trainingProvenance"]["sourceCandidate"] = json!("other-source"),
                2 => bad_t["training"]["stepsRun"] = json!(119),
                3 => {
                    bad_t["training"]["losses"].as_array_mut().unwrap().pop();
                }
                4 => bad_m["adapters"][0]["sha256"] = json!("wrong-donor"),
                5 => bad_t["training"]["editProtocol"]["evaluationReferenceCount"] = json!(1),
                6 => {
                    bad_t["training"]["editProtocol"]["evaluationCaption"] =
                        json!("changed caption")
                }
                _ => {
                    bad_t["training"]["editProtocol"]["trainingDataRecipe"]
                        ["heldoutSource99UsedForTraining"] = json!(true)
                }
            }
            assert!(std::panic::catch_unwind(|| validate_manifest(&bad_m, &bad_t)).is_err());
        }
    }
}
