//! Tensor-free immutable binding for the failed native-768 adapter selected by sc-24163.
use serde_json::Value;

pub(super) const SOURCE_BASE: &str = "ce820faecb332263b8c940eb937be71c9c055689";
pub(super) const SCENEWORKS_FBD: &str = "fbd74e46ef77811e42f0eae8266ab078881f17f7";
pub(super) const ADAPTER: &str = "33ab069111f5a085a2870e9d6ccf7a0f6e2357f2ed1bcb67c590a9561878e354";
pub(super) const TRAINING_RECEIPT: &str =
    "2728e1b79f92f5f7101358f72f0a65d2f121480ca8a6282065e98e8e1f14a223";
pub(super) const PROTOCOL_RECEIPT: &str =
    "a3a52a0b47511f6cdbf6fdd350273d3d76ed1d707b12b8fb3297760d05ac9c80";
pub(super) const DATASET: &str = "62d762a977a23d6df0a8435e73ee2abcc026909e3558c6a1f20bfb17e091ef9c";
pub(super) const CAPTION: &str = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour; image 2 is only the RGB level palette; do not copy its layout";
const TRAINING_CAPTION: &str = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour";
pub(super) const SHAPE: [i32; 3] = [1, 2304, 64];
pub(super) const REFERENCE_HASHES: [&str; 2] = [
    "a25c4aa67d3eb7c9107d353acc5813c1e3d95aeeb355fb41f137ffd3edc44a60",
    "da3be3ca711ec3d0e89df8f1e91bc662365fea188fd513d0108782b6c726189f",
];
pub(super) const TARGET_HASH: &str =
    "c2a972a40cb8fd33484ba0651ea50f922195992bdbb3fb1d0ad2a743854584f4";

const FACTOR_PATHS: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
];
const DATASET_FILES: [(&str, u64, &str, &str, u64, &str); 6] = [
    (
        "src_0.png",
        420_499,
        "d73f6c0e96e24cbef10c7ccf96858b382139cef11d1d49d53418e261228fa2f4",
        "tgt_0.png",
        10_528,
        "f2a73d842757ad8be8e8119bc1dda3027b13369ca7ddeff16d0ef8bf87e25a8c",
    ),
    (
        "src_1.png",
        429_645,
        "92fee6ac52bf4a0996bfeb1c65d891c740a2ff5ff3454069ea2084393cf5108b",
        "tgt_1.png",
        10_416,
        "9229dc87003fce7e4021095559d183452339d4e2533920ede9ed345d1e9f4b46",
    ),
    (
        "src_2.png",
        533_273,
        "c34be9c0313c0573e0ace28554a6a29996e4ba084a5c3639dbc8f7cab5e40948",
        "tgt_2.png",
        33_153,
        "b539f9964caf2c2cc763967bf43b4858cae7a31d1df8837c492c5b91e9e76e37",
    ),
    (
        "src_3.png",
        638_662,
        "4e74b9889e08f03ca466252221c330c0868acaf6db727df22a9fd754d246dd80",
        "tgt_3.png",
        29_742,
        "63313c84cb35512d545114997ba325957992f8e8b431f7f155878ad0711f4cc9",
    ),
    (
        "src_4.png",
        532_092,
        "b605d3ff793891410b562609a8fd67da511456a755335755fff9a2d002ce7660",
        "tgt_4.png",
        32_181,
        "53484a1b2a1d773cb994ddda1955546d1a6e01e1ffc9bd9b1c6931f3a6fb01f8",
    ),
    (
        "src_5.png",
        570_926,
        "f344033ff80668458331a240f5bedb5a4243a2690b131ed85646bc69fff59394",
        "tgt_5.png",
        33_393,
        "5073aac32900f92f0199fc870d08f7eb077f8f255617c0a1bbfe57abcd516b9a",
    ),
];

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
    let object = h.as_object().expect("current adapter header object");
    let keys = object
        .keys()
        .filter(|key| *key != "__metadata__")
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(keys, expected_keys(), "exact 672-factor/224-target key set");
    let metadata = &h["__metadata__"];
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
        assert_eq!(metadata[key], value, "current adapter metadata {key}");
    }
    for key in keys {
        let row = &h[&key];
        assert_eq!(row["dtype"], "F32");
        let shape = row["shape"].as_array().expect("factor shape array");
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

pub(super) fn validate_reference_order(hashes: &[String]) {
    assert_eq!(hashes, REFERENCE_HASHES, "exact ordered native references");
}

pub(super) fn validate_manifest(manifest: &Value, training: &Value, protocol: &Value) {
    assert_eq!(manifest["kind"], "DIAGNOSTIC_ONLY");
    assert_eq!(manifest["purpose"], "DIAGNOSTIC_ONLY");
    assert_eq!(manifest["acceptanceEvidence"], false);
    assert_eq!(manifest["sourceBase"], SOURCE_BASE);
    assert_eq!(manifest["sceneWorksFbdCommit"], SCENEWORKS_FBD);
    let provenance = &manifest["trainingProvenance"];
    assert_eq!(provenance["sourceCandidate"], SOURCE_BASE);
    assert_eq!(provenance["runId"], 37_392_084_691_u64);
    assert_eq!(provenance["jobId"], 112_039_296_411_u64);
    assert_eq!(provenance["steps"], 120);
    assert_eq!(provenance["datasetSha256"], DATASET);

    let adapters = manifest["adapters"].as_array().expect("one adapter array");
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0]["name"], "current_failed_native768_edit_lokr");
    assert_eq!(adapters[0]["kind"], "lokr");
    assert_eq!(adapters[0]["file"], "qwen21_edit_lokr.safetensors");
    assert_eq!(adapters[0]["sha256"], ADAPTER);
    assert_eq!(adapters[0]["size"], 6_759_417);
    assert_eq!(manifest["trainingReceipt"]["file"], "edit_lokr.json");
    assert_eq!(manifest["trainingReceipt"]["sha256"], TRAINING_RECEIPT);
    assert_eq!(manifest["trainingReceipt"]["size"], 156_097);
    assert_eq!(
        manifest["protocolReceipt"]["file"],
        "edit-training-protocol.json"
    );
    assert_eq!(manifest["protocolReceipt"]["sha256"], PROTOCOL_RECEIPT);
    assert_eq!(manifest["protocolReceipt"]["size"], 91_960);

    let t = &training["training"];
    assert_eq!(t["steps"], 120);
    assert_eq!(t["stepsRun"], 120);
    assert_eq!(t["rank"], 16);
    assert_eq!(t["networkType"], "Lokr");
    assert_eq!(t["adapterSha256"], ADAPTER);
    assert_eq!(t["dataset"]["sha256"], DATASET);
    assert_eq!(t["learningRate"], f64::from(1e-4_f32));
    assert_eq!(t["resolution"], 768);
    assert_eq!(t["gradientCheckpointing"], true);
    assert_eq!(t["editProtocol"]["trainingReferenceCount"], 1);
    assert_eq!(t["editProtocol"]["evaluationReferenceCount"], 2);
    assert_eq!(t["editProtocol"]["trainingCaption"], TRAINING_CAPTION);
    assert_eq!(t["editProtocol"]["evaluationCaption"], CAPTION);
    assert_eq!(t["editProtocol"]["trainingTargetEdge"], 768);
    assert_eq!(t["editProtocol"]["trainingReferenceNativeEdge"], 768);
    assert_eq!(t["editProtocol"]["evaluationTargetEdge"], 768);
    assert_eq!(t["editProtocol"]["evaluationKeyNativeEdge"], 512);
    assert_eq!(
        t["editProtocol"]["trainingDataRecipe"]["heldoutSource99UsedForTraining"],
        false
    );
    let losses = t["losses"].as_array().expect("120 recorded losses");
    assert_eq!(losses.len(), 120);
    assert!(losses
        .iter()
        .all(|value| value.as_f64().is_some_and(f64::is_finite)));
    let items = t["dataset"]["items"]
        .as_array()
        .expect("six ordered training pairs");
    assert_eq!(items.len(), DATASET_FILES.len());
    for (index, (item, expected)) in items.iter().zip(DATASET_FILES).enumerate() {
        let (source_file, source_bytes, source_sha, target_file, target_bytes, target_sha) =
            expected;
        assert_eq!(item["index"], index);
        assert_eq!(item["caption"], t["editProtocol"]["trainingCaption"]);
        assert_eq!(item["referenceCount"], 1);
        let references = item["orderedReferences"]
            .as_array()
            .expect("ordered one-reference array");
        assert_eq!(references.len(), 1);
        assert_eq!(references[0]["file"], source_file);
        assert_eq!(references[0]["bytes"], source_bytes);
        assert_eq!(references[0]["sha256"], source_sha);
        assert_eq!(item["target"]["file"], target_file);
        assert_eq!(item["target"]["bytes"], target_bytes);
        assert_eq!(item["target"]["sha256"], target_sha);
    }

    assert_eq!(
        protocol["kind"],
        "representative_one_reference_training_two_reference_evaluation"
    );
    assert_eq!(protocol["trainingReferenceCount"], 1);
    assert_eq!(protocol["evaluationReferenceCount"], 2);
    assert_eq!(protocol["trainingCaption"], TRAINING_CAPTION);
    assert_eq!(protocol["evaluationCaption"], CAPTION);
    assert_eq!(protocol["trainingTargetEdge"], 768);
    assert_eq!(protocol["trainingReferenceNativeEdge"], 768);
    assert_eq!(protocol["evaluationTargetEdge"], 768);
    assert_eq!(protocol["evaluationKeyNativeEdge"], 512);
    assert_eq!(protocol["stepsRequested"], 120);
    assert_eq!(protocol["dataset"]["sha256"], DATASET);
    assert_eq!(protocol["dataset"], t["dataset"]);
    assert_eq!(protocol["evaluationKeySha256"], REFERENCE_HASHES[1]);
    assert_eq!(
        protocol["trainingDataRecipe"]["heldoutSource99UsedForTraining"],
        false
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header_fixture() -> Value {
        let mut header = json!({"__metadata__":{
            "rank":"16","alpha":"16","family":"qwen-image-2-1",
            "baseModel":"qwen_image_2_1","trainingMode":"edit","networkType":"lokr",
            "license":"Qwen Research License Agreement (research/evaluation only)"}});
        for key in expected_keys() {
            let shape = if key.ends_with("lokr_w2_a") {
                vec![64, 16]
            } else if key.ends_with("lokr_w2_b") {
                vec![16, 64]
            } else {
                vec![64, 64]
            };
            header[&key] = json!({"dtype":"F32","shape":shape});
        }
        header
    }

    fn fixtures() -> (Value, Value, Value) {
        let losses = vec![json!(0.25); 120];
        let items = DATASET_FILES
            .into_iter()
            .enumerate()
            .map(
                |(
                    index,
                    (source_file, source_bytes, source_sha, target_file, target_bytes, target_sha),
                )| json!({
                    "caption":TRAINING_CAPTION,"index":index,"referenceCount":1,
                    "orderedReferences":[{"file":source_file,"bytes":source_bytes,"sha256":source_sha}],
                    "target":{"file":target_file,"bytes":target_bytes,"sha256":target_sha}
                }),
            )
            .collect::<Vec<_>>();
        let dataset = json!({"sha256":DATASET,"items":items});
        let protocol = json!({
            "kind":"representative_one_reference_training_two_reference_evaluation",
            "trainingReferenceCount":1,"evaluationReferenceCount":2,
            "trainingCaption":TRAINING_CAPTION,"evaluationCaption":CAPTION,"trainingTargetEdge":768,
            "trainingReferenceNativeEdge":768,"evaluationTargetEdge":768,
            "evaluationKeyNativeEdge":512,"stepsRequested":120,
            "dataset":dataset.clone(),"evaluationKeySha256":REFERENCE_HASHES[1],
            "trainingDataRecipe":{"heldoutSource99UsedForTraining":false}
        });
        let training = json!({"training":{
            "steps":120,"stepsRun":120,"rank":16,"networkType":"Lokr",
            "adapterSha256":ADAPTER,"dataset":dataset,
            "learningRate":f64::from(1e-4_f32),"resolution":768,
            "gradientCheckpointing":true,"losses":losses,"editProtocol":{
                "trainingReferenceCount":1,"evaluationReferenceCount":2,
                "trainingCaption":TRAINING_CAPTION,"evaluationCaption":CAPTION,"trainingTargetEdge":768,
                "trainingReferenceNativeEdge":768,"evaluationTargetEdge":768,
                "evaluationKeyNativeEdge":512,
                "trainingDataRecipe":{"heldoutSource99UsedForTraining":false}
            }
        }});
        let manifest = json!({
            "kind":"DIAGNOSTIC_ONLY","purpose":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,
            "sourceBase":SOURCE_BASE,"sceneWorksFbdCommit":SCENEWORKS_FBD,"directory":"/absolute",
            "trainingProvenance":{"sourceCandidate":SOURCE_BASE,"runId":37392084691_u64,
                "jobId":112039296411_u64,"steps":120,"datasetSha256":DATASET},
            "adapters":[{"name":"current_failed_native768_edit_lokr","kind":"lokr",
                "file":"qwen21_edit_lokr.safetensors","sha256":ADAPTER,"size":6759417}],
            "trainingReceipt":{"file":"edit_lokr.json","sha256":TRAINING_RECEIPT,"size":156097},
            "protocolReceipt":{"file":"edit-training-protocol.json","sha256":PROTOCOL_RECEIPT,"size":91960}
        });
        (manifest, training, protocol)
    }

    #[test]
    fn current_binding_accepts_only_exact_header_and_provenance() {
        validate_header(&header_fixture());
        let (manifest, training, protocol) = fixtures();
        validate_manifest(&manifest, &training, &protocol);
        validate_reference_order(&REFERENCE_HASHES.map(str::to_owned));

        for mutation in 0..16 {
            let (mut manifest, mut training, mut protocol) = fixtures();
            match mutation {
                0 => manifest["sourceBase"] = json!("other"),
                1 => manifest["sceneWorksFbdCommit"] = json!("other"),
                2 => manifest["trainingProvenance"]["runId"] = json!(1),
                3 => manifest["trainingProvenance"]["jobId"] = json!(1),
                4 => manifest["adapters"][0]["sha256"] = json!("other"),
                5 => manifest["adapters"][0]["size"] = json!(1),
                6 => manifest["trainingReceipt"]["sha256"] = json!("other"),
                7 => manifest["protocolReceipt"]["sha256"] = json!("other"),
                8 => manifest["trainingProvenance"]["datasetSha256"] = json!("other"),
                9 => training["training"]["resolution"] = json!(512),
                10 => training["training"]["adapterSha256"] = json!("other"),
                11 => training["training"]["losses"] = json!([0.1]),
                12 => protocol["evaluationKeySha256"] = json!("other"),
                13 => protocol["trainingReferenceCount"] = json!(2),
                14 => {
                    training["training"]["dataset"]["items"][0]["orderedReferences"][0]["sha256"] =
                        json!("other")
                }
                _ => manifest["acceptanceEvidence"] = json!(true),
            }
            assert!(std::panic::catch_unwind(|| {
                validate_manifest(&manifest, &training, &protocol)
            })
            .is_err());
        }

        for mutation in 0..4 {
            let mut header = header_fixture();
            let key = expected_keys().into_iter().next().unwrap();
            match mutation {
                0 => {
                    header.as_object_mut().unwrap().remove(&key);
                }
                1 => header["__metadata__"]["alpha"] = json!("8"),
                2 => header[&key]["dtype"] = json!("BF16"),
                _ => header["__metadata__"]["license"] = json!("other"),
            };
            assert!(std::panic::catch_unwind(|| validate_header(&header)).is_err());
        }
    }
}
