use super::*;
use crate::{
    artifacts::NativeDitStorage, quantization::DitFormat, BASE_MODEL_ID, DISTILLED_MODEL_ID,
};
use gen_core::{weightsmeta::Dtype, SafetensorsTensorHeader};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

const REAL_HEADERS: &str = include_str!("../tests/fixtures/flux3-real-checkpoints/headers.json");

pub(crate) fn real_config_json(flow: bool, format: DitFormat) -> Value {
    let text = match (flow, format) {
        (false, DitFormat::Dense) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/base-dense.json")
        }
        (true, DitFormat::Dense) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/distilled-dense.json")
        }
        (false, DitFormat::Fp8Rowwise) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/base-fp8r.json")
        }
        (true, DitFormat::Fp8Rowwise) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/distilled-fp8r.json")
        }
        (false, DitFormat::Nvfp4) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/base-nvfp4.json")
        }
        (true, DitFormat::Nvfp4) => {
            include_str!("../tests/fixtures/flux3-real-checkpoints/distilled-nvfp4.json")
        }
    };
    serde_json::from_str(text).unwrap()
}

fn rows(value: &Value) -> Vec<SafetensorsTensorHeader> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let dtype = Dtype::parse(row[1].as_str().unwrap()).unwrap();
            let shape: Vec<usize> = row[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            let data_bytes = shape.iter().fold(dtype.size() as u64, |n, d| {
                n.checked_mul(*d as u64).unwrap()
            });
            SafetensorsTensorHeader {
                name: row[0].as_str().unwrap().into(),
                dtype,
                shape,
                data_bytes,
            }
        })
        .collect()
}

pub(crate) fn real_headers(flow: bool, format: DitFormat) -> Vec<SafetensorsTensorHeader> {
    let source: Value = serde_json::from_str(REAL_HEADERS).unwrap();
    let key = match format {
        DitFormat::Dense => "dense",
        DitFormat::Fp8Rowwise => "fp8r",
        DitFormat::Nvfp4 => "nvfp4",
    };
    let mut headers = rows(&source["base"][key]);
    if flow {
        headers.extend(rows(&source["distilled_extra"][key]));
    }
    headers
}

/// Real names/dtypes/ranks, tiny all-ones shapes and an ACTUAL small zero payload. These files
/// are legal metadata stubs, never claimed to be geometry-valid or loadable model weights.
pub(crate) fn stub_bytes(headers: &[SafetensorsTensorHeader]) -> Vec<u8> {
    let mut map = serde_json::Map::new();
    let mut offset = 0;
    for header in headers {
        let size = header.dtype.size();
        let shape = vec![1; header.shape.len()];
        map.insert(header.name.clone(), json!({"dtype":header.dtype.as_str(),"shape":shape,"data_offsets":[offset,offset+size]}));
        offset += size;
    }
    map.insert(
        "__metadata__".into(),
        json!({"format":"pt","model":"f3-image-small","dtype":"bfloat16"}),
    );
    let mut json = serde_json::to_vec(&map).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut output = (json.len() as u64).to_le_bytes().to_vec();
    output.extend(json);
    output.resize(output.len() + offset, 0);
    output
}

pub(crate) fn dense_stub_bytes(flow: bool) -> Vec<u8> {
    stub_bytes(&real_headers(flow, DitFormat::Dense))
}

fn package(config: &Value, flow: bool, format: DitFormat) -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(
        temp.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("model.safetensors"),
        stub_bytes(&real_headers(flow, format)),
    )
    .unwrap();
    temp
}

#[test]
fn real_configs_and_independent_headers_cover_all_six_native_inventories() {
    for (format, base_count, distilled_count) in [
        (DitFormat::Dense, 357, 359),
        (DitFormat::Fp8Rowwise, 591, 593),
        (DitFormat::Nvfp4, 951, 953),
    ] {
        for flow in [false, true] {
            let raw = real_config_json(flow, format);
            let metadata = Flux3MetadataConfig::parse(&raw.to_string()).unwrap();
            assert_eq!(metadata.dit.flow_map, if flow { Some(true) } else { None });
            assert_eq!(metadata.model, "flux-3-image-klein");
            let headers = real_headers(flow, format);
            assert_eq!(
                headers.len(),
                if flow { distilled_count } else { base_count }
            );
            let temp = package(&raw, flow, format);
            let config = resolve_checkpoint(temp.path()).unwrap();
            assert_eq!(config.dit.flow_map, flow);
            config
                .validate_route(if flow {
                    DISTILLED_MODEL_ID
                } else {
                    BASE_MODEL_ID
                })
                .unwrap();
            assert!(config
                .validate_route(if flow {
                    BASE_MODEL_ID
                } else {
                    DISTILLED_MODEL_ID
                })
                .is_err());
            // These are independently captured REAL shapes, not the tiny stub's shapes and
            // not a fixture manufactured by expected_tensors itself.
            NativeDitStorage::Publisher(format)
                .validate_headers(&config.dit, &headers)
                .unwrap();
            assert_eq!(config.sampler.steps, if flow { 16 } else { 50 });
            assert_eq!(config.sampler.guidance, if flow { None } else { Some(3.5) });
        }
    }
}

#[test]
fn metadata_is_config_only_and_absent_flag_never_claims_base_identity() {
    let temp = tempfile::tempdir().unwrap();
    let config = real_config_json(false, DitFormat::Dense);
    std::fs::write(temp.path().join("config.json"), config.to_string()).unwrap();
    let metadata = Flux3MetadataConfig::from_dir(temp.path()).unwrap();
    assert_eq!(metadata.dit.flow_map, None);
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
    let plan = metadata.for_model(BASE_MODEL_ID).unwrap();
    assert!(!plan.dit.flow_map);
    assert_eq!(
        metadata.dit.flow_map, None,
        "planning never mutates descriptive metadata"
    );
    assert!(
        matches!(resolve_checkpoint(temp.path()),Err(Error::Unsupported(message)) if message.contains("missing transformer"))
    );
    assert!(metadata.for_model("foreign").is_err());
}

#[test]
fn optional_flag_types_and_header_agreement_fail_closed() {
    for wrong in [json!(null), json!("false"), json!(0), json!({})] {
        let mut config = real_config_json(false, DitFormat::Dense);
        config["dit"]["flow_map"] = wrong;
        assert!(Flux3MetadataConfig::parse(&config.to_string()).is_err());
    }
    for flow in [false, true] {
        let mut config = real_config_json(flow, DitFormat::Dense);
        config["dit"]["flow_map"] = json!(!flow);
        let temp = package(&config, flow, DitFormat::Dense);
        assert!(
            matches!(resolve_checkpoint(temp.path()),Err(Error::Unsupported(message)) if message.contains("flow_map"))
        );
        config["dit"]["flow_map"] = json!(flow);
        std::fs::write(temp.path().join("config.json"), config.to_string()).unwrap();
        assert_eq!(resolve_checkpoint(temp.path()).unwrap().dit.flow_map, flow);
    }
    // The same descriptive base label cannot override a real distilled header. The explicit
    // base sampler also contradicts that header and must be refused, not silently rewritten.
    let temp = package(
        &real_config_json(false, DitFormat::Dense),
        true,
        DitFormat::Dense,
    );
    assert!(resolve_checkpoint(temp.path()).is_err());
}

#[test]
fn missing_sampler_defaults_follow_the_header_not_an_absent_flag() {
    for flow in [false, true] {
        let mut config = real_config_json(flow, DitFormat::Dense);
        config["dit"].as_object_mut().unwrap().remove("flow_map");
        config.as_object_mut().unwrap().remove("sampler");
        let temp = package(&config, flow, DitFormat::Dense);
        let resolved = resolve_checkpoint(temp.path()).unwrap();
        assert_eq!(resolved.dit.flow_map, flow);
        assert_eq!(resolved.sampler.steps, if flow { 16 } else { 50 });
        assert_eq!(
            resolved.sampler.guidance,
            if flow { None } else { Some(3.5) }
        );
        config["sampler"] = json!({"steps":50,"guidance":3.5});
        std::fs::write(temp.path().join("config.json"), config.to_string()).unwrap();
        assert_eq!(resolve_checkpoint(temp.path()).is_ok(), !flow);
    }
}

#[test]
fn missing_truncated_oversized_and_malformed_headers_never_default_to_base() {
    let config = real_config_json(false, DitFormat::Dense);
    let temp = package(&config, false, DitFormat::Dense);
    let path = temp.path().join("model.safetensors");
    let good = dense_stub_bytes(false);
    std::fs::write(&path, &good[..good.len() - 1]).unwrap();
    assert!(
        resolve_checkpoint(temp.path()).is_err(),
        "past-EOF payload offsets are never metadata leniency"
    );
    // A structurally valid oversized header, not just a truncated prefix: without the FLUX3
    // cap the generic 100 MB reader would accept this padded JSON and its exact tiny payload.
    let old_length = u64::from_le_bytes(good[..8].try_into().unwrap()) as usize;
    let oversized_length = MAX_HEADER_BYTES as usize + 8;
    let mut oversized = (oversized_length as u64).to_le_bytes().to_vec();
    oversized.extend_from_slice(&good[8..8 + old_length]);
    oversized.resize(8 + oversized_length, b' ');
    oversized.extend_from_slice(&good[8 + old_length..]);
    std::fs::write(&path, oversized).unwrap();
    assert!(
        matches!(resolve_checkpoint(temp.path()),Err(Error::Unsupported(message)) if message.contains("16 MiB"))
    );
    for bad in [b"{}".as_slice(), b"not json".as_slice()] {
        let mut bytes = (bad.len() as u64).to_le_bytes().to_vec();
        bytes.extend(bad);
        std::fs::write(&path, bytes).unwrap();
        assert!(resolve_checkpoint(temp.path()).is_err());
    }
    std::fs::remove_file(path).unwrap();
    assert!(matches!(
        resolve_checkpoint(temp.path()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn header_schema_names_dtype_shape_and_ref_time_pair_are_validated() {
    let temp = package(
        &real_config_json(true, DitFormat::Dense),
        true,
        DitFormat::Dense,
    );
    let path = temp.path().join("model.safetensors");
    let mut headers = real_headers(true, DitFormat::Dense);
    headers.retain(|h| h.name != FLOW_MAP_OUT_KEY);
    std::fs::write(&path, stub_bytes(&headers)).unwrap();
    assert!(resolve_checkpoint(temp.path()).is_err());
    for key in ["dtype", "shape", "data_offsets"] {
        let good = dense_stub_bytes(true);
        let n = u64::from_le_bytes(good[..8].try_into().unwrap()) as usize;
        let mut header: Value = serde_json::from_slice(&good[8..8 + n]).unwrap();
        header["emb_in.image.weight"][key] = json!("invalid");
        let json = serde_json::to_vec(&header).unwrap();
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend_from_slice(&good[8 + n..]);
        std::fs::write(&path, bytes).unwrap();
        assert!(resolve_checkpoint(temp.path()).is_err(), "invalid {key}");
    }
}

#[test]
fn index_only_checkpoint_layout_is_typed_unsupported_before_any_shard_read() {
    for flow in [false, true] {
        let temp = package(
            &real_config_json(flow, DitFormat::Dense),
            flow,
            DitFormat::Dense,
        );
        std::fs::remove_file(temp.path().join("model.safetensors")).unwrap();
        std::fs::write(
            temp.path().join(INDEX_FILE),
            b"poison index must not be read",
        )
        .unwrap();
        let expected = match resolve_checkpoint(temp.path()).unwrap_err() {
            Error::Unsupported(message) => message,
            other => panic!("expected typed layout refusal, got {other:?}"),
        };
        assert!(expected.contains("unsupported checkpoint layout"));
        assert!(
            matches!(checkpoint_tensor_headers(temp.path()),Err(Error::Unsupported(message)) if message == expected)
        );
    }
}

#[test]
fn diffusers_headers_are_not_silently_classified_as_base() {
    let temp = package(
        &real_config_json(false, DitFormat::Dense),
        false,
        DitFormat::Dense,
    );
    let mut headers = real_headers(false, DitFormat::Dense);
    for header in &mut headers {
        header.name = format!("diffusers.{}", header.name);
    }
    std::fs::write(temp.path().join("model.safetensors"), stub_bytes(&headers)).unwrap();
    assert!(
        matches!(resolve_checkpoint(temp.path()),Err(Error::Unsupported(message)) if message.contains("native FLUX3 anchor"))
    );
}

#[test]
fn checkpoint_reader_consumes_zero_tensor_payload_bytes() {
    struct HeaderOnly {
        cursor: Cursor<Vec<u8>>,
        limit: u64,
    }
    impl Read for HeaderOnly {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if !output.is_empty() && self.cursor.position() + output.len() as u64 > self.limit {
                return Err(std::io::Error::other("tensor payload read forbidden"));
            }
            self.cursor.read(output)
        }
    }
    impl Seek for HeaderOnly {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(from)
        }
    }
    let bytes = dense_stub_bytes(false);
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut reader = HeaderOnly {
        cursor: Cursor::new(bytes[..8 + n].to_vec()),
        limit: (8 + n) as u64,
    };
    let (headers, _) =
        read_header_from(&mut reader, bytes.len() as u64, Path::new("metadata-only")).unwrap();
    assert_eq!(headers.len(), 357);
    assert_eq!(reader.cursor.position(), (8 + n) as u64);
}

#[test]
fn independent_ae_te_headers_and_real_lora_geometry_match_both_packages() {
    let fixture: Value = serde_json::from_str(REAL_HEADERS).unwrap();
    let ae = rows(&fixture["autoencoder"]);
    assert_eq!(ae.len(), 251);
    assert!(ae.iter().any(|h| h.name == "bn.num_batches_tracked"
        && h.dtype == Dtype::I64
        && h.shape.is_empty()));
    let mut te = BTreeMap::new();
    let index: Value = serde_json::from_str(include_str!(
        "../tests/fixtures/flux3-real-checkpoints/text-encoder-model.safetensors.index.json"
    ))
    .unwrap();
    for (file, value) in fixture["text_encoder"].as_object().unwrap() {
        for header in rows(value) {
            assert_eq!(header.dtype, Dtype::BF16);
            assert_eq!(
                index["weight_map"][&header.name],
                file.trim_start_matches("text_encoder/")
            );
            assert!(te.insert(header.name.clone(), header).is_none());
        }
    }
    assert_eq!(te.len(), 713);
    assert_eq!(index["weight_map"].as_object().unwrap().len(), te.len());
    let config = Flux3MetadataConfig::parse(&real_config_json(false, DitFormat::Dense).to_string())
        .unwrap()
        .for_model(BASE_MODEL_ID)
        .unwrap();
    let targets = crate::training::lora_targets(&config.dit, &[]).unwrap();
    assert_eq!(targets.len(), 258);
    let real: BTreeMap<_, _> = real_headers(false, DitFormat::Dense)
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
    for target in targets {
        assert_eq!(
            real[&format!("{}.weight", target.native)].shape,
            [target.output, target.input]
        );
    }
    assert_eq!(config.text_in_channels_for(2560), 20480);
    assert!(config.text_encoder.layers.iter().all(|layer| *layer <= 36));
}

#[test]
fn catalog_plan_and_header_resolved_training_resume_hashes_are_identical() {
    use crate::training::{identity, options::TrainingOptions};
    use gen_core::{train::TrainingConfig, train::TrainingItem, train::TrainingRequest};
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let raw = real_config_json(false, DitFormat::Dense).to_string();
    std::fs::write(temp.path().join("config.json"), &raw).unwrap();
    let metadata = Flux3MetadataConfig::from_dir(temp.path()).unwrap();
    assert_eq!(metadata.dit.flow_map, None);
    let planned = metadata.for_model(BASE_MODEL_ID).unwrap();
    identity::validate_release_config(&planned).unwrap();
    assert_eq!(
        std::fs::read_dir(temp.path()).unwrap().count(),
        1,
        "pre-download planning reads config only"
    );
    let image = temp.path().join("dataset-input");
    std::fs::write(&image, b"synthetic hash input, never decoded").unwrap();
    let mut request = TrainingRequest {
        items: vec![TrainingItem::captioned(image, "caption".into())],
        config: TrainingConfig {
            optimizer: "adamw".into(),
            train_dtype: "f32".into(),
            timestep_type: "uniform".into(),
            resolution: 64,
            rank: 2,
            alpha: 2.0,
            ..Default::default()
        },
        output_dir: temp.path().join("output"),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec![],
        cancel: gen_core::CancelFlag::new(),
    };
    let options = TrainingOptions::parse(&request).unwrap();
    let targets = crate::training::lora_targets(&planned.dit, &[]).unwrap();
    // Declared artifact pins are identical inputs to this hash-only unit test; production
    // observe_pinned_resources separately authenticates the actual complete files on resume.
    let artifacts = BTreeMap::from([
        (
            "dit/model.safetensors".into(),
            crate::BASE_DIT_SHA256.into(),
        ),
        (
            "autoencoder/ae.safetensors".into(),
            crate::BASE_AUTOENCODER_SHA256.into(),
        ),
        (
            "package/config.json".into(),
            format!("{:x}", Sha256::digest(raw.as_bytes())),
        ),
    ]);
    let before = identity::build(
        &request,
        &planned,
        &options,
        artifacts.clone(),
        &targets,
        &[],
        "candle",
    )
    .unwrap();
    std::fs::write(
        temp.path().join("model.safetensors"),
        dense_stub_bytes(false),
    )
    .unwrap();
    let resolved = resolve_checkpoint(temp.path()).unwrap();
    resolved.validate_route(BASE_MODEL_ID).unwrap();
    identity::validate_release_config(&resolved).unwrap();
    request.config.resume = true;
    let after = identity::build(
        &request,
        &resolved,
        &options,
        artifacts,
        &targets,
        &[],
        "candle",
    )
    .unwrap();
    assert_eq!(
        before, after,
        "same catalog plan must not rekey resume after header resolution"
    );
}

#[test]
fn duplicate_header_names_are_refused_instead_of_overwritten() {
    let temp = package(
        &real_config_json(false, DitFormat::Dense),
        false,
        DitFormat::Dense,
    );
    let original = dense_stub_bytes(false);
    let n = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let map: Value = serde_json::from_slice(&original[8..8 + n]).unwrap();
    let duplicate = map["emb_in.image.weight"].clone();
    let body = serde_json::to_string(&map).unwrap();
    let json = format!("{{\"emb_in.image.weight\":{duplicate},{}", &body[1..]);
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json.as_bytes());
    bytes.extend_from_slice(&original[8 + n..]);
    std::fs::write(temp.path().join("model.safetensors"), bytes).unwrap();
    assert!(resolve_checkpoint(temp.path()).is_err());
}

/// Explicit caller-provisioned snapshots, HEADERS/JSON ONLY. No tensor/provider/device path.
#[test]
#[ignore = "header-only real snapshot audit: provide FLUX3_REAL_BASE_HEADER_DIR and FLUX3_REAL_DISTILLED_HEADER_DIR"]
fn explicitly_provisioned_real_snapshot_headers_only() {
    let fixture: Value = serde_json::from_str(REAL_HEADERS).unwrap();
    for (flow, variable) in [
        (false, "FLUX3_REAL_BASE_HEADER_DIR"),
        (true, "FLUX3_REAL_DISTILLED_HEADER_DIR"),
    ] {
        let root = std::path::PathBuf::from(
            std::env::var_os(variable).expect("provide explicit snapshot directory"),
        );
        for (format, relative) in [
            (DitFormat::Dense, ""),
            (DitFormat::Fp8Rowwise, "variants/fp8r"),
            (DitFormat::Nvfp4, "variants/nvfp4"),
        ] {
            let dir = root.join(relative);
            let raw: Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                    .unwrap();
            assert_eq!(raw, real_config_json(flow, format));
            let config = resolve_checkpoint(&dir).unwrap();
            config
                .validate_route(if flow {
                    DISTILLED_MODEL_ID
                } else {
                    BASE_MODEL_ID
                })
                .unwrap();
            let actual = checkpoint_tensor_headers(&dir).unwrap();
            NativeDitStorage::Publisher(format)
                .validate_headers(&config.dit, &actual)
                .unwrap();
            let by_name = |headers: Vec<SafetensorsTensorHeader>| {
                headers
                    .into_iter()
                    .map(|h| (h.name.clone(), h))
                    .collect::<BTreeMap<_, _>>()
            };
            assert_eq!(by_name(actual), by_name(real_headers(flow, format)));
            let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(dir.clone()));
            let resources = crate::GenerationResources::from_spec(
                &spec,
                &config,
                if flow {
                    DISTILLED_MODEL_ID
                } else {
                    BASE_MODEL_ID
                },
            )
            .unwrap();
            assert_eq!(resources.text_encoder_dir, root.join("text_encoder"));
            assert_eq!(resources.autoencoder_file, root.join("ae.safetensors"));
        }
        let by_name = |headers: Vec<SafetensorsTensorHeader>| {
            headers
                .into_iter()
                .map(|h| (h.name.clone(), h))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(
            by_name(
                gen_core::safetensors_path_tensor_headers(root.join("ae.safetensors")).unwrap()
            ),
            by_name(rows(&fixture["autoencoder"]))
        );
        for (relative, value) in fixture["text_encoder"].as_object().unwrap() {
            assert_eq!(
                by_name(gen_core::safetensors_path_tensor_headers(root.join(relative)).unwrap()),
                by_name(rows(value))
            );
        }
    }
}

fn component_files(root: &Path) {
    std::fs::create_dir_all(root.join("text_encoder")).unwrap();
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        std::fs::write(
            root.join("text_encoder").join(name),
            b"path fixture, never loaded",
        )
        .unwrap();
    }
    std::fs::write(root.join("ae.safetensors"), b"path fixture, never loaded").unwrap();
}

#[test]
fn publisher_variant_shared_components_resolve_from_root_and_overrides_win() {
    use gen_core::{LoadSpec, WeightsSource};
    for flow in [false, true] {
        for (format, name) in [(DitFormat::Fp8Rowwise, "fp8r"), (DitFormat::Nvfp4, "nvfp4")] {
            let root = package(
                &real_config_json(flow, DitFormat::Dense),
                flow,
                DitFormat::Dense,
            );
            component_files(root.path());
            let variant = root.path().join("variants").join(name);
            std::fs::create_dir_all(&variant).unwrap();
            std::fs::write(
                variant.join("config.json"),
                real_config_json(flow, format).to_string(),
            )
            .unwrap();
            std::fs::write(
                variant.join("model.safetensors"),
                stub_bytes(&real_headers(flow, format)),
            )
            .unwrap();
            let resolved = resolve_checkpoint(&variant).unwrap();
            let id = if flow {
                DISTILLED_MODEL_ID
            } else {
                BASE_MODEL_ID
            };
            let mut spec = LoadSpec::new(WeightsSource::Dir(variant.clone()));
            let resources = crate::GenerationResources::from_spec(&spec, &resolved, id).unwrap();
            assert_eq!(resources.package_dir, variant);
            assert_eq!(resources.text_encoder_dir, root.path().join("text_encoder"));
            assert_eq!(
                resources.autoencoder_file,
                root.path().join("ae.safetensors")
            );
            assert!(crate::GenerationResources::from_spec(
                &spec,
                &resolved,
                if flow {
                    BASE_MODEL_ID
                } else {
                    DISTILLED_MODEL_ID
                }
            )
            .is_err());
            let overrides = tempfile::tempdir().unwrap();
            component_files(overrides.path());
            spec.components.insert(
                crate::TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(overrides.path().join("text_encoder")),
            );
            spec.components.insert(
                crate::AUTOENCODER_COMPONENT.into(),
                WeightsSource::File(overrides.path().join("ae.safetensors")),
            );
            std::fs::remove_file(root.path().join("config.json")).unwrap();
            let resources = crate::GenerationResources::from_spec(&spec, &resolved, id).unwrap();
            assert_eq!(
                resources.text_encoder_dir,
                overrides.path().join("text_encoder")
            );
            assert_eq!(
                resources.autoencoder_file,
                overrides.path().join("ae.safetensors")
            );
        }
    }
}

#[test]
fn publisher_shared_root_must_describe_the_same_graph() {
    use gen_core::{LoadSpec, WeightsSource};
    let mut root_config = real_config_json(false, DitFormat::Dense);
    root_config["dit"]["hidden"] = json!(2048);
    root_config["dit"]["heads"] = json!(16);
    let root = package(&root_config, false, DitFormat::Dense);
    component_files(root.path());
    let variant = root.path().join("variants/fp8r");
    std::fs::create_dir_all(&variant).unwrap();
    std::fs::write(
        variant.join("config.json"),
        real_config_json(false, DitFormat::Fp8Rowwise).to_string(),
    )
    .unwrap();
    std::fs::write(
        variant.join("model.safetensors"),
        stub_bytes(&real_headers(false, DitFormat::Fp8Rowwise)),
    )
    .unwrap();
    let resolved = resolve_checkpoint(&variant).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(variant));
    assert!(
        matches!(crate::GenerationResources::from_spec(&spec,&resolved,BASE_MODEL_ID),Err(Error::Unsupported(message)) if message.contains("different graphs"))
    );
}
