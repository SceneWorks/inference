use super::*;
use serde_json::json;

const TEST_TOKENIZER: EncoderTokenizerContract = EncoderTokenizerContract {
    family: "test-qwen3",
    binding: EncoderTokenizerBinding::RetainBase,
    artifact_candidates: &["tokenizer/tokenizer.json"],
    required_tokens: &[EncoderRequiredToken {
        role: "test_special",
        literal: "<test>",
        id: 1,
        config_field: None,
    }],
};
const TEST_PROMPTS: &[EncoderPromptExecutionContract] = &[EncoderPromptExecutionContract {
    purpose: "test_prompt",
    template: EncoderPromptTemplate::QwenInstruct,
    add_special_tokens: true,
    length: EncoderPromptLengthPolicy::RightTruncate { max_tokens: 8 },
    padding: EncoderPromptPadding::None,
    prefix_trim: 0,
}];

const CONTRACT: EncoderContract = EncoderContract {
    architecture: "qwen3",
    hidden_size: 8,
    intermediate_size: 12,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 4,
    vocab_size: 16,
    output_width: 8,
    loaded_hidden_layers: 2,
    requires_final_norm: true,
    requires_lm_head: false,
    hidden_activation: "silu",
    attention_dropout: EncoderConfigFloat::new(0.0),
    rms_norm_eps: EncoderConfigFloat::new(1e-6),
    qk_norm_eps: Some(EncoderConfigFloat::new(1e-6)),
    rope_theta: EncoderConfigFloat::new(1_000_000.0),
    max_position_embeddings: 4_096,
    attention_bias: EncoderConfigBool::Required(false),
    tie_word_embeddings: EncoderConfigBool::Required(true),
    tokenizer: TEST_TOKENIZER,
    prompt_executions: TEST_PROMPTS,
    bos_token_id: None,
    eos_token_id: None,
    image_token_id: None,
    vision_start_token_id: None,
    vision_end_token_id: None,
    mrope_section: &[],
    mrope_interleaved: None,
    selected_hidden_layers: &[2],
    packing: Some(EncoderPackingContract {
        group_size: 64,
        pack_embedding: false,
        pack_lm_head: false,
        supports_file: true,
    }),
    dense_storage_dtype_probe: Some("model.layers.0.input_layernorm.weight"),
};

#[test]
fn malformed_provider_contract_fails_before_source_inspection() {
    let malformed = EncoderContract {
        num_key_value_heads: 3,
        ..CONTRACT
    };
    let error = malformed
        .validate_source(&WeightsSource::Dir(PathBuf::from(
            "definitely-not-inspected",
        )))
        .expect_err("malformed provider contract must reject first")
        .to_string();
    assert!(error.contains("attention heads (2)"), "{error}");
    assert!(error.contains("key/value heads (3)"), "{error}");
    assert!(!error.contains("No such file"), "{error}");
}

fn write_fixture_with_shapes(root: &Path, embedding_hidden: usize, q_projection_output: usize) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen3",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 16,
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 4096,
            "attention_bias": false,
            "tie_word_embeddings": true
        }))
        .unwrap(),
    )
    .unwrap();
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    let mut tensors = vec![(
        "model.embed_tokens.weight".to_owned(),
        vec![16, embedding_hidden],
    )];
    for layer in 0..2 {
        let base = format!("model.layers.{layer}");
        tensors.extend([
            (
                format!("{base}.self_attn.q_proj.weight"),
                vec![if layer == 0 { q_projection_output } else { 8 }, 8],
            ),
            (format!("{base}.self_attn.k_proj.weight"), vec![4, 8]),
            (format!("{base}.self_attn.v_proj.weight"), vec![4, 8]),
            (format!("{base}.self_attn.o_proj.weight"), vec![8, 8]),
            (format!("{base}.self_attn.q_norm.weight"), vec![4]),
            (format!("{base}.self_attn.k_norm.weight"), vec![4]),
            (format!("{base}.mlp.gate_proj.weight"), vec![12, 8]),
            (format!("{base}.mlp.up_proj.weight"), vec![12, 8]),
            (format!("{base}.mlp.down_proj.weight"), vec![8, 12]),
            (format!("{base}.input_layernorm.weight"), vec![8]),
            (format!("{base}.post_attention_layernorm.weight"), vec![8]),
        ]);
    }
    tensors.push(("model.norm.weight".to_owned(), vec![8]));
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<usize>() * 4;
        header.insert(
            name,
            json!({"dtype":"F32", "shape":shape, "data_offsets":[offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut file = Vec::with_capacity(8 + encoded.len() + offset);
    file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    file.extend_from_slice(&encoded);
    file.resize(8 + encoded.len() + offset, 0);
    std::fs::write(root.join("model.safetensors"), file).unwrap();
}

fn write_fixture(root: &Path, embedding_hidden: usize) {
    write_fixture_with_shapes(root, embedding_hidden, 8);
}

fn write_tokenizer_fixture(root: &Path) {
    let tokenizer_dir = root.join("tokenizer");
    std::fs::create_dir_all(&tokenizer_dir).unwrap();
    std::fs::write(
        tokenizer_dir.join("tokenizer.json"),
        serde_json::to_vec(&json!({
            "added_tokens": [{"id": 1, "content": "<test>"}],
            "model": {"vocab": {}}
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_prefixed_fixture(root: &Path, outer_prefix: &str) -> PathBuf {
    let source = std::fs::read(root.join("model.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(source[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> =
        serde_json::from_slice(&source[8..8 + header_len]).unwrap();
    let mut prefixed = serde_json::Map::new();
    let mut data_len = 0usize;
    for (name, value) in header {
        data_len = data_len.max(
            value["data_offsets"][1]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .unwrap(),
        );
        prefixed.insert(format!("{outer_prefix}{name}"), value);
    }
    let encoded = serde_json::to_vec(&prefixed).unwrap();
    let mut file = Vec::with_capacity(8 + encoded.len() + data_len);
    file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    file.extend_from_slice(&encoded);
    file.resize(8 + encoded.len() + data_len, 0);
    let path = root.join("combined.safetensors");
    std::fs::write(&path, file).unwrap();
    path
}

fn rewrite_config_field(root: &Path, field: &str, value: usize) {
    let path = root.join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config[field] = json!(value);
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
}

fn rewrite_config_quantization(root: &Path, bits: i32) {
    let path = root.join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["quantization"] = json!({"bits": bits, "group_size": 64});
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
}

fn authorized_test_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        std::path::absolute(root).unwrap(),
        std::fs::canonicalize(root).unwrap(),
    ];
    roots.sort();
    roots.dedup();
    roots
}

#[derive(Clone, Copy, Debug)]
enum DiscoverySourceShape {
    File,
    DirectDirectory,
    CompleteSnapshot,
}

fn write_discovery_source(
    root: &Path,
    shape: DiscoverySourceShape,
    embedding_hidden: usize,
) -> (WeightsSource, PathBuf) {
    match shape {
        DiscoverySourceShape::File => {
            let component = root.join("file");
            write_fixture(&component, embedding_hidden);
            (
                WeightsSource::File(component.join("model.safetensors")),
                component,
            )
        }
        DiscoverySourceShape::DirectDirectory => {
            let component = root.join("direct");
            write_fixture(&component, embedding_hidden);
            (WeightsSource::Dir(component.clone()), component)
        }
        DiscoverySourceShape::CompleteSnapshot => {
            let snapshot = root.join("snapshot");
            let component = snapshot.join("text_encoder");
            write_fixture(&component, embedding_hidden);
            (WeightsSource::Dir(snapshot), component)
        }
    }
}

#[test]
fn validates_config_and_headers_before_load() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap();
    validated
        .read_unchanged::<(), Error>(|source| match source {
            WeightsSource::Dir(_) => Ok(()),
            WeightsSource::File(path) => Err(Error::Msg(format!(
                "unexpected file encoder source: {}",
                path.display()
            ))),
        })
        .unwrap();
}

#[test]
fn discovery_validation_does_no_full_hash_work_while_load_validation_still_seals() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let source = WeightsSource::Dir(selected);
    let allowed_roots = authorized_test_roots(temp.path());

    let before_discovery = crate::runtime::test_full_hash_work_count();
    let facts = CONTRACT
        .validate_source_for_discovery(&source, &allowed_roots)
        .unwrap();
    assert!(!facts.materialized_language_tensor_headers().is_empty());
    assert!(facts.source_bytes() > 0);
    let inventory = text_encoder_source_inventory_for_discovery(&source, &allowed_roots).unwrap();
    assert!(!inventory.tensor_headers().is_empty());
    assert_eq!(inventory.source_bytes(), facts.source_bytes());
    let discovery_planning =
        text_encoder_planning_facts_for_discovery(&source, &allowed_roots).unwrap();
    assert_eq!(
        discovery_planning.source_layout(),
        TextEncoderSourceLayout::DirectDirectory
    );
    assert_eq!(
        discovery_planning.direct_shard_bytes(),
        facts.source_bytes()
    );
    let comfyui_source = WeightsSource::File(match &source {
        WeightsSource::Dir(path) => path.join("model.safetensors"),
        WeightsSource::File(_) => unreachable!("the fixture source is a directory"),
    });
    let comfyui = CONTRACT
        .validate_comfyui_source_for_discovery(&comfyui_source, &allowed_roots)
        .unwrap();
    assert_eq!(comfyui.source_bytes(), facts.source_bytes());
    assert_eq!(
        crate::runtime::test_full_hash_work_count(),
        before_discovery,
        "catalog discovery must not acquire and discard a full-content artifact seal"
    );

    let before_load = crate::runtime::test_full_hash_work_count();
    let sealed = CONTRACT
        .validate_source_against_base(&source, &base)
        .unwrap();
    assert!(
        crate::runtime::test_full_hash_work_count() > before_load,
        "executable validation must retain full acquisition sealing"
    );

    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base));
    sealed.prepare_load_spec(&mut spec).unwrap();
    let before_prepared_planning = crate::runtime::test_full_hash_work_count();
    let prepared_planning = spec
        .prepared_text_encoder_planning_facts()
        .unwrap()
        .expect("validated source installed a receipt");
    assert_eq!(prepared_planning, discovery_planning);
    assert_eq!(
        crate::runtime::test_full_hash_work_count(),
        before_prepared_planning,
        "prepared planning must reuse the retained seal instead of hashing content again"
    );

    let selected_shard = match &source {
        WeightsSource::Dir(path) => path.join("model.safetensors"),
        WeightsSource::File(_) => unreachable!("the fixture source is a directory"),
    };
    let shard_len = std::fs::metadata(&selected_shard).unwrap().len();
    std::fs::write(&selected_shard, vec![1_u8; shard_len as usize]).unwrap();
    let error = spec
        .prepared_text_encoder_planning_facts()
        .expect_err("prepared planning must reject a mutated selected source")
        .to_string();
    assert!(error.contains("receipt changed"), "{error}");
}

#[test]
fn discovery_and_sealed_validation_have_source_shape_and_contract_parity() {
    #[derive(Clone, Copy, Debug)]
    enum Mutation {
        None,
        Config,
        Header,
        Quantization,
    }

    for shape in [
        DiscoverySourceShape::File,
        DiscoverySourceShape::DirectDirectory,
        DiscoverySourceShape::CompleteSnapshot,
    ] {
        for mutation in [
            Mutation::None,
            Mutation::Config,
            Mutation::Header,
            Mutation::Quantization,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let header_width = if matches!(mutation, Mutation::Header) {
                7
            } else {
                8
            };
            let (source, component) = write_discovery_source(temp.path(), shape, header_width);
            match mutation {
                Mutation::None | Mutation::Header => {}
                Mutation::Config => rewrite_config_field(&component, "hidden_size", 7),
                Mutation::Quantization => rewrite_config_quantization(&component, 4),
            }

            let discovery = CONTRACT
                .validate_source_for_discovery(&source, &authorized_test_roots(temp.path()));
            let sealed = CONTRACT.validate_source_for_planning(&source);
            match mutation {
                Mutation::None => {
                    let facts = discovery.unwrap_or_else(|error| {
                        panic!("{shape:?} discovery unexpectedly failed: {error}")
                    });
                    let sealed = sealed.unwrap_or_else(|error| {
                        panic!("{shape:?} sealed validation unexpectedly failed: {error}")
                    });
                    assert_eq!(facts.source_bytes(), sealed.source_bytes().unwrap());
                    assert_eq!(
                        facts.materialized_language_tensor_headers(),
                        sealed
                            .materialized_language_tensor_headers(&CONTRACT)
                            .unwrap()
                    );
                }
                Mutation::Config => {
                    let discovery = discovery.unwrap_err().to_string();
                    let sealed = sealed.unwrap_err().to_string();
                    assert!(discovery.contains("field hidden_size"), "{discovery}");
                    assert!(sealed.contains("field hidden_size"), "{sealed}");
                }
                Mutation::Header => {
                    let discovery = discovery.unwrap_err().to_string();
                    let sealed = sealed.unwrap_err().to_string();
                    assert!(discovery.contains("field hidden_size"), "{discovery}");
                    assert!(sealed.contains("field hidden_size"), "{sealed}");
                }
                Mutation::Quantization => {
                    let discovery = discovery.unwrap_err().to_string();
                    let sealed = sealed.unwrap_err().to_string();
                    assert!(discovery.contains("quantization mismatch"), "{discovery}");
                    assert!(sealed.contains("quantization mismatch"), "{sealed}");
                }
            }
        }
    }
}

#[test]
fn required_config_is_rejected_before_shard_enumeration() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("selected");
    std::fs::create_dir_all(source.join("broken.safetensors")).unwrap();
    let error = CONTRACT
        .validate_source_for_discovery(
            &WeightsSource::Dir(source),
            &authorized_test_roots(temp.path()),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no config.json"), "{error}");
    assert!(!error.contains("not a regular file"), "{error}");
}

#[test]
fn discovery_validation_rejects_config_header_and_source_shape_mutations() {
    let temp = tempfile::tempdir().unwrap();
    let allowed_roots = authorized_test_roots(temp.path());

    let config_mutation = temp.path().join("config-mutation");
    write_fixture(&config_mutation, 8);
    rewrite_config_field(&config_mutation, "hidden_size", 7);
    let error = CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(config_mutation), &allowed_roots)
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");

    let header_mutation = temp.path().join("header-mutation");
    write_fixture(&header_mutation, 7);
    let error = CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(header_mutation), &allowed_roots)
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");

    let source_shape_mutation = temp.path().join("source-shape-mutation");
    write_fixture(&source_shape_mutation, 8);
    CONTRACT
        .validate_source_for_discovery(
            &WeightsSource::Dir(source_shape_mutation.clone()),
            &allowed_roots,
        )
        .unwrap();
    let nested = source_shape_mutation.join("text_encoder");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::copy(
        source_shape_mutation.join("config.json"),
        nested.join("config.json"),
    )
    .unwrap();
    let error = CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(source_shape_mutation), &allowed_roots)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no direct .safetensors shards"), "{error}");
}

#[test]
fn discovery_and_sealed_validation_share_the_config_size_policy() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    std::fs::OpenOptions::new()
        .write(true)
        .open(temp.path().join("config.json"))
        .unwrap()
        .set_len(MAX_SELECTED_ENCODER_CONFIG_BYTES + 1)
        .unwrap();
    let source = WeightsSource::Dir(temp.path().to_path_buf());
    let discovery_error = CONTRACT
        .validate_source_for_discovery(&source, &authorized_test_roots(temp.path()))
        .unwrap_err()
        .to_string();
    let sealed_error = CONTRACT
        .validate_source_for_planning(&source)
        .unwrap_err()
        .to_string();
    assert!(
        discovery_error.contains("exceeds the 4194304-byte maximum"),
        "{discovery_error}"
    );
    assert!(
        sealed_error.contains("exceeds the 4194304-byte maximum"),
        "{sealed_error}"
    );
}

#[cfg(unix)]
#[test]
fn discovery_validation_rejects_canonical_target_outside_authorized_roots() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let allowed = temp.path().join("allowed");
    let outside = temp.path().join("outside");
    write_fixture(&outside, 8);
    std::fs::create_dir_all(&allowed).unwrap();
    std::fs::copy(outside.join("config.json"), allowed.join("config.json")).unwrap();
    symlink(
        outside.join("model.safetensors"),
        allowed.join("model.safetensors"),
    )
    .unwrap();

    let allowed_roots = authorized_test_roots(&allowed);
    let error = CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(allowed.clone()), &allowed_roots)
        .unwrap_err()
        .to_string();
    assert!(error.contains("canonical target"), "{error}");
    assert!(error.contains("escapes authorized model roots"), "{error}");
}

#[cfg(unix)]
#[test]
fn discovery_confinement_covers_file_direct_directory_and_complete_snapshot() {
    use std::os::unix::fs::symlink;

    for shape in [
        DiscoverySourceShape::File,
        DiscoverySourceShape::DirectDirectory,
        DiscoverySourceShape::CompleteSnapshot,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let outside = temp.path().join("outside");
        let (source, component) = write_discovery_source(&allowed, shape, 8);
        write_fixture(&outside, 8);
        std::fs::remove_file(component.join("model.safetensors")).unwrap();
        symlink(
            outside.join("model.safetensors"),
            component.join("model.safetensors"),
        )
        .unwrap();

        let error = CONTRACT
            .validate_source_for_discovery(&source, &authorized_test_roots(&allowed))
            .unwrap_err()
            .to_string();
        assert!(error.contains("canonical target"), "{shape:?}: {error}");
        assert!(
            error.contains("escapes authorized model roots"),
            "{shape:?}: {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn discovery_rejects_snapshot_component_symlink_before_inspecting_its_config() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let allowed = temp.path().join("allowed");
    let snapshot = allowed.join("snapshot");
    let outside_component = temp.path().join("outside/text_encoder");
    std::fs::create_dir_all(&snapshot).unwrap();
    write_fixture(&outside_component, 8);
    symlink(&outside_component, snapshot.join("text_encoder")).unwrap();

    let error = CONTRACT
        .validate_source_for_discovery(
            &WeightsSource::Dir(snapshot),
            &authorized_test_roots(&allowed),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("canonical target"), "{error}");
    assert!(error.contains("escapes authorized model roots"), "{error}");
}

#[test]
fn duplicate_direct_shard_keys_fail_before_backend_selection() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    std::fs::copy(
        temp.path().join("model.safetensors"),
        temp.path().join("model-00002.safetensors"),
    )
    .unwrap();

    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .expect_err("cross-backend duplicate semantics must fail closed")
        .to_string();
    assert!(error.contains("duplicate tensor key"), "{error}");
}

#[test]
fn configless_file_rejects_unprovable_behavior_and_head_topology() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = temp.path().join("model.safetensors");
    std::fs::remove_file(temp.path().join("config.json")).unwrap();
    let error = CONTRACT
        .validate_source(&WeightsSource::File(file))
        .expect_err("headers cannot prove behavior or tokenizer compatibility")
        .to_string();
    assert!(error.contains("has no config.json"), "{error}");
    assert!(error.contains("tokenizer"), "{error}");
}

#[test]
fn provider_owned_comfyui_file_preserves_the_legacy_configless_surface() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = temp.path().join("model.safetensors");
    std::fs::remove_file(temp.path().join("config.json")).unwrap();

    let validated = CONTRACT
        .validate_comfyui_source(&WeightsSource::File(file.clone()))
        .unwrap();
    validated
        .read_unchanged::<(), Error>(|source| match source {
            WeightsSource::File(actual) if actual == &file => Ok(()),
            other => Err(Error::Msg(format!("unexpected encoder source: {other:?}"))),
        })
        .unwrap();
    assert!(!validated.has_config());
}

#[test]
fn provider_owned_comfyui_file_still_rejects_a_present_wrong_config() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    rewrite_config_field(temp.path(), "hidden_size", 7);

    let error = CONTRACT
        .validate_comfyui_source(&WeightsSource::File(temp.path().join("model.safetensors")))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");
    assert!(error.contains("expected 8, got 7"), "{error}");
}

#[test]
fn validates_dense_encoder_embedded_in_fused_checkpoint() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = write_prefixed_fixture(temp.path(), "text_encoder.");
    CONTRACT
        .validate_embedded_file(&file, &["text_encoder."])
        .unwrap();
}

#[test]
fn provider_owned_comfyui_checkpoint_preserves_the_legacy_configless_surface() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = write_prefixed_fixture(temp.path(), "text_encoder.");
    std::fs::remove_file(temp.path().join("config.json")).unwrap();
    CONTRACT
        .validate_embedded_comfyui_file(&file, &["text_encoder."])
        .unwrap();
}

#[test]
fn ordinary_embedded_encoder_still_requires_its_own_config() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = write_prefixed_fixture(temp.path(), "text_encoder.");
    std::fs::remove_file(temp.path().join("config.json")).unwrap();
    let error = CONTRACT
        .validate_embedded_file(&file, &["text_encoder."])
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires sibling config.json"), "{error}");
}

#[test]
fn embedded_encoder_header_mismatch_rejects_before_payload_load() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 7);
    let file = write_prefixed_fixture(temp.path(), "text_encoder.");
    let error = CONTRACT
        .validate_embedded_file(&file, &["text_encoder."])
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");
    assert!(error.contains("expected 8, got 7"), "{error}");
}

#[test]
fn wrong_tensor_shape_names_the_exact_contract_field() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 7);
    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");
    assert!(error.contains("expected 8, got 7"), "{error}");
}

#[test]
fn wrong_config_hidden_size_names_the_exact_contract_field() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    rewrite_config_field(temp.path(), "hidden_size", 7);
    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field hidden_size"), "{error}");
    assert!(error.contains("expected 8, got 7"), "{error}");
}

#[test]
fn wrong_config_layer_count_names_the_exact_contract_field() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    rewrite_config_field(temp.path(), "num_hidden_layers", 3);
    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field num_hidden_layers"), "{error}");
    assert!(error.contains("expected 2, got 3"), "{error}");
}

#[test]
fn wrong_projection_shape_names_the_exact_head_contract_field() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture_with_shapes(temp.path(), 8, 7);
    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("field num_attention_heads/head_dim"),
        "{error}"
    );
    assert!(error.contains("expected 8, got 7"), "{error}");
}

#[test]
fn absent_override_validates_and_preserves_the_builtin_source() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(base);
    let source = CONTRACT
        .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
        .unwrap();
    source
        .read_unchanged::<(), Error>(|source| match source {
            WeightsSource::Dir(path) if path.as_path() == base.join("text_encoder") => Ok(()),
            other => Err(Error::Msg(format!("unexpected encoder source: {other:?}"))),
        })
        .unwrap();
}

#[test]
fn retained_tokenizer_rejects_replacement_before_runtime_open() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(base);
    let source = CONTRACT
        .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
        .unwrap();
    let tokenizer = base.join("tokenizer/tokenizer.json");
    let replacement = base.join("tokenizer/replacement.json");
    std::fs::write(
        &replacement,
        br#"{"added_tokens":[{"id":1,"content":"<test>"}],"model":{"vocab":{"replacement":2}}}"#,
    )
    .unwrap();
    std::fs::rename(replacement, tokenizer).unwrap();

    let opened = std::cell::Cell::new(false);
    let error = source
        .read_tokenizer_unchanged::<(), Error>(|_| {
            opened.set(true);
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(!opened.get(), "mutated tokenizer bytes must not be opened");
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn retained_tokenizer_rejects_swap_and_restore_inside_read_bracket() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(base);
    let source = CONTRACT
        .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
        .unwrap();
    let tokenizer = base.join("tokenizer/tokenizer.json");
    let original = base.join("tokenizer/original.json");

    let error = source
            .read_tokenizer_unchanged::<(), Error>(|path| {
                std::fs::rename(path, &original)?;
                std::fs::write(
                    path,
                    br#"{"added_tokens":[{"id":1,"content":"<test>"}],"model":{"vocab":{"swapped":2}}}"#,
                )?;
                std::fs::remove_file(path)?;
                std::fs::rename(&original, path)?;
                Ok(())
            })
            .unwrap_err()
            .to_string();
    assert!(error.contains("pinned weights"), "{error}");
    assert!(tokenizer.is_file());
}

#[cfg(unix)]
#[test]
fn retained_tokenizer_rejects_symlink_retarget_before_runtime_open() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(base);
    let tokenizer = base.join("tokenizer/tokenizer.json");
    let first = base.join("tokenizer/first.json");
    let second = base.join("tokenizer/second.json");
    std::fs::rename(&tokenizer, &first).unwrap();
    std::fs::copy(&first, &second).unwrap();
    symlink(&first, &tokenizer).unwrap();
    let source = CONTRACT
        .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
        .unwrap();

    std::fs::remove_file(&tokenizer).unwrap();
    symlink(&second, &tokenizer).unwrap();
    let opened = std::cell::Cell::new(false);
    let error = source
        .read_tokenizer_unchanged::<(), Error>(|_| {
            opened.set(true);
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(!opened.get(), "retargeted tokenizer must fail before open");
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn semantically_identical_complete_snapshot_retains_the_base_runtime_tokenizer() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(&base);
    write_fixture(&selected.join("text_encoder"), 8);
    std::fs::create_dir_all(selected.join("tokenizer")).unwrap();
    std::fs::write(
        selected.join("tokenizer/tokenizer.json"),
        br#"{ "model": { "vocab": {} }, "added_tokens": [ { "content": "<test>", "id": 1 } ] }"#,
    )
    .unwrap();
    let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
        .with_text_encoder(WeightsSource::Dir(selected));
    let source = CONTRACT.source_for_load(&spec, &base).unwrap();

    assert_eq!(
        source.tokenizer_disposition(),
        Some(EncoderTokenizerDisposition::MatchedSelectedTokenizer)
    );
    let runtime_path = source
        .read_tokenizer_unchanged::<PathBuf, Error>(|path| Ok(path.to_path_buf()))
        .unwrap();
    assert_eq!(
        runtime_path,
        std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap()
    );
}

#[test]
fn file_selection_prepares_shard_sibling_config_and_base_tokenizer_identity() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let selected_file = selected.join("model.safetensors");
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::File(selected_file.clone()), &base)
        .unwrap();

    validated.prepare_load_spec(&mut spec).unwrap();

    assert_eq!(
        spec.text_encoder,
        Some(WeightsSource::File(selected_file.clone()))
    );
    let paths = spec
        .prepared_file_pins()
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert!(paths.contains(&std::path::absolute(&selected_file).unwrap()));
    assert!(paths.contains(&std::path::absolute(selected.join("config.json")).unwrap()));
    assert!(paths.contains(&std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap()));

    std::fs::write(selected.join("config.json"), b"{}").unwrap();
    let opened = std::cell::Cell::new(false);
    let error = spec
        .read_prepared_files_unchanged::<(), Error>(|| {
            opened.set(true);
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(
        !opened.get(),
        "mutated config must fail before provider load"
    );
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn complete_snapshot_prepares_selected_and_base_tokenizer_identity() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected.join("text_encoder"), 8);
    write_tokenizer_fixture(&selected);
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
        .unwrap();

    validated.prepare_load_spec(&mut spec).unwrap();

    assert_eq!(
        spec.text_encoder,
        Some(WeightsSource::Dir(selected.clone()))
    );
    let paths = spec
        .prepared_file_pins()
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    for path in [
        selected.join("text_encoder/model.safetensors"),
        selected.join("text_encoder/config.json"),
        selected.join("tokenizer/tokenizer.json"),
        base.join("tokenizer/tokenizer.json"),
    ] {
        assert!(
            paths.contains(&std::path::absolute(path).unwrap()),
            "missing prepared receipt path"
        );
    }

    std::fs::write(
        selected.join("tokenizer/tokenizer.json"),
        br#"{"added_tokens":[]}"#,
    )
    .unwrap();
    let opened = std::cell::Cell::new(false);
    let error = spec
        .read_prepared_files_unchanged::<(), Error>(|| {
            opened.set(true);
            Ok(())
        })
        .expect_err("mutated selected tokenizer must fail before provider load")
        .to_string();
    assert!(!opened.get());
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn prepared_directory_receipt_rejects_shard_inventory_mutation() {
    for mutation in ["addition", "removal", "rename", "type"] {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let shard = selected.join("model.safetensors");
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        match mutation {
            "addition" => {
                std::fs::copy(&shard, selected.join("added.safetensors")).unwrap();
            }
            "removal" => std::fs::remove_file(&shard).unwrap(),
            "rename" => {
                std::fs::rename(&shard, selected.join("renamed.safetensors")).unwrap();
            }
            "type" => {
                std::fs::remove_file(&shard).unwrap();
                std::fs::create_dir(&shard).unwrap();
            }
            _ => unreachable!(),
        }

        let opened = std::cell::Cell::new(false);
        let error = spec
            .read_prepared_files_unchanged::<(), Error>(|| {
                opened.set(true);
                Ok(())
            })
            .expect_err("mutated direct-shard inventory must fail before provider load")
            .to_string();
        assert!(!opened.get(), "{mutation} reached the provider callback");
        assert!(
            error.contains("receipt changed") || error.contains("pinned weights"),
            "{mutation}: {error}"
        );
    }
}

#[test]
fn source_for_load_rechecks_prepared_directory_inventory() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let shard = selected.join("model.safetensors");
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    std::fs::copy(shard, selected.join("added.safetensors")).unwrap();

    let error = CONTRACT
        .source_for_load(&spec, &base)
        .expect_err("direct provider validation must retain the prepared shard inventory")
        .to_string();
    assert!(error.contains("receipt changed"), "{error}");
}

#[test]
fn prepared_encoder_receipt_rejects_selected_source_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    spec.text_encoder = Some(WeightsSource::Dir(temp.path().join("replacement")));
    let error = spec
        .validate_prepared_file_pins()
        .expect_err("prepared encoder source replacement must fail closed")
        .to_string();
    assert!(error.contains("no longer matches"), "{error}");
}

#[test]
fn prepared_file_receipt_rejects_later_config_sidecar() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let selected_file = selected.join("model.safetensors");
    std::fs::remove_file(selected.join("config.json")).unwrap();
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_comfyui_source_against_base(&WeightsSource::File(selected_file), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    std::fs::write(selected.join("config.json"), b"{}").unwrap();
    let opened = std::cell::Cell::new(false);
    let error = spec
        .read_prepared_files_unchanged::<(), Error>(|| {
            opened.set(true);
            Ok(())
        })
        .expect_err("a newly added behavior sidecar must fail before provider load")
        .to_string();
    assert!(!opened.get());
    assert!(error.contains("receipt changed"), "{error}");
}

#[test]
fn prepared_direct_directory_rejects_complete_snapshot_reinterpretation() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    write_fixture(&selected.join("text_encoder"), 8);
    let error = spec
        .validate_prepared_file_pins()
        .expect_err("a direct component cannot become a complete snapshot after preparation")
        .to_string();
    assert!(error.contains("source shape changed"), "{error}");
}

#[test]
fn prepared_receipt_rejects_higher_priority_tokenizer_candidate_addition() {
    const CANDIDATE_TOKENIZER: EncoderTokenizerContract = EncoderTokenizerContract {
        artifact_candidates: &["tokenizer/tokenizer.json", "processor/tokenizer.json"],
        ..TEST_TOKENIZER
    };
    const CANDIDATE_CONTRACT: EncoderContract = EncoderContract {
        tokenizer: CANDIDATE_TOKENIZER,
        ..CONTRACT
    };

    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    write_tokenizer_fixture(&base);
    std::fs::create_dir_all(base.join("processor")).unwrap();
    std::fs::rename(
        base.join("tokenizer/tokenizer.json"),
        base.join("processor/tokenizer.json"),
    )
    .unwrap();
    write_fixture(&selected, 8);
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CANDIDATE_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    write_tokenizer_fixture(&base);
    let error = spec
        .validate_prepared_file_pins()
        .expect_err("a new higher-priority tokenizer must not replace the retained artifact")
        .to_string();
    assert!(error.contains("tokenizer selection changed"), "{error}");
}

#[cfg(unix)]
#[test]
fn prepared_directory_receipt_rejects_shard_symlink_retarget() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let selected = temp.path().join("selected");
    std::fs::create_dir_all(&base).unwrap();
    write_tokenizer_fixture(&base);
    write_fixture(&selected, 8);
    let selected_shard = selected.join("model.safetensors");
    let target_a = temp.path().join("target-a.safetensors");
    let target_b = temp.path().join("target-b.safetensors");
    std::fs::rename(&selected_shard, &target_a).unwrap();
    std::fs::copy(&target_a, &target_b).unwrap();
    symlink(&target_a, &selected_shard).unwrap();
    let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
    let validated = CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
        .unwrap();
    validated.prepare_load_spec(&mut spec).unwrap();

    let staged = selected.join("staged.safetensors");
    symlink(&target_b, &staged).unwrap();
    std::fs::rename(staged, &selected_shard).unwrap();

    let opened = std::cell::Cell::new(false);
    let error = spec
        .read_prepared_files_unchanged::<(), Error>(|| {
            opened.set(true);
            Ok(())
        })
        .expect_err("retargeted shard symlink must fail before provider load")
        .to_string();
    assert!(!opened.get());
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn direct_component_inherits_and_pins_the_base_runtime_tokenizer() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let component = temp.path().join("alternate-component");
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(&base);
    write_fixture(&component, 8);
    let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
        .with_text_encoder(WeightsSource::Dir(component));
    let source = CONTRACT.source_for_load(&spec, &base).unwrap();

    assert_eq!(
        source.tokenizer_disposition(),
        Some(EncoderTokenizerDisposition::InheritedBase)
    );
    let runtime_path = source
        .read_tokenizer_unchanged::<PathBuf, Error>(|path| Ok(path.to_path_buf()))
        .unwrap();
    assert_eq!(
        runtime_path,
        std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap()
    );
}

#[test]
fn validated_inventory_and_pricing_ignore_nested_safetensors() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let direct = temp.path().join("model.safetensors");
    let direct_bytes = std::fs::metadata(&direct).unwrap().len();
    write_fixture(&temp.path().join("nested"), 8);

    let selected = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap();
    assert_eq!(selected.source_bytes().unwrap(), direct_bytes);
    assert_eq!(
        selected.tensor_headers().unwrap().len(),
        safetensors_path_tensor_headers(&direct).unwrap().len()
    );
}

#[test]
fn direct_safetensors_directory_is_rejected_as_a_non_file_shard() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    std::fs::create_dir(temp.path().join("bogus.safetensors")).unwrap();

    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a regular file"), "{error}");
    assert!(error.contains("bogus.safetensors"), "{error}");
}

#[cfg(unix)]
#[test]
fn broken_direct_safetensors_symlink_is_rejected_during_inventory() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    symlink(
        temp.path().join("missing-target"),
        temp.path().join("broken.safetensors"),
    )
    .unwrap();

    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot be inspected"), "{error}");
    assert!(error.contains("broken.safetensors"), "{error}");
}

#[test]
fn absent_override_rejects_a_doctored_builtin_config_and_names_the_field() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();
    let encoder = base.join("text_encoder");
    write_fixture(&encoder, 8);
    write_tokenizer_fixture(base);
    let config_path = encoder.join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("num_hidden_layers");
    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();

    let error = CONTRACT
        .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
        .unwrap_err()
        .to_string();
    assert!(error.contains("field num_hidden_layers"), "{error}");
    assert!(error.contains("expected 2, got missing"), "{error}");
}

#[test]
fn dense_selected_encoder_inherits_the_effective_quant_tier() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    assert_eq!(
        text_encoder_load_time_quant_bits(
            &WeightsSource::Dir(temp.path().to_path_buf()),
            Some(8),
            "fixture",
        )
        .unwrap(),
        Some(8)
    );
}

#[test]
fn matching_prepacked_selected_encoder_needs_no_requantization() {
    let temp = tempfile::tempdir().unwrap();
    write_packed_fixture(temp.path(), 8, true);
    assert_eq!(
        text_encoder_load_time_quant_bits(
            &WeightsSource::Dir(temp.path().to_path_buf()),
            Some(4),
            "fixture",
        )
        .unwrap(),
        None
    );
}

#[test]
fn prepacked_selected_encoder_rejects_a_different_effective_tier() {
    let temp = tempfile::tempdir().unwrap();
    write_packed_fixture(temp.path(), 8, true);
    let error = text_encoder_load_time_quant_bits(
        &WeightsSource::Dir(temp.path().to_path_buf()),
        Some(8),
        "fixture",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("pre-quantized Q4"), "{error}");
    assert!(error.contains("model policy is Q8"), "{error}");
}

#[test]
fn prepacked_selected_encoder_rejects_a_dense_model_policy() {
    let temp = tempfile::tempdir().unwrap();
    write_packed_fixture(temp.path(), 8, true);
    let error = text_encoder_load_time_quant_bits(
        &WeightsSource::Dir(temp.path().to_path_buf()),
        None,
        "fixture",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("pre-quantized Q4"), "{error}");
    assert!(error.contains("model policy is dense"), "{error}");
}

#[test]
fn dense_headers_reject_a_false_packed_config_marker() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    rewrite_config_quantization(temp.path(), 4);

    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("config declares Q4"), "{error}");
    assert!(error.contains("direct-shard surface is dense"), "{error}");
}

/// sc-22727: a Hugging Face cache snapshot's files are symlinks into the repository's sibling
/// `blobs/` tree, so the snapshot directory alone can never confine their canonical targets.
/// [`hf_cache_discovery_roots`] authorizes the `models--<org>--<repo>` directory for a path
/// below its `snapshots/<revision>`, and nothing above it.
#[cfg(unix)]
#[test]
fn hf_cache_discovery_roots_authorize_the_repository_blobs_and_nothing_above() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let hub = temp.path().join("hub");
    let repository = hub.join("models--org--repo");
    let blobs = repository.join("blobs");
    let snapshot = repository.join("snapshots/0123456789abcdef");
    let component = snapshot.join("q4/text_encoder");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&component).unwrap();
    let staged = temp.path().join("staged");
    write_fixture(&staged, 8);
    for (name, blob) in [("config.json", "aaaa"), ("model.safetensors", "bbbb")] {
        std::fs::copy(staged.join(name), blobs.join(blob)).unwrap();
        symlink(
            Path::new("../../../../blobs").join(blob),
            component.join(name),
        )
        .unwrap();
    }

    let roots = hf_cache_discovery_roots(&component).unwrap();
    assert!(roots.contains(&std::path::absolute(&component).unwrap()));
    assert!(roots.contains(&std::path::absolute(&repository).unwrap()));
    assert!(!roots.contains(&hub));
    // The snapshot-only roots the old callers computed still refuse the blobs target …
    let error = CONTRACT
        .validate_source_for_discovery(
            &WeightsSource::Dir(component.clone()),
            &authorized_test_roots(&component),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("escapes authorized model roots"), "{error}");
    // … and the repository-aware roots admit it.
    CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(component.clone()), &roots)
        .unwrap();

    // A symlink escaping the repository directory stays refused under the wider roots.
    let outside = temp.path().join("outside");
    write_fixture(&outside, 8);
    std::fs::remove_file(component.join("model.safetensors")).unwrap();
    symlink(
        outside.join("model.safetensors"),
        component.join("model.safetensors"),
    )
    .unwrap();
    let error = CONTRACT
        .validate_source_for_discovery(&WeightsSource::Dir(component.clone()), &roots)
        .unwrap_err()
        .to_string();
    assert!(error.contains("canonical target"), "{error}");
    assert!(error.contains("escapes authorized model roots"), "{error}");

    // Outside a cache snapshot the roots are just the path itself.
    let plain = hf_cache_discovery_roots(&staged).unwrap();
    assert_eq!(plain, authorized_test_roots(&staged));
    // A directory merely named `snapshots` above the path does not widen the roots either.
    let decoy = temp.path().join("models--x--y/snapshots");
    std::fs::create_dir_all(&decoy).unwrap();
    assert_eq!(
        hf_cache_discovery_roots(&decoy).unwrap(),
        authorized_test_roots(&decoy)
    );
    let bare_repository = temp.path().join("models--x--y");
    assert_eq!(
        hf_cache_discovery_roots(&bare_repository).unwrap(),
        authorized_test_roots(&bare_repository)
    );

    // A `..`-bearing path whose real location is a *different* repository must not authorize
    // the repository its lexical ancestors name. `std::path::absolute` keeps `..`, so the
    // repository has to be derived from the canonical path.
    let sibling = hub.join("models--evil--x/text_encoder");
    std::fs::create_dir_all(&sibling).unwrap();
    let traversal = snapshot.join("../../../models--evil--x/text_encoder");
    let traversal_roots = hf_cache_discovery_roots(&traversal).unwrap();
    assert!(
        !traversal_roots.contains(&std::path::absolute(&repository).unwrap()),
        "{traversal_roots:?}"
    );
    assert!(
        !traversal_roots
            .iter()
            .any(|root| std::fs::canonicalize(root).unwrap()
                == std::fs::canonicalize(&repository).unwrap()),
        "{traversal_roots:?}"
    );
    assert!(
        traversal_roots.contains(&std::fs::canonicalize(&sibling).unwrap()),
        "{traversal_roots:?}"
    );
}

#[test]
fn ignored_vision_packing_does_not_determine_the_language_tier() {
    let headers = vec![
        SafetensorsTensorHeader {
            name: "model.embed_tokens.weight".into(),
            dtype: Dtype::F16,
            shape: vec![16, 64],
            data_bytes: 2_048,
        },
        SafetensorsTensorHeader {
            name: "visual.proj.weight".into(),
            dtype: Dtype::U32,
            shape: vec![64, 1],
            data_bytes: 256,
        },
        SafetensorsTensorHeader {
            name: "visual.proj.scales".into(),
            dtype: Dtype::F16,
            shape: vec![64, 1],
            data_bytes: 128,
        },
        SafetensorsTensorHeader {
            name: "visual.proj.biases".into(),
            dtype: Dtype::F16,
            shape: vec![64],
            data_bytes: 128,
        },
    ];
    let language = language_quantization_evidence_headers(&headers);
    assert_eq!(
        validate_quantization_evidence(&language, Path::new("fixture"), None).unwrap(),
        None
    );
    let error = validate_quantization_evidence(
        &language,
        Path::new("fixture"),
        Some(PackedQuantization {
            bits: 4,
            group_size: 64,
        }),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("direct-shard surface is dense"), "{error}");
}

#[test]
fn ignored_decoder_tail_packing_does_not_determine_the_language_tier() {
    let headers = vec![
        SafetensorsTensorHeader {
            name: "model.embed_tokens.weight".into(),
            dtype: Dtype::F16,
            shape: vec![16, 64],
            data_bytes: 2_048,
        },
        SafetensorsTensorHeader {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: Dtype::F16,
            shape: vec![64, 64],
            data_bytes: 8_192,
        },
        SafetensorsTensorHeader {
            name: "model.layers.1.self_attn.q_proj.weight".into(),
            dtype: Dtype::U32,
            shape: vec![64, 8],
            data_bytes: 2_048,
        },
        SafetensorsTensorHeader {
            name: "model.layers.1.self_attn.q_proj.scales".into(),
            dtype: Dtype::F16,
            shape: vec![64, 1],
            data_bytes: 128,
        },
        SafetensorsTensorHeader {
            name: "model.layers.1.self_attn.q_proj.biases".into(),
            dtype: Dtype::F16,
            shape: vec![64, 1],
            data_bytes: 128,
        },
    ];
    let language = language_quantization_evidence_headers(&headers);
    assert_eq!(language.len(), 2);
    assert_eq!(
        validate_quantization_evidence(&language, Path::new("fixture"), None).unwrap(),
        None
    );
}

#[test]
fn ignored_decoder_tail_packed_components_are_not_validated() {
    let contract = EncoderContract {
        loaded_hidden_layers: 1,
        ..CONTRACT
    };
    let headers = vec![
        SafetensorsTensorHeader {
            name: "model.layers.1.self_attn.q_proj.weight".into(),
            dtype: Dtype::U32,
            shape: vec![64, usize::MAX],
            data_bytes: 0,
        },
        SafetensorsTensorHeader {
            name: "model.layers.1.mlp.up_proj.scales".into(),
            dtype: Dtype::F16,
            shape: vec![1],
            data_bytes: 2,
        },
    ];
    contract
        .validate_all_packed_triples(&headers, Path::new("fixture"), None, &BTreeSet::new())
        .unwrap();
}

#[test]
fn candle_krea_storage_probe_rejects_only_the_lossy_bf16_branch() {
    let header = |name: &str, dtype| SafetensorsTensorHeader {
        name: name.into(),
        dtype,
        shape: vec![8],
        data_bytes: 16,
    };
    let probe = CONTRACT.dense_storage_dtype_probe.unwrap();
    let mixed_f32_store = vec![
        header(probe, Dtype::F32),
        header("model.layers.0.self_attn.q_proj.weight", Dtype::BF16),
    ];
    CONTRACT
        .validate_dense_storage_probe(&mixed_f32_store, Path::new("fixture"))
        .unwrap();

    let lossy_bf16_store = vec![
        header(probe, Dtype::BF16),
        header("model.layers.0.self_attn.q_proj.weight", Dtype::F32),
    ];
    let error = CONTRACT
        .validate_dense_storage_probe(&lossy_bf16_store, Path::new("fixture"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("dense_storage_dtype"), "{error}");
    assert!(error.contains("q_proj.weight=F32"), "{error}");
}

#[test]
fn candle_krea_storage_probe_ignores_unloaded_decoder_tail() {
    let contract = EncoderContract {
        loaded_hidden_layers: 1,
        ..CONTRACT
    };
    let header = |name: &str, dtype| SafetensorsTensorHeader {
        name: name.into(),
        dtype,
        shape: vec![8],
        data_bytes: 16,
    };
    let probe = contract.dense_storage_dtype_probe.unwrap();
    contract
        .validate_dense_storage_probe(
            &[
                header(probe, Dtype::BF16),
                header("model.layers.0.self_attn.q_proj.weight", Dtype::BF16),
                header("model.layers.1.self_attn.q_proj.weight", Dtype::F32),
            ],
            Path::new("fixture"),
        )
        .unwrap();
}

#[test]
fn model_type_is_authoritative_over_a_stale_matching_architectures_alias() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["model_type"] = json!("mistral");
    config["architectures"] = json!(["Qwen3ForCausalLM"]);

    let error = CONTRACT
        .validate_config(&config, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("architecture.model_type"), "{error}");
    assert!(error.contains("mistral"), "{error}");
}

#[test]
fn every_architectures_alias_must_match_the_model_type_contract() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["architectures"] = json!(["Qwen3ForCausalLM", "MistralForCausalLM"]);

    let error = CONTRACT
        .validate_config(&config, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("architecture.architectures"), "{error}");
    assert!(error.contains("MistralForCausalLM"), "{error}");
}

#[test]
fn every_rope_theta_and_type_alias_must_match() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut theta: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    theta["rope_parameters"] = json!({"rope_theta": 500000.0});
    let error = CONTRACT
        .validate_config(&theta, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rope_theta"), "{error}");
    assert!(error.contains("rope_parameters.rope_theta"), "{error}");

    let mut rope_type: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    rope_type["rope_type"] = json!("default");
    rope_type["rope_scaling"] = json!({"type": "linear"});
    let error = CONTRACT
        .validate_config(&rope_type, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rope_type"), "{error}");
    assert!(error.contains("rope_scaling.type"), "{error}");
}

#[test]
fn every_mrope_section_and_interleaving_alias_must_match() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut section: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    section["rope_parameters"] = json!({"mrope_section": []});
    section["rope_scaling"] = json!({"mrope_section": [1]});
    let error = CONTRACT
        .validate_config(&section, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("mrope_section"), "{error}");
    assert!(error.contains("rope_scaling.mrope_section"), "{error}");

    let mut interleaving: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    interleaving["rope_parameters"] = json!({"mrope_interleaved": false});
    interleaving["rope_scaling"] = json!({"mrope_interleaved": true});
    let error = CONTRACT
        .validate_config(&interleaving, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("mrope_interleaved"), "{error}");
    assert!(error.contains("rope_scaling.mrope_interleaved"), "{error}");
}

#[test]
fn root_and_text_config_bool_and_token_id_duplicates_must_match() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut nested: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    nested["eos_token_id"] = json!(2);
    let bool_conflict = json!({
        "attention_bias": true,
        "text_config": nested,
    });
    let error = CONTRACT
        .validate_config(&bool_conflict, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("attention_bias"), "{error}");
    assert!(error.contains("root=true"), "{error}");

    let mut nested: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    nested["eos_token_id"] = json!(2);
    let token_conflict = json!({
        "eos_token_id": 3,
        "text_config": nested,
    });
    let error = CONTRACT
        .validate_config(&token_conflict, &path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("eos_token_id"), "{error}");
    assert!(error.contains("root=3"), "{error}");
}

#[test]
fn optional_behavior_bool_allows_omission_but_rejects_every_authored_conflict() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("attention_bias");
    let contract = EncoderContract {
        attention_bias: EncoderConfigBool::Optional(false),
        ..CONTRACT
    };
    contract
        .validate_config(&config, &path)
        .expect("omission selects the provider's fixed false runtime behavior");

    let mut root_conflict = config.clone();
    root_conflict["attention_bias"] = json!(true);
    let mut nested = config.clone();
    nested["attention_bias"] = json!(true);
    let nested_conflict = json!({ "text_config": nested });
    for authored in [root_conflict, nested_conflict] {
        let error = contract
            .validate_config(&authored, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("attention_bias"), "{error}");
        assert!(error.contains("expected false"), "{error}");
    }
}

#[test]
fn nested_text_architecture_is_authoritative_over_matching_root_wrapper() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut nested = config.clone();
    nested["model_type"] = json!("qwen2_5_vl_text");
    config["model_type"] = json!("qwen3");
    config["text_config"] = nested;
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();

    let error = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field architecture"), "{error}");
    assert!(error.contains("qwen2_5_vl_text"), "{error}");
}

#[test]
fn dense_file_without_sibling_config_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    std::fs::remove_file(temp.path().join("config.json")).unwrap();
    let file = temp.path().join("model.safetensors");

    let error = CONTRACT
        .validate_source(&WeightsSource::File(file))
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no config.json"), "{error}");
}

#[test]
fn file_override_never_borrows_the_builtin_config_as_architecture_proof() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join("base");
    let external = temp.path().join("external");
    write_fixture(&base.join("text_encoder"), 8);
    write_tokenizer_fixture(&base);
    write_fixture(&external, 8);
    std::fs::remove_file(external.join("config.json")).unwrap();
    let contract = EncoderContract {
        architecture: "qwen3_vl_text",
        ..CONTRACT
    };
    let base_config = base.join("text_encoder/config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&base_config).unwrap()).unwrap();
    config["model_type"] = json!("qwen3_vl_text");
    std::fs::write(&base_config, serde_json::to_vec(&config).unwrap()).unwrap();
    let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
        .with_text_encoder(WeightsSource::File(external.join("model.safetensors")));

    let error = contract
        .source_for_load(&spec, &base)
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no config.json"), "{error}");
    assert!(error.contains("cannot be proven"), "{error}");
}

fn write_packed_fixture(root: &Path, q_width: usize, include_q_biases: bool) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
            root.join("config.json"),
            br#"{"model_type":"qwen3","hidden_size":64,"intermediate_size":96,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":16,"hidden_act":"silu","attention_dropout":0.0,"rms_norm_eps":0.000001,"rope_theta":1000000,"max_position_embeddings":4096,"attention_bias":false,"tie_word_embeddings":true,"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
    let mut entries = vec![("model.embed_tokens.weight".to_owned(), "F16", vec![16, 64])];
    for layer in 0..2 {
        let base = format!("model.layers.{layer}");
        let packed = |name: String, output: usize, input: usize| {
            vec![
                (
                    format!("{name}.weight"),
                    "U32",
                    vec![output, input * 4 / 32],
                ),
                (format!("{name}.scales"), "F16", vec![output, input / 64]),
                (format!("{name}.biases"), "F16", vec![output, input / 64]),
            ]
        };
        let mut q = packed(format!("{base}.self_attn.q_proj"), 64, 64);
        if layer == 0 {
            q[0].2[1] = q_width;
            if !include_q_biases {
                q.retain(|(name, _, _)| !name.ends_with(".biases"));
            }
        }
        entries.extend(q);
        entries.extend(packed(format!("{base}.self_attn.k_proj"), 32, 64));
        entries.extend(packed(format!("{base}.self_attn.v_proj"), 32, 64));
        entries.extend(packed(format!("{base}.self_attn.o_proj"), 64, 64));
        entries.extend(packed(format!("{base}.mlp.gate_proj"), 96, 64));
        entries.extend(packed(format!("{base}.mlp.up_proj"), 96, 64));
        entries.extend(packed(format!("{base}.mlp.down_proj"), 64, 96));
        entries.extend([
            (format!("{base}.self_attn.q_norm.weight"), "F16", vec![32]),
            (format!("{base}.self_attn.k_norm.weight"), "F16", vec![32]),
            (format!("{base}.input_layernorm.weight"), "F16", vec![64]),
            (
                format!("{base}.post_attention_layernorm.weight"),
                "F16",
                vec![64],
            ),
        ]);
    }
    entries.push(("model.norm.weight".to_owned(), "F16", vec![64]));
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    for (name, dtype, shape) in entries {
        let element_bytes = if matches!(dtype, "F16" | "BF16") {
            2
        } else {
            4
        };
        let bytes = shape.iter().product::<usize>() * element_bytes;
        header.insert(
            name,
            json!({"dtype":dtype, "shape":shape, "data_offsets":[offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut file = Vec::with_capacity(8 + encoded.len() + offset);
    file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    file.extend_from_slice(&encoded);
    file.resize(8 + encoded.len() + offset, 0);
    std::fs::write(root.join("model.safetensors"), file).unwrap();
}

const PACKED_CONTRACT: EncoderContract = EncoderContract {
    architecture: "qwen3",
    hidden_size: 64,
    intermediate_size: 96,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 32,
    vocab_size: 16,
    output_width: 64,
    loaded_hidden_layers: 2,
    requires_final_norm: true,
    requires_lm_head: false,
    hidden_activation: "silu",
    attention_dropout: EncoderConfigFloat::new(0.0),
    rms_norm_eps: EncoderConfigFloat::new(1e-6),
    qk_norm_eps: Some(EncoderConfigFloat::new(1e-6)),
    rope_theta: EncoderConfigFloat::new(1_000_000.0),
    max_position_embeddings: 4_096,
    attention_bias: EncoderConfigBool::Required(false),
    tie_word_embeddings: EncoderConfigBool::Required(true),
    tokenizer: TEST_TOKENIZER,
    prompt_executions: TEST_PROMPTS,
    bos_token_id: None,
    eos_token_id: None,
    image_token_id: None,
    vision_start_token_id: None,
    vision_end_token_id: None,
    mrope_section: &[],
    mrope_interleaved: None,
    selected_hidden_layers: &[2],
    packing: Some(EncoderPackingContract {
        group_size: 64,
        pack_embedding: false,
        pack_lm_head: false,
        supports_file: true,
    }),
    dense_storage_dtype_probe: Some("model.layers.0.input_layernorm.weight"),
};

#[test]
fn packed_contract_rejects_corrupt_second_dimension_and_incomplete_triple() {
    let corrupt = tempfile::tempdir().unwrap();
    write_packed_fixture(corrupt.path(), 9, true);
    let error = PACKED_CONTRACT
        .validate_source(&WeightsSource::Dir(corrupt.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field packed_weight_shape"), "{error}");
    assert!(error.contains("expected [64, 8], got [64, 9]"), "{error}");

    let incomplete = tempfile::tempdir().unwrap();
    write_packed_fixture(incomplete.path(), 8, false);
    let error = PACKED_CONTRACT
        .validate_source(&WeightsSource::Dir(incomplete.path().to_path_buf()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("field packed_components"), "{error}");
}

#[test]
fn packed_file_without_sibling_metadata_is_unprovable_and_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_packed_fixture(temp.path(), 8, true);
    std::fs::remove_file(temp.path().join("config.json")).unwrap();
    let error = PACKED_CONTRACT
        .validate_source(&WeightsSource::File(temp.path().join("model.safetensors")))
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no config.json"), "{error}");
    assert!(
        error.contains("exact behavior, tokenizer, head-topology, and precision compatibility"),
        "{error}"
    );
}

#[test]
fn validated_file_rejects_replacement_before_payload_open() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let file = temp.path().join("model.safetensors");
    let validated = CONTRACT
        .validate_source(&WeightsSource::File(file.clone()))
        .unwrap();
    let replacement = temp.path().join("replacement.safetensors");
    std::fs::copy(&file, &replacement).unwrap();
    std::fs::rename(&replacement, &file).unwrap();

    let error = validated
        .read_unchanged::<(), Error>(|_| Ok(()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn validated_directory_rejects_shard_inventory_and_config_mutation() {
    let added = tempfile::tempdir().unwrap();
    write_fixture(added.path(), 8);
    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(added.path().to_path_buf()))
        .unwrap();
    std::fs::copy(
        added.path().join("model.safetensors"),
        added.path().join("model-00002-of-00002.safetensors"),
    )
    .unwrap();
    let error = validated
        .read_unchanged::<(), Error>(|_| Ok(()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("shard inventory changed"), "{error}");

    let config = tempfile::tempdir().unwrap();
    write_fixture(config.path(), 8);
    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(config.path().to_path_buf()))
        .unwrap();
    rewrite_config_field(config.path(), "vocab_size", 17);
    let error = validated
        .read_unchanged::<(), Error>(|_| Ok(()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn read_unchanged_brackets_backend_directory_enumeration() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap();
    let root = temp.path().to_path_buf();
    let error = validated
        .read_unchanged::<(), Error>(|_| {
            std::fs::copy(
                root.join("model.safetensors"),
                root.join("late-added.safetensors"),
            )
            .map_err(Error::from)?;
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("shard inventory changed"), "{error}");
}

#[test]
fn retained_quant_evidence_rejects_post_validation_config_mutation() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap();
    rewrite_config_quantization(temp.path(), 4);

    let error = validated
        .load_time_quant_bits(Some(4), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("pinned weights"), "{error}");
}

#[test]
fn selected_encoder_bytes_equal_the_direct_loader_shards_only() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path(), 8);
    let direct = temp.path().join("model.safetensors");
    let direct_bytes = std::fs::metadata(&direct).unwrap().len();
    let nested = temp.path().join("ignored/nested.safetensors");
    std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
    std::fs::copy(&direct, &nested).unwrap();

    let validated = CONTRACT
        .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
        .unwrap();
    assert_eq!(validated.source_bytes().unwrap(), direct_bytes);
    assert_eq!(
        text_encoder_source_bytes(&WeightsSource::Dir(temp.path().to_path_buf())).unwrap(),
        direct_bytes
    );
}
