use super::*;
use std::fs;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn model_bytes() -> Vec<u8> {
    super::super::checkpoint::tests::dense_stub_bytes(true)
}
fn config_bytes() -> Vec<u8> {
    let mut value = super::super::checkpoint::tests::real_config_json(
        true,
        super::super::quantization::DitFormat::Dense,
    );
    value["quantization"] = super::super::artifacts::NativeDitStorage::MlxAffine {
        bits: 4,
        group_size: 64,
    }
    .marker()
    .unwrap();
    serde_json::to_vec(&value).unwrap()
}
fn receipt() -> ConversionReceipt {
    ConversionReceipt {
        source_repository: super::super::DISTILLED_WEIGHTS_REPO.into(),
        source_revision: super::super::DISTILLED_WEIGHTS_REVISION.into(),
        source_artifacts: BTreeMap::from([("model.safetensors".into(), digest(b"source"))]),
        algorithm: "mlx-affine".into(),
        algorithm_version: 1,
        bits: Some(4),
        group_size: Some(64),
        components: vec![
            ComponentInventory {
                source_repository: super::super::DISTILLED_WEIGHTS_REPO.into(),
                source_revision: super::super::DISTILLED_WEIGHTS_REVISION.into(),
                source_artifacts: BTreeMap::from([("model.safetensors".into(), digest(b"source"))]),
                role: ComponentRole::Dit,
                placement: ComponentPlacement::Owned,
                representation: "mlx-affine-q4-g64/bf16-compute".into(),
                files: BTreeMap::from([(
                    "model.safetensors".into(),
                    OutputFile {
                        bytes: model_bytes().len() as u64,
                        sha256: digest(&model_bytes()),
                    },
                )]),
                quantized_targets: vec!["single_blocks.0.q_proj".into()],
                dense_targets: vec!["txt_in".into()],
            },
            ComponentInventory {
                source_repository: super::super::DISTILLED_WEIGHTS_REPO.into(),
                source_revision: super::super::DISTILLED_WEIGHTS_REVISION.into(),
                source_artifacts: BTreeMap::from([("model.safetensors".into(), digest(b"source"))]),
                role: ComponentRole::TextEncoder,
                placement: ComponentPlacement::Owned,
                representation: "bf16-language-tower".into(),
                files: BTreeMap::from([
                    (
                        "text_encoder/config.json".into(),
                        OutputFile {
                            bytes: 2,
                            sha256: digest(b"{}"),
                        },
                    ),
                    (
                        "text_encoder/model.safetensors".into(),
                        OutputFile {
                            bytes: 2,
                            sha256: digest(b"te"),
                        },
                    ),
                    (
                        "text_encoder/tokenizer_config.json".into(),
                        OutputFile {
                            bytes: 2,
                            sha256: digest(b"{}"),
                        },
                    ),
                    (
                        "text_encoder/tokenizer.json".into(),
                        OutputFile {
                            bytes: 2,
                            sha256: digest(b"{}"),
                        },
                    ),
                ]),
                quantized_targets: vec![],
                dense_targets: vec![],
            },
            ComponentInventory {
                source_repository: super::super::DISTILLED_WEIGHTS_REPO.into(),
                source_revision: super::super::DISTILLED_WEIGHTS_REVISION.into(),
                source_artifacts: BTreeMap::from([("model.safetensors".into(), digest(b"source"))]),
                role: ComponentRole::Autoencoder,
                placement: ComponentPlacement::Owned,
                representation: "f32".into(),
                files: BTreeMap::from([(
                    "ae.safetensors".into(),
                    OutputFile {
                        bytes: 2,
                        sha256: digest(b"ae"),
                    },
                )]),
                quantized_targets: vec![],
                dense_targets: vec![],
            },
        ],
        package_files: ["config.json", "LICENSE.md", "NOTICE.md"]
            .into_iter()
            .map(|p| {
                let bytes = if p == "config.json" {
                    config_bytes()
                } else {
                    b"{}".to_vec()
                };
                (
                    p.into(),
                    OutputFile {
                        bytes: bytes.len() as u64,
                        sha256: digest(&bytes),
                    },
                )
            })
            .collect(),
    }
}
fn populate(dir: &Path) -> Result<()> {
    fs::create_dir(dir.join("text_encoder"))?;
    for (p, bytes) in [
        ("model.safetensors", model_bytes()),
        ("ae.safetensors", b"ae".to_vec()),
        ("config.json", config_bytes()),
        ("LICENSE.md", b"{}".to_vec()),
        ("NOTICE.md", b"{}".to_vec()),
        ("text_encoder/config.json", b"{}".to_vec()),
        ("text_encoder/model.safetensors", b"te".to_vec()),
        ("text_encoder/tokenizer.json", b"{}".to_vec()),
        ("text_encoder/tokenizer_config.json", b"{}".to_vec()),
    ] {
        fs::write(dir.join(p), bytes)?;
    }
    Ok(())
}

#[test]
fn receipt_round_trip_keeps_every_component_and_algorithm_parameter() {
    let r = receipt();
    let parsed = ConversionReceipt::from_json(&r.to_json().unwrap()).unwrap();
    assert_eq!(r, parsed);
    let mut bad = r.to_json().unwrap();
    bad["group_size"] = serde_json::json!(17);
    assert!(ConversionReceipt::from_json(&bad).is_err());
    let mut bad = r.clone();
    bad.source_revision = super::super::BASE_WEIGHTS_REVISION.into();
    assert!(bad.validate().is_err());
    let mut bad = r.clone();
    bad.components.pop();
    assert!(bad.validate().is_err());
    let mut bad = r.clone();
    bad.components[0]
        .dense_targets
        .push("single_blocks.0.q_proj".into());
    assert!(bad.validate().is_err());
    let mut bad = r.clone();
    bad.components[0].files.insert(
        "../foreign".into(),
        OutputFile {
            bytes: 1,
            sha256: digest(b"x"),
        },
    );
    assert!(bad.validate().is_err());
}

#[test]
fn atomic_owned_output_does_not_publish_partial_or_corrupt_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("converted");
    let fail = publish_conversion(&target, |dir| {
        populate(dir)?;
        Err(Error::Msg("injected cancellation".into()))
    });
    assert!(fail.is_err());
    assert!(!target.exists());
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
    let bad = publish_conversion(&target, |dir| {
        populate(dir)?;
        fs::write(dir.join("model.safetensors"), b"broken")?;
        Ok(receipt())
    });
    assert!(bad.is_err());
    assert!(!target.exists());
    publish_conversion(&target, |dir| {
        populate(dir)?;
        Ok(receipt())
    })
    .unwrap();
    let r = open_conversion(&target).unwrap();
    assert_eq!(r, receipt());
    assert!(
        publish_conversion(&target, |_| panic!("existing output must not be replaced")).is_err()
    );
    fs::write(target.join("ae.safetensors"), b"xx").unwrap();
    assert!(open_conversion(&target).is_err());
}

#[test]
fn unlisted_or_duplicate_output_and_foreign_lock_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("converted");
    let bad = publish_conversion(&target, |dir| {
        populate(dir)?;
        fs::write(dir.join("extra"), b"x")?;
        Ok(receipt())
    });
    assert!(bad.is_err());
    assert!(!target.exists());
    let lock = temp.path().join(".converted.flux3-conversion.lock");
    fs::write(&lock, b"pid 4242").unwrap();
    let refusal = publish_conversion(&target, |_| panic!("foreign lock must be respected"))
        .unwrap_err()
        .to_string();
    // A stale lock is never deleted for the caller; the refusal gives the exact recovery.
    assert!(refusal.contains("pid 4242"), "{refusal}");
    assert!(refusal.contains("delete"), "{refusal}");
    assert!(
        refusal.contains(".converted.flux3-conversion-*"),
        "{refusal}"
    );
    assert_eq!(fs::read(&lock).unwrap(), b"pid 4242");
    fs::remove_file(&lock).unwrap();
    assert_eq!(
        publish_conversion(&target, |dir| {
            populate(dir)?;
            Ok(receipt())
        })
        .unwrap(),
        PublishStatus::Durable,
        "after the documented recovery the publish succeeds"
    );
    assert!(
        !temp
            .path()
            .join(".converted.flux3-conversion.lock")
            .exists(),
        "a finished conversion releases its own lock"
    );
    let mut r = receipt();
    r.package_files.insert(
        "ae.safetensors".into(),
        OutputFile {
            bytes: 2,
            sha256: digest(b"ae"),
        },
    );
    assert!(r.validate().is_err());
}

#[test]
fn torn_shard_index_never_publishes_ready() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("converted");
    let result = publish_conversion(&target, |dir| {
        populate(dir)?;
        fs::remove_file(dir.join("text_encoder/model.safetensors"))?;
        let bytes = br#"{"weight_map":{"model.language_model.weight":"missing.safetensors"}}"#;
        fs::write(dir.join("text_encoder/model.safetensors.index.json"), bytes)?;
        let mut r = receipt();
        let encoder = r
            .components
            .iter_mut()
            .find(|c| c.role == ComponentRole::TextEncoder)
            .unwrap();
        encoder.files.remove("text_encoder/model.safetensors");
        encoder.files.insert(
            "text_encoder/model.safetensors.index.json".into(),
            OutputFile {
                bytes: bytes.len() as u64,
                sha256: digest(bytes),
            },
        );
        Ok(r)
    });
    assert!(result.is_err());
    assert!(!target.exists());
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn ggml_algorithms_require_their_exact_block_parameters() {
    for (algorithm, bits) in [("ggml-q4_1", 4), ("ggml-q8_0", 8)] {
        let mut r = receipt();
        r.algorithm = algorithm.into();
        r.bits = Some(bits);
        r.group_size = Some(32);
        r.validate().unwrap();
        r.group_size = Some(64);
        assert!(
            r.validate().is_err(),
            "{algorithm}: GGML blocks are 32 wide"
        );
        r.group_size = Some(32);
        r.bits = Some(12 - bits);
        assert!(r.validate().is_err(), "{algorithm}: bits never alias");
    }
}

/// External encoder/AE files stay in the original root: they are excluded from the owned output
/// inventory, full-hash verified against the source root, and a DiT is never external.
///
/// *Mutations that red this:* counting external files in `verify_outputs`; skipping the
/// checksum in `verify_external`; allowing an external DiT.
#[test]
fn external_components_are_verified_in_the_source_root_not_the_output() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    fs::create_dir(&source).unwrap();
    populate(&source).unwrap();
    let mut r = receipt();
    for component in &mut r.components {
        if component.role != ComponentRole::Dit {
            component.placement = ComponentPlacement::External;
        }
    }
    assert!(r.has_external_components());
    assert_eq!(
        ConversionReceipt::from_json(&r.to_json().unwrap()).unwrap(),
        r
    );
    let target = temp.path().join("converted");
    let ext = r.clone();
    publish_conversion(&target, move |dir| {
        for (p, bytes) in [
            ("model.safetensors", model_bytes()),
            ("config.json", config_bytes()),
            ("LICENSE.md", b"{}".to_vec()),
            ("NOTICE.md", b"{}".to_vec()),
        ] {
            fs::write(dir.join(p), bytes)?;
        }
        Ok(ext)
    })
    .unwrap();
    assert!(!target.join("ae.safetensors").exists(), "nothing copied");
    assert_eq!(open_conversion(&target).unwrap(), r);
    r.verify_external(&source).unwrap();
    fs::write(source.join("ae.safetensors"), b"zz").unwrap();
    assert!(
        r.verify_external(&source).is_err(),
        "a changed source AE is refused"
    );
    let mut bad = r.clone();
    bad.components[0].placement = ComponentPlacement::External;
    assert!(bad.validate().is_err());
    let mut json = r.to_json().unwrap();
    json["components"][0]["placement"] = serde_json::json!("copied");
    assert!(ConversionReceipt::from_json(&json).is_err());
}

fn hashed_bytes() -> u64 {
    HASHED_BYTES.with(std::cell::Cell::get)
}

fn reset_hashed_bytes() {
    HASHED_BYTES.with(|hashed| hashed.set(0));
}

/// Publication full-hashes the owned outputs once and records a produced identity for each; every
/// later open reads ZERO bytes of them, and a same-size rewrite of an owned output is still
/// refused (its identity no longer matches, so it is hashed and the checksum disagrees).
///
/// *Mutations that red this:* trusting a recorded identity without `matches` (the rewrite opens);
/// not recording identities at publish (the reopen hashes every byte).
#[test]
fn a_published_conversion_reopens_without_rehashing_its_outputs() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("converted");
    publish_conversion(&target, |dir| {
        populate(dir)?;
        Ok(receipt())
    })
    .unwrap();
    let value: Value =
        serde_json::from_slice(&fs::read(target.join(RECEIPT_FILE)).unwrap()).unwrap();
    let identities = produced_identities(&value);
    let owned: BTreeSet<String> = receipt().files().map(|(p, _)| p.clone()).collect();
    assert_eq!(identities.keys().cloned().collect::<BTreeSet<_>>(), owned);
    assert!(identities
        .values()
        .all(ProducedFileIdentity::is_trustworthy));

    for _ in 0..2 {
        reset_hashed_bytes();
        assert_eq!(open_conversion(&target).unwrap(), receipt());
        assert_eq!(hashed_bytes(), 0, "a sealed open re-hashed an owned output");
    }

    let model = target.join("model.safetensors");
    let mut bytes = fs::read(&model).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&model, &bytes).unwrap();
    let refusal = open_conversion(&target).unwrap_err().to_string();
    assert!(
        refusal.contains("output size or checksum mismatch"),
        "{refusal}"
    );
    assert!(hashed_bytes() > 0, "the changed output was not hashed");
}

/// A receipt written before produced identities existed opens exactly as before: every owned
/// output full-hashed, a changed one refused.
#[test]
fn a_receipt_without_identities_still_verifies_by_hash() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("converted");
    publish_conversion(&target, |dir| {
        populate(dir)?;
        Ok(receipt())
    })
    .unwrap();
    let path = target.join(RECEIPT_FILE);
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert!(value
        .as_object_mut()
        .unwrap()
        .remove(PRODUCED_IDENTITIES)
        .is_some());
    assert_eq!(
        value,
        receipt().to_json().unwrap(),
        "the pre-identity receipt"
    );
    fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let total: u64 = receipt().files().map(|(_, f)| f.bytes).sum();
    reset_hashed_bytes();
    assert_eq!(open_conversion(&target).unwrap(), receipt());
    assert_eq!(hashed_bytes(), total, "every owned output is hashed");
    fs::write(target.join("NOTICE.md"), b"[]").unwrap();
    assert!(open_conversion(&target).is_err());
}

/// External source files are checked through the artifact seal, never the raw per-load hash, and a
/// changed source file is still refused.
#[test]
fn external_files_are_checked_through_the_artifact_seal() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    fs::create_dir(&source).unwrap();
    populate(&source).unwrap();
    let mut r = receipt();
    for component in &mut r.components {
        if component.role != ComponentRole::Dit {
            component.placement = ComponentPlacement::External;
        }
    }
    reset_hashed_bytes();
    r.verify_external(&source).unwrap();
    r.verify_external(&source).unwrap();
    assert_eq!(hashed_bytes(), 0, "an external file took the raw hash");
    // A different size, so the refusal never depends on the file-time tick a same-size rewrite
    // lands in (the seal's memo key is the file's stat identity).
    fs::write(source.join("text_encoder/model.safetensors"), b"tef").unwrap();
    let refusal = r.verify_external(&source).unwrap_err().to_string();
    assert!(
        refusal.contains("does not match the recorded source checksum"),
        "{refusal}"
    );
}
