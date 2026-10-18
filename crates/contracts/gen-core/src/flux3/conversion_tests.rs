use super::*;
use std::fs;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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
                        bytes: 6,
                        sha256: digest(b"packed"),
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
                (
                    p.into(),
                    OutputFile {
                        bytes: 2,
                        sha256: digest(b"{}"),
                    },
                )
            })
            .collect(),
    }
}
fn populate(dir: &Path) -> Result<()> {
    fs::create_dir(dir.join("text_encoder"))?;
    for (p, bytes) in [
        ("model.safetensors", b"packed".as_slice()),
        ("ae.safetensors", b"ae".as_slice()),
        ("config.json", b"{}".as_slice()),
        ("LICENSE.md", b"{}".as_slice()),
        ("NOTICE.md", b"{}".as_slice()),
        ("text_encoder/config.json", b"{}".as_slice()),
        ("text_encoder/model.safetensors", b"te".as_slice()),
        ("text_encoder/tokenizer.json", b"{}".as_slice()),
        ("text_encoder/tokenizer_config.json", b"{}".as_slice()),
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
            ("model.safetensors", b"packed".as_slice()),
            ("config.json", b"{}".as_slice()),
            ("LICENSE.md", b"{}".as_slice()),
            ("NOTICE.md", b"{}".as_slice()),
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
