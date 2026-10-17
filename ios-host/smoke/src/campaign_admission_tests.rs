use super::*;

fn adapter(registry: &ProviderRegistry, root: PathBuf) -> DeviceProvider<'_> {
    DeviceProvider {
        registry,
        root,
        device_allowed: true,
        artifact: Value::Null,
        refusal_kind: None,
        cancel: CancelFlag::new(),
        deadline: Instant::now() + Duration::from_secs(30),
        spec: None,
        context: None,
        request: None,
        refusal_only: false,
        sdxl: None,
    }
}
fn tensor(path: &Path, key: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let header =
        format!("{{\"{key}\":{{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}}}");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.resize(bytes.len() + 256, 0);
    std::fs::write(path, bytes).unwrap();
}
fn sana(root: &Path) {
    tensor(&root.join("transformer/a.safetensors"), "proj.weight");
    tensor(
        &root.join("text_encoder/gemma-2-2b-it.safetensors"),
        "model.embed_tokens.weight",
    );
    tensor(&root.join("vae/a.safetensors"), "conv.weight");
}
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}
#[test]
fn campaign_injected_device_sana_profiles_errors_rungs_and_modes() {
    let catalog = runtime_ios::catalog().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("sana_1600m/bf16");
    sana(&root);
    for (mode, rung, kind) in [
        ("text", 2, "no_evidence"),
        ("reference", 2, "mode"),
        ("text", 4, "strategy"),
    ] {
        let case = Case::parse(&format!("sana_1600m/sana_1600m/bf16/512/{mode}/{rung}")).unwrap();
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{result:?}");
        assert_eq!(p.refusal_kind, Some(kind));
        assert!(p.artifact["layout_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64));
    }
    std::fs::write(
        root.join("text_encoder/gemma-2-2b-it.safetensors"),
        b"malformed",
    )
    .unwrap();
    let mut p = adapter(catalog.media(), temp.path().into());
    let case = Case::parse("sana_1600m/sana_1600m/bf16/512/text/2").unwrap();
    assert_eq!(
        campaign::run(&mut p, &case, false).0,
        Status::DiagnosticFailure
    );
    assert_eq!(p.refusal_kind, None);
}
#[test]
fn campaign_injected_device_sdxl_anima_iris_no_evidence_is_not_capacity() {
    let catalog = runtime_ios::catalog().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let sdxl = temp.path().join("sdxl/bf16");
    for component in ["unet", "vae", "text_encoder", "text_encoder_2"] {
        tensor(&sdxl.join(component).join("model.safetensors"), "x");
    }
    let anima = temp.path().join("anima_turbo/bf16");
    for component in ["diffusion_models", "text_encoders", "vae"] {
        tensor(
            &anima
                .join("split_files")
                .join(component)
                .join("fixture.safetensors"),
            "x",
        );
    }
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/media/mlx-gen/mlx-gen-iris/tests/fixtures");
    copy_tree(
        &fixtures.join("tiny-snapshot/iris"),
        &temp.path().join("iris_3b/bf16"),
    );
    copy_tree(
        &fixtures.join("tiny-snapshot/text_encoder"),
        &temp.path().join("iris_3b/bf16/text_encoder"),
    );
    std::fs::write(
        temp.path().join("iris_3b/bf16/components.json"),
        r#"{"text_encoder":"text_encoder"}"#,
    )
    .unwrap();
    for provider in ["sdxl", "anima_turbo", "iris_3b"] {
        let case = Case::parse(&format!("{provider}/{provider}/bf16/512/text/0")).unwrap();
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{provider}: {result:?}");
        assert_eq!(
            p.refusal_kind,
            Some("no_evidence"),
            "{provider}: {result:?}"
        );
    }
    let case = Case::parse("iris_3b/iris_3b/bf16/512/prompt_batch/0").unwrap();
    let mut p = adapter(catalog.media(), temp.path().into());
    p.prepare(&case).unwrap();
    assert_eq!(p.request.as_ref().unwrap().prompt_batch.len(), 2);
    assert_ne!(
        p.request.as_ref().unwrap().prompt_batch[0],
        p.request.as_ref().unwrap().prompt_batch[1]
    );
}
#[test]
fn campaign_oversized_component_manifest_fails_without_full_allocation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("anima_turbo/q4");
    std::fs::create_dir_all(&root).unwrap();
    let file = std::fs::File::create(root.join("components.json")).unwrap();
    file.set_len(1 << 30).unwrap();
    let case = Case::parse("anima_turbo/anima_turbo/q4/512/text/0").unwrap();
    assert!(snapshot(temp.path(), &case).unwrap_err().contains("64 KiB"));
}
#[test]
fn campaign_kernel_ledger_uses_exact_peak_not_sampled_current() {
    let (current, peak) = support::read_footprint(|words, count| {
        assert_eq!(*count, 44);
        words[36] = 123;
        words[37] = 1;
        words[40] = 999;
        words[42] = 456;
        words[43] = 2;
        0
    })
    .unwrap();
    assert_eq!(current, (1u64 << 32) + 123);
    assert_eq!(peak, Some((2u64 << 32) + 456));
    let (_, peak) = support::read_footprint(|words, count| {
        words[36] = 321;
        *count = 38;
        0
    })
    .unwrap();
    assert_eq!(peak, None);
    assert!(support::read_footprint(|_, _| -1).is_none());
}
#[test]
fn campaign_sdxl_malformed_shard_is_failure_before_missing_evidence() {
    let catalog = runtime_ios::catalog().unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("sdxl/bf16/unet/model.safetensors");
    tensor(&path, "x");
    std::fs::write(path, b"invalid").unwrap();
    let mut p = adapter(catalog.media(), root.path().into());
    let case = Case::parse("sdxl/sdxl/bf16/512/text/2").unwrap();
    assert_eq!(
        campaign::run(&mut p, &case, false).0,
        Status::DiagnosticFailure
    );
    assert_eq!(p.refusal_kind, None);
}
#[test]
fn campaign_case_renders_are_unique_and_registered_in_manifest() {
    let root = tempfile::tempdir().unwrap();
    let mut case = Case::parse("iris_3b/iris_3b/bf16/256/prompt_batch/0").unwrap();
    let image = Image {
        width: 256,
        height: 256,
        pixels: (0..256 * 256 * 3).map(|i| (i % 251) as u8).collect(),
    };
    support::persist_images(root.path(), &case, &[image.clone(), image.clone()]).unwrap();
    case.tier = "q8".into();
    support::persist_images(root.path(), &case, &[image]).unwrap();
    let manifest = std::fs::read_to_string(root.path().join(crate::RENDERS_MANIFEST)).unwrap();
    let names: Vec<_> = manifest.lines().collect();
    assert_eq!(names.len(), 3);
    assert_eq!(
        names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3
    );
    for name in names {
        assert!(root.path().join(name).is_file());
        assert!(name.starts_with("campaign-iris_3b-iris_3b-"));
    }
}
#[test]
fn campaign_deadline_cooperatively_cancels_without_signals() {
    let cancel = CancelFlag::new();
    let mut deadline = Deadline::start(cancel.clone(), Duration::ZERO);
    deadline.thread.take().unwrap().join().unwrap();
    assert!(cancel.is_cancelled());
}
