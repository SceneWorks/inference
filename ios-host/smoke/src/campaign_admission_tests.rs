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
        candidate: None,
        evidence_key: None,
        permit: None,
        refused_quantum: None,
        refusal_only: false,
    }
}
fn tokenizer(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut vocab = String::from("{");
    for (i, c) in ('a'..='z').enumerate() {
        vocab += &format!("\"{c}\":{},\"{c}</w>\":{},", 2 * i, 2 * i + 1);
    }
    vocab += "\"<|startoftext|>\":49406,\"<|endoftext|>\":49407}";
    std::fs::write(dir.join("vocab.json"), vocab).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
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
    // SANA image-to-image is admitted-reachable: refused for EVIDENCE, not for its mode.
    for (mode, rung, kind) in [
        ("text", 2, "no_evidence"),
        ("reference", 2, "no_evidence"),
        ("control", 2, "mode"),
        ("text", 4, "strategy"),
    ] {
        let case = Case::parse(&format!("sana_1600m/sana_1600m/bf16/512/{mode}/{rung}")).unwrap();
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{result:?}");
        assert_eq!(p.refusal_kind, Some(kind), "{mode}/{rung}: {result:?}");
        if kind == "no_evidence" {
            assert!(p.artifact["layout_sha256"]
                .as_str()
                .is_some_and(|s| s.len() == 64));
            assert!(p.evidence_key.is_some(), "the memory key is recorded");
        }
    }
    // A dense snapshot requested as q4 is a typed ARTIFACT refusal (the provider publishes no
    // calibration for a load-time requantization), never an untyped failure or a wrong price.
    copy_tree(&root, &temp.path().join("sana_1600m/q4"));
    let case = Case::parse("sana_1600m/sana_1600m/q4/512/text/2").unwrap();
    let mut p = adapter(catalog.media(), temp.path().into());
    let result = campaign::run(&mut p, &case, false);
    assert_eq!(result.0, Status::AdmittedRefused, "{result:?}");
    assert_eq!(p.refusal_kind, Some("artifact"), "{result:?}");
    assert!(
        p.evidence_key.is_none(),
        "no key/price is computed for the wrong tier"
    );
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
    tokenizer(&sdxl.join("tokenizer"));
    tokenizer(&sdxl.join("tokenizer_2"));
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
    // Iris prompt batches are admitted-reachable too; a mode Iris does not serve is a typed
    // refusal, never a silent text-to-image substitute.
    for (mode, kind) in [("prompt_batch", "no_evidence"), ("reference", "mode")] {
        let case = Case::parse(&format!("iris_3b/iris_3b/bf16/512/{mode}/0")).unwrap();
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{mode}: {result:?}");
        assert_eq!(p.refusal_kind, Some(kind), "{mode}: {result:?}");
    }
    let case = Case::parse("iris_3b/iris_3b/bf16/512/prompt_batch/0").unwrap();
    let mut p = adapter(catalog.media(), temp.path().into());
    p.prepare(&case).unwrap();
    let runtime_ios::media::IosWork::Generate(request) = &p.candidate.as_ref().unwrap().work else {
        panic!("prompt batch is a generation")
    };
    assert_eq!(request.prompt_batch.len(), 2);
    assert_ne!(request.prompt_batch[0], request.prompt_batch[1]);
    assert_eq!(p.candidate.as_ref().unwrap().context.geometry.batch, 2);
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
    use support::{PeakReading, TASK_VM_INFO_COUNT, TASK_VM_INFO_REV3_COUNT};
    // The SDK layout: phys_footprint at words 36-37, ledger_phys_footprint_peak at 42-43, REV2 = 42
    // words, REV3 = 84, the full REV7 struct = 93.
    assert_eq!((TASK_VM_INFO_REV3_COUNT, TASK_VM_INFO_COUNT), (84, 93));
    let fill = |words: &mut [u32]| {
        words[36] = 123;
        words[37] = 1;
        words[40] = 999;
        words[42] = 456;
        words[43] = 2;
    };
    // REV3+ reply (the kernel returns its own count): the exact lifetime peak.
    for returned in [TASK_VM_INFO_REV3_COUNT, TASK_VM_INFO_COUNT] {
        let (current, peak) = support::read_footprint(|words, count| {
            assert_eq!(*count, TASK_VM_INFO_COUNT, "request the full struct");
            assert_eq!(words.len(), TASK_VM_INFO_COUNT as usize);
            fill(words);
            *count = returned;
            0
        })
        .unwrap();
        assert_eq!(current, (1u64 << 32) + 123);
        assert_eq!(peak, PeakReading::Read((2u64 << 32) + 456));
    }
    // A REV2-sized reply (42 words) carries no peak: an explicit typed reading, not a silent None.
    let (current, peak) = support::read_footprint(|words, count| {
        fill(words);
        *count = 42;
        0
    })
    .unwrap();
    assert_eq!(current, (1u64 << 32) + 123);
    assert_eq!(peak, PeakReading::Unavailable { returned_words: 42 });
    assert!(support::read_footprint(|_, count| {
        *count = 30;
        0
    })
    .is_none());
    assert!(support::read_footprint(|_, _| -1).is_none());
}
/// The mirrored layout must equal the installed SDK's `struct task_vm_info` (macOS hosts with
/// Xcode; skipped elsewhere): same field order and widths under `#pragma pack(4)`.
#[cfg(target_os = "macos")]
#[test]
fn task_vm_info_layout_matches_the_sdk_header() {
    let sdk = std::process::Command::new("xcrun")
        .args(["--sdk", "iphoneos", "--show-sdk-path"])
        .output();
    let Some(sdk) = sdk
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    else {
        eprintln!("no iPhoneOS SDK; layout cross-check skipped");
        return;
    };
    let header = std::fs::read_to_string(format!("{sdk}/usr/include/mach/task_info.h")).unwrap();
    assert!(header.contains("#pragma pack(push, 4)"));
    let body = header
        .split("struct task_vm_info {")
        .nth(1)
        .unwrap()
        .split("};")
        .next()
        .unwrap();
    let mut words = 0usize;
    let mut offsets = std::collections::HashMap::new();
    for line in body.lines() {
        let line = line.split("/*").next().unwrap().trim();
        let Some(decl) = line.strip_suffix(';') else {
            continue;
        };
        let mut parts = decl.split_whitespace();
        let ty = parts.next().unwrap();
        let name = parts.last().unwrap().to_string();
        offsets.insert(name, words);
        words += match ty {
            "integer_t" => 1,
            "mach_vm_size_t" | "mach_vm_address_t" | "int64_t" | "uint64_t" => 2,
            other => panic!("unmapped task_vm_info field type {other}"),
        };
    }
    assert_eq!(words as u32, support::TASK_VM_INFO_COUNT);
    assert_eq!(offsets["phys_footprint"], 36);
    assert_eq!(offsets["ledger_phys_footprint_peak"], 42);
    assert_eq!(
        offsets["limit_bytes_remaining"] as u32,
        support::TASK_VM_INFO_REV3_COUNT
    );
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
    support::persist_images(
        root.path(),
        &case,
        &[image.clone(), image.clone()],
        Some(256),
    )
    .unwrap();
    case.tier = "q8".into();
    support::persist_images(root.path(), &case, std::slice::from_ref(&image), None).unwrap();
    assert!(support::persist_images(root.path(), &case, &[image], Some(512)).is_err());
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

/// Finding 6: refusal kinds come TYPED from runtime-ios. An SDXL capacity refusal whose message
/// reads "...candidate 0: unsupported: <reason>" is recorded as capacity, not a diagnostic failure,
/// and no message text decides the kind.
#[test]
fn campaign_typed_refusals_classify_sdxl_capacity_without_parsing_messages() {
    use runtime_ios::media::{AdmissionError as Typed, Refusal, RefusalKind};
    for (kind, label) in [
        (RefusalKind::Capacity, "capacity"),
        (RefusalKind::NoEvidence, "no_evidence"),
        (RefusalKind::Mode, "mode"),
        (RefusalKind::Strategy, "strategy"),
        (RefusalKind::Artifact, "artifact"),
    ] {
        let mut recorded = None;
        // Deliberately misleading text: only the typed kind may decide the classification.
        let error = Typed::Refused(Refusal {
            kind,
            reason: "iOS admission on iPhone17,2: no candidate admitted; candidate 0: \
                     unsupported: no applicable artifact/device/request capture"
                .into(),
        });
        assert!(matches!(
            support::classify(&mut recorded, error),
            AdmissionError::Refused(_)
        ));
        assert_eq!(recorded, Some(label));
    }
    let mut recorded = None;
    let failed = Typed::Failed(runtime_ios::gen_core::Error::Unsupported(
        "iOS admission: estimated footprint 9999 MiB exceeds the cap".into(),
    ));
    assert!(matches!(
        support::classify(&mut recorded, failed),
        AdmissionError::Failure(_)
    ));
    assert_eq!(
        recorded, None,
        "a failure is never a refusal, whatever its text"
    );
}

/// Finding 1: the diagnostic flag selects the bounded CAPTURE permit, which never consults the
/// evidence table. Off-device it is a typed failure (no bounded quanta), never a pass and never an
/// admission.
#[cfg(not(target_os = "ios"))]
#[test]
fn campaign_capture_requires_bounded_quanta_and_skips_evidence() {
    let catalog = runtime_ios::catalog().unwrap();
    let temp = tempfile::tempdir().unwrap();
    sana(&temp.path().join("sana_1600m/bf16"));
    let case = Case::parse("sana_1600m/sana_1600m/bf16/512/text/2").unwrap();
    let mut p = adapter(catalog.media(), temp.path().into());
    let (status, reason) = campaign::run(&mut p, &case, true);
    assert_eq!(status, Status::DiagnosticFailure, "{reason}");
    assert!(reason.contains("bounded per-quantum"), "{reason}");
    assert_eq!(p.refusal_kind, None);
    assert!(p.permit.is_none());
}
