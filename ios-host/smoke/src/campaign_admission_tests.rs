use super::*;

/// The cell of a `provider/route/tier/edge/mode/rung` prefix at its provider's FIRST
/// (quality-first) selection for that tier and rung (else the parameterless one), running the
/// anchors' short step count. Any configuration of the catalog's domain, anchor or derived.
pub(super) fn cell(prefix: &str) -> Case {
    static SURFACES: std::sync::OnceLock<Vec<Surface>> = std::sync::OnceLock::new();
    let surfaces =
        SURFACES.get_or_init(|| surfaces(runtime_ios::catalog().unwrap().media()).unwrap());
    let mut c = Case::parse(&format!(
        "{prefix}/{}/{}",
        campaign::BASE_SELECTION,
        campaign::ANCHOR_STEPS
    ))
    .unwrap();
    if let Some(s) = surfaces.iter().find(|s| s.provider == c.provider) {
        c.selection = s.first_selection(&c.route, &c.tier, c.rung);
    }
    c
}
pub(super) fn adapter(registry: &ProviderRegistry, root: PathBuf) -> DeviceProvider<'_> {
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
        admitted: None,
        telemetry: None,
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

// These tests exercise metadata admission/identity, never a render or model
// load. Supply valid small tensor dictionaries over the COMPLETE constructor
// namespaces and affine group-64 packing; do not substitute an arbitrary `x`
// tensor for a base projection which the real overlay normalizer must inspect.
#[derive(Default)]
struct MetadataTensors(Vec<(String, &'static str, Vec<usize>)>);
impl MetadataTensors {
    fn put(&mut self, name: impl Into<String>, dtype: &'static str, shape: &[usize]) {
        self.0.push((name.into(), dtype, shape.to_vec()));
    }
    fn linear(&mut self, name: &str, rows: usize, columns: usize, bits: Option<u32>) {
        if let Some(bits) = bits {
            assert!(columns.is_multiple_of(64));
            self.put(
                format!("{name}.weight"),
                "U32",
                &[rows, columns * bits as usize / 32],
            );
            self.put(format!("{name}.scales"), "BF16", &[rows, columns / 64]);
            self.put(format!("{name}.biases"), "BF16", &[rows, columns / 64]);
        } else {
            self.put(format!("{name}.weight"), "BF16", &[rows, columns]);
        }
        self.put(format!("{name}.bias"), "BF16", &[rows]);
    }
    fn conv(&mut self, name: &str) {
        self.put(format!("{name}.weight"), "BF16", &[64, 64, 1, 1]);
        self.put(format!("{name}.bias"), "BF16", &[64]);
    }
    fn save(&self, path: &Path) {
        let tensors: Vec<_> = self
            .0
            .iter()
            .map(|(name, dtype, shape)| (name.as_str(), *dtype, shape.as_slice()))
            .collect();
        gen_core_testkit::wan_i2v::write_safetensors(path, &tensors);
    }
}

fn zimage_transformer_fixture(path: &Path, bits: Option<u32>) {
    let mut t = MetadataTensors::default();
    // ZImageTransformerConfig::turbo and the SAME block/global header schema
    // used by configured_overlay_semantic_facts_for (not a one-key workaround).
    for (stack, blocks, adaln) in [
        ("layers", 30, true),
        ("noise_refiner", 2, true),
        ("context_refiner", 2, false),
    ] {
        for block in 0..blocks {
            let base = format!("{stack}.{block}");
            if adaln {
                t.linear(&format!("{base}.adaLN_modulation.0"), 256, 64, bits);
            }
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                t.linear(&format!("{base}.attention.{leaf}"), 64, 64, bits);
            }
            for leaf in ["w1", "w2", "w3"] {
                t.linear(&format!("{base}.feed_forward.{leaf}"), 64, 64, bits);
            }
        }
    }
    for base in [
        "all_x_embedder.2-1",
        "cap_embedder.1",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "all_final_layer.2-1.linear",
        "all_final_layer.2-1.adaLN_modulation.0",
    ] {
        t.linear(base, 64, 64, bits);
    }
    t.save(path);
}
fn zimage_fixture(root: &Path, bits: Option<u32>) {
    zimage_transformer_fixture(&root.join("transformer/model.safetensors"), bits);
    tensor(&root.join("vae/model.safetensors"), "x");
    let config = bits.map_or_else(
        || "{}".to_owned(),
        |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
    );
    std::fs::write(root.join("transformer/config.json"), &config).unwrap();
    std::fs::write(root.join("vae/config.json"), config).unwrap();
    gen_core_testkit::write_encoder_contract_fixture_with_quant(
        &root.join("text_encoder"),
        mlx_gen_z_image::test_support::active_encoder_contract(),
        bits.map(|bits| bits as i32),
    )
    .unwrap();
}

fn anima_fixture(root: &Path) {
    let mut t = MetadataTensors::default();
    // Shipped Anima DiT: 28 blocks; Qwen conditioner: 6 blocks. These are
    // metadata-only, group-aligned projection facts, not runnable model weights.
    for block in 0..28 {
        for attention in ["self_attn", "cross_attn"] {
            for leaf in ["q_proj", "k_proj", "v_proj", "output_proj"] {
                t.linear(
                    &format!("net.blocks.{block}.{attention}.{leaf}"),
                    64,
                    64,
                    None,
                );
            }
        }
        for leaf in ["layer1", "layer2"] {
            t.linear(&format!("net.blocks.{block}.mlp.{leaf}"), 64, 64, None);
        }
        for norm in ["self_attn", "cross_attn", "mlp"] {
            for leaf in [1, 2] {
                t.linear(
                    &format!("net.blocks.{block}.adaln_modulation_{norm}.{leaf}"),
                    64,
                    64,
                    None,
                );
            }
        }
    }
    for leaf in [
        "x_embedder.proj.1",
        "t_embedder.1.linear_1",
        "t_embedder.1.linear_2",
        "final_layer.adaln_modulation.1",
        "final_layer.adaln_modulation.2",
        "final_layer.linear",
        "llm_adapter.out_proj",
    ] {
        t.linear(&format!("net.{leaf}"), 64, 64, None);
    }
    for block in 0..6 {
        for attention in ["self_attn", "cross_attn"] {
            for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                t.linear(
                    &format!("net.llm_adapter.blocks.{block}.{attention}.{leaf}"),
                    64,
                    64,
                    None,
                );
            }
        }
        for leaf in [0, 2] {
            t.linear(
                &format!("net.llm_adapter.blocks.{block}.mlp.{leaf}"),
                64,
                64,
                None,
            );
        }
    }
    t.save(&root.join("split_files/diffusion_models/anima-turbo-v1.0.safetensors"));
    tensor(
        &root.join("split_files/text_encoders/qwen_3_06b_base.safetensors"),
        "x",
    );
    tensor(
        &root.join("split_files/vae/qwen_image_vae.safetensors"),
        "x",
    );
}

fn sdxl_unet(path: &Path, include_up: bool) {
    fn transformer(t: &mut MetadataTensors, base: &str, depth: usize) {
        t.linear(&format!("{base}.proj_in"), 64, 64, None);
        t.linear(&format!("{base}.proj_out"), 64, 64, None);
        for block in 0..depth {
            for attention in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    t.linear(
                        &format!("{base}.transformer_blocks.{block}.{attention}.{leaf}"),
                        64,
                        64,
                        None,
                    );
                }
            }
            t.linear(
                &format!("{base}.transformer_blocks.{block}.ff.net.0.proj"),
                128,
                64,
                None,
            );
            t.linear(
                &format!("{base}.transformer_blocks.{block}.ff.net.2"),
                64,
                64,
                None,
            );
        }
    }
    fn resnet(t: &mut MetadataTensors, base: &str) {
        for leaf in ["conv1", "conv2"] {
            t.conv(&format!("{base}.{leaf}"));
        }
    }
    let mut t = MetadataTensors::default();
    t.conv("conv_in");
    if include_up {
        t.conv("conv_out");
    }
    // Exact SDXL-base BlockSpec order; optional time/shortcut projections are
    // genuinely absent, not fabricated selectors or missing required headers.
    for (stack, index, count, depth) in [
        ("down", 0, 2, 0),
        ("down", 1, 2, 2),
        ("down", 2, 2, 10),
        ("up", 0, 3, 10),
        ("up", 1, 3, 2),
        ("up", 2, 3, 0),
    ] {
        if stack == "up" && !include_up {
            continue;
        }
        for block in 0..count {
            resnet(&mut t, &format!("{stack}_blocks.{index}.resnets.{block}"));
            if depth != 0 {
                transformer(
                    &mut t,
                    &format!("{stack}_blocks.{index}.attentions.{block}"),
                    depth,
                );
            }
        }
        if stack == "down" && index < 2 {
            t.conv(&format!("down_blocks.{index}.downsamplers.0.conv"));
        }
        if stack == "up" && index < 2 {
            t.conv(&format!("up_blocks.{index}.upsamplers.0.conv"));
        }
    }
    resnet(&mut t, "mid_block.resnets.0");
    transformer(&mut t, "mid_block.attentions.0", 10);
    resnet(&mut t, "mid_block.resnets.1");
    t.save(path);
}
fn sdxl_fixture(root: &Path) {
    sdxl_unet(&root.join("unet/diffusion_pytorch_model.safetensors"), true);
    tensor(&root.join("vae/diffusion_pytorch_model.safetensors"), "x");
    for component in ["text_encoder", "text_encoder_2"] {
        tensor(&root.join(component).join("model.safetensors"), "x");
    }
    tokenizer(&root.join("tokenizer"));
    tokenizer(&root.join("tokenizer_2"));
}
fn sdxl_ip(root: &Path) {
    tensor(&root.join("models/image_encoder/model.safetensors"), "x");
    let mut t = MetadataTensors::default();
    // SAME complete SDXL-base cross-attention walk: down 4+20, mid 10,
    // up 30+6; each index has a real K/V pair in the h94 snapshot directory.
    for index in 0..70 {
        for role in ["k", "v"] {
            t.linear(
                &format!("ip_adapter.{}.to_{role}_ip", 2 * index + 1),
                64,
                64,
                None,
            );
        }
    }
    t.save(&root.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"));
}

fn sana(root: &Path) {
    tensor(&root.join("transformer/a.safetensors"), "proj.weight");
    tensor(
        &root.join("text_encoder/gemma-2-2b-it.safetensors"),
        "model.embed_tokens.weight",
    );
    tensor(&root.join("vae/a.safetensors"), "conv.weight");
}
pub(super) fn copy_tree(from: &Path, to: &Path) {
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
        let case = cell(&format!("sana_1600m/sana_1600m/bf16/512/{mode}/{rung}"));
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{result:?}");
        assert_eq!(p.refusal_kind, Some(kind), "{mode}/{rung}: {result:?}");
        if kind == "no_evidence" {
            assert!(p.artifact["layout_sha256"]
                .as_str()
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())));
            assert!(p.evidence_key.is_some(), "the memory key is recorded");
        }
    }
    // A dense snapshot requested as q4 is a typed ARTIFACT refusal (the provider publishes no
    // calibration for a load-time requantization), never an untyped failure or a wrong price.
    copy_tree(&root, &temp.path().join("sana_1600m/q4"));
    let case = cell("sana_1600m/sana_1600m/q4/512/text/2");
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
    let case = cell("sana_1600m/sana_1600m/bf16/512/text/2");
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
    let sdxl_root = temp.path().join("sdxl/bf16");
    sdxl_fixture(&sdxl_root);
    let anima_root = temp.path().join("anima_turbo/bf16");
    anima_fixture(&anima_root);
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
        let case = cell(&format!("{provider}/{provider}/bf16/512/text/0"));
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
        let case = cell(&format!("iris_3b/iris_3b/bf16/512/{mode}/0"));
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        assert_eq!(result.0, Status::AdmittedRefused, "{mode}: {result:?}");
        assert_eq!(p.refusal_kind, Some(kind), "{mode}: {result:?}");
    }
    let case = cell("iris_3b/iris_3b/bf16/512/prompt_batch/0");
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
    let case = cell("anima_turbo/anima_turbo/q4/512/text/0");
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
    let path = root
        .path()
        .join("sdxl/bf16/unet/diffusion_pytorch_model.safetensors");
    tensor(&path, "x");
    std::fs::write(path, b"invalid").unwrap();
    let mut p = adapter(catalog.media(), root.path().into());
    let case = cell("sdxl/sdxl/bf16/512/text/2");
    assert_eq!(
        campaign::run(&mut p, &case, false).0,
        Status::DiagnosticFailure
    );
    assert_eq!(p.refusal_kind, None);
}
#[test]
fn campaign_case_renders_are_unique_and_registered_in_manifest() {
    let root = tempfile::tempdir().unwrap();
    let mut case = cell("iris_3b/iris_3b/bf16/256/prompt_batch/0");
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
    let case = cell("sana_1600m/sana_1600m/bf16/512/text/2");
    let mut p = adapter(catalog.media(), temp.path().into());
    let (status, reason) = campaign::run(&mut p, &case, true);
    assert_eq!(status, Status::DiagnosticFailure, "{reason}");
    assert!(reason.contains("bounded per-quantum"), "{reason}");
    assert_eq!(p.refusal_kind, None);
    assert!(p.permit.is_none());
}

/// SDXL ControlNet and IP-Adapter are provisioned through components.json roles onto the load
/// spec's typed `control` / `ip_adapter` sources; `ip_adapter` is an enumerated SDXL mode, and a
/// case missing its component is a provisioning failure, never evidence.
#[test]
fn campaign_sdxl_control_and_ip_adapter_are_provisioned_roles() {
    let catalog = runtime_ios::catalog().unwrap();
    let sdxl = surfaces(catalog.media())
        .unwrap()
        .into_iter()
        .find(|s| s.provider == "sdxl")
        .unwrap();
    assert!(sdxl.modes.contains(&"ip_adapter".to_string()));
    assert!(sdxl.modes.contains(&"control".to_string()));
    let temp = tempfile::tempdir().unwrap();
    let slot = temp.path().join("sdxl/q4");
    sdxl_fixture(&slot);
    for mode in ["control", "ip_adapter"] {
        let case = cell(&format!("sdxl/sdxl/q4/512/{mode}/2"));
        let mut p = adapter(catalog.media(), temp.path().into());
        let (status, reason) = campaign::run(&mut p, &case, false);
        assert_eq!(status, Status::DiagnosticFailure, "{mode}: {reason}");
        assert!(reason.contains("provisioned"), "{reason}");
        assert_eq!(p.refusal_kind, None);
    }
    sdxl_unet(&slot.join("control/model.safetensors"), false);
    sdxl_ip(&slot.join("ip_adapter"));
    std::fs::write(
        slot.join("components.json"),
        r#"{"control":"control","ip_adapter":"ip_adapter"}"#,
    )
    .unwrap();
    let spec =
        |mode: &str| snapshot(temp.path(), &cell(&format!("sdxl/sdxl/q4/512/{mode}/2"))).unwrap();
    assert!(matches!(
        spec("control").control,
        Some(WeightsSource::Dir(_))
    ));
    assert!(spec("control").ip_adapter.is_none());
    assert!(matches!(
        spec("ip_adapter").ip_adapter,
        Some(WeightsSource::Dir(_))
    ));
    assert!(spec("ip_adapter").control.is_none());
    // An image-to-image case never loads the IP-Adapter (it would take the reference image).
    let img2img = spec("reference");
    assert!(img2img.ip_adapter.is_none() && img2img.control.is_none());
    assert!(
        img2img.components.is_empty(),
        "roles are not named components"
    );
    // Provisioned: both reach admission and are refused for EVIDENCE under their own routes.
    for mode in ["control", "ip_adapter"] {
        let case = cell(&format!("sdxl/sdxl/q4/512/{mode}/2"));
        let mut p = adapter(catalog.media(), temp.path().into());
        let (status, reason) = campaign::run(&mut p, &case, false);
        assert_eq!(status, Status::AdmittedRefused, "{mode}: {reason}");
        assert_eq!(p.refusal_kind, Some("no_evidence"), "{mode}: {reason}");
        assert_eq!(p.evidence_key.as_ref().unwrap().mode, mode);
    }
}
/// Z-Image Turbo's cells: both routes, every mode, refused for evidence (never capacity) on an
/// injected device before any load; the control route loads the `z_image_turbo` slot with that
/// slot's provisioned `control` role (and is a provisioning failure without it); a slot whose stored
/// tier is not the case's is a typed artifact refusal.
#[test]
fn campaign_injected_device_z_image_turbo_routes_and_tiers() {
    let _encoder_scope = mlx_gen_z_image::test_support::scoped_encoder_contract();
    let catalog = runtime_ios::catalog().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let slot = temp.path().join("z_image_turbo/q4");
    zimage_fixture(&slot, Some(4));
    let run = |prefix: &str| {
        let case = cell(prefix);
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = campaign::run(&mut p, &case, false);
        (result, p.refusal_kind)
    };
    // No control checkpoint provisioned yet: the control route is a provisioning failure.
    let ((status, reason), kind) =
        run("z_image_turbo_control/z_image_turbo_control/q4/512/control_pose/4");
    assert_eq!(status, Status::DiagnosticFailure, "{reason}");
    assert!(reason.contains("`control`"), "{reason}");
    assert_eq!(kind, None);
    tensor(&slot.join("control.safetensors"), "x");
    std::fs::write(
        slot.join("components.json"),
        r#"{"control":"control.safetensors"}"#,
    )
    .unwrap();
    for key in [
        "z_image_turbo/z_image_turbo/q4/512/text/4",
        "z_image_turbo/z_image_turbo/q4/768/reference/4",
        "z_image_turbo/z_image_turbo/q4/1024/text/0",
        "z_image_turbo_control/z_image_turbo_control/q4/512/control_pose/4",
        "z_image_turbo_control/z_image_turbo_control/q4/512/control_canny/4",
        "z_image_turbo_control/z_image_turbo_control/q4/512/control_depth/4",
        "z_image_turbo_control/z_image_turbo_control/q4/512/reference_depth/4",
    ] {
        let ((status, reason), kind) = run(key);
        assert_eq!(status, Status::AdmittedRefused, "{key}: {reason}");
        assert_eq!(kind, Some("no_evidence"), "{key}: {reason}");
    }
    // The q4 slot provisioned as the q8 tier: the stored tier is not the case's.
    copy_tree(&slot, &temp.path().join("z_image_turbo/q8"));
    let ((status, reason), kind) = run("z_image_turbo/z_image_turbo/q8/512/text/4");
    assert_eq!(status, Status::AdmittedRefused, "{reason}");
    assert_eq!(kind, Some("artifact"), "{reason}");
    // The control route carries its required control image, of the kind its mode names, in every
    // mode, ahead of a reference mode's init image; the base route never carries one.
    let carried = |prefix: &str| {
        let c = cell(prefix);
        let Ok(runtime_ios::media::IosWork::Generate(r)) =
            work(&c, &CancelFlag::new(), catalog.media())
        else {
            panic!("{prefix}: not a generate request");
        };
        r.conditioning
            .iter()
            .map(|x| match x {
                Conditioning::Control { kind, .. } => format!("control:{kind:?}"),
                Conditioning::Reference { .. } => "reference".to_string(),
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>()
    };
    for (kind, label) in [("pose", "Pose"), ("canny", "Canny"), ("depth", "Depth")] {
        assert_eq!(
            carried(&format!(
                "z_image_turbo_control/z_image_turbo_control/q4/512/control_{kind}/4"
            )),
            [format!("control:{label}")]
        );
        assert_eq!(
            carried(&format!(
                "z_image_turbo_control/z_image_turbo_control/q4/512/reference_{kind}/4"
            )),
            [format!("control:{label}"), "reference".to_string()]
        );
    }
    assert_eq!(
        carried("z_image_turbo/z_image_turbo/q4/512/reference/4"),
        ["reference"]
    );
    assert!(carried("z_image_turbo/z_image_turbo/q4/512/text/4").is_empty());
    // A mode naming a kind the route does not accept is a typed mode refusal.
    let mut odd = cell("z_image_turbo_control/z_image_turbo_control/q4/512/control_pose/4");
    odd.mode = "control_scribble".into();
    assert!(matches!(
        work(&odd, &CancelFlag::new(), catalog.media()),
        Err(AdmissionError::Refused(_))
    ));
    // Families (weights-free listing: no checkpoint collapse): the base route's text and reference
    // share its text-anchored family at every tier; the control route's six modes (one per accepted
    // kind, no plain text mode) share one family of their own; every family has both anchors.
    let listing: Value = serde_json::from_str(&listing_with(None).unwrap()).unwrap();
    let families = listing["families"].as_array().unwrap();
    for tier in ["q4", "q8", "bf16"] {
        let family = |anchor: &str| {
            families
                .iter()
                .find(|f| f["tier"] == tier && f["anchor"].as_str().unwrap().starts_with(anchor))
                .unwrap_or_else(|| panic!("{anchor} {tier}"))
        };
        let base = family("z_image_turbo/z_image_turbo/");
        assert_eq!(base["anchor"], "z_image_turbo/z_image_turbo/text");
        assert_eq!(
            base["members"],
            json!([
                "z_image_turbo/z_image_turbo/text",
                "z_image_turbo/z_image_turbo/reference"
            ])
        );
        let control = family("z_image_turbo_control/");
        let members: Vec<&str> = control["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap())
            .collect();
        for kind in ["pose", "canny", "depth"] {
            for mode in ["control", "reference"] {
                assert!(members.contains(
                    &format!("z_image_turbo_control/z_image_turbo_control/{mode}_{kind}").as_str()
                ));
            }
        }
        assert!(!members.iter().any(|m| m.ends_with("/text")));
    }
}

/// Every cell carries its own selection from the provider's declared space, enumerated quality
/// first, executed as selected and recorded in the evidence key: a bf16 Z-Image cell can stream the
/// text encoder (`te` / `both`), a q4 1024 cell can decode at tile 512, an SDXL cell reaches a
/// non-first tile; a token outside the space is a typed strategy refusal.
#[test]
fn campaign_cells_execute_their_own_declared_selection() {
    let _encoder_scope = mlx_gen_z_image::test_support::scoped_encoder_contract();
    let catalog = runtime_ios::catalog().unwrap();
    let discovered = surfaces(catalog.media()).unwrap();
    // The provider's declared space for `provider/tier/rung`, quality first.
    let tokens = |provider: &str, tier: &str, rung: u8| {
        discovered
            .iter()
            .find(|s| s.provider == provider)
            .and_then(|s| {
                s.selections
                    .get(&(provider.to_string(), tier.to_string(), rung))
                    .cloned()
            })
            .unwrap_or_default()
    };
    // Z-Image rung 4: staging x window scope x tile, quality first (unstaged, DiT-only scope,
    // largest tile first; staged, both streamed, smallest tile last).
    let z = tokens("z_image_turbo", "bf16", 4);
    assert_eq!(z.len(), 2 * 3 * 7, "{z:?}");
    assert_eq!(z[0], "t768_o64_c67108864_w1_dit");
    assert_eq!(z[1], "t640_o64_c67108864_w1_dit");
    assert_eq!(z[z.len() - 1], "s_t256_o64_c67108864_w1_both");
    assert!(z.contains(&"t512_o64_c67108864_w1_te".to_string()));
    assert!(z.contains(&"s_t512_o64_c67108864_w1_dit".to_string()));
    // SANA rung 2 enumerates every declared tile (then the same staged); SDXL rung 2 its native
    // route's tiles.
    assert_eq!(
        tokens("sana_1600m", "q4", 2),
        [
            "t512_o48",
            "t384_o48",
            "t256_o48",
            "t192_o48",
            "s_t512_o48",
            "s_t384_o48",
            "s_t256_o48",
            "s_t192_o48"
        ]
    );
    let sdxl = tokens("sdxl", "q4", 2);
    assert_eq!(sdxl[0], "t896_o64");
    assert!(sdxl.contains(&"t512_o64".to_string()));

    // Executed as selected, on provisioned (header-only) slots.
    let temp = tempfile::tempdir().unwrap();
    let zslot = |tier: &str, bits: Option<u32>| {
        let slot = temp.path().join("z_image_turbo").join(tier);
        zimage_fixture(&slot, bits);
    };
    zslot("bf16", None);
    zslot("q4", Some(4));
    let prepared = |key: &str| {
        let case = Case::parse(key).unwrap();
        let mut p = adapter(catalog.media(), temp.path().into());
        let result = p.prepare(&case);
        (result, p)
    };
    let selected = |key: &str| {
        let (result, p) = prepared(key);
        result.unwrap_or_else(|e| panic!("{key}: {e:?}"));
        let candidate = p.candidate.unwrap();
        let execution = p.evidence_key.unwrap().execution;
        (
            candidate.context.selection.parameters,
            candidate.execution,
            execution,
        )
    };
    for scope in ["te", "both"] {
        let (parameters, memory, key) = selected(&format!(
            "z_image_turbo/z_image_turbo/bf16/512/text/4/t768_o64_c67108864_w1_{scope}/1"
        ));
        let want = if scope == "te" {
            TransformerComponent::TextEncoder
        } else {
            TransformerComponent::Both
        };
        assert_eq!(parameters.transformer_window_component, Some(want));
        assert_eq!(memory.transformer_window_component, Some(want));
        assert!(key.contains(&format!("{want:?}")), "{key}");
    }
    let (parameters, memory, key) =
        selected("z_image_turbo/z_image_turbo/q4/1024/text/4/s_t512_o64_c67108864_w1_dit/1");
    assert_eq!(parameters.decode_tile_edge, Some(512));
    assert_eq!(memory.decode_tile_edge, Some(512));
    assert!(memory.stage_residency, "the staged selection evicts phases");
    assert!(key.contains("stage_residency: Some(true)"), "{key}");
    let (_, _, first) =
        selected("z_image_turbo/z_image_turbo/q4/1024/text/4/t768_o64_c67108864_w1_dit/1");
    assert_ne!(key, first, "the evidence key separates the selections");
    assert!(key.contains("decode_tile_edge: Some(512)"), "{key}");
    // SDXL: a non-first native tile.
    let slot = temp.path().join("sdxl/q4");
    sdxl_fixture(&slot);
    let (parameters, memory, _) = selected("sdxl/sdxl/q4/1024/text/2/t512_o64/1");
    assert_eq!(parameters.decode_tile_edge, Some(512));
    assert_eq!(memory.decode_tile_edge, Some(512));
    // A token outside the declared space: typed strategy refusal, nothing built.
    let (result, p) =
        prepared("z_image_turbo/z_image_turbo/q4/1024/text/4/t500_o64_c67108864_w1_dit/1");
    assert!(matches!(result, Err(AdmissionError::Refused(_))));
    assert_eq!(p.refusal_kind, Some("strategy"));
    assert!(p.candidate.is_none());
}

/// HOST: the campaign states a finite, source-derived host bound for ITS OWN callbacks: no
/// allocation per progress or preview event (its progress closures are `|_| {}`), and per quantum
/// mark at least the phase key, the context Vec and the `Mark` slot, and one byte per label byte
/// (labels are stored in full, never capped).
#[test]
fn the_campaign_states_its_listener_callback_host_bound() {
    use runtime_ios::media::evidence::{MAX_SAMPLING_MARKS, MAX_SAMPLING_NAME_BYTES};
    let runtime_ios::media::IosCallbackHost::Bounded {
        fixed_bytes,
        per_progress_event_bytes,
        per_preview_event_bytes,
        per_quantum_mark_bytes,
        per_quantum_label_byte_bytes,
    } = support::capture_listener_callback_host("iris_3b").unwrap()
    else {
        panic!("the campaign's listener allocates, so its term is Bounded");
    };
    assert_eq!((per_progress_event_bytes, per_preview_event_bytes), (0, 0));
    // Labels are kept in full: their copy is a per-byte term, never a cap.
    assert_eq!(per_quantum_label_byte_bytes, 1);
    let floor = "conditioning".len() as u64
        + 4 * MAX_SAMPLING_MARKS as u64
        + std::mem::size_of::<support::Mark>() as u64;
    assert!(
        per_quantum_mark_bytes >= floor,
        "{per_quantum_mark_bytes} < {floor}"
    );
    assert!(fixed_bytes >= MAX_SAMPLING_MARKS as u64 * 2 * MAX_SAMPLING_NAME_BYTES as u64);
    // The provider String the listener clones is part of the installation term.
    let longer = support::capture_listener_callback_host("iris_3b_restore").unwrap();
    let runtime_ios::media::IosCallbackHost::Bounded {
        fixed_bytes: longer_fixed,
        ..
    } = longer
    else {
        panic!("Bounded");
    };
    assert_eq!(longer_fixed - fixed_bytes, "_restore".len() as u64);
}

#[test]
fn compact_zimage_contract_is_thread_local_nested_and_restores_production() {
    use mlx_gen_z_image::{test_support, ENCODER_CONTRACT};
    assert_eq!(test_support::active_encoder_contract(), ENCODER_CONTRACT);
    {
        let _outer = test_support::scoped_encoder_contract();
        let compact = test_support::encoder_contract();
        compact.validate_definition().unwrap();
        assert_eq!(compact.hidden_size, ENCODER_CONTRACT.hidden_size);
        assert_eq!(compact.output_width, ENCODER_CONTRACT.output_width);
        assert_eq!(
            compact.selected_hidden_layers,
            &[compact.loaded_hidden_layers - 1]
        );
        assert_eq!(test_support::active_encoder_contract(), compact);
        for (actual, production) in compact
            .tokenizer
            .required_tokens
            .iter()
            .zip(ENCODER_CONTRACT.tokenizer.required_tokens)
        {
            assert_eq!(actual.literal, production.literal);
            assert_eq!(actual.role, production.role);
            assert!((actual.id as usize) < compact.vocab_size);
        }
        {
            let _inner = test_support::scoped_encoder_contract();
            assert_eq!(test_support::active_encoder_contract(), compact);
        }
        assert_eq!(test_support::active_encoder_contract(), compact);
        std::thread::spawn(|| {
            assert_eq!(test_support::active_encoder_contract(), ENCODER_CONTRACT)
        })
        .join()
        .unwrap();
        assert_eq!(
            mlx_gen_z_image::model::descriptor().encoder_contract,
            Some(compact)
        );
        let config = mlx_gen_z_image::text_encoder::ZTextEncoderConfig::z_image();
        assert_eq!(config.hidden_size as usize, compact.hidden_size);
        assert_eq!(config.vocab_size as usize, compact.vocab_size);
        assert_eq!(config.n_layers, compact.loaded_hidden_layers);
    }
    assert_eq!(test_support::active_encoder_contract(), ENCODER_CONTRACT);
    assert_eq!(
        mlx_gen_z_image::model::descriptor().encoder_contract,
        Some(ENCODER_CONTRACT)
    );
}
