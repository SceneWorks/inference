//! `REG-4` / `PIPE-12` / `TOOL-4`: narrow structural guards and boundary contract tests for
//! typed requests, outputs, per-run execution reports, and id-free shared dispatch.

use std::path::PathBuf;

use gen_core::generator::ExecutionIdentity;
use gen_core::{
    Capabilities, FamilyExecutionReport, FamilyExtension, FamilyExtensionKind, FeatureParams,
    Flux3Extension, GenerationOutput, GenerationReport, GenerationRequest, GuidanceMethodId,
    H3Extension, Image, Modality, ModelDescriptor, MossImageExtension, PidParams, ScheduleId,
    SolverId, SvdExtension, VectorFinishReason, VectorOutput, VectorTerminalMetadata, VideoMode,
};

/// Blank comments, strings, raw strings, and char literals with ASCII spaces while preserving
/// newlines so reported line numbers match the original source.
fn strip_comments_and_strings(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    let blank = |out: &mut [u8], from: usize, to: usize| {
        for byte in out.iter_mut().take(to).skip(from) {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                let start = i;
                i += 2;
                let mut depth = 1usize;
                while i < b.len() && depth > 0 {
                    if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                blank(&mut out, start, i);
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("ASCII blanks preserve UTF-8")
}

/// Strip comments only (keeping string literals intact) so we can scan non-test production code
/// for hardcoded model-id string comparisons.
fn strip_comments_keep_strings(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    let blank = |out: &mut [u8], from: usize, to: usize| {
        for byte in out.iter_mut().take(to).skip(from) {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                let start = i;
                i += 2;
                let mut depth = 1usize;
                while i < b.len() && depth > 0 {
                    if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                blank(&mut out, start, i);
            }
            b'"' => {
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("ASCII blanks preserve UTF-8")
}

/// Truncate a source file at its first `#[cfg(test)]` module so unit-test fixtures are excluded.
fn non_test_prefix(src: &str) -> &str {
    match src.find("#[cfg(test)]") {
        Some(idx) => &src[..idx],
        None => src,
    }
}

/// Extract the brace-delimited body immediately following `needle`.
fn extract_braced_block<'a>(src: &'a str, needle: &str) -> Option<&'a str> {
    let start = src.find(needle)? + needle.len();
    let rest = &src[start..];
    let open = rest.find('{')?;
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    for (idx, &ch) in bytes.iter().enumerate().skip(open) {
        if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(&rest[open + 1..idx]);
            }
        }
    }
    None
}

const FORBIDDEN_FAMILY_TOKENS: &[&str] = &[
    "moss",
    "flux",
    "h3",
    "veda",
    "ltx",
    "svd",
    "pulid",
    "seedvr",
    "wan",
    "krea",
    "qwen",
    "sdxl",
    "yue",
    "scail",
    "pid",
    "kolors",
    "iris",
    "zimage",
    "z_image",
    "anima",
    "boogu",
    "chroma",
    "bernini",
    "sensenova",
    "starvector",
];

const FORBIDDEN_FAMILY_REQUEST_FIELDS: &[&str] = &[
    "motion_bucket_id",
    "noise_aug_strength",
    "decode_chunk_size",
    "conditioning_fps",
    "num_generated_keyframes",
    "temporal_upsample_rounds",
    "auto_duration",
    "timestep_to_start_cfg",
    "trim_first_frames",
    "softness",
    "text_style_gain",
    "use_uncensored_enhancer",
    "reference_image_short_edge",
    "reference_image_resize_mode",
];

fn contains_family_token(ident: &str) -> bool {
    let lower = ident.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split('_').collect();
    FORBIDDEN_FAMILY_TOKENS.iter().any(|token| {
        parts.contains(token)
            || lower.starts_with(&format!("{token}_"))
            || lower.contains(&format!("_{token}_"))
            || lower.ends_with(&format!("_{token}"))
            || lower == *token
    })
}

fn check_generator_trait_methods(src: &str) -> Vec<String> {
    let stripped = strip_comments_and_strings(src);
    let Some(body) = extract_braced_block(&stripped, "pub trait Generator") else {
        return vec!["could not locate `pub trait Generator` block".into()];
    };
    let mut violations = Vec::new();
    let mut depth = 0usize;
    for line in body.lines() {
        let trimmed = line.trim();
        if depth == 0 {
            if let Some(after_fn) = trimmed.strip_prefix("fn ") {
                let name = after_fn
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                if contains_family_token(name) {
                    violations.push(format!("family-named Generator method `{name}`"));
                }
                if name == "adapter_apply_reports"
                    || (name.ends_with("_execution_report") && !name.starts_with("generate_"))
                {
                    violations.push(format!(
                        "per-run side-channel getter `{name}` must return on GenerationReport"
                    ));
                }
            }
        }
        for ch in trimmed.bytes() {
            if ch == b'{' {
                depth += 1;
            } else if ch == b'}' {
                depth = depth.saturating_sub(1);
            }
        }
    }
    violations
}

fn check_core_request_fields(src: &str) -> Vec<String> {
    let stripped = strip_comments_and_strings(src);
    let Some(body) = extract_braced_block(&stripped, "pub struct GenerationRequest") else {
        return vec!["could not locate `pub struct GenerationRequest` block".into()];
    };
    let mut violations = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(after_pub) = trimmed.strip_prefix("pub ") {
            let field = after_pub.split(':').next().unwrap_or("").trim();
            if field.is_empty() {
                continue;
            }
            if contains_family_token(field) || FORBIDDEN_FAMILY_REQUEST_FIELDS.contains(&field) {
                violations.push(format!("family-specific top-level request field `{field}`"));
            }
        }
    }
    violations
}

const FORBIDDEN_MODEL_ID_GATES: &[&str] = &[
    "id == \"",
    "id != \"",
    "id.starts_with(\"",
    "id.contains(\"",
    "descriptor.id == \"",
    "descriptor.id != \"",
    "descriptor.id.starts_with(\"",
    "descriptor.id.contains(\"",
    "descriptor.family == \"",
    "descriptor.family != \"",
    "DISTILLED_MODEL_ID",
    "BASE_MODEL_ID",
    "MossImageVariant::from_provider_id(",
];

fn check_shared_model_id_gates(src: &str) -> Vec<String> {
    let code = strip_comments_keep_strings(non_test_prefix(src));
    let mut violations = Vec::new();
    for (idx, line) in code.lines().enumerate() {
        for pattern in FORBIDDEN_MODEL_ID_GATES {
            if line.contains(pattern) {
                violations.push(format!(
                    "line {}: hardcoded model/family id gate `{pattern}`",
                    idx + 1
                ));
            }
        }
    }
    violations
}

fn gen_core_src(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(rel)
}

#[test]
fn tool4_generator_trait_has_zero_family_named_methods_and_side_channel_getters() {
    let traits_src = std::fs::read_to_string(gen_core_src("generator/traits.rs"))
        .expect("read generator/traits.rs");
    let violations = check_generator_trait_methods(&traits_src);
    assert!(
        violations.is_empty(),
        "Generator trait violates PIPE-12 / TOOL-4:\n{}",
        violations.join("\n")
    );

    // Positive fixture: family-free trait passes.
    let positive = r#"
        pub trait Generator {
            fn descriptor(&self) -> &ModelDescriptor;
            fn retained_weight_bytes(&self) -> Option<u64> { None }
            fn request_workspace_bound(&self, req: &GenerationRequest) -> Result<Option<RequestWorkspaceBound>> { Ok(None) }
        }
    "#;
    assert!(check_generator_trait_methods(positive).is_empty());

    // Negative fixtures: family-named methods and side-channel getters must be rejected.
    let negative_family = r#"
        pub trait Generator {
            fn moss_image_owned_weight_bytes(&self) -> Option<u64> { None }
            fn veda_workspace_bound(&self) -> Option<u64> { None }
        }
    "#;
    let neg_hits = check_generator_trait_methods(negative_family);
    assert_eq!(
        neg_hits.len(),
        2,
        "expected 2 family-method hits: {neg_hits:?}"
    );

    let negative_getter = r#"
        pub trait Generator {
            fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> { Vec::new() }
        }
    "#;
    let getter_hits = check_generator_trait_methods(negative_getter);
    assert_eq!(
        getter_hits.len(),
        1,
        "expected side-channel getter hit: {getter_hits:?}"
    );
}

#[test]
fn tool4_core_request_has_zero_family_specific_top_level_fields() {
    let request_src = std::fs::read_to_string(gen_core_src("generator/request.rs"))
        .expect("read generator/request.rs");
    let violations = check_core_request_fields(&request_src);
    assert!(
        violations.is_empty(),
        "GenerationRequest / CoreRequest violates REG-4 / TOOL-4:\n{}",
        violations.join("\n")
    );

    // Positive fixture: core fields + features + extensions passes.
    let positive = r#"
        pub struct GenerationRequest {
            pub prompt: String,
            pub sampler: Option<SolverId>,
            pub features: FeatureParams,
            pub extensions: FamilyExtensions,
        }
    "#;
    assert!(check_core_request_fields(positive).is_empty());

    // Negative fixture: family-specific fields must be rejected.
    let negative = r#"
        pub struct GenerationRequest {
            pub prompt: String,
            pub moss_image: Option<MossImageOptions>,
            pub motion_bucket_id: Option<f32>,
            pub use_pid: bool,
        }
    "#;
    let neg_hits = check_core_request_fields(negative);
    assert_eq!(
        neg_hits.len(),
        3,
        "expected 3 family-field hits: {neg_hits:?}"
    );
}

#[test]
fn tool4_shared_validation_and_dispatch_compare_zero_model_ids() {
    for rel in [
        "generator/capabilities.rs",
        "registry/providers.rs",
        "feature.rs",
    ] {
        let src = std::fs::read_to_string(gen_core_src(rel)).expect("read shared gen-core file");
        let violations = check_shared_model_id_gates(&src);
        assert!(
            violations.is_empty(),
            "{rel} violates REG-4 / TOOL-4:\n{}",
            violations.join("\n")
        );
    }

    // Positive fixture: capability-driven checks pass.
    let positive = r#"
        fn validate(caps: &Capabilities, id: &str) -> bool {
            let _ = id;
            caps.supports_ar_load_controls && caps.allows_low_res_start_single_ref_img2img
        }
    "#;
    assert!(check_shared_model_id_gates(positive).is_empty());

    // Negative fixture: hardcoded model id string comparisons must be rejected.
    let negative = r#"
        fn validate(id: &str, descriptor: &ModelDescriptor) -> bool {
            if id != "yue2" { return false; }
            if id.starts_with("flux1_") { return false; }
            if descriptor.family == "pulid" { return false; }
            true
        }
    "#;
    let neg_hits = check_shared_model_id_gates(negative);
    assert_eq!(neg_hits.len(), 3, "expected 3 id-gate hits: {neg_hits:?}");
}

#[test]
fn reg4_boundary_selectors_and_extensions_roundtrip_and_compose() {
    // Selectors parse canonical names into typed variants and preserve unknown strings losslessly.
    assert_eq!(SolverId::from("euler"), SolverId::Euler);
    assert_eq!(SolverId::Euler.as_str(), "euler");
    assert_eq!(
        SolverId::from("bogus_solver"),
        SolverId::Custom("bogus_solver".into())
    );
    assert_eq!(
        serde_json::from_str::<SolverId>(&serde_json::to_string(&SolverId::Dpmpp2m).unwrap())
            .unwrap(),
        SolverId::Dpmpp2m
    );

    assert_eq!(ScheduleId::from("karras"), ScheduleId::Karras);
    assert_eq!(ScheduleId::Karras.as_str(), "karras");
    assert_eq!(
        serde_json::from_str::<ScheduleId>(&serde_json::to_string(&ScheduleId::Beta57).unwrap())
            .unwrap(),
        ScheduleId::Beta57
    );

    assert_eq!(
        GuidanceMethodId::from("cfg_rescale"),
        GuidanceMethodId::CfgRescale
    );
    assert_eq!(GuidanceMethodId::CfgPp.as_str(), "cfg_pp");
    assert_eq!(
        serde_json::from_str::<GuidanceMethodId>(
            &serde_json::to_string(&GuidanceMethodId::Apg).unwrap()
        )
        .unwrap(),
        GuidanceMethodId::Apg
    );

    assert_eq!(VideoMode::from("animation"), VideoMode::Animation);
    assert_eq!(VideoMode::VideoBridge.as_str(), "video_bridge");

    // Composable FamilyExtensions and FeatureParams.
    let req = GenerationRequest {
        prompt: "composed".into(),
        sampler: Some(SolverId::Euler),
        scheduler: Some(ScheduleId::Simple),
        guidance_method: Some(GuidanceMethodId::Cfg),
        video_mode: Some(VideoMode::TextToVideo),
        features: FeatureParams {
            pid: Some(PidParams::with_capture_sigma(0.25)),
            ..Default::default()
        },
        ..Default::default()
    }
    .with_extension(FamilyExtension::H3(H3Extension {
        terminal_telemetry: true,
        reference_image_short_edge: Some(1536),
        ..Default::default()
    }))
    .with_extension(FamilyExtension::Svd(SvdExtension {
        motion_bucket_id: Some(127.0),
        ..Default::default()
    }));
    assert!(req.use_pid());
    assert_eq!(req.pid_capture_sigma(), Some(0.25));
    assert!(!req.extensions.is_empty());
    assert_eq!(
        req.extensions
            .h3
            .as_ref()
            .unwrap()
            .reference_image_short_edge,
        Some(1536)
    );
    assert_eq!(req.extensions.svd.unwrap().motion_bucket_id, Some(127.0));

    // Capability-driven extension gating refuses unsupported family extensions without checking model id.
    let base_caps = Capabilities {
        min_size: 256,
        max_size: 2048,
        max_count: 1,
        ..Default::default()
    };
    let flux3_req =
        GenerationRequest::default().with_extension(FamilyExtension::Flux3(Flux3Extension {
            reference_output: Some(gen_core::flux3::references::ReferenceOutputSize {
                width: Some(512),
                height: None,
            }),
            ..Default::default()
        }));
    assert!(base_caps
        .validate_request("flux3_klein", &flux3_req)
        .is_err());
    let flux3_caps = Capabilities {
        supported_family_extensions: vec![FamilyExtensionKind::Flux3],
        ..base_caps.clone()
    };
    assert!(flux3_caps
        .validate_request("any_route_id", &flux3_req)
        .is_ok());

    let moss_req = GenerationRequest::default().with_extension(FamilyExtension::MossImage(
        Box::new(MossImageExtension::from_options(Default::default())),
    ));
    assert!(base_caps
        .validate_request("moss_image_sft", &moss_req)
        .is_err());
    let moss_caps = Capabilities {
        supported_family_extensions: vec![FamilyExtensionKind::MossImage],
        ..base_caps
    };
    assert!(moss_caps
        .validate_request("any_moss_route", &moss_req)
        .is_ok());
}

#[test]
fn pipe12_vector_output_and_per_run_execution_reports_are_typed() {
    let vector = VectorOutput {
        svg: "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\"><circle cx=\"32\" cy=\"32\" r=\"20\"/></svg>".into(),
        preview: Some(Image {
            width: 2,
            height: 2,
            pixels: vec![255; 12],
        }),
        finish_reason: VectorFinishReason::Stop,
        generated_tokens: 42,
        prompt_tokens: Some(12),
        truncated: false,
        terminal: VectorTerminalMetadata {
            sanitizer_id: Some("starvector-svg-v1".into()),
            viewport: Some([64, 64]),
            svg_sha256: Some("0".repeat(64)),
            element_count: Some(1),
            kv_cache: Some(gen_core::core_llm::KvCacheReport::without_table_family(
                gen_core::core_llm::KvCompressionPolicy::Qualified,
            )),
            ..VectorTerminalMetadata::default()
        },
    };
    let json = serde_json::to_string(&vector).expect("serialize VectorOutput");
    let decoded: VectorOutput = serde_json::from_str(&json).expect("deserialize VectorOutput");
    assert_eq!(decoded.svg, vector.svg);
    assert_eq!(decoded.finish_reason, VectorFinishReason::Stop);
    assert_eq!(decoded.generated_tokens, 42);
    assert_eq!(decoded.prompt_tokens, Some(12));
    assert!(!decoded.truncated);
    assert_eq!(decoded.terminal, vector.terminal);
    assert!(
        decoded.preview.is_none(),
        "preview pixels are skipped in JSON"
    );

    let descriptor = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "starvector_1b",
        family: "starvector",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            max_count: 1,
            min_size: 64,
            max_size: 1024,
            ..Default::default()
        },
    };
    let mut report = GenerationReport::from_output(GenerationOutput::Vector(vector));
    report.family_report = Some(FamilyExecutionReport::Veda(Default::default()));
    assert!(report.veda_report().is_some());
    assert!(report.moss_image_report().is_none());
    assert!(report.moss_image_edit_report().is_none());

    let completed = gen_core::generator::CompletedGeneration::new(
        report,
        ExecutionIdentity::provider_default(),
        &descriptor,
    )
    .expect("VectorOutput counts as 1 output unit");
    assert!(matches!(
        completed.generation.output,
        Some(GenerationOutput::Vector(_))
    ));

    // PIPE-12: typed family-agnostic latent batch output validates shape/length and counts batch
    // elements as output units through CompletedGeneration::new.
    let latent_report = GenerationReport::from_output(GenerationOutput::Latents {
        values: vec![0.25; 2 * 128],
        shape: [2, 128, 1, 1],
    });
    let latent_completed = gen_core::generator::CompletedGeneration::new(
        latent_report,
        ExecutionIdentity::provider_default(),
        &descriptor,
    )
    .expect("valid Latents batch validates through CompletedGeneration::new");
    assert!(matches!(
        latent_completed.generation.output,
        Some(GenerationOutput::Latents {
            shape: [2, 128, 1, 1],
            ..
        })
    ));

    let bad_latent = GenerationReport::from_output(GenerationOutput::Latents {
        values: vec![0.25; 4],
        shape: [2, 128, 1, 1],
    });
    assert!(
        gen_core::generator::CompletedGeneration::new(
            bad_latent,
            ExecutionIdentity::provider_default(),
            &descriptor,
        )
        .is_err(),
        "mismatched Latents value length must fail CompletedGeneration::new"
    );
}
