use super::*;
use crate::{LoadSpec, WeightsSource};

#[test]
fn bounded_decode_params_require_a_published_edge_and_the_exact_overlap() {
    let check = |edge, overlap| validate_bounded_decode_params("m", &[512, 256], 64, edge, overlap);
    assert!(check(Some(512), Some(64)).is_ok());
    assert!(check(Some(256), Some(64)).is_ok());
    for (edge, overlap, needle) in [
        (None, Some(64), "m: bounded decode requires a tile edge"),
        (Some(512), None, "m: bounded decode requires a tile overlap"),
        (
            Some(384),
            Some(64),
            "m: decode tile 384/64 is outside the published domain",
        ),
        (
            Some(512),
            Some(32),
            "m: decode tile 512/32 is outside the published domain",
        ),
    ] {
        let error = check(edge, overlap).unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(error.to_string().contains(needle), "{error}");
    }
}

#[test]
fn adapter_residency_distinguishes_folded_additive_and_missing_evidence() {
    // sc-17755: a `TempDir` guard, so the tree leaves on `Drop` even when an assertion below
    // panics — the trailing `remove_dir_all` this replaced never ran on a failing test.
    let tmp = tempfile::Builder::new()
        .prefix("gen-core-adapter-residency-")
        .tempdir()
        .expect("fixture temp dir");
    let root = tmp.path();
    let first = root.join("first.safetensors");
    let second = root.join("second.safetensors");
    std::fs::write(&first, vec![0_u8; 11]).unwrap();
    std::fs::write(&second, vec![0_u8; 17]).unwrap();
    let adapters = vec![
        AdapterSpec::new(first, 1.0, crate::AdapterKind::Lora),
        AdapterSpec::new(second, 0.5, crate::AdapterKind::Lokr),
    ];

    assert_eq!(
        adapter_stack_resident_bytes(&adapters, AdapterResidencyMode::Folded),
        Some(0)
    );
    assert_eq!(
        adapter_stack_resident_bytes(&adapters, AdapterResidencyMode::Additive),
        Some(28)
    );

    let missing = vec![AdapterSpec::new(
        root.join("missing.safetensors"),
        1.0,
        crate::AdapterKind::Lora,
    )];
    assert_eq!(
        adapter_stack_resident_bytes(&missing, AdapterResidencyMode::Additive),
        None
    );
}

/// Write a safetensors file with one tensor per `(name, dtype, shape)`, zero-filled.
fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize], usize)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, width) in tensors {
        let bytes = shape.iter().product::<usize>() * width;
        header.insert(
                (*name).to_owned(),
                serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes] }),
            );
        offset += bytes;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.resize(file.len() + offset, 0);
    std::fs::write(path, file).unwrap();
}

/// sc-24163: an upcasting installer's stack is priced at f32 per float element plus the
/// compute-width copy, whatever the file stores; an f32 compute dtype adds no copy; a non-float
/// tensor keeps its stored bytes; a missing or unreadable file fails closed; a folded stack is
/// zero. The file-length pricing stays what it was, and is below the upcast one for a bf16 file.
#[test]
fn upcast_adapter_residency_prices_f32_factors_and_the_prepared_copy() {
    let tmp = tempfile::Builder::new()
        .prefix("gen-core-upcast-residency-")
        .tempdir()
        .expect("fixture temp dir");
    let root = tmp.path();
    let bf16 = root.join("bf16.safetensors");
    let f32 = root.join("f32.safetensors");
    write_safetensors(
        &bf16,
        &[
            ("a.lora_A.weight", "BF16", &[16, 64], 2),
            ("a.lora_B.weight", "BF16", &[32, 16], 2),
            ("a.alpha", "I64", &[], 8),
        ],
    );
    write_safetensors(
        &f32,
        &[
            ("a.lora_A.weight", "F32", &[16, 64], 4),
            ("a.lora_B.weight", "F32", &[32, 16], 4),
        ],
    );
    let elements = 16 * 64 + 32 * 16;
    let spec =
        |path: &std::path::Path| AdapterSpec::new(path.to_owned(), 1.0, crate::AdapterKind::Lora);
    let bf16_stack = vec![spec(&bf16)];
    let f32_stack = vec![spec(&f32)];
    let additive = AdapterResidencyMode::Additive;
    let cached = UpcastLoraCopy::Cached;
    let price = |stack: &[AdapterSpec], width| {
        adapter_stack_upcast_resident_bytes(stack, additive, width, cached)
    };

    assert_eq!(
        price(&bf16_stack, 2),
        Some(elements * (4 + 2) + 8),
        "f32 factors + a bf16 prepared copy, and the i64 alpha at its stored width"
    );
    assert_eq!(
        price(&f32_stack, 2),
        Some(elements * (4 + 2)),
        "the same factors in an f32 file cost the same"
    );
    assert_eq!(
        price(&f32_stack, 4),
        Some(elements * 4),
        "an f32 compute dtype shares the factors' storage"
    );
    assert_eq!(
        adapter_stack_upcast_resident_bytes(&f32_stack, additive, 2, UpcastLoraCopy::PerForward),
        Some(elements * 4),
        "a residual that casts per forward keeps only the f32 factors"
    );
    let both = vec![spec(&bf16), spec(&f32)];
    assert_eq!(
        price(&both, 2),
        Some(2 * elements * 6 + 8),
        "a stack is the sum of its members"
    );
    let file_priced = adapter_stack_resident_bytes(&bf16_stack, additive).unwrap();
    assert!(file_priced < elements * 6, "{file_priced}");

    // A LoKr is priced per module from its Kronecker dims, not its stored elements: the low-rank
    // `w2_a [16, 2]·w2_b [2, 16]` (64 stored) is materialized to `w2 [16, 16]` (256), plus the
    // f32 `w1 [2, 4]`, the bf16 `w1` copy and the `w2ᵀ` copy. The i64 alpha keeps its 8 bytes.
    let lokr = root.join("lokr.safetensors");
    write_safetensors(
        &lokr,
        &[
            ("m.lokr_w1", "BF16", &[2, 4], 2),
            ("m.lokr_w2_a", "BF16", &[16, 2], 2),
            ("m.lokr_w2_b", "BF16", &[2, 16], 2),
            ("m.alpha", "I64", &[], 8),
        ],
    );
    let lokr_stack = vec![spec(&lokr)];
    let dims = LokrKroneckerDims {
        a: 2,
        b: 16,
        c: 4,
        d: 16,
    };
    assert_eq!(dims.resident_bytes(2), (8 + 256) * 4 + 8 * 2 + 256 * 2);
    assert_eq!(
        dims.resident_bytes(4),
        (8 + 256) * 4 + 256 * 4,
        "at f32 the w1 copy is shared; the transposed w2 is still copied"
    );
    assert_eq!(price(&lokr_stack, 2), Some(dims.resident_bytes(2) + 8));
    let stored = (8 + 32 + 32) * 6;
    assert!(
        price(&lokr_stack, 2).unwrap() > stored,
        "not the stored elements"
    );
    // The low-rank `w1_a·w1_b` and tucker `w2_a [r, b]` / `w2_b [r, d]` forms.
    let shapes = std::collections::HashMap::from([
        ("lokr_w1_a", vec![2usize, 3]),
        ("lokr_w1_b", vec![3usize, 4]),
        ("lokr_t2", vec![3usize, 3, 1, 1]),
        ("lokr_w2_a", vec![3usize, 16]),
        ("lokr_w2_b", vec![3usize, 8]),
    ]);
    assert_eq!(
        LokrKroneckerDims::from_factor_shapes(|n| shapes.get(n).map(Vec::as_slice)),
        Some(LokrKroneckerDims {
            a: 2,
            b: 16,
            c: 4,
            d: 8
        })
    );
    // A module with no w2 cannot be sized: fail closed.
    let broken = root.join("broken.safetensors");
    write_safetensors(&broken, &[("m.lokr_w1", "BF16", &[2, 4], 2)]);
    assert_eq!(price(&[spec(&broken)], 2), None);

    assert_eq!(
        adapter_stack_upcast_resident_bytes(&bf16_stack, AdapterResidencyMode::Folded, 2, cached),
        Some(0)
    );
    assert_eq!(price(&[], 2), Some(0));
    let missing = vec![spec(&root.join("missing.safetensors"))];
    assert_eq!(price(&missing, 2), None);
    let garbage = root.join("garbage.safetensors");
    std::fs::write(&garbage, b"not a safetensors file").unwrap();
    assert_eq!(price(&[spec(&garbage)], 2), None);
}

#[test]
fn adapter_stack_identity_binds_order_path_kind_and_exact_scale() {
    let first = AdapterSpec::new(
        "/adapters/first.safetensors".into(),
        1.0,
        crate::AdapterKind::Lora,
    );
    let second = AdapterSpec::new(
        "/adapters/second.safetensors".into(),
        0.5,
        crate::AdapterKind::Lokr,
    );
    let expected = adapter_stack_identity(&[first.clone(), second.clone()]).unwrap();
    assert_eq!(
        adapter_stack_identity(&[first.clone(), second.clone()]),
        Some(expected.clone())
    );
    assert_ne!(
        adapter_stack_identity(&[second.clone(), first.clone()]),
        Some(expected.clone())
    );

    let mut changed_path = first.clone();
    changed_path.path = "/adapters/other.safetensors".into();
    assert_ne!(
        adapter_stack_identity(&[changed_path, second.clone()]),
        Some(expected.clone())
    );
    let mut changed_kind = first.clone();
    changed_kind.kind = crate::AdapterKind::Lokr;
    assert_ne!(
        adapter_stack_identity(&[changed_kind, second.clone()]),
        Some(expected.clone())
    );
    let mut changed_scale = first;
    changed_scale.scale = f32::from_bits(1.0_f32.to_bits() + 1);
    assert_ne!(
        adapter_stack_identity(&[changed_scale, second]),
        Some(expected)
    );
    assert_eq!(adapter_stack_identity(&[]), None);
}

fn mlx_backend() -> MemoryBackendRealization {
    MemoryBackendRealization::MlxMetal {
        bounded_wired_residency: true,
        lazy_or_mmap_materialization: true,
        explicit_evaluation_and_synchronization: true,
        cache_eviction: true,
    }
}

fn implemented(strategy: MemoryStrategy) -> MemoryStrategyCapability {
    MemoryStrategyCapability {
        strategy,
        support: MemoryStrategySupport::Implemented,
        parameters: MemoryParameterRanges::default(),
    }
}

fn adopted_contract() -> MemoryProviderContract {
    let mut strategies: Vec<_> = MemoryStrategy::ALL.into_iter().map(implemented).collect();
    strategies[2].parameters.decode_tile_edges = vec![512, 384];
    strategies[2].parameters.decode_overlaps = vec![64];
    strategies[3].parameters.attention_chunk_sizes = vec![256, 128];
    strategies[4].parameters.transformer_window_sizes = vec![2, 1];
    MemoryProviderContract {
        phase_facts: None,
        architecture_facts: crate::MemoryArchitectureFacts::default(),
        provider_id: "test-provider".to_owned(),
        backend: mlx_backend(),
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: LoadShape::DeferredMaterialization,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            "test-layout-v1",
            LoadShape::DeferredMaterialization,
        )),
        asset_facts: MemoryAssetFacts::default(),
        runtime: MemoryRuntimeSemantics::default(),
    }
}

fn admitted_context(contract: &MemoryProviderContract) -> MemoryRunContext {
    let calibration = contract.calibration.as_ref().expect("calibration");
    MemoryRunContext {
        reference_resources: None,
        selection: MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        },
        optimization_authority: MemoryOptimizationAuthority::Resident,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: 512,
            height: 512,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 512,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "test".to_owned(),
    }
}

/// An arbitrary well-formed source stamp, deliberately unrelated to anything in this build.
///
/// Before sc-19728 this fixture read the compiled
/// `MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT`, which is why
/// [`MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256`] churned every time an unrelated source
/// file changed. Pinning a literal here is what makes that canonical digest track the
/// serialization contract it is meant to pin, and nothing else.
const DECODE_QUALITY_STAMP: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn decode_quality_policy(contract: &MemoryProviderContract) -> MemoryDecodeGeometryPolicy {
    MemoryDecodeGeometryPolicy {
        quality_abi: MEMORY_DECODE_QUALITY_ABI,
        family: "test-family".to_owned(),
        resolved_route: contract.provider_id.clone(),
        backend: MemoryBackend::Mlx,
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        load_shape: contract.load_shape,
        artifact: MemoryDecodeArtifactIdentity {
            repository: "SceneWorks/test-model".to_owned(),
            revision: "a".repeat(40),
            variant: "q4".to_owned(),
            fingerprint: format!("SceneWorks/test-model@{}:q4", "a".repeat(40)),
        },
        implementation_fingerprint: DECODE_QUALITY_STAMP.to_owned(),
        mode: MemoryMode::TextToImage,
        overlay: None,
        geometry: MemoryGeometry {
            width: 512,
            height: 512,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        use_pid: false,
        tile_edge: 512,
        overlap: 64,
        metric: "max_abs_rgb_u8".to_owned(),
        maximum_error: 48,
        fixtures: [(7, 'a', 'b', 'c', 31), (99, 'd', 'e', 'f', 42)]
            .into_iter()
            .map(
                |(seed, latent, dense, tiled, observed_error)| MemoryDecodeQualityFixture {
                    seed,
                    production_latent_provenance: format!("test-model@revision seed={seed}"),
                    production_latent_sha256: latent.to_string().repeat(64),
                    dense_output_sha256: dense.to_string().repeat(64),
                    tiled_output_sha256: tiled.to_string().repeat(64),
                    observed_error,
                },
            )
            .collect(),
        production_evidence_sha256: String::new(),
        disposition: MemoryDecodeQualityDisposition::Admitted,
    }
    .seal()
}

fn decode_quality_runtime_identity(
    policy: &MemoryDecodeGeometryPolicy,
) -> MemoryDecodeQualityRuntimeIdentity {
    MemoryDecodeQualityRuntimeIdentity {
        artifact: policy.artifact.clone(),
    }
}

#[test]
fn geometry_decode_quality_is_exact_fail_closed_and_self_authenticating() {
    let mut contract = adopted_contract();
    contract.provider_id = "sdxl".to_owned();
    let mut policy = decode_quality_policy(&contract);
    policy.resolved_route = "realvisxl".to_owned();
    policy = policy.seal();
    assert_eq!(
        policy.production_evidence_sha256, MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256,
        "Rust and the correctness-only collector must share one canonical quality ABI"
    );
    assert!(contract
        .adopt_decode_geometry_policies("test-family", vec![policy.clone()])
        .unwrap());
    assert!(contract.lifecycle.decode_tiling);
    let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
    assert_eq!(decode.parameters.decode_tile_edges, [512]);
    assert_eq!(decode.parameters.decode_overlaps, [64]);
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );

    let query = MemoryDecodePolicyQuery {
        tier: policy.tier,
        load_shape: policy.load_shape,
        mode_key: "text_to_image",
        overlay: None,
        geometry: policy.geometry,
        use_pid: false,
    };
    let admitted = contract
        .decode_geometry_policy(query, None)
        .unwrap()
        .expect("policy adopter returns the exact row");
    assert_eq!((admitted.tile_edge, admitted.overlap), (512, 64));
    assert!(contract
        .decode_geometry_policy(
            query,
            Some(MemoryStrategyParameters {
                decode_tile_edge: Some(384),
                decode_overlap: Some(64),
                ..Default::default()
            })
        )
        .unwrap_err()
        .to_string()
        .contains("do not match"));

    let off_grid = MemoryDecodePolicyQuery {
        geometry: MemoryGeometry {
            width: 513,
            ..policy.geometry
        },
        ..query
    };
    assert!(contract
        .decode_geometry_policy(off_grid, None)
        .unwrap_err()
        .to_string()
        .contains("no admitted"));

    contract.strategies[MemoryStrategy::BoundedDecode as usize]
        .parameters
        .decode_geometry_policies[0]
        .fixtures[0]
        .observed_error += 1;
    assert!(contract
        .conformance_errors()
        .iter()
        .any(|error| error.contains("does not bind the canonical")));
}

#[test]
fn load_quality_table_requires_the_exact_resolved_route() {
    let contract = adopted_contract();
    let mut policy = decode_quality_policy(&contract);
    policy.resolved_route = "realvisxl".to_owned();
    policy = policy.seal();

    let unbound = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
        .with_decode_geometry_policies(vec![policy.clone()])
        .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&policy));
    assert!(unbound.route_bound_decode_geometry_policies().is_err());

    let sibling = unbound.clone().with_resolved_route("realvisxl_lightning");
    assert!(sibling.route_bound_decode_geometry_policies().is_err());

    let empty = unbound.clone().with_resolved_route("   ");
    assert!(empty.route_bound_decode_geometry_policies().is_err());

    let exact = unbound.with_resolved_route("realvisxl");
    assert_eq!(
        exact.route_bound_decode_geometry_policies().unwrap(),
        [policy]
    );
}

#[test]
fn load_quality_table_requires_every_artifact_identity_axis() {
    let contract = adopted_contract();
    let mut policy = decode_quality_policy(&contract);
    policy.resolved_route = "realvisxl".to_owned();
    policy = policy.seal();
    let base = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
        .with_resolved_route("realvisxl")
        .with_decode_geometry_policies(vec![policy.clone()]);
    assert!(base
        .route_bound_decode_geometry_policies()
        .unwrap_err()
        .to_string()
        .contains("independently resolved"));

    let exact = decode_quality_runtime_identity(&policy);
    for (label, identity) in [
        ("repository", {
            let mut value = exact.clone();
            value.artifact.repository.push_str("-sibling");
            value
        }),
        ("revision", {
            let mut value = exact.clone();
            value.artifact.revision = "b".repeat(40);
            value
        }),
        ("variant", {
            let mut value = exact.clone();
            value.artifact.variant = "q8".to_owned();
            value
        }),
        ("fingerprint", {
            let mut value = exact.clone();
            value.artifact.fingerprint.push_str("-drift");
            value
        }),
    ] {
        let error = base
            .clone()
            .with_decode_quality_runtime_identity(identity)
            .route_bound_decode_geometry_policies()
            .expect_err(label);
        assert!(
            error
                .to_string()
                .contains("cannot authorize loaded identity"),
            "{label}: {error}"
        );
    }
    assert_eq!(
        base.with_decode_quality_runtime_identity(exact)
            .route_bound_decode_geometry_policies()
            .unwrap(),
        [policy]
    );
}

/// sc-19728: a row measured by a different source closure is **admitted**, not refused.
///
/// This is the observable proof that currency enforcement is gone rather than merely
/// unreachable. Before sc-19728 each of these stamps made
/// [`MemoryProviderContract::adopt_decode_geometry_policies`] return `Err`, because
/// `validate_decode_geometry_policies` compared the row against a constant compiled into this
/// build. `"800d06ac…"` is the stamp on all 69 rows shipped in SceneWorks'
/// `builtin.models.jsonc`; it equals the constant at SceneWorks' current inference pin, which is
/// why tiled decode works there today and would have stopped working — silently, for every one
/// of the 69 — at the next pin bump. `"63c6f89d…"` is what the constant had already drifted to
/// on this branch. Flipping the removed comparison back on turns this test red.
#[test]
fn a_foreign_source_closure_stamp_no_longer_refuses_a_measured_row() {
    for stamp in [
        "800d06acf579a36e604d91955fd6a6852ec70bc39701f7a320f1fdd2bf5ff29d",
        "63c6f89da806e02f098598c595d1e5b46a619d05f3beb269fe40334e60c39cd9",
        &"f".repeat(64),
    ] {
        let mut contract = adopted_contract();
        let mut policy = decode_quality_policy(&contract);
        policy.implementation_fingerprint = stamp.to_owned();
        let policy = policy.seal();

        assert!(
            contract
                .adopt_decode_geometry_policies("test-family", vec![policy.clone()])
                .expect("a measurement stands until it is remeasured"),
            "{stamp}: bounded decode must remain admitted"
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{stamp}: {:?}",
            contract.conformance_errors()
        );

        let query = MemoryDecodePolicyQuery {
            tier: policy.tier,
            load_shape: policy.load_shape,
            mode_key: "text_to_image",
            overlay: None,
            geometry: policy.geometry,
            use_pid: false,
        };
        assert_eq!(
            contract
                .decode_geometry_policy(query, None)
                .unwrap()
                .expect("the row resolves")
                .implementation_fingerprint,
            stamp,
            "the stamp is recorded verbatim for forensics"
        );
    }
}

/// The guard that survived still discriminates: the stamp must be a real digest.
///
/// Recording garbage would defeat the forensic purpose the field was kept for, so shape is
/// enforced even though currency is not. Each of these is rejected for a distinct reason —
/// empty, short, long, uppercase, and non-hex — so the test cannot pass on a length check alone.
#[test]
fn a_malformed_source_closure_stamp_is_still_refused() {
    for malformed in [
        "",
        &"a".repeat(63),
        &"a".repeat(65),
        &"A".repeat(64),
        &"g".repeat(64),
    ] {
        let mut contract = adopted_contract();
        let mut policy = decode_quality_policy(&contract);
        policy.implementation_fingerprint = malformed.to_owned();
        let policy = policy.seal();

        let error = contract
            .adopt_decode_geometry_policies("test-family", vec![policy])
            .expect_err(malformed)
            .to_string();
        assert!(
            error.contains("must be a lowercase SHA-256 stamp"),
            "{malformed:?}: {error}"
        );
    }
}

#[test]
fn exact_tile_pair_is_policy_identity_and_selection_controls_decode_engagement() {
    let mut contract = adopted_contract();
    let first = decode_quality_policy(&contract);
    let mut second = first.clone();
    second.tile_edge = 384;
    second.overlap = 32;
    second = second.seal();
    contract
        .adopt_decode_geometry_policies("test-family", vec![first.clone(), second.clone()])
        .unwrap();
    let query = MemoryDecodePolicyQuery {
        tier: first.tier,
        load_shape: first.load_shape,
        mode_key: "text_to_image",
        overlay: None,
        geometry: first.geometry,
        use_pid: false,
    };
    assert!(contract
        .decode_geometry_policy(query, None)
        .unwrap_err()
        .to_string()
        .contains("multiple tile pairs"));
    let select_pair = |tile_edge, overlap| MemoryStrategyParameters {
        decode_tile_edge: Some(tile_edge),
        decode_overlap: Some(overlap),
        ..Default::default()
    };
    assert_eq!(
        contract
            .decode_geometry_policy(query, Some(select_pair(384, 32)))
            .unwrap()
            .unwrap()
            .production_evidence_sha256,
        second.production_evidence_sha256
    );

    let independent_attention = MemorySelection {
        strategy: MemoryStrategy::BoundedAttention,
        parameters: MemoryStrategyParameters {
            attention_chunk_size: Some(128),
            ..Default::default()
        },
        tier: first.tier,
    };
    contract.validate_selection(&independent_attention).unwrap();
    assert!(!contract.engages_selection(&independent_attention, MemoryStrategy::BoundedDecode));
    let request = contract
        .generation_memory(&independent_attention)
        .expect("optimized request memory");
    assert!(!request.tile_vae_decode);
    assert!(request.chunk_attention);

    let admitted_attention = MemorySelection {
        parameters: MemoryStrategyParameters {
            decode_tile_edge: Some(384),
            decode_overlap: Some(32),
            attention_chunk_size: Some(128),
            ..Default::default()
        },
        ..independent_attention
    };
    contract.validate_selection(&admitted_attention).unwrap();
    assert!(contract.engages_selection(&admitted_attention, MemoryStrategy::BoundedDecode));
    assert!(
        contract
            .generation_memory(&admitted_attention)
            .unwrap()
            .tile_vae_decode
    );

    let decode_without_pair = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: MemoryStrategyParameters::default(),
        tier: first.tier,
    };
    assert!(contract.validate_selection(&decode_without_pair).is_err());

    let mut duplicate = first.clone();
    duplicate.fixtures[0].seed += 1;
    duplicate = duplicate.seal();
    let mut duplicate_contract = adopted_contract();
    assert!(duplicate_contract
        .adopt_decode_geometry_policies("test-family", vec![first, duplicate])
        .unwrap_err()
        .to_string()
        .contains("duplicate decode quality coordinate"));
}

#[test]
fn calibration_free_optimization_requires_explicit_estimate_authority() {
    let mut contract = adopted_contract();
    let mut context = admitted_context(&contract);
    contract.calibration = None;
    context.selection.strategy = MemoryStrategy::StagedResidency;
    context.optimization_authority = MemoryOptimizationAuthority::Estimated;
    assert_eq!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Accept,
        "a caller-admitted conservative estimate authorizes an implemented calibration-free route"
    );

    context.optimization_authority = MemoryOptimizationAuthority::Calibrated;
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("requires explicit estimate authority")
    ));
    context.optimization_authority = MemoryOptimizationAuthority::Resident;
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("no explicit measured or estimated authority")
    ));
}

#[test]
fn loaded_quality_table_selects_the_exact_tier_and_load_shape() {
    let contract = adopted_contract();
    let mut q4 = decode_quality_policy(&contract);
    q4.resolved_route = "realvisxl".to_owned();
    q4 = q4.seal();
    let mut q8 = q4.clone();
    q8.tier.quant = Some(Quant::Q8);
    q8.artifact.variant = "q8".to_owned();
    q8.artifact.fingerprint = format!(
        "{}@{}:{}",
        q8.artifact.repository, q8.artifact.revision, q8.artifact.variant
    );
    q8 = q8.seal();

    let spec = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
        .with_resolved_route("realvisxl")
        .with_decode_geometry_policies(vec![q4.clone(), q8.clone()])
        .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&q4))
        .with_quant(Quant::Q4)
        .with_load_shape(LoadShape::DeferredMaterialization);
    assert_eq!(
        spec.decode_geometry_policies_for_loaded_contract(
            MemoryBackend::Mlx,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        )
        .unwrap(),
        [q4.clone()]
    );
    let q8_spec = spec
        .clone()
        .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&q8));
    assert_eq!(
        q8_spec
            .decode_geometry_policies_for_loaded_contract(
                MemoryBackend::Mlx,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
            )
            .unwrap(),
        [q8]
    );
    let eager = spec.with_load_shape(LoadShape::EagerMaterialization);
    assert!(
        eager.decode_geometry_policy_authoritative,
        "an exact projection that yields no row must retain the caller-declared semantic scope"
    );
    assert_eq!(
        eager
            .decode_geometry_policies_for_loaded_contract(
                MemoryBackend::Mlx,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            )
            .unwrap(),
        Vec::<MemoryDecodeGeometryPolicy>::new()
    );
}

#[test]
fn loaded_quality_table_cannot_mix_semantic_families() {
    let mut contract = adopted_contract();
    let first = decode_quality_policy(&contract);
    let mut second = first.clone();
    second.family = "another-family".to_owned();
    second.geometry.width = 768;
    second = second.seal();

    let error = contract
        .adopt_decode_geometry_policies("test-family", vec![first, second])
        .unwrap_err();
    assert!(error.to_string().contains("does not match expected family"));
}

#[test]
fn empty_decode_quality_table_preserves_legacy_parameter_fallback() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    assert!(contract
        .decode_geometry_policy(
            MemoryDecodePolicyQuery {
                tier: context.selection.tier,
                load_shape: context.load_shape,
                mode_key: context.mode.as_key(),
                overlay: context.overlay.as_deref(),
                geometry: context.geometry,
                use_pid: context.use_pid,
            },
            None,
        )
        .unwrap()
        .is_none());
}

#[test]
fn declared_empty_decode_quality_scope_fails_closed_without_blocking_independent_rungs() {
    let mut contract = adopted_contract();
    contract.decode_geometry_policy_authoritative = true;
    let context = admitted_context(&contract);
    let query = MemoryDecodePolicyQuery {
        tier: context.selection.tier,
        load_shape: context.load_shape,
        mode_key: context.mode.as_key(),
        overlay: context.overlay.as_deref(),
        geometry: context.geometry,
        use_pid: context.use_pid,
    };
    let error = contract
        .decode_geometry_policy(query, None)
        .expect_err("declared semantic scope cannot fall back to route-blind decode");
    assert!(error.to_string().contains("declared quality scope"));

    let independent = MemorySelection {
        strategy: MemoryStrategy::BoundedAttention,
        parameters: MemoryStrategyParameters {
            attention_chunk_size: Some(256),
            ..Default::default()
        },
        tier: context.selection.tier,
    };
    assert!(!contract.engages_selection(&independent, MemoryStrategy::BoundedDecode));
    assert!(contract.engages_selection(&independent, MemoryStrategy::BoundedAttention));
    let memory = contract
        .generation_memory(&independent)
        .expect("optimized selection memory controls");
    assert!(!memory.tile_vae_decode);
    assert!(memory.chunk_attention);
}

#[test]
fn behavior_fixture_preserves_exact_reference_cardinality() {
    let contract = adopted_contract();
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: Some(Quant::Q4),
        component_precision_floors: &[],
    };
    let context = standard_memory_behavior_context(
        &contract,
        MemoryStrategy::Resident,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 4,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )
    .expect("behavior context");
    let fixture = crate::MemoryBehaviorFixture::new(context);
    assert!(fixture.context.has_reference);
    assert_eq!(fixture.context.geometry.reference_count, 4);
    assert_eq!(fixture.request.conditioning.len(), 4);
    assert!(fixture
        .request
        .conditioning
        .iter()
        .all(|conditioning| matches!(conditioning, crate::Conditioning::Reference { .. })));
}

/// Build a weights-free behavior context whose geometry declares `frames`, leaving every other
/// axis at [`standard_memory_behavior_context`]'s values.
fn behavior_context_with_frames(frames: u32) -> MemoryRunContext {
    let contract = adopted_contract();
    let mut context = standard_memory_behavior_context(
        &contract,
        MemoryStrategy::Resident,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )
    .expect("behavior context");
    context.geometry.frames = frames;
    context
}

/// sc-19591. A multi-frame geometry reaches the request, so a provider re-grading the request
/// against the geometry the same fixture admitted sees the same count on both sides. Before
/// propagation `request.frames` was `GenerationRequest::default()`'s `None` for every geometry,
/// which is why the expectation here is the *input* frame count rather than a literal restated
/// on both sides, and why several counts are walked — no single constant satisfies them all,
/// and none of them is the default.
#[test]
fn behavior_fixture_propagates_a_multi_frame_geometry() {
    let default_frames = crate::GenerationRequest::default().frames;
    assert_eq!(
        default_frames, None,
        "the defaulted request states no frames"
    );
    for frames in [17u32, 81, 124, 145, 345] {
        let fixture = crate::MemoryBehaviorFixture::new(behavior_context_with_frames(frames));
        assert_eq!(fixture.request.frames, Some(frames));
        assert_eq!(
            fixture.request.frames,
            Some(fixture.context.geometry.frames)
        );
        assert_ne!(fixture.request.frames, default_frames);
    }
}

/// sc-19591. A one-frame geometry — [`standard_memory_behavior_context`]'s own value, and what
/// every image provider on the ladder declares — states that one frame explicitly. `Some(1)`
/// and `None` resolve identically for those providers, since each reads the field as
/// `request.frames.unwrap_or(<its own default>)` and theirs is 1; stating it keeps the fixture
/// self-consistent for a provider whose default is not 1.
#[test]
fn behavior_fixture_propagates_a_single_frame_geometry() {
    let fixture = crate::MemoryBehaviorFixture::new(behavior_context_with_frames(1));
    assert_eq!(fixture.context.geometry.frames, 1);
    assert_eq!(fixture.request.frames, Some(1));
}

/// sc-19591. Zero frames states no clip length at all and no provider can render it, so it maps
/// back to the unstated `None` rather than to an explicit zero-frame request that would ask
/// every provider for something unrenderable.
#[test]
fn behavior_fixture_leaves_a_zero_frame_geometry_unstated() {
    let fixture = crate::MemoryBehaviorFixture::new(behavior_context_with_frames(0));
    assert_eq!(fixture.context.geometry.frames, 0);
    assert_eq!(fixture.request.frames, None);
}

#[test]
fn default_safety_check_rejects_mutated_calibration_handshake() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    assert_eq!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Accept
    );

    let mut wrong_fingerprint = context.clone();
    wrong_fingerprint.calibration_fingerprint.push_str("-stale");
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &wrong_fingerprint),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("calibration handshake mismatch")
    ));

    let mut wrong_shape = context.clone();
    wrong_shape.load_shape = LoadShape::EagerMaterialization;
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &wrong_shape),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("calibration handshake mismatch")
    ));

    let mut wrong_abi = context;
    wrong_abi.calibration_abi += 1;
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &wrong_abi),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("calibration handshake mismatch")
    ));
}

#[test]
fn calibration_identity_load_shape_must_match_contract() {
    let mut contract = adopted_contract();
    contract.calibration.as_mut().unwrap().load_shape = LoadShape::EagerMaterialization;
    assert!(contract
        .conformance_errors()
        .iter()
        .any(|error| error.contains("calibration load shape")));
}

#[test]
fn standard_safety_check_runs_tier_route_and_budget_gates() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    let loaded_tier = context.selection.tier;
    assert_eq!(
        standard_memory_strategy_safety_check(&contract, &context, Some(loaded_tier), None),
        MemorySafetyDecision::Accept
    );
    let matching_spec =
        crate::LoadSpec::new(crate::WeightsSource::Dir("/weights".into())).with_quant(Quant::Q4);
    assert_eq!(
        default_registered_memory_strategy_safety_check(&matching_spec, &contract, &context),
        MemorySafetyDecision::Accept
    );
    let wrong_spec = matching_spec.with_quant(Quant::Q8);
    assert!(matches!(
        default_registered_memory_strategy_safety_check(&wrong_spec, &contract, &context),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("does not match loaded tier")
    ));

    let mut wrong_tier = context.clone();
    wrong_tier.selection.tier.quant = Some(Quant::Q8);
    assert!(matches!(
        standard_memory_strategy_safety_check(
            &contract,
            &wrong_tier,
            Some(loaded_tier),
            None
        ),
        MemorySafetyDecision::Reject { reason } if reason.contains("does not match loaded tier")
    ));

    let route_gate = || Err(Error::Unsupported("route refused".to_owned()));
    assert!(matches!(
        standard_memory_strategy_safety_check(
            &contract,
            &context,
            Some(loaded_tier),
            Some(&route_gate)
        ),
        MemorySafetyDecision::Reject { reason } if reason.contains("route refused")
    ));

    let mut invalid_reclaim_credit = context.clone();
    invalid_reclaim_credit.budget.committed_bytes = 10;
    invalid_reclaim_credit.budget.reclaimable_bytes = 11;
    assert!(matches!(
        standard_memory_strategy_safety_check(
            &contract,
            &invalid_reclaim_credit,
            Some(loaded_tier),
            None
        ),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("reclaim credit must be a subset")
    ));

    let mut over_budget = context;
    over_budget.predicted_peak_bytes = 1025;
    assert!(matches!(
        standard_memory_strategy_safety_check(
            &contract,
            &over_budget,
            Some(loaded_tier),
            None
        ),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("incremental live demand 1025 exceeds effective budget 1024")
    ));
}

fn pid_contract() -> MemoryProviderContract {
    let mut contract = adopted_contract();
    contract.strategies[2].parameters.decode_tile_edges = vec![512, 2048];
    contract.strategies[2].parameters.decode_overlaps = vec![64, 256];
    contract.pid_decode_routes = Some(MemoryPidDecodeRoutes {
        native: MemoryDecodeRouteDomain {
            tile_edges: vec![512],
            tile_overlap: 64,
        },
        pid: MemoryDecodeRouteDomain {
            tile_edges: vec![2048],
            tile_overlap: 256,
        },
    });
    contract
}

#[test]
fn pid_decode_routes_must_be_disjoint_and_match_the_published_ranges() {
    let valid = pid_contract();
    assert!(valid.conformance_errors().is_empty());

    let mut overlapping = valid.clone();
    overlapping.pid_decode_routes.as_mut().unwrap().pid = MemoryDecodeRouteDomain {
        tile_edges: vec![512],
        tile_overlap: 64,
    };
    let errors = overlapping.conformance_errors();
    assert!(errors
        .iter()
        .any(|error| error.contains("must be disjoint")));

    let mut under_declared = valid;
    under_declared.strategies[2]
        .parameters
        .decode_tile_edges
        .retain(|edge| *edge != 2048);
    let errors = under_declared.conformance_errors();
    assert!(errors
        .iter()
        .any(|error| error.contains("must equal the native/PiD route union")));
}

#[test]
fn compatibility_default_never_advertises_an_optimized_rung() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<MemoryAssetFacts>();
    let legacy_literal = MemoryAssetFacts {
        base_bytes: 1,
        conditioning_bytes: 2,
        transformer_bytes: 3,
        decoder_bytes: 4,
        overlay_bytes: 5,
    };
    assert_eq!(legacy_literal.overlay_bytes, 5);

    let contract = MemoryProviderContract::compatibility_default("legacy", mlx_backend());
    assert!(contract.conformance_errors().is_empty());
    assert!(contract.calibration.is_none());
    assert!(contract.formula.resident_components().is_empty());
    assert_eq!(contract.asset_facts.overlay_bytes, 0);
    let prediction = contract.predicted_peak_from_base(123);
    assert_eq!(prediction.predicted_peak_bytes(), 123);
    assert_eq!(prediction.unattributed_bytes, 123);
    assert!(prediction.components.is_empty());
    for strategy in MemoryStrategy::ALL {
        let support = &contract.capability(strategy).unwrap().support;
        if strategy == MemoryStrategy::Resident {
            assert_eq!(support, &MemoryStrategySupport::Implemented);
        } else {
            assert_eq!(support, &MemoryStrategySupport::Missing);
        }
    }

    let mut context = admitted_context(&adopted_contract());
    assert_eq!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Accept,
        "calibration-free compatibility contracts preserve resident admission"
    );
    context.selection.strategy = MemoryStrategy::StagedResidency;
    assert!(matches!(
        default_memory_strategy_safety_check(&contract, &context),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("has no explicit measured or estimated authority")
    ));

    let mut mutated = contract.clone();
    mutated
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .unwrap()
        .support = MemoryStrategySupport::Implemented;
    assert!(matches!(
        default_memory_strategy_safety_check(&mutated, &context),
        MemorySafetyDecision::Reject { reason }
            if reason.contains("has no explicit measured or estimated authority")
    ));
    context.optimization_authority = MemoryOptimizationAuthority::Estimated;
    assert_eq!(
        default_memory_strategy_safety_check(&mutated, &context),
        MemorySafetyDecision::Accept,
        "explicitly authorized conservative estimates preserve calibration-free optimized fallback"
    );
}

#[test]
fn prompt_upsampler_is_a_separate_once_charged_auxiliary_network() {
    let mut contract = adopted_contract();
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: 60,
        conditioning_bytes: 10,
        transformer_bytes: 40,
        decoder_bytes: 10,
        overlay_bytes: 0,
    };
    assert_eq!(contract.total_resident_bytes(), 60);
    contract.asset_facts.overlay_bytes = 37;
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: vec![MemoryResidentComponent {
            id: "prompt_upsampler".into(),
            kind: MemoryComponentKind::PromptUpsampler,
            resident_bytes: 37,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }],
    };
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
    assert_eq!(contract.asset_facts.base_bytes, 60);
    assert_eq!(contract.asset_facts.conditioning_bytes, 10);
    assert_eq!(contract.auxiliary_resident_bytes(), 37);
    assert_eq!(contract.total_resident_bytes(), 97);
    let breakdown = contract.decompose_predicted_peak(1000);
    assert_eq!(breakdown.unattributed_bytes, 963);
    assert_eq!(breakdown.components.len(), 1);
    assert_eq!(
        breakdown.components[0].kind,
        MemoryComponentKind::PromptUpsampler
    );
}

// --- SC-18665: intra-transformer evictable sub-stacks -------------------------------------

/// A contract carrying only `WholeRender` declarations reads **byte-identically** through every
/// accessor SC-18665 added, so adopting the field cannot move a provider that did not opt in.
///
/// The auxiliary component here declares 7 B against a 40 B transformer, and the assertions
/// name those literals rather than a default: comparing `evicted_component_bytes()` with `0` on
/// a contract that declares no components at all would pass with the whole feature deleted.
#[test]
fn a_whole_render_declaration_leaves_every_derived_quantity_unmoved() {
    let mut contract = adopted_contract();
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: 60,
        conditioning_bytes: 10,
        transformer_bytes: 40,
        decoder_bytes: 10,
        overlay_bytes: 7,
    };
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: vec![MemoryResidentComponent {
            id: "control".to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: 7,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }],
    };
    assert!(contract.conformance_errors().is_empty());

    let component = &contract.resident_components()[0];
    assert_eq!(component.steady_state_bytes(), 7, "nothing is dropped");
    assert_eq!(component.evicted_bytes(), 0);
    assert_eq!(contract.evicted_component_bytes(), 0);
    assert_eq!(
        contract.steady_state_transformer_bytes(),
        40,
        "the transformer charge is the declared 40 B, untouched by an auxiliary component"
    );
    assert_eq!(contract.auxiliary_resident_bytes(), 7);
    assert_eq!(contract.total_resident_bytes(), 67);
    assert_eq!(
        contract.decompose_predicted_peak(1_000).unattributed_bytes,
        993
    );
}

/// An intra-transformer sub-stack removes its **net** bytes from the transformer charge, and
/// only from there: the aggregate `asset_facts` fields and the auxiliary accounting are
/// untouched, because a sub-stack is not an auxiliary network.
#[test]
fn a_precomputed_sub_stack_reduces_only_the_steady_state_transformer_charge() {
    let mut contract = adopted_contract();
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: 160,
        conditioning_bytes: 10,
        transformer_bytes: 140,
        decoder_bytes: 10,
        overlay_bytes: 0,
    };
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![MemoryFormulaVariable::AssetBytes],
        resident_components: vec![MemoryResidentComponent {
            id: "adaln".to_owned(),
            kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            resident_bytes: 100,
            bounded_by: None,
            residency: MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in: MemoryPhase::Denoise,
                retained_bytes: 30,
                evidence: "a measurement".to_owned(),
            },
        }],
    };
    assert_eq!(contract.conformance_errors(), Vec::<String>::new());

    let component = &contract.resident_components()[0];
    assert_eq!(component.steady_state_bytes(), 30);
    assert_eq!(component.evicted_bytes(), 70, "net, not the gross 100");
    assert_eq!(contract.evicted_component_bytes(), 70);
    assert_eq!(contract.steady_state_transformer_bytes(), 140 - 70);
    // A sub-stack is inside the base-model total, so none of this may touch the overlay legs.
    assert_eq!(contract.auxiliary_resident_bytes(), 0);
    assert_eq!(contract.total_resident_bytes(), 160);
    assert_eq!(contract.asset_facts.transformer_bytes, 140);
}

/// Every SC-18665 conformance rule, mutated **one at a time** off a known-good contract, so
/// each guard is shown to detect its own breakage rather than the set detecting something.
#[test]
fn each_sub_stack_eviction_rule_is_independently_detected() {
    fn good() -> MemoryProviderContract {
        let mut contract = adopted_contract();
        contract.asset_facts = MemoryAssetFacts {
            base_bytes: 160,
            conditioning_bytes: 10,
            transformer_bytes: 140,
            decoder_bytes: 10,
            overlay_bytes: 0,
        };
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![MemoryFormulaVariable::AssetBytes],
            resident_components: vec![MemoryResidentComponent {
                id: "adaln".to_owned(),
                kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
                resident_bytes: 100,
                bounded_by: None,
                residency: MemoryComponentResidency::PrecomputedThenEvicted {
                    precomputed_in: MemoryPhase::Denoise,
                    retained_bytes: 30,
                    evidence: "a measurement".to_owned(),
                },
            }],
        };
        contract
    }
    fn residency(contract: &mut MemoryProviderContract) -> &mut MemoryComponentResidency {
        match &mut contract.formula {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                resident_components,
                ..
            } => &mut resident_components[0].residency,
            other => panic!("the control declares a component envelope, got {other:?}"),
        }
    }
    assert!(
        good().conformance_errors().is_empty(),
        "the control must conform, or every mutation below is vacuous"
    );

    /// One named mutation, the error substring it must produce, and the edit itself.
    type SubStackMutation = (&'static str, &'static str, fn(&mut MemoryProviderContract));

    let mutations: Vec<SubStackMutation> = vec![
        (
            "retaining everything it materialized",
            "excludes nothing",
            |c| {
                if let MemoryComponentResidency::PrecomputedThenEvicted { retained_bytes, .. } =
                    residency(c)
                {
                    *retained_bytes = 100;
                }
            },
        ),
        ("an unevidenced eviction", "no verification evidence", |c| {
            if let MemoryComponentResidency::PrecomputedThenEvicted { evidence, .. } = residency(c)
            {
                *evidence = "   ".to_owned();
            }
        }),
        (
            "a precompute phase the provider does not run",
            "not a declared lifecycle phase",
            |c| {
                // The formula's own phase list is left alone: what this mutates is the
                // provider's declared lifecycle, which is what the rule reads.
                c.lifecycle.phases.retain(|p| *p != MemoryPhase::Denoise);
            },
        ),
        (
            "a sub-stack larger than the stack containing it",
            "cannot exceed the stack that contains it",
            |c| c.asset_facts.transformer_bytes = 99,
        ),
    ];

    for (name, expected, mutate) in mutations {
        let mut contract = good();
        mutate(&mut contract);
        let errors = contract.conformance_errors();
        assert!(
            errors.iter().any(|error| error.contains(expected)),
            "conformance must reject {name} with an error naming {expected:?}, got {errors:?}"
        );
    }
}

/// The containment rule is skipped, not tripped, by a weights-free fixture: an
/// architecture-derived sub-stack against zero asset facts is a legitimate declaration, and if
/// this errored no adopting provider could publish a catalog-conformance fixture.
#[test]
fn a_weights_free_fixture_may_declare_a_sub_stack_against_zero_asset_facts() {
    let mut contract = adopted_contract();
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![MemoryFormulaVariable::AssetBytes],
        resident_components: vec![MemoryResidentComponent {
            id: "adaln".to_owned(),
            kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            // Obviously synthetic round numbers. These used to be MiniMax-H3's real 26.02 GB
            // stack beside `3_890_073_600` — a mis-transcription of that provider's
            // `ADALN_MODULATION_TABLE_MAX_BYTES` (3_870_720_000). A near-miss copy of a real
            // provider's figures reads as authoritative and is the harder defect: gen-core owns
            // the rule, not any provider's bytes, and nothing here grades a magnitude.
            resident_bytes: 4_000_000_000,
            bounded_by: None,
            residency: MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in: MemoryPhase::Denoise,
                retained_bytes: 1_000_000_000,
                evidence: "a measurement".to_owned(),
            },
        }],
    };
    assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
    assert_eq!(contract.conformance_errors(), Vec::<String>::new());
    // …and the accessor saturates rather than wrapping when the fixture charges nothing.
    assert_eq!(contract.steady_state_transformer_bytes(), 0);
}

/// A sub-stack is not auxiliary. If it were, every adopting provider would have to move its
/// bytes into `overlay_bytes` — where they would be charged a second time, on top of the
/// transformer that already contains them.
#[test]
fn a_transformer_sub_stack_is_not_auxiliary() {
    assert!(!MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit).is_auxiliary());
    assert!(!MemoryComponentKind::Transformer(TransformerComponent::Dit).is_auxiliary());
    for auxiliary in [
        MemoryComponentKind::ControlBranch,
        MemoryComponentKind::AdapterStack,
        MemoryComponentKind::IpAdapter,
        MemoryComponentKind::IdentityEncoder,
        MemoryComponentKind::ReferenceEncoder,
    ] {
        assert!(
            auxiliary.is_auxiliary(),
            "{auxiliary:?} must stay auxiliary"
        );
    }
}

#[test]
fn asset_facts_reject_overlay_contamination_and_ambiguous_overlay_formulas() {
    let mut contract = adopted_contract();
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: 60,
        conditioning_bytes: 10,
        transformer_bytes: 40,
        decoder_bytes: 10,
        overlay_bytes: 7,
    };
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: vec![MemoryResidentComponent {
            id: "control".to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: 7,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }],
    };
    assert!(contract.conformance_errors().is_empty());

    let mut contaminated = contract.clone();
    contaminated.asset_facts.base_bytes += contaminated.asset_facts.overlay_bytes;
    assert!(contaminated
        .conformance_errors()
        .iter()
        .any(|error| error.contains("exclude overlay_bytes")));

    let mut ambiguous = contract;
    ambiguous.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: ambiguous.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    assert!(ambiguous
        .conformance_errors()
        .iter()
        .any(|error| error.contains("typed auxiliary resident components")));

    let mut candle_krea_shape = adopted_contract();
    candle_krea_shape.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: candle_krea_shape.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    assert!(candle_krea_shape.conformance_errors().is_empty());
}

#[test]
fn optimized_evidence_requires_every_dimension_and_matching_fingerprint() {
    let contract = adopted_contract();
    let mut evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            model_family: "test-family".to_owned(),
            resolved_route: "test".to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            load_shape: contract.load_shape,
            mode: MemoryMode::TextToImage,
            reference_shape: MemoryReferenceShape::None,
            overlay: None,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            frames_per_second: None,
            strategy: MemoryStrategy::BoundedDecode,
            engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
        },
        conformance: MemoryConformanceState::Verified,
        dimensions: MemoryEvidenceDimensions::VERIFIED,
        calibration_abi: MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "test-layout-v1".to_owned(),
        sceneworks_revision: "scene".to_owned(),
        inference_revision: "inference".to_owned(),
        harness_version: "v1".to_owned(),
        predicted_peak_bytes: 10,
        observed_peak_bytes: Some(9),
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::Passed,
    };
    assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

    let canonical_composition = evidence.key.engaged_composition.clone();
    evidence.key.engaged_composition = vec![MemoryStrategy::Resident, MemoryStrategy::Resident];
    assert!(evidence
        .validation_errors()
        .iter()
        .any(|error| error.contains("canonical strategy set")));
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Invalid)
    );
    evidence.key.engaged_composition = canonical_composition;

    evidence.key.load_shape = LoadShape::EagerMaterialization;
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::FingerprintMismatch)
    );
    evidence.key.load_shape = contract.load_shape;

    let mut eager_contract = contract.clone();
    eager_contract.load_shape = LoadShape::EagerMaterialization;
    eager_contract.calibration.as_mut().unwrap().load_shape = LoadShape::EagerMaterialization;
    assert_eq!(
        evidence.optimized_eligibility(&eager_contract),
        Err(MemoryEvidenceVerdict::FingerprintMismatch)
    );

    evidence.calibration_abi = 1;
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::FingerprintMismatch)
    );
    evidence.calibration_abi = MEMORY_CALIBRATION_ABI;

    let mut changed_composition = contract.clone();
    changed_composition.additional_prerequisites.push((
        MemoryStrategy::BoundedDecode,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    assert_eq!(
        evidence.optimized_eligibility(&changed_composition),
        Err(MemoryEvidenceVerdict::CompositionMismatch)
    );

    evidence.dimensions.current_environment_verification = MemoryEvidenceVerdict::Stale;
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Stale)
    );
    evidence.dimensions = MemoryEvidenceDimensions::VERIFIED;
    evidence.calibration_fingerprint = "old-layout".to_owned();
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::FingerprintMismatch)
    );
}

#[test]
fn engaged_composition_tracks_a_non_default_prerequisite_edge() {
    let mut contract = adopted_contract();
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedDecode),
        vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode,]
    );

    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedDecode,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedDecode),
        vec![
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ]
    );
}

#[test]
fn effective_budget_reserves_headroom_from_free_and_reclaimable_memory() {
    let budget = MemoryBudget {
        total_bytes: 100,
        committed_bytes: 80,
        reclaimable_bytes: 50,
        reserved_headroom_bytes: 10,
    };
    assert_eq!(budget.effective_bytes(), 60);
    assert!(budget.fits(60));
    assert!(!budget.fits(61));

    assert_eq!(
        MemoryBudget {
            reclaimable_bytes: 0,
            ..budget
        }
        .effective_bytes(),
        10
    );
}

#[test]
fn fits_consumes_incremental_demand_not_the_absolute_request_peak() {
    let budget = MemoryBudget {
        total_bytes: 100,
        committed_bytes: 40,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 10,
    };
    let absolute_request_peak = 80;
    let request_resident_already_committed = 30;
    let incremental_live_demand = absolute_request_peak - request_resident_already_committed;

    assert_eq!(budget.effective_bytes(), 50);
    assert!(budget.fits(incremental_live_demand));
    assert_eq!(budget.required_total_bytes(incremental_live_demand), 100);
    assert!(!budget.fits(absolute_request_peak));
    assert_eq!(budget.required_total_bytes(absolute_request_peak), 130);
    assert!(!budget.fits(incremental_live_demand + 1));
}

#[test]
fn effective_budget_saturates_at_zero_and_total_minus_headroom() {
    assert_eq!(
        MemoryBudget {
            total_bytes: 100,
            committed_bytes: 110,
            reclaimable_bytes: 5,
            reserved_headroom_bytes: 10,
        }
        .effective_bytes(),
        0
    );
    assert_eq!(
        MemoryBudget {
            total_bytes: 100,
            committed_bytes: 0,
            reclaimable_bytes: u64::MAX,
            reserved_headroom_bytes: 10,
        }
        .effective_bytes(),
        90
    );
    assert_eq!(
        MemoryBudget {
            total_bytes: 100,
            committed_bytes: 0,
            reclaimable_bytes: u64::MAX,
            reserved_headroom_bytes: 101,
        }
        .effective_bytes(),
        0
    );
}

#[test]
fn effective_budget_preserves_an_overcommit_deficit_before_reclamation() {
    let budget = MemoryBudget {
        total_bytes: 100,
        committed_bytes: 110,
        reclaimable_bytes: 50,
        reserved_headroom_bytes: 10,
    };
    assert_eq!(budget.effective_bytes(), 30);
    assert!(budget.fits(30));
    assert!(!budget.fits(31));

    assert_eq!(
        MemoryBudget {
            committed_bytes: 160,
            ..budget
        }
        .effective_bytes(),
        0,
        "an overcommit larger than all reclaimable memory must leave no request budget"
    );
}

#[test]
fn evidence_axes_have_stable_protocol_keys_without_stringly_runtime_identity() {
    assert_eq!(MemoryBackend::Mlx.as_key(), "mlx");
    assert_eq!(MemoryBackend::Candle.as_key(), "candle");
    assert_eq!(MemoryMode::TextToImage.as_key(), "text_to_image");
    assert_eq!(MemoryMode::ImageToImage.as_key(), "image_to_image");
    assert_eq!(MemoryMode::Edit.as_key(), "edit");
    assert_eq!(MemoryMode::Other("custom".to_owned()).as_key(), "custom");
}

#[test]
fn provider_contract_requires_hooks_for_implemented_rungs() {
    let mut contract = adopted_contract();
    contract.lifecycle.attention_chunking = false;
    assert!(contract
        .conformance_errors()
        .iter()
        .any(|error| error.contains("attention_chunking")));
}

#[test]
fn provider_contract_rejects_structural_absence_when_an_implementation_hook_exists() {
    let mut contract = adopted_contract();
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedAttention)
        .expect("bounded attention capability")
        .support = MemoryStrategySupport::StructurallyNotApplicable {
        reason: "mutated contradiction".to_owned(),
    };
    assert!(contract.conformance_errors().iter().any(|error| {
        error.contains("BoundedAttention cannot be StructurallyNotApplicable")
            && error.contains("implementation hook")
    }));
}

/// **SC-16090.** A rung-4 realization that converts formats per window may ship, but not silently:
/// it must name the story that removes it, and say what it converts. The arms are asserted against
/// each other so the test cannot pass with the rule deleted (an unconditional error would fail the
/// declared arm, an absent one the undeclared arms), and the unrelated-rung arm pins that the rule
/// is scoped to rung 4.
#[test]
fn a_nonconforming_window_materialization_must_name_its_owner_story() {
    let converting = |converts: &str, owner_story: &str| MemoryBackendRealization::CandleCuda {
        device_residency: true,
        host_backed_weights: true,
        host_to_device_block_materialization: true,
        block_materialization: MemoryWindowMaterialization::HostFormatConversion {
            converts: converts.to_owned(),
            owner_story: owner_story.to_owned(),
        },
    };
    const CONVERTS: &str = "MLX affine triple -> GGML Q4_1 per projection per window";
    let errors_matching = |contract: &MemoryProviderContract, needle: &str| {
        contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains(needle))
    };

    let mut unnamed = adopted_contract();
    unnamed.backend = converting(CONVERTS, "");
    assert!(
        errors_matching(&unnamed, "owner_story"),
        "a per-window format conversion with no owner story must be a conformance error"
    );
    // Whitespace is not a name — otherwise the escape hatch is one space wide.
    let mut blank = adopted_contract();
    blank.backend = converting(CONVERTS, "   ");
    assert!(
        errors_matching(&blank, "owner_story"),
        "a blank owner_story must not satisfy it"
    );

    // `converts` is the pointer a reviewer follows to the code, so it is required too. Asserted
    // separately from `owner_story` so satisfying one cannot mask the other.
    let mut unexplained = adopted_contract();
    unexplained.backend = converting("", "sc-16096");
    assert!(
        errors_matching(&unexplained, "say what it converts"),
        "an unexplained conversion must be an error even with an owner story named"
    );

    let mut named = adopted_contract();
    named.backend = converting(CONVERTS, "sc-16096");
    assert!(
            named.conformance_errors().is_empty(),
            "a fully declared conversion is the escape hatch, and must leave the contract clean — \
             a contract-level error would cost the shared selector rungs 0-4, not just rung 4: {:?}",
            named.conformance_errors()
        );

    // Scoped to rung 4: the same realization with rung 4 unimplemented is silent, because nothing
    // is materializing windows.
    let mut without_rung_four = adopted_contract();
    without_rung_four.backend = converting("", "");
    without_rung_four.strategies[4].support = MemoryStrategySupport::Missing;
    without_rung_four.strategies[4].parameters = MemoryParameterRanges::default();
    assert!(
        without_rung_four.conformance_errors().is_empty(),
        "the rule must not fire for a provider that does not implement rung 4: {:?}",
        without_rung_four.conformance_errors()
    );
}

/// **SC-16090.** The obligation is asked backend-neutrally and answered per realization: one shared
/// question, not a forked ladder. MLX's mapped materialization satisfies it structurally; an MLX
/// realization that declares neither lazy nor mmap materialization is **unstated** — refused, but
/// with the fix it can actually perform, since `MlxMetal` has no `owner_story` to name.
#[test]
fn window_materialization_is_asked_of_every_backend_and_not_granted_by_backend_identity() {
    assert!(mlx_backend()
        .window_materialization()
        .is_some_and(MemoryWindowMaterialization::is_conforming));

    let unstated = MemoryBackendRealization::MlxMetal {
        bounded_wired_residency: true,
        lazy_or_mmap_materialization: false,
        explicit_evaluation_and_synchronization: true,
        cache_eviction: true,
    };
    assert_eq!(
        unstated.window_materialization(),
        None,
        "being MLX is not itself evidence that a window is a mapped read — but nor is it a \
             declared host conversion, which would assert something unmeasured"
    );

    // Unstated is refused, and the message must name the fix that exists (declare the mechanism),
    // not the one MlxMetal cannot perform (name an owner_story it has no field for).
    let mut under_declared = adopted_contract();
    under_declared.backend = unstated.clone();
    let errors = under_declared.conformance_errors();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("lazy_or_mmap_materialization")),
        "the refusal must name the achievable fix: {errors:?}"
    );
    assert!(
        !errors.iter().any(|error| error.contains("owner_story")),
        "must not demand an owner_story from a realization with no such field: {errors:?}"
    );

    let candle_conforming = MemoryBackendRealization::CandleCuda {
        device_residency: true,
        host_backed_weights: true,
        host_to_device_block_materialization: true,
        block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
    };
    assert!(
        candle_conforming
            .window_materialization()
            .is_some_and(MemoryWindowMaterialization::is_conforming),
        "a host-to-device copy of device-format bytes conforms — the rung is not MLX-only"
    );
    assert_eq!(candle_conforming.backend_id(), "candle");

    // The contract-level accessor is the seam a caller uses (mirroring how `engages` is layered),
    // so it must agree with the realization rather than being a second source of truth.
    let mut contract = adopted_contract();
    contract.backend = candle_conforming.clone();
    assert_eq!(
        contract.window_materialization(),
        candle_conforming.window_materialization()
    );
    assert!(
        contract.conformance_errors().is_empty(),
        "a conforming candle realization needs no escape hatch: {:?}",
        contract.conformance_errors()
    );
}

#[test]
fn selection_preserves_tier_and_rejects_unknown_parameter_values() {
    let contract = adopted_contract();
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: Some(Quant::Q4),
        component_precision_floors: &[],
    };
    let mut selection = MemorySelection {
        strategy: MemoryStrategy::BoundedDecode,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(64),
            ..Default::default()
        },
        tier,
    };
    contract.validate_selection(&selection).unwrap();
    assert_eq!(selection.tier, tier);

    selection.parameters.decode_tile_edge = Some(320);
    assert!(matches!(
        contract.validate_selection(&selection),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn cumulative_parameters_are_required_and_irrelevant_parameters_are_rejected() {
    let contract = adopted_contract();
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    };
    let mut selection = MemorySelection {
        strategy: MemoryStrategy::BoundedAttention,
        parameters: MemoryStrategyParameters {
            stage_residency: None,
            decode_tile_edge: Some(512),
            decode_overlap: Some(64),
            attention_chunk_size: Some(128),
            transformer_window_size: None,
            transformer_window_component: None,
        },
        tier,
    };
    contract.validate_selection(&selection).unwrap();

    selection.parameters.decode_overlap = None;
    assert!(contract
        .validate_selection(&selection)
        .unwrap_err()
        .to_string()
        .contains("decode_overlap is required"));

    selection.strategy = MemoryStrategy::Resident;
    selection.parameters = MemoryStrategyParameters {
        attention_chunk_size: Some(128),
        ..Default::default()
    };
    assert!(contract
        .validate_selection(&selection)
        .unwrap_err()
        .to_string()
        .contains("irrelevant"));
}

#[test]
fn optimized_evidence_requires_valid_passing_parity_and_an_observed_peak() {
    let contract = adopted_contract();
    let mut evidence = MemoryEvidence {
        key: MemoryEvidenceKey {
            model_family: "test-family".to_owned(),
            resolved_route: "test".to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            load_shape: contract.load_shape,
            mode: MemoryMode::TextToImage,
            reference_shape: MemoryReferenceShape::None,
            overlay: None,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            frames_per_second: None,
            strategy: MemoryStrategy::BoundedDecode,
            engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
        },
        conformance: MemoryConformanceState::Verified,
        dimensions: MemoryEvidenceDimensions::VERIFIED,
        calibration_abi: MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "test-layout-v1".to_owned(),
        sceneworks_revision: "scene".to_owned(),
        inference_revision: "inference".to_owned(),
        harness_version: "v1".to_owned(),
        predicted_peak_bytes: 10,
        observed_peak_bytes: Some(9),
        parity: MemoryParityContract::Tolerance {
            metric: "max_abs".to_owned(),
            maximum_error: 0.001,
        },
        parity_result: MemoryParityResult::Passed,
    };
    assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

    evidence.parity = MemoryParityContract::Tolerance {
        metric: String::new(),
        maximum_error: f64::NAN,
    };
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Invalid)
    );
    evidence.parity = MemoryParityContract::Golden {
        fixture: String::new(),
        metric: "max_abs".to_owned(),
        maximum_error: -1.0,
    };
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Invalid)
    );
    evidence.parity = MemoryParityContract::Exact;
    evidence.observed_peak_bytes = None;
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Invalid)
    );
    evidence.observed_peak_bytes = Some(9);
    evidence.parity_result = MemoryParityResult::NotRun;
    assert_eq!(
        evidence.optimized_eligibility(&contract),
        Err(MemoryEvidenceVerdict::Unverified)
    );
}
/// SC-15794: the component scope's accessors and its DiT-only default.
///
/// The default is the whole backward-compatibility story — every provider and every selection
/// written before this type existed must keep meaning "stream the DiT", so `None` resolving to
/// anything else would silently re-scope work already in production.
#[test]
fn the_transformer_component_defaults_to_dit_and_reports_its_membership() {
    assert_eq!(TransformerComponent::default(), TransformerComponent::Dit);
    assert_eq!(
        MemoryStrategyParameters::default().window_component(),
        TransformerComponent::Dit
    );
    let explicit = MemoryStrategyParameters {
        transformer_window_component: Some(TransformerComponent::TextEncoder),
        ..Default::default()
    };
    assert_eq!(
        explicit.window_component(),
        TransformerComponent::TextEncoder
    );

    // Membership, asserted per variant rather than via a loop so a swapped pair cannot pass.
    assert!(TransformerComponent::Dit.includes_dit());
    assert!(!TransformerComponent::Dit.includes_text_encoder());
    assert!(!TransformerComponent::TextEncoder.includes_dit());
    assert!(TransformerComponent::TextEncoder.includes_text_encoder());
    assert!(TransformerComponent::Both.includes_dit());
    assert!(TransformerComponent::Both.includes_text_encoder());
}

/// The selected scope is validated against what the provider actually declared — the check that
/// stops a selector asking for an encoder stream from a provider that only implements the DiT.
#[test]
fn a_selected_component_must_be_one_the_provider_declared() {
    let contract = |components: Vec<TransformerComponent>| {
        let mut c = adopted_contract();
        let rung4 = c
            .strategies
            .iter_mut()
            .find(|s| s.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4 capability");
        rung4.parameters.transformer_window_sizes = vec![1];
        rung4.parameters.transformer_window_components = components;
        c
    };
    let select = |component| MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: MemoryStrategyParameters {
            stage_residency: None,
            // Rung 4 is cumulative, so the lower rungs' parameters are required too; these are
            // `adopted_contract`'s own declared candidates.
            decode_tile_edge: Some(512),
            decode_overlap: Some(64),
            attention_chunk_size: Some(256),
            transformer_window_size: Some(1),
            transformer_window_component: Some(component),
        },
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
    };

    let dit_only = contract(vec![TransformerComponent::Dit]);
    assert!(validate_selected_parameters(&select(TransformerComponent::Dit), &dit_only).is_ok());
    let err = validate_selected_parameters(&select(TransformerComponent::TextEncoder), &dit_only)
        .expect_err("a provider that declares only Dit must reject a TextEncoder selection");
    assert!(err.contains("TextEncoder"), "{err}");

    // An empty declaration means DiT-only, so DiT is still selectable and nothing else is.
    let empty = contract(Vec::new());
    assert!(validate_selected_parameters(&select(TransformerComponent::Dit), &empty).is_ok());
    assert!(validate_selected_parameters(&select(TransformerComponent::Both), &empty).is_err());

    // A provider that declares the encoder scope accepts it.
    let both = contract(vec![TransformerComponent::Dit, TransformerComponent::Both]);
    assert!(validate_selected_parameters(&select(TransformerComponent::Both), &both).is_ok());
}

/// A provider may only own component candidates on the rung that owns the window they scope.
#[test]
fn only_rung_four_may_own_component_candidates() {
    let with_components = |strategy| MemoryStrategyCapability {
        strategy,
        support: MemoryStrategySupport::Implemented,
        parameters: MemoryParameterRanges {
            transformer_window_components: vec![TransformerComponent::Both],
            ..Default::default()
        },
    };
    let mut errors = Vec::new();
    validate_owned_parameter_domain(&with_components(MemoryStrategy::BoundedDecode), &mut errors);
    assert!(
        errors.iter().any(|e| e.contains("component candidates")),
        "a non-rung-4 strategy owning component candidates must be rejected, got {errors:?}"
    );

    // Duplicates are a declaration bug the selector would silently tolerate.
    let mut errors = Vec::new();
    validate_owned_parameter_domain(
        &MemoryStrategyCapability {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            support: MemoryStrategySupport::Implemented,
            parameters: MemoryParameterRanges {
                transformer_window_sizes: vec![1],
                transformer_window_components: vec![
                    TransformerComponent::Dit,
                    TransformerComponent::Dit,
                ],
                ..Default::default()
            },
        },
        &mut errors,
    );
    assert!(
        errors.iter().any(|e| e.contains("duplicate")),
        "duplicate component candidates must be rejected, got {errors:?}"
    );

    // An EMPTY declaration is legal and means DiT-only — that is what keeps every pre-SC-15794
    // provider valid rather than retroactively incomplete.
    let mut errors = Vec::new();
    validate_owned_parameter_domain(
        &MemoryStrategyCapability {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            support: MemoryStrategySupport::Implemented,
            parameters: MemoryParameterRanges {
                transformer_window_sizes: vec![1],
                ..Default::default()
            },
        },
        &mut errors,
    );
    assert!(
        errors.is_empty(),
        "an empty component declaration must be legal, got {errors:?}"
    );
}

/// Set one rung's support, returning the mutated contract.
fn with_support(
    mut contract: MemoryProviderContract,
    strategy: MemoryStrategy,
    support: MemoryStrategySupport,
) -> MemoryProviderContract {
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == strategy)
        .expect("declared rung")
        .support = support;
    contract
}

fn bf16() -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    }
}

/// The selection space is the full validated product of the engaged rungs' declared domains,
/// quality first (largest tile / chunk / window, narrowest scope), the tile edge varying fastest.
#[test]
fn selection_space_is_the_declared_product_in_quality_order() {
    let mut contract = adopted_contract();
    for capability in &mut contract.strategies {
        if capability.strategy == MemoryStrategy::BoundedTransformerResidency {
            capability.parameters.transformer_window_components = vec![
                TransformerComponent::Both,
                TransformerComponent::Dit,
                TransformerComponent::TextEncoder,
            ];
        }
    }
    let mut widest = 0;
    for strategy in MemoryStrategy::ALL {
        let Ok(space) = contract.selection_space(strategy, bf16(), false) else {
            assert!(contract
                .representative_selection(strategy, bf16(), false)
                .is_err());
            continue;
        };
        assert!(!space.is_empty(), "{strategy:?}");
        for selection in &space {
            contract.validate_selection(selection).unwrap();
        }
        let domain = |s: MemoryStrategy, f: fn(&MemoryParameterRanges) -> usize| {
            if contract.engages(strategy, s) {
                contract
                    .capability(s)
                    .map_or(1, |c| f(&c.parameters).max(1))
            } else {
                1
            }
        };
        let tiles = match (
            &contract.pid_decode_routes,
            contract.engages(strategy, MemoryStrategy::BoundedDecode),
        ) {
            (Some(routes), true) => routes.native.tile_edges.len().max(1),
            _ => domain(MemoryStrategy::BoundedDecode, |p| {
                p.decode_tile_edges.len() * p.decode_overlaps.len().max(1)
            }),
        };
        let expected = tiles
            * domain(MemoryStrategy::BoundedAttention, |p| {
                p.attention_chunk_sizes.len()
            })
            * domain(MemoryStrategy::BoundedTransformerResidency, |p| {
                p.transformer_window_sizes.len() * p.transformer_window_components.len().max(1)
            });
        let staged = strategy != MemoryStrategy::Resident
            && !contract.engages(strategy, MemoryStrategy::StagedResidency)
            && matches!(
                contract.support(MemoryStrategy::StagedResidency),
                Some(MemoryStrategySupport::Implemented)
            );
        let expected = expected * if staged { 2 } else { 1 };
        assert_eq!(space.len(), expected, "{strategy:?}: {space:?}");
        if staged {
            assert_eq!(space[0].parameters.stage_residency, None, "unstaged first");
            assert_eq!(
                space[space.len() - 1].parameters.stage_residency,
                Some(true)
            );
        }
        let first = space[0].parameters;
        let last = space[space.len() - 1].parameters;
        if let (Some(a), Some(b)) = (first.decode_tile_edge, last.decode_tile_edge) {
            assert!(a >= b, "{strategy:?}: largest tile first");
            if tiles > 1 {
                assert!(space[1].parameters.decode_tile_edge < first.decode_tile_edge);
            }
        }
        if contract.engages(strategy, MemoryStrategy::BoundedTransformerResidency) {
            assert_eq!(
                first.transformer_window_component,
                Some(TransformerComponent::Dit)
            );
            assert_eq!(
                last.transformer_window_component,
                Some(TransformerComponent::Both)
            );
        }
        let mut keys: Vec<_> = space
            .iter()
            .map(|s| format!("{:?}", s.parameters))
            .collect();
        keys.dedup();
        assert_eq!(
            keys.len(),
            space.len(),
            "{strategy:?}: duplicate selections"
        );
        widest = widest.max(space.len());
    }
    assert!(
        widest > 3,
        "the fixture must exercise a multi-lever space ({widest})"
    );
}

/// SC-15998: the graph itself, asserted edge by edge so a regression to the old rung-1 edge
/// cannot pass.
#[test]
fn optional_staging_changes_execution_without_changing_the_default() {
    let mut contract = adopted_contract();
    let mut selection = contract
        .representative_selection(
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            false,
        )
        .unwrap();
    assert!(!contract.engages_selection(&selection, MemoryStrategy::StagedResidency));
    assert!(
        !contract
            .generation_memory(&selection)
            .unwrap()
            .stage_residency
    );
    selection.parameters.stage_residency = Some(true);
    contract.validate_selection(&selection).unwrap();
    assert!(
        contract
            .generation_memory(&selection)
            .unwrap()
            .stage_residency
    );
    assert!(contract
        .engaged_composition_for_selection(&selection)
        .contains(&MemoryStrategy::StagedResidency));
    contract
        .strategies
        .iter_mut()
        .find(|cap| cap.strategy == MemoryStrategy::StagedResidency)
        .unwrap()
        .support = MemoryStrategySupport::Missing;
    assert!(contract.validate_selection(&selection).is_err());
    selection.parameters.stage_residency = None;
    contract.validate_selection(&selection).unwrap();
    selection.parameters.stage_residency = Some(false);
    assert!(contract.validate_selection(&selection).is_err());
}

#[test]
fn rung_four_requires_deferred_materialization_not_rung_one() {
    assert_eq!(MemoryStrategy::Resident.requires(), &[]);
    assert_eq!(MemoryStrategy::StagedResidency.requires(), &[]);
    assert_eq!(MemoryStrategy::BoundedDecode.requires(), &[]);
    assert_eq!(MemoryStrategy::BoundedAttention.requires(), &[]);
    assert_eq!(
        MemoryStrategy::BoundedTransformerResidency.requires(),
        &[MemoryStrategyPrerequisite::LoadShape(
            LoadShape::DeferredMaterialization
        )]
    );
    // The per-rung accessor is the same declaration, so a provider cannot diverge from it.
    for capability in &adopted_contract().strategies {
        assert_eq!(capability.requires(), capability.strategy.requires());
    }
}

/// SC-15805 — the case that is impossible today: rung 1 `Missing` must not make rungs 2 and 3
/// unselectable. Bounding decoder or attention scratch has nothing to do with whether the
/// conditioning component was shed.
#[test]
fn a_missing_rung_one_leaves_rungs_two_and_three_selectable() {
    let contract = with_support(
        adopted_contract(),
        MemoryStrategy::StagedResidency,
        MemoryStrategySupport::Missing,
    );
    assert!(contract.conformance_errors().is_empty());

    contract
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
            tier: bf16(),
        })
        .expect("rung 2 depends on nothing and must stay selectable under a Missing rung 1");

    contract
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedAttention,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                ..Default::default()
            },
            tier: bf16(),
        })
        .expect("rung 3 depends on nothing and must stay selectable under a Missing rung 1");
}

/// SC-15998 — rung 4 is refused for an eager load, and accepted for deferred materialization
/// even when rung 1 is unavailable.
#[test]
fn rung_four_is_gated_by_load_shape_not_staged_residency() {
    let rung_four = |contract: &MemoryProviderContract| {
        contract.validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            },
            tier: bf16(),
        })
    };

    let deferred_without_rung_one = with_support(
        adopted_contract(),
        MemoryStrategy::StagedResidency,
        MemoryStrategySupport::Missing,
    );
    rung_four(&deferred_without_rung_one)
        .expect("Resident+Deferred rung 4 must not require staged residency");

    let mut eager = adopted_contract();
    eager.load_shape = LoadShape::EagerMaterialization;
    let error = rung_four(&eager)
        .expect_err("a block window over an eagerly materialized trunk must be refused")
        .to_string();
    assert!(error.contains("DeferredMaterialization"), "{error}");
    assert!(error.contains("EagerMaterialization"), "{error}");

    // Component scope does not change the physical prerequisite.
    let mut encoder_scoped = eager;
    encoder_scoped
        .strategies
        .iter_mut()
        .find(|c| c.strategy == MemoryStrategy::BoundedTransformerResidency)
        .expect("rung 4")
        .parameters
        .transformer_window_components = vec![TransformerComponent::TextEncoder];
    let error = encoder_scoped
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::TextEncoder),
            },
            tier: bf16(),
        })
        .expect_err("a TextEncoder-scoped window does not escape the load-shape prerequisite")
        .to_string();
    assert!(error.contains("DeferredMaterialization"), "{error}");
}

/// A backend can append its real implementation coupling without changing the shared graph.
#[test]
fn an_additive_backend_prerequisite_can_require_staged_residency() {
    let mut contract = adopted_contract();
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    contract
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            },
            tier: bf16(),
        })
        .expect("the additive rung prerequisite engages staged residency");

    let missing = with_support(
        contract,
        MemoryStrategy::StagedResidency,
        MemoryStrategySupport::Missing,
    );
    let error = missing
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            },
            tier: bf16(),
        })
        .expect_err("the backend-specific rung edge must be enforced")
        .to_string();
    assert!(error.contains("StagedResidency"), "{error}");
    assert!(error.contains("ENGAGED IN THE SAME REQUEST"), "{error}");
}

/// SC-15998: rung 1 is neither part of the cost-order default nor rung 4's shared prerequisite.
///
/// Rungs 2 and 3 bound scratch and are free, so a cumulative default costs the caller nothing.
/// Rung 1 bounds residency and *"may evict the warm cross-request cache"* — a real cost paid by
/// the next request. Dragging it into a rung-2 selection that never needed it is a straight loss.
///
#[test]
fn rung_four_keeps_phase_residency_warm_while_cost_order_still_engages_scratch_rungs() {
    // 1. The behavior chosen here: a free scratch rung does not drag in the costly residency one.
    assert!(
        !MemoryStrategy::BoundedDecode.engages(MemoryStrategy::StagedResidency),
        "rung 2 bounds scratch and must not evict the warm cache by engaging rung 1"
    );
    assert!(
        !MemoryStrategy::BoundedAttention.engages(MemoryStrategy::StagedResidency),
        "rung 3 bounds scratch and must not evict the warm cache by engaging rung 1"
    );
    assert!(!MemoryStrategy::Resident.engages(MemoryStrategy::StagedResidency));

    // THE REGRESSION GUARD: rung 4 alone does not evict the warm cross-request cache.
    assert!(!MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::StagedResidency));
    assert!(MemoryStrategy::StagedResidency.engages(MemoryStrategy::StagedResidency));
    // End to end: the independent deferred load shape keeps rung 4 selectable.
    adopted_contract()
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            },
            tier: bf16(),
        })
        .expect("Deferred materialization satisfies rung 4 without engaging rung 1");

    // The AC's stated cost-order default is untouched: rung 4 still engages 2 and 3.
    assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedDecode));
    assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedAttention));
    assert!(MemoryStrategy::BoundedAttention.engages(MemoryStrategy::BoundedDecode));
    assert!(!MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedAttention));
    // Rung 0 stays engaged by everything — baseline, not a lever.
    for strategy in MemoryStrategy::ALL {
        assert!(strategy.engages(MemoryStrategy::Resident), "{strategy:?}");
    }
}

/// The cost-order default is engagement, is defeasible, and is NOT the prerequisite graph.
///
/// A provider that publishes a cheaper verified composition — rung 4 without rung 2 — must not
/// have to fight the validator: the unimplemented rung is simply not engaged, so its parameters
/// become irrelevant rather than required.
#[test]
fn the_cost_order_default_is_defeasible_selector_policy_not_a_dependency() {
    // Pure policy: a deeper selection engages every cheaper SCRATCH rung, and never the reverse.
    // (Rung 1 is deliberately outside this default — see
    // `rung_one_is_not_dragged_in_by_cost_order_but_the_rung_four_edge_still_engages_it`.)
    assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedDecode));
    assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::Resident));
    assert!(MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedDecode));
    assert!(!MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedAttention));

    // …and the contract intersects it with what the provider implements.
    let mut cheaper = with_support(
        adopted_contract(),
        MemoryStrategy::BoundedDecode,
        MemoryStrategySupport::Missing,
    );
    cheaper
        .strategies
        .iter_mut()
        .find(|c| c.strategy == MemoryStrategy::BoundedDecode)
        .expect("rung 2")
        .parameters = MemoryParameterRanges::default();
    cheaper.lifecycle.decode_tiling = false;
    assert!(cheaper.conformance_errors().is_empty());
    assert!(!cheaper.engages(
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategy::BoundedDecode
    ));

    // Rung 4 without rung 2 — accepted, and the decode parameters are now the irrelevant ones.
    let mut selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: MemoryStrategyParameters {
            stage_residency: None,
            decode_tile_edge: None,
            decode_overlap: None,
            attention_chunk_size: Some(256),
            transformer_window_size: Some(1),
            transformer_window_component: None,
        },
        tier: bf16(),
    };
    cheaper
        .validate_selection(&selection)
        .expect("a verified cheaper composition must not fight the validator");
    selection.parameters.decode_tile_edge = Some(512);
    assert!(cheaper
        .validate_selection(&selection)
        .unwrap_err()
        .to_string()
        .contains("irrelevant"));
}

#[test]
fn a_verified_exclusion_defeats_only_the_default_engagement_edge() {
    let mut contract = adopted_contract();
    contract
        .default_engagement_exclusions
        .push(MemoryStrategyEngagementExclusion {
            selection: MemoryStrategy::BoundedAttention,
            excluded_rung: MemoryStrategy::BoundedDecode,
            evidence: "direct calibration: attention chunking is independent of decode tiling"
                .to_owned(),
        });
    assert!(contract.conformance_errors().is_empty());
    assert!(!contract.engages(
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedDecode
    ));
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedAttention),
        vec![MemoryStrategy::Resident, MemoryStrategy::BoundedAttention]
    );
    contract
        .validate_selection(&MemorySelection {
            strategy: MemoryStrategy::BoundedAttention,
            parameters: MemoryStrategyParameters {
                attention_chunk_size: Some(256),
                ..Default::default()
            },
            tier: bf16(),
        })
        .expect("the documented cheaper composition must validate without decode parameters");

    contract.default_engagement_exclusions[0].evidence.clear();
    assert!(contract
        .conformance_errors()
        .iter()
        .any(|error| error.contains("has no verification evidence")));
}

fn evidence_log_record() -> MemoryEvidenceLogRecord {
    let calibration =
        MemoryCalibrationIdentity::new("test-layout-v1", LoadShape::EagerMaterialization);
    MemoryEvidenceLogRecord {
        key: MemoryEvidenceKey {
            model_family: "test-family".to_owned(),
            resolved_route: "test_provider".to_owned(),
            backend: MemoryBackend::Mlx,
            tier: bf16(),
            load_shape: LoadShape::EagerMaterialization,
            mode: MemoryMode::TextToImage,
            reference_shape: MemoryReferenceShape::None,
            overlay: None,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            frames_per_second: None,
            strategy: MemoryStrategy::StagedResidency,
            engaged_composition: vec![MemoryStrategy::Resident, MemoryStrategy::StagedResidency],
            parameters: MemoryStrategyParameters::default(),
        },
        declared_calibration: calibration.clone(),
        observed_calibration: calibration,
        predicted_peak_bytes: 8_000_000_000,
        observed_peak_bytes: 7_500_000_000,
        inference_revision: "a".repeat(40),
        sceneworks_revision: "b".repeat(40),
        model_revision: "d".repeat(40),
        model_inventory_sha256: "e".repeat(64),
        harness_version: "test-harness-v1".to_owned(),
        output_sha256: "c".repeat(64),
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::Passed,
    }
}

#[test]
fn optional_staging_keeps_legacy_key_bytes_and_names_the_new_composition() {
    let mut key = evidence_log_record().key;
    key.strategy = MemoryStrategy::BoundedDecode;
    key.engaged_composition = vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode];
    let legacy = evidence_key_json(&key);
    assert_eq!(
        legacy["parameters"],
        serde_json::json!({
            "decode_tile_edge":null,"decode_overlap":null,"attention_chunk_size":null,
            "transformer_window_size":null,"transformer_window_component":null
        })
    );
    key.parameters.stage_residency = Some(true);
    key.engaged_composition
        .insert(1, MemoryStrategy::StagedResidency);
    let staged = evidence_key_json(&key);
    assert_eq!(staged["parameters"]["stage_residency"], true);
    assert_ne!(
        serde_json::to_vec(&legacy).unwrap(),
        serde_json::to_vec(&staged).unwrap()
    );
    key.parameters.stage_residency = None;
    key.engaged_composition.remove(1);
    assert_eq!(
        serde_json::to_vec(&legacy).unwrap(),
        serde_json::to_vec(&evidence_key_json(&key)).unwrap()
    );
}

#[test]
fn memory_evidence_v1_line_contains_the_complete_typed_key_and_provenance() {
    let line = evidence_log_record().to_json_line().unwrap();
    assert!(line.starts_with(MEMORY_EVIDENCE_V1_PREFIX));
    let value: serde_json::Value =
        serde_json::from_str(&line[MEMORY_EVIDENCE_V1_PREFIX.len()..]).unwrap();
    assert_eq!(value["schema_version"], MEMORY_EVIDENCE_SCHEMA_VERSION);
    assert_eq!(value["key"]["backend"], "mlx");
    assert_eq!(value["key"]["model_family"], "test-family");
    assert_eq!(value["key"]["mode"], "text_to_image");
    assert_eq!(value["key"]["reference_shape"], "none");
    assert_eq!(value["key"]["frames_per_second"], serde_json::Value::Null);
    assert_eq!(value["key"]["load_shape"], "eager_materialization");
    assert_eq!(value["key"]["geometry"]["reference_count"], 0);
    assert_eq!(value["key"]["strategy"], "staged_residency");
    assert_eq!(
        value["key"]["engaged_composition"],
        serde_json::json!(["resident", "staged_residency"])
    );
    assert_eq!(value["declared_calibration"], value["observed_calibration"]);
    assert_eq!(value["predicted_peak_bytes"], 8_000_000_000_u64);
    assert_eq!(value["observed_peak_bytes"], 7_500_000_000_u64);
    assert_eq!(value["model_revision"], "d".repeat(40));
    assert_eq!(value["model_inventory_sha256"], "e".repeat(64));
    assert_eq!(value["output_sha256"], "c".repeat(64));
    assert_eq!(value["parity_result"]["kind"], "passed");
}

#[test]
fn evidence_writer_rejects_identity_revision_and_fingerprint_drift() {
    let mut record = evidence_log_record();
    record.key.reference_shape = MemoryReferenceShape::Image;
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("reference shape"));

    let mut record = evidence_log_record();
    record.key.frames_per_second = Some(0);
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("frame rate"));

    let mut record = evidence_log_record();
    record.key.engaged_composition = vec![MemoryStrategy::Resident];
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("end with the measured strategy"));

    let mut record = evidence_log_record();
    record.observed_calibration.fingerprint = "other-layout-v1".to_owned();
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("identities do not match"));

    let mut record = evidence_log_record();
    record.inference_revision = "main".to_owned();
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("exact 40-character"));

    let mut record = evidence_log_record();
    record.model_inventory_sha256 = "mutable-cache".to_owned();
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("model inventory SHA-256"));

    let mut record = evidence_log_record();
    record.output_sha256 = "not-a-sha".to_owned();
    assert!(record
        .to_json_line()
        .unwrap_err()
        .to_string()
        .contains("output SHA-256"));

    for invalid in [
        "",
        "Layout-v1",
        "layout_v1",
        "layout",
        "layout-v0",
        "layout-v01",
        "layout-v1-v2",
    ] {
        assert!(
            validate_calibration_fingerprint(invalid).is_err(),
            "{invalid:?}"
        );
    }
    for valid in ["layout-v1", "sc-16594-layout-v2", "layout-q4-512-v3"] {
        validate_calibration_fingerprint(valid).unwrap();
    }
}

/// sc-20799: the structural Resident receipt is a **generic** contract. Its domain is a caller
/// parameter (SCAIL-2's spelling is preserved for its existing parsers), and every optional
/// request knob is tag-hashed so an absent knob cannot borrow the receipt of a zero-valued one.
#[test]
fn structural_resident_receipt_is_domain_scoped_and_separates_absent_from_zero() {
    let evidence = MemoryStructuralResidentEvidence {
        abi: MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
        provider_id: "provider".to_owned(),
        repository: "org/repo".to_owned(),
        revision: "f".repeat(40),
        variant: "q4".to_owned(),
        receipt_sha256: "a".repeat(64),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        load_shape: LoadShape::EagerMaterialization,
        asset_facts: MemoryAssetFacts {
            base_bytes: 4,
            conditioning_bytes: 1,
            transformer_bytes: 2,
            decoder_bytes: 1,
            overlay_bytes: 0,
        },
        request_transient_bytes: 0,
        direct_file_count: 1,
        adapter_count: 0,
    };
    let base = MemoryStructuralResidentRequestIdentity {
        source_digest: "b".repeat(64),
        mode: "replacement".to_owned(),
        carrier_shape: "replacement:reference:16x16:control:16x16x4".to_owned(),
        width: 832,
        height: 480,
        frames: 45,
        fps: 16,
        reference_count: 1,
        seed: None,
        sampler: None,
        scheduler: None,
        steps: None,
        guidance: None,
        scheduler_shift: None,
        selection: MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        },
    };

    // SCAIL-2's two provider crates parse `parts[0]`; that spelling must not move.
    let scail2 = evidence.evidence_revision(&base).unwrap();
    assert_eq!(
        scail2.split(':').next(),
        Some(MemoryStructuralResidentEvidence::SCAIL2_RESIDENT_EVIDENCE_DOMAIN)
    );
    assert_eq!(
        scail2,
        evidence
            .evidence_revision_in_domain(
                MemoryStructuralResidentEvidence::SCAIL2_RESIDENT_EVIDENCE_DOMAIN,
                &base
            )
            .unwrap()
    );

    // A second adopter of this generic contract gets a disjoint namespace, digest included.
    let other = evidence
        .evidence_revision_in_domain("other-resident-v1", &base)
        .unwrap();
    assert_eq!(other.split(':').next(), Some("other-resident-v1"));
    assert_ne!(
        scail2.rsplit(':').next(),
        other.rsplit(':').next(),
        "the domain must be hashed, not merely prefixed"
    );
    for bad in ["", "has:colon"] {
        assert!(evidence.evidence_revision_in_domain(bad, &base).is_err());
    }

    // Absent must never collide with the zero/empty value of the same knob.
    let mut digests = vec![scail2.clone()];
    type Cross = Box<dyn Fn(&mut MemoryStructuralResidentRequestIdentity)>;
    let mutations: Vec<Cross> = vec![
        Box::new(|request| request.seed = Some(0)),
        Box::new(|request| request.steps = Some(0)),
        Box::new(|request| request.guidance = Some(0.0)),
        Box::new(|request| request.scheduler_shift = Some(0.0)),
        Box::new(|request| request.sampler = Some(String::new())),
        Box::new(|request| request.scheduler = Some(String::new())),
    ];
    for mutate in &mutations {
        let mut crossed = base.clone();
        mutate(&mut crossed);
        digests.push(evidence.evidence_revision(&crossed).unwrap());
    }
    let unique = digests.iter().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        digests.len(),
        "an absent knob shares a receipt with its zero/empty value"
    );

    // A neighbouring-field slide must not be reachable either: moving a character from the mode
    // into the carrier shape has to change the digest.
    let mut slid = base.clone();
    slid.mode = "replacemen".to_owned();
    slid.carrier_shape = format!("t{}", base.carrier_shape);
    assert_ne!(scail2, evidence.evidence_revision(&slid).unwrap());
}

#[test]
fn checked_byte_total_sums_and_refuses_overflow() {
    assert_eq!(checked_byte_total("p", "vae", [1, 2, 3]).unwrap(), 6);
    assert_eq!(checked_byte_total("p", "vae", []).unwrap(), 0);
    match checked_byte_total("p", "vae", [u64::MAX, 1]) {
        Err(Error::Msg(message)) => assert_eq!(message, "p: vae byte total overflows u64"),
        other => panic!("expected a typed overflow, got {other:?}"),
    }
}
