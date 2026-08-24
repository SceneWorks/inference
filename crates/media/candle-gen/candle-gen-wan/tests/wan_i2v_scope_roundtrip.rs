//! sc-20799: drive an admitted Wan-VACE request all the way through the **request scope** the
//! provider actually installs, not just through `validate_context`.
//!
//! gen-core's own Wan tests construct a `MemoryRunContext` directly and assert `validate_context`
//! accepts it. That checks only half of a two-sided gate. The other half is
//! `CandleRequestScopeCore::configure_request`, which re-derives the reference count from the
//! executing request via [`gen_core::GenerationRequest::memory_reference_count`] and refuses any
//! mismatch with the admitted `MemoryGeometry`. Because nothing drove both halves with the same
//! request, the two counts were free to disagree — and they did: the Wan-VACE `extend_clip` carrier
//! (exactly one `ControlClip`) was admitted at `reference_count == 1` while
//! `memory_reference_count()` scored it `0`, so admission and execution could never both pass and
//! the mode was unreachable by construction.
//!
//! Every case below therefore ends at `configure_request`, and the `video_bridge` /
//! `replace_person` cases pin the neighbouring carriers that must keep their own counts.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Conditioning, GenerationRequest, Image, LoadSpec, MemoryBudget, MemoryCacheState,
    MemoryMode, MemoryOptimizationAuthority, MemoryRunContext, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, WeightsSource,
};
use candle_gen_wan::i2v_memory_strategy as wan_memory;
use gen_core::wan_i2v_memory::{self, WanI2vBackend};

const WIDTH: u32 = 832;
const HEIGHT: u32 = 480;
const FRAMES: u32 = 45;

fn write_safetensors(path: &std::path::Path, name: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = 4 * 4 * 2; // [4, 4] BF16
    let header = serde_json::json!({
        name: {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, bytes],
        }
    });
    let mut json = serde_json::to_vec(&header).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend(json);
    out.extend(std::iter::repeat_n(name.len() as u8, bytes));
    std::fs::write(path, out).unwrap();
}

/// The Candle Wan-VACE fixture root, shaped exactly like gen-core's own `vace_route_fixture`.
fn vace_fixture() -> (tempfile::TempDir, LoadSpec) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("vace");
    for component in ["transformer", "text_encoder", "vae"] {
        std::fs::create_dir_all(root.join(component)).unwrap();
        std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
        write_safetensors(&root.join(component).join("model.safetensors"), "weight");
    }
    std::fs::create_dir_all(root.join("tokenizer")).unwrap();
    std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
    wan_i2v_memory::ensure_wan_vace_source_receipt(&root, WanI2vBackend::Candle).unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route("wan_vace");
    wan_memory::prepare_load_spec(&mut spec, "wan_vace").unwrap();
    (tmp, spec)
}

fn image(pixel: u8) -> Image {
    Image {
        width: WIDTH,
        height: HEIGHT,
        pixels: vec![pixel; (WIDTH * HEIGHT * 3) as usize],
    }
}

fn base_request(video_mode: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: "hold the source motion".to_owned(),
        width: WIDTH,
        height: HEIGHT,
        count: 1,
        seed: Some(29),
        steps: Some(50),
        guidance: Some(5.0),
        frames: Some(FRAMES),
        fps: Some(16),
        video_mode: Some(video_mode.to_owned()),
        control_scale: Some(1.0),
        ..Default::default()
    }
}

fn extend_request() -> GenerationRequest {
    GenerationRequest {
        conditioning: vec![Conditioning::ControlClip {
            frames: vec![image(17); FRAMES as usize],
            mask: vec![image(17); FRAMES as usize],
            masking_strength: 1.0,
            start_frame: 0,
            mode: gen_core::ReplacementMode::default(),
        }],
        ..base_request("extend_clip")
    }
}

/// A VACE bridge: black left-tail / right-head anchors around a white-mask mid-gray generated gap.
fn bridge_request() -> GenerationRequest {
    let mut mask = vec![image(0); FRAMES as usize];
    let mut frames = vec![image(17); FRAMES as usize];
    for index in 5..FRAMES as usize - 5 {
        mask[index] = image(255);
        frames[index] = image(128);
    }
    GenerationRequest {
        conditioning: vec![Conditioning::ControlClip {
            frames,
            mask,
            masking_strength: 1.0,
            start_frame: 0,
            mode: gen_core::ReplacementMode::default(),
        }],
        ..base_request("video_bridge")
    }
}

fn replace_person_request(references: usize) -> GenerationRequest {
    let mut conditioning = vec![Conditioning::ControlClip {
        frames: vec![image(37); FRAMES as usize],
        mask: vec![image(255); FRAMES as usize],
        masking_strength: 0.625,
        start_frame: 0,
        mode: gen_core::ReplacementMode::default(),
    }];
    conditioning.extend((0..references).map(|ordinal| Conditioning::Reference {
        image: Image {
            width: 16,
            height: 16,
            pixels: vec![ordinal as u8 + 1; 16 * 16 * 3],
        },
        strength: None,
    }));
    GenerationRequest {
        negative_prompt: Some("identity drift".to_owned()),
        control_scale: Some(0.75),
        conditioning,
        ..base_request(&format!("replace_person@{}", "a".repeat(64)))
    }
}

/// Admit `request` at `strategy` and hand back the scope the provider would install.
fn admit(
    prepared: &wan_memory::PreparedWanI2vMemory,
    request: &mut GenerationRequest,
    strategy: MemoryStrategy,
    mode_key: &str,
) -> MemoryRunContext {
    let contract = wan_i2v_memory::request_contract(prepared, request).unwrap();
    let selection = MemorySelection {
        strategy,
        parameters: if strategy == MemoryStrategy::BoundedDecode {
            MemoryStrategyParameters {
                decode_tile_edge: Some(wan_i2v_memory::DECODE_TILE_EDGES[0]),
                decode_overlap: Some(wan_i2v_memory::DECODE_OVERLAPS[0]),
                ..Default::default()
            }
        } else {
            MemoryStrategyParameters::default()
        },
        tier: prepared.tier,
    };
    request.memory = contract.generation_memory(&selection);
    let evidence_revision = wan_i2v_memory::request_evidence_revision(prepared, request).unwrap();
    let geometry = wan_i2v_memory::geometry_from_request(request);
    MemoryRunContext {
        selection,
        optimization_authority: if strategy.is_optimized() {
            MemoryOptimizationAuthority::Estimated
        } else {
            MemoryOptimizationAuthority::Resident
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        load_shape: prepared.contract.load_shape,
        mode: MemoryMode::Other(mode_key.to_owned()),
        has_reference: geometry.reference_count > 0,
        use_pid: false,
        has_phases: false,
        geometry,
        overlay: Some(prepared.adapter_identity.clone()),
        budget: MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision,
    }
}

fn round_trip(mut request: GenerationRequest, mode_key: &str, expected_references: u32) {
    let (_tmp, spec) = vace_fixture();
    let prepared = wan_memory::prepare(&spec, "wan_vace").unwrap();
    for strategy in [MemoryStrategy::Resident, MemoryStrategy::BoundedDecode] {
        let context = admit(&prepared, &mut request, strategy, mode_key);
        assert_eq!(
            context.geometry.reference_count, expected_references,
            "{mode_key}: admitted carrier count"
        );
        assert_eq!(
            request.memory_reference_count(),
            context.geometry.reference_count,
            "{mode_key}: the executing request must score the same carrier count admission did, or \
             the request scope refuses every admitted request"
        );
        wan_i2v_memory::validate_context(&prepared, &context).expect("admission");
        let mut scope = wan_memory::begin_request(&prepared, Device::Cpu, &context)
            .expect("scope")
            .expect("wan installs a scope");
        // The load-bearing half: this is what admission could never reach before.
        scope
            .configure_request(&mut request)
            .unwrap_or_else(|error| {
                panic!("{mode_key} at {strategy:?} crossed its own scope: {error}")
            });
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }
}

#[test]
fn vace_extend_clip_survives_admission_and_the_request_scope() {
    round_trip(extend_request(), "extend_clip", 1);
}

#[test]
fn vace_video_bridge_survives_admission_and_the_request_scope() {
    round_trip(bridge_request(), "video_bridge", 0);
}

#[test]
fn vace_replace_person_survives_admission_and_the_request_scope() {
    for references in 1..=4 {
        round_trip(
            replace_person_request(references),
            "replace_person",
            references as u32,
        );
    }
}

/// The invariant the three cases above depend on, stated once over every admitted Wan carrier:
/// the admission-side counter and the execution-side counter are the same function.
#[test]
fn admission_and_execution_count_every_wan_carrier_identically() {
    let mut cases = vec![
        ("extend_clip", extend_request()),
        ("video_bridge", bridge_request()),
    ];
    for references in 1..=4 {
        cases.push(("replace_person", replace_person_request(references)));
    }
    for (mode, request) in cases {
        assert_eq!(
            wan_i2v_memory::geometry_from_request(&request).reference_count,
            request.memory_reference_count(),
            "{mode}: admission and execution disagree on the carrier count"
        );
    }
}
