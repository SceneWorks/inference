/// `default_seed()` never hands out the same value twice in one process, even when calls land
/// in one clock tick, and never hands out the zero sentinel (sc-24114). Back to back the values
/// are strictly increasing; across threads they are pairwise distinct.
///
/// *Mutation that reds this:* returning the bare clock reading (the pre-sc-24114 body): the
/// tight loop below draws duplicates on any platform whose clock is coarser than the loop.
#[test]
fn default_seed_is_nonzero_and_never_repeats_in_process() {
    use std::collections::HashSet;

    let burst: Vec<u64> = (0..10_000).map(|_| super::default_seed()).collect();
    assert!(burst.iter().all(|seed| *seed != 0));
    assert!(
        burst.windows(2).all(|pair| pair[1] > pair[0]),
        "consecutive default seeds must be strictly increasing"
    );

    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                (0..2_000)
                    .map(|_| super::default_seed())
                    .collect::<Vec<u64>>()
            })
        })
        .collect();
    let mut all: Vec<u64> = burst;
    for handle in handles {
        all.extend(handle.join().unwrap());
    }
    let distinct: HashSet<u64> = all.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        all.len(),
        "a default seed was handed out twice"
    );
    // Deliberately no wall-clock assertion here (the clock ratchet): that the value lands
    // back on the clock once it moves past the last seed follows from `max(now, last + 1)`.
}

/// `image_reference_count()` must price a **transparent** reference exactly like an opaque one
/// (sc-24111).
///
/// This is not cosmetic arithmetic. `memory_reference_count()` delegates here, and both
/// providers' request scopes refuse a request whose `memory_reference_count()` differs from
/// the admitted `MemoryGeometry::reference_count`. A `ReferenceRgba` scored as zero is
/// therefore admitted at `reference_count = 1` and then **refused at execution**
/// ("references=0 does not fit admitted … references=1"), or priced with zero reference
/// tokens — the exact admission/execution disagreement the doc comment on
/// `memory_reference_count` says must be impossible by construction.
#[test]
fn a_transparent_reference_is_priced_like_an_opaque_one() {
    fn rgb() -> Image {
        Image {
            width: 8,
            height: 8,
            pixels: vec![0; 8 * 8 * 3],
        }
    }
    fn rgba() -> RgbaImage {
        RgbaImage {
            width: 8,
            height: 8,
            pixels: vec![0; 8 * 8 * 4],
        }
    }

    let lone = GenerationRequest {
        conditioning: vec![Conditioning::ReferenceRgba {
            image: rgba(),
            strength: None,
        }],
        ..Default::default()
    };
    assert_eq!(
        lone.image_reference_count(),
        1,
        "a lone transparent reference is one image reference"
    );
    assert_eq!(
        lone.memory_reference_count(),
        1,
        "the execution-side count must agree with admission"
    );

    let mixed = GenerationRequest {
        conditioning: vec![
            Conditioning::Reference {
                image: rgb(),
                strength: None,
            },
            Conditioning::ReferenceRgba {
                image: rgba(),
                strength: None,
            },
            Conditioning::MultiReference {
                images: vec![rgb()],
            },
        ],
        ..Default::default()
    };
    assert_eq!(
        mixed.image_reference_count(),
        3,
        "RGB and RGBA references share one ordered list and are priced alike"
    );
    assert_eq!(mixed.memory_reference_count(), 3);
}

use super::*;
use crate::execution_domains::{CfgBatchingDomain, ExecutionValueDomain};

#[test]
fn staged_residency_availability_preserves_the_two_independent_capabilities() {
    let classify = |unconditional, selectable| {
        Capabilities {
            unconditionally_engages_staged_residency: unconditional,
            supports_sequential_offload: selectable,
            ..Default::default()
        }
        .staged_residency_availability()
    };

    assert_eq!(classify(false, false), StagedResidencyAvailability::Absent);
    assert_eq!(
        classify(false, true),
        StagedResidencyAvailability::Selectable
    );
    assert_eq!(
        classify(true, false),
        StagedResidencyAvailability::UnconditionallyEngaged
    );
    assert_eq!(
        classify(true, true),
        StagedResidencyAvailability::UnconditionallyEngaged,
        "unconditional physical staging wins the derived tri-state without erasing that the \
             separate supports_sequential_offload bit is also true"
    );
}

#[test]
fn default_scope_is_allowed_only_for_resident_only_contracts() {
    let mut contract = MemoryProviderContract::compatibility_default(
        "fixture",
        crate::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    assert!(!requires_memory_request_scope(&contract));
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .unwrap()
        .support = crate::MemoryStrategySupport::Implemented;
    assert!(requires_memory_request_scope(&contract));
}

#[test]
fn a_component_cannot_be_raised_without_a_visible_floor_declaration() {
    assert_eq!(
        effective_component_quant(&[], PrecisionFloorComponent::TextEncoder, Quant::Q4),
        Quant::Q4
    );
    let declared = [ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TextEncoder,
        selected_tier: Quant::Q4,
        resident_tier: Quant::Q8,
    }];
    assert_eq!(
        effective_component_quant(&declared, PrecisionFloorComponent::TextEncoder, Quant::Q4),
        Quant::Q8
    );
    assert_eq!(
        effective_component_quant(
            &declared,
            PrecisionFloorComponent::TransformerHead,
            Quant::Q4
        ),
        Quant::Q4,
        "an unrelated component cannot inherit another component's floor"
    );
}

#[test]
fn generation_memory_is_opt_in_and_quality_preserving_levers_default_off() {
    let request = GenerationRequest::default();
    assert_eq!(request.memory, None);
    assert_eq!(
        GenerationMemory::default(),
        GenerationMemory {
            stage_residency: false,
            tile_vae_decode: false,
            chunk_attention: false,
            stream_transformer_blocks: false,
            decode_tile_edge: None,
            decode_overlap: None,
            attention_chunk_size: None,
            transformer_window_size: None,
            transformer_window_component: None,
            // sc-18317's typed execution domains. `None` is the provider's own historical
            // schedule for each, which is what keeps an untouched request byte-for-byte the
            // pre-sc-18317 render.
            graph_eval_cadence: None,
            ffn_chunk: None,
            cfg_batching: None,
            // sc-25905: unset is the prefill path — the cache is opt-in per request.
            prefix_kv_cache: None,
            calibration_error_phase: None,
            calibration_fault_harness_authorized: false,
        }
    );
}

/// SC-15510: the strategy-parameter carriers are additive and default-inert. This is deliberately
/// an exhaustive literal — adding a lever without deciding its default here fails to compile,
/// which is the point: a new field that defaults to anything but "the provider's own historical
/// constant" would silently change every existing render.
#[test]
fn strategy_parameters_default_to_the_providers_own_constants() {
    let memory = GenerationMemory::default();
    assert!(!memory.stage_residency);
    assert_eq!(memory.decode_tile_edge, None);
    assert_eq!(memory.decode_overlap, None);
    assert_eq!(memory.attention_chunk_size, None);
    assert_eq!(memory.transformer_window_size, None);
    // Setting a parameter does NOT turn its rung on: the boolean is the switch, the parameter is
    // only the value. A selector that set an edge but not `tile_vae_decode` gets no tiling.
    let parameterized = GenerationMemory {
        decode_tile_edge: Some(384),
        decode_overlap: Some(64),
        attention_chunk_size: Some(128),
        transformer_window_size: Some(2),
        ..Default::default()
    };
    assert!(!parameterized.tile_vae_decode);
    assert!(!parameterized.stream_transformer_blocks);
}

fn img(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![0u8; (w * h * 3) as usize],
    }
}

#[test]
fn scail2_typed_carrier_is_one_reference_and_rejects_crossed_shapes() {
    let exact = GenerationRequest {
        video_mode: Some("animation".to_owned()),
        conditioning: vec![
            Conditioning::Reference {
                image: img(2, 2),
                strength: None,
            },
            Conditioning::Mask { image: img(2, 2) },
            Conditioning::ControlClip {
                frames: vec![img(2, 2), img(2, 2)],
                mask: vec![img(2, 2), img(2, 2)],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
        ..Default::default()
    };
    let carrier = exact.scail2_animation_conditioning().unwrap();
    assert_eq!(carrier.reference_count(), 1);
    assert_eq!(exact.memory_reference_count(), 1);
    assert_eq!(
        exact.image_reference_count(),
        2,
        "physical Mask remains visible generically"
    );

    let mut extra = exact.clone();
    extra.conditioning.push(Conditioning::Reference {
        image: img(2, 2),
        strength: None,
    });
    assert!(extra.scail2_animation_conditioning().is_err());

    let mut mismatched = exact.clone();
    let Conditioning::ControlClip { mask, .. } = &mut mismatched.conditioning[2] else {
        unreachable!()
    };
    mask.pop();
    assert!(mismatched.scail2_animation_conditioning().is_err());

    let mut replacement = exact;
    replacement.video_mode = Some("replacement".to_owned());
    assert_eq!(replacement.memory_reference_count(), 1);
    let carrier = replacement.scail2_animation_conditioning().unwrap();
    assert_eq!(
        carrier.identity_shape("replacement").unwrap(),
        "replacement:reference:2x2:control:2x2x2"
    );
}

#[test]
fn keyframes_accessor_collects_in_order() {
    // first_last_frame: two keyframes at 0 and the last latent frame.
    let req = GenerationRequest {
        conditioning: vec![
            Conditioning::Keyframe {
                image: img(2, 2),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::Reference {
                image: img(2, 2),
                strength: None,
            },
            Conditioning::Keyframe {
                image: img(4, 4),
                frame_idx: 8,
                strength: 0.75,
            },
        ],
        ..Default::default()
    };
    let kf = req.keyframes();
    assert_eq!(kf.len(), 2);
    assert_eq!((kf[0].frame_idx, kf[0].strength), (0, 1.0));
    assert_eq!((kf[1].frame_idx, kf[1].strength), (8, 0.75));
    assert_eq!((kf[1].image.width, kf[1].image.height), (4, 4));
    // Reference is not a keyframe and is not a video clip / control clip.
    assert!(req.video_clips().is_empty());
    assert!(req.control_clip().is_none());
}

#[test]
fn video_clips_accessor_collects_clips() {
    // video_bridge: left clip @0, right clip @tail.
    let req = GenerationRequest {
        conditioning: vec![
            Conditioning::VideoClip {
                frames: vec![img(2, 2), img(2, 2)],
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::VideoClip {
                frames: vec![img(2, 2)],
                frame_idx: 24,
                strength: 0.9,
            },
        ],
        ..Default::default()
    };
    let clips = req.video_clips();
    assert_eq!(clips.len(), 2);
    assert_eq!((clips[0].frames.len(), clips[0].frame_idx), (2, 0));
    assert_eq!((clips[1].frames.len(), clips[1].frame_idx), (1, 24));
    assert!(req.keyframes().is_empty());
}

#[test]
fn control_clip_accessor_returns_first() {
    let req = GenerationRequest {
        conditioning: vec![Conditioning::ControlClip {
            frames: vec![img(2, 2), img(2, 2)],
            mask: vec![img(2, 2), img(2, 2)],
            masking_strength: 0.8,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }],
        ..Default::default()
    };
    let cc = req.control_clip().expect("control clip present");
    assert_eq!((cc.frames.len(), cc.mask.len()), (2, 2));
    assert_eq!(cc.masking_strength, 0.8);
    assert_eq!(cc.mode, ReplacementMode::FaceOnly);
}

#[test]
fn accessors_empty_by_default() {
    let req = GenerationRequest::default();
    assert!(req.keyframes().is_empty());
    assert!(req.video_clips().is_empty());
    assert!(req.control_clip().is_none());
}

/// A capability surface that turns nothing extra on: a single 256..=1024 image, no
/// negative/guidance/true_cfg, no samplers/schedulers, only `Reference` conditioning.
fn caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::Reference],
        samplers: vec!["euler"],
        min_size: 256,
        max_size: 1024,
        max_count: 1,
        ..Default::default()
    }
}

/// **sc-18317: the typed execution domains are fail-closed at the shared floor.**
///
/// The defect this closes is the pre-story state of all three knobs: a per-provider ad hoc
/// parameter a caller could not set, could not discover, and — had it been threaded through the
/// request without a declaration — would have been silently dropped by every provider that does
/// not implement it. So the floor's contract is exactly two things: an unset field is inert, and
/// a set field on a non-declaring descriptor is a typed `Unsupported` naming the field and the
/// remedy.
#[test]
fn execution_domains_are_refused_by_name_on_a_non_declaring_descriptor() {
    let unsupported = caps();
    assert!(
        unsupported.execution.is_inert(),
        "the default capability surface must declare no execution domain"
    );

    for (label, memory) in [
        (
            "graph_eval_cadence",
            GenerationMemory {
                graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
                ..Default::default()
            },
        ),
        (
            "ffn_chunk",
            GenerationMemory {
                ffn_chunk: Some(FfnChunk::new(4096).unwrap()),
                ..Default::default()
            },
        ),
        (
            "cfg_batching",
            GenerationMemory {
                cfg_batching: Some(CfgBatching::Sequential),
                ..Default::default()
            },
        ),
        (
            "prefix_kv_cache",
            GenerationMemory {
                prefix_kv_cache: Some(PrefixKvCacheMode::Cached),
                ..Default::default()
            },
        ),
    ] {
        let request = GenerationRequest {
            memory: Some(memory),
            ..base_req()
        };
        let error = unsupported
            .validate_request("m", &request)
            .expect_err("a non-declaring descriptor must refuse a selected execution domain");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "{label} must be a capability gap, not a range error: {error:?}"
        );
        let message = error.to_string();
        assert!(message.contains(label), "{label}: {message}");
        assert!(
            message.contains("unset"),
            "{label} refusal must name the remedy: {message}"
        );
        // The size-skipping floors run every non-spatial check, so the gate must be reached
        // through them too — a provider on the auto-size convention must not lose the refusal.
        assert!(
            unsupported
                .validate_request_skip_size("m", &request)
                .is_err(),
            "{label} must also be refused on the size-skipping floor"
        );
        assert!(
            unsupported.validate_request_audio("m", &request).is_err(),
            "{label} must also be refused on the audio floor"
        );
    }
}

/// The other half: a declaring descriptor admits exactly its declared values, and the ladder
/// parameters are untouched by the execution gate.
#[test]
fn execution_domains_admit_declared_values_and_leave_the_ladder_alone() {
    let declaring = Capabilities {
        execution: ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
            ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![2048]),
            cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
            ..ExecutionSurface::default()
        },
        ..caps()
    };
    assert!(declaring.execution.declaration_errors().is_empty());

    let admitted = GenerationRequest {
        memory: Some(GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::new(4).unwrap()),
            ffn_chunk: Some(FfnChunk::new(2048).unwrap()),
            cfg_batching: Some(CfgBatching::Batched),
            ..Default::default()
        }),
        ..base_req()
    };
    declaring
        .validate_request("m", &admitted)
        .expect("every declared value must reach the provider");

    let off_grid = GenerationRequest {
        memory: Some(GenerationMemory {
            ffn_chunk: Some(FfnChunk::new(2047).unwrap()),
            ..Default::default()
        }),
        ..base_req()
    };
    let message = declaring
        .validate_request("m", &off_grid)
        .expect_err("an off-domain chunk must be refused")
        .to_string();
    assert!(
        message.contains("[2048]"),
        "domain must be named: {message}"
    );

    let unimplemented_mode = GenerationRequest {
        memory: Some(GenerationMemory {
            cfg_batching: Some(CfgBatching::Sequential),
            ..Default::default()
        }),
        ..base_req()
    };
    assert!(
        declaring
            .validate_request("m", &unimplemented_mode)
            .is_err(),
        "a mode the provider does not implement must be refused even though CFG batching is \
             declared"
    );

    // An unset execution selection carrying a ladder rung still validates on the inert surface:
    // the execution gate is independent of the memory ladder.
    let ladder_only = GenerationRequest {
        memory: Some(GenerationMemory {
            stage_residency: true,
            chunk_attention: true,
            attention_chunk_size: Some(128),
            ..Default::default()
        }),
        ..base_req()
    };
    caps()
        .validate_request("m", &ladder_only)
        .expect("the execution gate must not touch ladder parameters");
}

#[test]
fn preview_support_is_weights_free_and_defaults_to_unsupported() {
    let unsupported = Capabilities::default();
    let supported_waiting_for_first_frame = Capabilities {
        supports_preview: true,
        ..Default::default()
    };

    assert!(!unsupported.supports_preview);
    assert!(supported_waiting_for_first_frame.supports_preview);
}

#[test]
fn prompt_enhancement_is_fail_closed_and_tuning_is_bounded() {
    let unsupported = caps();
    let requested = GenerationRequest {
        enhance_prompt: true,
        ..base_req()
    };
    assert!(matches!(
        unsupported.validate_request("m", &requested),
        Err(Error::Unsupported(message)) if message.contains("prompt enhancement")
    ));

    let supported = Capabilities {
        supports_prompt_enhancement: true,
        ..caps()
    };
    for request in [
        GenerationRequest {
            enhance_max_tokens: Some(32),
            ..base_req()
        },
        GenerationRequest {
            enhance_temperature: Some(0.15),
            ..base_req()
        },
    ] {
        assert!(supported
            .validate_request("m", &request)
            .unwrap_err()
            .to_string()
            .contains("requires enhance_prompt=true"));
    }
    for tokens in [0, MAX_ENHANCE_TOKENS + 1] {
        let request = GenerationRequest {
            enhance_prompt: true,
            enhance_max_tokens: Some(tokens),
            ..base_req()
        };
        assert!(supported.validate_request("m", &request).is_err());
    }
    for temperature in [-0.1, MAX_ENHANCE_TEMPERATURE + 0.1] {
        let request = GenerationRequest {
            enhance_prompt: true,
            enhance_temperature: Some(temperature),
            ..base_req()
        };
        assert!(supported.validate_request("m", &request).is_err());
    }
    assert!(supported
        .validate_request(
            "m",
            &GenerationRequest {
                enhance_prompt: true,
                enhance_max_tokens: Some(MAX_ENHANCE_TOKENS),
                enhance_temperature: Some(MAX_ENHANCE_TEMPERATURE),
                ..base_req()
            }
        )
        .is_ok());
}

#[test]
fn enhancement_token_limits_are_canonical_model_scoped_without_changing_defaults() {
    let supported = Capabilities {
        supports_prompt_enhancement: true,
        ..caps()
    };
    for (id, limit) in [
        ("flux2_dev", 2_048),
        ("ltx_2_3", 2_048),
        ("unknown", 2_048),
        ("flux3_klein_alias", 2_048),
        ("flux3_klein", 3_584),
        ("flux3_klein_base", 3_584),
    ] {
        assert_eq!(max_enhance_tokens_for_model(id), limit, "{id}");
        for tokens in [0, 2_048, 2_049, 3_584, 3_585] {
            let request = GenerationRequest {
                enhance_prompt: true,
                enhance_max_tokens: Some(tokens),
                ..base_req()
            };
            assert_eq!(
                supported.validate_request(id, &request).is_ok(),
                tokens > 0 && tokens <= limit,
                "{id}: {tokens} tokens"
            );
        }
        for enabled in [false, true] {
            let request = GenerationRequest {
                enhance_prompt: enabled,
                ..base_req()
            };
            assert!(supported.validate_request(id, &request).is_ok(), "{id}");
            assert!(request.enhance_max_tokens.is_none());
        }
        let disabled_tuning = GenerationRequest {
            enhance_max_tokens: Some(limit),
            ..base_req()
        };
        assert!(supported.validate_request(id, &disabled_tuning).is_err());
    }
    assert_eq!(MAX_ENHANCE_TOKENS, 2_048);
}

fn base_req() -> GenerationRequest {
    GenerationRequest {
        prompt: "x".into(),
        width: 512,
        height: 512,
        ..Default::default()
    }
}

#[test]
fn validate_request_accepts_in_surface() {
    let c = caps();
    assert!(c.validate_request("m", &base_req()).is_ok());
    // An advertised sampler + an accepted conditioning kind are fine.
    assert!(c
        .validate_request(
            "m",
            &GenerationRequest {
                sampler: Some("euler".into()),
                conditioning: vec![Conditioning::Reference {
                    image: img(8, 8),
                    strength: None,
                }],
                ..base_req()
            }
        )
        .is_ok());
}

#[test]
fn calibration_fault_injection_requires_a_complete_harness_authorization_pair() {
    let capabilities = caps();

    let mut phase_without_authorization = base_req();
    phase_without_authorization.memory = Some(GenerationMemory {
        calibration_error_phase: Some(MemoryPhase::Denoise),
        ..Default::default()
    });
    let error = capabilities
        .validate_request("m", &phase_without_authorization)
        .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported(message) if message.contains("explicit harness authorization"))
    );

    let mut authorization_without_phase = base_req();
    authorization_without_phase.memory = Some(GenerationMemory {
        calibration_fault_harness_authorized: true,
        ..Default::default()
    });
    let error = capabilities
        .validate_request_skip_size("m", &authorization_without_phase)
        .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported(message) if message.contains("requires an error phase"))
    );

    let mut authorized = base_req();
    let mut memory = GenerationMemory::default();
    memory.authorize_calibration_fault(MemoryPhase::Decode);
    authorized.memory = Some(memory);
    capabilities
        .validate_request("m", &authorized)
        .expect("an explicitly authorized calibration harness request reaches the provider");
}

/// `ResolvedDownstream` exempts **the sentinel**, not every size.
///
/// The variant's own doc says "`width`/`height == 0` is a resolve-from-driving-media sentinel and
/// the range check does not apply *to it*". Scoping matters: a provider advertising this is
/// saying one specific pair of values is special, not that it has opted out of the shared floor.
///
/// Written as a regression gate because the first implementation of `size_floor` read the
/// variant as a blanket opt-out and disabled the range check entirely. On main, SCAIL-2 got the
/// full check whenever dimensions were explicit — its `pipeline.rs` hand-rolls exactly this
/// branch, with a comment saying so — and routing that branch through a blanket
/// `ResolvedDownstream` silently stopped rejecting an explicit 16x16 or 4096x4096 against
/// declared bounds of 32..=1280. Nothing downstream catches it: `min_size`/`max_size` appear
/// nowhere in `mlx-gen-scail2` outside the descriptor.
#[test]
fn resolved_downstream_exempts_the_sentinel_but_still_range_checks_explicit_sizes() {
    let c = Capabilities {
        size_floor: SizeFloor::ResolvedDownstream,
        ..caps()
    };

    // The sentinel is the whole point of the variant: accepted.
    assert!(
        c.validate_request(
            "m",
            &GenerationRequest {
                width: 0,
                height: 0,
                ..base_req()
            }
        )
        .is_ok(),
        "the 0x0 resolve-downstream sentinel must be accepted"
    );

    // Explicit dimensions are NOT the sentinel and must still meet the advertised range.
    for (w, h, why) in [
        (16, 16, "below min_size"),
        (4096, 4096, "above max_size"),
        (
            0,
            512,
            "half-sentinel: only width is 0, so this is not the sentinel",
        ),
        (512, 0, "half-sentinel: only height is 0"),
    ] {
        assert!(
            c.validate_request(
                "m",
                &GenerationRequest {
                    width: w,
                    height: h,
                    ..base_req()
                }
            )
            .is_err(),
            "{w}x{h} ({why}) must still be rejected under ResolvedDownstream"
        );
    }
}

#[test]
fn advertised_explicit_grid_rejects_only_off_grid_explicit_sizes() {
    let ranged = Capabilities {
        size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 32 },
        ..caps()
    };
    let resolved = Capabilities {
        size_floor: SizeFloor::ResolvedDownstreamExplicitGrid { multiple: 32 },
        ..caps()
    };

    for c in [&ranged, &resolved] {
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    width: 512,
                    height: 480,
                    ..base_req()
                }
            )
            .is_ok());
        for (width, height) in [(512, 481), (513, 480)] {
            let err = c
                .validate_request(
                    "m",
                    &GenerationRequest {
                        width,
                        height,
                        ..base_req()
                    },
                )
                .expect_err("an explicit off-grid size must be refused")
                .to_string();
            assert!(
                err.contains("multiples of 32") && err.contains(&format!("{width}×{height}")),
                "got: {err}"
            );
        }
    }

    assert!(ranged
        .validate_request(
            "m",
            &GenerationRequest {
                width: 0,
                height: 0,
                ..base_req()
            }
        )
        .is_err());
    assert!(resolved
        .validate_request(
            "m",
            &GenerationRequest {
                width: 0,
                height: 0,
                ..base_req()
            }
        )
        .is_ok());
    assert_eq!(ranged.size_floor.explicit_size_multiple(), Some(32));
    assert_eq!(resolved.size_floor.explicit_size_multiple(), Some(32));
    assert_eq!(SizeFloor::RangeChecked.explicit_size_multiple(), None);
}

/// The default variant is unaffected — a descriptor that says nothing behaves exactly as before.
#[test]
fn range_checked_is_the_default_and_rejects_out_of_range_sizes() {
    let c = caps();
    assert_eq!(c.size_floor, SizeFloor::RangeChecked);
    for (w, h) in [(16, 16), (4096, 4096), (0, 0)] {
        assert!(
            c.validate_request(
                "m",
                &GenerationRequest {
                    width: w,
                    height: h,
                    ..base_req()
                }
            )
            .is_err(),
            "{w}x{h} must be rejected when the floor is RangeChecked"
        );
    }
}

/// `ResolvedDownstream` exempts **size**, and nothing else.
///
/// The sibling above pins one half of the scoping — an *explicit* size is still range-checked.
/// This pins the half a mis-scoping would break in the other direction: routing the sentinel
/// around the **whole** floor rather than around one check. That is not a hypothetical failure
/// mode for the provider this variant exists for. SCAIL-2's `generate` re-runs the floor itself
/// precisely so a `guidance: NaN` cannot NaN-poison a multi-minute video render into
/// garbage-as-success, and `0x0` — the auto-size path — is its *normal* request shape, not an
/// edge case. A sentinel that skipped the floor would therefore be un-validated in the common
/// case and validated only in the rare one.
///
/// Every case is a single-field mutation of a request the test first proves is accepted, so a
/// rejection can only have come from the field named: no case can pass vacuously.
#[test]
fn resolved_downstream_sentinel_still_runs_every_non_size_check() {
    let c = Capabilities {
        size_floor: SizeFloor::ResolvedDownstream,
        ..caps()
    };
    let sentinel = GenerationRequest {
        width: 0,
        height: 0,
        ..base_req()
    };
    assert!(
        c.validate_request("m", &sentinel).is_ok(),
        "the 0x0 sentinel must be accepted, or every case below would pass vacuously"
    );

    let cases: Vec<(&str, &str, GenerationRequest)> = vec![
        (
            "count above max_count",
            "count",
            GenerationRequest {
                count: 2,
                ..sentinel.clone()
            },
        ),
        (
            "an explicit steps: 0 (F-007)",
            "steps must be >= 1",
            GenerationRequest {
                steps: Some(0),
                ..sentinel.clone()
            },
        ),
        (
            "an unadvertised sampler",
            "unsupported sampler",
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..sentinel.clone()
            },
        ),
        (
            "an unadvertised scheduler",
            "unsupported scheduler",
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..sentinel.clone()
            },
        ),
        (
            "a conditioning kind outside the allowlist",
            "conditioning is not supported",
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..sentinel.clone()
            },
        ),
        (
            "a non-finite float knob (F-053)",
            "must be finite",
            GenerationRequest {
                strength: Some(f32::NAN),
                ..sentinel.clone()
            },
        ),
        (
            "a negative prompt on a model that does not advertise one",
            "negative prompts are not supported",
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..sentinel.clone()
            },
        ),
        (
            "guidance on a distilled/CFG-free model",
            "guidance is not supported",
            GenerationRequest {
                guidance: Some(3.5),
                ..sentinel.clone()
            },
        ),
    ];
    for (why, expected, req) in cases {
        let Err(err) = c.validate_request("m", &req) else {
            panic!(
                "{why} must still be rejected on the 0x0 sentinel path — \
                     ResolvedDownstream exempts SIZE only, not the rest of the floor"
            );
        };
        // Asserting the *message* and not merely `is_err` is what makes a case non-vacuous: a
        // future reordering that made some earlier check fire first would otherwise look green
        // while the check this row is actually about had stopped running.
        assert!(
            err.to_string().contains(expected),
            "{why}: expected a rejection naming {expected:?}, got: {err}"
        );
    }
}

#[test]
fn validate_request_enforces_advertised_surface() {
    let c = caps();
    let cases: Vec<GenerationRequest> = vec![
        // count out of range
        GenerationRequest {
            count: 0,
            ..base_req()
        },
        GenerationRequest {
            count: 2,
            ..base_req()
        },
        // size out of range (below min, above max)
        GenerationRequest {
            width: 128,
            ..base_req()
        },
        GenerationRequest {
            height: 2048,
            ..base_req()
        },
        // capability flags not advertised
        GenerationRequest {
            negative_prompt: Some("n".into()),
            ..base_req()
        },
        GenerationRequest {
            guidance: Some(3.5),
            ..base_req()
        },
        GenerationRequest {
            true_cfg: Some(4.0),
            ..base_req()
        },
        // sampler / scheduler not advertised
        GenerationRequest {
            sampler: Some("unipc".into()),
            ..base_req()
        },
        GenerationRequest {
            scheduler: Some("linear".into()),
            ..base_req()
        },
        // conditioning kind not accepted
        GenerationRequest {
            conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
            ..base_req()
        },
    ];
    for (i, req) in cases.iter().enumerate() {
        assert!(
            c.validate_request("m", req).is_err(),
            "case {i} should have been rejected"
        );
    }
}

#[test]
fn validate_request_skip_size_runs_every_non_size_check() {
    // F-158: the auto-size floor. `validate_request_skip_size` must still enforce the whole floor
    // *except* the size range — so a sentinel `0x0` size passes, but an out-of-surface count /
    // sampler / conditioning / non-finite knob is still rejected. Used by providers that resolve
    // size downstream (SCAIL-2 sizing from the driving video).
    let c = caps();
    // Auto-size sentinel (0x0) is accepted where the full floor would reject it for being < min.
    let auto = GenerationRequest {
        width: 0,
        height: 0,
        ..base_req()
    };
    assert!(
        c.validate_request("m", &auto).is_err(),
        "size 0x0 is below min for the full floor"
    );
    assert!(
        c.validate_request_skip_size("m", &auto).is_ok(),
        "skip_size must accept the 0x0 auto-size sentinel"
    );
    // Every non-size violation must still fire on the auto-size path. Labelled rather than
    // indexed: "skip_size case 3 failed" does not tell you which guarantee stopped holding.
    //
    // The last three rows close a real gap. This method's doc promises "negative-prompt /
    // guidance / true_cfg support gating" and "scheduler … membership" on this path, and
    // neither had a single case here — the *capability-gap* family (typed `Error::Unsupported`,
    // F-008) was represented only by `sampler`. Support gating is the family a consumer relies
    // on to tell "this backend can't do that" from "that value is out of range", so a
    // size-skipping floor that quietly dropped it would let a negative prompt reach a
    // CFG-free distilled model and be silently ignored.
    let rejected: Vec<(&str, GenerationRequest)> = vec![
        (
            "oversized count",
            GenerationRequest {
                count: 2,
                ..auto.clone()
            },
        ),
        (
            "explicit zero steps",
            GenerationRequest {
                steps: Some(0),
                ..auto.clone()
            },
        ),
        (
            "unadvertised sampler",
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..auto.clone()
            },
        ),
        (
            "disallowed conditioning kind",
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..auto.clone()
            },
        ),
        (
            // A non-finite knob is not support-gated and would NaN-poison the run.
            "non-finite strength (F-053)",
            GenerationRequest {
                strength: Some(f32::NAN),
                ..auto.clone()
            },
        ),
        (
            "negative prompt on a model that does not advertise one",
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..auto.clone()
            },
        ),
        (
            "guidance on a model that does not advertise it",
            GenerationRequest {
                guidance: Some(3.5),
                ..auto.clone()
            },
        ),
        (
            "true_cfg on a model that does not advertise it",
            GenerationRequest {
                true_cfg: Some(4.0),
                ..auto.clone()
            },
        ),
        (
            "unadvertised scheduler",
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..auto.clone()
            },
        ),
    ];
    for (why, req) in &rejected {
        assert!(
            c.validate_request_skip_size("m", req).is_err(),
            "{why} must still be rejected on the auto-size path"
        );
    }
}

#[test]
fn validate_request_rejects_explicit_zero_steps() {
    // F-007: the floor now enforces the steps>=1 claim the schedule builders rely on.
    let c = caps();
    let bad = GenerationRequest {
        steps: Some(0),
        ..base_req()
    };
    let err = c.validate_request("m", &bad).unwrap_err();
    assert!(matches!(err, Error::Msg(_)), "steps=0 is a range error");
    assert!(err.to_string().contains("steps must be >= 1"));
    // `None` and a positive count still pass.
    assert!(c.validate_request("m", &base_req()).is_ok());
    assert!(c
        .validate_request(
            "m",
            &GenerationRequest {
                steps: Some(1),
                ..base_req()
            }
        )
        .is_ok());
}

/// sc-19502 — the exact-step surface refuses an off-schedule count, admits every count it
/// advertises, and admits `None`.
///
/// The `None` half is the load-bearing one: it is what keeps "omit `steps`, get the baked
/// schedule" working, and inverting it would break every caller of a distilled model who does
/// not know its magic number.
#[test]
fn advertised_supported_steps_reject_every_off_schedule_count() {
    let distilled = Capabilities {
        supported_steps: StepSupport::Exact(vec![8]),
        ..caps()
    };
    let at = |steps: Option<u32>| {
        distilled.validate_request(
            "ltx_2_3",
            &GenerationRequest {
                steps,
                ..base_req()
            },
        )
    };

    // Admitted: the advertised count, and "the model picks".
    assert!(at(Some(8)).is_ok(), "the advertised count must be admitted");
    assert!(
        at(None).is_ok(),
        "an omitted count must use the baked schedule, not be refused"
    );

    // Refused on BOTH sides of the advertised value — a floor-shaped guard would let 30 through,
    // which is the precise half `limits.hardMinSteps` could not express (sc-19426).
    for steps in [1, 4, 7, 9, 30] {
        let err = at(Some(steps)).unwrap_err().to_string();
        assert!(
            err.contains("ltx_2_3") && err.contains(&format!("steps={steps}")) && err.contains("8"),
            "the refusal must name the model, the request and the legal value: {err}"
        );
    }
}

/// sc-19502 — the polarity guard. Empty `supported_steps` is NO constraint, so every descriptor
/// that never opts in is byte-identical to before the field existed.
///
/// Deliberately its own test rather than an assertion inside the one above: this is the
/// property that makes the field safe to add to a 128-descriptor repo, and if it regressed, a
/// bare `Default::default()` would refuse every step count in the workspace.
#[test]
fn an_unconstrained_model_admits_every_step_count_the_sanity_caps_admit() {
    let c = caps();
    assert!(
        c.supported_steps.is_unconstrained(),
        "the default must be unconstrained"
    );
    for steps in [1, 2, 8, 30, MAX_STEPS] {
        assert!(
            c.validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(steps),
                    ..base_req()
                }
            )
            .is_ok(),
            "an undeclared model must still admit {steps} steps"
        );
    }
}

/// sc-19502 — a multi-value schedule is a SET, not a range: the gap between two advertised
/// counts is refused.
///
/// Pins the shape choice. A future distilled model with two baked schedules can declare both
/// without the guard degrading into "anything between the smallest and largest".
#[test]
fn supported_steps_is_a_set_not_a_range() {
    let two_schedules = Capabilities {
        supported_steps: StepSupport::Exact(vec![4, 8]),
        ..caps()
    };
    let at = |steps: u32| {
        two_schedules.validate_request(
            "m",
            &GenerationRequest {
                steps: Some(steps),
                ..base_req()
            },
        )
    };
    assert!(at(4).is_ok());
    assert!(at(8).is_ok());
    let gap = at(6)
        .expect_err("6 sits between the two schedules and belongs to neither")
        .to_string();
    // The plural arm names every legal count, so the caller can pick one rather than guess.
    assert!(
        gap.contains("4 or 8") && gap.contains("steps=6"),
        "the multi-schedule refusal must list the legal counts: {gap}"
    );
}

/// sc-19559 — the ceiling. SVD-XT's `MAX_STEPS = 200` used to live only inside the two SVD
/// providers' `validate`; the shared floor now enforces the advertised range, and refuses on
/// BOTH ends of it.
///
/// The `200` boundary pair is the load-bearing part: an off-by-one that made the range
/// exclusive would refuse the model's own top count, and one that never fired would admit
/// `201` — asserting only the far-outside `10_000` would catch neither.
#[test]
fn an_advertised_step_range_refuses_both_ends_and_admits_the_interior() {
    let bounded = Capabilities {
        supported_steps: StepSupport::Range { min: 2, max: 200 },
        ..caps()
    };
    let at = |steps: Option<u32>| {
        bounded.validate_request(
            "svd_xt",
            &GenerationRequest {
                steps,
                ..base_req()
            },
        )
    };

    for ok in [2, 3, 25, 199, 200] {
        assert!(at(Some(ok)).is_ok(), "{ok} is inside 2..=200 and must pass");
    }
    assert!(
        at(None).is_ok(),
        "an omitted count must use the model's default, not be refused"
    );

    for bad in [1, 201, 10_000] {
        let err = at(Some(bad)).unwrap_err().to_string();
        assert!(
            err.contains("svd_xt") && err.contains(&bad.to_string()) && err.contains("2..=200"),
            "the refusal must name the model, the request and the legal range: {err}"
        );
    }
}

/// sc-19559 — the ceiling is readable from the descriptor, which is the whole point: a
/// consumer must not have to dispatch a job to learn the bound.
///
/// Also pins that a bound is NOT a default: `floor()` is the smallest legal count, and the
/// step count a request omitting `steps` actually gets is the model's own baked default,
/// which the capability surface deliberately does not carry.
#[test]
fn a_declared_bound_is_readable_and_is_not_a_default() {
    assert_eq!(StepSupport::Unconstrained.ceiling(), None);
    assert_eq!(StepSupport::Unconstrained.floor(), None);

    let range = StepSupport::Range { min: 2, max: 200 };
    assert_eq!(range.ceiling(), Some(200));
    assert_eq!(range.floor(), Some(2));

    // An exact menu's ceiling is its largest member, not its declaration order.
    let menu = StepSupport::Exact(vec![8, 4]);
    assert_eq!(menu.ceiling(), Some(8));
    assert_eq!(menu.floor(), Some(4));

    // A model that declares a bound still renders its own default when `steps` is omitted —
    // the floor never substitutes the bound for the missing value.
    let bounded = Capabilities {
        supported_steps: range,
        ..caps()
    };
    let req = GenerationRequest {
        steps: None,
        ..base_req()
    };
    assert!(bounded.validate_request("m", &req).is_ok());
    assert_eq!(
        req.steps, None,
        "validation must not seed `steps` from a bound"
    );
}

#[test]
fn non_finite_extended_float_knobs_are_rejected() {
    // F-001: the finiteness floor now covers every `Option<f32>` knob added after F-053, not just
    // guidance/true_cfg. A guidance-capable caps so the support gate never fires; each knob is
    // exercised with NaN/±Inf and must produce a typed `Msg` naming the field.
    let c = Capabilities {
        supports_guidance: true,
        supports_true_cfg: true,
        conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
        ..caps()
    };
    type Build = fn(f32) -> GenerationRequest;
    let mk: [(&str, Build); 12] = [
        ("guidance_eta", |v| GenerationRequest {
            guidance_eta: Some(v),
            ..base_req()
        }),
        ("guidance_momentum", |v| GenerationRequest {
            guidance_momentum: Some(v),
            ..base_req()
        }),
        ("guidance_norm_threshold", |v| GenerationRequest {
            guidance_norm_threshold: Some(v),
            ..base_req()
        }),
        ("strength", |v| GenerationRequest {
            strength: Some(v),
            ..base_req()
        }),
        ("control_scale", |v| GenerationRequest {
            control_scale: Some(v),
            ..base_req()
        }),
        ("image_guidance", |v| GenerationRequest {
            image_guidance: Some(v),
            ..base_req()
        }),
        ("scheduler_shift", |v| GenerationRequest {
            scheduler_shift: Some(v),
            ..base_req()
        }),
        ("motion_bucket_id", |v| GenerationRequest {
            motion_bucket_id: Some(v),
            ..base_req()
        }),
        ("noise_aug_strength", |v| GenerationRequest {
            noise_aug_strength: Some(v),
            ..base_req()
        }),
        ("softness", |v| GenerationRequest {
            softness: Some(v),
            ..base_req()
        }),
        ("enhance_temperature", |v| GenerationRequest {
            enhance_temperature: Some(v),
            ..base_req()
        }),
        ("pid_capture_sigma", |v| GenerationRequest {
            pid_capture_sigma: Some(v),
            ..base_req()
        }),
    ];
    for (field, build) in mk {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let req = build(bad);
            let err = c.validate_request("m", &req).unwrap_err();
            assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
            assert!(
                err.to_string().contains(field) && err.to_string().contains("must be finite"),
                "{field} {bad}: got {err}"
            );
        }
    }
    // The Control-branch scale carried inside a conditioning entry is guarded too (F-001).
    let ctrl = GenerationRequest {
        conditioning: vec![Conditioning::Control {
            image: img(8, 8),
            kind: ControlKind::Pose,
            scale: Some(f32::NAN),
        }],
        ..base_req()
    };
    let err = c.validate_request("m", &ctrl).unwrap_err();
    assert!(
        err.to_string().contains("conditioning.control.scale"),
        "got {err}"
    );
    // A fully-finite request across the extended knobs still passes.
    assert!(c
        .validate_request(
            "m",
            &GenerationRequest {
                guidance_eta: Some(1.0),
                guidance_momentum: Some(0.0),
                strength: Some(0.6),
                control_scale: Some(1.0),
                ..base_req()
            }
        )
        .is_ok());
}

#[test]
fn oversized_counters_hit_the_sanity_cap() {
    // F-004: the floor now rejects a pathological `steps` / video-counter value that would launch
    // an effectively-unbounded run. Model-realistic values still pass.
    let c = Capabilities {
        max_size: 4096,
        ..caps()
    };
    let steps = GenerationRequest {
        steps: Some(u32::MAX),
        ..base_req()
    };
    assert!(
        c.validate_request("m", &steps)
            .unwrap_err()
            .to_string()
            .contains("steps"),
        "u32::MAX steps must be capped"
    );
    let frames = GenerationRequest {
        frames: Some(u32::MAX),
        ..base_req()
    };
    assert!(
        c.validate_request("m", &frames).is_err(),
        "u32::MAX frames capped"
    );
    let fps = GenerationRequest {
        fps: Some(u32::MAX),
        ..base_req()
    };
    assert!(
        c.validate_request("m", &fps).is_err(),
        "u32::MAX fps capped"
    );
    let zero_fps = GenerationRequest {
        fps: Some(0),
        ..base_req()
    };
    let err = c
        .validate_request("m", &zero_fps)
        .expect_err("zero fps must be rejected before a provider derives durations");
    assert!(matches!(err, Error::Msg(_)), "got: {err:?}");
    assert!(err.to_string().contains("fps must be greater than zero"));
    // Realistic values pass.
    assert!(c
        .validate_request(
            "m",
            &GenerationRequest {
                steps: Some(50),
                frames: Some(121),
                fps: Some(24),
                ..base_req()
            }
        )
        .is_ok());
}

#[test]
fn capability_gaps_return_typed_unsupported() {
    // F-008: capability-gap branches must be `Error::Unsupported`, not `Msg`, so candle gating /
    // the worker can distinguish them. Malformed-value branches (range/finiteness) stay `Msg`.
    let c = caps();
    let gap_cases: Vec<GenerationRequest> = vec![
        GenerationRequest {
            negative_prompt: Some("n".into()),
            ..base_req()
        },
        GenerationRequest {
            guidance: Some(3.5),
            ..base_req()
        },
        GenerationRequest {
            true_cfg: Some(4.0),
            ..base_req()
        },
        GenerationRequest {
            sampler: Some("unipc".into()),
            ..base_req()
        },
        GenerationRequest {
            scheduler: Some("linear".into()),
            ..base_req()
        },
        GenerationRequest {
            guidance_method: Some("apg".into()),
            ..base_req()
        },
        GenerationRequest {
            conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
            ..base_req()
        },
    ];
    for (i, req) in gap_cases.iter().enumerate() {
        let err = c.validate_request("m", req).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "gap case {i} should be typed Unsupported, got {err:?}"
        );
    }
}

/// A pure-audio capability surface (sc-12834): no visual size bounds (unused for
/// `Modality::Audio`), an advertised voice/language/sample-rate surface, a 60 s cap, and
/// `ReferenceAudio` conditioning.
fn audio_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::ReferenceAudio],
        audio_sample_rates: vec![24_000, 48_000],
        max_audio_duration_secs: Some(60.0),
        audio_voices: vec!["nova"],
        audio_languages: vec!["en"],
        max_count: 1,
        ..Default::default()
    }
}

fn track() -> AudioTrack {
    AudioTrack {
        samples: vec![0.0; 16],
        sample_rate: 24_000,
        channels: 1,
        ..Default::default()
    }
}

/// A TTS-shaped request: prompt + typed audio sub-block, size left at the unused 0x0.
fn audio_req() -> GenerationRequest {
    GenerationRequest {
        prompt: "read this aloud".into(),
        width: 0,
        height: 0,
        audio: Some(AudioParams {
            voice: Some("nova".into()),
            language: Some("en".into()),
            target_duration: Some(12.5),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn audio_request_validates_against_audio_descriptor() {
    // sc-12834 acceptance: a `GenerationRequest { prompt, audio: Some(..) }` validates against
    // an audio Capabilities descriptor through the size-skipping audio floor.
    let c = audio_caps();
    assert!(c.validate_request_audio("tts", &audio_req()).is_ok());
    // A reference-audio conditioned request (voice cloning) passes when advertised.
    let cloned = GenerationRequest {
        conditioning: vec![Conditioning::ReferenceAudio {
            audio: track(),
            strength: Some(0.8),
        }],
        ..audio_req()
    };
    assert!(c.validate_request_audio("tts", &cloned).is_ok());
    // The music-shaped knobs are free-form/positive-gated, not membership-gated.
    let music = GenerationRequest {
        audio: Some(AudioParams {
            target_duration: Some(30.0),
            bpm: Some(120.0),
            musical_key: Some("C minor".into()),
            lyrics: Some("la la la".into()),
            ..Default::default()
        }),
        ..audio_req()
    };
    assert!(c.validate_request_audio("music", &music).is_ok());
}

#[test]
fn audio_floor_rejects_visual_only_mismatches_with_typed_errors() {
    // sc-12834 acceptance: the audio floor still runs every non-size check — capability gaps
    // are typed `Unsupported`, malformed values are `Msg`.
    let c = audio_caps();
    // Capability gaps (visual-only surface the audio descriptor does not advertise).
    let gap_cases: Vec<GenerationRequest> = vec![
        GenerationRequest {
            negative_prompt: Some("n".into()),
            ..audio_req()
        },
        GenerationRequest {
            guidance: Some(3.5),
            ..audio_req()
        },
        GenerationRequest {
            sampler: Some("euler".into()),
            ..audio_req()
        },
        GenerationRequest {
            conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
            ..audio_req()
        },
    ];
    for (i, req) in gap_cases.iter().enumerate() {
        let err = c.validate_request_audio("tts", req).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "gap case {i} should be typed Unsupported, got {err:?}"
        );
    }
    // Malformed values stay `Msg`.
    let msg_cases: Vec<GenerationRequest> = vec![
        GenerationRequest {
            count: 0,
            ..audio_req()
        },
        GenerationRequest {
            count: 2,
            ..audio_req()
        },
        GenerationRequest {
            steps: Some(0),
            ..audio_req()
        },
    ];
    for (i, req) in msg_cases.iter().enumerate() {
        let err = c.validate_request_audio("tts", req).unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)),
            "msg case {i} should be a Msg range error, got {err:?}"
        );
    }
}

#[test]
fn multi_speaker_script_gating_is_additive_and_typed() {
    // sc-12848: a script is a capability gap on a non-multi-speaker model (the default), and
    // gated by supports_multi_speaker / max_speakers / the closed voice surface when advertised.
    let seg = |sp: &str| SpeechSegment {
        text: "hello".into(),
        speaker: Some(sp.into()),
        ..Default::default()
    };
    let script_req =
        |caps_voices: Vec<&'static str>, ms: bool, max: Option<u32>, segs: Vec<SpeechSegment>| {
            let c = Capabilities {
                audio_voices: caps_voices,
                supports_multi_speaker: ms,
                max_speakers: max,
                max_count: 1,
                ..Default::default()
            };
            let req = GenerationRequest {
                audio: Some(AudioParams {
                    script: Some(segs),
                    ..Default::default()
                }),
                ..audio_req()
            };
            c.validate_request_audio("tts", &req)
        };

    // A single-voice model with no script capability: a script is a typed Unsupported.
    assert!(matches!(
        script_req(vec![], false, None, vec![seg("S1"), seg("S2")]),
        Err(Error::Unsupported(_))
    ));

    // Advertised multi-speaker, opaque labels (empty voice surface): a valid script passes.
    assert!(script_req(vec![], true, Some(2), vec![seg("S1"), seg("S2")]).is_ok());

    // An empty script is a malformed request (Msg), not a capability gap.
    assert!(matches!(
        script_req(vec![], true, None, vec![]),
        Err(Error::Msg(_))
    ));

    // Over the max_speakers cap → range error (Msg).
    assert!(matches!(
        script_req(vec![], true, Some(2), vec![seg("S1"), seg("S2"), seg("S3")]),
        Err(Error::Msg(_))
    ));
    // At the cap, distinct-count dedups repeated labels → OK.
    assert!(script_req(vec![], true, Some(2), vec![seg("S1"), seg("S1"), seg("S2")]).is_ok());

    // A closed voice surface gates script speakers exactly like `audio.voice` (typed Unsupported).
    assert!(script_req(
        vec!["nova", "onyx"],
        true,
        None,
        vec![seg("nova"), seg("onyx")]
    )
    .is_ok());
    assert!(matches!(
        script_req(vec!["nova"], true, None, vec![seg("nova"), seg("ghost")]),
        Err(Error::Unsupported(_))
    ));

    // The additive floor: a request with NO script behaves exactly as before (single-voice).
    let c = Capabilities {
        audio_voices: vec!["nova"],
        max_count: 1,
        ..Default::default()
    };
    let single = GenerationRequest {
        audio: Some(AudioParams {
            voice: Some("nova".into()),
            ..Default::default()
        }),
        ..audio_req()
    };
    assert!(c.validate_request_audio("tts", &single).is_ok());
}

#[test]
fn non_finite_song_sampling_is_named_by_the_float_floor() {
    // sc-22994: the symbolic-song sampling floats are classified by the same non-finite floor
    // as every other request float, and the reported name says which phase and field.
    type Set = fn(&mut TokenSampling, f64);
    let fields: [(&str, Set); 3] = [
        ("temperature", |s, v| s.temperature = Some(v)),
        ("top_p", |s, v| s.top_p = Some(v)),
        ("repetition_penalty", |s, v| s.repetition_penalty = Some(v)),
    ];
    for semantic in [false, true] {
        for (field, set) in fields {
            for bad in [f64::NAN, f64::INFINITY] {
                let mut sampling = TokenSampling::default();
                set(&mut sampling, bad);
                let song = if semantic {
                    SongParams {
                        semantic_sampling: Some(sampling),
                        ..Default::default()
                    }
                } else {
                    SongParams {
                        score_sampling: Some(sampling),
                        ..Default::default()
                    }
                };
                let req = GenerationRequest {
                    audio: Some(AudioParams {
                        song: Some(song),
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let phase = if semantic { "semantic" } else { "score" };
                let (name, _) = req.first_nonfinite_float().expect("non-finite is caught");
                assert_eq!(name, format!("audio.song.{phase}_sampling.{field}"));
            }
        }
    }
    let finite = GenerationRequest {
        audio: Some(AudioParams {
            song: Some(SongParams {
                score_sampling: Some(TokenSampling {
                    temperature: Some(0.7),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert_eq!(finite.first_nonfinite_float(), None);
}

#[test]
fn segmented_song_controls_are_refused_unless_advertised() {
    // sc-19382: each field is a typed capability gap on a model that does not read it, and
    // passes once its own flag is advertised.
    type Set = fn(&mut AudioParams);
    type Flag = fn(&mut Capabilities);
    let cases: [(&str, Set, Flag); 7] = [
        (
            "segments",
            |a| a.segments = Some(3),
            |c| c.supports_segmented_lyrics = true,
        ),
        (
            "max_new_tokens_per_segment",
            |a| a.max_new_tokens_per_segment = Some(100),
            |c| c.supports_segmented_lyrics = true,
        ),
        (
            "repetition_penalty",
            |a| a.repetition_penalty = Some(1.1),
            |c| c.supports_repetition_penalty = true,
        ),
        (
            "reference_region",
            |a| {
                a.reference_region = Some(TimeRegion {
                    start_secs: 0.0,
                    end_secs: Some(10.0),
                })
            },
            |c| c.supports_reference_region = true,
        ),
        (
            "output_limiter",
            |a| a.output_limiter = Some(OutputLimiter::Rescale),
            |c| c.supports_output_limiter = true,
        ),
        (
            "song",
            |a| {
                a.song = Some(SongParams {
                    planning: Some(SongPlanning::Melody),
                    ..Default::default()
                })
            },
            |c| c.supports_symbolic_song = true,
        ),
        (
            "artifacts",
            |a| {
                a.artifacts = Some(AudioArtifacts {
                    dir: "run".into(),
                    resume: false,
                })
            },
            |c| c.supports_audio_artifacts = true,
        ),
    ];
    for (name, set, flag) in cases {
        let mut req = audio_req();
        set(req.audio.as_mut().unwrap());
        let mut c = Capabilities {
            audio_voices: vec!["nova"],
            audio_languages: vec!["en"],
            audio_sample_rates: vec![24_000],
            max_count: 1,
            ..Default::default()
        };
        assert!(
            matches!(
                c.validate_request_audio("tts", &req),
                Err(Error::Unsupported(_))
            ),
            "{name} must be refused when not advertised"
        );
        flag(&mut c);
        assert!(
            c.validate_request_audio("tts", &req).is_ok(),
            "{name} must pass when advertised"
        );
    }
}

#[test]
fn plan_only_and_cover_are_gated_and_plan_only_needs_artifacts() {
    // sc-22988: each is a typed capability gap unless its own flag is advertised; a plan-only
    // request without a record to publish into is a malformed request, not a capability gap.
    let symbolic = || Capabilities {
        audio_sample_rates: vec![48_000],
        max_count: 1,
        supports_symbolic_song: true,
        supports_audio_artifacts: true,
        ..Default::default()
    };
    let req = |song: SongParams, artifacts: bool| GenerationRequest {
        prompt: "a song".into(),
        audio: Some(AudioParams {
            song: Some(song),
            artifacts: artifacts.then(|| AudioArtifacts {
                dir: "run".into(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let plan_only = SongParams {
        plan_only: true,
        ..Default::default()
    };
    let cover = SongParams {
        cover: Some(SongCover {
            mode: SongCoverMode::Melody,
            score: "X:1".into(),
            keep: None,
            translated_from: None,
        }),
        ..Default::default()
    };
    let mut caps = symbolic();
    for (name, song) in [("plan_only", &plan_only), ("cover", &cover)] {
        let got = caps.validate_request_audio("song", &req(song.clone(), true));
        assert!(
            matches!(&got, Err(Error::Unsupported(m)) if m.contains(name)),
            "{name} must be refused when not advertised: {got:?}"
        );
    }
    caps.supports_song_plan_only = true;
    caps.supports_song_cover = true;
    caps.validate_request_audio("song", &req(plan_only.clone(), true))
        .unwrap();
    caps.validate_request_audio("song", &req(cover, false))
        .unwrap();
    let orphan = caps.validate_request_audio("song", &req(plan_only, false));
    assert!(
        matches!(&orphan, Err(Error::Msg(m)) if m.contains("requires audio.artifacts")),
        "{orphan:?}"
    );
}

#[test]
fn cfg_interval_prompt_batch_and_caption_overflow_are_gated_on_the_floor() {
    // sc-25681: each control is refused by a model that does not advertise it, so it can never
    // be accepted and ignored; an advertising model still gets the shape checks.
    let plain = Capabilities {
        min_size: 64,
        max_size: 2048,
        max_count: 4,
        ..Default::default()
    };
    let all = Capabilities {
        supports_cfg_interval: true,
        supports_prompt_batch: true,
        supports_caption_overflow_policy: true,
        ..plain.clone()
    };
    let base = || GenerationRequest {
        prompt: "p".into(),
        ..Default::default()
    };
    let interval = GenerationRequest {
        cfg_interval: Some((0.1, 0.9)),
        ..base()
    };
    let batch = GenerationRequest {
        prompt: String::new(),
        prompt_batch: vec!["a".into(), "b".into()],
        count: 2,
        ..Default::default()
    };
    let overflow = GenerationRequest {
        caption_overflow: Some(CaptionOverflowPolicy::Error),
        ..base()
    };
    for req in [&interval, &batch, &overflow] {
        assert!(matches!(
            plain.validate_request("m", req),
            Err(Error::Unsupported(_))
        ));
        all.validate_request("m", req).unwrap();
    }
    let inverted = GenerationRequest {
        cfg_interval: Some((0.9, 0.1)),
        ..base()
    };
    assert!(all.validate_request("m", &inverted).is_err());
    let nan = GenerationRequest {
        cfg_interval: Some((f32::NAN, 0.5)),
        ..base()
    };
    assert_eq!(
        nan.first_nonfinite_float().map(|(n, _)| n),
        Some("cfg_interval.lo")
    );
    let both = GenerationRequest {
        prompt: "p".into(),
        ..batch.clone()
    };
    assert!(all.validate_request("m", &both).is_err());
    let too_many = GenerationRequest { count: 3, ..batch };
    assert!(all.validate_request("m", &too_many).is_err());
    for policy in [
        CaptionOverflowPolicy::Warn,
        CaptionOverflowPolicy::Error,
        CaptionOverflowPolicy::Silent,
    ] {
        assert_eq!(
            CaptionOverflowPolicy::from_name(policy.name()),
            Some(policy)
        );
    }
}

#[test]
fn generate_with_report_defaults_to_the_bare_output() {
    // sc-22988: the additive entry point wraps `generate` for every provider that does not
    // override it — no record, no warnings, the same output.
    struct Plain(ModelDescriptor);
    impl Generator for Plain {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.0
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            on_progress(Progress::Decoding);
            Ok(GenerationOutput::Audio(AudioTrack {
                samples: vec![0.25, -0.25],
                sample_rate: 48_000,
                channels: 2,
                stems: Vec::new(),
            }))
        }
    }
    let generator = Plain(ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "plain",
        family: "test",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            max_count: 1,
            ..Default::default()
        },
    });
    assert_eq!(generator.moss_image_owned_weight_bytes(), None);
    let mut progress = Vec::new();
    let report = generator
        .generate_with_report(&GenerationRequest::default(), &mut |p| progress.push(p))
        .unwrap();
    assert_eq!(progress, [Progress::Decoding], "progress is forwarded");
    assert!(report.artifacts.is_none() && report.warnings.is_empty());
    match report.output {
        Some(GenerationOutput::Audio(track)) => assert_eq!(track.samples, [0.25, -0.25]),
        other => panic!("expected the generated audio, got {other:?}"),
    }
}

#[test]
fn conversation_history_gating_is_additive_and_typed() {
    // sc-14150: a ConversationHistory is a capability gap on a non-conversational model, gated by
    // supports_conversation_history (+ the conditioning allowlist); when supported the shape must
    // be well-formed; a request with no conversation is byte-for-byte unaffected.
    let user = |t: &str| ConversationTurn {
        role: ConversationRole::User,
        text: t.into(),
        audio: Some(track()),
    };
    let asst = |t: &str, audio: Option<AudioTrack>| ConversationTurn {
        role: ConversationRole::Assistant,
        text: t.into(),
        audio,
    };
    // A bare audio request (no voice/language sub-block, so only the conversation is exercised).
    let conv_req = |caps: Capabilities, turns: Vec<ConversationTurn>| {
        let req = GenerationRequest {
            prompt: "read this aloud".into(),
            width: 0,
            height: 0,
            conditioning: vec![Conditioning::ConversationHistory { turns }],
            ..Default::default()
        };
        caps.validate_request_audio("tts", &req)
    };
    let conv_caps = || Capabilities {
        conditioning: vec![ConditioningKind::ConversationHistory],
        supports_conversation_history: true,
        max_count: 1,
        ..Default::default()
    };

    // A non-conversational model (neither the flag nor the kind): a conversation is a typed gap.
    let plain = Capabilities {
        max_count: 1,
        ..Default::default()
    };
    assert!(matches!(
        conv_req(plain, vec![user("hi"), asst("hello", None)]),
        Err(Error::Unsupported(_))
    ));

    // Advertised: a well-formed conversation (context user turn + an assistant reply to
    // synthesize) passes.
    assert!(conv_req(conv_caps(), vec![user("hi"), asst("hello", None)]).is_ok());

    // Empty conversation → malformed (Msg), not a capability gap.
    assert!(matches!(conv_req(conv_caps(), vec![]), Err(Error::Msg(_))));

    // A blank-text turn → Msg.
    assert!(matches!(
        conv_req(conv_caps(), vec![user("   "), asst("hi", None)]),
        Err(Error::Msg(_))
    ));

    // A User turn with no audio (provided context must carry its audio) → Msg.
    assert!(matches!(
        conv_req(
            conv_caps(),
            vec![
                ConversationTurn {
                    role: ConversationRole::User,
                    text: "hi".into(),
                    audio: None,
                },
                asst("hello", None),
            ]
        ),
        Err(Error::Msg(_))
    ));

    // No assistant turn to synthesize (every turn already carries audio) → Msg.
    assert!(matches!(
        conv_req(
            conv_caps(),
            vec![user("hi"), asst("was said", Some(track()))]
        ),
        Err(Error::Msg(_))
    ));

    // Keyed gate is authoritative: a descriptor listing the kind but leaving the flag unset
    // (inconsistent — the registry conformance sweep also flags it) still rejects a conversation
    // as the typed Unsupported.
    let flag_off = Capabilities {
        conditioning: vec![ConditioningKind::ConversationHistory],
        supports_conversation_history: false,
        max_count: 1,
        ..Default::default()
    };
    assert!(matches!(
        conv_req(flag_off, vec![user("hi"), asst("hello", None)]),
        Err(Error::Unsupported(_))
    ));

    // Additive: a plain request with no conversation validates exactly as before.
    let bare = GenerationRequest {
        prompt: "hi".into(),
        width: 0,
        height: 0,
        ..Default::default()
    };
    let plain2 = Capabilities {
        max_count: 1,
        ..Default::default()
    };
    assert!(plain2.validate_request_audio("tts", &bare).is_ok());
}

#[test]
fn audio_surface_membership_and_ranges_are_enforced() {
    let c = audio_caps();
    let with_audio = |a: AudioParams| GenerationRequest {
        audio: Some(a),
        ..audio_req()
    };
    // Membership gaps → typed Unsupported naming the field.
    let gaps: [(&str, AudioParams); 3] = [
        (
            "audio.voice",
            AudioParams {
                voice: Some("santa".into()),
                ..Default::default()
            },
        ),
        (
            "audio.language",
            AudioParams {
                language: Some("xx".into()),
                ..Default::default()
            },
        ),
        (
            "audio.sample_rate",
            AudioParams {
                sample_rate: Some(44_100),
                ..Default::default()
            },
        ),
    ];
    for (field, params) in gaps {
        let err = c
            .validate_request_audio("tts", &with_audio(params))
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{field}: got {err:?}");
        assert!(err.to_string().contains(field), "{field}: got {err}");
    }
    // Range violations → Msg.
    let over_cap = with_audio(AudioParams {
        target_duration: Some(61.0),
        ..Default::default()
    });
    let err = c.validate_request_audio("tts", &over_cap).unwrap_err();
    assert!(matches!(err, Error::Msg(_)), "got {err:?}");
    assert!(
        err.to_string().contains("audio.target_duration"),
        "got {err}"
    );
    for bad in [0.0, -3.0] {
        let req = with_audio(AudioParams {
            target_duration: Some(bad),
            ..Default::default()
        });
        assert!(
            matches!(
                c.validate_request_audio("tts", &req).unwrap_err(),
                Error::Msg(_)
            ),
            "target_duration {bad} must be rejected"
        );
    }
    let bad_bpm = with_audio(AudioParams {
        bpm: Some(0.0),
        ..Default::default()
    });
    assert!(matches!(
        c.validate_request_audio("tts", &bad_bpm).unwrap_err(),
        Error::Msg(_)
    ));
    // No advertised duration cap ⇒ only the sanity cap applies.
    let uncapped = Capabilities {
        max_audio_duration_secs: None,
        ..audio_caps()
    };
    assert!(uncapped
        .validate_request_audio(
            "tts",
            &with_audio(AudioParams {
                target_duration: Some(3600.0),
                ..Default::default()
            })
        )
        .is_ok());
    assert!(uncapped
        .validate_request_audio(
            "tts",
            &with_audio(AudioParams {
                target_duration: Some(MAX_DURATION_SECS + 1.0),
                ..Default::default()
            })
        )
        .is_err());
}

#[test]
fn audio_floats_join_the_finiteness_floor() {
    // The `AudioParams` floats inherit the F-053/F-001 finiteness floor by construction.
    let c = audio_caps();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (field, params) in [
            (
                "audio.target_duration",
                AudioParams {
                    target_duration: Some(bad),
                    ..Default::default()
                },
            ),
            (
                "audio.bpm",
                AudioParams {
                    bpm: Some(bad),
                    ..Default::default()
                },
            ),
        ] {
            let req = GenerationRequest {
                audio: Some(params),
                ..audio_req()
            };
            let err = c.validate_request_audio("tts", &req).unwrap_err();
            assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
            // ±Inf may trip the range/sanity-cap branch first (same convention as the
            // request-level `duration`); NaN falls through every comparison and must be
            // caught by the finiteness floor naming the field.
            assert!(err.to_string().contains(field), "{field} {bad}: got {err}");
            if bad.is_nan() {
                assert!(
                    err.to_string().contains("must be finite"),
                    "{field} NaN: got {err}"
                );
            }
        }
    }
    // The ReferenceAudio conditioning strength is guarded too.
    let cloned = GenerationRequest {
        conditioning: vec![Conditioning::ReferenceAudio {
            audio: track(),
            strength: Some(f32::NAN),
        }],
        ..audio_req()
    };
    let err = c.validate_request_audio("tts", &cloned).unwrap_err();
    assert!(
        err.to_string()
            .contains("conditioning.reference_audio.strength"),
        "got {err}"
    );
}

#[test]
fn reference_audio_is_gated_by_the_conditioning_allowlist() {
    // A visual descriptor that does not advertise ReferenceAudio rejects it, typed.
    let visual = caps();
    let req = GenerationRequest {
        conditioning: vec![Conditioning::ReferenceAudio {
            audio: track(),
            strength: None,
        }],
        ..base_req()
    };
    let err = visual.validate_request("m", &req).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    assert_eq!(
        Conditioning::ReferenceAudio {
            audio: track(),
            strength: None
        }
        .kind(),
        ConditioningKind::ReferenceAudio
    );
}

#[test]
fn audio_output_and_modality_variants_carry_the_track() {
    // The additive output variant round-trips the host-type track (tensor-free invariant).
    let out = GenerationOutput::Audio(track());
    match out {
        GenerationOutput::Audio(t) => {
            assert_eq!(t.sample_rate, 24_000);
            assert_eq!(t.channels, 1);
            assert_eq!(t.samples.len(), 16);
        }
        other => panic!("expected Audio output, got {other:?}"),
    }
    assert_ne!(Modality::Audio, Modality::Both);
    // A visual request is untouched by the audio block: `Default` carries `audio: None`.
    assert!(GenerationRequest::default().audio.is_none());
}

#[test]
fn non_finite_guidance_and_true_cfg_are_rejected() {
    // F-053: a NaN passes `x > 1.0`-style checks; the floor rejects non-finite explicitly. Uses
    // a caps that advertises guidance/true_cfg so the finiteness branch (not the support gate) runs.
    let c = Capabilities {
        supports_guidance: true,
        supports_true_cfg: true,
        ..caps()
    };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let g = GenerationRequest {
            guidance: Some(bad),
            ..base_req()
        };
        let err = c.validate_request("m", &g).unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)),
            "guidance {bad} → Msg range error"
        );
        assert!(err.to_string().contains("guidance must be finite"));
        let t = GenerationRequest {
            true_cfg: Some(bad),
            ..base_req()
        };
        assert!(matches!(
            c.validate_request("m", &t).unwrap_err(),
            Error::Msg(_)
        ));
    }
    // Finite guidance/true_cfg still pass.
    assert!(c
        .validate_request(
            "m",
            &GenerationRequest {
                guidance: Some(3.5),
                true_cfg: Some(2.0),
                ..base_req()
            }
        )
        .is_ok());
}

// ---- Prompted audio editing (sc-12847) -------------------------------------------------

/// An audio-edit capability surface: admits the `AudioEdit` kind and advertises two modes.
fn edit_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::AudioEdit],
        audio_edit_modes: vec![AudioEditMode::Repaint, AudioEditMode::Extend],
        min_size: 1,
        max_size: 4096,
        max_count: 1,
        ..Default::default()
    }
}

fn edit_req(
    mode: AudioEditMode,
    region: Option<TimeRegion>,
    strength: Option<f32>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: "x".into(),
        width: 512,
        height: 512,
        conditioning: vec![Conditioning::AudioEdit {
            audio: track(),
            mode,
            region,
            strength,
        }],
        ..Default::default()
    }
}

#[test]
fn audio_edit_kind_and_accessor_round_trip() {
    let region = TimeRegion {
        start_secs: 4.0,
        end_secs: Some(8.0),
    };
    let req = edit_req(AudioEditMode::Repaint, Some(region), Some(0.7));
    assert_eq!(
        req.conditioning[0].kind(),
        ConditioningKind::AudioEdit,
        "AudioEdit maps to its own kind"
    );
    let e = req.audio_edit().expect("audio_edit present");
    assert_eq!(e.mode, AudioEditMode::Repaint);
    assert_eq!(e.region, Some(region));
    assert_eq!(e.strength, Some(0.7));
    assert_eq!(e.audio.sample_rate, 24_000);
    // A request without an AudioEdit yields None.
    assert!(GenerationRequest::default().audio_edit().is_none());
}

#[test]
fn audio_edit_mode_is_gated_by_the_advertised_surface() {
    let c = edit_caps();
    // Advertised modes pass; the region is well-formed.
    assert!(c
        .validate_request(
            "m",
            &edit_req(
                AudioEditMode::Repaint,
                Some(TimeRegion {
                    start_secs: 4.0,
                    end_secs: Some(8.0),
                }),
                None,
            ),
        )
        .is_ok());
    // An unadvertised mode is a typed capability gap.
    let err = c
        .validate_request("m", &edit_req(AudioEditMode::Cover, None, None))
        .unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "unlisted mode → Unsupported"
    );
    assert!(err.to_string().contains("unsupported audio edit mode"));
    // The whole kind is rejected when not admitted at all.
    let no_edit = Capabilities {
        conditioning: vec![ConditioningKind::Reference],
        ..edit_caps()
    };
    assert!(matches!(
        no_edit
            .validate_request("m", &edit_req(AudioEditMode::Repaint, None, None))
            .unwrap_err(),
        Error::Unsupported(_)
    ));
}

#[test]
fn audio_edit_region_and_strength_are_floored() {
    let c = edit_caps();
    // start < 0 and end <= start are malformed ranges → Msg.
    for region in [
        TimeRegion {
            start_secs: -1.0,
            end_secs: Some(4.0),
        },
        TimeRegion {
            start_secs: 8.0,
            end_secs: Some(4.0),
        },
        TimeRegion {
            start_secs: 4.0,
            end_secs: Some(4.0),
        },
    ] {
        let err = c
            .validate_request("m", &edit_req(AudioEditMode::Repaint, Some(region), None))
            .unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "{region:?} → Msg range error");
    }
    // Non-finite strength / region bounds are caught by the finiteness floor.
    for bad in [f32::NAN, f32::INFINITY] {
        assert!(c
            .validate_request("m", &edit_req(AudioEditMode::Repaint, None, Some(bad)))
            .is_err());
        assert!(c
            .validate_request(
                "m",
                &edit_req(
                    AudioEditMode::Repaint,
                    Some(TimeRegion {
                        start_secs: bad,
                        end_secs: None,
                    }),
                    None,
                ),
            )
            .is_err());
        assert!(c
            .validate_request(
                "m",
                &edit_req(
                    AudioEditMode::Repaint,
                    Some(TimeRegion {
                        start_secs: 1.0,
                        end_secs: Some(bad),
                    }),
                    None,
                ),
            )
            .is_err());
    }
    // A well-formed open-ended region (end None) passes.
    assert!(c
        .validate_request(
            "m",
            &edit_req(
                AudioEditMode::Extend,
                Some(TimeRegion {
                    start_secs: 2.0,
                    end_secs: None,
                }),
                Some(0.5),
            ),
        )
        .is_ok());
}

// ---- Multi-region prompted audio editing (sc-14549) ------------------------------------

/// A **multi-region** capability surface: admits the distinct `AudioEditRegions` kind.
fn multi_edit_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::AudioEditRegions],
        ..edit_caps()
    }
}

fn multi_edit_req(mode: AudioEditMode, regions: Vec<TimeRegion>) -> GenerationRequest {
    GenerationRequest {
        prompt: "x".into(),
        width: 512,
        height: 512,
        conditioning: vec![Conditioning::AudioEditRegions {
            audio: track(),
            mode,
            regions,
            strength: None,
        }],
        ..Default::default()
    }
}

fn span(start_secs: f32, end_secs: f32) -> TimeRegion {
    TimeRegion {
        start_secs,
        end_secs: Some(end_secs),
    }
}

#[test]
fn multi_region_audio_edit_kind_and_accessor_round_trip() {
    let regions = vec![span(2.0, 6.0), span(14.0, 18.0)];
    let req = multi_edit_req(AudioEditMode::Repaint, regions.clone());
    assert_eq!(
        req.conditioning[0].kind(),
        ConditioningKind::AudioEditRegions,
        "AudioEditRegions maps to its OWN kind — that is what makes default-deny free"
    );
    let e = req
        .audio_edit_regions()
        .expect("audio_edit_regions present");
    assert_eq!(e.mode, AudioEditMode::Repaint);
    assert_eq!(e.regions, regions.as_slice());
    assert_eq!(e.strength, None);
    assert_eq!(e.audio.sample_rate, 24_000);
    assert!(GenerationRequest::default().audio_edit_regions().is_none());
    // The two carriers do not alias: a multi-region request is invisible to the single-region
    // accessor, and vice versa. This is what keeps the legacy path byte-identical.
    assert!(
        req.audio_edit().is_none(),
        "a multi-region carrier must not be visible through the single-region accessor"
    );
    let single = edit_req(AudioEditMode::Repaint, Some(span(2.0, 6.0)), None);
    assert!(single.audio_edit_regions().is_none());
}

/// Default-deny, with **no capability flag and no provider code**: a provider advertising only
/// the single-region `AudioEdit` kind refuses a multi-region request as typed `Unsupported`.
///
/// This is the acceptance criterion "every non-multi-region audio provider rejects multi-region
/// cleanly", proven against the shared allowlist rather than against any one provider.
#[test]
fn a_single_region_provider_refuses_multi_region_as_typed_unsupported() {
    let single_only = edit_caps();
    assert!(
        single_only.accepts(ConditioningKind::AudioEdit)
            && !single_only.accepts(ConditioningKind::AudioEditRegions),
        "precondition: the surface advertises single-region editing only"
    );
    let err = single_only
        .validate_request(
            "m",
            &multi_edit_req(
                AudioEditMode::Repaint,
                vec![span(2.0, 6.0), span(14.0, 18.0)],
            ),
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "multi-region on a single-region surface → Unsupported, got {err:?}"
    );
    assert!(err.to_string().contains("AudioEditRegions"));
    // And the same surface still accepts the single-region request it always did — so the
    // refusal is a real discrimination, not a blanket rejection.
    assert!(single_only
        .validate_request(
            "m",
            &edit_req(AudioEditMode::Repaint, Some(span(2.0, 6.0)), None)
        )
        .is_ok());
}

#[test]
fn multi_region_reuses_the_advertised_audio_edit_mode_surface() {
    let c = multi_edit_caps();
    assert!(c
        .validate_request(
            "m",
            &multi_edit_req(
                AudioEditMode::Repaint,
                vec![span(2.0, 6.0), span(14.0, 18.0)]
            )
        )
        .is_ok());
    // Order is not significant: the reversed list is equally valid.
    assert!(c
        .validate_request(
            "m",
            &multi_edit_req(
                AudioEditMode::Repaint,
                vec![span(14.0, 18.0), span(2.0, 6.0)]
            )
        )
        .is_ok());
    // Overlapping / touching / duplicate spans are ACCEPTED — the provider merges them.
    for regions in [
        vec![span(2.0, 6.0), span(4.0, 8.0)], // overlapping
        vec![span(2.0, 6.0), span(6.0, 8.0)], // touching
        vec![span(2.0, 6.0), span(2.0, 6.0)], // duplicate
        vec![span(2.0, 6.0), span(3.0, 4.0)], // contained
    ] {
        assert!(
            c.validate_request(
                "m",
                &multi_edit_req(AudioEditMode::Repaint, regions.clone())
            )
            .is_ok(),
            "{regions:?} must be accepted and normalized by the provider, not rejected"
        );
    }
    // An unadvertised mode is the same typed capability gap the single-region path gives.
    let err = c
        .validate_request(
            "m",
            &multi_edit_req(AudioEditMode::Cover, vec![span(2.0, 6.0)]),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(err.to_string().contains("unsupported audio edit mode"));
}

#[test]
fn multi_region_list_shape_is_gated() {
    let c = multi_edit_caps();
    // Empty list → malformed request, not a whole-clip edit.
    let err = c
        .validate_request("m", &multi_edit_req(AudioEditMode::Repaint, vec![]))
        .unwrap_err();
    assert!(matches!(err, Error::Msg(_)));
    assert!(err.to_string().contains("carries no regions"));
    // `end_secs: None` is refused OUTRIGHT, on any position — "to the end of the clip" has no
    // unambiguous meaning when order is not significant.
    for regions in [
        vec![TimeRegion {
            start_secs: 2.0,
            end_secs: None,
        }],
        vec![
            span(2.0, 6.0),
            TimeRegion {
                start_secs: 14.0,
                end_secs: None,
            },
        ],
        vec![
            TimeRegion {
                start_secs: 2.0,
                end_secs: None,
            },
            span(14.0, 18.0),
        ],
    ] {
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(AudioEditMode::Repaint, regions.clone()),
            )
            .unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)) && err.to_string().contains("must state an explicit"),
            "{regions:?} → open-ended refusal, got {err:?}"
        );
    }
}

/// **The list defeats the exhaustive-destructuring mechanism, so this is the gate that replaces
/// it.** Every malformed value is placed in region **two**, never region one.
///
/// `first_nonfinite_float` destructures `GenerationRequest` without `..` so a new *field* breaks
/// the build and must be classified. A `Vec` field satisfies that once and then hides an
/// unbounded number of floats behind it: a guard checking only `regions[0]` compiles, passes
/// every pre-existing test, and lets a NaN in region two reach the provider's mask.
#[test]
fn every_region_is_floored_not_just_the_first() {
    let c = multi_edit_caps();
    let good = span(2.0, 6.0);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Non-finite START in region TWO.
        let req = multi_edit_req(
            AudioEditMode::Repaint,
            vec![
                good,
                TimeRegion {
                    start_secs: bad,
                    end_secs: Some(18.0),
                },
            ],
        );
        assert_eq!(
            req.first_nonfinite_float().map(|(f, _)| f),
            Some("conditioning.audio_edit_regions.regions.start_secs"),
            "a non-finite start in region TWO must be found ({bad})"
        );
        assert!(c.validate_request("m", &req).is_err());

        // Non-finite END in region TWO.
        let req = multi_edit_req(
            AudioEditMode::Repaint,
            vec![
                good,
                TimeRegion {
                    start_secs: 14.0,
                    end_secs: Some(bad),
                },
            ],
        );
        assert_eq!(
            req.first_nonfinite_float().map(|(f, _)| f),
            Some("conditioning.audio_edit_regions.regions.end_secs"),
            "a non-finite end in region TWO must be found ({bad})"
        );
        assert!(c.validate_request("m", &req).is_err());

        // And in region THREE, so the gate does not merely check "the last one".
        let req = multi_edit_req(
            AudioEditMode::Repaint,
            vec![
                good,
                span(8.0, 10.0),
                TimeRegion {
                    start_secs: bad,
                    end_secs: Some(18.0),
                },
            ],
        );
        assert!(req.first_nonfinite_float().is_some());
        assert!(c.validate_request("m", &req).is_err());

        // Strength still joins the floor.
        let mut req = multi_edit_req(AudioEditMode::Repaint, vec![good]);
        let Conditioning::AudioEditRegions { strength, .. } = &mut req.conditioning[0] else {
            unreachable!()
        };
        *strength = Some(bad);
        assert_eq!(
            req.first_nonfinite_float().map(|(f, _)| f),
            Some("conditioning.audio_edit_regions.strength")
        );
    }
    // Control: a wholly well-formed multi-region request has no non-finite float at all, so the
    // assertions above discriminate rather than firing on everything.
    let ok = multi_edit_req(
        AudioEditMode::Repaint,
        vec![good, span(8.0, 10.0), span(14.0, 18.0)],
    );
    assert_eq!(ok.first_nonfinite_float(), None);
    assert!(c.validate_request("m", &ok).is_ok());
}

/// The non-finite refusal must name **which** region is malformed.
///
/// `first_nonfinite_float` keys are `&'static str`, so its multi-region arm can only report
/// `…regions.start_secs` with no index — and the two range guards that *do* carry an index
/// (`start < 0`, `end <= start`) never fire on a NaN, because both comparisons evaluate `false`
/// for NaN. Without the indexed pass in `validate_request` the caller of a ten-region repaint is
/// told a region is malformed but not which one. The bad value is placed in region **two** so a
/// guard that reached only `regions[0]` fails here, and in region **one** so a pass that skips
/// `regions[0]` fails too.
#[test]
fn a_non_finite_region_bound_is_reported_with_its_index() {
    let c = multi_edit_caps();
    let good = span(2.0, 6.0);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Non-finite START in region TWO (index 1).
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![
                        good,
                        TimeRegion {
                            start_secs: bad,
                            end_secs: Some(18.0),
                        },
                    ],
                ),
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, Error::Msg(_)) && msg.contains("region 1") && msg.contains("start"),
            "a non-finite start in region TWO must name `region 1`, got {msg:?} ({bad})"
        );

        // Non-finite END in region THREE (index 2), so the message is not merely reporting a
        // constant index — and so the pass is not checking only "the second one".
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![
                        good,
                        span(8.0, 10.0),
                        TimeRegion {
                            start_secs: 14.0,
                            end_secs: Some(bad),
                        },
                    ],
                ),
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, Error::Msg(_)) && msg.contains("region 2") && msg.contains("end"),
            "a non-finite end in region THREE must name `region 2`, got {msg:?} ({bad})"
        );

        // Non-finite START in region ONE (index 0), the symmetric twin of the two cases above:
        // a pass that *skips* `regions[0]` drops the index and falls through to the index-free
        // backstop, which the region-1 and region-2 cases alone do not catch.
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![
                        TimeRegion {
                            start_secs: bad,
                            end_secs: Some(6.0),
                        },
                        span(8.0, 10.0),
                    ],
                ),
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, Error::Msg(_)) && msg.contains("region 0") && msg.contains("start"),
            "a non-finite start in region ONE must name `region 0`, got {msg:?} ({bad})"
        );
    }
    // Discrimination: a well-formed multi-region request produces no error at all, so the
    // assertions above are not firing on everything. And the *index-free* backstop is still
    // wired — a bespoke provider that calls `ensure_finite_floats` directly (never entering
    // `validate_request`) still rejects the same request.
    let ok = multi_edit_req(AudioEditMode::Repaint, vec![good, span(8.0, 10.0)]);
    assert!(c.validate_request("m", &ok).is_ok());
    let nan_req = multi_edit_req(
        AudioEditMode::Repaint,
        vec![
            good,
            TimeRegion {
                start_secs: f32::NAN,
                end_secs: Some(18.0),
            },
        ],
    );
    assert!(
        nan_req.ensure_finite_floats().is_err(),
        "the index-free floor stays the backstop for callers that bypass validate_request"
    );
}

/// Malformed ranges are gated on **every** region, again with the bad value in region two.
#[test]
fn every_region_range_is_gated_not_just_the_first() {
    let c = multi_edit_caps();
    for bad in [span(-1.0, 4.0), span(8.0, 4.0), span(4.0, 4.0)] {
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(AudioEditMode::Repaint, vec![span(20.0, 24.0), bad]),
            )
            .unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)) && err.to_string().contains("region 1"),
            "{bad:?} in region two → Msg naming index 1, got {err:?}"
        );
    }
}

// ---- Video→audio (Foley) sync conditioning (sc-13436) ----------------------------------

/// A video→audio (Foley) capability surface: a `Modality::Audio` model that admits the
/// `VideoSync` kind and advertises a duration cap. No visual size bounds (audio floor).
fn foley_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::VideoSync],
        max_audio_duration_secs: Some(30.0),
        max_count: 1,
        ..Default::default()
    }
}

/// A video→audio request: a silent clip's frames plus a prompt, size left at the unused 0x0 and
/// the frame rate on `fps` (never on the variant).
fn foley_req(frame_count: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: "footsteps on gravel".into(),
        width: 0,
        height: 0,
        fps: Some(24),
        conditioning: vec![Conditioning::VideoSync {
            frames: (0..frame_count).map(|_| img(8, 8)).collect(),
        }],
        ..Default::default()
    }
}

#[test]
fn video_sync_maps_to_its_own_kind() {
    // The variant is distinct from the LTX clip kinds — its discriminant is VideoSync, not
    // VideoClip / ControlClip.
    let req = foley_req(3);
    assert_eq!(req.conditioning[0].kind(), ConditioningKind::VideoSync);
    // It is not collected by the LTX in-context clip / control accessors (a Foley clip is not an
    // extend_clip / replace_person input).
    assert!(req.video_clips().is_empty());
    assert!(req.control_clip().is_none());
    assert!(req.keyframes().is_empty());
}

#[test]
fn video_sync_accepted_when_advertised() {
    let c = foley_caps();
    assert!(c.validate_request_audio("foley", &foley_req(4)).is_ok());
}

#[test]
fn video_sync_unsupported_on_a_non_advertising_model() {
    // F-008: a model whose `conditioning` does not list `VideoSync` rejects it as the typed
    // Error::Unsupported (a capability gap), not a stringified Msg.
    let c = Capabilities {
        conditioning: vec![ConditioningKind::ReferenceAudio],
        max_count: 1,
        ..Default::default()
    };
    let err = c.validate_request_audio("tts", &foley_req(2)).unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "un-advertised VideoSync → typed Unsupported, got {err:?}"
    );
}

#[test]
fn video_sync_empty_frames_is_a_msg_range_error() {
    // An empty clip is a malformed request (nothing to condition on) → Error::Msg, even on a model
    // that admits the kind.
    let c = foley_caps();
    let err = c
        .validate_request_audio("foley", &foley_req(0))
        .unwrap_err();
    assert!(
        matches!(err, Error::Msg(_)),
        "empty VideoSync frames → Msg, got {err:?}"
    );
    assert!(err.to_string().contains("carries no frames"));
}

// ---- Reference video conditioning (sc-17149) -------------------------------------------

/// A video model that admits the `ReferenceVideo` kind.
fn ref_video_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::ReferenceVideo],
        max_count: 1,
        min_size: 8,
        max_size: 2048,
        ..Default::default()
    }
}

/// A reference-video request. `fps` is on the **variant** (the clip's own rate), and the
/// request's own `fps` is deliberately left unset — the two are different quantities.
fn ref_video_req(frame_count: usize, fps: f32, audio: Option<AudioTrack>) -> GenerationRequest {
    GenerationRequest {
        prompt: "the same subject, new scene".into(),
        width: 64,
        height: 64,
        conditioning: vec![Conditioning::ReferenceVideo {
            frames: (0..frame_count).map(|_| img(8, 8)).collect(),
            fps,
            audio,
        }],
        ..Default::default()
    }
}

#[test]
fn reference_video_maps_to_its_own_kind() {
    // Distinct from every other video-frame carrier. If this ever collapsed onto VideoClip, an
    // in-context-clip provider would start receiving references it would splice into the output.
    let req = ref_video_req(3, 30.0, None);
    assert_eq!(req.conditioning[0].kind(), ConditioningKind::ReferenceVideo);
    // And it is not collected by the epic-3040 in-context accessors: a reference has no position
    // in the generated timeline, so it is not an extend_clip / keyframe / replace_person input.
    assert!(req.video_clips().is_empty());
    assert!(req.control_clip().is_none());
    assert!(req.keyframes().is_empty());
}

#[test]
fn reference_video_accepted_when_advertised() {
    let c = ref_video_caps();
    c.validate_request("refvid", &ref_video_req(4, 30.0, None))
        .expect("an advertised reference clip is legal");
    // Carrying its own soundtrack changes nothing at the floor — the pairing rules that do
    // exist are per-model and layered by the provider's own validate.
    c.validate_request("refvid", &ref_video_req(4, 24.0, Some(track())))
        .expect("a reference clip with its own soundtrack is legal");
}

#[test]
fn reference_video_unsupported_on_a_non_advertising_model() {
    // F-008: default-deny. A provider that advertises the in-context clip kind but not the
    // reference kind must still reject a reference — this is the arm that makes the two kinds
    // worth separating in the first place.
    let c = Capabilities {
        conditioning: vec![ConditioningKind::VideoClip],
        max_count: 1,
        min_size: 8,
        max_size: 2048,
        ..Default::default()
    };
    let err = c
        .validate_request("ltx", &ref_video_req(2, 24.0, None))
        .unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "un-advertised ReferenceVideo → typed Unsupported, got {err:?}"
    );
    // The capability verdict outranks a payload-*shape* problem: an empty reference on a model
    // that does not admit the kind is still `Unsupported`, because the caller's first problem is
    // that this model cannot do this at all.
    let err = c
        .validate_request("ltx", &ref_video_req(0, 24.0, None))
        .unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "capability gap outranks payload shape, got {err:?}"
    );
    // The finiteness floor is the one exception, and it is deliberate and pre-existing: it runs
    // ahead of the conditioning allowlist for *every* float in the request, so a NaN rate is
    // reported as the non-finite float it is even on a model that would have refused the kind.
    // Pinned here so a future reordering of the floor is a visible decision rather than a
    // silent change of which error a caller sees.
    let err = c
        .validate_request("ltx", &ref_video_req(2, f32::NAN, None))
        .unwrap_err();
    assert!(
        matches!(&err, Error::Msg(m) if m.contains("conditioning.reference_video.fps")),
        "the finiteness floor precedes the allowlist, got {err:?}"
    );
}

#[test]
fn reference_video_empty_frames_is_a_msg_range_error() {
    let c = ref_video_caps();
    let err = c
        .validate_request("refvid", &ref_video_req(0, 24.0, None))
        .unwrap_err();
    assert!(
        matches!(err, Error::Msg(_)),
        "empty ReferenceVideo frames → Msg, got {err:?}"
    );
    assert!(err.to_string().contains("carries no frames"));
}

#[test]
fn reference_video_rejects_a_rate_that_has_no_reading() {
    // Zero and negative are refused as well as non-finite, and that is deliberate: a rate is the
    // divisor of a resample stride, so unlike every other conditioning float 0.0 is not a
    // meaningful inert value here — it makes the set of frames the model reads arbitrary.
    let c = ref_video_caps();
    for bad in [0.0f32, -24.0, f32::NAN, f32::INFINITY] {
        let err = c
            .validate_request("refvid", &ref_video_req(2, bad, None))
            .unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "fps {bad} → Msg, got {err:?}");
    }
    // A legal rate on the same shape passes, so the loop above is not passing because the
    // request is malformed for some unrelated reason.
    c.validate_request("refvid", &ref_video_req(2, 30.0, None))
        .expect("30 fps is a legal rate");
}

#[test]
fn reference_video_rate_joins_the_finiteness_floor() {
    // The floor is a separate mechanism from validate()'s range check above, and it names the
    // offending field. Both must cover the rate: the floor is what providers reading the
    // conditioning directly rely on.
    let (field, value) = ref_video_req(2, f32::NAN, None)
        .first_nonfinite_float()
        .expect("NaN fps must be caught");
    assert_eq!(field, "conditioning.reference_video.fps");
    assert!(value.is_nan());
    assert!(ref_video_req(2, 30.0, None)
        .first_nonfinite_float()
        .is_none());
}

/// sc-19571 — the four conditioning floats the old `_ => {}` arm swallowed.
///
/// Each of these is a `1 − strength` denoise mask or a blend weight, i.e. the same math the
/// floor already protected on `Reference` — they were missed only because the conditioning
/// `match` ended in a wildcard, so a new float-bearing variant compiled clean. The wildcard is
/// gone; this test is what proves the four arms that replaced it actually fire.
///
/// Mutation guard: delete any ONE of the four arms and exactly one sub-assertion below goes
/// red — they are checked individually, not as a batch.
#[test]
fn keyframe_video_clip_control_clip_and_redux_floats_join_the_finiteness_floor() {
    let with = |c: Conditioning| GenerationRequest {
        prompt: "x".into(),
        conditioning: vec![c],
        ..Default::default()
    };

    let (field, value) = with(Conditioning::Keyframe {
        image: img(8, 8),
        frame_idx: 0,
        strength: f32::NAN,
    })
    .first_nonfinite_float()
    .expect("NaN keyframe strength must be caught");
    assert_eq!(field, "conditioning.keyframe.strength");
    assert!(value.is_nan());

    let (field, value) = with(Conditioning::VideoClip {
        frames: vec![img(8, 8)],
        frame_idx: 0,
        strength: f32::INFINITY,
    })
    .first_nonfinite_float()
    .expect("infinite clip strength must be caught");
    assert_eq!(field, "conditioning.video_clip.strength");
    assert!(value.is_infinite());

    let (field, value) = with(Conditioning::ControlClip {
        frames: vec![img(8, 8)],
        mask: vec![img(8, 8)],
        masking_strength: f32::NAN,
        start_frame: 0,
        mode: ReplacementMode::FaceOnly,
    })
    .first_nonfinite_float()
    .expect("NaN masking_strength must be caught");
    assert_eq!(field, "conditioning.control_clip.masking_strength");
    assert!(value.is_nan());

    // The bad value sits in ref **two** — a guard that only reached `refs[0]` would compile and
    // pass every other assertion here (the `AudioEditRegions` lesson, applied).
    let (field, value) = with(Conditioning::ReduxRefs {
        refs: vec![(img(8, 8), 0.5), (img(8, 8), f32::NAN)],
    })
    .first_nonfinite_float()
    .expect("NaN in the SECOND redux ref must be caught");
    assert_eq!(field, "conditioning.redux_refs.strength");
    assert!(value.is_nan());

    // Finite values on all four pass cleanly — the floor rejects non-finiteness, not the knob.
    let clean = GenerationRequest {
        prompt: "x".into(),
        conditioning: vec![
            Conditioning::Keyframe {
                image: img(8, 8),
                frame_idx: -1,
                strength: 0.25,
            },
            Conditioning::VideoClip {
                frames: vec![img(8, 8)],
                frame_idx: 0,
                strength: 0.75,
            },
            Conditioning::ControlClip {
                frames: vec![img(8, 8)],
                mask: vec![img(8, 8)],
                masking_strength: 0.5,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            },
            Conditioning::ReduxRefs {
                refs: vec![(img(8, 8), 0.5), (img(8, 8), 0.25)],
            },
        ],
        ..Default::default()
    };
    assert!(clean.first_nonfinite_float().is_none());
    assert!(clean.ensure_finite_floats().is_ok());
}

/// sc-13884: the default request carries no phases (single-phase, byte-for-byte the pre-13884
/// behavior), and a `phases: Some([...])` request round-trips its typed phase list through a clone.
#[test]
fn phases_default_none_and_round_trip() {
    // Default = single-phase, unaffected.
    assert_eq!(GenerationRequest::default().phases, None);

    // A two-phase Raw→Raw+turbo-LoRA split: phase 1 = 20 steps, CFG on, base-only; phase 2 = 8
    // steps, CFG off, turbo LoRA (load-time adapter 0) at weight 0.8.
    let phases = vec![
        GenerationPhase {
            steps: 20,
            guidance: Some(3.5),
            adapters: vec![],
        },
        GenerationPhase {
            steps: 8,
            guidance: Some(0.0),
            adapters: vec![PhaseAdapter {
                adapter: 0,
                weight: Some(0.8),
            }],
        },
    ];
    let req = GenerationRequest {
        prompt: "a phased render".into(),
        phases: Some(phases.clone()),
        ..Default::default()
    };
    // The typed phases survive a clone unchanged (no serde in this contract — Clone is the
    // transport the worker uses to hand a request across the thread boundary).
    assert_eq!(req.clone().phases, Some(phases));
    // The flat steps knob is left at its default None — the total budget is the phases' sum.
    assert_eq!(req.steps, None);
}

/// sc-13884: a NaN/Inf on a phase guidance OR a phase-adapter weight is caught by the shared
/// finiteness floor, exactly like the flat float knobs (a NaN in the phase forward would silently
/// poison the guidance / adapter-scale math otherwise).
#[test]
fn phase_floats_join_the_finiteness_floor() {
    let bad_guidance = GenerationRequest {
        phases: Some(vec![GenerationPhase {
            steps: 4,
            guidance: Some(f32::NAN),
            adapters: vec![],
        }]),
        ..Default::default()
    };
    assert_eq!(
        bad_guidance.first_nonfinite_float().map(|(f, _)| f),
        Some("phases.guidance")
    );
    assert!(bad_guidance.ensure_finite_floats().is_err());

    let bad_weight = GenerationRequest {
        phases: Some(vec![GenerationPhase {
            steps: 4,
            guidance: Some(0.0),
            adapters: vec![PhaseAdapter {
                adapter: 0,
                weight: Some(f32::INFINITY),
            }],
        }]),
        ..Default::default()
    };
    assert_eq!(
        bad_weight.first_nonfinite_float().map(|(f, _)| f),
        Some("phases.adapter.weight")
    );

    // A finite two-phase request passes the floor untouched.
    let ok = GenerationRequest {
        phases: Some(vec![
            GenerationPhase {
                steps: 4,
                guidance: Some(3.5),
                adapters: vec![],
            },
            GenerationPhase {
                steps: 4,
                guidance: None,
                adapters: vec![PhaseAdapter {
                    adapter: 1,
                    weight: None,
                }],
            },
        ]),
        ..Default::default()
    };
    assert_eq!(ok.first_nonfinite_float(), None);
}

/// sc-23402 — the reference-image short edge is admitted over 1024..=2048 INCLUSIVE, and an
/// out-of-range value is refused rather than clamped.
///
/// The two boundary pairs are the whole point: an exclusive upper bound, an off-by-one floor, or
/// a `clamp` in place of the refusal each flips exactly one of these four.
#[test]
fn reference_image_short_edge_admits_1024_through_2048_inclusive_and_refuses_outside() {
    let with = |edge: Option<u32>| GenerationRequest {
        reference_image_short_edge: edge,
        ..Default::default()
    };

    validate_reference_image_short_edge("m", &with(Some(1024))).expect("1024 is admitted");
    validate_reference_image_short_edge("m", &with(Some(2048))).expect("2048 is admitted");
    validate_reference_image_short_edge("m", &with(None)).expect("an absent knob is vacuous");

    for bad in [0, 1, 1023, 2049, u32::MAX] {
        let message = validate_reference_image_short_edge("m", &with(Some(bad)))
            .expect_err("out of range is refused, never clamped")
            .to_string();
        assert!(
            message.contains("reference_image_short_edge")
                && message.contains(&bad.to_string())
                && message.contains("1024..=2048"),
            "the refusal must name the field, the value and the range, got: {message}"
        );
    }
}

/// The effective value is the request's, or upstream's 2048 when absent — the one resolver the
/// engine applies and the consumer records, so a recipe cannot drift from the render.
#[test]
fn effective_reference_image_short_edge_defaults_to_upstreams_2048() {
    let default_request = GenerationRequest::default();
    assert_eq!(default_request.reference_image_short_edge, None);
    assert_eq!(
        effective_reference_image_short_edge(&default_request),
        REFERENCE_IMAGE_SHORT_EDGE_DEFAULT
    );
    assert_eq!(REFERENCE_IMAGE_SHORT_EDGE_DEFAULT, 2048);
    assert_eq!(REFERENCE_IMAGE_SHORT_EDGE_MAX, 2048);
    assert_eq!(REFERENCE_IMAGE_SHORT_EDGE_MIN, 1024);

    for asked in [
        REFERENCE_IMAGE_SHORT_EDGE_MIN,
        1536,
        REFERENCE_IMAGE_SHORT_EDGE_MAX,
    ] {
        let req = GenerationRequest {
            reference_image_short_edge: Some(asked),
            ..Default::default()
        };
        assert_eq!(
            effective_reference_image_short_edge(&req),
            asked,
            "the resolver must return the REQUESTED value, not a constant"
        );
    }
}
