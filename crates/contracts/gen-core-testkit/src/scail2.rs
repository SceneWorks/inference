//! Weights-free public SCAIL-2 video registration checks shared by MLX and Candle.
use gen_core::{
    Conditioning, GenerationRequest, Generator, Image, LoadSpec, ProviderRegistry, ReplacementMode,
    WeightsSource,
};

/// Compare metadata for the 90-frame driving clip with the exact native window and decoder,
/// including all 1, 2 or 6 pairs that the registered adapter passes to the denoiser.
pub fn assert_registered_characters_and_windows(
    registry: ProviderRegistry,
    generator: &dyn Generator,
    resolve: impl for<'a> Fn(
        &'a GenerationRequest,
    ) -> gen_core::generator::Scail2CharacterReferences<'a>,
) {
    let mlx = generator.descriptor().backend == "mlx";
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-scail2-snapshot".into()));
    let image = |tag, w, h| Image {
        width: w,
        height: h,
        pixels: vec![tag; (w * h * 3) as usize],
    };
    for characters in [1, 2, 6] {
        let mut req = GenerationRequest {
            width: 0,
            height: 0,
            video_mode: Some("animation".into()),
            ..Default::default()
        };
        for i in 0..characters {
            req.conditioning.push(Conditioning::Reference {
                image: image(i, 2, 2),
                strength: None,
            });
            req.conditioning.push(Conditioning::Mask {
                image: image(100 + i, 2, 2),
            });
        }
        let driving = image(3, 64, 32);
        req.conditioning.push(Conditioning::ControlClip {
            frames: vec![driving.clone(); 90],
            mask: vec![driving; 90],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        });
        let plan = registry
            .compile_video_plan(generator.descriptor().id, &spec, &req)
            .unwrap()
            .unwrap();
        let layout = plan.layout();
        assert_eq!(plan.geometry().reference_count, u32::from(characters));
        assert_eq!(plan.geometry().frames, 81);
        assert_eq!(
            (
                layout.output.width,
                layout.output.height,
                layout.output.frames
            ),
            (64, 32, if mlx { 84 } else { 89 })
        );
        let source_bytes = 2 * 90 * 64 * 32 * 3 + 2 * u64::from(characters) * 2 * 2 * 3;
        let host = if layout.domain == gen_core::MemoryBudgetDomain::UnifiedMemory {
            layout.domain
        } else {
            gen_core::MemoryBudgetDomain::HostRam
        };
        let source_on_device = if host == layout.domain {
            source_bytes
        } else {
            0
        };
        let rgb = 64 * 32 * 3;
        let stacks = 90 * 2 * 4 * rgb;
        let retained = &layout.retained_allocations;
        assert_eq!(retained.len(), 1);
        assert_eq!(
            retained[0],
            gen_core::PricedAllocation::new("scail2_driving_rgb_and_masks", host, stacks)
        );
        let output = if mlx { 84 } else { 89 };
        let (round_peak, export, host_staging) = if mlx {
            (84 * 4 * rgb, (84 * 4 + 84) * rgb, 0)
        } else {
            // 81 first frames + all 13 in second backing + distinct 5 history + 89 stitched.
            let retained = (81 + 13 + 5 + 89) * 4 * rgb;
            (
                retained + 89 * (6 * 4 + 1) * rgb,
                retained + 89 * 4 * rgb,
                (89 + 1) * rgb,
            )
        };
        assert_eq!(
            layout.phases.last().unwrap().decoded_tensor_bytes,
            round_peak
        );
        assert_eq!(layout.output_conversion.device_bytes, export);
        assert_eq!(layout.output_conversion.host_bytes, host_staging);
        let host_buffers = output * rgb + host_staging;
        let expected = source_on_device
            + if host == layout.domain { stacks } else { 0 }
            + round_peak.max(
                export
                    + if host == layout.domain {
                        host_buffers
                    } else {
                        0
                    },
            );
        let minimum = plan.minimum_decode_live_bytes(layout.domain).unwrap();
        assert_eq!(minimum, expected);
        let mut budget = gen_core::MemoryBudget {
            total_bytes: expected - 1,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        };
        assert!(!budget.fits(minimum));
        budget.total_bytes += 1;
        assert!(budget.fits(minimum));
        assert_eq!(plan.minimum_live_bytes(layout.domain).unwrap(), minimum);
        if host != layout.domain {
            assert_eq!(
                plan.minimum_live_bytes(host).unwrap(),
                source_bytes + stacks + host_buffers
            );
        }
        assert_eq!(layout.references.len(), 2 * characters as usize + 2);
        let roles = &layout.references[layout.references.len() - 2..];
        assert_eq!(
            (
                roles[0].role,
                roles[0].width,
                roles[0].height,
                roles[0].frames
            ),
            ("driving-pose", 32, 16, if mlx { 81 } else { 89 })
        );
        assert_eq!(roles[1].role, "driving-mask");
        let phase = &layout.phases[0];
        assert_eq!(
            (
                phase.source.clone(),
                phase.latent_frames,
                phase.latent_bytes
            ),
            (0..81, 21, 16 * 21 * 8 * 4 * 4)
        );
        assert_eq!(
            (phase.decoded_frames, phase.decode_chunk_frames),
            if mlx { (84, 84) } else { (81, 4) }
        );
        assert_eq!(
            phase.retained_bytes,
            32 * 4 * (44 * u64::from(characters) + 39 * 21)
        );
        if mlx {
            assert_eq!(layout.phases.len(), 1);
        } else {
            let last = &layout.phases[1];
            assert_eq!(layout.phases.len(), 2);
            assert_eq!(
                (
                    last.source.clone(),
                    last.history_frames,
                    last.latent_frames,
                    last.decoded_frames
                ),
                (76..89, 5, 4, 13)
            );
            assert_eq!(
                last.retained_bytes,
                (81 + 5) * 64 * 32 * 3 * 4
                    + 32 * 4 * (44 * u64::from(characters) + 39 * 4 + 16 * 2 + 4 * 4)
            );
        }
        let pairs = resolve(&req);
        assert_eq!(pairs.additional.len(), characters as usize - 1);
        for (i, pair) in std::iter::once(pairs.primary)
            .chain(pairs.additional)
            .enumerate()
        {
            assert_eq!(pair.image.pixels[0], i as u8);
            assert_eq!(pair.mask.pixels[0], 100 + i as u8);
        }
        assert!(generator
            .video_memory_evidence_tier(&plan, &req)
            .unwrap()
            .is_none());
        assert_eq!(req.memory_reference_count(), u32::from(characters));
        let mut crossed = req.clone();
        if let Conditioning::Mask { image } = &mut crossed.conditioning[2 * characters as usize - 1]
        {
            image.pixels[0] ^= 1;
        }
        assert!(plan.check_inputs(&crossed).is_err());
        if characters == 1 {
            for (n, emitted) in [
                (45, if mlx { 48 } else { 45 }),
                (157, if mlx { 163 } else { 157 }),
            ] {
                let mut clip = req.clone();
                let Some(Conditioning::ControlClip { frames, mask, .. }) =
                    clip.conditioning.last_mut()
                else {
                    unreachable!()
                };
                frames.resize(n, frames[0].clone());
                mask.resize(n, mask[0].clone());
                let layout = registry
                    .compile_video_plan(generator.descriptor().id, &spec, &clip)
                    .unwrap()
                    .unwrap();
                assert_eq!(layout.layout().output.frames, emitted);
                if n == 157 {
                    let segments = &layout.layout().phases;
                    assert_eq!(segments.len(), 2);
                    assert_eq!(segments[1].source, 76..157);
                    assert_eq!(segments[1].history_frames, 5);
                    // Candle's second narrow retains its complete 81-frame decoder backing,
                    // whereas MLX stores only the newly stitched part as a separate array.
                    let retained = if mlx { 163 } else { 162 };
                    let device = if mlx {
                        (retained + 5 + 163) * 4 + 163
                    } else {
                        (retained + 5 + 157 + 157) * 4
                    };
                    assert_eq!(
                        layout.layout().output_conversion.device_bytes,
                        device * 64 * 32 * 3
                    );
                }
            }
        }
        let mut malformed = req.clone();
        malformed.conditioning.swap(0, 1);
        assert!(
            matches!(registry.compile_video_plan(generator.descriptor().id,&spec,&malformed),
            Err(gen_core::Error::Unsupported(reason)) if reason.contains("without a preceding Reference"))
        );
        req.video_mode = Some("replacement".into());
        assert_eq!(
            registry
                .compile_video_plan(generator.descriptor().id, &spec, &req)
                .is_ok(),
            characters == 1
        );
    }
}

/// The worker selected one Resident row. A valid projection may translate its mode, overlay
/// and revision, but may never replace its budget, selection, authority or cache state.
pub fn assert_projected_context(
    registry: ProviderRegistry,
    generator: &dyn Generator,
    spec: &LoadSpec,
    req: &GenerationRequest,
    native: gen_core::MemoryRunContext,
) {
    use gen_core::generator::Scail2AnimationConditioningRef;
    let id = generator.descriptor().id;
    let plan = registry.compile_video_plan(id, spec, req).unwrap().unwrap();
    let mode = req.video_mode.as_deref().unwrap();
    let catalog_mode = if mode == "animation" {
        "animate_character"
    } else {
        "replace_person"
    };
    let receipt = Scail2AnimationConditioningRef::catalog_receipt(
        generator.descriptor().backend,
        catalog_mode,
        req.width,
        req.height,
        &req.conditioning,
    )
    .unwrap();
    let adapter = if spec.adapters.is_empty() {
        ""
    } else {
        "adapter+"
    };
    let mut selected = native.clone();
    selected.overlay = Some(format!("{adapter}provider_video_mode:{mode}+{receipt}"));
    selected.mode = gen_core::MemoryMode::Other(catalog_mode.into());
    selected.evidence_revision = "selected-worker-evidence".into();
    assert_eq!(
        generator.video_memory_evidence_tier(&plan, req).unwrap(),
        Some(native.selection.tier)
    );
    assert_eq!(plan.geometry(), native.geometry);
    let mut crossed = selected.clone();
    crossed.overlay.as_mut().unwrap().push('0');
    assert!(generator
        .project_video_memory_context(&plan, req, &mut crossed)
        .is_err());
    let mut crossed = selected.clone();
    crossed.geometry.frames -= 4;
    assert!(generator
        .project_video_memory_context(&plan, req, &mut crossed)
        .is_err());
    let mut crossed = selected.clone();
    crossed.selection.strategy = gen_core::MemoryStrategy::BoundedDecode;
    assert!(generator
        .project_video_memory_context(&plan, req, &mut crossed)
        .is_err());
    let mut crossed = native.clone();
    crossed.evidence_revision.push('0');
    assert!(generator
        .project_video_memory_context(&plan, req, &mut crossed)
        .is_err());
    assert!(generator
        .project_video_memory_context(&plan, req, &mut selected)
        .unwrap());
    assert_eq!(selected, native);
    let mut other = spec.clone();
    other.resolved_route = Some("other".into());
    let wrong = registry
        .compile_video_plan(id, &other, req)
        .unwrap()
        .unwrap();
    assert!(generator
        .project_video_memory_context(&wrong, req, &mut selected)
        .is_err());
    assert_eq!(
        generator.memory_strategy_safety_check(&native),
        gen_core::MemorySafetyDecision::Accept
    );
    let mut scope = generator
        .begin_memory_strategy_request(&selected)
        .unwrap()
        .unwrap();
    let mut configured = req.clone();
    scope.configure_request(&mut configured).unwrap();
    let bound = plan
        .bind_admission(spec, Some(&selected), &configured)
        .unwrap();
    generator.validate_video_plan(&bound, &configured).unwrap();
    generator.validate(&configured).unwrap();
    let phase = &bound.layout().phases[0];
    assert_eq!(phase.identity.geometry, native.geometry);
    scope
        .enter_video_phase(&phase.identity, phase.latent_frames)
        .unwrap();
    scope.leave_phase(gen_core::MemoryPhase::Denoise).unwrap();
    scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
}
