use std::convert::Infallible;

use super::*;

const GEOMETRY: MemoryGeometry = MemoryGeometry {
    width: 256,
    height: 256,
    batch: 1,
    frames: 1,
    reference_count: 0,
};

fn phase_arguments(adapters: Vec<PhaseAdapterScale>, guidance: f32) -> PhaseArguments {
    let branches = if guidance != 0.0 {
        ConditioningBranches::PositiveAndNegative
    } else {
        ConditioningBranches::Positive
    };
    PhaseArguments::new(adapters, guidance, branches)
}

#[test]
fn phase_preparation_matches_forward_and_only_reuses_complete_dependencies() {
    let adapter_a = PhaseAdapterScale::new(5, 0.25);
    let adapter_b = PhaseAdapterScale::new(2, 0.5);
    let phases = [
        (phase_arguments(vec![], 0.0), GEOMETRY),
        (phase_arguments(vec![adapter_a], 2.0), GEOMETRY),
        (phase_arguments(vec![], 0.0), GEOMETRY),
        (phase_arguments(vec![], 2.0), GEOMETRY),
        (
            PhaseArguments::new(vec![adapter_a], 2.0, ConditioningBranches::Positive),
            GEOMETRY,
        ),
        (phase_arguments(vec![adapter_a], -3.0), GEOMETRY),
        (
            phase_arguments(vec![adapter_a], 2.0),
            MemoryGeometry {
                width: 512,
                ..GEOMETRY
            },
        ),
        (
            phase_arguments(vec![PhaseAdapterScale::new(5, 0.5)], 2.0),
            GEOMETRY,
        ),
        (phase_arguments(vec![adapter_a, adapter_b], 2.0), GEOMETRY),
        (phase_arguments(vec![adapter_b, adapter_a], 2.0), GEOMETRY),
        (phase_arguments(vec![adapter_a, adapter_a], 2.0), GEOMETRY),
    ];
    let mut prepared_count = 0;
    let prepared = prepare_phases(phases.clone(), |args, geometry| {
        prepared_count += 1;
        Ok::<_, Infallible>((
            prepared_count,
            args.adapters().to_vec(),
            args.uses_classifier_free_guidance(),
            geometry,
        ))
    })
    .unwrap();

    let cache_instances: Vec<_> = prepared
        .iter()
        .zip(phases)
        .map(|(phase, (original_args, original_geometry))| {
            let (forward_args, (instance, prepared_adapters, prepared_cfg, geometry)) =
                phase.parts();
            assert_eq!(forward_args.adapters(), prepared_adapters);
            assert_eq!(forward_args.uses_classifier_free_guidance(), *prepared_cfg);
            assert_eq!(
                forward_args.guidance().to_bits(),
                original_args.guidance().to_bits()
            );
            assert_eq!(*geometry, original_geometry);
            *instance
        })
        .collect();
    assert_eq!(cache_instances, [1, 2, 1, 3, 4, 2, 5, 6, 7, 8, 9]);
    assert_eq!(prepared_count, 9);
}

#[test]
fn interleaved_requests_own_distinct_model_and_conditioning_preparation() {
    // Numerically identical private slots can refer to different files in different requests.
    let arguments = phase_arguments(vec![PhaseAdapterScale::new(0, 0.5)], 1.0);
    let phase = (arguments, GEOMETRY);
    let first = prepare_phases([phase.clone(), phase.clone()], |_, _| {
        let second = prepare_phases([phase.clone()], |_, _| {
            Ok::<_, Infallible>(("second model", "second adapter", "second prompt"))
        })
        .unwrap();
        assert_eq!(
            second[0].parts().1,
            &("second model", "second adapter", "second prompt")
        );
        Ok::<_, Infallible>(("first model", "first adapter", "first prompt"))
    })
    .unwrap();
    assert_eq!(
        first[0].parts().1,
        &("first model", "first adapter", "first prompt")
    );
    assert!(std::ptr::eq(first[0].parts().1, first[1].parts().1));
}

#[test]
fn each_preparation_lives_only_as_long_as_the_phase_plans_using_it() {
    let base = (phase_arguments(vec![], 0.0), GEOMETRY);
    let adapted = (
        phase_arguments(vec![PhaseAdapterScale::new(4, 1.0)], 1.0),
        GEOMETRY,
    );
    let mut plans = prepare_phases(
        [base.clone(), adapted, base],
        |_, _| Ok::<_, Infallible>(()),
    )
    .unwrap()
    .into_iter();
    let first = plans.next().unwrap();
    let middle = plans.next().unwrap();
    let last = plans.next().unwrap();
    let base_lifetime = Rc::downgrade(&first.prepared);
    let adapted_lifetime = Rc::downgrade(&middle.prepared);
    assert!(Rc::ptr_eq(&first.prepared, &last.prepared));

    drop(first);
    assert!(base_lifetime.upgrade().is_some());
    drop(middle);
    assert!(adapted_lifetime.upgrade().is_none());
    drop(last);
    assert!(base_lifetime.upgrade().is_none());
}

#[test]
fn effective_scale_keys_use_exact_bits_and_projection_scale_math_preserves_precision() {
    assert_ne!(
        PhaseAdapterScale::new(4, 0.0),
        PhaseAdapterScale::new(4, -0.0)
    );
    assert_ne!(
        PhaseAdapterScale::new(4, 0.125),
        PhaseAdapterScale::new(5, 0.125)
    );
    let scale = 1.0_f64 + f64::EPSILON;
    let context = ScaleContext::new().with_pass(2);
    assert_eq!(context.scale_for_f64(0, scale).to_bits(), scale.to_bits());
    let context = ScaleContext {
        adapter_scales: Some(&[0.1]),
        uniform_scale: Some(0.3),
        pass: 2,
    };
    assert_eq!(
        context.scale_for(0, 2.0).to_bits(),
        (0.1_f32 * 0.3).to_bits()
    );
    assert_eq!(
        context.scale_for_f64(0, 2.0).to_bits(),
        (f64::from(0.1_f32) * f64::from(0.3_f32)).to_bits()
    );
}
