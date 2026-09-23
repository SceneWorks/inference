//! Weights-free, default-run descriptor-level gen-core conformance (sc-9098, F-009): every
//! registration this provider explicitly exports (including any reused sibling providers) satisfies
//! the descriptor/capability invariants checkable without loading weights — id/family/backend
//! shape, coherent size/count bounds, duplicate-free curated names and conditioning kinds,
//! modality-consistent conditioning, and per-kind registry id uniqueness. Behavioral conformance
//! (progress/cancel/seed) stays weights-gated in the crate's `#[ignore]`d suites.

#[test]
fn registered_descriptors_conform() {
    let registry = mlx_gen_qwen_image::provider_registry().expect("provider registry should build");
    assert!(
        registry.generators().len() > 0,
        "provider registry must contain a generator"
    );
    let errs = registry.descriptor_conformance_errors();
    assert!(
        errs.is_empty(),
        "descriptor conformance FAILED ({} violations):\n  - {}",
        errs.len(),
        errs.join("\n  - ")
    );
}

/// **RGBA default-deny (sc-24111).** This family's decoder is three-channel, so every descriptor
/// it registers must leave [`Capabilities::supports_alpha_output`] at the `Default` `false` and
/// the shared request floor must refuse an `OutputChannels::Rgba` request against it as a typed
/// `Unsupported`.
///
/// The negative half of the capability lives here, on a provider that has nothing to do with
/// transparency, precisely because that is where a regression would otherwise go unnoticed: the
/// alpha-capable provider's own suite cannot prove that *other* providers still refuse.
#[test]
fn rgba_output_is_refused_by_every_descriptor_in_this_family() {
    use mlx_gen::gen_core::{Error, GenerationRequest, Modality, OutputChannels};

    let registry = mlx_gen_qwen_image::provider_registry().expect("provider registry should build");
    let mut checked = 0;
    for registration in registry.generators() {
        let descriptor = (registration.descriptor)();
        let descriptor = &descriptor;
        if descriptor.modality == Modality::Audio {
            continue;
        }
        let caps = &descriptor.capabilities;
        assert!(
            !caps.supports_alpha_output,
            "{}: this family decodes three channels; it must not advertise an alpha output",
            descriptor.id
        );

        let mut req = GenerationRequest {
            prompt: "a red fox".into(),
            width: caps.min_size.max(64),
            height: caps.min_size.max(64),
            ..Default::default()
        };
        // The RGB default is accepted (so the refusal below is about alpha, not about the
        // request being malformed).
        caps.validate_request(descriptor.id, &req)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: the default RGB request was rejected: {e}",
                    descriptor.id
                )
            });

        req.output_channels = OutputChannels::Rgba;
        match caps.validate_request(descriptor.id, &req) {
            Err(Error::Unsupported(message)) => {
                assert!(
                    message.contains("RGBA"),
                    "{}: the refusal must name the capability, got {message:?}",
                    descriptor.id
                );
            }
            Err(other) => panic!(
                "{}: an RGBA request must be refused as Unsupported, got {other:?}",
                descriptor.id
            ),
            Ok(()) => panic!(
                "{}: an RGBA request was ACCEPTED by a three-channel provider — it would return \
                 RGB with nothing to say the alpha was dropped",
                descriptor.id
            ),
        }
        checked += 1;
    }
    assert!(checked > 0, "no image/video descriptors were checked");
}
