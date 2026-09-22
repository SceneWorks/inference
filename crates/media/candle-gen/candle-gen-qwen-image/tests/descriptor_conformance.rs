//! Weights-free, default-run descriptor-level gen-core conformance for the candle Qwen-Image
//! provider (sc-24111).
//!
//! Deliberately NOT in `conformance.rs`, which is `#![cfg(feature = "cuda")]` and so never
//! compiles off the CUDA lane: these are capability-surface assertions that cost nothing, open no
//! weights and must run on every lane, which is the only way they catch a regression early.

/// **RGBA default-deny (sc-24111).** This family's decoder is three-channel, so every descriptor
/// it registers must leave `Capabilities::supports_alpha_output` at the `Default` `false` and the
/// shared request floor must refuse an `OutputChannels::Rgba` request against it as a typed
/// `Unsupported`.
///
/// The candle twin of the MLX 2512 gate. The negative half of the capability lives on providers
/// that have nothing to do with transparency, because that is where a regression would otherwise
/// go unnoticed: the alpha-capable provider's own suite cannot prove that *other* providers still
/// refuse, and it cannot prove it on the other backend at all.
#[test]
fn rgba_output_is_refused_by_every_descriptor_in_this_family() {
    use candle_gen::gen_core::{Error, GenerationRequest, Modality, OutputChannels};

    let registry =
        candle_gen_qwen_image::provider_registry().expect("provider registry should build");
    let mut checked = 0;
    for registration in registry.generators() {
        let descriptor = (registration.descriptor)();
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
        // The RGB default is accepted, so the refusal below is about alpha rather than about the
        // request being malformed.
        caps.validate_request(descriptor.id, &req)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: the default RGB request was rejected: {e}",
                    descriptor.id
                )
            });

        req.output_channels = OutputChannels::Rgba;
        match caps.validate_request(descriptor.id, &req) {
            Err(Error::Unsupported(message)) => assert!(
                message.contains("RGBA"),
                "{}: the refusal must name the capability, got {message:?}",
                descriptor.id
            ),
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
