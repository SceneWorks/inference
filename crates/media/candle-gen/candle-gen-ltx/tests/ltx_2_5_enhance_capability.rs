//! sc-18764 / R2 — the candle LTX provider must not advertise prompt enhancement.
//!
//! Prompt enhancement is an **mlx-only** path in this family: `mlx-gen-ltx` ships
//! `src/enhance.rs` (Gemma 3 / LTX-2.3), this crate ships no enhancer for either checkpoint
//! generation. R2 is that a backend advertises exactly what it can do, so the candle descriptor
//! must carry `supports_prompt_enhancement: false` and the shared gen-core floor must turn that
//! into a typed refusal rather than letting `enhance_prompt` be silently dropped on the floor.
//!
//! Both halves are asserted, because the flag alone is not the fault site: a consumer never reads
//! `Capabilities` directly, it sends a request. The refusal assertion is placed on
//! `Capabilities::validate_request` — the exact call `<Pipeline as Generator>::validate` makes —
//! so it fails where a dishonest descriptor would actually do damage.
//!
//! Weights-free and device-free: no CUDA feature gate, no snapshot, runs by default in CI.
//!
//! Mutation that must make this file fail: flip `supports_prompt_enhancement` to `true` in
//! `descriptor()` (`src/lib.rs`). Both tests below then fail — the first on the flag, the second
//! because the floor stops refusing.

use candle_gen::gen_core::{Error, GenerationRequest};

/// A request that differs from an accepted one **only** in `enhance_prompt`, so a failure here can
/// only be about enhancement support.
fn enhance_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a person crosses the room".into(),
        width: 704,
        height: 512,
        frames: Some(49),
        enhance_prompt: true,
        ..Default::default()
    }
}

#[test]
fn candle_descriptor_does_not_advertise_prompt_enhancement() {
    let descriptor = candle_gen_ltx::descriptor();
    assert!(
        !descriptor.capabilities.supports_prompt_enhancement,
        "{}/{}: the candle LTX provider has no enhancer, so it must not advertise \
         supports_prompt_enhancement",
        descriptor.backend,
        descriptor.id,
    );
}

#[test]
fn candle_refuses_an_enhance_prompt_request() {
    let descriptor = candle_gen_ltx::descriptor();

    // The control: the same request without the flag is accepted, so the refusal below is
    // attributable to `enhance_prompt` and not to the rest of the request being invalid.
    let accepted = GenerationRequest {
        enhance_prompt: false,
        ..enhance_request()
    };
    descriptor
        .capabilities
        .validate_request(descriptor.id, &accepted)
        .expect("the same request without enhance_prompt must be accepted");

    match descriptor
        .capabilities
        .validate_request(descriptor.id, &enhance_request())
    {
        Err(Error::Unsupported(message)) => assert!(
            message.contains("prompt enhancement"),
            "refusal must name the unsupported capability, got: {message}"
        ),
        other => panic!(
            "enhance_prompt=true must be refused as Error::Unsupported by the shared floor, got \
             {other:?}"
        ),
    }
}
