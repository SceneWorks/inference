//! `mlx_gen_krea::model::validate_request` is reachable — and useful — from **outside** the crate.
//!
//! This file exists for one reason, and it is not line coverage: it is the only test that fails if
//! someone re-privatises that function. It was `pub(crate)`, which is why SceneWorks' Aether Studio
//! hand-mirrors Krea's request rules to type-check a request before paying for a 12B-parameter load —
//! a copy that can only drift. An in-crate unit test cannot notice a visibility change (`pub(crate)`
//! is visible to it), so the check has to live in an integration target, where narrowing the function
//! again is a **compile error** rather than a silent loss of API.
//!
//! The second claim pinned here is "without weights". Every call below passes a `ModelDescriptor`
//! from [`mlx_gen_krea::descriptor`] and a plain `GenerationRequest`; there is no `Generator`, no
//! snapshot directory and no `mlx_rs::Array` in this file, so it runs on a fresh clone with no
//! weights and no Metal device — which is precisely the situation the caller this change serves is
//! in when it wants an answer.
//!
//! Rule-by-rule coverage of the ~20 individual constraints lives in the crate's own unit tests
//! (`model.rs`'s `mod tests`) and is deliberately not duplicated. The cases below are chosen to span
//! each of the function's three layers exactly once — a Krea-specific rule, a shared
//! `Capabilities`-floor rule, and the typed capability-gap error — so that a refactor which dropped
//! the `capabilities.validate_request(...)` call from the middle of the function is caught here too,
//! not just the visibility.

use mlx_gen::{Error, GenerationRequest};
use mlx_gen_krea::descriptor;
use mlx_gen_krea::model::validate_request;

/// An otherwise-legal Turbo request at the given size. `prompt` is non-empty because Krea rejects an
/// empty one before it reaches anything else.
fn req(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a photo of a cat".into(),
        width,
        height,
        ..Default::default()
    }
}

#[test]
fn an_in_surface_request_type_checks_without_weights() {
    validate_request(&descriptor(), &req(1024, 1024))
        .expect("1024x1024 is in range, a multiple of 16, and inside every advertised bound");
}

#[test]
fn out_of_surface_requests_are_rejected_across_all_three_layers() {
    let d = descriptor();

    // Layer 1 — Krea's OWN constraint. The DiT patchifies on a 16px stride, so a non-multiple is
    // unrenderable. This rule exists nowhere in `Capabilities`: it is exactly the knowledge a caller
    // could not get before this function was public, and the reason mirroring it by hand was the only
    // alternative.
    let stride = validate_request(&d, &req(1000, 1024))
        .expect_err("1000 is not a multiple of the 16px DiT patch stride");
    assert!(
        stride.to_string().contains("must be a multiple of"),
        "expected the 16px-stride rejection, got: {stride}"
    );

    // Layer 2 — the SHARED floor, reached *through* this function. Asserting it here rather than only
    // in gen-core is what proves `validate_request` still composes `Capabilities::validate_request`
    // instead of having quietly become Krea-only checks; a caller that used this as its pre-load gate
    // would then be missing every advertised bound.
    let range = validate_request(&d, &req(128, 128))
        .expect_err("128x128 is below krea_2_turbo's advertised min_size of 256");
    assert!(
        range.to_string().contains("outside supported range"),
        "expected the shared size-range rejection, got: {range}"
    );

    // Layer 3 — the error TYPE survives the trip out of the crate. A capability gap is
    // `Error::Unsupported` and a malformed value is `Error::Msg` (F-008); a consumer routes on that
    // distinction to say "this backend can't do that" rather than "your request is wrong", so a
    // pre-load gate that flattened both to a string would be strictly worse than the mirror it
    // replaces. Turbo is CFG-free and advertises no negative prompt.
    let gap = validate_request(
        &d,
        &GenerationRequest {
            negative_prompt: Some("blurry".into()),
            ..req(1024, 1024)
        },
    )
    .expect_err("krea_2_turbo is CFG-free and advertises no negative prompt");
    assert!(
        matches!(gap, Error::Unsupported(_)),
        "a capability gap must stay a typed Error::Unsupported, got: {gap:?}"
    );
}
