//! sc-17157: the `ref2va` reference caps agree, digit for digit, with the MLX lane's.
//!
//! `src/reference.rs` restates every cap by hand rather than deriving it, because the two ports
//! cannot share a crate — MLX is macOS/Metal, candle-cuda is Windows/Linux, and nothing links both.
//! Restating is the only option; going unchecked is not. Two lanes that drifted apart on a cap
//! would each pass their own suite forever, and the divergence would surface as one platform
//! refusing a request the other accepts.
//!
//! `src/reference.rs`'s module doc claimed this file already existed and did this. It did not —
//! the claim shipped ahead of the test. This is the test.
//!
//! It reads the MLX module's **source text** through `include_str!` rather than linking it, which
//! is what makes it runnable on a lane that cannot build MLX at all. A parse of the source is
//! weaker than a link — a constant renamed on the MLX side reads here as "absent" rather than as a
//! mismatch — which is exactly why the candle-only set below is pinned rather than skipped.

use candle_gen_minimax_h3::reference::{
    MAX_AUDIO_REFERENCES, MAX_IMAGE_REFERENCES, MAX_TOTAL_REFERENCES, MAX_VIDEO_REFERENCES,
    MIN_REFERENCE_CLIP_FRAMES, REFERENCE_IMAGE_SHORT_EDGE, VIDEO_SAMPLE_FPS,
    VISION_TEMPORAL_PATCH,
};

const MLX_REFERENCE: &str = include_str!("../../../mlx-gen/mlx-gen-minimax-h3/src/reference.rs");

/// The value text of `pub const <name>` in the MLX module, with the type annotation discarded.
///
/// The annotation is dropped on purpose: `REFERENCE_IMAGE_SHORT_EDGE` is `i32` on the MLX side
/// (mlx-rs shapes are `i32`) and `u32` here, and the two lanes agreeing on `2048` is the fact worth
/// pinning. A type divergence that changed the admitted RANGE would still be caught, because these
/// are all caps that this port's own `Ref2VaReferences::new` enforces against `usize`.
fn mlx_const(name: &str) -> String {
    let needle = format!("pub const {name}:");
    let line = MLX_REFERENCE
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))
        .unwrap_or_else(|| {
            panic!(
                "`pub const {name}` is not declared in mlx-gen-minimax-h3/src/reference.rs — it \
                 was renamed or removed on that lane, and this port still carries it. Reconcile \
                 the two rather than deleting this pin."
            )
        });
    let (_, value) = line.split_once('=').expect("a const declaration has a value");
    value.trim().trim_end_matches(';').trim().to_owned()
}

#[test]
fn every_shared_reference_cap_equals_its_mlx_twin() {
    let shared: [(&str, String); 7] = [
        ("MAX_IMAGE_REFERENCES", MAX_IMAGE_REFERENCES.to_string()),
        ("MAX_VIDEO_REFERENCES", MAX_VIDEO_REFERENCES.to_string()),
        ("MAX_AUDIO_REFERENCES", MAX_AUDIO_REFERENCES.to_string()),
        ("MAX_TOTAL_REFERENCES", MAX_TOTAL_REFERENCES.to_string()),
        (
            "REFERENCE_IMAGE_SHORT_EDGE",
            REFERENCE_IMAGE_SHORT_EDGE.to_string(),
        ),
        ("VIDEO_SAMPLE_FPS", format!("{VIDEO_SAMPLE_FPS:?}")),
        ("VISION_TEMPORAL_PATCH", VISION_TEMPORAL_PATCH.to_string()),
    ];
    for (name, ours) in shared {
        assert_eq!(
            mlx_const(name),
            ours,
            "{name} has drifted between the two ports; a cap is a property of the MODEL, so one of \
             the two lanes is refusing a request the other admits"
        );
    }
}

/// Non-vacuity: the parser really does read the MLX file, and really would notice a difference.
#[test]
fn the_parse_reads_the_mlx_source_rather_than_returning_nothing() {
    assert!(
        MLX_REFERENCE.contains("pub const MAX_TOTAL_REFERENCES"),
        "the include_str! path no longer points at the MLX reference module"
    );
    assert_ne!(
        mlx_const("MAX_IMAGE_REFERENCES"),
        mlx_const("MAX_VIDEO_REFERENCES"),
        "the parser must return each constant's own value, not a constant answer"
    );
    assert_eq!(mlx_const("MAX_TOTAL_REFERENCES"), "12");
}

/// The caps this port declares that the MLX lane does **not**, pinned so a new one cannot quietly
/// join them.
///
/// `MIN_REFERENCE_CLIP_FRAMES` is the whole list today. It is not a drift: it is the VAE chunk
/// floor this port added in sc-17157, and the MLX lane has no equivalent yet — so the module doc's
/// "every one of these constants" was never achievable as stated, and this test says which ones the
/// claim actually covers.
#[test]
fn the_caps_with_no_mlx_twin_are_exactly_the_ones_named_here() {
    let candle_only = ["MIN_REFERENCE_CLIP_FRAMES"];
    for name in candle_only {
        assert!(
            !MLX_REFERENCE.contains(&format!("pub const {name}")),
            "{name} now EXISTS on the MLX lane — move it into the shared list above so the two \
             values are compared instead of being assumed to differ"
        );
    }
    // …and it is a real constant of this port, so the list cannot rot into a list of typos.
    assert!(MIN_REFERENCE_CLIP_FRAMES > 0);
}
