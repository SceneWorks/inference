#![allow(dead_code)]
//! Cross-backend fixture geometry for the Bernini parity goldens (sc-19496).
//!
//! `mod common;` is compiled into every including test binary, so only a subset is used by any one of
//! them.

// --- Cross-backend fixture geometry (sc-19496) ---------------------------------------------------
//
// Ten of the fixture files under `tests/fixtures/` here are committed **byte-identical** to the file
// `candle-gen-bernini` commits under the same name (`assembly`, `clip_diff`, `handoff`, `mar`,
// `process`, `qwen_backbone`, `template`, `vision_tower`, `vit_guidance` and `vit_preprocess`
// goldens). Both lanes load the same bytes, so a drift in either lane's hand-typed geometry leaves
// both lanes internally consistent and both parity suites green while the two backends compare
// tensors dumped at one shape against a model built at another. Nothing could see that: the two
// crates cannot import each other, because `mlx-gen-*` builds on macOS only.
//
// Most of this family's fixture geometry does not need pinning here at all, and deliberately is not:
// the goldens carry their own `__metadata__` and both lanes parse it (`Weights::metadata` here,
// `Golden::meta_req` there), so the fixture's own bytes are the single source and drift is
// impossible by construction. What follows is the remainder — the values both lanes genuinely
// hand-type because the golden does not record them. `check_cross_backend_geometry` in
// `scripts/check-workspace.py` compares every `SHARED_FIXTURE_*` declaration under this crate's
// `tests/` against the candle crate's, by name set and by value.

/// Assembly-fixture backbone depth: 0 layers, so only the token embedding is exercised.
pub const SHARED_FIXTURE_ASSEMBLY_NUM_LAYERS: i32 = 0;
/// Assembly-fixture attention heads.
pub const SHARED_FIXTURE_ASSEMBLY_NUM_HEADS: i32 = 2;
/// Assembly-fixture key/value heads (GQA).
pub const SHARED_FIXTURE_ASSEMBLY_NUM_KV_HEADS: i32 = 1;
/// Assembly-fixture per-head width.
pub const SHARED_FIXTURE_ASSEMBLY_HEAD_DIM: i32 = 8;
/// Assembly-fixture feed-forward width.
pub const SHARED_FIXTURE_ASSEMBLY_INTERMEDIATE_SIZE: i32 = 32;
/// Assembly-fixture RMSNorm epsilon.
pub const SHARED_FIXTURE_ASSEMBLY_RMS_NORM_EPS: f32 = 1e-6;
/// Assembly-fixture RoPE base.
pub const SHARED_FIXTURE_ASSEMBLY_ROPE_THETA: f32 = 1_000_000.0;
/// Assembly-fixture MRoPE per-axis (T/H/W) frequency counts.
pub const SHARED_FIXTURE_ASSEMBLY_MROPE_SECTION: [usize; 3] = [1, 2, 1];

/// ViT-guidance fixture: the image-conditioned guidance weight the golden was dumped at.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_IMG: f32 = 4.5;
/// ViT-guidance fixture: the text-conditioned guidance weight.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_TXT: f32 = 4.0;
/// ViT-guidance fixture: the target-conditioned guidance weight.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_TGT: f32 = 3.0;
/// ViT-guidance fixture: the video-conditioned guidance weight (the `rv2v` chain's first rung).
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_VID: f32 = 1.25;
/// ViT-guidance fixture: `apg_delta`'s eta (the parallel-component retention).
pub const SHARED_FIXTURE_VIT_GUIDANCE_APG_ETA: f32 = 0.2;
/// ViT-guidance fixture: `apg_delta`'s norm threshold.
pub const SHARED_FIXTURE_VIT_GUIDANCE_APG_NORM_THRESHOLD: f32 = 1.0;

/// Template-fixture task mixes, in the order the golden dumps them.
pub const SHARED_FIXTURE_TEMPLATE_TASKS: [&str; 4] = ["t2i", "i2i", "r2v", "rv2v"];
/// Template-fixture prompts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_PROMPTS: [&str; 4] = ["a cat", "edit", "subj", "edit v"];
/// Template-fixture input reference images `(h, w)`, one list per task.
pub const SHARED_FIXTURE_TEMPLATE_INPUT_IMAGE_HW: [&[(i64, i64)]; 4] =
    [&[], &[(48, 72)], &[(72, 48)], &[]];
/// Template-fixture input reference-video counts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_INPUT_VIDEO_COUNT: [usize; 4] = [0, 0, 0, 1];
/// Template-fixture output frame counts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_T: [i64; 4] = [1, 1, 9, 9];
/// Template-fixture output height.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_H: i64 = 64;
/// Template-fixture output width.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_W: i64 = 64;
/// Template-fixture image token counts, one list per task — grids match the process golden, so
/// `token_num = t·(h/2)·(w/2)`.
pub const SHARED_FIXTURE_TEMPLATE_IMAGE_TOKEN_NUMS: [&[i64]; 4] = [&[4], &[6, 4], &[6], &[]];
/// Template-fixture video token counts, one list per task.
pub const SHARED_FIXTURE_TEMPLATE_VIDEO_TOKEN_NUMS: [&[i64]; 4] = [&[], &[], &[12], &[12, 20]];
