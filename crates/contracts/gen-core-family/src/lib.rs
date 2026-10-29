//! # gen-core-family
//!
//! Small tensor-free per-family host policy split out of gen-core in sc-27316, each module used by
//! one model family's MLX and Candle providers: schedulers (Ideogram, Krea multiphase/schedule),
//! resolution and geometry (Lens, SeedVR2 video, SDXL LDM), keypoint and face-alignment math
//! (InstantID, face/FX), the Qwen-Image 2.1 config/reference/scheduler, and the SD3 text-encoder
//! artifact resolution and seeded-output mapping. Grouped so a change to one of them rebuilds
//! only these families, never gen-core.

pub mod face_align;
pub mod fx_program;
pub mod ideogram_scheduler;
pub mod instantid_kps;
pub mod krea_multiphase;
pub mod krea_schedule;
pub mod lens_resolution;
pub mod qwen_image_2_1;
pub mod sd3_encoder_artifacts;
pub mod sd3_request;
pub mod sdxl_ldm;
pub mod seedvr2_video;

pub use sd3_encoder_artifacts::{
    resolve_sd3_text_encoder_artifacts, Sd3TextEncoderArtifactError, Sd3TextEncoderArtifacts,
};
pub use sd3_request::map_sd3_seeded_outputs;
