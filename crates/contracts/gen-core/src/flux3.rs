//! FLUX 3 \[klein\] request-surface contract: the provider ids, the prompt-upsampler token cap, the
//! layout document and reference bindings, and the reference output size a
//! [`GenerationRequest`](crate::GenerationRequest) carries. Everything else (resources, conversion,
//! PUP, references, training) lives in the sibling `gen-core-flux3` crate (sc-27316).

/// Registry id of the distilled text-to-image route (the SceneWorks worker's `payload.model`).
pub const DISTILLED_MODEL_ID: &str = "flux3_klein";
/// Registry id of the CFG base checkpoint.
pub const BASE_MODEL_ID: &str = "flux3_klein_base";

pub mod layouts;
pub mod pup;
pub mod references;
