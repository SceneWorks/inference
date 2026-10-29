//! The reference output size a FLUX 3 [`GenerationRequest`](crate::GenerationRequest) carries.
//! Reference planning and preparation live in `gen_core_flux3::references` (sc-27316).

/// Original optional dimensions, before a consumer inserts its ordinary T2I defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReferenceOutputSize {
    pub width: Option<u32>,
    pub height: Option<u32>,
}
