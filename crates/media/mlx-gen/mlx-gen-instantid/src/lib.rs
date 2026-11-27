//! # mlx-gen-instantid
//!
//! InstantID (epic 3109) — identity-preserving SDXL. Builds on the existing crates rather than
//! re-implementing them:
//! - the face **Resampler** (ArcFace 512 → 16×2048 face tokens) is `mlx-gen-sdxl`'s generic IP-Adapter
//!   `Resampler` under `ResamplerConfig::instantid_face()` (sc-3110);
//! - the **IdentityNet** is a stock diffusers SDXL `ControlNet` (sc-3112) — `ControlNet::from_weights`
//!   + `UNetConfig::sdxl_base()`, no conversion;
//! - the **decoupled cross-attention** (16 face tokens) reuses `load_ip_kv_pairs` + the UNet's
//!   `install_ip_adapter` (sc-3113).
//!
//! This crate adds the InstantID-specific glue. [`kps`] is the keypoint control-image renderer
//! (sc-3111) — a bit-exact port of the vendored `draw_kps` (OpenCV 4.13) plus the letterbox aspect
//! handling and the canonical multi-view landmark sets.

pub mod kps;
pub mod memory_strategy;
pub mod model;
pub mod openpose;
mod registry;
pub mod restore;

pub use kps::{draw_kps, letterbox, view_angle_kps, ANGLE_SET_ORDER, VIEW_ANGLE_KPS};
pub use model::{
    InstantId, InstantIdPaths, InstantIdRequest, DEFAULT_CONTROLNET_SCALE, DEFAULT_IP_SCALE,
    DEFAULT_OPENPOSE_SCALE, FACE_RESTORE_PROMPT, SIZE_MULTIPLE,
};
pub use openpose::{
    draw_bodypose, face_box_from_keypoints, normalize_keypoints, square_fit, BodyPoint,
    NUM_BODY_KEYPOINTS, STICKWIDTH,
};
pub use registry::{descriptor, load_instantid};
pub use restore::{feather_mask, paste_alpha};

/// Add the MLX InstantID generator to an explicit media registry builder (`SW-6982`, `REG-6`).
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_route(registry::route_instantid())
}

/// Build the complete explicit MLX InstantID provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(explicit, ["instantid"]);
        assert!(registry.route_conformance_errors().is_empty());
        assert!(registry.descriptor_conformance_errors().is_empty());
    }
}
