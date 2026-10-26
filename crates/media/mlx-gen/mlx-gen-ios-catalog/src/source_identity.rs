//! Engineering evidence names executable sources separately from its capture commit. Evidence
//! rows and reviewed policy data may be committed afterwards without claiming a new device run.
//!
//! Rust TypeIds of crate-owned provider anchors bind the compiled dependency units. On the
//! pinned Cargo/rustc toolchain their crate disambiguators include Cargo's recursive unit
//! metadata, including features unified by downstream users of local, registry and git crates.
//! This is a build-system identity, not a stable Rust ABI: compiler/Cargo identity is bound too,
//! and the regression verifies transitive feature changes, rebuild and checkout stability.
//! Native MLX artifacts remain separately bound. The catalog itself has no anchor so its
//! capture-only permit and evidence/policy DATA do not invalidate the production closure.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

include!(concat!(env!("OUT_DIR"), "/ios_source_identity.rs"));

/// Source and effective build identity carried by engineering captures and reviewed policy.
/// `build_sha` remains a separate, original capture provenance field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub source_closure_sha256: String,
    pub backend: String,
    pub target: String,
    pub rustc: String,
    /// Enabled catalog features. Resolved dependency features are sealed in build_config_sha256.
    pub features: String,
    pub build_config_sha256: String,
    /// Hash of the actual mlx-sys manifest, archives and metallib; "unavailable" is unverified.
    pub native_mlx_fingerprint: String,
}

impl BuildIdentity {
    pub fn current() -> Self {
        Self {
            source_closure_sha256: SOURCE_CLOSURE_SHA256.into(),
            backend: "mlx".into(),
            target: BUILD_TARGET.into(),
            rustc: BUILD_RUSTC.into(),
            features: BUILD_FEATURES.into(),
            build_config_sha256: compiled_dependency_identity(),
            native_mlx_fingerprint: NATIVE_MLX_FINGERPRINT.into(),
        }
    }

    /// The capture-only permit exposes an entrypoint without changing production execution.
    /// All other features, including model-inference-proof, remain identity-bearing.
    pub fn same_production_execution(&self, other: &Self) -> bool {
        let production_features = |features: &str| {
            let mut features: Vec<_> = features
                .split(',')
                .filter(|f| !f.is_empty() && *f != "device-capture")
                .map(str::to_owned)
                .collect();
            features.sort();
            features.dedup();
            features
        };
        let digest = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        digest(&self.source_closure_sha256)
            && digest(&self.build_config_sha256)
            && digest(&self.native_mlx_fingerprint)
            && !self.target.is_empty()
            && !self.rustc.is_empty()
            && self.source_closure_sha256 == other.source_closure_sha256
            && self.backend == "mlx"
            && self.backend == other.backend
            && self.target == other.target
            && self.rustc == other.rustc
            && self.build_config_sha256 == other.build_config_sha256
            && self.native_mlx_fingerprint == other.native_mlx_fingerprint
            && production_features(&self.features) == production_features(&other.features)
    }
}

fn compiled_dependency_identity() -> String {
    if !(mlx_gen::IOS_ENGINEERING_FEATURES_SUPPORTED
        && mlx_gen_sana::IOS_ENGINEERING_FEATURES_SUPPORTED
        && mlx_gen_anima::IOS_ENGINEERING_FEATURES_SUPPORTED
        && mlx_gen_iris::IOS_ENGINEERING_FEATURES_SUPPORTED
        && mlx_gen_sdxl::IOS_ENGINEERING_FEATURES_SUPPORTED
        && mlx_gen_z_image::IOS_ENGINEERING_FEATURES_SUPPORTED)
    {
        // Not a SHA256: even an identical capture cannot authorize unsupported compiled code.
        return "unsupported-compiled-features".into();
    }
    use std::any::TypeId;
    let units = [
        TypeId::of::<mlx_gen::IosCompiledDependencyAnchor>(),
        TypeId::of::<mlx_gen_sana::IosCompiledDependencyAnchor>(),
        TypeId::of::<mlx_gen_anima::IosCompiledDependencyAnchor>(),
        TypeId::of::<mlx_gen_iris::IosCompiledDependencyAnchor>(),
        TypeId::of::<mlx_gen_sdxl::IosCompiledDependencyAnchor>(),
        TypeId::of::<mlx_gen_z_image::IosCompiledDependencyAnchor>(),
    ];
    format!(
        "{:x}",
        Sha256::digest(
            format!("ios-compiled-dependency-units-v1\0{BUILD_CONFIG_SHA256}\0{units:?}")
                .as_bytes()
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> BuildIdentity {
        BuildIdentity {
            source_closure_sha256: "1".repeat(64),
            backend: "mlx".into(),
            target: "aarch64-apple-ios".into(),
            rustc: "rustc test fixture".into(),
            features: String::new(),
            build_config_sha256: "2".repeat(64),
            native_mlx_fingerprint: "3".repeat(64),
        }
    }

    #[test]
    fn only_the_capture_permit_is_compatible_with_the_product() {
        let product = known();
        let mut capture = product.clone();
        capture.features = "device-capture".into();
        assert!(product.same_production_execution(&capture));
        capture.features = "device-capture,model-inference-proof".into();
        assert!(!product.same_production_execution(&capture));
        capture = product.clone();
        capture.native_mlx_fingerprint = "4".repeat(64);
        assert!(!product.same_production_execution(&capture));
        capture = product.clone();
        capture.build_config_sha256 = "5".repeat(64);
        assert!(!product.same_production_execution(&capture));
    }

    #[test]
    fn missing_native_identity_never_matches_itself() {
        let mut identity = known();
        identity.native_mlx_fingerprint = "unavailable".into();
        assert!(!identity.same_production_execution(&identity));
    }
}
