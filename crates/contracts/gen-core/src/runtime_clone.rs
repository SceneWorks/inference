//! Exhaustive owned-field clone law; see bounded_clone for container authority.
use super::*;
use crate::source_inventory::bounded_clone::{scalar, sum, CloneLaw};

impl CloneLaw for WeightsSource {
    fn clone_bytes(&self) -> crate::Result<u64> {
        match self {
            Self::File(path) | Self::Dir(path) => path.clone_bytes(),
        }
    }
}
impl CloneLaw for FileStatFingerprint {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            size,
            modified,
            is_symlink,
            symlink_target,
            device,
            inode,
            changed_seconds,
            changed_nanoseconds,
        } = self;
        sum([
            scalar(size),
            scalar(modified),
            scalar(is_symlink),
            symlink_target.clone_bytes(),
            scalar(device),
            scalar(inode),
            scalar(changed_seconds),
            scalar(changed_nanoseconds),
        ])
    }
}
impl CloneLaw for PathComponentFingerprint {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            path,
            is_symlink,
            symlink_target,
            device,
            inode,
        } = self;
        sum([
            path.clone_bytes(),
            scalar(is_symlink),
            symlink_target.clone_bytes(),
            scalar(device),
            scalar(inode),
        ])
    }
}
impl CloneLaw for ArtifactSeal {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            loader_path,
            canonical_target_path,
            path_component_fingerprints,
            entry_fingerprint,
            target_fingerprint,
            sha256,
            #[cfg(test)]
            hash_work,
        } = self;
        #[cfg(test)]
        let _shared_counter = hash_work.clone_bytes()?;
        sum([
            loader_path.clone_bytes(),
            canonical_target_path.clone_bytes(),
            path_component_fingerprints.clone_bytes(),
            entry_fingerprint.clone_bytes(),
            target_fingerprint.clone_bytes(),
            scalar(sha256),
        ])
    }
}
impl CloneLaw for PreparedFilePins {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            prepared,
            finalized,
            pins,
        } = self;
        sum([scalar(prepared), scalar(finalized), pins.clone_bytes()])
    }
}
impl CloneLaw for PidWeights {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self { checkpoint, gemma } = self;
        sum([checkpoint.clone_bytes(), gemma.clone_bytes()])
    }
}
impl CloneLaw for IdentityWeights {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            encoder,
            eva,
            face_dir,
        } = self;
        sum([
            encoder.clone_bytes(),
            eva.clone_bytes(),
            face_dir.clone_bytes(),
        ])
    }
}
impl CloneLaw for AdapterSpec {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            path,
            scale,
            kind,
            pass_scales,
            moe_expert,
        } = self;
        sum([
            path.clone_bytes(),
            scalar(scale),
            scalar(kind),
            pass_scales.clone_bytes(),
            scalar(moe_expert),
        ])
    }
}
impl CloneLaw for LoadSpec {
    fn clone_bytes(&self) -> crate::Result<u64> {
        let Self {
            weights,
            resolved_route,
            decode_geometry_policies,
            decode_geometry_policy_authoritative,
            decode_quality_runtime_identity,
            prepared_file_pins,
            prepared_receipt_paths,
            prepared_encoder_receipt,
            source_inventory,
            quantize,
            precision,
            yue2_ar_mode,
            yue2_compute_policy,
            control,
            extra_controls,
            ip_adapter,
            adapters,
            pid,
            identity,
            text_encoder,
            offload_policy,
            load_shape,
            load_shape_declaration_result,
            components,
        } = self;
        sum([
            weights.clone_bytes(),
            resolved_route.clone_bytes(),
            decode_geometry_policies.clone_bytes(),
            scalar(decode_geometry_policy_authoritative),
            decode_quality_runtime_identity.clone_bytes(),
            prepared_file_pins.clone_bytes(),
            prepared_receipt_paths.clone_bytes(),
            prepared_encoder_receipt.clone_bytes(),
            source_inventory.clone_bytes(),
            scalar(quantize),
            scalar(precision),
            scalar(yue2_ar_mode),
            scalar(yue2_compute_policy),
            control.clone_bytes(),
            extra_controls.clone_bytes(),
            ip_adapter.clone_bytes(),
            adapters.clone_bytes(),
            pid.clone_bytes(),
            identity.clone_bytes(),
            text_encoder.clone_bytes(),
            scalar(offload_policy),
            scalar(load_shape),
            scalar(load_shape_declaration_result),
            components.clone_bytes(),
        ])
    }
}
