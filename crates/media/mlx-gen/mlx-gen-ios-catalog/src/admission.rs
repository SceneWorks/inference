//! Artifact applicability is separate from measurement currency. No revision/pin freshness gate:
//! memory evidence applies only to the hardware, provider and actual stored tensor layout measured.

use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype};
use mlx_gen::gen_core::{Error, LoadSpec, Precision, Result, WeightsSource};
use sha2::{Digest, Sha256};

use crate::{FootprintBound, IOS_OFFLOAD_POLICY};

/// Header-only memory-layout identity, independent of directory/file names and weight values.
/// All tensor keys, shapes, storage dtypes and component membership are hashed, not just the
/// embedding or the caller's requested quant tier. This is not a weight-content quality claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SanaSnapshotProfile {
    pub layout_sha256: [u8; 32],
    pub component_bytes: [u64; 3],
    pub embedding_dtype: Dtype,
    pub embedding_shape: Vec<usize>,
}

/// Inspect headers only. No MLX arrays or tensor payload reads.
pub fn sana_snapshot_profile(spec: &LoadSpec) -> Result<SanaSnapshotProfile> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Unsupported(
            "iOS SANA evidence requires a snapshot directory".into(),
        ));
    };
    if spec.offload_policy != IOS_OFFLOAD_POLICY
        || spec.precision != Precision::Bf16
        || !spec.components.is_empty()
        || !spec.adapters.is_empty()
        || spec.control.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(Error::Unsupported(
            "iOS SANA evidence requires native stored precision, Sequential, and no load overrides"
                .into(),
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"sana-ios-memory-layout-v1");
    let mut component_bytes = [0u64; 3];
    let mut embedding = None;
    for (i, component) in ["transformer", "text_encoder", "vae"].iter().enumerate() {
        let dir = root.join(component);
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            // Match the actual loader inventory, not the header helper's recursive directory mode.
            if path.is_dir() && !mlx_gen::gen_core::weightsmeta::is_hidden_dir(&path) {
                return Err(Error::Unsupported(format!(
                    "iOS SANA evidence: nested component {}",
                    path.display()
                )));
            }
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors")
                && !mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
            {
                files.push(path);
            }
        }
        files.sort();
        // The TE loader opens this specific file. This is an inventory check, NOT evidence based
        // on its filename: every tensor header still participates in the identity below.
        if files.is_empty()
            || (*component == "text_encoder"
                && (files.len() != 1
                    || files[0].file_name().and_then(|s| s.to_str())
                        != Some("gemma-2-2b-it.safetensors")))
        {
            return Err(Error::Unsupported(format!(
                "iOS SANA evidence: incompatible {component} shard inventory"
            )));
        }
        let mut headers = std::collections::BTreeMap::new();
        for file in files {
            for header in safetensors_path_tensor_headers(file)? {
                if headers.insert(header.name.clone(), header).is_some() {
                    return Err(Error::Unsupported(format!(
                        "iOS SANA evidence: duplicate {component} tensor"
                    )));
                }
            }
        }
        digest.update(component.as_bytes());
        for header in headers.values() {
            component_bytes[i] = component_bytes[i]
                .checked_add(header.data_bytes)
                .ok_or_else(|| Error::Unsupported("iOS SANA tensor bytes overflow".into()))?;
            digest.update((header.name.len() as u64).to_le_bytes());
            digest.update(header.name.as_bytes());
            digest.update(header.dtype.as_str().as_bytes());
            digest.update((header.shape.len() as u64).to_le_bytes());
            for &dim in &header.shape {
                digest.update((dim as u64).to_le_bytes());
            }
            digest.update(header.data_bytes.to_le_bytes());
            if *component == "text_encoder" && header.name == "model.embed_tokens.weight" {
                embedding = Some((header.dtype, header.shape.clone()));
            }
        }
    }
    let (embedding_dtype, embedding_shape) = embedding
        .ok_or_else(|| Error::Unsupported("iOS SANA snapshot lacks token embedding".into()))?;
    Ok(SanaSnapshotProfile {
        layout_sha256: digest.finalize().into(),
        component_bytes,
        embedding_dtype,
        embedding_shape,
    })
}

/// A capture binds its OWN footprint rows, never silently borrows historical rows. Private so an
/// arbitrary caller/sidecar cannot manufacture admission authority from its own snapshot profile.
struct SanaDeviceEvidence<'a> {
    machine: &'a str,
    provider: &'a str,
    layout_sha256: [u8; 32],
    footprints: &'a [(u32, u32, FootprintBound)],
}

fn applicable_bound(
    profile: &SanaSnapshotProfile,
    machine: &str,
    provider: &str,
    evidence: Option<&SanaDeviceEvidence<'_>>,
    image: u32,
    tile: u32,
) -> Option<FootprintBound> {
    let evidence = evidence?;
    if evidence.machine != machine
        || evidence.provider != provider
        || evidence.layout_sha256 != profile.layout_sha256
    {
        return None;
    }
    evidence
        .footprints
        .iter()
        .find(|(edge, decode, _)| *edge == image && *decode == tile)
        .map(|(_, _, bound)| *bound)
}

/// Product admission checks the actual snapshot AND hardware/provider before returning a bound.
/// Historical iPhone 17 observations have no recoverable artifact layout identity, so currently
/// no product snapshot is authorized. Preserve them as planning data, not target-device evidence.
/// A new validated capture can populate an owned, artifact-bound record here; the explicitly
/// UNADMITTED diagnostic route is retained to collect it. Launch headroom is not such a capture.
pub fn sana_snapshot_footprint_bound(
    provider: &str,
    spec: &LoadSpec,
    machine: &str,
    image: u32,
    tile: u32,
) -> Result<FootprintBound> {
    let profile = sana_snapshot_profile(spec)?;
    applicable_bound(&profile, machine, provider, None, image, tile).ok_or_else(|| {
        Error::Unsupported(format!(
            "iOS SANA admission: no artifact-bound device footprint for {provider} on {machine}; \
             layout {:02x?}, component bytes {:?}, embedding {:?} {:?}. Historical iPhone17 \
             observations do not authorize this snapshot/device",
            profile.layout_sha256,
            profile.component_bytes,
            profile.embedding_dtype,
            profile.embedding_shape
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    struct Fixture(tempfile::TempDir);
    impl Fixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            for name in ["transformer", "text_encoder", "vae"] {
                std::fs::create_dir(root.path().join(name)).unwrap();
            }
            write_tensor(
                &root.path().join("transformer/a.safetensors"),
                "proj.weight",
                "U32",
                "[8,16]",
                512,
            );
            write_tensor(
                &root.path().join("vae/a.safetensors"),
                "conv.weight",
                "F32",
                "[8,8]",
                256,
            );
            Self(root)
        }
        fn spec(&self) -> LoadSpec {
            crate::sana_load_spec(WeightsSource::Dir(self.0.path().to_path_buf()))
        }
        fn embedding(&self, dtype: &str, shape: &str, bytes: usize) {
            write_tensor(
                &self.0.path().join("text_encoder/gemma-2-2b-it.safetensors"),
                "model.embed_tokens.weight",
                dtype,
                shape,
                bytes,
            );
        }
    }

    // Genuine tiny safetensors fixtures; no MLX backend, GPU or model loads.
    fn write_tensor(path: &Path, key: &str, dtype: &str, shape: &str, bytes: usize) {
        let header = format!("{{\"{key}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[0,{bytes}]}}}}");
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.resize(file.len() + bytes, 0);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn dense_larger_or_different_component_cannot_borrow_packed_evidence() {
        let fixture = Fixture::new();
        fixture.embedding("U32", "[7,16]", 448);
        let packed = sana_snapshot_profile(&fixture.spec()).unwrap();
        let rows = [(1024, 192, FootprintBound::Measured(3000))];
        let evidence = SanaDeviceEvidence {
            machine: "iPhone17,2",
            provider: "sana_1600m",
            layout_sha256: packed.layout_sha256,
            footprints: &rows,
        };
        let bound = |profile: &SanaSnapshotProfile, machine, provider| {
            applicable_bound(profile, machine, provider, Some(&evidence), 1024, 192)
        };
        assert_eq!(
            bound(&packed, "iPhone17,2", "sana_1600m"),
            Some(FootprintBound::Measured(3000))
        );
        assert_eq!(bound(&packed, "iPhone18,2", "sana_1600m"), None);
        assert_eq!(bound(&packed, "iPhone17,2", "sana_sprint_1600m"), None);
        assert_eq!(
            applicable_bound(
                &packed,
                "iPhone17,2",
                "sana_1600m",
                Some(&evidence),
                512,
                192
            ),
            None
        );
        assert!(
            sana_snapshot_footprint_bound("sana_1600m", &fixture.spec(), "iPhone17,2", 1024, 192)
                .is_err(),
            "unbound historical evidence cannot authorize even packed storage"
        );
        // Same paths and a requested q4 label cannot hide the much larger dense embedding.
        fixture.embedding("BF16", "[7,128]", 1792);
        let dense = sana_snapshot_profile(&fixture.spec()).unwrap();
        assert!(dense.component_bytes[1] > packed.component_bytes[1]);
        assert_eq!(bound(&dense, "iPhone17,2", "sana_1600m"), None);
        assert!(sana_snapshot_footprint_bound(
            "sana_1600m",
            &fixture.spec(),
            "iPhone17,2",
            1024,
            192
        )
        .is_err());
        fixture.embedding("F32", "[7,16]", 448); // same byte count and shape, different precision
        assert_eq!(
            bound(
                &sana_snapshot_profile(&fixture.spec()).unwrap(),
                "iPhone17,2",
                "sana_1600m"
            ),
            None
        );
        fixture.embedding("U32", "[14,8]", 448); // same byte count and dtype, different shape
        assert_eq!(
            bound(
                &sana_snapshot_profile(&fixture.spec()).unwrap(),
                "iPhone17,2",
                "sana_1600m"
            ),
            None
        );
        fixture.embedding("U32", "[7,32]", 896); // q8-shaped storage, not q4
        assert_eq!(
            bound(
                &sana_snapshot_profile(&fixture.spec()).unwrap(),
                "iPhone17,2",
                "sana_1600m"
            ),
            None
        );
        fixture.embedding("U32", "[7,16]", 448);
        write_tensor(
            &fixture.0.path().join("vae/a.safetensors"),
            "conv.weight",
            "BF16",
            "[8,8]",
            128,
        );
        assert_eq!(
            bound(
                &sana_snapshot_profile(&fixture.spec()).unwrap(),
                "iPhone17,2",
                "sana_1600m"
            ),
            None
        );
    }

    #[test]
    fn inventory_overrides_and_names_are_not_quantization_evidence() {
        let fixture = Fixture::new();
        fixture.embedding("U32", "[7,16]", 448);
        let profile = sana_snapshot_profile(&fixture.spec()).unwrap();
        std::fs::rename(
            fixture.0.path().join("transformer/a.safetensors"),
            fixture.0.path().join("transformer/renamed.safetensors"),
        )
        .unwrap();
        assert_eq!(
            sana_snapshot_profile(&fixture.spec()).unwrap(),
            profile,
            "identity is tensor-layout based, not a filename shortcut"
        );
        let mut spec = fixture.spec();
        spec.offload_policy = mlx_gen::gen_core::OffloadPolicy::Resident;
        assert!(sana_snapshot_profile(&spec).is_err());
        spec = fixture.spec();
        spec.quantize = Some(mlx_gen::gen_core::Quant::Q4);
        // SANA's quantize knob is advisory; the loader auto-detects actual on-disk storage.
        // It neither changes the memory identity nor manufactures evidence for a dense artifact.
        assert_eq!(sana_snapshot_profile(&spec).unwrap(), profile);
        assert!(
            sana_snapshot_footprint_bound("sana_1600m", &spec, "iPhone17,2", 1024, 192).is_err()
        );
        write_tensor(
            &fixture.0.path().join("transformer/duplicate.safetensors"),
            "proj.weight",
            "U32",
            "[8,16]",
            512,
        );
        assert!(sana_snapshot_profile(&fixture.spec()).is_err());
    }
}
