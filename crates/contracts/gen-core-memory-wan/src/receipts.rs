//! Pinned upstream revisions and the VACE / VACE-Fun source receipts.

use super::*;

/// Receipt spelling for the Wan video structural evidence revision.
///
/// `v5` (sc-20799) inserted the request frame rate as a **plain-text** field between the mode key
/// and the artifact identity:
/// `wan-video-structural-v5:<mode>:fps<fps>:<artifact_identity>:<selection_receipt>:<sha256>`.
/// The rate was always hashed, but the hash is opaque to a capture record and to
/// [`validate_context`], so a 16-fps and a 24-fps Ti2V-5B cell were indistinguishable at the
/// receipt surface and admission had to OR both public rate menus. With the rate readable,
/// admission pins the exact menu and an evidence collector can key the two cells apart.
pub const RECEIPT_VERSION: &str = "wan-video-structural-v5";
pub const LIGHTNING_REPOSITORY: &str = "lightx2v/Wan2.2-Lightning";
pub const LIGHTNING_REVISION: &str = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e";
pub const NATIVE_SCHEDULE: &str = "wan-flow-match-native";
pub const DECODE_TILE_EDGES: &[u32] = &[192, 256, 384];
pub const DECODE_OVERLAPS: &[u32] = &[64];
/// Immutable source receipt carried beside assembled Wan-VACE artifacts.  VACE repositories ship
/// the control transformer only; the UMT5, z16 VAE and tokenizer are separate upstream sources.
pub const VACE_SOURCE_RECEIPT_FILE: &str = "wan-vace-source-receipt.json";
pub const VACE_FUN_SOURCE_RECEIPT_FILE: &str = "wan-vace-fun-source-receipt.json";
pub(super) const VACE_SOURCE_RECEIPT_VERSION: &str = "wan-vace-source-receipt-v1";
pub(super) const VACE_FUN_SOURCE_RECEIPT_VERSION: &str = "wan-vace-fun-source-receipt-v1";
pub(super) const VACE_FUN_REPOSITORY: &str = "linoyts/Wan2.2-VACE-Fun-14B-diffusers";
pub(super) const VACE_FUN_REVISION: &str = "1abfb95801b7bd8f952083ebf80b93448ddb0ce4";
pub(super) const WAN21_T2V_REPOSITORY: &str = "Wan-AI/Wan2.1-T2V-14B-Diffusers";
pub(super) const WAN21_T2V_REVISION: &str = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68";
pub(super) const UMT5_REPOSITORY: &str = "google/umt5-xxl";
pub(super) const UMT5_REVISION: &str = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad";

pub(super) fn vace_source_receipt(backend: WanI2vBackend) -> serde_json::Value {
    let (transformer_repository, transformer_revision) = match backend {
        WanI2vBackend::Mlx => (
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "ec4d2cb062b548996b179d493fdd05340de702a1",
        ),
        WanI2vBackend::Candle => (
            "Wan-AI/Wan2.1-VACE-14B-diffusers",
            "db79b90c60bbb45ceec9e41b9d5a4df934538ac4",
        ),
    };
    serde_json::json!({
        "version": VACE_SOURCE_RECEIPT_VERSION,
        "backend": backend.key(),
        "components": [
            {
                "component": "transformer",
                "repository": transformer_repository,
                "revision": transformer_revision,
                "sourcePath": "transformer"
            },
            {
                "component": "t5_encoder",
                "repository": WAN21_T2V_REPOSITORY,
                "revision": WAN21_T2V_REVISION,
                "sourcePath": "text_encoder"
            },
            {
                "component": "vae",
                "repository": WAN21_T2V_REPOSITORY,
                "revision": WAN21_T2V_REVISION,
                "sourcePath": "vae"
            },
            {
                "component": "tokenizer",
                "repository": UMT5_REPOSITORY,
                "revision": UMT5_REVISION,
                "sourcePath": "tokenizer.json"
            }
        ]
    })
}

pub(super) fn vace_fun_source_receipt(backend: WanI2vBackend) -> serde_json::Value {
    serde_json::json!({
        "version": VACE_FUN_SOURCE_RECEIPT_VERSION,
        "backend": backend.key(),
        "components": [
            {"component":"high_expert","repository":VACE_FUN_REPOSITORY,"revision":VACE_FUN_REVISION,"sourcePath":"transformer"},
            {"component":"low_expert","repository":VACE_FUN_REPOSITORY,"revision":VACE_FUN_REVISION,"sourcePath":"transformer_2"},
            {"component":"t5_encoder","repository":WAN21_T2V_REPOSITORY,"revision":WAN21_T2V_REVISION,"sourcePath":"text_encoder"},
            {"component":"vae","repository":WAN21_T2V_REPOSITORY,"revision":WAN21_T2V_REVISION,"sourcePath":"vae"},
            {"component":"tokenizer","repository":UMT5_REPOSITORY,"revision":UMT5_REVISION,"sourcePath":"tokenizer.json"}
        ]
    })
}

/// Materialize the fixed provenance receipt once when the worker assembles or resolves the
/// artifact.  A pre-existing divergent receipt is a supply-chain error, never a value to replace.
pub fn ensure_wan_vace_source_receipt(root: &Path, backend: WanI2vBackend) -> gen_core::Result<()> {
    let expected = serde_json::to_vec_pretty(&vace_source_receipt(backend)).map_err(|error| {
        gen_core::Error::Unsupported(format!("wan_vace: cannot encode source receipt: {error}"))
    })?;
    let path = root.join(VACE_SOURCE_RECEIPT_FILE);
    match std::fs::read(&path) {
        Ok(actual) if actual == expected => Ok(()),
        Ok(_) => Err(gen_core::Error::Unsupported(format!(
            "wan_vace: immutable source receipt at {} does not match the selected backend",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::write(&path, expected)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

pub fn ensure_wan_vace_fun_source_receipt(
    root: &Path,
    backend: WanI2vBackend,
) -> gen_core::Result<()> {
    let expected =
        serde_json::to_vec_pretty(&vace_fun_source_receipt(backend)).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "wan2_2_vace_fun_14b: cannot encode source receipt: {error}"
            ))
        })?;
    let path = root.join(VACE_FUN_SOURCE_RECEIPT_FILE);
    match std::fs::read(&path) {
        Ok(actual) if actual == expected => Ok(()),
        Ok(_) => Err(gen_core::Error::Unsupported(format!(
            "wan2_2_vace_fun_14b: immutable source receipt at {} does not match the selected backend",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::write(&path, expected)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn vace_source_receipt_revision(
    root: &Path,
    backend: WanI2vBackend,
) -> gen_core::Result<String> {
    let path = root.join(VACE_SOURCE_RECEIPT_FILE);
    let actual: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "wan_vace: invalid source receipt {}: {error}",
                path.display()
            ))
        })?;
    if actual != vace_source_receipt(backend) {
        return Err(gen_core::Error::Unsupported(format!(
            "wan_vace: source receipt {} does not name the exact immutable transformer/T5/VAE/tokenizer origins",
            path.display()
        )));
    }
    sha256_file(&path)
}

pub(super) fn vace_fun_source_receipt_revision(
    root: &Path,
    backend: WanI2vBackend,
) -> gen_core::Result<String> {
    let path = root.join(VACE_FUN_SOURCE_RECEIPT_FILE);
    let actual: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "wan2_2_vace_fun_14b: invalid source receipt {}: {error}",
                path.display()
            ))
        })?;
    if actual != vace_fun_source_receipt(backend) {
        return Err(gen_core::Error::Unsupported(format!(
            "wan2_2_vace_fun_14b: source receipt {} does not name the exact immutable experts/T5/VAE/tokenizer origins",
            path.display()
        )));
    }
    sha256_file(&path)
}
