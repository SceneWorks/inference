//! Pinned-checkpoint resolution, weight-license surface, and the load entry point for the
//! Synchformer visual encoder.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core::WeightsSource;
use candle_audio::hub::hf_get_pinned;
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

use crate::sync::SynchformerVisualEncoder;

/// Stable identity of the encoder (used for the weight-license entry key). Not a shipping provider
/// id — this crate registers nothing this slice.
pub const MODEL_ID: &str = "synchformer_vfeat";

/// Hub pin: MMAudio's model repo, which mirrors the GitHub-release `ext_weights/`. Immutable commit
/// SHA (F-029 discipline). The Synchformer checkpoint is `ext_weights/synchformer_state_dict.pth`.
pub const HUB_REPO: &str = "hkchengrex/MMAudio";
pub const HUB_REVISION: &str = "eb13a1a98fdbec91753775c57b074ccdfc60587c";
/// The Synchformer visual-encoder state dict (~907 MB pickle) inside the pinned repo.
pub const WEIGHTS_PATH: &str = "ext_weights/synchformer_state_dict.pth";

/// The license of the pinned Synchformer weight checkpoint (sc-13332 framework) — surfaced for
/// SceneWorks' end-product licenses page.
///
/// The Synchformer **code** (`v-iashin/Synchformer`, vendored under MMAudio's
/// `mmaudio/ext/synchformer/LICENSE`) is **MIT**, © 2024 Vladimir Iashin — verified against the
/// repository `LICENSE` file. The released checkpoints ship alongside the MIT code with no separate
/// weights license; the restriction note records the training-data provenance
/// (VGGSound/AudioSet/LRS3) whose dataset terms a downstream *commercial* use would need a legal
/// read on, so the fact is surfaced rather than buried even though MIT itself is permissive.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "MIT",
        name: "MIT License",
        source_url: "https://github.com/v-iashin/Synchformer",
        attribution: Some(
            "Synchformer © 2024 Vladimir Iashin — MIT License; checkpoint distributed via MMAudio \
             (hkchengrex/MMAudio) ext_weights",
        ),
        commercial_use: true,
        restriction: Some(
            "Code is MIT. Released weights were trained on VGGSound/AudioSet/LRS3; those datasets \
             carry their own (YouTube/BBC-sourced, research-oriented) terms — a legal read is \
             warranted before any commercial redistribution of the weights.",
        ),
    };

/// This encoder's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation once a
/// shipping MMAudio generator registers it in a later slice.
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        license: WEIGHT_LICENSE,
    };

/// Resolve the pinned Synchformer checkpoint file through the audio lane's F-029 hub path.
/// Returns a [`WeightsSource::File`] pointing at the resolved `synchformer_state_dict.pth`.
pub fn resolve_pinned_weights() -> Result<WeightsSource> {
    Ok(WeightsSource::File(hf_get_pinned(
        HUB_REPO,
        HUB_REVISION,
        WEIGHTS_PATH,
    )?))
}

/// Load the Synchformer visual encoder from a `synchformer_state_dict.pth` file path.
///
/// The `.pth` holds the full Synchformer state dict (visual + audio branches + the AV sync
/// transformer); MMAudio keeps only the `vfeat_extractor.*` keys, so we root the `VarBuilder` there
/// and ignore the rest. Weights load as f32 (the encoder is deterministic and CPU-first).
pub fn load_from_pth(weights: &Path, device: &Device) -> Result<SynchformerVisualEncoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID}: weights file {} not found (resolve_pinned_weights materializes {WEIGHTS_PATH})",
            weights.display()
        )));
    }
    let vb = VarBuilder::from_pth(weights, DType::F32, device).map_err(AudioError::from)?;
    SynchformerVisualEncoder::load(vb.pp("vfeat_extractor"), device.clone())
        .map_err(AudioError::from)
}

/// Load from a [`WeightsSource`] (a `File` path to the `.pth`, or a `Dir` containing it under
/// `ext_weights/` or at its root).
pub fn load(source: &WeightsSource, device: &Device) -> Result<SynchformerVisualEncoder> {
    let path: PathBuf = match source {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => {
            let nested = d.join(WEIGHTS_PATH);
            if nested.exists() {
                nested
            } else {
                d.join("synchformer_state_dict.pth")
            }
        }
    };
    load_from_pth(&path, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_license_is_well_formed_mit() {
        assert!(WEIGHT_LICENSE.is_well_formed());
        assert_eq!(WEIGHT_LICENSE.spdx_id, "MIT");
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, MODEL_ID);
    }

    #[test]
    fn hub_revision_is_a_full_commit_sha() {
        assert_eq!(HUB_REVISION.len(), 40);
        assert!(HUB_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn missing_weights_file_errors_clearly() {
        let dev = candle_audio::candle_core::Device::Cpu;
        let err = match load_from_pth(std::path::Path::new("/nonexistent/synchformer.pth"), &dev) {
            Ok(_) => panic!("loading a nonexistent path must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("not found"));
    }
}
