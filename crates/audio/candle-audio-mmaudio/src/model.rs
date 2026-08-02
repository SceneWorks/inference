//! Pinned-checkpoint resolution, weight-license surface, and the load entry point for the
//! Synchformer visual encoder.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core::WeightsSource;
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

/// The schema-3 licence row for the pinned Synchformer visual-encoder checkpoint (sc-16663).
///
/// **Disclosure only.** The Synchformer repository declares MIT, © 2024 Vladimir Iashin, read from
/// the GitHub licence API on `retrieved`; the checkpoint ships beside that MIT code with no separate
/// weights licence, and is distributed via MMAudio's `ext_weights`.
///
/// The v2 row additionally carried a prose note about the training-data provenance
/// (VGGSound/AudioSet/LRS3). That is a fact about datasets, not a term the MIT text states, and this
/// surface transcribes licence texts — so it is not re-encoded as a term here. The note lives in this
/// doc comment, where it informs a reader without pretending to be a joinable obligation.
pub const COMPONENT_LICENSE: candle_audio::gen_core::ComponentLicense =
    candle_audio::gen_core::ComponentLicense {
        component: MODEL_ID,
        source_url: "https://github.com/v-iashin/Synchformer",
        gated: false,
        declared: "mit",
        family: "mit",
        attribution: Some(
            "Synchformer © 2024 Vladimir Iashin — MIT License; checkpoint distributed via MMAudio \
             (hkchengrex/MMAudio) ext_weights",
        ),
        retrieved: "2026-08-02",
    };

/// Load the Synchformer visual encoder from a `synchformer_state_dict.pth` file path.
///
/// The `.pth` holds the full Synchformer state dict (visual + audio branches + the AV sync
/// transformer); MMAudio keeps only the `vfeat_extractor.*` keys, so we root the `VarBuilder` there
/// and ignore the rest. Weights load as f32 (the encoder is deterministic and CPU-first).
pub fn load_from_pth(weights: &Path, device: &Device) -> Result<SynchformerVisualEncoder> {
    if !weights.exists() {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID}: weights file {} not found (pass {WEIGHTS_PATH} in via the LoadSpec)",
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
    fn component_licence_resolves_to_the_mit_family() {
        use candle_audio::gen_core::{resolve_family, LICENSE_FAMILIES};
        assert!(COMPONENT_LICENSE.is_well_formed(LICENSE_FAMILIES));
        assert_eq!(COMPONENT_LICENSE.component, MODEL_ID);
        assert_eq!(COMPONENT_LICENSE.declared, "mit");
        let family = resolve_family(LICENSE_FAMILIES, COMPONENT_LICENSE.family).unwrap();
        assert_eq!(family.spdx_id, "MIT");
        // Both MMAudio providers load this one artifact, so it is one row, not two (sc-16663).
        assert_eq!(
            crate::PROVIDER_COMPONENTS
                .iter()
                .filter(|p| p.components.contains(&MODEL_ID))
                .count(),
            2
        );
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
