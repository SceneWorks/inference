//! LTX **checkpoint-layout selection and per-component resolution** for the MLX engine (sc-18757).
//!
//! The shared, tensor-free machinery — version parsing, component classification, per-component
//! config isolation, and the `gemma_source_checkpoint` ⇄ text-encoder `gemma_version` assertion —
//! lives in [`mlx_gen::gen_core::ltx_checkpoint`] so mlx-gen-ltx and candle-gen-ltx behave
//! identically. This module is the MLX-side adapter: it turns a [`LoadSpec`] into a resolved bundle
//! and answers "which layout is this model directory?" **from `model_version`**, never from a file
//! name.
//!
//! # Where the version comes from
//!
//! * A SceneWorks-converted LTX-2.3 tree (`ltx_2_3_base_q4/`, the [`crate::convert`] output) carries
//!   its version in the `split_model.json` manifest (`"model_version": "2.3.0"`). Its per-component
//!   `.safetensors` are re-emitted by the converter and carry no `__metadata__`, so the manifest is
//!   the only declaration.
//! * An LTX-2.5 bundle carries `__metadata__["model_version"] = "2.5.0"` on **every** component file
//!   and ships no manifest.
//! * A raw upstream LTX-2.3 checkpoint carries `__metadata__["model_version"]` on its one flat file.
//!
//! [`declared_model_version`] reads them in that order and stops at the first declaration, so the
//! answer never depends on which components happen to be present.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::ltx_checkpoint::{
    GemmaEncoderIdentity, GemmaVersionCheck, LtxBundle, LtxBundleBuilder, LtxCheckpointLayout,
    LtxComponent,
};
use mlx_gen::gen_core::{LoadSpec, WeightsSource};
use mlx_gen::Result;

/// The `model_version` a model directory (or single checkpoint file) declares — the
/// `split_model.json` manifest first, then any component file's `__metadata__`. `None` when nothing
/// there declares one.
pub fn declared_model_version(root: &Path) -> Result<Option<String>> {
    Ok(mlx_gen::gen_core::ltx_checkpoint::declared_model_version(
        root,
    )?)
}

/// The layout a model directory declares. An undeclared version is
/// [`LtxCheckpointLayout::AllInOne`] — the oldest layout, matching upstream's fallback — which keeps
/// every pre-`model_version` LTX-2.3 tree on exactly the path it has always taken.
pub fn declared_layout(root: &Path) -> Result<LtxCheckpointLayout> {
    Ok(mlx_gen::gen_core::ltx_checkpoint::declared_layout(root)?)
}

/// The [`LoadSpec::components`] ids an LTX-2.5 load recognizes: one per split component, plus the
/// text encoder, which may instead ride the typed [`LoadSpec::text_encoder`] slot.
pub fn split_component_ids() -> Vec<&'static str> {
    LtxComponent::ALL.iter().map(|c| c.id()).collect()
}

/// Resolve an LTX-2.5 split bundle from a [`LoadSpec`].
///
/// Each component is resolved **independently**, in this order of authority:
///
/// 1. an explicit [`LoadSpec::components`] entry under the component's id (the caller-staged route —
///    SceneWorks resolves every path before load; there is no environment side-channel);
/// 2. for the text encoder only, the typed [`LoadSpec::text_encoder`] slot;
/// 3. otherwise, discovery under the bundle root by **classifying each file's own metadata** — never
///    by its name.
///
/// A component that none of the three produce is absent, and [`LtxBundle::require`] then reports it
/// by name along with every path searched.
pub fn resolve_split_bundle(spec: &LoadSpec) -> Result<LtxBundle> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };

    // Collect the caller's explicit choices first. Discovery then SKIPS those slots: a bundle that
    // ships two plausible candidates for a component the caller already picked must not be refused
    // as ambiguous before that pick is applied.
    let mut explicit: Vec<(LtxComponent, PathBuf)> = Vec::new();
    for component in LtxComponent::ALL {
        if let Some(source) = spec.components.get(component.id()) {
            let (WeightsSource::Dir(p) | WeightsSource::File(p)) = source;
            explicit.push((*component, p.clone()));
        }
    }
    // The text encoder is not bundled with the diffusion weights in either LTX generation, so the
    // typed slot is its canonical home; an explicit `text_encoder` component entry still wins.
    if !spec.components.contains_key(LtxComponent::TextEncoder.id()) {
        if let Some(WeightsSource::Dir(p) | WeightsSource::File(p)) = spec.text_encoder.as_ref() {
            explicit.push((LtxComponent::TextEncoder, p.clone()));
        }
    }
    let skip: Vec<LtxComponent> = explicit.iter().map(|(c, _)| *c).collect();

    let discovered =
        mlx_gen::gen_core::ltx_checkpoint::discover_split_bundle_skipping(&root, &skip)?;
    let mut builder = LtxBundleBuilder::new();
    for resolved in discovered.components() {
        builder = builder.with_component(resolved.component(), resolved.path());
    }
    for path in discovered.searched() {
        builder = builder.with_searched(path);
    }
    for (component, path) in explicit {
        builder = builder.with_component(component, path);
    }

    Ok(builder.build()?)
}

/// Read the text encoder's declared identity (`model_type` + `gemma_version`) from whichever layout
/// it ships as: a packed single-file encoder (LTX-2.5) or an HF snapshot directory (LTX-2.3).
pub fn text_encoder_identity(bundle: &LtxBundle) -> Result<GemmaEncoderIdentity> {
    let te = bundle.require(LtxComponent::TextEncoder)?;
    Ok(GemmaEncoderIdentity::load(te.path())?)
}

/// Assert the bundle's transformer and its text encoder agree on the Gemma generation.
///
/// A mismatch is a hard, message-bearing error — upstream `_check_gemma_version` raises, and a
/// warning-and-continue would run an LTX-2.5 DiT against Gemma-3 embeddings and emit plausible
/// garbage rather than failing.
pub fn assert_gemma_version(bundle: &LtxBundle) -> Result<GemmaVersionCheck> {
    let encoder = text_encoder_identity(bundle)?;
    Ok(bundle.check_gemma_version(&encoder)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_safetensors(path: &Path, metadata: &[(&str, &str)]) {
        let meta_json: String = metadata
            .iter()
            .map(|(k, v)| format!("{}:{}", serde_json::json!(k), serde_json::json!(v)))
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"__metadata__":{{{meta_json}}},"w":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// The SceneWorks-converted LTX-2.3 tree: a `split_model.json` manifest beside per-component
    /// files that carry no `__metadata__` of their own.
    fn write_converted_2_3(root: &Path) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join(mlx_gen::gen_core::ltx_checkpoint::SPLIT_MANIFEST_FILE),
            r#"{"format":"split","model_version":"2.3.0","variant":"distilled","quantized":true,
                "quantization_bits":4,"quantization_group_size":64}"#,
        )
        .unwrap();
        for name in ["transformer", "connector", "vae_decoder", "audio_vae"] {
            write_safetensors(&root.join(format!("{name}.safetensors")), &[]);
        }
    }

    fn write_2_5_bundle(root: &Path) {
        write_safetensors(
            &root.join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            &[
                ("model_version", "2.5.0"),
                (
                    "gemma_source_checkpoint",
                    r#"{"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}"#,
                ),
                (
                    "config",
                    r#"{"transformer":{"_class_name":"AVTransformer3DModel","num_layers":48},"vae":null,"audio_vae":null,"vocoder":null}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            &[
                ("model_version", "2.5.0"),
                (
                    "config",
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128,"patch_size":4}}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
            &[
                ("model_version", "2.5.0"),
                (
                    "config",
                    r#"{"audio_vae":{"model":{"params":{"ddconfig":{"ch":128,"z_channels":8}}}},"vocoder":{"vocoder":{"resblock":"AMP1"}}}"#,
                ),
            ],
        );
    }

    fn packed_text_encoder(path: &Path, gemma_version: &str, model_type: &str) {
        write_safetensors(
            path,
            &[
                ("model_version", "2.5.0"),
                (
                    "gemma_config",
                    &format!(
                        r#"{{"model_type":"{model_type}","gemma_version":"{gemma_version}"}}"#
                    ),
                ),
            ],
        );
    }

    #[test]
    fn a_converted_2_3_tree_declares_the_all_in_one_layout_from_its_manifest() {
        let dir = tempfile::tempdir().unwrap();
        write_converted_2_3(dir.path());
        assert_eq!(
            declared_model_version(dir.path()).unwrap().as_deref(),
            Some("2.3.0")
        );
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn a_2_5_bundle_declares_the_split_layout_from_its_component_metadata() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        assert_eq!(
            declared_model_version(dir.path()).unwrap().as_deref(),
            Some("2.5.0")
        );
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
    }

    #[test]
    fn an_undeclared_tree_stays_on_the_all_in_one_path() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("ltx-2.3-22b-distilled.safetensors"), &[]);
        assert_eq!(declared_model_version(dir.path()).unwrap(), None);
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn the_layout_does_not_change_when_a_component_is_removed() {
        // Selection is keyed on `model_version`, not on which files exist: deleting the audio VAE
        // must not demote a 2.5 bundle to the 2.3 path.
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        std::fs::remove_file(dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors")).unwrap();
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
    }

    #[test]
    fn resolution_discovers_components_and_honours_explicit_overrides() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        let te = dir.path().join("text_encoders/gemma4-12b.safetensors");
        packed_text_encoder(&te, "gemma4-12b-ltx-v1", "gemma4_unified");

        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let bundle = resolve_split_bundle(&spec).unwrap();
        assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
        // Discovered by metadata classification, not by name.
        assert!(bundle
            .require(LtxComponent::Transformer)
            .unwrap()
            .path()
            .ends_with("ltx-2.5-22b-distilled-transformer-bf16.safetensors"));
        assert!(bundle.require(LtxComponent::AudioVae).is_ok());
        assert!(bundle.require(LtxComponent::TextEncoder).is_ok());

        // An explicit component entry overrides discovery.
        let other = dir.path().join("alt/te.safetensors");
        packed_text_encoder(&other, "gemma4-12b-ltx-v1", "gemma4_unified");
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()))
            .with_component("text_encoder", WeightsSource::File(other.clone()));
        let bundle = resolve_split_bundle(&spec).unwrap();
        assert_eq!(
            bundle.require(LtxComponent::TextEncoder).unwrap().path(),
            other
        );
    }

    #[test]
    fn the_typed_text_encoder_slot_fills_the_text_encoder_component() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        let te = dir.path().join("external/te.safetensors");
        packed_text_encoder(&te, "gemma4-12b-ltx-v1", "gemma4_unified");
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.text_encoder = Some(WeightsSource::File(te.clone()));
        let bundle = resolve_split_bundle(&spec).unwrap();
        assert_eq!(
            bundle.require(LtxComponent::TextEncoder).unwrap().path(),
            te
        );
        assert!(matches!(
            assert_gemma_version(&bundle).unwrap(),
            GemmaVersionCheck::Matched(_)
        ));
    }

    #[test]
    fn a_2_5_bundle_with_a_gemma_3_text_encoder_fails_the_version_assertion() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        // The LTX-2.3 Gemma-3-12B snapshot directory, pointed at a 2.5 bundle.
        let gemma3 = dir.path().join("gemma-3-12b-it");
        std::fs::create_dir_all(&gemma3).unwrap();
        std::fs::write(
            gemma3.join("config.json"),
            r#"{"model_type":"gemma3","text_config":{"hidden_size":3840}}"#,
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.text_encoder = Some(WeightsSource::Dir(gemma3));
        let bundle = resolve_split_bundle(&spec).unwrap();
        let err = assert_gemma_version(&bundle).expect_err("Gemma 3 cannot serve LTX-2.5");
        let text = err.to_string();
        assert!(text.contains("Gemma version mismatch"), "{text}");
        assert!(text.contains("gemma4-12b-ltx-v1"), "{text}");
    }

    #[test]
    fn a_missing_component_names_itself_and_the_paths_searched() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let bundle = resolve_split_bundle(&spec).unwrap();
        let err = bundle
            .require(LtxComponent::DurationHead)
            .expect_err("this bundle ships no duration head");
        let text = err.to_string();
        assert!(text.contains("duration_head"), "{text}");
        assert!(text.contains("searched:"), "{text}");
        assert!(
            text.contains("ltx-2.5-video-vae-conv-bf16.safetensors"),
            "{text}"
        );
    }

    #[test]
    fn split_component_ids_cover_every_component() {
        let ids = split_component_ids();
        assert_eq!(ids.len(), LtxComponent::ALL.len());
        assert!(ids.contains(&"transformer"));
        assert!(ids.contains(&"temporal_upsampler"));
    }
}
