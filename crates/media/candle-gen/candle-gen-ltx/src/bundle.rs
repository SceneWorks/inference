//! LTX **checkpoint-layout selection and per-component resolution** for the candle engine
//! (sc-18757) — the mirror of `mlx-gen-ltx::bundle`.
//!
//! The shared, tensor-free machinery lives in [`gen_core::ltx_checkpoint`]: version parsing,
//! component classification, per-component config isolation, and the `gemma_source_checkpoint` ⇄
//! text-encoder `gemma_version` assertion. Both engines resolve a bundle through it, so a checkpoint
//! that loads on MLX resolves identically on candle and a version mismatch is refused on both.
//!
//! This module is the candle-side adapter: [`LoadSpec`] → resolved bundle, and "which layout is
//! this?" answered **from `model_version`** rather than from a file name or from which files happen
//! to be present.
//!
//! Candle's LTX-2.3 route accepts either a directory (the dense snapshot, or a packed MLX tier) or a
//! single `.safetensors`; [`declared_model_version`] handles both, so the gate is the same shape on
//! either input.

use std::fs;
use std::path::{Component, Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::gen_core::ltx_checkpoint::{
    GemmaEncoderIdentity, GemmaVersionCheck, LtxBundle, LtxBundleBuilder, LtxCheckpointLayout,
    LtxComponent,
};
use candle_gen::gen_core::{self, LoadSpec, WeightsSource};
use safetensors::Dtype as SafetensorDtype;

/// The `model_version` a checkpoint location declares — the `split_model.json` manifest of a packed
/// MLX tier first, then any component file's `__metadata__`. `None` when nothing declares one.
pub fn declared_model_version(root: &Path) -> gen_core::Result<Option<String>> {
    gen_core::ltx_checkpoint::declared_model_version(root)
}

/// The layout a checkpoint location declares. An undeclared version is
/// [`LtxCheckpointLayout::AllInOne`] — the oldest layout, matching upstream's fallback — so every
/// pre-`model_version` LTX-2.3 checkpoint stays on exactly the path it has always taken.
pub fn declared_layout(root: &Path) -> gen_core::Result<LtxCheckpointLayout> {
    gen_core::ltx_checkpoint::declared_layout(root)
}

/// The [`LoadSpec::components`] ids an LTX-2.5 load recognizes: one per split component. The text
/// encoder may instead ride the typed [`LoadSpec::text_encoder`] slot.
pub fn split_component_ids() -> Vec<&'static str> {
    LtxComponent::ALL.iter().map(|c| c.id()).collect()
}

/// Resolve an LTX-2.5 split bundle from a [`LoadSpec`].
///
/// Each component resolves **independently**, in this order of authority:
///
/// 1. an explicit [`LoadSpec::components`] entry under the component's id (the caller-staged route);
/// 2. for the text encoder only, the typed [`LoadSpec::text_encoder`] slot;
/// 3. otherwise, discovery under the bundle root by **classifying each file's own metadata**.
///
/// A component none of the three produce is absent, and [`LtxBundle::require`] reports it by name
/// together with every path searched.
pub fn resolve_split_bundle(spec: &LoadSpec) -> gen_core::Result<LtxBundle> {
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
    if !spec.components.contains_key(LtxComponent::TextEncoder.id()) {
        if let Some(WeightsSource::Dir(p) | WeightsSource::File(p)) = spec.text_encoder.as_ref() {
            explicit.push((LtxComponent::TextEncoder, p.clone()));
        }
    }
    let skip: Vec<LtxComponent> = explicit.iter().map(|(c, _)| *c).collect();

    let discovered = gen_core::ltx_checkpoint::discover_split_bundle_skipping(&root, &skip)?;
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

    builder.build()
}

/// Bind an explicitly selected nested bundle to its complete immutable snapshot.
///
/// Discovery is confined to `bundle_subdir`, then every discovered component is copied into an
/// explicit [`LoadSpec::components`] slot while [`LoadSpec::weights`] remains the full snapshot
/// root. This is the terminal campaign and production-replay shape: component choice is narrow,
/// but inventory and runtime provenance cover the complete public/upstream snapshot.
pub fn select_snapshot_bundle(
    snapshot_root: &Path,
    bundle_subdir: &Path,
) -> gen_core::Result<(LoadSpec, String)> {
    if bundle_subdir.is_absolute()
        || bundle_subdir.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(gen_core::Error::Unsupported(format!(
            "LTX-2.5 bundle subdir {} must be relative and cannot contain parent traversal",
            bundle_subdir.display()
        )));
    }
    let snapshot_root = fs::canonicalize(snapshot_root).map_err(|error| {
        gen_core::Error::Msg(format!(
            "canonicalize LTX-2.5 snapshot {}: {error}",
            snapshot_root.display()
        ))
    })?;
    let bundle_root = fs::canonicalize(snapshot_root.join(bundle_subdir)).map_err(|error| {
        gen_core::Error::Msg(format!(
            "canonicalize selected LTX-2.5 bundle {}: {error}",
            snapshot_root.join(bundle_subdir).display()
        ))
    })?;
    if !bundle_root.is_dir() || !bundle_root.starts_with(&snapshot_root) {
        return Err(gen_core::Error::Unsupported(format!(
            "selected LTX-2.5 bundle {} must be a directory inside snapshot {}",
            bundle_root.display(),
            snapshot_root.display()
        )));
    }
    let logical = bundle_root
        .strip_prefix(&snapshot_root)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?
        .to_string_lossy()
        .replace('\\', "/");
    let logical = if logical.is_empty() {
        ".".to_owned()
    } else {
        logical
    };

    let discovered = resolve_split_bundle(&LoadSpec::new(WeightsSource::Dir(bundle_root)))?;
    let mut selected = LoadSpec::new(WeightsSource::Dir(snapshot_root));
    for component in discovered.components() {
        let path = component.path().to_path_buf();
        let source = if path.is_dir() {
            WeightsSource::Dir(path)
        } else {
            WeightsSource::File(path)
        };
        selected = selected.with_component(component.component().id(), source);
    }
    Ok((selected, logical))
}

/// Read the text encoder's declared identity (`model_type` + `gemma_version`) from whichever layout
/// it ships as: a packed single-file encoder (LTX-2.5) or an HF snapshot directory (LTX-2.3).
pub fn text_encoder_identity(bundle: &LtxBundle) -> gen_core::Result<GemmaEncoderIdentity> {
    GemmaEncoderIdentity::load(bundle.require(LtxComponent::TextEncoder)?.path())
}

/// Assert the bundle's transformer and its text encoder agree on the Gemma generation. A mismatch is
/// a hard, message-bearing error — upstream `_check_gemma_version` raises, and a warning-and-continue
/// would run an LTX-2.5 DiT on Gemma-3 embeddings and emit plausible garbage instead of failing.
pub fn assert_gemma_version(bundle: &LtxBundle) -> gen_core::Result<GemmaVersionCheck> {
    let encoder = text_encoder_identity(bundle)?;
    bundle.check_gemma_version(&encoder)
}

/// Require the advanced CUDA lane's explicitly selected upstream Gemma-4 text encoder to be a
/// genuinely dense BF16 safetensors source. Comfy's I8 encoder must never be interpreted through the
/// Candle dense path merely because it declares the same Gemma generation.
pub fn assert_bf16_text_encoder(bundle: &LtxBundle) -> gen_core::Result<()> {
    let path = bundle.require(LtxComponent::TextEncoder)?.path();
    if !path.is_file() {
        return Err(gen_core::Error::Unsupported(format!(
            "LTX-2.5 advanced text encoder {} must be one explicit BF16 safetensors file",
            path.display()
        )));
    }
    // SAFETY: read-only safetensors header inspection; no tensor/device allocation occurs.
    let safetensors = unsafe { MmapedSafetensors::new(path) }
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
    let tensors = safetensors.tensors();
    if tensors.is_empty()
        || tensors
            .iter()
            .any(|(_, tensor)| tensor.dtype() != SafetensorDtype::BF16)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "LTX-2.5 advanced text encoder {} is not an all-BF16 upstream Gemma source; Comfy/I8 or mixed sources are forbidden",
            path.display()
        )));
    }
    Ok(())
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
                    r#"{"transformer":{"_class_name":"AVTransformer3DModel","num_layers":48,"num_attention_heads":32,"attention_head_dim":128},"vae":null,"audio_vae":null,"vocoder":null}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            &[
                ("model_version", "2.5.0"),
                (
                    "config",
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128}}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
            &[
                ("model_version", "2.5.0"),
                (
                    "config",
                    r#"{"audio_vae":{"model":{"params":{"ddconfig":{"ch":128,"z_channels":8}}}},"vocoder":{"vocoder":{"resblock":"AMP1","activation":"snakebeta"}}}"#,
                ),
            ],
        );
    }

    fn packed_text_encoder(path: &Path, model_type: &str, gemma_version: &str) {
        write_safetensors(
            path,
            &[
                // Ground truth (sc-18756): a packed TE declares no `model_version`.
                ("format", "pt"),
                (
                    "gemma_config",
                    &format!(
                        r#"{{"model_type":"{model_type}","gemma_version":"{gemma_version}"}}"#
                    ),
                ),
            ],
        );
    }

    fn write_typed_safetensors(path: &Path, dtype: &str, payload: &[u8]) {
        let header = format!(
            r#"{{"weight":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_dense_2_3_checkpoint_declares_the_all_in_one_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ltx-2.3-22b-distilled.safetensors");
        write_safetensors(
            &path,
            &[
                ("model_version", "2.3.0"),
                ("config", r#"{"transformer":{},"vae":{},"audio_vae":{}}"#),
            ],
        );
        assert_eq!(
            declared_layout(&path).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn an_unversioned_checkpoint_stays_on_the_all_in_one_path() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("10Eros_v1_bf16.safetensors"), &[]);
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn nested_bundle_selection_keeps_the_full_snapshot_as_weights_and_explicitly_pins_components() {
        let snapshot = tempfile::tempdir().unwrap();
        let nested = snapshot.path().join("packages/distilled-bf16");
        write_2_5_bundle(&nested);
        std::fs::write(snapshot.path().join("upstream-readme.md"), b"full snapshot").unwrap();

        let (spec, logical) =
            select_snapshot_bundle(snapshot.path(), Path::new("packages/distilled-bf16")).unwrap();
        assert_eq!(logical, "packages/distilled-bf16");
        assert_eq!(
            spec.weights,
            WeightsSource::Dir(std::fs::canonicalize(snapshot.path()).unwrap())
        );
        assert!(!spec.components.is_empty());
        let nested = std::fs::canonicalize(nested).unwrap();
        assert!(spec.components.values().all(|source| {
            let (WeightsSource::Dir(path) | WeightsSource::File(path)) = source;
            path.starts_with(&nested)
        }));
    }

    #[test]
    fn nested_bundle_selection_refuses_parent_traversal() {
        let snapshot = tempfile::tempdir().unwrap();
        let error = select_snapshot_bundle(snapshot.path(), Path::new("../outside"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("parent traversal"), "{error}");
    }

    #[test]
    fn selected_bundle_digest_allows_unrelated_snapshot_files_but_rejects_component_mutation() {
        use crate::dev_sampler::TransformerVariant;
        use crate::quant_eval::{
            inventory_for_snapshot, selected_bundle_identity_sha256, Ltx25QuantMode,
        };

        let source = tempfile::tempdir().unwrap();
        let public = tempfile::tempdir().unwrap();
        for root in [source.path(), public.path()] {
            write_2_5_bundle(&root.join("packages/distilled-bf16"));
        }
        std::fs::write(source.path().join("source-only.txt"), b"upstream").unwrap();
        std::fs::write(public.path().join("public-card.md"), b"SceneWorks").unwrap();
        let (source_spec, source_subdir) =
            select_snapshot_bundle(source.path(), Path::new("packages/distilled-bf16")).unwrap();
        let (public_spec, public_subdir) =
            select_snapshot_bundle(public.path(), Path::new("packages/distilled-bf16")).unwrap();
        let source_bundle = resolve_split_bundle(&source_spec).unwrap();
        let public_bundle = resolve_split_bundle(&public_spec).unwrap();
        let source_root = std::fs::canonicalize(source.path()).unwrap();
        let public_root = std::fs::canonicalize(public.path()).unwrap();
        let source_inventory = inventory_for_snapshot(&source_root).unwrap();
        let public_inventory = inventory_for_snapshot(&public_root).unwrap();
        let source_digest = selected_bundle_identity_sha256(
            &source_bundle,
            &source_root,
            &source_subdir,
            &source_inventory,
            TransformerVariant::Distilled,
            Ltx25QuantMode::Bf16,
        )
        .unwrap();
        let public_digest = selected_bundle_identity_sha256(
            &public_bundle,
            &public_root,
            &public_subdir,
            &public_inventory,
            TransformerVariant::Distilled,
            Ltx25QuantMode::Bf16,
        )
        .unwrap();
        assert_eq!(source_digest, public_digest);

        let transformer = public
            .path()
            .join("packages/distilled-bf16/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors");
        let mut bytes = std::fs::read(&transformer).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&transformer, bytes).unwrap();
        let mutated_inventory = inventory_for_snapshot(&public_root).unwrap();
        let mutated = selected_bundle_identity_sha256(
            &public_bundle,
            &public_root,
            &public_subdir,
            &mutated_inventory,
            TransformerVariant::Distilled,
            Ltx25QuantMode::Bf16,
        )
        .unwrap();
        assert_ne!(source_digest, mutated);

        let renamed = transformer.with_file_name("renamed-transformer.safetensors");
        std::fs::rename(&transformer, renamed).unwrap();
        let renamed_inventory = inventory_for_snapshot(&public_root).unwrap();
        assert!(selected_bundle_identity_sha256(
            &public_bundle,
            &public_root,
            &public_subdir,
            &renamed_inventory,
            TransformerVariant::Distilled,
            Ltx25QuantMode::Bf16,
        )
        .is_err());
    }

    #[test]
    fn a_2_5_bundle_declares_the_split_layout_and_keeps_it_when_a_component_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
        std::fs::remove_file(dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors")).unwrap();
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
    }

    #[test]
    fn resolution_finds_components_by_metadata_and_honours_the_typed_te_slot() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
        let te = dir.path().join("external/gemma4-12b.safetensors");
        packed_text_encoder(&te, "gemma4_unified", "gemma4-12b-ltx-v1");

        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.text_encoder = Some(WeightsSource::File(te.clone()));
        let bundle = resolve_split_bundle(&spec).unwrap();
        assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
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
    fn advanced_text_encoder_accepts_only_explicit_all_bf16_upstream_weights() {
        let dir = tempfile::tempdir().unwrap();
        let bf16 = dir.path().join("upstream-gemma4-bf16.safetensors");
        write_typed_safetensors(&bf16, "BF16", &[0, 0]);
        let bundle = LtxBundleBuilder::new()
            .with_component(LtxComponent::TextEncoder, bf16)
            .build()
            .unwrap();
        assert_bf16_text_encoder(&bundle).unwrap();

        let int8 = dir.path().join("comfy-gemma4-int8.safetensors");
        write_typed_safetensors(&int8, "I8", &[0]);
        let bundle = LtxBundleBuilder::new()
            .with_component(LtxComponent::TextEncoder, int8)
            .build()
            .unwrap();
        let error = assert_bf16_text_encoder(&bundle)
            .expect_err("the Comfy I8 text encoder must fail before device loading");
        assert!(error.to_string().contains("Comfy/I8"));
    }

    #[test]
    fn a_2_5_bundle_with_a_gemma_3_text_encoder_fails_the_version_assertion() {
        let dir = tempfile::tempdir().unwrap();
        write_2_5_bundle(dir.path());
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
            .require(LtxComponent::SpatialUpsampler)
            .expect_err("this bundle ships no upsampler");
        let text = err.to_string();
        assert!(text.contains("spatial_upsampler"), "{text}");
        assert!(text.contains("searched:"), "{text}");
    }
}
