//! Supra2-IMG converter and tokenizer conformance (sc-24213).
//!
//! The tiny checkpoints under `tests/fixtures/supra2/` were written by `torch.save` from the frozen
//! upstream `SupraDiT` class (`crates/media/mlx-gen/tools/dump_supra2_reference.py tiny`); their
//! per-tensor SHA-256 digests are the upstream truth the converted bundle must reproduce byte-for-byte.
//! The `#[ignore]`d rows run the same checks on the real pinned checkpoint and the real Flan-T5
//! tokenizer, against reference files dumped by the same script in `real` mode.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use gen_core_checkpoint::supra2::*;
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gen-core/tests/fixtures/supra2")
}

fn tiny_config() -> Supra2Config {
    Supra2Config {
        latent_channels: 4,
        latent_size: 16,
        patch: 2,
        d_model: 32,
        depth: 2,
        num_heads: 2,
        mlp_hidden: 128,
        ctx_dim: 24,
        ctx_len: 8,
        freq_dim: 256,
    }
}

fn opts(config: Supra2Config) -> ConvertOptions {
    ConvertOptions {
        config,
        require_pinned_source: false,
    }
}

fn digests(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// SHA-256 of every tensor's raw (F32 LE) bytes in a safetensors file.
fn tensor_digests(path: &Path) -> BTreeMap<String, (Vec<usize>, String)> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}");
            (
                name,
                (
                    view.shape().to_vec(),
                    format!("{:x}", Sha256::digest(view.data())),
                ),
            )
        })
        .collect()
}

fn scratch(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("supra2-{tag}-{}-", std::process::id()))
        .tempdir()
        .unwrap()
}

fn assert_denoiser_matches(bundle: &Supra2Bundle, reference: &serde_json::Value) {
    let converted = tensor_digests(&bundle.denoiser_path());
    let expected = reference["denoiser_f32_sha256"].as_object().unwrap();
    assert_eq!(converted.len(), expected.len());
    assert_eq!(
        converted.len(),
        bundle.manifest.config.expected_tensors().len()
    );
    for (name, shape) in bundle.manifest.config.expected_tensors() {
        let (got_shape, got) = &converted[&name];
        assert_eq!(got_shape, &shape, "{name}");
        assert_eq!(got, expected[&name].as_str().unwrap(), "{name}");
    }
}

#[test]
fn converts_stored_unconditional_checkpoint_byte_exact() {
    let out = scratch("uncond");
    let manifest = convert_checkpoint(
        &fixtures().join("tiny_model_with_uncond.pt"),
        out.path(),
        &opts(tiny_config()),
    )
    .unwrap();
    assert_eq!(manifest.source.weights_key, "model");
    assert_eq!(
        manifest.source.repo, None,
        "a tiny fixture is not the pinned source"
    );
    let bundle = read_bundle(out.path()).unwrap();
    let reference = digests(&fixtures().join("tiny_digests.json"));
    assert_denoiser_matches(&bundle, &reference);
    let cond = tensor_digests(&bundle.conditioning_path().unwrap());
    assert_eq!(cond["uncond_text"].0, vec![8, 24]);
    assert_eq!(cond["uncond_text"].1, reference["uncond_text_f32_sha256"]);
    assert_eq!(cond["uncond_mask"].0, vec![8]);
    assert_eq!(cond["uncond_mask"].1, reference["uncond_mask_f32_sha256"]);
    let vae_revision = SD_VAE.revision;
    assert_eq!(
        manifest
            .components
            .iter()
            .map(|c| (c.component.as_str(), c.revision.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (TEXT_ENCODER_COMPONENT, FLAN_T5_BASE.revision),
            (VAE_COMPONENT, vae_revision)
        ]
    );
}

/// Upstream prefers `ema` over `model`; a checkpoint without stored unconditional embeddings
/// converts without a conditioning file (the pipeline then uses the empty-string fallback).
#[test]
fn prefers_ema_and_omits_absent_conditioning() {
    let out = scratch("ema");
    let manifest = convert_checkpoint(
        &fixtures().join("tiny_ema_without_uncond.pt"),
        out.path(),
        &opts(tiny_config()),
    )
    .unwrap();
    assert_eq!(manifest.source.weights_key, "ema");
    assert!(manifest.conditioning.is_none());
    let bundle = read_bundle(out.path()).unwrap();
    assert!(bundle.conditioning_path().is_none());
    assert!(!out.path().join(CONDITIONING_FILE).exists());
    assert_denoiser_matches(&bundle, &digests(&fixtures().join("tiny_digests.json")));
}

fn rewrite_manifest(dir: &Path, edit: impl FnOnce(&mut BundleManifest)) {
    let path = dir.join(BUNDLE_MANIFEST);
    let mut m: BundleManifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut m);
    std::fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

/// The production read refuses each way a self-consistent bundle can be off-pin: an unpinned
/// source, non-published geometry, component revisions other than the pins, a missing stored
/// conditioning, and converted bytes other than the pinned conversion. The accepting half is
/// `check_pinned_manifest`'s unit test and, on the real checkpoint, `real_checkpoint_converts_byte_exact`.
#[test]
fn pinned_read_refuses_unpinned_source_geometry_and_components() {
    let out = scratch("pinned");
    convert_checkpoint(
        &fixtures().join("tiny_model_with_uncond.pt"),
        out.path(),
        &opts(tiny_config()),
    )
    .unwrap();
    // Self-consistent, so the unpinned reader accepts it…
    read_bundle(out.path()).unwrap();
    // …but the provider's reader names the pin it requires.
    let err = read_pinned_bundle(out.path(), BundleCheck::Digests)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unpinned source"), "{err}");
    assert!(
        err.contains("SupraLabs/Supra2-IMG@10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b"),
        "{err}"
    );

    let pin = SUPRA2_SOURCE.files[0];
    rewrite_manifest(out.path(), |m| {
        m.source.repo = Some(SUPRA2_SOURCE.repo.into());
        m.source.revision = Some(SUPRA2_SOURCE.revision.into());
        m.source.sha256 = pin.sha256.into();
        m.source.bytes = pin.bytes;
    });
    let err = read_pinned_bundle(out.path(), BundleCheck::SizesOnly)
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-published geometry"), "{err}");

    rewrite_manifest(out.path(), |m| {
        m.config = Supra2Config::published();
        m.denoiser_tensors = Supra2Config::published().expected_tensors().len();
        m.components[1].revision = "0000000000000000000000000000000000000000".into();
    });
    let err = read_pinned_bundle(out.path(), BundleCheck::SizesOnly)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unpinned components"), "{err}");

    // A manifest that drops the stored conditioning is not the pinned checkpoint's bundle: it
    // would silently switch CFG to the empty-string fallback.
    let vae_revision = SD_VAE.revision;
    let conditioning = rewrite_manifest_returning(out.path(), |m| {
        m.components[1].revision = vae_revision.into();
        m.conditioning.take()
    });
    let err = read_pinned_bundle(out.path(), BundleCheck::SizesOnly)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("stores no unconditional conditioning"),
        "{err}"
    );

    // With every manifest pin satisfied, the tiny denoiser is still not the pinned conversion —
    // for both check depths (the files are untouched, so this is the pin, not a digest mismatch).
    rewrite_manifest(out.path(), |m| m.conditioning = conditioning);
    for check in [BundleCheck::SizesOnly, BundleCheck::Digests] {
        let err = read_pinned_bundle(out.path(), check)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("denoiser `denoiser.safetensors`")
                && err.contains("is not the pinned conversion")
                && err.contains(PINNED_DENOISER.sha256),
            "{err}"
        );
    }
}

fn rewrite_manifest_returning<T>(dir: &Path, edit: impl FnOnce(&mut BundleManifest) -> T) -> T {
    let mut out = None;
    rewrite_manifest(dir, |m| out = Some(edit(m)));
    out.unwrap()
}

/// The sizes-only check (footprint/admission) still refuses a truncated file but does not hash.
#[test]
fn sizes_only_check_catches_size_changes_without_hashing() {
    let out = scratch("sizes");
    convert_checkpoint(
        &fixtures().join("tiny_model_with_uncond.pt"),
        out.path(),
        &opts(tiny_config()),
    )
    .unwrap();
    let path = out.path().join(DENOISER_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    // Same size, edited byte: only the digest check sees it.
    read_bundle_checked(out.path(), BundleCheck::SizesOnly).unwrap();
    assert!(read_bundle_checked(out.path(), BundleCheck::Digests).is_err());
    bytes.pop();
    std::fs::write(&path, &bytes).unwrap();
    let err = read_bundle_checked(out.path(), BundleCheck::SizesOnly)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match its manifest"), "{err}");
}

#[test]
fn refuses_mismatched_geometry_unpinned_sources_and_tampered_bundles() {
    let src = fixtures().join("tiny_model_with_uncond.pt");
    // Wrong depth ⇒ missing block tensors; wrong ctx_len ⇒ config mismatch.
    let out = scratch("bad");
    let deeper = Supra2Config {
        depth: 3,
        ..tiny_config()
    };
    assert!(convert_checkpoint(&src, out.path(), &opts(deeper)).is_err());
    let longer = Supra2Config {
        ctx_len: 16,
        ..tiny_config()
    };
    assert!(convert_checkpoint(&src, out.path(), &opts(longer)).is_err());
    assert!(!out.path().join(BUNDLE_MANIFEST).exists());
    // The production entry point accepts only the pinned upstream bytes.
    let err = convert_pinned_checkpoint(&src, out.path()).unwrap_err();
    assert!(
        err.to_string().contains("is not SupraLabs/Supra2-IMG"),
        "{err}"
    );

    // A converted bundle whose denoiser was edited after conversion no longer loads.
    let good = scratch("tamper");
    convert_checkpoint(&src, good.path(), &opts(tiny_config())).unwrap();
    read_bundle(good.path()).unwrap();
    let path = good.path().join(DENOISER_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    let err = read_bundle(good.path()).unwrap_err();
    assert!(
        err.to_string().contains("does not match its manifest"),
        "{err}"
    );
    assert!(read_bundle(scratch("empty").path()).is_err());
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

/// Real pinned checkpoint → bundle; every converted tensor and the stored conditioning are
/// byte-identical to upstream's `torch.load` (digests dumped by the reference script).
/// `SUPRA2_SOURCE_PT=<snapshot>/model_final_ema.pt`; digests: `fixtures/supra2_reference_digests.json`.
#[test]
#[ignore = "needs the pinned Supra2 checkpoint"]
fn real_checkpoint_converts_byte_exact() {
    let out = scratch("real");
    let manifest = convert_pinned_checkpoint(&env_path("SUPRA2_SOURCE_PT"), out.path()).unwrap();
    assert_eq!(
        manifest.source.revision.as_deref(),
        Some(SUPRA2_SOURCE.revision)
    );
    let reference = digests(&fixtures().join("supra2_reference_digests.json"));
    assert_eq!(manifest.source.weights_key, reference["weights_key"]);
    let bundle = read_bundle(out.path()).unwrap();
    assert_denoiser_matches(&bundle, &reference);
    // The conversion is deterministic: it reproduces the recorded pins, so the provider's reader
    // accepts it at both check depths.
    assert_eq!(manifest.denoiser.sha256, PINNED_DENOISER.sha256);
    assert_eq!(manifest.denoiser.bytes, PINNED_DENOISER.bytes);
    let conditioning = manifest.conditioning.as_ref().unwrap();
    assert_eq!(conditioning.sha256, PINNED_CONDITIONING.sha256);
    assert_eq!(conditioning.bytes, PINNED_CONDITIONING.bytes);
    read_pinned_bundle(out.path(), BundleCheck::Digests).unwrap();
    read_pinned_bundle(out.path(), BundleCheck::SizesOnly).unwrap();
    let cond = tensor_digests(&bundle.conditioning_path().unwrap());
    assert_eq!(cond["uncond_text"].1, reference["uncond_text_f32_sha256"]);
    assert_eq!(cond["uncond_mask"].1, reference["uncond_mask_f32_sha256"]);
}

/// Flan-T5 tokenization (short / empty / truncated-long / unicode) is id-for-id and mask-for-mask
/// identical to `AutoTokenizer(..., padding="max_length", truncation=True, max_length=128)`.
/// `SUPRA2_T5_SNAPSHOT=<flan-t5-base snapshot>`; cases: `fixtures/tokenizer_cases.json`.
#[test]
#[ignore = "needs the pinned Flan-T5 tokenizer"]
fn tokenizer_matches_hf_reference() {
    let tok = Supra2Tokenizer::from_dir(&env_path("SUPRA2_T5_SNAPSHOT")).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("tokenizer_cases.json")).unwrap())
            .unwrap();
    let mut saw_truncation = false;
    for (name, case) in cases.as_object().unwrap() {
        let got = tok.encode(case["prompt"].as_str().unwrap(), 128).unwrap();
        let ids: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
        let mask: Vec<i32> = serde_json::from_value(case["mask"].clone()).unwrap();
        assert_eq!(got.ids, ids, "{name}");
        assert_eq!(got.mask, mask, "{name}");
        assert_eq!(
            got.prompt_tokens as u64,
            case["untruncated_len"].as_u64().unwrap(),
            "{name}"
        );
        saw_truncation |= got.truncated;
    }
    assert!(
        saw_truncation,
        "the reference cases must include a truncated prompt"
    );
}

/// Install-time verification of the pinned component snapshots (sizes + SHA-256).
#[test]
#[ignore = "needs the pinned component snapshots"]
fn pinned_component_snapshots_verify() {
    FLAN_T5_BASE
        .verify(&env_path("SUPRA2_T5_SNAPSHOT"))
        .unwrap();
    SD_VAE.verify(&env_path("SUPRA2_VAE_SNAPSHOT")).unwrap();
}
