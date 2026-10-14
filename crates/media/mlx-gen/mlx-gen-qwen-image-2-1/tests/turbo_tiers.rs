//! Installable Q8/Q4 tiers of the distilled **Qwen-Image 2.1 Turbo** checkpoint (sc-25906), on a
//! Turbo-shaped copy of the committed miniature snapshot ([`crate::common::turbo_snapshot`]: saved
//! `sample_sigmas`, static-shift scheduler config, bf16 VAE, single-file text encoder).
//!
//! Every assertion here is made on what the converter **wrote** (the files of the tier directory)
//! or on what the loader **holds** (the tensors and the priced contract of a generator loaded from
//! that directory through the explicit provider catalog), never on a helper's return value.
//!
//! As in `tiers.rs`, the miniature geometry packs only part of itself at the one group size a tier
//! may declare, so the parity bars bound a partially-packed tier; the released geometry packs
//! completely (`convert::released_geometries_are_fully_group_aligned`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{Error as CoreError, Progress};
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_gen_qwen_image_2_1::convert::{
    digest_tree, prequantize_turnkey, source_variant, CHANGES_FILE, NOTICE_FILE, SHA256SUMS_FILE,
};
use mlx_gen_qwen_image_2_1::memory_strategy::derived;
use mlx_gen_qwen_image_2_1::quant::{installed_tier, Tier, GROUP_SIZE};
use mlx_gen_qwen_image_2_1::Variant;
use sha2::{Digest, Sha256};

use crate::common::{tiny_snapshot, turbo_snapshot, TURBO_SAMPLE_SIGMAS};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

fn spec(root: &Path, quant: Option<Quant>, policy: OffloadPolicy) -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy);
    match quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    }
}

fn load(
    id: &str,
    root: &Path,
    quant: Option<Quant>,
    policy: OffloadPolicy,
) -> mlx_gen::gen_core::Result<Box<dyn mlx_gen::Generator>> {
    mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(id, &spec(root, quant, policy))
}

fn convert(source: &Path, dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    prequantize_turnkey(source, &out, tier).expect("the snapshot converts");
    out
}

fn request(steps: Option<u32>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        steps,
        seed: Some(42),
        ..Default::default()
    }
}

/// The generator's RGB8 pixels and the `Step` events it reported.
fn render(
    generator: &dyn mlx_gen::Generator,
    req: &GenerationRequest,
) -> (Vec<u8>, Vec<(u32, u32)>) {
    let mut steps = Vec::new();
    let out = generator
        .generate(req, &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        })
        .expect("render");
    let GenerationOutput::Images(mut images) = out else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    (images.remove(0).pixels, steps)
}

/// `(max |Δ|, mean |Δ|)` over two RGB8 buffers, in 0..=255 units.
fn pixel_errors(got: &[u8], want: &[u8]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "buffer length");
    let mut max = 0f32;
    let mut sum = 0f32;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as f32 - *b as f32).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / got.len().max(1) as f32)
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}")))
        .unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

fn element_width(dtype: mlx_rs::Dtype) -> u64 {
    match dtype {
        mlx_rs::Dtype::Bfloat16 | mlx_rs::Dtype::Float16 => 2,
        mlx_rs::Dtype::Float32 | mlx_rs::Dtype::Uint32 | mlx_rs::Dtype::Int32 => 4,
        other => panic!("unexpected tensor dtype {other:?}"),
    }
}

fn numel(shape: &[i32]) -> u64 {
    shape.iter().map(|&d| d as u64).product()
}

/// Every tensor of the safetensors files under `dir`: name → `(element count, element width)`,
/// read from what is on disk.
fn tensor_inventory(dir: &Path) -> BTreeMap<String, (u64, u64)> {
    let weights = Weights::from_dir(dir).unwrap_or_else(|e| panic!("{dir:?}: {e}"));
    weights
        .keys()
        .map(|name| {
            let tensor = weights.require(name).unwrap();
            (
                name.to_owned(),
                (numel(tensor.shape()), element_width(tensor.dtype())),
            )
        })
        .collect()
}

/// The data region of one safetensors file: its length minus the 8-byte header length and the
/// JSON header itself — the bytes its tensors occupy, read from the file the converter wrote.
fn safetensors_payload_bytes(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    bytes.len() as u64 - 8 - header
}

// ── layout ───────────────────────────────────────────────────────────────────────────────────────

/// A converted Turbo tier is a complete standalone snapshot that **is still a Turbo snapshot**: the
/// saved schedule, the static-shift scheduler config and the bf16 VAE ride through byte for byte,
/// the single-file tower and the DiT are packed at the tier's one width, and the licence artefacts
/// name the Turbo upstream.
///
/// *Mutations that red this:* the converter dropping `sample_sigmas` from the tier's
/// `model_index.json`; casting the VAE on the way through; writing the base repository/revision
/// into a Turbo tier's `CHANGES.md`; leaving a changed file out of `CHANGES.md`; not writing the
/// tier's `NOTICE`, or writing it after the manifest.
#[test]
fn a_converted_turbo_tier_is_a_complete_standalone_turbo_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    assert_eq!(source_variant(&source).unwrap(), Variant::Turbo);

    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert(&source, tmp.path(), tier);
        let bits = tier.transformer_bits().unwrap();
        let name = tier.dir_name();

        // The schedule signals: unchanged files, and the values the Turbo loader reads from them.
        for rel in ["model_index.json", "scheduler/scheduler_config.json"] {
            assert_eq!(
                std::fs::read(out.join(rel)).unwrap(),
                std::fs::read(source.join(rel)).unwrap(),
                "{name}: {rel} must be carried through unchanged"
            );
        }
        assert_eq!(
            json(&out.join("model_index.json"))["sample_sigmas"],
            serde_json::json!(TURBO_SAMPLE_SIGMAS),
            "{name}: the tier keeps the source's saved sigmas"
        );
        let scheduler = json(&out.join("scheduler/scheduler_config.json"));
        assert_eq!(scheduler["use_dynamic_shifting"], serde_json::json!(false));
        assert_eq!(scheduler["shift"], serde_json::json!(1.0));
        assert!(scheduler["shift_terminal"].is_null());
        assert_eq!(source_variant(&out).unwrap(), Variant::Turbo);

        // The VAE is not packable: the same bytes, still bf16 in every tensor.
        let vae = "vae/diffusion_pytorch_model.safetensors";
        assert_eq!(
            std::fs::read(out.join(vae)).unwrap(),
            std::fs::read(source.join(vae)).unwrap(),
            "{name}: the VAE must be copied unchanged"
        );
        let vae_tensors = tensor_inventory(&out.join("vae"));
        assert!(!vae_tensors.is_empty());
        for (tensor, (_, width)) in &vae_tensors {
            assert_eq!(*width, 2, "{name}: vae tensor {tensor} must stay bf16");
        }

        // Both packed components are one file each, at the tier's one width.
        for component in ["transformer", "text_encoder"] {
            let mut files: Vec<String> = std::fs::read_dir(out.join(component))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|file| file.ends_with(".safetensors"))
                .collect();
            files.sort();
            assert_eq!(files, ["model.safetensors"], "{name}: {component}");
            assert_eq!(
                json(&out.join(component).join("config.json"))["quantization"],
                serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE }),
                "{name}: {component} marker"
            );
            assert!(
                tensor_inventory(&out.join(component))
                    .keys()
                    .any(|key| key.ends_with(".scales")),
                "{name}: {component} holds packed weights"
            );
        }
        assert_eq!(installed_tier(&out).unwrap(), tier);

        // The licence rides along, byte for byte.
        assert_eq!(
            std::fs::read(out.join("LICENSE")).unwrap(),
            std::fs::read(source.join("LICENSE")).unwrap(),
            "{name}: LICENSE must be copied unchanged"
        );

        // The §3(c) attribution travels IN the tier: a `NOTICE` file holding the licence's notice
        // verbatim (the same text for both checkpoints — their LICENSE files are one file).
        assert_eq!(
            std::fs::read_to_string(out.join(NOTICE_FILE)).unwrap(),
            format!("{}\n", mlx_gen_qwen_image_2_1::UPSTREAM_LICENSE_NOTICE),
            "{name}: NOTICE must be the attribution notice, verbatim"
        );
        assert_eq!(
            std::fs::read_to_string(out.join(NOTICE_FILE)).unwrap(),
            "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 \
             Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.\n"
        );

        // The change record names every changed file with the tier and group size, and the Turbo
        // upstream — never the base repository or its revision.
        let changes = std::fs::read_to_string(out.join(CHANGES_FILE)).unwrap();
        for needle in [
            format!("`{name}` tier"),
            format!("Q{bits}, group size {GROUP_SIZE}"),
            "`transformer/model.safetensors`".to_owned(),
            "`transformer/config.json`".to_owned(),
            "`text_encoder/model.safetensors`".to_owned(),
            "`text_encoder/config.json`".to_owned(),
            "`vae/`".to_owned(),
            "`model_index.json`".to_owned(),
            "`sample_sigmas`".to_owned(),
            format!("`{}`", mlx_gen_qwen_image_2_1::UPSTREAM_TURBO_HF_REPO),
            format!("`{}`", mlx_gen_qwen_image_2_1::UPSTREAM_TURBO_HF_REVISION),
            format!("`{TURBO}`"),
            "the §3(c) attribution is in the `NOTICE` file in this directory".to_owned(),
            "§3(b)".to_owned(),
            SHA256SUMS_FILE.to_owned(),
        ] {
            assert!(changes.contains(&needle), "{name}: missing {needle:?}");
        }
        for wrong in [
            format!("`{}`", mlx_gen_qwen_image_2_1::UPSTREAM_HF_REPO),
            mlx_gen_qwen_image_2_1::UPSTREAM_HF_REVISION.to_owned(),
            format!("Q{}", if bits == 4 { 8 } else { 4 }),
            "`NOTICE` of the producing crate".to_owned(),
        ] {
            assert!(
                !changes.contains(&wrong),
                "{name}: the Turbo record must not carry {wrong:?}:\n{changes}"
            );
        }

        // Every file whose bytes differ from the source's (or that the source does not have) is
        // either named in the change record or is one of the two artefacts the record itself
        // introduces. Read from the two trees, not from the converter's own list.
        let (written, original) = (digest_tree(&out).unwrap(), digest_tree(&source).unwrap());
        for (rel, digest) in &written {
            if original.get(rel) == Some(digest) || rel == CHANGES_FILE {
                continue;
            }
            assert!(
                changes.contains(&format!("`{rel}`")),
                "{name}: {rel} was changed but CHANGES.md does not name it:\n{changes}"
            );
        }

        // The manifest binds every other file to its bytes.
        let manifest = std::fs::read_to_string(out.join(SHA256SUMS_FILE)).unwrap();
        let mut listed = Vec::new();
        for line in manifest.lines() {
            let (digest, rel) = line
                .split_once("  ")
                .unwrap_or_else(|| panic!("malformed manifest line {line:?}"));
            assert_eq!(
                digest,
                format!(
                    "{:x}",
                    Sha256::digest(std::fs::read(out.join(rel)).unwrap())
                ),
                "{name}: {rel} digest"
            );
            listed.push(rel.to_owned());
        }
        for rel in [
            "LICENSE",
            NOTICE_FILE,
            CHANGES_FILE,
            "model_index.json",
            vae,
        ] {
            assert!(
                listed.iter().any(|l| l == rel),
                "{name}: {rel} is not bound"
            );
        }
        assert_eq!(listed.len(), written.len(), "{name}: every file is listed");
    }
}

/// Converting the same Turbo source twice produces a byte-identical tier — every file, the change
/// record and the manifest included.
///
/// *Mutation that reds this:* anything run-dependent in the tier (a timestamp in `CHANGES.md`).
#[test]
fn turbo_conversion_is_byte_reproducible_across_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    for tier in [Tier::Q8, Tier::Q4] {
        let a = tmp.path().join(format!("a-{}", tier.dir_name()));
        let b = tmp.path().join(format!("b-{}", tier.dir_name()));
        prequantize_turnkey(&source, &a, tier).unwrap();
        // Long enough that a wall-clock stamp at any practical resolution would differ.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        prequantize_turnkey(&source, &b, tier).unwrap();

        let (da, db) = (digest_tree(&a).unwrap(), digest_tree(&b).unwrap());
        assert!(da.contains_key(CHANGES_FILE) && da.contains_key("model_index.json"));
        assert_eq!(da, db, "{}: the two conversions differ", tier.dir_name());
        assert_eq!(
            std::fs::read(a.join(SHA256SUMS_FILE)).unwrap(),
            std::fs::read(b.join(SHA256SUMS_FILE)).unwrap(),
            "{}: the manifests differ",
            tier.dir_name()
        );
    }
}

/// A Turbo tier differs from the base tier of the same width **only where the two snapshots do**:
/// the saved schedule, the scheduler config, the VAE, and the change record that names the
/// source — plus the `NOTICE` file only a Turbo tier carries (a base tier is written exactly as it
/// always was, so published base tiers still reproduce). The packed `transformer/` and `text_encoder/` — weights and markers — are the same
/// bytes, because the converter treats both checkpoints' weights identically. (The fixture's DiT
/// and tower are the base fixture's own, as the released checkpoints' architectures are; this is
/// what lets the Candle twin compose a Turbo tier from the committed packed components.)
///
/// *Mutation that reds this:* any variant-dependent branch in the weight path of the converter.
#[test]
fn a_turbo_tier_differs_from_the_base_tier_only_where_the_snapshots_do() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    for tier in [Tier::Q8, Tier::Q4] {
        let turbo = digest_tree(&convert(&source, &tmp.path().join("turbo"), tier)).unwrap();
        let base = digest_tree(&convert(&tiny_snapshot(), &tmp.path().join("base"), tier)).unwrap();
        // The one file a Turbo tier holds that a base tier does not.
        let only_turbo: Vec<&str> = turbo
            .keys()
            .filter(|rel| !base.contains_key(*rel))
            .map(String::as_str)
            .collect();
        assert_eq!(only_turbo, [NOTICE_FILE], "{}", tier.dir_name());
        assert!(
            base.keys().all(|rel| turbo.contains_key(rel)),
            "{}: a Turbo tier holds every file a base tier does",
            tier.dir_name()
        );
        assert!(
            !tmp.path()
                .join("base")
                .join(tier.dir_name())
                .join(NOTICE_FILE)
                .exists(),
            "{}: the base tier is written as it always was",
            tier.dir_name()
        );
        let differing: Vec<&str> = turbo
            .iter()
            .filter(|(rel, digest)| base.get(*rel).is_some_and(|other| other != *digest))
            .map(|(rel, _)| rel.as_str())
            .collect();
        assert_eq!(
            differing,
            [
                CHANGES_FILE,
                "model_index.json",
                "scheduler/scheduler_config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
            "{}",
            tier.dir_name()
        );
    }
}

// ── load + render ────────────────────────────────────────────────────────────────────────────────

/// A converted Turbo tier loads under `qwen_image_2_1_turbo` with its matching `quantize` selector
/// and renders through the generator contract, over the snapshot's saved sigmas, under both
/// residency policies. The loaded contract prices the bf16 VAE the loader holds, under the Turbo
/// id's own calibration identity.
#[test]
fn turbo_tiers_load_and_render_under_the_turbo_id() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    let dense = load(TURBO, &source, None, OffloadPolicy::Resident).unwrap();
    let (dense_pixels, _) = render(dense.as_ref(), &request(None));
    let vae_bf16_bytes: u64 = tensor_inventory(&source.join("vae"))
        .values()
        .map(|(count, width)| {
            assert_eq!(*width, 2);
            count * width
        })
        .sum();

    let profile = gen_core_testkit::Profile {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        steps: 3,
        seed: 42,
        cancel_steps: 6,
    };
    let mut mean_by_tier = Vec::new();
    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert(&source, tmp.path(), tier);
        let quant = tier.selected_quant();
        let name = tier.dir_name();

        let resident = load(TURBO, &out, quant, OffloadPolicy::Resident)
            .unwrap_or_else(|e| panic!("the {name} Turbo tier must load under {TURBO}: {e}"));
        assert_eq!(resident.descriptor().id, TURBO);
        // What the loader holds: packed weights at the tier's width, and a bf16 VAE.
        assert_eq!(
            mlx_gen_qwen_image_2_1::load_vae(&out)
                .unwrap()
                .compute_dtype(),
            mlx_rs::Dtype::Bfloat16
        );
        let contract = resident
            .memory_strategy_contract()
            .expect("the loaded contract");
        assert_eq!(contract.provider_id, TURBO);
        assert_eq!(
            contract.asset_facts.decoder_bytes, vae_bf16_bytes,
            "{name}: the contract prices the bf16 VAE on disk"
        );
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            mlx_gen_qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT
        );

        // No step count: one step per saved sigma, read from the TIER's own model_index.json.
        let (pixels, steps) = render(resident.as_ref(), &request(None));
        assert_eq!(
            steps,
            (1..=8).map(|current| (current, 8)).collect::<Vec<_>>(),
            "{name}: the tier renders over its eight saved sigmas"
        );
        let (max, mean) = pixel_errors(&pixels, &dense_pixels);
        eprintln!("turbo {name} vs dense turbo: max|Δ|={max:.2}/255 mean|Δ|={mean:.4}/255");
        // Bars for a PARTIALLY packed miniature tier, as in `tiers.rs`.
        let (max_bound, mean_bound) = match tier {
            Tier::Q8 => (16.0, 1.0),
            _ => (64.0, 8.0),
        };
        assert!(max <= max_bound, "{name}: max|Δ|={max} exceeds {max_bound}");
        assert!(
            mean <= mean_bound,
            "{name}: mean|Δ|={mean} exceeds {mean_bound}"
        );
        mean_by_tier.push(mean);

        // Staging is an execution decision: the same pixels.
        let sequential = load(TURBO, &out, quant, OffloadPolicy::Sequential).unwrap();
        assert_eq!(render(sequential.as_ref(), &request(None)).0, pixels);

        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            gen_core_testkit::conformance(|| load(TURBO, &out, quant, policy).unwrap(), &profile);
        }
    }
    assert!(
        mean_by_tier[1] > mean_by_tier[0],
        "Q4 ({:.4}) must be strictly further from the dense render than Q8 ({:.4})",
        mean_by_tier[1],
        mean_by_tier[0]
    );
}

// ── refusals ─────────────────────────────────────────────────────────────────────────────────────

fn unsupported(
    result: mlx_gen::gen_core::Result<Box<dyn mlx_gen::Generator>>,
    what: &str,
) -> String {
    match result {
        Err(CoreError::Unsupported(message)) => message,
        Err(other) => panic!("{what}: expected a typed Unsupported refusal, got {other:?}"),
        Ok(_) => panic!("{what}: must not load"),
    }
}

/// A converted tier carries the same `model_index.json`/scheduler signals as its dense source, so
/// each engine id refuses the other checkpoint's **tier** with exactly the typed refusal the dense
/// snapshot gets — through `load` and through the memory registration alike.
///
/// *Mutation that reds this:* dropping `refuse_saved_sigma_snapshot` from the base load path (a
/// Turbo tier then renders 40 base steps under `qwen_image_2_1`).
#[test]
fn each_id_refuses_the_other_checkpoints_tier_like_its_dense_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let turbo_source = turbo_snapshot(tmp.path());
    let turbo_tiers = tmp.path().join("turbo-tiers");
    let base_tiers = tmp.path().join("base-tiers");
    let resident = OffloadPolicy::Resident;

    let dense_turbo_under_base =
        unsupported(load(BASE, &turbo_source, None, resident), "dense turbo");
    let dense_base_under_turbo =
        unsupported(load(TURBO, &tiny_snapshot(), None, resident), "dense base");
    assert!(dense_turbo_under_base.contains("load it as `qwen_image_2_1_turbo`"));
    assert!(dense_base_under_turbo.contains("as `qwen_image_2_1`"));

    for tier in [Tier::Q8, Tier::Q4] {
        let quant = tier.selected_quant();
        let name = tier.dir_name();
        let turbo_tier = convert(&turbo_source, &turbo_tiers, tier);
        let base_tier = convert(&tiny_snapshot(), &base_tiers, tier);
        assert_eq!(source_variant(&base_tier).unwrap(), Variant::Base);

        assert_eq!(
            unsupported(load(BASE, &turbo_tier, quant, resident), name),
            dense_turbo_under_base,
            "{name}: a Turbo tier under the base id"
        );
        assert_eq!(
            unsupported(load(TURBO, &base_tier, quant, resident), name),
            dense_base_under_turbo,
            "{name}: a base tier under the Turbo id"
        );
        // Each tier still loads under its own id.
        assert!(load(TURBO, &turbo_tier, quant, resident).is_ok());
        assert!(load(BASE, &base_tier, quant, resident).is_ok());

        // The memory registration never prices a snapshot the id would refuse to load.
        for (id, root, want) in [
            (BASE, &turbo_tier, &dense_turbo_under_base),
            (TURBO, &base_tier, &dense_base_under_turbo),
        ] {
            match mlx_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract(
                id,
                &spec(root, quant, resident),
            ) {
                Err(CoreError::Unsupported(message)) => assert_eq!(&message, want, "{name}: {id}"),
                other => panic!("{name}: {id} priced the other checkpoint's tier: {other:?}"),
            }
        }
    }
}

/// The tier/request rules are the base checkpoint's, unchanged, under the Turbo id: a dense Turbo
/// snapshot with a quantize request gets the refusal the dense base snapshot gets for the same
/// request, a request that disagrees with a packed tier is refused, and the converter still
/// refuses to emit a bf16 tier.
///
/// Every refusal reached through the Turbo id names it (E3, sc-25909): the Turbo message is the
/// base one with `qwen_image_2_1:` replaced by `qwen_image_2_1_turbo:`, nothing else.
///
/// *Mutations that red this:* removing the bf16 refusal from `prequantize_turnkey` (it would write
/// a "bf16 tier" directory); dropping `labelled` from `load_variant` (the Turbo refusal keeps the
/// base label).
#[test]
fn turbo_keeps_the_base_tier_refusals() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    let resident = OffloadPolicy::Resident;
    let error = |result: mlx_gen::gen_core::Result<Box<dyn mlx_gen::Generator>>| match result {
        Ok(_) => panic!("the request must be refused"),
        Err(err) => (std::mem::discriminant(&err), err.to_string()),
    };

    // Dense snapshot + quantize request: the same typed refusal on both ids, request for request,
    // each under its own id.
    for quant in [Quant::Q8, Quant::Q4, Quant::Nvfp4] {
        let turbo = error(load(TURBO, &source, Some(quant), resident));
        let (kind, base) = error(load(BASE, &tiny_snapshot(), Some(quant), resident));
        assert!(base.contains("qwen_image_2_1: "), "{quant:?}: {base}");
        assert_eq!(
            turbo,
            (
                kind,
                base.replacen("qwen_image_2_1: ", "qwen_image_2_1_turbo: ", 1)
            ),
            "{quant:?}"
        );
    }
    let (_, tower) = error(load(TURBO, &source, Some(Quant::Q8), resident));
    assert!(tower.contains("would leave the tower dense"), "{tower}");

    // A packed Turbo tier is selected exactly.
    let q8 = convert(&source, tmp.path(), Tier::Q8);
    let (_, crossed) = error(load(TURBO, &q8, Some(Quant::Q4), resident));
    assert!(crossed.contains("silently serve q8"), "{crossed}");
    let (_, unrequested) = error(load(TURBO, &q8, None, resident));
    assert!(
        unrequested.contains("no quantization was requested"),
        "{unrequested}"
    );
    for message in [&crossed, &unrequested] {
        assert!(
            message.contains("qwen_image_2_1_turbo: ") && !message.contains("qwen_image_2_1: "),
            "the Turbo id's tier refusal names the Turbo id: {message}"
        );
    }

    // The dense tier is the source snapshot itself: refused, and nothing is written.
    let bf16 = tmp.path().join("bf16");
    let err = prequantize_turnkey(&source, &bf16, Tier::Bf16)
        .unwrap_err()
        .to_string();
    assert!(err.contains("mirror it"), "{err}");
    assert!(!bf16.exists(), "a refused conversion must write nothing");

    // A source that is neither checkpoint — saved sigmas beside the base's dynamic-shift scheduler
    // config — is refused before anything is written, with the loader's own typed refusal.
    let hybrid = tmp.path().join("hybrid");
    mlx_gen::quant::copy_dir(&source, &hybrid).unwrap();
    std::fs::copy(
        tiny_snapshot().join("scheduler/scheduler_config.json"),
        hybrid.join("scheduler/scheduler_config.json"),
    )
    .unwrap();
    let out = tmp.path().join("hybrid-q8");
    let err = prequantize_turnkey(&hybrid, &out, Tier::Q8)
        .unwrap_err()
        .to_string();
    assert!(err.contains("`use_dynamic_shifting: true`"), "{err}");
    assert!(!out.exists(), "a refused conversion must write nothing");
}

// ── derived sizes ────────────────────────────────────────────────────────────────────────────────

/// **The per-tier size derivation holds on the files the converter writes.** Every packed Linear
/// occupies exactly `derived::packed_bytes(params, bits)`, and a packed component's data region is
/// the source's with each packed Linear's dense bytes replaced by that —
/// `derived::packed_component_payload_bytes`, the rule `derived::tier_tree_bytes` applies to the
/// released file sizes. Held on both packed components of both Turbo tiers.
#[test]
fn packed_turbo_components_weigh_what_the_derivation_says() {
    let tmp = tempfile::tempdir().unwrap();
    let source = turbo_snapshot(tmp.path());
    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert(&source, tmp.path(), tier);
        let bits = tier.transformer_bits().unwrap() as u32;
        for (component, source_file) in [
            ("transformer", "diffusion_pytorch_model.safetensors"),
            ("text_encoder", "model.safetensors"),
        ] {
            let before = tensor_inventory(&source.join(component));
            let after = tensor_inventory(&out.join(component));
            let mut packed_params = 0_u64;
            let mut dense_width = None;
            for base in after.keys().filter_map(|key| key.strip_suffix(".scales")) {
                let (params, width) = before[&format!("{base}.weight")];
                let stored: u64 = ["weight", "scales", "biases"]
                    .iter()
                    .map(|part| {
                        let (count, width) = after[&format!("{base}.{part}")];
                        count * width
                    })
                    .sum();
                assert_eq!(
                    stored,
                    derived::packed_bytes(params, bits),
                    "{}: {component} {base}",
                    tier.dir_name()
                );
                packed_params += params;
                assert_eq!(*dense_width.get_or_insert(width), width);
            }
            assert!(packed_params > 0, "{component} packs at least one Linear");
            assert_eq!(
                safetensors_payload_bytes(&out.join(component).join("model.safetensors")),
                derived::packed_component_payload_bytes(
                    safetensors_payload_bytes(&source.join(component).join(source_file)),
                    packed_params,
                    dense_width.unwrap(),
                    bits,
                ),
                "{}: {component} data region",
                tier.dir_name()
            );
        }
    }
}
