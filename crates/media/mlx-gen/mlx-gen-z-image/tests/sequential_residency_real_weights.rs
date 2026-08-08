//! sc-10839 (epic 10834 Phase 1): the `Sequential` component-residency A/B on real Z-Image weights.
//!
//! `#[ignore]`d — needs a real Z-Image-Turbo snapshot (`ZIMAGE_SNAPSHOT`, else the HF cache). Run:
//!   cargo test -p mlx-gen-z-image --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL A/B (see that file): (1) `Sequential` peaks LOWER than `Resident`
//! because the Qwen text encoder is dropped (+ `clear_cache()`) before the DiT materializes, and
//! (2) the output is BYTE-IDENTICAL. Z-Image's Qwen encoder is comparable to the DiT, so the saving
//! is proportionally larger than SDXL's. A repeat-job check confirms nothing stays resident across
//! jobs. Set `ZIMAGE_SEQ_Q8=1` for the Q8 case, `ZIMAGE_SEQ_STEPS`/`ZIMAGE_SEQ_SIZE` to tune.
//! Exact output artifacts are written under `MEMORY_EVIDENCE_OUTPUT_DIR` (or the system temporary
//! directory) so the strict verifier can independently bind each record to its rendered bytes.

mod common;

use common::snapshot;
use mlx_gen::gen_core::{
    MemoryBackend, MemoryCalibrationIdentity, MemoryEvidenceKey, MemoryEvidenceLogRecord,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract, MemoryParityResult,
    MemoryStrategy, MemoryStrategyParameters,
};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // Turbo is guidance-distilled (no CFG / negative prompt), 4-step default. A fixed seed makes the
    // byte-identity assertion meaningful; quality is irrelevant (Resident vs Sequential, not a golden).
    let size = env_u32("ZIMAGE_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_SEQ_STEPS", 4)),
        ..Default::default()
    }
}

/// A base-`z_image` probe (F-172, sc-11124): the base is undistilled, so it runs **real CFG** — a
/// negative prompt + guidance — which exercises the seam's pos+neg encode/materialize/drop path (the
/// Turbo probe only encodes a single cond). Small step count keeps the ignored test tractable.
fn base_probe_request() -> GenerationRequest {
    let size = env_u32("ZIMAGE_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(4.0),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_SEQ_STEPS", 8)),
        ..Default::default()
    }
}

/// The base `Tongyi-MAI/Z-Image` snapshot, from `ZIMAGE_BASE_SNAPSHOT` (a distinct repo from the Turbo
/// snapshot). Returns `None` so the base A/B skips rather than panics when only the Turbo snapshot is
/// available.
fn base_model_snapshot_opt() -> Option<PathBuf> {
    std::env::var("ZIMAGE_BASE_SNAPSHOT")
        .ok()
        .map(PathBuf::from)
}

fn spec_for(snapshot: PathBuf) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    if std::env::var("ZIMAGE_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn base_spec() -> LoadSpec {
    spec_for(snapshot())
}

fn render_measured(
    policy: OffloadPolicy,
    req: &GenerationRequest,
) -> (Vec<u8>, usize, MemoryEvidenceLogRecord) {
    render_measured_id("z_image_turbo", base_spec(), policy, req)
}

/// The generalized A/B render: load `model_id` from `spec` under `policy`, measure peak, return the
/// single output image's bytes + peak. Shared by the Turbo flagship probe and the base `z_image`
/// sibling probe (sc-11124) so both exercise the identical shared [`mlx_gen::Residency`] seam.
fn render_measured_id(
    model_id: &str,
    spec: LoadSpec,
    policy: OffloadPolicy,
    req: &GenerationRequest,
) -> (Vec<u8>, usize, MemoryEvidenceLogRecord) {
    let spec = spec.with_offload_policy(policy);
    let mut request = req.clone();
    let mut memory = request.memory.unwrap_or_default();
    memory.stage_residency = matches!(policy, OffloadPolicy::Sequential);
    request.memory = Some(memory);
    let model = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load(model_id, &spec)
        .expect("load model");
    let contract = model
        .memory_strategy_contract()
        .expect("Z-Image provider declares a memory strategy contract")
        .clone();
    let observed_calibration = contract
        .calibration
        .clone()
        .expect("Z-Image provider declares a calibration identity");
    reset_peak_memory();
    let out = model.generate(&request, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "expected a single image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    let strategy = if matches!(policy, OffloadPolicy::Sequential) {
        MemoryStrategy::StagedResidency
    } else {
        MemoryStrategy::Resident
    };
    let record = MemoryEvidenceLogRecord {
        key: MemoryEvidenceKey {
            resolved_route: model_id.to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
            load_shape: spec.load_shape,
            mode: MemoryMode::TextToImage,
            overlay: None,
            geometry: MemoryGeometry {
                width: req.width,
                height: req.height,
                batch: req.count,
                frames: 1,
                reference_count: 0,
            },
            strategy,
            engaged_composition: contract.engaged_composition(strategy),
            parameters: MemoryStrategyParameters::default(),
        },
        declared_calibration: MemoryCalibrationIdentity::new(
            mlx_gen_z_image::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ),
        observed_calibration,
        // This is the calibration cell itself: its measured high-water becomes the table's
        // prediction at this exact key. Out-of-sample validation is a separate evidence record.
        predicted_peak_bytes: peak as u64,
        observed_peak_bytes: peak as u64,
        inference_revision: required_revision("INFERENCE_REVISION"),
        sceneworks_revision: required_revision("SCENEWORKS_REVISION"),
        model_revision: required_revision("MEMORY_MODEL_REVISION"),
        model_inventory_sha256: required_sha256("MEMORY_MODEL_INVENTORY_SHA256"),
        harness_version: "inference-z-image-sequential-v1".to_owned(),
        output_sha256: format!("{:x}", Sha256::digest(&pixels)),
        parity: MemoryParityContract::Exact,
        // `NotRun` is honest HERE and only here: this single leg renders one output, and by itself
        // proves nothing about output preservation. The A/B tests upgrade both legs to `Passed`
        // through [`assert_exact_parity_and_mark_passed`], which establishes byte-identity first so
        // the verdict is earned by this run (sc-17861). The repeat-job test never emits its records,
        // so its legs stay `NotRun` — nobody compared them.
        parity_result: MemoryParityResult::NotRun,
    };
    (pixels, peak, record)
}

/// Establish the `Exact` parity contract both records declare, then — and only then — record the
/// verdict (sc-17861, following the per-row pattern from `mlx-gen-sana`, sc-15523).
///
/// Three checks, each load-bearing:
/// 1. equal length — the positional byte diff below zips, and `zip` silently truncates to the
///    shorter side, so a truncated output would otherwise sail through with `diff == 0`;
/// 2. zero positional byte diff — the headline byte-identity claim, with a readable count;
/// 3. the two records' own `output_sha256` fields agree — binds the verdict to the exact bytes each
///    record was built over, so a record constructed from other bytes than the ones compared here
///    cannot be stamped `Passed`.
///
/// Only after all three does either record's `parity_result` become `Passed`; on any failure both
/// stay `NotRun` and the test dies before printing them.
fn assert_exact_parity_and_mark_passed(
    model_id: &str,
    pixels_resident: &[u8],
    pixels_sequential: &[u8],
    resident_record: &mut MemoryEvidenceLogRecord,
    sequential_record: &mut MemoryEvidenceLogRecord,
) {
    assert_eq!(
        pixels_resident.len(),
        pixels_sequential.len(),
        "{model_id}: Sequential residency changed the output LENGTH ({} vs {} bytes) — the byte \
         diff below would silently truncate to the shorter side",
        pixels_resident.len(),
        pixels_sequential.len(),
    );
    let diff = pixels_resident
        .iter()
        .zip(pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "{model_id}: Sequential residency changed the output: {diff}/{} bytes differ (must be \
         byte-identical)",
        pixels_resident.len()
    );
    assert_eq!(
        resident_record.output_sha256, sequential_record.output_sha256,
        "{model_id}: the records' output_sha256 fields disagree even though the compared buffers \
         are byte-identical — a record was built over different bytes than the ones compared here",
    );
    resident_record.parity_result = MemoryParityResult::Passed;
    sequential_record.parity_result = MemoryParityResult::Passed;
}

fn required_revision(name: &str) -> String {
    let revision =
        std::env::var(name).unwrap_or_else(|_| panic!("set {name} to an exact Git commit"));
    assert!(
        revision.len() == 40
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be an exact lowercase 40-character Git commit"
    );
    revision
}

fn required_sha256(name: &str) -> String {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("set {name} to an exact SHA-256"));
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be an exact lowercase 64-character SHA-256"
    );
    value
}

fn persist_evidence_outputs(model_id: &str, resident: &[u8], staged: &[u8]) -> (PathBuf, PathBuf) {
    let root = std::env::var_os("MEMORY_EVIDENCE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("inference-memory-evidence-v1"));
    std::fs::create_dir_all(&root).expect("create memory evidence output directory");
    let resident_path = root.join(format!("{model_id}-resident.rgb"));
    let staged_path = root.join(format!("{model_id}-staged.rgb"));
    std::fs::write(&resident_path, resident).expect("write resident evidence artifact");
    std::fs::write(&staged_path, staged).expect("write staged evidence artifact");
    (resident_path, staged_path)
}

#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident, mut resident_record) =
        render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential, mut sequential_record) =
        render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Z-Image {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("ZIMAGE_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    // Persist BEFORE the parity assertion so a failing run leaves both outputs on disk for
    // diagnosis — a byte-identity failure with no bytes to diff is undebuggable.
    let (resident_path, staged_path) =
        persist_evidence_outputs("z_image_turbo", &pixels_resident, &pixels_sequential);
    assert_exact_parity_and_mark_passed(
        "z_image_turbo",
        &pixels_resident,
        &pixels_sequential,
        &mut resident_record,
        &mut sequential_record,
    );
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
    println!("{}", resident_record.to_json_line().unwrap());
    println!("{}", sequential_record.to_json_line().unwrap());
    println!(
        "MEMORY_EVIDENCE_ARTIFACTS resident={} staged={}",
        resident_path.display(),
        staged_path.display()
    );
}

/// A weights-free record over `pixels` for the parity-helper mutation checks below. Everything the
/// helper does not read is a placeholder; `output_sha256` is computed from `pixels` exactly the way
/// [`render_measured_id`] computes it, so the sha-binding check is exercised for real.
fn placeholder_record(pixels: &[u8]) -> MemoryEvidenceLogRecord {
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/nonexistent/z-image")));
    let calibration = MemoryCalibrationIdentity::new(
        mlx_gen_z_image::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    );
    MemoryEvidenceLogRecord {
        key: MemoryEvidenceKey {
            resolved_route: "z_image_turbo".to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
            load_shape: spec.load_shape,
            mode: MemoryMode::TextToImage,
            overlay: None,
            geometry: MemoryGeometry {
                width: 2,
                height: 1,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            strategy: MemoryStrategy::Resident,
            engaged_composition: vec![MemoryStrategy::Resident],
            parameters: MemoryStrategyParameters::default(),
        },
        declared_calibration: calibration.clone(),
        observed_calibration: calibration,
        predicted_peak_bytes: 1,
        observed_peak_bytes: 1,
        inference_revision: "a".repeat(40),
        sceneworks_revision: "b".repeat(40),
        model_revision: "c".repeat(40),
        model_inventory_sha256: "d".repeat(64),
        harness_version: "inference-z-image-sequential-v1".to_owned(),
        output_sha256: format!("{:x}", Sha256::digest(pixels)),
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    }
}

/// sc-17861 acceptance: every `Passed` these harnesses print is earned by
/// [`assert_exact_parity_and_mark_passed`], and the four tests below pin each escape direction —
/// weights-free, so the seam reddens on every `cargo test`, not only on the ignored real-weight A/B.
#[test]
fn parity_helper_marks_passed_only_after_byte_identity() {
    let pixels = [7u8, 11, 13];
    let mut resident = placeholder_record(&pixels);
    let mut sequential = placeholder_record(&pixels);
    assert_eq!(resident.parity_result, MemoryParityResult::NotRun);
    assert_exact_parity_and_mark_passed("unit", &pixels, &pixels, &mut resident, &mut sequential);
    assert_eq!(resident.parity_result, MemoryParityResult::Passed);
    assert_eq!(sequential.parity_result, MemoryParityResult::Passed);
}

#[test]
#[should_panic(expected = "bytes differ")]
fn parity_helper_reddens_when_an_optimized_byte_changes() {
    let resident_pixels = [7u8, 11, 13];
    let mut sequential_pixels = resident_pixels;
    sequential_pixels[1] ^= 1;
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_exact_parity_and_mark_passed(
        "unit",
        &resident_pixels,
        &sequential_pixels,
        &mut resident,
        &mut sequential,
    );
}

#[test]
#[should_panic(expected = "changed the output LENGTH")]
fn parity_helper_reddens_when_the_optimized_output_is_truncated() {
    // `zip` truncates to the shorter side, so without the explicit length check a truncated output
    // would pass the positional byte diff with `diff == 0`.
    let resident_pixels = [7u8, 11, 13];
    let sequential_pixels = [7u8, 11];
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_exact_parity_and_mark_passed(
        "unit",
        &resident_pixels,
        &sequential_pixels,
        &mut resident,
        &mut sequential,
    );
}

#[test]
#[should_panic(expected = "output_sha256")]
fn parity_helper_reddens_when_a_record_was_built_over_other_bytes() {
    // The compared buffers are identical, but one record's `output_sha256` was computed over
    // different bytes — `Passed` must not be stampable onto a record the comparison never covered.
    let pixels = [7u8, 11, 13];
    let mut resident = placeholder_record(&pixels);
    let mut sequential = placeholder_record(&[13u8, 11, 7]);
    assert_exact_parity_and_mark_passed("unit", &pixels, &pixels, &mut resident, &mut sequential);
}

#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1, _e1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2, _e2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Z-Image Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    let slop = peak1 / 10;
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real base Z-Image snapshot (set ZIMAGE_BASE_SNAPSHOT)"]
fn base_z_image_sequential_bounds_peak_and_is_byte_identical() {
    // F-172 (sc-11124): the base `z_image` sibling now honors `offload_policy` via the SAME shared
    // `mlx_gen::Residency` seam as the Turbo flagship — before the fix it ignored the policy and always
    // loaded `Resident` (silently OOMing a fit-gated Sequential request). This is the base analog of
    // `sequential_bounds_peak_and_is_byte_identical`, gated on a distinct base snapshot; it runs REAL
    // CFG (pos+neg encode), the harder seam path. The two control siblings share this same seam via
    // `load_control_residency` (weight-free routing covered by their unit tests); a control real-weight
    // A/B is a follow-up needing the base + control checkpoints (overlaps sc-11126 F-180).
    let Some(snap) = base_model_snapshot_opt() else {
        eprintln!("skipping: set ZIMAGE_BASE_SNAPSHOT to run the base z_image residency A/B");
        return;
    };
    let req = base_probe_request();
    let (pixels_resident, peak_resident, mut resident_record) = render_measured_id(
        "z_image",
        spec_for(snap.clone()),
        OffloadPolicy::Resident,
        &req,
    );
    let (pixels_sequential, peak_sequential, mut sequential_record) =
        render_measured_id("z_image", spec_for(snap), OffloadPolicy::Sequential, &req);

    println!(
        "base z_image {}x{} @ {} steps (CFG):\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB",
        req.width,
        req.height,
        req.steps.unwrap(),
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
    );

    // Persist BEFORE the parity assertion so a failing run leaves both outputs on disk for
    // diagnosis — a byte-identity failure with no bytes to diff is undebuggable.
    let (resident_path, staged_path) =
        persist_evidence_outputs("z_image", &pixels_resident, &pixels_sequential);
    assert_exact_parity_and_mark_passed(
        "z_image",
        &pixels_resident,
        &pixels_sequential,
        &mut resident_record,
        &mut sequential_record,
    );
    assert!(
        peak_sequential < peak_resident,
        "base z_image Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder \
         drop did not reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
    println!("{}", resident_record.to_json_line().unwrap());
    println!("{}", sequential_record.to_json_line().unwrap());
    println!(
        "MEMORY_EVIDENCE_ARTIFACTS resident={} staged={}",
        resident_path.display(),
        staged_path.display()
    );
}
