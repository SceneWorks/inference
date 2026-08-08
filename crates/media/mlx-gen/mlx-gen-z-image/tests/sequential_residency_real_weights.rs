//! sc-10839 (epic 10834 Phase 1): the `Sequential` component-residency A/B on real Z-Image weights.
//!
//! `#[ignore]`d — needs a real Z-Image-Turbo snapshot (`ZIMAGE_SNAPSHOT`, else the HF cache). Run:
//!   cargo test -p mlx-gen-z-image --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL A/B (see that file), with one adjudicated amendment (sc-18149):
//! (1) `Sequential` peaks LOWER than `Resident` because the Qwen text encoder is dropped
//! (+ `clear_cache()`) before the DiT materializes, and (2) the output is within the **declared
//! decode-drift tolerance** of `Resident` — not byte-identical, and that is by construction, not
//! by accident:
//!
//! ## The sc-18149 adjudication: why `Exact` was unsatisfiable and what replaced it
//!
//! sc-10839 (2026-07-11) landed this A/B with a byte-identity assertion, and at that revision it
//! held: `Sequential` and `Resident` both decoded the VAE whole-image. Nine days later sc-13571
//! (`36ab0a91`, 2026-07-20) gated the **tiled** VAE decode on the `Sequential` signal to bound the
//! ~14 GiB untiled decode transient that OOMs an 8 GB Mac (GitHub #1658) — a deliberate,
//! GroupNorm-approximate route ("tiling is gated on the Sequential signal so large-memory Macs
//! still decode exactly", its own commit message). From that commit on, the two legs of this A/B
//! run *different decodes by design*, so `MemoryParityContract::Exact` was structurally
//! unsatisfiable; the contradiction stayed invisible because this test was `#[ignore]`d and
//! rarely run (found by sc-17861's sweep: 1,556,876/1,769,472 subpixel bytes differ at 768²,
//! max 50/255, mean 2.30 — deterministic and policy-linked).
//!
//! The A/B therefore now renders **three** legs and splits the claim to match the mechanism:
//!
//! - `Resident` (untiled decode) vs `Sequential` (staged + tiled decode): the shipping routes,
//!   bounded by the declared [`MemoryParityContract::Tolerance`] on the **mean** absolute u8
//!   subpixel delta ([`DECODE_TILING_MEAN_ABS_U8`]), with the p99 tail pinned alongside
//!   ([`DECODE_TILING_P99_ABS_U8`]). The mean is the non-saturating statistic (the same
//!   adjudication shape as SANA's sc-17863, where maxD saturated at 255 and carried no
//!   information).
//! - `Resident` **+ forced tiled decode** (the isolator probe, not a shipping route, no evidence
//!   record) vs `Sequential`: **byte-identical**. This leg attributes the entire drift to the
//!   tiled decode: component staging itself — encode → drop → rematerialize, phase-boundary
//!   `eval`s, the offload load path — is numerically EXACT. A future change that makes staging
//!   itself drift (an mlx-rs bump, a residency-seam change) reddens this assertion even though
//!   the tolerance above might still absorb it.
//!
//! A repeat-job check confirms nothing stays resident across jobs. Set `ZIMAGE_SEQ_Q8=1` for the
//! Q8 case, `ZIMAGE_SEQ_STEPS`/`ZIMAGE_SEQ_SIZE` to tune. Exact output artifacts are written under
//! `MEMORY_EVIDENCE_OUTPUT_DIR` (or the system temporary directory) so the strict verifier can
//! independently bind each record to its rendered bytes and re-check the declared tolerance from
//! the artifacts alone. Set `ZIMAGE_AB_RENDER_OUT` to also dump viewable PPMs (each leg plus an
//! 8x-amplified diff) for adjudication with eyes on the renders.

mod common;

use common::snapshot;
use mlx_gen::gen_core::{
    GenerationMemory, MemoryBackend, MemoryCalibrationIdentity, MemoryEvidenceKey,
    MemoryEvidenceLogRecord, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract,
    MemoryParityResult, MemoryStrategy, MemoryStrategyParameters,
};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The declared ceiling on the **mean** absolute u8 subpixel delta between the `Resident`
/// (whole-image decode) and `Sequential` (tiled decode) legs — the tolerance the evidence records
/// carry (sc-18149).
///
/// Measured at 768², 4 steps, bf16, tile edge 512 / overlap 64 (the provider defaults the
/// Sequential route decodes at), across three seeds
/// ([`staged_drift_is_attributed_to_the_tiled_decode_across_seeds`], run on nax-macos
/// 2026-08-08, snapshot `bb2bc989`):
///
/// | seed | bytes differing | meanD | p99D | maxD | staged vs resident+tiled |
/// |---|---:|---:|---:|---:|---:|
/// | 1234 | 1,556,876 / 1,769,472 | 2.298 | 7 | 50 | **0 bytes** |
/// | 7 | 1,512,925 / 1,769,472 | 2.342 | 7 | 45 | **0 bytes** |
/// | 8675309 | 1,575,195 / 1,769,472 | 2.793 | 9 | 53 | **0 bytes** |
///
/// The last column is the adjudication's backbone: across every seed the Sequential output is
/// byte-identical to a Resident render forced onto the same tiled decode, so the drift is 100%
/// the tiled decode and 0% the residency staging.
///
/// At 512² and below the drift is measured EXACTLY zero (max 0): a 512-pixel tile covers the whole
/// output, and the degenerate single-tile plan reproduces the whole-image decode byte-for-byte.
/// The evidence lane used to pin `ZIMAGE_SEQ_SIZE: 512` to sit in that regime ("so residency is
/// the only mechanism that differs"), which kept `Exact` green at the measured cell while the
/// shipping 768²+ configuration drifted uncharacterized — sc-18149 removed the pin so the lane
/// exercises the geometry this ceiling actually bounds.
///
/// The ceiling is 4.0: ~1.43x headroom over the worst measured seed (2.793), and below the 6.0
/// SANA ships under a 32x-scale DC-AE — this VAE is 8x-scale with a 512/64 tile geometry, so its
/// drift floor is lower and the ceiling follows the measurement rather than copying SANA's. The
/// mean is declared (not max) because the max is an extreme-order statistic over ~1.8M subpixels;
/// sc-17863 measured it saturating at 255 on SANA and sc-17743 measured a 2.9x seed spread — the
/// mean's measured spread here is 1.22x. The p99 tail rides alongside as a secondary pin
/// ([`DECODE_TILING_P99_ABS_U8`]) so a pathological redistribution of the same mean (a few huge
/// deltas hiding under many zeros) cannot pass unnoticed.
const DECODE_TILING_MEAN_ABS_U8: f64 = 4.0;

/// The p99 companion to [`DECODE_TILING_MEAN_ABS_U8`]: 99% of subpixel deltas must sit at or below
/// this value. Derived exactly the way the mean's ceiling is: worst measured seed (9, from the
/// table above; seed spread 7..9, 1.29x) times the same ~1.43x headroom factor the mean carries
/// (9 x 1.43 = 12.9, rounded up to the next integer quantile step). The verifier recomputes this
/// quantile from the two bound artifacts and the lane pins it via `--max-p99-abs-u8`, so the pin
/// is enforced independently of this harness (sc-18149 review).
const DECODE_TILING_P99_ABS_U8: u32 = 13;

/// The declared parity contract both A/B evidence records carry (sc-18149): the drift the
/// Sequential route's forced tiled decode (sc-13571) introduces, bounded on the non-saturating
/// mean. See [`DECODE_TILING_MEAN_ABS_U8`] for the measurement and the module docs for why `Exact`
/// was structurally unsatisfiable.
fn decode_drift_tolerance() -> MemoryParityContract {
    MemoryParityContract::Tolerance {
        metric: "mean_abs_u8_subpixel".to_owned(),
        maximum_error: DECODE_TILING_MEAN_ABS_U8,
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // Turbo is guidance-distilled (no CFG / negative prompt), 4-step default. A fixed seed makes the
    // parity assertions meaningful; quality is irrelevant (Resident vs Sequential, not a golden).
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

/// Load `model_id` from `spec` under `policy` with the request-scoped `memory` controls, render one
/// image, measure peak. The record-free core shared by the two evidence legs and the isolator
/// probe.
fn render_under(
    model_id: &str,
    spec: LoadSpec,
    policy: OffloadPolicy,
    memory: GenerationMemory,
    req: &GenerationRequest,
) -> (Vec<u8>, usize, mlx_gen::gen_core::MemoryProviderContract) {
    let spec = spec.with_offload_policy(policy);
    let mut request = req.clone();
    request.memory = Some(memory);
    let model = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load(model_id, &spec)
        .expect("load model");
    let contract = model
        .memory_strategy_contract()
        .expect("Z-Image provider declares a memory strategy contract")
        .clone();
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
    (pixels, peak, contract)
}

/// The **isolator probe** (sc-18149): `Resident` load and residency, with the tiled decode the
/// `Sequential` route is forced onto (same default tile geometry — the request names none, exactly
/// like the Sequential leg). Everything about this render is the Resident leg except the decode, so
/// comparing it byte-for-byte against the Sequential leg attributes the drift: identical bytes
/// prove component staging is numerically exact and the tiled decode is the sole divergence. Not a
/// shipping route; deliberately emits **no evidence record** (the strict verifier requires exactly
/// one record per strategy in the lane log).
fn render_resident_tiled(model_id: &str, spec: LoadSpec, req: &GenerationRequest) -> Vec<u8> {
    let memory = GenerationMemory {
        tile_vae_decode: true,
        ..Default::default()
    };
    render_under(model_id, spec, OffloadPolicy::Resident, memory, req).0
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
    let memory = GenerationMemory {
        stage_residency: matches!(policy, OffloadPolicy::Sequential),
        ..req.memory.unwrap_or_default()
    };
    let contract_spec = spec.clone().with_offload_policy(policy);
    let (pixels, peak, contract) = render_under(model_id, spec, policy, memory, req);
    let observed_calibration = contract
        .calibration
        .clone()
        .expect("Z-Image provider declares a calibration identity");
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
                precision: contract_spec.precision,
                quant: contract_spec.quantize,
                component_precision_floors: &[],
            },
            load_shape: contract_spec.load_shape,
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
            contract_spec.load_shape,
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
        parity: decode_drift_tolerance(),
        // `NotRun` is honest HERE and only here: this single leg renders one output, and by itself
        // proves nothing about output preservation. The A/B tests upgrade both legs to `Passed`
        // through [`assert_decode_drift_within_ceiling_and_mark_passed`], which establishes the
        // declared tolerance first so the verdict is earned by this run (sc-17861). The repeat-job
        // test never emits its records, so its legs stay `NotRun` — nobody compared them.
        parity_result: MemoryParityResult::NotRun,
    };
    (pixels, peak, record)
}

/// The mean absolute u8 subpixel delta — the declared, non-saturating tolerance metric.
fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
    let total: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.len() as f64
}

/// The `q`-quantile of the absolute u8 subpixel difference, computed exactly from a 256-bin
/// histogram — the tail statistic that, unlike the max, does not saturate on this route (sc-18149,
/// following sc-17863).
fn quantile_delta(a: &[u8], b: &[u8], q: f64) -> u32 {
    assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
    let mut histogram = [0u64; 256];
    for (x, y) in a.iter().zip(b) {
        histogram[x.abs_diff(*y) as usize] += 1;
    }
    let need = (q * a.len() as f64).ceil() as u64;
    let mut seen = 0u64;
    for (delta, count) in histogram.iter().enumerate() {
        seen += count;
        if seen >= need {
            return delta as u32;
        }
    }
    255
}

/// The maximum absolute u8 subpixel delta — printed for the record, never declared (extreme-order
/// statistic; see [`DECODE_TILING_MEAN_ABS_U8`]).
fn max_delta(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| u32::from(x.abs_diff(*y)))
        .max()
        .unwrap_or(0)
}

/// The per-subpixel absolute difference, amplified 8x and clamped — the seam-structure image.
fn amplified_diff(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b)
        .map(|(x, y)| (u32::from(x.abs_diff(*y)) * 8).min(255) as u8)
        .collect()
}

/// Dump one RGB8 buffer as a binary PPM into `ZIMAGE_AB_RENDER_OUT` (no-op when unset) so the
/// adjudication can be made with eyes on the renders rather than on statistics alone (sc-18149).
fn dump_ppm(name: &str, width: u32, height: u32, pixels: &[u8]) {
    let Ok(dir) = std::env::var("ZIMAGE_AB_RENDER_OUT") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create ZIMAGE_AB_RENDER_OUT");
    let mut data = format!("P6\n{width} {height}\n255\n").into_bytes();
    data.extend_from_slice(pixels);
    std::fs::write(dir.join(format!("{name}.ppm")), data).expect("write ppm");
}

/// The isolator's claim (sc-18149): `staged` (Sequential: staged residency + tiled decode) must be
/// **byte-identical** to `resident_tiled` (Resident residency + the same tiled decode). Identical
/// bytes attribute the whole A/B drift to the tiled decode and prove component staging itself is
/// numerically exact; any difference here means the residency seam itself started drifting — a
/// mechanism change the drift tolerance must never silently absorb.
fn assert_staging_is_exact(model_id: &str, staged: &[u8], resident_tiled: &[u8]) {
    assert_eq!(
        staged.len(),
        resident_tiled.len(),
        "{model_id}: the staged and resident+tiled outputs differ in LENGTH ({} vs {} bytes) — \
         the positional byte diff below would silently truncate to the shorter side",
        staged.len(),
        resident_tiled.len(),
    );
    let diff = staged
        .iter()
        .zip(resident_tiled)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "{model_id}: component staging itself drifted: {diff}/{} bytes differ between the \
         Sequential leg and the Resident+tiled isolator, which run the SAME decode — the residency \
         seam (encode/drop/rematerialize, phase evals, offload load path) is no longer numerically \
         exact and the decode-drift tolerance must not be allowed to absorb it",
        staged.len(),
    );
}

/// Establish the declared decode-drift tolerance both records carry, then — and only then — record
/// the verdict (sc-17861 discipline, sc-18149 contract).
///
/// Five checks, each load-bearing:
/// 1. equal length — the positional metrics below zip, and `zip` silently truncates to the shorter
///    side, so a truncated output would otherwise sail through;
/// 2. each record's `output_sha256` matches the exact buffer it is being judged over — a record
///    constructed from other bytes than the ones compared here cannot be stamped `Passed`;
/// 3. both records declare exactly the adjudicated tolerance ([`decode_drift_tolerance`]) — this
///    helper must not promote a record carrying a contract (for example `Exact`) it never checked;
/// 4. the mean absolute delta is within the declared ceiling — the contract itself;
/// 5. the p99 delta is within its companion pin — so a pathological redistribution of the same
///    mean cannot pass.
///
/// Only after all five does either record's `parity_result` become `Passed`; on any failure both
/// stay `NotRun` and the test dies before printing them.
fn assert_decode_drift_within_ceiling_and_mark_passed(
    model_id: &str,
    pixels_resident: &[u8],
    pixels_sequential: &[u8],
    resident_record: &mut MemoryEvidenceLogRecord,
    sequential_record: &mut MemoryEvidenceLogRecord,
) {
    assert_eq!(
        pixels_resident.len(),
        pixels_sequential.len(),
        "{model_id}: Sequential residency changed the output LENGTH ({} vs {} bytes) — the \
         positional metrics below would silently truncate to the shorter side",
        pixels_resident.len(),
        pixels_sequential.len(),
    );
    let resident_sha = format!("{:x}", Sha256::digest(pixels_resident));
    let sequential_sha = format!("{:x}", Sha256::digest(pixels_sequential));
    assert_eq!(
        resident_record.output_sha256, resident_sha,
        "{model_id}: the resident record's output_sha256 does not match the compared resident \
         buffer — the record was built over different bytes than the ones judged here",
    );
    assert_eq!(
        sequential_record.output_sha256, sequential_sha,
        "{model_id}: the staged record's output_sha256 does not match the compared staged buffer \
         — the record was built over different bytes than the ones judged here",
    );
    let expected = decode_drift_tolerance();
    for (leg, record) in [
        ("resident", &*resident_record),
        ("staged", &*sequential_record),
    ] {
        assert_eq!(
            record.parity, expected,
            "{model_id}: the {leg} record declares a parity contract this helper never checked — \
             only the adjudicated decode-drift tolerance can be promoted here",
        );
    }
    let mean = mean_abs_delta(pixels_resident, pixels_sequential);
    assert!(
        mean <= DECODE_TILING_MEAN_ABS_U8,
        "{model_id}: the tiled-decode drift mean {mean:.4} exceeds the declared ceiling \
         {DECODE_TILING_MEAN_ABS_U8} (mean_abs_u8_subpixel)",
    );
    let p99 = quantile_delta(pixels_resident, pixels_sequential, 0.99);
    assert!(
        p99 <= DECODE_TILING_P99_ABS_U8,
        "{model_id}: the tiled-decode drift p99 {p99} exceeds its companion pin \
         {DECODE_TILING_P99_ABS_U8} — the tail redistributed even though the mean may be within \
         its ceiling",
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

/// Persist the three rendered legs for the strict verifier. The isolator artifact is deliberately
/// part of the signature: the verifier independently re-checks `sha(staged) == sha(isolator)`
/// (lane-wired via `--isolator-output`), so the residency-exactness claim is enforced outside this
/// harness — a harness that stops rendering the isolator leg cannot compile past this call, and a
/// lane run without the artifact fails the verifier (sc-18149 review).
fn persist_evidence_outputs(
    model_id: &str,
    resident: &[u8],
    staged: &[u8],
    resident_tiled: &[u8],
) -> (PathBuf, PathBuf, PathBuf) {
    let root = std::env::var_os("MEMORY_EVIDENCE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("inference-memory-evidence-v1"));
    std::fs::create_dir_all(&root).expect("create memory evidence output directory");
    let resident_path = root.join(format!("{model_id}-resident.rgb"));
    let staged_path = root.join(format!("{model_id}-staged.rgb"));
    let isolator_path = root.join(format!("{model_id}-resident-tiled.rgb"));
    std::fs::write(&resident_path, resident).expect("write resident evidence artifact");
    std::fs::write(&staged_path, staged).expect("write staged evidence artifact");
    std::fs::write(&isolator_path, resident_tiled).expect("write isolator evidence artifact");
    (resident_path, staged_path, isolator_path)
}

#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_within_declared_decode_drift() {
    let req = probe_request();
    let (pixels_resident, peak_resident, mut resident_record) =
        render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential, mut sequential_record) =
        render_measured(OffloadPolicy::Sequential, &req);
    // The isolator probe (sc-18149): Resident residency, Sequential's tiled decode. Rendered last
    // so the two evidence legs' peaks are measured exactly as they always were.
    let pixels_resident_tiled = render_resident_tiled("z_image_turbo", base_spec(), &req);

    println!(
        "Z-Image {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)\n  drift vs Resident: mean {:.4}, p99 {}, max {}",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("ZIMAGE_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
        mean_abs_delta(&pixels_resident, &pixels_sequential),
        quantile_delta(&pixels_resident, &pixels_sequential, 0.99),
        max_delta(&pixels_resident, &pixels_sequential),
    );

    // Persist BEFORE the parity assertions so a failing run leaves all outputs on disk for
    // diagnosis — a parity failure with no bytes to diff is undebuggable.
    let (resident_path, staged_path, isolator_path) = persist_evidence_outputs(
        "z_image_turbo",
        &pixels_resident,
        &pixels_sequential,
        &pixels_resident_tiled,
    );
    dump_ppm(
        "z_image_turbo_resident",
        req.width,
        req.height,
        &pixels_resident,
    );
    dump_ppm(
        "z_image_turbo_staged",
        req.width,
        req.height,
        &pixels_sequential,
    );
    dump_ppm(
        "z_image_turbo_resident_tiled",
        req.width,
        req.height,
        &pixels_resident_tiled,
    );
    dump_ppm(
        "z_image_turbo_drift_diff8x",
        req.width,
        req.height,
        &amplified_diff(&pixels_resident, &pixels_sequential),
    );
    assert_staging_is_exact("z_image_turbo", &pixels_sequential, &pixels_resident_tiled);
    assert_decode_drift_within_ceiling_and_mark_passed(
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
        "MEMORY_EVIDENCE_ARTIFACTS resident={} staged={} isolator={}",
        resident_path.display(),
        staged_path.display(),
        isolator_path.display()
    );
}

/// The multi-seed resample behind [`DECODE_TILING_MEAN_ABS_U8`] (sc-18149): per seed, render the
/// three legs and (1) attribute the drift — the Sequential leg must be byte-identical to the
/// Resident+tiled isolator, and (2) bound it — the Resident-vs-Sequential drift must sit within
/// the declared mean ceiling and its p99 companion. Statistics print per seed so the ceiling can
/// be re-derived from a run log; PPMs (legs + 8x diff) land in `ZIMAGE_AB_RENDER_OUT` when set.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn staged_drift_is_attributed_to_the_tiled_decode_across_seeds() {
    const SEEDS: [u64; 3] = [1234, 7, 8_675_309];
    for seed in SEEDS {
        let mut req = probe_request();
        req.seed = Some(seed);
        let (pixels_resident, _, _) = render_under(
            "z_image_turbo",
            base_spec(),
            OffloadPolicy::Resident,
            GenerationMemory::default(),
            &req,
        );
        let (pixels_sequential, _, _) = render_under(
            "z_image_turbo",
            base_spec(),
            OffloadPolicy::Sequential,
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
            &req,
        );
        let pixels_resident_tiled = render_resident_tiled("z_image_turbo", base_spec(), &req);
        let differing = pixels_resident
            .iter()
            .zip(&pixels_sequential)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "[sc-18149 seed {seed}] resident vs staged: {differing}/{} bytes differ, mean {:.4}, \
             p99 {}, max {}; staged vs resident+tiled: {} bytes differ",
            pixels_resident.len(),
            mean_abs_delta(&pixels_resident, &pixels_sequential),
            quantile_delta(&pixels_resident, &pixels_sequential, 0.99),
            max_delta(&pixels_resident, &pixels_sequential),
            pixels_sequential
                .iter()
                .zip(&pixels_resident_tiled)
                .filter(|(a, b)| a != b)
                .count(),
        );
        dump_ppm(
            &format!("seed{seed}_resident"),
            req.width,
            req.height,
            &pixels_resident,
        );
        dump_ppm(
            &format!("seed{seed}_staged"),
            req.width,
            req.height,
            &pixels_sequential,
        );
        dump_ppm(
            &format!("seed{seed}_resident_tiled"),
            req.width,
            req.height,
            &pixels_resident_tiled,
        );
        dump_ppm(
            &format!("seed{seed}_drift_diff8x"),
            req.width,
            req.height,
            &amplified_diff(&pixels_resident, &pixels_sequential),
        );
        assert_staging_is_exact("z_image_turbo", &pixels_sequential, &pixels_resident_tiled);
        let mean = mean_abs_delta(&pixels_resident, &pixels_sequential);
        assert!(
            mean <= DECODE_TILING_MEAN_ABS_U8,
            "seed {seed}: the tiled-decode drift mean {mean:.4} exceeds the declared ceiling \
             {DECODE_TILING_MEAN_ABS_U8}",
        );
        let p99 = quantile_delta(&pixels_resident, &pixels_sequential, 0.99);
        assert!(
            p99 <= DECODE_TILING_P99_ABS_U8,
            "seed {seed}: the tiled-decode drift p99 {p99} exceeds its companion pin \
             {DECODE_TILING_P99_ABS_U8}",
        );
    }
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
        parity: decode_drift_tolerance(),
        parity_result: MemoryParityResult::NotRun,
    }
}

/// sc-17861 acceptance, restated for the sc-18149 contract: every `Passed` these harnesses print is
/// earned by [`assert_decode_drift_within_ceiling_and_mark_passed`], and the tests below pin each
/// escape direction — weights-free, so the seam reddens on every `cargo test`, not only on the
/// ignored real-weight A/B.
#[test]
fn parity_helper_marks_passed_only_after_the_declared_tolerance() {
    let pixels = [7u8, 11, 13];
    let mut resident = placeholder_record(&pixels);
    let mut sequential = placeholder_record(&pixels);
    assert_eq!(resident.parity_result, MemoryParityResult::NotRun);
    assert_decode_drift_within_ceiling_and_mark_passed(
        "unit",
        &pixels,
        &pixels,
        &mut resident,
        &mut sequential,
    );
    assert_eq!(resident.parity_result, MemoryParityResult::Passed);
    assert_eq!(sequential.parity_result, MemoryParityResult::Passed);
}

#[test]
fn parity_helper_passes_bounded_drift_below_the_ceiling() {
    // A real (non-zero) drift below both pins must pass: one subpixel off by 1 out of three is a
    // mean of 1/3 and a p99 of 1 — comfortably inside the ceilings, exactly the situation the
    // tolerance downgrade exists to admit.
    let resident_pixels = [7u8, 11, 13];
    let mut sequential_pixels = resident_pixels;
    sequential_pixels[1] ^= 1;
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_decode_drift_within_ceiling_and_mark_passed(
        "unit",
        &resident_pixels,
        &sequential_pixels,
        &mut resident,
        &mut sequential,
    );
    assert_eq!(resident.parity_result, MemoryParityResult::Passed);
    assert_eq!(sequential.parity_result, MemoryParityResult::Passed);
}

#[test]
#[should_panic(expected = "exceeds the declared ceiling")]
fn parity_helper_reddens_when_the_mean_breaches_the_ceiling() {
    // Every subpixel off by 5 (mean 5.0 > 4.0): the declared ceiling must be able to fail — the
    // mutation check that makes the tolerance a contract rather than a number.
    let resident_pixels = [7u8, 11, 13];
    let sequential_pixels = [12u8, 16, 18];
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_decode_drift_within_ceiling_and_mark_passed(
        "unit",
        &resident_pixels,
        &sequential_pixels,
        &mut resident,
        &mut sequential,
    );
}

#[test]
#[should_panic(expected = "exceeds its companion pin")]
fn parity_helper_reddens_when_the_tail_breaches_p99_with_the_mean_within_ceiling() {
    // 11 of 1000 subpixels off by 255, the rest identical: mean = 2.805 (within the 4.0 ceiling)
    // but more than 1% of subpixels sit at 255, so p99 = 255 — the pathological redistribution the
    // companion pin exists to catch.
    let resident_pixels = [0u8; 1000];
    let mut sequential_pixels = [0u8; 1000];
    for delta in sequential_pixels.iter_mut().take(11) {
        *delta = 255;
    }
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_decode_drift_within_ceiling_and_mark_passed(
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
    // would pass the positional metrics unnoticed.
    let resident_pixels = [7u8, 11, 13];
    let sequential_pixels = [7u8, 11];
    let mut resident = placeholder_record(&resident_pixels);
    let mut sequential = placeholder_record(&sequential_pixels);
    assert_decode_drift_within_ceiling_and_mark_passed(
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
    // The compared buffers are within tolerance, but one record's `output_sha256` was computed over
    // different bytes — `Passed` must not be stampable onto a record the comparison never covered.
    let pixels = [7u8, 11, 13];
    let mut resident = placeholder_record(&pixels);
    let mut sequential = placeholder_record(&[13u8, 11, 7]);
    assert_decode_drift_within_ceiling_and_mark_passed(
        "unit",
        &pixels,
        &pixels,
        &mut resident,
        &mut sequential,
    );
}

#[test]
#[should_panic(expected = "parity contract this helper never checked")]
fn parity_helper_refuses_a_record_declaring_a_different_contract() {
    // A record declaring `Exact` (or any other contract) must not be promotable by the tolerance
    // helper — the verdict must be earned against the exact contract the record carries.
    let pixels = [7u8, 11, 13];
    let mut resident = placeholder_record(&pixels);
    resident.parity = MemoryParityContract::Exact;
    let mut sequential = placeholder_record(&pixels);
    assert_decode_drift_within_ceiling_and_mark_passed(
        "unit",
        &pixels,
        &pixels,
        &mut resident,
        &mut sequential,
    );
}

#[test]
#[should_panic(expected = "component staging itself drifted")]
fn staging_exactness_check_reddens_when_a_byte_changes() {
    let staged = [7u8, 11, 13];
    let mut resident_tiled = staged;
    resident_tiled[1] ^= 1;
    assert_staging_is_exact("unit", &staged, &resident_tiled);
}

#[test]
#[should_panic(expected = "differ in LENGTH")]
fn staging_exactness_check_reddens_when_an_output_is_truncated() {
    let staged = [7u8, 11, 13];
    let resident_tiled = [7u8, 11];
    assert_staging_is_exact("unit", &staged, &resident_tiled);
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
fn base_z_image_sequential_bounds_peak_within_declared_decode_drift() {
    // F-172 (sc-11124): the base `z_image` sibling now honors `offload_policy` via the SAME shared
    // `mlx_gen::Residency` seam as the Turbo flagship — before the fix it ignored the policy and always
    // loaded `Resident` (silently OOMing a fit-gated Sequential request). This is the base analog of
    // `sequential_bounds_peak_within_declared_decode_drift`, gated on a distinct base snapshot; it runs
    // REAL CFG (pos+neg encode), the harder seam path. The two control siblings share this same seam via
    // `load_control_residency` (weight-free routing covered by their unit tests); a control real-weight
    // A/B is a follow-up needing the base + control checkpoints (overlaps sc-11126 F-180). The base
    // pipeline shares `decode_tiling` with Turbo, so its Sequential leg is tiled the same way and
    // carries the same adjudicated tolerance (sc-18149) at the same default tile geometry.
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
    let (pixels_sequential, peak_sequential, mut sequential_record) = render_measured_id(
        "z_image",
        spec_for(snap.clone()),
        OffloadPolicy::Sequential,
        &req,
    );
    let pixels_resident_tiled = render_resident_tiled("z_image", spec_for(snap), &req);

    println!(
        "base z_image {}x{} @ {} steps (CFG):\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  drift vs Resident: mean {:.4}, p99 {}, max {}",
        req.width,
        req.height,
        req.steps.unwrap(),
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        mean_abs_delta(&pixels_resident, &pixels_sequential),
        quantile_delta(&pixels_resident, &pixels_sequential, 0.99),
        max_delta(&pixels_resident, &pixels_sequential),
    );

    // Persist BEFORE the parity assertions so a failing run leaves all outputs on disk for
    // diagnosis — a parity failure with no bytes to diff is undebuggable.
    let (resident_path, staged_path, isolator_path) = persist_evidence_outputs(
        "z_image",
        &pixels_resident,
        &pixels_sequential,
        &pixels_resident_tiled,
    );
    assert_staging_is_exact("z_image", &pixels_sequential, &pixels_resident_tiled);
    assert_decode_drift_within_ceiling_and_mark_passed(
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
        "MEMORY_EVIDENCE_ARTIFACTS resident={} staged={} isolator={}",
        resident_path.display(),
        staged_path.display(),
        isolator_path.display()
    );
}
