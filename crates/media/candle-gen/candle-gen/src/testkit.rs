//! Shared test-support helpers (sc-9055 / F-069) — the single home for the PPM read/write, cosine,
//! env-path, and GPU peak-VRAM helpers that had been hand-copied into ~16 `#[cfg(test)]` validation
//! modules across the provider crates.
//!
//! Why one home: the copies had already **drifted** — two PPM header tokenizers (one comment-tolerant,
//! one not) and an f32- vs f64-accumulating cosine. A comment-bearing PPM passed some harnesses and
//! failed others, and a methodology fix had to be mirrored by hand. Concentrating them here makes the
//! behaviour canonical.
//!
//! Weight snapshots are **not** resolved here: inference never self-fetches or derives an HF-cache
//! location (epic 13657); real-weight tests take explicit passed-in env paths.
//!
//! Behaviour is preserved for every caller:
//! * [`read_ppm`] is the **comment-tolerant** tokenizer (a strict superset — the non-tolerant callers only
//!   ever read comment-free `P6` files written by [`write_ppm`], for which the two agree byte-for-byte).
//! * [`cosine`] is the full normalized cosine (`0.0` when either input is the zero vector); [`cosine_dot`]
//!   is the bare dot product for callers whose inputs are already L2-normalized (SDXL/Kolors/FLUX IP).
//!
//! Gated behind the crate `testkit` feature so this test-only surface (and its `std::process` /
//! `nvidia-smi` dependency) never compiles into a production build. Provider crates enable it as a
//! dev-dependency feature: `candle-gen = { path = "...", features = ["testkit"] }` under
//! `[dev-dependencies]`, or `candle-gen/testkit` in a test-only feature.

#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use sha2::{Digest, Sha256};

use crate::candle_core::{Device, Tensor};
use crate::gen_core::{
    Image, LoadShape, MemoryBackend, MemoryCalibrationIdentity, MemoryEvidenceKey,
    MemoryEvidenceLogRecord, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract,
    MemoryParityResult, MemoryStrategy, MemoryStrategyParameters,
};

/// Typed inputs for one real-weight calibration observation.
pub struct MemoryEvidenceProbe<'a> {
    pub resolved_route: &'a str,
    pub declared_calibration: MemoryCalibrationIdentity,
    pub observed_calibration: MemoryCalibrationIdentity,
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub mode: MemoryMode,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    pub strategy: MemoryStrategy,
    pub engaged_composition: Vec<MemoryStrategy>,
    pub parameters: MemoryStrategyParameters,
    pub observed_peak_bytes: u64,
    pub harness_version: &'a str,
    pub output_bytes: &'a [u8],
}

/// Emit one strict `MEMORY_EVIDENCE_V1` line from a Candle real-weight probe.
///
/// The measurement establishes the prediction at this exact calibration cell, so the observed
/// high-water is written to both peak fields. A later out-of-sample validation must construct
/// [`MemoryEvidenceLogRecord`] directly with the previously promoted prediction.
pub fn memory_evidence_v1_line(probe: MemoryEvidenceProbe<'_>) -> String {
    memory_evidence_v1_line_with_parity(
        probe,
        MemoryParityContract::Exact,
        MemoryParityResult::NotRun,
    )
}

/// Emit one strict observation with the parity contract/result established by a provider-owned
/// real-weight comparator. The ordinary helper above remains conservative (`Exact` / `NotRun`) for
/// harnesses that only capture an output and leave comparison to a later verifier.
pub fn memory_evidence_v1_line_with_parity(
    probe: MemoryEvidenceProbe<'_>,
    parity: MemoryParityContract,
    parity_result: MemoryParityResult,
) -> String {
    let output_sha256 = format!("{:x}", Sha256::digest(probe.output_bytes));
    MemoryEvidenceLogRecord {
        key: MemoryEvidenceKey {
            resolved_route: probe.resolved_route.to_owned(),
            backend: MemoryBackend::Candle,
            tier: probe.tier,
            load_shape: probe.load_shape,
            mode: probe.mode,
            overlay: probe.overlay,
            geometry: probe.geometry,
            strategy: probe.strategy,
            engaged_composition: probe.engaged_composition,
            parameters: probe.parameters,
        },
        declared_calibration: probe.declared_calibration,
        observed_calibration: probe.observed_calibration,
        predicted_peak_bytes: probe.observed_peak_bytes,
        observed_peak_bytes: probe.observed_peak_bytes,
        inference_revision: required_git_revision("INFERENCE_REVISION"),
        sceneworks_revision: required_git_revision("SCENEWORKS_REVISION"),
        model_revision: required_git_revision("MEMORY_MODEL_REVISION"),
        model_inventory_sha256: required_sha256("MEMORY_MODEL_INVENTORY_SHA256"),
        harness_version: probe.harness_version.to_owned(),
        output_sha256,
        parity,
        parity_result,
    }
    .to_json_line()
    .expect("real-weight probe must produce a valid MEMORY_EVIDENCE_V1 record")
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

/// The calibration identity the operator/workflow expected before loading the provider.
///
/// The observed identity comes from the provider's exported constant or executable contract and is
/// passed separately to [`MemoryEvidenceProbe`]. Keeping these sources distinct makes a stale runner
/// fail at the writer instead of stamping its expectation onto the observed record.
pub fn expected_memory_calibration(load_shape: LoadShape) -> MemoryCalibrationIdentity {
    let fingerprint = std::env::var("MEMORY_EXPECTED_FINGERPRINT")
        .expect("set MEMORY_EXPECTED_FINGERPRINT to the provider's exported fingerprint");
    let abi = std::env::var("MEMORY_EXPECTED_ABI")
        .expect("set MEMORY_EXPECTED_ABI to the provider's exported ABI")
        .parse::<u32>()
        .expect("MEMORY_EXPECTED_ABI must be an unsigned integer");
    MemoryCalibrationIdentity {
        abi,
        fingerprint,
        load_shape,
    }
}

fn required_git_revision(name: &str) -> String {
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

// ---------------------------------------------------------------------------------------------------
// Env paths
// ---------------------------------------------------------------------------------------------------

/// A required env-var path for an opt-in real-weight test. Panics with a clear message if unset —
/// these tests are `#[ignore]`d and only run when the caller exports the env var.
pub fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| {
        panic!("set ${key} (see the test module docs for the real-weight run)")
    }))
}

/// An optional env-var path — `None` when unset (for tests that skip gracefully rather than panic).
pub fn env_path_opt(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

static PROCESS_ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Serialize and restore one process-global environment variable for deterministic tests.
pub struct EnvVarGuard {
    _lock: MutexGuard<'static, ()>,
    key: &'static str,
    prior: Option<OsString>,
}

impl EnvVarGuard {
    pub fn set(key: &'static str, value: Option<&str>) -> Self {
        let lock = PROCESS_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        Self {
            _lock: lock,
            key,
            prior,
        }
    }
}

#[cfg(all(test, unix))]
mod env_tests {
    use super::*;

    #[test]
    fn env_guard_restores_non_utf8_value_exactly() {
        use std::os::unix::ffi::OsStringExt;

        const KEY: &str = "CANDLE_GEN_TESTKIT_NON_UTF8";
        let original = OsString::from_vec(vec![b'x', 0xff, b'y']);
        std::env::set_var(KEY, &original);
        {
            let _guard = EnvVarGuard::set(KEY, None);
            assert!(std::env::var_os(KEY).is_none());
        }
        assert_eq!(std::env::var_os(KEY), Some(original));
        std::env::remove_var(KEY);
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.prior.take() {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// PPM image IO (codec-less — the harnesses own their image IO)
// ---------------------------------------------------------------------------------------------------

/// Minimal binary-`P6` PPM reader — `P6 <w> <h> <maxval>` header then `w*h*3` raw RGB bytes. Tolerant
/// of a single (or several) `#`-comment line and arbitrary header whitespace; enough for hand-prepared
/// reference images (the `image` dep in these crates is built codec-less).
///
/// This is the comment-tolerant tokenizer; on the comment-free files [`write_ppm`] produces it agrees
/// byte-for-byte with the older whitespace-only readers it replaced.
pub fn read_ppm(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut i = 0usize;
    let mut tok = || -> String {
        // skip whitespace + comment lines
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        String::from_utf8_lossy(&bytes[start..i]).to_string()
    };
    assert_eq!(tok(), "P6", "{} is not a binary (P6) PPM", path.display());
    let w: usize = tok().parse().expect("ppm width");
    let h: usize = tok().parse().expect("ppm height");
    let _max: usize = tok().parse().expect("ppm maxval");
    i += 1; // single whitespace after maxval, before the pixel block
    let pixels = bytes[i..i + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Write a binary-`P6` PPM (`P6\n<w> <h>\n255\n<rgb bytes>`). Convert to PNG out-of-band for viewing.
pub fn write_ppm(path: &Path, img: &Image) {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Mean absolute per-byte difference between two equal-size RGB renders (the
/// injection-changes-the-output sanity metric). Panics on a size mismatch.
pub fn mean_abs_diff(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "render size mismatch");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.pixels.len() as f32
}

// ---------------------------------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------------------------------

/// Cosine similarity of two equal-length embeddings, normalizing internally (inputs need NOT be
/// pre-normalized). Returns `0.0` when either input is the zero vector.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Bare dot product of two equal-length vectors — the cosine metric when both inputs are already
/// L2-normalized (the SDXL / Kolors / FLUX IP-adapter feature extractors normalize before comparing).
pub fn cosine_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity for Candle tensors, accumulated in `f64` to keep quantization fixture
/// comparisons stable. The tensors are flattened and converted from `f32` on the CPU.
pub fn tensor_cosine(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Build the canonical MLX Q4 test fixture used by the sibling Candle providers.
pub fn q4_packed(
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> (Tensor, Tensor, Tensor, Vec<f32>) {
    q4_packed_with(
        out_dim,
        in_dim,
        group_size,
        |i| ((i * 7 + i / 13) % 16) as u8,
        |group| 0.0625 * (group as f32 + 1.0),
        |group| -0.5 - 0.25 * group as f32,
    )
}

/// Build an MLX Q4 packed triple and its exact affine grid while allowing a test to retain its
/// fixture-specific code, scale, and bias patterns.
pub fn q4_packed_with<C, S, B>(
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    code: C,
    scale: S,
    bias: B,
) -> (Tensor, Tensor, Tensor, Vec<f32>)
where
    C: Fn(usize) -> u8,
    S: Fn(usize) -> f32,
    B: Fn(usize) -> f32,
{
    assert_eq!(in_dim % group_size, 0);
    assert_eq!(in_dim % 8, 0);
    let codes: Vec<u8> = (0..out_dim * in_dim).map(code).collect();
    assert!(
        codes.iter().all(|&code| code <= 0xF),
        "Q4 fixture codes must fit in a nibble"
    );
    let groups_per_row = in_dim / group_size;
    let groups = out_dim * groups_per_row;
    let scales: Vec<f32> = (0..groups).map(scale).collect();
    let biases: Vec<f32> = (0..groups).map(bias).collect();
    let grid = (0..out_dim * in_dim)
        .map(|i| {
            let group = (i / in_dim) * groups_per_row + (i % in_dim) / group_size;
            scales[group] * codes[i] as f32 + biases[group]
        })
        .collect();
    let words = codes
        .chunks_exact(8)
        .map(|codes| {
            codes.iter().enumerate().fold(0u32, |word, (i, &code)| {
                word | ((code as u32 & 0xF) << (4 * i))
            })
        })
        .collect::<Vec<_>>();
    let dev = Device::Cpu;
    (
        Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
        Tensor::from_vec(scales, (out_dim, groups_per_row), &dev).unwrap(),
        Tensor::from_vec(biases, (out_dim, groups_per_row), &dev).unwrap(),
        grid,
    )
}

#[cfg(test)]
mod quant_fixture_tests {
    use super::q4_packed_with;

    #[test]
    fn q4_packed_with_keeps_packed_codes_and_affine_grid_in_sync() {
        let (weight, scales, biases, grid) =
            q4_packed_with(1, 8, 8, |i| i as u8, |_| 2.0, |_| -1.0);

        assert_eq!(
            weight.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            [0x7654_3210]
        );
        assert_eq!(
            scales.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [2.0]
        );
        assert_eq!(
            biases.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [-1.0]
        );
        assert_eq!(grid, [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);
    }

    #[test]
    #[should_panic(expected = "Q4 fixture codes must fit in a nibble")]
    fn q4_packed_with_rejects_codes_above_four_bits() {
        let _ = q4_packed_with(1, 8, 8, |_| 0x10, |_| 1.0, |_| 0.0);
    }
}

// ---------------------------------------------------------------------------------------------------
// GPU peak-VRAM sampler (device-level `nvidia-smi memory.used`) — used by the video-VAE decode sweeps
// ---------------------------------------------------------------------------------------------------

pub use gpu_peak::{probe_gpu, used_mib, PeakSampler};
pub use vram_probe::{VramProbe, VramReport};

/// The driver memory-pool probe moved to [`crate::cuda_mempool`] in SC-15792 so it is compiled and
/// linted by the CUDA lane (which enables `cuda` but not `testkit`) and so the rung-4 harnesses have
/// one implementation to share instead of a fork. Re-exported here unchanged for the provider
/// real-weight tests that already import it from `testkit`.
///
/// New code should reach for [`crate::cuda_mempool::MemPool`], which also exposes the RESERVED
/// counters an admission gate actually reads, the release threshold, and a non-default pool handle.
#[cfg(feature = "cuda")]
pub use crate::cuda_mempool::{cuda_mempool_used_high_bytes, reset_cuda_mempool_high_water};

mod vram_probe {
    //! sc-9094 — the per-tier VRAM measuring harness (epic 9083's packed-load rollout). Wraps the
    //! device-level [`PeakSampler`] into the three phase quantities the manifest's per-variant
    //! `minMemoryGb` gate is derived from:
    //!
    //! * **load peak** — the transient high-water mark *during* model load (weights → device,
    //!   packed-repack, CPU-staging). For flux2-dev this is the headline: the dense CPU-stage path
    //!   peaked ~105 GB; the packed Q4 load lands the quantized footprint on-device directly.
    //! * **steady resident** — device VRAM after load settles, *before* denoise — the persistent
    //!   weight + component footprint a job holds for its whole lifetime.
    //! * **overall peak** — the max across the whole generate (load + denoise + VAE decode). This is
    //!   the number the card must physically hold; `minMemoryGb` = this + headroom.
    //!
    //! All three are **device-level** `nvidia-smi memory.used` deltas over a recorded `baseline`
    //! (WDDM reports per-process `used_memory` as `[N/A]`, and the card must fit the *whole* device's
    //! resident bytes anyway). Run on an otherwise-idle GPU; the report prints the baseline so a
    //! non-zero pre-run residency is visible. The sampler is an in-process helper thread (part of the
    //! measurement, not a background job) polling every ~40 ms — fast enough to catch a multi-hundred-ms
    //! load/decode transient.
    //!
    //! Usage from a provider example (load and generate are separate phases, so their peaks separate):
    //! ```ignore
    //! let mut probe = VramProbe::start_rendered(); // records the rendered GPU's idle baseline
    //! let load = probe.phase();                     // sample across load
    //! let gen = provider_registry.load(id, &spec)?; //   ... weights → device ...
    //! probe.end_load(load);                         // load peak recorded; steady sampled now
    //! let run = probe.phase();                      // sample across generate
    //! let out = gen.generate(&req, &mut cb)?;       //   ... denoise + decode ...
    //! probe.end_gen(run);                           // overall peak recorded
    //! println!("{}", probe.report());               // load-peak / steady / overall-peak (GB)
    //! ```

    use super::gpu_peak::{probe_gpu, used_mib, PeakSampler};

    /// MiB → GB (10⁹ bytes — the manifest's `minMemoryGb` is base-10 GB, matching the MLX footprint
    /// numbers). `1 MiB = 2²⁰ bytes`.
    fn mib_to_gb(mib: u64) -> f64 {
        (mib as f64) * (1024.0 * 1024.0) / 1.0e9
    }

    /// The three phase quantities (GB) plus the idle baseline they were measured over.
    #[derive(Clone, Copy, Debug)]
    pub struct VramReport {
        /// Device VRAM already resident before the run (GB) — should be ~0 on an idle GPU.
        pub baseline_gb: f64,
        /// Transient high-water mark during load, over baseline (GB).
        pub load_peak_gb: f64,
        /// Resident VRAM after load settles, before denoise, over baseline (GB).
        pub steady_gb: f64,
        /// Max over the whole generate (load + denoise + decode), over baseline (GB).
        pub peak_gb: f64,
    }

    impl std::fmt::Display for VramReport {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "load-peak {:.1} GB | steady {:.1} GB | overall-peak {:.1} GB (baseline {:.1} GB)",
                self.load_peak_gb, self.steady_gb, self.peak_gb, self.baseline_gb
            )
        }
    }

    impl VramReport {
        /// Fail rather than publish a peak sampled from a busy or unreadable GPU. Returns `self` so a
        /// harness can validate and then print/use the same report value.
        pub fn assert_trustworthy(self, max_baseline_gb: f64) -> Self {
            assert!(
                self.baseline_gb < max_baseline_gb,
                "sampled GPU was not idle (baseline {:.1} GB, required < {:.1} GB); the peak is contaminated",
                self.baseline_gb,
                max_baseline_gb
            );
            assert!(
                self.peak_gb > 0.0,
                "probe reported a 0.0 GB peak; nvidia-smi is unavailable or the query failed"
            );
            self
        }
    }

    /// A phase-scoped [`PeakSampler`] the caller starts around a load or generate call and hands back
    /// to the matching `end_*` to fold its peak into the report.
    pub struct Phase(PeakSampler);

    /// The per-run VRAM probe. [`start`](Self::start) records the idle baseline; each phase is bracketed
    /// by [`phase`](Self::phase) → the work → `end_load` / `end_gen`.
    pub struct VramProbe {
        gpu: usize,
        baseline_mib: u64,
        load_peak_mib: u64,
        steady_mib: u64,
        overall_peak_mib: u64,
    }

    impl VramProbe {
        /// Record the idle baseline on the physical GPU that Candle's logical `cuda:0` renders on.
        /// This derives the ordinal from `CUDA_VISIBLE_DEVICES` via [`probe_gpu`] so a multi-GPU run
        /// cannot silently render on one card while sampling another (sc-12107).
        pub fn start_rendered() -> Self {
            Self::start(probe_gpu())
        }

        /// Record the idle device baseline (used MiB) for GPU ordinal `gpu`.
        ///
        /// A failed query is fatal: treating it as a zero baseline would turn the absence of a
        /// measurement into a plausible low peak that could understate a manifest requirement.
        pub fn start(gpu: usize) -> Self {
            let baseline = used_mib(gpu).unwrap_or_else(|| {
                panic!(
                    "cannot read VRAM for physical GPU {gpu} with nvidia-smi; refusing to record an untrustworthy peak"
                )
            });
            Self {
                gpu,
                baseline_mib: baseline,
                load_peak_mib: baseline,
                steady_mib: baseline,
                overall_peak_mib: baseline,
            }
        }

        /// Fail if the recorded baseline is not approximately idle. Returns `self` so callers that
        /// need phase-relative measurements after a deliberate resident load can retain the trusted
        /// pre-run baseline as provenance.
        pub fn assert_idle(self, max_baseline_gb: f64) -> Self {
            let baseline_gb = mib_to_gb(self.baseline_mib);
            assert!(
                baseline_gb < max_baseline_gb,
                "sampled GPU was not idle (baseline {baseline_gb:.1} GB, required < {max_baseline_gb:.1} GB); the peak is contaminated"
            );
            self
        }

        /// Begin sampling a phase (load or generate). Keep the returned [`Phase`] alive across the work
        /// and pass it to the matching `end_*`.
        pub fn phase(&self) -> Phase {
            Phase(PeakSampler::start(self.gpu))
        }

        /// Close an arbitrary observed sub-phase and return its device peak over this probe's idle
        /// baseline. Provider measurement harnesses use this to split a generate call into physical
        /// text/denoise/decode residency phases while the ordinary load/generate report continues to
        /// sample the whole call.
        pub fn end_observed(&self, phase: Phase) -> f64 {
            mib_to_gb(phase.0.stop().saturating_sub(self.baseline_mib))
        }

        /// Close the **load** phase: fold its peak into `load_peak`, and sample the settled resident
        /// (`steady`) right now (load done, denoise not started). Also seeds the overall peak.
        pub fn end_load(&mut self, phase: Phase) {
            let load_peak = phase.0.stop();
            self.load_peak_mib = self.load_peak_mib.max(load_peak);
            self.overall_peak_mib = self.overall_peak_mib.max(load_peak);
            // Steady = the instantaneous resident after load, before any denoise allocation.
            if let Some(m) = used_mib(self.gpu) {
                self.steady_mib = m;
                self.overall_peak_mib = self.overall_peak_mib.max(m);
            }
        }

        /// Close the **generate** phase: fold its peak into the overall peak.
        pub fn end_gen(&mut self, phase: Phase) {
            let gen_peak = phase.0.stop();
            self.overall_peak_mib = self.overall_peak_mib.max(gen_peak);
        }

        /// The three phase quantities in GB, over the idle baseline (clamped at 0 — a slightly-lower
        /// late sample must not read as negative usage).
        pub fn report(&self) -> VramReport {
            let over = |m: u64| mib_to_gb(m.saturating_sub(self.baseline_mib));
            VramReport {
                baseline_gb: mib_to_gb(self.baseline_mib),
                load_peak_gb: over(self.load_peak_mib),
                steady_gb: over(self.steady_mib),
                peak_gb: over(self.overall_peak_mib),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mib_to_gb_is_base10_gb() {
            // 1024 MiB = 2³⁰ bytes ≈ 1.0737 GB (base-10).
            assert!((mib_to_gb(1024) - 1.0737).abs() < 1e-3);
        }

        #[test]
        fn report_is_delta_over_baseline_and_nonnegative() {
            // A probe with a synthetic baseline: deltas subtract it, and a below-baseline sample
            // clamps to 0 rather than going negative.
            let mut p = VramProbe {
                gpu: 0,
                baseline_mib: 1000,
                load_peak_mib: 5000,
                steady_mib: 3000,
                overall_peak_mib: 6000,
            };
            let r = p.report();
            assert!((r.load_peak_gb - mib_to_gb(4000)).abs() < 1e-6);
            assert!((r.steady_gb - mib_to_gb(2000)).abs() < 1e-6);
            assert!((r.peak_gb - mib_to_gb(5000)).abs() < 1e-6);
            // A late sample below baseline must not underflow.
            p.steady_mib = 500;
            assert_eq!(p.report().steady_gb, 0.0);
        }

        #[test]
        fn trustworthy_report_rejects_busy_and_zero_peak_samples() {
            let good = VramReport {
                baseline_gb: 0.2,
                load_peak_gb: 2.0,
                steady_gb: 1.5,
                peak_gb: 3.0,
            };
            assert_eq!(good.assert_trustworthy(1.0).peak_gb, 3.0);

            let busy = VramReport {
                baseline_gb: 2.0,
                ..good
            };
            assert!(std::panic::catch_unwind(|| busy.assert_trustworthy(1.0)).is_err());

            let unreadable = VramReport {
                peak_gb: 0.0,
                ..good
            };
            assert!(std::panic::catch_unwind(|| unreadable.assert_trustworthy(1.0)).is_err());
        }

        #[test]
        fn idle_probe_rejects_a_contaminated_baseline() {
            let probe = |baseline_mib| VramProbe {
                gpu: 0,
                baseline_mib,
                load_peak_mib: baseline_mib,
                steady_mib: baseline_mib,
                overall_peak_mib: baseline_mib,
            };

            assert_eq!(probe(200).assert_idle(1.0).baseline_mib, 200);
            assert!(std::panic::catch_unwind(|| probe(2_000).assert_idle(1.0)).is_err());
        }
    }
}

mod gpu_peak {
    //! sc-7148 — shared `nvidia-smi` peak-VRAM sampler for the video-VAE decode sweeps. Polls
    //! device-level `memory.used` in a background thread and tracks the max while a decode runs.
    //!
    //! Device-level (not per-process) is deliberate: Windows WDDM reports per-process `used_memory` as
    //! `[N/A]`, and the budgeted decode's safe ceiling is *total* VRAM × 0.85, so the honest "did it
    //! fit" quantity is the whole device's used bytes during the decode. Run the sweep on an
    //! otherwise-idle GPU (the harness prints the pre-decode `baseline` so you can confirm nothing else
    //! was resident).

    use std::process::Command;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    /// nvidia-smi poll cadence. ~40 ms is well under a multi-second VAE decode (so the peak is
    /// captured) while keeping the subprocess-spawn overhead negligible.
    const POLL: Duration = Duration::from_millis(40);

    fn parse_probe_gpu(raw: Option<&str>) -> Result<usize, String> {
        let Some(raw) = raw else {
            return Ok(0);
        };
        let first = raw.split(',').next().unwrap_or_default().trim();
        if first.is_empty() {
            return Err("CUDA_VISIBLE_DEVICES is set but its first entry is empty".into());
        }
        first.parse::<usize>().map_err(|_| {
            format!(
                "CUDA_VISIBLE_DEVICES={raw:?} does not start with a physical GPU ordinal; \
                 nvidia-smi cannot safely map UUID/MIG handles here"
            )
        })
    }

    /// Physical GPU ordinal sampled by `nvidia-smi` for Candle's rendered `cuda:0`.
    ///
    /// `CUDA_VISIBLE_DEVICES` remaps Candle's logical device indices but is ignored by `nvidia-smi`.
    /// Deriving the first visible physical ordinal here keeps render and probe on the same card by
    /// construction. Unset defaults to physical GPU 0. An empty, UUID, MIG handle, or otherwise
    /// non-numeric first entry panics instead of silently sampling the wrong card.
    pub fn probe_gpu() -> usize {
        let raw = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        parse_probe_gpu(raw.as_deref()).unwrap_or_else(|message| panic!("{message}"))
    }

    /// Device-level used VRAM (MiB) for GPU ordinal `gpu` via `nvidia-smi`, or `None` if the query
    /// fails.
    pub fn used_mib(gpu: usize) -> Option<u64> {
        let exe = crate::gpu::resolve_nvidia_smi()?;
        let out = Command::new(exe)
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                &gpu.to_string(),
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    /// Background sampler: from [`PeakSampler::start`] until [`PeakSampler::stop`], polls
    /// `used_mib(gpu)` every [`POLL`] and keeps the running max (MiB).
    pub struct PeakSampler {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicU64>,
        handle: Option<JoinHandle<()>>,
    }

    impl PeakSampler {
        /// Start sampling the physical GPU that Candle's logical `cuda:0` renders on. Prefer this for
        /// generation harnesses; explicit [`start`](Self::start) remains for sweep tools with their own
        /// GPU-selection environment variables.
        pub fn start_rendered() -> Self {
            Self::start(probe_gpu())
        }

        pub fn start(gpu: usize) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicU64::new(0));
            let (s, p) = (stop.clone(), peak.clone());
            let handle = thread::spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    if let Some(m) = used_mib(gpu) {
                        p.fetch_max(m, Ordering::Relaxed);
                    }
                    thread::sleep(POLL);
                }
                // One last sample after the stop signal — the true peak may land in the final window.
                if let Some(m) = used_mib(gpu) {
                    p.fetch_max(m, Ordering::Relaxed);
                }
            });
            Self {
                stop,
                peak,
                handle: Some(handle),
            }
        }

        /// Signal the sampler thread to stop, join it, and return the peak used VRAM (MiB).
        pub fn stop(mut self) -> u64 {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
            self.peak.load(Ordering::Relaxed)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::parse_probe_gpu;

        #[test]
        fn probe_gpu_defaults_to_zero_when_visibility_is_unset() {
            assert_eq!(parse_probe_gpu(None), Ok(0));
        }

        #[test]
        fn probe_gpu_uses_the_first_visible_physical_ordinal() {
            assert_eq!(parse_probe_gpu(Some(" 1,0 ")), Ok(1));
            assert_eq!(parse_probe_gpu(Some("7")), Ok(7));
        }

        #[test]
        fn probe_gpu_rejects_empty_uuid_mig_and_junk_without_guessing() {
            for raw in ["", " ,1", "GPU-a1b2", "MIG-GPU-a/b/c", "wat"] {
                assert!(parse_probe_gpu(Some(raw)).is_err(), "{raw:?} must fail");
            }
        }

        #[test]
        fn sampler_has_no_bare_nvidia_smi_spawn() {
            let source = include_str!("testkit.rs");
            let bare_spawn = ["Command", "::new(\"", "nvidia-smi", "\")"].concat();
            assert!(!source.contains(&bare_spawn));
        }
    }
}
