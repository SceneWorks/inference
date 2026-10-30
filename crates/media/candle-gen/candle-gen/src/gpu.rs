//! Shared VRAM-budget probe: the trusted-path `nvidia-smi` resolver used by the video-VAE decode
//! tilers (`candle-gen-seedvr2`, `candle-gen-wan`, `candle-gen-ltx`).
//!
//! # Why a resolver, not a bare `Command::new("nvidia-smi")` (sc-9014 / F-030)
//!
//! The budget probes used to spawn `Command::new("nvidia-smi")` **unqualified**. On Windows,
//! `CreateProcessW` searches the application directory (and the legacy current directory) *before*
//! `PATH`, so a `nvidia-smi.exe` planted next to the worker binary — or anywhere earlier on a
//! hijacked `PATH` — would run with the worker's privileges and silently rewrite the VRAM budget.
//! Because the probe runs on **every** budgeted decode, that is a repeatable, hard-to-observe
//! injection point.
//!
//! This module resolves `nvidia-smi` from a **trusted absolute location** (System32 /
//! `CUDA_PATH\bin` on Windows, the standard `/usr/bin` etc. on Unix) once, caches the result in a
//! [`OnceLock`], and only ever executes that absolute path. If no trusted binary is found the probe
//! degrades cleanly (returns `None` → the caller falls back to its env override / conservative
//! default) rather than executing whatever happens to be first on `PATH`.
//!
//! The whole query stays **best-effort**: any failure (no CUDA, sandboxed, non-zero exit, unparsable
//! output) yields `None`, exactly matching the previous behaviour when `nvidia-smi` was absent.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Cached resolution of the trusted `nvidia-smi` absolute path. `None` = not found at any trusted
/// location on this host (so the probe degrades to the caller's fallback). Resolved once per process.
static NVIDIA_SMI_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Candidate **absolute** locations for `nvidia-smi`, in priority order, for the current OS. Only
/// these trusted directories are ever considered — a bare `PATH` lookup is deliberately NOT among
/// them (that is the F-030 hijack vector). The CUDA-toolkit `bin` dirs are included because a CUDA
/// dev/deploy box always has the driver utilities there even if System32 somehow does not.
fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();

    #[cfg(windows)]
    {
        // The driver installs nvidia-smi.exe into System32 on every NVIDIA Windows host — the
        // canonical trusted location. `%SystemRoot%` (usually C:\Windows) is not attacker-writable.
        if let Some(root) = std::env::var_os("SystemRoot") {
            out.push(Path::new(&root).join("System32").join("nvidia-smi.exe"));
        }
        // CUDA toolkit bin (dev boxes): CUDA_PATH\bin\nvidia-smi.exe.
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            out.push(Path::new(&cuda).join("bin").join("nvidia-smi.exe"));
        }
    }

    #[cfg(not(windows))]
    {
        // The NVIDIA Linux driver installs nvidia-smi into /usr/bin (some distros /usr/local/bin).
        out.push(PathBuf::from("/usr/bin/nvidia-smi"));
        out.push(PathBuf::from("/usr/local/bin/nvidia-smi"));
        // CUDA toolkit bin, if the env points at it.
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            out.push(Path::new(&cuda).join("bin").join("nvidia-smi"));
        }
    }

    out
}

/// Pick the first candidate that exists on disk. Split out so it is unit-testable without touching
/// the real host: [`resolve_nvidia_smi`] wraps this over [`candidate_paths`].
fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

/// The trusted absolute `nvidia-smi` path for this host, or `None` if none of the trusted locations
/// contains it. Cached in a [`OnceLock`] — resolved (and the filesystem probed) at most once.
pub fn resolve_nvidia_smi() -> Option<&'static Path> {
    NVIDIA_SMI_PATH
        .get_or_init(|| first_existing(&candidate_paths()))
        .as_deref()
}

/// Total VRAM (GiB) of the smallest visible CUDA device, read from a **trusted absolute**
/// `nvidia-smi` (never a bare `PATH` lookup — see the module note / sc-9014). The MIN across devices
/// is conservative on a heterogeneous box. `None` when no trusted `nvidia-smi` exists or the query
/// fails — the caller then falls back to its env override / conservative default.
///
/// This is the shared implementation the SeedVR2 / Wan / LTX budget probes route through (the F-030
/// de-dup that had been tracked as a follow-up).
pub fn nvidia_smi_min_total_gib() -> Option<f64> {
    query_min_gib("memory.total")
}

/// **Free** VRAM (GiB) of the least-free visible CUDA device — what is ACTUALLY unallocated on the
/// device *right now*, read from the same trusted absolute `nvidia-smi` (never a bare `PATH` lookup;
/// sc-9014 / F-030). Because `nvidia-smi memory.free` is the driver's live `total − used`, this value
/// already nets out **resident model weights + the cudarc pool + anything else on the device**, i.e.
/// it IS `(total − resident)`. A decode tiler can therefore budget against it instead of `0.85×TOTAL`
/// and stop over-budgeting on top of what the denoise left resident (sc-12734).
///
/// The MIN across devices is conservative on a heterogeneous box. `None` when no trusted `nvidia-smi`
/// exists or the query fails — the caller then falls back to its env override / conservative default.
///
/// NOTE: on Windows WDDM the driver reports **device-level** used/free (per-process is `[N/A]`), so a
/// large *other* allocation on the same GPU shrinks this. That is the intended behaviour: the tiler
/// should fit whatever is genuinely free, not an optimistic per-process figure.
pub fn nvidia_smi_min_free_gib() -> Option<f64> {
    query_min_gib("memory.free")
}

/// Current free VRAM on Candle's logical device zero. Numeric visibility selects the
/// physical index (under the worker's PCI_BUS_ID ordering); UUID visibility selects
/// exactly one matching driver UUID. nvidia-smi ignores CUDA_VISIBLE_DEVICES, so query
/// metadata and resolve the first entry explicitly. Missing/ambiguous mappings return
/// None; an unrelated card's memory is never substituted for the selected card.
pub fn nvidia_smi_rendered_free_gib() -> Option<f64> {
    let raw = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    selected_free_gib(
        raw.as_deref(),
        &query_gpu_field("index,uuid,memory.free", None)?,
    )
}

fn valid_gpu_uuid(uuid: &str) -> bool {
    let Some(body) = uuid.strip_prefix("GPU-") else {
        return false;
    };
    body.len() == 36
        && body.bytes().enumerate().all(|(i, byte)| {
            if [8, 13, 18, 23].contains(&i) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

/// Pure selected-device resolution, also used by the metadata regression tests.
fn selected_free_gib(visible: Option<&str>, inventory: &str) -> Option<f64> {
    let selected = visible.unwrap_or("0").split(',').next()?.trim();
    let ordinal = selected.parse::<usize>().ok();
    let uuid_prefix = selected.strip_prefix("GPU-").filter(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_hexdigit() || c == b'-')
    });
    if ordinal.is_none() && uuid_prefix.is_none() {
        return None;
    }
    let mut matching = Vec::new();
    for row in inventory.lines().filter(|row| !row.trim().is_empty()) {
        let fields = row.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 3 {
            return None;
        }
        let index = fields[0].parse::<usize>().ok()?;
        if !valid_gpu_uuid(fields[1]) {
            return None;
        }
        if ordinal.map_or_else(|| fields[1].starts_with(selected), |wanted| index == wanted) {
            let mib = fields[2].parse::<f64>().ok()?;
            if !mib.is_finite() || mib < 0.0 {
                return None;
            }
            matching.push(mib / 1024.0);
        }
    }
    (matching.len() == 1).then(|| matching[0])
}

/// Effective allocation headroom for Candle on the rendered GPU: physical driver-free memory plus
/// bytes reserved by this process's current async pool that are no longer live and can be reused.
///
/// Dropping a sequential phase returns its allocations to cudarc's in-process pool, not to the
/// driver. Consequently `nvidia-smi memory.free` alone undercounts what the following VAE allocation
/// can use and may reject a viable tile after a large UNet drop. The current pool reports both
/// `RESERVED_MEM_CURRENT` and `USED_MEM_CURRENT`; their difference is already owned by this process
/// and immediately reusable without another physical allocation.
///
/// The pool query is best-effort. Non-CUDA builds and driver failures retain the conservative
/// `nvidia-smi` value. CUDA logical device 0 is the rendered device after `CUDA_VISIBLE_DEVICES`
/// remapping, matching Candle's `default_device`.
pub fn rendered_effective_free_gib() -> Option<f64> {
    nvidia_smi_rendered_free_gib()
        .map(|driver_free| effective_free_gib(driver_free, cuda_current_pool_reusable_gib()))
}

fn effective_free_gib(driver_free_gib: f64, reusable_pool_gib: Option<f64>) -> f64 {
    driver_free_gib + reusable_pool_gib.unwrap_or(0.0).max(0.0)
}

#[cfg(feature = "cuda")]
fn cuda_current_pool_reusable_gib() -> Option<f64> {
    // The pool candle actually allocates from (cuda_mempool trap 1); in production no custom pool is
    // installed, so this is the default async pool (sc-27280).
    let pool = crate::cuda_mempool::MemPool::device_current(0)?;
    Some(pool.reserved()?.saturating_sub(pool.used()?) as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(not(feature = "cuda"))]
fn cuda_current_pool_reusable_gib() -> Option<f64> {
    None
}

/// Run the trusted-absolute `nvidia-smi` for a single `--query-gpu=<field>` (one MiB value per line,
/// one line per GPU) and reduce to the MIN across GPUs in GiB. Shared by the total and free probes so
/// the trusted-path resolution + parsing lives in exactly one place.
fn query_min_gib(field: &str) -> Option<f64> {
    parse_min_mib_line_gib(&query_gpu_field(field, None)?)
}

/// The one `nvidia-smi --query-gpu=<field> --format=csv,noheader,nounits [-i <gpu>]` spawn (sc-27374):
/// the trusted absolute executable, the query, and the exit-status check. Returns stdout (lossy UTF-8)
/// on success; `None` when no trusted `nvidia-smi` exists, the spawn fails, or it exits nonzero.
/// Callers own the parsing. Shared with `testkit::used_mib`.
pub(crate) fn query_gpu_field(field: &str, gpu: Option<usize>) -> Option<String> {
    let exe = resolve_nvidia_smi()?;
    let mut command = Command::new(exe);
    command.args([
        &format!("--query-gpu={field}"),
        "--format=csv,noheader,nounits",
    ]);
    if let Some(gpu) = gpu {
        command.args(["-i", &gpu.to_string()]);
    }
    let out = command.output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse `--query-gpu=<memory field> --format=csv,noheader,nounits` output (one MiB value per line,
/// one line per GPU) into the MIN across GPUs, in GiB. Field-agnostic (used for both `memory.total`
/// and `memory.free`). Split out for unit testing.
fn parse_min_mib_line_gib(text: &str) -> Option<f64> {
    let min_mb = text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .filter(|&mb| mb > 0.0)
        .fold(f64::INFINITY, f64::min);
    min_mb.is_finite().then_some(min_mb / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_paths_are_all_absolute_and_trusted() {
        // Every candidate must be an ABSOLUTE path (the whole point of F-030): a relative or bare
        // name would reintroduce the process-search-order hijack. There must also be at least one
        // candidate on a normal host (SystemRoot on Windows / /usr/bin on Unix are always present).
        let cands = candidate_paths();
        assert!(!cands.is_empty(), "expected at least one trusted candidate");
        for p in &cands {
            assert!(p.is_absolute(), "candidate must be absolute, got {p:?}");
            assert!(
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("nvidia-smi")),
                "candidate must end in the nvidia-smi binary, got {p:?}"
            );
        }
    }

    #[test]
    fn first_existing_picks_the_first_present_path() {
        // With a made-up non-existent dir first and a guaranteed-existent one second, the resolver
        // must skip the missing one and return the real one — proving it validates on disk rather
        // than blindly returning candidate[0] (or falling through to PATH).
        let missing = PathBuf::from(if cfg!(windows) {
            r"C:\definitely\not\here\nvidia-smi.exe"
        } else {
            "/definitely/not/here/nvidia-smi"
        });
        // A path that is guaranteed to exist on the test host.
        let present = std::env::current_exe().expect("current exe path");
        let picked = first_existing(&[missing.clone(), present.clone()]);
        assert_eq!(picked.as_deref(), Some(present.as_path()));
    }

    #[test]
    fn first_existing_is_none_when_nothing_exists() {
        // No trusted candidate present → None (degrade cleanly to the caller's fallback, never PATH).
        let a = PathBuf::from(if cfg!(windows) {
            r"C:\nope\a\nvidia-smi.exe"
        } else {
            "/nope/a/nvidia-smi"
        });
        let b = PathBuf::from(if cfg!(windows) {
            r"C:\nope\b\nvidia-smi.exe"
        } else {
            "/nope/b/nvidia-smi"
        });
        assert_eq!(first_existing(&[a, b]), None);
    }

    #[test]
    fn parse_min_mib_line_gib_takes_min_across_gpus() {
        // Two GPUs: 24576 MiB and 49152 MiB → min = 24576 MiB = 24 GiB. (Same reducer for total and
        // free: the MIN is the conservative choice for both — smallest total / least-free device.)
        let got = parse_min_mib_line_gib("24576\n49152\n").expect("parses");
        assert!((got - 24.0).abs() < 1e-9, "expected 24.0 GiB, got {got}");
    }

    #[test]
    fn parse_min_mib_line_gib_none_on_empty_or_garbage() {
        assert_eq!(parse_min_mib_line_gib(""), None);
        assert_eq!(parse_min_mib_line_gib("no gpus\n\n"), None);
        // Zero/negative are filtered out (a driver quirk) → None, not 0.
        assert_eq!(parse_min_mib_line_gib("0\n"), None);
    }

    #[test]
    fn effective_free_adds_only_reusable_pool_capacity() {
        assert!((effective_free_gib(7.5, Some(4.25)) - 11.75).abs() < 1e-9);
        assert_eq!(effective_free_gib(7.5, None), 7.5);
        assert_eq!(effective_free_gib(7.5, Some(-1.0)), 7.5);
    }

    #[test]
    fn selected_device_metadata_resolves_numeric_uuid_and_first_visible() {
        let uuid0 = "GPU-00000000-0000-0000-0000-000000000000";
        let uuid1 = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
        let inventory = format!("0, {uuid0}, 1024\n1, {uuid1}, 61440\n");
        assert_eq!(selected_free_gib(None, &inventory), Some(1.0));
        assert_eq!(selected_free_gib(Some("1"), &inventory), Some(60.0));
        assert_eq!(selected_free_gib(Some(" 1,0 "), &inventory), Some(60.0));
        assert_eq!(selected_free_gib(Some(uuid1), &inventory), Some(60.0));
        assert_eq!(
            selected_free_gib(Some(&format!("{uuid1},{uuid0}")), &inventory),
            Some(60.0)
        );
        assert_eq!(
            selected_free_gib(Some("GPU-e4b79931"), &inventory),
            Some(60.0)
        );
        for selected in ["", "2", "GPU-missing", "MIG-abc", "-1", "GPU-"] {
            assert_eq!(selected_free_gib(Some(selected), &inventory), None);
        }
        let duplicate = format!("{inventory}2, {uuid1}, 1234\n");
        assert_eq!(selected_free_gib(Some(uuid1), &duplicate), None);
        assert_eq!(selected_free_gib(Some("1"), "1, invalid, 61440"), None);
        assert_eq!(
            selected_free_gib(Some(uuid1), &format!("1, {uuid1}, 0")),
            Some(0.0)
        );
        assert_eq!(
            selected_free_gib(Some(uuid1), &format!("1, {uuid1}, NaN")),
            None
        );
    }

    /// Real-hardware proof (sc-13298) that `nvidia_smi_rendered_free_gib` samples the PINNED device,
    /// not the all-GPU min. Needs a box with ≥2 GPUs at different free levels; run manually, once per
    /// card: `CUDA_VISIBLE_DEVICES=0 cargo test ... rendered_free_reads_the_pinned_device -- --ignored
    /// --nocapture`, then `=1`. `rendered_free` must track the selected card (so on the MORE-free card
    /// it is STRICTLY above the all-GPU min — the poison the old min-based probe suffered), while
    /// `nvidia_smi_min_free_gib` stays the least-free card regardless of `CUDA_VISIBLE_DEVICES`.
    #[test]
    #[ignore = "needs a real multi-GPU box; run per-card with CUDA_VISIBLE_DEVICES set + --ignored"]
    fn rendered_free_reads_the_pinned_device_not_the_all_gpu_min() {
        let rendered = nvidia_smi_rendered_free_gib().expect("rendered free (needs nvidia-smi)");
        let min = nvidia_smi_min_free_gib().expect("all-gpu min free");
        let raw = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let inventory = query_gpu_field("index,uuid,memory.free", None).expect("GPU metadata");
        let pinned_direct = selected_free_gib(raw.as_deref(), &inventory).expect("selected device");
        let cvd = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into());
        eprintln!(
            "CUDA_VISIBLE_DEVICES={cvd}: rendered_free={rendered:.3} GiB | \
             pinned-direct={pinned_direct:.3} GiB | all-gpu min={min:.3} GiB"
        );
        // The real guard: rendered_free must equal the PINNED device's free, not the all-GPU min. This
        // FAILS if the probe is wired back to `nvidia_smi_min_free_gib` on any box where the pinned card
        // is not the least-free one (the sc-13298 regression) — pin the MORE-free card to exercise it.
        assert!(
            (rendered - pinned_direct).abs() < 0.05,
            "rendered_free {rendered} must equal the pinned device's free {pinned_direct}, \
             not the all-gpu min {min}"
        );
    }
}
