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

/// **Free** VRAM (GiB) of the GPU the render is PINNED to — Candle's `cuda:0`, i.e. the FIRST entry of
/// `CUDA_VISIBLE_DEVICES` (unset ⇒ physical 0), resolved the same way `testkit::probe_gpu` derives the
/// sampled ordinal so a decode and the peak probe stay on the same card. Reads a trusted absolute
/// `nvidia-smi -i <ordinal> --query-gpu=memory.free` (never a bare `PATH` lookup; sc-9014 / F-030).
///
/// Unlike [`nvidia_smi_min_free_gib`], a busy CO-TENANT GPU cannot shrink this: a decode pinned to an
/// idle card budgets against THAT card's free, not the least-free card on the box (sc-13298). The
/// worker pins one GPU per render via `CUDA_VISIBLE_DEVICES` (worker `supervisor.rs`), so `cuda:0` IS
/// that device and its CVD entry IS the physical ordinal `-i` wants.
///
/// ⚠️ The `-i <ordinal>` mapping assumes Candle's `CUDA_VISIBLE_DEVICES` ordinal equals nvidia-smi's
/// PCI-bus index — true when `CUDA_DEVICE_ORDER=PCI_BUS_ID` or the GPUs are HOMOGENEOUS (the prod boxes
/// run 2× identical cards, where the default `FASTEST_FIRST` order tie-breaks on PCI id and coincides).
/// On a HETEROGENEOUS box under the default order the two can diverge and this could sample the wrong
/// card (over-budgeting if that card has more free); that is the SAME assumption the existing
/// `testkit::probe_gpu` / `used_mib` already make, so this stays consistent with them rather than
/// forking a second contract. Export `CUDA_DEVICE_ORDER=PCI_BUS_ID` to make it exact on mixed hardware.
///
/// Falls back to [`nvidia_smi_min_free_gib`] (the all-GPU min — the prior behaviour, conservative) when
/// the pinned ordinal cannot be resolved (an empty "no devices" value, or a UUID/MIG handle nvidia-smi
/// cannot map) or the per-device query fails, so the safe direction is preserved rather than lost.
/// `None` only when even that min has no trusted `nvidia-smi` to read.
pub fn nvidia_smi_rendered_free_gib() -> Option<f64> {
    rendered_gpu_ordinal()
        .and_then(query_free_gib_for)
        .or_else(nvidia_smi_min_free_gib)
}

/// The physical GPU ordinal Candle's `cuda:0` renders on, from `CUDA_VISIBLE_DEVICES` (`std::env`).
/// The production-path twin of `testkit::probe_gpu`, delegating to [`parse_rendered_ordinal`].
fn rendered_gpu_ordinal() -> Option<usize> {
    parse_rendered_ordinal(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
}

/// Pure core of [`rendered_gpu_ordinal`] — split out so the `CUDA_VISIBLE_DEVICES` parsing is unit-
/// testable without mutating process-global env (mirrors `testkit::parse_probe_gpu`). Unset (`None`) ⇒
/// physical 0; a numeric FIRST entry ⇒ that ordinal (`CUDA_VISIBLE_DEVICES` remaps Candle's logical
/// indices but nvidia-smi ignores it, so `cuda:0` = the first visible physical ordinal). An empty /
/// UUID / MIG / otherwise non-numeric first entry ⇒ `None`, so the caller falls back to the all-GPU
/// min rather than sample the wrong card. Returns `None` instead of panicking (unlike the test-harness
/// probe) because this runs inside a live render.
fn parse_rendered_ordinal(raw: Option<&str>) -> Option<usize> {
    match raw {
        None => Some(0),
        Some(raw) => raw
            .split(',')
            .next()
            .unwrap_or_default()
            .trim()
            .parse::<usize>()
            .ok(),
    }
}

/// **Free** VRAM (GiB) of a SINGLE physical GPU `gpu`, via a trusted absolute
/// `nvidia-smi -i <gpu> --query-gpu=memory.free`. `None` when no trusted `nvidia-smi` exists, the query
/// fails (e.g. `gpu` is out of range), or the value is missing. Reuses [`parse_min_mib_line_gib`] — with
/// `-i` the output is a single line, so its "MIN across lines" is just that one device's value.
fn query_free_gib_for(gpu: usize) -> Option<f64> {
    let exe = resolve_nvidia_smi()?;
    let out = Command::new(exe)
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            &gpu.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_min_mib_line_gib(&String::from_utf8_lossy(&out.stdout))
}

/// Run the trusted-absolute `nvidia-smi` for a single `--query-gpu=<field>` (one MiB value per line,
/// one line per GPU) and reduce to the MIN across GPUs in GiB. Shared by the total and free probes so
/// the trusted-path resolution + parsing lives in exactly one place.
fn query_min_gib(field: &str) -> Option<f64> {
    let exe = resolve_nvidia_smi()?;
    let out = Command::new(exe)
        .args([
            &format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_min_mib_line_gib(&String::from_utf8_lossy(&out.stdout))
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
    fn parse_rendered_ordinal_maps_cuda_visible_devices() {
        // Unset ⇒ every device visible; Candle's cuda:0 is physical 0.
        assert_eq!(parse_rendered_ordinal(None), Some(0));
        // The worker pins one GPU: CUDA_VISIBLE_DEVICES=<gpu_id> ⇒ that physical ordinal.
        assert_eq!(parse_rendered_ordinal(Some("1")), Some(1));
        // Multiple visible ⇒ cuda:0 is the FIRST entry (what `-i` must sample).
        assert_eq!(parse_rendered_ordinal(Some("2,3")), Some(2));
        assert_eq!(parse_rendered_ordinal(Some(" 1 ")), Some(1));
        // Set-but-empty ("no devices"), a UUID, or a MIG handle cannot map to an `-i` ordinal ⇒ None,
        // so the caller falls back to the all-GPU min rather than sample the wrong card. Unlike
        // `testkit::probe_gpu`, this NEVER panics — it runs inside a live render.
        assert_eq!(parse_rendered_ordinal(Some("")), None);
        assert_eq!(parse_rendered_ordinal(Some("GPU-1a2b3c4d")), None);
        assert_eq!(parse_rendered_ordinal(Some("MIG-abc")), None);
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
        let ordinal = rendered_gpu_ordinal().expect("a resolvable pinned ordinal");
        let pinned_direct = query_free_gib_for(ordinal).expect("pinned device direct free");
        let cvd = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into());
        eprintln!(
            "CUDA_VISIBLE_DEVICES={cvd} (ordinal {ordinal}): rendered_free={rendered:.3} GiB | \
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
