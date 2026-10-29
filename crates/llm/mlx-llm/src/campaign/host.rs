use super::*;

pub(crate) fn checked_git_revision(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    let revision = String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .trim()
        .to_string();
    if !output.status.success()
        || revision.len() != 40
        || !revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("cannot resolve immutable producer revision".into());
    }
    Ok(revision)
}

/// Normalize only an authenticated GitHub origin form; callers compare the result with a frozen
/// repository identity instead of stamping a path supplied by the environment.
pub fn canonical_github_repository(remote: &str) -> Result<String, String> {
    let remote = remote.trim().trim_end_matches('/');
    let path = if let Some(path) = remote.strip_prefix("git@github.com:") {
        path
    } else if let Some(path) = remote.strip_prefix("ssh://git@github.com/") {
        path
    } else if let Some(path) = remote.strip_prefix("https://github.com/") {
        path
    } else {
        return Err(format!(
            "repository origin is not an authenticated GitHub URL: {remote}"
        ));
    };
    let path = path.strip_suffix(".git").unwrap_or(path);
    let components = path.split('/').collect::<Vec<_>>();
    if components.len() != 2
        || components.iter().any(|component| {
            component.is_empty()
                || *component == "."
                || *component == ".."
                || component
                    .chars()
                    .any(|character| matches!(character, '?' | '#' | '\\'))
        })
    {
        return Err(format!(
            "repository origin has a non-canonical GitHub path: {remote}"
        ));
    }
    Ok(format!("github.com/{}/{}", components[0], components[1]))
}

/// Read and validate a worktree's `origin` against the source-owned repository identity.
pub fn checked_repository_identity(root: &Path, expected: &str) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .map_err(|e| e.to_string())?;
    let remote = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("cannot read repository origin".into());
    }
    let canonical = canonical_github_repository(&remote)?;
    if canonical != expected {
        return Err(format!(
            "repository origin is {canonical}, expected {expected}"
        ));
    }
    Ok(canonical)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockedMlxIdentity {
    pub version: String,
    pub source: String,
    pub revision: String,
}

pub fn locked_mlx_identity(lock: &[u8]) -> Result<LockedMlxIdentity, String> {
    let text = std::str::from_utf8(lock).map_err(|e| e.to_string())?;
    let quoted = |block: &str, key: &str| {
        block.lines().find_map(|line| {
            line.trim()
                .strip_prefix(&format!("{key} = \""))
                .and_then(|value| value.strip_suffix('\"'))
                .map(str::to_owned)
        })
    };
    for block in text.split("[[package]]") {
        if quoted(block, "name").as_deref() != Some("pmetal-mlx-rs") {
            continue;
        }
        let version = quoted(block, "version").ok_or("pmetal-mlx-rs lock entry has no version")?;
        let source = quoted(block, "source").ok_or("pmetal-mlx-rs lock entry has no source")?;
        let revision = source
            .rsplit_once('#')
            .map(|(_, revision)| revision.to_owned())
            .ok_or("pmetal-mlx-rs lock source has no immutable revision")?;
        let identity = LockedMlxIdentity {
            version,
            source,
            revision,
        };
        if !valid_locked_mlx_fields(&identity.version, &identity.source, &identity.revision) {
            return Err("pmetal-mlx-rs lock identity is not the frozen Git source/revision".into());
        }
        return Ok(identity);
    }
    Err("Cargo.lock does not contain pmetal-mlx-rs identity".into())
}

pub(super) fn required_campaign_env(name: &str) -> Result<String, String> {
    let value =
        std::env::var(name).map_err(|_| format!("missing required campaign environment {name}"))?;
    if value.trim().is_empty() || value.contains("product-probed-at-worker") {
        return Err(format!(
            "campaign environment {name} is not a real probed value"
        ));
    }
    Ok(value)
}

pub(super) fn probed_command(program: &str, args: &[&str], label: &str) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("probe {label}: {e}"))?;
    if !output.status.success() {
        return Err(format!("probe {label} failed"));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|e| format!("probe {label} output: {e}"))?
        .trim()
        .to_string();
    if value.is_empty() || value.contains("product-probed-at-worker") {
        return Err(format!("probe {label} returned no real value"));
    }
    Ok(value)
}

/// `CPU_Speed_Limit` from `pmset -g therm` when it reports CPU power status (`None` when it
/// prints only its no-history notes). The rest of the text is recorded verbatim, never parsed, so
/// unknown note lines are tolerated; only a malformed or contradictory limit is refused.
pub fn pmset_cpu_speed_limit(value: &str) -> Result<Option<u64>, String> {
    let mut limit = None;
    for line in value.lines() {
        let Some((name, raw)) = line.split_once('=') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("CPU_Speed_Limit") {
            continue;
        }
        let parsed = raw
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("pmset CPU_Speed_Limit is not a number: {:?}", raw.trim()))?;
        if limit.is_some_and(|prior| prior != parsed) {
            return Err("pmset reports contradictory CPU_Speed_Limit values".into());
        }
        limit = Some(parsed);
    }
    Ok(limit)
}

/// Row boundaries at which the host's power mode and thermal state are observed, in order.
pub const HOST_STATE_BOUNDARIES: [&str; 2] = ["row-start", "row-end"];

/// The host state recorded beside each measured timing sample, so throughput drift is attributable.
pub const TIMING_SAMPLE_HOST_BOUNDARY: &str = "timing-sample";

/// Normalized macOS energy modes (`pmset` `powermode` 0/1/2, or the older `lowpowermode`).
pub const POWER_MODES: [&str; 3] = ["automatic", "low-power", "high-power"];

/// `NSProcessInfo.thermalState` 0..=3.
pub const THERMAL_STATES: [&str; 4] = ["nominal", "fair", "serious", "critical"];

/// A real throttle signal: `NSProcessInfo.thermalState` serious or critical, or a `pmset`
/// `CPU_Speed_Limit` below 100%.
pub fn host_state_throttled(thermal_state: &str, cpu_speed_limit: Option<u64>) -> bool {
    matches!(thermal_state, "serious" | "critical")
        || cpu_speed_limit.is_some_and(|limit| limit < 100)
}

/// The host's power mode and thermal state at one boundary. Only a throttled row start refuses
/// the row; later states are recorded and flagged on the provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptHostState {
    pub boundary: String,
    pub captured_at: String,
    pub power_mode: String,
    /// `NSProcessInfo.thermalState`, one of [`THERMAL_STATES`].
    pub thermal_state: String,
    /// `pmset -g therm` `CPU_Speed_Limit` (percent), when reported.
    pub cpu_speed_limit: Option<u64>,
    /// The raw `pmset -g therm` text, recorded verbatim.
    pub pmset_thermal_raw: String,
    /// [`host_state_throttled`] of this state.
    pub throttled: bool,
}

impl ReceiptHostState {
    pub(super) fn validate(&self, boundary: &str) -> Result<(), String> {
        if self.boundary != boundary
            || !valid_utc_timestamp(&self.captured_at)
            || !POWER_MODES.contains(&self.power_mode.as_str())
            || !THERMAL_STATES.contains(&self.thermal_state.as_str())
            || self.pmset_thermal_raw.trim().is_empty()
            || pmset_cpu_speed_limit(&self.pmset_thermal_raw)? != self.cpu_speed_limit
            || self.throttled != host_state_throttled(&self.thermal_state, self.cpu_speed_limit)
        {
            return Err(format!(
                "host power/thermal provenance: {boundary} state is malformed or does not recompute"
            ));
        }
        Ok(())
    }
}

/// Whether the thermal state or the power mode changed from `start` across `later` observations.
/// A throttled later observation counts as a thermal change.
pub fn host_state_changes<'a>(
    start: &ReceiptHostState,
    later: impl IntoIterator<Item = &'a ReceiptHostState>,
) -> (bool, bool) {
    later
        .into_iter()
        .fold((false, false), |(thermal, power), state| {
            (
                thermal || state.thermal_state != start.thermal_state || state.throttled,
                power || state.power_mode != start.power_mode,
            )
        })
}

/// Refuse a row whose start is thermally throttled; every other state is recorded.
pub fn refuse_throttled_row_start(state: &ReceiptHostState) -> Result<(), String> {
    if state.throttled {
        return Err(format!(
            "row-start host is thermally throttled (thermalState={}, CPU_Speed_Limit={:?}); the row is refused before it runs",
            state.thermal_state, state.cpu_speed_limit
        ));
    }
    Ok(())
}

/// Normalize the active energy mode from `pmset -g` (its "Currently in use" settings): `powermode`
/// 0/1/2 is Automatic/Low Power/High Power; hosts without it report `lowpowermode` (and, on some
/// releases, `highpowermode`) 0/1. A missing, unknown, or contradictory setting is refused.
pub fn normalize_pmset_power_mode(value: &str) -> Result<String, String> {
    let mut in_use = false;
    let mut settings = std::collections::BTreeMap::new();
    for line in value.lines() {
        let line = line.trim();
        if line.eq_ignore_ascii_case("currently in use:") {
            in_use = true;
            continue;
        }
        if !in_use {
            continue;
        }
        let mut fields = line.split_whitespace();
        if let (Some(key @ ("powermode" | "lowpowermode" | "highpowermode")), Some(setting)) =
            (fields.next(), fields.next())
        {
            if settings.insert(key, setting).is_some() {
                return Err(format!("pmset reports {key} twice"));
            }
        }
    }
    let flag = |key: &str| match settings.get(key).copied() {
        None => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(other) => Err(format!("pmset reports unknown {key} {other}")),
    };
    let (low, high) = (flag("lowpowermode")?, flag("highpowermode")?);
    let mode = match settings.get("powermode").copied() {
        Some("0") => "automatic",
        Some("1") => "low-power",
        Some("2") => "high-power",
        Some(other) => return Err(format!("pmset reports unknown powermode {other}")),
        None => match (low, high) {
            (Some(true), Some(true)) => {
                return Err("pmset reports both low and high power mode".into())
            }
            (Some(true), _) => "low-power",
            (_, Some(true)) => "high-power",
            (Some(false), _) | (_, Some(false)) => "automatic",
            (None, None) => {
                return Err("pmset does not report the active power mode".into());
            }
        },
    };
    if (low == Some(true) && mode != "low-power") || (high == Some(true) && mode != "high-power") {
        return Err("pmset power mode settings contradict each other".into());
    }
    Ok(mode.into())
}

/// `NSProcessInfo.thermalState` (0 nominal, 1 fair, 2 serious, 3 critical) as its name.
pub fn normalize_process_thermal_state(state: i64) -> Result<String, String> {
    usize::try_from(state)
        .ok()
        .and_then(|index| THERMAL_STATES.get(index))
        .map(|name| (*name).to_string())
        .ok_or_else(|| format!("unknown process thermal state {state}"))
}

/// One boundary's host state from its raw probes: `pmset -g`, `pmset -g therm`, and
/// `NSProcessInfo.thermalState`. Throttling is recorded here, never refused.
pub fn host_state_from_probes(
    boundary: &str,
    captured_at: String,
    pmset: &str,
    pmset_thermal: &str,
    process_thermal_state: i64,
) -> Result<ReceiptHostState, String> {
    if !HOST_STATE_BOUNDARIES.contains(&boundary) && boundary != TIMING_SAMPLE_HOST_BOUNDARY {
        return Err(format!("unknown host-state boundary {boundary}"));
    }
    let power_mode = normalize_pmset_power_mode(pmset)?;
    let thermal_state = normalize_process_thermal_state(process_thermal_state)?;
    let cpu_speed_limit = pmset_cpu_speed_limit(pmset_thermal)?;
    Ok(ReceiptHostState {
        boundary: boundary.into(),
        captured_at,
        power_mode,
        throttled: host_state_throttled(&thermal_state, cpu_speed_limit),
        thermal_state,
        cpu_speed_limit,
        pmset_thermal_raw: pmset_thermal.into(),
    })
}

/// Probe this host's power mode and thermal state at `boundary`.
pub fn capture_host_state(boundary: &str) -> Result<ReceiptHostState, String> {
    let pmset = probed_command("pmset", &["-g"], "power mode")?;
    let pmset_thermal = probed_command("pmset", &["-g", "therm"], "thermal state")?;
    let process_thermal = process_thermal_state()?;
    host_state_from_probes(
        boundary,
        timestamp_now(),
        &pmset,
        &pmset_thermal,
        process_thermal,
    )
    .map_err(|e| format!("{boundary} host state: {e}"))
}

/// `[[NSProcessInfo processInfo] thermalState]` through the Objective-C runtime.
#[cfg(target_os = "macos")]
pub(super) fn process_thermal_state() -> Result<i64, String> {
    use std::ffi::{c_char, c_void};
    type Id = *mut c_void;
    type Sel = *mut c_void;
    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}
    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
    }
    // SAFETY: the class and selector names are C string literals; `objc_msgSend` is called
    // through the exact signatures of `+[NSProcessInfo processInfo]` (returns an object) and
    // `-[NSProcessInfo thermalState]` (returns an `NSInteger`), as the arm64/x86_64 ABIs require.
    unsafe {
        let class = objc_getClass(c"NSProcessInfo".as_ptr());
        if class.is_null() {
            return Err("NSProcessInfo is unavailable".into());
        }
        let send_object = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(Id, Sel) -> Id,
        >(objc_msgSend);
        let info = send_object(class, sel_registerName(c"processInfo".as_ptr()));
        if info.is_null() {
            return Err("NSProcessInfo returned no process info".into());
        }
        let send_integer = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(Id, Sel) -> isize,
        >(objc_msgSend);
        Ok(send_integer(info, sel_registerName(c"thermalState".as_ptr())) as i64)
    }
}

#[cfg(not(target_os = "macos"))]
pub(super) fn process_thermal_state() -> Result<i64, String> {
    Err("the process thermal state probe requires macOS".into())
}
