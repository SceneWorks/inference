use super::*;

/// The runtime admission a row ran under. It is recorded in every receipt so the stated child cap,
/// the static planning estimate, and the [`campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE`]
/// decision (rule, estimate source and bytes, host measurement) travel with the evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptAdmission {
    pub mode: String,
    pub rule: String,
    pub child_footprint_cap_bytes: u64,
    pub host_free_reserve_bytes: u64,
    pub static_footprint_floor_bytes: u64,
    /// Where [`Self::estimate_bytes`] came from (see [`campaign_supervisor::AdmissionEstimate`]).
    pub estimate_source: String,
    /// The unit's estimated peak: admission required host available memory of at least this plus
    /// the reserve. Never below the static floor, never above the cap.
    pub estimate_bytes: u64,
    /// Every vm_stat component of the pre-spawn host measurement the row was admitted (or
    /// refused) on; `availableBytes` is the measure the rule compared. Absent only from a
    /// parent-side static admission computed before any measurement; a receipt requires it (see
    /// [`Self::validate_admitted`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory_components: Option<campaign_supervisor::HostMemory>,
}

impl ReceiptAdmission {
    /// The recorded cap and reserve must be the captured campaign policy's, not caller values.
    pub fn validate_against(&self, policy: &CampaignSafetyPolicy) -> Result<(), String> {
        self.validate_admitted()?;
        if self.child_footprint_cap_bytes != policy.child_footprint_cap_bytes
            || self.host_free_reserve_bytes != policy.host_free_reserve_bytes
        {
            return Err("row admission cap/reserve differs from the captured safety policy".into());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.mode != RUNTIME_GUARDED_ADMISSION
            || self.rule != campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE
            || self.child_footprint_cap_bytes == 0
            || self.host_free_reserve_bytes == 0
            || self
                .host_free_reserve_bytes
                .checked_add(self.child_footprint_cap_bytes)
                .is_none()
            || self.static_footprint_floor_bytes == 0
            || self.static_footprint_floor_bytes > self.child_footprint_cap_bytes
            || self.estimate_bytes < self.static_footprint_floor_bytes
            || self
                .estimate()
                .validate(self.child_footprint_cap_bytes)
                .is_err()
        {
            return Err(format!(
                "row admission is not a runtime-guarded cap with its {} decision",
                campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE
            ));
        }
        if let Some(host) = &self.host_memory_components {
            host.validate()?;
        }
        Ok(())
    }

    /// A receipt's admission: the row ran, so its recorded host measurement must exist,
    /// recompute, and cover estimate plus reserve.
    pub fn validate_admitted(&self) -> Result<(), String> {
        self.validate()?;
        self.host_memory_components
            .as_ref()
            .ok_or("row admission lacks its host memory components")?
            .validate_admits(self.host_free_reserve_bytes, self.estimate_bytes)
    }

    /// The estimate the supervisor admits the row on.
    pub fn estimate(&self) -> campaign_supervisor::AdmissionEstimate {
        campaign_supervisor::AdmissionEstimate {
            source: self.estimate_source.clone(),
            bytes: self.estimate_bytes,
        }
    }

    /// The same admission with the supervisor's pre-spawn measurement (for unaccepted records).
    pub fn with_host_memory(&self, host: Option<campaign_supervisor::HostMemory>) -> Self {
        Self {
            host_memory_components: host,
            ..self.clone()
        }
    }
}

/// Admit a row by its runtime guards rather than by a static whole-process MLX peak proof, which is
/// unobtainable for lazy long-context graphs. The row is admitted only when the mandatory policy
/// configures every supervisor guard (deadline, sampling, termination grace, host RAM reserve,
/// child `phys_footprint` watchdog cap) and the conservative static floor still fits under the
/// cap. The floor is the row's admission estimate (`estimate_source` names it): the supervisor
/// refuses before spawn unless host available RAM ([`campaign_supervisor::HostMemory`]) covers
/// estimate plus reserve, and terminates the child if the cap or reserve watchdog trips; neither
/// outcome is ever an accepted row. No SC-20671/76/77 unit has an already-completed identical
/// unit to raise the estimate from (each runs once per resume directory, and resume skips accepted
/// units), so no measured peak enters it.
pub(crate) fn runtime_guarded_admission(
    policy: &CampaignSafetyPolicy,
    static_footprint_floor_bytes: u64,
    estimate_source: &str,
) -> Result<ReceiptAdmission, String> {
    policy.validate()?;
    let estimate = campaign_supervisor::AdmissionEstimate::resolve(
        Some((estimate_source, static_footprint_floor_bytes)),
        None,
        policy.child_footprint_cap_bytes,
    )?;
    let admission = ReceiptAdmission {
        mode: RUNTIME_GUARDED_ADMISSION.into(),
        rule: campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE.into(),
        child_footprint_cap_bytes: policy.child_footprint_cap_bytes,
        host_free_reserve_bytes: policy.host_free_reserve_bytes,
        static_footprint_floor_bytes,
        estimate_source: estimate.source,
        estimate_bytes: estimate.bytes,
        host_memory_components: None,
    };
    admission.validate()?;
    Ok(admission)
}

/// Worker side: the runtime-guarded admission plus the decision its supervisor admitted it on
/// (handed over in [`campaign_supervisor::HOST_MEMORY_ADMISSION_ENV`]), so the worker's own
/// receipt records every component of the decision. The supervisor's estimate must be exactly the
/// one the worker derives from the same sealed inputs.
pub(crate) fn supervised_admission(
    policy: &CampaignSafetyPolicy,
    static_footprint_floor_bytes: u64,
    estimate_source: &str,
) -> Result<ReceiptAdmission, String> {
    let admission =
        runtime_guarded_admission(policy, static_footprint_floor_bytes, estimate_source)?;
    let decision = campaign_supervisor::admitted_host(
        policy.host_free_reserve_bytes,
        policy.child_footprint_cap_bytes,
    )?;
    if decision.estimate() != admission.estimate() {
        return Err(format!(
            "supervisor admitted the worker on estimate {} bytes ({}), not the worker's own {} bytes ({})",
            decision.estimate_bytes,
            decision.estimate_source,
            admission.estimate_bytes,
            admission.estimate_source
        ));
    }
    let admission = admission.with_host_memory(Some(decision.host_memory));
    admission.validate_admitted()?;
    Ok(admission)
}

/// First attempt prefix whose stdout, stderr, and unaccepted record are all absent. The supervisor
/// removes its logs when a spawn fails, so the record alone must also reserve a prefix.
pub(crate) fn unused_attempt_prefix(logs: &Path, slug: &str) -> Option<String> {
    (0_u64..)
        .map(|attempt| format!("{slug}.attempt-{attempt}"))
        .find(|prefix| {
            ["stdout.log", "stderr.log", "unaccepted.json"]
                .iter()
                .all(|suffix| !logs.join(format!("{prefix}.{suffix}")).exists())
        })
}

/// Why a row stopped: reason, detail, the owned child PID (after spawn), and the live host sample
/// that tripped the host-reserve watchdog, when one did.
pub(crate) type UnacceptedStop<'a> = (
    &'a str,
    &'a str,
    Option<u32>,
    Option<&'a campaign_supervisor::HostMemory>,
);

/// Record an unaccepted row and return the row's own failure. A record-write failure is appended
/// to, never substituted for, the original refusal/abort/exit reason.
pub(crate) fn unaccepted_row_error(
    path: &Path,
    kind: &str,
    slug: &str,
    admission: &ReceiptAdmission,
    stop: UnacceptedStop<'_>,
    failure: String,
) -> String {
    match write_unaccepted_row_record(path, kind, slug, admission, stop) {
        Ok(()) => format!("{failure}; not accepted ({})", path.display()),
        Err(error) => format!(
            "{failure}; not accepted, and its record {} could not be written: {error}",
            path.display()
        ),
    }
}

/// Persist a refused (pre-spawn), aborted (post-spawn guard), or failed row as a sealed, explicitly
/// unaccepted log record. It never becomes a receipt and the resume validator never reads it.
pub(crate) fn write_unaccepted_row_record(
    path: &Path,
    kind: &str,
    slug: &str,
    admission: &ReceiptAdmission,
    stop: UnacceptedStop<'_>,
) -> Result<(), String> {
    let (reason, detail, pid, watchdog_host_memory) = stop;
    let mut record = serde_json::json!({
        "schemaVersion": 1,
        "kind": kind,
        "coordinate": slug,
        "accepted": false,
        "outcome": match (reason, pid) {
            ("ChildExit", _) => "failed",
            (_, None) => "refused",
            (_, Some(_)) => "aborted",
        },
        "reason": reason,
        "detail": detail,
        "pid": pid,
        "admission": admission,
    });
    if let Some(sample) = watchdog_host_memory {
        record["watchdogHostMemory"] = serde_json::to_value(sample).map_err(|e| e.to_string())?;
    }
    let bytes = seal_json(&record)?.0;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("write unaccepted row record: {e}"))?;
    std::io::Write::write_all(&mut file, &bytes).map_err(|e| e.to_string())
}

pub(super) fn valid_locked_mlx_fields(version: &str, source: &str, revision: &str) -> bool {
    !version.is_empty()
        && revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && source == format!("git+{PMETAL_MLX_REPOSITORY}?rev={revision}#{revision}")
}

pub(super) fn valid_locked_mlx_identity(provenance: &ReceiptProvenance) -> bool {
    valid_locked_mlx_fields(
        &provenance.mlx_version,
        &provenance.mlx_source,
        &provenance.mlx_revision,
    )
}
