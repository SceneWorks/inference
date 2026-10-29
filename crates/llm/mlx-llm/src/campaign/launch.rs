use super::*;

/// A durable "stopped by operator before row N" status. Rows `0..before_row` are accepted (or
/// resumed) in the resume directory; row `before_row` and later were never started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorStop {
    pub before_row: usize,
    pub row: String,
    pub rows_total: usize,
    /// The sealed status record written under the resume directory's `logs/`.
    pub record: PathBuf,
}

/// How a campaign parent invocation ended without an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CampaignOutcome {
    Completed,
    StoppedByOperator(OperatorStop),
    /// Every row was attempted but these were refused before spawn; nothing is published.
    IncompleteWithRefusals(Vec<String>),
}

impl CampaignOutcome {
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Completed => 0,
            Self::StoppedByOperator(_) => OPERATOR_STOP_EXIT_CODE,
            Self::IncompleteWithRefusals(_) => 1,
        }
    }
}

/// The operator stop files a parent honours: always `<resume-dir>/STOP`, plus `--stop-file <path>`
/// when given (either present stops the campaign). Stop files are operator control only: never
/// part of a resume identity and never forwarded to a worker.
pub(crate) fn operator_stop_files(
    args: &[String],
    resume_dir: &Path,
) -> Result<Vec<PathBuf>, String> {
    let mut files = vec![resume_dir.join(OPERATOR_STOP_FILE_NAME)];
    if args.iter().any(|arg| arg == "--stop-file") {
        let custom = PathBuf::from(required_flag(args, "--stop-file")?);
        if !files.contains(&custom) {
            files.push(custom);
        }
    }
    Ok(files)
}

/// Whether resume-directory entry `name` is an operator stop file rather than campaign state.
pub(crate) fn is_operator_stop_entry(
    resume_dir: &Path,
    stop_files: &[PathBuf],
    name: &str,
) -> bool {
    name == OPERATOR_STOP_FILE_NAME || stop_files.contains(&resume_dir.join(name))
}

/// A stop file is present when anything (even a dangling symlink) exists at its path. Only
/// `NotFound` means absent; any other stat failure is an error, never a silent "keep going".
pub(super) fn operator_stop_present(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "stat operator stop file {}: {error}",
            path.display()
        )),
    }
}

/// Create `path` exclusively and write `bytes` durably (fsync).
pub(super) fn write_new_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    std::io::Write::write_all(&mut file, bytes)?;
    file.sync_all()
}

/// Checked by a campaign parent only between rows, immediately before it would spawn row
/// `before_row`'s worker; a running worker is never signalled (killing an MLX render mid command
/// buffer can wedge the GPU). When any stop file exists this writes a sealed, never-overwritten
/// `logs/operator-stop.attempt-<n>.json` status record plus its `.sha256` sidecar and returns the
/// stop.
pub(crate) fn operator_stop_before_row(
    stop_files: &[PathBuf],
    logs: &Path,
    kind: &str,
    before_row: usize,
    row: &str,
    rows_total: usize,
) -> Result<Option<OperatorStop>, String> {
    let mut present = Vec::new();
    for path in stop_files {
        if operator_stop_present(path)? {
            present.push(path.display().to_string());
        }
    }
    if present.is_empty() {
        return Ok(None);
    }
    fs::create_dir_all(logs).map_err(|e| format!("create operator stop log directory: {e}"))?;
    let value = serde_json::json!({
        "schemaVersion": 1,
        "kind": kind,
        "status": "stopped-by-operator",
        "beforeRow": before_row,
        "beforeRowSlug": row,
        "rowsTotal": rows_total,
        "rowsAccepted": before_row,
        "stopFiles": present,
        "recordedAt": timestamp_now(),
        "resume": "remove the stop file and rerun the same command with the same resume directory",
    });
    let (bytes, sha256) = seal_json(&value)?;
    let mut attempt = 0_u64;
    let record = loop {
        let name = format!("operator-stop.attempt-{attempt}.json");
        let path = logs.join(&name);
        match write_new_durable(&path, &bytes) {
            Ok(()) => {
                write_new_durable(
                    &logs.join(format!("{name}.sha256")),
                    format!("{sha256}  {name}\n").as_bytes(),
                )
                .map_err(|e| format!("write operator stop record seal: {e}"))?;
                break path;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                attempt = attempt
                    .checked_add(1)
                    .ok_or("no unused operator stop record path")?;
            }
            Err(error) => return Err(format!("write operator stop record: {error}")),
        }
    };
    eprintln!(
        "stopped by operator before row {}/{rows_total} ({row}); stop file(s) {}; status {}",
        before_row + 1,
        present.join(", "),
        record.display(),
    );
    Ok(Some(OperatorStop {
        before_row,
        row: row.into(),
        rows_total,
        record,
    }))
}

/// One parent-loop step for a scheduled row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowStep {
    /// Accept the row from the resume directory; `Ok(false)` means it must run.
    Resume,
    /// Spawn, supervise, and accept the row's worker; `Ok(false)` records a pre-spawn refusal
    /// and the loop continues with the next row.
    Run,
}

/// The campaign parent's sequential row loop: resumed rows are accepted without a worker, and the
/// operator stop file is consulted between rows, before each worker spawn.
pub(crate) fn drive_rows(
    slugs: &[String],
    stop_files: &[PathBuf],
    logs: &Path,
    kind: &str,
    mut row: impl FnMut(usize, RowStep) -> Result<bool, String>,
) -> Result<DriveOutcome, String> {
    let mut refused = Vec::new();
    for (index, slug) in slugs.iter().enumerate() {
        if row(index, RowStep::Resume)? {
            eprintln!("coordinate {}/{} resumed: {slug}", index + 1, slugs.len());
            continue;
        }
        if let Some(stop) =
            operator_stop_before_row(stop_files, logs, kind, index, slug, slugs.len())?
        {
            return Ok(DriveOutcome {
                stop: Some(stop),
                refused,
            });
        }
        if row(index, RowStep::Run)? {
            eprintln!("coordinate {}/{} accepted: {slug}", index + 1, slugs.len());
        } else {
            eprintln!(
                "coordinate {}/{} refused before spawn: {slug}",
                index + 1,
                slugs.len()
            );
            refused.push(slug.clone());
        }
    }
    Ok(DriveOutcome {
        stop: None,
        refused,
    })
}

/// How the parent's row loop ended: an operator stop, and the rows refused before spawn (whose
/// unaccepted records are in the logs).
#[derive(Debug)]
pub(crate) struct DriveOutcome {
    pub(crate) stop: Option<OperatorStop>,
    pub(crate) refused: Vec<String>,
}

/// Immutable parent inputs.  The only model-related choices are snapshot paths; model identity,
/// geometry, memory, timing, and quality fields are collected by the child from the loaded product
/// and are never accepted by this command-line boundary.
#[derive(Clone, Debug)]
pub struct CampaignLaunch {
    pub executable: PathBuf,
    /// One candidate/reference snapshot pair per [`SC20671_FAMILIES`] entry, in that order.
    pub snapshots: [FamilySnapshots; SC20671_FAMILIES.len()],
    pub prompt_file: PathBuf,
    pub destination: PathBuf,
    pub resume_dir: PathBuf,
    /// Operator stop files checked between rows; never part of the resume identity.
    pub stop_files: Vec<PathBuf>,
    pub safety_policy: PathBuf,
    /// `Some` launches every scheduled row in `compressed` mode with this KV method; `None` is the
    /// dense baseline campaign.
    pub compressed: Option<CompressedKvMethod>,
    /// `--only-coordinate <slug>`: run just this scheduled row. The run publishes a partial,
    /// non-publishable manifest ([`SC20671_PARTIAL_RUN_KIND`]), never a campaign.
    pub only_coordinate: Option<String>,
}

/// One family's snapshot directories: `--<family>-snapshot` (the measured 4-bit candidate) and
/// `--<family>-fp32-reference-snapshot` (its pinned bf16 reference).
#[derive(Clone, Debug)]
pub struct FamilySnapshots {
    pub candidate: PathBuf,
    pub reference: PathBuf,
}

impl CampaignLaunch {
    /// The snapshot pair of a scheduled `family`.
    pub(super) fn family_snapshots(&self, family: &str) -> Result<&FamilySnapshots, String> {
        SC20671_FAMILIES
            .iter()
            .position(|candidate| *candidate == family)
            .map(|index| &self.snapshots[index])
            .ok_or_else(|| format!("SC-20671 has no snapshots for family {family:?}"))
    }

    /// Every pinned snapshot's inventory, candidates then references, each in
    /// [`SC20671_FAMILIES`] order (the [`resume_identity`] key order).
    pub(super) fn validated_inventories(&self) -> Result<Vec<SnapshotInventory>, String> {
        [false, true]
            .into_iter()
            .flat_map(|reference| {
                SC20671_FAMILIES
                    .iter()
                    .zip(&self.snapshots)
                    .map(move |(family, pair)| {
                        let path = if reference {
                            &pair.reference
                        } else {
                            &pair.candidate
                        };
                        validate_benchmark_snapshot(path, benchmark_model(family, reference)?)
                    })
            })
            .collect()
    }
}

/// The resume-identity key of a family's candidate or reference inventory (`llamaCandidate`,
/// `qwen8bReference`, ...).
pub(super) fn inventory_identity_key(family: &str, reference: bool) -> String {
    format!(
        "{family}{}",
        if reference { "Reference" } else { "Candidate" }
    )
}

/// Parse `--<family>-snapshot` and `--<family>-fp32-reference-snapshot` for every scheduled family.
pub(super) fn family_snapshot_flags(
    args: &[String],
) -> Result<[FamilySnapshots; SC20671_FAMILIES.len()], String> {
    SC20671_FAMILIES
        .iter()
        .map(|family| {
            Ok(FamilySnapshots {
                candidate: PathBuf::from(required_flag(args, &format!("--{family}-snapshot"))?),
                reference: PathBuf::from(required_flag(
                    args,
                    &format!("--{family}-fp32-reference-snapshot"),
                )?),
            })
        })
        .collect::<Result<Vec<_>, String>>()?
        .try_into()
        .map_err(|_| "one snapshot pair per scheduled family".to_string())
}

/// Indices into [`required_schedule`] a launch runs: all of them, or the one `--only-coordinate`
/// names (an unscheduled name is refused, listing the schedule).
pub(super) fn selected_schedule_indices(
    only_coordinate: Option<&str>,
) -> Result<Vec<usize>, String> {
    let slugs = required_coordinates()
        .iter()
        .map(coordinate_slug)
        .collect::<Vec<_>>();
    match only_coordinate {
        None => Ok((0..slugs.len()).collect()),
        Some(slug) => slugs
            .iter()
            .position(|candidate| candidate == slug)
            .map(|index| vec![index])
            .ok_or_else(|| {
                format!("--only-coordinate {slug:?} is not a scheduled coordinate; expected one of {slugs:?}")
            }),
    }
}

pub(crate) fn file_seal(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(&digest.finalize()))
}

pub(super) fn resume_identity(
    launch: &CampaignLaunch,
    policy_sha256: &str,
    inventories: &[SnapshotInventory],
) -> Result<serde_json::Value, String> {
    if inventories.len() != 2 * SC20671_FAMILIES.len() {
        return Err(
            "resume identity needs every family's candidate and reference inventory".into(),
        );
    }
    let inference_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let scene_works_root = PathBuf::from(required_campaign_env("SCENEWORKS_ROOT")?);
    checked_repository_identity(inference_root, INFERENCE_REPOSITORY)?;
    checked_repository_identity(&scene_works_root, SCENEWORKS_REPOSITORY)?;
    let inventory_value = |index: usize| {
        serde_json::json!({
            "sha256": inventories[index].sha256,
            "bytes": inventories[index].bytes,
        })
    };
    let mut identity = serde_json::json!({
        "schemaVersion": 1,
        "kind": "sc-20671-resume-identity",
        "scheduleVersion": SC20671_SCHEDULE_VERSION,
        "coordinates": required_coordinates().iter().map(coordinate_slug).collect::<Vec<_>>(),
        "inferenceRevision": checked_git_revision(inference_root)?,
        "sceneWorksRevision": checked_git_revision(&scene_works_root)?,
        "executableSha256": file_seal(&launch.executable)?,
        "promptSha256": file_seal(&launch.prompt_file)?,
        "policySha256": policy_sha256,
    });
    for (index, reference) in [false, true].into_iter().enumerate() {
        for (offset, family) in SC20671_FAMILIES.iter().enumerate() {
            identity[inventory_identity_key(family, reference)] =
                inventory_value(index * SC20671_FAMILIES.len() + offset);
        }
    }
    bind_resume_mode(&mut identity, launch.compressed);
    if let Some(slug) = &launch.only_coordinate {
        // A single-row run never shares a resume directory with a full campaign.
        identity["onlyCoordinate"] = slug.as_str().into();
    }
    Ok(identity)
}

/// A compressed campaign's resume identity names its mode and method, so dense and compressed
/// rows (or two methods) can never resume into each other. Dense identities keep their exact
/// historical bytes.
pub(super) fn bind_resume_mode(
    identity: &mut serde_json::Value,
    compressed: Option<CompressedKvMethod>,
) {
    if let Some(method) = compressed {
        identity["mode"] = "compressed".into();
        identity["kvMethod"] = method.id().into();
    }
}

/// The row mode a sealed resume identity launched: `None` for dense.
pub(super) fn resume_identity_mode(
    identity: &serde_json::Value,
) -> Result<Option<CompressedKvMethod>, String> {
    match (
        identity.get("mode").map(serde_json::Value::as_str),
        identity.get("kvMethod").map(serde_json::Value::as_str),
    ) {
        (None, None) => Ok(None),
        (Some(Some("compressed")), Some(Some(method))) => {
            CompressedKvMethod::parse(method).map(Some)
        }
        _ => Err("resume identity has a malformed compressed mode binding".into()),
    }
}

pub(super) fn seal_json(value: &serde_json::Value) -> Result<(Vec<u8>, String), String> {
    let bytes = canonical_json_bytes(value).map_err(|e| e.to_string())?;
    let sha = seal_bytes(&bytes);
    Ok((bytes, sha))
}

pub(super) fn preserve_captured_source(
    root: &Path,
    identity: &mut serde_json::Value,
) -> Result<(), String> {
    if root.exists() {
        let prior: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("identity.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        // A moved source ref cannot stale identical executable/model/contract bytes. Preserve the
        // captured refs as provenance; all new rows must stamp that same executable provenance.
        for key in ["inferenceRevision", "sceneWorksRevision"] {
            let captured = prior
                .get(key)
                .and_then(serde_json::Value::as_str)
                .ok_or("resume identity lacks captured source revision")?;
            identity[key] = serde_json::Value::String(captured.to_owned());
        }
    }
    Ok(())
}

pub(super) fn prepare_resume_root(
    root: &Path,
    identity: &mut serde_json::Value,
) -> Result<String, String> {
    preserve_captured_source(root, identity)?;
    let (bytes, sha) = seal_json(identity)?;
    let sidecar = format!("{sha}  identity.json\n");
    if root.exists() {
        if !root.is_dir()
            || fs::read(root.join("identity.json")).map_err(|e| e.to_string())? != bytes
            || fs::read_to_string(root.join("identity.json.sha256")).map_err(|e| e.to_string())?
                != sidecar
        {
            return Err("SC-20671 resume identity differs from current source, models, executable, schedule, prompt, or policy".into());
        }
    } else {
        fs::create_dir(root).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json"), &bytes).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json.sha256"), sidecar).map_err(|e| e.to_string())?;
    }
    Ok(sha)
}

/// Read-only preflight used to hand the exact producer-calculated identity and policy seal to
/// an independent consumer before any model child is started.
pub fn preflight_complete_campaign(launch: &CampaignLaunch) -> Result<serde_json::Value, String> {
    let policy = load_campaign_safety_policy(&launch.safety_policy)?;
    let inventories = launch.validated_inventories()?;
    let prompt = fs::read_to_string(&launch.prompt_file).map_err(|e| e.to_string())?;
    if prompt.trim().is_empty() {
        return Err("campaign prompt must not be empty".into());
    }
    let mut row_admission = Vec::new();
    let schedule = required_schedule();
    for row in selected_schedule_indices(launch.only_coordinate.as_deref())?
        .into_iter()
        .map(|index| schedule[index].clone())
    {
        let pair = launch.family_snapshots(row.coordinate.family)?;
        let (candidate, reference) = (&pair.candidate, &pair.reference);
        let coordinate = coordinate_slug(&row.coordinate);
        match static_row_requirements(
            &row.coordinate,
            candidate,
            reference,
            &prompt,
            &policy,
            launch.compressed.is_some(),
        )
        .and_then(|(total, request, footprint)| {
            Ok((
                total,
                request,
                runtime_guarded_admission(&policy, footprint, SC20671_ESTIMATE_SOURCE)?,
            ))
        }) {
            Ok((total, request, admission)) => row_admission.push(serde_json::json!({
                "coordinate": coordinate, "staticPreflightPassed": true,
                "totalLiveTokenBound": total, "requestTokenBound": request,
                "knownFootprintBudgetBytes": admission.static_footprint_floor_bytes,
                "admission": admission,
            })),
            Err(reason) => row_admission.push(serde_json::json!({
                "coordinate": coordinate, "staticPreflightPassed": false, "reason": reason,
            })),
        }
    }
    let policy_sha256 = policy.seal()?;
    let mut identity = resume_identity(launch, &policy_sha256, &inventories)?;
    preserve_captured_source(&launch.resume_dir, &mut identity)?;
    let (identity_bytes, resume_identity_sha256) = seal_json(&identity)?;
    if launch.resume_dir.exists()
        && (fs::read(launch.resume_dir.join("identity.json")).map_err(|e| e.to_string())?
            != identity_bytes
            || fs::read_to_string(launch.resume_dir.join("identity.json.sha256"))
                .map_err(|e| e.to_string())?
                != format!("{resume_identity_sha256}  identity.json\n"))
    {
        return Err(
            "SC-20671 existing resume identity does not match current behavior inputs".into(),
        );
    }
    Ok(serde_json::json!({
        "schemaVersion": 1,
        "policySha256": policy_sha256,
        "resumeIdentitySha256": resume_identity_sha256,
        "identity": identity,
        "completeStaticPreflightPassed": row_admission.iter().all(|row|
            row.get("staticPreflightPassed").and_then(serde_json::Value::as_bool) == Some(true)),
        "runtimeAdmission": "not-evaluated-by-read-only-preflight",
        "rows": row_admission,
    }))
}

pub(super) fn row_binding_path(root: &Path, slug: &str) -> PathBuf {
    root.join(format!("{slug}.binding.json"))
}

pub(super) fn row_binding(
    identity_sha256: &str,
    receipt: &Receipt,
    receipt_bytes: &[u8],
) -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1,
        "resumeIdentitySha256": identity_sha256,
        "receiptSha256": receipt.receipt_sha256,
        "receiptFileSha256": seal_bytes(receipt_bytes),
    })
}

pub(super) fn validate_receipt_launch_identity(
    receipt: &Receipt,
    coordinate: &Coordinate,
    identity: &serde_json::Value,
) -> Result<(), String> {
    let slug = coordinate_slug(coordinate);
    let p = &receipt.provenance;
    let field = |key| identity.get(key).and_then(serde_json::Value::as_str);
    let inventory = |key: &str| {
        identity
            .get(key)
            .and_then(|v| v.get("sha256"))
            .and_then(serde_json::Value::as_str)
    };
    let inventory_bytes = |key: &str| {
        identity
            .get(key)
            .and_then(|v| v.get("bytes"))
            .and_then(serde_json::Value::as_u64)
    };
    let (candidate, reference) = (
        inventory_identity_key(coordinate.family, false),
        inventory_identity_key(coordinate.family, true),
    );
    let (candidate, reference) = (candidate.as_str(), reference.as_str());
    let method = resume_identity_mode(identity)?;
    if receipt.mode
        != if method.is_some() {
            "compressed"
        } else {
            "dense"
        }
        || receipt.compression.as_ref().map(|c| c.method.as_str()) != method.map(|m| m.id())
    {
        return Err(format!(
            "resume row {slug} has a stale dense/compressed mode"
        ));
    }
    if p.inference_revision != field("inferenceRevision").unwrap_or("")
        || p.scene_works_revision != field("sceneWorksRevision").unwrap_or("")
        || p.model_file_sha256 != inventory(candidate).unwrap_or("")
        || Some(p.model_file_bytes) != inventory_bytes(candidate)
        || p.reference_model_sha256 != inventory(reference).unwrap_or("")
        || Some(p.reference_model_bytes) != inventory_bytes(reference)
        || receipt.matrix.family != coordinate.family
        || receipt.matrix.context_band != coordinate.context_band
        || receipt.matrix.request_mode != coordinate.request_mode
        || receipt.matrix.prefill_mode != coordinate.prefill_mode
        || receipt.matrix.process_temperature != coordinate.process_temperature
    {
        return Err(format!(
            "resume row {slug} has stale source, model, or coordinate identity"
        ));
    }
    Ok(())
}

pub(super) fn validate_resume_row(
    root: &Path,
    coordinate: &Coordinate,
    identity: &serde_json::Value,
    identity_sha256: &str,
) -> Result<PreparedCoordinateReceipt, String> {
    let slug = coordinate_slug(coordinate);
    let item = load_prepared_coordinate_receipt(&root.join(&slug), coordinate.clone())?;
    let receipt: Receipt =
        serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
    validate_sealed_receipt(&receipt)?;
    validate_receipt_launch_identity(&receipt, coordinate, identity)?;
    let expected = seal_json(&row_binding(
        identity_sha256,
        &receipt,
        &item.bundle.receipt,
    ))?
    .0;
    if fs::read(row_binding_path(root, &slug)).map_err(|e| e.to_string())? != expected {
        return Err(format!(
            "resume row {slug} has missing or stale identity binding"
        ));
    }
    Ok(item)
}

/// Run one row at a time under a mandatory bounded policy. Valid sealed rows stay in an
/// identity-bound resume directory after a failure; the destination appears only after all eight
/// scheduled rows have been accepted. Between rows — never during a worker — the parent honours
/// the operator stop file: it records a durable stopped-before-row status and returns
/// [`CampaignOutcome::StoppedByOperator`]; the same command later resumes at that row.
pub fn launch_complete_campaign(launch: &CampaignLaunch) -> Result<CampaignOutcome, String> {
    if launch.destination.exists() {
        return Err("campaign destination already exists".into());
    }
    if !launch.resume_dir.is_absolute() || !launch.destination.is_absolute() {
        return Err("campaign destination and resume directory must be absolute paths".into());
    }
    if launch.resume_dir == launch.destination {
        return Err("campaign resume directory must differ from destination".into());
    }
    for path in [
        &launch.executable,
        &launch.prompt_file,
        &launch.safety_policy,
    ]
    .into_iter()
    .chain(
        launch
            .snapshots
            .iter()
            .flat_map(|pair| [&pair.candidate, &pair.reference]),
    ) {
        if !path.exists() {
            return Err(format!(
                "required campaign input is absent: {}",
                path.display()
            ));
        }
    }
    let policy = load_campaign_safety_policy(&launch.safety_policy)?;
    let policy_sha256 = policy.seal()?;
    let inventories = launch.validated_inventories()?;
    let schedule = required_schedule();
    let selected = selected_schedule_indices(launch.only_coordinate.as_deref())?;
    let prompt = fs::read_to_string(&launch.prompt_file).map_err(|e| e.to_string())?;
    if prompt.trim().is_empty() {
        return Err("campaign prompt must not be empty".into());
    }
    let mut identity = resume_identity(launch, &policy_sha256, &inventories)?;
    let identity_sha256 = prepare_resume_root(&launch.resume_dir, &mut identity)?;
    let slugs = selected
        .iter()
        .map(|index| coordinate_slug(&schedule[*index].coordinate))
        .collect::<Vec<_>>();
    validate_resume_entries(&launch.resume_dir, &launch.stop_files, &slugs)?;
    let mut prepared = Vec::with_capacity(schedule.len());
    let stopped = drive_rows(
        &slugs,
        &launch.stop_files,
        &launch.resume_dir.join("logs"),
        "sc-20671-operator-stop",
        |index, step| {
            let schedule_index = selected[index];
            let row = &schedule[schedule_index];
            let slug = coordinate_slug(&row.coordinate);
            let child_dir = launch.resume_dir.join(&slug);
            let binding_path = row_binding_path(&launch.resume_dir, &slug);
            if step == RowStep::Resume {
                if !child_dir.exists() && !binding_path.exists() {
                    return Ok(false);
                }
                if !child_dir.exists() || !binding_path.exists() {
                    return Err(format!(
                        "partial resume row {slug} must be repaired explicitly"
                    ));
                }
                prepared.push(validate_resume_row(
                    &launch.resume_dir,
                    &row.coordinate,
                    &identity,
                    &identity_sha256,
                )?);
                return Ok(true);
            }
            let pair = launch.family_snapshots(row.coordinate.family)?;
            let (snapshot, reference_snapshot) = (&pair.candidate, &pair.reference);
            let (total_tokens, request_tokens, static_footprint_floor) = static_row_requirements(
                &row.coordinate,
                snapshot,
                reference_snapshot,
                &prompt,
                &policy,
                launch.compressed.is_some(),
            )?;
            let admission = runtime_guarded_admission(
                &policy,
                static_footprint_floor,
                SC20671_ESTIMATE_SOURCE,
            )?;
            let mut command = Command::new(&launch.executable);
            command
                .arg("worker")
                .arg("--coordinate-index")
                .arg(schedule_index.to_string())
                .arg("--snapshot")
                .arg(snapshot)
                .arg("--prompt-file")
                .arg(&launch.prompt_file)
                .arg("--fp32-reference-snapshot")
                .arg(reference_snapshot)
                .arg("--safety-policy")
                .arg(&launch.safety_policy)
                .arg("--policy-sha256")
                .arg(&policy_sha256)
                .arg("--resume-identity")
                .arg(launch.resume_dir.join("identity.json"))
                .arg("--resume-identity-sha256")
                .arg(&identity_sha256)
                .arg("--out")
                .arg(&child_dir);
            if let Some(method) = launch.compressed {
                command
                    .arg("--mode")
                    .arg("compressed")
                    .arg("--kv-method")
                    .arg(method.id());
            }
            let logs = launch.resume_dir.join("logs");
            let log_prefix =
                unused_attempt_prefix(&logs, &slug).ok_or("no unused bounded worker log path")?;
            let request = RunRequest {
                context_tokens: total_tokens,
                request_tokens,
                estimate: admission.estimate(),
                stdout_path: logs.join(format!("{log_prefix}.stdout.log")),
                stderr_path: logs.join(format!("{log_prefix}.stderr.log")),
            };
            let unaccepted = logs.join(format!("{log_prefix}.unaccepted.json"));
            // Pre-row duration estimate from completed same-family rows. A row whose estimate
            // exceeds the deadline by the margin is refused before it spawns; the campaign then
            // continues with the next row and ends incomplete-with-refusals.
            let basis = prepared
                .iter()
                .filter_map(|item| serde_json::from_slice::<Receipt>(&item.bundle.receipt).ok())
                .filter_map(|receipt| row_duration_basis(&receipt))
                .collect::<Vec<_>>();
            let estimate = estimate_row_seconds(
                row.coordinate.family,
                row.coordinate.context_band,
                row_work(
                    row.coordinate.process_temperature,
                    row.coordinate.request_mode,
                    row.coordinate.prefill_mode,
                ),
                &basis,
            );
            let deadline_seconds = policy.row_deadline_seconds;
            let estimate_record = serde_json::json!({
                "schemaVersion": 2,
                "kind": "sc-20671-row-duration-estimate",
                "coordinate": slug,
                "estimatedSeconds": estimate.as_ref().map(|e| e.seconds),
                "fixedSeconds": estimate.as_ref().map(|e| e.fixed_seconds),
                "prefillSeconds": estimate.as_ref().map(|e| e.prefill_seconds),
                "basisCoordinate": estimate.as_ref().map(|e| e.basis_coordinate.clone()),
                "rowDeadlineSeconds": deadline_seconds,
                "refusalMargin": DURATION_REFUSAL_MARGIN,
                "method": "basis wall time split into measured full-context prefills and a fixed remainder; fixed scaled by measurement units (timing runs + quality halves), prefill by count x T(1+T/Tattn) ratio",
            });
            fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
            fs::write(
                logs.join(format!("{log_prefix}.estimate.json")),
                seal_json(&estimate_record)?.0,
            )
            .map_err(|e| e.to_string())?;
            match &estimate {
                Some(estimate) => eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} estimated at {:.0}s (fixed {:.0}s + prefill {:.0}s, from {}); row deadline {deadline_seconds}s",
                    estimate.seconds, estimate.fixed_seconds, estimate.prefill_seconds, estimate.basis_coordinate
                ),
                None => eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} has no completed full-prefill {} row to estimate from; row deadline {deadline_seconds}s",
                    row.coordinate.family
                ),
            }
            if let Some(estimate) = estimate
                .as_ref()
                .filter(|estimate| duration_estimate_refuses(estimate, deadline_seconds))
            {
                let detail = format!(
                    "estimated duration {:.0}s (from {}) exceeds the {deadline_seconds}s row deadline by more than {DURATION_REFUSAL_MARGIN}x",
                    estimate.seconds, estimate.basis_coordinate
                );
                let message = unaccepted_row_error(
                    &unaccepted,
                    "sc-20671-unaccepted-row",
                    &slug,
                    &admission,
                    ("DurationEstimate", &detail, None, None),
                    format!("coordinate {slug} refused before spawn: {detail}"),
                );
                eprintln!("sc20671-kv-baseline: {message}; continuing with the next row");
                return Ok(false);
            }
            let status = match campaign_supervisor::run_guarded(
                &mut command,
                &request,
                &policy.supervisor(),
                &mut SystemProbe,
            ) {
                Ok(status) => status,
                Err(failure) => {
                    let reason = format!("{:?}", failure.reason);
                    return Err(unaccepted_row_error(
                        &unaccepted,
                        "sc-20671-unaccepted-row",
                        &slug,
                        &admission.with_host_memory(failure.host_memory.as_deref().cloned()),
                        (
                &reason,
                &failure.detail,
                failure.pid,
                failure.watchdog_host_memory.as_deref(),
            ),
                        format!(
                            "coordinate {slug} stopped ({reason}): {}; child {:?} reaped; valid earlier rows remain in {}",
                            failure.detail, failure.pid, launch.resume_dir.display(),
                        ),
                    ));
                }
            };
            if !status.success() {
                return Err(unaccepted_row_error(
                    &unaccepted,
                    "sc-20671-unaccepted-row",
                    &slug,
                    &admission,
                    ("ChildExit", &status.to_string(), None, None),
                    format!(
                        "coordinate {slug} failed with {status}; stderr: {}; valid earlier rows remain in {}",
                        request.stderr_path.display(), launch.resume_dir.display(),
                    ),
                ));
            }
            // Child publication itself is atomic. Bind the row to this exact immutable launch before
            // permitting a future resume to accept it.
            let item = load_prepared_coordinate_receipt(&child_dir, row.coordinate.clone())?;
            let receipt: Receipt =
                serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
            if let Some(gate) = &receipt.quality.quality_gate {
                eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} accepted as measured; quality gate {}",
                    quality_gate_summary(gate)
                );
            }
            let binding = seal_json(&row_binding(
                &identity_sha256,
                &receipt,
                &item.bundle.receipt,
            ))?
            .0;
            fs::write(&binding_path, binding).map_err(|e| e.to_string())?;
            prepared.push(validate_resume_row(
                &launch.resume_dir,
                &row.coordinate,
                &identity,
                &identity_sha256,
            )?);
            Ok(true)
        },
    )?;
    if let Some(stop) = stopped.stop {
        return Ok(CampaignOutcome::StoppedByOperator(stop));
    }
    if !stopped.refused.is_empty() {
        // Every other row ran; the campaign is not complete and is never published.
        eprintln!(
            "sc20671-kv-baseline: campaign incomplete: {} row(s) refused before spawn ({}); accepted rows remain in {}",
            stopped.refused.len(),
            stopped.refused.join(", "),
            launch.resume_dir.display()
        );
        return Ok(CampaignOutcome::IncompleteWithRefusals(stopped.refused));
    }
    match &launch.only_coordinate {
        Some(slug) => {
            publish_campaign(
                &launch.destination,
                &prepared,
                &identity,
                &policy,
                Some(slug),
            )?;
            eprintln!(
                "sc20671-kv-baseline: PARTIAL run of {slug} only, published as a non-publishable {SC20671_PARTIAL_RUN_KIND} at {}; it is not a campaign",
                launch.destination.display()
            );
        }
        None => publish_complete_campaign(&launch.destination, &prepared, &identity, &policy)?,
    }
    report_campaign_quality_gates(
        &prepared
            .iter()
            .map(|item| serde_json::from_slice::<Receipt>(&item.bundle.receipt))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    )?;
    Ok(CampaignOutcome::Completed)
}

/// Every resume-directory entry must be campaign state for this schedule or the operator stop file;
/// the stop file is tolerated here and never enters the resume identity.
pub(super) fn validate_resume_entries(
    resume_dir: &Path,
    stop_files: &[PathBuf],
    slugs: &[String],
) -> Result<(), String> {
    for entry in fs::read_dir(resume_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !["identity.json", "identity.json.sha256", "logs"].contains(&name.as_str())
            && !is_operator_stop_entry(resume_dir, stop_files, &name)
            && !slugs.contains(&name)
            && !slugs
                .iter()
                .any(|slug| name == format!("{slug}.binding.json"))
        {
            return Err(format!("unexpected or partial resume artifact: {name}"));
        }
    }
    Ok(())
}

/// `name <value>` when present (a flag without a value is refused).
pub(super) fn optional_flag(args: &[String], name: &str) -> Result<Option<String>, String> {
    match args.iter().position(|arg| arg == name) {
        None => Ok(None),
        Some(index) => args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("{name} requires a value")),
    }
}

pub(super) fn required_flag(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then(|| window[1].clone()))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {name}"))
}

/// `--mode dense|compressed` (default dense) and, for compressed, the required `--kv-method`.
pub(super) fn compressed_mode_flags(args: &[String]) -> Result<Option<CompressedKvMethod>, String> {
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|_| required_flag(args, name))
            .transpose()
    };
    match (flag("--mode")?.as_deref(), flag("--kv-method")?) {
        (None | Some("dense"), None) => Ok(None),
        (Some("compressed"), Some(method)) => CompressedKvMethod::parse(&method).map(Some),
        (Some("compressed"), None) => Err("--mode compressed requires --kv-method".into()),
        (None | Some("dense"), Some(_)) => Err("--kv-method requires --mode compressed".into()),
        (Some(mode), _) => Err(format!("--mode must be dense or compressed, not {mode:?}")),
    }
}

/// CLI entrypoint used by the standalone `sc20671-kv-baseline` binary.  The child accepts no
/// caller-authored evidence: identity, geometry, timing, quality, cache, and fixtures are bound
/// from the loaded product session before receipt publication.
///
/// `--mode compressed --kv-method <method>` (parent, preflight, and worker) runs every scheduled
/// row with its KV held in that method's compressed cache and fused decode attention, gated
/// against a dense-KV reference arm on the same candidate weights (SC-20676). Methods:
/// `group-affine` (2-bit codes), `group-affine-4` (4-bit) and `group-affine-8` (8-bit), all group
/// 32. `--only-coordinate <coordinate>` runs one scheduled row and publishes a partial,
/// non-publishable `sc-20671-partial-coordinate-run` manifest (never a campaign). The GPU-window
/// launch of the compressed campaign is one command from the inference checkout:
///
/// ```text
/// eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH && \
/// SCENEWORKS_ROOT=/abs/SceneWorks cargo run --locked --release -p mlx-llm --features campaign --bin sc20671_kv_baseline -- \
///   parent --mode compressed --kv-method group-affine \
///   --llama-snapshot <llama-4bit> --qwen-snapshot <qwen-4bit> \
///   --llama-fp32-reference-snapshot <llama-bf16> --qwen-fp32-reference-snapshot <qwen-bf16> \
///   --llama8b-snapshot <llama8b-4bit> --qwen8b-snapshot <qwen8b-4bit> \
///   --llama8b-fp32-reference-snapshot <llama8b-bf16> --qwen8b-fp32-reference-snapshot <qwen8b-bf16> \
///   --prompt-file <prompt.txt> --safety-policy <policy.json> \
///   --resume-dir /abs/sc20676-compressed-resume --out /abs/sc20676-compressed-campaign
/// ```
///
/// Safe operator stop: `touch /abs/sc20676-compressed-resume/STOP` (always honoured) or the
/// `--stop-file <path>` given to `parent` (honoured as well). The parent finishes the row whose
/// worker is running — it never signals a worker — then, before spawning the next row, writes
/// `<resume-dir>/logs/operator-stop.attempt-<n>.json` ("stopped-by-operator", `beforeRow`) with its
/// `.sha256` seal and exits with status [`OPERATOR_STOP_EXIT_CODE`] (75). Remove the stop file(s)
/// and rerun the same command: accepted rows resume and the campaign continues at that row.
pub fn sc20671_cli(args: &[String]) -> Result<CampaignOutcome, String> {
    let Some(mode) = args.first().map(String::as_str) else {
        return Err("usage: sc20671-kv-baseline parent|worker [options]".into());
    };
    match mode {
        "parent" | "preflight" => {
            let resume_dir = PathBuf::from(required_flag(args, "--resume-dir")?);
            let launch = CampaignLaunch {
                executable: std::env::current_exe().map_err(|e| e.to_string())?,
                snapshots: family_snapshot_flags(args)?,
                prompt_file: PathBuf::from(required_flag(args, "--prompt-file")?),
                destination: PathBuf::from(required_flag(args, "--out")?),
                stop_files: operator_stop_files(args, &resume_dir)?,
                resume_dir,
                safety_policy: PathBuf::from(required_flag(args, "--safety-policy")?),
                compressed: compressed_mode_flags(args)?,
                only_coordinate: optional_flag(args, "--only-coordinate")?,
            };
            if mode == "preflight" {
                let value = preflight_complete_campaign(&launch)?;
                println!(
                    "{}",
                    String::from_utf8(canonical_json_bytes(&value).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?
                );
                Ok(CampaignOutcome::Completed)
            } else {
                launch_complete_campaign(&launch)
            }
        }
        "worker" => {
            let policy = load_campaign_safety_policy(&PathBuf::from(required_flag(
                args,
                "--safety-policy",
            )?))?;
            if policy.seal()? != required_flag(args, "--policy-sha256")? {
                return Err("worker safety policy seal differs from parent's frozen policy".into());
            }
            let identity_bytes = fs::read(required_flag(args, "--resume-identity")?)
                .map_err(|e| format!("read parent resume identity: {e}"))?;
            let identity_sha256 = seal_bytes(&identity_bytes);
            if identity_sha256 != required_flag(args, "--resume-identity-sha256")? {
                return Err(
                    "worker resume identity seal differs from parent's frozen identity".into(),
                );
            }
            let identity: serde_json::Value =
                serde_json::from_slice(&identity_bytes).map_err(|e| e.to_string())?;
            let identity_field = |key| {
                identity
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("resume identity lacks {key}"))
            };
            let compressed = compressed_mode_flags(args)?;
            if resume_identity_mode(&identity)? != compressed {
                return Err("worker dense/compressed mode differs from resume identity".into());
            }
            if identity_field("policySha256")? != policy.seal()?
                || identity_field("executableSha256")?
                    != file_seal(&std::env::current_exe().map_err(|e| e.to_string())?)?
                || identity
                    .get("scheduleVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(SC20671_SCHEDULE_VERSION)
            {
                return Err(
                    "worker executable, policy, or schedule differs from resume identity".into(),
                );
            }
            let captured_source = (
                identity_field("sceneWorksRevision")?,
                identity_field("inferenceRevision")?,
            );
            let index = required_flag(args, "--coordinate-index")?
                .parse::<usize>()
                .map_err(|_| "--coordinate-index must be an integer".to_string())?;
            let row = required_schedule()
                .get(index)
                .cloned()
                .ok_or("--coordinate-index is outside the frozen covering schedule")?;
            if let Some(only) = identity.get("onlyCoordinate") {
                if only.as_str() != Some(coordinate_slug(&row.coordinate).as_str()) {
                    return Err(
                        "worker coordinate differs from the resume identity's coordinate filter"
                            .into(),
                    );
                }
            }
            let snapshot = PathBuf::from(required_flag(args, "--snapshot")?);
            let reference_snapshot =
                PathBuf::from(required_flag(args, "--fp32-reference-snapshot")?);
            let prompt_path = PathBuf::from(required_flag(args, "--prompt-file")?);
            if file_seal(&prompt_path)? != identity_field("promptSha256")? {
                return Err(
                    "worker prompt bytes differ from the parent-sealed resume identity".into(),
                );
            }
            let prompt =
                fs::read_to_string(&prompt_path).map_err(|e| format!("read prompt file: {e}"))?;
            if prompt.trim().is_empty() {
                return Err("campaign prompt must not be empty".into());
            }
            let (candidate_inventory, reference_inventory) = validate_coordinate_model_contract(
                &row.coordinate,
                &snapshot,
                &reference_snapshot,
            )?;
            let (candidate_key, reference_key) = (
                inventory_identity_key(row.coordinate.family, false),
                inventory_identity_key(row.coordinate.family, true),
            );
            for (key, inventory) in [
                (candidate_key.as_str(), &candidate_inventory),
                (reference_key.as_str(), &reference_inventory),
            ] {
                if identity
                    .get(key)
                    .and_then(|model| model.get("sha256"))
                    .and_then(serde_json::Value::as_str)
                    != Some(inventory.sha256.as_str())
                    || identity
                        .get(key)
                        .and_then(|model| model.get("bytes"))
                        .and_then(serde_json::Value::as_u64)
                        != Some(inventory.bytes)
                {
                    return Err(format!(
                        "worker {key} inventory differs from parent-sealed resume identity"
                    ));
                }
            }
            let (_, _, static_footprint_floor) = static_row_requirements(
                &row.coordinate,
                &snapshot,
                &reference_snapshot,
                &prompt,
                &policy,
                compressed.is_some(),
            )?;
            let admission =
                supervised_admission(&policy, static_footprint_floor, SC20671_ESTIMATE_SOURCE)?;
            // Observed before any model loads: a throttled host refuses the row before it runs.
            let row_start = capture_host_state("row-start")?;
            refuse_throttled_row_start(&row_start)?;
            // Compressed rows: the measured arm holds KV in the method's compressed cache, and its
            // quality denominator is the dense-KV run on the SAME candidate weights.
            let candidate_session = match compressed {
                Some(method) => CampaignSession::load_compressed(&snapshot, method),
                None => CampaignSession::load(&snapshot),
            }
            .map_err(|e| format!("load candidate campaign session: {e}"))?;
            let candidate_baseline = candidate_session.load_start_sample.mlx_active_bytes;
            // Candidate lifecycle evidence is collected before the reference model exists in the
            // process. This keeps every global MLX phase sample attributable to the candidate.
            // Contract v5: the timing measurements (a warm row's warmups, then the five repeats)
            // run first, so a cold row's repeat 0 is the process's first product dispatch; quality
            // is then measured once per arm, because every fixture is deterministic in-process.
            let timing_run = |kind: &str, index: usize| {
                run_timing_on_session(
                    &candidate_session,
                    &prompt,
                    &row.coordinate,
                    policy.max_request_tokens,
                )
                .map_err(|e| {
                    format!(
                        "timing {kind} {index} for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })
            };
            let mut timing_warmups = Vec::new();
            if row.coordinate.process_temperature == "warm" {
                for warmup_index in 0..TIMING_WARMUPS {
                    timing_warmups.push(timing_run("warmup", warmup_index)?);
                }
            }
            let warmup_cache_state_version =
                (!timing_warmups.is_empty()).then(|| candidate_session.cache_state_version());
            let mut timing_repeats = Vec::with_capacity(TIMING_REPEATS);
            let mut repeat_host_states = Vec::with_capacity(TIMING_REPEATS);
            for repeat in 0..TIMING_REPEATS {
                timing_repeats.push(timing_run("repeat", repeat)?);
                // Outside every timed window: recorded so throughput drift is attributable.
                repeat_host_states.push(capture_host_state(TIMING_SAMPLE_HOST_BOUNDARY)?);
            }
            // The kernel fixture's coordinate operation is timing repeat 0's: the same prompt,
            // operation and session, so it is reused rather than dispatched again.
            let candidate_quality = run_product_fixture_half_on_session_bounded(
                &candidate_session,
                &prompt,
                &row.coordinate,
                policy.max_request_tokens,
                Some(timing_repeats[0].coordinate.clone()),
            )
            .map_err(|e| {
                format!(
                    "candidate quality measurement for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;
            eprintln!(
                "{}",
                fixture_half_attempt_diagnostic(
                    &candidate_quality,
                    &row.coordinate,
                    &identity_sha256,
                    "candidate",
                    "quality",
                    0,
                )
            );
            let candidate_repeats = vec![candidate_quality];
            let compressed_parity = candidate_session
                .compressed()
                .map(CompressedKvArm::kernel_parity_errors)
                .transpose()
                .map_err(|e| format!("compressed kernel parity: {e}"))?;
            drop(candidate_session);
            quiesce_campaign_active_memory(candidate_baseline).map_err(|e| {
                format!(
                    "candidate session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let (reference_session_snapshot, reference_kind) = if compressed.is_some() {
                (&snapshot, QualityReference::DenseKvSameWeights)
            } else {
                (&reference_snapshot, QualityReference::Bf16Characterization)
            };
            let reference_session = CampaignSession::load(reference_session_snapshot)
                .map_err(|e| format!("load reference campaign session: {e}"))?;
            let reference_baseline = reference_session.load_start_sample.mlx_active_bytes;
            // The reference arm is never timed: its one quality measurement is all it runs.
            let reference_quality = run_product_fixture_half_on_session_bounded(
                &reference_session,
                &prompt,
                &row.coordinate,
                policy.max_request_tokens,
                None,
            )
            .map_err(|e| {
                format!(
                    "reference quality measurement for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;
            eprintln!(
                "{}",
                fixture_half_attempt_diagnostic(
                    &reference_quality,
                    &row.coordinate,
                    &identity_sha256,
                    "reference",
                    "quality",
                    0,
                )
            );
            let reference_repeats = vec![reference_quality];
            // Compressed rows: the same-weights dense-KV session greedily continues the kernel
            // fixture prompt through every stop token, once per row (the stream is deterministic),
            // scoring its own likelihood of every token of that stream.
            let forced_reference = compressed
                .map(|_| {
                    let kernel_prompt =
                        kernel_fixture_prompt(&reference_session, &prompt, &row.coordinate)?;
                    reference_session.provider.campaign_forced_continuation(
                        &kernel_prompt,
                        FORCED_CONTINUATION_TOKENS as usize,
                        None,
                        None,
                    )
                })
                .transpose()
                .map_err(|e| {
                    format!(
                        "dense forced continuation for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
            // Contract v4: compressed rows' multi-turn prompt-cache agreement is measured on the
            // same-weights dense-KV session's turn-2 forced continuation, after its own turn-2
            // prompt-cache hit, once per row.
            let multi_turn_reference = compressed
                .map(|_| {
                    let turn1_prompt =
                        multi_turn_fixture_prompt(&reference_session, &prompt, &row.coordinate)?;
                    reference_session
                        .provider
                        .campaign_multi_turn_forced_continuation(
                            &fixture_request(turn1_prompt, Vec::new()),
                            MULTI_TURN_FIXTURE_FOLLOW_UP,
                            FORCED_CONTINUATION_TOKENS as usize,
                            None,
                            None,
                        )
                })
                .transpose()
                .map_err(|e| {
                    format!(
                        "dense turn-2 forced continuation for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
            drop(reference_session);
            quiesce_campaign_active_memory(reference_baseline).map_err(|e| {
                format!(
                    "reference session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            // Greedy agreement (quality contract v3) is teacher-forced: the candidate, reloaded in
            // its own representation, is evaluated on each reference repeat's kernel stream. This
            // runs after every timed dispatch, so it never touches timing or compile attribution.
            let forcing_session = match compressed {
                Some(method) => CampaignSession::load_compressed(&snapshot, method),
                None => CampaignSession::load(&snapshot),
            }
            .map_err(|e| format!("load teacher-forcing candidate session: {e}"))?;
            let forcing_baseline = forcing_session.load_start_sample.mlx_active_bytes;
            // A compressed row is teacher-forced once on its dense forced continuation (every
            // position compared by argmax); a dense row keeps one forced pass per distinct natural
            // reference stream (greedy reference repeats are usually identical, and each pass is
            // a full-context prefill). Each pass also scores the candidate's likelihood of the
            // reference tokens, so the perplexity delta compares both arms on one stream.
            let (teacher_forced, forced_continuation, stream_likelihoods) = match &forced_reference
            {
                Some(reference) => {
                    let reference_stream = &reference.choices;
                    let candidate = kernel_fixture_prompt(&forcing_session, &prompt, &row.coordinate)
                        .and_then(|kernel_prompt| {
                            forcing_session.provider.campaign_forced_continuation(
                                &kernel_prompt,
                                reference_stream.len(),
                                forcing_session.compressed(),
                                Some(reference_stream),
                            )
                        })
                        .map_err(|e| {
                            format!(
                                "teacher-forced continuation for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })?;
                    let continuation =
                        forced_continuation_evidence(reference_stream, &candidate.choices)?;
                    let likelihood = StreamLikelihood::from_probabilities(
                        &reference.stream_probabilities,
                        &candidate.stream_probabilities,
                    )?;
                    eprintln!(
                        "sc20671-kv-baseline: coordinate {} forced continuation agreement {} ({}/{}; first flips {:?})",
                        coordinate_slug(&row.coordinate),
                        continuation.agreement,
                        continuation.matches,
                        continuation.tokens,
                        continuation.first_flip_positions
                    );
                    (
                        vec![None; reference_repeats.len()],
                        Some(continuation),
                        vec![Some(likelihood); reference_repeats.len()],
                    )
                }
                None => {
                    let stop_tokens = forcing_session.provider.campaign_stop_tokens().to_vec();
                    let streams = reference_repeats
                        .iter()
                        .map(|reference| {
                            stream_tokens(
                                &reference.kernel.quality_observation.token_probabilities,
                            )
                        })
                        .collect::<Vec<_>>();
                    let passes = force_distinct_streams(&streams, |stream| {
                        teacher_forced_kernel_choices(
                            &forcing_session,
                            &prompt,
                            &row.coordinate,
                            stream,
                        )
                        .map_err(|e| {
                            format!(
                                "teacher-forced pass for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })
                    })?;
                    // The reference's natural stream is its own greedy output, so its recorded
                    // probabilities are already its likelihood of that stream.
                    let likelihoods = reference_repeats
                        .iter()
                        .zip(&passes)
                        .map(|(reference, pass)| {
                            StreamLikelihood::from_probabilities(
                                &reference
                                    .kernel
                                    .quality_observation
                                    .token_probabilities
                                    .iter()
                                    .map(|(_, probability)| *probability)
                                    .collect::<Vec<_>>(),
                                &pass.stream_probabilities,
                            )
                            .map(Some)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let forced = passes
                        .into_iter()
                        .map(|pass| {
                            Some(TeacherForcedChoices {
                                choices: pass.choices,
                                stop_tokens: stop_tokens.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    (forced, None, likelihoods)
                }
            };
            // Multi-turn prompt-cache agreement (contract v4) is teacher-forced the same way, on
            // turn 2 after the candidate's own turn-2 prompt-cache hit: the row's turn-2 forced
            // continuation (compressed) or each reference repeat's natural turn-2 stream (dense).
            let (teacher_forced_cache, multi_turn_continuation, multi_turn_pass) =
                match &multi_turn_reference {
                Some((reference_stream, reference_turns)) => {
                    let (choices, candidate_turns) =
                        multi_turn_fixture_prompt(&forcing_session, &prompt, &row.coordinate)
                        .and_then(|turn1_prompt| {
                            forcing_session
                                .provider
                                .campaign_multi_turn_forced_continuation(
                                    &fixture_request(turn1_prompt, Vec::new()),
                                    MULTI_TURN_FIXTURE_FOLLOW_UP,
                                    reference_stream.len(),
                                    forcing_session.compressed(),
                                    Some(reference_stream),
                                )
                        })
                        .map_err(|e| {
                            format!(
                                "teacher-forced turn-2 continuation for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })?;
                    // The compressed arm is forced on the reference stream of the same turn-2
                    // prompt token ids, or the row is refused.
                    let pass = MultiTurnForcedPass {
                        reference: reference_turns.clone(),
                        candidate: candidate_turns,
                    };
                    pass.validate().map_err(|e| {
                        format!(
                            "turn-2 forced continuation for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        )
                    })?;
                    let continuation =
                        multi_turn_forced_continuation_evidence(reference_stream, &choices)?;
                    eprintln!(
                        "sc20671-kv-baseline: coordinate {} multi-turn turn-2 forced continuation agreement {} ({}/{}; first flips {:?})",
                        coordinate_slug(&row.coordinate),
                        continuation.agreement,
                        continuation.matches,
                        continuation.tokens,
                        continuation.first_flip_positions
                    );
                    (
                        vec![None; reference_repeats.len()],
                        Some(continuation),
                        Some(pass),
                    )
                }
                None => {
                    let stop_tokens = forcing_session.provider.campaign_stop_tokens().to_vec();
                    let streams = reference_repeats
                        .iter()
                        .map(|reference| {
                            stream_tokens(&reference.cache.quality_observation.token_probabilities)
                        })
                        .collect::<Vec<_>>();
                    let forced = force_distinct_streams(&streams, |stream| {
                        teacher_forced_multi_turn_choices(
                            &forcing_session,
                            &prompt,
                            &row.coordinate,
                            stream,
                        )
                        .map_err(|e| {
                            format!(
                                "teacher-forced turn-2 pass for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })
                    })?
                    .into_iter()
                    .map(|choices| {
                        Some(TeacherForcedChoices {
                            choices,
                            stop_tokens: stop_tokens.clone(),
                        })
                    })
                    .collect::<Vec<_>>();
                    (forced, None, None)
                }
            };
            drop(forcing_session);
            quiesce_campaign_active_memory(forcing_baseline).map_err(|e| {
                format!(
                    "teacher-forcing session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let suites = candidate_repeats
                .into_iter()
                .zip(reference_repeats)
                .zip(teacher_forced)
                .zip(teacher_forced_cache)
                .zip(stream_likelihoods)
                .enumerate()
                .map(|(repeat, ((((candidate, reference), forced), forced_cache), likelihood))| {
                    let mut suite =
                        pair_product_fixture_halves(candidate, reference, reference_kind)?;
                    suite.kernel_parity_errors = compressed_parity.clone();
                    suite.teacher_forced_candidate = forced;
                    suite.forced_continuation = forced_continuation.clone();
                    suite.stream_likelihood = likelihood;
                    suite.teacher_forced_cache = forced_cache;
                    suite.multi_turn_forced_continuation = multi_turn_continuation.clone();
                    suite.multi_turn_forced_pass = multi_turn_pass.clone();
                    suite.quality().map_err(|e| {
                        core_llm::Error::InvalidRequest(format!(
                            "product quality measurement {repeat} for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        ))
                    })?;
                    Ok(suite)
                })
                .collect::<core_llm::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            let kernel_runs = timing_repeats
                .iter()
                .map(|run| &run.coordinate)
                .collect::<Vec<_>>();
            let warmup_runs = timing_warmups
                .iter()
                .map(|run| &run.coordinate)
                .collect::<Vec<_>>();
            let steady_decodes = timing_repeats
                .iter()
                .map(|run| &run.steady_decode)
                .collect::<Vec<_>>();
            let mut timings = timing_samples_from_product_repeats(
                &kernel_runs,
                &steady_decodes,
                &row.coordinate,
                row.coordinate.process_temperature,
                &warmup_runs,
            )?;
            for (sample, state) in timings.samples.iter_mut().zip(repeat_host_states) {
                sample.host_state = Some(state);
            }
            let executable = std::env::current_exe().map_err(|e| e.to_string())?;
            let fixtures = sealed_product_fixture_artifacts(&suites, &row.coordinate)?;
            let receipt = product_receipt(
                &row.coordinate,
                &suites,
                timings,
                &executable,
                &fixtures,
                warmup_cache_state_version,
                &captured_source,
                admission,
                compressed.map(|method| {
                    (
                        method,
                        timing_warmups
                            .iter()
                            .chain(&timing_repeats)
                            .map(|run| &run.coordinate)
                            .collect::<Vec<_>>(),
                    )
                }),
                &reference_inventory,
                row_start,
            )?;
            let bundle = assemble_artifacts_with_fixtures(receipt, fixtures)?;
            write_artifacts(
                &PathBuf::from(required_flag(args, "--out")?),
                "receipt.json",
                "receipt.md",
                &bundle,
            )
            .map_err(|e| e.to_string())?;
            Ok(CampaignOutcome::Completed)
        }
        "noise-floor-parent" => noise_floor_parent(args),
        "noise-floor" => {
            let value = dense_noise_floor(
                Path::new(&required_flag(args, "--snapshot")?),
                Path::new(&required_flag(args, "--prompt-file")?),
                &required_flag(args, "--coordinate")?,
                optional_flag(args, "--prefill-chunk")?
                    .map(|chunk| chunk.parse::<usize>().map_err(|e| e.to_string()))
                    .transpose()?
                    .unwrap_or(crate::primitives::attention::SDPA_PREFILL_BLOCK_QLEN as usize),
                load_campaign_safety_policy(Path::new(&required_flag(args, "--safety-policy")?))?
                    .max_request_tokens,
            )?;
            let bytes = canonical_json_bytes(&value).map_err(|e| e.to_string())?;
            fs::write(required_flag(args, "--out")?, &bytes).map_err(|e| e.to_string())?;
            println!("{}", String::from_utf8(bytes).map_err(|e| e.to_string())?);
            Ok(CampaignOutcome::Completed)
        }
        _ => Err("usage: sc20671-kv-baseline parent|preflight|worker [--mode dense|compressed --kv-method <method>] [--only-coordinate <coordinate>] [--stop-file <path>] [options] | noise-floor --snapshot <dir> --prompt-file <file> --coordinate <coordinate> --safety-policy <json> --out <json> [--prefill-chunk <tokens>] | noise-floor-parent --llama-snapshot <dir> --qwen-snapshot <dir> --llama8b-snapshot <dir> --qwen8b-snapshot <dir> [--<family>-fp32-reference-snapshot <dir> ...] --prompt-file <file> --safety-policy <json> --resume-dir <dir> --out <dir> [--prefill-chunk <tokens>] [--stop-file <path>]".into()),
    }
}
