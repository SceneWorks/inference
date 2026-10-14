//! Private operator acceptance authority. Never populated from authored controls.
//! The worker must validate the canonical original request against its actual job before binding.
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub campaign_id: String,
    pub request_key: String,
    pub request_sha256: String,
    pub source_freeze_sha256: String,
    pub proof_id: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    SelfTestOutputFault,
    UnsupportedCapability,
    Capture,
}
impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SelfTestOutputFault => "self_test_output_fault",
            Self::UnsupportedCapability => "unsupported_capability",
            Self::Capture => "capture",
        }
    }
}
#[derive(Clone, Debug)]
struct Scopes {
    steps: Vec<usize>,
    layers: Vec<usize>,
    heads: Vec<usize>,
}
#[derive(Clone, Debug)]
struct Transport {
    directory: PathBuf,
    oracle_identity: String,
    timeout_ms: u64,
    deadline_ms: u64,
}
fn unix_ms() -> Result<u64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())
        .and_then(|d| u64::try_from(d.as_millis()).map_err(|e| e.to_string()))
}
#[derive(Debug)]
struct State {
    begun: bool,
    finished: bool,
    closed: bool,
    failure: Option<String>,
    file: Option<File>,
    hash: Sha256,
    records: u64,
    native_outputs: u64,
    selection_records: u64,
    dense_records: u64,
    pending_raw: bool,
    captured: std::collections::BTreeSet<(usize, usize, usize, usize)>,
    events: Vec<serde_json::Value>,
}
#[derive(Clone, Debug)]
pub struct Binding {
    context: Context,
    kind: Kind,
    job_id: String,
    worker_pid: u32,
    path: PathBuf,
    digest: String,
    scopes: Scopes,
    transport: Transport,
    state: Arc<Mutex<State>>,
}
impl PartialEq for Binding {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Binding {
    pub fn from_json(json: &str, expected: &Context, job_id: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("H3 acceptance authority JSON: {e}"))?;
        reject_duplicate_keys(json)?;
        let obj = value
            .as_object()
            .ok_or("H3 acceptance authority must be an object")?;
        let fields = [
            "schemaVersion",
            "campaignId",
            "requestKey",
            "requestSha256",
            "sourceFreezeSha256",
            "canonicalRequestJson",
            "proofId",
            "kind",
            "capturePath",
            "captureScopes",
            "captureTransport",
        ];
        if obj.len() != fields.len()
            || fields.iter().any(|k| !obj.contains_key(*k))
            || value["schemaVersion"] != 1
        {
            return Err("H3 acceptance authority schema mismatch".into());
        }
        let text = |key: &str| -> Result<&str, String> {
            value[key]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("H3 acceptance authority {key} must be a nonempty string"))
        };
        for (key, actual) in [
            ("campaignId", &expected.campaign_id),
            ("requestKey", &expected.request_key),
            ("requestSha256", &expected.request_sha256),
            ("sourceFreezeSha256", &expected.source_freeze_sha256),
            ("proofId", &expected.proof_id),
        ] {
            if text(key)? != actual {
                return Err(format!("H3 acceptance authority {key} mismatch"));
            }
        }
        if job_id.is_empty()
            || !digest(&expected.request_sha256)
            || !digest(&expected.source_freeze_sha256)
            || sha256(text("canonicalRequestJson")?.as_bytes()) != expected.request_sha256
        {
            return Err("H3 acceptance canonical request/source/job binding mismatch".into());
        }
        let request: serde_json::Value = serde_json::from_str(text("canonicalRequestJson")?)
            .map_err(|_| "H3 acceptance canonical request is malformed")?;
        reject_duplicate_keys(text("canonicalRequestJson")?)?;
        if !request.is_object() {
            return Err("H3 acceptance canonical request must be an object".into());
        }
        let kind = match text("kind")? {
            "self_test_output_fault" => Kind::SelfTestOutputFault,
            "unsupported_capability" => Kind::UnsupportedCapability,
            "capture" => Kind::Capture,
            _ => return Err("H3 acceptance unsupported proof kind".into()),
        };
        let path = PathBuf::from(text("capturePath")?);
        if !path.is_absolute()
            || path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(
                "H3 acceptance capture path must be absolute without parent traversal".into(),
            );
        }
        let scoped = &value["captureScopes"];
        if !scoped.as_object().is_some_and(|o| {
            o.len() == 3
                && ["steps", "layers", "heads"]
                    .iter()
                    .all(|k| o.contains_key(*k))
        }) {
            return Err("H3 acceptance captureScopes schema mismatch".into());
        }
        let indices = |key: &str| -> Result<Vec<usize>, String> {
            let a = scoped[key]
                .as_array()
                .filter(|a| a.len() <= 4096)
                .ok_or("H3 acceptance capture scope must be a bounded integer list")?;
            let mut v = Vec::new();
            for x in a {
                v.push(
                    x.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| *n < 4096)
                        .ok_or("H3 acceptance invalid scope index")?,
                );
            }
            let mut sorted = v.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != v.len() {
                return Err("H3 acceptance duplicate scope index".into());
            }
            Ok(v)
        };
        let scopes = Scopes {
            steps: indices("steps")?,
            layers: indices("layers")?,
            heads: indices("heads")?,
        };
        if kind != Kind::Capture
            && (!scopes.steps.is_empty() || !scopes.layers.is_empty() || !scopes.heads.is_empty())
        {
            return Err("H3 acceptance fault proof cannot declare raw capture scopes".into());
        }
        if [
            scopes.steps.is_empty(),
            scopes.layers.is_empty(),
            scopes.heads.is_empty(),
        ]
        .iter()
        .any(|x| *x)
            && !(scopes.steps.is_empty() && scopes.layers.is_empty() && scopes.heads.is_empty())
        {
            return Err("H3 acceptance partial empty capture scope".into());
        }
        let t = &value["captureTransport"];
        let tf = [
            "kind",
            "spoolDirectory",
            "oracleIdentitySha256",
            "ackTimeoutMs",
            "ownedDeadlineUnixMs",
            "maxPendingRecords",
        ];
        if !t
            .as_object()
            .is_some_and(|o| o.len() == tf.len() && tf.iter().all(|k| o.contains_key(*k)))
            || t["kind"] != "spool_ack_v1"
            || t["maxPendingRecords"] != 1
        {
            return Err("H3 acceptance captureTransport schema mismatch".into());
        }
        let directory = PathBuf::from(
            t["spoolDirectory"]
                .as_str()
                .ok_or("H3 acceptance spool directory missing")?,
        );
        let oracle_identity = t["oracleIdentitySha256"]
            .as_str()
            .filter(|s| digest(s))
            .ok_or("H3 acceptance oracle identity missing")?
            .to_owned();
        let timeout_ms = t["ackTimeoutMs"]
            .as_u64()
            .filter(|n| (1..=30000).contains(n))
            .ok_or("H3 acceptance ack timeout exceeds 30 seconds")?;
        let deadline_ms = t["ownedDeadlineUnixMs"]
            .as_u64()
            .ok_or("H3 acceptance owned deadline missing")?;
        let now = unix_ms()?;
        if deadline_ms <= now || deadline_ms - now > 3600000 {
            return Err("H3 acceptance expired or over-hour owned deadline".into());
        }
        if !directory.is_absolute()
            || directory.parent() != path.parent()
            || directory
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err("H3 acceptance spool must be an owned capture sibling directory".into());
        }
        let transport = Transport {
            directory,
            oracle_identity,
            timeout_ms,
            deadline_ms,
        };
        let worker_pid = std::process::id();
        let digest = sha256(
            &serde_json::to_vec(
                &serde_json::json!({"authority": value, "jobId":job_id,"workerPid":worker_pid}),
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(Self {
            context: expected.clone(),
            kind,
            job_id: job_id.into(),
            worker_pid,
            path,
            digest,
            scopes,
            transport,
            state: Arc::new(Mutex::new(State {
                begun: false,
                finished: false,
                closed: false,
                failure: None,
                file: None,
                hash: Sha256::new(),
                records: 0,
                native_outputs: 0,
                selection_records: 0,
                dense_records: 0,
                pending_raw: false,
                captured: std::collections::BTreeSet::new(),
                events: vec![],
            })),
        })
    }
    pub fn kind(&self) -> Kind {
        self.kind
    }
    pub fn begin(&self, native_source: serde_json::Value) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "H3 acceptance state poisoned")?;
        if self.worker_pid != std::process::id() || state.begun {
            return Err("H3 acceptance stale/reused worker binding".into());
        }
        // Consume even when opening fails; authority never migrates to a second generation.
        state.begun = true;
        state.file = Some(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&self.path)
                .map_err(|e| format!("H3 acceptance capture create_new: {e}"))?,
        );
        drop(state);
        self.record(serde_json::json!({"event":"binding","bindingSha256":self.digest,"campaignId":self.context.campaign_id,"requestKey":self.context.request_key,"requestSha256":self.context.request_sha256,"sourceFreezeSha256":self.context.source_freeze_sha256,"proofId":self.context.proof_id,"kind":self.kind.as_str(),"jobId":self.job_id,"workerPid":self.worker_pid,"nativeSource":native_source,"captureScopes":{"steps":self.scopes.steps,"layers":self.scopes.layers,"heads":self.scopes.heads},"oracleIdentitySha256":self.transport.oracle_identity}))
    }
    pub fn record(&self, mut value: serde_json::Value) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "H3 acceptance state poisoned")?;
        if self.worker_pid != std::process::id() || !state.begun || state.closed {
            return Err("H3 acceptance event outside bound generation lifetime".into());
        }
        if !value.is_object()
            || !matches!(
                value["event"].as_str(),
                Some(
                    "binding"
                        | "selection"
                        | "dense_attention"
                        | "self_test"
                        | "capability_guard"
                        | "native_output"
                        | "raw_record"
                        | "attention_call"
                        | "finish"
                        | "abort"
                )
            )
        {
            return Err("H3 acceptance malformed native event".into());
        }
        if state.pending_raw && value["event"] != "abort" {
            return Err("H3 acceptance compact record overlaps pending raw capture".into());
        }
        value["bindingSha256"] = self.digest.clone().into();
        value["recordIndex"] = state.records.into();
        let mut bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        state
            .file
            .as_mut()
            .ok_or("H3 acceptance capture is not open")?
            .write_all(&bytes)
            .map_err(|e| format!("H3 acceptance capture write: {e}"))?;
        state.hash.update(&bytes);
        state.records += 1;
        match value["event"].as_str() {
            Some("native_output") => state.native_outputs += 1,
            Some("selection" | "raw_record") => state.selection_records += 1,
            Some("dense_attention") => state.dense_records += 1,
            _ => {}
        }

        if matches!(
            value["event"].as_str(),
            Some("self_test" | "capability_guard")
        ) {
            state.events.push(value);
        }
        Ok(())
    }
    pub fn finish(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "H3 acceptance state poisoned")?;
        if !state.begun || state.closed || self.worker_pid != std::process::id() {
            return Err("H3 acceptance finish outside bound generation lifetime".into());
        }
        state
            .file
            .as_mut()
            .ok_or("H3 acceptance capture is not open")?
            .sync_all()
            .map_err(|e| format!("H3 acceptance capture sync: {e}"))?;
        if self.kind != Kind::Capture
            && !state.events.iter().any(|e| {
                e["applied"] == true
                    && e["rejected"] == true
                    && e["event"]
                        == (if self.kind == Kind::SelfTestOutputFault {
                            "self_test"
                        } else {
                            "capability_guard"
                        })
            })
        {
            return Err("H3 acceptance requested fault was not applied and rejected by its native guard/verifier".into());
        }
        if state.pending_raw {
            return Err("H3 acceptance raw capture remains unacknowledged".into());
        }
        for &layer in &self.scopes.layers {
            for &step in &self.scopes.steps {
                for &head in &self.scopes.heads {
                    if !state
                        .captured
                        .iter()
                        .any(|&(l, s, _, h)| l == layer && s == step && h == head)
                    {
                        return Err(format!("H3 acceptance declared scope missing layer={layer} step={step} head={head}"));
                    }
                }
            }
        }
        if state.native_outputs != 1 {
            return Err(
                "H3 acceptance requires exactly one actual native generation output".into(),
            );
        }
        let records_before = state.records;
        drop(state);
        self.record(serde_json::json!({"event":"finish","finished":true,"valid":true,"recordsBeforeFinish":records_before}))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| "H3 acceptance state poisoned")?;
        state
            .file
            .as_mut()
            .ok_or("H3 acceptance capture is not open")?
            .sync_all()
            .map_err(|e| format!("H3 acceptance capture sync: {e}"))?;
        let expected_hash = state.hash.clone().finalize();
        let file = state
            .file
            .as_mut()
            .ok_or("H3 acceptance capture is not open")?;
        file.rewind()
            .map_err(|e| format!("H3 acceptance capture rewind: {e}"))?;
        let mut observed_hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let len = file
                .read(&mut buffer)
                .map_err(|e| format!("H3 acceptance capture readback: {e}"))?;
            if len == 0 {
                break;
            }
            observed_hash.update(&buffer[..len]);
        }
        if observed_hash.finalize() != expected_hash {
            return Err("H3 acceptance capture mutated during generation".into());
        }
        state.file.take();
        state.finished = true;
        state.closed = true;
        Ok(())
    }
    /// Seal a failed/cancelled generation; no report clone can append after its job ends.
    pub fn abort(&self, error: &str) {
        let state = self.state.lock().expect("H3 acceptance state poisoned");
        if state.closed {
            return;
        }
        drop(state);
        let _ = self.record(
            serde_json::json!({"event":"abort","finished":false,"valid":false,"error":error}),
        );
        let mut state = self.state.lock().expect("H3 acceptance state poisoned");
        state.file.take();
        state.closed = true;
        state.failure = Some(error.into());
    }
    pub fn has_raw_scopes(&self) -> bool {
        self.kind == Kind::Capture && !self.scopes.steps.is_empty()
    }
    pub fn captures(&self, layer: usize, step: usize, head: usize) -> bool {
        self.kind == Kind::Capture
            && self.scopes.layers.contains(&layer)
            && self.scopes.steps.contains(&step)
            && self.scopes.heads.contains(&head)
    }
    pub fn record_raw(&self, value: serde_json::Value) -> Result<(), String> {
        let result = self.spool_raw(value);
        if let Err(error) = &result {
            self.abort(error);
        }
        result
    }
    fn spool_raw(&self, mut value: serde_json::Value) -> Result<(), String> {
        let index = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "H3 acceptance state poisoned")?;
            if !state.begun || state.closed || self.worker_pid != std::process::id() {
                return Err("H3 acceptance raw record outside generation lifetime".into());
            }
            if state.pending_raw {
                return Err("H3 acceptance more than one pending raw record".into());
            }
            state.pending_raw = true;
            state.records
        };
        let coord = |key: &str| {
            value[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("H3 acceptance raw {key} missing"))
        };
        let (layer, step, batch) = (coord("layer")?, coord("step")?, coord("batch")?);
        let heads = value["heads"]
            .as_array()
            .filter(|a| a.len() == 1)
            .ok_or("H3 acceptance raw capture must contain exactly one head")?;
        let head = heads[0]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("H3 acceptance raw head invalid")?;
        if value["event"] != "selection" || !self.captures(layer, step, head) {
            return Err("H3 acceptance raw capture outside declared scope".into());
        }
        let meta = std::fs::symlink_metadata(&self.transport.directory)
            .map_err(|e| format!("H3 acceptance spool directory: {e}"))?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("H3 acceptance unsafe spool directory".into());
        }
        value["bindingSha256"] = self.digest.clone().into();
        value["recordIndex"] = index.into();
        let raw = self
            .transport
            .directory
            .join(format!("{}-{index}.raw.json", self.digest));
        let ack = self
            .transport
            .directory
            .join(format!("{}-{index}.ack.json", self.digest));
        let bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
        drop(value);
        let raw_sha = sha256(&bytes);
        let raw_bytes = bytes.len();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&raw)
            .map_err(|e| format!("H3 acceptance raw create_new: {e}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))
                .map_err(|e| e.to_string())?;
        }
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        drop(bytes);
        // Ready marker is published only after the entire raw record is immutable and closed.
        let ready = self
            .transport
            .directory
            .join(format!("{}-{index}.ready.json", self.digest));
        let notice=serde_json::to_vec(&serde_json::json!({"schemaVersion":1,"bindingSha256":self.digest,"recordIndex":index,"rawSha256":raw_sha,"rawBytes":raw_bytes,"rawPath":raw,"ackPath":ack,"oracleIdentitySha256":self.transport.oracle_identity})).map_err(|e|e.to_string())?;
        let pending = ready.with_extension("pending");
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)
            .map_err(|e| e.to_string())?;
        f.write_all(&notice).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        std::fs::hard_link(&pending, &ready).map_err(|e| e.to_string())?;
        std::fs::remove_file(&pending).map_err(|e| e.to_string())?;
        let started = std::time::Instant::now();
        loop {
            if unix_ms()? >= self.transport.deadline_ms
                || started.elapsed().as_millis() >= u128::from(self.transport.timeout_ms)
            {
                return Err("H3 acceptance oracle ack timeout/owned deadline".into());
            }
            match std::fs::symlink_metadata(&ack) {
                Ok(meta) => {
                    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 65536 {
                        return Err("H3 acceptance unsafe oracle ack".into());
                    }
                    let bytes = std::fs::read(&ack).map_err(|e| e.to_string())?;
                    let a: serde_json::Value = serde_json::from_slice(&bytes)
                        .map_err(|e| format!("H3 acceptance oracle ack malformed: {e}"))?;
                    reject_duplicate_keys(std::str::from_utf8(&bytes).map_err(|e| e.to_string())?)?;
                    let fields = [
                        "schemaVersion",
                        "bindingSha256",
                        "recordIndex",
                        "rawSha256",
                        "oracleIdentitySha256",
                        "verdict",
                        "ledgerEntrySha256",
                    ];
                    if !a.as_object().is_some_and(|o| {
                        o.len() == fields.len() && fields.iter().all(|k| o.contains_key(*k))
                    }) || a["schemaVersion"] != 1
                        || a["bindingSha256"] != self.digest
                        || a["recordIndex"] != index
                        || a["rawSha256"] != raw_sha
                        || a["oracleIdentitySha256"] != self.transport.oracle_identity
                        || a["verdict"] != "verified"
                        || !a["ledgerEntrySha256"].as_str().is_some_and(digest)
                    {
                        return Err(
                            "H3 acceptance oracle ack binding/index/raw/identity/verdict mismatch"
                                .into(),
                        );
                    }
                    if file_sha256(&raw)? != raw_sha {
                        return Err("H3 acceptance raw capture mutated after oracle read".into());
                    }
                    {
                        let mut state = self
                            .state
                            .lock()
                            .map_err(|_| "H3 acceptance state poisoned")?;
                        if state.records != index || !state.pending_raw || state.closed {
                            return Err("H3 acceptance pending raw lifetime changed".into());
                        }
                        state.pending_raw = false;
                        if !state.captured.insert((layer, step, batch, head)) {
                            return Err("H3 acceptance duplicate raw scope tuple".into());
                        }
                    }
                    self.record(serde_json::json!({"event":"raw_record","layer":layer,"step":step,"batch":batch,"heads":[head],"rawEvent":"selection","rawSha256":raw_sha,"rawBytes":raw_bytes,"rawPath":raw,"ackSha256":sha256(&bytes),"oracleIdentitySha256":self.transport.oracle_identity,"ledgerEntrySha256":a["ledgerEntrySha256"],"transportAcknowledged":true,"rawRetention":"deleted_after_independent_oracle_ack"}))?;
                    for p in [&raw, &ack, &ready] {
                        std::fs::remove_file(p)
                            .map_err(|e| format!("H3 acceptance raw retirement: {e}"))?;
                    }
                    return Ok(());
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    std::thread::sleep(std::time::Duration::from_millis(25))
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }
    pub fn receipt(&self) -> serde_json::Value {
        let state = self.state.lock().expect("H3 acceptance state poisoned");
        serde_json::json!({"schemaVersion":1,"bindingSha256":self.digest,"campaignId":self.context.campaign_id,"requestKey":self.context.request_key,"requestSha256":self.context.request_sha256,"sourceFreezeSha256":self.context.source_freeze_sha256,"proofId":self.context.proof_id,"kind":self.kind.as_str(),"jobId":self.job_id,"workerPid":self.worker_pid,"injectedProof":self.kind != Kind::Capture,"capturePath":self.path,"captureSha256":format!("{:x}",state.hash.clone().finalize()),"records":state.records,"nativeOutputRecords":state.native_outputs,"selectionRecords":state.selection_records,"denseRecords":state.dense_records,"capturedScopeTuples":state.captured.len(),"pendingRaw":state.pending_raw,"finished":state.finished,"aborted":state.closed && !state.finished,"error":state.failure,"observations":state.events})
    }
}

pub fn f32_sha256(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for x in values {
        hash.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
pub fn output_record(output: &crate::GenerationOutput) -> serde_json::Value {
    match output {
        crate::GenerationOutput::Video { frames, fps, audio } => {
            serde_json::json!({"event":"native_output","fps":fps,"frames":frames.iter().map(|f| serde_json::json!({"width":f.width,"height":f.height,"pixelsSha256":sha256(&f.pixels)})).collect::<Vec<_>>(),"audio":audio.as_ref().map(|a| serde_json::json!({"sampleRate":a.sample_rate,"channels":a.channels,"sampleCount":a.samples.len(),"samplesSha256":f32_sha256(&a.samples)}))})
        }
        _ => serde_json::json!({"event":"native_output","unsupportedOutputKind":true}),
    }
}
pub fn budget_value(value: crate::VedaBudget) -> serde_json::Value {
    match value {
        crate::VedaBudget::SparsityPercent(p) => serde_json::json!({"ratio":1.0-p/100.0}),
        crate::VedaBudget::KeyTiles(n) => serde_json::json!({"tiles":n}),
    }
}

// Parse only top-level key token boundaries after serde_json has validated the syntax.
// Escaped keys are decoded before comparison, so aliases cannot bypass the closed schema.
fn reject_duplicate_keys(json: &str) -> Result<(), String> {
    let bytes = json.as_bytes();
    let mut objects: Vec<std::collections::BTreeSet<String>> = vec![];
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                objects.push(std::collections::BTreeSet::new());
                i += 1;
            }
            b'}' => {
                objects.pop();
                i += 1;
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                let mut next = i;
                while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                    next += 1;
                }
                if bytes.get(next) == Some(&b':') {
                    let key: String =
                        serde_json::from_str(&json[start..i]).map_err(|e| e.to_string())?;
                    if !objects
                        .last_mut()
                        .ok_or("H3 acceptance JSON key outside object")?
                        .insert(key)
                    {
                        return Err("H3 acceptance duplicate authority key".into());
                    }
                }
            }
            _ => i += 1,
        }
    }
    Ok(())
}

fn file_sha256(path: &std::path::Path) -> Result<String, String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err("H3 acceptance raw capture identity changed".into());
    }
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut bytes = [0u8; 65536];
    loop {
        let n = file.read(&mut bytes).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Conservative one-head capture serialization bound, separate from allocator/device peaks.
/// It includes readback/cast copies, promoted projections, JSON Value arrays and the growable
/// serialized byte buffer. The separately admitted oracle process is not included here.
pub fn capture_bound(
    layout: &super::contract::Layout,
    dim: usize,
) -> Result<serde_json::Value, String> {
    if dim == 0
        || dim > 128
        || layout.video > layout.valid.len()
        || layout.valid.len().checked_mul(super::contract::TILE) != Some(layout.perm.len())
    {
        return Err("H3 acceptance unsupported capture bound layout/dimension".into());
    }
    let slots = layout.perm.len() as u128;
    let tiles = layout.valid.len() as u128;
    let video = layout.video as u128;
    let dim = dim as u128;
    let qk = 2 * slots * dim;
    let projections = 6 * dim * dim;
    let logits = video * video;
    let floats = qk + projections + logits;
    let integers = slots + tiles + tiles * tiles;
    // Finite f32-promoted f64 JSON numbers fit in 32 bytes including separator; signed
    // indices/usize fit in 24. Allow 4 MiB for names, structural punctuation and row vectors.
    let metadata = 4u128 << 20;
    let encoded = 8 * qk + 32 * (projections + logits) + 24 * integers + metadata;
    let native = 2 * encoded
        + 16 * qk
        + 64 * (projections + logits + integers)
        + 8 * qk
        + 4 * projections
        + metadata;
    let n = |v| u64::try_from(v).map_err(|_| "H3 acceptance capture bound overflow".to_owned());
    Ok(
        serde_json::json!({"scope":"one head complete declared selection record; oracle process excluded","heads":1,"slots":layout.perm.len(),"dim":n(dim)?,"floatScalars":n(floats)?,"maxIntegerScalars":n(integers)?,"encodedBytesUpperBound":n(encoded)?,"nativeTransientBytesUpperBound":n(native)?,"deviceReadbackAndCastBytesUpperBound":n(8*qk)?,"method":"8bytes_lossless_QK_hex_2x_string_32bytes_other_finite_float_24bytes_index_64bytes_JSON_value_2x_encoded_buffer_plus_cast_readback_and_projection"}),
    )
}

pub fn capture_spans(layout: &super::contract::Layout) -> Result<serde_json::Value, String> {
    let spans = layout
        .capture_spans
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or("H3 acceptance actual construction spans were not retained")?;
    Ok(serde_json::Value::Array(
        spans
            .iter()
            .map(|s| serde_json::json!({"start":s.start,"grid":s.grid,"shape":s.shape.0}))
            .collect(),
    ))
}
/// Lossless float-bit transport; avoids a JSON/Python float object for every QK scalar.
pub fn f32_capture(values: &[f32], shape: &[usize]) -> Result<serde_json::Value, String> {
    if shape.contains(&0)
        || shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)) != Some(values.len())
        || values.iter().any(|x| !x.is_finite())
    {
        return Err("H3 acceptance float capture shape/nonfinite mismatch".into());
    }
    let size = values
        .len()
        .checked_mul(8)
        .ok_or("H3 acceptance float capture size overflow")?;
    let mut data = String::new();
    data.try_reserve_exact(size)
        .map_err(|e| format!("H3 acceptance float capture allocation: {e}"))?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            data.push(HEX[(byte >> 4) as usize] as char);
            data.push(HEX[(byte & 15) as usize] as char);
        }
    }
    Ok(
        serde_json::json!({"encoding":"f32_le_hex","shape":shape,"data":data,"sha256":f32_sha256(values)}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn authority(root: &std::path::Path, kind: &str) -> (String, Context) {
        let canonical = r#"{"project":"s9","advanced":{"s9Harness":"exact"},"duration":5.0}"#;
        let context = Context {
            campaign_id: "campaign".into(),
            request_key: "row".into(),
            request_sha256: sha256(canonical.as_bytes()),
            source_freeze_sha256: "ab".repeat(32),
            proof_id: "proof".into(),
        };
        (serde_json::json!({"schemaVersion":1,"campaignId":context.campaign_id,"requestKey":context.request_key,"requestSha256":context.request_sha256,"sourceFreezeSha256":context.source_freeze_sha256,"proofId":context.proof_id,"canonicalRequestJson":canonical,"kind":kind,"capturePath":root.join("capture.jsonl"),"captureScopes":{"steps":[],"layers":[],"heads":[]},"captureTransport":{"kind":"spool_ack_v1","spoolDirectory":root.join("raw"),"oracleIdentitySha256":"cd".repeat(32),"ackTimeoutMs":30000,"ownedDeadlineUnixMs":unix_ms().unwrap()+60000,"maxPendingRecords":1}}).to_string(),context)
    }
    #[test]
    fn source_owned_descriptor_matches_actual_native_proof_and_capture_surfaces() {
        let descriptor: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../config/acceptance/h3-owned-product.json"
        ))
        .unwrap();
        assert_eq!(descriptor["schemaVersion"], 1);
        assert_eq!(descriptor["kind"], "h3_owned_product_acceptance_contract");
        for flag in [
            "defaultOff",
            "realVerifier",
            "realCapabilityGuard",
            "requestPrivate",
        ] {
            assert_eq!(descriptor[flag], true);
        }
        let mlx = include_str!("../../../../media/mlx-gen/mlx-gen-minimax-h3/src/veda/mod.rs");
        let candle = include_str!("../../../../media/candle-gen/candle-gen-minimax-h3/src/veda.rs");
        for (backend, source) in [("mlx", mlx), ("candle_cuda", candle)] {
            for kind in ["self_test_output_fault", "unsupported_capability"] {
                assert!(source.contains(
                    descriptor["fallbackReasons"][backend][kind]
                        .as_str()
                        .unwrap()
                ));
            }
            for field in [
                "scores",
                "rows",
                "perm",
                "valid",
                "refTokens",
                "targetTokens",
                "generatedBudget",
                "referenceBudget",
                "qTiled",
                "kTiled",
                "projectionQ",
                "projectionK",
                "predictorSha256",
                "spans",
                "layer",
                "step",
                "batch",
                "heads",
                "dtype",
            ] {
                assert!(
                    source.contains(&format!("\"{field}\"")),
                    "{backend}/{field}"
                );
            }
            assert!(source.contains("Kind::SelfTestOutputFault"));
            assert!(source.contains("Kind::UnsupportedCapability"));
            assert!(source.contains("sparseDeviceComputeCompleted"));
        }
        assert!(mlx.contains("error > SELF_TEST_ATOL"));
        assert!(candle.contains("2. * ulp * b.abs() + 2e-4"));
    }
    #[test]
    fn mutation_of_emitted_capture_before_finish_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        let binding = Binding::from_json(&json, &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        binding
            .record(serde_json::json!({"event":"native_output"}))
            .unwrap();
        std::fs::write(root.path().join("capture.jsonl"), "changed").unwrap();
        assert!(binding.finish().unwrap_err().contains("mutated"));
        assert_eq!(binding.receipt()["finished"], false);
    }
    #[test]
    fn scope_and_transport_schema_mutations_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        for scope in [
            serde_json::json!({"steps":[0],"layers":[],"heads":[0]}),
            serde_json::json!({"steps":[0,0],"layers":[0],"heads":[0]}),
            serde_json::json!({"steps":[-1],"layers":[0],"heads":[0]}),
            serde_json::json!({"steps":[0],"layers":[0],"heads":[0],"unknown":true}),
        ] {
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value["captureScopes"] = scope;
            assert!(Binding::from_json(&value.to_string(), &context, "job").is_err());
        }
        for key in [
            "kind",
            "spoolDirectory",
            "oracleIdentitySha256",
            "ackTimeoutMs",
            "ownedDeadlineUnixMs",
            "maxPendingRecords",
        ] {
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value["captureTransport"][key] = "bad".into();
            assert!(
                Binding::from_json(&value.to_string(), &context, "job").is_err(),
                "{key}"
            );
        }
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["captureScopes"] = serde_json::json!({"steps":[0],"layers":[0],"heads":[0]});
        let binding = Binding::from_json(&value.to_string(), &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        binding
            .record(serde_json::json!({"event":"native_output"}))
            .unwrap();
        assert!(binding.finish().unwrap_err().contains("scope missing"));
    }
    #[test]
    fn raw_spool_retires_only_matching_ack_and_seals_an_explicit_finish() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["captureScopes"] = serde_json::json!({"steps":[0],"layers":[0],"heads":[0]});
        let binding = Binding::from_json(&value.to_string(), &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        let rawdir = root.path().join("raw");
        std::fs::create_dir(&rawdir).unwrap();
        let dir = rawdir.clone();
        let consumer = std::thread::spawn(move || {
            let notice = loop {
                let next = std::fs::read_dir(&dir).unwrap().find_map(|e| {
                    let p = e.ok()?.path();
                    p.file_name()?
                        .to_str()?
                        .ends_with(".ready.json")
                        .then_some(p)
                });
                if let Some(path) = next {
                    break path;
                }
                std::thread::yield_now();
            };
            let n: serde_json::Value =
                serde_json::from_slice(&std::fs::read(notice).unwrap()).unwrap();
            let path = std::path::Path::new(n["rawPath"].as_str().unwrap());
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(sha256(&bytes), n["rawSha256"]);
            let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(raw["bindingSha256"], n["bindingSha256"]);
            assert_eq!(raw["recordIndex"], n["recordIndex"]);
            let ack = serde_json::json!({"schemaVersion":1,"bindingSha256":n["bindingSha256"],"recordIndex":n["recordIndex"],"rawSha256":n["rawSha256"],"oracleIdentitySha256":n["oracleIdentitySha256"],"verdict":"verified","ledgerEntrySha256":sha256(b"transport fixture only, no device/oracle acceptance claim")});
            let ackpath = std::path::Path::new(n["ackPath"].as_str().unwrap());
            let temp = ackpath.with_extension("pending");
            std::fs::write(&temp, serde_json::to_vec(&ack).unwrap()).unwrap();
            std::fs::rename(temp, ackpath).unwrap();
        });
        binding.record_raw(serde_json::json!({"event":"selection","layer":0,"step":0,"batch":0,"heads":[0],"qTiled":[1.0],"kTiled":[2.0]})).unwrap();
        consumer.join().unwrap();
        assert_eq!(std::fs::read_dir(&rawdir).unwrap().count(), 0);
        binding
            .record(serde_json::json!({"event":"native_output"}))
            .unwrap();
        binding.finish().unwrap();
        let capture = std::fs::read_to_string(root.path().join("capture.jsonl")).unwrap();
        let last: serde_json::Value =
            serde_json::from_str(capture.lines().last().unwrap()).unwrap();
        assert_eq!(last["event"], "finish");
        assert_eq!(last["valid"], true);
        assert_eq!(last["recordsBeforeFinish"], 3);
        assert_eq!(binding.receipt()["capturedScopeTuples"], 1);
    }
    #[test]
    fn a_foreign_ack_cannot_retire_raw_or_finish_successfully() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["captureScopes"] = serde_json::json!({"steps":[0],"layers":[0],"heads":[0]});
        let binding = Binding::from_json(&value.to_string(), &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        let rawdir = root.path().join("raw");
        std::fs::create_dir(&rawdir).unwrap();
        let ack = rawdir.join(format!("{}-1.ack.json", binding.digest));
        std::fs::write(ack, b"{}").unwrap();
        assert!(binding
            .record_raw(
                serde_json::json!({"event":"selection","layer":0,"step":0,"batch":0,"heads":[0]})
            )
            .is_err());
        assert_eq!(binding.receipt()["aborted"], true);
        assert_eq!(binding.receipt()["finished"], false);
        assert!(binding.finish().is_err());
        assert!(std::fs::read_dir(&rawdir).unwrap().any(|e| e
            .unwrap()
            .path()
            .to_string_lossy()
            .ends_with(".raw.json")));
        let capture = std::fs::read_to_string(root.path().join("capture.jsonl")).unwrap();
        let last: serde_json::Value =
            serde_json::from_str(capture.lines().last().unwrap()).unwrap();
        assert_eq!(last["event"], "abort");
        assert_eq!(last["valid"], false);
    }
    #[test]
    fn one_head_bound_covers_lossless_encoded_payload_and_rejects_invalid_layout() {
        use super::super::contract::{Layout, TileShape, TILE};
        let layout = Layout::t2va(17, [2, 9, 17], TileShape([2, 8, 8]), 323).unwrap();
        let dim = 16;
        let bound = capture_bound(&layout, dim).unwrap();
        let values = vec![0.25; layout.perm.len() * dim];
        let shape = [1, layout.valid.len(), TILE, dim];
        let payload = serde_json::json!({"qTiled":f32_capture(&values,&shape).unwrap(),"kTiled":f32_capture(&values,&shape).unwrap(),"projectionQ":vec![0.5;3*dim*dim],"projectionK":vec![0.5;3*dim*dim],"scores":vec![0.1;layout.video*layout.video],"rows":vec![(0..layout.valid.len()).collect::<Vec<_>>();layout.valid.len()],"valid":layout.valid,"perm":layout.perm});
        assert!(
            serde_json::to_vec(&payload).unwrap().len() as u64
                <= bound["encodedBytesUpperBound"].as_u64().unwrap()
        );
        assert!(
            bound["nativeTransientBytesUpperBound"].as_u64().unwrap()
                > bound["encodedBytesUpperBound"].as_u64().unwrap()
        );
        assert!(capture_bound(&layout, usize::MAX).is_err());
        assert!(capture_bound(&layout, 0).is_err());
        let mut invalid = layout;
        invalid.video = usize::MAX;
        assert!(capture_bound(&invalid, 128).is_err());
    }
    #[test]
    fn captured_spans_are_exact_ordered_production_inputs_and_required_for_raw_rebuild() {
        use super::super::contract::{Layout, TileShape, TiledSpan};
        let spans = vec![
            TiledSpan {
                start: 4,
                grid: [1, 3, 17],
                shape: TileShape([1, 8, 16]),
            },
            TiledSpan {
                start: 60,
                grid: [3, 5, 19],
                shape: TileShape([2, 4, 16]),
            },
        ];
        let mut layout = Layout::from_spans(&spans, 349).unwrap();
        assert!(capture_spans(&layout).is_err());
        layout.capture_spans = Some(spans.clone());
        let value = capture_spans(&layout).unwrap();
        assert_eq!(value[0]["start"], 4);
        assert_eq!(value[1]["grid"], serde_json::json!([3, 5, 19]));
        let rebuilt =
            Layout::from_spans(layout.capture_spans.as_ref().unwrap(), layout.seq).unwrap();
        assert_eq!(rebuilt.perm, layout.perm);
        assert_eq!(rebuilt.valid, layout.valid);
        let mut mutated = spans;
        mutated[1].start += 1;
        let different = Layout::from_spans(&mutated, layout.seq).unwrap();
        assert_ne!(different.perm, layout.perm);
        mutated.swap(0, 1);
        assert!(Layout::from_spans(&mutated, layout.seq).is_err());
    }
    #[test]
    fn lossless_qk_capture_preserves_actual_bits_without_decimal_roundtrips() {
        let packed = f32_capture(&[0., -0., 1., -1.], &[1, 4]).unwrap();
        assert_eq!(packed["encoding"], "f32_le_hex");
        assert_eq!(packed["data"], "00000000000000800000803f000080bf");
        assert_eq!(packed["sha256"], f32_sha256(&[0., -0., 1., -1.]));
        assert!(f32_capture(&[1.], &[2]).is_err());
        assert!(f32_capture(&[f32::NAN], &[1]).is_err());
        assert!(f32_capture(&[], &[0]).is_err());
    }
    #[test]
    fn defaults_do_not_create_acceptance_authority_or_report_fields() {
        let req = crate::GenerationRequest::default();
        assert!(req.h3_terminal_acceptance.is_none());
        assert!(crate::VedaExecutionReport::default()
            .outcome_json()
            .get("acceptanceProof")
            .is_none());
    }
    #[test]
    fn malformed_cross_request_stale_source_and_mutated_authority_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        for key in [
            "schemaVersion",
            "campaignId",
            "requestKey",
            "requestSha256",
            "sourceFreezeSha256",
            "proofId",
            "canonicalRequestJson",
            "kind",
            "capturePath",
        ] {
            let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
            v[key] = serde_json::json!("invalid");
            assert!(
                Binding::from_json(&v.to_string(), &context, "job").is_err(),
                "{key}"
            );
        }
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        v["unknown"] = true.into();
        assert!(Binding::from_json(&v.to_string(), &context, "job").is_err());
        assert!(Binding::from_json(&json, &context, "").is_err());
        assert!(Binding::from_json(
            &format!("{{\"campaignId\":\"campaign\",{}", &json[1..]),
            &context,
            "job"
        )
        .is_err());
        assert!(Binding::from_json(&json.replace("5.0", "5.00"), &context, "job").is_err());
    }
    #[test]
    fn generation_lifetime_is_single_use_shared_by_clones_and_process_bound() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        let binding = Binding::from_json(&json, &context, "job").unwrap();
        assert!(binding
            .record(serde_json::json!({"event":"too_early"}))
            .is_err());
        binding
            .begin(serde_json::json!({"source":"actual"}))
            .unwrap();
        assert!(binding.clone().begin(serde_json::json!({})).is_err());
        binding
            .record(serde_json::json!({"event":"native_output"}))
            .unwrap();
        binding.finish().unwrap();
        assert!(binding.record(serde_json::json!({})).is_err());
        assert!(binding.finish().is_err());
        let receipt = binding.receipt();
        assert_eq!(
            receipt["captureSha256"],
            sha256(&std::fs::read(root.path().join("capture.jsonl")).unwrap())
        );
        assert_eq!(receipt["records"], 3);
        assert_eq!(receipt["finished"], true);
        assert_eq!(receipt["workerPid"], std::process::id());
        let records = std::fs::read_to_string(root.path().join("capture.jsonl")).unwrap();
        for (i, line) in records.lines().enumerate() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["bindingSha256"], receipt["bindingSha256"]);
            assert_eq!(v["recordIndex"], i);
        }
        std::fs::write(root.path().join("capture.jsonl"), "mutated").unwrap();
        assert_ne!(
            receipt["captureSha256"],
            sha256(&std::fs::read(root.path().join("capture.jsonl")).unwrap())
        );
        let mut forged = binding.clone();
        forged.worker_pid = std::process::id().wrapping_add(1);
        assert!(forged.begin(serde_json::json!({})).is_err());
    }
    #[test]
    fn failed_generation_seals_capture_and_report_clone_cannot_append() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "capture");
        let binding = Binding::from_json(&json, &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        binding.abort("cancelled");
        assert!(binding
            .clone()
            .record(serde_json::json!({"event":"native_output"}))
            .is_err());
        assert!(binding.finish().is_err());
        assert_eq!(binding.receipt()["aborted"], true);
        assert_eq!(binding.receipt()["error"], "cancelled");
    }
    #[test]
    fn missing_fault_reachability_never_finishes_as_a_successful_proof() {
        let root = tempfile::tempdir().unwrap();
        let (json, context) = authority(root.path(), "self_test_output_fault");
        let binding = Binding::from_json(&json, &context, "job").unwrap();
        binding.begin(serde_json::json!({})).unwrap();
        binding
            .record(serde_json::json!({"event":"self_test","applied":false,"rejected":true}))
            .unwrap();
        assert!(binding.finish().is_err());
        binding
            .record(serde_json::json!({"event":"capability_guard","applied":true,"rejected":true}))
            .unwrap();
        assert!(binding.finish().is_err());
        binding
            .record(serde_json::json!({"event":"self_test","applied":true,"rejected":true}))
            .unwrap();
        binding
            .record(serde_json::json!({"event":"native_output"}))
            .unwrap();
        binding.finish().unwrap();
        assert_eq!(binding.receipt()["injectedProof"], true);
    }
    #[test]
    fn byte_identity_keeps_signed_zero_and_all_frames_and_audio() {
        assert_ne!(f32_sha256(&[0.]), f32_sha256(&[-0.]));
        let video = crate::GenerationOutput::Video {
            frames: vec![
                crate::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![1, 2, 3],
                },
                crate::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![4, 5, 6],
                },
            ],
            fps: 25,
            audio: Some(crate::AudioTrack {
                samples: vec![0., -0., 1.],
                sample_rate: 32000,
                channels: 1,
                stems: vec![],
            }),
        };
        let v = output_record(&video);
        assert_eq!(v["frames"].as_array().unwrap().len(), 2);
        assert_eq!(v["audio"]["samplesSha256"], f32_sha256(&[0., -0., 1.]));
        assert_eq!(v["audio"]["sampleCount"], 3);
        assert_ne!(
            v["frames"][0]["pixelsSha256"],
            v["frames"][1]["pixelsSha256"]
        );
    }
}
