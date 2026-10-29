//! Artifact seals: content-digest identity, the proven-digest cache, single-flight sealing, and sealed reads.

use super::*;

/// What work an artifact-seal observation actually represents. Metadata is not a content read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactSealAcquisition {
    Metadata,
    ColdContentRead,
    MemoryMemo,
    PersistedMemo,
    DriftAudit,
    /// A caller-supplied digest whose bound target identity matched at pin time (sc-26691).
    Attested,
    /// A seal carried across a hard link this seal itself created (sc-26691).
    Relinked,
}

/// Completing a read does not authorize a seal; only Acquired follows identity validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactSealProgressState {
    Reading,
    ReadComplete,
    Acquired,
    Canceled,
    Failed,
}

/// Observational source work, never an authorization token or an estimated fraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSealProgress {
    pub loader_path: PathBuf,
    pub acquisition: ArtifactSealAcquisition,
    pub state: ArtifactSealProgressState,
    /// Actual bytes fed to SHA-256 in this read; memo acquisition always reports zero.
    pub hashed_bytes: u64,
    /// Declared file size at capture/open, not a cap on actual reads. Unknown before metadata.
    pub source_bytes: Option<u64>,
}

pub(super) fn seal_progress(
    path: &Path,
    acquisition: ArtifactSealAcquisition,
    state: ArtifactSealProgressState,
    hashed_bytes: u64,
    source_bytes: Option<u64>,
) -> ArtifactSealProgress {
    ArtifactSealProgress {
        loader_path: path.to_path_buf(),
        acquisition,
        state,
        hashed_bytes,
        source_bytes,
    }
}

/// A re-openable single-file weights source pinned without canonicalizing its loader path.
///
/// The retained path is absolute but otherwise lexical: an extension-bearing snapshot symlink stays
/// extension-bearing. Every window can call [`ensure_unchanged`](Self::ensure_unchanged) before
/// re-opening it, which detects persistent or identity-changing mutation/replacement of the lexical
/// parent chain, entry (`lstat`), resolution, and resolved target. The fingerprints are also suitable
/// for cache/provenance identity without hashing a multi-gigabyte checkpoint on each request. This
/// remains a path-token model; see [`read_unchanged`](Self::read_unchanged) for its active-swap bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactSeal {
    pub(super) loader_path: PathBuf,
    pub(super) canonical_target_path: PathBuf,
    pub(super) path_component_fingerprints: Vec<PathComponentFingerprint>,
    pub(super) entry_fingerprint: FileStatFingerprint,
    pub(super) target_fingerprint: FileStatFingerprint,
    pub(super) sha256: [u8; 32],
    #[cfg(test)]
    pub(super) hash_work: Arc<SealHashWork>,
}

/// Backwards-compatible name for the shared artifact seal.
///
/// Keep this alias while providers migrate their local terminology. Every existing file-pin
/// consumer therefore retains the canonical target, no-follow entry pin, and acquisition-time
/// content identity supplied by [`ArtifactSeal`].
pub type PinnedWeightsFile = ArtifactSeal;

/// Read-only collection of caller-prepared File identities carried by a [`LoadSpec`].
///
/// The private `prepared` bit distinguishes ordinary compatibility mode from an intentionally
/// prepared spec even when the spec has no File sources. Callers can inspect the map for cache-key
/// construction, but can only transition or add tokens through [`LoadSpec`] methods; in particular,
/// there is no mutable-map `clear` that could silently downgrade a prepared spec to compatibility
/// mode. Besides configured [`WeightsSource::File`] slots, a prepared spec may retain exact files
/// below a configured [`WeightsSource::Dir`] (for example a transformer directory's config and
/// weights). Those directory-member tokens participate in cache identity and guard the provider
/// callback without changing the provider's directory-shaped load contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PreparedFilePins {
    pub(super) prepared: bool,
    pub(super) finalized: bool,
    // Sorted unique lexical keys. A Vec makes the requested payload layout public
    // and finite for guarded binding; no private std BTree node ABI is assumed.
    pub(super) pins: Vec<(PathBuf, PinnedWeightsFile)>,
}

impl PreparedFilePins {
    /// Whether this spec has explicitly entered prepared mode, including a prepared Dir-only spec.
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }

    /// Whether the caller has installed the complete token set and finalized it for consumption.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pins.len()
    }

    pub fn get(&self, path: &Path) -> Option<&PinnedWeightsFile> {
        self.pins
            .binary_search_by(|(key, _)| key.as_path().cmp(path))
            .ok()
            .map(|index| &self.pins[index].1)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PathBuf, &PinnedWeightsFile)> {
        self.pins.iter().map(|(path, pin)| (path, pin))
    }

    pub fn keys(&self) -> impl Iterator<Item = &PathBuf> {
        self.pins.iter().map(|(path, _)| path)
    }
}

/// Everything [`ArtifactSeal::ensure_unchanged`] compares, captured before the content read.
///
/// Deliberately the WHOLE of the seal's own "unchanged" predicate — lexical path, every parent
/// component's fingerprint, the no-follow entry, the resolution, and the resolved target — not a
/// subset. Two pins agreeing on all of it are, by the seal's own definition, pins of the same
/// unmutated object, which is what makes reusing a digest between them sound rather than merely
/// convenient.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct SealIdentity {
    pub(super) loader_path: PathBuf,
    pub(super) canonical_target_path: PathBuf,
    pub(super) path_component_fingerprints: Vec<PathComponentFingerprint>,
    pub(super) entry_fingerprint: FileStatFingerprint,
    pub(super) target_fingerprint: FileStatFingerprint,
}

/// Digests already proven for an exact [`SealIdentity`], reused for the life of the process.
///
/// [`ArtifactSeal`]'s own doc promises fingerprints "suitable for cache/provenance identity without
/// hashing a multi-gigabyte checkpoint on each request". [`ArtifactSeal::pin`] hashed on every
/// acquisition anyway, and nothing upstream pins once: a single image job acquires the same text
/// encoder from at least two independent pre-load queries — `ProviderRegistry::footprint` and
/// `ProviderRegistry::memory_strategy_contract` — before the loader runs at all.
///
/// Measured on an Apple Silicon host, one krea_2_turbo job with an 8.3 GB bf16 encoder: a 55-second
/// stack sample spent 74% of the window under one of those queries, 97% of that subtree inside the
/// SHA-256 compressor and 3% in `read`, while a parallel I/O trace showed the source disk idle for
/// all 58 seconds. It was re-proving a page-cached file the previous query had just proven.
///
/// This does not weaken the seal. Every pin still captures and verifies the full fingerprint set
/// before consulting the map, and a hit requires all of it to be identical — so a mutation the seal
/// could detect is a mutation that misses the memo and re-hashes. On Unix the entry and target
/// fingerprints carry `ctime`, which a content write always moves and which userspace cannot
/// backdate the way it can `mtime`.
static PROVEN_DIGESTS: OnceLock<Mutex<HashMap<SealIdentity, [u8; 32]>>> = OnceLock::new();

/// Bound on retained digests. Keyed per file identity, so the live set is the number of distinct
/// weight files a process touches — small for one model, and growing only across a long-lived worker
/// that loads many. Clearing wholesale at the bound costs a re-hash on the next acquisition of an
/// evicted file, which is exactly the pre-memo behaviour; an LRU would buy nothing at this size.
const PROVEN_DIGEST_CAPACITY: usize = 1024;

fn proven_digests() -> &'static Mutex<HashMap<SealIdentity, [u8; 32]>> {
    PROVEN_DIGESTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn proven_digest(identity: &SealIdentity) -> Option<[u8; 32]> {
    proven_digests()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(identity)
        .copied()
}

pub(super) fn remember_proven_digest(identity: SealIdentity, sha256: [u8; 32]) {
    let mut digests = proven_digests()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if digests.len() >= PROVEN_DIGEST_CAPACITY {
        digests.clear();
    }
    digests.insert(identity, sha256);
}

/// Consumer-chosen directory that carries [`PROVEN_DIGESTS`] across process restarts.
///
/// The in-process memo dies with the worker, and a consumer restarts its worker routinely. On a
/// USB model drive (sc-26610) one MOSS-Image job then spent 6.5 minutes re-hashing 25 GB of
/// weights that the previous process had already proven, before a 10-second load.
///
/// A persisted entry is keyed on a digest of the WHOLE [`SealIdentity`], the same predicate the
/// in-process memo trusts, so it never authorizes anything the in-memory memo would not: any
/// change to the path, parent chain, entry, resolution, or target (including `ctime`/inode on
/// Unix) yields a different key and a full re-read. Unset by default; gen-core never picks a
/// location on its own.
///
/// The key also binds the identity of the VOLUME holding the content (see [`volume_identity`]).
/// Device numbers are assigned per mount, and some filesystems (exFAT/FAT, some SMB) synthesize
/// inodes and `ctime`, so a different drive mounted at the same path could otherwise present a
/// same-looking file. With no volume identity available nothing is persisted.
static PROVEN_DIGEST_STORE: OnceLock<PathBuf> = OnceLock::new();

/// Version prefix for the persisted key. Bump it if [`SealIdentity`]'s fields change meaning;
/// older entries then simply miss.
const PROVEN_DIGEST_STORE_KEY_VERSION: &str = "artifact-seal-identity-v2";

pub(super) fn proven_digest_store_entry(
    store: &Path,
    identity: &SealIdentity,
    volume: &[u8],
) -> PathBuf {
    // `Debug` renders every identity field, escaping paths, so equal renderings mean equal
    // identities. A future std rendering change can only turn hits into misses.
    let key = Sha256::digest(format!(
        "{PROVEN_DIGEST_STORE_KEY_VERSION}\n{}\n{identity:?}",
        hex_lower(volume)
    ));
    store.join(hex_lower(&key))
}

fn stored_proven_digest(entry: &Path) -> Option<[u8; 32]> {
    let text = std::fs::read_to_string(entry).ok()?;
    parse_hex_digest(text.trim())
}

fn store_proven_digest(entry: &Path, sha256: &[u8; 32]) {
    if entry
        .parent()
        .is_none_or(|store| std::fs::create_dir_all(store).is_err())
    {
        return;
    }
    // Write-then-rename so a concurrent reader in another worker never sees a torn entry.
    let staging = entry.with_extension(format!("tmp-{}", std::process::id()));
    if std::fs::write(&staging, hex_lower(sha256)).is_err()
        || std::fs::rename(&staging, entry).is_err()
    {
        let _ = std::fs::remove_file(&staging);
    }
}

/// Stable identity of the volume holding `path`, or `None` when the platform cannot supply one.
///
/// macOS: the volume UUID of the mount containing `path`. Other Unix: the filesystem id from
/// `statvfs`, which ext4/btrfs/xfs derive from the superblock UUID; a zero id (tmpfs and other
/// synthetic filesystems) counts as unavailable. Windows: the volume serial number is already part
/// of every [`FileStatFingerprint`], so the identity itself carries the volume.
#[cfg(target_os = "macos")]
pub(super) fn volume_identity(path: &Path) -> Option<Vec<u8>> {
    use std::ffi::{CStr, CString};
    use std::mem::{size_of, MaybeUninit};
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut fs = MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `path` is NUL-terminated and `fs` is an output buffer of the exact libc type.
    if unsafe { libc::statfs(path.as_ptr(), fs.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: a successful `statfs` initialized the whole structure.
    let fs = unsafe { fs.assume_init() };
    // SAFETY: `f_mntonname` is a NUL-terminated C string inside `fs`.
    let mount = unsafe { CStr::from_ptr(fs.f_mntonname.as_ptr()) };

    // Volume attributes are only reported for the mount point itself.
    let mut request = libc::attrlist {
        bitmapcount: libc::ATTR_BIT_MAP_COUNT,
        reserved: 0,
        commonattr: 0,
        volattr: libc::ATTR_VOL_INFO | libc::ATTR_VOL_UUID,
        dirattr: 0,
        fileattr: 0,
        forkattr: 0,
    };
    #[repr(C, packed(4))]
    struct VolumeUuidReply {
        length: u32,
        uuid: [u8; 16],
    }
    let mut reply = VolumeUuidReply {
        length: 0,
        uuid: [0; 16],
    };
    // SAFETY: valid C path, request, and a reply buffer whose size is passed alongside it.
    let status = unsafe {
        libc::getattrlist(
            mount.as_ptr(),
            (&mut request as *mut libc::attrlist).cast(),
            (&mut reply as *mut VolumeUuidReply).cast(),
            size_of::<VolumeUuidReply>(),
            0,
        )
    };
    let uuid = reply.uuid;
    (status == 0 && uuid != [0; 16]).then(|| uuid.to_vec())
}

#[cfg(all(unix, not(target_os = "macos")))]
pub(super) fn volume_identity(path: &Path) -> Option<Vec<u8>> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut fs = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is NUL-terminated and `fs` is an output buffer of the exact libc type.
    if unsafe { libc::statvfs(path.as_ptr(), fs.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: a successful `statvfs` initialized the whole structure.
    let fsid = unsafe { fs.assume_init() }.f_fsid;
    // `f_fsid` is `c_ulong`: `u64` on 64-bit targets, narrower elsewhere.
    #[allow(clippy::useless_conversion)]
    let fsid = u64::from(fsid);
    (fsid != 0).then(|| fsid.to_le_bytes().to_vec())
}

#[cfg(windows)]
pub(super) fn volume_identity(_path: &Path) -> Option<Vec<u8>> {
    Some(Vec::new())
}

#[cfg(not(any(unix, windows)))]
pub(super) fn volume_identity(_path: &Path) -> Option<Vec<u8>> {
    None
}

/// Drop this process's in-memory digests for `path`, as a worker restart would.
#[cfg(test)]
pub(super) fn forget_proven_digests_for(path: &Path) {
    proven_digests()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|identity, _| identity.loader_path != path);
}

/// Per-key mutual exclusion for expensive one-time work such as a full-content hash.
///
/// Callers that memoize a result per key claim the key, re-check their memo, and only then do the
/// work, so concurrent first uses of one key do the work once instead of once per caller. The
/// [`SingleFlightGuard`] releases the key on every exit path, including unwinding, so a caller
/// that fails or panics while holding it never strands the callers waiting on it.
pub struct SingleFlight<K> {
    busy: Mutex<HashSet<K>>,
    released: Condvar,
}

/// Holds one key of a [`SingleFlight`] until dropped.
pub struct SingleFlightGuard<'a, K: Eq + std::hash::Hash> {
    flight: &'a SingleFlight<K>,
    key: Option<K>,
}

/// Longest a waiter sleeps before re-running its cancellation check.
const SINGLE_FLIGHT_WAIT_SLICE: Duration = Duration::from_millis(50);

impl<K: Eq + std::hash::Hash + Clone> Default for SingleFlight<K> {
    fn default() -> Self {
        Self {
            busy: Mutex::new(HashSet::new()),
            released: Condvar::new(),
        }
    }
}

impl<K: Eq + std::hash::Hash + Clone> SingleFlight<K> {
    /// Block until no other caller holds `key`, then hold it until the returned guard drops.
    ///
    /// `check` runs between bounded waits, so a cancelled caller stops waiting with its error.
    pub fn claim(
        &self,
        key: K,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<SingleFlightGuard<'_, K>> {
        loop {
            {
                let mut busy = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
                if busy.insert(key.clone()) {
                    return Ok(SingleFlightGuard {
                        flight: self,
                        key: Some(key),
                    });
                }
                let _ = self
                    .released
                    .wait_timeout(busy, SINGLE_FLIGHT_WAIT_SLICE)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            check()?;
        }
    }
}

impl<K: Eq + std::hash::Hash> Drop for SingleFlightGuard<'_, K> {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            self.flight
                .busy
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&key);
        }
        self.flight.released.notify_all();
    }
}

/// Loader paths that reached the seal's single-flight claim, so a test can tell a waiter is parked.
#[cfg(test)]
pub(super) fn seal_claim_attempts() -> &'static Mutex<Vec<PathBuf>> {
    static ATTEMPTS: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    ATTEMPTS.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(test)]
fn record_seal_claim_attempt(path: &Path) {
    seal_claim_attempts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(path.to_path_buf());
}

/// Seal identities whose cold content read is in progress; see [`SingleFlight`].
pub(super) fn seal_flights() -> &'static SingleFlight<SealIdentity> {
    static FLIGHTS: OnceLock<SingleFlight<SealIdentity>> = OnceLock::new();
    FLIGHTS.get_or_init(SingleFlight::default)
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_hex_digest(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(digest)
}

/// A SHA-256 the caller already proved for one file, bound to that file's resolved-target identity
/// at the moment of proof (sc-26691).
///
/// The motivating producer is a derivative written by a hashing writer: every byte went through
/// SHA-256 on its way to disk, so re-reading the finished file to seal it is pure cost (a 34 GB
/// MiniMax-H3 derivative re-read cold before every first load). [`ArtifactSeal::pin_attested`]
/// redeems the digest only while a fresh capture of the file's target fingerprint is IDENTICAL to
/// the one bound here — size, mtime, file id/inode, volume/device, and Windows ChangeTime / Unix
/// ctime, which a content write always moves and userspace cannot backdate. Any difference is a full
/// read, exactly as [`ArtifactSeal::pin`] would perform. The trust is the same predicate the
/// in-process digest memo already applies; the attestation only lets the hashing party seed it.
///
/// Capture it only after the bytes are final (after the hashing writer closed), never before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactDigestAttestation {
    target_fingerprint: FileStatFingerprint,
    sha256: [u8; 32],
}

impl ArtifactDigestAttestation {
    /// Bind `sha256` to `path`'s resolved-target identity as it is right now.
    pub fn capture(path: impl AsRef<Path>, sha256: [u8; 32]) -> crate::Result<Self> {
        let target_fingerprint = stat_fingerprint(path.as_ref(), true)?;
        if target_fingerprint.is_symlink || !std::fs::metadata(path.as_ref())?.is_file() {
            return Err(crate::Error::Msg(format!(
                "attested weights source is not a regular file: {}",
                path.as_ref().display()
            )));
        }
        Ok(Self {
            target_fingerprint,
            sha256,
        })
    }

    pub fn target_fingerprint(&self) -> &FileStatFingerprint {
        &self.target_fingerprint
    }

    pub fn content_sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}

/// `fingerprint` with the fields a new hard link legitimately moves (Windows ChangeTime, Unix ctime)
/// cleared. Every other field — size, mtime, file id/inode, volume/device — must still agree.
fn without_change_time(fingerprint: &FileStatFingerprint) -> FileStatFingerprint {
    #[allow(unused_mut)]
    let mut fingerprint = fingerprint.clone();
    #[cfg(unix)]
    {
        fingerprint.changed_seconds = 0;
        fingerprint.changed_nanoseconds = 0;
    }
    #[cfg(windows)]
    {
        fingerprint.change_time = 0;
    }
    fingerprint
}

#[cfg(unix)]
fn same_file_object(left: &FileStatFingerprint, right: &FileStatFingerprint) -> bool {
    left.device == right.device && left.inode == right.inode
}

#[cfg(windows)]
fn same_file_object(left: &FileStatFingerprint, right: &FileStatFingerprint) -> bool {
    left.volume_serial_number == right.volume_serial_number && left.file_id == right.file_id
}

#[cfg(not(any(unix, windows)))]
fn same_file_object(_left: &FileStatFingerprint, _right: &FileStatFingerprint) -> bool {
    false
}

impl ArtifactSeal {
    /// Persist proven seal digests under `dir` for the rest of this process (first call wins).
    ///
    /// Store failures are never errors: an unreadable or unwritable store only costs the full read
    /// that pinning performs without one.
    pub fn persist_proven_digests_in(dir: impl Into<PathBuf>) {
        let _ = PROVEN_DIGEST_STORE.set(dir.into());
    }

    pub fn pin(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::pin_with_check(path, || Ok(()))
    }

    /// Pin the same immutable source identity with cooperative cancellation/progress checks.
    /// Checks bracket bounded hash reads and metadata/memo paths; errors never authorize a pin.
    /// Digest reuse and once-at-acquisition hashing are identical to [`Self::pin`].
    pub fn pin_with_check(
        path: impl AsRef<Path>,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<Self> {
        Self::pin_with_progress(path, &mut check, |_| {})
    }

    /// Acquire the same seal with typed, truthful work observations. The observer cannot authorize
    /// a source or suppress cancellation. To cancel on progress, latch a token read by `check`.
    /// Reads report their first chunk, at most 256 MiB between updates, EOF and exact abort bytes.
    /// Memo acquisitions report zero hashed bytes; metadata checks never fake content progress.
    pub fn pin_with_progress(
        path: impl AsRef<Path>,
        mut check: impl FnMut() -> crate::Result<()>,
        mut observe: impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<Self> {
        let path = path.as_ref();
        let loader_path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let mut last = seal_progress(
            &loader_path,
            ArtifactSealAcquisition::Metadata,
            ArtifactSealProgressState::Reading,
            0,
            None,
        );
        let result = {
            let mut record = |progress: &ArtifactSealProgress| {
                last = progress.clone();
                observe(progress);
            };
            Self::pin_with_store_and_check(
                path,
                PROVEN_DIGEST_STORE.get().map(PathBuf::as_path),
                &mut check,
                &mut record,
            )
        };
        if let Err(error) = &result {
            let state = if matches!(error, crate::Error::Canceled) {
                ArtifactSealProgressState::Canceled
            } else {
                ArtifactSealProgressState::Failed
            };
            if last.state != state {
                last.state = state;
                observe(&last);
            }
        }
        result
    }

    #[cfg(test)]
    pub(super) fn pin_with_store(path: &Path, store: Option<&Path>) -> crate::Result<Self> {
        Self::pin_with_store_and_check(path, store, &mut || Ok(()), &mut |_| {})
    }

    /// Capture and self-verify every identity field of `path` without reading its content. The
    /// returned seal carries a zero digest and must never escape before one is assigned.
    pub(super) fn capture_identity(path: &Path) -> crate::Result<Self> {
        let loader_path = std::path::absolute(path)?;
        let canonical_target_path = std::fs::canonicalize(&loader_path)?;
        let path_component_fingerprints = path_component_fingerprints(&loader_path)?;
        let entry_fingerprint = stat_fingerprint(&loader_path, false)?;
        let target_fingerprint = stat_fingerprint(&loader_path, true)?;
        if target_fingerprint.is_symlink || !std::fs::metadata(&loader_path)?.is_file() {
            return Err(crate::Error::Msg(format!(
                "weights source is not a regular file: {}",
                loader_path.display()
            )));
        }
        let pinned = Self {
            loader_path,
            canonical_target_path,
            path_component_fingerprints,
            entry_fingerprint,
            target_fingerprint,
            sha256: [0; 32],
            #[cfg(test)]
            hash_work: Arc::new(SealHashWork(AtomicUsize::new(0))),
        };
        // Prove the entry, parent chain, resolution, and target still agree after the multi-stat
        // capture. This closes persistent changes during pin construction itself.
        pinned.ensure_unchanged()?;
        Ok(pinned)
    }

    pub(super) fn seal_identity(&self) -> SealIdentity {
        SealIdentity {
            loader_path: self.loader_path.clone(),
            canonical_target_path: self.canonical_target_path.clone(),
            path_component_fingerprints: self.path_component_fingerprints.clone(),
            entry_fingerprint: self.entry_fingerprint.clone(),
            target_fingerprint: self.target_fingerprint.clone(),
        }
    }

    /// Record a digest proven for this seal's exact identity in the process memo and, when one is
    /// configured, the persisted store — the same two places a cold read records it.
    fn remember_digest(&self, store: Option<&Path>, sha256: [u8; 32]) {
        let identity = self.seal_identity();
        if let Some(entry) = store.and_then(|store| {
            let volume = volume_identity(&self.canonical_target_path)?;
            Some(proven_digest_store_entry(store, &identity, &volume))
        }) {
            store_proven_digest(&entry, &sha256);
        }
        remember_proven_digest(identity, sha256);
    }

    /// [`Self::pin`] for a file whose digest the caller already proved (see
    /// [`ArtifactDigestAttestation`]). No content is read when the attested target identity still
    /// matches; otherwise this is exactly [`Self::pin`], memo and cold read included.
    pub fn pin_attested(
        path: impl AsRef<Path>,
        attestation: &ArtifactDigestAttestation,
    ) -> crate::Result<Self> {
        Self::pin_attested_with_progress(path, attestation, || Ok(()), |_| {})
    }

    /// Checked, observed form of [`Self::pin_attested`]. A redeemed attestation reports
    /// [`ArtifactSealAcquisition::Attested`] with zero hashed bytes.
    pub fn pin_attested_with_progress(
        path: impl AsRef<Path>,
        attestation: &ArtifactDigestAttestation,
        mut check: impl FnMut() -> crate::Result<()>,
        mut observe: impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<Self> {
        let path = path.as_ref();
        check()?;
        let pinned = Self::capture_identity(path)?;
        check()?;
        if pinned.target_fingerprint == attestation.target_fingerprint {
            pinned.remember_digest(
                PROVEN_DIGEST_STORE.get().map(PathBuf::as_path),
                attestation.sha256,
            );
            observe(&seal_progress(
                &pinned.loader_path,
                ArtifactSealAcquisition::Attested,
                ArtifactSealProgressState::Acquired,
                0,
                Some(pinned.target_fingerprint.size),
            ));
            return Ok(Self {
                sha256: attestation.sha256,
                ..pinned
            });
        }
        Self::pin_with_progress(path, check, observe)
    }

    /// Create a hard link to this seal's target at `link` and return the seal re-captured after it,
    /// carrying the SAME digest without re-reading content (sc-26691).
    ///
    /// Adding a hard link moves the shared file's Windows ChangeTime (Unix ctime), which every
    /// seal reads as drift; re-pinning the file afterwards was a full cold read of bytes the link
    /// never touched. Here the seal performs the link itself, so it can prove what changed: the
    /// seal must be unchanged immediately before the link, the re-capture must agree on every
    /// field except the change time (size, mtime, file id/inode, volume, resolution, parent chain),
    /// and `link` must name the very same file object. Anything else refuses — the link is left in
    /// place and the caller's ordinary re-pin re-reads and compares the content.
    ///
    /// # The window in which the change time is not compared
    ///
    /// The change time is ignored only between two stats that bracket the `hard_link` syscall:
    /// the last pre-link stat of the target and of the entry (each compared IN FULL, change time
    /// included, against this seal — after [`Self::ensure_unchanged`] has verified the parent
    /// chain and resolution) and the first post-link stat of the target and of the entry (compared
    /// against this seal with only the change time cleared). Everything captured after those
    /// post-link stats — the full re-capture and the link's own stat — must then agree with them
    /// IN FULL, so a write after the bracket is caught by its change time as usual. The only write
    /// this cannot distinguish from the link is one that lands inside that bracket (two stats,
    /// the link syscall, two stats) AND preserves size, mtime and file id.
    pub fn hard_link_with_check(
        &self,
        link: impl AsRef<Path>,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<Self> {
        let link = link.as_ref();
        let changed = || {
            crate::Error::Unsupported(format!(
                "pinned weights changed while being hard-linked: {}",
                self.loader_path.display()
            ))
        };
        check()?;
        self.ensure_unchanged()?;
        // Bracket open: the last full-identity proof before the link.
        if stat_fingerprint(&self.loader_path, true)? != self.target_fingerprint
            || stat_fingerprint(&self.loader_path, false)? != self.entry_fingerprint
        {
            return Err(changed());
        }
        std::fs::hard_link(&self.canonical_target_path, link)?;
        // Bracket close: the first stats after the link; only the change time may have moved.
        let target_after = stat_fingerprint(&self.loader_path, true)?;
        let entry_after = stat_fingerprint(&self.loader_path, false)?;
        if without_change_time(&target_after) != without_change_time(&self.target_fingerprint)
            || without_change_time(&entry_after) != without_change_time(&self.entry_fingerprint)
        {
            return Err(changed());
        }
        let relinked = Self::capture_identity(&self.loader_path)?;
        let linked = stat_fingerprint(link, false)?;
        // From here on the full identity, change time included, must equal the bracket-close one.
        let unchanged_but_for_the_link = relinked.loader_path == self.loader_path
            && relinked.canonical_target_path == self.canonical_target_path
            && relinked.path_component_fingerprints == self.path_component_fingerprints
            && relinked.entry_fingerprint == entry_after
            && relinked.target_fingerprint == target_after
            && !linked.is_symlink
            && same_file_object(&linked, &relinked.target_fingerprint)
            && linked == target_after;
        if !unchanged_but_for_the_link {
            return Err(changed());
        }
        check()?;
        relinked.remember_digest(PROVEN_DIGEST_STORE.get().map(PathBuf::as_path), self.sha256);
        Ok(Self {
            sha256: self.sha256,
            ..relinked
        })
    }

    pub(super) fn pin_with_store_and_check(
        path: &Path,
        store: Option<&Path>,
        check: &mut impl FnMut() -> crate::Result<()>,
        observe: &mut impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<Self> {
        let (pinned, identity, _flight) = loop {
            check()?;
            let pinned = Self::capture_identity(path)?;
            check()?;
            // Only AFTER `capture_identity`'s self-verification is the identity worth trusting as a
            // memo key: the lookup below is allowed to stand in for the content read precisely
            // because everything it compares has just been re-verified against the filesystem.
            let identity = pinned.seal_identity();
            #[cfg(test)]
            record_seal_claim_attempt(&pinned.loader_path);
            // Concurrent first pins of one identity read it once: later callers wait here and then
            // hit the memo the first caller filled.
            let flight = seal_flights().claim(identity.clone(), &mut *check)?;
            // A waiter may have blocked for a whole content read, so the identity it captured
            // before waiting must be proven again before the memo may stand in for its content:
            // a file replaced or rewritten meanwhile is re-captured (and re-read) instead.
            if pinned.ensure_unchanged().is_ok() {
                break (pinned, identity, flight);
            }
        };
        if let Some(sha256) = proven_digest(&identity) {
            check()?;
            observe(&seal_progress(
                &pinned.loader_path,
                ArtifactSealAcquisition::MemoryMemo,
                ArtifactSealProgressState::Acquired,
                0,
                Some(pinned.target_fingerprint.size),
            ));
            return Ok(Self { sha256, ..pinned });
        }
        let persisted = store.and_then(|store| {
            let volume = volume_identity(&pinned.canonical_target_path)?;
            Some(proven_digest_store_entry(store, &identity, &volume))
        });
        check()?;
        if let Some(sha256) = persisted.as_deref().and_then(stored_proven_digest) {
            check()?;
            remember_proven_digest(identity, sha256);
            observe(&seal_progress(
                &pinned.loader_path,
                ArtifactSealAcquisition::PersistedMemo,
                ArtifactSealProgressState::Acquired,
                0,
                Some(pinned.target_fingerprint.size),
            ));
            return Ok(Self { sha256, ..pinned });
        }
        #[cfg(test)]
        {
            pinned.hash_work.0.fetch_add(1, Ordering::Relaxed);
            record_full_hash_work();
        }
        let (sha256, hashed_bytes) =
            pinned.read_unchanged_with_check(check, observe, |path, check, observe| {
                hash_file_sha256_with_progress(
                    path,
                    check,
                    ArtifactSealAcquisition::ColdContentRead,
                    observe,
                )
            })?;
        check()?;
        if let Some(entry) = &persisted {
            store_proven_digest(entry, &sha256);
        }
        remember_proven_digest(identity, sha256);
        observe(&seal_progress(
            &pinned.loader_path,
            ArtifactSealAcquisition::ColdContentRead,
            ArtifactSealProgressState::Acquired,
            hashed_bytes,
            Some(pinned.target_fingerprint.size),
        ));
        Ok(Self { sha256, ..pinned })
    }

    pub fn loader_path(&self) -> &Path {
        &self.loader_path
    }

    /// Canonical target resolved by the same pinning operation that captured this token.
    ///
    /// A caller can root-confine this path before installing the token on a [`LoadSpec`], avoiding
    /// a second resolution that could authorize a different filesystem object.
    pub fn canonical_target_path(&self) -> &Path {
        &self.canonical_target_path
    }

    pub fn entry_fingerprint(&self) -> &FileStatFingerprint {
        &self.entry_fingerprint
    }

    pub fn target_fingerprint(&self) -> &FileStatFingerprint {
        &self.target_fingerprint
    }

    /// Full-content digest captured exactly once when this seal was acquired.
    pub fn content_sha256(&self) -> &[u8; 32] {
        &self.sha256
    }

    /// [`content_sha256`](Self::content_sha256) as 64 lowercase hex digits.
    pub fn content_sha256_hex(&self) -> String {
        hex_lower(&self.sha256)
    }

    /// Number of full-content reads performed by this sealed artifact in test builds.
    #[cfg(test)]
    pub fn full_hash_work_count(&self) -> usize {
        self.hash_work.0.load(Ordering::Relaxed)
    }

    fn hash_content(&self, path: &Path) -> crate::Result<[u8; 32]> {
        self.hash_content_with_check(path, &mut || Ok(()), &mut |_| {})
    }

    fn hash_content_with_check(
        &self,
        path: &Path,
        check: &mut impl FnMut() -> crate::Result<()>,
        observe: &mut impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<[u8; 32]> {
        #[cfg(test)]
        {
            self.hash_work.0.fetch_add(1, Ordering::Relaxed);
            record_full_hash_work();
        }
        hash_file_sha256_with_progress(path, check, ArtifactSealAcquisition::DriftAudit, observe)
            .map(|(digest, _)| digest)
    }

    pub fn ensure_unchanged(&self) -> crate::Result<()> {
        let path_components = path_component_fingerprints(&self.loader_path)?;
        if path_components != self.path_component_fingerprints {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights path component changed after load: {}",
                self.loader_path.display()
            )));
        }
        let entry = stat_fingerprint(&self.loader_path, false)?;
        if entry != self.entry_fingerprint {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights entry changed after load: {}",
                self.loader_path.display()
            )));
        }
        let canonical_target_path = std::fs::canonicalize(&self.loader_path)?;
        if canonical_target_path != self.canonical_target_path {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights resolution changed after load: {}",
                self.loader_path.display()
            )));
        }
        let target = stat_fingerprint(&self.loader_path, true)?;
        if target != self.target_fingerprint {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights target changed after load: {}",
                self.loader_path.display()
            )));
        }
        Ok(())
    }

    /// Verify the complete replacement-sensitive identity through caller-owned Darwin
    /// metadata buffers. This never hashes the payload on a deferred read, including a
    /// mismatch (which is unconditionally rejected). Native compatibility APIs retain
    /// their historical audit behavior. See source_inventory::bounded_fs for authority.
    pub fn verify_unchanged_with_capacity(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use crate::source_inventory::bounded_fs as fs;
            if bounded_path_components(&self.loader_path, next)? != self.path_component_fingerprints
                || bounded_stat_fingerprint(&self.loader_path, false, next)?
                    != self.entry_fingerprint
                || fs::canonical(&self.loader_path, next)? != self.canonical_target_path
                || bounded_stat_fingerprint(&self.loader_path, true, next)?
                    != self.target_fingerprint
            {
                return crate::SourceInventory::metadata_gap(
                    next,
                    "artifact seal mismatch after load",
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        crate::SourceInventory::metadata_gap(next, crate::SourceInventory::PIN_VERIFY_METADATA_GAP)
    }

    /// Fresh acquisition with the same lexical binding, component/entry/target identity
    /// and full-content SHA-256 as pin(). The bounded path intentionally does not insert
    /// into the process-global digest memo: that would allocate private HashMap buckets.
    /// Already prepared tokens are reused by SourceInventory instead of reacquired.
    pub fn pin_with_capacity(
        path: &Path,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            use crate::source_inventory::bounded_fs as fs;
            let loader_path = fs::absolute(path, next)?;
            let canonical_target_path = fs::canonical(&loader_path, next)?;
            let path_component_fingerprints = bounded_path_components(&loader_path, next)?;
            let entry_fingerprint = bounded_stat_fingerprint(&loader_path, false, next)?;
            let target_fingerprint = bounded_stat_fingerprint(&loader_path, true, next)?;
            if !fs::is_file(&fs::stat(&loader_path, true, next)?) {
                return crate::SourceInventory::metadata_gap(
                    next,
                    "weights source is not a regular file",
                );
            }
            #[cfg(test)]
            next(bounded_arc_layout::<SealHashWork>()?)?;
            let mut pinned = Self {
                loader_path,
                canonical_target_path,
                path_component_fingerprints,
                entry_fingerprint,
                target_fingerprint,
                sha256: [0; 32],
                #[cfg(test)]
                hash_work: Arc::new(SealHashWork(AtomicUsize::new(0))),
            };
            pinned.verify_unchanged_with_capacity(next)?;
            #[cfg(test)]
            {
                pinned.hash_work.0.fetch_add(1, Ordering::Relaxed);
                record_full_hash_work();
            }
            pinned.sha256 = fs::hash(&pinned.loader_path, next)?;
            pinned.verify_unchanged_with_capacity(next)?;
            Ok(pinned)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = path;
            crate::SourceInventory::metadata_gap(
                next,
                crate::SourceInventory::PIN_CAPTURE_METADATA_GAP,
            )
        }
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn clone_with_capacity(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<Self> {
        use crate::source_inventory::bounded_fs as fs;
        let mut components = fs::vec(self.path_component_fingerprints.len(), next)?;
        for component in &self.path_component_fingerprints {
            components.push(PathComponentFingerprint {
                path: fs::copy_path(&component.path, next)?,
                is_symlink: component.is_symlink,
                symlink_target: component
                    .symlink_target
                    .as_deref()
                    .map(|p| fs::copy_path(p, next))
                    .transpose()?,
                device: component.device,
                inode: component.inode,
            });
        }
        Ok(Self {
            loader_path: fs::copy_path(&self.loader_path, next)?,
            canonical_target_path: fs::copy_path(&self.canonical_target_path, next)?,
            path_component_fingerprints: components,
            entry_fingerprint: bounded_clone_stat(&self.entry_fingerprint, next)?,
            target_fingerprint: bounded_clone_stat(&self.target_fingerprint, next)?,
            sha256: self.sha256,
            #[cfg(test)]
            hash_work: self.hash_work.clone(),
        })
    }

    /// Same explicit check before and after a pinned read, including failed reads. The existing
    /// native read_unchanged/verify_unchanged APIs remain unchanged for compatibility.
    pub fn read_unchanged_with_capacity<T, E: From<crate::Error>>(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
        read: impl FnOnce(&Path, &mut dyn FnMut(u64) -> crate::Result<()>) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.verify_unchanged_with_capacity(next).map_err(E::from)?;
        let result = read(&self.loader_path, next);
        self.verify_unchanged_with_capacity(next).map_err(E::from)?;
        result
    }

    pub fn verify_unchanged(&self) -> crate::Result<()> {
        self.verify_unchanged_with_check(&mut || Ok(()), &mut |_| {})
    }

    fn verify_unchanged_with_check(
        &self,
        check: &mut impl FnMut() -> crate::Result<()>,
        observe: &mut impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<()> {
        check()?;
        let unchanged = self.ensure_unchanged().is_ok();
        check()?;
        if unchanged {
            return Ok(());
        }

        // Identity drift can never re-authorize this seal, even if a full audit agrees.
        // Preserve the unchecked API's mismatch authority for I/O errors, but do not swallow
        // cooperative callback errors or silently turn its audit into an unchecked full read.
        let mut interrupted = false;
        let audit = self.hash_content_with_check(
            &self.loader_path,
            &mut || check().inspect_err(|_| interrupted = true),
            observe,
        );
        if interrupted {
            return audit.map(|_| ());
        }
        check()?;
        Err(crate::Error::Unsupported(format!(
            "artifact seal mismatch after load: {}",
            self.loader_path.display()
        )))
    }

    pub(super) fn read_unchanged_with_check<T, C, O>(
        &self,
        check: &mut C,
        observe: &mut O,
        read: impl FnOnce(&Path, &mut C, &mut O) -> crate::Result<T>,
    ) -> crate::Result<T>
    where
        C: FnMut() -> crate::Result<()>,
        O: FnMut(&ArtifactSealProgress),
    {
        self.verify_unchanged_with_check(check, observe)?;
        let result = read(&self.loader_path, check, observe);
        // A canceled acquisition delivers no source bytes or seal. Do not start a post-read
        // drift audit (or invoke a one-shot callback again) after that terminal cancellation.
        // The public unchecked read_unchanged retains its stronger post-read mutation priority.
        if matches!(result, Err(crate::Error::Canceled)) {
            return result;
        }
        self.verify_unchanged_with_check(check, observe)?;
        result
    }

    /// Explicit, operator-requested full-content audit of an otherwise stable seal.
    pub fn audit_content(&self) -> crate::Result<()> {
        self.ensure_unchanged()?;
        if self.hash_content(&self.loader_path)? != self.sha256 {
            return Err(crate::Error::Unsupported(format!(
                "artifact seal content mismatch after load: {}",
                self.loader_path.display()
            )));
        }
        Ok(())
    }

    /// Run one source read against the retained loader path, validating the pin both immediately
    /// before the path is opened and immediately after the read completes.
    ///
    /// The second check detects replacement that remains visible in the lexical entry, parent
    /// components, resolution, or target identity after the callback. Callers should keep this same
    /// pin for every lazy or sequential reopen. This is a path-token guard, not an opened-handle
    /// lease: an active actor that swaps in B and restores the *original* A pathname object wholly
    /// between the two checks is outside this guarantee.
    ///
    /// The post-read check runs even when `read` returns an error. If both the read and the pin fail,
    /// the mutation error wins because it is the stronger source-identity diagnosis.
    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        self.verify_unchanged().map_err(E::from)?;
        let result = read(&self.loader_path);
        self.verify_unchanged().map_err(E::from)?;
        result
    }
}

#[cfg(feature = "model-inference-proof")]
pub(crate) fn record_sealed_input_read<'a>(
    reader: &str,
    pins: impl IntoIterator<Item = &'a ArtifactSeal>,
) {
    record_sealed_input_read_with_role(
        reader,
        crate::model_input_proof::InputRole::ArtifactInput,
        pins,
    );
}

#[cfg(feature = "model-inference-proof")]
pub(crate) fn record_sealed_input_read_with_role<'a>(
    reader: &str,
    role: crate::model_input_proof::InputRole,
    pins: impl IntoIterator<Item = &'a ArtifactSeal>,
) {
    if std::env::var_os("MODEL_INFERENCE_CASE_FQN").is_none() {
        return;
    }
    let identities: Vec<_> = pins
        .into_iter()
        .map(|pin| {
            crate::model_input_proof::InputIdentity::validated(
                pin.loader_path(),
                *pin.content_sha256(),
                "gen-core::ArtifactSeal",
                "existing retained seal, actual read and pre/post path checks; not an opened-handle or ABA guarantee",
            )
        })
        .collect();
    crate::model_input_proof::record_inputs(reader, role, &identities);
}

/// Read size for full-content hashing. 64 KiB reads held a USB model drive to ~63 MB/s against
/// ~216 MB/s for 4 MiB reads of the same disk (sc-26610); the sequential hash is I/O-bound there.
pub(super) const HASH_READ_BYTES: usize = 8 * 1024 * 1024;

#[cfg(test)]
pub(super) fn hash_file_sha256(path: &Path) -> crate::Result<[u8; 32]> {
    hash_file_sha256_with_check(path, &mut || Ok(()))
}

#[cfg(test)]
pub(super) fn hash_file_sha256_with_check(
    path: &Path,
    check: &mut impl FnMut() -> crate::Result<()>,
) -> crate::Result<[u8; 32]> {
    hash_file_sha256_with_progress(
        path,
        check,
        ArtifactSealAcquisition::ColdContentRead,
        &mut |_| {},
    )
    .map(|(digest, _)| digest)
}

fn hash_file_sha256_with_progress(
    path: &Path,
    check: &mut impl FnMut() -> crate::Result<()>,
    acquisition: ArtifactSealAcquisition,
    observe: &mut impl FnMut(&ArtifactSealProgress),
) -> crate::Result<([u8; 32], u64)> {
    let mut checked = |hashed_bytes, source_bytes| {
        check().inspect_err(|error| {
            let state = if matches!(error, crate::Error::Canceled) {
                ArtifactSealProgressState::Canceled
            } else {
                ArtifactSealProgressState::Failed
            };
            observe(&seal_progress(
                path,
                acquisition,
                state,
                hashed_bytes,
                source_bytes,
            ));
        })
    };
    checked(0, None)?;
    let mut reader = File::open(path)?;
    let source_bytes = Some(reader.metadata()?.len());
    checked(0, source_bytes)?;
    let _activity = crate::LoadActivityScope::enter(crate::LoadActivity::VerifyingFiles);
    // Release the callback borrow so progress is reported BEFORE the post-read cancellation check.
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; HASH_READ_BYTES];
    let mut hashed_bytes = 0u64;
    let mut reported = 0u64;
    loop {
        if let Err(error) = check() {
            let state = if matches!(error, crate::Error::Canceled) {
                ArtifactSealProgressState::Canceled
            } else {
                ArtifactSealProgressState::Failed
            };
            observe(&seal_progress(
                path,
                acquisition,
                state,
                hashed_bytes,
                source_bytes,
            ));
            return Err(error);
        }
        let read = match reader.read(&mut buffer) {
            Ok(read) => read,
            Err(error) => {
                observe(&seal_progress(
                    path,
                    acquisition,
                    ArtifactSealProgressState::Failed,
                    hashed_bytes,
                    source_bytes,
                ));
                return Err(error.into());
            }
        };
        hash.update(&buffer[..read]);
        hashed_bytes += read as u64;
        #[cfg(test)]
        THREAD_HASH_BYTES_READ.with(|bytes| bytes.set(bytes.get() + read));
        if read == 0 || reported == 0 || hashed_bytes - reported >= 256 * 1024 * 1024 {
            let state = if read == 0 {
                ArtifactSealProgressState::ReadComplete
            } else {
                ArtifactSealProgressState::Reading
            };
            observe(&seal_progress(
                path,
                acquisition,
                state,
                hashed_bytes,
                source_bytes,
            ));
            reported = hashed_bytes;
        }
        if let Err(error) = check() {
            let state = if matches!(error, crate::Error::Canceled) {
                ArtifactSealProgressState::Canceled
            } else {
                ArtifactSealProgressState::Failed
            };
            observe(&seal_progress(
                path,
                acquisition,
                state,
                hashed_bytes,
                source_bytes,
            ));
            return Err(error);
        }
        if read == 0 {
            break;
        }
    }
    Ok((hash.finalize().into(), hashed_bytes))
}

/// Test support shared by this module's tests and `registry`'s prepared-file callback tests: the
/// typed artifact-seal rejection every load / footprint / memory-contract callback boundary must
/// surface when the sealed source is rebound underneath it.
///
/// One definition, `pub(crate)`, on purpose (SC-22667 review, blocker): a second private copy inside
/// `registry`'s `mod tests` was deleted as dead code on Windows — where its only caller is
/// `#[cfg(unix)]` — and that deletion was a hard `E0425` on every Unix lane. A single crate-visible
/// helper has a caller on every host, so no lane can see it as dead and no lane can lose it.
#[cfg(test)]
pub(crate) fn assert_artifact_seal_mismatch(error: crate::Error) {
    match error {
        crate::Error::Unsupported(reason)
            if reason.starts_with("artifact seal mismatch after load: ") => {}
        crate::Error::Unsupported(reason) => {
            panic!("expected the shared artifact-seal rejection, got: {reason}")
        }
        other => panic!("expected a typed artifact-seal rejection, got: {other:?}"),
    }
}
