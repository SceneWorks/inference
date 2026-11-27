//! Produced-file identity: how a host trusts bytes it wrote itself without ever hashing them.
//!
//! A file the host produced (a normalized checkpoint, a saved quantization, a locally converted
//! tier) has exactly one author. Re-reading it to prove a SHA-256 the host would compute from its
//! own output protects nothing; what a later process must still detect is that the file CHANGED
//! after its producer finished it. [`ProducedFileIdentity`] is the durable record of that: the
//! producer captures it once, after the bytes are final, and stores it beside the file (a cache
//! journal, a conversion receipt). A later process compares a fresh capture instead of reading
//! content, and treats any difference as a stale file to rebuild, never as something to accept.
//!
//! # What makes the comparison sound
//!
//! The record binds size, modification time, the file object (Windows volume serial number + file
//! id, Unix device + inode) and the change time (Windows ChangeTime, Unix ctime). Every content
//! write moves the change time, and no ordinary API lets userspace restore it, so a same-size
//! rewrite that puts the old modification time back still differs.
//!
//! Two kinds of record are never trusted, so callers always rebuild (or fall back to a digest
//! they hold) instead:
//!
//! * **Weak**: on Windows, a zero change time (FAT/exFAT and some network volumes report none).
//! * **Racy** (git's racy-index rule): the file's modification time is not comfortably older than
//!   the moment the record was taken, so a rewrite inside the same timestamp tick could leave every
//!   field unchanged. [`ProducedFileIdentity::seal_written`] avoids this by setting the file's
//!   modification time [`PRODUCED_FILE_BACKDATE`] in the past before capturing; any later write then stamps the
//!   file with the current time, which can never reproduce the recorded one.

use super::*;

use serde::{Deserialize, Serialize};
use std::time::UNIX_EPOCH;

/// How far into the past [`ProducedFileIdentity::seal_written`] sets the finished file's
/// modification time. Comfortably coarser than every timestamp granularity (FAT's 2 s included).
pub const PRODUCED_FILE_BACKDATE: Duration = Duration::from_secs(10);

/// A record younger than this (modification time to observation) is racy and never trusted.
const RACY_WINDOW_NANOS: i128 = 2_000_000_000;

/// The durable identity of one file its host produced. See the [module docs](self).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProducedFileIdentity {
    /// Bytes in the file.
    pub size_bytes: u64,
    /// Modification time, nanoseconds since the Unix epoch.
    pub modified_nanos: Option<i128>,
    /// The file object: `win:<volume serial>:<file id>` or `unix:<device>:<inode>`.
    pub object: String,
    /// Windows ChangeTime (100 ns ticks) or Unix ctime (nanoseconds since the epoch).
    pub changed: i128,
    /// When the record was taken, nanoseconds since the Unix epoch. Not part of the identity: it
    /// dates the observation, which the racy rule measures the modification time against.
    pub observed_nanos: i128,
}

fn epoch_nanos(time: SystemTime) -> Option<i128> {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => Some(after.as_nanos() as i128),
        Err(before) => Some(-(before.duration().as_nanos() as i128)),
    }
}

impl ProducedFileIdentity {
    fn from_fingerprint(fingerprint: &FileStatFingerprint, observed: SystemTime) -> Self {
        #[cfg(windows)]
        let (object, changed) = (
            format!(
                "win:{:016x}:{}",
                fingerprint.volume_serial_number,
                fingerprint
                    .file_id
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
            i128::from(fingerprint.change_time),
        );
        #[cfg(unix)]
        let (object, changed) = (
            format!("unix:{}:{}", fingerprint.device, fingerprint.inode),
            i128::from(fingerprint.changed_seconds) * 1_000_000_000
                + i128::from(fingerprint.changed_nanoseconds),
        );
        #[cfg(not(any(unix, windows)))]
        let (object, changed) = (
            String::from("unsupported"),
            fingerprint.created.and_then(epoch_nanos).unwrap_or(0),
        );
        Self {
            size_bytes: fingerprint.size,
            modified_nanos: fingerprint.modified.and_then(epoch_nanos),
            object,
            changed,
            observed_nanos: epoch_nanos(observed).unwrap_or(0),
        }
    }

    fn fingerprint_of(path: &Path) -> crate::Result<FileStatFingerprint> {
        let fingerprint = stat_fingerprint(path, true)?;
        if !std::fs::metadata(path)?.is_file() {
            return Err(crate::Error::Msg(format!(
                "produced file is not a regular file: {}",
                path.display()
            )));
        }
        Ok(fingerprint)
    }

    /// Record `path` as it is right now. Only for bytes that are already final.
    pub fn capture(path: impl AsRef<Path>) -> crate::Result<Self> {
        let observed = SystemTime::now();
        let fingerprint = Self::fingerprint_of(path.as_ref())?;
        Ok(Self::from_fingerprint(&fingerprint, observed))
    }

    /// Finish a file the caller has just written and closed: set its modification time
    /// [`PRODUCED_FILE_BACKDATE`] before now (so the record is not racy), then record it. This is the one call a
    /// producer makes per output file.
    pub fn seal_written(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref();
        let backdated = SystemTime::now()
            .checked_sub(PRODUCED_FILE_BACKDATE)
            .unwrap_or(UNIX_EPOCH);
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .set_modified(backdated)?;
        Self::capture(path)
    }

    /// Whether this record cannot detect a same-size rewrite that restores the modification time,
    /// so it must never stand in for content: a Windows volume that reports no change time.
    pub fn is_weak(&self) -> bool {
        cfg!(windows) && self.changed == 0
    }

    /// Git's racy-index rule: trusted only when the file was last modified comfortably before the
    /// record was taken. A missing modification time is racy.
    pub fn is_racy(&self) -> bool {
        self.modified_nanos
            .is_none_or(|modified| modified + RACY_WINDOW_NANOS > self.observed_nanos)
    }

    /// Whether the record can vouch for the file at all ([`Self::is_weak`] / [`Self::is_racy`]).
    pub fn is_trustworthy(&self) -> bool {
        !self.is_weak() && !self.is_racy()
    }

    fn identifies(&self, current: &Self) -> bool {
        self.size_bytes == current.size_bytes
            && self.modified_nanos == current.modified_nanos
            && self.object == current.object
            && self.changed == current.changed
    }

    /// Whether `path` is still exactly the file this record describes, and the record is
    /// trustworthy. `false` means stale: rebuild it (or verify it against a digest you hold).
    pub fn matches(&self, path: impl AsRef<Path>) -> bool {
        self.attest(path).is_some()
    }

    /// The seal attestation for `path`, from ONE fresh capture that this record identifies, so no
    /// change can slip between the comparison and the seal. `None` when stale or untrustworthy.
    pub fn attest(&self, path: impl AsRef<Path>) -> Option<ArtifactIdentityAttestation> {
        if !self.is_trustworthy() {
            return None;
        }
        let fingerprint = Self::fingerprint_of(path.as_ref()).ok()?;
        let current = Self::from_fingerprint(&fingerprint, SystemTime::now());
        self.identifies(&current)
            .then(|| ArtifactIdentityAttestation::from_fingerprint(fingerprint))
    }
}

impl ArtifactSeal {
    /// Seal a file its host produced from the durable [`ProducedFileIdentity`] recorded for it:
    /// no content is read and the seal carries no digest. A file that no longer matches its record
    /// is refused, typed, rather than hashed: with no digest on record there is nothing a hash
    /// could be compared against, and a changed produced file is stale, so the caller rebuilds it.
    pub fn pin_produced_with_progress(
        path: impl AsRef<Path>,
        record: &ProducedFileIdentity,
        check: impl FnMut() -> crate::Result<()>,
        observe: impl FnMut(&ArtifactSealProgress),
    ) -> crate::Result<Self> {
        let path = path.as_ref();
        let Some(attestation) = record.attest(path) else {
            return Err(crate::Error::Unsupported(format!(
                "produced file changed since its producer recorded it: {}",
                path.display()
            )));
        };
        let seal = Self::pin_identity_attested_with_progress(path, &attestation, check, observe)?;
        if seal.is_identity_attested() {
            Ok(seal)
        } else {
            // The file moved between the record check and the seal's own capture.
            Err(crate::Error::Unsupported(format!(
                "produced file changed while it was being sealed: {}",
                path.display()
            )))
        }
    }

    /// [`Self::pin_produced_with_progress`] without cancellation or progress.
    pub fn pin_produced(
        path: impl AsRef<Path>,
        record: &ProducedFileIdentity,
    ) -> crate::Result<Self> {
        Self::pin_produced_with_progress(path, record, || Ok(()), |_| {})
    }
}
