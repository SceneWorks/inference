//! Shared checkpoint cursor and no-clobber atomic publication. Tensor serialization
//! remains backend-owned; this transport never initializes a device or reads user images.
use gen_core::{CancelFlag, Error, Result};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 checkpoint: {message}"))
}

/// Limits resolved from the already validated run configuration/dataset.
#[derive(Clone, Copy, Debug)]
pub struct ResumePolicy {
    pub total_micro_steps: u64,
    pub accumulation: u32,
    /// Prebuilt full MICRO-BATCHES per epoch after drop-last-per-bucket, not image count.
    pub sampler_len: u64,
}
impl ResumePolicy {
    pub fn fingerprint(self) -> String {
        format!(
            "micro_steps={};accumulation={};sampler_len={}",
            self.total_micro_steps, self.accumulation, self.sampler_len
        )
    }
}

/// Every cursor needed to continue the *same* native trajectory. RNG encoding is
/// named/versioned in ResumeIdentity.configuration; an empty state is never a reset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingCursor {
    pub micro_step: u64,
    pub update_step: u64,
    /// Pending micro-batch gradients, already divided by configured N before backward.
    pub accumulation_count: u32,
    pub sampler_epoch: u64,
    /// Consumed batches in this epoch; may equal sampler_len just before epoch wrap.
    pub sampler_cursor: u64,
    pub rng_state: Vec<u8>,
    /// Last actually observed raw micro-batch loss, for truthful completed-run resume/no-op.
    /// Optional for low-level codec fixtures/older snapshots; never fabricate a zero metric.
    pub last_loss_bits: Option<u32>,
}

impl TrainingCursor {
    pub fn last_loss(&self) -> Result<Option<f32>> {
        self.last_loss_bits
            .map(|bits| {
                let value = f32::from_bits(bits);
                if !value.is_finite() || value < 0.0 {
                    Err(invalid("saved last loss must be finite and nonnegative"))
                } else {
                    Ok(value)
                }
            })
            .transpose()
    }

    pub fn validate(
        &self,
        total_micro_steps: u64,
        accumulation: u32,
        sampler_len: u64,
    ) -> Result<()> {
        self.last_loss()?;
        if accumulation == 0
            || sampler_len == 0
            || self.micro_step > total_micro_steps
            || self.sampler_cursor > sampler_len
            || self.rng_state.is_empty()
            || self.rng_state.len() > 4096
        {
            return Err(invalid(
                "invalid step/sampler/RNG state; cannot resume reproducibly",
            ));
        }
        let accumulation = accumulation as u64;
        let expected_micro = self
            .sampler_epoch
            .checked_mul(sampler_len)
            .and_then(|n| n.checked_add(self.sampler_cursor))
            .ok_or_else(|| invalid("epoch/batch cursor overflows"))?;
        if self.micro_step != expected_micro {
            return Err(invalid(
                "micro-step does not match saved epoch/batch cursor",
            ));
        }
        // Accelerator flushes and resets accumulation at EACH dataloader exhaustion.
        // A global micro_step/accumulation quotient would lose short epoch tails.
        let previous_updates = self
            .sampler_epoch
            .checked_mul(sampler_len.div_ceil(accumulation))
            .ok_or_else(|| invalid("epoch update count overflows"))?;
        let closed = previous_updates
            .checked_add(self.sampler_cursor / accumulation)
            .ok_or_else(|| invalid("update cursor overflows"))?;
        let regular = self.update_step == closed
            && self.accumulation_count as u64 == self.sampler_cursor % accumulation;
        let ready_full = self.sampler_cursor > 0
            && self.sampler_cursor.is_multiple_of(accumulation)
            && self.accumulation_count as u64 == accumulation
            && self.update_step.checked_add(1) == Some(closed);
        let flushed_updates = previous_updates
            .checked_add(self.sampler_cursor.div_ceil(accumulation))
            .ok_or_else(|| invalid("flushed update cursor overflows"))?;
        let flushed_tail = (self.sampler_cursor == sampler_len
            || self.micro_step == total_micro_steps)
            && self.accumulation_count == 0
            && self.update_step == flushed_updates;
        if !regular && !ready_full && !flushed_tail {
            return Err(invalid("micro-step/update/accumulation counters disagree; refusing to discard pending gradients"));
        }
        Ok(())
    }

    pub fn to_metadata(&self) -> Result<String> {
        self.last_loss()?;
        if self.rng_state.is_empty() || self.rng_state.len() > 4096 {
            return Err(invalid("missing/oversized RNG state"));
        }
        Ok(serde_json::json!({
            "schema": 1, "micro_step": self.micro_step, "update_step": self.update_step,
            "accumulation_count": self.accumulation_count, "sampler_epoch": self.sampler_epoch,
            "sampler_cursor": self.sampler_cursor, "rng_state": self.rng_state,
            "last_loss_bits": self.last_loss_bits,
        })
        .to_string())
    }

    pub fn from_metadata(text: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| invalid(format!("invalid cursor JSON: {e}")))?;
        const KEYS: [&str; 8] = [
            "last_loss_bits",
            "schema",
            "micro_step",
            "update_step",
            "accumulation_count",
            "sampler_epoch",
            "sampler_cursor",
            "rng_state",
        ];
        let object = value
            .as_object()
            .ok_or_else(|| invalid("cursor must be an object"))?;
        if !(7..=8).contains(&object.len())
            || object.keys().any(|k| !KEYS.contains(&k.as_str()))
            || value["schema"] != 1
        {
            return Err(invalid("unknown/incomplete cursor schema"));
        }
        let number = |key: &str| {
            value[key]
                .as_u64()
                .ok_or_else(|| invalid(format!("invalid cursor {key}")))
        };
        let cursor =
            Self {
                micro_step: number("micro_step")?,
                update_step: number("update_step")?,
                accumulation_count: u32::try_from(number("accumulation_count")?)
                    .map_err(|_| invalid("accumulation count overflow"))?,
                sampler_epoch: number("sampler_epoch")?,
                sampler_cursor: number("sampler_cursor")?,
                rng_state: serde_json::from_value(value["rng_state"].clone())
                    .map_err(|e| invalid(format!("invalid RNG state: {e}")))?,
                last_loss_bits: match value.get("last_loss_bits") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(value) => Some(
                        u32::try_from(value.as_u64().ok_or_else(|| {
                            invalid("last loss bits must be an unsigned integer")
                        })?)
                        .map_err(|_| invalid("last loss bits overflow"))?,
                    ),
                },
            };
        cursor.last_loss()?;
        if cursor.rng_state.is_empty() || cursor.rng_state.len() > 4096 {
            return Err(invalid("missing/oversized RNG state"));
        }
        Ok(cursor)
    }
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Publication {
    temporary: PathBuf,
    final_path: PathBuf,
    owns_final: bool,
    committed: bool,
}
impl Drop for Publication {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.temporary);
        if self.owns_final && !self.committed {
            let _ = std::fs::remove_file(&self.final_path);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    BeforePublish,
    AfterPublish,
}

/// Write through an exclusively created file handle, sync it, then atomically
/// publish via a same-filesystem hard link. Unlike rename, this never replaces an
/// existing checkpoint on Unix. A link-capable local filesystem is required.
/// Failed/canceled attempts remove ONLY files they exclusively created.
pub fn publish_bytes(
    root: &Path,
    name: &str,
    bytes: &[u8],
    cancel: &CancelFlag,
) -> Result<PathBuf> {
    publish_with_hook(root, name, bytes, cancel, |_| {})
}

fn publish_with_hook(
    root: &Path,
    name: &str,
    bytes: &[u8],
    cancel: &CancelFlag,
    mut hook: impl FnMut(Stage),
) -> Result<PathBuf> {
    let check_cancel = || {
        if cancel.is_cancelled() {
            Err(Error::Canceled)
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    let mut components = Path::new(name).components();
    if !root.is_absolute()
        || name.is_empty()
        || name.contains(['/', '\\', ':', '\0'])
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(invalid(
            "output requires an absolute owned directory and a single file name",
        ));
    }
    let io = |action: &str, error: std::io::Error| {
        Error::Msg(format!("flux3 checkpoint: {action}: {error}"))
    };
    std::fs::create_dir_all(root).map_err(|e| io("create output directory", e))?;
    let root = root
        .canonicalize()
        .map_err(|e| io("resolve output directory", e))?;
    let final_path = root.join(name);
    let mut opened = None;
    for _ in 0..16 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = root.join(format!(".{name}.{}.{sequence}.tmp", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => {
                opened = Some((temporary, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io("create exclusive temporary checkpoint", error)),
        }
    }
    let (temporary, mut file) =
        opened.ok_or_else(|| invalid("could not allocate an exclusive temporary checkpoint"))?;
    let mut publication = Publication {
        temporary,
        final_path,
        owns_final: false,
        committed: false,
    };
    file.write_all(bytes)
        .map_err(|e| io("write checkpoint", e))?;
    file.sync_all().map_err(|e| io("sync checkpoint", e))?;
    drop(file);
    hook(Stage::BeforePublish);
    check_cancel()?;
    std::fs::hard_link(&publication.temporary, &publication.final_path).map_err(|e| {
        io(
            "publish checkpoint without replacing an existing file (requires hard links)",
            e,
        )
    })?;
    publication.owns_final = true;
    hook(Stage::AfterPublish);
    check_cancel()?;
    #[cfg(unix)]
    std::fs::File::open(&root)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| io("sync checkpoint directory", e))?;
    check_cancel()?;
    publication.committed = true;
    Ok(publication.final_path.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn late_cancel_at_both_publication_boundaries_withholds_output() {
        for boundary in [Stage::BeforePublish, Stage::AfterPublish] {
            let root = tempfile::tempdir().unwrap();
            let cancel = CancelFlag::new();
            let result = publish_with_hook(
                root.path(),
                "checkpoint.safetensors",
                b"synthetic",
                &cancel,
                |stage| {
                    if stage == boundary {
                        cancel.cancel();
                    }
                },
            );
            assert!(matches!(result, Err(Error::Canceled)));
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        }
    }
}
