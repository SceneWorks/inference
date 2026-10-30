//! **Checkpoint lineage** of the Iris trainers (story sc-25690): which checkpoints under a run's
//! checkpoint root belong to the run, which ranks publish into a root, which checkpoint every rank
//! resumes from, and the atomic publication of a multi-file export directory.
//!
//! * **Lineage.** A run's retention ([`super::prune_checkpoints`]) and its `latest` resolution
//!   ([`super::latest_checkpoint`]) see every `step_*` directory under the root, so a root holding
//!   another run's checkpoints would keep that run's high steps, delete the new run's, and resume
//!   the old run. [`retire_foreign_checkpoints`] moves every checkpoint that is not this run's
//!   lineage aside (into the one sibling [`prior_checkpoint_root`], `<root>.prior`, which the next
//!   retirement replaces: bounded disk, part of the training's footprint) before the run
//!   writes its first one.
//! * **Writers.** `checkpoint_io: per_node` keeps one consolidated copy per node, written by each
//!   node's local rank 0. Nodes that share a filesystem would publish into the same directory
//!   concurrently (remove-then-rename races), so [`elect_checkpoint_writer`] probes which writers
//!   see the same storage and lets exactly one of them (the lowest node) publish there.
//! * **Agreement.** Each rank resolves its resume checkpoint on its own filesystem; a partial
//!   per-node publish, or `rank0` checkpoints without a shared filesystem, would make the ranks
//!   resume at different steps and silently average diverged replicas. [`agree_resume`] makes
//!   every rank resume from the same step with the same saved state, or refuses typed.
//!   `last_loss` is excluded from the state fingerprint: it is the writing rank's own loss.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::dist::{CheckpointIo, Collective, DistError, ReduceOp};
use super::{list_checkpoints, read_checkpoint_state, sync_dir_files, CKPT_LATEST};
use gen_core::{Error, Result};

fn io_err(what: &str, p: &Path, e: std::io::Error) -> Error {
    Error::Msg(format!("iris checkpoint: {what} {}: {e}", p.display()))
}

/// Where a run's retired checkpoints live: the one sibling `<root>.prior` of its checkpoint root
/// `<output_dir>/<stem>.checkpoints` ([`super::checkpoint_root`]). Bounded to one retirement — the
/// newest replaces the previous — and part of a training's on-disk footprint: the product deletes
/// it together with the checkpoint root (sc-25690).
pub fn prior_checkpoint_root(output_dir: &Path, file_name: &str) -> PathBuf {
    prior_of(&super::checkpoint_root(output_dir, file_name))
}

fn prior_of(root: &Path) -> PathBuf {
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoints");
    root.with_file_name(format!("{name}.prior"))
}

/// Make room for a new retirement: the previous one is removed (bounded disk, one prior).
fn fresh_prior(root: &Path) -> Result<PathBuf> {
    let dest = prior_of(root);
    if dest.exists() {
        std::fs::remove_dir_all(&dest).map_err(|e| io_err("replace the prior", &dest, e))?;
        eprintln!(
            "iris training: removed the previous retirement {} (one prior is kept)",
            dest.display()
        );
    }
    Ok(dest)
}

/// The per-election probe files [`elect_checkpoint_writer`] leaves only when a crash interrupts
/// it: never content of the root.
const PROBE_PREFIX: &str = ".writer-probe-";

/// Whether `dir` is a checkpoint directory directly under `root`.
fn is_under(dir: &Path, root: &Path) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    dir.parent().map(canon) == Some(canon(root))
}

/// Move every checkpoint under `root` that is not this run's lineage aside, before the run
/// publishes its first checkpoint; returns where they went (`None`: nothing foreign).
///
/// * A fresh run, or one resuming from a checkpoint outside `root`: everything under `root` is
///   another run's, and the whole root moves (its checkpoints, `latest` pointer and metrics).
/// * A run resuming from `root/step_K`: the checkpoints newer than `K` are an abandoned future of
///   this lineage; they move, the older ones (this run's history) stay.
///
/// The moved checkpoints stay readable under [`prior_checkpoint_root`] until the next retirement
/// replaces them (one prior is kept: a retirement removes the previous one).
pub fn retire_foreign_checkpoints(
    root: &Path,
    resume_from: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if !root.is_dir() {
        return Ok(None);
    }
    // The step from the directory name: every task's checkpoint (generation, depth and the
    // restoration trainer's own state format) is published as `step_XXXXXXXX`.
    let lineage = resume_from.filter(|d| is_under(d, root)).and_then(|d| {
        d.file_name()?
            .to_str()?
            .strip_prefix("step_")?
            .parse::<u64>()
            .ok()
    });
    match lineage {
        None => {
            let empty = std::fs::read_dir(root)
                .map_err(|e| io_err("list", root, e))?
                .filter_map(|e| e.ok())
                .all(|e| e.file_name().to_string_lossy().starts_with(PROBE_PREFIX));
            if empty {
                return Ok(None);
            }
            let dest = fresh_prior(root)?;
            std::fs::rename(root, &dest).map_err(|e| io_err("move aside", root, e))?;
            eprintln!(
                "iris training: {} held another run's checkpoints; moved to {} (this run starts \
                 its own lineage)",
                root.display(),
                dest.display()
            );
            Ok(Some(dest))
        }
        Some(k) => {
            let newer: Vec<(u64, PathBuf)> = list_checkpoints(root)
                .into_iter()
                .filter(|(s, _)| *s > k)
                .collect();
            if newer.is_empty() {
                return Ok(None);
            }
            let dest = fresh_prior(root)?;
            std::fs::create_dir_all(&dest).map_err(|e| io_err("create", &dest, e))?;
            for (_, dir) in &newer {
                let to = dest.join(dir.file_name().expect("a step directory"));
                std::fs::rename(dir, &to).map_err(|e| io_err("move aside", dir, e))?;
            }
            // `latest` named a moved checkpoint: repoint it at the resume point.
            super::atomic_write(
                &root.join(CKPT_LATEST),
                super::checkpoint_dir_name(k).as_bytes(),
            )?;
            eprintln!(
                "iris training: resuming from step {k}; the newer checkpoints {:?} of the abandoned \
                 continuation moved to {}",
                newer.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
                dest.display()
            );
            Ok(Some(dest))
        }
    }
}

/// Whether this rank publishes consolidated checkpoints into `root`, collectively (every rank of
/// the world calls it, in the same order relative to its other collectives).
///
/// `rank0` / `dcp`: global rank 0. `per_node`: each node's local rank 0 is a candidate; the
/// candidates write a probe file into `root`, and a candidate publishes only when its node is the
/// lowest whose probe it sees: one writer per storage, however the nodes share filesystems. The
/// probes are removed before returning.
pub fn elect_checkpoint_writer(
    coll: &mut dyn Collective,
    io: CheckpointIo,
    root: &Path,
) -> Result<bool> {
    let world = coll.world().clone();
    if io != CheckpointIo::PerNode || world.world_size == 1 {
        return Ok(io.writes(&world));
    }
    let candidate = world.local_rank == 0;
    // A per-run token, so probes of a concurrent or crashed earlier election never count.
    let mut token = if world.rank == 0 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        format!("{:x}{:x}", std::process::id(), nanos).into_bytes()
    } else {
        Vec::new()
    };
    coll.broadcast(&mut token, 0)?;
    let token = String::from_utf8_lossy(&token).into_owned();
    let prefix = format!("{PROBE_PREFIX}{token}-node");
    let mine = root.join(format!("{prefix}{}", world.node_rank));
    let elect = |coll: &mut dyn Collective| -> Result<bool> {
        if candidate {
            std::fs::create_dir_all(root).map_err(|e| io_err("create", root, e))?;
            super::atomic_write(&mine, b"probe")?;
        }
        coll.barrier("checkpoint writer probe")?;
        let mut writes = false;
        if candidate {
            let lowest = std::fs::read_dir(root)
                .map_err(|e| io_err("list", root, e))?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.file_name()
                        .to_str()?
                        .strip_prefix(&prefix)?
                        .parse::<usize>()
                        .ok()
                })
                .min();
            writes = lowest == Some(world.node_rank);
        }
        coll.barrier("checkpoint writer elected")?;
        Ok(writes)
    };
    let result = elect(coll);
    if candidate {
        // On every path: this election's own probe, and any probe an interrupted earlier election
        // left (another token) — never one of this election's (a peer may still be reading).
        let _ = std::fs::remove_file(&mine);
        if let Ok(entries) = std::fs::read_dir(root) {
            for e in entries.filter_map(|e| e.ok()) {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with(PROBE_PREFIX) && !n.starts_with(&prefix) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
    result
}

/// A 48-bit fingerprint of a checkpoint's saved state (exact in an `f64`), `last_loss` excluded.
fn state_fingerprint(dir: &Path) -> Result<f64> {
    let mut v = read_checkpoint_state(dir)?.to_json();
    if let Some(o) = v.as_object_mut() {
        o.remove("last_loss");
    }
    let digest = Sha256::digest(v.to_string().as_bytes());
    let mut b = [0u8; 8];
    b[2..].copy_from_slice(&digest[..6]);
    Ok(u64::from_be_bytes(b) as f64)
}

/// `[min, max]` of one value over the world.
fn min_max(coll: &mut dyn Collective, x: f64) -> Result<(f64, f64)> {
    let mut v = [x, -x];
    coll.all_reduce_f64(&mut v, ReduceOp::Max)?;
    Ok((-v[1], v[0]))
}

/// Make every rank resume from the same checkpoint, collectively (every rank calls it).
///
/// `chosen` is this rank's own resolution; `search_root` is the directory it resolved `latest`
/// in (`None` when the request named one checkpoint directory). When the ranks resolved
/// different steps under a `latest` resolution, they fall back to the newest step **every** rank
/// holds (a crash during a per-node publish leaves the slower nodes one step behind); otherwise a
/// disagreement is a typed refusal ([`DistError::SplitWorld`]), as is a checkpoint whose saved
/// state differs between the ranks' copies ([`DistError::Desync`]). Returns the agreed checkpoint
/// on this rank's filesystem (`None`: no rank resumes).
pub fn agree_resume(
    coll: &mut dyn Collective,
    chosen: Option<PathBuf>,
    search_root: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if coll.world().world_size == 1 {
        return Ok(chosen);
    }
    let step = match &chosen {
        Some(d) => read_checkpoint_state(d)?.step as f64,
        None => -1.0,
    };
    let (lo, hi) = min_max(coll, step)?;
    if hi < 0.0 {
        return Ok(None);
    }
    let mut agreed = chosen;
    if lo != hi {
        let split = |why: String| -> Error {
            DistError::SplitWorld(format!(
                "the ranks resolved different resume checkpoints (steps {lo} .. {hi}; -1 = none): \
                 {why}. checkpoint_io rank0 needs one filesystem shared by every rank; per_node \
                 keeps a copy on every node"
            ))
            .into_error()
        };
        if lo < 0.0 {
            return Err(split(
                "a rank sees no checkpoint while another resumes".into(),
            ));
        }
        let Some(root) = search_root else {
            return Err(split(
                "the named resume checkpoint differs between the ranks".into(),
            ));
        };
        let target = lo as u64;
        let local = list_checkpoints(root)
            .into_iter()
            .find(|(s, _)| *s == target)
            .map(|(_, p)| p);
        let (all_have, _) = min_max(coll, if local.is_some() { 1.0 } else { 0.0 })?;
        if all_have < 1.0 {
            return Err(split(format!(
                "not every rank still holds step {target} (retention pruned it)"
            )));
        }
        eprintln!(
            "iris resume: the ranks' newest checkpoints differ (steps {lo} .. {hi}); every rank \
             resumes from step {target}, the newest one all of them hold"
        );
        agreed = local;
    }
    let dir = agreed.expect("a checkpoint every rank agreed on");
    let (flo, fhi) = min_max(coll, state_fingerprint(&dir)?)?;
    if flo != fhi {
        return Err(DistError::Desync {
            rank: coll.world().rank,
            expected: format!("one saved state for checkpoint {}", dir.display()),
            got: "copies whose state.json differ between the ranks".into(),
        }
        .into_error());
    }
    Ok(Some(dir))
}

/// The filesystem operations of [`publish_export_dir`] that can fail mid-publication (a seam for
/// the failure-path tests).
trait ExportFs {
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()>;
    fn remove_dir_all(&self, dir: &Path) -> std::io::Result<()>;
}

struct RealFs;

impl ExportFs for RealFs {
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        std::fs::rename(from, to)
    }
    fn remove_dir_all(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::remove_dir_all(dir)
    }
}

/// Publish a multi-file export directory atomically: `write` fills a sibling staging directory,
/// every file is synced, then the staging directory replaces `final_dir` by rename (the previous
/// export, if any, is moved out first and removed after). A reader never sees a half-written
/// export; a failure at any step removes the staging directory and leaves the previous export in
/// place (moved back if the publishing rename failed); a failure to remove the replaced export
/// after a successful publish is only reported. Leftovers of an interrupted earlier publication
/// (`.<name>.export-tmp*`, `.<name>.export-old*`) are swept first — a hidden previous export is
/// restored as `final_dir` when none exists (sc-25690).
pub fn publish_export_dir<E: From<Error>>(
    final_dir: &Path,
    write: impl FnOnce(&Path) -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    publish_export_dir_with(&RealFs, final_dir, write)
}

fn publish_export_dir_with<E: From<Error>>(
    fs: &dyn ExportFs,
    final_dir: &Path,
    write: impl FnOnce(&Path) -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
    let parent = final_dir.parent().unwrap_or(Path::new("."));
    let name = final_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("export")
        .to_string();
    std::fs::create_dir_all(parent).map_err(|e| io_err("create", parent, e))?;
    sweep_export_leftovers(fs, parent, &name, final_dir);
    let pid = std::process::id();
    let staging = parent.join(format!(".{name}.export-tmp{pid}"));
    let old = parent.join(format!(".{name}.export-old{pid}"));
    let abandon = |e: E| -> E {
        let _ = fs.remove_dir_all(&staging);
        e
    };
    std::fs::create_dir_all(&staging).map_err(|e| abandon(io_err("create", &staging, e).into()))?;
    write(&staging)
        .and_then(|()| sync_dir_files(&staging).map_err(E::from))
        .map_err(abandon)?;
    let moved_old = final_dir.exists();
    if moved_old {
        fs.rename(final_dir, &old)
            .map_err(|e| abandon(io_err("replace", final_dir, e).into()))?;
    }
    if let Err(e) = fs.rename(&staging, final_dir) {
        if moved_old {
            if let Err(back) = fs.rename(&old, final_dir) {
                eprintln!(
                    "iris export: could not restore the previous export {} from {}: {back}",
                    final_dir.display(),
                    old.display()
                );
            }
        }
        return Err(abandon(io_err("publish", final_dir, e).into()));
    }
    if let Ok(d) = std::fs::File::open(parent) {
        let _ = d.sync_all();
    }
    if moved_old {
        if let Err(e) = fs.remove_dir_all(&old) {
            eprintln!(
                "iris export: published {}; the replaced export {} could not be removed ({e}) and \
                 is swept by the next publication",
                final_dir.display(),
                old.display()
            );
        }
    }
    Ok(())
}

/// Sweep what an interrupted publication of `name` left beside it (best effort): staging
/// directories are removed; a moved-out previous export is restored when `final_dir` is missing,
/// otherwise removed.
fn sweep_export_leftovers(fs: &dyn ExportFs, parent: &Path, name: &str, final_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    let (tmp, old) = (format!(".{name}.export-tmp"), format!(".{name}.export-old"));
    let mut olds = Vec::new();
    for e in entries.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.starts_with(&tmp) {
            let _ = fs.remove_dir_all(&e.path());
        } else if n.starts_with(&old) {
            olds.push(e.path());
        }
    }
    olds.sort();
    if !final_dir.exists() {
        if let Some(prev) = olds.pop() {
            if fs.rename(&prev, final_dir).is_ok() {
                eprintln!(
                    "iris export: restored the previous export {} from {}",
                    final_dir.display(),
                    prev.display()
                );
            } else {
                olds.push(prev);
            }
        }
    }
    for o in olds {
        let _ = fs.remove_dir_all(&o);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{checkpoint_dir_name, latest_checkpoint, CheckpointState, CKPT_STATE};
    use crate::IrisTask;

    /// A minimal complete checkpoint at `root/step_<step>` with the given `lr` (part of the state
    /// fingerprint) and `last_loss` (excluded from it).
    fn checkpoint(root: &Path, step: u64, lr: f64, loss: f64) -> PathBuf {
        let dir = root.join(checkpoint_dir_name(step));
        std::fs::create_dir_all(&dir).unwrap();
        let state = CheckpointState {
            step,
            epoch: 1,
            batches_consumed: step as usize,
            scheduler_step: step,
            lr,
            nan_count: 0,
            last_loss: loss,
            artifact: "lora".into(),
            optimizer: "adamw".into(),
            ema_decay: Some(0.99),
            seed: 7,
            backend: "candle".into(),
            state_identity: "s".into(),
            data_identity: "d".into(),
            dataset_fingerprint: "f".into(),
            base_identity: "b".into(),
            task: IrisTask::Generation,
            stage: None,
            parent_stage: None,
            data_position: None,
            dist: None,
            repa_projector: None,
            grad_scaler: None,
        };
        std::fs::write(dir.join(CKPT_STATE), state.to_json().to_string()).unwrap();
        dir
    }

    #[test]
    fn a_fresh_run_moves_another_runs_checkpoints_aside() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("run.checkpoints");
        checkpoint(&root, 4, 1e-3, 0.5);
        checkpoint(&root, 8, 1e-3, 0.5);
        std::fs::write(root.join(CKPT_LATEST), "step_00000008").unwrap();
        let moved = retire_foreign_checkpoints(&root, None).unwrap().unwrap();
        assert!(!root.exists(), "the new run starts an empty root");
        assert_eq!(moved, tmp.path().join("run.checkpoints.prior"));
        assert_eq!(moved, prior_checkpoint_root(tmp.path(), "run.safetensors"));
        assert_eq!(
            list_checkpoints(&moved).len(),
            2,
            "the retired run stays readable"
        );
        // a second retirement replaces the first: one prior, bounded disk
        checkpoint(&root, 2, 1e-3, 0.5);
        let moved = retire_foreign_checkpoints(&root, None).unwrap().unwrap();
        assert_eq!(moved, tmp.path().join("run.checkpoints.prior"));
        let kept: Vec<u64> = list_checkpoints(&moved)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        assert_eq!(kept, [2], "only the newest retirement is kept");
        let siblings = std::fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(siblings, 1, "no other prior directory accumulates");
        // an empty root, or one holding only a crashed election's probe, stays
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(retire_foreign_checkpoints(&root, None).unwrap(), None);
        std::fs::write(root.join(".writer-probe-dead-node0"), b"probe").unwrap();
        assert_eq!(retire_foreign_checkpoints(&root, None).unwrap(), None);
    }

    #[test]
    fn resuming_inside_the_root_retires_only_the_abandoned_future() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("run.checkpoints");
        let at4 = checkpoint(&root, 4, 1e-3, 0.5);
        checkpoint(&root, 2, 1e-3, 0.5);
        checkpoint(&root, 8, 1e-3, 0.5);
        std::fs::write(root.join(CKPT_LATEST), "step_00000008").unwrap();
        let moved = retire_foreign_checkpoints(&root, Some(&at4))
            .unwrap()
            .unwrap();
        let kept: Vec<u64> = list_checkpoints(&root)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        assert_eq!(kept, [2, 4]);
        assert_eq!(latest_checkpoint(&root), Some(at4.clone()));
        assert_eq!(
            list_checkpoints(&moved)
                .into_iter()
                .map(|(s, _)| s)
                .collect::<Vec<_>>(),
            [8]
        );
        // the latest checkpoint itself: nothing newer, nothing moves
        assert_eq!(retire_foreign_checkpoints(&root, Some(&at4)).unwrap(), None);
        // a resume source outside the root: the whole root is foreign
        let other = tempfile::tempdir().unwrap();
        let ext = checkpoint(&other.path().join("x.checkpoints"), 3, 1e-3, 0.5);
        assert!(retire_foreign_checkpoints(&root, Some(&ext))
            .unwrap()
            .is_some());
        assert!(!root.exists());
    }

    #[test]
    fn an_export_directory_is_replaced_whole_and_a_failed_write_keeps_the_old_one() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        publish_export_dir::<Error>(&dir, |s| {
            std::fs::write(s.join("config.yaml"), "a")?;
            std::fs::write(s.join("stale.bin"), "x")?;
            Ok(())
        })
        .unwrap();
        publish_export_dir::<Error>(&dir, |s| {
            std::fs::write(s.join("config.yaml"), "b")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("config.yaml")).unwrap(),
            "b"
        );
        assert!(
            !dir.join("stale.bin").exists(),
            "the whole directory is replaced"
        );
        let e = publish_export_dir::<Error>(&dir, |s| {
            std::fs::write(s.join("config.yaml"), "c")?;
            Err(Error::Msg("disk full".into()))
        });
        assert!(e.is_err());
        assert_eq!(
            std::fs::read_to_string(dir.join("config.yaml")).unwrap(),
            "b"
        );
        let leftovers: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(leftovers, ["model"], "no staging directory is left behind");
    }

    /// An [`ExportFs`] failing one step of a publication.
    struct FailFs(&'static str);

    impl ExportFs for FailFs {
        fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            let (f, t) = (from.to_string_lossy(), to.to_string_lossy());
            let step = if t.contains(".export-old") {
                "final->old"
            } else if f.contains(".export-tmp") {
                "staging->final"
            } else {
                "other"
            };
            if step == self.0 {
                return Err(std::io::Error::other(format!("injected {step}")));
            }
            std::fs::rename(from, to)
        }
        fn remove_dir_all(&self, dir: &Path) -> std::io::Result<()> {
            if self.0 == "remove-old" && dir.to_string_lossy().contains(".export-old") {
                return Err(std::io::Error::other("injected remove-old"));
            }
            std::fs::remove_dir_all(dir)
        }
    }

    fn config_of(dir: &Path) -> String {
        std::fs::read_to_string(dir.join("config.yaml")).unwrap()
    }

    fn siblings(parent: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(parent)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    fn publish(fs: &dyn ExportFs, dir: &Path, text: &str) -> Result<()> {
        publish_export_dir_with::<Error>(fs, dir, |s| {
            std::fs::write(s.join("config.yaml"), text)?;
            Ok(())
        })
    }

    /// sc-25690: every failing step of a publication leaves the previous export in place and no
    /// staging directory; a failed removal of the replaced export does not fail the publication,
    /// and the next publication sweeps it.
    #[test]
    fn a_failed_publication_step_keeps_the_previous_export_and_no_leftovers() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        publish(&RealFs, &dir, "v1").unwrap();
        for step in ["final->old", "staging->final"] {
            assert!(publish(&FailFs(step), &dir, "v2").is_err(), "{step}");
            assert_eq!(config_of(&dir), "v1", "{step}: the previous export stays");
            assert_eq!(siblings(tmp.path()), ["model"], "{step}: no leftovers");
        }
        publish(&FailFs("remove-old"), &dir, "v3").unwrap();
        assert_eq!(config_of(&dir), "v3");
        assert_eq!(
            siblings(tmp.path()).len(),
            2,
            "the unremovable old export is left"
        );
        publish(&RealFs, &dir, "v4").unwrap();
        assert_eq!(config_of(&dir), "v4");
        assert_eq!(
            siblings(tmp.path()),
            ["model"],
            "swept by the next publication"
        );
    }

    /// sc-25690: an interrupted publication's leftovers are swept, and an export it had moved out
    /// (no export under the final name) is restored, even when the new publication fails.
    #[test]
    fn an_interrupted_publication_is_recovered_by_the_next() {
        let tmp = tempfile::tempdir().unwrap();
        let hidden = tmp.path().join(".model.export-old4242");
        std::fs::create_dir_all(&hidden).unwrap();
        std::fs::write(hidden.join("config.yaml"), "previous").unwrap();
        let staged = tmp.path().join(".model.export-tmp4242");
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(staged.join("model.safetensors"), "partial").unwrap();
        let dir = tmp.path().join("model");
        let failed = publish_export_dir::<Error>(&dir, |_| Err(Error::Msg("disk full".into())));
        assert!(failed.is_err());
        assert_eq!(config_of(&dir), "previous", "the hidden export is restored");
        assert_eq!(siblings(tmp.path()), ["model"]);
    }
}
