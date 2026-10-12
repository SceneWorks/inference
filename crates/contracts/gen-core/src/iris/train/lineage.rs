//! **Checkpoint lineage** of the Iris trainers (story sc-25690): which checkpoints under a run's
//! checkpoint root belong to the run, which ranks publish into a root, which checkpoint every rank
//! resumes from, and the atomic publication of a multi-file export directory.
//!
//! * **Lineage.** A run's retention ([`super::prune_checkpoints`]) and its `latest` resolution
//!   ([`super::latest_checkpoint`]) see every `step_*` directory under the root, so a root holding
//!   another run's checkpoints would keep that run's high steps, delete the new run's, and resume
//!   the old run. [`retire_foreign_checkpoints`] moves every checkpoint that is not this run's
//!   lineage aside (into a sibling `<root>.prior-<n>` directory, never deleted) before the run
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
use crate::{Error, Result};

fn io_err(what: &str, p: &Path, e: std::io::Error) -> Error {
    Error::Msg(format!("iris checkpoint: {what} {}: {e}", p.display()))
}

/// The first free sibling `<root name>.prior-<n>` of `root`.
fn prior_dir(root: &Path) -> PathBuf {
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoints")
        .to_string();
    let parent = root.parent().unwrap_or(Path::new("."));
    (1u32..)
        .map(|n| parent.join(format!("{name}.prior-{n}")))
        .find(|p| !p.exists())
        .expect("a free prior directory")
}

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
/// Nothing is deleted: the moved checkpoints stay readable under `<root>.prior-<n>`.
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
                .next()
                .is_none();
            if empty {
                return Ok(None);
            }
            let dest = prior_dir(root);
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
            let dest = prior_dir(root);
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
    let prefix = format!(".writer-probe-{token}-node");
    let mine = root.join(format!("{prefix}{}", world.node_rank));
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
    if candidate {
        std::fs::remove_file(&mine).map_err(|e| io_err("remove probe", &mine, e))?;
    }
    Ok(writes)
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

/// Publish a multi-file export directory atomically: `write` fills a sibling staging directory,
/// every file is synced, then the staging directory replaces `final_dir` by rename (the previous
/// export, if any, is moved out first and removed after). A reader never sees a half-written
/// export, and a failed write leaves the previous export in place.
pub fn publish_export_dir<E: From<Error>>(
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
    let pid = std::process::id();
    let staging = parent.join(format!(".{name}.export-tmp{pid}"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(|e| io_err("clear", &staging, e))?;
    }
    std::fs::create_dir_all(&staging).map_err(|e| io_err("create", &staging, e))?;
    if let Err(e) = write(&staging).and_then(|()| sync_dir_files(&staging).map_err(E::from)) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    let old = parent.join(format!(".{name}.export-old{pid}"));
    if final_dir.exists() {
        if old.exists() {
            std::fs::remove_dir_all(&old).map_err(|e| io_err("clear", &old, e))?;
        }
        std::fs::rename(final_dir, &old).map_err(|e| io_err("replace", final_dir, e))?;
    }
    std::fs::rename(&staging, final_dir).map_err(|e| io_err("publish", final_dir, e))?;
    if let Ok(d) = std::fs::File::open(parent) {
        let _ = d.sync_all();
    }
    if old.exists() {
        std::fs::remove_dir_all(&old).map_err(|e| io_err("remove previous", &old, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iris::train::dist::{TcpCollective, World};
    use crate::iris::train::{checkpoint_dir_name, latest_checkpoint, CheckpointState, CKPT_STATE};
    use crate::iris::IrisTask;
    use crate::CancelFlag;

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
        assert_eq!(moved, tmp.path().join("run.checkpoints.prior-1"));
        assert_eq!(list_checkpoints(&moved).len(), 2, "nothing is deleted");
        // a second fresh run moves the next one to prior-2; an empty root stays
        checkpoint(&root, 2, 1e-3, 0.5);
        let moved = retire_foreign_checkpoints(&root, None).unwrap().unwrap();
        assert_eq!(moved, tmp.path().join("run.checkpoints.prior-2"));
        std::fs::create_dir_all(&root).unwrap();
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

    fn world(rank: usize, n: usize, nodes: usize, port: u16) -> World {
        let mut w = World::single();
        w.rank = rank;
        w.world_size = n;
        w.local_world_size = n / nodes;
        w.local_rank = rank % (n / nodes);
        w.node_rank = rank / (n / nodes);
        w.num_nodes = nodes;
        w.master_port = port;
        w
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    /// Run `f(rank, collective)` on every rank of an `n`-rank, `nodes`-node loopback world. The
    /// worlds declare several nodes on loopback, which `assert_single_world` refuses, so the
    /// multi-node cases pass `nodes = 1` worlds and model the node layout in `f`.
    fn on_ranks<T: Send + 'static>(
        n: usize,
        f: impl Fn(usize, &mut dyn Collective) -> T + Send + Sync + Clone + 'static,
    ) -> Vec<T> {
        let port = free_port();
        let handles: Vec<_> = (0..n)
            .map(|r| {
                let f = f.clone();
                std::thread::spawn(move || {
                    let mut c =
                        TcpCollective::connect(&world(r, n, 1, port), &CancelFlag::new()).unwrap();
                    f(r, &mut c)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn ranks_resume_from_one_agreed_checkpoint_or_refuse_typed() {
        // per-node copies, node 1 one step behind (its publish of step 8 was lost): both resume 4
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        for node in 0..2 {
            let root = base.join(format!("node{node}"));
            checkpoint(&root, 4, 1e-3, 0.1 * node as f64);
            if node == 0 {
                checkpoint(&root, 8, 1e-3, 0.5);
            }
        }
        let got = on_ranks(2, move |r, c| {
            let root = base.join(format!("node{r}"));
            agree_resume(c, latest_checkpoint(&root), Some(&root))
                .map(|d| d.map(|d| read_checkpoint_state(&d).unwrap().step))
                .map_err(|e| e.to_string())
        });
        assert_eq!(got, vec![Ok(Some(4)), Ok(Some(4))]);

        // rank0 checkpoints without a shared filesystem: rank 1 sees none -> typed refusal
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        checkpoint(&base.join("node0"), 4, 1e-3, 0.5);
        let got = on_ranks(2, move |r, c| {
            let root = base.join(format!("node{r}"));
            let e = agree_resume(c, latest_checkpoint(&root), Some(&root)).unwrap_err();
            matches!(
                crate::iris::train::dist::dist_error(&e),
                Some(DistError::SplitWorld(_))
            )
        });
        assert_eq!(got, vec![true, true]);

        // the same step with a different saved state -> typed desync; a differing last_loss (the
        // writer's own loss) is not a difference
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        checkpoint(&base.join("node0"), 4, 1e-3, 0.5);
        checkpoint(&base.join("node1"), 4, 2e-3, 0.7);
        let got = on_ranks(2, move |r, c| {
            let root = base.join(format!("node{r}"));
            let e = agree_resume(c, latest_checkpoint(&root), Some(&root)).unwrap_err();
            matches!(
                crate::iris::train::dist::dist_error(&e),
                Some(DistError::Desync { .. })
            )
        });
        assert_eq!(got, vec![true, true]);

        // nobody has one: a fresh start everywhere
        let got = on_ranks(2, |_, c| agree_resume(c, None, None).unwrap());
        assert_eq!(got, vec![None, None]);
    }

    /// `per_node` writers: nodes on one shared filesystem elect one writer; nodes on separate
    /// filesystems each write their own copy.
    #[test]
    fn per_node_writers_are_one_per_storage() {
        // 4 ranks as 2 nodes x 2 local ranks (the node layout lives in the closure: loopback
        // worlds are single-node by assert_single_world).
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared");
        let elect = |storage: fn(usize, &Path) -> PathBuf, base: PathBuf| {
            on_ranks(4, move |r, c| {
                let mut w = c.world().clone();
                w.local_world_size = 2;
                w.local_rank = r % 2;
                w.node_rank = r / 2;
                w.num_nodes = 2;
                let mut view = NodeView { inner: c, world: w };
                let root = storage(r / 2, &base);
                elect_checkpoint_writer(&mut view, CheckpointIo::PerNode, &root).unwrap()
            })
        };
        let one_fs = elect(|_, base| base.join("run.checkpoints"), shared.clone());
        assert_eq!(one_fs, vec![true, false, false, false]);
        let two_fs = elect(
            |node, base| base.join(format!("node{node}/run.checkpoints")),
            shared.clone(),
        );
        assert_eq!(two_fs, vec![true, false, true, false]);
        // the probes are gone
        for d in [
            shared.join("run.checkpoints"),
            shared.join("node0/run.checkpoints"),
            shared.join("node1/run.checkpoints"),
        ] {
            assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "{}", d.display());
        }
        // rank0 / single rank: no election
        let got = on_ranks(2, |_, c| {
            elect_checkpoint_writer(c, CheckpointIo::Rank0, Path::new("/nonexistent")).unwrap()
        });
        assert_eq!(got, vec![true, false]);
    }

    /// A collective whose world reports a multi-node layout (loopback test worlds are single-node).
    struct NodeView<'a> {
        inner: &'a mut dyn Collective,
        world: World,
    }

    impl Collective for NodeView<'_> {
        fn world(&self) -> &World {
            &self.world
        }
        fn all_reduce(
            &mut self,
            data: &mut [f32],
            op: ReduceOp,
            precision: super::super::dist::Precision,
            groups: &super::super::dist::Partition,
        ) -> Result<()> {
            self.inner.all_reduce(data, op, precision, groups)
        }
        fn all_gather(
            &mut self,
            data: &[f32],
            groups: &super::super::dist::Partition,
        ) -> Result<Vec<Vec<f32>>> {
            self.inner.all_gather(data, groups)
        }
        fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> Result<()> {
            self.inner.broadcast(data, root)
        }
        fn barrier(&mut self, tag: &str) -> Result<()> {
            self.inner.barrier(tag)
        }
        fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> Result<()> {
            self.inner.all_reduce_f64(data, op)
        }
    }
}
