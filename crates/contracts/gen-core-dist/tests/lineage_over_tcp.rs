//! Iris checkpoint lineage (`gen_core::iris::train::lineage`) driven over the real host TCP
//! transport: resume agreement, per-node writer election and probe cleanup across loopback ranks.
//! These ran as gen-core unit tests until the transport moved here (sc-27316).

use std::path::{Path, PathBuf};

use gen_core::iris::train::dist::{
    dist_error, CheckpointIo, Collective, DistError, Partition, Precision, ReduceOp, World,
};
use gen_core::iris::train::lineage::{agree_resume, elect_checkpoint_writer};
use gen_core::iris::train::{
    checkpoint_dir_name, latest_checkpoint, read_checkpoint_state, CheckpointState, CKPT_STATE,
};
use gen_core::iris::IrisTask;
use gen_core::{CancelFlag, Error, Result};
use gen_core_dist::TcpCollective;

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
        matches!(dist_error(&e), Some(DistError::SplitWorld(_)))
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
        matches!(dist_error(&e), Some(DistError::Desync { .. }))
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
        precision: Precision,
        groups: &Partition,
    ) -> Result<()> {
        self.inner.all_reduce(data, op, precision, groups)
    }
    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> Result<Vec<Vec<f32>>> {
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

/// A collective whose `tag` barrier fails.
struct FailBarrier<'a> {
    inner: &'a mut dyn Collective,
    world: World,
    tag: &'static str,
}

impl Collective for FailBarrier<'_> {
    fn world(&self) -> &World {
        &self.world
    }
    fn all_reduce(
        &mut self,
        data: &mut [f32],
        op: ReduceOp,
        precision: Precision,
        groups: &Partition,
    ) -> Result<()> {
        self.inner.all_reduce(data, op, precision, groups)
    }
    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> Result<Vec<Vec<f32>>> {
        self.inner.all_gather(data, groups)
    }
    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> Result<()> {
        self.inner.broadcast(data, root)
    }
    fn barrier(&mut self, tag: &str) -> Result<()> {
        if tag == self.tag {
            return Err(Error::Msg(format!("injected failure at {tag}")));
        }
        self.inner.barrier(tag)
    }
    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> Result<()> {
        self.inner.all_reduce_f64(data, op)
    }
}

/// sc-25690: an election that fails leaves no probe of its own, and every election sweeps the
/// probes an interrupted earlier one left — so a later fresh run never mistakes the root for
/// another run's.
#[test]
fn an_election_leaves_no_probe_behind() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("run.checkpoints");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join(".writer-probe-deadbeef-node0"), b"probe").unwrap();
    let base = root.clone();
    let got = on_ranks(2, move |r, c| {
        let mut w = c.world().clone();
        w.local_world_size = 1;
        w.local_rank = 0;
        w.node_rank = r;
        w.num_nodes = 2;
        let mut view = FailBarrier {
            inner: c,
            world: w,
            tag: if r == 1 {
                "checkpoint writer elected"
            } else {
                "never"
            },
        };
        elect_checkpoint_writer(&mut view, CheckpointIo::PerNode, &base).is_err()
    });
    assert!(got[1], "rank 1's election failed");
    assert_eq!(
        std::fs::read_dir(&root).unwrap().count(),
        0,
        "no probe is left behind, the stale one swept"
    );
}
