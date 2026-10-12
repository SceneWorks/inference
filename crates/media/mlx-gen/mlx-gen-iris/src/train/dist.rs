//! **Distributed** execution of the MLX Iris trainer (story sc-25689): MLX's own distributed layer
//! (`mlx::core::distributed` — the `ring` TCP backend, `jaccl` RDMA over Thunderbolt, `mpi`; bound
//! through the pinned mlx-rs fork's `mlx-sys`) behind the backend-neutral
//! [`gen_core::iris::train::dist::Collective`], plus the parameter / gradient / state movement the
//! replicated and sharded strategies need on MLX arrays.
//!
//! MLX's collectives run on the CPU stream (Metal has no implementation of them in the pinned
//! MLX) and read host-visible unified memory. The `ring` backend has neither group `split` nor
//! `sum_scatter`, so a collective over a sub-group (hybrid sharding's shard / replicate groups) is
//! the world all-gather of every rank's contribution reduced per group locally — the same values,
//! more traffic. Liveness: MLX's socket backends block on a vanished peer and take no timeout, so
//! every MLX collective is preceded by a bounded barrier on a host TCP side channel
//! ([`TcpCollective`], the run's rendezvous) **and** runs on a worker thread the trainer waits on
//! for at most the world's timeout: a rank lost before the collective fails the barrier, a rank
//! lost (or stalled) inside it expires the wait — either way a typed [`DistError::RankLost`]
//! within the timeout instead of a hang. A worker that never returns stays blocked inside MLX: it
//! is abandoned, the collective is poisoned (every later call fails with the same `RankLost`),
//! and the process must exit — MLX offers no way to cancel a collective.

use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use gen_core::iris::train::dist::{
    reduce_in_rank_order, Collective, DistError, DistPlan, Partition, Precision, ReduceOp,
    TcpCollective, World,
};
use mlx_gen::gen_core;
use mlx_gen::gen_core::CancelFlag;
use mlx_gen::{Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use super::optim::Params;

/// An MLX array's f32 values on the host.
pub fn to_host(a: &Array) -> Result<Vec<f32>> {
    if a.size() == 0 {
        // A zero-row shard (more shards than rows) has no host buffer to read.
        return Ok(Vec::new());
    }
    let f = a.as_dtype(Dtype::Float32)?;
    eval([&f])?;
    Ok(f.as_slice::<f32>().to_vec())
}

/// Host f32 values as an MLX array of `shape`.
pub fn from_host(v: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(v, shape)
}

/// MLX's distributed group as a [`Collective`].
pub struct MlxCollective {
    world: World,
    group: GroupHandle,
    /// The bounded liveness barrier + small host reductions.
    side: TcpCollective,
    /// Set once an MLX collective outlived the bounded wait: its worker is still blocked inside
    /// MLX, so the group can never be used again.
    poisoned: Option<DistError>,
    /// The run's cancel flag: a collective wait ends [`gen_core::Error::Canceled`] once a pending
    /// cancel outlived [`World::cancel_grace`] (sc-25690).
    cancel: CancelFlag,
}

/// The MLX group handle, moved to the collective's worker thread.
#[derive(Clone, Copy)]
struct GroupHandle(mlx_sys::mlx_distributed_group);

// SAFETY: the handle is an opaque pointer to MLX's group, which MLX's collectives use from the
// thread that issues them; one collective is in flight at a time (the trainer waits on it).
unsafe impl Send for GroupHandle {}

// SAFETY: the group handle is only used from the trainer's thread (and the one worker it waits on).
unsafe impl Send for MlxCollective {}

/// An MLX collective entry point (`mlx_distributed_all_sum` / `_all_gather`).
type MlxCollectiveFn = unsafe extern "C" fn(
    *mut mlx_sys::mlx_array,
    mlx_sys::mlx_array,
    mlx_sys::mlx_distributed_group,
    mlx_sys::mlx_stream,
) -> std::os::raw::c_int;

/// One MLX collective over host values, on the calling thread: the array is built from `data`,
/// reduced on an explicit CPU stream and read back (no default-device op).
fn run_collective(
    group: GroupHandle,
    data: &[f32],
    shape: &[i32],
    f: MlxCollectiveFn,
) -> std::result::Result<Vec<f32>, String> {
    let x = from_host(data, shape);
    // SAFETY: `res` is a fresh array handle owned by the returned `Array`; `x`, the group and the
    // stream outlive the call.
    unsafe {
        let stream = mlx_sys::mlx_default_cpu_stream_new();
        let mut res = mlx_sys::mlx_array_new();
        let status = f(&mut res, x.as_ptr(), group.0, stream);
        if status != 0 {
            mlx_sys::mlx_stream_free(stream);
            mlx_sys::mlx_array_free(res);
            return Err(format!("status {status}"));
        }
        let out = Array::from_ptr(res);
        let r = eval([&out]).map_err(|e| e.to_string());
        mlx_sys::mlx_stream_free(stream);
        r?;
        Ok(out.as_slice::<f32>().to_vec())
    }
}

/// How [`bounded_wait`] ended without a result.
#[derive(Debug)]
enum WaitEnd {
    /// The worker failed or vanished: the detail.
    Failed(String),
    /// No answer within the bound; the worker is abandoned (still blocked inside MLX).
    Expired,
    /// A pending cancel outlived the grace; the worker is abandoned (still blocked inside MLX).
    Canceled,
}

/// Run `work` (a call that may block inside MLX forever: MLX's socket backends take no timeout)
/// on a worker thread and wait for it in short slices: at most `bound`, and at most `grace` once
/// `cancel` is set. On either expiry the worker is abandoned, still blocked in MLX.
fn bounded_wait<T: Send + 'static>(
    name: &str,
    work: impl FnOnce() -> std::result::Result<T, String> + Send + 'static,
    bound: Duration,
    cancel: &CancelFlag,
    grace: Duration,
) -> std::result::Result<T, WaitEnd> {
    const SLICE: Duration = Duration::from_millis(200);
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name(format!("iris-mlx-{name}"))
        .spawn(move || {
            let _ = tx.send(work());
        })
        .map_err(|e| WaitEnd::Failed(format!("could not start the worker: {e}")))?;
    let deadline = Instant::now() + bound;
    let mut cancel_seen: Option<Instant> = None;
    loop {
        match rx.recv_timeout(SLICE) {
            Ok(Ok(v)) => return Ok(v),
            Ok(Err(e)) => return Err(WaitEnd::Failed(e)),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(WaitEnd::Failed("the worker panicked".into()))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if cancel.is_cancelled() {
                    let seen = *cancel_seen.get_or_insert_with(Instant::now);
                    if seen.elapsed() >= grace {
                        return Err(WaitEnd::Canceled);
                    }
                }
                if Instant::now() >= deadline {
                    return Err(WaitEnd::Expired);
                }
            }
        }
    }
}

/// Run one MLX collective on a worker thread, bounded by `timeout` and the cancel grace: MLX's
/// socket backends block forever on a peer that died or stalled inside the collective. On expiry
/// the worker is abandoned (still blocked in MLX) and the wait is a [`DistError::RankLost`]; on a
/// cancel past its grace, [`gen_core::Error::Canceled`] (the collective is poisoned either way).
#[allow(clippy::too_many_arguments)]
fn bounded_collective(
    group: GroupHandle,
    data: Vec<f32>,
    shape: Vec<i32>,
    f: MlxCollectiveFn,
    timeout: Duration,
    cancel: &CancelFlag,
    grace: Duration,
    rank: usize,
    op: &str,
) -> std::result::Result<Vec<f32>, (DistError, bool)> {
    let lost = |detail: String| DistError::RankLost {
        rank,
        op: op.to_string(),
        detail,
    };
    match bounded_wait(
        op,
        move || run_collective(group, &data, &shape, f),
        timeout,
        cancel,
        grace,
    ) {
        Ok(v) => Ok(v),
        Err(WaitEnd::Failed(e)) => Err((lost(format!("MLX distributed {op} failed ({e})")), false)),
        Err(WaitEnd::Expired) => Err((
            lost(format!(
                "no answer within the {}s transport timeout: a peer died or stalled inside the MLX \
                 {op} (the blocked collective is abandoned; this process must exit)",
                timeout.as_secs()
            )),
            false,
        )),
        Err(WaitEnd::Canceled) => Err((
            lost(format!(
                "the run was cancelled while waiting inside the MLX {op} (the blocked collective \
                 is abandoned; this process must exit)"
            )),
            true,
        )),
    }
}

/// The length header of a padded all-gather row, carried **bit-exactly** in an f32 slot (a
/// `len as f32` is exact only to 2^24 elements; the gather copies bits).
pub fn encode_len(len: usize) -> gen_core::Result<f32> {
    u32::try_from(len).map(f32::from_bits).map_err(|_| {
        DistError::Unsupported(format!(
            "an all-gather contribution of {len} elements exceeds the 2^32 the row header carries"
        ))
        .into_error()
    })
}

/// [`encode_len`] read back.
pub fn decode_len(slot: f32) -> usize {
    slot.to_bits() as usize
}

impl MlxCollective {
    /// Initialize MLX distributed with `backend` (`ring`, `jaccl`, `mpi`, or `any`). `ring` /
    /// `jaccl` read `MLX_RANK` and `MLX_HOSTFILE`; the world's `hostfile` is installed for them.
    ///
    /// The side channel's rendezvous honours `cancel` at once and is bounded by
    /// [`World::rendezvous_bound`]; `mlx::distributed::init` (which blocks on its own peers without
    /// a timeout) runs on a worker under the same bound and cancel: on expiry it is abandoned and
    /// the process must exit.
    pub fn init(world: &World, backend: &str, cancel: &CancelFlag) -> gen_core::Result<Self> {
        let side = TcpCollective::connect(world, cancel)?;
        if let Some(h) = &world.hostfile {
            std::env::set_var("MLX_HOSTFILE", h);
        }
        std::env::set_var("MLX_RANK", world.rank.to_string());
        let bk = CString::new(if backend == "any" { "" } else { backend })
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let bk_ptr = if backend == "any" {
            std::ptr::null()
        } else {
            bk.as_ptr()
        };
        // SAFETY: plain C calls with a valid NUL-terminated string (or null).
        let available = unsafe { mlx_sys::mlx_distributed_is_available(bk_ptr) };
        if !available {
            return Err(DistError::Unsupported(format!(
                "the MLX distributed backend {backend:?} is not available in this MLX build"
            ))
            .into_error());
        }
        let bk_owned: Option<CString> = (backend != "any").then(|| bk.clone());
        let init = move || {
            let ptr = bk_owned.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            // SAFETY: a valid NUL-terminated string (or null) owned by this closure.
            Ok(GroupHandle(unsafe {
                mlx_sys::mlx_distributed_init(true, ptr)
            }))
        };
        let group = match bounded_wait(
            "init",
            init,
            world.rendezvous_bound(),
            cancel,
            Duration::ZERO,
        ) {
            Ok(g) => g.0,
            Err(WaitEnd::Canceled) => return Err(gen_core::Error::Canceled),
            Err(WaitEnd::Expired) => {
                return Err(DistError::RankLost {
                    rank: world.rank,
                    op: "MLX distributed init".into(),
                    detail: format!(
                        "the {backend} backend's peers did not connect within {:?} (the blocked \
                         init is abandoned; this process must exit)",
                        world.rendezvous_bound()
                    ),
                }
                .into_error())
            }
            Err(WaitEnd::Failed(e)) => {
                return Err(DistError::Unsupported(format!(
                    "MLX distributed init ({backend}) failed: {e}"
                ))
                .into_error())
            }
        };
        if group.ctx.is_null() {
            return Err(DistError::Unsupported(format!(
                "MLX distributed init ({backend}) failed for rank {} of {}",
                world.rank, world.world_size
            ))
            .into_error());
        }
        let (rank, size) = unsafe {
            (
                mlx_sys::mlx_distributed_group_rank(group) as usize,
                mlx_sys::mlx_distributed_group_size(group) as usize,
            )
        };
        if rank != world.rank || size != world.world_size {
            return Err(DistError::SplitWorld(format!(
                "MLX distributed reports rank {rank} of {size}; the launcher declared rank {} of {}",
                world.rank, world.world_size
            ))
            .into_error());
        }
        Ok(Self {
            world: world.clone(),
            group: GroupHandle(group),
            side,
            poisoned: None,
            cancel: cancel.clone(),
        })
    }

    /// The bounded liveness barrier every MLX collective `op` is preceded by (public so a test
    /// can stand a rank at the barrier and then fail inside the collective).
    #[doc(hidden)]
    pub fn liveness_barrier(&mut self, op: &str) -> gen_core::Result<()> {
        if let Some(e) = &self.poisoned {
            return Err(e.clone().into_error());
        }
        self.side.barrier(op)
    }

    /// One MLX collective (after the liveness barrier), bounded by the world's timeout.
    fn mlx_op(
        &mut self,
        op: &str,
        data: Vec<f32>,
        shape: Vec<i32>,
        f: MlxCollectiveFn,
    ) -> gen_core::Result<Vec<f32>> {
        self.liveness_barrier(op)?;
        let r = bounded_collective(
            self.group,
            data,
            shape,
            f,
            self.world.timeout,
            &self.cancel,
            self.world.cancel_grace,
            self.world.rank,
            op,
        );
        r.map_err(|(e, canceled)| {
            if matches!(&e, DistError::RankLost { detail, .. } if detail.contains("abandoned")) {
                self.poisoned = Some(e.clone());
            }
            if canceled {
                gen_core::Error::Canceled
            } else {
                e.into_error()
            }
        })
    }

    /// The world all-gather of equal-length rows (`[world, n]`).
    fn gather_rows(&mut self, data: &[f32]) -> gen_core::Result<Vec<Vec<f32>>> {
        let n = data.len();
        let flat = self.mlx_op(
            "all_gather",
            data.to_vec(),
            vec![1, n as i32],
            mlx_sys::mlx_distributed_all_gather,
        )?;
        Ok(flat.chunks(n.max(1)).map(<[f32]>::to_vec).collect())
    }
}

impl Collective for MlxCollective {
    fn world(&self) -> &World {
        &self.world
    }

    fn all_reduce(
        &mut self,
        data: &mut [f32],
        op: ReduceOp,
        precision: Precision,
        groups: &Partition,
    ) -> gen_core::Result<()> {
        let whole = groups.0.iter().all(|&g| g == groups.0[0]);
        if whole && op == ReduceOp::Sum && precision == Precision::Fp32 {
            let out = self.mlx_op(
                "all_reduce",
                data.to_vec(),
                vec![data.len() as i32],
                mlx_sys::mlx_distributed_all_sum,
            )?;
            data.copy_from_slice(&out);
            return Ok(());
        }
        // Sub-groups, min/max and rounded precisions: the exact rank-order reduction of the
        // gathered contributions (identical on every member). Through the PADDED gather: a
        // sub-group's contributions may differ in length across groups (hybrid sharding's
        // replicate groups reduce row shards of different lengths), and MLX's ring all-gather
        // needs every rank's row equally long (sc-25690).
        let rows = self.all_gather(data, groups)?;
        let members = groups.members(self.world.rank);
        if let Some((m, r)) = members
            .iter()
            .zip(&rows)
            .find(|(_, r)| r.len() != data.len())
        {
            return Err(DistError::Desync {
                rank: *m,
                expected: format!("all_reduce of {} values matching its group", data.len()),
                got: format!("{} values", r.len()),
            }
            .into_error());
        }
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        data.copy_from_slice(&reduce_in_rank_order(&refs, op, precision));
        Ok(())
    }

    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> gen_core::Result<Vec<Vec<f32>>> {
        // Ring all-gather needs equal lengths: pad to the world maximum, carrying each length.
        let mut len = [data.len() as f64];
        self.side.all_reduce_f64(&mut len, ReduceOp::Max)?;
        let max = len[0] as usize;
        let mut padded = vec![encode_len(data.len())?];
        padded.extend_from_slice(data);
        padded.resize(max + 1, 0.0);
        let rows = self.gather_rows(&padded)?;
        groups
            .members(self.world.rank)
            .into_iter()
            .map(|m| {
                let n = decode_len(rows[m][0]);
                rows[m].get(1..=n).map(<[f32]>::to_vec).ok_or_else(|| {
                    DistError::Desync {
                        rank: m,
                        expected: format!("an all_gather row of at most {max} values"),
                        got: format!("a header of {n}"),
                    }
                    .into_error()
                })
            })
            .collect()
    }

    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> gen_core::Result<()> {
        self.side.broadcast(data, root)
    }

    fn barrier(&mut self, tag: &str) -> gen_core::Result<()> {
        self.side.barrier(tag)
    }

    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> gen_core::Result<()> {
        self.side.all_reduce_f64(data, op)
    }
}

/// The run's collective for `world`: none on one rank, the host TCP transport, or MLX's. The
/// rendezvous and every later wait honour `cancel` (sc-25690).
pub fn make_collective(
    world: &World,
    cancel: &CancelFlag,
) -> gen_core::Result<Box<dyn Collective>> {
    if world.world_size == 1 {
        return Ok(Box::new(gen_core::iris::train::dist::LocalCollective::new()));
    }
    Ok(match world.transport.as_str() {
        "tcp" => Box::new(TcpCollective::connect(world, cancel)?),
        "mlx_ring" => Box::new(MlxCollective::init(world, "ring", cancel)?),
        "mlx_jaccl" => Box::new(MlxCollective::init(world, "jaccl", cancel)?),
        "mlx_mpi" => Box::new(MlxCollective::init(world, "mpi", cancel)?),
        "mlx_any" => Box::new(MlxCollective::init(world, "any", cancel)?),
        other => {
            return Err(gen_core::Error::Msg(format!(
                "iris training: transport {other:?} is not available on the MLX backend"
            )))
        }
    })
}

// =============================================================================================
// The data-parallel state movement
// =============================================================================================

/// One rank's place in the replicated / sharded topology.
pub struct Topology {
    pub plan: DistPlan,
    pub coll: Box<dyn Collective>,
    /// Full shapes of every trained tensor (the sharded layout's reference).
    pub shapes: HashMap<Rc<str>, Vec<i32>>,
}

fn rows_of(shape: &[i32], n: usize, index: usize) -> std::ops::Range<usize> {
    let rows = shape.first().copied().unwrap_or(1) as usize;
    gen_core::iris::train::dist::chunk_rows(rows, n, index)
}

impl Topology {
    pub fn world(&self) -> &World {
        &self.plan.world
    }

    pub fn is_main(&self) -> bool {
        self.plan.world.rank == 0
    }

    /// This rank's rows of a full tensor (the whole tensor when not sharded).
    pub fn shard_of(&self, key: &str, full: &Array) -> Result<Array> {
        if !self.plan.sharded() {
            return Ok(full.clone());
        }
        let r = rows_of(full.shape(), self.plan.dp_shard, self.plan.shard_index());
        let _ = key;
        full.take_along_rows(r)
    }

    /// Every rank's shard of `key` gathered back into the full tensor (within the shard group).
    pub fn gather_full(&mut self, key: &str, shard: &Array) -> Result<Array> {
        if !self.plan.sharded() {
            return Ok(shard.clone());
        }
        let full = self
            .shapes
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Msg(format!("iris distributed: no shape for {key}")))?;
        let parts = self
            .coll
            .all_gather(&to_host(shard)?, &self.plan.shard_groups())?;
        let flat: Vec<f32> = parts.into_iter().flatten().collect();
        let numel: i32 = full.iter().product();
        if flat.len() != numel as usize {
            return Err(Error::Msg(format!(
                "iris distributed: the shards of {key} gather to {} values, its full shape {full:?} \
                 holds {numel} (local shard {:?})",
                flat.len(),
                shard.shape()
            )));
        }
        Ok(from_host(&flat, &full))
    }

    /// Gather every tensor of a sharded map back to full tensors.
    pub fn gather_map(&mut self, shards: &Params) -> Result<Params> {
        let mut keys: Vec<Rc<str>> = shards.keys().cloned().collect();
        keys.sort();
        let mut out = Params::with_capacity(shards.len());
        for k in keys {
            let full = self.gather_full(&k, &shards[&k])?;
            out.insert(k, full);
        }
        Ok(out)
    }

    /// Reduce one window's local gradients (the local mean over this rank's micro-batches) to the
    /// global mean: DDP all-reduces every gradient (`bucket_cap_mb` buckets); FSDP reduce-scatters
    /// within the shard group and all-reduces across replicate groups, returning this rank's
    /// gradient **rows**. Reductions run in `reduce_dtype`.
    pub fn reduce_grads(&mut self, grads: Params) -> Result<Params> {
        let world = self.plan.world.world_size as f32;
        if world == 1.0 {
            return Ok(grads);
        }
        let mut keys: Vec<Rc<str>> = grads.keys().cloned().collect();
        keys.sort();
        let prec = self.plan.reduce_dtype;
        let whole = Partition::whole(self.plan.world.world_size);
        let mut out = Params::with_capacity(grads.len());
        if !self.plan.sharded() {
            let cap = self.plan.ddp.bucket_elems();
            let mut bucket: Vec<(Rc<str>, Vec<i32>, usize)> = Vec::new();
            let mut buf: Vec<f32> = Vec::new();
            let flush = |bucket: &mut Vec<(Rc<str>, Vec<i32>, usize)>,
                         buf: &mut Vec<f32>,
                         out: &mut Params,
                         coll: &mut Box<dyn Collective>|
             -> Result<()> {
                if bucket.is_empty() {
                    return Ok(());
                }
                coll.all_reduce(buf, ReduceOp::Sum, prec, &whole)?;
                let mut off = 0;
                for (k, shape, n) in bucket.drain(..) {
                    let v: Vec<f32> = buf[off..off + n].iter().map(|x| x / world).collect();
                    out.insert(k, from_host(&v, &shape));
                    off += n;
                }
                buf.clear();
                Ok(())
            };
            for k in keys {
                let g = &grads[&k];
                let host = to_host(g)?;
                if !buf.is_empty() && buf.len() + host.len() > cap {
                    flush(&mut bucket, &mut buf, &mut out, &mut self.coll)?;
                }
                bucket.push((k, g.shape().to_vec(), host.len()));
                buf.extend(host);
            }
            flush(&mut bucket, &mut buf, &mut out, &mut self.coll)?;
            return Ok(out);
        }
        let shard_groups = self.plan.shard_groups();
        let rep_groups = self.plan.replicate_groups();
        for k in keys {
            let g = &grads[&k];
            let mut host = to_host(g)?;
            // reduce-scatter within the shard group (sum), then all-reduce the rows across the
            // replicate group (sum): the global sum, this rank's rows.
            self.coll
                .all_reduce(&mut host, ReduceOp::Sum, prec, &shard_groups)?;
            let shape = g.shape().to_vec();
            let r = rows_of(&shape, self.plan.dp_shard, self.plan.shard_index());
            let row_len: usize = shape.iter().skip(1).map(|&d| d as usize).product();
            let mut rows = host[r.start * row_len..r.end * row_len].to_vec();
            if self.plan.dp_replicate > 1 {
                self.coll
                    .all_reduce(&mut rows, ReduceOp::Sum, prec, &rep_groups)?;
            }
            for v in rows.iter_mut() {
                *v /= world;
            }
            let mut rshape = shape.clone();
            if !rshape.is_empty() {
                rshape[0] = (r.end - r.start) as i32;
            }
            out.insert(k, from_host(&rows, &rshape));
        }
        Ok(out)
    }

    /// `f64` scalars reduced over the whole world.
    pub fn reduce_scalars(&mut self, v: &mut [f64], op: ReduceOp) -> Result<()> {
        Ok(self.coll.all_reduce_f64(v, op)?)
    }

    /// The squared-norm sum of a (possibly sharded) gradient map, over the world's rows.
    pub fn global_sq_norm(&mut self, local_sq: f64) -> Result<f64> {
        if !self.plan.sharded() || self.plan.dp_shard == 1 {
            return Ok(local_sq);
        }
        // Each shard group holds every row once; replicate groups hold identical copies.
        let mut v = [local_sq / self.plan.dp_replicate as f64];
        self.coll.all_reduce_f64(&mut v, ReduceOp::Sum)?;
        Ok(v[0])
    }
}

/// Row selection on an MLX array.
trait TakeRows {
    fn take_along_rows(&self, r: std::ops::Range<usize>) -> Result<Array>;
}

impl TakeRows for Array {
    fn take_along_rows(&self, r: std::ops::Range<usize>) -> Result<Array> {
        use mlx_rs::ops::indexing::IndexOp;
        if self.ndim() == 0 {
            return Ok(self.clone());
        }
        Ok(self.index_device(
            (r.start as i32)..(r.end as i32),
            mlx_rs::Stream::task_local_or_default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_row_lengths_survive_past_two_to_the_24() {
        for len in [
            0usize,
            1,
            7,
            1 << 24,
            (1 << 24) + 1,
            (1 << 31) + 3,
            u32::MAX as usize,
        ] {
            assert_eq!(decode_len(encode_len(len).unwrap()), len, "{len}");
        }
        // the old `len as f32` header lost the length past 2^24
        assert_ne!(((1usize << 24) + 1) as f32 as usize, (1 << 24) + 1);
        assert!(encode_len(u32::MAX as usize + 1).is_err());
    }

    /// [`bounded_wait`] with its bound and grace in milliseconds.
    fn run_bounded<T: Send + 'static>(
        work: impl FnOnce() -> std::result::Result<T, String> + Send + 'static,
        bound_ms: u64,
        flag: &CancelFlag,
        grace_ms: u64,
    ) -> std::result::Result<T, WaitEnd> {
        bounded_wait(
            "t",
            work,
            Duration::from_millis(bound_ms),
            flag,
            Duration::from_millis(grace_ms),
        )
    }

    /// A call that blocks forever (MLX's init or a collective on a vanished peer): the wait ends
    /// typed at its bound, at once on a cancel with no grace, and a cancel within the grace still
    /// lets a late answer through (sc-25690). The bounds are long enough that only the cancel can
    /// end the cancelled waits as `Canceled`.
    #[test]
    fn a_blocked_mlx_call_is_bounded_and_cancellable() {
        fn forever() -> std::result::Result<(), String> {
            let (_tx, rx) = mpsc::channel::<()>();
            let _ = rx.recv();
            Ok(())
        }
        fn late() -> std::result::Result<u32, String> {
            std::thread::sleep(Duration::from_millis(500));
            Ok(7)
        }
        let flag = CancelFlag::new();
        let expired = run_bounded(forever, 300, &flag, 0);
        assert!(matches!(expired, Err(WaitEnd::Expired)), "{expired:?}");
        flag.cancel();
        let cancelled = run_bounded(forever, 3_600_000, &flag, 0);
        assert!(matches!(cancelled, Err(WaitEnd::Canceled)), "{cancelled:?}");
        let answered = run_bounded(late, 3_600_000, &flag, 60_000);
        assert!(matches!(answered, Ok(7)), "{answered:?}");
    }
}
