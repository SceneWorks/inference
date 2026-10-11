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
//! more traffic. Liveness: MLX's socket backends block on a vanished peer, so every MLX collective
//! is preceded by a bounded barrier on a host TCP side channel ([`TcpCollective`], the run's
//! rendezvous): a lost rank surfaces as a typed [`DistError::RankLost`] within the timeout instead
//! of a hang.

use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;

use gen_core::iris::train::dist::{
    reduce_in_rank_order, Collective, DistError, DistPlan, Partition, Precision, ReduceOp,
    TcpCollective, World,
};
use mlx_gen::gen_core;
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
    group: mlx_sys::mlx_distributed_group,
    /// The bounded liveness barrier + small host reductions.
    side: TcpCollective,
}

// SAFETY: the group handle is only used from the trainer's thread.
unsafe impl Send for MlxCollective {}

impl MlxCollective {
    /// Initialize MLX distributed with `backend` (`ring`, `jaccl`, `mpi`, or `any`). `ring` /
    /// `jaccl` read `MLX_RANK` and `MLX_HOSTFILE`; the world's `hostfile` is installed for them.
    pub fn init(world: &World, backend: &str) -> gen_core::Result<Self> {
        let side = TcpCollective::connect(world)?;
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
        let group = unsafe { mlx_sys::mlx_distributed_init(true, bk_ptr) };
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
            group,
            side,
        })
    }

    fn live(&mut self, op: &str) -> gen_core::Result<()> {
        self.side.barrier(op)
    }

    fn mlx_op(
        &self,
        op: &str,
        x: &Array,
        f: unsafe extern "C" fn(
            *mut mlx_sys::mlx_array,
            mlx_sys::mlx_array,
            mlx_sys::mlx_distributed_group,
            mlx_sys::mlx_stream,
        ) -> std::os::raw::c_int,
    ) -> gen_core::Result<Array> {
        // SAFETY: `res` is a fresh array handle owned by the returned `Array`; `x`, the group and
        // the stream outlive the call.
        unsafe {
            let stream = mlx_sys::mlx_default_cpu_stream_new();
            let mut res = mlx_sys::mlx_array_new();
            let status = f(&mut res, x.as_ptr(), self.group, stream);
            mlx_sys::mlx_stream_free(stream);
            if status != 0 {
                mlx_sys::mlx_array_free(res);
                return Err(DistError::RankLost {
                    rank: self.world.rank,
                    op: op.to_string(),
                    detail: format!("MLX distributed {op} failed (status {status})"),
                }
                .into_error());
            }
            let out = Array::from_ptr(res);
            eval([&out]).map_err(|e| {
                DistError::RankLost {
                    rank: self.world.rank,
                    op: op.to_string(),
                    detail: e.to_string(),
                }
                .into_error()
            })?;
            Ok(out)
        }
    }

    /// The world all-gather of equal-length rows (`[world, n]`).
    fn gather_rows(&mut self, data: &[f32]) -> gen_core::Result<Vec<Vec<f32>>> {
        self.live("all_gather")?;
        let n = data.len();
        let x = from_host(data, &[1, n as i32]);
        let g = self.mlx_op("all_gather", &x, mlx_sys::mlx_distributed_all_gather)?;
        let flat = g.as_slice::<f32>().to_vec();
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
            self.live("all_reduce")?;
            let x = from_host(data, &[data.len() as i32]);
            let out = self.mlx_op("all_sum", &x, mlx_sys::mlx_distributed_all_sum)?;
            data.copy_from_slice(out.as_slice::<f32>());
            return Ok(());
        }
        // Sub-groups, min/max and rounded precisions: the exact rank-order reduction of the
        // gathered contributions (identical on every member).
        let rows = self.gather_rows(data)?;
        let members = groups.members(self.world.rank);
        let refs: Vec<&[f32]> = members.iter().map(|&m| rows[m].as_slice()).collect();
        data.copy_from_slice(&reduce_in_rank_order(&refs, op, precision));
        Ok(())
    }

    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> gen_core::Result<Vec<Vec<f32>>> {
        // Ring all-gather needs equal lengths: pad to the world maximum, carrying each length.
        let mut len = [data.len() as f64];
        self.side.all_reduce_f64(&mut len, ReduceOp::Max)?;
        let max = len[0] as usize;
        let mut padded = vec![data.len() as f32];
        padded.extend_from_slice(data);
        padded.resize(max + 1, 0.0);
        let rows = self.gather_rows(&padded)?;
        Ok(groups
            .members(self.world.rank)
            .into_iter()
            .map(|m| {
                let n = rows[m][0] as usize;
                rows[m][1..=n].to_vec()
            })
            .collect())
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

/// The run's collective for `world`: none on one rank, the host TCP transport, or MLX's.
pub fn make_collective(world: &World) -> gen_core::Result<Box<dyn Collective>> {
    if world.world_size == 1 {
        return Ok(Box::new(gen_core::iris::train::dist::LocalCollective::new()));
    }
    Ok(match world.transport.as_str() {
        "tcp" => Box::new(TcpCollective::connect(world)?),
        "mlx_ring" => Box::new(MlxCollective::init(world, "ring")?),
        "mlx_jaccl" => Box::new(MlxCollective::init(world, "jaccl")?),
        "mlx_mpi" => Box::new(MlxCollective::init(world, "mpi")?),
        "mlx_any" => Box::new(MlxCollective::init(world, "any")?),
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
        Ok(full.take_along_rows(r)?)
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
