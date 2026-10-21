//! **Distributed** execution of the Candle Iris trainer (story sc-25689): the replicated and
//! sharded strategies over the backend-neutral [`gen_core::iris::train::dist::Collective`].
//!
//! Transport: the host-staged [`TcpCollective`] — NCCL does not exist on Windows (the repo's CUDA
//! runners) and the repo had no distributed layer, so ranks exchange host copies of their tensors
//! over TCP (bounded waits, typed failures). The MLX transports (`mlx_*`) are refused typed here.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core;
use candle_gen::gen_core::iris::train::dist::{
    chunk_rows, Collective, DistError, DistPlan, Partition, ReduceOp, TcpCollective, World,
};
use candle_gen::{CandleError as Error, Result};

use super::optim::Tensors;

/// A tensor's f32 values on the host.
pub fn to_host(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

/// The run's collective for `world`: none on one rank, else the host TCP transport. The rendezvous
/// and every later wait honour `cancel` (sc-25690).
pub fn make_collective(
    world: &World,
    cancel: &gen_core::CancelFlag,
) -> gen_core::Result<Box<dyn Collective>> {
    if world.world_size == 1 {
        return Ok(Box::new(gen_core::iris::train::dist::LocalCollective::new()));
    }
    match world.transport.as_str() {
        "tcp" => Ok(Box::new(TcpCollective::connect(world, cancel)?)),
        other => Err(DistError::Unsupported(format!(
            "transport {other:?} is MLX's distributed layer; the Candle trainer's transport is tcp"
        ))
        .into_error()),
    }
}

/// One rank's place in the replicated / sharded topology.
pub struct Topology {
    pub plan: DistPlan,
    pub coll: Box<dyn Collective>,
    /// Full shapes of every trained tensor.
    pub shapes: HashMap<String, Vec<usize>>,
    pub device: Device,
}

fn rows_of(shape: &[usize], n: usize, index: usize) -> std::ops::Range<usize> {
    chunk_rows(shape.first().copied().unwrap_or(1), n, index)
}

impl Topology {
    pub fn world(&self) -> &World {
        &self.plan.world
    }

    pub fn is_main(&self) -> bool {
        self.plan.world.rank == 0
    }

    /// This rank's rows of a full tensor (the whole tensor when not sharded).
    pub fn shard_of(&self, full: &Tensor) -> Result<Tensor> {
        if !self.plan.sharded() || full.rank() == 0 {
            return Ok(full.clone());
        }
        let r = rows_of(full.dims(), self.plan.dp_shard, self.plan.shard_index());
        Ok(full.narrow(0, r.start, r.end - r.start)?.contiguous()?)
    }

    /// Every rank's shard of `key` gathered back into the full tensor (within the shard group).
    pub fn gather_full(&mut self, key: &str, shard: &Tensor) -> Result<Tensor> {
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
        Ok(Tensor::from_vec(flat, full, &self.device)?)
    }

    /// Gather every tensor of a sharded map back to full tensors.
    pub fn gather_map(&mut self, shards: &Tensors) -> Result<Tensors> {
        let mut out = Tensors::new();
        for (k, v) in shards {
            let full = self.gather_full(k, v)?;
            out.insert(k.clone(), full);
        }
        Ok(out)
    }

    /// Reduce one window's local gradients to the global mean (DDP: all-reduce in
    /// `bucket_cap_mb` buckets; FSDP: reduce within the shard group, keep this rank's rows,
    /// all-reduce them across replicate groups), in `reduce_dtype`.
    pub fn reduce_grads(&mut self, grads: Tensors) -> Result<Tensors> {
        let world = self.plan.world.world_size as f32;
        if world == 1.0 {
            return Ok(grads);
        }
        let prec = self.plan.reduce_dtype;
        let whole = Partition::whole(self.plan.world.world_size);
        let mut out = Tensors::new();
        if !self.plan.sharded() {
            let cap = self.plan.ddp.bucket_elems();
            let mut bucket: Vec<(String, Vec<usize>, usize)> = Vec::new();
            let mut buf: Vec<f32> = Vec::new();
            let device = self.device.clone();
            let flush = |bucket: &mut Vec<(String, Vec<usize>, usize)>,
                         buf: &mut Vec<f32>,
                         out: &mut Tensors,
                         coll: &mut Box<dyn Collective>|
             -> Result<()> {
                if bucket.is_empty() {
                    return Ok(());
                }
                coll.all_reduce(buf, ReduceOp::Sum, prec, &whole)?;
                let mut off = 0;
                for (k, shape, n) in bucket.drain(..) {
                    let v: Vec<f32> = buf[off..off + n].iter().map(|x| x / world).collect();
                    out.insert(k, Tensor::from_vec(v, shape, &device)?);
                    off += n;
                }
                buf.clear();
                Ok(())
            };
            for (k, g) in &grads {
                let host = to_host(g)?;
                if !buf.is_empty() && buf.len() + host.len() > cap {
                    flush(&mut bucket, &mut buf, &mut out, &mut self.coll)?;
                }
                bucket.push((k.clone(), g.dims().to_vec(), host.len()));
                buf.extend(host);
            }
            flush(&mut bucket, &mut buf, &mut out, &mut self.coll)?;
            return Ok(out);
        }
        let shard_groups = self.plan.shard_groups();
        let rep_groups = self.plan.replicate_groups();
        for (k, g) in &grads {
            let mut host = to_host(g)?;
            self.coll
                .all_reduce(&mut host, ReduceOp::Sum, prec, &shard_groups)?;
            let shape = g.dims().to_vec();
            let r = rows_of(&shape, self.plan.dp_shard, self.plan.shard_index());
            let row_len: usize = shape.iter().skip(1).product();
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
                rshape[0] = r.end - r.start;
            }
            out.insert(k.clone(), Tensor::from_vec(rows, rshape, &self.device)?);
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
        let mut v = [local_sq / self.plan.dp_replicate as f64];
        self.coll.all_reduce_f64(&mut v, ReduceOp::Sum)?;
        Ok(v[0])
    }
}
