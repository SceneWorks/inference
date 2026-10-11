//! **Distributed training** contract of the Iris trainers (story sc-25689): the backend-neutral
//! half of upstream's `train/parallel.py`, `train/world.py` and the trainer's global-batch /
//! checkpoint-IO rules @ [`super::super::UPSTREAM_CODE_REVISION`] — the strategy and mesh
//! resolution, the world description and its single-world proof, the per-tensor shard layout,
//! the typed failures, and a host-side collective transport.
//!
//! ## Semantics (what every topology preserves)
//!
//! * **Global batch** = `batch_size × world_size × grad_accum` (refused against
//!   `expected_global_batch` when set — a wrong rank count can never silently change the batch).
//!   Each rank walks its own contiguous slice of the ranged plan; the gradient is the **mean** of
//!   the per-rank gradients (DDP's average), i.e. the gradient of the global-batch mean loss.
//! * **Replication** (`ddp`): every rank holds the full parameters, EMA and optimizer state; the
//!   averaged gradient makes every rank take the identical step.
//! * **Sharding** (`fsdp2`): each tensor's rows (dim 0, `torch.chunk` boundaries — FSDP2's
//!   `Shard(0)`) are partitioned over the `dp_shard` ranks of a shard group; a rank persists only
//!   its rows of the parameters (`full` / `hybrid`), of the averaged gradient, of the optimizer
//!   state and of the EMA (`ShardEMA`, so the shard of the EMA is the EMA of the shard). The full
//!   parameters are all-gathered for the forward (per micro-batch under `reshard_after_forward`,
//!   once per window otherwise; `grad_op` keeps them gathered). The gradient is reduce-scattered
//!   within the shard group and (hybrid) all-reduced across the `dp_replicate` groups. AdamW and
//!   the EMA are elementwise on the shard; a Muon matrix is orthogonalized **whole** (its updated
//!   momentum all-gathered — Dion's DTensor path), so the update equals the unsharded one.
//! * **Clipping** uses the global norm (the squared shard norms all-reduced).
//! * **Non-finite steps** are skipped **collectively**: the guard (the loss under
//!   `lazy_sync = false`, the clipped gradient norm under `lazy_sync = true`) is all-reduced, so
//!   every rank discards the same window (upstream's rank-local pre-backward guard deadlocks a
//!   multi-rank run, which is why `scripts/train.py` demands `lazy_sync` across nodes — kept).
//! * **Failure** is observable: a lost rank, a timeout, a split world, an invalid mesh or a
//!   global-batch mismatch is a typed [`DistError`] (boxed in [`Error::Backend`]), never a hang
//!   (every transport wait is bounded) and never a silently changed batch.
//!
//! ## Native transports
//! * Candle (CPU / CUDA, incl. Windows where NCCL does not exist): [`TcpCollective`], a host-staged
//!   transport over `std::net` with bounded socket timeouts — the minimal all-reduce / all-gather /
//!   reduce-scatter / broadcast primitive (the repo had no distributed layer). Deterministic: the
//!   coordinator reduces in rank order and returns the same bytes to every member.
//! * MLX (Apple Silicon, single- or multi-host): MLX's own distributed layer (`mlx::distributed`,
//!   the `ring` / `jaccl` / `mpi` backends, exposed by the pinned mlx-rs fork's `mlx-sys`) behind
//!   the same [`Collective`] trait, in `mlx-gen-iris`.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use serde_json::{json, Map as JsonMap, Value as JsonValue};

use crate::{Error, Result};

// =============================================================================================
// Typed failures
// =============================================================================================

/// A typed distributed failure. Trainers return it boxed in [`Error::Backend`];
/// [`dist_error`] recovers it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DistError {
    /// The mesh does not tile the world, straddles a node, or names an impossible factorization.
    #[error("iris distributed: invalid mesh: {0}")]
    InvalidMesh(String),
    /// `batch_size × world_size × grad_accum` differs from `expected_global_batch`.
    #[error(
        "iris distributed: global batch {actual} (batch_size {batch_size} x {world_size} ranks x \
         grad_accum {grad_accum}) does not match expected_global_batch={expected}"
    )]
    GlobalBatchMismatch {
        actual: usize,
        expected: usize,
        batch_size: usize,
        world_size: usize,
        grad_accum: usize,
    },
    /// The launcher's world description is inconsistent (`assert_single_world`), or the ranks that
    /// joined disagree about it.
    #[error("iris distributed: split world: {0}")]
    SplitWorld(String),
    /// A peer stopped answering: it exited, its connection closed, or it exceeded the bounded wait.
    #[error("iris distributed: rank {rank} lost during {op} ({detail})")]
    RankLost {
        rank: usize,
        op: String,
        detail: String,
    },
    /// The ranks issued different collective sequences.
    #[error("iris distributed: rank {rank} desynchronized: expected {expected}, got {got}")]
    Desync {
        rank: usize,
        expected: String,
        got: String,
    },
    /// The configured strategy cannot run on this transport / backend.
    #[error("iris distributed: {0}")]
    Unsupported(String),
}

impl DistError {
    pub fn into_error(self) -> Error {
        Error::Backend(Box::new(self))
    }
}

impl From<DistError> for Error {
    fn from(e: DistError) -> Self {
        e.into_error()
    }
}

/// The typed distributed failure inside `e`, if it is one.
pub fn dist_error(e: &Error) -> Option<&DistError> {
    match e {
        Error::Backend(b) => b.downcast_ref::<DistError>(),
        _ => None,
    }
}

// =============================================================================================
// The plan
// =============================================================================================

/// `train.dist.strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Ddp,
    Fsdp2,
}

/// `train.dist.sharding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sharding {
    /// Parameters + gradients + optimizer state sharded.
    Full,
    /// Gradients + optimizer state sharded, parameters kept gathered.
    GradOp,
    /// `Full` inside each shard group, replicated across groups.
    Hybrid,
    /// `GradOp` inside each shard group, replicated across groups.
    HybridGradOp,
}

impl Sharding {
    pub fn hybrid(self) -> bool {
        matches!(self, Self::Hybrid | Self::HybridGradOp)
    }
    /// Whether the parameters persist sharded between steps.
    pub fn shards_params(self) -> bool {
        matches!(self, Self::Full | Self::Hybrid)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::GradOp => "grad_op",
            Self::Hybrid => "hybrid",
            Self::HybridGradOp => "hybrid_grad_op",
        }
    }
}

/// A floating precision named by the dist config (`param_dtype`, `reduce_dtype`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Bf16,
    Fp16,
}

impl Precision {
    fn parse(key: &str, s: &str) -> Result<Self> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            "fp16" => Ok(Self::Fp16),
            other => Err(Error::Msg(format!(
                "iris training: dist.{key} must be one of bf16, fp16, fp32, got {other:?}"
            ))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
    /// Round an f32 to this precision (round to nearest, ties to even) and back.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Self::Fp32 => x,
            Self::Bf16 => round_bf16(x),
            Self::Fp16 => round_f16(x),
        }
    }
}

/// Round to the nearest bfloat16 (ties to even), returned as f32.
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    let r = b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// Round to the nearest IEEE half (ties to even, overflow to ±inf, subnormals kept), as f32.
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    // Quantum of the half grid at |x|: 2^(e − 10) for normals (e ≥ −14), 2^−24 below.
    let e = if a == 0.0 {
        -24
    } else {
        a.log2().floor() as i32
    };
    let q = 2f32.powi((e.max(-14)) - 10);
    let v = a / q;
    let f = v.floor();
    let rem = v - f;
    let n = if rem > 0.5 || (rem == 0.5 && (f as i64) % 2 == 1) {
        f + 1.0
    } else {
        f
    };
    (n * q).copysign(x)
}

/// `train.dist.checkpoint_io`, resolved (never `auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointIo {
    /// One consolidated checkpoint written by global rank 0.
    Rank0,
    /// One consolidated copy per node, written by each node's local rank 0.
    PerNode,
    /// Sharded: every rank writes its own shard; a meta file published after all shards land.
    Dcp,
}

impl CheckpointIo {
    /// `resolve_checkpoint_io`: `auto` = rank0 on one node, per_node beyond.
    pub fn resolve(mode: &str, num_nodes: usize) -> Result<Self> {
        match mode {
            "auto" => Ok(if num_nodes <= 1 {
                Self::Rank0
            } else {
                Self::PerNode
            }),
            "rank0" => Ok(Self::Rank0),
            "per_node" => Ok(Self::PerNode),
            "dcp" => Ok(Self::Dcp),
            other => Err(Error::Msg(format!(
                "iris training: unknown dist.checkpoint_io '{other}' (auto | rank0 | per_node | dcp)"
            ))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rank0 => "rank0",
            Self::PerNode => "per_node",
            Self::Dcp => "dcp",
        }
    }
    /// Whether `world`'s rank writes the consolidated payload (`CheckpointPlan.writes`).
    pub fn writes(self, world: &World) -> bool {
        match self {
            Self::PerNode => world.local_rank == 0,
            Self::Rank0 | Self::Dcp => world.rank == 0,
        }
    }
}

/// `train.dist.ddp`: the replicated strategy's bucketing / reduction knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct DdpKnobs {
    pub find_unused_parameters: String,
    pub gradient_as_bucket_view: bool,
    pub batched_grad_copy: bool,
    pub skip_all_reduce_unused_params: bool,
    /// Gradient bucket size of the all-reduce (0 = torch's 25 MB default).
    pub bucket_cap_mb: usize,
    pub forward_sync_buffers: bool,
}

impl Default for DdpKnobs {
    fn default() -> Self {
        Self {
            find_unused_parameters: "auto".into(),
            gradient_as_bucket_view: false,
            batched_grad_copy: false,
            skip_all_reduce_unused_params: false,
            bucket_cap_mb: 0,
            forward_sync_buffers: true,
        }
    }
}

impl DdpKnobs {
    /// The all-reduce bucket size in f32 elements (`bucket_cap_mb`, 25 MB when 0).
    pub fn bucket_elems(&self) -> usize {
        let mb = if self.bucket_cap_mb == 0 {
            25
        } else {
            self.bucket_cap_mb
        };
        (mb * 1024 * 1024 / 4).max(1)
    }
}

/// How this process was launched: its place in the job (`train/world.py` `World`) plus the native
/// transport's rendezvous.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    pub rank: usize,
    pub local_rank: usize,
    pub world_size: usize,
    pub local_world_size: usize,
    pub node_rank: usize,
    pub num_nodes: usize,
    pub master_addr: String,
    pub master_port: u16,
    /// The operator's declared node count (`--expected-nodes` / `NNODES`), checked against
    /// `num_nodes`.
    pub expected_nodes: Option<usize>,
    /// `tcp` (the host transport) or an MLX backend (`mlx_ring`, `mlx_jaccl`, `mlx_mpi`, `mlx_any`).
    pub transport: String,
    /// MLX ring/jaccl hostfile (JSON) the MLX transport hands to `mlx::distributed::init`.
    pub hostfile: Option<String>,
    /// Bound on every transport wait.
    pub timeout: Duration,
}

impl World {
    /// A single local process.
    pub fn single() -> Self {
        Self {
            rank: 0,
            local_rank: 0,
            world_size: 1,
            local_world_size: 1,
            node_rank: 0,
            num_nodes: 1,
            master_addr: "127.0.0.1".into(),
            master_port: 29500,
            expected_nodes: None,
            transport: "tcp".into(),
            hostfile: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    pub fn is_main(&self) -> bool {
        self.rank == 0
    }
}

/// The default bound on a collective wait (seconds). Upstream's process-group timeout is 5400 s;
/// natively the bound is per transport wait and configurable (`dist.world.timeout_secs`).
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

const LOOPBACK: [&str; 6] = [
    "localhost",
    "localhost.localdomain",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "",
];

/// `assert_single_world`: refuse a world that cannot be one job.
pub fn assert_single_world(w: &World) -> std::result::Result<(), DistError> {
    if w.world_size == 0 || w.local_world_size == 0 || w.num_nodes == 0 {
        return Err(DistError::SplitWorld(
            "world_size, local_world_size and num_nodes must be positive".into(),
        ));
    }
    if w.world_size != w.num_nodes * w.local_world_size {
        return Err(DistError::SplitWorld(format!(
            "WORLD_SIZE={} but GROUP_WORLD_SIZE={} x LOCAL_WORLD_SIZE={} = {}: this rank can only \
             see part of the job",
            w.world_size,
            w.num_nodes,
            w.local_world_size,
            w.num_nodes * w.local_world_size
        )));
    }
    if w.num_nodes > 1 && LOOPBACK.contains(&w.master_addr.as_str()) {
        return Err(DistError::SplitWorld(format!(
            "MASTER_ADDR={:?} is loopback but the job declares {} nodes",
            w.master_addr, w.num_nodes
        )));
    }
    if let Some(n) = w.expected_nodes {
        if n != w.num_nodes {
            return Err(DistError::SplitWorld(format!(
                "node count mismatch: {n} node(s) expected, {} observed",
                w.num_nodes
            )));
        }
    }
    if w.rank >= w.world_size || w.local_rank >= w.local_world_size || w.node_rank >= w.num_nodes {
        return Err(DistError::SplitWorld(format!(
            "rank {}/{} local {}/{} node {}/{} out of range",
            w.rank, w.world_size, w.local_rank, w.local_world_size, w.node_rank, w.num_nodes
        )));
    }
    let expected = w.node_rank * w.local_world_size + w.local_rank;
    if expected != w.rank {
        return Err(DistError::SplitWorld(format!(
            "rank numbering is node-local: RANK={} but GROUP_RANK={} x LOCAL_WORLD_SIZE={} + \
             LOCAL_RANK={} = {expected}",
            w.rank, w.node_rank, w.local_world_size, w.local_rank
        )));
    }
    Ok(())
}

/// The resolved distributed plan of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct DistPlan {
    pub strategy: Strategy,
    pub sharding: Sharding,
    pub dp_replicate: usize,
    pub dp_shard: usize,
    pub param_dtype: Precision,
    pub reduce_dtype: Precision,
    pub reshard_after_forward: bool,
    pub checkpoint_io: CheckpointIo,
    pub ddp: DdpKnobs,
    pub world: World,
}

impl DistPlan {
    pub fn single() -> Self {
        Self {
            strategy: Strategy::Ddp,
            sharding: Sharding::Full,
            dp_replicate: 1,
            dp_shard: 1,
            param_dtype: Precision::Bf16,
            reduce_dtype: Precision::Fp32,
            reshard_after_forward: true,
            checkpoint_io: CheckpointIo::Rank0,
            ddp: DdpKnobs::default(),
            world: World::single(),
        }
    }

    pub fn sharded(&self) -> bool {
        self.strategy == Strategy::Fsdp2
    }

    /// Whether this rank holds only a shard of the persistent state.
    pub fn shard_count(&self) -> usize {
        if self.sharded() {
            self.dp_shard
        } else {
            1
        }
    }

    /// This rank's index within its shard group.
    pub fn shard_index(&self) -> usize {
        if self.sharded() {
            self.world.rank % self.dp_shard
        } else {
            0
        }
    }

    /// The partition of the world into shard groups (`rank / dp_shard`; row-major mesh).
    pub fn shard_groups(&self) -> Partition {
        Partition(
            (0..self.world.world_size)
                .map(|r| if self.sharded() { r / self.dp_shard } else { r })
                .collect(),
        )
    }

    /// The partition of the world into replicate groups (`rank % dp_shard`).
    pub fn replicate_groups(&self) -> Partition {
        Partition(
            (0..self.world.world_size)
                .map(|r| if self.sharded() { r % self.dp_shard } else { 0 })
                .collect(),
        )
    }

    /// One line naming the strategy and mesh (`_dist_summary`).
    pub fn summary(&self) -> String {
        if !self.sharded() {
            return format!("distributed: ddp over {} rank(s)", self.world.world_size);
        }
        format!(
            "distributed: fsdp2 {} mesh dp_replicate={} x dp_shard={} param={} reduce={}",
            self.sharding.as_str(),
            self.dp_replicate,
            self.dp_shard,
            self.param_dtype.as_str(),
            self.reduce_dtype.as_str()
        )
    }

    /// Identity a resume records (the topology does not have to match — the consolidated state
    /// reshards — but it is reported).
    pub fn identity(&self) -> JsonValue {
        json!({
            "strategy": match self.strategy { Strategy::Ddp => "ddp", Strategy::Fsdp2 => "fsdp2" },
            "sharding": self.sharding.as_str(),
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "world_size": self.world.world_size,
            "param_dtype": self.param_dtype.as_str(),
            "reduce_dtype": self.reduce_dtype.as_str(),
            "checkpoint_io": self.checkpoint_io.as_str(),
        })
    }
}

/// `resolve_mesh`: `(dp_replicate, dp_shard)` for `world_size` ranks.
pub fn resolve_mesh(
    sharding: Sharding,
    dp_shard: usize,
    dp_replicate: usize,
    world_size: usize,
    local_world_size: usize,
) -> std::result::Result<(usize, usize), DistError> {
    if !sharding.hybrid() {
        if !(dp_shard == 0 || dp_shard == world_size) || !(dp_replicate == 0 || dp_replicate == 1) {
            return Err(DistError::InvalidMesh(format!(
                "sharding={} shards over every rank, so mesh.dp_shard must be 0 or {world_size} \
                 (got {dp_shard}) and mesh.dp_replicate must be 0 or 1 (got {dp_replicate}); use \
                 sharding=hybrid or hybrid_grad_op to replicate across groups",
                sharding.as_str()
            )));
        }
        return Ok((1, world_size));
    }
    if dp_shard == 0 {
        return Err(DistError::InvalidMesh(
            "hybrid sharding needs mesh.dp_shard (the ranks one copy is sharded over, normally \
             the GPU count of one node)"
                .into(),
        ));
    }
    let rep = if dp_replicate == 0 {
        world_size / dp_shard
    } else {
        dp_replicate
    };
    if rep * dp_shard != world_size {
        return Err(DistError::InvalidMesh(format!(
            "mesh does not tile the world: dp_replicate={rep} x dp_shard={dp_shard} = {}, but \
             world_size={world_size}",
            rep * dp_shard
        )));
    }
    if local_world_size > 1
        && !dp_shard.is_multiple_of(local_world_size)
        && !local_world_size.is_multiple_of(dp_shard)
    {
        return Err(DistError::InvalidMesh(format!(
            "mesh.dp_shard={dp_shard} straddles a node boundary at {local_world_size} ranks per \
             node"
        )));
    }
    Ok((rep, dp_shard))
}

/// `_effective_batch`: the global batch, checked against the launcher's pin when set.
pub fn effective_batch(
    batch_size: usize,
    world_size: usize,
    grad_accum: usize,
    expected: Option<usize>,
) -> std::result::Result<usize, DistError> {
    let actual = batch_size * world_size * grad_accum;
    match expected {
        Some(e) if e != actual => Err(DistError::GlobalBatchMismatch {
            actual,
            expected: e,
            batch_size,
            world_size,
            grad_accum,
        }),
        _ => Ok(actual),
    }
}

/// Read the `dist` object of the Iris options bag. Unknown keys are refused. `log_block_grad_norms`
/// and `resume_optimizer` take part in upstream's `DistConfig.validate`.
pub fn parse_dist(
    raw: Option<&JsonValue>,
    log_block_grad_norms: bool,
    resume_optimizer_full: bool,
) -> Result<DistPlan> {
    let empty = JsonMap::new();
    let obj = match raw {
        None | Some(JsonValue::Null) => &empty,
        Some(JsonValue::Object(m)) => m,
        Some(other) => {
            return Err(Error::Msg(format!(
                "iris training: dist must be an object, got {other}"
            )))
        }
    };
    const KEYS: [&str; 9] = [
        "strategy",
        "sharding",
        "mesh",
        "param_dtype",
        "reduce_dtype",
        "reshard_after_forward",
        "checkpoint_io",
        "ddp",
        "world",
    ];
    refuse_unknown("dist", obj, &KEYS)?;
    let s = |k: &str, d: &str| -> Result<String> {
        match obj.get(k) {
            None | Some(JsonValue::Null) => Ok(d.into()),
            Some(JsonValue::String(v)) => Ok(v.clone()),
            Some(v) => Err(Error::Msg(format!(
                "iris training: dist.{k} must be a string, got {v}"
            ))),
        }
    };
    let strategy = match s("strategy", "ddp")?.as_str() {
        "ddp" => Strategy::Ddp,
        "fsdp2" => Strategy::Fsdp2,
        other => {
            return Err(Error::Msg(format!(
                "iris training: dist.strategy must be ddp or fsdp2, got {other:?}"
            )))
        }
    };
    let sharding = match s("sharding", "full")?.as_str() {
        "full" => Sharding::Full,
        "grad_op" => Sharding::GradOp,
        "hybrid" => Sharding::Hybrid,
        "hybrid_grad_op" => Sharding::HybridGradOp,
        other => {
            return Err(Error::Msg(format!(
                "iris training: dist.sharding must be full, grad_op, hybrid or hybrid_grad_op, \
                 got {other:?}"
            )))
        }
    };
    let mesh = sub_object(obj, "mesh")?;
    refuse_unknown("dist.mesh", &mesh, &["dp_shard", "dp_replicate"])?;
    let dp_shard = uint(&mesh, "dp_shard", "dist.mesh")?.unwrap_or(0) as usize;
    let dp_replicate = uint(&mesh, "dp_replicate", "dist.mesh")?.unwrap_or(0) as usize;
    let param_dtype = Precision::parse("param_dtype", &s("param_dtype", "bf16")?)?;
    let reduce_dtype = Precision::parse("reduce_dtype", &s("reduce_dtype", "fp32")?)?;
    if reduce_dtype == Precision::Fp16 {
        return Err(Error::Msg(
            "iris training: dist.reduce_dtype must be fp32 or bf16".into(),
        ));
    }
    let reshard_after_forward = boolean(obj, "reshard_after_forward", "dist")?.unwrap_or(true);
    let checkpoint_io_name = s("checkpoint_io", "auto")?;
    let ddp_obj = sub_object(obj, "ddp")?;
    refuse_unknown(
        "dist.ddp",
        &ddp_obj,
        &[
            "find_unused_parameters",
            "gradient_as_bucket_view",
            "batched_grad_copy",
            "skip_all_reduce_unused_params",
            "bucket_cap_mb",
            "forward_sync_buffers",
        ],
    )?;
    let fup = match ddp_obj.get("find_unused_parameters") {
        None | Some(JsonValue::Null) => "auto".to_string(),
        Some(JsonValue::String(v)) if matches!(v.as_str(), "auto" | "true" | "false") => v.clone(),
        Some(JsonValue::Bool(b)) => b.to_string(),
        Some(v) => return Err(Error::Msg(format!(
            "iris training: dist.ddp.find_unused_parameters must be auto, true or false, got {v}"
        ))),
    };
    let ddp = DdpKnobs {
        find_unused_parameters: fup,
        gradient_as_bucket_view: boolean(&ddp_obj, "gradient_as_bucket_view", "dist.ddp")?
            .unwrap_or(false),
        batched_grad_copy: boolean(&ddp_obj, "batched_grad_copy", "dist.ddp")?.unwrap_or(false),
        skip_all_reduce_unused_params: boolean(
            &ddp_obj,
            "skip_all_reduce_unused_params",
            "dist.ddp",
        )?
        .unwrap_or(false),
        bucket_cap_mb: uint(&ddp_obj, "bucket_cap_mb", "dist.ddp")?.unwrap_or(0) as usize,
        forward_sync_buffers: boolean(&ddp_obj, "forward_sync_buffers", "dist.ddp")?
            .unwrap_or(true),
    };
    let world = parse_world(obj.get("world"))?;
    assert_single_world(&world).map_err(DistError::into_error)?;

    // DistConfig.validate
    if strategy == Strategy::Ddp {
        if sharding != Sharding::Full {
            return Err(DistError::InvalidMesh(format!(
                "dist.sharding applies to fsdp2 only; strategy=ddp shards nothing (got {})",
                sharding.as_str()
            ))
            .into_error());
        }
        if dp_shard != 0 || dp_replicate != 0 {
            return Err(
                DistError::InvalidMesh("dist.mesh applies to fsdp2 only".into()).into_error(),
            );
        }
        if checkpoint_io_name == "dcp" {
            return Err(DistError::InvalidMesh(
                "dist.checkpoint_io=dcp requires a sharded strategy".into(),
            )
            .into_error());
        }
    } else {
        if log_block_grad_norms {
            return Err(Error::Msg(
                "iris training: dist.strategy=fsdp2 does not support log_block_grad_norms \
                 (per-parameter grads are local shards)"
                    .into(),
            ));
        }
        if !resume_optimizer_full {
            return Err(Error::Msg(
                "iris training: dist.strategy=fsdp2 requires resume_optimizer=full".into(),
            ));
        }
    }
    let (dp_replicate, dp_shard) = if strategy == Strategy::Fsdp2 {
        resolve_mesh(
            sharding,
            dp_shard,
            dp_replicate,
            world.world_size,
            world.local_world_size,
        )
        .map_err(DistError::into_error)?
    } else {
        (world.world_size, 1)
    };
    let checkpoint_io = CheckpointIo::resolve(&checkpoint_io_name, world.num_nodes)?;
    Ok(DistPlan {
        strategy,
        sharding,
        dp_replicate,
        dp_shard,
        param_dtype,
        reduce_dtype,
        reshard_after_forward,
        checkpoint_io,
        ddp,
        world,
    })
}

fn parse_world(raw: Option<&JsonValue>) -> Result<World> {
    let empty = JsonMap::new();
    let obj = match raw {
        None | Some(JsonValue::Null) => &empty,
        Some(JsonValue::Object(m)) => m,
        Some(other) => {
            return Err(Error::Msg(format!(
                "iris training: dist.world must be an object, got {other}"
            )))
        }
    };
    refuse_unknown(
        "dist.world",
        obj,
        &[
            "rank",
            "local_rank",
            "world_size",
            "local_world_size",
            "node_rank",
            "num_nodes",
            "master_addr",
            "master_port",
            "expected_nodes",
            "transport",
            "hostfile",
            "timeout_secs",
        ],
    )?;
    let u = |k: &str, d: u64| -> Result<u64> { Ok(uint(obj, k, "dist.world")?.unwrap_or(d)) };
    let world_size = u("world_size", 1)? as usize;
    let local_world_size = u("local_world_size", world_size as u64)? as usize;
    let rank = u("rank", 0)? as usize;
    let num_nodes = u(
        "num_nodes",
        (world_size / local_world_size.max(1)).max(1) as u64,
    )? as usize;
    let node_rank = u("node_rank", (rank / local_world_size.max(1)) as u64)? as usize;
    let local_rank = u("local_rank", (rank % local_world_size.max(1)) as u64)? as usize;
    let str_of = |k: &str| -> Result<Option<String>> {
        match obj.get(k) {
            None | Some(JsonValue::Null) => Ok(None),
            Some(JsonValue::String(v)) => Ok(Some(v.clone())),
            Some(v) => Err(Error::Msg(format!(
                "iris training: dist.world.{k} must be a string, got {v}"
            ))),
        }
    };
    let transport = str_of("transport")?.unwrap_or_else(|| "tcp".into());
    if !matches!(
        transport.as_str(),
        "tcp" | "mlx_ring" | "mlx_jaccl" | "mlx_mpi" | "mlx_any"
    ) {
        return Err(Error::Msg(format!(
            "iris training: dist.world.transport {transport:?} (tcp | mlx_ring | mlx_jaccl | \
             mlx_mpi | mlx_any)"
        )));
    }
    let port = u("master_port", 29500)?;
    if port == 0 || port > u16::MAX as u64 {
        return Err(Error::Msg(
            "iris training: dist.world.master_port must be 1..=65535".into(),
        ));
    }
    let timeout = u("timeout_secs", DEFAULT_TIMEOUT_SECS)?;
    if timeout == 0 {
        return Err(Error::Msg(
            "iris training: dist.world.timeout_secs must be >= 1".into(),
        ));
    }
    Ok(World {
        rank,
        local_rank,
        world_size,
        local_world_size,
        node_rank,
        num_nodes,
        master_addr: str_of("master_addr")?.unwrap_or_else(|| "127.0.0.1".into()),
        master_port: port as u16,
        expected_nodes: uint(obj, "expected_nodes", "dist.world")?.map(|v| v as usize),
        transport,
        hostfile: str_of("hostfile")?,
        timeout: Duration::from_secs(timeout),
    })
}

fn refuse_unknown(path: &str, obj: &JsonMap<String, JsonValue>, known: &[&str]) -> Result<()> {
    for k in obj.keys() {
        if !known.contains(&k.as_str()) {
            return Err(Error::Unsupported(format!(
                "iris training: {path}.{k} is not an Iris training option (known: {})",
                known.join(", ")
            )));
        }
    }
    Ok(())
}

fn sub_object(obj: &JsonMap<String, JsonValue>, key: &str) -> Result<JsonMap<String, JsonValue>> {
    match obj.get(key) {
        None | Some(JsonValue::Null) => Ok(JsonMap::new()),
        Some(JsonValue::Object(m)) => Ok(m.clone()),
        Some(v) => Err(Error::Msg(format!(
            "iris training: {key} must be an object, got {v}"
        ))),
    }
}

fn uint(obj: &JsonMap<String, JsonValue>, key: &str, path: &str) -> Result<Option<u64>> {
    match obj.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            Error::Msg(format!(
                "iris training: {path}.{key} must be a non-negative integer, got {v}"
            ))
        }),
    }
}

fn boolean(obj: &JsonMap<String, JsonValue>, key: &str, path: &str) -> Result<Option<bool>> {
    match obj.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(Error::Msg(format!(
            "iris training: {path}.{key} must be a bool, got {v}"
        ))),
    }
}

// =============================================================================================
// Shard layout (FSDP2 `Shard(0)`, `torch.chunk` boundaries)
// =============================================================================================

/// The rows of dim 0 shard `index` of `n` owns (`torch.chunk`: chunks of `ceil(rows / n)`, the
/// trailing ones shorter or empty).
pub fn chunk_rows(rows: usize, n: usize, index: usize) -> std::ops::Range<usize> {
    let chunk = rows.div_ceil(n.max(1));
    let start = (index * chunk).min(rows);
    start..(start + chunk).min(rows)
}

/// The flat element range of shard `index` of a row-major tensor of `shape`.
pub fn shard_elems(shape: &[usize], n: usize, index: usize) -> std::ops::Range<usize> {
    let rows = shape.first().copied().unwrap_or(1);
    let row_len: usize = shape.iter().skip(1).product();
    let r = chunk_rows(rows, n, index);
    r.start * row_len..r.end * row_len
}

// =============================================================================================
// Collectives
// =============================================================================================

/// A partition of the world into groups: `0[rank]` is the group id of `rank`. Collectives run
/// independently inside each group; every rank calls every collective (lockstep).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition(pub Vec<usize>);

impl Partition {
    pub fn whole(world_size: usize) -> Self {
        Self(vec![0; world_size])
    }
    /// The ranks sharing `rank`'s group, ascending.
    pub fn members(&self, rank: usize) -> Vec<usize> {
        let g = self.0[rank];
        (0..self.0.len()).filter(|&r| self.0[r] == g).collect()
    }
}

/// A reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
}

/// The collectives the native distributed trainers need. Every rank calls every method in the same
/// order; a peer that stops answering within the world's timeout is a typed
/// [`DistError::RankLost`].
pub trait Collective {
    fn world(&self) -> &World;

    /// Sum (or min/max) `data` elementwise across each group of `groups`; every member receives
    /// the group's result. With `precision = Bf16`, every input and partial sum is rounded to
    /// bfloat16 in rank order (the reduce precision of `dist.reduce_dtype`).
    fn all_reduce(
        &mut self,
        data: &mut [f32],
        op: ReduceOp,
        precision: Precision,
        groups: &Partition,
    ) -> Result<()>;

    /// Gather each member's `data` (lengths may differ) to every member of its group, in rank order.
    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> Result<Vec<Vec<f32>>>;

    /// Broadcast `root`'s bytes to every rank.
    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> Result<()>;

    /// Every rank reaches this point.
    fn barrier(&mut self, tag: &str) -> Result<()>;

    /// `f64` reduction across the whole world (counters, flags).
    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> Result<()>;
}

/// The single-process collective (world size 1): every operation is the identity.
pub struct LocalCollective {
    world: World,
}

impl LocalCollective {
    pub fn new() -> Self {
        Self {
            world: World::single(),
        }
    }
}

impl Default for LocalCollective {
    fn default() -> Self {
        Self::new()
    }
}

impl Collective for LocalCollective {
    fn world(&self) -> &World {
        &self.world
    }
    fn all_reduce(
        &mut self,
        data: &mut [f32],
        _: ReduceOp,
        p: Precision,
        _: &Partition,
    ) -> Result<()> {
        if p != Precision::Fp32 {
            for v in data.iter_mut() {
                *v = p.round(*v);
            }
        }
        Ok(())
    }
    fn all_gather(&mut self, data: &[f32], _: &Partition) -> Result<Vec<Vec<f32>>> {
        Ok(vec![data.to_vec()])
    }
    fn broadcast(&mut self, _: &mut Vec<u8>, _: usize) -> Result<()> {
        Ok(())
    }
    fn barrier(&mut self, _: &str) -> Result<()> {
        Ok(())
    }
    fn all_reduce_f64(&mut self, _: &mut [f64], _: ReduceOp) -> Result<()> {
        Ok(())
    }
}

/// Reduce `inputs` (one per member, rank order) with `op` at `precision`.
pub fn reduce_in_rank_order(inputs: &[&[f32]], op: ReduceOp, precision: Precision) -> Vec<f32> {
    let n = inputs[0].len();
    let r = |v: f32| precision.round(v);
    let mut acc: Vec<f32> = inputs[0].iter().map(|&v| r(v)).collect();
    for inp in &inputs[1..] {
        for i in 0..n {
            let v = r(inp[i]);
            acc[i] = match op {
                ReduceOp::Sum => r(acc[i] + v),
                ReduceOp::Min => acc[i].min(v),
                ReduceOp::Max => acc[i].max(v),
            };
        }
    }
    acc
}

// ---- the host TCP transport ------------------------------------------------------------------

const MAGIC: u64 = 0x4952_4953_4453_5431; // "IRISDST1"

/// Message kinds of the TCP protocol.
const OP_REDUCE: u8 = 1;
const OP_GATHER: u8 = 2;
const OP_BCAST: u8 = 3;
const OP_BARRIER: u8 = 4;
const OP_REDUCE64: u8 = 5;

/// A host-staged collective over TCP: rank 0 coordinates (every rank holds one connection to it);
/// each collective is "every rank sends its contribution, the coordinator reduces per group in rank
/// order and answers every rank". Every socket wait is bounded by the world's timeout; an EOF, an
/// error or an expired wait is a [`DistError::RankLost`] naming the peer and the operation.
#[derive(Debug)]
pub struct TcpCollective {
    world: World,
    /// Coordinator: one stream per peer rank (index = rank, `None` at 0). Peer: the stream to 0.
    peers: Vec<Option<TcpStream>>,
    seq: u64,
}

fn io_lost(rank: usize, op: &str, e: impl std::fmt::Display) -> Error {
    DistError::RankLost {
        rank,
        op: op.to_string(),
        detail: e.to_string(),
    }
    .into_error()
}

/// A transport I/O failure as [`DistError::RankLost`], naming an expired bounded wait as such.
fn stream_lost(s: &TcpStream, peer: usize, op: &str, e: std::io::Error) -> Error {
    use std::io::ErrorKind::{TimedOut, WouldBlock};
    if matches!(e.kind(), WouldBlock | TimedOut) {
        let bound = s.read_timeout().ok().flatten().unwrap_or_default();
        return io_lost(
            peer,
            op,
            format!("no answer within the {}s transport timeout", bound.as_secs()),
        );
    }
    io_lost(peer, op, e)
}

fn write_all(s: &mut TcpStream, buf: &[u8], peer: usize, op: &str) -> Result<()> {
    s.write_all(buf).map_err(|e| stream_lost(s, peer, op, e))
}

fn read_exact(s: &mut TcpStream, buf: &mut [u8], peer: usize, op: &str) -> Result<()> {
    s.read_exact(buf).map_err(|e| stream_lost(s, peer, op, e))
}

fn send_frame(
    s: &mut TcpStream,
    kind: u8,
    seq: u64,
    payload: &[u8],
    peer: usize,
    op: &str,
) -> Result<()> {
    let mut head = Vec::with_capacity(17);
    head.push(kind);
    head.extend_from_slice(&seq.to_le_bytes());
    head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    write_all(s, &head, peer, op)?;
    write_all(s, payload, peer, op)?;
    s.flush().map_err(|e| io_lost(peer, op, e))
}

fn recv_frame(s: &mut TcpStream, kind: u8, seq: u64, peer: usize, op: &str) -> Result<Vec<u8>> {
    let mut head = [0u8; 17];
    read_exact(s, &mut head, peer, op)?;
    let got_kind = head[0];
    let got_seq = u64::from_le_bytes(head[1..9].try_into().expect("8"));
    let len = u64::from_le_bytes(head[9..17].try_into().expect("8")) as usize;
    if got_kind != kind || got_seq != seq {
        return Err(DistError::Desync {
            rank: peer,
            expected: format!("{op} #{seq}"),
            got: format!("kind {got_kind} #{got_seq}"),
        }
        .into_error());
    }
    let mut payload = vec![0u8; len];
    read_exact(s, &mut payload, peer, op)?;
    Ok(payload)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect()
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8")))
        .collect()
}

fn op_code(op: ReduceOp) -> u8 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Min => 1,
        ReduceOp::Max => 2,
    }
}

fn prec_code(p: Precision) -> u8 {
    match p {
        Precision::Fp32 => 0,
        Precision::Bf16 => 1,
        Precision::Fp16 => 2,
    }
}

impl TcpCollective {
    /// Join the job described by `world`: rank 0 listens on `master_addr:master_port` and accepts
    /// `world_size − 1` peers; every other rank connects (retrying until the timeout). Each peer
    /// announces its world description and the coordinator refuses a world that disagrees.
    pub fn connect(world: &World) -> Result<Self> {
        assert_single_world(world).map_err(DistError::into_error)?;
        let deadline = Instant::now() + world.timeout;
        let mut peers: Vec<Option<TcpStream>> = (0..world.world_size).map(|_| None).collect();
        let hello = json!({
            "magic": MAGIC,
            "rank": world.rank,
            "world_size": world.world_size,
            "local_world_size": world.local_world_size,
            "num_nodes": world.num_nodes,
            "node_rank": world.node_rank,
        })
        .to_string();
        if world.world_size == 1 {
            return Ok(Self {
                world: world.clone(),
                peers,
                seq: 0,
            });
        }
        if world.rank == 0 {
            let bind = format!("{}:{}", bind_host(&world.master_addr), world.master_port);
            let listener = TcpListener::bind(&bind)
                .map_err(|e| io_lost(0, "rendezvous bind", format!("{bind}: {e}")))?;
            listener
                .set_nonblocking(true)
                .map_err(|e| io_lost(0, "rendezvous", e))?;
            let mut joined = 1usize;
            while joined < world.world_size {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false)
                            .map_err(|e| io_lost(0, "rendezvous", e))?;
                        configure(&s, world.timeout)?;
                        let payload = recv_frame(&mut s, 0, 0, usize::MAX, "rendezvous")?;
                        let v: JsonValue = serde_json::from_slice(&payload).map_err(|e| {
                            DistError::SplitWorld(format!("bad hello: {e}")).into_error()
                        })?;
                        let r = v["rank"].as_u64().unwrap_or(u64::MAX) as usize;
                        for (k, mine) in [
                            ("world_size", world.world_size),
                            ("local_world_size", world.local_world_size),
                            ("num_nodes", world.num_nodes),
                        ] {
                            if v[k].as_u64() != Some(mine as u64) {
                                return Err(DistError::SplitWorld(format!(
                                    "rank {r} declares {k}={} but rank 0 declares {mine}",
                                    v[k]
                                ))
                                .into_error());
                            }
                        }
                        if v["magic"].as_u64() != Some(MAGIC) {
                            return Err(DistError::SplitWorld(
                                "a peer speaks another protocol".into(),
                            )
                            .into_error());
                        }
                        if r == 0 || r >= world.world_size || peers[r].is_some() {
                            return Err(DistError::SplitWorld(format!(
                                "duplicate or out-of-range rank {r} joined"
                            ))
                            .into_error());
                        }
                        peers[r] = Some(s);
                        joined += 1;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() > deadline {
                            let missing: Vec<usize> = (1..world.world_size)
                                .filter(|&r| peers[r].is_none())
                                .collect();
                            return Err(DistError::RankLost {
                                rank: missing[0],
                                op: "rendezvous".into(),
                                detail: format!(
                                    "ranks {missing:?} never joined within {:?}",
                                    world.timeout
                                ),
                            }
                            .into_error());
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) => return Err(io_lost(0, "rendezvous accept", e)),
                }
            }
            // Release every peer only once the whole world joined.
            for r in 1..world.world_size {
                let s = peers[r].as_mut().expect("joined");
                send_frame(s, 0, 0, b"ok", r, "rendezvous")?;
            }
        } else {
            let addr = resolve(&world.master_addr, world.master_port)?;
            let mut stream = loop {
                match TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
                    Ok(s) => break s,
                    Err(e) => {
                        if Instant::now() > deadline {
                            return Err(io_lost(0, "rendezvous connect", e));
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            };
            configure(&stream, world.timeout)?;
            send_frame(&mut stream, 0, 0, hello.as_bytes(), 0, "rendezvous")?;
            let ok = recv_frame(&mut stream, 0, 0, 0, "rendezvous")?;
            if ok != b"ok" {
                return Err(DistError::SplitWorld("rendezvous refused".into()).into_error());
            }
            peers[0] = Some(stream);
        }
        Ok(Self {
            world: world.clone(),
            peers,
            seq: 0,
        })
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Coordinator side of one exchange: gather every rank's payload, compute every rank's answer
    /// with `answer(rank, payloads)`, send it.
    fn coordinate(
        &mut self,
        kind: u8,
        mine: Vec<u8>,
        op: &str,
        answer: impl Fn(usize, &[Vec<u8>]) -> Vec<u8>,
    ) -> Result<Vec<u8>> {
        let seq = self.next_seq();
        if self.world.world_size == 1 {
            return Ok(answer(0, &[mine]));
        }
        if self.world.rank != 0 {
            let s = self.peers[0].as_mut().expect("connected");
            send_frame(s, kind, seq, &mine, 0, op)?;
            return recv_frame(s, kind, seq, 0, op);
        }
        let mut payloads = vec![mine];
        for r in 1..self.world.world_size {
            let s = self.peers[r].as_mut().expect("connected");
            payloads.push(recv_frame(s, kind, seq, r, op)?);
        }
        for r in 1..self.world.world_size {
            let out = answer(r, &payloads);
            let s = self.peers[r].as_mut().expect("connected");
            send_frame(s, kind, seq, &out, r, op)?;
        }
        Ok(answer(0, &payloads))
    }
}

fn bind_host(addr: &str) -> String {
    if addr.contains(':') && !addr.starts_with('[') {
        format!("[{addr}]")
    } else {
        addr.to_string()
    }
}

fn resolve(host: &str, port: u16) -> Result<SocketAddr> {
    (bind_host(host).as_str(), port)
        .to_socket_addrs()
        .map_err(|e| io_lost(0, "rendezvous resolve", e))?
        .next()
        .ok_or_else(|| io_lost(0, "rendezvous resolve", format!("{host} has no address")))
}

fn configure(s: &TcpStream, timeout: Duration) -> Result<()> {
    s.set_nodelay(true)
        .map_err(|e| io_lost(0, "configure", e))?;
    s.set_read_timeout(Some(timeout))
        .map_err(|e| io_lost(0, "configure", e))?;
    s.set_write_timeout(Some(timeout))
        .map_err(|e| io_lost(0, "configure", e))?;
    Ok(())
}

impl Collective for TcpCollective {
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
        let mut msg = vec![op_code(op), prec_code(precision)];
        msg.extend(f32_bytes(data));
        let parts = groups.clone();
        let out = self.coordinate(OP_REDUCE, msg, "all_reduce", move |rank, payloads| {
            let members = parts.members(rank);
            let vals: Vec<Vec<f32>> = members
                .iter()
                .map(|&m| bytes_f32(&payloads[m][2..]))
                .collect();
            let refs: Vec<&[f32]> = vals.iter().map(Vec::as_slice).collect();
            let op = match payloads[0][0] {
                1 => ReduceOp::Min,
                2 => ReduceOp::Max,
                _ => ReduceOp::Sum,
            };
            let prec = match payloads[0][1] {
                1 => Precision::Bf16,
                2 => Precision::Fp16,
                _ => Precision::Fp32,
            };
            f32_bytes(&reduce_in_rank_order(&refs, op, prec))
        })?;
        let v = bytes_f32(&out);
        if v.len() != data.len() {
            return Err(DistError::Desync {
                rank: self.world.rank,
                expected: format!("{} elements", data.len()),
                got: format!("{} elements", v.len()),
            }
            .into_error());
        }
        data.copy_from_slice(&v);
        Ok(())
    }

    fn all_gather(&mut self, data: &[f32], groups: &Partition) -> Result<Vec<Vec<f32>>> {
        let parts = groups.clone();
        let out = self.coordinate(
            OP_GATHER,
            f32_bytes(data),
            "all_gather",
            move |rank, payloads| {
                let mut buf = Vec::new();
                for m in parts.members(rank) {
                    buf.extend((payloads[m].len() as u64 / 4).to_le_bytes());
                    buf.extend_from_slice(&payloads[m]);
                }
                buf
            },
        )?;
        let mut res = Vec::new();
        let mut i = 0;
        while i < out.len() {
            let n = u64::from_le_bytes(out[i..i + 8].try_into().expect("8")) as usize;
            i += 8;
            res.push(bytes_f32(&out[i..i + 4 * n]));
            i += 4 * n;
        }
        Ok(res)
    }

    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> Result<()> {
        let mine = if self.world.rank == root {
            data.clone()
        } else {
            Vec::new()
        };
        let out = self.coordinate(OP_BCAST, mine, "broadcast", move |_, payloads| {
            payloads[root].clone()
        })?;
        *data = out;
        Ok(())
    }

    fn barrier(&mut self, tag: &str) -> Result<()> {
        let op = format!("barrier {tag}");
        self.coordinate(OP_BARRIER, tag.as_bytes().to_vec(), &op, |_, _| Vec::new())
            .map(|_| ())
    }

    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> Result<()> {
        let mut msg = vec![op_code(op)];
        msg.extend(f64_bytes(data));
        let out = self.coordinate(OP_REDUCE64, msg, "all_reduce_f64", |_, payloads| {
            let vals: Vec<Vec<f64>> = payloads.iter().map(|p| bytes_f64(&p[1..])).collect();
            let mut acc = vals[0].clone();
            for v in &vals[1..] {
                for (a, b) in acc.iter_mut().zip(v) {
                    *a = match payloads[0][0] {
                        1 => a.min(*b),
                        2 => a.max(*b),
                        _ => *a + b,
                    };
                }
            }
            f64_bytes(&acc)
        })?;
        data.copy_from_slice(&bytes_f64(&out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(rank: usize, n: usize, port: u16) -> World {
        World {
            rank,
            local_rank: rank,
            world_size: n,
            local_world_size: n,
            node_rank: 0,
            num_nodes: 1,
            master_addr: "127.0.0.1".into(),
            master_port: port,
            expected_nodes: None,
            transport: "tcp".into(),
            hostfile: None,
            timeout: Duration::from_secs(10),
        }
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn meshes_resolve_and_refuse_like_upstream() {
        assert_eq!(resolve_mesh(Sharding::Full, 0, 0, 8, 8).unwrap(), (1, 8));
        assert_eq!(resolve_mesh(Sharding::Hybrid, 4, 0, 8, 4).unwrap(), (2, 4));
        assert!(matches!(
            resolve_mesh(Sharding::Hybrid, 3, 0, 8, 8),
            Err(DistError::InvalidMesh(_))
        ));
        assert!(matches!(
            resolve_mesh(Sharding::Full, 4, 0, 8, 8),
            Err(DistError::InvalidMesh(_))
        ));
        assert!(matches!(
            resolve_mesh(Sharding::Hybrid, 0, 0, 8, 8),
            Err(DistError::InvalidMesh(_))
        ));
        assert!(matches!(
            resolve_mesh(Sharding::Hybrid, 3, 0, 6, 4),
            Err(DistError::InvalidMesh(_))
        ));
        assert_eq!(effective_batch(16, 64, 1, Some(1024)).unwrap(), 1024);
        assert!(matches!(
            effective_batch(16, 1, 1, Some(1024)),
            Err(DistError::GlobalBatchMismatch { actual: 16, .. })
        ));
    }

    #[test]
    fn worlds_must_be_one_job() {
        let mut w = world(0, 2, 1);
        assert!(assert_single_world(&w).is_ok());
        w.num_nodes = 2;
        assert!(assert_single_world(&w).is_err());
        let mut w = world(0, 4, 1);
        w.local_world_size = 2;
        w.num_nodes = 2;
        assert!(
            matches!(assert_single_world(&w), Err(DistError::SplitWorld(m)) if m.contains("loopback"))
        );
        let mut w = world(1, 2, 1);
        w.expected_nodes = Some(2);
        assert!(assert_single_world(&w).is_err());
    }

    #[test]
    fn dist_options_parse_and_validate() {
        let p = parse_dist(None, false, true).unwrap();
        assert_eq!(p.world.world_size, 1);
        let v = json!({"strategy": "fsdp2", "sharding": "hybrid", "mesh": {"dp_shard": 2},
                       "world": {"rank": 1, "world_size": 4}});
        let p = parse_dist(Some(&v), false, true).unwrap();
        assert_eq!((p.dp_replicate, p.dp_shard, p.shard_index()), (2, 2, 1));
        assert_eq!(p.shard_groups().members(1), vec![0, 1]);
        assert_eq!(p.replicate_groups().members(1), vec![1, 3]);
        let bad = json!({"strategy": "ddp", "sharding": "hybrid"});
        let e = parse_dist(Some(&bad), false, true).unwrap_err();
        assert!(matches!(dist_error(&e), Some(DistError::InvalidMesh(_))));
        let bad = json!({"strategy": "fsdp2", "sharding": "hybrid", "mesh": {"dp_shard": 3},
                         "world": {"world_size": 4}});
        let e = parse_dist(Some(&bad), false, true).unwrap_err();
        assert!(matches!(dist_error(&e), Some(DistError::InvalidMesh(_))));
        assert!(parse_dist(Some(&json!({"bogus": 1})), false, true).is_err());
        assert!(parse_dist(Some(&json!({"strategy": "fsdp2"})), true, true).is_err());
    }

    #[test]
    fn chunks_follow_torch_chunk() {
        assert_eq!(chunk_rows(10, 4, 0), 0..3);
        assert_eq!(chunk_rows(10, 4, 3), 9..10);
        assert_eq!(chunk_rows(2, 4, 3), 2..2);
        assert_eq!(shard_elems(&[5, 3], 2, 1), 9..15);
    }

    #[test]
    fn rounding_matches_the_formats() {
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0); // tie → even
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(round_f16(1.0 + 1.0 / 4096.0), 1.0);
        assert_eq!(round_f16(65504.0), 65504.0);
        assert!(round_f16(70000.0).is_infinite());
        assert_eq!(round_f16(1e-8), 0.0);
    }

    #[test]
    fn tcp_collectives_reduce_gather_and_broadcast_across_processes_threads() {
        let port = free_port();
        let handles: Vec<_> = (0..3)
            .map(|r| {
                std::thread::spawn(move || {
                    let mut c = TcpCollective::connect(&world(r, 3, port)).unwrap();
                    let mut v = vec![r as f32 + 1.0, 10.0 * r as f32];
                    c.all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(3))
                        .unwrap();
                    // groups {0, 2} and {1}
                    let groups = Partition(vec![0, 1, 0]);
                    let mut g = vec![r as f32];
                    c.all_reduce(&mut g, ReduceOp::Max, Precision::Fp32, &groups)
                        .unwrap();
                    let gathered = c.all_gather(&vec![r as f32; r + 1], &groups).unwrap();
                    let mut b = if r == 2 {
                        b"hello".to_vec()
                    } else {
                        Vec::new()
                    };
                    c.broadcast(&mut b, 2).unwrap();
                    let mut f = [r as f64];
                    c.all_reduce_f64(&mut f, ReduceOp::Min).unwrap();
                    c.barrier("end").unwrap();
                    (v, g, gathered, b, f[0])
                })
            })
            .collect();
        let out: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (r, (v, g, gathered, b, f)) in out.into_iter().enumerate() {
            assert_eq!(v, vec![6.0, 30.0]);
            assert_eq!(g, vec![if r == 1 { 1.0 } else { 2.0 }]);
            if r == 1 {
                assert_eq!(gathered, vec![vec![1.0, 1.0]]);
            } else {
                assert_eq!(gathered, vec![vec![0.0], vec![2.0, 2.0, 2.0]]);
            }
            assert_eq!(b, b"hello");
            assert_eq!(f, 0.0);
        }
    }

    #[test]
    fn a_lost_rank_is_a_typed_bounded_failure() {
        let port = free_port();
        let mut w0 = world(0, 2, port);
        w0.timeout = Duration::from_secs(2);
        let mut w1 = w0.clone();
        w1.rank = 1;
        w1.local_rank = 1;
        let peer = std::thread::spawn(move || {
            let c = TcpCollective::connect(&w1).unwrap();
            drop(c); // rank 1 dies right after the rendezvous
        });
        let mut c = TcpCollective::connect(&w0).unwrap();
        peer.join().unwrap();
        let mut v = vec![1.0f32];
        let e = c
            .all_reduce(&mut v, ReduceOp::Sum, Precision::Fp32, &Partition::whole(2))
            .unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { rank: 1, .. })),
            "{e}"
        );
        // a rank that never joins is bounded too
        let port = free_port();
        let mut w = world(0, 2, port);
        w.timeout = Duration::from_millis(300);
        let e = TcpCollective::connect(&w).unwrap_err();
        assert!(
            matches!(dist_error(&e), Some(DistError::RankLost { op, .. }) if op == "rendezvous")
        );
    }
}
