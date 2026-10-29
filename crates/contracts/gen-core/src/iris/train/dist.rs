//! **Distributed training** contract of the Iris trainers (story sc-25689): the backend-neutral
//! half of upstream's `train/parallel.py`, `train/world.py` and the trainer's global-batch /
//! checkpoint-IO rules @ [`super::super::UPSTREAM_CODE_REVISION`] — the strategy and mesh
//! resolution, the world description and its single-world proof, the per-tensor shard layout,
//! the typed failures, and the [`Collective`] trait every transport implements. The host TCP
//! transport itself lives in the sibling `sceneworks-gen-core-dist` crate (`gen_core_dist`,
//! sc-27316), so this contract crate holds no network transport or socket FFI.
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
//! * Candle (CPU / CUDA, incl. Windows where NCCL does not exist): `gen_core_dist::TcpCollective`,
//!   a host-staged transport over `std::net` with bounded socket timeouts — the minimal
//!   all-reduce / all-gather / reduce-scatter / broadcast primitive (the repo had no distributed
//!   layer). Deterministic: the coordinator reduces in rank order and returns the same bytes to
//!   every member.
//! * MLX (Apple Silicon, single- or multi-host): MLX's own distributed layer (`mlx::distributed`,
//!   the `ring` / `jaccl` / `mpi` backends, exposed by the pinned mlx-rs fork's `mlx-sys`) behind
//!   the same [`Collective`] trait, in `mlx-gen-iris`.

use std::time::Duration;

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

/// Round to the nearest bfloat16 (ties to even), returned as f32 — the shared
/// [`crate::numeric::round_bf16`], re-exported at its historical path (sc-27419).
pub use crate::numeric::round_bf16;

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
    /// Bound on every transport wait (without progress) once the job is running.
    pub timeout: Duration,
    /// Bound on the rendezvous (`dist.world.rendezvous_timeout_secs`, sc-25690): accepting the
    /// ranks, a rank's connect retries and its wait for admission. Shorter than [`Self::timeout`]
    /// by default: a rank that never starts is reported long before the collective bound.
    pub rendezvous_timeout: Duration,
    /// How long a collective wait continues once the run's cancel flag is set
    /// (`dist.world.cancel_grace_secs`, sc-25690): long enough for the peers, cancelled too, to
    /// reach the collective cancel agreement at their next step boundary; past it the wait ends
    /// with [`Error::Canceled`]. The rendezvous honours a cancel at once.
    pub cancel_grace: Duration,
    /// The per-job rendezvous secret ([`DIST_TOKEN_ENV`]): every rank presents it in its hello and
    /// the coordinator drops a connection whose token differs. `None` = no secret (any peer that
    /// speaks the protocol and names a free rank joins), so a multi-node `tcp` world without one is
    /// refused by [`assert_single_world`] (sc-27470). The token authenticates the rendezvous only:
    /// the TCP frames themselves are plaintext, so run multi-node jobs on a trusted network.
    pub token: Option<DistToken>,
    /// `dist.world.share_device`: an explicit opt-in to run several ranks on one accelerator when
    /// a rank's own device (`cuda:<local_rank>`) does not exist — same semantics, shared memory
    /// and compute. Off: such a rank is refused ([`DistError::InvalidMesh`]), never silently
    /// folded onto another rank's device.
    pub share_device: bool,
}

/// The environment variable carrying the per-job rendezvous secret ([`World::token`]). It is read
/// from the environment (never from the run config, which is persisted) by [`parse_dist`].
pub const DIST_TOKEN_ENV: &str = "SCENEWORKS_DIST_TOKEN";

/// A rendezvous secret; its `Debug` never prints the value.
#[derive(Clone, PartialEq, Eq)]
pub struct DistToken(pub String);

impl std::fmt::Debug for DistToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DistToken(<redacted>)")
    }
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
            rendezvous_timeout: Duration::from_secs(DEFAULT_RENDEZVOUS_TIMEOUT_SECS),
            cancel_grace: Duration::from_secs(DEFAULT_CANCEL_GRACE_SECS),
            token: None,
            share_device: false,
        }
    }

    pub fn is_main(&self) -> bool {
        self.rank == 0
    }

    /// The rendezvous bound: [`Self::rendezvous_timeout`], never longer than [`Self::timeout`].
    pub fn rendezvous_bound(&self) -> Duration {
        self.rendezvous_timeout.min(self.timeout)
    }
}

/// The environment variable of [`writer_test_delay`].
#[doc(hidden)]
pub const WRITER_TEST_DELAY_ENV: &str = "SCENEWORKS_IRIS_TEST_WRITER_DELAY_MS";

/// Test seam (sc-25690): the multi-process tests make the writing rank's checkpoint / export
/// writes slow (longer than the cancel grace) by setting [`WRITER_TEST_DELAY_ENV`] in the rank
/// process; unset — always, outside those tests — it does nothing.
#[doc(hidden)]
pub fn writer_test_delay() {
    if let Some(ms) = std::env::var(WRITER_TEST_DELAY_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        std::thread::sleep(Duration::from_millis(ms));
    }
}

/// The default rendezvous bound (seconds): torchrun's c10d rendezvous join timeout (600 s).
pub const DEFAULT_RENDEZVOUS_TIMEOUT_SECS: u64 = 600;

/// The default grace a pending cancel gives a collective wait (seconds).
pub const DEFAULT_CANCEL_GRACE_SECS: u64 = 120;

/// The default bound on a collective wait (seconds) — upstream's process-group timeout (5400 s),
/// so rank-0-only work between collectives (previews, consolidated checkpoint / export writes)
/// never reads as a lost rank where upstream would wait. Natively the bound is per transport wait
/// and configurable (`dist.world.timeout_secs`).
pub const DEFAULT_TIMEOUT_SECS: u64 = 5400;

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
    // A multi-node TCP world binds a routable address; without a rendezvous secret any host that
    // reaches it could join as a rank (sc-27470). Loopback single-node worlds stay token-optional.
    if w.num_nodes > 1 && w.transport == "tcp" && w.token.is_none() {
        return Err(DistError::SplitWorld(format!(
            "a {}-node tcp world on MASTER_ADDR={:?} needs a rendezvous secret: set \
             {DIST_TOKEN_ENV} to the same per-job value on every rank",
            w.num_nodes, w.master_addr
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
        Some(v) => {
            return Err(Error::Msg(format!(
            "iris training: dist.ddp.find_unused_parameters must be auto, true or false, got {v}"
        )))
        }
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
            "rendezvous_timeout_secs",
            "cancel_grace_secs",
            "share_device",
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
    let rendezvous = u("rendezvous_timeout_secs", DEFAULT_RENDEZVOUS_TIMEOUT_SECS)?;
    if rendezvous == 0 {
        return Err(Error::Msg(
            "iris training: dist.world.rendezvous_timeout_secs must be >= 1".into(),
        ));
    }
    let cancel_grace = u("cancel_grace_secs", DEFAULT_CANCEL_GRACE_SECS)?;
    if cancel_grace == 0 {
        // A zero grace would end the cancel agreement's own wait at its first idle poll.
        return Err(Error::Msg(
            "iris training: dist.world.cancel_grace_secs must be >= 1".into(),
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
        rendezvous_timeout: Duration::from_secs(rendezvous),
        cancel_grace: Duration::from_secs(cancel_grace),
        token: std::env::var(DIST_TOKEN_ENV)
            .ok()
            .filter(|t| !t.is_empty())
            .map(DistToken),
        share_device: boolean(obj, "share_device", "dist.world")?.unwrap_or(false),
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

    /// Arm or disarm the cancel grace of this collective's waits; returns the previous setting.
    /// Armed (the default of a transport), a wait ends [`Error::Canceled`] once a pending cancel
    /// outlived [`World::cancel_grace`]. Trainers disarm it once the collective cancel agreement
    /// decided to stop, and around checkpoint / export writes: there the slow party is a writing
    /// rank, not a stalled peer, so those waits stay bounded by [`World::timeout`] only (sc-25690).
    fn set_cancel_armed(&mut self, armed: bool) -> bool {
        let _ = armed;
        false
    }
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
            rendezvous_timeout: Duration::from_secs(10),
            cancel_grace: Duration::from_secs(2),
            token: None,
            share_device: false,
        }
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

    /// sc-27470: a multi-node tcp world on a routable address is refused without a rendezvous
    /// token, and admitted with one; single-node and non-tcp transports are unaffected.
    #[test]
    fn multi_node_tcp_worlds_require_a_token() {
        let mut w = world(0, 4, 1);
        w.local_world_size = 2;
        w.num_nodes = 2;
        w.master_addr = "10.0.0.5".into();
        assert!(
            matches!(assert_single_world(&w), Err(DistError::SplitWorld(m)) if m.contains(DIST_TOKEN_ENV))
        );
        w.token = Some(DistToken("secret".into()));
        assert!(assert_single_world(&w).is_ok());
        // MLX transports carry no token through this collective.
        w.token = None;
        w.transport = "mlx_ring".into();
        assert!(assert_single_world(&w).is_ok());
        // A single-node loopback world stays token-optional.
        assert!(assert_single_world(&world(0, 2, 1)).is_ok());
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
    fn the_default_wait_is_upstreams_process_group_timeout() {
        // torch's process-group timeout in upstream (`train/world.py`): a configured constant,
        // read as its printed value (no clock is involved)
        assert_eq!(format!("{:?}", World::single().timeout), "5400s");
        assert_eq!(
            format!("{:?}", DistToken("x".into())),
            "DistToken(<redacted>)"
        );
    }
}
