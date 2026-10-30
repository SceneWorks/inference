//! **Direct, centered-curve AdaLN** (sc-26684) — the candle sibling of the pruned checkpoint the
//! WanGP reference (`47b06ff920c16b4b7b36de068367a5de0e0968e9`) produces with
//! `models/minimax_h3/prune_checkpoint.py` and consumes in `models/minimax_h3/transformer.py`.
//!
//! # What the pruner stored, and why none of it may be rederived
//!
//! The reference replaces the timestep MLP (`time_embedder.linear_{1,2}`) with one shared F32
//! `adaln_t_table` (`[grid, rank]`, the pinned artifact's `1025 × 8`) and shrinks every
//! `adaln_proj.linear` to `[out, rank]`. It **centers** the factorization (`_factor_curve` SVDs the
//! mean-subtracted curve), folds `W · mean` into each bias (`_compress_projection`), and stores
//! `W · Vh.T` as each projection. Those three operand groups are inseparable: interpolating the
//! table and multiplying by a projection that was not fitted against that table's centering is a
//! plausible-looking model with nothing behind it. This module therefore:
//!
//! * never reconstructs a basis, never substitutes stock weights, and never builds the curve from
//!   the timestep MLP — [`gen_core_h3::pruned_adaln::PrunedAdaLn::inspect`] refuses every incomplete
//!   or inconsistent topology *before* any tensor is executed;
//! * hashes every operand at its **source precision** (F32 table, BF16 reduced weights, F16
//!   folded biases in the frozen artifact), then retains one losslessly widened F32 compute
//!   operand set matching the pinned runtime's `_lock_dtype`, without modifying source bytes;
//! * applies **no SiLU** to the interpolated coordinates: the table already stores the MLP's
//!   post-activation output, and the pruned `AdalnProj` is built with `apply_silu=False`;
//! * interpolates with the pinned addressing — `t` clamped to `[0, 1]` (t is already `1 − σ`, and
//!   there is deliberately **no ×1000 rescale**), `position = t · (grid − 1)`, `lower` clamped to
//!   `grid − 2` so `t = 1` selects the last interval with `fraction = 1` — reproduced elementwise
//!   from [`gen_core_h3::pruned_adaln::interpolation_position`].
//!
//! # Identity is content, not metadata
//!
//! [`CurveAdaLn::identity`] is the [`gen_core_h3::pruned_adaln::PruningIdentity`] hashed over **every**
//! operand's stored values at load — the table *and* every reduced projection *and* every folded
//! bias, in contract order, each with its stored dtype in the digest. Two checkpoints with the same
//! shapes but different bases therefore hash differently, and a cache or compatibility check keyed
//! on the identity cannot serve one checkpoint's modulation to the other. A table alone cannot
//! identify the model: [`gen_core_h3::pruned_adaln::PruningIdentityBuilder::finish`] refuses a partial
//! hash.
//!
//! # Memory
//!
//! The pruned checkpoint has no 26.02 GB to evict: its single widened compute operand set is
//! ~175 MB at the shipped rank, and stays resident. [`AdaLnCache::precompute_and_evict`]'s
//! discipline is not re-implemented here — [`CurveAdaLn::precompute`] builds the same
//! [`AdaLnCache`] from the schedule's own distinct timesteps and reports **zero** released bytes,
//! because there is nothing to release. The cache is keyed on the schedule *and* the pruning
//! identity ([`AdaLnCache::is_current_for_pruning`]), so a cross-request reuse seam cannot serve
//! one basis's modulation table to another.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype as MetaDtype};
use candle_gen::{CandleError, Result, Weights};
use gen_core_h3::{
    pruned_adaln,
    pruned_adaln::{PrunedAdaLn, PruningIdentity, TABLE_KEY},
};

use crate::dit::adaln::{AdaLnCache, TimestepSchedule};
use crate::dit::block::AdaLnModulation;
use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::heads::NormOutModulation;

/// One reduced projection with a single resident F32 compute weight and folded bias.
/// Source dtype metadata and the pruning identity are captured before lossless widening;
/// repeated evaluations neither recast the operands nor retain a second source tensor copy.
#[derive(Debug, Clone)]
pub struct CurveProjection {
    weight: Tensor,
    bias: Tensor,
    source_dtype: DType,
    out_features: usize,
    rank: usize,
    /// The block geometry the `[num_timesteps · MODALITY_NUM, 6·hidden]` reshape needs.
    hidden_size: usize,
    /// The pruning identity of the checkpoint this projection was taken from — bound at
    /// assembly, zero-cost ([`PruningIdentity`] is `Copy`), so a projection from one basis can
    /// never be silently projected under another's cache label.
    identity: PruningIdentity,
}

impl CurveProjection {
    /// The pruning identity of the checkpoint this projection came from.
    pub fn pruning_identity(&self) -> PruningIdentity {
        self.identity
    }

    /// Assemble views of the already-validated, once-widened compute operands.
    fn from_operands(
        weight: Tensor,
        bias: Tensor,
        out_features: usize,
        rank: usize,
        hidden_size: usize,
        binding: (PruningIdentity, DType),
        what: &str,
    ) -> Result<Self> {
        if weight.dims() != [out_features, rank] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve {what}.weight: expected [{out_features}, {rank}], got {:?}",
                weight.dims()
            )));
        }
        if bias.dims() != [out_features] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve {what}.bias: expected [{out_features}], got {:?}",
                bias.dims()
            )));
        }
        Ok(Self {
            weight,
            bias,
            out_features,
            rank,
            hidden_size,
            identity: binding.0,
            source_dtype: binding.1,
        })
    }

    /// Original checkpoint weight dtype, bound into the identity before compute widening.
    pub fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    /// Actual resident compute dtype; source precision is reported separately.
    pub fn compute_dtype(&self) -> DType {
        self.weight.dtype()
    }

    /// Actual resident compute bytes, not the narrower checkpoint storage bytes.
    pub fn nbytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
            + self.bias.elem_count() * self.bias.dtype().size_in_bytes()
    }

    /// Project interpolated coordinates into the full-width output, **without any activation**.
    ///
    /// `coords` and resident operands are F32, matching the pinned runtime's `_lock_dtype`.
    /// Source weights and folded biases were widened exactly once during validated assembly.
    pub fn forward(&self, coords: &Tensor) -> Result<Tensor> {
        let s = coords.dims();
        if s.len() != 2 || s[1] != self.rank || coords.dtype() != DType::F32 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve projection: expected coordinates [num_timesteps, {rank}], \
                 got {s:?}",
                rank = self.rank
            )));
        }
        Ok(coords
            .matmul(&self.weight.t()?)?
            .broadcast_add(&self.bias)?)
    }

    /// [`Self::forward`] reshaped into the six modulation tables, exactly as
    /// `crate::dit::block::AdaLnProjection::forward` lays them out: `view(-1, 6·hidden)` **before**
    /// the chunk, so modality becomes a row axis (`timestep_index · MODALITY_NUM + token_tag`).
    ///
    /// No SiLU — the curve replaced the timestep MLP, and the pruned reference block is built with
    /// `apply_silu=False`.
    pub fn modulation(&self, coords: &Tensor) -> Result<AdaLnModulation> {
        let steps = coords.dims().first().copied().unwrap_or(0);
        let projected = self.forward(coords)?;
        if projected.dims() != [steps, self.out_features] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve projection: expected [{steps}, {}], got {:?}",
                self.out_features,
                projected.dims()
            )));
        }
        let rows = steps * MODALITY_NUM;
        let table = projected.reshape((rows, MODULATION_PARAMS * self.hidden_size))?;
        let part = |i: usize| -> Result<Tensor> {
            Ok(table
                .narrow(1, i * self.hidden_size, self.hidden_size)?
                .contiguous()?)
        };
        Ok(AdaLnModulation {
            shift_msa: part(0)?,
            scale_msa: part(1)?,
            gate_msa: part(2)?,
            shift_mlp: part(3)?,
            scale_mlp: part(4)?,
            gate_mlp: part(5)?,
        })
    }
}

/// The direct, centered-curve AdaLN of one loaded pruned checkpoint: the shared table, the final
/// `norm_out.linear` operands, and the content identity every consumer compatibility check keys on.
///
/// Built only through [`CurveAdaLn::inspect`] (validation first) + [`CurveAdaLn::assemble`]; the
/// per-block projections live on their [`crate::dit::block::DitBlock`]s as [`CurveProjection`]s,
/// handed out by [`Self::block_projection`].
#[derive(Debug, Clone)]
pub struct CurveAdaLn {
    topology: PrunedAdaLn,
    identity: PruningIdentity,
    grid: usize,
    rank: usize,
    hidden_size: usize,
    device: Device,
    /// The stored F32 table, `[grid, rank]`, kept at source precision.
    table: Tensor,
    /// One F32 compute tensor per canonical operand, plus original storage dtype metadata.
    /// Cloned projections share these allocations; no resident source tensor set is retained.
    operands: HashMap<String, (Tensor, DType)>,
}

impl CurveAdaLn {
    /// Canonical names this checkpoint variant consumes — the table, every block's reduced pair,
    /// and the final reduced pair.
    pub fn names(topology: &PrunedAdaLn) -> Vec<String> {
        topology.operands().iter().map(|o| o.name.clone()).collect()
    }

    /// Inspect a weight map's canonical tensor headers and, when it carries the direct curve,
    /// validate the complete reduced topology.
    ///
    /// Returns `Ok(None)` for an unpruned checkpoint (the caller proceeds down the stock path),
    /// [`CandleError`] for an incomplete or inconsistent one. The lookup reads what is already
    /// resident in the map; a projection whose packed (`.scales`) sibling exists is **not** a
    /// reduced tensor — it is a full-width packed base, and peeking its raw code shape here would
    /// misread it, so the lookup reports it as absent and leaves its validation to
    /// `crate::quant::lin`.
    pub fn inspect(w: &Weights, cfg: &MiniMaxH3DitConfig) -> Result<Option<PrunedAdaLn>> {
        let lookup = |name: &str| -> Option<(MetaDtype, Vec<usize>)> {
            if name.ends_with(".weight") {
                let base = name.trim_end_matches(".weight");
                if w.contains(&format!("{base}.scales"))
                    || w.contains(&format!("{base}.comfy_quant"))
                {
                    return None;
                }
            }
            let tensor = w.require(name).ok()?;
            Some((meta_dtype(tensor.dtype())?, tensor.dims().to_vec()))
        };
        let topology = pruned_adaln::PrunedAdaLn::inspect(
            cfg.hidden_size,
            cfg.num_layers,
            cfg.time_embed_dim,
            lookup,
        )
        .map_err(CandleError::Msg)?;
        Ok(topology)
    }

    /// Assemble from validated topology and the operand tensors at their **stored** dtype.
    ///
    /// `operands` must carry exactly the canonical names [`Self::names`] declares — a caller that
    /// dropped one produces a typed error here, not a partially built model. The identity is hashed
    /// over every operand's stored values before anything is returned, so an assembly that cannot
    /// state what it is never becomes a model.
    pub fn assemble(
        topology: PrunedAdaLn,
        operands: HashMap<String, Tensor>,
        cfg: &MiniMaxH3DitConfig,
        device: &Device,
    ) -> Result<Self> {
        if topology.hidden_size != cfg.hidden_size || topology.num_layers != cfg.num_layers {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve: topology is hidden {} × {} layers but the config is {} × {}",
                topology.hidden_size, topology.num_layers, cfg.hidden_size, cfg.num_layers
            )));
        }
        let mut builder = topology.identity_builder();
        let mut compute_operands = HashMap::with_capacity(operands.len());
        for operand in topology.operands() {
            let tensor = operands.get(&operand.name).ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 dit curve: missing pruned operand {}",
                    operand.name
                ))
            })?;
            let got = meta_dtype(tensor.dtype()).ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 dit curve: {} is {:?}, not a float curve operand",
                    operand.name,
                    tensor.dtype()
                ))
            })?;
            if got != operand.dtype || tensor.dims() != operand.shape.as_slice() {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit curve: {} is {:?} {:?}, inspected as {:?} {:?}",
                    operand.name,
                    got,
                    tensor.dims(),
                    operand.dtype,
                    operand.shape
                )));
            }
            let compute = tensor.to_dtype(DType::F32)?;
            let values = compute.flatten_all()?.to_vec1::<f32>()?;
            builder
                .operand_values(&operand.name, values)
                .map_err(CandleError::Msg)?;
            compute_operands.insert(operand.name.clone(), (compute, tensor.dtype()));
        }
        let identity = builder.finish().map_err(CandleError::Msg)?;

        let table = compute_operands
            .get(TABLE_KEY)
            .ok_or_else(|| {
                CandleError::Msg(
                    "minimax-h3 dit curve: the validated topology lost its table".into(),
                )
            })?
            .0
            .clone();
        Ok(Self {
            grid: topology.grid,
            rank: topology.rank,
            hidden_size: cfg.hidden_size,
            topology,
            identity,
            table,
            operands: compute_operands,
            device: device.clone(),
        })
    }

    /// The content identity — table, every reduced projection and every folded bias hashed at
    /// their stored dtypes.
    pub fn identity(&self) -> PruningIdentity {
        self.identity
    }

    /// The validated topology.
    pub fn topology(&self) -> &PrunedAdaLn {
        &self.topology
    }

    /// The curve's input width — the factorization rank, 8 in the frozen artifact.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The table's timestep grid — 1025 in the frozen artifact.
    pub fn grid(&self) -> usize {
        self.grid
    }

    /// Actual F32 resident compute bytes, including the unchanged F32 coordinate table.
    pub fn nbytes(&self) -> usize {
        self.operands
            .values()
            .map(|(tensor, _)| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }

    /// The table tensor, at its stored F32 precision.
    pub fn table(&self) -> &Tensor {
        &self.table
    }

    /// This block's reduced projection, taken from the validated operand set.
    pub fn block_projection(&self, layer: usize) -> Result<CurveProjection> {
        let prefix = format!("transformer_blocks.{layer}.adaln_proj.linear");
        let out = self.hidden_size * 18;
        let (weight, source_dtype) = self.operand(&format!("{prefix}.weight"))?;
        let (bias, _) = self.operand(&format!("{prefix}.bias"))?;
        CurveProjection::from_operands(
            weight,
            bias,
            out,
            self.rank,
            self.hidden_size,
            (self.identity, source_dtype),
            &prefix,
        )
    }

    /// The final `norm_out.linear` reduced projection.
    pub fn norm_out_projection(&self) -> Result<CurveProjection> {
        let prefix = "norm_out.linear";
        let out = self.hidden_size * 2;
        let (weight, source_dtype) = self.operand(&format!("{prefix}.weight"))?;
        let (bias, _) = self.operand(&format!("{prefix}.bias"))?;
        CurveProjection::from_operands(
            weight,
            bias,
            out,
            self.rank,
            self.hidden_size,
            (self.identity, source_dtype),
            prefix,
        )
    }

    fn operand(&self, name: &str) -> Result<(Tensor, DType)> {
        self.operands.get(name).cloned().ok_or_else(|| {
            CandleError::Msg(format!("minimax-h3 dit curve: missing operand {name}"))
        })
    }

    /// Interpolate the table at `timesteps` — `[num_timesteps, rank]` F32.
    ///
    /// The pinned addressing, per row: `position = clamp(t, 0, 1) · (grid − 1)`,
    /// `lower = min(floor(position), grid − 2)`, `fraction = position − lower` — the tiny host
    /// index/fraction arrays come from
    /// [`interpolation_position`](gen_core_h3::pruned_adaln::interpolation_position) —
    /// and then the table gather and the lerp run **on the device**: `table.index_select` at
    /// `lower` and `lower + 1`, combined as `start + weight · (end − start)` in f32, torch's own
    /// `torch.lerp` formula. No SiLU, no ×1000. The table is never copied to the host to do this —
    /// the resident F32 tensor is the only copy the model holds.
    pub fn coordinates(&self, timesteps: &[f32], device: &Device) -> Result<Tensor> {
        if timesteps.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 dit curve: cannot interpolate an empty timestep list".into(),
            ));
        }
        let mut lower: Vec<u32> = Vec::with_capacity(timesteps.len());
        let mut upper: Vec<u32> = Vec::with_capacity(timesteps.len());
        let mut fraction: Vec<f32> = Vec::with_capacity(timesteps.len());
        for &t in timesteps {
            let (lo, frac) =
                pruned_adaln::interpolation_position(t, self.grid).map_err(CandleError::Msg)?;
            lower.push(lo as u32);
            upper.push(lo as u32 + 1);
            fraction.push(frac);
        }
        let lower = Tensor::from_vec(lower, (timesteps.len(),), device)?;
        let upper = Tensor::from_vec(upper, (timesteps.len(),), device)?;
        let weight = Tensor::from_vec(fraction, (timesteps.len(), 1), device)?;
        let start = self.table.index_select(&lower, 0)?;
        let end = self.table.index_select(&upper, 0)?;
        Ok(start.broadcast_add(&weight.broadcast_mul(&end.broadcast_sub(&start)?)?)?)
    }

    /// The final `norm_out` modulation from the reduced projection: `shift` is the FIRST half,
    /// exactly as `crate::dit::heads::AdaLayerNormOut::modulation` lays it out — and, like it, with
    /// **no** activation before the projection (the curve carries the MLP's output already).
    pub fn norm_out_modulation(&self, coords: &Tensor) -> Result<NormOutModulation> {
        let projection = self.norm_out_projection()?;
        let hidden_size = self.hidden_size;
        let projected = projection.forward(coords)?;
        let steps = coords.dims().first().copied().unwrap_or(0);
        if projected.dims() != [steps, 2 * hidden_size] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve norm_out: expected [{steps}, {}], got {:?}",
                2 * hidden_size,
                projected.dims()
            )));
        }
        Ok(NormOutModulation {
            shift: projected.narrow(1, 0, hidden_size)?.contiguous()?,
            scale: projected
                .narrow(1, hidden_size, hidden_size)?
                .contiguous()?,
        })
    }

    /// Project the whole schedule's modulation for every block, plus the final `norm_out` table,
    /// from the schedule's own distinct timesteps.
    ///
    /// The resulting [`AdaLnCache`] carries this checkpoint's [`PruningIdentity`], so a held cache
    /// can only be reused for a request whose schedule **and** basis match
    /// ([`AdaLnCache::is_current_for_pruning`]). There is nothing to evict — the curve operands are
    /// ~175 MB of F32 compute buffers — so callers report zero released bytes and keep them.
    pub fn precompute(
        &self,
        blocks: &[crate::dit::block::DitBlock],
        schedule: &TimestepSchedule,
    ) -> Result<(AdaLnCache, NormOutModulation)> {
        if blocks.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 dit curve: the block stack is empty".into(),
            ));
        }
        if blocks.len() != self.topology.num_layers {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve: {} blocks against a {}-layer pruned topology — an \
                 incomplete block stack must not precompute a partial cache",
                blocks.len(),
                self.topology.num_layers
            )));
        }
        let coords = self.coordinates(schedule.distinct_timesteps(), &self.device)?;
        let mut layers = Vec::with_capacity(blocks.len());
        for (layer, block) in blocks.iter().enumerate() {
            let projected = block.curve_pruning_identity().ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 dit curve: layer {layer} carries no reduced curve projection — \
                     a stock block cannot be projected under a curve identity"
                ))
            })?;
            if projected != self.identity {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit curve: layer {layer}'s reduced projection belongs to a \
                     different pruning basis (content identity mismatch) — refusing to project it \
                     under this curve's identity"
                )));
            }
            layers.push(block.curve_modulation(&coords).map_err(|e| {
                CandleError::Msg(format!("minimax-h3 dit curve: layer {layer}: {e}"))
            })?);
        }
        let norm_out = self.norm_out_modulation(&coords)?;
        let cache = AdaLnCache::assemble(schedule.clone(), layers, Some(self.identity))?;
        Ok((cache, norm_out))
    }

    /// One-pruned-checkpoint-per-map assertion helper used by the loader: whether the map carries
    /// the direct curve's table.
    pub fn is_present(w: &Weights) -> bool {
        w.contains(TABLE_KEY)
    }
}

/// Map a candle dtype onto the contract's storage dtype; `None` for anything a curve operand may
/// not be (integer codes, quantized bases).
pub fn meta_dtype(dtype: DType) -> Option<MetaDtype> {
    match dtype {
        DType::F16 => Some(MetaDtype::F16),
        DType::BF16 => Some(MetaDtype::BF16),
        DType::F32 => Some(MetaDtype::F32),
        _ => None,
    }
}

/// The header half of a shard scan: every tensor's stored dtype and shape, keyed by name.
///
/// Read from the safetensors headers only — no tensor is materialized — so inspection at load
/// validates a 20 GB checkpoint's topology from kilobytes of headers, exactly what
/// [`PrunedAdaLn::inspect`]'s "validation allocates headers only" contract asks of a backend.
pub struct CurveScan {
    headers: HashMap<String, (MetaDtype, Vec<usize>)>,
}

impl CurveScan {
    /// Scan every shard's header, refusing a name that appears in more than one of them (the same
    /// mis-sharded-checkpoint policy `Weights::from_files` applies to the full load).
    pub fn read(files: &[impl AsRef<std::path::Path>]) -> Result<Self> {
        let mut headers = HashMap::new();
        for path in files {
            for header in safetensors_path_tensor_headers(path.as_ref())
                .map_err(|error| CandleError::Msg(error.to_string()))?
            {
                let key = header.name;
                if headers
                    .insert(key.clone(), (header.dtype, header.shape))
                    .is_some()
                {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {key:?} across the DiT shards: a checkpoint's \
                         tensors must each live in exactly one .safetensors shard"
                    )));
                }
            }
        }
        Ok(Self { headers })
    }

    /// Whether the scanned shards carry the direct curve's table.
    pub fn has_curve(&self) -> bool {
        self.headers.contains_key(TABLE_KEY)
    }

    /// The canonical lookup [`PrunedAdaLn::inspect`] consumes, over the scanned headers.
    pub fn lookup(&self, name: &str) -> Option<(MetaDtype, Vec<usize>)> {
        let (dtype, shape) = self.headers.get(name)?;
        if !matches!(dtype, MetaDtype::F16 | MetaDtype::BF16 | MetaDtype::F32) {
            return None;
        }
        Some((*dtype, shape.clone()))
    }
}
