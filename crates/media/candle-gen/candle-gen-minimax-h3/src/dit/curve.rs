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
//!   the timestep MLP — [`gen_core::pruned_adaln::PrunedAdaLn::inspect`] refuses every incomplete
//!   or inconsistent topology *before* any tensor is executed;
//! * stores every operand at its **source precision** (table F32, reduced weights BF16, folded
//!   biases F16 in the frozen artifact) and widens to F32 only inside the projection — exactly the
//!   pinned runtime's `_lock_dtype = torch.float32`, without converting the checkpoint's bytes;
//! * applies **no SiLU** to the interpolated coordinates: the table already stores the MLP's
//!   post-activation output, and the pruned `AdalnProj` is built with `apply_silu=False`;
//! * interpolates with the pinned addressing — `t` clamped to `[0, 1]` (t is already `1 − σ`, and
//!   there is deliberately **no ×1000 rescale**), `position = t · (grid − 1)`, `lower` clamped to
//!   `grid − 2` so `t = 1` selects the last interval with `fraction = 1` — reproduced elementwise
//!   from [`gen_core::pruned_adaln::interpolation_position`].
//!
//! # Identity is content, not metadata
//!
//! [`CurveAdaLn::identity`] is the [`gen_core::pruned_adaln::PruningIdentity`] hashed over **every**
//! operand's stored values at load — the table *and* every reduced projection *and* every folded
//! bias, in contract order, each with its stored dtype in the digest. Two checkpoints with the same
//! shapes but different bases therefore hash differently, and a cache or compatibility check keyed
//! on the identity cannot serve one checkpoint's modulation to the other. A table alone cannot
//! identify the model: [`gen_core::pruned_adaln::PruningIdentityBuilder::finish`] refuses a partial
//! hash.
//!
//! # Memory
//!
//! The pruned checkpoint has no 26.02 GB to evict: all 103 curve operands together are ~90 MB at
//! the shipped rank, and the projections stay resident. [`AdaLnCache::precompute_and_evict`]'s
//! discipline is not re-implemented here — [`CurveAdaLn::precompute`] builds the same
//! [`AdaLnCache`] from the schedule's own distinct timesteps and reports **zero** released bytes,
//! because there is nothing to release. The cache is keyed on the schedule *and* the pruning
//! identity ([`AdaLnCache::is_current_for_pruning`]), so a cross-request reuse seam cannot serve
//! one basis's modulation table to another.

use std::collections::HashMap;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::pruned_adaln::{
    self, PrunedAdaLn, PruningIdentity, TABLE_KEY,
};
use candle_gen::gen_core::weightsmeta::Dtype as MetaDtype;
use candle_gen::{CandleError, Result, Weights};

use crate::dit::adaln::{AdaLnCache, TimestepSchedule};
use crate::dit::block::AdaLnModulation;
use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::heads::NormOutModulation;

/// One reduced `adaln_proj.linear` (or `norm_out.linear`): `[out_features, rank]` weight plus its
/// `[out_features]` folded bias, at their **stored** dtype.
///
/// The widening to F32 happens inside [`Self::forward`], per evaluation — never at load, never
/// persisted — so the checkpoint's own bytes remain the single source of truth and the identity
/// hash reads them back unchanged.
#[derive(Debug, Clone)]
pub struct CurveProjection {
    weight: Tensor,
    bias: Tensor,
    out_features: usize,
    rank: usize,
    /// The block geometry the `[num_timesteps · MODALITY_NUM, 6·hidden]` reshape needs.
    hidden_size: usize,
}

impl CurveProjection {
    /// Assemble from the two stored operand tensors, checking shapes.
    fn from_operands(
        weight: Tensor,
        bias: Tensor,
        out_features: usize,
        rank: usize,
        hidden_size: usize,
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
        })
    }

    /// The dtype the stored weight carries — BF16 in the frozen artifact.
    pub fn dtype(&self) -> DType {
        self.weight.dtype()
    }

    /// Device bytes the stored operands hold.
    pub fn nbytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
            + self.bias.elem_count() * self.bias.dtype().size_in_bytes()
    }

    /// Project interpolated coordinates into the full-width output, **without any activation**.
    ///
    /// `coords` is `[num_timesteps, rank]` F32 ([`CurveAdaLn::coordinates`]); the result is
    /// `[num_timesteps, out_features]` F32. Both operands are widened to F32 for the multiply —
    /// the pinned runtime's `_lock_dtype` — and the folded bias is added broadcast.
    pub fn forward(&self, coords: &Tensor) -> Result<Tensor> {
        let s = coords.dims();
        if s.len() != 2 || s[1] != self.rank {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit curve projection: expected coordinates [num_timesteps, {rank}], \
                 got {s:?}",
                rank = self.rank
            )));
        }
        let weight = self.weight.to_dtype(DType::F32)?;
        let bias = self.bias.to_dtype(DType::F32)?;
        coords
            .to_dtype(DType::F32)?
            .matmul(&weight.t()?)?
            .broadcast_add(&bias)
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
    /// The table's F32 values on the host — interpolation is a pure host read of these (the pinned
    /// runtime lerps in f32), and 8200 f32 do not justify a device round-trip per call.
    table_values: Vec<f32>,
    /// Every canonical operand at its stored dtype, keyed by canonical name. The per-block
    /// projections are handed to their blocks from here; `norm_out.linear` stays resident here.
    operands: HashMap<String, Tensor>,
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
                if w.contains(&format!("{base}.scales")) || w.contains(&format!("{base}.comfy_quant"))
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
        for operand in topology.operands() {
            let tensor = operands.get(&operand.name).ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 dit curve: missing pruned operand {}",
                    operand.name
                ))
            })?;
            let got = meta_dtype(tensor.dtype())
                .ok_or_else(|| {
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
            let values = tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            builder
                .operand_values(&operand.name, values)
                .map_err(CandleError::Msg)?;
        }
        let identity = builder.finish().map_err(CandleError::Msg)?;

        let table = operands
            .get(TABLE_KEY)
            .ok_or_else(|| {
                CandleError::Msg("minimax-h3 dit curve: the validated topology lost its table".into())
            })?
            .clone();
        let table_values = table.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            grid: topology.grid,
            rank: topology.rank,
            hidden_size: cfg.hidden_size,
            topology,
            identity,
            table,
            table_values,
            operands,
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

    /// Device bytes every stored curve operand holds.
    pub fn nbytes(&self) -> usize {
        self.operands
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
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
        let weight = self.operand(&format!("{prefix}.weight"))?;
        let bias = self.operand(&format!("{prefix}.bias"))?;
        CurveProjection::from_operands(weight, bias, out, self.rank, self.hidden_size, &prefix)
    }

    /// The final `norm_out.linear` reduced projection.
    pub fn norm_out_projection(&self) -> Result<CurveProjection> {
        let prefix = "norm_out.linear";
        let out = self.hidden_size * 2;
        let weight = self.operand(&format!("{prefix}.weight"))?;
        let bias = self.operand(&format!("{prefix}.bias"))?;
        CurveProjection::from_operands(weight, bias, out, self.rank, self.hidden_size, prefix)
    }

    fn operand(&self, name: &str) -> Result<Tensor> {
        self.operands
            .get(name)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("minimax-h3 dit curve: missing operand {name}")))
    }

    /// Interpolate the table at `timesteps` — `[num_timesteps, rank]` F32.
    ///
    /// The pinned addressing, per row: `position = clamp(t, 0, 1) · (grid − 1)`,
    /// `lower = min(floor(position), grid − 2)`, `fraction = position − lower`, and
    /// `lerp(table[lower], table[lower + 1], fraction)` in f32 — `start + weight · (end − start)`,
    /// torch's own formula. No SiLU, no ×1000.
    pub fn coordinates(&self, timesteps: &[f32], device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(timesteps.len() * self.rank);
        for &t in timesteps {
            let (lower, fraction) =
                pruned_adaln::interpolation_position(t, self.grid).map_err(CandleError::Msg)?;
            let start = &self.table_values[lower * self.rank..(lower + 1) * self.rank];
            let end = &self.table_values[(lower + 1) * self.rank..(lower + 2) * self.rank];
            for j in 0..self.rank {
                out.push(start[j] + fraction * (end[j] - start[j]));
            }
        }
        Ok(Tensor::from_vec(out, (timesteps.len(), self.rank), device)?)
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
    /// ~90 MB resident — so callers report zero released bytes and keep the projections.
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
        let coords = self.coordinates(schedule.distinct_timesteps(), &self.device)?;
        let mut layers = Vec::with_capacity(blocks.len());
        for (layer, block) in blocks.iter().enumerate() {
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
    headers: HashMap<String, (safetensors::Dtype, Vec<usize>)>,
}

impl CurveScan {
    /// Scan every shard's header, refusing a name that appears in more than one of them (the same
    /// mis-sharded-checkpoint policy `Weights::from_files` applies to the full load).
    pub fn read(files: &[impl AsRef<std::path::Path>]) -> Result<Self> {
        let mut headers = HashMap::new();
        for path in files {
            // SAFETY: read-only, process-owned checkpoint, mapped for header metadata only — the
            // standard candle weight-loading invariant.
            let st = unsafe { MmapedSafetensors::new(path.as_ref())? };
            for (key, view) in st.tensors() {
                if headers
                    .insert(key.to_string(), (view.dtype(), view.shape().to_vec()))
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
        let dtype = match dtype {
            safetensors::Dtype::F16 => MetaDtype::F16,
            safetensors::Dtype::BF16 => MetaDtype::BF16,
            safetensors::Dtype::F32 => MetaDtype::F32,
            _ => return None,
        };
        Some((dtype, shape.clone()))
    }

    /// Load every canonical operand at its **stored** dtype, from whichever shard holds it.
    ///
    /// The pruned artifact's AdaLN operands are dense by construction
    /// (`adalnQuantDescriptorCount = 0`); a packed (`.comfy_quant`) sibling under a curve name is a
    /// topology the contract refuses, and the missing-name error below says so rather than
    /// silently loading garbage codes.
    pub fn load_operands(&self, files: &[impl AsRef<std::path::Path>], device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut stores = Vec::with_capacity(files.len());
        for path in files {
            // SAFETY: read-only, process-owned weight file, mapped only for this load — the
            // standard candle weight-loading invariant.
            let st = unsafe { MmapedSafetensors::new(path.as_ref())? };
            stores.push(st);
        }
        let mut out = HashMap::new();
        for (key, (dtype, _)) in &self.headers {
            if !is_curve_key(key) {
                continue;
            }
            if !matches!(
                dtype,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 | safetensors::Dtype::F32
            ) {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit curve: {key} is stored as {dtype:?}; a curve operand must be \
                     dense F16/BF16/F32, never packed codes"
                )));
            }
            for st in &stores {
                if st.get(key).is_ok() {
                    out.insert(key.clone(), st.load(key, device)?);
                    break;
                }
            }
            if !out.contains_key(key) {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit curve: {key} is in the scanned headers but no shard carries it"
                )));
            }
        }
        Ok(out)
    }
}

/// Whether a tensor name is one of the curve's canonical operands — the table, a block's reduced
/// `adaln_proj.linear` pair, or the final `norm_out.linear` pair.
///
/// Shape-blind on purpose: the header scan runs before [`PrunedAdaLn::inspect`] has confirmed the
/// geometry, so the filter must not depend on a width that could itself be the inconsistency under
/// test.
fn is_curve_key(name: &str) -> bool {
    if name == TABLE_KEY {
        return true;
    }
    if let Some(rest) = name.strip_prefix("transformer_blocks.") {
        let Some((layer, tail)) = rest.split_once('.') else {
            return false;
        };
        return layer
            .parse::<usize>()
            .is_ok()
            && matches!(tail, "adaln_proj.linear.weight" | "adaln_proj.linear.bias");
    }
    matches!(name, "norm_out.linear.weight" | "norm_out.linear.bias")
}
