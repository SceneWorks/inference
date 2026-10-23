//! **Direct, centered-curve AdaLN topology** (sc-26684) — the pruned MiniMax-H3 timestep path on
//! MLX.
//!
//! The pinned WanGP curve runtime (`47b06ff920c16b4b7b36de068367a5de0e0968e9`) does not reconstruct
//! the full `[T, 2688]` timestep MLP. It stores the SVD of the *centered, post-SiLU* embedding curve
//! directly:
//!
//! * `adaln_t_table` — `U*S`, F32 `[grid, rank]`;
//! * every block / final projection — `W*Vh.T`, BF16 `[out, rank]`;
//! * every folded bias — `b + W*mean`, F16 `[out]`.
//!
//! Coordinates are interpolated (`torch.lerp`) against the table at `clamp(t,0,1) * (grid - 1)` with
//! **no factor of 1000 and no SiLU**: the SiLU is already inside the basis the curve was factored
//! from, so the reduced projections are plain Linears over the interpolated coordinates.
//!
//! Storage precision is per operand (F32 table, BF16 weight, F16 bias — the frozen INT8 artifact's
//! *reduced* dtypes, not the pruner's F32 intermediates). Compute widens to F32, matching the pinned
//! curve runtime's `_lock_dtype`, without ever recasting the stored source arrays.
//!
//! The complete reduced topology — and nothing else — is what
//! [`gen_core::pruned_adaln`] validates and hashes. Body weights and quantization codes are
//! deliberately **not** read or hashed here: the source-binding/ConvRot layers own those, and a
//! second basis with identical shapes must still produce a different identity.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::pruned_adaln::{
    interpolation_position, PrunedAdaLn, PruningIdentity, TABLE_KEY,
};
use mlx_gen::gen_core::weightsmeta::Dtype as GenDtype;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::adaln::{AdaLnCache, TimestepSchedule};
use crate::dit::block::AdaLnModulation;
use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::heads::NormOutModulation;
use crate::tensor::slice_axis;

/// Map an MLX array dtype onto the gen_core safetensors dtype vocabulary. `None` for an integer /
/// compact dtype, which a pruned operand can never legitimately carry — the caller then treats the
/// key as absent so [`PrunedAdaLn::inspect`]'s own missing/conflict rules decide the verdict.
fn gen_dtype(dtype: Dtype) -> Option<GenDtype> {
    match dtype {
        Dtype::Float32 => Some(GenDtype::F32),
        Dtype::Float16 => Some(GenDtype::F16),
        Dtype::Bfloat16 => Some(GenDtype::BF16),
        _ => None,
    }
}

/// Header-only detection: is `w` a complete pruned topology, a full MLP topology (`Ok(None)`), or an
/// inconsistent mixture that must refuse before any execution (`Err`)?
///
/// `full_width` is the *configured* timestep-embedding width (`time_embed_dim`) on the unpruned
/// path; a reduced `[out, rank]` projection without a table, or a table alongside a full timestep
/// MLP, is a hard error — never a fallback to stock weights.
pub fn inspect(w: &Weights, cfg: &MiniMaxH3DitConfig) -> Result<Option<PrunedAdaLn>> {
    let lookup = |key: &str| {
        let array = w.get(key)?;
        let dtype = gen_dtype(array.dtype())?;
        Some((dtype, array.shape().iter().map(|&d| d as usize).collect()))
    };
    PrunedAdaLn::inspect(
        cfg.hidden_size as usize,
        cfg.num_layers as usize,
        cfg.time_embed_dim as usize,
        lookup,
    )
    .map_err(Error::Msg)
}

/// Content-binding identity over exactly the pruning operands, in contract order, losslessly widened
/// to F32 with the stored dtype folded into the digest. A missing/incomplete/non-finite operand, or a
/// same-shape table/basis that differs in value, fails here rather than silently identifying.
pub fn binding_identity(w: &Weights, topology: &PrunedAdaLn) -> Result<PruningIdentity> {
    let mut builder = topology.identity_builder();
    for operand in topology.operands() {
        let array = w.get(&operand.name).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 pruned adaln: identity operand {} is missing",
                operand.name
            ))
        })?;
        // The loaded source must agree with the topology's *declared* dtype and shape before it is
        // hashed under that declaration — otherwise a caller could bind a topology from one basis
        // and feed values from another, mislabelling whose bytes the identity digests.
        let source_dtype = gen_dtype(array.dtype()).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 pruned adaln: {} has dtype {:?}, not a dense F16/BF16/F32 source",
                operand.name,
                array.dtype()
            ))
        })?;
        if source_dtype != operand.dtype {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {} declares {:?} but the loaded source is {:?}",
                operand.name, operand.dtype, source_dtype
            )));
        }
        let actual: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
        if actual.as_slice() != operand.shape.as_slice() {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {} declares {:?} but the loaded source is {actual:?}",
                operand.name, operand.shape
            )));
        }
        let widened = array.as_dtype(Dtype::Float32)?;
        builder
            .operand_values(&operand.name, widened.as_slice::<f32>().iter().copied())
            .map_err(Error::Msg)?;
    }
    builder.finish().map_err(Error::Msg)
}

/// Inspect then build: `Ok(None)` on the full MLP topology, `Ok(Some(..))` on a complete pruned
/// one, `Err` on an inconsistent mixture. The pruned runtime binds every operand and hashes the
/// content identity before returning.
pub fn detect(w: &Weights, cfg: &MiniMaxH3DitConfig) -> Result<Option<PrunedAdaLnRuntime>> {
    match inspect(w, cfg)? {
        None => Ok(None),
        Some(topology) => Ok(Some(PrunedAdaLnRuntime::from_weights(w, &topology, cfg)?)),
    }
}

/// The interpolated coordinate table — `U*S`, F32 `[grid, rank]`.
#[derive(Debug, Clone)]
pub struct PrunedTable {
    table: Array,
    grid: usize,
    rank: i32,
}

impl PrunedTable {
    /// Load `adaln_t_table` at its published F32 dtype, `[grid, rank]`.
    pub fn from_weights(w: &Weights, grid: usize, rank: usize) -> Result<Self> {
        let table = w.require(TABLE_KEY)?.clone();
        // Refuse a non-F32 table *before* any cast — widening a BF16/F16 table into validity
        // would fake the contract the header-only `inspect` already enforced.
        if table.dtype() != Dtype::Float32 {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {TABLE_KEY} must be F32, got {:?}",
                table.dtype()
            )));
        }
        let want = [grid as i32, rank as i32];
        if table.shape() != want {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {TABLE_KEY} must be F32 {want:?}, got {:?}",
                table.shape()
            )));
        }
        Ok(Self {
            table,
            grid,
            rank: rank as i32,
        })
    }

    pub fn grid(&self) -> usize {
        self.grid
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Interpolate coordinates for `timesteps` (already `t = 1 − σ`, never ×1000): one `[rank]` row
    /// per timestep, at `clamp(t,0,1) * (grid − 1)` — pinned `torch.lerp(table[lower],
    /// table[lower + 1], fraction)` addressing with no SiLU and no factor of 1000.
    pub fn coordinates(&self, timesteps: &[f32]) -> Result<Array> {
        if timesteps.is_empty() {
            return Err(Error::Msg("minimax-h3 pruned adaln: no timesteps".into()));
        }
        let mut lowers = Vec::with_capacity(timesteps.len());
        let mut fractions = Vec::with_capacity(timesteps.len());
        for &t in timesteps {
            let (lower, fraction) = interpolation_position(t, self.grid).map_err(Error::Msg)?;
            lowers.push(lower as i32);
            fractions.push(fraction);
        }
        let n = timesteps.len() as i32;
        // `lower` is at most `grid − 2`, so `lower + 1` is in bounds at the off-grid endpoint.
        let lower_idx = Array::from_slice(&lowers, &[n]);
        let upper_idx = Array::from_slice(&lowers.iter().map(|l| l + 1).collect::<Vec<_>>(), &[n]);
        let fraction = Array::from_slice(&fractions, &[n, 1]);
        let lo = self.table.take_axis(&lower_idx, 0)?;
        let hi = self.table.take_axis(&upper_idx, 0)?;
        let delta = subtract(&hi, &lo)?;
        Ok(add(&lo, &multiply(&delta, &fraction)?)?)
    }

    /// [`Self::coordinates`] under the name the timestep-embedding seam expects.
    pub fn forward(&self, timesteps: &[f32]) -> Result<Array> {
        self.coordinates(timesteps)
    }
}

/// A reduced dense projection — `W*Vh.T` BF16 `[out, rank]` plus folded bias `b + W*mean` F16
/// `[out]` — held at their **stored** dtypes and computed at F32 (the pinned curve runtime's
/// `_lock_dtype`), with no SiLU.
#[derive(Debug, Clone)]
pub struct PrunedReducedLinear {
    weight: Array,
    bias: Array,
    out_features: i32,
    rank: i32,
}

impl PrunedReducedLinear {
    /// Load `{prefix}.weight` (BF16) and `{prefix}.bias` (F16) at their stored dtypes.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        out_features: usize,
        rank: usize,
    ) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let bias = w.require(&format!("{prefix}.bias"))?.clone();
        let want = vec![out_features as i32, rank as i32];
        if weight.shape() != want.as_slice() {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {prefix}.weight must be {want:?}, got {:?}",
                weight.shape()
            )));
        }
        if bias.shape() != [out_features as i32] {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {prefix}.bias must be [{out_features}], got {:?}",
                bias.shape()
            )));
        }
        if !matches!(
            weight.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        ) || !matches!(
            bias.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        ) {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: {prefix} must be dense F16/BF16/F32, got {:?}/{:?}",
                weight.dtype(),
                bias.dtype()
            )));
        }
        Ok(Self {
            weight,
            bias,
            out_features: out_features as i32,
            rank: rank as i32,
        })
    }

    /// The stored weight, at its source dtype — exposed so a test can assert **which** precision the
    /// checkpoint ships, not just that the forward produced the right numbers.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// The stored folded bias, at its source dtype.
    pub fn bias(&self) -> &Array {
        &self.bias
    }

    pub fn out_features(&self) -> i32 {
        self.out_features
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// `coords @ Wᵀ + b` at F32 — the stored BF16/F16 operands are widened losslessly for the
    /// matmul and never recast in place. No SiLU: the coordinates are already in the post-SiLU
    /// curve basis.
    pub fn forward(&self, coords: &Array) -> Result<Array> {
        let coords = coords.as_dtype(Dtype::Float32)?;
        let weight = self.weight.as_dtype(Dtype::Float32)?;
        let bias = self.bias.as_dtype(Dtype::Float32)?;
        mlx_gen::nn::linear(&coords, &weight, &bias)
    }
}

/// One block's reduced AdaLN projection: the six modulation tables split out of `[T·3, 6·hidden]`,
/// exactly as [`crate::dit::block::AdaLnProjection`] does — but from the reduced weight and with no
/// SiLU on the coordinates.
#[derive(Debug, Clone)]
pub struct PrunedAdaLnProjection {
    linear: PrunedReducedLinear,
    hidden_size: i32,
}

impl PrunedAdaLnProjection {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        hidden_size: usize,
        rank: usize,
    ) -> Result<Self> {
        let out = hidden_size * 18;
        Ok(Self {
            linear: PrunedReducedLinear::from_weights(w, prefix, out, rank)?,
            hidden_size: hidden_size as i32,
        })
    }

    pub fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// The raw `[steps, 18·hidden]` projection, before the modality reshape and chunk — the exact
    /// tensor the oracle records as a block's `block_modulations`.
    pub fn projected(&self, coords: &Array) -> Result<Array> {
        self.linear.forward(coords)
    }

    /// `forward(coords)` is `[steps, 18·hidden]`; modality becomes a row axis before the chunk, so
    /// the six tables are each `[steps·3, hidden]` — the same reshape the full projection does.
    pub fn forward(&self, coords: &Array) -> Result<AdaLnModulation> {
        let projected = self.projected(coords)?;
        let s = projected.shape();
        if s.len() != 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: expected coords as [steps, rank], got {:?}",
                coords.shape()
            )));
        }
        let steps = s[0];
        let rows = steps * MODALITY_NUM;
        let table = projected.reshape(&[rows, MODULATION_PARAMS * self.hidden_size])?;
        let part = |i: i32| slice_axis(&table, 1, i * self.hidden_size, (i + 1) * self.hidden_size);
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

/// The final `norm_out.linear` reduced projection — `shift` then `scale`, per row, no SiLU.
#[derive(Debug, Clone)]
pub struct PrunedFinalProjection {
    linear: PrunedReducedLinear,
    hidden_size: i32,
}

impl PrunedFinalProjection {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        hidden_size: usize,
        rank: usize,
    ) -> Result<Self> {
        Ok(Self {
            linear: PrunedReducedLinear::from_weights(w, prefix, 2 * hidden_size, rank)?,
            hidden_size: hidden_size as i32,
        })
    }
    /// The raw `[T, 2·hidden]` projection, before the shift/scale split.
    pub fn projected(&self, coords: &Array) -> Result<Array> {
        self.linear.forward(coords)
    }

    /// `[T, 2·hidden]` → `(shift, scale)`, shift FIRST — the same half order the full `norm_out`
    /// uses, but from the reduced weight with no SiLU.
    pub fn modulation(&self, coords: &Array) -> Result<NormOutModulation> {
        let projected = self.projected(coords)?;
        Ok(NormOutModulation {
            shift: slice_axis(&projected, 1, 0, self.hidden_size)?,
            scale: slice_axis(&projected, 1, self.hidden_size, 2 * self.hidden_size)?,
        })
    }
}

/// The complete validated pruned runtime: the coordinate table, every reduced block projection, the
/// reduced final projection, and their content identity.
#[derive(Debug, Clone)]
pub struct PrunedAdaLnRuntime {
    table: PrunedTable,
    block_projections: Vec<PrunedAdaLnProjection>,
    final_projection: PrunedFinalProjection,
    identity: PruningIdentity,
}

impl PrunedAdaLnRuntime {
    pub fn from_weights(
        w: &Weights,
        topology: &PrunedAdaLn,
        cfg: &MiniMaxH3DitConfig,
    ) -> Result<Self> {
        let table = PrunedTable::from_weights(w, topology.grid, topology.rank)?;
        let block_projections = (0..cfg.num_layers)
            .map(|i| {
                PrunedAdaLnProjection::from_weights(
                    w,
                    &format!("transformer_blocks.{i}.adaln_proj.linear"),
                    cfg.hidden_size as usize,
                    topology.rank,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let final_projection = PrunedFinalProjection::from_weights(
            w,
            "norm_out.linear",
            cfg.hidden_size as usize,
            topology.rank,
        )?;
        let identity = binding_identity(w, topology)?;
        Ok(Self {
            table,
            block_projections,
            final_projection,
            identity,
        })
    }

    /// Project the whole schedule's modulation from the reduced projections — the pruned twin of
    /// [`AdaLnCache::precompute`]. The projections are a few tens of MB, so there is no eviction
    /// step: [`AdaLnCache::from_layers_pruned`] still forces evaluation per block (the lazy-eval
    /// discipline) and binds the cache to this topology's content identity.
    ///
    /// `num_blocks` must equal the reduced projection count exactly; the modulation is projected
    /// **only** from this runtime's own `block_projections` — never from caller-passed block bodies,
    /// whose identity belongs to the source binding, not the pruning identity.
    pub fn precompute(&self, schedule: TimestepSchedule, num_blocks: usize) -> Result<AdaLnCache> {
        if num_blocks != self.block_projections.len() {
            return Err(Error::Msg(format!(
                "minimax-h3 pruned adaln: projected {} blocks but the stack has {num_blocks}",
                self.block_projections.len()
            )));
        }
        let coords = self.table.coordinates(schedule.distinct_timesteps())?;
        let layers = self
            .block_projections
            .iter()
            .map(|projection| projection.forward(&coords))
            .collect::<Result<Vec<_>>>()?;
        AdaLnCache::from_layers_pruned(schedule, layers, self.identity)
    }

    pub fn table(&self) -> &PrunedTable {
        &self.table
    }

    pub fn num_layers(&self) -> usize {
        self.block_projections.len()
    }

    pub fn identity(&self) -> PruningIdentity {
        self.identity
    }

    /// Interpolated coordinates for a schedule's distinct timesteps.
    pub fn coords(&self, timesteps: &[f32]) -> Result<Array> {
        self.table.coordinates(timesteps)
    }

    /// The reduced block projection for `layer`.
    pub fn block_projection(&self, layer: usize) -> Result<&PrunedAdaLnProjection> {
        self.block_projections
            .get(layer)
            .ok_or_else(|| Error::Msg(format!("minimax-h3 pruned adaln: no layer {layer}")))
    }

    /// Project one block's modulation from the interpolated coordinates.
    pub fn block_modulation(&self, layer: usize, coords: &Array) -> Result<AdaLnModulation> {
        self.block_projection(layer)?.forward(coords)
    }

    pub fn final_projection(&self) -> &PrunedFinalProjection {
        &self.final_projection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_vocabulary_maps_the_pruned_operand_dtypes() {
        assert_eq!(gen_dtype(Dtype::Float32), Some(GenDtype::F32));
        assert_eq!(gen_dtype(Dtype::Float16), Some(GenDtype::F16));
        assert_eq!(gen_dtype(Dtype::Bfloat16), Some(GenDtype::BF16));
        assert_eq!(gen_dtype(Dtype::Uint8), None);
    }
}
