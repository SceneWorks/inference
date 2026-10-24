//! Direct, centered-curve AdaLN topology (sc-26684).
//!
//! WanGP `47b06ff920c16b4b7b36de068367a5de0e0968e9` stores `U*S` in
//! `adaln_t_table`, folds `W*mean` into each bias and stores `W*Vh.T` as each
//! projection. These operands are inseparable: no full basis is reconstructed,
//! and no SiLU is applied to the interpolated coordinates. Storage precision is
//! per operand (the frozen INT8 artifact uses F32/BF16/F16 respectively).
//! Compatible producers may store either reduced operand as F16, BF16 or F32;
//! the contract preserves and hashes that dtype instead of imposing one artifact's
//! mixed precision on unrelated fine-tunes. The table itself is always F32.

use sha2::{Digest, Sha256};

use crate::weightsmeta::Dtype;

/// Arithmetic/topology identity; not a dense tensor codec or conversion permission.
pub const TOPOLOGY_ID: &str = "h3-centered-adaln-curve-v1";
/// Canonical key shared by imported layouts and both native backends.
pub const TABLE_KEY: &str = "adaln_t_table";

/// One required source operand, in canonical (not filename) space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PruningOperand {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

/// Validated complete reduced topology; validation allocates headers only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrunedAdaLn {
    pub grid: usize,
    pub rank: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    operands: Vec<PruningOperand>,
}

impl PrunedAdaLn {
    /// Inspect canonical tensor headers before constructing executable projections.
    /// `full_width` is the configured time embedding width on the unpruned path.
    /// Missing tables, mixed full/reduced projections, missing folded biases and
    /// simultaneous full timestep MLPs are errors, never a fallback to stock weights.
    pub fn inspect(
        hidden_size: usize,
        num_layers: usize,
        full_width: usize,
        mut lookup: impl FnMut(&str) -> Option<(Dtype, Vec<usize>)>,
    ) -> Result<Option<Self>, String> {
        let time_names = [
            "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias",
        ];
        let Some((table_dtype, table_shape)) = lookup(TABLE_KEY) else {
            for name in time_names {
                if lookup(name).is_none() {
                    return Err(format!(
                        "H3 AdaLN: missing {TABLE_KEY} or complete timestep MLP ({name})"
                    ));
                }
            }
            for layer in 0..=num_layers {
                let name = if layer == num_layers {
                    "norm_out.linear.weight".to_owned()
                } else {
                    format!("transformer_blocks.{layer}.adaln_proj.linear.weight")
                };
                if let Some((_, shape)) = lookup(&name) {
                    if shape.len() == 2 && shape[1] != full_width {
                        return Err(format!(
                            "H3 AdaLN: {name} has reduced width without {TABLE_KEY}"
                        ));
                    }
                }
            }
            return Ok(None);
        };
        if hidden_size == 0
            || num_layers == 0
            || table_dtype != Dtype::F32
            || table_shape.len() != 2
            || table_shape[0] < 2
            || table_shape[1] == 0
            || table_shape[0] > i32::MAX as usize
            || table_shape[1] > i32::MAX as usize
            || table_shape[0].checked_mul(table_shape[1]).is_none()
        {
            return Err("H3 AdaLN: table must be finite F32 [grid >= 2, rank > 0]".into());
        }
        for name in time_names {
            if lookup(name).is_some() {
                return Err(format!(
                    "H3 AdaLN: {TABLE_KEY} conflicts with full timestep MLP {name}"
                ));
            }
        }
        let grid = table_shape[0];
        let rank = table_shape[1];
        let mut operands = vec![PruningOperand {
            name: TABLE_KEY.into(),
            dtype: table_dtype,
            shape: table_shape,
        }];
        for layer in 0..=num_layers {
            let (prefix, expansion) = if layer == num_layers {
                ("norm_out.linear".to_owned(), 2usize)
            } else {
                (
                    format!("transformer_blocks.{layer}.adaln_proj.linear"),
                    18usize,
                )
            };
            let out = hidden_size
                .checked_mul(expansion)
                .ok_or("H3 AdaLN: output width overflow")?;
            for (suffix, shape) in [("weight", vec![out, rank]), ("bias", vec![out])] {
                let name = format!("{prefix}.{suffix}");
                let (dtype, actual) = lookup(&name)
                    .ok_or_else(|| format!("H3 AdaLN: missing pruned operand {name}"))?;
                if !matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32) || actual != shape {
                    return Err(format!("H3 AdaLN: {name} must be dense F16/BF16/F32 {shape:?}, got {dtype:?} {actual:?}"));
                }
                operands.push(PruningOperand { name, dtype, shape });
            }
        }
        Ok(Some(Self {
            grid,
            rank,
            hidden_size,
            num_layers,
            operands,
        }))
    }

    /// Table followed by every block's reduced weight/bias and the final weight/bias.
    pub fn operands(&self) -> &[PruningOperand] {
        &self.operands
    }

    /// Hash the actual stored values, not paths, filenames, metadata claims or a rederived basis.
    pub fn identity_builder(&self) -> PruningIdentityBuilder<'_> {
        let mut hash = Sha256::new();
        hash.update(TOPOLOGY_ID.as_bytes());
        PruningIdentityBuilder {
            topology: self,
            next: 0,
            hash,
        }
    }
}

/// Content identity for compatibility and modulation caches. Includes every reduced projection
/// and folded bias as well as the table: another basis with the same shapes is incompatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PruningIdentity(pub [u8; 32]);

/// Streaming load-boundary content hashing, requiring every operand exactly once in contract order.
/// Values are losslessly widened to F32, with the stored dtype included in the digest. This avoids
/// materializing another full source tensor merely to hash BF16/F16 bytes on a native backend.
pub struct PruningIdentityBuilder<'a> {
    topology: &'a PrunedAdaLn,
    next: usize,
    hash: Sha256,
}

impl PruningIdentityBuilder<'_> {
    /// Supply source values before any projection/bias casts. Nonfinite data fails admission.
    pub fn operand_values(
        &mut self,
        name: &str,
        values: impl IntoIterator<Item = f32>,
    ) -> Result<(), String> {
        let operand = self
            .topology
            .operands
            .get(self.next)
            .ok_or("H3 AdaLN: extra identity operand")?;
        if operand.name != name {
            return Err(format!(
                "H3 AdaLN: identity expected {}, got {name}",
                operand.name
            ));
        }
        self.hash.update((name.len() as u64).to_le_bytes());
        self.hash.update(name.as_bytes());
        self.hash.update(operand.dtype.as_str().as_bytes());
        let mut count = 1usize;
        for dim in &operand.shape {
            self.hash.update((*dim as u64).to_le_bytes());
            count = count
                .checked_mul(*dim)
                .ok_or("H3 AdaLN: identity size overflow")?;
        }
        let mut seen = 0usize;
        for value in values {
            if !value.is_finite() || seen == count {
                return Err(format!("H3 AdaLN: invalid values for {name}"));
            }
            self.hash.update(value.to_bits().to_le_bytes());
            seen += 1;
        }
        if seen != count {
            return Err(format!("H3 AdaLN: incomplete values for {name}"));
        }
        self.next += 1;
        Ok(())
    }

    /// Refuses partial identity: a table alone cannot identify its centered projection basis.
    pub fn finish(self) -> Result<PruningIdentity, String> {
        if self.next != self.topology.operands.len() {
            return Err("H3 AdaLN: incomplete table/basis identity".into());
        }
        Ok(PruningIdentity(self.hash.finalize().into()))
    }
}

/// Pinned `torch.lerp(table[lower], table[lower+1], fraction)` addressing. At t=1 the
/// last interval is selected with fraction=1. There is deliberately no factor of 1000.
pub fn interpolation_position(t: f32, grid: usize) -> Result<(usize, f32), String> {
    if !t.is_finite() || grid < 2 || grid > i32::MAX as usize {
        return Err("H3 AdaLN: interpolation needs finite t and grid >= 2".into());
    }
    let position = t.clamp(0.0, 1.0) * (grid - 1) as f32;
    let lower = (position.floor() as usize).min(grid - 2);
    Ok((lower, position - lower as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn headers() -> BTreeMap<String, (Dtype, Vec<usize>)> {
        BTreeMap::from([
            (TABLE_KEY.into(), (Dtype::F32, vec![1025, 8])),
            (
                "transformer_blocks.0.adaln_proj.linear.weight".into(),
                (Dtype::BF16, vec![36, 8]),
            ),
            (
                "transformer_blocks.0.adaln_proj.linear.bias".into(),
                (Dtype::F16, vec![36]),
            ),
            ("norm_out.linear.weight".into(), (Dtype::BF16, vec![4, 8])),
            ("norm_out.linear.bias".into(), (Dtype::F16, vec![4])),
        ])
    }
    fn inspect(h: &BTreeMap<String, (Dtype, Vec<usize>)>) -> Result<Option<PrunedAdaLn>, String> {
        PrunedAdaLn::inspect(2, 1, 2688, |k| h.get(k).cloned())
    }
    fn identity(topology: &PrunedAdaLn, change: Option<usize>) -> PruningIdentity {
        let mut hash = topology.identity_builder();
        for (i, op) in topology.operands().iter().enumerate() {
            hash.operand_values(
                &op.name,
                std::iter::repeat_n(
                    if change == Some(i) { 0.5 } else { 0.25 },
                    op.shape.iter().product(),
                ),
            )
            .unwrap();
        }
        hash.finish().unwrap()
    }
    #[test]
    fn accepts_stored_mixed_precision_and_binds_every_operand() {
        let topology = inspect(&headers()).unwrap().unwrap();
        assert_eq!((topology.grid, topology.rank), (1025, 8));
        let original = identity(&topology, None);
        for i in 0..topology.operands().len() {
            assert_ne!(original, identity(&topology, Some(i)));
        }
        let mut changed = headers();
        changed.get_mut("norm_out.linear.bias").unwrap().0 = Dtype::BF16;
        assert_ne!(
            original,
            identity(&inspect(&changed).unwrap().unwrap(), None)
        );
        assert!(topology.identity_builder().finish().is_err());
    }
    #[test]
    fn refuses_missing_inconsistent_and_full_topologies() {
        for key in headers().keys() {
            let mut h = headers();
            h.remove(key);
            assert!(inspect(&h).is_err(), "{key}");
        }
        for key in headers().keys() {
            let mut h = headers();
            h.get_mut(key).unwrap().1[0] += 1;
            if key == TABLE_KEY {
                h.get_mut(key).unwrap().0 = Dtype::BF16;
            }
            assert!(inspect(&h).is_err(), "{key}");
        }
        let mut h = headers();
        h.insert(
            "time_embedder.linear_1.weight".into(),
            (Dtype::F32, vec![16, 16]),
        );
        assert!(inspect(&h).is_err());
        let topology = inspect(&headers()).unwrap().unwrap();
        let mut hash = topology.identity_builder();
        assert!(hash.operand_values(TABLE_KEY, [f32::NAN]).is_err());
    }
    #[test]
    fn reduced_final_projection_without_table_is_not_a_full_model() {
        let mut h = headers();
        h.remove(TABLE_KEY);
        for name in [
            "time_embedder.linear_1.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight",
            "time_embedder.linear_2.bias",
        ] {
            h.insert(name.into(), (Dtype::F32, vec![1]));
        }
        h.get_mut("transformer_blocks.0.adaln_proj.linear.weight")
            .unwrap()
            .1[1] = 2688;
        let error = inspect(&h).unwrap_err();
        assert!(error.contains("norm_out.linear.weight"), "{error}");
        h.get_mut("norm_out.linear.weight").unwrap().1[1] = 2688;
        assert!(inspect(&h).unwrap().is_none());
    }
    #[test]
    fn pinned_endpoints_off_grid_and_clamping() {
        assert_eq!(interpolation_position(-1.0, 1025).unwrap(), (0, 0.0));
        assert_eq!(interpolation_position(1.0, 1025).unwrap(), (1023, 1.0));
        assert_eq!(interpolation_position(2.0, 1025).unwrap(), (1023, 1.0));
        assert_eq!(
            interpolation_position(128.5 / 1024.0, 1025).unwrap(),
            (128, 0.5)
        );
        assert!(interpolation_position(f32::NAN, 1025).is_err());
    }
}
