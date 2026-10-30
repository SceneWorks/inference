//! **Frozen-grid holdout validation** of the Iris generation trainer (story sc-25689): upstream's
//! `Trainer._val_loss` / `_build_val_loader` (`train.val_every_steps`, `train.val_samples`,
//! `data.val_data_dirs`) — the flow loss of a fixed holdout slice under a deterministic
//! `(timestep, noise)` grid, reported overall and per sigma-index quartile.
//!
//! The grid is a pure function of each holdout sample's **absolute index** `k` in the holdout set:
//! timestep index `(k · 619) mod N`, noise keyed `mix_seed(0x5AF1, k)`; captions are pinned to
//! `(seed 0, epoch 0)`; no CFG dropout. So the grid is invariant across steps, runs, batch sizes,
//! world sizes and batching policies. The noise is drawn from the shared host stream
//! ([`super::HostRng`], as every other positional draw of the native trainers), not torch's.
//!
//! Holdout items are the request's [`gen_core::train::TrainingItem`]s whose
//! `model_options.split` is `"holdout"` (the native counterpart of `data.val_data_dirs`, which
//! upstream reads as separate WIDS shards); they never enter the training walk. Every rank
//! evaluates its slice of the holdout walk and the sums/counts are all-reduced.

use serde_json::{json, Value as JsonValue};

use super::{mix_seed, HostRng};
use gen_core::train::TrainingItem;
use gen_core::{Error, Result};

/// `VAL_GRID_SEED`.
pub const VAL_GRID_SEED: u64 = 0x5AF1;
/// `VAL_T_STRIDE` (coprime with 1000; low-discrepancy timestep coverage).
pub const VAL_T_STRIDE: usize = 619;
/// `TrainingItem::model_options` key that assigns an item to the holdout (`"holdout"`) or the
/// training set (`"train"`, the default).
pub const ITEM_SPLIT_KEY: &str = "split";
/// The metrics file a run appends its holdout and step logs to, inside the checkpoint root.
pub const METRICS_FILE: &str = "metrics.jsonl";

/// The resolved holdout validation of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutPlan {
    /// Every `every` optimizer steps (`train.val_every_steps`); 0 = off.
    pub every: u32,
    /// `train.val_samples`: the leading holdout items evaluated (0 = all).
    pub samples: usize,
}

/// Whether `item` belongs to the holdout set (refusing an unknown split value).
pub fn is_holdout(item: &TrainingItem) -> Result<bool> {
    match item.model_options.get(ITEM_SPLIT_KEY) {
        None | Some(JsonValue::Null) => Ok(false),
        Some(JsonValue::String(s)) if s == "train" => Ok(false),
        Some(JsonValue::String(s)) if s == "holdout" => Ok(true),
        Some(other) => Err(Error::Msg(format!(
            "iris training: item {} has model_options.{ITEM_SPLIT_KEY} = {other} (train | holdout)",
            item.image_path.display()
        ))),
    }
}

/// Split the request items into `(train, holdout)` index lists, preserving order.
pub fn split_items(items: &[TrainingItem]) -> Result<(Vec<usize>, Vec<usize>)> {
    let (mut train, mut holdout) = (Vec::new(), Vec::new());
    for (i, item) in items.iter().enumerate() {
        if is_holdout(item)? {
            holdout.push(i);
        } else {
            train.push(i);
        }
    }
    Ok((train, holdout))
}

/// The number of holdout items the grid covers (`min(len, val_samples)`, 0 = all).
pub fn holdout_limit(len: usize, samples: usize) -> usize {
    if samples == 0 {
        len
    } else {
        len.min(samples)
    }
}

/// Holdout sample `k`'s timestep index on an `n`-point grid.
pub fn grid_timestep(k: usize, n: usize) -> usize {
    (k * VAL_T_STRIDE) % n
}

/// Holdout sample `k`'s noise (`numel` standard normals, the sample's own shape).
pub fn grid_noise(k: usize, numel: usize) -> Vec<f32> {
    HostRng::new(mix_seed(&[VAL_GRID_SEED, k as u64])).normals_f32(numel)
}

/// The sigma-index quartile band of a timestep index (`(t · 4) // N`).
pub fn band(t_idx: usize, n: usize) -> usize {
    (t_idx * 4) / n
}

/// The accumulated sums/counts of one evaluation: `[total, q0, q1, q2, q3]` (all-reduced across
/// ranks before [`ValSums::logs`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValSums {
    pub sums: [f64; 5],
    pub counts: [f64; 5],
}

impl ValSums {
    /// Add one sample's loss at timestep index `t_idx`.
    pub fn add(&mut self, loss: f64, t_idx: usize, n: usize) {
        self.sums[0] += loss;
        self.counts[0] += 1.0;
        let q = band(t_idx, n);
        self.sums[1 + q] += loss;
        self.counts[1 + q] += 1.0;
    }

    /// The flat `[sums.., counts..]` vector a collective reduces.
    pub fn to_flat(&self) -> Vec<f32> {
        self.sums
            .iter()
            .chain(self.counts.iter())
            .map(|&v| v as f32)
            .collect()
    }

    pub fn from_flat(v: &[f32]) -> Self {
        let mut s = Self::default();
        for i in 0..5 {
            s.sums[i] = v[i] as f64;
            s.counts[i] = v[5 + i] as f64;
        }
        s
    }

    /// `val/loss` and `val/loss_t{0..3}` (sum / max(count, 1)).
    pub fn logs(&self) -> Vec<(String, f64)> {
        let mean = |i: usize| self.sums[i] / self.counts[i].max(1.0);
        let mut out = vec![("val/loss".to_string(), mean(0))];
        for q in 0..4 {
            out.push((format!("val/loss_t{q}"), mean(1 + q)));
        }
        out
    }

    /// One metrics-file record.
    pub fn record(&self, step: u64) -> JsonValue {
        let mut m = serde_json::Map::new();
        m.insert("step".into(), json!(step));
        m.insert("kind".into(), json!("validation"));
        for (k, v) in self.logs() {
            m.insert(k, json!(v));
        }
        JsonValue::Object(m)
    }
}

/// Append one JSON record (a line) to `dir/metrics.jsonl`, fsynced.
pub fn append_metrics(dir: &std::path::Path, record: &JsonValue) -> Result<()> {
    use std::io::Write as _;
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::Msg(format!("iris metrics: create {}: {e}", dir.display())))?;
    let path = dir.join(METRICS_FILE);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| Error::Msg(format!("iris metrics: open {}: {e}", path.display())))?;
    let mut line = record.to_string();
    line.push('\n');
    f.write_all(line.as_bytes())
        .and_then(|_| f.sync_all())
        .map_err(|e| Error::Msg(format!("iris metrics: write {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_is_a_function_of_the_sample_index() {
        assert_eq!(grid_timestep(0, 1000), 0);
        assert_eq!(grid_timestep(3, 1000), 857);
        assert_eq!(grid_noise(7, 4), grid_noise(7, 4));
        assert_ne!(grid_noise(7, 4), grid_noise(8, 4));
        assert_eq!(band(999, 1000), 3);
        assert_eq!(band(250, 1000), 1);
        let mut s = ValSums::default();
        s.add(1.0, 10, 1000);
        s.add(3.0, 900, 1000);
        let logs = s.logs();
        assert_eq!(logs[0], ("val/loss".to_string(), 2.0));
        assert_eq!(logs[1].1, 1.0);
        assert_eq!(logs[4].1, 3.0);
        assert_eq!(logs[2].1, 0.0);
        assert_eq!(ValSums::from_flat(&s.to_flat()), s);
    }

    #[test]
    fn items_split_by_their_flag() {
        let mut a = TrainingItem::captioned("a.png".into(), "a".into());
        let b = TrainingItem::captioned("b.png".into(), "b".into());
        a.model_options
            .insert(ITEM_SPLIT_KEY.into(), JsonValue::String("holdout".into()));
        assert_eq!(
            split_items(&[a.clone(), b, a.clone()]).unwrap(),
            (vec![1], vec![0, 2])
        );
        a.model_options
            .insert(ITEM_SPLIT_KEY.into(), JsonValue::String("val".into()));
        assert!(split_items(&[a]).is_err());
        assert_eq!(holdout_limit(10, 0), 10);
        assert_eq!(holdout_limit(10, 4), 4);
    }
}
