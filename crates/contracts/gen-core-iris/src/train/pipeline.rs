//! The generation trainers' **data pipeline** (story sc-25689): the request items split into the
//! training set and the holdout, each training item's native size and shape key, and the per-rank
//! walk — upstream's `build_dataloader` (`RangedSampler` for a uniform policy, `ShapeBatchSampler`
//! with `drop_last` otherwise), the epoch bookkeeping of `Trainer.run` (`set_epoch`,
//! `limit_batches(epoch_batches)`, the resume cursor, `_checkpoint_data_position`), and the
//! frozen-grid holdout walk of `_build_val_loader`.
//!
//! Host-side and tensor-free, shared verbatim by the MLX and Candle trainers. The caller performs
//! the one collective upstream needs here (`_agree_batch_count`: the global minimum of the
//! per-rank shape-batch counts) and hands the agreed count back.

use serde_json::{json, Value as JsonValue};

use super::holdout::{holdout_limit, split_items};
use super::shapes::{RangedSampler, ShapeBatchSampler, ShapePolicy};
use gen_core::train::TrainingItem;
use gen_core::{Error, Result};

#[derive(Clone, Debug)]
enum Walker {
    Ranged(RangedSampler),
    Shape(ShapeBatchSampler),
}

/// One rank's view of the training data.
#[derive(Clone, Debug)]
pub struct DataPipeline {
    /// Request item index of each training position.
    pub train: Vec<usize>,
    /// Request item index of each holdout position.
    pub holdout: Vec<usize>,
    /// Native `(height, width)` of each training position.
    pub sizes: Vec<(usize, usize)>,
    /// The shape key of each training position (`None`: an unreadable sample, skipped).
    pub keys: Vec<Option<String>>,
    pub policy: ShapePolicy,
    pub batch_size: usize,
    pub rank: usize,
    pub world_size: usize,
    walker: Walker,
    /// The agreed full-epoch batch count (`epoch_batches` upstream).
    epoch_batches: usize,
    /// Batches skipped at the start of the current pass (`skip_batches`).
    pub skip: usize,
}

fn image_size(path: &std::path::Path) -> Result<(usize, usize)> {
    let (w, h) = image::image_dimensions(path).map_err(|e| {
        Error::Msg(format!(
            "iris training: item {} is not a readable image: {e}",
            path.display()
        ))
    })?;
    Ok((h as usize, w as usize))
}

impl DataPipeline {
    /// Split the request, read every training image's header, build this rank's walker.
    pub fn new(
        items: &[TrainingItem],
        policy: ShapePolicy,
        batch_size: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let (train, holdout) = split_items(items)?;
        if train.is_empty() {
            return Err(Error::Msg(
                "iris training: the dataset has no training items (every item is holdout)".into(),
            ));
        }
        let mut sizes = Vec::with_capacity(train.len());
        let mut keys = Vec::with_capacity(train.len());
        for &i in &train {
            let (h, w) = image_size(&items[i].image_path)?;
            sizes.push((h, w));
            keys.push(Some(policy.key(h, w)));
        }
        let sampler = RangedSampler::new(train.len(), rank, world_size)?;
        let walker = if policy.uniform() {
            Walker::Ranged(sampler)
        } else {
            Walker::Shape(ShapeBatchSampler::new(sampler, batch_size, true)?)
        };
        let mut p = Self {
            train,
            holdout,
            sizes,
            keys,
            policy,
            batch_size,
            rank,
            world_size,
            walker,
            epoch_batches: 0,
            skip: 0,
        };
        p.epoch_batches = p.local_fresh_batches();
        Ok(p)
    }

    /// The dataset items the ranged walk never visits (the tail beyond `chunks · chunk_size`).
    pub fn unused_tail_warning(&self) -> Option<String> {
        let covered = match &self.walker {
            Walker::Ranged(s) => s.covered(),
            Walker::Shape(s) => s.sampler.covered(),
        };
        let n = self.train.len();
        (covered < n).then(|| {
            format!(
                "iris training: warning — {} of {n} training items (positions {covered}..{n}) are \
                 never trained on: the upstream ranged sampler covers min(640, n) · ⌊n / chunks⌋ \
                 items",
                n - covered
            )
        })
    }

    /// This rank's batch count of a fresh epoch (before any cross-rank agreement).
    pub fn local_fresh_batches(&mut self) -> usize {
        match &mut self.walker {
            Walker::Ranged(s) => {
                s.set_epoch(1);
                s.len().div_ceil(self.batch_size)
            }
            Walker::Shape(s) => {
                s.set_epoch(1);
                let keys = self.keys.clone();
                s.len(&keys)
            }
        }
    }

    /// Install the agreed full-epoch batch count (`_agree_batch_count`'s global minimum; the local
    /// count on one rank). Refuses zero (`a rank produced zero shape-homogeneous batches`).
    pub fn set_epoch_batches(&mut self, agreed: usize) -> Result<()> {
        if agreed == 0 {
            return Err(Error::Msg(
                "iris training: a rank produced zero batches; lower batch_size".into(),
            ));
        }
        self.epoch_batches = agreed;
        Ok(())
    }

    /// The agreed full-epoch batch count (`len(dataloader)`).
    pub fn epoch_batches(&self) -> usize {
        self.epoch_batches
    }

    /// Whether a resumed pass needs a second agreement (`_agree_batch_count(allow_empty=True)`).
    pub fn regroups(&self) -> bool {
        matches!(self.walker, Walker::Shape(_))
    }

    /// Start `epoch`: a fresh pass, or the resumed first pass from a saved cursor. Returns the
    /// batches already consumed in it (`skip_batches`).
    pub fn begin_epoch(&mut self, epoch: u32, resume: Option<&JsonValue>) -> Result<usize> {
        let keys = self.keys.clone();
        match &mut self.walker {
            Walker::Ranged(s) => {
                s.set_epoch(epoch);
                match resume {
                    Some(state) => s.load_resume_state(state)?,
                    None => s.set_start_batches(0, self.batch_size)?,
                }
            }
            Walker::Shape(s) => {
                s.set_epoch(epoch);
                s.limit_batches(Some(self.epoch_batches));
                match resume {
                    Some(state) => s.load_resume_state(state, &keys)?,
                    None => s.set_start_batches(0, self.batch_size, &keys)?,
                }
            }
        }
        self.skip = match resume {
            Some(state) => state
                .get("batches_consumed")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| {
                    Error::Msg("iris resume: data_position has no batches_consumed".into())
                })? as usize,
            None => 0,
        };
        Ok(self.skip)
    }

    /// This rank's remaining batch count of the current pass (for the resumed-pass agreement).
    pub fn remaining_batches(&mut self) -> usize {
        let keys = self.keys.clone();
        match &mut self.walker {
            Walker::Ranged(s) => s.len().div_ceil(self.batch_size),
            Walker::Shape(s) => s.len(&keys),
        }
    }

    /// Cap the current pass at `agreed` remaining batches (shape batching only).
    pub fn limit_remaining(&mut self, agreed: usize) {
        if let Walker::Shape(s) = &mut self.walker {
            s.limit_batches(Some(agreed));
        }
    }

    /// The current pass's batches, as training positions.
    pub fn batches(&mut self) -> Vec<Vec<usize>> {
        let keys = self.keys.clone();
        match &mut self.walker {
            Walker::Ranged(s) => {
                let idx = s.indices();
                idx.chunks(self.batch_size).map(<[usize]>::to_vec).collect()
            }
            Walker::Shape(s) => s.batches(&keys),
        }
    }

    /// `_checkpoint_data_position`: the sampler's resume state plus `epoch` and the yielded batch
    /// count. `plan_batches` / `plan_samples` count what this pass yielded.
    pub fn data_position(
        &mut self,
        epoch: u32,
        batches_consumed: usize,
        plan_batches: usize,
        plan_samples: usize,
    ) -> Result<JsonValue> {
        let keys = self.keys.clone();
        let mut state = match &mut self.walker {
            Walker::Ranged(s) => {
                // The ranged walk's progress is installed by trimming at load; its saved cursor is
                // the samples this pass yielded on top of what the plan already excludes.
                s.resume_state(plan_batches, self.batch_size, None, Some(plan_samples))?
            }
            Walker::Shape(s) => {
                s.resume_state(plan_batches, self.batch_size, Some(plan_samples), &keys)?
            }
        };
        let obj = state.as_object_mut().expect("object");
        obj.insert("epoch".into(), json!(epoch));
        obj.insert("batches_consumed".into(), json!(batches_consumed));
        Ok(state)
    }

    /// The `new_phase` cursor: epoch 1, nothing consumed, over the **current** dataset.
    pub fn fresh_position(&mut self) -> Result<JsonValue> {
        match &mut self.walker {
            Walker::Ranged(s) => s.set_epoch(1),
            Walker::Shape(s) => s.set_epoch(1),
        }
        self.data_position(1, 0, 0, 0)
    }

    /// A legacy (pre-sc-25689, single-rank, uniform) cursor: `batches` micro-batches of the
    /// **saved** `saved_batch` size ([`super::CheckpointState::legacy_batch`]) consumed in `epoch`.
    /// The cursor records the saved batch size, so the load resumes by samples exactly as a
    /// current checkpoint does across a micro-batch change.
    pub fn legacy_position(
        &mut self,
        epoch: u32,
        batches: usize,
        saved_batch: usize,
    ) -> Result<JsonValue> {
        if self.world_size != 1 || !self.policy.uniform() {
            return Err(Error::Msg(
                "iris resume: a checkpoint without a data_position resumes on one rank with the \
                 fixed-square policy only"
                    .into(),
            ));
        }
        let samples = (batches * saved_batch).min(match &self.walker {
            Walker::Ranged(s) => s.covered(),
            Walker::Shape(_) => 0,
        });
        match &mut self.walker {
            Walker::Ranged(s) => {
                s.set_epoch(epoch);
                let mut st = s.resume_state(batches, saved_batch, None, Some(samples))?;
                let o = st.as_object_mut().expect("object");
                o.insert("epoch".into(), json!(epoch));
                o.insert("batches_consumed".into(), json!(batches));
                Ok(st)
            }
            Walker::Shape(_) => unreachable!("checked uniform above"),
        }
    }

    /// The holdout walk of this rank: `(holdout positions)` batches over the leading
    /// `val_samples`, grouped by shape (keeping partial groups) when the policy varies shapes.
    /// `sizes` are the holdout items' native sizes.
    pub fn holdout_batches(
        &self,
        holdout_sizes: &[(usize, usize)],
        val_samples: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let limit = holdout_limit(self.holdout.len(), val_samples);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let sampler = RangedSampler::new(limit, self.rank, self.world_size)?;
        if self.policy.uniform() {
            let idx = sampler.indices();
            return Ok(idx.chunks(self.batch_size).map(<[usize]>::to_vec).collect());
        }
        let keys: Vec<Option<String>> = holdout_sizes
            .iter()
            .take(limit)
            .map(|&(h, w)| Some(self.policy.key(h, w)))
            .collect();
        let mut s = ShapeBatchSampler::new(sampler, self.batch_size, false)?;
        s.set_epoch(0);
        Ok(s.batches(&keys))
    }

    /// The holdout items' native sizes.
    pub fn holdout_sizes(&self, items: &[TrainingItem]) -> Result<Vec<(usize, usize)>> {
        self.holdout
            .iter()
            .map(|&i| image_size(&items[i].image_path))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::holdout::ITEM_SPLIT_KEY;

    fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let p = dir.join(name);
        image::RgbImage::from_pixel(w, h, image::Rgb([10, 20, 30]))
            .save(&p)
            .unwrap();
        p
    }

    #[test]
    fn pipelines_split_walk_and_resume() {
        let d = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        for i in 0..9 {
            let (w, h) = if i % 2 == 0 { (32, 16) } else { (16, 32) };
            let mut it =
                TrainingItem::captioned(write_png(d.path(), &format!("{i}.png"), w, h), "c".into());
            if i == 8 {
                it.model_options
                    .insert(ITEM_SPLIT_KEY.into(), json!("holdout"));
            }
            items.push(it);
        }
        let policy = ShapePolicy::build("bucket", 512, 16, "shared21-512", 32, 4.0).unwrap();
        let mut p = DataPipeline::new(&items, policy, 2, 0, 1).unwrap();
        assert_eq!(
            (p.train.len(), p.holdout.clone(), p.epoch_batches()),
            (8, vec![8], 4)
        );
        p.begin_epoch(1, None).unwrap();
        let all = p.batches();
        assert_eq!(all.len(), 4);
        assert!(all.iter().all(|b| b.len() == 2));
        // resume after 1 batch
        p.begin_epoch(1, None).unwrap();
        let pos = p.data_position(1, 1, 1, 2).unwrap();
        let mut q = DataPipeline::new(&items, p.policy.clone(), 2, 0, 1).unwrap();
        assert_eq!(q.begin_epoch(1, Some(&pos)).unwrap(), 1);
        assert_eq!(q.batches(), all[1..].to_vec());
        // holdout walk keeps partial groups
        let sizes = q.holdout_sizes(&items).unwrap();
        assert_eq!(q.holdout_batches(&sizes, 0).unwrap(), vec![vec![0]]);
        // a uniform policy is the contiguous walk
        let fixed = ShapePolicy::build("fixed", 16, 16, "", 32, 4.0).unwrap();
        let mut f = DataPipeline::new(&items, fixed, 3, 0, 1).unwrap();
        f.begin_epoch(1, None).unwrap();
        assert_eq!(f.batches(), vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]);
        let pos = f.data_position(1, 2, 2, 6).unwrap();
        let mut g = DataPipeline::new(&items, f.policy.clone(), 3, 0, 1).unwrap();
        g.begin_epoch(1, Some(&pos)).unwrap();
        assert_eq!(g.batches(), vec![vec![6, 7]]);
        let legacy = g.legacy_position(1, 2, 3).unwrap();
        assert_eq!(legacy["samples_per_rank"], json!(6));
        // A legacy checkpoint counts micro-batches of ITS batch size: 2 batches of 2 = 4 samples,
        // resumed by a batch-3 run at sample 4.
        let legacy = g.legacy_position(1, 2, 2).unwrap();
        assert_eq!(legacy["samples_per_rank"], json!(4));
        let mut h = DataPipeline::new(&items, f.policy.clone(), 3, 0, 1).unwrap();
        h.begin_epoch(1, Some(&legacy)).unwrap();
        assert_eq!(h.batches(), vec![vec![4, 5, 6], vec![7]]);
    }

    /// A cursor written by a 2-rank run resumes on 1 and on 4 ranks exactly as upstream's
    /// `RangedSampler.load_resume_state` re-splits it: every rank's unread tail, in rank order,
    /// cut into equal contiguous per-rank plans (`_split_ranges`) — no sample dropped or repeated.
    #[test]
    fn a_cursor_reshards_to_another_world_like_upstreams_sampler() {
        let d = tempfile::tempdir().unwrap();
        let items: Vec<TrainingItem> = (0..8)
            .map(|i| {
                TrainingItem::captioned(
                    write_png(d.path(), &format!("{i}.png"), 16, 16),
                    "c".into(),
                )
            })
            .collect();
        let fixed = ShapePolicy::build("fixed", 16, 16, "", 32, 4.0).unwrap();
        // world 2: rank 0 walks [0, 4), rank 1 [4, 8); each consumed one batch of 2
        let mut w2: Vec<DataPipeline> = (0..2)
            .map(|r| DataPipeline::new(&items, fixed.clone(), 2, r, 2).unwrap())
            .collect();
        for p in w2.iter_mut() {
            p.begin_epoch(1, None).unwrap();
        }
        assert_eq!(w2[0].batches(), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(w2[1].batches(), vec![vec![4, 5], vec![6, 7]]);
        w2[0].begin_epoch(1, None).unwrap();
        let pos = w2[0].data_position(1, 1, 1, 2).unwrap();
        // every rank writes the same cursor (equal progress over the shared plan)
        w2[1].begin_epoch(1, None).unwrap();
        assert_eq!(w2[1].data_position(1, 1, 1, 2).unwrap(), pos);
        // upstream: unread = [2, 4) + [6, 8); one rank takes it all, four ranks one sample each
        let mut w1 = DataPipeline::new(&items, fixed.clone(), 2, 0, 1).unwrap();
        w1.begin_epoch(1, Some(&pos)).unwrap();
        assert_eq!(w1.batches(), vec![vec![2, 3], vec![6, 7]]);
        let w4: Vec<Vec<Vec<usize>>> = (0..4)
            .map(|r| {
                let mut p = DataPipeline::new(&items, fixed.clone(), 1, r, 4).unwrap();
                p.begin_epoch(1, Some(&pos)).unwrap();
                p.batches()
            })
            .collect();
        assert_eq!(
            w4,
            vec![vec![vec![2]], vec![vec![3]], vec![vec![6]], vec![vec![7]]]
        );
        // a world that cannot split the remainder evenly is refused, as upstream refuses
        let mut w3 = DataPipeline::new(&items, fixed, 1, 0, 3);
        if let Ok(p) = w3.as_mut() {
            assert!(p.begin_epoch(1, Some(&pos)).is_err());
        }
    }
}
