//! Pinned training batch semantics: shuffle bucket membership once, drop incomplete
//! DATA batches per bucket, then shuffle only prebuilt batch ORDER each epoch.
use super::rng::NativeRng;
use crate::{CancelFlag, Error, Result};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct BatchSampler {
    batches: Vec<Vec<usize>>,
    seed: u64,
}
impl BatchSampler {
    pub fn new(bucket_ids: &[usize], batch_size: u32, seed: u64) -> Result<Self> {
        if batch_size == 0 || bucket_ids.is_empty() {
            return Err(Error::Unsupported(
                "flux3 sampler needs nonempty data and positive batch size".into(),
            ));
        }
        let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
        for (index, bucket) in bucket_ids.iter().enumerate() {
            grouped.entry(*bucket).or_default().push(index);
        }
        let mut batches = Vec::new();
        for (bucket, mut indices) in grouped {
            NativeRng::keyed(seed, "initial-within-bucket-order", bucket as u64)
                .shuffle(&mut indices)?;
            for batch in indices.chunks(batch_size as usize) {
                if batch.len() == batch_size as usize {
                    batches.push(batch.to_vec());
                }
            }
        }
        if batches.is_empty() {
            return Err(Error::Unsupported("flux3 sampler has no full batch after per-bucket drop_last; reduce batch size or add examples".into()));
        }
        Ok(Self { batches, seed })
    }
    pub fn advance_epoch(
        &self,
        batches: &mut [Vec<usize>],
        epoch: u64,
        cancel: &CancelFlag,
    ) -> Result<()> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if batches.len() != self.batches.len() {
            return Err(Error::Unsupported(
                "flux3 sampler epoch batch set mismatch".into(),
            ));
        }
        NativeRng::keyed(self.seed, "epoch-batch-order", epoch).shuffle(batches)
    }
    pub fn len(&self) -> usize {
        self.batches.len()
    }
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
    /// Reconstruct the same in-place epoch-order permutations from seed+epoch;
    /// membership/within-batch order is immutable after construction.
    pub fn epoch_batches(&self, epoch: u64, cancel: &CancelFlag) -> Result<Vec<Vec<usize>>> {
        let mut order = (0..self.batches.len()).collect::<Vec<_>>();
        for current in 0..=epoch {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            NativeRng::keyed(self.seed, "epoch-batch-order", current).shuffle(&mut order)?;
        }
        Ok(order
            .into_iter()
            .map(|index| self.batches[index].clone())
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn membership_is_built_once_and_short_data_batches_drop_per_bucket() {
        let buckets = [0, 0, 0, 1, 1, 1, 1];
        let sampler = BatchSampler::new(&buckets, 2, 73).unwrap();
        assert_eq!(sampler.len(), 3);
        let cancel = CancelFlag::new();
        let mut initial = sampler.epoch_batches(0, &cancel).unwrap();
        let mut later = sampler.epoch_batches(4, &cancel).unwrap();
        for batch in &initial {
            assert_eq!(batch.len(), 2);
            assert_eq!(buckets[batch[0]], buckets[batch[1]]);
        }
        initial.sort();
        later.sort();
        assert_eq!(
            initial, later,
            "epochs must not reshuffle membership or merge leftover buckets"
        );
        assert_eq!(
            sampler.epoch_batches(3, &cancel).unwrap(),
            BatchSampler::new(&buckets, 2, 73)
                .unwrap()
                .epoch_batches(3, &cancel)
                .unwrap()
        );
        assert!(BatchSampler::new(&[0, 1], 2, 73).is_err());
    }
}
