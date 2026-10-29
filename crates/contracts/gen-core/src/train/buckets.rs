//! Resolution buckets and the bucketed per-step sample order.

use super::*;

/// Most resolution buckets one training run may declare (epic 2123 sc-2127). Each bucket adds one
/// cached latent per dataset item and one more latent size the trainer must fit, so the list is
/// kept short; SceneWorks validates the same bound at submit time.
pub const MAX_RESOLUTION_BUCKETS: usize = 8;

/// Every [`ResolutionBucket::resolution`] must be a multiple of this (sc-2127): the trainers floor
/// each training edge to a multiple of 32, so an off-stride bucket would silently train at another
/// size — and two buckets that floor to the same edge (e.g. 512 and 520) would cache the same
/// latent twice and silently double that size's weight. SceneWorks validates the same stride.
pub const RESOLUTION_BUCKET_STRIDE: u32 = 32;

/// One training resolution bucket (see [`TrainingConfig::resolution_buckets`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolutionBucket {
    /// Square training edge in pixels, floored to the trainer's latent stride exactly like
    /// [`TrainingConfig::resolution`].
    pub resolution: u32,
    /// How many times each item is visited at this resolution per epoch (`>= 1`).
    pub repeats: u32,
}

impl TrainingConfig {
    /// The buckets this run trains on: [`resolution_buckets`](Self::resolution_buckets) when set,
    /// otherwise the single legacy bucket `{ resolution, repeats: 1 }`. Every trainer caches one
    /// latent per item per returned bucket, in this order.
    pub fn training_buckets(&self) -> Vec<ResolutionBucket> {
        if self.resolution_buckets.is_empty() {
            vec![ResolutionBucket {
                resolution: self.resolution,
                repeats: 1,
            }]
        } else {
            self.resolution_buckets.clone()
        }
    }

    /// The largest training edge of [`training_buckets`](Self::training_buckets) — the resolution
    /// every memory pre-flight / estimate must size for (epic 2123 E7).
    pub fn max_training_resolution(&self) -> u32 {
        self.training_buckets()
            .iter()
            .map(|b| b.resolution)
            .max()
            .unwrap_or(self.resolution)
    }
}

/// The per-step sample order over a bucketed latent cache (epic 2123 sc-2127).
///
/// Trainers cache `n_items × n_buckets` latents laid out item-major (`item * n_buckets + bucket`,
/// buckets in [`TrainingConfig::training_buckets`] order) and ask
/// [`cache_index`](Self::cache_index) which entry the `k`-th sample (0-based; usually `step - 1`)
/// reads.
///
/// - **One bucket** — `k % n_items`, unshuffled: exactly the round-robin walk every trainer used
///   before buckets existed, so a single bucket at today's resolution gives today's order. (A lone
///   bucket's repeat count cannot change a one-bucket per-image mix, so it is not consulted.)
/// - **Several buckets** — each epoch holds every `(item, bucket)` pair `repeats[bucket]` times
///   (`epoch_len = n_items · Σ repeats`), shuffled with a Fisher–Yates permutation whose RNG is
///   derived from the job seed and the epoch index (E4): reproducible, and a fresh order each
///   epoch.
#[derive(Clone, Debug)]
pub struct BucketSchedule {
    n_items: usize,
    n_buckets: usize,
    seed: u64,
    /// One epoch's unshuffled `(item, bucket)` multiset (empty for the single-bucket walk).
    epoch: Vec<(usize, usize)>,
    /// The most recently shuffled epoch `(epoch index, order)`, so consecutive steps reuse one
    /// permutation instead of reshuffling the whole epoch per sample.
    shuffled: std::cell::RefCell<Option<ShuffledEpoch>>,
}

/// A shuffled epoch: `(epoch index, (item, bucket) order)`.
type ShuffledEpoch = (usize, Vec<(usize, usize)>);

impl BucketSchedule {
    /// Shared Fisher–Yates engine for versioned, domain-separated rectangular schedules.
    /// The legacy square shuffle keeps its historical seed construction unchanged.
    pub(crate) fn shuffle_items<T>(items: &mut [T], seed: u64, epoch: u64, domain: u64) {
        let mut state = seed
            .wrapping_add(epoch.wrapping_mul(0xD6E8_FEB8_6659_FD93))
            .wrapping_add(domain.wrapping_mul(0xA076_1D64_78BD_642F));
        for i in (1..items.len()).rev() {
            let j = (splitmix64_next(&mut state) % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }

    /// Build the schedule for `n_items` cached items over `buckets` (from
    /// [`TrainingConfig::training_buckets`]).
    pub fn new(n_items: usize, buckets: &[ResolutionBucket], seed: u64) -> Self {
        let n_buckets = buckets.len().max(1);
        let mut epoch = Vec::new();
        if n_buckets > 1 {
            for (b, bucket) in buckets.iter().enumerate() {
                for item in 0..n_items {
                    for _ in 0..bucket.repeats {
                        epoch.push((item, b));
                    }
                }
            }
        }
        Self {
            n_items,
            n_buckets,
            seed,
            epoch,
            shuffled: std::cell::RefCell::new(None),
        }
    }

    /// Number of buckets each item is cached at (the cache stride).
    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }

    /// Samples per epoch (`n_items` for one bucket; `n_items · Σ repeats` otherwise).
    pub fn epoch_len(&self) -> usize {
        if self.n_buckets == 1 {
            self.n_items
        } else {
            self.epoch.len()
        }
    }

    /// The `(item, bucket)` the `k`-th sample (0-based) trains on.
    ///
    /// # Panics
    /// When the schedule is empty (no items); every trainer refuses an empty cache before looping.
    pub fn sample(&self, k: usize) -> (usize, usize) {
        let len = self.epoch_len();
        assert!(len > 0, "BucketSchedule::sample on an empty schedule");
        if self.n_buckets == 1 {
            return (k % len, 0);
        }
        let (epoch_idx, pos) = (k / len, k % len);
        let mut shuffled = self.shuffled.borrow_mut();
        if let Some((cached_epoch, order)) = shuffled.as_ref() {
            if *cached_epoch == epoch_idx {
                return order[pos];
            }
        }
        let order = self.shuffle_epoch(epoch_idx);
        let sample = order[pos];
        *shuffled = Some((epoch_idx, order));
        sample
    }

    /// Epoch `epoch_idx`'s order: Fisher–Yates over a splitmix64 stream seeded from (job seed,
    /// epoch index) — a pure function of both, so resume and re-runs see the same order.
    fn shuffle_epoch(&self, epoch_idx: usize) -> Vec<(usize, usize)> {
        let mut order = self.epoch.clone();
        let mut state = self
            .seed
            .wrapping_add(0x5EED_B0C4_E7A0_0001)
            .wrapping_add((epoch_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
        for i in (1..order.len()).rev() {
            let j = (splitmix64_next(&mut state) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        order
    }

    /// The cache entry (`item * n_buckets + bucket`) the `k`-th sample (0-based) reads.
    pub fn cache_index(&self, k: usize) -> usize {
        let (item, bucket) = self.sample(k);
        item * self.n_buckets + bucket
    }
}

/// One step of a splitmix64 *stream* (advances `state`) — the bucket shuffle's RNG; the stateless
/// [`splitmix64`] hash serves the technique-noise keys.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
