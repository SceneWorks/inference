//! **AdaLN precompute + evict** (sc-17145) — the single biggest memory lever in the MiniMax-H3
//! port.
//!
//! # The arithmetic
//!
//! Every block carries `adaln_proj.linear`, a `[96768, 2688]` weight plus its bias:
//! `6 · MODALITY_NUM · hidden_size` columns from `time_embed_dim`. That is 260_209_152 parameters
//! per block, **13.01 B over the 50-block stack — 26_020_915_200 B (26.02 GB) at bf16**, out of the
//! DiT's ~33 B / 62 GB.
//!
//! Those weights are a function of the **timestep embedding only** — never of the tokens. So the
//! whole schedule's modulation can be projected up front and the projections then released:
//! denoise-resident drops from ~62 GB to ~36 GB at bf16, proportionally at q8/q4. This is what the
//! model card means by "AdaLN parameters pre-cacheable, reducing inference load".
//!
//! ```text
//! ┌ before denoise ────────────────────────────────────────────────────────────┐
//! │  embed(schedule.distinct_timesteps())  ->  temb [T, time_embed_dim]        │
//! │  per block: adaln_proj(temb)           ->  6 × [T · MODALITY_NUM, hidden]  │
//! │  FORCE EVALUATION                          <- see "the lazy-eval trap"     │
//! │  take + drop every adaln_proj                                              │
//! │  drain the mlx allocator cache             <- the explicit drain, retried  │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # The timestep axis is per ROW, not per step
//!
//! MiniMax-H3's packed sequence carries **several different timesteps simultaneously** inside one
//! forward pass: the video rows at the video schedule's `t`, the audio rows at the audio schedule's
//! `t` (a different sigma shift — 3.0 against 12.0), the `fl2va` conditioning rows pinned at
//! `max(video_t, 0.999)`, and the `ref2va` reference-soundtrack rows at a clean `1.0`.
//! `t = 1 − σ` on a unit scale, with `t = 1` meaning clean — the opposite direction from every
//! other flow-match family in-tree.
//!
//! **The text rows sit at the VIDEO timestep**, not at `1.0`. This module and the sc-17242 spike
//! comment (sc-17146 activity 18717) both originally recorded them as clean; sc-17146 read
//! `build_row_timesteps` and found that `row_timesteps` is *filled* with the video timestep and the
//! text rows are then never reassigned, so they keep it. The `1.0` class is real but belongs to
//! `ref2va`, which `t2va` and `fl2va` have no rows of. See [`crate::denoise::packing`], whose
//! `text_rows_carry_the_video_timestep` pins the corrected reading against the reference's own
//! golden.
//!
//! So the cache is **not** "one row per step". It is one modulation row per
//! `(distinct timestep, modality)` pair over the whole run, and [`TimestepSchedule`] is what turns a
//! per-step list of distinct timesteps into that global table plus the per-step remap into it.
//!
//! Deduplicating across steps is not tidiness — it is roughly a 2× saving on the cache, because the
//! keyframe (0.999) and reference-audio (1.0) rows are the *same* timestep at every step while only
//! the video and audio ones move. See [`TimestepSchedule::distinct_timesteps`].
//!
//! # The lazy-eval trap
//!
//! MLX is lazily evaluated. `adaln_proj(temb)` returns a *graph node*, and that node holds a
//! reference to `adaln_proj.weight`. Dropping the block's `AdaLnProjection` before the modulation
//! has been evaluated therefore frees **nothing**: the buffer stays alive through the pending graph,
//! and the first denoise step silently re-materializes it — at which point the peak is worse than if
//! nothing had been evicted at all. A timer wrapped around an unforced precompute reads ~0 for the
//! same reason.
//!
//! [`AdaLnCache::precompute`] therefore calls [`mlx_rs::transforms::eval`] over all
//! `6 · num_layers` tables **before** returning, and [`AdaLnCache::precompute_and_evict`] only
//! takes the projections after that call has returned.
//!
//! # What the two memory measurements each prove
//!
//! Neither one alone is evidence, which is why the eviction path performs both.
//!
//! | measurement | proves | does NOT prove |
//! |---|---|---|
//! | [`mlx_rs::memory::get_active_memory`] / [`get_peak_memory`](mlx_rs::memory::get_peak_memory) drops by the projections' bytes | no live MLX array (and no pending graph node) references those buffers any more | that the memory left the process — MLX's allocator retains freed buffers in its own cache, so "active" can fall while RSS does not move at all |
//! | [`mlx_rs::memory::get_cache_memory`] is still small after [`clear_cache`](mlx_rs::memory::clear_cache) | the freed bytes were returned to the system allocator rather than merely migrating active → cache | nothing about *which* buffers were released |
//!
//! A dropped Rust handle proves neither. The spike's torch/MPS finding — a 62 GB conditioner that
//! `.to("cpu")`, `del`, `gc.collect()` and `torch.mps.empty_cache()` all failed to release inside a
//! process — is the reason "the memory actually went away" is treated here as a claim needing
//! evidence rather than an assumption. `tests/adaln_evict_memory.rs` measures both quantities on a
//! synthetic stack, and `tests/adaln_evict_real_weights.rs` measures them on the real 62 GB
//! `transformer/`.
//!
//! # Samplers whose evaluation timesteps are not knowable up front
//!
//! Eviction is **irreversible within a run**: once the projections are gone, servicing an
//! unforeseen timestep would mean re-reading 26 GB from disk mid-denoise, which costs more than the
//! render. Invalidate-and-rebuild is therefore not a real option, and the decision recorded here is
//! to **exclude** such samplers — see [`AdaLnResidency`].
//!
//! The distinction that matters is *not* "adaptive or not" but **whether the set of timesteps the
//! model is evaluated at is a pure function of the declared σ grid**. Of the ten curated in-tree
//! solvers, nine evaluate only at grid points — including Heun, whose second evaluation is at
//! `s_next` (`gen-core/src/sampling/solvers.rs:126`). Exactly one does not: `dpmpp_sde` takes a
//! stochastic midpoint and evaluates at `sigma_s = sigma_of(t + h·R)`
//! (`gen-core/src/sampling/solvers.rs:243-256`), which is **not** a grid point. It is still a pure
//! function of the grid, so it remains precomputable — but only if the enumeration includes those
//! midpoints. A port that enumerated the grid alone would build a cache that is *missing rows* at
//! exactly the second evaluation of every step.
//!
//! That is why [`TimestepSchedule::index_of`] resolves timesteps by **exact value** and returns a
//! typed error for one it was not given, instead of gathering a neighbouring row. A mis-enumerated
//! schedule fails at its first unlisted evaluation rather than computing plausible garbage.

use std::collections::HashMap;

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::{Error, Result};

use crate::dit::block::{AdaLnModulation, DitBlock};
use crate::dit::config::MODALITY_NUM;

/// A compact digest of a [`TimestepSchedule`], for deciding whether a held [`AdaLnCache`] is still
/// current for an incoming request.
///
/// Two independent 64-bit FNV-1a streams (different offset bases) over the step count, each step's
/// length, and the **exact bit pattern** of every timestep. Bit patterns rather than values because
/// a cache row is only reusable for a timestep that is bitwise the one it was built from — see
/// [`TimestepSchedule::index_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScheduleKey(u64, u64);

impl ScheduleKey {
    fn of(steps: &[Vec<f32>]) -> Self {
        // FNV-1a, and the same stream with a different offset basis, so a collision needs both to
        // agree. The prime is FNV's 64-bit prime.
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut a: u64 = 0xcbf2_9ce4_8422_2325;
        let mut b: u64 = 0x9dcf_1a4e_63f0_925d;
        let mut feed = |word: u64| {
            for byte in word.to_le_bytes() {
                a = (a ^ u64::from(byte)).wrapping_mul(PRIME);
                b = (b ^ u64::from(byte)).wrapping_mul(PRIME);
            }
        };
        feed(steps.len() as u64);
        for step in steps {
            feed(step.len() as u64);
            for t in step {
                feed(u64::from(t.to_bits()));
            }
        }
        Self(a, b)
    }
}

impl std::fmt::Display for ScheduleKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}{:016x}", self.0, self.1)
    }
}

/// Whether the AdaLN projections may be evicted for a given run.
///
/// **The recorded decision for a sampler whose evaluation timesteps are not enumerable up front is
/// [`Resident`](Self::Resident) — exclude it from the eviction, do not try to invalidate.** Once
/// 26 GB of projections have been dropped and the allocator drained, rebuilding them means
/// re-reading the checkpoint mid-denoise; an invalidation path would be a memory cliff wearing a
/// correctness hat. Keeping the projections resident for those samplers costs the memory win and
/// keeps the numerics identical (both paths run the same [`crate::dit::block::AdaLnProjection`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaLnResidency {
    /// Every timestep the model will be evaluated at is enumerated in the [`TimestepSchedule`].
    /// Precompute the whole table and release the projections.
    PrecomputeAndEvict,
    /// The solver picks its evaluation points from the state of the run, so no finite schedule
    /// describes it. Keep `adaln_proj` loaded and project per step via
    /// [`DitBlock::forward_with_temb`]; the 26.02 GB stays resident.
    Resident,
}

/// Every timestep a whole denoise run evaluates at, plus each step's map from its **row classes**
/// into them.
///
/// A MiniMax-H3 packed sequence has a small number of *row classes*, each at its own timestep —
/// exactly what the reference's `build_row_timesteps` describes (`before_denoise.py`): the video
/// rows **and the text rows** at the video schedule's `t`, the audio rows at the audio schedule's
/// `t`, the `fl2va` conditioning rows at `max(video_t, 0.999)` and the `ref2va` reference-soundtrack
/// rows at `1.0`. Every row of the sequence belongs to one class, and its AdaLN table row is
/// `timestep_index · MODALITY_NUM + token_tag`. [`crate::denoise::packing::RowClass`] is that class
/// set, in the order this type is built with.
///
/// This type is constructed from **one timestep per row class per step**, in a caller-fixed class
/// order, and globalizes it: the per-step lists are unioned into one deduplicated table (first
/// appearance order) so the modulation is projected once per *distinct* timestep across the whole
/// run rather than once per `(step, class)`, and each step keeps a remap from class index to global
/// table row.
///
/// **Two classes may share a timestep**, and do: at `σ = 1` the video and audio schedules coincide
/// (both shifts map 1 to 1, so both `t` are 0) even though they diverge at every later step. Owning
/// the dedup here rather than asking callers to pre-deduplicate is what keeps the class → row
/// mapping stable across steps whose distinct-timestep count changes.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestepSchedule {
    steps: Vec<Vec<f32>>,
    distinct: Vec<f32>,
    remap: Vec<Vec<i32>>,
    lookup: HashMap<u32, i32>,
    key: ScheduleKey,
}

impl TimestepSchedule {
    /// Build a schedule from the per-step, per-row-class timesteps.
    ///
    /// `steps[i][c]` is row class `c`'s timestep at step `i`. The class order is the caller's and is
    /// preserved; it is what [`Self::adaln_indices`]' per-row class indices address.
    ///
    /// Rejects an empty schedule, an empty step, a non-finite timestep, and a step that declares a
    /// different number of row classes than the others — a caller that dropped the conditioning
    /// class at one step would otherwise shift every later class index by one. `-0.0` is normalized
    /// to `0.0` so the bitwise table cannot hold two rows that compare equal.
    pub fn new(steps: Vec<Vec<f32>>) -> Result<Self> {
        if steps.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 adaln schedule: a denoise schedule needs at least one step".into(),
            ));
        }
        let steps: Vec<Vec<f32>> = steps
            .into_iter()
            .map(|s| {
                s.into_iter()
                    .map(|t| if t == 0.0 { 0.0 } else { t })
                    .collect()
            })
            .collect();
        let classes = steps[0].len();
        if classes == 0 {
            return Err(Error::Msg(
                "minimax-h3 adaln schedule: step 0 declares no row classes; every packed sequence \
                 carries at least the text rows"
                    .into(),
            ));
        }

        let mut distinct: Vec<f32> = Vec::new();
        let mut lookup: HashMap<u32, i32> = HashMap::new();
        let mut remap: Vec<Vec<i32>> = Vec::with_capacity(steps.len());
        for (i, step) in steps.iter().enumerate() {
            if step.len() != classes {
                return Err(Error::Msg(format!(
                    "minimax-h3 adaln schedule: step {i} declares {} row classes but step 0 \
                     declares {classes}; the class order must be the same at every step or every \
                     per-row class index shifts",
                    step.len()
                )));
            }
            let mut row = Vec::with_capacity(step.len());
            for &t in step {
                if !t.is_finite() {
                    return Err(Error::Msg(format!(
                        "minimax-h3 adaln schedule: step {i} carries a non-finite timestep {t}"
                    )));
                }
                let global = *lookup.entry(t.to_bits()).or_insert_with(|| {
                    distinct.push(t);
                    (distinct.len() - 1) as i32
                });
                row.push(global);
            }
            remap.push(row);
        }

        let key = ScheduleKey::of(&steps);
        Ok(Self {
            steps,
            distinct,
            remap,
            lookup,
            key,
        })
    }

    /// Row classes every step declares — 4 for the shipped `fl2va` layout (video, audio,
    /// conditioning, text).
    pub fn num_row_classes(&self) -> usize {
        self.steps[0].len()
    }

    /// Denoise steps in the schedule.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// The deduplicated union of every step's row-class timesteps, in first-appearance order — one
    /// row block of the modulation table each.
    ///
    /// This is far shorter than `num_steps · num_row_classes` by construction: the conditioning
    /// rows' `0.999` and the text rows' `1.0` are the same value at *every* step, so only the video
    /// and audio timesteps actually move. A 20-step joint run has 80 `(step, class)` pairs and ~41
    /// distinct timesteps — a ~2× saving on the cache, before any of the 26.02 GB win.
    pub fn distinct_timesteps(&self) -> &[f32] {
        &self.distinct
    }

    /// `distinct_timesteps().len()`, as the `i32` MLX shapes are expressed in.
    pub fn num_distinct_timesteps(&self) -> i32 {
        self.distinct.len() as i32
    }

    /// Rows of the modulation table this schedule needs: `num_distinct_timesteps · MODALITY_NUM`.
    pub fn modulation_rows(&self) -> i32 {
        self.num_distinct_timesteps() * MODALITY_NUM
    }

    /// Step `i`'s timestep per row class, in the caller's class order.
    pub fn step_timesteps(&self, step: usize) -> Result<&[f32]> {
        self.steps
            .get(step)
            .map(Vec::as_slice)
            .ok_or_else(|| self.no_such_step(step))
    }

    /// Map row class `class` at step `step` to its row block in the global table.
    pub fn global_timestep_index(&self, step: usize, class: i32) -> Result<i32> {
        let row = self
            .remap
            .get(step)
            .ok_or_else(|| self.no_such_step(step))?;
        if class < 0 || class as usize >= row.len() {
            return Err(Error::Msg(format!(
                "minimax-h3 adaln schedule: row class {class} is outside step {step}'s {} declared \
                 classes",
                row.len()
            )));
        }
        Ok(row[class as usize])
    }

    /// Resolve a timestep **value** to its row block in the global table.
    ///
    /// Matching is on the exact bit pattern, and a value the schedule was not built with is an
    /// error rather than a nearest-row match. That strictness is the gate that catches a
    /// mis-enumerated schedule — the `dpmpp_sde` midpoint case in the module docs — at the first
    /// unlisted evaluation instead of letting it gather a plausible neighbouring row. Callers
    /// should take their timesteps *from* the schedule ([`Self::step_timesteps`]) rather than
    /// recomputing them, so this stays a defensive check rather than a routine lookup.
    pub fn index_of(&self, timestep: f32) -> Result<i32> {
        let t = if timestep == 0.0 { 0.0 } else { timestep };
        self.lookup.get(&t.to_bits()).copied().ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 adaln schedule: timestep {timestep} is not in this schedule's {} \
                 declared timesteps, so the precomputed modulation has no row for it. The AdaLN \
                 projections were evicted and cannot be re-run: enumerate EVERY timestep the \
                 sampler evaluates at (a midpoint solver evaluates off the sigma grid), or load \
                 with AdaLnResidency::Resident",
                self.distinct.len()
            ))
        })
    }

    /// The digest an [`AdaLnCache`] is keyed by.
    pub fn key(&self) -> ScheduleKey {
        self.key
    }

    /// Build step `step`'s AdaLN gather index for the packed sequence.
    ///
    /// `row_classes` is one entry per sequence row — the class that row belongs to, in the same
    /// order [`Self::new`] was given; `token_tags` is the per-row modality tag (video 0, text 1,
    /// audio 2). The result addresses the **global** table the cache holds:
    /// `global_timestep_index · MODALITY_NUM + tag`.
    ///
    /// Both inputs are bounds-checked here because **MLX does not bounds-check a gather** — an
    /// index past the end reads whatever is adjacent in the buffer and the block computes silent
    /// garbage. This is the one place the class → global remap happens, so it is the one place that
    /// can catch a stale step index.
    pub fn adaln_indices(
        &self,
        step: usize,
        row_classes: &Array,
        token_tags: &Array,
    ) -> Result<Array> {
        let row = self
            .remap
            .get(step)
            .ok_or_else(|| self.no_such_step(step))?;
        if row_classes.shape() != token_tags.shape() {
            return Err(Error::Msg(format!(
                "minimax-h3 adaln schedule: row_classes {:?} and token_tags {:?} must be one entry \
                 per sequence row",
                row_classes.shape(),
                token_tags.shape()
            )));
        }
        let classes = row_classes.as_dtype(mlx_rs::Dtype::Int32)?;
        let tags = token_tags.as_dtype(mlx_rs::Dtype::Int32)?;
        bounded(&classes, row.len() as i32, "row class", step)?;
        bounded(&tags, MODALITY_NUM, "token tag", step)?;

        let table = Array::from_slice(row, &[row.len() as i32]);
        let global = table.take_axis(&classes, 0)?;
        Ok(add(
            &multiply(&global, Array::from_int(MODALITY_NUM))?,
            &tags,
        )?)
    }

    fn no_such_step(&self, step: usize) -> Error {
        Error::Msg(format!(
            "minimax-h3 adaln schedule: step {step} is outside this schedule's {} steps",
            self.steps.len()
        ))
    }
}

/// Drain attempts [`drain_allocator_cache`] makes before giving up. 8 × 2 ms bounds the whole drain
/// at ~16 ms, against a denoise measured in minutes.
const DRAIN_ATTEMPTS: usize = 8;

/// Return the evicted buffers to the system allocator, retrying while MLX's cache keeps refilling.
///
/// **One [`mlx_rs::memory::clear_cache`] is not reliably enough.** A buffer reaches MLX's allocator
/// cache when its last reference drops, but the Metal command buffer that used it retains its
/// resources until it is retired, and that retirement is not guaranteed to have happened by the
/// time [`mlx_rs::transforms::eval`] returns. A single drain can therefore run *before* some of the
/// weights have been handed back, and those land in the cache with nothing left to sweep them.
///
/// Measured on a synthetic 8-block stack under concurrent test load: a single drain left one
/// block's 18 MiB projection still counted as **active** — its last real reference being an
/// unretired command buffer — in roughly one run in five, while the same measurement in isolation
/// released all 8 every time.
///
/// The loop therefore watches **active**, not the cache. Stopping as soon as the cache reads empty
/// would return on the first pass — the seven already-freed buffers sweep cleanly and the cache
/// *is* empty — leaving the straggler to reach the cache a moment later with nothing left to sweep
/// it. Stopping when active stops falling is the condition that actually describes "the runtime has
/// finished handing buffers back".
///
/// This cannot mask a real leak: a buffer that is still referenced never reaches the cache at all,
/// so no amount of draining releases it, and the loop exits after two passes having freed nothing.
/// `tests/adaln_evict_memory.rs`'s control arm is the proof — with the forced evaluation removed
/// the drain frees 0.0 of 144.3 MiB, however many times it runs.
fn drain_allocator_cache() {
    let mut previous = usize::MAX;
    for _ in 0..DRAIN_ATTEMPTS {
        mlx_rs::memory::clear_cache();
        let active = mlx_rs::memory::get_active_memory();
        if active >= previous {
            return;
        }
        previous = active;
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    mlx_rs::memory::clear_cache();
}

/// Reject an index tensor that would gather out of bounds. See [`TimestepSchedule::adaln_indices`].
fn bounded(idx: &Array, limit: i32, what: &str, step: usize) -> Result<()> {
    let min: i32 = idx.min(None)?.item();
    let max: i32 = idx.max(None)?.item();
    if min < 0 || max >= limit {
        return Err(Error::Msg(format!(
            "minimax-h3 adaln schedule: {what} range [{min}, {max}] at step {step} is outside \
             [0, {limit}); MLX gathers out of bounds silently rather than failing"
        )));
    }
    Ok(())
}

/// The precomputed AdaLN modulation for a whole denoise run — one table per block, covering every
/// distinct timestep the run will evaluate at.
///
/// Built by [`Self::precompute_and_evict`], which is also what releases the 26.02 GB of
/// `adaln_proj` weights. See the module docs for the lazy-eval ordering that makes the release real.
#[derive(Debug, Clone)]
pub struct AdaLnCache {
    schedule: TimestepSchedule,
    layers: Vec<AdaLnModulation>,
    bytes: usize,
}

impl AdaLnCache {
    /// Project the schedule's modulation for every block, forcing evaluation before returning.
    ///
    /// `embed` is the timestep MLP — `time_embedder(time_proj(t))`, which the DiT wrapper owns
    /// (sc-17147) — applied to [`TimestepSchedule::distinct_timesteps`]. It is a closure rather
    /// than a `temb` argument so that the embedding is taken from the schedule's own timesteps and
    /// cannot be mis-bound to a different one.
    ///
    /// Does **not** evict; [`Self::precompute_and_evict`] is the memory lever.
    pub fn precompute<F>(blocks: &[DitBlock], schedule: TimestepSchedule, embed: F) -> Result<Self>
    where
        F: FnOnce(&[f32]) -> Result<Array>,
    {
        if blocks.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 adaln cache: the block stack is empty".into(),
            ));
        }
        let first = blocks[0].adaln_proj().ok_or_else(|| {
            Error::Msg(
                "minimax-h3 adaln cache: block 0's adaln_proj has already been evicted; a cache \
                 can only be built once per load"
                    .into(),
            )
        })?;
        let time_embed_dim = first.time_embed_dim();

        let temb = embed(schedule.distinct_timesteps())?;
        let want = [schedule.num_distinct_timesteps(), time_embed_dim];
        if temb.shape() != want {
            return Err(Error::Msg(format!(
                "minimax-h3 adaln cache: the timestep embedding must be {want:?} — one row per \
                 distinct timestep at the projection's time_embed_dim — got {:?}",
                temb.shape()
            )));
        }

        let mut layers = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            layers.push(
                block
                    .modulation(&temb)
                    .map_err(|e| Error::Msg(format!("minimax-h3 adaln cache: layer {i}: {e}")))?,
            );
        }

        // FORCE EVALUATION. Until this returns, every table above is an un-evaluated graph node
        // holding a reference to its block's `adaln_proj.weight`, so dropping the projection would
        // free nothing at all and the first denoise step would re-materialize all 26 GB. This is
        // also the only point at which the precompute's cost is actually incurred — a timer that
        // does not span this call reads ~0.
        let flat: Vec<&Array> = layers.iter().flat_map(AdaLnModulation::tables).collect();
        mlx_rs::transforms::eval(flat)?;

        let bytes = layers
            .iter()
            .map(|m| m.tables().map(Array::nbytes).sum::<usize>())
            .sum();
        Ok(Self {
            schedule,
            layers,
            bytes,
        })
    }

    /// [`Self::precompute`], then release every block's AdaLN projection and drain MLX's allocator
    /// cache — the whole 26.02 GB lever in one call.
    ///
    /// `residency` records the caller's decision. [`AdaLnResidency::Resident`] is **rejected**
    /// here: it is a valid choice, but it is the choice *not* to build a cache, and silently
    /// building one anyway would evict weights a run-state-dependent sampler still needs. Callers
    /// in that mode keep the projections and use [`DitBlock::forward_with_temb`].
    ///
    /// The order is load-bearing and is the reason this exists as one function rather than three
    /// steps at the call site:
    ///
    /// 1. project + **force evaluation** (so nothing references the weights through a pending
    ///    graph),
    /// 2. take the projection out of every block and drop it (so nothing references the weights
    ///    through a Rust handle),
    /// 3. drain MLX's allocator cache (so the freed buffers go back to the system allocator instead
    ///    of sitting in MLX's own cache, where `get_active_memory` would report them as released
    ///    while RSS had not moved — and retried, because the Metal backend does not always hand
    ///    them back before the first drain runs).
    ///
    /// Returns the cache and the bytes of projection weight released.
    pub fn precompute_and_evict<F>(
        blocks: &mut [DitBlock],
        schedule: TimestepSchedule,
        residency: AdaLnResidency,
        embed: F,
    ) -> Result<(Self, usize)>
    where
        F: FnOnce(&[f32]) -> Result<Array>,
    {
        if residency != AdaLnResidency::PrecomputeAndEvict {
            return Err(Error::Msg(
                "minimax-h3 adaln cache: AdaLnResidency::Resident does not precompute — the \
                 projections must stay loaded for a sampler whose evaluation timesteps are not \
                 enumerable up front, because eviction cannot be undone without re-reading the \
                 checkpoint. Call DitBlock::forward_with_temb per step instead"
                    .into(),
            ));
        }

        let cache = Self::precompute(blocks, schedule, embed)?;

        let mut released = 0usize;
        for block in blocks.iter_mut() {
            if let Some(proj) = block.evict_adaln() {
                released += proj.nbytes();
                drop(proj);
            }
        }
        // The explicit drain. `get_active_memory` falling is not on its own evidence that anything
        // left the process — without this the bytes migrate from active into MLX's allocator cache.
        drain_allocator_cache();

        Ok((cache, released))
    }

    /// The modulation tables for block `layer`.
    pub fn modulation(&self, layer: usize) -> Result<&AdaLnModulation> {
        self.layers.get(layer).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 adaln cache: layer {layer} is outside the cached stack's {} blocks",
                self.layers.len()
            ))
        })
    }

    /// Blocks this cache covers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The schedule this cache was built for.
    pub fn schedule(&self) -> &TimestepSchedule {
        &self.schedule
    }

    /// Device bytes the cache itself occupies — `6 · modulation_rows · hidden_size · num_layers`
    /// elements at the block dtype.
    ///
    /// Independent of resolution and of duration by construction: nothing here has a token axis.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether this cache can serve `schedule`, i.e. whether the schedule is bitwise the one it was
    /// built from. A changed step count, a changed sigma shift or a changed solver all produce a
    /// different [`ScheduleKey`] and therefore a `false` here.
    pub fn is_current_for(&self, schedule: &TimestepSchedule) -> bool {
        self.schedule.key() == schedule.key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::config::MODULATION_PARAMS;

    /// A joint run's four row classes at each of `evals` model evaluations: video `t` (which the
    /// text rows share), audio `t` (a different sigma shift), the `fl2va` conditioning rows at
    /// `max(video_t, 0.999)` and the `ref2va` reference-soundtrack rows at `1.0`.
    ///
    /// A hand-rolled stand-in for [`crate::denoise::JointSchedule`], kept so this module's tests
    /// stay independent of the denoise loop.
    ///
    /// `σ` descends from 1 but never reaches 0: `MiniMaxH3Scheduler` includes the terminal zero in
    /// `num_inference_steps` and the loop runs one evaluation fewer, so the model is never
    /// evaluated at `σ = 0`.
    fn joint_schedule(evals: usize) -> Vec<Vec<f32>> {
        (0..evals)
            .map(|i| {
                let sigma = 1.0 - (i as f32) / (evals as f32);
                let video_t = 1.0 - shift(sigma, 12.0);
                let audio_t = 1.0 - shift(sigma, 3.0);
                vec![video_t, audio_t, video_t.max(0.999), 1.0]
            })
            .collect()
    }

    /// `σ' = s·σ / (1 + (s−1)·σ)`.
    fn shift(sigma: f32, s: f32) -> f32 {
        s * sigma / (1.0 + (s - 1.0) * sigma)
    }

    /// The union is much shorter than the concatenation, because two of the four row classes are the
    /// SAME timestep at every step. That is the reason the cache is keyed on distinct timesteps
    /// rather than on `(step, row class)`.
    #[test]
    fn the_constant_row_classes_are_deduplicated_across_steps() {
        let steps = joint_schedule(20);
        let total: usize = steps.iter().map(Vec::len).sum();
        assert_eq!(total, 80);
        let s = TimestepSchedule::new(steps).unwrap();
        assert_eq!(s.num_steps(), 20);
        // 20 video + 20 audio + `1.0` (text, and the keyframe max for all but the last steps)…
        assert!(
            s.num_distinct_timesteps() < 45,
            "expected the constant classes to collapse, got {}",
            s.num_distinct_timesteps()
        );
        assert_eq!(
            s.modulation_rows(),
            s.num_distinct_timesteps() * MODALITY_NUM
        );
    }

    /// A different step count and a different sigma shift must each produce a different key —
    /// otherwise a held cache would be silently reused across schedules.
    #[test]
    fn the_key_separates_step_count_and_schedule_shape() {
        let a = TimestepSchedule::new(joint_schedule(20)).unwrap();
        let b = TimestepSchedule::new(joint_schedule(20)).unwrap();
        let c = TimestepSchedule::new(joint_schedule(4)).unwrap();
        assert_eq!(a.key(), b.key(), "the same schedule must key identically");
        assert_ne!(a.key(), c.key(), "a changed step count must re-key");

        // Same step count, one timestep nudged by a single ULP.
        let mut nudged = joint_schedule(20);
        let t = nudged[3][0];
        nudged[3][0] = f32::from_bits(t.to_bits() + 1);
        let d = TimestepSchedule::new(nudged).unwrap();
        assert_ne!(a.key(), d.key(), "a one-ULP change must re-key");
    }

    /// The class→global remap: the video class addresses a different global row at every step,
    /// while the text class addresses the same one throughout.
    #[test]
    fn row_classes_remap_per_step() {
        let s = TimestepSchedule::new(joint_schedule(4)).unwrap();
        assert_eq!(s.num_row_classes(), 4);
        assert_ne!(
            s.global_timestep_index(0, 0).unwrap(),
            s.global_timestep_index(1, 0).unwrap(),
            "the video timestep moves between steps"
        );
        let text = s.global_timestep_index(0, 3).unwrap();
        for step in 0..s.num_steps() {
            assert_eq!(s.global_timestep_index(step, 3).unwrap(), text);
        }
        assert!(s.global_timestep_index(0, 4).is_err(), "out of range class");
        assert!(s.global_timestep_index(99, 0).is_err(), "out of range step");
    }

    /// At `σ = 1` the two sigma shifts coincide, so the video and audio classes share a timestep.
    /// That must dedup to ONE table row while both classes keep their own stable class index.
    #[test]
    fn two_row_classes_may_share_a_timestep() {
        let s = TimestepSchedule::new(joint_schedule(4)).unwrap();
        let step0 = s.step_timesteps(0).unwrap();
        assert_eq!(step0[0], step0[1], "σ=1 maps to t=0 under both shifts");
        assert_eq!(
            s.global_timestep_index(0, 0).unwrap(),
            s.global_timestep_index(0, 1).unwrap(),
            "one table row serves both classes"
        );
        assert_ne!(
            s.global_timestep_index(1, 0).unwrap(),
            s.global_timestep_index(1, 1).unwrap(),
            "…and they diverge at the next step"
        );
    }

    /// A timestep the schedule was not built with is an ERROR, not a nearest-row match. This is the
    /// gate for a mis-enumerated schedule — a midpoint solver evaluating off the sigma grid.
    #[test]
    fn an_unlisted_timestep_is_rejected_rather_than_rounded() {
        let s = TimestepSchedule::new(joint_schedule(4)).unwrap();
        let listed = s.distinct_timesteps()[0];
        assert_eq!(s.index_of(listed).unwrap(), 0);

        // One ULP away — a "close enough" lookup would silently return row 0.
        let nudged = f32::from_bits(listed.to_bits() + 1);
        let e = s.index_of(nudged).unwrap_err().to_string();
        assert!(e.contains("not in this schedule"), "{e}");
        assert!(
            e.contains("Resident"),
            "the error must name the way out: {e}"
        );
    }

    /// `-0.0` and `0.0` compare equal but have different bit patterns; the table must not hold two
    /// rows for them.
    #[test]
    fn negative_zero_is_normalized() {
        let s = TimestepSchedule::new(vec![vec![0.0, 1.0], vec![-0.0, 1.0]]).unwrap();
        assert_eq!(s.num_distinct_timesteps(), 2);
        assert_eq!(s.index_of(-0.0).unwrap(), s.index_of(0.0).unwrap());
    }

    #[test]
    fn malformed_schedules_are_rejected() {
        assert!(TimestepSchedule::new(vec![]).is_err(), "empty schedule");
        assert!(
            TimestepSchedule::new(vec![vec![]]).is_err(),
            "no row classes"
        );
        assert!(
            TimestepSchedule::new(vec![vec![f32::NAN]]).is_err(),
            "non-finite timestep"
        );
        // A step that dropped a row class would shift every later class index by one.
        let e = TimestepSchedule::new(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5]])
            .unwrap_err()
            .to_string();
        assert!(e.contains("row classes"), "a ragged schedule: {e}");
    }

    /// `global · MODALITY_NUM + tag`, with both inputs bounds-checked because MLX gathers out of
    /// bounds silently.
    #[test]
    fn adaln_indices_compose_the_global_row() {
        let s = TimestepSchedule::new(joint_schedule(4)).unwrap();
        // Three rows: a video row (class 0), a text row (class 3), an audio row (class 1).
        let classes = Array::from_slice(&[0i32, 3, 1], &[3]);
        let tags = Array::from_slice(&[0i32, 1, 2], &[3]);
        let got = s.adaln_indices(1, &classes, &tags).unwrap();
        let want = [
            s.global_timestep_index(1, 0).unwrap() * MODALITY_NUM,
            s.global_timestep_index(1, 3).unwrap() * MODALITY_NUM + 1,
            s.global_timestep_index(1, 1).unwrap() * MODALITY_NUM + 2,
        ];
        assert_eq!(got.as_slice::<i32>(), &want);
        assert!(got.max(None).unwrap().item::<i32>() < s.modulation_rows());

        for (label, classes, tags) in [
            (
                "class past the step",
                Array::from_slice(&[9i32], &[1]),
                Array::from_slice(&[0i32], &[1]),
            ),
            (
                "tag beyond MODALITY_NUM",
                Array::from_slice(&[0i32], &[1]),
                Array::from_slice(&[MODALITY_NUM], &[1]),
            ),
            (
                "negative tag",
                Array::from_slice(&[0i32], &[1]),
                Array::from_slice(&[-1i32], &[1]),
            ),
        ] {
            assert!(
                s.adaln_indices(0, &classes, &tags).is_err(),
                "{label} must be rejected"
            );
        }
        // Mismatched lengths are a caller bug, not a broadcast.
        assert!(s
            .adaln_indices(
                0,
                &Array::from_slice(&[0i32, 1], &[2]),
                &Array::from_slice(&[0i32], &[1])
            )
            .is_err());
    }

    /// The cache-size claim: `6 · rows · hidden` per layer, and **no dependence on resolution or
    /// duration** — nothing in the table has a token axis.
    #[test]
    fn cache_bytes_are_independent_of_resolution_and_duration() {
        // 42 distinct timesteps at 20 steps × 3 modalities × 6 params × 5376 hidden × 50 layers.
        let s = TimestepSchedule::new(joint_schedule(20)).unwrap();
        let rows = s.modulation_rows() as usize;
        let bf16 = 2;
        let bytes = MODULATION_PARAMS as usize * rows * 5376 * 50 * bf16;
        assert!(
            (380..=460).contains(&(bytes / (1000 * 1000))),
            "expected ~400 MB at 20 steps, got {} MB",
            bytes / (1000 * 1000)
        );
        // The 26.02 GB it replaces.
        let evicted = 50usize * (96_768 * 2688 + 96_768) * bf16;
        assert_eq!(evicted, 26_020_915_200);
        assert!(evicted / bytes > 50, "the trade must be lopsided");
    }
}
