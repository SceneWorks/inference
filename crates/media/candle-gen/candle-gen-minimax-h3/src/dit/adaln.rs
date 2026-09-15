//! **AdaLN precompute + evict** — the single biggest memory lever in the MiniMax-H3 port, and the
//! place where this lane's answer genuinely differs from the MLX sibling's.
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
//! # The timestep axis is per ROW, not per step
//!
//! MiniMax-H3's packed sequence carries **several different timesteps simultaneously** inside one
//! forward pass: the video rows at the video schedule's `t`, the audio rows at the audio schedule's
//! `t` (a different sigma shift — 3.0 against 12.0), the `fl2va` conditioning rows pinned at
//! `max(video_t, 0.999)`, and the `ref2va` reference-soundtrack rows at a clean `1.0`.
//! `t = 1 − σ` on a unit scale, with `t = 1` meaning clean — the opposite direction from every
//! other flow-match family in-tree.
//!
//! **The text rows sit at the VIDEO timestep**, not at `1.0`: `build_row_timesteps` *fills*
//! `row_timesteps` with the video timestep and the text rows are then never reassigned, so they
//! keep it. See [`crate::denoise::packing`].
//!
//! So the cache is **not** "one row per step". It is one modulation row per
//! `(distinct timestep, modality)` pair over the whole run, and [`TimestepSchedule`] is what turns a
//! per-step list of distinct timesteps into that global table plus the per-step remap into it.
//!
//! Deduplicating across steps is not tidiness — it is roughly a 2× saving on the cache, because the
//! keyframe (0.999) and reference-audio (1.0) rows are the *same* timestep at every step while only
//! the video and audio ones move. See [`TimestepSchedule::distinct_timesteps`].
//!
//! # What "released" means on candle, and how it differs from MLX
//!
//! This is the part that does **not** transfer from sc-17145, and getting it wrong would leave a
//! green test proving nothing.
//!
//! **MLX is lazily evaluated.** `adaln_proj(temb)` returns a graph node holding a reference to
//! `adaln_proj.weight`, so the MLX path must force evaluation *before* dropping, then explicitly
//! drain MLX's allocator cache (retried, because the Metal command buffer may not have retired).
//! Neither step has a counterpart here:
//!
//! * **candle is eager.** `AdaLnProjection::forward` has already computed and materialized the
//!   modulation by the time it returns. There is no pending node, so there is nothing to force, and
//!   a "force evaluation" step would be a no-op wearing a correctness hat.
//! * **candle has no user-visible allocator cache to drain.** Its CUDA allocations go through
//!   `cuMemAllocAsync` on the device's stream-ordered pool, whose *release threshold is 0* — the
//!   driver default, which neither candle nor `candle-gen` raises. `MemPool::trim` therefore has no
//!   retained cache to return and is a measured **no-op**; it must not be credited with any
//!   recovery (SC-15791).
//!
//! What candle needs instead is a **synchronize**, and what it is exposed to instead is
//! **`Arc` sharing**:
//!
//! | step | why it is required here |
//! |---|---|
//! | drop every last `Tensor` handle to the projection | a candle `Tensor` is an `Arc` over its storage. `Weights::require` returns a **clone sharing that storage**, and `Tensor::to_dtype` is a no-op clone when the dtype already matches — so a surviving loader map keeps all 26.02 GB resident while every block reports itself evicted |
//! | [`Device::synchronize`](candle_gen::candle_core::Device::synchronize) | at a release threshold of 0 the pool hands pages back to the driver **on synchronize**. Measured in SC-15791: `pool reserved: live 160.0 │ after drop 160.0 │ after drop+sync 0.0 MiB` — a bare drop recovers *nothing* driver-visible |
//!
//! ## What each counter proves, and what it does not
//!
//! | counter | proves | does NOT prove |
//! |---|---|---|
//! | a counting `#[global_allocator]`'s live bytes falling (CPU device) | the storage was really returned to the system allocator — candle's CPU storage is a plain Rust allocation, so this is exact and transient-free | anything about a CUDA device |
//! | `MemPool::used` falling (CUDA) | no live `Tensor` references those buffers | that the driver got anything back — `used` falls on a bare drop while `nvidia-smi` moves 0.0 MiB |
//! | `MemPool::reserved` falling (CUDA) | the pages went back to the driver, i.e. another process or another allocation can now have them. **This is the counter an admission gate reads** | which buffers were released |
//!
//! `tests/adaln_evict_memory.rs` measures the CPU arm with a counting global allocator and runs a
//! control arm that holds a surviving clone. **The CUDA arm is written but is not executed by any
//! lane this story can reach** — see that file's module docs for exactly what that does and does not
//! establish.
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
//! `s_next`. Exactly one does not: `dpmpp_sde` takes a stochastic midpoint and evaluates at
//! `sigma_s = sigma_of(t + h·R)`, which is **not** a grid point. It is still a pure function of the
//! grid, so it remains precomputable — but only if the enumeration includes those midpoints. A port
//! that enumerated the grid alone would build a cache that is *missing rows* at exactly the second
//! evaluation of every step.
//!
//! That is why [`TimestepSchedule::index_of`] resolves timesteps by **exact value** and returns a
//! typed error for one it was not given, instead of gathering a neighbouring row. A mis-enumerated
//! schedule fails at its first unlisted evaluation rather than computing plausible garbage.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::dit::block::{AdaLnModulation, DitBlock};
use crate::dit::config::MODALITY_NUM;

/// A compact digest of a [`TimestepSchedule`], for deciding whether a held [`AdaLnCache`] is still
/// current for an incoming request.
///
/// Two independent 64-bit FNV-1a streams (different offset bases) over the step count, each step's
/// length, and the **exact bit pattern** of every timestep. Bit patterns rather than values because
/// a cache row is only reusable for a timestep that is bitwise the one it was built from — see
/// [`TimestepSchedule::index_of`].
///
/// **A future-seam hook, wired to nothing in production today.** [`crate::dit::JointDit`] builds a
/// fresh cache per request and consumes the schedule doing so, so there is no held-cache path that
/// could go stale — the staleness this key detects only becomes reachable if a cache is ever
/// retained across requests (the natural lever if per-request precompute cost ever matters). It is
/// kept, with [`AdaLnCache::is_current_for`], so that seam starts from a bitwise-exact currency
/// check instead of re-deriving one.
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
/// 26 GB of projections have been dropped, rebuilding them means re-reading the checkpoint
/// mid-denoise; an invalidation path would be a memory cliff wearing a correctness hat. Keeping the
/// projections resident for those samplers costs the memory win and keeps the numerics identical
/// (both paths run the same [`crate::dit::block::AdaLnProjection`]).
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
/// exactly what the reference's `build_row_timesteps` describes: the video rows **and the text
/// rows** at the video schedule's `t`, the audio rows at the audio schedule's `t`, the `fl2va`
/// conditioning rows at `max(video_t, 0.999)` and the `ref2va` reference-soundtrack rows at `1.0`.
/// Every row of the sequence belongs to one class, and its AdaLN table row is
/// `timestep_index · MODALITY_NUM + token_tag`. [`crate::denoise::packing::RowClass`] is that class
/// set, in the order this type is built with.
///
/// **Two classes may share a timestep**, and do: at `σ = 1` the video and audio schedules coincide
/// (both shifts map 1 to 1, so both `t` are 0) even though they diverge at every later step. Owning
/// the dedup here rather than asking callers to pre-deduplicate is what keeps the class → row
/// mapping stable across steps whose distinct-timestep count changes.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestepSchedule {
    steps: Vec<Vec<f32>>,
    distinct: Vec<f32>,
    remap: Vec<Vec<u32>>,
    lookup: HashMap<u32, u32>,
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
            return Err(CandleError::Msg(
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
            return Err(CandleError::Msg(
                "minimax-h3 adaln schedule: step 0 declares no row classes; every packed sequence \
                 carries at least the text rows"
                    .into(),
            ));
        }

        let mut distinct: Vec<f32> = Vec::new();
        let mut lookup: HashMap<u32, u32> = HashMap::new();
        let mut remap: Vec<Vec<u32>> = Vec::with_capacity(steps.len());
        for (i, step) in steps.iter().enumerate() {
            if step.len() != classes {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 adaln schedule: step {i} declares {} row classes but step 0 \
                     declares {classes}; the class order must be the same at every step or every \
                     per-row class index shifts",
                    step.len()
                )));
            }
            let mut row = Vec::with_capacity(step.len());
            for &t in step {
                if !t.is_finite() {
                    return Err(CandleError::Msg(format!(
                        "minimax-h3 adaln schedule: step {i} carries a non-finite timestep {t}"
                    )));
                }
                let global = *lookup.entry(t.to_bits()).or_insert_with(|| {
                    distinct.push(t);
                    (distinct.len() - 1) as u32
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
    /// rows' `0.999` and the reference-audio rows' `1.0` are the same value at *every* step, so only
    /// the video and audio timesteps actually move. A 20-step joint run has 80 `(step, class)` pairs
    /// and ~41 distinct timesteps — a ~2× saving on the cache, before any of the 26.02 GB win.
    pub fn distinct_timesteps(&self) -> &[f32] {
        &self.distinct
    }

    /// `distinct_timesteps().len()`.
    pub fn num_distinct_timesteps(&self) -> usize {
        self.distinct.len()
    }

    /// Rows of the modulation table this schedule needs: `num_distinct_timesteps · MODALITY_NUM`.
    pub fn modulation_rows(&self) -> usize {
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
    pub fn global_timestep_index(&self, step: usize, class: usize) -> Result<u32> {
        let row = self
            .remap
            .get(step)
            .ok_or_else(|| self.no_such_step(step))?;
        row.get(class).copied().ok_or_else(|| {
            CandleError::Msg(format!(
                "minimax-h3 adaln schedule: row class {class} is outside step {step}'s {} declared \
                 classes",
                row.len()
            ))
        })
    }

    /// Resolve a timestep **value** to its row block in the global table.
    ///
    /// Matching is on the exact bit pattern, and a value the schedule was not built with is an
    /// error rather than a nearest-row match. That strictness is the gate that catches a
    /// mis-enumerated schedule — the `dpmpp_sde` midpoint case in the module docs — at the first
    /// unlisted evaluation instead of letting it gather a plausible neighbouring row. Callers
    /// should take their timesteps *from* the schedule ([`Self::step_timesteps`]) rather than
    /// recomputing them, so this stays a defensive check rather than a routine lookup.
    pub fn index_of(&self, timestep: f32) -> Result<u32> {
        let t = if timestep == 0.0 { 0.0 } else { timestep };
        self.lookup.get(&t.to_bits()).copied().ok_or_else(|| {
            CandleError::Msg(format!(
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
    /// `global_timestep_index · MODALITY_NUM + tag`, as `u32` — the dtype candle's `index_select`
    /// takes.
    ///
    /// Both inputs are bounds-checked here on the host. This is the one place the class → global
    /// remap happens, so it is the one place that can catch a stale step index, and a typed error
    /// naming the schedule is far more legible than whatever a backend kernel would produce.
    pub fn adaln_indices(
        &self,
        step: usize,
        row_classes: &[u32],
        token_tags: &[u32],
        device: &Device,
    ) -> Result<Tensor> {
        let row = self
            .remap
            .get(step)
            .ok_or_else(|| self.no_such_step(step))?;
        if row_classes.len() != token_tags.len() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adaln schedule: row_classes ({}) and token_tags ({}) must be one entry \
                 per sequence row",
                row_classes.len(),
                token_tags.len()
            )));
        }
        if row_classes.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 adaln schedule: the packed sequence has no rows".into(),
            ));
        }
        let mut out = Vec::with_capacity(row_classes.len());
        for (&class, &tag) in row_classes.iter().zip(token_tags) {
            let class = class as usize;
            if class >= row.len() {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 adaln schedule: row class {class} at step {step} is outside \
                     [0, {})",
                    row.len()
                )));
            }
            if tag as usize >= MODALITY_NUM {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 adaln schedule: token tag {tag} at step {step} is outside \
                     [0, {MODALITY_NUM})"
                )));
            }
            out.push(row[class] * MODALITY_NUM as u32 + tag);
        }
        Ok(Tensor::from_vec(out, (row_classes.len(),), device)?)
    }

    fn no_such_step(&self, step: usize) -> CandleError {
        CandleError::Msg(format!(
            "minimax-h3 adaln schedule: step {step} is outside this schedule's {} steps",
            self.steps.len()
        ))
    }
}

/// Return the evicted buffers to the system — the candle counterpart of MLX's allocator drain.
///
/// **This is a `Device::synchronize`, and that is not an incidental detail.** candle allocates every
/// CUDA tensor through `cuMemAllocAsync` on the device's stream-ordered pool. Dropping the last
/// `Tensor` decrements that pool's `used` immediately, but the pages stay charged to the process —
/// `nvidia-smi`, and therefore every admission gate in the product, still sees them — until
/// something synchronizes, because the pool's release threshold is 0 (the driver default; neither
/// candle nor `candle-gen` raises it). SC-15791 measured exactly this:
///
/// ```text
/// pool reserved: live 160.0 | after drop 160.0 | after drop+sync 0.0   MiB
/// ```
///
/// The corollary matters as much as the call: **`MemPool::trim` is a no-op here** and must not be
/// credited with the recovery. At a release threshold of 0 there is no retained cache for it to
/// return.
///
/// On the CPU device this is a cheap no-op — candle's CPU storage is a plain Rust allocation that
/// the global allocator reclaims at drop — so the call is unconditional and the eviction path reads
/// the same on both backends.
///
/// # No test in this crate gates this call
///
/// Stated here because the alternative is a reader assuming coverage that does not exist. Deleting
/// this call and running the whole crate with `--no-fail-fast` leaves it **green**: every suite runs
/// on the CPU device, where `synchronize` is by definition a no-op. The claim above is SC-15791's
/// measurement, not this crate's.
///
/// `tests/adaln_evict_memory.rs`'s `#[cfg(feature = "cuda")]` arm is the only thing that could gate
/// it, and it needs a GPU to run. Deleting the *drop* that precedes this call, by contrast, is
/// caught immediately.
pub fn release_device_memory(device: &Device) -> Result<()> {
    Ok(device.synchronize()?)
}

/// The precomputed AdaLN modulation for a whole denoise run — one table per block, covering every
/// distinct timestep the run will evaluate at.
///
/// Built by [`Self::precompute_and_evict`], which is also what releases the 26.02 GB of
/// `adaln_proj` weights.
#[derive(Debug, Clone)]
pub struct AdaLnCache {
    schedule: TimestepSchedule,
    layers: Vec<AdaLnModulation>,
    bytes: usize,
}

impl AdaLnCache {
    /// Project the schedule's modulation for every block.
    ///
    /// `embed` is the timestep MLP — `time_embedder(time_proj(t))` — applied to
    /// [`TimestepSchedule::distinct_timesteps`]. It is a closure rather than a `temb` argument so
    /// that the embedding is taken from the schedule's own timesteps and cannot be mis-bound to a
    /// different one.
    ///
    /// Unlike the MLX sibling this performs **no forced-evaluation step**: candle is eager, so every
    /// table below is already materialized when `forward` returns. Adding an `eval`-shaped call here
    /// would be inert, and inert ceremony next to a memory claim is worse than none — it reads as
    /// evidence.
    ///
    /// Does **not** evict; [`Self::precompute_and_evict`] is the memory lever.
    pub fn precompute<F>(blocks: &[DitBlock], schedule: TimestepSchedule, embed: F) -> Result<Self>
    where
        F: FnOnce(&[f32]) -> Result<Tensor>,
    {
        if blocks.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 adaln cache: the block stack is empty".into(),
            ));
        }
        let first = blocks[0].adaln_proj().ok_or_else(|| {
            CandleError::Msg(
                "minimax-h3 adaln cache: block 0's adaln_proj has already been evicted; a cache \
                 can only be built once per load"
                    .into(),
            )
        })?;
        let time_embed_dim = first.time_embed_dim();

        let temb = embed(schedule.distinct_timesteps())?;
        let want = [schedule.num_distinct_timesteps(), time_embed_dim];
        if temb.dims() != want {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adaln cache: the timestep embedding must be {want:?} — one row per \
                 distinct timestep at the projection's time_embed_dim — got {:?}",
                temb.dims()
            )));
        }

        let mut layers = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            layers.push(block.modulation(&temb).map_err(|e| {
                CandleError::Msg(format!("minimax-h3 adaln cache: layer {i}: {e}"))
            })?);
        }

        let bytes = layers.iter().map(AdaLnModulation::nbytes).sum();
        Ok(Self {
            schedule,
            layers,
            bytes,
        })
    }

    /// [`Self::precompute`], then release every block's AdaLN projection and hand the pages back —
    /// the whole 26.02 GB lever in one call.
    ///
    /// `residency` records the caller's decision. [`AdaLnResidency::Resident`] is **rejected**
    /// here: it is a valid choice, but it is the choice *not* to build a cache, and silently
    /// building one anyway would evict weights a run-state-dependent sampler still needs. Callers
    /// in that mode keep the projections and use [`DitBlock::forward_with_temb`].
    ///
    /// The order is load-bearing and is the reason this exists as one function rather than three
    /// steps at the call site:
    ///
    /// 1. project (already materialized — candle is eager);
    /// 2. take the projection out of every block and drop it, so **no `Tensor` handle** references
    ///    the weights. The caller must not be holding the loader's `Weights` map at this point:
    ///    `require` returns a storage-sharing clone, and one surviving map pins all 26.02 GB with
    ///    every block still reporting itself evicted;
    /// 3. [`release_device_memory`] — the synchronize that actually returns the pool's pages to the
    ///    driver. Without it `MemPool::used` falls while `nvidia-smi` moves 0.0 MiB.
    ///
    /// Returns the cache and the bytes of projection weight released. **Note the return is what the
    /// blocks were holding, not a measurement**: it is the arithmetic the lever is sized on, and
    /// `tests/adaln_evict_memory.rs` is what turns it into evidence.
    pub fn precompute_and_evict<F>(
        blocks: &mut [DitBlock],
        schedule: TimestepSchedule,
        residency: AdaLnResidency,
        device: &Device,
        embed: F,
    ) -> Result<(Self, usize)>
    where
        F: FnOnce(&[f32]) -> Result<Tensor>,
    {
        if residency != AdaLnResidency::PrecomputeAndEvict {
            return Err(CandleError::Msg(
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
        // The synchronize. `MemPool::used` falling is not on its own evidence that anything left
        // the process — without this the pages stay charged to it until something else synchronizes.
        release_device_memory(device)?;

        Ok((cache, released))
    }

    /// The modulation tables for block `layer`.
    pub fn modulation(&self, layer: usize) -> Result<&AdaLnModulation> {
        self.layers.get(layer).ok_or_else(|| {
            CandleError::Msg(format!(
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
    /// Honest because [`AdaLnModulation`]'s six tables are contiguous: they are six allocations,
    /// not six views of one.
    ///
    /// Independent of resolution and of duration by construction: nothing here has a token axis.
    /// (sc-17152 separately measured that the *whole model's* peak is flat to 0.5 % across duration
    /// and canvas on MLX, so this is not the only such quantity — but it is the only one that is
    /// flat by construction rather than by measurement.)
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether this cache can serve `schedule`, i.e. whether the schedule is bitwise the one it was
    /// built from. A changed step count, a changed sigma shift or a changed solver all produce a
    /// different [`ScheduleKey`] and therefore a `false` here.
    ///
    /// **Unwired in production today** — see [`ScheduleKey`]: [`crate::dit::JointDit`] builds a
    /// fresh cache per request, so nothing currently holds a cache long enough for this to answer
    /// `false`. It is the check a cross-request cache-reuse seam must run before serving.
    pub fn is_current_for(&self, schedule: &TimestepSchedule) -> bool {
        self.schedule.key() == schedule.key()
    }

    /// The dtype the cached tables carry.
    pub fn dtype(&self) -> Option<DType> {
        self.layers.first().map(|m| m.shift_msa.dtype())
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
    /// while the reference-audio class addresses the same one throughout.
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
        assert!(s.step_timesteps(99).is_err());
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

    /// `global · MODALITY_NUM + tag`, with both inputs bounds-checked on the host.
    #[test]
    fn adaln_indices_compose_the_global_row() {
        let dev = Device::Cpu;
        let s = TimestepSchedule::new(joint_schedule(4)).unwrap();
        // Three rows: a video row (class 0), a text row (class 3), an audio row (class 1).
        let classes = [0u32, 3, 1];
        let tags = [0u32, 1, 2];
        let got = s.adaln_indices(1, &classes, &tags, &dev).unwrap();
        let m = MODALITY_NUM as u32;
        let want = [
            s.global_timestep_index(1, 0).unwrap() * m,
            s.global_timestep_index(1, 3).unwrap() * m + 1,
            s.global_timestep_index(1, 1).unwrap() * m + 2,
        ];
        assert_eq!(got.to_vec1::<u32>().unwrap(), want);
        assert!(*got.to_vec1::<u32>().unwrap().iter().max().unwrap() < s.modulation_rows() as u32);

        for (label, classes, tags) in [
            ("class past the step", vec![9u32], vec![0u32]),
            (
                "tag beyond MODALITY_NUM",
                vec![0u32],
                vec![MODALITY_NUM as u32],
            ),
        ] {
            assert!(
                s.adaln_indices(0, &classes, &tags, &dev).is_err(),
                "{label} must be rejected"
            );
        }
        // Mismatched lengths are a caller bug, not a broadcast.
        assert!(s.adaln_indices(0, &[0u32, 1], &[0u32], &dev).is_err());
        assert!(s.adaln_indices(0, &[], &[], &dev).is_err());
        assert!(s.adaln_indices(99, &classes, &tags, &dev).is_err());
    }

    /// The cache-size claim: `6 · rows · hidden` per layer, and **no dependence on resolution or
    /// duration** — nothing in the table has a token axis.
    #[test]
    fn cache_bytes_are_independent_of_resolution_and_duration() {
        // 42 distinct timesteps at 20 steps × 3 modalities × 6 params × 5376 hidden × 50 layers.
        let s = TimestepSchedule::new(joint_schedule(20)).unwrap();
        let rows = s.modulation_rows();
        let bf16 = 2;
        let bytes = MODULATION_PARAMS * rows * 5376 * 50 * bf16;
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

    /// The synchronize is callable on the CPU device and is a cheap no-op there, so the eviction
    /// path reads the same on both backends rather than branching.
    #[test]
    fn releasing_device_memory_is_a_no_op_on_cpu() {
        release_device_memory(&Device::Cpu).unwrap();
    }
}
