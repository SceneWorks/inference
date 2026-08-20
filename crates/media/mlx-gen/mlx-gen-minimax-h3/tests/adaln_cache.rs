//! sc-17145: the precomputed AdaLN path is the naive per-step path, and a stale or mis-indexed
//! cache is observable.
//!
//! Fixture `tests/fixtures/dit_block.safetensors` ← `tools/dump_minimax_h3_dit.py`, the same
//! **official diffusers** `MiniMaxH3Transformer3DModel` golden `dit_parity.rs` uses.
//!
//! # What has to be paranoid here
//!
//! The precompute replaces `num_steps` projections with **one**, and then throws the projection
//! weights away. Every way that can go wrong is shape-identical to the right answer:
//!
//! * a cache built for a *different* schedule still has the right shape and gathers cleanly;
//! * a per-step remap that is off by one step still produces in-range rows;
//! * using a row's **class** index where its **global** table row belongs is in-range for every
//!   schedule long enough to have the rows, and wrong at every step but the first.
//!
//! So the gate is `rel` — relative max-abs-diff — and never a norm, a checksum or a cosine.
//! sc-18740's half-swap left the output norm essentially unchanged and cosine at 0.73-0.78, and
//! this crate has now shipped two defects that a magnitude assertion could not see. Each mutation
//! below prints its measured margin over the suite's own residual so the margin stays auditable
//! rather than assumed.
//!
//! Memory is deliberately **not** measured here: this binary runs alongside the rest of the suite,
//! and MLX's allocator counters are process-global. `tests/adaln_evict_memory.rs` and
//! `tests/adaln_evict_real_weights.rs` own that, one measurement per process.

mod common;

use common::{assert_parity, dit_fixture_config, rel, DIT_FIXTURE, DIT_LAYOUT};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{
    AdaLnCache, AdaLnResidency, DitBlock, MiniMaxH3DitConfig, MmRope, TimestepSchedule,
    MODALITY_NUM,
};

/// The mlx-gen house parity tolerance, as in `dit_parity.rs`.
const TOL: f32 = 1e-2;

/// A mutation has to clear the suite's own residual by a wide margin or "the output moved" is
/// reduced-precision jitter. `precompute_residual_bounds_the_mutation_floor` re-measures the
/// residual on every run and fails if the margin ever closes.
const MUTATION_FLOOR: f32 = 1e-2;

fn fixture() -> Weights {
    Weights::from_file(DIT_FIXTURE).unwrap()
}

fn model_weights() -> Weights {
    let mut w = fixture();
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    w
}

/// The fixture's two-block stack, freshly loaded so each test owns its own projections.
fn stack(cfg: &MiniMaxH3DitConfig) -> Vec<DitBlock> {
    let mut w = model_weights();
    (0..2)
        .map(|i| {
            DitBlock::from_weights(
                &mut w,
                &format!("transformer_blocks.{i}"),
                cfg,
                Dtype::Float32,
            )
            .unwrap()
        })
        .collect()
}

/// A stand-in for the timestep MLP sc-17147 owns (`time_embedder(time_proj(t))`).
///
/// Built on the host and **row-independent by construction**: `temb[i]` depends only on `t[i]`, so
/// embedding a step's four timesteps and embedding the whole run's distinct set produce bitwise
/// identical rows for the same timestep. That isolates the thing under test — if the naive and
/// cached block outputs diverge, it is the projection or the indexing, never the embedding.
fn embed(cfg: &MiniMaxH3DitConfig) -> impl Fn(&[f32]) -> mlx_gen::Result<Array> + '_ {
    move |ts: &[f32]| {
        let d = cfg.time_embed_dim;
        let mut v = Vec::with_capacity(ts.len() * d as usize);
        for &t in ts {
            for j in 0..d {
                let f = 0.37 * (j as f32 + 1.0);
                v.push((t * f).sin() * 0.7 + (t * f * 0.5).cos() * 0.3);
            }
        }
        Ok(Array::from_slice(&v, &[ts.len() as i32, d]))
    }
}

/// `σ' = s·σ / (1 + (s−1)·σ)` — the exponential sigma shift, video 12.0 / audio 3.0.
fn shift(sigma: f32, s: f32) -> f32 {
    s * sigma / (1.0 + (s - 1.0) * sigma)
}

/// The four row classes a MiniMax-H3 step's packed sequence carries: video `t`, audio `t` (a
/// different shift), the conditioning rows at `max(video_t, 0.999)` and the text rows at `1.0`.
///
/// `t = 1 − σ`, with `t = 1` clean — the reversed convention `MiniMaxH3Scheduler` documents
/// (sc-17146 activity 18717). `σ` descends from 1 but never reaches 0: the terminal zero is part of
/// `num_inference_steps` and the loop runs one evaluation fewer, so `evals` here is the number of
/// **model evaluations**, not the requested step count.
fn joint_schedule(evals: usize) -> TimestepSchedule {
    let steps = (0..evals)
        .map(|i| {
            let sigma = 1.0 - (i as f32) / (evals as f32);
            let video_t = 1.0 - shift(sigma, 12.0);
            let audio_t = 1.0 - shift(sigma, 3.0);
            vec![video_t, audio_t, video_t.max(0.999), 1.0]
        })
        .collect();
    TimestepSchedule::new(steps).unwrap()
}

/// One packed sequence's `(row class, modality tag)` per row, laid out like `DIT_LAYOUT`: text rows
/// on the text class, audio rows on the audio one, video rows on the video one.
fn packed_rows() -> (Vec<i32>, Vec<i32>) {
    let text = DIT_LAYOUT.num_text_tokens;
    let audio = DIT_LAYOUT.num_audio_latents * DIT_LAYOUT.audio_channels;
    let video = DIT_LAYOUT.num_latent_frames
        * (DIT_LAYOUT.latent_height / 2)
        * (DIT_LAYOUT.latent_width / 2);
    let mut classes = Vec::new();
    let mut tags = Vec::new();
    // Classes index `joint_schedule`'s `[video_t, audio_t, keyframe_t, 1.0]`.
    for _ in 0..text {
        classes.push(3);
        tags.push(1);
    }
    for _ in 0..audio {
        classes.push(1);
        tags.push(2);
    }
    for _ in 0..video {
        classes.push(0);
        tags.push(0);
    }
    (classes, tags)
}

/// A deterministic hidden state of the right shape.
fn hidden(seq: i32, width: i32) -> Array {
    let n = seq * width;
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 0.5).collect();
    Array::from_slice(&v, &[1, seq, width])
}

/// Run the whole schedule the **cached** way: one projection up front, projections evicted, then
/// `DitBlock::forward` against the global table at every step.
fn run_cached(cfg: &MiniMaxH3DitConfig, schedule: TimestepSchedule) -> (Array, AdaLnCache) {
    let mut blocks = stack(cfg);
    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule,
        AdaLnResidency::PrecomputeAndEvict,
        embed(cfg),
    )
    .unwrap();
    assert!(released > 0, "the eviction must have released bytes");
    assert!(
        blocks.iter().all(|b| !b.holds_adaln()),
        "every block's projection must be gone"
    );

    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    let classes = Array::from_slice(&classes, &[seq]);
    let tags = Array::from_slice(&tags, &[seq]);

    let mut x = hidden(seq, cfg.hidden_size);
    for step in 0..cache.schedule().num_steps() {
        let idx = cache
            .schedule()
            .adaln_indices(step, &classes, &tags)
            .unwrap();
        for (layer, block) in blocks.iter().enumerate() {
            x = block
                .forward(&x, cache.modulation(layer).unwrap(), &idx, &rope, &tables)
                .unwrap();
        }
    }
    (x, cache)
}

/// Run the same schedule the **naive** way: re-project this step's own timesteps at every step,
/// against blocks that never gave up their weights, and index with the step-local rows.
fn run_naive(cfg: &MiniMaxH3DitConfig, schedule: &TimestepSchedule) -> Array {
    let blocks = stack(cfg);
    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    // The naive AdaLN row is `row_class · MODALITY_NUM + tag` — the reference's own addressing when
    // the projected table holds only this step's own timesteps.
    let naive_rows: Vec<i32> = classes
        .iter()
        .zip(&tags)
        .map(|(c, t)| c * MODALITY_NUM + t)
        .collect();
    let naive_idx = Array::from_slice(&naive_rows, &[seq]);

    let mut x = hidden(seq, cfg.hidden_size);
    for step in 0..schedule.num_steps() {
        let temb = embed(cfg)(schedule.step_timesteps(step).unwrap()).unwrap();
        for block in &blocks {
            x = block
                .forward_with_temb(&x, &temb, &naive_idx, &rope, &tables)
                .unwrap();
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------
// numeric identity
// ---------------------------------------------------------------------------------------------

/// The precomputed table IS the reference's `adaln_proj(temb)` — checked against the diffusers
/// golden directly, at the fixture's own two timesteps, so the cache is anchored to the reference
/// and not only to this crate's naive path.
#[test]
fn the_cache_reproduces_the_reference_modulation_table() {
    let cfg = dit_fixture_config();
    let f = fixture();
    // The fixture's timesteps, in the order `in.temb`'s rows carry them.
    let ts: Vec<f32> = f
        .require("in.timestep")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(ts.len(), 2, "the fixture dumps two distinct timesteps");

    let schedule = TimestepSchedule::new(vec![ts.clone()]).unwrap();
    assert_eq!(schedule.num_distinct_timesteps(), 2);
    let mut blocks = stack(&cfg);
    let (cache, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule,
        AdaLnResidency::PrecomputeAndEvict,
        // The reference's own `time_embedder(time_proj(t))` output, dumped as `in.temb`.
        |_| Ok(f.require("in.temb").unwrap().clone()),
    )
    .unwrap();

    let m = cache.modulation(0).unwrap();
    for (name, got) in [
        ("shift_msa", &m.shift_msa),
        ("scale_msa", &m.scale_msa),
        ("gate_msa", &m.gate_msa),
        ("shift_mlp", &m.shift_mlp),
        ("scale_mlp", &m.scale_mlp),
        ("gate_mlp", &m.gate_mlp),
    ] {
        assert_eq!(
            got.shape(),
            &[2 * MODALITY_NUM, cfg.hidden_size],
            "{name} shape"
        );
        assert_parity(
            got,
            f.require(&format!("out.modulation.{name}")).unwrap(),
            TOL,
            name,
        );
    }
}

/// **The acceptance criterion.** A whole run through the precomputed table equals the same run
/// re-projecting at every step, at three different schedule lengths.
///
/// The two paths are genuinely different computations, not a restatement: the naive one runs
/// `num_steps` projections of `[4, time_embed_dim]` and indexes a 12-row table with step-local
/// rows; the cached one runs **one** projection of `[num_distinct, time_embed_dim]` and indexes a
/// global table through the schedule's remap, with the projection weights already released.
#[test]
fn the_precomputed_path_reproduces_the_naive_per_step_run() {
    let cfg = dit_fixture_config();
    for steps in [2usize, 5, 9] {
        let schedule = joint_schedule(steps);
        let naive = run_naive(&cfg, &schedule);
        let (cached, cache) = run_cached(&cfg, schedule);
        let (peak, mean) = rel(&cached, &naive);
        println!(
            "  {steps:>2} steps: {} distinct timesteps, {} table rows, cache {} B — peak rel \
             {peak:.3e} (mean {mean:.3e})",
            cache.schedule().num_distinct_timesteps(),
            cache.schedule().modulation_rows(),
            cache.bytes()
        );
        assert!(
            peak < TOL,
            "{steps} steps: the precomputed path diverges from the naive one by {peak:.3e}"
        );
    }
}

/// The residual the mutations below have to clear. Printed on every run and asserted to stay well
/// under `MUTATION_FLOOR`, so a suite whose mutations sat in its own jitter would fail here first.
#[test]
fn precompute_residual_bounds_the_mutation_floor() {
    let cfg = dit_fixture_config();
    let schedule = joint_schedule(5);
    let naive = run_naive(&cfg, &schedule);
    let (cached, _) = run_cached(&cfg, schedule);
    let (peak, _) = rel(&cached, &naive);
    println!("  precompute residual {peak:.3e}, mutation floor {MUTATION_FLOOR:.1e}");
    assert!(
        peak * 10.0 < MUTATION_FLOOR,
        "the residual {peak:.3e} is within 10x of the mutation floor {MUTATION_FLOOR:.1e}; the \
         mutation tests would be reporting jitter"
    );
}

// ---------------------------------------------------------------------------------------------
// the cache going stale or mis-indexed is OBSERVABLE
// ---------------------------------------------------------------------------------------------

/// A cache built for one schedule, used to run another, is wrong — and `is_current_for` is the
/// guard that says so.
///
/// Both schedules have the same step count and the same table shape, so nothing structural
/// separates them. This is the failure a held-across-requests cache would have.
#[test]
fn a_cache_from_another_schedule_is_stale_and_observably_wrong() {
    let cfg = dit_fixture_config();
    // Two schedules built to be structurally INDISTINGUISHABLE: same steps, same row classes,
    // same distinct-timestep count, so the tables are the same shape and every index is in range.
    // Only the values differ — a different sigma shift, i.e. a different creative request.
    let explicit = |video0: f32| {
        let steps = (0..5)
            .map(|i| {
                let video_t = video0 + 0.05 * i as f32;
                let audio_t = 0.32 + 0.05 * i as f32;
                vec![video_t, audio_t, 0.999, 1.0]
            })
            .collect();
        TimestepSchedule::new(steps).unwrap()
    };
    let a = explicit(0.10);
    let b = explicit(0.11);
    assert_eq!(a.num_steps(), b.num_steps());
    assert_eq!(a.modulation_rows(), b.modulation_rows());
    assert_ne!(a.key(), b.key(), "the two schedules must key differently");

    let (correct, cache_b) = run_cached(&cfg, b.clone());
    assert!(
        !cache_b.is_current_for(&a),
        "a cache built for schedule B must report itself stale for schedule A"
    );
    assert!(cache_b.is_current_for(&b));

    // Run schedule B's steps against schedule A's table — the stale-cache mutation.
    let mut blocks = stack(&cfg);
    let (cache_a, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        a,
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    let classes = Array::from_slice(&classes, &[seq]);
    let tags = Array::from_slice(&tags, &[seq]);
    let mut x = hidden(seq, cfg.hidden_size);
    for step in 0..b.num_steps() {
        let idx = cache_a
            .schedule()
            .adaln_indices(step, &classes, &tags)
            .unwrap();
        for (layer, block) in blocks.iter().enumerate() {
            x = block
                .forward(&x, cache_a.modulation(layer).unwrap(), &idx, &rope, &tables)
                .unwrap();
        }
    }
    let (peak, _) = rel(&x, &correct);
    println!("  stale cache (schedule A's table for schedule B): peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a stale cache moved the output only {peak:.3e} — the parity test could not see it"
    );
}

/// Using step `i-1`'s remap at step `i` — the classic off-by-one in a cached loop. Every index
/// stays in range, so nothing fails structurally.
#[test]
fn a_step_off_by_one_in_the_remap_is_observable() {
    let cfg = dit_fixture_config();
    let schedule = joint_schedule(5);
    let (correct, cache) = run_cached(&cfg, schedule.clone());

    let mut blocks = stack(&cfg);
    let (cache2, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule.clone(),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    assert!(cache2.is_current_for(&schedule));
    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    let classes = Array::from_slice(&classes, &[seq]);
    let tags = Array::from_slice(&tags, &[seq]);
    let mut x = hidden(seq, cfg.hidden_size);
    for step in 0..schedule.num_steps() {
        let stale = step.saturating_sub(1);
        let idx = schedule.adaln_indices(stale, &classes, &tags).unwrap();
        for (layer, block) in blocks.iter().enumerate() {
            x = block
                .forward(&x, cache.modulation(layer).unwrap(), &idx, &rope, &tables)
                .unwrap();
        }
    }
    let (peak, _) = rel(&x, &correct);
    println!("  remap off by one step: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a one-step-stale remap moved the output only {peak:.3e}"
    );
}

/// Addressing the **global** table with a step's **local** timestep index — i.e. forgetting the
/// remap entirely. In range for every step (the table has more rows than a step has timesteps),
/// and wrong at every step but the first.
#[test]
fn skipping_the_local_to_global_remap_is_observable() {
    let cfg = dit_fixture_config();
    let schedule = joint_schedule(5);
    let (correct, cache) = run_cached(&cfg, schedule.clone());

    let blocks = stack(&cfg);
    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    let unremapped: Vec<i32> = classes
        .iter()
        .zip(&tags)
        .map(|(c, t)| c * MODALITY_NUM + t)
        .collect();
    let idx = Array::from_slice(&unremapped, &[seq]);
    assert!(
        unremapped
            .iter()
            .all(|&r| r < cache.schedule().modulation_rows()),
        "the un-remapped rows are IN RANGE — nothing structural catches this"
    );

    let mut x = hidden(seq, cfg.hidden_size);
    for _ in 0..schedule.num_steps() {
        for (layer, block) in blocks.iter().enumerate() {
            x = block
                .forward(&x, cache.modulation(layer).unwrap(), &idx, &rope, &tables)
                .unwrap();
        }
    }
    let (peak, _) = rel(&x, &correct);
    println!("  local index used as a global row: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "skipping the remap moved the output only {peak:.3e}"
    );
}

// ---------------------------------------------------------------------------------------------
// schedule keying
// ---------------------------------------------------------------------------------------------

/// The cache is keyed on the schedule, not fixed at a step count. Every pair of the five lengths
/// tested must key differently and must reject each other's caches.
#[test]
fn the_cache_is_keyed_by_the_whole_schedule() {
    let cfg = dit_fixture_config();
    let schedules: Vec<TimestepSchedule> = [2usize, 4, 5, 8, 20]
        .iter()
        .map(|&n| joint_schedule(n))
        .collect();
    for (i, a) in schedules.iter().enumerate() {
        let mut blocks = stack(&cfg);
        let (cache, _) = AdaLnCache::precompute_and_evict(
            &mut blocks,
            a.clone(),
            AdaLnResidency::PrecomputeAndEvict,
            embed(&cfg),
        )
        .unwrap();
        assert_eq!(cache.num_layers(), 2);
        assert!(cache.is_current_for(a));
        for (j, b) in schedules.iter().enumerate() {
            if i != j {
                assert!(
                    !cache.is_current_for(b),
                    "the {}-step cache claimed to serve the {}-step schedule",
                    a.num_steps(),
                    b.num_steps()
                );
            }
        }
        println!(
            "  {:>2} steps -> {:>2} distinct timesteps, key {}",
            a.num_steps(),
            a.num_distinct_timesteps(),
            a.key()
        );
    }
}

/// The cache does not grow with resolution or duration — only with the schedule. Same schedule,
/// two very different packed-sequence lengths, identical cache bytes.
#[test]
fn the_cache_size_tracks_the_schedule_and_nothing_else() {
    let cfg = dit_fixture_config();
    let mut blocks = stack(&cfg);
    let (small, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        joint_schedule(5),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    let mut blocks = stack(&cfg);
    let (big, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        joint_schedule(20),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    // 6 tables × rows × hidden × 4 B (f32) × 2 layers.
    let expect =
        |s: &TimestepSchedule| 6 * s.modulation_rows() as usize * cfg.hidden_size as usize * 4 * 2;
    assert_eq!(small.bytes(), expect(small.schedule()));
    assert_eq!(big.bytes(), expect(big.schedule()));
    assert!(big.bytes() > small.bytes(), "more steps, more rows");
    println!(
        "  5 steps {} B, 20 steps {} B — a function of the schedule only",
        small.bytes(),
        big.bytes()
    );
}

/// Two of the four row classes are the SAME timestep at every step, so the global union is far
/// shorter than the per-step concatenation. That dedup is roughly a 2x saving on the cache.
#[test]
fn the_constant_row_classes_collapse_across_the_schedule() {
    let s = joint_schedule(20);
    let concatenated: usize = (0..s.num_steps())
        .map(|i| s.step_timesteps(i).unwrap().len())
        .sum();
    assert_eq!(concatenated, 80);
    assert!(
        s.num_distinct_timesteps() <= 45,
        "expected the 0.999 / 1.0 classes to collapse, got {}",
        s.num_distinct_timesteps()
    );
    println!(
        "  20 steps: {concatenated} (step, class) pairs -> {} distinct timesteps",
        s.num_distinct_timesteps()
    );
}

// ---------------------------------------------------------------------------------------------
// eviction semantics and the sampler exclusion
// ---------------------------------------------------------------------------------------------

/// After the evict the block still runs — `forward` never touched the projection — while every
/// path that needs the projection is a typed error naming the way out.
#[test]
fn an_evicted_block_still_runs_but_cannot_project() {
    let cfg = dit_fixture_config();
    let mut blocks = stack(&cfg);
    let per_block = blocks[0].adaln_proj().unwrap().nbytes();
    assert_eq!(
        blocks[0].adaln_proj().unwrap().time_embed_dim(),
        cfg.time_embed_dim
    );
    assert_eq!(
        blocks[0].adaln_proj().unwrap().out_features(),
        cfg.adaln_out_features()
    );

    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        joint_schedule(3),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    assert_eq!(released, 2 * per_block, "both blocks' projections released");

    let f = fixture();
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let (classes, tags) = packed_rows();
    let seq = classes.len() as i32;
    let idx = cache
        .schedule()
        .adaln_indices(
            0,
            &Array::from_slice(&classes, &[seq]),
            &Array::from_slice(&tags, &[seq]),
        )
        .unwrap();
    let x = hidden(seq, cfg.hidden_size);
    let out = blocks[0]
        .forward(&x, cache.modulation(0).unwrap(), &idx, &rope, &tables)
        .unwrap();
    assert_eq!(out.shape(), x.shape());
    assert!(out.sum(None).unwrap().item::<f32>().is_finite());

    for block in &blocks {
        assert!(!block.holds_adaln());
        assert!(block.adaln_proj().is_none());
        let e = block
            .modulation(f.require("in.temb").unwrap())
            .unwrap_err()
            .to_string();
        assert!(e.contains("evicted"), "{e}");
        assert!(
            e.contains("Resident"),
            "the error must name the way out: {e}"
        );
    }
    // Idempotent.
    assert!(blocks[0].evict_adaln().is_none());
    // A cache cannot be rebuilt from an evicted stack.
    let e = AdaLnCache::precompute(&blocks, joint_schedule(3), embed(&cfg))
        .unwrap_err()
        .to_string();
    assert!(e.contains("already been evicted"), "{e}");
}

/// **The recorded decision.** A sampler whose evaluation timesteps are not enumerable up front is
/// EXCLUDED from the eviction, not serviced by invalidating the cache — invalidation after the
/// projections are gone would mean re-reading 26 GB mid-denoise. `AdaLnResidency::Resident` is
/// therefore a rejection here, and the error names the per-step path that replaces it.
#[test]
fn resident_residency_refuses_to_precompute_and_evict() {
    let cfg = dit_fixture_config();
    let mut blocks = stack(&cfg);
    let e = AdaLnCache::precompute_and_evict(
        &mut blocks,
        joint_schedule(4),
        AdaLnResidency::Resident,
        embed(&cfg),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("does not precompute"), "{e}");
    assert!(e.contains("forward_with_temb"), "{e}");
    assert!(
        blocks.iter().all(DitBlock::holds_adaln),
        "a rejected precompute must not have evicted anything"
    );
    // …and the per-step path it names still works.
    let f = fixture();
    blocks[0]
        .modulation(f.require("in.temb").unwrap())
        .expect("Resident keeps the projection");
}

/// The other half of the exclusion: a timestep the schedule was not built with is a typed error,
/// not a nearest-row gather.
///
/// The concrete case is `dpmpp_sde`, the one curated in-tree solver that evaluates **off the sigma
/// grid** — a stochastic midpoint at `sigma_s = sigma_of(t + h·R)`
/// (`gen-core/src/sampling/solvers.rs:243-256`). A port that enumerated only the grid would build a
/// cache missing a row for the second evaluation of every step. This makes that fail loudly at the
/// first unlisted evaluation instead of silently gathering a neighbour.
#[test]
fn an_off_grid_midpoint_timestep_is_rejected_rather_than_rounded() {
    let s = joint_schedule(5);
    let a = s.step_timesteps(0).unwrap()[0];
    let b = s.step_timesteps(1).unwrap()[0];
    assert_eq!(
        s.index_of(a).unwrap(),
        s.global_timestep_index(0, 0).unwrap()
    );

    // The midpoint between two grid timesteps — exactly what dpmpp_sde's second eval lands on.
    let midpoint = 0.5 * (a + b);
    assert!(midpoint != a && midpoint != b, "a genuine off-grid value");
    let e = s.index_of(midpoint).unwrap_err().to_string();
    assert!(e.contains("not in this schedule"), "{e}");
    assert!(
        e.contains("midpoint"),
        "the error must explain the case: {e}"
    );

    // Enumerating the midpoints makes it precomputable again — the fix, not a dead end. The
    // midpoint becomes a fifth row class, declared at EVERY step so the class order stays stable
    // (the last step, which has no successor, repeats its own timestep — classes may coincide).
    let with_midpoints = TimestepSchedule::new(
        (0..s.num_steps())
            .map(|i| {
                let mut step = s.step_timesteps(i).unwrap().to_vec();
                let next = s.step_timesteps((i + 1).min(s.num_steps() - 1)).unwrap()[0];
                step.push(0.5 * (step[0] + next));
                step
            })
            .collect(),
    )
    .unwrap();
    assert_eq!(with_midpoints.num_row_classes(), 5);
    assert!(with_midpoints.index_of(midpoint).is_ok());
    assert!(with_midpoints.num_distinct_timesteps() > s.num_distinct_timesteps());
}

/// A `temb` that is not one row per distinct timestep is rejected before anything is evicted — a
/// caller that embedded per step rather than per distinct timestep would otherwise build a table of
/// the wrong height and index it out of range.
#[test]
fn a_mis_shaped_timestep_embedding_is_rejected_before_eviction() {
    let cfg = dit_fixture_config();
    let mut blocks = stack(&cfg);
    let e = AdaLnCache::precompute_and_evict(
        &mut blocks,
        joint_schedule(5),
        AdaLnResidency::PrecomputeAndEvict,
        // One row per STEP, not per distinct timestep.
        |_| Ok(Array::zeros::<f32>(&[5, dit_fixture_config().time_embed_dim]).unwrap()),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("one row per distinct timestep"), "{e}");
    assert!(
        blocks.iter().all(DitBlock::holds_adaln),
        "a rejected precompute must not have evicted anything"
    );
}
