//! sc-17149: **which of the two 66 GB DiT checkpoints a task loads, and that they are never both
//! resident.**
//!
//! # Nothing structural can tell `transformer/` from `transformer_ref/`
//!
//! Measured on the published snapshot `MiniMaxAI/MiniMax-H3 @ 939557dc`:
//!
//! * their `config.json` files are **byte-identical**;
//! * both ship **the same 638 tensor names**, with the same shapes and dtypes;
//! * only the tensor *values* differ.
//!
//! So a loader that reads the wrong one produces a model that loads cleanly, runs at the same
//! speed, allocates the same memory and emits plausible video. This is exactly the class
//! [`mlx_gen_minimax_h3::layout`] documents for the gated-FFN half-swap, and the answer is the
//! same: **only an explicit assertion against real bytes can pin it.** A shape check, a key-count
//! check or a config diff would all pass on the wrong checkpoint.
//!
//! `proj_in.bias` is the probe. It is a 5376-wide float32 tensor living in shard 1 of both
//! partitions, it differs between them, and it is reachable through
//! [`MiniMaxH3Dit::projections`] — so the assertion can be made against *the loaded model* rather
//! than against a file the loader might not have read.
//!
//! # Loading a checkpoint materializes nothing — measured
//!
//! The first thing this file established, and the reason its memory test looks the way it does:
//! **after `MiniMaxH3Dit::load` returns on the 66 GB `transformer_ref`, peak device memory is
//! 33 KB.** MLX memory-maps the shards and materializes a tensor on first use, so "loaded" and
//! "resident" are different states and the loader only ever reaches the first.
//!
//! That makes the obvious co-residency test worthless: `peak < 2 · checkpoint` after a bare load
//! passes unconditionally, including on a build that loaded *both* partitions. The guard that
//! caught this is still in the test — the "implausibly small" assertion — so the measurement cannot
//! silently revert to a no-op.
//!
//! `two_checkpoints_are_never_co_resident` therefore forces a fixed ~3.8 GB slab of **each**
//! partition in turn and gates on the peak across the pair. Measured: phase A peak one slab,
//! footprint back to **0** after the retried drain, phase B peak one slab — not two.
//!
//! What it can show: the release is real and the two partitions do not accumulate. What it
//! *cannot*: anything about a caller holding both handles at once, or about the remaining 62 GB of
//! each checkpoint. The first is structural, gated by `every_dit_load_site_is_driven_by_the_task`.
//!
//! The second is now gated too: `two_full_checkpoints_are_never_co_resident` (sc-17151) runs the
//! identical property over **every tensor of both partitions**. The slab test is kept because it is
//! the affordable arm — ~7.6 GB rather than ~133 GB, so it is the one that can be re-run under a
//! source mutation. It is **not** the tighter of the two: the full-scale arm is tighter on both
//! bounds (peak `bytes + bytes/8`, 1.125x, against the slab's `2 * slab`, 2.00x; residual
//! `bytes_a/16`, 6.25%, against the slab's `slab/4`, 25%). Both now release through the same
//! `mlx_gen::residency::drain_allocator_cache` and read active **plus cache**; the slab arm did not
//! until sc-17151's review, which is how it came to name `get_active_memory`-alone blindness in its
//! own assertion string while measuring with it.

mod common;

use std::path::Path;

use mlx_rs::Dtype;

use mlx_gen::gen_core::{LoadSpec, WeightsSource};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::model::MiniMaxH3Dit;
use mlx_gen_minimax_h3::model::{load, MiniMaxH3Task, BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION};

/// The tensor that distinguishes the two partitions. float32, `[5376]`, in shard 1 of each.
const PROBE: &str = "proj_in.bias";

/// Shard 1 of a DiT partition — the one holding [`PROBE`], so the probe costs one shard rather than
/// the whole 66 GB.
fn shard_one(root: &Path, partition: &str) -> Weights {
    Weights::from_file(
        root.join(partition)
            .join("diffusion_pytorch_model-00001-of-00014.safetensors"),
    )
    .unwrap_or_else(|e| panic!("reading shard 1 of {partition}: {e}"))
}

fn probe_bytes(root: &Path, partition: &str) -> Vec<f32> {
    let w = shard_one(root, partition);
    let t = w
        .require(PROBE)
        .unwrap_or_else(|e| panic!("{partition} has no {PROBE}: {e}"));
    t.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// **The two partitions really are byte-different and structurally identical.**
///
/// This is the premise every other test here rests on, so it is measured rather than assumed. If a
/// future snapshot made them identical, the wrong-checkpoint test below would become vacuous — and
/// this fails first, saying so.
#[test]
#[ignore = "needs the real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn the_two_partitions_are_indistinguishable_except_by_value() {
    let root = common::snapshot();
    let base_cfg =
        std::fs::read_to_string(root.join(BASE_DIT_PARTITION).join("config.json")).unwrap();
    let ref_cfg =
        std::fs::read_to_string(root.join(REFERENCE_DIT_PARTITION).join("config.json")).unwrap();
    assert_eq!(
        base_cfg, ref_cfg,
        "the two configs are byte-identical on the published snapshot — if this ever diverges, \
         the config becomes a legitimate discriminator and this file's premise changes"
    );

    let (base, refr) = (
        shard_one(&root, BASE_DIT_PARTITION),
        shard_one(&root, REFERENCE_DIT_PARTITION),
    );
    let mut base_keys: Vec<&str> = base.keys().collect();
    let mut ref_keys: Vec<&str> = refr.keys().collect();
    base_keys.sort_unstable();
    ref_keys.sort_unstable();
    assert_eq!(
        base_keys, ref_keys,
        "same tensor names — nothing structural separates the two partitions"
    );

    let (b, r) = (
        probe_bytes(&root, BASE_DIT_PARTITION),
        probe_bytes(&root, REFERENCE_DIT_PARTITION),
    );
    assert_eq!(b.len(), r.len(), "the probe has the same shape in both");
    assert_ne!(
        b, r,
        "{PROBE} must differ between the two partitions, or it cannot discriminate them"
    );
}

/// **The wrong-checkpoint test: a `ref2va` request loads `transformer_ref`, and this fails if it
/// loads `transformer`.**
///
/// Asserted against **the loaded model's own weights**, not against the path string — a test that
/// only compared partition names would pass on a loader that built the path correctly and then read
/// somewhere else.
///
/// The `assert_ne!` against the base checkpoint is the half that matters. Without it, a loader that
/// returned the *same* tensor for every partition would satisfy the positive assertion.
#[test]
#[ignore = "loads the 66 GB transformer_ref (MINIMAX_H3_SNAPSHOT)"]
fn a_ref2va_request_loads_transformer_ref_and_not_transformer() {
    let root = common::snapshot();
    let expected = probe_bytes(&root, REFERENCE_DIT_PARTITION);
    let base = probe_bytes(&root, BASE_DIT_PARTITION);
    assert_ne!(expected, base, "premise: the probe discriminates");

    // The task -> partition mapping is what the render path uses; it is NOT hardcoded here.
    let partition = MiniMaxH3Task::Ref2va.partition();
    assert_eq!(partition, REFERENCE_DIT_PARTITION);

    let dit = MiniMaxH3Dit::load(&root, partition, Dtype::Bfloat16).unwrap();
    let loaded = dit
        .projections()
        .proj_in
        .bias()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();

    assert_eq!(
        loaded, expected,
        "a ref2va load must carry transformer_ref's weights"
    );
    assert_ne!(
        loaded, base,
        "a ref2va load must NOT carry transformer/'s weights — this is the assertion that fails \
         if the wrong 66 GB checkpoint was read"
    );

    // …and the base task really does select the other one, so the mapping is not constant.
    assert_eq!(MiniMaxH3Task::T2va.partition(), BASE_DIT_PARTITION);
    assert_eq!(MiniMaxH3Task::Fl2va.partition(), BASE_DIT_PARTITION);
}

/// Force a fixed, measurable **slab** of one partition into device memory and return its size.
///
/// The 50 blocks' `attn.to_q.weight` — one tensor per layer, ~3.8 GB at bf16. Big enough that two
/// slabs summing is unmistakable in the counters, small enough that materializing both partitions'
/// in one process is not itself the OOM the test is about.
///
/// Every tensor is `eval`'d and then reduced through `.item()`, because MLX is lazy in both
/// directions: an unforced graph node has not allocated, and `eval` alone can leave the reduction
/// unrealized.
fn force_slab(root: &Path, partition: &str) -> (Vec<mlx_rs::Array>, u64) {
    let w = Weights::from_dir(root.join(partition)).unwrap();
    let mut held = Vec::new();
    let mut bytes = 0u64;
    for i in 0..50 {
        let key = format!("transformer_blocks.{i}.attn.to_q.weight");
        let t = w
            .require(&key)
            .unwrap_or_else(|e| panic!("{key}: {e}"))
            .clone();
        let elems: u64 = t.shape().iter().map(|&d| d as u64).product();
        // bf16 is 2 bytes; read the dtype rather than assuming, so a precision change is visible.
        let width = match t.dtype() {
            Dtype::Float32 | Dtype::Int32 | Dtype::Uint32 => 4u64,
            Dtype::Float64 => 8,
            Dtype::Int8 | Dtype::Uint8 | Dtype::Bool => 1,
            _ => 2,
        };
        bytes += elems * width;
        held.push(t);
    }
    let refs: Vec<&mlx_rs::Array> = held.iter().collect();
    mlx_rs::transforms::eval(refs).unwrap();
    // Reduce each one so the bytes are genuinely touched, not merely mapped.
    for t in &held {
        let s = t.sum(None).unwrap();
        mlx_rs::transforms::eval([&s]).unwrap();
        let _ = s.item::<f32>();
    }
    (held, bytes)
}

/// **The two checkpoints are never co-resident — measured across a real release, not asserted.**
///
/// # What this had to be rewritten to measure, and why
///
/// The obvious version — load the 66 GB `transformer_ref` and read the peak — **measures nothing**.
/// Measured on this snapshot: after `MiniMaxH3Dit::load` returns, peak device memory is
/// **33 KB**. MLX memory-maps the shards and materializes a tensor only on first use, so "the
/// checkpoint is loaded" and "the checkpoint is resident" are different states and the loader only
/// ever reaches the first. A test that asserted `peak < 2 · checkpoint` after a bare load would
/// have passed unconditionally — including on a build that loaded *both* partitions.
///
/// So this forces a fixed slab of **each** partition in turn and gates on the peak across the pair.
/// The property under test is the one that matters: after the first partition is released, the
/// second does not add to it.
///
/// # What it can and cannot show
///
/// **Can**: that materializing partition A, releasing it, and materializing partition B leaves the
/// process peak at roughly one slab rather than two — i.e. the release is real and the two do not
/// accumulate. **Cannot**: anything about a caller that holds both handles at once; that is a
/// structural property, gated by `every_dit_load_site_is_driven_by_the_task`. It also cannot speak
/// to the *whole* 66 GB, only to the slab — the full-checkpoint residency contract is sc-17151.
///
/// The `.item()` reduction and the retried drain are what sc-17145 established, and the 33 KB
/// reading above is what a build that forces neither looks like. The bulk `eval` that precedes the
/// reduction is *not* separately established — every tensor is held and then individually reduced,
/// so it may be redundant — and no measurement here distinguishes them.
#[test]
#[ignore = "materializes ~3.8 GB from each of the two DiT partitions (MINIMAX_H3_SNAPSHOT)"]
fn two_checkpoints_are_never_co_resident() {
    let root = common::snapshot();

    // Active **plus cache**, and the shared retried drain — the same two the full-scale sibling
    // below uses. This arm carried neither until sc-17151's review: it ran a bare
    // `for _ in 0..4 { clear_cache(); }` and read `get_active_memory` alone, while its own residual
    // assertion named reading active alone as the sc-17145 failure. A no-op drain migrates the slab
    // from active into the allocator cache, which is invisible to the counter it was using.
    let footprint =
        || (mlx_rs::memory::get_active_memory() + mlx_rs::memory::get_cache_memory()) as u64;

    mlx_gen::residency::drain_allocator_cache();
    mlx_rs::memory::reset_peak_memory();
    let baseline = footprint();

    // --- phase A: the ref2va partition ------------------------------------------------------
    let (held_a, slab) = force_slab(&root, MiniMaxH3Task::Ref2va.partition());
    let peak_a = mlx_rs::memory::get_peak_memory() as u64;
    assert!(
        peak_a > slab / 2,
        "peak {peak_a} is implausibly small for a {slab}-byte slab — nothing was materialized, so \
         this test would pass on a no-op (a bare `load` reads 33 KB)"
    );
    let resident_a = footprint();

    // --- release, through the production drain --------------------------------------------------
    drop(held_a);
    mlx_gen::residency::drain_allocator_cache();
    let after_release = footprint();
    assert!(
        after_release < baseline + slab / 4,
        "after releasing partition A, active+cached memory is {after_release} against a {baseline} \
         baseline and a {slab}-byte slab — the checkpoint did not leave. This reads active PLUS \
         cache because `get_active_memory` alone reporting success while buffers sit in the cache \
         is exactly the sc-17145 failure"
    );

    // --- phase B: the base partition ----------------------------------------------------------
    mlx_rs::memory::reset_peak_memory();
    let (held_b, slab_b) = force_slab(&root, MiniMaxH3Task::T2va.partition());
    let peak_b = mlx_rs::memory::get_peak_memory() as u64;
    assert_eq!(slab, slab_b, "the two partitions have identical geometry");
    assert!(peak_b > slab / 2, "partition B was not materialized either");

    // **The gate.** Materializing the second partition after releasing the first must not reach the
    // size of two slabs. If the release were a no-op — or if both were held — this is where it
    // shows, because peak is the one counter that cannot be satisfied by moving memory between the
    // active and cached pools.
    assert!(
        peak_b < 2 * slab,
        "peak {peak_b} while materializing the second partition reached two slabs ({}) — the two \
         checkpoints were co-resident",
        2 * slab
    );
    drop(held_b);
    mlx_gen::residency::drain_allocator_cache();

    println!(
        "slab {slab} B | phase A peak {peak_a} active+cache {resident_a} | released to \
         {after_release} | phase B peak {peak_b}"
    );
}

/// Materialize a **whole** DiT partition and hold it, returning the handles and their byte count.
///
/// The full-scale counterpart of [`force_slab`]: same forcing steps, every tensor. The `.item()`
/// reduction is the step that reads the bytes — without it the 33 KB reading in this file's header
/// is what comes back. The bulk `eval` that precedes it is kept because it is the shape the
/// production loaders use, not because removing it was measured to change a reading.
fn force_partition(root: &Path, partition: &str) -> (Vec<mlx_rs::Array>, u64) {
    let w = Weights::from_dir(root.join(partition)).unwrap();
    let mut keys: Vec<String> = w.keys().map(str::to_owned).collect();
    keys.sort_unstable();
    let mut held = Vec::with_capacity(keys.len());
    let mut bytes = 0u64;
    for key in &keys {
        let t = w
            .require(key)
            .unwrap_or_else(|e| panic!("{key}: {e}"))
            .clone();
        let elems: u64 = t.shape().iter().map(|&d| d as u64).product();
        let width = match t.dtype() {
            Dtype::Float32 | Dtype::Int32 | Dtype::Uint32 => 4u64,
            Dtype::Float64 => 8,
            Dtype::Int8 | Dtype::Uint8 | Dtype::Bool => 1,
            _ => 2,
        };
        bytes += elems * width;
        held.push(t);
    }
    let refs: Vec<&mlx_rs::Array> = held.iter().collect();
    mlx_rs::transforms::eval(refs).unwrap();
    for t in &held {
        let s = t.sum(None).unwrap();
        mlx_rs::transforms::eval([&s]).unwrap();
        let _ = s.item::<f32>();
    }
    (held, bytes)
}

/// **The two 66 GB checkpoints are never co-resident — at FULL checkpoint scale** (sc-17151).
///
/// `two_checkpoints_are_never_co_resident` above gates the same property over a ~3.8 GB slab, which
/// is what sc-17149 could afford and what its own header defers from: the slab shows the release
/// mechanism works, not that the *whole* checkpoint leaves. This materializes every tensor of
/// `transformer_ref`, releases it through the production drain
/// ([`mlx_gen::residency::drain_allocator_cache`]), and materializes every tensor of `transformer`.
///
/// The gate is the same one, at 17× the scale: peak while materializing the second partition must
/// stay under one checkpoint plus slack. Two co-resident checkpoints are ~133 GB against this box's
/// 107.52 GiB Metal `recommendedMaxWorkingSetSize`, so a regression here does not merely fail an
/// assertion — but the assertion is what names it, rather than leaving an allocator abort to be
/// interpreted.
///
/// The "implausibly small" guard is carried over verbatim, because it is the one that catches the
/// lazy-mmap no-op: a bare load of either partition reads 33 KB and would satisfy every bound below.
#[test]
#[ignore = "materializes each of the two 66 GB DiT partitions in turn (MINIMAX_H3_SNAPSHOT)"]
fn two_full_checkpoints_are_never_co_resident() {
    let root = common::snapshot();

    mlx_gen::residency::drain_allocator_cache();
    mlx_rs::memory::reset_peak_memory();
    // Active **plus cache**: `get_active_memory` alone reports a shed as complete while the buffers
    // sit in MLX's allocator cache with RSS unmoved (sc-17145).
    let footprint =
        || (mlx_rs::memory::get_active_memory() + mlx_rs::memory::get_cache_memory()) as u64;
    let baseline = footprint();

    // --- phase A: the whole ref2va checkpoint -------------------------------------------------
    let (held_a, bytes_a) = force_partition(&root, MiniMaxH3Task::Ref2va.partition());
    let peak_a = mlx_rs::memory::get_peak_memory() as u64;
    assert!(
        peak_a > bytes_a / 2,
        "peak {peak_a} is implausibly small for a {bytes_a}-byte checkpoint — nothing was \
         materialized, so this test would pass on a no-op (a bare `load` reads 33 KB)"
    );

    // --- release, through the production drain --------------------------------------------------
    drop(held_a);
    mlx_gen::residency::drain_allocator_cache();
    let after_release = footprint();
    assert!(
        after_release < baseline + bytes_a / 16,
        "after releasing the whole {bytes_a}-byte transformer_ref, active+cached memory is \
         {after_release} against a {baseline} baseline — the checkpoint did not leave"
    );

    // --- phase B: the whole base checkpoint -----------------------------------------------------
    mlx_rs::memory::reset_peak_memory();
    let (held_b, bytes_b) = force_partition(&root, MiniMaxH3Task::T2va.partition());
    let peak_b = mlx_rs::memory::get_peak_memory() as u64;
    assert_eq!(
        bytes_a, bytes_b,
        "the two partitions have identical geometry, so one bound covers both"
    );
    assert!(
        peak_b > bytes_b / 2,
        "partition B was not materialized either — peak {peak_b} for {bytes_b} bytes"
    );

    // **The gate.** One checkpoint plus allocator slack, never two.
    let ceiling = bytes_b + bytes_b / 8;
    assert!(
        peak_b < ceiling,
        "peak {peak_b} while materializing the second full checkpoint exceeded the {ceiling}-byte \
         single-checkpoint ceiling — the two 66 GB checkpoints were co-resident"
    );

    drop(held_b);
    mlx_gen::residency::drain_allocator_cache();

    println!(
        "full checkpoints: {:.2} GB each | phase A peak {:.2} GB | released to {:.2} GB (baseline \
         {:.2} GB) | phase B peak {:.2} GB",
        bytes_a as f64 / 1e9,
        peak_a as f64 / 1e9,
        after_release as f64 / 1e9,
        baseline as f64 / 1e9,
        peak_b as f64 / 1e9
    );
}

/// **The render path has exactly one DiT load site, and it is handed the task's partition.**
///
/// The structural half of the co-residency property, and the half the memory measurement cannot
/// reach: a measurement of one load says nothing about a second load elsewhere in the file. This
/// reads the source and pins that there is nowhere else for one to be.
///
/// Source-scanning is a blunt instrument, so it is bounded: it asserts a **count**, not a pattern
/// match, and it asserts the bare partition literals never appear at a call site — the state before
/// this story, where `"transformer"` was a string in the `load` arguments, would fail here.
#[test]
fn every_dit_load_site_is_driven_by_the_task() {
    // **Production code only.** `model.rs`'s own `#[cfg(test)]` module legitimately names both
    // partitions — that is where the mapping is asserted — so scanning the whole file would count
    // the assertions as violations and make this gate unpassable for the right reasons.
    let whole = include_str!("../src/model.rs");
    let src = whole
        .split_once("\n#[cfg(test)]\n")
        .map_or(whole, |(before, _)| before);
    assert!(
        src.len() < whole.len(),
        "expected a #[cfg(test)] module in model.rs; if it moved, this scan is now reading the \
         whole file and its counts mean something different"
    );

    // There are two load sites — the `t2va`/`fl2va` arm and the `ref2va` arm — and they are
    // mutually exclusive branches of `generate`, so at most one runs per render. What matters is
    // that **neither** picks its partition by hand.
    //
    // Both go through `load_dir` rather than `load(root, partition)` since sc-17150 made the DiT a
    // tiered component: a `q4` install stages it outside the snapshot root entirely, so there is no
    // `(root, partition)` pair that names it. `task_dit_dir` is the single place the task chooses a
    // directory, and it is still driven by `MiniMaxH3Task::partition()` — asserted directly below,
    // because a source scan cannot see that the helper it is counting does the right thing.
    // Comment lines are excluded throughout: prose legitimately names the load site, and counting
    // it would make the gate track the docs rather than the code.
    let code = || {
        src.lines()
            .map(str::trim_start)
            .filter(|l| !l.starts_with("//"))
    };
    // Two accepted forms since sc-18662: the resident loader and the deferred (rung-4) one. Both
    // must resolve through `task_dit_dir(task)` — the streamed path reads its own checkpoint per
    // task exactly like the resident one, or a `ref2va` streamed render would rebuild windows out
    // of the WRONG 66 GB, plausibly, forever.
    let total = code().filter(|l| l.contains("MiniMaxH3Dit::load")).count();
    let by_task = code()
        .filter(|l| {
            l.contains("MiniMaxH3Dit::load_dir(self.task_dit_dir(task), self.dtype)")
                || l.contains(
                    "MiniMaxH3Dit::load_dir_deferred(self.task_dit_dir(task), self.dtype)",
                )
        })
        .count();
    assert_eq!(
        total,
        by_task,
        "every DiT load must resolve its directory through task_dit_dir(task); {} of {total} \
         site(s) do not",
        total - by_task
    );
    assert!(total > 0, "the render path must load a DiT somewhere");

    // ...and that helper is the ONE place a partition is chosen, from the task.
    assert!(
        src.contains("match task.partition() {"),
        "task_dit_dir must switch on MiniMaxH3Task::partition(), not on the task enum directly — \
         otherwise `partition()` is decorative and the two can drift"
    );

    // The bare partition literals survive only as named constants' values. Before this story
    // `"transformer"` was a string at the call site, which is the state this rules out. Counting
    // raw occurrences broke when sc-17150 added `DIT_COMPONENT` (a third, legitimate one), so this
    // asserts the shape that actually matters: every occurrence is a `const … = "…";` binding, and
    // none is an argument.
    for literal in ["\"transformer\"", "\"transformer_ref\""] {
        let as_const = code()
            .filter(|l| l.starts_with("pub const") && l.contains(literal))
            .count();
        let in_code = code().filter(|l| l.contains(literal)).count();
        assert_eq!(
            in_code, as_const,
            "{literal} appears in {} non-comment line(s) but only {as_const} are `pub const` \
             bindings; a partition literal at a call site is exactly what this gate rules out",
            in_code
        );
        assert!(as_const > 0, "{literal} must be bound to a named constant");
    }

    // The `ref2va` arm resolves its task from the request rather than assuming it.
    assert!(
        src.contains("let task = MiniMaxH3Task::resolve("),
        "the task must be resolved from the request, not assumed per arm"
    );
}

/// **`ref2va`'s checkpoint is probed at load, beside the resolved DiT — not under the root.**
///
/// The sc-17149 / sc-17150 seam, from the caller's side. `ref2va` is a first-class task of this
/// engine, so a snapshot that cannot serve it must fail at load naming the path, rather than 20
/// minutes into a render when the reference arm finally reaches for it.
///
/// Costs no weights: `load` probes for `config.json` before it reads a single tensor.
#[test]
fn a_snapshot_without_the_reference_partition_is_refused_at_load() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let dit = root.join(BASE_DIT_PARTITION);
    std::fs::create_dir_all(&dit).unwrap();
    std::fs::write(dit.join("config.json"), "{}").unwrap();

    // 1. The base partition is staged, the reference one is not.
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a snapshot with no {REFERENCE_DIT_PARTITION} must not load"),
    };
    assert!(
        message.contains(REFERENCE_DIT_PARTITION),
        "the refusal must name the missing partition: {message}"
    );
    assert!(
        message.contains(&root.join(REFERENCE_DIT_PARTITION).display().to_string()),
        "the refusal must name the PATH it looked at, so a split install can see where: {message}"
    );

    // 2. Staging it gets past this probe — so the gate above is a real ordering, not a load that
    //    always fails. The next refusal is a DIFFERENT, later component.
    let reference = root.join(REFERENCE_DIT_PARTITION);
    std::fs::create_dir_all(&reference).unwrap();
    std::fs::write(reference.join("config.json"), "{}").unwrap();
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the remaining components are still missing"),
    };
    assert!(
        !message.contains(REFERENCE_DIT_PARTITION),
        "the reference partition is staged now; the refusal must have moved on: {message}"
    );
    // The text encoder is the second **tiered** component (sc-19120), so like the two DiT
    // partitions it is probed at its resolved location, ahead of the always-root shared ones.
    assert!(
        message.contains("text_encoder"),
        "the next missing component is the text encoder: {message}"
    );

    // 3. …and staging that moves the refusal on again, to the first shared component. Two hops
    //    rather than one, so what this pins is a chain and not a single lucky comparison.
    let te = root.join("text_encoder");
    std::fs::create_dir_all(&te).unwrap();
    std::fs::write(te.join("config.json"), "{}").unwrap();
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the remaining components are still missing"),
    };
    assert!(
        !message.contains("text_encoder"),
        "the text encoder is staged now; the refusal must have moved on: {message}"
    );
    assert!(
        message.contains("vae"),
        "the next missing component is the video VAE: {message}"
    );
}

/// The same probe on a **tiered** install looks beside the staged DiT, not under the snapshot root.
///
/// This is the case `root.join(REFERENCE_DIT_PARTITION)` got wrong. The root here holds a complete
/// pair of partitions and is still refused, because the *staged* tier is what `ref2va` would read
/// and that tier carries only its base partition.
#[test]
fn a_tiered_install_probes_the_reference_partition_inside_the_tier() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("snapshot");
    let tier = dir.path().join("tiers").join("q4");
    for base in [&root, &tier] {
        let dit = base.join(BASE_DIT_PARTITION);
        std::fs::create_dir_all(&dit).unwrap();
        std::fs::write(dit.join("config.json"), "{}").unwrap();
    }
    // The ROOT has a reference partition. The staged tier does not — and the tier is what counts.
    let root_reference = root.join(REFERENCE_DIT_PARTITION);
    std::fs::create_dir_all(&root_reference).unwrap();
    std::fs::write(root_reference.join("config.json"), "{}").unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    spec.components.insert(
        mlx_gen_minimax_h3::model::DIT_COMPONENT.to_string(),
        WeightsSource::Dir(tier.join(BASE_DIT_PARTITION)),
    );
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the staged tier has no {REFERENCE_DIT_PARTITION}"),
    };
    assert!(
        message.contains(&tier.join(REFERENCE_DIT_PARTITION).display().to_string()),
        "the probe must look inside the TIER, not under the root: {message}"
    );
    assert!(
        !message.contains(&root_reference.display().to_string()),
        "the root's reference partition belongs to a different tier and must not satisfy this: \
         {message}"
    );
}
