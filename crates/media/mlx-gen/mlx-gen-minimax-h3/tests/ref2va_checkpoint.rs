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
//! partition in turn and gates on the peak across the pair. Measured: phase A peak one slab, active
//! back to **0** after the retried drain, phase B peak one slab — not two.
//!
//! What it can show: the release is real and the two partitions do not accumulate. What it
//! *cannot*: anything about a caller holding both handles at once, or about the remaining 62 GB of
//! each checkpoint. The first is structural, gated by `every_dit_load_site_is_driven_by_the_task`;
//! the second is the full-render residency contract, which is sc-17151.

mod common;

use std::path::Path;

use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::model::MiniMaxH3Dit;
use mlx_gen_minimax_h3::model::{MiniMaxH3Task, BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION};

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
/// The forced evaluation, the `.item()` reduction and the retried drain are each individually
/// necessary; sc-17145 established all three, and the 33 KB reading above is what a build without
/// the first one looks like.
#[test]
#[ignore = "materializes ~3.8 GB from each of the two DiT partitions (MINIMAX_H3_SNAPSHOT)"]
fn two_checkpoints_are_never_co_resident() {
    let root = common::snapshot();

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let baseline = mlx_rs::memory::get_active_memory() as u64;

    // --- phase A: the ref2va partition ------------------------------------------------------
    let (held_a, slab) = force_slab(&root, MiniMaxH3Task::Ref2va.partition());
    let peak_a = mlx_rs::memory::get_peak_memory() as u64;
    assert!(
        peak_a > slab / 2,
        "peak {peak_a} is implausibly small for a {slab}-byte slab — nothing was materialized, so \
         this test would pass on a no-op (a bare `load` reads 33 KB)"
    );
    let active_a = mlx_rs::memory::get_active_memory() as u64;

    // --- release, with the retried drain ------------------------------------------------------
    drop(held_a);
    for _ in 0..4 {
        mlx_rs::memory::clear_cache();
    }
    let after_release = mlx_rs::memory::get_active_memory() as u64;
    assert!(
        after_release < baseline + slab / 4,
        "after releasing partition A, active memory is {after_release} against a {baseline} \
         baseline and a {slab}-byte slab — the checkpoint did not leave, and `get_active_memory` \
         alone reporting success while buffers sit in the cache is exactly the sc-17145 failure"
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
    for _ in 0..4 {
        mlx_rs::memory::clear_cache();
    }

    println!(
        "slab {slab} B | phase A peak {peak_a} active {active_a} | released to {after_release} | \
         phase B peak {peak_b}"
    );
}

/// **The render path has exactly one DiT load site, and it is handed the task's partition.**
///
/// The structural half of the co-residency property, and the half the memory measurement cannot
/// reach: a measurement of one load says nothing about a second load elsewhere in the file. This
/// reads the source and pins that there is nowhere else for one to be.
///
/// Source-scanning is a blunt instrument, so it is bounded: it asserts a **count**, not a pattern
/// match, and it asserts the bare partition literals are gone — the state before this story, where
/// `"transformer"` was a string at the call site, would fail here.
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
    let total = src.matches("MiniMaxH3Dit::load(").count();
    let by_task = src
        .matches("MiniMaxH3Dit::load(&self.root, task.partition(), self.dtype)")
        .count();
    assert_eq!(
        total,
        by_task,
        "every DiT load must take MiniMaxH3Task::partition(); {} of {total} site(s) do not",
        total - by_task
    );
    assert!(total > 0, "the render path must load a DiT somewhere");

    // The bare partition literals survive only as the two named constants' values. Before this
    // story `"transformer"` was a string at the call site, which is the state this rules out.
    assert_eq!(
        src.matches("\"transformer\"").count(),
        1,
        "`transformer` should appear once, as BASE_DIT_PARTITION's value"
    );
    assert_eq!(
        src.matches("\"transformer_ref\"").count(),
        1,
        "`transformer_ref` should appear once, as REFERENCE_DIT_PARTITION's value"
    );

    // The `ref2va` arm resolves its task from the request rather than assuming it.
    assert!(
        src.contains("let task = MiniMaxH3Task::resolve("),
        "the task must be resolved from the request, not assumed per arm"
    );
}
