//! sc-17157: **which of the two 66 GB DiT checkpoints a task loads, and that a render maps exactly
//! one of them.**
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
//! [`candle_gen_minimax_h3::layout`] documents for the gated-FFN half-swap, and the answer is the
//! same: **only an explicit assertion against real bytes can pin it.** A shape check, a key-count
//! check or a config diff would all pass on the wrong checkpoint.
//!
//! `proj_in.bias` is the probe. It is a 5376-wide float32 tensor living in shard 1 of both
//! partitions, it differs between them, and it is reachable through
//! `MiniMaxH3Dit::projections()` — so the assertion can be made against *the loaded model* rather
//! than against a file the loader might not have read.
//!
//! # What is gated WITHOUT weights, and why that matters
//!
//! The real-bytes assertion is necessarily `#[ignore]`d: it needs 66 GB. An `#[ignore]`d test that
//! is never run reports nothing, so the load-bearing guards here are the **three weights-free**
//! ones, which run on every CI lane:
//!
//! * `the_task_mapping_is_not_a_constant` — the selection function itself;
//! * `a_snapshot_without_the_reference_partition_is_refused` — the sc-19517 hosting gap fails LOUD
//!   at load rather than degrading to the base checkpoint;
//! * `every_dit_load_site_is_driven_by_the_task` — a source scan proving there is nowhere in the
//!   render path for a hardcoded partition string to be.
//!
//! The last one is what covers "never both checkpoints resident" structurally: a render maps a DiT
//! at exactly two sites, and both are handed `task.partition()`, so the two 66 GB partitions cannot
//! both be mapped by one render. `model.rs`'s
//! `every_heavy_component_is_released_before_the_next_one_is_mapped` covers the release ordering of
//! whichever one it picked.

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use common::{safetensors_keys, snapshot};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{LoadSpec, WeightsSource};
use candle_gen_minimax_h3::{
    MiniMaxH3, MiniMaxH3Dit, MiniMaxH3Task, BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION,
};

/// The tensor that distinguishes the two partitions. float32, `[inner_dim]`, in shard 1 of each.
const PROBE: &str = "proj_in.bias";

/// The component directories a snapshot must carry for `MiniMaxH3::load` to succeed.
const COMPONENTS: [&str; 5] = [
    "text_encoder",
    BASE_DIT_PARTITION,
    REFERENCE_DIT_PARTITION,
    "vae",
    "audio_vae",
];

// ---------------------------------------------------------------------------------------------
// Weights-free
// ---------------------------------------------------------------------------------------------

/// **The task mapping is not a constant, and only `ref2va` moves.**
///
/// The negative half is the one that matters: a mapping that returned `transformer_ref` for
/// everything, or `transformer` for everything, satisfies "ref2va loads transformer_ref" or "t2va
/// loads transformer" respectively. Both are asserted, plus that the two names differ at all.
#[test]
fn the_task_mapping_is_not_a_constant() {
    assert_eq!(MiniMaxH3Task::Ref2va.partition(), REFERENCE_DIT_PARTITION);
    assert_eq!(MiniMaxH3Task::T2va.partition(), BASE_DIT_PARTITION);
    assert_eq!(MiniMaxH3Task::Fl2va.partition(), BASE_DIT_PARTITION);
    assert_ne!(
        BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION,
        "the two partitions must be different directories, or the selection is vacuous"
    );
    assert_ne!(
        MiniMaxH3Task::Ref2va.partition(),
        MiniMaxH3Task::T2va.partition()
    );
}

/// **A snapshot carrying only `transformer/` is REFUSED at load, naming the missing partition.**
///
/// This is the sc-19517 hosting gap made loud: `SceneWorks/minimax-h3-mlx` publishes no
/// `q4/transformer_ref`, so a pure-`q4` install has exactly this shape. The failure mode being
/// prevented is a `ref2va` request silently rendering off `transformer/` — which produces plausible
/// video and is wrong.
///
/// The control is the point: the SAME root **with** the reference partition loads, so the refusal
/// is attributable to that directory and not to a broken fixture.
#[test]
fn a_snapshot_without_the_reference_partition_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    for c in COMPONENTS.iter().filter(|c| **c != REFERENCE_DIT_PARTITION) {
        std::fs::create_dir_all(root.join(c)).unwrap();
    }
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    let e = match MiniMaxH3::load(&spec) {
        Ok(_) => panic!(
            "a snapshot without `{REFERENCE_DIT_PARTITION}/` must NOT load — ref2va would render \
             off the base checkpoint"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        e.contains(REFERENCE_DIT_PARTITION),
        "the refusal must name the missing partition: {e}"
    );

    // Control: the same root with the partition present loads.
    std::fs::create_dir_all(root.join(REFERENCE_DIT_PARTITION)).unwrap();
    MiniMaxH3::load(&spec).expect("the control must load, else the refusal proves nothing");
}

/// **The render path has exactly two DiT load sites, and BOTH are handed the task's partition.**
///
/// The structural half of the never-both-resident property, and the half a memory measurement
/// cannot reach: measuring one load says nothing about a second load elsewhere in the file. This
/// reads the production source and pins that there is nowhere else for one to be.
///
/// Source-scanning is a blunt instrument, so it is bounded: it asserts a **count**, and it asserts
/// the bare partition literals never appear at a call site.
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

    // Comment lines are excluded throughout: prose legitimately narrates the load site, and
    // counting it would make the gate track the documentation rather than the code.
    let code = || {
        src.lines()
            .map(str::trim_start)
            .filter(|l| !l.starts_with("//"))
    };
    let total = code().filter(|l| l.contains("MiniMaxH3Dit::load(")).count();
    let by_task = code()
        .filter(|l| {
            l.contains("MiniMaxH3Dit::load(&self.root, task.partition(), &self.device, self.dtype)")
        })
        .count();
    assert_eq!(
        total, 2,
        "model.rs has {total} `MiniMaxH3Dit::load(` call site(s); the render path has exactly two \
         (the t2va/fl2va arm and the ref2va arm), which are mutually exclusive branches"
    );
    assert_eq!(
        by_task,
        total,
        "{} of {total} DiT load site(s) do not take their partition from `task.partition()` — a \
         hardcoded partition is how a ref2va render reads the wrong 66 GB",
        total - by_task
    );

    // ...and neither partition literal appears at a call site outside its own `const` declaration.
    for literal in ["\"transformer\"", "\"transformer_ref\""] {
        let uses: Vec<&str> = code()
            .filter(|l| l.contains(literal) && !l.contains("pub const"))
            .collect();
        assert!(
            uses.is_empty(),
            "the bare literal {literal} appears outside its const declaration: {uses:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Real weights
// ---------------------------------------------------------------------------------------------

/// The `[out]` float32 values of `PROBE` in one partition, read straight off the shards.
fn probe_bytes(root: &Path, partition: &str) -> Vec<f32> {
    let dir = root.join(partition);
    let shards = candle_gen::loader::sorted_safetensors(&dir, "minimax-h3 dit").unwrap();
    let w = candle_gen::Weights::from_files_filtered(&shards, &Device::Cpu, DType::F32, &[PROBE])
        .unwrap();
    w.require(PROBE)
        .unwrap_or_else(|e| panic!("{partition}/{PROBE}: {e}"))
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// **The two partitions really are byte-different and structurally identical.**
///
/// This is the premise every other real-weight assertion rests on, so it is measured rather than
/// assumed. If a future snapshot made them identical, the wrong-checkpoint test below would become
/// vacuous — and this fails first, saying so.
#[test]
#[ignore = "reads shard headers plus one tensor from each 66 GB DiT partition (MINIMAX_H3_SNAPSHOT)"]
fn the_two_partitions_are_indistinguishable_except_by_value() {
    let root = snapshot();
    let base = root.join(BASE_DIT_PARTITION);
    let reference = root.join(REFERENCE_DIT_PARTITION);

    let base_cfg = std::fs::read(base.join("config.json")).unwrap();
    let ref_cfg = std::fs::read(reference.join("config.json")).unwrap();
    assert_eq!(
        base_cfg, ref_cfg,
        "the two partitions' config.json are supposed to be byte-identical; if they diverged, the \
         selection could be made structurally and this whole file is over-engineered"
    );

    let base_keys: BTreeSet<String> = candle_gen::loader::sorted_safetensors(&base, "dit")
        .unwrap()
        .iter()
        .flat_map(|p| safetensors_keys(p))
        .collect();
    let ref_keys: BTreeSet<String> = candle_gen::loader::sorted_safetensors(&reference, "dit")
        .unwrap()
        .iter()
        .flat_map(|p| safetensors_keys(p))
        .collect();
    assert_eq!(
        base_keys, ref_keys,
        "the two partitions ship the same tensor names"
    );
    println!("  both partitions carry {} tensors", base_keys.len());

    // …and only the VALUES differ.
    let a = probe_bytes(&root, BASE_DIT_PARTITION);
    let b = probe_bytes(&root, REFERENCE_DIT_PARTITION);
    assert_eq!(a.len(), b.len(), "{PROBE} has the same shape in both");
    assert_ne!(
        a, b,
        "{PROBE} is identical in both partitions, so it cannot discriminate them; pick another \
         probe before trusting the wrong-checkpoint test"
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
    let root = snapshot();
    let expected = probe_bytes(&root, REFERENCE_DIT_PARTITION);
    let base = probe_bytes(&root, BASE_DIT_PARTITION);
    assert_ne!(expected, base, "premise: the probe discriminates");

    // The task -> partition mapping is what the render path uses; it is NOT hardcoded here.
    let partition = MiniMaxH3Task::Ref2va.partition();
    assert_eq!(partition, REFERENCE_DIT_PARTITION);

    let dit = MiniMaxH3Dit::load(&root, partition, &Device::Cpu, DType::BF16).unwrap();
    let loaded = dit
        .projections()
        .proj_in
        .bias()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(
        loaded, expected,
        "a ref2va load must carry transformer_ref's weights"
    );
    assert_ne!(
        loaded, base,
        "a ref2va load must NOT carry transformer/'s weights — this is the assertion that fails \
         if the wrong 66 GB checkpoint was read"
    );
}
