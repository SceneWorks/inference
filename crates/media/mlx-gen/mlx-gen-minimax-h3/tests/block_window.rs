//! sc-18662 — **rung 4 (`BoundedTransformerResidency`) reachability and equivalence**, on the
//! committed DiT fixture.
//!
//! Runs in CI: no snapshot, no tier directory, no `#[ignore]`. The fixture's two-block stack is
//! staged into a `tempfile` directory with its own `config.json`, which is the whole apparatus a
//! [`DitBlockStream`] needs — it reopens a directory per window and rebuilds blocks out of it.
//!
//! # What a null memory result would not excuse
//!
//! Rung 4's peak claim needs real weights and is measured elsewhere. What is asserted here is the
//! part that must hold whatever the peak does, and that a memory measurement cannot see:
//!
//! | claim | asserted by |
//! |---|---|
//! | a deferred load holds **zero** blocks | [`a_deferred_load_materializes_no_blocks`] |
//! | a window produces the **same output** as the resident stack | [`windowed_and_resident_stacks_agree`] |
//! | the window is **reached** — every block really is re-materialized | [`windowed_and_resident_stacks_agree`] |
//! | `adaln_proj` is **not** in a denoise window | `block_stream`'s own unit test, plus the typed refusal below |
//! | the two residency modes cannot be silently mixed | [`the_two_residency_modes_are_mutually_exclusive`] |
//!
//! The `adaln_proj` one is the trap this family adds to the rung: re-reading it per window would
//! cost 39.3 % of the DiT every step and produce **bit-identical output**, so the equivalence test
//! below is exactly the test that cannot catch it.

mod common;

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::{AdaLnCache, DitBlock, MiniMaxH3DitConfig, MmRope};
use mlx_gen_minimax_h3::DitBlockStream;

use common::{dit_fixture_config, DIT_FIXTURE};

/// The fixture's model weights, written into `dir` as a component directory a stream can reopen.
///
/// A `DitBlockStream` is deliberately a **path**, not a loaded view, so exercising it needs a real
/// directory. `tempfile::TempDir` removes the tree while unwinding, which is the repo's guarded
/// temp-root contract (sc-17791).
fn stage_fixture(dir: &Path) -> MiniMaxH3DitConfig {
    let cfg = dit_fixture_config();
    // The fixture file verbatim, reference-side extras and all: `Weights::from_dir` maps lazily and
    // a stream only ever `require`s `transformer_blocks.{i}.…`, so the `src.`/`in.`/`out.`/`layout.`
    // entries cost nothing and re-writing a filtered copy would introduce a second serializer whose
    // agreement with the committed bytes nothing checks.
    std::fs::copy(DIT_FIXTURE, dir.join("model.safetensors")).unwrap();

    // The tier marker. `MiniMaxH3DitConfig::from_diffusers_json` is what a stream's consumers read,
    // so the fixture is staged through the same parser the shipped tiers go through rather than by
    // handing the config in directly — a divergence between the two would otherwise be invisible.
    let config = serde_json::json!({
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.num_refiner_layers,
        "ffn_dim": cfg.ffn_dim,
        "in_channels": cfg.in_channels,
        "audio_in_channels": cfg.audio_in_channels,
        "patch_size": cfg.patch_size,
        "text_dim": cfg.text_dim,
        "freq_dim": cfg.freq_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_dim,
        "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_freq_dim,
        "rope_theta": cfg.rope_theta,
        "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps,
        "final_norm_eps": cfg.final_norm_eps,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    cfg
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    a.subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item()
}

/// **The residency claim, as a count rather than as a byte figure.**
///
/// A deferred load must hold *no* materialized blocks. `resident_blocks()` is the observable, and it
/// is separate from `num_layers()` on purpose: the forward still runs all of them.
#[test]
fn a_deferred_load_materializes_no_blocks() {
    let staged = tempfile::tempdir().unwrap();
    let cfg = stage_fixture(staged.path());

    let resident =
        mlx_gen_minimax_h3::dit::MiniMaxH3Dit::load_dir(staged.path(), Dtype::Float32).unwrap();
    assert_eq!(resident.resident_blocks(), cfg.num_layers as usize);
    assert_eq!(resident.num_layers(), cfg.num_layers as usize);
    assert!(!resident.is_deferred());
    assert!(
        resident.holds_adaln(),
        "premise: a fresh resident load is pre-eviction"
    );

    let deferred =
        mlx_gen_minimax_h3::dit::MiniMaxH3Dit::load_dir_deferred(staged.path(), Dtype::Float32)
            .unwrap();
    assert_eq!(
        deferred.resident_blocks(),
        0,
        "a deferred load must hold no blocks — that is the entire rung"
    );
    assert_eq!(
        deferred.num_layers(),
        cfg.num_layers as usize,
        "it still RUNS every block; only the residency changed"
    );
    assert!(deferred.is_deferred());
    // `[].iter().all(..)` is `true`, so a naive implementation would report a deferred load as
    // pre-eviction — which would make every AdaLN assertion downstream read the wrong state.
    assert!(
        !deferred.holds_adaln(),
        "a streamed block is materialized body-only, so no block holds a projection"
    );
    assert!(deferred.stream().is_some());
}

/// **Equivalence and reachability in one test**, because either alone is a false green.
///
/// The window is driven at size 1 — the floor case, one block materialized at a time — and its
/// output must match the resident stack's exactly. Both arms run the identical
/// `DitBlock::forward_bounded` on the identical `AdaLnCache`, so the only difference is *when* each
/// block's weights exist; anything but zero here is a wiring error, not numerics.
///
/// Reachability is asserted from the plan rather than inferred: at window 1 over a 2-block stack the
/// driver must run two windows, and a plan that collapsed to one would make the equivalence above
/// trivially true with the streaming gone.
#[test]
fn windowed_and_resident_stacks_agree() {
    let staged = tempfile::tempdir().unwrap();
    let cfg = stage_fixture(staged.path());
    let f = Weights::from_file(DIT_FIXTURE).unwrap();

    let stream = DitBlockStream::new(staged.path(), Dtype::Float32, cfg.clone()).unwrap();
    assert_eq!(stream.n_blocks(), cfg.num_layers as usize);
    let plan = stream.plan(1).unwrap();
    assert!(
        plan.is_bounded() && plan.window_count() == cfg.num_layers as usize,
        "premise: window 1 over {} blocks must run {} windows, got {}",
        cfg.num_layers,
        cfg.num_layers,
        plan.window_count()
    );

    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let adaln_indices = f
        .require("layout.adaln_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();

    // The resident arm, block by block out of a fully loaded stack.
    let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    let mut resident_out = x.clone();
    for i in 0..cfg.num_layers {
        let block = DitBlock::from_weights(
            &mut w,
            &format!("transformer_blocks.{i}"),
            &cfg,
            Dtype::Float32,
        )
        .unwrap();
        let modulation = block.modulation(temb).unwrap();
        resident_out = block
            .forward(&resident_out, &modulation, &adaln_indices, &rope, &tables)
            .unwrap();
    }

    // The windowed arm: one block materialized at a time, out of a reopened directory.
    let mut windowed_out = x.clone();
    let mut materialized = 0usize;
    for range in plan.windows() {
        let mut view = stream.open().unwrap();
        for layer in range {
            let block = stream.materialize(&mut view, layer).unwrap();
            assert!(
                !block.holds_adaln(),
                "a denoise window materialized adaln_proj — 39.3 % of the DiT, re-read every step, \
                 for a table the precompute already holds. Bit-identical output, so nothing else \
                 here can see it."
            );
            // A body-only block cannot project its own modulation; that is the typed refusal, and
            // it is what forces the precompute to have run.
            assert!(
                block.modulation(temb).is_err(),
                "a body-only block must refuse to project rather than fabricate a table"
            );
            let modulation = {
                let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
                for prefix in ["src.", "in.", "out.", "layout."] {
                    w.remove_prefix(prefix);
                }
                DitBlock::from_weights(
                    &mut w,
                    &format!("transformer_blocks.{layer}"),
                    &cfg,
                    Dtype::Float32,
                )
                .unwrap()
                .modulation(temb)
                .unwrap()
            };
            windowed_out = block
                .forward(&windowed_out, &modulation, &adaln_indices, &rope, &tables)
                .unwrap();
            materialized += 1;
        }
        mlx_rs::transforms::eval([&windowed_out]).unwrap();
        drop(view);
        mlx_rs::memory::clear_cache();
    }
    assert_eq!(
        materialized, cfg.num_layers as usize,
        "every block must have been re-materialized"
    );

    let delta = max_abs(&resident_out, &windowed_out);
    eprintln!("[fixture rung 4, window 1] resident vs windowed max|Δ| {delta:.3e}");
    assert_eq!(
        delta, 0.0,
        "windowing changed the stack's output by {delta:e}; the two arms run the identical \
         arithmetic and differ only in when a block's weights exist"
    );

    // **The comparison is not vacuous.** Dropping one block from the windowed arm must move it far
    // off zero, so a test that accidentally compared an array with itself fails here.
    let mut one_short = x.clone();
    let mut view = stream.open().unwrap();
    let block = stream.materialize(&mut view, 0).unwrap();
    let modulation = {
        let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
        for prefix in ["src.", "in.", "out.", "layout."] {
            w.remove_prefix(prefix);
        }
        DitBlock::from_weights(&mut w, "transformer_blocks.0", &cfg, Dtype::Float32)
            .unwrap()
            .modulation(temb)
            .unwrap()
    };
    one_short = block
        .forward(&one_short, &modulation, &adaln_indices, &rope, &tables)
        .unwrap();
    let short_delta = max_abs(&resident_out, &one_short);
    assert!(
        short_delta > 1e-4,
        "mutation check: running one fewer block moved the metric only {short_delta:e}"
    );
}

/// A stack cannot be both resident and streamed, and neither can it be neither.
///
/// The failure this refuses is a deferred load that also kept its blocks: it would satisfy every
/// output assertion, report itself as rung 4, and bound nothing.
#[test]
fn the_two_residency_modes_are_mutually_exclusive() {
    let staged = tempfile::tempdir().unwrap();
    stage_fixture(staged.path());
    let resident =
        mlx_gen_minimax_h3::dit::MiniMaxH3Dit::load_dir(staged.path(), Dtype::Float32).unwrap();
    let deferred =
        mlx_gen_minimax_h3::dit::MiniMaxH3Dit::load_dir_deferred(staged.path(), Dtype::Float32)
            .unwrap();
    assert!(resident.stream().is_none() && resident.resident_blocks() > 0);
    assert!(deferred.stream().is_some() && deferred.resident_blocks() == 0);
}

/// The windowed AdaLN precompute must produce the **same cache** as the resident one.
///
/// This is the half of rung 4 that is peculiar to this family: every prior adopter windows only the
/// per-step trunk walk, while here the once-per-request projection pass is the larger residency and
/// has to be windowed too. Compared table by table against `AdaLnCache::precompute`, which is the
/// shipped resident path.
#[test]
fn the_windowed_adaln_precompute_matches_the_resident_one() {
    let staged = tempfile::tempdir().unwrap();
    let cfg = stage_fixture(staged.path());
    let f = Weights::from_file(DIT_FIXTURE).unwrap();
    let temb = f.require("in.temb").unwrap();
    // The fixture's `in.temb` is `[2, 48]`, so the schedule must carry exactly **two** distinct
    // timesteps across its four row classes — `AdaLnCache::precompute` validates the embedding
    // against `[num_distinct_timesteps, time_embed_dim]` and would reject a mismatch.
    assert_eq!(temb.shape(), &[2, cfg.time_embed_dim]);
    let schedule =
        mlx_gen_minimax_h3::dit::TimestepSchedule::new(vec![vec![0.9_f32, 0.9, 0.8, 0.8]]).unwrap();
    assert_eq!(schedule.num_distinct_timesteps(), 2);

    let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    let blocks: Vec<DitBlock> = (0..cfg.num_layers)
        .map(|i| {
            DitBlock::from_weights(
                &mut w,
                &format!("transformer_blocks.{i}"),
                &cfg,
                Dtype::Float32,
            )
            .unwrap()
        })
        .collect();
    let resident = AdaLnCache::precompute(&blocks, schedule.clone(), |_| Ok(temb.clone())).unwrap();

    let stream = DitBlockStream::new(staged.path(), Dtype::Float32, cfg.clone()).unwrap();
    let plan = stream.plan(1).unwrap();
    let cancel = mlx_gen::CancelFlag::default();
    let windowed =
        mlx_gen_minimax_h3::precompute_adaln_windowed(&stream, &plan, &cancel, schedule, temb)
            .unwrap();

    assert_eq!(windowed.num_layers(), resident.num_layers());
    assert_eq!(
        windowed.bytes(),
        resident.bytes(),
        "the windowed cache must retain the same table bytes; a different figure means a different \
         schedule or a dropped layer"
    );
    for layer in 0..resident.num_layers() {
        let (want, got) = (
            resident.modulation(layer).unwrap(),
            windowed.modulation(layer).unwrap(),
        );
        for (i, (a, b)) in want.tables().zip(got.tables()).enumerate() {
            let delta = max_abs(a, b);
            assert_eq!(
                delta, 0.0,
                "layer {layer} table {i}: the windowed projection moved by {delta:e}"
            );
        }
    }
}

/// **Every window must get its own view.** A stream that cached one and handed it out repeatedly
/// would satisfy every output assertion above while the release freed nothing — the failure
/// `gen_core::block_window::BlockWindowBackend::open_view` warns about in as many words ("a view
/// retained across windows keeps every materialized buffer alive through its own map").
///
/// Asserted behaviourally rather than structurally: `materialize` **drains** the handles it read, so
/// if two `open()` calls returned the same map the second window would find block 0's tensors gone.
/// A shared view therefore fails as a missing tensor, which is exactly the shape of the real defect.
#[test]
fn every_window_opens_an_independent_view() {
    let staged = tempfile::tempdir().unwrap();
    let cfg = stage_fixture(staged.path());
    let stream = DitBlockStream::new(staged.path(), Dtype::Float32, cfg).unwrap();

    let mut first = stream.open().unwrap();
    let _ = stream.materialize(&mut first, 0).unwrap();
    assert!(
        first.get("transformer_blocks.0.norm1.weight").is_none(),
        "premise: materialize must drain the handles it read, or a window keeps them resident"
    );

    let mut second = stream.open().unwrap();
    assert!(
        second.get("transformer_blocks.0.norm1.weight").is_some(),
        "the second window inherited the first window's drained map — every window is sharing one \
         view, and `run_windowed`'s release frees nothing"
    );
    stream.materialize(&mut second, 0).unwrap();
}

/// **The TE arm is reachable in CI, and a window is invisible in the output** (sc-18662 AC4).
///
/// Until this test the text-encoder window had only real-weight coverage, which CI never runs —
/// the arm's reachability rested on one `#[ignore]`d test. The fixture encoder is staged into a
/// directory exactly the way a tier is, loaded through the same `from_dir_deferred` +
/// `set_block_window` calls `model::build_te` makes for a streamed render, and compared against
/// the resident `from_weights` load — bit-identically, because the windowed walk runs the same
/// arithmetic and differs only in when a layer's weights exist.
///
/// Window sizes cover AC5's floor (1), a ragged tail (3 over 4 tapped layers), and the
/// all-covering plan (4), so the boundary bookkeeping of `run_layers`' carried activation is
/// exercised at every plan shape the scheduler can produce.
#[test]
fn the_windowed_te_matches_the_resident_one_on_the_fixture() {
    use mlx_gen_minimax_h3::text_encoder::MiniMaxH3TextEncoder;

    let staged = tempfile::tempdir().unwrap();
    std::fs::copy(common::TE_FIXTURE, staged.path().join("model.safetensors")).unwrap();
    // The tier marker `TeBlockStream::new` probes. Content is not read for the fixture prefix —
    // the config is handed in directly, as `te_parity.rs` does.
    std::fs::write(staged.path().join("config.json"), "{}").unwrap();
    let cfg = common::te_fixture_config();

    let w = Weights::from_file(common::TE_FIXTURE).unwrap();
    let ids = w.get("in.input_ids").unwrap().clone();
    let mask = w.get("in.attention_mask").unwrap().clone();

    let resident = MiniMaxH3TextEncoder::from_weights(&w, "language_model", &cfg).unwrap();
    assert!(!resident.is_deferred());
    let reference = resident.forward(&ids, &mask).unwrap();

    let tapped = cfg.select_hidden;
    for window in [1usize, 3, tapped] {
        let mut te =
            MiniMaxH3TextEncoder::from_dir_deferred(staged.path(), "language_model", &cfg).unwrap();
        assert!(te.is_deferred());
        assert_eq!(
            te.resident_layers(),
            0,
            "a deferred encoder must hold no layers — that is the residency claim"
        );
        assert_eq!(
            te.num_loaded_layers(),
            tapped,
            "the deferred walk must run the tapped depth, not the checkpoint's num_layers"
        );
        te.set_block_window(window, mlx_gen::CancelFlag::default())
            .unwrap();
        let out = te.forward(&ids, &mask).unwrap();
        assert_eq!(
            max_abs(&reference, &out),
            0.0,
            "window {window} changed the context — the windowed walk must be bit-identical"
        );
    }

    // The refusal `run_layers` types out: a deferred encoder without a window is a programming
    // error and must not silently fall back to anything.
    let deferred =
        MiniMaxH3TextEncoder::from_dir_deferred(staged.path(), "language_model", &cfg).unwrap();
    let err = deferred.forward(&ids, &mask).unwrap_err().to_string();
    assert!(err.contains("set_block_window"), "{err}");
}
