//! sc-17151 — **the staged-residency tripwire: no phase holds the previous phase's weights.**
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<upstream snapshot root> \
//!   cargo test -p mlx-gen-minimax-h3 --test staged_residency -- --ignored --nocapture
//! ```
//!
//! # Two arms run in CI; the rest are OPERATOR-ONLY, and that is a disk fact
//!
//! [`the_video_decoder_hands_off_cleanly`] and [`a_held_decoder_trips_both_handoff_gates`] are
//! selected by name from the `mlx-minimax-h3` job in `.github/workflows/real-weights.yml`, and
//! `test_minimax_h3_lanes_select_tests_that_exist_and_pin_their_run_count` binds those two names on
//! every PR. They are what CI pulls because `release/real-weight-models.toml` fetches an 11.640 GB
//! slice of the 196 GB repository that holds `vae/` and `audio_vae/` whole and carries **neither
//! `transformer/` nor `text_encoder/` shards** — and every arm below that touches either would fail
//! inside `Weights::from_dir` rather than at a gate. Those arms, plus every `#[ignore]`d arm of
//! `ref2va_checkpoint.rs` and `te_tier_generate_stages.rs`, are **operator-only**: they run where
//! someone holds the full snapshot, with the command above.
//!
//! [`a_phase_that_skips_the_reduction_fails_the_materialization_gate`] is the exception on both
//! counts — it reads only `audio_vae/`, so the lane's bytes could run it, and it is deliberately not
//! selected: it is a control arm whose *expected* outcome is unmeasured (see [`force`]), and a lane
//! is not where an open question belongs.
//!
//! # The mechanism shipped before the enforcement did
//!
//! `crate::model`'s render paths have always been staged — text encoder forced then released, the
//! keyframe VAE released before the DiT is mapped, the DiT released before the decoder is loaded.
//! Nothing failed if one of those releases stopped happening. `tests/te_tier_generate_stages.rs`
//! *printed* a per-stage table and asserted only that the stages existed; a printed table is not an
//! assertion. This file is the assertion, and `te_tier_generate_stages.rs` now carries the same
//! predicates over the shipped `generate`.
//!
//! # Three ways the measurement lies, and what each one forces here
//!
//! 1. **Lazy mmap.** `MiniMaxH3Dit::load` on the 66 GB `transformer_ref` leaves peak device memory
//!    at **33 KB** — MLX maps the shards and materializes a tensor on first use. A tripwire that
//!    read the peak after a bare load would pass on a build that loaded every component at once.
//!    [`force`] therefore evaluates *and reduces* every tensor, and [`was_materialized`] fails the
//!    phase if the peak never reached half the component, so the gate cannot silently revert to a
//!    no-op.
//! 2. **`get_peak_memory` is ACTIVE, not cache.** So is `get_active_memory`. sc-17145 measured a
//!    shed reporting `active 151.4 -> 11.2 MiB` — a complete success — with 147.7 MiB sitting in
//!    MLX's own allocator cache and RSS unmoved. Every handoff here is therefore gated on
//!    [`footprint`] = **active + cache**, which is the only reading a drain that never drained can
//!    not satisfy.
//! 3. **A single `clear_cache()` is not the drain.** The release path calls
//!    [`mlx_gen::residency::drain_allocator_cache`], which retries while active keeps falling.
//!
//! # What the gates can and cannot show
//!
//! **Can**: that a component of a given size was genuinely materialized, that releasing it returns
//! the process footprint to its baseline, and that the next component's peak is one component
//! rather than two. **Cannot**: anything about a code path this file does not drive — the shipped
//! render order is pinned by `te_tier_generate_stages.rs`, and the structural
//! "there is only one DiT load site" property by `ref2va_checkpoint.rs`.
//!
//! # THIS IS A LARGE-LEAK GATE. Its floor is ~2.15 GB, and that is deliberate
//!
//! sc-17151 asked for a tripwire on *component* residency — a phase holding its predecessor's 66 GB
//! weights — so the thresholds are sized to components. The arithmetic is stated here rather than
//! left implied, because "every handoff is protected" reads as a general claim and is not one.
//!
//! [`handoff_is_clean`] takes `RESIDUAL_ALLOWANCE.min(bytes / 2)`, and the **smallest thing any
//! gated phase here sheds is the ~8.5 GB bounded slab** (the rest are 10.42, 11.02, 66.28 and
//! 66.72 GB), so `bytes / 2` is never below 4.25 GB, the `min` always selects the constant, and the
//! threshold is always the flat **2.147 GB**. Concretely invisible to it, each a figure measured
//! elsewhere in this crate:
//!
//! | leak | size | under the 2.147 GB floor by |
//! |---|---|---|
//! | the AdaLN projection table | 0.164 GB | 13.1x |
//! | the measured AdaLN precompute transient (218,988,544 B) | 0.219 GB | 9.8x |
//! | the `audio_vae/` decode half | 0.242 GiB / 0.26 GB | 8.3x |
//!
//! `tests/adaln_evict_memory.rs` and `tests/adaln_evict_real_weights.rs` are what gate the AdaLN
//! sizes; nothing in *this* file would notice any of the three going resident. Reaching them needs
//! a different instrument — a per-phase output budget rather than one constant, since a released
//! phase legitimately leaves hundreds of MB of text context, condition rows or latents behind — and
//! is deliberately not attempted here.
//!
//! `a_held_component_trips_both_handoff_gates` is the control arm: it holds a phase's weights
//! across the boundary on purpose and asserts the gates' **own predicates** come back false. Both
//! arms call the same [`handoff_is_clean`] / [`peak_is_one_component`] functions, so the control
//! cannot drift away from the gate it is supposed to be validating.

mod common;

use std::path::{Path, PathBuf};

use mlx_rs::memory::{get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory};

use mlx_gen::weights::Weights;

/// Bytes the process may still be carrying after a phase releases, over the baseline it started
/// from, before the handoff counts as dirty.
///
/// A released phase legitimately leaves its *output* behind — a text context, a set of condition
/// rows, the latents — which are hundreds of MB at the geometries this model runs.
///
/// The discrimination it actually buys, as ratios rather than as "an order of magnitude" (which was
/// true of the 66 GB phases and of nothing else): the smallest full-scale phase sheds the 11.02 GB
/// decode pair, so **5.1x**; the bounded slab arms shed ~8.5 GB, so **~4x**. And the `bytes / 2`
/// branch of [`handoff_is_clean`] would take over below a ~4.3 GB component, where the margin
/// collapses toward **2x** — no arm here is that small today, so the constant is what binds
/// everywhere. See the module header for what this floor cannot see at all.
const RESIDUAL_ALLOWANCE: u64 = 2 << 30;

/// Slack over a component's own size that its materialization peak may reach: allocator alignment
/// and the reduction transients, not another component.
///
/// **1/8, and nothing added.** [`RESIDUAL_ALLOWANCE`] used to be a third term here and it should not
/// have been: peak already tolerates a leftover output by construction (the leftover is part of the
/// `bytes` the next phase is measured against), so adding it again spent the discrimination twice.
/// With it, the ceiling on the 66.28 GB DiT was 76.71 GB — 10.43 GB of slack against the 11.02 GB
/// decode pair, a **0.59 GB / 5% margin**, which a slightly larger decode pair or a quantized DiT
/// tier would have flipped to a pass. Without it the slack is 8.29 GB and the margin is **2.74 GB /
/// 33%**, and the bound is the same `bytes + bytes / 8` that `ref2va_checkpoint.rs`'s
/// `two_full_checkpoints_are_never_co_resident` measured against at 66 GB scale.
fn peak_ceiling(bytes: u64) -> u64 {
    bytes + bytes / 8
}

/// **The handoff gate's predicate.** `residual` is the footprint growth over the baseline after the
/// phase released; `bytes` is what the phase had resident.
///
/// One function, called by the gate *and* by the control arm, so "the gate would have caught a held
/// component" is measured rather than asserted in a comment.
fn handoff_is_clean(residual: u64, bytes: u64) -> bool {
    residual < RESIDUAL_ALLOWANCE.min(bytes / 2)
}

/// **The accumulation gate's predicate**: this phase's peak is one component, not two.
fn peak_is_one_component(peak: u64, bytes: u64) -> bool {
    peak < peak_ceiling(bytes)
}

/// **The lazy-mmap gate's predicate**: the phase actually put its bytes on the device.
///
/// Without this, every other gate here passes unconditionally on a build that materialized nothing
/// — the 33 KB reading in the module docs is what that looks like.
fn was_materialized(peak: u64, bytes: u64) -> bool {
    peak > bytes / 2
}

/// Active **plus cached** device memory: the footprint a release has to actually give back.
///
/// `get_active_memory` alone reports a shed as complete while the buffers sit in MLX's allocator
/// cache with RSS unmoved (sc-17145). Summing the two is what makes a no-op drain visible.
fn footprint() -> u64 {
    (get_active_memory() + get_cache_memory()) as u64
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Bytes one tensor occupies once materialized. The dtype is **read**, not assumed, so a precision
/// change moves the bound rather than silently invalidating it.
///
/// **Exhaustive, with no `_` arm.** A catch-all sized the 8-byte dtypes (`Int64`, `Uint64`,
/// `Complex64`) at 2, undercounting them 4x, and `bytes` is on both sides of the gates: too small a
/// `bytes` loosens [`was_materialized`] and tightens [`peak_is_one_component`] into a false red.
/// `mlx_rs::Dtype` is not `#[non_exhaustive]`, so listing every variant makes a future dtype a
/// compile error here rather than a silent 4x error in a bound.
fn tensor_bytes(t: &mlx_rs::Array) -> u64 {
    use mlx_rs::Dtype::*;
    let elems: u64 = t.shape().iter().map(|&d| d as u64).product();
    let width = match t.dtype() {
        Complex64 | Float64 | Int64 | Uint64 => 8u64,
        Float32 | Int32 | Uint32 => 4,
        Bfloat16 | Float16 | Int16 | Uint16 => 2,
        Bool | Int8 | Uint8 => 1,
    };
    elems * width
}

/// A phase's weights, materialized and held.
struct Resident {
    held: Vec<mlx_rs::Array>,
    bytes: u64,
}

/// Materialize a phase's components and hold them.
///
/// `take` bounds how many tensors per directory are forced, so the cheap arms of this file drive
/// the identical code path over a few GB instead of 66. `None` is the whole component.
///
/// The `.item()` reduction is the step sc-17145 established as load-bearing: `eval` alone can leave
/// a mapped tensor unrealized, and the reduction is what actually reads the bytes.
///
/// The bulk `eval(refs)` below it is **not** separately established, and this used to claim both
/// steps were. Every tensor is retained in `held` and then individually `sum` → `eval` → `item`ed,
/// so the bulk pass may be entirely redundant; it is kept because it is the shape the production
/// loaders use, not because removing it was measured to change any reading here.
/// `a_phase_that_skips_the_reduction_fails_the_materialization_gate` is what would settle it — if
/// that arm passes the reduction is the whole of the forcing, and if it reds the bulk `eval` is.
fn force(dirs: &[PathBuf], take: Option<usize>) -> Resident {
    force_with(dirs, take, true)
}

/// [`force`], with the `.item()` reduction switchable off.
///
/// `reduce: false` is the shape a build that stopped forcing has — it maps and `eval`s and puts
/// (nearly) nothing on the device — and it exists so
/// `a_phase_that_skips_the_reduction_fails_the_materialization_gate` can drive
/// [`was_materialized`] to **false** through the same loader the real arms use. Nothing in
/// production calls it with `false`.
fn force_with(dirs: &[PathBuf], take: Option<usize>, reduce: bool) -> Resident {
    let mut held: Vec<mlx_rs::Array> = Vec::new();
    let mut bytes = 0u64;
    for dir in dirs {
        let w = Weights::from_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        // Sorted so a bounded `take` selects the same tensors on every run rather than whatever the
        // hash map happened to yield — a run-to-run different slab would make the bounds noise.
        let mut keys: Vec<String> = w.keys().map(str::to_owned).collect();
        keys.sort_unstable();
        if let Some(n) = take {
            keys.truncate(n);
        }
        assert!(
            !keys.is_empty(),
            "{} yielded no tensors — the bounds below would all be vacuous",
            dir.display()
        );
        for k in &keys {
            let t = w.require(k).unwrap_or_else(|e| panic!("{k}: {e}")).clone();
            bytes += tensor_bytes(&t);
            held.push(t);
        }
    }
    let refs: Vec<&mlx_rs::Array> = held.iter().collect();
    mlx_rs::transforms::eval(refs).unwrap();
    if reduce {
        for t in &held {
            let s = t.sum(None).unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
            let _ = s.item::<f32>();
        }
    }
    Resident { held, bytes }
}

/// Run one staged phase and gate it: materialize, check it was real, release through the
/// **production** drain, check the process footprint came back.
///
/// Returns the phase's peak, so the caller can print what it measured.
fn phase(name: &str, dirs: &[PathBuf], take: Option<usize>, baseline: u64) -> (u64, u64) {
    reset_peak_memory();
    let resident = force(dirs, take);
    let bytes = resident.bytes;
    let peak = get_peak_memory() as u64;

    assert!(
        was_materialized(peak, bytes),
        "{name}: peak {:.2} GB against a {:.2} GB component — nothing was materialized, so every \
         other gate in this phase would pass on a no-op (a bare load reads 33 KB)",
        gb(peak),
        gb(bytes)
    );
    assert!(
        peak_is_one_component(peak, bytes),
        "{name}: peak {:.2} GB exceeds the {:.2} GB ceiling for a {:.2} GB component — this phase \
         is holding a previous phase's weights",
        gb(peak),
        gb(peak_ceiling(bytes)),
        gb(bytes)
    );

    // The production release path, byte for byte: drop the handles, then the retried drain.
    drop(resident.held);
    mlx_gen::residency::drain_allocator_cache();

    let after = footprint();
    let residual = after.saturating_sub(baseline);
    assert!(
        handoff_is_clean(residual, bytes),
        "{name}: {:.2} GB of active+cached memory is still held over the {:.2} GB baseline after \
         releasing a {:.2} GB component. `get_active_memory` alone reporting success while the \
         buffers sit in MLX's allocator cache is exactly the sc-17145 failure, which is why this \
         gate reads active + cache",
        gb(residual),
        gb(baseline),
        gb(bytes)
    );
    (peak, bytes)
}

/// Settle the allocator and read the footprint every phase is measured against.
fn baseline() -> u64 {
    mlx_gen::residency::drain_allocator_cache();
    reset_peak_memory();
    footprint()
}

fn components(root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) {
    (
        vec![root.join("text_encoder")],
        vec![root.join("transformer")],
        vec![root.join("vae"), root.join("audio_vae")],
    )
}

// ── the two arms CI can actually run ───────────────────────────────────────────────────────────
//
// The bounded slab arms below look like the cheap ones to wire into a lane, and they are cheap in
// MEMORY (60 tensors each) and not in DISK: `Weights::from_dir` opens the whole component directory,
// and `release/real-weight-models.toml` provisions a deliberate 11.640 GB slice of the 196 GB
// `minimax-h3` repository carrying NO `transformer/` shards and NO `text_encoder/` shards (only the
// TE index). They would fail inside `from_dir`, before reaching a gate; provisioning the shards is
// +124 GB on both Macs and is a separate decision from this story.
//
// The two arms here run the identical `phase` helper and the identical predicates over the
// components that slice DOES carry whole: `vae/` (10.42 GB) and `audio_vae/` (0.605 GB).

/// **The CI-runnable clean arm.** Two successive materializations of the video VAE, gated exactly as
/// the full-scale phases are.
///
/// The second phase re-loads the *same* component rather than a different one because the lane's
/// slice carries only one component over 1 GB. That is a constraint, not a preference, and what it
/// costs is worth naming: if MLX ever returned phase A's buffers for phase B, phase B's peak would
/// read as one component and this arm would pass while the control arm below went red. Each phase
/// is a separate [`Weights::from_dir`] over freshly mapped shards, so it should allocate its own
/// buffers — but that is an expectation about the runtime, not something this arm proves.
///
/// What it does buy: both phases are exactly the same size, so `peak_ceiling` is exact for each
/// rather than approximate for a small one.
///
/// What goes red: a `drain_allocator_cache` that stopped draining leaves phase A's 10.42 GB in the
/// allocator cache, which `handoff_is_clean` reads (active **+ cache**) and fails on.
#[test]
#[ignore = "materializes the 10.42 GB video VAE twice in turn (MINIMAX_H3_SNAPSHOT)"]
fn the_video_decoder_hands_off_cleanly() {
    let root = common::snapshot();
    let vae = vec![root.join("vae")];

    let base = baseline();
    let (peak_a, bytes_a) = phase("decode A (vae)", &vae, None, base);
    let (peak_b, bytes_b) = phase("decode B (vae, reloaded)", &vae, None, base);
    assert_eq!(
        bytes_a, bytes_b,
        "the same component loaded twice must size identically, or the two phases are not \
         comparable and the bound below means something else"
    );
    println!(
        "video VAE handoff: {:.2} GB component | phase A peak {:.2} GB | phase B peak {:.2} GB | \
         baseline {:.2} GB",
        gb(bytes_a),
        gb(peak_a),
        gb(peak_b),
        gb(base)
    );
}

/// **The CI-runnable control arm: both handoff predicates come back false on a held component.**
///
/// The counterpart of [`a_held_component_trips_both_handoff_gates`] on the lane's own bytes. It
/// holds the 10.42 GB video VAE across the boundary and materializes the 0.605 GB audio VAE behind
/// it, so both predicates are violated by a wide margin (residual 10.42 GB against a 2.147 GB
/// allowance; peak ~11.0 GB against a 0.68 GB one-component ceiling) rather than by a few percent.
#[test]
#[ignore = "holds the 10.42 GB video VAE while materializing the audio VAE (MINIMAX_H3_SNAPSHOT)"]
fn a_held_decoder_trips_both_handoff_gates() {
    let root = common::snapshot();
    let base = baseline();

    // Phase A, deliberately NOT released.
    reset_peak_memory();
    let held_a = force(&[root.join("vae")], None);
    let bytes_a = held_a.bytes;
    assert!(
        was_materialized(get_peak_memory() as u64, bytes_a),
        "the control arm materialized nothing, so it proves nothing"
    );

    mlx_gen::residency::drain_allocator_cache();
    let residual = footprint().saturating_sub(base);
    assert!(
        !handoff_is_clean(residual, bytes_a),
        "holding a {:.2} GB component left only {:.2} GB of residual footprint, which the handoff \
         gate calls clean — the gate cannot see a held component",
        gb(bytes_a),
        gb(residual)
    );

    reset_peak_memory();
    let held_b = force(&[root.join("audio_vae")], None);
    let peak_b = get_peak_memory() as u64;
    assert!(
        !peak_is_one_component(peak_b, held_b.bytes),
        "a {:.2} GB peak while holding {:.2} GB from the previous phase and materializing {:.2} GB \
         still reads as one component — the accumulation gate is inert",
        gb(peak_b),
        gb(bytes_a),
        gb(held_b.bytes)
    );

    println!(
        "decoder control: held {:.2} GB, residual {:.2} GB, second-phase peak {:.2} GB",
        gb(bytes_a),
        gb(residual),
        gb(peak_b)
    );

    drop((held_a, held_b));
    mlx_gen::residency::drain_allocator_cache();
}

/// **The third predicate's control arm: [`was_materialized`] comes back false when nothing was.**
///
/// [`a_held_component_trips_both_handoff_gates`] deflects the other two, and without this one the
/// materialization gate is the only predicate in the file asserted exclusively in the direction that
/// passes — which is the shape a gate wired to a constant `true` would also have.
///
/// [`force_with`]`(.., false)` maps and `eval`s but skips the `.item()` reduction: the 33 KB shape
/// this file's header describes. Cheap — it is the arm that allocates nothing by construction.
#[test]
#[ignore = "maps the audio VAE without reducing it (MINIMAX_H3_SNAPSHOT)"]
fn a_phase_that_skips_the_reduction_fails_the_materialization_gate() {
    let root = common::snapshot();
    baseline();

    reset_peak_memory();
    let unreduced = force_with(&[root.join("audio_vae")], None, false);
    let peak = get_peak_memory() as u64;
    assert!(
        !was_materialized(peak, unreduced.bytes),
        "mapping a {:.2} GB component without reducing it still peaked at {:.2} GB, over half of \
         it — so the materialization gate cannot come back false and every use of it above is a \
         formality",
        gb(unreduced.bytes),
        gb(peak)
    );
    println!(
        "unreduced: peak {:.2} GB against a {:.2} GB mapped component",
        gb(peak),
        gb(unreduced.bytes)
    );

    drop(unreduced);
    mlx_gen::residency::drain_allocator_cache();
}

/// **The tripwire, at full component scale.** Conditioning → denoise → decode, each component
/// materialized whole, released through the production drain, and gated on both the handoff and the
/// accumulation predicate.
///
/// If any phase regressed to holding its predecessor's weights, `peak_is_one_component` fails on
/// the phase that inherited them and `handoff_is_clean` fails on the phase that kept them.
#[test]
#[ignore = "materializes the whole 66 GB text encoder, 66 GB DiT and 11 GB decoder pair in turn \
            (MINIMAX_H3_SNAPSHOT)"]
fn no_phase_holds_the_previous_phases_weights() {
    let root = common::snapshot();
    let (te, dit, decoders) = components(&root);

    let base = baseline();
    let (te_peak, te_bytes) = phase("conditioning (text_encoder)", &te, None, base);
    let (dit_peak, dit_bytes) = phase("denoise (transformer)", &dit, None, base);
    let (dec_peak, dec_bytes) = phase("decode (vae + audio_vae)", &decoders, None, base);

    println!("── staged residency, full component scale ─────────────────────");
    println!("  baseline                     {:>8.2} GB", gb(base));
    for (name, peak, bytes) in [
        ("conditioning (text_encoder)", te_peak, te_bytes),
        ("denoise (transformer)", dit_peak, dit_bytes),
        ("decode (vae + audio_vae)", dec_peak, dec_bytes),
    ] {
        println!(
            "  {name:<28} {:>8.2} GB peak   {:>8.2} GB component",
            gb(peak),
            gb(bytes)
        );
    }
    println!("  process (max of phases)      {:>8.2} GB", {
        gb(te_peak.max(dit_peak).max(dec_peak))
    });

    // **Peak is max(phase), not sum** — the property the whole staged design exists for, stated
    // against the measured phases rather than against a constant.
    let process = te_peak.max(dit_peak).max(dec_peak);
    let resident_sum = te_bytes + dit_bytes + dec_bytes;
    assert!(
        process < resident_sum / 2,
        "the process peak {:.2} GB is not meaningfully below the {:.2} GB all-resident sum — the \
         phases are accumulating",
        gb(process),
        gb(resident_sum)
    );
}

/// The same gates over a **bounded slab** of the same two components — cheap enough to re-run under
/// a source mutation, and driving the identical [`phase`] helper the full-scale test does.
///
/// This is the arm a production regression is verified against: making
/// `mlx_gen::residency::drain_allocator_cache` a no-op turns this red on the active+cache handoff
/// gate without a 144 GB run.
#[test]
#[ignore = "materializes a bounded slab of the text encoder and the DiT (MINIMAX_H3_SNAPSHOT)"]
fn a_bounded_slab_handoff_leaves_nothing_behind() {
    let root = common::snapshot();
    let (te, dit, _) = components(&root);

    let base = baseline();
    let (te_peak, te_bytes) = phase("conditioning slab", &te, Some(SLAB_TENSORS), base);
    let (dit_peak, dit_bytes) = phase("denoise slab", &dit, Some(SLAB_TENSORS), base);
    println!(
        "slab: te {:.2} GB peak / {:.2} GB | dit {:.2} GB peak / {:.2} GB | baseline {:.2} GB",
        gb(te_peak),
        gb(te_bytes),
        gb(dit_peak),
        gb(dit_bytes),
        gb(base)
    );
}

/// Tensors per component in the bounded arms. Enough that the slab is several GB — comfortably over
/// [`RESIDUAL_ALLOWANCE`], which is what makes the control arm below discriminating.
const SLAB_TENSORS: usize = 60;

/// **The control arm: a phase that holds its predecessor's weights fails both gates.**
///
/// The gates above are only worth having if they can come back false, and neither a peak nor a
/// footprint reading says so on its own. This materializes a slab, *keeps* it across the boundary,
/// materializes a second one, and asserts the gates' own predicates — [`handoff_is_clean`] and
/// [`peak_is_one_component`], the same functions [`phase`] calls — return **false**.
///
/// Run it and the full-scale test together: this one says the instrument deflects, that one says
/// the shipped components do not deflect it.
#[test]
#[ignore = "materializes two bounded slabs and holds both (MINIMAX_H3_SNAPSHOT)"]
fn a_held_component_trips_both_handoff_gates() {
    let root = common::snapshot();
    let (te, dit, _) = components(&root);

    let base = baseline();

    // Phase A, deliberately NOT released.
    reset_peak_memory();
    let held_a = force(&te, Some(SLAB_TENSORS));
    let bytes_a = held_a.bytes;
    assert!(
        was_materialized(get_peak_memory() as u64, bytes_a),
        "the control arm materialized nothing, so it proves nothing"
    );

    // The handoff gate, evaluated where the release should have been.
    mlx_gen::residency::drain_allocator_cache();
    let residual = footprint().saturating_sub(base);
    assert!(
        !handoff_is_clean(residual, bytes_a),
        "holding a {:.2} GB component left only {:.2} GB of residual footprint, which the handoff \
         gate calls clean — the gate cannot see a held component and every use of it above is \
         inert",
        gb(bytes_a),
        gb(residual)
    );

    // The accumulation gate, on a second phase materialized while the first is still held.
    reset_peak_memory();
    let held_b = force(&dit, Some(SLAB_TENSORS));
    let peak_b = get_peak_memory() as u64;
    assert!(
        !peak_is_one_component(peak_b, held_b.bytes),
        "a {:.2} GB peak while holding {:.2} GB from the previous phase and materializing {:.2} GB \
         still reads as one component — the accumulation gate is inert",
        gb(peak_b),
        gb(bytes_a),
        gb(held_b.bytes)
    );

    println!(
        "control: held {:.2} GB across the boundary, residual {:.2} GB, second-phase peak {:.2} GB",
        gb(bytes_a),
        gb(residual),
        gb(peak_b)
    );

    drop((held_a, held_b));
    mlx_gen::residency::drain_allocator_cache();
}
