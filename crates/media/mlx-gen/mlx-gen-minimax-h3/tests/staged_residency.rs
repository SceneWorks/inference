//! sc-17151 — **the staged-residency tripwire: no phase holds the previous phase's weights.**
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<upstream snapshot root> \
//!   cargo test -p mlx-gen-minimax-h3 --test staged_residency -- --ignored --nocapture
//! ```
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
/// rows, the latents — which are hundreds of MB at the geometries this model runs. Every component
/// this file sheds is at least 0.6 GB and the two big ones are 66 GB, so the allowance discriminates
/// a leftover output from a leftover component by more than an order of magnitude on every phase.
const RESIDUAL_ALLOWANCE: u64 = 2 << 30;

/// Slack over a component's own size that its materialization peak may reach: allocator alignment
/// and the reduction transients, not another component. 1/8 of 66 GB is 8 GB; the smallest thing a
/// held prior phase would add here is 11 GB.
fn peak_ceiling(bytes: u64) -> u64 {
    bytes + bytes / 8 + RESIDUAL_ALLOWANCE
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
fn tensor_bytes(t: &mlx_rs::Array) -> u64 {
    let elems: u64 = t.shape().iter().map(|&d| d as u64).product();
    let width = match t.dtype() {
        mlx_rs::Dtype::Float64 => 8u64,
        mlx_rs::Dtype::Float32 | mlx_rs::Dtype::Int32 | mlx_rs::Dtype::Uint32 => 4,
        mlx_rs::Dtype::Int8 | mlx_rs::Dtype::Uint8 | mlx_rs::Dtype::Bool => 1,
        _ => 2,
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
/// Both forcing steps are load-bearing and were established by sc-17145: `eval` alone can leave a
/// mapped tensor unrealized, and the `.item()` reduction is what actually reads the bytes.
fn force(dirs: &[PathBuf], take: Option<usize>) -> Resident {
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
    for t in &held {
        let s = t.sum(None).unwrap();
        mlx_rs::transforms::eval([&s]).unwrap();
        let _ = s.item::<f32>();
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
