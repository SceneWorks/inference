//! Real-weight conformance and **measurement** for the Chroma1 shared memory ladder (SC-15520), on
//! Apple/Metal.
//!
//! Every number `crate::memory_strategy` publishes comes from this file. Nothing is inherited from
//! FLUX.1, from Kolors, or from any other family: the epic's standing rule is that a rung's
//! presence, magnitude, mechanism and candidate set are per family per backend, and Chroma is the
//! first **pruned-adaLN MMDiT with a T5-only conditioning phase** on this ladder.
//!
//! ## Measurement discipline (SC-17679)
//!
//! * **MLX's own accounting**, never timer-sampled RSS. `mlx_rs::memory::get_peak_memory` reports
//!   ACTIVE bytes; a sampled RSS measures how fast the machine happened to run.
//! * **A fresh generator per measured row** ([`measure`]). A reused heavy bundle lets the first row
//!   materialize the lazily-loaded stack, and every later row then reads a peak including work it
//!   did not do.
//! * **`reset_peak_memory` after the load**, so a row measures the *request*, not the load.
//! * **One discarded warm-up row of the same shape** before any published peak ([`warm_up`]) — and
//!   that alone is *not* sufficient, which is the whole lesson of SC-17679. Whether a cell can
//!   support a claim is answered by
//!   [`identical_requests_reproduce_once_the_allocator_has_settled`] (its resolution) and by
//!   [`probe_order`] (whether an apparent effect follows the cadence or the position).
//! * Rejected candidates are recorded **with their numbers**, and every rejection is re-asserted
//!   against the production path.
//!
//! ## Mutation proofs (SC-15520)
//!
//! Each rung's implementation was stubbed to its no-op path and the corresponding test confirmed to
//! REDDEN, then reverted. A test that cannot fail is worthless, and byte-identity assertions in
//! particular pass trivially with the feature off.
//!
//! | rung / claim | stub | reddened |
//! |---|---|---|
//! | 1 | `rung_plan` ignores `memory.stage_residency`, falls back to the load-time default | `staged_residency_bounds_the_request_peak_and_preserves_output` |
//! | 2 | the production refusal of the withheld rung removed | `the_withheld_rungs_are_refused_by_the_production_path` |
//! | 2 quality | the layer-wise decoder stops using full-image GroupNorm statistics | `layerwise_decode_quality_is_resampled_across_seeds` |
//! | 3 | the `sdpa` kernel discards the plan and always calls `AttentionPlan::UNBOUNDED` | `attention_chunking_is_measured_at_the_dit_seam` |
//! | 4 | `finalize_block_stream` and `block_window` stubbed to no-ops — rung 4 declared and not executed | `transformer_window_sweep_and_streamed_output_identity` |
//! | step-independence | [`measure`] scales the peak by the request's step count | `the_request_peak_is_step_independent` |
//! | cadence flatness | [`measure`] scales the peak by the selected window cadence | `transformer_window_sweep_and_streamed_output_identity` |
//!
//! The unit-level fixes from the same review are proven the same way, in their own crates:
//! `ArmedScope::get` forced to `None`, `resident_overlay_components` forced empty, the calibration
//! key's tier discriminant bypassed, and each `quantize`-after-arm guard disabled — every one
//! reddened its test and was reverted.
//!
//! ## Weights
//!
//! One env var per catalog entry, each pointing at that entry's snapshot **root** (the tier is a
//! subdirectory: `bf16` / `q4` / `q8`). Nothing self-fetches or derives a cache location
//! (epic 13657).
//!
//! Every test that names a **specific** entry/tier resolves it through [`require_tier`], which
//! panics `SKIPPED-BY-ABSENCE: <var>` rather than early-returning green. The one exception is
//! deliberate and is stated here rather than glossed: [`every_cached_entry_and_tier_publishes_its_own_evidence`]
//! sweeps whatever is present, **prints every absent cell by name**, and fails only if *nothing* was
//! measured. Its claim is per-cell — "each entry supplies its own evidence" — which an absent
//! snapshot cannot falsify, so it reports coverage instead of demanding it. Read its printed
//! `absent: [...]` line before treating a green run as full coverage.
//!
//! | env var | entry |
//! |---|---|
//! | `CHROMA_LADDER_BASE` | `chroma1_base` — the representative entry |
//! | `CHROMA_LADDER_HD` | `chroma1_hd` |
//! | `CHROMA_LADDER_FLASH` | `chroma1_flash` |
//!
//! ```text
//! CHROMA_LADDER_BASE=<snapshot root containing q4/> \
//!   cargo test -p mlx-gen-chroma --release --test memory_ladder_real_weights \
//!   -- --ignored --test-threads=1 --nocapture
//! ```

#![allow(clippy::items_after_test_module)]

use std::{collections::BTreeMap, path::PathBuf};

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, MemoryStrategy, MemoryStrategySupport,
    Progress, TransformerComponent,
};
use mlx_gen::memory::MEMORY_CAP_ENV;
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen_chroma::memory_strategy as ms;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The three catalog entries this provider serves, with the env var carrying each one's snapshot
/// root. They are one architecture and one loader; they differ in checkpoint, sampler default and
/// schedule, which is exactly why each still owes its own evidence.
const ENTRIES: &[(&str, &str)] = &[
    ("chroma1_base", "CHROMA_LADDER_BASE"),
    ("chroma1_hd", "CHROMA_LADDER_HD"),
    ("chroma1_flash", "CHROMA_LADDER_FLASH"),
];

/// The entry the single-entry measurements run against — `chroma1_base` by default, overridable with
/// `CHROMA_LADDER_ENTRY`.
///
/// Parameterised rather than hardcoded, because "the representative entry" is a convenience and not
/// a claim: sharing this provider's code is explicitly not what makes a sibling entry Verified, so
/// every rung-4 and scope conclusion has to be re-takeable per entry. `chroma1_hd` and
/// `chroma1_flash` are measured through exactly these tests with this variable set (sc-17695).
fn representative() -> &'static str {
    match std::env::var("CHROMA_LADDER_ENTRY").ok().as_deref() {
        Some("chroma1_hd") => "chroma1_hd",
        Some("chroma1_flash") => "chroma1_flash",
        Some("chroma1_base") | None => "chroma1_base",
        Some(other) => panic!("CHROMA_LADDER_ENTRY: unknown entry {other:?}"),
    }
}

/// The env var carrying [`representative`]'s snapshot root.
fn representative_env() -> &'static str {
    ENTRIES
        .iter()
        .find(|(entry, _)| *entry == representative())
        .map(|(_, var)| *var)
        .expect("every entry has an env var")
}

/// The tier every default-mode measurement runs at — the manifest's own `mlx.quantize: 4`, so the
/// asserted rows describe what a caller who names nothing actually gets.
const DEFAULT_TIER: &str = "q4";

/// The three advertised tiers, in catalog order.
const TIERS: &[&str] = &["q4", "q8", "bf16"];

/// Steps for a measured row — **the variant's real production schedule**, not a convenient short
/// one (review of PR #496).
///
/// An earlier revision hardcoded 4 and justified it by peak step-independence. That justification is
/// sound for a *peak* and worthless for a *quality* verdict, and rung 2's is a quality verdict: the
/// tiled-decode drift is a property of the latent the denoiser actually produces, and a 4-step
/// latent is not that latent. Re-measured at the real 28, the same 7x4 sweep's best cell moved from
/// 105/255 to 53 — the difference between "more than double the bar" and "10% over it", which is a
/// different claim about the same mechanism.
///
/// So the default binds to `ChromaVariant::Base::default_steps()`. `CHROMA_LADDER_STEPS` overrides
/// it for iteration, and [`the_request_peak_is_step_independent`] is what makes a shorter override
/// legitimate for the peak-bearing rows — it measures the claim instead of asserting it in prose.
fn steps() -> u32 {
    std::env::var("CHROMA_LADDER_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| mlx_gen_chroma::ChromaVariant::Base.default_steps())
}

/// The drift bar this family is judged against, in 8-bit levels out of 255.
///
/// SC-19753 retains 48/255 as the product decode-quality admission bar across the sealed coordinate
/// matrix. Historically it came from Z's old 48-admitted/64-rejected split; Z now admits every
/// measured edge with a much lower worst result, so that split is provenance rather than current
/// Z-domain evidence.
const SIBLING_DRIFT_BAR: u32 = 48;

/// One sealed policy coordinate resampled across seeds. Overlap remains part of policy identity but
/// no longer affects layer-wise decode arithmetic.
const MEASURED_POLICY_EDGE: u32 = 832;
const MEASURED_POLICY_OVERLAP: u32 = 256;

fn entry_root(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from)
}

fn tier_dir(var: &str, tier: &str) -> Option<PathBuf> {
    let root = entry_root(var)?;
    let dir = root.join(tier);
    dir.is_dir().then_some(dir)
}

#[track_caller]
fn require_tier(var: &str, tier: &str) -> PathBuf {
    match tier_dir(var, tier) {
        Some(dir) => dir,
        None => panic!("SKIPPED-BY-ABSENCE: set {var} to a snapshot root containing {tier}/"),
    }
}

/// The shipped tiers are already packed, so a tier is selected by pointing at its directory — never
/// by asking the loader to re-quantize a dense one. `quant_for` exists only for the fail-closed
/// precondition test, which needs exactly that refused combination.
fn spec(dir: &std::path::Path, shape: LoadShape) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(shape)
}

fn request(memory: Option<GenerationMemory>, edge: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, lowres".into()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(steps),
        seed: Some(1234),
        memory,
        ..Default::default()
    }
}

/// One measured row: the request's ACTIVE-bytes peak, its pixels, and its wall clock.
///
/// `wall` is here because rung 4's re-materialization latency is a real hazard on exactly the small
/// Macs the rung exists for, and a peak-only row cannot answer it. It is the softest number in this
/// file — it moves with thermal state — so it is *reported* and bounded loosely, never asserted to a
/// tight figure.
struct Row {
    peak_gib: f64,
    pixels: Vec<u8>,
    wall: std::time::Duration,
}

/// Render one row on a **fresh** generator and return its request peak.
///
/// The freshness is the whole contract of this helper. Chroma's DiT weights are lazy MLX handles
/// until something evaluates them, so a generator reused across rows carries the previous row's
/// materialization into this row's peak — exactly how a rung-4 sweep can report a saving that is
/// really just "the first row paid for the stack". Every row in this file goes through here.
#[track_caller]
fn measure(entry: &str, dir: &std::path::Path, shape: LoadShape, req: &GenerationRequest) -> Row {
    let registry = mlx_gen_chroma::provider_registry().expect("provider registry");
    let model = registry
        .load(entry, &spec(dir, shape))
        .unwrap_or_else(|error| panic!("load {entry}: {error}"));
    clear_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();
    let out = model
        .generate(req, &mut |_: Progress| {})
        .expect("generate must succeed");
    let peak = get_peak_memory();
    // Timed inside the reset/read window, so the row's clock covers exactly the request its peak
    // covers — the load is excluded from both.
    let wall = started.elapsed();
    let pixels = match out {
        GenerationOutput::Images(images) => images.first().expect("one image").pixels.clone(),
        other => panic!("expected images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    Row {
        peak_gib: peak as f64 / GIB,
        pixels,
        wall,
    }
}

fn ms_per_step(row: &Row, steps: u32) -> f64 {
    row.wall.as_secs_f64() * 1000.0 / f64::from(steps)
}

fn max_delta(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y) as u32)
        .max()
        .unwrap_or(0)
}

fn mean_delta(a: &[u8], b: &[u8]) -> f64 {
    let sum: u64 = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    sum as f64 / a.len() as f64
}

/// The all-rungs-off baseline: an explicit block, so the load-time `Sequential` policy cannot leak a
/// phase release into the "resident" row.
fn resident_memory() -> GenerationMemory {
    GenerationMemory::default()
}

fn staged() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..Default::default()
    }
}

/// The **rung-2** production composition. Note it does NOT stage: the shared cost order deliberately
/// excludes rung 1 (bounding residency may evict the warm cross-request pair, a cost the next
/// request pays), so a selector choosing rung 2 gets tiling and nothing else — see
/// `MemoryStrategy::engages`. Measuring rung 2 with staging bolted on would credit it with rung 1's
/// saving.
fn rung2(edge: u32, overlap: u32) -> GenerationMemory {
    GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        ..Default::default()
    }
}

/// The **rung-3** production composition: rung 3 engages rung 2 by cost order, still not rung 1 —
/// *and only where rung 2 is `Implemented`*. `MemoryProviderContract::engages` does not engage a
/// rung the provider declares `Missing`, so a helper that hardcoded the tiling would compose a
/// request the selector can never produce and the production path refuses.
fn rung3() -> GenerationMemory {
    GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(ms::ATTENTION_CHUNK_SIZE),
        ..if ms::DECODE_SUPPORT {
            rung2(ms::DECODE_TILE_EDGE, ms::DECODE_OVERLAP)
        } else {
            GenerationMemory::default()
        }
    }
}

/// The **rung-4 control**: everything rung 4's composition engages *except* the window itself, which
/// is the only way to isolate what the window buys. Rung 4 engages rungs 2 and 3 by cost order —
/// again only where they are `Implemented` — and rung 1 through this provider's declared
/// `EngagedInSameRequest` prerequisite.
fn rung4_control() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..if ms::ATTENTION_SUPPORT {
            rung3()
        } else if ms::DECODE_SUPPORT {
            rung2(ms::DECODE_TILE_EDGE, ms::DECODE_OVERLAP)
        } else {
            GenerationMemory::default()
        }
    }
}

/// The full rung-4 composition at a given cadence.
fn full_ladder(window: u32) -> GenerationMemory {
    full_ladder_scoped(window, ms::TRANSFORMER_WINDOW_COMPONENT)
}

fn full_ladder_scoped(window: u32, component: TransformerComponent) -> GenerationMemory {
    GenerationMemory {
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(component),
        ..rung4_control()
    }
}

/// Discard one measured row before publishing any peak from this process (SC-17679).
///
/// **This is a measurement-integrity control, not hygiene.** `get_peak_memory` reads ACTIVE bytes,
/// and the very first `generate` in a process reads them against a cold allocator. The sibling
/// Kolors harness measured that bias directly and it is not small: a windowed row measured first
/// read 4.4632 GiB for a configuration that reads 4.6924 once warm — a 4.9% phantom spread that
/// looked exactly like a real finding — and an SDXL seam pair measured cold published a +12.54%
/// regression which is +0.00% warm.
///
/// **A warm-up alone is not sufficient discipline.** Discarding one row removes the cold-allocator
/// bias on the row that follows, but where the reading is indexed by ordinal it merely shifts every
/// value one slot. That residual question is answered by
/// [`identical_requests_reproduce_once_the_allocator_has_settled`] and [`probe_order`], not here.
///
/// The warm-up is deliberately a *windowed* row, because that is the shape the bias was observed on
/// (rung 4 calls `clear_cache()` at every window boundary, which is what interacts with the cold
/// allocator). It runs at the smallest advertised output for one step.
#[track_caller]
fn warm_up(entry: &str, dir: &std::path::Path) {
    let _ = measure(
        entry,
        dir,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
    );
}

/// The order the cadence sweep executes its rows in — **the control that tells a cadence effect
/// apart from a positional one** (SC-17679).
///
/// Defaults to [`ms::TRANSFORMER_WINDOW_SIZES_SWEPT`]' own order. `CHROMA_WINDOW_PROBE_ORDER`
/// overrides it with a comma-separated permutation. If the peaks follow the *positions* rather than
/// the cadences, the cell is unresolvable and must be withdrawn as evidence — a genuine
/// weight-residency effect cannot move with execution order.
///
/// This is a permanent instrument, not scaffolding: a single discarded warm-up row does not remove
/// a positional bias, it shifts it by one slot.
fn probe_order() -> Vec<u32> {
    let Ok(spec) = std::env::var("CHROMA_WINDOW_PROBE_ORDER") else {
        return ms::TRANSFORMER_WINDOW_SIZES_SWEPT.to_vec();
    };
    let order: Vec<u32> = spec
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("CHROMA_WINDOW_PROBE_ORDER: not a u32")
        })
        .collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    let mut domain = ms::TRANSFORMER_WINDOW_SIZES_SWEPT.to_vec();
    domain.sort_unstable();
    assert_eq!(
        sorted, domain,
        "CHROMA_WINDOW_PROBE_ORDER must be a PERMUTATION of the swept domain — an order probe that \
         also changed which cadences ran would confound the two things it exists to separate"
    );
    println!("[sc-17679 order probe] executing cadences in the order {order:?}");
    order
}

fn probe_tier() -> String {
    std::env::var("CHROMA_WINDOW_PROBE_TIER").unwrap_or_else(|_| DEFAULT_TIER.to_owned())
}

fn probe_size() -> u32 {
    std::env::var("CHROMA_WINDOW_PROBE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

fn probing() -> bool {
    probe_tier() != DEFAULT_TIER || probe_size() != 1024
}

// ── The instrument ───────────────────────────────────────────────────────────────────────────────

/// **Does this harness's peak reading depend on a row's ORDINAL rather than on its request?**
///
/// The instrument check the whole ladder rests on. It renders the **identical** request
/// `SETTLE_PROBE_ROWS` times on a fresh generator each time and prints every row's peak. A request
/// that is byte-for-byte the same each time can only produce different peaks if the reading is a
/// function of something other than the request.
///
/// The claim this asserts is not "MLX always reproduces". It is "**the tolerance the flatness
/// assertion uses is not finer than the instrument's resolution at the cell it runs on**". A cell
/// whose resolution is, say, 5.67% (SDXL measured exactly that at 512² q8) can support no cadence
/// claim at all, in either direction.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn identical_requests_reproduce_once_the_allocator_has_settled() {
    const SETTLE_PROBE_ROWS: usize = 8;
    /// The tolerance `transformer_window_sweep_and_streamed_output_identity`'s flatness assertion
    /// uses. The instrument must be at least this good wherever that assertion runs.
    const SETTLE_TOLERANCE: f64 = 0.01;

    let tier = probe_tier();
    let edge = probe_size();
    let dir = require_tier(representative_env(), &tier);
    // Eight rows at the 28-step production schedule is ~40 minutes for a quantity
    // [`the_request_peak_is_step_independent`] measures to be step-invariant (0.00% over 1, 4 and
    // 28 steps). `CHROMA_LADDER_STEPS` is the licensed shortening, and the license is a measurement
    // rather than a claim — which is the whole reason that test exists.
    let count = steps();

    let peaks: Vec<f64> = (0..SETTLE_PROBE_ROWS)
        .map(|_| {
            measure(
                representative(),
                &dir,
                LoadShape::DeferredMaterialization,
                &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), edge, count),
            )
            .peak_gib
        })
        .collect();
    println!(
        "[sc-17679 settle {tier} {edge}² cadence {} {count} step(s)] identical request \
         x{SETTLE_PROBE_ROWS}: {}",
        ms::TRANSFORMER_WINDOW_SIZE,
        peaks
            .iter()
            .enumerate()
            .map(|(i, p)| format!("#{}: {p:.4}", i + 1))
            .collect::<Vec<_>>()
            .join("  ")
    );

    // Row 1 is the one `warm_up` discards. Everything after it is a row this file would PUBLISH.
    let published = &peaks[1..];
    let (min, max) = published
        .iter()
        .fold((f64::MAX, 0f64), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let spread = (max - min) / min;
    println!(
        "[sc-17679 settle {tier} {edge}²] INSTRUMENT RESOLUTION {:.2}% over rows 2..={SETTLE_PROBE_ROWS} \
         ({min:.4}..{max:.4} GiB; row 1 is the discarded warm-up){}",
        100.0 * spread,
        if probing() {
            " — reported, not asserted (this is a per-cell property)"
        } else {
            ""
        }
    );
    if probing() {
        return;
    }
    assert!(
        spread < SETTLE_TOLERANCE,
        "identical requests no longer reproduce at the DEFAULT configuration: rows 2..= span \
         {min:.4}..{max:.4} GiB ({:.2}%), against a flatness tolerance of {:.0}%. The flatness \
         assertion would then be running finer than the instrument's own resolution and could not \
         tell a real cadence effect from allocator noise",
        100.0 * spread,
        100.0 * SETTLE_TOLERANCE
    );
}

/// **Is the request peak a function of the step count?**
///
/// The harness runs at the variant's real 28-step schedule by default, which makes a full sweep
/// expensive; `CHROMA_LADDER_STEPS` exists so a peak-bearing row can be taken at a shorter one. That
/// override is only legitimate if the peak really is step-independent, and an earlier revision of
/// this file asserted exactly that **in prose** while publishing every number from a 4-step run.
///
/// This measures it. The same composition at 1, 4 and the production 28 steps must land on the same
/// peak within the instrument's own resolution — a denoise loop that accumulated per-step residency
/// (a retained preview, an un-evaluated graph growing with the schedule) would show up here.
///
/// It is *not* a licence for a quality verdict at a short schedule: rung 2's drift is a property of
/// the latent, and this says nothing about that. See [`steps`].
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_request_peak_is_step_independent() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    warm_up(representative(), &dir);
    let production = mlx_gen_chroma::ChromaVariant::Base.default_steps();
    let mut rows = Vec::new();
    for count in [1_u32, 4, production] {
        let row = measure(
            representative(),
            &dir,
            LoadShape::EagerMaterialization,
            &request(Some(staged()), 1024, count),
        );
        println!(
            "[sc-15520 step-independence {DEFAULT_TIER} 1024²] {count} step(s): {:.4} GiB, \
             {:.0} ms/step",
            row.peak_gib,
            ms_per_step(&row, count)
        );
        rows.push((count, row.peak_gib));
    }
    let (min, max) = rows.iter().fold((f64::MAX, 0f64), |(lo, hi), (_, p)| {
        (lo.min(*p), hi.max(*p))
    });
    let spread = 100.0 * (max - min) / min;
    println!("[sc-15520 step-independence {DEFAULT_TIER} 1024²] spread {spread:.2}% over {rows:?}");
    assert!(
        spread < 1.0,
        "the request peak is NOT step-independent ({spread:.2}% over {rows:?}), so \
         CHROMA_LADDER_STEPS cannot be used to shorten a peak-bearing row and every peak in this \
         file must be re-taken at the production schedule"
    );
}

// ── Rung 0/1 ─────────────────────────────────────────────────────────────────────────────────────

/// **Rung 1 is request-scoped and it moves the request peak.**
///
/// One cached generator serves resident → staged, and the staged request must peak strictly lower
/// while producing a **byte-identical** image: rung 1 sheds the T5-XXL encoder before the DiT/VAE
/// load, which is a residency change and not an arithmetic one.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn staged_residency_bounds_the_request_peak_and_preserves_output() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    warm_up(representative(), &dir);
    let shape = LoadShape::EagerMaterialization;
    let resident = measure(
        representative(),
        &dir,
        shape,
        &request(Some(resident_memory()), 1024, steps()),
    );
    let staged_row = measure(
        representative(),
        &dir,
        shape,
        &request(Some(staged()), 1024, steps()),
    );
    println!(
        "[sc-15520 rung1 {DEFAULT_TIER} 1024²] resident {:.4} GiB -> staged {:.4} GiB ({:+.2}%)  \
         {:.0} -> {:.0} ms/step",
        resident.peak_gib,
        staged_row.peak_gib,
        100.0 * (staged_row.peak_gib - resident.peak_gib) / resident.peak_gib,
        ms_per_step(&resident, steps()),
        ms_per_step(&staged_row, steps()),
    );
    assert_eq!(
        max_delta(&resident.pixels, &staged_row.pixels),
        0,
        "rung 1 is a residency schedule, not an arithmetic change: the image must be byte-identical"
    );
    // **A margin, not a bare `<`.** A bare inequality passes on floating-point noise: with the
    // request-scoped resolver stubbed out both rows land on the same schedule and the two peaks
    // differ in the fourth decimal, which satisfies `<` about half the time. 3% is a floor a no-op
    // implementation cannot clear in either direction.
    assert!(
        staged_row.peak_gib < resident.peak_gib * 0.97,
        "staged residency must bound the request peak by a real margin: {:.4} vs {:.4} GiB",
        staged_row.peak_gib,
        resident.peak_gib
    );
}

/// **Which phase bears the request peak** — the measurement the rung-4 component scope turns on.
///
/// Not a `Dit`-by-inheritance decision (SC-15794 / the Kolors precedent in SC-15521): if the
/// conditioning phase binds, a `Dit`-scoped window is inert on the request peak no matter how well
/// it bounds the denoise, and the `TextEncoder`/`Both` scopes are the ones worth implementing.
///
/// Chroma's shipped q4 tier packs only the DiT block linears; the T5-XXL encoder ships **dense
/// bf16**, so this is not a hypothetical question for this family. sc-16462 (inference PR #443) is
/// the story that packs those auxiliaries, and when it lands this row is re-derived rather than
/// inherited.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_request_peak_bearing_phase_is_measured_not_assumed() {
    use mlx_gen_chroma::{loader, ChromaTransformerConfig};
    use mlx_rs::Array;

    let dir = require_tier(representative_env(), &probe_tier());
    let edge = probe_size();

    // Each phase is measured doing its REAL work at the production shape. MLX is lazy, so a
    // "load and read the peak" probe measures nothing at all — the handles exist and the bytes do
    // not until something forces evaluation.

    // Phase A: the T5-XXL encoder + the production prompt encode.
    clear_cache();
    reset_peak_memory();
    let tokenizer = loader::load_tokenizer().expect("tokenizer");
    let t5 = loader::load_t5_encoder(&dir).expect("t5");
    let (embeds, mask) =
        mlx_gen_chroma::encode_prompt(&tokenizer, &t5, "a red fox in a snowy forest, photograph")
            .expect("encode");
    mlx_rs::transforms::eval([&embeds, &mask]).expect("eval conditioning");
    let conditioning_peak = get_peak_memory() as f64 / GIB;
    let text_len = embeds.shape()[1];
    drop(t5);
    drop(tokenizer);
    clear_cache();

    // Phase B: the DiT + one forward at the production shape (one CFG branch).
    let h2 = (edge / 16) as i32;
    let si = h2 * h2;
    let mut img = vec![0f32; (si * 3) as usize];
    for i in 0..h2 {
        for j in 0..h2 {
            let o = ((i * h2 + j) * 3) as usize;
            img[o + 1] = i as f32;
            img[o + 2] = j as f32;
        }
    }
    let img_ids = Array::from_slice(&img, &[si, 3]);
    let txt_ids = Array::from_slice(&vec![0f32; (text_len * 3) as usize], &[text_len, 3]);
    let full_mask =
        mlx_rs::ops::concatenate_axis(&[&mask, &Array::ones::<f32>(&[1, si]).unwrap()], 1)
            .expect("full mask");
    let latents = mlx_gen_flux::create_noise(1234, edge, edge).expect("latents");
    reset_peak_memory();
    let transformer =
        loader::load_transformer(&dir, ChromaTransformerConfig::default()).expect("transformer");
    let (double, single) = transformer.resident_block_counts();
    let velocity = transformer
        .forward(
            &latents,
            &embeds,
            &Array::from_slice(&[1.0f32], &[1]),
            &img_ids,
            &txt_ids,
            Some(&full_mask),
        )
        .expect("dit forward");
    mlx_rs::transforms::eval([&velocity]).expect("eval denoise");
    let dit_peak = get_peak_memory() as f64 / GIB;
    drop(transformer);
    drop(embeds);
    drop(velocity);
    clear_cache();

    // Phase C: the VAE + one decode at the production shape, untiled AND at the published rung-2
    // geometry. Both are needed: rung 4's composition engages rung 2, so the phase that binds a
    // rung-4 request is the one that binds once the decode is already bounded.
    let unpacked = mlx_gen_flux::unpack_latents(&latents, edge, edge).expect("unpack");
    reset_peak_memory();
    let vae = loader::load_vae(&dir).expect("vae");
    let decoded = mlx_gen::LatentDecoder::decode(&vae, &unpacked).expect("decode");
    mlx_rs::transforms::eval([&decoded]).expect("eval decode");
    let decode_peak = get_peak_memory() as f64 / GIB;
    drop(decoded);
    clear_cache();
    reset_peak_memory();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(
        ms::DECODE_TILE_EDGE as i32,
        ms::DECODE_OVERLAP as i32,
    );
    let tiled = vae
        .decode_tiled(&unpacked, &tiling, None)
        .expect("tiled decode");
    mlx_rs::transforms::eval([&tiled]).expect("eval tiled decode");
    let tiled_decode_peak = get_peak_memory() as f64 / GIB;
    drop(vae);
    drop(tiled);
    clear_cache();

    println!(
        "[sc-15520 scope {} {edge}²] conditioning (T5-XXL) {conditioning_peak:.4} GiB | denoise \
         (DiT {double}+{single} blocks) {dit_peak:.4} GiB | decode (VAE) {decode_peak:.4} GiB \
         untiled, {tiled_decode_peak:.4} GiB at edge {}/overlap {}",
        probe_tier(),
        ms::DECODE_TILE_EDGE,
        ms::DECODE_OVERLAP,
    );
    assert_eq!(
        (double, single),
        (19, 38),
        "the Chroma trunk is 19 double + 38 single blocks; a different shape invalidates every \
         window plan in this file"
    );

    let binding = |decode: f64| {
        if conditioning_peak >= dit_peak.max(decode) {
            "CONDITIONING"
        } else if dit_peak >= decode {
            "DENOISE"
        } else {
            "DECODE"
        }
    };
    let unbounded_binding = binding(decode_peak);
    // The composition rung 4 actually runs in: it engages rung 2 by cost order, so its decode is
    // already tiled and the question "which phase does a window have to address" is answered here.
    let rung_four_binding = binding(tiled_decode_peak);
    println!(
        "[sc-15520 scope {}] binding phase: {unbounded_binding} unbounded; {rung_four_binding} in \
         the rung-4 composition (decode already tiled)",
        probe_tier()
    );

    // Reported, not asserted: which phase binds is the INPUT to the scope decision, and the scope
    // decision itself is measured end to end by
    // `the_window_component_scopes_are_measured_not_inherited` — a request-peak comparison, which is
    // the only thing the epic counts as a saving. Asserting a phase ordering here as well would pin
    // an intermediate quantity twice and reject a legitimate re-tiering (sc-16462 packing the T5)
    // for the wrong reason.
    assert!(
        conditioning_peak > 0.0 && dit_peak > 0.0 && decode_peak > 0.0,
        "every phase probe must do real work — MLX is lazy, so a probe that only loads reads zero"
    );
    assert!(
        tiled_decode_peak < decode_peak,
        "the tiled decode must bound the decode phase ({tiled_decode_peak:.4} vs {decode_peak:.4} \
         GiB); the rung-2 verdict rests on the mechanism working and the QUALITY failing, not on \
         the mechanism failing"
    );
}

/// **Which component scope actually moves the request peak** — measured, never inherited.
///
/// The epic's rule is that a strategy which bounds a phase but does not move the REQUEST peak is not
/// a saving. Rung 4 can be scoped at the DiT, at the text encoder, or at both, and which of those
/// moves the request peak is a property of where the binding phase is — which is a property of the
/// tier's packing and of which cheaper rungs are engaged, not of the architecture. So it is measured
/// here, against rung 4's own control, and the published default must be the winner.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_window_component_scopes_are_measured_not_inherited() {
    let tier = probe_tier();
    let edge = probe_size();
    let dir = require_tier(representative_env(), &tier);
    warm_up(representative(), &dir);

    let control = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(Some(rung4_control()), edge, steps()),
    );
    println!(
        "[sc-15520 scope {tier} {edge}²] rung-4 control (no window) {:.4} GiB, {:.0} ms/step",
        control.peak_gib,
        ms_per_step(&control, steps())
    );

    let mut rows: Vec<(TransformerComponent, f64, f64)> = Vec::new();
    for component in ms::TRANSFORMER_WINDOW_COMPONENTS {
        let row = measure(
            representative(),
            &dir,
            LoadShape::DeferredMaterialization,
            &request(
                Some(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, *component)),
                edge,
                steps(),
            ),
        );
        println!(
            "[sc-15520 scope {tier} {edge}² {component:?}] {:.4} GiB ({:+.2}% vs control)  \
             {:.0} ms/step  max Δ {}",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib,
            ms_per_step(&row, steps()),
            max_delta(&control.pixels, &row.pixels),
        );
        // Every scope is a residency change, not an arithmetic one, so every scope must be
        // byte-identical to the control regardless of whether it moves the peak.
        assert_eq!(
            max_delta(&control.pixels, &row.pixels),
            0,
            "scope {component:?} changed the image — a re-materialized block is not reproducing its \
             resident twin"
        );
        rows.push((*component, row.peak_gib, ms_per_step(&row, steps())));
    }

    if probing() {
        println!("[sc-15520 scope {tier} {edge}²] probe mode: rows reported, not asserted");
        return;
    }
    let best = rows
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a non-empty scope domain");
    // The published DEFAULT must be a scope that actually moves the request peak, by the same 3%
    // margin every implemented rung has to clear.
    let published = rows
        .iter()
        .find(|(component, _, _)| *component == ms::TRANSFORMER_WINDOW_COMPONENT)
        .expect("the default scope must be published");
    assert!(
        published.1 < control.peak_gib * 0.97,
        "the published default scope {:?} peaked at {:.4} GiB against a {:.4} GiB control — a \
         window scoped at a non-binding phase bounds something real and moves the request peak by \
         nothing, which is not a saving. The measured rows are {rows:?}",
        ms::TRANSFORMER_WINDOW_COMPONENT,
        published.1,
        control.peak_gib
    );
    // And it must be the best of the published scopes, within the instrument's own resolution.
    assert!(
        published.1 <= best.1 * 1.01,
        "scope {:?} peaks at {:.4} GiB but the published default {:?} peaks at {:.4} — the default \
         must be the scope that bounds the request peak hardest",
        best.0,
        best.1,
        ms::TRANSFORMER_WINDOW_COMPONENT,
        published.1
    );
}

// ── Rung 2 ───────────────────────────────────────────────────────────────────────────────────────

/// **The mechanism-level tile sweep** that decides which edges the ladder may publish.
///
/// Isolated from the request envelope on purpose: the *mechanism* column measures the tiled decode
/// against the **exact untiled decode of the same latent**, which is the only way to see the
/// deviation a tile actually introduces. The latent is the **production** one — what the denoiser
/// hands the decode phase — rather than a re-encoded finished image, whose statistics have already
/// been through the VAE round trip and are systematically friendlier (SDXL's rung-2 verdict turned
/// on exactly that difference: 38/255 on a re-encoded latent, 84/255 on the production one).
///
/// Driving `Vae::decode_tiled` directly also reaches geometries the production resolver refuses,
/// which is how a rejected candidate gets a NUMBER instead of an omission.
///
/// The grid remains evidence for a human, and SC-19753 pins every cell against the production bar
/// plus byte-identical output across overlaps at each edge.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn decode_tile_mechanism_sweep_on_the_production_latent() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let list = |var: &str, default: Vec<u32>| -> Vec<u32> {
        match std::env::var(var) {
            Ok(v) => v
                .split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect(),
            Err(_) => default,
        }
    };
    let edges = list("CHROMA_SWEEP_EDGES", ms::DECODE_TILE_EDGES_SWEPT.to_vec());
    let overlaps = list("CHROMA_SWEEP_OVERLAPS", ms::DECODE_OVERLAPS_SWEPT.to_vec());
    let size = list("CHROMA_SWEEP_SIZE", vec![1024])[0];

    // The production latent: the denoiser's own output at the production schedule, not a re-encode.
    let latent = production_latent(&dir, size);
    let vae = mlx_gen_chroma::loader::load_vae(&dir).expect("load vae");
    let unpacked = mlx_gen_flux::unpack_latents(&latent, size, size).expect("unpack");
    clear_cache();
    reset_peak_memory();
    let reference = mlx_gen::LatentDecoder::decode(&vae, &unpacked).expect("untiled decode");
    reference.eval().expect("eval reference");
    let untiled_peak = get_peak_memory() as f64 / GIB;
    let ref_px =
        mlx_gen::image::decoded_to_image(&reference.as_dtype(mlx_rs::Dtype::Float32).unwrap())
            .expect("reference image")
            .pixels;
    drop(reference);
    clear_cache();
    println!(
        "[sc-15520 rung2 mechanism {DEFAULT_TIER} {size}²] untiled decode peak {untiled_peak:.4} GiB"
    );

    let mut best: Option<(u32, u32, u32, f64)> = None;
    let mut overlap_outputs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    println!("| edge | overlap | tiles | isolated peak (GiB) | vs untiled | max Δ | mean Δ |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for overlap in &overlaps {
        for edge in &edges {
            let cfg = mlx_gen::tiling::TilingConfig::spatial_only(*edge as i32, *overlap as i32);
            // The SHARED geometry constant and the axes `Vae::decode_tiled` itself slices on
            // (`sh[2], sh[3]`). A hand-rolled `VaeTiling` literal and `shape()[1]` — the 16-channel
            // axis — under-reported the grid as 2 where the decode ran 2x2 (review of PR #496).
            let plan = cfg.plan(
                mlx_gen::tiling::VaeTiling::QWEN_IMAGE,
                1,
                unpacked.shape()[2],
                unpacked.shape()[3],
            );
            let tiles = plan.h.len() * plan.w.len();
            clear_cache();
            reset_peak_memory();
            let tiled = vae
                .decode_tiled(&unpacked, &cfg, None)
                .expect("tiled decode");
            tiled.eval().expect("eval tiled");
            let peak = get_peak_memory() as f64 / GIB;
            let px =
                mlx_gen::image::decoded_to_image(&tiled.as_dtype(mlx_rs::Dtype::Float32).unwrap())
                    .expect("tiled image")
                    .pixels;
            let drift = max_delta(&ref_px, &px);
            println!(
                "| {edge} | {overlap} | {tiles} | {peak:.4} | {:+.2}% | {drift} | {:.4} |",
                100.0 * (peak - untiled_peak) / untiled_peak,
                mean_delta(&ref_px, &px),
            );
            assert!(
                drift <= SIBLING_DRIFT_BAR,
                "layer-wise decode exceeds the {SIBLING_DRIFT_BAR}/255 bar at edge {edge} overlap {overlap}: {drift}"
            );
            if let Some(reference) = overlap_outputs.get(edge) {
                assert_eq!(
                    &px, reference,
                    "overlap is policy identity only; edge {edge} changed output at overlap {overlap}"
                );
            } else {
                overlap_outputs.insert(*edge, px.clone());
            }
            if best.is_none_or(|(_, _, d, _)| drift < d) {
                best = Some((*edge, *overlap, drift, peak));
            }
            drop(tiled);
            clear_cache();
        }
    }
    let (best_edge, best_overlap, best_drift, best_peak) = best.expect("a non-empty sweep");
    println!(
        "[sc-15520 rung2 mechanism {DEFAULT_TIER} {size}²] best cell: edge {best_edge} overlap \
         {best_overlap} — max Δ {best_drift}/255, isolated peak {best_peak:.4} GiB against \
         {untiled_peak:.4} untiled"
    );

    if edges != ms::DECODE_TILE_EDGES_SWEPT || overlaps != ms::DECODE_OVERLAPS_SWEPT || size != 1024
    {
        return;
    }
    // The route-blind constant remains fail-closed; exact sealed policy rows adopt the measured
    // coordinates. The arithmetic itself must now clear the shared production quality bar.
    assert!(
        best_drift <= SIBLING_DRIFT_BAR,
        "the best swept geometry exceeds {SIBLING_DRIFT_BAR}/255: edge {best_edge} overlap \
         {best_overlap} produced {best_drift}"
    );
    assert!(
        best_peak < untiled_peak * 0.97,
        "the tiled decode no longer bounds the decode phase ({best_peak:.4} vs {untiled_peak:.4} \
         GiB) — the mechanism this rung rests on has stopped working"
    );
}

/// A **production** final latent: the denoiser's own output at the production schedule and sampler.
///
/// Not a re-encoded finished image. Under the retired whole-tail decoder the two instruments exposed
/// different per-crop GroupNorm drift; the layer-wise regression continues to use the production
/// latent because that is what exact-coordinate admission promises to users.
#[track_caller]
fn production_latent(dir: &std::path::Path, size: u32) -> mlx_rs::Array {
    let seed = 1234;
    let spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Resident);
    let model = mlx_gen_chroma::load_chroma(mlx_gen_chroma::ChromaVariant::Base, &spec)
        .expect("load chroma resident");
    let latents = mlx_gen_flux::create_noise(seed, size, size).expect("noise");
    let out = model
        .denoise_with_sampler_name(
            "a red fox in a snowy forest, photograph",
            "blurry, lowres",
            size,
            size,
            steps(),
            4.0,
            latents,
            None,
            &mlx_gen::CancelFlag::new(),
            &mut |_| {},
        )
        .expect("denoise the production latent");
    mlx_rs::transforms::eval([&out]).expect("eval latent");
    drop(model);
    clear_cache();
    out
}

/// Resample one sealed Chroma coordinate across production latents.
///
/// The pre-SC-19753 whole-tail decoder produced `[53, 82, 74, 63, 28]` here because each crop used
/// different GroupNorm statistics. That historical `UNRESOLVED` class is now the defect signature:
/// full-activation normalization must keep every seed within [`SIBLING_DRIFT_BAR`].
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn layerwise_decode_quality_is_resampled_across_seeds() {
    const SEEDS: [u64; 5] = [1234, 7, 99, 20260805, 424242];
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let size = 1024_u32;
    let (edge, overlap) = (MEASURED_POLICY_EDGE, MEASURED_POLICY_OVERLAP);
    let vae = mlx_gen_chroma::loader::load_vae(&dir).expect("load vae");
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(edge as i32, overlap as i32);

    // ONE resident load for the whole sample. Reloading the 14 GiB bundle per seed put the machine
    // into allocator thrash and turned a 20-minute measurement into an hour — and the latent is a
    // pure function of the seed, so a shared load produces the identical five.
    let latents: Vec<(u64, mlx_rs::Array)> = {
        let spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
            .with_offload_policy(OffloadPolicy::Resident);
        let model = mlx_gen_chroma::load_chroma(mlx_gen_chroma::ChromaVariant::Base, &spec)
            .expect("load chroma resident");
        let latents = SEEDS
            .iter()
            .map(|seed| {
                let noise = mlx_gen_flux::create_noise(*seed, size, size).expect("noise");
                let out = model
                    .denoise_with_sampler_name(
                        "a red fox in a snowy forest, photograph",
                        "blurry, lowres",
                        size,
                        size,
                        steps(),
                        4.0,
                        noise,
                        None,
                        &mlx_gen::CancelFlag::new(),
                        &mut |_| {},
                    )
                    .expect("denoise");
                mlx_rs::transforms::eval([&out]).expect("eval latent");
                (*seed, out)
            })
            .collect();
        drop(model);
        clear_cache();
        latents
    };

    let mut drifts = Vec::new();
    for (seed, latent) in latents {
        let unpacked = mlx_gen_flux::unpack_latents(&latent, size, size).expect("unpack");
        let reference = mlx_gen::LatentDecoder::decode(&vae, &unpacked).expect("untiled decode");
        reference.eval().expect("eval reference");
        let ref_px =
            mlx_gen::image::decoded_to_image(&reference.as_dtype(mlx_rs::Dtype::Float32).unwrap())
                .expect("reference image")
                .pixels;
        drop(reference);
        clear_cache();
        let tiled = vae
            .decode_tiled(&unpacked, &cfg, None)
            .expect("tiled decode");
        tiled.eval().expect("eval tiled");
        let px = mlx_gen::image::decoded_to_image(&tiled.as_dtype(mlx_rs::Dtype::Float32).unwrap())
            .expect("tiled image")
            .pixels;
        let drift = max_delta(&ref_px, &px);
        println!(
            "[sc-15520 rung2 resample {DEFAULT_TIER} {size}² edge {edge} overlap {overlap}] seed \
             {seed}: max Δ {drift}/255, mean Δ {:.4}",
            mean_delta(&ref_px, &px)
        );
        drifts.push(drift);
        drop(tiled);
        clear_cache();
    }

    let lo = *drifts.iter().min().expect("a non-empty sample");
    let hi = *drifts.iter().max().expect("a non-empty sample");
    println!(
        "[sc-15520 rung2 resample {DEFAULT_TIER} {size}²] max Δ over {} seeds: {drifts:?} — range \
         {lo}..{hi}/255 against a {SIBLING_DRIFT_BAR}/255 bar",
        SEEDS.len()
    );

    let verdict = if hi <= SIBLING_DRIFT_BAR {
        "ADMISSIBLE"
    } else if lo > SIBLING_DRIFT_BAR {
        "FAILS"
    } else {
        "UNRESOLVED"
    };
    println!("[sc-15520 rung2 resample] verdict class: {verdict}");
    assert_eq!(
        verdict, "ADMISSIBLE",
        "layer-wise decode must clear {SIBLING_DRIFT_BAR}/255 on every production latent: \
         {drifts:?} (range {lo}..{hi})"
    );
}

/// **Rungs 2 and 3 are refused by the PRODUCTION path** wherever they are declared `Missing`, on
/// real weights.
///
/// The withheld mechanisms are still in the crate — `Vae::decode_tiled` and `sdpa_budgeted_bhsd` are
/// both reachable — which is exactly why this exists: the only thing between a measured-and-rejected
/// mechanism and a production render is the refusal, and a refusal that lives in a doc comment is
/// not one.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_withheld_rungs_are_refused_by_the_production_path() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let registry = mlx_gen_chroma::provider_registry().expect("provider registry");
    let model = registry
        .load(
            representative(),
            &spec(&dir, LoadShape::EagerMaterialization),
        )
        .expect("load chroma");

    let mut checked = 0_usize;
    let mut admitted = Vec::new();
    for edge in ms::DECODE_TILE_EDGES_SWEPT {
        for overlap in ms::DECODE_OVERLAPS_SWEPT {
            let published = ms::DECODE_SUPPORT
                && ms::DECODE_TILE_EDGES.contains(edge)
                && *overlap == ms::DECODE_OVERLAP;
            let outcome =
                model.generate(&request(Some(rung2(*edge, *overlap)), 1024, 1), &mut |_| {});
            println!(
                "[sc-15520 rung2 domain] DECODE-REQUEST edge={edge} overlap={overlap} \
                 published={published} admitted={}",
                outcome.is_ok()
            );
            match (published, outcome) {
                (true, Ok(_)) => {}
                (true, Err(error)) => {
                    panic!("published geometry {edge}/{overlap} was refused: {error}")
                }
                (false, Ok(_)) => admitted.push(format!("{edge}/{overlap}")),
                (false, Err(error)) => {
                    let error = error.to_string();
                    assert!(
                        error.contains("bounded decode is not selectable")
                            || error.contains("decode tile edge")
                            || error.contains("decode overlap"),
                        "edge {edge} overlap {overlap} was refused for the wrong reason: {error}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        admitted.is_empty(),
        "geometries {admitted:?} are outside the published domain and were ADMITTED — a clamped or \
         ignored tile executes a strategy the selector did not choose"
    );
    assert!(
        checked > 0,
        "the swept set contains no rejected geometry, so this test asserts nothing"
    );

    if !ms::ATTENTION_SUPPORT {
        let error = match model.generate(&request(Some(rung3()), 1024, 1), &mut |_| {}) {
            Ok(_) => panic!("bounded attention must not render on this provider"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("bounded attention is not selectable")
                || error.contains("bounded decode is not selectable"),
            "{error}"
        );
    }

    // The control: the published composition renders. Without it, a generator that refused every
    // request would satisfy every assertion above.
    model
        .generate(&request(Some(staged()), 1024, 1), &mut |_| {})
        .expect("the published rung-1 composition must render");
}

// ── Rung 3 ───────────────────────────────────────────────────────────────────────────────────────

/// **What bounded attention does on this family — measured at the DiT seam.**
///
/// The production path refuses a bounded-attention request (`ATTENTION_SUPPORT` is `false`), so the
/// mechanism is driven directly through `ChromaTransformer::forward_with_attention_plan`. That is
/// deliberate and it is the only shape that keeps the `Missing` verdict falsifiable: a rung whose
/// evidence can no longer be re-taken once it is withheld is a comment, not a measurement.
///
/// The request-level numbers behind the verdict, taken before the refusal landed, were
/// **28.0779 GiB unbounded → 28.0776 GiB chunked at 1024², 1 step: −0.001%.** This test re-takes the
/// same comparison at the seam, where it is isolated from the residency schedule, and asserts it in
/// **both** directions with a margin: below means chunking has started bounding this family and the
/// `Missing` declaration is stale; above means it has started adding transients. A one-sided floor
/// could not tell a real seam change from no change at all.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn attention_chunking_is_measured_at_the_dit_seam() {
    use mlx_gen::attention::{AttentionBudget, AttentionPlan};
    use mlx_gen_chroma::{loader, ChromaTransformerConfig};
    use mlx_rs::Array;

    let dir = require_tier(representative_env(), DEFAULT_TIER);
    const EDGE: u32 = 1024;

    // The production conditioning, so the joint sequence is the real `[text, image]` length and the
    // `[B,1,S,S]` MMDiT mask is the real one — the tensor the bounded kernel has to narrow per chunk.
    let tokenizer = loader::load_tokenizer().expect("tokenizer");
    let t5 = loader::load_t5_encoder(&dir).expect("t5");
    let (embeds, mask) =
        mlx_gen_chroma::encode_prompt(&tokenizer, &t5, "a red fox in a snowy forest, photograph")
            .expect("encode");
    mlx_rs::transforms::eval([&embeds, &mask]).expect("eval conditioning");
    let text_len = embeds.shape()[1];
    drop(t5);
    drop(tokenizer);
    clear_cache();

    let h2 = (EDGE / 16) as i32;
    let si = h2 * h2;
    let mut ids = vec![0f32; (si * 3) as usize];
    for i in 0..h2 {
        for j in 0..h2 {
            let o = ((i * h2 + j) * 3) as usize;
            ids[o + 1] = i as f32;
            ids[o + 2] = j as f32;
        }
    }
    let img_ids = Array::from_slice(&ids, &[si, 3]);
    let txt_ids = Array::from_slice(&vec![0f32; (text_len * 3) as usize], &[text_len, 3]);
    let full_mask =
        mlx_rs::ops::concatenate_axis(&[&mask, &Array::ones::<f32>(&[1, si]).unwrap()], 1)
            .expect("full mask");
    let latents = mlx_gen_flux::create_noise(1234, EDGE, EDGE).expect("latents");
    let timestep = Array::from_slice(&[1.0f32], &[1]);
    let transformer =
        loader::load_transformer(&dir, ChromaTransformerConfig::default()).expect("transformer");

    let run = |plan: AttentionPlan<'_>| -> (f64, Vec<f32>) {
        clear_cache();
        reset_peak_memory();
        let out = transformer
            .forward_with_attention_plan(
                &latents,
                &embeds,
                &timestep,
                &img_ids,
                &txt_ids,
                Some(&full_mask),
                plan,
            )
            .expect("dit forward");
        out.eval().expect("eval");
        let peak = get_peak_memory() as f64 / GIB;
        let v = out
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        (peak, v)
    };

    // **The discarded warm-up row (SC-17679).** This test drives the seam directly rather than
    // through `measure`, so it does not go through `warm_up` — and the first forward in the process
    // reads its peak against a cold allocator whichever plan it executes. The sibling Kolors harness
    // published a +12.54% "seam regression" from exactly this, which is +0.00% once warmed. The
    // bounded plan warms up, because it has the larger allocation set.
    let bounded_plan = || AttentionPlan::budgeted(AttentionBudget::CONSTRAINED);
    let (warm_up_peak, _) = run(bounded_plan());

    let (unbounded_peak, unbounded) = run(AttentionPlan::UNBOUNDED);
    let (bounded_peak, bounded) = run(bounded_plan());
    let max_abs = unbounded
        .iter()
        .zip(&bounded)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let delta_pct = 100.0 * (bounded_peak - unbounded_peak) / unbounded_peak;
    println!(
        "[sc-15520 rung3 dit-seam {DEFAULT_TIER} {EDGE}²] discarded warm-up {warm_up_peak:.4} GiB; \
         unbounded {unbounded_peak:.4} GiB -> bounded {bounded_peak:.4} GiB ({delta_pct:+.2}%)  \
         max|Δv| {max_abs:.3e}"
    );

    // **Half one: the mechanism is wired and it is not a no-op.** This is the assertion that would
    // catch rung 3 shipping with the chunking deleted — a defect a sibling PR actually shipped, with
    // green CI, because every assertion it carried passed more easily with the feature off.
    assert!(
        delta_pct < -1.0,
        "bounded attention no longer bounds the DiT seam at all ({delta_pct:+.2}%, {bounded_peak:.4} \
         vs {unbounded_peak:.4} GiB). The plan is threaded through both block stacks and both \
         attention sites; if it has stopped engaging, this file's rung-3 evidence is measuring \
         nothing"
    );
    drop(transformer);
    clear_cache();

    // **Half two, and the one that decides the rung: bounding a phase is not moving the request
    // peak.** The saving above is real and it is confined to the denoise phase, which on this family
    // is not the phase that binds — so a caller pays wall clock for a request peak that does not
    // move. Asserted as arithmetic over two measured quantities rather than as a remembered figure:
    // the seam saving must be smaller than the headroom between the denoise phase and the request
    // peak. If the DiT ever becomes the binding phase, or the saving ever outgrows that headroom,
    // this reddens and ATTENTION_SUPPORT is re-decided.
    warm_up(representative(), &dir);
    let request_peak = measure(
        representative(),
        &dir,
        LoadShape::EagerMaterialization,
        &request(Some(staged()), EDGE, 1),
    )
    .peak_gib;
    let saving = unbounded_peak - bounded_peak;
    let headroom = request_peak - unbounded_peak;
    println!(
        "[sc-15520 rung3 request {DEFAULT_TIER} {EDGE}²] staged request peak {request_peak:.4} GiB \
         against a {unbounded_peak:.4} GiB denoise seam — headroom {headroom:.4} GiB, seam saving \
         {saving:.4} GiB"
    );
    if ms::ATTENTION_SUPPORT {
        assert!(
            saving > headroom,
            "bounded attention is declared Implemented but its {saving:.4} GiB seam saving is \
             inside the {headroom:.4} GiB headroom between the denoise phase and the request peak, \
             so it cannot move the request peak"
        );
    } else {
        assert!(
            saving < headroom,
            "bounded attention's {saving:.4} GiB seam saving now exceeds the {headroom:.4} GiB \
             headroom between the denoise phase and the request peak — the denoise phase may have \
             become peak-bearing, so ATTENTION_SUPPORT must be re-decided rather than left Missing \
             on a superseded measurement"
        );
    }
}

// ── Rung 4 ───────────────────────────────────────────────────────────────────────────────────────

/// **The cadence sweep**: does the window bound the request peak, is the domain a real frontier, and
/// is the streamed render byte-identical to its resident twin?
///
/// Rows execute in [`probe_order`], and the aggregate assertions sort by cadence first, so the
/// conclusions cannot depend on execution order — the SC-17679 control.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn transformer_window_sweep_and_streamed_output_identity() {
    let tier = probe_tier();
    let edge = probe_size();
    let dir = require_tier(representative_env(), &tier);
    warm_up(representative(), &dir);

    // The control is rung 4's OWN composition minus the window. Comparing against a rung-1 row
    // instead would confound the window with rung 2's tiled decode, which rung 4 engages by cost
    // order and which is an arithmetic change — a byte-identity assertion against that control could
    // never hold, and a peak comparison against it would credit the window with rung 2's saving.
    let control = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(Some(rung4_control()), edge, steps()),
    );
    println!(
        "[sc-15520 rung4 {tier} {edge}²] rung-4 control (its own composition, no window) {:.4} \
         GiB, {:.0} ms/step",
        control.peak_gib,
        ms_per_step(&control, steps())
    );

    let mut rows: Vec<(u32, f64, f64)> = Vec::new();
    for window in probe_order() {
        let row = measure(
            representative(),
            &dir,
            LoadShape::DeferredMaterialization,
            &request(Some(full_ladder(window)), edge, steps()),
        );
        println!(
            "[sc-15520 rung4 {tier} {edge}² cadence {window}] {:.4} GiB ({:+.2}% vs control)  \
             {:.0} ms/step ({:.1}x)  max Δ {}",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib,
            ms_per_step(&row, steps()),
            ms_per_step(&row, steps()) / ms_per_step(&control, steps()),
            max_delta(&control.pixels, &row.pixels),
        );
        // Byte-identity is asserted in BOTH modes: a streamed block is re-materialized through the
        // same constructor over the same file, so only residency differs.
        assert_eq!(
            max_delta(&control.pixels, &row.pixels),
            0,
            "cadence {window} changed the image — the re-materialized blocks are NOT reproducing \
             the resident ones (tier replay? adapter replay?)"
        );
        rows.push((window, row.peak_gib, ms_per_step(&row, steps())));
    }

    // Sorted by cadence, so `rows[0]`/`rows.last()` mean tightest/widest regardless of the order the
    // rows were EXECUTED in (`probe_order`).
    rows.sort_unstable_by_key(|(window, _, _)| *window);
    assert!(rows.len() >= 2, "a cadence sweep needs at least two rows");

    if probing() {
        println!("[sc-15520 rung4 {tier} {edge}²] probe mode: rows reported, not asserted");
        return;
    }
    for (window, peak, _) in &rows {
        // **The assertion that makes this test able to fail.** Byte-identity alone passes with the
        // windowing deleted — a resident forward is trivially identical to itself. The rung must
        // also MOVE the request peak.
        assert!(
            *peak < control.peak_gib * 0.97,
            "cadence {window} did not bound the request peak: {peak:.4} vs control {:.4} GiB — the \
             block stream is not actually replacing the resident stack",
            control.peak_gib
        );
    }
    let (tightest, tight_peak, tight_ms) = rows[0];
    let (widest, _wide_peak, wide_ms) = *rows.last().expect("a non-empty sweep");
    let spread = rows
        .iter()
        .map(|(_, peak, _)| 100.0 * (peak - tight_peak).abs() / tight_peak)
        .fold(0f64, f64::max);
    println!(
        "[sc-15520 rung4 {tier} {edge}²] cadence {tightest}..{widest}: peak spread {spread:.2}%, \
         {tight_ms:.0} -> {wide_ms:.0} ms/step"
    );
    // **The peak is cadence-independent, and the latency ordering is WITHDRAWN as evidence.**
    //
    // The peaks are flat to the fourth decimal in every execution order, and the instrument's
    // resolution at this cell is 0.00% over eight identical requests
    // (`identical_requests_reproduce_once_the_allocator_has_settled`), so 1% is a tolerance the
    // instrument can support. That is the claim this domain rests on: every published cadence
    // bounds the request peak to the same value, so a caller may pick any of them.
    //
    // The ms/step column is **printed and not asserted**, and that is a measured decision rather
    // than caution. Under [`probe_order`] the wall clock follows the row's POSITION, not its
    // cadence (SC-17679): re-running the same four cadences in a different execution order moves the
    // times with the slots. Cadence 10 has measured both the slowest row and the fastest one across
    // orders, and an independent re-run reproduced neither ordering — which is the point. No table
    // of those numbers is reproduced here, because a printed table reads as a finding and this one
    // is not: the only publishable statement is that the ordering is unresolvable on this
    // instrument.
    //
    // So `TRANSFORMER_WINDOW_SIZE`'s default rests on being the tightest weight bound — what the
    // rung exists for — and not on a latency argument.
    assert!(
        spread < 1.0,
        "the published cadences no longer bound the request peak to the same value ({spread:.2}% \
         spread over {rows:?}) against an instrument resolution of 0.00% at this cell. The domain \
         is published as equal-peak alternatives, so that has to keep holding — or it must be \
         narrowed to the cadence that was actually measured"
    );
    let _ = (tightest, widest, tight_ms, wide_ms);
}

/// **The published cadence domain is enforced in BOTH directions on the production path.**
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_published_window_domain_is_enforced_and_reachable() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let registry = mlx_gen_chroma::provider_registry().expect("provider registry");
    let model = registry
        .load(
            representative(),
            &spec(&dir, LoadShape::DeferredMaterialization),
        )
        .expect("load chroma");

    // Reachable: every published cadence renders.
    for window in ms::TRANSFORMER_WINDOW_SIZES {
        model
            .generate(&request(Some(full_ladder(*window)), 512, 1), &mut |_| {})
            .unwrap_or_else(|error| panic!("published cadence {window} must render, got: {error}"));
    }

    // Collected rather than asserted in-loop, so a regression reports every cadence it affects.
    const OUT_OF_DOMAIN: [u32; 8] = [0, 3, 6, 7, 11, 20, 39, 70];
    let mut silently_admitted = Vec::new();
    let mut refused_elsewhere = Vec::new();
    let mut refused_by_the_window_validator = 0_usize;
    for bad in OUT_OF_DOMAIN {
        assert!(
            !ms::TRANSFORMER_WINDOW_SIZES.contains(&bad),
            "the negative list must stay disjoint from the published domain"
        );
        let outcome = model.generate(&request(Some(full_ladder(bad)), 512, 1), &mut |_| {});
        println!(
            "[sc-15520 rung4 domain] WINDOW-REQUEST size={bad} admitted={} refused={}",
            outcome.is_ok(),
            outcome.is_err()
        );
        match outcome {
            Ok(_) => silently_admitted.push(bad),
            Err(error) if error.to_string().contains("transformer window") => {
                refused_by_the_window_validator += 1;
            }
            Err(error) => refused_elsewhere.push(format!("{bad}: {error}")),
        }
    }
    assert!(
        silently_admitted.is_empty(),
        "cadences {silently_admitted:?} are outside {:?} and were ADMITTED — a clamped or ignored \
         window executes a strategy the selector did not choose",
        ms::TRANSFORMER_WINDOW_SIZES
    );
    // Refused-by-something-else is a weaker guarantee: `BlockPlan::new` rejects a zero window on its
    // own, so a provider whose own validator silently stopped working would still look fine at that
    // end of the range while admitting the interior gaps.
    assert!(
        refused_elsewhere.is_empty(),
        "these cadences were refused, but NOT by memory_strategy::transformer_window — the \
         provider's own domain gate is not the thing rejecting them: {refused_elsewhere:?}"
    );
    assert_eq!(
        refused_by_the_window_validator,
        OUT_OF_DOMAIN.len(),
        "every out-of-domain cadence must be refused by the provider's window validator"
    );

    // The component scope is a published domain too, and the unimplemented scopes are refused.
    for component in [
        TransformerComponent::TextEncoder,
        TransformerComponent::Both,
    ] {
        if ms::TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
            continue;
        }
        let outcome = model.generate(
            &request(
                Some(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, component)),
                512,
                1,
            ),
            &mut |_| {},
        );
        let error = match outcome {
            Ok(_) => panic!("unpublished scope {component:?} was admitted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("component"), "scope {component:?}: {error}");
    }
}

/// **Rung 4's preconditions fail closed on real weights**, with a positive control.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn rung_four_preconditions_fail_closed_on_real_weights() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let registry = mlx_gen_chroma::provider_registry().expect("provider registry");

    // 1. Eager materialization: no reopenable stream.
    let eager = registry
        .load(
            representative(),
            &spec(&dir, LoadShape::EagerMaterialization),
        )
        .expect("load chroma eagerly");
    let error = match eager.generate(
        &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
        &mut |_| {},
    ) {
        Ok(_) => panic!("an eager load must not stream its blocks"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("cannot stream its blocks"), "{error}");
    drop(eager);
    clear_cache();

    let deferred = registry
        .load(
            representative(),
            &spec(&dir, LoadShape::DeferredMaterialization),
        )
        .expect("load chroma deferred");

    // 2. Rung 4 without rung 1 in the same request.
    let unstaged = GenerationMemory {
        stage_residency: false,
        ..full_ladder(ms::TRANSFORMER_WINDOW_SIZE)
    };
    let error = match deferred.generate(&request(Some(unstaged), 512, 1), &mut |_| {}) {
        Ok(_) => panic!("rung 4 must require rung 1 in the same request"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("staged residency"), "{error}");

    // 3. A dense source the loader would re-quantize per window: refused at load, not at render.
    let mut requantizing = spec(&dir, LoadShape::DeferredMaterialization);
    requantizing.quantize = Some(Quant::Q8);
    assert!(
        !ms::structurally_streamable(&requantizing),
        "a load-time quantization must not arm rung 4 — a window would convert host formats every \
         re-materialization"
    );
    let error = match registry
        .load(representative(), &requantizing)
        .expect("a re-quantizing load still builds")
        .generate(
            &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
            &mut |_| {},
        ) {
        Ok(_) => panic!("a re-quantizing load must not stream its blocks"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("cannot stream its blocks"), "{error}");

    // The positive control: without it a generator that refused everything would satisfy the above.
    deferred
        .generate(
            &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
            &mut |_| {},
        )
        .expect("the published rung-4 composition must render");
}

/// **The rung-4 saving is bounded by the block weight set it actually holds**, cross-checked against
/// the snapshot's own safetensors `data_offsets` totals.
///
/// This is the arithmetic that caught a misattributed mechanism in SDXL: a saving larger than the
/// whole windowable weight set is measuring something other than block residency.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_rung_four_saving_is_inside_the_block_weight_set() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let (block_bytes, trunk_bytes) = transformer_weight_arithmetic(&dir);
    println!(
        "[sc-15520 rung4 arithmetic {DEFAULT_TIER}] windowable block weights {:.4} GiB; resident \
         trunk (embedders + Approximator + proj_out) {:.4} GiB",
        block_bytes / GIB,
        trunk_bytes / GIB
    );
    assert!(
        block_bytes > trunk_bytes,
        "the windowable stacks must dominate the resident trunk, else rung 4 has little to bound"
    );

    warm_up(representative(), &dir);
    let control = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(Some(rung4_control()), 1024, steps()),
    );
    let windowed = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(
            Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)),
            1024,
            steps(),
        ),
    );
    let saving = (control.peak_gib - windowed.peak_gib) * GIB;
    println!(
        "[sc-15520 rung4 arithmetic {DEFAULT_TIER} 1024²] staged {:.4} -> windowed {:.4} GiB, \
         saving {:.4} GiB against a {:.4} GiB block weight set",
        control.peak_gib,
        windowed.peak_gib,
        saving / GIB,
        block_bytes / GIB
    );
    assert!(
        saving < block_bytes * 1.15,
        "the measured saving ({:.4} GiB) exceeds the whole windowable block weight set ({:.4} GiB) \
         by more than measurement slack — rung 4 cannot bound more than it holds, so the row is \
         measuring something else",
        saving / GIB,
        block_bytes / GIB
    );
}

/// Sum the `transformer/` safetensors payload split into the windowable block stacks and the trunk
/// rung 4 leaves resident. Header arithmetic only — no tensor is materialized.
fn transformer_weight_arithmetic(dir: &std::path::Path) -> (f64, f64) {
    let component = dir.join("transformer");
    let mut block = 0f64;
    let mut trunk = 0f64;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&component)
        .unwrap_or_else(|error| panic!("read {}: {error}", component.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no safetensors under {}",
        component.display()
    );
    for path in files {
        // Read ONLY the 8-byte length prefix and the JSON header. Slurping the whole file pulled a
        // ~5.06 GiB shard onto the heap to parse a few KiB of JSON, on an epic where this harness is
        // itself the OOM risk (review of PR #496).
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&path).expect("open safetensors");
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .expect("safetensors length prefix");
        let header_len = u64::from_le_bytes(prefix) as usize;
        assert!(
            header_len > 0 && header_len < 64 * 1024 * 1024,
            "{}: implausible safetensors header length {header_len}",
            path.display()
        );
        file.seek(SeekFrom::Start(8)).expect("seek past the prefix");
        let mut header_bytes = vec![0_u8; header_len];
        file.read_exact(&mut header_bytes)
            .expect("safetensors header");
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).expect("safetensors header");
        for (name, entry) in header.as_object().expect("header object") {
            if name == "__metadata__" {
                continue;
            }
            let offsets = entry["data_offsets"].as_array().expect("data_offsets");
            let size = (offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap()) as f64;
            if name.starts_with("transformer_blocks.")
                || name.starts_with("single_transformer_blocks.")
            {
                block += size;
            } else {
                trunk += size;
            }
        }
    }
    (block, trunk)
}

// ── Per-entry and per-tier evidence ──────────────────────────────────────────────────────────────

/// **Every cached entry/tier loads and publishes a coherent ladder**, and each supplies its OWN
/// rung-1 saving — sharing this provider's code is explicitly not what makes an entry Verified.
///
/// Absent entries/tiers are reported by name and counted; the test fails if NOTHING was measured, so
/// it can never pass having checked an empty set.
#[test]
#[ignore = "needs the real Chroma1 snapshots (see the module docs for the env vars)"]
fn every_cached_entry_and_tier_publishes_its_own_evidence() {
    let mut measured = 0_usize;
    let mut absent = Vec::new();
    for (entry, var) in ENTRIES {
        for tier in TIERS {
            let Some(dir) = tier_dir(var, tier) else {
                absent.push(format!("{entry}/{tier}"));
                continue;
            };
            let deferred = spec(&dir, LoadShape::DeferredMaterialization);
            let contract =
                mlx_gen_chroma::memory_strategy::contract_for(entry, &deferred).expect("contract");
            assert!(
                contract.conformance_errors().is_empty(),
                "{entry}/{tier}: {:?}",
                contract.conformance_errors()
            );
            let streams = ms::structurally_streamable(&deferred);
            let rung_four = &contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4 capability")
                .support;
            assert_eq!(
                streams,
                matches!(rung_four, MemoryStrategySupport::Implemented),
                "{entry}/{tier}: `structurally_streamable` and the published rung-4 support \
                 disagree — the contract must never advertise a rung this load cannot execute"
            );
            assert!(
                contract.asset_facts.base_bytes > 0,
                "{entry}/{tier}: asset facts must be filesystem-backed"
            );

            warm_up(entry, &dir);
            let resident = measure(
                entry,
                &dir,
                LoadShape::EagerMaterialization,
                &request(Some(resident_memory()), 1024, 1),
            );
            let staged_row = measure(
                entry,
                &dir,
                LoadShape::EagerMaterialization,
                &request(Some(staged()), 1024, 1),
            );
            println!(
                "[sc-15520 entries] {entry}/{tier}: resident {:.4} -> staged {:.4} GiB ({:+.2}%), \
                 rung4 {rung_four:?}, base asset bytes {:.4} GiB",
                resident.peak_gib,
                staged_row.peak_gib,
                100.0 * (staged_row.peak_gib - resident.peak_gib) / resident.peak_gib,
                contract.asset_facts.base_bytes as f64 / GIB,
            );
            assert_eq!(
                max_delta(&resident.pixels, &staged_row.pixels),
                0,
                "{entry}/{tier}: rung 1 must be byte-identical"
            );
            assert!(
                staged_row.peak_gib < resident.peak_gib * 0.97,
                "{entry}/{tier}: rung 1 must bound the request peak by a real margin ({:.4} vs \
                 {:.4} GiB)",
                staged_row.peak_gib,
                resident.peak_gib
            );
            measured += 1;
        }
    }
    println!("[sc-15520 entries] measured {measured} cell(s); absent: {absent:?}");
    assert!(
        measured > 0,
        "SKIPPED-BY-ABSENCE: no entry/tier was cached under {:?} — this test's claim is per-entry \
         and cannot be made from an empty set",
        ENTRIES.iter().map(|(_, var)| *var).collect::<Vec<_>>()
    );
}

/// **The full ladder renders under a memory cap** that the unwindowed composition does not fit.
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_full_ladder_renders_under_a_memory_cap() {
    let dir = require_tier(representative_env(), DEFAULT_TIER);
    warm_up(representative(), &dir);
    let staged_row = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(Some(rung4_control()), 1024, steps()),
    );
    let windowed = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(
            Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)),
            1024,
            steps(),
        ),
    );
    // The cap sits between the two peaks, so it is a cap the windowed composition needs.
    let cap = ((staged_row.peak_gib + windowed.peak_gib) / 2.0).ceil() as u64;
    println!(
        "[sc-15520 cap] staged {:.4} GiB, windowed {:.4} GiB -> cap {cap} GB",
        staged_row.peak_gib, windowed.peak_gib
    );
    std::env::set_var(MEMORY_CAP_ENV, cap.to_string());
    let capped = measure(
        representative(),
        &dir,
        LoadShape::DeferredMaterialization,
        &request(
            Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)),
            1024,
            steps(),
        ),
    );
    std::env::remove_var(MEMORY_CAP_ENV);
    assert_eq!(
        capped.pixels.len(),
        1024 * 1024 * 3,
        "a capped full-ladder render must still produce the requested 1024² RGB image"
    );
    let first = capped.pixels[0];
    assert!(
        capped.pixels.iter().any(|p| *p != first),
        "the capped render produced a uniform image — the window streamed blocks that decoded to \
         nothing rather than reproducing the resident stack"
    );
    assert_eq!(
        max_delta(&windowed.pixels, &capped.pixels),
        0,
        "the capped render must be byte-identical to the uncapped windowed one"
    );
}

/// **The SC-15449 calibration fault fires at each phase boundary, and a fresh request follows.**
#[test]
#[ignore = "needs a real Chroma1 snapshot (see the module docs for the env vars)"]
fn the_calibration_fault_fires_at_every_phase_and_a_fresh_request_recovers() {
    use mlx_gen::gen_core::MemoryPhase;

    let dir = require_tier(representative_env(), DEFAULT_TIER);
    let registry = mlx_gen_chroma::provider_registry().expect("provider registry");
    let model = registry
        .load(
            representative(),
            &spec(&dir, LoadShape::DeferredMaterialization),
        )
        .expect("load chroma");

    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let mut memory = staged();
        memory.authorize_calibration_fault(phase);
        let error = match model.generate(&request(Some(memory), 512, 1), &mut |_| {}) {
            Ok(_) => panic!("the authorized {phase:?} fault must fire"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(&format!("{phase:?}")), "{error}");
        // And a fresh request recovers, so the fault leaves no retained provider state.
        model
            .generate(&request(Some(staged()), 512, 1), &mut |_| {})
            .unwrap_or_else(|error| panic!("a fresh request after the {phase:?} fault: {error}"));
    }

    // A phase without its authorization never reaches the provider's injection seam: the SHARED
    // request floor refuses the half-set pair outright, which is a stronger guarantee than the
    // provider ignoring it would be.
    let unauthorized = GenerationMemory {
        calibration_error_phase: Some(MemoryPhase::Decode),
        ..staged()
    };
    let error = match model.generate(&request(Some(unauthorized), 512, 1), &mut |_| {}) {
        Ok(_) => panic!("an unauthorized calibration phase must be refused by the shared floor"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("authorization"), "{error}");
}
