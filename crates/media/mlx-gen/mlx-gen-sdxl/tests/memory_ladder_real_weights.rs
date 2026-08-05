//! Real-weight conformance and **measurement** for the SDXL shared memory ladder (SC-15525,
//! closing SC-16355), on Apple/Metal.
//!
//! Every number published in `crate::memory_strategy` comes from this file. Nothing here is
//! inherited from another family or another backend: the epic's standing rule is that a rung's
//! presence, magnitude, mechanism and candidate set are per family per backend, and SDXL is the
//! first **U-Net** on this ladder, so it has no architectural predecessor to inherit from even in
//! principle.
//!
//! ## Measurement discipline
//!
//! * **MLX's own accounting**, never timer-sampled RSS. `mlx_rs::memory::get_peak_memory` reports
//!   ACTIVE bytes; a sampled RSS measures how fast the machine happened to run.
//! * **A fresh generator per measured row.** This is not hygiene, it is the difference between a
//!   number and an artifact: a reused heavy bundle lets the first row materialize the lazily-loaded
//!   trunk and every later row then reads a peak that includes work it did not do. The
//!   `#[track_caller]` helper below is the only way a row is produced.
//! * **`reset_peak_memory` after the load**, so a row measures the *request*, not the load.
//! * Rejected candidates are recorded **with their numbers**, and the rejection is re-asserted
//!   against the production path — not left in a doc comment.
//!
//! ## Weights
//!
//! One env var per catalog entry, each pointing at that entry's snapshot **root** (the tier is a
//! subdirectory: `bf16` / `q4` / `q8`). Nothing self-fetches or derives a cache location
//! (epic 13657). A test whose entry/tier is absent **skips loudly by name** rather than passing
//! silently.
//!
//! | env var | entry |
//! |---|---|
//! | `SDXL_LADDER_SDXL` | `sdxl` |
//! | `SDXL_LADDER_REALVISXL` | `realvisxl` — the representative entry (the only one with all three tiers) |
//! | `SDXL_LADDER_REALVISXL_LIGHTNING` | `realvisxl_lightning` |
//! | `SDXL_LADDER_ILLUSTRIOUS_XL_V1` | `illustrious_xl_v1` |
//! | `SDXL_LADDER_ILLUSTRIOUS_XL_V2` | `illustrious_xl_v2` |

#![allow(clippy::items_after_test_module)]

use std::path::PathBuf;

use mlx_gen::gen_core::{GenerationMemory, GenerationOutput, GenerationRequest, Progress};
use mlx_gen::memory::MEMORY_CAP_ENV;
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen_sdxl::memory_strategy as ms;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The five catalog entries this provider serves, with the env var carrying each one's snapshot
/// root and the tiers cached for it at authoring time.
const ENTRIES: &[(&str, &str)] = &[
    ("sdxl", "SDXL_LADDER_SDXL"),
    ("realvisxl", "SDXL_LADDER_REALVISXL"),
    ("realvisxl_lightning", "SDXL_LADDER_REALVISXL_LIGHTNING"),
    ("illustrious_xl_v1", "SDXL_LADDER_ILLUSTRIOUS_XL_V1"),
    ("illustrious_xl_v2", "SDXL_LADDER_ILLUSTRIOUS_XL_V2"),
];

/// The representative entry: the one catalog entry with **all three** advertised tiers cached, so a
/// per-tier sweep is possible without a download. Every sibling entry still owes its own evidence —
/// sharing this provider's code is explicitly not what makes an entry Verified.
const REPRESENTATIVE: &str = "SDXL_LADDER_REALVISXL";

fn entry_root(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from)
}

/// Resolve one entry's tier directory, or `None` when it is not cached.
fn tier_dir(var: &str, tier: &str) -> Option<PathBuf> {
    let root = entry_root(var)?;
    let dir = root.join(tier);
    dir.is_dir().then_some(dir)
}

fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// A load spec for one entry/tier at the shape the ladder needs.
fn spec(dir: &std::path::Path, tier: &str, shape: LoadShape) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(shape);
    spec.quantize = quant_for(tier);
    spec
}

fn request(memory: Option<GenerationMemory>, edge: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, lowres".into()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(steps),
        guidance: Some(7.0),
        seed: Some(1234),
        memory,
        ..Default::default()
    }
}

/// One measured row: the request's ACTIVE-bytes peak, its pixels, and its wall clock.
///
/// `wall` exists because SC-16355 carried re-materialization latency as an explicit hazard — "70
/// blocks across 11 re-opens could be a severe regression on exactly the small Macs this rung exists
/// for" — and a peak-only row cannot answer it. A memory rung that halves the peak and triples the
/// render is not obviously a win, and the selector cannot weigh a cost nobody measured.
///
/// It is a wall clock, so it is the softest number in this file: it moves with thermal state and with
/// whatever else the machine is doing. It is therefore *reported* and bounded loosely, never asserted
/// to a tight figure.
struct Row {
    peak_gib: f64,
    pixels: Vec<u8>,
    wall: std::time::Duration,
}

/// Render one row on a **fresh** generator and return its request peak.
///
/// The freshness is the whole contract of this helper. SDXL's U-Net weights are lazy MLX handles
/// until something evaluates them, so a generator reused across rows carries the previous row's
/// materialization into this row's peak — which is exactly how a rung-4 sweep can report a saving
/// that is really just "the first row paid for the stack". Every row in this file goes through here.
#[track_caller]
fn measure(dir: &std::path::Path, tier: &str, shape: LoadShape, req: &GenerationRequest) -> Row {
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let model = registry
        .load("sdxl", &spec(dir, tier, shape))
        .expect("load sdxl");
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

/// Milliseconds per denoise step for a row, which is the unit rung 4's re-materialization cost is
/// actually paid in: the window re-opens the snapshot once per sub-stack per step, so the overhead
/// scales with steps rather than with the request.
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

/// The all-rungs-off baseline: an explicit Resident block, so the load-time `Sequential` policy
/// cannot leak a phase release into the "resident" row.
fn resident_memory() -> GenerationMemory {
    GenerationMemory::default()
}

fn staged() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..Default::default()
    }
}

fn staged_chunked() -> GenerationMemory {
    GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(ms::ATTENTION_CHUNK_SIZE),
        ..staged()
    }
}

fn full_ladder(window: u32) -> GenerationMemory {
    GenerationMemory {
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(ms::TRANSFORMER_WINDOW_COMPONENT),
        ..staged()
    }
}

// ── Rung 0/1 ─────────────────────────────────────────────────────────────────────────────────────

/// **Rung 1 is request-scoped and it moves the request peak.**
///
/// The same cached generator serves resident → staged, and the staged request must peak strictly
/// lower while producing a **byte-identical** image: rung 1 sheds the two CLIP towers before the
/// heavy bundle loads, which is a residency change and not an arithmetic one.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn staged_residency_bounds_the_request_peak_and_preserves_output() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let shape = LoadShape::EagerMaterialization;
    let resident = measure(
        &dir,
        "bf16",
        shape,
        &request(Some(resident_memory()), 1024, 6),
    );
    let staged_row = measure(&dir, "bf16", shape, &request(Some(staged()), 1024, 6));
    println!(
        "[sc-15525 rung1 bf16 1024²] resident {:.3} GiB -> staged {:.3} GiB",
        resident.peak_gib, staged_row.peak_gib
    );
    assert_eq!(
        max_delta(&resident.pixels, &staged_row.pixels),
        0,
        "rung 1 is a residency schedule, not an arithmetic change: the image must be byte-identical"
    );
    // **A margin, not a bare `<`.** A bare inequality here passes on floating-point noise: with the
    // request-scoped resolver stubbed out, both rows land on the same schedule and the two peaks
    // differ in the fourth decimal — which satisfies `<` about half the time. The measured saving is
    // −7.4%, so 3% is a floor a no-op implementation cannot clear in either direction.
    assert!(
        staged_row.peak_gib < resident.peak_gib * 0.97,
        "staged residency must bound the request peak by a real margin: {:.4} vs {:.4} GiB",
        staged_row.peak_gib,
        resident.peak_gib
    );
}

// ── Rung 2 ───────────────────────────────────────────────────────────────────────────────────────

/// **Rung 2 and rung 3 are refused by the production path**, on real weights.
///
/// Both were measured (see [`decode_tile_mechanism_sweep`] and
/// [`attention_chunking_is_measured_against_the_rung_two_top`]) and both are declared `Missing`.
/// Their mechanisms are still in the crate — which is exactly why this test exists: the only thing
/// between `Autoencoder::decode_tiled` and a production render is the refusal.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_rejected_rungs_are_refused_by_the_production_path() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let model = registry
        .load("sdxl", &spec(&dir, "bf16", LoadShape::EagerMaterialization))
        .expect("load sdxl");

    for edge in ms::DECODE_TILE_EDGES_SWEPT {
        let memory = GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(*edge),
            decode_overlap: Some(ms::DECODE_OVERLAP),
            ..staged()
        };
        let err = match model.generate(&request(Some(memory), 1024, 2), &mut |_| {}) {
            Ok(_) => panic!("a bounded decode must not render on this provider (edge {edge})"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("bounded decode is not selectable"),
            "edge {edge}: {err}"
        );
    }

    let err = match model.generate(&request(Some(staged_chunked()), 1024, 2), &mut |_| {}) {
        Ok(_) => panic!("bounded attention must not render on this provider"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("bounded attention is not selectable"),
        "{err}"
    );

    // The control: the published composition renders. Without it, a generator that refused every
    // request would satisfy every assertion above.
    model
        .generate(&request(Some(staged()), 1024, 2), &mut |_| {})
        .expect("the published rung-1 composition must render");
}

/// **The mechanism-level tile sweep** that decides which edges the ladder may publish.
///
/// Isolated from the request envelope on purpose, and that split is the same one Anima's rung-2
/// evidence uses: the *mechanism* column measures the decode against the **exact untiled decode of
/// the same latent**, which is the only way to see the deviation a tile actually introduces, while
/// the request column (above) measures what a caller pays. Driving it through
/// [`Autoencoder::decode_tiled`] also reaches geometries the production resolver refuses, which is
/// how a rejected candidate gets a *number* instead of an omission.
///
/// The latent is a real one: a full render at 1024², re-encoded through the same VAE, so the
/// GroupNorm statistics the tail tiles are the statistics of a real image rather than of noise.
///
/// **This is an exploration harness, and it asserts one thing only.** The 32-cell grid it prints is
/// evidence for a human reading the rung-2 write-up, not a gate: pinning 32 drift values would fail
/// on any MLX kernel change without telling anyone anything useful. What it *does* pin is the single
/// cell the verdict cites — the best geometry at the default output size — because that number
/// appears in `memory_strategy::DECODE_SUPPORT` and in `refuse_decode`'s user-facing message. The
/// range argument that actually decides the rung is asserted separately, in
/// `no_single_decode_tile_edge_clears_the_bar_across_the_advertised_output_range`.
///
/// Set `SDXL_SWEEP_EDGES` / `SDXL_SWEEP_OVERLAPS` (comma-separated) to explore; the defaults are the
/// published ladder and its rejection set.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn decode_tile_mechanism_sweep() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let list = |var: &str, default: Vec<u32>| -> Vec<u32> {
        match std::env::var(var) {
            Ok(v) => v
                .split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect(),
            Err(_) => default,
        }
    };
    let edges = list("SDXL_SWEEP_EDGES", ms::DECODE_TILE_EDGES_SWEPT.to_vec());
    let overlaps = list("SDXL_SWEEP_OVERLAPS", ms::DECODE_OVERLAPS_SWEPT.to_vec());
    // The render edge the sweep decodes at. Parameterized because whether a tile edge DIVIDES the
    // output evenly turns out to matter more than its size: a ragged remainder tile normalizes its
    // GroupNorms over a sliver of the image.
    let size = list("SDXL_SWEEP_SIZE", vec![1024])[0];

    // A real latent: render once, then re-encode through the same VAE.
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let model = registry
        .load("sdxl", &spec(&dir, "bf16", LoadShape::EagerMaterialization))
        .expect("load sdxl");
    let out = model
        .generate(&request(Some(staged()), size, 6), &mut |_| {})
        .expect("render one image for the sweep latent");
    let image = match out {
        GenerationOutput::Images(mut images) => images.swap_remove(0),
        other => panic!("expected images, got {other:?}"),
    };
    drop(model);
    clear_cache();

    let vae = mlx_gen_sdxl::load_vae(&dir).expect("load vae");
    let nhwc = mlx_gen_sdxl::preprocess_init_image(&image, size, size).expect("preprocess");
    let latent = vae.encode_mean(&nhwc).expect("encode");
    // The untiled decode is both the drift reference AND the peak this mechanism is measured
    // against. Recording only the tiled peaks left the "what does tiling buy" half of the evidence
    // with no denominator from this test.
    clear_cache();
    reset_peak_memory();
    let reference = vae.decode(&latent).expect("untiled decode");
    reference.eval().expect("eval reference");
    let untiled_peak = get_peak_memory() as f64 / GIB;
    let ref_px = mlx_gen_sdxl::decoded_to_image(&reference)
        .expect("reference image")
        .pixels;

    println!("[sc-15525 rung2 mechanism] latent {:?}", latent.shape());
    println!("[sc-15525 rung2 mechanism] untiled decode peak {untiled_peak:.3} GiB at {size}²");
    let mut cited: Option<u32> = None;
    println!("| edge | overlap | tiles | isolated peak (GiB) | max Δ | mean Δ |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for overlap in &overlaps {
        for edge in &edges {
            let cfg = mlx_gen::tiling::TilingConfig::spatial_only(*edge as i32, *overlap as i32);
            let plan = cfg.plan(
                mlx_gen::tiling::VaeTiling {
                    spatial_scale: 8,
                    temporal_scale: 1,
                    causal_temporal: false,
                    full_res_channels: 128,
                },
                1,
                latent.shape()[1],
                latent.shape()[2],
            );
            let tiles = plan.h.len() * plan.w.len();
            clear_cache();
            reset_peak_memory();
            let tiled = vae.decode_tiled(&latent, &cfg, None).expect("tiled decode");
            tiled.eval().expect("eval tiled");
            let peak = get_peak_memory() as f64 / GIB;
            let px = mlx_gen_sdxl::decoded_to_image(&tiled)
                .expect("tiled image")
                .pixels;
            let drift = max_delta(&ref_px, &px);
            println!(
                "| {edge} | {overlap} | {tiles} | {peak:.3} | {drift} | {:.4} |",
                mean_delta(&ref_px, &px)
            );
            if size == 1024 && *edge == ms::DECODE_TILE_EDGE && *overlap == BEST_DECODE_OVERLAP {
                cited = Some(drift);
            }
        }
    }
    // The one cell this harness gates: the number `memory_strategy::DECODE_SUPPORT` publishes and
    // `refuse_decode` puts in front of a user. Only asserted when the sweep is run at its defaults —
    // an explicitly narrowed sweep is exploring, and should not be forced to reproduce it.
    if let Some(drift) = cited {
        assert!(
            drift <= SIBLING_DRIFT_BAR,
            "the best swept geometry now drifts {drift}/255 at 1024², above the \
             {SIBLING_DRIFT_BAR}/255 sibling bar. DECODE_SUPPORT and `refuse_decode`'s message both \
             quote the old figure — re-derive them"
        );
    }
}

/// **The measurement that decides rung 2**, and the one the SC-15525 review's "publish `[896]`@192"
/// proposal turns on.
///
/// The review was right about the datum and right about the bar. At 1024² the best geometry in the
/// whole sweep — edge [`ms::DECODE_TILE_EDGE`] at overlap [`BEST_DECODE_OVERLAP`] — drifts **38/255**,
/// which clears [`SIBLING_DRIFT_BAR`], the worst drift Z-Image *admits* into its shipped ladder. On
/// that evidence alone the candidate should be published.
///
/// What the 1024²-only sweep could not see is that **the datum is not a property of the tile edge**.
/// A contract's `decode_tile_edges` is an absolute pixel domain with no geometry axis, so publishing
/// 896 publishes it across everything this provider advertises — and `descriptor()` advertises
/// `max_size: 2048`. Re-measured across that range, the same geometry walks straight through the bar:
///
/// | output | tiles | max Δ | verdict |
/// |---:|---:|---:|---|
/// | 1024² | 4 | **38** | clears 48 |
/// | 1280² | 4 | 64 | fails |
/// | 1536² | 4 | 120 | fails |
/// | 2048² | 9 | 77 | fails |
///
/// The mechanism explains the shape and predicts it: the tail's GroupNorms normalize over the spatial
/// extent each tile is handed, so what governs the drift is the tile's **fraction of the image**, not
/// its pixel size. 896 covers 87.5% of a 1024² output and 43.75% of a 2048² one. No fixed pixel edge
/// can hold a constant fraction across a 4× range, so no fixed edge is admissible across it — which
/// is a different and much stronger statement than the one this module used to make, and unlike that
/// one it is consistent with the table above.
///
/// So the rung stays `Missing`, and this test is why: it fails if 1024² ever stops clearing the bar
/// (the candidate was never real) **and** it fails if the larger outputs ever start clearing it (the
/// candidate is now publishable and the declaration is stale). Either way the verdict gets revisited
/// instead of inherited.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn no_single_decode_tile_edge_clears_the_bar_across_the_advertised_output_range() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let vae = mlx_gen_sdxl::load_vae(&dir).expect("load vae");

    let mut rows: Vec<(u32, u32, f64)> = Vec::new();
    // 2048² is included because `refuse_decode`'s user-facing message quotes its number. A published
    // figure — especially one in an error string a user reads — must come from a test, which is the
    // defect class the first review blocked this PR on. It is the slowest row here by far; that is
    // the cost of quoting it.
    for size in [1024_u32, 1280, 1536, 2048] {
        // Fresh generator per size: the latent must be a real image's, and a reused bundle would
        // carry the previous size's materialization into this one's isolated decode peak.
        let model = registry
            .load("sdxl", &spec(&dir, "bf16", LoadShape::EagerMaterialization))
            .expect("load sdxl");
        let out = model
            .generate(&request(Some(staged()), size, 6), &mut |_| {})
            .expect("render the sweep latent");
        let image = match out {
            GenerationOutput::Images(mut images) => images.swap_remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        drop(model);
        clear_cache();

        let nhwc = mlx_gen_sdxl::preprocess_init_image(&image, size, size).expect("preprocess");
        let latent = vae.encode_mean(&nhwc).expect("encode");
        let reference = vae.decode(&latent).expect("untiled decode");
        reference.eval().expect("eval reference");
        let ref_px = mlx_gen_sdxl::decoded_to_image(&reference)
            .expect("reference image")
            .pixels;

        let cfg = mlx_gen::tiling::TilingConfig::spatial_only(
            ms::DECODE_TILE_EDGE as i32,
            BEST_DECODE_OVERLAP as i32,
        );
        clear_cache();
        let tiled = vae.decode_tiled(&latent, &cfg, None).expect("tiled decode");
        tiled.eval().expect("eval tiled");
        let px = mlx_gen_sdxl::decoded_to_image(&tiled)
            .expect("tiled image")
            .pixels;
        let drift = max_delta(&ref_px, &px);
        let coverage = f64::from(ms::DECODE_TILE_EDGE) / f64::from(size);
        println!(
            "[sc-15525 rung2 range] {size}² edge {} overlap {BEST_DECODE_OVERLAP}: max Δ \
             {drift}/255 (tile covers {:.1}% of the output)",
            ms::DECODE_TILE_EDGE,
            100.0 * coverage,
        );
        rows.push((size, drift, coverage));
    }

    let at = |size: u32| rows.iter().find(|(s, _, _)| *s == size).expect("row").1;
    // Half one: the candidate is REAL at the size it was swept. Without this the test would pass on a
    // decoder that drifts everywhere, and the withholding reason would be unfalsifiable.
    assert!(
        at(1024) <= SIBLING_DRIFT_BAR,
        "edge {} overlap {BEST_DECODE_OVERLAP} no longer clears the sibling bar at 1024² ({} > \
         {SIBLING_DRIFT_BAR}) — the whole premise of the rung-2 write-up is stale",
        ms::DECODE_TILE_EDGE,
        at(1024),
    );
    // Half two: and it is NOT admissible across the advertised range, which is why it is withheld.
    for size in [1280_u32, 1536, 2048] {
        assert!(
            at(size) > SIBLING_DRIFT_BAR,
            "edge {} overlap {BEST_DECODE_OVERLAP} now clears {SIBLING_DRIFT_BAR}/255 at {size}² \
             ({}) — a fixed edge may have become admissible across the advertised range, so rung 2 \
             must be re-decided rather than left Missing on a superseded measurement",
            ms::DECODE_TILE_EDGE,
            at(size),
        );
    }
}

/// **What rung 2 would have bought at the REQUEST level, and what it costs on the latent a real
/// render actually produces.** Both are numbers the mechanism sweep cannot supply.
///
/// [`decode_tile_mechanism_sweep`] measures the *isolated* decode — the right scope for a drift
/// comparison, the wrong scope for the saving, because a selector admits against the whole request.
/// `generate` cannot supply this row either (`memory_strategy::decode_tiling` refuses every
/// bounded-decode request), so it is assembled from the same public entry points as the rung-3 row,
/// under the same staged schedule, with the tiled decode substituted for the single-pass one.
///
/// **The saving is real and it is the one this ladder claims**: 19.0032 → 15.8660 GiB, **−16.51%**,
/// at 876 → 936 ms/step (tiling costs ~7% wall clock, not a multiple). An earlier revision published
/// that figure with no test behind it; this is the test.
///
/// **And the drift is worse here than the sweep suggested — 84/255, not 38.** That is not a
/// contradiction, it is the sweep's latent being the friendlier one. The sweep decodes a latent
/// obtained by *re-encoding a finished image*, whose statistics have already been through the VAE
/// round trip; this row decodes what the denoiser actually hands the decode phase. The production
/// latent is the one a user would get, and at 1024² — the single output size where the swept
/// candidate cleared [`SIBLING_DRIFT_BAR`] — it does **not** clear it.
///
/// So this test carries the second, independent half of rung 2's `Missing` verdict:
/// `no_single_decode_tile_edge_clears_the_bar_across_the_advertised_output_range` shows the candidate
/// fails everywhere except 1024²; this shows it fails at 1024² too, once the latent is the real one.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_withheld_decode_geometry_is_priced_at_the_request_level() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let untiled = measure_end_to_end(&dir, 6, false, None);
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(
        ms::DECODE_TILE_EDGE as i32,
        BEST_DECODE_OVERLAP as i32,
    );
    let tiled = measure_end_to_end(&dir, 6, false, Some(&cfg));
    let saving = 100.0 * (tiled.peak_gib - untiled.peak_gib) / untiled.peak_gib;
    println!(
        "[sc-15525 rung2 request bf16 1024² 6 steps] untiled {:.4} GiB -> edge {} overlap \
         {BEST_DECODE_OVERLAP} {:.4} GiB ({saving:+.2}%), max Δ {}/255, {:.0} -> {:.0} ms/step",
        untiled.peak_gib,
        ms::DECODE_TILE_EDGE,
        tiled.peak_gib,
        max_delta(&untiled.pixels, &tiled.pixels),
        ms_per_step(&untiled, 6),
        ms_per_step(&tiled, 6),
    );
    // Half one: the mechanism must genuinely bound the request, by the same 3% margin an implemented
    // rung has to clear. If it ever stops doing so, the withholding note's "near miss with a real
    // prize" framing is wrong and the candidate is not a near miss at all.
    assert!(
        tiled.peak_gib < untiled.peak_gib * 0.97,
        "the tiled decode no longer bounds the request peak ({:.4} vs {:.4} GiB) — rung 2's \
         withholding note claims a real saving it would be giving up",
        tiled.peak_gib,
        untiled.peak_gib
    );
    // Half two, and the one that decides the rung: on the PRODUCTION latent the best swept geometry
    // does not clear the sibling bar even at 1024². Asserted rather than only printed, because this
    // is the load-bearing fact — if a future VAE or blend change brought it under the bar, the whole
    // `Missing` verdict would be resting on the range argument alone and would need re-deciding.
    let drift = max_delta(&untiled.pixels, &tiled.pixels);
    assert!(
        drift > SIBLING_DRIFT_BAR,
        "edge {} overlap {BEST_DECODE_OVERLAP} now drifts only {drift}/255 on the production latent \
         at 1024², which clears the {SIBLING_DRIFT_BAR}/255 sibling bar — rung 2's verdict is no \
         longer supported at this output size and must be re-decided on the range argument alone",
        ms::DECODE_TILE_EDGE,
    );
}

/// The drift bar this family is judged against, in 8-bit levels out of 255.
///
/// It is not invented here. It is the worst max-Δ that a **sibling MLX provider on the same shared
/// tiling machinery** admits into a shipped ladder: `mlx_gen_z_image`'s `DECODE_TILE_EDGES` tops out
/// at 48/255 (its 768 px tile), and its rejected set starts at 64. Z-Image's VAE is the same diffusers
/// `AutoencoderKL` with the same spatial-extent GroupNorms and the same head/tail split, so it is the
/// closest thing to a precedent that exists — using anything looser here would be inventing a bar to
/// clear, and using anything tighter would be inventing one to fail.
const SIBLING_DRIFT_BAR: u32 = 48;

/// The overlap that minimises drift at [`ms::DECODE_TILE_EDGE`] — the sweep's best cell.
const BEST_DECODE_OVERLAP: u32 = 192;

// ── Rung 3 ───────────────────────────────────────────────────────────────────────────────────────

/// One end-to-end row for a rung the production path **refuses**, measured over the whole request
/// rather than at one U-Net forward.
///
/// `generate` cannot produce this row: `memory_strategy::attention_plan` rejects a bounded-attention
/// selection on every layer, which is the point of rung 3 being `Missing`. So the request is
/// re-assembled here from the same public pipeline entry points `Sdxl::generate_impl` calls, in the
/// same order, under the same **rung-1 staged schedule** — encode with both CLIP towers alive, `eval`
/// the conditioning, drop the towers, *then* load the heavy pair and denoise + decode. That schedule
/// is what makes the peak comparable to the rung-1 row: it is bounded by
/// `max(CLIP-L + bigG, U-Net + VAE)` rather than by their sum.
///
/// Everything is rebuilt per row (`#[track_caller]`, same discipline as [`measure`]): the U-Net is a
/// lazy MLX handle until something forces it, so a bundle shared between the chunked and unchunked
/// arms would charge the first arm for the materialization and hand the second a free ride.
///
/// One deliberate difference from [`measure`]: the peak window opens **before** the loads rather
/// than after. That is not sloppiness, it is what a staged schedule *is* — the claim rung 1 makes is
/// that the encoder load and the heavy load never coexist, and a window that opened after both would
/// measure the thing the schedule exists to avoid. The reconstruction is checked by its own result:
/// it reads 19.0032 GiB, which is `generate`'s staged 19.003 to the millibyte.
#[track_caller]
fn measure_end_to_end(
    dir: &std::path::Path,
    steps: usize,
    chunked: bool,
    decode_tiling: Option<&mlx_gen::tiling::TilingConfig>,
) -> Row {
    use mlx_rs::Dtype::{Float16, Float32};

    let prompt = "a red fox in a snowy forest, photograph";
    let negative = "blurry, lowres";

    clear_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();

    // Conditioning phase: both towers alive, then released.
    let tokenizer = mlx_gen_sdxl::load_tokenizer(dir).expect("tokenizer");
    let te1 = mlx_gen_sdxl::load_text_encoder_1_dtype(dir, Float16).expect("clip-l");
    let te2 = mlx_gen_sdxl::load_text_encoder_2_dtype(dir, Float16).expect("bigg");
    let tokens = tokenizer
        .tokenize_batch(prompt, Some(negative))
        .expect("tokenize");
    let (conditioning, pooled) =
        mlx_gen_sdxl::encode_conditioning(&te1, &te2, &tokens).expect("encode");
    // Force the encode before dropping the towers — an unevaluated output keeps them referenced
    // through the lazy graph, so the drop would free nothing.
    mlx_rs::transforms::eval([&conditioning, &pooled]).expect("eval conditioning");
    drop((te1, te2, tokenizer));
    clear_cache();

    // Denoise + decode phase.
    let unet = mlx_gen_sdxl::load_unet_dtype(dir, Float16).expect("unet");
    let vae = mlx_gen_sdxl::load_vae(dir).expect("vae");
    let euler = mlx_gen_sdxl::EulerSampler::new_with_dtype(
        &mlx_gen_sdxl::config::DiffusionConfig::sdxl_base(),
        true,
        Float32,
    )
    .expect("sampler");
    let latents = mlx_gen_sdxl::seeded_prior(&euler, 1234, 1024, 1024).expect("prior");
    let sampler = mlx_gen_sdxl::sampler::AncestralEuler::new(&euler, steps, euler.max_time())
        .expect("ancestral");
    let plan = if chunked {
        mlx_gen_sdxl::SdxlForwardPlan::with_attention(mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::CONSTRAINED,
        ))
    } else {
        mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED
    };
    let denoiser = mlx_gen_sdxl::pipeline::Denoiser::with_plan(&unet, &sampler, plan);
    let time_ids = mlx_gen_sdxl::text_time_ids(pooled.shape()[0]);
    let cancel = mlx_gen::gen_core::CancelFlag::default();
    let out = mlx_gen_sdxl::denoise(
        &denoiser,
        latents,
        &conditioning,
        &pooled,
        &time_ids,
        7.0,
        &cancel,
        &mut |_: Progress| {},
    )
    .expect("denoise");
    let image = mlx_gen_sdxl::pipeline::decode_image_tiled(&vae, &out, None, decode_tiling, None)
        .expect("decode");

    let peak = get_peak_memory();
    let wall = started.elapsed();
    drop((unet, vae));
    clear_cache();
    Row {
        peak_gib: peak as f64 / GIB,
        pixels: image.pixels,
        wall,
    }
}

/// **The re-assembly is bound to the real `generate`, because two withholding verdicts rest on it.**
///
/// [`measure_end_to_end`] rebuilds the staged request from the public pipeline entry points, and both
/// rung 2's and rung 3's `Missing` verdicts are measured through it — because the production path
/// refuses both selections, so `generate` cannot produce either row. Its doc claimed the
/// reconstruction reproduces `generate`'s staged peak "to the millibyte" and **nothing bound them**.
/// A change to `Sdxl::generate_impl` — a dtype, an added component, a reordered phase — would leave
/// both verdicts resting on a harness that no longer models the thing it is standing in for, with
/// nothing red.
///
/// This is that binding. It compares the reconstruction's unchunked, untiled peak against a real
/// `generate` row under the same staged selection at the same geometry and step count.
///
/// The margin is 1%, not a bare equality: the two paths are the same phases in the same order, and
/// they agree to **0.001%** today (19.0030 through `generate`, 19.0032 through the re-assembly), but
/// MLX peak accounting is exact only for a fixed allocation sequence and the harness does not
/// reproduce `generate`'s progress/cancel plumbing. 1% is a thousand times the observed disagreement
/// and far tighter than any structural divergence: pointing the re-assembly at 768² while `generate`
/// renders 1024² reddens it at 47.8%.
///
/// **What it binds is the PEAK, and that is not the same as total fidelity.** A divergence that does
/// not move the peak passes — enabling attention chunking on one side only was tried as a mutation
/// and stayed green, because rung 3 moves this family's request peak 0.00%, which is itself one of
/// the two reasons `memory_strategy::ATTENTION_SUPPORT` is `Missing`. That is the honest scope: this
/// test is what makes the rung-2 and rung-3 *peak* numbers attributable to `generate`; it is not a
/// general equivalence proof between the two paths.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_end_to_end_reassembly_reproduces_the_real_generate_peak() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    const STEPS: u32 = 6;
    let production = measure(
        &dir,
        "bf16",
        LoadShape::EagerMaterialization,
        &request(Some(staged()), 1024, STEPS),
    );
    let reassembled = measure_end_to_end(&dir, STEPS as usize, false, None);
    let drift = 100.0 * (reassembled.peak_gib - production.peak_gib).abs() / production.peak_gib;
    println!(
        "[sc-15525 harness fidelity bf16 1024² {STEPS} steps] generate {:.4} GiB vs re-assembly \
         {:.4} GiB ({drift:.3}% apart)",
        production.peak_gib, reassembled.peak_gib
    );
    assert!(
        drift < 1.0,
        "the rung-2/rung-3 harness no longer models `generate`'s staged request ({:.4} vs {:.4} \
         GiB, {drift:.2}% apart). Both withholding verdicts are measured through that re-assembly, \
         so they are only as good as this agreement — re-derive the harness against \
         `Sdxl::generate_impl` before trusting either number again",
        reassembled.peak_gib,
        production.peak_gib
    );
}

/// **The rung-3 REQUEST-level measurement** — the first of the two independent reasons rung 3 is
/// declared `Missing`, and the one [`attention_chunking_is_measured_at_the_unet_seam`] cannot supply.
///
/// The seam test answers "does chunking bound one U-Net forward?". This one answers the question the
/// contract actually turns on: **does it move the number a selector admits against, and does the
/// image survive?** Those are different measurements, and publishing the second while only running
/// the first is what the SC-15525 review caught.
///
/// Both step counts are load-bearing and neither is redundant:
///
/// * **one step** is the control that separates kernel rounding from a wiring bug — the ancestral
///   schedule cannot amplify anything across a single step, so whatever Δ appears there is the raw
///   per-forward divergence;
/// * **six steps** is the production schedule, and shows what that per-forward divergence becomes
///   once a chaos-sensitive sampler has fed it back into itself.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn attention_chunking_is_measured_against_the_rung_two_top() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let mut rows = Vec::new();
    for steps in [1_usize, 6] {
        let plain = measure_end_to_end(&dir, steps, false, None);
        let chunked = measure_end_to_end(&dir, steps, true, None);
        let delta_pct = 100.0 * (chunked.peak_gib - plain.peak_gib) / plain.peak_gib;
        println!(
            "[sc-15525 rung3 end-to-end bf16 1024² {steps} step(s)] unchunked {:.4} GiB -> chunked \
             {:.4} GiB ({delta_pct:+.2}%)  max Δ {}/255  mean Δ {:.2}",
            plain.peak_gib,
            chunked.peak_gib,
            max_delta(&plain.pixels, &chunked.pixels),
            mean_delta(&plain.pixels, &chunked.pixels),
        );
        rows.push((steps, plain, chunked, delta_pct));
    }

    // Claim 1: the rung does not pay for itself at the request level. Asserted with the SAME 3%
    // margin the implemented rungs must CLEAR, in the opposite direction — so a future change that
    // makes chunking actually bound a request reddens here and forces the `Missing` declaration to be
    // revisited, rather than letting it calcify.
    for (steps, plain, chunked, _) in &rows {
        assert!(
            chunked.peak_gib > plain.peak_gib * 0.97,
            "bounded attention now bounds the REQUEST peak at {steps} step(s) ({:.4} vs {:.4} GiB) \
             — re-open memory_strategy::ATTENTION_SUPPORT",
            chunked.peak_gib,
            plain.peak_gib
        );
    }
    // Claim 2: it is not output-preserving on this path. The one-step row carries the claim, because
    // it is the one the sampler cannot have amplified.
    let (_, plain, chunked, _) = &rows[0];
    assert!(
        max_delta(&plain.pixels, &chunked.pixels) > 0,
        "chunked and unchunked agreed exactly at one step — if MLX now dispatches the same kernel at \
         every query-block size, the output-preservation half of ATTENTION_SUPPORT no longer holds \
         and the rung must be re-measured, not left Missing on a stale reason"
    );
}

/// **The rung-3 mechanism measurement**, which is why rung 3 is declared `Missing`.
///
/// Driven at the U-Net level rather than through `generate`, for the same reason the rung-2 sweep
/// is: the production resolver *refuses* a bounded-attention request
/// (`memory_strategy::attention_plan`), so the only way to measure a rung this family does not ship
/// is the mechanism seam. [`UNet2DConditionModel::forward_planned`] takes the plan raw.
///
/// Two facts are recorded, either of which alone is sufficient to withhold the rung:
///
/// 1. the bounded forward's ACTIVE peak against the unbounded one — MLX's fused SDPA never
///    materializes the score tensor, so there may be nothing to bound;
/// 2. whether the output moves — chunking is mathematically identity, but MLX dispatches a
///    different kernel per query-block size and SDXL runs fp16.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn attention_chunking_is_measured_at_the_unet_seam() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let unet = mlx_gen_sdxl::load_unet_dtype(&dir, mlx_rs::Dtype::Float16).expect("load unet");
    let key = mlx_rs::random::key(7).unwrap();
    let f16 = |a: mlx_rs::Array| a.as_dtype(mlx_rs::Dtype::Float16).unwrap();
    // The production CFG batch at 1024²: latent [2, 128, 128, 4], dual-CLIP context [2, 77, 2048].
    let x = f16(mlx_rs::random::normal::<f32>(&[2, 128, 128, 4], None, None, Some(&key)).unwrap());
    let ctx = f16(mlx_rs::random::normal::<f32>(&[2, 77, 2048], None, None, Some(&key)).unwrap());
    let pooled = f16(mlx_rs::random::normal::<f32>(&[2, 1280], None, None, Some(&key)).unwrap());
    let time_ids = mlx_gen_sdxl::text_time_ids(2);

    let run = |plan: mlx_gen_sdxl::SdxlForwardPlan<'_>| -> (f64, Vec<f32>) {
        clear_cache();
        reset_peak_memory();
        let out = unet
            .forward_planned(&x, 500.0, &ctx, &pooled, &time_ids, None, None, plan)
            .expect("unet forward");
        out.eval().expect("eval");
        let peak = get_peak_memory() as f64 / GIB;
        let v = out
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        (peak, v)
    };

    let (unbounded_peak, unbounded) = run(mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED);
    let (bounded_peak, bounded) = run(mlx_gen_sdxl::SdxlForwardPlan::with_attention(
        mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::CONSTRAINED,
        ),
    ));
    let max_abs = unbounded
        .iter()
        .zip(&bounded)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "[sc-15525 rung3 unet-seam bf16 1024² CFG] unbounded {unbounded_peak:.4} GiB -> bounded \
         {bounded_peak:.4} GiB ({:+.2}%)  max|Δeps| {max_abs:.3e}",
        100.0 * (bounded_peak - unbounded_peak) / unbounded_peak
    );
    // The published verdict: the rung does not move this forward's peak. Asserted with the same
    // margin the implemented rungs use, so a future change that DOES make it pay reddens here and
    // forces the declaration to be revisited rather than silently staying `Missing`.
    assert!(
        bounded_peak > unbounded_peak * 0.97,
        "bounded attention now bounds something on this family ({bounded_peak:.4} vs \
         {unbounded_peak:.4} GiB) — re-open memory_strategy::ATTENTION_SUPPORT"
    );
}

// ── Rung 4 ───────────────────────────────────────────────────────────────────────────────────────

/// **The rung-4 cadence sweep** — the whole published domain, peak *and* wall clock — plus the
/// bit-identity proof that the eleven streamed sub-stacks reproduce the resident ones exactly.
///
/// The first revision of this test swept a one-value domain, which made it a sweep in name only. It
/// now drives every cadence in [`ms::TRANSFORMER_WINDOW_SIZES`] and asserts the three facts the
/// domain's publication rests on:
///
/// 1. **every cadence bounds the peak** (each row < control by a 3% margin, against a measured
///    −12.31%);
/// 2. **every cadence bounds it to the SAME value** — this is the finding that made the domain worth
///    publishing, because it means the wider cadences give up no memory at all. Asserted at 1%, which
///    is far outside the observed spread: the peak rows reproduce **to the millibyte** (15.516 GiB at
///    all four cadences, across four separate runs), so unlike the latency numbers this one can be
///    pinned tightly;
/// 3. **widening the cadence buys time back monotonically**, which is what makes the domain a
///    frontier rather than four spellings of one choice.
///
/// Latency assertions are deliberately loose. Four runs of the window-1 row measured 3654 / 3674 /
/// 3698 / 3790 ms/step — a ~4% spread — so the margins below are set well outside that, and the
/// failure they exist to catch is a re-open path that starts *thrashing*, not one that drifts.
///
/// `SDXL_WINDOW_PROBE_SIZE` re-runs the sweep at another output edge. The 768² numbers in
/// [`ms::TRANSFORMER_WINDOW_SIZES`]' table come from it, and they are what establish that the flat
/// peak is a property of this family rather than of 1024².
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn transformer_window_sweep_and_streamed_output_identity() {
    // `SDXL_WINDOW_PROBE_TIER` / `SDXL_WINDOW_PROBE_SIZE` re-run the sweep at another tier or output
    // edge. The bf16/q4 and 512² rows in `ms::TRANSFORMER_WINDOW_SIZES`' tables come from them, and
    // they are what establish that the flat peak is a property of the PHASE SEPARATION rather than of
    // one tier at one geometry.
    // Probe mode: any off-default tier/size. The peak-flatness and wall-clock-frontier assertions
    // below are claims about the DEFAULT configuration — flatness is measured false at 512² q8, and a
    // wall clock is meaningless on a loaded machine — so in probe mode they are reported instead of
    // asserted. What stays asserted in BOTH modes is what is true of every configuration: each
    // cadence bounds the request peak below the control, and every row is byte-identical to it.
    // The 12x latency ceiling is NOT in that set — an earlier draft of this comment claimed it was,
    // and probing 512² falsified the claim at 12.6x. The re-open cost is area-independent while the
    // control's forward is not, so that ratio is geometry-bound by construction; see the ceiling
    // itself for the numbers.
    let probing = std::env::var("SDXL_WINDOW_PROBE_TIER").is_ok()
        || std::env::var("SDXL_WINDOW_PROBE_SIZE").is_ok();
    let tier = std::env::var("SDXL_WINDOW_PROBE_TIER").unwrap_or_else(|_| "q8".to_owned());
    let tier: &str = &tier;
    let Some(dir) = tier_dir(REPRESENTATIVE, tier) else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing {tier}/");
    };
    let deferred = LoadShape::DeferredMaterialization;
    // The attribution control: the same composition WITHOUT rung 4, on a deferred load, so the only
    // difference between the two rows is the window itself.
    const STEPS: u32 = 6;
    let edge: u32 = std::env::var("SDXL_WINDOW_PROBE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let control = measure(&dir, tier, deferred, &request(Some(staged()), edge, STEPS));
    println!(
        "[sc-15525 rung4 {tier} {edge}²] staged, no window (attribution control) {:.3} GiB, \
         {:.0} ms/step",
        control.peak_gib,
        ms_per_step(&control, STEPS)
    );
    let mut rows: Vec<(u32, f64, f64)> = Vec::new();
    for window in ms::TRANSFORMER_WINDOW_SIZES {
        let row = measure(
            &dir,
            tier,
            deferred,
            &request(Some(full_ladder(*window)), edge, STEPS),
        );
        rows.push((*window, row.peak_gib, ms_per_step(&row, STEPS)));
        // SC-16355 named re-materialization latency as an explicit hazard, so it is reported beside
        // the peak rather than left for a user to discover. The window re-opens the U-Net snapshot
        // once per `Transformer2D` per step — eleven re-opens covering 70 blocks — and what that
        // costs is a fact about this rung, not an implementation detail.
        println!(
            "[sc-15525 rung4 {tier} {edge}² window {window}] request peak {:.3} GiB ({:+.2}% vs control), \
             {:.0} ms/step ({:+.1}% vs control)",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib,
            ms_per_step(&row, STEPS),
            100.0 * (ms_per_step(&row, STEPS) - ms_per_step(&control, STEPS))
                / ms_per_step(&control, STEPS),
        );
        assert_eq!(
            max_delta(&control.pixels, &row.pixels),
            0,
            "a streamed block must be byte-identical to its resident twin: window {window} changed \
             the image, which means the re-materialized blocks are NOT reproducing the resident \
             ones (tier replay? adapter replay? IP-Adapter pairs?)"
        );
        // **The assertion that makes this test able to fail.** Byte-identity alone passes with the
        // windowing deleted — a resident forward is trivially identical to itself. The rung must
        // also MOVE the request peak, and the measured margin is −12.3%, so 3% is a floor a
        // no-op implementation cannot clear.
        assert!(
            row.peak_gib < control.peak_gib * 0.97,
            "window {window} did not bound the request peak: {:.3} vs control {:.3} GiB — the \
             block stream is not actually replacing the resident stack",
            row.peak_gib,
            control.peak_gib
        );
        // A LOOSE latency ceiling, and deliberately loose. The tightest cadence measures ~4.9x at
        // 1024² and ~7.4x at 768²; a wall clock moves with thermal state and with whatever else the
        // machine is doing, so asserting anything near the measurement would be asserting the
        // weather. 12x still leaves a substantial worsening detectable, which is the failure worth
        // catching — a re-open path that starts thrashing rather than one that drifts 15%.
        //
        // **The ceiling is a claim about the DEFAULT geometry, so probe mode reports it instead.**
        // Re-materialization cost is weight I/O: it is fixed per re-open and does not scale with
        // output area, while the control's per-step forward very much does. So the RATIO is a
        // function of geometry by construction, and it grows as the output shrinks — measured 4.9x at
        // 1024², 7.4x at 768², and 12.6x at 512², which trips a ceiling calibrated at 1024². That is
        // the mechanism behaving exactly as described, not a regression, and asserting a single
        // constant across a 4x area range would be asserting something this rung never claimed.
        let slowdown = ms_per_step(&row, STEPS) / ms_per_step(&control, STEPS);
        if probing {
            println!(
                "[sc-15525 rung4 {tier} {edge}² cadence {window}] {slowdown:.1}x the control's \
                 ms/step — ceiling NOT asserted in probe mode (it is calibrated at 1024²)"
            );
        } else {
            assert!(
                slowdown < 12.0,
                "window {window} re-materialization cost {slowdown:.1}x the control's ms/step — \
                 SC-16355 flagged exactly this as a severe-regression risk, and past 12x the rung \
                 stops being a trade a selector could reasonably make"
            );
        }
    }

    // ── The two facts that make this a DOMAIN rather than four spellings of one choice ───────────
    assert!(
        rows.len() >= 2,
        "a cadence sweep needs at least two published cadences to compare; got {rows:?}"
    );
    // (a) Every cadence bounds the peak to the SAME value — **at this configuration, where the
    //     decode is the peak-bearing phase.** So here the wider cadences give up no memory and the
    //     latency they buy back is free. That is a property of the configuration and NOT of the
    //     family: at 512² q8 the decode transient no longer dominates, the spread opens to 5.7% and
    //     goes non-monotonic, which is why this assertion is skipped in probe mode rather than
    //     relaxed, and why the DEFAULT is the tightest cadence rather than the cheapest one. See
    //     `ms::TRANSFORMER_WINDOW_SIZES` for the mechanism and the condition it depends on.
    //
    //     1% is far outside the observed spread at the default configuration — the peak rows
    //     reproduce to the millibyte across four runs — so unlike the latency margins this one is
    //     tight on purpose. If a future change makes the peak cadence-sensitive *here*, the domain
    //     needs per-cadence calibration and this reddens to say so.
    let (tightest, tight_peak, tight_ms) = rows[0];
    let (widest, _wide_peak, wide_ms) = *rows.last().expect("at least one row");
    for (window, peak, _) in &rows {
        if probing {
            println!(
                "[sc-15525 rung4 {tier} {edge}² cadence {window}] peak {peak:.4} GiB \
                 ({:+.2}% vs cadence {tightest}) — flatness NOT asserted in probe mode",
                100.0 * (peak - tight_peak) / tight_peak
            );
            continue;
        }
        assert!(
            (peak - tight_peak).abs() < tight_peak * 0.01,
            "cadence {window} peaked at {peak:.4} GiB against cadence {tightest}'s {tight_peak:.4} \
             — the published domain claims every cadence bounds the peak identically AT THIS \
             CONFIGURATION, which is what lets a selector trade cadence for latency here without \
             re-calibrating. (It is not why cadence {tightest} is the default: the default is the \
             tightest weight bound, the one this provider can make good on at every advertised \
             geometry, and 512² q8 is already measured non-flat.) If this row is genuinely no \
             longer flat, the flat region has moved and the domain owes fresh per-cadence evidence"
        );
    }
    // (b) And widening it buys time back. Measured 3654 -> 1318 ms/step (2.77x) at 1024² and
    //     3355 -> 979 (3.43x) at 768²; run-to-run latency spread is ~4%, so a 0.75 ratio is a floor
    //     noise cannot clear while still failing loudly if the re-open count stops tracking the
    //     cadence at all (which is what a `BlockPlan` that ignored the window would look like).
    assert!(
        probing || wide_ms < tight_ms * 0.75,
        "widening the cadence {tightest} -> {widest} did not buy time back ({tight_ms:.0} -> \
         {wide_ms:.0} ms/step). The domain is published as a time/memory frontier; if every cadence \
         costs the same, it is not one and only the tightest should ship"
    );
    println!(
        "[sc-15525 rung4 {tier} {edge}² frontier] cadence {tightest} -> {widest}: peak flat at \
         {tight_peak:.3} GiB, wall clock {tight_ms:.0} -> {wide_ms:.0} ms/step ({:.2}x cheaper)",
        tight_ms / wide_ms
    );
}

/// **The published window domain is enforced on both layers, and it is REACHABLE on both.**
///
/// The Anima PR shipped a published rung-4 domain whose out-of-domain values were never checked
/// through the production path; this is the SDXL twin of that check, written so it cannot pass
/// vacuously. For every cadence it emits a `WINDOW-REQUEST` receipt naming what happened, and it
/// asserts **both** directions:
///
/// * every value in [`ms::TRANSFORMER_WINDOW_SIZES`] is *admitted* and renders — a gate that refused
///   the whole domain would otherwise satisfy every negative case here;
/// * every value outside it is *refused*, including interior gaps (3, 4, 6, 7, 9) that a
///   "clamp to the nearest legal cadence" bug would silently absorb, and values past the deepest
///   sub-stack (11, 70).
///
/// The interior gaps are the ones worth having. A domain of `[1, 2, 5, 10]` with a clamping resolver
/// would accept 7 and quietly execute 5 — under-predicting nothing, but executing a strategy the
/// selector did not choose, which is precisely what the shared contract forbids.
///
/// **This exercises the request-side layer specifically**, and that is the point of driving
/// `generate` with a hand-built `GenerationMemory`: it is the path a calibration harness takes, and
/// it never crosses admission, so the shared parameter validator is not in play. Disarming
/// `memory_strategy::validate_window` and re-running proves how thin the remaining margin is — 7 of
/// these 8 cadences are then **admitted and rendered**, and only `0` still fails, caught much deeper
/// by `BlockPlan::new`'s own `>= 1` guard. That one provider-side function is the whole gate on this
/// path.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_published_window_domain_is_enforced_and_reachable_on_the_production_path() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "q8") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing q8/");
    };
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let model = registry
        .load(
            "sdxl",
            &spec(&dir, "q8", LoadShape::DeferredMaterialization),
        )
        .expect("load sdxl deferred");

    let mut admitted = 0_usize;
    for size in ms::TRANSFORMER_WINDOW_SIZES {
        let outcome = model.generate(&request(Some(full_ladder(*size)), 1024, 2), &mut |_| {});
        println!(
            "[sc-15525 rung4 domain] WINDOW-REQUEST size={size} admitted={} refused={}",
            outcome.is_ok(),
            outcome.is_err()
        );
        outcome.unwrap_or_else(|err| {
            panic!("published cadence {size} must be admitted by the production path: {err}")
        });
        admitted += 1;
    }
    assert_eq!(
        admitted,
        ms::TRANSFORMER_WINDOW_SIZES.len(),
        "every published cadence must render"
    );

    // Collected rather than asserted in-loop, so a regression reports EVERY cadence it affects
    // instead of stopping at the first. That matters here: the interesting failure mode is "the
    // interior gaps got clamped while the extremes still error", and a fail-fast loop that trips on
    // cadence 0 would hide exactly that.
    const OUT_OF_DOMAIN: [u32; 8] = [0, 3, 4, 6, 7, 9, 11, 70];
    let mut silently_admitted = Vec::new();
    let mut refused_elsewhere = Vec::new();
    let mut refused_by_the_window_validator = 0_usize;
    for bad in OUT_OF_DOMAIN {
        assert!(
            !ms::TRANSFORMER_WINDOW_SIZES.contains(&bad),
            "the negative list must stay disjoint from the published domain"
        );
        let outcome = model.generate(&request(Some(full_ladder(bad)), 1024, 2), &mut |_| {});
        println!(
            "[sc-15525 rung4 domain] WINDOW-REQUEST size={bad} admitted={} refused={}",
            outcome.is_ok(),
            outcome.is_err()
        );
        match outcome {
            Ok(_) => silently_admitted.push(bad),
            Err(err) if err.to_string().contains("transformer window") => {
                refused_by_the_window_validator += 1;
            }
            Err(err) => refused_elsewhere.push(format!("{bad}: {err}")),
        }
    }
    assert!(
        silently_admitted.is_empty(),
        "cadences {silently_admitted:?} are outside {:?} and were ADMITTED — a clamped or ignored \
         window executes a strategy the selector did not choose, which is the defect this domain's \
         publication depends on not existing",
        ms::TRANSFORMER_WINDOW_SIZES
    );
    // Refused-by-something-else is a weaker guarantee than refused-by-the-window-validator, and the
    // difference is worth failing over: `BlockPlan::new` rejects a zero window on its own, so a
    // provider whose own validator silently stopped working would still look fine at that end of the
    // range while admitting the interior gaps. Requiring the provider's own refusal keeps the two
    // layers independently proven rather than letting the deeper one mask the nearer one.
    assert!(
        refused_elsewhere.is_empty(),
        "these cadences were refused, but NOT by memory_strategy::validate_window — the provider's \
         own domain gate is not the thing rejecting them: {refused_elsewhere:?}"
    );
    assert_eq!(
        refused_by_the_window_validator,
        OUT_OF_DOMAIN.len(),
        "every out-of-domain cadence must be refused by the provider's window validator"
    );
}

/// **Rung 4 fails closed on every precondition**, on real weights, through the production path.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn rung_four_preconditions_fail_closed_on_real_weights() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "q8") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing q8/");
    };
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    // 1. An EAGER load cannot window — the blocks are already committed, so a window would add a
    //    second copy rather than bounding anything.
    let eager = registry
        .load("sdxl", &spec(&dir, "q8", LoadShape::EagerMaterialization))
        .expect("load sdxl eager");
    let err = match eager.generate(&request(Some(full_ladder(1)), 1024, 2), &mut |_| {}) {
        Ok(_) => panic!("a window on an eager load must be refused"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("cannot stream"), "got: {err}");
    drop(eager);
    clear_cache();

    let deferred = registry
        .load(
            "sdxl",
            &spec(&dir, "q8", LoadShape::DeferredMaterialization),
        )
        .expect("load sdxl deferred");

    // 2. Rung 4 without rung 1 engaged in the same request.
    let unstaged = GenerationMemory {
        stage_residency: false,
        ..full_ladder(1)
    };
    let err = match deferred.generate(&request(Some(unstaged), 1024, 2), &mut |_| {}) {
        Ok(_) => panic!("rung 4 without rung 1 must be refused"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("staged residency"), "got: {err}");

    // 3. A window CADENCE outside the published domain. (The exhaustive both-directions check is
    //    `the_published_window_domain_is_enforced_and_reachable_on_the_production_path`; these are
    //    the fail-closed spot checks that belong beside the other preconditions.)
    for bad in [0_u32, 3, 7, 70] {
        assert!(
            !ms::TRANSFORMER_WINDOW_SIZES.contains(&bad),
            "the negative list must stay disjoint from the published domain"
        );
        let err = match deferred.generate(&request(Some(full_ladder(bad)), 1024, 2), &mut |_| {}) {
            Ok(_) => panic!("an out-of-domain window {bad} must be refused"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("transformer window"),
            "window {bad}: {err}"
        );
    }

    // 4. A component scope this family does not implement — never narrowed to `Dit`.
    for component in [
        mlx_gen::gen_core::TransformerComponent::TextEncoder,
        mlx_gen::gen_core::TransformerComponent::Both,
    ] {
        let memory = GenerationMemory {
            transformer_window_component: Some(component),
            ..full_ladder(1)
        };
        let err = match deferred.generate(&request(Some(memory), 1024, 2), &mut |_| {}) {
            Ok(_) => panic!("an unimplemented window component must be refused"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("component"), "got: {err}");
    }

    // The control: the published composition renders. Without it every assertion above would pass
    // against a generator that refuses everything.
    deferred
        .generate(&request(Some(full_ladder(1)), 1024, 2), &mut |_| {})
        .expect("the published rung-4 composition must render");
}

/// The eleven sub-stacks are the shape SC-16355 describes — asserted against the **built** U-Net,
/// not re-read from the config, so a config/checkpoint disagreement shows up here.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_unet_has_eleven_windowable_sub_stacks_at_two_depths() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "bf16") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing bf16/");
    };
    let unet = mlx_gen_sdxl::load_unet_dtype(&dir, mlx_rs::Dtype::Float16).expect("load unet");
    let depths = unet.transformer_stack_depths();
    println!("[sc-15525 rung4] Transformer2D depths: {depths:?}");
    assert_eq!(depths.len(), 11, "eleven Transformer2D sub-stacks");
    assert_eq!(
        depths.iter().filter(|d| **d == 10).count(),
        6,
        "six 10-deep"
    );
    assert_eq!(
        depths.iter().filter(|d| **d == 2).count(),
        5,
        "five 2-deep — SC-16355 said four; the up path has layers_per_block + 1 = 3 attentions"
    );
    assert_eq!(
        depths.iter().sum::<usize>(),
        70,
        "70 windowable blocks, which is also the IP-Adapter K/V pair count"
    );
}

// ── Per-entry coverage ───────────────────────────────────────────────────────────────────────────

/// The rung-4 support each catalog entry/tier is **expected** to publish, and why.
///
/// This is the table the SC-15525 review forced into existence. The first revision of
/// [`every_catalog_entry_loads_and_publishes_the_ladder`] recorded a staged peak per entry and never
/// looked at *which ladder the entry published* — so "every catalog entry publishes the ladder" was
/// asserted by its own name and by nothing else, while rung 4 was in fact `Missing` on two of the
/// five. The declaration is per (entry, tier) because `streamable` is a fact about the **snapshot**,
/// not about the family.
///
/// `illustrious_xl_v1` / `illustrious_xl_v2` at q8 are the two that do not stream, and the cause is
/// not the architecture — it is their published snapshot. Their `unet/` weights are genuinely packed
/// (u32 codes plus `.scales`/`.biases`, byte-for-byte the same shape as `realvisxl`'s q8), but their
/// `unet/config.json` carries **no `quantization` marker**. `mlx_gen::quant::packed_quant_bits` reads
/// that marker and only that marker, so `needs_load_time_quant` answers "yes", `load_leaves_blocks_lazy`
/// answers "no", and `streamable` refuses the rung.
///
/// That refusal is **correct as a fail-closed**, and it is deliberately not papered over here by
/// sniffing the tensors instead: the same marker is what `model.rs`'s F-144 tier guard reads to reject
/// a requested-vs-packed mismatch, so a provider that trusted the tensors while the guard trusted the
/// marker would disagree with itself about what tier is loaded — and rung 4 replays *the recorded
/// tier* into every re-materialized block. The fix belongs in the published snapshot (add the marker
/// these two are missing), not in a second, looser definition of "packed" on this side.
const EXPECTED_RUNG_FOUR: &[(&str, &str, bool)] = &[
    ("sdxl", "bf16", true),
    ("realvisxl", "bf16", true),
    ("realvisxl", "q4", true),
    ("realvisxl", "q8", true),
    ("realvisxl_lightning", "q4", true),
    // Packed on disk, but the `quantization` marker is absent from `unet/config.json`.
    ("illustrious_xl_v1", "q8", false),
    ("illustrious_xl_v2", "q8", false),
];

/// **Every catalog entry** resolves to this provider, loads, and publishes a ladder — and this test
/// asserts **which** ladder, per entry and per tier, not merely that a render happened.
///
/// Each entry's own staged peak is still recorded, because sharing this provider's code is explicitly
/// not what makes a catalog entry Verified. What is new is that the published rung-4 capability is
/// checked against [`EXPECTED_RUNG_FOUR`] in **both** directions: an entry that silently stops
/// streaming reddens, and so does one that silently starts. The second direction is the one that
/// matters for the two `illustrious` entries — the day their snapshot grows its `quantization`
/// marker, this test fails and says so, instead of leaving the evidence overstating coverage for
/// another cycle.
///
/// **And where rung 4 is declared Implemented, this test now measures what it is worth there** —
/// SC-16355's "request peak measured per tier" AC. Publishing a capability per entry per tier while
/// measuring its value on one entry is the same class of substitution as publishing a peak without
/// saying which row it came from; the other four Implemented declarations were inheriting
/// `realvisxl`'s number by family resemblance. Each now carries its own.
#[test]
#[ignore = "needs the real SDXL-family snapshots (see the module docs for the env vars)"]
fn every_catalog_entry_loads_and_publishes_the_ladder() {
    use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};

    let mut covered = Vec::new();
    let mut absent = Vec::new();
    let mut exercised: Vec<(&str, &str)> = Vec::new();
    let mut streaming = Vec::new();
    let mut not_streaming = Vec::new();
    for (entry, var) in ENTRIES {
        let mut tiers = Vec::new();
        for tier in ["bf16", "q4", "q8"] {
            let Some(dir) = tier_dir(var, tier) else {
                continue;
            };
            let shape = LoadShape::DeferredMaterialization;
            let load_spec = spec(&dir, tier, shape);

            // What THIS entry/tier publishes, read off the contract the provider actually builds for
            // its own `LoadSpec` — not off the family's representative entry.
            let contract = ms::memory_strategy_contract(mlx_gen_sdxl::MODEL_ID, &load_spec)
                .expect("contract must build for a cached snapshot");
            let streams = ms::streamable(&load_spec);
            let rung_four = &contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4 capability")
                .support;
            assert_eq!(
                streams,
                matches!(rung_four, MemoryStrategySupport::Implemented),
                "{entry}/{tier}: `streamable` and the published rung-4 support disagree — the \
                 contract must never advertise a rung this load cannot execute"
            );
            let expected = EXPECTED_RUNG_FOUR
                .iter()
                .find(|(e, t, _)| *e == *entry && *t == tier)
                .unwrap_or_else(|| {
                    panic!(
                        "{entry}/{tier} is cached and was measured, but has no EXPECTED_RUNG_FOUR \
                         row. A newly-cached tier must be added to that table (and to the contract \
                         evidence and the SceneWorks matrix) rather than measured unverified"
                    )
                });
            assert_eq!(
                streams, expected.2,
                "{entry}/{tier}: rung-4 availability changed. EXPECTED_RUNG_FOUR (and the contract \
                 evidence, and the SceneWorks matrix cell) must be updated together — see that \
                 table for why the illustrious entries differ"
            );
            exercised.push((*entry, tier));

            let row = measure(&dir, tier, shape, &request(Some(staged()), 1024, 4));
            println!(
                "[sc-15525 entry {entry} tier {tier}] staged request peak {:.3} GiB, {:.0} ms/step, \
                 rung 4 {rung_four:?}",
                row.peak_gib,
                ms_per_step(&row, 4),
            );
            if streams {
                // **SC-16355's "request peak measured per tier" AC.** The row above is rung *1*'s
                // staged peak; it says nothing about what rung 4 is worth on THIS entry at THIS
                // tier. Every `true` row of `EXPECTED_RUNG_FOUR` declares rung 4 Implemented, and
                // before this the only entry/tier with a rung-4 request-level number was
                // `realvisxl` — the other declarations inherited its figure by family resemblance,
                // which is exactly the substitution the per-entry table exists to refuse.
                let windowed = measure(
                    &dir,
                    tier,
                    shape,
                    &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 1024, 4),
                );
                let delta = 100.0 * (windowed.peak_gib - row.peak_gib) / row.peak_gib;
                println!(
                    "[sc-15525 entry {entry} tier {tier}] rung-4 request peak at the shipped \
                     cadence {} = {:.3} GiB ({delta:+.2}% vs this entry's staged row), {:.0} ms/step",
                    ms::TRANSFORMER_WINDOW_SIZE,
                    windowed.peak_gib,
                    ms_per_step(&windowed, 4),
                );
                // A declaration of Implemented that does not move the request peak is a false
                // green — the rung would be reaching the blocks and buying nothing. The bar is 3%,
                // set well under the smallest measured saving (q4's ~7%, where the block set is
                // smallest relative to the resident remainder) and far above the noise, which for
                // MLX's per-process peak accounting is exactly zero across repeated runs.
                assert!(
                    delta < -3.0,
                    "{entry}/{tier} publishes rung 4 as Implemented but streaming it moved the \
                     request peak only {delta:+.2}% ({:.3} -> {:.3} GiB). A rung that reaches \
                     every block and saves nothing must be declared Missing with that measurement, \
                     not Implemented",
                    row.peak_gib,
                    windowed.peak_gib
                );
                streaming.push(format!("{entry}/{tier} rung4 {delta:+.2}%"));
            } else {
                not_streaming.push(format!("{entry}/{tier}"));
            }
            tiers.push(tier);
        }
        if tiers.is_empty() {
            absent.push(*entry);
        } else {
            covered.push((*entry, tiers));
        }
    }
    println!("[sc-15525] entries measured: {covered:?}");
    println!("[sc-15525] entries with NO cached tier: {absent:?}");
    println!("[sc-15525] rung 4 IMPLEMENTED on: {streaming:?}");
    println!("[sc-15525] rung 4 MISSING on:     {not_streaming:?}");
    assert!(
        !covered.is_empty(),
        "at least one catalog entry must be cached; set the env vars in the module docs"
    );
    // The control that keeps this test honest in the direction that matters: at least one measured
    // entry/tier must actually be on each side. A run that saw only streaming loads would pass every
    // assertion above while proving nothing about the refusal, and vice versa.
    // Every row of the table must have been exercised. `.find()` alone let a thinner machine go green
    // with rows unverified — the table would then be documentation, not a check.
    let unexercised: Vec<_> = EXPECTED_RUNG_FOUR
        .iter()
        .filter(|(e, t, _)| !exercised.iter().any(|(ee, tt)| ee == e && tt == t))
        .map(|(e, t, _)| format!("{e}/{t}"))
        .collect();
    assert!(
        unexercised.is_empty(),
        "EXPECTED_RUNG_FOUR rows {unexercised:?} were never measured — cache those snapshots or \
         remove the rows; an unexercised row asserts nothing"
    );
    assert!(
        !streaming.is_empty() && !not_streaming.is_empty(),
        "this test's per-entry claim needs both a streaming and a non-streaming snapshot cached; \
         got streaming={streaming:?} not_streaming={not_streaming:?}"
    );
}

/// A memory-capped host is where this ladder exists to help, so the full composition must still
/// render under a cap that the resident path would not fit.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn the_full_ladder_renders_under_a_memory_cap() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "q8") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing q8/");
    };
    // SAFETY: `RUST_TEST_THREADS=1` is forced repo-wide, so no other test observes this env var
    // concurrently, and it is cleared before the function returns.
    unsafe { std::env::set_var(MEMORY_CAP_ENV, "8") };
    // EVERY published cadence, not just one — and emphatically not the cadence that used to be the
    // default. This is the only test that exercises the small Mac this rung exists for, so pinning it
    // to a single hardcoded window meant the shipped default was never the one proven to render
    // under a cap. A domain the selector may choose from must be a domain that renders.
    let rows: Vec<(u32, Row)> = ms::TRANSFORMER_WINDOW_SIZES
        .iter()
        .map(|window| {
            (
                *window,
                measure(
                    &dir,
                    "q8",
                    LoadShape::DeferredMaterialization,
                    &request(Some(full_ladder(*window)), 1024, 4),
                ),
            )
        })
        .collect();
    unsafe { std::env::remove_var(MEMORY_CAP_ENV) };
    for (window, row) in &rows {
        println!(
            "[sc-15525 capped 8 GiB q8 1024² window {window}] full-ladder request peak {:.3} GiB, \
             {:.0} ms/step",
            row.peak_gib,
            ms_per_step(row, 4)
        );
    }
    assert!(
        rows.iter().any(|(w, _)| *w == ms::TRANSFORMER_WINDOW_SIZE),
        "the shipped default cadence {} must be among the cadences proven to render under a cap",
        ms::TRANSFORMER_WINDOW_SIZE
    );
    for (window, row) in &rows {
        let _ = window;
        // `!pixels.is_empty()` was the whole assertion in the first revision, which a decode returning a
        // single black pixel would satisfy. Bind the two facts a capped render actually has to deliver:
        // the requested geometry, and an image with content in it. Neither is satisfiable by a no-op.
        assert_eq!(
            row.pixels.len(),
            1024 * 1024 * 3,
            "a capped full-ladder render must still produce the requested 1024² RGB image"
        );
        let first = row.pixels[0];
        assert!(
        row.pixels.iter().any(|p| *p != first),
        "the capped render produced a uniform image — the window streamed blocks that decoded to \
         nothing rather than reproducing the resident stack"
    );
    }
    // Every cadence must render the SAME image under the cap. The cap changes MLX's allocator
    // behaviour, not the arithmetic, so a cadence that diverges here is re-materializing differently
    // under memory pressure — the failure mode a capped host would hit and an uncapped one would not.
    let reference = &rows[0].1.pixels;
    for (window, row) in &rows {
        assert_eq!(
            max_delta(reference, &row.pixels),
            0,
            "cadence {window} rendered a different image under the cap than cadence {} did",
            rows[0].0
        );
    }
}
