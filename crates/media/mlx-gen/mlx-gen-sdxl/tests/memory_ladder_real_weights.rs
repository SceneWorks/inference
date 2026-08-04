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

/// One measured row: the request's ACTIVE-bytes peak and its pixels.
struct Row {
    peak_gib: f64,
    pixels: Vec<u8>,
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
    let out = model
        .generate(req, &mut |_: Progress| {})
        .expect("generate must succeed");
    let peak = get_peak_memory();
    let pixels = match out {
        GenerationOutput::Images(images) => images.first().expect("one image").pixels.clone(),
        other => panic!("expected images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    Row {
        peak_gib: peak as f64 / GIB,
        pixels,
    }
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
    let reference = vae.decode(&latent).expect("untiled decode");
    reference.eval().expect("eval reference");
    let ref_px = mlx_gen_sdxl::decoded_to_image(&reference)
        .expect("reference image")
        .pixels;

    println!("[sc-15525 rung2 mechanism] latent {:?}", latent.shape());
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
            println!(
                "| {edge} | {overlap} | {tiles} | {peak:.3} | {} | {:.4} |",
                max_delta(&ref_px, &px),
                mean_delta(&ref_px, &px)
            );
        }
    }
}

// ── Rung 3 ───────────────────────────────────────────────────────────────────────────────────────

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

/// **The rung-4 window sweep**, and the bit-identity proof that the eleven streamed sub-stacks
/// reproduce the resident ones exactly.
#[test]
#[ignore = "needs a real SDXL-family snapshot (see the module docs for the env vars)"]
fn transformer_window_sweep_and_streamed_output_identity() {
    let Some(dir) = tier_dir(REPRESENTATIVE, "q8") else {
        panic!("SKIPPED-BY-ABSENCE: set {REPRESENTATIVE} to a snapshot root containing q8/");
    };
    let deferred = LoadShape::DeferredMaterialization;
    // The attribution control: the same composition WITHOUT rung 4, on a deferred load, so the only
    // difference between the two rows is the window itself.
    let control = measure(&dir, "q8", deferred, &request(Some(staged()), 1024, 6));
    println!(
        "[sc-15525 rung4 q8 1024²] staged, no window (attribution control) {:.3} GiB",
        control.peak_gib
    );
    for window in ms::TRANSFORMER_WINDOW_SIZES {
        let row = measure(
            &dir,
            "q8",
            deferred,
            &request(Some(full_ladder(*window)), 1024, 6),
        );
        println!(
            "[sc-15525 rung4 window {window}] request peak {:.3} GiB  ({:+.2}% vs control)",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib
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
    }
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

    // 3. A window CADENCE outside the published domain.
    for bad in [0_u32, 3, 7, 70] {
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

/// **Every catalog entry** resolves to this provider, loads, and publishes the same ladder — and
/// each one's *own* peak is recorded, because sharing code is not what makes an entry Verified.
#[test]
#[ignore = "needs the real SDXL-family snapshots (see the module docs for the env vars)"]
fn every_catalog_entry_loads_and_publishes_the_ladder() {
    let mut covered = Vec::new();
    let mut absent = Vec::new();
    for (entry, var) in ENTRIES {
        let mut tiers = Vec::new();
        for tier in ["bf16", "q4", "q8"] {
            let Some(dir) = tier_dir(var, tier) else {
                continue;
            };
            let shape = LoadShape::DeferredMaterialization;
            let row = measure(&dir, tier, shape, &request(Some(staged()), 1024, 4));
            println!(
                "[sc-15525 entry {entry} tier {tier}] staged request peak {:.3} GiB",
                row.peak_gib
            );
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
    assert!(
        !covered.is_empty(),
        "at least one catalog entry must be cached; set the env vars in the module docs"
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
    let row = measure(
        &dir,
        "q8",
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(1)), 1024, 4),
    );
    unsafe { std::env::remove_var(MEMORY_CAP_ENV) };
    println!(
        "[sc-15525 capped 8 GiB q8 1024²] full-ladder request peak {:.3} GiB",
        row.peak_gib
    );
    assert!(!row.pixels.is_empty());
}
