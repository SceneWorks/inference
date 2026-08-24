//! Real-weight conformance and **measurement** for the Kolors shared memory ladder (SC-15521), on
//! Apple/Metal.
//!
//! Every number published in `crate::memory_strategy` comes from this file. **Nothing is inherited
//! from SC-15525**, even though Kolors re-exports SDXL's `UNet2DConditionModel` unchanged: the
//! epic's standing rule is that a rung's presence, magnitude, mechanism and candidate set are per
//! family per backend, and Kolors swaps SDXL's ~0.8B dual-CLIP conditioning for a 6B ChatGLM3 tower
//! — which is exactly the kind of difference that moves which phase carries the request peak.
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
//! * **A discarded warm-up row ahead of every published peak** ([`warm_up`]). MLX's first
//!   measurement in a process reads against a cold allocator and comes in low, and the bias is
//!   large enough to look like a finding: it manufactured a 4.9% phantom cadence spread in one test
//!   and a 12.54% phantom "seam regression" in another. Both were published before they were
//!   caught. Every test in this file that publishes a peak now discards a row first —
//!   [`attention_chunking_is_measured_at_the_unet_seam`] drives the U-Net seam directly rather than
//!   through [`measure`], so it carries its own inline warm-up instead of calling [`warm_up`].
//! * Rejected candidates are recorded **with their numbers**, and the rejection is re-asserted
//!   against the production path — not left in a doc comment.
//!
//! ## Weights
//!
//! One env var, pointing at the `SceneWorks/kolors-mlx` snapshot **root** (the tier is a
//! subdirectory: `q4` / `q8` / `bf16`). Nothing self-fetches or derives a cache location
//! (epic 13657). A test whose tier is absent **skips loudly by name** rather than passing silently.
//!
//! | env var | entry |
//! |---|---|
//! | `KOLORS_LADDER_ROOT` | `kolors` — the single catalog entry, all three advertised tiers |
//!
//! ```text
//! KOLORS_LADDER_ROOT=<the SceneWorks/kolors-mlx snapshot root, containing q4/ q8/ bf16/> \
//!   cargo test -p mlx-gen-kolors --test integration memory_ladder_real_weights:: -- --ignored --test-threads=1
//! ```
//!
//! The env var is the only input: nothing here derives a cache location (epic 13657).

#![allow(clippy::items_after_test_module)]

use std::path::PathBuf;

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Progress, TransformerComponent,
};
use mlx_gen::memory::MEMORY_CAP_ENV;
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen_kolors::memory_strategy as ms;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The env var carrying the one catalog entry's snapshot root.
const ROOT_ENV: &str = "KOLORS_LADDER_ROOT";

/// The three advertised tiers, in catalog order (`q4` is the manifest default).
const TIERS: &[&str] = &["q4", "q8", "bf16"];

/// The tier every default-mode measurement runs at — the catalog's own default, so the asserted
/// rows describe what a caller who names nothing actually gets.
const DEFAULT_TIER: &str = "q4";

/// Resolve one tier directory, or `None` when it is not cached.
fn tier_dir(tier: &str) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var(ROOT_ENV).ok()?);
    let dir = root.join(tier);
    dir.is_dir().then_some(dir)
}

#[track_caller]
fn require_tier(tier: &str) -> PathBuf {
    match tier_dir(tier) {
        Some(dir) => dir,
        None => panic!("SKIPPED-BY-ABSENCE: set {ROOT_ENV} to a snapshot root containing {tier}/"),
    }
}

fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// A load spec for one tier at the shape the ladder needs.
fn spec(dir: &std::path::Path, tier: &str, shape: LoadShape) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(shape);
    spec.quantize = quant_for(tier);
    spec
}

/// The env var carrying the converted `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot — the layout
/// [`mlx_gen_kolors::ip_adapter::load_kolors_ip_adapter`] reads (`image_encoder/model.safetensors`
/// plus `ip_adapter_plus_general.safetensors`).
const IP_ADAPTER_ENV: &str = "KOLORS_IP_ADAPTER";

#[track_caller]
fn require_ip_adapter() -> PathBuf {
    let root = PathBuf::from(std::env::var(IP_ADAPTER_ENV).unwrap_or_else(|_| {
        panic!("SKIPPED-BY-ABSENCE: set {IP_ADAPTER_ENV} to a Kolors-IP-Adapter-Plus snapshot dir")
    }));
    assert!(
        root.join("ip_adapter_plus_general.safetensors").is_file(),
        "SKIPPED-BY-ABSENCE: {IP_ADAPTER_ENV} must point at a snapshot containing \
         ip_adapter_plus_general.safetensors (the upstream repo ships .bin; it needs converting)"
    );
    root
}

/// A flat mid-grey image prompt — the IP-Adapter identity. Its *content* is irrelevant here: the
/// claim under test is that the streamed and resident stacks agree, not what they draw.
fn ip_reference_image() -> mlx_gen::gen_core::Image {
    const EDGE: u32 = 336;
    mlx_gen::gen_core::Image {
        width: EDGE,
        height: EDGE,
        pixels: vec![128_u8; (EDGE * EDGE * 3) as usize],
    }
}

fn request(memory: Option<GenerationMemory>, edge: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, lowres".into()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(steps),
        guidance: Some(5.0),
        seed: Some(1234),
        memory,
        ..Default::default()
    }
}

/// One measured row: the request's ACTIVE-bytes peak, its pixels, and diagnostic wall clock.
///
/// Rung 4 re-materializes blocks, so the clock remains useful output for a human investigating a
/// run. It is deliberately not evidence: thermal state and unrelated host work can change it
/// without changing the allocation or output contract this harness grades.
struct Row {
    peak_gib: f64,
    pixels: Vec<u8>,
    wall: std::time::Duration,
}

/// Render one row on a **fresh** generator and return its request peak.
///
/// The freshness is the whole contract of this helper. Kolors' U-Net weights are lazy MLX handles
/// until something evaluates them, so a generator reused across rows carries the previous row's
/// materialization into this row's peak — which is exactly how a rung-4 sweep can report a saving
/// that is really just "the first row paid for the stack". Every row in this file goes through here.
#[track_caller]
fn measure(dir: &std::path::Path, tier: &str, shape: LoadShape, req: &GenerationRequest) -> Row {
    let registry = mlx_gen_kolors::provider_registry().expect("provider registry");
    let model = registry
        .load("kolors", &spec(dir, tier, shape))
        .expect("load kolors");
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

/// Grade the rung-4 product claim, never a host-dependent duration.
///
/// The independent mutation test below makes all three legs non-vacuous: a missing evaluation or
/// output check, an inert stream, and corrupt peak accounting each reject their synthetic row.
fn assert_rung_four_evidence(control: &Row, row: &Row, window: u32, require_peak_reduction: bool) {
    assert!(
        control.peak_gib.is_finite()
            && row.peak_gib.is_finite()
            && control.peak_gib > 0.0
            && row.peak_gib > 0.0,
        "cadence {window} has invalid ACTIVE-byte accounting: control {:.4} GiB, row {:.4} GiB",
        control.peak_gib,
        row.peak_gib
    );
    assert_eq!(
        control.pixels, row.pixels,
        "cadence {window} is a residency change, not an arithmetic one — a streamed block must be \
         byte-identical to its resident twin"
    );
    if require_peak_reduction {
        assert!(
            row.peak_gib < control.peak_gib * 0.97,
            "cadence {window} must bound the request peak by more than the 3% margin ({:.4} vs \
             {:.4} GiB)",
            row.peak_gib,
            control.peak_gib
        );
    }
}

#[test]
fn rung_four_evidence_rejects_output_loss_inert_stream_and_corrupt_peak_accounting() {
    let control = Row {
        peak_gib: 100.0,
        pixels: vec![1, 2, 3],
        wall: std::time::Duration::ZERO,
    };
    let valid = Row {
        peak_gib: 90.0,
        pixels: control.pixels.clone(),
        wall: std::time::Duration::ZERO,
    };
    assert_rung_four_evidence(&control, &valid, 1, true);

    let changed_output = Row {
        pixels: vec![1, 2, 4],
        ..valid
    };
    assert!(std::panic::catch_unwind(|| {
        assert_rung_four_evidence(&control, &changed_output, 1, true)
    })
    .is_err());

    let inert_stream = Row {
        peak_gib: 100.0,
        pixels: control.pixels.clone(),
        wall: std::time::Duration::ZERO,
    };
    assert!(std::panic::catch_unwind(|| {
        assert_rung_four_evidence(&control, &inert_stream, 1, true)
    })
    .is_err());

    let corrupt_accounting = Row {
        peak_gib: 0.0,
        pixels: control.pixels.clone(),
        wall: std::time::Duration::ZERO,
    };
    assert!(std::panic::catch_unwind(|| {
        assert_rung_four_evidence(&control, &corrupt_accounting, 1, true)
    })
    .is_err());
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

fn full_ladder(window: u32) -> GenerationMemory {
    full_ladder_scoped(window, ms::TRANSFORMER_WINDOW_COMPONENT)
}

fn full_ladder_scoped(window: u32, component: TransformerComponent) -> GenerationMemory {
    GenerationMemory {
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(component),
        ..staged()
    }
}

/// Discard one measured row before publishing any peak from this process.
///
/// **This is a measurement-integrity control, not hygiene.** MLX's `get_peak_memory` reads ACTIVE
/// bytes, and the very first `generate` in a process reads them against a cold allocator: an earlier
/// revision of `the_cadence_flatness_condition_is_checked_not_assumed` measured a windowed row first
/// and got **4.4632 GiB** for a configuration `transformer_window_sweep_and_streamed_output_identity`
/// reads as **4.6924 GiB**, a 4.9% phantom spread that looked exactly like the flat region breaking
/// at the advertised `min_size` — i.e. like a real and publishable finding. It was not. Warming up
/// removes it: the same row then reads 4.6924 to the millibyte.
///
/// The warm-up is deliberately a *windowed* row, because that is the shape the bias was observed on
/// (rung 4 calls `clear_cache()` at every window boundary, which is what interacts with the cold
/// allocator). It runs at the smallest advertised output for one step, so it costs seconds.
#[track_caller]
fn warm_up(dir: &std::path::Path, tier: &str) {
    let _ = measure(
        dir,
        tier,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
    );
}

/// The default step count for a measured row. Four is enough for the sampler to have fed a
/// per-forward divergence back into itself, and cheap enough that a five-row sweep is minutes.
const STEPS: u32 = 4;

// ── Rung 0 / rung 1 ──────────────────────────────────────────────────────────────────────────────

/// **Rung 1 is request-scoped and it moves the request peak — by far the most, on this family.**
///
/// The same cached generator serves resident → staged, and the staged request must peak strictly
/// lower while producing a **byte-identical** image: rung 1 sheds the ChatGLM3-6B encoder before the
/// heavy bundle loads, which is a residency change and not an arithmetic one.
///
/// The margin is what makes this Kolors' evidence rather than SDXL's. SDXL's rung 1 buys −7.4%,
/// because the dual CLIP pair it sheds is small next to the U-Net + decode transient it uncovers.
/// Kolors sheds a component that is **2.2× the U-Net at q4 and 2.4× at bf16**, so the assertion here
/// is set against a much larger measured saving. A change that reduced rung 1 to SDXL's magnitude
/// would redden this test rather than quietly re-describing the family.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn staged_residency_bounds_the_request_peak_and_preserves_output() {
    let dir = require_tier(DEFAULT_TIER);
    warm_up(&dir, DEFAULT_TIER);
    let resident = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::EagerMaterialization,
        &request(Some(resident_memory()), 1024, STEPS),
    );
    let staged_row = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::EagerMaterialization,
        &request(Some(staged()), 1024, STEPS),
    );
    let saved = 100.0 * (resident.peak_gib - staged_row.peak_gib) / resident.peak_gib;
    println!(
        "[sc-15521 rung1 {DEFAULT_TIER} 1024² {STEPS} steps] resident {:.4} GiB -> staged {:.4} GiB \
         ({saved:.2}%)  {:.0} -> {:.0} ms/step",
        resident.peak_gib,
        staged_row.peak_gib,
        ms_per_step(&resident, STEPS),
        ms_per_step(&staged_row, STEPS),
    );
    assert!(
        staged_row.peak_gib < resident.peak_gib * 0.97,
        "rung 1 must bound the request peak by more than the 3% margin ({:.4} vs {:.4} GiB)",
        staged_row.peak_gib,
        resident.peak_gib
    );
    assert_eq!(
        resident.pixels, staged_row.pixels,
        "rung 1 is a residency change, not an arithmetic one — the image must be byte-identical"
    );
}

// ── Rungs 2 and 3: refused by the production path ────────────────────────────────────────────────

/// **Rung 2 and rung 3 are refused by the production path**, on real weights.
///
/// Both were measured (see [`decode_tile_mechanism_sweep`] and
/// [`attention_chunking_is_measured_against_the_rung_two_top`]) and both are declared `Missing`.
/// Their mechanisms are still reachable through the crate — which is exactly why this test exists:
/// the only thing between `Autoencoder::decode_tiled` and a production render is the refusal.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_rejected_rungs_are_refused_by_the_production_path() {
    let dir = require_tier(DEFAULT_TIER);
    let registry = mlx_gen_kolors::provider_registry().expect("provider registry");
    let model = registry
        .load(
            "kolors",
            &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
        )
        .expect("load kolors");

    let tiled = GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(ms::DECODE_TILE_EDGE),
        decode_overlap: Some(ms::DECODE_OVERLAP),
        ..staged()
    };
    let err = model
        .generate(&request(Some(tiled), 1024, 1), &mut |_: Progress| {})
        .expect_err("a bounded-decode request must be refused by the production path");
    assert!(
        err.to_string().contains("rung 2 is declared"),
        "the refusal must name the withheld rung, got: {err}"
    );

    let chunked = GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(ms::ATTENTION_CHUNK_SIZE),
        ..staged()
    };
    let err = model
        .generate(&request(Some(chunked), 1024, 1), &mut |_: Progress| {})
        .expect_err("a bounded-attention request must be refused by the production path");
    assert!(
        err.to_string().contains("rung 3 is declared"),
        "the refusal must name the withheld rung, got: {err}"
    );

    // The control: without it, a `generate` that rejected everything would pass this test.
    model
        .generate(&request(Some(staged()), 512, 1), &mut |_: Progress| {})
        .expect("a plain staged request must still render");
}

// ── The staged re-assembly, for the two rungs `generate` refuses ─────────────────────────────────

/// One end-to-end row for a rung the production path **refuses**, measured over the whole request
/// rather than at one U-Net forward.
///
/// `generate` cannot produce this row: `memory_strategy::decode_tiling` and
/// `memory_strategy::attention_plan` reject their selections on every layer, which is the point of
/// rungs 2 and 3 being `Missing`. So the request is re-assembled here from the same public entry
/// points `KolorsGenerator::generate_impl` calls, in the same order, under the same **rung-1 staged
/// schedule** — encode with the ChatGLM3-6B tower alive, `eval` the conditioning, drop the tower,
/// *then* load the heavy bundle and denoise + decode.
///
/// Everything is rebuilt per row (`#[track_caller]`, same discipline as [`measure`]).
///
/// One deliberate difference from [`measure`]: the peak window opens **before** the loads rather
/// than after. That is not sloppiness, it is what a staged schedule *is* — the claim rung 1 makes is
/// that the encoder load and the heavy load never coexist, and a window that opened after both would
/// measure the thing the schedule exists to avoid. The reconstruction is bound to the real
/// `generate` by [`the_end_to_end_reassembly_reproduces_the_real_generate_peak`].
#[track_caller]
fn measure_end_to_end(
    dir: &std::path::Path,
    tier: &str,
    edge: u32,
    steps: usize,
    chunked: bool,
    decode_tiling: Option<&mlx_gen::tiling::TilingConfig>,
) -> Row {
    let (row, _) = measure_end_to_end_phased(dir, tier, edge, steps, chunked, decode_tiling);
    row
}

/// The same re-assembly, additionally reporting the **conditioning-phase** peak separately from the
/// whole-request peak.
///
/// The split is what answers the `TransformerComponent::TextEncoder` scope question
/// (`the_text_encoder_window_scope_cannot_move_the_request_peak`): a text-encoder window can only
/// move the request peak if the conditioning phase is the peak-bearing one, and on a family whose
/// encoder is 6B that is a live question rather than SDXL's foregone one.
///
/// **The conditioning split is not bound to production anywhere, and that limit is real.**
/// [`the_end_to_end_reassembly_reproduces_the_real_generate_peak`] binds this re-assembly to the
/// real `generate` at exactly one cell (q4, 1024²) and only on the WHOLE-REQUEST peak — `generate`
/// publishes no per-phase peak, so there is nothing to bind the split against. The nine-cell table
/// in `memory_strategy::TRANSFORMER_WINDOW_COMPONENTS` is therefore harness evidence. Its *decisive*
/// cell is separately corroborated on the production path
/// (`the_text_encoder_window_bounds_the_conditioning_bearing_cell` drives real `generate` rows at
/// bf16 512² and measures the TextEncoder scope moving the request peak, which can only happen if
/// the conditioning phase carries it); the other eight cells are not.
#[track_caller]
fn measure_end_to_end_phased(
    dir: &std::path::Path,
    tier: &str,
    edge: u32,
    steps: usize,
    chunked: bool,
    decode_tiling: Option<&mlx_gen::tiling::TilingConfig>,
) -> (Row, f64) {
    use mlx_gen_kolors::model::{KolorsHeavy, KolorsText};
    use mlx_rs::Dtype::Float16;

    let prompt = "a red fox in a snowy forest, photograph";
    let negative = "blurry, lowres";
    let bits = quant_for(tier).map(|q| q.bits());

    clear_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();

    // ── Conditioning phase: the ChatGLM3-6B tower alive, then released.
    let mut text = KolorsText::load(dir, Float16).expect("chatglm3");
    if let Some(bits) = bits {
        text.quantize(bits).expect("quantize text encoder");
    }
    let pos = text.encode(prompt).expect("encode positive");
    let neg = text.encode(negative).expect("encode negative");
    // Force the encode before dropping the tower — an unevaluated output keeps it referenced
    // through the lazy graph, so the drop would free nothing.
    mlx_rs::transforms::eval([&pos.0, &pos.1, &neg.0, &neg.1]).expect("eval conditioning");
    let conditioning_peak = get_peak_memory() as f64 / GIB;
    drop(text);
    clear_cache();

    // ── Denoise + decode phase.
    let mut heavy = KolorsHeavy::load(dir, Float16).expect("unet + vae");
    if let Some(bits) = bits {
        heavy.quantize_unet(bits).expect("quantize unet");
    }
    let (lh, lw) = ((edge / 8) as i32, (edge / 8) as i32);
    mlx_rs::random::seed(1234).expect("seed");
    let noise = mlx_rs::random::normal::<f32>(&[1, lh, lw, 4], None, None, None).expect("noise");
    let plan = if chunked {
        mlx_gen_sdxl::SdxlForwardPlan::with_attention(mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::CONSTRAINED,
        ))
    } else {
        mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED
    };
    let cancel = mlx_gen::gen_core::CancelFlag::default();
    let latents = heavy
        .denoise_latents_with_preview(
            &noise,
            &pos,
            Some(&neg),
            steps,
            5.0,
            edge as i32,
            edge as i32,
            None,
            &cancel,
            &mut |_: Progress| {},
            &mlx_gen::PreviewSink::default(),
            plan,
            mlx_gen::gen_core::CfgBatching::Batched,
        )
        .expect("denoise");
    let image =
        mlx_gen_sdxl::decode_image_tiled(heavy.vae(), &latents, None, decode_tiling, Some(&cancel))
            .expect("decode");

    let peak = get_peak_memory();
    let wall = started.elapsed();
    drop(heavy);
    clear_cache();
    (
        Row {
            peak_gib: peak as f64 / GIB,
            pixels: image.pixels,
            wall,
        },
        conditioning_peak,
    )
}

/// **The re-assembly is bound to the real `generate`, because two withholding verdicts rest on it.**
///
/// [`measure_end_to_end`] rebuilds the staged request from the public entry points, and both rung
/// 2's and rung 3's `Missing` verdicts are measured through it — because the production path refuses
/// both selections, so `generate` cannot produce either row. Without this binding a change to
/// `KolorsGenerator::generate_impl` — a dtype, an added component, a reordered phase — would leave
/// both verdicts resting on a harness that no longer models the thing it stands in for, with nothing
/// red.
///
/// The margin is 1%, not a bare equality: the two paths are the same phases in the same order, but
/// MLX peak accounting is exact only for a fixed allocation sequence and the harness does not
/// reproduce `generate`'s progress/cancel plumbing or its per-image count loop. 1% is far tighter
/// than any structural divergence — pointing the re-assembly at 512² while `generate` renders 1024²
/// reddens it by more than 50%.
///
/// **What it binds is the PEAK**, which is not the same as total fidelity. A divergence that does
/// not move the peak passes; that is the honest scope.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_end_to_end_reassembly_reproduces_the_real_generate_peak() {
    let dir = require_tier(DEFAULT_TIER);
    warm_up(&dir, DEFAULT_TIER);
    let production = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::EagerMaterialization,
        &request(Some(staged()), 1024, STEPS),
    );
    let reassembled = measure_end_to_end(&dir, DEFAULT_TIER, 1024, STEPS as usize, false, None);
    let drift = 100.0 * (reassembled.peak_gib - production.peak_gib).abs() / production.peak_gib;
    println!(
        "[sc-15521 harness fidelity {DEFAULT_TIER} 1024² {STEPS} steps] generate {:.4} GiB vs \
         re-assembly {:.4} GiB ({drift:.3}% apart)",
        production.peak_gib, reassembled.peak_gib
    );
    assert!(
        drift < 1.0,
        "the rung-2/rung-3 harness no longer models `generate`'s staged request ({:.4} vs {:.4} \
         GiB, {drift:.2}% apart). Both withholding verdicts are measured through that re-assembly, \
         so they are only as good as this agreement",
        reassembled.peak_gib,
        production.peak_gib
    );
}

// ── Rung 2 ───────────────────────────────────────────────────────────────────────────────────────

/// Overlap retained in the sealed policy identity; SC-19753 removed it from decode arithmetic.
const POLICY_DECODE_OVERLAP: u32 = 256;

/// Render one real Kolors image at `edge` and return the **production latent** — exactly what the
/// denoiser hands the decode phase.
///
/// This is the instrument choice that decides rung 2, and it is deliberately the harsher one. An
/// earlier revision swept a latent obtained by *re-encoding a finished image*, whose statistics have
/// already been through the VAE round trip; on Kolors that instrument reads **29/255** at the best
/// geometry where the production latent reads **41/255** at the same one. A user gets the production
/// latent, so the absolute bar has to be judged on it. (Both were measured; the re-encoded row is
/// what the first revision of this file published, and it was too generous by ~40%.)
#[track_caller]
fn production_latent(dir: &std::path::Path, tier: &str, edge: u32) -> mlx_rs::Array {
    use mlx_gen_kolors::model::{KolorsHeavy, KolorsText};
    use mlx_rs::Dtype::Float16;

    let bits = quant_for(tier).map(|q| q.bits());
    let mut text = KolorsText::load(dir, Float16).expect("chatglm3");
    if let Some(bits) = bits {
        text.quantize(bits).expect("quantize text encoder");
    }
    let pos = text
        .encode("a red fox in a snowy forest, photograph")
        .expect("pos");
    let neg = text.encode("blurry, lowres").expect("neg");
    mlx_rs::transforms::eval([&pos.0, &pos.1, &neg.0, &neg.1]).expect("eval");
    drop(text);
    clear_cache();

    let mut heavy = KolorsHeavy::load(dir, Float16).expect("heavy");
    if let Some(bits) = bits {
        heavy.quantize_unet(bits).expect("quantize unet");
    }
    let (lh, lw) = ((edge / 8) as i32, (edge / 8) as i32);
    mlx_rs::random::seed(1234).expect("seed");
    let noise = mlx_rs::random::normal::<f32>(&[1, lh, lw, 4], None, None, None).expect("noise");
    let cancel = mlx_gen::gen_core::CancelFlag::default();
    let latents = heavy
        .denoise_latents_with_preview(
            &noise,
            &pos,
            Some(&neg),
            STEPS as usize,
            5.0,
            edge as i32,
            edge as i32,
            None,
            &cancel,
            &mut |_: Progress| {},
            &mlx_gen::PreviewSink::default(),
            mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED,
            mlx_gen::gen_core::CfgBatching::Batched,
        )
        .expect("denoise");
    mlx_rs::transforms::eval([&latents]).expect("eval latent");
    drop(heavy);
    clear_cache();
    latents
}

/// **The mechanism-level tile sweep** that decides which edges the ladder may publish.
///
/// Isolated from the request envelope on purpose: the *mechanism* column measures the decode against
/// the **exact untiled decode of the same latent**, which is the only way to see the deviation a
/// tile actually introduces. Driving it through [`Autoencoder::decode_tiled`] also reaches
/// geometries the production resolver refuses, which is how a rejected candidate gets a *number*
/// instead of an omission.
///
/// Before SC-19753 this sweep selected an overlap by per-crop GroupNorm drift. The layer-wise decoder
/// makes overlap policy identity only: every overlap at an edge must produce identical pixels, and
/// every swept cell must clear [`ms::DECODE_DRIFT_BAR`].
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn decode_tile_mechanism_sweep() {
    let dir = require_tier(DEFAULT_TIER);
    let vae = mlx_gen_sdxl::load_vae(&dir).expect("vae");
    let latent = production_latent(&dir, DEFAULT_TIER, 1024);

    let untiled = mlx_gen_sdxl::decode_image(&vae, &latent, None).expect("untiled decode");
    let mut best = (u32::MAX, 0u32, 0u32);
    println!(
        "[sc-15521 rung2 mechanism sweep {DEFAULT_TIER} 1024²] edge/overlap -> max Δ (mean Δ)"
    );
    for edge in ms::DECODE_TILE_EDGES_SWEPT {
        let mut line = format!("  edge {edge:>4}:");
        let mut overlap_reference = None;
        for overlap in ms::DECODE_OVERLAPS_SWEPT {
            let cfg = mlx_gen::tiling::TilingConfig::spatial_only(*edge as i32, *overlap as i32);
            let tiled = mlx_gen_sdxl::decode_image_tiled(&vae, &latent, None, Some(&cfg), None)
                .expect("tiled decode");
            let max = max_delta(&untiled.pixels, &tiled.pixels);
            let mean = mean_delta(&untiled.pixels, &tiled.pixels);
            line.push_str(&format!("  o{overlap:>3}: {max:>3} ({mean:.2})"));
            assert!(
                max <= ms::DECODE_DRIFT_BAR,
                "layer-wise decode exceeds the {}/255 bar at edge {edge} overlap {overlap}: {max}",
                ms::DECODE_DRIFT_BAR
            );
            if let Some(reference) = &overlap_reference {
                assert_eq!(
                    &tiled.pixels, reference,
                    "overlap is policy identity only; edge {edge} changed output at overlap {overlap}"
                );
            } else {
                overlap_reference = Some(tiled.pixels.clone());
            }
            if max < best.0 {
                best = (max, *edge, *overlap);
            }
        }
        println!("{line}");
    }
    let (best_max, best_edge, best_overlap) = best;
    println!(
        "[sc-15521 rung2 mechanism sweep] best cell: edge {best_edge} overlap {best_overlap} at \
         {best_max}/255 against a {}/255 bar",
        ms::DECODE_DRIFT_BAR
    );
    // The best cell is reported for continuity, but all cells above are now gated against the bar.
    assert!(
        best_max <= ms::DECODE_DRIFT_BAR,
        "the best decode geometry in the whole sweep no longer clears the sibling bar at 1024² \
         ({best_max}/255 at edge {best_edge} overlap {best_overlap}, bar {}/255). \
         layer-wise normalization is no longer preserving the production quality contract",
        ms::DECODE_DRIFT_BAR
    );
}

/// SC-19753 regression across the advertised output range using production latents. Whole-tail
/// tiling failed at the large end; full-activation GroupNorm semantics must now keep every row under
/// the 48/255 bar. Exact sealed coordinate policies decide production admission.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn layerwise_decode_clears_the_bar_across_the_advertised_output_range() {
    let dir = require_tier(DEFAULT_TIER);
    let vae = mlx_gen_sdxl::load_vae(&dir).expect("vae");
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(
        ms::DECODE_TILE_EDGE as i32,
        POLICY_DECODE_OVERLAP as i32,
    );
    let mut cleared: Vec<u32> = Vec::new();
    let mut worst = 0u32;
    for edge in [1024u32, 1280, 1536, 2048] {
        // A PRODUCTION latent at each geometry — rendered, not synthesised. A synthetic
        // unit-normal latent is a materially friendlier instrument here (it reads 35/255 at 1024²
        // where the rendered one reads 41), and a rung admitted on the friendly instrument would be
        // admitted on evidence no user ever sees.
        let latent = production_latent(&dir, DEFAULT_TIER, edge);
        let untiled = mlx_gen_sdxl::decode_image(&vae, &latent, None).expect("untiled");
        let tiled =
            mlx_gen_sdxl::decode_image_tiled(&vae, &latent, None, Some(&cfg), None).expect("tiled");
        let max = max_delta(&untiled.pixels, &tiled.pixels);
        worst = worst.max(max);
        println!(
            "[sc-15521 rung2 range {DEFAULT_TIER}] {edge}²: tile covers {:.1}% of the edge, max Δ \
             {max}/255 (bar {})",
            100.0 * f64::from(ms::DECODE_TILE_EDGE) / f64::from(edge),
            ms::DECODE_DRIFT_BAR
        );
        if max <= ms::DECODE_DRIFT_BAR {
            cleared.push(edge);
        }
        clear_cache();
    }
    println!(
        "[sc-15521 rung2 range {DEFAULT_TIER}] edge {} overlap {POLICY_DECODE_OVERLAP} clears the \
         {}/255 bar at {cleared:?}; worst cell {worst}/255",
        ms::DECODE_TILE_EDGE,
        ms::DECODE_DRIFT_BAR
    );
    assert_eq!(
        cleared,
        [1024_u32, 1280, 1536, 2048],
        "layer-wise decode must clear the {}/255 bar across the advertised range; worst={worst}",
        ms::DECODE_DRIFT_BAR
    );
}

/// Request-level memory and quality for layer-wise decode on a production latent.
///
/// [`decode_tile_mechanism_sweep`] measures the *isolated* decode — the right scope for a drift
/// comparison, the wrong scope for the saving, because a selector admits against the whole request.
/// `generate` cannot supply this row either (`memory_strategy::decode_tiling` refuses every
/// bounded-decode request), so it is assembled from the same public entry points as the rung-3 row,
/// under the same staged schedule, with the tiled decode substituted for the single-pass one.
///
/// The route-blind harness remains refused; sealed exact-coordinate policy adoption is tested by the
/// shared request-scope contract. This row proves the decoder that policy reaches both saves memory
/// and clears the quality bar.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn layerwise_decode_is_priced_at_the_request_level() {
    let dir = require_tier(DEFAULT_TIER);
    let cfg = mlx_gen::tiling::TilingConfig::spatial_only(
        ms::DECODE_TILE_EDGE as i32,
        POLICY_DECODE_OVERLAP as i32,
    );
    warm_up(&dir, DEFAULT_TIER);
    let plain = measure_end_to_end(&dir, DEFAULT_TIER, 1024, STEPS as usize, false, None);
    let tiled = measure_end_to_end(&dir, DEFAULT_TIER, 1024, STEPS as usize, false, Some(&cfg));
    let saved = 100.0 * (plain.peak_gib - tiled.peak_gib) / plain.peak_gib;
    let drift = max_delta(&plain.pixels, &tiled.pixels);
    println!(
        "[sc-15521 rung2 request-level {DEFAULT_TIER} 1024² {STEPS} steps] untiled {:.4} GiB -> \
         tiled {:.4} GiB ({saved:.2}%)  {:.0} -> {:.0} ms/step  max Δ {drift}/255  mean Δ {:.2}",
        plain.peak_gib,
        tiled.peak_gib,
        ms_per_step(&plain, STEPS),
        ms_per_step(&tiled, STEPS),
        mean_delta(&plain.pixels, &tiled.pixels),
    );
    // **The prize is real and this is the number `DECODE_SUPPORT` records.** A rung withheld with a
    // number tells the next story where to look; one withheld without a number tells it nothing —
    // and on this family the number is the largest single saving anywhere on the ladder, which is
    // exactly why the withholding argument had to be a strong one.
    assert!(
        tiled.peak_gib < plain.peak_gib * 0.97,
        "the tiled decode must bound the request peak by more than the 3% margin ({:.4} vs {:.4} \
         GiB); if it does not, the rung-2 write-up's recorded prize is wrong",
        tiled.peak_gib,
        plain.peak_gib
    );
    // The production latent must clear the same bar used by sealed coordinate admission.
    assert!(
        drift <= ms::DECODE_DRIFT_BAR,
        "the anchored geometry no longer preserves the production latent at 1024² ({drift}/255 \
         against a {}/255 bar)",
        ms::DECODE_DRIFT_BAR
    );
}

// ── Rung 3 ───────────────────────────────────────────────────────────────────────────────────────

/// **The rung-3 REQUEST-level measurement** — the first of the two independent reasons rung 3 is
/// declared `Missing`, and the one [`attention_chunking_is_measured_at_the_unet_seam`] cannot
/// supply.
///
/// The seam test answers "does chunking bound one U-Net forward?". This one answers the question the
/// contract actually turns on: **does it move the number a selector admits against, and does the
/// image survive?**
///
/// Both step counts are load-bearing:
///
/// * **one step** is the control that separates kernel rounding from a wiring bug — the schedule
///   cannot amplify anything across a single step, so whatever Δ appears there is the raw
///   per-forward divergence;
/// * **four steps** shows what that per-forward divergence becomes once the sampler has fed it back
///   into itself.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn attention_chunking_is_measured_against_the_rung_two_top() {
    let dir = require_tier(DEFAULT_TIER);
    warm_up(&dir, DEFAULT_TIER);
    let mut rows = Vec::new();
    for steps in [1_usize, STEPS as usize] {
        let plain = measure_end_to_end(&dir, DEFAULT_TIER, 1024, steps, false, None);
        let chunked = measure_end_to_end(&dir, DEFAULT_TIER, 1024, steps, true, None);
        let delta_pct = 100.0 * (chunked.peak_gib - plain.peak_gib) / plain.peak_gib;
        println!(
            "[sc-15521 rung3 end-to-end {DEFAULT_TIER} 1024² {steps} step(s)] unchunked {:.4} GiB \
             -> chunked {:.4} GiB ({delta_pct:+.2}%)  max Δ {}/255  mean Δ {:.2}",
            plain.peak_gib,
            chunked.peak_gib,
            max_delta(&plain.pixels, &chunked.pixels),
            mean_delta(&plain.pixels, &chunked.pixels),
        );
        rows.push((steps, plain, chunked));
    }

    // Claim 1: the rung does not pay for itself at the request level. Asserted with the SAME 3%
    // margin the implemented rungs must CLEAR, in the opposite direction — so a future change that
    // makes chunking actually bound a request reddens here and forces the `Missing` declaration to
    // be revisited rather than letting it calcify.
    for (steps, plain, chunked) in &rows {
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
    let (_, plain, chunked) = &rows[0];
    assert!(
        max_delta(&plain.pixels, &chunked.pixels) > 0,
        "chunked and unchunked agreed exactly at one step — if MLX now dispatches the same kernel at \
         every query-block size, the output-preservation half of ATTENTION_SUPPORT no longer holds \
         and the rung must be re-measured, not left Missing on a stale reason"
    );
}

/// **The rung-3 mechanism measurement.**
///
/// Driven at the U-Net level rather than through `generate`, for the same reason the rung-2 sweep
/// is: the production resolver *refuses* a bounded-attention request, so the only way to measure a
/// rung this family does not ship is the mechanism seam.
/// [`UNet2DConditionModel::forward_planned`] takes the plan raw.
///
/// The Kolors-specific shape matters: the cross-attention memory is the **ChatGLM3 context**, 256
/// tokens at 4096 dims projected to 2048, against SDXL's 77×2048. A longer key axis is exactly where
/// a bounded-attention rung would be most likely to pay, which is why this is measured here rather
/// than inherited.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn attention_chunking_is_measured_at_the_unet_seam() {
    let dir = require_tier(DEFAULT_TIER);
    let unet =
        mlx_gen_sdxl::load_unet_kolors_dtype(&dir, mlx_rs::Dtype::Float16).expect("load unet");
    let key = mlx_rs::random::key(7).unwrap();
    let f16 = |a: mlx_rs::Array| a.as_dtype(mlx_rs::Dtype::Float16).unwrap();
    // The production CFG batch at 1024²: latent [2, 128, 128, 4], ChatGLM3 context [2, 256, 4096]
    // (the U-Net's auto-detected `encoder_hid_proj` projects it to 2048), pooled [2, 4096].
    let x = f16(mlx_rs::random::normal::<f32>(&[2, 128, 128, 4], None, None, Some(&key)).unwrap());
    let ctx = f16(mlx_rs::random::normal::<f32>(&[2, 256, 4096], None, None, Some(&key)).unwrap());
    let pooled = f16(mlx_rs::random::normal::<f32>(&[2, 4096], None, None, Some(&key)).unwrap());
    let time_ids = mlx_gen_kolors::model::kolors_time_ids(2, 1024, 1024);

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

    let bounded_plan = || {
        mlx_gen_sdxl::SdxlForwardPlan::with_attention(mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::CONSTRAINED,
        ))
    };

    // **The discarded warm-up row, and it is the difference between a number and an artifact.**
    //
    // This test drives the seam directly rather than through [`measure`], so it does not go through
    // [`warm_up`] — and for one revision it therefore had no warm-up at all, which made it the only
    // peak-publishing test in this file without one. The consequence was not subtle: the FIRST `run`
    // in this process reads **~4.6–4.8 GiB whichever plan it executes** (it is also the only row
    // here that does NOT reproduce run to run — measured 4.6104 / 4.6104 / 4.6600), against a
    // rock-steady **5.3652** for either plan once the allocator is warm. Measuring
    // unbounded-then-bounded cold therefore produced a **+12.54%** "seam regression" that was
    // entirely the cold-allocator bias — the two plans read the SAME peak when both are warm. That
    // fabricated figure reached `ATTENTION_SUPPORT`, two user-facing refusal strings and the
    // Shortcut record before it was caught.
    //
    // The warm-up runs the BOUNDED plan on purpose: it is the plan with the larger allocation set
    // (the per-chunk transients), so a discarded bounded row leaves the allocator warm for both
    // published rows rather than only for the one whose shape it happened to match.
    let (warm_up_peak, _) = run(bounded_plan());

    let (unbounded_peak, unbounded) = run(mlx_gen_sdxl::SdxlForwardPlan::UNBOUNDED);
    let (bounded_peak, bounded) = run(bounded_plan());
    let max_abs = unbounded
        .iter()
        .zip(&bounded)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let delta_pct = 100.0 * (bounded_peak - unbounded_peak) / unbounded_peak;
    println!(
        "[sc-15521 rung3 unet-seam {DEFAULT_TIER} 1024² CFG] discarded warm-up {warm_up_peak:.4} \
         GiB; unbounded {unbounded_peak:.4} GiB -> bounded {bounded_peak:.4} GiB \
         ({delta_pct:+.2}%)  max|Δeps| {max_abs:.3e}"
    );
    // **Asserted in BOTH directions, at 1%.** The measured movement is 0.00% — the two plans read
    // the same peak to the millibyte once the allocator is warm — so a one-sided 3% floor (which is
    // what this test carried while it was measuring the cold-allocator artifact) could not tell a
    // real seam change from no change at all: it passed on the fabricated +12.54% and would pass on
    // a genuine +2.9% regression too.
    //
    // Either direction is a publishable change and both must redden here:
    //   * BELOW  — chunking has started bounding the seam, and rung 3's first `Missing` reason
    //              ("the mechanism does not bound this family") no longer holds;
    //   * ABOVE  — chunking has started ADDING transients, which is a *new* finding and not the one
    //              `ATTENTION_SUPPORT` currently records (it records no movement at all).
    assert!(
        delta_pct.abs() < 1.0,
        "bounded attention moved the U-Net seam peak by {delta_pct:+.2}% ({bounded_peak:.4} vs \
         {unbounded_peak:.4} GiB). memory_strategy::ATTENTION_SUPPORT records that the two plans \
         peak IDENTICALLY at this seam — re-measure and re-publish it rather than widening this \
         margin"
    );
}

// ── Rung 4 ───────────────────────────────────────────────────────────────────────────────────────

/// **The rung-4 cadence sweep** — allocation and output evidence for every published cadence.
///
/// Two deterministic facts the product claim rests on:
///
/// 1. **every cadence bounds the peak** (each row below the staged control by a 3% margin);
/// 2. **every cadence preserves the output exactly**. The default row also asserts flat peak
///    accounting; a timing sample is printed only as diagnostic context.
///
/// `KOLORS_WINDOW_PROBE_TIER` / `KOLORS_WINDOW_PROBE_SIZE` re-run the sweep at another tier or
/// output edge. In probe mode flatness is reported rather than asserted. What stays asserted in
/// BOTH modes is what is true of every configuration: each cadence bounds the request peak below
/// the control, and every row is byte-identical to it.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn transformer_window_sweep_and_streamed_output_identity() {
    let tier = std::env::var("KOLORS_WINDOW_PROBE_TIER").unwrap_or_else(|_| DEFAULT_TIER.into());
    let edge: u32 = std::env::var("KOLORS_WINDOW_PROBE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let probe = tier != DEFAULT_TIER || edge != 1024;
    let dir = require_tier(&tier);
    warm_up(&dir, &tier);

    let control = measure(
        &dir,
        &tier,
        LoadShape::DeferredMaterialization,
        &request(Some(staged()), edge, STEPS),
    );
    println!(
        "[sc-15521 rung4 sweep {tier} {edge}² {STEPS} steps] staged control {:.4} GiB  {:.0} ms/step",
        control.peak_gib,
        ms_per_step(&control, STEPS)
    );

    let mut rows = Vec::new();
    for window in ms::TRANSFORMER_WINDOW_SIZES {
        let row = measure(
            &dir,
            &tier,
            LoadShape::DeferredMaterialization,
            &request(Some(full_ladder(*window)), edge, STEPS),
        );
        println!(
            "  cadence {window:>2}: {:.4} GiB ({:+.2}%)  {:.0} ms/step ({:.1}× the control)  max Δ {}",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib,
            ms_per_step(&row, STEPS),
            row.wall.as_secs_f64() / control.wall.as_secs_f64(),
            max_delta(&control.pixels, &row.pixels),
        );
        rows.push((*window, row));
    }

    // The default `Dit` cell is peak-bearing and must reduce the request peak. A probe can target a
    // conditioning-bearing cell where a correct `Dit` stream does not move the whole-request peak;
    // it still grades accounting and output identity, but reports rather than asserts that reduction.
    // Scope note: every `LoadSpec` in this sweep carries `ip_adapter: None` and no adapters, so
    // `ip_expected` is false and the `BlockAdapters` are empty — this loop discriminates the TIER
    // replay and nothing about the IP/adapter replay. That half is carried end to end by
    // `the_rung_four_stream_replays_an_installed_ip_adapter`, and at the unit level by
    // `mlx_gen_sdxl::block_stream`'s own tests.
    for (window, row) in &rows {
        assert_rung_four_evidence(&control, row, *window, !probe);
    }

    let tightest = &rows.first().expect("a non-empty domain").1;
    let widest = &rows.last().expect("a non-empty domain").1;
    let spread = 100.0 * (widest.peak_gib - tightest.peak_gib).abs() / tightest.peak_gib;
    println!(
        "[sc-15521 rung4 sweep {tier} {edge}²] peak spread across the domain {spread:.2}%{}",
        if probe {
            " (probe mode: reported, not asserted)"
        } else {
            ""
        }
    );
    if probe {
        return;
    }
    assert!(
        spread < 1.0,
        "the published cadences no longer bound the peak to the same value ({spread:.2}% spread). \
         TRANSFORMER_WINDOW_SIZES' flat column and the phase-separation mechanism it rests on must \
         be re-derived before the domain is published again"
    );
}

/// **Rung 4 with an IP-Adapter actually installed** — the end-to-end half of the replay guard.
///
/// Rung 4 is deliberately permissive for IP and control (`memory_strategy`'s route gate refuses only
/// the geometry, not the mode), so an **IP-armed rung-4 render is reachable in production**. The
/// mechanism that makes it correct is `mlx_gen_sdxl::block_stream`'s replay: each `Transformer2D`'s
/// stream captures the installed K/V projections off its FINISHED resident blocks (`registry`'s load
/// arms the streams LAST, after the IP install), and a re-materialized block that comes back without
/// its pair is refused loudly rather than rendering a plausible, wrong image.
///
/// That mechanism is unit-gated in `block_stream`, but until this test existed nothing exercised it
/// through `generate` on real weights: every `LoadSpec` in this file carried `ip_adapter: None`, so
/// the sweep's byte-identity loop ran with `ip_expected == false` and empty `BlockAdapters` and
/// discriminated nothing about replay. A capture that silently came back empty on the production
/// path would have been invisible here.
///
/// One row at the default tier, at the advertised `min_size` to keep it cheap: staged control vs the
/// shipped cadence, byte-identical. Byte-identity is the whole assertion, and it is a strong one —
/// dropping the image prompt for even one sub-stack's layers changes the image.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot + a converted Kolors-IP-Adapter-Plus \
            snapshot (set KOLORS_LADDER_ROOT and KOLORS_IP_ADAPTER)"]
fn the_rung_four_stream_replays_an_installed_ip_adapter() {
    const EDGE: u32 = 512;
    let dir = require_tier(DEFAULT_TIER);
    let ip_root = require_ip_adapter();
    warm_up(&dir, DEFAULT_TIER);

    let ip_spec = |shape: LoadShape| {
        spec(&dir, DEFAULT_TIER, shape).with_ip_adapter(WeightsSource::Dir(ip_root.clone()))
    };
    // In IP mode the Reference IS the image prompt, and the model refuses the request without one.
    let ip_request = |memory: GenerationMemory| {
        let mut req = request(Some(memory), EDGE, STEPS);
        req.conditioning = vec![mlx_gen::gen_core::Conditioning::Reference {
            image: ip_reference_image(),
            strength: Some(0.6),
        }];
        req
    };

    let render = |memory: GenerationMemory| -> Row {
        let registry = mlx_gen_kolors::provider_registry().expect("provider registry");
        let model = registry
            .load("kolors", &ip_spec(LoadShape::DeferredMaterialization))
            .expect("load kolors with an IP-Adapter");
        clear_cache();
        reset_peak_memory();
        let started = std::time::Instant::now();
        let out = model
            .generate(&ip_request(memory), &mut |_: Progress| {})
            .expect("an IP-armed generate must succeed");
        let peak = get_peak_memory();
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
    };

    let control = render(staged());
    let windowed = render(full_ladder(ms::TRANSFORMER_WINDOW_SIZE));
    println!(
        "[sc-15521 rung4 IP replay {DEFAULT_TIER} {EDGE}² {STEPS} steps] staged control {:.4} GiB \
         -> windowed {:.4} GiB ({:+.2}%)  {:.0} -> {:.0} ms/step  max Δ {}",
        control.peak_gib,
        windowed.peak_gib,
        100.0 * (windowed.peak_gib - control.peak_gib) / control.peak_gib,
        ms_per_step(&control, STEPS),
        ms_per_step(&windowed, STEPS),
        max_delta(&control.pixels, &windowed.pixels),
    );
    assert_eq!(
        control.pixels, windowed.pixels,
        "a streamed block did not reproduce its resident twin WITH an IP-Adapter installed. The \
         replay captures each block's K/V pair off the finished resident stack, so a divergence \
         here means the streamed cross-attention ran without the image prompt for some layers — a \
         plausible, wrong image. See mlx_gen_sdxl::block_stream's replay guard"
    );
}

/// **The rung-4 saving is the whole `transformer_blocks` weight set**, cross-checked against the
/// snapshot's own safetensors headers rather than against a doc comment.
///
/// This is the arithmetic that identifies the *mechanism*, and it is the check that caught SDXL's
/// misattributed one. Eleven `Transformer2D` sub-stacks run in sequence and `run_windowed` releases
/// one before opening the next, so at most one cadence-worth of blocks is materialized at any
/// instant. **If the peak occurred during the windowed forward**, the saving could be at most
/// `block set − w × (one deep block)`. A measured saving that exceeds that bound proves zero window
/// weights are resident at the peak moment — i.e. the peak is in another phase entirely (the
/// decode), and cadence is invisible to it.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_rung_four_saving_is_the_whole_transformer_block_weight_set() {
    let dir = require_tier(DEFAULT_TIER);
    warm_up(&dir, DEFAULT_TIER);
    let (total, blocks, deepest_block) = unet_weight_arithmetic(&dir);
    println!(
        "[sc-15521 rung4 arithmetic {DEFAULT_TIER}] U-Net {:.4} GiB, transformer_blocks {:.4} GiB, \
         one deep block {:.4} GiB",
        total / GIB,
        blocks / GIB,
        deepest_block / GIB
    );

    let control = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(staged()), 1024, STEPS),
    );
    let windowed = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 1024, STEPS),
    );
    let saving = (control.peak_gib - windowed.peak_gib) * GIB;
    let block_set = blocks;
    let forward_bound = block_set - f64::from(ms::TRANSFORMER_WINDOW_SIZE) * deepest_block;
    println!(
        "[sc-15521 rung4 arithmetic {DEFAULT_TIER} 1024²] measured saving {:.4} GiB = {:.3}× the \
         block set; the windowed-forward bound at cadence {} is {:.4} GiB",
        saving / GIB,
        saving / block_set,
        ms::TRANSFORMER_WINDOW_SIZE,
        forward_bound / GIB
    );
    assert!(
        saving > forward_bound,
        "the measured saving ({:.4} GiB) is inside the bound a peak-bearing windowed forward could \
         produce ({:.4} GiB). The phase-separation mechanism in TRANSFORMER_WINDOW_SIZES claims the \
         peak is the DECODE, and that claim rests on this inequality",
        saving / GIB,
        forward_bound / GIB
    );
    assert!(
        saving < block_set * 1.15,
        "the measured saving ({:.4} GiB) exceeds the whole transformer_blocks weight set ({:.4} \
         GiB) by more than measurement slack — rung 4 cannot bound more than it holds, so the row \
         is measuring something else",
        saving / GIB,
        block_set / GIB
    );
}

/// Sum the U-Net snapshot's `data_offsets`: (whole file, `transformer_blocks.*`, one deep block).
fn unet_weight_arithmetic(dir: &std::path::Path) -> (f64, f64, f64) {
    let file = mlx_gen_sdxl::resolve_unet_weight_file(dir, mlx_rs::Dtype::Float16)
        .expect("resolve unet weight file");
    let bytes = std::fs::read(&file).expect("read unet safetensors");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header");
    let map = header.as_object().expect("header object");
    let (mut total, mut blocks) = (0f64, 0f64);
    let mut per_block: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut stack_depths: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for (key, value) in map {
        if key == "__metadata__" {
            continue;
        }
        let offsets = value["data_offsets"].as_array().expect("data_offsets");
        let size = offsets[1].as_f64().unwrap() - offsets[0].as_f64().unwrap();
        total += size;
        if let Some((prefix, rest)) = key.split_once(".transformer_blocks.") {
            blocks += size;
            let index = rest.split('.').next().unwrap_or("0");
            *per_block.entry(format!("{prefix}.{index}")).or_default() += size;
            stack_depths
                .entry(prefix.to_owned())
                .or_default()
                .insert(index.to_owned());
        }
    }
    // "One deep block" is a block of a 10-deep sub-stack — the depth the widest published cadence
    // addresses. Taking a bare max over all blocks would pick the same value here, but a bare max
    // would silently follow a topology change to a different stack.
    let deep_prefix = stack_depths
        .iter()
        .find(|(_, indices)| indices.len() == 10)
        .map(|(prefix, _)| prefix.clone())
        .expect("a 10-deep Transformer2D sub-stack");
    let deepest_block = *per_block
        .get(&format!("{deep_prefix}.0"))
        .expect("block 0 of a deep sub-stack");
    (total, blocks, deepest_block)
}

/// **The cadence-flatness CONDITION, checked rather than assumed.**
///
/// `TRANSFORMER_WINDOW_SIZES` explains its flat peak column by phase separation: flatness holds only
/// while `decode transient + resident remainder` exceeds the windowed forward's own peak at the
/// widest published cadence. On SDXL that inequality **reverses** at its advertised `min_size`, and
/// the flat column stops being flat.
///
/// A test that only re-measured the flat column would report the *consequence* and never the
/// condition, so a family that happened to stay flat for a different reason would look identical.
/// This measures the two sides separately, at the advertised `min_size` where the inequality is
/// tightest, and asserts the ordering.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_cadence_flatness_condition_is_checked_not_assumed() {
    let dir = require_tier(DEFAULT_TIER);
    // 512² is `descriptor().capabilities.min_size` — the smallest advertised output, where the
    // decode transient is smallest and the inequality is therefore hardest to satisfy.
    const MIN_EDGE: u32 = 512;
    warm_up(&dir, DEFAULT_TIER);
    let widest = *ms::TRANSFORMER_WINDOW_SIZES
        .last()
        .expect("a non-empty domain");
    let tightest = ms::TRANSFORMER_WINDOW_SIZES[0];

    // The staged control runs FIRST and is not merely context: the first `measure` in a process
    // reads a peak biased by MLX's cold allocator. An earlier revision of this test measured the
    // widest cadence first and read 4.4632 GiB for a row `transformer_window_sweep_and_streamed_
    // output_identity` reads as 4.6924 GiB in the same configuration — a 4.9% phantom spread that
    // looked exactly like the flat region breaking. Every peak this file publishes therefore has a
    // row ahead of it in its own process, and this comment is the reason.
    let control = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(staged()), MIN_EDGE, STEPS),
    );
    let at_tightest = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(tightest)), MIN_EDGE, STEPS),
    );
    let at_widest = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(widest)), MIN_EDGE, STEPS),
    );
    let spread = 100.0 * (at_widest.peak_gib - at_tightest.peak_gib).abs() / at_tightest.peak_gib;
    println!(
        "[sc-15521 rung4 flatness condition {DEFAULT_TIER} {MIN_EDGE}² (advertised min_size)] \
         staged control {:.4} GiB, cadence {tightest} {:.4} GiB, cadence {widest} {:.4} GiB — \
         spread {spread:.2}%",
        control.peak_gib, at_tightest.peak_gib, at_widest.peak_gib
    );
    assert!(
        at_tightest.peak_gib < control.peak_gib * 0.97,
        "rung 4 must still bound the request peak at the advertised min_size ({:.4} vs {:.4} GiB)",
        at_tightest.peak_gib,
        control.peak_gib
    );
    assert_eq!(
        at_tightest.pixels, at_widest.pixels,
        "both cadences must render the same image at the advertised min_size"
    );
    // The condition, read off the two sides: if the decode still dominates at the SMALLEST
    // advertised output then it dominates everywhere larger, because the decode transient grows
    // with area while the windowed forward's peak is geometry-insensitive. If this reddens, the
    // flat column in TRANSFORMER_WINDOW_SIZES is a coincidence at the measured points rather than a
    // property, and the default cadence's justification changes.
    assert!(
        spread < 2.0,
        "the cadence-flatness condition no longer holds at the advertised min_size \
         ({spread:.2}% spread between cadence {tightest} and cadence {widest}). \
         TRANSFORMER_WINDOW_SIZES documents this as the boundary of the flat region — record the \
         new boundary rather than leaving the prose describing a region that has moved"
    );
}

/// **The `TransformerComponent::TextEncoder` scope question, measured — the one place SDXL's answer
/// could not have been reused.**
///
/// SDXL declares `Dit` only, because its CLIP pair is shed by rung 1 and could never carry the
/// request peak. Kolors' tower is ChatGLM3-6B — larger than the U-Net at every tier — so "rung 1
/// already releases it" is not the end of the argument: rung 1 releases it before the *denoise*, but
/// the conditioning phase itself still has to hold it, and if that phase were the peak-bearing one a
/// text-encoder window would move the request peak.
///
/// This measures both phases separately at every advertised geometry corner and reports which one
/// carries the peak. The assertion is deliberately narrow: it pins the fact
/// `TRANSFORMER_WINDOW_COMPONENTS` cites — that the conditioning phase is **not** peak-bearing at
/// the geometry the catalog defaults to — so a change that made it peak-bearing there forces the
/// scope decision to be revisited rather than inherited.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_text_encoder_window_scope_cannot_move_the_request_peak() {
    let mut bearing: Vec<(String, u32)> = Vec::new();
    let mut default_cell = None;
    let mut measured = 0;
    for tier in TIERS {
        let Some(dir) = tier_dir(tier) else {
            println!("SKIPPED-BY-ABSENCE: tier {tier} is not cached under {ROOT_ENV}");
            continue;
        };
        warm_up(&dir, tier);
        for edge in [512u32, 1024, 2048] {
            let (row, conditioning) = measure_end_to_end_phased(&dir, tier, edge, 1, false, None);
            // The whole-request peak is the max over phases, so the conditioning phase is the
            // peak-bearing one exactly when the request peak never rose above it.
            let conditioning_bearing = conditioning >= row.peak_gib;
            println!(
                "[sc-15521 rung4 TextEncoder scope {tier} {edge}²] conditioning phase {conditioning:.4} \
                 GiB, request {:.4} GiB -> peak-bearing phase: {}",
                row.peak_gib,
                if conditioning_bearing { "conditioning" } else { "denoise + decode" }
            );
            if conditioning_bearing {
                bearing.push(((*tier).to_owned(), edge));
            }
            if *tier == DEFAULT_TIER && edge == 1024 {
                default_cell = Some(conditioning_bearing);
            }
            measured += 1;
            clear_cache();
        }
    }
    // The nine-cell grid is the claim; a subset of it is not. `bearing` is asserted below to contain
    // only 512² cells, and that shape argument is vacuous if the 1024²/2048² cells of a missing tier
    // were never measured.
    assert_eq!(
        measured,
        TIERS.len() * 3,
        "SKIPPED-BY-ABSENCE: only {measured} of the {} advertised (tier × geometry) cells were \
         measured under {ROOT_ENV}; the scope finding is a claim about the whole grid",
        TIERS.len() * 3
    );
    println!(
        "[sc-15521 rung4 TextEncoder scope] the conditioning phase carries the request peak at \
         {bearing:?} of the advertised (tier × geometry) range"
    );
    assert_eq!(
        default_cell,
        Some(false),
        "the conditioning phase now carries the request peak at the catalog's DEFAULT tier and \
         geometry, so a TextEncoder-scoped window would move the request peak for the median \
         caller. TRANSFORMER_WINDOW_COMPONENTS declares `Dit` only, and its recorded reason is that \
         the scope pays at the small-output corner and nowhere else — re-open it"
    );
    assert!(
        ms::TRANSFORMER_WINDOW_COMPONENTS.contains(&TransformerComponent::TextEncoder),
        "the conditioning phase carries the request peak at {bearing:?}, so the TextEncoder scope \
         must be published and implemented rather than declared away"
    );
    assert_eq!(
        ms::TRANSFORMER_WINDOW_COMPONENT,
        TransformerComponent::Dit,
        "the DEFAULT scope must be the one that pays at the catalog's default tier and geometry"
    );
    // Pin the SHAPE of the finding, not just the default cell: the scope's value is confined to the
    // small-output corner, which is what makes `Dit`-only defensible. A cell at 1024² or above
    // turning conditioning-bearing changes that argument.
    assert!(
        bearing.iter().all(|(_, edge)| *edge <= 512),
        "the conditioning phase now carries the request peak at an output above the advertised \
         min_size ({bearing:?}) — TRANSFORMER_WINDOW_COMPONENTS' recorded reason no longer holds"
    );
}

/// **The published cadence domain is enforced AND reachable on the production path.**
///
/// Two directions, and the second is the one a declaration-only test misses: an out-of-domain
/// cadence must be refused by `generate` itself (`WINDOW-REQUEST size=N admitted=false
/// refused=true`), and every in-domain cadence must actually execute a render.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_published_window_domain_is_enforced_and_reachable_on_the_production_path() {
    let dir = require_tier(DEFAULT_TIER);
    let registry = mlx_gen_kolors::provider_registry().expect("provider registry");
    let model = registry
        .load(
            "kolors",
            &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
        )
        .expect("load kolors");

    for size in ms::TRANSFORMER_WINDOW_SIZES {
        let out = model.generate(
            &request(Some(full_ladder(*size)), 512, 1),
            &mut |_: Progress| {},
        );
        println!(
            "WINDOW-REQUEST size={size} admitted={} refused={}",
            out.is_ok(),
            out.is_err()
        );
        assert!(
            out.is_ok(),
            "published cadence {size} must be reachable on the production path: {:?}",
            out.err()
        );
    }
    for bad in [0_u32, 3, 4, 6, 7, 9, 11, 28, 70] {
        let out = model.generate(
            &request(Some(full_ladder(bad)), 512, 1),
            &mut |_: Progress| {},
        );
        println!(
            "WINDOW-REQUEST size={bad} admitted={} refused={}",
            out.is_ok(),
            out.is_err()
        );
        assert!(
            out.is_err(),
            "cadence {bad} is outside the published domain and must be refused by the PRODUCTION \
             path, not clamped to the nearest legal value"
        );
    }

    // Every PUBLISHED scope is reachable on the production path and none is silently narrowed.
    // A scope that resolved to `Dit` behind the caller's back would render identically to the `Dit`
    // row, which is precisely what the byte-identity check below cannot distinguish — so the peak
    // rows in `the_text_encoder_window_bounds_the_conditioning_bearing_cell` carry that half.
    for component in ms::TRANSFORMER_WINDOW_COMPONENTS {
        let out = model.generate(
            &request(
                Some(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, *component)),
                512,
                1,
            ),
            &mut |_: Progress| {},
        );
        println!(
            "WINDOW-REQUEST component={component:?} admitted={} refused={}",
            out.is_ok(),
            out.is_err()
        );
        assert!(
            out.is_ok(),
            "published scope {component:?} must be reachable on the production path: {:?}",
            out.err()
        );
    }
}

/// **Rung 4's preconditions fail closed on real weights.**
///
/// Three shapes, each of which would otherwise produce a wrong render or a meaningless one:
///
/// 1. an **eager** load — its blocks are already committed, so a window adds a copy rather than
///    bounding anything;
/// 2. rung 4 **without rung 1** — the ChatGLM3-6B tower stays resident through the denoise and the
///    request peak does not move;
/// 3. a **load-time quantization over a dense snapshot** — every block packs at load, so a window
///    over the materialized trunk bounds nothing. This is the shape the upstream
///    `Kwai-Kolors/Kolors-diffusers` snapshot has, and it is the tier-level discriminator on this
///    family.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn rung_four_preconditions_fail_closed_on_real_weights() {
    let dir = require_tier(DEFAULT_TIER);
    let registry = mlx_gen_kolors::provider_registry().expect("provider registry");

    // 1. Eager load.
    let eager = registry
        .load(
            "kolors",
            &spec(&dir, DEFAULT_TIER, LoadShape::EagerMaterialization),
        )
        .expect("load kolors eager");
    let err = eager
        .generate(
            &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, 1),
            &mut |_: Progress| {},
        )
        .expect_err("an eager load must refuse rung 4");
    assert!(
        err.to_string().contains("cannot stream its blocks"),
        "got: {err}"
    );

    // 2. Rung 4 without rung 1.
    let deferred = registry
        .load(
            "kolors",
            &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
        )
        .expect("load kolors deferred");
    let unstaged = GenerationMemory {
        stage_residency: false,
        ..full_ladder(ms::TRANSFORMER_WINDOW_SIZE)
    };
    let err = deferred
        .generate(&request(Some(unstaged), 512, 1), &mut |_: Progress| {})
        .expect_err("rung 4 without rung 1 must be refused");
    assert!(err.to_string().contains("staged residency"), "got: {err}");

    // 3. Load-time quantization over a DENSE snapshot. The bf16 tier is dense, so requesting a
    //    packed tier over it is exactly the shape `load_leaves_blocks_lazy` refuses.
    //
    // `require_tier`, not a `println!` + early return: this arm is the tier-level discriminator on
    // this family, and skipping it while still reporting the test as PASSED is exactly what the
    // module docs forbid ("a test whose tier is absent skips loudly by name rather than passing
    // silently"). An absent bf16/ must fail the test by name, not shrink it to two arms.
    let dense = require_tier("bf16");
    let mut dense_q8 = spec(&dense, "bf16", LoadShape::DeferredMaterialization);
    dense_q8.quantize = Some(Quant::Q8);
    assert!(
        !ms::streamable(&dense_q8),
        "a load-time quantization over the dense bf16 tier must not arm rung 4"
    );
    assert!(!ms::load_leaves_blocks_lazy(&dense_q8));
}

/// **Kolors' U-Net has eleven windowable sub-stacks at two depths — measured on the snapshot, not
/// read off the config.**
///
/// A config/checkpoint disagreement would otherwise surface as a window that silently covers the
/// wrong number of blocks. This reads the real key set.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_unet_has_eleven_windowable_sub_stacks_at_two_depths() {
    let dir = require_tier(DEFAULT_TIER);
    let file = mlx_gen_sdxl::resolve_unet_weight_file(&dir, mlx_rs::Dtype::Float16)
        .expect("resolve unet weight file");
    let bytes = std::fs::read(&file).expect("read unet safetensors");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header");
    let mut stacks: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for key in header.as_object().expect("header").keys() {
        if let Some((prefix, rest)) = key.split_once(".transformer_blocks.") {
            stacks
                .entry(prefix.to_owned())
                .or_default()
                .insert(rest.split('.').next().unwrap_or("0").to_owned());
        }
    }
    let mut depths: Vec<usize> = stacks.values().map(|v| v.len()).collect();
    depths.sort_unstable();
    let total: usize = depths.iter().sum();
    println!(
        "[sc-15521 rung4 topology {DEFAULT_TIER}] {} Transformer2D sub-stacks, depths {depths:?}, \
         {total} windowable blocks",
        stacks.len()
    );
    assert_eq!(stacks.len(), 11, "eleven Transformer2D sub-stacks");
    assert_eq!(depths, vec![2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10]);
    assert_eq!(total, 70, "70 windowable TransformerBlocks");
    assert_eq!(
        *depths.last().unwrap() as u32,
        *ms::TRANSFORMER_WINDOW_SIZES.last().unwrap(),
        "the widest published cadence must equal the deepest sub-stack"
    );
}

/// **Every advertised tier loads and publishes the ladder it actually implements.**
///
/// The epic's rule is that no cell becomes Verified by sharing code, and the failure mode this
/// closes is a per-tier test that records a peak and never checks *which* ladder each tier
/// published. Both are checked here: the rung-4 declaration per tier, and a measured rung-1 saving
/// per tier so a tier whose numbers diverge cannot hide behind the default tier's.
#[test]
#[ignore = "needs the real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn every_advertised_tier_loads_and_publishes_the_ladder() {
    let mut measured = 0;
    for tier in TIERS {
        let Some(dir) = tier_dir(tier) else {
            println!("SKIPPED-BY-ABSENCE: tier {tier} is not cached under {ROOT_ENV}");
            continue;
        };
        warm_up(&dir, tier);
        let load_spec = spec(&dir, tier, LoadShape::DeferredMaterialization);
        let contract =
            ms::memory_strategy_contract("kolors", &load_spec).expect("contract at this tier");
        let rung4 = contract
            .capability(mlx_gen::gen_core::MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4 capability")
            .support
            .clone();
        let staged_row = measure(
            &dir,
            tier,
            LoadShape::DeferredMaterialization,
            &request(Some(staged()), 1024, STEPS),
        );
        let windowed = measure(
            &dir,
            tier,
            LoadShape::DeferredMaterialization,
            &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 1024, STEPS),
        );
        println!(
            "[sc-15521 per-tier {tier} 1024² {STEPS} steps] rung4 {rung4:?}  staged {:.4} GiB -> \
             windowed {:.4} GiB ({:+.2}%)  conditioning {:.4} GiB / transformer {:.4} GiB / \
             decoder {:.4} GiB",
            staged_row.peak_gib,
            windowed.peak_gib,
            100.0 * (windowed.peak_gib - staged_row.peak_gib) / staged_row.peak_gib,
            contract.asset_facts.conditioning_bytes as f64 / GIB,
            contract.asset_facts.transformer_bytes as f64 / GIB,
            contract.asset_facts.decoder_bytes as f64 / GIB,
        );
        assert_eq!(
            rung4,
            mlx_gen::gen_core::MemoryStrategySupport::Implemented,
            "tier {tier} must publish rung 4 — all three shipped tiers are re-openable"
        );
        assert!(
            windowed.peak_gib < staged_row.peak_gib * 0.97,
            "tier {tier}: rung 4 must bound the request peak by more than the 3% margin"
        );
        assert_eq!(
            staged_row.pixels, windowed.pixels,
            "tier {tier}: the streamed render must be byte-identical to the resident one"
        );
        measured += 1;
    }
    // **Every** advertised tier, not "at least one". A `> 0` guard let a test whose whole promise is
    // per-tier evidence pass having checked the default tier and skipped the other two — which is
    // precisely the failure mode its own doc comment says it exists to close ("no cell becomes
    // Verified by sharing code").
    assert_eq!(
        measured,
        TIERS.len(),
        "SKIPPED-BY-ABSENCE: only {measured} of the {} advertised tiers were cached under \
         {ROOT_ENV}; this test's claim is per-tier and cannot be made from a subset",
        TIERS.len()
    );
}

/// **The whole ladder renders under a memory cap that the unwindowed composition does not fit.**
///
/// This is the end the rung exists for: on a host where the resident composition does not fit, even
/// a 3× wall-clock cost is the difference between a render and no render.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_full_ladder_renders_under_a_memory_cap() {
    let dir = require_tier(DEFAULT_TIER);
    warm_up(&dir, DEFAULT_TIER);
    // The cap is set between the measured staged peak and the measured windowed peak at 512², so it
    // is a real discriminator rather than a formality.
    let control = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(staged()), 512, STEPS),
    );
    let windowed = measure(
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, STEPS),
    );
    let cap_gb = (windowed.peak_gib + control.peak_gib) / 2.0 * (GIB / 1e9);
    println!(
        "[sc-15521 full ladder under a cap {DEFAULT_TIER} 512²] staged {:.4} GiB, windowed {:.4} \
         GiB, cap {cap_gb:.2} GB",
        control.peak_gib, windowed.peak_gib
    );
    // SAFETY: single-threaded by construction (`--test-threads=1` in the module docs) and the var is
    // cleared before the function returns.
    unsafe { std::env::set_var(MEMORY_CAP_ENV, format!("{cap_gb:.2}")) };
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        measure(
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &request(Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)), 512, STEPS),
        )
    }));
    unsafe { std::env::remove_var(MEMORY_CAP_ENV) };
    let capped = out.expect("the full ladder must render under a cap the staged row exceeds");
    assert_eq!(
        capped.pixels, windowed.pixels,
        "the capped render must be byte-identical to the uncapped windowed one"
    );
}

/// **The `TextEncoder` scope measured where it pays** — the `bf16` tier at the advertised
/// `min_size`, the one cell of nine where the ChatGLM3-6B conditioning phase carries the request
/// peak (`the_text_encoder_window_scope_cannot_move_the_request_peak`).
///
/// Three rows, and the contrast between them is the whole point:
///
/// * `Dit` — bounds the U-Net's 70 blocks, in a phase that is **not** the peak-bearing one here, so
///   the request peak barely moves;
/// * `TextEncoder` — bounds the 28 GLM blocks, in the phase that **is**, so the request peak falls;
/// * `Both` — bounds both, and must be at least as good as either.
///
/// Every row must be byte-identical to the staged control: a windowed block is re-materialized
/// through the same constructor with the same replayed tier, so only residency differs. That
/// identity is also the guard against a stream that silently dropped per-block state — Kolors
/// installs none on a `GlmBlock` today, and this is what would notice if that changed.
#[test]
#[ignore = "needs a real SceneWorks/kolors-mlx snapshot (set KOLORS_LADDER_ROOT)"]
fn the_text_encoder_window_bounds_the_conditioning_bearing_cell() {
    // The cell is `bf16` at `descriptor().capabilities.min_size`.
    const TIER: &str = "bf16";
    const EDGE: u32 = 512;
    let dir = require_tier(TIER);
    warm_up(&dir, TIER);
    let control = measure(
        &dir,
        TIER,
        LoadShape::DeferredMaterialization,
        &request(Some(staged()), EDGE, STEPS),
    );
    let mut rows = Vec::new();
    for component in ms::TRANSFORMER_WINDOW_COMPONENTS {
        let row = measure(
            &dir,
            TIER,
            LoadShape::DeferredMaterialization,
            &request(
                Some(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, *component)),
                EDGE,
                STEPS,
            ),
        );
        println!(
            "[sc-15521 rung4 scope {TIER} {EDGE}² {STEPS} steps] {component:?}: {:.4} GiB ({:+.2}%) \
              {:.0} ms/step  max Δ {}",
            row.peak_gib,
            100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib,
            ms_per_step(&row, STEPS),
            max_delta(&control.pixels, &row.pixels),
        );
        assert_eq!(
            control.pixels, row.pixels,
            "scope {component:?} is a residency change, not an arithmetic one"
        );
        rows.push((*component, row.peak_gib));
    }
    println!(
        "[sc-15521 rung4 scope {TIER} {EDGE}²] staged control {:.4} GiB",
        control.peak_gib
    );
    let peak_of = |c: TransformerComponent| {
        rows.iter()
            .find(|(component, _)| *component == c)
            .map(|(_, peak)| *peak)
            .expect("every published scope was measured")
    };
    let dit = peak_of(TransformerComponent::Dit);
    let text = peak_of(TransformerComponent::TextEncoder);
    let both = peak_of(TransformerComponent::Both);

    // The claim `TRANSFORMER_WINDOW_COMPONENTS` publishes the second scope on: at this cell the
    // text-encoder window moves the REQUEST peak, and by more than the 3% margin the ladder holds
    // every implemented rung to.
    assert!(
        text < control.peak_gib * 0.97,
        "the TextEncoder scope must bound the request peak at the one conditioning-bearing cell \
         ({text:.4} vs {:.4} GiB) — that measurement is why the scope is published at all",
        control.peak_gib
    );
    // And the contrast that makes it a distinct scope rather than a spelling of `Dit`: the U-Net
    // window cannot reach this peak, because the phase it bounds is not the peak-bearing one here.
    assert!(
        text < dit,
        "the TextEncoder scope must beat the Dit scope at the cell where the CONDITIONING phase \
         carries the peak ({text:.4} vs {dit:.4} GiB). If it does not, the two scopes are not \
         distinguishable here and publishing both puts a meaningless choice in front of a selector"
    );
    // **`Both` must be MATERIALLY better than `TextEncoder`, not merely no worse.** The measured
    // ratio is 4.5436 / 8.8396 = **0.514** — `Both` additionally bounds the U-Net's 70 blocks, in a
    // phase `TextEncoder` leaves fully resident — so the previous `both <= text * 1.01` granted a 1%
    // free pass against a 49% margin: roughly 50x looser than the relationship it was checking, and
    // wide enough to pass with the `Dit` half of `Both` deleted entirely. 0.60 keeps real headroom
    // over the measurement while still reddening on that collapse.
    assert!(
        both < text * 0.60,
        "`Both` no longer materially beats `TextEncoder` alone ({both:.4} vs {text:.4} GiB, ratio \
         {:.3} against a measured 0.514). Either the Dit half of the `Both` scope has stopped \
         bounding anything — which would make `Both` a spelling of `TextEncoder` and the third \
         published scope meaningless — or the phase balance at this cell has moved and \
         TRANSFORMER_WINDOW_COMPONENTS' table owes fresh numbers",
        both / text
    );
}
