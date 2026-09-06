//! Real-weight conformance and **measurement** for the SANA shared memory ladder — the SC-15523
//! rung-3 / rung-4 half — on Apple/Metal.
//!
//! Every rung-3 and rung-4 number `crate::memory_strategy` publishes comes from this file. Nothing
//! is inherited from Z-Image, Chroma, SDXL or any other family: the epic's standing rule is that a
//! rung's presence, magnitude, mechanism and candidate set are per family per backend, and SANA is
//! the only **ReLU-linear-attention Linear-DiT with a deep-compression (32x) autoencoder** on this
//! ladder. Its rung-3 lever in particular is the opposite of the sibling MLX families': theirs is a
//! graph cut around a fused kernel that never materializes scores, SANA's is an actually-materialized
//! `[B, H, N, 300]` f32 score tensor.
//!
//! ## Measurement discipline (SC-17679, SC-17743)
//!
//! * **MLX's own accounting**, never timer-sampled RSS. `mlx_rs::memory::get_peak_memory` reports
//!   ACTIVE bytes; a sampled RSS measures how fast the machine happened to run.
//! * **A fresh generator per measured row** ([`measure`]). A reused heavy bundle lets the first row
//!   materialize the lazily-loaded stack, and every later row then reads a peak including work it
//!   did not do — which, for rung 4 specifically, is exactly how a windowed sweep can report a
//!   saving that is really "the first row paid for the stack".
//! * **`reset_peak_memory` after the load**, so a row measures the *request*, not the load.
//! * **One discarded warm-up row of the same shape** ([`warm_up`]) — necessary and NOT sufficient.
//!   Whether a cell can support a claim is answered by
//!   [`identical_requests_reproduce_once_the_allocator_has_settled`] (its resolution) and by
//!   [`probe_order`] (whether an apparent effect follows the cadence or the row's position).
//! * **In distribution.** Base SANA runs at its own `DEFAULT_STEPS = 20` / guidance 4.5 true-CFG
//!   schedule and Sprint at `SPRINT_DEFAULT_STEPS = 2` CFG-free. A Chroma sweep run at 4 steps
//!   against a 28-step model published drift that was ~2x wrong; step count is bound to the
//!   variant's real schedule here rather than to whatever is cheap.
//! * Rejected candidates are recorded **with their numbers**, and every rejection is re-asserted
//!   against the production path.
//!
//! ## Mutation proofs (SC-15523)
//!
//! Each rung's implementation was stubbed to its no-op path and the corresponding test confirmed to
//! REDDEN, then reverted. A test that cannot fail is worthless, and byte-identity assertions in
//! particular pass trivially with the feature off — which is why every rung test here also asserts
//! that the rung MOVED something (a peak, or a probe count).
//!
//! | # | stub | reddened | weights |
//! |---|---|---|---|
//! | 1 | `CrossAttn::forward` ignores `budget`, always the single-call branch | `chunked_cross_attention_is_bit_exact_and_actually_chunks` | no |
//! | 2 | …the same stub, on real weights | `attention_chunking_is_measured_at_the_dit_seam` — all three rows read 3.2172 GiB, +0.00% | **yes** |
//! | 3 | `SanaBlock::forward` drops the budget on the way to `attn2` | `a_block_forward_threads_the_attention_budget_to_attn2` | no |
//! | 4 | the chunked path drops the caption mask | `the_caption_mask_changes_the_chunked_result` | no |
//! | 5 | `validate_request_memory` returns `Ok(())` unconditionally | `request_scoped_parameters_are_refused_outside_the_published_domain` — 17 selections admitted | no |
//! | 6 | …the same stub, on real weights | `the_published_domains_are_enforced_by_the_production_path` | **yes** |
//! | 7 | the rung-4 withholding check removed from `validate_request_memory` | `the_withheld_rung_four_is_refused_by_the_production_path` | **yes** |
//! | 8 | the phase probe's decode left unbounded, so the peak may scale | `the_request_peak_bearing_phase_is_measured_not_assumed` — +343.77% over 16x tokens | **yes** |
//! | 9 | `is_streamable` returns `true` | `rung_four_availability_reads_source_load_shape_and_the_staged_prerequisite` | no |
//! | 10 | `windowed` drops its rung-1 half | the same test | no |
//! | 11 | the window view is never drained | `block_stream_drains_exactly_what_the_block_read` | no |
//! | 12 | the drain is `remove_prefix`, not `remove_accessed` | the same test's un-read-key half | no |
//! | 13 | the block stream ignores the trunk's config | `a_materialized_block_matches_its_resident_twin` | no |
//! | 14 | the stream drops **only** the Sprint `qk_norm` gate | `a_base_config_stream_over_sprint_weights_is_present_but_wrong` | no |
//!
//! **Rung 4's execution has no real-weight mutation, and cannot have one.** The rung is withheld, so
//! its mechanism is unreachable from the production `generate` path by construction — #7 proves the
//! withholding is enforced and #11-#14 prove the mechanism weights-free. Stating that plainly is the
//! point: an earlier revision of this table cited a real-weight rung-4 test that had been *deleted*
//! when the rung was withdrawn, which is the fourth phantom citation this epic has produced.
//!
//! **Two mutations deliberately recorded as NOT reddening**, because a mutation that fails to redden
//! bounds an assertion's reach and that is worth knowing:
//!
//! - Forcing degenerate one-row chunks — measured NOT bit-exact at ~1e-6 in the latent — leaves the
//!   rendered image byte-identical. The exactness boundary is real in the latent and does not reach
//!   uint8 pixels, which is why `a_single_query_row_chunk_is_not_bit_exact_and_the_domain_cannot_reach_one`
//!   asserts on tensors rather than on an image.
//! - Dropping the caption mask on the chunked path reddens the unit test (#4) but leaves the real
//!   render byte-identical: at the CHI prompt this file drives, the mask is not load-bearing. So the
//!   real-weight byte-identity rows cannot stand in for #4, and both are kept.
//!
//! ## Weights
//!
//! One env var per catalog entry, each pointing at that entry's snapshot **root** (the tier is a
//! subdirectory: `bf16` / `q4` / `q8`). Nothing self-fetches or derives a cache location
//! (epic 13657). A test whose entry/tier is absent **fails loudly by name** rather than passing
//! silently.
//!
//! | env var | entry |
//! |---|---|
//! | `SANA_LADDER_1600M` | `sana_1600m` |
//! | `SANA_LADDER_SPRINT` | `sana_sprint_1600m` — the representative entry for the heavy sweeps (2 CFG-free steps against base SANA's 40 trunk forwards per image) |
//!
//! Evidence minting additionally requires `INFERENCE_REVISION`, `SCENEWORKS_REVISION`, and — **per
//! entry**, because the two entries are two different HF repositories — `<VAR>_MODEL_REVISION` and
//! `<VAR>_INVENTORY_SHA256`, both taken from
//! `scripts/release/verify_model_snapshot.py --model sana-1600m-mlx --snapshot <root>
//! --inventory-output <json>` against the pin in `release/real-weight-models.toml`.
//!
//! ```text
//! SANA_LADDER_1600M=<root containing q4/> SANA_LADDER_SPRINT=<root containing q4/> \
//!   cargo test -p mlx-gen-sana --release --test integration memory_ladder_real_weights:: \
//!   -- --ignored --test-threads=1 --nocapture
//! ```

#![allow(clippy::items_after_test_module)]

use std::path::PathBuf;

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, MemoryBackend,
    MemoryCalibrationIdentity, MemoryEvidenceKey, MemoryEvidenceLogRecord, MemoryGeometry,
    MemoryMode, MemoryNumericTier, MemoryParityContract, MemoryParityResult, MemoryReferenceShape,
    MemoryStrategy, MemoryStrategyParameters, MemoryStrategySupport, Progress,
    TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use sha2::{Digest, Sha256};

use mlx_gen_sana::memory_strategy as ms;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The two catalog entries this provider serves. One architecture and one loader; they differ in
/// checkpoint, sampler and guidance mode, which is exactly why each still owes its own evidence.
const ENTRIES: &[(&str, &str, bool)] = &[
    ("sana_1600m", "SANA_LADDER_1600M", false),
    ("sana_sprint_1600m", "SANA_LADDER_SPRINT", true),
];

/// The representative entry for the multi-row sweeps — **chosen by measured instrument resolution,
/// not by cost**.
///
/// SANA-Sprint is the cheaper entry (2 CFG-free steps against base SANA's 40 trunk forwards per
/// image) and was the obvious choice on that basis. It is the wrong choice on the only basis that
/// matters. Measured over five whole-ladder runs at 1024sq q4:
///
/// | cell | `sana_1600m` spread | `sana_sprint_1600m` spread |
/// |---|---:|---:|
/// | Resident | 0.00% | 0.00% |
/// | StagedResidency | 0.00% | 0.00% |
/// | BoundedDecode | 0.00% | 1.38% |
/// | BoundedAttention | 0.00% | 6.33% |
/// | BoundedTransformerResidency | 0.00% | 3.62% |
///
/// Base SANA reproduces to the fourth decimal at every rung; Sprint does not, and eight identical
/// windowed requests on Sprint span **9.17%**. The deltas this file publishes are single-digit
/// percentages, so on Sprint they are inside the instrument's own noise and on base they are three
/// orders of magnitude clear of it. The likely reason is the schedule itself — base runs its
/// in-distribution 20 steps and the allocator settles; Sprint runs 2 and the reading is dominated by
/// load transients.
///
/// Sprint still renders its own rows in the ladder walk, **reported with its own measured
/// resolution rather than asserted against a margin its instrument cannot support**. Sharing this
/// provider's code is explicitly not what makes an entry Verified.
const REPRESENTATIVE: &str = "sana_1600m";
const REPRESENTATIVE_ENV: &str = "SANA_LADDER_1600M";
/// Whether [`REPRESENTATIVE`] is the Sprint variant — the sweeps need it for the request shape.
const REPRESENTATIVE_IS_SPRINT: bool = false;
/// The turnkey tier both entries ship and the only one cached at authoring time. Per-tier fan-out is
/// the catalog stories' work (SC-15490 / SC-15491); this file owns the provider rungs.
const DEFAULT_TIER: &str = "q4";

/// The number of denoise steps each variant's real schedule runs. Bound to the engine defaults so a
/// sweep cannot drift out of distribution.
fn steps_for(sprint: bool) -> u32 {
    if sprint {
        mlx_gen_sana::pipeline::SPRINT_DEFAULT_STEPS as u32
    } else {
        mlx_gen_sana::pipeline::DEFAULT_STEPS as u32
    }
}

fn tier_dir(var: &str, tier: &str) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var(var).ok()?);
    let dir = root.join(tier);
    dir.is_dir().then_some(dir)
}

#[track_caller]
fn require_tier(var: &str, tier: &str) -> PathBuf {
    tier_dir(var, tier).unwrap_or_else(|| {
        panic!("SKIPPED-BY-ABSENCE: set {var} to a snapshot root containing {tier}/")
    })
}

fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

fn spec(dir: &std::path::Path, tier: &str, shape: LoadShape) -> LoadSpec {
    spec_with_policy(dir, tier, shape, OffloadPolicy::Sequential)
}

/// **SANA's rung 1 is a LOAD-time mechanism**, so a rung-0 row has to be a `Resident` LOAD.
///
/// The shared `Residency` seam is driven by `spec.offload_policy`; `GenerationMemory::stage_residency`
/// is a contract-level marker for this provider, not a lever the engine reads (sc-16783 measured and
/// published rung 1 that way). A row rendered on a Sequential load with `stage_residency: false`
/// therefore still stages its phases — and labelling that row `MemoryStrategy::Resident` in an
/// evidence record would name a composition the engine did not run, which is exactly the mislabelled
/// evidence this story's last acceptance criterion is about.
fn spec_with_policy(
    dir: &std::path::Path,
    tier: &str,
    shape: LoadShape,
    policy: OffloadPolicy,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(policy)
        .with_load_shape(shape);
    spec.quantize = quant_for(tier);
    spec
}

fn request(
    sprint: bool,
    memory: Option<GenerationMemory>,
    edge: u32,
    steps: u32,
) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        // Sprint is CFG-free and advertises no negative prompt; base SANA is true-CFG.
        negative_prompt: (!sprint).then(|| "blurry, lowres".to_owned()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(steps),
        guidance: Some(4.5),
        seed: Some(1234),
        memory,
        ..Default::default()
    }
}

// ── Compositions ─────────────────────────────────────────────────────────────────────────────────

/// The all-rungs-off baseline: an explicit block, so the load-time `Sequential` policy cannot leak a
/// phase release (or SANA's Sequential decode-tiling default) into the "resident" row.
fn resident_memory() -> GenerationMemory {
    GenerationMemory::default()
}

/// The **rung-2** production composition. It deliberately does NOT stage: the shared cost order
/// excludes rung 1 (bounding residency may evict the warm cross-request pair, a cost the next
/// request pays), so a selector choosing rung 2 gets tiling and nothing else.
fn rung2() -> GenerationMemory {
    GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(mlx_gen_sana::pipeline::DECODE_TILE_EDGE as u32),
        decode_overlap: Some(mlx_gen_sana::pipeline::DECODE_OVERLAP as u32),
        ..Default::default()
    }
}

/// The **rung-3** production composition: rung 3 engages rung 2 by cost order, still not rung 1.
fn rung3(chunk: u32) -> GenerationMemory {
    GenerationMemory {
        chunk_attention: true,
        attention_chunk_size: Some(chunk),
        ..rung2()
    }
}

/// The **published production ceiling**: rungs 1-3, which is what a selector can actually compose
/// now that rung 4 is withheld. It is also the control the withheld rung was measured against.
fn rung4_control() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..rung3(ms::ATTENTION_CHUNK_SIZE)
    }
}

/// A rung-4 selection — **refused by the production path**, since the rung is withheld. Retained so
/// the refusal is exercised rather than assumed.
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

// ── The instrument ───────────────────────────────────────────────────────────────────────────────

/// One measured row: the request's ACTIVE-bytes peak, its pixels, and its wall clock.
struct Row {
    peak_bytes: u64,
    peak_gib: f64,
    pixels: Vec<u8>,
    wall: std::time::Duration,
}

/// Render one row on a **fresh** generator and return its request peak.
///
/// The freshness is the whole contract of this helper. SANA's trunk weights are lazy MLX handles
/// until something evaluates them, so a generator reused across rows carries the previous row's
/// materialization into this row's peak. Every row in this file goes through here.
#[track_caller]
fn measure(
    entry: &str,
    dir: &std::path::Path,
    tier: &str,
    shape: LoadShape,
    req: &GenerationRequest,
) -> Row {
    measure_with_policy(entry, dir, tier, shape, OffloadPolicy::Sequential, req)
}

#[track_caller]
fn measure_with_policy(
    entry: &str,
    dir: &std::path::Path,
    tier: &str,
    shape: LoadShape,
    policy: OffloadPolicy,
    req: &GenerationRequest,
) -> Row {
    let registry = mlx_gen_sana::provider_registry().expect("provider registry");
    let model = registry
        .load(entry, &spec_with_policy(dir, tier, shape, policy))
        .unwrap_or_else(|error| panic!("load {entry}: {error}"));
    clear_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();
    let out = model
        .generate(req, &mut |_: Progress| {})
        .unwrap_or_else(|error| panic!("generate must succeed: {error}"));
    let peak = get_peak_memory();
    let wall = started.elapsed();
    let pixels = match out {
        GenerationOutput::Images(images) => images.first().expect("one image").pixels.clone(),
        other => panic!("expected images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    Row {
        peak_bytes: peak as u64,
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

/// Discard one measured row before publishing any peak from this process (SC-17679).
///
/// `get_peak_memory` reads ACTIVE bytes, and the very first `generate` in a process reads them
/// against a cold allocator. The sibling Kolors harness measured that bias directly: a windowed row
/// measured first read 4.4632 GiB for a configuration that reads 4.6924 once warm — a 4.9% phantom
/// spread that looked exactly like a real finding.
///
/// The warm-up runs the **published ceiling** (rungs 1-3) at the SAME edge the caller will publish
/// at, because a row's transients are a function of the token count.
#[track_caller]
fn warm_up(entry: &str, dir: &std::path::Path, sprint: bool, edge: u32) {
    let _ = measure(
        entry,
        dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(sprint, Some(rung4_control()), edge, steps_for(sprint)),
    );
}

fn probe_size() -> u32 {
    std::env::var("SANA_PROBE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

/// The order the cadence sweep executes its rows in — **the control that tells a cadence effect
/// apart from a positional one** (SC-17679).
///
/// Defaults to [`ms::TRANSFORMER_WINDOW_SIZES`]' own order. `SANA_WINDOW_PROBE_ORDER` overrides it
/// with a comma-separated permutation. If the peaks follow the *positions* rather than the cadences,
/// the cell is unresolvable and must be withdrawn as evidence — a genuine weight-residency effect
/// cannot move with execution order.
///
/// **This is the control that withdrew SANA's rung 4** (see
/// [`the_withheld_rung_four_is_refused_by_the_production_path`]): cadence 10 run first and cadence 4
/// run first both read exactly 2.8602 GiB. It stays LIVE rather than `#[allow(dead_code)]` — that
/// test drives its refusal loop through it, so a permutation must refuse exactly the same set.
/// Deleting it would delete the instrument the withdrawal rests on.
fn probe_order() -> Vec<u32> {
    let Ok(spec) = std::env::var("SANA_WINDOW_PROBE_ORDER") else {
        return ms::TRANSFORMER_WINDOW_SIZES.to_vec();
    };
    let order: Vec<u32> = spec
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("SANA_WINDOW_PROBE_ORDER: not a u32")
        })
        .collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    let mut domain = ms::TRANSFORMER_WINDOW_SIZES.to_vec();
    domain.sort_unstable();
    assert_eq!(
        sorted, domain,
        "SANA_WINDOW_PROBE_ORDER must be a PERMUTATION of the published domain — an order probe \
         that also changed which cadences ran would confound the two things it exists to separate"
    );
    println!("[sc-17679 order probe] executing cadences in the order {order:?}");
    order
}

/// The order the rung-3 sweep executes its budgets in — the same control [`probe_order`] is for
/// cadences, and the one that settled what the budget column actually measures here.
///
/// `SANA_BUDGET_PROBE_ORDER` overrides [`ms::ATTENTION_CHUNK_SIZES`]' own order with a
/// comma-separated permutation. If the peaks follow the *positions* rather than the budgets, the
/// budget ordering is unresolvable and must be withdrawn as evidence.
fn budget_probe_order() -> Vec<u32> {
    let Ok(spec) = std::env::var("SANA_BUDGET_PROBE_ORDER") else {
        return ms::ATTENTION_CHUNK_SIZES.to_vec();
    };
    let order: Vec<u32> = spec
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("SANA_BUDGET_PROBE_ORDER: not a u32")
        })
        .collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    let mut domain = ms::ATTENTION_CHUNK_SIZES.to_vec();
    domain.sort_unstable();
    assert_eq!(
        sorted, domain,
        "SANA_BUDGET_PROBE_ORDER must be a PERMUTATION of the published domain — an order probe \
         that also changed which budgets ran would confound the two things it exists to separate"
    );
    println!("[sc-17679 order probe] executing budgets in the order {order:?}");
    order
}

/// The resolution this cell's instrument is required to stay inside, and the floor every published
/// claim in this file must clear.
///
/// **Measured: 4.92%, and the noise is not random.** Eight identical requests produce a
/// deterministic five-value cycle — 2.8053 / 2.9108 / 2.8602 / 2.8270 / 2.9434 GiB, repeating with
/// period five — so a row's reading is a function of its ORDINAL as much as of its request. That is
/// why nothing here is published from a single row, why the rung-3 sweep runs under
/// [`budget_probe_order`], and why rung 4 was withdrawn: under [`probe_order`] its peak followed the
/// row's position and not its cadence.
///
/// **The baseline the deltas are measured against is exact.** Measured, not assumed: the rung-2
/// control cell reads 3.2172..3.2173 GiB over eight rows — **0.00%** — and 3.2170..3.2173 over
/// eleven (0.01%). The cycle therefore belongs to the rung-3-engaged path, not to the harness. That
/// is also what makes mutation #2's "+0.00% to four decimals" legitimate rather than the false-green
/// signature it superficially resembles: the stubbed rows land on the control's own exact value.
///
/// ## The aliasing rule, corrected — it is about STRIDE, not repeat count
///
/// Five independent whole-ladder runs earlier reported a 0.00% spread at every rung, which read as a
/// perfect instrument. An earlier revision of this doc explained it as *"a repeat count that is a
/// multiple of the cycle length cannot see the cycle"*. **That is false**, and it is the sentence
/// the next family would have copied. Falsified by measurement: at `SANA_SETTLE_ROWS=11` the
/// published window is rows 2..=11 — exactly ten rows, a multiple of five — and it still reports
/// **4.92%**, showing all five values.
///
/// The real invariant is about the **stride** a comparison takes through the cycle:
///
/// > A comparison whose stride is a multiple of the cycle period cannot see the cycle.
///
/// The five-rung ladder walk is exactly that: each rung is sampled once per walk, so each rung's
/// stride is 5 ≡ 0 (mod 5) and every rung always lands on the same phase — five runs, five identical
/// readings, 0.00%. The repeat COUNT was never the mechanism.
///
/// **Corollary, and the reason the sibling probes are trustworthy:** a settle probe samples
/// *consecutive* rows, so its stride is 1, and 1 is coprime to every period. A stride-1 probe is
/// therefore immune at any repeat count — which is why `mlx-gen-sdxl`'s and `mlx-gen-chroma`'s
/// 8-repeat 0.00% readings mean what they say. Design the probe stride-1; do not tune the count.
///
/// **Caveat: the cycle is deterministic per FRESH PROCESS, not invariant to process history.**
/// Observed directly in the eleven-row run, where the control cell's eleven rows execute first in
/// the same process: the rung-3 rows then open `2.8643 / 2.8563 / 2.8270 / 2.8784 / 2.8053 …`, three
/// of which are outside the five-value set, before settling into the cycle from row #5. The period
/// and the amplitude are stable; the phase and the exact members are not portable across a process
/// that has already allocated differently. Every number this file publishes therefore comes from a
/// **targeted `--exact` run**, and the 6% ceiling exists partly to absorb this.
///
/// The only effect this file publishes is rung 3's, whose WORST row is -8.51% — comfortably clear of
/// both the cycle and the ceiling, and measured against a baseline that is itself exact.
const INSTRUMENT_CEILING: f64 = 0.06;

/// **Does this harness's peak reading depend on a row's ORDINAL rather than on its request?**
///
/// The instrument check the whole file rests on. It renders the **identical** request
/// `SETTLE_PROBE_ROWS` times on a fresh generator each time and prints every row's peak. A request
/// that is byte-for-byte the same each time can only produce different peaks if the reading is a
/// function of something other than the request.
///
/// The claim this asserts is not "MLX always reproduces". It is "**the tolerance this file's
/// assertions use is not finer than the instrument's resolution at the cell they run on**". A cell
/// whose resolution is, say, 5.67% (SDXL measured exactly that at 512 q8) can support no claim at
/// all, in either direction.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn identical_requests_reproduce_once_the_allocator_has_settled() {
    // Overridable so the stride invariant above is CHECKABLE rather than asserted: with
    // `SANA_SETTLE_ROWS=11` the published window is rows 2..=11, exactly ten — a multiple of the
    // period — and it still reports the full 4.92%.
    let settle_probe_rows: usize = std::env::var("SANA_SETTLE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    /// The ceiling this file's published claims are required to clear. See [`INSTRUMENT_CEILING`].
    const SETTLE_TOLERANCE: f64 = INSTRUMENT_CEILING;

    let edge = probe_size();
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);

    // **Both cells, because a delta needs its baseline resolved too.** Probing only the
    // rung-3-engaged cell would leave a reader unable to tell a cycle that belongs to the rung from
    // one that belongs to the harness — and mutation #2's "+0.00% to four decimals" would be
    // indistinguishable from the false-green signature it superficially resembles.
    let sample = |memory: GenerationMemory| -> Vec<f64> {
        (0..settle_probe_rows)
            .map(|_| {
                measure(
                    REPRESENTATIVE,
                    &dir,
                    DEFAULT_TIER,
                    LoadShape::DeferredMaterialization,
                    &request(REPRESENTATIVE_IS_SPRINT, Some(memory), edge, steps),
                )
                .peak_gib
            })
            .collect()
    };
    let spread_of = |peaks: &[f64]| {
        let published = &peaks[1..];
        let (min, max) = published
            .iter()
            .fold((f64::MAX, 0f64), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        (min, max, (max - min) / min)
    };

    let control_peaks = sample(rung2());
    let (cmin, cmax, cspread) = spread_of(&control_peaks);
    println!(
        "[sc-15523 settle {DEFAULT_TIER} {edge}sq rung-2 CONTROL] x{settle_probe_rows}: {}",
        control_peaks
            .iter()
            .enumerate()
            .map(|(i, p)| format!("#{}: {p:.4}", i + 1))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!(
        "[sc-15523 settle {DEFAULT_TIER} {edge}sq] CONTROL RESOLUTION {:.2}% ({cmin:.4}..{cmax:.4} GiB)",
        100.0 * cspread
    );
    assert!(
        cspread < INSTRUMENT_CEILING,
        "the rung-2 control is no longer a stable baseline: {cmin:.4}..{cmax:.4} GiB ({:.2}%) — \
         every published delta inherits this cell's noise",
        100.0 * cspread
    );

    let peaks = sample(rung4_control());
    println!(
        "[sc-15523 settle {DEFAULT_TIER} {edge}sq rungs 1-3] identical request x{settle_probe_rows}: {}",
        peaks
            .iter()
            .enumerate()
            .map(|(i, p)| format!("#{}: {p:.4}", i + 1))
            .collect::<Vec<_>>()
            .join("  ")
    );

    // Row 1 is the one `warm_up` discards. Everything after it is a row this file would PUBLISH.
    let (min, max, spread) = spread_of(&peaks);
    println!(
        "[sc-15523 settle {DEFAULT_TIER} {edge}sq] INSTRUMENT RESOLUTION {:.2}% over rows \
         2..={settle_probe_rows} ({min:.4}..{max:.4} GiB; row 1 is the discarded warm-up)",
        100.0 * spread
    );
    assert!(
        spread < SETTLE_TOLERANCE,
        "identical requests no longer reproduce: rows 2..= span {min:.4}..{max:.4} GiB ({:.2}%), \
         against the tightest published margin of {:.0}%. Every finding in this file would then be \
         running finer than the instrument's own resolution",
        100.0 * spread,
        100.0 * SETTLE_TOLERANCE
    );
}

// ── Rung 3 ───────────────────────────────────────────────────────────────────────────────────────

/// **Rung 3: does bounding `attn2`'s materialized score tensor move the REQUEST peak, and is the
/// image preserved?**
///
/// Two claims, both required, and the epic's rule is that the second without the first is not a
/// saving. The sweep runs every published budget plus the rejected sibling constant, so the domain
/// is recorded WITH its numbers whichever way the answer comes out.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn attention_chunking_is_measured_at_the_dit_seam() {
    let edge = probe_size();
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);
    warm_up(REPRESENTATIVE, &dir, REPRESENTATIVE_IS_SPRINT, edge);

    let control = measure(
        REPRESENTATIVE,
        &dir,
        DEFAULT_TIER,
        LoadShape::DeferredMaterialization,
        &request(REPRESENTATIVE_IS_SPRINT, Some(rung2()), edge, steps),
    );
    let mut rows = Vec::new();
    let swept = budget_probe_order();
    for budget in swept.iter().chain(ms::ATTENTION_CHUNK_SIZES_REJECTED) {
        // The rejected constant is not in the published domain, so the production path refuses it —
        // which is itself the measurement being recorded for it.
        let row = if ms::ATTENTION_CHUNK_SIZES.contains(budget) {
            Some(measure(
                REPRESENTATIVE,
                &dir,
                DEFAULT_TIER,
                LoadShape::DeferredMaterialization,
                &request(REPRESENTATIVE_IS_SPRINT, Some(rung3(*budget)), edge, steps),
            ))
        } else {
            None
        };
        rows.push((*budget, row));
    }

    println!(
        "[sc-15523 rung3 {DEFAULT_TIER} {edge}sq] rung-2 control {:.4} GiB",
        control.peak_gib
    );
    let mut best = f64::MAX;
    let mut worst = 0f64;
    for (budget, row) in &rows {
        match row {
            Some(row) => {
                let delta = 100.0 * (row.peak_gib - control.peak_gib) / control.peak_gib;
                println!(
                    "[sc-15523 rung3 {DEFAULT_TIER} {edge}sq] budget {budget:>9}: {:.4} GiB \
                     ({delta:+.2}%), maxD {}, meanD {:.4}, {:.0} ms/step",
                    row.peak_gib,
                    max_delta(&control.pixels, &row.pixels),
                    mean_delta(&control.pixels, &row.pixels),
                    ms_per_step(row, steps)
                );
                best = best.min(row.peak_gib);
                worst = worst.max(row.peak_gib);
                // Rung 3 is a scratch bound, not an arithmetic change: query-row chunking keeps each
                // output row's complete k/v and both reductions, so the image must be identical.
                assert_eq!(
                    max_delta(&control.pixels, &row.pixels),
                    0,
                    "budget {budget} changed the image — query-row chunking must be bit-exact"
                );
            }
            None => println!(
                "[sc-15523 rung3 {DEFAULT_TIER} {edge}sq] budget {budget:>9}: REJECTED (outside \
                 the published domain; the shared 64 Mi constant never chunks SANA)"
            ),
        }
    }
    // **The assertion that can fail.** Byte-identity alone passes with the chunking deleted — a
    // resident forward is trivially identical to itself — so the rung must also MOVE the request
    // peak. Measured on this entry: 3.2172 -> 2.9434 GiB, **-8.51%**, against a cell resolution of
    // 0.00% over five independent runs. 3% is the floor a no-op cannot clear in either direction.
    let best_delta = 100.0 * (best - control.peak_gib) / control.peak_gib;
    let worst_delta = 100.0 * (worst - control.peak_gib) / control.peak_gib;
    println!(
        "[sc-15523 rung3 {DEFAULT_TIER} {edge}sq] VERDICT published budgets {worst_delta:+.2}%..\
         {best_delta:+.2}% vs the rung-2 control, every row byte-identical"
    );
    // **The assertion that can fail, and it is keyed to the instrument rather than to a round
    // number.** Byte-identity alone passes with the chunking deleted. The claim is that EVERY
    // published budget beats the rung-2 control by more than this cell's own resolution — measured
    // -8.51% at the worst row against a 4.92% five-value positional cycle. A no-op would land inside
    // the cycle, which is exactly where rung 4 landed and why rung 4 is withheld.
    assert!(
        worst < control.peak_gib * (1.0 - INSTRUMENT_CEILING),
        "bounded attention did not bound the request peak by more than the instrument can resolve: \
         worst published row {worst:.4} vs control {:.4} GiB ({worst_delta:+.2}%), against a \
         {:.0}% ceiling. SANA's attn2 materializes its scores, so a real bound is expected here \
         rather than the graph-cut-only effect the fused-kernel families measure",
        control.peak_gib,
        100.0 * INSTRUMENT_CEILING
    );
    // The three budgets are published as EQUAL-PEAK alternatives. Their ordering is not published,
    // because under `SANA_BUDGET_PROBE_ORDER` it follows the row's position: two different orders
    // both produced 2.8602 / 2.8270 / 2.9434 in positional sequence.
    assert!(
        100.0 * (worst - best) / best < 100.0 * INSTRUMENT_CEILING,
        "the published budgets stopped being equal-peak alternatives ({best:.4}..{worst:.4} GiB); \
         either narrow the domain to the one that was measured, or publish an ordering the order \
         permutation can actually support"
    );
}

// ── Rung 4 ───────────────────────────────────────────────────────────────────────────────────────

/// **Rung 4 is implemented, output-preserving, measured, and WITHHELD — and the withdrawal is
/// enforced by the production path.**
///
/// The numbers behind the withdrawal, measured on `sana_1600m` q4 at 1024sq (Sequential+Deferred),
/// live on [`ms::TRANSFORMER_WINDOW_WITHHELD`]. In short: the window bounds the trunk's weight
/// residency from 1871.9 MiB to 93.59 MiB and costs +42% wall, the image is byte-identical across
/// five production latents, and the request peak moves -1.74% — inside a **deterministic five-value
/// positional cycle spanning 4.92%** whose value follows the row's ORDINAL and not its cadence
/// (cadence 10 first and cadence 4 first both read exactly 2.8602 GiB under
/// `SANA_WINDOW_PROBE_ORDER`). Bounding a phase that is not the request peak is not a saving.
///
/// What this test asserts is the consequence: **every rung-4 selection is refused**, by name, on the
/// production `generate` path — not merely absent from the contract. A withheld rung a caller can
/// still execute is a rung that ships unmeasured.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn the_withheld_rung_four_is_refused_by_the_production_path() {
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let model = mlx_gen_sana::provider_registry()
        .expect("provider registry")
        .load(
            REPRESENTATIVE,
            &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
        )
        .expect("load sana");
    const { assert!(ms::TRANSFORMER_WINDOW_WITHHELD) };

    // Driven through [`probe_order`] so the cadence-order control stays live rather than becoming a
    // relic of the sweep it withdrew: a permutation must refuse exactly the same set.
    let mut admitted = Vec::new();
    for window in &probe_order() {
        match model.generate(
            &request(REPRESENTATIVE_IS_SPRINT, Some(full_ladder(*window)), 256, 1),
            &mut |_| {},
        ) {
            Ok(_) => admitted.push(format!("cadence {window}")),
            Err(error) => assert!(
                error.to_string().contains("WITHHELD"),
                "the refusal must name the withdrawal, got: {error}"
            ),
        }
    }
    for component in [
        TransformerComponent::Dit,
        TransformerComponent::TextEncoder,
        TransformerComponent::Both,
    ] {
        if model
            .generate(
                &request(
                    REPRESENTATIVE_IS_SPRINT,
                    Some(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, component)),
                    256,
                    1,
                ),
                &mut |_| {},
            )
            .is_ok()
        {
            admitted.push(format!("component {component:?}"));
        }
    }
    assert!(
        admitted.is_empty(),
        "the production path executed a withheld rung-4 selection: {admitted:?}"
    );

    // …and the published ceiling still renders, so the refusal is specific rather than a blanket
    // rejection of every request block.
    model
        .generate(
            &request(REPRESENTATIVE_IS_SPRINT, Some(rung4_control()), 256, 1),
            &mut |_| {},
        )
        .expect("the published rungs 1-3 composition must still render");
}

/// **Which PHASE bears the request peak — measured, not inferred from a byte table.**
///
/// The rung-4 withdrawal rests on the claim that after rungs 1-3 the peak is no longer the denoise
/// weight residency. That claim began as an *inference*: the peak is ~2.87 GiB, the windowed denoise
/// phase holds ~1.28 GiB, and the only component large enough to account for the difference is the
/// Gemma-2 caption encoder at 2211.4 MiB. Sound arithmetic, but arithmetic.
///
/// This measures it, using the one lever that separates the phases without instrumenting them:
/// **geometry**. SANA's conditioning phase is geometry-INDEPENDENT — the CHI prompt pads to a fixed
/// 300 caption slots at every output size — while denoise and decode both scale with the token count
/// `N = (edge/32)²`. So sweeping the advertised edge range separates them:
///
/// - a peak that stays **flat** as the edge grows is borne by the conditioning phase;
/// - a peak that **scales** with pixels is borne by denoise or decode.
///
/// Measured: 2.9108 / 2.8602 / 2.8270 GiB at 256² / 512² / 1024² — **-2.88% across a 16x token
/// increase**, and every value a member of the five-cycle. Flat. A denoise- or decode-borne peak
/// cannot do that, which is what makes rung 4 a withdrawal rather than a defect: bounding the trunk
/// cannot move a peak the trunk does not set. It also tells sc-17859 it is aimed at the right
/// component.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn the_request_peak_bearing_phase_is_measured_not_assumed() {
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);
    warm_up(REPRESENTATIVE, &dir, REPRESENTATIVE_IS_SPRINT, 1024);

    let mut rows = Vec::new();
    for edge in [256_u32, 512, 1024] {
        let row = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &request(REPRESENTATIVE_IS_SPRINT, Some(rung4_control()), edge, steps),
        );
        let pixels = u64::from(edge) * u64::from(edge);
        println!(
            "[sc-15523 phase {DEFAULT_TIER}] {edge}sq ({pixels} px): {:.4} GiB, {:.0} ms/step",
            row.peak_gib,
            ms_per_step(&row, steps)
        );
        rows.push((edge, row.peak_gib));
    }

    let (_, smallest) = rows[0];
    let (_, largest) = rows[rows.len() - 1];
    let growth = 100.0 * (largest - smallest) / smallest;
    println!(
        "[sc-15523 phase {DEFAULT_TIER}] 256sq -> 1024sq is 16x the tokens: peak {growth:+.2}% \
         => the peak-bearing phase is {}",
        if growth.abs() < 100.0 * INSTRUMENT_CEILING {
            "GEOMETRY-INDEPENDENT (the conditioning phase — the Gemma-2 caption encoder)"
        } else {
            "GEOMETRY-SCALING (denoise or decode)"
        }
    );
    assert!(
        growth.abs() < 100.0 * INSTRUMENT_CEILING,
        "the request peak now scales with geometry ({growth:+.2}% over a 16x token increase), so \
         it is no longer borne by the geometry-independent conditioning phase — the rung-4 \
         withdrawal was argued on the opposite finding and has to be re-measured"
    );
    // …and the flat peak must be large enough to BE the caption encoder, rather than something
    // smaller that merely happens not to scale. Gemma-2 q4 is 2211.37 MiB of weights alone.
    const GEMMA_WEIGHT_GIB: f64 = 2211.37 / 1024.0;
    assert!(
        smallest > GEMMA_WEIGHT_GIB,
        "the flat peak {smallest:.4} GiB is below the Gemma-2 weight floor {GEMMA_WEIGHT_GIB:.4} \
         GiB, so the conditioning phase cannot be what sets it"
    );
}

// ── Domain enforcement ───────────────────────────────────────────────────────────────────────────

/// **Every published parameter is reachable and every withheld one is refused — by the production
/// `generate` path, not only by shared-contract admission.**
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn the_published_domains_are_enforced_by_the_production_path() {
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let registry = mlx_gen_sana::provider_registry().expect("provider registry");
    let model = registry
        .load(
            REPRESENTATIVE,
            &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
        )
        .expect("load sana sprint");
    // One step at the smallest advertised output: this test is about admission, not about peaks.
    let probe = |memory: GenerationMemory| {
        model.generate(
            &request(REPRESENTATIVE_IS_SPRINT, Some(memory), 256, 1),
            &mut |_| {},
        )
    };

    for budget in ms::ATTENTION_CHUNK_SIZES {
        probe(rung3(*budget))
            .unwrap_or_else(|error| panic!("published budget {budget} must render, got: {error}"));
    }

    // Collected rather than asserted in-loop, so a regression reports every value it affects.
    let mut silently_admitted: Vec<String> = Vec::new();
    // Rung 4 is withheld, so EVERY cadence is out of domain — published-shaped and not alike.
    for bad in ms::TRANSFORMER_WINDOW_SIZES
        .iter()
        .copied()
        .chain([0_u32, 3, 6, 7, 11, 20, 21, 70])
    {
        if probe(full_ladder(bad)).is_ok() {
            silently_admitted.push(format!("cadence {bad}"));
        }
    }
    for bad in ms::ATTENTION_CHUNK_SIZES_REJECTED
        .iter()
        .chain(&[0, 7, 999])
    {
        assert!(!ms::ATTENTION_CHUNK_SIZES.contains(bad));
        if probe(rung3(*bad)).is_ok() {
            silently_admitted.push(format!("budget {bad}"));
        }
    }
    for bad in ms::DECODE_TILE_EDGES_REJECTED {
        let memory = GenerationMemory {
            decode_tile_edge: Some(*bad),
            ..rung2()
        };
        if probe(memory).is_ok() {
            silently_admitted.push(format!("decode edge {bad}"));
        }
    }
    for component in [
        TransformerComponent::TextEncoder,
        TransformerComponent::Both,
    ] {
        if probe(full_ladder_scoped(ms::TRANSFORMER_WINDOW_SIZE, component)).is_ok() {
            silently_admitted.push(format!("window component {component:?}"));
        }
    }
    assert!(
        silently_admitted.is_empty(),
        "the production path silently admitted these unpublished selections: {silently_admitted:?}"
    );
}

/// **Rung 4 is unavailable on an eager load, and the refusal reaches the production path.**
///
/// The contract declares it per LOAD; a generator that accepted the request anyway would execute an
/// unmeasured shape over an already-committed stack and report a false saving.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn an_eager_load_declares_and_refuses_rung_four() {
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let eager = spec(&dir, DEFAULT_TIER, LoadShape::EagerMaterialization);
    let contract = ms::memory_strategy_contract(REPRESENTATIVE, &eager).expect("contract");
    assert!(contract.conformance_errors().is_empty());
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .support,
        MemoryStrategySupport::Missing
    );
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap()
            .support,
        MemoryStrategySupport::Implemented,
        "rung 3 bounds scratch, so it is load-shape independent — an eager load keeps it"
    );
    let model = mlx_gen_sana::provider_registry()
        .expect("registry")
        .load(REPRESENTATIVE, &eager)
        .expect("load");
    let error = model
        .generate(
            &request(
                REPRESENTATIVE_IS_SPRINT,
                Some(full_ladder(ms::TRANSFORMER_WINDOW_SIZE)),
                256,
                1,
            ),
            &mut |_| {},
        )
        .expect_err("an eager load must refuse rung 4");
    assert!(
        error.to_string().contains("WITHHELD"),
        "the refusal must name the withdrawal, got: {error}"
    );
}

// ── Per-entry conformance + evidence ─────────────────────────────────────────────────────────────

/// **Every catalog entry exercises every implemented rung on this backend, and each mints its own
/// `MEMORY_EVIDENCE_V1` record.**
///
/// This is the story's "at least one representative entry exercises every implemented rung on each
/// advertised backend" AC and its "measurement recorded separately" AC. Sharing this provider's code
/// is explicitly not what makes an entry Verified, so both entries render their own rows at their
/// own in-distribution schedules.
///
/// The records are printed as strict `MEMORY_EVIDENCE_V1` lines; the paired SceneWorks delivery
/// ingests them into `docs/generated/memory-calibration-evidence.json`. The four revision/inventory
/// environment variables are REQUIRED — an evidence record that cannot name the exact revisions it
/// was measured at is the "stale or fingerprint-mismatched evidence" this story's last AC forbids.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn every_entry_exercises_every_implemented_rung_and_mints_evidence() {
    let edge = probe_size();
    let mut minted = 0_usize;
    for (entry, var, sprint) in ENTRIES {
        let dir = require_tier(var, DEFAULT_TIER);
        let steps = steps_for(*sprint);
        warm_up(entry, &dir, *sprint, edge);
        let load = spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization);
        let contract = ms::memory_strategy_contract(entry, &load).expect("contract");
        assert!(
            contract.conformance_errors().is_empty(),
            "{entry}: {:?}",
            contract.conformance_errors()
        );

        // The rung-0 baseline is a RESIDENT LOAD, not a Sequential load with the request flags off —
        // see `spec_with_policy`. Its contract is a different contract, so it is resolved per row.
        let resident_load = spec_with_policy(
            &dir,
            DEFAULT_TIER,
            LoadShape::EagerMaterialization,
            OffloadPolicy::Resident,
        );
        let resident_contract =
            ms::memory_strategy_contract(entry, &resident_load).expect("contract");
        assert!(resident_contract.conformance_errors().is_empty());

        let mut previous: Option<(MemoryStrategy, f64)> = None;
        let mut previous_pixels: Option<Vec<u8>> = None;
        for (strategy, memory, policy) in [
            (
                MemoryStrategy::Resident,
                resident_memory(),
                OffloadPolicy::Resident,
            ),
            (
                MemoryStrategy::StagedResidency,
                GenerationMemory {
                    stage_residency: true,
                    ..Default::default()
                },
                OffloadPolicy::Sequential,
            ),
            (
                MemoryStrategy::BoundedDecode,
                rung2(),
                OffloadPolicy::Sequential,
            ),
            (
                MemoryStrategy::BoundedAttention,
                rung3(ms::ATTENTION_CHUNK_SIZE),
                OffloadPolicy::Sequential,
            ),
        ] {
            let resident_row = matches!(policy, OffloadPolicy::Resident);
            let (row_load, row_contract) = if resident_row {
                (&resident_load, &resident_contract)
            } else {
                (&load, &contract)
            };
            assert_eq!(
                row_contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{entry} must implement {strategy:?} on the route its row runs"
            );
            let row = measure_with_policy(
                entry,
                &dir,
                DEFAULT_TIER,
                row_load.load_shape,
                policy,
                &request(*sprint, Some(memory), edge, steps),
            );
            println!(
                "[sc-15523 ladder {entry} {DEFAULT_TIER} {edge}sq] {strategy:?} ({policy:?}): \
                 {:.4} GiB, {:.0} ms/step",
                row.peak_gib,
                ms_per_step(&row, steps)
            );
            if let Some((prior, prior_peak)) = previous {
                println!(
                    "[sc-15523 ladder {entry}] {prior:?} -> {strategy:?}: {:+.2}%",
                    100.0 * (row.peak_gib - prior_peak) / prior_peak
                );
            }
            previous = Some((strategy, row.peak_gib));

            // **The parity verdict each record carries is established HERE, and it is per rung.**
            //
            // Each rung is compared against the composition it EXTENDS, not against rung 0. That is
            // the only comparison that isolates what the rung itself does — and getting it wrong is
            // not hypothetical: an earlier revision compared every row to rung 0 and reddened on
            // `BoundedDecode`, because **tiled decode is not byte-identical to whole-image decode**.
            // It is an approximation with a real drift, which is exactly why rung 2 declares a
            // TOLERANCE contract while the others declare `Exact`.
            if let Some(previous) = &previous_pixels {
                let max = max_delta(previous, &row.pixels);
                let mean = mean_delta(previous, &row.pixels);
                println!(
                    "[sc-15523 parity {entry}] {strategy:?} vs the composition it extends: \
                     maxD {max}, meanD {mean:.6}"
                );
                match parity_contract(strategy) {
                    MemoryParityContract::Exact => assert_eq!(
                        max, 0,
                        "{entry} {strategy:?} declares an EXACT parity contract but moved the \
                         image (maxD {max}) — it is an arithmetic change, not a memory schedule"
                    ),
                    // The declared metric is the MEAN, so the assertion reads the mean. Asserting
                    // on `max` here would assert on a statistic that saturates at 255 on this route.
                    MemoryParityContract::Tolerance { maximum_error, .. } => assert!(
                        mean <= maximum_error,
                        "{entry} {strategy:?} drifted meanD {mean:.4} past its declared tolerance \
                         {maximum_error} (maxD {max}, which saturates on this route and bounds \
                         nothing)"
                    ),
                    other => panic!("unexpected parity contract {other:?}"),
                }
            }
            previous_pixels = Some(row.pixels.clone());
            let record = evidence(
                entry,
                var,
                row_load,
                row_contract,
                strategy,
                memory,
                edge,
                &row,
            );
            println!("{}", record.to_json_line().expect("serialize evidence"));
            minted += 1;
        }
    }
    assert_eq!(
        minted,
        ENTRIES.len() * 4,
        "every entry must mint one record per PUBLISHED rung — rung 4 is withheld, and a withheld \
         rung must not mint evidence that could select a fit"
    );
}

/// The numerical contract each rung declares, and the reason it is not uniform.
///
/// Rungs 0/1/3 are **exact**: a residency schedule and a query-row split that leaves every output
/// row's complete k/v and both reductions intact. Rung 2 is **not** — a tiled DC-AE decode blends
/// overlapping tiles and is an approximation of the whole-image decode by construction. Declaring
/// `Exact` for it would be declaring a contract the engine cannot honor, and the harness measured
/// exactly that when an earlier revision tried.
///
/// The tolerance is on the MEAN absolute uint8 subpixel difference, not the max: the max saturates
/// at 255 on this route and therefore bounds nothing. See [`DECODE_TILING_MEAN_ABS_U8`], which is set
/// from a five-latent resample rather than from a single row, and which bounds the SHIPPING route
/// (edge 192, overlap 48) only.
fn parity_contract(strategy: MemoryStrategy) -> MemoryParityContract {
    match strategy {
        MemoryStrategy::BoundedDecode => MemoryParityContract::Tolerance {
            metric: "mean_abs_u8_subpixel".to_owned(),
            maximum_error: DECODE_TILING_MEAN_ABS_U8,
        },
        _ => MemoryParityContract::Exact,
    }
}

/// The measured drift ceiling for the shipping tiled-decode route (edge 192, overlap 48) against the
/// whole-image decode — **on the MEAN, because the max saturates and carries no information.**
///
/// SC-16783 shipped the tiled decode as the Sequential default and A/B'd Resident against Sequential
/// — both tiled — so the tiled-vs-untiled comparison had never been run. Run here across five
/// production latents (`the_tiled_decode_drift_is_resampled_across_production_latents`):
///
/// | seed | maxD | meanD |
/// |---|---:|---:|
/// | 1234 | 231 | 4.569 |
/// | 7 | **255** | 4.322 |
/// | 99991 | 205 | 4.444 |
/// | 424242 | 190 | 3.786 |
/// | 8675309 | **255** | 4.580 |
///
/// `maxD` hits **255 on two of five latents** — the metric's ceiling, meaning some subpixel goes all
/// the way from one end of the range to the other. A saturated statistic cannot bound anything: any
/// tolerance below 255 fails and 255 itself is a no-op. This is SC-17743's lesson arriving in its
/// strongest form — the extreme-order statistic is not merely noisy here, it is uninformative.
///
/// `meanD` is the resolvable one: **3.786..4.580 over the same five latents, a 1.21x spread**. So the
/// declared contract is on the mean, with headroom above the worst case.
///
/// ## The sc-17863 verdict: ACCEPTED as the shipping Sequential default, and the ceiling stands
///
/// Adjudicated with eyes on real renders, not statistics alone
/// (`the_published_decode_tile_domain_is_swept_against_the_whole_image_decode`, three production
/// latents, `sana_1600m` q4 at 1024²). At 1:1 the shipping route is visually indistinguishable from
/// the whole-image decode in every inspected crop — smooth snow across tile boundaries, and
/// high-frequency fur, where the drift concentrates as perceptually-neutral texture jitter. The
/// failure mode that IS visible — blocky tonal patches — appears only when the blend is removed
/// (overlap 24 px = ZERO blended latent cells), and that configuration also **measurably breaches
/// this ceiling** (meanD 6.31..6.43 > 6.0 on all three latents): the bound fails exactly when the
/// image visibly degrades, which is what makes it a real contract rather than a number.
///
/// The full drift/peak trade, measured (request peak; whole-image decode reads 13.6213 GiB):
///
/// | edge @ overlap 48 | peak GiB | vs whole | meanD (3 seeds) | p99D |
/// |---|---:|---:|---|---:|
/// | 512 | 5.0250 | −63.11% | 2.271..2.410 | 24..28 |
/// | 384 | 4.5315 | −66.73% | 2.595..2.857 | 30..32 |
/// | 256 | 3.3837 | −75.16% | 3.624..3.809 | 33..40 |
/// | **192 (shipping)** | **3.2172** | **−76.38%** | 3.786..4.580 (5 latents) | 39..45 |
///
/// **The OVERLAP, not the edge, is the seam lever** — measured at the shipping edge 192 by widening
/// the admission domain for the probe (the production path refuses these):
///
/// | overlap px (latent cells) | peak GiB | meanD (3 seeds) | p99D |
/// |---|---:|---|---:|
/// | 24 (0 — no blend) | 3.2172 | 6.315..6.427 **breaches the ceiling** | 54..60 |
/// | 48 (1 — shipping) | 3.2172 | 4.322..4.580 | 39..45 |
/// | 64 (2) | 3.2172 | 3.394..3.632 | 33..38 |
/// | 96 (3) | 3.2172 | 3.036..3.278 | 30..35 |
///
/// Overlap 96 removes the residual (8x-amplified-only) seam grid at a request peak IDENTICAL to
/// four decimals — tile area sets the decode transient, and overlap adds tiles, not bigger tiles —
/// but costs +1..4 s of decode wall per 1024² render, which on Sprint's 2-step schedule is a
/// 20..50% wall regression. Larger edges buy fidelity with the ladder's scarce resource (peak:
/// edge 512 is +56% over the floor) and still show a faint amplified seam grid. So the default
/// STAYS at edge 192 / overlap 48: the floor keeps its −76%, the drift has no visible artifact,
/// and the numbers above are the measured menu for any future quality-first move of the route.
///
/// The ceiling therefore stays 6.0: non-saturating metric, 1.31x headroom over the measured worst
/// published-domain row (4.580), and demonstrated able to fail — by the no-blend breach above and
/// by mutation (tightening it to 2.0 reddens the sweep on the first published row; see the sc-17863
/// PR for the run).
const DECODE_TILING_MEAN_ABS_U8: f64 = 6.0;

#[allow(clippy::too_many_arguments)]
fn evidence(
    entry: &str,
    var: &str,
    load: &LoadSpec,
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    strategy: MemoryStrategy,
    memory: GenerationMemory,
    edge: u32,
    row: &Row,
) -> MemoryEvidenceLogRecord {
    MemoryEvidenceLogRecord {
        key: MemoryEvidenceKey {
            model_family: "sana".to_owned(),
            resolved_route: entry.to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: load.precision,
                quant: load.quantize,
                component_precision_floors: &[],
            },
            load_shape: load.load_shape,
            mode: MemoryMode::TextToImage,
            reference_shape: MemoryReferenceShape::None,
            overlay: None,
            geometry: MemoryGeometry {
                width: edge,
                height: edge,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            frames_per_second: None,
            strategy,
            engaged_composition: contract.engaged_composition(strategy),
            // The exact parameters the measured row RAN with, taken from the request block rather
            // than from the declared defaults, so a record can never claim a cell it did not drive.
            parameters: MemoryStrategyParameters {
                decode_tile_edge: memory.tile_vae_decode.then(|| {
                    memory
                        .decode_tile_edge
                        .unwrap_or(mlx_gen_sana::pipeline::DECODE_TILE_EDGE as u32)
                }),
                decode_overlap: memory.tile_vae_decode.then(|| {
                    memory
                        .decode_overlap
                        .unwrap_or(mlx_gen_sana::pipeline::DECODE_OVERLAP as u32)
                }),
                attention_chunk_size: memory.chunk_attention.then(|| {
                    memory
                        .attention_chunk_size
                        .unwrap_or(ms::ATTENTION_CHUNK_SIZE)
                }),
                transformer_window_size: memory.stream_transformer_blocks.then(|| {
                    memory
                        .transformer_window_size
                        .unwrap_or(ms::TRANSFORMER_WINDOW_SIZE)
                }),
                transformer_window_component: memory
                    .stream_transformer_blocks
                    .then_some(ms::TRANSFORMER_WINDOW_COMPONENT),
            },
        },
        // sc-22731: the identity is per (provider, tier, policy), so the DECLARED half is asked of
        // this row's own entry and its own loaded tier rather than of the offload policy alone.
        declared_calibration: MemoryCalibrationIdentity::new(
            ms::production_calibration_fingerprint(entry, load)
                .expect("a shipped SANA route at a shipped tier declares an identity"),
            load.load_shape,
        ),
        observed_calibration: contract
            .calibration
            .clone()
            .expect("SANA declares a calibration identity"),
        // This IS the calibration cell: its measured high-water becomes the table's prediction at
        // this exact key. Out-of-sample validation is a separate evidence record.
        predicted_peak_bytes: row.peak_bytes,
        observed_peak_bytes: row.peak_bytes,
        inference_revision: required_revision("INFERENCE_REVISION"),
        sceneworks_revision: required_revision("SCENEWORKS_REVISION"),
        // PER ENTRY, not global. The two entries are two different HF repositories at two
        // different revisions, so a single `MEMORY_MODEL_REVISION` would make one of the two
        // records name weights it was not measured over — the exact "mismatched evidence" this
        // story's last acceptance criterion is about. Both values come from
        // `scripts/release/verify_model_snapshot.py --inventory-output` over the entry's own
        // snapshot, which is the only tool that binds a pinned fixture to the bytes on disk.
        model_revision: required_revision(&format!("{var}_MODEL_REVISION")),
        model_inventory_sha256: required_sha256(&format!("{var}_INVENTORY_SHA256")),
        harness_version: "inference-sana-memory-ladder-v1".to_owned(),
        output_sha256: format!("{:x}", Sha256::digest(&row.pixels)),
        parity: parity_contract(strategy),
        // **Measured, not deferred.** The declared contract is `Exact` and this run establishes it:
        // the ladder walk asserts every optimized row's `output_sha256` equals its entry's rung-0
        // row, and `output_preservation_is_resampled_across_production_latents` re-establishes it
        // across five production latents. `NotRun` is honest only for a harness that captures an
        // output and leaves comparison to a later verifier; here it would understate evidence this
        // run actually produced. (`mlx-gen-z-image` earns its `Passed` the same way since the
        // sc-17861 sweep: its A/B upgrades `NotRun` records only after asserting byte-identity.)
        parity_result: MemoryParityResult::Passed,
    }
}

fn required_revision(name: &str) -> String {
    let revision =
        std::env::var(name).unwrap_or_else(|_| panic!("set {name} to an exact Git commit"));
    assert!(
        revision.len() == 40
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be an exact lowercase 40-character Git commit"
    );
    revision
}

fn required_sha256(name: &str) -> String {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("set {name} to an exact SHA-256"));
    assert!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must be an exact lowercase 64-character SHA-256"
    );
    value
}

/// **The tiled-decode drift, resampled — because a single `maxD` is not a verdict (SC-17743).**
///
/// `maxD` is an extreme-order statistic over ~3M subpixels and showed a 2.9x seed-to-seed spread on a
/// fixed geometry, so the one-latent 231 the ladder walk prints cannot set a bound on its own. This
/// renders rung 1 and rung 2 at five production latents and reports the three-way class.
///
/// The rung being characterized here is **rung 2**, which SC-16783 shipped and this story does not
/// own. It is measured anyway because [`parity_contract`] has to declare a bound for it, and a bound
/// nobody measured is exactly the "unknown evidence" this epic exists to remove.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn the_tiled_decode_drift_is_resampled_across_production_latents() {
    const SEEDS: [u64; 5] = [1234, 7, 99_991, 424_242, 8_675_309];
    let edge = probe_size();
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);
    warm_up(REPRESENTATIVE, &dir, REPRESENTATIVE_IS_SPRINT, edge);

    let untiled = GenerationMemory {
        stage_residency: true,
        ..Default::default()
    };
    let mut maxima: Vec<(u32, f64)> = Vec::new();
    for seed in SEEDS {
        let mut a = request(REPRESENTATIVE_IS_SPRINT, Some(untiled), edge, steps);
        a.seed = Some(seed);
        let mut b = request(REPRESENTATIVE_IS_SPRINT, Some(rung2()), edge, steps);
        b.seed = Some(seed);
        let whole = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &a,
        );
        let tiled = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &b,
        );
        let max = max_delta(&whole.pixels, &tiled.pixels);
        println!(
            "[sc-15523 rung2 drift seed {seed}] maxD {max}, meanD {:.6}",
            mean_delta(&whole.pixels, &tiled.pixels)
        );
        let _ = max;
        maxima.push((max, mean_delta(&whole.pixels, &tiled.pixels)));
    }
    let worst_max = maxima.iter().map(|(m, _)| *m).max().unwrap_or(0);
    let saturated = maxima.iter().filter(|(m, _)| *m == 255).count();
    let worst_mean = maxima.iter().map(|(_, m)| *m).fold(0f64, f64::max);
    let best_mean = maxima.iter().map(|(_, m)| *m).fold(f64::MAX, f64::min);
    let class = if worst_mean == 0.0 {
        "ADMISSIBLE"
    } else if worst_mean <= DECODE_TILING_MEAN_ABS_U8 {
        "BOUNDED"
    } else {
        "FAILS"
    };
    println!(
        "[sc-15523 rung2 drift] tiled vs whole-image decode over {} latents: maxD saturates at 255 \
         on {saturated}/{} (worst {worst_max}) and bounds NOTHING; meanD \
         {best_mean:.3}..{worst_mean:.3} ({:.2}x spread) => {class} against the declared ceiling \
         {DECODE_TILING_MEAN_ABS_U8}",
        SEEDS.len(),
        SEEDS.len(),
        worst_mean / best_mean
    );
    assert!(
        worst_mean <= DECODE_TILING_MEAN_ABS_U8,
        "the shipping tiled-decode route drifted meanD {worst_mean:.4} past its declared ceiling \
         {DECODE_TILING_MEAN_ABS_U8}; the declared parity contract no longer bounds the engine"
    );
}

/// The `q`-quantile of the absolute uint8 subpixel difference — the tail statistic that, unlike
/// `maxD`, does NOT saturate on this route (sc-17863). Computed exactly from a 256-bin histogram.
fn quantile_delta(a: &[u8], b: &[u8], q: f64) -> u32 {
    assert_eq!(a.len(), b.len(), "pixel buffers differ in length");
    let mut histogram = [0u64; 256];
    for (x, y) in a.iter().zip(b) {
        histogram[x.abs_diff(*y) as usize] += 1;
    }
    let need = (q * a.len() as f64).ceil() as u64;
    let mut seen = 0u64;
    for (delta, count) in histogram.iter().enumerate() {
        seen += count;
        if seen >= need {
            return delta as u32;
        }
    }
    255
}

/// The fraction of subpixels whose absolute difference exceeds `threshold`.
fn fraction_above(a: &[u8], b: &[u8], threshold: u8) -> f64 {
    let count = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.abs_diff(**y) > threshold)
        .count();
    count as f64 / a.len() as f64
}

/// Dump one RGB8 buffer as a binary PPM into `SANA_SWEEP_OUT` (no-op when unset) so the sweep's
/// verdict can be made with eyes on the renders rather than on statistics alone (sc-17863).
fn dump_ppm(name: &str, edge: u32, pixels: &[u8]) {
    let Ok(dir) = std::env::var("SANA_SWEEP_OUT") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create SANA_SWEEP_OUT");
    let mut data = format!("P6\n{edge} {edge}\n255\n").into_bytes();
    data.extend_from_slice(pixels);
    std::fs::write(dir.join(format!("{name}.ppm")), data).expect("write ppm");
}

/// The per-subpixel absolute difference, amplified 8x and clamped — the seam-structure image.
fn amplified_diff(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b)
        .map(|(x, y)| (u32::from(x.abs_diff(*y)) * 8).min(255) as u8)
        .collect()
}

/// **The tiled-decode drift/peak trade, swept over the PUBLISHED domain (sc-17863).**
///
/// SC-16783 shipped edge 192 / overlap 48 as the Sequential default and sc-15523's resample bounded
/// that one cell. This sweeps every published edge at the fixed 48-pixel overlap against the
/// whole-image decode, per production latent, so the declared ceiling bounds the DOMAIN rather than
/// one cell — and so the edge-vs-overlap question is answered by measurement:
///
/// * the EDGE is the lever: drift falls monotonically as the tile grows (more decoder context per
///   tile), while the request peak rises with tile area;
/// * the OVERLAP is quantized by the 32x DC-AE scale (`overlap_px / 32` latent cells — 48 px is ONE
///   blended latent cell), so the unpublished probe overlaps below document the refusal of the
///   out-of-domain values rather than silently measuring an unshippable configuration.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn the_published_decode_tile_domain_is_swept_against_the_whole_image_decode() {
    const SEEDS: [u64; 3] = [1234, 7, 8_675_309];
    /// Unpublished overlaps at the shipping edge: 24 px is ZERO blended latent cells (a hard seam),
    /// 64 and 96 px are two and three. Outside the published `{48}` domain, so the production path
    /// refuses them — rows print REFUSED unless the domain is deliberately widened for a probe.
    const PROBE_OVERLAPS: [u32; 3] = [24, 64, 96];
    let edge = probe_size();
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);
    warm_up(REPRESENTATIVE, &dir, REPRESENTATIVE_IS_SPRINT, edge);

    let untiled = GenerationMemory {
        stage_residency: true,
        ..Default::default()
    };
    let registry = mlx_gen_sana::provider_registry().expect("provider registry");
    for seed in SEEDS {
        let mut whole_req = request(REPRESENTATIVE_IS_SPRINT, Some(untiled), edge, steps);
        whole_req.seed = Some(seed);
        let whole = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &whole_req,
        );
        dump_ppm(&format!("seed{seed}_whole"), edge, &whole.pixels);
        println!(
            "[sc-17863 sweep seed {seed}] whole-image decode: {:.4} GiB, {:.0} ms/step",
            whole.peak_gib,
            ms_per_step(&whole, steps)
        );
        let mut by_edge: Vec<(u32, f64)> = Vec::new();
        for tile_edge in ms::DECODE_TILE_EDGES {
            let memory = GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(*tile_edge),
                decode_overlap: Some(mlx_gen_sana::pipeline::DECODE_OVERLAP as u32),
                ..Default::default()
            };
            let mut req = request(REPRESENTATIVE_IS_SPRINT, Some(memory), edge, steps);
            req.seed = Some(seed);
            let row = measure(
                REPRESENTATIVE,
                &dir,
                DEFAULT_TIER,
                LoadShape::DeferredMaterialization,
                &req,
            );
            let mean = mean_delta(&whole.pixels, &row.pixels);
            println!(
                "[sc-17863 sweep seed {seed}] edge {tile_edge:>3} overlap 48: {:.4} GiB \
                 ({:+.2}% vs whole), maxD {}, meanD {mean:.4}, p99D {}, >8/255 {:.2}%, >32/255 \
                 {:.2}%, {:.0} ms/step",
                row.peak_gib,
                100.0 * (row.peak_gib - whole.peak_gib) / whole.peak_gib,
                max_delta(&whole.pixels, &row.pixels),
                quantile_delta(&whole.pixels, &row.pixels, 0.99),
                100.0 * fraction_above(&whole.pixels, &row.pixels, 8),
                100.0 * fraction_above(&whole.pixels, &row.pixels, 32),
                ms_per_step(&row, steps)
            );
            dump_ppm(
                &format!("seed{seed}_edge{tile_edge}_overlap48"),
                edge,
                &row.pixels,
            );
            dump_ppm(
                &format!("seed{seed}_edge{tile_edge}_overlap48_diff8x"),
                edge,
                &amplified_diff(&whole.pixels, &row.pixels),
            );
            by_edge.push((*tile_edge, mean));
            // The declared tolerance bounds the DOMAIN, not one cell: a caller may select any
            // published edge, so every published edge must honor the contract's ceiling.
            assert!(
                mean <= DECODE_TILING_MEAN_ABS_U8,
                "published edge {tile_edge} drifted meanD {mean:.4} past the declared ceiling \
                 {DECODE_TILING_MEAN_ABS_U8} on seed {seed}"
            );
        }
        // The lever's direction, pinned at every step: `by_edge` descends 512..192, so drift must
        // be non-decreasing across each adjacent pair — a smaller tile gives the decoder less
        // context per tile and may not drift LESS than its larger neighbor. The measured gaps
        // (meanD 2.27 -> 2.85 -> 3.61 -> 4.58 across 512/384/256/192) are an order of magnitude
        // wider than the observed envelope variance (~1.24%), so this cannot flake on a re-run.
        for pair in by_edge.windows(2) {
            let (larger, smaller) = (pair[0], pair[1]);
            assert!(
                larger.1 <= smaller.1,
                "edge {} (meanD {:.4}) should bound edge {} (meanD {:.4}) from below on seed \
                 {seed} — the drift/peak trade inverted between adjacent published edges",
                larger.0,
                larger.1,
                smaller.0,
                smaller.1
            );
        }
        // Overlap probe at the shipping edge. Published domain is {48}: these rows REFUSE on the
        // production path, and that refusal is the record. Widening the domain for a measurement
        // probe admits them, which is how sc-17863's overlap-vs-edge answer was measured — and an
        // admitted row is still bound by the declared ceiling (sc-18249), so a widened domain
        // cannot quietly ship a configuration the contract does not cover.
        for overlap in PROBE_OVERLAPS {
            let memory = GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(mlx_gen_sana::pipeline::DECODE_TILE_EDGE as u32),
                decode_overlap: Some(overlap),
                ..Default::default()
            };
            let mut req = request(REPRESENTATIVE_IS_SPRINT, Some(memory), edge, steps);
            req.seed = Some(seed);
            let model = registry
                .load(
                    REPRESENTATIVE,
                    &spec(&dir, DEFAULT_TIER, LoadShape::DeferredMaterialization),
                )
                .expect("load sana");
            clear_cache();
            reset_peak_memory();
            let started = std::time::Instant::now();
            match model.generate(&req, &mut |_: Progress| {}) {
                Ok(GenerationOutput::Images(images)) => {
                    let pixels = &images.first().expect("one image").pixels;
                    let peak = get_peak_memory() as f64 / GIB;
                    let mean = mean_delta(&whole.pixels, pixels);
                    println!(
                        "[sc-17863 overlap probe seed {seed}] edge {} overlap {overlap}: \
                         {peak:.4} GiB, maxD {}, meanD {mean:.4}, p99D {}, {:.0} ms/step",
                        mlx_gen_sana::pipeline::DECODE_TILE_EDGE,
                        max_delta(&whole.pixels, pixels),
                        quantile_delta(&whole.pixels, pixels, 0.99),
                        started.elapsed().as_secs_f64() * 1000.0 / f64::from(steps)
                    );
                    dump_ppm(
                        &format!("seed{seed}_edge192_overlap{overlap}"),
                        edge,
                        pixels,
                    );
                    dump_ppm(
                        &format!("seed{seed}_edge192_overlap{overlap}_diff8x"),
                        edge,
                        &amplified_diff(&whole.pixels, pixels),
                    );
                    // An admitted row is measured AND bound, never exempt (sc-18249). On the
                    // production path this arm is unreachable — the published domain is {48} and
                    // the Err arm below asserts that exact refusal — so reaching it at all means
                    // the domain enforcement was widened or lost. The ceiling then has to hold
                    // here too, and it binds by construction: overlap 24, the FIRST probe in
                    // `PROBE_OVERLAPS`, measures meanD 6.315..6.427 on all three seeds (see
                    // [`DECODE_TILING_MEAN_ABS_U8`]) — above the 6.0 ceiling — so a lost refusal
                    // panics on the first admitted row rather than printing numbers into a log
                    // nothing reads. The renders are dumped BEFORE this assertion so a breach
                    // still leaves its visual evidence behind.
                    assert!(
                        mean <= DECODE_TILING_MEAN_ABS_U8,
                        "admitted overlap probe {overlap} on seed {seed} drifted meanD {mean:.4} \
                         past the declared ceiling {DECODE_TILING_MEAN_ABS_U8}; an off-domain row \
                         generated AND breached the contract the domain refusal exists to protect"
                    );
                }
                Ok(other) => panic!("expected images, got {other:?}"),
                Err(error) => {
                    // Only the published-domain refusal counts as REFUSED. Any other error (an
                    // OOM, a snapshot failure) must fail the probe rather than masquerade as the
                    // domain rejection this row exists to record.
                    let message = error.to_string();
                    let refusal = format!(
                        "decode overlap is {}, got {overlap}",
                        mlx_gen_sana::pipeline::DECODE_OVERLAP
                    );
                    assert!(
                        message.contains(&refusal),
                        "overlap probe {overlap} on seed {seed} failed with something other than \
                         the published-domain refusal (expected \"{refusal}\" in the message): \
                         {message}"
                    );
                    println!(
                        "[sc-17863 overlap probe seed {seed}] overlap {overlap}: REFUSED \
                         ({message})"
                    );
                }
            }
            drop(model);
            clear_cache();
        }
    }
}

/// **A quality verdict is not a single image (SC-17743).**
///
/// `max delta` is an extreme-order statistic over ~3M subpixels and showed a 2.9x seed-to-seed
/// spread on a fixed geometry, so a one-seed comparison cannot classify a rung. Both SANA rungs
/// claim EXACT preservation, which makes this a strong check rather than a tolerance: resampled
/// across five production latents, every seed must be byte-identical, and a single non-zero seed
/// makes the class `FAILS` rather than being averaged away.
#[test]
#[ignore = "needs a real SANA snapshot (see the module docs for the env vars)"]
fn output_preservation_is_resampled_across_production_latents() {
    const SEEDS: [u64; 5] = [1234, 7, 99_991, 424_242, 8_675_309];
    let edge = probe_size();
    let dir = require_tier(REPRESENTATIVE_ENV, DEFAULT_TIER);
    let steps = steps_for(REPRESENTATIVE_IS_SPRINT);
    warm_up(REPRESENTATIVE, &dir, REPRESENTATIVE_IS_SPRINT, edge);

    let mut deltas = Vec::new();
    for seed in SEEDS {
        let mut control_req = request(REPRESENTATIVE_IS_SPRINT, Some(rung2()), edge, steps);
        control_req.seed = Some(seed);
        let mut full_req = request(
            REPRESENTATIVE_IS_SPRINT,
            Some(rung3(ms::ATTENTION_CHUNK_SIZE)),
            edge,
            steps,
        );
        full_req.seed = Some(seed);
        let control = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &control_req,
        );
        let full = measure(
            REPRESENTATIVE,
            &dir,
            DEFAULT_TIER,
            LoadShape::DeferredMaterialization,
            &full_req,
        );
        let max = max_delta(&control.pixels, &full.pixels);
        let mean = mean_delta(&control.pixels, &full.pixels);
        println!("[sc-15523 quality seed {seed}] maxD {max}, meanD {mean:.6}");
        deltas.push(max);
    }
    let worst = deltas.iter().copied().max().unwrap_or(0);
    let class = if worst == 0 {
        "ADMISSIBLE"
    } else if worst <= 1 {
        "UNRESOLVED"
    } else {
        "FAILS"
    };
    println!(
        "[sc-15523 quality] rung 3 over {} production latents: worst maxD {worst} => {class}",
        SEEDS.len()
    );
    assert_eq!(
        worst, 0,
        "rung 3 claims EXACT preservation over its published domain ({class} over {deltas:?}); a \
         non-zero seed means it is an arithmetic change, not a memory schedule"
    );
}
