//! Measured component sizes, stage peaks and rung evidence the contract declares.

use super::*;

/// The DiT block count, mirrored from `MiniMaxH3DitConfig::default().num_layers`.
pub const DIT_BLOCKS: u32 = 50;

// --- measured asset facts -------------------------------------------------------------------
//
// Exact `.safetensors` bytes under each component directory of the upstream bf16 snapshot
// (`MiniMaxAI/MiniMax-H3` @ `939557dc`), which is the same accounting
// `gen_core::safetensors_path_bytes` performs at load. These are on-disk footprints, NOT resident
// peaks — see the module docs for the stage-attributed peaks.

/// Qwen3-VL-32B text encoder — 14 shards, the conditioning component. 66.71 GB.
pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;

/// One 33 B DiT partition at bf16 — 14 shards. 66.28 GB.
///
/// `transformer` and `transformer_ref` are byte-identical in size and a render loads **exactly
/// one**, so this is charged once (`crate::model::MiniMaxH3Task` resolves which).
pub const DIT_BF16_BYTES: u64 = 66_280_504_216;

/// Video VAE — 3 shards; the decoder is a 36-layer transformer, not a conv stack. 10.42 GB.
pub const VIDEO_VAE_BYTES: u64 = 10_415_558_888;

/// Audio VAE — 1 shard, DAC-lineage encoder plus BigVGAN decoder. 0.61 GB.
pub const AUDIO_VAE_BYTES: u64 = 605_429_340;

// --- measured stage peaks -------------------------------------------------------------------
//
// MLX active bytes, decimal GB, measured to 0.01 GB. These are NOT contract fields: the contract
// has no place for a measured stage peak (that is calibration evidence, sc-17153). They live here
// so the attribution is greppable, testable and cannot silently drift.

/// **Conditioning stage.** The dense Qwen3-VL-32B text encoder measured in isolation: 53.07 GB.
/// This, not the DiT, is the ~53 GB floor.
pub const CONDITIONING_STAGE_PEAK_BYTES: u64 = 53_070_000_000;

/// **Conditioning stage, packed `q4` tier** — 14.43 GB (sc-19120), against the 53.07 GB dense
/// datum. A **38.64 GB** reduction, and the single largest memory change in this model.
///
/// Measured by `tests/te_tier_real_weights.rs` on the stage in its own process, with the encoder
/// forced through a forward first: MLX mmaps lazily, and a bare 66 GB load leaves the peak at
/// ~33 KB. The same run's `get_active_memory` is 14.19 GB against the encoder's own
/// `nbytes()` accounting of 14.15 GB, so essentially nothing dense survives the tiering.
///
/// End to end at `384x224 / 124 frames`, per stage (`tests/te_tier_generate_stages.rs`):
///
/// | stage | dense TE | q8 TE | q4 TE |
/// |---|---:|---:|---:|
/// | conditioning | **52.80** | **26.91** | **14.68** |
/// | DiT load + AdaLN precompute | 12.66 | 12.66 | 12.66 |
/// | denoise (q4 DiT) | 12.75 | 12.75 | 12.75 |
/// | decode | 5.77 | 5.77 | 5.77 |
/// | **process** | **52.80** | **26.91** | **14.68** |
///
/// Two things that table settles. The DiT column is finally *legible* — every later stage was
/// sitting under a 40 GB shadow before. And the conditioning stage is still the tallest one even
/// packed, but by 1.93 GB rather than 40.05 GB, so the model's floor is now the DiT's residency
/// rather than its conditioner's.
///
/// **This is not the 12 GB ComfyUI footprint and does not claim to be.** The whole DiT is still
/// held resident through the denoise; bounding *that* is sc-18662's block streaming, and the two
/// stories are a pair — neither reaches consumer hardware alone.
pub const CONDITIONING_STAGE_PEAK_Q4_BYTES: u64 = 14_430_000_000;

/// **Conditioning stage, packed `q8` tier** — 26.94 GB in isolation (26.91 GB end to end).
///
/// Sits between the two: half the dense stage, and still above the q4 DiT's 12.75 GB denoise, so
/// q8 leaves the conditioner as the binding stage by a wide margin where q4 does not.
pub const CONDITIONING_STAGE_PEAK_Q8_BYTES: u64 = 26_940_000_000;

/// **Denoise stage.** DiT weights resident through the denoise loop at bf16, after the AdaLN
/// precompute-and-evict: 40.43 GB — 12.64 GB *below* the conditioning stage's mark.
pub const DENOISE_RESIDENT_BF16_BYTES: u64 = 40_430_000_000;

/// **Denoise stage.** The same residency at q4: 11.63 GB. The 28.80 GB delta against bf16 is the
/// measured tiering win, and it is real — the peak looked flat only because the conditioning stage
/// set the high-water mark first.
pub const DENOISE_RESIDENT_Q4_BYTES: u64 = 11_630_000_000;

/// **Denoise stage.** Exact bytes the AdaLN precompute-and-evict drops (64.56 → 38.70 GB active):
/// the 50-block stack's `adaln_proj` at bf16. Asserted against the loader in
/// `crate::dit::adaln`. Both actives are measured in `tests/adaln_evict_real_weights.rs`, which
/// force-materializes the whole stack first; the render path does not, so neither is a render-time
/// resident.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

/// **The same 50 projections on the packed `q8` tier — 13.83 GB.**
///
/// [`ADALN_EVICTED_BYTES`] is a **bf16** figure and the lever shrinks with the tier exactly as the
/// DiT that contains it does (`crate::dit::block::AdaLnProjection::nbytes`). Derived rather than
/// measured: the private `adaln_stack_bytes` re-derives it from the shipped configuration and
/// `the_packed_tier_stack_sizes_are_the_loaders_own_accounting` holds the two together.
///
/// The 13.02 GB `crate::quant` quotes for this tier is the **code buffer only**; this figure also
/// carries the per-group bf16 scale and bias the packed triple cannot run without, which is what
/// `crate::quant::nbytes` sums and therefore what is actually resident.
pub const ADALN_EVICTED_Q8_BYTES: u64 = 13_828_147_200;

/// **The same 50 projections on the packed `q4` tier — 7.33 GB.**
///
/// The figure `tests/common/mod.rs`'s (g) section quotes as "≈7.3 GB (≈6.5 GB if the group metadata
/// is not counted)". See [`ADALN_EVICTED_Q8_BYTES`] for why the group metadata is counted here.
pub const ADALN_EVICTED_Q4_BYTES: u64 = 7_325_337_600;

/// **Denoise stage.** Bytes of the modulation table the precompute keeps in the projections' place,
/// at the **longest schedule this model admits** — `MODULATION_PARAMS · modulation_rows ·
/// hidden_size · DIT_BLOCKS` elements at bf16, with `modulation_rows` read off a real
/// [`crate::model::MAX_STEPS`]-evaluation schedule rather than derived by hand
/// (`the_retained_table_is_the_worst_case_over_the_admitted_schedule`).
///
/// **The evict is not free, and this is the price.** [`ADALN_EVICTED_BYTES`] is what the projections
/// cost; this is what replaces them. The contract declares the *net* difference, because declaring
/// the gross figure claims a saving the runtime does not deliver — the exact overstatement
/// `the_declared_saving_does_not_exceed_the_measured_residency_drop` refuses.
///
/// **Why the worst case rather than the default schedule.** The table is linear in the schedule's
/// distinct-timestep count and independent of resolution and duration (nothing in it has a token
/// axis — `AdaLnCache::bytes`). The contract is static and has no request to read a step count
/// from, so it must pick one point of that line. Under-declaring what the precompute keeps
/// over-declares the saving, and an over-declared saving turns a suppressed configuration into an
/// OOM; over-declaring it only leaves some of the win on the table. The asymmetry picks the end.
///
/// **This figure does NOT scale with the tier, and that asymmetry is the whole reason the two
/// quantities are declared separately.** The table is the projections' *output*, so its dtype is the
/// **compute** dtype — `crate::quant::compute_dtype`, which reads bf16 off a packed tier's own
/// scales — not the tier's bit width. `tests/common/mod.rs`'s (f) section states the same fact from
/// the measurement side ("tier-free … the retained table, whose dtype is the *compute* dtype …
/// rather than the tier's bit width"). So the resident side falls 26.02 → 13.83 → 7.33 GB across
/// bf16/q8/q4 while this stays at 3.87 GB, and applying one factor to both would be wrong at every
/// tier but bf16.
pub const ADALN_MODULATION_TABLE_MAX_BYTES: u64 = 3_870_720_000;

/// Contract-stable identity of the evictable AdaLN sub-stack, so a consumer can find the
/// declaration by name rather than by matching on bytes.
pub const ADALN_COMPONENT_ID: &str = "dit_adaln_proj_stack";

// --- rung 2: the bounded-decode domain ---------------------------------------------------------

/// The **only** decode tile edge this provider admits, in output pixels.
///
/// Derived from [`crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE`] rather than re-typed, so a change
/// to the reference geometry cannot leave the contract publishing a domain the decode no longer
/// executes. It is a singleton because the tile edge is an **output-correctness** input copied from
/// the published model (sc-18786), not a tunable — see the module docs.
pub const DECODE_TILE_EDGE: u32 = crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE as u32;

/// The **only** decode tile overlap this provider admits, in output pixels. Derived from
/// [`crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP`] for the same reason as [`DECODE_TILE_EDGE`].
///
/// A zero overlap is the tile-starvation failure mode this rung must not be able to select: it
/// abuts the tiles with no cross-fade, and the corruption shows up **across frames** rather than
/// within one, so no memory number can see it. `validate_ranges` independently forbids a zero
/// entry in a published range, so the domain cannot be widened to include it by accident.
pub const DECODE_OVERLAP: u32 = crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP as u32;

// --- rung 3: the measured inapplicability -------------------------------------------------------

/// **The rung-3 verdict's evidence, as data rather than prose** (sc-18661).
///
/// Denoise-phase MLX peak under each bounded attention configuration, as a fraction of the same
/// geometry's unbounded peak, measured on the real 50-block `q4` DiT by
/// `tests/bounded_attention_real.rs`. Positive means the rung **cost** peak.
///
/// Rows are `(tier, frames, packed_seq_len, head-axis @ 64 Mi, query-row @ 64 Mi, head-axis
/// one-head)` — **tier-keyed since the sweep was extended to all three** (the q4-only table it
/// started as made the bf16/q8 verdict an argument rather than a measurement).
/// Both ends of the `17n + 5` frame lattice are present because `frames` is an exact-match evidence
/// key (`mlx_fit_gate.rs:1381`) and fitted-curve extrapolation scales by area only — a single
/// duration cell may not stand in for another one.
///
/// The third column is the configuration whose budget is derived from the geometry (`B · Sq²`, one
/// head per chunk with the query axis whole). It is measured because the shipped 64 Mi operating
/// point is **Z-Image's**, and at 56 heads it stops being bit-exact above `Sq = 8192` — inside this
/// family's legal duration range. Without that column the verdict would be about a budget rather
/// than about the rung.
/// Measured 2026-08-15 at 576x320 on the `SceneWorks/minimax-h3-mlx` `q4` transformer
/// (`f22bc294`), two evaluations per arm, `tests/bounded_attention_real.rs`:
///
/// | frames | seq | unbounded | heads @ 64 Mi | query rows @ 64 Mi | heads @ one-head |
/// |---:|---:|---:|---:|---:|---:|
/// | 124 | 7 586 | 13.4894 GB | 13.5614 GB (+0.53 %) | 13.5640 GB (+0.55 %) | 13.5666 GB (+0.57 %) |
/// | 345 | 20 022 | 16.4951 GB | 16.7053 GB (+1.27 %) | 16.7123 GB (+1.32 %) | 16.7194 GB (+1.36 %) |
///
/// **The cost grows with the sequence**, which is the direction that settles the verdict: the axis
/// the rung exists to relieve is the one on which it gets worse. Wall-clock moves the same way at the
/// shipped budget — +1.6 %/+15.6 % at 124 frames, +44.3 %/+71.7 % at 345.
///
/// Output agreement was **exactly zero** relative max-abs on all six bounded arms, so nothing here is
/// a quality trade that memory could be bought with.
pub const RUNG3_MEASURED_PEAK_DELTAS: [(&str, i32, i32, f64, f64, f64); 4] = [
    ("q4", 124, 7_586, 0.005341, 0.005532, 0.005723),
    ("q4", 345, 20_022, 0.012740, 0.013168, 0.013596),
    ("q8", 124, 7_586, 0.003065, 0.003174, 0.003284),
    ("q8", 345, 20_022, 0.007931, 0.008197, 0.008464),
];

/// **`bf16` is measured but UNRESOLVED, and is deliberately absent from
/// [`RUNG3_MEASURED_PEAK_DELTAS`] rather than recorded with a number.**
///
/// The q4 and q8 sweeps agree with each other and with the verdict: every bounded arm costs peak,
/// the cost grows with sequence, and the magnitude shrinks as the tier's resident baseline grows
/// (+0.53%/+1.27% at q4 against +0.31%/+0.79% at q8). Extrapolating that trend to bf16 predicts a
/// small positive delta.
///
/// **It measured NEGATIVE.** At 124 frames the unbounded arm peaked at 42.3451 GB and all three
/// bounded arms at 42.145-42.151 GB — a **-0.47%** delta, i.e. chunking appearing to *save* peak.
/// That is the sign the verdict says cannot happen, so it is recorded here rather than declared
/// either way.
///
/// Two readings, and this constant exists because they have not been separated:
///
/// * **an ordering artifact.** The three bounded arms agree with each other to 0.01% while the
///   unbounded arm — which runs FIRST in the process — sits 0.47% above all of them. That is the
///   shape of a first-arm allocator transient, and it is a known hazard on this exact comparison:
///   `tests/sequence_cost_real.rs` documents the same harness reporting **+0.2% first-in-binary and
///   +50.3% under a filter** for one pair of arms, and adds a warm-up arm for it. This harness
///   reloads the 62 GB DiT per arm and clears the cache between them, which is stronger — but it has
///   no unmeasured warm-up arm, and at bf16 the load is 3.7x the q4 one it was validated at.
/// * **a real bf16 effect.** The dtype is the same on every tier's activations, but the *weights*
///   are not, and the graph a dense DiT builds differs from a packed one's.
///
/// Resolving it needs a warm-up arm plus an arm-order permutation on this tier; until then the
/// contract's inapplicability reason quotes q4 and q8 only, and `rung3_cells_for_tier("bf16")`
/// returns empty so the real-weight harness fails loudly rather than validating against a guess.
/// **The claim in PR #649 that q4 bounds the other tiers from above is contradicted at bf16/124f
/// and must not be relied on.**
pub const RUNG3_BF16_IS_UNRESOLVED: &str =
    "bf16 measured -0.47% at 124 frames (42.3451 GB unbounded against 42.145-42.151 GB bounded);      unseparated first-arm allocator transient vs a real dense-tier effect";

/// The rung-3 cells measured for one tier.
pub fn rung3_cells_for_tier(tier: &str) -> Vec<(i32, i32, f64, f64, f64)> {
    RUNG3_MEASURED_PEAK_DELTAS
        .iter()
        .filter(|(t, ..)| *t == tier)
        .map(|(_, frames, seq, a, b, c)| (*frames, *seq, *a, *b, *c))
        .collect()
}

/// The audio VAE is out of scope for rung 2, declared rather than implied.
///
/// It is 0.61 GB against the video VAE's 10.42 GB (17× smaller), and structurally has nothing to
/// tile: `crate::audio_vae` is a DAC-lineage 1-D stack whose decode is a BigVGAN vocoder over a
/// waveform. There is no spatial grid, so the `decode_tile_edge` / `decode_overlap` domain has no
/// meaning for it and the rung's mechanism does not run on that path. Named as a constant so the
/// boundary is greppable and testable rather than a sentence in a doc comment.
pub const AUDIO_VAE_IS_OUT_OF_SCOPE_FOR_TILING: bool = true;

/// The load shape a **resident** load has, kept as the provider's default and its calibration
/// identity's shape.
///
/// Until sc-18662 this was the only shape this provider had, because `MiniMaxH3Dit::load_dir` builds
/// the whole 50-block stack no matter what a caller asks for. It is no longer: `load_dir_deferred`
/// and `MiniMaxH3TextEncoder::from_dir_deferred` implement
/// [`LoadShape::DeferredMaterialization`] for both transformers, so the contract now **honours the
/// spec's shape** through `resolved_load_shape` instead of pinning one.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

// --- rung 4: the bounded transformer-residency domain -------------------------------------------

/// The **only** transformer window size this provider admits: one block or layer at a time.
///
/// **The singleton is the finding, not a placeholder** — the same shape rung 2's
/// [`DECODE_TILE_EDGE`] has, for its own separate reason. What makes it the right point is that
/// **1 is the measured residency floor**, not that the parameter is inert.
///
/// # The inertness claim was wrong, and it was wrong because it was measured on one tier (sc-17153)
///
/// This constant used to say the window parameter is *measured inert* above 1, citing a **1.09x**
/// spread across `[1, 5, 10, 50]`. That spread was taken on the **dense** encoder, where a single
/// Qwen3 layer's f32 upcast dominates the peak at every window size and flattens the curve
/// (re-measured: 1.39x). It does not describe the encoder a shipped tiered render loads. Measured
/// on the packed tiers, per window, on real weights:
///
/// | tier | resident | window 1 | window 5 | window 10 | window 50 | spread |
/// | --- | ---: | ---: | ---: | ---: | ---: | ---: |
/// | `bf16` (dense) | 53.07 GB | **4.50 GB** | 6.24 GB | 6.24 GB | 6.24 GB | 1.39x |
/// | `q8` | 26.98 GB | **1.43 GB** | 2.65 GB | 2.78 GB | 3.10 GB | 2.17x |
/// | `q4` | 14.43 GB | **0.80 GB** | 1.78 GB | 1.95 GB | 1.87 GB | 2.45x |
///
/// The packed spread runs 2.17-3.12x across runs (sc-17153's terminal campaign independently
/// measured 3.12x on `q4`), so the parameter is a **real lever** on every encoder a tiered render
/// uses, and the retired `< 2.0` spread assertion in `tests/block_window_real.rs` was asserting
/// something false about them.
///
/// What holds on **every encoder tier measured**, with a wide margin, is that window 1 is the
/// minimum: 4.50 GB against 6.24 (dense), 1.43 against 2.65-3.10 (`q8`), 0.80 against 1.78-1.95
/// (`q4`). The curve above 1 is not strictly monotonic — `q4` runs 1.78 -> 1.95 -> 1.87 across
/// windows 5/10/50, so the ordering *among the larger windows* is counter noise — but the floor is
/// separated from all of them by 1.9x or more. That is what justifies publishing the singleton:
/// `[1]` is the cheapest point of a real curve, and the other values would advertise a lever that
/// trades memory *away*.
///
/// `tests/block_window_real.rs::measure_te_arm` asserts the floor (with the harness's 256 MiB / 5 %
/// margin, not as a strict ordering) and bounds the spread, on **both TE arms**. The DiT arm is
/// deliberately not covered by that claim: it measures resident against window 1 only, so whether
/// 1 is *its* floor is unmeasured — see [`RUNG4_DIT_MEASURED_PEAKS`].
///
/// # ⚠ The DENSE encoder has been observed returning an all-zero context — UNFIXED
///
/// Five sightings in ~32 real-weight forwards on this hardware (sc-17153): identical shapes, clean
/// exit, no Metal diagnostic, `max = 0.0` where the correct answer is `1.55e4`. What the sightings
/// have in common is now specific enough to be actionable:
///
/// * **Every sighting is the `bf16` dense encoder.** Zero sightings across ~14 `q4` / `q8` arms.
/// * **It is NOT windowing.** Three sightings were *resident* forwards, two windowed — including
///   two resident ones after this module's loader fix, on a path that never goes near
///   `from_dir_deferred`. Rung 4 neither causes it nor is exposed to it more than the resident
///   path is, and the loader fix neither introduced nor removed it.
/// * **It is NOT the window size.** Reversing the order of `[1, 5, 10, 50]` moved the zero from
///   window 1 to window 50.
/// * **It IS the second multi-gigabyte encoder load in one process, reading cold.** Four of the
///   five fit exactly that: two dense resident loads that followed the packed arm in the same test
///   binary (167.2 s and 166.9 s cold — both zero), against two that were the *first* load in
///   their process (156.0 s and 72.6 s cold — both correct), and twelve warm loads at any position
///   (all correct). The two windowed sightings are the same shape: a window that followed a 53 GB
///   resident load in the same process. The one outlier was a 2-layer probe taken while ~20
///   compilers were running.
///
/// The mechanism is below this crate and is **unattributed**; nothing here claims to have removed
/// it. Two things this crate does about it:
///
/// 1. `tests/block_window_real.rs` compares the context against a reference on every arm and
///    refuses a degenerate reference — that assertion caught sightings four and five live. The peak
///    counter reads *lower* when the work does not happen, so no memory assertion in this rung can
///    see it, and neither can a norm, a cosine or a checksum.
/// 2. That file now **requires** each arm to be its own process invocation, which is what its own
///    module text had always claimed and what `--test-threads=1` does not provide.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// **The TE arm** (sc-18662, AC3), measured on the real Qwen3-VL-32B tap by
/// `tests/block_window_real.rs`: resident bytes, and the window-1 floor.
///
/// 53.07 GB -> 4.50 GB, an **11.79x** reduction and the single largest memory change this rung
/// produces on this model. The resident figure independently reproduces
/// [`CONDITIONING_STAGE_PEAK_BYTES`]'s 53.07 GB from a different harness, which is what makes it
/// attributable to the encoder rather than to a process-wide mark.
///
/// Output was **bit-identical** (`0.000e0` relative max-abs) at every window size, so nothing is
/// traded for it — but read the fault note on [`TRANSFORMER_WINDOW_SIZE`] before treating
/// "bit-identical" as a property of the path rather than of the runs that were measured.
///
/// # The windowed figure moved with the mlx-rs pin (sc-17153): 5.81 GB -> 4.50 GB
///
/// `RUNG4_TE_WINDOWED_PEAK_BYTES` was declared at `5_814_845_376`. It does not reproduce: eight
/// runs on real weights land at `4_501_440_448` B ± ~0.5 MB, 1.31 GB (22.6 %) below the declared
/// value and far outside `common::assert_declared_peak_constant`'s bound — which is the
/// per-constant absolute bound sc-17153 introduced precisely so a stale member of a pair could not
/// hide inside a quotient. The resident member is unchanged and reproduces to 2.4 MB, so this was
/// exactly the *single-member* drift a ratio band cancels.
///
/// **The cause is the MLX pin, not this crate.** `git diff` from the sc-18662 measurement
/// (`19efeb265`) to the current tree over `src/text_encoder/`, `src/block_stream.rs` and the
/// harness is doc- and comment-only — nothing in the crate accounts for a 22.6 % drop. What moved
/// is `Cargo.toml`'s mlx-rs revision, `932beb4e6` -> **`7151a9b27`**, in `dfd76b5ae`
/// ("fix(mlx): preserve global VAE normalization in tiled decode", sc-19753). So:
///
/// * this figure is **pinned to mlx-rs `7151a9b27`**, and an MLX pin bump is the event that should
///   be expected to move it again — when it does, the assertion will red with a *stale constant*
///   message, and the pin is the first thing to check rather than this crate's loaders;
/// * it also independently rules out the alternative reading, that 4.50 GB is what a *degenerate*
///   (all-zero) run happens to peak at. It is not: the zero-output runs peak lower still, and
///   `tests/block_window_real.rs` compares output before it records a cell, so no run that
///   produced zeros can contribute a peak to this constant.
pub const RUNG4_TE_RESIDENT_PEAK_BYTES: u64 = 53_074_721_216;

/// The TE arm at [`TRANSFORMER_WINDOW_SIZE`]. See [`RUNG4_TE_RESIDENT_PEAK_BYTES`].
pub const RUNG4_TE_WINDOWED_PEAK_BYTES: u64 = 4_501_440_448;

/// **The DiT arm**, per tier: `(tier, resident_peak_bytes, window_1_peak_bytes)`.
///
/// Measured by `tests/block_window_real.rs` over the 50-block stack at `seq = 4096`, walking block
/// bodies with no modulation and no rotary — the thinnest call that touches every one of a block's
/// ten body tensors, so the ratio is weight residency rather than activation.
///
/// | tier | resident | window 1 | ratio |
/// |---|---:|---:|---:|
/// | `q4` | 12.00 GB | 1.36 GB | **8.85x** |
/// | `q8` | 21.64 GB | 1.58 GB | **13.73x** |
/// | `bf16` | 39.58 GB | 1.90 GB | **20.82x** |
///
/// The ratio scales with the tier — 8.85x / 13.73x / 20.82x across q4 / q8 / bf16 — because the
/// *window* is the same activation-plus-one-block at every tier while the resident baseline is not.
/// That is the rung behaving exactly as declared, and it is the **opposite** of rung 3's tier story
/// on this model, where the transient the rung acts on is bf16 regardless of tier so the lower tier
/// was the favourable one. The two rungs' tier arguments do not transfer to each other.
///
/// Both resident figures independently reproduce the provider's own stage constants from a different
/// harness — [`DENOISE_RESIDENT_Q4_BYTES`]'s 11.63 GB and [`DENOISE_RESIDENT_BF16_BYTES`]'s 40.43 GB
/// — which is what makes them attributable to the DiT rather than to a process-wide mark. Output was
/// bit-identical at every window on every tier.
pub const RUNG4_DIT_MEASURED_PEAKS: [(&str, u64, u64); 3] = [
    ("q4", 12_002_937_856, 1_355_656_192),
    ("q8", 21_636_046_848, 1_575_493_632),
    ("bf16", 39_578_684_416, 1_901_221_376),
];

/// **The rung-4 REQUEST peak** (sc-18662): the max over one whole streamed render's stage peaks,
/// measured on the shipped `generate` by `tests/streamed_generate_real.rs` — q4 DiT tier, q4
/// packed text encoder, 124 frames at 384x224, 4 steps, window 1, `TransformerComponent::Both`.
///
/// This is the number the per-arm cells above cannot stand in for: they measure each phase in its
/// own process window, while the SceneWorks survey's `requestPeak` axis asks about the composition
/// — and its own note says a static proxy will not do. The resident request's high-water is the
/// conditioning stage's mark ([`CONDITIONING_STAGE_PEAK_BYTES`], 53.07 GB dense); the streamed
/// request pins at the decode floor instead, because conditioning and denoise are both windowed
/// and the VAE decode is untouched by this rung (rung 2 owns it).
///
/// Measured per stage: conditioning 0.78 GB, dit-load + adaln-precompute 2.03 GB, denoise
/// 2.03 GB, decode **5.80 GB** — so the request is decode-bound, which is what the phase
/// envelope's `max(TE, DiT, decode)` arithmetic predicted (5.77 GB) before the request ran.
///
/// **Graded to ±290 MB, not to a ±25% band** (sc-17153). `tests/streamed_generate_real.rs` puts
/// this constant through `common::assert_declared_peak_constant`, whose tolerance is the larger of
/// 256 MiB and 5 % of the declared value — 5 % wins here, at 290_218_452 B. That grader replaced the
/// `|measured/declared − 1| < 0.25` ratio band, which was multiplicatively asymmetric and, on a
/// quotient of two constants, blind to pair-proportional drift.
///
/// The retired "~1 GB of run-to-run spread on this machine" figure was **retracted** with it: it was
/// not reproduced on either arm of the sc-17153 terminal campaign (the q4 DiT windowed peak came
/// back bit-identical across two runs, its resident peak moved 0.36 %). Do not size a new bound
/// from it. The grading is TIGHTER than the retired comment claimed, so a red here is a real drift
/// or a pin bump — `assert_declared_peak_constant`'s own docs carry the two limits of the bound.
///
/// Only graded when the run is q4 DiT **and** q4 packed TE, the conditions this figure was captured
/// under; the harness prints that it asserted nothing when either differs.
pub const RUNG4_REQUEST_PEAK_Q4_BYTES: u64 = 5_804_369_044;
