//! sc-19120 — **the conditioning stage measured in isolation, per tier**, on real weights.
//!
//! `#[ignore]`d: needs a real `text_encoder/` component (61.85 GB dense, or a built tier) and
//! Metal.
//!
//! ```sh
//! # one stage, one tier, one process — the peak is only meaningful per process
//! MINIMAX_H3_TE=<text_encoder dir> \
//!   cargo test -p mlx-gen-minimax-h3 --test te_tier_real_weights -- --ignored --nocapture \
//!   --test-threads=1 --exact conditioning_stage_peak_is_the_staged_tier
//!
//! # the packed tier against the dense one it was built from, in one process
//! MINIMAX_H3_TE_DENSE=<dense text_encoder> MINIMAX_H3_TE_PACKED=<tier text_encoder> \
//!   cargo test -p mlx-gen-minimax-h3 --test te_tier_real_weights -- --ignored --nocapture \
//!   --test-threads=1 --exact the_packed_context_tracks_the_dense_one
//! ```
//!
//! # Why this is a per-stage measurement and not a render
//!
//! Every tier of this model measured an identical ~53 GB *generate* peak, and three separate
//! explanations were recorded for it before the right one: the conditioning stage runs **first**,
//! `reset_peak_memory()` fires before `generate`, and the dense text encoder's high-water masks
//! every later stage. A process-wide peak set by a tier-independent stage reads as a defect in the
//! tiered one. So this measures one stage, in its own process, with nothing else loaded — and
//! [`mlx_gen_minimax_h3::text_encoder::map_shards`] is the production mapping, not a copy of it.
//!
//! # Three hazards, all of which have produced a wrong number in this epic
//!
//! 1. **MLX mmaps lazily.** A bare 66 GB load leaves the peak at ~33 KB. Nothing here reports a
//!    figure that has not been through [`force`], which runs the encoder and evaluates its output.
//! 2. **`get_active_memory` is ACTIVE, not cache.** A `clear_cache` that "succeeds" can leave
//!    buffers in the allocator's cache, so [`drain`] retries and watches *active* fall.
//! 3. **`apply_gpu_memory_limit` is a no-op**, so there is no ceiling to lean on — the numbers are
//!    what they are.

use std::path::{Path, PathBuf};

use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use mlx_rs::{Array, Dtype};

use mlx_gen_minimax_h3::text_encoder::{
    map_shards, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, LM_PREFIX, SELECT_HIDDEN,
};

/// The dense conditioning-stage high-water this story exists to lower —
/// [`CONDITIONING_STAGE_PEAK_BYTES`](mlx_gen_minimax_h3::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES).
const DENSE_STAGE_PEAK_BYTES: u64 =
    mlx_gen_minimax_h3::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES;

fn env(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// Return everything MLX is holding to the system, retrying until *active* stops falling.
///
/// One `clear_cache` reports success while buffers migrate active → cache and sit there; the
/// distinguishing observation is `get_active_memory` itself, not the call's return.
fn drain() {
    let mut previous = usize::MAX;
    for _ in 0..8 {
        clear_cache();
        let active = get_active_memory();
        if active >= previous {
            break;
        }
        previous = active;
    }
}

/// Run the encoder and evaluate its output — the only thing that materializes a lazily mmapped
/// component. Returns the context so the caller can keep it alive across the measurement.
fn force(te: &MiniMaxH3TextEncoder, ids: &Array, mask: &Array) -> Array {
    let context = te.forward(ids, mask).expect("te forward");
    mlx_rs::transforms::eval([&context]).expect("eval");
    context
}

/// A short, fixed presentation. The binding stage's cost is a function of prompt tokens only, so a
/// short probe keeps activations negligible against the weights and the number reported is the
/// **residency**, which is what a tier moves.
fn probe(cfg: &MiniMaxH3TeConfig) -> (Array, Array) {
    let ids: Vec<i32> = (0..24).map(|i| (1000 + i * 37) % cfg.vocab_size).collect();
    let n = ids.len() as i32;
    (
        Array::from_slice(&ids, &[1, n]),
        Array::from_slice(&vec![1i32; ids.len()], &[1, n]),
    )
}

/// On-disk bytes of the shards the `t2va` window actually maps — shards 1-12. Reported alongside
/// the resident figure so the tolerance below is against a quantity a reader can check with `ls`.
fn mapped_shard_bytes(dir: &Path) -> u64 {
    mlx_gen_minimax_h3::text_encoder::TE_SHARDS
        .map(|i| dir.join(format!("model-{i:05}-of-00014.safetensors")))
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum()
}

/// **AC 1 + AC 2.** Load the staged text encoder, force materialization, and report the
/// conditioning stage's peak and residency — then assert the residency matches what the loaded
/// encoder says it is holding.
///
/// The assertion is against [`MiniMaxH3TextEncoder::nbytes`] rather than against the tier's
/// directory size, because the `t2va` window maps shards that carry layers 50-58 it never builds:
/// the directory is larger than the stage by a quantity that has nothing to do with the tier. What
/// must hold is that **active tracks what was actually built** — if something dense were still
/// resident that the tier believed it had packed, active would exceed `nbytes` by that amount.
#[test]
#[ignore = "needs a real text_encoder component and Metal"]
fn conditioning_stage_peak_is_the_staged_tier() {
    let dir = env("MINIMAX_H3_TE").expect("MINIMAX_H3_TE=<text_encoder dir>");
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    let (ids, mask) = probe(&cfg);

    drain();
    reset_peak_memory();
    let baseline = get_active_memory();

    let w = map_shards(&dir, false).expect("map text-encoder shards");
    let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg).expect("build encoder");

    // The lazy-mmap trap: everything up to here can leave the peak in the kilobytes.
    let lazy_peak = get_peak_memory();
    let context = force(&te, &ids, &mask);

    let peak = get_peak_memory();
    let active = get_active_memory();
    let cache = get_cache_memory();
    let accounted = te.nbytes();
    let bits = te.packed_bits().expect("uniform width");

    println!("── conditioning stage ─────────────────────────────────────────");
    println!("  component      {}", dir.display());
    println!(
        "  tier           {bits:?} (token table packed: {})",
        te.token_table_is_quantized()
    );
    println!("  layers built   {}", te.num_loaded_layers());
    println!(
        "  mapped shards  {:.2} GB on disk",
        gb(mapped_shard_bytes(&dir) as usize)
    );
    println!("  encoder holds  {:.2} GB (nbytes)", gb(accounted));
    println!(
        "  peak           {:.2} GB   (pre-force {:.4} GB)",
        gb(peak),
        gb(lazy_peak)
    );
    println!(
        "  active         {:.2} GB   cache {:.2} GB",
        gb(active),
        gb(cache)
    );
    println!("  baseline       {:.2} GB", gb(baseline));
    println!(
        "  dense datum    {:.2} GB",
        DENSE_STAGE_PEAK_BYTES as f64 / 1e9
    );
    println!("  context        {:?}", context.shape());

    assert_eq!(
        te.num_loaded_layers(),
        SELECT_HIDDEN,
        "the layer-50 tap must survive tiering"
    );
    // **A two-sided band, and the lower half is the one that matters.**
    //
    // The obvious guard — "the peak grew by orders of magnitude after forcing" — is blind, and was
    // caught being blind by mutation: delete the `eval` inside `force` and *nothing* materializes,
    // so `lazy_peak` and `peak` are both a few MB and any ratio between them still passes. What
    // cannot be faked is the residency landing on the encoder's own accounting:
    //
    // * **below** the band ⇒ the component never materialized (an unforced lazy mmap reads ~33 KB
    //   against a 14-50 GB `nbytes`), so every figure this test prints would be fiction;
    // * **above** it ⇒ something dense is resident that the tier believed it had packed.
    //
    // Measured ratios sit at 1.001-1.010 across dense / q8 / q4, so the band is wide either way.
    let (floor, ceiling) = (0.90, 1.20);
    let ratio = active as f64 / accounted as f64;
    assert!(
        ratio > floor,
        "active {:.2} GB is only {ratio:.3}x the encoder's accounted {:.2} GB — the component was \
         never materialized, so nothing here was measured (MLX mmaps lazily; `force` must eval)",
        gb(active),
        gb(accounted)
    );
    assert!(
        ratio < ceiling,
        "active {:.2} GB exceeds the encoder's accounted {:.2} GB by {ratio:.3}x — something dense \
         is resident that the tier believed it had packed",
        gb(active),
        gb(accounted)
    );
    assert!(
        peak >= active && peak > lazy_peak,
        "peak {peak} must bound the residency and exceed the pre-force {lazy_peak}"
    );

    if bits.is_some() {
        assert!(
            te.token_table_is_quantized(),
            "a packed tier that left the 1.56 GB token table dense is a mis-built tier"
        );
        assert!(
            (peak as u64) < DENSE_STAGE_PEAK_BYTES,
            "a packed tier must peak below the {:.2} GB dense datum, got {:.2} GB",
            DENSE_STAGE_PEAK_BYTES as f64 / 1e9,
            gb(peak)
        );
    }
}

/// **AC 4.** The packed context against the **real dense text encoder's** context, not against a
/// fixture.
///
/// A fixture generated from reference modules cannot validate a loader reading a converted
/// checkpoint — golden and loader can share a layout and both disagree with the shipped weights.
/// So the reference here is the dense component itself, run in this same process on the same ids,
/// released before the packed one is mapped so the two never co-reside.
///
/// The gate is **relative max-abs-diff**. Not a cosine (scale-invariant: a tier that uniformly
/// halved every activation would score 1.0), not a norm (averages a local catastrophe away over
/// 5120 channels), not a checksum (answers a question nobody asked). Seven checks in this epic were
/// blind for exactly those reasons.
#[test]
#[ignore = "needs both a dense and a packed text_encoder component, and Metal"]
fn the_packed_context_tracks_the_dense_one() {
    let dense_dir = env("MINIMAX_H3_TE_DENSE").expect("MINIMAX_H3_TE_DENSE=<dense text_encoder>");
    let packed_dir = env("MINIMAX_H3_TE_PACKED").expect("MINIMAX_H3_TE_PACKED=<tier text_encoder>");
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    let (ids, mask) = probe(&cfg);

    let context_of = |dir: &Path| -> (Vec<f32>, Option<i32>) {
        drain();
        let w = map_shards(dir, false).expect("map shards");
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg).expect("build encoder");
        let bits = te.packed_bits().expect("uniform width");
        // ── sc-17153: DO NOT "fix" this by retrying, `#[ignore]`ing, or loosening it ─────────────
        //
        // The **bf16 dense** encoder intermittently returns an all-zero forward — `max|out| = 0.0`,
        // correct shapes, no Metal error — at a measured ~13% of forwards (4 sightings in ~30).
        // The mechanism is below this crate; see `src/text_encoder/degeneracy.rs` for the evidence
        // and the non-causes ruled out by counter-measurement.
        //
        // `force()` runs the real encoder, so when the hazard fires the dense arm dies **here**, on
        // the encoder's own degeneracy refusal, with a message naming sc-17153. That is
        // **deliberate and wanted**: this test is `#[ignore]`d and runs only on the dispatch-only
        // real-weights lane, never as a PR gate, so an occasional honest red costs nothing and is
        // the only incidence signal the investigation still gets. A retry, a relaxed bound, or a
        // silenced arm would destroy it and leave the hazard shipping unobserved.
        //
        // If you are here because this failed: record the run (`max|out|`, tier, resident vs
        // windowed, whether a repeat in a fresh process reproduces) against sc-17153, then re-run.
        let context = force(&te, &ids, &mask);
        let host: Vec<f32> = context
            .as_dtype(Dtype::Float32)
            .expect("f32")
            .as_slice::<f32>()
            .to_vec();
        drop((context, te, w));
        drain();
        (host, bits)
    };

    let (reference, dense_bits) = context_of(&dense_dir);
    assert_eq!(
        dense_bits, None,
        "MINIMAX_H3_TE_DENSE is not a dense component"
    );
    let (packed, packed_bits) = context_of(&packed_dir);
    let bits = packed_bits.expect("MINIMAX_H3_TE_PACKED is not a packed tier");

    assert_eq!(reference.len(), packed.len(), "context shapes differ");
    // No `assert!(scale > 0.0)` here any more: the encoder's own sc-17153 screen refuses a zero
    // context inside `context_of`'s `force()`, so a zero reference can no longer reach this line.
    // An assertion that cannot fail is a trap — it reads as protection and provides none.
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = reference
        .iter()
        .zip(&packed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let relative = worst / scale;

    println!("── hidden_states[{SELECT_HIDDEN}] parity ──────────────────────");
    println!("  tier                    q{bits}");
    println!("  rows x hidden           {}", reference.len());
    println!("  max|ref|                {scale:.6}");
    println!("  max|Δ|                  {worst:.6}");
    println!("  relative max-abs-diff   {relative:.6}");

    // The two halves that matter, and they pull in opposite directions:
    assert!(
        relative > 0.0,
        "the packed context is bit-identical to the dense one — nothing was packed"
    );
    // A q4 conditioning context that has drifted by more than a quarter of its own dynamic range is
    // not a tier, it is a different prompt. Stated as a ceiling rather than a target: the number
    // this actually measures is recorded on sc-19120.
    let ceiling = if bits == 4 { 0.25 } else { 0.10 };
    assert!(
        relative < ceiling,
        "q{bits} relative max-abs-diff {relative} exceeds the {ceiling} ceiling"
    );
}
