//! **sc-15325 — is the LTX video VAE's tiled decode faithful to a single pass?**
//!
//! **Answer, measured: at the one bucket measured, yes — LTX did not reproduce the z16 defect.** This
//! test is kept as the standing evidence for that verdict, and as the guard that would catch it
//! changing. Read the scope carefully: the clearing is **empirical and narrow**, the mechanism is
//! **unexplained**, and post-fix LTX cannot reach the starved candidates anyway (see "How far this
//! verdict actually goes" below).
//!
//! ## Why it was asked
//!
//! The sibling z16 Wan VAE was shipping a temporal decode window of **2 latent frames**, which starves
//! its temporal convolutions and corrupts the *content* of every tile — 24.4/255 mean abs err against
//! a single-pass decode of the same latents, with 30.8 % of a worst frame blown to white. LTX was
//! explicitly **not cleared** by that work: its `temporal_scale` is 8, so the `(24, 8)` entry in
//! `LTX_VAE_TEMPORAL_FR` is a latent tile of **3**, below the latent-4 tile that still measured
//! 6.4/255 on z16. The reasoning was sound; it was simply never measured.
//!
//! ## What the measurement says (640×384 × 89 frames, q8, smooth source)
//!
//! ⚠️ **This table is a dated snapshot** (2026-07, one host, one tier, one source clip), reproduced
//! verbatim in `sceneworks-gen-core`'s `MIN_TEMPORAL_TILE_LATENT_FRAMES` doc and in
//! `mlx-gen-ltx::pipeline`'s `ltx_tiling_never_selects_a_starved_temporal_tile`. Numbers in a doc
//! comment cannot go red. What is actually *asserted* — and therefore what the "6.6× less error for
//! 17 % more peak" argument rests on — is the **ratio** between the latent-3 and latent-8 rows, gated
//! in `ltx_tiled_decode_tracks_single_pass` below (`err8 < err3 · 0.5`, `peak8 ≤ peak3 · 1.5`).
//! Treat the absolute values here as illustration; treat the ratio as the contract.
//!
//! | temporal tile | latent tile / overlap | mean abs err vs single-pass | clipping mean / worst | MLX active peak |
//! |---|---|---|---|---|
//! | single-pass | — | 0 (reference) | 0.00 % / 0.00 % | 10.68 GiB |
//! | 24 / 8 — the old smallest candidate | 3 / 1 | **1.73 /255** | 0.00 % / 0.00 % | 7.34 GiB |
//! | 48 / 16 | 6 / 2 | 0.50 | 0.00 % / 0.00 % | 8.31 GiB |
//! | 64 / 16 | 8 / 2 | 0.26 | 0.00 % / 0.00 % | 8.57 GiB |
//! | 64 / 32 | 8 / 4 | 0.16 | 0.00 % / 0.00 % | 9.49 GiB |
//!
//! LTX degraded **gracefully** where z16 collapses: at its worst candidate it is 1.73/255 with zero
//! highlight clipping, against z16's 24.4/255 and 30.8 % clipping at a comparable starvation. (The
//! zero-clipping half of that comes partly from the stimulus — see below.)
//!
//! ## The mechanism is UNKNOWN — and the obvious explanation is wrong
//!
//! The tempting story is that LTX's temporal tiling is **causal**, so each tile is handed a preceding
//! context frame (`split_temporal`'s `starts[i] -= 1`). That cannot be the reason:
//!
//!  * `causal_temporal` is **also `true` for `VaeTiling::WAN22`** (z48, shipping). A causal-VAE
//!    explanation is not LTX-specific — it would predict z48 is equally immune, which nobody has
//!    measured.
//!  * At **matched context** the gap survives. z16's `4 / 2` row is a 4-latent tile with 1 latent of
//!    overlap; an LTX interior tile at `24 / 8` is a 3-latent tile with 1 latent of overlap **plus**
//!    1 causal context frame — the same 4 latent frames of input. z16 reads 6.4/255 there, LTX
//!    1.73/255: a **3.7× gap that context-counting does not explain**.
//!
//! Something else is responsible — decoder depth, channel width, the ×8 vs ×4 temporal scale, the
//! latent distribution itself. It has not been identified, and no replacement mechanism is asserted
//! here. **The verdict is measured; the explanation is absent.** Anyone deciding whether to make the
//! floor per-VAE should start from that, not from a causal-tiling story.
//!
//! ## How far this verdict actually goes
//!
//! Narrower than "LTX is fine":
//!
//!  * **One bucket, and one that never tiles in production.** 640×384 × 89 frames peaks ~10.7 GiB
//!    single-pass, so `auto_tiling_budgeted_ltx` returns `Ok(None)` at any realistic budget — this
//!    test has to *force* the tiling config by hand to measure it at all. Nothing here says anything
//!    about 1080p × 241, which is where LTX genuinely tiles.
//!  * **The source cannot clip.** `smooth_video`'s channel amplitudes are 0.70/0.70/0.60 on [−1, 1],
//!    never near the 0.9608 that reads as ≥250/255. LTX's "0.00 % / 0.00 %" clipping is therefore
//!    **partly structural** — the stimulus had little headroom to blow out. The mean-|Δ| column is
//!    the load-bearing one; the clipping column mostly says the source was well-behaved.
//!  * **It is also moot post-fix.** The floor removes `(48, 16)` and `(24, 8)` from the selectable
//!    set, so production LTX cannot reach a starved tile regardless of how this verdict is read.
//!
//! Stated plainly: **cleared empirically at one low-dynamic-range bucket, mechanism unexplained, and
//! unreachable in production anyway.**
//!
//! ## Why the shared floor still applies to LTX
//!
//! Not as a defect fix — as a cheap quality win. Moving from the old smallest candidate to the floor's
//! smallest survivor is **6.6× less error for 17 % more decode memory** (1.73 → 0.26 /255, 7.34 →
//! 8.57 GiB), and on a tight budget the selector simply tiles further *spatially* rather than refusing.
//! Reverting the floor for LTX specifically would buy ~1.2 GiB and cost most of that accuracy; it is
//! not worth a second policy.
//!
//! ```text
//! LTX_VAE_DIR=/path/to/ltx-2.3-mlx/q8 \
//!   cargo test -p mlx-gen-ltx --release --test vae_decode_tiling_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::tiling::{
    TemporalTiling, TilingConfig, VaeTiling, MIN_TEMPORAL_TILE_LATENT_FRAMES,
    MIN_TEMPORAL_TILE_LATENT_OVERLAP,
};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_ltx::{LtxVaeConfig, LtxVideoVae};
use mlx_rs::Array;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy().to_string();
        match s.strip_prefix("~/") {
            Some(rest) => PathBuf::from(std::env::var("HOME").unwrap()).join(rest),
            None => PathBuf::from(s),
        }
    })
}

fn env_i32(var: &str, default: i32) -> i32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// A smooth (band-limited) moving source clip as `[1, 3, F, H, W]` in [-1, 1]. Deliberately smooth: a
/// VAE cannot reproduce a hard discontinuity, so a sawtooth source would bury the tiling error under
/// the VAE's own round-trip error and make the comparison meaningless.
fn smooth_video(f: i32, h: i32, w: i32) -> Array {
    let mut v = Vec::with_capacity((3 * f * h * w) as usize);
    for c in 0..3 {
        for t in 0..f {
            let p = t as f32 * 0.21;
            for y in 0..h {
                let fy = y as f32 / h as f32;
                for x in 0..w {
                    let fx = x as f32 / w as f32;
                    v.push(match c {
                        0 => 0.70 * (fx * 6.0 + p).sin(),
                        1 => 0.70 * (fy * 5.0 - p).cos(),
                        _ => 0.60 * ((fx + fy) * 4.0 + p * 0.5).sin(),
                    });
                }
            }
        }
    }
    Array::from_slice(&v, &[1, 3, f, h, w])
}

/// Per-pixel mean |Δ| between two decoded videos, expressed on the 0-255 scale a viewer sees (the
/// decode is roughly [-1, 1], so ×127.5). MLX is lazily evaluated — both sides are `eval`ed before the
/// read-back, or the comparison silently reads an unmaterialized graph.
fn mean_abs_delta_255(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    let m = mlx_rs::ops::mean(&d, None).unwrap();
    mlx_rs::transforms::eval([&m]).unwrap();
    m.item::<f32>() as f64 * 127.5
}

/// Percentage of pixels clipped to near-white (>= 250/255, i.e. >= 0.9608 on the [-1, 1] scale), as
/// `(mean over frames, worst frame)`. This is the metric that separates a starved tile from a healthy
/// one — mean |Δ| can stay small while a handful of frames blow out.
fn clip_stats(v: &Array) -> (f64, f64) {
    let sh = v.shape().to_vec();
    let (f, hh, ww) = (sh[2], sh[3], sh[4]);
    mlx_rs::transforms::eval([v]).unwrap();
    let flat = v.as_slice::<f32>();
    let per_frame = (hh as usize) * (ww as usize);
    let mut pcts = Vec::with_capacity(f as usize);
    for t in 0..f as usize {
        let mut clipped = 0usize;
        for i in 0..per_frame {
            // channels-first: [1, 3, F, H, W]
            let base = t * per_frame + i;
            let stride = (f as usize) * per_frame;
            let mx = flat[base]
                .max(flat[base + stride])
                .max(flat[base + 2 * stride]);
            if mx >= 0.9608 {
                clipped += 1;
            }
        }
        pcts.push(100.0 * clipped as f64 / per_frame as f64);
    }
    let mean = pcts.iter().sum::<f64>() / pcts.len().max(1) as f64;
    (mean, pcts.iter().cloned().fold(0.0f64, f64::max))
}

fn temporal(tile: i32, overlap: i32) -> TilingConfig {
    TilingConfig {
        spatial: None,
        temporal: Some(TemporalTiling {
            tile_frames: tile,
            overlap_frames: overlap,
        }),
    }
}

#[test]
#[ignore = "needs an ltx-2.3-mlx snapshot with vae_encoder + vae_decoder (LTX_VAE_DIR); GPU-heavy"]
fn ltx_tiled_decode_tracks_single_pass() {
    let Some(dir) = env_path("LTX_VAE_DIR") else {
        panic!("set LTX_VAE_DIR to a snapshot dir holding vae_encoder/vae_decoder.safetensors");
    };
    // LTX VAE: spatial ×32, temporal ×8 **causal** ⇒ F = 1 + 8·k, H/W multiples of 32.
    let w = env_i32("LTX_W", 640);
    let h = env_i32("LTX_H", 384);
    let f = env_i32("LTX_FRAMES", 89); // 12 latent frames — room for a latent-8 tile to actually tile
    assert_eq!((f - 1) % 8, 0, "LTX_FRAMES must be 1 + 8·k (got {f})");
    assert_eq!(h % 32, 0, "LTX_H must be a multiple of 32");
    assert_eq!(w % 32, 0, "LTX_W must be a multiple of 32");

    // ⚠️ The single-pass reference is only valid below the write cap — past it MLX writes silently
    // wrong pixels and every tiled candidate would look catastrophic against a corrupt reference
    // (sc-15402). LTX's `full_res_channels` is 8, so this is enormously slack here, but assert it
    // rather than assume it: this guard is the reason the z16 numbers are trustworthy.
    let cap = VaeTiling::LTX.writable_frame_cap(h, w);
    assert!(
        f as i64 <= cap,
        "needs a valid single-pass reference: {f} output frames exceeds the LTX write cap {cap} at \
         {w}x{h}"
    );

    let cfg = LtxVaeConfig::from_model_dir(&dir).expect("read LtxVaeConfig (embedded_config.json)");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
    let enc = Weights::from_file(dir.join("vae_encoder.safetensors")).expect("vae_encoder");
    let vae = LtxVideoVae::from_weights(&dec, Some(&enc), &cfg).expect("build LtxVideoVae");

    let latent = {
        let video = smooth_video(f, h, w);
        let z = vae.encode(&video).expect("VAE encode");
        mlx_rs::transforms::eval([&z]).expect("materialize the latent");
        z
    };
    let ls = latent.shape().to_vec();
    let t_lat = ls[2];
    println!(
        "=== LTX tiled-decode parity: {w}x{h}x{f} out, latent [z{}, T{t_lat}, {}, {}] ===",
        ls[1], ls[3], ls[4]
    );
    mlx_rs::memory::clear_cache();

    let decode = |cfg: Option<&TilingConfig>| -> (Array, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = match cfg {
            Some(c) => vae
                .decode_tiled(&latent, c, &CancelFlag::default())
                .expect("tiled decode"),
            None => vae.decode(&latent).expect("single-pass decode"),
        };
        // MLX is lazy: without this the peak reads ~0 and the comparison reads an unbuilt graph.
        mlx_rs::transforms::eval([&out]).expect("materialize the decode");
        let peak = mlx_rs::memory::get_peak_memory();
        (out, peak)
    };

    let (single, single_peak) = decode(None);
    let (s_cm, s_cw) = clip_stats(&single);
    println!(
        "  single-pass (reference): clipping {s_cm:.2}% / {s_cw:.2}%, peak {:.2} GiB",
        gib(single_peak)
    );

    // Every entry the candidate grid held before sc-15325, plus the latent-8 tile the floor now
    // guarantees. `(96, 24)` is latent 12 = the whole sequence here, so it is a no-op and omitted.
    let ladder: [(i32, i32, &str); 4] = [
        (24, 8, "the old smallest candidate — latent 3 / 1"),
        (48, 16, "latent 6 / 2 — also removed by the floor"),
        (64, 16, "latent 8 / 2 — the floor's smallest survivor"),
        (64, 32, "latent 8 / 4 — the floor's overlap policy"),
    ];
    // (latent tile, mean abs err, worst-frame clipping %, MLX active peak GiB)
    let mut err_at: Vec<(i32, f64, f64, f64)> = Vec::new();
    for (tile, overlap, note) in ladder {
        let (out, peak) = decode(Some(&temporal(tile, overlap)));
        let d = mean_abs_delta_255(&single, &out);
        let (cm, cw) = clip_stats(&out);
        println!(
            "  tile {tile:>3} / overlap {overlap:>2} (latent {} / {}) [{note}]: mean abs err \
             {d:.2}/255, clipping {cm:.2}% / {cw:.2}%, peak {:.2} GiB",
            tile / 8,
            overlap / 8,
            gib(peak)
        );
        err_at.push((tile / 8, d, cw, gib(peak)));
    }

    let at = |lat: i32| -> (f64, f64, f64) {
        err_at
            .iter()
            .filter(|(l, ..)| *l == lat)
            .fold((f64::MAX, 0.0, f64::MAX), |a, b| {
                if b.1 < a.0 {
                    (b.1, b.2, b.3)
                } else {
                    a
                }
            })
    };
    let (err3, clip3, peak3) = at(3);
    let (err8, _, peak8) = at(8);

    // 1. **The verdict, at the width the evidence supports.** At THIS bucket LTX does not reproduce
    //    the sc-15325 defect: even its most starved candidate stays far from the z16 collapse
    //    (24.4/255 with 30.8 % worst-frame clipping) on both metrics. This is the assertion that
    //    carries the finding — if LTX ever starts behaving like z16 here, the "cleared" verdict in the
    //    story, in `MIN_TEMPORAL_TILE_LATENT_FRAMES`'s doc and in `mlx-gen-ltx::pipeline` is wrong and
    //    this goes red rather than the regression shipping.
    //
    //    What it does NOT establish (see the module header): a mechanism, anything about buckets that
    //    actually tile in production, or that a high-dynamic-range source would clip as little. The
    //    clipping assertion below is the weaker of the two precisely because this source's amplitudes
    //    (0.70/0.70/0.60 on [-1, 1]) never approach the 0.9608 clip threshold — near-zero clipping is
    //    partly structural here. `err3` is the load-bearing number.
    assert!(
        err3 < 5.0,
        "the old latent-3 LTX tile is {err3:.2}/255 from single-pass — LTX now looks like the z16 \
         defect, and this test's (and sc-15325's) 'LTX is cleared' verdict must be revisited"
    );
    assert!(
        clip3 <= s_cw + 1.5,
        "latent-3 LTX clipping is {clip3:.2}% against {s_cw:.2}% single-pass — the highlight blowout \
         that defines the z16 defect has appeared on LTX"
    );

    // 2. **Why the shared floor still applies here anyway**: it is a cheap quality win, not a defect
    //    fix. Pin both halves of that trade, because the trade is the justification — if the accuracy
    //    gain stops being large, or the memory cost stops being small, keeping one policy across both
    //    VAE families needs re-arguing.
    //
    //    ⚠️ **Deliberate trip-wire, and it fires on a GOOD change too.** `err8 < err3 · 0.5` is a
    //    cost/benefit assertion, not a correctness one: it goes red if the floor's survivor regresses
    //    *or* if the removed latent-3 candidate ever gets BETTER (a decoder improvement, a new tier, a
    //    changed blend). Both readings are worth a human look, which is why it is a hard assert rather
    //    than a println — but a red here is not automatically a bug. If it fires, read the printed
    //    ladder first: if `err3` improved rather than `err8` regressing, the correct response is to
    //    re-argue whether LTX still wants the shared floor (and possibly make it per-VAE), NOT to
    //    loosen this number until it passes.
    assert!(
        err8 < err3 * 0.5,
        "the floor's smallest survivor ({err8:.2}/255) is no longer materially better than the \
         removed latent-3 candidate ({err3:.2}/255) — the floor is buying nothing on LTX"
    );
    assert!(
        peak8 <= peak3 * 1.5,
        "the floor costs {:.0}% more LTX decode memory ({peak3:.2} -> {peak8:.2} GiB) — it was \
         adopted here on the strength of being nearly free",
        100.0 * (peak8 / peak3 - 1.0)
    );

    // 3. And the floor is actually wired into the grid this engine uses.
    let planned = mlx_gen_ltx::pipeline::auto_tiling_budgeted_ltx(h, w, f)
        .expect("the LTX decode must remain feasible");
    if let Some(t) = planned.and_then(|c| c.temporal) {
        let lat = t.tile_frames / 8;
        assert!(
            lat >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
            "the LTX planner still emits a {lat}-latent-frame temporal tile"
        );
        assert!(
            (t.overlap_frames / 8).min(lat - 1) >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
            "the LTX planner's temporal overlap is under the blend floor"
        );
    }
}
