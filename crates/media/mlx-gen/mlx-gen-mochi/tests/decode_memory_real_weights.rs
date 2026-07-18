//! sc-12291: **measure** the real Mochi AsymmVAE decode — peak memory and correctness, untiled vs
//! chunked. This is the story's evidence artifact; the numbers in `vae.rs`'s docs come from here.
//!
//! sc-11991 (B1) declared `mlx.minMemoryGb: 96` from a *derived* decode peak, and B2 (sc-11992) built
//! its pre-flight fit gate on a derived formula. Both disclosed the derivation as unmeasured. This
//! measures it: it loads only the real AsymmVAE decoder (~0.86 GiB — no DiT, no T5), decodes
//! real-geometry 848×480 latents, and reports `get_peak_memory` per path.
//!
//! **Two things it pins that no golden can.** The `vae_parity` golden is dumped at 64×64/7 frames
//! (6.3e6 elements); production is 848×480/151 frames (8.13e9). That gap hid a real bug:
//!  1. the untiled decode's peak grows with clip length (the story's premise), and
//!  2. on **MLX 0.31.2** the untiled decode returned *wrong pixels* past `T_lat = 6` (31 frames, ~1 s)
//!     at 848×480 — the boundary where `block_out` crosses `i32::MAX`. **sc-12748: FIXED on MLX
//!     0.32.0.** The over-bound write is `block_out`'s conv3d; #3524 promotes its output offset to
//!     `size_t` (probe-verified), and this decoder is attention-free, so the untiled decode is now
//!     correct past the ceiling — see `untiled_matches_chunked_past_the_bound`. Chunking is retained as
//!     a **memory** tool (point 1), no longer a correctness one. (Originally sc-12349.)
//!
//! Measured 2026-07-16 on an M5 Max / 137 GB (MLX limit 121.6 GiB), 848×480, f32:
//!
//! | path | 19 f | 61 f | 151 f | 163 f |
//! |---|---|---|---|---|
//! | untiled | 65.21 | 145.89 | *(wrong pixels — past the bound)* | *(wrong)* |
//! | chunked (chunk=1, cold) | 23.09 | 23.28 | **23.70** | 23.75 |
//! | chunked (chunk=2, warm) | 37.08 | 37.25 | 37.69 | 37.75 |
//!
//! ⚠️ **These allocate tens of GiB and can SIGKILL the host.** The untiled arm costs ~11.5 GiB per
//! latent frame and is deliberately capped at [`MAX_UNTILED_T_LAT`] (~65 GiB); even so, run them
//! **one at a time** (`--test-threads=1` is not enough — the arms accumulate in one process). Driving
//! the untiled path further than this cap SIGKILLed a 137 GB M5 Max at ~88 GiB, above the ~63%-of-RAM
//! wired ceiling. MLX's default error handler is `exit(-1)`, so a memory blow-up here takes the
//! process, not a Rust error.
//!
//! Run (one test at a time):
//! ```text
//! MOCHI_SNAPSHOT=~/.cache/huggingface/hub/models--genmo--mochi-1-preview/snapshots/<sha> \
//!   cargo test -p mlx-gen-mochi --release --test decode_memory_real_weights \
//!   -- --ignored --nocapture --exact decode_peak_is_flat_in_clip_length
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mochi::{
    load_vae_decoder, MochiVaeConfig, MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES,
};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 848×480 — Mochi's native design point and the manifest's shipped resolution.
const W: i32 = 848;
const H: i32 = 480;

/// The largest clip this suite will decode **untiled** at 848×480.
///
/// Two separate ceilings bound this, and (sc-12748) only the memory one now binds:
///  - *correctness*: `block_out` (`128 × 6·T_lat × 480 × 848`) crosses `i32::MAX` between T_lat 6 and
///    7. On MLX 0.31.2 the untiled decode returned wrong pixels past that; on **0.32.0 it is correct**
///    (#3524), so this is no longer a bound — the past-the-bound decode is exercised at bf16 in
///    `untiled_matches_chunked_past_the_bound`;
///  - *memory*: the untiled peak runs ~11.5 GiB per latent frame (measured: 65.21 GiB at T_lat 4,
///    110.77 at 7, 145.89 at 11). T_lat 6 is ~88 GiB, which exceeds this machine class's wired ceiling
///    (~63% of RAM) and gets the process **SIGKILL**ed rather than an error — observed while writing
///    this suite on a 137 GB M5 Max.
///
/// 4 (~65 GiB, measured safe) is as far as the untiled reference can be driven without risking the
/// host. That it cannot be driven further *is itself the story*: this is what chunking removes.
const MAX_UNTILED_T_LAT: i32 = 4;

fn snapshot_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("MOCHI_SNAPSHOT") {
        return p.into();
    }
    discover_snapshot().expect(
        "set MOCHI_SNAPSHOT to a mochi snapshot dir (no genmo/mochi-1-preview or SceneWorks/mochi-1-mlx \
         snapshot with a vae/ dir found under the HF cache)",
    )
}

/// Auto-discover a Mochi snapshot whose `vae/` dir holds the AsymmVAE weights: the genmo
/// `mochi-1-preview` cache first, then the `SceneWorks/mochi-1-mlx` mirror.
fn discover_snapshot() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    for repo in [
        "models--genmo--mochi-1-preview",
        "models--SceneWorks--mochi-1-mlx",
    ] {
        let snaps = hub.join(repo).join("snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            continue;
        };
        for entry in rd.flatten() {
            let dir = entry.path();
            if dir.join("vae").is_dir() {
                return Some(dir);
            }
        }
    }
    None
}

/// Load the AsymmVAE decoder at **bf16** compute precision (the [`load_vae_decoder`] default is f32).
/// bf16 halves the untiled decode's ~11.5 GiB/latent-frame peak, which is what lets the past-the-bound
/// geometry (`T_lat = 7`, ~55 GiB bf16 vs ~110 GiB f32) fit under a 128 GB machine's wired ceiling.
fn load_bf16() -> MochiVaeDecoder {
    let root = snapshot_dir();
    let cfg = MochiVaeConfig::from_model_dir(&root).expect("mochi vae config");
    let w = Weights::from_dir(root.join("vae")).expect("read vae weights");
    MochiVaeDecoder::from_weights_dtype(&w, &cfg, Dtype::Bfloat16).expect("bf16 vae decoder")
}

fn rnd(shape: &[i32], seed: u64) -> Array {
    let n: i64 = shape.iter().map(|&x| x as i64).product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 0.000_001).sin()
                * 0.5
        })
        .collect();
    Array::from_slice(&data, shape)
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `F` output frames ⇒ `T_lat` latent frames (`F = (T_lat − 1)·6 + 1`).
fn latent_frames(out_frames: i32) -> i32 {
    (out_frames - 1) / 6 + 1
}

fn latent_for(out_frames: i32) -> Array {
    rnd(&[1, 12, latent_frames(out_frames), H / 8, W / 8], 7)
}

/// Peak GiB + wall-clock seconds while running `f`, from a clean cache/peak baseline.
fn measure(f: impl FnOnce() -> Array) -> (f64, f64, Array) {
    clear_cache();
    reset_peak_memory();
    let t0 = Instant::now();
    let out = f();
    mlx_rs::transforms::eval([&out]).expect("eval");
    let secs = t0.elapsed().as_secs_f64();
    (get_peak_memory() as f64 / GIB, secs, out)
}

fn load() -> MochiVaeDecoder {
    load_vae_decoder(&snapshot_dir()).expect("load vae decoder")
}

/// **The headline: the chunked decode peak is ~flat in clip length; the untiled one is not.**
#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (real AsymmVAE weights) + a large-memory Metal Mac"]
fn decode_peak_is_flat_in_clip_length() {
    let dec = load();
    println!("\n=== chunked decode peak vs clip length @ {W}×{H} (chunk={DEFAULT_DECODE_CHUNK_FRAMES}) ===");
    println!(
        "{:>7} {:>7} {:>12} {:>9}",
        "frames", "T_lat", "peak GiB", "secs"
    );

    let mut peaks = Vec::new();
    for &f in &[19i32, 61, 151, 163] {
        let (p, s, _v) = measure(|| {
            dec.decode_denormalized_chunked(&latent_for(f), DEFAULT_DECODE_CHUNK_FRAMES, None)
                .expect("chunked")
        });
        println!("{f:>7} {:>7} {p:>12.2} {s:>9.1}", latent_frames(f));
        peaks.push(p);
    }

    let (lo, hi) = (
        peaks.iter().cloned().fold(f64::MAX, f64::min),
        peaks.iter().cloned().fold(0.0, f64::max),
    );
    println!(
        "spread over 19→163 frames: {lo:.2}..{hi:.2} GiB ({:.2}×)",
        hi / lo
    );
    assert!(
        hi / lo < 1.35,
        "chunked decode peak must be ~flat in clip length (8.6× more frames), got {lo:.2}..{hi:.2} GiB"
    );
}

/// The chunk knob's memory/time tradeoff — the evidence behind [`DEFAULT_DECODE_CHUNK_FRAMES`].
#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (real AsymmVAE weights) + a large-memory Metal Mac"]
fn decode_peak_vs_chunk_size() {
    let dec = load();
    println!("\n=== chunk-size tradeoff @ {W}×{H}, 151 frames ===");
    println!("{:>6} {:>12} {:>9}", "chunk", "peak GiB", "secs");
    let mut prev = 0.0f64;
    for &c in &[1usize, 2, 4] {
        let (p, s, _v) = measure(|| {
            dec.decode_denormalized_chunked(&latent_for(151), c, None)
                .expect("chunked")
        });
        println!("{c:>6} {p:>12.2} {s:>9.1}");
        assert!(
            p > prev,
            "a bigger chunk must cost more memory, not less (chunk={c})"
        );
        prev = p;
    }
    println!(
        "max safe chunk @{W}×{H}: {}",
        dec.max_safe_chunk_frames(H / 8, W / 8)
    );
}

/// **Chunked == untiled on real weights**, at the real 848×480 geometry, up to [`MAX_UNTILED_T_LAT`] —
/// as far as the untiled reference can safely be driven on this machine class. Exact equality is the
/// seam check: the chunked decode is not an approximation to blend, so a pass means there is no seam
/// to see.
///
/// The equality itself is geometry-independent (it also holds bit-for-bit on synthetic weights at
/// every chunk size — see `chunked_decode.rs`); what this adds is the *real weights, real resolution*
/// confirmation that nothing about the true kernels or channel widths breaks it.
#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (real AsymmVAE weights) + a large-memory Metal Mac"]
fn chunked_matches_untiled_on_real_weights() {
    let dec = load();
    for t_lat in 2..=MAX_UNTILED_T_LAT {
        let latent = rnd(&[1, 12, t_lat, H / 8, W / 8], 7);
        let single = dec.decode_denormalized(&latent).expect("untiled");
        // A valid decode is ~[-1, 1]. If the untiled arm ever drifts far outside that, it has crossed
        // the element ceiling and is no longer a reference worth comparing against.
        let signal = max_abs(&single);
        assert!(
            (1e-3..1.6).contains(&signal),
            "T_lat={t_lat}: untiled decode range ±{signal:.3} is not a sane video — it is over the \
             element ceiling and cannot serve as the reference (see sc-12349)"
        );
        for chunk in [1usize, 2] {
            let chunked = dec
                .decode_denormalized_chunked(&latent, chunk, None)
                .expect("chunked");
            assert_eq!(
                chunked.shape(),
                single.shape(),
                "T_lat={t_lat} chunk={chunk}: shape"
            );
            let d = max_abs(&subtract(&chunked, &single).unwrap());
            println!("T_lat={t_lat} chunk={chunk}: max abs diff {d:.3e} (signal {signal:.4})");
            assert_eq!(
                d, 0.0,
                "T_lat={t_lat} chunk={chunk}: chunked must equal untiled exactly, got {d:.3e}"
            );
        }
        clear_cache();
    }
}

/// **sc-12748 — THE PAYOFF: the untiled decode is now correct PAST the element ceiling on MLX 0.32.0.**
///
/// On MLX 0.31.2 an untiled decode whose `block_out` crossed `i32::MAX` returned ±2.67 garbage
/// (sc-12291), and sc-12349 added a refusal so the corruption could not reach a caller. That refusal is
/// **retired here**: MLX 0.32.0's #3524 fixes the conv output-offset overflow that caused it
/// (probe-verified in `mlx-gen/tests/mlx_write_bound_probe.rs::conv3d_8to128_output_across_i32max`), and
/// this decoder is attention-free, so conv + elementwise (both int64-safe on this pin) are its only
/// over-bound writes.
///
/// This is the sc-12349 "re-check against a future MLX bump" method made a standing test: decode
/// `T_lat = 7` (`block_out ≈ 2.31e9`, 1.07× the ceiling) **untiled** and compare it to the chunked
/// reference (every chunk stays far below the ceiling). Mochi's chunked decode is *numerically
/// identical* to the untiled one (per-frame + causal, no blend), so a correct untiled decode must match
/// it **exactly** — and be a sane video (±~[-1, 1]), not the ±2.67 corruption. Run at bf16 so the
/// ~11.5 GiB/latent-frame untiled peak (~55 GiB at `T_lat = 7`) fits a 128 GB machine.
#[test]
#[ignore = "sc-12748 real AsymmVAE untiled-past-the-bound decode; auto-discovers snapshot, ~55 GiB bf16"]
fn untiled_matches_chunked_past_the_bound() {
    let dec = load_bf16();
    let t_lat = 7i32; // block_out ≈ 128·(6·7)·480·848 = 2.19e9 > i32::MAX (1.02×), the exact probe class

    // Precondition: this untiled decode genuinely crosses the ceiling the guard used to refuse.
    let elems = 128i64 * (6 * t_lat) as i64 * H as i64 * W as i64;
    assert!(
        elems > i32::MAX as i64,
        "T_lat={t_lat}: block_out {elems} must exceed i32::MAX ({}) to exercise the retired guard",
        i32::MAX
    );

    let latent = rnd(&[1, 12, t_lat, H / 8, W / 8], 7);

    // Untiled — the path sc-12349 refused; must now RENDER (no error), measured for peak + wall-clock.
    let (peak, secs, single) = measure(|| {
        dec.decode_denormalized(&latent)
            .expect("untiled decode past the bound must now succeed (sc-12748), not refuse")
    });
    let out_frames = (t_lat - 1) * 6 + 1;
    let signal = max_abs(&single);
    println!(
        "[sc-12748] untiled @ {W}×{H} T_lat={t_lat} ({out_frames}f, block_out≈{elems} = {:.3}× i32::MAX): \
         ±{signal:.4}  peak={peak:.2} GiB  {secs:.1}s",
        elems as f64 / i32::MAX as f64
    );
    assert!(
        (1e-3..1.6).contains(&signal),
        "untiled past-the-bound decode range ±{signal:.3} is not a sane video — it looks like the \
         0.31.2 ±2.67 corruption, i.e. MLX's conv fix regressed. Re-instate the Mochi decode guard."
    );

    // Chunked reference — every intermediate stays below the ceiling; numerically identical to untiled.
    let chunked = dec
        .decode_denormalized_chunked(&latent, DEFAULT_DECODE_CHUNK_FRAMES, None)
        .expect("chunked reference decode");
    assert_eq!(chunked.shape(), single.shape(), "shape");
    let d = max_abs(&subtract(&chunked, &single).unwrap());
    println!("[sc-12748] untiled vs chunked max|Δ| = {d:.3e} (a valid video is ~[-1, 1])");
    assert!(
        d < 1e-2,
        "T_lat={t_lat}: the untiled past-the-bound decode diverged from the below-bound chunked \
         reference by {d:.3e} — the >i32::MAX path is NOT correct. (bf16 tolerance; the f32 path is \
         bit-exact — see chunked_matches_untiled_on_real_weights.)"
    );
}
