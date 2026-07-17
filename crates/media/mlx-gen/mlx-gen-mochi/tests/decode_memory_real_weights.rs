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
//!  2. the untiled decode returns *wrong pixels* past `T_lat = 6` (31 frames, ~1 s) at 848×480 — the
//!     boundary where `block_out` crosses `i32::MAX` elements. The mechanism is **not** established
//!     (a conv3d at that size is fine; an elementwise op at exactly that size is not), so treat it as
//!     a measured property of this decoder, not an MLX law. See sc-12349.
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

use std::time::Instant;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen_mochi::{load_vae_decoder, MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 848×480 — Mochi's native design point and the manifest's shipped resolution.
const W: i32 = 848;
const H: i32 = 480;

/// The largest clip this suite will decode **untiled** at 848×480.
///
/// Two separate ceilings bound this, and the memory one binds first:
///  - *correctness*: `block_out` (`128 × 6·T_lat × 480 × 848`) crosses `i32::MAX` between T_lat 6 and
///    7, and `decode_denormalized` now refuses past that rather than return wrong pixels;
///  - *memory*: the untiled peak runs ~11.5 GiB per latent frame (measured: 65.21 GiB at T_lat 4,
///    110.77 at 7, 145.89 at 11). T_lat 6 is ~88 GiB, which exceeds this machine class's wired ceiling
///    (~63% of RAM) and gets the process **SIGKILL**ed rather than an error — observed while writing
///    this suite on a 137 GB M5 Max.
///
/// 4 (~65 GiB, measured safe) is as far as the untiled reference can be driven without risking the
/// host. That it cannot be driven further *is itself the story*: this is what chunking removes.
const MAX_UNTILED_T_LAT: i32 = 4;

fn snapshot_dir() -> std::path::PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
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

/// **The chunked decode is correct where the untiled one cannot even be attempted** (sc-12349). A 5 s
/// clip (T_lat 26) is 8.13e9 elements untiled — 3.8× past the bound — so `decode_denormalized` refuses
/// it, while the chunked path decodes it to a sane video.
///
/// This is the *reachable* half of the finding. The raw corruption (untiled at T_lat 7 → ±2.67 instead
/// of ±0.50) is no longer observable through the public API precisely because the guard now refuses it,
/// and reproducing it costs ~111 GiB — see sc-12349, which carries the measurement and the method for
/// re-checking it against a future MLX bump (temporarily drop the guard).
#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (real AsymmVAE weights) + a large-memory Metal Mac"]
fn untiled_refuses_the_shipped_default_that_chunked_decodes() {
    let dec = load();
    let latent = latent_for(151); // the shipped 5 s default
    let elems = 128i64 * (6 * latent_frames(151)) as i64 * H as i64 * W as i64;
    assert!(
        elems > i32::MAX as i64,
        "the 151-frame default should be over the element ceiling ({elems} elements)"
    );

    match dec.decode_denormalized(&latent) {
        Err(mlx_gen::Error::Msg(m)) => println!("untiled @151f correctly refused: {m}"),
        Err(e) => panic!("expected an element-ceiling Msg error, got {e}"),
        Ok(_) => panic!(
            "untiled decode of the 151-frame default ({elems} elements) must refuse — MLX returns \
             wrong pixels here without a guard. If MLX fixed this, update MAX_TENSOR_ELEMS + sc-12349."
        ),
    }

    let chunked = dec
        .decode_denormalized_chunked(&latent, DEFAULT_DECODE_CHUNK_FRAMES, None)
        .expect("chunked must decode the shipped default");
    let c = max_abs(&chunked);
    println!("chunked @151f: ±{c:.4} (a valid video is ~[-1, 1])");
    assert!(
        (1e-3..1.6).contains(&c),
        "the chunked decode of the shipped default must be a sane video, got ±{c:.4}"
    );
}
