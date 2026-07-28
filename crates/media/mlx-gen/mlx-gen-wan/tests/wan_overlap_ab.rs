//! sc-15445 — real-weight A/B for Wan's raised temporal decode overlap.
//!
//! sc-15325 raised the shared Wan overlap from the candidate grid's historical value to half of
//! the latent tile. That is peak-memory-neutral, but it shortens the stride and can add decode
//! iterations. This harness measures the trade on the *same deterministic latent* with the real z16
//! and z48 VAEs. It deliberately runs one family/bucket/clip point per process so an OOM cannot erase
//! earlier results.
//!
//! The two recorded operating points are the shipping 640×384 and 832×480 buckets, at product clip
//! lengths 81 and 121. z16 is non-causal and emits `4*T_lat`, so those become 84 and 124 decoded
//! frames before the product's normal trim; z48 is causal and emits exactly 81 and 121.
//!
//! ```text
//! WAN_OVERLAP_AB_FAMILY=z16 WAN_OVERLAP_AB_VAE=/path/to/z16/vae.safetensors \
//! WAN_OVERLAP_AB_W=640 WAN_OVERLAP_AB_H=384 WAN_OVERLAP_AB_FRAMES=81 \
//! WAN_OVERLAP_AB_TILE_FRAMES=32 \
//!   cargo test -p mlx-gen-wan --release --test wan_overlap_ab -- --ignored --nocapture
//!
//! WAN_OVERLAP_AB_FAMILY=z48 WAN_OVERLAP_AB_VAE=/path/to/z48/vae.safetensors \
//! WAN_OVERLAP_AB_W=832 WAN_OVERLAP_AB_H=480 WAN_OVERLAP_AB_FRAMES=121 \
//! WAN_OVERLAP_AB_TILE_FRAMES=48 \
//!   cargo test -p mlx-gen-wan --release --test wan_overlap_ab -- --ignored --nocapture
//! ```
//!
//! Each row warms both A/B paths once, then reports the median and range of three alternating
//! materialized decodes. Quality is mean absolute error on the viewer's 0–255 scale plus mean/worst
//! near-white clipping. The reference uses the same 192/64 shipping spatial candidate but a 96-frame
//! temporal tile with half overlap: it is temporally single-pass at the 81-frame point and the
//! highest-context writable/affordable tiled reference at 121 frames.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen_wan::{Wan22Vae, WanVae};
use mlx_rs::{random, Array};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Z16,
    Z48,
}

impl Family {
    fn from_env() -> Self {
        match std::env::var("WAN_OVERLAP_AB_FAMILY")
            .unwrap_or_else(|_| "z16".into())
            .as_str()
        {
            "z16" => Self::Z16,
            "z48" => Self::Z48,
            other => panic!("WAN_OVERLAP_AB_FAMILY must be z16 or z48, got {other:?}"),
        }
    }

    fn tiling(self) -> VaeTiling {
        match self {
            Self::Z16 => VaeTiling::WAN,
            Self::Z48 => VaeTiling::WAN22,
        }
    }

    fn latent_and_output_frames(self, product_frames: i32) -> (i32, i32) {
        match self {
            Self::Z16 => {
                let latent = (product_frames - 1) / 4 + 1;
                (latent, latent * 4)
            }
            Self::Z48 => {
                assert_eq!(
                    (product_frames - 1) % 4,
                    0,
                    "z48 product frames must be 1+4*k"
                );
                ((product_frames - 1) / 4 + 1, product_frames)
            }
        }
    }
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn config(tile_frames: i32, overlap_frames: i32) -> TilingConfig {
    TilingConfig {
        // 192/64 is the smallest shipping spatial candidate. It bounds the real-weight run below
        // 128 GiB while leaving the A/B's temporal geometry as the only changed variable.
        spatial: Some(SpatialTiling {
            tile_px: 192,
            overlap_px: 64,
        }),
        temporal: Some(TemporalTiling {
            tile_frames,
            overlap_frames,
        }),
    }
}

fn historical_overlap(tile_frames: i32) -> i32 {
    match tile_frames {
        96 | 64 => 24,
        48 => 16,
        32 => 8,
        other => panic!(
            "WAN_OVERLAP_AB_TILE_FRAMES must be a shipping candidate (96,64,48,32), got {other}"
        ),
    }
}

fn half_overlap(tile_frames: i32) -> i32 {
    tile_frames / 2
}

fn mean_abs_delta_255(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    let m = mlx_rs::ops::mean(&d, None).unwrap();
    mlx_rs::transforms::eval([&m]).unwrap();
    m.item::<f32>() as f64 * 127.5
}

/// `(mean %, worst-frame %)` of pixels having any channel at or above 250/255.
fn clip_stats(v: &Array, family: Family) -> (f64, f64) {
    mlx_rs::transforms::eval([v]).unwrap();
    let sh = v.shape();
    let flat = v.as_slice::<f32>();
    let (frames, height, width) = match family {
        Family::Z16 => (sh[2] as usize, sh[3] as usize, sh[4] as usize),
        Family::Z48 => (sh[1] as usize, sh[2] as usize, sh[3] as usize),
    };
    let pixels = height * width;
    let mut frame_pcts = Vec::with_capacity(frames);
    for t in 0..frames {
        let mut clipped = 0usize;
        for i in 0..pixels {
            let max_channel = match family {
                // NCTHW, with channel planes separated by `frames * pixels`.
                Family::Z16 => {
                    let base = t * pixels + i;
                    let channel_stride = frames * pixels;
                    flat[base]
                        .max(flat[base + channel_stride])
                        .max(flat[base + 2 * channel_stride])
                }
                // NTHWC, with three adjacent channels.
                Family::Z48 => {
                    let base = (t * pixels + i) * 3;
                    flat[base].max(flat[base + 1]).max(flat[base + 2])
                }
            };
            if max_channel >= 0.9608 {
                clipped += 1;
            }
        }
        frame_pcts.push(100.0 * clipped as f64 / pixels as f64);
    }
    let mean = frame_pcts.iter().sum::<f64>() / frame_pcts.len() as f64;
    let worst = frame_pcts.iter().copied().fold(0.0, f64::max);
    (mean, worst)
}

fn median_range(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    (
        values[values.len() / 2],
        values[0],
        values[values.len() - 1],
    )
}

fn iteration_count(cfg: &TilingConfig, family: Family, latent: &Array) -> (usize, usize) {
    let sh = latent.shape();
    let (f, h, w) = match family {
        Family::Z16 => (sh[2], sh[3], sh[4]),
        Family::Z48 => (sh[1], sh[2], sh[3]),
    };
    let plan = cfg.plan(family.tiling(), f, h, w);
    let temporal = plan.t.len();
    (temporal, temporal * plan.h.len() * plan.w.len())
}

#[test]
#[ignore = "needs a real Wan z16 or z48 vae.safetensors and Metal; GPU-heavy"]
fn wan_overlap_wall_time_and_quality_ab() {
    let family = Family::from_env();
    let vae_path = env_path("WAN_OVERLAP_AB_VAE")
        .expect("set WAN_OVERLAP_AB_VAE to the selected family's real vae.safetensors");
    let width = env_i32("WAN_OVERLAP_AB_W", 640);
    let height = env_i32("WAN_OVERLAP_AB_H", 384);
    let product_frames = env_i32("WAN_OVERLAP_AB_FRAMES", 81);
    let tile_frames = env_i32("WAN_OVERLAP_AB_TILE_FRAMES", 32);
    let repeats = env_usize("WAN_OVERLAP_AB_REPEATS", 3);
    let quality_only = env_usize("WAN_OVERLAP_AB_QUALITY_ONLY", 0) == 1;
    assert!(
        quality_only || repeats >= 3,
        "use at least three measured repeats"
    );
    assert_eq!(width % family.tiling().spatial_scale, 0);
    assert_eq!(height % family.tiling().spatial_scale, 0);

    let (latent_frames, output_frames) = family.latent_and_output_frames(product_frames);
    let old = config(tile_frames, historical_overlap(tile_frames));
    let current = config(tile_frames, half_overlap(tile_frames));
    let reference = config(96, 48);

    let key = random::key(15_445).unwrap();
    let latent = match family {
        Family::Z16 => random::normal::<f32>(
            &[
                1,
                16,
                latent_frames,
                height / VaeTiling::WAN.spatial_scale,
                width / VaeTiling::WAN.spatial_scale,
            ],
            None,
            None,
            Some(&key),
        )
        .unwrap(),
        Family::Z48 => random::normal::<f32>(
            &[
                48,
                latent_frames,
                height / VaeTiling::WAN22.spatial_scale,
                width / VaeTiling::WAN22.spatial_scale,
            ],
            None,
            None,
            Some(&key),
        )
        .unwrap(),
    };
    mlx_rs::transforms::eval([&latent]).unwrap();

    let old_iters = iteration_count(&old, family, &latent);
    let current_iters = iteration_count(&current, family, &latent);
    let ref_iters = iteration_count(&reference, family, &latent);
    println!(
        "\n=== sc-15445 Wan overlap A/B ===\n\
         family={family:?} weights={} stimulus=normal(seed=15445) \n\
         product={width}x{height}x{product_frames} decoded={width}x{height}x{output_frames} \
         latent={:?}\n\
         spatial=192/64 old={tile_frames}/{} current={tile_frames}/{} reference=96/48\n\
         iterations temporal/all: old={}/{} current={}/{} reference={}/{} repeats={repeats}",
        vae_path.display(),
        latent.shape(),
        historical_overlap(tile_frames),
        half_overlap(tile_frames),
        old_iters.0,
        old_iters.1,
        current_iters.0,
        current_iters.1,
        ref_iters.0,
        ref_iters.1,
    );
    println!(
        "memory before VAE load: limit={:.3} GiB active={:.3} GiB cache={:.3} GiB",
        gib(mlx_rs::memory::get_memory_limit()),
        gib(mlx_rs::memory::get_active_memory()),
        gib(mlx_rs::memory::get_cache_memory()),
    );

    let mut weights = Weights::from_file(&vae_path).expect("open VAE weights");
    // The shipping TI2V-5B decode casts z48's conv-heavy body to bf16 (model.rs, sc-5039).
    // Benchmark that path rather than the slower f32 diagnostic/reference mode.
    if family == Family::Z48 {
        weights
            .cast_all(mlx_rs::Dtype::Bfloat16)
            .expect("cast z48 VAE weights to the shipping bf16 compute dtype");
    }
    let z16 =
        (family == Family::Z16).then(|| WanVae::from_weights(&weights).expect("load z16 VAE"));
    let z48 =
        (family == Family::Z48).then(|| Wan22Vae::from_weights(&weights).expect("load z48 VAE"));
    println!(
        "memory after VAE load: limit={:.3} GiB active={:.3} GiB cache={:.3} GiB",
        gib(mlx_rs::memory::get_memory_limit()),
        gib(mlx_rs::memory::get_active_memory()),
        gib(mlx_rs::memory::get_cache_memory()),
    );
    let decode = |cfg: &TilingConfig| -> (Array, f64, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let started = Instant::now();
        let out = match family {
            Family::Z16 => z16
                .as_ref()
                .unwrap()
                .decode_tiled(&latent, cfg, None)
                .expect("z16 tiled decode"),
            Family::Z48 => z48
                .as_ref()
                .unwrap()
                .decode_tiled(&latent, cfg, None)
                .expect("z48 tiled decode"),
        };
        mlx_rs::transforms::eval([&out]).expect("materialize decode");
        (
            out,
            started.elapsed().as_secs_f64(),
            mlx_rs::memory::get_peak_memory(),
        )
    };

    let (
        old_median,
        old_min,
        old_max,
        current_median,
        current_min,
        current_max,
        old_peak,
        current_peak,
    ) = if quality_only {
        println!("progress quality-only mode: skipping warm-up and repeated timing");
        (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0,
            0,
        )
    } else {
        // Warm both paths once before collecting wall time. Alternate old/current each round so a
        // slow thermal drift cannot systematically favor one side.
        let (_, warm_old, _) = decode(&old);
        println!("progress warm old={warm_old:.3}s");
        let (_, warm_current, _) = decode(&current);
        println!("progress warm current={warm_current:.3}s");
        let mut old_times = Vec::with_capacity(repeats);
        let mut current_times = Vec::with_capacity(repeats);
        let mut old_peak = 0usize;
        let mut current_peak = 0usize;
        for round in 0..repeats {
            let (first, second) = if round % 2 == 0 {
                (&old, &current)
            } else {
                (&current, &old)
            };
            let (_, first_secs, first_peak) = decode(first);
            let (_, second_secs, second_peak) = decode(second);
            println!(
                    "progress measured round={} order={} first={first_secs:.3}s second={second_secs:.3}s",
                    round + 1,
                    if std::ptr::eq(first, &old) {
                        "old,current"
                    } else {
                        "current,old"
                    }
                );
            if std::ptr::eq(first, &old) {
                old_times.push(first_secs);
                current_times.push(second_secs);
                old_peak = old_peak.max(first_peak);
                current_peak = current_peak.max(second_peak);
            } else {
                current_times.push(first_secs);
                old_times.push(second_secs);
                current_peak = current_peak.max(first_peak);
                old_peak = old_peak.max(second_peak);
            }
        }
        let (old_median, old_min, old_max) = median_range(old_times);
        let (current_median, current_min, current_max) = median_range(current_times);
        (
            old_median,
            old_min,
            old_max,
            current_median,
            current_min,
            current_max,
            old_peak,
            current_peak,
        )
    };

    // One deterministic materialized output per path is enough for quality; decoding is
    // deterministic and the timing loop already established repeatability.
    let (reference_out, reference_secs, reference_peak) = decode(&reference);
    println!("progress quality reference={reference_secs:.3}s");
    let (old_out, _, _) = decode(&old);
    println!("progress quality old materialized");
    let (current_out, _, _) = decode(&current);
    println!("progress quality current materialized");
    let old_error = mean_abs_delta_255(&reference_out, &old_out);
    let current_error = mean_abs_delta_255(&reference_out, &current_out);
    let reference_clip = clip_stats(&reference_out, family);
    let old_clip = clip_stats(&old_out, family);
    let current_clip = clip_stats(&current_out, family);

    println!(
        "RESULT family={family:?} bucket={width}x{height} product_frames={product_frames} \
         decoded_frames={output_frames} tile={tile_frames} \
         old_overlap={} current_overlap={} temporal_iterations={}=>{} all_iterations={}=>{} \
         old_seconds_median={old_median:.3} old_seconds_range={old_min:.3}..{old_max:.3} \
         current_seconds_median={current_median:.3} \
         current_seconds_range={current_min:.3}..{current_max:.3} \
         wall_ratio={:.4} old_peak_gib={:.3} current_peak_gib={:.3} \
         old_mae255={old_error:.4} current_mae255={current_error:.4} \
         reference_clip_mean={:.4} reference_clip_worst={:.4} \
         old_clip_mean={:.4} old_clip_worst={:.4} \
         current_clip_mean={:.4} current_clip_worst={:.4} \
         reference_seconds={reference_secs:.3} reference_peak_gib={:.3}",
        historical_overlap(tile_frames),
        half_overlap(tile_frames),
        old_iters.0,
        current_iters.0,
        old_iters.1,
        current_iters.1,
        current_median / old_median,
        gib(old_peak),
        gib(current_peak),
        reference_clip.0,
        reference_clip.1,
        old_clip.0,
        old_clip.1,
        current_clip.0,
        current_clip.1,
        gib(reference_peak),
    );
}
