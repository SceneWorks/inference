//! sc-6894 (F-004) — **LTX-2.3 video VAE decode anchor sweep.** Measures the real MLX peak GPU
//! allocation of [`LtxVideoVae::decode`] / [`LtxVideoVae::decode_tiled`] across output sizes and tile
//! sizes, so the budgeted decode cost model (`estimated_ltx_decode_peak_gib`) can be **fit from real
//! measurements** — the way sc-4998 fit the Wan z48 model and sc-6894 fit the z16 model. Decode-only:
//! loads just `vae_decoder.safetensors` (no encoder). One config **per process** (env-driven) so an
//! OOM on the largest configs kills only this process; the driving shell loop keeps earlier anchors.
//! Synthetic (random) latents — we measure cost/scaling, not parity.
//!
//! ```text
//! LTX_VAE_DIR=~/.cache/huggingface/hub/models--SceneWorks--ltx-2.3-mlx/snapshots/<h>/q8 \
//! LTX_W=768 LTX_H=768 LTX_FRAMES=25 \
//!   cargo test -p mlx-gen-ltx --test vae_decode_sweep --release -- --ignored --nocapture
//! # add LTX_TILE_PX=256 [LTX_OVERLAP_PX=32 LTX_TILE_FRAMES=.. LTX_OVERLAP_FRAMES=..] for a tiled run
//! # add LTX_LIMIT_GB=48 to simulate a smaller-RAM tier
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::memory::{get_memory_limit, get_peak_memory, reset_peak_memory, set_memory_limit};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{random, Array};

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::{LtxVaeConfig, LtxVideoVae};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));
            }
        }
        PathBuf::from(s.to_string())
    })
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "needs the ltx-2.3-mlx vae_decoder.safetensors (LTX_VAE_DIR); GPU-heavy"]
fn ltx_vae_decode_sweep() {
    let dir = match env_path("LTX_VAE_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set LTX_VAE_DIR to a snapshot dir holding vae_decoder.safetensors");
            return;
        }
    };
    let w_out = env_usize("LTX_W", 768) as i32;
    let h_out = env_usize("LTX_H", 768) as i32;
    let frames = env_usize("LTX_FRAMES", 25) as i32;
    // LTX VAE: spatial /32, temporal /8 **causal** (out_f = 1 + (T_lat−1)·8 ⇒ frames = 1 + 8·k).
    assert_eq!(
        (frames - 1) % 8,
        0,
        "LTX_FRAMES must be 1 + 8·k (got {frames})"
    );
    assert_eq!(h_out % 32, 0, "LTX_H must be a multiple of 32");
    assert_eq!(w_out % 32, 0, "LTX_W must be a multiple of 32");
    let (z, t_lat, h_lat, w_lat) = (128, (frames - 1) / 8 + 1, h_out / 32, w_out / 32);

    if let Ok(lim) = std::env::var("LTX_LIMIT_GB") {
        if let Ok(g) = lim.parse::<usize>() {
            let prev = set_memory_limit(g << 30);
            println!(
                "[limit] pinned MLX memory limit {g} GB (was {:.0} GB)",
                gb(prev)
            );
        }
    }

    let vae_cfg = LtxVaeConfig::from_model_dir(&dir).expect("read LtxVaeConfig");
    let decoder_w =
        Weights::from_file(dir.join("vae_decoder.safetensors")).expect("read vae_decoder");
    let vae =
        LtxVideoVae::from_weights(&decoder_w, None, &vae_cfg).expect("LtxVideoVae::from_weights");

    // Synthetic latent [B=1, 128, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let key = random::key(0).unwrap();
    let latent =
        random::normal::<f32>(&[1, z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();

    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 8, h_lat * 32, w_lat * 32);

    // Fixed-tile iff LTX_TILE_PX is set; otherwise single-pass. (Budgeted mode is added once the cost
    // model exists — see `auto_tiling_budgeted_ltx`.)
    let tile_px = env_usize("LTX_TILE_PX", 0) as i32;
    let cfg = if env_usize("LTX_BUDGETED", 0) == 1 {
        mlx_gen_ltx::pipeline::auto_tiling_budgeted_ltx(out_w, out_h, out_f)
            .expect("ltx decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        let overlap_px = env_usize("LTX_OVERLAP_PX", 32) as i32;
        let spatial = Some(SpatialTiling {
            tile_px,
            overlap_px,
        });
        let tf = env_usize("LTX_TILE_FRAMES", 0) as i32;
        let temporal = (tf > 0).then(|| TemporalTiling {
            tile_frames: tf,
            overlap_frames: env_usize("LTX_OVERLAP_FRAMES", (tf / 2).max(1) as usize) as i32,
        });
        Some(TilingConfig { spatial, temporal })
    } else {
        None
    };

    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
    let (tile_f, tile_h, tile_w) = match &cfg {
        Some(c) => (
            c.temporal
                .map(|t| (t.tile_frames as i64).min(out_f as i64))
                .unwrap_or(out_f as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_h as i64))
                .unwrap_or(out_h as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_w as i64))
                .unwrap_or(out_w as i64),
        ),
        None => (out_f as i64, out_h as i64, out_w as i64),
    };
    let tile_vox = tile_f * tile_h * tile_w;

    println!(
        "\n=== ltx sweep: out {out_w}x{out_h}x{out_f}  latent[z{z},T{t_lat},{h_lat},{w_lat}]  \
         tiled={}  MLX limit={:.0} GB ===",
        cfg.is_some(),
        gb(get_memory_limit())
    );

    reset_peak_memory();
    let t = Instant::now();
    let video = match &cfg {
        Some(c) => vae.decode_tiled(&latent, c, &Default::default()).unwrap(),
        None => vae.decode(&latent).unwrap(),
    };
    mlx_rs::transforms::eval([&video]).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak_bytes = get_peak_memory();
    let peak = gb(peak_bytes);

    println!(
        "[LTX decode] -> {:?}  {secs:.1}s  peak={peak:.2} GB",
        video.shape()
    );
    println!(
        "ANCHOR out_vox={out_vox} tile_vox={tile_vox} peak_gb={peak:.4} \
         peak_bytes_per_out_vox={:.1} peak_bytes_per_tile_vox={:.1}",
        peak_bytes as f64 / out_vox as f64,
        peak_bytes as f64 / tile_vox as f64,
    );
}

/// Discover the LTX VAE decoder dir (`vae_decoder.safetensors`): `LTX_VAE_DIR` first, else the `q4`
/// tier of the cached `SceneWorks/ltx-2.3-mlx` snapshot.
fn discover_ltx_vae() -> Option<PathBuf> {
    if let Some(p) = env_path("LTX_VAE_DIR") {
        return Some(p);
    }
    let home = std::env::var_os("HOME")?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--ltx-2.3-mlx/snapshots");
    for entry in std::fs::read_dir(&snaps).ok()?.flatten() {
        for tier in ["bf16", "q8", "q4"] {
            let d = entry.path().join(tier);
            if d.join("vae_decoder.safetensors").exists() {
                return Some(d);
            }
        }
    }
    None
}

/// **sc-12748 — THE PAYOFF (mode 2): a tiled LTX decode whose ASSEMBLED output crosses `i32::MAX`
/// renders on real weights and validates against a below-bound reference.**
///
/// sc-12438 (PR #74) *refused* a tiled decode whose assembled RGB output exceeded the write bound — on
/// MLX 0.31.2 the `pad`-and-accumulate that builds it corrupted past the bound, and `reshape`/`from_slice`
/// overflowed, so it could not be produced at all. sc-12748 lifts that refusal: on this pin the assembly
/// ops (`pad`/`add`/`divide`) and read-back (`reshape`+`as_slice`) are all probe-verified int64-safe
/// (`mlx-gen/tests/mlx_write_bound_probe.rs`; sc-12746 pad copy-gate + #3524).
///
/// Geometry: **1280²×441f** (t_lat 56) → assembled RGB output `3·441·1280·1280 = 2.168e9 = 1.009×
/// i32::MAX` — the exact "1280²·441f class" the `check_output_writable` unit test names. Each decode
/// TILE stays below the bound (only the assembled output crosses), so this isolates the assembled-output
/// path. Validation honours the sc-12438 probe rule, in three tiers:
///  1. the causal LTX prefix `[0..32]` is *the same content* whether the clip is 441 or 65 frames long,
///     so it must match a below-bound reference decode of the frame-prefix latent (same tiling, so the
///     shared first tile is bit-identical) — sub-bound-offset exactness;
///  2. a tail voxel at a **>i32::MAX flat offset** is read back (via `as_slice`, the proven path) and
///     checked finite/sane — the over-bound region is addressed, not garbage;
///  3. **sc-12926 — position-dependent OVER-bound exactness**: the tiled loop's per-tile closure is a
///     plain slice-decode, so a standalone decode of the LAST temporal tile's latent slice must agree
///     (≤1e-3) with the assembled output wherever that tile is the sole contributor at blend weight 1
///     (there the accumulate is `dec·1.0` + zero-pads and the normalize divides by exactly 1). Channel-2
///     late frames sit past `i32::MAX`, so a ±2.67-class corruption of the over-bound region — which
///     tier 2's finite+|v|<5 check would PASS — fails this comparison. Over-bound exactness is thereby
///     render-covered, not just probe-covered.
///
/// Peak + wall-clock are reported.
#[test]
#[ignore = "sc-12748 real LTX VAE over-bound assembled-output render (peak ~46 GiB measured); auto-discovers the q4 vae"]
fn over_bound_output_matches_below_bound_reference() {
    let Some(dir) = discover_ltx_vae() else {
        eprintln!("skip: no LTX_VAE_DIR and no ltx-2.3-mlx vae_decoder.safetensors under the HF cache");
        return;
    };
    if let Some(g) = std::env::var("LTX_LIMIT_GB").ok().and_then(|s| s.parse::<usize>().ok()) {
        set_memory_limit(g << 30);
    }
    let vae_cfg = LtxVaeConfig::from_model_dir(&dir).expect("read LtxVaeConfig");
    let decoder_w =
        Weights::from_file(dir.join("vae_decoder.safetensors")).expect("read vae_decoder");
    let vae =
        LtxVideoVae::from_weights(&decoder_w, None, &vae_cfg).expect("LtxVideoVae::from_weights");

    // 1280²×441f: latent [1,128,56,40,40] → out_f = 1+(56-1)·8 = 441, out 1280×1280.
    let (z, t_lat, h_lat, w_lat) = (128i32, 56i32, 40i32, 40i32);
    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 8, h_lat * 32, w_lat * 32);
    let assembled = 3i64 * out_f as i64 * out_h as i64 * out_w as i64;
    const I32_MAX: i64 = i32::MAX as i64;
    assert!(
        assembled > I32_MAX,
        "precondition: assembled output {assembled} must cross i32::MAX ({I32_MAX})"
    );

    let key = random::key(0).unwrap();
    let latent =
        random::normal::<f32>(&[1, z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();

    // Tile temporally in 64-output-frame tiles (each tile's write stays far below the bound); leave
    // the 1280² spatial extent as one tile. Only the ASSEMBLED output crosses the bound.
    let cfg = TilingConfig {
        spatial: None,
        temporal: Some(TemporalTiling {
            tile_frames: 64,
            overlap_frames: 16,
        }),
    };

    reset_peak_memory();
    let t = Instant::now();
    // Must RENDER — the sc-12438 refusal is retired. (A refusal would surface here as an Err.)
    let video = vae
        .decode_tiled(&latent, &cfg, &Default::default())
        .expect("over-bound assembled-output decode must render, not refuse (sc-12748)");
    mlx_rs::transforms::eval([&video]).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak = gb(get_peak_memory());
    assert_eq!(video.shape(), &[1, 3, out_f, out_h, out_w], "output geometry");
    println!(
        "\n[sc-12748 LTX mode-2] out {out_w}x{out_h}x{out_f}  assembled={assembled} ({:.3}× i32::MAX)  \
         peak={peak:.2} GiB  {secs:.1}s",
        assembled as f64 / I32_MAX as f64
    );

    // Below-bound reference: a 65-frame prefix (t_lat 9 → assembled 3·65·1280² = 3.2e8, far under the
    // bound), same tiling ⇒ its first temporal tile is bit-identical to the full decode's first tile.
    let ref_t_lat = 9i32;
    let ref_latent = latent
        .take_axis(Array::from_slice(&(0..ref_t_lat).collect::<Vec<i32>>(), &[ref_t_lat]), 2)
        .unwrap();
    let reference = vae
        .decode_tiled(&ref_latent, &cfg, &Default::default())
        .expect("below-bound reference decode");
    mlx_rs::transforms::eval([&reference]).unwrap();

    // Compare the causal prefix [0..32] (inside the shared first tile's blend-weight-1 region, at
    // sub-bound offsets): identical tiles + identical latent ⇒ bit-exact.
    let cmp_frames = 32i32;
    let idx = Array::from_slice(&(0..cmp_frames).collect::<Vec<i32>>(), &[cmp_frames]);
    let o_pre = video.take_axis(&idx, 2).unwrap();
    let r_pre = reference.take_axis(&idx, 2).unwrap();
    let d = max(abs(subtract(&o_pre, &r_pre).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    println!("[sc-12748 LTX mode-2] prefix[0..{cmp_frames}] over-bound-vs-below-bound max|Δ| = {d:.3e}");
    assert!(
        d < 1e-3,
        "the over-bound tiled decode's prefix diverged from the below-bound reference by {d:.3e} — the \
         assembled-output path is not correct (a shared tile must decode bit-identically)"
    );

    // Read a TAIL voxel at a >i32::MAX flat offset via `as_slice` (the proven read path) and check it is
    // finite and in a sane video range — proving the over-bound region is addressed, not aliased/garbage.
    let flat = video.as_slice::<f32>();
    assert_eq!(flat.len() as i64, assembled, "as_slice must expose the whole over-bound buffer");
    let tail_off = (out_f as i64 - 1) * out_h as i64 * out_w as i64 * 3 + 12345; // frame 440, > i32::MAX
    assert!(tail_off > I32_MAX, "tail sample offset {tail_off} must be past i32::MAX");
    let mut worst = 0f32;
    for k in 0..8i64 {
        let v = flat[(tail_off + k) as usize];
        assert!(v.is_finite(), "over-bound tail voxel {} is not finite: {v}", tail_off + k);
        worst = worst.max(v.abs());
    }
    println!("[sc-12748 LTX mode-2] tail voxels @ flat offset {tail_off} (>i32::MAX): max|v| = {worst:.4}");
    assert!(
        worst < 5.0,
        "over-bound tail voxels are out of a sane video range (|v|={worst:.3}) — the >i32::MAX region \
         read back as garbage"
    );

    // Tier 3 (sc-12926): position-dependent over-bound exactness. Recompute the temporal plan the
    // tiled decode used, standalone-decode the LAST tile's latent slice, and compare it against the
    // assembled output in that tile's sole-contributor weight-1 frames — at flat offsets PAST i32::MAX.
    let plan = cfg.plan(VaeTiling::LTX, t_lat, h_lat, w_lat);
    assert!(plan.t.len() >= 2, "geometry must produce multiple temporal tiles");
    let last = plan.t.last().unwrap();
    let prev_stop = plan.t[plan.t.len() - 2].out_stop; // frames ≥ this are covered only by `last`
    let tile_idx: Vec<i32> = (last.start..last.end).collect();
    let tile_latent = latent
        .take_axis(Array::from_slice(&tile_idx, &[tile_idx.len() as i32]), 2)
        .unwrap();
    let tile_ref = vae.decode(&tile_latent).expect("standalone decode of the last temporal tile");
    mlx_rs::transforms::eval([&tile_ref]).unwrap();
    let rf = tile_ref.shape()[2] as i64;
    assert!(
        rf >= (last.out_stop - last.out_start) as i64,
        "last-tile standalone decode must cover the tile's output span"
    );
    let ref_flat = tile_ref.as_slice::<f32>();

    // Frames where the last tile is the sole contributor at blend weight 1 (assembled == standalone).
    let sole: Vec<i64> = (prev_stop.max(last.out_start)..last.out_stop)
        .filter(|&f| last.mask[(f - last.out_start) as usize] >= 1.0 - 1e-6)
        .map(i64::from)
        .collect();
    assert!(!sole.is_empty(), "no sole-contributor weight-1 frames in the last tile");

    // Check EVERY sole-contributor frame (the decodes are already paid for; this is host reads) at
    // four positions × 3 channels. Channel-2 frames from ~429 sit past i32::MAX, so the over-bound
    // band gets dense coverage, not a token sample.
    let (oh, ow) = (out_h as i64, out_w as i64);
    let mut max_d = 0f32;
    let mut above = 0i64;
    let mut checked = 0i64;
    for &f in &sole {
        for c in 0..3i64 {
            for &(y, x) in &[(0i64, 0i64), (oh / 2, ow / 2), (oh - 1, ow - 1), (123i64, 456i64)] {
                let off = ((c * out_f as i64 + f) * oh + y) * ow + x;
                let r_off = ((c * rf + (f - last.out_start as i64)) * oh + y) * ow + x;
                let dd = (flat[off as usize] - ref_flat[r_off as usize]).abs();
                checked += 1;
                if off > I32_MAX {
                    above += 1;
                }
                if dd > max_d {
                    max_d = dd;
                }
            }
        }
    }
    println!(
        "[sc-12926 LTX tier-3] last-tile frames {}..={}: checked={checked} \
         above_2^31_offsets={above} max|Δ|={max_d:.3e} vs standalone tile decode",
        sole[0],
        sole[sole.len() - 1]
    );
    assert!(
        above > 0,
        "tier-3 sampled no flat offsets past i32::MAX — the over-bound region went unvalidated"
    );
    assert!(
        max_d < 1e-3,
        "assembled over-bound output diverged from the standalone last-tile decode by {max_d:.3e} — \
         the >i32::MAX region of the pad-and-accumulate assembly is corrupted (sc-12926 tier-3)"
    );
}
