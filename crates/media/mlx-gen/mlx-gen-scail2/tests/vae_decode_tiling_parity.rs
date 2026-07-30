//! **sc-15325 — SCAIL-2's tiled VAE decode, measured against a single pass.**
//!
//! SCAIL-2 shipped a byte-for-byte copy of the Krea Realtime decode-window arithmetic: a local
//! `DECODE_TILE_BUDGET_PXFRAMES = 3.5e6` output px·frame budget, snapped to the z16 ×4 temporal
//! stride, with `overlap = (tile_frames / 4).max(1)`. At every bucket ≥ ~233k px/frame that is an
//! **8-output-frame** window — *two* latent frames — and after `TilePlan::plan`'s ÷4 the overlap is
//! **0** latent frames (SCAIL-2 never got the sc-8446 `.max(temporal_scale)` floor, so it was strictly
//! worse than krea's). The krea measurements put that at 18.5/255 mean abs err against a single-pass
//! decode of the same latents, with 26.6 % of a worst frame blown to near-white.
//!
//! It was never measured *here*, on SCAIL-2's own VAE weights, which is why sc-15325 refuses to call
//! it either way on inference alone. This test does the measurement: encode one clip through
//! SCAIL-2's `vae.safetensors`, then decode the **same latent** single-pass, through the old policy,
//! and through the policy the crate now uses — reporting mean abs err, highlight clipping and MLX
//! active peak for each.
//!
//! ```text
//! SCAIL2_SNAPSHOT_DIR=/path/to/scail2-mlx \
//!   cargo test -p mlx-gen-scail2 --release --test vae_decode_tiling_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_wan::pipeline::auto_tiling_budgeted_z16_quality_overlap;
use mlx_gen_wan::WanVae;
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

/// A smooth (band-limited) moving clip `[1, 3, F, H, W]` in [-1, 1]. Smooth on purpose: a z16 VAE
/// cannot reproduce a hard discontinuity, so a sawtooth source would bury the tiling error under the
/// VAE's own round-trip error.
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

/// Mean |Δ| on the 0-255 scale a viewer sees. MLX is lazy — `eval` before the read-back or this
/// silently measures an unmaterialized graph (the failure mode that produced three ~0 readings in this
/// epic).
fn mean_abs_delta_255(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    let m = mlx_rs::ops::mean(&d, None).unwrap();
    mlx_rs::transforms::eval([&m]).unwrap();
    m.item::<f32>() as f64 * 127.5
}

/// `(mean %, worst-frame %)` of pixels clipped to near-white — the metric a viewer actually notices.
fn clip_stats(v: &Array) -> (f64, f64) {
    let sh = v.shape().to_vec();
    let (f, hh, ww) = (sh[2], sh[3], sh[4]);
    mlx_rs::transforms::eval([v]).unwrap();
    let flat = v.as_slice::<f32>();
    let per_frame = (hh as usize) * (ww as usize);
    let stride = (f as usize) * per_frame;
    let mut pcts = Vec::with_capacity(f as usize);
    for t in 0..f as usize {
        let mut clipped = 0usize;
        for i in 0..per_frame {
            let base = t * per_frame + i;
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

/// The policy SCAIL-2 shipped before sc-15325, reproduced verbatim from the deleted code. This is what
/// makes the test a guard rather than a snapshot: it is the exact config the old crate emitted.
fn old_scail2_policy(out_h: i32, out_w: i32, out_frames: i32) -> TilingConfig {
    const BUDGET_PXFRAMES: i64 = 3_500_000;
    let px_per_frame = (out_h as i64) * (out_w as i64);
    let budget_frames = BUDGET_PXFRAMES / px_per_frame.max(1);
    let write_cap = VaeTiling::WAN.writable_frame_cap(out_h, out_w);
    let win = budget_frames.min(write_cap) as i32;
    let tile_frames = (win / 4 * 4).clamp(8, out_frames.max(8));
    // NOTE: `.max(1)`, not krea's `.max(4)` — SCAIL-2 never received the sc-8446 overlap floor, so
    // this divides to **0** latent frames: hard-cut windows with no blend at all.
    TilingConfig::temporal_only(tile_frames, (tile_frames / 4).max(1))
}

#[test]
#[ignore = "needs a scail2-mlx snapshot holding vae.safetensors (SCAIL2_SNAPSHOT_DIR); GPU-heavy"]
fn scail2_tiled_decode_tracks_single_pass() {
    let Some(dir) = env_path("SCAIL2_SNAPSHOT_DIR") else {
        panic!("set SCAIL2_SNAPSHOT_DIR to a scail2-mlx snapshot dir holding vae.safetensors");
    };
    // 832×480 is SCAIL-2's reference bucket and one of the five the old budget collapsed at.
    let w = env_i32("SCAIL2_W", 832);
    let h = env_i32("SCAIL2_H", 480);
    let t_lat = env_i32("SCAIL2_LATENT_FRAMES", 9); // 36 output frames — under the write cap
    let f_in = 1 + (t_lat - 1) * 4; // VAE-aligned source length

    let vae = {
        let wts = Weights::from_file(dir.join("vae.safetensors")).expect("open scail2 vae");
        WanVae::from_weights(&wts).expect("load the z16 Wan VAE")
    };
    let latent = {
        let z = vae.encode(&smooth_video(f_in, h, w)).expect("VAE encode");
        mlx_rs::transforms::eval([&z]).expect("materialize the latent");
        z
    };
    let ls = latent.shape().to_vec();
    let out_frames = ls[2] * 4; // z16 is non-causal: T_lat → 4·T_lat
    println!(
        "=== SCAIL-2 tiled-decode parity: {w}x{h}, latent [z{}, T{}, {}, {}] -> {out_frames} frames ===",
        ls[1], ls[2], ls[3], ls[4]
    );

    // ⚠️ The single-pass reference is only valid below the z16 write cap (56 output frames at
    // 832×480). Past it MLX writes silently wrong pixels and every tiled candidate looks catastrophic
    // against a corrupt reference (sc-15402). Refuse rather than measure garbage.
    let cap = VaeTiling::WAN.writable_frame_cap(h, w);
    assert!(
        out_frames as i64 <= cap,
        "needs a VALID single-pass reference: {out_frames} output frames exceeds the z16 write cap \
         {cap} at {w}x{h} — lower SCAIL2_LATENT_FRAMES to {}",
        cap / 4
    );
    mlx_rs::memory::clear_cache();

    let decode = |cfg: Option<&TilingConfig>| -> (Array, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = match cfg {
            Some(c) => vae
                .decode_tiled(&latent, c, Some(&CancelFlag::default()))
                .expect("tiled decode"),
            None => vae.decode(&latent).expect("single-pass decode"),
        };
        // MLX is lazy: without this the peak reads ~0 and the delta reads an unbuilt graph.
        mlx_rs::transforms::eval([&out]).expect("materialize the decode");
        (out.clone(), mlx_rs::memory::get_peak_memory())
    };

    let (single, single_peak) = decode(None);
    let (s_cm, s_cw) = clip_stats(&single);
    println!(
        "  single-pass (reference): clipping {s_cm:.2}% / {s_cw:.2}%, peak {:.2} GiB",
        gib(single_peak)
    );

    let old = old_scail2_policy(h, w, out_frames);
    let ot = old.temporal.expect("the old policy is temporal-only");
    let (old_out, old_peak) = decode(Some(&old));
    let old_err = mean_abs_delta_255(&single, &old_out);
    let (o_cm, o_cw) = clip_stats(&old_out);
    println!(
        "  OLD policy tile {} / overlap {} (latent {} / {}): mean abs err {old_err:.2}/255, \
         clipping {o_cm:.2}% / {o_cw:.2}%, peak {:.2} GiB",
        ot.tile_frames,
        ot.overlap_frames,
        ot.tile_frames / 4,
        (ot.overlap_frames / 4).min(ot.tile_frames / 4 - 1),
        gib(old_peak)
    );

    // The NEW policy, budget-pinned to what the OLD one actually cost, so quality is not bought with
    // memory. This is the *same* call `generate.rs` now makes.
    let budget = env_i32("SCAIL2_DECODE_BUDGET_GIB", gib(old_peak).ceil() as i32);
    std::env::set_var("WAN_VAE_BUDGET_GIB", budget.to_string());
    let new = auto_tiling_budgeted_z16_quality_overlap(h, w, out_frames)
        .expect("the new policy must be feasible at the old policy's peak")
        .expect("the clip must still tile at that budget");
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    let (new_out, new_peak) = decode(Some(&new));
    let new_err = mean_abs_delta_255(&single, &new_out);
    let (n_cm, n_cw) = clip_stats(&new_out);
    println!(
        "  NEW policy {new:?} (budget pinned {budget} GiB): mean abs err {new_err:.2}/255, clipping \
         {n_cm:.2}% / {n_cw:.2}%, peak {:.2} GiB",
        gib(new_peak)
    );

    // 1. The defect is real on SCAIL-2's own weights — the verdict this test exists to deliver.
    assert!(
        old_err > 5.0,
        "SCAIL-2's old decode window is only {old_err:.2}/255 from single-pass — it does NOT share \
         the krea defect, and sc-15325's exposure analysis needs revisiting"
    );
    // 2. The policy it uses now is at single-pass level.
    assert!(
        new_err < 4.0 && new_err < old_err * 0.35,
        "SCAIL-2's new decode policy is {new_err:.2}/255 from single-pass (old: {old_err:.2})"
    );
    // 3. ...on the metric a viewer sees.
    assert!(
        n_cw <= s_cw + 1.5,
        "the new policy's worst-frame clipping is {n_cw:.2}% against {s_cw:.2}% single-pass"
    );
    // 4. ...without exceeding the pinned budget.
    assert!(
        gib(new_peak) <= budget as f64 * 1.15,
        "the new policy peaked at {:.2} GiB against a {budget} GiB budget — the z16 cost model is \
         under-predicting and `mlx.minMemoryGb` cannot be trusted",
        gib(new_peak)
    );
}
