//! sc-12758 — **Wan2.2 A14B z16 video VAE decode CUDA peak-VRAM sweep.** The z16 twin of the z48
//! `vae_decode_sweep.rs` (sc-7148): measures the *real* concurrent GPU memory peak of
//! [`WanVae16::decode`] (streaming) / [`WanVae16::decode_tiled`] / [`WanVae16::decode_budgeted`] across
//! a grid of output sizes × **spatial** tile sizes on the real Wan2.2-T2V-A14B z16 VAE, so the budgeted
//! decode cost model (`estimated_wan_z16_decode_peak_gib`, the `WAN_Z16_VAE_ACCUM_BYTES_PER_VOXEL` /
//! `WAN_Z16_VAE_FRAME_BYTES_PER_OUT_PX` constants in `vae16.rs`) can be **fit from CUDA measurements**.
//!
//! Loads only the VAE component of a Wan2.2-T2V-A14B snapshot (the ~500 MB z16 VAE, not the two ~28 GB
//! experts), so it is cheap. One config **per process** (env-driven) so an OOM on the largest configs
//! kills only this process and the driving shell loop keeps the earlier anchors. Synthetic (random)
//! latents — cost/scaling, not parity (parity lives in `vae16_tiling_cuda.rs`).
//!
//! **Streaming model note:** candle's `WanVae16::decode` decodes one latent frame at a time, so the
//! temporal axis is already memory-bounded — the per-tile activation spike scales with one output
//! *frame's* area (`tile_h·tile_w`), not the whole tile's volume. The sweep therefore tiles **only the
//! spatial axes** and reports `bytes_per_frame_px` (fits `FRAME`) alongside `bytes_per_out_vox`
//! (bounds `ACCUM`). Peak is sampled device-wide via `nvidia-smi` (see `candle_gen::testkit`).
//!
//! ```text
//! set CUDA_VISIBLE_DEVICES=0
//! set WAN_VAE16_SNAPSHOT=E:\staged\12402\wan-t2v-q4
//! set WAN_W=1280& set WAN_H=720& set WAN_FRAMES=81
//!   cargo test -p candle-gen-wan --features cuda --release --test vae16_decode_sweep -- --ignored --nocapture
//! # add WAN_TILE_PX=512 [WAN_OVERLAP_PX=64] for a fixed spatial-tile run (anchor fitting)
//! # add WAN_BUDGETED=1 [WAN_VAE_BUDGET_GIB=20] to exercise the production budgeted selector
//! # add WAN_GPU=0 to pick which GPU ordinal nvidia-smi samples (default 0; pair with CUDA_VISIBLE_DEVICES)
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen::testkit::{
    cuda_mempool_used_high_bytes, reset_cuda_mempool_high_water, used_mib, PeakSampler,
};
use candle_gen_wan::config::Vae16Config;
use candle_gen_wan::vae16::{auto_tiling_budgeted_wan_z16, WanVae16};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The VAE weight dtype for the sweep: `WAN_VAE_DTYPE=bf16` loads the A14B's shipped bf16 z16 VAE
/// (sc-12818); anything else (default) loads f32. The bf16-vs-f32 pair isolates whether running the
/// VAE bf16 shrinks the *decode's own* concurrent-live peak — independent of the denoise.
fn vae_dtype() -> DType {
    match std::env::var("WAN_VAE_DTYPE").ok().as_deref() {
        Some("bf16") | Some("BF16") => DType::BF16,
        _ => DType::F32,
    }
}

fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

/// Load just the z16 VAE from `$WAN_VAE16_SNAPSHOT/vae/diffusion_pytorch_model.safetensors` onto
/// cuda:`gpu`. Decode-only (`WanVae16::new`) — the sweep never encodes.
fn load_vae(gpu: usize) -> Option<(WanVae16, Device)> {
    let snap = std::env::var("WAN_VAE16_SNAPSHOT").ok()?;
    let dev = Device::new_cuda(gpu).expect("cuda device");
    let f = PathBuf::from(snap)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    // SAFETY: mmap of a read-only, process-owned weight file resolved from `$WAN_VAE16_SNAPSHOT`; not
    // mutated behind the mapping — the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[f], vae_dtype(), &dev).unwrap() };
    let vae = WanVae16::new(&Vae16Config::wan21(), vb).unwrap();
    Some((vae, dev))
}

#[test]
#[ignore = "needs a Wan2.2-T2V-A14B snapshot dir (WAN_VAE16_SNAPSHOT) + CUDA; GPU-heavy"]
fn wan_z16_vae_decode_sweep() {
    let gpu = env_usize("WAN_GPU", 0);
    let Some((vae, dev)) = load_vae(gpu) else {
        eprintln!("skip: set WAN_VAE16_SNAPSHOT to a Wan2.2-T2V-A14B snapshot dir");
        return;
    };
    let w_out = env_usize("WAN_W", 1280) as i32;
    let h_out = env_usize("WAN_H", 720) as i32;
    let frames = env_usize("WAN_FRAMES", 81) as i32;
    // z16: spatial ×8, temporal ×4 **causal** (out_f = 1 + (T_lat−1)·4 ⇒ frames = 1 + 4·k).
    assert_eq!(
        (frames - 1) % 4,
        0,
        "WAN_FRAMES must be 1 + 4·k (got {frames})"
    );
    assert_eq!(h_out % 8, 0, "WAN_H must be a multiple of 8");
    assert_eq!(w_out % 8, 0, "WAN_W must be a multiple of 8");
    let (t_lat, h_lat, w_lat) = ((frames - 1) / 4 + 1, h_out / 8, w_out / 8);

    // Warm the CUDA context / cuBLAS handles with a tiny streaming decode so the measured peak reflects
    // the real decode's working set, not one-time context creation.
    let warm = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
    let _ = vae.decode(&warm).unwrap();
    dev.synchronize().unwrap();

    // Synthetic latent [B=1, 16, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let latent = Tensor::randn(
        0f32,
        1f32,
        (1, 16, t_lat as usize, h_lat as usize, w_lat as usize),
        &dev,
    )
    .unwrap();
    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 4, h_lat * 8, w_lat * 8);

    // Tile selection: WAN_BUDGETED=1 exercises the production `auto_tiling_budgeted_wan_z16` selector
    // (honors WAN_VAE_BUDGET_GIB); else WAN_TILE_PX sets a fixed spatial tile (for anchor fitting); else
    // single-pass. The candle decode streams temporally, so there is no temporal tiling here.
    let tile_px = env_usize("WAN_TILE_PX", 0) as i32;
    let cfg: Option<TilingConfig> = if env_usize("WAN_BUDGETED", 0) == 1 {
        auto_tiling_budgeted_wan_z16(out_h, out_w, out_f)
            .expect("wan z16 decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        Some(TilingConfig::spatial_only(
            tile_px,
            env_usize("WAN_OVERLAP_PX", 64) as i32,
        ))
    } else {
        None
    };

    // Largest-tile extents in the cost model's convention (min(tile, out_dim) per axis); a None spatial
    // axis means full extent. `tile_f` is unused by the streaming model but kept for the anchor line.
    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
    let (tile_h, tile_w) = match &cfg {
        Some(c) => (
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_h as i64))
                .unwrap_or(out_h as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_w as i64))
                .unwrap_or(out_w as i64),
        ),
        None => (out_h as i64, out_w as i64),
    };
    let frame_px = tile_h * tile_w;

    let baseline_mib = used_mib(gpu).unwrap_or(0);
    let dtype = vae_dtype();
    println!(
        "\n=== wan z16 sweep [gpu {gpu}]: out {out_w}x{out_h}x{out_f}  latent[z16,T{t_lat},{h_lat},{w_lat}]  \
         tiled={}  cfg={cfg:?}  dtype={dtype:?}  baseline={baseline_mib} MiB ===",
        cfg.is_some(),
    );

    // sc-12818: measure the decode's TRUE concurrent-live peak via the driver mempool USED_MEM_HIGH
    // (accurate where the nvidia-smi sampler under-samples the im2col transients ~2×), reset right
    // before the decode so it isolates the decode's own working set. The nvidia-smi PeakSampler runs
    // alongside for the historical anchor line.
    reset_cuda_mempool_high_water(gpu as i32);
    let sampler = PeakSampler::start(gpu);
    let t = Instant::now();
    let video = match &cfg {
        Some(c) => vae.decode_tiled(&latent, c).unwrap(),
        None => vae.decode(&latent).unwrap(),
    };
    dev.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak_mib = sampler.stop();
    let decode_high_bytes = cuda_mempool_used_high_bytes(gpu as i32).unwrap_or(0);
    let decode_high_gib = gib(decode_high_bytes as f64);

    // Finiteness / range / shape (cheap sanity — the parity bound lives in vae16_tiling_cuda.rs). Cast
    // to f32 on host: a bf16 decode (WAN_VAE_DTYPE=bf16) cannot `to_vec1::<f32>` directly.
    let v = video
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "decode produced non-finite values"
    );
    assert_eq!(
        video.dims(),
        &[1, 3, out_f as usize, out_h as usize, out_w as usize],
        "unexpected output shape"
    );

    let peak_bytes = (peak_mib as f64) * 1024.0 * 1024.0;
    let peak_gib = gib(peak_bytes);
    println!(
        "[WAN z16 decode] dtype={dtype:?} -> {:?}  {secs:.1}s  nvidia-smi peak={peak_gib:.2} GiB \
         ({peak_mib} MiB) | USED_MEM_HIGH true peak={decode_high_gib:.2} GiB",
        video.dims()
    );
    // Parse-friendly anchor line (grep `^ANCHOR`): peak vs out voxels (ACCUM floor) and per-frame px
    // (the streaming per-tile FRAME term). `true_peak_gib` is the accurate USED_MEM_HIGH number.
    println!(
        "ANCHOR wanz16 dtype={dtype:?} out_vox={out_vox} frame_px={frame_px} peak_gib={peak_gib:.4} \
         true_peak_gib={decode_high_gib:.4} peak_mib={peak_mib} baseline_mib={baseline_mib} \
         bytes_per_out_vox={:.2} bytes_per_frame_px={:.2}",
        peak_bytes / out_vox as f64,
        peak_bytes / frame_px as f64,
    );
}
