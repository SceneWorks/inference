//! Real-weight CUDA validation of the z16 **spatial-tiling** decode (sc-12758) — the A14B twin of the
//! z48 `vae_tiling_cuda.rs`.
//!
//! Loads only the z16 VAE component of a Wan2.2-T2V-A14B snapshot (~500 MB, not the two ~28 GB experts),
//! so it is cheap, and exercises [`WanVae16::decode_tiled`] / [`WanVae16::decode_budgeted`] on real z16
//! weights. `#[ignore]`d by default — needs the CUDA backend, a snapshot dir, and the weights:
//!
//! ```text
//! set WAN_VAE16_SNAPSHOT=E:\staged\12402\wan-t2v-q4
//! cargo test -p candle-gen-wan --features cuda --release --test vae16_tiling_cuda -- --ignored --nocapture
//! ```
//!
//! NOTE: the z16 decoder has **global per-frame spatial attention** (`MidAttn` softmaxes over all H·W),
//! so spatial tiling is intentionally an *approximation* — each tile attends only within itself and the
//! overlapping trapezoidal blend softens the seams. So `decode_tiled` is **not** bit-exact vs. the
//! single-pass `decode`; seam-free quality is asserted as **PSNR ~35 dB** (the standing tiled-VAE rule),
//! NOT a tight max-abs-diff. The test asserts shape/finiteness/range exactly, the budgeted-routing
//! equivalences exactly, PSNR ≥ 30 dB for the tiled decode, and prints the measured PSNR / max-abs-diff.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen_wan::config::Vae16Config;
use candle_gen_wan::vae16::WanVae16;

/// Load just the z16 VAE from `$WAN_VAE16_SNAPSHOT/vae` onto CUDA:0, or `None` if the env var is unset.
fn load_vae() -> Option<(WanVae16, Device)> {
    let snap = std::env::var("WAN_VAE16_SNAPSHOT").ok()?;
    let dev = Device::new_cuda(0).expect("cuda:0");
    let f = PathBuf::from(snap)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    // SAFETY: mmap of a read-only, process-owned weight file resolved from `$WAN_VAE16_SNAPSHOT`; not
    // mutated behind the mapping — the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[f], DType::F32, &dev).unwrap() };
    let vae = WanVae16::new(&Vae16Config::wan21(), vb).unwrap();
    Some((vae, dev))
}

fn assert_finite_and_in_range(t: &Tensor, label: &str) {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{label}: produced non-finite values"
    );
    let (lo, hi) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    assert!(
        lo >= -1.01 && hi <= 1.01,
        "{label}: out of [-1,1] range: [{lo}, {hi}]"
    );
}

/// PSNR (dB) over the `[-1,1]` decode range (peak-to-peak 2.0) + max-abs-diff, between two
/// equally-shaped tensors on host. The seam-free metric for tiled VAE parity (my standing rule).
fn psnr_and_maxdiff(a: &Tensor, b: &Tensor) -> (f64, f32) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    let (mut sse, mut maxd) = (0f64, 0f32);
    for (&x, &y) in a.iter().zip(&b) {
        let d = (x - y) as f64;
        sse += d * d;
        maxd = maxd.max((x - y).abs());
    }
    let mse = sse / a.len() as f64;
    // Range is [-1,1] ⇒ peak-to-peak = 2.0. PSNR = 10·log10(peak² / MSE).
    let psnr = if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (4.0 / mse).log10()
    };
    (psnr, maxd)
}

#[test]
#[ignore = "needs WAN_VAE16_SNAPSHOT (a Wan2.2-T2V-A14B snapshot dir) + CUDA weights"]
fn wan_z16_spatial_tiling_decode() {
    let Some((vae, dev)) = load_vae() else {
        eprintln!("WAN_VAE16_SNAPSHOT unset — skipping");
        return;
    };

    // 640²×5-frame output (80×80 latent, 2 latent frames). 640 > 512 ⇒ a single high-res frame. Small
    // enough that the ×8 write bound (96 ch) does not force tiling on the huge-budget check below.
    let z = Tensor::randn(0f32, 1f32, (1, 16, 2, 80, 80), &dev).unwrap();

    // Baseline: the streaming single-pass decode.
    let base = vae.decode(&z).unwrap();
    assert_eq!(base.dims(), &[1, 3, 5, 640, 640], "unexpected output shape");
    assert_finite_and_in_range(&base, "decode");

    // Spatial tiling: 256 px tiles (32 latent), 64 px overlap ⇒ several tiles per axis.
    let cfg = TilingConfig::spatial_only(256, 64);
    let tiled = vae.decode_tiled(&z, &cfg).unwrap();
    assert_eq!(tiled.dims(), base.dims(), "tiled shape != baseline");
    assert_finite_and_in_range(&tiled, "decode_tiled");

    let (psnr, maxd) = psnr_and_maxdiff(&base, &tiled);
    eprintln!("[sc-12758] decode vs decode_tiled(256/64): PSNR={psnr:.2} dB  max_abs_diff={maxd:.4}");
    // Global-attention tiling perturbs the result, but the overlapping trapezoidal blend keeps it
    // seam-free by PSNR. 30 dB floor leaves margin while still cratering on a blend/offset regression
    // (a broken stitch drops PSNR to the teens). The anchor record is the printed PSNR above.
    assert!(
        psnr > 30.0,
        "tiled decode diverged from baseline: PSNR={psnr:.2} dB (expected seam-free ~35 dB)"
    );

    // Budgeted routing — huge budget ⇒ single-pass ⇒ bit-exact with `decode`.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "100000");
    let big = vae.decode_budgeted(&z).unwrap();
    let (_, maxd_big) = psnr_and_maxdiff(&base, &big);
    assert!(
        maxd_big < 1e-6,
        "decode_budgeted(huge budget) must equal single-pass decode, max_abs_diff={maxd_big}"
    );

    // Budgeted routing — tight budget ⇒ must tile, stay finite/in-range, and differ from single-pass.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "4");
    let small = vae.decode_budgeted(&z).unwrap();
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    assert_eq!(small.dims(), base.dims());
    assert_finite_and_in_range(&small, "decode_budgeted(tiny)");
    let (psnr_small, maxd_small) = psnr_and_maxdiff(&base, &small);
    eprintln!("[sc-12758] decode vs decode_budgeted(4 GiB): PSNR={psnr_small:.2} dB  max_abs_diff={maxd_small:.4}");
    assert!(
        maxd_small > 0.0,
        "tiny-budget decode_budgeted should have tiled (differ from single-pass)"
    );
    assert!(
        psnr_small > 30.0,
        "tiny-budget tiled decode must stay seam-free: PSNR={psnr_small:.2} dB"
    );
}

/// Production-scale seam-free check (sc-12758 AC): decode the **real A14B T2V render geometry**
/// (1280×720 / 81 frames) three ways on the SAME latent and compare by PSNR:
///  - `ref256` — a fine 256 px spatial tiling. This is the **trusted reference**: every per-tile
///    conv2d im2col stays small (~56M elems, well under candle's CUDA conv2d overflow zone), so it
///    decodes correctly at any output resolution.
///  - `prod448` — the production budgeted selector at an emulated ~24 GB card (`WAN_VAE_BUDGET_GIB=20`
///    ⇒ the 448 px tile) — the thing that actually ships.
///  - `untiled` — the single-pass `decode` (the ~48 GB spike this story tiles away).
///
/// The seam-free AC is asserted on **`prod448` vs `ref256`** (~35 dB standing rule) — both are tiled
/// paths through the same blend/stitch, so this isolates the seam quality. The `untiled` comparison is
/// recorded for the finding that the single-pass decode of a 1280×720 frame drives its final conv2d
/// im2col (~796M elems) into candle's CUDA overflow zone (this box's raw z16 conv2d is NOT chunked),
/// so it corrupts — which the tiling avoids (the MLX write-cap + the budget both force tiling at this
/// geometry, so the shipped A14B decode never takes the untiled path here).
#[test]
#[ignore = "needs WAN_VAE16_SNAPSHOT + CUDA; decodes an untiled 1280×720×81 frame (~48 GB) — big-VRAM box only"]
fn wan_z16_render_scale_seamfree_parity() {
    let Some((vae, dev)) = load_vae() else {
        eprintln!("WAN_VAE16_SNAPSHOT unset — skipping");
        return;
    };
    // 1280×720 / 81 output frames ⇒ latent [1,16,21,90,160] (×8 spatial, ×4 causal temporal).
    let z = Tensor::randn(0f32, 1f32, (1, 16, 21, 90, 160), &dev).unwrap();

    // Trusted reference: 256 px tiling (small per-tile conv2d ⇒ no im2col overflow).
    let ref256 = vae
        .decode_tiled(&z, &TilingConfig::spatial_only(256, 64))
        .unwrap();
    assert_eq!(ref256.dims(), &[1, 3, 81, 720, 1280]);
    assert_finite_and_in_range(&ref256, "decode_tiled(256, render-scale)");

    // Production budgeted decode at an emulated ~24 GB card. The selector tiles spatially (448 px).
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");
    let prod448 = vae.decode_budgeted(&z).unwrap();
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    assert_eq!(prod448.dims(), ref256.dims());
    assert_finite_and_in_range(&prod448, "decode_budgeted(render-scale, 20 GiB)");

    let (psnr, maxd) = psnr_and_maxdiff(&ref256, &prod448);
    eprintln!(
        "[sc-12758] RENDER-SCALE 1280x720x81 decode_budgeted(20 GiB, 448px) vs decode_tiled(256px): \
         PSNR={psnr:.2} dB  max_abs_diff={maxd:.4}"
    );

    // Record the untiled comparison (the ~48 GB single-pass spike — expected to diverge, corrupted by
    // the un-chunked conv2d im2col overflow at 1280×720; the tiling is what avoids it).
    let untiled = vae.decode(&z).unwrap();
    let (psnr_u_ref, _) = psnr_and_maxdiff(&ref256, &untiled);
    let (psnr_u_prod, _) = psnr_and_maxdiff(&prod448, &untiled);
    eprintln!(
        "[sc-12758] RENDER-SCALE untiled decode vs ref256={psnr_u_ref:.2} dB, vs prod448={psnr_u_prod:.2} dB \
         (low ⇒ the untiled 796M-elem conv2d im2col is corrupted; tiling is the correct path)"
    );

    assert!(
        psnr > 35.0,
        "render-scale production tiling must be seam-free vs the trusted 256 px reference \
         (~35 dB rule): PSNR={psnr:.2} dB"
    );
}
