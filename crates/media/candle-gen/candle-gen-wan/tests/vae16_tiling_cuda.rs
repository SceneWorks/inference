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

/// Load just the z16 VAE from `$WAN_VAE16_SNAPSHOT/vae` onto CUDA:0 at `dtype`, or `None` if the env
/// var is unset.
fn load_vae_dtype(dtype: DType) -> Option<(WanVae16, Device)> {
    let snap = std::env::var("WAN_VAE16_SNAPSHOT").ok()?;
    let dev = Device::new_cuda(0).expect("cuda:0");
    let f = PathBuf::from(snap)
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    // SAFETY: mmap of a read-only, process-owned weight file resolved from `$WAN_VAE16_SNAPSHOT`; not
    // mutated behind the mapping — the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[f], dtype, &dev).unwrap() };
    let vae = WanVae16::new(&Vae16Config::wan21(), vb).unwrap();
    Some((vae, dev))
}

/// The f32 VAE — the existing tiling tests' loader (the A14B ships bf16; see the sc-12818 parity test).
fn load_vae() -> Option<(WanVae16, Device)> {
    load_vae_dtype(DType::F32)
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

/// sc-12818 acceptance gate: the **bf16** z16 VAE decode must match the **f32** decode of the same
/// latent within the seam-free tolerance (~35 dB PSNR). Running the z16 VAE bf16 is the lever that
/// halves the A14B decode's otherwise-fixed ~30 GiB VRAM floor so it fits a 24 GiB card; the VAE is the
/// sensitive part, so if bf16 degraded the decode below tolerance it would NOT be viable and a different
/// lever would be needed. Uses the untiled streaming [`WanVae16::decode`] on a moderate 512²×21f latent
/// so the comparison isolates the bf16-vs-f32 decode numerics from the tiling approximation.
#[test]
#[ignore = "needs WAN_VAE16_SNAPSHOT + CUDA; the bf16-VAE acceptance gate (sc-12818)"]
fn wan_z16_bf16_vs_f32_decode_parity() {
    let Some((vae_f32, dev)) = load_vae_dtype(DType::F32) else {
        eprintln!("WAN_VAE16_SNAPSHOT unset — skipping");
        return;
    };
    let (vae_bf16, _) = load_vae_dtype(DType::BF16).unwrap();

    // The bf16 VAE really loaded bf16 weights (the VRAM-floor win); the f32 baseline stayed f32.
    assert_eq!(
        vae_bf16.dtype(),
        DType::BF16,
        "the A14B z16 VAE must load bf16 weights (sc-12818)"
    );
    assert_eq!(vae_f32.dtype(), DType::F32);

    // 512²×21f output (64×64 latent, 6 latent frames) — under the im2col cap, so a single untiled pass.
    let z = Tensor::randn(0f32, 1f32, (1, 16, 6, 64, 64), &dev).unwrap();
    let dec_f32 = vae_f32.decode(&z).unwrap();
    // candle's CPU has no bf16 matmul, but this decode runs on CUDA; cast to f32 on host for comparison.
    let dec_bf16 = vae_bf16.decode(&z).unwrap().to_dtype(DType::F32).unwrap();
    assert_eq!(dec_f32.dims(), &[1, 3, 21, 512, 512], "unexpected output shape");
    assert_eq!(dec_bf16.dims(), dec_f32.dims());
    assert_finite_and_in_range(&dec_bf16, "bf16 decode");

    let (psnr, maxd) = psnr_and_maxdiff(&dec_f32, &dec_bf16);
    eprintln!("[sc-12818] bf16-VAE vs f32-VAE decode: PSNR={psnr:.2} dB  max_abs_diff={maxd:.4}");
    assert!(
        psnr > 35.0,
        "bf16 VAE decode degrades vs f32 below the seam-free tolerance: PSNR={psnr:.2} dB (need >35) \
         — bf16 VAE would not be viable and a different VRAM lever would be required"
    );
}

#[test]
#[ignore = "needs WAN_VAE16_SNAPSHOT (a Wan2.2-T2V-A14B snapshot dir) + CUDA weights"]
fn wan_z16_spatial_tiling_decode() {
    let Some((vae, dev)) = load_vae() else {
        eprintln!("WAN_VAE16_SNAPSHOT unset — skipping");
        return;
    };

    // 640²×5-frame output (80×80 latent, 2 latent frames). 640 > the im2col SAFE_PX cap (512), so the
    // budgeted decode is forced to the 512 px cap tile even at a huge budget (see the huge-budget check).
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

    // Budgeted routing — huge budget, but 640 > the im2col SAFE_PX cap (512), so `decode_budgeted` is
    // FORCED to the 512 px cap tile regardless of budget (sc-12758 `cap_spatial_for_im2col`). It is
    // therefore the 512-cap tiling — bit-exactly (same `decode_tiled` routing) — NOT the untiled
    // single-pass `decode`. (Pre-cap this asserted `== decode`; the cap diverts any >512 px frame to
    // tiling, and the untiled path stays correct on its own via the sc-12773 conv2d chunking — exercised
    // by `wan_z16_render_scale_seamfree_parity`.)
    let cap_tiled = vae
        .decode_tiled(&z, &TilingConfig::spatial_only(512, 64))
        .unwrap();
    std::env::set_var("WAN_VAE_BUDGET_GIB", "100000");
    let big = vae.decode_budgeted(&z).unwrap();
    let (_, maxd_big) = psnr_and_maxdiff(&cap_tiled, &big);
    assert!(
        maxd_big < 1e-6,
        "decode_budgeted(huge budget) at 640² must equal the im2col-capped 512 px tiling (sc-12758 cap), max_abs_diff={maxd_big}"
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
/// paths through the same blend/stitch, so this isolates the seam quality. sc-12773: the **`untiled`**
/// single-pass decode used to silently corrupt at this geometry — its final conv2d im2col (~796M elems)
/// overflowed candle's CUDA conv2d, so the untiled decode diverged from BOTH tiled decodes by an
/// identical ~15.6 dB while the two tiled decodes agreed at ~55 dB (the untiled path was the corrupt
/// one). Now that the z16 VAE conv2d is im2col-chunked (`chunked_conv2d`), the untiled decode is
/// **correct**: it must agree with the trusted 256 px reference far above that corrupt floor (measured
/// ~52.6 dB — the untiled path attends globally, so it is if anything *closer* to the true decode than
/// the tiled approximation). This test now asserts that untiled parity as the sc-12773 acceptance.
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

    // sc-12773: the *untiled* single-pass decode (the ~48 GB spike). It used to silently corrupt at this
    // geometry (final conv2d im2col ~796M elems → candle's CUDA overflow band), diverging from BOTH tiled
    // decodes by an identical ~15.6 dB. With the VAE conv2d now im2col-chunked it decodes correctly.
    let untiled = vae.decode(&z).unwrap();
    assert_eq!(untiled.dims(), ref256.dims());
    assert_finite_and_in_range(&untiled, "decode(untiled render-scale)");
    let (psnr_u_ref, _) = psnr_and_maxdiff(&ref256, &untiled);
    let (psnr_u_prod, _) = psnr_and_maxdiff(&prod448, &untiled);
    eprintln!(
        "[sc-12773] RENDER-SCALE untiled decode vs ref256={psnr_u_ref:.2} dB, vs prod448={psnr_u_prod:.2} dB \
         (was ~15.6 dB pre-fix ⇒ corrupt; high now ⇒ the chunked conv2d decodes the untiled path correctly)"
    );

    assert!(
        psnr > 35.0,
        "render-scale production tiling must be seam-free vs the trusted 256 px reference \
         (~35 dB rule): PSNR={psnr:.2} dB"
    );
    // sc-12773 acceptance: the untiled hi-res decode is no longer corrupt — it agrees with the trusted
    // 256 px tiling far above the old ~15.6 dB corruption floor (a conv2d-chunking regression craters it
    // back down). Threshold 48 dB honors the AC's "~50+ dB": measured ~52.6 (stable — PSNR averages over
    // 224M output elems), corrupt ~15.6 — a wide, unambiguous separation.
    assert!(
        psnr_u_ref > 48.0,
        "untiled hi-res decode must be uncorrupted (sc-12773): vs trusted ref256 PSNR={psnr_u_ref:.2} dB \
         (corrupt was ~15.6 dB) — the VAE conv2d im2col chunking regressed"
    );
}
