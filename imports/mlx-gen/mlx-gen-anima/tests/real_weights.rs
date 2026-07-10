//! Real-weights tests for mlx-gen-anima (sc-10515). `#[ignore]`d — they need the licensed
//! `circlestone-labs/Anima` snapshot in the HF cache and Metal. Run with:
//!   cargo test -p mlx-gen-anima --release --test real_weights -- --ignored --nocapture
//!
//! The snapshot dir is resolved by glob (no hardcoded sha); PNG output goes to `$ANIMA_OUT`
//! (default `/tmp/anima_sc10515`).

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::runtime::CancelFlag;
use mlx_gen::weights::Weights;
use mlx_gen::{Progress, WeightsSource};

use mlx_gen_anima::conditioner::AnimaTextConditioner;
use mlx_gen_anima::config::{ConditionerConfig, DitConfig, Variant};
use mlx_gen_anima::loader::split_anima_keys;
use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};
use mlx_gen_anima::tokenizer::AnimaTokenizers;
use mlx_gen_anima::transformer::CosmosDiT;

/// Glob the Anima snapshot's `split_files/` dir from the HF cache (no hardcoded sha).
fn split_files() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("ANIMA_OUT").unwrap_or_else(|_| "/tmp/anima_sc10515".into()))
}

fn dit_file(split: &std::path::Path, v: Variant) -> PathBuf {
    split.join("diffusion_models").join(v.dit_filename())
}

/// `(grayscale_std, coarse_std)`. `std` is the full-image contrast (≈0 ⇒ blank wash). `coarse_std` is
/// the std of a 32×32 block-averaged downsample: real images keep strong coarse layout (subject vs
/// background) so it stays high, while noise — even the VAE's 8×8-upsampled latent noise — averages
/// out to a near-uniform coarse map (low `coarse_std`). Together they separate {blank wash, noise,
/// coherent image}. (A naive neighbor-diff ratio is fooled by the VAE's spatial upsampling.)
fn coherence(img: &Image) -> (f32, f32) {
    let (w, h) = (img.width as usize, img.height as usize);
    let gray: Vec<f32> = img
        .pixels
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = gray.iter().sum::<f32>() / gray.len() as f32;
    let std = (gray.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gray.len() as f32).sqrt();
    let g = 32usize;
    let (bh, bw) = (h / g, w / g);
    let mut coarse = vec![0f32; g * g];
    for by in 0..g {
        for bx in 0..g {
            let mut s = 0.0f32;
            for y in 0..bh {
                for x in 0..bw {
                    s += gray[(by * bh + y) * w + (bx * bw + x)];
                }
            }
            coarse[by * g + bx] = s / (bh * bw) as f32;
        }
    }
    let cm = coarse.iter().sum::<f32>() / coarse.len() as f32;
    let coarse_std =
        (coarse.iter().map(|&x| (x - cm).powi(2)).sum::<f32>() / coarse.len() as f32).sqrt();
    (std, coarse_std)
}

/// Flatten a (possibly bf16) array to host f32.
fn to_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Max |a - b| over two equally-shaped arrays (host f32).
fn maxabs_diff(a: &Array, b: &Array) -> f32 {
    let (av, bv) = (to_f32(a), to_f32(b));
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter()
        .zip(&bv)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn save_png(img: &Image, path: &std::path::Path) {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("valid RGB buffer");
    buf.save(path).expect("save png");
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(img.width, 1024);
    assert_eq!(img.height, 1024);
    let (std, coarse_std) = coherence(img);
    println!("[{label}] grayscale std = {std:.2}, coarse32 std = {coarse_std:.2}");
    assert!(std > 8.0, "{label}: image is near-blank (std {std:.2})");
    // Real generations carry strong coarse layout (coarse32 std ~30-40); VAE-decoded noise averages to
    // a near-uniform coarse map (coarse32 std < ~8). A coherent anime image clears this easily.
    assert!(
        coarse_std > 12.0,
        "{label}: output lacks coarse structure — looks like noise, not a coherent image (coarse32 std {coarse_std:.2})"
    );
}

// -------------------------------------------------------------------------------------------------
// Structural / shape tests
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn checkpoint_split_is_118_adapter_567_dit() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let (dit, adapter) = split_anima_keys(&w);
    println!("dit keys = {}, adapter keys = {}", dit.len(), adapter.len());
    assert_eq!(adapter.len(), 118, "expected 118 llm_adapter tensors");
    assert_eq!(dit.len(), 567, "expected 567 DiT tensors");
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn conditioner_output_is_b_512_1024() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let cond =
        AnimaTextConditioner::from_weights(&w, "net.llm_adapter", ConditionerConfig::anima())
            .unwrap();
    // dummy Qwen3 source states [1, 4, 1024] + T5 ids [1, 3] → conditioner must right-pad to 512.
    let source = mlx_rs::random::normal::<f32>(&[1, 4, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let t5_ids = Array::from_slice(&[10i32, 42, 7], &[1, 3]);
    let out = cond.forward(&source, &t5_ids, Dtype::Bfloat16).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 512, 1024],
        "conditioner must emit exactly 512 text tokens"
    );
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn dit_forward_17ch_patch_and_5d_latent_roundtrip() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let dit = CosmosDiT::from_weights(&w, "net", DitConfig::anima()).unwrap();
    // 5-D latent [B,16,1,Hl,Wl] (Hl=Wl=64 → 512² image); patch-embed prepends the 17th (mask) channel.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 64, 64], None, None, None).unwrap();
    let sigma = Array::from_slice(&[1.0f32], &[1]);
    let encoder = mlx_rs::random::normal::<f32>(&[1, 512, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let v = dit
        .forward(&latent, &sigma, &encoder, Dtype::Bfloat16)
        .unwrap();
    // velocity must have the same 5-D latent shape (proves patchify(17ch) + unpatchify roundtrip).
    assert_eq!(v.shape(), &[1, 16, 1, 64, 64]);
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn dit_rejects_out_of_range_size_per_axis() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let dit = CosmosDiT::from_weights(&w, "net", DitConfig::anima()).unwrap();
    // latent Hl=250 → post-patch 125 > max_size 120 ⇒ RoPE must reject, not index OOB.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 250, 64], None, None, None).unwrap();
    let sigma = Array::from_slice(&[1.0f32], &[1]);
    let encoder = mlx_rs::random::normal::<f32>(&[1, 512, 1024], None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    assert!(dit
        .forward(&latent, &sigma, &encoder, Dtype::Bfloat16)
        .is_err());
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn vae_decode_shape() {
    let split = split_files().expect("Anima snapshot");
    let vae = mlx_gen_anima::load_vae(split.join("vae/qwen_image_vae.safetensors")).unwrap();
    // a 5-D latent [1,16,1,128,128] → 1024² image.
    let latent = mlx_rs::random::normal::<f32>(&[1, 16, 1, 128, 128], None, None, None).unwrap();
    let img = vae.decode(&latent).unwrap();
    assert_eq!(img.shape(), &[1, 3, 1, 1024, 1024]);
}

// NOTE: the flow-velocity convention (the DiT is a standard flow denoiser, `v ≈ ε − x0`) and the raw-σ
// timestep have a dedicated `cos(v, ε − x0)` regression guard against a KNOWN VAE-encoded latent
// (≈0.9+ with timestep=σ; a negated velocity flips it strongly negative; a σ·1000 timestep collapses
// it toward 0). That check materializes many arrays, so — rather than run it in this shared binary,
// where mlx-rs's single Metal default stream can cross-contaminate — it lives in its own
// integration-test binary at `tests/velocity_convention.rs` (also `#[ignore]`d / real-weights-gated).
// The end-to-end `generate_*` test below is a second, coarser guard: a wrong sign or timestep collapses
// the output into a wash/noise that `assert_coherent` rejects. See `pipeline.rs` for the convention.

// -------------------------------------------------------------------------------------------------
// Acceptance: generate a real, coherent image for all three variants (bf16, 1024², fixed seed).
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (real 2B DiT denoise)"]
fn generate_all_three_variants_1024() {
    let split = split_files().expect("Anima snapshot");
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();
    let prompt =
        "an anime girl with long silver hair and blue eyes, detailed illustration, masterpiece";

    // Turbo first (cheapest: 10 steps, CFG-free) so a fundamental pipeline bug surfaces fast.
    for variant in [Variant::Turbo, Variant::Base, Variant::Aesthetic] {
        let pipeline =
            AnimaPipeline::from_source(&WeightsSource::Dir(split.clone()), variant).unwrap();

        // Default (recommended er_sde) solver, story-default steps/CFG.
        let opts = GenOptions {
            width: 1024,
            height: 1024,
            steps: variant.default_steps() as usize,
            guidance: variant.default_guidance(),
            seed: 42,
            sampler: None,
        };
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        let img = pipeline
            .generate(prompt, "", variant, &opts, &cancel, &mut prog)
            .unwrap();
        let path = out.join(format!("{}_1024_er_sde.png", variant.id()));
        save_png(&img, &path);
        println!("wrote {}", path.display());
        assert_coherent(&img, variant.id());

        // Base: also render with the reference Euler solver (matches diffusers FlowMatchEuler) as a
        // cross-check that the DiT/conditioner port is correct independent of the stochastic solver.
        if variant == Variant::Base {
            let opts_euler = GenOptions {
                sampler: Some("euler".into()),
                ..GenOptions {
                    width: 1024,
                    height: 1024,
                    steps: variant.default_steps() as usize,
                    guidance: variant.default_guidance(),
                    seed: 42,
                    sampler: None,
                }
            };
            let img_e = pipeline
                .generate(prompt, "", variant, &opts_euler, &cancel, &mut prog)
                .unwrap();
            let path_e = out.join(format!("{}_1024_euler.png", variant.id()));
            save_png(&img_e, &path_e);
            println!("wrote {}", path_e.display());
            assert_coherent(&img_e, "anima_base_euler");
        }

        mlx_rs::memory::clear_cache();
    }
}

// -------------------------------------------------------------------------------------------------
// Prompt weighting (sc-10566) — the EPIC ACCEPTANCE proof that `(chibi:2)` changes the output, and
// that the weight hits the **T5 query-token path**, not Qwen.
//
// Reference convention (read, NOT vendored — ComfyUI is GPL-3.0, mlx-gen is Apache-2.0):
//   * comfy/text_encoders/anima.py L26-27  — Qwen token weights forced to 1.0; T5 weights preserved.
//   * comfy/ldm/anima/model.py    L198-206 — `out = self.llm_adapter(...); out = out * t5xxl_weights`
//                                            (per-token scale of the adapter OUTPUT, before pad-to-512).
//   * comfy/model_base.py         L1470    — `t5xxl_weights.unsqueeze(0).unsqueeze(-1)` → `[1, St, 1]`.
// We implement exactly `out[:, i, :] *= w[i]` on the conditioner output; Qwen is untouched.
// -------------------------------------------------------------------------------------------------

/// **STRUCTURAL characterization** (mutation-sensitive): the weight scales *only* the weighted T5
/// query-token OUTPUT rows, by exactly their factor, leaving every other row bit-identical. This
/// uniquely pins `out * w` at the conditioner output: a wrong-place implementation (scaling the
/// embedding INPUT) or wrong-tower implementation (scaling the Qwen cross-attn source) smears the
/// change across ALL output rows via the 6 attention blocks, so the unweighted-rows-unchanged
/// assertion FAILS. Uses a fixed random Qwen source (no text encoder needed) so it isolates the
/// conditioner. `2.0` is a power of two ⇒ the bf16 product is exact, so equality is checked exactly.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn prompt_weighting_scales_only_weighted_t5_rows() {
    let split = split_files().expect("Anima snapshot");
    let w = Weights::from_file(dit_file(&split, Variant::Base)).unwrap();
    let cond =
        AnimaTextConditioner::from_weights(&w, "net.llm_adapter", ConditionerConfig::anima())
            .unwrap();
    let tk = AnimaTokenizers::load().unwrap();

    // Weighted prompt → T5 ids + per-token weights (the `chibi` span carries 2.0; the rest 1.0).
    let (ids, weights) = tk
        .encode_t5_weighted("1girl, (chibi:2.0), masterpiece")
        .unwrap();
    let st = ids.shape()[1] as usize;
    assert_eq!(weights.len(), st);
    assert!(
        weights.iter().any(|&x| (x - 2.0).abs() < 1e-6) && weights.contains(&1.0),
        "expected a mix of 2.0 and 1.0 weights, got {weights:?}"
    );

    // A fixed random Qwen source (same `St` length so the structure is unambiguous).
    let key = mlx_rs::random::key(7).unwrap();
    let source = mlx_rs::random::normal::<f32>(&[1, st as i32, 1024], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    // Identical inputs, weights off vs on.
    let base = cond
        .forward_weighted(&source, &ids, None, Dtype::Bfloat16)
        .unwrap();
    let wt = cond
        .forward_weighted(&source, &ids, Some(&weights), Dtype::Bfloat16)
        .unwrap();
    assert_eq!(base.shape(), &[1, 512, 1024]);
    assert_eq!(wt.shape(), &[1, 512, 1024]);

    let (bv, wv) = (to_f32(&base), to_f32(&wt));
    let d = 1024usize;
    let (mut changed_rows, mut unchanged_max, mut weighted_max) = (0usize, 0f32, 0f32);
    for (r, (base_row, wt_row)) in bv.chunks(d).zip(wv.chunks(d)).enumerate() {
        let w_r = if r < st { weights[r] } else { 1.0 };
        let mut row_diff = 0f32;
        for (c, (&b, &w)) in base_row.iter().zip(wt_row).enumerate() {
            // Exact per-row characterization: weighted == base * w_r (bf16-exact for w ∈ {1.0, 2.0}).
            let expected = b * w_r;
            assert!(
                (w - expected).abs() == 0.0,
                "row {r} col {c}: weighted {w} != base*{w_r} {expected}"
            );
            row_diff = row_diff.max((w - b).abs());
        }
        if (w_r - 1.0).abs() < 1e-6 {
            unchanged_max = unchanged_max.max(row_diff);
        } else {
            weighted_max = weighted_max.max(row_diff);
            if row_diff > 0.0 {
                changed_rows += 1;
            }
        }
    }
    println!(
        "[per-row] St={st} weighted-rows-changed={changed_rows} \
         max|Δ| weighted-rows={weighted_max:.4} unweighted+pad-rows={unchanged_max:.6}"
    );
    // The weighted rows changed; the unweighted/pad rows did NOT (mutation-sensitive assertion).
    assert!(changed_rows > 0, "the weighted T5 rows must change");
    assert!(
        weighted_max > 0.0,
        "the weighted rows must differ from base"
    );
    assert_eq!(
        unchanged_max, 0.0,
        "unweighted / pad rows must be bit-identical — a nonzero delta means the weight leaked off \
         the T5 output path (wrong place / wrong tower)"
    );
}

/// **BOTH DIRECTIONS + IMAGE** through the full pipeline: (1) T5-side `(chibi:2.0)` changes the
/// conditioner output; (2) the Qwen tower is weight-blind (its ids are identical for the weighted and
/// de-weighted prompt, so it contributes 0 — the whole change is the T5 path); and the weighting
/// visibly changes a generated image. Turbo (CFG-free, 10 steps) keeps it cheap.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (real 2B DiT denoise)"]
fn prompt_weighting_both_directions_and_image() {
    let split = split_files().expect("Anima snapshot");
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();

    let plain = "1girl, chibi, masterpiece, silver hair, detailed illustration";
    let weighted = "1girl, (chibi:2.0), masterpiece, silver hair, detailed illustration";

    let pipeline =
        AnimaPipeline::from_source(&WeightsSource::Dir(split.clone()), Variant::Turbo).unwrap();

    // Direction 1 — T5-side weighting CHANGES the conditioner output. This `max|Δ|` is only a smoke
    // bound (a wrong-but-nonzero impl clears it); the REAL correctness proof is the bf16-exact per-row
    // check in Direction 2 below and in `prompt_weighting_scales_only_weighted_t5_rows`.
    let base_c = pipeline.encode_prompt(plain).unwrap();
    let wt_c = pipeline.encode_prompt(weighted).unwrap();
    let d_t5 = maxabs_diff(&base_c, &wt_c);
    println!("[both-directions] T5-side (chibi:2.0) vs plain — conditioner max|Δ| = {d_t5:.4}");
    assert!(
        d_t5 > 0.5,
        "T5-side weighting must change the conditioner output (max|Δ| {d_t5})"
    );

    // Direction 2 — the weight VALUE is independent of the Qwen tower (the real Qwen-blindness proof;
    // the old "ids match" leg was true by construction). Encode the SAME prompt at two powers-of-two
    // weights (2.0 vs 4.0 ⇒ bf16-exact) through the production `encode_prompt`. The stripped text,
    // Qwen tokens and T5 ids are identical, so the conditioner output may differ ONLY on the weighted
    // T5 rows (scaled 2.0 vs 4.0 ⇒ enc4 = enc2 × 2.0 exactly). If the weight leaked into the Qwen
    // path, its cross-attention source would differ between the two weights and smear a change into
    // EVERY output row — so the unweighted/pad rows would NOT be bit-identical.
    let p2 = "1girl, (chibi:2.0), masterpiece, silver hair, detailed illustration";
    let p4 = "1girl, (chibi:4.0), masterpiece, silver hair, detailed illustration";
    let (ids2, w2) = pipeline
        .components()
        .tokenizers
        .encode_t5_weighted(p2)
        .unwrap();
    let (ids4, w4) = pipeline
        .components()
        .tokenizers
        .encode_t5_weighted(p4)
        .unwrap();
    assert_eq!(
        ids2.as_slice::<i32>(),
        ids4.as_slice::<i32>(),
        "same T5 tokens — only the weight VALUE differs between the two prompts"
    );
    let st = w2.len();
    let enc2 = pipeline.encode_prompt(p2).unwrap();
    let enc4 = pipeline.encode_prompt(p4).unwrap();
    let (e2, e4) = (to_f32(&enc2), to_f32(&enc4));
    let d = 1024usize;
    let (mut qwen_row_maxabs, mut weighted_rows) = (0f32, 0usize);
    for (r, (r2, r4)) in e2.chunks(d).zip(e4.chunks(d)).enumerate() {
        let (a, b) = (
            if r < st { w2[r] } else { 1.0 },
            if r < st { w4[r] } else { 1.0 },
        );
        if (a - 1.0).abs() < 1e-6 && (b - 1.0).abs() < 1e-6 {
            // Unweighted / pad row — must be BIT-IDENTICAL across the two weight values: the Qwen
            // contribution (and unweighted T5 rows) is invariant to the weight VALUE.
            let m = r2
                .iter()
                .zip(r4)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            qwen_row_maxabs = qwen_row_maxabs.max(m);
        } else {
            // Weighted row — scaled by exactly b/a (4.0 / 2.0 = 2.0, bf16-exact).
            let ratio = b / a;
            for (x, y) in r2.iter().zip(r4) {
                assert!(
                    (y - x * ratio).abs() == 0.0,
                    "row {r}: enc4 {y} != enc2*{ratio} {}",
                    x * ratio
                );
            }
            weighted_rows += 1;
        }
    }
    println!(
        "[both-directions] Qwen weight-value blindness: weight 2.0 vs 4.0 — unweighted/pad rows \
         max|Δ| = {qwen_row_maxabs:.6} (weighted rows scaled exactly ×2.0: {weighted_rows})"
    );
    assert!(weighted_rows > 0, "expected some weighted T5 rows");
    assert_eq!(
        qwen_row_maxabs, 0.0,
        "unweighted / pad rows must be identical across weight VALUES — a nonzero delta means the \
         weight value leaked into the Qwen tower (cross-attn smears it into every row)"
    );

    // Image — (chibi:2.0) must visibly change the generated image vs the unweighted prompt.
    let opts = GenOptions {
        width: 1024,
        height: 1024,
        steps: Variant::Turbo.default_steps() as usize,
        guidance: Variant::Turbo.default_guidance(),
        seed: 42,
        sampler: None,
    };
    let cancel = CancelFlag::default();
    let mut prog = |_p: Progress| {};
    let img_plain = pipeline
        .generate(plain, "", Variant::Turbo, &opts, &cancel, &mut prog)
        .unwrap();
    let img_wt = pipeline
        .generate(weighted, "", Variant::Turbo, &opts, &cancel, &mut prog)
        .unwrap();
    let p_plain = out.join("prompt_weight_turbo_plain.png");
    let p_wt = out.join("prompt_weight_turbo_chibi2.png");
    save_png(&img_plain, &p_plain);
    save_png(&img_wt, &p_wt);
    println!("wrote {}", p_plain.display());
    println!("wrote {}", p_wt.display());
    assert_coherent(&img_plain, "prompt_weight_plain");
    assert_coherent(&img_wt, "prompt_weight_chibi2");

    // Mean absolute per-pixel difference (0..255) between the two renders (same seed).
    let mad = img_plain
        .pixels
        .iter()
        .zip(&img_wt.pixels)
        .map(|(a, b)| (*a as f32 - *b as f32).abs())
        .sum::<f32>()
        / img_plain.pixels.len() as f32;
    println!("[both-directions] image mean-abs-pixel-diff (plain vs (chibi:2.0)) = {mad:.2}");
    assert!(
        mad > 1.0,
        "(chibi:2.0) must visibly change the generated image (MAD {mad})"
    );

    mlx_rs::memory::clear_cache();
}
