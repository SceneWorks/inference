//! sc-5112 acceptance gate: the candle Lens DiT must match the vendor `LensTransformer2DModel`.
//!
//! Loads the real `transformer/` weights (the cached `microsoft/Lens-Turbo` snapshot) **as f32** and
//! checks, against `scripts/dump_lens_dit_golden.py`:
//!   1. **per-block** — block 0 reproduces the reference block output given the golden's block-0
//!      inputs (`img_in_out`, `txt_in_out`, `temb`), with the Rust-built RoPE tables (tight gate);
//!   2. **full forward** — the whole 48-block DiT reproduces the reference output for the same
//!      synthetic inputs.
//!
//! f32 on both sides makes this a tight correctness gate — bf16 cross-backend accumulation over 48
//! residual blocks would obscure subtle bugs (wrong RoPE axis, transposed weight, mis-ordered
//! modulation). Heavy + machine-specific (loads the f32 DiT and needs the GPU), so it is **gated** on
//! env vars and skips cleanly when they are unset (CPU CI has neither weights nor a GPU):
//!   LENS_DIT_DIR     — the Lens-Turbo `transformer` snapshot dir (config.json + model-*.safetensors)
//!   LENS_DIT_GOLDENS — lens_dit_golden.safetensors (default: .scratch/lens-dit-goldens/…)
//! Run with the `cuda` feature (absolute goldens path — cargo test cwd is the crate dir):
//!   cargo test -p candle-gen-lens --features cuda --test dit_parity -- --nocapture

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::rope::LensRope;
use candle_gen_lens::transformer::{LensDitConfig, LensTransformer, LensTransformerBlock};

/// Cosine similarity over all elements (flattened), computed in f64 on CPU.
fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in cosine");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32)
}

/// Peak relative error `max|a-b| / max|b|`, in f64 on CPU.
fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f64;
    let mut max_b = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).abs() as f64);
        max_b = max_b.max((*y).abs() as f64);
    }
    Ok((max_diff / max_b.max(1e-12)) as f32)
}

#[test]
fn lens_dit_matches_reference() -> Result<()> {
    let dit_dir = match std::env::var("LENS_DIT_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_DIT_DIR to the Lens-Turbo transformer snapshot dir");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_DIT_GOLDENS")
        .unwrap_or_else(|_| ".scratch/lens-dit-goldens/lens_dit_golden.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_dit_golden.py)"
        );
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dit_dir)
        .expect("read transformer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {dit_dir}");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };

    let cfg = LensDitConfig::lens();

    // Goldens (f32) + the exact synthetic inputs used by the reference.
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let f32g = |k: &str| -> Result<Tensor> { g[k].to_dtype(DType::F32) };
    let grid = g["grid_fhw"].to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let (frame, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let timestep = g["timestep"].to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
    let txt_len = f32g("feat_0")?.dim(1)?;
    eprintln!("grid=({frame},{h},{w}) timestep={timestep:.5} txt_len={txt_len}");

    // --- 1. per-block: block 0 (tight gate) ---
    let block0 = LensTransformerBlock::new(&cfg, vb.pp("transformer_blocks").pp(0))?;
    let rope = LensRope::new(cfg.rope_theta, cfg.axes_dims_rope);
    let (img_cos, img_sin) = rope.img_cos_sin(frame, h, w, &device)?;
    let (txt_cos, txt_sin) = rope.txt_cos_sin(txt_len, h, w, &device)?;
    let (enc0, hid0) = block0.forward(
        &f32g("img_in_out")?,
        &f32g("txt_in_out")?,
        &f32g("temb")?,
        &img_cos,
        &img_sin,
        &txt_cos,
        &txt_sin,
        None,
    )?;
    let blk_enc_pr = peak_rel(&enc0, &f32g("block0_enc")?)?;
    let blk_hid_pr = peak_rel(&hid0, &f32g("block0_hidden")?)?;
    let blk_enc_cos = cosine(&enc0, &f32g("block0_enc")?)?;
    let blk_hid_cos = cosine(&hid0, &f32g("block0_hidden")?)?;
    eprintln!(
        "block0: enc peak_rel={blk_enc_pr:.3e} cosine={blk_enc_cos:.7} | hidden peak_rel={blk_hid_pr:.3e} cosine={blk_hid_cos:.7}"
    );

    // --- 2. full forward ---
    let transformer = LensTransformer::new(&cfg, vb)?;
    let feats: Vec<Tensor> = (0..cfg.num_text_layers)
        .map(|i| f32g(&format!("feat_{i}")))
        .collect::<Result<_>>()?;
    let out = transformer.forward(&f32g("hidden_states")?, &feats, None, timestep, frame, h, w)?;
    let out_pr = peak_rel(&out, &f32g("out")?)?;
    let out_cos = cosine(&out, &f32g("out")?)?;
    eprintln!("full forward: peak_rel={out_pr:.3e} cosine={out_cos:.7}");

    // Per-block, fed the exact reference block-0 inputs, isolates every sub-op (fused QKV, QK-norm,
    // complex RoPE, AdaLN modulation, SwiGLU GateMLP, gated residuals). The full forward then
    // accumulates the CUDA-vs-CPU f32-matmul floor over 48 residual blocks; cosine stays near 1 — a
    // real bug (wrong axis/transpose/order) would crater it.
    assert!(
        blk_enc_pr < 5e-3,
        "block0 enc peak_rel {blk_enc_pr:.3e} ≥ 5e-3"
    );
    assert!(
        blk_hid_pr < 5e-3,
        "block0 hidden peak_rel {blk_hid_pr:.3e} ≥ 5e-3"
    );
    assert!(out_cos > 0.999, "full forward cosine {out_cos:.7} ≤ 0.999");
    assert!(out_pr < 2e-2, "full forward peak_rel {out_pr:.3e} ≥ 2e-2");
    eprintln!("ALL PASS");
    Ok(())
}
