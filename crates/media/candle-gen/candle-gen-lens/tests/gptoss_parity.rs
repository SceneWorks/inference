//! sc-5108 acceptance gate: the candle gpt-oss encoder must match the HF torch reference.
//!
//! Compares `GptOssTextEncoder`'s per-layer hidden states against goldens dumped from the HF
//! `microsoft/Lens` text_encoder (transformers 5.8, MXFP4 → bf16 dequant) by
//! `scripts/dump_gptoss_goldens.py`. Both sides dequantize the experts to bf16 and run bf16, so this
//! is an apples-to-apples comparison of the forward, not of the quantization.
//!
//! Heavy + machine-specific (loads ~13 GB and needs the GPU for a 20B forward), so it is **gated** on
//! env vars and skips cleanly when they are unset (CPU CI has neither weights nor a GPU):
//!   LENS_TEXT_ENCODER_DIR — the Lens `text_encoder` snapshot dir (config.json + model-*.safetensors)
//!   LENS_GOLDENS          — gptoss_goldens.safetensors (default: .scratch/gptoss-goldens/…)
//! Run with the `cuda` feature:
//!   cargo test -p candle-gen-lens --features cuda --test gptoss_parity -- --nocapture

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::text_encoder::{Config, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};

/// Cosine similarity over all elements (flattened), computed in f32 on CPU.
fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in cosine");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32)
}

/// Relative L2 error ||a-b|| / ||b||, in f32 on CPU.
fn rel_l2(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((*x - *y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    Ok((num.sqrt() / (den.sqrt() + 1e-12)) as f32)
}

#[test]
fn gptoss_encoder_matches_torch_reference() -> Result<()> {
    let te_dir = match std::env::var("LENS_TEXT_ENCODER_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_TEXT_ENCODER_DIR to the Lens text_encoder snapshot dir");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_GOLDENS")
        .unwrap_or_else(|_| ".scratch/gptoss-goldens/gptoss_goldens.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!("SKIP: goldens not found at {goldens_path} (run scripts/dump_gptoss_goldens.py)");
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    // Load the encoder weights (bf16; experts MXFP4 dequantized to bf16 inside the module).
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&te_dir)
        .expect("read text_encoder dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {te_dir}");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::BF16, &device)? };
    let encoder = GptOssTextEncoder::new(&Config::gpt_oss_20b(), vb)?;

    // Goldens (f32) + the exact input ids used by the reference.
    let goldens = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let input_ids = goldens["input_ids"].to_dtype(DType::U32)?;
    let seq = input_ids.dim(0)?;
    let input_ids = input_ids.reshape((1, seq))?;

    let out = encoder.forward(&input_ids)?;

    // Compare every dumped layer index (all <= 23 → pre-norm residual outputs in both) and the final
    // normed state. HF hidden_states[i] == our hidden_states[i] for i in 0..=23.
    let mut worst_cos = 1f32;
    for i in [0usize, 1, 5, 11, 12, 17, 23] {
        let key = format!("hidden_{i:02}");
        let Some(golden) = goldens.get(&key) else {
            continue;
        };
        let mine = out.hidden_states[i].squeeze(0)?; // [seq, hidden]
        let c = cosine(&mine, golden)?;
        let r = rel_l2(&mine, golden)?;
        eprintln!("hidden[{i:>2}]: cosine={c:.6}  rel_l2={r:.4}");
        worst_cos = worst_cos.min(c);
        assert!(c > 0.99, "hidden[{i}] cosine {c:.6} too low (port bug?)");
        assert!(r < 0.06, "hidden[{i}] rel_l2 {r:.4} too high (port bug?)");
    }

    let last = out.last_hidden_state.squeeze(0)?;
    let golden_last = &goldens["last_hidden_state"];
    let c = cosine(&last, golden_last)?;
    let r = rel_l2(&last, golden_last)?;
    eprintln!("last_hidden_state: cosine={c:.6}  rel_l2={r:.4}");
    assert!(c > 0.99, "last_hidden_state cosine {c:.6} too low");
    assert!(r < 0.06, "last_hidden_state rel_l2 {r:.4} too high");

    // sc-5110: multi-layer capture — the OUTPUT of decoder layers [5,11,17,23] (the LensGptOssEncoder
    // feature path) vs the reference's raw layer-output captures. capture[k] == hidden_states[s+1].
    let caps = encoder.capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
    for (cap, &layer) in caps.iter().zip(DEFAULT_SELECTED_LAYERS.iter()) {
        let Some(golden) = goldens.get(&format!("cap_{layer:02}")) else {
            continue;
        };
        let mine = cap.squeeze(0)?; // [seq, hidden]
        let c = cosine(&mine, golden)?;
        let r = rel_l2(&mine, golden)?;
        eprintln!("capture[L{layer:>2}]: cosine={c:.6}  rel_l2={r:.4}");
        assert!(
            c > 0.99,
            "capture L{layer} cosine {c:.6} too low (capture-index bug?)"
        );
        assert!(r < 0.06, "capture L{layer} rel_l2 {r:.4} too high");
        worst_cos = worst_cos.min(c);
    }

    eprintln!("PASS — worst layer cosine {worst_cos:.6}");
    Ok(())
}
