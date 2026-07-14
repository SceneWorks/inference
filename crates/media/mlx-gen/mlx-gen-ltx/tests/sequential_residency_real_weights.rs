//! sc-10976 (epic 10975 — MLX **video**-lane sequential component residency): the residency proof for
//! LTX-2.3 on real weights. LTX now stages the ~24 GB Gemma-3-12B text encoder load → `encode_av` →
//! **drop + `clear_cache()`** BEFORE the AvDiT materializes (`mlx-gen-ltx/src/model.rs`), mirroring
//! Wan's `encode_text_staged`. This test measures that the two GIANTS — the Gemma TE and the AvDiT —
//! never co-reside during a real `generate`.
//!
//! **Why this is NOT the image-lane `OffloadPolicy::Resident`↔`Sequential` A/B.** Per epic 10975 the
//! video lane stages **unconditionally** (Wan-style: always load → use → drop, no `offload_policy` /
//! fit-gate branch — video is slow enough that a cross-job warm cache is worth ~nothing against the
//! encoder's memory pressure). So there is no production "Resident" mode to flip. Instead we bound the
//! staged `generate` peak below a **co-residence estimate** = (measured TE resident peak) + (the AvDiT's
//! on-disk `transformer.safetensors` bytes). The pre-sc-10976 `load()` held BOTH giants resident for the
//! whole job, so its peak was ≥ that estimate; the staged path holds at most one giant at a time.
//!
//! **The LTX-specific picture** (measured on `SceneWorks/ltx-2.3-mlx`, q4): TE bf16 ≈ 24.6 GiB, q4 DiT ≈
//! 10.6 GiB. The TE is the LARGER phase, so the staged peak floor is the **text phase** (~TE), and the
//! win is dropping the DiT (~10 GiB) out of co-residence with the TE — a ~38 → ~28 GiB drop that lets
//! LTX fit Macs the resident build would OOM. (Quantizing the Gemma snapshot — `resolve_gemma_quant`,
//! ~6–12 GiB — is the complementary lever for the text-phase floor; orthogonal to this staging story.)
//!
//! **Output correctness** is covered by the existing parity gates (`te_parity`, `pipeline_parity`,
//! `i2v_parity`, `s0_parity`): staging changes only WHEN each component is built/freed, not the encode /
//! denoise / decode math, so those bit-exact/parity tests remain the authority. This test owns the
//! MEMORY invariant + a non-degenerate-output sanity check.
//!
//! `#[ignore]`d — needs the real snapshot. Defaults to the HF cache `SceneWorks/ltx-2.3-mlx` (model dir
//! = its `q4` subdir; Gemma = its bundled `gemma/`); override with `LTX_MODEL_DIR` / `LTX_GEMMA_DIR`.
//! Run: `cargo test -p mlx-gen-ltx --release --test sequential_residency_real_weights -- --ignored
//! --nocapture`.

use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_ltx::gemma::GemmaConfig;
use mlx_gen_ltx::{LtxConfig, LtxTextEncoder, LtxTokenizer};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::Dtype;
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> Option<PathBuf> {
    let snaps = home()
        .join(".cache/huggingface/hub")
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// The LTX model dir (split-weight snapshot) — `LTX_MODEL_DIR`, else the HF-cache
/// `SceneWorks/ltx-2.3-mlx/q4`.
fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LTX_MODEL_DIR") {
        return Some(PathBuf::from(d));
    }
    hf_snapshot("models--SceneWorks--ltx-2.3-mlx").map(|s| s.join("q4"))
}

/// The bundled Gemma-3-12B TE dir — `LTX_GEMMA_DIR`, else the snapshot's `gemma/` (the sibling of the
/// model `q4` dir).
fn gemma_dir(model: &std::path::Path) -> PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return PathBuf::from(d);
    }
    model
        .parent()
        .expect("model dir has a parent")
        .join("gemma")
}

/// A small deterministic T2V request. 256×256, 9 frames (= 1 + 8·1), audio decode skipped (`no_audio`)
/// — enough to drive the full staged text → denoise → VAE-decode path without the audio tail.
fn request() -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox trotting through a snowy forest, cinematic".into(),
        width: 256,
        height: 256,
        frames: Some(9),
        fps: Some(24),
        video_mode: Some("no_audio".into()),
        seed: Some(1234),
        ..Default::default()
    }
}

/// Measure the resident footprint of the Gemma text phase ALONE: build the AudioVideo TE exactly as
/// `load()` does (bf16; the bundled `…/gemma` is dense bf16, so `gemma_quant = None`), run a real
/// `encode_av`, and `eval` so every layer's weights are forced resident. Returns the peak bytes.
fn te_resident_peak(model: &std::path::Path, gemma: &std::path::Path) -> usize {
    let cfg = LtxConfig::from_model_dir(model).expect("LtxConfig::from_model_dir");
    let gemma_w = Weights::from_dir(gemma).expect("gemma weights");
    let connector_w =
        Weights::from_file(model.join("connector.safetensors")).expect("connector weights");
    reset_peak_memory();
    let te = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        None, // the bundled `gemma/` is dense bf16 (no `config.json` quantization block)
        &cfg,
        Dtype::Bfloat16,
    )
    .expect("build TE");
    let tok = LtxTokenizer::from_dir(gemma).expect("tokenizer");
    // Pad to the production prompt length (`MAX_PROMPT_TOKENS` = 1024) so this baseline's Gemma encode
    // footprint matches the real text phase — and clears the connector's 128-register minimum.
    let (ids, mask) = tok
        .encode("a red fox trotting through a snowy forest", 1024)
        .expect("tokenize");
    let (video_ctx, audio_ctx) = te.encode_av(&ids, &mask).expect("encode_av");
    mlx_rs::transforms::eval([&video_ctx, &audio_ctx]).expect("eval");
    let peak = get_peak_memory();
    drop(te);
    clear_cache();
    peak
}

/// Run the real staged `generate`, returning the video frames + the process peak unified memory.
fn staged_generate(model: &std::path::Path, gemma: &std::path::Path) -> (Vec<Image>, usize) {
    let spec = LoadSpec {
        text_encoder: Some(WeightsSource::Dir(gemma.to_path_buf())),
        ..LoadSpec::new(WeightsSource::Dir(model.to_path_buf()))
    };
    let m = mlx_gen_ltx::provider_registry()
        .expect("build explicit LTX provider registry")
        .load("ltx_2_3", &spec)
        .expect("load ltx_2_3");
    reset_peak_memory();
    let out = m.generate(&request(), &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let frames = match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video, got {other:?}"),
    };
    drop(m);
    clear_cache();
    (frames, peak)
}

#[test]
#[ignore = "needs SceneWorks/ltx-2.3-mlx (LTX_MODEL_DIR/LTX_GEMMA_DIR or the HF cache); ~25 GB+ RAM"]
fn ltx_staged_peak_stays_below_te_plus_dit_coresidence() {
    let Some(model) = model_dir() else {
        eprintln!("skip: no LTX_MODEL_DIR and no SceneWorks/ltx-2.3-mlx in the HF cache");
        return;
    };
    let gemma = gemma_dir(&model);
    if !model.join("transformer.safetensors").exists() || !gemma.exists() {
        eprintln!(
            "skip: missing model/{{transformer.safetensors}} or gemma dir\n  model={}\n  gemma={}",
            model.display(),
            gemma.display()
        );
        return;
    }

    // The AvDiT's resident weight proxy: the on-disk `transformer.safetensors` bytes (follow symlink).
    let dit_bytes = std::fs::metadata(model.join("transformer.safetensors"))
        .expect("stat transformer.safetensors")
        .len() as usize;

    // Staged production path first, then the TE-alone resident peak (each brackets its own
    // reset/clear so neither inflates the other).
    let (frames, staged_peak) = staged_generate(&model, &gemma);
    let te_peak = te_resident_peak(&model, &gemma);

    // The pre-sc-10976 `load()` held BOTH giants resident for the whole job, so its peak was AT LEAST
    // this (it also carried the small components + activations, which this estimate omits — i.e. the
    // bound is conservative).
    let coresident_estimate = te_peak + dit_bytes;
    let saved = coresident_estimate.saturating_sub(staged_peak);

    println!(
        "\nLTX sequential residency ({}×{}, {} frames):\n  \
         TE resident peak (Gemma text phase) = {:.2} GiB\n  \
         AvDiT weights (transformer.safetensors) = {:.2} GiB\n  \
         co-resident estimate (TE + DiT, pre-sc-10976 floor) = {:.2} GiB\n  \
         staged generate peak = {:.2} GiB\n  \
         saved vs co-residence ≈ {:.2} GiB ({:.1}%)",
        request().width,
        request().height,
        request().frames.unwrap(),
        te_peak as f64 / GIB,
        dit_bytes as f64 / GIB,
        coresident_estimate as f64 / GIB,
        staged_peak as f64 / GIB,
        saved as f64 / GIB,
        100.0 * saved as f64 / coresident_estimate as f64,
    );

    // (1) Non-degenerate output: the right number of frames at the requested size, and frame 0 is not a
    // flat single-color buffer (a smoke test that the staged denoise + decode actually produced pixels).
    assert_eq!(frames.len(), 9, "expected 9 video frames");
    let f0 = &frames[0];
    assert_eq!(
        f0.pixels.len(),
        (256 * 256 * 3) as usize,
        "frame 0 is {}×{} — wrong pixel count",
        f0.width,
        f0.height
    );
    assert!(
        f0.pixels.iter().any(|&p| p != f0.pixels[0]),
        "frame 0 is a flat single-color buffer — the staged denoise/decode produced no image"
    );

    // (2) The residency invariant: the staged generate NEVER reaches TE+DiT co-residence. If staging
    // regressed (TE held through the denoise), staged_peak would be ≈ TE + DiT + smalls > this estimate.
    assert!(
        staged_peak < coresident_estimate,
        "staged peak {:.2} GiB was NOT below the TE+DiT co-residence estimate {:.2} GiB — the Gemma \
         drop before the DiT did not bound peak (staging regressed?)",
        staged_peak as f64 / GIB,
        coresident_estimate as f64 / GIB,
    );
    // (3) Tripwire: the DiT really left co-residence — the win should be multiple GiB (q4 DiT ≈ 10 GiB),
    // well above measurement noise. A tiny/zero saving means the DiT stayed resident alongside the TE.
    assert!(
        saved as f64 / GIB > 2.0,
        "saved only {:.2} GiB — expected several GiB (≈ the DiT dropped out of co-residence); staging \
         may not be freeing the AvDiT / TE",
        saved as f64 / GIB,
    );
}
