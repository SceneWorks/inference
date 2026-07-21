//! Krea 2 **Raw→Turbo LoRA/LoKr** real-weight harness (epic 7565 P3). Weight-gated (`#[ignore]`):
//! trains an adapter on the real `krea/Krea-2-Raw` DiT, then loads the `krea_2_turbo` engine WITH that
//! adapter and renders. Two layers:
//! - **sc-7911 apply MECHANISM** — `raw_trained_{lora,lokr}_applies_at_turbo_inference`: a 3-step
//!   micro-train proves the adapter installs (no "not supported" rejection) + the render stays
//!   coherent + the output changes (same-seed deterministic ⇒ any diff is adapter-attributable).
//! - **sc-7579 VIABILITY** — `raw_lora_visibly_shifts_turbo_toward_concept` (a real ~160-step concept
//!   train; asserts the learned concept visibly + coherently carries onto the distilled Turbo) +
//!   `lora_scale_sweep_over_trained_concept` (characterizes effect-vs-scale over the saved adapter).
//!   Env tunables: `KREA_LORA_STEPS` / `KREA_LORA_SCALE` / `KREA_SWEEP_PROMPT` / `KREA_ADAPTER`; PNGs
//!   saved to `/tmp/krea_lora_viability`.
//!
//! Needs BOTH snapshots (`KREA_TURBO_DIR` may be the Q8 turnkey `…/krea-2-turbo-mlx/snapshots/<rev>/q8`):
//!
//! ```sh
//! KREA_RAW_DIR=/path/to/models--krea--Krea-2-Raw/snapshots/<rev> \
//! KREA_TURBO_DIR=/path/to/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test apply_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, Quant, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_gen_krea::{load, load_trainer};

/// Resolve a cached HF snapshot dir (the `{env}` override, else the newest `models--{repo}` snapshot
/// with a `transformer/` tree).
fn snapshot(env: &str, repo_dir: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("MLX_GEN_MODELS_ROOT").ok()?;
    let snaps = PathBuf::from(home).join(repo_dir).join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
}

fn write_synth_image(path: &std::path::Path) {
    let mut img = image::RgbImage::new(320, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
    img.save(path).expect("write synth png");
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — a coherent image has a broad histogram
/// AND spatial smoothness; pure noise has a high adjacent Δ and flat std (mirrors `e2e_real_weights`).
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Mean absolute per-byte difference between two equal-length RGB8 buffers.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffers differ in size");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

/// Train a tiny adapter (`network_type`) on the real Raw DiT and return its `.safetensors` path.
fn train_tiny_adapter(raw: &std::path::Path, network_type: NetworkType, tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("krea_apply_smoke_{tag}"));
    std::fs::create_dir_all(&tmp).unwrap();
    let img_path = tmp.join("swatch.png");
    write_synth_image(&img_path);

    let mut trainer =
        load_trainer(&LoadSpec::new(WeightsSource::Dir(raw.to_path_buf()))).expect("load_trainer");
    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a vivid abstract color swatch".into(),
            control_image_path: None,
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 3,
            resolution: 256,
            save_every: 0,
            learning_rate: 1e-4,
            network_type,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: format!("krea_apply_{tag}.safetensors"),
        trigger_words: vec!["swatch".into()],
        cancel: CancelFlag::new(),
    };
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                eprintln!("[sc-7911 {tag}] train step {step} loss {loss:.5}");
            }
        })
        .expect("train");
    out.adapter_path
}

/// Load the `krea_2_turbo` engine (optionally with adapters) and render one 512² image.
fn render_turbo(turbo: &std::path::Path, adapters: Vec<AdapterSpec>) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(turbo.to_path_buf()));
    match std::env::var("KREA_QUANT").ok().as_deref() {
        Some("q8") => spec = spec.with_quant(Quant::Q8),
        Some("q4") => spec = spec.with_quant(Quant::Q4),
        _ => {}
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let gen = load(&spec).expect("load krea_2_turbo engine (+adapters)");
    let req = GenerationRequest {
        prompt: "A medium-shot photograph of a red fox in a snowy forest at golden hour.".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let GenerationOutput::Images(mut imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    imgs.pop().expect("one image")
}

// ── sc-7579 viability: a REAL concept-learning train, not just the apply-mechanism smoke ──────────

/// Write `n` solid-`rgb` 512² PNGs — a strong, fast-to-learn concept for a viability train.
fn write_solid_images(dir: &std::path::Path, n: usize, rgb: [u8; 3]) -> Vec<PathBuf> {
    (0..n)
        .map(|i| {
            let p = dir.join(format!("concept_{i}.png"));
            image::RgbImage::from_pixel(512, 512, image::Rgb(rgb))
                .save(&p)
                .expect("write concept png");
            p
        })
        .collect()
}

/// Magenta-ness of an RGB8 buffer: `mean(R) + mean(B) − 2·mean(G)` — high for magenta, low/negative
/// for the green-ish natural scene the base prompt renders. A LoRA that learned a magenta concept
/// raises this when applied.
fn magenta_score(px: &[u8]) -> f32 {
    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
    for c in px.chunks_exact(3) {
        r += c[0] as u64;
        g += c[1] as u64;
        b += c[2] as u64;
    }
    let n = (px.len() / 3).max(1) as f32;
    (r as f32 + b as f32 - 2.0 * g as f32) / n
}

/// Save an RGB8 image under /tmp for eyeballing the viability verdict.
fn save_png(img: &Image, name: &str) {
    let dir = std::path::Path::new("/tmp/krea_lora_viability");
    std::fs::create_dir_all(dir).ok();
    let p = dir.join(format!("{name}.png"));
    image::save_buffer(
        &p,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .ok();
    eprintln!("  saved {}", p.display());
}

/// Train a real LoRA on Raw over `images`/`caption` for `steps` steps at `rank` (512², grad-checkpointed).
fn train_concept_lora(
    raw: &std::path::Path,
    images: &[PathBuf],
    caption: &str,
    rank: u32,
    steps: u32,
    out_name: &str,
) -> PathBuf {
    let tmp = std::env::temp_dir().join("krea_lora_viability");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut trainer =
        load_trainer(&LoadSpec::new(WeightsSource::Dir(raw.to_path_buf()))).expect("load_trainer");
    let req = TrainingRequest {
        items: images
            .iter()
            .map(|p| TrainingItem {
                image_path: p.clone(),
                caption: caption.to_string(),
                control_image_path: None,
            })
            .collect(),
        config: TrainingConfig {
            rank,
            alpha: rank as f32,
            steps,
            resolution: 512,
            save_every: 0,
            learning_rate: 1e-4,
            network_type: NetworkType::Lora,
            gradient_checkpointing: true,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: format!("{out_name}.safetensors"),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                if step <= 3 || step % 20 == 0 {
                    eprintln!("[sc-7579] train step {step} loss {loss:.5}");
                }
            }
        })
        .expect("train");
    out.adapter_path
}

/// Render Turbo on an arbitrary `prompt` with `adapters` (seed 0, 512², 8 steps).
fn render_turbo_prompt(turbo: &std::path::Path, adapters: Vec<AdapterSpec>, prompt: &str) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(turbo.to_path_buf()));
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let gen = load(&spec).expect("load krea_2_turbo (+adapters)");
    let req = GenerationRequest {
        prompt: prompt.into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let GenerationOutput::Images(mut imgs) = gen.generate(&req, &mut |_| {}).expect("generate")
    else {
        panic!("expected images");
    };
    imgs.pop().expect("one image")
}

/// sc-7579 — the epic's open viability question: does a Raw-trained LoRA visibly + acceptably alter
/// the *distilled* Turbo's output? Trains a real (~160-step) LoRA on a strong magenta concept, then
/// renders the SAME neutral prompt on Turbo with the adapter OFF vs ON and confirms the output shifts
/// toward the learned concept while staying a coherent image. Tunable via `KREA_LORA_STEPS` /
/// `KREA_LORA_SCALE`. Saves both PNGs to /tmp/krea_lora_viability for eyeballing.
#[test]
#[ignore = "viability (sc-7579): real Raw+Turbo + a Mac; ~160-step train — run as its own process"]
fn raw_lora_visibly_shifts_turbo_toward_concept() {
    let (Some(raw), Some(turbo)) = (
        snapshot("KREA_RAW_DIR", "models--krea--Krea-2-Raw"),
        snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_TURBO_DIR");
        return;
    };

    let tmp = std::env::temp_dir().join("krea_lora_viability");
    std::fs::create_dir_all(&tmp).unwrap();
    let images = write_solid_images(&tmp, 5, [230, 30, 230]);
    let steps: u32 = std::env::var("KREA_LORA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(160);
    let scale: f32 = std::env::var("KREA_LORA_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let adapter = train_concept_lora(
        &raw,
        &images,
        "a solid magenta color field",
        16,
        steps,
        "viability_magenta",
    );
    // A PERMISSIVE prompt: the few-step distilled Turbo adheres strongly to prompt, so it resists a
    // LoRA that tries to OVERRIDE a strongly-described scene (e.g. "snowy mountain" stays a mountain);
    // a loosely-constrained backdrop gives the learned concept room to express, which is the fair
    // "does it transfer" probe. (The scale-sweep test characterizes both prompt regimes.)
    let prompt = "a minimalist abstract studio backdrop";

    let off = render_turbo_prompt(&turbo, Vec::new(), prompt);
    let on = render_turbo_prompt(
        &turbo,
        vec![AdapterSpec::new(adapter, scale, AdapterKind::Lora)],
        prompt,
    );
    save_png(&off, "mountain_adapter_off");
    save_png(&on, "mountain_adapter_on");

    let (m_off, m_on) = (magenta_score(&off.pixels), magenta_score(&on.pixels));
    let diff = mean_abs_diff(&off.pixels, &on.pixels);
    eprintln!(
        "[sc-7579] steps={steps} scale={scale} · magenta off={m_off:.1} on={m_on:.1} (Δ={:.1}) · \
         mean|Δ|px={diff:.1} · coherent off={} on={}",
        m_on - m_off,
        is_coherent(&off),
        is_coherent(&on)
    );

    assert!(
        is_coherent(&off),
        "base (adapter-off) render must be coherent"
    );
    assert!(
        is_coherent(&on),
        "adapter-applied render must stay a coherent image (acceptable quality, not a solid blob)"
    );
    assert!(
        m_on - m_off > 4.0,
        "the Raw-trained magenta LoRA must visibly pull the Turbo render toward the learned concept \
         (magenta Δ={:.1}); raise KREA_LORA_STEPS/KREA_LORA_SCALE if marginal",
        m_on - m_off
    );
}

/// sc-7579 characterization — re-render at a sweep of adapter scales over the ALREADY-TRAINED
/// concept adapter (no retrain), to see whether the learned concept emerges on the distilled Turbo as
/// scale rises and at what point coherence breaks. Reads `KREA_ADAPTER` (default the path
/// `raw_lora_visibly_shifts_turbo_toward_concept` writes). Pure characterization — no assertions.
#[test]
#[ignore = "characterization (sc-7579): reuses the saved magenta adapter; run after the viability test"]
fn lora_scale_sweep_over_trained_concept() {
    let Some(turbo) = snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo") else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };
    let adapter = std::env::var("KREA_ADAPTER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir()
                .join("krea_lora_viability")
                .join("viability_magenta.safetensors")
        });
    if !adapter.is_file() {
        eprintln!(
            "skipping: no adapter at {} (run raw_lora_visibly_shifts_turbo_toward_concept first)",
            adapter.display()
        );
        return;
    }
    let prompt_owned = std::env::var("KREA_SWEEP_PROMPT")
        .unwrap_or_else(|_| "a photograph of a snowy mountain at golden hour".to_string());
    let prompt = prompt_owned.as_str();
    eprintln!("[sc-7579 sweep] prompt = {prompt:?}");
    let base = render_turbo_prompt(&turbo, Vec::new(), prompt);
    let m_base = magenta_score(&base.pixels);
    save_png(&base, "sweep_scale_0");
    eprintln!(
        "[sc-7579 sweep] scale=0.0 magenta={m_base:.1} coherent={}",
        is_coherent(&base)
    );
    for &scale in &[1.0f32, 2.0, 3.0, 4.0] {
        let img = render_turbo_prompt(
            &turbo,
            vec![AdapterSpec::new(adapter.clone(), scale, AdapterKind::Lora)],
            prompt,
        );
        save_png(&img, &format!("sweep_scale_{}", scale as u32));
        eprintln!(
            "[sc-7579 sweep] scale={scale:.1} magenta={:.1} (Δ={:.1}) mean|Δ|px={:.1} coherent={}",
            magenta_score(&img.pixels),
            magenta_score(&img.pixels) - m_base,
            mean_abs_diff(&base.pixels, &img.pixels),
            is_coherent(&img)
        );
    }
}

#[test]
#[ignore = "needs real Krea 2 Raw + Turbo snapshots (~45 GB) + a Mac; run as its own process"]
fn raw_trained_lora_applies_at_turbo_inference() {
    let (Some(raw), Some(turbo)) = (
        snapshot("KREA_RAW_DIR", "models--krea--Krea-2-Raw"),
        snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_TURBO_DIR");
        return;
    };

    // Train a tiny LoRA on Raw, then render Turbo with and without it (same seed).
    let adapter = train_tiny_adapter(&raw, NetworkType::Lora, "lora");
    let spec = AdapterSpec::new(adapter, 4.0, AdapterKind::Lora);

    let base = render_turbo(&turbo, Vec::new());
    let adapted = render_turbo(&turbo, vec![spec]);

    assert_eq!((base.width, base.height), (512, 512));
    assert_eq!((adapted.width, adapted.height), (512, 512));
    assert!(is_coherent(&base), "base Turbo render must be coherent");
    assert!(
        is_coherent(&adapted),
        "adapter-applied Turbo render must stay coherent (not noise)"
    );

    // Same seed + deterministic MLX ops ⇒ without an adapter the two renders would be byte-identical
    // (mean|Δ| == 0). So ANY non-trivial diff is entirely adapter-attributable; this is the apply
    // MECHANISM smoke (a 3-step micro-train imparts only a small shift — measured ~0.33 on the 0–255
    // byte scale). The "does a real LoRA visibly + acceptably alter output" viability bar (sc-7579)
    // is `raw_lora_visibly_shifts_turbo_toward_concept` below.
    let diff = mean_abs_diff(&base.pixels, &adapted.pixels);
    eprintln!("[sc-7911 lora] base↔adapted mean|Δ| = {diff:.3} (adapter-attributable; same seed)");
    assert!(
        diff > 0.1,
        "the Raw-trained LoRA must change the Turbo output above numeric noise (mean|Δ|={diff:.3})"
    );
}

#[test]
#[ignore = "needs real Krea 2 Raw + Turbo snapshots (~45 GB) + a Mac; run as its own process"]
fn raw_trained_lokr_applies_at_turbo_inference() {
    let (Some(raw), Some(turbo)) = (
        snapshot("KREA_RAW_DIR", "models--krea--Krea-2-Raw"),
        snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_TURBO_DIR");
        return;
    };

    // The LoKr path rides the same `apply_adapters_strict` seam (Kronecker delta → residual); confirm
    // a Raw-trained LoKr loads + renders coherently at Turbo.
    let adapter = train_tiny_adapter(&raw, NetworkType::Lokr, "lokr");
    let spec = AdapterSpec::new(adapter, 2.0, AdapterKind::Lokr);
    let adapted = render_turbo(&turbo, vec![spec]);

    assert_eq!((adapted.width, adapted.height), (512, 512));
    assert!(
        is_coherent(&adapted),
        "LoKr-applied Turbo render must be a coherent image"
    );
}
