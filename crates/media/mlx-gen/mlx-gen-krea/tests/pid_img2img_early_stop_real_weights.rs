//! **sc-10121 (epic 8588 × PiD epic 7840) — PiD `from_ldm` early-stop × img2img on Krea 2 Turbo.**
//! Weight-gated (`#[ignore]`). The integrated go/no-go for the ONE genuinely-crossed feature: a PiD
//! `from_ldm` early-stop capture (σ>0 partial-denoise → PiD super-res) applied on the img2img surface
//! (reference latent-init, denoise a SLICED schedule). The risk the story flags is a σ desync — the
//! img2img denoise runs `sigmas[start..]`, so the capture index and the decoder's degrade σ must be
//! resolved against that sliced window (start-aware, via `flow_capture_for_request`'s `start_step`), or
//! the PiD decoder is bound to a σ the latent never reached → garbage. This test proves it does NOT
//! desync: through the real [`Generator`], img2img + `use_pid` + `pid_capture_sigma` produces a coherent
//! 4× image (a desynced decoder would render noise/structure-collapse, which `is_coherent` rejects).
//!
//! Runs three generations off ONE reference image: (1) img2img + clean σ=0 PiD (the productionized A1
//! path), (2) img2img + from_ldm early-stop PiD (the sc-10121 combo), (3) — implicit — the crossover is
//! eyeballed over the saved PNGs. Asserts each is 4× native and coherent.
//!
//! ```sh
//! KREA_TURBO_DIR=/path/to/models--SceneWorks--krea-2-turbo-mlx/snapshots/<rev>/q8 \
//!   PID_QWEN_SAFETENSORS=tools/golden/pid/qwenimage_2kto4k.safetensors \
//!   cargo test -p mlx-gen-krea --release --test pid_img2img_early_stop_real_weights \
//!     -- --ignored --nocapture
//! ```
//! (With no env, auto-resolves the newest cached `SceneWorks/krea-2-turbo-mlx` q8 turnkey, the golden
//! PiD checkpoint, and a cached `gemma-2-2b-it`.) PNGs land in `tools/golden/pid`.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::img2img::init_time_step;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_krea::pipeline::turbo_schedule;
use mlx_gen_krea::{load, TURBO_STEPS};
use mlx_gen_pid::flow_capture_for_request;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn krea_snapshot() -> PathBuf {
    env_path("KREA_TURBO_DIR").unwrap_or_else(|| panic!("set KREA_TURBO_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"))
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_QWEN_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        ))
    })
}

fn gemma_dir() -> PathBuf {
    env_path("PID_GEMMA_DIR").unwrap_or_else(|| panic!("set PID_GEMMA_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"))
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — coherent = broad histogram + spatial
/// smoothness; a σ-desynced PiD decode collapses to noise (high adjacent Δ) or a flat field (narrow
/// std). Mirrors `img2img_spike_real_weights::image_stats`.
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
        if i % stride >= 3 {
            adj_sum += (v as f64 - px[i - 3] as f64).abs();
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
    // A real rendered image has a broad histogram (>40 levels), meaningful contrast (std > 12), and is
    // locally smooth (mean adjacent |Δ| < 60 — pure noise sits far above that). A σ-desync fails ≥1.
    distinct > 40 && std > 12.0 && adj < 60.0
}

fn only_image(out: GenerationOutput) -> Image {
    match out {
        GenerationOutput::Images(v) => v.into_iter().next().expect("one image"),
        _ => panic!("expected images"),
    }
}

fn save_png(img: &Image, path: &str) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs the Krea Turbo snapshot + converted PiD checkpoint + gemma-2-2b-it"]
fn krea_turbo_img2img_pid_from_ldm_early_stop() {
    let size: u32 = std::env::var("KREA_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    // A mid-window strength: enough of the reference survives to blend, enough steps remain for the
    // capture ceiling to land after `start` (else `flow_capture_for_request` drops the capture as
    // "no benefit"). σ ceiling between two schedule steps of the SLICED window.
    let strength: f32 = std::env::var("KREA_IMG2IMG_STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4);
    // Default ceiling MUST land strictly inside the img2img-sliced window `sigmas[start..]` or the
    // capture collapses to the clean σ=0 tail and this test false-greens by running the clean path
    // twice (sc-11869). For strength 0.4 (start=3) on the 8-step Turbo schedule
    // `[1.0, .957, .905, .840, .760, .655, .513, .311, 0.0]`, `0.55` resolves to `sigmas[6]=0.513`
    // (`keep=7 > start+1`) — an ACTIVE capture. The active-capture assert below enforces this for
    // any override too. The OLD default `0.3` sat below every positive σ → inactive.
    let capture_sigma: f32 = std::env::var("KREA_PID_CAPTURE_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.55);

    let spec = LoadSpec::new(WeightsSource::Dir(krea_snapshot())).with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );
    eprintln!("loading Krea 2 Turbo (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = load(&spec).expect("load Krea + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    // --- Reference image R (plain t2i, native VAE) ---
    let ref_req = GenerationRequest {
        prompt:
            "a photograph of a mountain landscape with a still lake and pine trees, clear blue \
                 sky, midday"
                .into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        ..Default::default()
    };
    let reference = only_image(
        model
            .generate(&ref_req, &mut |_| {})
            .expect("reference t2i"),
    );
    assert_eq!(reference.width, size, "reference is native-res");
    assert!(is_coherent(&reference), "reference itself must be coherent");

    // The restyle prompt + reference conditioning shared by both PiD runs below.
    let restyle = "a mountain landscape at sunset, warm orange and violet sky, glowing autumn \
                   foliage";
    let base_img2img = GenerationRequest {
        prompt: restyle.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(11),
        use_pid: true,
        conditioning: vec![Conditioning::Reference {
            image: reference.clone(),
            strength: Some(strength),
        }],
        ..Default::default()
    };

    // --- (1) img2img + clean σ=0 PiD (the A1 productionized path) ---
    let t = Instant::now();
    let clean = only_image(
        model
            .generate(&base_img2img, &mut |_| {})
            .expect("img2img + clean PiD"),
    );
    let clean_dt = t.elapsed().as_secs_f32();
    let (cs, cd, ca) = image_stats(&clean.pixels, clean.width);
    eprintln!(
        "img2img clean σ=0 PiD: {}x{} in {clean_dt:.2}s  std {cs:.1} distinct {cd} adj {ca:.1}",
        clean.width, clean.height
    );
    assert_eq!(clean.width, size * 4, "clean PiD width == 4× native");
    assert_eq!(clean.height, size * 4, "clean PiD height == 4× native");
    assert!(is_coherent(&clean), "img2img clean-PiD output incoherent");

    // --- (2) img2img + from_ldm early-stop PiD (the sc-10121 combo) ---
    let early_req = GenerationRequest {
        pid_capture_sigma: Some(capture_sigma),
        ..base_img2img.clone()
    };

    // sc-11869: PROVE the from_ldm capture is ACTIVE for these params BEFORE generating — otherwise
    // the run below would silently execute the clean σ=0 tail (identical to `clean`) and the
    // coherence/4× asserts would pass without ever exercising the crossed path. Mirror the Turbo
    // img2img resolution in `Krea::generate_impl`: `start = init_time_step(steps, strength)`, then
    // resolve the capture against the FULL schedule with that `start`. An active capture is
    // `keep < len` (a ceiling stopping at/before `start` collapses to `keep == len`, the clean tail).
    let sched = turbo_schedule(TURBO_STEPS, None);
    let start = init_time_step(TURBO_STEPS, Some(strength)).min(sched.len() - 1);
    let (resolved_sigma, keep) = flow_capture_for_request(&early_req, &sched, start);
    assert!(
        keep < sched.len() && keep > start + 1,
        "sc-11869: from_ldm capture INACTIVE at strength={strength}, σ_ceiling={capture_sigma} \
         (start={start}, keep={keep}, len={}) — the gate would false-green on the clean σ=0 tail. \
         Set KREA_PID_CAPTURE_SIGMA above the smallest positive σ of sigmas[start..].",
        sched.len(),
    );
    eprintln!(
        "sc-11869 active-capture check OK: start={start} keep={keep} resolved_sigma={resolved_sigma:.4} \
         (schedule len {})",
        sched.len(),
    );

    let t = Instant::now();
    let early = only_image(
        model
            .generate(&early_req, &mut |_| {})
            .expect("img2img + from_ldm early-stop PiD (sc-10121)"),
    );
    let early_dt = t.elapsed().as_secs_f32();
    let (es, ed, ea) = image_stats(&early.pixels, early.width);
    eprintln!(
        "img2img from_ldm σ≤{capture_sigma} PiD: {}x{} in {early_dt:.2}s  std {es:.1} distinct {ed} \
         adj {ea:.1}  ({:.0}% wall-clock vs clean)",
        early.width,
        early.height,
        100.0 * (1.0 - early_dt / clean_dt.max(1e-3)),
    );
    assert_eq!(early.width, size * 4, "early-stop PiD width == 4× native");
    assert_eq!(early.height, size * 4, "early-stop PiD height == 4× native");
    // THE σ-desync gate: a decoder bound to a σ the sliced latent never reached renders garbage. A
    // coherent 4× image is the proof the capture resolved against the img2img-sliced schedule.
    assert!(
        is_coherent(&early),
        "img2img + from_ldm early-stop PiD is incoherent — likely a σ desync (sc-10121)"
    );
    // sc-11869: the active capture MUST change the output vs the clean σ=0 decode. Byte-identical
    // pixels mean the capture silently collapsed to the clean tail (the false-green this guards) —
    // coherence alone can't tell "the crossed path rendered a good image" from "the clean path ran
    // twice". (The pre-capture assert above already proves `keep < len`; this is the on-image check.)
    assert_ne!(
        early.pixels, clean.pixels,
        "sc-11869: early-stop output is byte-identical to the clean σ=0 decode — the from_ldm \
         capture did not fire, so the crossed path was never exercised"
    );

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(dir);
    save_png(
        &reference,
        &format!("{dir}/krea_img2img_ref_{}.png", reference.width),
    );
    save_png(
        &clean,
        &format!("{dir}/krea_img2img_pid_clean_{}.png", clean.width),
    );
    save_png(
        &early,
        &format!("{dir}/krea_img2img_pid_earlystop_{}.png", early.width),
    );
    eprintln!("wrote reference + clean + early-stop PNGs to {dir}");
}
