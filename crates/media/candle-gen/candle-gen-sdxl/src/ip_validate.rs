//! SDXL IP-Adapter-Plus real-weight GPU validation (sc-5488, epic 5480) — an env-driven, `#[ignore]`d
//! integration test that drives the REAL [`IpAdapterSdxl`] stack on the deployed hardware
//! (RealVisXL/SDXL diffusers + `ip-adapter-plus_sdxl_vit-h` + the CLIP ViT-H image encoder + a
//! reference image). The analog of the InstantID Phase-5 harness.
//!
//! **Quantitative gate (no extra deps).** IP-Adapter conditions on CLIP image features, so the natural
//! metric is the CLIP-feature cosine between the reference and the generated output — using the SAME
//! ViT-H tower the provider uses. We generate twice at one seed: **with** IP (`ip_adapter_scale > 0`)
//! and **without** (`ip_adapter_scale = 0`, the branch gated off → plain SDXL), and assert the IP run's
//! reference-cosine is meaningfully higher — i.e. the IP path actually pulls the output toward the
//! reference. Plus the cancel contract (pre + mid-denoise). Outputs are written as PPM for eyeballing.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set IP_SDXL_BASE=...\RealVisXL_V5.0           # diffusers tree (unet/, text_encoder{,_2}/, …)
//! rem  The IP-Adapter bundle + ViT-H image encoder come from one pinned h94/IP-Adapter snapshot
//! rem  (`ip-adapter-plus-sdxl-vit-h` @ 018e402774aeeddd60609b4ecdb7e298259dc729 in
//! rem  release/real-weight-models.toml — materialize with ensure_model_snapshot.py; sc-13963):
//! set IP_ADAPTER_SNAPSHOT=...\ip-adapter        # h94/IP-Adapter repo snapshot dir
//! rem  … or override either component explicitly (an explicit var wins over the snapshot dir):
//! rem  set IP_BUNDLE=...\ip-adapter-plus_sdxl_vit-h.safetensors
//! rem  set IP_IMAGE_ENCODER=...\image_encoder   # dir with model.safetensors (or the file)
//! set SDXL_TOKENIZER_CLIP_L_DIR=...             # CLIP-L tokenizer dir (tokenizer.json) — passed-in component
//! set SDXL_TOKENIZER_CLIP_BIGG_DIR=...          # CLIP-bigG tokenizer dir (tokenizer.json)
//! set SDXL_VAE_FP16_FIX_DIR=...                 # madebyollin/sdxl-vae-fp16-fix dir (diffusion_pytorch_model.safetensors)
//! set IP_REF=...\ref.ppm                        # a reference image (P6 PPM)
//! set IP_OUT=...\out                            # output dir
//! cargo test -p candle-gen-sdxl --features cuda --release ip_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::DType;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress, WeightsSource};
// The sdxl-local `cosine` is a BARE dot product (inputs are already L2-normalized by
// `ClipMetric::feature`), so map it to the shared `cosine_dot` — NOT `cosine`, which would
// re-normalize and change the metric.
use candle_gen::testkit::{cosine_dot as cosine, env_path, read_ppm, write_ppm};

use crate::ip_adapter::preprocess_clip_image_sized;
use crate::ip_provider::{IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest};
use crate::vision_encoder::{ClipVisionEncoder, VisionConfig};
use crate::weights::Weights;

/// In-repo relative path of the IP-Adapter-Plus SDXL ViT-H bundle inside an `h94/IP-Adapter`
/// snapshot (`ip-adapter-plus-sdxl-vit-h` in `release/real-weight-models.toml`; sc-13963).
const IP_BUNDLE_REL: &str = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors";
/// In-repo relative path of the CLIP ViT-H image-encoder dir inside an `h94/IP-Adapter` snapshot.
const IP_IMAGE_ENCODER_REL: &str = "models/image_encoder";

/// Resolve one IP-Adapter component path (sc-13963): an explicit per-component override wins;
/// otherwise it is derived from the pinned `IP_ADAPTER_SNAPSHOT` repo-snapshot dir by joining `rel`
/// — the same explicit-path-or-snapshot-dir bridge the mmaudio harness uses (`tests/common/mod.rs`).
/// Returns `None` only when neither is set (the env wrapper turns that into an actionable panic).
fn resolve_ip_component(
    explicit: Option<String>,
    snapshot: Option<String>,
    rel: &str,
) -> Option<PathBuf> {
    explicit
        .map(PathBuf::from)
        .or_else(|| snapshot.map(|s| PathBuf::from(s).join(rel)))
}

/// Env wrapper over [`resolve_ip_component`]: `explicit_var` (a full path) wins, else
/// `IP_ADAPTER_SNAPSHOT` (the `h94/IP-Adapter` repo snapshot dir) joined with `rel`.
fn ip_component(explicit_var: &str, rel: &str) -> PathBuf {
    resolve_ip_component(
        std::env::var(explicit_var).ok(),
        std::env::var("IP_ADAPTER_SNAPSHOT").ok(),
        rel,
    )
    .unwrap_or_else(|| {
        panic!("set {explicit_var} (a path) or IP_ADAPTER_SNAPSHOT (an h94/IP-Adapter repo snapshot dir)")
    })
}

/// The snapshot-dir bridge derives the two IP-Adapter component paths, and an explicit per-component
/// override always wins — a pure-logic guard (no env / no GPU) for the sc-13963 wiring.
#[test]
fn ip_adapter_snapshot_derives_component_paths() {
    // Explicit per-component override wins, even when a snapshot is also set.
    assert_eq!(
        resolve_ip_component(
            Some("/x/bundle.safetensors".into()),
            Some("/snap".into()),
            IP_BUNDLE_REL,
        ),
        Some(PathBuf::from("/x/bundle.safetensors")),
    );
    // Else: the pinned snapshot dir joined with the in-repo relative path.
    assert_eq!(
        resolve_ip_component(None, Some("/snap".into()), IP_BUNDLE_REL),
        Some(PathBuf::from("/snap").join(IP_BUNDLE_REL)),
    );
    assert_eq!(
        resolve_ip_component(None, Some("/snap".into()), IP_IMAGE_ENCODER_REL),
        Some(PathBuf::from("/snap").join("models/image_encoder")),
    );
    // Neither set → None (the env wrapper turns this into an actionable panic).
    assert_eq!(resolve_ip_component(None, None, IP_BUNDLE_REL), None);
}

/// A standalone CLIP ViT-H feature extractor for the cosine metric (independent of the model's private
/// encoder): preprocess → penultimate → mean-pool over tokens → L2-normalize. Returns a 1280-vec.
struct ClipMetric {
    encoder: ClipVisionEncoder,
    size: usize,
    device: candle_core::Device,
}

impl ClipMetric {
    fn load(image_encoder: &Path) -> Self {
        let device = candle_gen::default_device().unwrap();
        let cfg = VisionConfig::vit_h_14();
        // Resolve a dir to model.safetensors; a file is used directly.
        let path = if image_encoder.is_file() {
            image_encoder.to_path_buf()
        } else {
            ["model.safetensors", "model.fp16.safetensors"]
                .iter()
                .map(|n| image_encoder.join(n))
                .find(|p| p.is_file())
                .unwrap_or_else(|| panic!("no model.safetensors under {}", image_encoder.display()))
        };
        let w = Weights::from_file(&path, &device, DType::F32).unwrap();
        let encoder = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        Self {
            encoder,
            size: cfg.image_size,
            device,
        }
    }

    fn feature(&self, img: &Image) -> Vec<f32> {
        let px = preprocess_clip_image_sized(img, self.size, &self.device).unwrap();
        let penult = self.encoder.penultimate(&px).unwrap(); // [1, N, 1280]
        let pooled = penult.mean(1).unwrap().flatten_all().unwrap(); // [1280]
        let v = pooled
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }
}

/// Drive the real SDXL IP-Adapter stack: a with-IP vs no-IP ablation (the IP run must score a higher
/// reference cosine) + the cancel contract. Visually-inspectable PPMs land in `IP_OUT`.
#[test]
#[ignore = "real-weight GPU validation; set IP_SDXL_BASE/IP_BUNDLE/IP_IMAGE_ENCODER/IP_REF/IP_OUT"]
fn real_weight_ip_adapter() {
    let out_dir = env_path("IP_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    // The bundle + image encoder resolve from one pinned IP_ADAPTER_SNAPSHOT repo dir, or from an
    // explicit IP_BUNDLE / IP_IMAGE_ENCODER override (sc-13963).
    let image_encoder = ip_component("IP_IMAGE_ENCODER", IP_IMAGE_ENCODER_REL);

    let paths = IpAdapterSdxlPaths {
        sdxl_base: env_path("IP_SDXL_BASE"),
        ip_adapter: ip_component("IP_BUNDLE", IP_BUNDLE_REL),
        image_encoder: image_encoder.clone(),
        // epic 13657 / sc-13663: tokenizers + fp16-fix VAE are passed-in components (env-pointed dirs).
        tokenizer_clip_l: WeightsSource::Dir(env_path("SDXL_TOKENIZER_CLIP_L_DIR")),
        tokenizer_clip_bigg: WeightsSource::Dir(env_path("SDXL_TOKENIZER_CLIP_BIGG_DIR")),
        vae_fp16_fix: WeightsSource::Dir(env_path("SDXL_VAE_FP16_FIX_DIR")),
    };
    let reference = read_ppm(&env_path("IP_REF"));
    println!(
        "reference {}x{}; loading IpAdapterSdxl …",
        reference.width, reference.height
    );

    let t0 = std::time::Instant::now();
    let mut model = IpAdapterSdxl::load(&paths).expect("load IpAdapterSdxl");
    println!("loaded in {:?}", t0.elapsed());

    let base = IpAdapterSdxlRequest {
        prompt: "a cinematic portrait photo, soft natural light, photorealistic, sharp focus"
            .into(),
        negative: "blurry, lowres, deformed, watermark, text".into(),
        width: 1024,
        height: 1024,
        steps: 30,
        guidance: 5.0,
        ip_adapter_scale: 0.7,
        sampler: None,
        scheduler: None,
        seed: 12345,
        // Native VAE: this harness validates the IP-Adapter pipeline, not the optional PiD SR (sc-8044).
        use_pid: false,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With IP.
    let t = std::time::Instant::now();
    let out_ip = model
        .generate(&base, &reference, &mut noop)
        .expect("generate (ip)");
    println!("[ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("ip.ppm"), &out_ip);

    // Without IP (scale 0 → branch gated off → plain SDXL at the same seed/prompt).
    let plain_req = IpAdapterSdxlRequest {
        ip_adapter_scale: 0.0,
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &reference, &mut noop)
        .expect("generate (no ip)");
    println!("[no-ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("no_ip.ppm"), &out_plain);

    // CLIP-feature cosine to the reference: the IP run must pull meaningfully closer.
    let metric = ClipMetric::load(&image_encoder);
    let ref_feat = metric.feature(&reference);
    let cos_ip = cosine(&ref_feat, &metric.feature(&out_ip));
    let cos_plain = cosine(&ref_feat, &metric.feature(&out_plain));
    println!("=== SDXL IP-Adapter validation ===");
    println!("  clip cosine (ip)    : {cos_ip:.4}");
    println!("  clip cosine (no-ip) : {cos_plain:.4}");
    println!("  delta               : {:.4}", cos_ip - cos_plain);
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel: a flag set before the first step short-circuits.
    let cancelled = IpAdapterSdxlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &reference, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel: flip the flag from the progress callback on the 3rd step; the next step's
    // start-of-loop check must short-circuit.
    let mid = CancelFlag::new();
    let mid_req = IpAdapterSdxlRequest {
        cancel: mid.clone(),
        ..base.clone()
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid.cancel();
            }
        }
    };
    let res = model.generate(&mid_req, &reference, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: IP conditioning pulls the output toward the reference in CLIP space.
    assert!(
        cos_ip > cos_plain + 0.02,
        "IP run cosine {cos_ip:.4} not meaningfully above no-IP {cos_plain:.4}"
    );
    assert!(out_ip.width == 1024 && out_ip.height == 1024);
    println!("SDXL IP-Adapter validation PASS ✅");
}
