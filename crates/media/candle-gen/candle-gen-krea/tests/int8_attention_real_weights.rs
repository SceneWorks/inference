//! SW-6131 (SW-F287 S6) — the opt-in INT8 QK attention engine option on real Krea 2 Turbo weights.
//!
//! Fixed-seed renders through the registered generator, `KreaEngineOptions::default()` (the
//! FlashAttention engine) vs `int8_attention: true`, at 1024² (8 steps) and 2048² (4 steps):
//!
//! - **AC2** the 1024² INT8 render is >= 35 dB PSNR against the flash render, and the default
//!   engine takes no INT8 route at all (zero INT8 calls and zero INT8 refusals in the counters);
//! - the s/step of both engines at both sizes (the POC gate's step-time question), from the renders
//!   and, more precisely, from alternating synchronized DiT forwards on one resident trunk;
//! - **AC1 on captured operands**: one production attention call's Q/K/V captured from each INT8
//!   render (S = 4096 + text at 1024², 16384 + text at 2048²) replayed through the INT8 kernel with
//!   K centering off / on / adaptive against the f32 materialized reference: cosine >= 0.999.
//!
//! ```sh
//! KREA_TURBO_DIR=<snapshot>/bf16 KREA_SMOKE_ARTIFACT_DIR=<out> \
//!   cargo test -p candle-gen-krea --release --features cuda,flash-attn --test integration \
//!   int8_attention_real_weights:: -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::fused_attention::{
    arm_int8_operand_capture, attention_path_counts, enter_int8_attention,
    take_int8_operand_capture, AttentionPathCounts, KCentering,
};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use candle_gen_krea::KreaEngineOptions;

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> PathBuf {
    std::env::var("KREA_TURBO_DIR")
        .map(PathBuf::from)
        .expect("set KREA_TURBO_DIR to the bf16 Krea 2 Turbo snapshot")
}

fn artifact_dir() -> PathBuf {
    let dir = std::env::var("KREA_SMOKE_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("krea_int8_attention");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One render: the image, the steady-state seconds per step, and the attention path counts of
/// the render.
struct Render {
    image: Image,
    s_per_step: f64,
    counts: AttentionPathCounts,
}

fn render(options: KreaEngineOptions, size: u32, steps: u32, capture_skip: Option<u64>) -> Render {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let generator = candle_gen_krea::load_with_options(&spec, options).expect("load krea_2_turbo");
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        steps: Some(steps),
        ..Default::default()
    };
    if let Some(skip) = capture_skip {
        arm_int8_operand_capture(skip);
    }
    let mut step_times = Vec::new();
    let before = attention_path_counts();
    let out = generator
        .generate(&request, &mut |progress| {
            if let Progress::Step { .. } = progress {
                step_times.push(Instant::now());
            }
        })
        .expect("generate");
    let counts = attention_path_counts().since(&before);
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected images");
    };
    assert_eq!(images.len(), 1);
    let image = images.pop().unwrap();
    assert_eq!((image.width, image.height), (size, size));
    // Steady state: the first step carries kernel compiles and warm-up.
    let s_per_step = match step_times.as_slice() {
        [first, .., last] if step_times.len() > 1 => {
            (*last - *first).as_secs_f64() / (step_times.len() - 1) as f64
        }
        _ => f64::NAN,
    };
    drop(generator);
    Render {
        image,
        s_per_step,
        counts,
    }
}

fn save(image: &Image, name: &str) {
    let path = artifact_dir().join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

/// PSNR (dB) of two RGB8 images.
fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let mse = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.pixels.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// `(cosine, rel-L2)` of `a` against `b`, in f32.
fn similarity(a: &Tensor, b: &Tensor) -> (f32, f32) {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let s = |t: Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let dot = s((&a * &b).unwrap());
    let na = s(a.sqr().unwrap()).sqrt();
    let nb = s(b.sqr().unwrap()).sqrt();
    let diff = s((&a - &b).unwrap().sqr().unwrap()).sqrt();
    (dot / (na * nb), diff / nb)
}

/// Replay one captured production call through the INT8 kernel (each centering mode) against
/// the f32 materialized reference; cosine >= 0.999 each.
fn replay_captured(label: &str) {
    use candle_gen::attention::{sdpa_planned_bhsd, sdpa_planned_bhsd_materialized};
    use candle_gen::candle_nn::ops::softmax_last_dim;
    use candle_gen::gen_core::attention_budget::AttentionPlan;
    let call = take_int8_operand_capture().expect("the armed INT8 call was captured");
    let plan = AttentionPlan::budgeted(candle_gen::attention::attention_budget_from_usize(
        candle_gen::ATTN_SCORES_BUDGET,
    ));
    let (_, h, s, d) = call.q.dims4().unwrap();
    let hk = call.k.dim(1).unwrap();
    let group = h / hk;
    let k = candle_gen::attention::repeat_kv(&call.k, group).unwrap();
    let v = candle_gen::attention::repeat_kv(&call.v, group).unwrap();
    let f32 = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
    let want = sdpa_planned_bhsd_materialized(
        &f32(&call.q),
        &f32(&k),
        &f32(&v),
        call.scale,
        None,
        softmax_last_dim,
        plan,
    )
    .unwrap();
    let flash = sdpa_planned_bhsd(
        &call.q,
        &call.k,
        &call.v,
        call.scale,
        None,
        softmax_last_dim,
        plan,
    )
    .unwrap();
    let (fc, fr) = similarity(&flash, &want);
    eprintln!(
        "[captured {label}] S={s} H={h}/{hk} d={d} {:?}: flash cosine {fc:.6} rel-L2 {fr:.3e}",
        call.q.dtype()
    );
    for centering in [KCentering::Off, KCentering::Always, KCentering::Adaptive] {
        let got = {
            let _scope = enter_int8_attention(centering);
            sdpa_planned_bhsd(
                &call.q,
                &call.k,
                &call.v,
                call.scale,
                None,
                softmax_last_dim,
                plan,
            )
            .unwrap()
        };
        let (cos, rel) = similarity(&got, &want);
        eprintln!("[captured {label}] int8 {centering:?}: cosine {cos:.6} rel-L2 {rel:.3e}");
        assert!(cos >= 0.999, "captured {label} {centering:?}: cosine {cos}");
    }
}

/// The default engine took no INT8 route at all, and took FlashAttention.
fn assert_default_engine(r: &Render) {
    assert_eq!(
        r.counts.int8, 0,
        "the default engine took INT8: {:?}",
        r.counts
    );
    assert_eq!(
        r.counts.int8_refusals.total(),
        0,
        "the default engine entered the INT8 scope: {:?}",
        r.counts
    );
    assert!(r.counts.flash > 0, "{:?}", r.counts);
}

/// One size, rendered flash -> INT8 -> flash (the board is power-capped, so the INT8 step time is
/// compared with the mean of the two flash renders around it). Captures one mid-trajectory block
/// call of the INT8 render. Returns the INT8-vs-flash PSNR.
fn compare(size: u32, steps: u32, blocks: u64) -> f64 {
    let flash = render(KreaEngineOptions::default(), size, steps, None);
    assert_default_engine(&flash);
    // Capture the middle block of the middle step (one INT8 call per block per step).
    let skip = (steps as u64 / 2) * blocks + blocks / 2;
    let int8 = render(
        KreaEngineOptions {
            int8_attention: true,
        },
        size,
        steps,
        Some(skip),
    );
    assert!(
        int8.counts.int8 >= steps as u64 * blocks,
        "every denoise block attention should take INT8: {:?}",
        int8.counts
    );
    let flash_again = render(KreaEngineOptions::default(), size, steps, None);
    assert_default_engine(&flash_again);
    // The option off is the flash engine, run to run: FlashAttention's forward is deterministic.
    let control = psnr(&flash_again.image, &flash.image);
    let db = psnr(&int8.image, &flash.image);
    save(&flash.image, &format!("fox_{size}_s{steps}_flash"));
    save(&int8.image, &format!("fox_{size}_s{steps}_int8"));
    let flash_s = (flash.s_per_step + flash_again.s_per_step) / 2.0;
    eprintln!(
        "[krea_2_turbo {size}x{size} {steps}-step seed 7] flash {:.3} / {:.3} s/step (mean {flash_s:.3}), \
         int8 {:.3} s/step ({:+.1}%); PSNR int8 vs flash {db:.2} dB, flash vs flash {control:.2} dB; \
         int8 counts {:?}",
        flash.s_per_step,
        flash_again.s_per_step,
        int8.s_per_step,
        (int8.s_per_step / flash_s - 1.0) * 100.0,
        int8.counts
    );
    assert!(
        control.is_infinite(),
        "two default-engine renders differ ({control:.2} dB)"
    );
    replay_captured(&format!("{size}x{size}"));
    db
}
/// The DiT step-time A/B with one resident real-weight trunk: one denoise forward with the INT8
/// option off, then on, alternating (`rounds` pairs per size after a warm pair), each synchronized
/// and timed — the board is power-capped, so render-to-render step times drift by more than the
/// effect. Prints the median seconds per forward (= per Turbo step) of each.
fn dit_forward_ab(sizes: &[usize], rounds: usize) {
    use candle_gen::candle_core::Device;
    use candle_gen_krea::loader::Weights;
    use candle_gen_krea::{Krea2Config, Krea2Transformer};
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let root = snapshot();
    let cfg = Krea2Config::from_snapshot(&root).expect("config");
    let w = Weights::from_dir(&root.join("transformer"), &dev, DType::BF16).expect("weights");
    let mut dit = Krea2Transformer::load(&w, &cfg).expect("trunk");
    drop(w);
    // A 128-token caption context (timing only: the values do not change the work).
    let ctx = Tensor::randn(
        0f32,
        1.0,
        (1, 128, cfg.num_text_layers, cfg.text_hidden_dim),
        &dev,
    )
    .unwrap();
    let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
    for &edge in sizes {
        let x = Tensor::randn(0f32, 1.0, (1, 16, edge / 8, edge / 8), &dev).unwrap();
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..=rounds {
            for (i, on) in [false, true].into_iter().enumerate() {
                dit.set_int8_attention(on);
                dev.synchronize().unwrap();
                let before = attention_path_counts();
                let start = Instant::now();
                drop(dit.forward(&x, &t, &ctx).expect("DiT forward"));
                dev.synchronize().unwrap();
                let s = start.elapsed().as_secs_f64();
                let counts = attention_path_counts().since(&before);
                assert_eq!(counts.int8 > 0, on, "{counts:?}");
                if round > 0 {
                    times[i].push(s);
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (off, on) = (median(&mut times[0]), median(&mut times[1]));
        eprintln!(
            "[krea DiT forward {edge}x{edge}, real weights, {rounds} alternating pairs] flash {off:.3} s, \
             int8 {on:.3} s ({:+.1}%)",
            (on / off - 1.0) * 100.0
        );
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn krea_turbo_int8_attention_matches_flash_render() {
    let blocks = 28; // Krea 2's single-stream blocks: one attention call each per forward.
    let db_1024 = compare(1024, 8, blocks);
    assert!(
        db_1024 >= 35.0,
        "1024x1024 INT8 vs flash PSNR {db_1024:.2} dB < 35"
    );
    let db_2048 = compare(2048, 4, blocks);
    eprintln!("PSNR 1024x1024 {db_1024:.2} dB, 2048x2048 {db_2048:.2} dB");
    dit_forward_ab(&[1024, 2048], 5);

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED int8_attention_real_weights::krea_turbo_int8_attention_matches_flash_render"
    );
}
