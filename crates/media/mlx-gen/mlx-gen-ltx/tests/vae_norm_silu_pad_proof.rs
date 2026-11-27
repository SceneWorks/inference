//! SW-6142 (SW-F287 M4) — **LTX-2.3 video VAE fused pixel-norm → SiLU → replicate pad**: the
//! real-weight proof and bench, one registered model proof for one dispatch.
//!
//! 1. The `mlx_gen::vae_norm_silu_pad::parity` harness over the LTX conv-input geometries (Metal
//!    route vs the reference op chain: bit identity + E2 tolerance) and the wrong-temporal-side
//!    mutation control on the causal replicate pad (must fail parity).
//! 2. Per route (`reference`, `compiled`, `metal`), in **its own child process** (MLX's peak
//!    counter is process-wide; the route is selected by `MLX_GEN_VAE_NORM_SILU_PAD`):
//!    - a bench of the conv input alone at the decoder's final-stage shape `[1, 128, 17, 192, 256]`
//!      (non-causal, one replicated frame each side) — ms/iter and peak;
//!    - the decode of a fixed latent `[1, 128, 3, 12, 16]` (→ 17 × 384 × 512) — time, peak, every
//!      norm→SiLU→conv site's route;
//!    - the causal encode of a fixed clip `[1, 3, 9, 256, 256]`.
//! 3. Each fused route's decode/encode vs the reference route: PSNR and bit identity.
//!
//! Everything is printed before anything is asserted. Asserted: Metal parity within E2 tolerance,
//! every mutation fails, every normed conv of the Metal child took the Metal kernel, and the Metal
//! decode and encode are ≥ 50 dB from the reference route. The compiled route is reported only.
//!
//! `LTX_VAE_SNAPSHOT` is the pinned `SceneWorks/ltx-2.3-mlx` q4 snapshot directory (its
//! `vae_decoder.safetensors` / `vae_encoder.safetensors` + `embedded_config.json`);
//! `model-inference-proofs.yml` stages it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
use mlx_rs::{random, Array};

use mlx_gen::vae_norm_silu_pad::{
    self as norm_silu_pad, parity, ChannelNorm, NormSiluPadRoute, PadSpec, TemporalPad,
};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::{LtxVaeConfig, LtxVideoVae};

const FQN: &str = "vae_norm_silu_pad_proof::ltx_vae_norm_silu_pad_decode_proof";
const SNAPSHOT_ENV: &str = "LTX_VAE_SNAPSHOT";
const CHILD_ENV: &str = "SW6142_LTX_PROOF_CHILD";
const OUT_ENV: &str = "SW6142_LTX_PROOF_OUT";

const LATENT: [i32; 5] = [1, 128, 3, 12, 16];
const CLIP: [i32; 5] = [1, 3, 9, 256, 256];
const BENCH_SHAPE: [i32; 5] = [1, 128, 17, 192, 256];
const BENCH_ITERS: usize = 3;
const ROUTES: [NormSiluPadRoute; 3] = [
    NormSiluPadRoute::Reference,
    NormSiluPadRoute::Compiled,
    NormSiluPadRoute::Metal,
];
const MIB: f64 = 1024.0 * 1024.0;

fn eval(x: &Array) {
    mlx_rs::transforms::eval([x]).unwrap();
}

/// Run `f` from an emptied allocator cache: `(result, ms, peak bytes above the starting live set)`.
fn measured<T>(f: impl FnOnce() -> T) -> (T, f64, u64) {
    clear_cache();
    reset_peak_memory();
    let base = get_active_memory() as u64;
    let start = Instant::now();
    let out = f();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let peak = (get_peak_memory() as u64).saturating_sub(base);
    (out, ms, peak)
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// PSNR of `got` against `want` over a peak-to-peak `range`.
fn psnr_db(got: &[f32], want: &[f32], range: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    let mse = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / want.len().max(1) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (range * range / mse).log10()
    }
}

fn peak_to_peak(values: &[f32]) -> f64 {
    let (lo, hi) = values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    f64::from(hi - lo)
}

fn chip() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace(' ', "_"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Normed conv sites of a block list: two per residual layer of every non-compress block.
fn res_sites(blocks: &[mlx_gen_ltx::VaeBlock]) -> u64 {
    blocks
        .iter()
        .filter(|b| !b.is_compress())
        .map(|b| 2 * b.num_layers as u64)
        .sum()
}

/// The conv input of the decoder's final-stage residual conv, by `route`, as the contiguous
/// NDHWC buffer the conv reads.
fn bench_conv_input(route: NormSiluPadRoute, x: &Array) -> Array {
    let norm = ChannelNorm::Pixel { eps: 1e-8 };
    let spec = PadSpec {
        temporal: TemporalPad::Replicate { front: 1, back: 1 },
        cache: None,
        ph: 1,
        pw: 1,
    };
    let ndhwc = if route == NormSiluPadRoute::Reference {
        norm_silu_pad::reference_conv_input(x, &norm, &spec, 0)
            .unwrap()
            .ndhwc
    } else {
        norm_silu_pad::try_fused(x, &norm, &spec, 0)
            .unwrap()
            .expect("the fused route serves the bench shape")
            .ndhwc
    };
    let out = mlx_gen::array::contiguous(&ndhwc).unwrap();
    eval(&out);
    out
}

/// One route's measurements, in its own process. Prints one `SW6142_LTX_CHILD` line.
fn child(route: NormSiluPadRoute, out_dir: &Path) {
    assert_eq!(norm_silu_pad::route(), route);
    let dir = PathBuf::from(std::env::var(SNAPSHOT_ENV).unwrap());
    let cfg = LtxVaeConfig::from_model_dir(&dir).unwrap();
    let ((dec_w, enc_w, vae), load_ms, _) = measured(|| {
        let dec_w = Weights::from_file(dir.join("vae_decoder.safetensors")).unwrap();
        let enc_w = Weights::from_file(dir.join("vae_encoder.safetensors")).unwrap();
        let vae = LtxVideoVae::from_weights(&dec_w, Some(&enc_w), &cfg).unwrap();
        (dec_w, enc_w, vae)
    });
    drop((dec_w, enc_w));
    let decode_expected = res_sites(&cfg.decoder_blocks) + 1;
    let encode_expected = res_sites(&cfg.encoder_blocks) + 1;

    // Warm-up: JIT-compile the kernel variants / compiled graphs outside every measurement.
    {
        let key = random::key(1).unwrap();
        let z = random::normal::<f32>(&[1, 128, 2, 2, 2], None, None, Some(&key)).unwrap();
        let clip = random::uniform::<f32, f32>(-1.0, 1.0, &[1, 3, 9, 64, 64], Some(&key)).unwrap();
        eval(&vae.decode(&z).unwrap());
        eval(&vae.encode(&clip).unwrap());
    }

    let key = random::key(61420).unwrap();
    let x = random::normal::<f32>(&BENCH_SHAPE, None, None, Some(&key)).unwrap();
    eval(&x);
    drop(bench_conv_input(route, &x));
    let mut bench = Vec::new();
    for _ in 0..BENCH_ITERS {
        let (out, ms, peak) = measured(|| bench_conv_input(route, &x));
        drop(out);
        bench.push((ms, peak));
    }
    drop(x);
    bench.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (bench_ms, _) = bench[BENCH_ITERS / 2];
    let bench_peak = bench.iter().map(|b| b.1).max().unwrap();

    let key = random::key(6142).unwrap();
    let latent = random::normal::<f32>(&LATENT, None, None, Some(&key)).unwrap();
    eval(&latent);
    let start = norm_silu_pad::thread_counts();
    let (video, decode_ms, decode_peak) = measured(|| {
        let video = vae.decode(&latent).unwrap();
        eval(&video);
        video
    });
    let decode_counts = norm_silu_pad::thread_counts().since(&start);
    write_f32(
        &out_dir.join(format!("decode-{}.f32", route.label())),
        &parity::host_f32(&video).unwrap(),
    );
    let video_shape = video.shape().to_vec();
    drop((video, latent));

    let key = random::key(6143).unwrap();
    let clip = random::uniform::<f32, f32>(-1.0, 1.0, &CLIP, Some(&key)).unwrap();
    eval(&clip);
    let start = norm_silu_pad::thread_counts();
    let (latent_out, encode_ms, encode_peak) = measured(|| {
        let z = vae.encode(&clip).unwrap();
        eval(&z);
        z
    });
    let encode_counts = norm_silu_pad::thread_counts().since(&start);
    write_f32(
        &out_dir.join(format!("encode-{}.f32", route.label())),
        &parity::host_f32(&latent_out).unwrap(),
    );

    let served = |c: &norm_silu_pad::RouteCounts| match route {
        NormSiluPadRoute::Reference => c.reference,
        NormSiluPadRoute::Compiled => c.compiled,
        NormSiluPadRoute::Metal => c.metal,
    };
    println!(
        "SW6142_LTX_CHILD route={} load_ms={load_ms:.1} bench_ms={bench_ms:.2} \
         bench_peak_bytes={bench_peak} decode_ms={decode_ms:.1} decode_peak_bytes={decode_peak} \
         decode_sites={} decode_expected={decode_expected} decode_refused={} \
         encode_ms={encode_ms:.1} encode_peak_bytes={encode_peak} encode_sites={} \
         encode_expected={encode_expected} encode_refused={} video_shape={video_shape:?} \
         latent_shape={:?}",
        route.label(),
        served(&decode_counts),
        decode_counts.refused,
        served(&encode_counts),
        encode_counts.refused,
        latent_out.shape(),
    );
}

fn parse_child(stdout: &str) -> HashMap<String, String> {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("SW6142_LTX_CHILD "))
        .unwrap_or_else(|| panic!("child printed no SW6142_LTX_CHILD line:\n{stdout}"));
    line.split_whitespace()
        .skip(1)
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn ltx_vae_norm_silu_pad_decode_proof() {
    if let Ok(route) = std::env::var(CHILD_ENV) {
        let route = NormSiluPadRoute::parse(&route).expect("child route");
        child(route, Path::new(&std::env::var(OUT_ENV).unwrap()));
        return;
    }
    let Some(snapshot) = std::env::var_os(SNAPSHOT_ENV) else {
        eprintln!("skip: set {SNAPSHOT_ENV} to the ltx-2.3-mlx q4 snapshot directory");
        crate::model_inference_device::record_model_inference_prerequisite_missing(
            "required work unavailable in ltx_vae_norm_silu_pad_decode_proof; see preceding skip diagnostic",
        );
        return;
    };
    assert!(
        Path::new(&snapshot)
            .join("vae_decoder.safetensors")
            .is_file(),
        "{SNAPSHOT_ENV} has no vae_decoder.safetensors"
    );
    let device = chip();
    let mut failures: Vec<String> = Vec::new();

    // 1. Kernel parity over the LTX geometries and the mutation control.
    let cases = parity::ltx_cases();
    let metal = parity::run_all(&cases, NormSiluPadRoute::Metal).unwrap();
    println!(
        "SW6142_LTX_PARITY_SUMMARY device={device} cases={} metal_bitwise={} metal_within_tol={}",
        cases.len(),
        metal.iter().filter(|r| r.bitwise()).count(),
        metal.iter().filter(|r| r.within_tolerance()).count(),
    );
    for report in metal.iter().filter(|r| !r.within_tolerance()) {
        failures.push(format!("metal parity: {}", report.line()));
    }
    for case in parity::mutation_cases()
        .into_iter()
        .filter(|case| case.name.starts_with("ltx_"))
    {
        let c = parity::run_mutation(&case).unwrap();
        println!(
            "SW6142_MUTATION case={} mismatches={} max_abs={:.3e}",
            case.name, c.mismatches, c.max_abs
        );
        if c.bitwise() || c.max_abs <= 1e-2 {
            failures.push(format!("mutation {} did not fail parity", case.name));
        }
    }

    // 2. One child per route.
    let dir = tempfile::tempdir().unwrap();
    let mut children = HashMap::new();
    for route in ROUTES {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                FQN,
                "--ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD_ENV, route.label())
            .env(norm_silu_pad::ROUTE_ENV, route.label())
            .env(OUT_ENV, dir.path())
            .output()
            .expect("spawn the per-route child");
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{} child failed:\n{stdout}\n{}",
            route.label(),
            String::from_utf8_lossy(&output.stderr)
        );
        // The child's result line, and its in-scope input receipts (the children are the ones
        // that read the pinned weights; libtest summaries stay in the child's own output).
        for line in stdout.lines().filter(|l| {
            l.starts_with("SW6142_LTX_CHILD ") || l.starts_with("MODEL_INFERENCE_INPUT_USED ")
        }) {
            println!("\n{line}");
        }
        children.insert(route, parse_child(&stdout));
    }

    // 3. Fused routes vs the reference route.
    let read = |what: &str, route: NormSiluPadRoute| {
        read_f32(&dir.path().join(format!("{what}-{}.f32", route.label())))
    };
    let (ref_video, ref_latent) = (
        read("decode", NormSiluPadRoute::Reference),
        read("encode", NormSiluPadRoute::Reference),
    );
    let latent_range = peak_to_peak(&ref_latent);
    let field =
        |route: NormSiluPadRoute, key: &str| -> f64 { children[&route][key].parse().unwrap() };
    let reference = NormSiluPadRoute::Reference;
    for route in [NormSiluPadRoute::Compiled, NormSiluPadRoute::Metal] {
        let (video, latent) = (read("decode", route), read("encode", route));
        let (decode_db, encode_db) = (
            psnr_db(&video, &ref_video, 2.0),
            psnr_db(&latent, &ref_latent, latent_range),
        );
        println!(
            "SW6142_LTX_DECODE device={device} route={} latent={LATENT:?} decode_psnr_db={decode_db:.2} \
             decode_bitwise={} encode_psnr_db={encode_db:.2} encode_bitwise={} \
             decode_ms={:.1} (reference {:.1}) decode_peak_mib={:.1} (reference {:.1}) \
             encode_ms={:.1} (reference {:.1}) encode_peak_mib={:.1} (reference {:.1})",
            route.label(),
            video == ref_video,
            latent == ref_latent,
            field(route, "decode_ms"),
            field(reference, "decode_ms"),
            field(route, "decode_peak_bytes") / MIB,
            field(reference, "decode_peak_bytes") / MIB,
            field(route, "encode_ms"),
            field(reference, "encode_ms"),
            field(route, "encode_peak_bytes") / MIB,
            field(reference, "encode_peak_bytes") / MIB,
        );
        if route == NormSiluPadRoute::Metal {
            if decode_db < 50.0 || encode_db < 50.0 {
                failures.push(format!(
                    "metal decode {decode_db:.2} dB / encode {encode_db:.2} dB below 50 dB"
                ));
            }
            let sites = [
                ("decode_sites", "decode_expected", "decode_refused"),
                ("encode_sites", "encode_expected", "encode_refused"),
            ];
            for (served, expected, refused) in sites {
                if field(route, served) != field(route, expected) || field(route, refused) != 0.0 {
                    failures.push(format!(
                        "metal child served {} of {} {served} with {} refusals",
                        field(route, served),
                        field(route, expected),
                        field(route, refused)
                    ));
                }
            }
        }
    }
    for (label, key_ms, key_peak, shape) in [
        (
            "vae_norm_silu_pad",
            "bench_ms",
            "bench_peak_bytes",
            format!("{BENCH_SHAPE:?}"),
        ),
        (
            "ltx_decode",
            "decode_ms",
            "decode_peak_bytes",
            format!("{LATENT:?}"),
        ),
        (
            "ltx_encode",
            "encode_ms",
            "encode_peak_bytes",
            format!("{CLIP:?}"),
        ),
    ] {
        println!(
            "SW6142_BENCH device={device} kernel={label} shape={shape} reference_ms={:.2} \
             compiled_ms={:.2} metal_ms={:.2} reference_peak_mib={:.1} compiled_peak_mib={:.1} \
             metal_peak_mib={:.1}",
            field(NormSiluPadRoute::Reference, key_ms),
            field(NormSiluPadRoute::Compiled, key_ms),
            field(NormSiluPadRoute::Metal, key_ms),
            field(NormSiluPadRoute::Reference, key_peak) / MIB,
            field(NormSiluPadRoute::Compiled, key_peak) / MIB,
            field(NormSiluPadRoute::Metal, key_peak) / MIB,
        );
    }

    assert!(
        failures.is_empty(),
        "SW-6142 proof failures:\n{}",
        failures.join("\n")
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED vae_norm_silu_pad_proof::ltx_vae_norm_silu_pad_decode_proof");
}
