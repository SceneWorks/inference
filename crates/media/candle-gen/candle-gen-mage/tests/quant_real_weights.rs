//! Physical q4/q8/bf16 acceptance for sc-14053.
//!
//! This is intentionally ignored in ordinary CI: it loads the complete checkpoint and must run on
//! the CUDA lane with enough host RAM/VRAM. The workflow invokes this binary once per tier so every
//! VRAM sample starts from a fresh process and idle CUDA context. It drives the registered production
//! loader, compares the final image to the independently generated Torch oracle, and samples the
//! load/steady/overall device peak that SceneWorks' admission gate consumes.

use candle_core::{DType, Device};
use candle_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use candle_gen::testkit::VramProbe;
use candle_gen_mage::REGISTRATION;

const PROMPT: &str = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug";

fn snapshot() -> std::path::PathBuf {
    std::env::var_os("CANDLE_MAGE_SNAPSHOT")
        .map(Into::into)
        .expect("set CANDLE_MAGE_SNAPSHOT to a complete Mage-Flow snapshot")
}

fn golden() -> Vec<u8> {
    let path = std::path::PathBuf::from(
        std::env::var("MAGE_GOLDEN_DIR")
            .expect("set MAGE_GOLDEN_DIR to the Torch oracle directory"),
    )
    .join("mage_flow_e2e_golden.safetensors");
    let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    let geometry = tensors["geometry"].to_vec1::<i32>().expect("geometry");
    assert_eq!(geometry, [1024, 1024, 20, 4], "Torch oracle geometry");
    tensors["image_u8"]
        .to_dtype(DType::U8)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap()
}

fn tier() -> (&'static str, Option<Quant>, f64, f64, f64) {
    match std::env::var("MAGE_QUANT_TIER")
        .expect("set MAGE_QUANT_TIER to q4, q8, or bf16")
        .to_ascii_lowercase()
        .as_str()
    {
        // Absolute output gates are intentionally tier-specific. BF16 inherits the independently
        // calibrated 0.10 Torch gate from cuda_1024; Q8/Q4 receive bounded quantization budgets rather
        // than the old relative-only Q8<Q4 assertion.
        "q4" => ("q4", Some(Quant::Q4), 15.0, 38.0, 0.24),
        "q8" => ("q8", Some(Quant::Q8), 17.0, 22.0, 0.14),
        "bf16" => ("bf16", None, 21.0, 16.0, 0.10),
        other => panic!("unknown MAGE_QUANT_TIER {other:?}; expected q4, q8, or bf16"),
    }
}

fn mae(left: &[u8], right: &[u8]) -> f64 {
    assert_eq!(left.len(), right.len(), "image shape mismatch");
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a.abs_diff(b)))
        .sum::<f64>()
        / left.len() as f64
}

fn mean_relative_error(got: &[u8], want: &[u8]) -> f64 {
    let sum_delta = got
        .iter()
        .zip(want)
        .map(|(&got, &want)| f64::from(got.abs_diff(want)))
        .sum::<f64>();
    let sum_want = want.iter().map(|&value| f64::from(value)).sum::<f64>();
    sum_delta / sum_want.max(f64::MIN_POSITIVE)
}

#[test]
#[ignore = "requires CANDLE_MAGE_SNAPSHOT, MAGE_GOLDEN_DIR, and a physical idle CUDA runner"]
fn registered_tier_matches_independent_oracle_and_vram_budget() {
    let (tier, quant, max_peak_gb, max_mae, max_mean_rel) = tier();
    let mut probe = VramProbe::start_rendered().assert_idle(1.0);
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = quant;
    let generator = (REGISTRATION.load)(&spec).expect("registered Mage load");
    probe.end_load(load_phase);

    let generation_phase = probe.phase();
    let output = generator
        .generate(
            &GenerationRequest {
                prompt: PROMPT.into(),
                width: 1024,
                height: 1024,
                steps: Some(20),
                guidance: Some(5.0),
                seed: Some(42),
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("Mage generation");
    probe.end_gen(generation_phase);
    let report = probe.report().assert_trustworthy(1.0);
    let pixels = match output {
        GenerationOutput::Images(images) => images.into_iter().next().expect("one image").pixels,
        _ => panic!("expected image output"),
    };
    assert_eq!(pixels.len(), 1024 * 1024 * 3, "{tier}: image geometry");
    let want = golden();
    let image_mae = mae(&pixels, &want);
    let mean_rel = mean_relative_error(&pixels, &want);
    assert!(
        image_mae <= max_mae,
        "{tier}: Torch-oracle MAE {image_mae:.3} exceeds tier gate {max_mae:.3}"
    );
    assert!(
        mean_rel <= max_mean_rel,
        "{tier}: Torch-oracle mean_rel {mean_rel:.6} exceeds tier gate {max_mean_rel:.6}"
    );
    assert!(
        report.peak_gb <= max_peak_gb,
        "{tier}: measured physical peak {:.2} GB exceeds SceneWorks gate row {max_peak_gb:.2} GB",
        report.peak_gb
    );
    println!(
        "[[MAGE_VRAM]] {{\"tier\":\"{tier}\",\"loadPeakGb\":{:.2},\"steadyGb\":{:.2},\
         \"peakGb\":{:.2},\"maxPeakGb\":{max_peak_gb:.2},\"oracleMae\":{image_mae:.3},\
         \"oracleMeanRel\":{mean_rel:.6}}}",
        report.load_peak_gb, report.steady_gb, report.peak_gb
    );
}
