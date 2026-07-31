//! SC-15800: Mage-Flow text-encoder residency on real Apple/Metal weights.
//!
//! The runner sweeps every numeric tier via `MAGE_REQUEST_SCOPE_QUANT`, measures both the
//! conditioning phase and complete request, checks exact identity, sweeps prompt length because
//! Mage packs variable-length segments without padding, and runs a lazy-carry mutation control.

#![cfg(target_os = "macos")]

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, Generator, LoadShape, LoadSpec, OffloadPolicy,
    Quant, WeightsSource,
};
use mlx_gen_mage::config::{QwenVlTextConfig, TE_RMS_NORM_EPS, TE_ROPE_THETA};
use mlx_gen_mage::model::{COMPONENT_TEXT_ENCODER, REGISTRATION};
use mlx_gen_mage::text_encoder::{
    load_dir, MageTextEncoder, PromptKind, Qwen3VlTextEncoder, LM_PREFIX,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const PROMPT: &str = "A weathered lighthouse keeper on a stone pier at dawn, sea spray on a heavy wool coat, low coastal fog, precise cinematic lighting";

fn snapshot() -> std::path::PathBuf {
    std::env::var("MAGE_REQUEST_SCOPE_SNAPSHOT")
        .expect("set MAGE_REQUEST_SCOPE_SNAPSHOT")
        .into()
}

fn quant() -> Option<Quant> {
    match std::env::var("MAGE_REQUEST_SCOPE_QUANT")
        .unwrap_or_else(|_| "bf16".into())
        .as_str()
    {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("MAGE_REQUEST_SCOPE_QUANT must be q4, q8, or bf16, got {other}"),
    }
}

fn env_dir(key: &str) -> Option<std::path::PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(Into::into)
}

/// Resolve a transfer-only text-encoder artifact for the requested tier. The real-weights runner
/// may provide the installed shared-components tree; otherwise build the packed component once in
/// the runner temp directory. Runtime quantization inside every window is deliberately not tested
/// or published as rung 4 because it is a repeated format conversion, not device-format transfer.
fn text_encoder_dir(tier: Option<Quant>) -> std::path::PathBuf {
    let Some(tier) = tier else {
        return snapshot().join("text_encoder");
    };
    let label = match tier {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => panic!("Mage/MLX does not offer an NVFP4 tier"),
    };
    if let Some(root) = env_dir("MAGE_COMPONENTS_ROOT") {
        return root.join(label).join("text_encoder");
    }
    let destination = std::env::temp_dir()
        .join("mage-sc-15800-components")
        .join(label);
    if !destination.join("text_encoder").is_dir() {
        mlx_gen_mage::convert::prequantize_shared_components(&snapshot(), &destination, label)
            .expect("prequantize transfer-only text encoder");
    }
    destination.join("text_encoder")
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn prepare(mut encoder: MageTextEncoder, tier: Option<Quant>) -> MageTextEncoder {
    if let Some(tier) = tier {
        encoder.quantize(tier.bits()).expect("quantize encoder");
    }
    encoder
}

fn condition(
    encoder: &MageTextEncoder,
    prompt: &str,
    window: Option<usize>,
) -> (Vec<f32>, usize, usize) {
    clear_cache();
    reset_peak_memory();
    let out = encoder
        .encode_with_window(&[prompt], PromptKind::Gen, window, &CancelFlag::default())
        .expect("conditioning");
    let txt = out
        .txt
        .as_dtype(mlx_rs::Dtype::Float32)
        .expect("f32 conditioning");
    mlx_rs::transforms::eval([&txt]).expect("evaluate conditioning");
    let peak = get_peak_memory();
    let values = txt.as_slice::<f32>().to_vec();
    (values, peak, out.seq_lens[0])
}

#[test]
#[ignore = "requires Apple MLX and MAGE_REQUEST_SCOPE_SNAPSHOT real weights"]
fn conditioning_is_exact_bounded_variable_length_and_materialization_is_load_bearing() {
    let tier = quant();
    let dir = text_encoder_dir(tier);

    let resident = prepare(load_dir(&dir).expect("resident encoder"), tier);
    assert!(!resident.is_streamable());
    let (expected, resident_peak, _) = condition(&resident, PROMPT, None);
    drop(resident);
    clear_cache();

    let streamed = prepare(
        mlx_gen_mage::text_encoder::load_dir_streamable(&dir).expect("streamed encoder"),
        tier,
    );
    assert!(streamed.is_streamable());
    let mut peaks = Vec::new();
    for window in [36usize, 8, 4, 2, 1] {
        let (got, peak, _) = condition(&streamed, PROMPT, Some(window));
        assert_eq!(got, expected, "window={window} changed conditioning");
        peaks.push((window, peak));
    }
    println!(
        "SC-15800 conditioning tier={tier:?}: resident {:.3} GiB",
        gib(resident_peak)
    );
    for (window, peak) in &peaks {
        println!("  window={window:<2} {:.3} GiB", gib(*peak));
    }
    let tight = peaks.last().unwrap().1;
    assert!(
        tight < resident_peak * 4 / 5,
        "window=1 did not materially bound conditioning: {:.3} vs {:.3} GiB",
        gib(tight),
        gib(resident_peak)
    );
    for pair in peaks.windows(2) {
        let ((loose_window, loose_peak), (tight_window, tight_peak)) = (pair[0], pair[1]);
        assert!(
            tight_peak <= loose_peak + loose_peak / 20,
            "window={tight_window} peaked at {:.3} GiB above window={loose_window}'s {:.3} GiB",
            gib(tight_peak),
            gib(loose_peak)
        );
    }

    // Mage does not pad: prove the sweep reaches materially different token lengths and record the
    // corresponding phase peaks at the production candidate.
    let long_prompt = std::iter::repeat_n(PROMPT, 12)
        .collect::<Vec<_>>()
        .join(" ");
    let prompts = ["red cube", PROMPT, long_prompt.as_str()];
    let mut lengths = Vec::new();
    for prompt in prompts {
        let (_, peak, tokens) = condition(&streamed, prompt, Some(1));
        println!("  prompt_tokens={tokens:<4} window=1 {:.3} GiB", gib(peak));
        lengths.push(tokens);
    }
    assert!(lengths.windows(2).all(|pair| pair[0] < pair[1]));
    drop(streamed);
    clear_cache();

    // Mutation control: preserve identical arithmetic while omitting the carried-state evaluation
    // at each window boundary. The peak must rebound because the lazy graph then retains prior
    // windows' weights, proving the production materialization owns the cross-window bound.
    let cfg = QwenVlTextConfig::mage_flow();
    let weights = Weights::from_dir(&dir).expect("weights");
    let mut mutated_lm = Qwen3VlTextEncoder::from_streamable_source_without_carry_materialization(
        &weights,
        WeightsSource::Dir(dir.clone()),
        LM_PREFIX,
        &cfg,
        TE_RMS_NORM_EPS,
        TE_ROPE_THETA,
    )
    .expect("mutation encoder");
    if let Some(tier) = tier {
        mutated_lm.quantize(tier.bits()).expect("quantize mutation");
    }
    let tokenizer = mlx_gen_mage::text_encoder::load_tokenizer_dir(&dir).expect("tokenizer");
    let mutated = MageTextEncoder::new(tokenizer, mutated_lm);
    drop(weights);
    let (mutated_values, mutated_peak, _) = condition(&mutated, PROMPT, Some(1));
    assert_eq!(mutated_values, expected, "mutation changed arithmetic");
    assert!(
        mutated_peak > tight + tight / 5,
        "lazy-carry mutation did not rebound: {:.3} vs production {:.3} GiB",
        gib(mutated_peak),
        gib(tight)
    );
}

fn spec(policy: OffloadPolicy, shape: LoadShape, tier: Option<Quant>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_offload_policy(policy);
    spec.load_shape = shape;
    spec.quantize = tier;
    if tier.is_some() {
        spec = spec.with_component(
            COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(text_encoder_dir(tier)),
        );
    }
    spec
}

fn render(generator: &dyn Generator, window: Option<u32>) -> (Vec<u8>, usize) {
    clear_cache();
    reset_peak_memory();
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: Some("artifact, blur".into()),
        width: 512,
        height: 512,
        count: 1,
        steps: Some(1),
        guidance: Some(5.0),
        seed: Some(15800),
        memory: window.map(|window| GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(window),
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        }),
        ..Default::default()
    };
    let GenerationOutput::Images(images) =
        generator.generate(&request, &mut |_| {}).expect("render")
    else {
        panic!("expected image output")
    };
    (images.into_iter().next().unwrap().pixels, get_peak_memory())
}

#[test]
#[ignore = "requires Apple MLX and MAGE_REQUEST_SCOPE_SNAPSHOT real weights"]
fn window_moves_the_complete_request_peak() {
    let tier = quant();
    let resident = (REGISTRATION.load)(&spec(
        OffloadPolicy::Resident,
        LoadShape::EagerMaterialization,
        tier,
    ))
    .expect("resident generator");
    let (expected, resident_peak) = render(resident.as_ref(), None);
    drop(resident);
    clear_cache();

    let streamed = (REGISTRATION.load)(&spec(
        OffloadPolicy::Sequential,
        LoadShape::DeferredMaterialization,
        tier,
    ))
    .expect("streamed generator");
    let mut peaks = Vec::new();
    for window in [36u32, 8, 4, 2, 1] {
        let (got, peak) = render(streamed.as_ref(), Some(window));
        assert_eq!(got, expected, "window={window} changed the image");
        peaks.push((window, peak));
    }
    println!(
        "SC-15800 request tier={tier:?}: resident {:.3} GiB",
        gib(resident_peak)
    );
    for (window, peak) in &peaks {
        println!("  window={window:<2} {:.3} GiB", gib(*peak));
    }
    let tight = peaks.last().unwrap().1;
    assert!(
        tight < resident_peak * 9 / 10,
        "window=1 did not move the request peak: {:.3} vs {:.3} GiB",
        gib(tight),
        gib(resident_peak)
    );
}
