//! The native model against upstream's own modules on a tiny model with the real architecture
//! (`testdata/tiny`, from `native_parity.py tiny`: seeded weights, MERT parent + rank-4 adapters
//! merged the upstream way, 2 Conformer blocks, a 2-layer BART decoder, the real tokenizer and
//! grammar over a 1 s window).
//!
//! Hidden states are compared by max-abs and relative error (max-abs over the reference's max-abs),
//! never by cosine alone; the greedy tokens must match exactly.

use std::collections::HashMap;

use candle_audio::candle_core::{Device, Tensor};

use super::*;
use crate::tokenizer::FULL_TASK_PROMPTS;

pub(crate) fn tiny_reference() -> serde_json::Value {
    serde_json::from_str(include_str!("../../testdata/tiny/reference.json")).unwrap()
}

fn tensors(bytes: &[u8]) -> HashMap<String, Tensor> {
    candle_audio::candle_core::safetensors::load_buffer(bytes, &Device::Cpu).unwrap()
}

pub(crate) fn tiny_model() -> SheetSage2Model {
    let reference = tiny_reference();
    let config = &reference["sheetsage2_config"];
    SheetSage2Model::load(
        config,
        &config["backbone_config"],
        Weights::from_map(
            "tiny head",
            tensors(include_bytes!(
                "../../testdata/tiny/sheetsage2_head.safetensors"
            )),
        ),
        Weights::from_map(
            "tiny parent",
            tensors(include_bytes!(
                "../../testdata/tiny/mert_parent.safetensors"
            )),
        ),
        &Device::Cpu,
    )
    .unwrap()
}

/// `(max_abs, relative)` of `ours − theirs`, relative to the reference's max-abs.
fn error(ours: &Tensor, theirs: &Tensor) -> (f32, f32) {
    assert_eq!(ours.dims(), theirs.dims());
    let a = ours.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = theirs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_abs, max_abs / scale.max(f32::MIN_POSITIVE))
}

/// Every encoder state and the decoder memory match upstream within measured float32 tolerances.
///
/// Measured on aarch64 (all ≤ ~3e-6 relative); the thresholds are 1e-4 relative. Mutations that must
/// fail: skip the LoRA merge (relative error ≈ 1), drop the GRN's `+ x` residual, use natural log
/// instead of log10 in the dB step, or reverse `rotate_half`.
#[test]
fn encoder_states_match_upstream() {
    let _serial = crate::test_lock();
    let model = tiny_model();
    let reference = tensors(include_bytes!("../../testdata/tiny/reference.safetensors"));
    let waveform = reference["input.waveform"].to_vec1::<f32>().unwrap();
    let features = model.audio_features(&waveform).unwrap();
    let get = |k: &str| reference[k].unsqueeze(0).unwrap();
    let mut checks = vec![
        ("mel", error(&features.mel, &get("output.mel"))),
        (
            "input_hidden",
            error(&features.input_hidden, &get("output.input_hidden")),
        ),
        ("mixed", error(&features.mixed, &get("output.mixed"))),
        ("memory", error(&features.memory, &get("output.memory"))),
    ];
    for (i, block) in features.blocks.iter().enumerate() {
        checks.push((
            ["block.0", "block.1"][i],
            error(block, &get(&format!("output.block.{i}"))),
        ));
    }
    for (name, (max_abs, relative)) in &checks {
        println!("{name}: max_abs {max_abs:.3e} relative {relative:.3e}");
        assert!(*relative < 1e-4, "{name}: relative error {relative:.3e}");
    }
    // The streamed production path produces the same memory.
    let memory = model.encode(&waveform).unwrap();
    let (max_abs, _) = error(&memory, &features.memory);
    assert_eq!(
        max_abs, 0.0,
        "streamed layer mix must equal the materialized one"
    );
}

/// Greedy decoding under the grammar reproduces upstream's tokens exactly, and the first step's raw
/// logits agree. The fixture's minimum greedy margin (> 0.02) shows the comparison is not a string
/// of near-ties.
///
/// Mutations that must fail: drop the BART position offset of 2, skip `layernorm_embedding`, or use
/// the grammar-unmasked argmax.
#[test]
fn greedy_tokens_match_upstream_exactly() {
    let _serial = crate::test_lock();
    let model = tiny_model();
    let reference = tiny_reference();
    let tensors = tensors(include_bytes!("../../testdata/tiny/reference.safetensors"));
    let waveform = tensors["input.waveform"].to_vec1::<f32>().unwrap();
    let memory = model.encode(&waveform).unwrap();
    let prefix = model.tokenizer().prompt_prefix(&FULL_TASK_PROMPTS).unwrap();
    let mut first_logits = None;
    let tokens = model
        .generate(
            &memory,
            &prefix,
            reference["max_sequence_length"].as_u64().unwrap() as usize,
            Some(reference["stop_time_seconds"].as_f64().unwrap()),
            |step| {
                if first_logits.is_none() {
                    first_logits = Some(step.logits.to_vec());
                }
                Ok(())
            },
        )
        .unwrap();
    let expected: Vec<u32> = reference["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(tokens, expected);
    assert!(reference["greedy_min_margin"].as_f64().unwrap() > 0.02);
    let ours = Tensor::new(first_logits.unwrap(), &Device::Cpu).unwrap();
    let (max_abs, relative) = error(&ours, &tensors["output.first_logits"]);
    println!("first logits: max_abs {max_abs:.3e} relative {relative:.3e}");
    assert!(relative < 1e-4);
}

/// A checkpoint that is not float32, has an extra tensor, or pairs with a different parent
/// architecture is refused at load, never silently accepted.
#[test]
fn load_refuses_mismatched_checkpoints() {
    let _serial = crate::test_lock();
    let reference = tiny_reference();
    let config = &reference["sheetsage2_config"];
    let head = || {
        tensors(include_bytes!(
            "../../testdata/tiny/sheetsage2_head.safetensors"
        ))
    };
    let parent = || {
        tensors(include_bytes!(
            "../../testdata/tiny/mert_parent.safetensors"
        ))
    };

    let mut extra = head();
    extra.insert(
        "stray".into(),
        Tensor::zeros(1, candle_audio::candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
    let err = SheetSage2Model::load(
        config,
        &config["backbone_config"],
        Weights::from_map("h", extra),
        Weights::from_map("p", parent()),
        &Device::Cpu,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("unexpected tensors"), "{err}");

    let mut missing = parent();
    missing.remove("layers.0.attn.query_proj.bias");
    let err = SheetSage2Model::load(
        config,
        &config["backbone_config"],
        Weights::from_map("h", head()),
        Weights::from_map("p", missing),
        &Device::Cpu,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("missing tensor"), "{err}");

    let mut other = config["backbone_config"].clone();
    other["rotary_embedding_base"] = serde_json::json!(5000);
    let err = SheetSage2Model::load(
        config,
        &other,
        Weights::from_map("h", head()),
        Weights::from_map("p", parent()),
        &Device::Cpu,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("architecture mismatch"), "{err}");

    let mut half = head();
    let w = half["layer_weight"]
        .to_dtype(candle_audio::candle_core::DType::F16)
        .unwrap();
    half.insert("layer_weight".into(), w);
    let err = SheetSage2Model::load(
        config,
        &config["backbone_config"],
        Weights::from_map("h", half),
        Weights::from_map("p", parent()),
        &Device::Cpu,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("float32"), "{err}");
}
