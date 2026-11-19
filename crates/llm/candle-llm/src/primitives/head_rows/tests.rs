//! The head-row access (SW-7197): which access each head format gets (construction only), and —
//! in the explicit model-proof workflow — gathered scores equal to the whole head's columns for a
//! dense, a Q8_0, a Q4_K and a dequantized-forward Q8_0 head.

use super::*;
use crate::primitives::quant::QuantizedLinear;
use crate::primitives::sampler::{SplitMix64, TokenRng};
use candle_core::quantized::GgmlDType;

const VOCAB: usize = 40;
/// A multiple of 256, so the Q4_K head is whole super-blocks.
const HIDDEN: usize = 256;

fn weight(device: &Device) -> Tensor {
    let mut rng = SplitMix64::new(0x7197_0004);
    let data: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| rng.next_f32() - 0.5).collect();
    Tensor::from_vec(data, (VOCAB, HIDDEN), &Device::Cpu)
        .and_then(|t| t.to_device(device))
        .unwrap()
}

/// Every head format this crate loads an LM head in, on `device`, with its label there.
fn heads(device: &Device, dtype: DType) -> Vec<(&'static str, Projection, &'static str)> {
    let w = weight(device);
    let blocks = if device.is_cuda() { "indexed" } else { "host" };
    vec![
        (
            "dense",
            Projection::Dense(candle_nn::Linear::new(w.to_dtype(dtype).unwrap(), None)),
            "dense",
        ),
        (
            "q8_0",
            Projection::Quantized(QuantizedLinear::quantize(&w, GgmlDType::Q8_0, None).unwrap()),
            blocks,
        ),
        (
            "q4_k",
            Projection::Quantized(QuantizedLinear::quantize(&w, GgmlDType::Q4K, None).unwrap()),
            blocks,
        ),
        (
            "q8_0 dequantized forward",
            Projection::Quantized(
                QuantizedLinear::quantize_dequantized(&w, GgmlDType::Q8_0, None).unwrap(),
            ),
            blocks,
        ),
    ]
}

/// Each head format gets the row access its storage allows: dense rows gathered, GGML blocks read
/// on the host, and a biased head (no indexed form) runs whole — named. Only the CUDA indexed
/// table holds device bytes, and a whole-head fallback prices the vocabulary-wide logits the
/// gathered forms avoid.
#[test]
fn each_head_format_names_its_row_access() {
    for (name, head, label) in heads(&Device::Cpu, DType::F32) {
        let rows = HeadRows::new(&head).unwrap();
        assert_eq!(rows.label(), label, "{name}");
        assert!(rows.is_gathered(), "{name}");
        assert_eq!(rows.resident_bytes(), 0, "{name}");
    }
    let biased = Projection::Dense(candle_nn::Linear::new(
        weight(&Device::Cpu),
        Some(Tensor::zeros(VOCAB, DType::F32, &Device::Cpu).unwrap()),
    ));
    let full = HeadRows::new(&biased).unwrap();
    assert_eq!(full.label(), "bias");
    assert!(!full.is_gathered());

    // 15 draft rows of 16 senders against 16 candidates over the 27B's head.
    let (vocab, hidden) = (248_320u64, 5_120u64);
    let price = |rows: &HeadRows| rows.workspace_bytes(15, 16, 16, vocab, hidden, 2).unwrap();
    let gathered = price(&HeadRows::Dense(weight(&Device::Cpu)));
    assert_eq!(gathered, 15 * 16 * hidden * 4 + 15 * 16 * 16 * 6);
    assert_eq!(price(&full), 15 * 16 * vocab * 6 + 15 * 16 * 16 * 6);
    assert!(price(&full) > 50 * gathered);
}

/// The gathered scores are the whole head's columns of the asked ids, for every head format, on
/// the proof device (the CUDA indexed GEMV there; the host block gather on the CPU, bit for bit).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn gathered_head_rows_score_as_the_whole_head() {
    let (_guard, device) = crate::models::block_draft::tests::proof_device();
    let dtype = crate::device::compute_dtype(&device);
    let (groups, senders, k) = (3usize, 4usize, 5usize);
    let mut rng = SplitMix64::new(0x7197_0005);
    let rows: Vec<f32> = (0..groups * senders * HIDDEN)
        .map(|_| rng.next_f32() - 0.5)
        .collect();
    let rows = Tensor::from_vec(rows, (groups, senders, HIDDEN), &Device::Cpu)
        .and_then(|t| t.to_device(&device)?.to_dtype(dtype))
        .unwrap();
    // Ascending ids per group, as a proposal's candidates are; group 2 repeats group 0's.
    let ids: Vec<u32> = [[0u32, 7, 13, 21, 39], [2, 3, 4, 30, 31], [0, 7, 13, 21, 39]].concat();
    let ids = Tensor::from_vec(ids, (groups, k), &device).unwrap();
    let host = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let ids_host: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    for (name, head, label) in heads(&device, dtype) {
        let access = HeadRows::new(&head).unwrap();
        assert_eq!(access.label(), label, "{name}");
        let got = access.scores(&head, &rows, &ids).unwrap();
        assert_eq!(got.dims(), &[groups, senders, k], "{name}");
        let got = host(&got);
        let full = host(
            &head
                .forward(&rows.reshape((groups * senders, HIDDEN)).unwrap())
                .unwrap(),
        );
        let bit_exact = device.is_cpu() && label == "host";
        let mut worst = 0f32;
        for g in 0..groups {
            for s in 0..senders {
                for j in 0..k {
                    let id = ids_host[g * k + j] as usize;
                    let want = full[(g * senders + s) * VOCAB + id];
                    let have = got[(g * senders + s) * k + j];
                    if bit_exact {
                        assert_eq!(have, want, "{name}: ({g}, {s}, {j})");
                    }
                    worst = worst.max((have - want).abs() / want.abs().max(1.0));
                }
            }
        }
        let tol = if device.is_cpu() { 1e-4 } else { 3e-2 };
        assert!(worst <= tol, "{name}: worst relative difference {worst}");
    }
    let fallback = HeadRows::Full("test");
    let (_, head, _) = heads(&device, dtype).remove(1);
    let gathered = host(
        &HeadRows::new(&head)
            .unwrap()
            .scores(&head, &rows, &ids)
            .unwrap(),
    );
    let whole = host(&fallback.scores(&head, &rows, &ids).unwrap());
    let worst = gathered
        .iter()
        .zip(&whole)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(worst <= 3e-2, "the whole-head fallback moved {worst}");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::head_rows::tests::gathered_head_rows_score_as_the_whole_head");
}
