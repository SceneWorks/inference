//! sc-18316 / sc-18318 reachability, driven through a **real MMDiT block forward**.
//!
//! Both stories shipped their optimization behind `diagnostics::toggle_requested`, which is false
//! unless a diagnostic scope names the toggle — and the only non-test installer was the benchmark
//! binary. Every production render therefore re-traced and re-leased a fresh `mx.compile` handle on
//! every block of every step, and ran the composed epilogue instead of the native one. The two
//! capabilities are now on by default (`mlx_gen::capability`), and this proves it where it matters:
//! on [`QwenTransformerBlock::forward`], the 60-times-per-step production hot path, with synthetic
//! weights and the same [`CompileGlueGuard`] the production denoise loop binds — and with **no**
//! toggle-selecting scope of any kind.
//!
//! One forward reaches four retained sites and one P3 operation:
//!
//! | site | per forward |
//! |---|---|
//! | `qwen_image::block::modulate` | 4 (img/txt × mod1/mod2) |
//! | `qwen_image::block::gated` | 4 |
//! | `qwen_image::attention::rope_rotate` | 4 (img/txt × q/k) |
//! | `qwen_image::feed_forward::gelu_ffn` | 2 |
//! | `silu` on the timestep embedding | 1 (P3 `EXACT_SILU`) |

use std::collections::HashMap;

use mlx_gen::capability::{CapabilityOptOut, PipelineCapability};
use mlx_gen::diagnostics::{
    self, CompileDisposition, DiagnosticCounter, DiagnosticReport, ToggleDisposition,
    EXACT_EPILOGUES, RETAINED_COMPILATION,
};
use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::transformer::{CompileGlueGuard, QwenTransformerBlock};
use mlx_rs::Array;

const RETAINED_SITES: [&str; 4] = [
    "qwen_image::block::modulate",
    "qwen_image::block::gated",
    "qwen_image::attention::rope_rotate",
    "qwen_image::feed_forward::gelu_ffn",
];

/// num_heads · head_dim, and the block's hidden width.
const DIM: i32 = 8;
const HEADS: i32 = 2;
const HEAD_DIM: i32 = 4;
const IMG_SEQ: i32 = 4;
const TXT_SEQ: i32 = 3;

fn patterned(shape: &[i32], phase: f32) -> Array {
    let len = shape.iter().map(|dim| *dim as usize).product::<usize>();
    let values: Vec<f32> = (0..len)
        .map(|index| (index as f32 * 0.037 + phase).sin() * 0.5)
        .collect();
    Array::from_slice(&values, shape)
}

fn push_linear<'a>(
    tensors: &mut Vec<(&'a str, &'a Array)>,
    prefix: &str,
    weight: &'a Array,
    bias: &'a Array,
) {
    tensors.push((
        Box::leak(format!("{prefix}.weight").into_boxed_str()),
        weight,
    ));
    tensors.push((Box::leak(format!("{prefix}.bias").into_boxed_str()), bias));
}

/// One real block built from a synthetic safetensors file — no weights, no accelerator fixture.
fn tiny_block(tmp: &tempfile::TempDir) -> QwenTransformerBlock {
    let w8 = patterned(&[DIM, DIM], 0.1);
    let b8 = patterned(&[DIM], 0.2);
    let norm = patterned(&[HEAD_DIM], 0.3);
    let w48 = patterned(&[6 * DIM, DIM], 0.4);
    let b48 = patterned(&[6 * DIM], 0.5);
    let w_in = patterned(&[2 * DIM, DIM], 0.6);
    let b_in = patterned(&[2 * DIM], 0.7);
    let w_out = patterned(&[DIM, 2 * DIM], 0.8);

    let path = tmp.path().join("block.safetensors");
    let mut tensors: Vec<(&str, &Array)> = Vec::new();
    push_linear(&mut tensors, "img_mod_linear", &w48, &b48);
    push_linear(&mut tensors, "txt_mod_linear", &w48, &b48);
    for projection in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.attn_to_out.0",
        "attn.to_add_out",
    ] {
        push_linear(&mut tensors, projection, &w8, &b8);
    }
    for qk_norm in [
        "attn.norm_q",
        "attn.norm_k",
        "attn.norm_added_q",
        "attn.norm_added_k",
    ] {
        tensors.push((
            Box::leak(format!("{qk_norm}.weight").into_boxed_str()),
            &norm,
        ));
    }
    for stream in ["img_ff", "txt_ff"] {
        push_linear(&mut tensors, &format!("{stream}.mlp_in"), &w_in, &b_in);
        push_linear(&mut tensors, &format!("{stream}.mlp_out"), &w_out, &b8);
    }
    Array::save_safetensors(tensors, None as Option<&HashMap<String, String>>, &path).unwrap();
    let weights = Weights::from_file(&path).unwrap();
    QwenTransformerBlock::from_weights(&weights, "", HEADS, HEAD_DIM).unwrap()
}

struct Inputs {
    hidden: Array,
    encoder_hidden: Array,
    temb: Array,
    img_cos: Array,
    img_sin: Array,
    txt_cos: Array,
    txt_sin: Array,
}

impl Inputs {
    fn new() -> Self {
        let half = HEAD_DIM / 2;
        Self {
            hidden: patterned(&[1, IMG_SEQ, DIM], 0.0),
            encoder_hidden: patterned(&[1, TXT_SEQ, DIM], 1.0),
            temb: patterned(&[1, DIM], 2.0),
            img_cos: patterned(&[IMG_SEQ, half], 3.0),
            img_sin: patterned(&[IMG_SEQ, half], 4.0),
            txt_cos: patterned(&[TXT_SEQ, half], 5.0),
            txt_sin: patterned(&[TXT_SEQ, half], 6.0),
        }
    }
}

/// Exactly what a production step does: the denoise loop's compile-glue guard and nothing else.
fn forward(block: &QwenTransformerBlock, inputs: &Inputs) -> (Array, Array) {
    let (text, image) = block
        .forward(
            &inputs.hidden,
            &inputs.encoder_hidden,
            &inputs.temb,
            &inputs.img_cos,
            &inputs.img_sin,
            &inputs.txt_cos,
            &inputs.txt_sin,
            None,
            None,
        )
        .unwrap();
    text.eval().unwrap();
    image.eval().unwrap();
    (text, image)
}

fn compile_count(report: &DiagnosticReport, site: &str, disposition: CompileDisposition) -> u64 {
    report
        .counters
        .iter()
        .filter_map(|counter| match counter {
            DiagnosticCounter::Compile {
                site: recorded,
                disposition: recorded_disposition,
                count,
            } if *recorded == site && *recorded_disposition == disposition => Some(*count),
            _ => None,
        })
        .sum()
}

fn assert_bit_exact(actual: &Array, expected: &Array, context: &str) {
    assert!(
        mlx_rs::ops::array_eq(actual, expected, false)
            .unwrap()
            .item::<bool>(),
        "{context} diverged from the pre-promotion baseline"
    );
}

/// A production block forward takes the retained and fused arms, and returns identical tensors.
#[test]
fn a_production_block_forward_reaches_retention_and_exact_epilogues() {
    let tmp = tempfile::tempdir().unwrap();
    let block = tiny_block(&tmp);
    let inputs = Inputs::new();

    // Pre-promotion baseline: one-shot compilation and composed epilogues, reachable now only
    // through the explicit opt-outs. This is the numeric reference the default path must match.
    let baseline = {
        let _glue = CompileGlueGuard::enable();
        let _retained = CapabilityOptOut::install(PipelineCapability::RetainedCompilation);
        let _epilogues = CapabilityOptOut::install(PipelineCapability::ExactEpilogues);
        let baseline_scope =
            diagnostics::begin_observed_request("qwen-block-baseline", "qwen_image").unwrap();
        let outputs = forward(&block, &inputs);
        let report = baseline_scope.finish();
        // The opted-out arm is genuinely the old one: one-shot per call, and no P3 admission.
        for site in RETAINED_SITES {
            assert!(
                compile_count(&report, site, CompileDisposition::OneShot) > 0,
                "the opt-out must restore the one-shot arm at {site}"
            );
            assert_eq!(
                compile_count(&report, site, CompileDisposition::RetainedHit),
                0
            );
            assert_eq!(
                compile_count(&report, site, CompileDisposition::RetainedMiss),
                0
            );
        }
        assert!(report.counters.contains(&DiagnosticCounter::Toggle {
            toggle: EXACT_EPILOGUES,
            disposition: ToggleDisposition::Unavailable,
            count: 1,
        }));
        outputs
    };

    // Production: the denoise loop's guard, no capability override, no opt-out. Clear MLX's compiler
    // cache first so the first observed call must report a real miss rather than inheriting the
    // baseline's graphs — the same cold-start the benchmark runner enforces per request.
    mlx_rs::transforms::compile::clear_cache();
    let _glue = CompileGlueGuard::enable();
    let scope = diagnostics::begin_observed_request("qwen-block-production", "qwen_image").unwrap();
    let first = forward(&block, &inputs);
    let second = forward(&block, &inputs);
    let report = scope.finish();

    for site in RETAINED_SITES {
        assert_eq!(
            compile_count(&report, site, CompileDisposition::OneShot),
            0,
            "a production forward must not construct a per-call handle at {site}"
        );
        assert_eq!(
            compile_count(&report, site, CompileDisposition::RetainedMiss),
            1,
            "{site} must compile exactly once across two production forwards"
        );
        assert!(
            compile_count(&report, site, CompileDisposition::RetainedHit) > 0,
            "{site} must reuse its retained graph on the second production forward"
        );
    }
    assert!(
        !report
            .counters
            .iter()
            .any(|counter| matches!(counter, DiagnosticCounter::Fallback { .. })),
        "a production forward must not report a fallback: {:?}",
        report.counters
    );
    assert!(
        report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                ..
            }
        )),
        "a production forward must carry the P1 activation receipt"
    );
    assert!(
        report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::ExactEpilogue {
                operation: "silu",
                disposition: ToggleDisposition::Applied,
                reason: None,
                count: 2,
            }
        )),
        "the block's timestep-embedding SiLU must take the native epilogue: {:?}",
        report.counters
    );
    assert!(report.counters.contains(&DiagnosticCounter::Toggle {
        toggle: EXACT_EPILOGUES,
        disposition: ToggleDisposition::Applied,
        count: 1,
    }));

    // Numeric safety: retention is compiled-graph reuse and the P3 epilogues are bit-exact, so both
    // production forwards must equal the pre-promotion baseline exactly.
    for (index, outputs) in [&first, &second].into_iter().enumerate() {
        assert_bit_exact(
            &outputs.0,
            &baseline.0,
            &format!("encoder_hidden_states, forward {index}"),
        );
        assert_bit_exact(
            &outputs.1,
            &baseline.1,
            &format!("hidden_states, forward {index}"),
        );
    }
}
