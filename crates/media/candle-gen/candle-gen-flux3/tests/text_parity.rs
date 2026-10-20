//! Qwen3-VL conditioning vs upstream `TextEncoder.encode` (real class, miniature
//! `Qwen3VLForConditionalGeneration` + WordLevel tokenizer carrying the PINNED chat template) —
//! the same fixture the MLX twin's `text_parity` reads: the single-string template tokenization,
//! the pad-to-multiple window with the clamp, the pad position ids, the padded attention and the
//! selected-layer stacking.
//!
//! Tolerances: the oracle ran the tower in bf16 (the release's `text_encoder.dtype`) over weights
//! pre-rounded to bf16, so its activations carry a bf16 rounding at every op of the 36-block
//! residual stream (ulp ≈ 0.03 at the stream's ≈ 6 peak), and no Candle backend rounds at the
//! same points as torch CPU: on CUDA the tower runs in bf16 through cuBLAS / fused kernels
//! (measured max 4.0–4.2 % of peak, mean 0.6–0.8 % across the four prompts on an RTX PRO 6000,
//! sc-26235); on Candle CPU (no half-precision GEMM) it runs in **f32** and the distance is the
//! oracle's own accumulated rounding, uncorrelated (measured max 2.7–6.7 %, mean 0.4–0.7 %). The
//! MLX twin's 4e-2 max bound on this fixture is therefore not the Candle bound: the bound here is
//! **8e-2 of peak on the max and 1.5e-2 on the mean** ([`MAX_TOL`], [`MEAN_TOL`]), held apart from
//! a structural error by [`stacking_the_wrong_layers_lands_far_outside_the_bound`]: shifting the
//! stacked layers by one measures 21 % max / 5.4 % mean of peak on the same fixture.
//!
//! Ids, masks and positions are exact on every backend.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen_flux3::Flux3TextEncoder;

use crate::common::{
    assert_close, cpu, device, fixture, host_f32, host_i32, tiny_config, tiny_text_encoder,
};

const PROMPTS: [(&str, &str); 4] = [
    ("short", "a red fox in the snow"),
    ("exact", "a red fox in the snow , a cat on a roof"),
    (
        "long",
        "a red fox in the snow , a cat on a roof with a hat , golden hour , blue sky and sun of the",
    ),
    ("empty", ""),
];

/// Candle tower (bf16 on CUDA, f32 on CPU) vs the bf16 torch-CPU oracle: max |Δ| of peak.
const MAX_TOL: f32 = 8e-2;
/// Candle tower vs the bf16 torch-CPU oracle: mean |Δ| of peak.
const MEAN_TOL: f32 = 1.5e-2;

fn encoder(device: &candle_gen::candle_core::Device) -> Flux3TextEncoder {
    let config = tiny_config();
    Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &config.text_encoder,
        config.dit.text_in_channels,
        device,
    )
    .expect("tiny Qwen3-VL snapshot loads")
}

/// The tower's bound against the bf16 oracle, `(max_tol, mean_tol)` — the same on every Candle
/// backend (see the module docs for the two measurements behind it).
fn tolerance(_te: &Flux3TextEncoder) -> (f32, f32) {
    (MAX_TOL, MEAN_TOL)
}

/// `assert_close` on the max plus a bound on the mean |Δ| (both relative to `max(1, peak)`).
fn assert_conditioning_close(
    name: &str,
    got: &Tensor,
    want: &Tensor,
    (max_tol, mean_tol): (f32, f32),
) {
    assert_close(name, got, want, max_tol);
    let (_, peak, mean) = crate::common::errors(got, want);
    assert!(
        mean <= mean_tol * peak.max(1.0),
        "{name}: mean|Δ|={mean:.3e} exceeds {:.3e} (tol {mean_tol:.0e} × peak {peak:.3e})",
        mean_tol * peak.max(1.0)
    );
}

fn check_windows_positions_and_stacked_states(te: &Flux3TextEncoder) {
    let tol = tolerance(te);
    let golden = fixture("flux3_text_golden.safetensors");
    assert_eq!(te.pad_id(), host_i32(golden.require("pad_id"))[0]);
    let mut lengths = Vec::new();
    for (name, prompt) in PROMPTS {
        let out = te.encode(prompt).unwrap();
        let w = &out.window;
        assert_eq!(
            w.input_ids,
            host_i32(golden.require(&format!("{name}/input_ids"))),
            "{name}: input ids"
        );
        assert_eq!(
            w.mask,
            host_i32(golden.require(&format!("{name}/mask"))),
            "{name}: mask"
        );
        assert_eq!(
            w.positions,
            host_i32(golden.require(&format!("{name}/positions"))),
            "{name}: positions"
        );
        let want: &Tensor = golden.require(&format!("{name}/tokens"));
        assert_eq!(out.tokens.dims(), want.dims(), "{name}: conditioning shape");
        assert_eq!(out.tokens.dtype(), DType::BF16);
        assert_conditioning_close(&format!("{name}/tokens"), &out.tokens, want, tol);
        let ids: Vec<f32> = out.ids.iter().flatten().copied().collect();
        assert_eq!(
            ids,
            host_f32(golden.require(&format!("{name}/ids"))),
            "{name}: ids"
        );
        lengths.push((name, w.real, w.length(), w.truncated_tokens));
    }
    // The window law under test: pad to the multiple (8 here), clamp at the max (24 here).
    assert_eq!(
        lengths,
        [
            ("short", 11, 16, 0),
            ("exact", 17, 24, 0),
            ("long", 24, 24, 6),
            ("empty", 5, 8, 0),
        ]
    );
}

#[test]
fn windows_positions_and_stacked_states_match_upstream() {
    let te = encoder(&cpu());
    assert_eq!(te.dtype(), DType::F32, "the CPU tower runs in f32");
    check_windows_positions_and_stacked_states(&te);
}

/// The same gate on the build's device: under `--features cuda` the tower runs the real bf16
/// kernels (on a CPU build this is the test above again).
#[test]
fn windows_positions_and_stacked_states_match_upstream_on_device() {
    let device = device();
    let te = encoder(&device);
    if !device.is_cpu() {
        assert_eq!(te.dtype(), DType::BF16, "the GPU tower runs in bf16");
    }
    check_windows_positions_and_stacked_states(&te);
}

/// The pad rows are real conditioning (the pad token's state at position 1 over the real keys),
/// identical to each other, and never zero — upstream feeds them to the DiT as text tokens.
///
/// *Mutation that reds this:* zeroing or dropping the pad rows, or giving the pads their
/// sequential positions (the rows then differ from each other).
#[test]
fn pad_rows_are_identical_states_not_zeros() {
    let te = encoder(&cpu());
    let out = te.encode("a red fox in the snow").unwrap();
    let real = out.window.real;
    let length = out.window.length();
    assert!(
        length > real + 1,
        "the short prompt pads by more than one row"
    );
    let first = host_f32(&out.tokens.narrow(0, real, 1).unwrap());
    assert!(first.iter().any(|v| *v != 0.0), "pad rows carry state");
    for p in real + 1..length {
        assert_eq!(
            host_f32(&out.tokens.narrow(0, p, 1).unwrap()),
            first,
            "every pad row is the same state"
        );
    }
    // And the real rows are unaffected by the pads: they match the golden's real rows.
    let golden = fixture("flux3_text_golden.safetensors");
    let want = golden.require("short/tokens");
    assert_conditioning_close(
        "short/real rows",
        &out.tokens.narrow(0, 0, real).unwrap(),
        &want.narrow(0, 0, real).unwrap(),
        tolerance(&te),
    );
    // The window IS what the tower ran: the same prompt through a window with every pad given
    // its sequential position (the wrong law) must land visibly elsewhere on the pad rows.
    let mut wrong = out.window.clone();
    for (p, pos) in wrong.positions.iter_mut().enumerate().skip(real) {
        *pos = p as i32;
    }
    let shifted = te.encode_window(wrong).unwrap();
    let (max_abs, peak, _) = crate::common::errors(
        &shifted.tokens.narrow(0, real, length - real).unwrap(),
        &out.tokens.narrow(0, real, length - real).unwrap(),
    );
    assert!(
        max_abs > 1e-2 * peak.max(1.0),
        "sequential pad positions must change the pad rows (max|Δ|={max_abs:.3e})"
    );
}

/// The evidence behind the CPU bound: stacking the wrong layers (every selected layer shifted by
/// one — the smallest structural error the layer selection can make) lands far outside it on
/// every prompt, on the max and on the mean alike, so the f32-vs-bf16 noise the bound admits
/// cannot hide a layer-selection, template or window error.
///
/// *Mutation that reds this:* loosening `MAX_TOL` / `MEAN_TOL` toward the
/// shifted-layer distance.
#[test]
fn stacking_the_wrong_layers_lands_far_outside_the_bound() {
    let config = tiny_config();
    let mut shifted_cfg = config.text_encoder.clone();
    shifted_cfg.layers = shifted_cfg.layers.iter().map(|l| l + 1).collect();
    let shifted = Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &shifted_cfg,
        config.dit.text_in_channels,
        &cpu(),
    )
    .unwrap();
    let golden = fixture("flux3_text_golden.safetensors");
    for (name, prompt) in PROMPTS {
        let want = golden.require(&format!("{name}/tokens"));
        let out = shifted.encode(prompt).unwrap();
        let (max_abs, peak, mean) = crate::common::errors(&out.tokens, want);
        let scale = peak.max(1.0);
        eprintln!(
            "{name}: layers shifted by one: max|Δ|={:.3} mean|Δ|={:.3} of peak",
            max_abs / scale,
            mean / scale
        );
        assert!(
            max_abs > 2.0 * MAX_TOL * scale,
            "{name}: a one-layer shift must exceed twice the CPU max bound"
        );
        assert!(
            mean > 3.0 * MEAN_TOL * scale,
            "{name}: a one-layer shift must exceed three times the CPU mean bound"
        );
    }
}
