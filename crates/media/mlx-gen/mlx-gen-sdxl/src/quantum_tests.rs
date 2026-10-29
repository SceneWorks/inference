//! sc-26270: SDXL iOS bounded-quantum guards, exercised on tiny random-weight models on MLX's CPU
//! device inside a child process that makes any Metal use fatal.
//!
//! For every guarded path family (U-Net plain / IP / ControlNet / rung-4 streamed, CLIP text, CLIP
//! vision + IP Resampler, VAE decode dense / evaluated / tiled, VAE encode, component loads) the
//! tests prove: guarded execution is byte-identical to the unguarded host path; every quantum is
//! checked before it is synchronized, quanta never interleave, and there is exactly one checked
//! quantum per block / layer / tile of the fixture; a refusal happens before the refused quantum
//! is constructed (nothing after it is checked or synchronized, and the typed record names it);
//! and the scratch is derived from the loaded tensors and grows with the request geometry.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::ios_quantum::{
    take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
    PROCESS_RESERVE_BYTES,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::{random, Array, Dtype};

use crate::config::{ClipActivation, ClipTextConfig, UNetConfig};
use crate::quantum::Backend;
use crate::quantum::*;
use crate::unet::UNet2DConditionModel;

/// Headroom no tiny fixture can exhaust.
const PLENTY: u64 = 1 << 34;

/// Re-run test `name` in a child process pinned to the CPU device with Metal made fatal. Returns
/// `true` inside the child (run the body), `false` in the parent (the child already asserted).
fn cpu_child(name: &str) -> bool {
    const CHILD: &str = "SDXL_IOS_QUANTUM_CPU_CHILD";
    if std::env::var_os(CHILD).is_some() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu()
        );
        // Pin the native stream too, before any fixture is built (not only inside diagnostic
        // scopes). The handle is retained by MLX's default-stream table.
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("quantum_tests::{name}"),
            "--nocapture",
            "--include-ignored",
        ])
        .env(CHILD, "1")
        .env("IOS_CAPTURE_IN_CHILD", "1")
        .status()
        .unwrap();
    assert!(status.success(), "CPU child for {name} failed");
    false
}

/// Completion is armed only at the actual end of each admitted test body. Because this guard is
/// bound before fixture locals, it drops last: after fixtures/Arrays, then CPU stream completion.
/// Panic or an unarmed early return never produces a success marker.
struct CpuBodyCompletion {
    body_completed: bool,
}

impl Drop for CpuBodyCompletion {
    fn drop(&mut self) {
        if self.body_completed && !std::thread::panicking() {
            assert_eq!(
                unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                0,
                "CPU body completion stream synchronization failed"
            );
            std::fs::write(
                std::env::var("SDXL_IOS_QUANTUM_CPU_BODY_DONE")
                    .expect("CPU body completion marker path"),
                b"ok",
            )
            .expect("write CPU normal-body completion marker");
        }
    }
}

/// Six admitted regression bodies opt into completion evidence without altering unrelated tests.
/// The existing helper installs its unchanged device/stream readbacks and fatal Metal trap
/// before this returns a guard and before any fixture can be constructed.
fn completed_cpu_child(name: &str) -> Option<CpuBodyCompletion> {
    if std::env::var_os("SDXL_IOS_QUANTUM_CPU_CHILD").is_some() {
        assert!(cpu_child(name));
        return Some(CpuBodyCompletion {
            body_completed: false,
        });
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("quantum_tests::{name}"),
            "--nocapture",
            "--include-ignored",
            "--test-threads=1",
        ])
        .env("SDXL_IOS_QUANTUM_CPU_CHILD", "1")
        .env("IOS_CAPTURE_IN_CHILD", "1")
        .env("SDXL_IOS_QUANTUM_CPU_BODY_DONE", &marker)
        .status()
        .unwrap();
    assert!(status.success(), "CPU child for {name} failed");
    assert_eq!(
        std::fs::read(&marker).expect("CPU child ran no completed normal body"),
        b"ok"
    );
    None
}

/// Cheap precision discriminator retained in ordinary CI after the full RGB matrix becomes
/// selected-only. Values deliberately distinguish F32 from a round-trip through F16.
#[test]
fn ios_boundary_preserves_actual_operand_dtype_and_values() {
    if !cpu_child("ios_boundary_preserves_actual_operand_dtype_and_values") {
        return;
    }
    for dtype in [Dtype::Float16, Dtype::Float32] {
        let source = Array::from_slice(&[0.1234567_f32, -0.9876543, 1.000123], &[1, 1, 1, 3])
            .as_dtype(dtype)
            .unwrap();
        let host = crate::pipeline::ios_unet_input(&source).unwrap();
        let bounded =
            with_cpu_diagnostic_headroom(PLENTY, || crate::pipeline::ios_unet_input(&source))
                .unwrap();
        assert_eq!(bounded.dtype(), dtype);
        assert_eq!(bits(&bounded), bits(&source));
        assert_eq!(bits(&host), bits(&source));
        if dtype == Dtype::Float32 {
            assert_ne!(
                bits(&source),
                bits(&source.as_dtype(Dtype::Float16).unwrap()),
                "fixture detects narrowing"
            );
        }
    }
}

/// Header-only target topology must equal the ACTUAL constructor's target surfaces, including
/// optional time projections/shortcuts. This is a native CPU construction test, not backing proof.
#[test]
fn configured_header_targets_match_actual_constructor_surfaces() {
    if !cpu_child("configured_header_targets_match_actual_constructor_surfaces") {
        return;
    }
    use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader};
    for optional in [true, false] {
        let cfg = crate::test_support::tiny_unet_config();
        let mut weights = crate::test_support::tiny_unet_weights_for(&cfg, 71).unwrap();
        if !optional {
            let keys: Vec<_> = weights
                .keys()
                .filter(|key| key.contains(".time_emb_proj.") || key.contains(".conv_shortcut."))
                .map(str::to_owned)
                .collect();
            for key in keys {
                weights.remove(&key);
            }
        }
        let headers: Vec<_> = weights
            .keys()
            .map(|name| {
                let array = weights.require(name).unwrap();
                let dtype = match array.dtype() {
                    Dtype::Float32 => Stored::F32,
                    Dtype::Float16 => Stored::F16,
                    Dtype::Bfloat16 => Stored::BF16,
                    other => panic!("unexpected tiny fixture dtype {other:?}"),
                };
                SafetensorsTensorHeader {
                    name: name.to_owned(),
                    dtype,
                    shape: array.shape().iter().map(|d| *d as usize).collect(),
                    data_bytes: array.size() as u64 * crate::quantum::dtype_bytes(array.dtype()),
                }
            })
            .collect();
        let actual = UNet2DConditionModel::from_weights(&weights, &cfg).unwrap();
        let (vendored, complete, convs) =
            crate::unet::adapter_target_paths_from_headers(&cfg, &headers);
        assert_eq!(
            vendored,
            actual.lora_target_paths(),
            "header/runtime vendored topology optional={optional}"
        );
        assert_eq!(
            complete,
            actual.lora_target_paths_complete(),
            "header/runtime complete topology optional={optional}"
        );
        assert_eq!(
            convs,
            actual.conv_target_paths(),
            "header/runtime convolution topology optional={optional}"
        );
    }
}

/// Safety is a hard lower bound; a conservative ratio above the reporting threshold is not a
/// safety failure (epic 16774). Keep the measured figures visible without weakening the bound.
fn assert_tracks_measured(label: &str, formula: u64, measured: u64, report_ratio: f64) {
    assert!(
        formula >= measured,
        "{label}: formula {formula} B is BELOW the measured peak {measured} B"
    );
    if formula as f64 > report_ratio * measured as f64 {
        eprintln!("OVERPRICE {label}: formula {formula} B / measured {measured} B exceeds {report_ratio}x (report only)");
    }
}

#[test]
fn measured_lower_bound_is_hard_and_overpricing_is_report_only() {
    assert_tracks_measured("report-only control", 151, 100, 1.5);
    assert!(std::panic::catch_unwind(|| {
        assert_tracks_measured("underpriced mutation", 99, 100, 1.5)
    })
    .is_err());
}

/// A quantum event with the (per-run) stream index dropped, so two runs compare.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ev {
    Checked(String, u64),
    Synced(String),
}

fn norm(events: &[QuantumEvent]) -> Vec<Ev> {
    events
        .iter()
        .map(|e| match e {
            QuantumEvent::Checked { label, scratch } => Ev::Checked(label.clone(), *scratch),
            QuantumEvent::Synchronized { label, .. } => Ev::Synced(label.clone()),
        })
        .collect()
}

/// Run `body` with the CPU fixture enabled at `headroom`, recording every quantum event.
fn observe<T>(headroom: u64, body: impl FnOnce() -> Result<T>) -> (Result<T>, Vec<Ev>) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = events.clone();
    let result = with_cpu_diagnostic_headroom(headroom, || {
        with_cpu_diagnostic_observer(move |e| sink.borrow_mut().push(e.clone()), body)
    });
    let events = norm(&events.borrow());
    (result, events)
}

fn bits(a: &Array) -> Vec<u32> {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

fn checked(events: &[Ev], label: &str) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            Ev::Checked(l, s) if l == label => Some(*s),
            _ => None,
        })
        .collect()
}

fn synced(events: &[Ev], label: &str) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Ev::Synced(l) if l == label))
        .count()
}

/// The per-core synchronization label of `mlx_gen::vae_tiling`'s tiled convolutions.
const TILED_CORE: &str = "tiled conv core";

/// Every synchronized quantum was checked first, quanta never interleave, every tile check lies
/// inside an open tiled VAE stage, and nothing is left open.
fn assert_well_ordered(events: &[Ev]) {
    assert!(!events.is_empty(), "the fixture must run guarded quanta");
    let mut open: Option<&str> = None;
    for (i, e) in events.iter().enumerate() {
        match e {
            Ev::Checked(label, _) if label == VAE_TILE || label == VAE_ENCODE_TILE => {
                let stage = if label == VAE_TILE {
                    VAE_TILED_STAGE
                } else {
                    VAE_ENCODE_TILED_STAGE
                };
                assert_eq!(
                    open,
                    Some(stage),
                    "tile check #{i} outside a tiled stage: {events:?}"
                )
            }
            Ev::Checked(label, _) => {
                assert!(
                    open.is_none(),
                    "{label} checked while {open:?} is still open: {events:?}"
                );
                open = Some(label);
            }
            // `mlx_gen::vae_tiling` synchronizes and drains each tiled core itself, inside the
            // open tiled stage, after that tile's own check.
            Ev::Synced(label) if label == TILED_CORE => assert!(
                matches!(open, Some(VAE_TILED_STAGE | VAE_ENCODE_TILED_STAGE)),
                "tiled core synchronized outside a tiled stage: {events:?}"
            ),
            Ev::Synced(label) => {
                assert_eq!(
                    open,
                    Some(label.as_str()),
                    "{label} synchronized without its own preceding check: {events:?}"
                );
                open = None;
            }
        }
    }
    assert!(open.is_none(), "{open:?} never synchronized");
}

/// Refusal before construction, twice: (1) with zero headroom the FIRST quantum refuses and no
/// event at all is observed; (2) at a headroom that admits every earlier quantum but not the first
/// one needing more scratch than all before it, the refusal names THAT quantum, carries its scratch,
/// and the observed events are exactly the full run's prefix up to (not including) its check — so
/// nothing after it was checked, built or synchronized. (2) needs such a quantum to exist (a path
/// whose first quantum is its widest has none); returns whether (2) ran.
fn assert_refusals<T>(full: &[Ev], run: impl Fn(u64) -> (Result<T>, Vec<Ev>)) -> bool {
    let _ = take_refusal();
    let first = full
        .iter()
        .find_map(|e| match e {
            Ev::Checked(l, _) => Some(l.clone()),
            Ev::Synced(_) => None,
        })
        .expect("a checked quantum");
    let (result, events) = run(0);
    assert!(result.is_err(), "zero headroom must refuse");
    assert_eq!(take_refusal().map(|r| r.label), Some(first));
    assert!(
        events.is_empty(),
        "nothing may run past a first-quantum refusal: {events:?}"
    );

    let checks: Vec<(usize, &str, u64)> = full
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Ev::Checked(l, s) => Some((i, l.as_str(), *s)),
            Ev::Synced(_) => None,
        })
        .collect();
    let Some((k, max_prev)) = (1..checks.len()).find_map(|k| {
        let max_prev = checks[..k].iter().map(|c| c.2).max().unwrap();
        (checks[k].2 > max_prev).then_some((k, max_prev))
    }) else {
        return false;
    };
    let (index, label, scratch) = checks[k];
    let (result, events) = run(PROCESS_RESERVE_BYTES + max_prev);
    assert!(result.is_err(), "{label} must refuse");
    let record = take_refusal().expect("a typed refusal record");
    assert_eq!(record.label, label);
    assert_eq!(record.scratch_bytes, scratch);
    assert_eq!(
        record.available_bytes,
        Some(PROCESS_RESERVE_BYTES + max_prev)
    );
    assert_eq!(
        events,
        full[..index],
        "the run must stop exactly before the refused {label}"
    );
    true
}

fn rnd(shape: &[i32], seed: u64, scale: f32) -> Array {
    let key = random::key(seed).unwrap();
    mlx_rs::ops::multiply(
        random::normal::<f32>(shape, None, None, Some(&key)).unwrap(),
        Array::from_f32(scale),
    )
    .unwrap()
}

/// `(latent, text memory, pooled, time ids)` for the tiny U-Net at `batch` rows of `h x w`.
fn unet_inputs(batch: i32, h: i32, w: i32) -> (Array, Array, Array, Array) {
    use crate::test_support::{TINY_CONTEXT_DIM, TINY_POOLED_DIM};
    let out = (
        rnd(&[batch, h, w, 4], 11, 1.0),
        rnd(&[batch, 5, TINY_CONTEXT_DIM], 12, 1.0),
        rnd(&[batch, TINY_POOLED_DIM], 13, 1.0),
        crate::pipeline::text_time_ids(batch),
    );
    mlx_rs::transforms::eval([&out.0, &out.1, &out.2, &out.3]).unwrap();
    out
}

fn resnet_count(cfg: &UNetConfig) -> usize {
    let l: i32 = cfg.layers_per_block.iter().sum();
    (l + 2 + l + cfg.num_blocks() as i32) as usize
}

/// The structural quantum counts of one U-Net forward.
fn assert_unet_counts(events: &[Ev], unet: &UNet2DConditionModel, cfg: &UNetConfig) {
    let depths = unet.transformer_stack_depths();
    let n = cfg.num_blocks();
    for (label, want) in [
        (UNET_STEM, 1),
        (UNET_RESNET, resnet_count(cfg)),
        (TRANSFORMER_PROJ_IN, depths.len()),
        (TRANSFORMER_BLOCK, depths.iter().sum()),
        (TRANSFORMER_PROJ_OUT, depths.len()),
        (UNET_DOWNSAMPLE, n - 1),
        (UNET_UPSAMPLE, n - 1),
        (UNET_CONV_OUT, 1),
    ] {
        assert_eq!(checked(events, label).len(), want, "{label} checks");
        assert_eq!(synced(events, label), want, "{label} synchronizations");
    }
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn unet_forward_is_bounded_identical_and_refuses_before_construction() {
    if !cpu_child("unet_forward_is_bounded_identical_and_refuses_before_construction") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let unet = crate::test_support::tiny_unet(7).unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let forward = || {
        (
            crate::test_device::record_model_inference_device(),
            unet.forward(&x, 500.0, &ctx, &pooled, &ids),
        )
            .1
    };
    let host = forward().unwrap();
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &unet, &cfg);
    // The stem is `conv_in` (output, bias add, CPU kernel transient) + the three latent-sized lazy
    // buffers of the forward's input, from the loaded weights; plus the time and add embedding
    // MLPs. Independent arithmetic for those MLPs: dense f32 weights over f32 inputs need no
    // operand casts, and each MLP holds three hidden planes (linear_1 out, SiLU, linear_2 in) and
    // one output plane of [batch, time_embed_dim] f32, plus four buffers' page slack.
    let mlp = 4
        * x.shape()[0] as u64
        * cfg.time_embed_dim() as u64
        * crate::quantum::dtype_bytes(Dtype::Float32)
        + 4 * crate::quantum::PAGE_BYTES;
    assert_eq!(
        checked(&events, UNET_STEM),
        [
            crate::unet::stem_bytes(&x, &ctx, unet.conv_in_weight(), None, 0, Backend::Cpu)
                + 2 * mlp
        ]
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::unet_forward_is_bounded_identical_and_refuses_before_construction");
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn unet_ip_forward_prices_the_ip_tokens_and_stays_identical() {
    if !cpu_child("unet_ip_forward_prices_the_ip_tokens_and_stays_identical") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let mut unet = crate::test_support::tiny_unet(7).unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let (_, plain) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            unet.forward(&x, 500.0, &ctx, &pooled, &ids),
        )
            .1
    });
    let layers: usize = unet.transformer_stack_depths().iter().sum();
    let pairs = (0..layers)
        .map(|i| {
            let c = 64; // every tiny-U-Net attention runs at the inner 64-channel level
            (
                rnd(
                    &[c, crate::test_support::TINY_CONTEXT_DIM],
                    100 + i as u64,
                    0.1,
                ),
                rnd(
                    &[c, crate::test_support::TINY_CONTEXT_DIM],
                    200 + i as u64,
                    0.1,
                ),
            )
        })
        .collect();
    unet.install_ip_adapter(pairs).unwrap();
    let tokens = rnd(&[1, 3, crate::test_support::TINY_CONTEXT_DIM], 31, 1.0);
    let forward = || {
        (
            crate::test_device::record_model_inference_device(),
            unet.forward_with_ip(&x, 500.0, &ctx, &pooled, &ids, (&tokens, 0.7)),
        )
            .1
    };
    let host = forward().unwrap();
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &unet, &cfg);
    // The IP branch prices its three extra keys and its output in the cross-attention phase: the
    // first block (16 tokens of the 4x4 grid) at 5 text + 3 IP keys with the IP buffers — never
    // below the plain block, whose widest phase may be its FFN.
    let (with_ip, without) = (
        checked(&events, TRANSFORMER_BLOCK),
        checked(&plain, TRANSFORMER_BLOCK),
    );
    assert_eq!(with_ip.len(), without.len());
    assert_eq!(
        with_ip[0],
        transformer_block_bytes(
            1,
            1,
            16,
            64,
            2,
            32,
            5 + 3,
            4 * 64,
            16,
            true,
            4,
            Backend::Cpu
        )
    );
    assert!(
        with_ip.iter().zip(&without).all(|(a, b)| a >= b),
        "{with_ip:?} vs {without:?}"
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::unet_ip_forward_prices_the_ip_tokens_and_stays_identical");
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn controlnet_branch_and_injection_are_bounded_and_identical() {
    if !cpu_child("controlnet_branch_and_injection_are_bounded_and_identical") {
        return;
    }
    let cfg = crate::test_support::tiny_control_unet_config();
    let unet = UNet2DConditionModel::from_weights(
        &crate::test_support::tiny_unet_weights_for(&cfg, 3).unwrap(),
        &cfg,
    )
    .unwrap();
    let control = crate::unet::ControlNet::from_weights(
        &crate::test_support::tiny_controlnet_weights(&cfg, 5).unwrap(),
        &cfg,
    )
    .unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let image = rnd(&[1, 64, 64, 3], 41, 0.5);
    image.eval().unwrap();
    let run = || -> Result<Vec<Array>> {
        let cond = control.embed_cond(&image)?;
        let residuals = (
            crate::test_device::record_model_inference_device(),
            control.forward(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8),
        )
            .1?;
        let eps = (
            crate::test_device::record_model_inference_device(),
            unet.forward_with_control(&x, 500.0, &ctx, &pooled, &ids, &residuals),
        )
            .1?;
        let mut out = residuals.down;
        out.push(residuals.mid);
        out.push(eps);
        Ok(out)
    };
    let host = run().unwrap();
    let (guarded, events) = observe(PLENTY, run);
    let guarded = guarded.unwrap();
    assert_eq!(host.len(), guarded.len());
    for (a, b) in host.iter().zip(&guarded) {
        assert_eq!(bits(a), bits(b));
    }
    assert_well_ordered(&events);
    for (label, want) in [
        (CONTROL_COND_CONV, 8),
        (CONTROL_STEM, 1),
        (CONTROL_ZERO_CONV, 10),
        (CONTROL_DOWN_INJECTION, 1),
        (CONTROL_MID_INJECTION, 1),
        (UNET_STEM, 1),
        (UNET_CONV_OUT, 1),
        // The ControlNet runs the encoder copy's down + mid resnets; the U-Net all of its own.
        (UNET_RESNET, 6 + 2 + resnet_count(&cfg)),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    assert_refusals(&events, |headroom| observe(headroom, run));
    // The full-resolution conditioning embedding is this chain's widest quantum, so prove the
    // mid-path refusal on the U-Net-with-control half (stem → … → injections) on its own.
    let cond = control.embed_cond(&image).unwrap();
    let residuals = (
        crate::test_device::record_model_inference_device(),
        control.forward(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8),
    )
        .1
        .unwrap();
    let inject = || {
        (
            crate::test_device::record_model_inference_device(),
            unet.forward_with_control(&x, 500.0, &ctx, &pooled, &ids, &residuals),
        )
            .1
    };
    let (_, inject_events) = observe(PLENTY, inject);
    assert_well_ordered(&inject_events);
    assert!(assert_refusals(&inject_events, |headroom| observe(
        headroom, inject
    )));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::controlnet_branch_and_injection_are_bounded_and_identical");
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn streamed_transformer_blocks_are_checked_before_materialization() {
    if !cpu_child("streamed_transformer_blocks_are_checked_before_materialization") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let weights = crate::test_support::tiny_unet_weights(7).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("unet.safetensors");
    let keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    let named: Vec<(&str, &Array)> = keys
        .iter()
        .map(|k| (k.as_str(), weights.get(k).unwrap()))
        .collect();
    Array::save_safetensors(named, None, &path).unwrap();
    let resident = UNet2DConditionModel::from_weights(&weights, &cfg).unwrap();
    let mut streamed =
        UNet2DConditionModel::from_weights(&Weights::from_file(&path).unwrap(), &cfg).unwrap();
    streamed.arm_block_streams(
        &mlx_gen::WeightsSource::File(path.clone()),
        Dtype::Float32,
        None,
    );
    assert!(streamed.can_stream_blocks());
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let cancel = mlx_gen::CancelFlag::default();
    let plan =
        crate::plan::SdxlForwardPlan::UNBOUNDED.with_window(Some(crate::plan::SdxlBlockWindow {
            size: 1,
            cancel: &cancel,
        }));
    let forward = || {
        (
            crate::test_device::record_model_inference_device(),
            streamed.forward_planned(&x, 500.0, &ctx, &pooled, &ids, None, None, plan),
        )
            .1
    };
    let host = forward().unwrap();
    assert_eq!(
        bits(&host),
        bits(
            &(
                crate::test_device::record_model_inference_device(),
                resident.forward(&x, 500.0, &ctx, &pooled, &ids)
            )
                .1
                .unwrap()
        )
    );
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &streamed, &cfg);
    // The streamed block's activations are its resident twin's (same loaded shapes, same
    // request); its quantum also reads the block from the snapshot (epic 16774): every stored
    // tensor of the block plus the GEGLU value/gate halves beside the projection they are sliced
    // from — read off the file, in forward order (down, mid, up).
    let (_, resident_events) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            resident.forward(&x, 500.0, &ctx, &pooled, &ids),
        )
            .1
    });
    let mut stacks: Vec<String> = Vec::new();
    let n = cfg.block_out_channels.len();
    for i in 0..n {
        if cfg.down_block_types[i].contains("CrossAttn") {
            for j in 0..cfg.layers_per_block[i] {
                stacks.push(format!("down_blocks.{i}.attentions.{j}"));
            }
        }
    }
    stacks.push("mid_block.attentions.0".into());
    for k in 0..n {
        let ci = n - 1 - k;
        if cfg.up_block_types[ci].contains("CrossAttn") {
            for j in 0..=cfg.layers_per_block[ci] {
                stacks.push(format!("up_blocks.{k}.attentions.{j}"));
            }
        }
    }
    let mut read = Vec::new();
    for stack in &stacks {
        let depth = weights
            .keys()
            .filter_map(|k| k.strip_prefix(&format!("{stack}.transformer_blocks.")))
            .filter_map(|rest| rest.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |d| d + 1);
        for b in 0..depth {
            let block = format!("{stack}.transformer_blocks.{b}.");
            read.push(
                weights
                    .keys()
                    .filter(|k| k.starts_with(&block))
                    .map(|k| {
                        let a = weights.get(k).unwrap();
                        let stored = mlx_gen::ios_quantum::allocated_bytes(bytes(a));
                        if k.contains(".ff.net.0.proj.") {
                            // The value/gate halves beside the projection, and (once, off the
                            // weight) the two `[hidden]` i32 row selections slicing them.
                            let selections = if k.ends_with(".weight") {
                                let half = a.shape()[0] as u64 / 2;
                                2 * mlx_gen::ios_quantum::allocated_bytes(4 * half)
                            } else {
                                0
                            };
                            2 * stored + selections
                        } else {
                            stored
                        }
                    })
                    .sum::<u64>(),
            );
        }
    }
    let resident_blocks = checked(&resident_events, TRANSFORMER_BLOCK);
    assert_eq!(resident_blocks.len(), read.len());
    assert_eq!(
        checked(&events, TRANSFORMER_BLOCK),
        resident_blocks
            .iter()
            .zip(&read)
            .map(|(activations, weights)| activations + weights)
            .collect::<Vec<_>>()
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::streamed_transformer_blocks_are_checked_before_materialization");
}

/// The block floor is `attention_block_bytes` over THIS block's loaded tensors (non-production
/// widths: 64 channels, 2 heads, a 4x GEGLU, 16-wide context) and the request (tokens, context
/// tokens, chunk rows) — and it grows with the latent grid.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn block_and_resnet_floors_come_from_loaded_tensors_and_grow_with_geometry() {
    if !cpu_child("block_and_resnet_floors_come_from_loaded_tensors_and_grow_with_geometry") {
        return;
    }
    let unet = crate::test_support::tiny_unet(7).unwrap();
    let observe_at = |h: i32, w: i32| {
        let (x, ctx, pooled, ids) = unet_inputs(1, h, w);
        observe(PLENTY, || {
            (
                crate::test_device::record_model_inference_device(),
                unet.forward(&x, 500.0, &ctx, &pooled, &ids),
            )
                .1
        })
        .1
    };
    let small = observe_at(8, 8);
    let large = observe_at(16, 16);
    // 64 channels, 2 heads of 32, a 4x GEGLU, 16 latent tokens at the 4x4 grid, 5 context tokens,
    // f32 activations, the CPU backend (unfused SDPA).
    let expected =
        transformer_block_bytes(1, 1, 16, 64, 2, 32, 5, 4 * 64, 16, false, 4, Backend::Cpu);
    // The first block is down_blocks.1's at the 4x4 grid (16 tokens) of an 8x8 latent.
    assert_eq!(checked(&small, TRANSFORMER_BLOCK)[0], expected);
    for label in [TRANSFORMER_BLOCK, UNET_RESNET, UNET_STEM, UNET_CONV_OUT] {
        let (s, l) = (checked(&small, label), checked(&large, label));
        assert!(
            s.iter().zip(&l).all(|(a, b)| b > a),
            "{label}: {s:?} vs {l:?}"
        );
    }
    // A resnet's floor uses its loaded conv channels: the first up resnet (4x4 grid) concatenates
    // the 64-channel mid output with a 64-channel skip.
    // (Weights shaped like the loaded `[out, 3, 3, in]` convs: the price reads only their shapes.)
    let conv1 = Array::zeros::<f32>(&[64, 3, 3, 128]).unwrap();
    let conv2 = Array::zeros::<f32>(&[64, 3, 3, 64]).unwrap();
    assert_eq!(
        checked(&small, UNET_RESNET)[4],
        resnet_bytes(&[1, 4, 4, 64], 4, 4, &conv1, &conv2, Backend::Cpu)
    );
    // A bounded attention plan is priced at its chunk's query rows: at a 16x16 latent the attention
    // grid is 64 tokens; 160 scores per chunk gives 16 cross-attention (5 keys x 2 heads) rows and
    // one self-attention row, so the block prices the wider score domain's chunk (16 rows) with
    // the chunk outputs and their concatenation.
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(160, false);
    assert_eq!(budget.query_block(1, 2, 64, 5), 16);
    let (x, ctx, pooled, ids) = unet_inputs(1, 16, 16);
    let plan = crate::plan::SdxlForwardPlan::with_attention(
        mlx_gen::attention::AttentionPlan::budgeted(budget),
    );
    let (bounded_out, chunked) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            unet.forward_planned(&x, 500.0, &ctx, &pooled, &ids, None, None, plan),
        )
            .1
    });
    bounded_out.unwrap();
    let (full, bounded) = (
        checked(&large, TRANSFORMER_BLOCK),
        checked(&chunked, TRANSFORMER_BLOCK),
    );
    assert_eq!(full.len(), bounded.len());
    assert_eq!(
        bounded[0],
        transformer_block_bytes(1, 1, 64, 64, 2, 32, 5, 4 * 64, 16, false, 4, Backend::Cpu)
    );
    assert_eq!(
        full[0],
        transformer_block_bytes(1, 1, 64, 64, 2, 32, 5, 4 * 64, 64, false, 4, Backend::Cpu)
    );

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::block_and_resnet_floors_come_from_loaded_tensors_and_grow_with_geometry");
}

// ---------------------------------------------------------------------------------------------
// CLIP text encoders.
// ---------------------------------------------------------------------------------------------

struct Gen {
    map: HashMap<String, Array>,
    seed: u64,
}

impl Gen {
    fn new(seed: u64) -> Self {
        Self {
            map: HashMap::new(),
            seed,
        }
    }
    fn put(&mut self, key: String, shape: &[i32], scale: f32) {
        self.seed += 1;
        self.map.insert(key, rnd(shape, self.seed, scale));
    }
    fn ones(&mut self, key: String, c: i32) {
        self.map.insert(key, Array::ones::<f32>(&[c]).unwrap());
    }
    fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) {
        self.put(
            format!("{p}.weight"),
            &[out, inp],
            (1.0 / inp as f32).sqrt(),
        );
        if bias {
            self.put(format!("{p}.bias"), &[out], 0.02);
        }
    }
    fn ln(&mut self, p: &str, c: i32) {
        self.ones(format!("{p}.weight"), c);
        self.put(format!("{p}.bias"), &[c], 0.02);
    }
    fn clip_layers(&mut self, p: &str, layers: i32, d: i32, mlp: i32) {
        for i in 0..layers {
            let l = format!("{p}.encoder.layers.{i}");
            self.ln(&format!("{l}.layer_norm1"), d);
            self.ln(&format!("{l}.layer_norm2"), d);
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                self.linear(&format!("{l}.self_attn.{proj}"), d, d, true);
            }
            self.linear(&format!("{l}.mlp.fc1"), mlp, d, true);
            self.linear(&format!("{l}.mlp.fc2"), d, mlp, true);
        }
    }
    fn weights(self) -> Weights {
        Weights::from_map(self.map)
    }
}

/// A tiny CLIP text encoder: `d` wide, three layers, a 3x MLP, optional projection.
fn tiny_clip(d: i32, heads: i32, projection: Option<i32>, seed: u64) -> (Weights, ClipTextConfig) {
    let cfg = ClipTextConfig {
        num_layers: 3,
        model_dims: d,
        num_heads: heads,
        max_length: 8,
        vocab_size: 40,
        projection_dim: projection,
        hidden_act: ClipActivation::Gelu,
    };
    let mut g = Gen::new(seed);
    g.put(
        "text_model.embeddings.token_embedding.weight".into(),
        &[40, d],
        0.5,
    );
    g.put(
        "text_model.embeddings.position_embedding.weight".into(),
        &[8, d],
        0.1,
    );
    g.clip_layers("text_model", 3, d, 3 * d);
    g.ln("text_model.final_layer_norm", d);
    if let Some(p) = projection {
        g.linear("text_projection", p, d, false);
    }
    (g.weights(), cfg)
}

fn clip_ids(n: i32) -> Array {
    let mut ids: Vec<i32> = (0..2 * n).map(|i| 1 + (i * 7) % 30).collect();
    ids[(n - 1) as usize] = 39; // EOS is the max id
    ids[(2 * n - 1) as usize] = 39;
    Array::from_slice(&ids, &[2, n])
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn clip_text_encoders_are_bounded_per_layer_and_identical() {
    if !cpu_child("clip_text_encoders_are_bounded_per_layer_and_identical") {
        return;
    }
    let (w1, c1) = tiny_clip(24, 3, None, 300);
    let (w2, c2) = tiny_clip(32, 4, Some(12), 400);
    let te1 = crate::text_encoder::ClipTextEncoder::from_weights(&w1, "text_model", &c1).unwrap();
    let te2 = crate::text_encoder::ClipTextEncoder::from_weights(&w2, "text_model", &c2).unwrap();
    let ids = clip_ids(8);
    let encode = || {
        (
            crate::test_device::record_model_inference_device(),
            crate::pipeline::encode_conditioning(&te1, &te2, &ids),
        )
            .1
    };
    let (cond, pooled) = encode().unwrap();
    let (guarded, events) = observe(PLENTY, encode);
    let (g_cond, g_pooled) = guarded.unwrap();
    assert_eq!(bits(&cond), bits(&g_cond));
    assert_eq!(bits(&pooled), bits(&g_pooled));
    assert_well_ordered(&events);
    for (label, want) in [(CLIP_EMBEDDINGS, 2), (CLIP_LAYER, 6), (CLIP_POOLED, 2)] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    // The layer floor is the loaded widths (3x MLP, configured heads), not CLIP-L/bigG constants.
    let layers = checked(&events, CLIP_LAYER);
    assert_eq!(
        layers[0],
        clip_layer_bytes(2, 8, 24, 72, 3, 8, true, 4, Backend::Cpu)
    );
    assert_eq!(
        layers[3],
        clip_layer_bytes(2, 8, 32, 96, 4, 8, true, 4, Backend::Cpu)
    );
    let short = clip_ids(4);
    let (_, short_events) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            te1.forward(&short),
        )
            .1
            .map(|o| o.pooled)
    });
    assert!(checked(&short_events, CLIP_LAYER)[0] < layers[0]);
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, encode
    )));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::clip_text_encoders_are_bounded_per_layer_and_identical");
}

// ---------------------------------------------------------------------------------------------
// IP-Adapter image encoder: CLIP vision tower + Resampler.
// ---------------------------------------------------------------------------------------------

fn tiny_ip_encoder() -> crate::ip_adapter::IpImageEncoder {
    let vcfg = crate::vision_encoder::VisionConfig {
        hidden: 32,
        num_layers: 3,
        num_heads: 4,
        patch: 4,
        image_size: 16,
        num_channels: 3,
        quick_gelu: false,
    };
    let mut g = Gen::new(500);
    g.put(
        "vision_model.embeddings.patch_embedding.weight".into(),
        &[32, 3, 4, 4],
        0.2,
    );
    g.put("vision_model.embeddings.class_embedding".into(), &[32], 0.1);
    g.put(
        "vision_model.embeddings.position_embedding.weight".into(),
        &[17, 32],
        0.1,
    );
    g.ln("vision_model.pre_layrnorm", 32);
    g.clip_layers("vision_model", 3, 32, 64);
    let encoder =
        crate::vision_encoder::ClipVisionEncoder::from_weights(&g.weights(), &vcfg).unwrap();
    let rcfg = crate::ip_adapter::ResamplerConfig {
        dim: 32,
        depth: 2,
        heads: 2,
        dim_head: 16,
        num_queries: 4,
        embed_dim: 32,
        output_dim: 16,
    };
    let mut g = Gen::new(600);
    g.put("image_proj.latents".into(), &[1, 4, 32], 0.5);
    g.linear("image_proj.proj_in", 32, 32, true);
    g.linear("image_proj.proj_out", 16, 32, true);
    g.ln("image_proj.norm_out", 16);
    for i in 0..2 {
        let a = format!("image_proj.layers.{i}.0");
        g.ln(&format!("{a}.norm1"), 32);
        g.ln(&format!("{a}.norm2"), 32);
        g.linear(&format!("{a}.to_q"), 32, 32, false);
        g.linear(&format!("{a}.to_kv"), 64, 32, false);
        g.linear(&format!("{a}.to_out"), 32, 32, false);
        let f = format!("image_proj.layers.{i}.1");
        g.ln(&format!("{f}.0"), 32);
        g.linear(&format!("{f}.1"), 48, 32, false);
        g.linear(&format!("{f}.3"), 32, 48, false);
    }
    let resampler =
        crate::ip_adapter::Resampler::from_weights(&g.weights(), "image_proj", &rcfg).unwrap();
    crate::ip_adapter::IpImageEncoder::with_image_size(encoder, resampler, 16)
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn ip_image_encoder_is_bounded_per_layer_and_identical() {
    if !cpu_child("ip_image_encoder_is_bounded_per_layer_and_identical") {
        return;
    }
    let encoder = tiny_ip_encoder();
    let image = mlx_gen::media::Image {
        width: 20,
        height: 18,
        pixels: (0..20 * 18 * 3).map(|i| (i * 37 % 251) as u8).collect(),
    };
    let tokens = || encoder.tokens(&image);
    let host = tokens().unwrap();
    let (guarded, events) = observe(PLENTY, tokens);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    for (label, want) in [
        (VISION_EMBEDDINGS, 1),
        // `penultimate` runs every layer but the last.
        (VISION_LAYER, 2),
        (RESAMPLER_PROJ_IN, 1),
        (RESAMPLER_LAYER, 2),
        (RESAMPLER_PROJ_OUT, 1),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    // 17 tokens (16 patches + class), the loaded 64-wide MLP, four heads of 8.
    assert_eq!(
        checked(&events, VISION_LAYER)[0],
        clip_layer_bytes(1, 17, 32, 64, 4, 8, false, 4, Backend::Cpu)
    );
    // Perceiver layers (CPU backend): 8 key-side buffers over 17 image tokens + 4 queries at the
    // 32-wide inner width (6 + the fallback's k/v copies), 15 query-side buffers (5 + three
    // 3-buffer CPU LayerNorms + the scaled q) with the loaded 48-wide FFN's two, the CPU's two
    // materialized score tensors (2 heads x 4 queries x 21 keys), and one page per buffer.
    assert_eq!(
        checked(&events, RESAMPLER_LAYER)[0],
        (8 * 21 * 32 + 4 * (15 * 32 + 2 * 48) + 2 * 2 * 4 * 21) * 4 + (8 + 15 + 4) * PAGE_BYTES
    );
    // The patch-embedding quantum is this chain's widest, so only the first-quantum refusal applies.
    assert_refusals(&events, |headroom| observe(headroom, tokens));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::ip_image_encoder_is_bounded_per_layer_and_identical");
}

// ---------------------------------------------------------------------------------------------
// VAE decode / encode.
// ---------------------------------------------------------------------------------------------

fn tiny_vae() -> crate::vae::Autoencoder {
    let (w, cfg) = crate::vae::tests::fixture();
    crate::vae::Autoencoder::from_weights(&w, &cfg).unwrap()
}

fn vae_latent() -> Array {
    let z = rnd(&[1, 8, 9, 4], 71, 1.0);
    z.eval().unwrap();
    z
}

/// The dense decode quanta of the tiny VAE: conv_in, mid resnet/attention/resnet, two up blocks of
/// two resnets, one upsample, the output conv.
fn assert_dense_decode_counts(events: &[Ev]) {
    for (label, want) in [
        (VAE_DECODE_IN, 1),
        (VAE_RESNET, 2 + 4),
        (VAE_ATTENTION, 1),
        (VAE_UPSAMPLE, 1),
        (VAE_DECODE_OUT, 1),
    ] {
        assert_eq!(checked(events, label).len(), want, "{label} checks");
        assert_eq!(synced(events, label), want, "{label} synchronizations");
    }
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn vae_dense_and_evaluated_decode_are_bounded_and_identical() {
    if !cpu_child("vae_dense_and_evaluated_decode_are_bounded_and_identical") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    for evaluated in [false, true] {
        let decode = || {
            if evaluated {
                (
                    crate::test_device::record_model_inference_device(),
                    vae.decode_evaluated(&z),
                )
                    .1
            } else {
                (
                    crate::test_device::record_model_inference_device(),
                    vae.decode(&z),
                )
                    .1
            }
        };
        let host = decode().unwrap();
        let (guarded, events) = observe(PLENTY, decode);
        assert_eq!(
            bits(&host),
            bits(&guarded.unwrap()),
            "evaluated={evaluated}"
        );
        assert_well_ordered(&events);
        assert_dense_decode_counts(&events);
        // The attention is priced per query chunk at the loaded width: the fixture's 72 latent
        // tokens fit one chunk, so it scores all of them against each other.
        assert_eq!(
            checked(&events, VAE_ATTENTION),
            [vae_attention_bytes(1, 72, 32, 72, 4, Backend::Cpu)]
        );
        assert!(assert_refusals(&events, |headroom| observe(
            headroom, decode
        )));
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::vae_dense_and_evaluated_decode_are_bounded_and_identical");
}

/// The production decode route (`pipeline::decode_image_tiled` through the NCHW decoder seam, both
/// dense and tiled) ends in a guarded pixel conversion and stays byte-identical.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn decode_image_route_is_bounded_end_to_end() {
    if !cpu_child("decode_image_route_is_bounded_end_to_end") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    for tiled in [None, Some(&tiling)] {
        let decode = || {
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::decode_image_tiled(&vae, &z, None, tiled, None),
            )
                .1
        };
        let host = decode().unwrap();
        let (guarded, events) = observe(PLENTY, decode);
        let guarded = guarded.unwrap();
        assert_eq!((host.width, host.height), (guarded.width, guarded.height));
        assert_eq!(host.pixels, guarded.pixels);
        assert_well_ordered(&events);
        assert_eq!(
            checked(&events, DECODED_TO_IMAGE),
            vec![16 * 18 * 3 * 4 + PAGE_BYTES]
        );
        assert_eq!(synced(&events, DECODED_TO_IMAGE), 1);
        assert_refusals(&events, |headroom| observe(headroom, decode));
    }

    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::decode_image_route_is_bounded_end_to_end"
    );
}

/// Tiles one layer-wise tiled conv of an `h x w` input cuts at `edge` (the halo-1 core plan of
/// `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc`).
fn tiles(h: i32, w: i32, edge: i32) -> usize {
    if h <= edge && w <= edge {
        return 1;
    }
    let core = edge - 2;
    (((h + core - 1) / core) * ((w + core - 1) / core)) as usize
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn vae_tiled_decode_checks_every_tile_before_its_graph() {
    if !cpu_child("vae_tiled_decode_checks_every_tile_before_its_graph") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    let decode = || {
        (
            crate::test_device::record_model_inference_device(),
            vae.decode_tiled(&z, &tiling, None),
        )
            .1
    };
    let host = decode().unwrap();
    let (guarded, events) = observe(PLENTY, decode);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    // Output edge 8 over one 2x upsample: block 0 tiles at edge 4 on the 8x9 latent grid (two
    // resnets, two convs each), its upsample conv at edge 8 on 16x18, block 1 likewise at 16x18,
    // then the output conv.
    let want_tiles =
        4 * tiles(8, 9, 4) + tiles(16, 18, 8) + 4 * tiles(16, 18, 8) + tiles(16, 18, 8);
    assert_eq!(checked(&events, VAE_TILE).len(), want_tiles);
    assert_eq!(checked(&events, VAE_TILED_STAGE).len(), 4 + 1 + 1);
    assert_eq!(synced(&events, VAE_TILED_STAGE), 4 + 1 + 1);
    // Each tile's core is synchronized and drained before the next tile's graph is built.
    assert_eq!(synced(&events, TILED_CORE), want_tiles);
    // The head is still dense and guarded.
    for (label, want) in [(VAE_DECODE_IN, 1), (VAE_RESNET, 2), (VAE_ATTENTION, 1)] {
        assert_eq!(checked(&events, label).len(), want, "{label}");
    }
    // A tile's floor is its own extent: the 4-edge tiles are smaller than the 8-edge ones.
    let tile_floors = checked(&events, VAE_TILE);
    assert!(tile_floors[0] < *tile_floors.last().unwrap());
    assert_refusals(&events, |headroom| observe(headroom, decode));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::vae_tiled_decode_checks_every_tile_before_its_graph");
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn vae_encode_is_bounded_and_identical() {
    if !cpu_child("vae_encode_is_bounded_and_identical") {
        return;
    }
    let vae = tiny_vae();
    let image = rnd(&[1, 16, 18, 3], 81, 0.5);
    image.eval().unwrap();
    let encode = || {
        (
            crate::test_device::record_model_inference_device(),
            vae.encode_mean(&image),
        )
            .1
    };
    let host = encode().unwrap();
    let (guarded, events) = observe(PLENTY, encode);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    for (label, want) in [
        (VAE_ENCODE_IN, 1),
        (VAE_ENCODE_RESNET, 2 + 2),
        (VAE_DOWNSAMPLE, 1),
        (VAE_ENCODE_ATTENTION, 1),
        (VAE_ENCODE_OUT, 1),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    assert_refusals(&events, |headroom| observe(headroom, encode));

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::vae_encode_is_bounded_and_identical");
}

// ---------------------------------------------------------------------------------------------
// Component loads.
// ---------------------------------------------------------------------------------------------

fn save(weights: &Weights, path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    let named: Vec<(&str, &Array)> = keys
        .iter()
        .map(|k| (k.as_str(), weights.get(k).unwrap()))
        .collect();
    Array::save_safetensors(named, None, path).unwrap();
}

/// The cast-aware load floor for dense `files` loaded at `target` (`width` bytes) small enough to
/// materialize in one batch: every tensor's stored bytes as read, plus the cast copy of every
/// tensor whose stored dtype differs — the read sources and the copies are live together until the
/// batch completes.
fn cast_load_floor(
    files: &[&std::path::Path],
    target: mlx_gen::gen_core::weightsmeta::Dtype,
    width: u64,
) -> u64 {
    files
        .iter()
        .flat_map(|f| mlx_gen::gen_core::safetensors_path_tensor_headers(f).unwrap())
        .map(|h| {
            let cast = if h.dtype == target {
                0
            } else {
                mlx_gen::ios_quantum::allocated_bytes(
                    h.shape.iter().product::<usize>() as u64 * width,
                )
            };
            mlx_gen::ios_quantum::allocated_bytes(h.data_bytes) + cast
        })
        .sum()
}

/// Every component loader checks its header floor before constructing anything; with no headroom
/// it refuses with a typed record naming the component and never reaches construction. The tiny
/// fixtures cannot satisfy the SDXL-base configs those loaders build, so with ample headroom each
/// load records its check FIRST and only then fails in construction — the order under test.
#[test]
fn component_loads_check_their_header_floor_before_construction() {
    if !cpu_child("component_loads_check_their_header_floor_before_construction") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (vae_w, _) = crate::vae::tests::fixture();
    let vae_file = root.join("vae/diffusion_pytorch_model.safetensors");
    save(&vae_w, &vae_file);
    let (clip_w, _) = tiny_clip(24, 3, None, 300);
    let clip_file = root.join("text_encoder/model.safetensors");
    save(&clip_w, &clip_file);
    let unet_file = root.join("unet/diffusion_pytorch_model.safetensors");
    save(
        &crate::test_support::tiny_unet_weights(7).unwrap(),
        &unet_file,
    );
    let fix_file = root.join("fix/vae.safetensors");
    save(&vae_w, &fix_file);

    let fix = mlx_gen::WeightsSource::File(fix_file.clone());
    let unet_source = mlx_gen::WeightsSource::File(unet_file.clone());
    type Load<'a> = Box<dyn Fn() -> Result<()> + 'a>;
    let cases: Vec<(&str, &std::path::Path, Load)> = vec![
        (
            "SDXL VAE load",
            &vae_file,
            Box::new(|| crate::loader::load_vae(root).map(drop)),
        ),
        (
            "SDXL fp16-fix VAE load",
            &fix_file,
            Box::new(|| crate::loader::load_vae_fp16_fix(&fix).map(drop)),
        ),
        (
            "SDXL text_encoder load",
            &clip_file,
            Box::new(|| crate::loader::load_text_encoder_1_dtype(root, Dtype::Float32).map(drop)),
        ),
        (
            "SDXL UNet load",
            &unet_file,
            Box::new(|| crate::loader::load_unet_dtype(root, Dtype::Float16).map(drop)),
        ),
        (
            "SDXL ControlNet load",
            &unet_file,
            Box::new(|| crate::loader::load_controlnet(&unet_source, Dtype::Float16).map(drop)),
        ),
    ];
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    for (label, file, load) in &cases {
        // Every loader casts after reading (`cast_for_load` / the VAEs' `cast_all`): the f32
        // fixtures loaded at f16 (the fp16-fix VAE, the U-Net, the ControlNet) price their f16
        // copies; the f32 loads (the stock VAE, CLIP here) cast nothing and price the stored bytes.
        let floor = match *label {
            "SDXL fp16-fix VAE load" | "SDXL ControlNet load" => {
                cast_load_floor(&[file], Stored::F16, 2)
            }
            // The U-Net's constructor also builds every block's GEGLU row selections and the
            // timestep / `text_time` sinusoid tables (the SDXL-base config's) before materializing.
            "SDXL UNet load" => {
                let tables = crate::unet::SinusoidalPositionalEncoding::timestep_table_bytes(320)
                    + crate::unet::SinusoidalPositionalEncoding::timestep_table_bytes(256);
                cast_load_floor(&[file], Stored::F16, 2)
                    + crate::loader::geglu_index_bytes(&[file.to_path_buf()], |_| true).unwrap()
                    + tables
            }
            _ => cast_load_floor(&[file], Stored::F32, 4),
        };
        let _ = take_refusal();
        let (refused, events) = observe(0, load);
        assert!(refused.is_err(), "{label}");
        let record = take_refusal().expect("typed load refusal");
        assert_eq!(record.label, *label);
        assert_eq!(record.scratch_bytes, floor);
        assert!(events.is_empty());
        let (_, events) = observe(PLENTY, load);
        assert_eq!(
            events.first(),
            Some(&Ev::Checked(label.to_string(), floor)),
            "{label} must be checked before construction: {events:?}"
        );
    }
    // A cast load's floor carries the copies: the f16 copies of the f32 U-Net, the f32 source
    // held with the largest copy — not the stored bytes alone.
    assert!(
        cast_load_floor(&[&unet_file], Stored::F16, 2)
            != cast_load_floor(&[&unet_file], Stored::F32, 4)
    );
}

/// The fused-LDM phase floors count only the components that phase builds.
#[test]
fn fused_checkpoint_floor_counts_only_the_phase_components() {
    if !cpu_child("fused_checkpoint_floor_counts_only_the_phase_components") {
        return;
    }
    use gen_core_family::sdxl_ldm::SdxlComponent;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("fused.safetensors");
    let mut g = Gen::new(900);
    g.put(
        "model.diffusion_model.input_blocks.7.0.out_layers.3.weight".into(),
        &[64],
        1.0,
    );
    g.put(
        "first_stage_model.decoder.up.1.block.2.nin_shortcut.weight".into(),
        &[16],
        1.0,
    );
    g.put(
        "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight".into(),
        &[8],
        1.0,
    );
    g.put(
        "conditioner.embedders.1.model.text_projection".into(),
        &[4, 2],
        1.0,
    );
    g.put("unrelated.metadata".into(), &[1000], 1.0);
    save(&g.weights(), &file);
    let floor = |components: &[SdxlComponent]| {
        let _ = take_refusal();
        let refused = with_cpu_diagnostic_headroom(0, || {
            crate::ldm::before_component_load("LDM", &file, components)
        });
        assert!(refused.is_err());
        take_refusal().unwrap().scratch_bytes
    };
    // Resident bytes after each component's cast (the f32 U-Net / CLIP tensors become f16 copies;
    // the VAE stays f32) plus the largest transient (an f32 source held with its f16 copy).
    assert_eq!(
        floor(&[SdxlComponent::ClipL, SdxlComponent::ClipBigG]),
        (16 + 16) + (16 + 32)
    );
    assert_eq!(
        floor(&[SdxlComponent::Unet, SdxlComponent::Vae]),
        (128 + 64) + (128 + 256)
    );
    assert_eq!(floor(&[SdxlComponent::Unet]), 128 + (128 + 256));
    // Off device it reads nothing and passes.
    crate::ldm::before_component_load("LDM", &file, &[SdxlComponent::Unet]).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Conditioning window count (weights-free; iOS admission keys evidence on it).
// ---------------------------------------------------------------------------------------------

/// A two-word CLIP vocabulary under `<root>/tokenizer/` (every `a` / `b` word is one token).
fn tiny_tokenizer_root(root: &std::path::Path) {
    let dir = root.join("tokenizer");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("vocab.json"),
        r#"{"<|startoftext|>": 49406, "<|endoftext|>": 49407, "a</w>": 320, "b</w>": 321}"#,
    )
    .unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
}

#[test]
fn conditioning_window_count_matches_the_encode_path() {
    if !cpu_child("conditioning_window_count_matches_the_encode_path") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    tiny_tokenizer_root(tmp.path());
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tmp.path().to_owned()));
    let tokenizer = crate::loader::load_tokenizer(tmp.path()).unwrap();
    let words = |n: usize| vec!["a"; n].join(" ");
    let count = |p: &str, n: Option<&str>| crate::conditioning_window_count(&spec, p, n).unwrap();
    let encoded = |p: &str, n: Option<&str>| tokenizer.tokenize_windows(p, n).unwrap().len() as u32;
    let short = words(10);
    let long = words(80); // 80 content tokens > 75 per window
    let longer = words(160);
    assert_eq!(count(&short, None), 1);
    assert_eq!(count(&short, Some("b b")), 1);
    assert_eq!(count(&long, None), 2);
    assert_eq!(
        count(&short, Some(&long)),
        2,
        "a longer negative raises the count"
    );
    assert_eq!(count(&long, Some(&longer)), 3);
    for (p, n) in [
        (short.as_str(), None),
        (short.as_str(), Some("b")),
        (long.as_str(), None),
        (short.as_str(), Some(long.as_str())),
        (long.as_str(), Some(longer.as_str())),
        (words(75).as_str(), None),
        (words(76).as_str(), Some("")),
    ] {
        assert_eq!(count(p, n), encoded(p, n), "{p:?} / {n:?}");
    }
    // A fused single-file checkpoint resolves the caller-staged tokenizer component, like `load`.
    let mut fused = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::File(
        tmp.path().join("model.safetensors"),
    ));
    assert!(crate::conditioning_window_count(&fused, &short, None).is_err());
    fused.components.insert(
        crate::LDM_TOKENIZER_COMPONENT.to_owned(),
        mlx_gen::WeightsSource::Dir(tmp.path().to_owned()),
    );
    assert_eq!(
        crate::conditioning_window_count(&fused, &long, None).unwrap(),
        2
    );
}

/// The f16-storage upsample convolution accumulates in f32 (`tiled_conv2d_3x3_nhwc_f32_accum`): its
/// stage prices the stage activations PLUS the f32 weight and bias copies that stay live across all
/// of the stage's tiles. The f32 path has no such copies.
#[test]
fn half_upsample_stage_prices_its_f32_accumulation_copies() {
    if !cpu_child("half_upsample_stage_prices_its_f32_accumulation_copies") {
        return;
    }
    let (b, h, w, cin, cout) = (1, 6, 6, 4, 8);
    let x = Array::ones::<f32>(&[b, h, w, cin]).unwrap();
    let weight = Array::ones::<f32>(&[cout, 3, 3, cin]).unwrap();
    let bias = Array::zeros::<f32>(&[cout]).unwrap();
    let stage = |x: &Array, weight: &Array, bias: &Array| {
        let (_, events) = observe(PLENTY, || {
            crate::vae::upsample_tiled(x, weight, bias, 8, None)
        });
        checked(&events, VAE_TILED_STAGE)
    };
    // The nearest-2x copy (input width) and two output-width images (row strips + assembly), at
    // the doubled extent and the activation width.
    let up = |width: u64| (b * 2 * h * 2 * w) as u64 * width;
    let activations = |width: u64| up(width) * (cin as u64 + 2 * cout as u64);
    assert_eq!(stage(&x, &weight, &bias), [activations(4)]);
    let half = |a: &Array| a.as_dtype(Dtype::Float16).unwrap();
    let wide_copies = (cout * 9 * cin) as u64 * 4 + cout as u64 * 4;
    assert_eq!(
        stage(&half(&x), &half(&weight), &half(&bias)),
        [activations(2) + wide_copies]
    );
}

/// The bounded (tiled) encode against the whole-image encode, held to the tiled-decode parity
/// standard (`vae::tests`): 1e-4 of peak for an f32 VAE against the f32 whole-image encode, 1e-2 of
/// peak for the fp16 VAE against the f32 whole-image encode of the same f16-rounded weights — at
/// tile edges that split the image evenly, raggedly and down to the 3-pixel floor (the stride-2
/// downsample tiles reach the padding row/column at the ragged edges).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn tiled_encode_meets_the_tiled_parity_standard() {
    if !cpu_child("tiled_encode_meets_the_tiled_parity_standard") {
        return;
    }
    let (w, cfg) = crate::vae::tests::fixture();
    let mut rounded = w.clone();
    rounded.cast_all(Dtype::Float16).unwrap();
    let mut half = rounded.clone();
    rounded.cast_all(Dtype::Float32).unwrap();
    half.cast_all(Dtype::Float16).unwrap();
    let reference = crate::vae::Autoencoder::from_weights(&rounded, &cfg).unwrap();
    let half = crate::vae::Autoencoder::from_weights(&half, &cfg).unwrap();
    for (h, wd) in [(16, 18), (17, 23)] {
        let image = rnd(&[1, h, wd, 3], 81, 0.5);
        image.eval().unwrap();
        let dense = (
            crate::test_device::record_model_inference_device(),
            reference.encode_mean(&image),
        )
            .1
            .unwrap();
        for edge in [3, 5, 8, 9] {
            let tiled = (
                crate::test_device::record_model_inference_device(),
                reference.encode_tiled(&image, edge, None),
            )
                .1
                .unwrap();
            let delta = relative_max_delta(&dense, &tiled);
            eprintln!("{h}x{wd} edge {edge}: f32 {delta:.3e}");
            assert!(
                delta <= 1e-4,
                "f32 tiled encode at edge {edge} drifted {delta:.3e}"
            );
            let tiled16 = (
                crate::test_device::record_model_inference_device(),
                half.encode_tiled(&image, edge, None),
            )
                .1
                .unwrap();
            assert_eq!(tiled16.dtype(), Dtype::Float32, "callers receive f32");
            let delta = relative_max_delta(&dense, &tiled16);
            eprintln!("{h}x{wd} edge {edge}: f16 {delta:.3e}");
            assert!(
                delta <= 1e-2,
                "fp16 tiled encode at edge {edge} drifted {delta:.3e}"
            );
        }
        // An image that fits one tile takes the exact whole-image encode.
        assert_eq!(
            bits(&dense),
            bits(
                &(
                    crate::test_device::record_model_inference_device(),
                    reference.encode_tiled(&image, 32, None)
                )
                    .1
                    .unwrap()
            )
        );
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::tiled_encode_meets_the_tiled_parity_standard");
}

/// The bounded encode on the iOS path: `encode_mean` selects it above one tile, every tiled stage
/// and tile is checked before its graph and refuses before construction, guarded execution equals
/// the host's run of the same bounded encode bit for bit, and its mid attention runs
/// query-chunked; off the iOS path `encode_mean` is the whole-image encode. A bounded-decode rung's
/// tiling routes the img2img init encode through the tiled encode (its cancel point proves the
/// route).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn bounded_encode_is_selected_on_ios_guarded_and_routed_by_the_rung() {
    if !cpu_child("bounded_encode_is_selected_on_ios_guarded_and_routed_by_the_rung") {
        return;
    }
    let vae = tiny_vae();
    let tile = crate::vae::IOS_ENCODE_TILE_PX;
    // One more than the iOS tile edge: the encoder's 2x downsample leaves a 133x133 latent
    // (17689 tokens), so the bounded attention chunks.
    let image = rnd(&[1, tile + 10, tile + 10, 3], 91, 0.5);
    image.eval().unwrap();
    let host = (
        crate::test_device::record_model_inference_device(),
        vae.encode_tiled(&image, tile, None),
    )
        .1
        .unwrap();
    assert!(
        mlx_gen::attention::last_chunk_count() > 1,
        "the bounded encode's mid attention must run query-chunked"
    );
    let (guarded, events) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            vae.encode_mean(&image),
        )
            .1
    });
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert!(
        !checked(&events, VAE_ENCODE_TILE).is_empty(),
        "iOS encode_mean must tile"
    );
    // conv_in, block 0's resnet, its downsample, block 1's resnet: one tiled stage each.
    assert_eq!(checked(&events, VAE_ENCODE_TILED_STAGE).len(), 4);
    assert_eq!(synced(&events, VAE_ENCODE_TILED_STAGE), 4);
    assert!(
        checked(&events, VAE_TILE).is_empty() && checked(&events, VAE_RESNET).is_empty(),
        "an encode checks only encode-phase labels"
    );
    assert_eq!(
        checked(&events, VAE_ENCODE_IN).len(),
        0,
        "no whole-image conv_in on iOS"
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom,
        || (
            crate::test_device::record_model_inference_device(),
            vae.encode_mean(&image)
        )
            .1
    )));

    // The rung's tiling routes the init encode through the tiled encode: a tripped cancel flag
    // stops it between tiles, while the untiled route ignores it.
    let pixels = mlx_gen::media::Image {
        width: 40,
        height: 32,
        pixels: (0..40 * 32 * 3).map(|i| (i * 31 % 251) as u8).collect(),
    };
    let tripped = mlx_gen::CancelFlag::default();
    tripped.cancel();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    let routed = (
        crate::test_device::record_model_inference_device(),
        crate::pipeline::encode_init_latents_tiled(
            &vae,
            &pixels,
            40,
            32,
            Some(&tiling),
            Some(&tripped),
        ),
    )
        .1;
    assert!(
        matches!(routed, Err(mlx_gen::Error::Canceled)),
        "{routed:?}"
    );
    (
        crate::test_device::record_model_inference_device(),
        crate::pipeline::encode_init_latents_tiled(&vae, &pixels, 40, 32, None, Some(&tripped)),
    )
        .1
        .unwrap();

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::bounded_encode_is_selected_on_ios_guarded_and_routed_by_the_rung");
}

/// Items 3 and 5 (sc-26270 round 3): every ancestral denoise step's CFG combine + sampler update is
/// materialized inside its own guarded quantum (checked, synchronized, byte-identical to the host),
/// and a sequential CFG half prices its transformer blocks at the ONE latent row it runs — only the
/// text k/v it projects at the pair's batch are pair-sized.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn denoise_steps_are_guarded_and_sequential_cfg_prices_the_latent_batch() {
    if !cpu_child("denoise_steps_are_guarded_and_sequential_cfg_prices_the_latent_batch") {
        return;
    }
    use mlx_gen::gen_core::CfgBatching;
    // This test isolates guard/evaluation identity, not the separately-proven iOS boundary
    // quantization. Match production compute dtypes so that those casts are identities here.
    // actual_noised_unet_input_is_cast_at_every_forward retains the f32-state counterexample;
    // the full-request RGB matrix owns the unchanged numerical cast tolerances.
    let mut unet = crate::test_support::tiny_unet(7).unwrap();
    unet.cast_weights(crate::model::DTYPE).unwrap();
    let euler = crate::sampler::EulerSampler::new_with_dtype(
        &crate::config::DiffusionConfig::sdxl_base(),
        true,
        crate::model::DTYPE,
    )
    .unwrap();
    let sampler = crate::sampler::AncestralEuler::new(&euler, 3, 1000.0).unwrap();
    let (_, ctx, pooled, ids) = unet_inputs(2, 8, 8);
    let ctx = ctx.as_dtype(crate::model::DTYPE).unwrap();
    let pooled = pooled.as_dtype(crate::model::DTYPE).unwrap();
    let latents = rnd(&[1, 8, 8, 4], 21, 1.0)
        .as_dtype(crate::model::DTYPE)
        .unwrap();
    latents.eval().unwrap();
    let cancel = mlx_gen::CancelFlag::default();
    let (unet, sampler, cancel) = (&unet, &sampler, &cancel);
    let (ctx, pooled, ids) = (&ctx, &pooled, &ids);
    let run = |batching: CfgBatching| {
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(batching);
        let d = crate::pipeline::Denoiser::with_plan(unet, sampler, plan);
        let latents = latents.clone();
        move || -> Result<Array> {
            mlx_rs::random::seed(5)?;
            (
                crate::test_device::record_model_inference_device(),
                crate::pipeline::denoise(
                    &d,
                    latents.clone(),
                    ctx,
                    pooled,
                    ids,
                    5.0,
                    cancel,
                    &mut |_| {},
                ),
            )
                .1
        }
    };
    for batching in [CfgBatching::Batched, CfgBatching::Sequential] {
        let host = run(batching)().unwrap();
        let (guarded, events) = observe(PLENTY, run(batching));
        assert_eq!(bits(&host), bits(&guarded.unwrap()), "{batching:?}");
        assert_well_ordered(&events);
        let steps = checked(&events, SAMPLER_STEP);
        assert_eq!(
            steps.len(),
            3,
            "{batching:?}: one sampler-step quantum per step"
        );
        assert_eq!(synced(&events, SAMPLER_STEP), 3);
        assert!(steps
            .iter()
            .all(|s| *s == sampler_step_bytes(&latents, false)));
        // At tiny f16 geometries the sequential path's pair-side k/v and page slack can
        // outweigh its one-row activation savings. Prove the actual row geometry, not an
        // unsupported ordering between aggregate budgets.
        let sequential = !batching.is_batched();
        let batch = if sequential { 1 } else { 2 };
        let x = shaped(&[batch, 8, 8, 4], crate::model::DTYPE).unwrap();
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
            .with_cfg_batching(batching)
            .with_cfg_row(sequential.then_some(0));
        let mut predicted = Vec::new();
        unet.price_forward(&x, ctx, pooled, None, None, plan, &mut predicted)
            .unwrap();
        let blocks: Vec<_> = predicted
            .iter()
            .filter(|(label, _)| *label == TRANSFORMER_BLOCK)
            .map(|(_, scratch)| *scratch)
            .collect();
        assert!(!blocks.is_empty());
        let forwards = 3 * if sequential { 2 } else { 1 };
        assert_eq!(checked(&events, TRANSFORMER_BLOCK), blocks.repeat(forwards),
            "{batching:?}: every actual block is priced at its explicit {batch}-row forward geometry");
        assert_eq!(
            checked(&events, UNET_STEM),
            vec![
                predicted
                    .iter()
                    .find(|(label, _)| *label == UNET_STEM)
                    .unwrap()
                    .1;
                forwards
            ],
            "{batching:?}: actual stem input has exactly {batch} latent rows"
        );
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::denoise_steps_are_guarded_and_sequential_cfg_prices_the_latent_batch");
}

/// F32 noising is part of the historical sampler arithmetic, not a request to narrow the
/// model input. Both actual forwards must price and preserve that promoted operand; the narrow
/// counterfactual keeps this native driver test sensitive to a reintroduced iOS-only cast.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn actual_noised_unet_input_preserves_sampler_precision_at_every_forward() {
    if !cpu_child("actual_noised_unet_input_preserves_sampler_precision_at_every_forward") {
        return;
    }
    let mut unet = crate::test_support::tiny_unet(7).unwrap();
    unet.cast_weights(crate::model::DTYPE).unwrap();
    let (pair, ctx, pooled, ids) = unet_inputs(2, 8, 8);
    let ctx = ctx.as_dtype(crate::model::DTYPE).unwrap();
    let pooled = pooled.as_dtype(crate::model::DTYPE).unwrap();
    let narrow = pair.as_dtype(crate::model::DTYPE).unwrap();
    let full_stem = |input: &Array| {
        let mut prices = Vec::new();
        unet.price_forward(
            input,
            &ctx,
            &pooled,
            None,
            None,
            crate::plan::SdxlForwardPlan::UNBOUNDED,
            &mut prices,
        )
        .unwrap();
        prices
            .into_iter()
            .find(|(label, _)| *label == UNET_STEM)
            .unwrap()
            .1
    };
    let expected = full_stem(&pair);
    assert_ne!(
        expected,
        full_stem(&narrow),
        "the fixture must distinguish f32 from compute-dtype stem inputs"
    );
    let euler = crate::sampler::EulerSampler::new_with_dtype(
        &crate::config::DiffusionConfig::sdxl_base(),
        true,
        crate::model::DTYPE,
    )
    .unwrap();
    let sampler = crate::sampler::AncestralEuler::new(&euler, 2, 800.0).unwrap();
    let d = crate::pipeline::Denoiser::with_plan(
        &unet,
        &sampler,
        crate::plan::SdxlForwardPlan::UNBOUNDED,
    );
    let init = rnd(&[1, 8, 8, 4], 4, 1.0)
        .as_dtype(crate::model::DTYPE)
        .unwrap();
    let noise = rnd(init.shape(), 5, 1.0);
    let noised = euler.add_noise_with(&init, &noise, 800.0).unwrap();
    assert_eq!(noised.dtype(), Dtype::Float32);
    let (out, events) = observe(PLENTY, || {
        (
            crate::test_device::record_model_inference_device(),
            crate::pipeline::denoise(
                &d,
                noised,
                &ctx,
                &pooled,
                &ids,
                5.0,
                &mlx_gen::CancelFlag::default(),
                &mut |_| {},
            ),
        )
            .1
    });
    assert_eq!(
        out.unwrap().dtype(),
        Dtype::Float32,
        "forward execution must not narrow the persistent sampler state"
    );
    assert_eq!(checked(&events, UNET_STEM), vec![expected; 2]);

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::actual_noised_unet_input_preserves_sampler_precision_at_every_forward");
}

/// Item 3 (sc-26270 round 3): the ControlNet branch keeps the request's bounded attention plan, so
/// its attention runs (and its blocks are priced) under the plan's chunking, not unbounded.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn controlnet_keeps_the_bounded_attention_plan() {
    if !cpu_child("controlnet_keeps_the_bounded_attention_plan") {
        return;
    }
    let cfg = crate::test_support::tiny_control_unet_config();
    let control = crate::unet::ControlNet::from_weights(
        &crate::test_support::tiny_controlnet_weights(&cfg, 5).unwrap(),
        &cfg,
    )
    .unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 16, 16);
    let image = rnd(&[1, 128, 128, 3], 41, 0.5);
    image.eval().unwrap();
    let cond = control.embed_cond(&image).unwrap();
    cond.eval().unwrap();
    let run = |plan: crate::plan::SdxlForwardPlan<'_>| {
        mlx_gen::attention::reset_chunk_count();
        let out = (
            crate::test_device::record_model_inference_device(),
            control.forward_planned(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8, plan),
        )
            .1
            .unwrap();
        (out.mid, mlx_gen::attention::last_chunk_count())
    };
    let (mid, unchunked) = run(crate::plan::SdxlForwardPlan::UNBOUNDED);
    assert_eq!(unchunked, 1, "the unbounded branch runs one SDPA call");
    // 40 scores per chunk over the mid block's 4x4 grid (16 tokens, 2 heads, 5 text keys): its last
    // (cross) attention runs 4-row chunks.
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(40, true);
    assert_eq!(budget.query_block(1, 2, 16, 5), 4);
    let (bounded_mid, chunks) = run(crate::plan::SdxlForwardPlan::with_attention(
        mlx_gen::attention::AttentionPlan::budgeted(budget),
    ));
    assert!(
        chunks > 1,
        "the ControlNet must run the request's bounded plan, ran {chunks} chunk(s)"
    );
    // Chunked attention is the same attention (fused per chunk): the branch's output holds.
    assert!(relative_max_delta(&mid, &bounded_mid) <= 1e-4);

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::controlnet_keeps_the_bounded_attention_plan");
}

/// The device's convolution transients, modelled from MLX's Metal dispatch (they cannot be measured
/// on the CPU): Winograd only for a 3x3 stride-1 call with 32-aligned channels, `N·H·W ≥ 4096` and
/// `C + O ≥ 256` — the padded input, the transformed input/output tiles and the 64-tap filter, plus
/// the weight's contiguous copy every call makes; and the f32-accumulating upsample tile's promoted
/// crop.
#[test]
fn metal_conv_transients() {
    if !cpu_child("metal_conv_transients") {
        return;
    }
    let w = |o: i32, k: i32, c: i32| {
        Array::zeros::<f32>(&[o, k, k, c])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap()
    };
    // A 1024² level-0 resnet conv: [1, 128, 128, 320] -> 320 at f16.
    let call = ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 3, 320), 1, 1);
    assert!(call.winograd());
    let tiles = 22 * 22; // ⌈128 / 6⌉²
    let padded = 6 * (128_u64 + 2 - 2).div_ceil(6) + 2; // 134
    assert_eq!(
        call.transient(Backend::Metal),
        320 * 9 * 320 * 2 // contiguous weight copy
            + padded * padded * 320 * 2
            + 64 * tiles * 320 * 2 * 2 // transformed input + output
            + 64 * 320 * 320 * 2 // transformed filter
    );
    // Too small (N·H·W = 1024), strided, 1x1, or misaligned: implicit GEMM, the weight copy only.
    for call in [
        ConvCall::new(&[1, 32, 32, 1280], 2, &w(1280, 3, 1280), 1, 1),
        ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 3, 320), 2, 1),
        ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 1, 320), 1, 0),
        ConvCall::new(&[1, 128, 128, 4], 2, &w(320, 3, 4), 1, 1),
    ] {
        assert!(!call.winograd());
        assert_eq!(call.transient(Backend::Metal), call.weight_elements() * 2);
    }
    // The fp16-fix VAE's f32-accumulating upsample tile prices its f32 promoted crop; the f32 path
    // has none.
    let wt = w(512, 3, 512);
    let tile = [1, 64, 64, 512];
    let crop = 64 * 64 * 512 * 2;
    // A raw crop of a 128-wide activation: its 128-wide row strip and the crop.
    let raw = TileSource {
        full_width: 128,
        normalized: false,
    };
    let strip = 64 * 128 * 512 * 2;
    let wide = ConvCall::new(&tile, 4, &wt, 1, 1);
    assert_eq!(
        tile_bytes(&tile, raw, 2, &wt, true, Backend::Metal),
        // strip + f16 crop, the f32 promoted crop, the f32 output, bias sum and core, the core's f16
        // cast, the f32 call's transients and the page slack.
        strip
            + crop
            + 64 * 64 * 512 * 4
            + 3 * wide.out_bytes()
            + 64 * 64 * 512 * 2
            + wide.transient(Backend::Metal)
            + page_slack(16)
    );
    let narrow = ConvCall::new(&tile, 2, &wt, 1, 1);
    assert_eq!(
        tile_bytes(&tile, raw, 2, &wt, false, Backend::Metal),
        strip + crop + 3 * narrow.out_bytes() + narrow.transient(Backend::Metal) + page_slack(16)
    );
    // Normalized at f16: + the grouped copy, centred and transposed-back f32 buffers, the f32
    // normalized product and the f16 cast back.
    let normalized = TileSource {
        full_width: 128,
        normalized: true,
    };
    assert_eq!(
        tile_bytes(&tile, normalized, 2, &wt, false, Backend::Metal),
        tile_bytes(&tile, raw, 2, &wt, false, Backend::Metal) + 4 * 64 * 64 * 512 * 4 + crop
    );
}

/// `max|a - b| / max|reference|`, at f32.
fn relative_max_delta(reference: &Array, candidate: &Array) -> f32 {
    let f = |a: &Array| {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    };
    let (r, c) = (f(reference), f(candidate));
    let peak = r.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.0);
    r.iter()
        .zip(&c)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
        / peak
}

/// sc-26270 blocker fix: the VAE mid attention runs query-chunked on iOS. Each chunk's rows attend
/// to every key — the same single-head attention, row for row — but a chunk changes the score
/// GEMM's row count, so the result is equal up to float reassociation, not bit-identical (measured
/// on the CPU). It is held to the tiled-decode parity standard (`vae::tests`): 1e-4 of peak for an
/// f32 VAE against the f32 single call, 1e-2 of peak for the fp16 VAE against the f32 single call
/// on the same f16-rounded weights — at chunk sizes that divide the tokens, leave a ragged tail,
/// and degenerate to one row. The chunked plan is selected only under the iOS path: off device the
/// plan is unbounded, so the host decode is the historical single SDPA call, byte for byte.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn vae_attention_query_chunks_meet_the_tiled_parity_standard_and_are_ios_only() {
    if !cpu_child("vae_attention_query_chunks_meet_the_tiled_parity_standard_and_are_ios_only") {
        return;
    }
    let mut g = Gen::new(7700);
    g.ln("a.group_norm", 64);
    for p in ["to_q", "to_k", "to_v", "to_out.0"] {
        g.linear(&format!("a.{p}"), 64, 64, true);
    }
    // f16-rounded weights, so the f32 reference and the f16 attention share their storage rounding.
    let mut rounded = g.weights();
    rounded.cast_all(Dtype::Float16).unwrap();
    let mut half = rounded.clone();
    rounded.cast_all(Dtype::Float32).unwrap();
    let attn = crate::vae::VaeAttention::from_weights(&rounded, "a").unwrap();
    let attn16 = crate::vae::VaeAttention::from_weights(&half, "a").unwrap();
    half.cast_all(Dtype::Float16).unwrap();
    let x = rnd(&[1, 24, 40, 64], 7, 2.0)
        .as_dtype(Dtype::Float16)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    x.eval().unwrap();
    let x16 = x.as_dtype(Dtype::Float16).unwrap();
    let tokens = 24 * 40;
    let unbounded = mlx_gen::attention::AttentionPlan::UNBOUNDED;
    let full = (
        crate::test_device::record_model_inference_device(),
        attn.forward_planned(&x, unbounded),
    )
        .1
        .unwrap();
    let full16 = (
        crate::test_device::record_model_inference_device(),
        attn16.forward_planned(&x16, unbounded),
    )
        .1
        .unwrap();
    for rows in [1, 37, 128, 480, tokens - 1] {
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(
            rows as u64 * tokens as u64,
            true,
        );
        assert_eq!(budget.query_block(1, 1, tokens, tokens), rows);
        let plan = mlx_gen::attention::AttentionPlan::budgeted(budget);
        let chunked = (
            crate::test_device::record_model_inference_device(),
            attn.forward_planned(&x, plan),
        )
            .1
            .unwrap();
        let delta = relative_max_delta(&full, &chunked);
        eprintln!(
            "f32 {rows}-row chunks: {delta:.3e} of peak, bit-exact {}",
            bits(&full) == bits(&chunked)
        );
        assert!(
            delta <= 1e-4,
            "f32 {rows}-row chunks drifted {delta:.3e} of peak"
        );
        let chunked16 = (
            crate::test_device::record_model_inference_device(),
            attn16.forward_planned(&x16, plan),
        )
            .1
            .unwrap();
        assert_eq!(
            chunked16.dtype(),
            Dtype::Float16,
            "the chunked f16 path stays at f16"
        );
        let delta = relative_max_delta(&full, &chunked16);
        eprintln!(
            "f16 {rows}-row chunks: {delta:.3e} of peak vs f32 (single f16 call {:.3e})",
            relative_max_delta(&full, &full16)
        );
        assert!(
            delta <= 1e-2,
            "f16 {rows}-row chunks drifted {delta:.3e} of peak"
        );
    }
    // Plan selection: unbounded on the host, the bounded score budget under the iOS path.
    assert!(crate::vae::VaeAttention::plan().budget.is_unbounded());
    let ios =
        with_cpu_diagnostic_headroom(PLENTY, || Ok(crate::vae::VaeAttention::plan())).unwrap();
    assert_eq!(
        ios.budget.max_score_elements(),
        VAE_ATTENTION_SCORE_ELEMENTS
    );
    assert!(ios.budget.eval_per_chunk());
    // A 1024² decode's 128x128 latent attention chunks to 1024 rows (16 chunks), so its priced
    // f32 scores are 64 MiB per chunk, not the 1 GiB of the full 16384² score tensor.
    assert_eq!(ios.budget.query_block(1, 1, 16384, 16384), 1024);
    assert!(
        vae_attention_bytes(1, 16384, 512, 1024, 4, Backend::Metal)
            < vae_attention_bytes(1, 16384, 512, 16384, 4, Backend::Metal) / 4
    );

    println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::vae_attention_query_chunks_meet_the_tiled_parity_standard_and_are_ios_only");
}

// ---------------------------------------------------------------------------------------------
// Measured-peak proofs (sc-26270 round 3): every counted SDXL quantum price, evaluated with the
// CPU backend's kernel transients, is at least MLX's measured CPU peak for that quantum's body and
// reports ratios above `MAX_RATIO`. Each body is the guarded quantum's own code (or the one-to-three-line
// op sequence it runs, spelled with the same `mlx_gen::nn` calls), at a small shape where the
// activations dominate the weights; weights and inputs are materialized before the measurement.
// ---------------------------------------------------------------------------------------------

mod proofs {
    use super::*;
    use mlx_gen::ios_quantum::measured_peak_bytes;

    /// Efficiency reporting threshold, not a safety gate.
    const MAX_RATIO: f64 = 1.5;

    /// Measurements per proof; the WORST (largest) peak is the one the formula must cover.
    const RUNS: usize = 5;

    /// Measure `run` (after one warm-up run that materializes lazy weights and kernel state)
    /// [`RUNS`] times, UNTHROTTLED (no memory limit is ever set; the fixture only enables the iOS
    /// guards and barriers), and prove `formula` tracks the worst measured peak. Prints the ratio for
    /// the record.
    pub(super) fn prove(label: &str, formula: u64, run: impl Fn() -> Result<Vec<Array>>) -> f64 {
        let warm = run().unwrap();
        mlx_rs::transforms::eval(warm.iter()).unwrap();
        drop(warm);
        // Measured on the iOS path: the CPU fixture enables every guard and barrier the device runs.
        let measured = (0..RUNS)
            .map(|_| with_cpu_diagnostic_headroom(PLENTY, || measured_peak_bytes(&run)).unwrap())
            .max()
            .unwrap();
        let ratio = formula as f64 / measured as f64;
        eprintln!(
            "PROOF {label}: formula {formula} B, worst of {RUNS} measured {measured} B, ratio {ratio:.3}"
        );
        assert_tracks_measured(label, formula, measured, MAX_RATIO);
        ratio
    }

    fn evaluated(a: Array) -> Array {
        a.eval().unwrap();
        a
    }

    fn rand(shape: &[i32], seed: u64) -> Array {
        evaluated(rnd(shape, seed, 1.0))
    }

    impl Gen {
        /// An NCHW conv `[out, in, k, k]` + bias, as checkpoints store it.
        fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) {
            self.put(
                format!("{p}.weight"),
                &[out, inp, k, k],
                (1.0 / (inp * k * k) as f32).sqrt(),
            );
            self.put(format!("{p}.bias"), &[out], 0.02);
        }
        /// A diffusers `ResnetBlock2D` (U-Net with a `temb` projection, VAE without).
        fn resnet(&mut self, p: &str, cin: i32, cout: i32, temb: Option<i32>) {
            self.ln(&format!("{p}.norm1"), cin);
            self.conv(&format!("{p}.conv1"), cout, cin, 3);
            if let Some(t) = temb {
                self.linear(&format!("{p}.time_emb_proj"), cout, t, true);
            }
            self.ln(&format!("{p}.norm2"), cout);
            self.conv(&format!("{p}.conv2"), cout, cout, 3);
            if cin != cout {
                self.conv(&format!("{p}.conv_shortcut"), cout, cin, 1);
            }
        }
        /// A U-Net `BasicTransformerBlock` (self-attn, cross-attn, GEGLU FFN).
        fn block(&mut self, p: &str, dim: i32, ctx: i32, ff: i32) {
            for n in ["norm1", "norm2", "norm3"] {
                self.ln(&format!("{p}.{n}"), dim);
            }
            for (a, kv) in [("attn1", dim), ("attn2", ctx)] {
                self.linear(&format!("{p}.{a}.to_q"), dim, dim, false);
                self.linear(&format!("{p}.{a}.to_k"), dim, kv, false);
                self.linear(&format!("{p}.{a}.to_v"), dim, kv, false);
                self.linear(&format!("{p}.{a}.to_out.0"), dim, dim, true);
            }
            self.linear(&format!("{p}.ff.net.0.proj"), 2 * ff, dim, true);
            self.linear(&format!("{p}.ff.net.2"), dim, ff, true);
        }
    }

    /// A conv weight in the production layout: the NHWC transposed VIEW of an NCHW checkpoint
    /// tensor (`unet::nchw_to_nhwc`), so the kernels copy it exactly as they do in production.
    fn conv_weight(out: i32, k: i32, inp: i32, seed: u64) -> Array {
        let nchw = evaluated(rnd(
            &[out, inp, k, k],
            seed,
            (1.0 / (inp * k * k) as f32).sqrt(),
        ));
        crate::unet::nchw_to_nhwc(&nchw).unwrap()
    }

    /// Native retained storage of the actual input operation forms: state alias (ancestral),
    /// owned scale/cast (DDPM), and owned f32 scale followed by a pre-CFG half cast (curated).
    /// Caller state remains held, as it does in the solver; aliases are never counted twice.
    #[test]
    fn forward_input_owners_match_native_alias_cast_and_cfg_storage() {
        if !cpu_child("proofs::forward_input_owners_match_native_alias_cast_and_cfg_storage") {
            return;
        }
        with_cpu_diagnostic_headroom(PLENTY, || {
            let _scope = mlx_gen::ios_quantum::Scope::new();
            for state_dtype in [Dtype::Float16, Dtype::Float32] {
                let state = evaluated(rnd(&[1, 8, 8, 4], 7700, 1.0).as_dtype(state_dtype)?);
                let coefficient = evaluated(Array::from_f32(0.75).as_dtype(state_dtype)?);
                for cfg_rows in [1u64, 2] {
                    for flow in [
                        ForwardInputOwnership { state: state_dtype, scaled: None, cast_before_cfg: false },
                        ForwardInputOwnership { state: state_dtype, scaled: Some(Dtype::Float16), cast_before_cfg: false },
                        ForwardInputOwnership { state: state_dtype, scaled: Some(Dtype::Float32), cast_before_cfg: true },
                    ] {
                        let owned = flow.bytes(elements(&state), crate::model::DTYPE, cfg_rows);
                        let before = mlx_rs::memory::get_active_memory() as u64;
                        // Construction temporaries have their own quantum; the assertion below
                        // inspects only retained owners AFTER its barrier, not sum-of-history peaks.
                        mlx_gen::ios_quantum::before("retained forward input unit", owned + bytes(&state).max(elements(&state) * 4))?;
                        let scaled = match flow.scaled {
                            None => state.clone(),
                            Some(dtype) => mlx_rs::ops::multiply(&state, &coefficient)?.as_dtype(dtype)?,
                        };
                        let pre_cfg = if flow.cast_before_cfg { scaled.as_dtype(crate::model::DTYPE)? } else { scaled.clone() };
                        let pair = if cfg_rows == 2 { mlx_rs::ops::concatenate_axis(&[&pre_cfg, &pre_cfg], 0)? } else { pre_cfg.clone() };
                        let forward = crate::pipeline::ios_unet_input(&pair)?;
                        mlx_gen::ios_quantum::finish("retained forward input unit", &[&scaled, &pre_cfg, &pair, &forward])?;
                        let retained = (mlx_rs::memory::get_active_memory() as u64).saturating_sub(before);
                        assert_eq!(retained, owned - bytes(&state), "{flow:?} CFGrows={cfg_rows}: distinct native owners only");
                        assert_eq!(forward.dtype(), if flow.cast_before_cfg { crate::model::DTYPE } else { flow.scaled.unwrap_or(state_dtype) });
                        assert_eq!(forward.shape()[0] as u64, cfg_rows);
                        eprintln!("FORWARD OWNERS {flow:?} rows={cfg_rows}: caller={} native_extra={retained} priced_total={owned}", bytes(&state));
                        drop((scaled, pre_cfg, pair, forward));
                    }
                }
            }
            Ok(())
        }).unwrap();
    }

    /// Exact native-counter proof at the actual allocator page boundaries. These are tiny
    /// operation/alias classes, not a model/sampler/rung measurement grid or process certificate.
    #[test]
    fn native_forward_backings_cover_page_boundaries_and_parent_views() {
        if !cpu_child("proofs::native_forward_backings_cover_page_boundaries_and_parent_views") {
            return;
        }
        use mlx_rs::ops::indexing::TryIndexOp;
        fn active() -> u64 {
            let mut bytes = 0usize;
            assert_eq!(unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) }, 0);
            bytes as u64
        }
        fn sync_cpu() -> Result<()> {
            let status = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
            if status == 0 {
                Ok(())
            } else {
                Err(mlx_gen::Error::Msg(
                    "native input fixture CPU sync failed".into(),
                ))
            }
        }
        fn verify(
            state: &Array,
            coefficient: &Array,
            flow: ForwardInputOwnership,
            rows: u64,
            parent_elements: u64,
            page: u64,
        ) -> Result<()> {
            let parent = mlx_gen::ios_quantum::fresh_metal_allocation_bytes(
                parent_elements
                    .checked_mul(dtype_bytes(state.dtype()))
                    .unwrap(),
                page,
            )
            .unwrap();
            let predicted = flow
                .native_bytes(
                    elements(state),
                    parent_elements,
                    crate::model::DTYPE,
                    rows,
                    page,
                    mlx_gen::ios_quantum::fresh_metal_allocation_bytes,
                )
                .expect("source-known fresh full parent and outputs");
            let before = active();
            mlx_gen::ios_quantum::before(
                "native unique input construction",
                predicted.checked_mul(2).unwrap(),
            )?;
            let scaled = match flow.scaled {
                None => state.clone(),
                Some(dtype) => mlx_rs::ops::multiply(state, coefficient)?.as_dtype(dtype)?,
            };
            let pre = if flow.cast_before_cfg {
                scaled.as_dtype(crate::model::DTYPE)?
            } else {
                scaled.clone()
            };
            let pair = if rows == 2 {
                mlx_rs::ops::concatenate_axis(&[&pre, &pre], 0)?
            } else {
                pre.clone()
            };
            let input = crate::pipeline::ios_unet_input(&pair)?;
            mlx_gen::ios_quantum::finish(
                "native unique input construction",
                &[&scaled, &pre, &pair, &input],
            )?;
            let retained = active()
                .checked_sub(before)
                .expect("held caller/coefficient cannot disappear");
            assert_eq!(retained, predicted.checked_sub(parent).unwrap(),
                "{flow:?} shape={:?} rows={rows} page={page} parent_elements={parent_elements}: unique physical backings", state.shape());
            assert_eq!(
                input.dtype(),
                if flow.cast_before_cfg {
                    crate::model::DTYPE
                } else {
                    flow.scaled.unwrap_or(state.dtype())
                }
            );
            eprintln!("NATIVE UNIQUE INPUT {flow:?} shape={:?} rows={rows} parent={parent} total={predicted} extra={retained} page={page}", state.shape());
            // Keep caller state, coefficient and the actual scope alive THROUGH the assertion.
            drop((scaled, pre, pair, input));
            Ok(())
        }
        with_cpu_diagnostic_headroom(PLENTY, || {
            let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
            let page = mlx_gen::ios_quantum::native_metal_page_bytes().expect("actual Darwin allocator operand");
            assert!(page.checked_mul(64).is_some_and(|n| n <= 32 * 1024 * 1024),
                "this finite fixture has a 32MiB allocation budget, not an unbounded allocator probe");
            let edge = i32::try_from(page / 8).unwrap();
            for shape in [[1, 1, edge-1, 4], [1, 1, edge, 4], [1, 1, edge+1, 4], [1, 31, 35, 4]] {
                for dtype in [Dtype::Float16, Dtype::Float32] {
                    let before_state = active();
                    let state = evaluated(rnd(&shape, 7710, 1.0).as_dtype(dtype)?);
                    sync_cpu()?;
                    let parent = mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes(&state), page).unwrap();
                    assert_eq!(active().checked_sub(before_state), Some(parent),
                        "the source-created full state must have the exact fresh native backing");
                    let coefficient = evaluated(Array::from_f32(0.75).as_dtype(dtype)?);
                    sync_cpu()?;
                    for rows in [1, 2] {
                        for flow in [
                            ForwardInputOwnership { state: dtype, scaled: None, cast_before_cfg: false },
                            ForwardInputOwnership { state: dtype, scaled: Some(Dtype::Float16), cast_before_cfg: false },
                            ForwardInputOwnership { state: dtype, scaled: Some(Dtype::Float32), cast_before_cfg: true },
                        ] { verify(&state, &coefficient, flow, rows, elements(&state), page)?; }
                    }
                    drop((state, coefficient));
                }
            }
            // A genuine offset slice/reshape keeps its FULL larger parent after that original
            // Rust handle drops. Logical nbytes or distinct data pointers do not identify backing.
            let logical_width = edge + 1;
            let parent_width = 2 * logical_width + 3;
            let before_parent = active();
            let parent = evaluated(rnd(&[1, 1, parent_width, 4], 7720, 1.0).as_dtype(Dtype::Float16)?);
            sync_cpu()?;
            let parent_elements = elements(&parent);
            let physical_parent = active().checked_sub(before_parent).unwrap();
            let view = parent.try_index((.., .., 1..1+logical_width, ..))?.reshape(&[1, 1, logical_width, 4])?;
            view.eval()?;
            sync_cpu()?;
            drop(parent);
            assert_eq!(active().checked_sub(before_parent), Some(physical_parent),
                "slice/reshape retains the full original allocation, not just its logical span");
            assert!(physical_parent > mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes(&view), page).unwrap());
            let coefficient = evaluated(Array::from_f32(0.75).as_dtype(Dtype::Float16)?);
            sync_cpu()?;
            verify(&view, &coefficient, ForwardInputOwnership {
                state: Dtype::Float16, scaled: None, cast_before_cfg: false,
            }, 1, parent_elements, page)?;
            drop((view, coefficient));
            // Conversely a broadcast can have a logical plane larger than its tiny shared parent.
            let parent = evaluated(rnd(&[1, 1, 1, 4], 7730, 1.0).as_dtype(Dtype::Float16)?);
            let broadcast = mlx_rs::ops::broadcast_to(&parent, &[1, 31, 35, 4])?;
            broadcast.eval()?;
            sync_cpu()?;
            let parent_elements = elements(&parent);
            drop(parent);
            let coefficient = evaluated(Array::from_f32(0.75).as_dtype(Dtype::Float16)?);
            sync_cpu()?;
            verify(&broadcast, &coefficient, ForwardInputOwnership {
                state: Dtype::Float16, scaled: None, cast_before_cfg: false,
            }, 1, parent_elements, page)?;
            Ok(())
        }).unwrap();
    }

    /// The frontend cast of a permuted half weight and CPU GEMM's layout copy coexist.
    /// The pre-promoted control keeps exactly the same layout, but has no pending dtype cast.
    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn mixed_conv_cast_and_layout_are_distinct_live_allocations() {
        if !cpu_child("proofs::mixed_conv_cast_and_layout_are_distinct_live_allocations") {
            return;
        }
        let x = rand(&[1, 16, 16, 128], 7601);
        let w = half_conv_weight(128, 3, 128, 7602);
        let promoted = evaluated(w.as_dtype(Dtype::Float32).unwrap());
        eprintln!(
            "CONV input {:?}/{:?}/{:?}, weight {:?}/{:?}/{:?}, promoted strides {:?}",
            x.shape(),
            x.dtype(),
            x.strides(),
            w.shape(),
            w.dtype(),
            w.strides(),
            promoted.strides()
        );
        let mixed = ConvCall::new(x.shape(), 4, &w, 1, 1);
        let ready = ConvCall::new(x.shape(), 4, &promoted, 1, 1);
        for (name, weight, call) in [("mixed", &w, mixed), ("pre-promoted", &promoted, ready)] {
            prove(name, single_conv_bytes(&call, 0, 0, Backend::Cpu), || {
                Ok(vec![mlx_gen::nn::conv2d(&x, weight, None, 1, 1)?])
            });
        }
        assert_eq!(
            mixed.transient(Backend::Cpu) - ready.transient(Backend::Cpu),
            elements(&w) * dtype_bytes(Dtype::Float32),
            "frontend weight cast only"
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::mixed_conv_cast_and_layout_are_distinct_live_allocations");
        }
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn affine_norm_fallback_tracks_both_compute_widths() {
        if !cpu_child("proofs::affine_norm_fallback_tracks_both_compute_widths") {
            return;
        }
        for dtype in [Dtype::Float16, Dtype::Float32] {
            let x = evaluated(rnd(&[2, 8, 64], 7610, 1.0).as_dtype(dtype).unwrap());
            let w = evaluated(Array::ones::<f32>(&[64]).unwrap().as_dtype(dtype).unwrap());
            let b = evaluated(rnd(&[64], 7611, 0.02).as_dtype(dtype).unwrap());
            eprintln!("NORM {:?}/{:?}/{:?}", x.shape(), x.dtype(), x.strides());
            prove(
                &format!("affine norm {dtype:?}"),
                final_norm_bytes(x.shape(), dtype, &w, &b),
                || {
                    Ok(vec![mlx_rs::fast::layer_norm(
                        &x,
                        Some(&w),
                        Some(&b),
                        1e-5,
                    )?])
                },
            );
        }

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::affine_norm_fallback_tracks_both_compute_widths");
        }
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn biased_projection_prices_all_addmm_operand_promotions() {
        if !cpu_child("proofs::biased_projection_prices_all_addmm_operand_promotions") {
            return;
        }
        use mlx_gen::adapters::AdaptableLinear;
        let x = half(&rand(&[1, 8, 32], 7620));
        let w = half(&rand(&[64, 32], 7621));
        let b = rand(&[64], 7622);
        let p = AdaptableLinear::dense(w.clone(), Some(b));
        let ready_x = evaluated(x.as_dtype(Dtype::Float32).unwrap());
        let ready_w = evaluated(w.as_dtype(Dtype::Float32).unwrap());
        let ready = AdaptableLinear::dense(ready_w, Some(rand(&[64], 7622)));
        assert_eq!(
            linear_output_dtype(&p, x.dtype()),
            Dtype::Float32,
            "bias participates in addmm result_type"
        );
        let out_bytes = 8 * 64 * 4;
        prove(
            "mixed biased addmm",
            linear_operand_bytes(&p, x.shape(), x.dtype()) + out_bytes,
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        p.forward(&x),
                    )
                        .1?,
                ])
            },
        );
        prove("pre-promoted biased addmm", out_bytes, || {
            Ok(vec![
                (
                    crate::test_device::record_model_inference_device(),
                    ready.forward(&ready_x),
                )
                    .1?,
            ])
        });
        assert_eq!(
            linear_operand_bytes(&p, x.shape(), x.dtype()),
            (elements(&x) + elements(&w)) * 4,
            "activation and weight casts, no invented bias-add output"
        );
        assert_eq!(
            linear_operand_bytes(&ready, ready_x.shape(), ready_x.dtype()),
            0
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::biased_projection_prices_all_addmm_operand_promotions");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::biased_projection_prices_all_addmm_operand_promotions");
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn control_injection_prices_lazy_and_materialized_owned_residuals() {
        if !cpu_child("proofs::control_injection_prices_lazy_and_materialized_owned_residuals") {
            return;
        }
        for dtype in [Dtype::Float16, Dtype::Float32] {
            let x = evaluated(rnd(&[1, 4, 4, 32], 7630, 1.0).as_dtype(dtype).unwrap());
            let source = rand(x.shape(), 7631);
            let ready = half(&source);
            let scratch = control_injection_bytes(&x, &ready);
            for lazy in [false, true] {
                prove(
                    &format!("control owned {dtype:?} lazy={lazy}"),
                    scratch,
                    || {
                        let c = if lazy {
                            source.as_dtype(Dtype::Float16)?
                        } else {
                            ready.clone()
                        };
                        let control = crate::unet::ControlResiduals {
                            down: vec![],
                            mid: c,
                        };
                        let y =
                            crate::unet::inject_control_mid_residual(x.clone(), Some(&control))?;
                        y.eval()?;
                        // Production retains ControlResiduals through the entire U-Net forward.
                        drop(control);
                        Ok(vec![y])
                    },
                );
            }
            assert_eq!(scratch, elements(&x) * (2 + dtype_bytes(dtype)));
        }

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::control_injection_prices_lazy_and_materialized_owned_residuals");
        }
    }

    #[ignore = "model inference; explicit model-proof workflow"]
    #[test]
    fn decode_input_dtype_alias_and_retention_are_priced_without_narrowing_state() {
        if !cpu_child(
            "proofs::decode_input_dtype_alias_and_retention_are_priced_without_narrowing_state",
        ) {
            return;
        }
        let (mut weights, cfg) = crate::vae::tests::fixture();
        weights.cast_all(Dtype::Float16).unwrap();
        let vae = crate::vae::Autoencoder::from_weights(&weights, &cfg).unwrap();
        let shape = [1, 8, 8, 4];
        let x32 = rand(&shape, 7640);
        let x16 = half(&x32);
        assert_eq!(vae.dtype(), Dtype::Float16);
        assert_eq!(
            (
                crate::test_device::record_model_inference_device(),
                vae.decode_input_retained_bytes(&x16)
            )
                .1,
            bytes(&x16),
            "compute input aliases caller"
        );
        assert_eq!(
            (
                crate::test_device::record_model_inference_device(),
                vae.decode_input_retained_bytes(&x32)
            )
                .1,
            bytes(&x32) + bytes(&x16),
            "caller plus distinct compute cast"
        );
        let euler = crate::sampler::EulerSampler::new_with_dtype(
            &crate::config::DiffusionConfig::sdxl_base(),
            false,
            Dtype::Float16,
        )
        .unwrap();
        let noised = euler.add_noise_with(&x16, &x32, 800.0).unwrap();
        assert_eq!(
            noised.dtype(),
            Dtype::Float32,
            "sampler state must NOT be narrowed to fix accounting"
        );
        for caller in [&x16, &x32] {
            let mut quanta = Vec::new();
            vae.price_decode(caller, None, &mut quanta).unwrap();
            let (decoded, events) = observe(PLENTY, || {
                (
                    crate::test_device::record_model_inference_device(),
                    vae.decode(caller),
                )
                    .1
            });
            decoded.unwrap().eval().unwrap();
            let actual = checked(&events, VAE_DECODE_IN);
            assert_eq!(
                actual,
                [quanta[0].1],
                "first guard and header price include the same pending cast"
            );
            assert_eq!(
                caller.dtype(),
                if std::ptr::eq(caller, &x16) {
                    Dtype::Float16
                } else {
                    Dtype::Float32
                }
            );
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::decode_input_dtype_alias_and_retention_are_priced_without_narrowing_state");
    }

    // -- U-Net ---------------------------------------------------------------------------------

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn unet_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::unet_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED;
        let mut g = Gen::new(7000);
        g.resnet("plain", 64, 64, Some(128));
        g.resnet("up", 128, 64, Some(128));
        let w = g.weights();
        let temb = rand(&[1, 128], 1);
        let x = rand(&[1, 32, 32, 64], 2);
        let skip = rand(&[1, 32, 32, 64], 3);

        // Resnet: the down/mid shape and the up path's skip concatenation.
        let plain = crate::unet::ResnetBlock2D::from_weights(&w, "plain").unwrap();
        prove(UNET_RESNET, plain.quantum_scratch(&x, Some(&temb)), || {
            Ok(vec![
                (
                    crate::test_device::record_model_inference_device(),
                    plain.forward_quantum(&x, None, Some(&temb), plan),
                )
                    .1?,
            ])
        });
        let up = crate::unet::ResnetBlock2D::from_weights(&w, "up").unwrap();
        prove(
            &format!("{UNET_RESNET} (skip concat)"),
            up.quantum_scratch(&x, Some(&temb)),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        up.forward_quantum(&x, Some(skip.clone()), Some(&temb), plan),
                    )
                        .1?,
                ])
            },
        );

        // Downsample (stride-2 conv) and upsample (nearest-2x + conv), as `UNetBlock2D` runs them.
        let wd = conv_weight(64, 3, 64, 4);
        let bd = rand(&[64], 5);
        let call = ConvCall::new(x.shape(), 4, &wd, 2, 1);
        prove(UNET_DOWNSAMPLE, single_conv_bytes(&call, 0, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(&x, &wd, Some(&bd), 2, 1)?])
        });
        let small = rand(&[1, 16, 16, 64], 6);
        let call = ConvCall::new(&[1, 32, 32, 64], 4, &wd, 1, 1);
        prove(UNET_UPSAMPLE, single_conv_bytes(&call, 1, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(
                &mlx_gen::nn::upsample_nearest(&small, 2)?,
                &wd,
                Some(&bd),
                1,
                1,
            )?])
        });

        // conv_out head: GroupNorm + SiLU + conv to 4 channels.
        let (gw, gb) = (
            evaluated(Array::ones::<f32>(&[64]).unwrap()),
            rand(&[64], 7),
        );
        let wo = conv_weight(4, 3, 64, 8);
        let bo = rand(&[4], 9);
        let call = ConvCall::new(x.shape(), 4, &wo, 1, 1);
        prove(UNET_CONV_OUT, norm_conv_bytes(&call, cpu), || {
            let y = mlx_gen::nn::group_norm(&x, &gw, &gb, 32, 1e-5)?;
            let y = crate::silu_glue(&y)?;
            Ok(vec![mlx_gen::nn::conv2d(&y, &wo, Some(&bo), 1, 1)?])
        });

        // Stem: conv_in + the time/text_time embeddings at the guided pair's batch, over the
        // curated route's lazily scaled and CFG-concatenated latent.
        let unet = crate::test_support::tiny_unet(7).unwrap();
        let (latent, ctx, pooled, ids) = unet_inputs(2, 32, 32);
        let one = rand(&[1, 32, 32, 4], 10);
        let lazy_x = || -> Result<Array> {
            let scaled = mlx_rs::ops::multiply(&one, Array::from_f32(0.5))?;
            Ok(mlx_rs::ops::concatenate_axis(&[&scaled, &scaled], 0)?)
        };
        prove(UNET_STEM, unet.stem_scratch_for_test(&latent, &ctx), || {
            let (x, temb, _) = unet.stem(
                &lazy_x()?,
                500.0,
                &ctx,
                &pooled,
                &ids,
                plan,
                2,
                Dtype::Float32,
            )?;
            Ok(vec![x, temb])
        });

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::unet_quanta_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::unet_quanta_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn transformer_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::transformer_quanta_track_their_measured_peaks") {
            return;
        }
        let (dim, ctx_dim, ff, heads) = (128, 64, 512, 2);
        let mut g = Gen::new(7100);
        g.block("b", dim, ctx_dim, ff);
        g.ln("t.norm", dim);
        g.linear("t.proj_in", dim, dim, true);
        g.linear("t.proj_out", dim, dim, true);
        g.block("t.transformer_blocks.0", dim, ctx_dim, ff);
        let w = g.weights();
        let block =
            crate::unet::transformer::TransformerBlock::from_weights(&w, "b", dim, heads).unwrap();
        let y = rand(&[1, 256, dim], 1);
        let memory = rand(&[1, 7, ctx_dim], 2);
        let pair_memory = rand(&[2, 7, ctx_dim], 3);
        let unbounded = crate::plan::SdxlForwardPlan::UNBOUNDED;

        // Batched / plain forward: one latent row, the text memory at that row.
        prove(
            TRANSFORMER_BLOCK,
            block.quantum_scratch(1, 1, 256, 7, false, Dtype::Float32, unbounded.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &memory, None, unbounded),
                    )
                        .1?,
                ])
            },
        );
        // Sequential CFG half: the latent at ONE row, the text k/v projected at the pair's two.
        let half = unbounded.with_cfg_row(Some(1));
        prove(
            &format!("{TRANSFORMER_BLOCK} (sequential CFG half)"),
            block.quantum_scratch(1, 2, 256, 7, false, Dtype::Float32, half.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &pair_memory, None, half),
                    )
                        .1?,
                ])
            },
        );
        // Bounded attention: 64 query rows per self-attention chunk, each chunk evaluated.
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(2 * 64 * 256, true);
        let bounded = crate::plan::SdxlForwardPlan::with_attention(
            mlx_gen::attention::AttentionPlan::budgeted(budget),
        );
        assert_eq!(budget.query_block(1, heads, 256, 256), 64);
        prove(
            &format!("{TRANSFORMER_BLOCK} (bounded attention)"),
            block.quantum_scratch(1, 1, 256, 7, false, Dtype::Float32, bounded.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &memory, None, bounded),
                    )
                        .1?,
                ])
            },
        );

        // proj_in / proj_out of a Transformer2D over a 16x16 grid.
        let t =
            crate::unet::transformer::Transformer2D::from_weights(&w, "t", dim, heads, 1).unwrap();
        let x = rand(&[1, 16, 16, dim], 4);
        let (proj_in, proj_out) = t.projection_scratch(&x);
        prove(TRANSFORMER_PROJ_IN, proj_in, || {
            Ok(vec![t.proj_in_body(&x)?])
        });
        let hidden = rand(&[1, 256, dim], 5);
        prove(TRANSFORMER_PROJ_OUT, proj_out, || {
            Ok(vec![t.proj_out_body(&hidden, &x)?])
        });

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::transformer_quanta_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::transformer_quanta_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn controlnet_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::controlnet_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        // The conditioning embedding's first (full-resolution, 3 -> 16) and a stride-2 (16 -> 32)
        // convolution, each with its SiLU epilogue.
        let image = rand(&[1, 64, 64, 3], 1);
        let w_in = conv_weight(16, 3, 3, 2);
        let b_in = rand(&[16], 3);
        prove(
            CONTROL_COND_CONV,
            single_conv_bytes(&ConvCall::new(image.shape(), 4, &w_in, 1, 1), 0, 1, cpu),
            || {
                Ok(vec![mlx_gen::nn::silu(&mlx_gen::nn::conv2d(
                    &image,
                    &w_in,
                    Some(&b_in),
                    1,
                    1,
                )?)?])
            },
        );
        let e = rand(&[1, 64, 64, 16], 4);
        let w_dn = conv_weight(32, 3, 16, 5);
        let b_dn = rand(&[32], 6);
        prove(
            &format!("{CONTROL_COND_CONV} (stride 2)"),
            single_conv_bytes(&ConvCall::new(e.shape(), 4, &w_dn, 2, 1), 0, 1, cpu),
            || {
                Ok(vec![mlx_gen::nn::silu(&mlx_gen::nn::conv2d(
                    &e,
                    &w_dn,
                    Some(&b_dn),
                    2,
                    1,
                )?)?])
            },
        );
        // A 1x1 zero-conv head, scaled.
        let r = rand(&[1, 32, 32, 64], 7);
        let wz = conv_weight(64, 1, 64, 8);
        let bz = rand(&[64], 9);
        let scale = Array::from_f32(0.8);
        prove(
            CONTROL_ZERO_CONV,
            single_conv_bytes(&ConvCall::new(r.shape(), 4, &wz, 1, 0), 0, 0, cpu),
            || {
                Ok(vec![mlx_rs::ops::multiply(
                    &mlx_gen::nn::conv2d(&r, &wz, Some(&bz), 1, 0)?,
                    &scale,
                )?])
            },
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::controlnet_quanta_track_their_measured_peaks");
        }
    }

    // -- f16 activations (the device's compute dtype) -------------------------------------------
    //
    // The device runs the U-Net, its ControlNets and both CLIP encoders at `model::DTYPE` (fp16):
    // every dense float leaf loads as f16 (`loader::cast_for_load`; the SceneWorks snapshots store
    // the U-Net norms, convs and dense linears as F16 already) and only a packed snapshot's
    // `.scales`/`.biases` keep their stored BF16. These proofs measure the same quanta at f16
    // activations over f16 weights, so every f16 -> f32 promotion inside them (the f32 sinusoidal
    // timestep table, the f32 im2col of the CPU convolution, the composed norms' f32 statistics,
    // the softmax) is inside the measurement, not assumed away.

    /// Every float tensor of `w` at f16 (production's dense load), materialized.
    fn half_weights(w: Weights) -> Weights {
        let mut w = w;
        w.cast_all(Dtype::Float16).unwrap();
        w.materialize().unwrap();
        w
    }

    fn half(a: &Array) -> Array {
        evaluated(a.as_dtype(Dtype::Float16).unwrap())
    }

    /// [`conv_weight`] stored f16, as the U-Net's convs load.
    fn half_conv_weight(out: i32, k: i32, inp: i32, seed: u64) -> Array {
        let nchw = half(&rnd(
            &[out, inp, k, k],
            seed,
            (1.0 / (inp * k * k) as f32).sqrt(),
        ));
        crate::unet::nchw_to_nhwc(&nchw).unwrap()
    }

    /// Re-store every dense linear `{p}.weight` of `w` named by `linears` as the packed Q4 triple a
    /// pre-quantized SceneWorks snapshot carries: u32 codes, BF16 `.scales`/`.biases`.
    fn packed(mut map: HashMap<String, Array>, linears: &[String]) -> Weights {
        for p in linears {
            let dense = map.remove(&format!("{p}.weight")).unwrap();
            let (wq, scales, biases) =
                mlx_rs::ops::quantize(&dense, crate::quant::GROUP_SIZE, 4).unwrap();
            let bf16 = |a: Array| evaluated(a.as_dtype(Dtype::Bfloat16).unwrap());
            map.insert(format!("{p}.weight"), evaluated(wq));
            map.insert(format!("{p}.scales"), bf16(scales));
            map.insert(format!("{p}.biases"), bf16(biases));
        }
        Weights::from_map(map)
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn f16_unet_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::f16_unet_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED;
        let mut g = Gen::new(8000);
        g.resnet("plain", 64, 64, Some(128));
        g.resnet("up", 128, 64, Some(128));
        let w = half_weights(g.weights());
        let temb = half(&rnd(&[1, 128], 1, 1.0));
        let x = half(&rnd(&[1, 32, 32, 64], 2, 1.0));
        let skip = half(&rnd(&[1, 32, 32, 64], 3, 1.0));

        let plain = crate::unet::ResnetBlock2D::from_weights(&w, "plain").unwrap();
        let out = (
            crate::test_device::record_model_inference_device(),
            plain.forward_quantum(&x, None, Some(&temb), plan),
        )
            .1
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Float16, "the f16 resnet stays f16");
        prove(
            &format!("{UNET_RESNET} (f16)"),
            plain.quantum_scratch(&x, Some(&temb)),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        plain.forward_quantum(&x, None, Some(&temb), plan),
                    )
                        .1?,
                ])
            },
        );
        let up = crate::unet::ResnetBlock2D::from_weights(&w, "up").unwrap();
        prove(
            &format!("{UNET_RESNET} (f16, skip concat)"),
            up.quantum_scratch(&x, Some(&temb)),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        up.forward_quantum(&x, Some(skip.clone()), Some(&temb), plan),
                    )
                        .1?,
                ])
            },
        );

        let wd = half_conv_weight(64, 3, 64, 4);
        let bd = half(&rnd(&[64], 5, 0.02));
        let call = ConvCall::new(x.shape(), 2, &wd, 2, 1);
        prove(
            &format!("{UNET_DOWNSAMPLE} (f16)"),
            single_conv_bytes(&call, 0, 0, cpu),
            || Ok(vec![mlx_gen::nn::conv2d(&x, &wd, Some(&bd), 2, 1)?]),
        );
        let small = half(&rnd(&[1, 16, 16, 64], 6, 1.0));
        let call = ConvCall::new(&[1, 32, 32, 64], 2, &wd, 1, 1);
        prove(
            &format!("{UNET_UPSAMPLE} (f16)"),
            single_conv_bytes(&call, 1, 0, cpu),
            || {
                Ok(vec![mlx_gen::nn::conv2d(
                    &mlx_gen::nn::upsample_nearest(&small, 2)?,
                    &wd,
                    Some(&bd),
                    1,
                    1,
                )?])
            },
        );
        let (gw, gb) = (
            half(&Array::ones::<f32>(&[64]).unwrap()),
            half(&rnd(&[64], 7, 0.02)),
        );
        let wo = half_conv_weight(4, 3, 64, 8);
        let bo = half(&rnd(&[4], 9, 0.02));
        let call = ConvCall::new(x.shape(), 2, &wo, 1, 1);
        prove(
            &format!("{UNET_CONV_OUT} (f16)"),
            norm_conv_bytes(&call, cpu),
            || {
                let y = mlx_gen::nn::group_norm(&x, &gw, &gb, 32, 1e-5)?;
                let y = crate::silu_glue(&y)?;
                Ok(vec![mlx_gen::nn::conv2d(&y, &wo, Some(&bo), 1, 1)?])
            },
        );

        // Stem at the f16 compute dtype: the f32 sinusoidal tables cast to f16, the f16 embedding
        // MLPs, conv_in over the curated route's lazily scaled and CFG-concatenated f16 latent.
        let mut unet = crate::test_support::tiny_unet(7).unwrap();
        unet.cast_weights(Dtype::Float16).unwrap();
        let (latent, ctx, pooled, ids) = unet_inputs(2, 32, 32);
        let (latent, ctx, pooled) = (half(&latent), half(&ctx), half(&pooled));
        let one = half(&rnd(&[1, 32, 32, 4], 10, 1.0));
        let lazy_x = || -> Result<Array> {
            let scaled =
                mlx_rs::ops::multiply(&one, Array::from_f32(0.5).as_dtype(Dtype::Float16)?)?;
            Ok(mlx_rs::ops::concatenate_axis(&[&scaled, &scaled], 0)?)
        };
        let (stem_x, stem_temb, _) = unet
            .stem(
                &lazy_x().unwrap(),
                500.0,
                &ctx,
                &pooled,
                &ids,
                plan,
                2,
                Dtype::Float16,
            )
            .unwrap();
        assert_eq!(
            (stem_x.dtype(), stem_temb.dtype()),
            (Dtype::Float16, Dtype::Float16)
        );
        prove(
            &format!("{UNET_STEM} (f16)"),
            unet.stem_scratch_for_test(&latent, &ctx),
            || {
                let (x, temb, _) = unet.stem(
                    &lazy_x()?,
                    500.0,
                    &ctx,
                    &pooled,
                    &ids,
                    plan,
                    2,
                    Dtype::Float16,
                )?;
                Ok(vec![x, temb])
            },
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::f16_unet_quanta_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::f16_unet_quanta_track_their_measured_peaks");
    }

    /// The packed tier's promotions: a pre-quantized snapshot's `time_emb_proj` (BF16 scales) turns
    /// the f16 resnet's second half f32 (the projected f32 time embedding widens the sum); after it
    /// the U-Net carries f32 activations through f16 convolutions, whose weights MLX casts up.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn packed_tier_resnet_and_conv_promotions_track_their_measured_peaks() {
        if !cpu_child("proofs::packed_tier_resnet_and_conv_promotions_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED;
        let mut g = Gen::new(8300);
        g.resnet("r", 64, 64, Some(128));
        let mut map = g.map;
        for v in map.values_mut() {
            *v = half(v);
        }
        let w = packed(map, &["r.time_emb_proj".to_string()]);
        let resnet = crate::unet::ResnetBlock2D::from_weights(&w, "r").unwrap();
        let x16 = half(&rnd(&[1, 32, 32, 64], 1, 1.0));
        let temb32 = evaluated(rnd(&[1, 128], 2, 1.0));
        let out = (
            crate::test_device::record_model_inference_device(),
            resnet.forward_quantum(&x16, None, Some(&temb32), plan),
        )
            .1
            .unwrap();
        assert_eq!(
            out.dtype(),
            Dtype::Float32,
            "the packed temb widens the resnet"
        );
        prove(
            &format!("{UNET_RESNET} (f16 in, packed f32 temb)"),
            resnet.quantum_scratch(&x16, Some(&temb32)),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        resnet.forward_quantum(&x16, None, Some(&temb32), plan),
                    )
                        .1?,
                ])
            },
        );
        // The promoted half is priced at f32 (on the CPU the GroupNorms' f32 working copies hide
        // the difference in the measurement above; on Metal they do not exist).
        let conv = |k: &str| crate::unet::nchw_to_nhwc(w.require(k).unwrap()).unwrap();
        let (c1, c2) = (conv("r.conv1.weight"), conv("r.conv2.weight"));
        assert_eq!(
            resnet.quantum_scratch(&x16, Some(&temb32)),
            resnet_bytes(x16.shape(), 2, 4, &c1, &c2, Backend::Cpu)
                // packed temb bias, norm2 weight/bias, conv2 bias: four promoted channel vectors.
                + 4 * c2.shape()[0] as u64 * dtype_bytes(Dtype::Float32)
                // native qmm casts the packed BF16 scales and offsets ([out, in/group] each) of
                // `time_emb_proj` to its f32 compute dtype.
                + 2 * (c2.shape()[0] as u64)
                    * (temb32.shape()[1] / crate::quant::GROUP_SIZE) as u64
                    * dtype_bytes(Dtype::Float32)
        );
        assert!(
            resnet_bytes(x16.shape(), 2, 4, &c1, &c2, Backend::Metal)
                > resnet_bytes(x16.shape(), 2, 2, &c1, &c2, Backend::Metal)
        );
        // Downstream: f32 activations through the f16 resnet's convolutions.
        let x32 = evaluated(rnd(&[1, 32, 32, 64], 3, 1.0));
        prove(
            &format!("{UNET_RESNET} (f32 activations, f16 convs)"),
            resnet.quantum_scratch(&x32, Some(&temb32)),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        resnet.forward_quantum(&x32, None, Some(&temb32), plan),
                    )
                        .1?,
                ])
            },
        );
        let wo = half_conv_weight(4, 3, 64, 4);
        let bo = half(&rnd(&[4], 5, 0.02));
        let (gw, gb) = (
            half(&Array::ones::<f32>(&[64]).unwrap()),
            half(&rnd(&[64], 6, 0.02)),
        );
        let call = ConvCall::new(x32.shape(), 4, &wo, 1, 1);
        prove(
            &format!("{UNET_CONV_OUT} (f32 activations, f16 conv)"),
            norm_conv_bytes(&call, cpu),
            || {
                let y = mlx_gen::nn::group_norm(&x32, &gw, &gb, 32, 1e-5)?;
                let y = crate::silu_glue(&y)?;
                Ok(vec![mlx_gen::nn::conv2d(&y, &wo, Some(&bo), 1, 1)?])
            },
        );
        // GPU AsType also preserves the dense transposed layout. Mixed precision holds a
        // distinct f32 frontend cast AND an f32 layout copy, versus one f16 layout copy.
        let narrow = ConvCall::new(x16.shape(), 2, &wo, 1, 1);
        assert_eq!(
            call.transient(Backend::Metal) - narrow.transient(Backend::Metal),
            call.weight_elements() * (4 + 4 - 2)
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::packed_tier_resnet_and_conv_promotions_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::packed_tier_resnet_and_conv_promotions_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn composed_conv_bias_promotion_is_separate_from_weight_layout() {
        if !cpu_child("proofs::composed_conv_bias_promotion_is_separate_from_weight_layout") {
            return;
        }
        let x = rand(&[1, 8, 8, 64], 7660);
        let b = half(&rand(&[64], 7661));
        let w = half_conv_weight(64, 3, 64, 7662);
        let casts = conv_bias_bytes(x.dtype(), &w, Some(&b), elements(&x));
        assert_eq!(
            casts,
            elements(&b) * 4,
            "only the bias vector needs promotion"
        );
        prove(
            "composed convolution bias promotion",
            bytes(&x) + casts,
            || Ok(vec![mlx_rs::ops::add(&x, &b)?]),
        );
        let x16 = half(&x);
        let b32 = rand(&[64], 7661);
        let widened = conv_bias_bytes(x16.dtype(), &w, Some(&b32), elements(&x16));
        assert_eq!(
            widened,
            elements(&x16) * 4,
            "wider output, not a second weight copy"
        );
        prove("composed convolution wider bias", widened, || {
            Ok(vec![mlx_rs::ops::add(&x16, &b32)?])
        });

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::composed_conv_bias_promotion_is_separate_from_weight_layout");
        }
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn resnet_affine_bias_and_shortcut_promotions_are_tensor_derived() {
        if !cpu_child("proofs::resnet_affine_bias_and_shortcut_promotions_are_tensor_derived") {
            return;
        }
        let (cin, cout) = (64, 32);
        let mut g = Gen::new(7650);
        g.resnet("r", cin, cout, None);
        let mut map = g.map;
        for value in map.values_mut() {
            *value = half(value);
        }
        let weights = Weights::from_map(map);
        let r = crate::unet::ResnetBlock2D::from_weights(&weights, "r").unwrap();
        let x = rand(&[1, 8, 8, cin], 7651);
        // norm1's two cin vectors, both conv biases + norm2's two cout vectors, and the
        // shortcut addmm's cout×cin weight and cout bias, all promoted from f16 to f32.
        let casts = (2 * cin + 5 * cout + cin * cout) as u64 * 4;
        assert_eq!(r.promotion_scratch(&x, None), casts);
        let c1 = crate::unet::nchw_to_nhwc(weights.require("r.conv1.weight").unwrap()).unwrap();
        let c2 = crate::unet::nchw_to_nhwc(weights.require("r.conv2.weight").unwrap()).unwrap();
        assert_eq!(
            r.quantum_scratch(&x, None),
            resnet_bytes(x.shape(), 4, 4, &c1, &c2, Backend::Cpu) + casts
        );
        prove(
            "resnet affine/bias/shortcut",
            r.quantum_scratch(&x, None),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        r.forward_quantum(&x, None, None, crate::plan::SdxlForwardPlan::UNBOUNDED),
                    )
                        .1?,
                ])
            },
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::resnet_affine_bias_and_shortcut_promotions_are_tensor_derived");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::resnet_affine_bias_and_shortcut_promotions_are_tensor_derived");
    }

    /// Both CLIP text encoders run f16 on the device; a packed snapshot's BF16 scales make each
    /// layer's projections, and so its residual stream, f32.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn f16_and_packed_clip_layers_track_their_measured_peaks() {
        if !cpu_child("proofs::f16_and_packed_clip_layers_track_their_measured_peaks") {
            return;
        }
        let cfg = ClipTextConfig {
            num_layers: 1,
            model_dims: 128,
            num_heads: 2,
            max_length: 77,
            vocab_size: 40,
            projection_dim: None,
            hidden_act: ClipActivation::Gelu,
        };
        let mut g = Gen::new(8400);
        g.put(
            "text_model.embeddings.token_embedding.weight".into(),
            &[40, 128],
            0.5,
        );
        g.put(
            "text_model.embeddings.position_embedding.weight".into(),
            &[77, 128],
            0.1,
        );
        g.clip_layers("text_model", 1, 128, 3 * 128);
        g.ln("text_model.final_layer_norm", 128);
        let mut map = g.map;
        for v in map.values_mut() {
            *v = half(v);
        }
        let x = half(&rnd(&[2, 77, 128], 1, 1.0));
        let dense = crate::text_encoder::ClipTextEncoder::from_weights(
            &Weights::from_map(map.clone()),
            "text_model",
            &cfg,
        )
        .unwrap();
        let (scratch, body) = dense.layer_quantum_for_test(0, &x);
        assert_eq!(body().unwrap().dtype(), Dtype::Float16);
        prove(&format!("{CLIP_LAYER} (f16)"), scratch, || {
            Ok(vec![body()?])
        });
        let linears: Vec<String> = ["q_proj", "k_proj", "v_proj", "out_proj"]
            .iter()
            .map(|p| format!("text_model.encoder.layers.0.self_attn.{p}"))
            .chain(
                ["fc1", "fc2"]
                    .iter()
                    .map(|p| format!("text_model.encoder.layers.0.mlp.{p}")),
            )
            .collect();
        let packed_te = crate::text_encoder::ClipTextEncoder::from_weights(
            &packed(map, &linears),
            "text_model",
            &cfg,
        )
        .unwrap();
        let (scratch, body) = packed_te.layer_quantum_for_test(0, &x);
        assert_eq!(
            body().unwrap().dtype(),
            Dtype::Float32,
            "the packed layer runs f32"
        );
        prove(
            &format!("{CLIP_LAYER} (f16 in, packed Q4 / BF16 scales)"),
            scratch,
            || Ok(vec![body()?]),
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::f16_and_packed_clip_layers_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::f16_and_packed_clip_layers_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn f16_transformer_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::f16_transformer_quanta_track_their_measured_peaks") {
            return;
        }
        let (dim, ctx_dim, ff, heads) = (128, 64, 512, 2);
        let mut g = Gen::new(8100);
        g.block("b", dim, ctx_dim, ff);
        g.ln("t.norm", dim);
        g.linear("t.proj_in", dim, dim, true);
        g.linear("t.proj_out", dim, dim, true);
        let linears: Vec<String> = ["attn1", "attn2"]
            .iter()
            .flat_map(|a| {
                ["to_q", "to_k", "to_v", "to_out.0"]
                    .iter()
                    .map(move |p| format!("b.{a}.{p}"))
            })
            .chain(["b.ff.net.0.proj".into(), "b.ff.net.2".into()])
            .collect();
        let mut map = g.map;
        for v in map.values_mut() {
            *v = half(v);
        }
        let w = Weights::from_map(map.clone());
        let mut block =
            crate::unet::transformer::TransformerBlock::from_weights(&w, "b", dim, heads).unwrap();
        let y = half(&rnd(&[1, 256, dim], 1, 1.0));
        let memory = half(&rnd(&[1, 7, ctx_dim], 2, 1.0));
        let pair_memory = half(&rnd(&[2, 7, ctx_dim], 3, 1.0));
        let long_memory = half(&rnd(&[2, 308, ctx_dim], 4, 1.0));
        let unbounded = crate::plan::SdxlForwardPlan::UNBOUNDED;
        let out = (
            crate::test_device::record_model_inference_device(),
            block.forward_ip(&y, &memory, None, unbounded),
        )
            .1
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Float16, "the f16 block stays f16");

        prove(
            &format!("{TRANSFORMER_BLOCK} (f16)"),
            block.quantum_scratch(1, 1, 256, 7, false, Dtype::Float16, unbounded.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &memory, None, unbounded),
                    )
                        .1?,
                ])
            },
        );
        let half_plan = unbounded.with_cfg_row(Some(1));
        prove(
            &format!("{TRANSFORMER_BLOCK} (f16, sequential CFG half)"),
            block.quantum_scratch(1, 2, 256, 7, false, Dtype::Float16, half_plan.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &pair_memory, None, half_plan),
                    )
                        .1?,
                ])
            },
        );
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(2 * 64 * 256, true);
        let bounded = crate::plan::SdxlForwardPlan::with_attention(
            mlx_gen::attention::AttentionPlan::budgeted(budget),
        );
        prove(
            &format!("{TRANSFORMER_BLOCK} (f16, bounded attention)"),
            block.quantum_scratch(1, 1, 256, 7, false, Dtype::Float16, bounded.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&y, &memory, None, bounded),
                    )
                        .1?,
                ])
            },
        );

        // Cross-attention-dominated: 256 latent tokens against a four-window long prompt's 308 text
        // keys at the guided pair's batch + 16 IP tokens (sequential CFG half, IP branch on), so the
        // cross-attention phase is the block's widest.
        let ip_k = half(&rnd(&[dim, ctx_dim], 5, 0.1));
        let ip_v = half(&rnd(&[dim, ctx_dim], 6, 0.1));
        block.set_ip_projections(
            mlx_gen::adapters::AdaptableLinear::dense(ip_k, None),
            mlx_gen::adapters::AdaptableLinear::dense(ip_v, None),
        );
        let few = half(&rnd(&[1, 256, dim], 7, 1.0));
        let tokens = half(&rnd(&[2, 16, ctx_dim], 8, 1.0));
        let cross = block.quantum_scratch(
            1,
            2,
            256,
            308 + 16,
            true,
            Dtype::Float16,
            half_plan.attention,
        );
        prove(
            &format!("{TRANSFORMER_BLOCK} (f16, cross-attention + IP, CFG half)"),
            cross,
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        block.forward_ip(&few, &long_memory, Some((&tokens, 0.6)), half_plan),
                    )
                        .1?,
                ])
            },
        );

        // proj_in / proj_out at f16.
        let mut tmap = map.clone();
        tmap.retain(|k, _| k.starts_with("t."));
        let mut g = Gen::new(8200);
        g.block("t.transformer_blocks.0", dim, ctx_dim, ff);
        for (k, v) in g.map {
            tmap.insert(k, half(&v));
        }
        let t = crate::unet::transformer::Transformer2D::from_weights(
            &Weights::from_map(tmap),
            "t",
            dim,
            heads,
            1,
        )
        .unwrap();
        let x = half(&rnd(&[1, 16, 16, dim], 9, 1.0));
        let (proj_in, proj_out) = t.projection_scratch(&x);
        prove(&format!("{TRANSFORMER_PROJ_IN} (f16)"), proj_in, || {
            Ok(vec![t.proj_in_body(&x)?])
        });
        let hidden = half(&rnd(&[1, 256, dim], 10, 1.0));
        prove(&format!("{TRANSFORMER_PROJ_OUT} (f16)"), proj_out, || {
            Ok(vec![t.proj_out_body(&hidden, &x)?])
        });

        // A pre-quantized (Q4) snapshot's block: u32 codes with BF16 scales/biases against f16
        // activations — the device's packed tier.
        let packed_w = packed(map, &linears);
        let packed_block =
            crate::unet::transformer::TransformerBlock::from_weights(&packed_w, "b", dim, heads)
                .unwrap();
        // MLX promotes the f16 activations through the BF16-scaled projections: the block runs f32.
        let out = (
            crate::test_device::record_model_inference_device(),
            packed_block.forward_ip(&y, &memory, None, unbounded),
        )
            .1
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Float32);
        assert_eq!(packed_block.activation_bytes(Dtype::Float16), 4);
        assert_eq!(block.activation_bytes(Dtype::Float16), 2);
        // Entered at f16, its self-attention q/k/v each cast the f16 LayerNorm output to f32
        // (three hidden-sized f32 copies an f32 entry does not make; on the CPU the measurement
        // below does not separate them from the f32 block's other buffers).
        let scratch = |x| packed_block.quantum_scratch(1, 1, 256, 7, false, x, unbounded.attention);
        assert_eq!(
            scratch(Dtype::Float16) - scratch(Dtype::Float32),
            3 * 256 * dim as u64 * 4 + 3 * PAGE_BYTES
        );
        prove(
            &format!("{TRANSFORMER_BLOCK} (f16, packed Q4 / BF16 scales)"),
            packed_block.quantum_scratch(1, 1, 256, 7, false, Dtype::Float16, unbounded.attention),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        packed_block.forward_ip(&y, &memory, None, unbounded),
                    )
                        .1?,
                ])
            },
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::f16_transformer_quanta_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::f16_transformer_quanta_track_their_measured_peaks");
    }

    // -- Text / vision encoders ------------------------------------------------------------------

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn encoder_layers_track_their_measured_peaks() {
        if !cpu_child("proofs::encoder_layers_track_their_measured_peaks") {
            return;
        }
        // CLIP text layer: 77 tokens, 128 wide, 2 heads, the causal additive mask.
        let cfg = ClipTextConfig {
            num_layers: 1,
            model_dims: 128,
            num_heads: 2,
            max_length: 77,
            vocab_size: 40,
            projection_dim: None,
            hidden_act: ClipActivation::Gelu,
        };
        let mut g = Gen::new(7200);
        g.put(
            "text_model.embeddings.token_embedding.weight".into(),
            &[40, 128],
            0.5,
        );
        g.put(
            "text_model.embeddings.position_embedding.weight".into(),
            &[77, 128],
            0.1,
        );
        g.clip_layers("text_model", 1, 128, 3 * 128);
        g.ln("text_model.final_layer_norm", 128);
        let te =
            crate::text_encoder::ClipTextEncoder::from_weights(&g.weights(), "text_model", &cfg)
                .unwrap();
        let x = rand(&[2, 77, 128], 1);
        let (scratch, body) = te.layer_quantum_for_test(0, &x);
        prove(CLIP_LAYER, scratch, || Ok(vec![body()?]));

        // CLIP vision layer: 257 tokens (16x16 patches + class), 128 wide, 2 heads, 2x MLP.
        let vcfg = crate::vision_encoder::VisionConfig {
            hidden: 128,
            num_layers: 1,
            num_heads: 2,
            patch: 4,
            image_size: 64,
            num_channels: 3,
            quick_gelu: false,
        };
        let mut g = Gen::new(7300);
        g.put(
            "vision_model.embeddings.patch_embedding.weight".into(),
            &[128, 3, 4, 4],
            0.2,
        );
        g.put(
            "vision_model.embeddings.class_embedding".into(),
            &[128],
            0.1,
        );
        g.put(
            "vision_model.embeddings.position_embedding.weight".into(),
            &[257, 128],
            0.1,
        );
        g.ln("vision_model.pre_layrnorm", 128);
        g.clip_layers("vision_model", 1, 128, 256);
        let vision =
            crate::vision_encoder::ClipVisionEncoder::from_weights(&g.weights(), &vcfg).unwrap();
        let v = rand(&[1, 257, 128], 2);
        let (scratch, body) = vision.layer_quantum_for_test(0, &v);
        prove(VISION_LAYER, scratch, || Ok(vec![body()?]));
        let pixels = rand(&[1, 64, 64, 3], 5);
        let (scratch, body) = vision.embed_quantum_for_test(&pixels);
        prove(VISION_EMBEDDINGS, scratch, || Ok(vec![body()?]));

        // IP Resampler perceiver layer: 257 image tokens + 16 queries, 2 heads of 64, 4x FFN.
        let rcfg = crate::ip_adapter::ResamplerConfig {
            dim: 128,
            depth: 1,
            heads: 2,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 128,
            output_dim: 64,
        };
        let mut g = Gen::new(7400);
        g.put("image_proj.latents".into(), &[1, 16, 128], 0.5);
        g.linear("image_proj.proj_in", 128, 128, true);
        g.linear("image_proj.proj_out", 64, 128, true);
        g.ln("image_proj.norm_out", 64);
        g.ln("image_proj.layers.0.0.norm1", 128);
        g.ln("image_proj.layers.0.0.norm2", 128);
        g.linear("image_proj.layers.0.0.to_q", 128, 128, false);
        g.linear("image_proj.layers.0.0.to_kv", 256, 128, false);
        g.linear("image_proj.layers.0.0.to_out", 128, 128, false);
        g.ln("image_proj.layers.0.1.0", 128);
        g.linear("image_proj.layers.0.1.1", 512, 128, false);
        g.linear("image_proj.layers.0.1.3", 128, 512, false);
        let resampler =
            crate::ip_adapter::Resampler::from_weights(&g.weights(), "image_proj", &rcfg).unwrap();
        let features = rand(&[1, 257, 128], 3);
        let latents = rand(&[1, 16, 128], 4);
        let (scratch, body) = resampler.layer_quantum_for_test(0, &features, &latents);
        prove(RESAMPLER_LAYER, scratch, || Ok(vec![body()?]));

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::encoder_layers_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::encoder_layers_track_their_measured_peaks");
    }

    // -- VAE -------------------------------------------------------------------------------------

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn vae_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::vae_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let mut g = Gen::new(7500);
        g.resnet("r", 64, 64, None);
        g.ln("a.group_norm", 64);
        for p in ["to_q", "to_k", "to_v", "to_out.0"] {
            g.linear(&format!("a.{p}"), 64, 64, true);
        }
        let w = g.weights();

        // A temb-free resnet (`resnet_quantum`'s lazy body).
        let resnet = crate::unet::ResnetBlock2D::from_weights(&w, "r").unwrap();
        let x = rand(&[1, 32, 32, 64], 1);
        prove(VAE_RESNET, resnet.quantum_scratch(&x, None), || {
            Ok(vec![
                (
                    crate::test_device::record_model_inference_device(),
                    resnet.forward(&x, None),
                )
                    .1?,
            ])
        });

        // Mid attention, unchunked (the host path) and query-chunked (the iOS plan) — priced at
        // the chunk's rows.
        let attn = crate::vae::VaeAttention::from_weights(&w, "a").unwrap();
        let small = rand(&[1, 16, 16, 64], 2);
        prove(
            VAE_ATTENTION,
            vae_attention_bytes(1, 256, 64, 256, 4, Backend::Cpu),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        attn.forward(&small),
                    )
                        .1?,
                ])
            },
        );
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(128 * 1024, true);
        assert_eq!(budget.query_block(1, 1, 1024, 1024), 128);
        let chunked = mlx_gen::attention::AttentionPlan::budgeted(budget);
        prove(
            &format!("{VAE_ATTENTION} (query-chunked)"),
            vae_attention_bytes(1, 1024, 64, 128, 4, Backend::Cpu),
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        attn.forward_planned(&x, chunked),
                    )
                        .1?,
                ])
            },
        );

        // Downsample: asymmetric pad + stride-2 conv.
        let wd = conv_weight(64, 3, 64, 3);
        let bd = rand(&[64], 4);
        let call = ConvCall::new(&[1, 33, 33, 64], 4, &wd, 2, 0);
        prove(VAE_DOWNSAMPLE, single_conv_bytes(&call, 1, 0, cpu), || {
            let padded = mlx_rs::ops::pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            Ok(vec![mlx_gen::nn::conv2d(&padded, &wd, Some(&bd), 2, 0)?])
        });

        // Decode conv_in over the lazily de-normalized latent; encode conv_in over the image.
        let z = rand(&[1, 32, 32, 4], 5);
        let wi = conv_weight(64, 3, 4, 6);
        let bi = rand(&[64], 7);
        let post = rand(&[4, 4], 8);
        let call = ConvCall::new(z.shape(), 4, &wi, 1, 1);
        prove(VAE_DECODE_IN, single_conv_bytes(&call, 2, 0, cpu), || {
            let scaled = mlx_rs::ops::multiply(&z, Array::from_f32(1.0 / 0.13025))?;
            let projected = mlx_rs::ops::matmul(&scaled, post.t())?;
            Ok(vec![mlx_gen::nn::conv2d(&projected, &wi, Some(&bi), 1, 1)?])
        });
        let image = rand(&[1, 64, 64, 3], 9);
        let we = conv_weight(64, 3, 3, 10);
        let call = ConvCall::new(image.shape(), 4, &we, 1, 1);
        prove(VAE_ENCODE_IN, single_conv_bytes(&call, 1, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(&image, &we, Some(&bi), 1, 1)?])
        });

        // Decode conv_out: GroupNorm + SiLU + conv to 3 channels.
        let (gw, gb) = (
            evaluated(Array::ones::<f32>(&[64]).unwrap()),
            rand(&[64], 11),
        );
        let wo = conv_weight(3, 3, 64, 12);
        let bo = rand(&[3], 13);
        let call = ConvCall::new(x.shape(), 4, &wo, 1, 1);
        prove(VAE_DECODE_OUT, norm_conv_bytes(&call, cpu), || {
            let y = mlx_gen::nn::group_norm(&x, &gw, &gb, 32, 1e-5)?;
            Ok(vec![mlx_gen::nn::conv2d(
                &mlx_gen::nn::silu(&y)?,
                &wo,
                Some(&bo),
                1,
                1,
            )?])
        });

        // Encode out (the fixture VAE's real head) over its encoder body.
        let vae = tiny_vae();
        let (body, scratch) = vae
            .encode_body_for_test(&rand(&[1, 64, 64, 3], 14))
            .unwrap();
        let body = evaluated(body);
        prove(VAE_ENCODE_OUT, scratch, || Ok(vec![vae.encode_out(&body)?]));

        // The decoded image's pixel conversion.
        let decoded = rand(&[1, 64, 64, 3], 15);
        prove(DECODED_TO_IMAGE, decoded_to_image_bytes(&decoded), || {
            crate::pipeline::decoded_to_image(&decoded)?;
            Ok(vec![])
        });

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::vae_quanta_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::vae_quanta_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    // Prior ignore attribute: #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn tiled_vae_stages_and_tiles_track_their_measured_peaks() {
        if !cpu_child("proofs::tiled_vae_stages_and_tiles_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let mut g = Gen::new(7600);
        g.resnet("r", 64, 64, None);
        let resnet = crate::unet::ResnetBlock2D::from_weights(&g.weights(), "r").unwrap();
        let wu = conv_weight(64, 3, 64, 1);
        let bu = rand(&[64], 2);
        // A tiled stage plus its widest tile (the 16-edge tiles of a 32x32 extent).
        let x = rand(&[1, 32, 32, 64], 3);
        let tile = tile_bytes(
            &[1, 16, 16, 64],
            TileSource {
                full_width: 32,
                normalized: true,
            },
            4,
            &wu,
            false,
            cpu,
        );
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (resnet)"),
            tiled_stage_bytes(&[1, 32, 32, 64], 64, 64, 4, true) + tile,
            || {
                Ok(vec![
                    (
                        crate::test_device::record_model_inference_device(),
                        resnet.forward_tiled_vae(&x, 16, None, crate::quantum::VAE_DECODE_LABELS),
                    )
                        .1?,
                ])
            },
        );
        // The bounded encode's stride-2 downsample, at a stage-dominated shape (8-pixel output
        // cores, 17x17 exact-halo crops, over a 128x128x32 input: the row strips + assembly carry
        // the peak) and a tile-dominated one (one 32-pixel core: its 65x65 padded crop, output and
        // im2col carry it).
        let wds = conv_weight(32, 3, 32, 5);
        let bds = rand(&[32], 9);
        let wide = rand(&[1, 128, 128, 32], 10);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode downsample, many tiles)"),
            downsample_stage_bytes(&[1, 64, 64, 32], 4)
                + downsample_tile_bytes(&[1, 17, 17, 32], 4, &wds, cpu),
            || {
                Ok(vec![crate::vae::downsample_tiled(
                    &wide, &wds, &bds, 8, None,
                )?])
            },
        );
        let wd = conv_weight(64, 3, 64, 5);
        let sq = rand(&[1, 64, 64, 64], 11);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode downsample, one tile)"),
            downsample_stage_bytes(&[1, 32, 32, 64], 4)
                + downsample_tile_bytes(&[1, 65, 65, 64], 4, &wd, cpu),
            || Ok(vec![crate::vae::downsample_tiled(&sq, &wd, &bu, 32, None)?]),
        );
        // The bounded encode's conv_in: a 3-channel image to 32 channels at 16-pixel tiles.
        let image = rand(&[1, 64, 64, 3], 6);
        let wi = conv_weight(32, 3, 3, 7);
        let bi = rand(&[32], 8);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode conv_in)"),
            tiled_stage_bytes(&[1, 64, 64, 32], 3, 32, 4, false)
                + tile_bytes(
                    &[1, 18, 18, 3],
                    TileSource {
                        full_width: 64,
                        normalized: false,
                    },
                    4,
                    &wi,
                    false,
                    cpu,
                ),
            || {
                Ok(vec![mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(
                    &image,
                    &wi,
                    Some(&bi),
                    16,
                    None,
                    |t| Ok(t.clone()),
                )?])
            },
        );
        let small = rand(&[1, 16, 16, 64], 4);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (upsample)"),
            tiled_stage_bytes(&[1, 32, 32, 64], 64, 64, 4, false) + tile,
            || {
                Ok(vec![crate::vae::upsample_tiled(
                    &small, &wu, &bu, 16, None,
                )?])
            },
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::tiled_vae_stages_and_tiles_track_their_measured_peaks");
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::proofs::tiled_vae_stages_and_tiles_track_their_measured_peaks");
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn sampler_step_tracks_its_measured_peak() {
        if !cpu_child("proofs::sampler_step_tracks_its_measured_peak") {
            return;
        }
        use mlx_gen::DiffusionSampler;
        let sampler =
            crate::sampler::EulerSampler::new(&crate::config::DiffusionConfig::sdxl_base(), true)
                .unwrap();
        let schedule = crate::sampler::AncestralEuler::new(&sampler, 20, 1000.0).unwrap();
        let latents = rand(&[1, 64, 64, 4], 1);
        let eps_text = rand(&[1, 64, 64, 4], 2);
        let eps_neg = rand(&[1, 64, 64, 4], 3);
        prove(SAMPLER_STEP, sampler_step_bytes(&latents, false), || {
            let eps =
                mlx_gen::gen_core::guidance::cfg(&mlx_gen::MlxLatentOps, &eps_text, &eps_neg, 7.0)?;
            Ok(vec![schedule.step(&eps, &latents, 3)?])
        });

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::sampler_step_tracks_its_measured_peak");
        }
    }

    /// One projection's load-time pack (epic 16774) over a resident dense f16 `[1280, 2048]`
    /// projection with its bias (an SDXL IP K/V or ControlNet attention projection's width), at
    /// both packed tiers.
    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn load_quantize_tracks_its_measured_peak() {
        if !cpu_child("proofs::load_quantize_tracks_its_measured_peak") {
            return;
        }
        let mut g = Gen::new(9300);
        g.linear("p", 1280, 2048, true);
        let w = g.weights();
        let mut half = std::collections::HashMap::new();
        for key in ["p.weight", "p.bias"] {
            half.insert(
                key.to_owned(),
                evaluated(w.get(key).unwrap().as_dtype(Dtype::Float16).unwrap()),
            );
        }
        let base = crate::quant::lin(&Weights::from_map(half), "p", true).unwrap();
        let (weight, bias) = base.dense_weight().unwrap();
        for bits in [4, 8] {
            let formula = crate::quant::quantize_bytes(weight, bias, bits);
            prove(LOAD_QUANTIZE, formula, || {
                let mut packed = base.clone();
                crate::quant::guarded_load_quantize(|| {
                    crate::quant::quantize_linear(&mut packed, bits)
                })?;
                let (wq, scales, biases, bias, _, _) = packed.quantized_params().unwrap();
                let mut out = vec![wq.clone(), scales.clone(), biases.clone()];
                out.extend(bias.cloned());
                Ok(out)
            });
        }

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::load_quantize_tracks_its_measured_peak");
        }
    }

    /// The IP tokens' CFG pair (epic 16774), at the production plus Resampler's 16 x 2048 tokens
    /// (f16, and f32 where a packed tier promotes them).
    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn ip_cfg_tokens_track_their_measured_peak() {
        if !cpu_child("proofs::ip_cfg_tokens_track_their_measured_peak") {
            return;
        }
        for dtype in [Dtype::Float16, Dtype::Float32] {
            let tokens = evaluated(rand(&[1, 16, 2048], 4).as_dtype(dtype).unwrap());
            prove(IP_CFG_TOKENS, ip_cfg_tokens_bytes(&tokens), || {
                Ok(vec![crate::ip_adapter::cfg_pair(&tokens)?])
            });
        }

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::proofs::ip_cfg_tokens_track_their_measured_peak");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reachable-sizes prediction (sc-26270 round 3, item 7). Host-derived PREDICTIONS, not evidence:
// resident bytes per phase from the on-disk safetensors headers under the iOS load-cast rule, the
// largest counted quantum of that phase at production shapes (Metal backend, f16 activations —
// f32 through a packed tier's U-Net and text encoders, whose BF16 scales promote them — sequential
// CFG, fused U-Net SDPA, the fp16-fix VAE with the chunked mid attention), and whether
// resident + scratch + the 512 MiB reserve fits the 6142 MiB entitled launch headroom.
// ---------------------------------------------------------------------------------------------

#[path = "quantum_tests_reachable.rs"]
mod reachable;

#[path = "quantum_tests_pricing.rs"]
mod pricing;

// ---------------------------------------------------------------------------------------------
// Adapter (LoRA / LoKr / LoHa) reads and installs (sc-26270 round 4, item 2).
// ---------------------------------------------------------------------------------------------

mod adapters {
    use super::*;
    use mlx_gen::adapters::AdaptableHost;
    use mlx_gen::ios_quantum::measured_peak_bytes;
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};

    use crate::adapters::{apply_sdxl_adapters_with, LoraCoverage};

    const Q: &str = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
    const PROJ_IN: &str = "down_blocks.1.attentions.0.proj_in";
    const CONV: &str = "down_blocks.1.resnets.0.conv2";
    const SHORTCUT: &str = "up_blocks.0.resnets.0.conv_shortcut";

    fn flat(dotted: &str) -> String {
        format!("lora_unet_{}", dotted.replace('.', "_"))
    }

    fn dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("sdxl-adapter-quanta-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write `tensors` (f16 unless already typed) to `name` in the scratch dir with `meta`.
    fn write(
        name: &str,
        tensors: Vec<(String, Array)>,
        meta: &[(&str, &str)],
    ) -> std::path::PathBuf {
        let path = dir().join(name);
        let meta: HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let named: Vec<(String, Array)> = tensors
            .into_iter()
            .map(|(k, a)| (k, evaluated(a)))
            .collect();
        Array::save_safetensors(named.iter().map(|(k, a)| (k.as_str(), a)), &meta, &path).unwrap();
        path
    }

    fn evaluated(a: Array) -> Array {
        a.eval().unwrap();
        a
    }

    fn half(a: Array) -> Array {
        a.as_dtype(Dtype::Float16).unwrap()
    }

    /// A kohya LoRA pair for the linear `dotted` (`[out, in]`), rank `r`, f16 factors.
    fn lora_linear(dotted: &str, out: i32, inp: i32, r: i32, seed: u64) -> Vec<(String, Array)> {
        let k = flat(dotted);
        vec![
            (
                format!("{k}.lora_down.weight"),
                half(rnd(&[r, inp], seed, 0.1)),
            ),
            (
                format!("{k}.lora_up.weight"),
                half(rnd(&[out, r], seed + 1, 0.1)),
            ),
            (format!("{k}.alpha"), Array::from_f32(r as f32)),
        ]
    }

    /// A kohya conv LoRA pair for `dotted` (`[out, in, k, k]`), rank `r`, f16 factors.
    fn lora_conv(
        dotted: &str,
        out: i32,
        inp: i32,
        kk: i32,
        r: i32,
        seed: u64,
    ) -> Vec<(String, Array)> {
        let k = flat(dotted);
        vec![
            (
                format!("{k}.lora_down.weight"),
                half(rnd(&[r, inp, kk, kk], seed, 0.1)),
            ),
            (
                format!("{k}.lora_up.weight"),
                half(rnd(&[out, r, 1, 1], seed + 1, 0.1)),
            ),
            (format!("{k}.alpha"), Array::from_f32(r as f32)),
        ]
    }

    fn spec(path: std::path::PathBuf, kind: AdapterKind) -> AdapterSpec {
        AdapterSpec::new(path, 0.8, kind)
    }

    /// The adapter-proof U-Net: 32/256 channels, so its targets are 64K-590K elements; f16 like the
    /// device's, materialized (the production route) unless `lazy` (the fused-LDM route, whose U-Net
    /// is still the file's lazy arrays when the adapters merge); `packed` packs the linear targets
    /// at Q4.
    fn proof_unet(lazy: bool, packed: bool) -> crate::unet::UNet2DConditionModel {
        let mut cfg = crate::test_support::tiny_unet_config();
        cfg.block_out_channels = vec![32, 256];
        cfg.num_attention_heads = vec![1, 4];
        let mut w = crate::test_support::tiny_unet_weights_for(&cfg, 21).unwrap();
        w.cast_all(Dtype::Float16).unwrap();
        let w = if lazy {
            let path = dir().join("proof-unet.safetensors");
            if !path.exists() {
                super::save(&w, &path);
            }
            Weights::from_file(&path).unwrap()
        } else {
            w.materialize().unwrap();
            w
        };
        let mut unet = crate::unet::UNet2DConditionModel::from_weights(&w, &cfg).unwrap();
        if packed {
            // Pack the adapter targets (the fixture's 16-wide context projections cannot pack).
            for target in [Q, PROJ_IN] {
                let parts: Vec<&str> = target.split('.').collect();
                crate::quant::quantize_linear(unet.adaptable_mut(&parts).unwrap(), 4).unwrap();
            }
        }
        unet
    }

    /// Actual loader/installed-state parity for the configured exporter. Tiny CPU-only installs,
    /// not generation, allocator measurements, or ProcessGrowth evidence.
    #[test]
    fn configured_adapter_facts_match_actual_install_branches() {
        let Some(mut completion) =
            completed_cpu_child("adapters::configured_adapter_facts_match_actual_install_branches")
        else {
            return;
        };
        use mlx_gen::adapters::Adapter;
        use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader};
        let cfg = crate::test_support::tiny_unet_config();
        for (name, packed, kind, scale, form) in [
            ("dense_lora", false, AdapterKind::Lora, 0.8, 0),
            ("packed_lora_zero", true, AdapterKind::Lora, 0.0, 0),
            ("structured_lokr_zero", true, AdapterKind::Lokr, 0.8, 1),
            ("thirdparty_lokr_zero", true, AdapterKind::Lora, 0.0, 2),
            ("thirdparty_loha_zero", true, AdapterKind::Lora, 0.0, 3),
            ("packed_conv_lora", true, AdapterKind::Lora, 0.8, 4),
        ] {
            let mut weights = crate::test_support::tiny_unet_weights_for(&cfg, 191).unwrap();
            if packed {
                let weight = weights
                    .require(&format!("{Q}.weight"))
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                let (codes, scales, biases) =
                    mlx_rs::ops::quantize(&weight, crate::quant::GROUP_SIZE, 4).unwrap();
                weights.insert(format!("{Q}.weight"), codes);
                weights.insert(format!("{Q}.scales"), scales);
                weights.insert(format!("{Q}.biases"), biases);
            }
            let headers: Vec<_> = weights
                .keys()
                .map(|key| {
                    let a = weights.require(key).unwrap();
                    let dtype = match a.dtype() {
                        Dtype::Float32 => Stored::F32,
                        Dtype::Bfloat16 => Stored::BF16,
                        Dtype::Float16 => Stored::F16,
                        Dtype::Uint32 => Stored::U32,
                        other => panic!("unexpected fixture dtype {other:?}"),
                    };
                    SafetensorsTensorHeader {
                        name: key.to_owned(),
                        dtype,
                        shape: a.shape().iter().map(|d| *d as usize).collect(),
                        data_bytes: a.size() as u64 * crate::quantum::dtype_bytes(a.dtype()),
                    }
                })
                .collect();
            let prefix = flat(Q);
            let (tensors, metadata) = match form {
                0 => (lora_linear(Q, 64, 64, 4, 201), vec![]),
                1 => (
                    vec![
                        (format!("{Q}.lokr_w1"), rnd(&[8, 8], 211, 0.1)),
                        (format!("{Q}.lokr_w2"), rnd(&[8, 8], 212, 0.1)),
                    ],
                    vec![("networkType", "lokr"), ("rank", "4"), ("alpha", "0")],
                ),
                2 => (
                    vec![
                        (format!("{prefix}.lokr_w1"), rnd(&[8, 8], 221, 0.1)),
                        (format!("{prefix}.lokr_w2"), rnd(&[8, 8], 222, 0.1)),
                        (format!("{prefix}.alpha"), Array::from_f32(0.0)),
                    ],
                    vec![],
                ),
                3 => (
                    vec![
                        (format!("{prefix}.hada_w1_a"), rnd(&[64, 4], 231, 0.1)),
                        (format!("{prefix}.hada_w1_b"), rnd(&[4, 64], 232, 0.1)),
                        (format!("{prefix}.hada_w2_a"), rnd(&[64, 4], 233, 0.1)),
                        (format!("{prefix}.hada_w2_b"), rnd(&[4, 64], 234, 0.1)),
                        (format!("{prefix}.alpha"), Array::from_f32(0.0)),
                    ],
                    vec![],
                ),
                4 => (lora_conv(Q, 64, 64, 1, 4, 241), vec![]),
                _ => unreachable!(),
            };
            let mut tensors = tensors;
            // Deliberately out of surface: the actual loader materializes this too, even when
            // the accepted adapter's forward is disabled. It must survive numeric read pricing.
            tensors.push((
                "lora_te1_ignored.weight".into(),
                half(rnd(&[3, 7], 249, 0.1)),
            ));
            let file = write(&format!("semantic-{name}.safetensors"), tensors, &metadata);
            let adapter = AdapterSpec::new(file, scale, kind);
            let mut spec = mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(dir()));
            spec.adapters.push(adapter.clone());
            let facts =
                crate::adapters::configured_adapter_semantic_facts_with(&spec, &headers, &cfg)
                    .unwrap();
            let entry = &facts["adapters"][0]["installed"][0];
            let (loaded, work) = crate::adapters::with_adapter_work(|| {
                with_cpu_diagnostic_headroom(PLENTY, || {
                    crate::loader::cast_for_load(&mut weights, crate::model::DTYPE).unwrap();
                    let mut unet = UNet2DConditionModel::from_weights(&weights, &cfg).unwrap();
                    apply_sdxl_adapters_with(
                        &mut unet,
                        std::slice::from_ref(&adapter),
                        crate::adapters::configured_lora_coverage(),
                    )
                    .unwrap();
                    Ok(unet)
                })
            });
            let mut unet = loaded.unwrap();
            let ledger = crate::ios_pricing::AdapterLoadLedger::from_work(&work, 123);
            let raw =
                mlx_gen::ios_quantum::load_floor_bytes(std::slice::from_ref(&adapter.path), |_| {
                    true
                })
                .unwrap();
            assert_eq!(
                ledger.raw_parent_upper, raw,
                "{name}: actual adapter producer lost a materialized raw key"
            );
            assert!(
                matches!(
                    work.first(),
                    Some(crate::adapters::AdapterWork::FileMaterialized(_))
                ) && matches!(
                    work.last(),
                    Some(crate::adapters::AdapterWork::FileScopeEnd)
                ),
                "{name}: actual adapter producer file-map scope order"
            );
            assert_eq!(
                ledger.quanta[0].scratch, raw,
                "{name}: actual adapter producer raw-file load price"
            );
            assert!(
                ledger
                    .quanta
                    .iter()
                    .filter(|q| q.label == ADAPTER_MERGE)
                    .all(|q| q.resident >= 123 + raw),
                "{name}: actual adapter producer dropped raw parents before install"
            );
            let target = unet
                .adaptable_mut(&Q.split('.').collect::<Vec<_>>())
                .unwrap();
            let describe = |facts: &mlx_gen::adapters::facts::AdapterFacts| {
                (
                    facts.disabled(),
                    facts
                        .tensors()
                        .iter()
                        .map(|tensor| (tensor.shape.clone(), tensor.dtype, tensor.bytes()))
                        .collect::<Vec<_>>(),
                )
            };
            let native: Vec<_> = target
                .adapters()
                .iter()
                .map(mlx_gen::adapters::facts::AdapterFacts::of)
                .collect();
            assert_eq!(
                ledger.residuals.iter().map(describe).collect::<Vec<_>>(),
                native.iter().map(describe).collect::<Vec<_>>(),
                "{name}: actual adapter producer declared residual/disable parity"
            );
            for dtype in [Dtype::Float16, Dtype::Float32] {
                let output = crate::quantum::linear_output_dtype(target, dtype);
                let residual = native.iter().fold(0u64, |n, fact| {
                    n.saturating_add(crate::adapters::adapter_forward_scratch(
                        fact,
                        &[2, 3, 64],
                        dtype,
                        output,
                    ))
                });
                assert!(
                    crate::quantum::linear_operand_bytes(target, &[2, 3, 64], dtype) >= residual,
                    "{name}: actual adapter producer residual absent from operand budget"
                );
            }
            let (storage, disabled) = match target.adapters() {
                [] => ("merged_dense", false),
                [Adapter::Lora { .. }] => ("factor_residual", target.adapters()[0].is_disabled()),
                [Adapter::LokrStructured { .. }] => {
                    ("structured_residual", target.adapters()[0].is_disabled())
                }
                [Adapter::Lokr { .. }] => (
                    "materialized_delta_residual",
                    target.adapters()[0].is_disabled(),
                ),
                other => panic!("unexpected installed stack length {}", other.len()),
            };
            assert_eq!(
                entry["installed"], storage,
                "{name}: exporter disagrees with actual installed storage"
            );
            assert_eq!(
                entry["configured_zero_branch"], disabled,
                "{name}: exporter disagrees with actual disable predicate"
            );
        }
        completion.body_completed = true;
    }

    fn weight_of(unet: &mut crate::unet::UNet2DConditionModel, dotted: &str) -> Array {
        let parts: Vec<&str> = dotted.split('.').collect();
        if let Some(conv) = unet.adaptable_conv_mut(&parts) {
            return conv.weight().clone();
        }
        let lin = unet.adaptable_mut(&parts).unwrap();
        lin.dense_weight().map(|(w, _)| w.clone()).unwrap()
    }

    /// One adapter file's measured proof: the guarded apply's adapter-load floor plus its largest
    /// install price, against MLX's worst-of-N measured peak of the whole apply (the file's tensors
    /// stay live while it installs), on a fresh U-Net per run.
    fn prove_install(
        label: &str,
        unet: impl Fn() -> crate::unet::UNet2DConditionModel,
        file: &std::path::Path,
        kind: AdapterKind,
        coverage: LoraCoverage,
    ) -> f64 {
        let specs = [spec(file.to_path_buf(), kind)];
        let (r, events) = observe(PLENTY, || {
            apply_sdxl_adapters_with(&mut unet(), &specs, coverage)
        });
        assert!(r.unwrap().merged >= 1);
        let load: u64 = checked(&events, ADAPTER_LOAD).iter().sum();
        let merge = checked(&events, ADAPTER_MERGE).into_iter().max().unwrap();
        let formula = load + merge;
        let measured = (0..5)
            .map(|_| {
                let mut u = unet();
                with_cpu_diagnostic_headroom(PLENTY, || {
                    measured_peak_bytes(|| {
                        apply_sdxl_adapters_with(&mut u, &specs, coverage)?;
                        Ok(vec![])
                    })
                })
                .unwrap()
            })
            .max()
            .unwrap();
        let ratio = formula as f64 / measured as f64;
        eprintln!(
            "PROOF {ADAPTER_MERGE} {label}: formula {formula} B (load {load} + install {merge}), \
             worst of 5 measured {measured} B, ratio {ratio:.3}"
        );
        assert_tracks_measured(label, formula, measured, 1.5);
        ratio
    }

    #[test]
    #[ignore = "expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
    fn adapter_installs_track_their_measured_peaks() {
        if !cpu_child("adapters::adapter_installs_track_their_measured_peaks") {
            return;
        }
        let dense = || proof_unet(false, false);
        let lazy = || proof_unet(true, false);
        let packed = || proof_unet(false, true);
        let complete = LoraCoverage::Complete;
        let lora = write("lora.safetensors", lora_linear(Q, 256, 256, 16, 1), &[]);
        prove_install(
            "LoRA, dense f16 base",
            dense,
            &lora,
            AdapterKind::Lora,
            complete,
        );
        prove_install(
            "LoRA, lazily read base",
            lazy,
            &lora,
            AdapterKind::Lora,
            complete,
        );
        let wide = write(
            "lora-wide.safetensors",
            lora_linear(PROJ_IN, 256, 256, 256, 3),
            &[],
        );
        prove_install(
            "LoRA, packed base",
            packed,
            &wide,
            AdapterKind::Lora,
            complete,
        );
        let conv = write("conv.safetensors", lora_conv(CONV, 256, 256, 3, 16, 5), &[]);
        prove_install(
            "conv LoRA, f16 conv",
            dense,
            &conv,
            AdapterKind::Lora,
            complete,
        );
        prove_install(
            "conv LoRA, lazily read conv",
            lazy,
            &conv,
            AdapterKind::Lora,
            complete,
        );
        let shortcut = write(
            "shortcut.safetensors",
            lora_conv(SHORTCUT, 256, 512, 1, 16, 7),
            &[],
        );
        prove_install(
            "1x1 shortcut LoRA",
            dense,
            &shortcut,
            AdapterKind::Lora,
            complete,
        );

        // peft LoKr (networkType stamp, global alpha/rank metadata).
        let lokr_meta = [("networkType", "lokr"), ("rank", "4"), ("alpha", "4")];
        let peft = write(
            "lokr.safetensors",
            vec![
                (format!("{Q}.lokr_w1"), half(rnd(&[16, 16], 9, 0.3))),
                (format!("{Q}.lokr_w2"), half(rnd(&[16, 16], 10, 0.3))),
            ],
            &lokr_meta,
        );
        prove_install(
            "peft LoKr, dense base",
            dense,
            &peft,
            AdapterKind::Lokr,
            complete,
        );
        let structured = write(
            "lokr-structured.safetensors",
            vec![
                (format!("{Q}.lokr_w1"), half(rnd(&[1, 1], 11, 0.3))),
                (format!("{Q}.lokr_w2"), half(rnd(&[256, 256], 12, 0.3))),
            ],
            &lokr_meta,
        );
        prove_install(
            "peft LoKr, packed base (structured)",
            packed,
            &structured,
            AdapterKind::Lokr,
            complete,
        );

        // Third-party LyCORIS LoKr and LoHa (kohya keys, per-module alpha, no stamp).
        let k = flat(Q);
        let tp_lokr = write(
            "tp-lokr.safetensors",
            vec![
                (format!("{k}.lokr_w1"), half(rnd(&[16, 16], 13, 0.3))),
                (format!("{k}.lokr_w2_a"), half(rnd(&[16, 4], 14, 0.3))),
                (format!("{k}.lokr_w2_b"), half(rnd(&[4, 16], 15, 0.3))),
                (format!("{k}.alpha"), Array::from_f32(4.0)),
            ],
            &[],
        );
        prove_install(
            "third-party LoKr, dense base",
            dense,
            &tp_lokr,
            AdapterKind::Lora,
            complete,
        );
        prove_install(
            "third-party LoKr, packed base",
            packed,
            &tp_lokr,
            AdapterKind::Lora,
            complete,
        );
        let loha = write(
            "loha.safetensors",
            vec![
                (format!("{k}.hada_w1_a"), half(rnd(&[256, 8], 16, 0.1))),
                (format!("{k}.hada_w1_b"), half(rnd(&[8, 256], 17, 0.1))),
                (format!("{k}.hada_w2_a"), half(rnd(&[256, 8], 18, 0.1))),
                (format!("{k}.hada_w2_b"), half(rnd(&[8, 256], 19, 0.1))),
                (format!("{k}.alpha"), Array::from_f32(8.0)),
            ],
            &[],
        );
        prove_install(
            "LoHa, dense base",
            dense,
            &loha,
            AdapterKind::Lora,
            complete,
        );
        prove_install(
            "LoHa, packed base",
            packed,
            &loha,
            AdapterKind::Lora,
            complete,
        );

        if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
            eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::adapters::adapter_installs_track_their_measured_peaks");
        }
    }

    /// Every adapter file is checked (header floor) before it is read and every target's install
    /// is its own checked-then-synchronized quantum; with zero headroom the first file's read
    /// refuses and nothing merges; at a headroom that admits the read but not the first install,
    /// the install refuses before it is built and the U-Net is untouched; guarded installs are
    /// byte-identical to the host's.
    #[ignore = "model inference; explicit model-proof workflow"]
    #[test]
    fn adapter_reads_and_installs_refuse_before_they_run_and_stay_identical() {
        if !cpu_child(
            "adapters::adapter_reads_and_installs_refuse_before_they_run_and_stay_identical",
        ) {
            return;
        }
        let lora_file = {
            let mut t = lora_linear(Q, 256, 256, 8, 31);
            t.extend(lora_linear(PROJ_IN, 256, 256, 8, 33));
            t.extend(lora_conv(CONV, 256, 256, 3, 8, 35));
            write("multi.safetensors", t, &[])
        };
        let k = flat(PROJ_IN);
        let loha_file = write(
            "multi-loha.safetensors",
            vec![
                (format!("{k}.hada_w1_a"), half(rnd(&[256, 4], 37, 0.1))),
                (format!("{k}.hada_w1_b"), half(rnd(&[4, 256], 38, 0.1))),
                (format!("{k}.hada_w2_a"), half(rnd(&[256, 4], 39, 0.1))),
                (format!("{k}.hada_w2_b"), half(rnd(&[4, 256], 40, 0.1))),
            ],
            &[],
        );
        let specs = [
            spec(lora_file, AdapterKind::Lora),
            spec(loha_file, AdapterKind::Lora),
        ];
        let targets = [Q, PROJ_IN, CONV];
        let apply = |unet: &mut crate::unet::UNet2DConditionModel| {
            apply_sdxl_adapters_with(unet, &specs, LoraCoverage::Complete)
        };
        let snapshot = |unet: &mut crate::unet::UNet2DConditionModel| -> Vec<Vec<u32>> {
            targets.iter().map(|t| bits(&weight_of(unet, t))).collect()
        };

        // Host (no guard) vs guarded: identical merged weights.
        let mut host = proof_unet(false, false);
        let base = snapshot(&mut host);
        let report = apply(&mut host).unwrap();
        assert_eq!(report.merged, 4);
        let merged = snapshot(&mut host);
        assert_ne!(merged, base, "the fixture must merge");
        let mut guarded = proof_unet(false, false);
        let (r, events) = observe(PLENTY, || apply(&mut guarded));
        assert_eq!(r.unwrap(), report);
        assert_eq!(
            snapshot(&mut guarded),
            merged,
            "guarded installs must be byte-identical"
        );

        // Ordering: each file's read check, then one checked + synchronized install per target.
        let expected_shape: Vec<(bool, &str)> = [
            (true, ADAPTER_LOAD),
            (true, ADAPTER_MERGE),
            (false, ADAPTER_MERGE),
            (true, ADAPTER_MERGE),
            (false, ADAPTER_MERGE),
            (true, ADAPTER_MERGE),
            (false, ADAPTER_MERGE),
            (true, ADAPTER_LOAD),
            (true, ADAPTER_MERGE),
            (false, ADAPTER_MERGE),
        ]
        .into();
        let shape: Vec<(bool, &str)> = events
            .iter()
            .map(|e| match e {
                Ev::Checked(l, _) => (true, l.as_str()),
                Ev::Synced(l) => (false, l.as_str()),
            })
            .collect();
        assert_eq!(shape, expected_shape, "{events:?}");
        let floor =
            mlx_gen::ios_quantum::load_floor_bytes(std::slice::from_ref(&specs[0].path), |_| true)
                .unwrap();
        assert_eq!(checked(&events, ADAPTER_LOAD)[0], floor);

        // Zero headroom: the first read refuses; nothing is read, built or merged.
        let _ = take_refusal();
        let mut refused = proof_unet(false, false);
        let (r, events) = observe(0, || apply(&mut refused));
        assert!(r.is_err());
        assert_eq!(
            take_refusal().map(|r| r.label),
            Some(ADAPTER_LOAD.to_string())
        );
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(snapshot(&mut refused), base);

        // A headroom that admits the read but not the first install: that install refuses before
        // it is built (no synchronization, the U-Net untouched).
        let mut refused = proof_unet(false, false);
        let (r, events) = observe(PROCESS_RESERVE_BYTES + floor, || apply(&mut refused));
        assert!(r.is_err());
        let record = take_refusal().unwrap();
        assert_eq!(record.label, ADAPTER_MERGE);
        assert!(record.scratch_bytes > floor);
        assert_eq!(events, vec![Ev::Checked(ADAPTER_LOAD.into(), floor)]);
        assert_eq!(snapshot(&mut refused), base);

        // Packed base: the residual installs are guarded and the forward stays byte-identical.
        let x = rnd(&[1, 7, 256], 41, 1.0).as_dtype(Dtype::Float16).unwrap();
        let forward = |unet: &mut crate::unet::UNet2DConditionModel| {
            let parts: Vec<&str> = Q.split('.').collect();
            bits(&unet.adaptable_mut(&parts).unwrap().forward(&x).unwrap())
        };
        let mut host = proof_unet(false, true);
        apply(&mut host).unwrap();
        let mut guarded = proof_unet(false, true);
        let (r, events) = observe(PLENTY, || apply(&mut guarded));
        r.unwrap();
        assert_eq!(forward(&mut guarded), forward(&mut host));
        assert_eq!(checked(&events, ADAPTER_MERGE).len(), 4);
        assert_eq!(synced(&events, ADAPTER_MERGE), 4);

        println!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::adapters::adapter_reads_and_installs_refuse_before_they_run_and_stay_identical");
    }
}

/// Direct constructor equivalence, not a native-memory or layout certificate. Mixed source
/// widths ensure metadata passes the actual PRE-CAST weight element width, not nbytes or act.
#[test]
fn conv_call_metadata_constructor_matches_native_element_width() {
    let Some(mut completed) =
        completed_cpu_child("conv_call_metadata_constructor_matches_native_element_width")
    else {
        return;
    };
    for (input, weight_shape, stride, pad) in [
        ([1, 3, 5, 32], [224, 3, 3, 32], 1, 1),
        ([1, 64, 64, 32], [224, 3, 3, 32], 1, 1),
        ([1, 7, 9, 32], [64, 3, 3, 32], 2, 0),
    ] {
        let count = weight_shape.iter().map(|d| *d as usize).product::<usize>();
        for (activation, weight_dtype) in [
            (Dtype::Float16, Dtype::Float32),
            (Dtype::Float32, Dtype::Float16),
            (Dtype::Float16, Dtype::Bfloat16),
            (Dtype::Float32, Dtype::Float32),
        ] {
            let weight = Array::from_slice(&vec![0.25f32; count], &weight_shape)
                .as_dtype(weight_dtype)
                .unwrap();
            let native = ConvCall::new(&input, dtype_bytes(activation), &weight, stride, pad);
            let metadata = ConvCall::from_metadata(
                &input,
                dtype_bytes(activation),
                &weight_shape,
                dtype_bytes(weight_dtype),
                stride,
                pad,
            );
            let describe = |call: ConvCall| {
                (
                    call.in_bytes(),
                    call.out_bytes(),
                    call.weight_elements(),
                    call.winograd(),
                    call.transient(Backend::Cpu),
                    call.transient(Backend::Metal),
                )
            };
            assert_eq!(describe(native), describe(metadata),
                "ConvCall metadata element-width equivalence input={input:?} weight={weight_dtype:?} activation={activation:?}");
        }
    }
    completed.body_completed = true;
}
