//! sc-21483 acceptance on the real pinned Krea 2 NVFP4 checkpoint: **adapters over a PackedNvfp4
//! base** (`Comfy-Org/Krea-2@952f49d…/diffusion_models/krea2_turbo_nvfp4.safetensors`).
//!
//! Two ignored real-weight probes, deliberately cheap — this is an acceptance check, not a campaign:
//!
//! 1. [`nvfp4_lora_render_differs_and_zero_scale_restores_the_base_exactly`] — three fixed-seed
//!    512²/2-step renders through the production checkpoint route. The un-adapted render is the
//!    baseline; the same checkpoint loaded with a real rank-16 Krea 2 LoRA renders **differently**
//!    (the adapter genuinely applied over the packed base — before this story the load could not
//!    even resolve a target and failed "matched neither a diff-patch nor a low-rank projection");
//!    and the same LoRA at strength 0 renders **byte-identically** to the baseline, which is the
//!    render-level form of "removing the adapter restores the exact base output".
//! 2. [`nvfp4_trunk_keeps_every_packed_projection_after_a_real_lora_install`] — GPU but render-free:
//!    install the same LoRA additively onto a real NVFP4 trunk and prove the NVFP4 accounting is
//!    *unchanged* — same quantized-projection count, same FP4-lit count, same packed byte total. No
//!    projection is dequantized, re-quantized, or converted to another regime to host the adapter.
//!
//! Run pinned to the quiet GPU (GPU 0 belongs to the CI runners):
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=1 \
//! KREAMANIA_VARIANT7=…/krea2_turbo_nvfp4.safetensors \
//! KREA_BASE_SNAPSHOT=…/Krea-2-Turbo/snapshots/<rev> \
//! KREA_2_LORA=…/krea2_<name>_v1_onetrainer.safetensors \
//!   cargo test -p candle-gen-krea --release --features cuda --test integration \
//!   nvfp4_adapters_real_weights:: -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::native_mapping::DeclaredLogicalShapes;
use candle_gen_krea::{Krea2Config, Krea2Transformer};

/// Floor on the number of projections the pinned Krea 2 LoRA must resolve onto the NVFP4 trunk.
///
/// **Measured**: `applied = 256, skipped_keys = 0, skipped_targets = 0` against
/// `krea2_turbo_nvfp4.safetensors` + `krea2_identity_edit_v1_1_r64.safetensors` on CUDA
/// (`nvfp4 quantized=160 fp4_lit=115`). A floor rather than an exact equality so a LoRA trained over
/// a *superset* of targets still passes — but it sits exactly at the measured count, so any partial
/// resolution (a renamed key class, a regime that quietly refused a subset) drops below it. The
/// `applied > 0` assertion this replaces could not tell "one projection matched" from "all did".
const MIN_APPLIED_TARGETS: usize = 256;

/// The pinned NVFP4 DiT, the base component snapshot, and a real Krea 2 LoRA. `None` (with a
/// printed reason) when the box is not provisioned, so the lane stays skippable.
fn fixtures() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let mut missing = Vec::new();
    let mut get = |name: &str| match std::env::var(name) {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            missing.push(name.to_owned());
            None
        }
    };
    let dit = get("KREAMANIA_VARIANT7");
    let base = get("KREA_BASE_SNAPSHOT");
    let lora = get("KREA_2_LORA");
    match (dit, base, lora) {
        (Some(dit), Some(base), Some(lora)) => Some((dit, base, lora)),
        _ => {
            eprintln!("skipping: set {}", missing.join(", "));
            None
        }
    }
}

fn lora_spec(path: &std::path::Path, scale: f32) -> Vec<AdapterSpec> {
    vec![AdapterSpec::new(
        path.to_path_buf(),
        scale,
        AdapterKind::Lora,
    )]
}

/// One fixed-seed render through the production checkpoint route (`load_from_native_dit_file`), the
/// same entry a SceneWorks imported-model load takes. The generator is dropped before returning so
/// each render owns the GPU alone.
fn render(dit: &std::path::Path, base: &std::path::Path, adapters: &[AdapterSpec]) -> Image {
    let request = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dusk".to_owned(),
        width: 512,
        height: 512,
        steps: Some(2),
        seed: Some(7),
        ..Default::default()
    };
    let generator = candle_gen_krea::load_from_native_dit_file(
        dit,
        base,
        adapters,
        candle_gen_krea::descriptor(),
    )
    .expect("the pinned NVFP4 checkpoint loads through the production checkpoint route");
    let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
    else {
        panic!("expected image output")
    };
    assert_eq!(images.len(), 1);
    images.into_iter().next().unwrap()
}

fn differing_pixels(a: &Image, b: &Image) -> usize {
    assert_eq!(
        a.pixels.len(),
        b.pixels.len(),
        "renders must share geometry"
    );
    a.pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| x != y)
        .count()
}

/// **AC#1 on real weights.** A real rank-16 Krea 2 LoRA installs over the pinned NVFP4 checkpoint
/// and changes the render; the same adapter at strength 0 leaves the base output bit-identical.
///
/// Strength 0 is an **install-neutrality** proof, NOT a removal proof: `Adapter::is_zero`
/// short-circuits a zero-scale residual before any residual math runs, so the forward is the bare
/// packed base and the only thing the byte-equal render can witness is that *installing* the adapter
/// did not perturb the base (no dequantize, no re-pack, no fold at load). The removal claim — that
/// clearing an installed, non-zero residual restores the base exactly — is carried by
/// `nvfp4_base_hosts_an_additive_lora_and_restores_exactly_on_removal` in the unit suite.
#[test]
#[ignore = "requires explicitly scheduled CUDA and the local pinned Krea 2 NVFP4 checkpoint + a Krea 2 LoRA"]
fn nvfp4_lora_render_differs_and_zero_scale_restores_the_base_exactly() {
    let Some((dit, base, lora)) = fixtures() else {
        return;
    };

    let baseline = render(&dit, &base, &[]);
    let adapted = render(&dit, &base, &lora_spec(&lora, 1.0));
    let zero_scale = render(&dit, &base, &lora_spec(&lora, 0.0));

    let moved = differing_pixels(&baseline, &adapted);
    println!(
        "sc-21483 NVFP4+LoRA: {moved}/{} subpixels changed at strength 1.0",
        baseline.pixels.len()
    );
    assert!(
        moved > baseline.pixels.len() / 100,
        "a rank-16 LoRA over the NVFP4 trunk must visibly change the render; only {moved} \
         subpixels moved"
    );

    assert_eq!(
        zero_scale.pixels, baseline.pixels,
        "a strength-0 adapter must leave the packed base output byte-identical — a difference \
         means installing the adapter perturbed the base"
    );
}

/// **AC#2 on real weights.** Installing a real LoRA must not move a single projection out of the
/// NVFP4 regime. The report is byte-accounting over the trunk's actual resident weight buffers, so
/// an equal report before and after is a direct statement that nothing was dequantized, re-packed,
/// or folded to host the residuals.
#[test]
#[ignore = "requires explicitly scheduled CUDA and the local pinned Krea 2 NVFP4 checkpoint + a Krea 2 LoRA"]
fn nvfp4_trunk_keeps_every_packed_projection_after_a_real_lora_install() {
    let Some((dit_path, _base, lora)) = fixtures() else {
        return;
    };

    let cfg = Krea2Config::turbo();
    let device = Device::new_cuda(0).expect("CUDA device");
    let weights = Weights::from_native_file_for(
        &dit_path,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("the pinned NVFP4 checkpoint plans");
    assert!(
        weights.is_native_nvfp4(),
        "the fixture must be the NVFP4 checkpoint"
    );

    let mut trunk = Krea2Transformer::load(&weights, &cfg).expect("NVFP4 trunk builds");
    let before = trunk.nvfp4_report();
    assert!(
        before.n_quantized > 0,
        "the NVFP4 plan must actually serve projections through Nvfp4Linear"
    );

    let report = candle_gen_krea::install_additive(&mut trunk, &lora_spec(&lora, 1.0), 0)
        .expect("a real Krea 2 LoRA installs over the NVFP4 trunk");
    println!(
        "sc-21483 install: applied={} skipped_keys={} skipped_targets={} | nvfp4 quantized={} \
         fp4_lit={} packed_bytes={}",
        report.applied,
        report.skipped_keys,
        report.skipped_targets.len(),
        before.n_quantized,
        before.fp4_lit,
        before.nvfp4_bytes,
    );
    // A count, not a `> 0`: the pinned rank-16 Krea 2 LoRA targets every attention/MLP projection of
    // all 48 blocks, so a partial resolution (a renamed key class, a regime that quietly refused a
    // subset) has to show up as a number below the floor rather than as a still-passing `> 0`.
    assert!(
        report.applied >= MIN_APPLIED_TARGETS,
        "the LoRA must resolve onto the trunk's NVFP4 projections — applied={} is below the \
         {MIN_APPLIED_TARGETS}-target floor for this checkpoint/LoRA pair (a zero is the \
         'matched no target' failure this story fixes; a low non-zero is a partial resolution)",
        report.applied
    );
    // Every key must land. `skipped_keys > 0` on this pair means a factor was dropped, and on an
    // NVFP4 base there is no fallback that could have rendered it anyway — the story's whole point
    // is that such a key is a refusal, never a silent skip.
    assert_eq!(
        report.skipped_keys, 0,
        "no key of the pinned LoRA may be skipped against the NVFP4 trunk"
    );
    assert!(
        report.skipped_targets.is_empty(),
        "every LoRA target must exist on the NVFP4 trunk surface; unmatched: {:?}",
        report.skipped_targets
    );

    let after = trunk.nvfp4_report();
    assert_eq!(
        (after.n_quantized, after.fp4_lit, after.nvfp4_bytes),
        (before.n_quantized, before.fp4_lit, before.nvfp4_bytes),
        "installing an adapter must not move any projection out of the NVFP4 regime"
    );

    // Clearing the stack is likewise regime-neutral.
    trunk.clear_adapters().expect("adapters clear");
    let cleared = trunk.nvfp4_report();
    assert_eq!(
        (cleared.n_quantized, cleared.fp4_lit, cleared.nvfp4_bytes),
        (before.n_quantized, before.fp4_lit, before.nvfp4_bytes),
    );
}
