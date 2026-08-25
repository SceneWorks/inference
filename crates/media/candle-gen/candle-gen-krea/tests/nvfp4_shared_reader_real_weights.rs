//! sc-21482 walking-skeleton acceptance on the real pinned Krea 2 NVFP4 checkpoint
//! (`Comfy-Org/Krea-2@952f49d…/diffusion_models/krea2_turbo_nvfp4.safetensors`).
//!
//! Three ignored real-weight probes:
//!
//! 1. [`linked_and_managed_copies_share_plan_and_fixed_seed_output`] — the SAME bytes opened from
//!    a managed (HF-cache) path and from a linked copy at an unrelated path compile identical
//!    logical-weight plans and render byte-identical fixed-seed output through the
//!    checkpoint-plan route. Source location is an identity fact, never an execution input.
//! 2. [`sm120_rows_stay_packed_and_the_receipt_matches_the_plan`] — on this box's `sm_120` CUDA
//!    device, every eligible NVFP4 row is priced `Packed`, the whole trunk constructs through the
//!    shared reader, and the measured receipt equals the plan's pricing.
//! 3. [`cpu_load_takes_the_declared_dense_fallback`] — the same checkpoint on the CPU prices every
//!    NVFP4 row `Dense` and materializes the declared bf16 fallback, receipt matching.
//!
//! Run pinned to the quiet GPU: `CUDA_VISIBLE_DEVICES=1`, with `KREAMANIA_VARIANT7` pointing at
//! the NVFP4 DiT file and `KREA_BASE_SNAPSHOT` at the Krea 2 base component snapshot.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::checkpoint_codec::{ResidencyMode, NVFP4_CODEC};
use candle_gen::gen_core::{GenerationOutput, GenerationRequest, Generator, Image};
use candle_gen::logical_weights::LogicalTensor;
use candle_gen_krea::loader::Weights;
use candle_gen_krea::native_mapping::DeclaredLogicalShapes;
use candle_gen_krea::{Krea2Config, Krea2Transformer};

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn render_one(generator: &dyn Generator, request: &GenerationRequest) -> Image {
    let GenerationOutput::Images(images) = generator.generate(request, &mut |_| {}).unwrap() else {
        panic!("expected image output")
    };
    assert_eq!(images.len(), 1);
    images.into_iter().next().unwrap()
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and the local pinned Krea 2 NVFP4 checkpoint"]
fn linked_and_managed_copies_share_plan_and_fixed_seed_output() {
    let managed = env_path("KREAMANIA_VARIANT7");
    let base = env_path("KREA_BASE_SNAPSHOT");
    // The "linked" copy: the same bytes at an unrelated path on the same volume. A hard link (not
    // a junction/symlink — candle refuses reparse points, os error 448) is what a user linking an
    // existing ComfyUI folder into the library amounts to at this seam.
    let link_dir = tempfile::Builder::new()
        .prefix("sc21482-linked-")
        .tempdir_in(managed.parent().expect("checkpoint has a parent dir"))
        .expect("temp dir beside the managed copy");
    let linked = link_dir.path().join("krea2_turbo_nvfp4.safetensors");
    std::fs::hard_link(&managed, &linked).expect("hard link on the same volume");

    // Same semantic plan: the compiled logical-weight plan is a pure function of the bytes, so the
    // two source locations must compile the identical plan (codec rows, companions, residency).
    let cfg = Krea2Config::turbo();
    let device = Device::new_cuda(0).expect("CUDA device");
    let managed_w = Weights::from_native_file_for(
        &managed,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("managed copy plans");
    let linked_w = Weights::from_native_file_for(
        &linked,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("linked copy plans");
    assert!(managed_w.is_native_nvfp4() && linked_w.is_native_nvfp4());
    assert_eq!(
        managed_w.logical_plan().expect("managed plan"),
        linked_w.logical_plan().expect("linked plan"),
        "source location must not leak into the logical-weight plan"
    );
    drop(managed_w);
    drop(linked_w);

    // Same fixed-seed output through the checkpoint-plan production route.
    let request = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dusk".to_owned(),
        width: 512,
        height: 512,
        steps: Some(2),
        seed: Some(7),
        ..Default::default()
    };
    let managed_generator = candle_gen_krea::load_from_native_dit_file(
        &managed,
        &base,
        &[],
        candle_gen_krea::descriptor(),
    )
    .expect("managed copy loads through the production checkpoint route");
    let managed_image = render_one(managed_generator.as_ref(), &request);
    drop(managed_generator);
    let linked_generator = candle_gen_krea::load_from_native_dit_file(
        &linked,
        &base,
        &[],
        candle_gen_krea::descriptor(),
    )
    .expect("linked copy loads through the production checkpoint route");
    let linked_image = render_one(linked_generator.as_ref(), &request);
    assert_eq!(
        managed_image.pixels, linked_image.pixels,
        "linked and managed copies must render byte-identical fixed-seed output"
    );
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and the local pinned Krea 2 NVFP4 checkpoint"]
fn sm120_rows_stay_packed_and_the_receipt_matches_the_plan() {
    let path = env_path("KREAMANIA_VARIANT7");
    let cfg = Krea2Config::turbo();
    let device = Device::new_cuda(0).expect("CUDA device");
    let w = Weights::from_native_file_for(
        &path,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("the pinned checkpoint plans");
    assert!(w.is_native_nvfp4());
    let plan = w.logical_plan().expect("plan").clone();

    // Every eligible NVFP4 row is priced Packed on sm_120; ineligible rows
    // (`full_precision_matrix_mult`, padded, misaligned) are priced Dense — and there must be at
    // least one genuinely packed row or this probe proves nothing.
    let packed_rows = plan
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.codec_id == NVFP4_CODEC.codec_id
                && tensor.residency.mode == ResidencyMode::Packed
        })
        .count();
    assert!(
        packed_rows > 0,
        "the pinned checkpoint must have packed-eligible rows on an sm_120 device"
    );

    // Construct the whole trunk through the shared reader, then compare the measured receipt to
    // the plan (E8: the receipt is what happened; the plan is what was declared; they must agree).
    let dit = Krea2Transformer::load(&w, &cfg).expect("the trunk constructs");
    let receipt = w.logical_weight_receipt().expect("receipt");
    assert_eq!(
        receipt.tensor_count,
        plan.tensor_count(),
        "constructing the trunk must materialize every planned tensor through the reader"
    );
    assert_eq!(
        receipt.resident_bytes(),
        plan.resident_bytes(),
        "the measured residency must equal the plan's pricing"
    );
    assert_eq!(receipt.source_bytes, plan.source_bytes);
    drop(dit);
}

#[test]
#[ignore = "requires the local pinned Krea 2 NVFP4 checkpoint (CPU-only probe)"]
fn cpu_load_takes_the_declared_dense_fallback() {
    let path = env_path("KREAMANIA_VARIANT7");
    let cfg = Krea2Config::turbo();
    let w = Weights::from_native_file_for(
        &path,
        &Device::Cpu,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("the pinned checkpoint plans on the CPU");
    assert!(w.is_native_nvfp4());
    let plan = w.logical_plan().expect("plan").clone();
    for tensor in plan
        .tensors
        .iter()
        .filter(|tensor| tensor.codec_id == NVFP4_CODEC.codec_id)
    {
        assert_eq!(
            tensor.residency.mode,
            ResidencyMode::Dense,
            "{}: a CPU load must price every NVFP4 row as the dense fallback",
            tensor.logical_key
        );
    }
    // Materialize one real projection: the declared dense fallback, measured to match the plan.
    let first_nvfp4 = plan
        .tensors
        .iter()
        .find(|tensor| tensor.codec_id == NVFP4_CODEC.codec_id)
        .expect("the checkpoint has NVFP4 rows");
    let read = w
        .get(&first_nvfp4.logical_key)
        .expect("the dense fallback materializes");
    assert_eq!(read.dims(), first_nvfp4.shape.as_slice());
    let receipt = w.logical_weight_receipt().expect("receipt");
    let row = receipt
        .residency
        .iter()
        .find(|row| row.codec_id == NVFP4_CODEC.codec_id)
        .expect("the materialized row is reported");
    assert_eq!(row.resident_bytes, first_nvfp4.residency.resident_bytes);
    // The dense fallback is the codec's decode, not raw bytes: a bf16 tensor of the logical shape.
    assert!(matches!(
        w.read_planned(&first_nvfp4.logical_key),
        Ok(LogicalTensor::Dense(_))
    ));
}
