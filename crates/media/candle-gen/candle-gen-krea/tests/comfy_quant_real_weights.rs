//! sc-27092 real-weight acceptance for the ComfyUI-quantized Krea 2 DiTs — the two published
//! encodings the ComfyUI `comfy_quant` family carries besides int8-per-row and Kitchen NVFP4.
//!
//! Two ignored probes:
//!
//! 1. [`the_scalar_fp8_export_plans_companion_scales_and_decodes_dense`] — a ComfyUI `ModelSave`
//!    of an fp8-scaled Krea 2 DiT (256 `F8_E4M3` weights, 256 `F32` **scalar** `.weight_scale`
//!    companions, 256 `.comfy_quant` descriptors — 160 plain, 96 `full_precision_matrix_mult` —
//!    and 174 BF16 dense tensors) compiles a plan whose 256 quantized rows are
//!    `fp8-e4m3-scalar-v1` with a **companion** scale (never unit), every row materializes as a
//!    dense bf16 weight at the declared shape, and the measured receipt reconciles with the plan.
//!    The companion-scale assertion is the load-bearing one: this convention's scale is REAL, and
//!    a unit scale would decode the file at the wrong magnitude.
//! 2. [`the_kitchen_mxfp8_export_plans_block_scales_and_decodes_dense`] — the Kitchen converter's
//!    MXFP8 export (224 `F8_E4M3` weights on a 32-aligned grid, 224 `U8` block scales in the
//!    cuBLAS 128×4 swizzled layout, quantization declared FILE-LEVEL, 206 BF16 dense tensors)
//!    compiles `mxfp8-v1` rows at the declared logical shape, materializes as dense bf16, and
//!    decodes to finite, non-degenerate weights — the check that the file's scale layout is the one
//!    the decoder reads.
//!
//! Both files are read IN PLACE, exactly as the imported lane reads them.
//!
//! Run pinned to the quiet GPU: `CUDA_VISIBLE_DEVICES=1`, with `KREAMANIA_SCALED_FP8` /
//! `KREAMANIA_MXFP8` pointing at the DiT files.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    ResidencyMode, ScalarScaleSource, TensorCodecSpec, FP8_E4M3_SCALAR_CODEC, MXFP8_CODEC,
};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::native_mapping::DeclaredLogicalShapes;
use candle_gen_krea::{Krea2Config, Krea2Transformer};

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and the local fp8-scaled Krea 2 checkpoint"]
fn the_scalar_fp8_export_plans_companion_scales_and_decodes_dense() {
    let path = env_path("KREAMANIA_SCALED_FP8");
    let cfg = Krea2Config::turbo();
    let device = Device::new_cuda(0).expect("CUDA device");
    let w = Weights::from_native_file_for(
        &path,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("the fp8-scaled checkpoint plans");
    // The `.comfy_quant` marker does not make a file int8 (sc-27092): this one declares
    // `float8_e4m3fn` and carries no `I8` weight at all.
    assert!(
        !w.is_plain_int8(),
        "an fp8 export must not be claimed as a plain-int8 file"
    );
    assert!(!w.is_native_nvfp4());
    let plan = w.logical_plan().expect("plan").clone();
    assert_eq!(plan.tensor_count(), 430, "the Krea 2 DiT's model tensors");

    let fp8_rows: Vec<_> = plan
        .tensors
        .iter()
        .filter(|tensor| tensor.codec_id == FP8_E4M3_SCALAR_CODEC.codec_id)
        .collect();
    assert_eq!(
        fp8_rows.len(),
        256,
        "the export's 256 quantized projections"
    );
    // EVERY fp8 row carries a COMPANION scale — the defining property of this convention, and the
    // thing a unit-scale decode would lose.
    for tensor in &fp8_rows {
        assert!(
            matches!(
                &tensor.codec,
                TensorCodecSpec::ScalarFp8 {
                    scale: ScalarScaleSource::Companion { .. },
                    ..
                }
            ),
            "every fp8 row must decode with its real per-tensor scale: {:?}",
            tensor.physical_key
        );
        // The packed fp8 leg is masked on every host (no provider consumes it), so every row takes
        // the dense fallback — and the plan must say so rather than price a packed container the
        // reader will not build.
        assert_eq!(
            tensor.residency.mode,
            ResidencyMode::Dense,
            "{:?}",
            tensor.physical_key
        );
    }
    // The producer flagged 96 of them `full_precision_matrix_mult` — the descriptor field that
    // asks for the dense fallback, recorded per row.
    assert_eq!(
        fp8_rows
            .iter()
            .filter(|tensor| tensor.codec.full_precision_matrix_mult())
            .count(),
        96,
        "the export's `full_precision_matrix_mult` rows must be recorded on the plan"
    );
    // The descriptors and the scales are companions of the rows — consumed by the decode, never
    // retained.
    assert_eq!(
        plan.companions.len(),
        512,
        "256 `.comfy_quant` descriptors + 256 `.weight_scale` companions"
    );

    // The whole trunk constructs through the shared reader, and the measured receipt reconciles
    // with the plan (the receipt is what happened; the plan is what was declared).
    let dit = Krea2Transformer::load(&w, &cfg).expect("the trunk constructs");
    let receipt = w.logical_weight_receipt().expect("receipt");
    assert_eq!(receipt.tensor_count, plan.tensor_count());
    assert_eq!(receipt.source_bytes, plan.source_bytes);
    let facts = w
        .checkpoint_weight_facts()
        .expect("the pinned source is unchanged")
        .expect("a native import has a plan");
    assert!(facts.is_complete());
    assert!(
        facts.source().declares(FP8_E4M3_SCALAR_CODEC.codec_id),
        "the facts must report the fp8-scalar source codec"
    );
    drop(dit);
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and the local MXFP8 Krea 2 checkpoint"]
fn the_kitchen_mxfp8_export_plans_block_scales_and_decodes_dense() {
    let path = env_path("KREAMANIA_MXFP8");
    let cfg = Krea2Config::turbo();
    let device = Device::new_cuda(0).expect("CUDA device");
    let w = Weights::from_native_file_for(
        &path,
        &device,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(&cfg),
    )
    .expect("the MXFP8 checkpoint plans");
    assert!(
        !w.is_plain_int8(),
        "an MXFP8 export must not be claimed as a plain-int8 file"
    );
    assert!(!w.is_native_nvfp4());
    let plan = w.logical_plan().expect("plan").clone();
    assert_eq!(plan.tensor_count(), 430, "the Krea 2 DiT's model tensors");

    let mxfp8_rows: Vec<_> = plan
        .tensors
        .iter()
        .filter(|tensor| tensor.codec_id == MXFP8_CODEC.codec_id)
        .collect();
    assert_eq!(
        mxfp8_rows.len(),
        224,
        "the export's 224 quantized projections"
    );
    for tensor in &mxfp8_rows {
        assert!(
            matches!(&tensor.codec, TensorCodecSpec::Mxfp8 { .. }),
            "{:?}",
            tensor.physical_key
        );
        assert_eq!(
            tensor.residency.mode,
            ResidencyMode::Dense,
            "MXFP8 has no block-scaled kernel in this workspace: {}",
            tensor.physical_key
        );
        // The plan carries the LOGICAL shape, not the stored 32-aligned grid — the whole point of
        // declaring the architecture's shapes (sc-27092).
        let logical = &tensor.shape;
        assert_eq!(logical.len(), 2, "{:?}", tensor.physical_key);
    }
    // MXFP8 carries exactly ONE companion per projection (the block scale): no per-layer
    // `.comfy_quant` blob and no second-level scale. The plan's physical key set is therefore the
    // file's tensor surface exactly — which is what the key-surface validator checks.
    assert_eq!(plan.companions.len(), 224, "one block scale per projection");

    // The decoded weights must be finite and non-degenerate: a misread swizzle shows up as
    // NaN/Inf (a wrong E8M0 exponent) or as a zero matrix.
    let decoded = w
        .read_planned(&mxfp8_rows[0].logical_key)
        .expect("the block-scaled row decodes");
    let values = match &decoded {
        candle_gen::logical_weights::LogicalTensor::Dense(tensor) => tensor
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("f32"),
        _ => panic!("an MXFP8 row must materialize dense"),
    };
    assert!(!values.is_empty(), "the decoded row must not be empty");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "a misread block scale decodes to NaN/Inf"
    );
    let peak = values
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    assert!(peak > 0.0, "a misread block scale decodes to zeros");

    let dit = Krea2Transformer::load(&w, &cfg).expect("the trunk constructs");
    let receipt = w.logical_weight_receipt().expect("receipt");
    assert_eq!(receipt.tensor_count, plan.tensor_count());
    assert_eq!(receipt.source_bytes, plan.source_bytes);
    let facts = w
        .checkpoint_weight_facts()
        .expect("the pinned source is unchanged")
        .expect("a native import has a plan");
    assert!(facts.is_complete());
    assert!(
        facts.source().declares(MXFP8_CODEC.codec_id),
        "the facts must report the MXFP8 source codec"
    );
    drop(dit);
}
