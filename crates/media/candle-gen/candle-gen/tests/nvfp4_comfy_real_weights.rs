//! sc-20641 real-weight validation of the ComfyUI NVFP4 codec against a **header-declared** NVFP4
//! checkpoint — `#[ignore]`, env-driven, run manually with the file on disk:
//!
//! ```text
//! SC20641_NVFP4=$HOME/models/kreamania_variant7.safetensors \
//!   cargo test -p candle-gen --test integration -- nvfp4_comfy_real_weights:: --ignored --nocapture
//! ```
//!
//! The reference artifact is `kreamania_variant7.safetensors` (8,167,318,440 B), written by the
//! *ComfyUI Kitchen NVFP4 Converter*. It is the case sc-20385 pinned as a typed refusal and this
//! story flips to support, and it is the awkward shape of the format:
//!
//! * **No `.comfy_quant` tensors at all.** The declaration is a file-level
//!   `__metadata__._quantization_metadata` object, and its 224 layer names are *relative*
//!   (`blocks.0.attn.wq`) to the `model.diffusion_model.` prefix the tensors carry.
//! * **Mixed**: 224 NVFP4 layers next to ~40 plain `BF16` weights, biases and norms, dispatched per
//!   layer by the same plan.
//! * **Multi-atom scale grids**: every NVFP4 layer is ≥ 1536 × 6144 logical, i.e. ≥ 12 row atoms ×
//!   ≥ 96 block atoms, so the 128×4 `to_blocked` swizzle is fully exercised.
//!
//! The test is header-only where it can be (planning never touches the data region, so the plan
//! assertions run in milliseconds on an 8 GB file) and decodes exactly one layer to keep the
//! runtime and the host-RAM cost bounded.

use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::logical_weights::{
    plan_logical_weights, read_logical_weights, CandleCodecResidency, LogicalTensor,
};
use gen_core::checkpoint_codec::{LogicalKeyMapping, ResidencyMode, TensorCodecSpec};

/// The checkpoint stores canonical keys already; the adapter surface under test is the codec, not a
/// dialect translation, so the mapping is the identity over every on-disk key.
struct IdentityKeys;

impl LogicalKeyMapping for IdentityKeys {
    fn mapping_id(&self) -> &'static str {
        "nvfp4-real-weights-identity"
    }
    fn logical_key(&self, physical_key: &str) -> Option<String> {
        Some(physical_key.to_owned())
    }
}

fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SC20641_NVFP4").ok()?);
    path.exists().then_some(path)
}

/// AC1 + AC2 on the real artifact: the header-declared NVFP4 checkpoint **plans** (this is the
/// sc-20385 refusal, flipped), dispatches per layer, prices the two residencies independently, and
/// decodes a real layer through the dense fallback.
#[test]
#[ignore = "needs kreamania_variant7.safetensors via SC20641_NVFP4"]
fn kreamania_variant7_plans_decodes_and_prices_both_residencies() {
    let Some(path) = checkpoint() else {
        panic!("set SC20641_NVFP4 to the kreamania_variant7 checkpoint");
    };

    // ---- plan (header-only; never reads the 8 GB data region) --------------------------------
    let plan = plan_logical_weights(&path, &IdentityKeys, &CandleCodecResidency::DENSE).expect(
        "a header-declared NVFP4 checkpoint must plan (sc-20385 refusal → sc-20641 support)",
    );

    let mut by_codec: BTreeMap<&str, usize> = BTreeMap::new();
    for tensor in &plan.tensors {
        *by_codec.entry(tensor.codec_id).or_default() += 1;
    }
    println!("codec census: {by_codec:?}");
    assert_eq!(
        by_codec.get("nvfp4-v1").copied(),
        Some(224),
        "the file declares 224 nvfp4 layers"
    );
    assert!(
        by_codec.len() > 1,
        "variant7 is mixed: NVFP4 layers next to dense bf16 rows, dispatched per layer"
    );

    // Every NVFP4 layer resolved BOTH scale levels, under the relative-name prefix.
    let nvfp4: Vec<_> = plan
        .tensors
        .iter()
        .filter(|tensor| tensor.codec_id == "nvfp4-v1")
        .collect();
    for tensor in &nvfp4 {
        let TensorCodecSpec::Nvfp4 {
            block_scale,
            global_scale,
            stored_shape,
            ..
        } = &tensor.codec
        else {
            panic!("{}: nvfp4 codec without an nvfp4 spec", tensor.physical_key);
        };
        let base = tensor
            .physical_key
            .strip_suffix(".weight")
            .expect("nvfp4 weights end in .weight");
        assert_eq!(*block_scale, format!("{base}.weight_scale"));
        assert_eq!(*global_scale, format!("{base}.weight_scale_2"));
        assert!(
            base.starts_with("model.diffusion_model."),
            "the metadata's relative layer names must resolve under the file's prefix: {base}"
        );
        // 16-padded on both axes, and multi-atom in both scale dimensions.
        assert!(stored_shape[0].is_multiple_of(16) && stored_shape[1].is_multiple_of(16));
        assert!(
            stored_shape[0] / 128 > 1 && (stored_shape[1] / 16) / 4 > 1,
            "{base}: expected a multi-atom scale grid, got {stored_shape:?}"
        );
        assert_eq!(tensor.shape, stored_shape.to_vec());
    }

    // ---- AC2: the two residencies are priced independently ------------------------------------
    let native = CandleCodecResidency {
        fp8_e4m3_native: false,
        nvfp4_native: true,
    };
    let packed_plan = plan_logical_weights(&path, &IdentityKeys, &native).expect("packed plan");
    let packed_rows: Vec<_> = packed_plan
        .tensors
        .iter()
        .filter(|tensor| tensor.codec_id == "nvfp4-v1")
        .collect();
    assert!(
        packed_rows
            .iter()
            .all(|tensor| tensor.residency.mode == ResidencyMode::Packed),
        "every variant7 NVFP4 layer is K/N-aligned, so sm_120 keeps all of them packed"
    );
    for (dense, packed) in nvfp4.iter().zip(packed_rows.iter()) {
        // Dense holds bf16 over the logical grid; packed holds the stored nibbles — a 4× ratio, and
        // each is computed from its own quantity rather than scaled from the other.
        assert_eq!(
            dense.residency.resident_bytes,
            packed.residency.resident_bytes * 4
        );
    }
    let (dense_bytes, packed_bytes) = (plan.resident_bytes(), packed_plan.resident_bytes());
    println!(
        "resident: dense {:.2} GiB, packed {:.2} GiB, source {:.2} GiB",
        dense_bytes as f64 / (1 << 30) as f64,
        packed_bytes as f64 / (1 << 30) as f64,
        plan.source_bytes as f64 / (1 << 30) as f64,
    );
    assert!(packed_bytes < dense_bytes);
    let file_len = std::fs::metadata(&path).expect("stat").len();
    assert!(
        plan.source_bytes < file_len && file_len - plan.source_bytes < 1 << 20,
        "source_bytes ({}) must cover the whole data region: the only difference from the file \
         length ({file_len}) is the 8-byte prefix and the header",
        plan.source_bytes
    );

    // ---- read the file and cross-check one real layer's decode -------------------------------
    //
    // The reader requires the plan's physical-key surface to equal the file's, so this is a whole-
    // file read. It uses the PACKED policy deliberately: packed keeps the 4-bit nibbles resident
    // (~4 GiB) instead of the ~25 GiB a dense bf16 read of a 224-layer DiT would need, and it
    // exercises the repack on 224 real multi-atom layers rather than on a fixture.
    let weights = read_logical_weights(&path, &packed_plan, &Device::Cpu)
        .expect("a real header-declared NVFP4 checkpoint must read");
    assert_eq!(
        weights.receipt.resident_bytes(),
        packed_plan.resident_bytes(),
        "the receipt measures residency off the decoded values; it must match what was priced"
    );

    let one = packed_rows[0];
    println!("cross-checking {} {:?}", one.physical_key, one.shape);
    let LogicalTensor::PackedNvfp4 { tensor, .. } = weights
        .tensors
        .get(one.logical_key.as_str())
        .expect("decoded layer")
    else {
        panic!("an sm_120-eligible NVFP4 layer must repack");
    };
    assert_eq!((tensor.rows, tensor.cols), (one.shape[0], one.shape[1]));

    // The repack **round-trips against the reference** on 48 x 96 real atoms: the container's own
    // dequant (block scales indexed through `Nvfp4Tensor::scale_offset_for`) must reproduce, element
    // for element, what `gen_core::decode_nvfp4` reads out of the file's `to_blocked` buffer.
    //
    // These are NOT two independent determinations of the atom order, and this assertion must not be
    // read as one: both routes resolve through `blocked_scale_index`, and the permutation
    // `from_kitchen_parts` applies is undone by the one `scale_offset_for` applies on the way back
    // out (the nibble rotate cancels the same way). What this DOES catch — and what it caught — is a
    // repack that copies the scale buffer verbatim instead of permuting it, plus any mis-shaping of
    // a real 6144x6144 layer. The independent check on the atom order itself is the
    // `nvfp4_golden_decodes_exactly_on_the_dense_fallback` golden, whose expectation comes from
    // `reference_to_blocked_index` — `comfy.float.to_blocked` transliterated from its definition,
    // never calling the code under test. Which order the FP4 GEMM actually wants is resolvable only
    // by sc-20651's live cuBLASLt run on Blackwell.
    let from_container = tensor.dequantize_to_vec();
    let st = unsafe {
        candle_gen::candle_core::safetensors::MmapedSafetensors::new(&path).expect("mmap")
    };
    let base = one.physical_key.strip_suffix(".weight").expect(".weight");
    let mut from_reference = Vec::new();
    let global = f32::from_le_bytes(
        st.get(&format!("{base}.weight_scale_2"))
            .expect("global scale")
            .data()
            .try_into()
            .expect("one F32"),
    );
    gen_core::decode_nvfp4(
        st.get(&one.physical_key).expect("weight").data(),
        st.get(&format!("{base}.weight_scale"))
            .expect("scales")
            .data(),
        global,
        [one.shape[0], one.shape[1]],
        [one.shape[0], one.shape[1]],
        &mut from_reference,
    )
    .expect("the reference decode must accept a real layer");
    assert_eq!(from_container.len(), from_reference.len());
    for (index, (container, reference)) in
        from_container.iter().zip(from_reference.iter()).enumerate()
    {
        assert_eq!(
            container,
            reference,
            "element {index} (row {}, col {}): the repacked container and the reference decode \
             disagree, so the repack does not round-trip on real multi-atom scales",
            index / one.shape[1],
            index % one.shape[1]
        );
    }

    // The values are real weights, so the guard is on their statistics, not a pinned constant.
    assert!(
        from_reference.iter().all(|value| value.is_finite()),
        "NaN/inf leaked into a decoded layer"
    );
    let nonzero = from_reference.iter().filter(|value| **value != 0.0).count();
    assert!(
        nonzero > from_reference.len() / 2,
        "a real DiT projection is dense: {nonzero} of {} elements are non-zero",
        from_reference.len()
    );
    let amax = from_reference
        .iter()
        .fold(0.0_f32, |max, v| max.max(v.abs()));
    println!(
        "decoded amax {amax}, non-zero {nonzero}/{}",
        from_reference.len()
    );
    assert!(
        amax > 0.0 && amax < 10.0,
        "implausible DiT weight range: {amax}"
    );
}
